#!/usr/bin/env python3
"""
Heorth — a self-contained local AI server with a browser GUI.
(Heorth is Old English for "hearth" — your models, at home. Formerly LocalMind.)

Quick start (Linux / macOS / Windows):
    python3 heorth.py            (Windows: python heorth.py)
    ...or whatever this file is currently named.

That's it. On first run Heorth creates its own private Python environment
next to itself (heorth_data/venv), installs what it needs, restarts inside
that environment and opens http://127.0.0.1:8317 in your browser.
Upgrading from LocalMind? Heorth finds and keeps using your existing
localmind_data folder — nothing is lost.

Text generation is powered by Ollama (https://ollama.com). If Ollama is not
installed yet, the GUI shows one-line install instructions for your OS.

Optional features (installed later, from inside the GUI, one click each):
  * Image generation  (Stable Diffusion via diffusers + torch)
  * MCP client        (connect Model Context Protocol servers to the agent)
  * Private web search (a SearXNG container — needs Docker; the GUI guides you)
  * Computer control  (let the agent see the screen and move the mouse/keyboard
                       via PyAutoGUI — OFF by default, behind a consent gate,
                       with a live action log and an emergency stop)
  * Vision input      (attach images to a chat message and ask about them
                       with a vision model such as gemma3/gemma4 or qwen2.5vl)
  * Regenerate & export (redo the last answer with one click; download any
                       conversation as Markdown or JSON)
  * Generation stats  (every answer shows measured tokens/second, token
                       count and wall time, straight from Ollama)
  * Remote access     (run with --host 0.0.0.0 and open Heorth from a phone
                       on the same network; optional access password protects
                       every device that is not this machine itself)
  * Hardened local API (Host/Origin checks block DNS-rebinding and cross-site
                       request forgery; "Run app" artifacts are sandboxed so
                       generated apps cannot call Heorth's own API)
  * Runnable artifacts (code blocks in chat get a Copy button, and HTML
                       apps get a "Run app" button that saves the file and
                       opens it in a new browser tab)
  * Coder mode        (an opencode-style coding agent that works on a real
                       project folder: tree/read/grep/edit/write/shell tools,
                       locked inside the folder you choose; read-only "plan
                       mode" by default, direct edits only after opt-in)

Chat modes: plain chat, Knowledge (answers grounded in your documents),
Agent (single-turn tool use), Loop (autonomous agent: it plans, calls
tools, observes and repeats until it declares the task complete), and
Council (a panel of consultants — each analyzes the question independently
and in parallel, then they read and critique each other over one or more
consultation rounds, and a chair writes the final synthesized answer), and
Coder (a coding agent in the spirit of opencode / Claude Code: point it at a
project folder and it explores, edits and tests the code autonomously).

Updating: download a newer heorth*.py (or legacy localmind*.py) file into
your Downloads folder, or into the data folder's updates/ directory, or drop
it onto the Settings page. The GUI notices the newer version and shows an
"Update" button; clicking it backs up the current version, swaps the file
and restarts in place.

Flags:
    --port N        listen on another port          (default 8317)
    --host ADDR     bind address, 0.0.0.0 for LAN   (default 127.0.0.1)
    --no-browser    do not open the browser on start
    --system        use the current Python env instead of a private venv
"""

__version__ = "1.8.1"

APP_NAME = "Heorth"
FILE_STEM = "heorth"                       # used for backups and new installs
FILE_STEMS = ("heorth", "localmind")       # update files may start with either
DEFAULT_PORT = 8317
# Filled in by main() with the host/port the server actually bound, so a
# self-update can restart on the exact same address the browser is using.
RUNTIME = {"host": "127.0.0.1", "port": DEFAULT_PORT}
# Session tokens handed out after a correct LAN password (memory-only: a
# server restart simply asks devices to unlock again).
_AUTH_TOKENS: set = set()

import difflib
import fnmatch
import hmac
import ipaddress
import secrets
import json
import os
import platform
import re
import shutil
import socket
import sqlite3
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

# --------------------------------------------------------------------------
# Paths (defined before bootstrap so the venv can live in the data folder)
# --------------------------------------------------------------------------

SCRIPT_PATH = Path(os.path.abspath(__file__))
BASE_DIR = SCRIPT_PATH.parent


def _default_data_dir() -> Path:
    for name in ("heorth_data", "localmind_data"):   # keep legacy data
        if (BASE_DIR / name).exists():
            return BASE_DIR / name
    return BASE_DIR / "heorth_data"


DATA_DIR = Path(os.environ.get("HEORTH_DATA")
                or os.environ.get("LOCALMIND_DATA")
                or _default_data_dir())

for _sub in ("", "images", "rag_docs", "workspace", "backups", "updates",
             "artifacts"):
    (DATA_DIR / _sub).mkdir(parents=True, exist_ok=True)


def _db_file() -> Path:
    for name in ("heorth.db", "localmind.db"):       # keep legacy database
        if (DATA_DIR / name).exists():
            return DATA_DIR / name
    return DATA_DIR / "heorth.db"


DB_PATH = _db_file()
CONFIG_PATH = DATA_DIR / "config.json"
VENV_DIR = DATA_DIR / "venv"

CORE_DEPS = [
    "fastapi>=0.110",
    "uvicorn>=0.29",
    "httpx>=0.27",
    "psutil>=5.9",
    "numpy>=1.26",
    "python-multipart>=0.0.9",
    "pypdf>=4.0",
]
CORE_IMPORTS = ["fastapi", "uvicorn", "httpx", "psutil", "numpy", "multipart", "pypdf"]


def _venv_python() -> Path:
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def _core_deps_ok(python_exe: str) -> bool:
    code = "import " + ",".join(CORE_IMPORTS)
    try:
        r = subprocess.run([python_exe, "-c", code], capture_output=True, timeout=120)
        return r.returncode == 0
    except Exception:
        return False


def _pip_install(python_exe: str, packages) -> bool:
    print(f"[{APP_NAME}] installing: {' '.join(packages)} (one-time, please wait)")
    cmd = [python_exe, "-m", "pip", "install", "--disable-pip-version-check", *packages]
    return subprocess.call(cmd) == 0


def ensure_environment() -> None:
    """Make sure core dependencies exist; if not, build a private venv and
    re-run this script inside it. Safe to call on every start."""
    if os.environ.get("HEORTH_BOOTSTRAPPED") == "1" \
            or os.environ.get("LOCALMIND_BOOTSTRAPPED") == "1":
        return  # already re-exec'd once; trust it and fail loudly if broken

    try:  # happy path: everything already importable in this interpreter
        for _m in CORE_IMPORTS:
            __import__(_m)
        return
    except Exception:
        pass

    if "--system" in sys.argv:
        if not _pip_install(sys.executable, CORE_DEPS) or not _core_deps_ok(sys.executable):
            print(f"[{APP_NAME}] could not install dependencies into the current "
                  f"Python environment. Re-run without --system to use a private venv.")
            sys.exit(1)
        return

    vpy = _venv_python()
    if not vpy.exists():
        print(f"[{APP_NAME}] first run: creating a private Python environment in\n"
              f"            {VENV_DIR}")
        import venv as _venv
        try:
            _venv.EnvBuilder(with_pip=True, clear=False).create(str(VENV_DIR))
        except Exception as e:
            print(f"[{APP_NAME}] failed to create venv ({e}). Trying current environment…")
            if _pip_install(sys.executable, CORE_DEPS) and _core_deps_ok(sys.executable):
                return
            sys.exit(1)

    if not _core_deps_ok(str(vpy)):
        if not _pip_install(str(vpy), CORE_DEPS) or not _core_deps_ok(str(vpy)):
            print(f"[{APP_NAME}] dependency install failed. Check your internet "
                  f"connection and try again, or delete {VENV_DIR} to retry from scratch.")
            sys.exit(1)

    env = dict(os.environ, HEORTH_BOOTSTRAPPED="1")
    print(f"[{APP_NAME}] restarting inside its private environment…")
    rc = subprocess.call([str(vpy), str(SCRIPT_PATH), *sys.argv[1:]], env=env)
    sys.exit(rc)


ensure_environment()

# --------------------------------------------------------------------------
# Heavy imports (safe after bootstrap)
# --------------------------------------------------------------------------

import asyncio
import base64
import hashlib
import io
import mimetypes
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Optional

import httpx
import numpy as np
import psutil
import uvicorn
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse

# --------------------------------------------------------------------------
# Database + settings
# --------------------------------------------------------------------------

_db_lock = threading.RLock()
_db = sqlite3.connect(DB_PATH, check_same_thread=False)
_db.row_factory = sqlite3.Row
_db.execute("PRAGMA journal_mode=WAL")

SCHEMA = """
CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS conversations(
  id TEXT PRIMARY KEY, title TEXT, created REAL);
CREATE TABLE IF NOT EXISTS messages(
  id TEXT PRIMARY KEY, conv_id TEXT, role TEXT, content TEXT,
  meta TEXT, created REAL);
CREATE TABLE IF NOT EXISTS docs(
  id TEXT PRIMARY KEY, name TEXT, created REAL, chunks INTEGER);
CREATE TABLE IF NOT EXISTS chunks(
  id TEXT PRIMARY KEY, doc_id TEXT, idx INTEGER, text TEXT, embedding BLOB);
CREATE TABLE IF NOT EXISTS images(
  id TEXT PRIMARY KEY, filename TEXT, prompt TEXT, model TEXT,
  width INTEGER, height INTEGER, seed INTEGER, created REAL);
CREATE TABLE IF NOT EXISTS mcp_servers(
  id TEXT PRIMARY KEY, name TEXT, command TEXT, args TEXT, env TEXT,
  enabled INTEGER DEFAULT 1);
"""


def db_init() -> None:
    with _db_lock:
        _db.executescript(SCHEMA)
        _db.commit()


def q(sql: str, params: tuple = ()) -> list:
    with _db_lock:
        return [dict(r) for r in _db.execute(sql, params).fetchall()]


def qx(sql: str, params: tuple = ()) -> None:
    with _db_lock:
        _db.execute(sql, params)
        _db.commit()


def qmany(sql: str, seq: list) -> None:
    """executemany + a single commit — much faster for bulk inserts."""
    with _db_lock:
        _db.executemany(sql, seq)
        _db.commit()


DEFAULT_SETTINGS = {
    "ollama_host": os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"),
    "system_prompt": "You are a helpful, precise assistant running fully "
                     "locally on the user's machine.",
    "embed_model": "nomic-embed-text",
    "rag_top_k": "5",
    "context_messages": "30",
    "allow_code_execution": "0",
    "allow_web_tools": "1",
    "auto_search": "1",
    "agent_max_steps": "8",
    "loop_max_steps": "15",
    "coder_root": "",
    "coder_allow_write": "0",
    "coder_max_steps": "25",
    "lan_password": "",
    "council_size": "4",
    "council_rounds": "1",
    "council_research": "0",
    "computer_control": "0",
    "computer_confirm": "1",
    "computer_pause": "0.4",
    "image_model": "sd-turbo",
    "image_force_fp32": "0",
    "search_backend": "auto",
    "searxng_url": "http://127.0.0.1:8888",
    "theme": "dark",
}


def get_setting(key: str) -> str:
    rows = q("SELECT value FROM settings WHERE key=?", (key,))
    if rows:
        return rows[0]["value"]
    return DEFAULT_SETTINGS.get(key, "")


def set_setting(key: str, value: str) -> None:
    qx("INSERT INTO settings(key,value) VALUES(?,?) "
       "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, str(value)))


def flag(key: str) -> bool:
    """True when a boolean setting is switched on."""
    return str(get_setting(key)) == "1"


def all_settings() -> dict:
    merged = dict(DEFAULT_SETTINGS)
    for row in q("SELECT key,value FROM settings"):
        merged[row["key"]] = row["value"]
    return merged


def now() -> float:
    return time.time()


def new_id() -> str:
    return uuid.uuid4().hex[:16]

# --------------------------------------------------------------------------
# Hardware detection + model recommendation
# --------------------------------------------------------------------------


def _nvidia_gpus() -> list:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        gpus = []
        for line in out.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2 and parts[-1].isdigit():
                gpus.append({"name": ",".join(parts[:-1]),
                             "vram_gb": round(int(parts[-1]) / 1024, 1),
                             "kind": "nvidia"})
        return gpus
    except Exception:
        return []


def _amd_gpus() -> list:
    exe = shutil.which("rocm-smi")
    if not exe:
        return []
    try:
        out = subprocess.run([exe, "--showmeminfo", "vram", "--json"],
                             capture_output=True, text=True, timeout=10)
        data = json.loads(out.stdout or "{}")
        gpus = []
        for key, card in data.items():
            total = card.get("VRAM Total Memory (B)") or card.get("vram_total")
            if total:
                gpus.append({"name": f"AMD GPU ({key})",
                             "vram_gb": round(int(total) / 1024 ** 3, 1),
                             "kind": "amd"})
        return gpus
    except Exception:
        return []


def detect_hardware() -> dict:
    vm = psutil.virtual_memory()
    ram_gb = round(vm.total / 1024 ** 3, 1)
    du = shutil.disk_usage(str(DATA_DIR))
    info = {
        "os": platform.system(),
        "os_pretty": f"{platform.system()} {platform.release()}",
        "arch": platform.machine(),
        "cpu": platform.processor() or platform.machine(),
        "cpu_cores": psutil.cpu_count(logical=False) or psutil.cpu_count() or 1,
        "cpu_threads": psutil.cpu_count() or 1,
        "ram_gb": ram_gb,
        "ram_used_gb": round((vm.total - vm.available) / 1024 ** 3, 1),
        "disk_free_gb": round(du.free / 1024 ** 3, 1),
        "gpus": [],
        "apple_silicon": False,
        "backend": "cpu",
        "usable_gb": round(min(ram_gb * 0.6, max(ram_gb - 6, 2)), 1),
    }
    if platform.system() == "Darwin" and platform.machine() in ("arm64", "aarch64"):
        info["apple_silicon"] = True
        info["backend"] = "metal"
        info["gpus"] = [{"name": "Apple Silicon GPU (unified memory)",
                         "vram_gb": ram_gb, "kind": "apple"}]
        info["usable_gb"] = round(ram_gb * 0.7, 1)
    else:
        gpus = _nvidia_gpus() or _amd_gpus()
        info["gpus"] = gpus
        if gpus:
            info["backend"] = gpus[0]["kind"]
            info["usable_gb"] = round(max(g["vram_gb"] for g in gpus) * 0.92, 1)
    return info


# Curated catalog. size_gb = approximate download, need_gb = memory to run
# comfortably at the default (Q4) quantization.
MODEL_CATALOG = [
 {"id": "llama3.2:1b", "name": "Llama 3.2 1B", "size_gb": 1.3, "need_gb": 2.5,
  "tags": ["chat", "tools", "tiny"], "desc": "Very small and fast; fine for quick drafting on any machine."},
 {"id": "llama3.2:3b", "name": "Llama 3.2 3B", "size_gb": 2.0, "need_gb": 4,
  "tags": ["chat", "tools", "tiny"], "desc": "Small all-rounder from Meta; good quality for its size."},
 {"id": "llama3.1:8b", "name": "Llama 3.1 8B", "size_gb": 4.9, "need_gb": 8,
  "tags": ["chat", "tools"], "desc": "The classic dependable 8B; strong general chat and tool calling."},
 {"id": "llama3.3:70b", "name": "Llama 3.3 70B", "size_gb": 43, "need_gb": 48,
  "tags": ["chat", "tools", "reasoning"], "desc": "Near-frontier quality; needs a large GPU or a big Mac."},
 {"id": "llama4:scout", "name": "Llama 4 Scout", "size_gb": 63, "need_gb": 72,
  "tags": ["chat", "tools", "vision", "reasoning"], "desc": "Meta's MoE flagship (17B active); multimodal, for big workstations."},
 {"id": "gpt-oss:20b", "name": "GPT-OSS 20B", "size_gb": 13, "need_gb": 16,
  "tags": ["chat", "tools", "reasoning"], "desc": "OpenAI's open-weight reasoner; excellent tool use on 16 GB."},
 {"id": "gemma4:12b", "name": "Gemma 4 12B", "size_gb": 8.5, "need_gb": 13,
  "tags": ["chat", "vision"], "desc": "Google's 2026 Gemma; strong writing and vision in 16 GB of RAM."},
 {"id": "qwen3-coder:30b", "name": "Qwen 3 Coder 30B", "size_gb": 19, "need_gb": 24,
  "tags": ["code", "tools", "reasoning"], "desc": "MoE coding specialist (3B active) — a top pick for Coder mode."},
 {"id": "devstral:24b", "name": "Devstral 24B", "size_gb": 14, "need_gb": 20,
  "tags": ["code", "tools"], "desc": "Mistral's agentic coding model — built for tool-driven Coder workflows."},
 {"id": "qwen3:0.6b", "name": "Qwen 3 0.6B", "size_gb": 0.5, "need_gb": 1.5,
  "tags": ["chat", "tools", "tiny"], "desc": "Tiny but surprisingly capable; runs on almost anything."},
 {"id": "qwen3:1.7b", "name": "Qwen 3 1.7B", "size_gb": 1.4, "need_gb": 3,
  "tags": ["chat", "tools", "tiny"], "desc": "Compact hybrid-reasoning model."},
 {"id": "qwen3:4b", "name": "Qwen 3 4B", "size_gb": 2.6, "need_gb": 5,
  "tags": ["chat", "tools", "reasoning"], "desc": "Excellent quality-per-GB; great laptop default."},
 {"id": "qwen3:8b", "name": "Qwen 3 8B", "size_gb": 5.2, "need_gb": 8.5,
  "tags": ["chat", "tools", "reasoning"], "desc": "Strong reasoning and tool use; top pick around 8 GB."},
 {"id": "qwen3:14b", "name": "Qwen 3 14B", "size_gb": 9.3, "need_gb": 13,
  "tags": ["chat", "tools", "reasoning"], "desc": "Noticeable step up in depth; great on 16 GB GPUs/Macs."},
 {"id": "qwen3:32b", "name": "Qwen 3 32B", "size_gb": 20, "need_gb": 24,
  "tags": ["chat", "tools", "reasoning"], "desc": "Heavyweight reasoning for 24 GB+ setups."},
 {"id": "gemma3:1b", "name": "Gemma 3 1B", "size_gb": 0.8, "need_gb": 2,
  "tags": ["chat", "tiny"], "desc": "Google's smallest Gemma 3; snappy on CPUs."},
 {"id": "gemma3:4b", "name": "Gemma 3 4B", "size_gb": 3.3, "need_gb": 5.5,
  "tags": ["chat", "vision"], "desc": "Small multimodal model — it can also look at images."},
 {"id": "gemma3:12b", "name": "Gemma 3 12B", "size_gb": 8.1, "need_gb": 12,
  "tags": ["chat", "vision"], "desc": "Great writing quality plus vision, for 12–16 GB."},
 {"id": "gemma3:27b", "name": "Gemma 3 27B", "size_gb": 17, "need_gb": 22,
  "tags": ["chat", "vision", "reasoning"], "desc": "Gemma flagship; superb general model for 24 GB+."},
 {"id": "phi4:14b", "name": "Phi-4 14B", "size_gb": 9.1, "need_gb": 13,
  "tags": ["chat", "reasoning"], "desc": "Microsoft's dense 14B, strong at math and logic."},
 {"id": "mistral:7b", "name": "Mistral 7B", "size_gb": 4.1, "need_gb": 7,
  "tags": ["chat", "tools"], "desc": "Fast, efficient European classic."},
 {"id": "mistral-small3.2:24b", "name": "Mistral Small 3.2 24B", "size_gb": 15, "need_gb": 20,
  "tags": ["chat", "tools", "vision"], "desc": "Punches far above its weight; vision + tools."},
 {"id": "qwen2.5-coder:7b", "name": "Qwen 2.5 Coder 7B", "size_gb": 4.7, "need_gb": 8,
  "tags": ["code", "tools"], "desc": "Dedicated coding model; great autocomplete and refactors."},
 {"id": "qwen2.5-coder:14b", "name": "Qwen 2.5 Coder 14B", "size_gb": 9.0, "need_gb": 13,
  "tags": ["code", "tools"], "desc": "Stronger coding, still laptop-friendly on 16 GB."},
 {"id": "deepseek-r1:8b", "name": "DeepSeek R1 8B", "size_gb": 5.2, "need_gb": 8.5,
  "tags": ["reasoning", "chat"], "desc": "Distilled reasoning model that thinks step by step."},
 {"id": "deepseek-r1:14b", "name": "DeepSeek R1 14B", "size_gb": 9.0, "need_gb": 13,
  "tags": ["reasoning", "chat"], "desc": "Bigger R1 distill; strong maths/logic for 16 GB."},
 {"id": "qwen2.5vl:7b", "name": "Qwen 2.5 VL 7B", "size_gb": 6.0, "need_gb": 9,
  "tags": ["vision", "chat"], "desc": "Vision-language model for screenshots, photos, documents."},
 {"id": "llava:7b", "name": "LLaVA 7B", "size_gb": 4.7, "need_gb": 8,
  "tags": ["vision", "chat"], "desc": "Classic open vision assistant."},
 {"id": "smollm2:1.7b", "name": "SmolLM2 1.7B", "size_gb": 1.8, "need_gb": 3,
  "tags": ["chat", "tiny"], "desc": "HuggingFace's small model; ideal for weak hardware."},
 {"id": "nomic-embed-text", "name": "Nomic Embed Text", "size_gb": 0.3, "need_gb": 1,
  "tags": ["embed"], "desc": "Embedding model used by the knowledge base (RAG)."},
 {"id": "mxbai-embed-large", "name": "MxBai Embed Large", "size_gb": 0.7, "need_gb": 1.5,
  "tags": ["embed"], "desc": "Higher-quality embeddings, slightly slower."},
]


def recommend_models(hw: dict) -> dict:
    usable = hw["usable_gb"]
    chat = [m for m in MODEL_CATALOG
            if "embed" not in m["tags"] and m["need_gb"] <= usable]
    chat.sort(key=lambda m: m["need_gb"], reverse=True)
    top = chat[:6]
    best = top[0] if top else None
    if usable < 3:
        tier = "Very limited — stick to tiny models; answers will be slow but usable."
    elif usable < 6:
        tier = "Entry level — small models (1–4B) will run nicely."
    elif usable < 11:
        tier = "Solid — 7–8B models run well; this is the sweet spot for daily use."
    elif usable < 20:
        tier = "Strong — 12–14B models fit comfortably."
    elif usable < 40:
        tier = "Enthusiast — 24–32B models fit; expect excellent quality."
    else:
        tier = "Workstation class — 70B models are within reach."
    if hw["backend"] == "cpu":
        tier += " (No GPU detected: generation runs on CPU, so prefer the smaller picks.)"
    return {"tier": tier, "usable_gb": usable, "best": best, "picks": top}

# --------------------------------------------------------------------------
# Ollama client
# --------------------------------------------------------------------------


def ollama_url(path: str) -> str:
    return get_setting("ollama_host").rstrip("/") + path


async def ollama_up() -> dict:
    try:
        async with httpx.AsyncClient(timeout=3) as c:
            r = await c.get(ollama_url("/api/version"))
            if r.status_code == 200:
                return {"up": True, "version": r.json().get("version", "?")}
    except Exception:
        pass
    return {"up": False, "version": None}


async def ollama_installed_models() -> list:
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(ollama_url("/api/tags"))
            models = []
            for m in r.json().get("models", []):
                models.append({
                    "name": m.get("name", ""),
                    "size_gb": round(m.get("size", 0) / 1024 ** 3, 1),
                    "family": (m.get("details") or {}).get("family", ""),
                    "params": (m.get("details") or {}).get("parameter_size", ""),
                    "quant": (m.get("details") or {}).get("quantization_level", ""),
                    "modified": m.get("modified_at", ""),
                })
            return models
    except Exception:
        return []


async def ollama_pull_stream(name: str) -> AsyncGenerator[str, None]:
    """Forward Ollama's pull progress as NDJSON lines."""
    try:
        async with httpx.AsyncClient(timeout=None) as c:
            async with c.stream("POST", ollama_url("/api/pull"),
                                json={"name": name, "stream": True}) as r:
                async for line in r.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        d = json.loads(line)
                    except Exception:
                        continue
                    out = {"type": "progress",
                           "status": d.get("status", ""),
                           "total": d.get("total"),
                           "completed": d.get("completed")}
                    if d.get("error"):
                        out = {"type": "error", "error": d["error"]}
                    yield json.dumps(out) + "\n"
        yield json.dumps({"type": "done"}) + "\n"
    except Exception as e:
        yield json.dumps({"type": "error", "error": f"Pull failed: {e}"}) + "\n"


async def ollama_delete(name: str) -> bool:
    try:
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.request("DELETE", ollama_url("/api/delete"),
                                json={"name": name})
            return r.status_code == 200
    except Exception:
        return False


async def ollama_chat(messages: list, model: str, tools: Optional[list] = None,
                      options: Optional[dict] = None,
                      think: Optional[bool] = None):
    """Non-streaming chat call (used by the agent for tool-calling steps)."""
    payload: dict = {"model": model, "messages": messages, "stream": False}
    if tools:
        payload["tools"] = tools
    if options:
        payload["options"] = options
    if think is not None:
        payload["think"] = think          # some models reason in a separate field
    async with httpx.AsyncClient(timeout=600) as c:
        r = await c.post(ollama_url("/api/chat"), json=payload)
        r.raise_for_status()
        return r.json()


async def ollama_stream_with_heartbeat(messages: list, model: str,
                                       think: Optional[bool] = None):
    """Wrap ollama_chat_stream and, during long silent stretches (model
    loading or prompt processing on slower machines), periodically yield a
    synthetic heartbeat so the UI can show progress instead of a frozen
    spinner. Heartbeats are dicts with {'_heartbeat': seconds}."""
    q_out: "asyncio.Queue" = asyncio.Queue()
    DONE = object()

    async def producer():
        try:
            async for part in ollama_chat_stream(messages, model, think=think):
                await q_out.put(part)
        except Exception as e:
            await q_out.put(("_error", e))
        finally:
            await q_out.put(DONE)

    task = asyncio.create_task(producer())
    waited = 0.0
    try:
        while True:
            try:
                item = await asyncio.wait_for(q_out.get(), timeout=2.0)
            except asyncio.TimeoutError:
                waited += 2.0
                yield {"_heartbeat": waited}
                continue
            if item is DONE:
                return
            if isinstance(item, tuple) and item and item[0] == "_error":
                raise item[1]
            waited = 0.0
            yield item
    finally:
        if not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass


async def ollama_chat_stream(messages: list, model: str,
                             think: Optional[bool] = None) -> AsyncGenerator[dict, None]:
    payload: dict = {"model": model, "messages": messages, "stream": True}
    if think is not None:
        payload["think"] = think          # some models reason in a separate field
    async with httpx.AsyncClient(timeout=None) as c:
        async with c.stream("POST", ollama_url("/api/chat"), json=payload) as r:
            if r.status_code != 200:
                body = (await r.aread()).decode(errors="replace")[:500]
                raise RuntimeError(f"Ollama error {r.status_code}: {body}")
            async for line in r.aiter_lines():
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    continue


def _gen_stats(part: dict) -> Optional[dict]:
    """Turn the statistics Ollama reports at the end of a generation into a
    small UI event: measured tokens/second, token counts, wall time."""
    try:
        ec = int(part.get("eval_count") or 0)
        ed = int(part.get("eval_duration") or 0)
        if ec <= 0 or ed <= 0:
            return None
        return {"type": "stats",
                "tps": round(ec / (ed / 1e9), 1),
                "tokens": ec,
                "prompt_tokens": int(part.get("prompt_eval_count") or 0),
                "seconds": round(int(part.get("total_duration") or ed) / 1e9, 1)}
    except Exception:
        return None


async def ollama_embed(texts: list) -> Optional[list]:
    model = get_setting("embed_model")
    try:
        async with httpx.AsyncClient(timeout=300) as c:
            r = await c.post(ollama_url("/api/embed"),
                             json={"model": model, "input": texts})
            if r.status_code == 200:
                return r.json().get("embeddings")
            # fall back to the legacy one-at-a-time endpoint
            vecs = []
            for t in texts:
                r2 = await c.post(ollama_url("/api/embeddings"),
                                  json={"model": model, "prompt": t})
                r2.raise_for_status()
                vecs.append(r2.json()["embedding"])
            return vecs
    except Exception:
        return None


def ollama_install_help(hw: dict) -> dict:
    os_name = hw["os"]
    if os_name == "Darwin":
        return {"os": "macOS",
                "steps": ["Download the app from https://ollama.com/download/mac, "
                          "open it once, then come back here.",
                          "Or with Homebrew:  brew install ollama   then run:  ollama serve"],
                "url": "https://ollama.com/download/mac"}
    if os_name == "Windows":
        return {"os": "Windows",
                "steps": ["Download and run the installer from "
                          "https://ollama.com/download/windows, then come back here."],
                "url": "https://ollama.com/download/windows"}
    return {"os": "Linux",
            "steps": ["Run in a terminal:  curl -fsSL https://ollama.com/install.sh | sh",
                      "If it did not start automatically:  ollama serve"],
            "url": "https://ollama.com/download/linux"}

# --------------------------------------------------------------------------
# Model search (curated catalog + Hugging Face GGUF)
# --------------------------------------------------------------------------


async def search_models(query: str) -> dict:
    ql = query.lower().strip()
    local = [m for m in MODEL_CATALOG
             if ql in m["id"].lower() or ql in m["name"].lower()
             or any(ql in t for t in m["tags"])]
    hf = []
    try:
        async with httpx.AsyncClient(timeout=12) as c:
            r = await c.get("https://huggingface.co/api/models",
                            params={"search": query, "filter": "gguf",
                                    "sort": "downloads", "limit": 12})
            if r.status_code == 200:
                for m in r.json():
                    mid = m.get("id") or m.get("modelId", "")
                    if not mid:
                        continue
                    hf.append({
                        "id": f"hf.co/{mid}",
                        "name": mid,
                        "downloads": m.get("downloads", 0),
                        "likes": m.get("likes", 0),
                        "source": "huggingface",
                        "desc": "GGUF repo on Hugging Face — pulls the default "
                                "quantization; add :Q4_K_M etc. to choose one.",
                    })
    except Exception:
        pass
    return {"catalog": local, "huggingface": hf}

# --------------------------------------------------------------------------
# RAG: extract → chunk → embed → search
# --------------------------------------------------------------------------

_rag_cache: dict = {"matrix": None, "rows": None, "dirty": True}


def _extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="replace")


def _chunk_text(text: str, size: int = 1400, overlap: int = 200) -> list:
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []
    chunks, start = [], 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            cut = text.rfind("\n", start + int(size * 0.5), end)
            if cut == -1:
                cut = text.rfind(". ", start + int(size * 0.5), end)
            if cut != -1:
                end = cut + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        start = max(end - overlap, start + 1)
    return chunks


async def rag_add_document(filename: str, raw: bytes) -> dict:
    safe = re.sub(r"[^A-Za-z0-9._ -]", "_", filename)[:120] or "document.txt"
    doc_id = new_id()
    stored = DATA_DIR / "rag_docs" / f"{doc_id}_{safe}"
    stored.write_bytes(raw)
    try:
        text = _extract_text(stored)
    except Exception as e:
        stored.unlink(missing_ok=True)
        return {"ok": False, "error": f"Could not read {safe}: {e}"}
    chunks = _chunk_text(text)
    if not chunks:
        stored.unlink(missing_ok=True)
        return {"ok": False, "error": f"No readable text found in {safe}."}
    embeddings = await ollama_embed(chunks)
    if embeddings is None:
        stored.unlink(missing_ok=True)
        return {"ok": False, "error":
                "Embedding failed. Make sure Ollama is running and the "
                f"embedding model '{get_setting('embed_model')}' is pulled "
                "(Models page → search 'embed')."}
    qx("INSERT INTO docs(id,name,created,chunks) VALUES(?,?,?,?)",
       (doc_id, safe, now(), len(chunks)))
    qmany("INSERT INTO chunks(id,doc_id,idx,text,embedding) VALUES(?,?,?,?,?)",
          [(new_id(), doc_id, i, piece,
            np.asarray(vec, dtype=np.float32).tobytes())
           for i, (piece, vec) in enumerate(zip(chunks, embeddings))])
    _rag_cache["dirty"] = True
    return {"ok": True, "doc": {"id": doc_id, "name": safe, "chunks": len(chunks)}}


def rag_delete_document(doc_id: str) -> None:
    qx("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
    qx("DELETE FROM docs WHERE id=?", (doc_id,))
    for f in (DATA_DIR / "rag_docs").glob(f"{doc_id}_*"):
        f.unlink(missing_ok=True)
    _rag_cache["dirty"] = True


def _rag_matrix():
    if _rag_cache["dirty"] or _rag_cache["matrix"] is None:
        rows = q("SELECT chunks.id, doc_id, idx, text, embedding, docs.name AS doc_name "
                 "FROM chunks JOIN docs ON docs.id = chunks.doc_id")
        if rows:
            mat = np.vstack([np.frombuffer(r["embedding"], dtype=np.float32)
                             for r in rows])
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms == 0] = 1
            _rag_cache["matrix"] = mat / norms
        else:
            _rag_cache["matrix"] = None
        _rag_cache["rows"] = rows
        _rag_cache["dirty"] = False
    return _rag_cache["matrix"], _rag_cache["rows"]


async def rag_search(query: str, top_k: Optional[int] = None) -> list:
    mat, rows = _rag_matrix()
    if mat is None or not rows:
        return []
    vecs = await ollama_embed([query])
    if not vecs:
        return []
    v = np.asarray(vecs[0], dtype=np.float32)
    n = np.linalg.norm(v)
    if n == 0:
        return []
    scores = mat @ (v / n)
    k = top_k or int(get_setting("rag_top_k") or 5)
    order = np.argsort(-scores)[:k]
    return [{"doc": rows[i]["doc_name"], "text": rows[i]["text"],
             "score": float(scores[i])} for i in order]

# --------------------------------------------------------------------------
# Background pip installs (image generation / MCP extras) with live logs
# --------------------------------------------------------------------------

_install_jobs: dict = {}   # name -> {"status": "running|done|failed", "log": [..]}


def _run_job(job_name: str, commands: list) -> None:
    """Run a sequence of commands, streaming their output into the job log."""
    job = _install_jobs[job_name]
    try:
        for cmd in commands:
            job["log"].append("$ " + " ".join(cmd))
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True,
                                    bufsize=1)
            for line in proc.stdout:  # type: ignore[union-attr]
                line = line.rstrip()
                if line:
                    job["log"].append(line[-300:])
                    if len(job["log"]) > 400:
                        del job["log"][:100]
            proc.wait()
            job["log"].append(f"[exit code {proc.returncode}]")
            if proc.returncode != 0:
                job["status"] = "failed"
                return
        job["status"] = "done"
    except Exception as e:
        job["status"] = "failed"
        job["log"].append(f"error: {e}")


def start_job(job_name: str, commands: list) -> dict:
    existing = _install_jobs.get(job_name)
    if existing and existing["status"] == "running":
        return {"ok": True, "already": True}
    _install_jobs[job_name] = {"status": "running", "log": [], "started": now()}
    threading.Thread(target=_run_job, args=(job_name, commands),
                     daemon=True).start()
    return {"ok": True}


def start_install(job_name: str, packages: list) -> dict:
    return start_job(job_name, [[sys.executable, "-m", "pip", "install",
                                 "--disable-pip-version-check", *packages]])

# --------------------------------------------------------------------------
# Optional private web search: SearXNG in Docker
# --------------------------------------------------------------------------

SEARXNG_CONTAINER = "heorth-searxng"
SEARXNG_CONTAINERS = ("heorth-searxng", "localmind-searxng")   # legacy kept
SEARXNG_DIR = DATA_DIR / "searxng"


def docker_state() -> dict:
    exe = shutil.which("docker")
    if not exe:
        return {"installed": False, "daemon": False}
    try:
        r = subprocess.run([exe, "info", "--format", "{{.ServerVersion}}"],
                           capture_output=True, text=True, timeout=8)
        return {"installed": True, "daemon": r.returncode == 0}
    except Exception:
        return {"installed": True, "daemon": False}


def docker_install_help(hw: dict) -> list:
    if hw["os"] == "Darwin":
        return ["Install Docker Desktop for Mac: "
                "https://www.docker.com/products/docker-desktop/",
                "Open it once and wait until the whale icon says it's running."]
    if hw["os"] == "Windows":
        return ["Install Docker Desktop for Windows: "
                "https://www.docker.com/products/docker-desktop/",
                "Open it once and wait until it reports 'Engine running'."]
    return ["Install Docker Engine:  curl -fsSL https://get.docker.com | sh",
            "Start it:  sudo systemctl enable --now docker",
            "Optional, to run Docker without sudo:  "
            "sudo usermod -aG docker $USER   (then log out and back in)"]


def _searxng_port() -> int:
    from urllib.parse import urlparse
    try:
        return urlparse(get_setting("searxng_url")).port or 8888
    except Exception:
        return 8888


def _searxng_write_config() -> None:
    """Minimal SearXNG settings with the JSON API enabled, so Heorth can
    query it programmatically. Only written once, never overwritten."""
    SEARXNG_DIR.mkdir(parents=True, exist_ok=True)
    cfg = SEARXNG_DIR / "settings.yml"
    if cfg.exists():
        return
    import secrets
    cfg.write_text(
        "# generated by Heorth — minimal SearXNG config, JSON API enabled\n"
        "use_default_settings: true\n"
        "server:\n"
        f"  secret_key: \"{secrets.token_hex(32)}\"\n"
        "  limiter: false\n"
        "  image_proxy: true\n"
        "search:\n"
        "  formats:\n"
        "    - html\n"
        "    - json\n", encoding="utf-8")


def searxng_container_state() -> dict:
    exe = shutil.which("docker")
    if not exe:
        return {"exists": False, "running": False, "status": "", "name": ""}
    for cname in SEARXNG_CONTAINERS:
        try:
            r = subprocess.run([exe, "ps", "-a", "--filter",
                                f"name=^{cname}$",
                                "--format", "{{.Status}}"],
                               capture_output=True, text=True, timeout=8)
            lines = (r.stdout or "").strip().splitlines()
            status = lines[0] if lines else ""
            if status:
                return {"exists": True,
                        "running": status.lower().startswith("up"),
                        "status": status, "name": cname}
        except Exception:
            continue
    return {"exists": False, "running": False, "status": "", "name": ""}


def searxng_manual_cmd() -> str:
    return ("docker run -d --name " + SEARXNG_CONTAINER +
            f" -p {_searxng_port()}:8080 -v \"{SEARXNG_DIR}\":/etc/searxng"
            " --restart unless-stopped searxng/searxng")


def searxng_start() -> dict:
    st = docker_state()
    if not st["installed"]:
        return {"ok": False, "error": "Docker is not installed."}
    if not st["daemon"]:
        return {"ok": False, "error": "Docker is installed but not running."}
    _searxng_write_config()
    cont = searxng_container_state()
    if cont["exists"]:
        cmd = ["docker", "start", cont["name"]]
    else:
        cmd = ["docker", "run", "-d", "--name", SEARXNG_CONTAINER,
               "-p", f"{_searxng_port()}:8080",
               "-v", f"{SEARXNG_DIR}:/etc/searxng",
               "--restart", "unless-stopped", "searxng/searxng"]
    return start_job("searxng", [cmd])


async def searxng_probe() -> dict:
    url = get_setting("searxng_url").rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=6) as c:
            r = await c.get(url + "/search",
                            params={"q": "localmind", "format": "json"})
        if r.status_code == 200:
            try:
                r.json()
                return {"reachable": True, "json_ok": True, "error": None}
            except Exception:
                return {"reachable": True, "json_ok": False,
                        "error": "SearXNG answered, but not with JSON."}
        if r.status_code == 403:
            return {"reachable": True, "json_ok": False,
                    "error": "SearXNG is up but its JSON API is off. Add "
                             "'json' under search.formats in settings.yml, "
                             "then restart the container."}
        return {"reachable": True, "json_ok": False,
                "error": f"SearXNG answered with HTTP {r.status_code}."}
    except Exception as e:
        return {"reachable": False, "json_ok": False, "error": str(e)[:200]}


async def _searxng_search(query: str) -> Optional[str]:
    url = get_setting("searxng_url").rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.get(url + "/search",
                            params={"q": query, "format": "json"})
            if r.status_code != 200:
                return None
            data = r.json()
        lines = []
        for a in (data.get("answers") or [])[:2]:
            text = a.get("answer") if isinstance(a, dict) else str(a)
            if text:
                lines.append(f"Answer: {str(text)[:300]}")
        for res in (data.get("results") or [])[:6]:
            title = (res.get("title") or "").strip()
            u = res.get("url") or ""
            content = re.sub(r"\s+", " ", res.get("content") or "").strip()
            lines.append(f"- {title}\n  {u}\n  {content[:240]}")
        if not lines:
            return "No results found.\n(via SearXNG)"
        return "\n".join(lines) + "\n(via SearXNG)"
    except Exception:
        return None

# --------------------------------------------------------------------------
# Image generation (Stable Diffusion via diffusers — optional)
# --------------------------------------------------------------------------

IMAGE_PRESETS = {
    "sd-turbo": {
        "repo": "stabilityai/sd-turbo", "steps": 2, "guidance": 0.0,
        "size": 512, "dl_gb": 2.5,
        "label": "SD Turbo — fast drafts (512px, ~2.5 GB)"},
    "dreamshaper-8": {
        "repo": "Lykon/dreamshaper-8", "steps": 30, "guidance": 6.5,
        "size": 512, "dl_gb": 2.1,
        "label": "DreamShaper 8 — quality SD1.5 (512px, ~2 GB)"},
    "sdxl-turbo": {
        "repo": "stabilityai/sdxl-turbo", "steps": 4, "guidance": 0.0,
        "size": 768, "dl_gb": 7.0,
        "label": "SDXL Turbo — fast, sharper (768px, ~7 GB, needs 10+ GB RAM/VRAM)"},
}

_pipelines: dict = {}
_image_lock = threading.Lock()
_image_cancel = threading.Event()


class _ImageCancelled(Exception):
    pass


def imagegen_available() -> dict:
    try:
        import torch  # noqa: F401
        import diffusers  # noqa: F401
        return {"installed": True, "device": _torch_device()}
    except Exception:
        return {"installed": False, "device": None}


def _torch_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def imagegen_commands(hw: dict) -> list:
    base = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check"]
    rest = ["diffusers>=0.31", "transformers", "accelerate", "safetensors",
            "pillow"]
    if hw["os"] == "Windows" and any(g["kind"] == "nvidia" for g in hw["gpus"]):
        # Windows CUDA wheels live on the PyTorch index; install torch on its
        # own first, then everything else from PyPI.
        return [base + ["torch", "--index-url",
                        "https://download.pytorch.org/whl/cu124"],
                base + rest]
    return [base + ["torch"] + rest]


_force_fp32_runtime = {"on": False}


def _image_dtype(device: str):
    """fp16 on MPS produces NaNs in the Stable Diffusion VAE → all-black
    images, so Apple Silicon always runs full precision. CUDA runs fp16 for
    speed unless a black frame was ever detected (then we remember fp32)."""
    import torch
    if device in ("cpu", "mps"):
        return torch.float32
    if _force_fp32_runtime["on"] or get_setting("image_force_fp32") == "1":
        return torch.float32
    return torch.float16


def _looks_black(img) -> bool:
    try:
        arr = np.asarray(img.convert("L"), dtype=np.float32)
        return float(arr.mean()) < 2.0 and float(arr.std()) < 2.0
    except Exception:
        return False


def _load_pipeline(preset_key: str, progress):
    import torch
    from diffusers import AutoPipelineForText2Image
    preset = IMAGE_PRESETS[preset_key]
    device = _torch_device()
    dtype = _image_dtype(device)
    cache_key = f"{preset_key}|{dtype}"
    if cache_key in _pipelines:
        return _pipelines[cache_key]
    progress(f"Loading {preset['repo']} (first time downloads ~{preset['dl_gb']} GB "
             f"— progress shows in the terminal)")
    pipe = AutoPipelineForText2Image.from_pretrained(
        preset["repo"], torch_dtype=dtype, safety_checker=None,
        requires_safety_checker=False)
    pipe = pipe.to(device)
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass
    _pipelines.clear()          # keep at most one pipeline in memory
    _pipelines[cache_key] = pipe
    return pipe


def generate_image_sync(prompt: str, preset_key: str, width: int, height: int,
                        steps: Optional[int], seed: Optional[int], progress) -> dict:
    import torch
    with _image_lock:
        _image_cancel.clear()
        preset = IMAGE_PRESETS.get(preset_key) or IMAGE_PRESETS["sd-turbo"]
        device = _torch_device()
        actual_seed = seed if seed not in (None, -1) else int.from_bytes(os.urandom(4), "big")
        nsteps = steps or preset["steps"]
        total = max(nsteps, 1)

        def cb(pipeline, step, timestep, kwargs):
            if _image_cancel.is_set():
                raise _ImageCancelled()
            progress(None, step + 1, total)
            return kwargs

        def render():
            pipe = _load_pipeline(preset_key, progress)
            gen = torch.Generator(device="cpu").manual_seed(actual_seed)
            progress(f"Generating on {device.upper()} — {nsteps} steps")
            result = pipe(prompt=prompt, num_inference_steps=nsteps,
                          guidance_scale=preset["guidance"],
                          width=width, height=height, generator=gen,
                          callback_on_step_end=cb)
            return result.images[0]

        img = render()
        if _looks_black(img) and _image_dtype(device) != torch.float32:
            # Known half-precision failure (NaNs in the VAE). Retry once in
            # full precision and, if that fixes it, remember forever.
            progress("The result was solid black — a known half-precision "
                     "bug. Retrying in full precision…")
            _force_fp32_runtime["on"] = True
            _pipelines.clear()
            img = render()
            if not _looks_black(img):
                set_setting("image_force_fp32", "1")

        img_id = new_id()
        filename = f"{img_id}.png"
        img.save(DATA_DIR / "images" / filename)
        qx("INSERT INTO images(id,filename,prompt,model,width,height,seed,created) "
           "VALUES(?,?,?,?,?,?,?,?)",
           (img_id, filename, prompt, preset_key, width, height, actual_seed, now()))
        return {"id": img_id, "filename": filename, "prompt": prompt,
                "model": preset_key, "width": width, "height": height,
                "seed": actual_seed, "created": now()}

# --------------------------------------------------------------------------
# Computer control (optional) — the agent sees the screen and drives the
# mouse/keyboard via PyAutoGUI. OFF by default; gated, logged, interruptible.
# --------------------------------------------------------------------------

_computer_stop = threading.Event()          # emergency stop for the running task
_computer_log: list = []                    # recent actions, newest last
_pending_confirms: dict = {}                 # id -> {"event", "approved", "action"}


def computer_available() -> dict:
    try:
        import pyautogui  # noqa: F401
        return {"installed": True}
    except Exception:
        return {"installed": False}


def computer_packages(hw: dict) -> list:
    base = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check"]
    pkgs = ["pyautogui", "pillow"]
    if hw["os"] == "Linux":
        pkgs.append("python-xlib")
    elif hw["os"] == "Darwin":
        pkgs += ["pyobjc-core", "pyobjc-framework-quartz"]
    return [base + pkgs]


def computer_os_notes(hw: dict) -> list:
    if hw["os"] == "Darwin":
        return ["macOS requires permission: System Settings → Privacy & "
                "Security → Accessibility, and enable the app running Heorth "
                "(your Terminal, or Python). Screen Recording permission is "
                "also needed for screenshots.",
                "You must grant this before the agent can control anything."]
    if hw["os"] == "Linux":
        return ["Linux needs an X11 session. On Wayland, mouse/keyboard "
                "control may not work — log in using an 'Xorg' session.",
                "If screenshots are blank, install scrot:  sudo apt install scrot"]
    return ["Windows works out of the box once installed.",
            "Run Heorth as a normal user; it will control your own session."]


def _pag():
    import pyautogui
    pyautogui.FAILSAFE = True     # slam mouse to a corner to abort
    try:
        pyautogui.PAUSE = max(0.0, float(get_setting("computer_pause") or 0.4))
    except Exception:
        pyautogui.PAUSE = 0.4
    return pyautogui


def computer_reset() -> None:
    _computer_stop.clear()
    for c in list(_pending_confirms.values()):
        c["approved"] = False
        c["event"].set()
    _pending_confirms.clear()


def _log_action(kind: str, detail: str, shot: Optional[str] = None) -> dict:
    entry = {"id": new_id(), "kind": kind, "detail": detail,
             "shot": shot, "ts": now()}
    _computer_log.append(entry)
    if len(_computer_log) > 200:
        del _computer_log[:80]
    return entry


def _screen_size() -> tuple:
    try:
        w, h = _pag().size()
        return int(w), int(h)
    except Exception:
        return (0, 0)


def take_screenshot(save: bool = True, tag: str = "screen") -> Optional[dict]:
    """Grab the primary screen, downscale for the model, optionally persist a
    copy to the images folder so the user sees exactly what the agent saw."""
    try:
        pag = _pag()
        img = pag.screenshot()
    except Exception as e:
        return {"error": f"Screenshot failed: {e}"}
    from PIL import Image
    w, h = img.size
    scale = min(1.0, 1280 / max(w, 1))
    small = img.resize((max(1, int(w * scale)), max(1, int(h * scale))),
                       Image.LANCZOS) if scale < 1 else img
    buf = io.BytesIO()
    small.convert("RGB").save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    stored = None
    if save:
        fn = f"cc_{int(now())}_{new_id()[:6]}.png"
        try:
            small.convert("RGB").save(DATA_DIR / "images" / fn)
            stored = fn
        except Exception:
            stored = None
    return {"b64": b64, "w": w, "h": h, "file": stored}


COMPUTER_TOOLS = [
    {"name": "screen_capture", "desc": "Take a screenshot and look at the screen. Call this first and whenever you need to see the current state.", "params": {}},
    {"name": "mouse_move", "desc": "Move the mouse to absolute pixel coordinates.", "params": {"x": "int", "y": "int"}},
    {"name": "mouse_click", "desc": "Click at coordinates (or current position). button is left/right/middle; clicks is 1 or 2.", "params": {"x": "int", "y": "int", "button": "string", "clicks": "int"}},
    {"name": "mouse_drag", "desc": "Drag from the current position to x,y (press, move, release).", "params": {"x": "int", "y": "int"}},
    {"name": "scroll", "desc": "Scroll vertically. Positive is up, negative is down; amount is 'clicks'.", "params": {"amount": "int"}},
    {"name": "type_text", "desc": "Type a string of text at the current cursor.", "params": {"text": "string"}},
    {"name": "press_keys", "desc": "Press a key or a hotkey combo. Give keys as space-separated names, e.g. 'ctrl c' or 'enter' or 'cmd space'.", "params": {"keys": "string"}},
]


def computer_tool_schemas() -> list:
    schemas = []
    for t in COMPUTER_TOOLS:
        props = {}
        for k, ty in t["params"].items():
            props[k] = {"type": "integer" if ty == "int" else "string",
                        "description": k}
        schemas.append({"type": "function", "function": {
            "name": t["name"], "description": "[computer] " + t["desc"],
            "parameters": {"type": "object", "properties": props,
                           "required": [k for k in t["params"]
                                        if k not in ("x", "y", "button",
                                                     "clicks")]}}})
    return schemas


async def _await_confirm(action_desc: str, emit) -> bool:
    """Block the tool until the user approves/denies in the UI (or stops)."""
    cid = new_id()
    ev = threading.Event()
    rec = {"event": ev, "approved": False, "action": action_desc}
    _pending_confirms[cid] = rec
    emit({"type": "computer_confirm", "id": cid, "action": action_desc})
    loop = asyncio.get_event_loop()
    # Wait for the user's decision, but not forever — an abandoned prompt
    # auto-denies after 5 minutes so the worker thread is never stuck.
    await loop.run_in_executor(None, lambda: ev.wait(timeout=300))
    _pending_confirms.pop(cid, None)
    if not ev.is_set():
        return False
    return rec["approved"] and not _computer_stop.is_set()


def resolve_confirm(cid: str, approved: bool) -> bool:
    rec = _pending_confirms.get(cid)
    if not rec:
        return False
    rec["approved"] = approved
    rec["event"].set()
    return True


async def call_computer_tool(name: str, args: dict, emit) -> str:
    """Execute one desktop action. `emit` streams UI events (log, confirm).
    Returns a text result; screen_capture additionally returns the image via
    a side channel handled by the agent loop."""
    if not flag("computer_control"):
        return ("Computer control is turned OFF. The user must enable it on "
                "the Computer page and accept the safety notice.")
    if not computer_available()["installed"]:
        return "Computer control isn't installed yet."
    if _computer_stop.is_set():
        return "Emergency stop is active — no further actions will run."

    # Read-only screenshots never need confirmation; actions may.
    needs_confirm = (name != "screen_capture" and
                     str(get_setting("computer_confirm")) == "1")
    human = _describe_action(name, args)
    if needs_confirm:
        ok = await _await_confirm(human, emit)
        if not ok:
            _log_action(name, "DENIED: " + human)
            emit({"type": "computer_action", "kind": name,
                  "detail": "✗ denied by user"})
            return f"The user denied this action: {human}"

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(None, _do_action, name, args)
    except Exception as e:
        try:
            import pyautogui
            if isinstance(e, pyautogui.FailSafeException):
                _computer_stop.set()
                _log_action(name, "FAIL-SAFE triggered (mouse to corner)")
                emit({"type": "computer_stopped",
                      "reason": "Fail-safe: mouse moved to a screen corner."})
                return ("Fail-safe triggered (mouse hit a screen corner). "
                        "All control stopped.")
        except Exception:
            pass
        return f"Action failed: {e}"

    shot = result.get("shot") if isinstance(result, dict) else None
    emit({"type": "computer_action", "kind": name,
          "detail": result.get("detail", human) if isinstance(result, dict)
          else human, "shot": shot})
    _log_action(name, result.get("detail", human) if isinstance(result, dict)
                else human, shot)
    if isinstance(result, dict):
        return result.get("text", "done")
    return "done"


def _describe_action(name: str, args: dict) -> str:
    if name == "screen_capture":
        return "look at the screen"
    if name == "mouse_move":
        return f"move mouse to ({args.get('x')}, {args.get('y')})"
    if name == "mouse_click":
        b = args.get("button", "left"); c = int(args.get("clicks", 1) or 1)
        at = (f" at ({args.get('x')}, {args.get('y')})"
              if args.get("x") is not None else "")
        return f"{'double-' if c == 2 else ''}{b}-click{at}"
    if name == "mouse_drag":
        return f"drag to ({args.get('x')}, {args.get('y')})"
    if name == "scroll":
        return f"scroll {args.get('amount')}"
    if name == "type_text":
        t = str(args.get("text", "")); short = t[:60] + ("…" if len(t) > 60 else "")
        return f"type: “{short}”"
    if name == "press_keys":
        return f"press keys: {args.get('keys')}"
    return name


def _do_action(name: str, args: dict) -> dict:
    pag = _pag()
    if name == "screen_capture":
        shot = take_screenshot(save=True)
        if shot and "error" in shot:
            return {"text": shot["error"], "detail": "screenshot failed"}
        w, h = (shot or {}).get("w", 0), (shot or {}).get("h", 0)
        return {"text": f"Screenshot captured. Screen is {w}x{h} pixels. "
                        "The image is attached for you to inspect.",
                "detail": f"captured screen ({w}×{h})",
                "shot": (shot or {}).get("file"),
                "_b64": (shot or {}).get("b64")}
    if name == "mouse_move":
        pag.moveTo(int(args["x"]), int(args["y"]))
        return {"text": "moved", "detail": _describe_action(name, args)}
    if name == "mouse_click":
        kw = {}
        if args.get("x") is not None:
            kw["x"] = int(args["x"]); kw["y"] = int(args["y"])
        kw["button"] = args.get("button", "left")
        kw["clicks"] = int(args.get("clicks", 1) or 1)
        pag.click(**kw)
        return {"text": "clicked", "detail": _describe_action(name, args)}
    if name == "mouse_drag":
        pag.dragTo(int(args["x"]), int(args["y"]), duration=0.4)
        return {"text": "dragged", "detail": _describe_action(name, args)}
    if name == "scroll":
        pag.scroll(int(args.get("amount", 0) or 0))
        return {"text": "scrolled", "detail": _describe_action(name, args)}
    if name == "type_text":
        pag.write(str(args.get("text", "")), interval=0.02)
        return {"text": "typed", "detail": _describe_action(name, args)}
    if name == "press_keys":
        keys = [k for k in str(args.get("keys", "")).replace("+", " ").split()
                if k]
        if len(keys) == 1:
            pag.press(keys[0])
        elif keys:
            pag.hotkey(*keys)
        return {"text": "pressed", "detail": _describe_action(name, args)}
    return {"text": f"unknown action {name}", "detail": name}


# --------------------------------------------------------------------------
# Built-in agent tools
# --------------------------------------------------------------------------

WORKSPACE = DATA_DIR / "workspace"


def _safe_workspace_path(rel: str) -> Path:
    p = (WORKSPACE / rel).resolve()
    if WORKSPACE.resolve() not in p.parents and p != WORKSPACE.resolve():
        raise ValueError("Path escapes the workspace folder")
    return p


def _strip_html(html: str) -> str:
    html = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", html)
    html = re.sub(r"(?s)<[^>]+>", " ", html)
    return re.sub(r"\s{2,}", " ", html).strip()


async def tool_fetch_url(url: str) -> str:
    if not flag("allow_web_tools"):
        return "Web tools are disabled in Settings."
    async with httpx.AsyncClient(timeout=20, follow_redirects=True,
                                 headers={"User-Agent": f"{APP_NAME}/{__version__}"}) as c:
        r = await c.get(url)
        text = _strip_html(r.text)
        return text[:8000] or "(empty page)"


async def _ddg_search(query: str) -> str:
    try:
        async with httpx.AsyncClient(timeout=20, headers={
                "User-Agent": "Mozilla/5.0 (Heorth agent)"}) as c:
            r = await c.post("https://html.duckduckgo.com/html/",
                             data={"q": query})
            html = r.text
        results = []
        for m in re.finditer(
                r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
                html, re.S):
            url, title = m.group(1), _strip_html(m.group(2))
            if "duckduckgo.com" in url and "uddg=" in url:
                from urllib.parse import parse_qs, unquote, urlparse
                qs = parse_qs(urlparse(url).query)
                url = unquote(qs.get("uddg", [url])[0])
            results.append(f"- {title}\n  {url}")
            if len(results) >= 6:
                break
        out = "\n".join(results) or "No results found."
        return out + "\n(via DuckDuckGo)"
    except Exception as e:
        return f"Search failed: {e}"


async def perform_web_search(query: str) -> str:
    """Route to the configured backend. 'auto' prefers SearXNG when it is
    reachable and quietly falls back to the built-in DuckDuckGo scraper."""
    backend = get_setting("search_backend")
    if backend in ("auto", "searxng"):
        out = await _searxng_search(query)
        if out is not None:
            return out
        if backend == "searxng":
            return ("SearXNG is not reachable at "
                    f"{get_setting('searxng_url')}. Check the Agent & Tools "
                    "page, or switch the search backend in Settings.")
    return await _ddg_search(query)


async def tool_web_search(query: str) -> str:
    if not flag("allow_web_tools"):
        return "Web tools are disabled in Settings."
    return await perform_web_search(query)


def tool_calculator(expression: str) -> str:
    import ast
    import operator as op
    ops = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
           ast.Pow: op.pow, ast.Mod: op.mod, ast.FloorDiv: op.floordiv,
           ast.USub: op.neg, ast.UAdd: op.pos}

    def ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            return ops[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in ops:
            return ops[type(node.op)](ev(node.operand))
        raise ValueError("Unsupported expression")
    try:
        return str(ev(ast.parse(expression, mode="eval").body))
    except Exception as e:
        return f"Could not evaluate: {e}"


def tool_list_files() -> str:
    items = []
    for p in sorted(WORKSPACE.rglob("*")):
        if p.is_file():
            rel = p.relative_to(WORKSPACE)
            items.append(f"{rel} ({p.stat().st_size} bytes)")
    return "\n".join(items) or "(workspace is empty)"


def tool_read_file(path: str) -> str:
    p = _safe_workspace_path(path)
    if not p.is_file():
        return f"No such file: {path}"
    return p.read_text(encoding="utf-8", errors="replace")[:12000]


def tool_write_file(path: str, content: str) -> str:
    p = _safe_workspace_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} characters to {path}"


def tool_run_python(code: str) -> str:
    if not flag("allow_code_execution"):
        return ("Code execution is disabled. Enable 'Allow code execution' "
                "in Settings to use this tool.")
    try:
        r = subprocess.run([sys.executable, "-I", "-c", code],
                           capture_output=True, text=True, timeout=60,
                           cwd=str(WORKSPACE))
        out = (r.stdout or "") + (("\n[stderr]\n" + r.stderr) if r.stderr else "")
        return out[:6000] or "(no output)"
    except subprocess.TimeoutExpired:
        return "Timed out after 60 seconds."
    except Exception as e:
        return f"Failed: {e}"


def tool_run_shell(command: str) -> str:
    if not flag("allow_code_execution"):
        return ("Code execution is disabled. Enable 'Allow code execution' "
                "in Settings to use this tool.")
    try:
        r = subprocess.run(command, shell=True, capture_output=True, text=True,
                           timeout=60, cwd=str(WORKSPACE))
        out = (r.stdout or "") + (("\n[stderr]\n" + r.stderr) if r.stderr else "")
        return out[:6000] or "(no output)"
    except subprocess.TimeoutExpired:
        return "Timed out after 60 seconds."
    except Exception as e:
        return f"Failed: {e}"


BUILTIN_TOOLS = [
    {"name": "web_search", "desc": "Search the web — uses your private SearXNG when set up, DuckDuckGo otherwise.",
     "params": {"query": "string"}},
    {"name": "fetch_url", "desc": "Fetch a web page and return its readable text.",
     "params": {"url": "string"}},
    {"name": "calculator", "desc": "Evaluate an arithmetic expression, e.g. (17*32)/4.",
     "params": {"expression": "string"}},
    {"name": "search_knowledge", "desc": "Search the user's local knowledge base (uploaded documents).",
     "params": {"query": "string"}},
    {"name": "list_files", "desc": "List files in the agent workspace folder.",
     "params": {}},
    {"name": "read_file", "desc": "Read a text file from the agent workspace.",
     "params": {"path": "string"}},
    {"name": "write_file", "desc": "Write/overwrite a text file in the agent workspace.",
     "params": {"path": "string", "content": "string"}},
    {"name": "generate_image", "desc": "Generate an image from a text prompt and return it inline.",
     "params": {"prompt": "string"}},
    {"name": "run_python", "desc": "Run a short Python script (only if enabled in Settings).",
     "params": {"code": "string"}},
    {"name": "run_shell", "desc": "Run a shell command in the workspace (only if enabled in Settings).",
     "params": {"command": "string"}},
]


def builtin_tool_schemas() -> list:
    schemas = []
    for t in BUILTIN_TOOLS:
        props = {k: {"type": "string", "description": k} for k in t["params"]}
        schemas.append({"type": "function", "function": {
            "name": t["name"], "description": t["desc"],
            "parameters": {"type": "object", "properties": props,
                           "required": list(t["params"].keys())}}})
    return schemas


async def call_builtin_tool(name: str, args: dict) -> str:
    try:
        if name == "web_search":
            return await tool_web_search(str(args.get("query", "")))
        if name == "fetch_url":
            return await tool_fetch_url(str(args.get("url", "")))
        if name == "calculator":
            return tool_calculator(str(args.get("expression", "")))
        if name == "search_knowledge":
            hits = await rag_search(str(args.get("query", "")))
            if not hits:
                return "The knowledge base is empty or nothing matched."
            return "\n\n".join(f"[{h['doc']}] {h['text'][:800]}" for h in hits)
        if name == "list_files":
            return tool_list_files()
        if name == "read_file":
            return tool_read_file(str(args.get("path", "")))
        if name == "write_file":
            return tool_write_file(str(args.get("path", "")),
                                   str(args.get("content", "")))
        if name == "generate_image":
            if not imagegen_available()["installed"]:
                return ("Image generation is not installed. The user can enable "
                        "it on the Images page.")
            loop = asyncio.get_event_loop()
            info = await loop.run_in_executor(
                None, lambda: generate_image_sync(
                    str(args.get("prompt", "")), get_setting("image_model"),
                    512, 512, None, None, lambda *a, **k: None))
            return (f"Image generated and saved to the gallery. Show it to the "
                    f"user with this exact markdown: "
                    f"![generated image](/api/images/file/{info['filename']})")
        if name == "run_python":
            return tool_run_python(str(args.get("code", "")))
        if name == "run_shell":
            return tool_run_shell(str(args.get("command", "")))
        return f"Unknown tool: {name}"
    except Exception as e:
        return f"Tool error: {e}"

# --------------------------------------------------------------------------
# Coder mode — an opencode-style coding agent over a real project folder.
# Every tool is locked inside the folder chosen in Settings; edits are off
# by default ("plan mode") and shell commands reuse the global code gate.
# --------------------------------------------------------------------------

CODER_SKIP_DIRS = {".git", ".hg", ".svn", "node_modules", "__pycache__",
                   ".venv", "venv", "env", ".tox", ".mypy_cache",
                   ".pytest_cache", ".ruff_cache", "dist", "build",
                   ".next", ".nuxt", "target", ".idea", ".vscode",
                   "heorth_data", "localmind_data"}


def _coder_root() -> Optional[Path]:
    raw = (get_setting("coder_root") or "").strip()
    if not raw:
        return None
    try:
        p = Path(raw).expanduser().resolve()
    except Exception:
        return None
    return p if p.is_dir() else None


def _coder_path(rel: str) -> Path:
    """Resolve a path inside the project root; refuse anything that escapes
    it (.., absolute paths, symlinks pointing outside)."""
    root = _coder_root()
    if root is None:
        raise ValueError("no project folder is set (Settings → Coder)")
    rel = (rel or "").strip().lstrip("/\\")
    p = (root / rel).resolve() if rel else root
    if p != root and root not in p.parents:
        raise ValueError(f"path escapes the project folder: {rel}")
    return p


def _coder_is_binary(p: Path) -> bool:
    try:
        with open(p, "rb") as fh:
            return b"\x00" in fh.read(1024)
    except OSError:
        return True


def tool_coder_tree() -> str:
    root = _coder_root()
    if root is None:
        return ("No project folder is set. Ask the user to set one in "
                "Settings → Coder.")
    lines: list = []
    count = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames
                             if d not in CODER_SKIP_DIRS
                             and not d.startswith("."))
        rel = os.path.relpath(dirpath, root)
        depth = 0 if rel == "." else rel.count(os.sep) + 1
        if depth > 5:
            dirnames[:] = []
            continue
        indent = "  " * depth
        if rel != ".":
            lines.append(f"{indent}{os.path.basename(dirpath)}/")
            count += 1
        for f in sorted(filenames):
            if count >= 350:
                lines.append("… (tree truncated — use coder_grep to "
                             "locate specific files)")
                return "\n".join(lines)
            try:
                size = os.path.getsize(os.path.join(dirpath, f))
            except OSError:
                size = 0
            lines.append(f"{indent}  {f}  ({size:,} B)")
            count += 1
    return "\n".join(lines) or "(the project folder is empty)"


def tool_coder_read(path: str, start: str = "", end: str = "") -> str:
    try:
        p = _coder_path(path)
    except ValueError as e:
        return f"Refused: {e}"
    if not p.is_file():
        return f"Not a file: {path}"
    try:
        if p.stat().st_size > 5_000_000:
            return (f"{path} is {p.stat().st_size:,} bytes — too large. "
                    "Use coder_grep to find the relevant part.")
        if _coder_is_binary(p):
            return f"{path} looks like a binary file."
        all_lines = p.read_text(errors="replace").splitlines()
    except OSError as e:
        return f"Failed to read {path}: {e}"
    total = len(all_lines)
    if total == 0:
        return f"{path} is empty (0 lines)."
    s = int(start) if str(start).strip().isdigit() and int(start) > 0 else 1
    e = int(end) if str(end).strip().isdigit() and int(end) > 0 else total
    s, e = max(1, min(s, total or 1)), max(1, min(e, total))
    chunk = all_lines[s - 1:e]
    clipped = False
    if len(chunk) > 400:
        chunk, e, clipped = chunk[:400], s + 399, True
    body = "\n".join(f"{i:>5}  {ln[:500]}"
                     for i, ln in enumerate(chunk, s))[:60_000]
    head = f"{path}  ({total} lines; showing {s}–{e})"
    tail = ("\n… (continue with a higher start line)" if clipped or e < total
            else "")
    return f"{head}\n{body}{tail}"


def tool_coder_grep(pattern: str, file_glob: str = "") -> str:
    root = _coder_root()
    if root is None:
        return ("No project folder is set. Ask the user to set one in "
                "Settings → Coder.")
    if not pattern:
        return "Empty search pattern."
    try:
        rx = re.compile(pattern)
    except re.error:
        rx = re.compile(re.escape(pattern))   # fall back to literal search
    hits: list = []
    scanned = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames
                       if d not in CODER_SKIP_DIRS and not d.startswith(".")]
        for f in sorted(filenames):
            if file_glob and not fnmatch.fnmatch(f, file_glob):
                continue
            fp = os.path.join(dirpath, f)
            try:
                if os.path.getsize(fp) > 1_000_000:
                    continue
                with open(fp, "rb") as fh:
                    if b"\x00" in fh.read(1024):
                        continue
                scanned += 1
                relf = os.path.relpath(fp, root)
                with open(fp, errors="replace") as fh:
                    for i, line in enumerate(fh, 1):
                        if rx.search(line):
                            hits.append(f"{relf}:{i}: "
                                        f"{line.rstrip()[:200]}")
                            if len(hits) >= 60:
                                hits.append("… (more matches truncated — "
                                            "narrow the pattern)")
                                return "\n".join(hits)
            except OSError:
                continue
    if not hits:
        return f"No matches for {pattern!r} in {scanned} text files."
    return "\n".join(hits)


def _coder_write_gate() -> Optional[str]:
    if not flag("coder_allow_write"):
        return ("File edits are off (plan mode). Present the change as a "
                "diff in your final answer instead, or ask the user to "
                "enable 'Allow file edits' in Settings → Coder.")
    return None


def tool_coder_edit(path: str, find: str, replace: str) -> str:
    gate = _coder_write_gate()
    if gate:
        return gate
    try:
        p = _coder_path(path)
    except ValueError as e:
        return f"Refused: {e}"
    if not p.is_file():
        return f"Not a file: {path} (use coder_write to create files)"
    if not find:
        return "Empty 'find' string."
    try:
        old = p.read_text(errors="replace")
    except OSError as e:
        return f"Failed to read {path}: {e}"
    n = old.count(find)
    if n == 0:
        return ("The 'find' text was not found — it must match the file "
                "exactly, including whitespace. Read the file again first.")
    if n > 1:
        return (f"The 'find' text occurs {n} times. Include surrounding "
                "lines so it is unique, then retry.")
    new = old.replace(find, replace, 1)
    try:
        p.write_text(new)
    except OSError as e:
        return f"Failed to write {path}: {e}"
    diff = "".join(difflib.unified_diff(
        old.splitlines(keepends=True), new.splitlines(keepends=True),
        fromfile=f"a/{path}", tofile=f"b/{path}", n=2))
    return f"Edited {path}.\n{diff[:3000]}"


def tool_coder_write(path: str, content: str) -> str:
    gate = _coder_write_gate()
    if gate:
        return gate
    try:
        p = _coder_path(path)
    except ValueError as e:
        return f"Refused: {e}"
    if p.is_dir():
        return f"{path} is a directory."
    if len(content) > 400_000:
        return "Content too large (400 KB limit) — write it in parts."
    old = ""
    if p.is_file():
        try:
            old = p.read_text(errors="replace")
        except OSError:
            old = ""
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    except OSError as e:
        return f"Failed to write {path}: {e}"
    if old:
        diff = "".join(difflib.unified_diff(
            old.splitlines(keepends=True), content.splitlines(keepends=True),
            fromfile=f"a/{path}", tofile=f"b/{path}", n=2))
        return f"Rewrote {path}.\n{diff[:3000]}"
    return f"Created {path} ({len(content):,} chars)."


def tool_coder_shell(command: str) -> str:
    if not flag("allow_code_execution"):
        return ("Code execution is disabled. The user can enable 'Allow "
                "code execution' in Settings to let you run commands and "
                "tests in the project.")
    root = _coder_root()
    if root is None:
        return "No project folder is set (Settings → Coder)."
    try:
        r = subprocess.run(command, shell=True, capture_output=True,
                           text=True, timeout=120, cwd=str(root))
        out = (r.stdout or "") + (("\n[stderr]\n" + r.stderr)
                                  if r.stderr else "")
        return (out[:6000] or "(no output)") + f"\n[exit {r.returncode}]"
    except subprocess.TimeoutExpired:
        return "Timed out after 120 seconds."
    except Exception as e:
        return f"Failed: {e}"


CODER_TOOLS = [
    {"name": "coder_tree",
     "desc": "Show the project's file tree with sizes (heavy folders like "
             "node_modules and .git are skipped).",
     "params": {}, "required": []},
    {"name": "coder_read",
     "desc": "Read a project file with line numbers. Optional start/end "
             "line numbers for large files.",
     "params": {"path": "string", "start": "string", "end": "string"},
     "required": ["path"]},
    {"name": "coder_grep",
     "desc": "Search every project file for a regex or plain text. Optional "
             "file_glob such as *.py to narrow it.",
     "params": {"pattern": "string", "file_glob": "string"},
     "required": ["pattern"]},
    {"name": "coder_edit",
     "desc": "Replace one exact, unique text snippet in a file and get a "
             "diff back. Read the file first; the match must be exact, "
             "including whitespace.",
     "params": {"path": "string", "find": "string", "replace": "string"},
     "required": ["path", "find", "replace"]},
    {"name": "coder_write",
     "desc": "Create a new file or overwrite one completely. Prefer "
             "coder_edit for small changes.",
     "params": {"path": "string", "content": "string"},
     "required": ["path", "content"]},
    {"name": "coder_shell",
     "desc": "Run a shell command inside the project folder (tests, git, "
             "build). Requires 'Allow code execution' in Settings.",
     "params": {"command": "string"}, "required": ["command"]},
]


def coder_tool_schemas() -> list:
    schemas = []
    for t in CODER_TOOLS:
        props = {k: {"type": "string", "description": k}
                 for k in t["params"]}
        schemas.append({"type": "function", "function": {
            "name": t["name"], "description": t["desc"],
            "parameters": {"type": "object", "properties": props,
                           "required": t["required"]}}})
    return schemas


def coder_toolset() -> list:
    """Coder tools plus the web tools (handy for looking up docs) when the
    user has web tools enabled."""
    tools = coder_tool_schemas()
    if flag("allow_web_tools"):
        keep = {"web_search", "fetch_url"}
        tools += [t for t in builtin_tool_schemas()
                  if t["function"]["name"] in keep]
    return tools


async def call_coder_tool(name: str, args: dict) -> str:
    try:
        if name == "coder_tree":
            return tool_coder_tree()
        if name == "coder_read":
            return tool_coder_read(str(args.get("path", "")),
                                   str(args.get("start", "")),
                                   str(args.get("end", "")))
        if name == "coder_grep":
            return tool_coder_grep(str(args.get("pattern", "")),
                                   str(args.get("file_glob", "")))
        if name == "coder_edit":
            return tool_coder_edit(str(args.get("path", "")),
                                   str(args.get("find", "")),
                                   str(args.get("replace", "")))
        if name == "coder_write":
            return tool_coder_write(str(args.get("path", "")),
                                    str(args.get("content", "")))
        if name == "coder_shell":
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, lambda: tool_coder_shell(str(args.get("command", ""))))
        return f"Unknown tool: {name}"
    except Exception as e:
        return f"Tool error: {e}"


def coder_protocol(root: Path) -> str:
    if flag("coder_allow_write"):
        mode = ("Build mode: you may change files with coder_edit and "
                "coder_write.")
    else:
        mode = ("Plan mode (read-only): file edits are disabled. "
                "Investigate the code, then present your plan and the exact "
                "proposed diffs in the final answer.")
    return (
        f"\nYou are Heorth's coding agent working on the user's project at "
        f"{root}. All coder_* tools operate inside that folder only.\n"
        f"{mode}\n"
        "Method: orient yourself with coder_tree and coder_grep; always "
        "coder_read code before changing it; make small, targeted edits; "
        "verify with coder_shell (tests/build) when code execution is "
        "enabled. Never invent file contents or tool output. When the task "
        "is fully done, call task_complete with a clear summary of what "
        "changed and why.")

# --------------------------------------------------------------------------
# MCP client manager (optional 'mcp' package)
# --------------------------------------------------------------------------


def mcp_available() -> bool:
    try:
        import mcp  # noqa: F401
        return True
    except Exception:
        return False


class MCPManager:
    """Keeps stdio connections to configured MCP servers and exposes
    their tools to the agent."""

    def __init__(self):
        self.sessions: dict = {}    # server_id -> {"session", "stack", "tools"}
        self.errors: dict = {}      # server_id -> last error string

    def list_servers(self) -> list:
        rows = q("SELECT * FROM mcp_servers ORDER BY name")
        out = []
        for r in rows:
            live = self.sessions.get(r["id"])
            out.append({**r, "args": json.loads(r["args"] or "[]"),
                        "env": json.loads(r["env"] or "{}"),
                        "connected": bool(live),
                        "tools": [t["name"] for t in (live or {}).get("tools", [])],
                        "error": self.errors.get(r["id"])})
        return out

    async def connect(self, server_id: str) -> dict:
        if not mcp_available():
            return {"ok": False, "error": "The 'mcp' package is not installed."}
        rows = q("SELECT * FROM mcp_servers WHERE id=?", (server_id,))
        if not rows:
            return {"ok": False, "error": "Unknown server."}
        cfg = rows[0]
        await self.disconnect(server_id)
        try:
            from contextlib import AsyncExitStack
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
            stack = AsyncExitStack()
            params = StdioServerParameters(
                command=cfg["command"],
                args=json.loads(cfg["args"] or "[]"),
                env={**os.environ, **json.loads(cfg["env"] or "{}")})
            read, write = await stack.enter_async_context(stdio_client(params))
            session = await stack.enter_async_context(ClientSession(read, write))
            await asyncio.wait_for(session.initialize(), timeout=30)
            listed = await asyncio.wait_for(session.list_tools(), timeout=30)
            tools = []
            for t in listed.tools:
                tools.append({"name": t.name,
                              "description": t.description or t.name,
                              "schema": t.inputSchema or
                              {"type": "object", "properties": {}}})
            self.sessions[server_id] = {"session": session, "stack": stack,
                                        "tools": tools}
            self.errors.pop(server_id, None)
            return {"ok": True, "tools": [t["name"] for t in tools]}
        except Exception as e:
            self.errors[server_id] = str(e)[:400]
            return {"ok": False, "error": str(e)[:400]}

    async def disconnect(self, server_id: str) -> None:
        live = self.sessions.pop(server_id, None)
        if live:
            try:
                await live["stack"].aclose()
            except Exception:
                pass

    async def connect_enabled(self) -> None:
        for row in q("SELECT id FROM mcp_servers WHERE enabled=1"):
            try:
                await self.connect(row["id"])
            except Exception:
                pass

    def agent_tools(self) -> list:
        """Ollama tool schemas for every connected MCP tool, namespaced."""
        schemas = []
        for sid, live in self.sessions.items():
            srv = q("SELECT name FROM mcp_servers WHERE id=?", (sid,))
            sname = srv[0]["name"] if srv else sid
            for t in live["tools"]:
                fname = f"mcp__{sid}__{t['name']}"
                schemas.append({"type": "function", "function": {
                    "name": fname,
                    "description": f"[MCP: {sname}] {t['description']}"[:900],
                    "parameters": t["schema"]}})
        return schemas

    async def call(self, namespaced: str, args: dict) -> str:
        try:
            _, sid, tool = namespaced.split("__", 2)
        except ValueError:
            return f"Bad MCP tool name: {namespaced}"
        live = self.sessions.get(sid)
        if not live:
            return "That MCP server is not connected."
        try:
            result = await asyncio.wait_for(
                live["session"].call_tool(tool, args or {}), timeout=120)
            parts = []
            for c in getattr(result, "content", []) or []:
                text = getattr(c, "text", None)
                parts.append(text if text is not None else str(c))
            out = "\n".join(parts) or "(no content returned)"
            return out[:8000]
        except Exception as e:
            return f"MCP tool error: {e}"


MCP = MCPManager()

# --------------------------------------------------------------------------
# Chat + agent orchestration (streams NDJSON events)
# --------------------------------------------------------------------------


def _context_messages(conv_id: str) -> list:
    limit = int(get_setting("context_messages") or 30)
    rows = q("SELECT role, content, meta FROM messages WHERE conv_id=? "
             "ORDER BY created DESC LIMIT ?", (conv_id, limit))
    rows.reverse()
    msgs = [{"role": "system", "content": get_setting("system_prompt")}]
    # Only real user/assistant turns with actual text. Empty assistant rows
    # (e.g. an agent turn that was stopped before writing an answer, or a
    # tool-only turn) must not be replayed — an empty assistant message in the
    # prompt makes many chat templates misbehave and can stall generation.
    for r in rows:
        if r["role"] not in ("user", "assistant"):
            continue
        content = (r["content"] or "").strip()
        if not content:
            continue
        m = {"role": r["role"], "content": content}
        if r["role"] == "user":
            try:
                raw = json.loads(r["meta"] or "{}").get("images") or []
            except Exception:
                raw = []
            imgs = []
            for im in raw[:4]:
                s = str(im).strip()
                if s.lower().startswith("data:") and "," in s:
                    s = s.split(",", 1)[1].strip()   # ollama wants bare base64
                if s:
                    imgs.append(s)
            if imgs:
                m["images"] = imgs
        msgs.append(m)
    budget = 4          # replay only the newest images — VRAM and context
    for m in reversed(msgs):
        if "images" in m:
            keep = m["images"][:budget]
            budget -= len(keep)
            if keep:
                m["images"] = keep
            else:
                del m["images"]
    return msgs


async def _rag_context(query: str):
    hits = await rag_search(query)
    if not hits:
        return None, []
    block = "\n\n".join(f"[Source: {h['doc']}]\n{h['text']}" for h in hits)
    ctx = ("Relevant excerpts from the user's local knowledge base are below. "
           "Use them when they answer the question and mention the source "
           "file names you used.\n\n" + block)
    return ctx, [{"doc": h["doc"], "score": round(h["score"], 3),
                  "snippet": h["text"][:220]} for h in hits]


# --- Automatic web search: decide per-message whether a search is needed ----

_NO_SEARCH_RE = re.compile(
    r"^\s*(hi|hey|hello|yo|sup|thanks|thank you|thx|ok|okay|cool|nice|"
    r"good (morning|afternoon|evening|night)|bye|goodbye|lol|haha)\b", re.I)
_MATH_RE = re.compile(r"^[\s\d\.\+\-\*/%\^\(\)=xX×÷,]+\??$")
_CREATIVE_RE = re.compile(
    r"^\s*(write|compose|draft|create|generate|make me|give me)\b.*"
    r"(poem|story|song|haiku|essay|joke|rap|limerick|script|dialogue|"
    r"tagline|slogan|caption)", re.I)
_CODE_RE = re.compile(
    r"\b(write|debug|fix|refactor|explain this|review)\b.*\b(code|function|"
    r"script|regex|bug|class|method|query|sql)\b", re.I)


def _obviously_no_search(msg: str) -> bool:
    """Cheap pre-filter: skip the classifier for messages that clearly never
    need the web (greetings, arithmetic, creative writing, code help)."""
    m = msg.strip()
    if len(m) < 3:
        return True
    if _NO_SEARCH_RE.match(m) and len(m.split()) <= 6:
        return True
    if _MATH_RE.match(m):
        return True
    if _CREATIVE_RE.search(m):
        return True
    if _CODE_RE.search(m) or "```" in m:
        return True
    return False


async def classify_search(message: str, recent: str, model: str) -> Optional[str]:
    """Ask the model whether the message needs a current/factual web search.
    Returns a search query string, or None. Works with any model (no tool
    support required); output is capped to keep it fast."""
    ctx = (f"Recent context:\n{recent}\n\n" if recent else "")
    prompt = (
        "You decide whether answering the user's latest message needs a live "
        "web search for current, factual, or niche information the assistant "
        "might not reliably know (news, prices, weather, sports, releases, "
        "people/companies now, specific facts, anything time-sensitive).\n"
        "Reply on ONE line, nothing else:\n"
        "  SEARCH: <a concise search query>   (if a search would clearly help)\n"
        "  NONE                                (for greetings, opinions, "
        "math, coding, translation, general knowledge, or things you already "
        "know well)\n\n"
        f"{ctx}User's latest message:\n{message}\n\nDecision:")
    try:
        # Reasoning models spend their token budget "thinking" before any
        # visible answer, so a tight num_predict starves the decision line
        # and the classifier silently returns None. Ask with thinking off
        # first; if the model or Ollama version rejects the flag, retry
        # without it but with enough room to think AND answer.
        try:
            resp = await asyncio.wait_for(
                ollama_chat([{"role": "user", "content": prompt}], model,
                            think=False,
                            options={"num_predict": 60, "temperature": 0}),
                timeout=90)
        except Exception:
            resp = await asyncio.wait_for(
                ollama_chat([{"role": "user", "content": prompt}], model,
                            options={"num_predict": 600, "temperature": 0}),
                timeout=90)
        msg = resp.get("message") or {}
        text = (msg.get("content", "") or "").strip()
        if not text:
            # the decision may have landed in the separate reasoning field
            text = (msg.get("thinking") or msg.get("reasoning") or "").strip()
        # take the first line, strip any reasoning that leaked through
        for line in text.splitlines():
            line = line.strip()
            up = line.upper()
            if up.startswith("SEARCH:"):
                query = line.split(":", 1)[1].strip().strip('"').strip()
                return query[:200] or None
            if up.startswith("NONE") or up == "NO":
                return None
        # if the model rambled, look for a SEARCH: anywhere
        m = re.search(r"SEARCH:\s*(.+)", text, re.I)
        if m:
            return m.group(1).splitlines()[0].strip().strip('"')[:200] or None
        return None
    except Exception:
        return None


async def run_chat(conv_id: str, user_message: str, model: str,
                   use_rag: bool, agent_mode: bool,
                   loop_mode: bool = False,
                   council_mode: bool = False,
                   computer_mode: bool = False,
                   coder_mode: bool = False,
                   images: Optional[list] = None,
                   save_user: bool = True) -> AsyncGenerator[str, None]:
    """Streams NDJSON events and persists both sides of the exchange.
    Persistence happens in a finally block so that when the user hits Stop
    (client disconnect), the partial answer is still saved."""

    def ev(obj: dict) -> str:
        return json.dumps(obj, ensure_ascii=False) + "\n"

    if save_user:
        clean_imgs = [str(x) for x in (images or []) if str(x).strip()][:4]
        umeta = json.dumps({"images": clean_imgs} if clean_imgs else {})
        qx("INSERT INTO messages(id,conv_id,role,content,meta,created) "
           "VALUES(?,?,?,?,?,?)",
           (new_id(), conv_id, "user", user_message, umeta, now()))

    status = await ollama_up()
    if not status["up"]:
        yield ev({"type": "error", "error":
                  "Ollama is not running. Open the Models page for setup help."})
        return

    messages = _context_messages(conv_id)
    events_meta: list = []
    sources = []
    if use_rag:
        ctx, sources = await _rag_context(user_message)
        if ctx:
            messages.insert(1, {"role": "system", "content": ctx})
            yield ev({"type": "sources", "items": sources})
            events_meta.append({"type": "sources", "items": sources})

    full_answer = []
    try:
        if computer_mode:
            async for line in _run_computer_agent(messages, model, events_meta,
                                                  full_answer):
                yield line
        elif council_mode:
            async for line in _run_council(messages, model, events_meta,
                                           full_answer):
                yield line
        elif coder_mode:
            root = _coder_root()
            if root is None:
                yield ev({"type": "error", "error":
                          "Coder mode needs a project folder. Set an "
                          "absolute path in Settings \u2192 Coder, then try "
                          "again."})
                return
            async for line in _run_agent(
                    messages, model, events_meta, full_answer,
                    loop_mode=True,
                    tools_override=coder_toolset(),
                    protocol_override=coder_protocol(root),
                    budget_override=int(get_setting("coder_max_steps") or 25),
                    label_override="Coder"):
                yield line
        elif agent_mode or loop_mode:
            async for line in _run_agent(messages, model, events_meta,
                                         full_answer, loop_mode=loop_mode):
                yield line
        else:
            # Automatic web search (plain chat only): decide if the message
            # needs current info, and if so search and ground the answer.
            answer_messages = messages
            if (flag("auto_search")
                    and flag("allow_web_tools")
                    and not _obviously_no_search(user_message)):
                yield ev({"type": "status",
                          "text": "Checking whether a web search would help…"})
                recent = ""
                for m in reversed(messages[:-1]):
                    if m.get("role") == "assistant" and m.get("content"):
                        recent = m["content"][:400]
                        break
                query = await classify_search(user_message, recent, model)
                if query:
                    yield ev({"type": "status",
                              "text": f"Searching the web: {query}"})
                    results = await perform_web_search(query)
                    yield ev({"type": "auto_search", "query": query,
                              "result": results[:1800]})
                    events_meta.append({"type": "auto_search", "query": query,
                                        "result": results[:1800]})
                    web_ctx = ("You just ran a live web search because the "
                               "question needs current information. Use these "
                               "results to answer, and mention that the info "
                               f"comes from a web search.\n\nQuery: {query}\n\n"
                               f"Results:\n{results}")
                    answer_messages = (messages[:1]
                                       + [{"role": "system", "content": web_ctx}]
                                       + messages[1:])
            got_content = False
            stats = None
            async for part in ollama_stream_with_heartbeat(answer_messages, model):
                if "_heartbeat" in part:
                    secs = int(part["_heartbeat"])
                    yield ev({"type": "status", "text":
                              "Loading the model or reading the conversation… "
                              f"({secs}s)" if secs < 20 else
                              "Still working — long conversations take a moment "
                              f"to process on the first reply… ({secs}s)"})
                    continue
                msg = part.get("message") or {}
                think = msg.get("thinking") or msg.get("reasoning") or ""
                if think:
                    yield ev({"type": "thinking", "text": think})
                token = msg.get("content", "")
                if token:
                    got_content = True
                    full_answer.append(token)
                    yield ev({"type": "token", "text": token})
                if part.get("done"):
                    stats = _gen_stats(part) or stats
                    break
            # Some reasoning models can stream only their thinking and finish
            # with an empty answer. Retry once with thinking turned off so the
            # user always gets a real reply.
            if not got_content and not "".join(full_answer).strip():
                yield ev({"type": "status",
                          "text": "Finalizing the answer…"})
                try:
                    async for part in ollama_chat_stream(messages, model,
                                                         think=False):
                        token = (part.get("message") or {}).get("content", "")
                        if token:
                            got_content = True
                            full_answer.append(token)
                            yield ev({"type": "token", "text": token})
                        if part.get("done"):
                            stats = _gen_stats(part) or stats
                            break
                except Exception:
                    pass
                if not "".join(full_answer).strip():
                    yield ev({"type": "error", "error":
                              "The model finished without producing an answer. "
                              "This can happen with some reasoning models — try "
                              "again, or pick a different model on the Models "
                              "page."})
            if stats and "".join(full_answer).strip():
                events_meta.append(stats)
                yield ev(stats)
    except Exception as e:
        yield ev({"type": "error", "error": f"{e}"})
    finally:
        answer = "".join(full_answer).strip()
        if answer or events_meta:
            qx("INSERT INTO messages(id,conv_id,role,content,meta,created) "
               "VALUES(?,?,?,?,?,?)",
               (new_id(), conv_id, "assistant", answer,
                json.dumps({"events": events_meta}), now()))
    yield ev({"type": "done"})


LOOP_PROTOCOL = (
    "\nYou are running in AUTONOMOUS LOOP mode. Work the task end to end by "
    "yourself: think briefly, call tools to make real progress, observe the "
    "results, and repeat. Do not ask the user questions — make sensible "
    "decisions and proceed. Keep each thought short. When, and only when, "
    "the task is genuinely finished, call the task_complete tool with the "
    "complete final answer for the user. Never invent tool output and never "
    "claim completion for work you did not do.")

TASK_COMPLETE_TOOL = {"type": "function", "function": {
    "name": "task_complete",
    "description": "Finish the autonomous loop. Call this exactly once, only "
                   "when the whole task is done, with the final answer.",
    "parameters": {"type": "object", "properties": {
        "answer": {"type": "string",
                   "description": "The complete final answer / result for "
                                  "the user."}},
        "required": ["answer"]}}}


async def _stream_text(text: str, emit) -> AsyncGenerator[str, None]:
    for i in range(0, len(text), 24):
        yield emit({"type": "token", "text": text[i:i + 24]})
        await asyncio.sleep(0)


async def _run_agent(messages: list, model: str, events_meta: list,
                     full_answer: list,
                     loop_mode: bool = False,
                     tools_override: Optional[list] = None,
                     protocol_override: Optional[str] = None,
                     budget_override: Optional[int] = None,
                     label_override: Optional[str] = None
                     ) -> AsyncGenerator[str, None]:
    def ev(obj: dict) -> str:
        if obj.get("type") in ("tool_call", "tool_result", "thought", "stats"):
            events_meta.append(obj)
        return json.dumps(obj, ensure_ascii=False) + "\n"

    tools = (tools_override if tools_override is not None
             else builtin_tool_schemas() + MCP.agent_tools())
    if loop_mode:
        tools = tools + [TASK_COMPLETE_TOOL]
        budget = int(get_setting("loop_max_steps") or 15)
        extra = LOOP_PROTOCOL
        label = "Loop"
    else:
        budget = int(get_setting("agent_max_steps") or 8)
        extra = ("\nYou can call tools. Use them whenever they help, then "
                 "give the user a clear final answer. Never invent tool "
                 "output.")
        label = "Thinking"
    if protocol_override is not None:
        extra = protocol_override
    if budget_override is not None:
        budget = budget_override
    if label_override is not None:
        label = label_override

    convo = list(messages)
    convo[0] = {"role": "system",
                "content": get_setting("system_prompt") + extra}

    for step in range(budget):
        yield ev({"type": "status",
                  "text": f"{label} — step {step + 1}/{budget}…"})
        try:
            resp = await ollama_chat(convo, model, tools=tools)
        except httpx.HTTPStatusError as e:
            body = e.response.text[:300]
            if "does not support tools" in body:
                yield ev({"type": "status",
                          "text": "This model has no tool support — answering "
                                  "directly instead."})
                async for part in ollama_chat_stream(messages, model):
                    msg = part.get("message") or {}
                    think = msg.get("thinking") or msg.get("reasoning") or ""
                    if think:
                        yield ev({"type": "thinking", "text": think})
                    token = msg.get("content", "")
                    if token:
                        full_answer.append(token)
                        yield ev({"type": "token", "text": token})
                    if part.get("done"):
                        st = _gen_stats(part)
                        if st:
                            yield ev(st)
                        return
                return
            yield ev({"type": "error", "error": f"Ollama error: {body}"})
            return

        msg = resp.get("message") or {}
        calls = msg.get("tool_calls") or []
        content = (msg.get("content") or "").strip()

        if calls:
            if content:   # models often narrate their plan alongside calls
                yield ev({"type": "thought", "text": content[:700]})
            convo.append(msg)
            for call in calls:
                fn = (call.get("function") or {})
                name = fn.get("name", "")
                args = fn.get("arguments") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {"input": args}

                if name == "task_complete":
                    answer = str(args.get("answer", "")).strip() \
                        or "(the loop finished without a written answer)"
                    full_answer.append(answer)
                    yield ev({"type": "status", "text": "Task complete."})
                    async for line in _stream_text(answer, ev):
                        yield line
                    st = _gen_stats(resp)
                    if st:
                        yield ev(st)
                    return

                yield ev({"type": "tool_call", "name": name, "args": args})
                if name.startswith("mcp__"):
                    result = await MCP.call(name, args)
                elif name.startswith("coder_"):
                    result = await call_coder_tool(name, args)
                else:
                    result = await call_builtin_tool(name, args)
                yield ev({"type": "tool_result", "name": name,
                          "result": result[:1500]})
                convo.append({"role": "tool", "content": result[:8000],
                              "tool_name": name})
            continue

        # ---- no tool calls this step ----
        if loop_mode:
            # In loop mode plain text is a thought, not the final answer:
            # keep it visible and nudge the model onward.
            if content:
                yield ev({"type": "thought", "text": content[:700]})
                convo.append({"role": "assistant", "content": content})
            convo.append({"role": "user", "content":
                          "(system) Continue working autonomously. Call tools "
                          "to make progress, or call task_complete with the "
                          "final answer once the task is fully done."})
            continue

        # regular agent mode: plain text is the final answer
        if content:
            full_answer.append(content)
            async for line in _stream_text(content, ev):
                yield line
            st = _gen_stats(resp)
            if st:
                yield ev(st)
            return
        # Empty final answer — ask the model once more, plainly, so the user
        # isn't left with a blank reply.
        yield ev({"type": "status", "text": "Composing the final answer…"})
        st = None
        async for part in ollama_chat_stream(messages, model, think=False):
            tok = (part.get("message") or {}).get("content", "")
            if tok:
                full_answer.append(tok)
                yield ev({"type": "token", "text": tok})
            if part.get("done"):
                st = _gen_stats(part)
                break
        if st and "".join(full_answer).strip():
            yield ev(st)
        if not "".join(full_answer).strip():
            yield ev({"type": "error", "error":
                      "The model finished without a written answer. Try again "
                      "or pick a different model."})
        return

    if loop_mode:
        yield ev({"type": "error",
                  "error": f"The loop reached its {budget}-step limit before "
                           "calling task_complete. Progress so far is shown "
                           "above — you can raise the limit in Settings and "
                           "ask it to continue."})
    else:
        yield ev({"type": "error",
                  "error": "The agent hit its step limit before finishing. "
                           "You can raise it in Settings."})

# --------------------------------------------------------------------------
# Computer-control agent — a screenshot-observe-act loop with vision
# --------------------------------------------------------------------------

COMPUTER_PROTOCOL = (
    "You can operate this computer by looking at the screen and controlling "
    "the mouse and keyboard. Work in a tight loop: call screen_capture to SEE "
    "the current screen, decide ONE next action, perform it, then capture "
    "again to check the result before continuing. Coordinates are absolute "
    "pixels with (0,0) at the top-left. Prefer keyboard shortcuts when they "
    "are more reliable than clicking. Take your time and verify each step. "
    "Never assume an action worked — look. If you appear stuck, are unsure, "
    "or the task looks risky or complete, stop and call task_complete with a "
    "short summary for the user. Do not attempt logins, payments, or "
    "destructive actions unless the user explicitly asked for them.")


async def _run_computer_agent(messages: list, model: str, events_meta: list,
                              full_answer: list) -> AsyncGenerator[str, None]:
    def ev(obj: dict) -> str:
        if obj.get("type") in ("computer_action", "computer_confirm",
                               "computer_stopped", "thought", "status"):
            if obj.get("type") in ("computer_action", "thought"):
                events_meta.append(obj)
        return json.dumps(obj, ensure_ascii=False) + "\n"

    if not flag("computer_control"):
        yield ev({"type": "error", "error":
                  "Computer control is turned off. Open the Computer page, "
                  "read the safety notice and enable it first."})
        return
    if not computer_available()["installed"]:
        yield ev({"type": "error", "error":
                  "Computer control isn't installed. Enable it on the "
                  "Computer page (one-click install)."})
        return

    computer_reset()
    budget = _clamp(get_setting("loop_max_steps"), 1, 60, 15)

    # A queue lets synchronous tool wrappers emit UI events mid-execution.
    emitted: "asyncio.Queue" = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def emit(obj):
        loop.call_soon_threadsafe(emitted.put_nowait, obj)

    w, h = _screen_size()
    convo = [{"role": "system", "content":
              get_setting("system_prompt") + "\n" + COMPUTER_PROTOCOL +
              f"\nThe screen is {w}x{h} pixels."}] + \
            [m for m in messages if m.get("role") != "system"]
    tools = computer_tool_schemas() + [TASK_COMPLETE_TOOL]

    yield ev({"type": "status", "text": "Starting computer control — taking a "
                                        "first look at the screen…"})

    # Always begin by showing the model the current screen.
    first = await loop.run_in_executor(None, _do_action, "screen_capture", {})
    ev_entry = {"type": "computer_action", "kind": "screen_capture",
                "detail": first.get("detail", "captured screen"),
                "shot": first.get("shot")}
    events_meta.append(ev_entry)
    yield ev(ev_entry)
    convo.append({"role": "user", "content":
                  "Here is the current screen.",
                  "images": [first["_b64"]] if first.get("_b64") else []})

    for step in range(budget):
        if _computer_stop.is_set():
            yield ev({"type": "error", "error": "Stopped by user."})
            return
        yield ev({"type": "status", "text": f"Deciding next action "
                                            f"(step {step + 1}/{budget})…"})
        try:
            resp = await ollama_chat(convo, model, tools=tools)
        except httpx.HTTPStatusError as e:
            body = e.response.text[:300]
            if "does not support tools" in body:
                yield ev({"type": "error", "error":
                          "This model can't call tools, so it can't control "
                          "the computer. Try a tool-capable model such as "
                          "qwen3, llama3.1/3.3 or mistral."})
            else:
                yield ev({"type": "error", "error": f"Ollama error: {body}"})
            return

        msg = resp.get("message") or {}
        calls = msg.get("tool_calls") or []
        content = (msg.get("content") or "").strip()
        if content:
            yield ev({"type": "thought", "text": content[:700]})

        if not calls:
            # No action chosen — nudge once, then treat as done.
            convo.append({"role": "assistant", "content": content or ""})
            convo.append({"role": "user", "content":
                          "(system) Either perform the next action with a "
                          "tool, or call task_complete if you are done."})
            continue

        convo.append(msg)
        for call in calls:
            if _computer_stop.is_set():
                yield ev({"type": "error", "error": "Stopped by user."})
                return
            fn = call.get("function") or {}
            cname = fn.get("name", "")
            cargs = fn.get("arguments") or {}
            if isinstance(cargs, str):
                try:
                    cargs = json.loads(cargs)
                except Exception:
                    cargs = {}

            if cname == "task_complete":
                answer = str(cargs.get("answer", "")).strip() or \
                    "Finished operating the computer."
                full_answer.append(answer)
                yield ev({"type": "status", "text": "Task complete."})
                async for line in _stream_text(answer, ev):
                    yield line
                return

            # Run the tool; drain any UI events it emits (confirm prompts etc).
            task = asyncio.create_task(call_computer_tool(cname, cargs, emit))
            while not task.done():
                try:
                    obj = await asyncio.wait_for(emitted.get(), timeout=0.1)
                    yield ev(obj)
                except asyncio.TimeoutError:
                    pass
            while not emitted.empty():
                yield ev(emitted.get_nowait())
            result = await task

            # Feed the result back. For screenshots, attach the new image.
            if cname == "screen_capture":
                shot = await loop.run_in_executor(
                    None, _do_action, "screen_capture", {})
                convo.append({"role": "tool", "content": result,
                              "tool_name": cname})
                convo.append({"role": "user",
                              "content": "Updated screen after your last "
                                         "action.",
                              "images": [shot["_b64"]]
                              if shot.get("_b64") else []})
            else:
                convo.append({"role": "tool", "content": result,
                              "tool_name": cname})

    yield ev({"type": "error", "error":
              f"Reached the {budget}-step limit. Raise 'Loop iteration limit' "
              "in Settings if the task needs more steps."})

# --------------------------------------------------------------------------
# Council mode — a panel of consultants deliberating in parallel
# --------------------------------------------------------------------------

FALLBACK_ROLES = [
    {"role": "Systems Analyst", "focus": "structure, constraints and second-order effects"},
    {"role": "Devil's Advocate", "focus": "attack the popular answer; surface failure modes"},
    {"role": "Domain Expert", "focus": "established best practice in this field"},
    {"role": "Pragmatist", "focus": "what actually works with limited time and money"},
    {"role": "Creative Strategist", "focus": "unconventional options others miss"},
    {"role": "Risk Assessor", "focus": "what can go wrong, likelihood and mitigation"},
    {"role": "Economist", "focus": "costs, incentives and trade-offs"},
    {"role": "End-user Advocate", "focus": "lived experience of the people affected"},
    {"role": "Engineer", "focus": "feasibility and implementation detail"},
    {"role": "Ethicist", "focus": "fairness, harm and long-term consequences"},
]


def _clamp(v, lo, hi, default):
    try:
        return max(lo, min(hi, int(v)))
    except Exception:
        return default


async def _council_roles(question: str, n: int, model: str) -> list:
    """Ask the model to design the panel; fall back to a fixed roster."""
    try:
        resp = await asyncio.wait_for(ollama_chat([
            {"role": "system", "content":
             "You assemble small expert panels. Reply with a JSON array only, "
             "no prose, no code fences."},
            {"role": "user", "content":
             f"Question or topic:\n{question[:1500]}\n\n"
             f"Propose exactly {n} sharply distinct consultant roles for a "
             "panel debating this. Include at least one contrarian red-team "
             "role. Each item must be an object with keys 'role' (2-4 word "
             "title) and 'focus' (one short line on their angle). "
             "JSON array only."}], model), timeout=300)
        text = (resp.get("message") or {}).get("content", "")
        m = re.search(r"\[.*\]", text, re.S)
        roles = json.loads(m.group(0)) if m else []
        clean = []
        for r in roles:
            if isinstance(r, dict) and r.get("role"):
                clean.append({"role": str(r["role"])[:48],
                              "focus": str(r.get("focus", ""))[:140]})
        if len(clean) >= 2:
            return clean[:n]
    except Exception:
        pass
    return FALLBACK_ROLES[:n]


async def _council_brief(question: str, model: str) -> Optional[str]:
    """Optional shared research: the model proposes a few searches, results
    are pooled into one briefing every consultant receives."""
    if not flag("allow_web_tools"):
        return None
    try:
        resp = await asyncio.wait_for(ollama_chat([
            {"role": "system", "content":
             "Reply with a JSON array of strings only."},
            {"role": "user", "content":
             f"Up to 3 web search queries that would inform a panel debating:"
             f"\n{question[:800]}\nJSON array of query strings only."}],
            model), timeout=180)
        m = re.search(r"\[.*\]", (resp.get("message") or {}).get("content", ""),
                      re.S)
        queries = [str(q)[:120] for q in (json.loads(m.group(0)) if m else [])
                   if str(q).strip()][:3]
        if not queries:
            return None
        parts = []
        for qy in queries:
            out = await perform_web_search(qy)
            parts.append(f"### Search: {qy}\n{out[:1200]}")
        return "\n\n".join(parts)[:3600]
    except Exception:
        return None


async def _run_council(messages: list, model: str, events_meta: list,
                       full_answer: list) -> AsyncGenerator[str, None]:
    def ev(obj: dict) -> str:
        if obj.get("type") in ("council_start", "council_round",
                               "council_take", "consultant_status",
                               "council_brief"):
            events_meta.append(obj)
        return json.dumps(obj, ensure_ascii=False) + "\n"

    n = _clamp(get_setting("council_size"), 2, 10, 4)
    rounds = _clamp(get_setting("council_rounds"), 0, 3, 1)
    question = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            question = m.get("content", "")
            break

    yield ev({"type": "status", "text": "Assembling the council…"})
    roles = await _council_roles(question, n, model)
    n = len(roles)
    yield ev({"type": "council_start", "size": n, "rounds": rounds,
              "consultants": [{"id": i, **roles[i]} for i in range(n)]})

    brief = None
    if flag("council_research"):
        yield ev({"type": "status", "text": "Preparing a shared research brief…"})
        brief = await _council_brief(question, model)
        if brief:
            yield ev({"type": "council_brief", "text": brief[:2000]})

    base_system = get_setting("system_prompt")
    convos: list = []
    for i, r in enumerate(roles):
        sysmsg = (f"{base_system}\nYou are consultant #{i + 1} of {n} on an "
                  f"expert panel: {r['role']} — {r['focus']}. Give YOUR "
                  "independent professional take on the user's latest "
                  "message: take a clear position, be concrete, name key "
                  "assumptions and risks from your angle. Under 180 words. "
                  "Address the user, not the other consultants (you have not "
                  "heard them yet).")
        convo = [{"role": "system", "content": sysmsg}] + [
            m for m in messages if m.get("role") != "system"]
        if brief:
            convo.insert(1, {"role": "system", "content":
                             "Shared research brief for the panel:\n" + brief})
        convos.append(convo)

    takes: list = [None] * n

    async def _one(i: int) -> tuple:
        try:
            resp = await asyncio.wait_for(
                ollama_chat(convos[i], model), timeout=600)
            text = ((resp.get("message") or {}).get("content", "") or "").strip()
            return i, (text[:2400] or "(no answer)")
        except Exception:
            return i, None

    for rnd in range(rounds + 1):
        label = ("Independent analysis" if rnd == 0
                 else f"Consultation — round {rnd}")
        yield ev({"type": "council_round", "round": rnd, "label": label})
        if rnd > 0:
            for i in range(n):
                if takes[i] is None:
                    continue
                digest = "\n\n".join(
                    f"[{roles[j]['role']}]\n{takes[j][:1200]}"
                    for j in range(n) if j != i and takes[j])
                convos[i].append({"role": "assistant", "content": takes[i]})
                convos[i].append({"role": "user", "content":
                                  f"(consultation round {rnd}) The other "
                                  "consultants said:\n\n" + digest +
                                  "\n\nBriefly respond: where do you agree or "
                                  "disagree, and why? Update your "
                                  "recommendation if warranted. Under 150 "
                                  "words."})
        done = 0
        tasks = [asyncio.create_task(_one(i)) for i in range(n)
                 if rnd == 0 or takes[i] is not None]
        new_takes = list(takes)
        for fut in asyncio.as_completed(tasks):
            i, text = await fut
            done += 1
            if text is None:
                new_takes[i] = None
                yield ev({"type": "consultant_status", "id": i,
                          "round": rnd, "state": "failed"})
            else:
                new_takes[i] = text
                yield ev({"type": "council_take", "id": i, "round": rnd,
                          "text": text})
            yield ev({"type": "status",
                      "text": f"{label} — {done}/{len(tasks)} consultants done…"})
        takes = new_takes
        if not any(takes):
            yield ev({"type": "error", "error":
                      "Every consultant failed to answer — is the model "
                      "loaded and Ollama healthy?"})
            return

    yield ev({"type": "status", "text": "The chair is synthesizing the "
                                        "final answer…"})
    digest_all = "\n\n".join(
        f"[{roles[j]['role']}]\n{takes[j][:1400]}"
        for j in range(n) if takes[j])
    chair = [{"role": "system", "content":
              base_system + "\nYou are the chair of a consultant panel. "
              "Write the final consolidated answer for the user."}] + \
            [m for m in messages if m.get("role") != "system"] + \
            [{"role": "user", "content":
              "(chair instruction) The panel has finished. Their final "
              "positions:\n\n" + digest_all +
              "\n\nWrite the definitive answer to my original question: the "
              "consensus, any notable disagreements (name the roles), and a "
              "clear recommendation. Do not pad."}]
    try:
        chair_stats = None
        async for part in ollama_chat_stream(chair, model):
            token = (part.get("message") or {}).get("content", "")
            if token:
                full_answer.append(token)
                yield ev({"type": "token", "text": token})
            if part.get("done"):
                chair_stats = _gen_stats(part)
                break
        if chair_stats:
            events_meta.append(chair_stats)
            yield ev(chair_stats)
    except Exception as e:
        yield ev({"type": "error", "error": f"Synthesis failed: {e}"})

# --------------------------------------------------------------------------
# Update manager — notices newer localmind*.py files and swaps itself
# --------------------------------------------------------------------------

VERSION_RE = re.compile(r"__version__\s*=\s*['\"]([0-9]+(?:\.[0-9]+)*)['\"]")


def _parse_version(text: str):
    m = VERSION_RE.search(text)
    if not m:
        return None
    return tuple(int(p) for p in m.group(1).split("."))


def _version_str(v: tuple) -> str:
    return ".".join(str(p) for p in v)


CURRENT_VERSION = _parse_version(f"__version__ = '{__version__}'") or (0,)


def update_scan_dirs() -> list:
    dirs = [DATA_DIR / "updates", BASE_DIR]
    dl = Path.home() / "Downloads"
    if dl.is_dir():
        dirs.append(dl)
    return dirs


def check_for_update() -> dict:
    best = None
    for d in update_scan_dirs():
        try:
            entries = []
            for stem in FILE_STEMS:
                entries.extend(d.glob(f"{stem}*.py"))
        except Exception:
            continue
        for f in entries:
            try:
                if f.resolve() == SCRIPT_PATH.resolve():
                    continue
                head = f.read_text(encoding="utf-8", errors="replace")[:6000]
                v = _parse_version(head)
                if not v or v <= CURRENT_VERSION:
                    continue
                cand = {"version": _version_str(v), "vtuple": v,
                        "path": str(f), "dir": str(d),
                        "size_kb": round(f.stat().st_size / 1024),
                        "modified": f.stat().st_mtime}
                if best is None or v > best["vtuple"]:
                    best = cand
            except Exception:
                continue
    if best:
        best.pop("vtuple", None)
        return {"available": True, "current": __version__, "update": best,
                "watched": [str(d) for d in update_scan_dirs()]}
    return {"available": False, "current": __version__,
            "watched": [str(d) for d in update_scan_dirs()]}


def apply_update(path_str: str) -> dict:
    src = Path(path_str)
    if not src.is_file():
        return {"ok": False, "error": "Update file not found."}
    text = src.read_text(encoding="utf-8", errors="replace")
    v = _parse_version(text)
    if not v:
        return {"ok": False, "error": "No __version__ found in that file."}
    if v <= CURRENT_VERSION:
        return {"ok": False, "error":
                f"That file is v{_version_str(v)}, not newer than v{__version__}."}
    try:
        compile(text, str(src), "exec")   # refuse files with syntax errors
    except SyntaxError as e:
        return {"ok": False, "error": f"Update has a syntax error: {e}"}

    backup = DATA_DIR / "backups" / f"{FILE_STEM}_v{__version__}_{int(now())}.py"
    try:
        shutil.copy2(SCRIPT_PATH, backup)
        tmp = SCRIPT_PATH.with_suffix(".py.new")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, SCRIPT_PATH)
    except Exception as e:
        return {"ok": False, "error": f"Could not write the update: {e}"}

    threading.Thread(target=_restart_process, daemon=True).start()
    return {"ok": True, "new_version": _version_str(v),
            "backup": str(backup)}


def _restart_argv(argv: list, host: str, port: int) -> list:
    """Child arguments for a restart: whatever the user passed originally,
    minus any --host/--port pair, plus the address we are actually bound to.
    Without this, a server that auto-picked a free port (or was started on a
    custom one) restarts on the default port and the browser is stranded."""
    out, skip = [], False
    for a in argv:
        if skip:
            skip = False
            continue
        if a in ("--host", "--port"):
            skip = True
            continue
        out.append(a)
    return out + ["--host", str(host), "--port", str(port)]


def _restart_process() -> None:
    time.sleep(0.8)  # let the HTTP response reach the browser
    print(f"[{APP_NAME}] restarting into the new version…")
    env = dict(os.environ, HEORTH_RESTART="1")
    env.pop("HEORTH_BOOTSTRAPPED", None)      # re-check deps for new version
    env.pop("LOCALMIND_BOOTSTRAPPED", None)
    kwargs: dict = {}
    if os.name == "nt":
        kwargs["creationflags"] = 0x00000008 | 0x00000200  # DETACHED|NEW_GROUP
    else:
        kwargs["start_new_session"] = True
    args = _restart_argv(sys.argv[1:], RUNTIME["host"], RUNTIME["port"])
    try:
        log = open(DATA_DIR / "restart.log", "ab")
        log.write((f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')} restarting "
                   f"on {RUNTIME['host']}:{RUNTIME['port']} "
                   f"(argv: {args}) ---\n").encode())
        log.flush()
        kwargs["stdout"] = log
        kwargs["stderr"] = subprocess.STDOUT
    except OSError:
        pass                     # no log is better than no restart
    subprocess.Popen([sys.executable, str(SCRIPT_PATH), *args],
                     env=env, **kwargs)
    time.sleep(0.2)
    os._exit(0)


def wait_for_port_free(host: str, port: int, timeout: float = 25) -> None:
    end = time.time() + timeout
    while time.time() < end:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            if s.connect_ex((host if host != "0.0.0.0" else "127.0.0.1",
                             port)) != 0:
                return
        time.sleep(0.5)

# --------------------------------------------------------------------------
# FastAPI application
# --------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(application):
    db_init()
    asyncio.create_task(MCP.connect_enabled())
    yield
    for sid in list(MCP.sessions):
        await MCP.disconnect(sid)


app = FastAPI(title=APP_NAME, lifespan=lifespan)


def _host_allowed(hostport: str) -> bool:
    """Only localhost and IP-literal hosts are legitimate for a local app.
    A DNS name that is not localhost means some website's domain was pointed
    at this machine (DNS rebinding) — always reject it."""
    h = (hostport or "").strip().lower()
    if h.startswith("["):                       # [::1]:8990
        h = h[1:h.index("]")] if "]" in h else h[1:]
    else:
        h = h.split(":")[0]
    bare = h[:-1] if h.endswith(".") else h
    if bare == "localhost" or bare.endswith(".localhost"):
        return True
    # Names public DNS can never serve cannot be DNS-rebound by a website:
    # single-label LAN hostnames (mypc), mDNS (*.local, RFC 6762) and home
    # networks (*.home.arpa, RFC 8375).
    if "." not in bare or bare.endswith((".local", ".home.arpa", ".ts.net")):
        return True    # .ts.net = Tailscale MagicDNS (zone owned by Tailscale)
    try:
        ipaddress.ip_address(bare)
        return True
    except ValueError:
        return False


@app.middleware("http")
async def _local_guard(request: Request, call_next):
    """Protect the unauthenticated local API from the two classic attacks
    on localhost apps: DNS rebinding (Host check) and cross-site request
    forgery from web pages (Origin check)."""
    if not _host_allowed(request.headers.get("host", "")):
        return JSONResponse({"ok": False, "error":
                             "Blocked: unexpected Host header "
                             "(DNS-rebinding protection)."}, status_code=403)
    origin = request.headers.get("origin")
    if origin:
        from urllib.parse import urlsplit
        onet = urlsplit(origin).netloc if origin != "null" else ""
        if origin == "null" or not _host_allowed(onet):
            return JSONResponse({"ok": False, "error":
                                 "Blocked: cross-site request "
                                 "(CSRF protection)."}, status_code=403)
    pw = (get_setting("lan_password") or "").strip()
    if pw and not _is_loopback_client(request):
        path = request.url.path
        if not (path.startswith("/artifact/") or path == "/api/auth"):
            if request.cookies.get("heorth_auth", "") not in _AUTH_TOKENS:
                if path.startswith("/api"):
                    return JSONResponse(
                        {"ok": False, "error":
                         "Unlock required — open Heorth in the browser and "
                         "enter the access password."}, status_code=401)
                return HTMLResponse(_LOGIN_HTML, status_code=401)
    resp = await call_next(request)
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    return resp


def _is_loopback_client(request) -> bool:
    """True only when the TCP peer is this same machine. Header-independent,
    so it cannot be spoofed by a remote client."""
    try:
        host = (request.client.host if request.client else "") or ""
        return ipaddress.ip_address(host.split("%")[0]).is_loopback
    except ValueError:
        return False


_LOGIN_HTML = """<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Heorth — unlock</title><style>
body{margin:0;min-height:100vh;display:flex;align-items:center;justify-content:center;
background:#0b0f14;color:#e6edf3;font-family:Inter,-apple-system,Segoe UI,Roboto,sans-serif}
.card{background:#121821;border:1px solid #1f2a37;border-radius:14px;padding:30px 28px;
width:min(88vw,340px);text-align:center}
h1{font-size:19px;margin:0 0 6px}p{color:#7d8da1;font-size:13px;margin:0 0 18px}
input{width:100%;box-sizing:border-box;background:#0e141c;border:1px solid #1f2a37;color:#e6edf3;
border-radius:9px;padding:11px 12px;font-size:15px;margin-bottom:12px;outline:none}
input:focus{border-color:#f5a623}
button{width:100%;background:#f5a623;color:#0b0f14;border:none;border-radius:9px;
padding:11px;font-size:15px;font-weight:600;cursor:pointer}
.err{color:#f87171;font-size:13px;min-height:17px;margin:10px 0 0}
</style></head><body><div class="card">
<h1>Heorth</h1><p>This server is protected. Enter the access password.</p>
<input id="pw" type="password" placeholder="Access password" autofocus>
<button id="go">Unlock</button><div class="err" id="err"></div>
<script>
async function unlock(){
  const r = await fetch("/api/auth", {method:"POST",
    headers:{"Content-Type":"application/json"},
    body: JSON.stringify({password: document.getElementById("pw").value})});
  if(r.ok){ location.reload(); }
  else { document.getElementById("err").textContent = "Wrong password."; }
}
document.getElementById("go").onclick = unlock;
document.getElementById("pw").addEventListener("keydown",
  e => { if(e.key === "Enter") unlock(); });
</script></div></body></html>"""


def jerr(message: str, code: int = 400) -> JSONResponse:
    return JSONResponse({"ok": False, "error": message}, status_code=code)


@app.get("/api/health")
async def api_health():
    return {"ok": True, "app": APP_NAME, "version": __version__}


@app.get("/api/system")
async def api_system():
    hw = detect_hardware()
    status = await ollama_up()
    return {"ok": True, "hardware": hw, "ollama": status,
            "recommendation": recommend_models(hw),
            "install_help": ollama_install_help(hw),
            "version": __version__,
            "data_dir": str(DATA_DIR),
            "imagegen": imagegen_available(),
            "mcp_installed": mcp_available(),
            "computer": {**computer_available(),
                         "enabled": get_setting("computer_control") == "1"}}


@app.get("/api/settings")
async def api_get_settings():
    return {"ok": True, "settings": all_settings()}


@app.post("/api/settings")
async def api_set_settings(req: Request):
    body = await req.json()
    for key, value in (body or {}).items():
        if key in DEFAULT_SETTINGS:
            set_setting(key, str(value))
    return {"ok": True, "settings": all_settings()}

# ---- models -----------------------------------------------------------------


@app.get("/api/models/installed")
async def api_models_installed():
    status = await ollama_up()
    return {"ok": True, "ollama": status,
            "models": await ollama_installed_models() if status["up"] else []}


@app.get("/api/models/catalog")
async def api_models_catalog():
    hw = detect_hardware()
    return {"ok": True, "catalog": MODEL_CATALOG,
            "recommendation": recommend_models(hw), "hardware": hw}


@app.get("/api/models/search")
async def api_models_search(q: str = ""):
    if not q.strip():
        return {"ok": True, "catalog": [], "huggingface": []}
    res = await search_models(q.strip())
    return {"ok": True, **res}


@app.post("/api/models/pull")
async def api_models_pull(req: Request):
    body = await req.json()
    name = (body or {}).get("name", "").strip()
    if not name:
        return jerr("Missing model name.")
    return StreamingResponse(ollama_pull_stream(name),
                             media_type="application/x-ndjson")


@app.post("/api/models/delete")
async def api_models_delete(req: Request):
    body = await req.json()
    name = (body or {}).get("name", "").strip()
    ok = await ollama_delete(name)
    return {"ok": ok} if ok else jerr("Delete failed — is Ollama running?")

# ---- conversations / chat ---------------------------------------------------


@app.get("/api/conversations")
async def api_conversations():
    rows = q("SELECT id, title, created FROM conversations ORDER BY created DESC")
    return {"ok": True, "conversations": rows}


@app.delete("/api/conversations/{conv_id}")
async def api_del_conversation(conv_id: str):
    qx("DELETE FROM messages WHERE conv_id=?", (conv_id,))
    qx("DELETE FROM conversations WHERE id=?", (conv_id,))
    return {"ok": True}


@app.get("/api/conversations/{conv_id}/messages")
async def api_conv_messages(conv_id: str):
    rows = q("SELECT role, content, meta, created FROM messages "
             "WHERE conv_id=? ORDER BY created", (conv_id,))
    for r in rows:
        try:
            r["meta"] = json.loads(r["meta"] or "{}")
        except Exception:
            r["meta"] = {}
    return {"ok": True, "messages": rows}


@app.post("/api/chat")
async def api_chat(req: Request):
    body = await req.json()
    message = (body or {}).get("message", "").strip()
    model = (body or {}).get("model", "").strip()
    if not message:
        return jerr("Empty message.")
    if not model:
        return jerr("Pick a model first (Models page).")
    conv_id = (body or {}).get("conversation_id")
    if not conv_id:
        conv_id = new_id()
        title = message[:70] + ("…" if len(message) > 70 else "")
        qx("INSERT INTO conversations(id,title,created) VALUES(?,?,?)",
           (conv_id, title, now()))
    elif not q("SELECT id FROM conversations WHERE id=?", (conv_id,)):
        # stale id from the client — recreate the row so it isn't orphaned
        title = message[:70] + ("…" if len(message) > 70 else "")
        qx("INSERT INTO conversations(id,title,created) VALUES(?,?,?)",
           (conv_id, title, now()))

    images = [x for x in ((body or {}).get("images") or [])
              if isinstance(x, str) and x.strip()][:4]
    if any(len(x) > 12_000_000 for x in images):
        return jerr("An attached image is too large (roughly 8 MB max).")

    async def gen():
        yield json.dumps({"type": "meta", "conversation_id": conv_id}) + "\n"
        async for line in run_chat(conv_id, message,
                                   model,
                                   bool((body or {}).get("use_rag")),
                                   bool((body or {}).get("agent_mode")),
                                   bool((body or {}).get("loop_mode")),
                                   bool((body or {}).get("council_mode")),
                                   bool((body or {}).get("computer_mode")),
                                   coder_mode=bool(
                                       (body or {}).get("coder_mode")),
                                   images=images):
            yield line
    return StreamingResponse(gen(), media_type="application/x-ndjson")

@app.post("/api/conversations/{conv_id}/regenerate")
async def api_regenerate(conv_id: str, req: Request):
    """Delete everything after the last user message and answer it again
    with the current model and mode toggles."""
    body = await req.json() or {}
    model = (body.get("model") or "").strip()
    if not model:
        return jerr("Pick a model first (Models page).")
    rows = q("SELECT role, content, meta, created FROM messages "
             "WHERE conv_id=? ORDER BY created", (conv_id,))
    last_user = next((r for r in reversed(rows)
                      if r["role"] == "user" and (r["content"] or "").strip()),
                     None)
    if last_user is None:
        return jerr("Nothing to regenerate yet.")
    qx("DELETE FROM messages WHERE conv_id=? AND created>?",
       (conv_id, last_user["created"]))

    async def gen():
        yield json.dumps({"type": "meta", "conversation_id": conv_id}) + "\n"
        async for line in run_chat(conv_id, last_user["content"], model,
                                   bool(body.get("use_rag")),
                                   bool(body.get("agent_mode")),
                                   bool(body.get("loop_mode")),
                                   bool(body.get("council_mode")),
                                   bool(body.get("computer_mode")),
                                   coder_mode=bool(body.get("coder_mode")),
                                   save_user=False):
            yield line
    return StreamingResponse(gen(), media_type="application/x-ndjson")


@app.get("/api/conversations/{conv_id}/export")
async def api_conv_export(conv_id: str, fmt: str = "md"):
    conv = q("SELECT * FROM conversations WHERE id=?", (conv_id,))
    if not conv:
        return jerr("No such conversation.", 404)
    rows = q("SELECT role, content, meta, created FROM messages "
             "WHERE conv_id=? ORDER BY created", (conv_id,))
    title = (conv[0]["title"] or "Conversation").strip()
    stamp = time.strftime("%Y-%m-%d %H:%M")
    if fmt == "json":
        payload = json.dumps({"title": title, "exported": stamp,
                              "app": f"{APP_NAME} v{__version__}",
                              "messages": rows},
                             ensure_ascii=False, indent=2, default=str)
        media, ext = "application/json", "json"
    else:
        lines = [f"# {title}", "",
                 f"*Exported from {APP_NAME} v{__version__} on {stamp}*", ""]
        for r in rows:
            if r["role"] not in ("user", "assistant"):
                continue
            body_text = (r["content"] or "").strip()
            if not body_text:
                continue
            lines += [f"## {'You' if r['role'] == 'user' else APP_NAME}",
                      "", body_text, ""]
        payload, media, ext = "\n".join(lines), "text/markdown", "md"
    safe = re.sub(r"[^A-Za-z0-9 _-]", "", title)[:40].strip() or "conversation"
    return Response(payload, media_type=media + "; charset=utf-8", headers={
        "Content-Disposition": f'attachment; filename="{safe}.{ext}"'})


@app.get("/api/coder/check")
async def api_coder_check(path: str = ""):
    """Used by the Coder set-up dialog to validate a folder as you type."""
    raw = (path or "").strip()
    if not raw:
        return {"ok": True, "exists": False, "is_dir": False,
                "resolved": "", "entries": 0}
    try:
        p = Path(raw).expanduser().resolve()
    except Exception:
        return {"ok": True, "exists": False, "is_dir": False,
                "resolved": raw, "entries": 0}
    entries = 0
    if p.is_dir():
        try:
            entries = sum(1 for _ in p.iterdir())
        except OSError:
            entries = 0
    return {"ok": True, "exists": p.exists(), "is_dir": p.is_dir(),
            "resolved": str(p), "entries": entries}


# ---- runnable artifacts (code blocks the user opens in the browser) --------


@app.post("/api/auth")
async def api_auth(req: Request):
    body = await req.json() or {}
    pw = (get_setting("lan_password") or "").strip()
    if not pw:
        return {"ok": True}                      # protection is switched off
    if not hmac.compare_digest(str(body.get("password", "")), pw):
        await asyncio.sleep(0.6)                 # take the edge off guessing
        return jerr("Wrong password.", 401)
    tok = secrets.token_urlsafe(32)
    _AUTH_TOKENS.add(tok)
    while len(_AUTH_TOKENS) > 200:               # keep the set bounded
        _AUTH_TOKENS.pop()
    resp = JSONResponse({"ok": True})
    resp.set_cookie("heorth_auth", tok, max_age=60 * 60 * 24 * 30,
                    httponly=True, samesite="lax")
    return resp


@app.post("/api/artifacts")
async def api_artifact_create(req: Request):
    """Save a generated single-file app (usually HTML) so the browser can
    open and run it. Files are named by content hash, so re-running the
    same code block reuses the same file."""
    body = await req.json() or {}
    content = str(body.get("content", ""))
    if not content.strip():
        return jerr("Empty content.")
    if len(content) > 2_000_000:
        return jerr("Too large (2 MB limit).")
    name = hashlib.sha1(content.encode("utf-8")).hexdigest()[:16] + ".html"
    path = DATA_DIR / "artifacts" / name
    if not path.is_file():
        path.write_text(content, encoding="utf-8")
    return {"ok": True, "url": f"/artifact/{name}", "name": name}


@app.get("/artifact/{filename}")
async def api_artifact_file(filename: str):
    safe = re.sub(r"[^A-Za-z0-9._-]", "", filename)
    path = DATA_DIR / "artifacts" / safe
    if not path.is_file():
        return jerr("Artifact not found.", 404)
    return FileResponse(str(path), media_type="text/html", headers={
        "Content-Security-Policy":
            "sandbox allow-scripts allow-forms allow-modals "
            "allow-popups allow-pointer-lock"})

# ---- knowledge base (RAG) ---------------------------------------------------


@app.get("/api/rag/docs")
async def api_rag_docs():
    rows = q("SELECT id, name, created, chunks FROM docs ORDER BY created DESC")
    status = await ollama_up()
    models = await ollama_installed_models() if status["up"] else []
    embed = get_setting("embed_model")
    have_embed = any(m["name"].split(":")[0] == embed.split(":")[0]
                     for m in models)
    return {"ok": True, "docs": rows, "embed_model": embed,
            "embed_ready": bool(status["up"] and have_embed),
            "ollama_up": status["up"]}


@app.post("/api/rag/upload")
async def api_rag_upload(file: UploadFile = File(...)):
    raw = await file.read()
    if len(raw) > 50 * 1024 * 1024:
        return jerr("File is larger than 50 MB.")
    result = await rag_add_document(file.filename or "document.txt", raw)
    return result if result.get("ok") else jerr(result.get("error", "Failed."))


@app.delete("/api/rag/docs/{doc_id}")
async def api_rag_delete(doc_id: str):
    rag_delete_document(doc_id)
    return {"ok": True}


@app.get("/api/rag/search")
async def api_rag_search(q: str = ""):
    hits = await rag_search(q) if q.strip() else []
    return {"ok": True, "hits": hits}

# ---- images -----------------------------------------------------------------


@app.get("/api/images/status")
async def api_images_status():
    hw = detect_hardware()
    job = _install_jobs.get("imagegen")
    return {"ok": True, **imagegen_available(),
            "presets": [{"key": k, **{kk: vv for kk, vv in v.items()
                                      if kk != "repo"}, "repo": v["repo"]}
                        for k, v in IMAGE_PRESETS.items()],
            "install_job": {"status": job["status"], "log": job["log"][-14:]}
            if job else None,
            "packages": [a for cmd in imagegen_commands(hw) for a in cmd[5:]
                         if not a.startswith("-")
                         and not a.startswith("http")]}


@app.post("/api/images/setup")
async def api_images_setup():
    hw = detect_hardware()
    return start_job("imagegen", imagegen_commands(hw))


@app.get("/api/images")
async def api_images():
    rows = q("SELECT * FROM images ORDER BY created DESC")
    return {"ok": True, "images": rows}


@app.get("/api/images/file/{filename}")
async def api_image_file(filename: str):
    safe = re.sub(r"[^A-Za-z0-9._-]", "", filename)
    path = DATA_DIR / "images" / safe
    if not path.is_file():
        return jerr("Image not found.", 404)
    return FileResponse(str(path), media_type="image/png")


@app.delete("/api/images/{image_id}")
async def api_image_delete(image_id: str):
    rows = q("SELECT filename FROM images WHERE id=?", (image_id,))
    if rows:
        (DATA_DIR / "images" / rows[0]["filename"]).unlink(missing_ok=True)
    qx("DELETE FROM images WHERE id=?", (image_id,))
    return {"ok": True}


@app.post("/api/images/generate")
async def api_images_generate(req: Request):
    if not imagegen_available()["installed"]:
        return jerr("Image generation is not installed yet — click "
                    "'Install image generation' first.")
    body = await req.json() or {}
    prompt = str(body.get("prompt", "")).strip()
    if not prompt:
        return jerr("Write a prompt first.")
    preset = body.get("model") or get_setting("image_model")
    if preset not in IMAGE_PRESETS:
        preset = "sd-turbo"
    size = IMAGE_PRESETS[preset]["size"]
    width = int(body.get("width") or size)
    height = int(body.get("height") or size)
    steps = body.get("steps")
    steps = int(steps) if steps else None
    seed = body.get("seed")
    seed = int(seed) if seed not in (None, "", -1, "-1") else None

    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def progress(text=None, step=None, total=None):
        item = {"type": "status", "text": text} if text else \
               {"type": "step", "step": step, "total": total}
        loop.call_soon_threadsafe(queue.put_nowait, item)

    def work():
        try:
            info = generate_image_sync(prompt, preset, width, height,
                                       steps, seed, progress)
            loop.call_soon_threadsafe(queue.put_nowait,
                                      {"type": "done", "image": info})
        except _ImageCancelled:
            loop.call_soon_threadsafe(queue.put_nowait,
                                      {"type": "cancelled"})
        except Exception as e:
            traceback.print_exc()
            loop.call_soon_threadsafe(queue.put_nowait,
                                      {"type": "error", "error": str(e)[:500]})

    threading.Thread(target=work, daemon=True).start()

    async def gen():
        while True:
            item = await queue.get()
            yield json.dumps(item) + "\n"
            if item["type"] in ("done", "error", "cancelled"):
                break
    return StreamingResponse(gen(), media_type="application/x-ndjson")


@app.post("/api/images/cancel")
async def api_images_cancel():
    _image_cancel.set()
    return {"ok": True}

# ---- tools + MCP ------------------------------------------------------------


@app.get("/api/tools")
async def api_tools():
    return {"ok": True, "builtin": BUILTIN_TOOLS,
            "mcp_installed": mcp_available(),
            "mcp_servers": MCP.list_servers(),
            "allow_code_execution": get_setting("allow_code_execution") == "1",
            "allow_web_tools": get_setting("allow_web_tools") == "1"}


# ---- web search (SearXNG, optional) ----------------------------------------


@app.get("/api/search/status")
async def api_search_status():
    hw = detect_hardware()
    job = _install_jobs.get("searxng")
    return {"ok": True,
            "backend": get_setting("search_backend"),
            "searxng_url": get_setting("searxng_url"),
            "docker": docker_state(),
            "docker_help": docker_install_help(hw),
            "container": searxng_container_state(),
            "searxng": await searxng_probe(),
            "job": {"status": job["status"], "log": job["log"][-14:]}
            if job else None,
            "manual_cmd": searxng_manual_cmd()}


# ---- computer control (optional) --------------------------------------------


@app.get("/api/computer/status")
async def api_computer_status():
    hw = detect_hardware()
    job = _install_jobs.get("computer")
    w, h = _screen_size() if computer_available()["installed"] else (0, 0)
    return {"ok": True, **computer_available(),
            "enabled": get_setting("computer_control") == "1",
            "confirm": get_setting("computer_confirm") == "1",
            "pause": get_setting("computer_pause"),
            "os": hw["os"], "os_notes": computer_os_notes(hw),
            "screen": {"w": w, "h": h},
            "stopped": _computer_stop.is_set(),
            "log": _computer_log[-40:],
            "job": {"status": job["status"], "log": job["log"][-14:]}
            if job else None}


@app.post("/api/computer/setup")
async def api_computer_setup():
    hw = detect_hardware()
    return start_job("computer", computer_packages(hw))


@app.post("/api/computer/stop")
async def api_computer_stop():
    _computer_stop.set()
    for c in list(_pending_confirms.values()):
        c["approved"] = False
        c["event"].set()
    _log_action("stop", "EMERGENCY STOP pressed by user")
    return {"ok": True}


@app.post("/api/computer/confirm")
async def api_computer_confirm(req: Request):
    body = await req.json() or {}
    cid = body.get("id", "")
    approved = bool(body.get("approved"))
    ok = resolve_confirm(cid, approved)
    return {"ok": ok}


@app.get("/api/computer/shot/{filename}")
async def api_computer_shot(filename: str):
    safe = re.sub(r"[^A-Za-z0-9._-]", "", filename)
    path = DATA_DIR / "images" / safe
    if not path.is_file():
        return jerr("Not found.", 404)
    return FileResponse(str(path), media_type="image/png")


@app.post("/api/search/setup")
async def api_search_setup():
    result = searxng_start()
    return result if result.get("ok") else jerr(result.get("error", "Failed."))


@app.get("/api/search/test")
async def api_search_test(q: str = "what is searxng"):
    return {"ok": True, "result": await perform_web_search(q)}


@app.post("/api/mcp/setup")
async def api_mcp_setup():
    return start_install("mcp", ["mcp"])


@app.get("/api/mcp/install_status")
async def api_mcp_install_status():
    job = _install_jobs.get("mcp")
    return {"ok": True, "installed": mcp_available(),
            "job": {"status": job["status"], "log": job["log"][-10:]}
            if job else None}


@app.post("/api/mcp/servers")
async def api_mcp_add(req: Request):
    body = await req.json() or {}
    name = str(body.get("name", "")).strip()
    command = str(body.get("command", "")).strip()
    if not name or not command:
        return jerr("Name and command are required.")
    args = body.get("args") or []
    if isinstance(args, str):
        args = [a for a in args.split() if a]
    env = body.get("env") or {}
    sid = new_id()
    qx("INSERT INTO mcp_servers(id,name,command,args,env,enabled) "
       "VALUES(?,?,?,?,?,1)",
       (sid, name, command, json.dumps(args), json.dumps(env)))
    result = await MCP.connect(sid)
    return {"ok": True, "id": sid, "connect": result}


@app.post("/api/mcp/servers/{sid}/connect")
async def api_mcp_connect(sid: str):
    return await MCP.connect(sid)


@app.delete("/api/mcp/servers/{sid}")
async def api_mcp_delete(sid: str):
    await MCP.disconnect(sid)
    qx("DELETE FROM mcp_servers WHERE id=?", (sid,))
    return {"ok": True}

# ---- updates ----------------------------------------------------------------


@app.get("/api/update/check")
async def api_update_check():
    return {"ok": True, **check_for_update()}


@app.post("/api/update/apply")
async def api_update_apply(req: Request):
    body = await req.json() or {}
    path = body.get("path", "")
    result = apply_update(path)
    return result if result.get("ok") else jerr(result.get("error", "Failed."))


@app.post("/api/update/upload")
async def api_update_upload(file: UploadFile = File(...)):
    name = os.path.basename(file.filename or "")
    if not (name.startswith(FILE_STEMS) and name.endswith(".py")):
        return jerr("Update files must be named heorth*.py or localmind*.py")
    raw = await file.read()
    text = raw.decode("utf-8", errors="replace")
    v = _parse_version(text)
    if not v:
        return jerr("No __version__ found in that file.")
    if v <= CURRENT_VERSION:
        return jerr(f"That file is v{_version_str(v)} — you are already on "
                    f"v{__version__}.")
    dest = DATA_DIR / "updates" / name
    dest.write_bytes(raw)
    return {"ok": True, "path": str(dest), "version": _version_str(v)}

# --------------------------------------------------------------------------
# Frontend (single page, served from the string below)
# --------------------------------------------------------------------------

_INDEX_B64 = (
    "PCFET0NUWVBFIGh0bWw+CjxodG1sIGxhbmc9ImVuIiBkYXRhLXRoZW1lPSJkYXJrIj4KPGhlYWQ+CjxtZXRhIGNoYXJzZXQ9InV0Zi04Ij4KPG1ldGEg"
    "bmFtZT0idmlld3BvcnQiIGNvbnRlbnQ9IndpZHRoPWRldmljZS13aWR0aCwgaW5pdGlhbC1zY2FsZT0xIj4KPHRpdGxlPkhlb3J0aDwvdGl0bGU+Cjxz"
    "dHlsZT4KLyogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09CiAgIEhlb3J0aCDigJQgImNv"
    "bnRyb2wgcm9vbSBmb3IgdGhlIG1hY2hpbmUgeW91IG93biIKICAgUGFsZXR0ZTogIGluayAjMGIwZjE0IMK3IHBhbmVsICMxMjE4MjEgwrcgbGluZSAj"
    "MWYyYTM3CiAgICAgICAgICAgICBzaWduYWwgKGFtYmVyKSAjZjVhNjIzIMK3IGxpdmUgKGdyZWVuKSAjNGFkZTgwCiAgICAgICAgICAgICB0ZXh0ICNl"
    "NmVkZjMgwrcgbXV0ZWQgIzdkOGRhMSDCtyB2aW9sZXQgI2E3OGJmYQogICBUeXBlOiAgICAgZGlzcGxheSA9IFNwYWNlIEdyb3Rlc2staXNoIHZpYSBz"
    "eXN0ZW0gc3RhY2sgZmFsbGJhY2ssCiAgICAgICAgICAgICBib2R5ID0gc3lzdGVtIHNhbnMsIGRhdGEgPSB1aS1tb25vc3BhY2UKICAgPT09PT09PT09"
    "PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09ICovCjpyb290ewogIC0taW5rOiMwYjBmMTQ7IC0tcGFuZWw6"
    "IzEyMTgyMTsgLS1wYW5lbC0yOiMwZTE0MWM7IC0tcmFpc2VkOiMxODIxMmQ7CiAgLS1saW5lOiMxZjJhMzc7IC0tbGluZS1zb2Z0OiMxNzIwMjk7CiAg"
    "LS10ZXh0OiNlNmVkZjM7IC0tbXV0ZWQ6IzdkOGRhMTsgLS1kaW06IzU1NjA3NDsKICAtLXNpZ25hbDojZjVhNjIzOyAtLXNpZ25hbC1kaW06IzdhNTQx"
    "MDsgLS1zaWduYWwtZ2xvdzpyZ2JhKDI0NSwxNjYsMzUsLjE2KTsKICAtLWxpdmU6IzRhZGU4MDsgLS1saXZlLWRpbTojMTQ1MzJkOyAtLXJlZDojZjg3"
    "MTcxOyAtLXZpb2xldDojYTc4YmZhOwogIC0tYmx1ZTojNjBhNWZhOwogIC0tcjoxNHB4OyAtLXItc206OXB4OyAtLXItbGc6MjBweDsKICAtLW1vbm86"
    "dWktbW9ub3NwYWNlLCJTRiBNb25vIiwiSmV0QnJhaW5zIE1vbm8iLE1lbmxvLENvbnNvbGFzLG1vbm9zcGFjZTsKICAtLXNhbnM6IkludGVyIiwtYXBw"
    "bGUtc3lzdGVtLEJsaW5rTWFjU3lzdGVtRm9udCwiU2Vnb2UgVUkiLFJvYm90byxzYW5zLXNlcmlmOwogIC0tZGlzcDoiU3BhY2UgR3JvdGVzayIsIklu"
    "dGVyIix2YXIoLS1zYW5zKTsKICAtLXNoYWRvdzowIDE4cHggNTBweCAtMjBweCByZ2JhKDAsMCwwLC43KTsKfQpbZGF0YS10aGVtZT0ibGlnaHQiXXsK"
    "ICAtLWluazojZThlZGYyOyAtLXBhbmVsOiNmN2Y5ZmI7IC0tcGFuZWwtMjojZWVmMmY3OyAtLXJhaXNlZDojZmZmZmZmOwogIC0tbGluZTojZDJkYmU1"
    "OyAtLWxpbmUtc29mdDojZTBlN2VmOwogIC0tdGV4dDojMTExZDJjOyAtLW11dGVkOiM1NDYxNzc7IC0tZGltOiM4YTkyYTQ7CiAgLS1zaWduYWw6I2E5"
    "NjkwYTsgLS1zaWduYWwtZ2xvdzpyZ2JhKDE2OSwxMDUsMTAsLjEyKTsgLS1zaWduYWwtZGltOiNkOWJlOTM7CiAgLS1saXZlOiMxNTgwM2Q7IC0tbGl2"
    "ZS1kaW06I2JiZjdkMDsgLS1yZWQ6I2RjMjYyNjsgLS12aW9sZXQ6IzZkNGJkODsKICAtLWJsdWU6IzI1NjNlYjsgLS1zaGFkb3c6MCAyMHB4IDUwcHgg"
    "LTI0cHggcmdiYSgzMCw0NSw2NSwuMzApOwp9Cip7Ym94LXNpemluZzpib3JkZXItYm94fQpbaGlkZGVuXXtkaXNwbGF5Om5vbmUhaW1wb3J0YW50fQpo"
    "dG1sLGJvZHl7bWFyZ2luOjA7aGVpZ2h0OjEwMCV9CmJvZHl7CiAgYmFja2dyb3VuZDp2YXIoLS1pbmspOyBjb2xvcjp2YXIoLS10ZXh0KTsgZm9udC1m"
    "YW1pbHk6dmFyKC0tc2Fucyk7CiAgZm9udC1zaXplOjE0LjVweDsgbGluZS1oZWlnaHQ6MS41NTsgLXdlYmtpdC1mb250LXNtb290aGluZzphbnRpYWxp"
    "YXNlZDsKICBvdmVyZmxvdzpoaWRkZW47Cn0KQGltcG9ydCB1cmwoJ2h0dHBzOi8vZm9udHMuZ29vZ2xlYXBpcy5jb20vY3NzMj9mYW1pbHk9SW50ZXI6"
    "d2dodEA0MDA7NTAwOzYwMDs3MDAmZmFtaWx5PVNwYWNlK0dyb3Rlc2s6d2dodEA1MDA7NjAwOzcwMCZkaXNwbGF5PXN3YXAnKTsKOjpzZWxlY3Rpb257"
    "YmFja2dyb3VuZDp2YXIoLS1zaWduYWwtZ2xvdyk7Y29sb3I6dmFyKC0tdGV4dCl9CmF7Y29sb3I6dmFyKC0tc2lnbmFsKTt0ZXh0LWRlY29yYXRpb246"
    "bm9uZX0KYnV0dG9ue2ZvbnQtZmFtaWx5OmluaGVyaXQ7Y29sb3I6aW5oZXJpdDtjdXJzb3I6cG9pbnRlcjtib3JkZXI6bm9uZTtiYWNrZ3JvdW5kOm5v"
    "bmV9CmlucHV0LHRleHRhcmVhLHNlbGVjdHtmb250LWZhbWlseTppbmhlcml0O2NvbG9yOmluaGVyaXR9CmgxLGgyLGgze2ZvbnQtZmFtaWx5OnZhcigt"
    "LWRpc3ApO2ZvbnQtd2VpZ2h0OjYwMDtsZXR0ZXItc3BhY2luZzotLjAxZW07bWFyZ2luOjB9CmNvZGUsa2Jke2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8p"
    "fQprYmR7YmFja2dyb3VuZDp2YXIoLS1yYWlzZWQpO2JvcmRlcjoxcHggc29saWQgdmFyKC0tbGluZSk7Ym9yZGVyLXJhZGl1czo1cHg7CiAgICBwYWRk"
    "aW5nOjFweCA2cHg7Zm9udC1zaXplOjExLjVweH0KLnNjcm9sbHtzY3JvbGxiYXItd2lkdGg6dGhpbjtzY3JvbGxiYXItY29sb3I6dmFyKC0tbGluZSkg"
    "dHJhbnNwYXJlbnR9Ci5zY3JvbGw6Oi13ZWJraXQtc2Nyb2xsYmFye3dpZHRoOjlweDtoZWlnaHQ6OXB4fQouc2Nyb2xsOjotd2Via2l0LXNjcm9sbGJh"
    "ci10aHVtYntiYWNrZ3JvdW5kOnZhcigtLWxpbmUpO2JvcmRlci1yYWRpdXM6NnB4OwogICAgYm9yZGVyOjJweCBzb2xpZCB0cmFuc3BhcmVudDtiYWNr"
    "Z3JvdW5kLWNsaXA6cGFkZGluZy1ib3h9CgovKiAtLS0tLS0tLS0tIHNoZWxsIC0tLS0tLS0tLS0gKi8KLmFwcHtkaXNwbGF5OmdyaWQ7Z3JpZC10ZW1w"
    "bGF0ZS1jb2x1bW5zOjIzMHB4IDFmcjtoZWlnaHQ6MTAwdmh9Ci5zaWRlewogIGJhY2tncm91bmQ6bGluZWFyLWdyYWRpZW50KDE4MGRlZyx2YXIoLS1w"
    "YW5lbC0yKSx2YXIoLS1pbmspKTsKICBib3JkZXItcmlnaHQ6MXB4IHNvbGlkIHZhcigtLWxpbmUpO2Rpc3BsYXk6ZmxleDtmbGV4LWRpcmVjdGlvbjpj"
    "b2x1bW47CiAgcGFkZGluZzoxOHB4IDE0cHg7Z2FwOjRweDtwb3NpdGlvbjpyZWxhdGl2ZTt6LWluZGV4OjU7Cn0KLmJyYW5ke2Rpc3BsYXk6ZmxleDth"
    "bGlnbi1pdGVtczpjZW50ZXI7Z2FwOjExcHg7cGFkZGluZzo2cHggOHB4IDIwcHh9Ci5icmFuZCAubWFya3sKICB3aWR0aDozNHB4O2hlaWdodDozNHB4"
    "O2JvcmRlci1yYWRpdXM6OXB4O2ZsZXg6bm9uZTtwb3NpdGlvbjpyZWxhdGl2ZTsKICBiYWNrZ3JvdW5kOnJhZGlhbC1ncmFkaWVudCgxMjAlIDEyMCUg"
    "YXQgMzAlIDI1JSwjMjQzMjQ0LCMwZDEzMWIpOwogIGJvcmRlcjoxcHggc29saWQgdmFyKC0tbGluZSk7ZGlzcGxheTpncmlkO3BsYWNlLWl0ZW1zOmNl"
    "bnRlcjsKICBib3gtc2hhZG93Omluc2V0IDAgMXB4IDAgcmdiYSgyNTUsMjU1LDI1NSwuMDUpOwp9Ci5icmFuZCAubWFyazo6YmVmb3Jle2NvbnRlbnQ6"
    "IiI7d2lkdGg6MTJweDtoZWlnaHQ6MTJweDtib3JkZXItcmFkaXVzOjUwJTsKICBiYWNrZ3JvdW5kOnZhcigtLXNpZ25hbCk7Ym94LXNoYWRvdzowIDAg"
    "MCA0cHggdmFyKC0tc2lnbmFsLWdsb3cpLAogIDAgMCAxOHB4IDJweCB2YXIoLS1zaWduYWwtZ2xvdyk7YW5pbWF0aW9uOmNvcmVwdWxzZSAzLjRzIGVh"
    "c2UtaW4tb3V0IGluZmluaXRlfQpAa2V5ZnJhbWVzIGNvcmVwdWxzZXswJSwxMDAle29wYWNpdHk6Ljg1O3RyYW5zZm9ybTpzY2FsZSgxKX0KICA1MCV7"
    "b3BhY2l0eToxO3RyYW5zZm9ybTpzY2FsZSgxLjE4KX19Ci5icmFuZCBoMXtmb250LXNpemU6MTdweDtsZXR0ZXItc3BhY2luZzotLjAyZW19Ci5icmFu"
    "ZCAudmVye2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMC41cHg7Y29sb3I6dmFyKC0tZGltKTsKICBtYXJnaW4tdG9wOjFweH0KCi5u"
    "YXZ7ZGlzcGxheTpmbGV4O2ZsZXgtZGlyZWN0aW9uOmNvbHVtbjtnYXA6MnB4O21hcmdpbi10b3A6MnB4fQoubmF2IGJ1dHRvbnsKICBkaXNwbGF5OmZs"
    "ZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDoxMXB4O3BhZGRpbmc6OXB4IDExcHg7Ym9yZGVyLXJhZGl1czp2YXIoLS1yLXNtKTsKICBjb2xvcjp2YXIo"
    "LS1tdXRlZCk7Zm9udC13ZWlnaHQ6NTAwO2ZvbnQtc2l6ZToxMy41cHg7dGV4dC1hbGlnbjpsZWZ0OwogIHRyYW5zaXRpb246YmFja2dyb3VuZCAuMTNz"
    "LGNvbG9yIC4xM3M7cG9zaXRpb246cmVsYXRpdmU7Cn0KLm5hdiBidXR0b246aG92ZXJ7YmFja2dyb3VuZDp2YXIoLS1wYW5lbCk7Y29sb3I6dmFyKC0t"
    "dGV4dCl9Ci5uYXYgYnV0dG9uLmFjdGl2ZXtiYWNrZ3JvdW5kOnZhcigtLXJhaXNlZCk7Y29sb3I6dmFyKC0tdGV4dCl9Ci5uYXYgYnV0dG9uLmFjdGl2"
    "ZTo6YmVmb3Jle2NvbnRlbnQ6IiI7cG9zaXRpb246YWJzb2x1dGU7bGVmdDotMTRweDt0b3A6NTAlOwogIHRyYW5zZm9ybTp0cmFuc2xhdGVZKC01MCUp"
    "O3dpZHRoOjNweDtoZWlnaHQ6MTlweDtib3JkZXItcmFkaXVzOjAgM3B4IDNweCAwOwogIGJhY2tncm91bmQ6dmFyKC0tc2lnbmFsKTtib3gtc2hhZG93"
    "OjAgMCAxMnB4IHZhcigtLXNpZ25hbCl9Ci5uYXYgYnV0dG9uIC5pY3t3aWR0aDoxN3B4O2hlaWdodDoxN3B4O2ZsZXg6bm9uZTtvcGFjaXR5Oi45fQou"
    "bmF2IC5iYWRnZXttYXJnaW4tbGVmdDphdXRvO2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMHB4OwogIGJhY2tncm91bmQ6dmFyKC0t"
    "c2lnbmFsKTtjb2xvcjojMWExMjA0O2JvcmRlci1yYWRpdXM6MjBweDtwYWRkaW5nOjFweCA3cHg7CiAgZm9udC13ZWlnaHQ6NzAwO2xldHRlci1zcGFj"
    "aW5nOi4wMmVtfQoubmF2LXNlcHtoZWlnaHQ6MXB4O2JhY2tncm91bmQ6dmFyKC0tbGluZS1zb2Z0KTttYXJnaW46OXB4IDZweH0KCi5zaWRlLWZvb3R7"
    "bWFyZ2luLXRvcDphdXRvO3BhZGRpbmctdG9wOjEycHg7ZGlzcGxheTpmbGV4O2ZsZXgtZGlyZWN0aW9uOmNvbHVtbjtnYXA6OXB4fQouc3RhdHVzYmFy"
    "e2Rpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7Z2FwOjhweDtwYWRkaW5nOjhweCAxMHB4OwogIGJvcmRlcjoxcHggc29saWQgdmFyKC0tbGlu"
    "ZSk7Ym9yZGVyLXJhZGl1czp2YXIoLS1yLXNtKTtiYWNrZ3JvdW5kOnZhcigtLXBhbmVsLTIpOwogIGZvbnQtc2l6ZToxMnB4fQouZG90e3dpZHRoOjhw"
    "eDtoZWlnaHQ6OHB4O2JvcmRlci1yYWRpdXM6NTAlO2JhY2tncm91bmQ6dmFyKC0tZGltKTtmbGV4Om5vbmV9Ci5kb3Qub257YmFja2dyb3VuZDp2YXIo"
    "LS1saXZlKTtib3gtc2hhZG93OjAgMCA5cHggdmFyKC0tbGl2ZSl9Ci5kb3Qub2Zme2JhY2tncm91bmQ6dmFyKC0tcmVkKTtib3gtc2hhZG93OjAgMCA5"
    "cHggdmFyKC0tcmVkKX0KLmRvdC53YXJue2JhY2tncm91bmQ6dmFyKC0tc2lnbmFsKTtib3gtc2hhZG93OjAgMCA5cHggdmFyKC0tc2lnbmFsKX0KLnRo"
    "ZW1ldG9nZ2xle2Rpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7anVzdGlmeS1jb250ZW50OmNlbnRlcjtnYXA6OHB4OwogIHBhZGRpbmc6N3B4"
    "O2JvcmRlcjoxcHggc29saWQgdmFyKC0tbGluZSk7Ym9yZGVyLXJhZGl1czp2YXIoLS1yLXNtKTsKICBjb2xvcjp2YXIoLS1tdXRlZCk7Zm9udC1zaXpl"
    "OjEycHg7YmFja2dyb3VuZDp2YXIoLS1wYW5lbC0yKX0KLnRoZW1ldG9nZ2xlOmhvdmVye2NvbG9yOnZhcigtLXRleHQpO2JvcmRlci1jb2xvcjp2YXIo"
    "LS1kaW0pfQoKLyogLS0tLS0tLS0tLSBtYWluIC0tLS0tLS0tLS0gKi8KLm1haW57b3ZlcmZsb3c6aGlkZGVuO2Rpc3BsYXk6ZmxleDtmbGV4LWRpcmVj"
    "dGlvbjpjb2x1bW47bWluLXdpZHRoOjA7CiAgYmFja2dyb3VuZDpyYWRpYWwtZ3JhZGllbnQoMTQwJSA4MCUgYXQgMTAwJSAwJSxyZ2JhKDI0NSwxNjYs"
    "MzUsLjAzKSx0cmFuc3BhcmVudCA2MCUpLHZhcigtLWluayl9Ci52aWV3e2ZsZXg6MTtvdmVyZmxvdy15OmF1dG87cGFkZGluZzoyNnB4IDMwcHggNjBw"
    "eH0KLnZpZXcuY2hhdHZpZXd7cGFkZGluZzowO2Rpc3BsYXk6ZmxleDtmbGV4LWRpcmVjdGlvbjpjb2x1bW59Ci5oZHtkaXNwbGF5OmZsZXg7YWxpZ24t"
    "aXRlbXM6ZmxleC1lbmQ7anVzdGlmeS1jb250ZW50OnNwYWNlLWJldHdlZW47Z2FwOjE2cHg7CiAgbWFyZ2luLWJvdHRvbToyMnB4O2ZsZXgtd3JhcDp3"
    "cmFwfQouaGQgLmV5ZWJyb3d7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjExcHg7bGV0dGVyLXNwYWNpbmc6LjE2ZW07CiAgdGV4dC10"
    "cmFuc2Zvcm06dXBwZXJjYXNlO2NvbG9yOnZhcigtLXNpZ25hbCk7bWFyZ2luLWJvdHRvbTo3cHh9Ci5oZCBoMntmb250LXNpemU6MjZweH0KLmhkIHAu"
    "c3Vie2NvbG9yOnZhcigtLW11dGVkKTttYXJnaW46NnB4IDAgMDttYXgtd2lkdGg6NjBjaDtmb250LXNpemU6MTMuNXB4fQoKLyogLS0tLS0tLS0tLSBw"
    "cmltaXRpdmVzIC0tLS0tLS0tLS0gKi8KLmNhcmR7YmFja2dyb3VuZDp2YXIoLS1wYW5lbCk7Ym9yZGVyOjFweCBzb2xpZCB2YXIoLS1saW5lKTtib3Jk"
    "ZXItcmFkaXVzOnZhcigtLXIpOwogIHBhZGRpbmc6MThweH0KLmNhcmQucGFkLWxne3BhZGRpbmc6MjJweH0KLmdyaWR7ZGlzcGxheTpncmlkO2dhcDox"
    "NnB4fQouYnRue2Rpc3BsYXk6aW5saW5lLWZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDo4cHg7cGFkZGluZzo5cHggMTVweDsKICBib3JkZXItcmFk"
    "aXVzOnZhcigtLXItc20pO2ZvbnQtd2VpZ2h0OjYwMDtmb250LXNpemU6MTNweDsKICBib3JkZXI6MXB4IHNvbGlkIHZhcigtLWxpbmUpO2JhY2tncm91"
    "bmQ6dmFyKC0tcmFpc2VkKTtjb2xvcjp2YXIoLS10ZXh0KTsKICB0cmFuc2l0aW9uOi4xNHM7d2hpdGUtc3BhY2U6bm93cmFwfQouYnRuOmhvdmVye2Jv"
    "cmRlci1jb2xvcjp2YXIoLS1kaW0pO3RyYW5zZm9ybTp0cmFuc2xhdGVZKC0xcHgpfQouYnRuOmFjdGl2ZXt0cmFuc2Zvcm06dHJhbnNsYXRlWSgwKX0K"
    "LmJ0bjpkaXNhYmxlZHtvcGFjaXR5Oi40NTtjdXJzb3I6bm90LWFsbG93ZWQ7dHJhbnNmb3JtOm5vbmV9Ci5idG4ucHJpbWFyeXtiYWNrZ3JvdW5kOnZh"
    "cigtLXNpZ25hbCk7Y29sb3I6IzFhMTIwNDtib3JkZXItY29sb3I6dHJhbnNwYXJlbnQ7CiAgYm94LXNoYWRvdzowIDZweCAyMHB4IC04cHggdmFyKC0t"
    "c2lnbmFsKX0KLmJ0bi5wcmltYXJ5OmhvdmVye2ZpbHRlcjpicmlnaHRuZXNzKDEuMDYpfQouYnRuLmdob3N0e2JhY2tncm91bmQ6dHJhbnNwYXJlbnR9"
    "Ci5idG4uZGFuZ2Vye2NvbG9yOnZhcigtLXJlZCk7Ym9yZGVyLWNvbG9yOnRyYW5zcGFyZW50O2JhY2tncm91bmQ6dHJhbnNwYXJlbnR9Ci5idG4uZGFu"
    "Z2VyOmhvdmVye2JhY2tncm91bmQ6cmdiYSgyNDgsMTEzLDExMywuMSl9Ci5idG4uc217cGFkZGluZzo2cHggMTFweDtmb250LXNpemU6MTJweH0KLmJ0"
    "bi5pY29ue3BhZGRpbmc6N3B4O3dpZHRoOjM0cHg7aGVpZ2h0OjM0cHg7anVzdGlmeS1jb250ZW50OmNlbnRlcn0KLmNoaXB7ZGlzcGxheTppbmxpbmUt"
    "ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7Z2FwOjVweDtmb250LWZhbWlseTp2YXIoLS1tb25vKTsKICBmb250LXNpemU6MTAuNXB4O3BhZGRpbmc6MnB4"
    "IDhweDtib3JkZXItcmFkaXVzOjIwcHg7Ym9yZGVyOjFweCBzb2xpZCB2YXIoLS1saW5lKTsKICBjb2xvcjp2YXIoLS1tdXRlZCk7dGV4dC10cmFuc2Zv"
    "cm06bG93ZXJjYXNlO2xldHRlci1zcGFjaW5nOi4wMmVtfQouY2hpcC5obHtib3JkZXItY29sb3I6dmFyKC0tc2lnbmFsLWRpbSk7Y29sb3I6dmFyKC0t"
    "c2lnbmFsKTsKICBiYWNrZ3JvdW5kOnZhcigtLXNpZ25hbC1nbG93KX0KLmNoaXAuZ3Jue2JvcmRlci1jb2xvcjp2YXIoLS1saXZlLWRpbSk7Y29sb3I6"
    "dmFyKC0tbGl2ZSl9Ci5jaGlwLnZpb3tib3JkZXItY29sb3I6IzRjM2Y3YTtjb2xvcjp2YXIoLS12aW9sZXQpfQouZmllbGR7ZGlzcGxheTpibG9jaztt"
    "YXJnaW4tYm90dG9tOjE0cHh9Ci5maWVsZCBsYWJlbHtkaXNwbGF5OmJsb2NrO2ZvbnQtc2l6ZToxMnB4O2NvbG9yOnZhcigtLW11dGVkKTttYXJnaW4t"
    "Ym90dG9tOjZweDsKICBmb250LXdlaWdodDo1MDB9Ci5pbnAsLnRhLC5zZWx7d2lkdGg6MTAwJTtiYWNrZ3JvdW5kOnZhcigtLXBhbmVsLTIpO2JvcmRl"
    "cjoxcHggc29saWQgdmFyKC0tbGluZSk7CiAgYm9yZGVyLXJhZGl1czp2YXIoLS1yLXNtKTtwYWRkaW5nOjEwcHggMTJweDtmb250LXNpemU6MTMuNXB4"
    "O2NvbG9yOnZhcigtLXRleHQpOwogIHRyYW5zaXRpb246Ym9yZGVyIC4xNHMsYm94LXNoYWRvdyAuMTRzO291dGxpbmU6bm9uZX0KLmlucDpmb2N1cywu"
    "dGE6Zm9jdXMsLnNlbDpmb2N1c3tib3JkZXItY29sb3I6dmFyKC0tc2lnbmFsKTsKICBib3gtc2hhZG93OjAgMCAwIDNweCB2YXIoLS1zaWduYWwtZ2xv"
    "dyl9Ci50YXtyZXNpemU6dmVydGljYWw7bWluLWhlaWdodDo3NHB4O2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMi41cHh9Ci5yb3d7"
    "ZGlzcGxheTpmbGV4O2dhcDoxMHB4O2FsaWduLWl0ZW1zOmNlbnRlcn0KLnNwcmVhZHtkaXNwbGF5OmZsZXg7anVzdGlmeS1jb250ZW50OnNwYWNlLWJl"
    "dHdlZW47YWxpZ24taXRlbXM6Y2VudGVyO2dhcDoxMnB4fQoubXV0ZWR7Y29sb3I6dmFyKC0tbXV0ZWQpfS5kaW17Y29sb3I6dmFyKC0tZGltKX0KLm1v"
    "bm97Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyl9Ci5oaW50e2ZvbnQtc2l6ZToxMnB4O2NvbG9yOnZhcigtLWRpbSk7bWFyZ2luLXRvcDo1cHh9Ci5lbXB0"
    "eXt0ZXh0LWFsaWduOmNlbnRlcjtwYWRkaW5nOjQ2cHggMjBweDtjb2xvcjp2YXIoLS1tdXRlZCl9Ci5lbXB0eSAuYmlne2ZvbnQtc2l6ZTozNHB4O21h"
    "cmdpbi1ib3R0b206MTBweDtvcGFjaXR5Oi41fQouZGl2aWRlcntoZWlnaHQ6MXB4O2JhY2tncm91bmQ6dmFyKC0tbGluZS1zb2Z0KTttYXJnaW46MThw"
    "eCAwfQoKLyogcHJvZ3Jlc3MgKi8KLmJhcntoZWlnaHQ6N3B4O2JvcmRlci1yYWRpdXM6NnB4O2JhY2tncm91bmQ6dmFyKC0tcGFuZWwtMik7b3ZlcmZs"
    "b3c6aGlkZGVuOwogIGJvcmRlcjoxcHggc29saWQgdmFyKC0tbGluZSl9Ci5iYXIgPiBpe2Rpc3BsYXk6YmxvY2s7aGVpZ2h0OjEwMCU7YmFja2dyb3Vu"
    "ZDpsaW5lYXItZ3JhZGllbnQoOTBkZWcsdmFyKC0tc2lnbmFsKSwjZmZjZDZiKTsKICB3aWR0aDowO3RyYW5zaXRpb246d2lkdGggLjI1cztib3JkZXIt"
    "cmFkaXVzOjZweH0KLnNwaW57d2lkdGg6MTVweDtoZWlnaHQ6MTVweDtib3JkZXI6MnB4IHNvbGlkIHZhcigtLWxpbmUpO2JvcmRlci10b3AtY29sb3I6"
    "dmFyKC0tc2lnbmFsKTsKICBib3JkZXItcmFkaXVzOjUwJTthbmltYXRpb246c3BpbiAuN3MgbGluZWFyIGluZmluaXRlO2ZsZXg6bm9uZX0KQGtleWZy"
    "YW1lcyBzcGlue3Rve3RyYW5zZm9ybTpyb3RhdGUoMzYwZGVnKX19CgovKiB0b2FzdCAqLwojdG9hc3Rze3Bvc2l0aW9uOmZpeGVkO2JvdHRvbToyMnB4"
    "O3JpZ2h0OjIycHg7ZGlzcGxheTpmbGV4O2ZsZXgtZGlyZWN0aW9uOmNvbHVtbjsKICBnYXA6MTBweDt6LWluZGV4OjIwMDttYXgtd2lkdGg6MzYwcHh9"
    "Ci50b2FzdHtiYWNrZ3JvdW5kOnZhcigtLXJhaXNlZCk7Ym9yZGVyOjFweCBzb2xpZCB2YXIoLS1saW5lKTtib3JkZXItbGVmdDozcHggc29saWQgdmFy"
    "KC0tc2lnbmFsKTsKICBib3JkZXItcmFkaXVzOnZhcigtLXItc20pO3BhZGRpbmc6MTFweCAxNHB4O2ZvbnQtc2l6ZToxM3B4O2JveC1zaGFkb3c6dmFy"
    "KC0tc2hhZG93KTsKICBhbmltYXRpb246c2xpZGVpbiAuMjJzIGVhc2V9Ci50b2FzdC5lcnJ7Ym9yZGVyLWxlZnQtY29sb3I6dmFyKC0tcmVkKX0gLnRv"
    "YXN0Lm9re2JvcmRlci1sZWZ0LWNvbG9yOnZhcigtLWxpdmUpfQpAa2V5ZnJhbWVzIHNsaWRlaW57ZnJvbXt0cmFuc2Zvcm06dHJhbnNsYXRlWCgyMHB4"
    "KTtvcGFjaXR5OjB9fQoKLyogbW9kYWwgKi8KLm1vZGFsLWJne3Bvc2l0aW9uOmZpeGVkO2luc2V0OjA7YmFja2dyb3VuZDpyZ2JhKDQsNywxMSwuNzIp"
    "O2JhY2tkcm9wLWZpbHRlcjpibHVyKDRweCk7CiAgei1pbmRleDoxNTA7ZGlzcGxheTpncmlkO3BsYWNlLWl0ZW1zOmNlbnRlcjtwYWRkaW5nOjIwcHg7"
    "YW5pbWF0aW9uOmZhZGUgLjE2c30KQGtleWZyYW1lcyBmYWRle2Zyb217b3BhY2l0eTowfX0KLm1vZGFse2JhY2tncm91bmQ6dmFyKC0tcGFuZWwpO2Jv"
    "cmRlcjoxcHggc29saWQgdmFyKC0tbGluZSk7Ym9yZGVyLXJhZGl1czp2YXIoLS1yLWxnKTsKICBtYXgtd2lkdGg6NTYwcHg7d2lkdGg6MTAwJTtib3gt"
    "c2hhZG93OnZhcigtLXNoYWRvdyk7bWF4LWhlaWdodDo4OHZoO292ZXJmbG93OmF1dG87CiAgYW5pbWF0aW9uOnBvcCAuMnMgZWFzZX0KLm1vZGFsLndp"
    "ZGV7bWF4LXdpZHRoOm1pbig5MnZ3LDkyMHB4KX0KQGtleWZyYW1lcyBwb3B7ZnJvbXt0cmFuc2Zvcm06c2NhbGUoLjk2KSB0cmFuc2xhdGVZKDhweCk7"
    "b3BhY2l0eTowfX0KLm1vZGFsIC5taHtwYWRkaW5nOjIwcHggMjJweDtib3JkZXItYm90dG9tOjFweCBzb2xpZCB2YXIoLS1saW5lKTsKICBkaXNwbGF5"
    "OmZsZXg7anVzdGlmeS1jb250ZW50OnNwYWNlLWJldHdlZW47YWxpZ24taXRlbXM6Y2VudGVyfQoubW9kYWwgLm1ie3BhZGRpbmc6MjJweH0KLm1vZGFs"
    "IC5tZntwYWRkaW5nOjE2cHggMjJweDtib3JkZXItdG9wOjFweCBzb2xpZCB2YXIoLS1saW5lKTtkaXNwbGF5OmZsZXg7CiAganVzdGlmeS1jb250ZW50"
    "OmZsZXgtZW5kO2dhcDoxMHB4fQoKQG1lZGlhIChtYXgtd2lkdGg6ODIwcHgpewogIC5hcHB7Z3JpZC10ZW1wbGF0ZS1jb2x1bW5zOjFmcn0KICAuc2lk"
    "ZXtwb3NpdGlvbjpmaXhlZDtsZWZ0OjA7dG9wOjA7Ym90dG9tOjA7d2lkdGg6MjMwcHg7dHJhbnNmb3JtOnRyYW5zbGF0ZVgoLTEwMCUpOwogICAgdHJh"
    "bnNpdGlvbjp0cmFuc2Zvcm0gLjIycztib3gtc2hhZG93OnZhcigtLXNoYWRvdyl9CiAgLnNpZGUub3Blbnt0cmFuc2Zvcm06bm9uZX0KICAubW9iaWxl"
    "YmFye2Rpc3BsYXk6ZmxleCFpbXBvcnRhbnR9CiAgLnZpZXd7cGFkZGluZzoxOHB4fQp9Ci5tb2JpbGViYXJ7ZGlzcGxheTpub25lO2FsaWduLWl0ZW1z"
    "OmNlbnRlcjtnYXA6MTJweDtwYWRkaW5nOjEycHggMTZweDsKICBib3JkZXItYm90dG9tOjFweCBzb2xpZCB2YXIoLS1saW5lKTtiYWNrZ3JvdW5kOnZh"
    "cigtLXBhbmVsLTIpfQoKLyogLS0tLS0tLS0tLSBkYXNoYm9hcmQgLS0tLS0tLS0tLSAqLwouaGVyby1wYW5lbHtwb3NpdGlvbjpyZWxhdGl2ZTtvdmVy"
    "ZmxvdzpoaWRkZW47Ym9yZGVyLXJhZGl1czp2YXIoLS1yLWxnKTsKICBib3JkZXI6MXB4IHNvbGlkIHZhcigtLWxpbmUpOwogIGJhY2tncm91bmQ6bGlu"
    "ZWFyLWdyYWRpZW50KDE1MGRlZyx2YXIoLS1yYWlzZWQpLHZhcigtLXBhbmVsLTIpIDcwJSk7cGFkZGluZzoyNnB4IDI4cHh9Ci5oZXJvLXBhbmVsOjph"
    "ZnRlcntjb250ZW50OiIiO3Bvc2l0aW9uOmFic29sdXRlO3JpZ2h0Oi02MHB4O3RvcDotNjBweDt3aWR0aDoyNDBweDtoZWlnaHQ6MjQwcHg7CiAgYmFj"
    "a2dyb3VuZDpyYWRpYWwtZ3JhZGllbnQoY2lyY2xlLHZhcigtLXNpZ25hbC1nbG93KSx0cmFuc3BhcmVudCA2MiUpO3BvaW50ZXItZXZlbnRzOm5vbmV9"
    "Ci5nYXVnZS1yb3d7ZGlzcGxheTpncmlkO2dyaWQtdGVtcGxhdGUtY29sdW1uczphdXRvIDFmcjtnYXA6MjRweDthbGlnbi1pdGVtczpjZW50ZXI7CiAg"
    "cG9zaXRpb246cmVsYXRpdmU7ei1pbmRleDoxfQouZ2F1Z2V7cG9zaXRpb246cmVsYXRpdmU7d2lkdGg6MTMycHg7aGVpZ2h0OjEzMnB4O2ZsZXg6bm9u"
    "ZX0KLmdhdWdlIHN2Z3t0cmFuc2Zvcm06cm90YXRlKC05MGRlZyl9Ci5nYXVnZSAubGJse3Bvc2l0aW9uOmFic29sdXRlO2luc2V0OjA7ZGlzcGxheTpn"
    "cmlkO3BsYWNlLWNvbnRlbnQ6Y2VudGVyO3RleHQtYWxpZ246Y2VudGVyfQouZ2F1Z2UgLmxibCBie2ZvbnQtZmFtaWx5OnZhcigtLWRpc3ApO2ZvbnQt"
    "c2l6ZToyNnB4O2Rpc3BsYXk6YmxvY2s7bGluZS1oZWlnaHQ6MX0KLmdhdWdlIC5sYmwgc3Bhbntmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNp"
    "emU6OS41cHg7Y29sb3I6dmFyKC0tbXV0ZWQpOwogIHRleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTtsZXR0ZXItc3BhY2luZzouMTJlbX0KLnNwZWNze2Rp"
    "c3BsYXk6Z3JpZDtncmlkLXRlbXBsYXRlLWNvbHVtbnM6cmVwZWF0KGF1dG8tZml0LG1pbm1heCgxMjBweCwxZnIpKTtnYXA6MTJweH0KLnNwZWN7YmFj"
    "a2dyb3VuZDp2YXIoLS1wYW5lbC0yKTtib3JkZXI6MXB4IHNvbGlkIHZhcigtLWxpbmUpO2JvcmRlci1yYWRpdXM6dmFyKC0tci1zbSk7CiAgcGFkZGlu"
    "ZzoxMXB4IDEzcHh9Ci5zcGVjIC5re2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMHB4O2NvbG9yOnZhcigtLWRpbSk7CiAgdGV4dC10"
    "cmFuc2Zvcm06dXBwZXJjYXNlO2xldHRlci1zcGFjaW5nOi4xZW19Ci5zcGVjIC52e2ZvbnQtZmFtaWx5OnZhcigtLWRpc3ApO2ZvbnQtc2l6ZToxNnB4"
    "O21hcmdpbi10b3A6M3B4fQouc3BlYyAudiBzbWFsbHtmb250LXNpemU6MTFweDtjb2xvcjp2YXIoLS1tdXRlZCk7Zm9udC1mYW1pbHk6dmFyKC0tc2Fu"
    "cyl9Ci50aWVyLW5vdGV7bWFyZ2luLXRvcDoxNnB4O3BhZGRpbmc6MTNweCAxNXB4O2JvcmRlci1yYWRpdXM6dmFyKC0tci1zbSk7CiAgYmFja2dyb3Vu"
    "ZDp2YXIoLS1zaWduYWwtZ2xvdyk7Ym9yZGVyOjFweCBzb2xpZCB2YXIoLS1zaWduYWwtZGltKTtmb250LXNpemU6MTNweDsKICBwb3NpdGlvbjpyZWxh"
    "dGl2ZTt6LWluZGV4OjF9CgoucmVjbGlzdHtkaXNwbGF5OmdyaWQ7Z2FwOjEwcHh9Ci5yZWNpdGVte2Rpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50"
    "ZXI7Z2FwOjE0cHg7cGFkZGluZzoxM3B4IDE1cHg7CiAgYm9yZGVyOjFweCBzb2xpZCB2YXIoLS1saW5lKTtib3JkZXItcmFkaXVzOnZhcigtLXIpO2Jh"
    "Y2tncm91bmQ6dmFyKC0tcGFuZWwpOwogIHRyYW5zaXRpb246LjE0c30KLnJlY2l0ZW06aG92ZXJ7Ym9yZGVyLWNvbG9yOnZhcigtLWRpbSl9Ci5yZWNp"
    "dGVtLmJlc3R7Ym9yZGVyLWNvbG9yOnZhcigtLXNpZ25hbC1kaW0pO2JhY2tncm91bmQ6bGluZWFyLWdyYWRpZW50KDkwZGVnLHZhcigtLXNpZ25hbC1n"
    "bG93KSx0cmFuc3BhcmVudCA1NSUpfQoucmVjaXRlbSAucmFua3tmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6MTJweDtjb2xvcjp2YXIo"
    "LS1kaW0pO3dpZHRoOjIycHg7ZmxleDpub25lfQoucmVjaXRlbSAuaW5mb3tmbGV4OjE7bWluLXdpZHRoOjB9Ci5yZWNpdGVtIC5pbmZvIC5ubXtmb250"
    "LXdlaWdodDo2MDA7Zm9udC1zaXplOjE0LjVweDtkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDo4cHg7ZmxleC13cmFwOndyYXB9Ci5y"
    "ZWNpdGVtIC5pbmZvIC5kc3tmb250LXNpemU6MTIuNXB4O2NvbG9yOnZhcigtLW11dGVkKTttYXJnaW4tdG9wOjJweH0KLnJlY2l0ZW0gLnNpemV7Zm9u"
    "dC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjExcHg7Y29sb3I6dmFyKC0tbXV0ZWQpO3RleHQtYWxpZ246cmlnaHQ7ZmxleDpub25lfQoKLyog"
    "LS0tLS0tLS0tLSBtb2RlbHMgcGFnZSAtLS0tLS0tLS0tICovCi5zZWFyY2hiYXJ7ZGlzcGxheTpmbGV4O2dhcDoxMHB4O21hcmdpbi1ib3R0b206MThw"
    "eH0KLnNlYXJjaGJhciAuaW5we2ZsZXg6MX0KLm1ncmlke2Rpc3BsYXk6Z3JpZDtncmlkLXRlbXBsYXRlLWNvbHVtbnM6cmVwZWF0KGF1dG8tZmlsbCxt"
    "aW5tYXgoMzIwcHgsMWZyKSk7Z2FwOjE0cHh9Ci5tY2FyZHtib3JkZXI6MXB4IHNvbGlkIHZhcigtLWxpbmUpO2JvcmRlci1yYWRpdXM6dmFyKC0tcik7"
    "YmFja2dyb3VuZDp2YXIoLS1wYW5lbCk7CiAgcGFkZGluZzoxNnB4O2Rpc3BsYXk6ZmxleDtmbGV4LWRpcmVjdGlvbjpjb2x1bW47Z2FwOjEwcHg7dHJh"
    "bnNpdGlvbjouMTRzfQoubWNhcmQ6aG92ZXJ7Ym9yZGVyLWNvbG9yOnZhcigtLWRpbSl9Ci5tY2FyZCAudG9we2Rpc3BsYXk6ZmxleDtqdXN0aWZ5LWNv"
    "bnRlbnQ6c3BhY2UtYmV0d2VlbjtnYXA6MTBweDthbGlnbi1pdGVtczpmbGV4LXN0YXJ0fQoubWNhcmQgLm5te2ZvbnQtd2VpZ2h0OjYwMDtmb250LXNp"
    "emU6MTVweH0KLm1jYXJkIC5pZHtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6MTFweDtjb2xvcjp2YXIoLS1kaW0pO3dvcmQtYnJlYWs6"
    "YnJlYWstYWxsfQoubWNhcmQgLmRze2ZvbnQtc2l6ZToxMi41cHg7Y29sb3I6dmFyKC0tbXV0ZWQpO2ZsZXg6MX0KLm1jYXJkIC50YWdze2Rpc3BsYXk6"
    "ZmxleDtnYXA6NXB4O2ZsZXgtd3JhcDp3cmFwfQoubWNhcmQgLmZvb3R7ZGlzcGxheTpmbGV4O2p1c3RpZnktY29udGVudDpzcGFjZS1iZXR3ZWVuO2Fs"
    "aWduLWl0ZW1zOmNlbnRlcjtnYXA6OHB4fQoucHVsbGJveHttYXJnaW4tdG9wOjhweH0KLnB1bGxib3ggLnN0YXR7Zm9udC1mYW1pbHk6dmFyKC0tbW9u"
    "byk7Zm9udC1zaXplOjExcHg7Y29sb3I6dmFyKC0tbXV0ZWQpOwogIG1hcmdpbi1ib3R0b206NXB4O2Rpc3BsYXk6ZmxleDtqdXN0aWZ5LWNvbnRlbnQ6"
    "c3BhY2UtYmV0d2Vlbn0KLmluc3RhbGxlZC1iYWRnZXtkaXNwbGF5OmlubGluZS1mbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6NXB4O2NvbG9yOnZh"
    "cigtLWxpdmUpOwogIGZvbnQtc2l6ZToxMnB4O2ZvbnQtd2VpZ2h0OjYwMH0KLnNlY3Rpb24tdGl0bGV7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9u"
    "dC1zaXplOjExcHg7bGV0dGVyLXNwYWNpbmc6LjE0ZW07CiAgdGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO2NvbG9yOnZhcigtLW11dGVkKTttYXJnaW46"
    "MjJweCAwIDEycHg7CiAgZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6MTBweH0KLnNlY3Rpb24tdGl0bGU6OmFmdGVye2NvbnRlbnQ6"
    "IiI7ZmxleDoxO2hlaWdodDoxcHg7YmFja2dyb3VuZDp2YXIoLS1saW5lLXNvZnQpfQoKLyogLS0tLS0tLS0tLSBjaGF0IC0tLS0tLS0tLS0gKi8KLmNo"
    "YXQtaGVhZHtkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDoxMnB4O3BhZGRpbmc6MTNweCAyMHB4OwogIGJvcmRlci1ib3R0b206MXB4"
    "IHNvbGlkIHZhcigtLWxpbmUpO2JhY2tncm91bmQ6dmFyKC0tcGFuZWwtMik7ZmxleC13cmFwOndyYXB9Ci5jaGF0LWhlYWQgLm1vZGVscGlja3tkaXNw"
    "bGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDo4cHh9Ci5jaGF0LWhlYWQgLnNlbHt3aWR0aDphdXRvO21pbi13aWR0aDoxNzBweDtwYWRkaW5n"
    "OjdweCAxMHB4O2ZvbnQtc2l6ZToxMi41cHh9Ci50b2dnbGV7ZGlzcGxheTppbmxpbmUtZmxleDthbGlnbi1pdGVtczpjZW50ZXI7Z2FwOjdweDtwYWRk"
    "aW5nOjZweCAxMXB4OwogIGJvcmRlcjoxcHggc29saWQgdmFyKC0tbGluZSk7Ym9yZGVyLXJhZGl1czoyMHB4O2ZvbnQtc2l6ZToxMnB4O2NvbG9yOnZh"
    "cigtLW11dGVkKTsKICB0cmFuc2l0aW9uOi4xNHM7dXNlci1zZWxlY3Q6bm9uZX0KLnRvZ2dsZTpob3Zlcntib3JkZXItY29sb3I6dmFyKC0tZGltKTtj"
    "b2xvcjp2YXIoLS10ZXh0KX0KLnRvZ2dsZS5vbntiYWNrZ3JvdW5kOnZhcigtLXNpZ25hbC1nbG93KTtib3JkZXItY29sb3I6dmFyKC0tc2lnbmFsLWRp"
    "bSk7Y29sb3I6dmFyKC0tc2lnbmFsKX0KLnRvZ2dsZS5vbi52aW97YmFja2dyb3VuZDpyZ2JhKDE2NywxMzksMjUwLC4xMik7Ym9yZGVyLWNvbG9yOiM0"
    "YzNmN2E7Y29sb3I6dmFyKC0tdmlvbGV0KX0KLnRvZ2dsZS5vbi5ibHV7YmFja2dyb3VuZDpyZ2JhKDk2LDE2NSwyNTAsLjEzKTtib3JkZXItY29sb3I6"
    "IzJjNGE3NTtjb2xvcjp2YXIoLS1ibHVlKX0KLnRvZ2dsZS5vbi5ncm57YmFja2dyb3VuZDpyZ2JhKDc0LDIyMiwxMjgsLjExKTtib3JkZXItY29sb3I6"
    "IzI2NWMzYTtjb2xvcjp2YXIoLS1saXZlKX0KLnRvZ2dsZS5vbi5yZWR7YmFja2dyb3VuZDpyZ2JhKDI0OCwxMTMsMTEzLC4xMyk7Ym9yZGVyLWNvbG9y"
    "OiM2YjJiMmI7Y29sb3I6dmFyKC0tcmVkKX0KLnRvZ2dsZSAuc3d7d2lkdGg6MjZweDtoZWlnaHQ6MTVweDtib3JkZXItcmFkaXVzOjIwcHg7YmFja2dy"
    "b3VuZDp2YXIoLS1saW5lKTsKICBwb3NpdGlvbjpyZWxhdGl2ZTt0cmFuc2l0aW9uOi4xNHM7ZmxleDpub25lfQoudG9nZ2xlIC5zdzo6YWZ0ZXJ7Y29u"
    "dGVudDoiIjtwb3NpdGlvbjphYnNvbHV0ZTt0b3A6MnB4O2xlZnQ6MnB4O3dpZHRoOjExcHg7aGVpZ2h0OjExcHg7CiAgYm9yZGVyLXJhZGl1czo1MCU7"
    "YmFja2dyb3VuZDp2YXIoLS1tdXRlZCk7dHJhbnNpdGlvbjouMTRzfQoudG9nZ2xlLm9uIC5zd3tiYWNrZ3JvdW5kOnZhcigtLXNpZ25hbCl9Ci50b2dn"
    "bGUub24gLnN3OjphZnRlcntsZWZ0OjEzcHg7YmFja2dyb3VuZDojMWExMjA0fQoudG9nZ2xlLm9uLnZpbyAuc3d7YmFja2dyb3VuZDp2YXIoLS12aW9s"
    "ZXQpfQoudG9nZ2xlLm9uLmJsdSAuc3d7YmFja2dyb3VuZDp2YXIoLS1ibHVlKX0KLnRvZ2dsZS5vbi5ncm4gLnN3e2JhY2tncm91bmQ6dmFyKC0tbGl2"
    "ZSl9Ci50b2dnbGUub24uZ3JuIC5zdzo6YWZ0ZXJ7YmFja2dyb3VuZDojMDQxNzBifQoudG9nZ2xlLm9uLnJlZCAuc3d7YmFja2dyb3VuZDp2YXIoLS1y"
    "ZWQpfQoKLyogY29tcHV0ZXIgY29udHJvbCAqLwouZGFuZ2VyLWNhcmR7Ym9yZGVyOjFweCBzb2xpZCAjNmIyYjJiO2JvcmRlci1yYWRpdXM6dmFyKC0t"
    "cik7CiAgYmFja2dyb3VuZDpsaW5lYXItZ3JhZGllbnQoMTgwZGVnLHJnYmEoMjQ4LDExMywxMTMsLjA2KSx2YXIoLS1wYW5lbCkpO3BhZGRpbmc6MjBw"
    "eCAyMnB4fQouZGFuZ2VyLWNhcmQgaDN7Y29sb3I6dmFyKC0tcmVkKX0KLndhcm5saXN0e21hcmdpbjoxMnB4IDA7cGFkZGluZzowO2xpc3Qtc3R5bGU6"
    "bm9uZTtkaXNwbGF5OmZsZXg7ZmxleC1kaXJlY3Rpb246Y29sdW1uO2dhcDo4cHh9Ci53YXJubGlzdCBsaXtkaXNwbGF5OmZsZXg7Z2FwOjlweDtmb250"
    "LXNpemU6MTNweDtjb2xvcjp2YXIoLS1tdXRlZCk7YWxpZ24taXRlbXM6ZmxleC1zdGFydH0KLndhcm5saXN0IGxpOjpiZWZvcmV7Y29udGVudDoi4pa4"
    "Ijtjb2xvcjp2YXIoLS1yZWQpO2ZsZXg6bm9uZX0KLmNvbnNlbnR7ZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmZsZXgtc3RhcnQ7Z2FwOjEwcHg7bWFy"
    "Z2luOjE0cHggMDtwYWRkaW5nOjEycHg7CiAgYm9yZGVyOjFweCBzb2xpZCB2YXIoLS1saW5lKTtib3JkZXItcmFkaXVzOnZhcigtLXItc20pO2JhY2tn"
    "cm91bmQ6dmFyKC0tcGFuZWwtMik7Y3Vyc29yOnBvaW50ZXJ9Ci5jb25zZW50IGlucHV0e21hcmdpbi10b3A6MnB4O3dpZHRoOjE2cHg7aGVpZ2h0OjE2"
    "cHg7ZmxleDpub25lO2FjY2VudC1jb2xvcjp2YXIoLS1yZWQpfQouZXN0b3B7d2lkdGg6MTAwJTtwYWRkaW5nOjE1cHg7Ym9yZGVyLXJhZGl1czp2YXIo"
    "LS1yKTtiYWNrZ3JvdW5kOnZhcigtLXJlZCk7Y29sb3I6I2ZmZjsKICBmb250LXdlaWdodDo4MDA7Zm9udC1zaXplOjE1cHg7bGV0dGVyLXNwYWNpbmc6"
    "LjAzZW07Ym9yZGVyOm5vbmU7ZGlzcGxheTpmbGV4OwogIGFsaWduLWl0ZW1zOmNlbnRlcjtqdXN0aWZ5LWNvbnRlbnQ6Y2VudGVyO2dhcDoxMHB4O2Jv"
    "eC1zaGFkb3c6MCA4cHggMjRweCAtOHB4IHZhcigtLXJlZCl9Ci5lc3RvcDpob3ZlcntmaWx0ZXI6YnJpZ2h0bmVzcygxLjA4KX0gLmVzdG9wOmFjdGl2"
    "ZXt0cmFuc2Zvcm06dHJhbnNsYXRlWSgxcHgpfQouZXN0b3Agc3Zne3dpZHRoOjE5cHg7aGVpZ2h0OjE5cHh9Ci5jYy1zdGF0dXN7ZGlzcGxheTpmbGV4"
    "O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6MTJweDtmbGV4LXdyYXA6d3JhcDttYXJnaW4tYm90dG9tOjE0cHh9Ci5jYy1waWxse2Rpc3BsYXk6aW5saW5l"
    "LWZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDo3cHg7cGFkZGluZzo2cHggMTJweDtib3JkZXItcmFkaXVzOjIwcHg7CiAgYm9yZGVyOjFweCBzb2xp"
    "ZCB2YXIoLS1saW5lKTtmb250LXNpemU6MTIuNXB4O2ZvbnQtd2VpZ2h0OjYwMH0KLmNjLXBpbGwubGl2ZXtib3JkZXItY29sb3I6IzI2NWMzYTtjb2xv"
    "cjp2YXIoLS1saXZlKTtiYWNrZ3JvdW5kOnJnYmEoNzQsMjIyLDEyOCwuMDgpfQouY2MtcGlsbC5vZmZ7Ym9yZGVyLWNvbG9yOnZhcigtLWxpbmUpO2Nv"
    "bG9yOnZhcigtLW11dGVkKX0KLmNjbG9ne2JvcmRlcjoxcHggc29saWQgdmFyKC0tbGluZSk7Ym9yZGVyLXJhZGl1czp2YXIoLS1yKTtiYWNrZ3JvdW5k"
    "OnZhcigtLXBhbmVsLTIpOwogIG1heC1oZWlnaHQ6MzQwcHg7b3ZlcmZsb3c6YXV0bztwYWRkaW5nOjhweH0KLmNjcm93e2Rpc3BsYXk6ZmxleDthbGln"
    "bi1pdGVtczpjZW50ZXI7Z2FwOjEwcHg7cGFkZGluZzo4cHggMTBweDtib3JkZXItcmFkaXVzOnZhcigtLXItc20pOwogIGZvbnQtc2l6ZToxMi41cHh9"
    "Ci5jY3Jvdzpob3ZlcntiYWNrZ3JvdW5kOnZhcigtLXBhbmVsKX0KLmNjcm93IC5jaXt3aWR0aDoyNHB4O2hlaWdodDoyNHB4O2JvcmRlci1yYWRpdXM6"
    "NnB4O2JhY2tncm91bmQ6dmFyKC0tcmFpc2VkKTtmbGV4Om5vbmU7CiAgZGlzcGxheTpncmlkO3BsYWNlLWl0ZW1zOmNlbnRlcjtmb250LXNpemU6MTJw"
    "eH0KLmNjcm93IC5jZHtmbGV4OjE7bWluLXdpZHRoOjB9IC5jY3JvdyAuY3R7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjEwcHg7Y29s"
    "b3I6dmFyKC0tZGltKX0KLmNjcm93IGltZ3toZWlnaHQ6MzRweDtib3JkZXItcmFkaXVzOjVweDtib3JkZXI6MXB4IHNvbGlkIHZhcigtLWxpbmUpO2N1"
    "cnNvcjp6b29tLWlufQoKLyogaW4tY2hhdCBjb21wdXRlciBhY3Rpb24gZmVlZCAqLwouY2NmZWVke2JvcmRlcjoxcHggc29saWQgIzZiMmIyYjtib3Jk"
    "ZXItcmFkaXVzOnZhcigtLXIpO2JhY2tncm91bmQ6dmFyKC0tcGFuZWwtMik7CiAgcGFkZGluZzoxMXB4IDEzcHg7bWFyZ2luOjRweCAwfQouY2NmZWVk"
    "IC5jZmh7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjExcHg7bGV0dGVyLXNwYWNpbmc6LjEyZW07dGV4dC10cmFuc2Zvcm06dXBwZXJj"
    "YXNlOwogIGNvbG9yOnZhcigtLXJlZCk7ZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6OHB4O21hcmdpbi1ib3R0b206OHB4fQouY2Nh"
    "Y3R7ZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6OXB4O3BhZGRpbmc6NXB4IDdweDtmb250LXNpemU6MTIuNXB4OwogIGJvcmRlci1y"
    "YWRpdXM6NnB4fQouY2NhY3QgLmFpY3t3aWR0aDoyMHB4O2hlaWdodDoyMHB4O2JvcmRlci1yYWRpdXM6NXB4O2JhY2tncm91bmQ6dmFyKC0tcmFpc2Vk"
    "KTtmbGV4Om5vbmU7CiAgZGlzcGxheTpncmlkO3BsYWNlLWl0ZW1zOmNlbnRlcjtmb250LXNpemU6MTFweH0KLmNjYWN0IGltZ3toZWlnaHQ6MzBweDti"
    "b3JkZXItcmFkaXVzOjRweDtib3JkZXI6MXB4IHNvbGlkIHZhcigtLWxpbmUpO2N1cnNvcjp6b29tLWluO21hcmdpbi1sZWZ0OmF1dG99Ci5jY2NvbmZp"
    "cm17Ym9yZGVyOjFweCBzb2xpZCB2YXIoLS1zaWduYWwtZGltKTtib3JkZXItcmFkaXVzOnZhcigtLXItc20pO2JhY2tncm91bmQ6dmFyKC0tc2lnbmFs"
    "LWdsb3cpOwogIHBhZGRpbmc6MTFweCAxM3B4O21hcmdpbjo4cHggMDtkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDoxMnB4O2ZsZXgt"
    "d3JhcDp3cmFwfQouY2Njb25maXJtIC5xe2ZsZXg6MTtmb250LXNpemU6MTNweDttaW4td2lkdGg6MTYwcHh9Ci5jY2NvbmZpcm0gLnEgYntjb2xvcjp2"
    "YXIoLS10ZXh0KX0KLm1pbmktZXN0b3B7ZGlzcGxheTppbmxpbmUtZmxleDthbGlnbi1pdGVtczpjZW50ZXI7Z2FwOjdweDtwYWRkaW5nOjdweCAxM3B4"
    "O2JvcmRlci1yYWRpdXM6OHB4OwogIGJhY2tncm91bmQ6dmFyKC0tcmVkKTtjb2xvcjojZmZmO2ZvbnQtd2VpZ2h0OjcwMDtmb250LXNpemU6MTIuNXB4"
    "O2JvcmRlcjpub25lfQoubWluaS1lc3RvcDpob3ZlcntmaWx0ZXI6YnJpZ2h0bmVzcygxLjA4KX0KCi8qIGNvdW5jaWwgcGFuZWwgKi8KLmNvdW5jaWx7"
    "Ym9yZGVyOjFweCBzb2xpZCB2YXIoLS1saW5lKTtib3JkZXItcmFkaXVzOnZhcigtLXIpO2JhY2tncm91bmQ6dmFyKC0tcGFuZWwtMik7CiAgcGFkZGlu"
    "ZzoxM3B4IDE0cHg7bWFyZ2luOjRweCAwfQouY291bmNpbCAuY2hlYWR7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjExcHg7bGV0dGVy"
    "LXNwYWNpbmc6LjEyZW07CiAgdGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO2NvbG9yOnZhcigtLWxpdmUpO2Rpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpj"
    "ZW50ZXI7Z2FwOjhweH0KLmNyb3VuZHtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6MTAuNXB4O2xldHRlci1zcGFjaW5nOi4xZW07dGV4"
    "dC10cmFuc2Zvcm06dXBwZXJjYXNlOwogIGNvbG9yOnZhcigtLW11dGVkKTttYXJnaW46MTNweCAwIDhweDtkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6"
    "Y2VudGVyO2dhcDo5cHh9Ci5jcm91bmQ6OmFmdGVye2NvbnRlbnQ6IiI7ZmxleDoxO2hlaWdodDoxcHg7YmFja2dyb3VuZDp2YXIoLS1saW5lLXNvZnQp"
    "fQouY2dyaWR7ZGlzcGxheTpncmlkO2dyaWQtdGVtcGxhdGUtY29sdW1uczpyZXBlYXQoYXV0by1maWxsLG1pbm1heCgyMTVweCwxZnIpKTtnYXA6OXB4"
    "fQouY2NhcmR7Ym9yZGVyOjFweCBzb2xpZCB2YXIoLS1saW5lKTtib3JkZXItcmFkaXVzOnZhcigtLXItc20pO2JhY2tncm91bmQ6dmFyKC0tcGFuZWwp"
    "OwogIHBhZGRpbmc6MTBweCAxMXB4O21pbi13aWR0aDowfQouY2NhcmQgLmNyb2xle2Rpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7Z2FwOjdw"
    "eDtmb250LWZhbWlseTp2YXIoLS1tb25vKTsKICBmb250LXNpemU6MTFweDtmb250LXdlaWdodDo2MDA7bWFyZ2luLWJvdHRvbToycHh9Ci5jY2FyZCAu"
    "Y2RvdHt3aWR0aDo3cHg7aGVpZ2h0OjdweDtib3JkZXItcmFkaXVzOjUwJTtmbGV4Om5vbmV9Ci5jY2FyZCAuY2ZvY3Vze2ZvbnQtc2l6ZToxMC41cHg7"
    "Y29sb3I6dmFyKC0tZGltKTttYXJnaW4tYm90dG9tOjZweDtsaW5lLWhlaWdodDoxLjM1fQouY2NhcmQgLmNzdHttYXJnaW4tbGVmdDphdXRvO2ZsZXg6"
    "bm9uZTtkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyfQouY2NhcmQgLmN0YWtle2ZvbnQtc2l6ZToxMnB4O2NvbG9yOnZhcigtLW11dGVkKTts"
    "aW5lLWhlaWdodDoxLjU7d2hpdGUtc3BhY2U6cHJlLXdyYXA7CiAgb3ZlcmZsb3ctd3JhcDphbnl3aGVyZTttYXgtaGVpZ2h0OjEwOHB4O292ZXJmbG93"
    "OmhpZGRlbjtwb3NpdGlvbjpyZWxhdGl2ZTtjdXJzb3I6cG9pbnRlcn0KLmNjYXJkIC5jdGFrZS5vcGVue21heC1oZWlnaHQ6bm9uZX0KLmNjYXJkIC5j"
    "dGFrZTpub3QoLm9wZW4pOjphZnRlcntjb250ZW50OiIiO3Bvc2l0aW9uOmFic29sdXRlO2xlZnQ6MDtyaWdodDowO2JvdHRvbTowO2hlaWdodDozNHB4"
    "OwogIGJhY2tncm91bmQ6bGluZWFyLWdyYWRpZW50KDBkZWcsdmFyKC0tcGFuZWwpLHRyYW5zcGFyZW50KX0KLmNjYXJkLmZhaWxlZHtvcGFjaXR5Oi41"
    "NX0KCi50aG91Z2h0e2JvcmRlci1sZWZ0OjJweCBzb2xpZCB2YXIoLS12aW9sZXQpO3BhZGRpbmc6N3B4IDExcHg7Zm9udC1zaXplOjEyLjVweDsKICBj"
    "b2xvcjp2YXIoLS1tdXRlZCk7YmFja2dyb3VuZDp2YXIoLS1wYW5lbC0yKTtib3JkZXItcmFkaXVzOjAgOHB4IDhweCAwOwogIHdoaXRlLXNwYWNlOnBy"
    "ZS13cmFwO292ZXJmbG93LXdyYXA6YW55d2hlcmV9Ci50aG91Z2h0IC50bHtkaXNwbGF5OmJsb2NrO2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQt"
    "c2l6ZTo5LjVweDtjb2xvcjp2YXIoLS12aW9sZXQpOwogIGxldHRlci1zcGFjaW5nOi4xZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO21hcmdpbi1i"
    "b3R0b206M3B4fQoKLyogbGl2ZSByZWFzb25pbmcgc3RyZWFtICh0aGlua2luZyBtb2RlbHMpICovCi50aGlua2luZy1ib3h7Ym9yZGVyOjFweCBzb2xp"
    "ZCB2YXIoLS1saW5lKTtib3JkZXItcmFkaXVzOnZhcigtLXItc20pO2JhY2tncm91bmQ6dmFyKC0tcGFuZWwtMik7CiAgbWFyZ2luOjRweCAwO292ZXJm"
    "bG93OmhpZGRlbn0KLnRoaW5raW5nLWJveCAudGh7ZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6OXB4O3BhZGRpbmc6OHB4IDEycHg7"
    "Y3Vyc29yOnBvaW50ZXI7CiAgZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjExLjVweDtjb2xvcjp2YXIoLS12aW9sZXQpO3VzZXItc2Vs"
    "ZWN0Om5vbmV9Ci50aGlua2luZy1ib3ggLnRoIC5sYmx7ZmxleDoxfQoudGhpbmtpbmctYm94IC50aCAuY3Z7dHJhbnNpdGlvbjp0cmFuc2Zvcm0gLjE4"
    "cztmb250LXNpemU6MTBweDtjb2xvcjp2YXIoLS1kaW0pfQoudGhpbmtpbmctYm94LmNvbGxhcHNlZCAudGggLmN2e3RyYW5zZm9ybTpyb3RhdGUoLTkw"
    "ZGVnKX0KLnRoaW5raW5nLWJveCAudGJ7cGFkZGluZzowIDEycHggMTFweDtmb250LXNpemU6MTJweDtjb2xvcjp2YXIoLS1tdXRlZCk7bGluZS1oZWln"
    "aHQ6MS41NTsKICB3aGl0ZS1zcGFjZTpwcmUtd3JhcDtvdmVyZmxvdy13cmFwOmFueXdoZXJlO21heC1oZWlnaHQ6MjAwcHg7b3ZlcmZsb3c6YXV0b30K"
    "LnRoaW5raW5nLWJveC5jb2xsYXBzZWQgLnRie2Rpc3BsYXk6bm9uZX0KLnRoaW5raW5nLWJveCAudGIuc3RyZWFtaW5ne2JvcmRlci1sZWZ0OjJweCBz"
    "b2xpZCB2YXIoLS12aW9sZXQpO3BhZGRpbmctbGVmdDoxMHB4O21hcmdpbi1sZWZ0OjJweH0KCi8qIGF1dG9tYXRpYyB3ZWItc2VhcmNoIGJveCAqLwou"
    "d2Vic2VhcmNoLWJveHtib3JkZXI6MXB4IHNvbGlkIHZhcigtLWxpbmUpO2JvcmRlci1yYWRpdXM6dmFyKC0tci1zbSk7YmFja2dyb3VuZDp2YXIoLS1w"
    "YW5lbC0yKTsKICBtYXJnaW46NHB4IDA7b3ZlcmZsb3c6aGlkZGVufQoud2Vic2VhcmNoLWJveCAud2h7ZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNl"
    "bnRlcjtnYXA6OXB4O3BhZGRpbmc6OHB4IDEycHg7Y3Vyc29yOnBvaW50ZXI7CiAgZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjExLjVw"
    "eDtjb2xvcjp2YXIoLS1ibHVlKTt1c2VyLXNlbGVjdDpub25lfQoud2Vic2VhcmNoLWJveCAud2ggLndxe2ZsZXg6MTtvdmVyZmxvdzpoaWRkZW47dGV4"
    "dC1vdmVyZmxvdzplbGxpcHNpczt3aGl0ZS1zcGFjZTpub3dyYXA7CiAgY29sb3I6dmFyKC0tdGV4dCl9Ci53ZWJzZWFyY2gtYm94IC53aCAuY3Z7dHJh"
    "bnNpdGlvbjp0cmFuc2Zvcm0gLjE4cztmb250LXNpemU6MTBweDtjb2xvcjp2YXIoLS1kaW0pfQoud2Vic2VhcmNoLWJveC5jb2xsYXBzZWQgLndoIC5j"
    "dnt0cmFuc2Zvcm06cm90YXRlKC05MGRlZyl9Ci53ZWJzZWFyY2gtYm94IC53YntwYWRkaW5nOjAgMTJweCAxMXB4O2ZvbnQtc2l6ZToxMnB4O2NvbG9y"
    "OnZhcigtLW11dGVkKTtsaW5lLWhlaWdodDoxLjU7CiAgd2hpdGUtc3BhY2U6cHJlLXdyYXA7b3ZlcmZsb3ctd3JhcDphbnl3aGVyZTttYXgtaGVpZ2h0"
    "OjIyMHB4O292ZXJmbG93OmF1dG99Ci53ZWJzZWFyY2gtYm94LmNvbGxhcHNlZCAud2J7ZGlzcGxheTpub25lfQoKLmNoYXRzY3JvbGx7ZmxleDoxO292"
    "ZXJmbG93LXk6YXV0bztwYWRkaW5nOjI0cHggMH0KLmNoYXR3cmFwe21heC13aWR0aDo4MjBweDttYXJnaW46MCBhdXRvO3BhZGRpbmc6MCAyNHB4O2Rp"
    "c3BsYXk6ZmxleDsKICBmbGV4LWRpcmVjdGlvbjpjb2x1bW47Z2FwOjIwcHh9Ci5tc2d7ZGlzcGxheTpmbGV4O2dhcDoxM3B4O2FuaW1hdGlvbjptc2dp"
    "biAuMjVzIGVhc2V9CkBrZXlmcmFtZXMgbXNnaW57ZnJvbXtvcGFjaXR5OjA7dHJhbnNmb3JtOnRyYW5zbGF0ZVkoNnB4KX19Ci5tc2cgLmF2e3dpZHRo"
    "OjMwcHg7aGVpZ2h0OjMwcHg7Ym9yZGVyLXJhZGl1czo5cHg7ZmxleDpub25lO2Rpc3BsYXk6Z3JpZDsKICBwbGFjZS1pdGVtczpjZW50ZXI7Zm9udC1m"
    "YW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjExcHg7Zm9udC13ZWlnaHQ6NzAwOwogIGJvcmRlcjoxcHggc29saWQgdmFyKC0tbGluZSl9Ci5tc2cu"
    "dXNlciAuYXZ7YmFja2dyb3VuZDp2YXIoLS1yYWlzZWQpO2NvbG9yOnZhcigtLW11dGVkKX0KLm1zZy5haSAuYXZ7YmFja2dyb3VuZDpyYWRpYWwtZ3Jh"
    "ZGllbnQoMTIwJSAxMjAlIGF0IDMwJSAyNSUsIzI0MzI0NCwjMGQxMzFiKTsKICBjb2xvcjp2YXIoLS1zaWduYWwpfQoubXNnIC5ib2R5e2ZsZXg6MTtt"
    "aW4td2lkdGg6MDtwYWRkaW5nLXRvcDozcHh9Ci5tc2cgLndob3tmb250LXNpemU6MTFweDtjb2xvcjp2YXIoLS1kaW0pO2ZvbnQtZmFtaWx5OnZhcigt"
    "LW1vbm8pO21hcmdpbi1ib3R0b206NHB4OwogIHRleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTtsZXR0ZXItc3BhY2luZzouMDhlbX0KLmJ1YmJsZXtmb250"
    "LXNpemU6MTQuNXB4O2xpbmUtaGVpZ2h0OjEuNjI7d29yZC13cmFwOmJyZWFrLXdvcmQ7b3ZlcmZsb3ctd3JhcDphbnl3aGVyZX0KLmJ1YmJsZSBwe21h"
    "cmdpbjowIDAgMTBweH0gLmJ1YmJsZSBwOmxhc3QtY2hpbGR7bWFyZ2luLWJvdHRvbTowfQouYnViYmxlIHByZXtiYWNrZ3JvdW5kOnZhcigtLXBhbmVs"
    "LTIpO2JvcmRlcjoxcHggc29saWQgdmFyKC0tbGluZSk7Ym9yZGVyLXJhZGl1czp2YXIoLS1yLXNtKTsKICBwYWRkaW5nOjEzcHggMTVweDtvdmVyZmxv"
    "dy14OmF1dG87bWFyZ2luOjExcHggMDtmb250LXNpemU6MTIuNXB4O2xpbmUtaGVpZ2h0OjEuNX0KLmJ1YmJsZSBjb2Rle2JhY2tncm91bmQ6dmFyKC0t"
    "cmFpc2VkKTtwYWRkaW5nOjEuNXB4IDZweDtib3JkZXItcmFkaXVzOjVweDtmb250LXNpemU6MTIuNXB4fQouYnViYmxlIHByZSBjb2Rle2JhY2tncm91"
    "bmQ6bm9uZTtwYWRkaW5nOjB9Ci5idWJibGUgdWwsLmJ1YmJsZSBvbHttYXJnaW46OHB4IDA7cGFkZGluZy1sZWZ0OjIycHh9IC5idWJibGUgbGl7bWFy"
    "Z2luOjNweCAwfQouYnViYmxlIGgxLC5idWJibGUgaDIsLmJ1YmJsZSBoM3ttYXJnaW46MTZweCAwIDhweH0KLmJ1YmJsZSBpbWd7bWF4LXdpZHRoOjEw"
    "MCU7Ym9yZGVyLXJhZGl1czp2YXIoLS1yLXNtKTtib3JkZXI6MXB4IHNvbGlkIHZhcigtLWxpbmUpO21hcmdpbjo4cHggMH0KLmNvZGV3cmFwe21hcmdp"
    "bjoxMXB4IDB9Ci5jb2Rld3JhcCAuY29kZWJhcntkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDo4cHg7YmFja2dyb3VuZDp2YXIoLS1y"
    "YWlzZWQpOwogIGJvcmRlcjoxcHggc29saWQgdmFyKC0tbGluZSk7Ym9yZGVyLWJvdHRvbTpub25lO2JvcmRlci1yYWRpdXM6dmFyKC0tci1zbSkgdmFy"
    "KC0tci1zbSkgMCAwOwogIHBhZGRpbmc6NXB4IDEwcHg7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjExcHg7Y29sb3I6dmFyKC0tbXV0"
    "ZWQpfQouY29kZXdyYXAgcHJle21hcmdpbjowO2JvcmRlci10b3AtbGVmdC1yYWRpdXM6MDtib3JkZXItdG9wLXJpZ2h0LXJhZGl1czowfQouY29kZWxh"
    "bmd7bGV0dGVyLXNwYWNpbmc6LjRweDt0ZXh0LXRyYW5zZm9ybTpsb3dlcmNhc2V9Ci5jYnRue2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6"
    "ZToxMXB4O2JhY2tncm91bmQ6bm9uZTtib3JkZXI6MXB4IHNvbGlkIHZhcigtLWxpbmUpOwogIGNvbG9yOnZhcigtLW11dGVkKTtib3JkZXItcmFkaXVz"
    "OjZweDtwYWRkaW5nOjIuNXB4IDlweDtjdXJzb3I6cG9pbnRlcn0KLmNidG46aG92ZXJ7Y29sb3I6dmFyKC0tdGV4dCk7Ym9yZGVyLWNvbG9yOnZhcigt"
    "LWRpbSl9Ci5jYnRuLm9re2NvbG9yOnZhcigtLWxpdmUpO2JvcmRlci1jb2xvcjp2YXIoLS1saXZlLWRpbSl9Ci5jYnRuOmRpc2FibGVke29wYWNpdHk6"
    "LjU7Y3Vyc29yOndhaXR9Ci5mbGFzaHthbmltYXRpb246Y2FyZGZsYXNoIDIuMnMgZWFzZS1vdXR9CkBrZXlmcmFtZXMgY2FyZGZsYXNoezAlLDYwJXti"
    "b3gtc2hhZG93OjAgMCAwIDJweCB2YXIoLS1zaWduYWwpLCB2YXIoLS1zaGFkb3cpfTEwMCV7Ym94LXNoYWRvdzpub25lfX0KLmF0dGFjaHJvd3tkaXNw"
    "bGF5OmZsZXg7Z2FwOjhweDtwYWRkaW5nOjAgNHB4IDhweDtmbGV4LXdyYXA6d3JhcH0KLmF0dGFjaGNoaXB7cG9zaXRpb246cmVsYXRpdmV9Ci5hdHRh"
    "Y2hjaGlwIGltZ3toZWlnaHQ6NTJweDtib3JkZXItcmFkaXVzOjhweDtib3JkZXI6MXB4IHNvbGlkIHZhcigtLWxpbmUpO2Rpc3BsYXk6YmxvY2t9Ci5h"
    "dHRhY2hjaGlwIGJ7cG9zaXRpb246YWJzb2x1dGU7dG9wOi03cHg7cmlnaHQ6LTdweDt3aWR0aDoxOHB4O2hlaWdodDoxOHB4O2xpbmUtaGVpZ2h0OjE2"
    "cHg7CiAgdGV4dC1hbGlnbjpjZW50ZXI7YmFja2dyb3VuZDp2YXIoLS1yYWlzZWQpO2JvcmRlcjoxcHggc29saWQgdmFyKC0tbGluZSk7Ym9yZGVyLXJh"
    "ZGl1czo1MCU7CiAgY3Vyc29yOnBvaW50ZXI7Zm9udC1zaXplOjEycHg7Y29sb3I6dmFyKC0tbXV0ZWQpfQouYXR0YWNoY2hpcCBiOmhvdmVye2NvbG9y"
    "OnZhcigtLXJlZCk7Ym9yZGVyLWNvbG9yOnZhcigtLXJlZCl9Ci5hdHRhY2hidG57YmFja2dyb3VuZDpub25lO2JvcmRlcjpub25lO2NvbG9yOnZhcigt"
    "LWRpbSk7Y3Vyc29yOnBvaW50ZXI7cGFkZGluZzo4cHggNHB4IDhweCA4cHg7ZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcn0KLmF0dGFjaGJ0"
    "bjpob3Zlcntjb2xvcjp2YXIoLS10ZXh0KX0KLm1zZ2ltZ3N7ZGlzcGxheTpmbGV4O2dhcDo4cHg7ZmxleC13cmFwOndyYXA7bWFyZ2luOjAgMCA4cHh9"
    "Ci5tc2dpbWdzIGltZ3ttYXgtaGVpZ2h0OjE0MHB4O21heC13aWR0aDoyMjBweDtib3JkZXItcmFkaXVzOjEwcHg7Ym9yZGVyOjFweCBzb2xpZCB2YXIo"
    "LS1saW5lKX0KLnJlZ2Vucm93e21hcmdpbi10b3A6N3B4fQouc3RhdGxpbmV7Zm9udC1zaXplOjExcHg7Y29sb3I6dmFyKC0tZGltKTtmb250LWZhbWls"
    "eTp2YXIoLS1tb25vKTttYXJnaW4tdG9wOjdweDsKICBkaXNwbGF5OmZsZXg7Z2FwOjZweDthbGlnbi1pdGVtczpjZW50ZXI7dXNlci1zZWxlY3Q6bm9u"
    "ZX0KLnBhdGhva3tjb2xvcjp2YXIoLS1saXZlKTtmb250LXNpemU6MTJweH0ucGF0aGJhZHtjb2xvcjp2YXIoLS1yZWQpO2ZvbnQtc2l6ZToxMnB4fQou"
    "YnViYmxlIGF7Ym9yZGVyLWJvdHRvbToxcHggc29saWQgdmFyKC0tc2lnbmFsLWRpbSl9Ci5idWJibGUgdGFibGV7Ym9yZGVyLWNvbGxhcHNlOmNvbGxh"
    "cHNlO21hcmdpbjoxMHB4IDA7Zm9udC1zaXplOjEyLjVweDt3aWR0aDoxMDAlfQouYnViYmxlIHRoLC5idWJibGUgdGR7Ym9yZGVyOjFweCBzb2xpZCB2"
    "YXIoLS1saW5lKTtwYWRkaW5nOjZweCAxMHB4O3RleHQtYWxpZ246bGVmdH0KLmN1cnNvci1ibGluazo6YWZ0ZXJ7Y29udGVudDoi4paLIjtjb2xvcjp2"
    "YXIoLS1zaWduYWwpO2FuaW1hdGlvbjpibGluayAxcyBzdGVwLWVuZCBpbmZpbml0ZTsKICBtYXJnaW4tbGVmdDoxcHh9CkBrZXlmcmFtZXMgYmxpbmt7"
    "NTAle29wYWNpdHk6MH19CgouYWdlbnRsb2d7bWFyZ2luOjZweCAwIDRweDtkaXNwbGF5OmZsZXg7ZmxleC1kaXJlY3Rpb246Y29sdW1uO2dhcDo2cHh9"
    "Ci5zdGVwe2JvcmRlcjoxcHggc29saWQgdmFyKC0tbGluZSk7Ym9yZGVyLXJhZGl1czp2YXIoLS1yLXNtKTtiYWNrZ3JvdW5kOnZhcigtLXBhbmVsLTIp"
    "OwogIG92ZXJmbG93OmhpZGRlbjtmb250LXNpemU6MTIuNXB4fQouc3RlcCAuc2h7ZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6OXB4"
    "O3BhZGRpbmc6OHB4IDEycHg7Y3Vyc29yOnBvaW50ZXJ9Ci5zdGVwIC5zaCAudG57Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjExLjVw"
    "eDtjb2xvcjp2YXIoLS12aW9sZXQpO2ZvbnQtd2VpZ2h0OjYwMH0KLnN0ZXAgLnNoIC5hcmd7Y29sb3I6dmFyKC0tbXV0ZWQpO2ZvbnQtZmFtaWx5OnZh"
    "cigtLW1vbm8pO2ZvbnQtc2l6ZToxMXB4OwogIG92ZXJmbG93OmhpZGRlbjt0ZXh0LW92ZXJmbG93OmVsbGlwc2lzO3doaXRlLXNwYWNlOm5vd3JhcDtm"
    "bGV4OjE7bWluLXdpZHRoOjB9Ci5zdGVwIC5zYntib3JkZXItdG9wOjFweCBzb2xpZCB2YXIoLS1saW5lKTtwYWRkaW5nOjEwcHggMTJweDtmb250LWZh"
    "bWlseTp2YXIoLS1tb25vKTsKICBmb250LXNpemU6MTEuNXB4O3doaXRlLXNwYWNlOnByZS13cmFwO2NvbG9yOnZhcigtLW11dGVkKTttYXgtaGVpZ2h0"
    "OjIzMHB4O292ZXJmbG93OmF1dG87CiAgYmFja2dyb3VuZDp2YXIoLS1pbmspfQouc3RhdHVzbGluZXtkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2Vu"
    "dGVyO2dhcDo5cHg7Zm9udC1zaXplOjEyLjVweDtjb2xvcjp2YXIoLS1tdXRlZCk7CiAgZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7cGFkZGluZzoycHgg"
    "MH0KLnNyY2JveHtib3JkZXI6MXB4IGRhc2hlZCB2YXIoLS1saW5lKTtib3JkZXItcmFkaXVzOnZhcigtLXItc20pO3BhZGRpbmc6MTBweCAxM3B4Owog"
    "IG1hcmdpbi1ib3R0b206NHB4O2JhY2tncm91bmQ6dmFyKC0tcGFuZWwtMil9Ci5zcmNib3ggLnN0e2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQt"
    "c2l6ZToxMC41cHg7dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlOwogIGxldHRlci1zcGFjaW5nOi4xZW07Y29sb3I6dmFyKC0tc2lnbmFsKTttYXJnaW4t"
    "Ym90dG9tOjdweH0KLnNyY2l0ZW17Zm9udC1zaXplOjEycHg7Y29sb3I6dmFyKC0tbXV0ZWQpO3BhZGRpbmc6NHB4IDA7Ym9yZGVyLXRvcDoxcHggc29s"
    "aWQgdmFyKC0tbGluZS1zb2Z0KX0KLnNyY2l0ZW06Zmlyc3Qtb2YtdHlwZXtib3JkZXItdG9wOm5vbmV9Ci5zcmNpdGVtIGJ7Y29sb3I6dmFyKC0tdGV4"
    "dCl9CgouY29tcG9zZXJ7Ym9yZGVyLXRvcDoxcHggc29saWQgdmFyKC0tbGluZSk7YmFja2dyb3VuZDp2YXIoLS1wYW5lbC0yKTtwYWRkaW5nOjE0cHgg"
    "MjRweCAxOHB4fQouY29tcG9zZXIgLmNib3h7bWF4LXdpZHRoOjgyMHB4O21hcmdpbjowIGF1dG87cG9zaXRpb246cmVsYXRpdmU7CiAgYm9yZGVyOjFw"
    "eCBzb2xpZCB2YXIoLS1saW5lKTtib3JkZXItcmFkaXVzOnZhcigtLXIpO2JhY2tncm91bmQ6dmFyKC0tcGFuZWwpOwogIHRyYW5zaXRpb246Ym9yZGVy"
    "IC4xNHMsYm94LXNoYWRvdyAuMTRzO2Rpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpmbGV4LWVuZDtnYXA6OHB4O3BhZGRpbmc6OHB4IDhweCA4cHggMTRw"
    "eH0KLmNvbXBvc2VyIC5jYm94OmZvY3VzLXdpdGhpbntib3JkZXItY29sb3I6dmFyKC0tc2lnbmFsKTtib3gtc2hhZG93OjAgMCAwIDNweCB2YXIoLS1z"
    "aWduYWwtZ2xvdyl9Ci5jb21wb3NlciB0ZXh0YXJlYXtmbGV4OjE7YmFja2dyb3VuZDpub25lO2JvcmRlcjpub25lO291dGxpbmU6bm9uZTtyZXNpemU6"
    "bm9uZTsKICBmb250LXNpemU6MTQuNXB4O2xpbmUtaGVpZ2h0OjEuNTU7bWF4LWhlaWdodDoyMDBweDtwYWRkaW5nOjhweCAwO2NvbG9yOnZhcigtLXRl"
    "eHQpfQouY29tcG9zZXIgdGV4dGFyZWE6OnBsYWNlaG9sZGVye2NvbG9yOnZhcigtLWRpbSl9Ci5zZW5kYnRue3dpZHRoOjM4cHg7aGVpZ2h0OjM4cHg7"
    "Ym9yZGVyLXJhZGl1czoxMHB4O2JhY2tncm91bmQ6dmFyKC0tc2lnbmFsKTtjb2xvcjojMWExMjA0OwogIGRpc3BsYXk6Z3JpZDtwbGFjZS1pdGVtczpj"
    "ZW50ZXI7ZmxleDpub25lO3RyYW5zaXRpb246LjE0czttYXJnaW4tYm90dG9tOjFweH0KLnNlbmRidG46aG92ZXJ7ZmlsdGVyOmJyaWdodG5lc3MoMS4w"
    "Nyl9IC5zZW5kYnRuOmRpc2FibGVke29wYWNpdHk6LjQ7Y3Vyc29yOm5vdC1hbGxvd2VkfQouc2VuZGJ0bi5zdG9we2JhY2tncm91bmQ6dmFyKC0tcmVk"
    "KTtjb2xvcjojZmZmfQouc2VuZGJ0bi5zdG9wOmhvdmVye2ZpbHRlcjpicmlnaHRuZXNzKDEuMSl9Ci5jaGF0LXNpZGV7d2lkdGg6MjUwcHg7Ym9yZGVy"
    "LWxlZnQ6MXB4IHNvbGlkIHZhcigtLWxpbmUpO2JhY2tncm91bmQ6dmFyKC0tcGFuZWwtMik7CiAgZGlzcGxheTpmbGV4O2ZsZXgtZGlyZWN0aW9uOmNv"
    "bHVtbjtmbGV4Om5vbmV9Ci5jaGF0LWxheW91dHtmbGV4OjE7ZGlzcGxheTpmbGV4O21pbi1oZWlnaHQ6MH0KLmNoYXQtbWFpbntmbGV4OjE7ZGlzcGxh"
    "eTpmbGV4O2ZsZXgtZGlyZWN0aW9uOmNvbHVtbjttaW4td2lkdGg6MH0KLmNvbnZsaXN0e2ZsZXg6MTtvdmVyZmxvdy15OmF1dG87cGFkZGluZzoxMHB4"
    "fQouY29udml0ZW17cGFkZGluZzoxMHB4IDEycHg7Ym9yZGVyLXJhZGl1czp2YXIoLS1yLXNtKTtjdXJzb3I6cG9pbnRlcjttYXJnaW4tYm90dG9tOjNw"
    "eDsKICBmb250LXNpemU6MTNweDtjb2xvcjp2YXIoLS1tdXRlZCk7ZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6OHB4O3RyYW5zaXRp"
    "b246LjEyczsKICBwb3NpdGlvbjpyZWxhdGl2ZX0KLmNvbnZpdGVtOmhvdmVye2JhY2tncm91bmQ6dmFyKC0tcGFuZWwpO2NvbG9yOnZhcigtLXRleHQp"
    "fQouY29udml0ZW0uYWN0aXZle2JhY2tncm91bmQ6dmFyKC0tcmFpc2VkKTtjb2xvcjp2YXIoLS10ZXh0KX0KLmNvbnZpdGVtIC50dHtmbGV4OjE7b3Zl"
    "cmZsb3c6aGlkZGVuO3RleHQtb3ZlcmZsb3c6ZWxsaXBzaXM7d2hpdGUtc3BhY2U6bm93cmFwfQouY29udml0ZW0gLmRlbHtvcGFjaXR5OjA7Y29sb3I6"
    "dmFyKC0tZGltKTtwYWRkaW5nOjJweH0KLmNvbnZpdGVtOmhvdmVyIC5kZWx7b3BhY2l0eToxfSAuY29udml0ZW0gLmRlbDpob3Zlcntjb2xvcjp2YXIo"
    "LS1yZWQpfQoubmV3Y2hhdHttYXJnaW46MTBweDtkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2p1c3RpZnktY29udGVudDpjZW50ZXI7Z2Fw"
    "OjhweH0KQG1lZGlhIChtYXgtd2lkdGg6MTAwMHB4KXsuY2hhdC1zaWRle2Rpc3BsYXk6bm9uZX19Ci5jaGF0LWVtcHR5e21hcmdpbjo5dmggYXV0byAw"
    "O3RleHQtYWxpZ246Y2VudGVyO3BhZGRpbmc6MjBweDttYXgtd2lkdGg6NDYwcHg7CiAgZGlzcGxheTpmbGV4O2ZsZXgtZGlyZWN0aW9uOmNvbHVtbjth"
    "bGlnbi1pdGVtczpjZW50ZXI7Z2FwOjlweH0KLmNoYXQtZW1wdHkgLm9yYnt3aWR0aDo1MnB4O2hlaWdodDo1MnB4O2JvcmRlci1yYWRpdXM6MTZweDsK"
    "ICBiYWNrZ3JvdW5kOnJhZGlhbC1ncmFkaWVudCgxMjAlIDEyMCUgYXQgMzAlIDI1JSwjMjQzMjQ0LCMwZDEzMWIpOwogIGJvcmRlcjoxcHggc29saWQg"
    "dmFyKC0tbGluZSk7ZGlzcGxheTpncmlkO3BsYWNlLWl0ZW1zOmNlbnRlcjttYXJnaW4tYm90dG9tOjVweH0KLmNoYXQtZW1wdHkgLm9yYjo6YmVmb3Jl"
    "e2NvbnRlbnQ6IiI7d2lkdGg6MTZweDtoZWlnaHQ6MTZweDtib3JkZXItcmFkaXVzOjUwJTsKICBiYWNrZ3JvdW5kOnZhcigtLXNpZ25hbCk7Ym94LXNo"
    "YWRvdzowIDAgMCA2cHggdmFyKC0tc2lnbmFsLWdsb3cpLAogIDAgMCAyNnB4IDRweCB2YXIoLS1zaWduYWwtZ2xvdyk7YW5pbWF0aW9uOmNvcmVwdWxz"
    "ZSAzLjRzIGVhc2UtaW4tb3V0IGluZmluaXRlfQouY2hhdC1lbXB0eSBoM3tmb250LXNpemU6MTlweH0KLmNoYXQtZW1wdHkgcHttYXJnaW46MH0KLmNo"
    "YXQtZW1wdHkgLmV4cm93e2Rpc3BsYXk6ZmxleDtmbGV4LWRpcmVjdGlvbjpjb2x1bW47Z2FwOjhweDt3aWR0aDoxMDAlO21hcmdpbi10b3A6NnB4fQpA"
    "bWVkaWEgKHByZWZlcnMtcmVkdWNlZC1tb3Rpb246cmVkdWNlKXsKICAqLCo6OmJlZm9yZSwqOjphZnRlcnthbmltYXRpb24tZHVyYXRpb246LjAxbXMh"
    "aW1wb3J0YW50O3RyYW5zaXRpb24tZHVyYXRpb246LjAxbXMhaW1wb3J0YW50fX0KCi8qIC0tLS0tLS0tLS0gaW1hZ2VzIC0tLS0tLS0tLS0gKi8KLmlt"
    "Z2xheW91dHtkaXNwbGF5OmdyaWQ7Z3JpZC10ZW1wbGF0ZS1jb2x1bW5zOjMzMHB4IDFmcjtnYXA6MjJweDthbGlnbi1pdGVtczpzdGFydH0KQG1lZGlh"
    "IChtYXgtd2lkdGg6OTAwcHgpey5pbWdsYXlvdXR7Z3JpZC10ZW1wbGF0ZS1jb2x1bW5zOjFmcn19Ci5nZW5wYW5lbHtwb3NpdGlvbjpzdGlja3k7dG9w"
    "OjB9Ci5pbWdwcmV2aWV3e2FzcGVjdC1yYXRpbzoxO2JvcmRlcjoxcHggc29saWQgdmFyKC0tbGluZSk7Ym9yZGVyLXJhZGl1czp2YXIoLS1yKTsKICBi"
    "YWNrZ3JvdW5kOnZhcigtLXBhbmVsLTIpO2Rpc3BsYXk6Z3JpZDtwbGFjZS1pdGVtczpjZW50ZXI7b3ZlcmZsb3c6aGlkZGVuO21hcmdpbi1ib3R0b206"
    "MTRweDsKICBwb3NpdGlvbjpyZWxhdGl2ZX0KLmltZ3ByZXZpZXcgaW1ne3dpZHRoOjEwMCU7aGVpZ2h0OjEwMCU7b2JqZWN0LWZpdDpjb250YWluO2N1"
    "cnNvcjp6b29tLWlufQouaW1ncHJldmlldyAucGh7dGV4dC1hbGlnbjpjZW50ZXI7Y29sb3I6dmFyKC0tZGltKTtwYWRkaW5nOjIwcHh9Ci5nYWxsZXJ5"
    "e2Rpc3BsYXk6Z3JpZDtncmlkLXRlbXBsYXRlLWNvbHVtbnM6cmVwZWF0KGF1dG8tZmlsbCxtaW5tYXgoMTUwcHgsMWZyKSk7Z2FwOjEycHh9Ci5naXRl"
    "bXtib3JkZXI6MXB4IHNvbGlkIHZhcigtLWxpbmUpO2JvcmRlci1yYWRpdXM6dmFyKC0tci1zbSk7b3ZlcmZsb3c6aGlkZGVuOwogIGJhY2tncm91bmQ6"
    "dmFyKC0tcGFuZWwpO3Bvc2l0aW9uOnJlbGF0aXZlO2FzcGVjdC1yYXRpbzoxO2N1cnNvcjpwb2ludGVyO3RyYW5zaXRpb246LjE0c30KLmdpdGVtOmhv"
    "dmVye2JvcmRlci1jb2xvcjp2YXIoLS1kaW0pO3RyYW5zZm9ybTp0cmFuc2xhdGVZKC0ycHgpfQouZ2l0ZW0gaW1ne3dpZHRoOjEwMCU7aGVpZ2h0OjEw"
    "MCU7b2JqZWN0LWZpdDpjb3Zlcn0KLmdpdGVtIC5vdntwb3NpdGlvbjphYnNvbHV0ZTtpbnNldDowO2JhY2tncm91bmQ6bGluZWFyLWdyYWRpZW50KDBk"
    "ZWcscmdiYSg0LDcsMTEsLjkpLHRyYW5zcGFyZW50IDQ1JSk7CiAgb3BhY2l0eTowO3RyYW5zaXRpb246LjE0cztkaXNwbGF5OmZsZXg7ZmxleC1kaXJl"
    "Y3Rpb246Y29sdW1uO2p1c3RpZnktY29udGVudDpmbGV4LWVuZDtwYWRkaW5nOjEwcHg7CiAgcG9pbnRlci1ldmVudHM6bm9uZX0KLmdpdGVtOmhvdmVy"
    "IC5vdntvcGFjaXR5OjF9Ci5naXRlbSAub3YgLnBye2ZvbnQtc2l6ZToxMXB4O2NvbG9yOnZhcigtLXRleHQpO2xpbmUtaGVpZ2h0OjEuNDsKICBkaXNw"
    "bGF5Oi13ZWJraXQtYm94Oy13ZWJraXQtbGluZS1jbGFtcDozOy13ZWJraXQtYm94LW9yaWVudDp2ZXJ0aWNhbDtvdmVyZmxvdzpoaWRkZW59Ci5naXRl"
    "bSAucm17cG9zaXRpb246YWJzb2x1dGU7dG9wOjdweDtyaWdodDo3cHg7d2lkdGg6MjZweDtoZWlnaHQ6MjZweDtib3JkZXItcmFkaXVzOjdweDsKICBi"
    "YWNrZ3JvdW5kOnJnYmEoNCw3LDExLC43NSk7Y29sb3I6I2ZmZjtkaXNwbGF5OmdyaWQ7cGxhY2UtaXRlbXM6Y2VudGVyO29wYWNpdHk6MDt0cmFuc2l0"
    "aW9uOi4xNHN9Ci5naXRlbTpob3ZlciAucm17b3BhY2l0eToxfSAuZ2l0ZW0gLnJtOmhvdmVye2JhY2tncm91bmQ6dmFyKC0tcmVkKX0KLnNpemVncmlk"
    "e2Rpc3BsYXk6Z3JpZDtncmlkLXRlbXBsYXRlLWNvbHVtbnM6cmVwZWF0KDMsMWZyKTtnYXA6OHB4fQouc2l6ZW9wdHtwYWRkaW5nOjlweDtib3JkZXI6"
    "MXB4IHNvbGlkIHZhcigtLWxpbmUpO2JvcmRlci1yYWRpdXM6dmFyKC0tci1zbSk7dGV4dC1hbGlnbjpjZW50ZXI7CiAgZm9udC1zaXplOjEycHg7Y3Vy"
    "c29yOnBvaW50ZXI7dHJhbnNpdGlvbjouMTJzO2JhY2tncm91bmQ6dmFyKC0tcGFuZWwtMil9Ci5zaXplb3B0OmhvdmVye2JvcmRlci1jb2xvcjp2YXIo"
    "LS1kaW0pfSAuc2l6ZW9wdC5vbntib3JkZXItY29sb3I6dmFyKC0tc2lnbmFsKTsKICBiYWNrZ3JvdW5kOnZhcigtLXNpZ25hbC1nbG93KTtjb2xvcjp2"
    "YXIoLS1zaWduYWwpfQouc2l6ZW9wdCAubW9ub3tmb250LXNpemU6MTBweDtjb2xvcjp2YXIoLS1kaW0pO2Rpc3BsYXk6YmxvY2t9Ci5zaXplb3B0Lm9u"
    "IC5tb25ve2NvbG9yOnZhcigtLXNpZ25hbCl9CgovKiAtLS0tLS0tLS0tIHRvb2xzIC8gbWNwIC8gcmFnIC8gc2V0dGluZ3MgLS0tLS0tLS0tLSAqLwou"
    "c2V0dXAtY2FyZHt0ZXh0LWFsaWduOmNlbnRlcjtwYWRkaW5nOjM0cHggMjZweH0KLnNldHVwLWNhcmQgLmlje2ZvbnQtc2l6ZTo0MHB4O21hcmdpbi1i"
    "b3R0b206MTRweH0KLmluc3RhbGxsb2d7YmFja2dyb3VuZDp2YXIoLS1pbmspO2JvcmRlcjoxcHggc29saWQgdmFyKC0tbGluZSk7Ym9yZGVyLXJhZGl1"
    "czp2YXIoLS1yLXNtKTsKICBwYWRkaW5nOjEycHg7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjExcHg7Y29sb3I6dmFyKC0tbXV0ZWQp"
    "O21heC1oZWlnaHQ6MjIwcHg7CiAgb3ZlcmZsb3c6YXV0bzt0ZXh0LWFsaWduOmxlZnQ7d2hpdGUtc3BhY2U6cHJlLXdyYXA7bWFyZ2luLXRvcDoxNHB4"
    "O2xpbmUtaGVpZ2h0OjEuNX0KLnRvb2xyb3d7ZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6MTNweDtwYWRkaW5nOjEycHggMTRweDti"
    "b3JkZXI6MXB4IHNvbGlkIHZhcigtLWxpbmUpOwogIGJvcmRlci1yYWRpdXM6dmFyKC0tci1zbSk7bWFyZ2luLWJvdHRvbTo4cHg7YmFja2dyb3VuZDp2"
    "YXIoLS1wYW5lbCl9Ci50b29scm93IC50aWN7d2lkdGg6MzBweDtoZWlnaHQ6MzBweDtib3JkZXItcmFkaXVzOjhweDtiYWNrZ3JvdW5kOnZhcigtLXBh"
    "bmVsLTIpOwogIGJvcmRlcjoxcHggc29saWQgdmFyKC0tbGluZSk7ZGlzcGxheTpncmlkO3BsYWNlLWl0ZW1zOmNlbnRlcjtmbGV4Om5vbmU7Y29sb3I6"
    "dmFyKC0tdmlvbGV0KX0KLnRvb2xyb3cgLnRpe2ZsZXg6MTttaW4td2lkdGg6MH0KLnRvb2xyb3cgLnRpIC5ubXtmb250LWZhbWlseTp2YXIoLS1tb25v"
    "KTtmb250LXNpemU6MTNweDtmb250LXdlaWdodDo2MDB9Ci50b29scm93IC50aSAuZHN7Zm9udC1zaXplOjEycHg7Y29sb3I6dmFyKC0tbXV0ZWQpfQou"
    "c3J2e2JvcmRlcjoxcHggc29saWQgdmFyKC0tbGluZSk7Ym9yZGVyLXJhZGl1czp2YXIoLS1yKTtwYWRkaW5nOjE0cHggMTZweDttYXJnaW4tYm90dG9t"
    "OjEwcHg7CiAgYmFja2dyb3VuZDp2YXIoLS1wYW5lbCl9Ci5zcnYgLnNoe2Rpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7Z2FwOjExcHh9Ci5z"
    "cnYgLnNoIC5ubXtmb250LXdlaWdodDo2MDA7Zm9udC1zaXplOjE0cHh9IC5zcnYgLmNtZHtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6"
    "MTFweDsKICBjb2xvcjp2YXIoLS1kaW0pO21hcmdpbi10b3A6NXB4O3dvcmQtYnJlYWs6YnJlYWstYWxsfQouc3J2IC50b29sY2hpcHN7ZGlzcGxheTpm"
    "bGV4O2dhcDo1cHg7ZmxleC13cmFwOndyYXA7bWFyZ2luLXRvcDo5cHh9Ci5kb2Nyb3d7ZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6"
    "MTNweDtwYWRkaW5nOjEycHggMTRweDtib3JkZXI6MXB4IHNvbGlkIHZhcigtLWxpbmUpOwogIGJvcmRlci1yYWRpdXM6dmFyKC0tci1zbSk7bWFyZ2lu"
    "LWJvdHRvbTo4cHg7YmFja2dyb3VuZDp2YXIoLS1wYW5lbCl9Ci5kb2Nyb3cgLmRpY3t3aWR0aDozNHB4O2hlaWdodDozNHB4O2JvcmRlci1yYWRpdXM6"
    "OHB4O2JhY2tncm91bmQ6dmFyKC0tcGFuZWwtMik7CiAgYm9yZGVyOjFweCBzb2xpZCB2YXIoLS1saW5lKTtkaXNwbGF5OmdyaWQ7cGxhY2UtaXRlbXM6"
    "Y2VudGVyO2ZsZXg6bm9uZTtjb2xvcjp2YXIoLS1ibHVlKX0KLmRvY3JvdyAuZGl7ZmxleDoxO21pbi13aWR0aDowfSAuZG9jcm93IC5kaSAubm17Zm9u"
    "dC13ZWlnaHQ6NjAwO2ZvbnQtc2l6ZToxMy41cHg7CiAgb3ZlcmZsb3c6aGlkZGVuO3RleHQtb3ZlcmZsb3c6ZWxsaXBzaXM7d2hpdGUtc3BhY2U6bm93"
    "cmFwfQouZG9jcm93IC5kaSAubXR7Zm9udC1zaXplOjExLjVweDtjb2xvcjp2YXIoLS1kaW0pO2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pfQouZHJvcHpv"
    "bmV7Ym9yZGVyOjJweCBkYXNoZWQgdmFyKC0tbGluZSk7Ym9yZGVyLXJhZGl1czp2YXIoLS1yKTtwYWRkaW5nOjMycHg7dGV4dC1hbGlnbjpjZW50ZXI7"
    "CiAgY29sb3I6dmFyKC0tbXV0ZWQpO3RyYW5zaXRpb246LjE1cztjdXJzb3I6cG9pbnRlcjtiYWNrZ3JvdW5kOnZhcigtLXBhbmVsLTIpfQouZHJvcHpv"
    "bmU6aG92ZXIsLmRyb3B6b25lLmRyYWd7Ym9yZGVyLWNvbG9yOnZhcigtLXNpZ25hbCk7YmFja2dyb3VuZDp2YXIoLS1zaWduYWwtZ2xvdyk7Y29sb3I6"
    "dmFyKC0tdGV4dCl9Ci5zZXR0aW5ncm93e2Rpc3BsYXk6ZmxleDtqdXN0aWZ5LWNvbnRlbnQ6c3BhY2UtYmV0d2VlbjthbGlnbi1pdGVtczpjZW50ZXI7"
    "Z2FwOjE2cHg7CiAgcGFkZGluZzoxNXB4IDA7Ym9yZGVyLWJvdHRvbToxcHggc29saWQgdmFyKC0tbGluZS1zb2Z0KX0KLnNldHRpbmdyb3c6bGFzdC1j"
    "aGlsZHtib3JkZXItYm90dG9tOm5vbmV9Ci5zZXR0aW5ncm93IC5sYWJ7Zm9udC1zaXplOjEzLjVweDtmb250LXdlaWdodDo1MDB9IC5zZXR0aW5ncm93"
    "IC5sYWIgLnN1Yntmb250LXNpemU6MTJweDsKICBjb2xvcjp2YXIoLS1tdXRlZCk7Zm9udC13ZWlnaHQ6NDAwO21hcmdpbi10b3A6MnB4O21heC13aWR0"
    "aDo0NmNofQouc2V0dGluZ3JvdyAuY3Rse2ZsZXg6bm9uZTttaW4td2lkdGg6MTgwcHh9Ci5zd2l0Y2h7d2lkdGg6NDRweDtoZWlnaHQ6MjRweDtib3Jk"
    "ZXItcmFkaXVzOjIwcHg7YmFja2dyb3VuZDp2YXIoLS1saW5lKTtwb3NpdGlvbjpyZWxhdGl2ZTsKICBjdXJzb3I6cG9pbnRlcjt0cmFuc2l0aW9uOi4x"
    "NXM7ZmxleDpub25lfQouc3dpdGNoOjphZnRlcntjb250ZW50OiIiO3Bvc2l0aW9uOmFic29sdXRlO3RvcDoycHg7bGVmdDoycHg7d2lkdGg6MThweDto"
    "ZWlnaHQ6MThweDsKICBib3JkZXItcmFkaXVzOjUwJTtiYWNrZ3JvdW5kOiNmZmY7dHJhbnNpdGlvbjouMTVzfQouc3dpdGNoLm9ue2JhY2tncm91bmQ6"
    "dmFyKC0tc2lnbmFsKX0gLnN3aXRjaC5vbjo6YWZ0ZXJ7bGVmdDoyMnB4fQoKLyogdXBkYXRlIGJhbm5lciAqLwoudXBkYXRlYmFubmVye21hcmdpbjow"
    "IDAgMjBweDtib3JkZXI6MXB4IHNvbGlkIHZhcigtLXNpZ25hbC1kaW0pO2JvcmRlci1yYWRpdXM6dmFyKC0tcik7CiAgYmFja2dyb3VuZDpsaW5lYXIt"
    "Z3JhZGllbnQoMTAwZGVnLHZhcigtLXNpZ25hbC1nbG93KSx0cmFuc3BhcmVudCk7cGFkZGluZzoxNnB4IDE4cHg7CiAgZGlzcGxheTpmbGV4O2FsaWdu"
    "LWl0ZW1zOmNlbnRlcjtnYXA6MTVweH0KLnVwZGF0ZWJhbm5lciAuaWN7d2lkdGg6NDBweDtoZWlnaHQ6NDBweDtib3JkZXItcmFkaXVzOjEwcHg7YmFj"
    "a2dyb3VuZDp2YXIoLS1zaWduYWwpOwogIGNvbG9yOiMxYTEyMDQ7ZGlzcGxheTpncmlkO3BsYWNlLWl0ZW1zOmNlbnRlcjtmbGV4Om5vbmV9Ci51cGRh"
    "dGViYW5uZXIgLnVpe2ZsZXg6MX0KLnVwZGF0ZWJhbm5lciAudWkgLnR7Zm9udC13ZWlnaHQ6NjAwO2ZvbnQtc2l6ZToxNHB4fQoudXBkYXRlYmFubmVy"
    "IC51aSAuZHtmb250LXNpemU6MTIuNXB4O2NvbG9yOnZhcigtLW11dGVkKTttYXJnaW4tdG9wOjJweH0KLm5hdiAudXBkb3R7cG9zaXRpb246YWJzb2x1"
    "dGU7cmlnaHQ6MTBweDt3aWR0aDo3cHg7aGVpZ2h0OjdweDtib3JkZXItcmFkaXVzOjUwJTsKICBiYWNrZ3JvdW5kOnZhcigtLXNpZ25hbCk7Ym94LXNo"
    "YWRvdzowIDAgOHB4IHZhcigtLXNpZ25hbCl9Cjwvc3R5bGU+CjwvaGVhZD4KPGJvZHk+CjxkaXYgY2xhc3M9ImFwcCI+CiAgPCEtLSA9PT09PT09PT09"
    "PT0gU0lERUJBUiA9PT09PT09PT09PT0gLS0+CiAgPGFzaWRlIGNsYXNzPSJzaWRlIiBpZD0ic2lkZSI+CiAgICA8ZGl2IGNsYXNzPSJicmFuZCI+CiAg"
    "ICAgIDxkaXYgY2xhc3M9Im1hcmsiPjwvZGl2PgogICAgICA8ZGl2PgogICAgICAgIDxoMT5IZW9ydGg8L2gxPgogICAgICAgIDxkaXYgY2xhc3M9InZl"
    "ciIgaWQ9ImJyYW5kVmVyIj524oCUPC9kaXY+CiAgICAgIDwvZGl2PgogICAgPC9kaXY+CgogICAgPG5hdiBjbGFzcz0ibmF2IiBpZD0ibmF2Ij4KICAg"
    "ICAgPGJ1dHRvbiBkYXRhLXZpZXc9ImRhc2hib2FyZCIgY2xhc3M9ImFjdGl2ZSI+CiAgICAgICAgPHN2ZyBjbGFzcz0iaWMiIHZpZXdCb3g9IjAgMCAy"
    "NCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJjdXJyZW50Q29sb3IiIHN0cm9rZS13aWR0aD0iMS44Ij48cmVjdCB4PSIzIiB5PSIzIiB3aWR0aD0iNyIg"
    "aGVpZ2h0PSI5IiByeD0iMS41Ii8+PHJlY3QgeD0iMTQiIHk9IjMiIHdpZHRoPSI3IiBoZWlnaHQ9IjUiIHJ4PSIxLjUiLz48cmVjdCB4PSIxNCIgeT0i"
    "MTIiIHdpZHRoPSI3IiBoZWlnaHQ9IjkiIHJ4PSIxLjUiLz48cmVjdCB4PSIzIiB5PSIxNiIgd2lkdGg9IjciIGhlaWdodD0iNSIgcng9IjEuNSIvPjwv"
    "c3ZnPgogICAgICAgIERhc2hib2FyZAogICAgICA8L2J1dHRvbj4KICAgICAgPGJ1dHRvbiBkYXRhLXZpZXc9ImNoYXQiPgogICAgICAgIDxzdmcgY2xh"
    "c3M9ImljIiB2aWV3Qm94PSIwIDAgMjQgMjQiIGZpbGw9Im5vbmUiIHN0cm9rZT0iY3VycmVudENvbG9yIiBzdHJva2Utd2lkdGg9IjEuOCI+PHBhdGgg"
    "ZD0iTTIxIDEyYTggOCAwIDAgMS0xMS41IDcuMkw0IDIwbDEtNC4zQTggOCAwIDEgMSAyMSAxMloiLz48L3N2Zz4KICAgICAgICBDaGF0CiAgICAgIDwv"
    "YnV0dG9uPgogICAgICA8YnV0dG9uIGRhdGEtdmlldz0ibW9kZWxzIj4KICAgICAgICA8c3ZnIGNsYXNzPSJpYyIgdmlld0JveD0iMCAwIDI0IDI0IiBm"
    "aWxsPSJub25lIiBzdHJva2U9ImN1cnJlbnRDb2xvciIgc3Ryb2tlLXdpZHRoPSIxLjgiPjxwYXRoIGQ9Ik0xMiAzIDMgNy41IDEyIDEybDktNC41TDEy"
    "IDNaIi8+PHBhdGggZD0iTTMgMTJsOSA0LjVMMjEgMTIiLz48cGF0aCBkPSJNMyAxNi41IDEyIDIxbDktNC41Ii8+PC9zdmc+CiAgICAgICAgTW9kZWxz"
    "CiAgICAgIDwvYnV0dG9uPgogICAgICA8YnV0dG9uIGRhdGEtdmlldz0iaW1hZ2VzIj4KICAgICAgICA8c3ZnIGNsYXNzPSJpYyIgdmlld0JveD0iMCAw"
    "IDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9ImN1cnJlbnRDb2xvciIgc3Ryb2tlLXdpZHRoPSIxLjgiPjxyZWN0IHg9IjMiIHk9IjMiIHdpZHRoPSIx"
    "OCIgaGVpZ2h0PSIxOCIgcng9IjIuNSIvPjxjaXJjbGUgY3g9IjguNSIgY3k9IjguNSIgcj0iMS44Ii8+PHBhdGggZD0ibTIxIDE1LTUtNUw1IDIxIi8+"
    "PC9zdmc+CiAgICAgICAgSW1hZ2VzCiAgICAgIDwvYnV0dG9uPgogICAgICA8YnV0dG9uIGRhdGEtdmlldz0ia25vd2xlZGdlIj4KICAgICAgICA8c3Zn"
    "IGNsYXNzPSJpYyIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9ImN1cnJlbnRDb2xvciIgc3Ryb2tlLXdpZHRoPSIxLjgiPjxw"
    "YXRoIGQ9Ik00IDUuNUEyLjUgMi41IDAgMCAxIDYuNSAzSDIwdjE1SDYuNUEyLjUgMi41IDAgMCAwIDQgMjAuNVY1LjVaIi8+PHBhdGggZD0iTTQgNS41"
    "VjIwIi8+PC9zdmc+CiAgICAgICAgS25vd2xlZGdlCiAgICAgIDwvYnV0dG9uPgogICAgICA8YnV0dG9uIGRhdGEtdmlldz0idG9vbHMiPgogICAgICAg"
    "IDxzdmcgY2xhc3M9ImljIiB2aWV3Qm94PSIwIDAgMjQgMjQiIGZpbGw9Im5vbmUiIHN0cm9rZT0iY3VycmVudENvbG9yIiBzdHJva2Utd2lkdGg9IjEu"
    "OCI+PHBhdGggZD0iTTE0LjcgNi4zYTQgNCAwIDAgMC01LjQgNS40bC02IDYgMiAyIDYtNmE0IDQgMCAwIDAgNS40LTUuNGwtMi41IDIuNS0yLTIgMi41"
    "LTIuNVoiLz48L3N2Zz4KICAgICAgICBBZ2VudCAmYW1wOyBUb29scwogICAgICA8L2J1dHRvbj4KICAgICAgPGJ1dHRvbiBkYXRhLXZpZXc9ImNvbXB1"
    "dGVyIj4KICAgICAgICA8c3ZnIGNsYXNzPSJpYyIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9ImN1cnJlbnRDb2xvciIgc3Ry"
    "b2tlLXdpZHRoPSIxLjgiPjxyZWN0IHg9IjIuNSIgeT0iNCIgd2lkdGg9IjE5IiBoZWlnaHQ9IjEyIiByeD0iMiIvPjxwYXRoIGQ9Ik04IDIwaDhNMTIg"
    "MTZ2NCIvPjwvc3ZnPgogICAgICAgIENvbXB1dGVyCiAgICAgIDwvYnV0dG9uPgogICAgICA8ZGl2IGNsYXNzPSJuYXYtc2VwIj48L2Rpdj4KICAgICAg"
    "PGJ1dHRvbiBkYXRhLXZpZXc9InNldHRpbmdzIj4KICAgICAgICA8c3ZnIGNsYXNzPSJpYyIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBz"
    "dHJva2U9ImN1cnJlbnRDb2xvciIgc3Ryb2tlLXdpZHRoPSIxLjgiPjxjaXJjbGUgY3g9IjEyIiBjeT0iMTIiIHI9IjMiLz48cGF0aCBkPSJNMTkuNCAx"
    "NWExLjYgMS42IDAgMCAwIC4zIDEuOGwuMS4xYTIgMiAwIDEgMS0yLjggMi44bC0uMS0uMWExLjYgMS42IDAgMCAwLTEuOC0uMyAxLjYgMS42IDAgMCAw"
    "LTEgMS41VjIxYTIgMiAwIDEgMS00IDB2LS4xYTEuNiAxLjYgMCAwIDAtMS0xLjUgMS42IDEuNiAwIDAgMC0xLjguM2wtLjEuMWEyIDIgMCAxIDEtMi44"
    "LTIuOGwuMS0uMWExLjYgMS42IDAgMCAwIC4zLTEuOCAxLjYgMS42IDAgMCAwLTEuNS0xSDNhMiAyIDAgMSAxIDAtNGguMWExLjYgMS42IDAgMCAwIDEu"
    "NS0xIDEuNiAxLjYgMCAwIDAtLjMtMS44bC0uMS0uMWEyIDIgMCAxIDEgMi44LTIuOGwuMS4xYTEuNiAxLjYgMCAwIDAgMS44LjNIOWExLjYgMS42IDAg"
    "MCAwIDEtMS41VjNhMiAyIDAgMSAxIDQgMHYuMWExLjYgMS42IDAgMCAwIDEgMS41IDEuNiAxLjYgMCAwIDAgMS44LS4zbC4xLS4xYTIgMiAwIDEgMSAy"
    "LjggMi44bC0uMS4xYTEuNiAxLjYgMCAwIDAtLjMgMS44VjlhMS42IDEuNiAwIDAgMCAxLjUgMUgyMWEyIDIgMCAxIDEgMCA0aC0uMWExLjYgMS42IDAg"
    "MCAwLTEuNSAxWiIvPjwvc3ZnPgogICAgICAgIFNldHRpbmdzCiAgICAgIDwvYnV0dG9uPgogICAgPC9uYXY+CgogICAgPGRpdiBjbGFzcz0ic2lkZS1m"
    "b290Ij4KICAgICAgPGRpdiBjbGFzcz0ic3RhdHVzYmFyIj4KICAgICAgICA8c3BhbiBjbGFzcz0iZG90IiBpZD0ib2xsYW1hRG90Ij48L3NwYW4+CiAg"
    "ICAgICAgPHNwYW4gaWQ9Im9sbGFtYVN0YXR1cyIgY2xhc3M9Im11dGVkIj5DaGVja2luZyBPbGxhbWHigKY8L3NwYW4+CiAgICAgIDwvZGl2PgogICAg"
    "ICA8YnV0dG9uIGNsYXNzPSJ0aGVtZXRvZ2dsZSIgaWQ9InRoZW1lQnRuIj4KICAgICAgICA8c3ZnIHdpZHRoPSIxNCIgaGVpZ2h0PSIxNCIgdmlld0Jv"
    "eD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9ImN1cnJlbnRDb2xvciIgc3Ryb2tlLXdpZHRoPSIxLjgiPjxwYXRoIGQ9Ik0yMSAxMi44QTkg"
    "OSAwIDEgMSAxMS4yIDNhNyA3IDAgMCAwIDkuOCA5LjhaIi8+PC9zdmc+CiAgICAgICAgPHNwYW4gaWQ9InRoZW1lTGFiZWwiPkxpZ2h0IG1vZGU8L3Nw"
    "YW4+CiAgICAgIDwvYnV0dG9uPgogICAgPC9kaXY+CiAgPC9hc2lkZT4KCiAgPCEtLSA9PT09PT09PT09PT0gTUFJTiA9PT09PT09PT09PT0gLS0+CiAg"
    "PG1haW4gY2xhc3M9Im1haW4iPgogICAgPGRpdiBjbGFzcz0ibW9iaWxlYmFyIj4KICAgICAgPGJ1dHRvbiBjbGFzcz0iYnRuIGljb24gZ2hvc3QiIGlk"
    "PSJtZW51QnRuIj4KICAgICAgICA8c3ZnIHdpZHRoPSIyMCIgaGVpZ2h0PSIyMCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9"
    "ImN1cnJlbnRDb2xvciIgc3Ryb2tlLXdpZHRoPSIyIj48cGF0aCBkPSJNNCA2aDE2TTQgMTJoMTZNNCAxOGgxNiIvPjwvc3ZnPgogICAgICA8L2J1dHRv"
    "bj4KICAgICAgPHN0cm9uZyBzdHlsZT0iZm9udC1mYW1pbHk6dmFyKC0tZGlzcCkiPkhlb3J0aDwvc3Ryb25nPgogICAgPC9kaXY+CgogICAgPCEtLSA9"
    "PT09PSBEQVNIQk9BUkQgPT09PT0gLS0+CiAgICA8c2VjdGlvbiBjbGFzcz0idmlldyIgZGF0YS12aWV3PSJkYXNoYm9hcmQiPgogICAgICA8ZGl2IGNs"
    "YXNzPSJoZCI+CiAgICAgICAgPGRpdj4KICAgICAgICAgIDxkaXYgY2xhc3M9ImV5ZWJyb3ciPllvdXIgbWFjaGluZTwvZGl2PgogICAgICAgICAgPGgy"
    "PkRhc2hib2FyZDwvaDI+CiAgICAgICAgICA8cCBjbGFzcz0ic3ViIj5IZW9ydGggc2Nhbm5lZCB5b3VyIGhhcmR3YXJlIGFuZCBwaWNrZWQgbW9kZWxz"
    "IHRoYXQKICAgICAgICAgICAgd2lsbCBydW4gd2VsbCBoZXJlLiBFdmVyeXRoaW5nIHJ1bnMgb24geW91ciBjb21wdXRlciDigJQgbm90aGluZyBsZWF2"
    "ZXMgaXQuPC9wPgogICAgICAgIDwvZGl2PgogICAgICAgIDxidXR0b24gY2xhc3M9ImJ0biBnaG9zdCBzbSIgaWQ9InJlc2NhbkJ0biI+CiAgICAgICAg"
    "ICA8c3ZnIHdpZHRoPSIxNCIgaGVpZ2h0PSIxNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9ImN1cnJlbnRDb2xvciIgc3Ry"
    "b2tlLXdpZHRoPSIyIj48cGF0aCBkPSJNMjEgMTJhOSA5IDAgMSAxLTIuNi02LjRNMjEgM3Y1aC01Ii8+PC9zdmc+CiAgICAgICAgICBSZXNjYW4KICAg"
    "ICAgICA8L2J1dHRvbj4KICAgICAgPC9kaXY+CgogICAgICA8ZGl2IGlkPSJ1cGRhdGVCYW5uZXJIb3N0Ij48L2Rpdj4KCiAgICAgIDxkaXYgY2xhc3M9"
    "Imhlcm8tcGFuZWwiIGlkPSJoZXJvUGFuZWwiPgogICAgICAgIDxkaXYgY2xhc3M9ImdhdWdlLXJvdyI+CiAgICAgICAgICA8ZGl2IGNsYXNzPSJnYXVn"
    "ZSIgaWQ9ImdhdWdlIj48ZGl2IGNsYXNzPSJzcGluIj48L2Rpdj48L2Rpdj4KICAgICAgICAgIDxkaXYgc3R5bGU9ImZsZXg6MSI+CiAgICAgICAgICAg"
    "IDxkaXYgY2xhc3M9InNwZWNzIiBpZD0ic3BlY3MiPjwvZGl2PgogICAgICAgICAgPC9kaXY+CiAgICAgICAgPC9kaXY+CiAgICAgICAgPGRpdiBjbGFz"
    "cz0idGllci1ub3RlIiBpZD0idGllck5vdGUiPlNjYW5uaW5n4oCmPC9kaXY+CiAgICAgIDwvZGl2PgoKICAgICAgPGRpdiBjbGFzcz0ic2VjdGlvbi10"
    "aXRsZSI+UmVjb21tZW5kZWQgZm9yIHlvdTwvZGl2PgogICAgICA8ZGl2IGNsYXNzPSJyZWNsaXN0IiBpZD0icmVjTGlzdCI+PC9kaXY+CiAgICA8L3Nl"
    "Y3Rpb24+CgogICAgPCEtLSA9PT09PSBDSEFUID09PT09IC0tPgogICAgPHNlY3Rpb24gY2xhc3M9InZpZXcgY2hhdHZpZXciIGRhdGEtdmlldz0iY2hh"
    "dCIgaGlkZGVuPgogICAgICA8ZGl2IGNsYXNzPSJjaGF0LWhlYWQiPgogICAgICAgIDxkaXYgY2xhc3M9Im1vZGVscGljayI+CiAgICAgICAgICA8c3Bh"
    "biBjbGFzcz0iZGltIG1vbm8iIHN0eWxlPSJmb250LXNpemU6MTFweCI+TU9ERUw8L3NwYW4+CiAgICAgICAgICA8c2VsZWN0IGNsYXNzPSJzZWwiIGlk"
    "PSJjaGF0TW9kZWwiPjxvcHRpb24+bG9hZGluZ+KApjwvb3B0aW9uPjwvc2VsZWN0PgogICAgICAgIDwvZGl2PgogICAgICAgIDxkaXYgY2xhc3M9InRv"
    "Z2dsZSIgaWQ9InJhZ1RvZ2dsZSIgdGl0bGU9IlVzZSB5b3VyIHVwbG9hZGVkIGRvY3VtZW50cyI+CiAgICAgICAgICA8c3BhbiBjbGFzcz0ic3ciPjwv"
    "c3Bhbj4gS25vd2xlZGdlCiAgICAgICAgPC9kaXY+CiAgICAgICAgPGRpdiBjbGFzcz0idG9nZ2xlIHZpbyIgaWQ9ImFnZW50VG9nZ2xlIiB0aXRsZT0i"
    "TGV0IHRoZSBtb2RlbCB1c2UgdG9vbHMiPgogICAgICAgICAgPHNwYW4gY2xhc3M9InN3Ij48L3NwYW4+IEFnZW50CiAgICAgICAgPC9kaXY+CiAgICAg"
    "ICAgPGRpdiBjbGFzcz0idG9nZ2xlIGJsdSIgaWQ9Imxvb3BUb2dnbGUiIHRpdGxlPSJBdXRvbm9tb3VzIGxvb3Ag4oCUIHRoZSBhZ2VudCBwbGFucywg"
    "YWN0cyBhbmQgcmVwZWF0cyB1bnRpbCBpdCBjYWxscyB0aGUgdGFzayBkb25lIj4KICAgICAgICAgIDxzcGFuIGNsYXNzPSJzdyI+PC9zcGFuPiBMb29w"
    "CiAgICAgICAgPC9kaXY+CiAgICAgICAgPGRpdiBjbGFzcz0idG9nZ2xlIGdybiIgaWQ9ImNvdW5jaWxUb2dnbGUiIHRpdGxlPSJBIHBhbmVsIG9mIGNv"
    "bnN1bHRhbnRzIGFuYWx5emVzIGluIHBhcmFsbGVsLCBjcml0aXF1ZXMgZWFjaCBvdGhlciwgdGhlbiBhIGNoYWlyIHN5bnRoZXNpemVzIHRoZSBhbnN3"
    "ZXIiPgogICAgICAgICAgPHNwYW4gY2xhc3M9InN3Ij48L3NwYW4+IENvdW5jaWwKICAgICAgICA8L2Rpdj4KICAgICAgICA8ZGl2IGNsYXNzPSJ0b2dn"
    "bGUgcmVkIiBpZD0iY29tcHV0ZXJUb2dnbGUiIHRpdGxlPSJMZXQgdGhlIG1vZGVsIHNlZSB0aGUgc2NyZWVuIGFuZCBjb250cm9sIHRoZSBtb3VzZSAm"
    "IGtleWJvYXJkIChtdXN0IGJlIGVuYWJsZWQgb24gdGhlIENvbXB1dGVyIHBhZ2UpIj4KICAgICAgICAgIDxzcGFuIGNsYXNzPSJzdyI+PC9zcGFuPiBD"
    "b21wdXRlcgogICAgICAgIDwvZGl2PgogICAgICAgIDxkaXYgY2xhc3M9InRvZ2dsZSIgaWQ9ImNvZGVyVG9nZ2xlIiB0aXRsZT0iQ29kaW5nIGFnZW50"
    "IMOgIGxhIG9wZW5jb2RlIOKAlCBleHBsb3JlcywgZWRpdHMgYW5kIHRlc3RzIGEgcmVhbCBwcm9qZWN0IGZvbGRlciAoc2V0IHRoZSBmb2xkZXIgaW4g"
    "U2V0dGluZ3Mg4oaSIENvZGVyKSI+CiAgICAgICAgICA8c3BhbiBjbGFzcz0ic3ciPjwvc3Bhbj4gQ29kZXIKICAgICAgICA8L2Rpdj4KICAgICAgICA8"
    "YnV0dG9uIGNsYXNzPSJidG4gc20gZ2hvc3QiIGlkPSJleHBvcnRCdG4iIHRpdGxlPSJEb3dubG9hZCB0aGlzIGNvbnZlcnNhdGlvbiBhcyBNYXJrZG93"
    "biIgc3R5bGU9Im1hcmdpbi1sZWZ0OmF1dG8iPgogICAgICAgICAgPHN2ZyB3aWR0aD0iMTQiIGhlaWdodD0iMTQiIHZpZXdCb3g9IjAgMCAyNCAyNCIg"
    "ZmlsbD0ibm9uZSIgc3Ryb2tlPSJjdXJyZW50Q29sb3IiIHN0cm9rZS13aWR0aD0iMiI+PHBhdGggZD0iTTIxIDE1djRhMiAyIDAgMCAxLTIgMkg1YTIg"
    "MiAwIDAgMS0yLTJ2LTRNNyAxMGw1IDUgNS01TTEyIDE1VjMiLz48L3N2Zz4KICAgICAgICAgIEV4cG9ydAogICAgICAgIDwvYnV0dG9uPgogICAgICAg"
    "IDxkaXYgc3R5bGU9ImZsZXg6MSI+PC9kaXY+CiAgICAgICAgPGJ1dHRvbiBjbGFzcz0iYnRuIHNtIGdob3N0IiBpZD0iY2xlYXJDaGF0QnRuIj5OZXcg"
    "Y2hhdDwvYnV0dG9uPgogICAgICA8L2Rpdj4KICAgICAgPGRpdiBjbGFzcz0iY2hhdC1sYXlvdXQiPgogICAgICAgIDxkaXYgY2xhc3M9ImNoYXQtbWFp"
    "biI+CiAgICAgICAgICA8ZGl2IGNsYXNzPSJjaGF0c2Nyb2xsIHNjcm9sbCIgaWQ9ImNoYXRTY3JvbGwiPgogICAgICAgICAgICA8ZGl2IGNsYXNzPSJj"
    "aGF0d3JhcCIgaWQ9ImNoYXRXcmFwIj48L2Rpdj4KICAgICAgICAgIDwvZGl2PgogICAgICAgICAgPGRpdiBjbGFzcz0iY29tcG9zZXIiPgogICAgICAg"
    "ICAgICA8ZGl2IGNsYXNzPSJhdHRhY2hyb3ciIGlkPSJhdHRhY2hSb3ciIGhpZGRlbj48L2Rpdj4KICAgICAgICAgICAgPGRpdiBjbGFzcz0iY2JveCI+"
    "CiAgICAgICAgICAgICAgPGJ1dHRvbiBjbGFzcz0iYXR0YWNoYnRuIiBpZD0iYXR0YWNoQnRuIiB0aXRsZT0iQXR0YWNoIGltYWdlcyDigJQgYXNrIGFi"
    "b3V0IHRoZW0gd2l0aCBhIHZpc2lvbiBtb2RlbCAoZ2VtbWEzLCBnZW1tYTQsIHF3ZW4yLjV2bCwgbGxhdmEpIj4KICAgICAgICAgICAgICAgIDxzdmcg"
    "d2lkdGg9IjE3IiBoZWlnaHQ9IjE3IiB2aWV3Qm94PSIwIDAgMjQgMjQiIGZpbGw9Im5vbmUiIHN0cm9rZT0iY3VycmVudENvbG9yIiBzdHJva2Utd2lk"
    "dGg9IjIiPjxwYXRoIGQ9Im0yMS40NCAxMS4wNS05LjE5IDkuMTlhNiA2IDAgMCAxLTguNDktOC40OWw4LjU3LTguNTdBNCA0IDAgMSAxIDE4IDguODRs"
    "LTguNTkgOC41N2EyIDIgMCAwIDEtMi44My0yLjgzbDguNDktOC40OCIvPjwvc3ZnPgogICAgICAgICAgICAgIDwvYnV0dG9uPgogICAgICAgICAgICAg"
    "IDxpbnB1dCB0eXBlPSJmaWxlIiBpZD0iYXR0YWNoSW5wdXQiIGFjY2VwdD0iaW1hZ2UvKiIgbXVsdGlwbGUgaGlkZGVuPgogICAgICAgICAgICAgIDx0"
    "ZXh0YXJlYSBpZD0iY2hhdElucHV0IiByb3dzPSIxIiBwbGFjZWhvbGRlcj0iQXNrIGFueXRoaW5n4oCmIChFbnRlciB0byBzZW5kLCBTaGlmdCtFbnRl"
    "ciBmb3IgYSBuZXcgbGluZSkiPjwvdGV4dGFyZWE+CiAgICAgICAgICAgICAgPGJ1dHRvbiBjbGFzcz0ic2VuZGJ0biIgaWQ9InNlbmRCdG4iIHRpdGxl"
    "PSJTZW5kIj4KICAgICAgICAgICAgICAgIDxzdmcgd2lkdGg9IjE4IiBoZWlnaHQ9IjE4IiB2aWV3Qm94PSIwIDAgMjQgMjQiIGZpbGw9Im5vbmUiIHN0"
    "cm9rZT0iY3VycmVudENvbG9yIiBzdHJva2Utd2lkdGg9IjIiPjxwYXRoIGQ9Ik03IDExIDEyIDZsNSA1TTEyIDZ2MTMiLz48L3N2Zz4KICAgICAgICAg"
    "ICAgICA8L2J1dHRvbj4KICAgICAgICAgICAgPC9kaXY+CiAgICAgICAgICA8L2Rpdj4KICAgICAgICA8L2Rpdj4KICAgICAgICA8ZGl2IGNsYXNzPSJj"
    "aGF0LXNpZGUiPgogICAgICAgICAgPGRpdiBjbGFzcz0ibmV3Y2hhdCI+PGJ1dHRvbiBjbGFzcz0iYnRuIHNtIGdob3N0IiBpZD0ibmV3Q29udkJ0biIg"
    "c3R5bGU9IndpZHRoOjEwMCUiPgogICAgICAgICAgICA8c3ZnIHdpZHRoPSIxNCIgaGVpZ2h0PSIxNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJu"
    "b25lIiBzdHJva2U9ImN1cnJlbnRDb2xvciIgc3Ryb2tlLXdpZHRoPSIyIj48cGF0aCBkPSJNMTIgNXYxNE01IDEyaDE0Ii8+PC9zdmc+CiAgICAgICAg"
    "ICAgIE5ldyBjb252ZXJzYXRpb248L2J1dHRvbj48L2Rpdj4KICAgICAgICAgIDxkaXYgY2xhc3M9ImNvbnZsaXN0IHNjcm9sbCIgaWQ9ImNvbnZMaXN0"
    "Ij48L2Rpdj4KICAgICAgICA8L2Rpdj4KICAgICAgPC9kaXY+CiAgICA8L3NlY3Rpb24+CgogICAgPCEtLSA9PT09PSBNT0RFTFMgPT09PT0gLS0+CiAg"
    "ICA8c2VjdGlvbiBjbGFzcz0idmlldyIgZGF0YS12aWV3PSJtb2RlbHMiIGhpZGRlbj4KICAgICAgPGRpdiBjbGFzcz0iaGQiPgogICAgICAgIDxkaXY+"
    "CiAgICAgICAgICA8ZGl2IGNsYXNzPSJleWVicm93Ij5MaWJyYXJ5PC9kaXY+CiAgICAgICAgICA8aDI+TW9kZWxzPC9oMj4KICAgICAgICAgIDxwIGNs"
    "YXNzPSJzdWIiPlNlYXJjaCB0aG91c2FuZHMgb2YgbW9kZWxzIG9yIHBpY2sgZnJvbSB0aGUgY3VyYXRlZCBsaXN0LgogICAgICAgICAgICBEb3dubG9h"
    "ZHMgYXJlIGhhbmRsZWQgYnkgT2xsYW1hIGFuZCBzdG9yZWQgbG9jYWxseS48L3A+CiAgICAgICAgPC9kaXY+CiAgICAgIDwvZGl2PgogICAgICA8ZGl2"
    "IGNsYXNzPSJzZWFyY2hiYXIiPgogICAgICAgIDxpbnB1dCBjbGFzcz0iaW5wIiBpZD0ibW9kZWxTZWFyY2giIHBsYWNlaG9sZGVyPSJTZWFyY2ggbW9k"
    "ZWxzIOKAlCBlLmcuIGxsYW1hLCBxd2VuIGNvZGVyLCB2aXNpb24sIGVtYmVk4oCmIj4KICAgICAgICA8YnV0dG9uIGNsYXNzPSJidG4gcHJpbWFyeSIg"
    "aWQ9Im1vZGVsU2VhcmNoQnRuIj5TZWFyY2g8L2J1dHRvbj4KICAgICAgPC9kaXY+CiAgICAgIDxkaXYgaWQ9Im1vZGVsc0JvZHkiPjwvZGl2PgogICAg"
    "PC9zZWN0aW9uPgoKICAgIDwhLS0gPT09PT0gSU1BR0VTID09PT09IC0tPgogICAgPHNlY3Rpb24gY2xhc3M9InZpZXciIGRhdGEtdmlldz0iaW1hZ2Vz"
    "IiBoaWRkZW4+CiAgICAgIDxkaXYgY2xhc3M9ImhkIj4KICAgICAgICA8ZGl2PgogICAgICAgICAgPGRpdiBjbGFzcz0iZXllYnJvdyI+R2VuZXJhdGU8"
    "L2Rpdj4KICAgICAgICAgIDxoMj5JbWFnZXM8L2gyPgogICAgICAgICAgPHAgY2xhc3M9InN1YiI+Q3JlYXRlIGltYWdlcyBmcm9tIHRleHQgd2l0aCBT"
    "dGFibGUgRGlmZnVzaW9uLCBydW5uaW5nCiAgICAgICAgICAgIGxvY2FsbHkuIEV2ZXJ5IGltYWdlIGlzIHNhdmVkIHRvIHlvdXIgZ2FsbGVyeSBhdXRv"
    "bWF0aWNhbGx5LjwvcD4KICAgICAgICA8L2Rpdj4KICAgICAgPC9kaXY+CiAgICAgIDxkaXYgaWQ9ImltYWdlc0JvZHkiPjwvZGl2PgogICAgPC9zZWN0"
    "aW9uPgoKICAgIDwhLS0gPT09PT0gS05PV0xFREdFID09PT09IC0tPgogICAgPHNlY3Rpb24gY2xhc3M9InZpZXciIGRhdGEtdmlldz0ia25vd2xlZGdl"
    "IiBoaWRkZW4+CiAgICAgIDxkaXYgY2xhc3M9ImhkIj4KICAgICAgICA8ZGl2PgogICAgICAgICAgPGRpdiBjbGFzcz0iZXllYnJvdyI+UmV0cmlldmFs"
    "PC9kaXY+CiAgICAgICAgICA8aDI+S25vd2xlZGdlIGJhc2U8L2gyPgogICAgICAgICAgPHAgY2xhc3M9InN1YiI+QWRkIGRvY3VtZW50cyBhbmQgdGhl"
    "IGFzc2lzdGFudCBjYW4gY2l0ZSB0aGVtIGluIENoYXQgd2hlbgogICAgICAgICAgICB5b3Ugc3dpdGNoIG9uIEtub3dsZWRnZS4gRmlsZXMgYXJlIGNo"
    "dW5rZWQgYW5kIGVtYmVkZGVkIGxvY2FsbHkuPC9wPgogICAgICAgIDwvZGl2PgogICAgICA8L2Rpdj4KICAgICAgPGRpdiBpZD0ia25vd2xlZGdlQm9k"
    "eSI+PC9kaXY+CiAgICA8L3NlY3Rpb24+CgogICAgPCEtLSA9PT09PSBUT09MUyA9PT09PSAtLT4KICAgIDxzZWN0aW9uIGNsYXNzPSJ2aWV3IiBkYXRh"
    "LXZpZXc9InRvb2xzIiBoaWRkZW4+CiAgICAgIDxkaXYgY2xhc3M9ImhkIj4KICAgICAgICA8ZGl2PgogICAgICAgICAgPGRpdiBjbGFzcz0iZXllYnJv"
    "dyI+Q2FwYWJpbGl0aWVzPC9kaXY+CiAgICAgICAgICA8aDI+QWdlbnQgJmFtcDsgVG9vbHM8L2gyPgogICAgICAgICAgPHAgY2xhc3M9InN1YiI+VGhl"
    "IGFnZW50IGNhbiBjYWxsIHRoZXNlIHRvb2xzIHdoaWxlIGl0IGFuc3dlcnMuIENvbm5lY3QKICAgICAgICAgICAgTUNQIHNlcnZlcnMgdG8gZ2l2ZSBp"
    "dCBldmVuIG1vcmUgYWJpbGl0aWVzLjwvcD4KICAgICAgICA8L2Rpdj4KICAgICAgPC9kaXY+CiAgICAgIDxkaXYgaWQ9InRvb2xzQm9keSI+PC9kaXY+"
    "CiAgICA8L3NlY3Rpb24+CgogICAgPCEtLSA9PT09PSBDT01QVVRFUiA9PT09PSAtLT4KICAgIDxzZWN0aW9uIGNsYXNzPSJ2aWV3IiBkYXRhLXZpZXc9"
    "ImNvbXB1dGVyIiBoaWRkZW4+CiAgICAgIDxkaXYgY2xhc3M9ImhkIj4KICAgICAgICA8ZGl2PgogICAgICAgICAgPGRpdiBjbGFzcz0iZXllYnJvdyI+"
    "RGlyZWN0IGNvbnRyb2w8L2Rpdj4KICAgICAgICAgIDxoMj5Db21wdXRlciBjb250cm9sPC9oMj4KICAgICAgICAgIDxwIGNsYXNzPSJzdWIiPkxldCBh"
    "IG1vZGVsIHNlZSB5b3VyIHNjcmVlbiBhbmQgb3BlcmF0ZSB0aGUgbW91c2UgYW5kCiAgICAgICAgICAgIGtleWJvYXJkIHRvIGRvIHRhc2tzIGZvciB5"
    "b3UuIFBvd2VyZnVsIOKAlCBhbmQgb2ZmIHVudGlsIHlvdSB0dXJuIGl0IG9uLjwvcD4KICAgICAgICA8L2Rpdj4KICAgICAgPC9kaXY+CiAgICAgIDxk"
    "aXYgaWQ9ImNvbXB1dGVyQm9keSI+PC9kaXY+CiAgICA8L3NlY3Rpb24+CgogICAgPCEtLSA9PT09PSBTRVRUSU5HUyA9PT09PSAtLT4KICAgIDxzZWN0"
    "aW9uIGNsYXNzPSJ2aWV3IiBkYXRhLXZpZXc9InNldHRpbmdzIiBoaWRkZW4+CiAgICAgIDxkaXYgY2xhc3M9ImhkIj4KICAgICAgICA8ZGl2PgogICAg"
    "ICAgICAgPGRpdiBjbGFzcz0iZXllYnJvdyI+Q29uZmlndXJhdGlvbjwvZGl2PgogICAgICAgICAgPGgyPlNldHRpbmdzPC9oMj4KICAgICAgICAgIDxw"
    "IGNsYXNzPSJzdWIiPlR1bmUgYmVoYXZpb3VyLCBtYW5hZ2UgdXBkYXRlcywgYW5kIHNlZSB3aGVyZSB5b3VyIGRhdGEgbGl2ZXMuPC9wPgogICAgICAg"
    "IDwvZGl2PgogICAgICA8L2Rpdj4KICAgICAgPGRpdiBpZD0ic2V0dGluZ3NCb2R5Ij48L2Rpdj4KICAgIDwvc2VjdGlvbj4KICA8L21haW4+CjwvZGl2"
    "PgoKPGRpdiBpZD0idG9hc3RzIj48L2Rpdj4KPGRpdiBpZD0ibW9kYWxIb3N0Ij48L2Rpdj4KPHNjcmlwdD4KLyogPT09PT09PT09PT09PT09PT09PT09"
    "PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09CiAgIEhlb3J0aCBmcm9udGVuZCDigJQgdmFuaWxsYSBKUywgbm8gYnVpbGQgc3Rl"
    "cAogICA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT0gKi8KY29uc3QgJCA9IChpZCkgPT4g"
    "ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoaWQpOwpjb25zdCBlbCA9ICh0YWcsIGNscywgaHRtbCkgPT4geyBjb25zdCBlID0gZG9jdW1lbnQuY3JlYXRl"
    "RWxlbWVudCh0YWcpOwogIGlmIChjbHMpIGUuY2xhc3NOYW1lID0gY2xzOyBpZiAoaHRtbCAhPSBudWxsKSBlLmlubmVySFRNTCA9IGh0bWw7IHJldHVy"
    "biBlOyB9Owpjb25zdCBlc2MgPSAocykgPT4gKHMgPT0gbnVsbCA/ICIiIDogU3RyaW5nKHMpKS5yZXBsYWNlKC9bJjw+IiddL2csCiAgYyA9PiAoeycm"
    "JzonJmFtcDsnLCc8JzonJmx0OycsJz4nOicmZ3Q7JywnIic6JyZxdW90OycsIiciOicmIzM5Oyd9W2NdKSk7CmNvbnN0IGZtdEdCID0gKG4pID0+IChu"
    "ID09IG51bGwgPyAiPyIgOiAobiA+PSAxMCA/IE1hdGgucm91bmQobikgOiBuLnRvRml4ZWQoMSkpKTsKCmNvbnN0IHN0YXRlID0geyBzeXN0ZW06bnVs"
    "bCwgc2V0dGluZ3M6e30sIGluc3RhbGxlZDpbXSwgY3VycmVudENvbnY6bnVsbCwKICBzZW5kaW5nOmZhbHNlLCB1cGRhdGVJbmZvOm51bGwsIGltZ1By"
    "ZXNldHM6W10sIHNlbEltZ1NpemU6bnVsbCB9OwoKLyogLS0tLS0tLS0tLSBBUEkgaGVscGVycyAtLS0tLS0tLS0tICovCmFzeW5jIGZ1bmN0aW9uIGFw"
    "aShwYXRoLCBvcHRzKXsKICBjb25zdCByID0gYXdhaXQgZmV0Y2gocGF0aCwgb3B0cyk7CiAgY29uc3QgY3QgPSByLmhlYWRlcnMuZ2V0KCJjb250ZW50"
    "LXR5cGUiKSB8fCAiIjsKICBpZihjdC5pbmNsdWRlcygiYXBwbGljYXRpb24vanNvbiIpKXsKICAgIGNvbnN0IGogPSBhd2FpdCByLmpzb24oKTsKICAg"
    "IGlmKGogJiYgai5vayA9PT0gZmFsc2UgJiYgai5lcnJvcikgdGhyb3cgbmV3IEVycm9yKGouZXJyb3IpOwogICAgcmV0dXJuIGo7CiAgfQogIHJldHVy"
    "biByOwp9CmFzeW5jIGZ1bmN0aW9uIHBvc3QocGF0aCwgYm9keSl7CiAgcmV0dXJuIGFwaShwYXRoLCB7bWV0aG9kOiJQT1NUIiwgaGVhZGVyczp7IkNv"
    "bnRlbnQtVHlwZSI6ImFwcGxpY2F0aW9uL2pzb24ifSwKICAgIGJvZHk6IEpTT04uc3RyaW5naWZ5KGJvZHl8fHt9KX0pOwp9CmFzeW5jIGZ1bmN0aW9u"
    "IGRlbChwYXRoKXsgcmV0dXJuIGFwaShwYXRoLCB7bWV0aG9kOiJERUxFVEUifSk7IH0KCi8qIHN0cmVhbSBOREpTT04gZnJvbSBhIFBPU1QgZW5kcG9p"
    "bnQsIGNhbGxpbmcgb25PYmogZm9yIGVhY2ggcGFyc2VkIGxpbmUgKi8KYXN5bmMgZnVuY3Rpb24gc3RyZWFtTkRKU09OKHBhdGgsIGJvZHksIG9uT2Jq"
    "LCBzaWduYWwpewogIGNvbnN0IHIgPSBhd2FpdCBmZXRjaChwYXRoLCB7bWV0aG9kOiJQT1NUIiwgc2lnbmFsLAogICAgaGVhZGVyczp7IkNvbnRlbnQt"
    "VHlwZSI6ImFwcGxpY2F0aW9uL2pzb24ifSwgYm9keTogSlNPTi5zdHJpbmdpZnkoYm9keXx8e30pfSk7CiAgaWYoIXIub2speyBsZXQgdD0iIjsgdHJ5"
    "e3Q9KGF3YWl0IHIuanNvbigpKS5lcnJvcn1jYXRjaChlKXt0PWF3YWl0IHIudGV4dCgpfQogICAgdGhyb3cgbmV3IEVycm9yKHQgfHwgKCJIVFRQICIr"
    "ci5zdGF0dXMpKTsgfQogIGNvbnN0IHJlYWRlciA9IHIuYm9keS5nZXRSZWFkZXIoKTsgY29uc3QgZGVjID0gbmV3IFRleHREZWNvZGVyKCk7IGxldCBi"
    "dWY9IiI7CiAgd2hpbGUodHJ1ZSl7CiAgICBjb25zdCB7ZG9uZSwgdmFsdWV9ID0gYXdhaXQgcmVhZGVyLnJlYWQoKTsgaWYoZG9uZSkgYnJlYWs7CiAg"
    "ICBidWYgKz0gZGVjLmRlY29kZSh2YWx1ZSwge3N0cmVhbTp0cnVlfSk7IGxldCBubDsKICAgIHdoaWxlKChubCA9IGJ1Zi5pbmRleE9mKCJcbiIpKSA+"
    "PSAwKXsKICAgICAgY29uc3QgbGluZSA9IGJ1Zi5zbGljZSgwLCBubCkudHJpbSgpOyBidWYgPSBidWYuc2xpY2UobmwrMSk7CiAgICAgIGlmKGxpbmUp"
    "eyB0cnl7IG9uT2JqKEpTT04ucGFyc2UobGluZSkpOyB9Y2F0Y2goZSl7fSB9CiAgICB9CiAgfQogIGlmKGJ1Zi50cmltKCkpeyB0cnl7IG9uT2JqKEpT"
    "T04ucGFyc2UoYnVmLnRyaW0oKSkpOyB9Y2F0Y2goZSl7fSB9Cn0KCi8qIC0tLS0tLS0tLS0gdG9hc3QgKyBtb2RhbCAtLS0tLS0tLS0tICovCmZ1bmN0"
    "aW9uIHRvYXN0KG1zZywga2luZCl7CiAgY29uc3QgdCA9IGVsKCJkaXYiLCAidG9hc3QiICsgKGtpbmQgPyAiICIra2luZCA6ICIiKSwgZXNjKG1zZykp"
    "OwogICQoInRvYXN0cyIpLmFwcGVuZENoaWxkKHQpOwogIHNldFRpbWVvdXQoKCk9PnsgdC5zdHlsZS5vcGFjaXR5PSIwIjsgdC5zdHlsZS50cmFuc2Zv"
    "cm09InRyYW5zbGF0ZVgoMjBweCkiOwogICAgdC5zdHlsZS50cmFuc2l0aW9uPSIuM3MiOyBzZXRUaW1lb3V0KCgpPT50LnJlbW92ZSgpLCAzMDApOyB9"
    "LCA0MjAwKTsKfQpmdW5jdGlvbiBtb2RhbCh7dGl0bGUsIGJvZHlIVE1MLCBhY3Rpb25zLCB3aWRlfSl7CiAgY2xvc2VNb2RhbCgpOwogIGNvbnN0IGJn"
    "ID0gZWwoImRpdiIsIm1vZGFsLWJnIik7IGJnLmlkPSJhY3RpdmVNb2RhbCI7CiAgY29uc3QgbSA9IGVsKCJkaXYiLCJtb2RhbCIrKHdpZGU/IiB3aWRl"
    "IjoiIikpOwogIG0uYXBwZW5kQ2hpbGQoZWwoImRpdiIsIm1oIiwgYDxoMz4ke2VzYyh0aXRsZSl9PC9oMz5gKSk7CiAgY29uc3QgYnRuWCA9IGVsKCJi"
    "dXR0b24iLCJidG4gaWNvbiBnaG9zdCIsCiAgICAnPHN2ZyB3aWR0aD0iMTgiIGhlaWdodD0iMTgiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9u"
    "ZSIgc3Ryb2tlPSJjdXJyZW50Q29sb3IiIHN0cm9rZS13aWR0aD0iMiI+PHBhdGggZD0iTTE4IDYgNiAxOE02IDZsMTIgMTIiLz48L3N2Zz4nKTsKICBi"
    "dG5YLm9uY2xpY2sgPSBjbG9zZU1vZGFsOyBtLnF1ZXJ5U2VsZWN0b3IoIi5taCIpLmFwcGVuZENoaWxkKGJ0blgpOwogIGNvbnN0IGJvZHkgPSBlbCgi"
    "ZGl2IiwibWIiKTsgYm9keS5pbm5lckhUTUwgPSBib2R5SFRNTDsgbS5hcHBlbmRDaGlsZChib2R5KTsKICBpZihhY3Rpb25zICYmIGFjdGlvbnMubGVu"
    "Z3RoKXsKICAgIGNvbnN0IG1mID0gZWwoImRpdiIsIm1mIik7CiAgICBhY3Rpb25zLmZvckVhY2goYT0+eyBjb25zdCBiID0gZWwoImJ1dHRvbiIsImJ0"
    "biAiKyhhLmNsc3x8IiIpLCBlc2MoYS5sYWJlbCkpOwogICAgICBiLm9uY2xpY2sgPSAoKT0+YS5vbkNsaWNrICYmIGEub25DbGljayhib2R5KTsgbWYu"
    "YXBwZW5kQ2hpbGQoYik7IH0pOwogICAgbS5hcHBlbmRDaGlsZChtZik7CiAgfQogIGJnLmFwcGVuZENoaWxkKG0pOyBiZy5vbmNsaWNrID0gKGUpPT57"
    "IGlmKGUudGFyZ2V0PT09YmcpIGNsb3NlTW9kYWwoKTsgfTsKICAkKCJtb2RhbEhvc3QiKS5hcHBlbmRDaGlsZChiZyk7IHJldHVybiBib2R5Owp9CmZ1"
    "bmN0aW9uIGNsb3NlTW9kYWwoKXsgY29uc3QgbSA9ICQoImFjdGl2ZU1vZGFsIik7IGlmKG0pIG0ucmVtb3ZlKCk7IH0KZG9jdW1lbnQuYWRkRXZlbnRM"
    "aXN0ZW5lcigia2V5ZG93biIsIGU9PnsgaWYoZS5rZXk9PT0iRXNjYXBlIikgY2xvc2VNb2RhbCgpOyB9KTsKCi8qIC0tLS0tLS0tLS0gY29kZSBibG9j"
    "ayBhY3Rpb25zIChjb3B5IC8gcnVuKSAtLS0tLS0tLS0tICovCmZ1bmN0aW9uIGxvb2tzSHRtbChsYW5nLCBjb2RlKXsKICBpZigvXmh0bWw/JC9pLnRl"
    "c3QobGFuZ3x8IiIpKSByZXR1cm4gdHJ1ZTsKICBjb25zdCBoZWFkID0gKGNvZGV8fCIiKS5zbGljZSgwLDYwMCkudG9Mb3dlckNhc2UoKTsKICByZXR1"
    "cm4gaGVhZC5pbmNsdWRlcygiPCFkb2N0eXBlIGh0bWwiKSB8fCAvPGh0bWxbXHM+XS8udGVzdChoZWFkKTsKfQphc3luYyBmdW5jdGlvbiBjb3B5VGV4"
    "dCh0KXsKICB0cnl7IGF3YWl0IG5hdmlnYXRvci5jbGlwYm9hcmQud3JpdGVUZXh0KHQpOyByZXR1cm4gdHJ1ZTsgfQogIGNhdGNoKGUpewogICAgdHJ5"
    "eyAgLy8gaHR0cCBvdmVyIExBTiAoLS1ob3N0IDAuMC4wLjApIGhhcyBubyBjbGlwYm9hcmQgQVBJCiAgICAgIGNvbnN0IHRhID0gZG9jdW1lbnQuY3Jl"
    "YXRlRWxlbWVudCgidGV4dGFyZWEiKTsKICAgICAgdGEudmFsdWUgPSB0OyB0YS5zdHlsZS5wb3NpdGlvbiA9ICJmaXhlZCI7IHRhLnN0eWxlLm9wYWNp"
    "dHkgPSAiMCI7CiAgICAgIGRvY3VtZW50LmJvZHkuYXBwZW5kQ2hpbGQodGEpOyB0YS5zZWxlY3QoKTsKICAgICAgY29uc3Qgb2sgPSBkb2N1bWVudC5l"
    "eGVjQ29tbWFuZCgiY29weSIpOyB0YS5yZW1vdmUoKTsgcmV0dXJuIG9rOwogICAgfWNhdGNoKGUyKXsgcmV0dXJuIGZhbHNlOyB9CiAgfQp9CmRvY3Vt"
    "ZW50LmFkZEV2ZW50TGlzdGVuZXIoImNsaWNrIiwgYXN5bmMgKGUpPT57CiAgY29uc3QgYnRuID0gZS50YXJnZXQuY2xvc2VzdCgiLmNidG4iKTsgaWYo"
    "IWJ0bikgcmV0dXJuOwogIGNvbnN0IHdyYXAgPSBidG4uY2xvc2VzdCgiLmNvZGV3cmFwIik7IGlmKCF3cmFwKSByZXR1cm47CiAgY29uc3QgY29kZUVs"
    "ID0gd3JhcC5xdWVyeVNlbGVjdG9yKCJwcmUgY29kZSIpOwogIGNvbnN0IGNvZGUgPSBjb2RlRWwgPyBjb2RlRWwudGV4dENvbnRlbnQgOiAiIjsKICBp"
    "ZihidG4uY2xhc3NMaXN0LmNvbnRhaW5zKCJjb3B5YnRuIikpewogICAgY29uc3Qgb2sgPSBhd2FpdCBjb3B5VGV4dChjb2RlKTsKICAgIGNvbnN0IG9s"
    "ZCA9IGJ0bi50ZXh0Q29udGVudDsKICAgIGJ0bi50ZXh0Q29udGVudCA9IG9rID8gIkNvcGllZCBcdTI3MTMiIDogIkNvcHkgZmFpbGVkIjsKICAgIGJ0"
    "bi5jbGFzc0xpc3QudG9nZ2xlKCJvayIsIG9rKTsKICAgIHNldFRpbWVvdXQoKCk9PnsgYnRuLnRleHRDb250ZW50ID0gb2xkOyBidG4uY2xhc3NMaXN0"
    "LnJlbW92ZSgib2siKTsgfSwgMTQwMCk7CiAgfSBlbHNlIGlmKGJ0bi5jbGFzc0xpc3QuY29udGFpbnMoInJ1bmJ0biIpKXsKICAgIGJ0bi5kaXNhYmxl"
    "ZCA9IHRydWU7CiAgICB0cnl7CiAgICAgIGNvbnN0IHIgPSBhd2FpdCBwb3N0KCIvYXBpL2FydGlmYWN0cyIsIHtjb250ZW50OiBjb2RlfSk7CiAgICAg"
    "IHdpbmRvdy5vcGVuKHIudXJsLCAiX2JsYW5rIik7CiAgICB9Y2F0Y2goZXJyKXsgdG9hc3QoIkNvdWxkIG5vdCBjcmVhdGUgdGhlIGFwcCBmaWxlOiAi"
    "ICsgZXJyLm1lc3NhZ2UsICJlcnIiKTsgfQogICAgYnRuLmRpc2FibGVkID0gZmFsc2U7CiAgfQp9KTsKCi8qIC0tLS0tLS0tLS0gbGlnaHR3ZWlnaHQg"
    "bWFya2Rvd24gLS0tLS0tLS0tLSAqLwpmdW5jdGlvbiBtZChzcmMpewogIGlmKCFzcmMpIHJldHVybiAiIjsKICBjb25zdCBibG9ja3MgPSBbXTsgLy8g"
    "c3Rhc2ggY29kZSBmZW5jZXMKICBsZXQgcyA9IHNyYy5yZXBsYWNlKC9gYGAoXHcqKVxuPyhbXHNcU10qPylgYGAvZywgKG0sIGxhbmcsIGNvZGUpPT57"
    "CiAgICBjb25zdCByYXcgPSBjb2RlLnJlcGxhY2UoL1xuJC8sIiIpOwogICAgY29uc3QgcnVuID0gbG9va3NIdG1sKGxhbmcsIHJhdykKICAgICAgPyBg"
    "PGJ1dHRvbiBjbGFzcz0iY2J0biBydW5idG4iIHRpdGxlPSJTYXZlIGFzIGFuIC5odG1sIGZpbGUgYW5kIG9wZW4gaXQgaW4gYSBuZXcgdGFiIj4mIzk2"
    "NTQ7IFJ1biBhcHA8L2J1dHRvbj5gIDogIiI7CiAgICBibG9ja3MucHVzaChgPGRpdiBjbGFzcz0iY29kZXdyYXAiPjxkaXYgY2xhc3M9ImNvZGViYXIi"
    "PmArCiAgICAgIGA8c3BhbiBjbGFzcz0iY29kZWxhbmciPiR7ZXNjKGxhbmd8fCJjb2RlIil9PC9zcGFuPjxzcGFuIHN0eWxlPSJmbGV4OjEiPjwvc3Bh"
    "bj5gKwogICAgICBydW4rYDxidXR0b24gY2xhc3M9ImNidG4gY29weWJ0biIgdGl0bGU9IkNvcHkgdGhpcyBjb2RlIj5Db3B5PC9idXR0b24+PC9kaXY+"
    "YCsKICAgICAgYDxwcmU+PGNvZGU+JHtlc2MocmF3KX08L2NvZGU+PC9wcmU+PC9kaXY+YCk7CiAgICByZXR1cm4gYFx1MDAwMCR7YmxvY2tzLmxlbmd0"
    "aC0xfVx1MDAwMGA7IH0pOwogIHMgPSBlc2Mocyk7CiAgLy8gaW1hZ2VzIHRoZW4gbGlua3MKICBzID0gcy5yZXBsYWNlKC8hXFsoW15cXV0qKVxdXCgo"
    "W14pXHNdKylcKS9nLAogICAgICAobSxhLHUpPT5gPGltZyBhbHQ9IiR7YX0iIHNyYz0iJHt1fSI+YCk7CiAgcyA9IHMucmVwbGFjZSgvXFsoW15cXV0r"
    "KVxdXCgoW14pXHNdKylcKS9nLAogICAgICAobSx0LHUpPT5gPGEgaHJlZj0iJHt1fSIgdGFyZ2V0PSJfYmxhbmsiIHJlbD0ibm9vcGVuZXIiPiR7dH08"
    "L2E+YCk7CiAgcyA9IHMucmVwbGFjZSgvYChbXmBdKylgL2csIChtLGMpPT5gPGNvZGU+JHtjfTwvY29kZT5gKTsKICBzID0gcy5yZXBsYWNlKC9cKlwq"
    "KFteKl0rKVwqXCovZywgIjxzdHJvbmc+JDE8L3N0cm9uZz4iKTsKICBzID0gcy5yZXBsYWNlKC8oXnxbXipdKVwqKFteKlxuXSspXCovZywgIiQxPGVt"
    "PiQyPC9lbT4iKTsKICAvLyBoZWFkaW5ncwogIHMgPSBzLnJlcGxhY2UoL14jIyNccysoLiopJC9nbSwgIjxoMz4kMTwvaDM+IikKICAgICAgIC5yZXBs"
    "YWNlKC9eIyNccysoLiopJC9nbSwgIjxoMj4kMTwvaDI+IikKICAgICAgIC5yZXBsYWNlKC9eI1xzKyguKikkL2dtLCAiPGgxPiQxPC9oMT4iKTsKICAv"
    "LyBsaXN0cwogIHMgPSBzLnJlcGxhY2UoLyg/Ol58XG4pKCg/OlxzKlstKl1ccysuKig/OlxufCQpKSspL2csIChtLCBsaXN0KT0+ewogICAgY29uc3Qg"
    "aXRlbXMgPSBsaXN0LnRyaW0oKS5zcGxpdCgiXG4iKS5tYXAobD0+CiAgICAgICI8bGk+IitsLnJlcGxhY2UoL15ccypbLSpdXHMrLywiIikrIjwvbGk+"
    "Iikuam9pbigiIik7IHJldHVybiAiXG48dWw+IitpdGVtcysiPC91bD4iOyB9KTsKICBzID0gcy5yZXBsYWNlKC8oPzpefFxuKSgoPzpccypcZCtcLlxz"
    "Ky4qKD86XG58JCkpKykvZywgKG0sIGxpc3QpPT57CiAgICBjb25zdCBpdGVtcyA9IGxpc3QudHJpbSgpLnNwbGl0KCJcbiIpLm1hcChsPT4KICAgICAg"
    "IjxsaT4iK2wucmVwbGFjZSgvXlxzKlxkK1wuXHMrLywiIikrIjwvbGk+Iikuam9pbigiIik7IHJldHVybiAiXG48b2w+IitpdGVtcysiPC9vbD4iOyB9"
    "KTsKICAvLyBwYXJhZ3JhcGhzCiAgcyA9IHMuc3BsaXQoL1xuezIsfS8pLm1hcChwPT57CiAgICBwID0gcC50cmltKCk7IGlmKCFwKSByZXR1cm4gIiI7"
    "CiAgICBpZigvXlx1MDAwMFxkK1x1MDAwMCQvLnRlc3QocCkpIHJldHVybiBwOyAgIC8vIGNvZGUgYmxvY2sgcGxhY2Vob2xkZXIKICAgIGlmKC9ePCho"
    "XGR8dWx8b2x8cHJlfHRhYmxlfGltZ3xibG9ja3F1b3RlKS8udGVzdChwKSkgcmV0dXJuIHA7CiAgICByZXR1cm4gIjxwPiIrcC5yZXBsYWNlKC9cbi9n"
    "LCI8YnI+IikrIjwvcD4iOwogIH0pLmpvaW4oIlxuIik7CiAgcyA9IHMucmVwbGFjZSgvXHUwMDAwKFxkKylcdTAwMDAvZywgKG0saSk9PmJsb2Nrc1sr"
    "aV0pOwogIHJldHVybiBzOwp9CgovKiAtLS0tLS0tLS0tIG5hdmlnYXRpb24gLS0tLS0tLS0tLSAqLwpjb25zdCBsb2FkZXJzID0ge307CmxldCBjdXJy"
    "ZW50VmlldyA9ICJkYXNoYm9hcmQiOwpmdW5jdGlvbiBzaG93KHZpZXcpewogIGN1cnJlbnRWaWV3ID0gdmlldzsKICBkb2N1bWVudC5xdWVyeVNlbGVj"
    "dG9yQWxsKCcubmF2IGJ1dHRvbltkYXRhLXZpZXddJykuZm9yRWFjaChiPT4KICAgIGIuY2xhc3NMaXN0LnRvZ2dsZSgiYWN0aXZlIiwgYi5kYXRhc2V0"
    "LnZpZXc9PT12aWV3KSk7CiAgZG9jdW1lbnQucXVlcnlTZWxlY3RvckFsbCgnLnZpZXdbZGF0YS12aWV3XScpLmZvckVhY2gocz0+CiAgICBzLmhpZGRl"
    "biA9IHMuZGF0YXNldC52aWV3IT09dmlldyk7CiAgJCgic2lkZSIpLmNsYXNzTGlzdC5yZW1vdmUoIm9wZW4iKTsKICBpZihsb2FkZXJzW3ZpZXddKSBs"
    "b2FkZXJzW3ZpZXddKCk7Cn0KZG9jdW1lbnQucXVlcnlTZWxlY3RvckFsbCgnLm5hdiBidXR0b25bZGF0YS12aWV3XScpLmZvckVhY2goYj0+CiAgYi5v"
    "bmNsaWNrID0gKCk9PnNob3coYi5kYXRhc2V0LnZpZXcpKTsKJCgibWVudUJ0biIpLm9uY2xpY2sgPSAoKT0+ICQoInNpZGUiKS5jbGFzc0xpc3QudG9n"
    "Z2xlKCJvcGVuIik7CgovKiAtLS0tLS0tLS0tIHRoZW1lIC0tLS0tLS0tLS0gKi8KZnVuY3Rpb24gYXBwbHlUaGVtZSh0KXsKICBkb2N1bWVudC5kb2N1"
    "bWVudEVsZW1lbnQuZGF0YXNldC50aGVtZSA9IHQ7CiAgJCgidGhlbWVMYWJlbCIpLnRleHRDb250ZW50ID0gdD09PSJkYXJrIiA/ICJMaWdodCBtb2Rl"
    "IiA6ICJEYXJrIG1vZGUiOwp9CiQoInRoZW1lQnRuIikub25jbGljayA9IGFzeW5jICgpPT57CiAgY29uc3QgbmV4dCA9IGRvY3VtZW50LmRvY3VtZW50"
    "RWxlbWVudC5kYXRhc2V0LnRoZW1lPT09ImRhcmsiID8gImxpZ2h0IjoiZGFyayI7CiAgYXBwbHlUaGVtZShuZXh0KTsgc3RhdGUuc2V0dGluZ3MudGhl"
    "bWUgPSBuZXh0OwogIHRyeXsgYXdhaXQgcG9zdCgiL2FwaS9zZXR0aW5ncyIsIHt0aGVtZTogbmV4dH0pOyB9Y2F0Y2goZSl7fQp9OwoKLyogPT09PT09"
    "PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09CiAgIERBU0hCT0FSRAogICA9PT09PT09PT09PT09PT09"
    "PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT0gKi8KZnVuY3Rpb24gZG9udXQocGN0LCBjZW50ZXJCaWcsIGNlbnRlclNt"
    "YWxsKXsKICBjb25zdCBSID0gNTIsIEMgPSAyKk1hdGguUEkqUiwgb2ZmID0gQyooMSAtIE1hdGgubWF4KDAsTWF0aC5taW4oMSxwY3QpKSk7CiAgcmV0"
    "dXJuIGA8c3ZnIHdpZHRoPSIxMzIiIGhlaWdodD0iMTMyIiB2aWV3Qm94PSIwIDAgMTMyIDEzMiI+CiAgICA8Y2lyY2xlIGN4PSI2NiIgY3k9IjY2IiBy"
    "PSIke1J9IiBmaWxsPSJub25lIiBzdHJva2U9InZhcigtLWxpbmUpIiBzdHJva2Utd2lkdGg9IjExIi8+CiAgICA8Y2lyY2xlIGN4PSI2NiIgY3k9IjY2"
    "IiByPSIke1J9IiBmaWxsPSJub25lIiBzdHJva2U9InZhcigtLXNpZ25hbCkiIHN0cm9rZS13aWR0aD0iMTEiCiAgICAgIHN0cm9rZS1saW5lY2FwPSJy"
    "b3VuZCIgc3Ryb2tlLWRhc2hhcnJheT0iJHtDfSIgc3Ryb2tlLWRhc2hvZmZzZXQ9IiR7b2ZmfSIKICAgICAgc3R5bGU9InRyYW5zaXRpb246c3Ryb2tl"
    "LWRhc2hvZmZzZXQgLjhzIGVhc2UiLz4KICA8L3N2Zz48ZGl2IGNsYXNzPSJsYmwiPjxiPiR7Y2VudGVyQmlnfTwvYj48c3Bhbj4ke2NlbnRlclNtYWxs"
    "fTwvc3Bhbj48L2Rpdj5gOwp9Cgphc3luYyBmdW5jdGlvbiBsb2FkRGFzaGJvYXJkKCl7CiAgdHJ5ewogICAgY29uc3QgZCA9IGF3YWl0IGFwaSgiL2Fw"
    "aS9zeXN0ZW0iKTsKICAgIHN0YXRlLnN5c3RlbSA9IGQ7IHJlbmRlclN5c3RlbShkKTsKICB9Y2F0Y2goZSl7ICQoInRpZXJOb3RlIikudGV4dENvbnRl"
    "bnQgPSAiQ291bGQgbm90IHJlYWQgc3lzdGVtIGluZm86ICIrZS5tZXNzYWdlOyB9Cn0KZnVuY3Rpb24gcmVuZGVyU3lzdGVtKGQpewogIGNvbnN0IGh3"
    "ID0gZC5oYXJkd2FyZSwgcmVjID0gZC5yZWNvbW1lbmRhdGlvbjsKICAkKCJicmFuZFZlciIpLnRleHRDb250ZW50ID0gInYiK2QudmVyc2lvbjsKICAv"
    "LyBnYXVnZTogdXNhYmxlIG1lbW9yeSBhcyBzaGFyZSBvZiB0b3RhbCBSQU0KICBjb25zdCBwY3QgPSBody5yYW1fZ2IgPyBody51c2FibGVfZ2IgLyBo"
    "dy5yYW1fZ2IgOiAwOwogICQoImdhdWdlIikuaW5uZXJIVE1MID0gZG9udXQocGN0LCBmbXRHQihody51c2FibGVfZ2IpKyI8c21hbGwgc3R5bGU9J2Zv"
    "bnQtc2l6ZToxM3B4Jz5HQjwvc21hbGw+IiwKICAgICJ1c2FibGUiKTsKICBjb25zdCBncHUgPSBody5ncHVzICYmIGh3LmdwdXMubGVuZ3RoID8gaHcu"
    "Z3B1c1swXSA6IG51bGw7CiAgY29uc3QgcGx1cmFsID0gKG4sdyk9PiBuKyIgIit3KyhuPT09MT8iIjoicyIpOwogIGNvbnN0IHNwZWNzID0gWwogICAg"
    "WyJTeXN0ZW0iLCBody5vc19wcmV0dHksIGh3LmFyY2hdLAogICAgWyJNZW1vcnkiLCBmbXRHQihody5yYW1fZ2IpKyIgR0IiLCBmbXRHQihody5yYW1f"
    "dXNlZF9nYikrIiBHQiBpbiB1c2UiXSwKICAgIFsiUHJvY2Vzc29yIiwgcGx1cmFsKGh3LmNwdV9jb3Jlc3x8MCwiY29yZSIpLCBwbHVyYWwoaHcuY3B1"
    "X3RocmVhZHN8fDAsInRocmVhZCIpXSwKICAgIFsiR3JhcGhpY3MiLCBncHUgPyAoZ3B1LmtpbmQ9PT0iYXBwbGUiPyJBcHBsZSBHUFUiOmdwdS5uYW1l"
    "LnNsaWNlKDAsMjIpKSA6ICJDUFUgb25seSIsCiAgICAgICBncHUgPyBmbXRHQihncHUudnJhbV9nYikrIiBHQiAiKyhncHUua2luZD09PSJhcHBsZSI/"
    "InVuaWZpZWQiOiJWUkFNIikgOiBody5iYWNrZW5kXSwKICAgIFsiRnJlZSBkaXNrIiwgZm10R0IoaHcuZGlza19mcmVlX2diKSsiIEdCIiwgImZvciBt"
    "b2RlbHMiXSwKICAgIFsiQmFja2VuZCIsIGh3LmJhY2tlbmQudG9VcHBlckNhc2UoKSwgaHcuYXBwbGVfc2lsaWNvbj8iTWV0YWwiOiJhY2NlbGVyYXRp"
    "b24iXSwKICBdOwogICQoInNwZWNzIikuaW5uZXJIVE1MID0gc3BlY3MubWFwKChbayx2LHNdKT0+CiAgICBgPGRpdiBjbGFzcz0ic3BlYyI+PGRpdiBj"
    "bGFzcz0iayI+JHtrfTwvZGl2PjxkaXYgY2xhc3M9InYiPiR7ZXNjKHYpfQogICAgIDxzbWFsbD4ke2VzYyhzfHwiIil9PC9zbWFsbD48L2Rpdj48L2Rp"
    "dj5gKS5qb2luKCIiKTsKICAkKCJ0aWVyTm90ZSIpLmlubmVySFRNTCA9ICI8Yj4iK2VzYyhyZWMudGllci5zcGxpdCgi4oCUIilbMF0udHJpbSgpKSsi"
    "PC9iPiDigJQgIisKICAgIGVzYyhyZWMudGllci5zcGxpdCgi4oCUIikuc2xpY2UoMSkuam9pbigi4oCUIikudHJpbSgpIHx8IHJlYy50aWVyKTsKCiAg"
    "Y29uc3QgaW5zdGFsbGVkTmFtZXMgPSBuZXcgU2V0KHN0YXRlLmluc3RhbGxlZC5tYXAobT0+bS5uYW1lKSk7CiAgY29uc3QgcGlja3MgPSByZWMucGlj"
    "a3MgfHwgW107CiAgJCgicmVjTGlzdCIpLmlubmVySFRNTCA9IHBpY2tzLmxlbmd0aCA/ICIiIDoKICAgICI8ZGl2IGNsYXNzPSdtdXRlZCc+Tm8gbW9k"
    "ZWxzIGZpdCB0aGUgZGV0ZWN0ZWQgbWVtb3J5LiBUcnkgdGhlIE1vZGVscyBwYWdlLjwvZGl2PiI7CiAgcGlja3MuZm9yRWFjaCgobSxpKT0+ewogICAg"
    "Y29uc3QgaGF2ZSA9IGluc3RhbGxlZE5hbWVzLmhhcyhtLmlkKTsKICAgIGNvbnN0IHJvdyA9IGVsKCJkaXYiLCJyZWNpdGVtIisoaT09PTA/IiBiZXN0"
    "IjoiIikpOwogICAgcm93LmlubmVySFRNTCA9IGA8ZGl2IGNsYXNzPSJyYW5rIj4ke1N0cmluZyhpKzEpLnBhZFN0YXJ0KDIsIjAiKX08L2Rpdj4KICAg"
    "ICAgPGRpdiBjbGFzcz0iaW5mbyI+PGRpdiBjbGFzcz0ibm0iPiR7ZXNjKG0ubmFtZSl9CiAgICAgICAgJHtpPT09MD8nPHNwYW4gY2xhc3M9ImNoaXAg"
    "aGwiPmJlc3QgZml0PC9zcGFuPic6Jyd9CiAgICAgICAgJHttLnRhZ3Muc2xpY2UoMCwyKS5tYXAodD0+YDxzcGFuIGNsYXNzPSJjaGlwIj4ke3R9PC9z"
    "cGFuPmApLmpvaW4oIiIpfTwvZGl2PgogICAgICAgIDxkaXYgY2xhc3M9ImRzIj4ke2VzYyhtLmRlc2MpfTwvZGl2PjwvZGl2PgogICAgICA8ZGl2IGNs"
    "YXNzPSJzaXplIj4ke2ZtdEdCKG0uc2l6ZV9nYil9IEdCPGJyPjxzcGFuIGNsYXNzPSJkaW0iPn4ke2ZtdEdCKG0ubmVlZF9nYil9IEdCIHJhbTwvc3Bh"
    "bj48L2Rpdj5gOwogICAgY29uc3QgYWN0ID0gZWwoImRpdiIpOyBhY3Quc3R5bGUuZmxleD0ibm9uZSI7CiAgICBpZihoYXZlKXsgYWN0LmlubmVySFRN"
    "TCA9ICc8c3BhbiBjbGFzcz0iaW5zdGFsbGVkLWJhZGdlIj7inJMgaW5zdGFsbGVkPC9zcGFuPic7IH0KICAgIGVsc2V7IGNvbnN0IGIgPSBlbCgiYnV0"
    "dG9uIiwiYnRuIHByaW1hcnkgc20iLCJEb3dubG9hZCIpOwogICAgICBiLm9uY2xpY2sgPSAoKT0+cHVsbE1vZGVsKG0uaWQsIGIpOyBhY3QuYXBwZW5k"
    "Q2hpbGQoYik7IH0KICAgIHJvdy5hcHBlbmRDaGlsZChhY3QpOyAkKCJyZWNMaXN0IikuYXBwZW5kQ2hpbGQocm93KTsKICB9KTsKfQokKCJyZXNjYW5C"
    "dG4iKS5vbmNsaWNrID0gKCk9PnsgJCgiZ2F1Z2UiKS5pbm5lckhUTUw9JzxkaXYgY2xhc3M9InNwaW4iPjwvZGl2Pic7CiAgbG9hZERhc2hib2FyZCgp"
    "OyB0b2FzdCgiUmVzY2FubmluZyBoYXJkd2FyZeKApiIpOyB9Owpsb2FkZXJzLmRhc2hib2FyZCA9IGxvYWREYXNoYm9hcmQ7CgovKiA9PT09PT09PT09"
    "PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT0KICAgTU9ERUxTCiAgID09PT09PT09PT09PT09PT09PT09PT09"
    "PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PSAqLwphc3luYyBmdW5jdGlvbiBsb2FkTW9kZWxzKCl7CiAgY29uc3QgaG9zdCA9ICQo"
    "Im1vZGVsc0JvZHkiKTsKICBob3N0LmlubmVySFRNTCA9ICc8ZGl2IGNsYXNzPSJyb3ciIHN0eWxlPSJnYXA6OXB4Ij48ZGl2IGNsYXNzPSJzcGluIj48"
    "L2Rpdj4nKwogICAgJzxzcGFuIGNsYXNzPSJtdXRlZCI+TG9hZGluZyBtb2RlbHPigKY8L3NwYW4+PC9kaXY+JzsKICBsZXQgaW5zdDsKICB0cnl7IGlu"
    "c3QgPSBhd2FpdCBhcGkoIi9hcGkvbW9kZWxzL2luc3RhbGxlZCIpOyB9CiAgY2F0Y2goZSl7IGhvc3QuaW5uZXJIVE1MID0gZXJyQ2FyZChlLm1lc3Nh"
    "Z2UpOyByZXR1cm47IH0KICBzdGF0ZS5pbnN0YWxsZWQgPSBpbnN0Lm1vZGVscyB8fCBbXTsKICBsZXQgaHRtbCA9ICIiOwogIGlmKCFpbnN0Lm9sbGFt"
    "YS51cCl7CiAgICBodG1sICs9IG9sbGFtYVNldHVwQ2FyZCgpOwogIH0KICAvLyBpbnN0YWxsZWQKICBodG1sICs9IGA8ZGl2IGNsYXNzPSJzZWN0aW9u"
    "LXRpdGxlIj5JbnN0YWxsZWQke2luc3Qub2xsYW1hLnVwPwogICAgIiDCtyAiK3N0YXRlLmluc3RhbGxlZC5sZW5ndGg6IiJ9PC9kaXY+YDsKICBpZihz"
    "dGF0ZS5pbnN0YWxsZWQubGVuZ3RoKXsKICAgIGh0bWwgKz0gJzxkaXYgY2xhc3M9Im1ncmlkIiBpZD0iaW5zdEdyaWQiPjwvZGl2Pic7CiAgfSBlbHNl"
    "IGlmKGluc3Qub2xsYW1hLnVwKXsKICAgIGh0bWwgKz0gJzxkaXYgY2xhc3M9ImVtcHR5Ij48ZGl2IGNsYXNzPSJiaWciPuKXtTwvZGl2Pk5vIG1vZGVs"
    "cyB5ZXQg4oCUIGRvd25sb2FkIG9uZSBiZWxvdyBvciBmcm9tIHRoZSBEYXNoYm9hcmQuPC9kaXY+JzsKICB9CiAgaHRtbCArPSAnPGRpdiBjbGFzcz0i"
    "c2VjdGlvbi10aXRsZSI+Q3VyYXRlZCBmb3IgeW91ciBoYXJkd2FyZTwvZGl2PjxkaXYgY2xhc3M9Im1ncmlkIiBpZD0iY2F0R3JpZCI+PC9kaXY+JzsK"
    "ICBodG1sICs9ICc8ZGl2IGlkPSJzZWFyY2hSZXN1bHRzIj48L2Rpdj4nOwogIGhvc3QuaW5uZXJIVE1MID0gaHRtbDsKCiAgaWYoc3RhdGUuaW5zdGFs"
    "bGVkLmxlbmd0aCl7CiAgICBjb25zdCBnID0gJCgiaW5zdEdyaWQiKTsKICAgIHN0YXRlLmluc3RhbGxlZC5mb3JFYWNoKG09PnsKICAgICAgY29uc3Qg"
    "YyA9IGVsKCJkaXYiLCJtY2FyZCIpOwogICAgICBjLmlubmVySFRNTCA9IGA8ZGl2IGNsYXNzPSJ0b3AiPjxkaXY+PGRpdiBjbGFzcz0ibm0iPiR7ZXNj"
    "KG0ubmFtZSl9PC9kaXY+CiAgICAgICAgPGRpdiBjbGFzcz0iaWQiPiR7ZXNjKFttLnBhcmFtcyxtLnF1YW50LG0uZmFtaWx5XS5maWx0ZXIoQm9vbGVh"
    "bikuam9pbigiIMK3ICIpfHwibW9kZWwiKX08L2Rpdj48L2Rpdj48L2Rpdj4KICAgICAgICA8ZGl2IGNsYXNzPSJmb290Ij48c3BhbiBjbGFzcz0iaW5z"
    "dGFsbGVkLWJhZGdlIj7inJMgJHtmbXRHQihtLnNpemVfZ2IpfSBHQiBvbiBkaXNrPC9zcGFuPjwvZGl2PmA7CiAgICAgIGNvbnN0IGJ0biA9IGVsKCJi"
    "dXR0b24iLCJidG4gZGFuZ2VyIHNtIiwiUmVtb3ZlIik7CiAgICAgIGJ0bi5vbmNsaWNrID0gKCk9PnJlbW92ZU1vZGVsKG0ubmFtZSwgYyk7CiAgICAg"
    "IGMucXVlcnlTZWxlY3RvcigiLmZvb3QiKS5hcHBlbmRDaGlsZChidG4pOyBnLmFwcGVuZENoaWxkKGMpOwogICAgfSk7CiAgfQogIC8vIGN1cmF0ZWQg"
    "Y2F0YWxvZwogIHRyeXsKICAgIGNvbnN0IGNhdCA9IGF3YWl0IGFwaSgiL2FwaS9tb2RlbHMvY2F0YWxvZyIpOwogICAgY29uc3QgaW5zdGFsbGVkTmFt"
    "ZXMgPSBuZXcgU2V0KHN0YXRlLmluc3RhbGxlZC5tYXAobT0+bS5uYW1lKSk7CiAgICBjb25zdCBmaXQgPSAoY2F0LmNhdGFsb2d8fFtdKS5maWx0ZXIo"
    "bT0+CiAgICAgIG0ubmVlZF9nYiA8PSBjYXQucmVjb21tZW5kYXRpb24udXNhYmxlX2diICsgMC41KTsKICAgIHJlbmRlck1vZGVsQ2FyZHMoJCgiY2F0"
    "R3JpZCIpLCBmaXQsIGluc3RhbGxlZE5hbWVzKTsKICB9Y2F0Y2goZSl7fQp9CmZ1bmN0aW9uIHJlbmRlck1vZGVsQ2FyZHMoZ3JpZCwgbW9kZWxzLCBp"
    "bnN0YWxsZWROYW1lcyl7CiAgZ3JpZC5pbm5lckhUTUwgPSAiIjsKICBpZighbW9kZWxzLmxlbmd0aCl7IGdyaWQuaW5uZXJIVE1MID0gJzxkaXYgY2xh"
    "c3M9Im11dGVkIj5Ob3RoaW5nIHRvIHNob3cuPC9kaXY+JzsgcmV0dXJuOyB9CiAgbW9kZWxzLmZvckVhY2gobT0+ewogICAgY29uc3QgaGF2ZSA9IGlu"
    "c3RhbGxlZE5hbWVzLmhhcyhtLmlkKTsKICAgIGNvbnN0IGMgPSBlbCgiZGl2IiwibWNhcmQiKTsKICAgIGMuaW5uZXJIVE1MID0gYDxkaXYgY2xhc3M9"
    "InRvcCI+PGRpdj48ZGl2IGNsYXNzPSJubSI+JHtlc2MobS5uYW1lKX08L2Rpdj4KICAgICAgPGRpdiBjbGFzcz0iaWQiPiR7ZXNjKG0uaWQpfTwvZGl2"
    "PjwvZGl2PgogICAgICA8ZGl2IGNsYXNzPSJ0YWdzIj4keyhtLnRhZ3N8fFtdKS5tYXAodD0+YDxzcGFuIGNsYXNzPSJjaGlwIj4ke3R9PC9zcGFuPmAp"
    "LmpvaW4oIiIpfTwvZGl2PjwvZGl2PgogICAgICA8ZGl2IGNsYXNzPSJkcyI+JHtlc2MobS5kZXNjfHwiIil9PC9kaXY+CiAgICAgIDxkaXYgY2xhc3M9"
    "ImZvb3QiPjxzcGFuIGNsYXNzPSJkaW0gbW9ubyIgc3R5bGU9ImZvbnQtc2l6ZToxMXB4Ij4KICAgICAgICAke20uc2l6ZV9nYj9mbXRHQihtLnNpemVf"
    "Z2IpKyIgR0IiOiIifSR7bS5kb3dubG9hZHM/IuKGkyAiK20uZG93bmxvYWRzLnRvTG9jYWxlU3RyaW5nKCk6IiJ9PC9zcGFuPjwvZGl2PmA7CiAgICBj"
    "b25zdCBmb290ID0gYy5xdWVyeVNlbGVjdG9yKCIuZm9vdCIpOwogICAgaWYoaGF2ZSl7IGZvb3QuYXBwZW5kQ2hpbGQoZWwoInNwYW4iLCJpbnN0YWxs"
    "ZWQtYmFkZ2UiLCLinJMgaW5zdGFsbGVkIikpOyB9CiAgICBlbHNleyBjb25zdCBiID0gZWwoImJ1dHRvbiIsImJ0biBwcmltYXJ5IHNtIiwiRG93bmxv"
    "YWQiKTsKICAgICAgYi5vbmNsaWNrID0gKCk9PnB1bGxNb2RlbChtLmlkLCBiLCBjKTsgZm9vdC5hcHBlbmRDaGlsZChiKTsgfQogICAgZ3JpZC5hcHBl"
    "bmRDaGlsZChjKTsKICB9KTsKfQphc3luYyBmdW5jdGlvbiBkb01vZGVsU2VhcmNoKCl7CiAgY29uc3QgcXYgPSAkKCJtb2RlbFNlYXJjaCIpLnZhbHVl"
    "LnRyaW0oKTsgaWYoIXF2KSByZXR1cm47CiAgY29uc3QgYm94ID0gJCgic2VhcmNoUmVzdWx0cyIpOwogIGJveC5pbm5lckhUTUwgPSAnPGRpdiBjbGFz"
    "cz0ic2VjdGlvbi10aXRsZSI+U2VhcmNoIHJlc3VsdHM8L2Rpdj4nKwogICAgJzxkaXYgY2xhc3M9InJvdyI+PGRpdiBjbGFzcz0ic3BpbiI+PC9kaXY+"
    "PHNwYW4gY2xhc3M9Im11dGVkIj5TZWFyY2hpbmfigKY8L3NwYW4+PC9kaXY+JzsKICB0cnl7CiAgICBjb25zdCByID0gYXdhaXQgYXBpKCIvYXBpL21v"
    "ZGVscy9zZWFyY2g/cT0iK2VuY29kZVVSSUNvbXBvbmVudChxdikpOwogICAgY29uc3QgaW5zdGFsbGVkTmFtZXMgPSBuZXcgU2V0KHN0YXRlLmluc3Rh"
    "bGxlZC5tYXAobT0+bS5uYW1lKSk7CiAgICBib3guaW5uZXJIVE1MID0gJzxkaXYgY2xhc3M9InNlY3Rpb24tdGl0bGUiPlNlYXJjaCByZXN1bHRzPC9k"
    "aXY+JzsKICAgIGNvbnN0IGFsbCA9IFsuLi4oci5jYXRhbG9nfHxbXSksIC4uLihyLmh1Z2dpbmdmYWNlfHxbXSldOwogICAgaWYoIWFsbC5sZW5ndGgp"
    "eyBib3guaW5uZXJIVE1MICs9ICc8ZGl2IGNsYXNzPSJtdXRlZCI+Tm8gbWF0Y2hlcy4gVHJ5IGFub3RoZXIgdGVybSwgJysKICAgICAgJ29yIHBhc3Rl"
    "IGFuIGV4YWN0IE9sbGFtYSB0YWcgbGlrZSA8c3BhbiBjbGFzcz0ibW9ubyI+cXdlbjM6OGI8L3NwYW4+LjwvZGl2Pic7CiAgICAgIGNvbnN0IGIgPSBl"
    "bCgiYnV0dG9uIiwiYnRuIHNtIGdob3N0IiwiUHVsbCBcIiIrcXYrIlwiIGFueXdheSIpOwogICAgICBiLnN0eWxlLm1hcmdpblRvcD0iMTBweCI7IGIu"
    "b25jbGljaz0oKT0+cHVsbE1vZGVsKHF2LGIpOyBib3guYXBwZW5kQ2hpbGQoYik7IHJldHVybjsgfQogICAgY29uc3QgZyA9IGVsKCJkaXYiLCJtZ3Jp"
    "ZCIpOyBib3guYXBwZW5kQ2hpbGQoZyk7CiAgICByZW5kZXJNb2RlbENhcmRzKGcsIGFsbCwgaW5zdGFsbGVkTmFtZXMpOwogIH1jYXRjaChlKXsgYm94"
    "LmlubmVySFRNTCArPSBlcnJDYXJkKGUubWVzc2FnZSk7IH0KfQokKCJtb2RlbFNlYXJjaEJ0biIpLm9uY2xpY2sgPSBkb01vZGVsU2VhcmNoOwokKCJt"
    "b2RlbFNlYXJjaCIpLmFkZEV2ZW50TGlzdGVuZXIoImtleWRvd24iLCBlPT57IGlmKGUua2V5PT09IkVudGVyIikgZG9Nb2RlbFNlYXJjaCgpOyB9KTsK"
    "CmFzeW5jIGZ1bmN0aW9uIHB1bGxNb2RlbChuYW1lLCBidG4sIGNhcmQpewogIGlmKGJ0bil7IGJ0bi5kaXNhYmxlZCA9IHRydWU7IGJ0bi50ZXh0Q29u"
    "dGVudCA9ICJTdGFydGluZ+KApiI7IH0KICBjb25zdCBob3N0ID0gY2FyZCB8fCAoYnRuICYmIGJ0bi5jbG9zZXN0KCIucmVjaXRlbSIpKSB8fCBkb2N1"
    "bWVudC5ib2R5OwogIGxldCBib3ggPSBob3N0LnF1ZXJ5U2VsZWN0b3IoIi5wdWxsYm94Iik7CiAgaWYoIWJveCl7IGJveCA9IGVsKCJkaXYiLCJwdWxs"
    "Ym94Iik7IGJveC5pbm5lckhUTUwgPQogICAgJzxkaXYgY2xhc3M9InN0YXQiPjxzcGFuIGNsYXNzPSJzIj5QcmVwYXJpbmfigKY8L3NwYW4+PHNwYW4g"
    "Y2xhc3M9InAiPjwvc3Bhbj48L2Rpdj4nKwogICAgJzxkaXYgY2xhc3M9ImJhciI+PGk+PC9pPjwvZGl2Pic7IGhvc3QuYXBwZW5kQ2hpbGQoYm94KTsg"
    "fQogIGNvbnN0IGJhciA9IGJveC5xdWVyeVNlbGVjdG9yKCJpIiksIHN0YXQgPSBib3gucXVlcnlTZWxlY3RvcigiLnMiKSwKICAgICAgICBwYyA9IGJv"
    "eC5xdWVyeVNlbGVjdG9yKCIucCIpOwogIHRyeXsKICAgIGF3YWl0IHN0cmVhbU5ESlNPTigiL2FwaS9tb2RlbHMvcHVsbCIsIHtuYW1lfSwgKG8pPT57"
    "CiAgICAgIGlmKG8udHlwZT09PSJlcnJvciIpeyB0aHJvdyBuZXcgRXJyb3Ioby5lcnJvcik7IH0KICAgICAgaWYoby50eXBlPT09ImRvbmUiKXsgc3Rh"
    "dC50ZXh0Q29udGVudCA9ICJEb25lIjsgYmFyLnN0eWxlLndpZHRoPSIxMDAlIjsgcmV0dXJuOyB9CiAgICAgIHN0YXQudGV4dENvbnRlbnQgPSBvLnN0"
    "YXR1cyB8fCAiRG93bmxvYWRpbmciOwogICAgICBpZihvLnRvdGFsICYmIG8uY29tcGxldGVkIT1udWxsKXsKICAgICAgICBjb25zdCBwID0gby5jb21w"
    "bGV0ZWQvby50b3RhbCoxMDA7IGJhci5zdHlsZS53aWR0aCA9IHAudG9GaXhlZCgxKSsiJSI7CiAgICAgICAgcGMudGV4dENvbnRlbnQgPSBmbXRHQihv"
    "LmNvbXBsZXRlZC8xZTkpKyIgLyAiK2ZtdEdCKG8udG90YWwvMWU5KSsiIEdCIjsKICAgICAgfQogICAgfSk7CiAgICB0b2FzdCgiRG93bmxvYWRlZCAi"
    "K25hbWUsICJvayIpOwogICAgc3RhdGUuaW5zdGFsbGVkLnB1c2goe25hbWUsIHNpemVfZ2I6MH0pOwogICAgaWYoY3VycmVudFZpZXc9PT0ibW9kZWxz"
    "IikgbG9hZE1vZGVscygpOwogICAgaWYoY3VycmVudFZpZXc9PT0iZGFzaGJvYXJkIikgbG9hZERhc2hib2FyZCgpOwogICAgcmVmcmVzaENoYXRNb2Rl"
    "bHMoKTsKICB9Y2F0Y2goZSl7CiAgICB0b2FzdCgiRG93bmxvYWQgZmFpbGVkOiAiK2UubWVzc2FnZSwgImVyciIpOwogICAgaWYoYnRuKXsgYnRuLmRp"
    "c2FibGVkPWZhbHNlOyBidG4udGV4dENvbnRlbnQ9IlJldHJ5IjsgfQogICAgc3RhdC50ZXh0Q29udGVudCA9ICJGYWlsZWQ6ICIrZS5tZXNzYWdlOwog"
    "IH0KfQphc3luYyBmdW5jdGlvbiByZW1vdmVNb2RlbChuYW1lLCBjYXJkKXsKICBjb25zdCBib2R5ID0gbW9kYWwoe3RpdGxlOiJSZW1vdmUgbW9kZWw/"
    "IiwKICAgIGJvZHlIVE1MOmA8cCBjbGFzcz0ibXV0ZWQiPkRlbGV0ZSA8c3BhbiBjbGFzcz0ibW9ubyI+JHtlc2MobmFtZSl9PC9zcGFuPiBmcm9tIGRp"
    "c2s/CiAgICAgIFlvdSBjYW4gZG93bmxvYWQgaXQgYWdhaW4gbGF0ZXIuPC9wPmAsCiAgICBhY3Rpb25zOlt7bGFiZWw6IkNhbmNlbCIsIG9uQ2xpY2s6"
    "Y2xvc2VNb2RhbH0sCiAgICAgIHtsYWJlbDoiUmVtb3ZlIiwgY2xzOiJkYW5nZXIiLCBvbkNsaWNrOmFzeW5jKCk9PnsKICAgICAgICBjbG9zZU1vZGFs"
    "KCk7CiAgICAgICAgdHJ5eyBhd2FpdCBwb3N0KCIvYXBpL21vZGVscy9kZWxldGUiLCB7bmFtZX0pOwogICAgICAgICAgdG9hc3QoIlJlbW92ZWQgIitu"
    "YW1lLCJvayIpOyBsb2FkTW9kZWxzKCk7IHJlZnJlc2hDaGF0TW9kZWxzKCk7IH0KICAgICAgICBjYXRjaChlKXsgdG9hc3QoZS5tZXNzYWdlLCJlcnIi"
    "KTsgfQogICAgICB9fV19KTsKfQpmdW5jdGlvbiBlcnJDYXJkKG1zZyl7IHJldHVybiBgPGRpdiBjbGFzcz0iY2FyZCIgc3R5bGU9ImJvcmRlci1jb2xv"
    "cjp2YXIoLS1yZWQpIj4KICA8YiBzdHlsZT0iY29sb3I6dmFyKC0tcmVkKSI+U29tZXRoaW5nIHdlbnQgd3Jvbmc8L2I+CiAgPHAgY2xhc3M9Im11dGVk"
    "IiBzdHlsZT0ibWFyZ2luOjZweCAwIDAiPiR7ZXNjKG1zZyl9PC9wPjwvZGl2PmA7IH0KZnVuY3Rpb24gb2xsYW1hU2V0dXBDYXJkKCl7CiAgY29uc3Qg"
    "aCA9IHN0YXRlLnN5c3RlbSA/IHN0YXRlLnN5c3RlbS5pbnN0YWxsX2hlbHAgOiBudWxsOwogIGNvbnN0IHN0ZXBzID0gaCA/IGguc3RlcHMgOiBbIklu"
    "c3RhbGwgT2xsYW1hIGZyb20gaHR0cHM6Ly9vbGxhbWEuY29tIl07CiAgcmV0dXJuIGA8ZGl2IGNsYXNzPSJjYXJkIHBhZC1sZyIgc3R5bGU9ImJvcmRl"
    "ci1jb2xvcjp2YXIoLS1zaWduYWwtZGltKTttYXJnaW4tYm90dG9tOjhweCI+CiAgICA8ZGl2IGNsYXNzPSJyb3ciIHN0eWxlPSJnYXA6MTBweDttYXJn"
    "aW4tYm90dG9tOjEwcHgiPgogICAgICA8c3BhbiBjbGFzcz0iZG90IG9mZiI+PC9zcGFuPjxiPk9sbGFtYSBpc24ndCBydW5uaW5nPC9iPjwvZGl2Pgog"
    "ICAgPHAgY2xhc3M9Im11dGVkIiBzdHlsZT0ibWFyZ2luOjAgMCAxMnB4Ij5IZW9ydGggdXNlcyBPbGxhbWEgdG8gcnVuIHRleHQgbW9kZWxzLgogICAg"
    "ICBJbnN0YWxsIGl0IG9uY2UgKCR7aD9lc2MoaC5vcyk6IiJ9KSwgdGhlbiB0aGlzIHBhZ2UgY29ubmVjdHMgYXV0b21hdGljYWxseS48L3A+CiAgICAk"
    "e3N0ZXBzLm1hcChzPT5gPGRpdiBjbGFzcz0ibW9ubyIgc3R5bGU9ImZvbnQtc2l6ZToxMnB4O2JhY2tncm91bmQ6dmFyKC0taW5rKTsKICAgICAgYm9y"
    "ZGVyOjFweCBzb2xpZCB2YXIoLS1saW5lKTtib3JkZXItcmFkaXVzOjhweDtwYWRkaW5nOjlweCAxMXB4O21hcmdpbi1ib3R0b206NnB4Ij4ke2VzYyhz"
    "KX08L2Rpdj5gKS5qb2luKCIiKX0KICAgIDxidXR0b24gY2xhc3M9ImJ0biBzbSBnaG9zdCIgc3R5bGU9Im1hcmdpbi10b3A6NnB4IiBvbmNsaWNrPSJs"
    "b2FkTW9kZWxzKCkiPkNoZWNrIGFnYWluPC9idXR0b24+CiAgPC9kaXY+YDsKfQpsb2FkZXJzLm1vZGVscyA9IGxvYWRNb2RlbHM7CgovKiA9PT09PT09"
    "PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT0KICAgQ0hBVAogICA9PT09PT09PT09PT09PT09PT09PT09"
    "PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT0gKi8KbGV0IHJhZ09uPWZhbHNlLCBhZ2VudE9uPWZhbHNlLCBsb29wT249ZmFsc2Us"
    "IGNvdW5jaWxPbj1mYWxzZSwgY29tcHV0ZXJPbj1mYWxzZSwgY29kZXJPbj1mYWxzZTsKZnVuY3Rpb24gc3luY1RvZ2dsZXMoKXsKICAkKCJhZ2VudFRv"
    "Z2dsZSIpLmNsYXNzTGlzdC50b2dnbGUoIm9uIixhZ2VudE9uKTsKICAkKCJsb29wVG9nZ2xlIikuY2xhc3NMaXN0LnRvZ2dsZSgib24iLGxvb3BPbik7"
    "CiAgJCgiY291bmNpbFRvZ2dsZSIpLmNsYXNzTGlzdC50b2dnbGUoIm9uIixjb3VuY2lsT24pOwogICQoImNvbXB1dGVyVG9nZ2xlIikuY2xhc3NMaXN0"
    "LnRvZ2dsZSgib24iLGNvbXB1dGVyT24pOwogICQoImNvZGVyVG9nZ2xlIikuY2xhc3NMaXN0LnRvZ2dsZSgib24iLGNvZGVyT24pOwp9CiQoInJhZ1Rv"
    "Z2dsZSIpLm9uY2xpY2sgPSAoKT0+eyByYWdPbj0hcmFnT247ICQoInJhZ1RvZ2dsZSIpLmNsYXNzTGlzdC50b2dnbGUoIm9uIixyYWdPbik7IH07CiQo"
    "ImFnZW50VG9nZ2xlIikub25jbGljayA9ICgpPT57IGFnZW50T249IWFnZW50T247CiAgaWYoYWdlbnRPbil7IGNvdW5jaWxPbj1mYWxzZTsgY29tcHV0"
    "ZXJPbj1mYWxzZTsgY29kZXJPbj1mYWxzZTsgfQogIGlmKCFhZ2VudE9uKSBsb29wT249ZmFsc2U7CiAgc3luY1RvZ2dsZXMoKTsgfTsKJCgibG9vcFRv"
    "Z2dsZSIpLm9uY2xpY2sgPSAoKT0+eyBsb29wT249IWxvb3BPbjsKICBpZihsb29wT24peyBhZ2VudE9uPXRydWU7IGNvdW5jaWxPbj1mYWxzZTsgY29t"
    "cHV0ZXJPbj1mYWxzZTsgY29kZXJPbj1mYWxzZTsgfQogIHN5bmNUb2dnbGVzKCk7IH07CiQoImNvdW5jaWxUb2dnbGUiKS5vbmNsaWNrID0gKCk9Pnsg"
    "Y291bmNpbE9uPSFjb3VuY2lsT247CiAgaWYoY291bmNpbE9uKXsgYWdlbnRPbj1mYWxzZTsgbG9vcE9uPWZhbHNlOyBjb21wdXRlck9uPWZhbHNlOyBj"
    "b2Rlck9uPWZhbHNlOyB9CiAgc3luY1RvZ2dsZXMoKTsgfTsKJCgiY29tcHV0ZXJUb2dnbGUiKS5vbmNsaWNrID0gYXN5bmMgKCk9PnsKICBpZighY29t"
    "cHV0ZXJPbil7CiAgICAvLyBtdXN0IGJlIGVuYWJsZWQgb24gdGhlIENvbXB1dGVyIHBhZ2UgZmlyc3QKICAgIGxldCBzPW51bGw7IHRyeXsgcz1hd2Fp"
    "dCBhcGkoIi9hcGkvY29tcHV0ZXIvc3RhdHVzIik7IH1jYXRjaChlKXt9CiAgICBpZighcyB8fCAhcy5pbnN0YWxsZWQgfHwgIXMuZW5hYmxlZCl7CiAg"
    "ICAgIHRvYXN0KHMgJiYgcy5pbnN0YWxsZWQgPyAiRW5hYmxlIGNvbXB1dGVyIGNvbnRyb2wgb24gdGhlIENvbXB1dGVyIHBhZ2UgZmlyc3QiCiAgICAg"
    "ICAgICAgICAgICAgICAgICAgICAgICAgOiAiSW5zdGFsbCBjb21wdXRlciBjb250cm9sIG9uIHRoZSBDb21wdXRlciBwYWdlIGZpcnN0IiwiZXJyIik7"
    "CiAgICAgIHNob3coImNvbXB1dGVyIik7IHJldHVybjsKICAgIH0KICAgIGNvbXB1dGVyT249dHJ1ZTsgYWdlbnRPbj1mYWxzZTsgbG9vcE9uPWZhbHNl"
    "OyBjb3VuY2lsT249ZmFsc2U7IGNvZGVyT249ZmFsc2U7CiAgfSBlbHNlIHsgY29tcHV0ZXJPbj1mYWxzZTsgfQogIHN5bmNUb2dnbGVzKCk7Cn07CiQo"
    "ImNvZGVyVG9nZ2xlIikub25jbGljayA9ICgpPT57CiAgaWYoY29kZXJPbil7IGNvZGVyT249ZmFsc2U7IHN5bmNUb2dnbGVzKCk7IHJldHVybjsgfQog"
    "IGNvbnN0IHMgPSBzdGF0ZS5zZXR0aW5ncyB8fCB7fTsKICBpZigoKHMuY29kZXJfcm9vdCl8fCIiKS50cmltKCkpewogICAgY29kZXJPbj10cnVlOyBh"
    "Z2VudE9uPWZhbHNlOyBsb29wT249ZmFsc2U7IGNvdW5jaWxPbj1mYWxzZTsgY29tcHV0ZXJPbj1mYWxzZTsKICAgIHN5bmNUb2dnbGVzKCk7IHJldHVy"
    "bjsKICB9CiAgY29kZXJTZXR1cE1vZGFsKCk7Cn07CmZ1bmN0aW9uIGNvZGVyU2V0dXBNb2RhbCgpewogIGNvbnN0IGJvZHkgPSBtb2RhbCh7dGl0bGU6"
    "IlNldCB1cCBDb2RlciBtb2RlIiwgYm9keUhUTUw6YAogICAgPHAgc3R5bGU9Im1hcmdpbi10b3A6MCI+Q29kZXIgaXMgYSBjb2RpbmcgYWdlbnQgdGhh"
    "dCB3b3JrcyBpbnNpZGUgPGI+b25lIHByb2plY3QKICAgIGZvbGRlcjwvYj4g4oCUIGl0IGNhbiBsaXN0LCByZWFkLCBzZWFyY2gkeyIifSBhbmQgKG9u"
    "Y2UgeW91IGFsbG93IGl0KSBlZGl0IHRoZSBmaWxlcwogICAgdGhlcmUsIGFuZCBydW4geW91ciB0ZXN0cy4gSXQgbmV2ZXIgdG91Y2hlcyBhbnl0aGlu"
    "ZyBvdXRzaWRlIHRoYXQgZm9sZGVyLjwvcD4KICAgIDxsYWJlbCBjbGFzcz0iZGltIiBzdHlsZT0iZm9udC1zaXplOjEycHgiPlByb2plY3QgZm9sZGVy"
    "IChhYnNvbHV0ZSBwYXRoKTwvbGFiZWw+CiAgICA8aW5wdXQgY2xhc3M9ImlucCBtb25vIiBpZD0iY29kZXJQYXRoSW5wIiBwbGFjZWhvbGRlcj0iL2hv"
    "bWUveW91L215cHJvamVjdCIKICAgICAgc3R5bGU9IndpZHRoOjEwMCU7bWFyZ2luOjZweCAwIDRweCI+CiAgICA8ZGl2IGlkPSJjb2RlclBhdGhIaW50"
    "IiBzdHlsZT0ibWluLWhlaWdodDoxOHB4Ij48L2Rpdj4KICAgIDxwIGNsYXNzPSJoaW50IiBzdHlsZT0ibWFyZ2luOjhweCAwIDAiPlN0YXJ0cyBpbiA8"
    "Yj5wbGFuIG1vZGU8L2I+IChyZWFkLW9ubHkpOiBpdCBwcm9wb3NlcwogICAgZGlmZnMgaW5zdGVhZCBvZiBlZGl0aW5nLiBUdXJuIG9uICJBbGxvdyBm"
    "aWxlIGVkaXRzIiBpbiBTZXR0aW5ncyB3aGVuIHlvdSdyZSByZWFkeQogICAgZm9yIGl0IHRvIGNoYW5nZSBmaWxlcyBkaXJlY3RseS48L3A+YCwKICAg"
    "IGFjdGlvbnM6WwogICAgICB7bGFiZWw6Ik9wZW4gZnVsbCBzZXR0aW5ncyIsIGNsczoiZ2hvc3QiLCBvbkNsaWNrOigpPT57IGNsb3NlTW9kYWwoKTsg"
    "cmV2ZWFsQ29kZXJTZXR0aW5ncygpOyB9fSwKICAgICAge2xhYmVsOiJTYXZlICYgdHVybiBvbiIsIGNsczoicHJpbWFyeSIsIG9uQ2xpY2s6YXN5bmMo"
    "Yik9PnsKICAgICAgICBjb25zdCB2ID0gYi5xdWVyeVNlbGVjdG9yKCIjY29kZXJQYXRoSW5wIikudmFsdWUudHJpbSgpOwogICAgICAgIGlmKCF2KXsg"
    "dG9hc3QoIlR5cGUgdGhlIGZvbGRlciBwYXRoIGZpcnN0IiwiZXJyIik7IHJldHVybjsgfQogICAgICAgIHRyeXsKICAgICAgICAgIGNvbnN0IGNoayA9"
    "IGF3YWl0IGFwaSgiL2FwaS9jb2Rlci9jaGVjaz9wYXRoPSIrZW5jb2RlVVJJQ29tcG9uZW50KHYpKTsKICAgICAgICAgIGlmKCFjaGsuaXNfZGlyKXsg"
    "dG9hc3QoIlRoYXQgZm9sZGVyIGRvZXNuJ3QgZXhpc3Qgb24gdGhpcyBtYWNoaW5lIiwiZXJyIik7IHJldHVybjsgfQogICAgICAgICAgYXdhaXQgcG9z"
    "dCgiL2FwaS9zZXR0aW5ncyIsIHtjb2Rlcl9yb290OiBjaGsucmVzb2x2ZWR9KTsKICAgICAgICAgIGNvbnN0IHIyID0gYXdhaXQgYXBpKCIvYXBpL3Nl"
    "dHRpbmdzIik7IHN0YXRlLnNldHRpbmdzID0gcjIuc2V0dGluZ3M7CiAgICAgICAgICBjb2Rlck9uPXRydWU7IGFnZW50T249ZmFsc2U7IGxvb3BPbj1m"
    "YWxzZTsgY291bmNpbE9uPWZhbHNlOyBjb21wdXRlck9uPWZhbHNlOwogICAgICAgICAgc3luY1RvZ2dsZXMoKTsgY2xvc2VNb2RhbCgpOwogICAgICAg"
    "ICAgdG9hc3QoIkNvZGVyIG1vZGUgaXMgb24g4oCUIHdvcmtpbmcgaW4gIitjaGsucmVzb2x2ZWQsIm9rIik7CiAgICAgICAgfWNhdGNoKGUpeyB0b2Fz"
    "dChlLm1lc3NhZ2UsImVyciIpOyB9CiAgICAgIH19CiAgICBdfSk7CiAgY29uc3QgaW5wID0gYm9keS5xdWVyeVNlbGVjdG9yKCIjY29kZXJQYXRoSW5w"
    "Iik7CiAgY29uc3QgaGludCA9IGJvZHkucXVlcnlTZWxlY3RvcigiI2NvZGVyUGF0aEhpbnQiKTsKICBsZXQgdD1udWxsOwogIGlucC5vbmlucHV0ID0g"
    "KCk9PnsgY2xlYXJUaW1lb3V0KHQpOyB0PXNldFRpbWVvdXQoYXN5bmMoKT0+ewogICAgY29uc3QgdiA9IGlucC52YWx1ZS50cmltKCk7CiAgICBpZigh"
    "dil7IGhpbnQuaW5uZXJIVE1MPSIiOyByZXR1cm47IH0KICAgIHRyeXsKICAgICAgY29uc3QgciA9IGF3YWl0IGFwaSgiL2FwaS9jb2Rlci9jaGVjaz9w"
    "YXRoPSIrZW5jb2RlVVJJQ29tcG9uZW50KHYpKTsKICAgICAgaGludC5pbm5lckhUTUwgPSByLmlzX2RpcgogICAgICAgID8gYDxzcGFuIGNsYXNzPSJw"
    "YXRob2siPlx1MjcxMyBGb2xkZXIgZm91bmQgXHUyMDE0ICR7ci5lbnRyaWVzfSBlbnRyJHtyLmVudHJpZXM9PT0xPyJ5IjoiaWVzIn08L3NwYW4+YAog"
    "ICAgICAgIDogYDxzcGFuIGNsYXNzPSJwYXRoYmFkIj5cdTI3MTcgTm90IGEgZm9sZGVyIG9uIHRoaXMgbWFjaGluZTwvc3Bhbj5gOwogICAgfWNhdGNo"
    "KGUpeyBoaW50LmlubmVySFRNTD0iIjsgfQogIH0sIDM1MCk7IH07CiAgc2V0VGltZW91dCgoKT0+aW5wLmZvY3VzKCksIDYwKTsKfQpmdW5jdGlvbiBy"
    "ZXZlYWxDb2RlclNldHRpbmdzKCl7CiAgc2hvdygic2V0dGluZ3MiKTsKICBsZXQgdHJpZXMgPSAwOwogIGNvbnN0IGZpbmQgPSAoKT0+ewogICAgY29u"
    "c3QgYyA9ICQoImNvZGVyQ2FyZCIpOwogICAgaWYoYyl7IGMuc2Nyb2xsSW50b1ZpZXcoe2JlaGF2aW9yOiJzbW9vdGgiLCBibG9jazoiY2VudGVyIn0p"
    "OwogICAgICBjLmNsYXNzTGlzdC5hZGQoImZsYXNoIik7IHNldFRpbWVvdXQoKCk9PmMuY2xhc3NMaXN0LnJlbW92ZSgiZmxhc2giKSwgMjMwMCk7IH0K"
    "ICAgIGVsc2UgaWYoKyt0cmllcyA8IDEyKSBzZXRUaW1lb3V0KGZpbmQsIDE1MCk7CiAgfTsKICBzZXRUaW1lb3V0KGZpbmQsIDIwMCk7Cn0KCi8qIC0t"
    "LS0tLS0tLS0gaW1hZ2UgYXR0YWNobWVudHMgKHZpc2lvbiBtb2RlbHMpIC0tLS0tLS0tLS0gKi8KbGV0IHBlbmRpbmdJbWFnZXMgPSBbXTsKZnVuY3Rp"
    "b24gcmVuZGVyQXR0YWNoKCl7CiAgY29uc3Qgcm93ID0gJCgiYXR0YWNoUm93Iik7IGlmKCFyb3cpIHJldHVybjsKICByb3cuaGlkZGVuID0gIXBlbmRp"
    "bmdJbWFnZXMubGVuZ3RoOwogIHJvdy5pbm5lckhUTUwgPSBwZW5kaW5nSW1hZ2VzLm1hcCgoZCxpKT0+CiAgICBgPHNwYW4gY2xhc3M9ImF0dGFjaGNo"
    "aXAiPjxpbWcgc3JjPSIke2R9Ij48YiBkYXRhLWk9IiR7aX0iIHRpdGxlPSJSZW1vdmUiPlx1MDBkNzwvYj48L3NwYW4+YCkuam9pbigiIik7CiAgcm93"
    "LnF1ZXJ5U2VsZWN0b3JBbGwoImIiKS5mb3JFYWNoKGI9PiBiLm9uY2xpY2sgPSAoKT0+ewogICAgcGVuZGluZ0ltYWdlcy5zcGxpY2UoK2IuZGF0YXNl"
    "dC5pLCAxKTsgcmVuZGVyQXR0YWNoKCk7IH0pOwp9CiQoImF0dGFjaEJ0biIpLm9uY2xpY2sgPSAoKT0+ICQoImF0dGFjaElucHV0IikuY2xpY2soKTsK"
    "JCgiYXR0YWNoSW5wdXQiKS5vbmNoYW5nZSA9IGFzeW5jIChlKT0+ewogIGZvcihjb25zdCBmIG9mIEFycmF5LmZyb20oZS50YXJnZXQuZmlsZXN8fFtd"
    "KSl7CiAgICBpZihwZW5kaW5nSW1hZ2VzLmxlbmd0aCA+PSA0KXsgdG9hc3QoIlVwIHRvIDQgaW1hZ2VzIHBlciBtZXNzYWdlIik7IGJyZWFrOyB9CiAg"
    "ICBpZihmLnNpemUgPiA4KjEwMjQqMTAyNCl7IHRvYXN0KGYubmFtZSsiIGlzIG92ZXIgOCBNQiIsImVyciIpOyBjb250aW51ZTsgfQogICAgY29uc3Qg"
    "ZCA9IGF3YWl0IG5ldyBQcm9taXNlKHJlcz0+eyBjb25zdCByID0gbmV3IEZpbGVSZWFkZXIoKTsKICAgICAgci5vbmxvYWQgPSAoKT0+cmVzKHIucmVz"
    "dWx0KTsgci5yZWFkQXNEYXRhVVJMKGYpOyB9KTsKICAgIHBlbmRpbmdJbWFnZXMucHVzaChkKTsKICB9CiAgZS50YXJnZXQudmFsdWUgPSAiIjsgcmVu"
    "ZGVyQXR0YWNoKCk7Cn07CiQoImV4cG9ydEJ0biIpLm9uY2xpY2sgPSAoKT0+ewogIGlmKCFzdGF0ZS5jdXJyZW50Q29udil7IHRvYXN0KCJOb3RoaW5n"
    "IHRvIGV4cG9ydCB5ZXQiKTsgcmV0dXJuOyB9CiAgY29uc3QgYSA9IGRvY3VtZW50LmNyZWF0ZUVsZW1lbnQoImEiKTsKICBhLmhyZWYgPSAiL2FwaS9j"
    "b252ZXJzYXRpb25zLyIrc3RhdGUuY3VycmVudENvbnYrIi9leHBvcnQiOwogIGEuZG93bmxvYWQgPSAiIjsgZG9jdW1lbnQuYm9keS5hcHBlbmRDaGls"
    "ZChhKTsgYS5jbGljaygpOyBhLnJlbW92ZSgpOwp9OwoKYXN5bmMgZnVuY3Rpb24gcmVmcmVzaENoYXRNb2RlbHMoKXsKICB0cnl7CiAgICBjb25zdCBp"
    "bnN0ID0gYXdhaXQgYXBpKCIvYXBpL21vZGVscy9pbnN0YWxsZWQiKTsKICAgIHN0YXRlLmluc3RhbGxlZCA9IGluc3QubW9kZWxzIHx8IFtdOwogICAg"
    "Y29uc3Qgc2VsID0gJCgiY2hhdE1vZGVsIik7IGNvbnN0IHByZXYgPSBzZWwudmFsdWU7CiAgICBjb25zdCBjaGF0ID0gc3RhdGUuaW5zdGFsbGVkLmZp"
    "bHRlcihtPT4hL2VtYmVkL2kudGVzdChtLm5hbWUpKTsKICAgIGlmKCFjaGF0Lmxlbmd0aCl7IHNlbC5pbm5lckhUTUwgPSAnPG9wdGlvbiB2YWx1ZT0i"
    "Ij5ObyBtb2RlbHMg4oCUIGRvd25sb2FkIG9uZTwvb3B0aW9uPic7IHJldHVybjsgfQogICAgc2VsLmlubmVySFRNTCA9IGNoYXQubWFwKG09PmA8b3B0"
    "aW9uIHZhbHVlPSIke2VzYyhtLm5hbWUpfSI+JHtlc2MobS5uYW1lKX08L29wdGlvbj5gKS5qb2luKCIiKTsKICAgIGlmKHByZXYgJiYgY2hhdC5zb21l"
    "KG09Pm0ubmFtZT09PXByZXYpKSBzZWwudmFsdWUgPSBwcmV2OwogIH1jYXRjaChlKXt9Cn0KYXN5bmMgZnVuY3Rpb24gbG9hZENvbnZzKCl7CiAgdHJ5"
    "ewogICAgY29uc3QgciA9IGF3YWl0IGFwaSgiL2FwaS9jb252ZXJzYXRpb25zIik7CiAgICBjb25zdCBsaXN0ID0gJCgiY29udkxpc3QiKTsgbGlzdC5p"
    "bm5lckhUTUwgPSAiIjsKICAgIGlmKCFyLmNvbnZlcnNhdGlvbnMubGVuZ3RoKXsgbGlzdC5pbm5lckhUTUwgPQogICAgICAnPGRpdiBjbGFzcz0iZGlt"
    "IiBzdHlsZT0icGFkZGluZzoxNHB4O2ZvbnQtc2l6ZToxMnB4O3RleHQtYWxpZ246Y2VudGVyIj5ObyBjb252ZXJzYXRpb25zIHlldDwvZGl2Pic7IHJl"
    "dHVybjsgfQogICAgci5jb252ZXJzYXRpb25zLmZvckVhY2goYz0+ewogICAgICBjb25zdCBpdCA9IGVsKCJkaXYiLCJjb252aXRlbSIrKGMuaWQ9PT1z"
    "dGF0ZS5jdXJyZW50Q29udj8iIGFjdGl2ZSI6IiIpKTsKICAgICAgaXQuaW5uZXJIVE1MID0gYDxzcGFuIGNsYXNzPSJ0dCI+JHtlc2MoYy50aXRsZXx8"
    "IlVudGl0bGVkIil9PC9zcGFuPgogICAgICAgIDxzcGFuIGNsYXNzPSJkZWwiIHRpdGxlPSJEZWxldGUiPgogICAgICAgIDxzdmcgd2lkdGg9IjE0IiBo"
    "ZWlnaHQ9IjE0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIGZpbGw9Im5vbmUiIHN0cm9rZT0iY3VycmVudENvbG9yIiBzdHJva2Utd2lkdGg9IjIiPjxwYXRo"
    "IGQ9Ik0zIDZoMThNOCA2VjRoOHYyTTYgNmwxIDE0aDEwbDEtMTQiLz48L3N2Zz48L3NwYW4+YDsKICAgICAgaXQucXVlcnlTZWxlY3RvcigiLnR0Iiku"
    "b25jbGljayA9ICgpPT5vcGVuQ29udihjLmlkKTsKICAgICAgaXQucXVlcnlTZWxlY3RvcigiLmRlbCIpLm9uY2xpY2sgPSBhc3luYyhlKT0+eyBlLnN0"
    "b3BQcm9wYWdhdGlvbigpOwogICAgICAgIGF3YWl0IGRlbCgiL2FwaS9jb252ZXJzYXRpb25zLyIrYy5pZCk7CiAgICAgICAgaWYoc3RhdGUuY3VycmVu"
    "dENvbnY9PT1jLmlkKXsgc3RhdGUuY3VycmVudENvbnY9bnVsbDsgc2hvd0NoYXRFbXB0eSgpOyB9CiAgICAgICAgbG9hZENvbnZzKCk7IH07CiAgICAg"
    "IGxpc3QuYXBwZW5kQ2hpbGQoaXQpOwogICAgfSk7CiAgfWNhdGNoKGUpe30KfQphc3luYyBmdW5jdGlvbiBvcGVuQ29udihpZCl7CiAgaWYoc3RhdGUu"
    "c2VuZGluZyAmJiBzdGF0ZS5hYm9ydCl7IHRyeXsgc3RhdGUuYWJvcnQuYWJvcnQoKTsgfWNhdGNoKGUpe30gfQogIHN0YXRlLmN1cnJlbnRDb252ID0g"
    "aWQ7IGxvYWRDb252cygpOwogIGNvbnN0IHdyYXAgPSAkKCJjaGF0V3JhcCIpOyB3cmFwLmlubmVySFRNTCA9ICIiOwogIHRyeXsKICAgIGNvbnN0IHIg"
    "PSBhd2FpdCBhcGkoIi9hcGkvY29udmVyc2F0aW9ucy8iK2lkKyIvbWVzc2FnZXMiKTsKICAgIHIubWVzc2FnZXMuZm9yRWFjaChtPT57CiAgICAgIGlm"
    "KG0ucm9sZT09PSJ1c2VyIikgYWRkVXNlck1zZyhtLmNvbnRlbnQsIChtLm1ldGEgJiYgbS5tZXRhLmltYWdlcykgfHwgW10pOwogICAgICBlbHNlIHsg"
    "Y29uc3Qge2J1YmJsZSwgc291cmNlcywgYWdlbnRsb2d9ID0gYWRkQUlNc2coKTsKICAgICAgICBjb25zdCBldiA9IChtLm1ldGEgJiYgbS5tZXRhLmV2"
    "ZW50cykgfHwgW107CiAgICAgICAgY29uc3Qgc3JjcyA9IGV2LmZpbHRlcihlPT5lLnR5cGU9PT0ic291cmNlcyIpLmZsYXRNYXAoZT0+ZS5pdGVtc3x8"
    "W10pOwogICAgICAgIGlmKHNyY3MubGVuZ3RoKSByZW5kZXJTb3VyY2VzKHNvdXJjZXMsIHNyY3MpOwogICAgICAgIHJlbmRlckFnZW50RXZlbnRzKGFn"
    "ZW50bG9nLCBldik7CiAgICAgICAgYnViYmxlLmlubmVySFRNTCA9IG1kKG0uY29udGVudCk7CiAgICAgICAgY29uc3Qgc3QgPSBldi5maWx0ZXIoZT0+"
    "ZS50eXBlPT09InN0YXRzIikucG9wKCk7CiAgICAgICAgaWYoc3QpIGFwcGVuZFN0YXQoYnViYmxlLCBzdCk7CiAgICAgIH0KICAgIH0pOwogICAgc2Ny"
    "b2xsQ2hhdCgpOwogIH1jYXRjaChlKXsgdG9hc3QoZS5tZXNzYWdlLCJlcnIiKTsgfQp9CmZ1bmN0aW9uIHNob3dDaGF0RW1wdHkoKXsKICBjb25zdCB3"
    "cmFwID0gJCgiY2hhdFdyYXAiKTsKICB3cmFwLmlubmVySFRNTCA9IGA8ZGl2IGNsYXNzPSJjaGF0LWVtcHR5Ij4KICAgIDxkaXYgY2xhc3M9Im9yYiI+"
    "PC9kaXY+CiAgICA8aDM+VGFsayB0byB5b3VyIG1hY2hpbmU8L2gzPgogICAgPHAgY2xhc3M9Im11dGVkIj5FdmVyeXRoaW5nIHN0YXlzIG9uIHRoaXMg"
    "Y29tcHV0ZXIuIFBpY2sgYSBtb2RlbCBhYm92ZSBhbmQgYXNrCiAgICAgIGFueXRoaW5nIOKAlCBvciBzdGFydCB3aXRoIG9uZSBvZiB0aGVzZTo8L3A+"
    "CiAgICA8ZGl2IGNsYXNzPSJleHJvdyI+CiAgICAgIDxidXR0b24gY2xhc3M9ImJ0biBzbSBnaG9zdCBleCI+RXhwbGFpbiBob3cgYSB2ZWN0b3IgZGF0"
    "YWJhc2Ugd29ya3M8L2J1dHRvbj4KICAgICAgPGJ1dHRvbiBjbGFzcz0iYnRuIHNtIGdob3N0IGV4Ij5TdW1tYXJpemUgdGhlIHByb3MgYW5kIGNvbnMg"
    "b2YgcnVubmluZyBBSSBsb2NhbGx5PC9idXR0b24+CiAgICAgIDxidXR0b24gY2xhc3M9ImJ0biBzbSBnaG9zdCBleCI+RHJhZnQgYSBwb2xpdGUgZW1h"
    "aWwgZGVjbGluaW5nIGEgbWVldGluZzwvYnV0dG9uPgogICAgPC9kaXY+CiAgICA8cCBjbGFzcz0iaGludCI+VGlwOiBIZW9ydGggYXV0by1zZWFyY2hl"
    "cyB0aGUgd2ViIHdoZW4gYSBxdWVzdGlvbiBuZWVkcyBjdXJyZW50IGluZm8uCiAgICAgIDxiPktub3dsZWRnZTwvYj4gdXNlcyB5b3VyIGRvY3VtZW50"
    "cywgPGI+QWdlbnQ8L2I+IGxldHMgdGhlIG1vZGVsIGNhbGwgdG9vbHMsIDxiPkxvb3A8L2I+CiAgICAgIHdvcmtzIGEgdGFzayBhdXRvbm9tb3VzbHks"
    "IDxiPkNvdW5jaWw8L2I+IGNvbnZlbmVzIGEgZGViYXRpbmcgcGFuZWwsIGFuZCA8Yj5Db21wdXRlcjwvYj4KICAgICAgbGV0cyBpdCBjb250cm9sIHlv"
    "dXIgbW91c2UgJmFtcDsga2V5Ym9hcmQuPC9wPjwvZGl2PmA7CiAgd3JhcC5xdWVyeVNlbGVjdG9yQWxsKCIuZXgiKS5mb3JFYWNoKGI9PmIub25jbGlj"
    "az0oKT0+ewogICAgJCgiY2hhdElucHV0IikudmFsdWUgPSBiLnRleHRDb250ZW50LnRyaW0oKTsgYXV0b0dyb3coKTsgJCgiY2hhdElucHV0IikuZm9j"
    "dXMoKTsgfSk7Cn0KZnVuY3Rpb24gYXBwZW5kU3RhdChidWJibGUsIHMpewogIGNvbnN0IG9sZCA9IGJ1YmJsZS5wYXJlbnROb2RlLnF1ZXJ5U2VsZWN0"
    "b3IoIi5zdGF0bGluZSIpOwogIGlmKG9sZCkgb2xkLnJlbW92ZSgpOwogIGNvbnN0IGQgPSBlbCgiZGl2Iiwic3RhdGxpbmUiKTsKICBpZihzLnByb21w"
    "dF90b2tlbnMpIGQudGl0bGUgPSAicHJvbXB0OiAiICsgcy5wcm9tcHRfdG9rZW5zICsgIiB0b2tlbnMiOwogIGQudGV4dENvbnRlbnQgPSAiXHUyNmEx"
    "ICIgKyBzLnRwcyArICIgdG9rL3MgXHUwMGI3ICIgKyBzLnRva2VucwogICAgKyAiIHRva2VucyBcdTAwYjcgIiArIHMuc2Vjb25kcyArICJzIjsKICBi"
    "dWJibGUucGFyZW50Tm9kZS5pbnNlcnRCZWZvcmUoZCwgYnViYmxlLm5leHRTaWJsaW5nKTsKfQpmdW5jdGlvbiBhZGRVc2VyTXNnKHRleHQsIGltZ3Mp"
    "ewogIGNvbnN0IHdyYXAgPSAkKCJjaGF0V3JhcCIpOwogIGNvbnN0IGVtcCA9IHdyYXAucXVlcnlTZWxlY3RvcigiLmNoYXQtZW1wdHkiKTsgaWYoZW1w"
    "KSBlbXAucmVtb3ZlKCk7CiAgY29uc3QgbSA9IGVsKCJkaXYiLCJtc2cgdXNlciIpOwogIGNvbnN0IHBpY3MgPSAoaW1ncyAmJiBpbWdzLmxlbmd0aCkK"
    "ICAgID8gYDxkaXYgY2xhc3M9Im1zZ2ltZ3MiPiR7aW1ncy5tYXAoZD0+YDxpbWcgc3JjPSIke2R9Ij5gKS5qb2luKCIiKX08L2Rpdj5gIDogIiI7CiAg"
    "bS5pbm5lckhUTUwgPSBgPGRpdiBjbGFzcz0iYXYiPllPVTwvZGl2PjxkaXYgY2xhc3M9ImJvZHkiPgogICAgPGRpdiBjbGFzcz0id2hvIj5Zb3U8L2Rp"
    "dj4ke3BpY3N9PGRpdiBjbGFzcz0iYnViYmxlIj4ke21kKHRleHQpfTwvZGl2PjwvZGl2PmA7CiAgd3JhcC5hcHBlbmRDaGlsZChtKTsgc2Nyb2xsQ2hh"
    "dCgpOyByZXR1cm4gbTsKfQpmdW5jdGlvbiBhZGRBSU1zZygpewogIGNvbnN0IHdyYXAgPSAkKCJjaGF0V3JhcCIpOwogIGNvbnN0IG0gPSBlbCgiZGl2"
    "IiwibXNnIGFpIik7CiAgbS5pbm5lckhUTUwgPSBgPGRpdiBjbGFzcz0iYXYiPkFJPC9kaXY+PGRpdiBjbGFzcz0iYm9keSI+CiAgICA8ZGl2IGNsYXNz"
    "PSJ3aG8iPkhlb3J0aDwvZGl2PgogICAgPGRpdiBjbGFzcz0ic3JjaG9zdCI+PC9kaXY+PGRpdiBjbGFzcz0ic3RhdHVzaG9zdCI+PC9kaXY+CiAgICA8"
    "ZGl2IGNsYXNzPSJhZ2VudGxvZyI+PC9kaXY+PGRpdiBjbGFzcz0iYnViYmxlIj48L2Rpdj48L2Rpdj5gOwogIHdyYXAuYXBwZW5kQ2hpbGQobSk7CiAg"
    "cmV0dXJuIHsgcm9vdDptLCBidWJibGU6bS5xdWVyeVNlbGVjdG9yKCIuYnViYmxlIiksCiAgICBzb3VyY2VzOm0ucXVlcnlTZWxlY3RvcigiLnNyY2hv"
    "c3QiKSwgc3RhdHVzOm0ucXVlcnlTZWxlY3RvcigiLnN0YXR1c2hvc3QiKSwKICAgIGFnZW50bG9nOm0ucXVlcnlTZWxlY3RvcigiLmFnZW50bG9nIikg"
    "fTsKfQpmdW5jdGlvbiByZW5kZXJTb3VyY2VzKGhvc3QsIGl0ZW1zKXsKICBpZighaXRlbXMubGVuZ3RoKSByZXR1cm47CiAgY29uc3QgYm94ID0gZWwo"
    "ImRpdiIsInNyY2JveCIpOwogIGJveC5pbm5lckhUTUwgPSAnPGRpdiBjbGFzcz0ic3QiPuKXhiBTb3VyY2VzIGZyb20geW91ciBrbm93bGVkZ2UgYmFz"
    "ZTwvZGl2PicrCiAgICBpdGVtcy5tYXAocz0+YDxkaXYgY2xhc3M9InNyY2l0ZW0iPjxiPiR7ZXNjKHMuZG9jKX08L2I+CiAgICAgIDxzcGFuIGNsYXNz"
    "PSJkaW0iPsK3ICR7KHMuc2NvcmUqMTAwfHwwKS50b0ZpeGVkKDApfSUgbWF0Y2g8L3NwYW4+PGJyPgogICAgICA8c3BhbiBjbGFzcz0ibXV0ZWQiPiR7"
    "ZXNjKChzLnNuaXBwZXR8fHMudGV4dHx8IiIpLnNsaWNlKDAsMTUwKSl94oCmPC9zcGFuPjwvZGl2PmApLmpvaW4oIiIpOwogIGhvc3QuYXBwZW5kQ2hp"
    "bGQoYm94KTsKfQpmdW5jdGlvbiBzdGVwRWwobmFtZSwgYXJncyl7CiAgY29uc3QgaXNNY3AgPSBuYW1lLnN0YXJ0c1dpdGgoIm1jcF9fIik7CiAgY29u"
    "c3QgZGlzcCA9IGlzTWNwID8gbmFtZS5zcGxpdCgiX18iKS5zbGljZSgyKS5qb2luKCJfXyIpKyIgKE1DUCkiIDogbmFtZTsKICBjb25zdCBzID0gZWwo"
    "ImRpdiIsInN0ZXAiKTsKICBjb25zdCBhcmdTdHIgPSB0eXBlb2YgYXJncz09PSJvYmplY3QiID8gSlNPTi5zdHJpbmdpZnkoYXJncykgOiBTdHJpbmco"
    "YXJnc3x8IiIpOwogIHMuaW5uZXJIVE1MID0gYDxkaXYgY2xhc3M9InNoIj48ZGl2IGNsYXNzPSJzcGluIiBzdHlsZT0id2lkdGg6MTJweDtoZWlnaHQ6"
    "MTJweCI+PC9kaXY+CiAgICA8c3BhbiBjbGFzcz0idG4iPiR7ZXNjKGRpc3ApfTwvc3Bhbj48c3BhbiBjbGFzcz0iYXJnIj4ke2VzYyhhcmdTdHIpfTwv"
    "c3Bhbj48L2Rpdj4KICAgIDxkaXYgY2xhc3M9InNiIiBoaWRkZW4+PC9kaXY+YDsKICBjb25zdCBzaCA9IHMucXVlcnlTZWxlY3RvcigiLnNoIiksIHNi"
    "ID0gcy5xdWVyeVNlbGVjdG9yKCIuc2IiKTsKICBzaC5zdHlsZS5jdXJzb3I9InBvaW50ZXIiOwogIHNoLm9uY2xpY2sgPSAoKT0+eyBzYi5oaWRkZW4g"
    "PSAhc2IuaGlkZGVuOyB9OwogIHJldHVybiBzOwp9CmZ1bmN0aW9uIHJlbmRlckFnZW50RXZlbnRzKGhvc3QsIGV2ZW50cyl7CiAgbGV0IHBlbmRpbmcg"
    "PSBudWxsOwogIGxldCBjcD1udWxsLCBjb25zPVtdOyBjb25zdCBjZmFpbD1uZXcgU2V0KCk7CiAgZXZlbnRzLmZvckVhY2goZT0+ewogICAgaWYoZS50"
    "eXBlPT09InRvb2xfY2FsbCIpeyBjb25zdCBzID0gc3RlcEVsKGUubmFtZSwgZS5hcmdzKTsKICAgICAgaG9zdC5hcHBlbmRDaGlsZChzKTsgcGVuZGlu"
    "ZyA9IHM7IH0KICAgIGVsc2UgaWYoZS50eXBlPT09InRvb2xfcmVzdWx0IiAmJiBwZW5kaW5nKXsKICAgICAgY29uc3Qgc3AgPSBwZW5kaW5nLnF1ZXJ5"
    "U2VsZWN0b3IoIi5zcGluIik7IGlmKHNwKSBzcC5vdXRlckhUTUwgPQogICAgICAgICc8c3BhbiBzdHlsZT0iY29sb3I6dmFyKC0tbGl2ZSkiPuKckzwv"
    "c3Bhbj4nOwogICAgICBjb25zdCBzYiA9IHBlbmRpbmcucXVlcnlTZWxlY3RvcigiLnNiIik7IHNiLnRleHRDb250ZW50ID0gZS5yZXN1bHR8fCIiOwog"
    "ICAgICBwZW5kaW5nID0gbnVsbDsKICAgIH0KICAgIGVsc2UgaWYoZS50eXBlPT09InRob3VnaHQiKXsgaG9zdC5hcHBlbmRDaGlsZCh0aG91Z2h0RWwo"
    "ZS50ZXh0KSk7IH0KICAgIGVsc2UgaWYoZS50eXBlPT09ImF1dG9fc2VhcmNoIil7IGhvc3QuYXBwZW5kQ2hpbGQod2ViU2VhcmNoRWwoZS5xdWVyeSwg"
    "ZS5yZXN1bHQpKTsgfQogICAgZWxzZSBpZihlLnR5cGU9PT0iY29tcHV0ZXJfYWN0aW9uIil7CiAgICAgIGxldCBmZWVkPWhvc3QucXVlcnlTZWxlY3Rv"
    "cigiLmNjZmVlZCIpOwogICAgICBpZighZmVlZCl7IGZlZWQ9ZWwoImRpdiIsImNjZmVlZCIpOwogICAgICAgIGZlZWQuaW5uZXJIVE1MPSc8ZGl2IGNs"
    "YXNzPSJjZmgiPvCflqUgQ29tcHV0ZXIgY29udHJvbDwvZGl2PjxkaXYgY2xhc3M9ImNjYm9keSI+PC9kaXY+JzsKICAgICAgICBob3N0LmFwcGVuZENo"
    "aWxkKGZlZWQpOyB9CiAgICAgIGNvbnN0IHJvdz1lbCgiZGl2IiwiY2NhY3QiKTsKICAgICAgcm93LmlubmVySFRNTD1gPHNwYW4gY2xhc3M9ImFpYyI+"
    "JHtDQ19JQ09OU1tlLmtpbmRdfHwi4oCiIn08L3NwYW4+CiAgICAgICAgPHNwYW4+JHtlc2MoZS5kZXRhaWx8fGUua2luZCl9PC9zcGFuPgogICAgICAg"
    "ICR7ZS5zaG90P2A8aW1nIHNyYz0iL2FwaS9jb21wdXRlci9zaG90LyR7ZS5zaG90fSIgdGl0bGU9IldoYXQgdGhlIG1vZGVsIHNhdyI+YDoiIn1gOwog"
    "ICAgICBpZihlLnNob3QpIHJvdy5xdWVyeVNlbGVjdG9yKCJpbWciKS5vbmNsaWNrPSgpPT53aW5kb3cub3BlbigiL2FwaS9jb21wdXRlci9zaG90LyIr"
    "ZS5zaG90KTsKICAgICAgZmVlZC5xdWVyeVNlbGVjdG9yKCIuY2Nib2R5IikuYXBwZW5kQ2hpbGQocm93KTsKICAgIH0KICAgIGVsc2UgaWYoZS50eXBl"
    "PT09ImNvdW5jaWxfc3RhcnQiKXsgY29ucz1lLmNvbnN1bHRhbnRzfHxbXTsKICAgICAgY3A9Y291bmNpbFBhbmVsRWwoZSk7IGhvc3QuYXBwZW5kQ2hp"
    "bGQoY3ApOyB9CiAgICBlbHNlIGlmKGUudHlwZT09PSJjb3VuY2lsX2JyaWVmIiAmJiBjcCl7IGNwLmFwcGVuZENoaWxkKGNvdW5jaWxCcmllZkVsKGUu"
    "dGV4dCkpOyB9CiAgICBlbHNlIGlmKGUudHlwZT09PSJjb3VuY2lsX3JvdW5kIiAmJiBjcCl7CiAgICAgIGNvdW5jaWxSb3VuZEVsKGNwLCBlLnJvdW5k"
    "LCBlLmxhYmVsLCBjb25zLCBjZmFpbCwgZmFsc2UpOyB9CiAgICBlbHNlIGlmKGUudHlwZT09PSJjb3VuY2lsX3Rha2UiICYmIGNwKXsgY291bmNpbEZp"
    "bGwoY3AsIGUuaWQsIGUucm91bmQsIGUudGV4dCk7IH0KICAgIGVsc2UgaWYoZS50eXBlPT09ImNvbnN1bHRhbnRfc3RhdHVzIiAmJiBjcCAmJiBlLnN0"
    "YXRlPT09ImZhaWxlZCIpewogICAgICBjZmFpbC5hZGQoZS5pZCk7IGNvdW5jaWxGYWlsKGNwLCBlLmlkLCBlLnJvdW5kKTsgfQogIH0pOwp9CgpmdW5j"
    "dGlvbiBzY3JvbGxDaGF0KCl7IGNvbnN0IHMgPSAkKCJjaGF0U2Nyb2xsIik7IHMuc2Nyb2xsVG9wID0gcy5zY3JvbGxIZWlnaHQ7IH0KZnVuY3Rpb24g"
    "YXV0b0dyb3coKXsgY29uc3QgdCA9ICQoImNoYXRJbnB1dCIpOyB0LnN0eWxlLmhlaWdodD0iYXV0byI7CiAgdC5zdHlsZS5oZWlnaHQgPSBNYXRoLm1p"
    "bih0LnNjcm9sbEhlaWdodCwgMjAwKSsicHgiOyB9CmNvbnN0IFNFTkRfSUNPTiA9ICc8c3ZnIHdpZHRoPSIxOCIgaGVpZ2h0PSIxOCIgdmlld0JveD0i"
    "MCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9ImN1cnJlbnRDb2xvciIgc3Ryb2tlLXdpZHRoPSIyIj48cGF0aCBkPSJNNyAxMSAxMiA2bDUgNU0x"
    "MiA2djEzIi8+PC9zdmc+JzsKY29uc3QgU1RPUF9JQ09OID0gJzxzdmcgd2lkdGg9IjE2IiBoZWlnaHQ9IjE2IiB2aWV3Qm94PSIwIDAgMjQgMjQiIGZp"
    "bGw9ImN1cnJlbnRDb2xvciI+PHJlY3QgeD0iNiIgeT0iNiIgd2lkdGg9IjEyIiBoZWlnaHQ9IjEyIiByeD0iMiIvPjwvc3ZnPic7CmZ1bmN0aW9uIHNl"
    "dFNlbmRNb2RlKHNlbmRpbmcpewogIGNvbnN0IGIgPSAkKCJzZW5kQnRuIik7CiAgYi5jbGFzc0xpc3QudG9nZ2xlKCJzdG9wIiwgc2VuZGluZyk7CiAg"
    "Yi50aXRsZSA9IHNlbmRpbmcgPyAiU3RvcCBnZW5lcmF0aW5nIiA6ICJTZW5kIjsKICBiLmlubmVySFRNTCA9IHNlbmRpbmcgPyBTVE9QX0lDT04gOiBT"
    "RU5EX0lDT047Cn0KJCgiY2hhdElucHV0IikuYWRkRXZlbnRMaXN0ZW5lcigiaW5wdXQiLCBhdXRvR3Jvdyk7CiQoImNoYXRJbnB1dCIpLmFkZEV2ZW50"
    "TGlzdGVuZXIoImtleWRvd24iLCBlPT57CiAgaWYoZS5rZXk9PT0iRW50ZXIiICYmICFlLnNoaWZ0S2V5KXsgZS5wcmV2ZW50RGVmYXVsdCgpOwogICAg"
    "aWYoIXN0YXRlLnNlbmRpbmcpIHNlbmRNZXNzYWdlKCk7IH0gfSk7CiQoInNlbmRCdG4iKS5vbmNsaWNrID0gKCk9PnsKICBpZihzdGF0ZS5zZW5kaW5n"
    "KXsgaWYoc3RhdGUuYWJvcnQpIHN0YXRlLmFib3J0LmFib3J0KCk7IHJldHVybjsgfQogIHNlbmRNZXNzYWdlKCk7Cn07CiQoImNsZWFyQ2hhdEJ0biIp"
    "Lm9uY2xpY2sgPSAkKCJuZXdDb252QnRuIikub25jbGljayA9ICgpPT57CiAgc3RhdGUuY3VycmVudENvbnY9bnVsbDsgc2hvd0NoYXRFbXB0eSgpOyBs"
    "b2FkQ29udnMoKTsKICAkKCJjaGF0SW5wdXQiKS5mb2N1cygpOyB9OwoKZnVuY3Rpb24gdGhvdWdodEVsKHRleHQpewogIGNvbnN0IGQgPSBlbCgiZGl2"
    "IiwidGhvdWdodCIpOwogIGQuaW5uZXJIVE1MID0gJzxzcGFuIGNsYXNzPSJ0bCI+4pyzIHRob3VnaHQ8L3NwYW4+JyArIGVzYyh0ZXh0fHwiIik7CiAg"
    "cmV0dXJuIGQ7Cn0KZnVuY3Rpb24gd2ViU2VhcmNoRWwocXVlcnksIHJlc3VsdCl7CiAgY29uc3QgYm94ID0gZWwoImRpdiIsIndlYnNlYXJjaC1ib3gg"
    "Y29sbGFwc2VkIik7CiAgYm94LmlubmVySFRNTCA9IGA8ZGl2IGNsYXNzPSJ3aCI+PHN2ZyB3aWR0aD0iMTMiIGhlaWdodD0iMTMiIHZpZXdCb3g9IjAg"
    "MCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJjdXJyZW50Q29sb3IiIHN0cm9rZS13aWR0aD0iMiI+PGNpcmNsZSBjeD0iMTEiIGN5PSIxMSIgcj0i"
    "NyIvPjxwYXRoIGQ9Im0yMSAyMS00LjMtNC4zIi8+PC9zdmc+CiAgICA8c3BhbiBjbGFzcz0id3EiPldlYiBzZWFyY2g6ICR7ZXNjKHF1ZXJ5fHwiIil9"
    "PC9zcGFuPjxzcGFuIGNsYXNzPSJjdiI+4pa8PC9zcGFuPjwvZGl2PgogICAgPGRpdiBjbGFzcz0id2IiPjwvZGl2PmA7CiAgYm94LnF1ZXJ5U2VsZWN0"
    "b3IoIi53YiIpLnRleHRDb250ZW50ID0gcmVzdWx0IHx8ICIiOwogIGJveC5xdWVyeVNlbGVjdG9yKCIud2giKS5vbmNsaWNrID0gKCk9PmJveC5jbGFz"
    "c0xpc3QudG9nZ2xlKCJjb2xsYXBzZWQiKTsKICByZXR1cm4gYm94Owp9Ci8qIC0tLS0gY291bmNpbCByZW5kZXJpbmcgKHNoYXJlZCBieSBsaXZlIHN0"
    "cmVhbSBhbmQgaGlzdG9yeSByZXBsYXkpIC0tLS0gKi8KY29uc3QgQ0NfQ09MT1JTID0gWyIjZjVhNjIzIiwiIzYwYTVmYSIsIiNhNzhiZmEiLCIjNGFk"
    "ZTgwIiwiI2Y4NzE3MSIsCiAgICAgICAgICAgICAgICAgICAiI2Y0NzJiNiIsIiNmYWNjMTUiLCIjMmRkNGJmIiwiI2ZiOTIzYyIsIiM5NGEzYjgiXTsK"
    "Y29uc3QgQ0MgPSAoaSk9PkNDX0NPTE9SU1tpICUgQ0NfQ09MT1JTLmxlbmd0aF07CmZ1bmN0aW9uIGNvdW5jaWxQYW5lbEVsKG8pewogIGNvbnN0IHAg"
    "PSBlbCgiZGl2IiwiY291bmNpbCIpOwogIGNvbnN0IHIgPSBvLnJvdW5kcyA/IGAgwrcgJHtvLnJvdW5kc30gY29uc3VsdGF0aW9uIHJvdW5kJHtvLnJv"
    "dW5kcz4xPyJzIjoiIn1gIDogIiI7CiAgcC5pbm5lckhUTUwgPSBgPGRpdiBjbGFzcz0iY2hlYWQiPuKsoiBDb3VuY2lsIG9mICR7by5zaXplfSR7cn08"
    "L2Rpdj5gOwogIHJldHVybiBwOwp9CmZ1bmN0aW9uIGNvdW5jaWxCcmllZkVsKHRleHQpewogIGNvbnN0IGIgPSBlbCgiZGl2Iiwic3JjYm94Iik7IGIu"
    "c3R5bGUubWFyZ2luVG9wID0gIjEwcHgiOwogIGIuaW5uZXJIVE1MID0gJzxkaXYgY2xhc3M9InN0Ij7il4YgU2hhcmVkIHJlc2VhcmNoIGJyaWVmPC9k"
    "aXY+JyArCiAgICAnPGRpdiBjbGFzcz0ibW9ubyIgc3R5bGU9ImZvbnQtc2l6ZToxMXB4O3doaXRlLXNwYWNlOnByZS13cmFwO21heC1oZWlnaHQ6MTUw"
    "cHg7b3ZlcmZsb3c6YXV0bztjb2xvcjp2YXIoLS1tdXRlZCkiPjwvZGl2Pic7CiAgYi5sYXN0RWxlbWVudENoaWxkLnRleHRDb250ZW50ID0gdGV4dCB8"
    "fCAiIjsKICByZXR1cm4gYjsKfQpmdW5jdGlvbiBjb3VuY2lsUm91bmRFbChwYW5lbCwgcm91bmQsIGxhYmVsLCBjb25zdWx0YW50cywgZmFpbGVkLCBs"
    "aXZlKXsKICBjb25zdCBzZWMgPSBlbCgiZGl2Iik7CiAgc2VjLmlubmVySFRNTCA9IGA8ZGl2IGNsYXNzPSJjcm91bmQiPiR7ZXNjKGxhYmVsfHwoIlJv"
    "dW5kICIrcm91bmQpKX08L2Rpdj5gOwogIGNvbnN0IGcgPSBlbCgiZGl2IiwiY2dyaWQiKTsKICAoY29uc3VsdGFudHN8fFtdKS5mb3JFYWNoKGM9PnsK"
    "ICAgIGlmKGZhaWxlZC5oYXMoYy5pZCkpIHJldHVybjsKICAgIGNvbnN0IGNhcmQgPSBlbCgiZGl2IiwiY2NhcmQiKTsKICAgIGNhcmQuZGF0YXNldC5j"
    "aWQgPSBjLmlkOyBjYXJkLmRhdGFzZXQucm91bmQgPSByb3VuZDsKICAgIGNhcmQuaW5uZXJIVE1MID0gYDxkaXYgY2xhc3M9ImNyb2xlIj48c3BhbiBj"
    "bGFzcz0iY2RvdCIgc3R5bGU9ImJhY2tncm91bmQ6JHtDQyhjLmlkKX0iPjwvc3Bhbj4KICAgICAgPHNwYW4+JHtlc2MoYy5yb2xlKX08L3NwYW4+PHNw"
    "YW4gY2xhc3M9ImNzdCI+JHtsaXZlPyc8c3BhbiBjbGFzcz0ic3BpbiIgc3R5bGU9IndpZHRoOjExcHg7aGVpZ2h0OjExcHgiPjwvc3Bhbj4nOicnfTwv"
    "c3Bhbj48L2Rpdj4KICAgICAgJHtyb3VuZD09PTA/YDxkaXYgY2xhc3M9ImNmb2N1cyI+JHtlc2MoYy5mb2N1c3x8IiIpfTwvZGl2PmA6IiJ9CiAgICAg"
    "IDxkaXYgY2xhc3M9ImN0YWtlIiB0aXRsZT0iQ2xpY2sgdG8gZXhwYW5kIj48L2Rpdj5gOwogICAgY2FyZC5xdWVyeVNlbGVjdG9yKCIuY3Rha2UiKS5v"
    "bmNsaWNrID0gKGUpPT5lLmN1cnJlbnRUYXJnZXQuY2xhc3NMaXN0LnRvZ2dsZSgib3BlbiIpOwogICAgZy5hcHBlbmRDaGlsZChjYXJkKTsKICB9KTsK"
    "ICBzZWMuYXBwZW5kQ2hpbGQoZyk7IHBhbmVsLmFwcGVuZENoaWxkKHNlYyk7Cn0KZnVuY3Rpb24gY291bmNpbEZpbGwocGFuZWwsIGlkLCByb3VuZCwg"
    "dGV4dCl7CiAgY29uc3QgY2FyZCA9IHBhbmVsLnF1ZXJ5U2VsZWN0b3IoYC5jY2FyZFtkYXRhLWNpZD0iJHtpZH0iXVtkYXRhLXJvdW5kPSIke3JvdW5k"
    "fSJdYCk7CiAgaWYoIWNhcmQpIHJldHVybjsKICBjYXJkLnF1ZXJ5U2VsZWN0b3IoIi5jdGFrZSIpLnRleHRDb250ZW50ID0gdGV4dCB8fCAiIjsKICBj"
    "YXJkLnF1ZXJ5U2VsZWN0b3IoIi5jc3QiKS5pbm5lckhUTUwgPSAnPHNwYW4gc3R5bGU9ImNvbG9yOnZhcigtLWxpdmUpO2ZvbnQtc2l6ZToxMnB4Ij7i"
    "nJM8L3NwYW4+JzsKfQpmdW5jdGlvbiBjb3VuY2lsRmFpbChwYW5lbCwgaWQsIHJvdW5kKXsKICBjb25zdCBjYXJkID0gcGFuZWwucXVlcnlTZWxlY3Rv"
    "cihgLmNjYXJkW2RhdGEtY2lkPSIke2lkfSJdW2RhdGEtcm91bmQ9IiR7cm91bmR9Il1gKTsKICBpZighY2FyZCkgcmV0dXJuOwogIGNhcmQuY2xhc3NM"
    "aXN0LmFkZCgiZmFpbGVkIik7CiAgY2FyZC5xdWVyeVNlbGVjdG9yKCIuY3N0IikuaW5uZXJIVE1MID0gJzxzcGFuIHN0eWxlPSJjb2xvcjp2YXIoLS1z"
    "aWduYWwpO2ZvbnQtc2l6ZToxMnB4Ij7imqA8L3NwYW4+JzsKICBjYXJkLnF1ZXJ5U2VsZWN0b3IoIi5jdGFrZSIpLnRleHRDb250ZW50ID0gIihubyBy"
    "ZXNwb25zZSkiOwp9CmFzeW5jIGZ1bmN0aW9uIHNlbmRNZXNzYWdlKHJlZ2VuPWZhbHNlKXsKICBpZihzdGF0ZS5zZW5kaW5nKSByZXR1cm47CiAgY29u"
    "c3QgaW5wdXQgPSAkKCJjaGF0SW5wdXQiKTsKICBjb25zdCBtb2RlbCA9ICQoImNoYXRNb2RlbCIpLnZhbHVlOwogIGlmKCFtb2RlbCl7IHRvYXN0KCJE"
    "b3dubG9hZCBhIG1vZGVsIGZpcnN0IChNb2RlbHMgcGFnZSkuIiwiZXJyIik7IHNob3coIm1vZGVscyIpOyByZXR1cm47IH0KICBsZXQgdGV4dCA9ICIi"
    "LCBpbWdzID0gW107CiAgaWYocmVnZW4pewogICAgaWYoIXN0YXRlLmN1cnJlbnRDb252KSByZXR1cm47CiAgfSBlbHNlIHsKICAgIHRleHQgPSBpbnB1"
    "dC52YWx1ZS50cmltKCk7CiAgICBpZighdGV4dCl7IGlmKHBlbmRpbmdJbWFnZXMubGVuZ3RoKSB0b2FzdCgiQWRkIGEgcXVlc3Rpb24gYWJvdXQgdGhl"
    "IGltYWdlIGZpcnN0Iik7IHJldHVybjsgfQogICAgaW1ncyA9IHBlbmRpbmdJbWFnZXMuc2xpY2UoKTsKICAgIGlucHV0LnZhbHVlPSIiOyBhdXRvR3Jv"
    "dygpOwogICAgYWRkVXNlck1zZyh0ZXh0LCBpbWdzKTsKICAgIHBlbmRpbmdJbWFnZXMgPSBbXTsgcmVuZGVyQXR0YWNoKCk7CiAgfQogIGRvY3VtZW50"
    "LnF1ZXJ5U2VsZWN0b3JBbGwoIi5yZWdlbnJvdyIpLmZvckVhY2goeD0+eC5yZW1vdmUoKSk7CiAgY29uc3QgdWkgPSBhZGRBSU1zZygpOwogIGNvbnN0"
    "IHN0YXR1c0xpbmUgPSBlbCgiZGl2Iiwic3RhdHVzbGluZSIsJzxkaXYgY2xhc3M9InNwaW4iPjwvZGl2PjxzcGFuPlRoaW5raW5n4oCmPC9zcGFuPicp"
    "OwogIHVpLnN0YXR1cy5hcHBlbmRDaGlsZChzdGF0dXNMaW5lKTsKICBzdGF0ZS5zZW5kaW5nID0gdHJ1ZTsgc3RhdGUuYWJvcnQgPSBuZXcgQWJvcnRD"
    "b250cm9sbGVyKCk7CiAgc2V0U2VuZE1vZGUodHJ1ZSk7CiAgbGV0IGFuc3dlciA9ICIiOyBsZXQgcGVuZGluZ1N0ZXAgPSBudWxsOyBsZXQgZ290VG9r"
    "ZW49ZmFsc2U7CiAgbGV0IHRoaW5rQm94PW51bGwsIHRoaW5rVGV4dD0iIjsKICBjb25zdCBmaW5hbGl6ZVRoaW5rID0gKGxhYmVsKT0+ewogICAgaWYo"
    "IXRoaW5rQm94KSByZXR1cm47CiAgICBjb25zdCBzcCA9IHRoaW5rQm94LnF1ZXJ5U2VsZWN0b3IoIi50aCAuc3BpbiIpOwogICAgaWYoc3ApIHNwLm91"
    "dGVySFRNTCA9ICc8c3BhbiBzdHlsZT0iY29sb3I6dmFyKC0tdmlvbGV0KSI+4pyzPC9zcGFuPic7CiAgICBjb25zdCBsYmwgPSB0aGlua0JveC5xdWVy"
    "eVNlbGVjdG9yKCIubGJsIik7CiAgICBpZihsYmwpIGxibC50ZXh0Q29udGVudCA9IGxhYmVsIHx8ICJUaG91Z2h0IHByb2Nlc3MiOwogICAgY29uc3Qg"
    "dGIgPSB0aGlua0JveC5xdWVyeVNlbGVjdG9yKCIudGIiKTsgaWYodGIpIHRiLmNsYXNzTGlzdC5yZW1vdmUoInN0cmVhbWluZyIpOwogICAgdGhpbmtC"
    "b3guY2xhc3NMaXN0LmFkZCgiY29sbGFwc2VkIik7CiAgfTsKICBsZXQgY291bmNpbFBhbmVsPW51bGwsIGNvbnN1bHRhbnRzPVtdOyBjb25zdCBmYWls"
    "ZWRJZHM9bmV3IFNldCgpOwogIGxldCBjY0ZlZWQ9bnVsbDsKICBmdW5jdGlvbiBlbnN1cmVDQ0ZlZWQoKXsKICAgIGlmKGNjRmVlZCkgcmV0dXJuIGNj"
    "RmVlZDsKICAgIGNjRmVlZCA9IGVsKCJkaXYiLCJjY2ZlZWQiKTsKICAgIGNjRmVlZC5pbm5lckhUTUwgPSBgPGRpdiBjbGFzcz0iY2ZoIj7wn5alIENv"
    "bXB1dGVyIGNvbnRyb2wKICAgICAgPGJ1dHRvbiBjbGFzcz0ibWluaS1lc3RvcCIgc3R5bGU9Im1hcmdpbi1sZWZ0OmF1dG8iPgogICAgICAgIDxzdmcg"
    "d2lkdGg9IjEyIiBoZWlnaHQ9IjEyIiB2aWV3Qm94PSIwIDAgMjQgMjQiIGZpbGw9ImN1cnJlbnRDb2xvciI+PHJlY3QgeD0iNiIgeT0iNiIgd2lkdGg9"
    "IjEyIiBoZWlnaHQ9IjEyIiByeD0iMiIvPjwvc3ZnPgogICAgICAgIFNUT1A8L2J1dHRvbj48L2Rpdj48ZGl2IGNsYXNzPSJjY2JvZHkiPjwvZGl2PmA7"
    "CiAgICBjY0ZlZWQucXVlcnlTZWxlY3RvcigiLm1pbmktZXN0b3AiKS5vbmNsaWNrID0gZW1lcmdlbmN5U3RvcDsKICAgIHVpLmFnZW50bG9nLmFwcGVu"
    "ZENoaWxkKGNjRmVlZCk7CiAgICByZXR1cm4gY2NGZWVkOwogIH0KICBjb25zdCBjbGVhclNwaW5uZXJzID0gKCk9PnsKICAgIGlmKHN0YXR1c0xpbmUu"
    "cGFyZW50Tm9kZSkgc3RhdHVzTGluZS5yZW1vdmUoKTsKICAgIHVpLmFnZW50bG9nLnF1ZXJ5U2VsZWN0b3JBbGwoIi5zdGF0dXNsaW5lIikuZm9yRWFj"
    "aCh4PT54LnJlbW92ZSgpKTsKICB9OwoKICB0cnl7CiAgICBjb25zdCBwYXlsb2FkID0gewogICAgICBtb2RlbCwgdXNlX3JhZzogcmFnT24sIGFnZW50"
    "X21vZGU6IGFnZW50T24sIGxvb3BfbW9kZTogbG9vcE9uLAogICAgICBjb3VuY2lsX21vZGU6IGNvdW5jaWxPbiwgY29tcHV0ZXJfbW9kZTogY29tcHV0"
    "ZXJPbiwgY29kZXJfbW9kZTogY29kZXJPbgogICAgfTsKICAgIGlmKCFyZWdlbil7CiAgICAgIHBheWxvYWQubWVzc2FnZSA9IHRleHQ7IHBheWxvYWQu"
    "Y29udmVyc2F0aW9uX2lkID0gc3RhdGUuY3VycmVudENvbnY7CiAgICAgIGlmKGltZ3MubGVuZ3RoKSBwYXlsb2FkLmltYWdlcyA9IGltZ3M7CiAgICB9"
    "CiAgICBhd2FpdCBzdHJlYW1OREpTT04ocmVnZW4KICAgICAgICA/ICIvYXBpL2NvbnZlcnNhdGlvbnMvIitzdGF0ZS5jdXJyZW50Q29udisiL3JlZ2Vu"
    "ZXJhdGUiCiAgICAgICAgOiAiL2FwaS9jaGF0IiwgcGF5bG9hZCwgKG8pPT57CiAgICAgIGlmKG8udHlwZT09PSJtZXRhIil7IGlmKCFzdGF0ZS5jdXJy"
    "ZW50Q29udil7IHN0YXRlLmN1cnJlbnRDb252PW8uY29udmVyc2F0aW9uX2lkOwogICAgICAgIGxvYWRDb252cygpOyB9IH0KICAgICAgZWxzZSBpZihv"
    "LnR5cGU9PT0ic291cmNlcyIpeyByZW5kZXJTb3VyY2VzKHVpLnNvdXJjZXMsIG8uaXRlbXN8fFtdKTsgfQogICAgICBlbHNlIGlmKG8udHlwZT09PSJh"
    "dXRvX3NlYXJjaCIpewogICAgICAgIGNsZWFyU3Bpbm5lcnMoKTsKICAgICAgICB1aS5hZ2VudGxvZy5hcHBlbmRDaGlsZCh3ZWJTZWFyY2hFbChvLnF1"
    "ZXJ5LCBvLnJlc3VsdCkpOwogICAgICAgIHVpLmFnZW50bG9nLmFwcGVuZENoaWxkKGVsKCJkaXYiLCJzdGF0dXNsaW5lIiwKICAgICAgICAgICc8ZGl2"
    "IGNsYXNzPSJzcGluIj48L2Rpdj48c3Bhbj5SZWFkaW5nIHJlc3VsdHPigKY8L3NwYW4+JykpOyBzY3JvbGxDaGF0KCk7CiAgICAgIH0KICAgICAgZWxz"
    "ZSBpZihvLnR5cGU9PT0ic3RhdHVzIil7IHN0YXR1c0xpbmUucXVlcnlTZWxlY3Rvcigic3BhbiIpLnRleHRDb250ZW50ID0gby50ZXh0OwogICAgICAg"
    "IGlmKCFzdGF0dXNMaW5lLnBhcmVudE5vZGUgJiYgIWdvdFRva2VuKSB1aS5zdGF0dXMuYXBwZW5kQ2hpbGQoc3RhdHVzTGluZSk7IH0KICAgICAgZWxz"
    "ZSBpZihvLnR5cGU9PT0iY29tcHV0ZXJfYWN0aW9uIil7CiAgICAgICAgY2xlYXJTcGlubmVycygpOwogICAgICAgIGNvbnN0IGJvZHkgPSBlbnN1cmVD"
    "Q0ZlZWQoKS5xdWVyeVNlbGVjdG9yKCIuY2Nib2R5Iik7CiAgICAgICAgY29uc3Qgcm93ID0gZWwoImRpdiIsImNjYWN0Iik7CiAgICAgICAgcm93Lmlu"
    "bmVySFRNTCA9IGA8c3BhbiBjbGFzcz0iYWljIj4ke0NDX0lDT05TW28ua2luZF18fCLigKIifTwvc3Bhbj4KICAgICAgICAgIDxzcGFuPiR7ZXNjKG8u"
    "ZGV0YWlsfHxvLmtpbmQpfTwvc3Bhbj4KICAgICAgICAgICR7by5zaG90P2A8aW1nIHNyYz0iL2FwaS9jb21wdXRlci9zaG90LyR7by5zaG90fSIgdGl0"
    "bGU9IldoYXQgdGhlIG1vZGVsIHNhdyI+YDoiIn1gOwogICAgICAgIGlmKG8uc2hvdCkgcm93LnF1ZXJ5U2VsZWN0b3IoImltZyIpLm9uY2xpY2s9KCk9"
    "PndpbmRvdy5vcGVuKCIvYXBpL2NvbXB1dGVyL3Nob3QvIitvLnNob3QpOwogICAgICAgIGJvZHkuYXBwZW5kQ2hpbGQocm93KTsKICAgICAgICB1aS5h"
    "Z2VudGxvZy5hcHBlbmRDaGlsZChlbCgiZGl2Iiwic3RhdHVzbGluZSIsCiAgICAgICAgICAnPGRpdiBjbGFzcz0ic3BpbiI+PC9kaXY+PHNwYW4+V29y"
    "a2luZ+KApjwvc3Bhbj4nKSk7IHNjcm9sbENoYXQoKTsKICAgICAgfQogICAgICBlbHNlIGlmKG8udHlwZT09PSJjb21wdXRlcl9jb25maXJtIil7CiAg"
    "ICAgICAgY2xlYXJTcGlubmVycygpOwogICAgICAgIGNvbnN0IGJvZHkgPSBlbnN1cmVDQ0ZlZWQoKS5xdWVyeVNlbGVjdG9yKCIuY2Nib2R5Iik7CiAg"
    "ICAgICAgY29uc3QgYm94ID0gZWwoImRpdiIsImNjY29uZmlybSIpOwogICAgICAgIGJveC5pbm5lckhUTUwgPSBgPGRpdiBjbGFzcz0icSI+QWxsb3cg"
    "dGhlIG1vZGVsIHRvIDxiPiR7ZXNjKG8uYWN0aW9uKX08L2I+PzwvZGl2PmA7CiAgICAgICAgY29uc3QgeWVzID0gZWwoImJ1dHRvbiIsImJ0biBzbSBw"
    "cmltYXJ5IiwiQWxsb3ciKTsKICAgICAgICBjb25zdCBubyA9IGVsKCJidXR0b24iLCJidG4gc20gZ2hvc3QiLCJEZW55Iik7CiAgICAgICAgY29uc3Qg"
    "ZG9uZSA9IChhcHByb3ZlZCk9PnsgYm94LnF1ZXJ5U2VsZWN0b3JBbGwoImJ1dHRvbiIpLmZvckVhY2goYj0+Yi5kaXNhYmxlZD10cnVlKTsKICAgICAg"
    "ICAgIGJveC5xdWVyeVNlbGVjdG9yKCIucSIpLmlubmVySFRNTCA9IChhcHByb3ZlZD8i4pyTIEFsbG93ZWQ6ICI6IuKclyBEZW5pZWQ6ICIpKyI8Yj4i"
    "K2VzYyhvLmFjdGlvbikrIjwvYj4iOwogICAgICAgICAgcG9zdCgiL2FwaS9jb21wdXRlci9jb25maXJtIix7aWQ6by5pZCwgYXBwcm92ZWR9KS5jYXRj"
    "aCgoKT0+e30pOyB9OwogICAgICAgIHllcy5vbmNsaWNrPSgpPT5kb25lKHRydWUpOyBuby5vbmNsaWNrPSgpPT5kb25lKGZhbHNlKTsKICAgICAgICBi"
    "b3guYXBwZW5kKHllcyxubyk7IGJvZHkuYXBwZW5kQ2hpbGQoYm94KTsKICAgICAgICB1aS5hZ2VudGxvZy5hcHBlbmRDaGlsZChlbCgiZGl2Iiwic3Rh"
    "dHVzbGluZSIsCiAgICAgICAgICAnPGRpdiBjbGFzcz0ic3BpbiI+PC9kaXY+PHNwYW4+V2FpdGluZyBmb3IgeW91ciBhcHByb3ZhbOKApjwvc3Bhbj4n"
    "KSk7IHNjcm9sbENoYXQoKTsKICAgICAgfQogICAgICBlbHNlIGlmKG8udHlwZT09PSJjb21wdXRlcl9zdG9wcGVkIil7CiAgICAgICAgY2xlYXJTcGlu"
    "bmVycygpOwogICAgICAgIGNvbnN0IGJvZHkgPSBlbnN1cmVDQ0ZlZWQoKS5xdWVyeVNlbGVjdG9yKCIuY2Nib2R5Iik7CiAgICAgICAgYm9keS5hcHBl"
    "bmRDaGlsZChlbCgiZGl2IiwiaGludCIsJ+KblCAnK2VzYyhvLnJlYXNvbnx8IlN0b3BwZWQuIikpKTsKICAgICAgICBzY3JvbGxDaGF0KCk7CiAgICAg"
    "IH0KICAgICAgZWxzZSBpZihvLnR5cGU9PT0iY291bmNpbF9zdGFydCIpewogICAgICAgIGNvbnN1bHRhbnRzID0gby5jb25zdWx0YW50c3x8W107CiAg"
    "ICAgICAgY291bmNpbFBhbmVsID0gY291bmNpbFBhbmVsRWwobyk7CiAgICAgICAgdWkuYWdlbnRsb2cuYXBwZW5kQ2hpbGQoY291bmNpbFBhbmVsKTsg"
    "c2Nyb2xsQ2hhdCgpOwogICAgICB9CiAgICAgIGVsc2UgaWYoby50eXBlPT09ImNvdW5jaWxfYnJpZWYiKXsKICAgICAgICBpZihjb3VuY2lsUGFuZWwp"
    "IGNvdW5jaWxQYW5lbC5hcHBlbmRDaGlsZChjb3VuY2lsQnJpZWZFbChvLnRleHQpKTsKICAgICAgfQogICAgICBlbHNlIGlmKG8udHlwZT09PSJjb3Vu"
    "Y2lsX3JvdW5kIil7CiAgICAgICAgaWYoY291bmNpbFBhbmVsKXsgY291bmNpbFJvdW5kRWwoY291bmNpbFBhbmVsLCBvLnJvdW5kLCBvLmxhYmVsLAog"
    "ICAgICAgICAgY29uc3VsdGFudHMsIGZhaWxlZElkcywgdHJ1ZSk7IHNjcm9sbENoYXQoKTsgfQogICAgICB9CiAgICAgIGVsc2UgaWYoby50eXBlPT09"
    "ImNvdW5jaWxfdGFrZSIpewogICAgICAgIGlmKGNvdW5jaWxQYW5lbCkgY291bmNpbEZpbGwoY291bmNpbFBhbmVsLCBvLmlkLCBvLnJvdW5kLCBvLnRl"
    "eHQpOwogICAgICB9CiAgICAgIGVsc2UgaWYoby50eXBlPT09ImNvbnN1bHRhbnRfc3RhdHVzIil7CiAgICAgICAgaWYoby5zdGF0ZT09PSJmYWlsZWQi"
    "ICYmIGNvdW5jaWxQYW5lbCl7IGZhaWxlZElkcy5hZGQoby5pZCk7CiAgICAgICAgICBjb3VuY2lsRmFpbChjb3VuY2lsUGFuZWwsIG8uaWQsIG8ucm91"
    "bmQpOyB9CiAgICAgIH0KICAgICAgZWxzZSBpZihvLnR5cGU9PT0idGhvdWdodCIpewogICAgICAgIGNsZWFyU3Bpbm5lcnMoKTsKICAgICAgICB1aS5h"
    "Z2VudGxvZy5hcHBlbmRDaGlsZCh0aG91Z2h0RWwoby50ZXh0KSk7CiAgICAgICAgdWkuYWdlbnRsb2cuYXBwZW5kQ2hpbGQoZWwoImRpdiIsInN0YXR1"
    "c2xpbmUiLAogICAgICAgICAgJzxkaXYgY2xhc3M9InNwaW4iPjwvZGl2PjxzcGFuPldvcmtpbmfigKY8L3NwYW4+JykpOwogICAgICAgIHNjcm9sbENo"
    "YXQoKTsKICAgICAgfQogICAgICBlbHNlIGlmKG8udHlwZT09PSJ0b29sX2NhbGwiKXsKICAgICAgICBjbGVhclNwaW5uZXJzKCk7CiAgICAgICAgY29u"
    "c3QgcyA9IHN0ZXBFbChvLm5hbWUsIG8uYXJncyk7IHVpLmFnZW50bG9nLmFwcGVuZENoaWxkKHMpOyBwZW5kaW5nU3RlcD1zOwogICAgICAgIHVpLmFn"
    "ZW50bG9nLmFwcGVuZENoaWxkKGVsKCJkaXYiLCJzdGF0dXNsaW5lIiwKICAgICAgICAgICc8ZGl2IGNsYXNzPSJzcGluIj48L2Rpdj48c3Bhbj5SdW5u"
    "aW5nIHRvb2zigKY8L3NwYW4+JykpOwogICAgICAgIHNjcm9sbENoYXQoKTsKICAgICAgfQogICAgICBlbHNlIGlmKG8udHlwZT09PSJ0b29sX3Jlc3Vs"
    "dCIpewogICAgICAgIGNvbnN0IHNsID0gdWkuYWdlbnRsb2cucXVlcnlTZWxlY3RvcigiLnN0YXR1c2xpbmUiKTsgaWYoc2wpIHNsLnJlbW92ZSgpOwog"
    "ICAgICAgIGlmKHBlbmRpbmdTdGVwKXsgY29uc3Qgc3A9cGVuZGluZ1N0ZXAucXVlcnlTZWxlY3RvcigiLnNwaW4iKTsKICAgICAgICAgIGlmKHNwKSBz"
    "cC5vdXRlckhUTUw9JzxzcGFuIHN0eWxlPSJjb2xvcjp2YXIoLS1saXZlKSI+4pyTPC9zcGFuPic7CiAgICAgICAgICBwZW5kaW5nU3RlcC5xdWVyeVNl"
    "bGVjdG9yKCIuc2IiKS50ZXh0Q29udGVudCA9IG8ucmVzdWx0fHwiIjsgcGVuZGluZ1N0ZXA9bnVsbDsgfQogICAgICAgIHVpLmFnZW50bG9nLmFwcGVu"
    "ZENoaWxkKGVsKCJkaXYiLCJzdGF0dXNsaW5lIiwKICAgICAgICAgICc8ZGl2IGNsYXNzPSJzcGluIj48L2Rpdj48c3Bhbj5UaGlua2luZ+KApjwvc3Bh"
    "bj4nKSk7IHNjcm9sbENoYXQoKTsKICAgICAgfQogICAgICBlbHNlIGlmKG8udHlwZT09PSJ0aGlua2luZyIpewogICAgICAgIGlmKCFnb3RUb2tlbil7"
    "CiAgICAgICAgICBpZihzdGF0dXNMaW5lLnBhcmVudE5vZGUpIHN0YXR1c0xpbmUucXVlcnlTZWxlY3Rvcigic3BhbiIpLnRleHRDb250ZW50PSJSZWFz"
    "b25pbmfigKYiOwogICAgICAgIH0KICAgICAgICBpZighdGhpbmtCb3gpewogICAgICAgICAgdGhpbmtCb3ggPSBlbCgiZGl2IiwidGhpbmtpbmctYm94"
    "Iik7CiAgICAgICAgICB0aGlua0JveC5pbm5lckhUTUwgPSBgPGRpdiBjbGFzcz0idGgiPjxzcGFuIGNsYXNzPSJzcGluIiBzdHlsZT0id2lkdGg6MTFw"
    "eDtoZWlnaHQ6MTFweCI+PC9zcGFuPgogICAgICAgICAgICA8c3BhbiBjbGFzcz0ibGJsIj5UaGlua2luZ+KApjwvc3Bhbj48c3BhbiBjbGFzcz0iY3Yi"
    "PuKWvDwvc3Bhbj48L2Rpdj4KICAgICAgICAgICAgPGRpdiBjbGFzcz0idGIgc3RyZWFtaW5nIj48L2Rpdj5gOwogICAgICAgICAgdGhpbmtCb3gucXVl"
    "cnlTZWxlY3RvcigiLnRoIikub25jbGljaz0oKT0+dGhpbmtCb3guY2xhc3NMaXN0LnRvZ2dsZSgiY29sbGFwc2VkIik7CiAgICAgICAgICB1aS5hZ2Vu"
    "dGxvZy5hcHBlbmRDaGlsZCh0aGlua0JveCk7CiAgICAgICAgfQogICAgICAgIHRoaW5rVGV4dCArPSBvLnRleHQ7CiAgICAgICAgdGhpbmtCb3gucXVl"
    "cnlTZWxlY3RvcigiLnRiIikudGV4dENvbnRlbnQgPSB0aGlua1RleHQ7CiAgICAgICAgc2Nyb2xsQ2hhdCgpOwogICAgICB9CiAgICAgIGVsc2UgaWYo"
    "by50eXBlPT09InRva2VuIil7CiAgICAgICAgaWYoIWdvdFRva2VuKXsgZ290VG9rZW49dHJ1ZTsgY2xlYXJTcGlubmVycygpOyBmaW5hbGl6ZVRoaW5r"
    "KCJUaG91Z2h0IHByb2Nlc3MiKTsgfQogICAgICAgIGFuc3dlciArPSBvLnRleHQ7IHVpLmJ1YmJsZS5pbm5lckhUTUwgPSBtZChhbnN3ZXIpOwogICAg"
    "ICAgIHVpLmJ1YmJsZS5jbGFzc0xpc3QuYWRkKCJjdXJzb3ItYmxpbmsiKTsgc2Nyb2xsQ2hhdCgpOwogICAgICB9CiAgICAgIGVsc2UgaWYoby50eXBl"
    "PT09InN0YXRzIil7IGFwcGVuZFN0YXQodWkuYnViYmxlLCBvKTsgfQogICAgICBlbHNlIGlmKG8udHlwZT09PSJlcnJvciIpewogICAgICAgIGNsZWFy"
    "U3Bpbm5lcnMoKTsgZmluYWxpemVUaGluaygiVGhvdWdodCBwcm9jZXNzIik7CiAgICAgICAgdWkuYnViYmxlLmNsYXNzTGlzdC5yZW1vdmUoImN1cnNv"
    "ci1ibGluayIpOwogICAgICAgIHVpLmJ1YmJsZS5pbm5lckhUTUwgKz0gYDxkaXYgY2xhc3M9ImNhcmQiIHN0eWxlPSJib3JkZXItY29sb3I6dmFyKC0t"
    "cmVkKTttYXJnaW4tdG9wOjhweCI+CiAgICAgICAgICA8c3BhbiBzdHlsZT0iY29sb3I6dmFyKC0tcmVkKSI+JHtlc2Moby5lcnJvcil9PC9zcGFuPjwv"
    "ZGl2PmA7CiAgICAgIH0KICAgICAgZWxzZSBpZihvLnR5cGU9PT0iZG9uZSIpeyB1aS5idWJibGUuY2xhc3NMaXN0LnJlbW92ZSgiY3Vyc29yLWJsaW5r"
    "Iik7IGZpbmFsaXplVGhpbmsoIlRob3VnaHQgcHJvY2VzcyIpOyB9CiAgICB9LCBzdGF0ZS5hYm9ydC5zaWduYWwpOwogIH1jYXRjaChlKXsKICAgIGNs"
    "ZWFyU3Bpbm5lcnMoKTsgZmluYWxpemVUaGluaygiVGhvdWdodCBwcm9jZXNzIik7CiAgICB1aS5idWJibGUuY2xhc3NMaXN0LnJlbW92ZSgiY3Vyc29y"
    "LWJsaW5rIik7CiAgICBpZihlLm5hbWU9PT0iQWJvcnRFcnJvciIpewogICAgICBpZihwZW5kaW5nU3RlcCl7IGNvbnN0IHNwPXBlbmRpbmdTdGVwLnF1"
    "ZXJ5U2VsZWN0b3IoIi5zcGluIik7CiAgICAgICAgaWYoc3ApIHNwLm91dGVySFRNTD0nPHNwYW4gY2xhc3M9ImRpbSI+4pagPC9zcGFuPic7IH0KICAg"
    "ICAgdWkuYnViYmxlLmluc2VydEFkamFjZW50SFRNTCgiYmVmb3JlZW5kIiwKICAgICAgICAnPGRpdiBjbGFzcz0iaGludCIgc3R5bGU9Im1hcmdpbi10"
    "b3A6OHB4Ij7ilqAgU3RvcHBlZCDigJQgcGFydGlhbCByZXBseSBrZXB0LjwvZGl2PicpOwogICAgfSBlbHNlIHsKICAgICAgdWkuYnViYmxlLmlubmVy"
    "SFRNTCArPSBgPGRpdiBjbGFzcz0iY2FyZCIgc3R5bGU9ImJvcmRlci1jb2xvcjp2YXIoLS1yZWQpO21hcmdpbi10b3A6OHB4Ij4KICAgICAgICA8c3Bh"
    "biBzdHlsZT0iY29sb3I6dmFyKC0tcmVkKSI+JHtlc2MoZS5tZXNzYWdlKX08L3NwYW4+PC9kaXY+YDsKICAgIH0KICB9ZmluYWxseXsKICAgIHN0YXRl"
    "LnNlbmRpbmc9ZmFsc2U7IHN0YXRlLmFib3J0PW51bGw7IHNldFNlbmRNb2RlKGZhbHNlKTsKICAgIGNsZWFyU3Bpbm5lcnMoKTsgZmluYWxpemVUaGlu"
    "aygiVGhvdWdodCBwcm9jZXNzIik7CiAgICB1aS5idWJibGUuY2xhc3NMaXN0LnJlbW92ZSgiY3Vyc29yLWJsaW5rIik7CiAgICBpZihzdGF0ZS5jdXJy"
    "ZW50Q29udil7CiAgICAgIGNvbnN0IHJvdyA9IGVsKCJkaXYiLCJyZWdlbnJvdyIpOwogICAgICByb3cuaW5uZXJIVE1MID0gJzxidXR0b24gY2xhc3M9"
    "ImNidG4iIHRpdGxlPSJEZWxldGUgdGhpcyBhbnN3ZXIgYW5kIGdlbmVyYXRlIGEgbmV3IG9uZSI+XHUyMWJiIFJlZ2VuZXJhdGU8L2J1dHRvbj4nOwog"
    "ICAgICByb3cucXVlcnlTZWxlY3RvcigiYnV0dG9uIikub25jbGljayA9ICgpPT57CiAgICAgICAgaWYoc3RhdGUuc2VuZGluZykgcmV0dXJuOwogICAg"
    "ICAgIHVpLnJvb3QucmVtb3ZlKCk7IHNlbmRNZXNzYWdlKHRydWUpOwogICAgICB9OwogICAgICB1aS5idWJibGUucGFyZW50Tm9kZS5hcHBlbmRDaGls"
    "ZChyb3cpOwogICAgfQogICAgc2Nyb2xsQ2hhdCgpOwogIH0KfQpsb2FkZXJzLmNoYXQgPSAoKT0+eyByZWZyZXNoQ2hhdE1vZGVscygpOyBsb2FkQ29u"
    "dnMoKTsKICBpZighc3RhdGUuY3VycmVudENvbnYgJiYgISQoImNoYXRXcmFwIikuY2hpbGRyZW4ubGVuZ3RoKSBzaG93Q2hhdEVtcHR5KCk7CiAgc2V0"
    "VGltZW91dCgoKT0+JCgiY2hhdElucHV0IikuZm9jdXMoKSw1MCk7IH07CgovKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09"
    "PT09PT09PT09PT09PT09PT09PT0KICAgSU1BR0VTCiAgID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09"
    "PT09PT09PSAqLwphc3luYyBmdW5jdGlvbiBsb2FkSW1hZ2VzKCl7CiAgY29uc3QgaG9zdCA9ICQoImltYWdlc0JvZHkiKTsKICBob3N0LmlubmVySFRN"
    "TCA9ICc8ZGl2IGNsYXNzPSJyb3ciPjxkaXYgY2xhc3M9InNwaW4iPjwvZGl2PjxzcGFuIGNsYXNzPSJtdXRlZCI+TG9hZGluZ+KApjwvc3Bhbj48L2Rp"
    "dj4nOwogIGxldCBzdDsKICB0cnl7IHN0ID0gYXdhaXQgYXBpKCIvYXBpL2ltYWdlcy9zdGF0dXMiKTsgfWNhdGNoKGUpeyBob3N0LmlubmVySFRNTD1l"
    "cnJDYXJkKGUubWVzc2FnZSk7IHJldHVybjsgfQogIHN0YXRlLmltZ1ByZXNldHMgPSBzdC5wcmVzZXRzfHxbXTsKICBpZighc3QuaW5zdGFsbGVkKXsK"
    "ICAgIGhvc3QuaW5uZXJIVE1MID0gYDxkaXYgY2xhc3M9ImltZ2xheW91dCI+CiAgICAgIDxkaXYgY2xhc3M9ImdlbnBhbmVsIiBpZD0ic2V0dXBIb3N0"
    "Ij48L2Rpdj4KICAgICAgPGRpdj48ZGl2IGNsYXNzPSJzZWN0aW9uLXRpdGxlIiBzdHlsZT0ibWFyZ2luLXRvcDowIj5HYWxsZXJ5PC9kaXY+CiAgICAg"
    "ICAgPGRpdiBjbGFzcz0iZ2FsbGVyeSIgaWQ9ImdhbGxlcnkiPjwvZGl2PjwvZGl2PgogICAgPC9kaXY+YDsKICAgIHJlbmRlckltYWdlU2V0dXAoJCgi"
    "c2V0dXBIb3N0IiksIHN0KTsKICAgIGxvYWRHYWxsZXJ5KCk7CiAgICByZXR1cm47CiAgfQogIGhvc3QuaW5uZXJIVE1MID0gYDxkaXYgY2xhc3M9Imlt"
    "Z2xheW91dCI+CiAgICA8ZGl2IGNsYXNzPSJnZW5wYW5lbCI+CiAgICAgIDxkaXYgY2xhc3M9ImNhcmQiPgogICAgICAgIDxkaXYgY2xhc3M9ImltZ3By"
    "ZXZpZXciIGlkPSJpbWdQcmV2aWV3Ij48ZGl2IGNsYXNzPSJwaCI+WW91ciBpbWFnZSBhcHBlYXJzIGhlcmU8L2Rpdj48L2Rpdj4KICAgICAgICA8ZGl2"
    "IGNsYXNzPSJmaWVsZCI+PGxhYmVsPlByb21wdDwvbGFiZWw+CiAgICAgICAgICA8dGV4dGFyZWEgY2xhc3M9InRhIiBpZD0iaW1nUHJvbXB0IiBwbGFj"
    "ZWhvbGRlcj0iYSBsaWdodGhvdXNlIGF0IGR1c2ssIG9pbCBwYWludGluZywgZHJhbWF0aWMgc2t5Ij48L3RleHRhcmVhPjwvZGl2PgogICAgICAgIDxk"
    "aXYgY2xhc3M9ImZpZWxkIj48bGFiZWw+TW9kZWw8L2xhYmVsPgogICAgICAgICAgPHNlbGVjdCBjbGFzcz0ic2VsIiBpZD0iaW1nTW9kZWwiPiR7c3Rh"
    "dGUuaW1nUHJlc2V0cy5tYXAocD0+CiAgICAgICAgICAgIGA8b3B0aW9uIHZhbHVlPSIke3Aua2V5fSI+JHtlc2MocC5sYWJlbCl9PC9vcHRpb24+YCku"
    "am9pbigiIil9PC9zZWxlY3Q+PC9kaXY+CiAgICAgICAgPGRpdiBjbGFzcz0iZmllbGQiPjxsYWJlbD5TaXplPC9sYWJlbD48ZGl2IGNsYXNzPSJzaXpl"
    "Z3JpZCIgaWQ9InNpemVHcmlkIj48L2Rpdj48L2Rpdj4KICAgICAgICA8ZGl2IGNsYXNzPSJmaWVsZCI+PGxhYmVsPlNlZWQgPHNwYW4gY2xhc3M9ImRp"
    "bSI+KGJsYW5rID0gcmFuZG9tKTwvc3Bhbj48L2xhYmVsPgogICAgICAgICAgPGlucHV0IGNsYXNzPSJpbnAgbW9ubyIgaWQ9ImltZ1NlZWQiIHBsYWNl"
    "aG9sZGVyPSJyYW5kb20iPjwvZGl2PgogICAgICAgIDxidXR0b24gY2xhc3M9ImJ0biBwcmltYXJ5IiBpZD0iZ2VuQnRuIiBzdHlsZT0id2lkdGg6MTAw"
    "JSI+CiAgICAgICAgICA8c3ZnIHdpZHRoPSIxNSIgaGVpZ2h0PSIxNSIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9ImN1cnJl"
    "bnRDb2xvciIgc3Ryb2tlLXdpZHRoPSIyIj48cGF0aCBkPSJNNSAzdjRNMyA1aDRNNiAxN3Y0TTQgMTloNE0xMyAzbDIuNSA2LjVMMjIgMTJsLTYuNSAy"
    "LjVMMTMgMjFsLTIuNS02LjVMNCAxMmw2LjUtMi41TDEzIDNaIi8+PC9zdmc+CiAgICAgICAgICBHZW5lcmF0ZTwvYnV0dG9uPgogICAgICAgIDxkaXYg"
    "aWQ9ImdlblByb2dyZXNzIj48L2Rpdj4KICAgICAgPC9kaXY+CiAgICAgIDxkaXYgY2xhc3M9ImhpbnQiPkRldmljZTogPHNwYW4gY2xhc3M9Im1vbm8i"
    "PiR7KHN0LmRldmljZXx8ImNwdSIpLnRvVXBwZXJDYXNlKCl9PC9zcGFuPiDCtwogICAgICAgIGltYWdlcyBzYXZlIHRvIHlvdXIgZ2FsbGVyeSBhdXRv"
    "bWF0aWNhbGx5PC9kaXY+CiAgICA8L2Rpdj4KICAgIDxkaXY+PGRpdiBjbGFzcz0ic2VjdGlvbi10aXRsZSIgc3R5bGU9Im1hcmdpbi10b3A6MCI+R2Fs"
    "bGVyeTwvZGl2PgogICAgICA8ZGl2IGNsYXNzPSJnYWxsZXJ5IiBpZD0iZ2FsbGVyeSI+PC9kaXY+PC9kaXY+CiAgPC9kaXY+YDsKICBjb25zdCBzaXpl"
    "cyA9IFtbNTEyLDUxMiwic3F1YXJlIl0sWzUxMiw3NjgsInBvcnRyYWl0Il0sWzc2OCw1MTIsImxhbmRzY2FwZSJdXTsKICBzdGF0ZS5zZWxJbWdTaXpl"
    "ID0gc3RhdGUuc2VsSW1nU2l6ZSB8fCAiNTEyeDUxMiI7CiAgJCgic2l6ZUdyaWQiKS5pbm5lckhUTUwgPSBzaXplcy5tYXAoKFt3LGgsbF0pPT4KICAg"
    "IGA8ZGl2IGNsYXNzPSJzaXplb3B0JHtzdGF0ZS5zZWxJbWdTaXplPT09dysieCIraD8iIG9uIjoiIn0iIGRhdGEtc2l6ZT0iJHt3fXgke2h9Ij4KICAg"
    "ICAgJHtsfTxzcGFuIGNsYXNzPSJtb25vIj4ke3d9w5cke2h9PC9zcGFuPjwvZGl2PmApLmpvaW4oIiIpOwogICQoInNpemVHcmlkIikucXVlcnlTZWxl"
    "Y3RvckFsbCgiLnNpemVvcHQiKS5mb3JFYWNoKG89Pm8ub25jbGljaz0oKT0+ewogICAgc3RhdGUuc2VsSW1nU2l6ZT1vLmRhdGFzZXQuc2l6ZTsKICAg"
    "ICQoInNpemVHcmlkIikucXVlcnlTZWxlY3RvckFsbCgiLnNpemVvcHQiKS5mb3JFYWNoKHg9PnguY2xhc3NMaXN0LnJlbW92ZSgib24iKSk7CiAgICBv"
    "LmNsYXNzTGlzdC5hZGQoIm9uIik7IH0pOwogICQoImdlbkJ0biIpLm9uY2xpY2sgPSBnZW5lcmF0ZUltYWdlOwogIGxvYWRHYWxsZXJ5KCk7Cn0KZnVu"
    "Y3Rpb24gcmVuZGVySW1hZ2VTZXR1cChob3N0LCBzdCl7CiAgY29uc3QgcnVubmluZyA9IHN0Lmluc3RhbGxfam9iICYmIHN0Lmluc3RhbGxfam9iLnN0"
    "YXR1cz09PSJydW5uaW5nIjsKICBob3N0LmlubmVySFRNTCA9IGA8ZGl2IGNsYXNzPSJjYXJkIHNldHVwLWNhcmQiPgogICAgPGRpdiBjbGFzcz0iaWMi"
    "PvCfjqg8L2Rpdj4KICAgIDxoMyBzdHlsZT0ibWFyZ2luLWJvdHRvbTo4cHgiPlR1cm4gb24gaW1hZ2UgZ2VuZXJhdGlvbjwvaDM+CiAgICA8cCBjbGFz"
    "cz0ibXV0ZWQiIHN0eWxlPSJtYXgtd2lkdGg6NTJjaDttYXJnaW46MCBhdXRvIDZweCI+VGhpcyBpbnN0YWxscyBTdGFibGUgRGlmZnVzaW9uCiAgICAg"
    "IChQeVRvcmNoICsgZGlmZnVzZXJzKSBpbnRvIEhlb3J0aCdzIHByaXZhdGUgZW52aXJvbm1lbnQuIEl0J3MgYSBvbmUtdGltZQogICAgICBkb3dubG9h"
    "ZCBvZiBhIGZldyBnaWdhYnl0ZXMuIE1vZGVscyBkb3dubG9hZCB0aGUgZmlyc3QgdGltZSB5b3UgZ2VuZXJhdGUuPC9wPgogICAgPGRpdiBzdHlsZT0i"
    "bWFyZ2luLXRvcDoxNnB4Ij4KICAgICAgPGJ1dHRvbiBjbGFzcz0iYnRuIHByaW1hcnkiIGlkPSJzZXR1cEltZ0J0biIgJHtydW5uaW5nPyJkaXNhYmxl"
    "ZCI6IiJ9PgogICAgICAgICR7cnVubmluZz8iSW5zdGFsbGluZ+KApiI6Ikluc3RhbGwgaW1hZ2UgZ2VuZXJhdGlvbiJ9PC9idXR0b24+PC9kaXY+CiAg"
    "ICA8ZGl2IGNsYXNzPSJpbnN0YWxsbG9nIiBpZD0iaW1nTG9nIiAke3N0Lmluc3RhbGxfam9iPyIiOiJoaWRkZW4ifT4kewogICAgICBzdC5pbnN0YWxs"
    "X2pvYj9lc2MoKHN0Lmluc3RhbGxfam9iLmxvZ3x8W10pLmpvaW4oIlxuIikpOiIifTwvZGl2PgogIDwvZGl2PmA7CiAgJCgic2V0dXBJbWdCdG4iKS5v"
    "bmNsaWNrID0gYXN5bmMoKT0+ewogICAgJCgic2V0dXBJbWdCdG4iKS5kaXNhYmxlZD10cnVlOyAkKCJzZXR1cEltZ0J0biIpLnRleHRDb250ZW50PSJJ"
    "bnN0YWxsaW5n4oCmIjsKICAgICQoImltZ0xvZyIpLmhpZGRlbj1mYWxzZTsKICAgIHRyeXsgYXdhaXQgcG9zdCgiL2FwaS9pbWFnZXMvc2V0dXAiKTsg"
    "cG9sbEltZ0luc3RhbGwoKTsgfQogICAgY2F0Y2goZSl7IHRvYXN0KGUubWVzc2FnZSwiZXJyIik7IH0KICB9OwogIGlmKHJ1bm5pbmcpIHBvbGxJbWdJ"
    "bnN0YWxsKCk7Cn0KYXN5bmMgZnVuY3Rpb24gcG9sbEltZ0luc3RhbGwoKXsKICBjb25zdCBsb2cgPSAkKCJpbWdMb2ciKTsKICBjb25zdCB0ID0gc2V0"
    "SW50ZXJ2YWwoYXN5bmMoKT0+ewogICAgdHJ5eyBjb25zdCBzdCA9IGF3YWl0IGFwaSgiL2FwaS9pbWFnZXMvc3RhdHVzIik7CiAgICAgIGlmKGxvZyAm"
    "JiBzdC5pbnN0YWxsX2pvYil7IGxvZy50ZXh0Q29udGVudCA9IChzdC5pbnN0YWxsX2pvYi5sb2d8fFtdKS5qb2luKCJcbiIpOwogICAgICAgIGxvZy5z"
    "Y3JvbGxUb3AgPSBsb2cuc2Nyb2xsSGVpZ2h0OyB9CiAgICAgIGlmKHN0Lmluc3RhbGxlZCl7IGNsZWFySW50ZXJ2YWwodCk7IHRvYXN0KCJJbWFnZSBn"
    "ZW5lcmF0aW9uIHJlYWR5Iiwib2siKTsgbG9hZEltYWdlcygpOyB9CiAgICAgIGVsc2UgaWYoc3QuaW5zdGFsbF9qb2IgJiYgc3QuaW5zdGFsbF9qb2Iu"
    "c3RhdHVzPT09ImZhaWxlZCIpeyBjbGVhckludGVydmFsKHQpOwogICAgICAgIHRvYXN0KCJJbnN0YWxsIGZhaWxlZCDigJQgc2VlIHRoZSBsb2ciLCJl"
    "cnIiKTsKICAgICAgICBjb25zdCBiPSQoInNldHVwSW1nQnRuIik7IGlmKGIpe2IuZGlzYWJsZWQ9ZmFsc2U7Yi50ZXh0Q29udGVudD0iUmV0cnkgaW5z"
    "dGFsbCI7fSB9CiAgICB9Y2F0Y2goZSl7IGNsZWFySW50ZXJ2YWwodCk7IH0KICB9LCAxNTAwKTsKfQphc3luYyBmdW5jdGlvbiBnZW5lcmF0ZUltYWdl"
    "KCl7CiAgY29uc3QgcHJvbXB0ID0gJCgiaW1nUHJvbXB0IikudmFsdWUudHJpbSgpOwogIGlmKCFwcm9tcHQpeyB0b2FzdCgiV3JpdGUgYSBwcm9tcHQg"
    "Zmlyc3QiLCJlcnIiKTsgcmV0dXJuOyB9CiAgY29uc3QgW3csaF0gPSAoc3RhdGUuc2VsSW1nU2l6ZXx8IjUxMng1MTIiKS5zcGxpdCgieCIpLm1hcChO"
    "dW1iZXIpOwogIGNvbnN0IG1vZGVsID0gJCgiaW1nTW9kZWwiKS52YWx1ZTsKICBjb25zdCBzZWVkViA9ICQoImltZ1NlZWQiKS52YWx1ZS50cmltKCk7"
    "CiAgY29uc3QgYnRuID0gJCgiZ2VuQnRuIik7IGJ0bi5kaXNhYmxlZD10cnVlOwogIGNvbnN0IHByb2cgPSAkKCJnZW5Qcm9ncmVzcyIpOwogIHByb2cu"
    "aW5uZXJIVE1MID0gJzxkaXYgY2xhc3M9InB1bGxib3giPjxkaXYgY2xhc3M9InN0YXQiPjxzcGFuIGNsYXNzPSJzIj5TdGFydGluZ+KApjwvc3Bhbj4n"
    "KwogICAgJzxzcGFuIGNsYXNzPSJwIj48L3NwYW4+PC9kaXY+PGRpdiBjbGFzcz0iYmFyIj48aT48L2k+PC9kaXY+JysKICAgICc8ZGl2IGNsYXNzPSJy"
    "b3ciIHN0eWxlPSJqdXN0aWZ5LWNvbnRlbnQ6ZmxleC1lbmQ7bWFyZ2luLXRvcDo4cHgiPicrCiAgICAnPGJ1dHRvbiBjbGFzcz0iYnRuIHNtIGRhbmdl"
    "ciIgaWQ9ImltZ1N0b3BCdG4iPlN0b3A8L2J1dHRvbj48L2Rpdj48L2Rpdj4nOwogIGNvbnN0IGJhciA9IHByb2cucXVlcnlTZWxlY3RvcigiaSIpLCBz"
    "dGF0ID0gcHJvZy5xdWVyeVNlbGVjdG9yKCIucyIpLCBwYyA9IHByb2cucXVlcnlTZWxlY3RvcigiLnAiKTsKICAkKCJpbWdTdG9wQnRuIikub25jbGlj"
    "ayA9IGFzeW5jKCk9PnsgJCgiaW1nU3RvcEJ0biIpLmRpc2FibGVkPXRydWU7CiAgICAkKCJpbWdTdG9wQnRuIikudGV4dENvbnRlbnQ9IlN0b3BwaW5n"
    "4oCmIjsgdHJ5eyBhd2FpdCBwb3N0KCIvYXBpL2ltYWdlcy9jYW5jZWwiKTsgfWNhdGNoKGUpe30gfTsKICAkKCJpbWdQcmV2aWV3IikuaW5uZXJIVE1M"
    "ID0gJzxkaXYgY2xhc3M9InNwaW4iIHN0eWxlPSJ3aWR0aDoyNnB4O2hlaWdodDoyNnB4Ij48L2Rpdj4nOwogIHRyeXsKICAgIGF3YWl0IHN0cmVhbU5E"
    "SlNPTigiL2FwaS9pbWFnZXMvZ2VuZXJhdGUiLAogICAgICB7cHJvbXB0LCBtb2RlbCwgd2lkdGg6dywgaGVpZ2h0OmgsIHNlZWQ6IHNlZWRWPT09IiI/"
    "bnVsbDpOdW1iZXIoc2VlZFYpfSwKICAgICAgKG8pPT57CiAgICAgICAgaWYoby50eXBlPT09InN0YXR1cyIpeyBzdGF0LnRleHRDb250ZW50PW8udGV4"
    "dDsgfQogICAgICAgIGVsc2UgaWYoby50eXBlPT09InN0ZXAiKXsgaWYoby50b3RhbCl7IGNvbnN0IHA9by5zdGVwL28udG90YWwqMTAwOwogICAgICAg"
    "ICAgYmFyLnN0eWxlLndpZHRoPXArIiUiOyBwYy50ZXh0Q29udGVudD1vLnN0ZXArIi8iK28udG90YWw7IHN0YXQudGV4dENvbnRlbnQ9IkdlbmVyYXRp"
    "bmciOyB9IH0KICAgICAgICBlbHNlIGlmKG8udHlwZT09PSJkb25lIil7IGJhci5zdHlsZS53aWR0aD0iMTAwJSI7IHN0YXQudGV4dENvbnRlbnQ9IkRv"
    "bmUiOwogICAgICAgICAgY29uc3QgaW09by5pbWFnZTsgJCgiaW1nUHJldmlldyIpLmlubmVySFRNTCA9CiAgICAgICAgICAgIGA8aW1nIHNyYz0iL2Fw"
    "aS9pbWFnZXMvZmlsZS8ke2ltLmZpbGVuYW1lfT90PSR7RGF0ZS5ub3coKX0iIHRpdGxlPSJDbGljayB0byB2aWV3IGxhcmdlciI+YDsKICAgICAgICAg"
    "ICQoImltZ1ByZXZpZXciKS5xdWVyeVNlbGVjdG9yKCJpbWciKS5vbmNsaWNrID0gKCk9PnZpZXdJbWFnZShpbSk7CiAgICAgICAgICB0b2FzdCgiSW1h"
    "Z2Ugc2F2ZWQgdG8gZ2FsbGVyeSIsIm9rIik7IGxvYWRHYWxsZXJ5KCk7CiAgICAgICAgICBzZXRUaW1lb3V0KCgpPT57cHJvZy5pbm5lckhUTUw9IiI7"
    "fSwxNTAwKTsgfQogICAgICAgIGVsc2UgaWYoby50eXBlPT09ImNhbmNlbGxlZCIpeyBzdGF0LnRleHRDb250ZW50PSJTdG9wcGVkIjsKICAgICAgICAg"
    "ICQoImltZ1ByZXZpZXciKS5pbm5lckhUTUw9JzxkaXYgY2xhc3M9InBoIj5TdG9wcGVkIOKAlCBub3RoaW5nIHdhcyBzYXZlZDwvZGl2Pic7CiAgICAg"
    "ICAgICB0b2FzdCgiR2VuZXJhdGlvbiBzdG9wcGVkIiwib2siKTsKICAgICAgICAgIHNldFRpbWVvdXQoKCk9Pntwcm9nLmlubmVySFRNTD0iIjt9LDEy"
    "MDApOyB9CiAgICAgICAgZWxzZSBpZihvLnR5cGU9PT0iZXJyb3IiKXsgdGhyb3cgbmV3IEVycm9yKG8uZXJyb3IpOyB9CiAgICAgIH0pOwogIH1jYXRj"
    "aChlKXsKICAgIHRvYXN0KCJHZW5lcmF0aW9uIGZhaWxlZDogIitlLm1lc3NhZ2UsImVyciIpOwogICAgc3RhdC50ZXh0Q29udGVudD0iRmFpbGVkOiAi"
    "K2UubWVzc2FnZTsKICAgICQoImltZ1ByZXZpZXciKS5pbm5lckhUTUw9JzxkaXYgY2xhc3M9InBoIj5HZW5lcmF0aW9uIGZhaWxlZDwvZGl2Pic7CiAg"
    "fWZpbmFsbHl7IGJ0bi5kaXNhYmxlZD1mYWxzZTsgfQp9CmFzeW5jIGZ1bmN0aW9uIGxvYWRHYWxsZXJ5KCl7CiAgdHJ5ewogICAgY29uc3QgciA9IGF3"
    "YWl0IGFwaSgiL2FwaS9pbWFnZXMiKTsKICAgIGNvbnN0IGcgPSAkKCJnYWxsZXJ5Iik7IGlmKCFnKSByZXR1cm47CiAgICBpZighci5pbWFnZXMubGVu"
    "Z3RoKXsgZy5pbm5lckhUTUwgPQogICAgICAnPGRpdiBjbGFzcz0iZW1wdHkiIHN0eWxlPSJncmlkLWNvbHVtbjoxLy0xIj48ZGl2IGNsYXNzPSJiaWci"
    "PvCflrw8L2Rpdj5ObyBpbWFnZXMgeWV0PC9kaXY+JzsgcmV0dXJuOyB9CiAgICBnLmlubmVySFRNTCA9ICIiOwogICAgci5pbWFnZXMuZm9yRWFjaChp"
    "bT0+ewogICAgICBjb25zdCBpdCA9IGVsKCJkaXYiLCJnaXRlbSIpOwogICAgICBpdC5pbm5lckhUTUwgPSBgPGltZyBzcmM9Ii9hcGkvaW1hZ2VzL2Zp"
    "bGUvJHtpbS5maWxlbmFtZX0iIGxvYWRpbmc9ImxhenkiPgogICAgICAgIDxkaXYgY2xhc3M9Im92Ij48ZGl2IGNsYXNzPSJwciI+JHtlc2MoaW0ucHJv"
    "bXB0fHwiIil9PC9kaXY+PC9kaXY+CiAgICAgICAgPGRpdiBjbGFzcz0icm0iIHRpdGxlPSJSZW1vdmUiPgogICAgICAgIDxzdmcgd2lkdGg9IjE0IiBo"
    "ZWlnaHQ9IjE0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIGZpbGw9Im5vbmUiIHN0cm9rZT0iY3VycmVudENvbG9yIiBzdHJva2Utd2lkdGg9IjIiPjxwYXRo"
    "IGQ9Ik0xOCA2IDYgMThNNiA2bDEyIDEyIi8+PC9zdmc+PC9kaXY+YDsKICAgICAgaXQub25jbGljayA9ICgpPT52aWV3SW1hZ2UoaW0pOwogICAgICBp"
    "dC5xdWVyeVNlbGVjdG9yKCIucm0iKS5vbmNsaWNrID0gYXN5bmMoZSk9PnsgZS5zdG9wUHJvcGFnYXRpb24oKTsKICAgICAgICBhd2FpdCBkZWwoIi9h"
    "cGkvaW1hZ2VzLyIraW0uaWQpOyB0b2FzdCgiSW1hZ2UgcmVtb3ZlZCIsIm9rIik7IGxvYWRHYWxsZXJ5KCk7IH07CiAgICAgIGcuYXBwZW5kQ2hpbGQo"
    "aXQpOwogICAgfSk7CiAgfWNhdGNoKGUpe30KfQpmdW5jdGlvbiB2aWV3SW1hZ2UoaW0pewogIGNvbnN0IGJvZHkgPSBtb2RhbCh7dGl0bGU6IkltYWdl"
    "Iiwgd2lkZTp0cnVlLCBib2R5SFRNTDpgCiAgICA8aW1nIHNyYz0iL2FwaS9pbWFnZXMvZmlsZS8ke2ltLmZpbGVuYW1lfSIgaWQ9Im1vZGFsSW1nIiB0"
    "aXRsZT0iQ2xpY2sgdG8gb3BlbiB0aGUgb3JpZ2luYWwgZmlsZSIKICAgICAgc3R5bGU9IndpZHRoOjEwMCU7bWF4LWhlaWdodDo3MHZoO29iamVjdC1m"
    "aXQ6Y29udGFpbjtib3JkZXItcmFkaXVzOjEwcHg7Ym9yZGVyOjFweCBzb2xpZCB2YXIoLS1saW5lKTtjdXJzb3I6em9vbS1pbjtiYWNrZ3JvdW5kOnZh"
    "cigtLWluaykiPgogICAgPGRpdiBjbGFzcz0iZmllbGQiIHN0eWxlPSJtYXJnaW4tdG9wOjE0cHgiPjxsYWJlbD5Qcm9tcHQ8L2xhYmVsPgogICAgICA8"
    "ZGl2IGNsYXNzPSJtb25vIiBzdHlsZT0iZm9udC1zaXplOjEycHg7YmFja2dyb3VuZDp2YXIoLS1wYW5lbC0yKTtwYWRkaW5nOjEwcHg7Ym9yZGVyLXJh"
    "ZGl1czo4cHg7Ym9yZGVyOjFweCBzb2xpZCB2YXIoLS1saW5lKSI+JHtlc2MoaW0ucHJvbXB0fHwiIil9PC9kaXY+PC9kaXY+CiAgICA8ZGl2IGNsYXNz"
    "PSJyb3ciIHN0eWxlPSJnYXA6MTZweDtmb250LXNpemU6MTJweCI+CiAgICAgIDxzcGFuIGNsYXNzPSJkaW0gbW9ubyI+JHtpbS53aWR0aH3DlyR7aW0u"
    "aGVpZ2h0fTwvc3Bhbj4KICAgICAgPHNwYW4gY2xhc3M9ImRpbSBtb25vIj5zZWVkICR7aW0uc2VlZH08L3NwYW4+CiAgICAgIDxzcGFuIGNsYXNzPSJk"
    "aW0gbW9ubyI+JHtlc2MoaW0ubW9kZWx8fCIiKX08L3NwYW4+PC9kaXY+YCwKICAgIGFjdGlvbnM6W3tsYWJlbDoiT3BlbiBmaWxlIiwgb25DbGljazoo"
    "KT0+d2luZG93Lm9wZW4oIi9hcGkvaW1hZ2VzL2ZpbGUvIitpbS5maWxlbmFtZSl9LAogICAgICB7bGFiZWw6IlJlbW92ZSIsIGNsczoiZGFuZ2VyIiwg"
    "b25DbGljazphc3luYygpPT57IGNsb3NlTW9kYWwoKTsKICAgICAgICBhd2FpdCBkZWwoIi9hcGkvaW1hZ2VzLyIraW0uaWQpOyB0b2FzdCgiUmVtb3Zl"
    "ZCIsIm9rIik7IGxvYWRHYWxsZXJ5KCk7IH19XX0pOwogIGJvZHkucXVlcnlTZWxlY3RvcigiI21vZGFsSW1nIikub25jbGljayA9ICgpPT53aW5kb3cu"
    "b3BlbigiL2FwaS9pbWFnZXMvZmlsZS8iK2ltLmZpbGVuYW1lKTsKfQpsb2FkZXJzLmltYWdlcyA9IGxvYWRJbWFnZXM7CgovKiA9PT09PT09PT09PT09"
    "PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT0KICAgS05PV0xFREdFIChSQUcpCiAgID09PT09PT09PT09PT09PT09"
    "PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PSAqLwphc3luYyBmdW5jdGlvbiBsb2FkS25vd2xlZGdlKCl7CiAgY29uc3Qg"
    "aG9zdCA9ICQoImtub3dsZWRnZUJvZHkiKTsKICBob3N0LmlubmVySFRNTCA9ICc8ZGl2IGNsYXNzPSJyb3ciPjxkaXYgY2xhc3M9InNwaW4iPjwvZGl2"
    "PjxzcGFuIGNsYXNzPSJtdXRlZCI+TG9hZGluZ+KApjwvc3Bhbj48L2Rpdj4nOwogIGxldCByOwogIHRyeXsgciA9IGF3YWl0IGFwaSgiL2FwaS9yYWcv"
    "ZG9jcyIpOyB9Y2F0Y2goZSl7IGhvc3QuaW5uZXJIVE1MPWVyckNhcmQoZS5tZXNzYWdlKTsgcmV0dXJuOyB9CiAgbGV0IHdhcm4gPSAiIjsKICBpZigh"
    "ci5lbWJlZF9yZWFkeSl7CiAgICB3YXJuID0gYDxkaXYgY2xhc3M9ImNhcmQiIHN0eWxlPSJib3JkZXItY29sb3I6dmFyKC0tc2lnbmFsLWRpbSk7bWFy"
    "Z2luLWJvdHRvbToxNnB4Ij4KICAgICAgPGRpdiBjbGFzcz0icm93IiBzdHlsZT0iZ2FwOjlweDttYXJnaW4tYm90dG9tOjZweCI+PHNwYW4gY2xhc3M9"
    "ImRvdCB3YXJuIj48L3NwYW4+CiAgICAgIDxiPkVtYmVkZGluZyBtb2RlbCBuZWVkZWQ8L2I+PC9kaXY+CiAgICAgIDxwIGNsYXNzPSJtdXRlZCIgc3R5"
    "bGU9Im1hcmdpbjowIj5UbyBzZWFyY2ggZG9jdW1lbnRzLCBIZW9ydGggbmVlZHMgdGhlIGVtYmVkZGluZwogICAgICAgIG1vZGVsIDxzcGFuIGNsYXNz"
    "PSJtb25vIj4ke2VzYyhyLmVtYmVkX21vZGVsKX08L3NwYW4+LgogICAgICAgICR7ci5vbGxhbWFfdXA/IiI6IlN0YXJ0IE9sbGFtYSwgdGhlbiAifWRv"
    "d25sb2FkIGl0IGZyb20gdGhlIE1vZGVscyBwYWdlCiAgICAgICAgKHNlYXJjaCDigJxlbWJlZOKAnSkuIFlvdSBjYW4gc3RpbGwgdXBsb2FkIG5vdywg"
    "YnV0IHNlYXJjaCBuZWVkcyBpdC48L3A+PC9kaXY+YDsKICB9CiAgaG9zdC5pbm5lckhUTUwgPSB3YXJuICsgYAogICAgPGRpdiBjbGFzcz0iZHJvcHpv"
    "bmUiIGlkPSJkcm9wem9uZSI+CiAgICAgIDxzdmcgd2lkdGg9IjMwIiBoZWlnaHQ9IjMwIiB2aWV3Qm94PSIwIDAgMjQgMjQiIGZpbGw9Im5vbmUiIHN0"
    "cm9rZT0iY3VycmVudENvbG9yIiBzdHJva2Utd2lkdGg9IjEuNiIgc3R5bGU9Im1hcmdpbi1ib3R0b206OHB4Ij48cGF0aCBkPSJNMTIgMTZWNG0wIDAg"
    "NCA0bS00LTQtNCA0Ii8+PHBhdGggZD0iTTQgMTZ2MmEyIDIgMCAwIDAgMiAyaDEyYTIgMiAwIDAgMCAyLTJ2LTIiLz48L3N2Zz4KICAgICAgPGRpdiBz"
    "dHlsZT0iZm9udC13ZWlnaHQ6NjAwIj5Ecm9wIGZpbGVzIGhlcmUgb3IgY2xpY2sgdG8gYnJvd3NlPC9kaXY+CiAgICAgIDxkaXYgY2xhc3M9ImhpbnQi"
    "PlBERiwgVFhULCBNYXJrZG93biwgY29kZSDigJQgdXAgdG8gNTAgTUIgZWFjaDwvZGl2PgogICAgICA8aW5wdXQgdHlwZT0iZmlsZSIgaWQ9ImZpbGVJ"
    "bnB1dCIgbXVsdGlwbGUgaGlkZGVuCiAgICAgICAgYWNjZXB0PSIucGRmLC50eHQsLm1kLC5tYXJrZG93biwucHksLmpzLC50cywuanNvbiwuY3N2LC5o"
    "dG1sLC5jc3MsLmphdmEsLmMsLmNwcCwuZ28sLnJzLC5yYiwuc2gsLnlhbWwsLnltbCwueG1sIj4KICAgIDwvZGl2PgogICAgPGRpdiBpZD0idXBsb2Fk"
    "UHJvZyI+PC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJzZWN0aW9uLXRpdGxlIj5Zb3VyIGRvY3VtZW50cyR7ci5kb2NzLmxlbmd0aD8iIMK3ICIrci5kb2Nz"
    "Lmxlbmd0aDoiIn08L2Rpdj4KICAgIDxkaXYgaWQ9ImRvY0xpc3QiPjwvZGl2PmA7CiAgY29uc3QgZHogPSAkKCJkcm9wem9uZSIpLCBmaSA9ICQoImZp"
    "bGVJbnB1dCIpOwogIGR6Lm9uY2xpY2sgPSAoKT0+ZmkuY2xpY2soKTsKICBmaS5vbmNoYW5nZSA9ICgpPT57IGhhbmRsZUZpbGVzKFsuLi5maS5maWxl"
    "c10pOyBmaS52YWx1ZT0iIjsgfTsKICBkei5vbmRyYWdvdmVyID0gKGUpPT57IGUucHJldmVudERlZmF1bHQoKTsgZHouY2xhc3NMaXN0LmFkZCgiZHJh"
    "ZyIpOyB9OwogIGR6Lm9uZHJhZ2xlYXZlID0gKCk9PmR6LmNsYXNzTGlzdC5yZW1vdmUoImRyYWciKTsKICBkei5vbmRyb3AgPSAoZSk9PnsgZS5wcmV2"
    "ZW50RGVmYXVsdCgpOyBkei5jbGFzc0xpc3QucmVtb3ZlKCJkcmFnIik7CiAgICBoYW5kbGVGaWxlcyhbLi4uZS5kYXRhVHJhbnNmZXIuZmlsZXNdKTsg"
    "fTsKICByZW5kZXJEb2NzKHIuZG9jcyk7Cn0KZnVuY3Rpb24gcmVuZGVyRG9jcyhkb2NzKXsKICBjb25zdCBsaXN0ID0gJCgiZG9jTGlzdCIpOyBpZigh"
    "bGlzdCkgcmV0dXJuOwogIGlmKCFkb2NzLmxlbmd0aCl7IGxpc3QuaW5uZXJIVE1MID0KICAgICc8ZGl2IGNsYXNzPSJlbXB0eSI+PGRpdiBjbGFzcz0i"
    "YmlnIj7wn5OEPC9kaXY+Tm8gZG9jdW1lbnRzIHlldDwvZGl2Pic7IHJldHVybjsgfQogIGxpc3QuaW5uZXJIVE1MID0gIiI7CiAgZG9jcy5mb3JFYWNo"
    "KGQ9PnsKICAgIGNvbnN0IHJvdyA9IGVsKCJkaXYiLCJkb2Nyb3ciKTsKICAgIHJvdy5pbm5lckhUTUwgPSBgPGRpdiBjbGFzcz0iZGljIj4KICAgICAg"
    "PHN2ZyB3aWR0aD0iMTgiIGhlaWdodD0iMTgiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJjdXJyZW50Q29sb3IiIHN0cm9r"
    "ZS13aWR0aD0iMS44Ij48cGF0aCBkPSJNMTQgM3Y1aDUiLz48cGF0aCBkPSJNMTQgM0g2YTIgMiAwIDAgMC0yIDJ2MTRhMiAyIDAgMCAwIDIgMmgxMmEy"
    "IDIgMCAwIDAgMi0yVjhsLTYtNVoiLz48L3N2Zz48L2Rpdj4KICAgICAgPGRpdiBjbGFzcz0iZGkiPjxkaXYgY2xhc3M9Im5tIj4ke2VzYyhkLm5hbWUp"
    "fTwvZGl2PgogICAgICAgIDxkaXYgY2xhc3M9Im10Ij4ke2QuY2h1bmtzfSBjaHVua3MgwrcgYWRkZWQgJHtuZXcgRGF0ZShkLmNyZWF0ZWQqMTAwMCku"
    "dG9Mb2NhbGVEYXRlU3RyaW5nKCl9PC9kaXY+PC9kaXY+YDsKICAgIGNvbnN0IGIgPSBlbCgiYnV0dG9uIiwiYnRuIGRhbmdlciBzbSIsIlJlbW92ZSIp"
    "OwogICAgYi5vbmNsaWNrID0gYXN5bmMoKT0+eyBhd2FpdCBkZWwoIi9hcGkvcmFnL2RvY3MvIitkLmlkKTsgdG9hc3QoIlJlbW92ZWQgIitkLm5hbWUs"
    "Im9rIik7IGxvYWRLbm93bGVkZ2UoKTsgfTsKICAgIHJvdy5hcHBlbmRDaGlsZChiKTsgbGlzdC5hcHBlbmRDaGlsZChyb3cpOwogIH0pOwp9CmFzeW5j"
    "IGZ1bmN0aW9uIGhhbmRsZUZpbGVzKGZpbGVzKXsKICBjb25zdCBwcm9nID0gJCgidXBsb2FkUHJvZyIpOwogIGZvcihjb25zdCBmIG9mIGZpbGVzKXsK"
    "ICAgIGNvbnN0IHJvdyA9IGVsKCJkaXYiLCJkb2Nyb3ciKTsKICAgIHJvdy5pbm5lckhUTUwgPSBgPGRpdiBjbGFzcz0ic3BpbiI+PC9kaXY+PGRpdiBj"
    "bGFzcz0iZGkiPjxkaXYgY2xhc3M9Im5tIj4ke2VzYyhmLm5hbWUpfTwvZGl2PgogICAgICA8ZGl2IGNsYXNzPSJtdCI+UmVhZGluZyAmIGVtYmVkZGlu"
    "Z+KApjwvZGl2PjwvZGl2PmA7CiAgICBwcm9nLmFwcGVuZENoaWxkKHJvdyk7CiAgICB0cnl7CiAgICAgIGNvbnN0IGZkID0gbmV3IEZvcm1EYXRhKCk7"
    "IGZkLmFwcGVuZCgiZmlsZSIsIGYpOwogICAgICBjb25zdCByID0gYXdhaXQgZmV0Y2goIi9hcGkvcmFnL3VwbG9hZCIsIHttZXRob2Q6IlBPU1QiLCBi"
    "b2R5OmZkfSk7CiAgICAgIGNvbnN0IGogPSBhd2FpdCByLmpzb24oKTsKICAgICAgaWYoai5vayl7IHRvYXN0KCJBZGRlZCAiK2YubmFtZSwib2siKTsg"
    "fQogICAgICBlbHNlIHsgdG9hc3Qoai5lcnJvcnx8IlVwbG9hZCBmYWlsZWQiLCJlcnIiKTsgfQogICAgfWNhdGNoKGUpeyB0b2FzdCgiVXBsb2FkIGZh"
    "aWxlZDogIitlLm1lc3NhZ2UsImVyciIpOyB9CiAgICByb3cucmVtb3ZlKCk7CiAgfQogIGxvYWRLbm93bGVkZ2UoKTsKfQpsb2FkZXJzLmtub3dsZWRn"
    "ZSA9IGxvYWRLbm93bGVkZ2U7CgovKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT0KICAg"
    "QUdFTlQgJiBUT09MUwogICA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT0gKi8KY29uc3Qg"
    "VE9PTF9JQ09OUyA9IHsKICB3ZWJfc2VhcmNoOifwn5SNJywgZmV0Y2hfdXJsOifwn4yQJywgY2FsY3VsYXRvcjon8J+nricsIHNlYXJjaF9rbm93bGVk"
    "Z2U6J/Cfk5onLAogIGxpc3RfZmlsZXM6J/Cfk4EnLCByZWFkX2ZpbGU6J/Cfk4QnLCB3cml0ZV9maWxlOifinI/vuI8nLCBnZW5lcmF0ZV9pbWFnZTon"
    "8J+OqCcsCiAgcnVuX3B5dGhvbjon8J+QjScsIHJ1bl9zaGVsbDon4oyYJyB9Owphc3luYyBmdW5jdGlvbiBsb2FkVG9vbHMoKXsKICBjb25zdCBob3N0"
    "ID0gJCgidG9vbHNCb2R5Iik7CiAgaG9zdC5pbm5lckhUTUwgPSAnPGRpdiBjbGFzcz0icm93Ij48ZGl2IGNsYXNzPSJzcGluIj48L2Rpdj48c3BhbiBj"
    "bGFzcz0ibXV0ZWQiPkxvYWRpbmfigKY8L3NwYW4+PC9kaXY+JzsKICBsZXQgdDsKICB0cnl7IHQgPSBhd2FpdCBhcGkoIi9hcGkvdG9vbHMiKTsgfWNh"
    "dGNoKGUpeyBob3N0LmlubmVySFRNTD1lcnJDYXJkKGUubWVzc2FnZSk7IHJldHVybjsgfQogIGxldCBodG1sID0gYDxkaXYgY2xhc3M9ImNhcmQgcGFk"
    "LWxnIiBzdHlsZT0ibWFyZ2luLWJvdHRvbToxOHB4Ij4KICAgIDxkaXYgY2xhc3M9InNwcmVhZCI+PGRpdj48Yj5Ib3cgdGhlIGFnZW50IHdvcmtzPC9i"
    "PgogICAgICA8cCBjbGFzcz0ibXV0ZWQiIHN0eWxlPSJtYXJnaW46NnB4IDAgMDttYXgtd2lkdGg6NjRjaCI+VHVybiBvbiA8Yj5BZ2VudDwvYj4gaW4g"
    "Q2hhdCBhbmQgdGhlCiAgICAgIG1vZGVsIGNhbiBjYWxsIHRoZXNlIHRvb2xzIHRvIHNlYXJjaCwgY2FsY3VsYXRlLCByZWFkIHlvdXIgZmlsZXMgYW5k"
    "IG1vcmUg4oCUIHRoZW4KICAgICAgYW5zd2VyIHVzaW5nIHdoYXQgaXQgZm91bmQuIFNvbWUgdG9vbHMgcmVzcGVjdCB5b3VyIHNhZmV0eSBzZXR0aW5n"
    "cyBiZWxvdy48L3A+PC9kaXY+PC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJyb3ciIHN0eWxlPSJnYXA6MjBweDttYXJnaW4tdG9wOjE0cHgiPgogICAgICA8"
    "c3BhbiBjbGFzcz0iY2hpcCAke3QuYWxsb3dfd2ViX3Rvb2xzPyJncm4iOiIifSI+d2ViIHRvb2xzICR7dC5hbGxvd193ZWJfdG9vbHM/Im9uIjoib2Zm"
    "In08L3NwYW4+CiAgICAgIDxzcGFuIGNsYXNzPSJjaGlwICR7dC5hbGxvd19jb2RlX2V4ZWN1dGlvbj8iZ3JuIjoiIn0iPmNvZGUgZXhlY3V0aW9uICR7"
    "dC5hbGxvd19jb2RlX2V4ZWN1dGlvbj8ib24iOiJvZmYifTwvc3Bhbj4KICAgIDwvZGl2PjwvZGl2PgogICAgPGRpdiBjbGFzcz0ic2VjdGlvbi10aXRs"
    "ZSI+QnVpbHQtaW4gdG9vbHM8L2Rpdj5gOwogIHQuYnVpbHRpbi5mb3JFYWNoKHRvb2w9PnsKICAgIGNvbnN0IGdhdGVkID0gKHRvb2wubmFtZT09PSJy"
    "dW5fcHl0aG9uInx8dG9vbC5uYW1lPT09InJ1bl9zaGVsbCIpOwogICAgaHRtbCArPSBgPGRpdiBjbGFzcz0idG9vbHJvdyI+PGRpdiBjbGFzcz0idGlj"
    "Ij4ke1RPT0xfSUNPTlNbdG9vbC5uYW1lXXx8IuKamSJ9PC9kaXY+CiAgICAgIDxkaXYgY2xhc3M9InRpIj48ZGl2IGNsYXNzPSJubSI+JHtlc2ModG9v"
    "bC5uYW1lKX08L2Rpdj4KICAgICAgICA8ZGl2IGNsYXNzPSJkcyI+JHtlc2ModG9vbC5kZXNjKX08L2Rpdj48L2Rpdj4KICAgICAgJHtnYXRlZD9gPHNw"
    "YW4gY2xhc3M9ImNoaXAgJHt0LmFsbG93X2NvZGVfZXhlY3V0aW9uPyJncm4iOiIifSI+JHt0LmFsbG93X2NvZGVfZXhlY3V0aW9uPyJlbmFibGVkIjoi"
    "bmVlZHMgb3B0LWluIn08L3NwYW4+YDoKICAgICAgICAodG9vbC5uYW1lPT09IndlYl9zZWFyY2gifHx0b29sLm5hbWU9PT0iZmV0Y2hfdXJsIik/CiAg"
    "ICAgICAgYDxzcGFuIGNsYXNzPSJjaGlwICR7dC5hbGxvd193ZWJfdG9vbHM/ImdybiI6IiJ9Ij4ke3QuYWxsb3dfd2ViX3Rvb2xzPyJvbiI6Im9mZiJ9"
    "PC9zcGFuPmA6CiAgICAgICAgJzxzcGFuIGNsYXNzPSJjaGlwIGdybiI+cmVhZHk8L3NwYW4+J308L2Rpdj5gOwogIH0pOwogIGh0bWwgKz0gYDxkaXYg"
    "Y2xhc3M9InNlY3Rpb24tdGl0bGUiPldlYiBzZWFyY2ggPHNwYW4gY2xhc3M9ImNoaXAiPm9wdGlvbmFsIHVwZ3JhZGU8L3NwYW4+PC9kaXY+CiAgICA8"
    "ZGl2IGlkPSJzZWFyY2hCb2R5Ij48L2Rpdj4KICAgIDxkaXYgY2xhc3M9InNlY3Rpb24tdGl0bGUiPk1DUCBzZXJ2ZXJzIDxzcGFuIGNsYXNzPSJjaGlw"
    "IHZpbyI+TW9kZWwgQ29udGV4dCBQcm90b2NvbDwvc3Bhbj48L2Rpdj4KICAgIDxkaXYgaWQ9Im1jcEJvZHkiPjwvZGl2PmA7CiAgaG9zdC5pbm5lckhU"
    "TUwgPSBodG1sOwogIHJlbmRlck1DUCh0KTsKICByZW5kZXJTZWFyY2goKTsKfQpmdW5jdGlvbiByZW5kZXJNQ1AodCl7CiAgY29uc3QgaG9zdCA9ICQo"
    "Im1jcEJvZHkiKTsKICBpZighdC5tY3BfaW5zdGFsbGVkKXsKICAgIGhvc3QuaW5uZXJIVE1MID0gYDxkaXYgY2xhc3M9ImNhcmQgc2V0dXAtY2FyZCI+"
    "CiAgICAgIDxkaXYgY2xhc3M9ImljIj7wn5SMPC9kaXY+CiAgICAgIDxoMyBzdHlsZT0ibWFyZ2luLWJvdHRvbTo4cHgiPkVuYWJsZSBNQ1AgY29ubmVj"
    "dGlvbnM8L2gzPgogICAgICA8cCBjbGFzcz0ibXV0ZWQiIHN0eWxlPSJtYXgtd2lkdGg6NTRjaDttYXJnaW46MCBhdXRvIDZweCI+TUNQIGxldHMgdGhl"
    "IGFnZW50IHVzZSBleHRlcm5hbAogICAgICAgIHRvb2wgc2VydmVycyDigJQgZmlsZSBzeXN0ZW1zLCBicm93c2VycywgZGF0YWJhc2VzLCBBUElzIGFu"
    "ZCBtb3JlLiBUaGlzIGluc3RhbGxzIHRoZQogICAgICAgIDxzcGFuIGNsYXNzPSJtb25vIj5tY3A8L3NwYW4+IHBhY2thZ2UgaW50byBIZW9ydGgncyBl"
    "bnZpcm9ubWVudCAoc21hbGwsIG9uZS10aW1lKS48L3A+CiAgICAgIDxkaXYgc3R5bGU9Im1hcmdpbi10b3A6MTRweCI+PGJ1dHRvbiBjbGFzcz0iYnRu"
    "IHByaW1hcnkiIGlkPSJzZXR1cE1jcEJ0biI+SW5zdGFsbCBNQ1Agc3VwcG9ydDwvYnV0dG9uPjwvZGl2PgogICAgICA8ZGl2IGNsYXNzPSJpbnN0YWxs"
    "bG9nIiBpZD0ibWNwTG9nIiBoaWRkZW4+PC9kaXY+PC9kaXY+YDsKICAgICQoInNldHVwTWNwQnRuIikub25jbGljayA9IGFzeW5jKCk9PnsKICAgICAg"
    "JCgic2V0dXBNY3BCdG4iKS5kaXNhYmxlZD10cnVlOyAkKCJzZXR1cE1jcEJ0biIpLnRleHRDb250ZW50PSJJbnN0YWxsaW5n4oCmIjsKICAgICAgJCgi"
    "bWNwTG9nIikuaGlkZGVuPWZhbHNlOwogICAgICB0cnl7IGF3YWl0IHBvc3QoIi9hcGkvbWNwL3NldHVwIik7IHBvbGxNY3BJbnN0YWxsKCk7IH1jYXRj"
    "aChlKXsgdG9hc3QoZS5tZXNzYWdlLCJlcnIiKTsgfQogICAgfTsKICAgIHJldHVybjsKICB9CiAgbGV0IGh0bWwgPSBgPHAgY2xhc3M9Im11dGVkIiBz"
    "dHlsZT0ibWFyZ2luOjAgMCAxNHB4Ij5Db25uZWN0ZWQgc2VydmVycyBleHBvc2UgdGhlaXIgdG9vbHMgdG8gdGhlCiAgICBhZ2VudCBhdXRvbWF0aWNh"
    "bGx5LiA8YnV0dG9uIGNsYXNzPSJidG4gc20gZ2hvc3QiIGlkPSJhZGRNY3BCdG4iPisgQWRkIHNlcnZlcjwvYnV0dG9uPjwvcD5gOwogIGlmKCF0Lm1j"
    "cF9zZXJ2ZXJzLmxlbmd0aCl7CiAgICBodG1sICs9ICc8ZGl2IGNsYXNzPSJlbXB0eSI+PGRpdiBjbGFzcz0iYmlnIj7wn5SMPC9kaXY+Tm8gTUNQIHNl"
    "cnZlcnMgeWV0Ljxicj4nKwogICAgICAnPHNwYW4gY2xhc3M9ImhpbnQiPkFkZCBvbmUgdG8gZ2l2ZSB0aGUgYWdlbnQgbmV3IGFiaWxpdGllcy48L3Nw"
    "YW4+PC9kaXY+JzsKICB9CiAgaHRtbCArPSAnPGRpdiBpZD0ic3J2TGlzdCI+PC9kaXY+JzsKICBob3N0LmlubmVySFRNTCA9IGh0bWw7CiAgJCgiYWRk"
    "TWNwQnRuIikub25jbGljayA9IHNob3dBZGRNY3A7CiAgY29uc3QgbGlzdCA9ICQoInNydkxpc3QiKTsKICB0Lm1jcF9zZXJ2ZXJzLmZvckVhY2gocz0+"
    "ewogICAgY29uc3Qgc3J2ID0gZWwoImRpdiIsInNydiIpOwogICAgc3J2LmlubmVySFRNTCA9IGA8ZGl2IGNsYXNzPSJzaCI+PHNwYW4gY2xhc3M9ImRv"
    "dCAke3MuY29ubmVjdGVkPyJvbiI6KHMuZXJyb3I/Im9mZiI6IiIpfSI+PC9zcGFuPgogICAgICA8ZGl2IHN0eWxlPSJmbGV4OjEiPjxkaXYgY2xhc3M9"
    "Im5tIj4ke2VzYyhzLm5hbWUpfTwvZGl2PgogICAgICAgIDxkaXYgY2xhc3M9ImNtZCI+JHtlc2Mocy5jb21tYW5kKX0gJHtlc2MoKHMuYXJnc3x8W10p"
    "LmpvaW4oIiAiKSl9PC9kaXY+PC9kaXY+CiAgICAgIDxkaXYgY2xhc3M9InJvdyIgc3R5bGU9ImdhcDo2cHgiPjwvZGl2PjwvZGl2PgogICAgICAke3Mu"
    "Y29ubmVjdGVkJiZzLnRvb2xzLmxlbmd0aD9gPGRpdiBjbGFzcz0idG9vbGNoaXBzIj4kewogICAgICAgIHMudG9vbHMubWFwKHRuPT5gPHNwYW4gY2xh"
    "c3M9ImNoaXAgdmlvIj4ke2VzYyh0bil9PC9zcGFuPmApLmpvaW4oIiIpfTwvZGl2PmA6IiJ9CiAgICAgICR7cy5lcnJvcj9gPGRpdiBjbGFzcz0iaGlu"
    "dCIgc3R5bGU9ImNvbG9yOnZhcigtLXJlZCk7bWFyZ2luLXRvcDo4cHgiPiR7ZXNjKHMuZXJyb3IpfTwvZGl2PmA6IiJ9YDsKICAgIGNvbnN0IGFjdGlv"
    "bnMgPSBzcnYucXVlcnlTZWxlY3RvcigiLnNoIC5yb3ciKTsKICAgIGNvbnN0IHJjID0gZWwoImJ1dHRvbiIsImJ0biBzbSBnaG9zdCIsIHMuY29ubmVj"
    "dGVkPyJSZWNvbm5lY3QiOiJDb25uZWN0Iik7CiAgICByYy5vbmNsaWNrID0gYXN5bmMoKT0+eyByYy5kaXNhYmxlZD10cnVlOyByYy50ZXh0Q29udGVu"
    "dD0i4oCmIjsKICAgICAgdHJ5eyBjb25zdCByID0gYXdhaXQgcG9zdCgiL2FwaS9tY3Avc2VydmVycy8iK3MuaWQrIi9jb25uZWN0Iik7CiAgICAgICAg"
    "aWYoci5vaykgdG9hc3QoIkNvbm5lY3RlZCAiK3MubmFtZSwib2siKTsgZWxzZSB0b2FzdChyLmVycm9yfHwiRmFpbGVkIiwiZXJyIik7IH0KICAgICAg"
    "Y2F0Y2goZSl7IHRvYXN0KGUubWVzc2FnZSwiZXJyIik7IH0gbG9hZFRvb2xzKCk7IH07CiAgICBjb25zdCBkbCA9IGVsKCJidXR0b24iLCJidG4gc20g"
    "ZGFuZ2VyIiwiUmVtb3ZlIik7CiAgICBkbC5vbmNsaWNrID0gYXN5bmMoKT0+eyBhd2FpdCBkZWwoIi9hcGkvbWNwL3NlcnZlcnMvIitzLmlkKTsgdG9h"
    "c3QoIlJlbW92ZWQiLCJvayIpOyBsb2FkVG9vbHMoKTsgfTsKICAgIGFjdGlvbnMuYXBwZW5kKHJjLCBkbCk7IGxpc3QuYXBwZW5kQ2hpbGQoc3J2KTsK"
    "ICB9KTsKfQpmdW5jdGlvbiBzaG93QWRkTWNwKCl7CiAgY29uc3QgYm9keSA9IG1vZGFsKHt0aXRsZToiQWRkIE1DUCBzZXJ2ZXIiLCBib2R5SFRNTDpg"
    "CiAgICA8cCBjbGFzcz0ibXV0ZWQiIHN0eWxlPSJtYXJnaW46MCAwIDE2cHgiPk1DUCBzZXJ2ZXJzIHJ1biBhcyBhIGxvY2FsIGNvbW1hbmQuIEZvciBl"
    "eGFtcGxlLCBhCiAgICAgIGZpbGVzeXN0ZW0gc2VydmVyOiBjb21tYW5kIDxzcGFuIGNsYXNzPSJtb25vIj5ucHg8L3NwYW4+LCBhcmd1bWVudHMKICAg"
    "ICAgPHNwYW4gY2xhc3M9Im1vbm8iPi15IEBtb2RlbGNvbnRleHRwcm90b2NvbC9zZXJ2ZXItZmlsZXN5c3RlbSAvcGF0aDwvc3Bhbj4uPC9wPgogICAg"
    "PGRpdiBjbGFzcz0iZmllbGQiPjxsYWJlbD5OYW1lPC9sYWJlbD48aW5wdXQgY2xhc3M9ImlucCIgaWQ9Im1jcE5hbWUiIHBsYWNlaG9sZGVyPSJGaWxl"
    "c3lzdGVtIj48L2Rpdj4KICAgIDxkaXYgY2xhc3M9ImZpZWxkIj48bGFiZWw+Q29tbWFuZDwvbGFiZWw+PGlucHV0IGNsYXNzPSJpbnAgbW9ubyIgaWQ9"
    "Im1jcENtZCIgcGxhY2Vob2xkZXI9Im5weCI+PC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJmaWVsZCI+PGxhYmVsPkFyZ3VtZW50cyA8c3BhbiBjbGFzcz0i"
    "ZGltIj4oc3BhY2Utc2VwYXJhdGVkKTwvc3Bhbj48L2xhYmVsPgogICAgICA8aW5wdXQgY2xhc3M9ImlucCBtb25vIiBpZD0ibWNwQXJncyIgcGxhY2Vo"
    "b2xkZXI9Ii15IEBtb2RlbGNvbnRleHRwcm90b2NvbC9zZXJ2ZXItZmlsZXN5c3RlbSAvVXNlcnMvbWUvZG9jcyI+PC9kaXY+CiAgICA8ZGl2IGNsYXNz"
    "PSJmaWVsZCI+PGxhYmVsPkVudmlyb25tZW50IDxzcGFuIGNsYXNzPSJkaW0iPihLRVk9dmFsdWUsIG9uZSBwZXIgbGluZSwgb3B0aW9uYWwpPC9zcGFu"
    "PjwvbGFiZWw+CiAgICAgIDx0ZXh0YXJlYSBjbGFzcz0idGEiIGlkPSJtY3BFbnYiIHBsYWNlaG9sZGVyPSJBUElfS0VZPS4uLiI+PC90ZXh0YXJlYT48"
    "L2Rpdj5gLAogICAgYWN0aW9uczpbe2xhYmVsOiJDYW5jZWwiLCBvbkNsaWNrOmNsb3NlTW9kYWx9LAogICAgICB7bGFiZWw6IkFkZCAmIGNvbm5lY3Qi"
    "LCBjbHM6InByaW1hcnkiLCBvbkNsaWNrOmFzeW5jKCk9PnsKICAgICAgICBjb25zdCBuYW1lPSQoIm1jcE5hbWUiKS52YWx1ZS50cmltKCksIGNtZD0k"
    "KCJtY3BDbWQiKS52YWx1ZS50cmltKCk7CiAgICAgICAgaWYoIW5hbWV8fCFjbWQpeyB0b2FzdCgiTmFtZSBhbmQgY29tbWFuZCByZXF1aXJlZCIsImVy"
    "ciIpOyByZXR1cm47IH0KICAgICAgICBjb25zdCBhcmdzPSQoIm1jcEFyZ3MiKS52YWx1ZS50cmltKCk7CiAgICAgICAgY29uc3QgZW52PXt9OyAkKCJt"
    "Y3BFbnYiKS52YWx1ZS5zcGxpdCgiXG4iKS5mb3JFYWNoKGw9PnsKICAgICAgICAgIGNvbnN0IGk9bC5pbmRleE9mKCI9Iik7IGlmKGk+MCkgZW52W2wu"
    "c2xpY2UoMCxpKS50cmltKCldPWwuc2xpY2UoaSsxKS50cmltKCk7IH0pOwogICAgICAgIGNsb3NlTW9kYWwoKTsgdG9hc3QoIkNvbm5lY3RpbmcgdG8g"
    "IituYW1lKyLigKYiKTsKICAgICAgICB0cnl7IGNvbnN0IHIgPSBhd2FpdCBwb3N0KCIvYXBpL21jcC9zZXJ2ZXJzIiwge25hbWUsIGNvbW1hbmQ6Y21k"
    "LCBhcmdzLCBlbnZ9KTsKICAgICAgICAgIGlmKHIuY29ubmVjdCAmJiByLmNvbm5lY3Qub2spIHRvYXN0KCJDb25uZWN0ZWQgIituYW1lLCJvayIpOwog"
    "ICAgICAgICAgZWxzZSB0b2FzdCgoci5jb25uZWN0JiZyLmNvbm5lY3QuZXJyb3IpfHwiQWRkZWQgYnV0IG5vdCBjb25uZWN0ZWQiLCJlcnIiKTsgfQog"
    "ICAgICAgIGNhdGNoKGUpeyB0b2FzdChlLm1lc3NhZ2UsImVyciIpOyB9IGxvYWRUb29scygpOwogICAgICB9fV19KTsKfQphc3luYyBmdW5jdGlvbiBw"
    "b2xsTWNwSW5zdGFsbCgpewogIGNvbnN0IGxvZyA9ICQoIm1jcExvZyIpOwogIGNvbnN0IHQgPSBzZXRJbnRlcnZhbChhc3luYygpPT57CiAgICB0cnl7"
    "IGNvbnN0IHN0ID0gYXdhaXQgYXBpKCIvYXBpL21jcC9pbnN0YWxsX3N0YXR1cyIpOwogICAgICBpZihsb2cgJiYgc3Quam9iKXsgbG9nLnRleHRDb250"
    "ZW50PShzdC5qb2IubG9nfHxbXSkuam9pbigiXG4iKTsgbG9nLnNjcm9sbFRvcD1sb2cuc2Nyb2xsSGVpZ2h0OyB9CiAgICAgIGlmKHN0Lmluc3RhbGxl"
    "ZCl7IGNsZWFySW50ZXJ2YWwodCk7IHRvYXN0KCJNQ1Agc3VwcG9ydCByZWFkeSIsIm9rIik7IGxvYWRUb29scygpOyB9CiAgICAgIGVsc2UgaWYoc3Qu"
    "am9iICYmIHN0LmpvYi5zdGF0dXM9PT0iZmFpbGVkIil7IGNsZWFySW50ZXJ2YWwodCk7CiAgICAgICAgdG9hc3QoIkluc3RhbGwgZmFpbGVkIiwiZXJy"
    "Iik7CiAgICAgICAgY29uc3QgYj0kKCJzZXR1cE1jcEJ0biIpOyBpZihiKXtiLmRpc2FibGVkPWZhbHNlO2IudGV4dENvbnRlbnQ9IlJldHJ5Ijt9IH0K"
    "ICAgIH1jYXRjaChlKXsgY2xlYXJJbnRlcnZhbCh0KTsgfQogIH0sIDE1MDApOwp9CmFzeW5jIGZ1bmN0aW9uIHJlbmRlclNlYXJjaCgpewogIGNvbnN0"
    "IGhvc3QgPSAkKCJzZWFyY2hCb2R5Iik7IGlmKCFob3N0KSByZXR1cm47CiAgaG9zdC5pbm5lckhUTUwgPSAnPGRpdiBjbGFzcz0icm93Ij48ZGl2IGNs"
    "YXNzPSJzcGluIj48L2Rpdj48c3BhbiBjbGFzcz0ibXV0ZWQiPkNoZWNraW5nIHNlYXJjaCBlbmdpbmVz4oCmPC9zcGFuPjwvZGl2Pic7CiAgbGV0IHM7"
    "CiAgdHJ5eyBzID0gYXdhaXQgYXBpKCIvYXBpL3NlYXJjaC9zdGF0dXMiKTsgfQogIGNhdGNoKGUpeyBob3N0LmlubmVySFRNTCA9IGVyckNhcmQoZS5t"
    "ZXNzYWdlKTsgcmV0dXJuOyB9CiAgY29uc3QgbGl2ZSA9IHMuc2VhcnhuZy5qc29uX29rOwogIGxldCBodG1sID0gYDxkaXYgY2xhc3M9InRvb2xyb3ci"
    "PjxkaXYgY2xhc3M9InRpYyI+8J+UjTwvZGl2PgogICAgPGRpdiBjbGFzcz0idGkiPjxkaXYgY2xhc3M9Im5tIj5BY3RpdmUgZW5naW5lOiAke2xpdmU/"
    "IlNlYXJYTkciOiJEdWNrRHVja0dvIChidWlsdC1pbiBmYWxsYmFjaykifTwvZGl2PgogICAgICA8ZGl2IGNsYXNzPSJkcyI+JHtsaXZlCiAgICAgICAg"
    "PyAiUHJpdmF0ZSBtZXRhc2VhcmNoIGF0IDxzcGFuIGNsYXNzPSdtb25vJz4iK2VzYyhzLnNlYXJ4bmdfdXJsKSsiPC9zcGFuPiDigJQgYWdncmVnYXRl"
    "cyBtYW55IGVuZ2luZXMsIG5vIHRyYWNraW5nLCBubyByYXRlIGxpbWl0cy4iCiAgICAgICAgOiAiV29ya3Mgb3V0IG9mIHRoZSBib3gsIGJ1dCBjYW4g"
    "YmUgc2xvdyBvciByYXRlLWxpbWl0ZWQuIFNlYXJYTkcgKGJlbG93KSBpcyB0aGUgcmVjb21tZW5kZWQgdXBncmFkZS4ifTwvZGl2PjwvZGl2PgogICAg"
    "PHNwYW4gY2xhc3M9ImNoaXAgJHtsaXZlPyJncm4iOiIifSI+JHtsaXZlPyJjb25uZWN0ZWQiOiJmYWxsYmFjayJ9PC9zcGFuPgogICAgPGJ1dHRvbiBj"
    "bGFzcz0iYnRuIHNtIGdob3N0IiBpZD0idGVzdFNlYXJjaEJ0biI+VGVzdDwvYnV0dG9uPjwvZGl2PmA7CiAgaWYoIWxpdmUpewogICAgaHRtbCArPSBg"
    "PGRpdiBjbGFzcz0iY2FyZCBwYWQtbGciIHN0eWxlPSJtYXJnaW46NHB4IDAgOHB4Ij4KICAgICAgPGI+U2V0IHVwIFNlYXJYTkcgPHNwYW4gY2xhc3M9"
    "ImNoaXAgdmlvIiBzdHlsZT0ibWFyZ2luLWxlZnQ6NnB4Ij5yZWNvbW1lbmRlZDwvc3Bhbj48L2I+CiAgICAgIDxwIGNsYXNzPSJtdXRlZCIgc3R5bGU9"
    "Im1hcmdpbjo3cHggMCAwO21heC13aWR0aDo2NGNoIj5TZWFyWE5HIGlzIGEgc2VsZi1ob3N0ZWQsIHByaXZhdGUKICAgICAgICBtZXRhLXNlYXJjaCBl"
    "bmdpbmUuIEhlb3J0aCBjYW4gc3RhcnQgb25lIGZvciB5b3UgYXMgYSBzbWFsbCBEb2NrZXIgY29udGFpbmVyCiAgICAgICAgKG9uZS10aW1lIH4zMDAg"
    "TUIgaW1hZ2UgZG93bmxvYWQpIGFuZCB3aWxsIHVzZSBpdCBhdXRvbWF0aWNhbGx5IGZvciB0aGUgYWdlbnQncwogICAgICAgIDxzcGFuIGNsYXNzPSJt"
    "b25vIj53ZWJfc2VhcmNoPC9zcGFuPiB0b29sLjwvcD4KICAgICAgPGRpdiBpZD0ic2VhcnhTdGVwcyIgc3R5bGU9Im1hcmdpbi10b3A6MTRweCI+PC9k"
    "aXY+CiAgICAgIDxkaXYgY2xhc3M9Imluc3RhbGxsb2ciIGlkPSJzZWFyeExvZyIgaGlkZGVuPjwvZGl2PjwvZGl2PmA7CiAgfQogIGhvc3QuaW5uZXJI"
    "VE1MID0gaHRtbDsKICAkKCJ0ZXN0U2VhcmNoQnRuIikub25jbGljayA9IHRlc3RTZWFyY2g7CiAgaWYobGl2ZSkgcmV0dXJuOwoKICBjb25zdCBzdGVw"
    "cyA9ICQoInNlYXJ4U3RlcHMiKTsKICBpZighcy5kb2NrZXIuaW5zdGFsbGVkKXsKICAgIHN0ZXBzLmlubmVySFRNTCA9IGA8ZGl2IGNsYXNzPSJyb3ci"
    "IHN0eWxlPSJnYXA6OXB4O21hcmdpbi1ib3R0b206MTBweCI+CiAgICAgIDxzcGFuIGNsYXNzPSJkb3Qgd2FybiI+PC9zcGFuPjxiPkRvY2tlciBpcyBy"
    "ZXF1aXJlZCBmaXJzdDwvYj48L2Rpdj4KICAgICAgPHAgY2xhc3M9Im11dGVkIiBzdHlsZT0ibWFyZ2luOjAgMCAxMHB4Ij5TZWFyWE5HIHJ1bnMgaW4g"
    "YSBjb250YWluZXIsIHNvIGluc3RhbGwgRG9ja2VyCiAgICAgICAgb25jZSDigJQgaXQgdGFrZXMgYSBjb3VwbGUgb2YgbWludXRlcywgdGhlbiBjb21l"
    "IGJhY2sgaGVyZTo8L3A+CiAgICAgICR7cy5kb2NrZXJfaGVscC5tYXAobD0+YDxkaXYgY2xhc3M9Im1vbm8iIHN0eWxlPSJmb250LXNpemU6MTJweDti"
    "YWNrZ3JvdW5kOnZhcigtLWluayk7Ym9yZGVyOjFweCBzb2xpZCB2YXIoLS1saW5lKTtib3JkZXItcmFkaXVzOjhweDtwYWRkaW5nOjlweCAxMXB4O21h"
    "cmdpbi1ib3R0b206NnB4Ij4ke2VzYyhsKX08L2Rpdj5gKS5qb2luKCIiKX0KICAgICAgPGJ1dHRvbiBjbGFzcz0iYnRuIHNtIGdob3N0IiBzdHlsZT0i"
    "bWFyZ2luLXRvcDo0cHgiIGlkPSJzZWFyeFJlY2hlY2siPkkndmUgaW5zdGFsbGVkIERvY2tlciDigJQgY2hlY2sgYWdhaW48L2J1dHRvbj5gOwogICAg"
    "JCgic2VhcnhSZWNoZWNrIikub25jbGljayA9IHJlbmRlclNlYXJjaDsKICAgIHJldHVybjsKICB9CiAgaWYoIXMuZG9ja2VyLmRhZW1vbil7CiAgICBz"
    "dGVwcy5pbm5lckhUTUwgPSBgPGRpdiBjbGFzcz0icm93IiBzdHlsZT0iZ2FwOjlweDttYXJnaW4tYm90dG9tOjEwcHgiPgogICAgICA8c3BhbiBjbGFz"
    "cz0iZG90IHdhcm4iPjwvc3Bhbj48Yj5Eb2NrZXIgaXMgaW5zdGFsbGVkIGJ1dCBub3QgcnVubmluZzwvYj48L2Rpdj4KICAgICAgPHAgY2xhc3M9Im11"
    "dGVkIiBzdHlsZT0ibWFyZ2luOjAgMCAxMnB4Ij5TdGFydCBEb2NrZXIgRGVza3RvcCAob24gTGludXg6CiAgICAgICAgPHNwYW4gY2xhc3M9Im1vbm8i"
    "PnN1ZG8gc3lzdGVtY3RsIHN0YXJ0IGRvY2tlcjwvc3Bhbj4pLCB0aGVuIGNoZWNrIGFnYWluLjwvcD4KICAgICAgPGJ1dHRvbiBjbGFzcz0iYnRuIHNt"
    "IGdob3N0IiBpZD0ic2VhcnhSZWNoZWNrIj5DaGVjayBhZ2FpbjwvYnV0dG9uPmA7CiAgICAkKCJzZWFyeFJlY2hlY2siKS5vbmNsaWNrID0gcmVuZGVy"
    "U2VhcmNoOwogICAgcmV0dXJuOwogIH0KICBjb25zdCBydW5uaW5nID0gcy5qb2IgJiYgcy5qb2Iuc3RhdHVzPT09InJ1bm5pbmciOwogIHN0ZXBzLmlu"
    "bmVySFRNTCA9IGAKICAgICR7cy5jb250YWluZXIuZXhpc3RzP2A8cCBjbGFzcz0iaGludCIgc3R5bGU9Im1hcmdpbjowIDAgMTBweCI+Q29udGFpbmVy"
    "CiAgICAgIDxzcGFuIGNsYXNzPSJtb25vIj4ke2VzYyhzLmNvbnRhaW5lci5uYW1lfHwiaGVvcnRoLXNlYXJ4bmciKX08L3NwYW4+IGV4aXN0cyDigJQg"
    "c3RhdHVzOiAke2VzYyhzLmNvbnRhaW5lci5zdGF0dXN8fCJzdG9wcGVkIil9LjwvcD5gOiIifQogICAgJHtzLnNlYXJ4bmcucmVhY2hhYmxlICYmIHMu"
    "c2VhcnhuZy5lcnJvcj9gPHAgY2xhc3M9ImhpbnQiIHN0eWxlPSJjb2xvcjp2YXIoLS1zaWduYWwpO21hcmdpbjowIDAgMTBweCI+JHtlc2Mocy5zZWFy"
    "eG5nLmVycm9yKX08L3A+YDoiIn0KICAgIDxkaXYgY2xhc3M9InJvdyIgc3R5bGU9ImdhcDoxMHB4O2ZsZXgtd3JhcDp3cmFwIj4KICAgICAgPGJ1dHRv"
    "biBjbGFzcz0iYnRuIHByaW1hcnkiIGlkPSJzdGFydFNlYXJ4QnRuIiAke3J1bm5pbmc/ImRpc2FibGVkIjoiIn0+CiAgICAgICAgJHtydW5uaW5nPyJT"
    "dGFydGluZ+KApiI6KHMuY29udGFpbmVyLmV4aXN0cz8iU3RhcnQgU2VhclhORyI6Ikluc3RhbGwgJiBzdGFydCBTZWFyWE5HIil9PC9idXR0b24+CiAg"
    "ICAgIDxidXR0b24gY2xhc3M9ImJ0biBzbSBnaG9zdCIgaWQ9InNlYXJ4UmVjaGVjayI+Q2hlY2sgYWdhaW48L2J1dHRvbj48L2Rpdj4KICAgIDxwIGNs"
    "YXNzPSJoaW50IiBzdHlsZT0ibWFyZ2luOjEycHggMCAwIj5QcmVmZXIgdG8gcnVuIGl0IHlvdXJzZWxmPyBVc2U6PGJyPgogICAgICA8c3BhbiBjbGFz"
    "cz0ibW9ubyIgc3R5bGU9InVzZXItc2VsZWN0OmFsbCI+JHtlc2Mocy5tYW51YWxfY21kKX08L3NwYW4+PC9wPmA7CiAgJCgic2VhcnhSZWNoZWNrIiku"
    "b25jbGljayA9IHJlbmRlclNlYXJjaDsKICAkKCJzdGFydFNlYXJ4QnRuIikub25jbGljayA9IGFzeW5jKCk9PnsKICAgICQoInN0YXJ0U2VhcnhCdG4i"
    "KS5kaXNhYmxlZCA9IHRydWU7ICQoInN0YXJ0U2VhcnhCdG4iKS50ZXh0Q29udGVudD0iU3RhcnRpbmfigKYiOwogICAgJCgic2VhcnhMb2ciKS5oaWRk"
    "ZW4gPSBmYWxzZTsKICAgIHRyeXsgYXdhaXQgcG9zdCgiL2FwaS9zZWFyY2gvc2V0dXAiKTsgcG9sbFNlYXJ4KCk7IH0KICAgIGNhdGNoKGUpeyB0b2Fz"
    "dChlLm1lc3NhZ2UsImVyciIpOyByZW5kZXJTZWFyY2goKTsgfQogIH07CiAgaWYocnVubmluZyl7ICQoInNlYXJ4TG9nIikuaGlkZGVuPWZhbHNlOwog"
    "ICAgJCgic2VhcnhMb2ciKS50ZXh0Q29udGVudD0ocy5qb2IubG9nfHxbXSkuam9pbigiXG4iKTsgcG9sbFNlYXJ4KCk7IH0KfQphc3luYyBmdW5jdGlv"
    "biB0ZXN0U2VhcmNoKCl7CiAgdG9hc3QoIlJ1bm5pbmcgYSB0ZXN0IHNlYXJjaOKApiIpOwogIHRyeXsKICAgIGNvbnN0IHIgPSBhd2FpdCBhcGkoIi9h"
    "cGkvc2VhcmNoL3Rlc3Q/cT0iK2VuY29kZVVSSUNvbXBvbmVudCgid2hhdCBpcyBzZWFyeG5nIikpOwogICAgbW9kYWwoe3RpdGxlOiJXZWIgc2VhcmNo"
    "IHRlc3QiLCBib2R5SFRNTDoKICAgICAgYDxwcmUgc3R5bGU9IndoaXRlLXNwYWNlOnByZS13cmFwO2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQt"
    "c2l6ZToxMnB4O21heC1oZWlnaHQ6MzYwcHg7b3ZlcmZsb3c6YXV0bzttYXJnaW46MCI+JHtlc2Moci5yZXN1bHQpfTwvcHJlPmAsCiAgICAgIGFjdGlv"
    "bnM6W3tsYWJlbDoiQ2xvc2UiLCBvbkNsaWNrOmNsb3NlTW9kYWx9XX0pOwogIH1jYXRjaChlKXsgdG9hc3QoZS5tZXNzYWdlLCJlcnIiKTsgfQp9Cmxl"
    "dCBzZWFyeFRpbWVyPW51bGw7CmZ1bmN0aW9uIHBvbGxTZWFyeCgpewogIGlmKHNlYXJ4VGltZXIpIGNsZWFySW50ZXJ2YWwoc2VhcnhUaW1lcik7CiAg"
    "bGV0IHRyaWVzPTA7CiAgc2VhcnhUaW1lciA9IHNldEludGVydmFsKGFzeW5jKCk9PnsKICAgIHRyaWVzKys7CiAgICB0cnl7CiAgICAgIGNvbnN0IHMg"
    "PSBhd2FpdCBhcGkoIi9hcGkvc2VhcmNoL3N0YXR1cyIpOwogICAgICBjb25zdCBsb2cgPSAkKCJzZWFyeExvZyIpOwogICAgICBpZihsb2cgJiYgcy5q"
    "b2IpeyBsb2cudGV4dENvbnRlbnQ9KHMuam9iLmxvZ3x8W10pLmpvaW4oIlxuIik7CiAgICAgICAgbG9nLnNjcm9sbFRvcD1sb2cuc2Nyb2xsSGVpZ2h0"
    "OyB9CiAgICAgIGlmKHMuc2VhcnhuZy5qc29uX29rKXsgY2xlYXJJbnRlcnZhbChzZWFyeFRpbWVyKTsKICAgICAgICB0b2FzdCgiU2VhclhORyBpcyBy"
    "dW5uaW5nIOKAlCB3ZWIgc2VhcmNoIHVwZ3JhZGVkIiwib2siKTsgcmVuZGVyU2VhcmNoKCk7IH0KICAgICAgZWxzZSBpZihzLmpvYiAmJiBzLmpvYi5z"
    "dGF0dXM9PT0iZmFpbGVkIil7IGNsZWFySW50ZXJ2YWwoc2VhcnhUaW1lcik7CiAgICAgICAgdG9hc3QoIlNlYXJYTkcgc3RhcnQgZmFpbGVkIOKAlCBz"
    "ZWUgdGhlIGxvZyIsImVyciIpOyB9CiAgICAgIGVsc2UgaWYodHJpZXM+NjApeyBjbGVhckludGVydmFsKHNlYXJ4VGltZXIpOwogICAgICAgIHRvYXN0"
    "KCJTZWFyWE5HIGlzIHRha2luZyBhIHdoaWxlIOKAlCBjaGVjayB0aGUgbG9nIG9yIHRyeSBhZ2FpbiIsImVyciIpOyB9CiAgICB9Y2F0Y2goZSl7IGNs"
    "ZWFySW50ZXJ2YWwoc2VhcnhUaW1lcik7IH0KICB9LCAxNjAwKTsKfQpsb2FkZXJzLnRvb2xzID0gbG9hZFRvb2xzOwoKLyogPT09PT09PT09PT09PT09"
    "PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09CiAgIENPTVBVVEVSIENPTlRST0wKICAgPT09PT09PT09PT09PT09PT09"
    "PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09ICovCmNvbnN0IENDX0lDT05TID0ge3NjcmVlbl9jYXB0dXJlOiLwn5O3Iiwg"
    "bW91c2VfbW92ZToi8J+WsSIsIG1vdXNlX2NsaWNrOiLwn5GGIiwKICBtb3VzZV9kcmFnOiLinIoiLCBzY3JvbGw6IuKGlSIsIHR5cGVfdGV4dDoi4oyo"
    "IiwgcHJlc3Nfa2V5czoi4oyoIiwgc3RvcDoi4puUIn07CmFzeW5jIGZ1bmN0aW9uIGxvYWRDb21wdXRlcigpewogIGNvbnN0IGhvc3QgPSAkKCJjb21w"
    "dXRlckJvZHkiKTsKICBob3N0LmlubmVySFRNTCA9ICc8ZGl2IGNsYXNzPSJyb3ciPjxkaXYgY2xhc3M9InNwaW4iPjwvZGl2PjxzcGFuIGNsYXNzPSJt"
    "dXRlZCI+Q2hlY2tpbmfigKY8L3NwYW4+PC9kaXY+JzsKICBsZXQgczsKICB0cnl7IHMgPSBhd2FpdCBhcGkoIi9hcGkvY29tcHV0ZXIvc3RhdHVzIik7"
    "IH1jYXRjaChlKXsgaG9zdC5pbm5lckhUTUw9ZXJyQ2FyZChlLm1lc3NhZ2UpOyByZXR1cm47IH0KCiAgaWYoIXMuaW5zdGFsbGVkKXsgcmVuZGVyQ29t"
    "cHV0ZXJTZXR1cChob3N0LCBzKTsgcmV0dXJuOyB9CiAgaWYoIXMuZW5hYmxlZCl7IHJlbmRlckNvbXB1dGVyQ29uc2VudChob3N0LCBzKTsgcmV0dXJu"
    "OyB9CiAgcmVuZGVyQ29tcHV0ZXJQYW5lbChob3N0LCBzKTsKfQpmdW5jdGlvbiByZW5kZXJDb21wdXRlclNldHVwKGhvc3QsIHMpewogIGNvbnN0IHJ1"
    "bm5pbmcgPSBzLmpvYiAmJiBzLmpvYi5zdGF0dXM9PT0icnVubmluZyI7CiAgaG9zdC5pbm5lckhUTUwgPSBgPGRpdiBjbGFzcz0iZGFuZ2VyLWNhcmQi"
    "PgogICAgPGRpdiBjbGFzcz0icm93IiBzdHlsZT0iZ2FwOjEwcHg7bWFyZ2luLWJvdHRvbTo2cHgiPjxzcGFuIHN0eWxlPSJmb250LXNpemU6MjRweCI+"
    "8J+Wpe+4jzwvc3Bhbj4KICAgICAgPGgzIHN0eWxlPSJtYXJnaW46MCI+SW5zdGFsbCBjb21wdXRlciBjb250cm9sPC9oMz48L2Rpdj4KICAgIDxwIGNs"
    "YXNzPSJtdXRlZCIgc3R5bGU9Im1hcmdpbjowIDAgNnB4O21heC13aWR0aDo2NmNoIj5UaGlzIGluc3RhbGxzIFB5QXV0b0dVSSBpbnRvIEhlb3J0aCdz"
    "CiAgICAgIHByaXZhdGUgZW52aXJvbm1lbnQgc28gYSBtb2RlbCBjYW4gbW92ZSB0aGUgbW91c2UsIHR5cGUsIGFuZCB0YWtlIHNjcmVlbnNob3RzLiBJ"
    "dCdzIGEgc21hbGwsCiAgICAgIG9uZS10aW1lIGluc3RhbGwuPC9wPgogICAgPHVsIGNsYXNzPSJ3YXJubGlzdCI+JHsocy5vc19ub3Rlc3x8W10pLm1h"
    "cChuPT5gPGxpPiR7ZXNjKG4pfTwvbGk+YCkuam9pbigiIil9PC91bD4KICAgIDxkaXYgc3R5bGU9Im1hcmdpbi10b3A6OHB4Ij48YnV0dG9uIGNsYXNz"
    "PSJidG4gcHJpbWFyeSIgaWQ9ImNjU2V0dXBCdG4iICR7cnVubmluZz8iZGlzYWJsZWQiOiIifT4KICAgICAgJHtydW5uaW5nPyJJbnN0YWxsaW5n4oCm"
    "IjoiSW5zdGFsbCBjb21wdXRlciBjb250cm9sIn08L2J1dHRvbj48L2Rpdj4KICAgIDxkaXYgY2xhc3M9Imluc3RhbGxsb2ciIGlkPSJjY0xvZyIgJHtz"
    "LmpvYj8iIjoiaGlkZGVuIn0+JHtzLmpvYj9lc2MoKHMuam9iLmxvZ3x8W10pLmpvaW4oIlxuIikpOiIifTwvZGl2PgogIDwvZGl2PmA7CiAgJCgiY2NT"
    "ZXR1cEJ0biIpLm9uY2xpY2sgPSBhc3luYygpPT57ICQoImNjU2V0dXBCdG4iKS5kaXNhYmxlZD10cnVlOwogICAgJCgiY2NTZXR1cEJ0biIpLnRleHRD"
    "b250ZW50PSJJbnN0YWxsaW5n4oCmIjsgJCgiY2NMb2ciKS5oaWRkZW49ZmFsc2U7CiAgICB0cnl7IGF3YWl0IHBvc3QoIi9hcGkvY29tcHV0ZXIvc2V0"
    "dXAiKTsgcG9sbENDSW5zdGFsbCgpOyB9Y2F0Y2goZSl7IHRvYXN0KGUubWVzc2FnZSwiZXJyIik7IH0gfTsKICBpZihydW5uaW5nKSBwb2xsQ0NJbnN0"
    "YWxsKCk7Cn0KYXN5bmMgZnVuY3Rpb24gcG9sbENDSW5zdGFsbCgpewogIGNvbnN0IGxvZz0kKCJjY0xvZyIpOwogIGNvbnN0IHQ9c2V0SW50ZXJ2YWwo"
    "YXN5bmMoKT0+ewogICAgdHJ5eyBjb25zdCBzPWF3YWl0IGFwaSgiL2FwaS9jb21wdXRlci9zdGF0dXMiKTsKICAgICAgaWYobG9nJiZzLmpvYil7IGxv"
    "Zy50ZXh0Q29udGVudD0ocy5qb2IubG9nfHxbXSkuam9pbigiXG4iKTsgbG9nLnNjcm9sbFRvcD1sb2cuc2Nyb2xsSGVpZ2h0OyB9CiAgICAgIGlmKHMu"
    "aW5zdGFsbGVkKXsgY2xlYXJJbnRlcnZhbCh0KTsgdG9hc3QoIkNvbXB1dGVyIGNvbnRyb2wgaW5zdGFsbGVkIiwib2siKTsgbG9hZENvbXB1dGVyKCk7"
    "IH0KICAgICAgZWxzZSBpZihzLmpvYiYmcy5qb2Iuc3RhdHVzPT09ImZhaWxlZCIpeyBjbGVhckludGVydmFsKHQpOyB0b2FzdCgiSW5zdGFsbCBmYWls"
    "ZWQiLCJlcnIiKTsKICAgICAgICBjb25zdCBiPSQoImNjU2V0dXBCdG4iKTsgaWYoYil7Yi5kaXNhYmxlZD1mYWxzZTtiLnRleHRDb250ZW50PSJSZXRy"
    "eSBpbnN0YWxsIjt9IH0KICAgIH1jYXRjaChlKXsgY2xlYXJJbnRlcnZhbCh0KTsgfQogIH0sMTUwMCk7Cn0KZnVuY3Rpb24gcmVuZGVyQ29tcHV0ZXJD"
    "b25zZW50KGhvc3QsIHMpewogIGhvc3QuaW5uZXJIVE1MID0gYDxkaXYgY2xhc3M9ImRhbmdlci1jYXJkIj4KICAgIDxkaXYgY2xhc3M9InJvdyIgc3R5"
    "bGU9ImdhcDoxMHB4O21hcmdpbi1ib3R0b206OHB4Ij4KICAgICAgPHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIg"
    "ZmlsbD0ibm9uZSIgc3Ryb2tlPSJ2YXIoLS1yZWQpIiBzdHJva2Utd2lkdGg9IjIiPjxwYXRoIGQ9Ik0xMiA5djRtMCA0aC4wMU0xMC4zIDMuOSAxLjgg"
    "MThhMiAyIDAgMCAwIDEuNyAzaDE3YTIgMiAwIDAgMCAxLjctM0wxMy43IDMuOWEyIDIgMCAwIDAtMy40IDBaIi8+PC9zdmc+CiAgICAgIDxoMyBzdHls"
    "ZT0ibWFyZ2luOjAiPlJlYWQgdGhpcyBiZWZvcmUgZW5hYmxpbmc8L2gzPjwvZGl2PgogICAgPHAgY2xhc3M9Im11dGVkIiBzdHlsZT0ibWFyZ2luOjAg"
    "MCA0cHg7bWF4LXdpZHRoOjY4Y2giPldoZW4gZW5hYmxlZCwgdGhlIG1vZGVsIGNhbiBjb250cm9sIHlvdXIKICAgICAgY29tcHV0ZXIgYXMgaWYgaXQg"
    "d2VyZSB5b3UuIExvY2FsIG1vZGVscyBtYWtlIG1pc3Rha2VzLiBQbGVhc2UgdW5kZXJzdGFuZDo8L3A+CiAgICA8dWwgY2xhc3M9Indhcm5saXN0Ij4K"
    "ICAgICAgPGxpPkl0IGNhbiBjbGljaywgdHlwZSwgb3BlbiBhcHBzLCBhbmQgY2hhbmdlIG9yIGRlbGV0ZSB0aGluZ3Mg4oCUIGFueXdoZXJlIG9uIHlv"
    "dXIgc2NyZWVuLjwvbGk+CiAgICAgIDxsaT5JdCBvbmx5IGRvZXMgdGhpcyB3aGlsZSA8Yj5Db21wdXRlcjwvYj4gbW9kZSBpcyBvbiBpbiBDaGF0IGFu"
    "ZCB5b3Ugc2VuZCBhIHJlcXVlc3QuPC9saT4KICAgICAgPGxpPktlZXAgPGI+4oCcQXNrIGJlZm9yZSBlYWNoIGFjdGlvbuKAnTwvYj4gb24gKGJlbG93"
    "KSB1bnRpbCB5b3UgdHJ1c3QgaXQg4oCUIHlvdSBhcHByb3ZlIGV2ZXJ5IGNsaWNrIGFuZCBrZXlzdHJva2UuPC9saT4KICAgICAgPGxpPkEgYmlnIHJl"
    "ZCA8Yj5FbWVyZ2VuY3kgU3RvcDwvYj4gaXMgYWx3YXlzIGF2YWlsYWJsZSwgYW5kIHNsYW1taW5nIHlvdXIgbW91c2UgaW50byBhbnkgc2NyZWVuIGNv"
    "cm5lciBpbnN0YW50bHkgYWJvcnRzIChQeUF1dG9HVUkgZmFpbC1zYWZlKS48L2xpPgogICAgICA8bGk+RG9uJ3QgbGVhdmUgaXQgdW5hdHRlbmRlZC4g"
    "QXZvaWQgdGFza3MgaW52b2x2aW5nIHBhc3N3b3JkcywgcGF5bWVudHMsIG9yIGlycmV2ZXJzaWJsZSBhY3Rpb25zLjwvbGk+CiAgICA8L3VsPgogICAg"
    "PGxhYmVsIGNsYXNzPSJjb25zZW50IiBpZD0iY2NDb25zZW50Ij4KICAgICAgPGlucHV0IHR5cGU9ImNoZWNrYm94IiBpZD0iY2NDb25zZW50Qm94Ij4K"
    "ICAgICAgPHNwYW4+SSB1bmRlcnN0YW5kIHRoZSByaXNrcyBhbmQgd2FudCB0byBsZXQgbW9kZWxzIGNvbnRyb2wgdGhpcyBjb21wdXRlci4gSSdtIHJl"
    "c3BvbnNpYmxlIGZvciB3aGF0IGl0IGRvZXMuPC9zcGFuPjwvbGFiZWw+CiAgICA8ZGl2IGNsYXNzPSJyb3ciIHN0eWxlPSJnYXA6MTBweCI+CiAgICAg"
    "IDxidXR0b24gY2xhc3M9ImJ0biBwcmltYXJ5IiBpZD0iY2NFbmFibGVCdG4iIGRpc2FibGVkIHN0eWxlPSJiYWNrZ3JvdW5kOnZhcigtLXJlZCk7Ym94"
    "LXNoYWRvdzpub25lIj5FbmFibGUgY29tcHV0ZXIgY29udHJvbDwvYnV0dG9uPgogICAgICA8c3BhbiBjbGFzcz0iaGludCI+WW91IGNhbiB0dXJuIHRo"
    "aXMgb2ZmIGFueXRpbWUsIGhlcmUgb3IgaW4gU2V0dGluZ3MuPC9zcGFuPjwvZGl2PgogIDwvZGl2PmA7CiAgJCgiY2NDb25zZW50Qm94Iikub25jaGFu"
    "Z2UgPSAoZSk9PnsgJCgiY2NFbmFibGVCdG4iKS5kaXNhYmxlZCA9ICFlLnRhcmdldC5jaGVja2VkOyB9OwogICQoImNjRW5hYmxlQnRuIikub25jbGlj"
    "ayA9IGFzeW5jKCk9PnsKICAgIHRyeXsgYXdhaXQgcG9zdCgiL2FwaS9zZXR0aW5ncyIsIHtjb21wdXRlcl9jb250cm9sOiIxIn0pOwogICAgICB0b2Fz"
    "dCgiQ29tcHV0ZXIgY29udHJvbCBlbmFibGVkIiwib2siKTsgc3RhdGUuc2V0dGluZ3MuY29tcHV0ZXJfY29udHJvbD0iMSI7IGxvYWRDb21wdXRlcigp"
    "OyB9CiAgICBjYXRjaChlKXsgdG9hc3QoZS5tZXNzYWdlLCJlcnIiKTsgfQogIH07Cn0KZnVuY3Rpb24gcmVuZGVyQ29tcHV0ZXJQYW5lbChob3N0LCBz"
    "KXsKICBob3N0LmlubmVySFRNTCA9IGAKICAgIDxkaXYgY2xhc3M9ImNjLXN0YXR1cyI+CiAgICAgIDxzcGFuIGNsYXNzPSJjYy1waWxsICR7cy5zdG9w"
    "cGVkPydvZmYnOidsaXZlJ30iPgogICAgICAgIDxzcGFuIGNsYXNzPSJkb3QgJHtzLnN0b3BwZWQ/J29mZic6J29uJ30iPjwvc3Bhbj4ke3Muc3RvcHBl"
    "ZD8nU3RvcHBlZCc6J1JlYWR5J308L3NwYW4+CiAgICAgIDxzcGFuIGNsYXNzPSJjYy1waWxsIG9mZiI+8J+WpSAke3Muc2NyZWVuLnd9w5cke3Muc2Ny"
    "ZWVuLmh9PC9zcGFuPgogICAgICA8c3BhbiBjbGFzcz0iY2MtcGlsbCBvZmYiPiR7cy5jb25maXJtPydBc2tzIGJlZm9yZSBlYWNoIGFjdGlvbic6J0Fj"
    "dHMgd2l0aG91dCBhc2tpbmcnfTwvc3Bhbj4KICAgICAgPGRpdiBzdHlsZT0iZmxleDoxIj48L2Rpdj4KICAgICAgPGJ1dHRvbiBjbGFzcz0iYnRuIHNt"
    "IGdob3N0IiBpZD0iY2NEaXNhYmxlQnRuIj5UdXJuIG9mZjwvYnV0dG9uPgogICAgPC9kaXY+CiAgICA8YnV0dG9uIGNsYXNzPSJlc3RvcCIgaWQ9ImNj"
    "RXN0b3BCdG4iPgogICAgICA8c3ZnIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJjdXJyZW50Q29sb3IiIHN0cm9rZS13aWR0"
    "aD0iMi40Ij48cmVjdCB4PSI2IiB5PSI2IiB3aWR0aD0iMTIiIGhlaWdodD0iMTIiIHJ4PSIyIi8+PC9zdmc+CiAgICAgIEVNRVJHRU5DWSBTVE9QPC9i"
    "dXR0b24+CiAgICA8cCBjbGFzcz0iaGludCIgc3R5bGU9InRleHQtYWxpZ246Y2VudGVyO21hcmdpbjo5cHggMCAxOHB4Ij5TdG9wcyBhbGwgYWN0aXZp"
    "dHkgaW1tZWRpYXRlbHkuCiAgICAgIFlvdSBjYW4gYWxzbyBzbGFtIHRoZSBtb3VzZSBpbnRvIGFueSBzY3JlZW4gY29ybmVyLjwvcD4KCiAgICA8ZGl2"
    "IGNsYXNzPSJjYXJkIHBhZC1sZyIgc3R5bGU9Im1hcmdpbi1ib3R0b206MTZweCI+CiAgICAgIDxiPkhvdyB0byB1c2UgaXQ8L2I+CiAgICAgIDxwIGNs"
    "YXNzPSJtdXRlZCIgc3R5bGU9Im1hcmdpbjo3cHggMCAwIj5HbyB0byA8Yj5DaGF0PC9iPiwgc3dpdGNoIG9uIHRoZSByZWQgPGI+Q29tcHV0ZXI8L2I+"
    "CiAgICAgICAgdG9nZ2xlLCBhbmQgZGVzY3JpYmUgdGhlIHRhc2sg4oCUIGUuZy4g4oCcb3BlbiB0aGUgY2FsY3VsYXRvciBhbmQgY29tcHV0ZSA0OCDD"
    "lyAxMuKAnSwgb3IKICAgICAgICDigJx0YWtlIGEgc2NyZWVuc2hvdCBhbmQgdGVsbCBtZSB3aGF0IGFwcCBpcyBpbiBmb2N1c+KAnS4gVGhlIG1vZGVs"
    "IHdpbGwgbG9vayBhdCB0aGUgc2NyZWVuIGFuZAogICAgICAgIGFjdCBzdGVwIGJ5IHN0ZXAuIEV2ZXJ5IGFjdGlvbiBzaG93cyB1cCBpbiBDaGF0IGFu"
    "ZCBpbiB0aGUgbG9nIGJlbG93LiBCZXN0IHdpdGggYQogICAgICAgIHRvb2wtY2FwYWJsZSwgdmlzaW9uLWZyaWVuZGx5IG1vZGVsLjwvcD4KICAgIDwv"
    "ZGl2PgoKICAgIDxkaXYgY2xhc3M9InNwcmVhZCIgc3R5bGU9Im1hcmdpbi1ib3R0b206MTBweCI+CiAgICAgIDxkaXYgY2xhc3M9InNlY3Rpb24tdGl0"
    "bGUiIHN0eWxlPSJtYXJnaW46MCI+UmVjZW50IGFjdGl2aXR5PC9kaXY+CiAgICAgIDxidXR0b24gY2xhc3M9ImJ0biBzbSBnaG9zdCIgaWQ9ImNjUmVm"
    "cmVzaExvZyI+UmVmcmVzaDwvYnV0dG9uPjwvZGl2PgogICAgPGRpdiBjbGFzcz0iY2Nsb2ciIGlkPSJjY0xvZ0xpc3QiPjwvZGl2PmA7CiAgJCgiY2NE"
    "aXNhYmxlQnRuIikub25jbGljayA9IGFzeW5jKCk9PnsgYXdhaXQgcG9zdCgiL2FwaS9zZXR0aW5ncyIse2NvbXB1dGVyX2NvbnRyb2w6IjAifSk7CiAg"
    "ICBzdGF0ZS5zZXR0aW5ncy5jb21wdXRlcl9jb250cm9sPSIwIjsgdG9hc3QoIkNvbXB1dGVyIGNvbnRyb2wgdHVybmVkIG9mZiIpOyBsb2FkQ29tcHV0"
    "ZXIoKTsgfTsKICAkKCJjY0VzdG9wQnRuIikub25jbGljayA9IGVtZXJnZW5jeVN0b3A7CiAgJCgiY2NSZWZyZXNoTG9nIikub25jbGljayA9IGxvYWRD"
    "b21wdXRlcjsKICByZW5kZXJDQ0xvZyhzLmxvZ3x8W10pOwp9CmZ1bmN0aW9uIHJlbmRlckNDTG9nKGxvZyl7CiAgY29uc3QgbGlzdD0kKCJjY0xvZ0xp"
    "c3QiKTsgaWYoIWxpc3QpIHJldHVybjsKICBpZighbG9nLmxlbmd0aCl7IGxpc3QuaW5uZXJIVE1MPSc8ZGl2IGNsYXNzPSJlbXB0eSIgc3R5bGU9InBh"
    "ZGRpbmc6MjZweCI+PGRpdiBjbGFzcz0iYmlnIj7wn5axPC9kaXY+Tm8gYWN0aW9ucyB5ZXQ8L2Rpdj4nOyByZXR1cm47IH0KICBsaXN0LmlubmVySFRN"
    "TD0iIjsKICBbLi4ubG9nXS5yZXZlcnNlKCkuZm9yRWFjaChhPT57CiAgICBjb25zdCByb3c9ZWwoImRpdiIsImNjcm93Iik7CiAgICByb3cuaW5uZXJI"
    "VE1MPWA8ZGl2IGNsYXNzPSJjaSI+JHtDQ19JQ09OU1thLmtpbmRdfHwi4oCiIn08L2Rpdj4KICAgICAgPGRpdiBjbGFzcz0iY2QiPjxkaXY+JHtlc2Mo"
    "YS5kZXRhaWx8fGEua2luZCl9PC9kaXY+CiAgICAgICAgPGRpdiBjbGFzcz0iY3QiPiR7bmV3IERhdGUoYS50cyoxMDAwKS50b0xvY2FsZVRpbWVTdHJp"
    "bmcoKX08L2Rpdj48L2Rpdj4KICAgICAgJHthLnNob3Q/YDxpbWcgc3JjPSIvYXBpL2NvbXB1dGVyL3Nob3QvJHthLnNob3R9IiB0aXRsZT0iQ2xpY2sg"
    "dG8gdmlldyI+YDoiIn1gOwogICAgaWYoYS5zaG90KSByb3cucXVlcnlTZWxlY3RvcigiaW1nIikub25jbGljaz0oKT0+d2luZG93Lm9wZW4oIi9hcGkv"
    "Y29tcHV0ZXIvc2hvdC8iK2Euc2hvdCk7CiAgICBsaXN0LmFwcGVuZENoaWxkKHJvdyk7CiAgfSk7Cn0KYXN5bmMgZnVuY3Rpb24gZW1lcmdlbmN5U3Rv"
    "cCgpewogIHRyeXsgYXdhaXQgcG9zdCgiL2FwaS9jb21wdXRlci9zdG9wIik7IHRvYXN0KCJFbWVyZ2VuY3kgc3RvcCBzZW50Iiwib2siKTsKICAgIGlm"
    "KGN1cnJlbnRWaWV3PT09ImNvbXB1dGVyIikgbG9hZENvbXB1dGVyKCk7IH0KICBjYXRjaChlKXsgdG9hc3QoZS5tZXNzYWdlLCJlcnIiKTsgfQp9Cmxv"
    "YWRlcnMuY29tcHV0ZXIgPSBsb2FkQ29tcHV0ZXI7CgovKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09"
    "PT09PT09PT0KICAgU0VUVElOR1MgICgrIHNlbGYtdXBkYXRlKQogICA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09"
    "PT09PT09PT09PT09PT0gKi8KZnVuY3Rpb24gc2V0dGluZ1JvdyhsYWIsIHN1YiwgY3RsSFRNTCl7CiAgcmV0dXJuIGA8ZGl2IGNsYXNzPSJzZXR0aW5n"
    "cm93Ij48ZGl2IGNsYXNzPSJsYWIiPiR7bGFifQogICAgJHtzdWI/YDxkaXYgY2xhc3M9InN1YiI+JHtzdWJ9PC9kaXY+YDoiIn08L2Rpdj48ZGl2IGNs"
    "YXNzPSJjdGwiPiR7Y3RsSFRNTH08L2Rpdj48L2Rpdj5gOwp9CmFzeW5jIGZ1bmN0aW9uIGxvYWRTZXR0aW5ncygpewogIGNvbnN0IGhvc3QgPSAkKCJz"
    "ZXR0aW5nc0JvZHkiKTsKICBsZXQgczsKICB0cnl7IGNvbnN0IHIgPSBhd2FpdCBhcGkoIi9hcGkvc2V0dGluZ3MiKTsgcyA9IHIuc2V0dGluZ3M7IHN0"
    "YXRlLnNldHRpbmdzID0gczsgfQogIGNhdGNoKGUpeyBob3N0LmlubmVySFRNTCA9IGVyckNhcmQoZS5tZXNzYWdlKTsgcmV0dXJuOyB9CiAgY29uc3Qg"
    "c3cgPSAoa2V5LCBvbik9PmA8ZGl2IGNsYXNzPSJzd2l0Y2ggJHtvbj8ib24iOiIifSIgZGF0YS1rZXk9IiR7a2V5fSI+PC9kaXY+YDsKICBob3N0Lmlu"
    "bmVySFRNTCA9IGAKICAgIDxkaXYgY2xhc3M9ImNhcmQgcGFkLWxnIiBzdHlsZT0ibWFyZ2luLWJvdHRvbToxNnB4Ij4KICAgICAgPGRpdiBjbGFzcz0i"
    "c2VjdGlvbi10aXRsZSIgc3R5bGU9Im1hcmdpbi10b3A6MCI+QmVoYXZpb3VyPC9kaXY+CiAgICAgICR7c2V0dGluZ1JvdygiU3lzdGVtIHByb21wdCIs"
    "IlNldHMgdGhlIGFzc2lzdGFudCdzIHBlcnNvbmFsaXR5IGFuZCBydWxlcy4iLAogICAgICAgIGA8dGV4dGFyZWEgY2xhc3M9InRhIiBpZD0ic2V0U3lz"
    "UHJvbXB0IiBzdHlsZT0ibWluLXdpZHRoOjI4MHB4Ij4ke2VzYyhzLnN5c3RlbV9wcm9tcHQpfTwvdGV4dGFyZWE+YCl9CiAgICAgICR7c2V0dGluZ1Jv"
    "dygiQ29udGV4dCBtZXNzYWdlcyIsIkhvdyBtYW55IHJlY2VudCBtZXNzYWdlcyB0byBzZW5kIGVhY2ggdHVybi4iLAogICAgICAgIGA8aW5wdXQgY2xh"
    "c3M9ImlucCBtb25vIiBpZD0ic2V0Q29udGV4dCIgdmFsdWU9IiR7ZXNjKHMuY29udGV4dF9tZXNzYWdlcyl9IiBzdHlsZT0id2lkdGg6OTBweCI+YCl9"
    "CiAgICAgICR7c2V0dGluZ1JvdygiV2ViIHRvb2xzIiwiTGV0IHRoZSBhZ2VudCBzZWFyY2ggdGhlIHdlYiBhbmQgZmV0Y2ggcGFnZXMuIiwKICAgICAg"
    "ICBzdygiYWxsb3dfd2ViX3Rvb2xzIiwgcy5hbGxvd193ZWJfdG9vbHM9PT0iMSIpKX0KICAgICAgJHtzZXR0aW5nUm93KCJBdXRvbWF0aWMgd2ViIHNl"
    "YXJjaCIsIkluIHBsYWluIGNoYXQsIEhlb3J0aCBxdWlldGx5IGNoZWNrcyBpZiBhIHF1ZXN0aW9uIG5lZWRzIGN1cnJlbnQgaW5mbyBhbmQgc2VhcmNo"
    "ZXMgb25seSB3aGVuIGl0IGhlbHBzIOKAlCBubyBuZWVkIHRvIHN3aXRjaCBvbiBBZ2VudC4gUmVxdWlyZXMgV2ViIHRvb2xzLiIsCiAgICAgICAgc3co"
    "ImF1dG9fc2VhcmNoIiwgcy5hdXRvX3NlYXJjaD09PSIxIikpfQogICAgICAke3NldHRpbmdSb3coIkFsbG93IGNvZGUgZXhlY3V0aW9uIiwKICAgICAg"
    "ICAiTGV0cyB0aGUgYWdlbnQgcnVuIFB5dGhvbiBhbmQgc2hlbGwgY29tbWFuZHMgaW4gaXRzIHdvcmtzcGFjZS4gT25seSBlbmFibGUgaWYgeW91IHRy"
    "dXN0IHRoZSBtb2RlbC4iLAogICAgICAgIHN3KCJhbGxvd19jb2RlX2V4ZWN1dGlvbiIsIHMuYWxsb3dfY29kZV9leGVjdXRpb249PT0iMSIpKX0KICAg"
    "ICAgJHtzZXR0aW5nUm93KCJBZ2VudCBzdGVwIGxpbWl0IiwiTWF4aW11bSB0b29sIGNhbGxzIGJlZm9yZSB0aGUgYWdlbnQgbXVzdCBhbnN3ZXIuIiwK"
    "ICAgICAgICBgPGlucHV0IGNsYXNzPSJpbnAgbW9ubyIgaWQ9InNldFN0ZXBzIiB2YWx1ZT0iJHtlc2Mocy5hZ2VudF9tYXhfc3RlcHMpfSIgc3R5bGU9"
    "IndpZHRoOjkwcHgiPmApfQogICAgICAke3NldHRpbmdSb3coIkxvb3AgaXRlcmF0aW9uIGxpbWl0IiwiQ2VpbGluZyBmb3IgYXV0b25vbW91cyBMb29w"
    "IHJ1bnMg4oCUIHRoZSBsb29wIHN0b3BzIGhlcmUgZXZlbiBpZiBpdCBoYXNuJ3QgY2FsbGVkIHRoZSB0YXNrIGRvbmUuIiwKICAgICAgICBgPGlucHV0"
    "IGNsYXNzPSJpbnAgbW9ubyIgaWQ9InNldExvb3BTdGVwcyIgdmFsdWU9IiR7ZXNjKHMubG9vcF9tYXhfc3RlcHMpfSIgc3R5bGU9IndpZHRoOjkwcHgi"
    "PmApfQogICAgICAke3NldHRpbmdSb3coIkNvdW5jaWwgc2l6ZSIsIkNvbnN1bHRhbnRzIHBlciBDb3VuY2lsIHJ1biAoMuKAkzEwKS4gM+KAkzUgaXMg"
    "dGhlIHN3ZWV0IHNwb3Qgb24gbG9jYWwgaGFyZHdhcmU7IDEwIHdvcmtzIGJ1dCBpcyBzbG93LiIsCiAgICAgICAgYDxpbnB1dCBjbGFzcz0iaW5wIG1v"
    "bm8iIGlkPSJzZXRDb3VuY2lsU2l6ZSIgdmFsdWU9IiR7ZXNjKHMuY291bmNpbF9zaXplKX0iIHN0eWxlPSJ3aWR0aDo5MHB4Ij5gKX0KICAgICAgJHtz"
    "ZXR0aW5nUm93KCJDb25zdWx0YXRpb24gcm91bmRzIiwiQWZ0ZXIgdGhlIGluZGVwZW5kZW50IHRha2VzLCBob3cgbWFueSByb3VuZHMgdGhlIGNvbnN1"
    "bHRhbnRzIHNwZW5kIGNyaXRpcXVpbmcgZWFjaCBvdGhlciAoMOKAkzMpLiIsCiAgICAgICAgYDxpbnB1dCBjbGFzcz0iaW5wIG1vbm8iIGlkPSJzZXRD"
    "b3VuY2lsUm91bmRzIiB2YWx1ZT0iJHtlc2Mocy5jb3VuY2lsX3JvdW5kcyl9IiBzdHlsZT0id2lkdGg6OTBweCI+YCl9CiAgICAgICR7c2V0dGluZ1Jv"
    "dygiQ291bmNpbCByZXNlYXJjaCBicmllZiIsIlJ1biBhIHF1aWNrIHNoYXJlZCB3ZWIgc2VhcmNoIGJlZm9yZSB0aGUgcGFuZWwgc3RhcnRzLCBzbyBl"
    "dmVyeSBjb25zdWx0YW50IGFyZ3VlcyBmcm9tIHRoZSBzYW1lIGZhY3RzLiIsCiAgICAgICAgc3coImNvdW5jaWxfcmVzZWFyY2giLCBzLmNvdW5jaWxf"
    "cmVzZWFyY2g9PT0iMSIpKX0KICAgIDwvZGl2PgogICAgPGRpdiBjbGFzcz0iY2FyZCBwYWQtbGciIGlkPSJjb2RlckNhcmQiIHN0eWxlPSJtYXJnaW4t"
    "Ym90dG9tOjE2cHgiPgogICAgICA8ZGl2IGNsYXNzPSJzZWN0aW9uLXRpdGxlIiBzdHlsZT0ibWFyZ2luLXRvcDowIj5Db2RlcjwvZGl2PgogICAgICAk"
    "e3NldHRpbmdSb3coIlByb2plY3QgZm9sZGVyIiwiQWJzb2x1dGUgcGF0aCB0byB0aGUgY29kZWJhc2UgQ29kZXIgbW9kZSB3b3JrcyBvbi4gQWxsIENv"
    "ZGVyIHRvb2xzIGFyZSBsb2NrZWQgaW5zaWRlIHRoaXMgZm9sZGVyLiIsCiAgICAgICAgYDxpbnB1dCBjbGFzcz0iaW5wIG1vbm8iIGlkPSJzZXRDb2Rl"
    "clJvb3QiIHZhbHVlPSIke2VzYyhzLmNvZGVyX3Jvb3R8fCIiKX0iIHBsYWNlaG9sZGVyPSIvaG9tZS95b3UvbXlwcm9qZWN0IiBzdHlsZT0ibWluLXdp"
    "ZHRoOjI2MHB4Ij5gKX0KICAgICAgJHtzZXR0aW5nUm93KCJBbGxvdyBmaWxlIGVkaXRzIiwiT2ZmID0gcGxhbiBtb2RlOiBDb2RlciByZWFkcywgc2Vh"
    "cmNoZXMgYW5kIHByb3Bvc2VzIGRpZmZzIGZvciB5b3UgdG8gYXBwbHkuIE9uID0gYnVpbGQgbW9kZTogaXQgZWRpdHMgZmlsZXMgZGlyZWN0bHkuIiwK"
    "ICAgICAgICBzdygiY29kZXJfYWxsb3dfd3JpdGUiLCBzLmNvZGVyX2FsbG93X3dyaXRlPT09IjEiKSl9CiAgICAgICR7c2V0dGluZ1JvdygiQ29kZXIg"
    "c3RlcCBsaW1pdCIsIk1heGltdW0gdG9vbCBjYWxscyBwZXIgQ29kZXIgcnVuIGJlZm9yZSBpdCBtdXN0IHN0b3AuIiwKICAgICAgICBgPGlucHV0IGNs"
    "YXNzPSJpbnAgbW9ubyIgaWQ9InNldENvZGVyU3RlcHMiIHZhbHVlPSIke2VzYyhzLmNvZGVyX21heF9zdGVwc3x8IjI1Iil9IiBzdHlsZT0id2lkdGg6"
    "OTBweCI+YCl9CiAgICAgIDxwIGNsYXNzPSJoaW50IiBzdHlsZT0ibWFyZ2luOjZweCAwIDAiPlJ1bm5pbmcgY29tbWFuZHMgYW5kIHRlc3RzIGluIHRo"
    "ZSBwcm9qZWN0IHJldXNlcyB0aGUKICAgICAgICAiQWxsb3cgY29kZSBleGVjdXRpb24iIHN3aXRjaCBhYm92ZS4gVGlwOiBrZWVwIHRoZSBwcm9qZWN0"
    "IGluIGdpdCBzbyBhbnkgZWRpdCBpcyBlYXN5IHRvIHJldmlldyBhbmQgdW5kby48L3A+CiAgICA8L2Rpdj4KICAgIDxkaXYgY2xhc3M9ImNhcmQgcGFk"
    "LWxnIiBzdHlsZT0ibWFyZ2luLWJvdHRvbToxNnB4Ij4KICAgICAgPGRpdiBjbGFzcz0ic2VjdGlvbi10aXRsZSIgc3R5bGU9Im1hcmdpbi10b3A6MCI+"
    "UmVtb3RlIGFjY2VzczwvZGl2PgogICAgICAke3NldHRpbmdSb3coIkFjY2VzcyBwYXNzd29yZCIsIlJlcXVpcmVkIGZyb20gZXZlcnkgZGV2aWNlIHRo"
    "YXQgaXMgbm90IHRoaXMgbWFjaGluZSAocGhvbmVzLCBsYXB0b3BzIG9uIHlvdXIgV2ktRmksIFRhaWxzY2FsZSkuIExlYXZlIGVtcHR5IHRvIHN3aXRj"
    "aCBwcm90ZWN0aW9uIG9mZi4gRGV2aWNlcyBzdGF5IHVubG9ja2VkIGZvciAzMCBkYXlzOyByZXN0YXJ0aW5nIHRoZSBzZXJ2ZXIgbG9ja3MgdGhlbSBh"
    "Z2Fpbi4iLAogICAgICAgIGA8aW5wdXQgY2xhc3M9ImlucCBtb25vIiBpZD0ic2V0TGFuUGFzcyIgdmFsdWU9IiR7ZXNjKHMubGFuX3Bhc3N3b3JkfHwi"
    "Iil9IiBwbGFjZWhvbGRlcj0iZW1wdHkgPSBubyBwYXNzd29yZCIgc3R5bGU9Im1pbi13aWR0aDoyMjBweCI+YCl9CiAgICAgIDxwIGNsYXNzPSJoaW50"
    "IiBzdHlsZT0ibWFyZ2luOjZweCAwIDAiPlRvIHVzZSBIZW9ydGggZnJvbSB5b3VyIHBob25lLCBzdGFydCBpdCB3aXRoCiAgICAgICAgPHNwYW4gY2xh"
    "c3M9Im1vbm8iPi0taG9zdCAwLjAuMC4wPC9zcGFuPiDigJQgdGhlIHRlcm1pbmFsIHRoZW4gcHJpbnRzIHRoZSBhZGRyZXNzIHRvIG9wZW4uCiAgICAg"
    "ICAgUmVxdWVzdHMgZnJvbSB0aGlzIG1hY2hpbmUgaXRzZWxmIG5ldmVyIG5lZWQgdGhlIHBhc3N3b3JkLjwvcD4KICAgIDwvZGl2PgogICAgPGRpdiBj"
    "bGFzcz0iZGFuZ2VyLWNhcmQiIHN0eWxlPSJtYXJnaW4tYm90dG9tOjE2cHgiPgogICAgICA8ZGl2IGNsYXNzPSJzZWN0aW9uLXRpdGxlIiBzdHlsZT0i"
    "bWFyZ2luLXRvcDowO2NvbG9yOnZhcigtLXJlZCkiPkNvbXB1dGVyIGNvbnRyb2w8L2Rpdj4KICAgICAgJHtzZXR0aW5nUm93KCJFbmFibGUgY29tcHV0"
    "ZXIgY29udHJvbCIsIk1hc3RlciBzd2l0Y2guIExldHMgYSBtb2RlbCBzZWUgdGhlIHNjcmVlbiBhbmQgZHJpdmUgdGhlIG1vdXNlL2tleWJvYXJkLiBU"
    "dXJuIG9uIHZpYSB0aGUgQ29tcHV0ZXIgcGFnZSwgd2hpY2ggZXhwbGFpbnMgdGhlIHJpc2tzLiIsCiAgICAgICAgc3coImNvbXB1dGVyX2NvbnRyb2wi"
    "LCBzLmNvbXB1dGVyX2NvbnRyb2w9PT0iMSIpKX0KICAgICAgJHtzZXR0aW5nUm93KCJBc2sgYmVmb3JlIGVhY2ggYWN0aW9uIiwiU3Ryb25nbHkgcmVj"
    "b21tZW5kZWQuIFlvdSBhcHByb3ZlIGV2ZXJ5IGNsaWNrIGFuZCBrZXlzdHJva2UgYmVmb3JlIGl0IGhhcHBlbnMuIiwKICAgICAgICBzdygiY29tcHV0"
    "ZXJfY29uZmlybSIsIHMuY29tcHV0ZXJfY29uZmlybT09PSIxIikpfQogICAgICAke3NldHRpbmdSb3coIlBhdXNlIGJldHdlZW4gYWN0aW9ucyIsIlNl"
    "Y29uZHMgSGVvcnRoIHdhaXRzIGFmdGVyIGVhY2ggYWN0aW9uLCBnaXZpbmcgeW91IHRpbWUgdG8gcmVhY3QgKGFuZCB0byByZWFjaCBhIHNjcmVlbiBj"
    "b3JuZXIgdG8gYWJvcnQpLiIsCiAgICAgICAgYDxpbnB1dCBjbGFzcz0iaW5wIG1vbm8iIGlkPSJzZXRDY1BhdXNlIiB2YWx1ZT0iJHtlc2Mocy5jb21w"
    "dXRlcl9wYXVzZSl9IiBzdHlsZT0id2lkdGg6OTBweCI+YCl9CiAgICA8L2Rpdj4KICAgIDxkaXYgY2xhc3M9ImNhcmQgcGFkLWxnIiBzdHlsZT0ibWFy"
    "Z2luLWJvdHRvbToxNnB4Ij4KICAgICAgPGRpdiBjbGFzcz0ic2VjdGlvbi10aXRsZSIgc3R5bGU9Im1hcmdpbi10b3A6MCI+TW9kZWxzICZhbXA7IHJl"
    "dHJpZXZhbDwvZGl2PgogICAgICAke3NldHRpbmdSb3coIk9sbGFtYSBhZGRyZXNzIiwiV2hlcmUgT2xsYW1hIGlzIGxpc3RlbmluZy4iLAogICAgICAg"
    "IGA8aW5wdXQgY2xhc3M9ImlucCBtb25vIiBpZD0ic2V0T2xsYW1hIiB2YWx1ZT0iJHtlc2Mocy5vbGxhbWFfaG9zdCl9IiBzdHlsZT0ibWluLXdpZHRo"
    "OjIwMHB4Ij5gKX0KICAgICAgJHtzZXR0aW5nUm93KCJFbWJlZGRpbmcgbW9kZWwiLCJVc2VkIHRvIGluZGV4IHlvdXIga25vd2xlZGdlIGJhc2UuIiwK"
    "ICAgICAgICBgPGlucHV0IGNsYXNzPSJpbnAgbW9ubyIgaWQ9InNldEVtYmVkIiB2YWx1ZT0iJHtlc2Mocy5lbWJlZF9tb2RlbCl9IiBzdHlsZT0ibWlu"
    "LXdpZHRoOjE4MHB4Ij5gKX0KICAgICAgJHtzZXR0aW5nUm93KCJLbm93bGVkZ2UgcmVzdWx0cyIsIkhvdyBtYW55IGRvY3VtZW50IGNodW5rcyB0byBy"
    "ZXRyaWV2ZSBwZXIgcXVlc3Rpb24uIiwKICAgICAgICBgPGlucHV0IGNsYXNzPSJpbnAgbW9ubyIgaWQ9InNldFRvcEsiIHZhbHVlPSIke2VzYyhzLnJh"
    "Z190b3Bfayl9IiBzdHlsZT0id2lkdGg6OTBweCI+YCl9CiAgICA8L2Rpdj4KICAgIDxkaXYgY2xhc3M9ImNhcmQgcGFkLWxnIiBzdHlsZT0ibWFyZ2lu"
    "LWJvdHRvbToxNnB4Ij4KICAgICAgPGRpdiBjbGFzcz0ic2VjdGlvbi10aXRsZSIgc3R5bGU9Im1hcmdpbi10b3A6MCI+V2ViIHNlYXJjaDwvZGl2Pgog"
    "ICAgICAke3NldHRpbmdSb3coIlNlYXJjaCBiYWNrZW5kIiwiQXV0byB1c2VzIFNlYXJYTkcgd2hlbmV2ZXIgaXQncyBydW5uaW5nIGFuZCBxdWlldGx5"
    "IGZhbGxzIGJhY2sgdG8gRHVja0R1Y2tHby4iLAogICAgICAgIGA8c2VsZWN0IGNsYXNzPSJzZWwiIGlkPSJzZXRTZWFyY2hCYWNrZW5kIj4KICAgICAg"
    "ICAgICA8b3B0aW9uIHZhbHVlPSJhdXRvIj5BdXRvIChyZWNvbW1lbmRlZCk8L29wdGlvbj4KICAgICAgICAgICA8b3B0aW9uIHZhbHVlPSJzZWFyeG5n"
    "Ij5TZWFyWE5HIG9ubHk8L29wdGlvbj4KICAgICAgICAgICA8b3B0aW9uIHZhbHVlPSJkdWNrZHVja2dvIj5EdWNrRHVja0dvIG9ubHk8L29wdGlvbj48"
    "L3NlbGVjdD5gKX0KICAgICAgJHtzZXR0aW5nUm93KCJTZWFyWE5HIGFkZHJlc3MiLCJXaGVyZSB5b3VyIFNlYXJYTkcgaW5zdGFuY2UgbGlzdGVucy4i"
    "LAogICAgICAgIGA8aW5wdXQgY2xhc3M9ImlucCBtb25vIiBpZD0ic2V0U2VhcnhVcmwiIHZhbHVlPSIke2VzYyhzLnNlYXJ4bmdfdXJsKX0iIHN0eWxl"
    "PSJtaW4td2lkdGg6MjAwcHgiPmApfQogICAgICA8cCBjbGFzcz0iaGludCIgc3R5bGU9Im1hcmdpbjo2cHggMCAwIj5TZXQgdXAgU2VhclhORyB3aXRo"
    "IG9uZSBjbGljayBvbiB0aGUKICAgICAgICBBZ2VudCAmYW1wOyBUb29scyBwYWdlIOKAlCBpdCBuZWVkcyBEb2NrZXIuPC9wPgogICAgICA8ZGl2IGNs"
    "YXNzPSJzZXR0aW5ncm93IiBzdHlsZT0iYm9yZGVyLXRvcDoxcHggc29saWQgdmFyKC0tbGluZS1zb2Z0KTttYXJnaW4tdG9wOjEycHgiPgogICAgICAg"
    "IDxkaXYgY2xhc3M9ImxhYiI+U2F2ZSBjaGFuZ2VzPC9kaXY+CiAgICAgICAgPGRpdiBjbGFzcz0iY3RsIj48YnV0dG9uIGNsYXNzPSJidG4gcHJpbWFy"
    "eSIgaWQ9InNhdmVTZXR0aW5nc0J0biI+U2F2ZSBzZXR0aW5nczwvYnV0dG9uPjwvZGl2PjwvZGl2PgogICAgPC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJj"
    "YXJkIHBhZC1sZyIgc3R5bGU9Im1hcmdpbi1ib3R0b206MTZweCIgaWQ9InVwZGF0ZUNhcmQiPjwvZGl2PgogICAgPGRpdiBjbGFzcz0iY2FyZCBwYWQt"
    "bGciPgogICAgICA8ZGl2IGNsYXNzPSJzZWN0aW9uLXRpdGxlIiBzdHlsZT0ibWFyZ2luLXRvcDowIj5Zb3VyIGRhdGE8L2Rpdj4KICAgICAgPHAgY2xh"
    "c3M9Im11dGVkIiBzdHlsZT0ibWFyZ2luOjAgMCA4cHgiPkV2ZXJ5dGhpbmcgSGVvcnRoIHN0b3JlcyDigJQgY29udmVyc2F0aW9ucywgZG9jdW1lbnRz"
    "LAogICAgICAgIGltYWdlcywgbW9kZWxzIGxpc3QsIGJhY2t1cHMg4oCUIGxpdmVzIGluIG9uZSBmb2xkZXIgb24geW91ciBjb21wdXRlcjo8L3A+CiAg"
    "ICAgIDxkaXYgY2xhc3M9Im1vbm8iIHN0eWxlPSJmb250LXNpemU6MTJweDtiYWNrZ3JvdW5kOnZhcigtLWluayk7Ym9yZGVyOjFweCBzb2xpZCB2YXIo"
    "LS1saW5lKTsKICAgICAgICBib3JkZXItcmFkaXVzOjhweDtwYWRkaW5nOjExcHggMTNweDt3b3JkLWJyZWFrOmJyZWFrLWFsbCIgaWQ9ImRhdGFEaXIi"
    "PuKApjwvZGl2PgogICAgICA8cCBjbGFzcz0iaGludCIgc3R5bGU9Im1hcmdpbi10b3A6MTBweCI+RGVsZXRlIHRoYXQgZm9sZGVyIHRvIHJlc2V0IEhl"
    "b3J0aCBjb21wbGV0ZWx5LgogICAgICAgIE1vZGVsIGZpbGVzIHRoZW1zZWx2ZXMgYXJlIG1hbmFnZWQgYnkgT2xsYW1hLjwvcD4KICAgIDwvZGl2PmA7"
    "CgogICQoInNldFNlYXJjaEJhY2tlbmQiKS52YWx1ZSA9IHMuc2VhcmNoX2JhY2tlbmQgfHwgImF1dG8iOwoKICAvLyBzd2l0Y2hlcwogIGhvc3QucXVl"
    "cnlTZWxlY3RvckFsbCgiLnN3aXRjaCIpLmZvckVhY2goZWwyPT57CiAgICBlbDIub25jbGljayA9IGFzeW5jKCk9PnsKICAgICAgY29uc3Qga2V5ID0g"
    "ZWwyLmRhdGFzZXQua2V5OyBjb25zdCBvbiA9ICFlbDIuY2xhc3NMaXN0LmNvbnRhaW5zKCJvbiIpOwogICAgICBlbDIuY2xhc3NMaXN0LnRvZ2dsZSgi"
    "b24iLCBvbik7CiAgICAgIGlmKGtleT09PSJjb21wdXRlcl9jb250cm9sIiAmJiBvbil7CiAgICAgICAgZWwyLmNsYXNzTGlzdC5yZW1vdmUoIm9uIik7"
    "CiAgICAgICAgc2hvdygiY29tcHV0ZXIiKTsKICAgICAgICB0b2FzdCgiUmV2aWV3IHRoZSBzYWZldHkgbm90aWNlIG9uIHRoZSBDb21wdXRlciBwYWdl"
    "IHRvIGVuYWJsZSB0aGlzIik7CiAgICAgICAgcmV0dXJuOwogICAgICB9CiAgICAgIGlmKGtleT09PSJhbGxvd19jb2RlX2V4ZWN1dGlvbiIgJiYgb24p"
    "ewogICAgICAgIC8vIGNvbmZpcm0gZGFuZ2Vyb3VzIHRvZ2dsZQogICAgICAgIGVsMi5jbGFzc0xpc3QucmVtb3ZlKCJvbiIpOwogICAgICAgIG1vZGFs"
    "KHt0aXRsZToiQWxsb3cgY29kZSBleGVjdXRpb24/IiwKICAgICAgICAgIGJvZHlIVE1MOmA8cCBjbGFzcz0ibXV0ZWQiPlRoaXMgbGV0cyB0aGUgbW9k"
    "ZWwgcnVuIFB5dGhvbiBhbmQgc2hlbGwgY29tbWFuZHMgb24geW91cgogICAgICAgICAgICBjb21wdXRlciAoaW5zaWRlIGl0cyB3b3Jrc3BhY2UgZm9s"
    "ZGVyKS4gT25seSBlbmFibGUgdGhpcyBpZiB5b3UgdW5kZXJzdGFuZCBhbmQgdHJ1c3QKICAgICAgICAgICAgd2hhdCB5b3UncmUgcnVubmluZy48L3A+"
    "YCwKICAgICAgICAgIGFjdGlvbnM6W3tsYWJlbDoiQ2FuY2VsIiwgb25DbGljazpjbG9zZU1vZGFsfSwKICAgICAgICAgICAge2xhYmVsOiJFbmFibGUi"
    "LCBjbHM6ImRhbmdlciIsIG9uQ2xpY2s6YXN5bmMoKT0+eyBjbG9zZU1vZGFsKCk7IGVsMi5jbGFzc0xpc3QuYWRkKCJvbiIpOwogICAgICAgICAgICAg"
    "IGF3YWl0IHBvc3QoIi9hcGkvc2V0dGluZ3MiLCB7YWxsb3dfY29kZV9leGVjdXRpb246IjEifSk7IHRvYXN0KCJDb2RlIGV4ZWN1dGlvbiBlbmFibGVk"
    "Iiwib2siKTsgfX1dfSk7CiAgICAgICAgcmV0dXJuOwogICAgICB9CiAgICAgIGF3YWl0IHBvc3QoIi9hcGkvc2V0dGluZ3MiLCB7W2tleV06IG9uPyIx"
    "IjoiMCJ9KTsKICAgICAgdG9hc3QoIlNhdmVkIiwib2siKTsKICAgIH07CiAgfSk7CiAgJCgic2F2ZVNldHRpbmdzQnRuIikub25jbGljayA9IGFzeW5j"
    "KCk9PnsKICAgIHRyeXsKICAgICAgYXdhaXQgcG9zdCgiL2FwaS9zZXR0aW5ncyIsIHsKICAgICAgICBzeXN0ZW1fcHJvbXB0OiQoInNldFN5c1Byb21w"
    "dCIpLnZhbHVlLAogICAgICAgIGNvbnRleHRfbWVzc2FnZXM6JCgic2V0Q29udGV4dCIpLnZhbHVlLAogICAgICAgIGFnZW50X21heF9zdGVwczokKCJz"
    "ZXRTdGVwcyIpLnZhbHVlLAogICAgICAgIGxvb3BfbWF4X3N0ZXBzOiQoInNldExvb3BTdGVwcyIpLnZhbHVlLAogICAgICAgIGNvdW5jaWxfc2l6ZTok"
    "KCJzZXRDb3VuY2lsU2l6ZSIpLnZhbHVlLAogICAgICAgIGNvdW5jaWxfcm91bmRzOiQoInNldENvdW5jaWxSb3VuZHMiKS52YWx1ZSwKICAgICAgICBj"
    "b2Rlcl9yb290OiQoInNldENvZGVyUm9vdCIpLnZhbHVlLAogICAgICAgIGxhbl9wYXNzd29yZDokKCJzZXRMYW5QYXNzIikudmFsdWUsCiAgICAgICAg"
    "Y29kZXJfbWF4X3N0ZXBzOiQoInNldENvZGVyU3RlcHMiKS52YWx1ZSwKICAgICAgICBjb21wdXRlcl9wYXVzZTokKCJzZXRDY1BhdXNlIikudmFsdWUs"
    "CiAgICAgICAgb2xsYW1hX2hvc3Q6JCgic2V0T2xsYW1hIikudmFsdWUsCiAgICAgICAgZW1iZWRfbW9kZWw6JCgic2V0RW1iZWQiKS52YWx1ZSwKICAg"
    "ICAgICByYWdfdG9wX2s6JCgic2V0VG9wSyIpLnZhbHVlLAogICAgICAgIHNlYXJjaF9iYWNrZW5kOiQoInNldFNlYXJjaEJhY2tlbmQiKS52YWx1ZSwK"
    "ICAgICAgICBzZWFyeG5nX3VybDokKCJzZXRTZWFyeFVybCIpLnZhbHVlIH0pOwogICAgICB0cnl7IGNvbnN0IHIyID0gYXdhaXQgYXBpKCIvYXBpL3Nl"
    "dHRpbmdzIik7IHN0YXRlLnNldHRpbmdzID0gcjIuc2V0dGluZ3M7IH1jYXRjaChlKXt9CiAgICAgIHRvYXN0KCJTZXR0aW5ncyBzYXZlZCIsIm9rIik7"
    "CiAgICB9Y2F0Y2goZSl7IHRvYXN0KGUubWVzc2FnZSwiZXJyIik7IH0KICB9OwogIHRyeXsgY29uc3Qgc3lzZCA9IGF3YWl0IGFwaSgiL2FwaS9zeXN0"
    "ZW0iKTsgJCgiZGF0YURpciIpLnRleHRDb250ZW50ID0gc3lzZC5kYXRhX2RpcjsgfWNhdGNoKGUpe30KICByZW5kZXJVcGRhdGVDYXJkKCk7Cn0KCi8q"
    "IC0tLS0tLS0tLS0gc2VsZi11cGRhdGUgLS0tLS0tLS0tLSAqLwphc3luYyBmdW5jdGlvbiByZW5kZXJVcGRhdGVDYXJkKCl7CiAgY29uc3QgY2FyZCA9"
    "ICQoInVwZGF0ZUNhcmQiKTsgaWYoIWNhcmQpIHJldHVybjsKICBjYXJkLmlubmVySFRNTCA9IGA8ZGl2IGNsYXNzPSJzZWN0aW9uLXRpdGxlIiBzdHls"
    "ZT0ibWFyZ2luLXRvcDowIj5VcGRhdGVzPC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJyb3ciIHN0eWxlPSJnYXA6OXB4Ij48ZGl2IGNsYXNzPSJzcGluIj48"
    "L2Rpdj4KICAgIDxzcGFuIGNsYXNzPSJtdXRlZCI+Q2hlY2tpbmcgZm9yIGEgbmV3ZXIgdmVyc2lvbuKApjwvc3Bhbj48L2Rpdj5gOwogIGxldCBpbmZv"
    "OwogIHRyeXsgaW5mbyA9IGF3YWl0IGFwaSgiL2FwaS91cGRhdGUvY2hlY2siKTsgfQogIGNhdGNoKGUpeyBjYXJkLmlubmVySFRNTCA9ICc8ZGl2IGNs"
    "YXNzPSJzZWN0aW9uLXRpdGxlIiBzdHlsZT0ibWFyZ2luLXRvcDowIj5VcGRhdGVzPC9kaXY+JysKICAgIGVyckNhcmQoZS5tZXNzYWdlKTsgcmV0dXJu"
    "OyB9CiAgc3RhdGUudXBkYXRlSW5mbyA9IGluZm87CiAgY29uc3Qgd2F0Y2hlZCA9IChpbmZvLndhdGNoZWR8fFtdKS5tYXAodz0+YDxzcGFuIGNsYXNz"
    "PSJtb25vIj4ke2VzYyh3KX08L3NwYW4+YCkuam9pbigiPGJyPiIpOwogIGxldCBpbm5lciA9IGA8ZGl2IGNsYXNzPSJzZWN0aW9uLXRpdGxlIiBzdHls"
    "ZT0ibWFyZ2luLXRvcDowIj5VcGRhdGVzPC9kaXY+YDsKICBpZihpbmZvLmF2YWlsYWJsZSl7CiAgICBjb25zdCB1ID0gaW5mby51cGRhdGU7CiAgICBp"
    "bm5lciArPSBgPGRpdiBjbGFzcz0idXBkYXRlYmFubmVyIiBzdHlsZT0ibWFyZ2luLWJvdHRvbToxNnB4Ij4KICAgICAgPGRpdiBjbGFzcz0iaWMiPgog"
    "ICAgICAgIDxzdmcgd2lkdGg9IjIwIiBoZWlnaHQ9IjIwIiB2aWV3Qm94PSIwIDAgMjQgMjQiIGZpbGw9Im5vbmUiIHN0cm9rZT0iY3VycmVudENvbG9y"
    "IiBzdHJva2Utd2lkdGg9IjIiPjxwYXRoIGQ9Ik0yMSAxMmE5IDkgMCAxIDEtMi42LTYuNE0yMSAzdjVoLTUiLz48L3N2Zz48L2Rpdj4KICAgICAgPGRp"
    "diBjbGFzcz0idWkiPjxkaXYgY2xhc3M9InQiPlZlcnNpb24gJHtlc2ModS52ZXJzaW9uKX0gaXMgcmVhZHkgdG8gaW5zdGFsbDwvZGl2PgogICAgICAg"
    "IDxkaXYgY2xhc3M9ImQiPkZvdW5kIDxzcGFuIGNsYXNzPSJtb25vIj4ke2VzYyh1LnBhdGguc3BsaXQoIi8iKS5wb3AoKSl9PC9zcGFuPgogICAgICAg"
    "ICAgKCR7dS5zaXplX2tifSBLQikuIFlvdSdyZSBvbiB2JHtlc2MoaW5mby5jdXJyZW50KX0uPC9kaXY+PC9kaXY+CiAgICAgIDxidXR0b24gY2xhc3M9"
    "ImJ0biBwcmltYXJ5IiBpZD0iYXBwbHlVcGRhdGVCdG4iPlVwZGF0ZSAmYW1wOyByZXN0YXJ0PC9idXR0b24+PC9kaXY+YDsKICB9IGVsc2UgewogICAg"
    "aW5uZXIgKz0gYDxkaXYgY2xhc3M9InJvdyIgc3R5bGU9ImdhcDo5cHg7bWFyZ2luLWJvdHRvbToxNHB4Ij4KICAgICAgPHNwYW4gY2xhc3M9ImRvdCBv"
    "biI+PC9zcGFuPjxzcGFuIGNsYXNzPSJtdXRlZCI+WW91J3JlIG9uIHRoZSBsYXRlc3QgdmVyc2lvbgogICAgICAgICg8c3BhbiBjbGFzcz0ibW9ubyI+"
    "diR7ZXNjKGluZm8uY3VycmVudCl9PC9zcGFuPikuIE5vIG5ld2VyIGZpbGUgZm91bmQuPC9zcGFuPjwvZGl2PmA7CiAgfQogIGlubmVyICs9IGA8cCBj"
    "bGFzcz0iaGludCIgc3R5bGU9Im1hcmdpbjowIDAgMTJweCI+SGVvcnRoIHdhdGNoZXMgdGhlc2UgbG9jYXRpb25zIGZvciBhIG5ld2VyCiAgICA8c3Bh"
    "biBjbGFzcz0ibW9ubyI+aGVvcnRoKi5weTwvc3Bhbj4gKG9yIDxzcGFuIGNsYXNzPSJtb25vIj5sb2NhbG1pbmQqLnB5PC9zcGFuPikgZmlsZTo8YnI+"
    "JHt3YXRjaGVkfTxicj4KICAgIERyb3AgYSBuZXdlciBmaWxlIGluIGFueSBvZiB0aGVtLCBvciB1cGxvYWQgb25lIGJlbG93LjwvcD4KICAgIDxkaXYg"
    "Y2xhc3M9InJvdyIgc3R5bGU9ImdhcDoxMHB4Ij4KICAgICAgPGJ1dHRvbiBjbGFzcz0iYnRuIHNtIGdob3N0IiBpZD0iY2hlY2tVcGRhdGVCdG4iPkNo"
    "ZWNrIGFnYWluPC9idXR0b24+CiAgICAgIDxidXR0b24gY2xhc3M9ImJ0biBzbSBnaG9zdCIgaWQ9InVwbG9hZFVwZGF0ZUJ0biI+VXBsb2FkIHVwZGF0"
    "ZSBmaWxl4oCmPC9idXR0b24+CiAgICAgIDxpbnB1dCB0eXBlPSJmaWxlIiBpZD0idXBkYXRlRmlsZUlucHV0IiBhY2NlcHQ9Ii5weSIgaGlkZGVuPgog"
    "ICAgPC9kaXY+YDsKICBjYXJkLmlubmVySFRNTCA9IGlubmVyOwoKICBjb25zdCBhcHBseSA9ICQoImFwcGx5VXBkYXRlQnRuIik7CiAgaWYoYXBwbHkp"
    "IGFwcGx5Lm9uY2xpY2sgPSAoKT0+YXBwbHlVcGRhdGUoc3RhdGUudXBkYXRlSW5mby51cGRhdGUucGF0aCk7CiAgJCgiY2hlY2tVcGRhdGVCdG4iKS5v"
    "bmNsaWNrID0gcmVuZGVyVXBkYXRlQ2FyZDsKICAkKCJ1cGxvYWRVcGRhdGVCdG4iKS5vbmNsaWNrID0gKCk9PiQoInVwZGF0ZUZpbGVJbnB1dCIpLmNs"
    "aWNrKCk7CiAgJCgidXBkYXRlRmlsZUlucHV0Iikub25jaGFuZ2UgPSBhc3luYyhlKT0+ewogICAgY29uc3QgZiA9IGUudGFyZ2V0LmZpbGVzWzBdOyBp"
    "ZighZikgcmV0dXJuOwogICAgY29uc3QgZmQgPSBuZXcgRm9ybURhdGEoKTsgZmQuYXBwZW5kKCJmaWxlIiwgZik7CiAgICB0b2FzdCgiVXBsb2FkaW5n"
    "ICIrZi5uYW1lKyLigKYiKTsKICAgIHRyeXsKICAgICAgY29uc3QgciA9IGF3YWl0IGZldGNoKCIvYXBpL3VwZGF0ZS91cGxvYWQiLCB7bWV0aG9kOiJQ"
    "T1NUIiwgYm9keTpmZH0pOwogICAgICBjb25zdCBqID0gYXdhaXQgci5qc29uKCk7CiAgICAgIGlmKGoub2speyB0b2FzdCgiVXBkYXRlIHYiK2oudmVy"
    "c2lvbisiIHJlYWR5Iiwib2siKTsgcmVuZGVyVXBkYXRlQ2FyZCgpOyBwb2xsVXBkYXRlQmFkZ2UoKTsgfQogICAgICBlbHNlIHRvYXN0KGouZXJyb3J8"
    "fCJVcGxvYWQgcmVqZWN0ZWQiLCJlcnIiKTsKICAgIH1jYXRjaChlcnIpeyB0b2FzdChlcnIubWVzc2FnZSwiZXJyIik7IH0KICAgIGUudGFyZ2V0LnZh"
    "bHVlPSIiOwogIH07Cn0KYXN5bmMgZnVuY3Rpb24gYXBwbHlVcGRhdGUocGF0aCl7CiAgY29uc3QgYm9keSA9IG1vZGFsKHt0aXRsZToiVXBkYXRlIGFu"
    "ZCByZXN0YXJ0PyIsCiAgICBib2R5SFRNTDpgPHAgY2xhc3M9Im11dGVkIj5IZW9ydGggd2lsbCBiYWNrIHVwIHRoZSBjdXJyZW50IHZlcnNpb24sIHJl"
    "cGxhY2UgaXQgd2l0aCB0aGUKICAgICAgbmV3IGZpbGUsIGFuZCByZXN0YXJ0IHRoZSBzZXJ2ZXIuIFRoaXMgcGFnZSB3aWxsIHJlY29ubmVjdCBhdXRv"
    "bWF0aWNhbGx5IGluIGEgZmV3IHNlY29uZHMuPC9wPmAsCiAgICBhY3Rpb25zOlt7bGFiZWw6IkNhbmNlbCIsIG9uQ2xpY2s6Y2xvc2VNb2RhbH0sCiAg"
    "ICAgIHtsYWJlbDoiVXBkYXRlIG5vdyIsIGNsczoicHJpbWFyeSIsIG9uQ2xpY2s6YXN5bmMoKT0+ewogICAgICAgIGJvZHkuaW5uZXJIVE1MID0gJzxk"
    "aXYgY2xhc3M9InJvdyIgc3R5bGU9ImdhcDoxMHB4Ij48ZGl2IGNsYXNzPSJzcGluIj48L2Rpdj4nKwogICAgICAgICAgJzxzcGFuPkFwcGx5aW5nIHVw"
    "ZGF0ZSBhbmQgcmVzdGFydGluZ+KApjwvc3Bhbj48L2Rpdj4nOwogICAgICAgIHRyeXsKICAgICAgICAgIGNvbnN0IHIgPSBhd2FpdCBwb3N0KCIvYXBp"
    "L3VwZGF0ZS9hcHBseSIsIHtwYXRofSk7CiAgICAgICAgICBpZihyLm9rKXsgd2FpdEZvclJlc3RhcnQoci5uZXdfdmVyc2lvbik7IH0KICAgICAgICAg"
    "IGVsc2UgeyB0b2FzdChyLmVycm9yfHwiVXBkYXRlIGZhaWxlZCIsImVyciIpOyBjbG9zZU1vZGFsKCk7IH0KICAgICAgICB9Y2F0Y2goZSl7CiAgICAg"
    "ICAgICAvLyB0aGUgc2VydmVyIG1heSBoYXZlIHJlc3RhcnRlZCBtaWQtcmVxdWVzdCDigJQgdHJlYXQgYXMgc3VjY2VzcyBhbmQgd2FpdAogICAgICAg"
    "ICAgd2FpdEZvclJlc3RhcnQoKTsKICAgICAgICB9CiAgICAgIH19XX0pOwp9CmZ1bmN0aW9uIHdhaXRGb3JSZXN0YXJ0KG5ld1Zlcil7CiAgY29uc3Qg"
    "Ym9keSA9IG1vZGFsKHt0aXRsZToiVXBkYXRpbmfigKYiLCBib2R5SFRNTDpgPGRpdiBzdHlsZT0idGV4dC1hbGlnbjpjZW50ZXI7cGFkZGluZzoxNHB4"
    "Ij4KICAgIDxkaXYgY2xhc3M9InNwaW4iIHN0eWxlPSJ3aWR0aDoyNnB4O2hlaWdodDoyNnB4O21hcmdpbjowIGF1dG8gMTRweCI+PC9kaXY+CiAgICA8"
    "cCBjbGFzcz0ibXV0ZWQiIGlkPSJyZXN0YXJ0TXNnIj5SZXN0YXJ0aW5nIGludG8gJHtuZXdWZXI/InZlcnNpb24gIitlc2MobmV3VmVyKToidGhlIG5l"
    "dyB2ZXJzaW9uIn0uCiAgICAgIFJlY29ubmVjdGluZyBhdXRvbWF0aWNhbGx54oCmPC9wPjwvZGl2PmB9KTsKICBsZXQgdHJpZXM9MDsKICBjb25zdCBw"
    "b2xsID0gKCk9PnsgY29uc3QgdCA9IHNldEludGVydmFsKGFzeW5jKCk9PnsKICAgIHRyaWVzKys7CiAgICB0cnl7CiAgICAgIGNvbnN0IHIgPSBhd2Fp"
    "dCBmZXRjaCgiL2FwaS9oZWFsdGgiLCB7Y2FjaGU6Im5vLXN0b3JlIn0pOwogICAgICBpZihyLm9rKXsgY29uc3QgaiA9IGF3YWl0IHIuanNvbigpOwog"
    "ICAgICAgIC8vIG9ubHkgY2VsZWJyYXRlIG9uY2UgdGhlIE5FVyB2ZXJzaW9uIGFuc3dlcnMg4oCUIGFuIG9sZCBzZXJ2ZXIgdGhhdAogICAgICAgIC8v"
    "IG5ldmVyIHJlc3RhcnRlZCBtdXN0IG5vdCBsb29rIGxpa2UgYSBzdWNjZXNzZnVsIHVwZGF0ZQogICAgICAgIGlmKCFuZXdWZXIgfHwgai52ZXJzaW9u"
    "ID09PSBuZXdWZXIpewogICAgICAgICAgY2xlYXJJbnRlcnZhbCh0KTsKICAgICAgICAgIGNsb3NlTW9kYWwoKTsgdG9hc3QoIlVwZGF0ZWQgdG8gdiIr"
    "ai52ZXJzaW9uKyIg4oCUIHJlbG9hZGluZyIsIm9rIik7CiAgICAgICAgICBzZXRUaW1lb3V0KCgpPT5sb2NhdGlvbi5yZWxvYWQoKSwgODAwKTsKICAg"
    "ICAgICB9CiAgICAgIH0KICAgIH1jYXRjaChlKXsgLyogc2VydmVyIHN0aWxsIGRvd24sIGtlZXAgd2FpdGluZyAqLyB9CiAgICBpZih0cmllcz43NSl7"
    "IGNsZWFySW50ZXJ2YWwodCk7CiAgICAgIGNvbnN0IG0gPSAkKCJyZXN0YXJ0TXNnIik7CiAgICAgIGlmKG0pIG0uaW5uZXJIVE1MID0gYFN0aWxsIGNh"
    "bid0IHJlYWNoIHRoZSBzZXJ2ZXIgb24gPGI+JHtlc2MobG9jYXRpb24uaG9zdCl9PC9iPi48YnI+PGJyPgogICAgICAgIENoZWNrIHRoZSB0ZXJtaW5h"
    "bCB3aW5kb3cgd2hlcmUgSGVvcnRoIHJ1bnMg4oCUIGlmIGl0IHJlc3RhcnRlZCBvbiBhCiAgICAgICAgPGI+ZGlmZmVyZW50IHBvcnQ8L2I+IGl0IHBy"
    "aW50cyB0aGUgbmV3IGFkZHJlc3MgdGhlcmUuIERldGFpbHMgYXJlIGFsc28KICAgICAgICBsb2dnZWQgdG8gPHNwYW4gY2xhc3M9Im1vbm8iPmhlb3J0"
    "aF9kYXRhL3Jlc3RhcnQubG9nPC9zcGFuPi48YnI+PGJyPgogICAgICAgIDxidXR0b24gY2xhc3M9ImJ0biBzbSIgaWQ9ImtlZXBXYWl0aW5nQnRuIj5L"
    "ZWVwIHdhaXRpbmc8L2J1dHRvbj5gOwogICAgICBjb25zdCBiID0gJCgia2VlcFdhaXRpbmdCdG4iKTsKICAgICAgaWYoYikgYi5vbmNsaWNrID0gKCk9"
    "PnsgdHJpZXMgPSAwOwogICAgICAgIGlmKG0pIG0uaW5uZXJIVE1MID0gIlJlY29ubmVjdGluZyBhdXRvbWF0aWNhbGx54oCmIjsgcG9sbCgpOyB9Owog"
    "ICAgfQogIH0sIDgwMCk7IH07CiAgcG9sbCgpOwp9CmxvYWRlcnMuc2V0dGluZ3MgPSBsb2FkU2V0dGluZ3M7CgovKiAtLS0tLS0tLS0tIHVwZGF0ZSBi"
    "YWRnZSBwb2xsaW5nIChnbG9iYWwpIC0tLS0tLS0tLS0gKi8KYXN5bmMgZnVuY3Rpb24gcG9sbFVwZGF0ZUJhZGdlKCl7CiAgdHJ5ewogICAgY29uc3Qg"
    "aW5mbyA9IGF3YWl0IGFwaSgiL2FwaS91cGRhdGUvY2hlY2siKTsKICAgIHN0YXRlLnVwZGF0ZUluZm8gPSBpbmZvOwogICAgY29uc3QgbmF2QnRuID0g"
    "ZG9jdW1lbnQucXVlcnlTZWxlY3RvcignLm5hdiBidXR0b25bZGF0YS12aWV3PSJzZXR0aW5ncyJdJyk7CiAgICBsZXQgZG90ID0gbmF2QnRuLnF1ZXJ5"
    "U2VsZWN0b3IoIi51cGRvdCIpOwogICAgaWYoaW5mby5hdmFpbGFibGUpewogICAgICBpZighZG90KXsgZG90ID0gZWwoInNwYW4iLCJ1cGRvdCIpOyBu"
    "YXZCdG4uYXBwZW5kQ2hpbGQoZG90KTsgfQogICAgICBjb25zdCBiYW5uZXIgPSAkKCJ1cGRhdGVCYW5uZXJIb3N0Iik7CiAgICAgIGlmKGJhbm5lciAm"
    "JiAhYmFubmVyLmRhdGFzZXQuc2hvd24pewogICAgICAgIGJhbm5lci5kYXRhc2V0LnNob3duPSIxIjsKICAgICAgICBiYW5uZXIuaW5uZXJIVE1MID0g"
    "YDxkaXYgY2xhc3M9InVwZGF0ZWJhbm5lciI+CiAgICAgICAgICA8ZGl2IGNsYXNzPSJpYyI+PHN2ZyB3aWR0aD0iMjAiIGhlaWdodD0iMjAiIHZpZXdC"
    "b3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJjdXJyZW50Q29sb3IiIHN0cm9rZS13aWR0aD0iMiI+PHBhdGggZD0iTTEyIDE2VjRtMCAw"
    "IDQgNG0tNC00LTQgNCIvPjxwYXRoIGQ9Ik00IDE2djJhMiAyIDAgMCAwIDIgMmgxMmEyIDIgMCAwIDAgMi0ydi0yIi8+PC9zdmc+PC9kaXY+CiAgICAg"
    "ICAgICA8ZGl2IGNsYXNzPSJ1aSI+PGRpdiBjbGFzcz0idCI+SGVvcnRoIHYke2VzYyhpbmZvLnVwZGF0ZS52ZXJzaW9uKX0gaXMgYXZhaWxhYmxlPC9k"
    "aXY+CiAgICAgICAgICAgIDxkaXYgY2xhc3M9ImQiPkEgbmV3ZXIgdmVyc2lvbiBmaWxlIHdhcyBmb3VuZCBvbiB5b3VyIGNvbXB1dGVyLiBVcGRhdGUg"
    "ZnJvbSBTZXR0aW5ncy48L2Rpdj48L2Rpdj4KICAgICAgICAgIDxidXR0b24gY2xhc3M9ImJ0biBwcmltYXJ5IiBvbmNsaWNrPSJzaG93KCdzZXR0aW5n"
    "cycpIj5HbyB0byB1cGRhdGU8L2J1dHRvbj48L2Rpdj5gOwogICAgICB9CiAgICB9IGVsc2UgaWYoZG90KXsgZG90LnJlbW92ZSgpOyB9CiAgfWNhdGNo"
    "KGUpe30KfQoKLyogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09CiAgIElOSVQKICAgPT09"
    "PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09ICovCmFzeW5jIGZ1bmN0aW9uIHJlZnJlc2hPbGxh"
    "bWFTdGF0dXMoKXsKICB0cnl7CiAgICBjb25zdCBzID0gYXdhaXQgYXBpKCIvYXBpL3N5c3RlbSIpOyBzdGF0ZS5zeXN0ZW0gPSBzOwogICAgY29uc3Qg"
    "dXAgPSBzLm9sbGFtYS51cDsKICAgICQoIm9sbGFtYURvdCIpLmNsYXNzTmFtZSA9ICJkb3QgIisodXA/Im9uIjoib2ZmIik7CiAgICAkKCJvbGxhbWFT"
    "dGF0dXMiKS50ZXh0Q29udGVudCA9IHVwID8gIk9sbGFtYSAiK3Mub2xsYW1hLnZlcnNpb24gOiAiT2xsYW1hIG9mZmxpbmUiOwogICAgJCgib2xsYW1h"
    "U3RhdHVzIikuY2xhc3NOYW1lID0gdXAgPyAiIiA6ICJtdXRlZCI7CiAgICAkKCJicmFuZFZlciIpLnRleHRDb250ZW50ID0gInYiK3MudmVyc2lvbjsK"
    "ICAgIGFwcGx5VGhlbWUocyAmJiBzdGF0ZS5zZXR0aW5ncy50aGVtZSA/IHN0YXRlLnNldHRpbmdzLnRoZW1lIDoKICAgICAgKGRvY3VtZW50LmRvY3Vt"
    "ZW50RWxlbWVudC5kYXRhc2V0LnRoZW1lfHwiZGFyayIpKTsKICB9Y2F0Y2goZSl7CiAgICAkKCJvbGxhbWFEb3QiKS5jbGFzc05hbWU9ImRvdCBvZmYi"
    "OyAkKCJvbGxhbWFTdGF0dXMiKS50ZXh0Q29udGVudD0iU2VydmVyIGVycm9yIjsKICB9Cn0KYXN5bmMgZnVuY3Rpb24gaW5pdCgpewogIHRyeXsgY29u"
    "c3QgciA9IGF3YWl0IGFwaSgiL2FwaS9zZXR0aW5ncyIpOyBzdGF0ZS5zZXR0aW5ncyA9IHIuc2V0dGluZ3M7CiAgICBhcHBseVRoZW1lKHIuc2V0dGlu"
    "Z3MudGhlbWV8fCJkYXJrIik7IH1jYXRjaChlKXt9CiAgYXdhaXQgcmVmcmVzaENoYXRNb2RlbHMoKTsKICBhd2FpdCByZWZyZXNoT2xsYW1hU3RhdHVz"
    "KCk7CiAgbG9hZERhc2hib2FyZCgpOwogIHBvbGxVcGRhdGVCYWRnZSgpOwogIHNldEludGVydmFsKHJlZnJlc2hPbGxhbWFTdGF0dXMsIDEyMDAwKTsK"
    "ICBzZXRJbnRlcnZhbChwb2xsVXBkYXRlQmFkZ2UsIDYwMDAwKTsKfQppbml0KCk7Cjwvc2NyaXB0Pgo8L2JvZHk+CjwvaHRtbD4K"
)
INDEX_HTML = base64.b64decode(_INDEX_B64).decode("utf-8")


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(INDEX_HTML)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def _lan_ips() -> list:
    ips = set()
    try:
        for item in socket.getaddrinfo(socket.gethostname(), None,
                                       socket.AF_INET):
            ips.add(item[4][0])
    except OSError:
        pass
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ips.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    return sorted(ip for ip in ips if not ip.startswith("127."))


def _free_port(preferred: int, host: str) -> int:
    for candidate in [preferred] + list(range(preferred + 1, preferred + 15)):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((host, candidate))
                return candidate
            except OSError:
                continue
    return preferred


def _arg(flag: str, default):
    if flag in sys.argv:
        i = sys.argv.index(flag)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default


def main() -> None:
    host = _arg("--host", "127.0.0.1")
    port = int(_arg("--port", str(DEFAULT_PORT)))

    _restarting = (os.environ.get("HEORTH_RESTART") == "1"
                   or os.environ.get("LOCALMIND_RESTART") == "1")
    if _restarting:
        wait_for_port_free(host, port)

    wanted = port
    port = _free_port(port, host)
    RUNTIME["host"], RUNTIME["port"] = host, port
    if _restarting and port != wanted:
        print(f"[{APP_NAME}] WARNING: port {wanted} was still busy — "
              f"now on port {port}. Reload the browser at the new address.")
    url = f"http://{'127.0.0.1' if host in ('0.0.0.0', '') else host}:{port}"
    lan_lines = ""
    if host in ("0.0.0.0", ""):
        lan_lines = "".join(
            f"     Phones on this network:  http://{ip}:{port}\n"
            for ip in _lan_ips()[:3])
        if lan_lines and not (get_setting("lan_password") or "").strip():
            lan_lines += ("     (tip: set an access password in Settings "
                          "\u2192 Remote access)\n")

    banner = f"""
  ┌───────────────────────────────────────────────┐
     {APP_NAME}  v{__version__}
     Open in your browser:  {url}
{lan_lines}     Data folder:  {DATA_DIR}
     Press Ctrl+C here to stop the server.
  └───────────────────────────────────────────────┘
"""
    print(banner)

    if "--no-browser" not in sys.argv and not _restarting:
        def _open():
            time.sleep(1.2)
            try:
                import webbrowser
                webbrowser.open(url)
            except Exception:
                pass
        threading.Thread(target=_open, daemon=True).start()

    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n[{APP_NAME}] stopped.")
