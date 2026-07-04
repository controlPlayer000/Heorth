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

Chat modes: plain chat, Knowledge (answers grounded in your documents),
Agent (single-turn tool use), Loop (autonomous agent: it plans, calls
tools, observes and repeats until it declares the task complete), and
Council (a panel of consultants — each analyzes the question independently
and in parallel, then they read and critique each other over one or more
consultation rounds, and a chair writes the final synthesized answer).

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

__version__ = "1.2.1"

APP_NAME = "Heorth"
FILE_STEM = "heorth"                       # used for backups and new installs
FILE_STEMS = ("heorth", "localmind")       # update files may start with either
DEFAULT_PORT = 8317

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

for _sub in ("", "images", "rag_docs", "workspace", "backups", "updates"):
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


DEFAULT_SETTINGS = {
    "ollama_host": os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"),
    "system_prompt": "You are a helpful, precise assistant running fully "
                     "locally on the user's machine.",
    "embed_model": "nomic-embed-text",
    "rag_top_k": "5",
    "context_messages": "30",
    "allow_code_execution": "0",
    "allow_web_tools": "1",
    "agent_max_steps": "8",
    "loop_max_steps": "15",
    "council_size": "4",
    "council_rounds": "1",
    "council_research": "0",
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
                      stream: bool = False):
    """Non-streaming chat call (used by the agent for tool-calling steps)."""
    payload: dict = {"model": model, "messages": messages, "stream": False}
    if tools:
        payload["tools"] = tools
    async with httpx.AsyncClient(timeout=600) as c:
        r = await c.post(ollama_url("/api/chat"), json=payload)
        r.raise_for_status()
        return r.json()


async def ollama_chat_stream(messages: list, model: str) -> AsyncGenerator[dict, None]:
    payload = {"model": model, "messages": messages, "stream": True}
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
    for i, (piece, vec) in enumerate(zip(chunks, embeddings)):
        blob = np.asarray(vec, dtype=np.float32).tobytes()
        qx("INSERT INTO chunks(id,doc_id,idx,text,embedding) VALUES(?,?,?,?,?)",
           (new_id(), doc_id, i, piece, blob))
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
    if not str(get_setting("allow_web_tools")) == "1":
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
    if str(get_setting("allow_web_tools")) != "1":
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
    if str(get_setting("allow_code_execution")) != "1":
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
    if str(get_setting("allow_code_execution")) != "1":
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
    rows = q("SELECT role, content FROM messages WHERE conv_id=? "
             "ORDER BY created DESC LIMIT ?", (conv_id, limit))
    rows.reverse()
    msgs = [{"role": "system", "content": get_setting("system_prompt")}]
    msgs += [{"role": r["role"], "content": r["content"]} for r in rows
             if r["role"] in ("user", "assistant")]
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


async def run_chat(conv_id: str, user_message: str, model: str,
                   use_rag: bool, agent_mode: bool,
                   loop_mode: bool = False,
                   council_mode: bool = False) -> AsyncGenerator[str, None]:
    """Streams NDJSON events and persists both sides of the exchange.
    Persistence happens in a finally block so that when the user hits Stop
    (client disconnect), the partial answer is still saved."""

    def ev(obj: dict) -> str:
        return json.dumps(obj, ensure_ascii=False) + "\n"

    qx("INSERT INTO messages(id,conv_id,role,content,meta,created) "
       "VALUES(?,?,?,?,?,?)",
       (new_id(), conv_id, "user", user_message, "{}", now()))

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
        if council_mode:
            async for line in _run_council(messages, model, events_meta,
                                           full_answer):
                yield line
        elif agent_mode or loop_mode:
            async for line in _run_agent(messages, model, events_meta,
                                         full_answer, loop_mode=loop_mode):
                yield line
        else:
            async for part in ollama_chat_stream(messages, model):
                token = (part.get("message") or {}).get("content", "")
                if token:
                    full_answer.append(token)
                    yield ev({"type": "token", "text": token})
                if part.get("done"):
                    break
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
                     loop_mode: bool = False) -> AsyncGenerator[str, None]:
    def ev(obj: dict) -> str:
        if obj.get("type") in ("tool_call", "tool_result", "thought"):
            events_meta.append(obj)
        return json.dumps(obj, ensure_ascii=False) + "\n"

    tools = builtin_tool_schemas() + MCP.agent_tools()
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
                    token = (part.get("message") or {}).get("content", "")
                    if token:
                        full_answer.append(token)
                        yield ev({"type": "token", "text": token})
                    if part.get("done"):
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
                    return

                yield ev({"type": "tool_call", "name": name, "args": args})
                if name.startswith("mcp__"):
                    result = await MCP.call(name, args)
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
        full_answer.append(content)
        async for line in _stream_text(content, ev):
            yield line
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
    if str(get_setting("allow_web_tools")) != "1":
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
    if str(get_setting("council_research")) == "1":
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
        async for part in ollama_chat_stream(chair, model):
            token = (part.get("message") or {}).get("content", "")
            if token:
                full_answer.append(token)
                yield ev({"type": "token", "text": token})
            if part.get("done"):
                break
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
    subprocess.Popen([sys.executable, str(SCRIPT_PATH), *sys.argv[1:]],
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
            "mcp_installed": mcp_available()}


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

    async def gen():
        yield json.dumps({"type": "meta", "conversation_id": conv_id}) + "\n"
        async for line in run_chat(conv_id, message,
                                   model,
                                   bool((body or {}).get("use_rag")),
                                   bool((body or {}).get("agent_mode")),
                                   bool((body or {}).get("loop_mode")),
                                   bool((body or {}).get("council_mode"))):
            yield line
    return StreamingResponse(gen(), media_type="application/x-ndjson")

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
    "IzI2NWMzYTtjb2xvcjp2YXIoLS1saXZlKX0KLnRvZ2dsZSAuc3d7d2lkdGg6MjZweDtoZWlnaHQ6MTVweDtib3JkZXItcmFkaXVzOjIwcHg7YmFja2dy"
    "b3VuZDp2YXIoLS1saW5lKTsKICBwb3NpdGlvbjpyZWxhdGl2ZTt0cmFuc2l0aW9uOi4xNHM7ZmxleDpub25lfQoudG9nZ2xlIC5zdzo6YWZ0ZXJ7Y29u"
    "dGVudDoiIjtwb3NpdGlvbjphYnNvbHV0ZTt0b3A6MnB4O2xlZnQ6MnB4O3dpZHRoOjExcHg7aGVpZ2h0OjExcHg7CiAgYm9yZGVyLXJhZGl1czo1MCU7"
    "YmFja2dyb3VuZDp2YXIoLS1tdXRlZCk7dHJhbnNpdGlvbjouMTRzfQoudG9nZ2xlLm9uIC5zd3tiYWNrZ3JvdW5kOnZhcigtLXNpZ25hbCl9Ci50b2dn"
    "bGUub24gLnN3OjphZnRlcntsZWZ0OjEzcHg7YmFja2dyb3VuZDojMWExMjA0fQoudG9nZ2xlLm9uLnZpbyAuc3d7YmFja2dyb3VuZDp2YXIoLS12aW9s"
    "ZXQpfQoudG9nZ2xlLm9uLmJsdSAuc3d7YmFja2dyb3VuZDp2YXIoLS1ibHVlKX0KLnRvZ2dsZS5vbi5ncm4gLnN3e2JhY2tncm91bmQ6dmFyKC0tbGl2"
    "ZSl9Ci50b2dnbGUub24uZ3JuIC5zdzo6YWZ0ZXJ7YmFja2dyb3VuZDojMDQxNzBifQoKLyogY291bmNpbCBwYW5lbCAqLwouY291bmNpbHtib3JkZXI6"
    "MXB4IHNvbGlkIHZhcigtLWxpbmUpO2JvcmRlci1yYWRpdXM6dmFyKC0tcik7YmFja2dyb3VuZDp2YXIoLS1wYW5lbC0yKTsKICBwYWRkaW5nOjEzcHgg"
    "MTRweDttYXJnaW46NHB4IDB9Ci5jb3VuY2lsIC5jaGVhZHtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6MTFweDtsZXR0ZXItc3BhY2lu"
    "ZzouMTJlbTsKICB0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6dmFyKC0tbGl2ZSk7ZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtn"
    "YXA6OHB4fQouY3JvdW5ke2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMC41cHg7bGV0dGVyLXNwYWNpbmc6LjFlbTt0ZXh0LXRyYW5z"
    "Zm9ybTp1cHBlcmNhc2U7CiAgY29sb3I6dmFyKC0tbXV0ZWQpO21hcmdpbjoxM3B4IDAgOHB4O2Rpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7"
    "Z2FwOjlweH0KLmNyb3VuZDo6YWZ0ZXJ7Y29udGVudDoiIjtmbGV4OjE7aGVpZ2h0OjFweDtiYWNrZ3JvdW5kOnZhcigtLWxpbmUtc29mdCl9Ci5jZ3Jp"
    "ZHtkaXNwbGF5OmdyaWQ7Z3JpZC10ZW1wbGF0ZS1jb2x1bW5zOnJlcGVhdChhdXRvLWZpbGwsbWlubWF4KDIxNXB4LDFmcikpO2dhcDo5cHh9Ci5jY2Fy"
    "ZHtib3JkZXI6MXB4IHNvbGlkIHZhcigtLWxpbmUpO2JvcmRlci1yYWRpdXM6dmFyKC0tci1zbSk7YmFja2dyb3VuZDp2YXIoLS1wYW5lbCk7CiAgcGFk"
    "ZGluZzoxMHB4IDExcHg7bWluLXdpZHRoOjB9Ci5jY2FyZCAuY3JvbGV7ZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6N3B4O2ZvbnQt"
    "ZmFtaWx5OnZhcigtLW1vbm8pOwogIGZvbnQtc2l6ZToxMXB4O2ZvbnQtd2VpZ2h0OjYwMDttYXJnaW4tYm90dG9tOjJweH0KLmNjYXJkIC5jZG90e3dp"
    "ZHRoOjdweDtoZWlnaHQ6N3B4O2JvcmRlci1yYWRpdXM6NTAlO2ZsZXg6bm9uZX0KLmNjYXJkIC5jZm9jdXN7Zm9udC1zaXplOjEwLjVweDtjb2xvcjp2"
    "YXIoLS1kaW0pO21hcmdpbi1ib3R0b206NnB4O2xpbmUtaGVpZ2h0OjEuMzV9Ci5jY2FyZCAuY3N0e21hcmdpbi1sZWZ0OmF1dG87ZmxleDpub25lO2Rp"
    "c3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXJ9Ci5jY2FyZCAuY3Rha2V7Zm9udC1zaXplOjEycHg7Y29sb3I6dmFyKC0tbXV0ZWQpO2xpbmUtaGVp"
    "Z2h0OjEuNTt3aGl0ZS1zcGFjZTpwcmUtd3JhcDsKICBvdmVyZmxvdy13cmFwOmFueXdoZXJlO21heC1oZWlnaHQ6MTA4cHg7b3ZlcmZsb3c6aGlkZGVu"
    "O3Bvc2l0aW9uOnJlbGF0aXZlO2N1cnNvcjpwb2ludGVyfQouY2NhcmQgLmN0YWtlLm9wZW57bWF4LWhlaWdodDpub25lfQouY2NhcmQgLmN0YWtlOm5v"
    "dCgub3Blbik6OmFmdGVye2NvbnRlbnQ6IiI7cG9zaXRpb246YWJzb2x1dGU7bGVmdDowO3JpZ2h0OjA7Ym90dG9tOjA7aGVpZ2h0OjM0cHg7CiAgYmFj"
    "a2dyb3VuZDpsaW5lYXItZ3JhZGllbnQoMGRlZyx2YXIoLS1wYW5lbCksdHJhbnNwYXJlbnQpfQouY2NhcmQuZmFpbGVke29wYWNpdHk6LjU1fQoKLnRo"
    "b3VnaHR7Ym9yZGVyLWxlZnQ6MnB4IHNvbGlkIHZhcigtLXZpb2xldCk7cGFkZGluZzo3cHggMTFweDtmb250LXNpemU6MTIuNXB4OwogIGNvbG9yOnZh"
    "cigtLW11dGVkKTtiYWNrZ3JvdW5kOnZhcigtLXBhbmVsLTIpO2JvcmRlci1yYWRpdXM6MCA4cHggOHB4IDA7CiAgd2hpdGUtc3BhY2U6cHJlLXdyYXA7"
    "b3ZlcmZsb3ctd3JhcDphbnl3aGVyZX0KLnRob3VnaHQgLnRse2Rpc3BsYXk6YmxvY2s7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjku"
    "NXB4O2NvbG9yOnZhcigtLXZpb2xldCk7CiAgbGV0dGVyLXNwYWNpbmc6LjFlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7bWFyZ2luLWJvdHRvbToz"
    "cHh9CgouY2hhdHNjcm9sbHtmbGV4OjE7b3ZlcmZsb3cteTphdXRvO3BhZGRpbmc6MjRweCAwfQouY2hhdHdyYXB7bWF4LXdpZHRoOjgyMHB4O21hcmdp"
    "bjowIGF1dG87cGFkZGluZzowIDI0cHg7ZGlzcGxheTpmbGV4OwogIGZsZXgtZGlyZWN0aW9uOmNvbHVtbjtnYXA6MjBweH0KLm1zZ3tkaXNwbGF5OmZs"
    "ZXg7Z2FwOjEzcHg7YW5pbWF0aW9uOm1zZ2luIC4yNXMgZWFzZX0KQGtleWZyYW1lcyBtc2dpbntmcm9te29wYWNpdHk6MDt0cmFuc2Zvcm06dHJhbnNs"
    "YXRlWSg2cHgpfX0KLm1zZyAuYXZ7d2lkdGg6MzBweDtoZWlnaHQ6MzBweDtib3JkZXItcmFkaXVzOjlweDtmbGV4Om5vbmU7ZGlzcGxheTpncmlkOwog"
    "IHBsYWNlLWl0ZW1zOmNlbnRlcjtmb250LWZhbWlseTp2YXIoLS1tb25vKTtmb250LXNpemU6MTFweDtmb250LXdlaWdodDo3MDA7CiAgYm9yZGVyOjFw"
    "eCBzb2xpZCB2YXIoLS1saW5lKX0KLm1zZy51c2VyIC5hdntiYWNrZ3JvdW5kOnZhcigtLXJhaXNlZCk7Y29sb3I6dmFyKC0tbXV0ZWQpfQoubXNnLmFp"
    "IC5hdntiYWNrZ3JvdW5kOnJhZGlhbC1ncmFkaWVudCgxMjAlIDEyMCUgYXQgMzAlIDI1JSwjMjQzMjQ0LCMwZDEzMWIpOwogIGNvbG9yOnZhcigtLXNp"
    "Z25hbCl9Ci5tc2cgLmJvZHl7ZmxleDoxO21pbi13aWR0aDowO3BhZGRpbmctdG9wOjNweH0KLm1zZyAud2hve2ZvbnQtc2l6ZToxMXB4O2NvbG9yOnZh"
    "cigtLWRpbSk7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7bWFyZ2luLWJvdHRvbTo0cHg7CiAgdGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO2xldHRlci1z"
    "cGFjaW5nOi4wOGVtfQouYnViYmxle2ZvbnQtc2l6ZToxNC41cHg7bGluZS1oZWlnaHQ6MS42Mjt3b3JkLXdyYXA6YnJlYWstd29yZDtvdmVyZmxvdy13"
    "cmFwOmFueXdoZXJlfQouYnViYmxlIHB7bWFyZ2luOjAgMCAxMHB4fSAuYnViYmxlIHA6bGFzdC1jaGlsZHttYXJnaW4tYm90dG9tOjB9Ci5idWJibGUg"
    "cHJle2JhY2tncm91bmQ6dmFyKC0tcGFuZWwtMik7Ym9yZGVyOjFweCBzb2xpZCB2YXIoLS1saW5lKTtib3JkZXItcmFkaXVzOnZhcigtLXItc20pOwog"
    "IHBhZGRpbmc6MTNweCAxNXB4O292ZXJmbG93LXg6YXV0bzttYXJnaW46MTFweCAwO2ZvbnQtc2l6ZToxMi41cHg7bGluZS1oZWlnaHQ6MS41fQouYnVi"
    "YmxlIGNvZGV7YmFja2dyb3VuZDp2YXIoLS1yYWlzZWQpO3BhZGRpbmc6MS41cHggNnB4O2JvcmRlci1yYWRpdXM6NXB4O2ZvbnQtc2l6ZToxMi41cHh9"
    "Ci5idWJibGUgcHJlIGNvZGV7YmFja2dyb3VuZDpub25lO3BhZGRpbmc6MH0KLmJ1YmJsZSB1bCwuYnViYmxlIG9se21hcmdpbjo4cHggMDtwYWRkaW5n"
    "LWxlZnQ6MjJweH0gLmJ1YmJsZSBsaXttYXJnaW46M3B4IDB9Ci5idWJibGUgaDEsLmJ1YmJsZSBoMiwuYnViYmxlIGgze21hcmdpbjoxNnB4IDAgOHB4"
    "fQouYnViYmxlIGltZ3ttYXgtd2lkdGg6MTAwJTtib3JkZXItcmFkaXVzOnZhcigtLXItc20pO2JvcmRlcjoxcHggc29saWQgdmFyKC0tbGluZSk7bWFy"
    "Z2luOjhweCAwfQouYnViYmxlIGF7Ym9yZGVyLWJvdHRvbToxcHggc29saWQgdmFyKC0tc2lnbmFsLWRpbSl9Ci5idWJibGUgdGFibGV7Ym9yZGVyLWNv"
    "bGxhcHNlOmNvbGxhcHNlO21hcmdpbjoxMHB4IDA7Zm9udC1zaXplOjEyLjVweDt3aWR0aDoxMDAlfQouYnViYmxlIHRoLC5idWJibGUgdGR7Ym9yZGVy"
    "OjFweCBzb2xpZCB2YXIoLS1saW5lKTtwYWRkaW5nOjZweCAxMHB4O3RleHQtYWxpZ246bGVmdH0KLmN1cnNvci1ibGluazo6YWZ0ZXJ7Y29udGVudDoi"
    "4paLIjtjb2xvcjp2YXIoLS1zaWduYWwpO2FuaW1hdGlvbjpibGluayAxcyBzdGVwLWVuZCBpbmZpbml0ZTsKICBtYXJnaW4tbGVmdDoxcHh9CkBrZXlm"
    "cmFtZXMgYmxpbmt7NTAle29wYWNpdHk6MH19CgouYWdlbnRsb2d7bWFyZ2luOjZweCAwIDRweDtkaXNwbGF5OmZsZXg7ZmxleC1kaXJlY3Rpb246Y29s"
    "dW1uO2dhcDo2cHh9Ci5zdGVwe2JvcmRlcjoxcHggc29saWQgdmFyKC0tbGluZSk7Ym9yZGVyLXJhZGl1czp2YXIoLS1yLXNtKTtiYWNrZ3JvdW5kOnZh"
    "cigtLXBhbmVsLTIpOwogIG92ZXJmbG93OmhpZGRlbjtmb250LXNpemU6MTIuNXB4fQouc3RlcCAuc2h7ZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNl"
    "bnRlcjtnYXA6OXB4O3BhZGRpbmc6OHB4IDEycHg7Y3Vyc29yOnBvaW50ZXJ9Ci5zdGVwIC5zaCAudG57Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9u"
    "dC1zaXplOjExLjVweDtjb2xvcjp2YXIoLS12aW9sZXQpO2ZvbnQtd2VpZ2h0OjYwMH0KLnN0ZXAgLnNoIC5hcmd7Y29sb3I6dmFyKC0tbXV0ZWQpO2Zv"
    "bnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMXB4OwogIG92ZXJmbG93OmhpZGRlbjt0ZXh0LW92ZXJmbG93OmVsbGlwc2lzO3doaXRlLXNw"
    "YWNlOm5vd3JhcDtmbGV4OjE7bWluLXdpZHRoOjB9Ci5zdGVwIC5zYntib3JkZXItdG9wOjFweCBzb2xpZCB2YXIoLS1saW5lKTtwYWRkaW5nOjEwcHgg"
    "MTJweDtmb250LWZhbWlseTp2YXIoLS1tb25vKTsKICBmb250LXNpemU6MTEuNXB4O3doaXRlLXNwYWNlOnByZS13cmFwO2NvbG9yOnZhcigtLW11dGVk"
    "KTttYXgtaGVpZ2h0OjIzMHB4O292ZXJmbG93OmF1dG87CiAgYmFja2dyb3VuZDp2YXIoLS1pbmspfQouc3RhdHVzbGluZXtkaXNwbGF5OmZsZXg7YWxp"
    "Z24taXRlbXM6Y2VudGVyO2dhcDo5cHg7Zm9udC1zaXplOjEyLjVweDtjb2xvcjp2YXIoLS1tdXRlZCk7CiAgZm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7"
    "cGFkZGluZzoycHggMH0KLnNyY2JveHtib3JkZXI6MXB4IGRhc2hlZCB2YXIoLS1saW5lKTtib3JkZXItcmFkaXVzOnZhcigtLXItc20pO3BhZGRpbmc6"
    "MTBweCAxM3B4OwogIG1hcmdpbi1ib3R0b206NHB4O2JhY2tncm91bmQ6dmFyKC0tcGFuZWwtMil9Ci5zcmNib3ggLnN0e2ZvbnQtZmFtaWx5OnZhcigt"
    "LW1vbm8pO2ZvbnQtc2l6ZToxMC41cHg7dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlOwogIGxldHRlci1zcGFjaW5nOi4xZW07Y29sb3I6dmFyKC0tc2ln"
    "bmFsKTttYXJnaW4tYm90dG9tOjdweH0KLnNyY2l0ZW17Zm9udC1zaXplOjEycHg7Y29sb3I6dmFyKC0tbXV0ZWQpO3BhZGRpbmc6NHB4IDA7Ym9yZGVy"
    "LXRvcDoxcHggc29saWQgdmFyKC0tbGluZS1zb2Z0KX0KLnNyY2l0ZW06Zmlyc3Qtb2YtdHlwZXtib3JkZXItdG9wOm5vbmV9Ci5zcmNpdGVtIGJ7Y29s"
    "b3I6dmFyKC0tdGV4dCl9CgouY29tcG9zZXJ7Ym9yZGVyLXRvcDoxcHggc29saWQgdmFyKC0tbGluZSk7YmFja2dyb3VuZDp2YXIoLS1wYW5lbC0yKTtw"
    "YWRkaW5nOjE0cHggMjRweCAxOHB4fQouY29tcG9zZXIgLmNib3h7bWF4LXdpZHRoOjgyMHB4O21hcmdpbjowIGF1dG87cG9zaXRpb246cmVsYXRpdmU7"
    "CiAgYm9yZGVyOjFweCBzb2xpZCB2YXIoLS1saW5lKTtib3JkZXItcmFkaXVzOnZhcigtLXIpO2JhY2tncm91bmQ6dmFyKC0tcGFuZWwpOwogIHRyYW5z"
    "aXRpb246Ym9yZGVyIC4xNHMsYm94LXNoYWRvdyAuMTRzO2Rpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpmbGV4LWVuZDtnYXA6OHB4O3BhZGRpbmc6OHB4"
    "IDhweCA4cHggMTRweH0KLmNvbXBvc2VyIC5jYm94OmZvY3VzLXdpdGhpbntib3JkZXItY29sb3I6dmFyKC0tc2lnbmFsKTtib3gtc2hhZG93OjAgMCAw"
    "IDNweCB2YXIoLS1zaWduYWwtZ2xvdyl9Ci5jb21wb3NlciB0ZXh0YXJlYXtmbGV4OjE7YmFja2dyb3VuZDpub25lO2JvcmRlcjpub25lO291dGxpbmU6"
    "bm9uZTtyZXNpemU6bm9uZTsKICBmb250LXNpemU6MTQuNXB4O2xpbmUtaGVpZ2h0OjEuNTU7bWF4LWhlaWdodDoyMDBweDtwYWRkaW5nOjhweCAwO2Nv"
    "bG9yOnZhcigtLXRleHQpfQouY29tcG9zZXIgdGV4dGFyZWE6OnBsYWNlaG9sZGVye2NvbG9yOnZhcigtLWRpbSl9Ci5zZW5kYnRue3dpZHRoOjM4cHg7"
    "aGVpZ2h0OjM4cHg7Ym9yZGVyLXJhZGl1czoxMHB4O2JhY2tncm91bmQ6dmFyKC0tc2lnbmFsKTtjb2xvcjojMWExMjA0OwogIGRpc3BsYXk6Z3JpZDtw"
    "bGFjZS1pdGVtczpjZW50ZXI7ZmxleDpub25lO3RyYW5zaXRpb246LjE0czttYXJnaW4tYm90dG9tOjFweH0KLnNlbmRidG46aG92ZXJ7ZmlsdGVyOmJy"
    "aWdodG5lc3MoMS4wNyl9IC5zZW5kYnRuOmRpc2FibGVke29wYWNpdHk6LjQ7Y3Vyc29yOm5vdC1hbGxvd2VkfQouc2VuZGJ0bi5zdG9we2JhY2tncm91"
    "bmQ6dmFyKC0tcmVkKTtjb2xvcjojZmZmfQouc2VuZGJ0bi5zdG9wOmhvdmVye2ZpbHRlcjpicmlnaHRuZXNzKDEuMSl9Ci5jaGF0LXNpZGV7d2lkdGg6"
    "MjUwcHg7Ym9yZGVyLWxlZnQ6MXB4IHNvbGlkIHZhcigtLWxpbmUpO2JhY2tncm91bmQ6dmFyKC0tcGFuZWwtMik7CiAgZGlzcGxheTpmbGV4O2ZsZXgt"
    "ZGlyZWN0aW9uOmNvbHVtbjtmbGV4Om5vbmV9Ci5jaGF0LWxheW91dHtmbGV4OjE7ZGlzcGxheTpmbGV4O21pbi1oZWlnaHQ6MH0KLmNoYXQtbWFpbntm"
    "bGV4OjE7ZGlzcGxheTpmbGV4O2ZsZXgtZGlyZWN0aW9uOmNvbHVtbjttaW4td2lkdGg6MH0KLmNvbnZsaXN0e2ZsZXg6MTtvdmVyZmxvdy15OmF1dG87"
    "cGFkZGluZzoxMHB4fQouY29udml0ZW17cGFkZGluZzoxMHB4IDEycHg7Ym9yZGVyLXJhZGl1czp2YXIoLS1yLXNtKTtjdXJzb3I6cG9pbnRlcjttYXJn"
    "aW4tYm90dG9tOjNweDsKICBmb250LXNpemU6MTNweDtjb2xvcjp2YXIoLS1tdXRlZCk7ZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6"
    "OHB4O3RyYW5zaXRpb246LjEyczsKICBwb3NpdGlvbjpyZWxhdGl2ZX0KLmNvbnZpdGVtOmhvdmVye2JhY2tncm91bmQ6dmFyKC0tcGFuZWwpO2NvbG9y"
    "OnZhcigtLXRleHQpfQouY29udml0ZW0uYWN0aXZle2JhY2tncm91bmQ6dmFyKC0tcmFpc2VkKTtjb2xvcjp2YXIoLS10ZXh0KX0KLmNvbnZpdGVtIC50"
    "dHtmbGV4OjE7b3ZlcmZsb3c6aGlkZGVuO3RleHQtb3ZlcmZsb3c6ZWxsaXBzaXM7d2hpdGUtc3BhY2U6bm93cmFwfQouY29udml0ZW0gLmRlbHtvcGFj"
    "aXR5OjA7Y29sb3I6dmFyKC0tZGltKTtwYWRkaW5nOjJweH0KLmNvbnZpdGVtOmhvdmVyIC5kZWx7b3BhY2l0eToxfSAuY29udml0ZW0gLmRlbDpob3Zl"
    "cntjb2xvcjp2YXIoLS1yZWQpfQoubmV3Y2hhdHttYXJnaW46MTBweDtkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2p1c3RpZnktY29udGVu"
    "dDpjZW50ZXI7Z2FwOjhweH0KQG1lZGlhIChtYXgtd2lkdGg6MTAwMHB4KXsuY2hhdC1zaWRle2Rpc3BsYXk6bm9uZX19Ci5jaGF0LWVtcHR5e21hcmdp"
    "bjo5dmggYXV0byAwO3RleHQtYWxpZ246Y2VudGVyO3BhZGRpbmc6MjBweDttYXgtd2lkdGg6NDYwcHg7CiAgZGlzcGxheTpmbGV4O2ZsZXgtZGlyZWN0"
    "aW9uOmNvbHVtbjthbGlnbi1pdGVtczpjZW50ZXI7Z2FwOjlweH0KLmNoYXQtZW1wdHkgLm9yYnt3aWR0aDo1MnB4O2hlaWdodDo1MnB4O2JvcmRlci1y"
    "YWRpdXM6MTZweDsKICBiYWNrZ3JvdW5kOnJhZGlhbC1ncmFkaWVudCgxMjAlIDEyMCUgYXQgMzAlIDI1JSwjMjQzMjQ0LCMwZDEzMWIpOwogIGJvcmRl"
    "cjoxcHggc29saWQgdmFyKC0tbGluZSk7ZGlzcGxheTpncmlkO3BsYWNlLWl0ZW1zOmNlbnRlcjttYXJnaW4tYm90dG9tOjVweH0KLmNoYXQtZW1wdHkg"
    "Lm9yYjo6YmVmb3Jle2NvbnRlbnQ6IiI7d2lkdGg6MTZweDtoZWlnaHQ6MTZweDtib3JkZXItcmFkaXVzOjUwJTsKICBiYWNrZ3JvdW5kOnZhcigtLXNp"
    "Z25hbCk7Ym94LXNoYWRvdzowIDAgMCA2cHggdmFyKC0tc2lnbmFsLWdsb3cpLAogIDAgMCAyNnB4IDRweCB2YXIoLS1zaWduYWwtZ2xvdyk7YW5pbWF0"
    "aW9uOmNvcmVwdWxzZSAzLjRzIGVhc2UtaW4tb3V0IGluZmluaXRlfQouY2hhdC1lbXB0eSBoM3tmb250LXNpemU6MTlweH0KLmNoYXQtZW1wdHkgcHtt"
    "YXJnaW46MH0KLmNoYXQtZW1wdHkgLmV4cm93e2Rpc3BsYXk6ZmxleDtmbGV4LWRpcmVjdGlvbjpjb2x1bW47Z2FwOjhweDt3aWR0aDoxMDAlO21hcmdp"
    "bi10b3A6NnB4fQpAbWVkaWEgKHByZWZlcnMtcmVkdWNlZC1tb3Rpb246cmVkdWNlKXsKICAqLCo6OmJlZm9yZSwqOjphZnRlcnthbmltYXRpb24tZHVy"
    "YXRpb246LjAxbXMhaW1wb3J0YW50O3RyYW5zaXRpb24tZHVyYXRpb246LjAxbXMhaW1wb3J0YW50fX0KCi8qIC0tLS0tLS0tLS0gaW1hZ2VzIC0tLS0t"
    "LS0tLS0gKi8KLmltZ2xheW91dHtkaXNwbGF5OmdyaWQ7Z3JpZC10ZW1wbGF0ZS1jb2x1bW5zOjMzMHB4IDFmcjtnYXA6MjJweDthbGlnbi1pdGVtczpz"
    "dGFydH0KQG1lZGlhIChtYXgtd2lkdGg6OTAwcHgpey5pbWdsYXlvdXR7Z3JpZC10ZW1wbGF0ZS1jb2x1bW5zOjFmcn19Ci5nZW5wYW5lbHtwb3NpdGlv"
    "bjpzdGlja3k7dG9wOjB9Ci5pbWdwcmV2aWV3e2FzcGVjdC1yYXRpbzoxO2JvcmRlcjoxcHggc29saWQgdmFyKC0tbGluZSk7Ym9yZGVyLXJhZGl1czp2"
    "YXIoLS1yKTsKICBiYWNrZ3JvdW5kOnZhcigtLXBhbmVsLTIpO2Rpc3BsYXk6Z3JpZDtwbGFjZS1pdGVtczpjZW50ZXI7b3ZlcmZsb3c6aGlkZGVuO21h"
    "cmdpbi1ib3R0b206MTRweDsKICBwb3NpdGlvbjpyZWxhdGl2ZX0KLmltZ3ByZXZpZXcgaW1ne3dpZHRoOjEwMCU7aGVpZ2h0OjEwMCU7b2JqZWN0LWZp"
    "dDpjb250YWluO2N1cnNvcjp6b29tLWlufQouaW1ncHJldmlldyAucGh7dGV4dC1hbGlnbjpjZW50ZXI7Y29sb3I6dmFyKC0tZGltKTtwYWRkaW5nOjIw"
    "cHh9Ci5nYWxsZXJ5e2Rpc3BsYXk6Z3JpZDtncmlkLXRlbXBsYXRlLWNvbHVtbnM6cmVwZWF0KGF1dG8tZmlsbCxtaW5tYXgoMTUwcHgsMWZyKSk7Z2Fw"
    "OjEycHh9Ci5naXRlbXtib3JkZXI6MXB4IHNvbGlkIHZhcigtLWxpbmUpO2JvcmRlci1yYWRpdXM6dmFyKC0tci1zbSk7b3ZlcmZsb3c6aGlkZGVuOwog"
    "IGJhY2tncm91bmQ6dmFyKC0tcGFuZWwpO3Bvc2l0aW9uOnJlbGF0aXZlO2FzcGVjdC1yYXRpbzoxO2N1cnNvcjpwb2ludGVyO3RyYW5zaXRpb246LjE0"
    "c30KLmdpdGVtOmhvdmVye2JvcmRlci1jb2xvcjp2YXIoLS1kaW0pO3RyYW5zZm9ybTp0cmFuc2xhdGVZKC0ycHgpfQouZ2l0ZW0gaW1ne3dpZHRoOjEw"
    "MCU7aGVpZ2h0OjEwMCU7b2JqZWN0LWZpdDpjb3Zlcn0KLmdpdGVtIC5vdntwb3NpdGlvbjphYnNvbHV0ZTtpbnNldDowO2JhY2tncm91bmQ6bGluZWFy"
    "LWdyYWRpZW50KDBkZWcscmdiYSg0LDcsMTEsLjkpLHRyYW5zcGFyZW50IDQ1JSk7CiAgb3BhY2l0eTowO3RyYW5zaXRpb246LjE0cztkaXNwbGF5OmZs"
    "ZXg7ZmxleC1kaXJlY3Rpb246Y29sdW1uO2p1c3RpZnktY29udGVudDpmbGV4LWVuZDtwYWRkaW5nOjEwcHg7CiAgcG9pbnRlci1ldmVudHM6bm9uZX0K"
    "LmdpdGVtOmhvdmVyIC5vdntvcGFjaXR5OjF9Ci5naXRlbSAub3YgLnBye2ZvbnQtc2l6ZToxMXB4O2NvbG9yOnZhcigtLXRleHQpO2xpbmUtaGVpZ2h0"
    "OjEuNDsKICBkaXNwbGF5Oi13ZWJraXQtYm94Oy13ZWJraXQtbGluZS1jbGFtcDozOy13ZWJraXQtYm94LW9yaWVudDp2ZXJ0aWNhbDtvdmVyZmxvdzpo"
    "aWRkZW59Ci5naXRlbSAucm17cG9zaXRpb246YWJzb2x1dGU7dG9wOjdweDtyaWdodDo3cHg7d2lkdGg6MjZweDtoZWlnaHQ6MjZweDtib3JkZXItcmFk"
    "aXVzOjdweDsKICBiYWNrZ3JvdW5kOnJnYmEoNCw3LDExLC43NSk7Y29sb3I6I2ZmZjtkaXNwbGF5OmdyaWQ7cGxhY2UtaXRlbXM6Y2VudGVyO29wYWNp"
    "dHk6MDt0cmFuc2l0aW9uOi4xNHN9Ci5naXRlbTpob3ZlciAucm17b3BhY2l0eToxfSAuZ2l0ZW0gLnJtOmhvdmVye2JhY2tncm91bmQ6dmFyKC0tcmVk"
    "KX0KLnNpemVncmlke2Rpc3BsYXk6Z3JpZDtncmlkLXRlbXBsYXRlLWNvbHVtbnM6cmVwZWF0KDMsMWZyKTtnYXA6OHB4fQouc2l6ZW9wdHtwYWRkaW5n"
    "OjlweDtib3JkZXI6MXB4IHNvbGlkIHZhcigtLWxpbmUpO2JvcmRlci1yYWRpdXM6dmFyKC0tci1zbSk7dGV4dC1hbGlnbjpjZW50ZXI7CiAgZm9udC1z"
    "aXplOjEycHg7Y3Vyc29yOnBvaW50ZXI7dHJhbnNpdGlvbjouMTJzO2JhY2tncm91bmQ6dmFyKC0tcGFuZWwtMil9Ci5zaXplb3B0OmhvdmVye2JvcmRl"
    "ci1jb2xvcjp2YXIoLS1kaW0pfSAuc2l6ZW9wdC5vbntib3JkZXItY29sb3I6dmFyKC0tc2lnbmFsKTsKICBiYWNrZ3JvdW5kOnZhcigtLXNpZ25hbC1n"
    "bG93KTtjb2xvcjp2YXIoLS1zaWduYWwpfQouc2l6ZW9wdCAubW9ub3tmb250LXNpemU6MTBweDtjb2xvcjp2YXIoLS1kaW0pO2Rpc3BsYXk6YmxvY2t9"
    "Ci5zaXplb3B0Lm9uIC5tb25ve2NvbG9yOnZhcigtLXNpZ25hbCl9CgovKiAtLS0tLS0tLS0tIHRvb2xzIC8gbWNwIC8gcmFnIC8gc2V0dGluZ3MgLS0t"
    "LS0tLS0tLSAqLwouc2V0dXAtY2FyZHt0ZXh0LWFsaWduOmNlbnRlcjtwYWRkaW5nOjM0cHggMjZweH0KLnNldHVwLWNhcmQgLmlje2ZvbnQtc2l6ZTo0"
    "MHB4O21hcmdpbi1ib3R0b206MTRweH0KLmluc3RhbGxsb2d7YmFja2dyb3VuZDp2YXIoLS1pbmspO2JvcmRlcjoxcHggc29saWQgdmFyKC0tbGluZSk7"
    "Ym9yZGVyLXJhZGl1czp2YXIoLS1yLXNtKTsKICBwYWRkaW5nOjEycHg7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7Zm9udC1zaXplOjExcHg7Y29sb3I6"
    "dmFyKC0tbXV0ZWQpO21heC1oZWlnaHQ6MjIwcHg7CiAgb3ZlcmZsb3c6YXV0bzt0ZXh0LWFsaWduOmxlZnQ7d2hpdGUtc3BhY2U6cHJlLXdyYXA7bWFy"
    "Z2luLXRvcDoxNHB4O2xpbmUtaGVpZ2h0OjEuNX0KLnRvb2xyb3d7ZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6MTNweDtwYWRkaW5n"
    "OjEycHggMTRweDtib3JkZXI6MXB4IHNvbGlkIHZhcigtLWxpbmUpOwogIGJvcmRlci1yYWRpdXM6dmFyKC0tci1zbSk7bWFyZ2luLWJvdHRvbTo4cHg7"
    "YmFja2dyb3VuZDp2YXIoLS1wYW5lbCl9Ci50b29scm93IC50aWN7d2lkdGg6MzBweDtoZWlnaHQ6MzBweDtib3JkZXItcmFkaXVzOjhweDtiYWNrZ3Jv"
    "dW5kOnZhcigtLXBhbmVsLTIpOwogIGJvcmRlcjoxcHggc29saWQgdmFyKC0tbGluZSk7ZGlzcGxheTpncmlkO3BsYWNlLWl0ZW1zOmNlbnRlcjtmbGV4"
    "Om5vbmU7Y29sb3I6dmFyKC0tdmlvbGV0KX0KLnRvb2xyb3cgLnRpe2ZsZXg6MTttaW4td2lkdGg6MH0KLnRvb2xyb3cgLnRpIC5ubXtmb250LWZhbWls"
    "eTp2YXIoLS1tb25vKTtmb250LXNpemU6MTNweDtmb250LXdlaWdodDo2MDB9Ci50b29scm93IC50aSAuZHN7Zm9udC1zaXplOjEycHg7Y29sb3I6dmFy"
    "KC0tbXV0ZWQpfQouc3J2e2JvcmRlcjoxcHggc29saWQgdmFyKC0tbGluZSk7Ym9yZGVyLXJhZGl1czp2YXIoLS1yKTtwYWRkaW5nOjE0cHggMTZweDtt"
    "YXJnaW4tYm90dG9tOjEwcHg7CiAgYmFja2dyb3VuZDp2YXIoLS1wYW5lbCl9Ci5zcnYgLnNoe2Rpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7"
    "Z2FwOjExcHh9Ci5zcnYgLnNoIC5ubXtmb250LXdlaWdodDo2MDA7Zm9udC1zaXplOjE0cHh9IC5zcnYgLmNtZHtmb250LWZhbWlseTp2YXIoLS1tb25v"
    "KTtmb250LXNpemU6MTFweDsKICBjb2xvcjp2YXIoLS1kaW0pO21hcmdpbi10b3A6NXB4O3dvcmQtYnJlYWs6YnJlYWstYWxsfQouc3J2IC50b29sY2hp"
    "cHN7ZGlzcGxheTpmbGV4O2dhcDo1cHg7ZmxleC13cmFwOndyYXA7bWFyZ2luLXRvcDo5cHh9Ci5kb2Nyb3d7ZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1z"
    "OmNlbnRlcjtnYXA6MTNweDtwYWRkaW5nOjEycHggMTRweDtib3JkZXI6MXB4IHNvbGlkIHZhcigtLWxpbmUpOwogIGJvcmRlci1yYWRpdXM6dmFyKC0t"
    "ci1zbSk7bWFyZ2luLWJvdHRvbTo4cHg7YmFja2dyb3VuZDp2YXIoLS1wYW5lbCl9Ci5kb2Nyb3cgLmRpY3t3aWR0aDozNHB4O2hlaWdodDozNHB4O2Jv"
    "cmRlci1yYWRpdXM6OHB4O2JhY2tncm91bmQ6dmFyKC0tcGFuZWwtMik7CiAgYm9yZGVyOjFweCBzb2xpZCB2YXIoLS1saW5lKTtkaXNwbGF5OmdyaWQ7"
    "cGxhY2UtaXRlbXM6Y2VudGVyO2ZsZXg6bm9uZTtjb2xvcjp2YXIoLS1ibHVlKX0KLmRvY3JvdyAuZGl7ZmxleDoxO21pbi13aWR0aDowfSAuZG9jcm93"
    "IC5kaSAubm17Zm9udC13ZWlnaHQ6NjAwO2ZvbnQtc2l6ZToxMy41cHg7CiAgb3ZlcmZsb3c6aGlkZGVuO3RleHQtb3ZlcmZsb3c6ZWxsaXBzaXM7d2hp"
    "dGUtc3BhY2U6bm93cmFwfQouZG9jcm93IC5kaSAubXR7Zm9udC1zaXplOjExLjVweDtjb2xvcjp2YXIoLS1kaW0pO2ZvbnQtZmFtaWx5OnZhcigtLW1v"
    "bm8pfQouZHJvcHpvbmV7Ym9yZGVyOjJweCBkYXNoZWQgdmFyKC0tbGluZSk7Ym9yZGVyLXJhZGl1czp2YXIoLS1yKTtwYWRkaW5nOjMycHg7dGV4dC1h"
    "bGlnbjpjZW50ZXI7CiAgY29sb3I6dmFyKC0tbXV0ZWQpO3RyYW5zaXRpb246LjE1cztjdXJzb3I6cG9pbnRlcjtiYWNrZ3JvdW5kOnZhcigtLXBhbmVs"
    "LTIpfQouZHJvcHpvbmU6aG92ZXIsLmRyb3B6b25lLmRyYWd7Ym9yZGVyLWNvbG9yOnZhcigtLXNpZ25hbCk7YmFja2dyb3VuZDp2YXIoLS1zaWduYWwt"
    "Z2xvdyk7Y29sb3I6dmFyKC0tdGV4dCl9Ci5zZXR0aW5ncm93e2Rpc3BsYXk6ZmxleDtqdXN0aWZ5LWNvbnRlbnQ6c3BhY2UtYmV0d2VlbjthbGlnbi1p"
    "dGVtczpjZW50ZXI7Z2FwOjE2cHg7CiAgcGFkZGluZzoxNXB4IDA7Ym9yZGVyLWJvdHRvbToxcHggc29saWQgdmFyKC0tbGluZS1zb2Z0KX0KLnNldHRp"
    "bmdyb3c6bGFzdC1jaGlsZHtib3JkZXItYm90dG9tOm5vbmV9Ci5zZXR0aW5ncm93IC5sYWJ7Zm9udC1zaXplOjEzLjVweDtmb250LXdlaWdodDo1MDB9"
    "IC5zZXR0aW5ncm93IC5sYWIgLnN1Yntmb250LXNpemU6MTJweDsKICBjb2xvcjp2YXIoLS1tdXRlZCk7Zm9udC13ZWlnaHQ6NDAwO21hcmdpbi10b3A6"
    "MnB4O21heC13aWR0aDo0NmNofQouc2V0dGluZ3JvdyAuY3Rse2ZsZXg6bm9uZTttaW4td2lkdGg6MTgwcHh9Ci5zd2l0Y2h7d2lkdGg6NDRweDtoZWln"
    "aHQ6MjRweDtib3JkZXItcmFkaXVzOjIwcHg7YmFja2dyb3VuZDp2YXIoLS1saW5lKTtwb3NpdGlvbjpyZWxhdGl2ZTsKICBjdXJzb3I6cG9pbnRlcjt0"
    "cmFuc2l0aW9uOi4xNXM7ZmxleDpub25lfQouc3dpdGNoOjphZnRlcntjb250ZW50OiIiO3Bvc2l0aW9uOmFic29sdXRlO3RvcDoycHg7bGVmdDoycHg7"
    "d2lkdGg6MThweDtoZWlnaHQ6MThweDsKICBib3JkZXItcmFkaXVzOjUwJTtiYWNrZ3JvdW5kOiNmZmY7dHJhbnNpdGlvbjouMTVzfQouc3dpdGNoLm9u"
    "e2JhY2tncm91bmQ6dmFyKC0tc2lnbmFsKX0gLnN3aXRjaC5vbjo6YWZ0ZXJ7bGVmdDoyMnB4fQoKLyogdXBkYXRlIGJhbm5lciAqLwoudXBkYXRlYmFu"
    "bmVye21hcmdpbjowIDAgMjBweDtib3JkZXI6MXB4IHNvbGlkIHZhcigtLXNpZ25hbC1kaW0pO2JvcmRlci1yYWRpdXM6dmFyKC0tcik7CiAgYmFja2dy"
    "b3VuZDpsaW5lYXItZ3JhZGllbnQoMTAwZGVnLHZhcigtLXNpZ25hbC1nbG93KSx0cmFuc3BhcmVudCk7cGFkZGluZzoxNnB4IDE4cHg7CiAgZGlzcGxh"
    "eTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6MTVweH0KLnVwZGF0ZWJhbm5lciAuaWN7d2lkdGg6NDBweDtoZWlnaHQ6NDBweDtib3JkZXItcmFk"
    "aXVzOjEwcHg7YmFja2dyb3VuZDp2YXIoLS1zaWduYWwpOwogIGNvbG9yOiMxYTEyMDQ7ZGlzcGxheTpncmlkO3BsYWNlLWl0ZW1zOmNlbnRlcjtmbGV4"
    "Om5vbmV9Ci51cGRhdGViYW5uZXIgLnVpe2ZsZXg6MX0KLnVwZGF0ZWJhbm5lciAudWkgLnR7Zm9udC13ZWlnaHQ6NjAwO2ZvbnQtc2l6ZToxNHB4fQou"
    "dXBkYXRlYmFubmVyIC51aSAuZHtmb250LXNpemU6MTIuNXB4O2NvbG9yOnZhcigtLW11dGVkKTttYXJnaW4tdG9wOjJweH0KLm5hdiAudXBkb3R7cG9z"
    "aXRpb246YWJzb2x1dGU7cmlnaHQ6MTBweDt3aWR0aDo3cHg7aGVpZ2h0OjdweDtib3JkZXItcmFkaXVzOjUwJTsKICBiYWNrZ3JvdW5kOnZhcigtLXNp"
    "Z25hbCk7Ym94LXNoYWRvdzowIDAgOHB4IHZhcigtLXNpZ25hbCl9Cjwvc3R5bGU+CjwvaGVhZD4KPGJvZHk+CjxkaXYgY2xhc3M9ImFwcCI+CiAgPCEt"
    "LSA9PT09PT09PT09PT0gU0lERUJBUiA9PT09PT09PT09PT0gLS0+CiAgPGFzaWRlIGNsYXNzPSJzaWRlIiBpZD0ic2lkZSI+CiAgICA8ZGl2IGNsYXNz"
    "PSJicmFuZCI+CiAgICAgIDxkaXYgY2xhc3M9Im1hcmsiPjwvZGl2PgogICAgICA8ZGl2PgogICAgICAgIDxoMT5IZW9ydGg8L2gxPgogICAgICAgIDxk"
    "aXYgY2xhc3M9InZlciIgaWQ9ImJyYW5kVmVyIj524oCUPC9kaXY+CiAgICAgIDwvZGl2PgogICAgPC9kaXY+CgogICAgPG5hdiBjbGFzcz0ibmF2IiBp"
    "ZD0ibmF2Ij4KICAgICAgPGJ1dHRvbiBkYXRhLXZpZXc9ImRhc2hib2FyZCIgY2xhc3M9ImFjdGl2ZSI+CiAgICAgICAgPHN2ZyBjbGFzcz0iaWMiIHZp"
    "ZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJjdXJyZW50Q29sb3IiIHN0cm9rZS13aWR0aD0iMS44Ij48cmVjdCB4PSIzIiB5PSIz"
    "IiB3aWR0aD0iNyIgaGVpZ2h0PSI5IiByeD0iMS41Ii8+PHJlY3QgeD0iMTQiIHk9IjMiIHdpZHRoPSI3IiBoZWlnaHQ9IjUiIHJ4PSIxLjUiLz48cmVj"
    "dCB4PSIxNCIgeT0iMTIiIHdpZHRoPSI3IiBoZWlnaHQ9IjkiIHJ4PSIxLjUiLz48cmVjdCB4PSIzIiB5PSIxNiIgd2lkdGg9IjciIGhlaWdodD0iNSIg"
    "cng9IjEuNSIvPjwvc3ZnPgogICAgICAgIERhc2hib2FyZAogICAgICA8L2J1dHRvbj4KICAgICAgPGJ1dHRvbiBkYXRhLXZpZXc9ImNoYXQiPgogICAg"
    "ICAgIDxzdmcgY2xhc3M9ImljIiB2aWV3Qm94PSIwIDAgMjQgMjQiIGZpbGw9Im5vbmUiIHN0cm9rZT0iY3VycmVudENvbG9yIiBzdHJva2Utd2lkdGg9"
    "IjEuOCI+PHBhdGggZD0iTTIxIDEyYTggOCAwIDAgMS0xMS41IDcuMkw0IDIwbDEtNC4zQTggOCAwIDEgMSAyMSAxMloiLz48L3N2Zz4KICAgICAgICBD"
    "aGF0CiAgICAgIDwvYnV0dG9uPgogICAgICA8YnV0dG9uIGRhdGEtdmlldz0ibW9kZWxzIj4KICAgICAgICA8c3ZnIGNsYXNzPSJpYyIgdmlld0JveD0i"
    "MCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9ImN1cnJlbnRDb2xvciIgc3Ryb2tlLXdpZHRoPSIxLjgiPjxwYXRoIGQ9Ik0xMiAzIDMgNy41IDEy"
    "IDEybDktNC41TDEyIDNaIi8+PHBhdGggZD0iTTMgMTJsOSA0LjVMMjEgMTIiLz48cGF0aCBkPSJNMyAxNi41IDEyIDIxbDktNC41Ii8+PC9zdmc+CiAg"
    "ICAgICAgTW9kZWxzCiAgICAgIDwvYnV0dG9uPgogICAgICA8YnV0dG9uIGRhdGEtdmlldz0iaW1hZ2VzIj4KICAgICAgICA8c3ZnIGNsYXNzPSJpYyIg"
    "dmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9ImN1cnJlbnRDb2xvciIgc3Ryb2tlLXdpZHRoPSIxLjgiPjxyZWN0IHg9IjMiIHk9"
    "IjMiIHdpZHRoPSIxOCIgaGVpZ2h0PSIxOCIgcng9IjIuNSIvPjxjaXJjbGUgY3g9IjguNSIgY3k9IjguNSIgcj0iMS44Ii8+PHBhdGggZD0ibTIxIDE1"
    "LTUtNUw1IDIxIi8+PC9zdmc+CiAgICAgICAgSW1hZ2VzCiAgICAgIDwvYnV0dG9uPgogICAgICA8YnV0dG9uIGRhdGEtdmlldz0ia25vd2xlZGdlIj4K"
    "ICAgICAgICA8c3ZnIGNsYXNzPSJpYyIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9ImN1cnJlbnRDb2xvciIgc3Ryb2tlLXdp"
    "ZHRoPSIxLjgiPjxwYXRoIGQ9Ik00IDUuNUEyLjUgMi41IDAgMCAxIDYuNSAzSDIwdjE1SDYuNUEyLjUgMi41IDAgMCAwIDQgMjAuNVY1LjVaIi8+PHBh"
    "dGggZD0iTTQgNS41VjIwIi8+PC9zdmc+CiAgICAgICAgS25vd2xlZGdlCiAgICAgIDwvYnV0dG9uPgogICAgICA8YnV0dG9uIGRhdGEtdmlldz0idG9v"
    "bHMiPgogICAgICAgIDxzdmcgY2xhc3M9ImljIiB2aWV3Qm94PSIwIDAgMjQgMjQiIGZpbGw9Im5vbmUiIHN0cm9rZT0iY3VycmVudENvbG9yIiBzdHJv"
    "a2Utd2lkdGg9IjEuOCI+PHBhdGggZD0iTTE0LjcgNi4zYTQgNCAwIDAgMC01LjQgNS40bC02IDYgMiAyIDYtNmE0IDQgMCAwIDAgNS40LTUuNGwtMi41"
    "IDIuNS0yLTIgMi41LTIuNVoiLz48L3N2Zz4KICAgICAgICBBZ2VudCAmYW1wOyBUb29scwogICAgICA8L2J1dHRvbj4KICAgICAgPGRpdiBjbGFzcz0i"
    "bmF2LXNlcCI+PC9kaXY+CiAgICAgIDxidXR0b24gZGF0YS12aWV3PSJzZXR0aW5ncyI+CiAgICAgICAgPHN2ZyBjbGFzcz0iaWMiIHZpZXdCb3g9IjAg"
    "MCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJjdXJyZW50Q29sb3IiIHN0cm9rZS13aWR0aD0iMS44Ij48Y2lyY2xlIGN4PSIxMiIgY3k9IjEyIiBy"
    "PSIzIi8+PHBhdGggZD0iTTE5LjQgMTVhMS42IDEuNiAwIDAgMCAuMyAxLjhsLjEuMWEyIDIgMCAxIDEtMi44IDIuOGwtLjEtLjFhMS42IDEuNiAwIDAg"
    "MC0xLjgtLjMgMS42IDEuNiAwIDAgMC0xIDEuNVYyMWEyIDIgMCAxIDEtNCAwdi0uMWExLjYgMS42IDAgMCAwLTEtMS41IDEuNiAxLjYgMCAwIDAtMS44"
    "LjNsLS4xLjFhMiAyIDAgMSAxLTIuOC0yLjhsLjEtLjFhMS42IDEuNiAwIDAgMCAuMy0xLjggMS42IDEuNiAwIDAgMC0xLjUtMUgzYTIgMiAwIDEgMSAw"
    "LTRoLjFhMS42IDEuNiAwIDAgMCAxLjUtMSAxLjYgMS42IDAgMCAwLS4zLTEuOGwtLjEtLjFhMiAyIDAgMSAxIDIuOC0yLjhsLjEuMWExLjYgMS42IDAg"
    "MCAwIDEuOC4zSDlhMS42IDEuNiAwIDAgMCAxLTEuNVYzYTIgMiAwIDEgMSA0IDB2LjFhMS42IDEuNiAwIDAgMCAxIDEuNSAxLjYgMS42IDAgMCAwIDEu"
    "OC0uM2wuMS0uMWEyIDIgMCAxIDEgMi44IDIuOGwtLjEuMWExLjYgMS42IDAgMCAwLS4zIDEuOFY5YTEuNiAxLjYgMCAwIDAgMS41IDFIMjFhMiAyIDAg"
    "MSAxIDAgNGgtLjFhMS42IDEuNiAwIDAgMC0xLjUgMVoiLz48L3N2Zz4KICAgICAgICBTZXR0aW5ncwogICAgICA8L2J1dHRvbj4KICAgIDwvbmF2PgoK"
    "ICAgIDxkaXYgY2xhc3M9InNpZGUtZm9vdCI+CiAgICAgIDxkaXYgY2xhc3M9InN0YXR1c2JhciI+CiAgICAgICAgPHNwYW4gY2xhc3M9ImRvdCIgaWQ9"
    "Im9sbGFtYURvdCI+PC9zcGFuPgogICAgICAgIDxzcGFuIGlkPSJvbGxhbWFTdGF0dXMiIGNsYXNzPSJtdXRlZCI+Q2hlY2tpbmcgT2xsYW1h4oCmPC9z"
    "cGFuPgogICAgICA8L2Rpdj4KICAgICAgPGJ1dHRvbiBjbGFzcz0idGhlbWV0b2dnbGUiIGlkPSJ0aGVtZUJ0biI+CiAgICAgICAgPHN2ZyB3aWR0aD0i"
    "MTQiIGhlaWdodD0iMTQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJjdXJyZW50Q29sb3IiIHN0cm9rZS13aWR0aD0iMS44"
    "Ij48cGF0aCBkPSJNMjEgMTIuOEE5IDkgMCAxIDEgMTEuMiAzYTcgNyAwIDAgMCA5LjggOS44WiIvPjwvc3ZnPgogICAgICAgIDxzcGFuIGlkPSJ0aGVt"
    "ZUxhYmVsIj5MaWdodCBtb2RlPC9zcGFuPgogICAgICA8L2J1dHRvbj4KICAgIDwvZGl2PgogIDwvYXNpZGU+CgogIDwhLS0gPT09PT09PT09PT09IE1B"
    "SU4gPT09PT09PT09PT09IC0tPgogIDxtYWluIGNsYXNzPSJtYWluIj4KICAgIDxkaXYgY2xhc3M9Im1vYmlsZWJhciI+CiAgICAgIDxidXR0b24gY2xh"
    "c3M9ImJ0biBpY29uIGdob3N0IiBpZD0ibWVudUJ0biI+CiAgICAgICAgPHN2ZyB3aWR0aD0iMjAiIGhlaWdodD0iMjAiIHZpZXdCb3g9IjAgMCAyNCAy"
    "NCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJjdXJyZW50Q29sb3IiIHN0cm9rZS13aWR0aD0iMiI+PHBhdGggZD0iTTQgNmgxNk00IDEyaDE2TTQgMThoMTYi"
    "Lz48L3N2Zz4KICAgICAgPC9idXR0b24+CiAgICAgIDxzdHJvbmcgc3R5bGU9ImZvbnQtZmFtaWx5OnZhcigtLWRpc3ApIj5IZW9ydGg8L3N0cm9uZz4K"
    "ICAgIDwvZGl2PgoKICAgIDwhLS0gPT09PT0gREFTSEJPQVJEID09PT09IC0tPgogICAgPHNlY3Rpb24gY2xhc3M9InZpZXciIGRhdGEtdmlldz0iZGFz"
    "aGJvYXJkIj4KICAgICAgPGRpdiBjbGFzcz0iaGQiPgogICAgICAgIDxkaXY+CiAgICAgICAgICA8ZGl2IGNsYXNzPSJleWVicm93Ij5Zb3VyIG1hY2hp"
    "bmU8L2Rpdj4KICAgICAgICAgIDxoMj5EYXNoYm9hcmQ8L2gyPgogICAgICAgICAgPHAgY2xhc3M9InN1YiI+SGVvcnRoIHNjYW5uZWQgeW91ciBoYXJk"
    "d2FyZSBhbmQgcGlja2VkIG1vZGVscyB0aGF0CiAgICAgICAgICAgIHdpbGwgcnVuIHdlbGwgaGVyZS4gRXZlcnl0aGluZyBydW5zIG9uIHlvdXIgY29t"
    "cHV0ZXIg4oCUIG5vdGhpbmcgbGVhdmVzIGl0LjwvcD4KICAgICAgICA8L2Rpdj4KICAgICAgICA8YnV0dG9uIGNsYXNzPSJidG4gZ2hvc3Qgc20iIGlk"
    "PSJyZXNjYW5CdG4iPgogICAgICAgICAgPHN2ZyB3aWR0aD0iMTQiIGhlaWdodD0iMTQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ry"
    "b2tlPSJjdXJyZW50Q29sb3IiIHN0cm9rZS13aWR0aD0iMiI+PHBhdGggZD0iTTIxIDEyYTkgOSAwIDEgMS0yLjYtNi40TTIxIDN2NWgtNSIvPjwvc3Zn"
    "PgogICAgICAgICAgUmVzY2FuCiAgICAgICAgPC9idXR0b24+CiAgICAgIDwvZGl2PgoKICAgICAgPGRpdiBpZD0idXBkYXRlQmFubmVySG9zdCI+PC9k"
    "aXY+CgogICAgICA8ZGl2IGNsYXNzPSJoZXJvLXBhbmVsIiBpZD0iaGVyb1BhbmVsIj4KICAgICAgICA8ZGl2IGNsYXNzPSJnYXVnZS1yb3ciPgogICAg"
    "ICAgICAgPGRpdiBjbGFzcz0iZ2F1Z2UiIGlkPSJnYXVnZSI+PGRpdiBjbGFzcz0ic3BpbiI+PC9kaXY+PC9kaXY+CiAgICAgICAgICA8ZGl2IHN0eWxl"
    "PSJmbGV4OjEiPgogICAgICAgICAgICA8ZGl2IGNsYXNzPSJzcGVjcyIgaWQ9InNwZWNzIj48L2Rpdj4KICAgICAgICAgIDwvZGl2PgogICAgICAgIDwv"
    "ZGl2PgogICAgICAgIDxkaXYgY2xhc3M9InRpZXItbm90ZSIgaWQ9InRpZXJOb3RlIj5TY2FubmluZ+KApjwvZGl2PgogICAgICA8L2Rpdj4KCiAgICAg"
    "IDxkaXYgY2xhc3M9InNlY3Rpb24tdGl0bGUiPlJlY29tbWVuZGVkIGZvciB5b3U8L2Rpdj4KICAgICAgPGRpdiBjbGFzcz0icmVjbGlzdCIgaWQ9InJl"
    "Y0xpc3QiPjwvZGl2PgogICAgPC9zZWN0aW9uPgoKICAgIDwhLS0gPT09PT0gQ0hBVCA9PT09PSAtLT4KICAgIDxzZWN0aW9uIGNsYXNzPSJ2aWV3IGNo"
    "YXR2aWV3IiBkYXRhLXZpZXc9ImNoYXQiIGhpZGRlbj4KICAgICAgPGRpdiBjbGFzcz0iY2hhdC1oZWFkIj4KICAgICAgICA8ZGl2IGNsYXNzPSJtb2Rl"
    "bHBpY2siPgogICAgICAgICAgPHNwYW4gY2xhc3M9ImRpbSBtb25vIiBzdHlsZT0iZm9udC1zaXplOjExcHgiPk1PREVMPC9zcGFuPgogICAgICAgICAg"
    "PHNlbGVjdCBjbGFzcz0ic2VsIiBpZD0iY2hhdE1vZGVsIj48b3B0aW9uPmxvYWRpbmfigKY8L29wdGlvbj48L3NlbGVjdD4KICAgICAgICA8L2Rpdj4K"
    "ICAgICAgICA8ZGl2IGNsYXNzPSJ0b2dnbGUiIGlkPSJyYWdUb2dnbGUiIHRpdGxlPSJVc2UgeW91ciB1cGxvYWRlZCBkb2N1bWVudHMiPgogICAgICAg"
    "ICAgPHNwYW4gY2xhc3M9InN3Ij48L3NwYW4+IEtub3dsZWRnZQogICAgICAgIDwvZGl2PgogICAgICAgIDxkaXYgY2xhc3M9InRvZ2dsZSB2aW8iIGlk"
    "PSJhZ2VudFRvZ2dsZSIgdGl0bGU9IkxldCB0aGUgbW9kZWwgdXNlIHRvb2xzIj4KICAgICAgICAgIDxzcGFuIGNsYXNzPSJzdyI+PC9zcGFuPiBBZ2Vu"
    "dAogICAgICAgIDwvZGl2PgogICAgICAgIDxkaXYgY2xhc3M9InRvZ2dsZSBibHUiIGlkPSJsb29wVG9nZ2xlIiB0aXRsZT0iQXV0b25vbW91cyBsb29w"
    "IOKAlCB0aGUgYWdlbnQgcGxhbnMsIGFjdHMgYW5kIHJlcGVhdHMgdW50aWwgaXQgY2FsbHMgdGhlIHRhc2sgZG9uZSI+CiAgICAgICAgICA8c3BhbiBj"
    "bGFzcz0ic3ciPjwvc3Bhbj4gTG9vcAogICAgICAgIDwvZGl2PgogICAgICAgIDxkaXYgY2xhc3M9InRvZ2dsZSBncm4iIGlkPSJjb3VuY2lsVG9nZ2xl"
    "IiB0aXRsZT0iQSBwYW5lbCBvZiBjb25zdWx0YW50cyBhbmFseXplcyBpbiBwYXJhbGxlbCwgY3JpdGlxdWVzIGVhY2ggb3RoZXIsIHRoZW4gYSBjaGFp"
    "ciBzeW50aGVzaXplcyB0aGUgYW5zd2VyIj4KICAgICAgICAgIDxzcGFuIGNsYXNzPSJzdyI+PC9zcGFuPiBDb3VuY2lsCiAgICAgICAgPC9kaXY+CiAg"
    "ICAgICAgPGRpdiBzdHlsZT0iZmxleDoxIj48L2Rpdj4KICAgICAgICA8YnV0dG9uIGNsYXNzPSJidG4gc20gZ2hvc3QiIGlkPSJjbGVhckNoYXRCdG4i"
    "Pk5ldyBjaGF0PC9idXR0b24+CiAgICAgIDwvZGl2PgogICAgICA8ZGl2IGNsYXNzPSJjaGF0LWxheW91dCI+CiAgICAgICAgPGRpdiBjbGFzcz0iY2hh"
    "dC1tYWluIj4KICAgICAgICAgIDxkaXYgY2xhc3M9ImNoYXRzY3JvbGwgc2Nyb2xsIiBpZD0iY2hhdFNjcm9sbCI+CiAgICAgICAgICAgIDxkaXYgY2xh"
    "c3M9ImNoYXR3cmFwIiBpZD0iY2hhdFdyYXAiPjwvZGl2PgogICAgICAgICAgPC9kaXY+CiAgICAgICAgICA8ZGl2IGNsYXNzPSJjb21wb3NlciI+CiAg"
    "ICAgICAgICAgIDxkaXYgY2xhc3M9ImNib3giPgogICAgICAgICAgICAgIDx0ZXh0YXJlYSBpZD0iY2hhdElucHV0IiByb3dzPSIxIiBwbGFjZWhvbGRl"
    "cj0iQXNrIGFueXRoaW5n4oCmIChFbnRlciB0byBzZW5kLCBTaGlmdCtFbnRlciBmb3IgYSBuZXcgbGluZSkiPjwvdGV4dGFyZWE+CiAgICAgICAgICAg"
    "ICAgPGJ1dHRvbiBjbGFzcz0ic2VuZGJ0biIgaWQ9InNlbmRCdG4iIHRpdGxlPSJTZW5kIj4KICAgICAgICAgICAgICAgIDxzdmcgd2lkdGg9IjE4IiBo"
    "ZWlnaHQ9IjE4IiB2aWV3Qm94PSIwIDAgMjQgMjQiIGZpbGw9Im5vbmUiIHN0cm9rZT0iY3VycmVudENvbG9yIiBzdHJva2Utd2lkdGg9IjIiPjxwYXRo"
    "IGQ9Ik03IDExIDEyIDZsNSA1TTEyIDZ2MTMiLz48L3N2Zz4KICAgICAgICAgICAgICA8L2J1dHRvbj4KICAgICAgICAgICAgPC9kaXY+CiAgICAgICAg"
    "ICA8L2Rpdj4KICAgICAgICA8L2Rpdj4KICAgICAgICA8ZGl2IGNsYXNzPSJjaGF0LXNpZGUiPgogICAgICAgICAgPGRpdiBjbGFzcz0ibmV3Y2hhdCI+"
    "PGJ1dHRvbiBjbGFzcz0iYnRuIHNtIGdob3N0IiBpZD0ibmV3Q29udkJ0biIgc3R5bGU9IndpZHRoOjEwMCUiPgogICAgICAgICAgICA8c3ZnIHdpZHRo"
    "PSIxNCIgaGVpZ2h0PSIxNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9ImN1cnJlbnRDb2xvciIgc3Ryb2tlLXdpZHRoPSIy"
    "Ij48cGF0aCBkPSJNMTIgNXYxNE01IDEyaDE0Ii8+PC9zdmc+CiAgICAgICAgICAgIE5ldyBjb252ZXJzYXRpb248L2J1dHRvbj48L2Rpdj4KICAgICAg"
    "ICAgIDxkaXYgY2xhc3M9ImNvbnZsaXN0IHNjcm9sbCIgaWQ9ImNvbnZMaXN0Ij48L2Rpdj4KICAgICAgICA8L2Rpdj4KICAgICAgPC9kaXY+CiAgICA8"
    "L3NlY3Rpb24+CgogICAgPCEtLSA9PT09PSBNT0RFTFMgPT09PT0gLS0+CiAgICA8c2VjdGlvbiBjbGFzcz0idmlldyIgZGF0YS12aWV3PSJtb2RlbHMi"
    "IGhpZGRlbj4KICAgICAgPGRpdiBjbGFzcz0iaGQiPgogICAgICAgIDxkaXY+CiAgICAgICAgICA8ZGl2IGNsYXNzPSJleWVicm93Ij5MaWJyYXJ5PC9k"
    "aXY+CiAgICAgICAgICA8aDI+TW9kZWxzPC9oMj4KICAgICAgICAgIDxwIGNsYXNzPSJzdWIiPlNlYXJjaCB0aG91c2FuZHMgb2YgbW9kZWxzIG9yIHBp"
    "Y2sgZnJvbSB0aGUgY3VyYXRlZCBsaXN0LgogICAgICAgICAgICBEb3dubG9hZHMgYXJlIGhhbmRsZWQgYnkgT2xsYW1hIGFuZCBzdG9yZWQgbG9jYWxs"
    "eS48L3A+CiAgICAgICAgPC9kaXY+CiAgICAgIDwvZGl2PgogICAgICA8ZGl2IGNsYXNzPSJzZWFyY2hiYXIiPgogICAgICAgIDxpbnB1dCBjbGFzcz0i"
    "aW5wIiBpZD0ibW9kZWxTZWFyY2giIHBsYWNlaG9sZGVyPSJTZWFyY2ggbW9kZWxzIOKAlCBlLmcuIGxsYW1hLCBxd2VuIGNvZGVyLCB2aXNpb24sIGVt"
    "YmVk4oCmIj4KICAgICAgICA8YnV0dG9uIGNsYXNzPSJidG4gcHJpbWFyeSIgaWQ9Im1vZGVsU2VhcmNoQnRuIj5TZWFyY2g8L2J1dHRvbj4KICAgICAg"
    "PC9kaXY+CiAgICAgIDxkaXYgaWQ9Im1vZGVsc0JvZHkiPjwvZGl2PgogICAgPC9zZWN0aW9uPgoKICAgIDwhLS0gPT09PT0gSU1BR0VTID09PT09IC0t"
    "PgogICAgPHNlY3Rpb24gY2xhc3M9InZpZXciIGRhdGEtdmlldz0iaW1hZ2VzIiBoaWRkZW4+CiAgICAgIDxkaXYgY2xhc3M9ImhkIj4KICAgICAgICA8"
    "ZGl2PgogICAgICAgICAgPGRpdiBjbGFzcz0iZXllYnJvdyI+R2VuZXJhdGU8L2Rpdj4KICAgICAgICAgIDxoMj5JbWFnZXM8L2gyPgogICAgICAgICAg"
    "PHAgY2xhc3M9InN1YiI+Q3JlYXRlIGltYWdlcyBmcm9tIHRleHQgd2l0aCBTdGFibGUgRGlmZnVzaW9uLCBydW5uaW5nCiAgICAgICAgICAgIGxvY2Fs"
    "bHkuIEV2ZXJ5IGltYWdlIGlzIHNhdmVkIHRvIHlvdXIgZ2FsbGVyeSBhdXRvbWF0aWNhbGx5LjwvcD4KICAgICAgICA8L2Rpdj4KICAgICAgPC9kaXY+"
    "CiAgICAgIDxkaXYgaWQ9ImltYWdlc0JvZHkiPjwvZGl2PgogICAgPC9zZWN0aW9uPgoKICAgIDwhLS0gPT09PT0gS05PV0xFREdFID09PT09IC0tPgog"
    "ICAgPHNlY3Rpb24gY2xhc3M9InZpZXciIGRhdGEtdmlldz0ia25vd2xlZGdlIiBoaWRkZW4+CiAgICAgIDxkaXYgY2xhc3M9ImhkIj4KICAgICAgICA8"
    "ZGl2PgogICAgICAgICAgPGRpdiBjbGFzcz0iZXllYnJvdyI+UmV0cmlldmFsPC9kaXY+CiAgICAgICAgICA8aDI+S25vd2xlZGdlIGJhc2U8L2gyPgog"
    "ICAgICAgICAgPHAgY2xhc3M9InN1YiI+QWRkIGRvY3VtZW50cyBhbmQgdGhlIGFzc2lzdGFudCBjYW4gY2l0ZSB0aGVtIGluIENoYXQgd2hlbgogICAg"
    "ICAgICAgICB5b3Ugc3dpdGNoIG9uIEtub3dsZWRnZS4gRmlsZXMgYXJlIGNodW5rZWQgYW5kIGVtYmVkZGVkIGxvY2FsbHkuPC9wPgogICAgICAgIDwv"
    "ZGl2PgogICAgICA8L2Rpdj4KICAgICAgPGRpdiBpZD0ia25vd2xlZGdlQm9keSI+PC9kaXY+CiAgICA8L3NlY3Rpb24+CgogICAgPCEtLSA9PT09PSBU"
    "T09MUyA9PT09PSAtLT4KICAgIDxzZWN0aW9uIGNsYXNzPSJ2aWV3IiBkYXRhLXZpZXc9InRvb2xzIiBoaWRkZW4+CiAgICAgIDxkaXYgY2xhc3M9Imhk"
    "Ij4KICAgICAgICA8ZGl2PgogICAgICAgICAgPGRpdiBjbGFzcz0iZXllYnJvdyI+Q2FwYWJpbGl0aWVzPC9kaXY+CiAgICAgICAgICA8aDI+QWdlbnQg"
    "JmFtcDsgVG9vbHM8L2gyPgogICAgICAgICAgPHAgY2xhc3M9InN1YiI+VGhlIGFnZW50IGNhbiBjYWxsIHRoZXNlIHRvb2xzIHdoaWxlIGl0IGFuc3dl"
    "cnMuIENvbm5lY3QKICAgICAgICAgICAgTUNQIHNlcnZlcnMgdG8gZ2l2ZSBpdCBldmVuIG1vcmUgYWJpbGl0aWVzLjwvcD4KICAgICAgICA8L2Rpdj4K"
    "ICAgICAgPC9kaXY+CiAgICAgIDxkaXYgaWQ9InRvb2xzQm9keSI+PC9kaXY+CiAgICA8L3NlY3Rpb24+CgogICAgPCEtLSA9PT09PSBTRVRUSU5HUyA9"
    "PT09PSAtLT4KICAgIDxzZWN0aW9uIGNsYXNzPSJ2aWV3IiBkYXRhLXZpZXc9InNldHRpbmdzIiBoaWRkZW4+CiAgICAgIDxkaXYgY2xhc3M9ImhkIj4K"
    "ICAgICAgICA8ZGl2PgogICAgICAgICAgPGRpdiBjbGFzcz0iZXllYnJvdyI+Q29uZmlndXJhdGlvbjwvZGl2PgogICAgICAgICAgPGgyPlNldHRpbmdz"
    "PC9oMj4KICAgICAgICAgIDxwIGNsYXNzPSJzdWIiPlR1bmUgYmVoYXZpb3VyLCBtYW5hZ2UgdXBkYXRlcywgYW5kIHNlZSB3aGVyZSB5b3VyIGRhdGEg"
    "bGl2ZXMuPC9wPgogICAgICAgIDwvZGl2PgogICAgICA8L2Rpdj4KICAgICAgPGRpdiBpZD0ic2V0dGluZ3NCb2R5Ij48L2Rpdj4KICAgIDwvc2VjdGlv"
    "bj4KICA8L21haW4+CjwvZGl2PgoKPGRpdiBpZD0idG9hc3RzIj48L2Rpdj4KPGRpdiBpZD0ibW9kYWxIb3N0Ij48L2Rpdj4KPHNjcmlwdD4KLyogPT09"
    "PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09CiAgIEhlb3J0aCBmcm9udGVuZCDigJQgdmFuaWxs"
    "YSBKUywgbm8gYnVpbGQgc3RlcAogICA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT0gKi8K"
    "Y29uc3QgJCA9IChpZCkgPT4gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoaWQpOwpjb25zdCBlbCA9ICh0YWcsIGNscywgaHRtbCkgPT4geyBjb25zdCBl"
    "ID0gZG9jdW1lbnQuY3JlYXRlRWxlbWVudCh0YWcpOwogIGlmIChjbHMpIGUuY2xhc3NOYW1lID0gY2xzOyBpZiAoaHRtbCAhPSBudWxsKSBlLmlubmVy"
    "SFRNTCA9IGh0bWw7IHJldHVybiBlOyB9Owpjb25zdCBlc2MgPSAocykgPT4gKHMgPT0gbnVsbCA/ICIiIDogU3RyaW5nKHMpKS5yZXBsYWNlKC9bJjw+"
    "IiddL2csCiAgYyA9PiAoeycmJzonJmFtcDsnLCc8JzonJmx0OycsJz4nOicmZ3Q7JywnIic6JyZxdW90OycsIiciOicmIzM5Oyd9W2NdKSk7CmNvbnN0"
    "IGZtdEdCID0gKG4pID0+IChuID09IG51bGwgPyAiPyIgOiAobiA+PSAxMCA/IE1hdGgucm91bmQobikgOiBuLnRvRml4ZWQoMSkpKTsKCmNvbnN0IHN0"
    "YXRlID0geyBzeXN0ZW06bnVsbCwgc2V0dGluZ3M6e30sIGluc3RhbGxlZDpbXSwgY3VycmVudENvbnY6bnVsbCwKICBzZW5kaW5nOmZhbHNlLCB1cGRh"
    "dGVJbmZvOm51bGwsIGltZ1ByZXNldHM6W10sIHNlbEltZ1NpemU6bnVsbCB9OwoKLyogLS0tLS0tLS0tLSBBUEkgaGVscGVycyAtLS0tLS0tLS0tICov"
    "CmFzeW5jIGZ1bmN0aW9uIGFwaShwYXRoLCBvcHRzKXsKICBjb25zdCByID0gYXdhaXQgZmV0Y2gocGF0aCwgb3B0cyk7CiAgY29uc3QgY3QgPSByLmhl"
    "YWRlcnMuZ2V0KCJjb250ZW50LXR5cGUiKSB8fCAiIjsKICBpZihjdC5pbmNsdWRlcygiYXBwbGljYXRpb24vanNvbiIpKXsKICAgIGNvbnN0IGogPSBh"
    "d2FpdCByLmpzb24oKTsKICAgIGlmKGogJiYgai5vayA9PT0gZmFsc2UgJiYgai5lcnJvcikgdGhyb3cgbmV3IEVycm9yKGouZXJyb3IpOwogICAgcmV0"
    "dXJuIGo7CiAgfQogIHJldHVybiByOwp9CmFzeW5jIGZ1bmN0aW9uIHBvc3QocGF0aCwgYm9keSl7CiAgcmV0dXJuIGFwaShwYXRoLCB7bWV0aG9kOiJQ"
    "T1NUIiwgaGVhZGVyczp7IkNvbnRlbnQtVHlwZSI6ImFwcGxpY2F0aW9uL2pzb24ifSwKICAgIGJvZHk6IEpTT04uc3RyaW5naWZ5KGJvZHl8fHt9KX0p"
    "Owp9CmFzeW5jIGZ1bmN0aW9uIGRlbChwYXRoKXsgcmV0dXJuIGFwaShwYXRoLCB7bWV0aG9kOiJERUxFVEUifSk7IH0KCi8qIHN0cmVhbSBOREpTT04g"
    "ZnJvbSBhIFBPU1QgZW5kcG9pbnQsIGNhbGxpbmcgb25PYmogZm9yIGVhY2ggcGFyc2VkIGxpbmUgKi8KYXN5bmMgZnVuY3Rpb24gc3RyZWFtTkRKU09O"
    "KHBhdGgsIGJvZHksIG9uT2JqLCBzaWduYWwpewogIGNvbnN0IHIgPSBhd2FpdCBmZXRjaChwYXRoLCB7bWV0aG9kOiJQT1NUIiwgc2lnbmFsLAogICAg"
    "aGVhZGVyczp7IkNvbnRlbnQtVHlwZSI6ImFwcGxpY2F0aW9uL2pzb24ifSwgYm9keTogSlNPTi5zdHJpbmdpZnkoYm9keXx8e30pfSk7CiAgaWYoIXIu"
    "b2speyBsZXQgdD0iIjsgdHJ5e3Q9KGF3YWl0IHIuanNvbigpKS5lcnJvcn1jYXRjaChlKXt0PWF3YWl0IHIudGV4dCgpfQogICAgdGhyb3cgbmV3IEVy"
    "cm9yKHQgfHwgKCJIVFRQICIrci5zdGF0dXMpKTsgfQogIGNvbnN0IHJlYWRlciA9IHIuYm9keS5nZXRSZWFkZXIoKTsgY29uc3QgZGVjID0gbmV3IFRl"
    "eHREZWNvZGVyKCk7IGxldCBidWY9IiI7CiAgd2hpbGUodHJ1ZSl7CiAgICBjb25zdCB7ZG9uZSwgdmFsdWV9ID0gYXdhaXQgcmVhZGVyLnJlYWQoKTsg"
    "aWYoZG9uZSkgYnJlYWs7CiAgICBidWYgKz0gZGVjLmRlY29kZSh2YWx1ZSwge3N0cmVhbTp0cnVlfSk7IGxldCBubDsKICAgIHdoaWxlKChubCA9IGJ1"
    "Zi5pbmRleE9mKCJcbiIpKSA+PSAwKXsKICAgICAgY29uc3QgbGluZSA9IGJ1Zi5zbGljZSgwLCBubCkudHJpbSgpOyBidWYgPSBidWYuc2xpY2Uobmwr"
    "MSk7CiAgICAgIGlmKGxpbmUpeyB0cnl7IG9uT2JqKEpTT04ucGFyc2UobGluZSkpOyB9Y2F0Y2goZSl7fSB9CiAgICB9CiAgfQogIGlmKGJ1Zi50cmlt"
    "KCkpeyB0cnl7IG9uT2JqKEpTT04ucGFyc2UoYnVmLnRyaW0oKSkpOyB9Y2F0Y2goZSl7fSB9Cn0KCi8qIC0tLS0tLS0tLS0gdG9hc3QgKyBtb2RhbCAt"
    "LS0tLS0tLS0tICovCmZ1bmN0aW9uIHRvYXN0KG1zZywga2luZCl7CiAgY29uc3QgdCA9IGVsKCJkaXYiLCAidG9hc3QiICsgKGtpbmQgPyAiICIra2lu"
    "ZCA6ICIiKSwgZXNjKG1zZykpOwogICQoInRvYXN0cyIpLmFwcGVuZENoaWxkKHQpOwogIHNldFRpbWVvdXQoKCk9PnsgdC5zdHlsZS5vcGFjaXR5PSIw"
    "IjsgdC5zdHlsZS50cmFuc2Zvcm09InRyYW5zbGF0ZVgoMjBweCkiOwogICAgdC5zdHlsZS50cmFuc2l0aW9uPSIuM3MiOyBzZXRUaW1lb3V0KCgpPT50"
    "LnJlbW92ZSgpLCAzMDApOyB9LCA0MjAwKTsKfQpmdW5jdGlvbiBtb2RhbCh7dGl0bGUsIGJvZHlIVE1MLCBhY3Rpb25zLCB3aWRlfSl7CiAgY2xvc2VN"
    "b2RhbCgpOwogIGNvbnN0IGJnID0gZWwoImRpdiIsIm1vZGFsLWJnIik7IGJnLmlkPSJhY3RpdmVNb2RhbCI7CiAgY29uc3QgbSA9IGVsKCJkaXYiLCJt"
    "b2RhbCIrKHdpZGU/IiB3aWRlIjoiIikpOwogIG0uYXBwZW5kQ2hpbGQoZWwoImRpdiIsIm1oIiwgYDxoMz4ke2VzYyh0aXRsZSl9PC9oMz5gKSk7CiAg"
    "Y29uc3QgYnRuWCA9IGVsKCJidXR0b24iLCJidG4gaWNvbiBnaG9zdCIsCiAgICAnPHN2ZyB3aWR0aD0iMTgiIGhlaWdodD0iMTgiIHZpZXdCb3g9IjAg"
    "MCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJjdXJyZW50Q29sb3IiIHN0cm9rZS13aWR0aD0iMiI+PHBhdGggZD0iTTE4IDYgNiAxOE02IDZsMTIg"
    "MTIiLz48L3N2Zz4nKTsKICBidG5YLm9uY2xpY2sgPSBjbG9zZU1vZGFsOyBtLnF1ZXJ5U2VsZWN0b3IoIi5taCIpLmFwcGVuZENoaWxkKGJ0blgpOwog"
    "IGNvbnN0IGJvZHkgPSBlbCgiZGl2IiwibWIiKTsgYm9keS5pbm5lckhUTUwgPSBib2R5SFRNTDsgbS5hcHBlbmRDaGlsZChib2R5KTsKICBpZihhY3Rp"
    "b25zICYmIGFjdGlvbnMubGVuZ3RoKXsKICAgIGNvbnN0IG1mID0gZWwoImRpdiIsIm1mIik7CiAgICBhY3Rpb25zLmZvckVhY2goYT0+eyBjb25zdCBi"
    "ID0gZWwoImJ1dHRvbiIsImJ0biAiKyhhLmNsc3x8IiIpLCBlc2MoYS5sYWJlbCkpOwogICAgICBiLm9uY2xpY2sgPSAoKT0+YS5vbkNsaWNrICYmIGEu"
    "b25DbGljayhib2R5KTsgbWYuYXBwZW5kQ2hpbGQoYik7IH0pOwogICAgbS5hcHBlbmRDaGlsZChtZik7CiAgfQogIGJnLmFwcGVuZENoaWxkKG0pOyBi"
    "Zy5vbmNsaWNrID0gKGUpPT57IGlmKGUudGFyZ2V0PT09YmcpIGNsb3NlTW9kYWwoKTsgfTsKICAkKCJtb2RhbEhvc3QiKS5hcHBlbmRDaGlsZChiZyk7"
    "IHJldHVybiBib2R5Owp9CmZ1bmN0aW9uIGNsb3NlTW9kYWwoKXsgY29uc3QgbSA9ICQoImFjdGl2ZU1vZGFsIik7IGlmKG0pIG0ucmVtb3ZlKCk7IH0K"
    "ZG9jdW1lbnQuYWRkRXZlbnRMaXN0ZW5lcigia2V5ZG93biIsIGU9PnsgaWYoZS5rZXk9PT0iRXNjYXBlIikgY2xvc2VNb2RhbCgpOyB9KTsKCi8qIC0t"
    "LS0tLS0tLS0gbGlnaHR3ZWlnaHQgbWFya2Rvd24gLS0tLS0tLS0tLSAqLwpmdW5jdGlvbiBtZChzcmMpewogIGlmKCFzcmMpIHJldHVybiAiIjsKICBj"
    "b25zdCBibG9ja3MgPSBbXTsgLy8gc3Rhc2ggY29kZSBmZW5jZXMKICBsZXQgcyA9IHNyYy5yZXBsYWNlKC9gYGAoXHcqKVxuPyhbXHNcU10qPylgYGAv"
    "ZywgKG0sIGxhbmcsIGNvZGUpPT57CiAgICBibG9ja3MucHVzaChgPHByZT48Y29kZT4ke2VzYyhjb2RlLnJlcGxhY2UoL1xuJC8sIiIpKX08L2NvZGU+"
    "PC9wcmU+YCk7CiAgICByZXR1cm4gYFx1MDAwMCR7YmxvY2tzLmxlbmd0aC0xfVx1MDAwMGA7IH0pOwogIHMgPSBlc2Mocyk7CiAgLy8gaW1hZ2VzIHRo"
    "ZW4gbGlua3MKICBzID0gcy5yZXBsYWNlKC8hXFsoW15cXV0qKVxdXCgoW14pXHNdKylcKS9nLAogICAgICAobSxhLHUpPT5gPGltZyBhbHQ9IiR7YX0i"
    "IHNyYz0iJHt1fSI+YCk7CiAgcyA9IHMucmVwbGFjZSgvXFsoW15cXV0rKVxdXCgoW14pXHNdKylcKS9nLAogICAgICAobSx0LHUpPT5gPGEgaHJlZj0i"
    "JHt1fSIgdGFyZ2V0PSJfYmxhbmsiIHJlbD0ibm9vcGVuZXIiPiR7dH08L2E+YCk7CiAgcyA9IHMucmVwbGFjZSgvYChbXmBdKylgL2csIChtLGMpPT5g"
    "PGNvZGU+JHtjfTwvY29kZT5gKTsKICBzID0gcy5yZXBsYWNlKC9cKlwqKFteKl0rKVwqXCovZywgIjxzdHJvbmc+JDE8L3N0cm9uZz4iKTsKICBzID0g"
    "cy5yZXBsYWNlKC8oXnxbXipdKVwqKFteKlxuXSspXCovZywgIiQxPGVtPiQyPC9lbT4iKTsKICAvLyBoZWFkaW5ncwogIHMgPSBzLnJlcGxhY2UoL14j"
    "IyNccysoLiopJC9nbSwgIjxoMz4kMTwvaDM+IikKICAgICAgIC5yZXBsYWNlKC9eIyNccysoLiopJC9nbSwgIjxoMj4kMTwvaDI+IikKICAgICAgIC5y"
    "ZXBsYWNlKC9eI1xzKyguKikkL2dtLCAiPGgxPiQxPC9oMT4iKTsKICAvLyBsaXN0cwogIHMgPSBzLnJlcGxhY2UoLyg/Ol58XG4pKCg/OlxzKlstKl1c"
    "cysuKig/OlxufCQpKSspL2csIChtLCBsaXN0KT0+ewogICAgY29uc3QgaXRlbXMgPSBsaXN0LnRyaW0oKS5zcGxpdCgiXG4iKS5tYXAobD0+CiAgICAg"
    "ICI8bGk+IitsLnJlcGxhY2UoL15ccypbLSpdXHMrLywiIikrIjwvbGk+Iikuam9pbigiIik7IHJldHVybiAiXG48dWw+IitpdGVtcysiPC91bD4iOyB9"
    "KTsKICBzID0gcy5yZXBsYWNlKC8oPzpefFxuKSgoPzpccypcZCtcLlxzKy4qKD86XG58JCkpKykvZywgKG0sIGxpc3QpPT57CiAgICBjb25zdCBpdGVt"
    "cyA9IGxpc3QudHJpbSgpLnNwbGl0KCJcbiIpLm1hcChsPT4KICAgICAgIjxsaT4iK2wucmVwbGFjZSgvXlxzKlxkK1wuXHMrLywiIikrIjwvbGk+Iiku"
    "am9pbigiIik7IHJldHVybiAiXG48b2w+IitpdGVtcysiPC9vbD4iOyB9KTsKICAvLyBwYXJhZ3JhcGhzCiAgcyA9IHMuc3BsaXQoL1xuezIsfS8pLm1h"
    "cChwPT57CiAgICBwID0gcC50cmltKCk7IGlmKCFwKSByZXR1cm4gIiI7CiAgICBpZigvXjwoaFxkfHVsfG9sfHByZXx0YWJsZXxpbWd8YmxvY2txdW90"
    "ZSkvLnRlc3QocCkpIHJldHVybiBwOwogICAgcmV0dXJuICI8cD4iK3AucmVwbGFjZSgvXG4vZywiPGJyPiIpKyI8L3A+IjsKICB9KS5qb2luKCJcbiIp"
    "OwogIHMgPSBzLnJlcGxhY2UoL1x1MDAwMChcZCspXHUwMDAwL2csIChtLGkpPT5ibG9ja3NbK2ldKTsKICByZXR1cm4gczsKfQoKLyogLS0tLS0tLS0t"
    "LSBuYXZpZ2F0aW9uIC0tLS0tLS0tLS0gKi8KY29uc3QgbG9hZGVycyA9IHt9OwpsZXQgY3VycmVudFZpZXcgPSAiZGFzaGJvYXJkIjsKZnVuY3Rpb24g"
    "c2hvdyh2aWV3KXsKICBjdXJyZW50VmlldyA9IHZpZXc7CiAgZG9jdW1lbnQucXVlcnlTZWxlY3RvckFsbCgnLm5hdiBidXR0b25bZGF0YS12aWV3XScp"
    "LmZvckVhY2goYj0+CiAgICBiLmNsYXNzTGlzdC50b2dnbGUoImFjdGl2ZSIsIGIuZGF0YXNldC52aWV3PT09dmlldykpOwogIGRvY3VtZW50LnF1ZXJ5"
    "U2VsZWN0b3JBbGwoJy52aWV3W2RhdGEtdmlld10nKS5mb3JFYWNoKHM9PgogICAgcy5oaWRkZW4gPSBzLmRhdGFzZXQudmlldyE9PXZpZXcpOwogICQo"
    "InNpZGUiKS5jbGFzc0xpc3QucmVtb3ZlKCJvcGVuIik7CiAgaWYobG9hZGVyc1t2aWV3XSkgbG9hZGVyc1t2aWV3XSgpOwp9CmRvY3VtZW50LnF1ZXJ5"
    "U2VsZWN0b3JBbGwoJy5uYXYgYnV0dG9uW2RhdGEtdmlld10nKS5mb3JFYWNoKGI9PgogIGIub25jbGljayA9ICgpPT5zaG93KGIuZGF0YXNldC52aWV3"
    "KSk7CiQoIm1lbnVCdG4iKS5vbmNsaWNrID0gKCk9PiAkKCJzaWRlIikuY2xhc3NMaXN0LnRvZ2dsZSgib3BlbiIpOwoKLyogLS0tLS0tLS0tLSB0aGVt"
    "ZSAtLS0tLS0tLS0tICovCmZ1bmN0aW9uIGFwcGx5VGhlbWUodCl7CiAgZG9jdW1lbnQuZG9jdW1lbnRFbGVtZW50LmRhdGFzZXQudGhlbWUgPSB0Owog"
    "ICQoInRoZW1lTGFiZWwiKS50ZXh0Q29udGVudCA9IHQ9PT0iZGFyayIgPyAiTGlnaHQgbW9kZSIgOiAiRGFyayBtb2RlIjsKfQokKCJ0aGVtZUJ0biIp"
    "Lm9uY2xpY2sgPSBhc3luYyAoKT0+ewogIGNvbnN0IG5leHQgPSBkb2N1bWVudC5kb2N1bWVudEVsZW1lbnQuZGF0YXNldC50aGVtZT09PSJkYXJrIiA/"
    "ICJsaWdodCI6ImRhcmsiOwogIGFwcGx5VGhlbWUobmV4dCk7IHN0YXRlLnNldHRpbmdzLnRoZW1lID0gbmV4dDsKICB0cnl7IGF3YWl0IHBvc3QoIi9h"
    "cGkvc2V0dGluZ3MiLCB7dGhlbWU6IG5leHR9KTsgfWNhdGNoKGUpe30KfTsKCi8qID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09"
    "PT09PT09PT09PT09PT09PT09PT09PQogICBEQVNIQk9BUkQKICAgPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09"
    "PT09PT09PT09PT09ICovCmZ1bmN0aW9uIGRvbnV0KHBjdCwgY2VudGVyQmlnLCBjZW50ZXJTbWFsbCl7CiAgY29uc3QgUiA9IDUyLCBDID0gMipNYXRo"
    "LlBJKlIsIG9mZiA9IEMqKDEgLSBNYXRoLm1heCgwLE1hdGgubWluKDEscGN0KSkpOwogIHJldHVybiBgPHN2ZyB3aWR0aD0iMTMyIiBoZWlnaHQ9IjEz"
    "MiIgdmlld0JveD0iMCAwIDEzMiAxMzIiPgogICAgPGNpcmNsZSBjeD0iNjYiIGN5PSI2NiIgcj0iJHtSfSIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ2YXIo"
    "LS1saW5lKSIgc3Ryb2tlLXdpZHRoPSIxMSIvPgogICAgPGNpcmNsZSBjeD0iNjYiIGN5PSI2NiIgcj0iJHtSfSIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ2"
    "YXIoLS1zaWduYWwpIiBzdHJva2Utd2lkdGg9IjExIgogICAgICBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1kYXNoYXJyYXk9IiR7Q30iIHN0"
    "cm9rZS1kYXNob2Zmc2V0PSIke29mZn0iCiAgICAgIHN0eWxlPSJ0cmFuc2l0aW9uOnN0cm9rZS1kYXNob2Zmc2V0IC44cyBlYXNlIi8+CiAgPC9zdmc+"
    "PGRpdiBjbGFzcz0ibGJsIj48Yj4ke2NlbnRlckJpZ308L2I+PHNwYW4+JHtjZW50ZXJTbWFsbH08L3NwYW4+PC9kaXY+YDsKfQoKYXN5bmMgZnVuY3Rp"
    "b24gbG9hZERhc2hib2FyZCgpewogIHRyeXsKICAgIGNvbnN0IGQgPSBhd2FpdCBhcGkoIi9hcGkvc3lzdGVtIik7CiAgICBzdGF0ZS5zeXN0ZW0gPSBk"
    "OyByZW5kZXJTeXN0ZW0oZCk7CiAgfWNhdGNoKGUpeyAkKCJ0aWVyTm90ZSIpLnRleHRDb250ZW50ID0gIkNvdWxkIG5vdCByZWFkIHN5c3RlbSBpbmZv"
    "OiAiK2UubWVzc2FnZTsgfQp9CmZ1bmN0aW9uIHJlbmRlclN5c3RlbShkKXsKICBjb25zdCBodyA9IGQuaGFyZHdhcmUsIHJlYyA9IGQucmVjb21tZW5k"
    "YXRpb247CiAgJCgiYnJhbmRWZXIiKS50ZXh0Q29udGVudCA9ICJ2IitkLnZlcnNpb247CiAgLy8gZ2F1Z2U6IHVzYWJsZSBtZW1vcnkgYXMgc2hhcmUg"
    "b2YgdG90YWwgUkFNCiAgY29uc3QgcGN0ID0gaHcucmFtX2diID8gaHcudXNhYmxlX2diIC8gaHcucmFtX2diIDogMDsKICAkKCJnYXVnZSIpLmlubmVy"
    "SFRNTCA9IGRvbnV0KHBjdCwgZm10R0IoaHcudXNhYmxlX2diKSsiPHNtYWxsIHN0eWxlPSdmb250LXNpemU6MTNweCc+R0I8L3NtYWxsPiIsCiAgICAi"
    "dXNhYmxlIik7CiAgY29uc3QgZ3B1ID0gaHcuZ3B1cyAmJiBody5ncHVzLmxlbmd0aCA/IGh3LmdwdXNbMF0gOiBudWxsOwogIGNvbnN0IHBsdXJhbCA9"
    "IChuLHcpPT4gbisiICIrdysobj09PTE/IiI6InMiKTsKICBjb25zdCBzcGVjcyA9IFsKICAgIFsiU3lzdGVtIiwgaHcub3NfcHJldHR5LCBody5hcmNo"
    "XSwKICAgIFsiTWVtb3J5IiwgZm10R0IoaHcucmFtX2diKSsiIEdCIiwgZm10R0IoaHcucmFtX3VzZWRfZ2IpKyIgR0IgaW4gdXNlIl0sCiAgICBbIlBy"
    "b2Nlc3NvciIsIHBsdXJhbChody5jcHVfY29yZXN8fDAsImNvcmUiKSwgcGx1cmFsKGh3LmNwdV90aHJlYWRzfHwwLCJ0aHJlYWQiKV0sCiAgICBbIkdy"
    "YXBoaWNzIiwgZ3B1ID8gKGdwdS5raW5kPT09ImFwcGxlIj8iQXBwbGUgR1BVIjpncHUubmFtZS5zbGljZSgwLDIyKSkgOiAiQ1BVIG9ubHkiLAogICAg"
    "ICAgZ3B1ID8gZm10R0IoZ3B1LnZyYW1fZ2IpKyIgR0IgIisoZ3B1LmtpbmQ9PT0iYXBwbGUiPyJ1bmlmaWVkIjoiVlJBTSIpIDogaHcuYmFja2VuZF0s"
    "CiAgICBbIkZyZWUgZGlzayIsIGZtdEdCKGh3LmRpc2tfZnJlZV9nYikrIiBHQiIsICJmb3IgbW9kZWxzIl0sCiAgICBbIkJhY2tlbmQiLCBody5iYWNr"
    "ZW5kLnRvVXBwZXJDYXNlKCksIGh3LmFwcGxlX3NpbGljb24/Ik1ldGFsIjoiYWNjZWxlcmF0aW9uIl0sCiAgXTsKICAkKCJzcGVjcyIpLmlubmVySFRN"
    "TCA9IHNwZWNzLm1hcCgoW2ssdixzXSk9PgogICAgYDxkaXYgY2xhc3M9InNwZWMiPjxkaXYgY2xhc3M9ImsiPiR7a308L2Rpdj48ZGl2IGNsYXNzPSJ2"
    "Ij4ke2VzYyh2KX0KICAgICA8c21hbGw+JHtlc2Moc3x8IiIpfTwvc21hbGw+PC9kaXY+PC9kaXY+YCkuam9pbigiIik7CiAgJCgidGllck5vdGUiKS5p"
    "bm5lckhUTUwgPSAiPGI+Iitlc2MocmVjLnRpZXIuc3BsaXQoIuKAlCIpWzBdLnRyaW0oKSkrIjwvYj4g4oCUICIrCiAgICBlc2MocmVjLnRpZXIuc3Bs"
    "aXQoIuKAlCIpLnNsaWNlKDEpLmpvaW4oIuKAlCIpLnRyaW0oKSB8fCByZWMudGllcik7CgogIGNvbnN0IGluc3RhbGxlZE5hbWVzID0gbmV3IFNldChz"
    "dGF0ZS5pbnN0YWxsZWQubWFwKG09Pm0ubmFtZSkpOwogIGNvbnN0IHBpY2tzID0gcmVjLnBpY2tzIHx8IFtdOwogICQoInJlY0xpc3QiKS5pbm5lckhU"
    "TUwgPSBwaWNrcy5sZW5ndGggPyAiIiA6CiAgICAiPGRpdiBjbGFzcz0nbXV0ZWQnPk5vIG1vZGVscyBmaXQgdGhlIGRldGVjdGVkIG1lbW9yeS4gVHJ5"
    "IHRoZSBNb2RlbHMgcGFnZS48L2Rpdj4iOwogIHBpY2tzLmZvckVhY2goKG0saSk9PnsKICAgIGNvbnN0IGhhdmUgPSBpbnN0YWxsZWROYW1lcy5oYXMo"
    "bS5pZCk7CiAgICBjb25zdCByb3cgPSBlbCgiZGl2IiwicmVjaXRlbSIrKGk9PT0wPyIgYmVzdCI6IiIpKTsKICAgIHJvdy5pbm5lckhUTUwgPSBgPGRp"
    "diBjbGFzcz0icmFuayI+JHtTdHJpbmcoaSsxKS5wYWRTdGFydCgyLCIwIil9PC9kaXY+CiAgICAgIDxkaXYgY2xhc3M9ImluZm8iPjxkaXYgY2xhc3M9"
    "Im5tIj4ke2VzYyhtLm5hbWUpfQogICAgICAgICR7aT09PTA/JzxzcGFuIGNsYXNzPSJjaGlwIGhsIj5iZXN0IGZpdDwvc3Bhbj4nOicnfQogICAgICAg"
    "ICR7bS50YWdzLnNsaWNlKDAsMikubWFwKHQ9PmA8c3BhbiBjbGFzcz0iY2hpcCI+JHt0fTwvc3Bhbj5gKS5qb2luKCIiKX08L2Rpdj4KICAgICAgICA8"
    "ZGl2IGNsYXNzPSJkcyI+JHtlc2MobS5kZXNjKX08L2Rpdj48L2Rpdj4KICAgICAgPGRpdiBjbGFzcz0ic2l6ZSI+JHtmbXRHQihtLnNpemVfZ2IpfSBH"
    "Qjxicj48c3BhbiBjbGFzcz0iZGltIj5+JHtmbXRHQihtLm5lZWRfZ2IpfSBHQiByYW08L3NwYW4+PC9kaXY+YDsKICAgIGNvbnN0IGFjdCA9IGVsKCJk"
    "aXYiKTsgYWN0LnN0eWxlLmZsZXg9Im5vbmUiOwogICAgaWYoaGF2ZSl7IGFjdC5pbm5lckhUTUwgPSAnPHNwYW4gY2xhc3M9Imluc3RhbGxlZC1iYWRn"
    "ZSI+4pyTIGluc3RhbGxlZDwvc3Bhbj4nOyB9CiAgICBlbHNleyBjb25zdCBiID0gZWwoImJ1dHRvbiIsImJ0biBwcmltYXJ5IHNtIiwiRG93bmxvYWQi"
    "KTsKICAgICAgYi5vbmNsaWNrID0gKCk9PnB1bGxNb2RlbChtLmlkLCBiKTsgYWN0LmFwcGVuZENoaWxkKGIpOyB9CiAgICByb3cuYXBwZW5kQ2hpbGQo"
    "YWN0KTsgJCgicmVjTGlzdCIpLmFwcGVuZENoaWxkKHJvdyk7CiAgfSk7Cn0KJCgicmVzY2FuQnRuIikub25jbGljayA9ICgpPT57ICQoImdhdWdlIiku"
    "aW5uZXJIVE1MPSc8ZGl2IGNsYXNzPSJzcGluIj48L2Rpdj4nOwogIGxvYWREYXNoYm9hcmQoKTsgdG9hc3QoIlJlc2Nhbm5pbmcgaGFyZHdhcmXigKYi"
    "KTsgfTsKbG9hZGVycy5kYXNoYm9hcmQgPSBsb2FkRGFzaGJvYXJkOwoKLyogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09"
    "PT09PT09PT09PT09PT09PT09CiAgIE1PREVMUwogICA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09"
    "PT09PT0gKi8KYXN5bmMgZnVuY3Rpb24gbG9hZE1vZGVscygpewogIGNvbnN0IGhvc3QgPSAkKCJtb2RlbHNCb2R5Iik7CiAgaG9zdC5pbm5lckhUTUwg"
    "PSAnPGRpdiBjbGFzcz0icm93IiBzdHlsZT0iZ2FwOjlweCI+PGRpdiBjbGFzcz0ic3BpbiI+PC9kaXY+JysKICAgICc8c3BhbiBjbGFzcz0ibXV0ZWQi"
    "PkxvYWRpbmcgbW9kZWxz4oCmPC9zcGFuPjwvZGl2Pic7CiAgbGV0IGluc3Q7CiAgdHJ5eyBpbnN0ID0gYXdhaXQgYXBpKCIvYXBpL21vZGVscy9pbnN0"
    "YWxsZWQiKTsgfQogIGNhdGNoKGUpeyBob3N0LmlubmVySFRNTCA9IGVyckNhcmQoZS5tZXNzYWdlKTsgcmV0dXJuOyB9CiAgc3RhdGUuaW5zdGFsbGVk"
    "ID0gaW5zdC5tb2RlbHMgfHwgW107CiAgbGV0IGh0bWwgPSAiIjsKICBpZighaW5zdC5vbGxhbWEudXApewogICAgaHRtbCArPSBvbGxhbWFTZXR1cENh"
    "cmQoKTsKICB9CiAgLy8gaW5zdGFsbGVkCiAgaHRtbCArPSBgPGRpdiBjbGFzcz0ic2VjdGlvbi10aXRsZSI+SW5zdGFsbGVkJHtpbnN0Lm9sbGFtYS51"
    "cD8KICAgICIgwrcgIitzdGF0ZS5pbnN0YWxsZWQubGVuZ3RoOiIifTwvZGl2PmA7CiAgaWYoc3RhdGUuaW5zdGFsbGVkLmxlbmd0aCl7CiAgICBodG1s"
    "ICs9ICc8ZGl2IGNsYXNzPSJtZ3JpZCIgaWQ9Imluc3RHcmlkIj48L2Rpdj4nOwogIH0gZWxzZSBpZihpbnN0Lm9sbGFtYS51cCl7CiAgICBodG1sICs9"
    "ICc8ZGl2IGNsYXNzPSJlbXB0eSI+PGRpdiBjbGFzcz0iYmlnIj7il7U8L2Rpdj5ObyBtb2RlbHMgeWV0IOKAlCBkb3dubG9hZCBvbmUgYmVsb3cgb3Ig"
    "ZnJvbSB0aGUgRGFzaGJvYXJkLjwvZGl2Pic7CiAgfQogIGh0bWwgKz0gJzxkaXYgY2xhc3M9InNlY3Rpb24tdGl0bGUiPkN1cmF0ZWQgZm9yIHlvdXIg"
    "aGFyZHdhcmU8L2Rpdj48ZGl2IGNsYXNzPSJtZ3JpZCIgaWQ9ImNhdEdyaWQiPjwvZGl2Pic7CiAgaHRtbCArPSAnPGRpdiBpZD0ic2VhcmNoUmVzdWx0"
    "cyI+PC9kaXY+JzsKICBob3N0LmlubmVySFRNTCA9IGh0bWw7CgogIGlmKHN0YXRlLmluc3RhbGxlZC5sZW5ndGgpewogICAgY29uc3QgZyA9ICQoImlu"
    "c3RHcmlkIik7CiAgICBzdGF0ZS5pbnN0YWxsZWQuZm9yRWFjaChtPT57CiAgICAgIGNvbnN0IGMgPSBlbCgiZGl2IiwibWNhcmQiKTsKICAgICAgYy5p"
    "bm5lckhUTUwgPSBgPGRpdiBjbGFzcz0idG9wIj48ZGl2PjxkaXYgY2xhc3M9Im5tIj4ke2VzYyhtLm5hbWUpfTwvZGl2PgogICAgICAgIDxkaXYgY2xh"
    "c3M9ImlkIj4ke2VzYyhbbS5wYXJhbXMsbS5xdWFudCxtLmZhbWlseV0uZmlsdGVyKEJvb2xlYW4pLmpvaW4oIiDCtyAiKXx8Im1vZGVsIil9PC9kaXY+"
    "PC9kaXY+PC9kaXY+CiAgICAgICAgPGRpdiBjbGFzcz0iZm9vdCI+PHNwYW4gY2xhc3M9Imluc3RhbGxlZC1iYWRnZSI+4pyTICR7Zm10R0IobS5zaXpl"
    "X2diKX0gR0Igb24gZGlzazwvc3Bhbj48L2Rpdj5gOwogICAgICBjb25zdCBidG4gPSBlbCgiYnV0dG9uIiwiYnRuIGRhbmdlciBzbSIsIlJlbW92ZSIp"
    "OwogICAgICBidG4ub25jbGljayA9ICgpPT5yZW1vdmVNb2RlbChtLm5hbWUsIGMpOwogICAgICBjLnF1ZXJ5U2VsZWN0b3IoIi5mb290IikuYXBwZW5k"
    "Q2hpbGQoYnRuKTsgZy5hcHBlbmRDaGlsZChjKTsKICAgIH0pOwogIH0KICAvLyBjdXJhdGVkIGNhdGFsb2cKICB0cnl7CiAgICBjb25zdCBjYXQgPSBh"
    "d2FpdCBhcGkoIi9hcGkvbW9kZWxzL2NhdGFsb2ciKTsKICAgIGNvbnN0IGluc3RhbGxlZE5hbWVzID0gbmV3IFNldChzdGF0ZS5pbnN0YWxsZWQubWFw"
    "KG09Pm0ubmFtZSkpOwogICAgY29uc3QgZml0ID0gKGNhdC5jYXRhbG9nfHxbXSkuZmlsdGVyKG09PgogICAgICBtLm5lZWRfZ2IgPD0gY2F0LnJlY29t"
    "bWVuZGF0aW9uLnVzYWJsZV9nYiArIDAuNSk7CiAgICByZW5kZXJNb2RlbENhcmRzKCQoImNhdEdyaWQiKSwgZml0LCBpbnN0YWxsZWROYW1lcyk7CiAg"
    "fWNhdGNoKGUpe30KfQpmdW5jdGlvbiByZW5kZXJNb2RlbENhcmRzKGdyaWQsIG1vZGVscywgaW5zdGFsbGVkTmFtZXMpewogIGdyaWQuaW5uZXJIVE1M"
    "ID0gIiI7CiAgaWYoIW1vZGVscy5sZW5ndGgpeyBncmlkLmlubmVySFRNTCA9ICc8ZGl2IGNsYXNzPSJtdXRlZCI+Tm90aGluZyB0byBzaG93LjwvZGl2"
    "Pic7IHJldHVybjsgfQogIG1vZGVscy5mb3JFYWNoKG09PnsKICAgIGNvbnN0IGhhdmUgPSBpbnN0YWxsZWROYW1lcy5oYXMobS5pZCk7CiAgICBjb25z"
    "dCBjID0gZWwoImRpdiIsIm1jYXJkIik7CiAgICBjLmlubmVySFRNTCA9IGA8ZGl2IGNsYXNzPSJ0b3AiPjxkaXY+PGRpdiBjbGFzcz0ibm0iPiR7ZXNj"
    "KG0ubmFtZSl9PC9kaXY+CiAgICAgIDxkaXYgY2xhc3M9ImlkIj4ke2VzYyhtLmlkKX08L2Rpdj48L2Rpdj4KICAgICAgPGRpdiBjbGFzcz0idGFncyI+"
    "JHsobS50YWdzfHxbXSkubWFwKHQ9PmA8c3BhbiBjbGFzcz0iY2hpcCI+JHt0fTwvc3Bhbj5gKS5qb2luKCIiKX08L2Rpdj48L2Rpdj4KICAgICAgPGRp"
    "diBjbGFzcz0iZHMiPiR7ZXNjKG0uZGVzY3x8IiIpfTwvZGl2PgogICAgICA8ZGl2IGNsYXNzPSJmb290Ij48c3BhbiBjbGFzcz0iZGltIG1vbm8iIHN0"
    "eWxlPSJmb250LXNpemU6MTFweCI+CiAgICAgICAgJHttLnNpemVfZ2I/Zm10R0IobS5zaXplX2diKSsiIEdCIjoiIn0ke20uZG93bmxvYWRzPyLihpMg"
    "IittLmRvd25sb2Fkcy50b0xvY2FsZVN0cmluZygpOiIifTwvc3Bhbj48L2Rpdj5gOwogICAgY29uc3QgZm9vdCA9IGMucXVlcnlTZWxlY3RvcigiLmZv"
    "b3QiKTsKICAgIGlmKGhhdmUpeyBmb290LmFwcGVuZENoaWxkKGVsKCJzcGFuIiwiaW5zdGFsbGVkLWJhZGdlIiwi4pyTIGluc3RhbGxlZCIpKTsgfQog"
    "ICAgZWxzZXsgY29uc3QgYiA9IGVsKCJidXR0b24iLCJidG4gcHJpbWFyeSBzbSIsIkRvd25sb2FkIik7CiAgICAgIGIub25jbGljayA9ICgpPT5wdWxs"
    "TW9kZWwobS5pZCwgYiwgYyk7IGZvb3QuYXBwZW5kQ2hpbGQoYik7IH0KICAgIGdyaWQuYXBwZW5kQ2hpbGQoYyk7CiAgfSk7Cn0KYXN5bmMgZnVuY3Rp"
    "b24gZG9Nb2RlbFNlYXJjaCgpewogIGNvbnN0IHF2ID0gJCgibW9kZWxTZWFyY2giKS52YWx1ZS50cmltKCk7IGlmKCFxdikgcmV0dXJuOwogIGNvbnN0"
    "IGJveCA9ICQoInNlYXJjaFJlc3VsdHMiKTsKICBib3guaW5uZXJIVE1MID0gJzxkaXYgY2xhc3M9InNlY3Rpb24tdGl0bGUiPlNlYXJjaCByZXN1bHRz"
    "PC9kaXY+JysKICAgICc8ZGl2IGNsYXNzPSJyb3ciPjxkaXYgY2xhc3M9InNwaW4iPjwvZGl2PjxzcGFuIGNsYXNzPSJtdXRlZCI+U2VhcmNoaW5n4oCm"
    "PC9zcGFuPjwvZGl2Pic7CiAgdHJ5ewogICAgY29uc3QgciA9IGF3YWl0IGFwaSgiL2FwaS9tb2RlbHMvc2VhcmNoP3E9IitlbmNvZGVVUklDb21wb25l"
    "bnQocXYpKTsKICAgIGNvbnN0IGluc3RhbGxlZE5hbWVzID0gbmV3IFNldChzdGF0ZS5pbnN0YWxsZWQubWFwKG09Pm0ubmFtZSkpOwogICAgYm94Lmlu"
    "bmVySFRNTCA9ICc8ZGl2IGNsYXNzPSJzZWN0aW9uLXRpdGxlIj5TZWFyY2ggcmVzdWx0czwvZGl2Pic7CiAgICBjb25zdCBhbGwgPSBbLi4uKHIuY2F0"
    "YWxvZ3x8W10pLCAuLi4oci5odWdnaW5nZmFjZXx8W10pXTsKICAgIGlmKCFhbGwubGVuZ3RoKXsgYm94LmlubmVySFRNTCArPSAnPGRpdiBjbGFzcz0i"
    "bXV0ZWQiPk5vIG1hdGNoZXMuIFRyeSBhbm90aGVyIHRlcm0sICcrCiAgICAgICdvciBwYXN0ZSBhbiBleGFjdCBPbGxhbWEgdGFnIGxpa2UgPHNwYW4g"
    "Y2xhc3M9Im1vbm8iPnF3ZW4zOjhiPC9zcGFuPi48L2Rpdj4nOwogICAgICBjb25zdCBiID0gZWwoImJ1dHRvbiIsImJ0biBzbSBnaG9zdCIsIlB1bGwg"
    "XCIiK3F2KyJcIiBhbnl3YXkiKTsKICAgICAgYi5zdHlsZS5tYXJnaW5Ub3A9IjEwcHgiOyBiLm9uY2xpY2s9KCk9PnB1bGxNb2RlbChxdixiKTsgYm94"
    "LmFwcGVuZENoaWxkKGIpOyByZXR1cm47IH0KICAgIGNvbnN0IGcgPSBlbCgiZGl2IiwibWdyaWQiKTsgYm94LmFwcGVuZENoaWxkKGcpOwogICAgcmVu"
    "ZGVyTW9kZWxDYXJkcyhnLCBhbGwsIGluc3RhbGxlZE5hbWVzKTsKICB9Y2F0Y2goZSl7IGJveC5pbm5lckhUTUwgKz0gZXJyQ2FyZChlLm1lc3NhZ2Up"
    "OyB9Cn0KJCgibW9kZWxTZWFyY2hCdG4iKS5vbmNsaWNrID0gZG9Nb2RlbFNlYXJjaDsKJCgibW9kZWxTZWFyY2giKS5hZGRFdmVudExpc3RlbmVyKCJr"
    "ZXlkb3duIiwgZT0+eyBpZihlLmtleT09PSJFbnRlciIpIGRvTW9kZWxTZWFyY2goKTsgfSk7Cgphc3luYyBmdW5jdGlvbiBwdWxsTW9kZWwobmFtZSwg"
    "YnRuLCBjYXJkKXsKICBpZihidG4peyBidG4uZGlzYWJsZWQgPSB0cnVlOyBidG4udGV4dENvbnRlbnQgPSAiU3RhcnRpbmfigKYiOyB9CiAgY29uc3Qg"
    "aG9zdCA9IGNhcmQgfHwgKGJ0biAmJiBidG4uY2xvc2VzdCgiLnJlY2l0ZW0iKSkgfHwgZG9jdW1lbnQuYm9keTsKICBsZXQgYm94ID0gaG9zdC5xdWVy"
    "eVNlbGVjdG9yKCIucHVsbGJveCIpOwogIGlmKCFib3gpeyBib3ggPSBlbCgiZGl2IiwicHVsbGJveCIpOyBib3guaW5uZXJIVE1MID0KICAgICc8ZGl2"
    "IGNsYXNzPSJzdGF0Ij48c3BhbiBjbGFzcz0icyI+UHJlcGFyaW5n4oCmPC9zcGFuPjxzcGFuIGNsYXNzPSJwIj48L3NwYW4+PC9kaXY+JysKICAgICc8"
    "ZGl2IGNsYXNzPSJiYXIiPjxpPjwvaT48L2Rpdj4nOyBob3N0LmFwcGVuZENoaWxkKGJveCk7IH0KICBjb25zdCBiYXIgPSBib3gucXVlcnlTZWxlY3Rv"
    "cigiaSIpLCBzdGF0ID0gYm94LnF1ZXJ5U2VsZWN0b3IoIi5zIiksCiAgICAgICAgcGMgPSBib3gucXVlcnlTZWxlY3RvcigiLnAiKTsKICB0cnl7CiAg"
    "ICBhd2FpdCBzdHJlYW1OREpTT04oIi9hcGkvbW9kZWxzL3B1bGwiLCB7bmFtZX0sIChvKT0+ewogICAgICBpZihvLnR5cGU9PT0iZXJyb3IiKXsgdGhy"
    "b3cgbmV3IEVycm9yKG8uZXJyb3IpOyB9CiAgICAgIGlmKG8udHlwZT09PSJkb25lIil7IHN0YXQudGV4dENvbnRlbnQgPSAiRG9uZSI7IGJhci5zdHls"
    "ZS53aWR0aD0iMTAwJSI7IHJldHVybjsgfQogICAgICBzdGF0LnRleHRDb250ZW50ID0gby5zdGF0dXMgfHwgIkRvd25sb2FkaW5nIjsKICAgICAgaWYo"
    "by50b3RhbCAmJiBvLmNvbXBsZXRlZCE9bnVsbCl7CiAgICAgICAgY29uc3QgcCA9IG8uY29tcGxldGVkL28udG90YWwqMTAwOyBiYXIuc3R5bGUud2lk"
    "dGggPSBwLnRvRml4ZWQoMSkrIiUiOwogICAgICAgIHBjLnRleHRDb250ZW50ID0gZm10R0Ioby5jb21wbGV0ZWQvMWU5KSsiIC8gIitmbXRHQihvLnRv"
    "dGFsLzFlOSkrIiBHQiI7CiAgICAgIH0KICAgIH0pOwogICAgdG9hc3QoIkRvd25sb2FkZWQgIituYW1lLCAib2siKTsKICAgIHN0YXRlLmluc3RhbGxl"
    "ZC5wdXNoKHtuYW1lLCBzaXplX2diOjB9KTsKICAgIGlmKGN1cnJlbnRWaWV3PT09Im1vZGVscyIpIGxvYWRNb2RlbHMoKTsKICAgIGlmKGN1cnJlbnRW"
    "aWV3PT09ImRhc2hib2FyZCIpIGxvYWREYXNoYm9hcmQoKTsKICAgIHJlZnJlc2hDaGF0TW9kZWxzKCk7CiAgfWNhdGNoKGUpewogICAgdG9hc3QoIkRv"
    "d25sb2FkIGZhaWxlZDogIitlLm1lc3NhZ2UsICJlcnIiKTsKICAgIGlmKGJ0bil7IGJ0bi5kaXNhYmxlZD1mYWxzZTsgYnRuLnRleHRDb250ZW50PSJS"
    "ZXRyeSI7IH0KICAgIHN0YXQudGV4dENvbnRlbnQgPSAiRmFpbGVkOiAiK2UubWVzc2FnZTsKICB9Cn0KYXN5bmMgZnVuY3Rpb24gcmVtb3ZlTW9kZWwo"
    "bmFtZSwgY2FyZCl7CiAgY29uc3QgYm9keSA9IG1vZGFsKHt0aXRsZToiUmVtb3ZlIG1vZGVsPyIsCiAgICBib2R5SFRNTDpgPHAgY2xhc3M9Im11dGVk"
    "Ij5EZWxldGUgPHNwYW4gY2xhc3M9Im1vbm8iPiR7ZXNjKG5hbWUpfTwvc3Bhbj4gZnJvbSBkaXNrPwogICAgICBZb3UgY2FuIGRvd25sb2FkIGl0IGFn"
    "YWluIGxhdGVyLjwvcD5gLAogICAgYWN0aW9uczpbe2xhYmVsOiJDYW5jZWwiLCBvbkNsaWNrOmNsb3NlTW9kYWx9LAogICAgICB7bGFiZWw6IlJlbW92"
    "ZSIsIGNsczoiZGFuZ2VyIiwgb25DbGljazphc3luYygpPT57CiAgICAgICAgY2xvc2VNb2RhbCgpOwogICAgICAgIHRyeXsgYXdhaXQgcG9zdCgiL2Fw"
    "aS9tb2RlbHMvZGVsZXRlIiwge25hbWV9KTsKICAgICAgICAgIHRvYXN0KCJSZW1vdmVkICIrbmFtZSwib2siKTsgbG9hZE1vZGVscygpOyByZWZyZXNo"
    "Q2hhdE1vZGVscygpOyB9CiAgICAgICAgY2F0Y2goZSl7IHRvYXN0KGUubWVzc2FnZSwiZXJyIik7IH0KICAgICAgfX1dfSk7Cn0KZnVuY3Rpb24gZXJy"
    "Q2FyZChtc2cpeyByZXR1cm4gYDxkaXYgY2xhc3M9ImNhcmQiIHN0eWxlPSJib3JkZXItY29sb3I6dmFyKC0tcmVkKSI+CiAgPGIgc3R5bGU9ImNvbG9y"
    "OnZhcigtLXJlZCkiPlNvbWV0aGluZyB3ZW50IHdyb25nPC9iPgogIDxwIGNsYXNzPSJtdXRlZCIgc3R5bGU9Im1hcmdpbjo2cHggMCAwIj4ke2VzYyht"
    "c2cpfTwvcD48L2Rpdj5gOyB9CmZ1bmN0aW9uIG9sbGFtYVNldHVwQ2FyZCgpewogIGNvbnN0IGggPSBzdGF0ZS5zeXN0ZW0gPyBzdGF0ZS5zeXN0ZW0u"
    "aW5zdGFsbF9oZWxwIDogbnVsbDsKICBjb25zdCBzdGVwcyA9IGggPyBoLnN0ZXBzIDogWyJJbnN0YWxsIE9sbGFtYSBmcm9tIGh0dHBzOi8vb2xsYW1h"
    "LmNvbSJdOwogIHJldHVybiBgPGRpdiBjbGFzcz0iY2FyZCBwYWQtbGciIHN0eWxlPSJib3JkZXItY29sb3I6dmFyKC0tc2lnbmFsLWRpbSk7bWFyZ2lu"
    "LWJvdHRvbTo4cHgiPgogICAgPGRpdiBjbGFzcz0icm93IiBzdHlsZT0iZ2FwOjEwcHg7bWFyZ2luLWJvdHRvbToxMHB4Ij4KICAgICAgPHNwYW4gY2xh"
    "c3M9ImRvdCBvZmYiPjwvc3Bhbj48Yj5PbGxhbWEgaXNuJ3QgcnVubmluZzwvYj48L2Rpdj4KICAgIDxwIGNsYXNzPSJtdXRlZCIgc3R5bGU9Im1hcmdp"
    "bjowIDAgMTJweCI+SGVvcnRoIHVzZXMgT2xsYW1hIHRvIHJ1biB0ZXh0IG1vZGVscy4KICAgICAgSW5zdGFsbCBpdCBvbmNlICgke2g/ZXNjKGgub3Mp"
    "OiIifSksIHRoZW4gdGhpcyBwYWdlIGNvbm5lY3RzIGF1dG9tYXRpY2FsbHkuPC9wPgogICAgJHtzdGVwcy5tYXAocz0+YDxkaXYgY2xhc3M9Im1vbm8i"
    "IHN0eWxlPSJmb250LXNpemU6MTJweDtiYWNrZ3JvdW5kOnZhcigtLWluayk7CiAgICAgIGJvcmRlcjoxcHggc29saWQgdmFyKC0tbGluZSk7Ym9yZGVy"
    "LXJhZGl1czo4cHg7cGFkZGluZzo5cHggMTFweDttYXJnaW4tYm90dG9tOjZweCI+JHtlc2Mocyl9PC9kaXY+YCkuam9pbigiIil9CiAgICA8YnV0dG9u"
    "IGNsYXNzPSJidG4gc20gZ2hvc3QiIHN0eWxlPSJtYXJnaW4tdG9wOjZweCIgb25jbGljaz0ibG9hZE1vZGVscygpIj5DaGVjayBhZ2FpbjwvYnV0dG9u"
    "PgogIDwvZGl2PmA7Cn0KbG9hZGVycy5tb2RlbHMgPSBsb2FkTW9kZWxzOwoKLyogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09"
    "PT09PT09PT09PT09PT09PT09PT09CiAgIENIQVQKICAgPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09"
    "PT09PT09ICovCmxldCByYWdPbj1mYWxzZSwgYWdlbnRPbj1mYWxzZSwgbG9vcE9uPWZhbHNlLCBjb3VuY2lsT249ZmFsc2U7CmZ1bmN0aW9uIHN5bmNU"
    "b2dnbGVzKCl7CiAgJCgiYWdlbnRUb2dnbGUiKS5jbGFzc0xpc3QudG9nZ2xlKCJvbiIsYWdlbnRPbik7CiAgJCgibG9vcFRvZ2dsZSIpLmNsYXNzTGlz"
    "dC50b2dnbGUoIm9uIixsb29wT24pOwogICQoImNvdW5jaWxUb2dnbGUiKS5jbGFzc0xpc3QudG9nZ2xlKCJvbiIsY291bmNpbE9uKTsKfQokKCJyYWdU"
    "b2dnbGUiKS5vbmNsaWNrID0gKCk9PnsgcmFnT249IXJhZ09uOyAkKCJyYWdUb2dnbGUiKS5jbGFzc0xpc3QudG9nZ2xlKCJvbiIscmFnT24pOyB9Owok"
    "KCJhZ2VudFRvZ2dsZSIpLm9uY2xpY2sgPSAoKT0+eyBhZ2VudE9uPSFhZ2VudE9uOwogIGlmKGFnZW50T24pIGNvdW5jaWxPbj1mYWxzZTsKICBpZigh"
    "YWdlbnRPbikgbG9vcE9uPWZhbHNlOwogIHN5bmNUb2dnbGVzKCk7IH07CiQoImxvb3BUb2dnbGUiKS5vbmNsaWNrID0gKCk9PnsgbG9vcE9uPSFsb29w"
    "T247CiAgaWYobG9vcE9uKXsgYWdlbnRPbj10cnVlOyBjb3VuY2lsT249ZmFsc2U7IH0KICBzeW5jVG9nZ2xlcygpOyB9OwokKCJjb3VuY2lsVG9nZ2xl"
    "Iikub25jbGljayA9ICgpPT57IGNvdW5jaWxPbj0hY291bmNpbE9uOwogIGlmKGNvdW5jaWxPbil7IGFnZW50T249ZmFsc2U7IGxvb3BPbj1mYWxzZTsg"
    "fQogIHN5bmNUb2dnbGVzKCk7IH07Cgphc3luYyBmdW5jdGlvbiByZWZyZXNoQ2hhdE1vZGVscygpewogIHRyeXsKICAgIGNvbnN0IGluc3QgPSBhd2Fp"
    "dCBhcGkoIi9hcGkvbW9kZWxzL2luc3RhbGxlZCIpOwogICAgc3RhdGUuaW5zdGFsbGVkID0gaW5zdC5tb2RlbHMgfHwgW107CiAgICBjb25zdCBzZWwg"
    "PSAkKCJjaGF0TW9kZWwiKTsgY29uc3QgcHJldiA9IHNlbC52YWx1ZTsKICAgIGNvbnN0IGNoYXQgPSBzdGF0ZS5pbnN0YWxsZWQuZmlsdGVyKG09PiEv"
    "ZW1iZWQvaS50ZXN0KG0ubmFtZSkpOwogICAgaWYoIWNoYXQubGVuZ3RoKXsgc2VsLmlubmVySFRNTCA9ICc8b3B0aW9uIHZhbHVlPSIiPk5vIG1vZGVs"
    "cyDigJQgZG93bmxvYWQgb25lPC9vcHRpb24+JzsgcmV0dXJuOyB9CiAgICBzZWwuaW5uZXJIVE1MID0gY2hhdC5tYXAobT0+YDxvcHRpb24gdmFsdWU9"
    "IiR7ZXNjKG0ubmFtZSl9Ij4ke2VzYyhtLm5hbWUpfTwvb3B0aW9uPmApLmpvaW4oIiIpOwogICAgaWYocHJldiAmJiBjaGF0LnNvbWUobT0+bS5uYW1l"
    "PT09cHJldikpIHNlbC52YWx1ZSA9IHByZXY7CiAgfWNhdGNoKGUpe30KfQphc3luYyBmdW5jdGlvbiBsb2FkQ29udnMoKXsKICB0cnl7CiAgICBjb25z"
    "dCByID0gYXdhaXQgYXBpKCIvYXBpL2NvbnZlcnNhdGlvbnMiKTsKICAgIGNvbnN0IGxpc3QgPSAkKCJjb252TGlzdCIpOyBsaXN0LmlubmVySFRNTCA9"
    "ICIiOwogICAgaWYoIXIuY29udmVyc2F0aW9ucy5sZW5ndGgpeyBsaXN0LmlubmVySFRNTCA9CiAgICAgICc8ZGl2IGNsYXNzPSJkaW0iIHN0eWxlPSJw"
    "YWRkaW5nOjE0cHg7Zm9udC1zaXplOjEycHg7dGV4dC1hbGlnbjpjZW50ZXIiPk5vIGNvbnZlcnNhdGlvbnMgeWV0PC9kaXY+JzsgcmV0dXJuOyB9CiAg"
    "ICByLmNvbnZlcnNhdGlvbnMuZm9yRWFjaChjPT57CiAgICAgIGNvbnN0IGl0ID0gZWwoImRpdiIsImNvbnZpdGVtIisoYy5pZD09PXN0YXRlLmN1cnJl"
    "bnRDb252PyIgYWN0aXZlIjoiIikpOwogICAgICBpdC5pbm5lckhUTUwgPSBgPHNwYW4gY2xhc3M9InR0Ij4ke2VzYyhjLnRpdGxlfHwiVW50aXRsZWQi"
    "KX08L3NwYW4+CiAgICAgICAgPHNwYW4gY2xhc3M9ImRlbCIgdGl0bGU9IkRlbGV0ZSI+CiAgICAgICAgPHN2ZyB3aWR0aD0iMTQiIGhlaWdodD0iMTQi"
    "IHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJjdXJyZW50Q29sb3IiIHN0cm9rZS13aWR0aD0iMiI+PHBhdGggZD0iTTMgNmgx"
    "OE04IDZWNGg4djJNNiA2bDEgMTRoMTBsMS0xNCIvPjwvc3ZnPjwvc3Bhbj5gOwogICAgICBpdC5xdWVyeVNlbGVjdG9yKCIudHQiKS5vbmNsaWNrID0g"
    "KCk9Pm9wZW5Db252KGMuaWQpOwogICAgICBpdC5xdWVyeVNlbGVjdG9yKCIuZGVsIikub25jbGljayA9IGFzeW5jKGUpPT57IGUuc3RvcFByb3BhZ2F0"
    "aW9uKCk7CiAgICAgICAgYXdhaXQgZGVsKCIvYXBpL2NvbnZlcnNhdGlvbnMvIitjLmlkKTsKICAgICAgICBpZihzdGF0ZS5jdXJyZW50Q29udj09PWMu"
    "aWQpeyBzdGF0ZS5jdXJyZW50Q29udj1udWxsOyBzaG93Q2hhdEVtcHR5KCk7IH0KICAgICAgICBsb2FkQ29udnMoKTsgfTsKICAgICAgbGlzdC5hcHBl"
    "bmRDaGlsZChpdCk7CiAgICB9KTsKICB9Y2F0Y2goZSl7fQp9CmFzeW5jIGZ1bmN0aW9uIG9wZW5Db252KGlkKXsKICBzdGF0ZS5jdXJyZW50Q29udiA9"
    "IGlkOyBsb2FkQ29udnMoKTsKICBjb25zdCB3cmFwID0gJCgiY2hhdFdyYXAiKTsgd3JhcC5pbm5lckhUTUwgPSAiIjsKICB0cnl7CiAgICBjb25zdCBy"
    "ID0gYXdhaXQgYXBpKCIvYXBpL2NvbnZlcnNhdGlvbnMvIitpZCsiL21lc3NhZ2VzIik7CiAgICByLm1lc3NhZ2VzLmZvckVhY2gobT0+ewogICAgICBp"
    "ZihtLnJvbGU9PT0idXNlciIpIGFkZFVzZXJNc2cobS5jb250ZW50KTsKICAgICAgZWxzZSB7IGNvbnN0IHtidWJibGUsIHNvdXJjZXMsIGFnZW50bG9n"
    "fSA9IGFkZEFJTXNnKCk7CiAgICAgICAgY29uc3QgZXYgPSAobS5tZXRhICYmIG0ubWV0YS5ldmVudHMpIHx8IFtdOwogICAgICAgIGNvbnN0IHNyY3Mg"
    "PSBldi5maWx0ZXIoZT0+ZS50eXBlPT09InNvdXJjZXMiKS5mbGF0TWFwKGU9PmUuaXRlbXN8fFtdKTsKICAgICAgICBpZihzcmNzLmxlbmd0aCkgcmVu"
    "ZGVyU291cmNlcyhzb3VyY2VzLCBzcmNzKTsKICAgICAgICByZW5kZXJBZ2VudEV2ZW50cyhhZ2VudGxvZywgZXYpOwogICAgICAgIGJ1YmJsZS5pbm5l"
    "ckhUTUwgPSBtZChtLmNvbnRlbnQpOwogICAgICB9CiAgICB9KTsKICAgIHNjcm9sbENoYXQoKTsKICB9Y2F0Y2goZSl7IHRvYXN0KGUubWVzc2FnZSwi"
    "ZXJyIik7IH0KfQpmdW5jdGlvbiBzaG93Q2hhdEVtcHR5KCl7CiAgY29uc3Qgd3JhcCA9ICQoImNoYXRXcmFwIik7CiAgd3JhcC5pbm5lckhUTUwgPSBg"
    "PGRpdiBjbGFzcz0iY2hhdC1lbXB0eSI+CiAgICA8ZGl2IGNsYXNzPSJvcmIiPjwvZGl2PgogICAgPGgzPlRhbGsgdG8geW91ciBtYWNoaW5lPC9oMz4K"
    "ICAgIDxwIGNsYXNzPSJtdXRlZCI+RXZlcnl0aGluZyBzdGF5cyBvbiB0aGlzIGNvbXB1dGVyLiBQaWNrIGEgbW9kZWwgYWJvdmUgYW5kIGFzawogICAg"
    "ICBhbnl0aGluZyDigJQgb3Igc3RhcnQgd2l0aCBvbmUgb2YgdGhlc2U6PC9wPgogICAgPGRpdiBjbGFzcz0iZXhyb3ciPgogICAgICA8YnV0dG9uIGNs"
    "YXNzPSJidG4gc20gZ2hvc3QgZXgiPkV4cGxhaW4gaG93IGEgdmVjdG9yIGRhdGFiYXNlIHdvcmtzPC9idXR0b24+CiAgICAgIDxidXR0b24gY2xhc3M9"
    "ImJ0biBzbSBnaG9zdCBleCI+U3VtbWFyaXplIHRoZSBwcm9zIGFuZCBjb25zIG9mIHJ1bm5pbmcgQUkgbG9jYWxseTwvYnV0dG9uPgogICAgICA8YnV0"
    "dG9uIGNsYXNzPSJidG4gc20gZ2hvc3QgZXgiPkRyYWZ0IGEgcG9saXRlIGVtYWlsIGRlY2xpbmluZyBhIG1lZXRpbmc8L2J1dHRvbj4KICAgIDwvZGl2"
    "PgogICAgPHAgY2xhc3M9ImhpbnQiPlRpcDogPGI+S25vd2xlZGdlPC9iPiB1c2VzIHlvdXIgZG9jdW1lbnRzLCA8Yj5BZ2VudDwvYj4gbGV0cyB0aGUg"
    "bW9kZWwgY2FsbAogICAgICB0b29scywgPGI+TG9vcDwvYj4gd29ya3MgYSB0YXNrIGF1dG9ub21vdXNseSwgYW5kIDxiPkNvdW5jaWw8L2I+IGNvbnZl"
    "bmVzIGEgcGFuZWwgb2YKICAgICAgY29uc3VsdGFudHMgd2hvIGRlYmF0ZSBiZWZvcmUgYW5zd2VyaW5nLjwvcD48L2Rpdj5gOwogIHdyYXAucXVlcnlT"
    "ZWxlY3RvckFsbCgiLmV4IikuZm9yRWFjaChiPT5iLm9uY2xpY2s9KCk9PnsKICAgICQoImNoYXRJbnB1dCIpLnZhbHVlID0gYi50ZXh0Q29udGVudC50"
    "cmltKCk7IGF1dG9Hcm93KCk7ICQoImNoYXRJbnB1dCIpLmZvY3VzKCk7IH0pOwp9CmZ1bmN0aW9uIGFkZFVzZXJNc2codGV4dCl7CiAgY29uc3Qgd3Jh"
    "cCA9ICQoImNoYXRXcmFwIik7CiAgY29uc3QgZW1wID0gd3JhcC5xdWVyeVNlbGVjdG9yKCIuY2hhdC1lbXB0eSIpOyBpZihlbXApIGVtcC5yZW1vdmUo"
    "KTsKICBjb25zdCBtID0gZWwoImRpdiIsIm1zZyB1c2VyIik7CiAgbS5pbm5lckhUTUwgPSBgPGRpdiBjbGFzcz0iYXYiPllPVTwvZGl2PjxkaXYgY2xh"
    "c3M9ImJvZHkiPgogICAgPGRpdiBjbGFzcz0id2hvIj5Zb3U8L2Rpdj48ZGl2IGNsYXNzPSJidWJibGUiPiR7bWQodGV4dCl9PC9kaXY+PC9kaXY+YDsK"
    "ICB3cmFwLmFwcGVuZENoaWxkKG0pOyBzY3JvbGxDaGF0KCk7IHJldHVybiBtOwp9CmZ1bmN0aW9uIGFkZEFJTXNnKCl7CiAgY29uc3Qgd3JhcCA9ICQo"
    "ImNoYXRXcmFwIik7CiAgY29uc3QgbSA9IGVsKCJkaXYiLCJtc2cgYWkiKTsKICBtLmlubmVySFRNTCA9IGA8ZGl2IGNsYXNzPSJhdiI+QUk8L2Rpdj48"
    "ZGl2IGNsYXNzPSJib2R5Ij4KICAgIDxkaXYgY2xhc3M9IndobyI+SGVvcnRoPC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJzcmNob3N0Ij48L2Rpdj48ZGl2"
    "IGNsYXNzPSJzdGF0dXNob3N0Ij48L2Rpdj4KICAgIDxkaXYgY2xhc3M9ImFnZW50bG9nIj48L2Rpdj48ZGl2IGNsYXNzPSJidWJibGUiPjwvZGl2Pjwv"
    "ZGl2PmA7CiAgd3JhcC5hcHBlbmRDaGlsZChtKTsKICByZXR1cm4geyByb290Om0sIGJ1YmJsZTptLnF1ZXJ5U2VsZWN0b3IoIi5idWJibGUiKSwKICAg"
    "IHNvdXJjZXM6bS5xdWVyeVNlbGVjdG9yKCIuc3JjaG9zdCIpLCBzdGF0dXM6bS5xdWVyeVNlbGVjdG9yKCIuc3RhdHVzaG9zdCIpLAogICAgYWdlbnRs"
    "b2c6bS5xdWVyeVNlbGVjdG9yKCIuYWdlbnRsb2ciKSB9Owp9CmZ1bmN0aW9uIHJlbmRlclNvdXJjZXMoaG9zdCwgaXRlbXMpewogIGlmKCFpdGVtcy5s"
    "ZW5ndGgpIHJldHVybjsKICBjb25zdCBib3ggPSBlbCgiZGl2Iiwic3JjYm94Iik7CiAgYm94LmlubmVySFRNTCA9ICc8ZGl2IGNsYXNzPSJzdCI+4peG"
    "IFNvdXJjZXMgZnJvbSB5b3VyIGtub3dsZWRnZSBiYXNlPC9kaXY+JysKICAgIGl0ZW1zLm1hcChzPT5gPGRpdiBjbGFzcz0ic3JjaXRlbSI+PGI+JHtl"
    "c2Mocy5kb2MpfTwvYj4KICAgICAgPHNwYW4gY2xhc3M9ImRpbSI+wrcgJHsocy5zY29yZSoxMDB8fDApLnRvRml4ZWQoMCl9JSBtYXRjaDwvc3Bhbj48"
    "YnI+CiAgICAgIDxzcGFuIGNsYXNzPSJtdXRlZCI+JHtlc2MoKHMuc25pcHBldHx8cy50ZXh0fHwiIikuc2xpY2UoMCwxNTApKX3igKY8L3NwYW4+PC9k"
    "aXY+YCkuam9pbigiIik7CiAgaG9zdC5hcHBlbmRDaGlsZChib3gpOwp9CmZ1bmN0aW9uIHN0ZXBFbChuYW1lLCBhcmdzKXsKICBjb25zdCBpc01jcCA9"
    "IG5hbWUuc3RhcnRzV2l0aCgibWNwX18iKTsKICBjb25zdCBkaXNwID0gaXNNY3AgPyBuYW1lLnNwbGl0KCJfXyIpLnNsaWNlKDIpLmpvaW4oIl9fIikr"
    "IiAoTUNQKSIgOiBuYW1lOwogIGNvbnN0IHMgPSBlbCgiZGl2Iiwic3RlcCIpOwogIGNvbnN0IGFyZ1N0ciA9IHR5cGVvZiBhcmdzPT09Im9iamVjdCIg"
    "PyBKU09OLnN0cmluZ2lmeShhcmdzKSA6IFN0cmluZyhhcmdzfHwiIik7CiAgcy5pbm5lckhUTUwgPSBgPGRpdiBjbGFzcz0ic2giPjxkaXYgY2xhc3M9"
    "InNwaW4iIHN0eWxlPSJ3aWR0aDoxMnB4O2hlaWdodDoxMnB4Ij48L2Rpdj4KICAgIDxzcGFuIGNsYXNzPSJ0biI+JHtlc2MoZGlzcCl9PC9zcGFuPjxz"
    "cGFuIGNsYXNzPSJhcmciPiR7ZXNjKGFyZ1N0cil9PC9zcGFuPjwvZGl2PgogICAgPGRpdiBjbGFzcz0ic2IiIGhpZGRlbj48L2Rpdj5gOwogIGNvbnN0"
    "IHNoID0gcy5xdWVyeVNlbGVjdG9yKCIuc2giKSwgc2IgPSBzLnF1ZXJ5U2VsZWN0b3IoIi5zYiIpOwogIHNoLnN0eWxlLmN1cnNvcj0icG9pbnRlciI7"
    "CiAgc2gub25jbGljayA9ICgpPT57IHNiLmhpZGRlbiA9ICFzYi5oaWRkZW47IH07CiAgcmV0dXJuIHM7Cn0KZnVuY3Rpb24gcmVuZGVyQWdlbnRFdmVu"
    "dHMoaG9zdCwgZXZlbnRzKXsKICBsZXQgcGVuZGluZyA9IG51bGw7CiAgbGV0IGNwPW51bGwsIGNvbnM9W107IGNvbnN0IGNmYWlsPW5ldyBTZXQoKTsK"
    "ICBldmVudHMuZm9yRWFjaChlPT57CiAgICBpZihlLnR5cGU9PT0idG9vbF9jYWxsIil7IGNvbnN0IHMgPSBzdGVwRWwoZS5uYW1lLCBlLmFyZ3MpOwog"
    "ICAgICBob3N0LmFwcGVuZENoaWxkKHMpOyBwZW5kaW5nID0gczsgfQogICAgZWxzZSBpZihlLnR5cGU9PT0idG9vbF9yZXN1bHQiICYmIHBlbmRpbmcp"
    "ewogICAgICBjb25zdCBzcCA9IHBlbmRpbmcucXVlcnlTZWxlY3RvcigiLnNwaW4iKTsgaWYoc3ApIHNwLm91dGVySFRNTCA9CiAgICAgICAgJzxzcGFu"
    "IHN0eWxlPSJjb2xvcjp2YXIoLS1saXZlKSI+4pyTPC9zcGFuPic7CiAgICAgIGNvbnN0IHNiID0gcGVuZGluZy5xdWVyeVNlbGVjdG9yKCIuc2IiKTsg"
    "c2IudGV4dENvbnRlbnQgPSBlLnJlc3VsdHx8IiI7CiAgICAgIHBlbmRpbmcgPSBudWxsOwogICAgfQogICAgZWxzZSBpZihlLnR5cGU9PT0idGhvdWdo"
    "dCIpeyBob3N0LmFwcGVuZENoaWxkKHRob3VnaHRFbChlLnRleHQpKTsgfQogICAgZWxzZSBpZihlLnR5cGU9PT0iY291bmNpbF9zdGFydCIpeyBjb25z"
    "PWUuY29uc3VsdGFudHN8fFtdOwogICAgICBjcD1jb3VuY2lsUGFuZWxFbChlKTsgaG9zdC5hcHBlbmRDaGlsZChjcCk7IH0KICAgIGVsc2UgaWYoZS50"
    "eXBlPT09ImNvdW5jaWxfYnJpZWYiICYmIGNwKXsgY3AuYXBwZW5kQ2hpbGQoY291bmNpbEJyaWVmRWwoZS50ZXh0KSk7IH0KICAgIGVsc2UgaWYoZS50"
    "eXBlPT09ImNvdW5jaWxfcm91bmQiICYmIGNwKXsKICAgICAgY291bmNpbFJvdW5kRWwoY3AsIGUucm91bmQsIGUubGFiZWwsIGNvbnMsIGNmYWlsLCBm"
    "YWxzZSk7IH0KICAgIGVsc2UgaWYoZS50eXBlPT09ImNvdW5jaWxfdGFrZSIgJiYgY3ApeyBjb3VuY2lsRmlsbChjcCwgZS5pZCwgZS5yb3VuZCwgZS50"
    "ZXh0KTsgfQogICAgZWxzZSBpZihlLnR5cGU9PT0iY29uc3VsdGFudF9zdGF0dXMiICYmIGNwICYmIGUuc3RhdGU9PT0iZmFpbGVkIil7CiAgICAgIGNm"
    "YWlsLmFkZChlLmlkKTsgY291bmNpbEZhaWwoY3AsIGUuaWQsIGUucm91bmQpOyB9CiAgfSk7Cn0KCmZ1bmN0aW9uIHNjcm9sbENoYXQoKXsgY29uc3Qg"
    "cyA9ICQoImNoYXRTY3JvbGwiKTsgcy5zY3JvbGxUb3AgPSBzLnNjcm9sbEhlaWdodDsgfQpmdW5jdGlvbiBhdXRvR3JvdygpeyBjb25zdCB0ID0gJCgi"
    "Y2hhdElucHV0Iik7IHQuc3R5bGUuaGVpZ2h0PSJhdXRvIjsKICB0LnN0eWxlLmhlaWdodCA9IE1hdGgubWluKHQuc2Nyb2xsSGVpZ2h0LCAyMDApKyJw"
    "eCI7IH0KY29uc3QgU0VORF9JQ09OID0gJzxzdmcgd2lkdGg9IjE4IiBoZWlnaHQ9IjE4IiB2aWV3Qm94PSIwIDAgMjQgMjQiIGZpbGw9Im5vbmUiIHN0"
    "cm9rZT0iY3VycmVudENvbG9yIiBzdHJva2Utd2lkdGg9IjIiPjxwYXRoIGQ9Ik03IDExIDEyIDZsNSA1TTEyIDZ2MTMiLz48L3N2Zz4nOwpjb25zdCBT"
    "VE9QX0lDT04gPSAnPHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0iY3VycmVudENvbG9yIj48cmVjdCB4"
    "PSI2IiB5PSI2IiB3aWR0aD0iMTIiIGhlaWdodD0iMTIiIHJ4PSIyIi8+PC9zdmc+JzsKZnVuY3Rpb24gc2V0U2VuZE1vZGUoc2VuZGluZyl7CiAgY29u"
    "c3QgYiA9ICQoInNlbmRCdG4iKTsKICBiLmNsYXNzTGlzdC50b2dnbGUoInN0b3AiLCBzZW5kaW5nKTsKICBiLnRpdGxlID0gc2VuZGluZyA/ICJTdG9w"
    "IGdlbmVyYXRpbmciIDogIlNlbmQiOwogIGIuaW5uZXJIVE1MID0gc2VuZGluZyA/IFNUT1BfSUNPTiA6IFNFTkRfSUNPTjsKfQokKCJjaGF0SW5wdXQi"
    "KS5hZGRFdmVudExpc3RlbmVyKCJpbnB1dCIsIGF1dG9Hcm93KTsKJCgiY2hhdElucHV0IikuYWRkRXZlbnRMaXN0ZW5lcigia2V5ZG93biIsIGU9PnsK"
    "ICBpZihlLmtleT09PSJFbnRlciIgJiYgIWUuc2hpZnRLZXkpeyBlLnByZXZlbnREZWZhdWx0KCk7CiAgICBpZighc3RhdGUuc2VuZGluZykgc2VuZE1l"
    "c3NhZ2UoKTsgfSB9KTsKJCgic2VuZEJ0biIpLm9uY2xpY2sgPSAoKT0+ewogIGlmKHN0YXRlLnNlbmRpbmcpeyBpZihzdGF0ZS5hYm9ydCkgc3RhdGUu"
    "YWJvcnQuYWJvcnQoKTsgcmV0dXJuOyB9CiAgc2VuZE1lc3NhZ2UoKTsKfTsKJCgiY2xlYXJDaGF0QnRuIikub25jbGljayA9ICQoIm5ld0NvbnZCdG4i"
    "KS5vbmNsaWNrID0gKCk9PnsKICBzdGF0ZS5jdXJyZW50Q29udj1udWxsOyBzaG93Q2hhdEVtcHR5KCk7IGxvYWRDb252cygpOwogICQoImNoYXRJbnB1"
    "dCIpLmZvY3VzKCk7IH07CgpmdW5jdGlvbiB0aG91Z2h0RWwodGV4dCl7CiAgY29uc3QgZCA9IGVsKCJkaXYiLCJ0aG91Z2h0Iik7CiAgZC5pbm5lckhU"
    "TUwgPSAnPHNwYW4gY2xhc3M9InRsIj7inLMgdGhvdWdodDwvc3Bhbj4nICsgZXNjKHRleHR8fCIiKTsKICByZXR1cm4gZDsKfQovKiAtLS0tIGNvdW5j"
    "aWwgcmVuZGVyaW5nIChzaGFyZWQgYnkgbGl2ZSBzdHJlYW0gYW5kIGhpc3RvcnkgcmVwbGF5KSAtLS0tICovCmNvbnN0IENDX0NPTE9SUyA9IFsiI2Y1"
    "YTYyMyIsIiM2MGE1ZmEiLCIjYTc4YmZhIiwiIzRhZGU4MCIsIiNmODcxNzEiLAogICAgICAgICAgICAgICAgICAgIiNmNDcyYjYiLCIjZmFjYzE1Iiwi"
    "IzJkZDRiZiIsIiNmYjkyM2MiLCIjOTRhM2I4Il07CmNvbnN0IENDID0gKGkpPT5DQ19DT0xPUlNbaSAlIENDX0NPTE9SUy5sZW5ndGhdOwpmdW5jdGlv"
    "biBjb3VuY2lsUGFuZWxFbChvKXsKICBjb25zdCBwID0gZWwoImRpdiIsImNvdW5jaWwiKTsKICBjb25zdCByID0gby5yb3VuZHMgPyBgIMK3ICR7by5y"
    "b3VuZHN9IGNvbnN1bHRhdGlvbiByb3VuZCR7by5yb3VuZHM+MT8icyI6IiJ9YCA6ICIiOwogIHAuaW5uZXJIVE1MID0gYDxkaXYgY2xhc3M9ImNoZWFk"
    "Ij7irKIgQ291bmNpbCBvZiAke28uc2l6ZX0ke3J9PC9kaXY+YDsKICByZXR1cm4gcDsKfQpmdW5jdGlvbiBjb3VuY2lsQnJpZWZFbCh0ZXh0KXsKICBj"
    "b25zdCBiID0gZWwoImRpdiIsInNyY2JveCIpOyBiLnN0eWxlLm1hcmdpblRvcCA9ICIxMHB4IjsKICBiLmlubmVySFRNTCA9ICc8ZGl2IGNsYXNzPSJz"
    "dCI+4peGIFNoYXJlZCByZXNlYXJjaCBicmllZjwvZGl2PicgKwogICAgJzxkaXYgY2xhc3M9Im1vbm8iIHN0eWxlPSJmb250LXNpemU6MTFweDt3aGl0"
    "ZS1zcGFjZTpwcmUtd3JhcDttYXgtaGVpZ2h0OjE1MHB4O292ZXJmbG93OmF1dG87Y29sb3I6dmFyKC0tbXV0ZWQpIj48L2Rpdj4nOwogIGIubGFzdEVs"
    "ZW1lbnRDaGlsZC50ZXh0Q29udGVudCA9IHRleHQgfHwgIiI7CiAgcmV0dXJuIGI7Cn0KZnVuY3Rpb24gY291bmNpbFJvdW5kRWwocGFuZWwsIHJvdW5k"
    "LCBsYWJlbCwgY29uc3VsdGFudHMsIGZhaWxlZCwgbGl2ZSl7CiAgY29uc3Qgc2VjID0gZWwoImRpdiIpOwogIHNlYy5pbm5lckhUTUwgPSBgPGRpdiBj"
    "bGFzcz0iY3JvdW5kIj4ke2VzYyhsYWJlbHx8KCJSb3VuZCAiK3JvdW5kKSl9PC9kaXY+YDsKICBjb25zdCBnID0gZWwoImRpdiIsImNncmlkIik7CiAg"
    "KGNvbnN1bHRhbnRzfHxbXSkuZm9yRWFjaChjPT57CiAgICBpZihmYWlsZWQuaGFzKGMuaWQpKSByZXR1cm47CiAgICBjb25zdCBjYXJkID0gZWwoImRp"
    "diIsImNjYXJkIik7CiAgICBjYXJkLmRhdGFzZXQuY2lkID0gYy5pZDsgY2FyZC5kYXRhc2V0LnJvdW5kID0gcm91bmQ7CiAgICBjYXJkLmlubmVySFRN"
    "TCA9IGA8ZGl2IGNsYXNzPSJjcm9sZSI+PHNwYW4gY2xhc3M9ImNkb3QiIHN0eWxlPSJiYWNrZ3JvdW5kOiR7Q0MoYy5pZCl9Ij48L3NwYW4+CiAgICAg"
    "IDxzcGFuPiR7ZXNjKGMucm9sZSl9PC9zcGFuPjxzcGFuIGNsYXNzPSJjc3QiPiR7bGl2ZT8nPHNwYW4gY2xhc3M9InNwaW4iIHN0eWxlPSJ3aWR0aDox"
    "MXB4O2hlaWdodDoxMXB4Ij48L3NwYW4+JzonJ308L3NwYW4+PC9kaXY+CiAgICAgICR7cm91bmQ9PT0wP2A8ZGl2IGNsYXNzPSJjZm9jdXMiPiR7ZXNj"
    "KGMuZm9jdXN8fCIiKX08L2Rpdj5gOiIifQogICAgICA8ZGl2IGNsYXNzPSJjdGFrZSIgdGl0bGU9IkNsaWNrIHRvIGV4cGFuZCI+PC9kaXY+YDsKICAg"
    "IGNhcmQucXVlcnlTZWxlY3RvcigiLmN0YWtlIikub25jbGljayA9IChlKT0+ZS5jdXJyZW50VGFyZ2V0LmNsYXNzTGlzdC50b2dnbGUoIm9wZW4iKTsK"
    "ICAgIGcuYXBwZW5kQ2hpbGQoY2FyZCk7CiAgfSk7CiAgc2VjLmFwcGVuZENoaWxkKGcpOyBwYW5lbC5hcHBlbmRDaGlsZChzZWMpOwp9CmZ1bmN0aW9u"
    "IGNvdW5jaWxGaWxsKHBhbmVsLCBpZCwgcm91bmQsIHRleHQpewogIGNvbnN0IGNhcmQgPSBwYW5lbC5xdWVyeVNlbGVjdG9yKGAuY2NhcmRbZGF0YS1j"
    "aWQ9IiR7aWR9Il1bZGF0YS1yb3VuZD0iJHtyb3VuZH0iXWApOwogIGlmKCFjYXJkKSByZXR1cm47CiAgY2FyZC5xdWVyeVNlbGVjdG9yKCIuY3Rha2Ui"
    "KS50ZXh0Q29udGVudCA9IHRleHQgfHwgIiI7CiAgY2FyZC5xdWVyeVNlbGVjdG9yKCIuY3N0IikuaW5uZXJIVE1MID0gJzxzcGFuIHN0eWxlPSJjb2xv"
    "cjp2YXIoLS1saXZlKTtmb250LXNpemU6MTJweCI+4pyTPC9zcGFuPic7Cn0KZnVuY3Rpb24gY291bmNpbEZhaWwocGFuZWwsIGlkLCByb3VuZCl7CiAg"
    "Y29uc3QgY2FyZCA9IHBhbmVsLnF1ZXJ5U2VsZWN0b3IoYC5jY2FyZFtkYXRhLWNpZD0iJHtpZH0iXVtkYXRhLXJvdW5kPSIke3JvdW5kfSJdYCk7CiAg"
    "aWYoIWNhcmQpIHJldHVybjsKICBjYXJkLmNsYXNzTGlzdC5hZGQoImZhaWxlZCIpOwogIGNhcmQucXVlcnlTZWxlY3RvcigiLmNzdCIpLmlubmVySFRN"
    "TCA9ICc8c3BhbiBzdHlsZT0iY29sb3I6dmFyKC0tc2lnbmFsKTtmb250LXNpemU6MTJweCI+4pqgPC9zcGFuPic7CiAgY2FyZC5xdWVyeVNlbGVjdG9y"
    "KCIuY3Rha2UiKS50ZXh0Q29udGVudCA9ICIobm8gcmVzcG9uc2UpIjsKfQphc3luYyBmdW5jdGlvbiBzZW5kTWVzc2FnZSgpewogIGlmKHN0YXRlLnNl"
    "bmRpbmcpIHJldHVybjsKICBjb25zdCBpbnB1dCA9ICQoImNoYXRJbnB1dCIpOyBjb25zdCB0ZXh0ID0gaW5wdXQudmFsdWUudHJpbSgpOwogIGNvbnN0"
    "IG1vZGVsID0gJCgiY2hhdE1vZGVsIikudmFsdWU7CiAgaWYoIXRleHQpIHJldHVybjsKICBpZighbW9kZWwpeyB0b2FzdCgiRG93bmxvYWQgYSBtb2Rl"
    "bCBmaXJzdCAoTW9kZWxzIHBhZ2UpLiIsImVyciIpOyBzaG93KCJtb2RlbHMiKTsgcmV0dXJuOyB9CiAgaW5wdXQudmFsdWU9IiI7IGF1dG9Hcm93KCk7"
    "CiAgYWRkVXNlck1zZyh0ZXh0KTsKICBjb25zdCB1aSA9IGFkZEFJTXNnKCk7CiAgY29uc3Qgc3RhdHVzTGluZSA9IGVsKCJkaXYiLCJzdGF0dXNsaW5l"
    "IiwnPGRpdiBjbGFzcz0ic3BpbiI+PC9kaXY+PHNwYW4+VGhpbmtpbmfigKY8L3NwYW4+Jyk7CiAgdWkuc3RhdHVzLmFwcGVuZENoaWxkKHN0YXR1c0xp"
    "bmUpOwogIHN0YXRlLnNlbmRpbmcgPSB0cnVlOyBzdGF0ZS5hYm9ydCA9IG5ldyBBYm9ydENvbnRyb2xsZXIoKTsKICBzZXRTZW5kTW9kZSh0cnVlKTsK"
    "ICBsZXQgYW5zd2VyID0gIiI7IGxldCBwZW5kaW5nU3RlcCA9IG51bGw7IGxldCBnb3RUb2tlbj1mYWxzZTsKICBsZXQgY291bmNpbFBhbmVsPW51bGws"
    "IGNvbnN1bHRhbnRzPVtdOyBjb25zdCBmYWlsZWRJZHM9bmV3IFNldCgpOwogIGNvbnN0IGNsZWFyU3Bpbm5lcnMgPSAoKT0+ewogICAgaWYoc3RhdHVz"
    "TGluZS5wYXJlbnROb2RlKSBzdGF0dXNMaW5lLnJlbW92ZSgpOwogICAgdWkuYWdlbnRsb2cucXVlcnlTZWxlY3RvckFsbCgiLnN0YXR1c2xpbmUiKS5m"
    "b3JFYWNoKHg9PngucmVtb3ZlKCkpOwogIH07CgogIHRyeXsKICAgIGF3YWl0IHN0cmVhbU5ESlNPTigiL2FwaS9jaGF0IiwgewogICAgICBtZXNzYWdl"
    "OnRleHQsIG1vZGVsLCBjb252ZXJzYXRpb25faWQ6IHN0YXRlLmN1cnJlbnRDb252LAogICAgICB1c2VfcmFnOiByYWdPbiwgYWdlbnRfbW9kZTogYWdl"
    "bnRPbiwgbG9vcF9tb2RlOiBsb29wT24sCiAgICAgIGNvdW5jaWxfbW9kZTogY291bmNpbE9uCiAgICB9LCAobyk9PnsKICAgICAgaWYoby50eXBlPT09"
    "Im1ldGEiKXsgaWYoIXN0YXRlLmN1cnJlbnRDb252KXsgc3RhdGUuY3VycmVudENvbnY9by5jb252ZXJzYXRpb25faWQ7CiAgICAgICAgbG9hZENvbnZz"
    "KCk7IH0gfQogICAgICBlbHNlIGlmKG8udHlwZT09PSJzb3VyY2VzIil7IHJlbmRlclNvdXJjZXModWkuc291cmNlcywgby5pdGVtc3x8W10pOyB9CiAg"
    "ICAgIGVsc2UgaWYoby50eXBlPT09InN0YXR1cyIpeyBzdGF0dXNMaW5lLnF1ZXJ5U2VsZWN0b3IoInNwYW4iKS50ZXh0Q29udGVudCA9IG8udGV4dDsK"
    "ICAgICAgICBpZighc3RhdHVzTGluZS5wYXJlbnROb2RlICYmICFnb3RUb2tlbikgdWkuc3RhdHVzLmFwcGVuZENoaWxkKHN0YXR1c0xpbmUpOyB9CiAg"
    "ICAgIGVsc2UgaWYoby50eXBlPT09ImNvdW5jaWxfc3RhcnQiKXsKICAgICAgICBjb25zdWx0YW50cyA9IG8uY29uc3VsdGFudHN8fFtdOwogICAgICAg"
    "IGNvdW5jaWxQYW5lbCA9IGNvdW5jaWxQYW5lbEVsKG8pOwogICAgICAgIHVpLmFnZW50bG9nLmFwcGVuZENoaWxkKGNvdW5jaWxQYW5lbCk7IHNjcm9s"
    "bENoYXQoKTsKICAgICAgfQogICAgICBlbHNlIGlmKG8udHlwZT09PSJjb3VuY2lsX2JyaWVmIil7CiAgICAgICAgaWYoY291bmNpbFBhbmVsKSBjb3Vu"
    "Y2lsUGFuZWwuYXBwZW5kQ2hpbGQoY291bmNpbEJyaWVmRWwoby50ZXh0KSk7CiAgICAgIH0KICAgICAgZWxzZSBpZihvLnR5cGU9PT0iY291bmNpbF9y"
    "b3VuZCIpewogICAgICAgIGlmKGNvdW5jaWxQYW5lbCl7IGNvdW5jaWxSb3VuZEVsKGNvdW5jaWxQYW5lbCwgby5yb3VuZCwgby5sYWJlbCwKICAgICAg"
    "ICAgIGNvbnN1bHRhbnRzLCBmYWlsZWRJZHMsIHRydWUpOyBzY3JvbGxDaGF0KCk7IH0KICAgICAgfQogICAgICBlbHNlIGlmKG8udHlwZT09PSJjb3Vu"
    "Y2lsX3Rha2UiKXsKICAgICAgICBpZihjb3VuY2lsUGFuZWwpIGNvdW5jaWxGaWxsKGNvdW5jaWxQYW5lbCwgby5pZCwgby5yb3VuZCwgby50ZXh0KTsK"
    "ICAgICAgfQogICAgICBlbHNlIGlmKG8udHlwZT09PSJjb25zdWx0YW50X3N0YXR1cyIpewogICAgICAgIGlmKG8uc3RhdGU9PT0iZmFpbGVkIiAmJiBj"
    "b3VuY2lsUGFuZWwpeyBmYWlsZWRJZHMuYWRkKG8uaWQpOwogICAgICAgICAgY291bmNpbEZhaWwoY291bmNpbFBhbmVsLCBvLmlkLCBvLnJvdW5kKTsg"
    "fQogICAgICB9CiAgICAgIGVsc2UgaWYoby50eXBlPT09InRob3VnaHQiKXsKICAgICAgICBjbGVhclNwaW5uZXJzKCk7CiAgICAgICAgdWkuYWdlbnRs"
    "b2cuYXBwZW5kQ2hpbGQodGhvdWdodEVsKG8udGV4dCkpOwogICAgICAgIHVpLmFnZW50bG9nLmFwcGVuZENoaWxkKGVsKCJkaXYiLCJzdGF0dXNsaW5l"
    "IiwKICAgICAgICAgICc8ZGl2IGNsYXNzPSJzcGluIj48L2Rpdj48c3Bhbj5Xb3JraW5n4oCmPC9zcGFuPicpKTsKICAgICAgICBzY3JvbGxDaGF0KCk7"
    "CiAgICAgIH0KICAgICAgZWxzZSBpZihvLnR5cGU9PT0idG9vbF9jYWxsIil7CiAgICAgICAgY2xlYXJTcGlubmVycygpOwogICAgICAgIGNvbnN0IHMg"
    "PSBzdGVwRWwoby5uYW1lLCBvLmFyZ3MpOyB1aS5hZ2VudGxvZy5hcHBlbmRDaGlsZChzKTsgcGVuZGluZ1N0ZXA9czsKICAgICAgICB1aS5hZ2VudGxv"
    "Zy5hcHBlbmRDaGlsZChlbCgiZGl2Iiwic3RhdHVzbGluZSIsCiAgICAgICAgICAnPGRpdiBjbGFzcz0ic3BpbiI+PC9kaXY+PHNwYW4+UnVubmluZyB0"
    "b29s4oCmPC9zcGFuPicpKTsKICAgICAgICBzY3JvbGxDaGF0KCk7CiAgICAgIH0KICAgICAgZWxzZSBpZihvLnR5cGU9PT0idG9vbF9yZXN1bHQiKXsK"
    "ICAgICAgICBjb25zdCBzbCA9IHVpLmFnZW50bG9nLnF1ZXJ5U2VsZWN0b3IoIi5zdGF0dXNsaW5lIik7IGlmKHNsKSBzbC5yZW1vdmUoKTsKICAgICAg"
    "ICBpZihwZW5kaW5nU3RlcCl7IGNvbnN0IHNwPXBlbmRpbmdTdGVwLnF1ZXJ5U2VsZWN0b3IoIi5zcGluIik7CiAgICAgICAgICBpZihzcCkgc3Aub3V0"
    "ZXJIVE1MPSc8c3BhbiBzdHlsZT0iY29sb3I6dmFyKC0tbGl2ZSkiPuKckzwvc3Bhbj4nOwogICAgICAgICAgcGVuZGluZ1N0ZXAucXVlcnlTZWxlY3Rv"
    "cigiLnNiIikudGV4dENvbnRlbnQgPSBvLnJlc3VsdHx8IiI7IHBlbmRpbmdTdGVwPW51bGw7IH0KICAgICAgICB1aS5hZ2VudGxvZy5hcHBlbmRDaGls"
    "ZChlbCgiZGl2Iiwic3RhdHVzbGluZSIsCiAgICAgICAgICAnPGRpdiBjbGFzcz0ic3BpbiI+PC9kaXY+PHNwYW4+VGhpbmtpbmfigKY8L3NwYW4+Jykp"
    "OyBzY3JvbGxDaGF0KCk7CiAgICAgIH0KICAgICAgZWxzZSBpZihvLnR5cGU9PT0idG9rZW4iKXsKICAgICAgICBpZighZ290VG9rZW4peyBnb3RUb2tl"
    "bj10cnVlOyBjbGVhclNwaW5uZXJzKCk7IH0KICAgICAgICBhbnN3ZXIgKz0gby50ZXh0OyB1aS5idWJibGUuaW5uZXJIVE1MID0gbWQoYW5zd2VyKTsK"
    "ICAgICAgICB1aS5idWJibGUuY2xhc3NMaXN0LmFkZCgiY3Vyc29yLWJsaW5rIik7IHNjcm9sbENoYXQoKTsKICAgICAgfQogICAgICBlbHNlIGlmKG8u"
    "dHlwZT09PSJlcnJvciIpewogICAgICAgIGNsZWFyU3Bpbm5lcnMoKTsKICAgICAgICB1aS5idWJibGUuY2xhc3NMaXN0LnJlbW92ZSgiY3Vyc29yLWJs"
    "aW5rIik7CiAgICAgICAgdWkuYnViYmxlLmlubmVySFRNTCArPSBgPGRpdiBjbGFzcz0iY2FyZCIgc3R5bGU9ImJvcmRlci1jb2xvcjp2YXIoLS1yZWQp"
    "O21hcmdpbi10b3A6OHB4Ij4KICAgICAgICAgIDxzcGFuIHN0eWxlPSJjb2xvcjp2YXIoLS1yZWQpIj4ke2VzYyhvLmVycm9yKX08L3NwYW4+PC9kaXY+"
    "YDsKICAgICAgfQogICAgICBlbHNlIGlmKG8udHlwZT09PSJkb25lIil7IHVpLmJ1YmJsZS5jbGFzc0xpc3QucmVtb3ZlKCJjdXJzb3ItYmxpbmsiKTsg"
    "fQogICAgfSwgc3RhdGUuYWJvcnQuc2lnbmFsKTsKICB9Y2F0Y2goZSl7CiAgICBjbGVhclNwaW5uZXJzKCk7CiAgICB1aS5idWJibGUuY2xhc3NMaXN0"
    "LnJlbW92ZSgiY3Vyc29yLWJsaW5rIik7CiAgICBpZihlLm5hbWU9PT0iQWJvcnRFcnJvciIpewogICAgICBpZihwZW5kaW5nU3RlcCl7IGNvbnN0IHNw"
    "PXBlbmRpbmdTdGVwLnF1ZXJ5U2VsZWN0b3IoIi5zcGluIik7CiAgICAgICAgaWYoc3ApIHNwLm91dGVySFRNTD0nPHNwYW4gY2xhc3M9ImRpbSI+4pag"
    "PC9zcGFuPic7IH0KICAgICAgdWkuYnViYmxlLmluc2VydEFkamFjZW50SFRNTCgiYmVmb3JlZW5kIiwKICAgICAgICAnPGRpdiBjbGFzcz0iaGludCIg"
    "c3R5bGU9Im1hcmdpbi10b3A6OHB4Ij7ilqAgU3RvcHBlZCDigJQgcGFydGlhbCByZXBseSBrZXB0LjwvZGl2PicpOwogICAgfSBlbHNlIHsKICAgICAg"
    "dWkuYnViYmxlLmlubmVySFRNTCArPSBgPGRpdiBjbGFzcz0iY2FyZCIgc3R5bGU9ImJvcmRlci1jb2xvcjp2YXIoLS1yZWQpO21hcmdpbi10b3A6OHB4"
    "Ij4KICAgICAgICA8c3BhbiBzdHlsZT0iY29sb3I6dmFyKC0tcmVkKSI+JHtlc2MoZS5tZXNzYWdlKX08L3NwYW4+PC9kaXY+YDsKICAgIH0KICB9Zmlu"
    "YWxseXsKICAgIHN0YXRlLnNlbmRpbmc9ZmFsc2U7IHN0YXRlLmFib3J0PW51bGw7IHNldFNlbmRNb2RlKGZhbHNlKTsKICAgIGNsZWFyU3Bpbm5lcnMo"
    "KTsgdWkuYnViYmxlLmNsYXNzTGlzdC5yZW1vdmUoImN1cnNvci1ibGluayIpOwogICAgc2Nyb2xsQ2hhdCgpOwogIH0KfQpsb2FkZXJzLmNoYXQgPSAo"
    "KT0+eyByZWZyZXNoQ2hhdE1vZGVscygpOyBsb2FkQ29udnMoKTsKICBpZighc3RhdGUuY3VycmVudENvbnYgJiYgISQoImNoYXRXcmFwIikuY2hpbGRy"
    "ZW4ubGVuZ3RoKSBzaG93Q2hhdEVtcHR5KCk7CiAgc2V0VGltZW91dCgoKT0+JCgiY2hhdElucHV0IikuZm9jdXMoKSw1MCk7IH07CgovKiA9PT09PT09"
    "PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT0KICAgSU1BR0VTCiAgID09PT09PT09PT09PT09PT09PT09"
    "PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PSAqLwphc3luYyBmdW5jdGlvbiBsb2FkSW1hZ2VzKCl7CiAgY29uc3QgaG9zdCA9"
    "ICQoImltYWdlc0JvZHkiKTsKICBob3N0LmlubmVySFRNTCA9ICc8ZGl2IGNsYXNzPSJyb3ciPjxkaXYgY2xhc3M9InNwaW4iPjwvZGl2PjxzcGFuIGNs"
    "YXNzPSJtdXRlZCI+TG9hZGluZ+KApjwvc3Bhbj48L2Rpdj4nOwogIGxldCBzdDsKICB0cnl7IHN0ID0gYXdhaXQgYXBpKCIvYXBpL2ltYWdlcy9zdGF0"
    "dXMiKTsgfWNhdGNoKGUpeyBob3N0LmlubmVySFRNTD1lcnJDYXJkKGUubWVzc2FnZSk7IHJldHVybjsgfQogIHN0YXRlLmltZ1ByZXNldHMgPSBzdC5w"
    "cmVzZXRzfHxbXTsKICBpZighc3QuaW5zdGFsbGVkKXsKICAgIGhvc3QuaW5uZXJIVE1MID0gYDxkaXYgY2xhc3M9ImltZ2xheW91dCI+CiAgICAgIDxk"
    "aXYgY2xhc3M9ImdlbnBhbmVsIiBpZD0ic2V0dXBIb3N0Ij48L2Rpdj4KICAgICAgPGRpdj48ZGl2IGNsYXNzPSJzZWN0aW9uLXRpdGxlIiBzdHlsZT0i"
    "bWFyZ2luLXRvcDowIj5HYWxsZXJ5PC9kaXY+CiAgICAgICAgPGRpdiBjbGFzcz0iZ2FsbGVyeSIgaWQ9ImdhbGxlcnkiPjwvZGl2PjwvZGl2PgogICAg"
    "PC9kaXY+YDsKICAgIHJlbmRlckltYWdlU2V0dXAoJCgic2V0dXBIb3N0IiksIHN0KTsKICAgIGxvYWRHYWxsZXJ5KCk7CiAgICByZXR1cm47CiAgfQog"
    "IGhvc3QuaW5uZXJIVE1MID0gYDxkaXYgY2xhc3M9ImltZ2xheW91dCI+CiAgICA8ZGl2IGNsYXNzPSJnZW5wYW5lbCI+CiAgICAgIDxkaXYgY2xhc3M9"
    "ImNhcmQiPgogICAgICAgIDxkaXYgY2xhc3M9ImltZ3ByZXZpZXciIGlkPSJpbWdQcmV2aWV3Ij48ZGl2IGNsYXNzPSJwaCI+WW91ciBpbWFnZSBhcHBl"
    "YXJzIGhlcmU8L2Rpdj48L2Rpdj4KICAgICAgICA8ZGl2IGNsYXNzPSJmaWVsZCI+PGxhYmVsPlByb21wdDwvbGFiZWw+CiAgICAgICAgICA8dGV4dGFy"
    "ZWEgY2xhc3M9InRhIiBpZD0iaW1nUHJvbXB0IiBwbGFjZWhvbGRlcj0iYSBsaWdodGhvdXNlIGF0IGR1c2ssIG9pbCBwYWludGluZywgZHJhbWF0aWMg"
    "c2t5Ij48L3RleHRhcmVhPjwvZGl2PgogICAgICAgIDxkaXYgY2xhc3M9ImZpZWxkIj48bGFiZWw+TW9kZWw8L2xhYmVsPgogICAgICAgICAgPHNlbGVj"
    "dCBjbGFzcz0ic2VsIiBpZD0iaW1nTW9kZWwiPiR7c3RhdGUuaW1nUHJlc2V0cy5tYXAocD0+CiAgICAgICAgICAgIGA8b3B0aW9uIHZhbHVlPSIke3Au"
    "a2V5fSI+JHtlc2MocC5sYWJlbCl9PC9vcHRpb24+YCkuam9pbigiIil9PC9zZWxlY3Q+PC9kaXY+CiAgICAgICAgPGRpdiBjbGFzcz0iZmllbGQiPjxs"
    "YWJlbD5TaXplPC9sYWJlbD48ZGl2IGNsYXNzPSJzaXplZ3JpZCIgaWQ9InNpemVHcmlkIj48L2Rpdj48L2Rpdj4KICAgICAgICA8ZGl2IGNsYXNzPSJm"
    "aWVsZCI+PGxhYmVsPlNlZWQgPHNwYW4gY2xhc3M9ImRpbSI+KGJsYW5rID0gcmFuZG9tKTwvc3Bhbj48L2xhYmVsPgogICAgICAgICAgPGlucHV0IGNs"
    "YXNzPSJpbnAgbW9ubyIgaWQ9ImltZ1NlZWQiIHBsYWNlaG9sZGVyPSJyYW5kb20iPjwvZGl2PgogICAgICAgIDxidXR0b24gY2xhc3M9ImJ0biBwcmlt"
    "YXJ5IiBpZD0iZ2VuQnRuIiBzdHlsZT0id2lkdGg6MTAwJSI+CiAgICAgICAgICA8c3ZnIHdpZHRoPSIxNSIgaGVpZ2h0PSIxNSIgdmlld0JveD0iMCAw"
    "IDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9ImN1cnJlbnRDb2xvciIgc3Ryb2tlLXdpZHRoPSIyIj48cGF0aCBkPSJNNSAzdjRNMyA1aDRNNiAxN3Y0"
    "TTQgMTloNE0xMyAzbDIuNSA2LjVMMjIgMTJsLTYuNSAyLjVMMTMgMjFsLTIuNS02LjVMNCAxMmw2LjUtMi41TDEzIDNaIi8+PC9zdmc+CiAgICAgICAg"
    "ICBHZW5lcmF0ZTwvYnV0dG9uPgogICAgICAgIDxkaXYgaWQ9ImdlblByb2dyZXNzIj48L2Rpdj4KICAgICAgPC9kaXY+CiAgICAgIDxkaXYgY2xhc3M9"
    "ImhpbnQiPkRldmljZTogPHNwYW4gY2xhc3M9Im1vbm8iPiR7KHN0LmRldmljZXx8ImNwdSIpLnRvVXBwZXJDYXNlKCl9PC9zcGFuPiDCtwogICAgICAg"
    "IGltYWdlcyBzYXZlIHRvIHlvdXIgZ2FsbGVyeSBhdXRvbWF0aWNhbGx5PC9kaXY+CiAgICA8L2Rpdj4KICAgIDxkaXY+PGRpdiBjbGFzcz0ic2VjdGlv"
    "bi10aXRsZSIgc3R5bGU9Im1hcmdpbi10b3A6MCI+R2FsbGVyeTwvZGl2PgogICAgICA8ZGl2IGNsYXNzPSJnYWxsZXJ5IiBpZD0iZ2FsbGVyeSI+PC9k"
    "aXY+PC9kaXY+CiAgPC9kaXY+YDsKICBjb25zdCBzaXplcyA9IFtbNTEyLDUxMiwic3F1YXJlIl0sWzUxMiw3NjgsInBvcnRyYWl0Il0sWzc2OCw1MTIs"
    "ImxhbmRzY2FwZSJdXTsKICBzdGF0ZS5zZWxJbWdTaXplID0gc3RhdGUuc2VsSW1nU2l6ZSB8fCAiNTEyeDUxMiI7CiAgJCgic2l6ZUdyaWQiKS5pbm5l"
    "ckhUTUwgPSBzaXplcy5tYXAoKFt3LGgsbF0pPT4KICAgIGA8ZGl2IGNsYXNzPSJzaXplb3B0JHtzdGF0ZS5zZWxJbWdTaXplPT09dysieCIraD8iIG9u"
    "IjoiIn0iIGRhdGEtc2l6ZT0iJHt3fXgke2h9Ij4KICAgICAgJHtsfTxzcGFuIGNsYXNzPSJtb25vIj4ke3d9w5cke2h9PC9zcGFuPjwvZGl2PmApLmpv"
    "aW4oIiIpOwogICQoInNpemVHcmlkIikucXVlcnlTZWxlY3RvckFsbCgiLnNpemVvcHQiKS5mb3JFYWNoKG89Pm8ub25jbGljaz0oKT0+ewogICAgc3Rh"
    "dGUuc2VsSW1nU2l6ZT1vLmRhdGFzZXQuc2l6ZTsKICAgICQoInNpemVHcmlkIikucXVlcnlTZWxlY3RvckFsbCgiLnNpemVvcHQiKS5mb3JFYWNoKHg9"
    "PnguY2xhc3NMaXN0LnJlbW92ZSgib24iKSk7CiAgICBvLmNsYXNzTGlzdC5hZGQoIm9uIik7IH0pOwogICQoImdlbkJ0biIpLm9uY2xpY2sgPSBnZW5l"
    "cmF0ZUltYWdlOwogIGxvYWRHYWxsZXJ5KCk7Cn0KZnVuY3Rpb24gcmVuZGVySW1hZ2VTZXR1cChob3N0LCBzdCl7CiAgY29uc3QgcnVubmluZyA9IHN0"
    "Lmluc3RhbGxfam9iICYmIHN0Lmluc3RhbGxfam9iLnN0YXR1cz09PSJydW5uaW5nIjsKICBob3N0LmlubmVySFRNTCA9IGA8ZGl2IGNsYXNzPSJjYXJk"
    "IHNldHVwLWNhcmQiPgogICAgPGRpdiBjbGFzcz0iaWMiPvCfjqg8L2Rpdj4KICAgIDxoMyBzdHlsZT0ibWFyZ2luLWJvdHRvbTo4cHgiPlR1cm4gb24g"
    "aW1hZ2UgZ2VuZXJhdGlvbjwvaDM+CiAgICA8cCBjbGFzcz0ibXV0ZWQiIHN0eWxlPSJtYXgtd2lkdGg6NTJjaDttYXJnaW46MCBhdXRvIDZweCI+VGhp"
    "cyBpbnN0YWxscyBTdGFibGUgRGlmZnVzaW9uCiAgICAgIChQeVRvcmNoICsgZGlmZnVzZXJzKSBpbnRvIEhlb3J0aCdzIHByaXZhdGUgZW52aXJvbm1l"
    "bnQuIEl0J3MgYSBvbmUtdGltZQogICAgICBkb3dubG9hZCBvZiBhIGZldyBnaWdhYnl0ZXMuIE1vZGVscyBkb3dubG9hZCB0aGUgZmlyc3QgdGltZSB5"
    "b3UgZ2VuZXJhdGUuPC9wPgogICAgPGRpdiBzdHlsZT0ibWFyZ2luLXRvcDoxNnB4Ij4KICAgICAgPGJ1dHRvbiBjbGFzcz0iYnRuIHByaW1hcnkiIGlk"
    "PSJzZXR1cEltZ0J0biIgJHtydW5uaW5nPyJkaXNhYmxlZCI6IiJ9PgogICAgICAgICR7cnVubmluZz8iSW5zdGFsbGluZ+KApiI6Ikluc3RhbGwgaW1h"
    "Z2UgZ2VuZXJhdGlvbiJ9PC9idXR0b24+PC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJpbnN0YWxsbG9nIiBpZD0iaW1nTG9nIiAke3N0Lmluc3RhbGxfam9i"
    "PyIiOiJoaWRkZW4ifT4kewogICAgICBzdC5pbnN0YWxsX2pvYj9lc2MoKHN0Lmluc3RhbGxfam9iLmxvZ3x8W10pLmpvaW4oIlxuIikpOiIifTwvZGl2"
    "PgogIDwvZGl2PmA7CiAgJCgic2V0dXBJbWdCdG4iKS5vbmNsaWNrID0gYXN5bmMoKT0+ewogICAgJCgic2V0dXBJbWdCdG4iKS5kaXNhYmxlZD10cnVl"
    "OyAkKCJzZXR1cEltZ0J0biIpLnRleHRDb250ZW50PSJJbnN0YWxsaW5n4oCmIjsKICAgICQoImltZ0xvZyIpLmhpZGRlbj1mYWxzZTsKICAgIHRyeXsg"
    "YXdhaXQgcG9zdCgiL2FwaS9pbWFnZXMvc2V0dXAiKTsgcG9sbEltZ0luc3RhbGwoKTsgfQogICAgY2F0Y2goZSl7IHRvYXN0KGUubWVzc2FnZSwiZXJy"
    "Iik7IH0KICB9OwogIGlmKHJ1bm5pbmcpIHBvbGxJbWdJbnN0YWxsKCk7Cn0KYXN5bmMgZnVuY3Rpb24gcG9sbEltZ0luc3RhbGwoKXsKICBjb25zdCBs"
    "b2cgPSAkKCJpbWdMb2ciKTsKICBjb25zdCB0ID0gc2V0SW50ZXJ2YWwoYXN5bmMoKT0+ewogICAgdHJ5eyBjb25zdCBzdCA9IGF3YWl0IGFwaSgiL2Fw"
    "aS9pbWFnZXMvc3RhdHVzIik7CiAgICAgIGlmKGxvZyAmJiBzdC5pbnN0YWxsX2pvYil7IGxvZy50ZXh0Q29udGVudCA9IChzdC5pbnN0YWxsX2pvYi5s"
    "b2d8fFtdKS5qb2luKCJcbiIpOwogICAgICAgIGxvZy5zY3JvbGxUb3AgPSBsb2cuc2Nyb2xsSGVpZ2h0OyB9CiAgICAgIGlmKHN0Lmluc3RhbGxlZCl7"
    "IGNsZWFySW50ZXJ2YWwodCk7IHRvYXN0KCJJbWFnZSBnZW5lcmF0aW9uIHJlYWR5Iiwib2siKTsgbG9hZEltYWdlcygpOyB9CiAgICAgIGVsc2UgaWYo"
    "c3QuaW5zdGFsbF9qb2IgJiYgc3QuaW5zdGFsbF9qb2Iuc3RhdHVzPT09ImZhaWxlZCIpeyBjbGVhckludGVydmFsKHQpOwogICAgICAgIHRvYXN0KCJJ"
    "bnN0YWxsIGZhaWxlZCDigJQgc2VlIHRoZSBsb2ciLCJlcnIiKTsKICAgICAgICBjb25zdCBiPSQoInNldHVwSW1nQnRuIik7IGlmKGIpe2IuZGlzYWJs"
    "ZWQ9ZmFsc2U7Yi50ZXh0Q29udGVudD0iUmV0cnkgaW5zdGFsbCI7fSB9CiAgICB9Y2F0Y2goZSl7IGNsZWFySW50ZXJ2YWwodCk7IH0KICB9LCAxNTAw"
    "KTsKfQphc3luYyBmdW5jdGlvbiBnZW5lcmF0ZUltYWdlKCl7CiAgY29uc3QgcHJvbXB0ID0gJCgiaW1nUHJvbXB0IikudmFsdWUudHJpbSgpOwogIGlm"
    "KCFwcm9tcHQpeyB0b2FzdCgiV3JpdGUgYSBwcm9tcHQgZmlyc3QiLCJlcnIiKTsgcmV0dXJuOyB9CiAgY29uc3QgW3csaF0gPSAoc3RhdGUuc2VsSW1n"
    "U2l6ZXx8IjUxMng1MTIiKS5zcGxpdCgieCIpLm1hcChOdW1iZXIpOwogIGNvbnN0IG1vZGVsID0gJCgiaW1nTW9kZWwiKS52YWx1ZTsKICBjb25zdCBz"
    "ZWVkViA9ICQoImltZ1NlZWQiKS52YWx1ZS50cmltKCk7CiAgY29uc3QgYnRuID0gJCgiZ2VuQnRuIik7IGJ0bi5kaXNhYmxlZD10cnVlOwogIGNvbnN0"
    "IHByb2cgPSAkKCJnZW5Qcm9ncmVzcyIpOwogIHByb2cuaW5uZXJIVE1MID0gJzxkaXYgY2xhc3M9InB1bGxib3giPjxkaXYgY2xhc3M9InN0YXQiPjxz"
    "cGFuIGNsYXNzPSJzIj5TdGFydGluZ+KApjwvc3Bhbj4nKwogICAgJzxzcGFuIGNsYXNzPSJwIj48L3NwYW4+PC9kaXY+PGRpdiBjbGFzcz0iYmFyIj48"
    "aT48L2k+PC9kaXY+JysKICAgICc8ZGl2IGNsYXNzPSJyb3ciIHN0eWxlPSJqdXN0aWZ5LWNvbnRlbnQ6ZmxleC1lbmQ7bWFyZ2luLXRvcDo4cHgiPicr"
    "CiAgICAnPGJ1dHRvbiBjbGFzcz0iYnRuIHNtIGRhbmdlciIgaWQ9ImltZ1N0b3BCdG4iPlN0b3A8L2J1dHRvbj48L2Rpdj48L2Rpdj4nOwogIGNvbnN0"
    "IGJhciA9IHByb2cucXVlcnlTZWxlY3RvcigiaSIpLCBzdGF0ID0gcHJvZy5xdWVyeVNlbGVjdG9yKCIucyIpLCBwYyA9IHByb2cucXVlcnlTZWxlY3Rv"
    "cigiLnAiKTsKICAkKCJpbWdTdG9wQnRuIikub25jbGljayA9IGFzeW5jKCk9PnsgJCgiaW1nU3RvcEJ0biIpLmRpc2FibGVkPXRydWU7CiAgICAkKCJp"
    "bWdTdG9wQnRuIikudGV4dENvbnRlbnQ9IlN0b3BwaW5n4oCmIjsgdHJ5eyBhd2FpdCBwb3N0KCIvYXBpL2ltYWdlcy9jYW5jZWwiKTsgfWNhdGNoKGUp"
    "e30gfTsKICAkKCJpbWdQcmV2aWV3IikuaW5uZXJIVE1MID0gJzxkaXYgY2xhc3M9InNwaW4iIHN0eWxlPSJ3aWR0aDoyNnB4O2hlaWdodDoyNnB4Ij48"
    "L2Rpdj4nOwogIHRyeXsKICAgIGF3YWl0IHN0cmVhbU5ESlNPTigiL2FwaS9pbWFnZXMvZ2VuZXJhdGUiLAogICAgICB7cHJvbXB0LCBtb2RlbCwgd2lk"
    "dGg6dywgaGVpZ2h0OmgsIHNlZWQ6IHNlZWRWPT09IiI/bnVsbDpOdW1iZXIoc2VlZFYpfSwKICAgICAgKG8pPT57CiAgICAgICAgaWYoby50eXBlPT09"
    "InN0YXR1cyIpeyBzdGF0LnRleHRDb250ZW50PW8udGV4dDsgfQogICAgICAgIGVsc2UgaWYoby50eXBlPT09InN0ZXAiKXsgaWYoby50b3RhbCl7IGNv"
    "bnN0IHA9by5zdGVwL28udG90YWwqMTAwOwogICAgICAgICAgYmFyLnN0eWxlLndpZHRoPXArIiUiOyBwYy50ZXh0Q29udGVudD1vLnN0ZXArIi8iK28u"
    "dG90YWw7IHN0YXQudGV4dENvbnRlbnQ9IkdlbmVyYXRpbmciOyB9IH0KICAgICAgICBlbHNlIGlmKG8udHlwZT09PSJkb25lIil7IGJhci5zdHlsZS53"
    "aWR0aD0iMTAwJSI7IHN0YXQudGV4dENvbnRlbnQ9IkRvbmUiOwogICAgICAgICAgY29uc3QgaW09by5pbWFnZTsgJCgiaW1nUHJldmlldyIpLmlubmVy"
    "SFRNTCA9CiAgICAgICAgICAgIGA8aW1nIHNyYz0iL2FwaS9pbWFnZXMvZmlsZS8ke2ltLmZpbGVuYW1lfT90PSR7RGF0ZS5ub3coKX0iIHRpdGxlPSJD"
    "bGljayB0byB2aWV3IGxhcmdlciI+YDsKICAgICAgICAgICQoImltZ1ByZXZpZXciKS5xdWVyeVNlbGVjdG9yKCJpbWciKS5vbmNsaWNrID0gKCk9PnZp"
    "ZXdJbWFnZShpbSk7CiAgICAgICAgICB0b2FzdCgiSW1hZ2Ugc2F2ZWQgdG8gZ2FsbGVyeSIsIm9rIik7IGxvYWRHYWxsZXJ5KCk7CiAgICAgICAgICBz"
    "ZXRUaW1lb3V0KCgpPT57cHJvZy5pbm5lckhUTUw9IiI7fSwxNTAwKTsgfQogICAgICAgIGVsc2UgaWYoby50eXBlPT09ImNhbmNlbGxlZCIpeyBzdGF0"
    "LnRleHRDb250ZW50PSJTdG9wcGVkIjsKICAgICAgICAgICQoImltZ1ByZXZpZXciKS5pbm5lckhUTUw9JzxkaXYgY2xhc3M9InBoIj5TdG9wcGVkIOKA"
    "lCBub3RoaW5nIHdhcyBzYXZlZDwvZGl2Pic7CiAgICAgICAgICB0b2FzdCgiR2VuZXJhdGlvbiBzdG9wcGVkIiwib2siKTsKICAgICAgICAgIHNldFRp"
    "bWVvdXQoKCk9Pntwcm9nLmlubmVySFRNTD0iIjt9LDEyMDApOyB9CiAgICAgICAgZWxzZSBpZihvLnR5cGU9PT0iZXJyb3IiKXsgdGhyb3cgbmV3IEVy"
    "cm9yKG8uZXJyb3IpOyB9CiAgICAgIH0pOwogIH1jYXRjaChlKXsKICAgIHRvYXN0KCJHZW5lcmF0aW9uIGZhaWxlZDogIitlLm1lc3NhZ2UsImVyciIp"
    "OwogICAgc3RhdC50ZXh0Q29udGVudD0iRmFpbGVkOiAiK2UubWVzc2FnZTsKICAgICQoImltZ1ByZXZpZXciKS5pbm5lckhUTUw9JzxkaXYgY2xhc3M9"
    "InBoIj5HZW5lcmF0aW9uIGZhaWxlZDwvZGl2Pic7CiAgfWZpbmFsbHl7IGJ0bi5kaXNhYmxlZD1mYWxzZTsgfQp9CmFzeW5jIGZ1bmN0aW9uIGxvYWRH"
    "YWxsZXJ5KCl7CiAgdHJ5ewogICAgY29uc3QgciA9IGF3YWl0IGFwaSgiL2FwaS9pbWFnZXMiKTsKICAgIGNvbnN0IGcgPSAkKCJnYWxsZXJ5Iik7IGlm"
    "KCFnKSByZXR1cm47CiAgICBpZighci5pbWFnZXMubGVuZ3RoKXsgZy5pbm5lckhUTUwgPQogICAgICAnPGRpdiBjbGFzcz0iZW1wdHkiIHN0eWxlPSJn"
    "cmlkLWNvbHVtbjoxLy0xIj48ZGl2IGNsYXNzPSJiaWciPvCflrw8L2Rpdj5ObyBpbWFnZXMgeWV0PC9kaXY+JzsgcmV0dXJuOyB9CiAgICBnLmlubmVy"
    "SFRNTCA9ICIiOwogICAgci5pbWFnZXMuZm9yRWFjaChpbT0+ewogICAgICBjb25zdCBpdCA9IGVsKCJkaXYiLCJnaXRlbSIpOwogICAgICBpdC5pbm5l"
    "ckhUTUwgPSBgPGltZyBzcmM9Ii9hcGkvaW1hZ2VzL2ZpbGUvJHtpbS5maWxlbmFtZX0iIGxvYWRpbmc9ImxhenkiPgogICAgICAgIDxkaXYgY2xhc3M9"
    "Im92Ij48ZGl2IGNsYXNzPSJwciI+JHtlc2MoaW0ucHJvbXB0fHwiIil9PC9kaXY+PC9kaXY+CiAgICAgICAgPGRpdiBjbGFzcz0icm0iIHRpdGxlPSJS"
    "ZW1vdmUiPgogICAgICAgIDxzdmcgd2lkdGg9IjE0IiBoZWlnaHQ9IjE0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIGZpbGw9Im5vbmUiIHN0cm9rZT0iY3Vy"
    "cmVudENvbG9yIiBzdHJva2Utd2lkdGg9IjIiPjxwYXRoIGQ9Ik0xOCA2IDYgMThNNiA2bDEyIDEyIi8+PC9zdmc+PC9kaXY+YDsKICAgICAgaXQub25j"
    "bGljayA9ICgpPT52aWV3SW1hZ2UoaW0pOwogICAgICBpdC5xdWVyeVNlbGVjdG9yKCIucm0iKS5vbmNsaWNrID0gYXN5bmMoZSk9PnsgZS5zdG9wUHJv"
    "cGFnYXRpb24oKTsKICAgICAgICBhd2FpdCBkZWwoIi9hcGkvaW1hZ2VzLyIraW0uaWQpOyB0b2FzdCgiSW1hZ2UgcmVtb3ZlZCIsIm9rIik7IGxvYWRH"
    "YWxsZXJ5KCk7IH07CiAgICAgIGcuYXBwZW5kQ2hpbGQoaXQpOwogICAgfSk7CiAgfWNhdGNoKGUpe30KfQpmdW5jdGlvbiB2aWV3SW1hZ2UoaW0pewog"
    "IGNvbnN0IGJvZHkgPSBtb2RhbCh7dGl0bGU6IkltYWdlIiwgd2lkZTp0cnVlLCBib2R5SFRNTDpgCiAgICA8aW1nIHNyYz0iL2FwaS9pbWFnZXMvZmls"
    "ZS8ke2ltLmZpbGVuYW1lfSIgaWQ9Im1vZGFsSW1nIiB0aXRsZT0iQ2xpY2sgdG8gb3BlbiB0aGUgb3JpZ2luYWwgZmlsZSIKICAgICAgc3R5bGU9Indp"
    "ZHRoOjEwMCU7bWF4LWhlaWdodDo3MHZoO29iamVjdC1maXQ6Y29udGFpbjtib3JkZXItcmFkaXVzOjEwcHg7Ym9yZGVyOjFweCBzb2xpZCB2YXIoLS1s"
    "aW5lKTtjdXJzb3I6em9vbS1pbjtiYWNrZ3JvdW5kOnZhcigtLWluaykiPgogICAgPGRpdiBjbGFzcz0iZmllbGQiIHN0eWxlPSJtYXJnaW4tdG9wOjE0"
    "cHgiPjxsYWJlbD5Qcm9tcHQ8L2xhYmVsPgogICAgICA8ZGl2IGNsYXNzPSJtb25vIiBzdHlsZT0iZm9udC1zaXplOjEycHg7YmFja2dyb3VuZDp2YXIo"
    "LS1wYW5lbC0yKTtwYWRkaW5nOjEwcHg7Ym9yZGVyLXJhZGl1czo4cHg7Ym9yZGVyOjFweCBzb2xpZCB2YXIoLS1saW5lKSI+JHtlc2MoaW0ucHJvbXB0"
    "fHwiIil9PC9kaXY+PC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJyb3ciIHN0eWxlPSJnYXA6MTZweDtmb250LXNpemU6MTJweCI+CiAgICAgIDxzcGFuIGNs"
    "YXNzPSJkaW0gbW9ubyI+JHtpbS53aWR0aH3DlyR7aW0uaGVpZ2h0fTwvc3Bhbj4KICAgICAgPHNwYW4gY2xhc3M9ImRpbSBtb25vIj5zZWVkICR7aW0u"
    "c2VlZH08L3NwYW4+CiAgICAgIDxzcGFuIGNsYXNzPSJkaW0gbW9ubyI+JHtlc2MoaW0ubW9kZWx8fCIiKX08L3NwYW4+PC9kaXY+YCwKICAgIGFjdGlv"
    "bnM6W3tsYWJlbDoiT3BlbiBmaWxlIiwgb25DbGljazooKT0+d2luZG93Lm9wZW4oIi9hcGkvaW1hZ2VzL2ZpbGUvIitpbS5maWxlbmFtZSl9LAogICAg"
    "ICB7bGFiZWw6IlJlbW92ZSIsIGNsczoiZGFuZ2VyIiwgb25DbGljazphc3luYygpPT57IGNsb3NlTW9kYWwoKTsKICAgICAgICBhd2FpdCBkZWwoIi9h"
    "cGkvaW1hZ2VzLyIraW0uaWQpOyB0b2FzdCgiUmVtb3ZlZCIsIm9rIik7IGxvYWRHYWxsZXJ5KCk7IH19XX0pOwogIGJvZHkucXVlcnlTZWxlY3Rvcigi"
    "I21vZGFsSW1nIikub25jbGljayA9ICgpPT53aW5kb3cub3BlbigiL2FwaS9pbWFnZXMvZmlsZS8iK2ltLmZpbGVuYW1lKTsKfQpsb2FkZXJzLmltYWdl"
    "cyA9IGxvYWRJbWFnZXM7CgovKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT0KICAgS05P"
    "V0xFREdFIChSQUcpCiAgID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PSAqLwphc3luYyBm"
    "dW5jdGlvbiBsb2FkS25vd2xlZGdlKCl7CiAgY29uc3QgaG9zdCA9ICQoImtub3dsZWRnZUJvZHkiKTsKICBob3N0LmlubmVySFRNTCA9ICc8ZGl2IGNs"
    "YXNzPSJyb3ciPjxkaXYgY2xhc3M9InNwaW4iPjwvZGl2PjxzcGFuIGNsYXNzPSJtdXRlZCI+TG9hZGluZ+KApjwvc3Bhbj48L2Rpdj4nOwogIGxldCBy"
    "OwogIHRyeXsgciA9IGF3YWl0IGFwaSgiL2FwaS9yYWcvZG9jcyIpOyB9Y2F0Y2goZSl7IGhvc3QuaW5uZXJIVE1MPWVyckNhcmQoZS5tZXNzYWdlKTsg"
    "cmV0dXJuOyB9CiAgbGV0IHdhcm4gPSAiIjsKICBpZighci5lbWJlZF9yZWFkeSl7CiAgICB3YXJuID0gYDxkaXYgY2xhc3M9ImNhcmQiIHN0eWxlPSJi"
    "b3JkZXItY29sb3I6dmFyKC0tc2lnbmFsLWRpbSk7bWFyZ2luLWJvdHRvbToxNnB4Ij4KICAgICAgPGRpdiBjbGFzcz0icm93IiBzdHlsZT0iZ2FwOjlw"
    "eDttYXJnaW4tYm90dG9tOjZweCI+PHNwYW4gY2xhc3M9ImRvdCB3YXJuIj48L3NwYW4+CiAgICAgIDxiPkVtYmVkZGluZyBtb2RlbCBuZWVkZWQ8L2I+"
    "PC9kaXY+CiAgICAgIDxwIGNsYXNzPSJtdXRlZCIgc3R5bGU9Im1hcmdpbjowIj5UbyBzZWFyY2ggZG9jdW1lbnRzLCBIZW9ydGggbmVlZHMgdGhlIGVt"
    "YmVkZGluZwogICAgICAgIG1vZGVsIDxzcGFuIGNsYXNzPSJtb25vIj4ke2VzYyhyLmVtYmVkX21vZGVsKX08L3NwYW4+LgogICAgICAgICR7ci5vbGxh"
    "bWFfdXA/IiI6IlN0YXJ0IE9sbGFtYSwgdGhlbiAifWRvd25sb2FkIGl0IGZyb20gdGhlIE1vZGVscyBwYWdlCiAgICAgICAgKHNlYXJjaCDigJxlbWJl"
    "ZOKAnSkuIFlvdSBjYW4gc3RpbGwgdXBsb2FkIG5vdywgYnV0IHNlYXJjaCBuZWVkcyBpdC48L3A+PC9kaXY+YDsKICB9CiAgaG9zdC5pbm5lckhUTUwg"
    "PSB3YXJuICsgYAogICAgPGRpdiBjbGFzcz0iZHJvcHpvbmUiIGlkPSJkcm9wem9uZSI+CiAgICAgIDxzdmcgd2lkdGg9IjMwIiBoZWlnaHQ9IjMwIiB2"
    "aWV3Qm94PSIwIDAgMjQgMjQiIGZpbGw9Im5vbmUiIHN0cm9rZT0iY3VycmVudENvbG9yIiBzdHJva2Utd2lkdGg9IjEuNiIgc3R5bGU9Im1hcmdpbi1i"
    "b3R0b206OHB4Ij48cGF0aCBkPSJNMTIgMTZWNG0wIDAgNCA0bS00LTQtNCA0Ii8+PHBhdGggZD0iTTQgMTZ2MmEyIDIgMCAwIDAgMiAyaDEyYTIgMiAw"
    "IDAgMCAyLTJ2LTIiLz48L3N2Zz4KICAgICAgPGRpdiBzdHlsZT0iZm9udC13ZWlnaHQ6NjAwIj5Ecm9wIGZpbGVzIGhlcmUgb3IgY2xpY2sgdG8gYnJv"
    "d3NlPC9kaXY+CiAgICAgIDxkaXYgY2xhc3M9ImhpbnQiPlBERiwgVFhULCBNYXJrZG93biwgY29kZSDigJQgdXAgdG8gNTAgTUIgZWFjaDwvZGl2Pgog"
    "ICAgICA8aW5wdXQgdHlwZT0iZmlsZSIgaWQ9ImZpbGVJbnB1dCIgbXVsdGlwbGUgaGlkZGVuCiAgICAgICAgYWNjZXB0PSIucGRmLC50eHQsLm1kLC5t"
    "YXJrZG93biwucHksLmpzLC50cywuanNvbiwuY3N2LC5odG1sLC5jc3MsLmphdmEsLmMsLmNwcCwuZ28sLnJzLC5yYiwuc2gsLnlhbWwsLnltbCwueG1s"
    "Ij4KICAgIDwvZGl2PgogICAgPGRpdiBpZD0idXBsb2FkUHJvZyI+PC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJzZWN0aW9uLXRpdGxlIj5Zb3VyIGRvY3Vt"
    "ZW50cyR7ci5kb2NzLmxlbmd0aD8iIMK3ICIrci5kb2NzLmxlbmd0aDoiIn08L2Rpdj4KICAgIDxkaXYgaWQ9ImRvY0xpc3QiPjwvZGl2PmA7CiAgY29u"
    "c3QgZHogPSAkKCJkcm9wem9uZSIpLCBmaSA9ICQoImZpbGVJbnB1dCIpOwogIGR6Lm9uY2xpY2sgPSAoKT0+ZmkuY2xpY2soKTsKICBmaS5vbmNoYW5n"
    "ZSA9ICgpPT57IGhhbmRsZUZpbGVzKFsuLi5maS5maWxlc10pOyBmaS52YWx1ZT0iIjsgfTsKICBkei5vbmRyYWdvdmVyID0gKGUpPT57IGUucHJldmVu"
    "dERlZmF1bHQoKTsgZHouY2xhc3NMaXN0LmFkZCgiZHJhZyIpOyB9OwogIGR6Lm9uZHJhZ2xlYXZlID0gKCk9PmR6LmNsYXNzTGlzdC5yZW1vdmUoImRy"
    "YWciKTsKICBkei5vbmRyb3AgPSAoZSk9PnsgZS5wcmV2ZW50RGVmYXVsdCgpOyBkei5jbGFzc0xpc3QucmVtb3ZlKCJkcmFnIik7CiAgICBoYW5kbGVG"
    "aWxlcyhbLi4uZS5kYXRhVHJhbnNmZXIuZmlsZXNdKTsgfTsKICByZW5kZXJEb2NzKHIuZG9jcyk7Cn0KZnVuY3Rpb24gcmVuZGVyRG9jcyhkb2NzKXsK"
    "ICBjb25zdCBsaXN0ID0gJCgiZG9jTGlzdCIpOyBpZighbGlzdCkgcmV0dXJuOwogIGlmKCFkb2NzLmxlbmd0aCl7IGxpc3QuaW5uZXJIVE1MID0KICAg"
    "ICc8ZGl2IGNsYXNzPSJlbXB0eSI+PGRpdiBjbGFzcz0iYmlnIj7wn5OEPC9kaXY+Tm8gZG9jdW1lbnRzIHlldDwvZGl2Pic7IHJldHVybjsgfQogIGxp"
    "c3QuaW5uZXJIVE1MID0gIiI7CiAgZG9jcy5mb3JFYWNoKGQ9PnsKICAgIGNvbnN0IHJvdyA9IGVsKCJkaXYiLCJkb2Nyb3ciKTsKICAgIHJvdy5pbm5l"
    "ckhUTUwgPSBgPGRpdiBjbGFzcz0iZGljIj4KICAgICAgPHN2ZyB3aWR0aD0iMTgiIGhlaWdodD0iMTgiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0i"
    "bm9uZSIgc3Ryb2tlPSJjdXJyZW50Q29sb3IiIHN0cm9rZS13aWR0aD0iMS44Ij48cGF0aCBkPSJNMTQgM3Y1aDUiLz48cGF0aCBkPSJNMTQgM0g2YTIg"
    "MiAwIDAgMC0yIDJ2MTRhMiAyIDAgMCAwIDIgMmgxMmEyIDIgMCAwIDAgMi0yVjhsLTYtNVoiLz48L3N2Zz48L2Rpdj4KICAgICAgPGRpdiBjbGFzcz0i"
    "ZGkiPjxkaXYgY2xhc3M9Im5tIj4ke2VzYyhkLm5hbWUpfTwvZGl2PgogICAgICAgIDxkaXYgY2xhc3M9Im10Ij4ke2QuY2h1bmtzfSBjaHVua3Mgwrcg"
    "YWRkZWQgJHtuZXcgRGF0ZShkLmNyZWF0ZWQqMTAwMCkudG9Mb2NhbGVEYXRlU3RyaW5nKCl9PC9kaXY+PC9kaXY+YDsKICAgIGNvbnN0IGIgPSBlbCgi"
    "YnV0dG9uIiwiYnRuIGRhbmdlciBzbSIsIlJlbW92ZSIpOwogICAgYi5vbmNsaWNrID0gYXN5bmMoKT0+eyBhd2FpdCBkZWwoIi9hcGkvcmFnL2RvY3Mv"
    "IitkLmlkKTsgdG9hc3QoIlJlbW92ZWQgIitkLm5hbWUsIm9rIik7IGxvYWRLbm93bGVkZ2UoKTsgfTsKICAgIHJvdy5hcHBlbmRDaGlsZChiKTsgbGlz"
    "dC5hcHBlbmRDaGlsZChyb3cpOwogIH0pOwp9CmFzeW5jIGZ1bmN0aW9uIGhhbmRsZUZpbGVzKGZpbGVzKXsKICBjb25zdCBwcm9nID0gJCgidXBsb2Fk"
    "UHJvZyIpOwogIGZvcihjb25zdCBmIG9mIGZpbGVzKXsKICAgIGNvbnN0IHJvdyA9IGVsKCJkaXYiLCJkb2Nyb3ciKTsKICAgIHJvdy5pbm5lckhUTUwg"
    "PSBgPGRpdiBjbGFzcz0ic3BpbiI+PC9kaXY+PGRpdiBjbGFzcz0iZGkiPjxkaXYgY2xhc3M9Im5tIj4ke2VzYyhmLm5hbWUpfTwvZGl2PgogICAgICA8"
    "ZGl2IGNsYXNzPSJtdCI+UmVhZGluZyAmIGVtYmVkZGluZ+KApjwvZGl2PjwvZGl2PmA7CiAgICBwcm9nLmFwcGVuZENoaWxkKHJvdyk7CiAgICB0cnl7"
    "CiAgICAgIGNvbnN0IGZkID0gbmV3IEZvcm1EYXRhKCk7IGZkLmFwcGVuZCgiZmlsZSIsIGYpOwogICAgICBjb25zdCByID0gYXdhaXQgZmV0Y2goIi9h"
    "cGkvcmFnL3VwbG9hZCIsIHttZXRob2Q6IlBPU1QiLCBib2R5OmZkfSk7CiAgICAgIGNvbnN0IGogPSBhd2FpdCByLmpzb24oKTsKICAgICAgaWYoai5v"
    "ayl7IHRvYXN0KCJBZGRlZCAiK2YubmFtZSwib2siKTsgfQogICAgICBlbHNlIHsgdG9hc3Qoai5lcnJvcnx8IlVwbG9hZCBmYWlsZWQiLCJlcnIiKTsg"
    "fQogICAgfWNhdGNoKGUpeyB0b2FzdCgiVXBsb2FkIGZhaWxlZDogIitlLm1lc3NhZ2UsImVyciIpOyB9CiAgICByb3cucmVtb3ZlKCk7CiAgfQogIGxv"
    "YWRLbm93bGVkZ2UoKTsKfQpsb2FkZXJzLmtub3dsZWRnZSA9IGxvYWRLbm93bGVkZ2U7CgovKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09"
    "PT09PT09PT09PT09PT09PT09PT09PT09PT09PT0KICAgQUdFTlQgJiBUT09MUwogICA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09"
    "PT09PT09PT09PT09PT09PT09PT09PT0gKi8KY29uc3QgVE9PTF9JQ09OUyA9IHsKICB3ZWJfc2VhcmNoOifwn5SNJywgZmV0Y2hfdXJsOifwn4yQJywg"
    "Y2FsY3VsYXRvcjon8J+nricsIHNlYXJjaF9rbm93bGVkZ2U6J/Cfk5onLAogIGxpc3RfZmlsZXM6J/Cfk4EnLCByZWFkX2ZpbGU6J/Cfk4QnLCB3cml0"
    "ZV9maWxlOifinI/vuI8nLCBnZW5lcmF0ZV9pbWFnZTon8J+OqCcsCiAgcnVuX3B5dGhvbjon8J+QjScsIHJ1bl9zaGVsbDon4oyYJyB9Owphc3luYyBm"
    "dW5jdGlvbiBsb2FkVG9vbHMoKXsKICBjb25zdCBob3N0ID0gJCgidG9vbHNCb2R5Iik7CiAgaG9zdC5pbm5lckhUTUwgPSAnPGRpdiBjbGFzcz0icm93"
    "Ij48ZGl2IGNsYXNzPSJzcGluIj48L2Rpdj48c3BhbiBjbGFzcz0ibXV0ZWQiPkxvYWRpbmfigKY8L3NwYW4+PC9kaXY+JzsKICBsZXQgdDsKICB0cnl7"
    "IHQgPSBhd2FpdCBhcGkoIi9hcGkvdG9vbHMiKTsgfWNhdGNoKGUpeyBob3N0LmlubmVySFRNTD1lcnJDYXJkKGUubWVzc2FnZSk7IHJldHVybjsgfQog"
    "IGxldCBodG1sID0gYDxkaXYgY2xhc3M9ImNhcmQgcGFkLWxnIiBzdHlsZT0ibWFyZ2luLWJvdHRvbToxOHB4Ij4KICAgIDxkaXYgY2xhc3M9InNwcmVh"
    "ZCI+PGRpdj48Yj5Ib3cgdGhlIGFnZW50IHdvcmtzPC9iPgogICAgICA8cCBjbGFzcz0ibXV0ZWQiIHN0eWxlPSJtYXJnaW46NnB4IDAgMDttYXgtd2lk"
    "dGg6NjRjaCI+VHVybiBvbiA8Yj5BZ2VudDwvYj4gaW4gQ2hhdCBhbmQgdGhlCiAgICAgIG1vZGVsIGNhbiBjYWxsIHRoZXNlIHRvb2xzIHRvIHNlYXJj"
    "aCwgY2FsY3VsYXRlLCByZWFkIHlvdXIgZmlsZXMgYW5kIG1vcmUg4oCUIHRoZW4KICAgICAgYW5zd2VyIHVzaW5nIHdoYXQgaXQgZm91bmQuIFNvbWUg"
    "dG9vbHMgcmVzcGVjdCB5b3VyIHNhZmV0eSBzZXR0aW5ncyBiZWxvdy48L3A+PC9kaXY+PC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJyb3ciIHN0eWxlPSJn"
    "YXA6MjBweDttYXJnaW4tdG9wOjE0cHgiPgogICAgICA8c3BhbiBjbGFzcz0iY2hpcCAke3QuYWxsb3dfd2ViX3Rvb2xzPyJncm4iOiIifSI+d2ViIHRv"
    "b2xzICR7dC5hbGxvd193ZWJfdG9vbHM/Im9uIjoib2ZmIn08L3NwYW4+CiAgICAgIDxzcGFuIGNsYXNzPSJjaGlwICR7dC5hbGxvd19jb2RlX2V4ZWN1"
    "dGlvbj8iZ3JuIjoiIn0iPmNvZGUgZXhlY3V0aW9uICR7dC5hbGxvd19jb2RlX2V4ZWN1dGlvbj8ib24iOiJvZmYifTwvc3Bhbj4KICAgIDwvZGl2Pjwv"
    "ZGl2PgogICAgPGRpdiBjbGFzcz0ic2VjdGlvbi10aXRsZSI+QnVpbHQtaW4gdG9vbHM8L2Rpdj5gOwogIHQuYnVpbHRpbi5mb3JFYWNoKHRvb2w9PnsK"
    "ICAgIGNvbnN0IGdhdGVkID0gKHRvb2wubmFtZT09PSJydW5fcHl0aG9uInx8dG9vbC5uYW1lPT09InJ1bl9zaGVsbCIpOwogICAgaHRtbCArPSBgPGRp"
    "diBjbGFzcz0idG9vbHJvdyI+PGRpdiBjbGFzcz0idGljIj4ke1RPT0xfSUNPTlNbdG9vbC5uYW1lXXx8IuKamSJ9PC9kaXY+CiAgICAgIDxkaXYgY2xh"
    "c3M9InRpIj48ZGl2IGNsYXNzPSJubSI+JHtlc2ModG9vbC5uYW1lKX08L2Rpdj4KICAgICAgICA8ZGl2IGNsYXNzPSJkcyI+JHtlc2ModG9vbC5kZXNj"
    "KX08L2Rpdj48L2Rpdj4KICAgICAgJHtnYXRlZD9gPHNwYW4gY2xhc3M9ImNoaXAgJHt0LmFsbG93X2NvZGVfZXhlY3V0aW9uPyJncm4iOiIifSI+JHt0"
    "LmFsbG93X2NvZGVfZXhlY3V0aW9uPyJlbmFibGVkIjoibmVlZHMgb3B0LWluIn08L3NwYW4+YDoKICAgICAgICAodG9vbC5uYW1lPT09IndlYl9zZWFy"
    "Y2gifHx0b29sLm5hbWU9PT0iZmV0Y2hfdXJsIik/CiAgICAgICAgYDxzcGFuIGNsYXNzPSJjaGlwICR7dC5hbGxvd193ZWJfdG9vbHM/ImdybiI6IiJ9"
    "Ij4ke3QuYWxsb3dfd2ViX3Rvb2xzPyJvbiI6Im9mZiJ9PC9zcGFuPmA6CiAgICAgICAgJzxzcGFuIGNsYXNzPSJjaGlwIGdybiI+cmVhZHk8L3NwYW4+"
    "J308L2Rpdj5gOwogIH0pOwogIGh0bWwgKz0gYDxkaXYgY2xhc3M9InNlY3Rpb24tdGl0bGUiPldlYiBzZWFyY2ggPHNwYW4gY2xhc3M9ImNoaXAiPm9w"
    "dGlvbmFsIHVwZ3JhZGU8L3NwYW4+PC9kaXY+CiAgICA8ZGl2IGlkPSJzZWFyY2hCb2R5Ij48L2Rpdj4KICAgIDxkaXYgY2xhc3M9InNlY3Rpb24tdGl0"
    "bGUiPk1DUCBzZXJ2ZXJzIDxzcGFuIGNsYXNzPSJjaGlwIHZpbyI+TW9kZWwgQ29udGV4dCBQcm90b2NvbDwvc3Bhbj48L2Rpdj4KICAgIDxkaXYgaWQ9"
    "Im1jcEJvZHkiPjwvZGl2PmA7CiAgaG9zdC5pbm5lckhUTUwgPSBodG1sOwogIHJlbmRlck1DUCh0KTsKICByZW5kZXJTZWFyY2goKTsKfQpmdW5jdGlv"
    "biByZW5kZXJNQ1AodCl7CiAgY29uc3QgaG9zdCA9ICQoIm1jcEJvZHkiKTsKICBpZighdC5tY3BfaW5zdGFsbGVkKXsKICAgIGhvc3QuaW5uZXJIVE1M"
    "ID0gYDxkaXYgY2xhc3M9ImNhcmQgc2V0dXAtY2FyZCI+CiAgICAgIDxkaXYgY2xhc3M9ImljIj7wn5SMPC9kaXY+CiAgICAgIDxoMyBzdHlsZT0ibWFy"
    "Z2luLWJvdHRvbTo4cHgiPkVuYWJsZSBNQ1AgY29ubmVjdGlvbnM8L2gzPgogICAgICA8cCBjbGFzcz0ibXV0ZWQiIHN0eWxlPSJtYXgtd2lkdGg6NTRj"
    "aDttYXJnaW46MCBhdXRvIDZweCI+TUNQIGxldHMgdGhlIGFnZW50IHVzZSBleHRlcm5hbAogICAgICAgIHRvb2wgc2VydmVycyDigJQgZmlsZSBzeXN0"
    "ZW1zLCBicm93c2VycywgZGF0YWJhc2VzLCBBUElzIGFuZCBtb3JlLiBUaGlzIGluc3RhbGxzIHRoZQogICAgICAgIDxzcGFuIGNsYXNzPSJtb25vIj5t"
    "Y3A8L3NwYW4+IHBhY2thZ2UgaW50byBIZW9ydGgncyBlbnZpcm9ubWVudCAoc21hbGwsIG9uZS10aW1lKS48L3A+CiAgICAgIDxkaXYgc3R5bGU9Im1h"
    "cmdpbi10b3A6MTRweCI+PGJ1dHRvbiBjbGFzcz0iYnRuIHByaW1hcnkiIGlkPSJzZXR1cE1jcEJ0biI+SW5zdGFsbCBNQ1Agc3VwcG9ydDwvYnV0dG9u"
    "PjwvZGl2PgogICAgICA8ZGl2IGNsYXNzPSJpbnN0YWxsbG9nIiBpZD0ibWNwTG9nIiBoaWRkZW4+PC9kaXY+PC9kaXY+YDsKICAgICQoInNldHVwTWNw"
    "QnRuIikub25jbGljayA9IGFzeW5jKCk9PnsKICAgICAgJCgic2V0dXBNY3BCdG4iKS5kaXNhYmxlZD10cnVlOyAkKCJzZXR1cE1jcEJ0biIpLnRleHRD"
    "b250ZW50PSJJbnN0YWxsaW5n4oCmIjsKICAgICAgJCgibWNwTG9nIikuaGlkZGVuPWZhbHNlOwogICAgICB0cnl7IGF3YWl0IHBvc3QoIi9hcGkvbWNw"
    "L3NldHVwIik7IHBvbGxNY3BJbnN0YWxsKCk7IH1jYXRjaChlKXsgdG9hc3QoZS5tZXNzYWdlLCJlcnIiKTsgfQogICAgfTsKICAgIHJldHVybjsKICB9"
    "CiAgbGV0IGh0bWwgPSBgPHAgY2xhc3M9Im11dGVkIiBzdHlsZT0ibWFyZ2luOjAgMCAxNHB4Ij5Db25uZWN0ZWQgc2VydmVycyBleHBvc2UgdGhlaXIg"
    "dG9vbHMgdG8gdGhlCiAgICBhZ2VudCBhdXRvbWF0aWNhbGx5LiA8YnV0dG9uIGNsYXNzPSJidG4gc20gZ2hvc3QiIGlkPSJhZGRNY3BCdG4iPisgQWRk"
    "IHNlcnZlcjwvYnV0dG9uPjwvcD5gOwogIGlmKCF0Lm1jcF9zZXJ2ZXJzLmxlbmd0aCl7CiAgICBodG1sICs9ICc8ZGl2IGNsYXNzPSJlbXB0eSI+PGRp"
    "diBjbGFzcz0iYmlnIj7wn5SMPC9kaXY+Tm8gTUNQIHNlcnZlcnMgeWV0Ljxicj4nKwogICAgICAnPHNwYW4gY2xhc3M9ImhpbnQiPkFkZCBvbmUgdG8g"
    "Z2l2ZSB0aGUgYWdlbnQgbmV3IGFiaWxpdGllcy48L3NwYW4+PC9kaXY+JzsKICB9CiAgaHRtbCArPSAnPGRpdiBpZD0ic3J2TGlzdCI+PC9kaXY+JzsK"
    "ICBob3N0LmlubmVySFRNTCA9IGh0bWw7CiAgJCgiYWRkTWNwQnRuIikub25jbGljayA9IHNob3dBZGRNY3A7CiAgY29uc3QgbGlzdCA9ICQoInNydkxp"
    "c3QiKTsKICB0Lm1jcF9zZXJ2ZXJzLmZvckVhY2gocz0+ewogICAgY29uc3Qgc3J2ID0gZWwoImRpdiIsInNydiIpOwogICAgc3J2LmlubmVySFRNTCA9"
    "IGA8ZGl2IGNsYXNzPSJzaCI+PHNwYW4gY2xhc3M9ImRvdCAke3MuY29ubmVjdGVkPyJvbiI6KHMuZXJyb3I/Im9mZiI6IiIpfSI+PC9zcGFuPgogICAg"
    "ICA8ZGl2IHN0eWxlPSJmbGV4OjEiPjxkaXYgY2xhc3M9Im5tIj4ke2VzYyhzLm5hbWUpfTwvZGl2PgogICAgICAgIDxkaXYgY2xhc3M9ImNtZCI+JHtl"
    "c2Mocy5jb21tYW5kKX0gJHtlc2MoKHMuYXJnc3x8W10pLmpvaW4oIiAiKSl9PC9kaXY+PC9kaXY+CiAgICAgIDxkaXYgY2xhc3M9InJvdyIgc3R5bGU9"
    "ImdhcDo2cHgiPjwvZGl2PjwvZGl2PgogICAgICAke3MuY29ubmVjdGVkJiZzLnRvb2xzLmxlbmd0aD9gPGRpdiBjbGFzcz0idG9vbGNoaXBzIj4kewog"
    "ICAgICAgIHMudG9vbHMubWFwKHRuPT5gPHNwYW4gY2xhc3M9ImNoaXAgdmlvIj4ke2VzYyh0bil9PC9zcGFuPmApLmpvaW4oIiIpfTwvZGl2PmA6IiJ9"
    "CiAgICAgICR7cy5lcnJvcj9gPGRpdiBjbGFzcz0iaGludCIgc3R5bGU9ImNvbG9yOnZhcigtLXJlZCk7bWFyZ2luLXRvcDo4cHgiPiR7ZXNjKHMuZXJy"
    "b3IpfTwvZGl2PmA6IiJ9YDsKICAgIGNvbnN0IGFjdGlvbnMgPSBzcnYucXVlcnlTZWxlY3RvcigiLnNoIC5yb3ciKTsKICAgIGNvbnN0IHJjID0gZWwo"
    "ImJ1dHRvbiIsImJ0biBzbSBnaG9zdCIsIHMuY29ubmVjdGVkPyJSZWNvbm5lY3QiOiJDb25uZWN0Iik7CiAgICByYy5vbmNsaWNrID0gYXN5bmMoKT0+"
    "eyByYy5kaXNhYmxlZD10cnVlOyByYy50ZXh0Q29udGVudD0i4oCmIjsKICAgICAgdHJ5eyBjb25zdCByID0gYXdhaXQgcG9zdCgiL2FwaS9tY3Avc2Vy"
    "dmVycy8iK3MuaWQrIi9jb25uZWN0Iik7CiAgICAgICAgaWYoci5vaykgdG9hc3QoIkNvbm5lY3RlZCAiK3MubmFtZSwib2siKTsgZWxzZSB0b2FzdChy"
    "LmVycm9yfHwiRmFpbGVkIiwiZXJyIik7IH0KICAgICAgY2F0Y2goZSl7IHRvYXN0KGUubWVzc2FnZSwiZXJyIik7IH0gbG9hZFRvb2xzKCk7IH07CiAg"
    "ICBjb25zdCBkbCA9IGVsKCJidXR0b24iLCJidG4gc20gZGFuZ2VyIiwiUmVtb3ZlIik7CiAgICBkbC5vbmNsaWNrID0gYXN5bmMoKT0+eyBhd2FpdCBk"
    "ZWwoIi9hcGkvbWNwL3NlcnZlcnMvIitzLmlkKTsgdG9hc3QoIlJlbW92ZWQiLCJvayIpOyBsb2FkVG9vbHMoKTsgfTsKICAgIGFjdGlvbnMuYXBwZW5k"
    "KHJjLCBkbCk7IGxpc3QuYXBwZW5kQ2hpbGQoc3J2KTsKICB9KTsKfQpmdW5jdGlvbiBzaG93QWRkTWNwKCl7CiAgY29uc3QgYm9keSA9IG1vZGFsKHt0"
    "aXRsZToiQWRkIE1DUCBzZXJ2ZXIiLCBib2R5SFRNTDpgCiAgICA8cCBjbGFzcz0ibXV0ZWQiIHN0eWxlPSJtYXJnaW46MCAwIDE2cHgiPk1DUCBzZXJ2"
    "ZXJzIHJ1biBhcyBhIGxvY2FsIGNvbW1hbmQuIEZvciBleGFtcGxlLCBhCiAgICAgIGZpbGVzeXN0ZW0gc2VydmVyOiBjb21tYW5kIDxzcGFuIGNsYXNz"
    "PSJtb25vIj5ucHg8L3NwYW4+LCBhcmd1bWVudHMKICAgICAgPHNwYW4gY2xhc3M9Im1vbm8iPi15IEBtb2RlbGNvbnRleHRwcm90b2NvbC9zZXJ2ZXIt"
    "ZmlsZXN5c3RlbSAvcGF0aDwvc3Bhbj4uPC9wPgogICAgPGRpdiBjbGFzcz0iZmllbGQiPjxsYWJlbD5OYW1lPC9sYWJlbD48aW5wdXQgY2xhc3M9Imlu"
    "cCIgaWQ9Im1jcE5hbWUiIHBsYWNlaG9sZGVyPSJGaWxlc3lzdGVtIj48L2Rpdj4KICAgIDxkaXYgY2xhc3M9ImZpZWxkIj48bGFiZWw+Q29tbWFuZDwv"
    "bGFiZWw+PGlucHV0IGNsYXNzPSJpbnAgbW9ubyIgaWQ9Im1jcENtZCIgcGxhY2Vob2xkZXI9Im5weCI+PC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJmaWVs"
    "ZCI+PGxhYmVsPkFyZ3VtZW50cyA8c3BhbiBjbGFzcz0iZGltIj4oc3BhY2Utc2VwYXJhdGVkKTwvc3Bhbj48L2xhYmVsPgogICAgICA8aW5wdXQgY2xh"
    "c3M9ImlucCBtb25vIiBpZD0ibWNwQXJncyIgcGxhY2Vob2xkZXI9Ii15IEBtb2RlbGNvbnRleHRwcm90b2NvbC9zZXJ2ZXItZmlsZXN5c3RlbSAvVXNl"
    "cnMvbWUvZG9jcyI+PC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJmaWVsZCI+PGxhYmVsPkVudmlyb25tZW50IDxzcGFuIGNsYXNzPSJkaW0iPihLRVk9dmFs"
    "dWUsIG9uZSBwZXIgbGluZSwgb3B0aW9uYWwpPC9zcGFuPjwvbGFiZWw+CiAgICAgIDx0ZXh0YXJlYSBjbGFzcz0idGEiIGlkPSJtY3BFbnYiIHBsYWNl"
    "aG9sZGVyPSJBUElfS0VZPS4uLiI+PC90ZXh0YXJlYT48L2Rpdj5gLAogICAgYWN0aW9uczpbe2xhYmVsOiJDYW5jZWwiLCBvbkNsaWNrOmNsb3NlTW9k"
    "YWx9LAogICAgICB7bGFiZWw6IkFkZCAmIGNvbm5lY3QiLCBjbHM6InByaW1hcnkiLCBvbkNsaWNrOmFzeW5jKCk9PnsKICAgICAgICBjb25zdCBuYW1l"
    "PSQoIm1jcE5hbWUiKS52YWx1ZS50cmltKCksIGNtZD0kKCJtY3BDbWQiKS52YWx1ZS50cmltKCk7CiAgICAgICAgaWYoIW5hbWV8fCFjbWQpeyB0b2Fz"
    "dCgiTmFtZSBhbmQgY29tbWFuZCByZXF1aXJlZCIsImVyciIpOyByZXR1cm47IH0KICAgICAgICBjb25zdCBhcmdzPSQoIm1jcEFyZ3MiKS52YWx1ZS50"
    "cmltKCk7CiAgICAgICAgY29uc3QgZW52PXt9OyAkKCJtY3BFbnYiKS52YWx1ZS5zcGxpdCgiXG4iKS5mb3JFYWNoKGw9PnsKICAgICAgICAgIGNvbnN0"
    "IGk9bC5pbmRleE9mKCI9Iik7IGlmKGk+MCkgZW52W2wuc2xpY2UoMCxpKS50cmltKCldPWwuc2xpY2UoaSsxKS50cmltKCk7IH0pOwogICAgICAgIGNs"
    "b3NlTW9kYWwoKTsgdG9hc3QoIkNvbm5lY3RpbmcgdG8gIituYW1lKyLigKYiKTsKICAgICAgICB0cnl7IGNvbnN0IHIgPSBhd2FpdCBwb3N0KCIvYXBp"
    "L21jcC9zZXJ2ZXJzIiwge25hbWUsIGNvbW1hbmQ6Y21kLCBhcmdzLCBlbnZ9KTsKICAgICAgICAgIGlmKHIuY29ubmVjdCAmJiByLmNvbm5lY3Qub2sp"
    "IHRvYXN0KCJDb25uZWN0ZWQgIituYW1lLCJvayIpOwogICAgICAgICAgZWxzZSB0b2FzdCgoci5jb25uZWN0JiZyLmNvbm5lY3QuZXJyb3IpfHwiQWRk"
    "ZWQgYnV0IG5vdCBjb25uZWN0ZWQiLCJlcnIiKTsgfQogICAgICAgIGNhdGNoKGUpeyB0b2FzdChlLm1lc3NhZ2UsImVyciIpOyB9IGxvYWRUb29scygp"
    "OwogICAgICB9fV19KTsKfQphc3luYyBmdW5jdGlvbiBwb2xsTWNwSW5zdGFsbCgpewogIGNvbnN0IGxvZyA9ICQoIm1jcExvZyIpOwogIGNvbnN0IHQg"
    "PSBzZXRJbnRlcnZhbChhc3luYygpPT57CiAgICB0cnl7IGNvbnN0IHN0ID0gYXdhaXQgYXBpKCIvYXBpL21jcC9pbnN0YWxsX3N0YXR1cyIpOwogICAg"
    "ICBpZihsb2cgJiYgc3Quam9iKXsgbG9nLnRleHRDb250ZW50PShzdC5qb2IubG9nfHxbXSkuam9pbigiXG4iKTsgbG9nLnNjcm9sbFRvcD1sb2cuc2Ny"
    "b2xsSGVpZ2h0OyB9CiAgICAgIGlmKHN0Lmluc3RhbGxlZCl7IGNsZWFySW50ZXJ2YWwodCk7IHRvYXN0KCJNQ1Agc3VwcG9ydCByZWFkeSIsIm9rIik7"
    "IGxvYWRUb29scygpOyB9CiAgICAgIGVsc2UgaWYoc3Quam9iICYmIHN0LmpvYi5zdGF0dXM9PT0iZmFpbGVkIil7IGNsZWFySW50ZXJ2YWwodCk7CiAg"
    "ICAgICAgdG9hc3QoIkluc3RhbGwgZmFpbGVkIiwiZXJyIik7CiAgICAgICAgY29uc3QgYj0kKCJzZXR1cE1jcEJ0biIpOyBpZihiKXtiLmRpc2FibGVk"
    "PWZhbHNlO2IudGV4dENvbnRlbnQ9IlJldHJ5Ijt9IH0KICAgIH1jYXRjaChlKXsgY2xlYXJJbnRlcnZhbCh0KTsgfQogIH0sIDE1MDApOwp9CmFzeW5j"
    "IGZ1bmN0aW9uIHJlbmRlclNlYXJjaCgpewogIGNvbnN0IGhvc3QgPSAkKCJzZWFyY2hCb2R5Iik7IGlmKCFob3N0KSByZXR1cm47CiAgaG9zdC5pbm5l"
    "ckhUTUwgPSAnPGRpdiBjbGFzcz0icm93Ij48ZGl2IGNsYXNzPSJzcGluIj48L2Rpdj48c3BhbiBjbGFzcz0ibXV0ZWQiPkNoZWNraW5nIHNlYXJjaCBl"
    "bmdpbmVz4oCmPC9zcGFuPjwvZGl2Pic7CiAgbGV0IHM7CiAgdHJ5eyBzID0gYXdhaXQgYXBpKCIvYXBpL3NlYXJjaC9zdGF0dXMiKTsgfQogIGNhdGNo"
    "KGUpeyBob3N0LmlubmVySFRNTCA9IGVyckNhcmQoZS5tZXNzYWdlKTsgcmV0dXJuOyB9CiAgY29uc3QgbGl2ZSA9IHMuc2VhcnhuZy5qc29uX29rOwog"
    "IGxldCBodG1sID0gYDxkaXYgY2xhc3M9InRvb2xyb3ciPjxkaXYgY2xhc3M9InRpYyI+8J+UjTwvZGl2PgogICAgPGRpdiBjbGFzcz0idGkiPjxkaXYg"
    "Y2xhc3M9Im5tIj5BY3RpdmUgZW5naW5lOiAke2xpdmU/IlNlYXJYTkciOiJEdWNrRHVja0dvIChidWlsdC1pbiBmYWxsYmFjaykifTwvZGl2PgogICAg"
    "ICA8ZGl2IGNsYXNzPSJkcyI+JHtsaXZlCiAgICAgICAgPyAiUHJpdmF0ZSBtZXRhc2VhcmNoIGF0IDxzcGFuIGNsYXNzPSdtb25vJz4iK2VzYyhzLnNl"
    "YXJ4bmdfdXJsKSsiPC9zcGFuPiDigJQgYWdncmVnYXRlcyBtYW55IGVuZ2luZXMsIG5vIHRyYWNraW5nLCBubyByYXRlIGxpbWl0cy4iCiAgICAgICAg"
    "OiAiV29ya3Mgb3V0IG9mIHRoZSBib3gsIGJ1dCBjYW4gYmUgc2xvdyBvciByYXRlLWxpbWl0ZWQuIFNlYXJYTkcgKGJlbG93KSBpcyB0aGUgcmVjb21t"
    "ZW5kZWQgdXBncmFkZS4ifTwvZGl2PjwvZGl2PgogICAgPHNwYW4gY2xhc3M9ImNoaXAgJHtsaXZlPyJncm4iOiIifSI+JHtsaXZlPyJjb25uZWN0ZWQi"
    "OiJmYWxsYmFjayJ9PC9zcGFuPgogICAgPGJ1dHRvbiBjbGFzcz0iYnRuIHNtIGdob3N0IiBpZD0idGVzdFNlYXJjaEJ0biI+VGVzdDwvYnV0dG9uPjwv"
    "ZGl2PmA7CiAgaWYoIWxpdmUpewogICAgaHRtbCArPSBgPGRpdiBjbGFzcz0iY2FyZCBwYWQtbGciIHN0eWxlPSJtYXJnaW46NHB4IDAgOHB4Ij4KICAg"
    "ICAgPGI+U2V0IHVwIFNlYXJYTkcgPHNwYW4gY2xhc3M9ImNoaXAgdmlvIiBzdHlsZT0ibWFyZ2luLWxlZnQ6NnB4Ij5yZWNvbW1lbmRlZDwvc3Bhbj48"
    "L2I+CiAgICAgIDxwIGNsYXNzPSJtdXRlZCIgc3R5bGU9Im1hcmdpbjo3cHggMCAwO21heC13aWR0aDo2NGNoIj5TZWFyWE5HIGlzIGEgc2VsZi1ob3N0"
    "ZWQsIHByaXZhdGUKICAgICAgICBtZXRhLXNlYXJjaCBlbmdpbmUuIEhlb3J0aCBjYW4gc3RhcnQgb25lIGZvciB5b3UgYXMgYSBzbWFsbCBEb2NrZXIg"
    "Y29udGFpbmVyCiAgICAgICAgKG9uZS10aW1lIH4zMDAgTUIgaW1hZ2UgZG93bmxvYWQpIGFuZCB3aWxsIHVzZSBpdCBhdXRvbWF0aWNhbGx5IGZvciB0"
    "aGUgYWdlbnQncwogICAgICAgIDxzcGFuIGNsYXNzPSJtb25vIj53ZWJfc2VhcmNoPC9zcGFuPiB0b29sLjwvcD4KICAgICAgPGRpdiBpZD0ic2VhcnhT"
    "dGVwcyIgc3R5bGU9Im1hcmdpbi10b3A6MTRweCI+PC9kaXY+CiAgICAgIDxkaXYgY2xhc3M9Imluc3RhbGxsb2ciIGlkPSJzZWFyeExvZyIgaGlkZGVu"
    "PjwvZGl2PjwvZGl2PmA7CiAgfQogIGhvc3QuaW5uZXJIVE1MID0gaHRtbDsKICAkKCJ0ZXN0U2VhcmNoQnRuIikub25jbGljayA9IHRlc3RTZWFyY2g7"
    "CiAgaWYobGl2ZSkgcmV0dXJuOwoKICBjb25zdCBzdGVwcyA9ICQoInNlYXJ4U3RlcHMiKTsKICBpZighcy5kb2NrZXIuaW5zdGFsbGVkKXsKICAgIHN0"
    "ZXBzLmlubmVySFRNTCA9IGA8ZGl2IGNsYXNzPSJyb3ciIHN0eWxlPSJnYXA6OXB4O21hcmdpbi1ib3R0b206MTBweCI+CiAgICAgIDxzcGFuIGNsYXNz"
    "PSJkb3Qgd2FybiI+PC9zcGFuPjxiPkRvY2tlciBpcyByZXF1aXJlZCBmaXJzdDwvYj48L2Rpdj4KICAgICAgPHAgY2xhc3M9Im11dGVkIiBzdHlsZT0i"
    "bWFyZ2luOjAgMCAxMHB4Ij5TZWFyWE5HIHJ1bnMgaW4gYSBjb250YWluZXIsIHNvIGluc3RhbGwgRG9ja2VyCiAgICAgICAgb25jZSDigJQgaXQgdGFr"
    "ZXMgYSBjb3VwbGUgb2YgbWludXRlcywgdGhlbiBjb21lIGJhY2sgaGVyZTo8L3A+CiAgICAgICR7cy5kb2NrZXJfaGVscC5tYXAobD0+YDxkaXYgY2xh"
    "c3M9Im1vbm8iIHN0eWxlPSJmb250LXNpemU6MTJweDtiYWNrZ3JvdW5kOnZhcigtLWluayk7Ym9yZGVyOjFweCBzb2xpZCB2YXIoLS1saW5lKTtib3Jk"
    "ZXItcmFkaXVzOjhweDtwYWRkaW5nOjlweCAxMXB4O21hcmdpbi1ib3R0b206NnB4Ij4ke2VzYyhsKX08L2Rpdj5gKS5qb2luKCIiKX0KICAgICAgPGJ1"
    "dHRvbiBjbGFzcz0iYnRuIHNtIGdob3N0IiBzdHlsZT0ibWFyZ2luLXRvcDo0cHgiIGlkPSJzZWFyeFJlY2hlY2siPkkndmUgaW5zdGFsbGVkIERvY2tl"
    "ciDigJQgY2hlY2sgYWdhaW48L2J1dHRvbj5gOwogICAgJCgic2VhcnhSZWNoZWNrIikub25jbGljayA9IHJlbmRlclNlYXJjaDsKICAgIHJldHVybjsK"
    "ICB9CiAgaWYoIXMuZG9ja2VyLmRhZW1vbil7CiAgICBzdGVwcy5pbm5lckhUTUwgPSBgPGRpdiBjbGFzcz0icm93IiBzdHlsZT0iZ2FwOjlweDttYXJn"
    "aW4tYm90dG9tOjEwcHgiPgogICAgICA8c3BhbiBjbGFzcz0iZG90IHdhcm4iPjwvc3Bhbj48Yj5Eb2NrZXIgaXMgaW5zdGFsbGVkIGJ1dCBub3QgcnVu"
    "bmluZzwvYj48L2Rpdj4KICAgICAgPHAgY2xhc3M9Im11dGVkIiBzdHlsZT0ibWFyZ2luOjAgMCAxMnB4Ij5TdGFydCBEb2NrZXIgRGVza3RvcCAob24g"
    "TGludXg6CiAgICAgICAgPHNwYW4gY2xhc3M9Im1vbm8iPnN1ZG8gc3lzdGVtY3RsIHN0YXJ0IGRvY2tlcjwvc3Bhbj4pLCB0aGVuIGNoZWNrIGFnYWlu"
    "LjwvcD4KICAgICAgPGJ1dHRvbiBjbGFzcz0iYnRuIHNtIGdob3N0IiBpZD0ic2VhcnhSZWNoZWNrIj5DaGVjayBhZ2FpbjwvYnV0dG9uPmA7CiAgICAk"
    "KCJzZWFyeFJlY2hlY2siKS5vbmNsaWNrID0gcmVuZGVyU2VhcmNoOwogICAgcmV0dXJuOwogIH0KICBjb25zdCBydW5uaW5nID0gcy5qb2IgJiYgcy5q"
    "b2Iuc3RhdHVzPT09InJ1bm5pbmciOwogIHN0ZXBzLmlubmVySFRNTCA9IGAKICAgICR7cy5jb250YWluZXIuZXhpc3RzP2A8cCBjbGFzcz0iaGludCIg"
    "c3R5bGU9Im1hcmdpbjowIDAgMTBweCI+Q29udGFpbmVyCiAgICAgIDxzcGFuIGNsYXNzPSJtb25vIj4ke2VzYyhzLmNvbnRhaW5lci5uYW1lfHwiaGVv"
    "cnRoLXNlYXJ4bmciKX08L3NwYW4+IGV4aXN0cyDigJQgc3RhdHVzOiAke2VzYyhzLmNvbnRhaW5lci5zdGF0dXN8fCJzdG9wcGVkIil9LjwvcD5gOiIi"
    "fQogICAgJHtzLnNlYXJ4bmcucmVhY2hhYmxlICYmIHMuc2VhcnhuZy5lcnJvcj9gPHAgY2xhc3M9ImhpbnQiIHN0eWxlPSJjb2xvcjp2YXIoLS1zaWdu"
    "YWwpO21hcmdpbjowIDAgMTBweCI+JHtlc2Mocy5zZWFyeG5nLmVycm9yKX08L3A+YDoiIn0KICAgIDxkaXYgY2xhc3M9InJvdyIgc3R5bGU9ImdhcDox"
    "MHB4O2ZsZXgtd3JhcDp3cmFwIj4KICAgICAgPGJ1dHRvbiBjbGFzcz0iYnRuIHByaW1hcnkiIGlkPSJzdGFydFNlYXJ4QnRuIiAke3J1bm5pbmc/ImRp"
    "c2FibGVkIjoiIn0+CiAgICAgICAgJHtydW5uaW5nPyJTdGFydGluZ+KApiI6KHMuY29udGFpbmVyLmV4aXN0cz8iU3RhcnQgU2VhclhORyI6Ikluc3Rh"
    "bGwgJiBzdGFydCBTZWFyWE5HIil9PC9idXR0b24+CiAgICAgIDxidXR0b24gY2xhc3M9ImJ0biBzbSBnaG9zdCIgaWQ9InNlYXJ4UmVjaGVjayI+Q2hl"
    "Y2sgYWdhaW48L2J1dHRvbj48L2Rpdj4KICAgIDxwIGNsYXNzPSJoaW50IiBzdHlsZT0ibWFyZ2luOjEycHggMCAwIj5QcmVmZXIgdG8gcnVuIGl0IHlv"
    "dXJzZWxmPyBVc2U6PGJyPgogICAgICA8c3BhbiBjbGFzcz0ibW9ubyIgc3R5bGU9InVzZXItc2VsZWN0OmFsbCI+JHtlc2Mocy5tYW51YWxfY21kKX08"
    "L3NwYW4+PC9wPmA7CiAgJCgic2VhcnhSZWNoZWNrIikub25jbGljayA9IHJlbmRlclNlYXJjaDsKICAkKCJzdGFydFNlYXJ4QnRuIikub25jbGljayA9"
    "IGFzeW5jKCk9PnsKICAgICQoInN0YXJ0U2VhcnhCdG4iKS5kaXNhYmxlZCA9IHRydWU7ICQoInN0YXJ0U2VhcnhCdG4iKS50ZXh0Q29udGVudD0iU3Rh"
    "cnRpbmfigKYiOwogICAgJCgic2VhcnhMb2ciKS5oaWRkZW4gPSBmYWxzZTsKICAgIHRyeXsgYXdhaXQgcG9zdCgiL2FwaS9zZWFyY2gvc2V0dXAiKTsg"
    "cG9sbFNlYXJ4KCk7IH0KICAgIGNhdGNoKGUpeyB0b2FzdChlLm1lc3NhZ2UsImVyciIpOyByZW5kZXJTZWFyY2goKTsgfQogIH07CiAgaWYocnVubmlu"
    "Zyl7ICQoInNlYXJ4TG9nIikuaGlkZGVuPWZhbHNlOwogICAgJCgic2VhcnhMb2ciKS50ZXh0Q29udGVudD0ocy5qb2IubG9nfHxbXSkuam9pbigiXG4i"
    "KTsgcG9sbFNlYXJ4KCk7IH0KfQphc3luYyBmdW5jdGlvbiB0ZXN0U2VhcmNoKCl7CiAgdG9hc3QoIlJ1bm5pbmcgYSB0ZXN0IHNlYXJjaOKApiIpOwog"
    "IHRyeXsKICAgIGNvbnN0IHIgPSBhd2FpdCBhcGkoIi9hcGkvc2VhcmNoL3Rlc3Q/cT0iK2VuY29kZVVSSUNvbXBvbmVudCgid2hhdCBpcyBzZWFyeG5n"
    "IikpOwogICAgbW9kYWwoe3RpdGxlOiJXZWIgc2VhcmNoIHRlc3QiLCBib2R5SFRNTDoKICAgICAgYDxwcmUgc3R5bGU9IndoaXRlLXNwYWNlOnByZS13"
    "cmFwO2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pO2ZvbnQtc2l6ZToxMnB4O21heC1oZWlnaHQ6MzYwcHg7b3ZlcmZsb3c6YXV0bzttYXJnaW46MCI+JHtl"
    "c2Moci5yZXN1bHQpfTwvcHJlPmAsCiAgICAgIGFjdGlvbnM6W3tsYWJlbDoiQ2xvc2UiLCBvbkNsaWNrOmNsb3NlTW9kYWx9XX0pOwogIH1jYXRjaChl"
    "KXsgdG9hc3QoZS5tZXNzYWdlLCJlcnIiKTsgfQp9CmxldCBzZWFyeFRpbWVyPW51bGw7CmZ1bmN0aW9uIHBvbGxTZWFyeCgpewogIGlmKHNlYXJ4VGlt"
    "ZXIpIGNsZWFySW50ZXJ2YWwoc2VhcnhUaW1lcik7CiAgbGV0IHRyaWVzPTA7CiAgc2VhcnhUaW1lciA9IHNldEludGVydmFsKGFzeW5jKCk9PnsKICAg"
    "IHRyaWVzKys7CiAgICB0cnl7CiAgICAgIGNvbnN0IHMgPSBhd2FpdCBhcGkoIi9hcGkvc2VhcmNoL3N0YXR1cyIpOwogICAgICBjb25zdCBsb2cgPSAk"
    "KCJzZWFyeExvZyIpOwogICAgICBpZihsb2cgJiYgcy5qb2IpeyBsb2cudGV4dENvbnRlbnQ9KHMuam9iLmxvZ3x8W10pLmpvaW4oIlxuIik7CiAgICAg"
    "ICAgbG9nLnNjcm9sbFRvcD1sb2cuc2Nyb2xsSGVpZ2h0OyB9CiAgICAgIGlmKHMuc2VhcnhuZy5qc29uX29rKXsgY2xlYXJJbnRlcnZhbChzZWFyeFRp"
    "bWVyKTsKICAgICAgICB0b2FzdCgiU2VhclhORyBpcyBydW5uaW5nIOKAlCB3ZWIgc2VhcmNoIHVwZ3JhZGVkIiwib2siKTsgcmVuZGVyU2VhcmNoKCk7"
    "IH0KICAgICAgZWxzZSBpZihzLmpvYiAmJiBzLmpvYi5zdGF0dXM9PT0iZmFpbGVkIil7IGNsZWFySW50ZXJ2YWwoc2VhcnhUaW1lcik7CiAgICAgICAg"
    "dG9hc3QoIlNlYXJYTkcgc3RhcnQgZmFpbGVkIOKAlCBzZWUgdGhlIGxvZyIsImVyciIpOyB9CiAgICAgIGVsc2UgaWYodHJpZXM+NjApeyBjbGVhcklu"
    "dGVydmFsKHNlYXJ4VGltZXIpOwogICAgICAgIHRvYXN0KCJTZWFyWE5HIGlzIHRha2luZyBhIHdoaWxlIOKAlCBjaGVjayB0aGUgbG9nIG9yIHRyeSBh"
    "Z2FpbiIsImVyciIpOyB9CiAgICB9Y2F0Y2goZSl7IGNsZWFySW50ZXJ2YWwoc2VhcnhUaW1lcik7IH0KICB9LCAxNjAwKTsKfQpsb2FkZXJzLnRvb2xz"
    "ID0gbG9hZFRvb2xzOwoKLyogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09CiAgIFNFVFRJ"
    "TkdTICAoKyBzZWxmLXVwZGF0ZSkKICAgPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09ICov"
    "CmZ1bmN0aW9uIHNldHRpbmdSb3cobGFiLCBzdWIsIGN0bEhUTUwpewogIHJldHVybiBgPGRpdiBjbGFzcz0ic2V0dGluZ3JvdyI+PGRpdiBjbGFzcz0i"
    "bGFiIj4ke2xhYn0KICAgICR7c3ViP2A8ZGl2IGNsYXNzPSJzdWIiPiR7c3VifTwvZGl2PmA6IiJ9PC9kaXY+PGRpdiBjbGFzcz0iY3RsIj4ke2N0bEhU"
    "TUx9PC9kaXY+PC9kaXY+YDsKfQphc3luYyBmdW5jdGlvbiBsb2FkU2V0dGluZ3MoKXsKICBjb25zdCBob3N0ID0gJCgic2V0dGluZ3NCb2R5Iik7CiAg"
    "bGV0IHM7CiAgdHJ5eyBjb25zdCByID0gYXdhaXQgYXBpKCIvYXBpL3NldHRpbmdzIik7IHMgPSByLnNldHRpbmdzOyBzdGF0ZS5zZXR0aW5ncyA9IHM7"
    "IH0KICBjYXRjaChlKXsgaG9zdC5pbm5lckhUTUwgPSBlcnJDYXJkKGUubWVzc2FnZSk7IHJldHVybjsgfQogIGNvbnN0IHN3ID0gKGtleSwgb24pPT5g"
    "PGRpdiBjbGFzcz0ic3dpdGNoICR7b24/Im9uIjoiIn0iIGRhdGEta2V5PSIke2tleX0iPjwvZGl2PmA7CiAgaG9zdC5pbm5lckhUTUwgPSBgCiAgICA8"
    "ZGl2IGNsYXNzPSJjYXJkIHBhZC1sZyIgc3R5bGU9Im1hcmdpbi1ib3R0b206MTZweCI+CiAgICAgIDxkaXYgY2xhc3M9InNlY3Rpb24tdGl0bGUiIHN0"
    "eWxlPSJtYXJnaW4tdG9wOjAiPkJlaGF2aW91cjwvZGl2PgogICAgICAke3NldHRpbmdSb3coIlN5c3RlbSBwcm9tcHQiLCJTZXRzIHRoZSBhc3Npc3Rh"
    "bnQncyBwZXJzb25hbGl0eSBhbmQgcnVsZXMuIiwKICAgICAgICBgPHRleHRhcmVhIGNsYXNzPSJ0YSIgaWQ9InNldFN5c1Byb21wdCIgc3R5bGU9Im1p"
    "bi13aWR0aDoyODBweCI+JHtlc2Mocy5zeXN0ZW1fcHJvbXB0KX08L3RleHRhcmVhPmApfQogICAgICAke3NldHRpbmdSb3coIkNvbnRleHQgbWVzc2Fn"
    "ZXMiLCJIb3cgbWFueSByZWNlbnQgbWVzc2FnZXMgdG8gc2VuZCBlYWNoIHR1cm4uIiwKICAgICAgICBgPGlucHV0IGNsYXNzPSJpbnAgbW9ubyIgaWQ9"
    "InNldENvbnRleHQiIHZhbHVlPSIke2VzYyhzLmNvbnRleHRfbWVzc2FnZXMpfSIgc3R5bGU9IndpZHRoOjkwcHgiPmApfQogICAgICAke3NldHRpbmdS"
    "b3coIldlYiB0b29scyIsIkxldCB0aGUgYWdlbnQgc2VhcmNoIHRoZSB3ZWIgYW5kIGZldGNoIHBhZ2VzLiIsCiAgICAgICAgc3coImFsbG93X3dlYl90"
    "b29scyIsIHMuYWxsb3dfd2ViX3Rvb2xzPT09IjEiKSl9CiAgICAgICR7c2V0dGluZ1JvdygiQWxsb3cgY29kZSBleGVjdXRpb24iLAogICAgICAgICJM"
    "ZXRzIHRoZSBhZ2VudCBydW4gUHl0aG9uIGFuZCBzaGVsbCBjb21tYW5kcyBpbiBpdHMgd29ya3NwYWNlLiBPbmx5IGVuYWJsZSBpZiB5b3UgdHJ1c3Qg"
    "dGhlIG1vZGVsLiIsCiAgICAgICAgc3coImFsbG93X2NvZGVfZXhlY3V0aW9uIiwgcy5hbGxvd19jb2RlX2V4ZWN1dGlvbj09PSIxIikpfQogICAgICAk"
    "e3NldHRpbmdSb3coIkFnZW50IHN0ZXAgbGltaXQiLCJNYXhpbXVtIHRvb2wgY2FsbHMgYmVmb3JlIHRoZSBhZ2VudCBtdXN0IGFuc3dlci4iLAogICAg"
    "ICAgIGA8aW5wdXQgY2xhc3M9ImlucCBtb25vIiBpZD0ic2V0U3RlcHMiIHZhbHVlPSIke2VzYyhzLmFnZW50X21heF9zdGVwcyl9IiBzdHlsZT0id2lk"
    "dGg6OTBweCI+YCl9CiAgICAgICR7c2V0dGluZ1JvdygiTG9vcCBpdGVyYXRpb24gbGltaXQiLCJDZWlsaW5nIGZvciBhdXRvbm9tb3VzIExvb3AgcnVu"
    "cyDigJQgdGhlIGxvb3Agc3RvcHMgaGVyZSBldmVuIGlmIGl0IGhhc24ndCBjYWxsZWQgdGhlIHRhc2sgZG9uZS4iLAogICAgICAgIGA8aW5wdXQgY2xh"
    "c3M9ImlucCBtb25vIiBpZD0ic2V0TG9vcFN0ZXBzIiB2YWx1ZT0iJHtlc2Mocy5sb29wX21heF9zdGVwcyl9IiBzdHlsZT0id2lkdGg6OTBweCI+YCl9"
    "CiAgICAgICR7c2V0dGluZ1JvdygiQ291bmNpbCBzaXplIiwiQ29uc3VsdGFudHMgcGVyIENvdW5jaWwgcnVuICgy4oCTMTApLiAz4oCTNSBpcyB0aGUg"
    "c3dlZXQgc3BvdCBvbiBsb2NhbCBoYXJkd2FyZTsgMTAgd29ya3MgYnV0IGlzIHNsb3cuIiwKICAgICAgICBgPGlucHV0IGNsYXNzPSJpbnAgbW9ubyIg"
    "aWQ9InNldENvdW5jaWxTaXplIiB2YWx1ZT0iJHtlc2Mocy5jb3VuY2lsX3NpemUpfSIgc3R5bGU9IndpZHRoOjkwcHgiPmApfQogICAgICAke3NldHRp"
    "bmdSb3coIkNvbnN1bHRhdGlvbiByb3VuZHMiLCJBZnRlciB0aGUgaW5kZXBlbmRlbnQgdGFrZXMsIGhvdyBtYW55IHJvdW5kcyB0aGUgY29uc3VsdGFu"
    "dHMgc3BlbmQgY3JpdGlxdWluZyBlYWNoIG90aGVyICgw4oCTMykuIiwKICAgICAgICBgPGlucHV0IGNsYXNzPSJpbnAgbW9ubyIgaWQ9InNldENvdW5j"
    "aWxSb3VuZHMiIHZhbHVlPSIke2VzYyhzLmNvdW5jaWxfcm91bmRzKX0iIHN0eWxlPSJ3aWR0aDo5MHB4Ij5gKX0KICAgICAgJHtzZXR0aW5nUm93KCJD"
    "b3VuY2lsIHJlc2VhcmNoIGJyaWVmIiwiUnVuIGEgcXVpY2sgc2hhcmVkIHdlYiBzZWFyY2ggYmVmb3JlIHRoZSBwYW5lbCBzdGFydHMsIHNvIGV2ZXJ5"
    "IGNvbnN1bHRhbnQgYXJndWVzIGZyb20gdGhlIHNhbWUgZmFjdHMuIiwKICAgICAgICBzdygiY291bmNpbF9yZXNlYXJjaCIsIHMuY291bmNpbF9yZXNl"
    "YXJjaD09PSIxIikpfQogICAgPC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJjYXJkIHBhZC1sZyIgc3R5bGU9Im1hcmdpbi1ib3R0b206MTZweCI+CiAgICAg"
    "IDxkaXYgY2xhc3M9InNlY3Rpb24tdGl0bGUiIHN0eWxlPSJtYXJnaW4tdG9wOjAiPk1vZGVscyAmYW1wOyByZXRyaWV2YWw8L2Rpdj4KICAgICAgJHtz"
    "ZXR0aW5nUm93KCJPbGxhbWEgYWRkcmVzcyIsIldoZXJlIE9sbGFtYSBpcyBsaXN0ZW5pbmcuIiwKICAgICAgICBgPGlucHV0IGNsYXNzPSJpbnAgbW9u"
    "byIgaWQ9InNldE9sbGFtYSIgdmFsdWU9IiR7ZXNjKHMub2xsYW1hX2hvc3QpfSIgc3R5bGU9Im1pbi13aWR0aDoyMDBweCI+YCl9CiAgICAgICR7c2V0"
    "dGluZ1JvdygiRW1iZWRkaW5nIG1vZGVsIiwiVXNlZCB0byBpbmRleCB5b3VyIGtub3dsZWRnZSBiYXNlLiIsCiAgICAgICAgYDxpbnB1dCBjbGFzcz0i"
    "aW5wIG1vbm8iIGlkPSJzZXRFbWJlZCIgdmFsdWU9IiR7ZXNjKHMuZW1iZWRfbW9kZWwpfSIgc3R5bGU9Im1pbi13aWR0aDoxODBweCI+YCl9CiAgICAg"
    "ICR7c2V0dGluZ1JvdygiS25vd2xlZGdlIHJlc3VsdHMiLCJIb3cgbWFueSBkb2N1bWVudCBjaHVua3MgdG8gcmV0cmlldmUgcGVyIHF1ZXN0aW9uLiIs"
    "CiAgICAgICAgYDxpbnB1dCBjbGFzcz0iaW5wIG1vbm8iIGlkPSJzZXRUb3BLIiB2YWx1ZT0iJHtlc2Mocy5yYWdfdG9wX2spfSIgc3R5bGU9IndpZHRo"
    "OjkwcHgiPmApfQogICAgPC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJjYXJkIHBhZC1sZyIgc3R5bGU9Im1hcmdpbi1ib3R0b206MTZweCI+CiAgICAgIDxk"
    "aXYgY2xhc3M9InNlY3Rpb24tdGl0bGUiIHN0eWxlPSJtYXJnaW4tdG9wOjAiPldlYiBzZWFyY2g8L2Rpdj4KICAgICAgJHtzZXR0aW5nUm93KCJTZWFy"
    "Y2ggYmFja2VuZCIsIkF1dG8gdXNlcyBTZWFyWE5HIHdoZW5ldmVyIGl0J3MgcnVubmluZyBhbmQgcXVpZXRseSBmYWxscyBiYWNrIHRvIER1Y2tEdWNr"
    "R28uIiwKICAgICAgICBgPHNlbGVjdCBjbGFzcz0ic2VsIiBpZD0ic2V0U2VhcmNoQmFja2VuZCI+CiAgICAgICAgICAgPG9wdGlvbiB2YWx1ZT0iYXV0"
    "byI+QXV0byAocmVjb21tZW5kZWQpPC9vcHRpb24+CiAgICAgICAgICAgPG9wdGlvbiB2YWx1ZT0ic2VhcnhuZyI+U2VhclhORyBvbmx5PC9vcHRpb24+"
    "CiAgICAgICAgICAgPG9wdGlvbiB2YWx1ZT0iZHVja2R1Y2tnbyI+RHVja0R1Y2tHbyBvbmx5PC9vcHRpb24+PC9zZWxlY3Q+YCl9CiAgICAgICR7c2V0"
    "dGluZ1JvdygiU2VhclhORyBhZGRyZXNzIiwiV2hlcmUgeW91ciBTZWFyWE5HIGluc3RhbmNlIGxpc3RlbnMuIiwKICAgICAgICBgPGlucHV0IGNsYXNz"
    "PSJpbnAgbW9ubyIgaWQ9InNldFNlYXJ4VXJsIiB2YWx1ZT0iJHtlc2Mocy5zZWFyeG5nX3VybCl9IiBzdHlsZT0ibWluLXdpZHRoOjIwMHB4Ij5gKX0K"
    "ICAgICAgPHAgY2xhc3M9ImhpbnQiIHN0eWxlPSJtYXJnaW46NnB4IDAgMCI+U2V0IHVwIFNlYXJYTkcgd2l0aCBvbmUgY2xpY2sgb24gdGhlCiAgICAg"
    "ICAgQWdlbnQgJmFtcDsgVG9vbHMgcGFnZSDigJQgaXQgbmVlZHMgRG9ja2VyLjwvcD4KICAgICAgPGRpdiBjbGFzcz0ic2V0dGluZ3JvdyIgc3R5bGU9"
    "ImJvcmRlci10b3A6MXB4IHNvbGlkIHZhcigtLWxpbmUtc29mdCk7bWFyZ2luLXRvcDoxMnB4Ij4KICAgICAgICA8ZGl2IGNsYXNzPSJsYWIiPlNhdmUg"
    "Y2hhbmdlczwvZGl2PgogICAgICAgIDxkaXYgY2xhc3M9ImN0bCI+PGJ1dHRvbiBjbGFzcz0iYnRuIHByaW1hcnkiIGlkPSJzYXZlU2V0dGluZ3NCdG4i"
    "PlNhdmUgc2V0dGluZ3M8L2J1dHRvbj48L2Rpdj48L2Rpdj4KICAgIDwvZGl2PgogICAgPGRpdiBjbGFzcz0iY2FyZCBwYWQtbGciIHN0eWxlPSJtYXJn"
    "aW4tYm90dG9tOjE2cHgiIGlkPSJ1cGRhdGVDYXJkIj48L2Rpdj4KICAgIDxkaXYgY2xhc3M9ImNhcmQgcGFkLWxnIj4KICAgICAgPGRpdiBjbGFzcz0i"
    "c2VjdGlvbi10aXRsZSIgc3R5bGU9Im1hcmdpbi10b3A6MCI+WW91ciBkYXRhPC9kaXY+CiAgICAgIDxwIGNsYXNzPSJtdXRlZCIgc3R5bGU9Im1hcmdp"
    "bjowIDAgOHB4Ij5FdmVyeXRoaW5nIEhlb3J0aCBzdG9yZXMg4oCUIGNvbnZlcnNhdGlvbnMsIGRvY3VtZW50cywKICAgICAgICBpbWFnZXMsIG1vZGVs"
    "cyBsaXN0LCBiYWNrdXBzIOKAlCBsaXZlcyBpbiBvbmUgZm9sZGVyIG9uIHlvdXIgY29tcHV0ZXI6PC9wPgogICAgICA8ZGl2IGNsYXNzPSJtb25vIiBz"
    "dHlsZT0iZm9udC1zaXplOjEycHg7YmFja2dyb3VuZDp2YXIoLS1pbmspO2JvcmRlcjoxcHggc29saWQgdmFyKC0tbGluZSk7CiAgICAgICAgYm9yZGVy"
    "LXJhZGl1czo4cHg7cGFkZGluZzoxMXB4IDEzcHg7d29yZC1icmVhazpicmVhay1hbGwiIGlkPSJkYXRhRGlyIj7igKY8L2Rpdj4KICAgICAgPHAgY2xh"
    "c3M9ImhpbnQiIHN0eWxlPSJtYXJnaW4tdG9wOjEwcHgiPkRlbGV0ZSB0aGF0IGZvbGRlciB0byByZXNldCBIZW9ydGggY29tcGxldGVseS4KICAgICAg"
    "ICBNb2RlbCBmaWxlcyB0aGVtc2VsdmVzIGFyZSBtYW5hZ2VkIGJ5IE9sbGFtYS48L3A+CiAgICA8L2Rpdj5gOwoKICAkKCJzZXRTZWFyY2hCYWNrZW5k"
    "IikudmFsdWUgPSBzLnNlYXJjaF9iYWNrZW5kIHx8ICJhdXRvIjsKCiAgLy8gc3dpdGNoZXMKICBob3N0LnF1ZXJ5U2VsZWN0b3JBbGwoIi5zd2l0Y2gi"
    "KS5mb3JFYWNoKGVsMj0+ewogICAgZWwyLm9uY2xpY2sgPSBhc3luYygpPT57CiAgICAgIGNvbnN0IGtleSA9IGVsMi5kYXRhc2V0LmtleTsgY29uc3Qg"
    "b24gPSAhZWwyLmNsYXNzTGlzdC5jb250YWlucygib24iKTsKICAgICAgZWwyLmNsYXNzTGlzdC50b2dnbGUoIm9uIiwgb24pOwogICAgICBpZihrZXk9"
    "PT0iYWxsb3dfY29kZV9leGVjdXRpb24iICYmIG9uKXsKICAgICAgICAvLyBjb25maXJtIGRhbmdlcm91cyB0b2dnbGUKICAgICAgICBlbDIuY2xhc3NM"
    "aXN0LnJlbW92ZSgib24iKTsKICAgICAgICBtb2RhbCh7dGl0bGU6IkFsbG93IGNvZGUgZXhlY3V0aW9uPyIsCiAgICAgICAgICBib2R5SFRNTDpgPHAg"
    "Y2xhc3M9Im11dGVkIj5UaGlzIGxldHMgdGhlIG1vZGVsIHJ1biBQeXRob24gYW5kIHNoZWxsIGNvbW1hbmRzIG9uIHlvdXIKICAgICAgICAgICAgY29t"
    "cHV0ZXIgKGluc2lkZSBpdHMgd29ya3NwYWNlIGZvbGRlcikuIE9ubHkgZW5hYmxlIHRoaXMgaWYgeW91IHVuZGVyc3RhbmQgYW5kIHRydXN0CiAgICAg"
    "ICAgICAgIHdoYXQgeW91J3JlIHJ1bm5pbmcuPC9wPmAsCiAgICAgICAgICBhY3Rpb25zOlt7bGFiZWw6IkNhbmNlbCIsIG9uQ2xpY2s6Y2xvc2VNb2Rh"
    "bH0sCiAgICAgICAgICAgIHtsYWJlbDoiRW5hYmxlIiwgY2xzOiJkYW5nZXIiLCBvbkNsaWNrOmFzeW5jKCk9PnsgY2xvc2VNb2RhbCgpOyBlbDIuY2xh"
    "c3NMaXN0LmFkZCgib24iKTsKICAgICAgICAgICAgICBhd2FpdCBwb3N0KCIvYXBpL3NldHRpbmdzIiwge2FsbG93X2NvZGVfZXhlY3V0aW9uOiIxIn0p"
    "OyB0b2FzdCgiQ29kZSBleGVjdXRpb24gZW5hYmxlZCIsIm9rIik7IH19XX0pOwogICAgICAgIHJldHVybjsKICAgICAgfQogICAgICBhd2FpdCBwb3N0"
    "KCIvYXBpL3NldHRpbmdzIiwge1trZXldOiBvbj8iMSI6IjAifSk7CiAgICAgIHRvYXN0KCJTYXZlZCIsIm9rIik7CiAgICB9OwogIH0pOwogICQoInNh"
    "dmVTZXR0aW5nc0J0biIpLm9uY2xpY2sgPSBhc3luYygpPT57CiAgICB0cnl7CiAgICAgIGF3YWl0IHBvc3QoIi9hcGkvc2V0dGluZ3MiLCB7CiAgICAg"
    "ICAgc3lzdGVtX3Byb21wdDokKCJzZXRTeXNQcm9tcHQiKS52YWx1ZSwKICAgICAgICBjb250ZXh0X21lc3NhZ2VzOiQoInNldENvbnRleHQiKS52YWx1"
    "ZSwKICAgICAgICBhZ2VudF9tYXhfc3RlcHM6JCgic2V0U3RlcHMiKS52YWx1ZSwKICAgICAgICBsb29wX21heF9zdGVwczokKCJzZXRMb29wU3RlcHMi"
    "KS52YWx1ZSwKICAgICAgICBjb3VuY2lsX3NpemU6JCgic2V0Q291bmNpbFNpemUiKS52YWx1ZSwKICAgICAgICBjb3VuY2lsX3JvdW5kczokKCJzZXRD"
    "b3VuY2lsUm91bmRzIikudmFsdWUsCiAgICAgICAgb2xsYW1hX2hvc3Q6JCgic2V0T2xsYW1hIikudmFsdWUsCiAgICAgICAgZW1iZWRfbW9kZWw6JCgi"
    "c2V0RW1iZWQiKS52YWx1ZSwKICAgICAgICByYWdfdG9wX2s6JCgic2V0VG9wSyIpLnZhbHVlLAogICAgICAgIHNlYXJjaF9iYWNrZW5kOiQoInNldFNl"
    "YXJjaEJhY2tlbmQiKS52YWx1ZSwKICAgICAgICBzZWFyeG5nX3VybDokKCJzZXRTZWFyeFVybCIpLnZhbHVlIH0pOwogICAgICB0b2FzdCgiU2V0dGlu"
    "Z3Mgc2F2ZWQiLCJvayIpOwogICAgfWNhdGNoKGUpeyB0b2FzdChlLm1lc3NhZ2UsImVyciIpOyB9CiAgfTsKICB0cnl7IGNvbnN0IHN5c2QgPSBhd2Fp"
    "dCBhcGkoIi9hcGkvc3lzdGVtIik7ICQoImRhdGFEaXIiKS50ZXh0Q29udGVudCA9IHN5c2QuZGF0YV9kaXI7IH1jYXRjaChlKXt9CiAgcmVuZGVyVXBk"
    "YXRlQ2FyZCgpOwp9CgovKiAtLS0tLS0tLS0tIHNlbGYtdXBkYXRlIC0tLS0tLS0tLS0gKi8KYXN5bmMgZnVuY3Rpb24gcmVuZGVyVXBkYXRlQ2FyZCgp"
    "ewogIGNvbnN0IGNhcmQgPSAkKCJ1cGRhdGVDYXJkIik7IGlmKCFjYXJkKSByZXR1cm47CiAgY2FyZC5pbm5lckhUTUwgPSBgPGRpdiBjbGFzcz0ic2Vj"
    "dGlvbi10aXRsZSIgc3R5bGU9Im1hcmdpbi10b3A6MCI+VXBkYXRlczwvZGl2PgogICAgPGRpdiBjbGFzcz0icm93IiBzdHlsZT0iZ2FwOjlweCI+PGRp"
    "diBjbGFzcz0ic3BpbiI+PC9kaXY+CiAgICA8c3BhbiBjbGFzcz0ibXV0ZWQiPkNoZWNraW5nIGZvciBhIG5ld2VyIHZlcnNpb27igKY8L3NwYW4+PC9k"
    "aXY+YDsKICBsZXQgaW5mbzsKICB0cnl7IGluZm8gPSBhd2FpdCBhcGkoIi9hcGkvdXBkYXRlL2NoZWNrIik7IH0KICBjYXRjaChlKXsgY2FyZC5pbm5l"
    "ckhUTUwgPSAnPGRpdiBjbGFzcz0ic2VjdGlvbi10aXRsZSIgc3R5bGU9Im1hcmdpbi10b3A6MCI+VXBkYXRlczwvZGl2PicrCiAgICBlcnJDYXJkKGUu"
    "bWVzc2FnZSk7IHJldHVybjsgfQogIHN0YXRlLnVwZGF0ZUluZm8gPSBpbmZvOwogIGNvbnN0IHdhdGNoZWQgPSAoaW5mby53YXRjaGVkfHxbXSkubWFw"
    "KHc9PmA8c3BhbiBjbGFzcz0ibW9ubyI+JHtlc2Modyl9PC9zcGFuPmApLmpvaW4oIjxicj4iKTsKICBsZXQgaW5uZXIgPSBgPGRpdiBjbGFzcz0ic2Vj"
    "dGlvbi10aXRsZSIgc3R5bGU9Im1hcmdpbi10b3A6MCI+VXBkYXRlczwvZGl2PmA7CiAgaWYoaW5mby5hdmFpbGFibGUpewogICAgY29uc3QgdSA9IGlu"
    "Zm8udXBkYXRlOwogICAgaW5uZXIgKz0gYDxkaXYgY2xhc3M9InVwZGF0ZWJhbm5lciIgc3R5bGU9Im1hcmdpbi1ib3R0b206MTZweCI+CiAgICAgIDxk"
    "aXYgY2xhc3M9ImljIj4KICAgICAgICA8c3ZnIHdpZHRoPSIyMCIgaGVpZ2h0PSIyMCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJv"
    "a2U9ImN1cnJlbnRDb2xvciIgc3Ryb2tlLXdpZHRoPSIyIj48cGF0aCBkPSJNMjEgMTJhOSA5IDAgMSAxLTIuNi02LjRNMjEgM3Y1aC01Ii8+PC9zdmc+"
    "PC9kaXY+CiAgICAgIDxkaXYgY2xhc3M9InVpIj48ZGl2IGNsYXNzPSJ0Ij5WZXJzaW9uICR7ZXNjKHUudmVyc2lvbil9IGlzIHJlYWR5IHRvIGluc3Rh"
    "bGw8L2Rpdj4KICAgICAgICA8ZGl2IGNsYXNzPSJkIj5Gb3VuZCA8c3BhbiBjbGFzcz0ibW9ubyI+JHtlc2ModS5wYXRoLnNwbGl0KCIvIikucG9wKCkp"
    "fTwvc3Bhbj4KICAgICAgICAgICgke3Uuc2l6ZV9rYn0gS0IpLiBZb3UncmUgb24gdiR7ZXNjKGluZm8uY3VycmVudCl9LjwvZGl2PjwvZGl2PgogICAg"
    "ICA8YnV0dG9uIGNsYXNzPSJidG4gcHJpbWFyeSIgaWQ9ImFwcGx5VXBkYXRlQnRuIj5VcGRhdGUgJmFtcDsgcmVzdGFydDwvYnV0dG9uPjwvZGl2PmA7"
    "CiAgfSBlbHNlIHsKICAgIGlubmVyICs9IGA8ZGl2IGNsYXNzPSJyb3ciIHN0eWxlPSJnYXA6OXB4O21hcmdpbi1ib3R0b206MTRweCI+CiAgICAgIDxz"
    "cGFuIGNsYXNzPSJkb3Qgb24iPjwvc3Bhbj48c3BhbiBjbGFzcz0ibXV0ZWQiPllvdSdyZSBvbiB0aGUgbGF0ZXN0IHZlcnNpb24KICAgICAgICAoPHNw"
    "YW4gY2xhc3M9Im1vbm8iPnYke2VzYyhpbmZvLmN1cnJlbnQpfTwvc3Bhbj4pLiBObyBuZXdlciBmaWxlIGZvdW5kLjwvc3Bhbj48L2Rpdj5gOwogIH0K"
    "ICBpbm5lciArPSBgPHAgY2xhc3M9ImhpbnQiIHN0eWxlPSJtYXJnaW46MCAwIDEycHgiPkhlb3J0aCB3YXRjaGVzIHRoZXNlIGxvY2F0aW9ucyBmb3Ig"
    "YSBuZXdlcgogICAgPHNwYW4gY2xhc3M9Im1vbm8iPmhlb3J0aCoucHk8L3NwYW4+IChvciA8c3BhbiBjbGFzcz0ibW9ubyI+bG9jYWxtaW5kKi5weTwv"
    "c3Bhbj4pIGZpbGU6PGJyPiR7d2F0Y2hlZH08YnI+CiAgICBEcm9wIGEgbmV3ZXIgZmlsZSBpbiBhbnkgb2YgdGhlbSwgb3IgdXBsb2FkIG9uZSBiZWxv"
    "dy48L3A+CiAgICA8ZGl2IGNsYXNzPSJyb3ciIHN0eWxlPSJnYXA6MTBweCI+CiAgICAgIDxidXR0b24gY2xhc3M9ImJ0biBzbSBnaG9zdCIgaWQ9ImNo"
    "ZWNrVXBkYXRlQnRuIj5DaGVjayBhZ2FpbjwvYnV0dG9uPgogICAgICA8YnV0dG9uIGNsYXNzPSJidG4gc20gZ2hvc3QiIGlkPSJ1cGxvYWRVcGRhdGVC"
    "dG4iPlVwbG9hZCB1cGRhdGUgZmlsZeKApjwvYnV0dG9uPgogICAgICA8aW5wdXQgdHlwZT0iZmlsZSIgaWQ9InVwZGF0ZUZpbGVJbnB1dCIgYWNjZXB0"
    "PSIucHkiIGhpZGRlbj4KICAgIDwvZGl2PmA7CiAgY2FyZC5pbm5lckhUTUwgPSBpbm5lcjsKCiAgY29uc3QgYXBwbHkgPSAkKCJhcHBseVVwZGF0ZUJ0"
    "biIpOwogIGlmKGFwcGx5KSBhcHBseS5vbmNsaWNrID0gKCk9PmFwcGx5VXBkYXRlKHN0YXRlLnVwZGF0ZUluZm8udXBkYXRlLnBhdGgpOwogICQoImNo"
    "ZWNrVXBkYXRlQnRuIikub25jbGljayA9IHJlbmRlclVwZGF0ZUNhcmQ7CiAgJCgidXBsb2FkVXBkYXRlQnRuIikub25jbGljayA9ICgpPT4kKCJ1cGRh"
    "dGVGaWxlSW5wdXQiKS5jbGljaygpOwogICQoInVwZGF0ZUZpbGVJbnB1dCIpLm9uY2hhbmdlID0gYXN5bmMoZSk9PnsKICAgIGNvbnN0IGYgPSBlLnRh"
    "cmdldC5maWxlc1swXTsgaWYoIWYpIHJldHVybjsKICAgIGNvbnN0IGZkID0gbmV3IEZvcm1EYXRhKCk7IGZkLmFwcGVuZCgiZmlsZSIsIGYpOwogICAg"
    "dG9hc3QoIlVwbG9hZGluZyAiK2YubmFtZSsi4oCmIik7CiAgICB0cnl7CiAgICAgIGNvbnN0IHIgPSBhd2FpdCBmZXRjaCgiL2FwaS91cGRhdGUvdXBs"
    "b2FkIiwge21ldGhvZDoiUE9TVCIsIGJvZHk6ZmR9KTsKICAgICAgY29uc3QgaiA9IGF3YWl0IHIuanNvbigpOwogICAgICBpZihqLm9rKXsgdG9hc3Qo"
    "IlVwZGF0ZSB2IitqLnZlcnNpb24rIiByZWFkeSIsIm9rIik7IHJlbmRlclVwZGF0ZUNhcmQoKTsgcG9sbFVwZGF0ZUJhZGdlKCk7IH0KICAgICAgZWxz"
    "ZSB0b2FzdChqLmVycm9yfHwiVXBsb2FkIHJlamVjdGVkIiwiZXJyIik7CiAgICB9Y2F0Y2goZXJyKXsgdG9hc3QoZXJyLm1lc3NhZ2UsImVyciIpOyB9"
    "CiAgICBlLnRhcmdldC52YWx1ZT0iIjsKICB9Owp9CmFzeW5jIGZ1bmN0aW9uIGFwcGx5VXBkYXRlKHBhdGgpewogIGNvbnN0IGJvZHkgPSBtb2RhbCh7"
    "dGl0bGU6IlVwZGF0ZSBhbmQgcmVzdGFydD8iLAogICAgYm9keUhUTUw6YDxwIGNsYXNzPSJtdXRlZCI+SGVvcnRoIHdpbGwgYmFjayB1cCB0aGUgY3Vy"
    "cmVudCB2ZXJzaW9uLCByZXBsYWNlIGl0IHdpdGggdGhlCiAgICAgIG5ldyBmaWxlLCBhbmQgcmVzdGFydCB0aGUgc2VydmVyLiBUaGlzIHBhZ2Ugd2ls"
    "bCByZWNvbm5lY3QgYXV0b21hdGljYWxseSBpbiBhIGZldyBzZWNvbmRzLjwvcD5gLAogICAgYWN0aW9uczpbe2xhYmVsOiJDYW5jZWwiLCBvbkNsaWNr"
    "OmNsb3NlTW9kYWx9LAogICAgICB7bGFiZWw6IlVwZGF0ZSBub3ciLCBjbHM6InByaW1hcnkiLCBvbkNsaWNrOmFzeW5jKCk9PnsKICAgICAgICBib2R5"
    "LmlubmVySFRNTCA9ICc8ZGl2IGNsYXNzPSJyb3ciIHN0eWxlPSJnYXA6MTBweCI+PGRpdiBjbGFzcz0ic3BpbiI+PC9kaXY+JysKICAgICAgICAgICc8"
    "c3Bhbj5BcHBseWluZyB1cGRhdGUgYW5kIHJlc3RhcnRpbmfigKY8L3NwYW4+PC9kaXY+JzsKICAgICAgICB0cnl7CiAgICAgICAgICBjb25zdCByID0g"
    "YXdhaXQgcG9zdCgiL2FwaS91cGRhdGUvYXBwbHkiLCB7cGF0aH0pOwogICAgICAgICAgaWYoci5vayl7IHdhaXRGb3JSZXN0YXJ0KHIubmV3X3ZlcnNp"
    "b24pOyB9CiAgICAgICAgICBlbHNlIHsgdG9hc3Qoci5lcnJvcnx8IlVwZGF0ZSBmYWlsZWQiLCJlcnIiKTsgY2xvc2VNb2RhbCgpOyB9CiAgICAgICAg"
    "fWNhdGNoKGUpewogICAgICAgICAgLy8gdGhlIHNlcnZlciBtYXkgaGF2ZSByZXN0YXJ0ZWQgbWlkLXJlcXVlc3Qg4oCUIHRyZWF0IGFzIHN1Y2Nlc3Mg"
    "YW5kIHdhaXQKICAgICAgICAgIHdhaXRGb3JSZXN0YXJ0KCk7CiAgICAgICAgfQogICAgICB9fV19KTsKfQpmdW5jdGlvbiB3YWl0Rm9yUmVzdGFydChu"
    "ZXdWZXIpewogIG1vZGFsKHt0aXRsZToiVXBkYXRpbmfigKYiLCBib2R5SFRNTDpgPGRpdiBzdHlsZT0idGV4dC1hbGlnbjpjZW50ZXI7cGFkZGluZzox"
    "NHB4Ij4KICAgIDxkaXYgY2xhc3M9InNwaW4iIHN0eWxlPSJ3aWR0aDoyNnB4O2hlaWdodDoyNnB4O21hcmdpbjowIGF1dG8gMTRweCI+PC9kaXY+CiAg"
    "ICA8cCBjbGFzcz0ibXV0ZWQiPlJlc3RhcnRpbmcgaW50byAke25ld1Zlcj8idmVyc2lvbiAiK2VzYyhuZXdWZXIpOiJ0aGUgbmV3IHZlcnNpb24ifS4K"
    "ICAgICAgUmVjb25uZWN0aW5nIGF1dG9tYXRpY2FsbHnigKY8L3A+PC9kaXY+YH0pOwogIGxldCB0cmllcz0wOwogIGNvbnN0IHQgPSBzZXRJbnRlcnZh"
    "bChhc3luYygpPT57CiAgICB0cmllcysrOwogICAgdHJ5ewogICAgICBjb25zdCByID0gYXdhaXQgZmV0Y2goIi9hcGkvaGVhbHRoIiwge2NhY2hlOiJu"
    "by1zdG9yZSJ9KTsKICAgICAgaWYoci5vayl7IGNvbnN0IGogPSBhd2FpdCByLmpzb24oKTsKICAgICAgICBjbGVhckludGVydmFsKHQpOwogICAgICAg"
    "IGNsb3NlTW9kYWwoKTsgdG9hc3QoIlVwZGF0ZWQgdG8gdiIrai52ZXJzaW9uKyIg4oCUIHJlbG9hZGluZyIsIm9rIik7CiAgICAgICAgc2V0VGltZW91"
    "dCgoKT0+bG9jYXRpb24ucmVsb2FkKCksIDgwMCk7CiAgICAgIH0KICAgIH1jYXRjaChlKXsgLyogc2VydmVyIHN0aWxsIGRvd24sIGtlZXAgd2FpdGlu"
    "ZyAqLyB9CiAgICBpZih0cmllcz40MCl7IGNsZWFySW50ZXJ2YWwodCk7IGNsb3NlTW9kYWwoKTsKICAgICAgdG9hc3QoIlJlc3RhcnQgaXMgdGFraW5n"
    "IGEgd2hpbGUg4oCUIHRyeSByZWxvYWRpbmcgdGhlIHBhZ2UiLCJlcnIiKTsgfQogIH0sIDgwMCk7Cn0KbG9hZGVycy5zZXR0aW5ncyA9IGxvYWRTZXR0"
    "aW5nczsKCi8qIC0tLS0tLS0tLS0gdXBkYXRlIGJhZGdlIHBvbGxpbmcgKGdsb2JhbCkgLS0tLS0tLS0tLSAqLwphc3luYyBmdW5jdGlvbiBwb2xsVXBk"
    "YXRlQmFkZ2UoKXsKICB0cnl7CiAgICBjb25zdCBpbmZvID0gYXdhaXQgYXBpKCIvYXBpL3VwZGF0ZS9jaGVjayIpOwogICAgc3RhdGUudXBkYXRlSW5m"
    "byA9IGluZm87CiAgICBjb25zdCBuYXZCdG4gPSBkb2N1bWVudC5xdWVyeVNlbGVjdG9yKCcubmF2IGJ1dHRvbltkYXRhLXZpZXc9InNldHRpbmdzIl0n"
    "KTsKICAgIGxldCBkb3QgPSBuYXZCdG4ucXVlcnlTZWxlY3RvcigiLnVwZG90Iik7CiAgICBpZihpbmZvLmF2YWlsYWJsZSl7CiAgICAgIGlmKCFkb3Qp"
    "eyBkb3QgPSBlbCgic3BhbiIsInVwZG90Iik7IG5hdkJ0bi5hcHBlbmRDaGlsZChkb3QpOyB9CiAgICAgIGNvbnN0IGJhbm5lciA9ICQoInVwZGF0ZUJh"
    "bm5lckhvc3QiKTsKICAgICAgaWYoYmFubmVyICYmICFiYW5uZXIuZGF0YXNldC5zaG93bil7CiAgICAgICAgYmFubmVyLmRhdGFzZXQuc2hvd249IjEi"
    "OwogICAgICAgIGJhbm5lci5pbm5lckhUTUwgPSBgPGRpdiBjbGFzcz0idXBkYXRlYmFubmVyIj4KICAgICAgICAgIDxkaXYgY2xhc3M9ImljIj48c3Zn"
    "IHdpZHRoPSIyMCIgaGVpZ2h0PSIyMCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9ImN1cnJlbnRDb2xvciIgc3Ryb2tlLXdp"
    "ZHRoPSIyIj48cGF0aCBkPSJNMTIgMTZWNG0wIDAgNCA0bS00LTQtNCA0Ii8+PHBhdGggZD0iTTQgMTZ2MmEyIDIgMCAwIDAgMiAyaDEyYTIgMiAwIDAg"
    "MCAyLTJ2LTIiLz48L3N2Zz48L2Rpdj4KICAgICAgICAgIDxkaXYgY2xhc3M9InVpIj48ZGl2IGNsYXNzPSJ0Ij5IZW9ydGggdiR7ZXNjKGluZm8udXBk"
    "YXRlLnZlcnNpb24pfSBpcyBhdmFpbGFibGU8L2Rpdj4KICAgICAgICAgICAgPGRpdiBjbGFzcz0iZCI+QSBuZXdlciB2ZXJzaW9uIGZpbGUgd2FzIGZv"
    "dW5kIG9uIHlvdXIgY29tcHV0ZXIuIFVwZGF0ZSBmcm9tIFNldHRpbmdzLjwvZGl2PjwvZGl2PgogICAgICAgICAgPGJ1dHRvbiBjbGFzcz0iYnRuIHBy"
    "aW1hcnkiIG9uY2xpY2s9InNob3coJ3NldHRpbmdzJykiPkdvIHRvIHVwZGF0ZTwvYnV0dG9uPjwvZGl2PmA7CiAgICAgIH0KICAgIH0gZWxzZSBpZihk"
    "b3QpeyBkb3QucmVtb3ZlKCk7IH0KICB9Y2F0Y2goZSl7fQp9CgovKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09"
    "PT09PT09PT09PT09PT0KICAgSU5JVAogICA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT0g"
    "Ki8KYXN5bmMgZnVuY3Rpb24gcmVmcmVzaE9sbGFtYVN0YXR1cygpewogIHRyeXsKICAgIGNvbnN0IHMgPSBhd2FpdCBhcGkoIi9hcGkvc3lzdGVtIik7"
    "IHN0YXRlLnN5c3RlbSA9IHM7CiAgICBjb25zdCB1cCA9IHMub2xsYW1hLnVwOwogICAgJCgib2xsYW1hRG90IikuY2xhc3NOYW1lID0gImRvdCAiKyh1"
    "cD8ib24iOiJvZmYiKTsKICAgICQoIm9sbGFtYVN0YXR1cyIpLnRleHRDb250ZW50ID0gdXAgPyAiT2xsYW1hICIrcy5vbGxhbWEudmVyc2lvbiA6ICJP"
    "bGxhbWEgb2ZmbGluZSI7CiAgICAkKCJvbGxhbWFTdGF0dXMiKS5jbGFzc05hbWUgPSB1cCA/ICIiIDogIm11dGVkIjsKICAgICQoImJyYW5kVmVyIiku"
    "dGV4dENvbnRlbnQgPSAidiIrcy52ZXJzaW9uOwogICAgYXBwbHlUaGVtZShzICYmIHN0YXRlLnNldHRpbmdzLnRoZW1lID8gc3RhdGUuc2V0dGluZ3Mu"
    "dGhlbWUgOgogICAgICAoZG9jdW1lbnQuZG9jdW1lbnRFbGVtZW50LmRhdGFzZXQudGhlbWV8fCJkYXJrIikpOwogIH1jYXRjaChlKXsKICAgICQoIm9s"
    "bGFtYURvdCIpLmNsYXNzTmFtZT0iZG90IG9mZiI7ICQoIm9sbGFtYVN0YXR1cyIpLnRleHRDb250ZW50PSJTZXJ2ZXIgZXJyb3IiOwogIH0KfQphc3lu"
    "YyBmdW5jdGlvbiBpbml0KCl7CiAgdHJ5eyBjb25zdCByID0gYXdhaXQgYXBpKCIvYXBpL3NldHRpbmdzIik7IHN0YXRlLnNldHRpbmdzID0gci5zZXR0"
    "aW5nczsKICAgIGFwcGx5VGhlbWUoci5zZXR0aW5ncy50aGVtZXx8ImRhcmsiKTsgfWNhdGNoKGUpe30KICBhd2FpdCByZWZyZXNoQ2hhdE1vZGVscygp"
    "OwogIGF3YWl0IHJlZnJlc2hPbGxhbWFTdGF0dXMoKTsKICBsb2FkRGFzaGJvYXJkKCk7CiAgcG9sbFVwZGF0ZUJhZGdlKCk7CiAgc2V0SW50ZXJ2YWwo"
    "cmVmcmVzaE9sbGFtYVN0YXR1cywgMTIwMDApOwogIHNldEludGVydmFsKHBvbGxVcGRhdGVCYWRnZSwgNjAwMDApOwp9CmluaXQoKTsKPC9zY3JpcHQ+"
    "CjwvYm9keT4KPC9odG1sPgo="
)
INDEX_HTML = base64.b64decode(_INDEX_B64).decode("utf-8")


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(INDEX_HTML)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

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

    port = _free_port(port, host)
    url = f"http://{'127.0.0.1' if host in ('0.0.0.0', '') else host}:{port}"

    banner = f"""
  ┌───────────────────────────────────────────────┐
     {APP_NAME}  v{__version__}
     Open in your browser:  {url}
     Data folder:  {DATA_DIR}
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
