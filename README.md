# Heorth

**A complete local AI server in a single Python file.**
Chat, knowledge base (RAG), tool-using agents, autonomous loops, a multi-agent
council, image generation, MCP and private web search — all behind a clean
browser GUI, all running on your own machine. Nothing leaves your computer.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Deployment](https://img.shields.io/badge/deployment-single%20file-orange)
![Platforms](https://img.shields.io/badge/platforms-Linux%20%7C%20macOS%20%7C%20Windows-8A2BE2)

> *Heorth* is Old English for "hearth" — your models, at home.

---

## Quick start

```bash
# 1. Install Ollama once (powers text generation)  →  https://ollama.com
# 2. Run Heorth:
python3 heorth.py          # Windows: python heorth.py
```

That's the whole install. On first run Heorth creates a private Python
environment next to itself (`heorth_data/venv`), installs its dependencies
into it, restarts inside it, and opens **http://127.0.0.1:8317** in your
browser. If Ollama isn't installed yet, the GUI shows the one-line install
command for your OS and reconnects automatically once it's up.

## What's inside

| Area | What you get |
|---|---|
| **Dashboard** | Scans your RAM/GPU/CPU and recommends models that actually fit, with one-click downloads and a live "usable memory" gauge. |
| **Chat** | Streaming responses, saved conversations, markdown, a Stop button that keeps partial replies. |
| **Models** | Curated catalog plus live search across Ollama tags and Hugging Face GGUF repos. Pull with progress bars, remove with one click. |
| **Knowledge (RAG)** | Drag-and-drop PDF / text / Markdown / code. Files are chunked and embedded locally; answers cite their sources. |
| **Agent** | The model can call 10 built-in tools: web search, fetch URL, calculator, knowledge search, workspace file I/O, image generation, and opt-in Python/shell execution. |
| **Loop** | Autonomous mode: the agent plans → acts → observes → repeats, thoughts visible live, until it calls `task_complete` or hits your iteration ceiling. |
| **Council** | A panel of 2–10 consultants. Roles are designed per-question (always including a contrarian), they analyze **in parallel**, critique each other over consultation rounds, and a chair synthesizes the final answer. |
| **Images** | Stable Diffusion (SD-Turbo / DreamShaper / SDXL-Turbo) installed from the GUI in one click. Every image auto-saves to a gallery; click to enlarge, hover to delete. |
| **MCP** | Connect any stdio [Model Context Protocol](https://modelcontextprotocol.io) server; its tools appear to the agent automatically. |
| **Web search** | Works out of the box (DuckDuckGo). One click sets up a private [SearXNG](https://docs.searxng.org) metasearch container via Docker — Heorth writes the config, starts the container, and routes the agent's searches through it. |
| **Self-update** | Drop a newer `heorth*.py` into Downloads (or upload it in Settings). Heorth notices, shows an Update button, backs up the current version, swaps the file and restarts in place. |

## Chat modes

- **Plain** — just talk to the model.
- **Knowledge** — answers grounded in your uploaded documents, with sources shown.
- **Agent** — single-turn tool use: the model calls tools, then answers.
- **Loop** — autonomous multi-step work with a hard iteration limit (Settings, default 15).
- **Council** — parallel multi-agent deliberation; size, consultation rounds and
  an optional shared web-research brief are configurable in Settings.

Knowledge combines with any mode. Council is exclusive with Agent/Loop.

## Optional features (one click each, from the GUI)

| Feature | What it installs | Size |
|---|---|---|
| Image generation | PyTorch + diffusers into Heorth's private venv | a few GB, once |
| MCP client | the `mcp` package | tiny |
| SearXNG search | a Docker container (`heorth-searxng`) with the JSON API pre-configured | ~300 MB image |

SearXNG needs Docker; if it's missing, the Agent & Tools page shows the
install steps for your OS first.

## Flags & configuration

```
python3 heorth.py --port 9000      # different port (default 8317)
python3 heorth.py --host 0.0.0.0   # expose on your LAN (default 127.0.0.1)
python3 heorth.py --no-browser     # don't open the browser
python3 heorth.py --system         # skip the private venv, use current Python
HEORTH_DATA=/path python3 heorth.py   # relocate the data folder
```

Everything else — system prompt, context window, tool permissions, agent/loop/
council limits, search backend, Ollama address, embedding model — lives in
**Settings** in the GUI.

## Your data

Everything Heorth stores lives in one folder next to the script:

```
heorth_data/
├── heorth.db      # conversations, settings, document index
├── images/        # generated images
├── rag_docs/      # your uploaded originals
├── workspace/     # the agent's sandbox for file tools
├── backups/       # previous versions, kept on every self-update
├── updates/       # drop update files here (Downloads works too)
└── venv/          # Heorth's private Python environment
```

Delete the folder to reset completely. Model weights themselves are managed
by Ollama. Nothing is sent anywhere except the requests you explicitly make
(web search / URL fetch tools, model downloads).

## How self-update works

Each release is a single file with a `__version__` string. Heorth watches its
own folder, `heorth_data/updates/` and `~/Downloads` for any newer
`heorth*.py` (legacy `localmind*.py` also accepted). Updating validates the
file compiles, backs up the running version to `backups/`, atomically swaps
the script and restarts — the browser reconnects by itself. Installs that
began life under the old LocalMind name keep their `localmind_data` folder
and database automatically.

## Troubleshooting

- **Black generated images** — fixed in v1.2.1. Apple Silicon now runs the
  VAE in full precision (fp16 on MPS produces NaN → black frames), and CUDA
  cards with broken fp16 (e.g. GTX 16-series) are auto-detected: Heorth
  retries in fp32 and remembers.
- **Port already in use** — Heorth automatically tries the next free port,
  or pass `--port`.
- **"Ollama isn't running"** — install/start Ollama; the Models page has
  per-OS instructions and a re-check button.
- **Council feels slow** — that's local hardware reality. Requests are
  dispatched in parallel, but true concurrency depends on your VRAM and
  Ollama's `OLLAMA_NUM_PARALLEL`. 3–5 consultants is the sweet spot.
- **Fresh dependency state** — delete `heorth_data/venv` and rerun; Heorth
  rebuilds it.

## Architecture (for the curious)

One file by design — it's what makes the self-update mechanism trivial.
Inside: a FastAPI backend (chat streaming over NDJSON, SQLite for state,
numpy cosine search for RAG), Ollama as the text-generation engine, diffusers
for images, and a zero-build vanilla-JS single-page frontend embedded as a
base64 blob. The venv bootstrap keeps the host Python untouched.

## Contributing

Issues and PRs welcome. Keep it single-file, keep it dependency-light, and
bump `__version__` in any PR that changes behavior — the update mechanism
relies on it.

## License

MIT — see [LICENSE](LICENSE).

## Credits

Built on the shoulders of [Ollama](https://ollama.com),
[Hugging Face diffusers](https://github.com/huggingface/diffusers),
[SearXNG](https://github.com/searxng/searxng),
[FastAPI](https://fastapi.tiangolo.com) and the
[Model Context Protocol](https://modelcontextprotocol.io).
