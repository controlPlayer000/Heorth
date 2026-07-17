# Heorth

![License: MIT](https://img.shields.io/badge/license-MIT-blue) ![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)

**A self-hosted AI workbench in a single Python file.**

Heorth runs open-weight language models on your own machine — chat, autonomous
agents, a coding assistant, document Q&A, vision, image generation and more,
behind one local web interface. No accounts, no cloud, no telemetry: your
conversations, documents and generated files never leave your computer.

Everything is one file: `heorth.py`.

## Quick start

1. Install [Python 3.10+](https://www.python.org) and [Ollama](https://ollama.com)
2. Download `heorth.py` and run it:

```bash
python3 heorth.py        # Windows: python heorth.py
```

That's it. On first run Heorth creates a private Python environment next to
itself (`heorth_data/venv`), installs its dependencies into it, restarts inside
that environment and opens `http://127.0.0.1:8317` in your browser. The
built-in model manager detects your hardware (GPU, VRAM, RAM) and recommends
models that will actually run well on it — one click to download, one click to
chat.

## Chat modes

| Mode | What it does |
|---|---|
| **Chat** | Plain conversation, with streaming, reasoning-model ("thinking") support and automatic web search when a question needs live information |
| **Knowledge** | Answers grounded in your own documents (PDF, text, markdown) via a local embedding index |
| **Agent** | Single-turn tool use: web search, fetch pages, run code, read/write workspace files, calculator, your MCP servers |
| **Loop** | Autonomous agent: plans, calls tools, observes and repeats until it declares the task complete |
| **Council** | A panel of models answer independently in parallel, critique each other over consultation rounds, and a chair writes the synthesis |
| **Computer** | The model sees your screen and controls mouse and keyboard (PyAutoGUI) — off by default, behind a consent gate, with a live action log and an emergency stop |
| **Coder** | A coding agent in the spirit of [opencode](https://opencode.ai) / Claude Code: point it at a project folder and it explores, edits and tests the code — locked inside that folder, read-only "plan mode" until you allow edits |

## More features

- **Runnable artifacts** — every code block gets a Copy button; HTML apps get a
  "▶ Run app" button that saves the file and opens it running in a new tab
  (served sandboxed, so generated apps can't touch Heorth's own API)
- **Vision input** — attach up to 4 images to a message and ask about them with
  a vision model (gemma3, qwen2.5vl, llava, …)
- **Regenerate & export** — redo the last answer with one click; download any
  conversation as Markdown or JSON
- **Image generation** — local Stable Diffusion (optional, one-click install
  from the GUI)
- **Private web search** — optional self-hosted SearXNG container (the GUI
  guides you), with DuckDuckGo as the fallback
- **MCP client** — connect Model Context Protocol servers and the agent can
  use their tools
- **Self-update** — drop a newer `heorth*.py` in your Downloads folder and the
  GUI offers to update, backs up the old version, and restarts itself on the
  same address

## Using it from your phone

Heorth's UI is responsive and works well in mobile browsers. On your Wi-Fi:

```bash
python3 heorth.py --host 0.0.0.0
```

The terminal prints the address to open on the phone (e.g.
`http://192.168.1.20:8317`). Set an **access password** in
Settings → Remote access — devices other than the server itself must unlock
before they can use anything. For access from outside your home, use a
WireGuard-based mesh like Tailscale or the fully open-source
[Headscale](https://github.com/juanfont/headscale) / [NetBird](https://netbird.io);
do **not** port-forward Heorth directly to the internet (it speaks plain HTTP).

## Security model

Heorth is local-first software with real teeth for the ways local apps get
attacked:

- **DNS-rebinding and CSRF protection** — requests with unexpected Host or
  cross-site Origin headers are rejected, so a malicious website you visit
  cannot reach the API on localhost
- **Optional access password** for every non-localhost device, with
  constant-time comparison and brute-force damping
- **Sandboxed artifacts** — model-generated HTML apps run with an opaque
  origin and cannot call Heorth's API
- **Consent gates** — code execution, computer control and Coder file edits
  are all off by default and must be switched on explicitly
- **Path confinement** — workspace, Coder and image tools each refuse to
  touch anything outside their designated folder (including symlink escapes)

Heorth has no authentication on `127.0.0.1` by design: anything running on
your own machine is already inside your trust boundary.

## Command line

```
python3 heorth.py [--host 127.0.0.1] [--port 8317] [--no-browser]
```

| Flag | Meaning |
|---|---|
| `--host` | Bind address. Use `0.0.0.0` to allow other devices on your network |
| `--port` | Preferred port (auto-bumps if busy) |
| `--no-browser` | Don't open the browser on start |

## Data & privacy

Everything Heorth stores lives in `heorth_data/` next to the script:
conversations (SQLite), your document index, generated images and artifacts,
the agent workspace, update backups and logs. Delete the folder and Heorth
starts fresh. Nothing is sent anywhere except the requests you explicitly
make (model downloads from Ollama, web search when enabled).

## Requirements

- Python 3.10 or newer
- [Ollama](https://ollama.com) for text generation (the GUI shows one-line
  install instructions if it's missing)
- Optional: Docker (private SearXNG search), a GPU (larger models, image
  generation)

## License

[MIT](LICENSE)
