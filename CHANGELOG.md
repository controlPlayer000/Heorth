# Changelog

## 1.10.0 — Images move into the chat

### Added
- **/imagine in chat**: type `/imagine a lighthouse at dusk, oil painting` in
  any conversation and the picture is generated right there, with live
  progress, saved into the conversation history and the gallery. Uses the
  default model chosen on the Images page; Regenerate works on image messages
  too.
- **Custom image models**: add any Hugging Face diffusers text-to-image repo
  on the Images page and it appears in the model picker (weights download on
  first use).
- The model picker now shows which models are already downloaded (✓) versus
  their first-use download size, and remembers your default choice.

### Changed
- The Images page is now primarily the gallery plus model management; the
  full generation form remains there for seeds, sizes and experimenting.

## 1.9.0 — Reliability round

### Fixed
- **Model downloads survive leaving the page.** Pulls previously streamed over
  the browser's own request, so navigating away aborted them and the Models
  page later showed a plain Download button again. Downloads now run as
  server-side background jobs; the Models page shows a live "Downloading"
  section that re-attaches to in-flight pulls whenever you come back, with a
  Retry card on failure.
- **Image generation no longer loops on "Requirement already satisfied".**
  When the packages installed fine but the engine failed to load, the UI spun
  on "Installing…" forever and re-ran pip on every attempt while hiding the
  real error. The actual import error is now captured and shown, with a
  one-click server restart (which usually fixes it) and a re-check button.

### Added
- **Clear warning when Ollama is missing**: trying to download a model
  without Ollama running now explains the problem immediately and points to
  the setup instructions instead of failing quietly.
- **Restart button** in Settings → Server: restarts the process on the same
  address and reconnects automatically — handy after installing optional
  features.

## 1.8.1 — Generation stats
- Every answer now shows measured performance under the bubble: tokens/second,
  token count and wall time, taken directly from Ollama's own timings (prompt
  token count in the tooltip). Works in plain chat, Agent/Loop/Coder, Council,
  on regenerated answers, and in conversation history.

## 1.8.0 — Remote access
- Optional access password for every device that is not the server machine
  itself (localhost is always exempt; 30-day unlock cookie; constant-time
  checks; brute-force damping)
- Clean unlock page for protected servers
- Startup banner prints the real LAN address(es) to open on a phone when
  running with `--host 0.0.0.0`
- Tailscale MagicDNS (`*.ts.net`) hostnames allowed through the
  request-origin protections

## 1.7.1 — Restart reliability
- Self-update restarts now reuse the exact host and port the browser is on
  (previously an auto-bumped or custom port reverted to the default and
  stranded the browser)
- Restarts are logged to `heorth_data/restart.log` for diagnosis
- Update dialog polls for the new version for 60 s, can't mistake a lingering
  old server for a successful update, and gives actionable guidance on timeout
- Host protection now also allows single-label LAN names, `*.local`,
  `*.localhost` and `*.home.arpa`

## 1.7.0 — Security, vision & polish
- Hardened the local API against DNS rebinding and cross-site request forgery
  (Host/Origin validation middleware)
- "Run app" artifacts are served sandboxed and can no longer call Heorth's API
- Vision input: attach images to a message and ask about them
- Regenerate the last answer; export conversations as Markdown or JSON
- Coder set-up dialog with live folder validation
- 2026 model catalog additions (Llama 4 Scout, GPT-OSS, Gemma 4, Qwen3-Coder,
  Devstral)
- Faster document indexing (batched inserts) and internal clean-ups

## 1.6.0 — Runnable artifacts
- Copy button on every code block (works over plain-HTTP LAN too)
- "▶ Run app" on HTML code blocks: saves the file and opens it in a new tab
- Full code review; renderer and small fixes

## 1.5.0 — Coder mode
- opencode-style coding agent over a real project folder: tree, read, grep,
  exact-match edit (with diffs), write, shell — all locked inside the chosen
  folder, read-only "plan mode" by default

## 1.4.1
- Automatic web search now works with reasoning ("thinking") models

## 1.0 – 1.4
- Initial releases: chat with thinking-model support, Knowledge (document
  Q&A), Agent and Loop modes, Council, computer control, local image
  generation, MCP client, private SearXNG search, hardware-aware model
  manager, self-update — and the rename from LocalMind to Heorth
