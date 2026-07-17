# Changelog

All notable changes from version 1.4.0 onward.

## 1.8.1 — Generation stats

### Added
- Every answer shows its measured performance under the bubble: **tokens/second,
  token count and wall time**, taken directly from Ollama's own generation
  timings rather than estimated. Hovering the line shows the prompt token count.
- Stats appear in plain chat, Agent, Loop and Coder runs, the Council chair's
  synthesis, and on regenerated answers. They are stored with each message, so
  conversation history shows the speed every answer had when it was generated.

## 1.8.0 — Remote access

### Added
- **Access password** for remote devices (Settings → Remote access). Every
  device that is not the server machine itself must unlock once; connections
  from the machine itself are always exempt. The check is based on the actual
  TCP peer address, so it cannot be spoofed with headers. Unlocks last 30 days
  per browser (HttpOnly cookie) and a server restart locks everyone again.
  Password comparison is constant-time and wrong guesses are slowed down.
- A clean unlock page for protected servers.
- When started with `--host 0.0.0.0`, the terminal banner now prints the real
  LAN address(es) to open on a phone, plus a reminder to set a password.

### Changed
- The request-origin protections now also allow Tailscale MagicDNS hostnames
  (`*.ts.net`), so remote access over Tailscale/Headscale works by name as
  well as by IP.

## 1.7.1 — Restart reliability

### Fixed
- **Self-update restarts no longer lose the port.** The restarted server used
  to compute its port from the original command line, so an auto-bumped or
  custom port reverted to the default and the browser was left polling a dead
  address. The restart now passes the exact host and port the server is bound
  to, and the new process waits for that same port.
- The update dialog polls for a full 60 seconds, only declares success once
  the **new** version answers (a lingering old server can no longer look like
  a successful update), and on timeout explains exactly where to look, with a
  "Keep waiting" option.

### Added
- Restarts are logged to `heorth_data/restart.log` (timestamp, target address,
  full output of the new process), so a failed restart is diagnosable instead
  of silent. A loud warning is printed if the original port could not be
  reclaimed.

### Changed
- Host protection now also accepts single-label LAN hostnames (`mypc`),
  `*.local` (mDNS), `*.localhost` and `*.home.arpa` — names that public DNS
  can never serve, so DNS-rebinding protection is unaffected. This fixes
  hostname-based access that 1.7.0's stricter checks had blocked.

## 1.7.0 — Security hardening, vision & workflow

### Security
- **The local API is now protected against DNS rebinding and cross-site
  request forgery.** Previously any website you visited could send requests to
  Heorth on localhost — including endpoints that change settings, register MCP
  servers or apply updates. A middleware now rejects requests with unexpected
  `Host` headers or foreign `Origin` headers.
- "Run app" artifacts are served with a CSP sandbox: generated HTML apps run
  with an opaque origin and cannot call Heorth's own API.
- All responses carry `X-Content-Type-Options: nosniff`.

### Added
- **Vision input**: attach up to 4 images to a message (on phones this opens
  the camera roll) and ask about them with a vision model. Images are stored
  with the conversation and replayed to the model on follow-ups, capped at the
  4 newest to keep VRAM and context in check.
- **Regenerate**: one click deletes the last answer and produces a new one
  with the current model and mode toggles, without duplicating your message.
- **Export**: download any conversation as Markdown or JSON.
- **Coder set-up dialog**: switching Coder on without a project folder now
  opens a dialog with a live folder check ("✓ Folder found — 87 entries"),
  an explanation of plan vs build mode, and a "Save & turn on" button.
- Model catalog additions: Llama 4 Scout, GPT-OSS 20B, Gemma 4 12B,
  Qwen3-Coder 30B and Devstral 24B.

### Changed
- Document indexing now writes all chunks in a single transaction instead of
  one commit per chunk — large PDFs index dramatically faster.
- Internal clean-ups: a shared `flag()` helper for boolean settings, a
  rewritten context builder, removal of dead parameters.

## 1.6.0 — Runnable artifacts

### Added
- **Copy button on every code block** in chat, with the language shown in a
  header bar. Works during streaming, on old messages, and over plain-HTTP
  LAN connections (clipboard fallback included).
- **"▶ Run app" on HTML code blocks**: saves the code as a real file under
  `heorth_data/artifacts/` and opens it running in a new browser tab. Files
  are named by content hash so re-running the same block reuses the same
  file; filenames are sanitized on serving.

### Fixed
- The markdown renderer wrapped code blocks (and other block elements) inside
  `<p>` tags, producing invalid HTML that browsers silently tolerated.
- `coder_read` on an empty file produced a nonsensical header.
- Removed a dead `stream` parameter from the Ollama client.

## 1.5.0 — Coder mode

### Added
- **Coder mode**: a coding agent in the spirit of opencode / Claude Code.
  Point it at a project folder and it explores, edits and tests the code
  autonomously with six tools — file tree, read with line numbers, regex
  search, exact-match edit (returns a diff), whole-file write, and shell in
  the project directory.
- Every Coder tool is hard-locked inside the chosen folder: `..` traversal,
  absolute paths and symlinks pointing outside are all refused.
- **Plan mode by default**: Coder starts read-only and proposes diffs; direct
  file edits require switching on "Allow file edits". Shell commands reuse the
  existing "Allow code execution" consent.
- Coder toggle in the chat header, a Coder section in Settings, and its own
  step limit. Web search is available to it (for documentation) when web
  tools are enabled.

## 1.4.1 — Auto-search with reasoning models

### Fixed
- The automatic web search silently failed with reasoning ("thinking")
  models: the search classifier capped output at 40 tokens, which the model
  spent entirely on thinking, so no search decision was ever produced and the
  model answered "I can't access the internet." The classifier now asks with
  thinking disabled, retries with a much larger budget when a model doesn't
  support that flag, and reads the reasoning field as a last resort.

## 1.4.0 — Baseline

The starting point for this changelog. At 1.4.0, Heorth (renamed from
LocalMind in an earlier release) already included: streaming chat with
thinking-model support, automatic web search, Knowledge (document Q&A over a
local index), Agent and autonomous Loop modes, Council (a debating panel of
model instances), Computer control behind a consent gate, local image
generation, an MCP client, private SearXNG search with DuckDuckGo fallback,
a hardware-aware model manager, and self-update from the Downloads folder.
