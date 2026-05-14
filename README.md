# lash

Terminal AI coding agent written from scratch in Rust. Works with Anthropic, any OpenAI-compatible API, OpenAI Codex Subscription, and Google Gemini.

![lash TUI](screenshot.png)

Patch-based editing, shell execution, file search, web search, planning, skills, host-backed subagents, session resume/retry, provider-native variants, and live token accounting.

## What's different

- **Two execution modes**
  - `standard` (default) uses the provider's native tool-calling protocol directly. The model can emit multiple independent tool calls in a single response, which the runtime executes concurrently.
  - `rlm` runs a persistent `lashlang` DSL runtime that keeps state across turns. The model emits `lashlang` programs that are evaluated locally, with `parallel { }` blocks for concurrent tool execution.
- **Plugin architecture**: tools, prompts, planning, UI activity, subagents, memory, and history are plugins. Host applications compose only what they need through the app-facing `lash` crate.
- **Layered workspace**:
  - `lash-sansio`: pure turn machine, prompt model, messages, effects, and responses.
  - `lash-core`: async runtime internals, plugin host, providers, persistence, session graph, child-session orchestration, and built-in tools.
  - `lash`: app-facing facade for runtime construction, sessions, turn streaming, provider/mode/plugin wiring, and host integrations.
  - `lash-mode-standard` / `lash-mode-rlm`: execution-mode plugins.
  - `lash-standard-plugins`, `lash-subagents`, `lash-plugin-*`, `lash-provider-*`: first-party tool, plugin, and provider crates.
  - `lash-cli`: end-user TUI, setup, provider auth, session bootstrap/resume/fork, and `--print` mode.

## Install

Download the relevant release, use the install script below, or build from source.

```bash
curl -fsSL https://github.com/SamGalanakis/lash/releases/latest/download/install_lash.sh | bash
```

```bash
cargo build -p lash-cli --release
```

## Quick Start

```bash
lash                           # interactive TUI
lash -p "summarize this repo"  # single-shot, output to stdout
```

## Run the example

The `examples/agent-service` crate is a localhost SQLite-backed chat app that
showcases the `lash` facade, RLM mode, and typed plugin input. From the repo
root, with an OpenRouter key in the environment:

```bash
OPENROUTER_API_KEY=sk-or-... cargo run -p agent-service
```

Then open <http://127.0.0.1:3000>. See
[`examples/agent-service/README.md`](examples/agent-service/README.md) for the
optional environment knobs (`OPENROUTER_MODEL`, `AGENT_SERVICE_ADDR`,
`AGENT_SERVICE_DATA_DIR`, `AGENT_SERVICE_TRACE`, …).

## Development

The CI runtime performance gate uses the quick synthetic profile:

```bash
python3 scripts/profile_runtime.py --profile quick --release --cargo-feature fff-zlob --out .benchmarks/runtime-perf/ci.json
```

The nightly/manual `Performance` workflow runs the full profile:

```bash
python3 scripts/profile_runtime.py --profile full --release --cargo-feature fff-zlob --out .benchmarks/runtime-perf/full.json
```

## Docs

Browse them online at <https://samgalanakis.github.io/lash/> (mirrors the
`docs/` tree for `main` and `staging`).

- `docs/quickstart.html` is the shortest app-facing setup path.
- `docs/embedding.html` covers the `lash` facade API, session specs, plugin stacks, turn streaming, storage, and MCP.
- `docs/plugins.html` covers plugin factories, tool providers, default plugin stacks, and `ToolContext` capabilities.
- `docs/architecture/index.html` documents the current workspace architecture (start at `docs/architecture/lashlang.html` for the RLM execution language).
- `docs/example-agent-service.html` walks through the runnable `examples/agent-service` app.
- `docs/design-language.html` documents the TUI design language.

## License

MIT
