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

## Docs

- `docs/architecture/index.html` documents the current workspace architecture.
- `docs/design-language.html` documents the TUI design language.

## License

MIT
