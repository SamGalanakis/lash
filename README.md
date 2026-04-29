# lash

Terminal AI coding agent written from scratch in Rust. Works with Anthropic, any OpenAI-compatible API, OpenAI Codex Subscription, and Google Gemini.

![lash TUI](screenshot.png)

Patch-based editing, shell execution, file search, web search, planning, skills, host-backed subagents, session resume/retry, provider-native variants, and live token accounting.

## What's different

- **Three execution modes**
  - `standard` (default) uses the provider's native tool-calling protocol directly. The model can emit multiple independent tool calls in a single response, which the runtime executes concurrently.
  - `rlm` runs a persistent `lashlang` DSL runtime that keeps state across turns. The model emits `lashlang` programs that are evaluated locally, with `parallel { }` blocks for concurrent tool execution.
  - `rlmpure` uses the same persistent `lashlang` runtime with a compact context projector that renders task inputs plus REPL trajectory instead of full chat history.
- **Plugin architecture**: tools, prompts, planning, UI activity, subagents, memory, and history are plugins. Embedders compose only what they need.
- **Embeddable workspace**:
  - `lash-sansio`: pure turn machine, prompt model, messages, effects, and responses.
  - `lash`: async runtime, plugin host, providers, persistence, session graph, child-session orchestration, and built-in tools.
  - `lash-mode-standard` / `lash-mode-rlm` / `lash-mode-rlmpure`: execution-mode plugins.
  - `lash-default-tools`, `lash-subagents`, `lash-plugin-*`, `lash-provider-*`: first-party tool, plugin, and provider crates.
  - `lash-cli`: end-user TUI, setup, provider auth, session bootstrap/resume/fork, and `--print` mode.

## Agent behavior

- Continue with the next concrete action while actionable work remains.
- Do not stop merely to report that work is incomplete.
- Only summarize remaining work when blocked, when a decision is required, or when no further feasible action remains in the current turn.
- Do not claim completion unless the required end state has actually been verified.

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
