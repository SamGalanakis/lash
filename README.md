# lash

Terminal AI coding agent written from scratch in Rust. Works with any OpenAI-compatible API, OpenAI Codex Subscription, or Google Gemini.

![lash TUI](screenshot.png)

Patch-based editing, shell execution, file search, web search, planning, skills, host-backed subagents, session resume/retry, and live token accounting.

## What's different

- **Two execution modes**
  - `rlm` (default) — runs a persistent `lashlang` DSL runtime that keeps state across turns. The model emits `lashlang` programs that are evaluated locally, with `parallel { }` blocks for concurrent tool execution. This gives the agent a programmable scratch space and control flow beyond what the provider protocol offers.
  - `standard` — uses the provider's native tool-calling protocol directly. The model can emit multiple independent tool calls in a single response, which the runtime executes concurrently. Simpler, no extra DSL layer, closer to how the provider intended the API to be used.
- **Plugin architecture** — tools, prompts, planning, subagents, and history are all plugins; embedders compose what they need
- **Embeddable** — split into three crates so you can use only what you need:
  - `lash-sansio` — sans-IO turn machine. Pure logic: takes messages in, yields effects out, no networking or filesystem. Embed this if you want full control over I/O and just need the turn loop.
  - `lash` — async host runtime built on `lash-sansio`. Adds plugin infrastructure, tool execution, session management, context approaches, and child-session orchestration. Use this to embed a full agent in a larger Rust application.
  - `lash-cli` — the end-user binary. TUI, provider auth, config, and the single-shot `--print` mode. Built on top of `lash`.

## Agent behavior

- Finish the task by continuing with the next concrete action while actionable work remains.
- Do not stop just to report that work is incomplete.
- Only summarize remaining work when blocked, when a decision is required, or when no further feasible action remains in the current turn.
- Do not claim completion unless the required end state has actually been verified.

## Install

Download the relevant release, use the install script below or build from source.

```bash
curl -fsSL https://github.com/SamGalanakis/lash/releases/latest/download/install_lash.sh | bash
```

```bash
cargo build -p lash-cli --release
```

## Quick start

```bash
lash                           # interactive TUI
lash -p "summarize this repo"  # single-shot, output to stdout
```

## Agent behavior

- Continue with the next concrete action while actionable work remains.
- Do not stop merely to report that work is incomplete.
- Only summarize remaining work when blocked, when a decision is required, or when no further feasible action remains in the current turn.
- Do not claim completion unless the required end state has actually been verified.

## Docs

See `docs/index.html` for architecture, design, and full reference.

## License

MIT
