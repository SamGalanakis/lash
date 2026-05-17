# lash

A Rust runtime for durable LLM agents.

Most agent stacks treat the LLM as the runtime and stitch state around it — a database for memory, a queue for retries, a sandbox for code. `lash` inverts that. The runtime is the durable end of the pair; the LLM is the variable call. Your app owns the outer boundaries — storage, auth, transport, product state. `lash` owns the turn — model calls, modes, tools, plugins, semantic stream events, usage, and terminal outcomes.

**Docs**: <https://lash.run/> — quickstart, embedding guide, plugins, persistence, durable workflow integration, architecture chapters.

> **Alpha:** works today, API still moving fast — pin to a commit when you embed.

## What's inside

### Durable per-turn commits

Every turn lands as one `RuntimeCommit` against a `SessionGraph` — graph delta, checkpoint blobs, usage deltas, and head revision in one SQLite transaction with optimistic CAS. Partial turn = no commit.

### Sans-IO state machine for workflow integration

`lash-sansio::TurnMachine` is a pure effect / response state machine with deterministic `EffectId`. Snapshot it, ship the bytes to another worker, resume against the same logical effect. Built for Temporal, Restate, and other durable-workflow runtimes.

### Two execution modes, one commit unit

`standard` uses the provider's native tool-calling protocol — the model emits multiple independent tool calls in a single response, the runtime dispatches them in parallel. `rlm` runs `lashlang` programs in a sandboxed VM with no filesystem, process, or network surface; every effect crosses `ToolHost`. Use RLM when the model should compose multiple tool calls per turn instead of one.

### Lashlang

A small typed DSL the model can emit and the runtime can execute deterministically. `parallel { }` blocks for concurrent tool batches, projected read-only bindings from the host, no syscalls, fully checkpointable.

### Plugin architecture

Tools, prompts, planning, UI activity, subagents, memory, history transforms, and tool-output budgeting are all plugins. Host applications compose only what they need through the `lash` facade.

### Provider portability

First-party crates for Anthropic, OpenAI Responses, any OpenAI-compatible Chat Completions endpoint, OpenAI Codex subscription, and Google Gemini / Code Assist. MCP servers attach through `lash-plugin-mcp` over stdio, streamable-HTTP, or SSE.

### Tracing as a first-class sink

JSONL by default with a self-contained HTML viewer; optional OpenTelemetry export.

## Workspace layout

- `lash-sansio` — pure turn machine, prompt model, messages, effects, responses, checkpoint / restore.
- `lash-core` — async runtime internals, plugin host, providers, persistence, session graph, child-session orchestration, built-in tools.
- `lash` — app-facing facade for runtime construction, sessions, turn streaming, provider / mode / plugin wiring, host integrations.
- `lash-mode-standard` / `lash-mode-rlm` — execution-mode plugins.
- `lash-standard-plugins`, `lash-subagents`, `lash-plugin-*`, `lash-provider-*` — first-party tool, plugin, and provider crates.
- `lashlang` — the RLM execution language: parser, VM, projection.
- `lash-cli` — first-party terminal frontend on top of the library.

## Embed it

The shortest path to a working turn. Pull the crates straight from the monorepo:

```toml
[dependencies]
lash                 = { git = "https://github.com/SamGalanakis/lash.git" }
lash-provider-openai = { git = "https://github.com/SamGalanakis/lash.git" }
anyhow = "1"
tokio  = { version = "1", features = ["full"] }
```

```rust
use lash::{provider::ProviderHandle, LashCore, TurnInput};
use lash_provider_openai::{OPENROUTER_BASE_URL, OpenAiCompatibleProvider};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let api_key = std::env::var("OPENROUTER_API_KEY")?;
    let provider = ProviderHandle::new(
        OpenAiCompatibleProvider::new(api_key, OPENROUTER_BASE_URL).into_components(),
    );

    let core = LashCore::standard()
        .provider(provider)
        .model("anthropic/claude-sonnet-4.6", None)
        .max_context_tokens(200_000)
        .build()?;

    let session = core.session("hello-1").open().await?;
    let result = session
        .turn(TurnInput::text("Say hi in one short sentence."))
        .run()
        .await?;

    println!("{}", result.assistant_message().unwrap_or_default());
    Ok(())
}
```

See [`docs/quickstart.html`](https://lash.run/quickstart.html) for the full walkthrough, and [`docs/embedding.html`](https://lash.run/embedding.html) for the complete facade API — session specs, plugin stacks, turn streaming, persistence, subagents, MCP wiring, and durable-workflow integration.

## Run the example

`examples/agent-service` is a localhost SQLite-backed chat app that exercises the `lash` facade end-to-end: RLM mode, typed plugin input, app tools, semantic streaming, and per-chat model selection.

```bash
OPENROUTER_API_KEY=sk-or-... cargo run -p agent-service
```

Then open <http://127.0.0.1:3000>. See [`examples/agent-service/README.md`](examples/agent-service/README.md) for the optional environment knobs (`OPENROUTER_MODEL`, `AGENT_SERVICE_ADDR`, `AGENT_SERVICE_DATA_DIR`, `AGENT_SERVICE_TRACE`, …).

## The CLI

`lash-cli` is a first-party terminal frontend on top of the library — coding-agent affordances (patch-based editing, shell execution, file search, web search, planning, skills, host-backed subagents, session resume / retry, provider-native variants, live token accounting). It's not the product, but it's a fully featured way to drive the runtime from a terminal and a useful reference for end-to-end integration.

![lash TUI](screenshot.png)

```bash
curl -fsSL https://github.com/SamGalanakis/lash/releases/latest/download/install_lash.sh | bash
```

```bash
cargo build -p lash-cli --release
```

```bash
lash                           # interactive TUI
lash -p "summarize this repo"  # single-shot, output to stdout
```

CLI reference: [`docs/cli.html`](https://lash.run/cli.html).

## Development

The CI runtime-performance gate uses the quick synthetic profile:

```bash
python3 scripts/profile_runtime.py --profile quick --release --cargo-feature fff-zlob --out .benchmarks/runtime-perf/ci.json
```

That default matrix covers standard mode, RLM, RLM tool batches, large tool surfaces, embed paths, streaming, and durable turn-checkpoint round trips. The nightly / manual `Performance` workflow runs the full profile:

```bash
python3 scripts/profile_runtime.py --profile full --release --cargo-feature fff-zlob --out .benchmarks/runtime-perf/full.json
```

## Contributing

Feature requests and bug reports welcome — open an [issue](https://github.com/SamGalanakis/lash/issues). At this stage detailed write-ups (what you tried, what you expected, what happened) help more than drive-by PRs; the internals are still moving and code may land in the wrong direction.

## License

MIT
