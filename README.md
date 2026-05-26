# lash

A Rust runtime for durable LLM agents.

Most agent stacks treat the LLM as the runtime and stitch state around it — a database for memory, a queue for retries, a sandbox for code. `lash` inverts that. The runtime is the durable end of the pair; the LLM is the variable call. Your app owns the outer boundaries — storage, auth, transport, product state. `lash` owns the turn — model calls, modes, tools, plugins, semantic stream events, usage, and terminal outcomes.

**Docs**: <https://lash.run/> — quickstart, embedding guide, plugins, persistence, durable workflow integration, architecture chapters.

> **Alpha:** works today, API still moving fast — pin to a commit when you embed.

## What's inside

### Durable per-turn commits

Every completed turn lands as one lease-fenced `RuntimeCommit` against a `SessionGraph` — graph delta, checkpoint blobs, usage deltas, and head revision in one SQLite transaction with optimistic CAS. In-flight turns persist separately as a lease-guarded `RuntimeTurnCheckpoint` plus effect-journal records. Scoped durable turns renew the same `RuntimeTurnLease` before checkpoint and journal writes, abandon ownership on non-commit exits while preserving resumable state, and clear the turn's checkpoint and journal only through a final commit that still owns an active, unexpired lease.

### Sans-IO state machine for workflow integration

`lash-core::RuntimeEffectController` is the durable boundary around nondeterministic work. LLM calls, individual tool calls, RLM exec, process control, checkpoints, retry sleeps, execution-surface sync, and direct/plugin LLM completions all cross it with stable invocation metadata, idempotency keys, checkpoint digests, and ref-only attachment specs. The default inline controller runs in process; workflow adapters pass a scoped controller with a stable turn id, while `LashRuntime::resume_turn(...)` / `LashSession::resume_turn(...)` reload the saved turn checkpoint and replay completed effects from the runtime journal. Process handles are explicit runtime support: install a `ProcessRegistry` such as `LocalProcessRegistry` when the host wants background process control; otherwise process start/list/await/cancel/transfer/cleanup fail loudly.

### Two execution modes, one commit unit

`standard` uses the provider's native tool-calling protocol — the model emits multiple independent tool calls in a single response, and the runtime dispatches them concurrently. `rlm` runs `lashlang` programs in a sandboxed VM with no direct filesystem, OS-process, or network surface; every effect crosses the Lashlang `ExecutionHost` and the linked host surface decides which resource/process abilities exist. Use RLM when the model should compose multiple tool calls per turn instead of one.

### Lashlang

A small typed DSL the model can emit and the runtime can execute deterministically. Receiver-first resource operations and named background `process` declarations are host-provided abilities: unavailable abilities still parse, but fail during linking and are omitted from the RLM prompt. Linked process modules are stored with the process input so nested starts survive later host ability drift. Trigger and cron schedule declarations are parsed and gated by the same host surface, but runtime activation for resource events and cron ticks remains a follow-up.

### Plugin architecture

Tools, prompts, planning, UI activity, subagents, memory, history transforms, and tool-output budgeting are all plugins. Host applications compose only what they need through the `lash` facade.

### Provider portability

First-party crates for Anthropic, OpenAI Responses, any OpenAI-compatible Chat Completions endpoint, OpenAI Codex subscription, and Google Gemini / Code Assist. MCP servers attach through `lash-plugin-mcp` over stdio, streamable-HTTP, or SSE.

### Tracing as a first-class sink

JSONL by default with a self-contained HTML viewer; optional OpenTelemetry export.

## Workspace layout

- `lash-sansio` — pure turn machine, prompt model, messages, effects, responses, checkpoints, tool contracts, and canonical tool-call output; no Lashlang dependency.
- `lash-core` — async runtime internals, plugin host, mode build input, providers, persistence, session graph, child-session orchestration, built-in tools, and Lashlang host-surface construction.
- `lash` — app-facing facade for runtime construction, sessions, turn streaming, provider / mode / plugin wiring, host integrations.
- `lash-mode-standard` / `lash-mode-rlm` — execution-mode plugins.
- `lash-standard-plugins`, `lash-subagents`, `lash-plugin-*`, `lash-provider-*` — first-party tool, plugin, and provider crates.
- `lashlang` — the RLM execution language: parser, VM, projection.
- `lash-cli` — first-party terminal frontend on top of the library.

## Embed it

The shortest path to a working turn. `lash` is shipped on crates.io as
`lash-runtime` (the bare name is owned by another project). During the
alpha series the versions carry an `-alpha.N` suffix, so the dep needs
the explicit pre-release tag:

```toml
[dependencies]
lash-runtime         = "=0.1.0-alpha.4"
lash-provider-openai = "=0.1.0-alpha.4"
anyhow               = "1"
tokio                = { version = "1", features = ["full"] }
```

The library is still imported as `lash` — only the crate name on
crates.io changes:

```rust
use lash::{provider::ProviderHandle, LashCore, ModelSpec, TurnInput};
use lash_provider_openai::{OPENROUTER_BASE_URL, OpenAiCompatibleProvider};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let api_key = std::env::var("OPENROUTER_API_KEY")?;
    let provider = ProviderHandle::new(
        OpenAiCompatibleProvider::new(api_key, OPENROUTER_BASE_URL).into_components(),
    );

    let model = ModelSpec::from_token_limits(
        "anthropic/claude-sonnet-4.6",
        None,
        200_000,
        None,
        None,
    )
    .map_err(anyhow::Error::msg)?;

    let core = LashCore::standard()
        .provider(provider)
        .model(model)
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

`examples/agent-service` is a localhost SQLite-backed chat app that exercises the `lash` facade end-to-end: RLM mode, typed session plugin activation, app-owned board tools, semantic streaming, per-chat model selection, SQLite runtime persistence, and optional Restate turn durability.

```bash
OPENROUTER_API_KEY=sk-or-... cargo run -p agent-service
```

Then open <http://127.0.0.1:3000>. See [`examples/agent-service/README.md`](examples/agent-service/README.md) for the optional environment knobs (`OPENROUTER_MODEL`, `AGENT_SERVICE_ADDR`, `AGENT_SERVICE_DATA_DIR`, `AGENT_SERVICE_TRACE`, `AGENT_SERVICE_DURABILITY`, …) and the one-command Restate E2E recipe.

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

That default matrix covers standard mode, RLM, RLM tool batches, large tool surfaces, observational-memory prompt and maintenance paths, embed paths, streaming, scoped effect controllers, store reopen, and durable turn-checkpoint round trips. The nightly / manual `Performance` workflow runs the full runtime profile:

```bash
python3 scripts/profile_runtime.py --profile full --release --cargo-feature fff-zlob --out .benchmarks/runtime-perf/full.json
```

It also runs the full UI profile and the Lashlang scenario sweep:

```bash
python3 scripts/profile_ui.py --profile full --release --cargo-feature fff-zlob --runs 5 --warmups 1 --out .benchmarks/ui-perf/full.json
python3 scripts/profile_lashlang.py --iterations 2500 --profile-iterations 2500 --out .benchmarks/lashlang-perf/full.json
```

## Contributing

Feature requests and bug reports welcome — open an [issue](https://github.com/SamGalanakis/lash/issues). At this stage detailed write-ups (what you tried, what you expected, what happened) help more than drive-by PRs; the internals are still moving and code may land in the wrong direction.

## License

MIT
