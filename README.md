# lash

A Rust runtime for durable LLM agents.

Most agent stacks treat the LLM as the runtime and stitch state around it — a database for memory, a queue for retries, a sandbox for code. `lash` inverts that. The runtime is the durable end of the pair; the LLM is the variable call. Your app owns the outer boundaries — storage, auth, transport, product state. `lash` owns the turn — model calls, modes, tools, plugins, semantic stream events, usage, and terminal outcomes.

**Docs**: <https://lash.run/> — quickstart, embedding guide, plugins, persistence, durable workflow integration, architecture chapters.

> **Alpha:** works today, API still moving fast — pin to a commit when you embed.

## What's inside

### Durable per-turn commits

Every completed turn lands as one semantic `RuntimeCommit` against a `SessionGraph` — graph delta, checkpoint blobs, usage deltas, queued-work completions, attachment manifests, and head revision in one optimistic transaction. Lash owns stable `turn_id`s, replay keys, causal metadata, and final commit idempotency. Stores persist committed Lash state and durable work records. Effect hosts own in-flight nondeterministic work: the inline host is local and non-durable, while durable workflow hosts such as Restate replay effects from host history and timers. Effects are the replay boundary; turns are the semantic commit boundary.

### Sans-IO state machine for workflow integration

`lash-core::EffectHost` is the host integration boundary around nondeterministic work. LLM calls, individual tool calls, RLM exec, process control, retry sleeps, execution-surface sync, and direct/plugin LLM completions all cross a scoped controller with a typed `RuntimeInvocation`: scoped session/turn coordinates, a subject, optional causal parent, `replay.key`, and ref-only attachment specs. The default `InlineEffectHost` runs in process and reopens only the last committed state after a local crash. Workflow adapters create handler-scoped `ScopedEffectController`s for stable `EffectScope`s; Restate recovery reruns the handler with the same turn id, replays effects from Restate history, and lets Lash retry the final idempotent commit. Process handles are explicit persistence support: install a deployment-level `ProcessRegistry` such as `lash-turso-store::TursoProcessRegistry` when the host wants background process control; otherwise process start/list/await/cancel/transfer/cleanup fail loudly. Host-facing process commands stay on `ProcessControl`; optional process observation attaches through trace sinks such as `TraceLashlangGraphStore`.

### Two execution modes, one commit unit

`standard` uses the provider's native tool-calling protocol — the model emits multiple independent tool calls in a single response, and the runtime dispatches them concurrently. `rlm` runs `lashlang` programs in a sandboxed VM with no direct filesystem, OS-process, or network surface; every effect crosses the Lashlang `ExecutionHost` and the linked host surface decides which resource/process abilities exist. Use RLM when the model should compose multiple tool calls per turn instead of one.

### Lashlang

A small typed DSL the model can emit and the runtime can execute deterministically. Host capabilities are exposed as lowercase module operations such as `web.search(...)`, `files.read(...)`, and `agents.spawn(...)`; named `process` declarations define reusable background work. `start name(...)` and trigger activations create process runs from those definitions. Unavailable abilities still parse, but fail during linking and are omitted from the RLM prompt. Trigger registration installs durable rules from host-provided source values to process definitions plus explicit input mappings; source owners activate exact handles or source types. Timers and recurring jobs are host/plugin concerns, not core syntax or built-in sources.

### Plugin architecture

Tools, prompts, planning, UI activity, subagents, memory, history transforms, and tool-output budgeting are all plugins. Host applications compose only what they need through the `lash` facade.

### Provider portability

First-party crates for Anthropic, OpenAI Responses, any OpenAI-compatible Chat Completions endpoint, OpenAI Codex subscription, and Google Gemini / Code Assist. MCP servers attach through `lash-plugin-mcp` over stdio, streamable-HTTP, or SSE.

### Tracing as a first-class sink

Attach a `TraceSink` for structured turn, tool, LLM, prompt, stream, and usage records. The bundled JSONL sink pairs with a self-contained HTML viewer; OpenTelemetry export is feature-gated. Lashlang execution graphs are a separate opt-in sink for foreground Lashlang blocks, durable processes, node/branch observations, and child execution links, so host observability can be richer without changing process registry state. `TraceLashlangGraphStore` reduces those records into host-safe graph snapshots for UIs, dashboards, tests, and debugging; the snapshots are trace-derived projections, not canonical process state. Process wake provenance is typed runtime metadata for hosts to inspect, while labels, colors, icons, and other presentation stay host-owned.

## Workspace layout

- `lash-sansio` — pure turn machine, prompt model, messages, effects, responses, checkpoints, tool contracts, and canonical tool-call output; no Lashlang dependency.
- `lash-core` — async runtime internals, plugin host, protocol build input, providers, persistence, session graph, child-session orchestration, built-in tools, and Lashlang host-surface construction.
- `lash-remote-protocol` — runtime-neutral canonical DTOs for wrapping Lash behind a service boundary: remote turn requests/results, LLM requests/responses, prompt patches, activity streams, and transport-neutral tool grants.
- `lash` — app-facing facade for runtime construction, sessions, turn streaming, provider / mode / plugin wiring, host integrations.
- `lash-protocol-standard` / `lash-protocol-rlm` — protocol plugins.
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
lash-runtime         = "=0.1.0-alpha.33"
lash-provider-openai = "=0.1.0-alpha.33"
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

### Remote service boundary

Hosts that expose Lash through HTTP, queues, callbacks, or workflow handlers should use the canonical remote DTOs from `lash::remote` or `lash-remote-protocol`. Wrap `RemoteTurnRequest` and `RemoteTurnResult` with product-owned auth, billing, routing, persistence, and tenant metadata; do not redefine Lash sub-DTOs in downstream services. Product-specific data belongs in the host wrapper or the DTO `metadata` maps, while Lash-owned fields such as prompt patches, tool grants, activities, LLM calls, usage, and terminal outcomes stay in the protocol crate.

## Examples

Two runnable apps under `examples/` drive the `lash` facade end-to-end — full hosts with a browser UI, real persistence, and optional durable execution. The docs walk through both at <https://lash.run/examples.html>.

`examples/agent-service` is a localhost SQLite-backed chat app: RLM protocol, typed session plugin activation, app-owned board tools, semantic streaming, per-chat model selection, Turso runtime persistence, and optional Restate-backed turns.

```bash
OPENROUTER_API_KEY=sk-or-... cargo run -p agent-service
```

Then open <http://127.0.0.1:3000>. See [`examples/agent-service/README.md`](examples/agent-service/README.md) for the optional environment knobs (`OPENROUTER_MODEL`, `AGENT_SERVICE_ADDR`, `AGENT_SERVICE_DATA_DIR`, `AGENT_SERVICE_TRACE`, `AGENT_SERVICE_DURABILITY`, …) and the one-command Restate E2E recipe.

`examples/agent-workbench` adds durable background work: Lashlang background processes, subagents, web tools, `ui.button.pressed` host events, and Restate-backed cron triggers. Restate is required — the bundled entrypoint starts it in Docker, registers the in-process endpoint, and opens the browser.

```bash
OPENROUTER_API_KEY=sk-or-... just agent-workbench
```

Then open <http://127.0.0.1:3030>. See [`examples/agent-workbench/README.md`](examples/agent-workbench/README.md) for the trigger sources, cron sync, and the full environment list.

## The CLI

`lash-cli` is a first-party terminal frontend on top of the library — coding-agent affordances (patch-based editing, shell execution, file search, web search, planning, skills, host-backed subagents, session resume / retry, model-native variants, live token accounting). It's not the product, but it's a fully featured way to drive the runtime from a terminal and a useful reference for end-to-end integration.

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

That default matrix covers standard mode, RLM, RLM tool batches, large tool surfaces, observational-memory prompt and maintenance paths, embed paths, streaming, scoped effect-controller turns, store reopen, and sans-IO turn-checkpoint round trips. The nightly / manual `Performance` workflow runs the full runtime profile:

```bash
python3 scripts/profile_runtime.py --profile full --release --cargo-feature fff-zlob --out .benchmarks/runtime-perf/full.json
```

For focused runtime regressions, the guard runner combines the normal runtime profile, stack-sensitivity matrix, and optional DHAT heap attribution:

```bash
python3 scripts/profile_runtime_guard.py --profile quick --release --cargo-feature fff-zlob --out .benchmarks/runtime-guard/guard.json
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
