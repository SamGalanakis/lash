# Agent Service

SQLite-backed localhost chat example for `lash`, RLM mode, and typed
session plugins.

Run it:

```bash
OPENROUTER_API_KEY=... cargo run -p agent-service
```

Optional environment:

```bash
OPENROUTER_MODEL=openai/gpt-5.5
OPENROUTER_MODEL_VARIANT=medium
AGENT_SERVICE_ADDR=127.0.0.1:3000
AGENT_SERVICE_DATA_DIR=.agent-service
AGENT_SERVICE_TRACE=.agent-service/trace.jsonl
AGENT_SERVICE_DURABILITY=local
```

The durability mode can also be passed as `--durability local`.

Restate mode is feature-gated and runnable:

```bash
docker run --rm -p 8080:8080 -p 9070:9070 -p 9071:9071 restatedev/restate:latest
OPENROUTER_API_KEY=... \
AGENT_SERVICE_DURABILITY=restate \
AGENT_SERVICE_RESTATE_ADDR=127.0.0.1:9080 \
RESTATE_INGRESS_URL=http://127.0.0.1:8080 \
cargo run -p agent-service --features restate -- --durability restate

restate deployments register http://host.docker.internal:9080
```

In Restate mode the Axum app still serves `AGENT_SERVICE_ADDR`, the same process
also serves a Restate endpoint on `AGENT_SERVICE_RESTATE_ADDR`, and browser
turns submit the app-specific `AgentServiceTurnWorkflow/{turn_id}/run` through
`RESTATE_INGRESS_URL`. The workflow creates a
`RestateRuntimeEffectController`, first attempts
`session.resume_turn(turn_id).stream_with_effect_scope(...)`, and falls back to
a fresh `session.turn(...).stream_with_effect_scope(...)` only when Lash has no
checkpoint for that turn id yet. Turn progress is written to an app-owned
SQLite outbox keyed by `turn_id`, so the NDJSON route can stream progress after
route restart or while the workflow is running in the Restate handler.

Then open `http://127.0.0.1:3000`.

The model and reasoning variant are also editable in the browser. The
environment values are just the defaults for new chats; each chat persists its
own OpenRouter model id and variant, and each turn applies that selection with
the public `TurnBuilder::model(...)` API.

The example opts into provider-level thinking exposure for demonstration,
attaches a trace sink to `LashCore`, and writes JSONL trace records to stderr
and `AGENT_SERVICE_TRACE` so provider payloads, RLM response, extracted
lashlang, terminal output, and tool calls are visible while you run it.

The app installs RLM explicitly with `ModePreset::rlm()`, activates
`DemoPlugin` per chat session with
`SessionBuilder::plugin::<DemoPlugin>(...)`, and lets the plugin provide
its fixed app tools through the normal `ToolProvider` hook.

The plugin demonstrates:

- Typed session activation through `PluginBinding::SessionConfig`.
- Typed per-turn UI input through `DemoTurnInput`.
- A required tic-tac-toe board input validated before the turn runs.
- A plugin-authored `BoardTurnExt` trait so route code uses `.with_board(...)`
  instead of the generic `with_plugin_input::<DemoPlugin>(...)` primitive.
- `read_board` and `play_move` app tools provided by the plugin's
  `ToolProvider`. Their handlers read the same typed turn input used by the
  prompt hook.
- Prompt contribution that reflects the current board state.
- Additive semantic streaming: thinking is shown live from
  `TurnEvent::ReasoningDelta`, assistant prose as
  `TurnEvent::AssistantProseDelta`, code/tool activity as structured cards, and
  RLM `submit` as `TurnEvent::SubmittedValue`.
- Runtime persistence is handled by `SqliteSessionStoreFactory`; each request
  opens the Lash session from the chat id and store instead of keeping runtime
  sessions in a process-global map.
- Product persistence is app-owned: chat rows, board snapshots, reasoning, code
  blocks, tool cards, and titles stay in the app database. The final assistant
  row is derived from `TurnOutput` terminal semantics, preferring `submit` /
  tool terminal values over streamed prose.
- The app database uses `rusqlite` on `tokio::task::spawn_blocking`, keeping the
  example dependency-light without blocking Axum worker tasks on SQLite calls.

This example opts into `.require_submit()`, so the assistant's final user-facing
text should be placed in `submit`. RLM also supports `.allow_prose_or_submit()`
for turns where direct prose may finish without a lashlang block.
`SubmittedValue` appears when a turn finishes through `submit`; `ToolValue`
appears when a tool terminal control finishes the turn. Prose-only completion is
already visible through assistant prose deltas.

In lashlang, the model can call the demo tool with:

```lashlang
board = (call read_board {})?
move = (call play_move { cell: 4 })?
submit "I played the center."
```

The browser also listens for submitted/tool value stream events and renders
their JSON-shaped value with the same display rule as Lash: strings pass
through, `null` is empty, and other values pretty-print.
