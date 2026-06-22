# Agent Service

SQLite-backed localhost chat example for `lash`, RLM mode, typed session
plugin activation, app-owned board tools, semantic streaming, and optional
Restate-backed turns.

Run it:

```bash
OPENROUTER_API_KEY=... cargo run -p agent-service
```

Optional environment:

```bash
OPENROUTER_MODEL=anthropic/claude-sonnet-4.6
OPENROUTER_MODEL_VARIANT=high
AGENT_SERVICE_ADDR=127.0.0.1:3000
AGENT_SERVICE_DATA_DIR=.agent-service
AGENT_SERVICE_TRACE=.agent-service/trace.jsonl
AGENT_SERVICE_DURABILITY=local
```

The durability mode can also be passed as `--durability local`. Restate mode is
feature-gated and uses these local defaults:

| Path | App | Restate endpoint | Ingress | Admin |
| --- | --- | --- | --- | --- |
| App run | `127.0.0.1:3000` | `127.0.0.1:9080` | `127.0.0.1:8080` | `127.0.0.1:9070` |
| Live E2E | in-process test | `127.0.0.1:19080` | `127.0.0.1:18080` | `127.0.0.1:19070` |

For the app run, start Restate on the host network, run the feature-gated binary,
then register the endpoint:

```bash
docker run --rm --network host restatedev/restate:1.7.0
OPENROUTER_API_KEY=... \
AGENT_SERVICE_DURABILITY=restate \
AGENT_SERVICE_RESTATE_ADDR=127.0.0.1:9080 \
RESTATE_INGRESS_URL=http://127.0.0.1:8080 \
cargo run -p agent-service --features restate -- --durability restate

restate deployments register http://127.0.0.1:9080
```

If you run Restate in Docker bridge mode instead, bind the app endpoint to a
container-reachable interface such as `AGENT_SERVICE_RESTATE_ADDR=0.0.0.0:9080`
and register `http://host.docker.internal:9080` (or add
`--add-host=host.docker.internal:host-gateway` on Linux).

For the live E2E, use the one-command recipe. It starts the agent-service
Restate endpoint in-process, registers it through the Restate Admin API, submits
a turn through Restate ingress, runs a named Lashlang background process against
the tic-tac-toe board through `LashProcessWorkflow`, verifies app
outbox/message persistence, and removes the container on exit:

```bash
just agent-service-restate-e2e
```

The recipe starts `restatedev/restate:1.7.0` with host networking on admin
`19070`, ingress `18080`, and node `15122`. Override
`AGENT_SERVICE_RESTATE_IMAGE`, `AGENT_SERVICE_RESTATE_CONTAINER`,
`RESTATE_ADMIN_PORT`, `RESTATE_INGRESS_PORT`, `RESTATE_NODE_PORT`,
`AGENT_SERVICE_E2E_ENDPOINT_BIND`, or `AGENT_SERVICE_E2E_ENDPOINT_URL` if your
local Docker networking needs different addresses.

In Restate mode the Axum app still serves `AGENT_SERVICE_ADDR`, the same process
also serves a Restate endpoint on `AGENT_SERVICE_RESTATE_ADDR`, and browser
turns submit the app-specific `AgentServiceTurnWorkflow/{turn_id}/run/send`
through `RESTATE_INGRESS_URL`. The endpoint also binds Lash's generic
`LashProcessWorkflow`, backed by `RestateCoreProcessRunner` and the same
deployment-level `processes.db`, so background process starts from a turn are
reconstructed from SQLite session stores instead of running in the route
process. `AgentServiceTurnWorkflowRequest` carries only stable turn, chat, text,
model, and model-variant data; board state stays in the app database. The
workflow creates a `RestateRuntimeEffectController` and calls
`session.turn(...).turn_id(...).effects(&controller).stream_to(...)`.
The stable chat/session id and turn id keep Restate replay and Lash final
commit addressed to the same operation.
Turn progress is written to an app-owned
SQLite outbox keyed by `turn_id`, so the NDJSON route can stream progress after
route restart or while the workflow is running in the Restate handler.
Semantic progress is sourced from the Lash session observation API rather than
from a route-local side channel: the route captures
`session.observe().current_observation().cursor` before the turn, emits that
opaque cursor as `replay_cursor`, and forwards live
`RemoteSessionObservationEvent` values from
`ObservableSession::subscribe_and_recover(...)` as `observation` NDJSON rows
until the session commit is observed. If the cursor is no longer in the bounded
replay window, the stream emits `replay_gap` with the latest cursor so a
reconnecting client can replace from a fresh observation. Restate mode uses the
same live replay path for turn activity and keeps the app outbox focused on
durable product rows such as user messages, assistant messages, errors, and
`done`.

Then open `http://127.0.0.1:3000`.

The model and reasoning variant are also editable in the browser. The
environment values are just the defaults for new chats; each chat persists its
own OpenRouter model id and variant, and each turn applies that selection with
the public `TurnBuilder::model(...)` API.

The example opts into provider-level thinking exposure for demonstration,
attaches a trace sink to `LashCore`, and writes JSONL trace records to stderr
and `AGENT_SERVICE_TRACE` so provider payloads, RLM response, extracted
lashlang, terminal output, and tool calls are visible while you run it.

The app builds an `RlmCore`, activates `DemoPlugin` per chat session with
`SessionBuilder::plugin::<DemoPlugin>(...)`, and lets the plugin provide
its fixed app tools through the normal `ToolProvider` hook.

The plugin demonstrates:

- Typed session activation through `PluginBinding::SessionConfig`.
- App-owned tic-tac-toe board state in the `chat_boards` SQLite table.
- User message payload board snapshots for browser replay only.
- `read_board` and `play_move` app tools provided by the plugin's
  `ToolProvider`. `read_board` reads the canonical board from SQLite;
  `play_move` validates and mutates that canonical board.
- Prompt contribution that reflects the current canonical board state.
- Additive semantic streaming: thinking is shown live from
  `TurnEvent::ReasoningDelta`, assistant prose as
  `TurnEvent::AssistantProseDelta`, code/tool activity as structured cards, and
  RLM `submit` as `TurnEvent::SubmittedValue`.
- Runtime persistence is handled by `SqliteSessionStoreFactory`; each request
  opens the Lash session from the chat id and store instead of keeping runtime
  sessions in a process-global map.
- Product persistence is app-owned: chat rows, board snapshots, reasoning, code
  blocks, tool cards, tool outbox events, and titles stay in the app database.
  The final assistant row is derived from `TurnOutput` terminal semantics,
  preferring `submit` / tool terminal values over streamed prose.
- The app database uses `rusqlite` on `tokio::task::spawn_blocking`, keeping the
  example dependency-light without blocking Axum worker tasks on SQLite calls.

This example opts into `.require_submit()`, so the assistant's final user-facing
text should be placed in `submit`. RLM also supports `.allow_prose_or_submit()`
for turns where direct prose may finish without a lashlang block.
`SubmittedValue` appears when a turn finishes through `submit`; `ToolValue`
appears when a tool terminal control finishes the turn. Prose-only completion is
already visible through assistant prose deltas.

In lashlang, the model calls the demo tools through their host-declared module
surface. `DemoPlugin` maps the underlying `read_board` tool to `board.read`
and `play_move` to `board.play` with `LashlangToolBinding`:

```text
<lashlang>
board = await board.read({})?
move = await board.play({ cell: 4 })?
submit "I played the center."
</lashlang>
```

The browser also listens for submitted/tool value stream events and renders
their JSON-shaped value with the same display rule as Lash: strings pass
through, `null` is empty, and other values pretty-print.
