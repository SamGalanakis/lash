# Agent Workbench

A standalone RLM chat demo for background processes, subagents, web tools,
button triggers, and Restate-backed cron triggers.

Restate is required. Run the example from the repo root with the bundled
entrypoint. The default command starts the workbench as a detached local
service, waits for readiness, registers the Restate deployment, and then exits
after printing the URL:

```bash
OPENROUTER_API_KEY=... just agent-workbench 3000
```

Open `http://127.0.0.1:3000`. Useful lifecycle commands:

```bash
just agent-workbench-status 3000
just agent-workbench-logs 3000
just agent-workbench-logs-follow 3000
just agent-workbench-restart 3000
just agent-workbench-down 3000
```

`restart` replaces only the workbench web process and preserves the Restate
container and its retained invocations. `down` stops both the workbench and any
Restate container started by the entrypoint.

Validate the example build and unit tests:

```bash
cargo test -p agent-workbench --all-targets
```

For the old attached process style, use `just agent-workbench-foreground 3000`.

The entrypoint checks for Restate ingress/admin on the configured ports. If
they are not already running, it starts `restatedev/restate:1.7.0` in Docker,
waits for ingress/admin, starts the workbench and its in-process Restate
endpoint, registers the endpoint through Restate Admin, then opens the browser.
It writes PID, log, and run metadata under `.agent-workbench/run/`; stale PID
files are cleaned up automatically. Readiness is checked with
`/healthz`, so a random process on the same port is reported as a port conflict
instead of being mistaken for the workbench.

Configuration is read from `.env` or the process environment:

- `OPENROUTER_API_KEY`: model provider key.
- `TAVILY_API_KEY`: Tavily key for `web.search(...)` and `web.fetch(...)`, matching the
  CLI web tools.
- `AGENT_WORKBENCH_ADDR`: bind address, default `127.0.0.1:3030`. Passing a
  port to the `just` recipes, for example `just agent-workbench 3000`, binds
  `127.0.0.1:<port>`.
- `AGENT_WORKBENCH_RESTATE_ADDR`: Restate endpoint bind address, default
  `127.0.0.1:9081`. The `just agent-workbench` entrypoint starts Restate with
  host networking, so Restate can call this localhost endpoint directly.
- `RESTATE_INGRESS_URL`: Restate ingress URL, default `http://127.0.0.1:8080`.
- `AGENT_WORKBENCH_DATA_DIR`: persistence directory, default
  `.agent-workbench`.
- `AGENT_WORKBENCH_TRACE`: JSONL trace path, default
  `.agent-workbench/trace.jsonl`.
- `AGENT_WORKBENCH_LASHLANG_EXECUTION_TRACE`: JSONL Lashlang execution graph
  trace path, default `.agent-workbench/lashlang-execution.jsonl`.
- `RESTATE_ADMIN_URL`: Restate Admin URL used by the entrypoint for deployment
  registration, default `http://127.0.0.1:19070` so the dev runner does not
  collide with other local Restate/admin listeners.
- `AGENT_WORKBENCH_RESTATE_ADMIN_PORT`: host port for the Restate Admin
  container started by the entrypoint, default `19070`.
- `AGENT_WORKBENCH_RESTATE_NODE_PORT`: host port for the Restate node endpoint
  started by the entrypoint, default `19071`.
- `AGENT_WORKBENCH_RESTATE_ENDPOINT_URL`: URL Restate should use to reach the
  workbench endpoint, default `http://127.0.0.1:9081` for the host-networked
  Docker Restate container started by the entrypoint.
- `AGENT_WORKBENCH_OPEN`: set to `0` to skip opening the browser.
- `AGENT_WORKBENCH_RESTATE_IMAGE`: Restate Docker image for the entrypoint,
  default `restatedev/restate:1.7.0`.
- `AGENT_WORKBENCH_RESTATE_CONTAINER`: Restate Docker container name for the
  entrypoint, default `lash-agent-workbench-dev-restate`.
- `AGENT_WORKBENCH_TOKIO_STACK_BYTES`: Tokio worker thread stack for the
  workbench process, default `8388608`. Override only when diagnosing stack
  regressions or comparing runtime stack-size lanes.
- `AGENT_WORKBENCH_LEASE_HOST_ID`: identity of this workbench's PID namespace
  among every instance sharing the session store. Set it to a unique pod or
  container id when `/etc/machine-id` may be baked into the image; otherwise
  the workbench uses machine id, then hostname as a fallback.
- `OPENROUTER_MODEL`: default `anthropic/claude-sonnet-4.6`.
- `OPENROUTER_MODEL_VARIANT`: default `high`; choose `provider default` in
  the UI to send no variant for models without configurable thinking.

Open the workbench at `http://127.0.0.1:3030` by default, or at the port passed
to the `just` recipe. Restate ingress is
`http://127.0.0.1:8080`; the local Restate admin/UI is on
`http://127.0.0.1:19070`.

The browser UI has three work areas: the left rail contains red and blue trigger
buttons, a cron schedule card, and per-turn model controls; the center
pane is a chat/event stream; and the right rail polls the process registry for
visible background work. A **chat / accounts** tab switch at the top of the
center pane opens a dedicated mock-email view (see below). The buttons emit
`ui.button.pressed` trigger occurrences. The cron card is
backed by Restate: ask the agent to schedule something and it can construct a
typed `cron.Schedule` source whose registrations sync to Restate virtual
objects. Started Lashlang background processes appear in the right rail.
The **stop turn** button (or **Esc**) cooperatively cancels the exact running
turn: `POST /api/turn/cancel` sends its stable session and turn address through
`TurnWorkDriver::request_cancel`. The request lives on Lash's durable
keyed-promise seam, so it survives a workbench web-process restart and is
observed by the current or replayed Restate owner. The authoritative terminal
result is `TurnStop::Cancelled` with the original request id, opaque
host-defined origin, and optional reason; the UI clears only after the request
is accepted or the turn has already won the completion race.

Cancellation is cooperative: detached effects and non-cooperative external
work are not guaranteed to stop. Session and turn ids are routing identity,
not authorization; a production host must authorize the caller before
forwarding a stop request.
The Lashlang graph panel is backed by `TraceLashlangGraphStore`, a public
trace-derived observation store for foreground blocks, durable process runs,
and child execution links; command operations still go through the session's
`SessionProcessAdmin` facade.

The browser stream is deliberately split the same way a production host would
split it: product rows such as user messages, assistant messages, trigger
notes, errors, and `done` come from the workbench app stream, while Lash turn
activity comes from `session.observe().current_observation().cursor` plus
`ObservableSession::subscribe_and_recover_remote(...)`. `/api/events` accepts
the last cursor as `?cursor=...`, emits `replay_cursor`, forwards
`RemoteSessionObservationEvent` rows as `observation`, and reports missed
bounded-replay windows as `replay_gap` with `RemoteLiveReplayGap` plus a fresh
remote observation snapshot. Resetting the workbench rotates the session id and
restarts the browser stream from a fresh cursor.

The **accounts** tab is a mocked multi-account inbox world you control live.
Type a name (for example `Work`) and press **add account** to connect one;
**delete** disconnects it. Each account card has a compose form that delivers a
message into its inbox and shows that inbox inline, with a per-message delete.
Each account is projected into the RLM Lashlang host environment as a typed module
authority of type `Inbox` at `inbox.<slug>`, exposing three operations — a
message is just a title and text, with no recipient address:

```text
<lashlang>
await inbox.work.send({ title: "Standup", text: "Notes attached." })?
listed = await inbox.work.list({})?            // { account, messages: [{ id, title, text }] }
await inbox.work.delete({ id: listed.messages[0].id })?
</lashlang>
```

Because every account shares the `Inbox` authority type, one account-parametric
process can be started against any account, which is the point of the
multi-account showcase:

```text
<lashlang>
process triage(box: Inbox) {
  items = await box.list({})?
  wake { kind: "triage", account: items.account, count: len(items.messages) }
  finish true
}

work = start triage(box: inbox.work)
personal = start triage(box: inbox.personal)
results = await { work: work, personal: personal }
</lashlang>
```

Adding or removing an account enqueues a durable tool-catalog refresh that a
Restate workflow drains and commits — nothing executes in the HTTP handler.
The next opened turn picks up the new `inbox.<slug>` authority automatically.
Inbox tools resolve by parsing the tool name rather than scanning live
accounts, so a session persisted with a since-removed account's tools still
reopens cleanly; the refresh then drops the stale entries, and executing one
fails with the world's unknown-account error.

Delivering a message is the third trigger
source in the demo: the host appends it to the inbox and emits `mail.received`
with payload `mail.Received { account: str, title: str, text: str }`. Like the
button, the emission runs inside a Restate execution scope
(`WorkbenchMailReceivedWorkflow`) so any registered trigger starts a durable
process. Register an inbox concierge once and it fires on every delivery:

```text
<lashlang>
process on_mail(event: mail.Received) {
  work = start inbox.work.list({})
  personal = start inbox.personal.list({})
  inboxes = await { work: work, personal: personal }
  wake { kind: "mail_brief", arrived_in: event.account, title: event.title }
  finish true
}

handle = await triggers.register({
  source: mail.received({}),
  target: on_mail,
  inputs: { event: trigger.event },
  name: "inbox concierge"
})?
finish format("Inbox concierge registered as `{}`.", handle)
</lashlang>
```

This gives the demo three kinds of trigger source — a UI button trigger
occurrence, an inbound email data occurrence, and a cron schedule tick — all
activating durable processes through the same registry. Source constructors such
as `cron.Schedule` and `mail.received` live in the plugin's
`lashlang_resources()` hook. The button source is zero-config and exposed from
its trigger declaration.

The button source config is `{}`. Red/blue selection arrives in the event
payload:

```text
<lashlang>
process on_button(event: ui.button.Pressed) {
  wake { kind: "button_pressed", button: event.button, message: event.message }
  finish true
}

handle = await triggers.register({
  source: ui.button.pressed({}),
  target: on_button,
  inputs: { event: trigger.event },
  name: "button watcher"
})?
registrations = await triggers.list({ name: "button watcher" })?
finish format("Registered button watcher `{}`. Active matching registrations: {}.", handle, len(registrations))
</lashlang>
```

The cron card is the schedule reference integration: there is no Lashlang
`schedule` syntax and no UI tick button. In this example, Restate owns the timer
policy. The workbench plugin declares the `cron.Schedule` source; Lashlang builds a
`cron.Schedule` value and registers it with the runtime trigger registry:

```text
<lashlang>
process daily_digest(tick: cron.Tick) {
  wake { kind: "daily_digest_due", tick: tick }
  finish true
}

source = cron.Schedule({ expr: "0 8 * * *", tz: "UTC" })
handle = await triggers.register({
  source: source,
  target: daily_digest,
  inputs: { tick: trigger.event },
  name: "daily_digest"
})?
registrations = await triggers.list({ target: daily_digest })?
finish format("Registered daily digest `{}`. Active matching registrations: {}.", handle, len(registrations))
</lashlang>
```

After a Restate-backed turn registers an enabled `cron.Schedule`, the workbench
syncs that source key to `WorkbenchCronJob/{session_id}:{source_key}`. The
virtual object stores the source request and next execution timestamp in
Restate K/V state. Its `run` handler emits a validated
`cron.Tick` trigger occurrence for that stored source key:

```json
{ "fired_at": "2026-06-02T12:00:00Z" }
```

The occurrence idempotency key includes the journaled fire time, so it is
unique per tick and stable across retries of the same tick. The `run` handler
re-arms the next delayed `run` *before* emitting, so a tick that fails cannot
kill the schedule, and a job whose session is no longer the live workbench
session terminates itself on its next fire. Resetting the workbench cancels
the old session's cron jobs derived from its durable trigger registrations
(plus anything armed in-process), clears the mocked mail world, and rotates
the session; an equal-request re-sync revives a chain whose stored next
execution is already in the past. The trace JSONL files include
`agent_workbench.cron.restate.sync_upserted`,
`agent_workbench.cron.restate.run`, and
`agent_workbench.cron.restate.zombie_cancelled` events.

Host wiring has two pieces: source constructors such as `cron.Schedule` and
`mail.received` are declared through the plugin's `lashlang_resources()` hook,
while the button is a zero-config source exposed by its trigger declaration. The
button payload is validated by that registration:

```rust
fn schedule_config_type() -> lashlang::TypeExpr {
    lashlang::TypeExpr::Object(vec![
        lashlang::TypeField {
            name: "expr".into(),
            ty: lashlang::TypeExpr::Str,
            optional: false,
        },
        lashlang::TypeField {
            name: "tz".into(),
            ty: lashlang::TypeExpr::Str,
            optional: true,
        },
    ])
}

fn cron_tick_event_type() -> lashlang::NamedDataType {
    lashlang::NamedDataType::object(
        "cron.Tick",
        vec![lashlang::TypeField {
            name: "fired_at".into(),
            ty: lashlang::TypeExpr::Str,
            optional: false,
        }],
    )
    .expect("valid cron tick type")
}

fn button_trigger_event_type() -> lashlang::NamedDataType {
    lashlang::NamedDataType::object(
        "ui.button.Pressed",
        vec![
            lashlang::TypeField {
                name: "button".into(),
                ty: lashlang::TypeExpr::Union(vec![
                    lashlang::TypeExpr::Enum(vec!["Red".into()]),
                    lashlang::TypeExpr::Enum(vec!["Blue".into()]),
                ]),
                optional: false,
            },
            lashlang::TypeField {
                name: "message".into(),
                ty: lashlang::TypeExpr::Str,
                optional: false,
            },
            lashlang::TypeField {
                name: "pressed_at".into(),
                ty: lashlang::TypeExpr::Str,
                optional: false,
            },
        ],
    )
    .expect("valid button trigger event type")
}

fn mail_received_event_type() -> lashlang::NamedDataType {
    lashlang::NamedDataType::object(
        "mail.Received",
        vec![
            lashlang::TypeField {
                name: "account".into(),
                ty: lashlang::TypeExpr::Str,
                optional: false,
            },
            lashlang::TypeField {
                name: "title".into(),
                ty: lashlang::TypeExpr::Str,
                optional: false,
            },
            lashlang::TypeField {
                name: "text".into(),
                ty: lashlang::TypeExpr::Str,
                optional: false,
            },
        ],
    )
    .expect("valid mail received event type")
}

fn workbench_lashlang_resources() -> lashlang::LashlangHostCatalog {
    let mut resources = lashlang::LashlangHostCatalog::new();
    resources.add_trigger_source_constructor(
        ["cron", "Schedule"],
        schedule_config_type(),
        cron_tick_event_type(),
    )
    .expect("valid cron trigger source");
    resources.add_trigger_source_constructor(
        ["mail", "received"],
        lashlang::TypeExpr::Object(vec![]),
        mail_received_event_type(),
    )
    .expect("valid mail trigger source");
    resources
}

reg.triggers().declare(
    TriggerEvent::new("Button", "ui.button", "pressed", button_trigger_event_type()),
)?;
```

For a model-free live smoke that starts Restate, registers the endpoint through
Admin, submits a deterministic turn through Restate `/send`, verifies the
returned invocation completed successfully, schedules
`cron.Schedule({ expr: "*/2 * * * * *", tz: "UTC" })`, waits for a tick, checks
the JSONL trace, and asserts no workbench/process invocation remains active,
run:

```bash
just agent-workbench-restate-e2e
```
