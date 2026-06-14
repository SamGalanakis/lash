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

For the old attached process style, use `just agent-workbench-foreground 3000`.

The entrypoint checks for Restate ingress/admin on the configured ports. If
they are not already running, it starts `restatedev/restate:1.6.2` in Docker,
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
  default `restatedev/restate:1.6.2`.
- `AGENT_WORKBENCH_RESTATE_CONTAINER`: Restate Docker container name for the
  entrypoint, default `lash-agent-workbench-dev-restate`.
- `AGENT_WORKBENCH_TOKIO_STACK_BYTES`: Tokio worker thread stack for the
  workbench process, default `2097152`. Override only when diagnosing stack
  regressions or comparing runtime stack-size lanes.
- `OPENROUTER_MODEL`: default `anthropic/claude-sonnet-4.6`.
- `OPENROUTER_MODEL_VARIANT`: default `high`; choose `provider default` in
  the UI to send no variant for models without configurable thinking.

Open the workbench at `http://127.0.0.1:3030` by default, or at the port passed
to the `just` recipe. Restate ingress is
`http://127.0.0.1:8080`; the local Restate admin/UI is on
`http://127.0.0.1:9070`.

The browser UI has three work areas: the left rail contains red and blue host
event buttons, a cron schedule card, and per-turn model controls; the center
pane is a chat/event stream; and the right rail polls the process registry for
visible background work. A **chat / accounts** tab switch at the top of the
center pane opens a dedicated mock-email view (see below). The buttons emit
`ui.button.pressed`. The cron card is
backed by Restate: ask the agent to schedule something and it can construct a
typed `cron.Schedule` source whose registrations sync to Restate virtual
objects. Started Lashlang background processes appear in the right rail.
The **stop turn** button (or **Esc**) cancels the running turn for real:
`POST /api/turn/cancel` calls `LashSession::cancel_running_turns()` on the
live session handle inside the Restate turn workflow, the runtime aborts the
in-flight provider call, and the turn commits as
`TurnOutcome::Stopped(Cancelled)` — the transcript shows `turn cancelled`.
The Lashlang graph panel is backed by `TraceLashlangGraphStore`, a public
trace-derived observation store for foreground blocks, durable process runs,
and child execution links; command operations still go through the session's
`SessionProcessAdmin` facade.

The **accounts** tab is a mocked multi-account inbox world you control live.
Type a name (for example `Work`) and press **add account** to connect one;
**delete** disconnects it. Each account card has a compose form that delivers a
message into its inbox and shows that inbox inline, with a per-message delete.
Each account is projected into the RLM Lashlang host environment as a typed module
authority of type `Inbox` at `inbox.<slug>`, exposing three operations — a
message is just a title and text, with no recipient address:

```lashlang
await inbox.work.send({ title: "Standup", text: "Notes attached." })?
listed = await inbox.work.list({})?            // { account, messages: [{ id, title, text }] }
await inbox.work.delete({ id: listed.messages[0].id })?
```

Because every account shares the `Inbox` authority type, one account-parametric
process can be started against any account, which is the point of the
multi-account showcase:

```lashlang
process triage(box: Inbox) {
  items = await box.list({})?
  wake { kind: "triage", account: items.account, count: len(items.messages) }
  finish true
}

work = start triage(box: inbox.work)
personal = start triage(box: inbox.personal)
results = await { work: work, personal: personal }
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
button, the emission runs inside a Restate effect scope
(`WorkbenchMailReceivedWorkflow`) so any registered trigger starts a durable
process. Register an inbox concierge once and it fires on every delivery:

```lashlang
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
submit format("Inbox concierge registered as `{}`.", handle)
```

This gives the demo three kinds of trigger source — a UI button (synthetic
event), an inbound email (data event), and a cron schedule (clock) — all
activating durable processes through the same registry. The host declares the
`mail.received` source through the plugin's `lashlang_resources()` hook and a
matching `mail` trigger registration, exactly like the button.

The button source config is `{}`. Red/blue selection arrives in the event
payload:

```lashlang
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
submit format("Registered button watcher `{}`. Active matching registrations: {}.", handle, len(registrations))
```

The cron card is the schedule reference integration: there is no Lashlang
`schedule` syntax and no UI tick button. Restate owns the timer policy. The
workbench plugin declares the `cron.Schedule` source; Lashlang builds a
`cron.Schedule` value and registers it with the runtime trigger registry:

```lashlang
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
submit format("Registered daily digest `{}`. Active matching registrations: {}.", handle, len(registrations))
```

After a Restate-backed turn registers an enabled `cron.Schedule`, the workbench
syncs that source key to `WorkbenchCronJob/{session_id}:{source_key}`. The
virtual object stores the source request, the next execution timestamp, and the
Restate invocation id in Restate K/V state. Its `run` handler emits a validated
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

The host side declares those source constructors through the plugin's
`lashlang_resources()` hook. The button is a zero-config source whose event
payload is validated separately through the trigger registration:

```rust
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
    .expect("valid button event type")
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
        ["ui", "button", "pressed"],
        lashlang::TypeExpr::Object(vec![]),
        button_trigger_event_type(),
    )
    .expect("valid button trigger source");
    resources
}

reg.triggers().declare(
    TriggerEvent::new("Button", "ui.button", "pressed", button_trigger_event_type()),
)?;
```

For a model-free live smoke that starts Restate, registers the endpoint through
Admin, schedules `cron.Schedule({ expr: "*/2 * * * * *", tz: "UTC" })`, waits
for a tick, and checks the JSONL trace, run:

```bash
just agent-workbench-restate-e2e
```
