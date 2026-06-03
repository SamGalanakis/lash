# Agent Workbench

A standalone RLM chat demo for background processes, subagents, web tools,
button host events, and Restate-backed cron triggers.

Restate is required. Run it from the repo root with a local Restate server,
then register the workbench endpoint:

```bash
# Terminal 1: Restate
docker run --rm \
  --add-host=host.docker.internal:host-gateway \
  -p 8080:8080 -p 9070:9070 -p 9071:9071 \
  restatedev/restate:latest

# Terminal 2: Workbench
OPENROUTER_API_KEY=... \
AGENT_WORKBENCH_RESTATE_ADDR=127.0.0.1:9081 \
RESTATE_INGRESS_URL=http://127.0.0.1:8080 \
cargo run -p agent-workbench

# Terminal 3, after the workbench endpoint starts
restate deployments register http://host.docker.internal:9081
```

Configuration is read from `.env` or the process environment:

- `OPENROUTER_API_KEY`: model provider key.
- `TAVILY_API_KEY`: Tavily key for `web.search(...)` and `web.fetch(...)`, matching the
  CLI web tools.
- `AGENT_WORKBENCH_ADDR`: bind address, default `127.0.0.1:3030`.
- `AGENT_WORKBENCH_RESTATE_ADDR`: Restate endpoint bind address, default
  `127.0.0.1:9081`.
- `RESTATE_INGRESS_URL`: Restate ingress URL, default `http://127.0.0.1:8080`.
- `AGENT_WORKBENCH_DATA_DIR`: persistence directory, default
  `.agent-workbench`.
- `AGENT_WORKBENCH_TRACE`: JSONL trace path, default
  `.agent-workbench/trace.jsonl`.
- `AGENT_WORKBENCH_LASHLANG_EXECUTION_TRACE`: JSONL Lashlang execution graph
  trace path, default `.agent-workbench/lashlang-execution.jsonl`.
- `OPENROUTER_MODEL`: default `anthropic/claude-sonnet-4.6`.
- `OPENROUTER_MODEL_VARIANT`: default `high`; choose `provider default` in
  the UI to send no variant for models without configurable thinking.

Open the workbench at `http://127.0.0.1:3030`. Restate ingress is
`http://127.0.0.1:8080`; the local Restate admin/UI is on
`http://127.0.0.1:9070`.

The browser UI has three work areas: the left rail contains red and blue host
event buttons, a cron schedule card, and per-turn model controls; the center
pane is a chat/event stream; and the right rail polls the process registry for
visible background work. The buttons emit `ui.button.pressed`. The cron card is
backed by Restate: ask the agent to schedule something and it can construct a
typed `cron.Schedule` source whose registrations sync to Restate virtual
objects. Started Lashlang background processes appear in the right rail.
The Lashlang graph panel is backed by `TraceLashlangGraphStore`, a public
trace-derived observation store for foreground blocks, durable process runs,
and child execution links; command operations still go through the session's
`ProcessControl` facade.

The button source config is `{}`. Red/blue selection arrives in the event
payload:

```lash
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
submit { handle: handle, registrations: registrations }
```

The cron card is the schedule reference integration: there is no Lashlang
`schedule` syntax and no UI tick button. Restate owns the timer policy. The
workbench plugin declares the `cron.Schedule` source; Lashlang builds a
`cron.Schedule` value and registers it with the runtime trigger registry:

```lash
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
submit { handle: handle, registrations: await triggers.list({ target: daily_digest })? }
```

After a Restate-backed turn registers an enabled `cron.Schedule`, the workbench
syncs that trigger to `WorkbenchCronJob/{session_id}:{trigger_handle}`. The
virtual object stores the source request, the next execution timestamp, and the
Restate invocation id in Restate K/V state. Its `run` handler activates that
exact trigger handle with a validated `cron.Tick` payload:

```json
{ "fired_at": "2026-06-02T12:00:00Z" }
```

Then it schedules its next `run` call with a delayed Restate send. Resetting the
workbench cancels known cron objects for the old session. The trace JSONL files
include `agent_workbench.cron.restate.sync_upserted` and
`agent_workbench.cron.restate.run` events.

The host side declares those source constructors through the plugin's
`lashlang_resources()` hook. The button is a zero-config source whose event
payload is validated separately through the host-event registration:

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

fn workbench_lashlang_resources() -> lashlang::ResourceCatalog {
    let mut resources = lashlang::ResourceCatalog::new();
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

reg.host_events().declare(
    HostEvent::new("Button", "ui.button", "pressed", button_trigger_event_type()),
)?;
```

For a model-free live smoke that starts Restate, registers the endpoint through
Admin, schedules `cron.Schedule({ expr: "*/2 * * * * *", tz: "UTC" })`, waits
for a tick, and checks the JSONL trace, run:

```bash
just agent-workbench-restate-e2e
```
