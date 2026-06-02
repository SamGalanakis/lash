# Agent Workbench

A standalone RLM chat demo for background processes, subagents, web tools, and
a host-owned event.

Run it from the repo root:

```sh
cargo run -p agent-workbench
```

Configuration is read from `.env` or the process environment:

- `OPENROUTER_API_KEY`: model provider key.
- `TAVILY_API_KEY`: Tavily key for `web.search(...)` and `web.fetch(...)`, matching the
  CLI web tools.
- `AGENT_WORKBENCH_ADDR`: bind address, default `127.0.0.1:3030`.
- `AGENT_WORKBENCH_DATA_DIR`: persistence directory, default
  `.agent-workbench`.
- `OPENROUTER_MODEL`: default `anthropic/claude-sonnet-4.6`.
- `OPENROUTER_MODEL_VARIANT`: default `high`; choose `provider default` in
  the UI to send no variant for models without configurable thinking.

The browser UI has three work areas: the left rail contains red and blue host
event buttons, a cron schedule card, and per-turn model controls; the center
pane is a chat/event stream; and the right rail polls the process registry for
visible background work. The buttons emit `ui.button.pressed`. The cron card is
instructional: ask the agent to schedule something and it can construct a typed
`cron.Schedule` source for a host-owned timer. Started Lashlang background
processes appear in the right rail.
The process graph panel is backed by `TraceProcessGraphStore`, a public
trace-derived observation store; command operations still go through the
session's `ProcessControl` facade.

The button source config is `{}`. Red/blue selection arrives in the event
payload:

```lash
type ButtonPressed = { button: "Red" | "Blue", message: str, pressed_at: str }

process on_button(event: ButtonPressed) {
  wake { kind: "button_pressed", button: event.button, message: event.message }
  finish true
}

handle = await triggers.register({
  source: ui.button.pressed({}),
  target: on_button,
  name: "button watcher"
})?
submit handle
```

The cron card is the schedule demo: there is no Lashlang `schedule` syntax and
no UI tick button. The host owns any cron/timer policy. The workbench plugin
declares the `cron.Schedule` source; core provides the generic trigger registry
plus source-type activation for host-owned schedulers. Exact-handle activation
remains available for schedulers that select specific due routes. Lashlang
builds a `cron.Schedule` value and registers it with the runtime trigger
registry:

```lash
process daily_digest(tick: cron.Tick) {
  wake { kind: "daily_digest_due", tick: tick }
  finish true
}

source = cron.Schedule({ expr: "0 8 * * *", tz: "UTC" })
handle = await triggers.register({
  source: source,
  target: daily_digest,
  name: "daily_digest"
})?
submit handle
```

The host side declares those source constructors through the plugin's
`lashlang_resources()` hook. The button is a zero-config source whose event
payload is validated separately through the host-event registration:

```rust
fn workbench_lashlang_resources() -> lashlang::ResourceCatalog {
    let mut resources = lashlang::ResourceCatalog::new();
    resources.add_trigger_source_constructor(
        ["cron", "Schedule"],
        schedule_config_type(),
        lashlang::TypeExpr::Ref("cron.Tick".into()),
    );
    resources.add_trigger_source_constructor(
        ["ui", "button", "pressed"],
        lashlang::TypeExpr::Object(vec![]),
        button_trigger_event_type(),
    );
    resources
}

reg.host_events().declare(
    HostEvent::new("Button", "ui.button", "pressed").payload(button_trigger_event_type()),
)?;
```
