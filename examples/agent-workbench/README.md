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
event buttons plus per-turn model controls, the center pane is a chat/event
stream, and the right rail polls the process registry for visible background
work. The two buttons emit the declared host event `ui.button.pressed`.
The agent installs behavior by declaring a Lashlang `trigger` on that event;
started Lashlang background processes appear in the right rail.
