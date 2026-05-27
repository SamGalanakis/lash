# Agent Workbench

A standalone RLM chat demo for background processes, subagents, web tools, and
a host-owned event.

Run it from the repo root:

```sh
cargo run -p agent-workbench
```

Configuration is read from `.env` or the process environment:

- `OPENROUTER_API_KEY`: model provider key.
- `TAVILY_API_KEY`: Tavily key for `search_web` and `fetch_url`, matching the
  CLI web tools.
- `AGENT_WORKBENCH_ADDR`: bind address, default `127.0.0.1:3030`.
- `AGENT_WORKBENCH_DATA_DIR`: persistence directory, default
  `.agent-workbench`.
- `OPENROUTER_MODEL`: default `openai/gpt-5.5`.
- `OPENROUTER_MODEL_VARIANT`: default `medium`.

The browser UI has three work areas: the left rail contains red and blue host
event buttons plus per-turn model controls, the center pane is a chat/event
stream, and the right rail polls the process registry for visible background
work. The two buttons emit the declared host event `TRIGGER.button.pressed`.
The agent installs behavior by writing a declaration-only Lashlang module with
a `trigger` on that event; started Lashlang background processes appear in the
right rail.
