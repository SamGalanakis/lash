# Agent Service

SQLite-backed localhost chat example for `lash-embed`, RLM mode, and typed
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
```

Then open `http://127.0.0.1:3000`.

The example opts into provider-level thinking exposure for demonstration,
attaches a trace sink to `LashCore`, and writes JSONL trace records to stderr
and `AGENT_SERVICE_TRACE` so provider payloads, RLM response, extracted
lashlang, terminal output, and tool calls are visible while you run it.

The app installs RLM explicitly with `ModePreset::rlm()`, registers
`DemoPlugin` on `LashCore`, activates it per chat session with
`SessionBuilder::use_plugin::<DemoPlugin>(...)`, and lets the plugin provide
its fixed app tools through the normal `ToolProvider` hook.

The plugin demonstrates:

- Typed session activation through `PluginBinding::SessionConfig`.
- Typed per-turn UI context through `DemoTurnContext`.
- A required tic-tac-toe board context validated before the turn runs.
- A plugin-authored `BoardTurnExt` trait so route code uses `.with_board(...)`
  instead of the generic `with_plugin_context::<DemoPlugin>(...)` primitive.
- `read_board` and `play_move` app tools provided by the plugin's
  `ToolProvider`. Their handlers read the same typed turn context used by the
  prompt hook.
- Prompt contribution that reflects the current board state.
- Additive semantic streaming: thinking is shown live from
  `TurnEvent::ReasoningDelta`, assistant prose as
  `TurnEvent::AssistantProseDelta`, code/tool activity as structured cards, and
  RLM `submit` as `TurnEvent::SubmittedValue`.
- Final persistence is app-owned: the stream sink accumulates assistant prose
  while rendering the same semantic activities live in the browser.

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
