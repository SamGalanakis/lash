# RLM Plugin Chat UI

SQLite-backed localhost chat example for `lash-embed`, RLM mode, and typed
session plugins.

Run it:

```bash
OPENROUTER_API_KEY=... cargo run -p openrouter-chat-ui
```

Optional environment:

```bash
OPENROUTER_MODEL=anthropic/claude-sonnet-4.6
OPENROUTER_CHAT_ADDR=127.0.0.1:3000
OPENROUTER_CHAT_DATA_DIR=.openrouter-chat-ui
```

Then open `http://127.0.0.1:3000`.

The example also attaches a trace sink to `LashCore` and writes JSONL trace
records to stderr so the RLM response, extracted lashlang, terminal output, and
tool calls are visible while you run it.

The app installs RLM explicitly with `ModePreset::rlm()`, registers
`DemoPlugin` on `LashCore`, and activates it per chat session with
`SessionBuilder::use_plugin::<DemoPlugin>(...)`.

The plugin demonstrates:

- Typed session activation through `EmbedPlugin::SessionConfig`.
- Typed per-turn UI inputs through `DemoTurnInput`.
- A required tic-tac-toe board input validated before the turn runs.
- `read_board` and `play_move` tools that read the same typed turn context from
  `ToolExecutionContext`.
- Prompt contribution that reflects the current board state.
- Additive semantic streaming: assistant prose is streamed as
  `TurnEvent::AssistantProseDelta`, code/tool activity is rendered as cards, and
  RLM `submit` is streamed as `TurnEvent::TerminalOutput`.
- Final persistence uses `TurnCollector::rendered_output()` so prose and typed
  terminal output are stored exactly through the same visible-output path.

This example opts into `RlmTermination::SubmitRequired { schema: None }`, so the
assistant's final user-facing text should be placed in `submit`. RLM also
supports `RlmTermination::ProseOrSubmit` for turns where direct prose may finish
without a lashlang block. `TerminalOutput` appears only when a turn finishes
through `submit` or a tool terminal control; prose-only completion is already
visible through assistant prose deltas.

In lashlang, the model can call the demo tool with:

```lashlang
board = (call read_board {})?
move = (call play_move { cell: 4 })?
submit "I played the center."
```

The browser also listens for `terminal_output` stream events and renders their
JSON-shaped value with the same display rule as Lash: strings pass through,
`null` is empty, and other values pretty-print.
