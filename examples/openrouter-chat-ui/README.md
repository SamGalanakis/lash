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

The app installs RLM explicitly with `ModePreset::rlm()`, registers
`DemoPlugin` on `LashCore`, and activates it per chat session with
`SessionBuilder::use_plugin::<DemoPlugin>(...)`.

The plugin demonstrates:

- Typed session activation through `EmbedPlugin::SessionConfig`.
- Typed per-turn UI inputs through `DemoTurnInput`.
- A required `DemoPageContext` turn input validated by `TurnBuilder::run()`.
- An optional `Tone` value selected by the composer.
- A `demo_lookup` tool that reads the same typed turn context from
  `ToolExecutionContext`.
- Prompt contribution that reflects the selected tone and page context.

In lashlang, the model can call the demo tool with:

```lashlang
call demo_lookup { "query": "current page" }
```

The browser streams assistant prose from `TurnEvent::TextDelta`, renders tool
calls as compact cards, and persists only `TurnResult.final_text`.
