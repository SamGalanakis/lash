# OpenRouter Chat UI

SQLite-backed localhost chat example for `lash-embed`.

The app owns the web server, browser protocol, chat list, message table, and
SQLite files. Lash owns turn execution through `LashCore` and per-chat runtime
state through a `SessionStoreFactory`.

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
