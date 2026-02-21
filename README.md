# lash

AI coding agent with a persistent Python REPL, hashline-safe file editing, and a host-friendly runtime API for TUI and headless frontends.

## Setup

```bash
cat > ~/.local/bin/lash << 'SCRIPT'
#!/bin/bash
cargo build --release --manifest-path ~/code/lash/Cargo.toml 2>/dev/null
exec ~/code/lash/target/release/lash "$@"
SCRIPT
chmod +x ~/.local/bin/lash
```

## CLI Usage

```bash
lash                               # start interactive TUI session
lash "fix auth error handling"      # start with an initial prompt
lash --print "summarize this repo"  # headless mode: run one prompt and print result
lash --provider                    # force provider setup flow
lash --reset                       # delete ~/.lash and ~/.cache/lash, then exit
```

## Slash Commands

| Command | Description |
|---|---|
| `/clear`, `/new` | Reset conversation |
| `/controls` | Show keyboard shortcuts |
| `/model <name>` | Switch LLM model |
| `/provider` | Open provider setup in-app |
| `/login` | Alias for `/provider` |
| `/logout` | Remove stored credentials from disk |
| `/resume [name]` | Browse/load previous sessions |
| `/skills` | Browse loaded skills |
| `/help`, `/?` | Show help |
| `/exit`, `/quit` | Quit |

`/logout` only clears persisted config. The active session may continue with in-memory credentials until you switch provider or restart.

## Keyboard Shortcuts

| Key | Action |
|---|---|
| `Esc` | Cancel running agent / dismiss prompt |
| `Shift+Enter` | Insert newline |
| `Ctrl+U` / `Ctrl+D` | Scroll half-page up / down |
| `PgUp` / `PgDn` | Scroll full page |
| `Ctrl+V` | Paste image as inline `[Image #n]` |
| `Ctrl+Shift+V` | Paste text only |
| `Ctrl+Y` | Copy last response |
| `Ctrl+O` | Cycle tool expansion level |
| `Ctrl+Shift+O` | Full expansion (code + stdout) |

## Provider Defaults

Default root model by provider:

| Provider | Default model |
|---|---|
| `OpenRouter` | `anthropic/claude-sonnet-4.6` |
| `Claude` | `claude-sonnet-4-6` |
| `Codex` | `gpt-5.3-codex` |
| `Google OAuth` | `gemini-3.1-pro-preview` |

Default sub-agent tier mapping (`quick` / `balanced` / `thorough`):

| Provider | quick | balanced | thorough |
|---|---|---|---|
| `OpenRouter` | `minimax/minimax-m2.5` | `z-ai/glm-5` | `anthropic/claude-sonnet-4.6` |
| `Claude` | `claude-haiku-4-5` | `claude-sonnet-4-6` | `claude-sonnet-4-6` |
| `Codex` | `gpt-5.3-codex-spark` | `gpt-5.3-codex` (`reasoning=medium`) | `gpt-5.3-codex` (`reasoning=high`) |
| `Google OAuth` | `gemini-3-flash-preview` | `gemini-3.1-pro-preview` | `gemini-3.1-pro-preview` |

## Frontend / Runtime Integration

`lash-core` exposes a unified turn input surface for third-party frontends (`lash/src/runtime.rs`).

- `TurnInput.items` is an ordered list of typed items.
- Supported item kinds:
  - `text`
  - `file_ref`
  - `dir_ref`
  - `image_ref`
  - `skill_ref`
- `TurnInput.image_blobs` carries raw image bytes keyed by `image_ref.id`.

This preserves interleaving and intent, e.g. text -> image -> text -> file reference in a single turn.

### Example payload

```json
{
  "items": [
    {"type": "text", "text": "Please update this"},
    {"type": "image_ref", "id": "img-1"},
    {"type": "text", "text": "and check"},
    {"type": "file_ref", "path": "/repo/src/main.rs"}
  ],
  "image_blobs": {
    "img-1": "<binary png bytes>"
  }
}
```

## Tool Exposure Model

Tools define `inject_into_prompt: bool` (`lash/src/lib.rs`).

- `inject_into_prompt=true`: tool appears directly in the LLM system prompt.
- `inject_into_prompt=false`: tool is omitted from prompt for brevity, but still callable at runtime.

Runtime discovery is available via:

- `list_tools(...)`
- `find_tools(query, mode="hybrid", ...)`

The REPL exposes the full tool namespace under `T` (for example `T.read_file(...)`, `T.find_tools(...)`).

All rights reserved.
