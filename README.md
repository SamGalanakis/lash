# lash

AI coding agent with persistent Python REPL and hashline editing.

## Setup

```
cat > ~/.local/bin/lash << 'SCRIPT'
#!/bin/bash
cargo build --release --manifest-path ~/code/lash/Cargo.toml 2>/dev/null
exec ~/code/lash/target/release/lash "$@"
SCRIPT
chmod +x ~/.local/bin/lash
```

## Usage

```
lash                        # start new session
lash "fix the login bug"    # start with an initial prompt
lash --resume               # resume last session
```

| Command | Description |
|---|---|
| `/clear`, `/new` | Reset conversation |
| `/model <name>` | Switch LLM model |
| `/provider` | Change provider |
| `/resume [name]` | Browse/load previous sessions |
| `/skills` | Browse loaded skills |
| `/help` | Show help |
| `!command` | Run shell command directly |

| Key | Action |
|---|---|
| `Shift+Tab` | Toggle plan mode |
| `Esc` | Cancel agent / dismiss prompt |
| `Ctrl+U` / `Ctrl+D` | Scroll half-page |
| `PgUp` / `PgDn` | Scroll full page |
| `Shift+Enter` | Insert newline |
| `Ctrl+V` | Paste image |
| `Ctrl+Y` | Copy last response |


All rights reserved.
