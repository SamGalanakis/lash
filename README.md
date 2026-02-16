# lash

AI coding agent with persistent Python REPL and hashline editing.

## Overview

Lash uses a **Code-Act loop**: the LLM outputs Python code each turn, executed in a persistent REPL where variables survive between turns. Tools are async Python functions—file edits, shell commands, web search, sub-agent delegation—called with `await`. The agent communicates back via `respond()` (final answer) or `say()` (progress update).

## Architecture

Cargo workspace with two crates:

- **`lash`** — core runtime: agent loop, tool implementations, session persistence, provider abstraction
- **`lash-cli`** — terminal UI built with Ratatui: rendering, input handling, setup wizard, theme

## Setup

Install the launcher script (auto-builds from source on each run):

```
cat > ~/.local/bin/lash << 'SCRIPT'
#!/bin/bash
cargo build --release --manifest-path ~/code/lash/Cargo.toml 2>/dev/null
exec ~/code/lash/target/release/lash "$@"
SCRIPT
chmod +x ~/.local/bin/lash
```

On first run, `lash` launches an interactive setup wizard:

1. **Select provider** — Claude (OAuth), Codex (device code), or OpenRouter (API key)
2. **Authenticate** — browser-based OAuth, device code flow, or paste an API key
3. **Optional** — Tavily API key for web search

You can also pass `--api-key` or set `OPENROUTER_API_KEY` directly:

```
lash --api-key sk-or-...
```

Re-run setup anytime with `lash --provider`.

## Usage

```
lash                        # start new session
lash "fix the login bug"    # start with an initial prompt
lash --resume               # resume last session
```

### Commands

| Command | Description |
|---|---|
| `/clear`, `/new` | Reset conversation |
| `/model <name>` | Switch LLM model |
| `/provider` | Change provider |
| `/resume [name]` | Browse/load previous sessions |
| `/skills` | Browse loaded skills |
| `/help` | Show help |
| `!command` | Run shell command directly |

### Keyboard shortcuts

| Key | Action |
|---|---|
| `Shift+Tab` | Toggle plan mode |
| `Esc` | Cancel agent / dismiss prompt |
| `Ctrl+U` / `Ctrl+D` | Scroll half-page |
| `PgUp` / `PgDn` | Scroll full page |
| `Shift+Enter` | Insert newline |
| `Ctrl+V` | Paste image |
| `Ctrl+Y` | Copy last response |

### Plan mode

`Shift+Tab` enters plan mode, which restricts the agent to read-only tools and writes plans to `.lash_plan`. Use it to think through changes before executing them.

### Instructions

Lash loads project instructions automatically:

- **Global**: `~/.lash/AGENT.md`
- **Project**: `AGENTS.md` or `CLAUDE.md` found by walking up from the working directory

### Skills

Custom skills extend the agent with domain-specific prompts and tool configurations. Browse available skills with `/skills`.

## Tools

**File ops** — ReadFile, WriteFile, EditFile, DiffFile, FindReplace, Glob, Ls
**Execution** — Shell (streaming), Grep
**Delegation** — DelegateSearch (read-only), DelegateTask (full tools), DelegateDeep (no turn limit)
**Web** — WebSearch (Tavily), FetchUrl
**Project** — TaskStore (planning/tracking), SkillStore, ViewMessage

## Project structure

```
lash/
  baml_src/agent.baml      # Code-Act prompt & execution model
  src/
    agent.rs                # BAML agent loop
    provider.rs             # LLM provider abstraction
    session.rs              # Session & REPL management
    store.rs                # SQLite session store
    instructions.rs         # Instruction file loader
    embedded.rs             # Embedded Python runtime
    tools/                  # Tool implementations
      mod.rs                # Tool registry
      shell.rs, read_file.rs, edit_file.rs, ...

lash-cli/
  src/
    main.rs                 # Entry point, CLI args, event loop
    app.rs                  # App state machine
    ui.rs                   # Ratatui rendering
    setup.rs                # Interactive provider setup
    theme.rs                # Design system (colors, fonts)
    markdown.rs             # Markdown parsing & rendering
    command.rs              # Slash-command handling

docs/
  design.html               # Visual design reference
```

## Session persistence

Sessions are stored in `~/.lash/sessions/` as paired `.jsonl` (event log) and `.db` (SQLite) files. REPL state is snapshot via Dill serialization, enabling full session resume with variables intact.

## License

MIT
