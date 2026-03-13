# lash

AI coding agent with a persistent `repl` runtime or `standard` tool-calling execution mode, `apply_patch`-based file editing, and a host-friendly runtime API for TUI and headless frontends.

## Install

```bash
curl -fsSL https://github.com/SamGalanakis/lash/releases/latest/download/install_lash.sh | bash
```

Install a pinned release instead:

```bash
curl -fsSL https://github.com/SamGalanakis/lash/releases/download/v0.2.0/install_lash.sh | bash
```

Successful pushes to `main` automatically bump the release patch version when needed, create the next `v*` tag, and trigger the release workflow. That workflow mints a GitHub Release with platform tarballs, `SHA256SUMS`, and a version-pinned `install_lash.sh`. The installer downloads the matching prebuilt asset for your platform, verifies `SHA256SUMS`, and installs `lash` to `~/.local/bin` by default.

From a local checkout, run `./install_lash.sh`.

## Runtime Modes

`lash`/`lash-core` support two execution backends:

- `repl` (default): a persistent `lashlang` runtime that executes agent-written workflow code with host-call boundaries for tools.
- `standard`: provider-native tool calling without the REPL sandbox.

Both backends are driven by a single sans-IO state machine (`TurnMachine` in `lash/src/sansio.rs`). All protocol logic (prompt assembly, fence parsing, retry/backoff, context folding, turn limits) lives in the synchronous machine; the async `LashRuntime` host driver in `lash/src/runtime.rs` fulfils I/O effects. This makes the core turn logic independently testable without a tokio runtime.

### Local dev

```bash
./dev.sh
```

### Build commands

```bash
cargo build -p lash-cli
cargo build -p lash-cli --release
```

### Downstream (`lash-core`) integration

```toml
[dependencies]
lash-core = { path = "../lash", default-features = false, features = ["full"] }
```

## CLI Usage

```bash
lash                               # start interactive TUI session
lash -p "summarize this repo"      # headless mode: run one prompt and print result
lash --print "summarize this repo"  # same, long form
lash --model gpt-5.4               # override model
lash --execution-mode standard # use provider-native tool calling instead of the repl sandbox
lash --no-mouse                    # disable mouse (re-enables terminal text selection)
lash --provider                    # force provider setup flow
lash --reset                       # delete ~/.lash and ~/.cache/lash, then exit
```

## Prompt Customization

Override system prompt sections at launch:

```bash
lash --prompt-replace "section=replacement text"
lash --prompt-replace-file "section=path/to/file.md"
lash --prompt-prepend "section=prepended text"
lash --prompt-prepend-file "section=path/to/file.md"
lash --prompt-append "section=appended text"
lash --prompt-append-file "section=path/to/file.md"
lash --prompt-disable "section"
```

Project instructions are loaded through the prompt-context plugin. By default it loads global `AGENT.md` plus repo-local `AGENTS.md` / `CLAUDE.md`, and that file set is configurable in `lash-core` through `InstructionLoaderConfig`. The same plugin also owns the stable environment/project context block, so those sections stay in sync with runtime prompt assembly.

## Skills

Skills are modular directories that extend lash with specialized knowledge and workflows.

Skill directories: `~/.lash/skills/` (global) and `.agents/lash/skills/` (repo-local, overrides global). The legacy `.lash/skills/` path is still loaded as a fallback.

Each skill is a directory containing a `SKILL.md` with YAML frontmatter (`name`, `description`) and a markdown body. Supporting files (scripts, references, templates) are included alongside.

```
~/.lash/skills/
  my-skill/
    SKILL.md          # required: frontmatter + instructions
    scripts/foo.py    # optional supporting files
    references/bar.md

.agents/lash/skills/
  repo-local-skill/
    SKILL.md
```

Use `/skills` to browse, `/<skill-name>` to invoke, `call load_skill { name: "..." }` inside `repl`, or `load_skill(...)` in `standard`.

## Terminal Bench

Use `scripts/run-terminalbench.sh` to run Harbor + Terminal Bench 2 with the in-repo lash adapter.

By default it runs `terminal-bench-sample@2.0`, builds a benchmark binary in `target-bullseye/release/lash` using `rust:1-bullseye`, and uses your local `~/.lash/config.json` inside benchmark containers (so OAuth/OpenAI-generic config is reused). You must specify `--execution-mode repl` or `--execution-mode standard` explicitly.

The Harbor adapter also appends benchmark-specific guidelines to lash's `guidelines` prompt section for these runs. That overlay makes exact verifier compliance explicit: match required final state exactly, self-verify ports/files/processes before finishing, and do not undo the final working state after verification. Override it with `LASH_BENCH_PROMPT_APPEND_GUIDELINES` if needed.

For exact task batches, use `--tasks a,b,c` or `--task-file path.txt`. The runner will set Harbor concurrency to the full explicit task count unless you override `--n-concurrent`, and it prints a per-task summary from the job artifacts after the run.

```bash
scripts/run-terminalbench.sh --sample --execution-mode repl
scripts/run-terminalbench.sh --full --execution-mode standard --task "git-*"
scripts/run-terminalbench.sh --sample --execution-mode repl --task chess-best-move --model gpt-5.4
scripts/run-terminalbench.sh --sample --execution-mode standard --tasks regex-log,sqlite-with-gcov
scripts/run-terminalbench.sh --sample --execution-mode standard --task-file bench/tasks.txt
```

## Slash Commands

| Command | Description |
|---|---|
| `/clear`, `/new` | Reset conversation |
| `/controls` | Show keyboard shortcuts |
| `/model [name]` | Show current model or switch LLM model |
| `/mode [name]` | Show the current execution mode (`repl` or `standard`) for this session |
| `/provider` | Open provider setup in-app |
| `/login` | Alias for `/provider` |
| `/logout` | Remove stored credentials from disk |
| `/retry` | Replay the previous turn payload exactly |
| `/resume [name]` | Browse/load previous sessions |
| `/skills` | Browse loaded skills |
| `/tools` | Inspect or edit dynamic tools |
| `/caps` | Inspect or edit dynamic capabilities |
| `/reconfigure` | Apply or inspect pending runtime reconfigure |
| `/help`, `/?` | Show help |
| `/exit`, `/quit` | Quit |

`/logout` only clears persisted config. The active session may continue with in-memory credentials until you switch provider or restart.

`repl` is the default execution mode and is always available. `standard` uses provider-native tool calling with a concrete toolset selected at session start, so it does not preserve arbitrary REPL locals across turns.
Low-intelligence sub-agents spawned with `agent_call(..., intelligence="low")` always run in `standard`; medium/high sub-agents inherit the parent session's execution mode.

Context folding is batched and cache-friendly. Lash keeps the prompt stable until the hard watermark is hit, then folds old history back to the soft watermark with one stable archive marker instead of mutating prompt status every turn. Defaults are `50%` soft and `60%` hard. Override them with:

```bash
lash --context-fold-soft-pct 45 --context-fold-hard-pct 58
```

`batch` is standard-only. Use it for 2-25 independent tool calls when you already know the arguments up front; it runs those calls concurrently and returns a structured per-call result summary.

Across both `repl` and `standard`, lash now renders active history as one cache-friendly chronological transcript inside a single user prompt, paired with a stable system prompt. The internal turn state and structured tool/code records are still preserved for execution and folding, but provider requests no longer depend on replaying provider-specific conversational role sequences.

## Planning

Planning now has one durable checklist tool and one optional planning-only mode.

- `update_plan` is for substantial multi-step execution work.
- Keep plans short, concrete, and easy to verify.
- The TUI renders the current checklist directly from `update_plan` calls.
- Plan mode remains a separate plugin-owned surface for planning-only turns and proposed-plan rendering.

## Keyboard Shortcuts

| Key | Action |
|---|---|
| `Esc` | Cancel running agent / dismiss prompt |
| `Shift+Enter` | Insert newline |
| `Ctrl+U` / `Ctrl+D` | Scroll half-page up / down |
| `PgUp` / `PgDn` | Scroll full page |
| `Ctrl+V` | Paste image as inline `[Image #n]` |
| `Ctrl+Shift+V` | Paste text only |
| `Shift+Tab` | Toggle persistent plan mode |
| `Ctrl+Y` | Copy last response |
| `Ctrl+O` | Cycle tool expansion level |
| `Alt+O` | Full expansion (code + stdout) |

## Provider Defaults

Default root model by provider:

| Provider | Default model |
|---|---|
| `openai-generic` | `anthropic/claude-sonnet-4.6` |
| `Claude` | `claude-opus-4-6` |
| `Codex` | `gpt-5.4` (`reasoning=high`) |
| `Google OAuth` | `gemini-3.1-pro-preview` |

`openai-generic` defaults to `https://openrouter.ai/api/v1` as its base URL.

Stored config lives at `~/.lash/config.json` and uses this shape:

```json
{
  "provider": {
    "type": "openai-generic",
    "api_key": "...",
    "base_url": "https://openrouter.ai/api/v1"
  },
  "auxiliary_secrets": {
    "tavily_api_key": "..."
  },
  "agent_models": {
    "low": "...",
    "medium": "...",
    "high": "..."
  }
}
```

`auxiliary_secrets.tavily_api_key` is the supported Tavily location. The old top-level `tavily_api_key` field is rejected, and legacy sub-agent tier aliases are no longer part of the documented config surface.

Default sub-agent tier mapping (`low` / `medium` / `high`):

| Provider | low | medium | high |
|---|---|---|---|
| `openai-generic` | `minimax/minimax-m2.5` | `z-ai/glm-5` | `anthropic/claude-sonnet-4.6` |
| `Claude` | `claude-haiku-4-5` | `claude-sonnet-4-6` | `claude-sonnet-4-6` |
| `Codex` | `gpt-5.3-codex-spark` | `gpt-5.4` (`reasoning=medium`) | `gpt-5.4` (`reasoning=high`) |
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

### Runtime APIs

`LashRuntime` exposes two first-class host APIs:

- `stream_turn(input, sink, cancel)`:
  - Streams low-level, mode-specific `AgentEvent` values to an `EventSink`.
  - Returns canonical `AssembledTurn` when the turn terminates.
- `run_turn_assembled(input, cancel)`:
  - Convenience path when the host only needs the high-level terminal result.

### Assembled turn contract

`AssembledTurn` is the canonical terminal result:

- `status`: `completed` | `interrupted` | `failed`
- `done_reason`: `model_stop` | `max_turns` | `user_abort` | `tool_failure` | `runtime_error`
- `execution.mode`: `repl` | `native_tools`
- `execution.had_tool_calls` and `execution.had_code_execution`: high-level execution summary
- `assistant_output.safe_text`: host-renderable output (always present, may be empty)
- `assistant_output.raw_text`: unsanitized terminal output (always present, may be empty)
- `assistant_output.state`: `usable` | `empty_output` | `traceback_only` | `sanitized` | `recovered_from_error`
- `tool_calls`, `code_outputs`, `errors`, `token_usage`, and updated `state`

High-level contract:

- `AssembledTurn` is stable across execution modes.
- `standard` is a subset of the `repl` result shape at this level.
- REPL-only detail is represented by `code_outputs`, which is empty for native tool-calling turns.
- The `batch` tool is an optimization at the execution layer only; batched work still folds back into the same high-level turn contract.

Host/runtime policy knobs are configured via `RuntimeConfig`:

- `host_profile` (`interactive` / `headless` / `embedded`)
- `context_folding.soft_limit_pct` / `context_folding.hard_limit_pct`
- `base_dir` + optional custom `path_resolver`
- `sanitizer` and `termination` policies

The same folding policy is persisted in CLI config under:

```json
{
  "runtime": {
    "context_folding": {
      "soft_limit_pct": 50,
      "hard_limit_pct": 60
    }
  }
}
```

Integration invariants:

- Keep one `LashRuntime` instance per active session.
- Do not rebuild message history manually between turns while a runtime is alive.
- Treat streamed `AgentEvent`s as low-level preview/progress; render terminal user output from `AssembledTurn.assistant_output.safe_text`.

## Tool Exposure Model

Tools define `inject_into_prompt: bool` (`lash/src/lib.rs`).

- `inject_into_prompt=true`: tool appears directly in the LLM system prompt.
- `inject_into_prompt=false`: tool is omitted from prompt for brevity, but still callable at runtime.
- Most tool identity is shared across execution modes, but mode-specific toolsets can expose different shell surfaces (`exec_command`/`write_stdin` in `standard`, `shell*` in `repl`).
- Prompt injection only controls which tools are described up front. It does not change the active runtime catalog.

In `repl`, tools are called through `lashlang`:

```txt
files = call glob { pattern: "src/**/*.rs" }
tool_catalog = call search_tools {}
observe files
finish tool_catalog
```

In `standard`, tools are provider-native tool calls.

Use `search_tools` only when the tool you need is not already listed in Available Tools. With no query, it returns the full active tool catalog; with a focused query, it returns ranked matches. Both modes search the same live runtime catalog.

### Default Standard Surface

With the default capability profile, `standard` can call:

- Core read: `read_file`, `glob`, `grep`, `ls`, `search_tools`
- Core write: `apply_patch`
- Shell: `exec_command`, `write_stdin`
- Planning: `update_plan`, `ask` (interactive sessions only)
- Delegation: `agent_call`, `agent_result`, `agent_kill`
- History: `search_history`
- Memory: `search_mem`, `mem_set`, `mem_get`, `mem_delete`, `mem_all`
- Skills: `load_skill`, `read_skill_file`, `search_skills`
- Web, when Tavily is configured: `search_web`, `fetch_url`
- Native-tools only: `batch`

This is the callable default surface, not the prompt-injected subset. Lash injects a smaller prompt-visible list and relies on `search_tools` only for genuine discovery of omitted tools.

`search_history(...)` searches all prior completed turns persisted by the runtime, not just turns folded out of the active prompt window.
For delegation, low-tier sub-agents use this standard surface by default even when the parent session is in `repl`.

## Filesystem Listing Output

`call glob { ... }` / `call ls { ... }` in `repl`, and `glob(...)` / `ls(...)` in `standard`, return a typed envelope rather than plain path/tree strings:

```json
{
  "__type__": "path_entries",
  "items": [
    {
      "path": "src/main.rs",
      "kind": "file",
      "size_bytes": 1234,
      "lines": 87,
      "modified_at": "2026-03-09T10:11:12Z"
    }
  ],
  "truncated": null
}
```

Each item includes:

- `path`
- `kind` (`file` / `dir` / `symlink` / `other`)
- `size_bytes`
- `lines` (`null` unless `with_lines=true`)
- `modified_at` (RFC3339 UTC)

Both tools support `limit` (`null` for uncapped) and expose truncation metadata on the returned object in REPL wrappers.

All rights reserved.
