# lash

Terminal AI coding agent with:

- an interactive TUI
- an autonomous single-shot CLI preset
- two execution modes: persistent `repl` and provider-native `standard`
- patch-based editing
- skills, delegated child-session workers, planning, shell, and optional web search
- a Harbor Terminal Bench runner with a local results UI

## Install

Latest release:

```bash
curl -fsSL https://github.com/SamGalanakis/lash/releases/latest/download/install_lash.sh | bash
```

From this repo:

```bash
./install_lash.sh
```

## Quick Start

First run opens provider setup if needed:

```bash
lash
```

Interactive setup currently supports:

- `codex`: OpenAI Codex OAuth
- `google_oauth`: Google OAuth for Gemini
- `openai-generic`: API key auth, defaulting to the OpenRouter base URL

Autonomous single-shot usage:

```bash
lash -p "summarize this repo"
lash --print "explain src/main.rs"
```

`--print` runs a single autonomous turn, prints the final response to stdout, and skips the interactive prompt bridge.
While it runs, live progress and tool activity stream to stderr.

Common flags:

```bash
lash --model gpt-5.4
lash --execution-mode standard
lash --check-update
lash --update
lash --provider
lash --no-mouse
lash --reset
```

## Docs

Start with the docs hub:

```txt
docs/index.html
```

It links to:

- `README.md`: install, CLI, execution modes, and integration overview
- `docs/architecture.html`: runtime, plugin, prompt, and tool architecture
- `docs/design.html`: TUI visual system and interaction design

## Key Features

- `repl` mode: persistent runtime across turns
- `standard` mode: provider-native tool calling
- `lashlang` REPL with `parallel { ... }` concurrency
- `apply_patch` editing flow
- shell execution and streamed output
- durable sessions with resume and retry
- live token/context accounting in the TUI status bar as usage streams in
- skills loaded from global and repo-local skill directories
- image/file/path references in prompts
- planning with `update_plan`
- benchmark runner and results browser

## Execution Modes

`repl` is the default.

- `repl`: persistent `lashlang` runtime, best when the agent benefits from state across turns
- `standard`: provider-native tool calling, including multiple native tool calls in one response for independent concurrent work

Choose explicitly:

```bash
lash --execution-mode repl
lash --execution-mode standard
```

Concurrency surface:

- `repl`: use `parallel { ... }`; in expression position it returns a source-ordered list of branch results, and bare expression branches contribute their values directly
- `standard`: emit multiple independent tool calls in the same response; the runtime executes them concurrently and returns all results before the next model step

Selected `lashlang` semantics:

- `slice(value, start, end)` treats `null` bounds as omitted: `start=null` means from the beginning, `end=null` means through the end
- negative list/string indices count from the end; out-of-bounds indices return `null`
- `contains(record, key)` checks record keys in addition to string substring and list membership
- string comparisons like `"abc" < "def"` are lexicographic
- `to_string([1, 2])` and `to_string({ a: 1 })` preserve integer formatting instead of forcing `1.0`

Shell surface:

- `standard` and `repl`: `exec_command` and `write_stdin` are PTY-backed and return incremental terminal output

## Useful Commands

Inside the TUI:

- `/help`
- `/clear`
- `/fork [prompt]`
- `/version`
- `/info`
- `/model [name]`
- `/variant [name]`
- `/mode [repl|standard]`
- `/provider`
- `/login`
- `/logout`
- `/retry`
- `/resume`
- `/skills`
- `/tools`
- `/reconfigure`
- `/exit`

Useful keys:

- `Esc`: cancel run / close prompt
- `Shift+Enter`: newline
- `Ctrl+V`: paste image
- `Ctrl+Shift+V`: paste text only
- `Ctrl+Y`: copy last response
- `Ctrl+O`: cycle expansion
- `Alt+O`: full expansion

## Skills

Skills are markdown-based workflow bundles.

Locations:

- global: `~/.lash/skills/`
- repo-local: `.agents/lash/skills/`

Use them with:

- `/skills` in the TUI
- `$<skill-name>` in your prompt

When a skill is selected, lash injects a `<skill>` block with that skill's
instructions into the turn input before the model runs.

## Prompt Customization

Override prompt sections at launch:

```bash
lash --prompt-replace "intro=You are terse."
lash --prompt-append "guidance=Always run tests."
lash --prompt-disable "available_tools"
lash --prompt-replace-file "guidance=./prompt.md"
```

## Terminal Bench

Run Harbor + Terminal Bench with the in-repo lash adapter:

```bash
scripts/run-terminalbench.sh --sample --execution-mode repl --model gpt-5.4 --variant high
scripts/run-terminalbench.sh --sample --preset trivial --execution-mode repl --model gpt-5.4 --variant high
scripts/run-terminalbench.sh --sample --preset smoke --execution-mode repl --model gpt-5.4 --variant high
scripts/run-terminalbench.sh --sample --preset fast-3 --execution-mode standard --model gpt-5.4 --variant high
scripts/run-terminalbench.sh --sample --preset fast-medium --execution-mode standard --model gpt-5.4 --variant high
scripts/run-terminalbench.sh --full --preset memory-3 --execution-mode standard --model gpt-5.4 --variant high
scripts/run-terminalbench.sh --full --preset recall-3 --execution-mode standard --model gpt-5.4 --variant high
scripts/run-terminalbench.sh --full --preset representative-10 --execution-mode standard --model gpt-5.4 --variant high
scripts/run-terminalbench.sh --sample --execution-mode standard --model gpt-5.4 --variant high --build-mode host
scripts/run-terminalbench.sh --sample --execution-mode standard --tasks regex-log,sqlite-with-gcov --model gpt-5.4 --variant high
scripts/run-terminalbench.sh --full --execution-mode standard --task "git-*" --model gpt-5.4 --variant high
```

Run the same harness with OpenCode:

```bash
scripts/run-terminalbench.sh --agent opencode --sample --model openrouter/openai/gpt-5 --variant high
scripts/run-terminalbench.sh --agent opencode --sample --model anthropic/claude-sonnet-4-5 --variant high
```

Notes:

- `lash` remains the default agent.
- `--execution-mode` only applies to `lash`; OpenCode uses its native execution path.
- `--preset trivial` expands to `log-summary-date-ranges`.
- `--preset smoke` expands to `log-summary-date-ranges,fix-code-vulnerability`.
- `--preset fast-3` expands to `log-summary-date-ranges,fix-code-vulnerability,regex-log`.
- `--preset fast-medium` expands to `regex-log,log-summary-date-ranges,fix-code-vulnerability,sqlite-with-gcov`.
- `--preset memory-3` expands to `password-recovery,db-wal-recovery,git-leak-recovery` and requires `--full` / `terminal-bench@2.0`.
- `--preset recall-3` expands to `password-recovery,git-leak-recovery,sanitize-git-repo` and requires `--full` / `terminal-bench@2.0`. It is meant to stress cross-step fact retention and cleanup consistency more than single-artifact forensic decoding.
- `--preset representative-10` expands to `build-cython-ext,configure-git-webserver,db-wal-recovery,fix-code-vulnerability,git-leak-recovery,log-summary-date-ranges,nginx-request-logging,polyglot-c-py,regex-log,sqlite-with-gcov` and requires `--full` / `terminal-bench@2.0`.
- `--variant` is required for all benchmark runs so provider-native reasoning settings are explicit and reproducible.
- `--build-mode docker-bookworm` is the default for lash benchmark builds so the binary matches the benchmark container ABI. Use `host` only when you intentionally want to benchmark against the host libc.
- OpenCode benchmark runs require an explicit `--model provider/model`.
- OpenCode benchmark runs automatically copy local `opencode auth login` credentials from `~/.local/share/opencode/auth.json` into the Harbor container when present.
- OpenCode can still fall back to provider env vars such as `OPENROUTER_API_KEY` or `OPENAI_API_KEY`.

Each run exports a structured snapshot to `.benchmarks/terminalbench/` by default, including:

- global stats
- per-task rollups
- per-trial timing and token usage
- per-trial CPU time and peak memory where available
- copied logs and verifier output
- run parameters such as agent, provider, model, variant, execution mode, concurrency, and timeouts

Open the local results UI:

```bash
python3 scripts/bench_ui.py --results-dir .benchmarks/terminalbench --open
```

The UI supports:

- browsing runs
- multi-run selection
- side-by-side comparison
- trial log inspection
- deleting runs end-to-end, including the exported snapshot and original Harbor job directory

Change the export location with:

```bash
scripts/run-terminalbench.sh --results-dir /path/to/results --sample --execution-mode repl
```

## Build

```bash
cargo build -p lash-cli
cargo build -p lash-cli --release
```

## Integration

`lash` is the host/runtime library, and `lash-sansio` is the portable turn-machine core:

```toml
[dependencies]
lash = { path = "../lash/lash", default-features = false, features = ["full"] }
lash-sansio = { path = "../lash/lash-sansio" }
```

Embedders provide model metadata explicitly and can choose their own catalog source and storage. The
first-party CLI is plugin-first: it builds the session from tool plugins plus stateful plugins such
as history, planning, and delegation. `PluginHost` can opt sessions into a
`DynamicToolProvider`, and the active `PluginSession` owns that live tool graph. The built-in
`tool_surface` plugin owns:

- which tools are injected into the prompt
- whether `search_tools()` is exposed because additional tools were omitted from the REPL prompt
- omitted-tool notes attached to the available tool list

For plugin authors, `lash` exposes grouped registrar namespaces on `PluginRegistrar` such as
`reg.tools()`, `reg.prompt()`, `reg.surface()`, `reg.turn()`, `reg.tool_calls()`, `reg.output()`,
`reg.messages()`, `reg.tool_results()`, `reg.session()`, and `reg.external()`. Small static
plugins can use `StaticPluginFactory`, context-sensitive declarative plugins can use
`PluginSpecFactory`, and bespoke `SessionPlugin` implementations can still register the full hook
set directly. Plugin factories build against a session context that includes both `agent_id` and
`execution_mode`, so a plugin can choose the correct per-mode tool instance up front. Turn
lifecycle orchestration is plugin-owned: the active `PluginSession` prepares turns, applies
checkpoint directives, and finalizes committed turns before history is persisted. The host-facing
`SessionManager` also exposes generic child-session orchestration primitives such as
`create_session`, `start_turn_stream`, `await_turn`, `cancel_turn`, and `close_session`, which is
how `agent_call` launches delegated workers without special core-level subagent state.

## Config

Stored config lives at:

```txt
~/.lash/config.json
```

Relevant runtime settings include:

- `runtime.context_strategy`
- `runtime.low_tier_subagent_execution_mode`
  `standard` by default for `agent_call` low-tier child sessions; set to `repl` only if you want low-tier delegates to execute via `lashlang`

Supported provider ids:

- `codex`
- `google_oauth`
- `openai-generic`

Supported context strategies:

- `rolling_context`
- `recall_agent`

Config shape:

```json
{
  "active_provider": "openai-generic",
  "providers": {
    "openai-generic": {
      "type": "openai-generic",
      "api_key": "...",
      "base_url": "https://openrouter.ai/api/v1"
    }
  },
  "auxiliary_secrets": {
    "tavily_api_key": "..."
  },
  "agent_models": {
    "low": "...",
    "medium": "...",
    "high": "..."
  },
  "runtime": {
    "context_strategy": {
      "type": "rolling_context"
    },
    "low_tier_subagent_execution_mode": "standard"
  }
}
```

The config file is saved with mode `0600` on Unix.

`openai-generic` defaults to:

```txt
https://openrouter.ai/api/v1
```

## License

All rights reserved.
