# lash

Terminal AI coding agent with:

- an interactive TUI
- an autonomous single-shot CLI preset
- two execution modes: persistent `repl` and provider-native `standard`
- patch-based editing
- skills, sub-agents, planning, shell, and optional web search
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

Autonomous single-shot usage:

```bash
lash -p "summarize this repo"
lash --print "explain src/main.rs"
```

`--print` runs a single autonomous turn, prints the final response to stdout, and skips the interactive prompt bridge.

Common flags:

```bash
lash --model gpt-5.4
lash --execution-mode standard
lash --provider
lash --no-mouse
lash --reset
```

## Key Features

- `repl` mode: persistent runtime across turns
- `standard` mode: provider-native tool calling
- `lashlang` REPL with `parallel { ... }` concurrency
- `apply_patch` editing flow
- shell execution and streamed output
- durable sessions with resume and retry
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

- `standard`: `exec_command` and `write_stdin` are PTY-backed and return incremental terminal output
- `repl`: `shell`, `shell_wait`, `shell_read`, `shell_write`, and `shell_kill` expose the same PTY-style terminal semantics through persistent handles

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
- `/caps`
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

Skills are markdown-based capability bundles.

Locations:

- global: `~/.lash/skills/`
- repo-local: `.agents/lash/skills/`

Use them with:

- `/skills` in the TUI
- `/<skill-name>`

## Prompt Customization

Override prompt sections at launch:

```bash
lash --prompt-replace "identity=You are terse."
lash --prompt-append "guidelines=Always run tests."
lash --prompt-disable "tool_guides"
lash --prompt-replace-file "guidelines=./prompt.md"
```

## Terminal Bench

Run Harbor + Terminal Bench with the in-repo lash adapter:

```bash
scripts/run-terminalbench.sh --sample --execution-mode repl
scripts/run-terminalbench.sh --sample --execution-mode standard --tasks regex-log,sqlite-with-gcov
scripts/run-terminalbench.sh --full --execution-mode standard --task "git-*"
```

Each run exports a structured snapshot to `.benchmarks/terminalbench/` by default, including:

- global stats
- per-task rollups
- per-trial timing and token usage
- copied logs and verifier output
- run parameters such as provider, model, variant, execution mode, concurrency, and timeouts

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

`lash-core` is available as a library:

```toml
[dependencies]
lash-core = { path = "../lash", default-features = false, features = ["full"] }
```

Embedders provide model metadata explicitly and can choose their own catalog source and storage. The
first-party CLI uses a modular `models.dev` adapter plus a built-in `tool_surface` plugin that owns:

- which tools are injected into the prompt
- whether `search_tools()` is exposed because additional tools were omitted from the prompt
- extra tool-usage guidance such as omitted-tool notes

## Config

Stored config lives at:

```txt
~/.lash/config.json
```

Relevant runtime settings include:

- `runtime.context_folding`
- `runtime.low_tier_subagent_execution_mode`
  `standard` by default for `agent_call` low-tier workers; set to `repl` only if you want low-tier sub-agents to execute via `lashlang`

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
    "context_folding": {
      "soft_limit_pct": 50,
      "hard_limit_pct": 60
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
