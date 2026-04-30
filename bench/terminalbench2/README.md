# Terminal Bench 2

This directory is the tracked home for lash's Terminal Bench 2 (Harbor) setup.

The lash agent uses **your local Codex OAuth subscription** by default — the harness rsyncs `~/.lash/config.json` into the benchmark container, so every task call bills through your active provider (typically Codex GPT-5.4). No OpenRouter / API-key juggling.

## Quickstart

```bash
# Smoke test — one easy task, RLM + gpt-5.4 at high effort.
bench/terminalbench2/run.sh --sample --preset trivial --execution-mode rlm --model gpt-5.4 --variant high

# Full sample dataset (~70 tasks), RLM.
bench/terminalbench2/run.sh --sample --execution-mode rlm --model gpt-5.4 --variant high

# Full dataset (~270 tasks), Standard mode + rolling history.
bench/terminalbench2/run.sh --full --execution-mode standard --model gpt-5.4 --variant high --context-approach rolling_history
```

`--model` is optional; if `~/.lash/config.json`'s active provider is Codex, passing just `--variant high` picks up whatever model that provider is configured with. We pin `--model gpt-5.4` in examples to make runs reproducible across config changes.

Or call the underlying script directly:

```bash
scripts/run-terminalbench.sh --help
```

## Presets

| Preset | Size | Notes |
|---|---|---|
| `trivial` | 1 task | Smoke test, one easy task. |
| `smoke` | 2 tasks | Quick sanity. |
| `fast-3` | 3 tasks | Three short tasks for fast iteration. |
| `fast-medium` | 4 tasks | Adds one medium task. |
| `memory-3` | 3 tasks | Memory-heavy tasks (full dataset only). |
| `recall-3` | 3 tasks | Recall-heavy tasks (full dataset only). |
| `representative-10` | 10 tasks | A mixed slice for leaderboard comparison (full dataset only). |

Custom selection:

```bash
# One task by name
bench/terminalbench2/run.sh --sample --task chess-best-move --execution-mode rlm --model gpt-5.4 --variant high

# Glob
bench/terminalbench2/run.sh --full --task "git-*" --execution-mode standard --model gpt-5.4 --variant high

# Explicit list
bench/terminalbench2/run.sh --sample --tasks regex-log,fix-code-vulnerability --execution-mode standard --variant high
```

## Flags that matter

| Flag | Required | Notes |
|---|---|---|
| `--execution-mode` | yes (for lash) | `rlm` or `standard`. No `repl` (old name). |
| `--variant` | yes | Provider-native effort level: `high`, `xhigh`, etc. |
| `--model` | optional for lash | Defaults to the active provider's model in `~/.lash/config.json`. Pass `--model gpt-5.4` to pin. |
| `--context-approach` | optional | Standard mode only: `rolling_history` or `observational_memory`. |
| `--build-mode` | optional | `docker-bookworm` (default) / `docker-bullseye` / `host`. |
| `--n-concurrent` | optional | Parallel trials. Default 1. |
| `--attempts` | optional | Attempts per trial. Default 1. |

## Outputs

- **Harbor jobs**: `jobs/<job-name>/` — raw harbor session logs, pulled container filesystems.
- **Structured results**: `.benchmarks/terminalbench2/<job-name>/` — gathered trial JSONs + a markdown summary.
- **Lash traces**: exported under each trial's `artifacts/sessions/*.trace.jsonl` when the lash agent runs; dashboard LLM-call counts come from typed `llm_call_completed` trace records.

## Implementation notes

- Entry point: `bench/terminalbench2/run.sh` → `scripts/run-terminalbench.sh` → `harbor run …`.
- Lash driver: `scripts/harbor_lash_agent.py` — `BaseInstalledAgent` that rsyncs the host's lash binary + `~/.lash/config.json` into the container, then runs `lash --print "<task>"` with the right flags.
- Tracing: `scripts/run-terminalbench.sh` sets `LASH_LOG=debug` for lash runs so the CLI writes typed per-session traces. `scripts/terminalbench_results.py` reads those traces for model, token, turn, and completed-call metadata.
- **Credentials**: your host's `~/.lash/config.json` (with Codex OAuth tokens, OpenRouter key, etc.) is copied into the container. The container's lash picks up the active provider from that config. Bypass with `--allow-no-config` if you're testing non-lash agents (opencode / codex).
- The benchmark-harness guidance ("you're being graded by exact verifier checks, don't ask questions…") is folded directly into the user prompt by the Python agent. The old `--prompt-replace` / `--prompt-append` / `--prompt-disable` CLI flags were removed from lash; the harness no longer relies on them.
- Agent build: default is `docker-bookworm`, which produces `target-bookworm/release/lash` on the host. Use `--no-build` to reuse a prior build.

## Requirements

- `harbor` CLI installed ([laude-institute/harbor](https://github.com/laude-institute/harbor)).
- `docker` (unless `--build-mode host --env host`).
- `~/.lash/config.json` with an active provider set up (`lash --provider` interactively, or by copying a working config).
