# Continual Learning Bench

This harness runs Lash as a local system inside
[pgasawa/continual-learning-bench](https://github.com/pgasawa/continual-learning-bench)
without vendoring that repository into this checkout.

Generated state lives under ignored `.benchmarks/continual-learning-bench/`:

- `vendor/continual-learning-bench/` - shallow upstream checkout
- `venv/` - Python environment for `clbench`
- `runs/` and upstream `results/` - CLBench outputs
- generated upstream `src/systems/lash/` adapter files copied from this directory

The Python CLBench system is only a thin shim. Each benchmark query is executed
by the local Rust runner `bench-clbench-lash`, which embeds Lash directly instead
of shelling through `lash-cli`.

## Quickstart

```bash
bench/continual-learning-bench/setup.sh
bench/continual-learning-bench/run.sh exploitable_poker \
  --schedule quick_test \
  --system lash \
  --system.provider-id codex \
  --system.model gpt-5.5 \
  --system.variant high \
  --system.execution-mode rlm
```

`run.sh` delegates to `clbench run`, so CLBench task and system flags pass
through unchanged. For example, use `--task.schedule` if your installed CLBench
revision requires the dynamic task-parameter spelling instead of `--schedule`.
For consistency with the other benchmark wrappers, `run.sh` also accepts
`--max-concurrency` and `--n-concurrent` aliases and forwards them to CLBench's
`--max-workers`.

## Useful Flags

```bash
bench/continual-learning-bench/run.sh exploitable_poker --schedule quick_test --system lash --dry-run
bench/continual-learning-bench/run.sh sales_prediction --schedule default --system lash --system.provider-id openai-compatible --system.model gpt-5.4 --system.variant high
bench/continual-learning-bench/run.sh --config .benchmarks/continual-learning-bench/vendor/continual-learning-bench/configs/exploitable_poker/exploitable_poker_icl.json --system lash --system.provider-id codex --system.model gpt-5.5 --system.variant high
bench/continual-learning-bench/run.sh exploitable_poker --task.schedule quick_test --system lash --runs 3 --max-concurrency 3
bench/continual-learning-bench/run_all.sh --name lash-full --system lash --task-parallelism 3 --per-task-parallelism 4 --skip-baseline --system.provider-id openai-compatible --system.model anthropic/claude-sonnet-4.6 --system.variant high
bench/continual-learning-bench/run_all.sh --background --name lash-full --system lash --task-parallelism 3 --per-task-parallelism 4 --skip-baseline --system.provider-id openai-compatible --system.model anthropic/claude-sonnet-4.6 --system.variant high
```

The Lash adapter keeps one Lash session across CLBench instances until CLBench
calls `reset()`. That preserves online-learning state during rollout runs while
letting CLBench reset the system for baseline instances.

Use `--skip-baseline` (or `LASH_CLBENCH_SKIP_BASELINE=1`) to report rollout
reward without running the stateless baseline comparison. Gain metrics are not
available in that mode.

Use `run_all.sh --background` to detach the full-suite run while keeping the
live dashboard server in the benchmark process. The wrapper prints the PID,
log path, and dashboard URL when available; it also writes
`.benchmarks/continual-learning-bench/run-all-background.pid` and
`.benchmarks/continual-learning-bench/run-all-background.env`.

The Rust runner registers an explicit RLM tool surface for benchmark hygiene:

- registered: `llm_query`, `spawn_agent` with `capability: "explore"`,
  `continue_as`, and async-handle helpers
- not registered: local shell, filesystem tools, search, web, editing, monitor,
  MCP, and AppWorld tools

Each turn passes CLBench's current `response_schema` into RLM as the required
`submit` schema, so the model receives runtime feedback if `submit <value>` does
not match the benchmark step.

## Requirements

- `uv`
- `git`
- Docker for CLBench tasks/systems that need containerized workspaces
- `~/.lash/config.json` with the requested provider configured

By default, `setup.sh` installs CLBench with all optional task dependencies and
runs `clbench setup --all`. Use `--skip-task-setup` to install only the Python
environment and adapter.
