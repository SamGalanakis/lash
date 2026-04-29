# SWE-bench Verified

Direct lash → [SWE-bench](https://github.com/SWE-bench/SWE-bench) Verified harness. Each instance runs through the embedded lash runtime (no CLI subprocess, no Harbor), the resulting `git diff` becomes a prediction, and the upstream Docker evaluator grades the batch.

## Quickstart

```bash
# One-time: clone upstream, install the evaluator, export the dataset to JSONL.
bench/swebench/setup.sh

# Smoke test — one instance, RLM mode + Codex OAuth (active provider).
bench/swebench/run.sh --limit 1 --variant high

# Ten instances, batch of 5.
bench/swebench/run.sh --limit 10 --variant high

# Grade predictions with the upstream Docker evaluator.
bench/swebench/evaluate.sh .benchmarks/swebench/runs/<run_id>
```

`--model` is optional; if omitted, the runner uses the active provider's default model from `~/.lash/config.json` (typically `gpt-5.4` on Codex).

## What the runner does

For each instance:

1. **Materialize the repo.** Clones `https://github.com/<repo>.git` as a bare mirror into `.benchmarks/swebench/workspace/<org__repo>.git` (shared across instances), then `git worktree add --detach <base_commit>` into the per-instance directory. Fetches on-demand when the base commit isn't present yet.
2. **Run lash.** Each instance is spawned as its own subprocess (lash's file tools resolve paths against the process CWD, so we need per-instance isolation to run `batch_size > 1`). The child starts the embedded runtime with the worktree as the project root, RLM execution mode, rolling-history context, and the default tool surface (shell, apply_patch, read_file, ls, grep, glob). No `ask` (autonomous) and no web tools.
3. **Capture the patch.** `git add -A && git diff HEAD --binary` against the base commit — that's the `model_patch`. Empty patches are recorded but graded as `fail`.
4. **Record artifacts.** `instances/<id>/` gets `model.patch`, `prompt.txt`, `result.json`, `events.jsonl`, `session.trace.jsonl`, `session.db`. Worktrees are removed after the instance finishes (they can be 100s of MB).

`predictions.jsonl` (top-level, one JSON object per line) is what the evaluator consumes. It follows upstream's schema:

```json
{"instance_id": "...", "model_name_or_path": "gpt-5.4", "model_patch": "diff --git ..."}
```

## Dashboard

Each run emits a `run.json` compatible with `scripts/bench_ui.py`. Launch it with:

```bash
bench/swebench/dashboard.sh --open      # pins --results-dir for you
# or equivalently:
scripts/bench_ui.py --results-dir .benchmarks/swebench --open
```

The standard terminalbench dashboard then shows pass/fail counts, token totals, per-instance artifacts, etc.

## Flags

| Flag | Default | Notes |
|---|---|---|
| `--dataset` | `.benchmarks/swebench/verified.jsonl` | Produced by `setup.sh`. Accepts `.jsonl` or `.json`. |
| `--instance-id` (repeatable) | — | Run only these IDs. |
| `--limit` | — | Cap instance count after filtering. |
| `--offset` | `0` | Skip the first N instances (after filtering). |
| `--batch-size` | `5` | Instances in flight simultaneously. |
| `--model` | active provider's default | e.g. `gpt-5.4`, `anthropic/claude-haiku-4.5`. |
| `--variant` | `high` | Reasoning effort. |
| `--execution-mode` | `rlm` | `rlm` or `standard`. |
| `--max-turns` | `60` | Per-instance turn cap. |
| `--run-id` | timestamp | Output dir name under `runs/`. |
| `--resume` | `false` | Skip instances whose prediction row already exists. |
| `--dry-run` | `false` | Print the plan without calling the model. |

## Requirements

- `git` + network access (clones the target repos on first use).
- `uv` for the upstream evaluator.
- `~/.lash/config.json` with an active provider (Codex OAuth, OpenRouter key, …). Run `lash --provider` once if you haven't.
- **Evaluator only:** Docker, enough disk for SWE-bench's instance images (GBs per task).
