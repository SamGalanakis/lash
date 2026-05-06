---
name: harness-opt
description: Operate Lash harness optimization runs, especially lash-harness-opt GEPA/CLBench runs. Use when starting, monitoring, inspecting, summarizing, or tuning optimizer runs; reviewing proposal artifacts; choosing budgets/selectors/frontier settings; or wiring a proposer/meta-agent prompt for harness optimization.
---

# Harness Opt

Use this skill to run and inspect `lash-harness-opt` as a batch optimizer. Treat the CLI and run directory as the source of truth; do not add Lash plugin or UI integration unless the user explicitly asks for it.

## First Checks

From the repo root:

```bash
cargo run -p lash-harness-opt -- check clbench --config <config.json>
```

If the check fails, fix the config or candidate surface before starting an optimization run.

## Start A Run

Use a new run directory per experiment. Keep names concrete enough to compare later.

```bash
cargo run -p lash-harness-opt -- optimize clbench \
  --config <config.json> \
  --run-dir runs/<run-name> \
  --max-metric-calls 200 \
  --minibatch-size 8 \
  --max-concurrency 4 \
  --candidate-selection pareto \
  --component-selection round-robin \
  --frontier instance \
  --proposer-prompt skills/harness-opt/references/gepa-proposer-prompt.md
```

Optional merge:

```bash
  --use-merge \
  --max-merge-invocations 10
```

For quick smoke runs, use `--max-metric-calls 10 --minibatch-size 2`.

## Monitor A Run

Use `stats` as the stable monitoring API:

```bash
cargo run -q -p lash-harness-opt -- stats runs/<run-name>
```

For live monitoring:

```bash
watch -n 5 'cargo run -q -p lash-harness-opt -- stats runs/<run-name> | jq'
```

Interpret the key fields:

- `metric_calls_used`: budget consumed; cache hits do not count.
- `accepted_proposals` / `rejected_proposals`: whether reflection is producing useful mutations.
- `best_mean_score`: current scalar best.
- `cache_hits` / `cache_misses`: whether resume and repeated minibatches are reusing evaluations.
- `pareto_coverage`: number of current frontier candidates.

## Inspect Artifacts

The run directory is the operational record:

```text
runs/<run-name>/harness-opt.sqlite
runs/<run-name>/proposals/<generation>/prompt.md
runs/<run-name>/proposals/<generation>/output.json
runs/<run-name>/examples/<candidate-id>/<example-id>/
```

When explaining a run, inspect both `stats` and the latest proposal artifacts. Prefer reading `output.json` for the model submission and `prompt.md` for the exact evidence/context given to the proposer.

Useful commands:

```bash
find runs/<run-name>/proposals -maxdepth 2 -type f | sort | tail -20
sqlite3 runs/<run-name>/harness-opt.sqlite '.tables'
sqlite3 runs/<run-name>/harness-opt.sqlite 'select generation, accepted, reason, before_score, after_score, candidate_id from proposals order by id desc limit 10;'
```

## Meta-Agent Prompt

The proposer/meta-agent prompt should be explicit, evidence-grounded, and patch-scoped. Pass a prompt template with:

```bash
--proposer-prompt skills/harness-opt/references/gepa-proposer-prompt.md
```

A default prompt template is in `references/gepa-proposer-prompt.md`.

The template supports these placeholders:

- `{{parent_candidate_id}}`
- `{{mutable_components_json}}`
- `{{evidence_json}}`
- `{{output_schema_json}}`

When customizing the prompt, preserve these constraints:

- Only patch listed mutable component ids.
- Preserve generic RLM execution protocol, Lashlang reference, tool contracts, tool availability, and typed response-schema enforcement.
- Use the supplied evidence and traces; do not invent benchmark results.
- Return only the required structured proposal submission.

## Realistic Operating Loop

1. Run `check clbench`.
2. Start with a small smoke optimization.
3. Monitor `stats` until completion or failure.
4. Inspect the latest accepted and rejected proposal artifacts.
5. Increase `--max-metric-calls` once the run is producing sensible accepted proposals.
6. Compare run dirs by `best_mean_score`, accepted proposal count, and proposal reasons.

## Defaults

Use these defaults unless the user gives stronger direction:

- `candidate-selection`: `pareto`
- `component-selection`: `round-robin`
- `frontier`: `instance` for scalar CLBench
- `minibatch-size`: `8` for real runs, `2` for smoke tests
- `max-concurrency`: `4`
- `skip-perfect-score`: enabled

Avoid building a Lash extension for this workflow until the CLI plus skill loop proves painful in real use.
