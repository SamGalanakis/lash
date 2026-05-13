# OOLONG RLM

Native Rust harness that runs OOLONG through Lash's lashlang-backed RLM mode.

The RLM paper's Table 1 uses the OOLONG `trec_coarse` split with 50 tasks at
the 131K-token setting. This harness makes that comparable slice the default
setup target while keeping the runner configurable for model, child model,
concurrency, turn limits, output limits, and context fallback thresholds.

All runtime artifacts live under ignored `.benchmarks/oolong/`.

## Setup

```bash
bench/oolong/setup.sh
```

Default setup exports:

```bash
oolongbench/oolong-synth
split:       test
dataset:     trec_coarse
context_len: 131072
limit:       50
output:      .benchmarks/oolong/data/oolong_synth.jsonl
```

Other useful slices:

```bash
bench/oolong/setup.sh --suite synth-with-labels --dataset trec_coarse --context-len 131072 --limit 50
bench/oolong/setup.sh --suite real --config dnd --limit 50
```

`setup.sh` uses `uv run --with datasets --with pyarrow` so it does not add
Python dependencies to the repo.

## Run

```bash
bench/oolong/run.sh --suite synth --batch-size 4
```

Comparable RLM-paper style knobs:

```bash
bench/oolong/run.sh \
  --suite synth \
  --dataset-name trec_coarse \
  --context-len 131072 \
  --max-questions 50 \
  --model openai/gpt-5 \
  --variant medium \
  --child-model openai/gpt-5-mini \
  --batch-size 4 \
  --max-turns 50 \
  --max-context-tokens 1000000 \
  --max-output-tokens 125000
```

The root and children run the same locked-down RLM tool surface:

- `llm_query`
- `spawn_agent` with `capability: "default"`
- `continue_as`
- `list_async_handles`

No shell, filesystem, web, MCP, monitor, or task-control tools are registered.

## Outputs

```
.benchmarks/oolong/runs/<run-id>/
  manifest.json
  results.json
  predictions.jsonl
  index.html
  questions/<question-id>/
    question.json
    prompt.txt
    answer.txt
    events.jsonl
    session.db
    session.trace.jsonl
    system_prompt.txt
    trace.html
    result.json
```

## Evaluate

```bash
bench/oolong/evaluate.sh .benchmarks/oolong/runs/<run-id>
```

This reports exact normalized match against the dataset `answer` field. It is
not an implementation of OOLONG-Pairs; the RLM paper describes OOLONG-Pairs as
a separate 20-query quadratic variant, so that should be added only from the
official query/data generation path.
