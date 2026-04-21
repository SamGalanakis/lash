# LongCoT

Native Rust harness that runs [LongCoT](https://github.com/LongHorizonReasoning/longcot) — Motwani et al.'s long-horizon chain-of-thought benchmark (2,500 expert-designed problems across logic, cs, chemistry, chess, math) — through lash as an RLM-style reasoning system.

The runner is a `bench-longcot` crate that builds directly against `lash` (no `lash-cli` subprocessing — each question runs inside its own `LashRuntime` in-process). Plugin stack, policy, and prompt template are configured to match the reference LongCoT writeup ([raw.works/longcot](https://raw.works/longcot-a-benchmark-worthy-of-a-rlms-attention/)) as closely as lash allows.

All runtime artifacts live under ignored `.benchmarks/longcot/`.

## Defaults

| Knob | Default | Reference |
|---|---|---|
| Model | `openai/gpt-5.2` | reference blog used `claude-sonnet-4.5` — change with `--model` |
| Provider | `openai-compatible` (OpenRouter OAI-compat endpoint) | `--provider-id` |
| Max turns | 50 | matches `dspy.RLM max_iterations=50` |
| Max output tokens | 64,000 | matches reference `max_tokens=64000`; plumbed through the openrouter adapter |
| Max context tokens | 1,000,000 | — |
| Execution mode | `rlm` (lashlang DSL) | the intentional delta vs dspy.RLM |
| Context approach | `rolling_history` | — |

## Quickstart

```bash
# One-time: clone upstream dataset and seed the evaluator venv.
bench/longcot/setup.sh

# Tiny probe — 5 easy questions, gpt-5.2 default.
bench/longcot/run.sh --difficulty longcot-mini --max-questions 5

# Score a run with upstream's evaluator.
bench/longcot/evaluate.sh .benchmarks/longcot/runs/<run-id>
```

lash reads credentials from the repo `.env`. Either of these works:

- `OPENROUTER_API_KEY` (default path)
- `OPENAI_COMPATIBLE_API_KEY` + optional `OPENAI_COMPATIBLE_BASE_URL`

## Test-set selection

The CLI mirrors the upstream `run_inference.py` selection flags so test sets compose:

```bash
# One difficulty, all domains
bench/longcot/run.sh --difficulty easy

# One domain, all difficulties
bench/longcot/run.sh --domain logic

# Two domains crossed with the "longcot-mini" preset (easy only)
bench/longcot/run.sh --domain math --domain logic --difficulty longcot-mini

# The full benchmark minus easies (medium + hard, every domain) — matches the
# blog's "longcot" preset.
bench/longcot/run.sh --difficulty longcot

# Exactly one question by id
bench/longcot/run.sh --question-id math_hard_0042

# Shuffled sample of 20 with a reproducible seed
bench/longcot/run.sh --max-questions 20 --shuffle-seed 7

# Resume a prior run that crashed halfway
bench/longcot/run.sh --run-id 20260420T172030Z --resume
```

Full flag reference: `cargo run -p bench-longcot -- --help` (or `bench/longcot/run.sh --help`).

### Overriding the model

```bash
# Back to the reference blog's model
bench/longcot/run.sh --model anthropic/claude-sonnet-4.5

# Route GPT-5.2 through a Codex subscription instead of OpenRouter
bench/longcot/run.sh --provider-id codex --model gpt-5.2

# A specific reasoning variant (supported on GPT-5.2/5.3/5.4 via openrouter)
bench/longcot/run.sh --variant xhigh
```

## Output layout

```
.benchmarks/longcot/runs/<run-id>/
  manifest.json           # run settings (model, mode, selection, etc.)
  results.json            # aggregate summary (by_domain, totals, usage)
  index.html              # clickable trace index
  responses/<label>.jsonl # per-question responses in upstream eval format
  questions/<qid>/
    question.json         # the raw question
    prompt.txt            # the problem text
    answer.txt            # the model's final answer
    events.jsonl          # streamed session events
    session.db            # full session graph
    session.llm.jsonl     # raw LLM request/response log
    system_prompt.txt     # exact system prompt sent for this question
    trace.html            # self-contained session trace (from lash-export)
    result.json           # per-question structured result
```

## Evaluating

`evaluate.sh` delegates to the upstream repo's `run_eval.py` via `uv`, so the
scores you get back are directly comparable to LongCoT leaderboard numbers:

```bash
bench/longcot/evaluate.sh .benchmarks/longcot/runs/<run-id>
```

Any arguments after the run path are forwarded:

```bash
bench/longcot/evaluate.sh .benchmarks/longcot/runs/<run-id> -- --judge-model gpt-4o
```
