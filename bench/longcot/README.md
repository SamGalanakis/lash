# LongCoT

Native Rust harness that runs [LongCoT](https://github.com/LongHorizonReasoning/longcot) — Motwani et al.'s long-horizon chain-of-thought benchmark (2,500 expert-designed problems across logic, cs, chemistry, chess, math) — through lash as an RLM-style reasoning system.

The runner is a `bench-longcot` crate that builds directly against `lash` (no `lash-cli` subprocessing — each question runs inside its own `LashRuntime` in-process). Defaults mirror upstream's `oai_gpt52.yaml` reference config (gpt-5.2, reasoning effort `high`, `max_output_tokens=125000`) so scores sit on the same axis as the published [LongCoT leaderboard](https://longcot.ai). The [raw.works writeup](https://raw.works/longcot-a-benchmark-worthy-of-a-rlms-attention/) is the secondary RLM-style reproduction we cross-reference for prompt/strategy choices.

All runtime artifacts live under ignored `.benchmarks/longcot/`.

## Defaults

| Knob | Default | Reference |
|---|---|---|
| Model | `openai/gpt-5.2` | matches upstream `src/configs/oai_gpt52.yaml` — change with `--model` |
| Provider | `openai-compatible` (OpenRouter OAI-compat endpoint) | `--provider-id` |
| Max turns | 50 | matches the reference RLM iteration cap |
| Max output tokens | 125,000 | matches upstream `oai_gpt52.yaml: max_output_tokens=125000`; plumbed via `LASH_MAX_OUTPUT_TOKENS` |
| Reasoning variant | `high` | matches upstream `oai_gpt52.yaml: reasoning.effort=high` |
| Max context tokens | 1,000,000 | — |
| Execution mode | `rlm` (lashlang DSL, fixed) | the intentional delta vs the upstream raw-LLM harness |

## Tool surface

The Rust runner registers an explicit RLM tool surface for benchmark hygiene, mirroring the continual-learning-bench setup:

- **registered (model-visible):** `llm_query`, `spawn_agent` with `capability: "default"`, `continue_as`, `list_async_handles`
- **inherited by `spawn_agent` children:** the same four tools, so recursive descents stay inside the same surface
- **not registered:** local shell, filesystem, search, web, editing, MCP, monitor, task-controls, AppWorld, and standard-mode context plugins (rolling-history / observational-memory)

LongCoT prompts explicitly forbid external tool use; this stack honors that while keeping recursive decomposition (via `spawn_agent`) available.

### `seed:` channel for child sessions

Both `spawn_agent` and `continue_as` accept `seed: { name: value, ... }`. Children inherit nothing else; pass everything they need through seed. Each entry's kind is preserved automatically by source:

- `seed: { problem: input.prompt }` — `input.prompt` is a host-projected binding on the parent, so the child receives `problem` as a read-only projected binding (visible in its system prompt under `Host Projected Variables`).
- `seed: { findings: findings }` — `findings` is a regular RLM global, so the child receives it as a regular global.
- `seed: { hint: slice(input.prompt, 0, 1000) }` — computed expression, defaults to global.

The wire encoding is a canonical `{"__projected__": <inner>}` wrapper any time lashlang serializes a `Value::Projected` to JSON; tools that don't care unwrap transparently via `lash::tools::unwrap_projected_arg`.

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

> Execution mode is fixed to `rlm`; `--execution-mode` and `--standard-context-approach` are no longer accepted. Standard-mode context plugins (rolling-history / observational-memory) and the monitor / task-controls tool surface are intentionally left unregistered — see "Tool surface" above.

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
    session.trace.jsonl   # structured runtime trace
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

The Gemini fallback judge (math/chemistry borderline cases) needs `GEMINI_API_KEY` or `GOOGLE_API_KEY` in `.env`. Pass `-- --no-fallback` to disable; scores tend to dip without it.

## Submitting to the leaderboard

Submissions are PRs against [LongHorizonReasoning/longcot](https://github.com/LongHorizonReasoning/longcot) with a JSON file shaped like:

```json
{
  "model": "lash + gpt-5.2 (RLM)",
  "provider": "your-org",
  "type": "open",
  "results": [
    {"question_id": "Sudoku_easy_1", "response": "...", "correct": true}
  ]
}
```

To produce that from a run, walk `responses/<label>.jsonl` (one row per question) and pair each `question_id` + `response_text` with the `correct` flag from `run_eval.py`'s output. The `--harness` flag on `run.sh` (`restricted` | `open`) sets the manifest column for the eventual PR; we always run the same lash RLM stack regardless.
