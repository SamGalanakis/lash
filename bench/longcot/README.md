# LongCoT

Native Rust harness that runs [LongCoT](https://github.com/LongHorizonReasoning/longcot) — Motwani et al.'s long-horizon chain-of-thought benchmark (2,500 expert-designed problems across logic, cs, chemistry, chess, math) — through Lash as an RLM-style reasoning system.

The runner is a `bench-longcot` crate that builds directly against `lash` (no `cargo run -p lash-cli --print` subprocessing). Each question gets its own `LashRuntime` session, with policy + plugin stack configured to match the reference blog post ([raw.works/longcot](https://raw.works/longcot-a-benchmark-worthy-of-a-rlms-attention/)) as closely as lash allows.

Everything runtime-generated lives under ignored `.benchmarks/longcot/`.

## Blog reference-point alignment

| Knob | Blog setup | This harness |
|---|---|---|
| Model | `claude-sonnet-4-5` | `anthropic/claude-sonnet-4.5` via OpenRouter |
| Max iterations | `dspy.RLM max_iterations=50` | `SessionPolicy.max_turns=50` |
| Max output tokens | `max_tokens=64000` | `LASH_MAX_OUTPUT_TOKENS=64000` (plumbed into the openrouter adapter) |
| Prompt | LongCoT prompt verbatim, wrapped in `dspy.LongCoTSolve` signature | LongCoT prompt verbatim as user input; minimal system template names only the `solution = …` answer contract |
| RLM engine | `dspy.RLM` (Python, meta-level recursive self-calls) | lash's `lashlang`-backed RLM (Rust, DSL-in-response) — **intentional delta** |
| Context approach | n/a | `rolling_history` |
| Provider | OpenRouter | OpenRouter |

The harness reproduces the prompts, model, iteration cap, and max-output budget exactly. The execution engine is the one thing we can't share with dspy.RLM — we use lash's own recursive-language-model runtime instead, which is the whole point of benchmarking lash.

## Quickstart

```bash
bench/longcot/setup.sh                                              # clone upstream + uv sync verifier env
bench/longcot/run.sh --difficulty longcot-mini --max-questions 5    # tiny probe
bench/longcot/evaluate.sh .benchmarks/longcot/runs/<run-id>         # official scoring
```

Lash reads credentials from the repo `.env`. Either of these works:

- `OPENROUTER_API_KEY` (default path)
- `OPENAI_COMPATIBLE_API_KEY` + optional `OPENAI_COMPATIBLE_BASE_URL`

## Defaults

- model: `anthropic/claude-sonnet-4.5`
- provider: `openai-compatible`
- execution mode: `rlm`
- context approach: `rolling_history`
- `max_turns`: `50`
- `max_context_tokens`: `1_000_000`
- `max_output_tokens`: `64000` (exported into the process env as `LASH_MAX_OUTPUT_TOKENS`)
- batch size: `4`

## Useful flags

```bash
# Benchmarks (match upstream flag names)
bench/longcot/run.sh --difficulty longcot-mini        # easy split (~500 questions)
bench/longcot/run.sh --difficulty longcot             # medium + hard (~2000 questions)
bench/longcot/run.sh --domain math --difficulty hard

# Subsetting
bench/longcot/run.sh --question-id Sudoku_easy_1
bench/longcot/run.sh --max-questions 5
bench/longcot/run.sh --offset 100 --max-questions 50
bench/longcot/run.sh --shuffle-seed 42 --max-questions 50

# Lash policy
bench/longcot/run.sh --execution-mode standard
bench/longcot/run.sh --context-approach observational_memory
bench/longcot/run.sh --variant high
bench/longcot/run.sh --max-turns 100 --max-output-tokens 96000

# Routing
bench/longcot/run.sh --model openai/gpt-5.2
bench/longcot/run.sh --base-url https://openrouter.ai/api/v1

# Concurrency + resume
bench/longcot/run.sh --batch-size 8
bench/longcot/run.sh --run-id 20260417T154000Z --resume
```

## Output layout

```
.benchmarks/longcot/
├── vendor/longcot/                          # upstream clone (dataset + verifier)
├── runs/
│   └── <run-id>/
│       ├── index.html                       # run-level table linking every trace
│       ├── manifest.json                    # config snapshot + blog reference
│       ├── results.json                     # per-run summary (successful, by-domain, usage)
│       ├── responses/
│       │   └── <domain>_<difficulty>_<run-name>-<model-slug>.jsonl
│       └── questions/<qid>/
│           ├── trace.html                   # rendered session transcript (via lash-export)
│           ├── question.json                # the upstream question record
│           ├── prompt.txt                   # full prompt sent to lash
│           ├── answer.txt                   # final assistant text
│           ├── events.jsonl                 # per-turn lash session events
│           ├── session.llm.jsonl            # lash's raw LLM trace log
│           ├── session.db                   # session persistence store
│           └── result.json                  # single-question summary
```

Each response JSONL row matches the upstream `run_eval.py` contract plus extra diagnostic fields:

```json
{
  "question_id": "BlocksWorld_easy_1",
  "successful": true,
  "response_text": "... solution = [...]",
  "model": "anthropic/claude-sonnet-4.5",
  "usage": { ... },
  "attempts": 1,
  "elapsed_seconds": 412.3,
  "iterations": 7,
  "solution_line_present": true,
  "status": "completed",
  "done_reason": "model_stop",
  "domain": "logic",
  "difficulty": "easy",
  "lash": { "execution_mode": "rlm", "context_approach": "rolling_history", "max_turns": 50, "max_output_tokens": 64000 }
}
```

## Evaluating

`bench/longcot/evaluate.sh` resolves a run directory to its responses JSONL and invokes the upstream verifier via `uv run python run_eval.py` inside the vendored repo:

```bash
bench/longcot/evaluate.sh .benchmarks/longcot/runs/<run-id>
bench/longcot/evaluate.sh .benchmarks/longcot/runs/<run-id> --no-fallback
bench/longcot/evaluate.sh path/to/responses.jsonl
```

Upstream prints:

- `correct`, `incorrect`, `failed`, `wrong_formatting`
- `accuracy = correct / (correct + incorrect)`
- `overall_accuracy = correct / total`

Math and chemistry use a Gemini LLM fallback when deterministic checks are inconclusive. Set `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) to enable it; pass `--no-fallback` to skip.

## Notes

- Every question runs in its own isolated `lash::Session` with its own store, events log, and LLM trace. Failures in one question do not taint others.
- Subagent plugins are enabled (`spawn_agent` is available for recursive decomposition) but no benchmark-specific retrieval/tool plugins are installed — LongCoT prompts explicitly forbid external tool use.
- The minimal prompt template adds only an "answer format" reminder; the LongCoT prompt itself is already self-contained.
- `LASH_MAX_OUTPUT_TOKENS` is a per-process env var plumbed into the openrouter adapter; the bench binary sets it once at startup from `--max-output-tokens`.
- Usage ledger aggregates across root and spawned subagents, so `results.json.usage` reflects the full token spend per run.
