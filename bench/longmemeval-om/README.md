# LongMemEval OM

This benchmark harness is dedicated to the LongMemEval setup used in Mastra's Observational Memory writeup, adapted to Lash's current surface:

- actor model default: `google/gemini-3-flash-preview`
- actor provider default: `openai-compatible` (`https://openrouter.ai/api/v1`)
- context approach: `observational_memory`
- execution mode: `rlm`
- dataset default: `longmemeval_s_cleaned.json`

Use the wrapper:

```bash
bench/longmemeval-om/run.sh --limit 10
```

If you only want to prepare the evaluator and dataset:

```bash
bench/longmemeval-om/setup.sh
```

Then evaluate the generated hypotheses with the official LongMemEval evaluator:

```bash
bench/longmemeval-om/evaluate.sh .benchmarks/longmemeval-om/runs/<run-id>/hypotheses.jsonl
```

Notes:

- The harness auto-downloads the cleaned LongMemEval dataset into `.benchmarks/longmemeval-om/data/`.
- It isolates all session state under `.benchmarks/longmemeval-om/runs/<run-id>/lash-home/`.
- It copies `~/.lash/config.json` into that isolated `LASH_HOME`, so your normal Lash provider setup is reused for the run.
- By default it activates the `openai-compatible` provider entry from that config, which in this repo is the OpenRouter-backed path.
- It keeps `rlm`, but overrides the execution prompt into a strict no-tools/plain-prose benchmark mode so ingest turns return exactly `stored` and answer turns do not improvise filesystem-backed memory.
- Historical sessions are replayed into Lash as transcript-ingestion turns because the current CLI surface accepts user turns, not raw historical assistant-message injection.
- The official evaluator is cloned into `.benchmarks/longmemeval-om/vendor/LongMemEval/` by `setup.sh`.
- The official evaluator still uses an OpenAI judge model by default, so `OPENAI_API_KEY` must be set when you run `evaluate.sh`.
