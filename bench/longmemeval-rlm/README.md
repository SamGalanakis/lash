# LongMemEval RLM

This benchmark harness evaluates Lash the same way the `rawwerks/longmemeval-rlm` repo evaluates an RLM system:

- one real Lash runtime per LongMemEval question
- full structured history bound into RLM globals
- no `stored` ingest dance
- default configuration: `rlm + rolling_history`
- optional benchmark-local session tools for retrieval over the haystack sessions

Everything runtime-generated lives under ignored `.benchmarks/longmemeval-rlm/`.

The runner does not depend on `lash --provider` state. It expects an API key in `.env`:

- `OPENROUTER_API_KEY` for the default OpenRouter path
- or `OPENAI_COMPATIBLE_API_KEY`

## Dataset presets

The runner can target the same named datasets used in `rawwerks/longmemeval-rlm`:

- `cleaned-s`
  Uses `longmemeval_s_cleaned.json` for the full cleaned `S` set.
- `flash-failures-64`
  Uses the exact `data/longmemeval_s_flash_failures_64.json` subset from the rawwerks repo.
- `discordant-110`
  Uses the exact `data/discordant_110.json` subset from the rawwerks repo.

If you want the exact rawwerks 64-question slice, use `--dataset-preset flash-failures-64`. `--limit 64` on `cleaned-s` is only the first 64 rows, not the same subset.

## Quickstart

```bash
bench/longmemeval-rlm/setup.sh
bench/longmemeval-rlm/run.sh --limit 10
```

Evaluate with the official LongMemEval judge:

```bash
bench/longmemeval-rlm/evaluate.sh .benchmarks/longmemeval-rlm/runs/<run-id>/hypotheses.jsonl
```

## Defaults

- model: `google/gemini-3-flash-preview`
- provider: `openai-compatible`
- execution mode: `rlm`
- benchmark-local session tools: enabled
- batch size: `10`
- per-question context-token budget: `3,000,000`

## Useful flags

```bash
bench/longmemeval-rlm/run.sh --dataset-preset flash-failures-64
bench/longmemeval-rlm/run.sh --dataset-preset cleaned-s --limit 64
bench/longmemeval-rlm/run.sh --question-id e47becba
bench/longmemeval-rlm/run.sh --batch-size 10
bench/longmemeval-rlm/run.sh --max-question-context-tokens 3000000
bench/longmemeval-rlm/run.sh --prompt-profile temporal-observations
bench/longmemeval-rlm/run.sh --execution-mode standard --context-approach observational_memory --await-background-work
bench/longmemeval-rlm/run.sh --no-session-tools
bench/longmemeval-rlm/run.sh --resume
```

## Notes

- The harness is a separate Rust crate under `bench/longmemeval-rlm/runner/`.
- It builds against `lash` directly, not `lash-cli`.
- It reads benchmark API credentials from `.env` instead of the user’s saved Lash provider config.
- Questions run in isolated parallel batches; each question gets its own runtime, store, and artifacts.
- The runner enforces a live per-question context-token budget across both root and delegated child usage. Set `--max-question-context-tokens 0` to disable it.
- `bench/longmemeval-rlm/setup.sh` prefetches the rawwerks `flash_failures_64` and `discordant_110` subset files into ignored `.benchmarks/longmemeval-rlm/data/`.
- The benchmark-local tools are harness-owned, not product tools:
  - `list_sessions`
  - `get_session`
  - `search_sessions`
  - `grep_sessions`
