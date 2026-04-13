# Terminal Bench 2

This directory is the tracked home for Lash's Terminal Bench 2 setup.

Use the wrapper:

```bash
bench/terminalbench2/run.sh --sample --execution-mode rlm --variant high
```

Or call the underlying script directly:

```bash
scripts/run-terminalbench.sh --sample --execution-mode rlm --variant high
```

Notes:

- The current Lash harness uses `rlm` and `standard`, not `repl`.
- The current Lash context setting is `--context-approach`, not `--context-strategy`.
- Structured run exports now default to `.benchmarks/terminalbench2`.
