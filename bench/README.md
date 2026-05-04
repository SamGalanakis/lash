# Benchmarks

- Runtime/UI/Lashlang profiling helpers live under `scripts/`:
  - `scripts/profile_runtime.py` writes `.benchmarks/runtime-perf/*.json`.
  - `scripts/profile_ui.py` writes `.benchmarks/ui-perf/*.json`.
  - `scripts/profile_lashlang.py` writes `.benchmarks/lashlang-perf/*.json`.
  - `scripts/perfreport.py <report.json>` summarizes or diffs those reports.
- [terminalbench2](./terminalbench2/README.md): Harbor + Terminal Bench 2 harness for Lash and peer agents.
- [appworld](./appworld/README.md): AppWorld MCP harness for evaluating Lash on task-world API workflows.
- [continual-learning-bench](./continual-learning-bench/README.md): Thin local wrapper for running Lash as a cached system inside `pgasawa/continual-learning-bench`.
- [longmemeval-rlm](./longmemeval-rlm/README.md): Native Rust LongMemEval harness that evaluates Lash as an RLM system over the full structured history, closer to `rawwerks/longmemeval-rlm`.
- [longbench-v2](./longbench-v2/README.md): LongBench-style benchmark harness for running Lash over LongBench/LongBench-v2 style datasets and exporting official-eval-friendly prediction files.
- [longcot](./longcot/README.md): Native Rust LongCoT harness — 2,500 long-horizon chain-of-thought problems (logic / cs / chemistry / chess / math) run through Lash's RLM mode. Defaults to GPT-5.2.
