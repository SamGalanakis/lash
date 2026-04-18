# Benchmarks

- [terminalbench2](./terminalbench2/README.md): Harbor + Terminal Bench 2 harness for Lash and peer agents.
- [longmemeval-rlm](./longmemeval-rlm/README.md): Native Rust LongMemEval harness that evaluates Lash as an RLM system over the full structured history, closer to `rawwerks/longmemeval-rlm`.
- [longbench-v2](./longbench-v2/README.md): LongBench-style benchmark harness for running Lash over LongBench/LongBench-v2 style datasets and exporting official-eval-friendly prediction files.
- [longcot](./longcot/README.md): Native Rust harness for the [LongCoT](https://github.com/LongHorizonReasoning/longcot) long-horizon CoT benchmark. Builds directly against the `lash` crate, mirrors the reference blog's model/iteration/token settings, and emits JSONL the upstream `run_eval.py` verifier accepts directly.
