# Performance and stack profiling refresh: PR material

## Summary

This refresh replaces the stack profiler's broad but shallow default with one
composed turn geometry, makes that geometry a required CI stack guard, and adds
deterministic runtime timing scenarios for the alpha.100–105 turn-lifecycle
paths. All scenarios use the test provider and local stores; none use provider
tokens or external services.

The calibration policy is **the measured release median plus 25%, rounded up
to a stable guard value**. The stack crash envelope uses the smallest passing
configured Tokio worker stack plus 25%. Measurements were taken on the code in
this branch with five measured runs, one warmup, and twelve turns per runtime
scenario. The timing calibration deliberately used the later, loaded-host run
so the checked-in wall-clock guards retain their headroom on a shared runner.

## Instrument audit

| Instrument | What it measured before this refresh | Invisible alpha.100–105 layers | Coverage added here |
| --- | --- | --- | --- |
| `scripts/profile_runtime.py` / `lash-perf` | Synthetic, non-inference runtime sessions. It reports phase and total wall time, allocator deltas, RSS/HWM, session shape, and configured worker-stack metadata. Runtime budgets live in `runtime_perf/report.rs`, not the JSON budget file. | No synchronous cancellation start-gate retry, live per-attempt cancel round trip, active ingress claim/projection, or trigger occurrence-to-delivery timing. Its stack field described configuration, not consumption. | Added `turn_start_gate`, `turn_cancel_round_trip`, and `ingress_claim_projection`; extended `rlm_trigger_mail_pipeline`; added required named phases and median phase-time guards. Added `deep_turn_composition` for the stack lane. |
| `scripts/profile_runtime_stack.py` | Ran every synthetic runtime scenario in fresh processes over configured Tokio worker-stack sizes. It found the first non-crashing configured size and verified that the report accounted for that size. It was a crash-boundary envelope, not a sampled stack high-water mark, and CI did not run it. | The default scenarios predated the parent/child/tool/cancel/ingress/wait composition that caused the child-session SIGABRTs. There was no checked-in per-scenario budget or enforcement mode. | The default is now `deep_turn_composition`. Added checked-in per-scenario budgets, `--enforce-budgets`, and a fast `--budget-only` CI mode. |
| `scripts/profile_lashlang.py` | Swept release/debug Lashlang example modes and scenarios for time, allocations, allocation bytes, and interpreted instruction counts. Its stack result was the applied `RUST_MIN_STACK`/process limit, not consumed stack. | Runtime turn composition, cancellation, ingress, triggers, process observation, and attempt reset are outside the Lashlang example boundary. | Recalibrated all three JSON guards from the exact nightly sweep; report rendering now shows the observed maxima and budgets. |
| `scripts/perfreport.py` | Rendered or diffed runtime, UI, Lashlang, aggregate guard, and dhat reports. It performed no measurement or enforcement. | A standalone runtime-stack matrix was rejected as an unknown format; the new phase guard results were only discoverable in raw JSON. | Added standalone stack-envelope rendering plus runtime phase-timing guard and Lashlang guard-max summaries. |
| `scripts/perf_guard_budgets.json` | Contained only three broad Lashlang ceilings. | No runtime stack geometry had a checked-in crash budget. Runtime timing budgets remained in the Rust report's existing guard pattern. | Added the deep-composition stack measurement, budget, and policy metadata; refreshed the three Lashlang values. |
| `scripts/ci-stack-budget.sh` | Ran the Lashlang stack test, `lash-runtime` tests named `stack_budget`, and one subagent child test with a 2 MiB process/thread setting. | It did not run either child-session geometry that had needed an explicit 8 MiB test thread, and it never drove the composed production turn future. A SIGABRT could therefore appear first in a regular shard. | Builds `lash-perf` and runs the deep-composition scenario in a fresh process at its checked-in worker-stack budget. A stack abort now fails this job before the general test shards. |
| CI `Stack budget` job | Installed Rust/protoc, restored the debug cache, and invoked only `ci-stack-budget.sh`. | Its coverage was exactly the shell script's stale test selection. | The same job now transitively enforces the deep runtime-stack envelope and produces `.benchmarks/runtime-stack/ci-deep-turn-composition.json`. |

Still intentionally outside these deterministic local profilers: concrete
Restate SDK journal/awakeable implementation cost, runtime-wide process-change
feed fan-out (the existing process-list stress covers registry reads, not
observer delivery), and the `ModelAttemptReset` retry/retraction path (the test
provider does not retry). Their new generic turn-future layers are present in
the composed stack shape where applicable, but service-specific timing needs a
separate Restate or retry-capable harness rather than invented local timings.

## Deep stack geometry and calibration

`deep_turn_composition` runs an identified parent turn with an active ingress
item already queued, the live cancellation watcher, an RLM process/tool loop,
timer and await-event waits, and a spawned child session whose own turn runs a
tool and process loop before the parent completes. Each stack size is tested in
a new process so an abort is data rather than a lost matrix run.

Debug crash envelope:

- 1,769,472 bytes and below: abort/fail.
- 1,835,008 bytes (1.75 MiB): first passing configured stack.
- 1,900,544, 1,966,080, 2,031,616, and 2,097,152 bytes: pass.
- Guard: 2,293,760 bytes (2.1875 MiB), exactly 1,835,008 + 25%; budget-only run passes.

This is a **minimum-passing crash envelope**, not a claim that 1,835,008 bytes
were all consumed. It catches the relevant stack-overflow/SIGABRT class because
the composed future is executed on a worker with the configured bound.

## Runtime timing and allocation calibration

| Scenario | Measured total median | Measured hot phase median | Total guard | Phase guard | Total / steady allocation median | Allocation guards |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| turn start gate (inline peek through two deterministic retries) | 1,274.951 ms | `turn_cancel.start_gate`: 939.643 ms | 1,600 ms | 1,200 ms | 68,619,516 / 4,720,693 B | 90,000,000 / 6,000,000 B |
| cancel request → token fire → terminal seal | 243.858 ms | `turn_cancel.request_to_token_to_seal`: 45.754 ms | 310 ms | 60 ms | 65,436,012 / 4,461,512 B | 85,000,000 / 6,000,000 B |
| ingress enqueue → claim → next-request projection | 364.830 ms | `turn_input_ingress.enqueue_to_claim_to_projection`: 153.060 ms | 460 ms | 195 ms | 141,495,058 / 10,745,794 B | 180,000,000 / 14,000,000 B |
| trigger occurrence → delivery | 409.025 ms | `trigger.occurrence_to_delivery`: 42.976 ms | 520 ms | 55 ms | 206,108,424 / 15,650,940 B | 260,000,000 / 20,000,000 B |

The trigger scenario's existing guards changed from 128,000,000 to
260,000,000 total allocation, 64,000,000 to 20,000,000 steady-turn
allocation, and 2,000 ms to 520 ms total wall time. The first change absorbs
the now-measured trigger lifecycle work; the latter two tighten previously
broad ceilings. The other three scenarios and all four named-phase time guards
are new.

The full 39-scenario guard also found five allocation ceilings invalidated by
the alpha.100–105 code shape. Their five-run total / steady-turn medians and
old → new guards are:

- `rlm_async_tool_completion`: 113,277,190 / 8,319,629 B;
  96,000,000 → 145,000,000 total and 64,000,000 → 11,000,000 steady.
- `rlm_process_async_tool_completion`: 231,801,232 / 18,091,766 B;
  160,000,000 → 290,000,000 total and 128,000,000 → 23,000,000 steady.
- `rlm_large_tool_catalog`: 4,457,889,543 / 343,433,271 B;
  1,000,000,000 → 5,600,000,000 total and 750,000,000 → 430,000,000 steady.
- `rlm_oblique_stack_mix`: 1,559,621,281 / 129,459,894 B;
  1,500,000,000 → 1,950,000,000 total and 1,000,000,000 → 165,000,000 steady.
- `tool_discovery_search`: 1,839,812,745 / 134,916,680 B;
  1,500,000,000 → 2,300,000,000 total and 1,000,000,000 → 170,000,000 steady.

The same full run caught one stale correctness fixture before enforcement:
`rlm_oblique_stack_mix` initialized `candidate_ids` from an untyped empty list,
which current Lashlang correctly retained as a null/any union. Seeding it from
the first typed search-result comprehension keeps the fixture `list[str]` and
lets the guard measure the intended path again.

## JSON guard calibration

The exact 2,500-iteration nightly Lashlang sweep produced 210 perf rows and one
aggregate profile row. Maxima and checked-in changes:

- Allocated bytes/iteration: measured 1,739,210.8 (`phase_breakdown` /
  `large_data`); guard 2,000,000 → 2,200,000 (+200,000).
- Allocations/iteration: measured 9,349.011 (`phase_breakdown` /
  `large_data`); guard 50,000 → 12,000 (-38,000).
- Instructions/iteration: measured 8,936; guard 500,000 → 12,000 (-488,000).
- Runtime deep stack: no previous JSON guard → measured 1,835,008 and guarded
  at 2,293,760 bytes.

The refreshed Lashlang sweep passes all three new ceilings.
