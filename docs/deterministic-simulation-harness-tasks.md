# Deterministic Simulation Harness Tasks

## Status

Backlog for the work deferred after the first provider-transport simulation
vertical slice reached the wholehog-loop implementation target of 90/100.

The authoritative design remains `docs/deterministic-simulation-harness-plan.md`
and `docs/adr/0009-deterministic-simulation-harness.md`. This file turns the
remaining work into concrete tasks with acceptance evidence.

## Completed Baseline

- Production-visible `LlmHttpTransport` seam exists in `lash-llm-transport`.
- OpenAI-compatible Chat Completions and direct OpenAI Responses use the seam.
- `lash-sim` is an unpublished, non-default workspace crate.
- Canonical Provider Wire Scripts drive mock wire data through real OpenAI
  provider crates without live LLM calls.
- Fixed-script runner writes deterministic manifests, script hashes, transcript
  hashes, and per-proof summaries.
- `scripts/confidence-gate.sh fast` runs the fixed-script sim lane.

Baseline proof:

```sh
cargo test -p lash-sim --locked
cargo test -p lash-llm-transport -p lash-provider-openai -p lash-sim --locked
cargo run -p lash-sim --locked -- fixed-scripts --out target/confidence/fast/sim
```

## P0: Finish Provider Slice Hardening

### T1. Enrich Fixed-Script Transcripts

Add redacted request/response forensic data to each fixed-script proof
transcript.

Acceptance:

- Transcripts include redacted request headers, endpoint, method, body shape,
  terminal provider result, terminal error envelope fields, response status,
  response headers, and response event names.
- Secrets and full prompt content are not emitted.
- Manifest schema test asserts the required transcript fields.

Proof:

```sh
cargo test -p lash-sim --locked fixed_script_manifest_schema_contains_required_proofs_and_artifact_fields
cargo run -p lash-sim --locked -- fixed-scripts --out target/confidence/transcript-smoke/sim
```

### T2. Scripted Timeout Parity

Prove response-start and stream-chunk timeout behavior through
`LlmHttpTransport` and `ScriptedLlmHttpTransport`.

Acceptance:

- One response-start timeout proof returns the same timeout kind/code expected
  from production transport timeout handling.
- One stream chunk timeout proof fails during stream consumption without
  committing partial successful output.
- Fixed-script runner records both proofs or a focused test proves the same
  behavior with explicit reasoning why the runner should not carry them.

Proof:

```sh
cargo test -p lash-sim --locked timeout
cargo test -p lash-llm-transport --locked timeout
```

### T3. Codex Transport Boundary Decision

Decide whether `crates/lash-provider-openai/src/codex.rs` is out of scope for
provider simulation or migrate its HTTP/SSE path to `LlmHttpTransport`.

Decision for the first provider-transport simulation slice: Codex OAuth
transport is out of scope. The accepted plan now assigns Codex to a later
Codex/OAuth transport phase because its primary boundary includes OAuth-owned
headers, WebSocket-first execution, SSE fallback, and continuation-cache
behavior. This slice remains limited to OpenAI-compatible Chat Completions and
direct OpenAI Responses flowing through production `LlmHttpTransport`.

Acceptance:

- If out of scope: the plan documents why Codex transport is not part of the
  first provider simulation contract and names the later phase that owns it.
- If migrated: Codex no longer stores or exposes a first-class concrete
  `reqwest::Client` execution seam except as a compatibility wrapper.
- No new provider code may depend on compatibility-only reqwest helpers.

Proof:

```sh
rg -n "with_client|reqwest::Client|send_request|read_response_text" crates/lash-provider-openai crates/lash-llm-transport
cargo test -p lash-provider-openai --locked
```

## P1: Runtime Scenario Integration

### T4. Tiny Runtime/Facade Provider Turn

Add one minimal runtime or facade turn proof using the scripted
OpenAI-compatible provider transport.

Acceptance:

- A real Lash runtime or facade path executes one simple provider-backed turn
  through `ScriptedLlmHttpTransport`.
- The proof asserts at least one runtime/session invariant and one provider
  output invariant.
- The proof does not start the full scheduler/model-store implementation.

Proof:

```sh
cargo test -p lash-sim --locked runtime
```

### T5. Shared Oracle Extraction

Extract narrow reusable oracles from Runtime, Standard, RLM, Agent, provider,
and persistence tests without importing test modules directly.

Acceptance:

- Shared checks live behind crate-owned `testing` APIs or support modules.
- `lash-sim` does not use `#[path = ...]` imports into test trees.
- At least one Runtime and one provider oracle are reused by `lash-sim`.

Proof:

```sh
rg -n "#\\[path|mod tests|testing" crates/lash-sim crates/lash-core crates/lash-protocol-standard crates/lash-protocol-rlm
cargo test -p lash-sim --locked
```

## P1: Provider Matrix Expansion

### T6. Anthropic Provider Transport Migration

Move Anthropic provider execution onto `LlmHttpTransport`.

Acceptance:

- Anthropic request construction/parsing remains provider-owned.
- Production default uses `ReqwestLlmHttpTransport`.
- Canonical Provider Wire Scripts cover at least text streaming, tool use or
  equivalent structured output, non-2xx provider error, and disconnect.

Proof:

```sh
cargo test -p lash-provider-anthropic -p lash-sim --locked anthropic
```

### T7. Google Provider Transport Migration

Move Google provider execution onto `LlmHttpTransport`.

Acceptance:

- Google request construction/parsing remains provider-owned.
- OAuth/device flows may remain outside v1 simulation unless explicitly
  targeted by a workload.
- Canonical Provider Wire Scripts cover text streaming, non-2xx provider error,
  and disconnect.

Proof:

```sh
cargo test -p lash-provider-google -p lash-sim --locked google
```

## P2: Deterministic Simulation Core

### T8. Boundary-Event Scheduler

Implement the scheduler that controls external boundary completions instead of
future polling.

Acceptance:

- Provider `at` timestamps move from metadata to deterministic scheduling.
- Two pending provider/tool/time boundaries can be delivered in seeded order.
- Trace records selected boundary, stable actor alias, event id, and observed
  output.
- No custom async executor is introduced.

Proof:

```sh
cargo test -p lash-sim --locked scheduler
```

### T9. Trace And Replay Script

Add the Simulation Replay Script schema and exact replay command.

Acceptance:

- Failed or selected runs write seed, generator version, script hashes, event
  trace, stable aliases, final summary, and oracle verdict.
- `lash-sim replay <trace>` reproduces the same terminal verdict and delivered
  boundary sequence.
- Incompatible generator/script hashes fail explicitly unless a migration mode
  is requested.

Proof:

```sh
cargo run -p lash-sim --locked -- replay target/lash-sim/failures/<id>/trace.json
cargo test -p lash-sim --locked replay
```

### T10. Generated Multi-Session Workloads

Add the first generated workload family with at least two sessions and provider
scripts.

Acceptance:

- Workload generation is seed and generator-version deterministic.
- The first workload interleaves one turn per session.
- Oracle proves cross-session isolation.

Proof:

```sh
cargo run -p lash-sim --locked -- run --profile fast-random --seeds 32 --max-boundaries 150
cargo test -p lash-sim --locked workload
```

## P2: Persistence And Durable Execution

### T11. Model Store

Add the high-volume deterministic model store needed by generated workloads.

Acceptance:

- Model-store rows cover every runtime facet used by current workloads.
- Workload generation rejects unsupported families instead of silently using
  incomplete rows.
- Store snapshots feed invariant checks.

Proof:

```sh
cargo test -p lash-sim --locked model_store
```

### T12. Durable Effect Boundary

Simulate Lash's durable effect boundary for sleeps, direct completions, and
exec-code results.

Acceptance:

- Scripted durable effects replay by key without re-running local side effects.
- Cancellation and replay are explicit boundary outcomes.
- Effect-host conformance passes for modeled durable-step behavior.

Proof:

```sh
cargo test -p lash-sim --locked durable_effect
```

### T13. Simulated Worker Topology

Model in-process worker identities, lease owners, incarnations, crash/restart,
failover, and lease contention.

Acceptance:

- Two simulated workers can contend for one process/session lease.
- Stale completion from a prior owner/incarnation is rejected.
- Session and process lease oracles pass under crash/restart/failover.

Proof:

```sh
cargo test -p lash-sim --locked worker_topology
```

### T14. Backend Replay Lanes

Replay selected traces through SQLite and full/nightly traces through Postgres.

Acceptance:

- Model store and SQLite replay produce matching oracle verdicts for selected
  traces.
- Postgres replay is integrated with the existing confidence-gate bootstrap or
  configured database path.
- Divergence artifacts identify store/backend operation, trace id, and oracle.

Proof:

```sh
cargo run -p lash-sim --locked -- replay-suite --backend sqlite --profile default
cargo run -p lash-sim --locked -- replay-suite --backend postgres --profile full
```

## P2: Confidence Gate Expansion

### T15. Default Sim Confidence Lane

Extend `scripts/confidence-gate.sh default` with generated seeds and selected
SQLite replay.

Acceptance:

- Default lane writes `target/confidence/default/sim/summary.json`,
  `events.jsonl`, `sqlite-replay.json`, `coverage-tags.json`, and failure
  artifacts when applicable.
- Environment knobs from the plan are supported.
- Existing default checks and e2e posture remain intact.

Proof:

```sh
scripts/confidence-gate.sh default
```

### T16. Full Sim Confidence Lane

Extend `scripts/confidence-gate.sh full` with long randomized simulation,
provider matrix, SQLite replay, and Postgres replay.

Acceptance:

- Full lane writes `target/confidence/full/sim/summary.json`,
  `events.jsonl`, `sqlite-replay.json`, `postgres-replay.json`,
  `provider-matrix.json`, and failure artifacts when applicable.
- Sharding and seed knobs are supported.
- Postgres requires configured URL or the existing Docker bootstrap path.

Proof:

```sh
scripts/confidence-gate.sh full
```

## P3: Test Strength

### T17. Parser And Protocol Property Tests

Add property tests for Provider Wire Scripts, Standard protocol, RLM protocol,
and runtime state machines.

Acceptance:

- Generators cover invalid and valid shapes.
- Shrunk failures become minimized fixtures.
- Tests run deterministically from recorded seeds.

Proof:

```sh
cargo test --workspace --locked property
```

### T18. Mutation Testing For Critical Crates

Add mutation testing for the critical sim, parser, provider, and runtime crates.

Acceptance:

- Mutants in provider wire matching, transport error classification, RLM
  protocol state, and runtime commit rules are killed by tests.
- Surviving mutants are triaged into task backlog or accepted with rationale.

Proof:

```sh
scripts/confidence-gate.sh full
```

### T19. Fault Injection Matrix

Expand generated simulation faults across crashes, retries, lease loss,
duplicate inputs, provider failures, cancellation, and durable effect replay.

Acceptance:

- Fault classes are explicit workload dimensions.
- Every fault class has at least one fixed replay and randomized generation.
- Failure artifacts include the fault class and minimizing command.

Proof:

```sh
cargo run -p lash-sim --locked -- run --profile full-random
```
