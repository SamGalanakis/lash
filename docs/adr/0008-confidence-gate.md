# Confidence Gate

## Status

accepted

## Decision

Lash has one executable confidence contract: `scripts/confidence-gate.sh`.
The gate has explicit lanes instead of an implicit pile of local commands:

- `fast`: deterministic Runtime, Standard Protocol, RLM Protocol, and Agent
  Scenario harnesses; runtime state-machine property checks; durable
  fault-matrix metadata; and performance guard identity tests.
- `default`: `fast` plus Sqlite backend conformance, production-backed backend
  contention evidence, coverage blind-spot artifacts, and targeted
  cargo-mutants evidence for high-risk direct/model and
  deterministic-simulation paths.
- `broad`: bounded broad evidence. It runs a full-profile generated simulation
  under explicit seed/boundary budgets, Postgres conformance when an env URL or
  Docker bootstrap is available, cross-backend replay for every generated trace
  and every minimized failing-regression trace, and targeted mutation. It is
  not a true full confidence claim.
- `full`: true full confidence. It includes broad semantics and full
  cargo-mutants over the same critical crates; the lane refuses non-full
  mutation scopes.

Coverage is not a percentage goal. The gate writes LCOV, missing-line text, and
summary JSON under `target/confidence/<lane>/coverage/` so uncovered source is
reviewed as a blind-spot map.

Mutation testing is required for lanes that claim it. `cargo-mutants` absence
fails `default`, `broad`, and `full` unless `LASH_CONFIDENCE_BOOTSTRAP=1`
installs the pinned tool version. Mutation success is never faked as a skipped
pass, and a targeted bounded run is never labeled as `full`.

## Why

Line coverage and flaky end-to-end-only tests do not establish confidence for
Lash's contracts. The high-value risks are invalid runtime states, durable
replay errors, duplicate ingress, retries, cancellation, lease loss, provider
failures, and backend drift. A single gate makes those risks visible and gives
CI and local development the same language for confidence.

## Consequences

- PR CI runs the `fast` lane after workspace tests to prove the confidence
  contract stays wired.
- The `Confidence` workflow runs `full` on a weekly schedule and supports
  manual `default`/`broad`/`full` dispatch.
- `just confidence`, `just confidence-fast`, `just confidence-broad`, and
  `just confidence-full` are the local entry points.
- Missing tools are actionable failures with deterministic bootstrap commands.
  Use `LASH_CONFIDENCE_BOOTSTRAP=1` when a machine should install the required
  cargo subcommands.
- The durable fault matrix lives in
  `crates/lash-core/src/runtime/tests/runtime_scenarios/fault_matrix.rs`; every
  row must point at an executable test or carry a concrete blocked rationale.
- `sim/backend-contention/backend-contention.json` records deterministic
  `RuntimePersistence` lease contention, stale completion fencing, reopen, and
  dead-owner reclaim evidence through SQLite and, when Postgres is configured,
  `lash-postgres-store` production-facing session-store APIs.
