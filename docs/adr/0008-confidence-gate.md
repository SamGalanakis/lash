# Confidence Gate

## Status

accepted

## Decision

Lash has one executable confidence contract: `scripts/confidence-gate.sh`.
The gate has explicit lanes instead of an implicit pile of local commands:

- `fast`: deterministic Runtime, Standard Protocol, RLM Protocol, and Agent
  Scenario harnesses; runtime state-machine property checks; durable
  fault-matrix metadata; and performance guard identity tests.
- `default`: `fast` plus Sqlite backend conformance, coverage blind-spot
  artifacts, and cargo-mutants smoke shards for `lash-core`, `lashlang`,
  `lash-protocol-rlm`, and `lash-protocol-standard`.
- `full`: `default` plus Postgres backend conformance and full cargo-mutants
  over the same critical crates.

Coverage is not a percentage goal. The gate writes LCOV, missing-line text, and
summary JSON under `target/confidence/<lane>/coverage/` so uncovered source is
reviewed as a blind-spot map.

Mutation testing is required for lanes that claim it. `cargo-mutants` absence
fails `default` and `full` unless `LASH_CONFIDENCE_BOOTSTRAP=1` installs the
pinned tool version. Mutation success is never faked as a skipped pass.

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
  manual `default`/`full` dispatch.
- `just confidence`, `just confidence-fast`, and `just confidence-full` are the
  local entry points.
- Missing tools are actionable failures with deterministic bootstrap commands.
  Use `LASH_CONFIDENCE_BOOTSTRAP=1` when a machine should install the required
  cargo subcommands.
- The durable fault matrix lives in
  `crates/lash-core/src/runtime/tests/runtime_scenarios/fault_matrix.rs`; every
  row must point at an executable test or carry a concrete blocked rationale.
