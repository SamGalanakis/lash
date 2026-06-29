# Deterministic Simulation Harness

## Status

accepted

## Decision

Lash's Deterministic Simulation Harness will be an unpublished `lash-sim`
workspace crate that composes the existing Runtime, Standard Protocol, RLM
Protocol, Agent, provider conformance, persistence conformance, and Durable
Fault Matrix contracts under a boundary-event scheduler. Provider simulation
will exercise real LLM Provider crates through a production-visible
provider-agnostic HTTP-ish transport seam in `lash-llm-transport`, using
Provider Wire Scripts instead of live LLM calls or a primary mock provider.
The v1 Simulated Worker Topology is bounded to in-process worker identities,
lease owners, incarnations, crash/restart/failover, and lease contention; it
models Lash durable effect boundaries, not Restate internals or physical
deployment behavior.

The accepted architecture is a true DST, not merely a deterministic
conformance/replay harness. The implementation is not done until the scheduler
drives at least two sessions' turns concurrently and resolves scripted provider
chunks plus final completions in seeded order; tool and durable-effect
completion paths pass through a real suspended turn; graph acyclicity, single
active Agent Frame, usage monotonicity, and Final Value semantics are executable
oracles; scenario-contract oracles have per-contract evidence or actual named
scenario runs; at least one real discovered regression has been minimized and
promoted to `crates/lash-sim/replays/`; and the generator's fast randomness,
provider mutations, queued-ingress modes, and worker failover are real.

## Why

The surprising part is that simulation belongs below the facade but above
provider/backend mocks: the harness must be deterministic and cheap enough for
randomized search, yet still use real provider serialization/parsing and real
Lash lease/effect semantics. A custom async executor, live provider tests, or a
new fifth scenario family would all create parallel contracts that drift from
the existing architecture.

## Consequences

- `lash-sim` owns generation, scheduling, replay, model-store simulation,
  worker topology, and sim artifacts, but it does not become part of the
  publishable SDK surface.
- `lash-llm-transport` grows a production-visible injectable transport seam;
  provider crates continue to own vendor schemas and parsing.
- Existing scenarios and conformance suites remain authoritative. Shared
  simulation oracles are extracted narrowly instead of importing test modules
  wholesale.
- Confidence lanes extend `scripts/confidence-gate.sh`: fast gets a tiny fixed
  replay/generator corpus; default adds local conformance, backend contention,
  coverage, and targeted mutation; broad adds bounded full-profile simulation,
  scheduled-depth search, and SQLite/Postgres replay evidence without claiming
  full confidence; full means broad semantics plus full critical-crate mutation
  and the true DST interleaving/effect/oracle/replay/generator criteria above.
- Broad confidence is honest bounded evidence. Full confidence is reserved for
  lanes that exercise the true DST criteria, including replayed promoted
  regressions rather than only generated coverage packages or negative fixtures.
