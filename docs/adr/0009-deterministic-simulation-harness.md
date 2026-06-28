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
  full mutation; full means broad semantics plus full critical-crate mutation.
