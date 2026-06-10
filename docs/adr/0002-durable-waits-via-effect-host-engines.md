# Durable waits lean on effect-host engines; lash never owns an effect journal

Long-lived processes need to suspend durably (waiting on a signal, a long timer, or a
child process) without holding a worker. We close this by growing the effect-host
contract by exactly one primitive — a durable, one-shot, keyed promise
(`AwaitEvent { key }` plus a resolve seam) — and leaning on whatever engine implements
the contract correctly (Restate today, Temporal or others tomorrow) for journaling,
replay, and suspension economics. All richer wait semantics (named typed signals,
child-process joins, timer wakes) are lash-defined compilations onto that one primitive
with deterministic, occurrence-sequenced keys. Lash never journals effect outcomes
itself.

## Considered Options

- **Lash-owned journal**: record effect outcomes into the process event log and replay
  from it, giving uniform suspension on every backend (sqlite/postgres/inline) and
  demoting engines to transports. Rejected: it reimplements what dedicated
  durable-execution engines already do well, and the entire point of the effect-host
  seam is pluggable durability — the registry tier's weaker guarantees are priced in by
  `DurabilityTier`.
- **Per-semantic contract growth** (`AwaitSignal`, `AwaitProcessTerminal`, …): rejected —
  the probability of a correct third-party engine implementation falls with contract
  surface area; one promise primitive is the smallest thing an engine must get right,
  and new wait flavors then cost zero contract change.

## Consequences

- The inline implementation satisfies the same contract as an in-memory wait over the
  process registry: correct semantics, no suspension economics (a waiting process holds
  a parked future and keeps its lease alive). Long-lived automation belongs on an
  engine-backed tier; this difference is the existing `DurabilityTier` distinction, not
  a bug.
- Signals are named and typed only: declared per-process as event types with payload
  schemas, validated at send time; the unnamed untyped `wait_signal()` is removed.
- Waiting is an observability facet on a running process (wait state on the record,
  mirrored by waiting/resumed events), not a fifth lifecycle status — terminal/lease
  semantics are untouched.
