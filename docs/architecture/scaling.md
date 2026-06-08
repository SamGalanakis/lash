# Scaling lash as a stateless microservice

**Status:** design. This note records the target deployment shape and the seams
that already make it reachable. lash is *already* a durable-object / actor-per-session
architecture: the durable truth lives entirely behind one async store trait, and
the in-process runtime holds only a rehydratable working copy. Running lash as a
horizontally-scaling microservice is therefore additive work against existing
seams — a networked store implementation, turns under a durable effect host, and
session-agnostic ingress — not a rewrite.

## The core property: no durable state in process memory

`LashRuntime.state` (`RuntimeSessionState`) is a **rehydratable working copy**, not
a source of truth. It is loaded via `RuntimePersistence::load_session` and
refreshed via `refresh_session_graph_from_store`; `Residency` controls whether a
load rehydrates the full graph or just the active path. The other in-memory fields
on `LashRuntime` are either ephemeral per-turn accumulators (`shared_token_ledger`,
drained into `state` at commit) or per-worker coordination (`managed_sessions`,
`managed_turns` — what *this* instance is currently running). None of it is durable
truth.

Durable truth lives behind a single trait, `RuntimePersistence`
(`crates/lash-core/src/store.rs`), whose surface is exactly a durable-object store:

- `load_session`, `load_node` — rehydrate working state;
- `commit_runtime_state` — **compare-and-swap** on the session head;
- `enqueue_queued_work`, `claim_ready_queued_work`, `renew_/abandon_/cancel_*` —
  a durable work queue with `claim_id` / `claim_owner_id` / `claim_token` /
  `claim_fencing_token`;
- `save_/load_session_meta`, `tombstone_nodes`, `vacuum`, `gc_unreachable`.

Any backend that satisfies this trait (and the `ProcessRegistry` /
`LashlangArtifactStore` traits) is a valid store. The backend-agnostic conformance
suite (`lash_core::testing::conformance`) is what guarantees a new backend behaves
identically.

## The model: a durable object per session

```
            ┌─ worker A ─┐      ┌─ worker B ─┐      ┌─ worker C ─┐   (identical, stateless)
  LB  ──►   │ load_session│      │ load_session│      │ load_session│
 (by         │   → turn    │      │   → turn    │      │   → turn    │
 session_id) │ commit (CAS)│      │ commit (CAS)│      │ commit (CAS)│
            └──────┬──────┘      └──────┬──────┘      └──────┬──────┘
                   └──────────────┬─────┴────────────────────┘
                          networked RuntimePersistence
                     (session head + CAS, durable work queue)
```

A request for `session_id` routes to any worker. The worker rehydrates the session
from the store, runs the turn (pure `lash-sansio` compute + replayed effects),
and commits under CAS. Concurrent writers are safe by construction: the head-revision
CAS in `commit_runtime_state` lets exactly one writer win; the loser retries. (The
"head revision conflict" surfaced by the two-process store test *is* this guard
firing — it is the intended multi-writer safety net, not a bug.)

You scale **across sessions**, not within one (see Limits).

## What already supports this

| primitive | role | where |
|---|---|---|
| `lash-sansio` pure `TurnMachine` | compute is deterministic `(state, input) → (state, effects)` | `crates/lash-sansio` |
| `RuntimePersistence` | the single durable store seam; backend-swappable | `store.rs`, conformance suite |
| `commit_runtime_state` CAS on `session_head` | optimistic concurrency for many writers | `store.rs` |
| queue with claim + fencing tokens | exactly-once work pickup by any worker | `queued_work_batches` |
| effect replay keys + `lash-restate` | re-run a turn on another worker, replay effects | `crates/lash-restate` |
| rehydratable `state` + `Residency` | no durable-only memory; reload anywhere | `runtime/session_ops.rs` |

## The gaps to close

1. **Networked store (the headline item).** The shipped store
   (`lash-sqlite-store`) is a *local file*. Multi-process WAL makes it safe for
   many processes on one host, but a distributed fleet needs a networked
   `RuntimePersistence` implementation (Postgres, libsql-server, FoundationDB,
   etc.). This is a new impl behind the existing trait, validated by the existing
   conformance suite — not a rearchitecture.
2. **Turn durability across workers.** The `InlineEffectHost` runs a turn in
   process; a crash mid-turn loses it. Deploying turns under **Restate** (already
   integrated) makes a turn survive worker death via effect replay. "Stateless
   turns" = mandate the durable effect host for the scaled deployment; the seam
   (`RuntimeEffectController` / `ScopedEffectController`) already exists.
3. **Fleet work dispatch.** Either Restate pushes handlers (the `agent-workbench`
   example already does this) or each worker polls the shared queue with claims.
   The claim/fencing columns are already present; only the fleet-level driver wiring
   is deployment work.
4. **Session-agnostic ingress.** External signals must route to whichever worker
   holds the session, so ingress must not be bolted to one session. See
   *Host events* below.
5. **Affinity vs pure reload.** A routing policy choice: `session_id` affinity
   (warm working-copy cache) vs reload-per-request (simpler, more store reads).
   CAS supports either; `Residency` tunes reload cost.

## Inherent limits (domain, not implementation)

- **Single-writer-per-session.** A session is a serial conversation; the CAS head
  enforces one logical writer. You scale across many sessions, not within one. This
  is ideal for multi-tenant workloads; a single session's throughput is bounded by
  the domain.
- **The store is the scaling ceiling.** Whatever networked backend is chosen must
  sustain the CAS commits and queue claims at fleet scale. The store choice *is* the
  scaling story.
- **Long LLM turns** hold or migrate a worker; durable execution (Restate) is the
  answer, at the cost of replay on recovery.

## What does *not* block it

Statelessness is about where durable truth lives, not about the async runtime.
lash-core staying on tokio is a deliberate choice and is orthogonal: each worker is
a tokio service; the durable truth is already 100% external to the process.

## Host events (ingress)

Host events are the session-agnostic ingress that completes the stateless model
(gap 4). A host event is *not* attached to a session: it is recorded to its own
backend-swappable seam and routed to whichever worker holds the interested
session/process. Only the resulting wake is session-coupled.

- **A second store seam.** Subscriptions, occurrences, and delivery reservations
  live behind a `HostEventStore` trait — parallel to `RuntimePersistence`. The
  in-memory implementation does not fan out across workers, so a distributed fleet
  needs a **networked `HostEventStore`** for the same reason it needs a networked
  `RuntimePersistence`. This is the one new seam ingress adds to the deployment.
- **The event carries no session; the subscription carries the target.** An
  emitted event is matched by `source_type` against registered subscriptions, each
  of which names its own target scope (`session_id`). A single event fans out to
  0..N subscribers across any sessions on any workers. Session identity enters only
  as a property of the matched subscription — never of the event.
- **Delivery is an idempotent effect.** Each matched delivery runs through the
  effect controller with a deterministic id (`occurrence_id` derived from
  `source_type` + source key + idempotency key; delivery process id derived from
  `occurrence_id` + `subscription_id`). So delivery is **exactly-once** and
  replay-safe: any worker can run it, and a crash-and-replay (under Restate) cannot
  double-start a trigger process. This is the same durable-execution model the rest
  of the runtime uses, applied to ingress.
- **Only the wake is session-coupled.** When a delivered process wakes an agent,
  that wake lands on the target session's input queue at `EarliestSafeBoundary`
  (see the runtime turn-ordering note). That is the sole point where session
  identity and turn-ordering apply — exactly where they should.

Net for the fleet: route external signals to a runtime-level emit → match →
idempotent effect-delivery → wake. The only addition over the core scaling model is
a second networked store seam; everything else reuses the existing CAS, claim, and
replay machinery. The full host-event design lives in its own note (owned by the
ingress effort); this section covers only its intersection with horizontal scaling.
