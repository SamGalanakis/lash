# Durable background process execution

**Status:** implemented. A background process is identically durable however it
is started: start writes durable intent, and a lash-owned `DurableProcessWorker`
is the *sole* executor of every non-terminal process, behind a single-owner
`ProcessLease`. There is no off-lease in-process spawn — out-of-turn starts no
longer run on a detached `tokio::spawn`. The registry's non-terminal rows are the
durable work queue, and a `ProcessWorkRunner` drains them **on poke, on a poll
tick, and once at startup**. So the *first* run of an out-of-turn start is itself
lease-protected and prompt (a poke fires after the start), not merely eventually
recovered by the next restart's sweep. This closes the turn-vs-trigger asymmetry
— processes started by matched host-event occurrences run durably and recover on
crash, exactly like turn-started ones — and removes the double-execution window
the startup-only sweep left open. Timers and recurring jobs are host-owned
scheduling sources: a source owner emits a host-event occurrence with a stored
`source_key`, and every matched subscription starts a deterministic process run
with the same lease-protected execution + recovery. This note records the design
and principle; the *Implementation* section below maps it to the code that
shipped.

## Problem (the asymmetry this resolved)

A background `process` can be started two ways:

- **in a turn** — the model emits `start name(...)` while a turn runs;
- **by a host-event occurrence** — a host-owned source emits an occurrence
  through the runtime router, which matches stored trigger subscriptions by
  `source_type` and `source_key`.

Before this work, only the first was durably re-executed in a Restate
deployment, and the asymmetry was silent.

Durability of execution used to be borrowed from the **turn-local effect
scope**. A turn run with `stream(&sink, scope)` threaded the
Restate-backed scoped controller into process control via a silent
`effect_controller.unwrap_or(current.host.core.effect_controller)` fallback, so a
turn's `start name(...)` scheduled a `LashProcessWorkflow` invocation that
Restate re-invoked on crash — but host-event emission runs **outside** a turn.
It cannot borrow a turn-scoped controller, so process starts must use the
host-supplied runtime effect controller for the occurrence emission. Before the
cutover, the old session-coupled trigger path hit the **build-time** inline effect host in a
Restate deployment (the
Restate-backed controller is constructed per-invocation from a Restate `ctx`, so
it can never be a process-global build-time controller). The inline path
`tokio::spawn`ed the
process and dropped the `JoinHandle`: the run completed
in-process under normal operation, but the spawn held **no lease**, so a recovery
sweep on another node or after a restart could re-run a process already running
(the off-lease/double-exec window), and on crash the in-memory task was gone with
**no prompt re-execution** — only the next restart's startup sweep would re-run
it. Registry rows — record, events, grants, wake inbox — survived a restart; the
*execution* was either off-lease or merely eventually-recovered.

The fix removes that silent fallback **and** the off-lease spawn. Process control
now routes through the controller the host **explicitly wired** for the path, and
a `Start` only *registers* the durable row; the lease-protected
`DurableProcessWorker` — driven promptly by a `ProcessWorkRunner` — is the single
executor for all out-of-turn process work. A durable host never silently
re-executes a process on an inline controller, and no process ever runs without
holding its `ProcessLease`.

## Principle

lash's thesis is that **the runtime is the durable end** for committed session
state and process control records, while the active effect host owns in-flight
effect replay. Restate supplies durable handler history and timers for turn
effects; SQLite supplies the committed session store and process registry.
Process execution follows that split: **lash owns the process durability logic;
the host supplies the backend.**

## What the OpenAI Agents SDK does — and why it is the wrong template

The OpenAI Agents Python SDK draws the boundary in the opposite place. Its
durability primitive is `RunState`: a run interrupts (e.g. a human-in-the-loop
tool approval), the host calls `result.to_state()`, serializes it
(`to_string()`/`to_json()`), **persists it itself**, and later resumes via
`Runner.run(agent, state)`. The SDK owns only the serialization surface and the
resume entry point; the host owns when/where to persist, when to re-drive, and
idempotency. There is no background/trigger/timer process concept — "long
running" means host-driven pause/resume; Temporal integration lives in the
orchestrator, not the SDK.

That is coherent because the SDK is a **stateless request/response agent loop** —
durability genuinely is not its job. Copying it into lash would (a) contradict
the "durable runtime" thesis and (b) push lease/recovery/idempotency onto every
host *per background process*. Right answer for a thin stateless SDK; wrong
answer for a durable runtime with first-class background processes and triggers.

## Rejected: emitter-supplied durable turn scopes

The tempting stopgap is to let the emitter carry a durable scope, mirroring the
turn API. It is the wrong layer:

- it couples a process's durability to **whoever emitted the event**;
- it does not generalize to host timers, whose scheduler should not carry a
  Lash turn scope;
- it conflates "produce start intent" with "execute it durably."

Adding it would ossify an emitter-carries-durability model exactly where the
clean worker-owns-execution-durability model belongs.

## Design: separate intent from execution

**1. Start = durable intent, full stop.** `start name(...)` in a turn and a
matched host-event delivery each do one thing: write a durable process record +
grant to the registry. The rows survive restart. Turn starts carry turn
causality; host-event deliveries carry `CausalRef::HostEvent { occurrence_id }`.
Neither path carries a borrowed execution scope.

**2. Execution = a lash-owned durable worker with a per-process lease +
recovery.** Turns use effect-host replay plus final commit stamps; processes use
registry events plus process leases:

| Unit    | Single-owner lease | Resumable state             | Recovery entry         |
| ------- | ------------------ | --------------------------- | ---------------------- |
| Turn    | effect-host history | host-recorded effects       | workflow handler replay |
| Process | `ProcessLease`     | registry events + journal   | `drive_pending_processes` (poke/poll/startup) |

The `DurableProcessWorker` — the Restate execution authority via
`RestateCoreProcessRunner` / `LashProcessWorkflow` — is the single executor for
**all** non-terminal registered processes, regardless of how they were started.
It:

- claims a per-process execution lease (single-owner, renewable, expiring);
- runs the process through the host-supplied durable controller;
- on startup / deploy / lease expiry, **sweeps the registry for non-terminal,
  unleased processes and (re-)runs them.**

**3. Host supplies the backend; lash owns the durability logic** — exactly as
with turns. The host wires the SQLite registry and runs the worker (under
Restate, the worker handler is a workflow Restate re-invokes). The host does not
re-implement recovery, leasing, or idempotency per deployment.

This deletes the asymmetry: a trigger-started process is identically durable to a
turn-started one, because both are just registered intent that the durable worker
executes. "How it was started" leaves the durability story entirely.

## The primitive (a process-side mirror of the turn machinery)

A generalization of code that already existed for turns — not a new subsystem:

- **`ProcessLease`** — claim / renew / expire, single-owner, so a recovered
  process is not double-run across nodes. Carries
  `schema_version`, `process_id`, `owner_id`, `lease_token`, `fencing_token`,
  `claimed_at_epoch_ms`, `expires_at_epoch_ms`, plus `ProcessLeaseCompletion`.
  The owner / lease-token / fencing-token triple is the distributed
  single-owner contract. Implemented on `SqliteProcessRegistry` (durable,
  fencing CAS) and `TestLocalProcessRegistry` (in-memory).
- **`ProcessRegistry::list_non_terminal()`** — the worklist query: the
  non-terminal rows *are* the durable work queue, alongside the claim / renew /
  complete lease ops.
- **`DurableProcessWorker::drive_pending_processes`** — one lease-protected
  drive: it lists non-terminal processes, claims each lease (skipping any held
  live by another owner), re-checks terminality after claiming, runs the claimed
  process on the worker's wired controller while renewing the lease across the
  execution, then writes the terminal outcome and releases the lease.
  Idempotent by `process_id`: terminal
  processes are never on the worklist, and a process that became terminal between
  the list and the claim is detected and skipped.
- **`ProcessWorkRunner` + `ProcessWorkPoke`** — the loop that *drives*
  `drive_pending_processes` promptly: a `tokio::select!` over a `Notify` (poke)
  and a poll interval, plus one startup drive that folds in what used to be a
  separate boot-time recovery sweep. A `ProcessRunHandle` selects the tier: the
  inline handle delegates to the worker's `drive_pending_processes`; the Restate
  handle fences-submits `LashProcessWorkflow/{process_id}` per non-terminal row.
  The control seam pokes after every successful `Start` — idempotently, since the
  runner skips leased/terminal rows and the durable submit is keyed-idempotent —
  so in-turn-inline, host-event delivery, and other explicit out-of-turn starts
  are all driven the same way.
  The poke handle lives on the runtime host, not the trait-object controller.

## Idempotency

Two layers, both in place. **Start-idempotency**: the registry's `process_id`
uniqueness is the replay boundary for process control issued outside a turn
lease — `ProcessCommandRunner::run` (`process_runners/control.rs`) routes every
command through the explicitly selected controller and pokes the work runner
after a successful `Start`, and a duplicate `Start` re-registers the same row
rather than spawning a second execution. **Execution-idempotency**: the
`ProcessLease` ensures a recovered, non-terminal process is re-run by exactly
one owner. Process execution identity is the persisted `process_id` end-to-end
(the Restate invocation is already keyed `LashProcessWorkflow/{process_id}`); a
retry — a Restate `run` re-invocation or a recovery sweep — must present that
stable id, and an empty/fresh id is rejected loudly
(`DurableProcessWorker::ensure_stable_process_id`), mirroring how
`EffectScope::turn` rejects an empty turn id when scoped controller creation
validates it.

## Non-goals

- Not "make every emitter supply a durable turn scope".
- Not "the host re-implements recovery" (the OpenAI `RunState` model).
- Not exactly-once side effects inside a process beyond what the active effect
  host can replay or dedupe — the same at-least-once-with-idempotency semantics
  apply.

## Implementation

Durability is a property of each execution path, derived from what the host
wired — there is no mode flag. Store traits and effect-host traits carry a
defaulted `durability_tier()` returning `DurabilityTier::Inline`; durable
implementations override it to `Durable`. The runtime never silently runs
nondeterministic work on a non-durable controller picked as a fallback, and a
durable effect host paired with any ephemeral store facet fails **loudly** at
the boundary where the tier is known:

- **Scope boundary** (`turn_loop.rs`, resume/stream turn-scoped): when the
  turn's effect host is `Durable`, all wired store facets must be `Durable`, else
  `DurableStoreRequired { facet }`.
- **Worker boundary** (`DurableProcessWorker::ensure_durable_store_facets`): the
  same check at process-run / runtime rebuild, including the `ProcessRegistry`
  store facet.
- **Facade build** (`lash/src/core.rs::ensure_store_peer_coherence`): store
  peer-coherence only — the per-invocation durable controller is not visible at
  build, so build checks the stores against each other (durable session store
  ⇒ durable attachment + artifact store; durable process registry ⇒ durable
  session store factory + durable host-event store), never the controller.

Out-of-turn starts (host-event deliveries; facade `start`; `api.rs`) write
durable intent and execute via the worker. The silent
`effect_controller.unwrap_or(...)` fallback at `process_runners/control.rs` is
gone — out of turn there is no turn-scoped controller, so the host's explicitly
named runtime effect controller is used for the effect that registers the row. A
`Start` only *registers* the row (the inline off-lease `tokio::spawn` is
deleted), and `ProcessCommandRunner::run` pokes the host's
`ProcessWorkRunner` after a successful `Start`. The runner is wired by the host:
the `lash` facade lazily spawns a default inline runner on first
`session().open()` when a process registry **and** a store factory are present
(it cannot rebuild session runtimes without one), suppressed by an explicit
`with_process_work_runner(...)` and guarded by
`EmbedError::ProcessRegistryWithoutWorkRunner` when the default is disabled with
no replacement; a Restate deployment registers the ingress-client runner at the
serve path (`lash-restate/src/lib.rs`) and hands the core its poke.

An emitter-supplied durability API was **not** added — the worker owns execution
durability, not the emitter.

Intentional cancellation now follows the host-owned ability seam instead of the
durable-start path. `ProcessCancelAbility` receives typed cancel requests with a
source (`HostApi`, `Tool`, or `Lashlang`) and reason; cancel-all first lists live
visible handles, then calls the same cancel-summary path for each. Internal
cleanup (`cancel_unreferenced`) stays on the low-level registry operation.

### Subagent collapse

A subagent spawn is now itself a process that runs a child lash session,
collapsed onto the generic `ProcessInput::SessionTurn` primitive
(`process_runners/session.rs::run_process_session_turn`). `agents.spawn`
(`crates/lash-subagents`) builds a `SessionCreateRequest` (capability → policy,
seed → initial nodes, RLM termination → plugin options, depth ceiling →
hidden tools) plus a `TurnInput`, and emits one `ProcessInput::SessionTurn` —
all deltas are request **config**, not new structure. The bespoke parallel
child-creation path (`rlm.rs::run_child_session`, the dead
`SubagentSessionConfigurator` / `NoopSubagentSessionConfigurator`) is deleted.
Because children now flow through the same generic path Phases A/B made durable,
they inherit provider re-supply, durability, and recovery — which is what
dissolves the `rlm_spawn` provider-inheritance failure: there is no bespoke
route left to drop the live provider handle. Recursion safety
(`MAX_SUBAGENT_DEPTH`, tool-hiding at the depth ceiling) and the `submit_error`
terminal are carried as request config / tool-access, not lost.

## References

- `crates/lash-core/src/runtime/process_worker.rs` — `DurableProcessWorker`:
  `drive_pending_processes`, `ensure_durable_store_facets`, `ensure_stable_process_id`,
  lease-renewing run.
- `crates/lash-core/src/runtime/process_work_runner.rs` — `ProcessWorkRunner`
  (poke + poll + startup loop), `ProcessWorkPoke`, `ProcessRunHandle` /
  `InlineProcessRunHandle`.
- `crates/lash/src/core.rs` — `ProcessWorkRunnerSlot` (lazy default-runner spawn
  on first open), `with_process_work_runner` / `disable_default_process_work_runner`.
- `crates/lash-core/src/runtime/process/model.rs` — `ProcessLease` /
  `ProcessLeaseCompletion`.
- `crates/lash-core/src/runtime/process/registry.rs` — `ProcessRegistry`
  defaulted `durability_tier()` + `list_non_terminal()` and the lease ops.
- `crates/lash-core/src/runtime/session_manager/process_runners/control.rs` —
  explicit-controller routing (the silent fallback removed), register-and-poke
  seam after a successful `Start`, + out-of-turn idempotency comment.
- `crates/lash-core/src/runtime/process/service.rs` —
  `ProcessService::start_from_request`, `ProcessCancelAbility`,
  `ProcessCancelAllRequest`, and typed cancel summaries.
- `crates/lash-core/src/runtime/effect/executor.rs` —
  `InlineEffectHost` / inline controller (stateless; the off-lease
  `tokio::spawn` deleted, `Start` only registers the row, cancel is a durable
  event append).
- `crates/lash-core/src/runtime/turn_loop.rs` — scope-boundary store-facet
  check.
- `crates/lash/src/core.rs` — `ensure_store_peer_coherence` (facade build).
- `crates/lash-core/src/runtime/error.rs` — `DurableStoreRequired { facet }`.
- `crates/lash-core/src/store.rs` — committed session state and turn commit stamps
  (the model the process lease mirrors).
- `crates/lash-subagents/src/rlm.rs` — `agents.spawn` emitting
  `ProcessInput::SessionTurn`.
- `crates/lash-restate/src/tests.rs` —
  `sqlite_trigger_started_process_recovered_after_worker_registry_reopen` and
  `sqlite_process_recovery_reopens_registry_worker_grants_wakes_and_cancel`.
- `crates/lash-core/src/testing/conformance.rs` — process-lease single-owner /
  fencing conformance suite.
