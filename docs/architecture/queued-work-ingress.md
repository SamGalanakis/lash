# ADR: Durable Work And Pending Turn-Input Ingress

## Status

Accepted.

## Context

Lash has several sources of deferred runtime work: user turn input, early turn injection, process wakes, and maintenance commands. Triggers are a separate runtime-level ingress, and timers are host-owned scheduling. User input has UX-visible ownership requirements that queued background work does not: an active steer may be accepted into the live turn, deferred exactly once after interrupt, cancelled before claim, or dispatched as the next idle turn. Background work must not accidentally look like foreground user work.

## Decision

Lash uses two durable ingress paths through `RuntimePersistence`:

- **Pending turn input** stores model-visible user input. Each row carries a stable input id/source key and a `TurnInputIngress`: `ActiveTurn { turn_id, min_boundary }` for input typed while a turn is running, or `NextTurn` for true follow-up work. A pending input is in exactly one state at a time: pending-active, accepted/claimed, deferred-next-turn, cancelled, or completed.
- **Queued work** stores non-user background work. It contains `SessionCommand` mutations and turn-producing `ProcessWake` delivery. User `TurnInput` is not a queued-work payload.

The CLI never treats local prepared drafts as authoritative queue records. It reconciles draft presentation data with pending-turn-input receipts from core, then renders active pending steers separately from queued next-turn input.

Queued payloads have two runtime classes derived from their existing payload variants:

- `SessionCommand`: session mutations such as `RefreshToolCatalog`.
- `TurnWork`: turn-producing non-user work, currently `ProcessWake`.

Core owns scheduling across those classes. A queued drain consumes ready leading `SessionCommand` batches in enqueue order before it claims the next ready `TurnWork` group. Selected batch-id drains follow the same rule: earlier ready session commands are completed first, and the selected ids must still be the next runnable turn-work group. Session commands never become prompt text.

The CLI projects durable ingress into separate user-visible surfaces:

- active-turn steer preview for pending `ActiveTurn` input,
- queued-turn preview for pending `NextTurn` input,
- process dock for durable runtime process handles,
- diagnostics/status details for background counts and failures.

Slash commands remain CLI host commands and are never queued as model work. Tool Catalog membership is configured outside chat slash commands.

## Consequences

- `LashSession::queued_work()` is an admin/introspection view of all pending work classes.
- `LashSession::pending_turn_inputs()` is the user-input view used for queue and active-steer previews.
- Idle dispatch claims durable pending `NextTurn` inputs. Active checkpoints claim only pending `ActiveTurn` inputs anchored to the live turn and admitted by the checkpoint boundary.
- Accepted active inputs stay with the interrupted turn and are never re-rendered or sent again. Unaccepted active inputs are deferred to `NextTurn` once during interrupt finalization.
- Process wakes and session commands never appear as queued user input.
- Triggers and timers never appear as queued work at all; a matched trigger occurrence may start a process, and only that process's wake enters queued work at a safe turn boundary.
- Background process work does not promote the foreground footer state to `Working`.
