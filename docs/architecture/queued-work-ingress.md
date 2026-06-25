# ADR: Single Durable Queued-Work Ingress

## Status

Accepted.

## Context

Lash has several sources of deferred runtime work: user turn input, early turn injection, process wakes, and maintenance commands. Triggers are a separate runtime-level ingress, and timers are host-owned scheduling; neither is queued session work. Splitting session-scoped work into local CLI queues or separate host-specific queues creates unclear authority: the UI can show or cancel work that the runtime has already claimed, and background work can accidentally look like foreground user work.

## Decision

Lash keeps one durable runtime ingress: **Queued Work**. Every queued payload is stored and claimed through the runtime persistence contract. The CLI never treats local prepared drafts as authoritative queue records.

Queued payloads have two runtime classes derived from their existing payload variants:

- `SessionCommand`: session mutations such as `RefreshToolCatalog`.
- `TurnWork`: turn-producing work, currently `TurnInput` and `ProcessWake`.

Core owns scheduling across those classes. A queued drain consumes ready leading `SessionCommand` batches in enqueue order before it claims the next ready `TurnWork` group. Selected batch-id drains follow the same rule: earlier ready session commands are completed first, and the selected ids must still be the next runnable turn-work group. Session commands never become prompt text.

The CLI projects that single ingress into separate user-visible surfaces:

- queued-turn preview for visible `TurnInput` payloads only,
- process dock for durable runtime process handles,
- diagnostics/status details for background counts and failures.

Slash commands remain CLI host commands and are never queued as model work. Tool Catalog membership is configured outside chat slash commands.

## Consequences

- `LashSession::queued_work()` is an admin/introspection view of all pending work classes.
- Queue preview is reconstructed by filtering that admin view to visible `TurnInput` batches.
- Dispatch claims durable turn-input batch ids selected from that filtered snapshot; core drains any earlier session commands before claiming the selected turn work.
- Accepted or claimed queued turns disappear from preview when the runtime claims or starts them.
- Process wakes and session commands never appear as queued user input.
- Triggers and timers never appear as queued work at all; a matched trigger occurrence may start a process, and only that process's wake enters queued work at a safe turn boundary.
- Background process work does not promote the foreground footer state to `Working`.
