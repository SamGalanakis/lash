# ADR: Single Durable Queued-Work Ingress

## Status

Accepted.

## Context

Lash has several sources of deferred runtime work: user turn input, early turn injection, process wakes, host events, timers, and maintenance commands. Splitting these into local CLI queues or separate host-specific queues creates unclear authority: the UI can show or cancel work that the runtime has already claimed, and background work can accidentally look like foreground user work.

## Decision

Lash keeps one durable runtime ingress: **Queued Work**. Every queued payload is stored and claimed through the runtime persistence contract. The CLI never treats local prepared drafts as authoritative queue records.

The CLI projects that single ingress into separate user-visible surfaces:

- queued-turn preview for visible `TurnInput` payloads only,
- process dock for durable runtime process handles,
- diagnostics/status details for background counts and failures.

Slash commands remain CLI host commands and are never queued as model work. Tool availability is configured outside chat slash commands.

## Consequences

- Queue preview is reconstructed from `LashSession::queued_work()`.
- Dispatch claims durable batch ids selected from that snapshot.
- Accepted or claimed queued turns disappear from preview when the runtime claims or starts them.
- Process wakes, host events, timers, and session commands never appear as queued user input.
- Background process work does not promote the foreground footer state to `Working`.
