# Turn cancellation is a first-party work-driver primitive on the keyed-promise seam

Foreground turns need a durable, externally addressable stop request without becoming Runtime
Processes and without adding coordination state to the session store. We therefore define exact
turn cancellation as `TurnAddress { session_id, turn_id }` on `TurnWorkDriver`, alongside (but
separate from) `ProcessWorkDriver`. Session and turn ids are routing identities, not authorization
credentials; every host boundary remains responsible for authentication and authorization.

The primitive is cooperative. A request races the turn's normal completion through a reserved,
first-writer-wins keyed promise. A cancellation winner carries typed request id, source, and reason
evidence; the running or replayed owner feeds that evidence into its internal cancellation token,
assembles `TurnStop::Cancelled`, and commits under the live session-execution lease. A normal
completion seals the same gate before commit, causing later requests to report
`CompletionWonRace`. A second reserved promise publishes terminal evidence after the commit so an
external caller can attach without polling storage. The promise key uses semantic session/turn
identity rather than lease generation: cancellation survives owner loss, while ADR 0029's normal
generation fence prevents the stale owner from committing.

Turn cancellation has three operational layers:

1. `TurnWorkDriver::request_cancel` is the durable cooperative foreground-turn primitive. It can
   unwind cancellable provider/tool waits, but it cannot guarantee that detached tasks,
   subprocesses, or non-cooperative providers have stopped.
2. Runtime Process cancellation remains the existing process event and worker-recovery protocol.
   Foreground turns do not acquire Process identity, ownership, or lifecycle (ADR 0003).
3. Engine invocation cancellation or kill is host-owned break-glass recovery. Per ADR 0019, owner
   destruction is not cooperative evidence and must never be projected as Lash `Cancelled`; the
   authoritative result is unknown unless a live/replayed owner commits one.

The keyed-promise implementation uses the existing `AwaitEventResolver` operations and engine
journal. Reserved `TurnCancelGate` and `TurnTerminal` identities are indexed as control promises:
ordinary durable-wait cancellation does not sweep them, while session deletion revokes them. This
adds no store method (ADR 0016), no Lash-owned effect journal (ADR 0012), no polling/watch path, and
no claim TTL. Restate attaches through `LashDurableWaitWorkflow`, not its Admin API.

We rejected store-side cancellation rows or a lease marker because they add store coordination,
polling, and recovery races; invocation-id cancellation because it leaks engine identity and can
destroy an owner without a Lash result; turns-as-processes because ADR 0003 keeps foreground turns
session-owned; and session-wide cancel-all because it needs an active-turn index and can touch the
wrong or a future turn. A host that offers “stop all visible work” retains the exact active turn
ids it submitted and fans out exact requests.
