# Child-turn and driver stack growth have canonical seams

Nested managed sessions used to poll a child turn directly from the tool call
that created it. Each level therefore re-entered the complete logical-turn
future on its parent's poll stack. The resulting failure mode was poll-stack
depth: ordinary one-level child turns already needed oversized test-thread
stacks, even though boxing showed that no individual future was exceptionally
large.

We designate two runtime boundaries as the only intentional stack-growth
seams. A managed child turn with an owned, shareable effect controller runs in
its own `tokio::spawn` task, giving every nesting level a fresh default-size
task stack. Inside a physical turn, `RuntimeTurnDriver::run` is boxed exactly
once before the event pump. Future growth in child-turn orchestration belongs
behind the task boundary; future growth in the driver belongs behind that one
box. Callers must not add opportunistic boxes around whichever future most
recently exceeded a stack budget.

The child task retains the managed runtime mutex for the whole turn and calls
`publish_from` while the post-turn state is still guarded. The mutex is the
child runtime's single-writer boundary, not a recursion guard, so shortening
its scope would change observation and mutation ordering. The existing event
channel remains the only child-to-parent observation path and preserves FIFO
delivery to the parent's session and activity sinks.

Cancellation uses an abort-on-drop guard around the child task. If the parent
select or turn is cancelled, dropping its child-turn future aborts the task
promptly; normal completion disarms the guard. A child task panic is resumed on
the awaiting parent so it keeps the former panic behavior instead of becoming
a silent or reclassified `JoinError`. Session execution leases, queued-work
claims, and turn-input claims remain owned and settled by the child runtime
under the same guarded turn path.

Handler-scoped effect controllers, whose borrow is tied to an engine
invocation, cannot satisfy Tokio's `'static` spawn contract without replacing
the engine's journal controller. Those controllers retain the inline path to
preserve durable replay semantics; the prepare/abort frame split, deferred
cancellation-message recovery, claim de-duplication, and single driver box
still reduce that path's stack footprint. Hosts that provide an owned shared
controller take the canonical child-task boundary.

The preparation and plugin-abort preamble also lives in separate async helper
frames. Its `SessionReadView` and abort-only locals are destroyed before the
normal driver path clones state into `TurnBoundary`, avoiding transient graph
double-holds. Cancellation reconstructs its message sequence from the turn
boundary only after cancellation wins, and lease-loss cleanup borrows the
original claim vectors instead of keeping error-only clones live through final
commit.
