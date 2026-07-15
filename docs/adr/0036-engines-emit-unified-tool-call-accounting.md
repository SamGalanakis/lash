# Engines emit unified tool-call accounting outside model projection

Lash has more than one execution engine. The standard engine asks the model for
native tool calls and dispatches them between model completions. The RLM engine
runs Lashlang, whose executor can dispatch tools internally before it returns an
`ExecResponse`. These are different control loops, but they are not different
kinds of tool use. Every completed call is part of the turn's accounting and
must therefore use the standard `SessionEvent::ToolCall` event shape.

The RLM driver emits one accounting event for each tool record returned by a
successful exec effect. It does so in `handle_exec_result`, after inspecting the
complete response for terminal tool control and before any terminal or
nonterminal branch returns. This is deliberately the response-handling seam,
not the live per-call execution seam. Live execution already emits trace and
activity start/completion pairs; emitting there again would duplicate those
channels. More importantly, a durable effect re-drive passes its recorded
`ExecResponse` through the same response-handling seam, so accounting and
attachment reachability are reconstructed after a cold restart.

The engine vocabulary carries three invariants:

1. Every engine emits standard tool-call accounting events.
2. Every engine's terminal outcome carries its full payload. Accounting bounds
   never determine or truncate the terminal control decision.
3. Engine internals reach model context only through that engine's projector.

`SessionEvent::ToolCall` is an accounting and host-observation event. It is not a
conversation or protocol graph node, and RLM trajectory entries remain free of
tool-call records. RLM emission consequently fills `AssembledTurn.tool_calls`,
host and remote turn summaries, and `ExecutionSummary.had_tool_calls` without
changing the bytes projected back to the model. Exec calls continue to consume
no model tokens and add nothing to the ADR 0032 LLM-attempt ledger.

Accounting remains bounded without sacrificing attachment reachability. Large
inline `ToolValue` scalars are replaced in place by an `omitted_bytes` marker,
recursing through success values and failure or cancellation raw values while
preserving the surrounding arrays and objects. A fixed per-exec record cap
collapses the tail into one marker carrying `omitted_records` and
`omitted_failures`; the marker is non-successful exactly when its omitted tail
contains a non-successful record. Attachment references are small, load-bearing
references rather than inline output bytes, so neither bound may remove them.
Tail attachments are carried by the overflow marker and remain visible to the
ordinary tool-output attachment scan.

## Why attachment commit has three sources

The final turn boundary commits the union of three attachment sources. This
union predates this ADR but had no recorded rationale:

1. `pending_manifest_commit_ids` is a local, conservative proxy for durable
   roots that core cannot inspect. In particular, the execution-state snapshot
   is opaque bytes and plugin trajectory nodes may carry plugin-typed values.
   An attachment created during the turn can remain reachable only through one
   of those roots.
2. Attachment references in tool-call outputs are the replay-safe reachability
   backbone. Standard calls have always populated this source; unified RLM exec
   emission now reconstructs exec-internal calls here when a recorded response
   is re-driven in another process.
3. Attachment references in message parts express ordinary message
   reachability.

The union intentionally over-retains some abandoned scratch: source 1 can commit
an attachment that executor code created and then discarded before the turn
ended. The alternative is unsafe because core cannot prove that the attachment
is absent from opaque executor or plugin state. This over-retention is bounded
to the session lifetime, and explicit `SessionAttachmentStore::delete()` drops a
session reference eagerly. Session deletion drops the remaining manifest roots,
after which normal attachment GC reclaims the bytes. The existing three-source
union therefore remains unchanged.
