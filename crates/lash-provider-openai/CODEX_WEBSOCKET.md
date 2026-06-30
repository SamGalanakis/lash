# Codex WebSocket Notes

The Codex provider owns a provider-local WebSocket session cache. The cache is
bounded by `MAX_SESSION_WEBSOCKET_CACHE_ENTRIES`, pruned by idle TTL, and dropped
synchronously because the provider API does not currently expose a lifecycle hook
for closing transport sessions on runtime shutdown or provider removal.

Known design concerns to keep explicit:

- Lash exposes one stable `LlmRequest::session_id`. Upstream Codex carries both a
  `session_id` and `thread_id`; PI uses this gateway path with one id for
  `session-id` and `x-client-request-id`. Lash intentionally uses
  `req.session_id` for both headers until core has a separate thread/turn
  affinity field.
- The provider can only send a cached WebSocket delta when the next full request
  starts with the previous request input plus previous response output. Runtime
  turn-local prompt insertions can break that prefix and force a full-context
  request.
- Auto transport records a short per-session WebSocket fallback cooldown after
  pre-output transport failures. That avoids repeatedly hitting a known-bad
  WebSocket path without turning an explicit `websocket` or `websocket_cached`
  user choice into SSE.

## Minimal Core Continuation Contract

Codex continuation needs a stable/transient distinction before runtime-level
prompt injection can reliably preserve cached WebSocket deltas.

The minimal core-facing contract is:

- Each prompt/history item that reaches a provider is marked
  `stable_for_continuation` or `transient_for_turn`.
- Stable items are durable conversation state: persisted user messages,
  assistant responses, tool calls, and tool results. For a cached delta, the next
  request's stable sequence must begin with the previous request's stable input
  followed by the previous response output, byte-for-byte after provider
  projection.
- Transient items are volatile turn-local context: current-turn reminders,
  scheduler/wake metadata, UI status, retry notes, and runtime hints. They must
  be included in full-context requests when needed, but must not participate in
  the cached-prefix baseline or invalidate a continuation by moving around.
- Non-input request properties that affect model semantics remain part of the
  continuation fingerprint: model, reasoning effort, tools, tool choice,
  instructions, output schema, cache retention, and attachment projections.
  Transient prompt text must not change this fingerprint unless it changes one of
  those semantic request properties.

With that contract, the Codex provider can compute `previous_response_id` deltas
from stable history while still sending transient current-turn context in the
new delta input. Without it, the provider must conservatively fall back to a
full-context WebSocket request on `input_prefix_mismatch`.

## Runtime Test Seam

The CI-safe scripted WebSocket tests live at the provider layer because the
normal Lash RPC/config path does not expose test URLs for the Codex Responses
HTTP and WebSocket endpoints. Adding those URLs to public provider config would
be a behavior surface change. A runtime-level RPC test should use a core test
seam that can inject an in-process provider instance or endpoint override without
serializing test-only fields into user config.
