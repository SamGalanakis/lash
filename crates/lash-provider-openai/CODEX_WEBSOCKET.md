# Codex WebSocket Notes

The Codex provider owns a provider-local WebSocket session cache. The cache is
bounded by `MAX_SESSION_WEBSOCKET_CACHE_ENTRIES`, pruned by idle TTL, and dropped
synchronously because the provider API does not currently expose a lifecycle hook
for closing transport sessions on runtime shutdown or provider removal.

Known design concerns to keep explicit:

- Lash attaches `LlmRequestScope` at every provider request boundary.
  Runtime LLM calls include `session_id`, `agent_frame_id`, and `request_id`.
  Direct calls without a user session get an explicit generated direct scope.
- The provider uses `session_id` for Codex gateway affinity, `request_id` for
  request correlation, and `session_id + agent_frame_id` as its local
  continuation/cache key. A frame switch must not inherit another frame's
  `previous_response_id`.
- The provider can only send a cached WebSocket delta when the next full request
  starts with the previous request input plus previous response output. Runtime
  turn-local prompt insertions can break that prefix and force a full-context
  request.
- Auto transport records a short per-scope WebSocket fallback cooldown after
  pre-output transport failures. That avoids repeatedly hitting a known-bad
  WebSocket path without turning an explicit `websocket` or `websocket_cached`
  user choice into SSE.

## Minimal Core Continuation Contract

Core does not expose provider-specific conversation state. It supplies request
scope only, and scope is always present on provider requests:

- `session_id`: logical Lash session.
- `agent_frame_id`: durable frame/branch within that session.
- `request_id`: one provider call, suitable for correlation/idempotency.

The Codex provider owns the WebSocket mechanics. It may reuse a cached
`previous_response_id` only when the current projected request has the same
non-input fingerprint and its input prefix exactly matches the prior request
input plus prior response output. Otherwise it sends the full context and records
the explicit cache miss reason, such as `input_prefix_mismatch`.

## Runtime Test Seam

The CI-safe scripted WebSocket tests live at the provider layer because the
normal Lash RPC/config path does not expose test URLs for the Codex Responses
HTTP and WebSocket endpoints. Adding those URLs to public provider config would
be a behavior surface change. A runtime-level RPC test should use a core test
seam that can inject an in-process provider instance or endpoint override without
serializing test-only fields into user config.
