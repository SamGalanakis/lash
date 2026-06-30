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
  request. A future core prompt API could mark transient turn-local messages to
  make cached deltas applicable more often.
- Auto transport records a short per-session WebSocket fallback cooldown after
  pre-output transport failures. That avoids repeatedly hitting a known-bad
  WebSocket path without turning an explicit `websocket` or `websocket_cached`
  user choice into SSE.
