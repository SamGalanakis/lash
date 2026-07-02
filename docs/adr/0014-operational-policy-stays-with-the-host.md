# Operational Policy Stays With the Host; Lash Exposes Levers

## Status

accepted

## Decision

Lash ships no shutdown or drain orchestrator. Operational policy — when to stop
admitting work, how long to drain, when to fail over, what to do with
stragglers — belongs to the embedding host, which owns the process, its
signals, and its deployment model. Lash's obligation is that every reasonable
host policy is implementable through explicit, lash-owned capabilities, because
the state those capabilities act on (leases, claims, waits, cached transports,
trace buffers) lives inside lash. The capability set:

- **Lease Timings**: every runtime lease and claim (session execution, effect
  replay, queued work, turn input, process leases) derives its TTL and renewal
  cadence from one host-configurable `LeaseTimings` on the core builder,
  validated against the survive-two-missed-renewals invariant
  (`ttl >= 3 * renew_interval`). The former hardcoded 30s constants are gone.
- **Quiesce and handoff**: `LashSession::park(self)` flushes dirty state
  through a fresh-lease commit and returns a resumable `ParkedSession`;
  `LashCore::resume` rebuilds it. `LashSession::close(self)` is park-without-
  a-handle (flush + discard). Both consume the session and fail with
  `SessionStillInUse` when other live handles exist, making mid-turn quiesce an
  explicit contract rather than a silent partial flush.
- **Transport and sink release**: `Provider::close()` (default no-op; Codex
  drains its websocket session cache with real close frames) and
  `TraceSink::flush()` (default no-op; the OTel sink documents that span-export
  durability is the host provider's duty).
- **Claim and wait handback**: host-facing `abandon_queued_work_claim` /
  `abandon_turn_input_claim` return claimed work immediately instead of waiting
  out the claim TTL, and `revoke_durable_waits` resolves a session's
  outstanding Durable Waits as `Cancelled` without deleting the session.
- **Failover parity**: process leases carry `LeaseOwnerIdentity` and support
  fenced, liveness-gated reclaim exactly like session execution leases, so
  provably-dead co-located holders are recoverable before TTL on every lane.

## Why

A capability audit showed the machinery (fencing, per-turn leases, cancellation,
observation cursors) was first-class while the host-facing lever layer was not:
TTLs were compile-time `pub(crate)` constants, the park/resume quiesce primitive
existed only inside `lash-core`, `close()` silently did less than its name, and
sub-TTL takeover was impossible for the opaque identities every distributed
deployment uses. The tempting fix — a `LashCore::shutdown()` drain loop — would
have moved host policy into the runtime and set the precedent for `health()`,
`readiness()`, and the rest of framework-hood. Lash's thesis is the opposite:
the app owns the outer boundaries; lash owns the turn. The same division that
keeps effect journals inside workflow engines keeps drain policy inside the
host.

## Consequences

- Hosts compose their own drain: stop admitting turns, cancel or await actives,
  `park()`/`close()` sessions, `close()` providers, `flush()` sinks, and exit —
  each step an explicit call, no hidden ordering.
- Failover latency is a host decision (`LeaseTimings`), traded explicitly
  against false-takeover risk, instead of a constant chosen by lash.
- `AwaitEventResolver` gained `cancel_await_events_for_session` with a
  loud-failing default, so durable effect hosts (Restate/Temporal adapters)
  must decide how wait revocation maps onto their engine rather than silently
  ignoring it. The inline registry (and the SQLite/Postgres boundaries that
  reuse it) resolves every outstanding wait for the session as `Cancelled`
  while leaving the session usable. The Restate boundary cannot honor the
  session-wide lever honestly: Restate models each Durable Wait as a promise
  addressable only by its exact per-wait key, those keys are process-scoped and
  carry no session id, and the SDK exposes no session→promise enumeration, so
  cancelling a whole session's waits would require an engine-side index Restate
  does not provide. Rather than fake success, the three Restate resolvers
  (`RestateEffectHost`, its controller, and `RestateRuntimeEffectController`)
  reject the call with a specific
  `restate_await_event_cancel_by_session_unsupported` error that names the
  limitation and the workaround: cancel each known wait individually via
  `resolve_await_event(key, Resolution::Cancelled)`, which the process workflow
  handler resolves durably by its exact promise key.
- Anything lash cannot expose as a lever without becoming an orchestrator
  (signal handling, drain deadlines, readiness endpoints) is documented as host
  territory in the production guide instead of API surface.

## Considered Alternatives

- **`LashCore::shutdown()` orchestrator.** Rejected: drain ordering and
  deadlines are policy; the runtime absorbing them starts the framework slide
  and still could not know the host's grace budget.
- **`Provider::shutdown()` hook alone.** Rejected: without the rest of the
  lever set nothing in core would call it, making it a footgun-by-convention.
- **Accept the status quo (drop everything, TTL recovers).** Rejected after the
  audit: TTL-only release put a fixed 30s floor under every drain and failover
  path and was not a policy the host chose.
