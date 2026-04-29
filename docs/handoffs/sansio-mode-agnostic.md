# ADR: Mode-Agnostic Sans-IO Boundary

**Status:** Implemented

**Date:** 2026-04-26

## Context

`lash-sansio` owns the protocol state machine for a turn. It must not know about any concrete execution mode such as RLM, rlmpure, or Standard. Mode-owned vocabulary lives outside the sans-IO crate:

- Standard mode uses the unit protocol shape.
- RLM and rlmpure share `lash_rlm_types::RlmModeProtocol`.
- The host crate, `lash`, erases mode-specific data at request and persistence boundaries.

## Decision

`lash-sansio` is parameterized by `ModeProtocol`, whose associated `Event` and `Termination` types are concrete for the active mode. Sans-IO turn execution remains typed and does not JSON-roundtrip mode data in the hot path.

`lash` stores mode-owned session events through `ModeEvent { mode_id, payload }`. Existing untagged RLM event records are accepted on load and normalized to `mode_id = "rlm"` when reserialized. Mode-specific code decodes only the events for its own mode.

`SessionCreateRequest::mode_extras` is open-ended:

```rust
ModeExtras {
    mode_id: ExecutionMode,
    payload: serde_json::Value,
}
```

Typed helpers serialize and deserialize extras at the request boundary. RLM and rlmpure extras are owned by `lash-rlm-types`.

Runtime turn options are mode-owned via `ModeTurnOptions`; RLM termination is represented as `ModeTurnOptions::Rlm(RlmTermination)`. This keeps the base turn input from carrying RLM-named fields.

## Consequences

- New modes do not need new `ModeExtras` enum variants.
- Persisted mode events carry an explicit `mode_id`.
- RLM vocabulary is imported from `lash_rlm_types`, not from top-level `lash`.
- Mode session APIs use neutral names such as `apply_mode_globals_patch`.
- Legacy RLM records remain readable through the load-time deserialization path, but new writes use the tagged mode-event envelope.
