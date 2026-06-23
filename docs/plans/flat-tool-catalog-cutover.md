# Flat Tool Catalog — Clean Cutover Plan

Wholehog cutover to the design recorded in
[ADR 0005](../adr/0005-tool-catalog-membership-replaces-availability-tiers.md).
No shims, no migration adapters, no interim tiers. Lash is pre-1.0
(`v0.1.0-alpha`), so breaking the persisted/wire shape of tool availability is
acceptable and expected — we delete the old model rather than translate it.

## End state

- **Tool Catalog is a flat callable set.** Membership is the only availability
  fact. A tool is in the catalog (callable) or it does not exist to the model.
  There is no `ToolAvailability` enum, no `Off`/`Searchable`/`Callable`/
  `Showcased`.
- **Catalog assembly is propose/remove by trusted plugins.** A `ToolProvider`
  contributes members; any plugin may remove a member. Authority hiding and
  plan-mode gating are removals, not tiers.
- **Presentation is protocol-owned.** Standard presents members via the LLM
  Provider's native tools array and renders no prompt tool docs. RLM renders
  *all* catalog members as full prompt docs (call-path idiom). No "showcased"
  bit — being a member *is* being presented.
- **Deferral is RLM-only.** A host `DeferredToolResolver` resolves a call-path
  absent from the link-time Lashlang Host Environment into a Tool Grant (+ Tool
  Execution Binding) or `NotAvailable`. `gather → resolve → link`; each
  resolution frozen as a per-link replayable effect; the flat catalog never
  mutates from resolution; replay never re-resolves.
- **Lash ships no discovery.** `lash-plugin-tool-discovery` is deleted. Its
  ranking index and the catalogue-preview formatter move into `lash-cli` as the
  reference MCP-discovery example. `search_tools` is a host tool, not a
  primitive.

Out of scope (leave untouched, verify orthogonal): `ToolActivation`
(`Always`/`Internal` — a who-may-call axis, not an availability tier) and
`SchemaProjectionOverride` (a request-size lever, independent of this work).

## Cutover order

Each phase must leave the workspace compiling (`cargo build --workspace`).

### Phase 0 — Reshape the catalog core (`lash-sansio`)

Foundation; everything else follows. In `tool_contract.rs` / `tool_catalog.rs`:

- Delete `ToolAvailability`, `ToolAvailabilityConfig`, `is_searchable/_callable/
  _showcased`, `availability_override`, `effective_availability`,
  `is_default_tool_availability_config`.
- `ToolManifest`: drop `availability` and `availability_override` fields and
  their serde. Membership comes from being in a provider's manifest list, not a
  field.
- `ToolCatalog`: delete `showcased_tools(_iter)`, `searchable_tools_iter`,
  `omitted_tools_iter`, `omitted_tool_count`, `prompt_tool_docs`,
  `rendered_prompt_tool_docs`, `tool_list_notes`. `callable_tools(_iter)`,
  `tool_names`, and `model_tool_specs` now range over **all** entries (no
  filter). `tool_availability`/`has_callable_tool` collapse to membership tests
  (`contains`).
- `ToolCatalogContribution`/`ToolCatalogOverride`: replace the
  `availability: Option<ToolAvailability>` override with a membership removal —
  e.g. `ToolCatalogContribution { remove: Vec<String> }`. Drop `tool_list_notes`
  (notes become ordinary prompt contributions authored by the host).
- `prompt.rs`/`plugin.rs`: `PromptContribution::requires_tool(name)` gates on
  membership only; delete the `minimum_availability` parameter and the
  `ToolGate` availability field.

### Phase 1 — Core wiring (`lash-core`)

- `plugin/session_obj/tools.rs::resolve_tool_catalog`: authority-hidden tools
  (`tool_access.hides`) become **removals** instead of `Off` overrides.
- `plugin/hooks.rs`, `tool_provider/session.rs`, `plugin/runtime_host.rs`:
  rename `set_tools_availability(names, Option<ToolAvailability>)` →
  `set_tool_membership(names, present: bool)` (or `add_tools`/`remove_tools`).
- `tool_registry/{registry_types,state,restore_execute}.rs`: drop availability
  from stored manifest state; membership only.
- `runtime/turn_driver/tool_catalog.rs`, `session.rs`, `testing/mod.rs`,
  `tool_dispatch/preparation.rs`: update to the membership API.
- `lib.rs`: drop `ToolAvailability` from the re-export surface.

### Phase 2 — Protocols

- **Standard** (`lash-protocol-standard`): remove `omitted_tool_count: 0` and any
  availability references; it already renders no tool docs. No behavioral change.
- **RLM** (`lash-protocol-rlm`):
  - Delete `rlm_tool_catalog` (the `Searchable → Callable` promotion),
    `catalogue_notes`, `has_catalogued_tools`, and the `"search_tools"`
    name special-casing in `tool_catalog.rs`.
  - `rlm_prompt_tool_docs` renders **all** catalog members (drop the
    `is_showcased` filter) under their Lashlang call-path.
  - `control_tools.rs`, `driver.rs`: drop availability references.

### Phase 3 — Deferred resolution (`lash-lashlang-runtime` + `lash-protocol-rlm`)

- Add to `lash-lashlang-runtime`:
  ```rust
  pub enum Resolution { Resolved(ToolGrant), NotAvailable } // ToolGrant carries the Tool Execution Binding
  #[async_trait] pub trait DeferredToolResolver: Send + Sync {
      async fn resolve(&self, path: &str) -> Resolution;
  }
  ```
- Linking: a `gather → resolve → link` pass around the synchronous
  `LinkedModule::link`. Parse, collect unresolved call-paths, `resolve` the
  unknowns (batched/concurrent — do not fail-fast on the first), fold
  `Resolved` grants into the `LashlangHostEnvironment`, then link. `NotAvailable`
  (and no-resolver) leaves the symbol unresolved → a clean model-visible link
  error.
- Record each resolution as a **per-link replayable effect** keyed by
  call-path within the execution scope, capturing the Tool Grant, its Tool
  Execution Binding, and negative `NotAvailable` results. Replay/recovery
  applies the record; the resolver is never called twice for the same link.
  Reuse the existing replay-key / effect-outcome machinery
  (`tool_provider.rs` replay keys; `runtime/effect`).
- The resolver is wired by the RLM protocol plugin from host-provided
  configuration; lash-core stays ignorant of it. The flat catalog is **not**
  mutated; resolution is link-scoped only.

### Phase 4 — Delete the discovery crate, relocate to CLI

- Remove `crates/lash-plugin-tool-discovery` and its workspace membership.
- Remove it from `lash-standard-plugins` (`Cargo.toml` + `src/lib.rs`).
- Move `ToolDiscoveryIndex` (BM25 / semantic / RRF) and the catalogue-preview
  formatter into `lash-cli` as example modules. The `semantic-tool-search`
  feature and `model2vec_rs` dependency follow them (feature-gated in the CLI).
- Build the reference MCP loop in `lash-cli`: enumerate MCP tools via
  `lash-plugin-mcp` → author a catalogue-preview `PromptContribution` →
  expose `search_tools` over the index → register a `DeferredToolResolver`
  that resolves MCP call-paths to a Tool Grant + Tool Execution Binding.

### Phase 5 — Downstream tool definitions and plugins

- `lash-llm-tools`, `lash-tools/web/{fetch_url,web_search}`: delete
  `with_availability(...)` calls; tools are members by default.
- `lash-plugin-plan-mode`: gating via `set_tool_membership`/removal.
- `lash-autoresearch` (`runtime/commands.rs`, `runtime/tools.rs`): toggle its
  tools via membership instead of `set_tools_availability`.
- `lash-plugin-process-controls`, `lash-subagents/rlm_support.rs`: drop
  availability references.

### Phase 6 — Remote protocol wire shape

- `lash-remote-protocol` (`core_conversions/tools.rs`, `protocol/tools.rs`,
  `protocol/prompt.rs`): remove availability from the serialized tool shape and
  from prompt-gate conversion. Clean break — bump the protocol version marker;
  do not translate the old field.

### Phase 7 — Tests, perf, docs

- Update/remove availability-centric tests across `lash-sansio`,
  `lash-core/tool_registry`, `lash-subagents`, `lash-protocol-rlm`,
  `lash-perf/runtime_perf/providers.rs`.
- Add: an RLM e2e that resolves a deferred call-path and a **replay test**
  proving a re-driven turn reuses the recorded resolution (and recorded
  `NotAvailable`) without calling the resolver again.
- Refresh `docs/tools*.html` / `docs/rlm.html` narrative for the flat model.

## Verification gates

1. `cargo build --workspace` after every phase.
2. `cargo test --workspace` green; deleted-concept tests removed, not skipped.
3. RLM e2e: deferred resolve succeeds; `NotAvailable` surfaces a clean model
   error; replay of both is deterministic with zero resolver re-calls.
4. CLI MCP example: discover → preview → `search_tools` → call → resolve →
   execute, end to end.
5. Perf snapshot: confirm the flat catalog keeps the request `tools` array and
   RLM prompt stable across turns (no resolution-driven churn → cache-friendly).

## Risks and mitigations

- **Wire/persistence break.** Remote workers and persisted sessions carrying
  availability fields will not load. Accepted for alpha; coordinate the
  remote-protocol version bump (Phase 6) with any deployed workers.
- **Replay determinism of the new resolution effect.** Highest-risk new code.
  Gate on the Phase 7 replay test before merge.
- **RLM prompt bloat.** Rendering *all* members as full docs punishes a host
  that over-populates the resident catalog. Mitigation is doctrine, not code:
  the resident catalog is the small documented core; the long tail is deferred.
- **`ToolActivation` entanglement.** Confirm `Internal` activation is a
  who-may-call axis independent of membership before deleting availability; if
  it secretly encoded a tier, fold it into the membership/visibility model
  explicitly rather than leaving a vestige.
