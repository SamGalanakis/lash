#![allow(clippy::items_after_test_module)]

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, RwLock};

use serde::{Deserialize, Serialize};

use crate::{
    PreparedToolCall, ProgressSender, ToolCall, ToolContext, ToolContract, ToolManifest,
    ToolPrepareCall, ToolProvider, ToolResult,
};

const PLUGIN_SOURCE_ID: &str = "plugins";

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ToolSourceHandle {
    id: String,
}

impl ToolSourceHandle {
    pub(crate) fn new(id: impl Into<String>) -> Self {
        Self { id: id.into() }
    }

    pub(crate) fn as_str(&self) -> &str {
        &self.id
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolStateEntry {
    manifest: ToolManifest,
    /// True when this tool was not resolvable from any registered source at
    /// export time (e.g. a detached MCP server). Orphaned entries keep their
    /// last-known manifest, surface as [`crate::ToolAvailability::Off`], and
    /// rebind automatically when a source re-advertises the same (name, id).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    orphaned: bool,
}

impl ToolStateEntry {
    #[cfg(test)]
    pub(crate) fn new(manifest: ToolManifest) -> Self {
        Self {
            manifest,
            orphaned: false,
        }
    }

    /// The manifest as exposed to callers. Orphaned entries are known to the
    /// session but unavailable until their source returns, so their public
    /// availability is always forced to `Off`.
    pub fn manifest(&self) -> ToolManifest {
        let mut manifest = self.manifest.clone();
        if self.orphaned {
            manifest.availability_override = Some(crate::ToolAvailability::Off);
        }
        manifest
    }

    fn stored_manifest(&self) -> &ToolManifest {
        &self.manifest
    }

    pub fn is_orphaned(&self) -> bool {
        self.orphaned
    }
}

#[derive(Clone, Debug, Default)]
pub struct ToolState {
    generation: u64,
    tools: Arc<BTreeMap<String, ToolStateEntry>>,
}

impl ToolState {
    pub(crate) fn new(generation: u64, tools: BTreeMap<String, ToolStateEntry>) -> Self {
        Self {
            generation,
            tools: Arc::new(tools),
        }
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn with_generation(mut self, generation: u64) -> Self {
        self.generation = generation;
        self
    }

    pub fn tool_manifests(&self) -> Vec<ToolManifest> {
        self.tools.values().map(ToolStateEntry::manifest).collect()
    }

    pub fn get(&self, name: &str) -> Option<&ToolStateEntry> {
        self.tools.get(name)
    }

    pub fn manifest_mut(&mut self, name: &str) -> Option<&mut ToolManifest> {
        Arc::make_mut(&mut self.tools)
            .get_mut(name)
            .map(|entry| &mut entry.manifest)
    }

    pub fn contains(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    pub fn len(&self) -> usize {
        self.tools.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &ToolStateEntry)> {
        self.tools
            .iter()
            .map(|(name, entry)| (name.as_str(), entry))
    }

    pub fn set_availability(
        &mut self,
        name: &str,
        availability: Option<crate::ToolAvailability>,
    ) -> Result<(), ReconfigureError> {
        let Some(entry) = Arc::make_mut(&mut self.tools).get_mut(name) else {
            return Err(ReconfigureError::Validation(format!(
                "unknown tool `{name}`"
            )));
        };
        entry.manifest.availability_override = availability;
        Ok(())
    }

    pub fn retain(&mut self, mut keep: impl FnMut(&str, &ToolStateEntry) -> bool) {
        Arc::make_mut(&mut self.tools).retain(|name, entry| keep(name, entry));
    }

    pub fn remove(&mut self, name: &str) -> Option<ToolStateEntry> {
        Arc::make_mut(&mut self.tools).remove(name)
    }

    pub(crate) fn entries(&self) -> &BTreeMap<String, ToolStateEntry> {
        self.tools.as_ref()
    }
}

impl Serialize for ToolState {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        #[derive(Serialize)]
        struct ToolStateRef<'a> {
            generation: u64,
            tools: &'a BTreeMap<String, ToolStateEntry>,
        }

        ToolStateRef {
            generation: self.generation,
            tools: self.tools.as_ref(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ToolState {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct ToolStateOwned {
            generation: u64,
            tools: BTreeMap<String, ToolStateEntry>,
        }

        let owned = ToolStateOwned::deserialize(deserializer)?;
        Ok(Self {
            generation: owned.generation,
            tools: Arc::new(owned.tools),
        })
    }
}

#[async_trait::async_trait]
pub(crate) trait ToolSourceExecutor: Send + Sync + 'static {
    fn id(&self) -> &str;
    fn advertised_tools(&self) -> Vec<ToolManifest>;
    fn resolve_manifest(&self, name: &str) -> Option<ToolManifest> {
        self.advertised_tools()
            .into_iter()
            .find(|manifest| manifest.name == name)
    }
    fn resolve_contract(&self, name: &str) -> Option<Arc<ToolContract>>;
    async fn prepare_tool_call(
        &self,
        call: ToolPrepareCall<'_>,
    ) -> Result<PreparedToolCall, ToolResult> {
        Ok(PreparedToolCall::identity(call.pending))
    }
    async fn execute(
        &self,
        tool: &str,
        args: &serde_json::Value,
        context: &ToolContext<'_>,
        progress: Option<&ProgressSender>,
    ) -> ToolResult;
}

struct ToolProviderSource {
    id: String,
    provider: Arc<dyn ToolProvider>,
}

impl ToolProviderSource {
    fn new(id: impl Into<String>, provider: Arc<dyn ToolProvider>) -> Self {
        Self {
            id: id.into(),
            provider,
        }
    }
}

struct ToolProviderGroupSource {
    id: String,
    tools: RwLock<BTreeMap<String, (ToolManifest, usize)>>,
    providers: Vec<Arc<dyn ToolProvider>>,
}

impl ToolProviderGroupSource {
    fn new(id: impl Into<String>, providers: Vec<Arc<dyn ToolProvider>>) -> Self {
        let mut tools = BTreeMap::new();
        for (provider_idx, provider) in providers.iter().enumerate() {
            for manifest in provider.tool_manifests() {
                tools.insert(manifest.name.clone(), (manifest, provider_idx));
            }
        }
        Self {
            id: id.into(),
            tools: RwLock::new(tools),
            providers,
        }
    }

    fn read_advertised_tools(&self) -> Vec<ToolManifest> {
        let mut tools = BTreeMap::new();
        for (provider_idx, provider) in self.providers.iter().enumerate() {
            for manifest in provider.tool_manifests() {
                tools.insert(manifest.name.clone(), (manifest, provider_idx));
            }
        }
        let manifests = tools
            .values()
            .map(|(manifest, _)| manifest.clone())
            .collect::<Vec<_>>();
        *self
            .tools
            .write()
            .expect("tool provider group lock poisoned") = tools;
        manifests
    }

    fn provider_index_for(&self, name: &str) -> Option<usize> {
        self.resolve_manifest(name).and_then(|_| {
            self.tools
                .read()
                .expect("tool provider group lock poisoned")
                .get(name)
                .map(|(_, provider_idx)| *provider_idx)
        })
    }
}

#[async_trait::async_trait]
impl ToolSourceExecutor for ToolProviderGroupSource {
    fn id(&self) -> &str {
        &self.id
    }

    fn advertised_tools(&self) -> Vec<ToolManifest> {
        self.read_advertised_tools()
    }

    fn resolve_manifest(&self, name: &str) -> Option<ToolManifest> {
        if let Some((manifest, _)) = self
            .tools
            .read()
            .expect("tool provider group lock poisoned")
            .get(name)
        {
            return Some(manifest.clone());
        }
        for (provider_idx, provider) in self.providers.iter().enumerate() {
            if let Some(manifest) = provider.resolve_manifest(name) {
                self.tools
                    .write()
                    .expect("tool provider group lock poisoned")
                    .insert(name.to_string(), (manifest.clone(), provider_idx));
                return Some(manifest);
            }
        }
        None
    }

    fn resolve_contract(&self, name: &str) -> Option<Arc<ToolContract>> {
        let provider_idx = self.provider_index_for(name)?;
        self.providers[provider_idx].resolve_contract(name)
    }

    async fn prepare_tool_call(
        &self,
        call: ToolPrepareCall<'_>,
    ) -> Result<PreparedToolCall, ToolResult> {
        let name = call.pending.tool_name.clone();
        let Some(provider_idx) = self.provider_index_for(&name) else {
            return Err(ToolResult::err_fmt(format_args!("Unknown tool: {name}")));
        };
        self.providers[provider_idx].prepare_tool_call(call).await
    }

    async fn execute(
        &self,
        tool: &str,
        args: &serde_json::Value,
        context: &ToolContext<'_>,
        progress: Option<&ProgressSender>,
    ) -> ToolResult {
        let Some(provider_idx) = self.provider_index_for(tool) else {
            return ToolResult::err_fmt(format_args!("Unknown tool: {tool}"));
        };
        self.providers[provider_idx]
            .execute(ToolCall {
                name: tool,
                args,
                context,
                progress,
            })
            .await
    }
}

#[async_trait::async_trait]
impl ToolSourceExecutor for ToolProviderSource {
    fn id(&self) -> &str {
        &self.id
    }

    fn advertised_tools(&self) -> Vec<ToolManifest> {
        self.provider.tool_manifests()
    }

    fn resolve_manifest(&self, name: &str) -> Option<ToolManifest> {
        self.provider.resolve_manifest(name)
    }

    fn resolve_contract(&self, name: &str) -> Option<Arc<ToolContract>> {
        self.provider.resolve_contract(name)
    }

    async fn prepare_tool_call(
        &self,
        call: ToolPrepareCall<'_>,
    ) -> Result<PreparedToolCall, ToolResult> {
        self.provider.prepare_tool_call(call).await
    }

    async fn execute(
        &self,
        tool: &str,
        args: &serde_json::Value,
        context: &ToolContext<'_>,
        progress: Option<&ProgressSender>,
    ) -> ToolResult {
        self.provider
            .execute(ToolCall {
                name: tool,
                args,
                context,
                progress,
            })
            .await
    }
}

/// How a registry entry is connected to its tool source.
#[derive(Clone, Debug, PartialEq, Eq)]
enum ToolBinding {
    /// Resolvable through the registered source with this id.
    Bound(String),
    /// Persisted in a session snapshot but not resolvable from any currently
    /// registered source. Appears as `Off`; execution fails loudly; rebinds
    /// when a source re-advertises the same (name, id).
    Orphaned,
}

impl ToolBinding {
    fn source_id(&self) -> Option<&str> {
        match self {
            Self::Bound(id) => Some(id),
            Self::Orphaned => None,
        }
    }
}

#[derive(Clone)]
struct ToolRegistryEntry {
    manifest: ToolManifest,
    binding: ToolBinding,
}

impl ToolRegistryEntry {
    fn new(manifest: ToolManifest, source_id: impl Into<String>) -> Self {
        Self {
            manifest,
            binding: ToolBinding::Bound(source_id.into()),
        }
    }

    fn orphaned(manifest: ToolManifest) -> Self {
        Self {
            manifest,
            binding: ToolBinding::Orphaned,
        }
    }

    fn is_orphaned(&self) -> bool {
        self.binding == ToolBinding::Orphaned
    }

    /// The manifest as exposed to surfaces, catalogs, and availability checks.
    /// Orphaned entries are forced to `Off` in the view without mutating the
    /// stored manifest, so the persisted override survives a later rebind and
    /// export/restore round-trips stay byte-identical.
    fn view_manifest(&self) -> ToolManifest {
        let mut manifest = self.manifest.clone();
        if self.is_orphaned() {
            manifest.availability_override = Some(crate::ToolAvailability::Off);
        }
        manifest
    }

    fn export(&self) -> ToolStateEntry {
        ToolStateEntry {
            manifest: self.manifest.clone(),
            orphaned: self.is_orphaned(),
        }
    }
}

#[derive(Clone)]
struct ToolRegistryState {
    generation: u64,
    tools: BTreeMap<String, ToolRegistryEntry>,
    next_live_source_id: u64,
}

/// Outcome of [`ToolRegistry::restore_state`]: the adopted generation plus the
/// names of persisted tools that no registered source currently resolves.
/// Hosts should surface a non-empty `orphaned` list to the user — the session
/// opened, but those tools are `Off` until their source returns.
#[derive(Clone, Debug, Default)]
pub struct ToolRestoreReport {
    pub generation: u64,
    pub orphaned: Vec<String>,
}

#[derive(Debug, thiserror::Error)]
pub enum ReconfigureError {
    #[error("validation error: {0}")]
    Validation(String),
    #[error("unknown tool source: {0}")]
    UnknownSource(String),
    #[error("generation mismatch: expected {expected}, actual {actual}")]
    GenerationMismatch { expected: u64, actual: u64 },
}

#[derive(Clone)]
pub struct ToolRegistry {
    sources: Arc<RwLock<BTreeMap<String, Arc<dyn ToolSourceExecutor>>>>,
    state: Arc<RwLock<ToolRegistryState>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SourceReconcilePolicy {
    RejectExternalConflicts,
    OverlayReplacingConflicts,
}

impl ToolRegistry {
    pub fn from_tool_provider(provider: Arc<dyn ToolProvider>) -> Result<Self, ReconfigureError> {
        let registry = Self::empty();
        registry.upsert_source(Arc::new(ToolProviderSource::new(
            PLUGIN_SOURCE_ID,
            provider,
        )))?;
        Ok(registry)
    }

    pub(crate) fn from_tool_providers(
        providers: Vec<Arc<dyn ToolProvider>>,
    ) -> Result<Self, ReconfigureError> {
        let registry = Self::empty();
        registry.upsert_source(Arc::new(ToolProviderGroupSource::new(
            PLUGIN_SOURCE_ID,
            providers,
        )))?;
        Ok(registry)
    }

    pub(crate) fn empty() -> Self {
        Self {
            sources: Arc::new(RwLock::new(BTreeMap::new())),
            state: Arc::new(RwLock::new(ToolRegistryState {
                generation: 0,
                tools: BTreeMap::new(),
                next_live_source_id: 0,
            })),
        }
    }

    pub fn generation(&self) -> u64 {
        self.state
            .read()
            .expect("tool registry state lock poisoned")
            .generation
    }

    pub fn export_state(&self) -> ToolState {
        let state = self
            .state
            .read()
            .expect("tool registry state lock poisoned");
        ToolState::new(state.generation, export_tool_state_entries(&state.tools))
    }

    pub fn apply_state(&self, next: ToolState) -> Result<u64, ReconfigureError> {
        let current_generation = self.generation();
        if next.generation != current_generation {
            return Err(ReconfigureError::GenerationMismatch {
                expected: next.generation,
                actual: current_generation,
            });
        }

        validate_unique_manifest_entries(next.entries().values())?;
        let rebound = {
            let sources = self.sources.read().expect("tool source lock poisoned");
            rebind_tool_state_entries(next.entries(), &sources, RebindMode::RejectUnresolved)?
        };

        let mut state = self
            .state
            .write()
            .expect("tool registry state lock poisoned");
        if state.generation != next.generation {
            return Err(ReconfigureError::GenerationMismatch {
                expected: next.generation,
                actual: state.generation,
            });
        }
        state.tools = rebound.tools;
        state.generation += 1;
        Ok(state.generation)
    }

    /// Restore a persisted [`ToolState`] snapshot onto a freshly-built registry,
    /// adopting the snapshot's generation verbatim.
    ///
    /// Unlike [`apply_state`](Self::apply_state) — which applies an incremental
    /// *delta* expected at the current generation and bumps it by one — a
    /// restore reconstructs the exact persisted state regardless of the fresh
    /// registry's base generation, and does **not** bump. This is idempotent: a
    /// snapshot exported at generation `G` restores to generation `G`, so a
    /// re-export round-trips. Cold rebuilds (the durable process worker, session
    /// resume) restore a session whose tool catalog reached generation `G ≥ 2`
    /// onto a base registry at generation 1 — `apply_state` would reject that
    /// (`expected G, actual 1`); `restore_state` adopts `G`. Entries are still
    /// rebound to the live sources, so source identity is reconnected.
    ///
    /// Restore is tolerant of missing sources: a persisted tool that no current
    /// source resolves becomes an orphaned entry (kept with its last-known
    /// manifest, surfaced as `Off`, rebound when its source returns) instead of
    /// failing the whole restore. Real conflicts — the name resolving with a
    /// different id, or multiple sources claiming the same tool — still fail.
    pub fn restore_state(
        &self,
        snapshot: ToolState,
    ) -> Result<ToolRestoreReport, ReconfigureError> {
        validate_unique_manifest_entries(snapshot.entries().values())?;
        let rebound = {
            let sources = self.sources.read().expect("tool source lock poisoned");
            rebind_tool_state_entries(snapshot.entries(), &sources, RebindMode::OrphanUnresolved)?
        };

        let mut state = self
            .state
            .write()
            .expect("tool registry state lock poisoned");
        state.tools = rebound.tools;
        state.generation = snapshot.generation();
        Ok(ToolRestoreReport {
            generation: state.generation,
            orphaned: rebound.orphaned,
        })
    }

    pub fn add_tool_provider(
        &self,
        provider: Arc<dyn ToolProvider>,
    ) -> Result<ToolSourceHandle, ReconfigureError> {
        let source_id = {
            let mut state = self
                .state
                .write()
                .expect("tool registry state lock poisoned");
            state.next_live_source_id += 1;
            format!("live:{}", state.next_live_source_id)
        };
        self.upsert_source(Arc::new(ToolProviderSource::new(
            source_id.clone(),
            provider,
        )))?;
        Ok(ToolSourceHandle::new(source_id))
    }

    pub(crate) fn compose_session_catalog(
        &self,
        include_base_tools: bool,
        context_providers: Vec<Arc<dyn ToolProvider>>,
    ) -> Result<Self, ReconfigureError> {
        let registry = if include_base_tools {
            self.fork_with_state(self.export_state())?
        } else {
            Self::empty()
        };
        registry.upsert_overlay_source(Arc::new(ToolProviderGroupSource::new(
            "context",
            context_providers,
        )))?;
        Ok(registry)
    }

    pub(crate) fn upsert_source(
        &self,
        source: Arc<dyn ToolSourceExecutor>,
    ) -> Result<u64, ReconfigureError> {
        self.reconcile_source(source, SourceReconcilePolicy::RejectExternalConflicts)
    }

    fn upsert_overlay_source(
        &self,
        source: Arc<dyn ToolSourceExecutor>,
    ) -> Result<u64, ReconfigureError> {
        self.reconcile_source(source, SourceReconcilePolicy::OverlayReplacingConflicts)
    }

    fn reconcile_source(
        &self,
        source: Arc<dyn ToolSourceExecutor>,
        policy: SourceReconcilePolicy,
    ) -> Result<u64, ReconfigureError> {
        let source_id = source.id().to_string();
        let advertised_tools = source
            .advertised_tools()
            .into_iter()
            .map(|manifest| manifest_with_compact_contract(source.as_ref(), manifest))
            .collect::<Vec<_>>();
        validate_unique_manifests(&advertised_tools)?;

        let advertised_names = advertised_tools
            .iter()
            .map(|manifest| manifest.name.clone())
            .collect::<BTreeSet<_>>();
        let advertised_ids = advertised_tools
            .iter()
            .map(|manifest| manifest.id.clone())
            .collect::<BTreeSet<_>>();
        let mut state = self
            .state
            .write()
            .expect("tool registry state lock poisoned");
        let previous_overrides = state
            .tools
            .iter()
            .map(|(name, entry)| (name.clone(), entry.manifest.availability_override))
            .collect::<BTreeMap<_, _>>();
        match policy {
            SourceReconcilePolicy::RejectExternalConflicts => {
                // Orphans never conflict: a live advertisement supersedes a
                // tool that is currently unresolvable. Matching (name, id)
                // means the source came back — the entry rebinds below, with
                // its availability override preserved via `previous_overrides`.
                for manifest in &advertised_tools {
                    if let Some(existing) = state.tools.get(&manifest.name)
                        && let Some(existing_source) = existing.binding.source_id()
                        && existing_source != source_id
                    {
                        return Err(ReconfigureError::Validation(format!(
                            "duplicate tool name `{}` from source `{}` conflicts with source `{}`",
                            manifest.name, source_id, existing_source
                        )));
                    }
                    if let Some((existing_name, existing_source)) =
                        state.tools.iter().find_map(|(name, entry)| {
                            let existing_source = entry.binding.source_id()?;
                            (existing_source != source_id && entry.manifest.id == manifest.id)
                                .then(|| (name.clone(), existing_source.to_string()))
                        })
                    {
                        return Err(ReconfigureError::Validation(format!(
                            "duplicate tool id `{}` from source `{}` conflicts with tool `{}` from source `{}`",
                            manifest.id, source_id, existing_name, existing_source
                        )));
                    }
                }
                state.tools.retain(|name, entry| match &entry.binding {
                    // Drop this source's previous surface; it is re-inserted
                    // from the fresh advertisement below.
                    ToolBinding::Bound(bound) => bound != &source_id,
                    // Drop orphans the fresh advertisement supersedes.
                    ToolBinding::Orphaned => {
                        !advertised_names.contains(name)
                            && !advertised_ids.contains(&entry.manifest.id)
                    }
                });
            }
            SourceReconcilePolicy::OverlayReplacingConflicts => {
                state.tools.retain(|name, entry| {
                    entry.binding.source_id() != Some(source_id.as_str())
                        && !advertised_names.contains(name)
                        && !advertised_ids.contains(&entry.manifest.id)
                });
            }
        }
        for mut manifest in advertised_tools {
            let name = manifest.name.clone();
            manifest.availability_override = previous_overrides
                .get(&name)
                .copied()
                .flatten()
                .or(manifest.availability_override);
            state
                .tools
                .insert(name, ToolRegistryEntry::new(manifest, source_id.clone()));
        }
        self.sources
            .write()
            .expect("tool source lock poisoned")
            .insert(source_id, source);
        state.generation += 1;
        Ok(state.generation)
    }

    pub fn remove_source(&self, handle: &ToolSourceHandle) -> Result<u64, ReconfigureError> {
        self.remove_source_id(handle.as_str())
    }

    pub fn refresh_sources(&self) -> Result<u64, ReconfigureError> {
        let sources = self
            .sources
            .read()
            .expect("tool source lock poisoned")
            .values()
            .cloned()
            .collect::<Vec<_>>();
        let mut generation = self.generation();
        for source in sources {
            generation = self.upsert_source(source)?;
        }
        Ok(generation)
    }

    pub(crate) fn remove_source_id(&self, source_id: &str) -> Result<u64, ReconfigureError> {
        {
            let mut sources = self.sources.write().expect("tool source lock poisoned");
            if sources.remove(source_id).is_none() {
                return Err(ReconfigureError::UnknownSource(source_id.to_string()));
            }
        }
        let mut state = self
            .state
            .write()
            .expect("tool registry state lock poisoned");
        state
            .tools
            .retain(|_, entry| entry.binding.source_id() != Some(source_id));
        state.generation += 1;
        Ok(state.generation)
    }

    pub(crate) fn fork_with_state(&self, snapshot: ToolState) -> Result<Self, ReconfigureError> {
        let sources = self
            .sources
            .read()
            .expect("tool source lock poisoned")
            .iter()
            .map(|(k, v)| (k.clone(), Arc::clone(v)))
            .collect::<BTreeMap<_, _>>();
        validate_unique_manifest_entries(snapshot.entries().values())?;
        // Tolerant like `restore_state`: a fork mirrors whatever the parent
        // registry holds, including orphans whose sources are still absent.
        let rebound =
            rebind_tool_state_entries(snapshot.entries(), &sources, RebindMode::OrphanUnresolved)?;
        let generation = snapshot.generation.max(1);
        Ok(Self {
            sources: Arc::new(RwLock::new(sources)),
            state: Arc::new(RwLock::new(ToolRegistryState {
                generation,
                tools: rebound.tools,
                next_live_source_id: 0,
            })),
        })
    }
}

impl ToolRegistry {
    /// Try to rebind an orphaned entry to a source that now resolves the same
    /// (name, id). Returns the live manifest on success. A source resolving
    /// the name with a *different* id is a replaced implementation — the
    /// orphan keeps the name and stays unavailable rather than silently
    /// swapping semantics under the session.
    fn try_rebind_orphan(&self, name: &str, orphan_id: &crate::ToolId) -> Option<ToolManifest> {
        let sources = self
            .sources
            .read()
            .expect("tool source lock poisoned")
            .iter()
            .map(|(source_id, source)| (source_id.clone(), Arc::clone(source)))
            .collect::<Vec<_>>();
        for (source_id, source) in sources {
            let Some(manifest) = source.resolve_manifest(name) else {
                continue;
            };
            if manifest.id != *orphan_id {
                continue;
            }
            let mut manifest = manifest_with_compact_contract(source.as_ref(), manifest);
            let mut state = self
                .state
                .write()
                .expect("tool registry state lock poisoned");
            let existing = state.tools.get(name)?;
            if !existing.is_orphaned() {
                return Some(existing.view_manifest());
            }
            manifest.availability_override = existing
                .manifest
                .availability_override
                .or(manifest.availability_override);
            state.tools.insert(
                name.to_string(),
                ToolRegistryEntry::new(manifest.clone(), source_id),
            );
            state.generation += 1;
            return Some(manifest);
        }
        None
    }

    /// Resolve the source for a registry entry, distinguishing "unknown tool"
    /// from "known but orphaned" so callers can fail with a precise error.
    fn resolve_execution_source(
        &self,
        name: &str,
    ) -> Result<Arc<dyn ToolSourceExecutor>, ToolResult> {
        if self.resolve_manifest(name).is_none() {
            return Err(ToolResult::err_fmt(format_args!("Unknown tool: {name}")));
        }
        let binding = {
            let state = self
                .state
                .read()
                .expect("tool registry state lock poisoned");
            state.tools.get(name).map(|entry| entry.binding.clone())
        };
        let source_id = match binding {
            Some(ToolBinding::Bound(source_id)) => source_id,
            Some(ToolBinding::Orphaned) => {
                return Err(ToolResult::err_fmt(format_args!(
                    "Tool `{name}` is unavailable: it was restored from a persisted session \
                     but its source is not currently registered"
                )));
            }
            None => return Err(ToolResult::err_fmt(format_args!("Unknown tool: {name}"))),
        };
        let source = {
            self.sources
                .read()
                .expect("tool source lock poisoned")
                .get(&source_id)
                .cloned()
        };
        source.ok_or_else(|| {
            ToolResult::err_fmt(format_args!("Tool source missing for tool `{name}`"))
        })
    }
}

#[async_trait::async_trait]
impl ToolProvider for ToolRegistry {
    fn tool_manifests(&self) -> Vec<ToolManifest> {
        let state = self
            .state
            .read()
            .expect("tool registry state lock poisoned");
        state
            .tools
            .values()
            .map(ToolRegistryEntry::view_manifest)
            .collect()
    }

    fn resolve_manifest(&self, name: &str) -> Option<ToolManifest> {
        enum Known {
            Bound(ToolManifest),
            Orphaned(ToolManifest, crate::ToolId),
        }
        let known = {
            let state = self
                .state
                .read()
                .expect("tool registry state lock poisoned");
            state.tools.get(name).map(|entry| {
                if entry.is_orphaned() {
                    Known::Orphaned(entry.view_manifest(), entry.manifest.id.clone())
                } else {
                    Known::Bound(entry.manifest.clone())
                }
            })
        };
        match known {
            Some(Known::Bound(manifest)) => return Some(manifest),
            Some(Known::Orphaned(off_manifest, orphan_id)) => {
                // A source may have come back since the restore: rebind on
                // (name, id) match, otherwise keep serving the Off view.
                return Some(
                    self.try_rebind_orphan(name, &orphan_id)
                        .unwrap_or(off_manifest),
                );
            }
            None => {}
        }

        let sources = self
            .sources
            .read()
            .expect("tool source lock poisoned")
            .iter()
            .map(|(source_id, source)| (source_id.clone(), Arc::clone(source)))
            .collect::<Vec<_>>();
        for (source_id, source) in sources {
            let Some(manifest) = source.resolve_manifest(name) else {
                continue;
            };
            let mut manifest = manifest_with_compact_contract(source.as_ref(), manifest);
            let previous_override = {
                let state = self
                    .state
                    .read()
                    .expect("tool registry state lock poisoned");
                state
                    .tools
                    .get(&manifest.name)
                    .and_then(|entry| entry.manifest.availability_override)
            };
            manifest.availability_override = previous_override.or(manifest.availability_override);
            let mut state = self
                .state
                .write()
                .expect("tool registry state lock poisoned");
            if let Some(existing) = state.tools.get(&manifest.name) {
                return (existing.binding.source_id() == Some(source_id.as_str()))
                    .then(|| existing.view_manifest());
            }
            if let Some((_, existing)) = state
                .tools
                .iter()
                .find(|(_, entry)| entry.manifest.id == manifest.id)
            {
                return (existing.binding.source_id() == Some(source_id.as_str()))
                    .then(|| existing.view_manifest());
            }
            state.tools.insert(
                manifest.name.clone(),
                ToolRegistryEntry::new(manifest.clone(), source_id),
            );
            state.generation += 1;
            return Some(manifest);
        }
        None
    }

    fn resolve_contract(&self, name: &str) -> Option<Arc<ToolContract>> {
        let source_id = self.resolve_manifest(name).and_then(|_| {
            let state = self
                .state
                .read()
                .expect("tool registry state lock poisoned");
            state
                .tools
                .get(name)
                .and_then(|entry| entry.binding.source_id().map(str::to_string))
        })?;
        self.sources
            .read()
            .expect("tool source lock poisoned")
            .get(&source_id)?
            .resolve_contract(name)
    }

    async fn prepare_tool_call(
        &self,
        call: ToolPrepareCall<'_>,
    ) -> Result<PreparedToolCall, ToolResult> {
        let name = call.pending.tool_name.clone();
        let source = self.resolve_execution_source(&name)?;
        source.prepare_tool_call(call).await
    }

    async fn execute(&self, call: ToolCall<'_>) -> ToolResult {
        let name = call.name;
        let source = match self.resolve_execution_source(name) {
            Ok(source) => source,
            Err(result) => return result,
        };
        source
            .execute(name, call.args, call.context, call.progress)
            .await
    }
}

fn validate_unique_manifests(manifests: &[ToolManifest]) -> Result<(), ReconfigureError> {
    let mut names = BTreeSet::new();
    let mut ids = BTreeSet::new();
    for manifest in manifests {
        if manifest.id.as_str().trim().is_empty() {
            return Err(ReconfigureError::Validation(
                "tool id cannot be empty".to_string(),
            ));
        }
        if !ids.insert(manifest.id.clone()) {
            return Err(ReconfigureError::Validation(format!(
                "duplicate tool id `{}` in source",
                manifest.id
            )));
        }
        if manifest.name.trim().is_empty() {
            return Err(ReconfigureError::Validation(
                "tool name cannot be empty".to_string(),
            ));
        }
        if !names.insert(manifest.name.clone()) {
            return Err(ReconfigureError::Validation(format!(
                "duplicate tool name `{}` in source",
                manifest.name
            )));
        }
    }
    Ok(())
}

fn manifest_with_compact_contract(
    source: &dyn ToolSourceExecutor,
    mut manifest: ToolManifest,
) -> ToolManifest {
    if manifest.compact_contract.is_none()
        && let Some(contract) = source.resolve_contract(&manifest.name)
    {
        manifest.compact_contract = Some(contract.compact_contract(&manifest));
    }
    manifest
}

fn export_tool_state_entries(
    entries: &BTreeMap<String, ToolRegistryEntry>,
) -> BTreeMap<String, ToolStateEntry> {
    entries
        .iter()
        .map(|(name, entry)| (name.clone(), entry.export()))
        .collect()
}

/// How [`rebind_tool_state_entries`] treats a persisted tool that no
/// registered source resolves by name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RebindMode {
    /// Restore/fork: keep the entry as an orphan instead of failing. Sessions
    /// must stay openable when a dynamic source (e.g. an MCP server) is down.
    OrphanUnresolved,
    /// Explicit `apply_state`: fail on unresolved tools, except entries the
    /// snapshot already marks as orphaned — otherwise a host could never
    /// round-trip `export_state` → edit → `apply_state` while an orphan exists.
    RejectUnresolved,
}

struct ReboundTools {
    tools: BTreeMap<String, ToolRegistryEntry>,
    orphaned: Vec<String>,
}

fn rebind_tool_state_entries(
    entries: &BTreeMap<String, ToolStateEntry>,
    sources: &BTreeMap<String, Arc<dyn ToolSourceExecutor>>,
    mode: RebindMode,
) -> Result<ReboundTools, ReconfigureError> {
    let mut rebound = BTreeMap::new();
    let mut orphaned = Vec::new();
    for (name, entry) in entries {
        if name != &entry.manifest.name {
            return Err(ReconfigureError::Validation(format!(
                "tool state key `{}` does not match manifest name `{}`",
                name, entry.manifest.name
            )));
        }

        let mut name_matches = Vec::new();
        for (source_id, source) in sources {
            let Some(manifest) = source.resolve_manifest(name) else {
                continue;
            };
            name_matches.push((
                source_id.clone(),
                manifest_with_compact_contract(source.as_ref(), manifest),
            ));
        }

        if name_matches.is_empty() {
            if mode == RebindMode::RejectUnresolved && !entry.orphaned {
                return Err(ReconfigureError::Validation(format!(
                    "no registered tool source resolves tool `{name}`"
                )));
            }
            orphaned.push(name.clone());
            rebound.insert(
                name.clone(),
                ToolRegistryEntry::orphaned(entry.manifest.clone()),
            );
            continue;
        }

        let matching_id = name_matches
            .iter()
            .filter(|(_, manifest)| manifest.id == entry.manifest.id)
            .collect::<Vec<_>>();

        if matching_id.len() == 1 {
            let source_id = matching_id[0].0.clone();
            rebound.insert(
                name.clone(),
                ToolRegistryEntry::new(entry.manifest.clone(), source_id),
            );
        } else if matching_id.is_empty() {
            let resolved_ids = name_matches
                .iter()
                .map(|(_, manifest)| manifest.id.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            return Err(ReconfigureError::Validation(format!(
                "tool `{name}` resolved with id(s) `{resolved_ids}`, expected `{}`",
                entry.manifest.id
            )));
        } else {
            return Err(ReconfigureError::Validation(format!(
                "tool `{name}` with id `{}` is resolved by multiple registered sources",
                entry.manifest.id
            )));
        }
    }
    Ok(ReboundTools {
        tools: rebound,
        orphaned,
    })
}

fn validate_unique_manifest_entries<'a>(
    entries: impl IntoIterator<Item = &'a ToolStateEntry>,
) -> Result<(), ReconfigureError> {
    let manifests = entries
        .into_iter()
        .map(|entry| entry.stored_manifest().clone())
        .collect::<Vec<_>>();
    validate_unique_manifests(&manifests)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ToolDefinition;
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct MockTool;
    struct MixedEnabledTool;
    struct ExternalMockSource;
    struct ExactResolvingSource {
        manifest_resolutions: Arc<AtomicUsize>,
        contract_resolutions: Arc<AtomicUsize>,
        executions: Arc<AtomicUsize>,
    }
    struct NamedExactSource {
        id: &'static str,
    }
    struct DynamicToolProvider {
        names: Arc<std::sync::Mutex<Vec<String>>>,
    }

    fn test_tool(
        name: &str,
        description: &str,
        availability: crate::ToolAvailabilityConfig,
    ) -> ToolDefinition {
        ToolDefinition::raw_with_id(
            format!("tool:{name}"),
            name,
            description,
            ToolDefinition::default_input_schema(),
            json!({ "type": "string" }),
        )
        .with_availability(availability)
    }

    fn manifests(definitions: Vec<ToolDefinition>) -> Vec<ToolManifest> {
        definitions
            .into_iter()
            .map(|tool| tool.manifest())
            .collect()
    }

    fn contract_from(definitions: Vec<ToolDefinition>, name: &str) -> Option<Arc<ToolContract>> {
        definitions
            .into_iter()
            .find(|tool| tool.name() == name)
            .map(|tool| Arc::new(tool.contract()))
    }

    fn dynamic_definition(name: &str) -> ToolDefinition {
        test_tool(name, "dynamic", crate::ToolAvailabilityConfig::callable())
    }

    fn test_tool_context() -> crate::ToolContext<'static> {
        crate::ToolContext::builder(
            "registry-test".to_string(),
            Arc::new(crate::testing::MockSessionManager::default()),
            Arc::new(crate::testing::MockSessionManager::default()),
            Arc::new(crate::testing::MockSessionManager::default()),
            Arc::new(crate::UnavailableProcessService),
            Arc::new(crate::DefaultProcessCancelAbility),
            crate::runtime::RuntimeEffectControllerHandle::shared(Arc::new(
                crate::InlineRuntimeEffectController,
            )),
            Arc::new(crate::InMemoryAttachmentStore::new()),
            crate::DirectCompletionClient::unavailable(
                "direct completions are unavailable in this test context",
            ),
        )
        .build()
    }

    #[async_trait::async_trait]
    impl ToolProvider for MockTool {
        fn tool_manifests(&self) -> Vec<ToolManifest> {
            manifests(vec![test_tool(
                "mock_tool",
                "mock",
                crate::ToolAvailabilityConfig::callable(),
            )])
        }

        fn resolve_contract(&self, name: &str) -> Option<Arc<ToolContract>> {
            contract_from(
                vec![test_tool(
                    "mock_tool",
                    "mock",
                    crate::ToolAvailabilityConfig::callable(),
                )],
                name,
            )
        }

        async fn execute(&self, _call: ToolCall<'_>) -> ToolResult {
            ToolResult::ok(serde_json::json!("ok"))
        }
    }

    #[async_trait::async_trait]
    impl ToolProvider for MixedEnabledTool {
        fn tool_manifests(&self) -> Vec<ToolManifest> {
            manifests(vec![
                test_tool(
                    "enabled_tool",
                    "enabled",
                    crate::ToolAvailabilityConfig::callable(),
                ),
                test_tool(
                    "disabled_tool",
                    "disabled",
                    crate::ToolAvailabilityConfig::off(),
                ),
            ])
        }

        fn resolve_contract(&self, name: &str) -> Option<Arc<ToolContract>> {
            contract_from(
                vec![
                    test_tool(
                        "enabled_tool",
                        "enabled",
                        crate::ToolAvailabilityConfig::callable(),
                    ),
                    test_tool(
                        "disabled_tool",
                        "disabled",
                        crate::ToolAvailabilityConfig::off(),
                    ),
                ],
                name,
            )
        }

        async fn execute(&self, _call: ToolCall<'_>) -> ToolResult {
            ToolResult::ok(serde_json::json!("ok"))
        }
    }

    #[async_trait::async_trait]
    impl ToolSourceExecutor for ExternalMockSource {
        fn id(&self) -> &str {
            "external"
        }

        fn advertised_tools(&self) -> Vec<ToolManifest> {
            manifests(vec![ToolDefinition::raw_with_id(
                "tool:mcp__demo__search",
                "mcp__demo__search",
                "search",
                json!({
                    "type": "object",
                    "properties": {
                        "query": { "type": "string" }
                    },
                    "required": ["query"],
                    "additionalProperties": false
                }),
                json!({ "type": "object", "additionalProperties": true }),
            )])
        }

        fn resolve_contract(&self, name: &str) -> Option<Arc<ToolContract>> {
            contract_from(
                vec![ToolDefinition::raw_with_id(
                    "tool:mcp__demo__search",
                    "mcp__demo__search",
                    "search",
                    json!({
                        "type": "object",
                        "properties": {
                            "query": { "type": "string" }
                        },
                        "required": ["query"],
                        "additionalProperties": false
                    }),
                    json!({ "type": "object", "additionalProperties": true }),
                )],
                name,
            )
        }

        async fn execute(
            &self,
            tool: &str,
            args: &serde_json::Value,
            _context: &ToolContext<'_>,
            _progress: Option<&ProgressSender>,
        ) -> ToolResult {
            ToolResult::ok(json!({
                "tool": tool,
                "args": args
            }))
        }
    }

    #[async_trait::async_trait]
    impl ToolSourceExecutor for ExactResolvingSource {
        fn id(&self) -> &str {
            "exact"
        }

        fn advertised_tools(&self) -> Vec<ToolManifest> {
            Vec::new()
        }

        fn resolve_manifest(&self, name: &str) -> Option<ToolManifest> {
            self.manifest_resolutions.fetch_add(1, Ordering::SeqCst);
            (name == "host_only").then(|| {
                test_tool(
                    "host_only",
                    "host-only",
                    crate::ToolAvailabilityConfig::callable(),
                )
                .manifest()
            })
        }

        fn resolve_contract(&self, name: &str) -> Option<Arc<ToolContract>> {
            self.contract_resolutions.fetch_add(1, Ordering::SeqCst);
            contract_from(
                vec![test_tool(
                    "host_only",
                    "host-only",
                    crate::ToolAvailabilityConfig::callable(),
                )],
                name,
            )
        }

        async fn execute(
            &self,
            tool: &str,
            _args: &serde_json::Value,
            _context: &ToolContext<'_>,
            _progress: Option<&ProgressSender>,
        ) -> ToolResult {
            self.executions.fetch_add(1, Ordering::SeqCst);
            ToolResult::ok(json!(tool))
        }
    }

    #[async_trait::async_trait]
    impl ToolSourceExecutor for NamedExactSource {
        fn id(&self) -> &str {
            self.id
        }

        fn advertised_tools(&self) -> Vec<ToolManifest> {
            Vec::new()
        }

        fn resolve_manifest(&self, name: &str) -> Option<ToolManifest> {
            (name == "host_only").then(|| {
                test_tool(
                    "host_only",
                    "host-only",
                    crate::ToolAvailabilityConfig::callable(),
                )
                .manifest()
            })
        }

        fn resolve_contract(&self, _name: &str) -> Option<Arc<ToolContract>> {
            None
        }

        async fn execute(
            &self,
            tool: &str,
            _args: &serde_json::Value,
            _context: &ToolContext<'_>,
            _progress: Option<&ProgressSender>,
        ) -> ToolResult {
            ToolResult::ok(json!(tool))
        }
    }

    #[async_trait::async_trait]
    impl ToolProvider for DynamicToolProvider {
        fn tool_manifests(&self) -> Vec<ToolManifest> {
            self.names
                .lock()
                .expect("dynamic tool names lock")
                .iter()
                .map(|name| dynamic_definition(name).manifest())
                .collect()
        }

        fn resolve_contract(&self, name: &str) -> Option<Arc<ToolContract>> {
            self.names
                .lock()
                .expect("dynamic tool names lock")
                .iter()
                .any(|tool_name| tool_name == name)
                .then(|| Arc::new(dynamic_definition(name).contract()))
        }

        async fn execute(&self, call: ToolCall<'_>) -> ToolResult {
            ToolResult::ok(json!(call.name))
        }
    }

    #[test]
    fn registry_preserves_initial_availability_state() {
        let registry =
            ToolRegistry::from_tool_provider(Arc::new(MixedEnabledTool)).expect("registry");
        let snapshot = registry.export_state();
        assert_eq!(
            snapshot
                .get("enabled_tool")
                .unwrap()
                .manifest()
                .effective_availability(),
            crate::ToolAvailability::Callable
        );
        assert_eq!(
            snapshot
                .get("disabled_tool")
                .unwrap()
                .manifest()
                .effective_availability(),
            crate::ToolAvailability::Off
        );
    }

    #[test]
    fn exported_tool_state_is_source_free() {
        let registry = ToolRegistry::from_tool_provider(Arc::new(MockTool)).expect("registry");
        registry
            .add_tool_provider(Arc::new(MixedEnabledTool))
            .expect("live provider registered");

        let value = serde_json::to_value(registry.export_state()).expect("serialized tool state");
        let serialized = value.to_string();

        assert!(!serialized.contains("source_id"));
        assert!(!serialized.contains(PLUGIN_SOURCE_ID));
        assert!(!serialized.contains("live:"));
    }

    #[test]
    fn apply_state_rebinds_source_free_snapshot_to_current_sources() {
        let source_registry =
            ToolRegistry::from_tool_provider(Arc::new(MixedEnabledTool)).expect("source registry");
        let snapshot = source_registry.export_state();

        let target_registry =
            ToolRegistry::from_tool_provider(Arc::new(MixedEnabledTool)).expect("target registry");
        let next_generation = target_registry
            .apply_state(snapshot.with_generation(target_registry.generation()))
            .expect("state rebound");

        assert_eq!(next_generation, target_registry.generation());
        assert!(target_registry.resolve_contract("enabled_tool").is_some());
    }

    #[test]
    fn apply_state_rejects_tools_not_advertised_by_source() {
        let registry = ToolRegistry::from_tool_provider(Arc::new(MockTool)).expect("registry");
        let snapshot = registry.export_state();
        let generation = snapshot.generation();
        let mut tools = snapshot.entries().clone();
        tools.insert(
            "missing".to_string(),
            ToolStateEntry::new(
                test_tool(
                    "missing",
                    "missing",
                    crate::ToolAvailabilityConfig::callable(),
                )
                .manifest(),
            ),
        );
        let snapshot = ToolState::new(generation, tools);
        assert!(matches!(
            registry.apply_state(snapshot),
            Err(ReconfigureError::Validation(_))
        ));
    }

    #[test]
    fn apply_state_rejects_snapshot_when_provider_is_absent() {
        let source_registry =
            ToolRegistry::from_tool_provider(Arc::new(MockTool)).expect("source registry");
        source_registry
            .upsert_source(Arc::new(ExternalMockSource))
            .expect("source registered");
        let snapshot = source_registry.export_state();

        let target_registry =
            ToolRegistry::from_tool_provider(Arc::new(MockTool)).expect("target registry");
        let err = target_registry
            .apply_state(snapshot.with_generation(target_registry.generation()))
            .expect_err("missing provider should fail");

        assert!(matches!(err, ReconfigureError::Validation(_)));
    }

    #[test]
    fn apply_state_rejects_ambiguous_current_source_binding() {
        let registry = ToolRegistry::empty();
        registry
            .upsert_source(Arc::new(NamedExactSource { id: "exact-a" }))
            .expect("source a registered");
        registry
            .upsert_source(Arc::new(NamedExactSource { id: "exact-b" }))
            .expect("source b registered");

        let mut tools = BTreeMap::new();
        tools.insert(
            "host_only".to_string(),
            ToolStateEntry::new(
                test_tool(
                    "host_only",
                    "host-only",
                    crate::ToolAvailabilityConfig::callable(),
                )
                .manifest(),
            ),
        );

        let err = registry
            .apply_state(ToolState::new(registry.generation(), tools))
            .expect_err("ambiguous source binding should fail");

        assert!(matches!(err, ReconfigureError::Validation(_)));
    }

    #[test]
    fn advertised_manifest_resolves_without_exact_host_lookup() {
        let manifest_resolutions = Arc::new(AtomicUsize::new(0));
        let registry = ToolRegistry::from_tool_provider(Arc::new(MockTool)).expect("registry");
        registry
            .upsert_source(Arc::new(ExactResolvingSource {
                manifest_resolutions: Arc::clone(&manifest_resolutions),
                contract_resolutions: Arc::new(AtomicUsize::new(0)),
                executions: Arc::new(AtomicUsize::new(0)),
            }))
            .expect("source registered");

        assert_eq!(
            registry
                .resolve_manifest("mock_tool")
                .map(|manifest| manifest.name),
            Some("mock_tool".to_string())
        );
        assert_eq!(manifest_resolutions.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn refresh_sources_re_reads_group_provider_manifests() {
        let names = Arc::new(std::sync::Mutex::new(vec!["dynamic_one".to_string()]));
        let provider: Arc<dyn ToolProvider> = Arc::new(DynamicToolProvider {
            names: Arc::clone(&names),
        });
        let registry = ToolRegistry::from_tool_providers(vec![provider]).expect("registry");

        let tool_names = || {
            registry
                .tool_manifests()
                .into_iter()
                .map(|manifest| manifest.name)
                .collect::<BTreeSet<_>>()
        };

        assert!(tool_names().contains("dynamic_one"));
        assert!(!tool_names().contains("dynamic_two"));

        names
            .lock()
            .expect("dynamic tool names lock")
            .push("dynamic_two".to_string());
        registry.refresh_sources().expect("refresh sources");
        let refreshed = tool_names();
        assert!(refreshed.contains("dynamic_one"));
        assert!(refreshed.contains("dynamic_two"));

        names
            .lock()
            .expect("dynamic tool names lock")
            .retain(|name| name != "dynamic_one");
        registry.refresh_sources().expect("refresh sources");
        let refreshed = tool_names();
        assert!(!refreshed.contains("dynamic_one"));
        assert!(refreshed.contains("dynamic_two"));
    }

    #[tokio::test]
    async fn unknown_manifest_exact_resolves_and_routes_to_owner() {
        let manifest_resolutions = Arc::new(AtomicUsize::new(0));
        let contract_resolutions = Arc::new(AtomicUsize::new(0));
        let executions = Arc::new(AtomicUsize::new(0));
        let registry = ToolRegistry::from_tool_provider(Arc::new(MockTool)).expect("registry");
        registry
            .upsert_source(Arc::new(ExactResolvingSource {
                manifest_resolutions: Arc::clone(&manifest_resolutions),
                contract_resolutions: Arc::clone(&contract_resolutions),
                executions: Arc::clone(&executions),
            }))
            .expect("source registered");

        assert_eq!(
            registry
                .resolve_manifest("host_only")
                .map(|manifest| manifest.name),
            Some("host_only".to_string())
        );
        assert_eq!(manifest_resolutions.load(Ordering::SeqCst), 1);

        let contract = registry.resolve_contract("host_only");
        assert!(contract.is_some());
        assert_eq!(manifest_resolutions.load(Ordering::SeqCst), 1);
        assert_eq!(contract_resolutions.load(Ordering::SeqCst), 1);

        let context = test_tool_context();
        let args = json!({});
        let result = registry
            .execute(crate::ToolCall {
                name: "host_only",
                args: &args,
                context: &context,
                progress: None,
            })
            .await;
        assert!(result.is_success());
        assert_eq!(result.value_for_projection(), json!("host_only"));
        assert_eq!(executions.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn unknown_manifest_without_host_resolver_is_unavailable() {
        let registry = ToolRegistry::from_tool_provider(Arc::new(MockTool)).expect("registry");

        assert!(registry.resolve_manifest("missing").is_none());
        assert!(registry.resolve_contract("missing").is_none());
    }

    #[tokio::test]
    async fn upsert_source_registers_and_executes_external_tools() {
        let registry = ToolRegistry::from_tool_provider(Arc::new(MockTool)).expect("registry");
        registry
            .upsert_source(Arc::new(ExternalMockSource))
            .expect("source registered");

        let defs = registry.tool_manifests();
        assert!(defs.iter().any(|def| def.name == "mcp__demo__search"));

        let context = test_tool_context();
        let args = json!({ "query": "hello" });
        let result = registry
            .execute(crate::ToolCall {
                name: "mcp__demo__search",
                args: &args,
                context: &context,
                progress: None,
            })
            .await;
        assert!(result.is_success());
        assert_eq!(
            result.value_for_projection()["tool"],
            json!("mcp__demo__search")
        );
        assert_eq!(
            result.value_for_projection()["args"]["query"],
            json!("hello")
        );
    }

    #[test]
    fn upsert_source_preserves_availability_override_on_refresh() {
        let registry = ToolRegistry::from_tool_provider(Arc::new(MockTool)).expect("registry");
        registry
            .upsert_source(Arc::new(ExternalMockSource))
            .expect("source registered");
        let mut snapshot = registry.export_state();
        snapshot
            .set_availability("mcp__demo__search", Some(crate::ToolAvailability::Off))
            .unwrap();
        registry.apply_state(snapshot).unwrap();
        registry
            .upsert_source(Arc::new(ExternalMockSource))
            .expect("source refreshed");
        let snapshot = registry.export_state();
        assert_eq!(
            snapshot
                .get("mcp__demo__search")
                .unwrap()
                .manifest()
                .effective_availability(),
            crate::ToolAvailability::Off
        );
    }

    #[test]
    fn restore_state_adopts_generation_at_or_above_three() {
        // Cold rebuild ratchet: a session whose tool catalog advanced to
        // generation >= 3 restores onto a fresh base-1 registry. `restore_state`
        // adopts the snapshot's generation verbatim; `apply_state` (a gen-matched
        // delta) rejects it. This is the exact divergence the durable worker /
        // session resume rebuild relies on `restore_state` to absorb.
        let source = ToolRegistry::from_tool_provider(Arc::new(MockTool)).expect("source registry");
        let snapshot = source.export_state().with_generation(3);

        let target = ToolRegistry::from_tool_provider(Arc::new(MockTool)).expect("target registry");
        assert_eq!(
            target.generation(),
            1,
            "a fresh registry starts at generation 1"
        );
        let restored = target
            .restore_state(snapshot.clone())
            .expect("restore adopts the snapshot generation");
        assert_eq!(
            restored.generation, 3,
            "restore returns the adopted generation"
        );
        assert!(
            restored.orphaned.is_empty(),
            "all tools resolve, so nothing orphans"
        );
        assert_eq!(
            target.generation(),
            3,
            "restore adopts gen 3 onto a base-1 registry without bumping"
        );
        // A re-export round-trips at the same generation (idempotent).
        assert_eq!(target.export_state().generation(), 3);

        // apply_state on the same high-generation snapshot is rejected — proving
        // the rebuild would have failed without restore_state.
        let fresh = ToolRegistry::from_tool_provider(Arc::new(MockTool)).expect("fresh registry");
        assert!(
            matches!(
                fresh.apply_state(snapshot),
                Err(ReconfigureError::GenerationMismatch {
                    expected: 3,
                    actual: 1
                })
            ),
            "apply_state must reject a gen-3 snapshot on a base-1 registry"
        );
    }

    /// Build a snapshot whose `mcp__demo__search` entry only resolves while
    /// `ExternalMockSource` is registered — restoring it elsewhere orphans it.
    fn snapshot_with_external_tool() -> ToolState {
        let source = ToolRegistry::from_tool_provider(Arc::new(MockTool)).expect("source registry");
        source
            .upsert_source(Arc::new(ExternalMockSource))
            .expect("source registered");
        source.export_state()
    }

    #[tokio::test]
    async fn restore_orphans_unresolved_tools_instead_of_failing() {
        let mut snapshot = snapshot_with_external_tool();
        snapshot
            .set_availability(
                "mcp__demo__search",
                Some(crate::ToolAvailability::Showcased),
            )
            .expect("override set");

        let target = ToolRegistry::from_tool_provider(Arc::new(MockTool)).expect("target");
        let report = target
            .restore_state(snapshot)
            .expect("restore tolerates the missing source");
        assert_eq!(report.orphaned, vec!["mcp__demo__search".to_string()]);

        // The orphan surfaces as Off without mutating the stored manifest.
        let view = target
            .tool_manifests()
            .into_iter()
            .find(|manifest| manifest.name == "mcp__demo__search")
            .expect("orphan stays in the surface listing");
        assert_eq!(
            view.effective_availability(),
            crate::ToolAvailability::Off,
            "orphans are forced Off in the view"
        );
        let exported = target.export_state();
        let exported_view = exported
            .tool_manifests()
            .into_iter()
            .find(|manifest| manifest.name == "mcp__demo__search")
            .expect("orphan is visible in exported tool state");
        assert_eq!(
            exported_view.effective_availability(),
            crate::ToolAvailability::Off,
            "exported ToolState exposes the same forced-Off orphan view"
        );
        let entry = exported.get("mcp__demo__search").expect("orphan exported");
        assert!(entry.is_orphaned());
        assert_eq!(
            entry.manifest().effective_availability(),
            crate::ToolAvailability::Off,
            "entry manifest is also the public forced-Off view"
        );
        assert_eq!(
            entry.stored_manifest().availability_override,
            Some(crate::ToolAvailability::Showcased),
            "the persisted override survives orphaning"
        );

        // Execution fails loudly with a precise error.
        let context = test_tool_context();
        let args = json!({ "query": "hello" });
        let result = target
            .execute(crate::ToolCall {
                name: "mcp__demo__search",
                args: &args,
                context: &context,
                progress: None,
            })
            .await;
        assert!(!result.is_success());
        assert!(
            format!("{result:?}").contains("unavailable"),
            "orphan execution error names the condition: {result:?}"
        );

        // Bound tools are unaffected.
        assert!(target.resolve_contract("mock_tool").is_some());
    }

    #[tokio::test]
    async fn orphan_rebinds_when_source_is_upserted_again() {
        let mut snapshot = snapshot_with_external_tool();
        snapshot
            .set_availability(
                "mcp__demo__search",
                Some(crate::ToolAvailability::Showcased),
            )
            .expect("override set");
        let target = ToolRegistry::from_tool_provider(Arc::new(MockTool)).expect("target");
        target.restore_state(snapshot).expect("restore");
        let orphaned_generation = target.generation();

        target
            .upsert_source(Arc::new(ExternalMockSource))
            .expect("the returning source must not conflict with its own orphan");
        assert!(
            target.generation() > orphaned_generation,
            "rebinding bumps the generation"
        );

        let exported = target.export_state();
        let entry = exported.get("mcp__demo__search").expect("entry kept");
        assert!(
            !entry.is_orphaned(),
            "the orphan rebound to the live source"
        );
        assert_eq!(
            entry.manifest().availability_override,
            Some(crate::ToolAvailability::Showcased),
            "rebinding preserves the persisted override"
        );

        let context = test_tool_context();
        let args = json!({ "query": "hello" });
        let result = target
            .execute(crate::ToolCall {
                name: "mcp__demo__search",
                args: &args,
                context: &context,
                progress: None,
            })
            .await;
        assert!(result.is_success(), "rebound tool executes: {result:?}");
    }

    #[tokio::test]
    async fn orphan_rebinds_lazily_via_resolve_manifest() {
        // `NamedExactSource` advertises nothing, so reconcile-on-upsert cannot
        // rebind; only the lazy `resolve_manifest` path can.
        let source_registry = ToolRegistry::empty();
        source_registry
            .upsert_source(Arc::new(NamedExactSource { id: "exact-a" }))
            .expect("source registered");
        assert!(source_registry.resolve_manifest("host_only").is_some());
        let snapshot = source_registry.export_state();

        let target = ToolRegistry::from_tool_provider(Arc::new(MockTool)).expect("target");
        let report = target.restore_state(snapshot).expect("restore");
        assert_eq!(report.orphaned, vec!["host_only".to_string()]);

        target
            .upsert_source(Arc::new(NamedExactSource { id: "exact-a" }))
            .expect("source returns");
        let manifest = target
            .resolve_manifest("host_only")
            .expect("resolves after the source returned");
        assert_eq!(
            manifest.effective_availability(),
            crate::ToolAvailability::Callable,
            "lazy rebind drops the forced-Off orphan view"
        );
        assert!(
            !target
                .export_state()
                .get("host_only")
                .expect("entry kept")
                .is_orphaned()
        );
    }

    #[test]
    fn restore_still_fails_when_name_resolves_with_different_id() {
        struct ReplacedSearchTool;
        #[async_trait::async_trait]
        impl ToolProvider for ReplacedSearchTool {
            fn tool_manifests(&self) -> Vec<ToolManifest> {
                manifests(vec![ToolDefinition::raw_with_id(
                    "tool:replaced",
                    "mcp__demo__search",
                    "a different implementation under the same name",
                    ToolDefinition::default_input_schema(),
                    json!({}),
                )])
            }
            fn resolve_contract(&self, _name: &str) -> Option<Arc<ToolContract>> {
                None
            }
            async fn execute(&self, _call: ToolCall<'_>) -> ToolResult {
                ToolResult::ok(json!("ok"))
            }
        }

        let snapshot = snapshot_with_external_tool();
        let target = ToolRegistry::from_tool_provider(Arc::new(MockTool)).expect("target");
        target
            .add_tool_provider(Arc::new(ReplacedSearchTool))
            .expect("replacement registered");
        let err = target
            .restore_state(snapshot)
            .expect_err("same name with a different id is a real conflict");
        assert!(matches!(err, ReconfigureError::Validation(_)));
    }

    #[test]
    fn apply_state_round_trips_while_orphans_exist() {
        // `export_state` → edit → `apply_state` must work with an orphan in
        // the snapshot: the exported orphan flag exempts it from strictness.
        let target = ToolRegistry::from_tool_provider(Arc::new(MockTool)).expect("target");
        target
            .restore_state(snapshot_with_external_tool())
            .expect("restore");

        let mut edited = target.export_state();
        edited
            .set_availability("mock_tool", Some(crate::ToolAvailability::Searchable))
            .expect("edit bound tool");
        target
            .apply_state(edited)
            .expect("apply accepts the snapshot it exported");
        let exported = target.export_state();
        assert!(exported.get("mcp__demo__search").unwrap().is_orphaned());
        assert_eq!(
            exported
                .get("mock_tool")
                .unwrap()
                .manifest()
                .effective_availability(),
            crate::ToolAvailability::Searchable
        );

        // But a snapshot that does NOT mark the tool orphaned still fails —
        // strictness is preserved for entries that were bound at export.
        let strict = snapshot_with_external_tool().with_generation(target.generation());
        assert!(matches!(
            target.apply_state(strict),
            Err(ReconfigureError::Validation(_))
        ));
    }

    #[test]
    fn orphan_flag_serializes_and_legacy_snapshots_deserialize_as_bound() {
        let target = ToolRegistry::from_tool_provider(Arc::new(MockTool)).expect("target");
        target
            .restore_state(snapshot_with_external_tool())
            .expect("restore");
        let value = serde_json::to_value(target.export_state()).expect("serializes");
        assert_eq!(value["tools"]["mcp__demo__search"]["orphaned"], json!(true));
        assert!(
            value["tools"]["mock_tool"].get("orphaned").is_none(),
            "bound entries omit the flag, keeping old and new snapshots byte-compatible"
        );

        let legacy: ToolStateEntry = serde_json::from_value(json!({
            "manifest": value["tools"]["mock_tool"]["manifest"]
        }))
        .expect("legacy entry without the flag deserializes");
        assert!(!legacy.is_orphaned());
    }

    #[test]
    fn remove_source_removes_all_source_tools() {
        let registry = ToolRegistry::from_tool_provider(Arc::new(MockTool)).expect("registry");
        registry
            .upsert_source(Arc::new(ExternalMockSource))
            .expect("source registered");
        registry
            .remove_source_id("external")
            .expect("source removed");
        let defs = registry.tool_manifests();
        assert!(!defs.iter().any(|def| def.name == "mcp__demo__search"));
    }

    #[test]
    fn project_tool_catalog_keeps_searchable_tools_with_catalog_metadata() {
        fn dummy_tool(name: &str) -> crate::ToolDefinition {
            let tool = crate::ToolDefinition::raw_with_id(
                format!("tool:{name}"),
                name,
                format!("desc for {name}"),
                crate::ToolDefinition::default_input_schema(),
                serde_json::json!({}),
            );
            match name {
                "read_file" => {
                    tool.with_lashlang_binding(crate::LashlangToolBinding::new(["files"], "read"))
                }
                "search_tools" => {
                    tool.with_lashlang_binding(crate::LashlangToolBinding::new(["tools"], "search"))
                }
                _ => tool,
            }
        }
        let catalog = project_tool_catalog([
            crate::ToolCatalogEntry {
                manifest: dummy_tool("read_file").manifest(),
                availability: crate::ToolAvailability::Showcased,
            },
            crate::ToolCatalogEntry {
                manifest: dummy_tool("search_tools").manifest(),
                availability: crate::ToolAvailability::Callable,
            },
        ]);
        assert_eq!(catalog.len(), 2);
        assert_eq!(catalog[0]["name"], serde_json::json!("read_file"));
        assert_eq!(
            catalog[0]["contract"]["signature"],
            serde_json::json!("await files.read({})?")
        );
        assert_eq!(catalog[0]["showcased"], serde_json::json!(true));
        assert_eq!(catalog[1]["callable"], serde_json::json!(true));
    }

    #[test]
    fn project_tool_catalog_preserves_dynamic_output_contracts() {
        fn dummy_tool(name: &str) -> crate::ToolDefinition {
            crate::ToolDefinition::raw_with_id(
                format!("tool:{name}"),
                name,
                format!("desc for {name}"),
                crate::ToolDefinition::default_input_schema(),
                serde_json::json!({}),
            )
            .with_lashlang_binding(crate::LashlangToolBinding::new(["llm"], "query"))
        }
        let catalog = project_tool_catalog([crate::ToolCatalogEntry {
            manifest: dummy_tool("llm_query")
                .with_output_from_input_schema(
                    "output",
                    Some(serde_json::json!({ "type": "string" })),
                )
                .manifest(),
            availability: crate::ToolAvailability::Searchable,
        }]);

        assert_eq!(
            catalog[0]["contract"]["signature"],
            serde_json::json!("await llm.query<T = str>({})?")
        );
        assert_eq!(catalog[0]["contract"]["returns"], serde_json::json!("T"));
    }
}

pub(crate) fn project_tool_catalog<I>(entries: I) -> Vec<serde_json::Value>
where
    I: IntoIterator<Item = crate::ToolCatalogEntry>,
{
    entries
        .into_iter()
        .filter(|entry| entry.availability.is_searchable())
        .map(|entry| {
            let manifest = entry.manifest;
            let availability = entry.availability;
            let lashlang_binding = manifest.lashlang_binding.executable_for(&manifest.name);
            let call = lashlang_binding.call_path();
            let mut projected = serde_json::json!({
                "id": manifest.id,
                "name": manifest.name,
                "module_path": lashlang_binding.module_path,
                "operation": lashlang_binding.operation,
                "authority_type": lashlang_binding.authority_type,
                "call": call,
                "description": manifest.description,
                "aliases": lashlang_binding.aliases,
                "availability": availability,
                "callable": availability.is_callable(),
                "showcased": availability.is_showcased(),
                "searchable": availability.is_searchable(),
                "activation": manifest.activation,
            });
            if let Some(contract) = manifest.compact_contract {
                projected
                    .as_object_mut()
                    .expect("projected tool catalog entry is an object")
                    .insert("contract".to_string(), serde_json::json!(contract));
            }
            projected
        })
        .collect()
}
