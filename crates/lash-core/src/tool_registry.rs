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
}

impl ToolStateEntry {
    pub(crate) fn new(manifest: ToolManifest) -> Self {
        Self { manifest }
    }

    pub fn manifest(&self) -> &ToolManifest {
        &self.manifest
    }

    pub fn manifest_mut(&mut self) -> &mut ToolManifest {
        &mut self.manifest
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
        self.tools
            .values()
            .map(|entry| entry.manifest.clone())
            .collect()
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
    providers: Vec<(Arc<dyn ToolProvider>, Vec<String>)>,
}

impl ToolProviderGroupSource {
    fn new(id: impl Into<String>, providers: Vec<Arc<dyn ToolProvider>>) -> Self {
        let mut tools = BTreeMap::new();
        let mut entries = Vec::new();
        for provider in providers {
            let tool_names = provider
                .tool_manifests()
                .into_iter()
                .map(|manifest| {
                    let name = manifest.name.clone();
                    tools.insert(name.clone(), (manifest, entries.len()));
                    name
                })
                .collect::<Vec<_>>();
            entries.push((provider, tool_names));
        }
        Self {
            id: id.into(),
            tools: RwLock::new(tools),
            providers: entries,
        }
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
        self.tools
            .read()
            .expect("tool provider group lock poisoned")
            .values()
            .map(|(manifest, _)| manifest.clone())
            .collect()
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
        for (provider_idx, (provider, _)) in self.providers.iter().enumerate() {
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
        self.providers[provider_idx].0.resolve_contract(name)
    }

    async fn prepare_tool_call(
        &self,
        call: ToolPrepareCall<'_>,
    ) -> Result<PreparedToolCall, ToolResult> {
        let name = call.pending.tool_name.clone();
        let Some(provider_idx) = self.provider_index_for(&name) else {
            return Err(ToolResult::err_fmt(format_args!("Unknown tool: {name}")));
        };
        self.providers[provider_idx].0.prepare_tool_call(call).await
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
            .0
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

#[derive(Clone)]
struct ToolRegistryEntry {
    manifest: ToolManifest,
    source_id: String,
}

impl ToolRegistryEntry {
    fn new(manifest: ToolManifest, source_id: impl Into<String>) -> Self {
        Self {
            manifest,
            source_id: source_id.into(),
        }
    }

    fn export(&self) -> ToolStateEntry {
        ToolStateEntry::new(self.manifest.clone())
    }
}

#[derive(Clone)]
struct ToolRegistryState {
    generation: u64,
    tools: BTreeMap<String, ToolRegistryEntry>,
    next_live_source_id: u64,
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
        let rebound_tools = {
            let sources = self.sources.read().expect("tool source lock poisoned");
            rebind_tool_state_entries(next.entries(), &sources)?
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
        state.tools = rebound_tools;
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
    /// resume) restore a session whose tool surface reached generation `G ≥ 2`
    /// onto a base registry at generation 1 — `apply_state` would reject that
    /// (`expected G, actual 1`); `restore_state` adopts `G`. Entries are still
    /// rebound to the live sources, so source identity is reconnected.
    pub fn restore_state(&self, snapshot: ToolState) -> Result<u64, ReconfigureError> {
        validate_unique_manifest_entries(snapshot.entries().values())?;
        let rebound_tools = {
            let sources = self.sources.read().expect("tool source lock poisoned");
            rebind_tool_state_entries(snapshot.entries(), &sources)?
        };

        let mut state = self
            .state
            .write()
            .expect("tool registry state lock poisoned");
        state.tools = rebound_tools;
        state.generation = snapshot.generation();
        Ok(state.generation)
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

    pub(crate) fn compose_session_surface(
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
        let source_id = source.id().to_string();
        let advertised_tools = source
            .advertised_tools()
            .into_iter()
            .map(|manifest| manifest_with_compact_contract(source.as_ref(), manifest))
            .collect::<Vec<_>>();
        validate_unique_manifests(&advertised_tools)?;

        let mut state = self
            .state
            .write()
            .expect("tool registry state lock poisoned");
        let previous_overrides = state
            .tools
            .iter()
            .map(|(name, entry)| (name.clone(), entry.manifest.availability_override))
            .collect::<BTreeMap<_, _>>();
        let same_source_names = state
            .tools
            .iter()
            .filter_map(|(name, entry)| (entry.source_id == source_id).then_some(name.clone()))
            .collect::<BTreeSet<_>>();
        for manifest in &advertised_tools {
            if let Some(existing) = state.tools.get(&manifest.name)
                && existing.source_id != source_id
            {
                return Err(ReconfigureError::Validation(format!(
                    "duplicate tool name `{}` from source `{}` conflicts with source `{}`",
                    manifest.name, source_id, existing.source_id
                )));
            }
            if let Some((existing_name, existing)) = state
                .tools
                .iter()
                .find(|(_, entry)| entry.source_id != source_id && entry.manifest.id == manifest.id)
            {
                return Err(ReconfigureError::Validation(format!(
                    "duplicate tool id `{}` from source `{}` conflicts with tool `{}` from source `{}`",
                    manifest.id, source_id, existing_name, existing.source_id
                )));
            }
        }
        state.tools.retain(|name, entry| {
            entry.source_id != source_id || !same_source_names.contains(name)
        });

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

    fn upsert_overlay_source(
        &self,
        source: Arc<dyn ToolSourceExecutor>,
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
        state.tools.retain(|name, entry| {
            entry.source_id != source_id
                && !advertised_names.contains(name)
                && !advertised_ids.contains(&entry.manifest.id)
        });
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
        state.tools.retain(|_, entry| entry.source_id != source_id);
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
        let tools = rebind_tool_state_entries(snapshot.entries(), &sources)?;
        let generation = snapshot.generation.max(1);
        Ok(Self {
            sources: Arc::new(RwLock::new(sources)),
            state: Arc::new(RwLock::new(ToolRegistryState {
                generation,
                tools,
                next_live_source_id: 0,
            })),
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
            .map(|entry| entry.manifest.clone())
            .collect()
    }

    fn resolve_manifest(&self, name: &str) -> Option<ToolManifest> {
        if let Some(manifest) = {
            let state = self
                .state
                .read()
                .expect("tool registry state lock poisoned");
            state.tools.get(name).map(|entry| entry.manifest.clone())
        } {
            return Some(manifest);
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
                return (existing.source_id == source_id).then(|| existing.manifest.clone());
            }
            if let Some((_, existing)) = state
                .tools
                .iter()
                .find(|(_, entry)| entry.manifest.id == manifest.id)
            {
                return (existing.source_id == source_id).then(|| existing.manifest.clone());
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
            state.tools.get(name).map(|entry| entry.source_id.clone())
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
        let source_id = self.resolve_manifest(&name).and_then(|_| {
            let state = self
                .state
                .read()
                .expect("tool registry state lock poisoned");
            state.tools.get(&name).map(|entry| entry.source_id.clone())
        });
        let Some(source_id) = source_id else {
            return Err(ToolResult::err_fmt(format_args!("Unknown tool: {name}")));
        };
        let source = {
            self.sources
                .read()
                .expect("tool source lock poisoned")
                .get(&source_id)
                .cloned()
        };
        let Some(source) = source else {
            return Err(ToolResult::err_fmt(format_args!(
                "Tool source missing for tool `{name}`"
            )));
        };
        source.prepare_tool_call(call).await
    }

    async fn execute(&self, call: ToolCall<'_>) -> ToolResult {
        let name = call.name;
        let source_id = self.resolve_manifest(name).and_then(|_| {
            let state = self
                .state
                .read()
                .expect("tool registry state lock poisoned");
            state.tools.get(name).map(|entry| entry.source_id.clone())
        });
        let Some(source_id) = source_id else {
            return ToolResult::err_fmt(format_args!("Unknown tool: {name}"));
        };
        let source = {
            self.sources
                .read()
                .expect("tool source lock poisoned")
                .get(&source_id)
                .cloned()
        };
        let Some(source) = source else {
            return ToolResult::err_fmt(format_args!("Tool source missing for tool `{name}`"));
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

fn rebind_tool_state_entries(
    entries: &BTreeMap<String, ToolStateEntry>,
    sources: &BTreeMap<String, Arc<dyn ToolSourceExecutor>>,
) -> Result<BTreeMap<String, ToolRegistryEntry>, ReconfigureError> {
    let mut rebound = BTreeMap::new();
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
            return Err(ReconfigureError::Validation(format!(
                "no registered tool source resolves tool `{name}`"
            )));
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
    Ok(rebound)
}

fn validate_unique_manifest_entries<'a>(
    entries: impl IntoIterator<Item = &'a ToolStateEntry>,
) -> Result<(), ReconfigureError> {
    let manifests = entries
        .into_iter()
        .map(|entry| entry.manifest.clone())
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
        // Cold rebuild ratchet: a session whose tool surface advanced to
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
        assert_eq!(restored, 3, "restore returns the adopted generation");
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
    fn project_tool_catalog_keeps_searchable_tools_with_surface_metadata() {
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
                    tool.with_agent_surface(crate::ToolAgentSurface::new(["files"], "read"))
                }
                "search_tools" => {
                    tool.with_agent_surface(crate::ToolAgentSurface::new(["tools"], "search"))
                }
                _ => tool,
            }
        }
        let catalog = project_tool_catalog([
            crate::ToolSurfaceEntry {
                manifest: dummy_tool("read_file").manifest(),
                availability: crate::ToolAvailability::Showcased,
            },
            crate::ToolSurfaceEntry {
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
            .with_agent_surface(crate::ToolAgentSurface::new(["llm"], "query"))
        }
        let catalog = project_tool_catalog([crate::ToolSurfaceEntry {
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
    I: IntoIterator<Item = crate::ToolSurfaceEntry>,
{
    entries
        .into_iter()
        .filter(|entry| entry.availability.is_searchable())
        .map(|entry| {
            let manifest = entry.manifest;
            let availability = entry.availability;
            let agent_surface = manifest.agent_surface.executable_for(&manifest.name);
            let call = agent_surface.call_path();
            let mut projected = serde_json::json!({
                "id": manifest.id,
                "name": manifest.name,
                "module_path": agent_surface.module_path,
                "operation": agent_surface.operation,
                "authority_type": agent_surface.authority_type,
                "call": call,
                "description": manifest.description,
                "aliases": agent_surface.aliases,
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
