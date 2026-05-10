#![allow(clippy::items_after_test_module)]

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, RwLock};

use serde::{Deserialize, Serialize};

use crate::{ProgressSender, ToolCall, ToolContext, ToolDefinition, ToolProvider, ToolResult};

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
    definition: ToolDefinition,
    source_id: String,
}

impl ToolStateEntry {
    pub fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    pub fn definition_mut(&mut self) -> &mut ToolDefinition {
        &mut self.definition
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ToolState {
    generation: u64,
    tools: BTreeMap<String, ToolStateEntry>,
}

impl ToolState {
    pub(crate) fn new(generation: u64, tools: BTreeMap<String, ToolStateEntry>) -> Self {
        Self { generation, tools }
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn with_generation(mut self, generation: u64) -> Self {
        self.generation = generation;
        self
    }

    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.tools
            .values()
            .map(|entry| entry.definition.clone())
            .collect()
    }

    pub fn get(&self, name: &str) -> Option<&ToolStateEntry> {
        self.tools.get(name)
    }

    pub fn definition_mut(&mut self, name: &str) -> Option<&mut ToolDefinition> {
        self.tools.get_mut(name).map(|entry| &mut entry.definition)
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
        let Some(entry) = self.tools.get_mut(name) else {
            return Err(ReconfigureError::Validation(format!(
                "unknown tool `{name}`"
            )));
        };
        entry.definition.availability_override = availability;
        Ok(())
    }

    pub fn retain(&mut self, mut keep: impl FnMut(&str, &ToolStateEntry) -> bool) {
        self.tools.retain(|name, entry| keep(name, entry));
    }

    pub fn remove(&mut self, name: &str) -> Option<ToolStateEntry> {
        self.tools.remove(name)
    }

    pub(crate) fn entries(&self) -> &BTreeMap<String, ToolStateEntry> {
        &self.tools
    }

    pub(crate) fn into_entries(self) -> BTreeMap<String, ToolStateEntry> {
        self.tools
    }
}

#[async_trait::async_trait]
pub(crate) trait ToolSourceExecutor: Send + Sync + 'static {
    fn id(&self) -> &str;
    fn advertised_tools(&self) -> Vec<ToolDefinition>;
    async fn execute(
        &self,
        tool: &str,
        args: &serde_json::Value,
        context: &ToolContext,
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

#[async_trait::async_trait]
impl ToolSourceExecutor for ToolProviderSource {
    fn id(&self) -> &str {
        &self.id
    }

    fn advertised_tools(&self) -> Vec<ToolDefinition> {
        self.provider.definitions()
    }

    async fn execute(
        &self,
        tool: &str,
        args: &serde_json::Value,
        context: &ToolContext,
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
struct ToolRegistryState {
    generation: u64,
    tools: BTreeMap<String, ToolStateEntry>,
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
        ToolState::new(state.generation, state.tools.clone())
    }

    pub fn apply_state(&self, next: ToolState) -> Result<u64, ReconfigureError> {
        let current_generation = self.generation();
        if next.generation != current_generation {
            return Err(ReconfigureError::GenerationMismatch {
                expected: next.generation,
                actual: current_generation,
            });
        }

        {
            let sources = self.sources.read().expect("tool source lock poisoned");
            for entry in next.entries().values() {
                let Some(source) = sources.get(&entry.source_id) else {
                    return Err(ReconfigureError::UnknownSource(entry.source_id.clone()));
                };
                let advertised: BTreeSet<String> = source
                    .advertised_tools()
                    .into_iter()
                    .map(|d| d.name)
                    .collect();
                if !advertised.contains(&entry.definition.name) {
                    return Err(ReconfigureError::Validation(format!(
                        "tool source `{}` does not advertise tool `{}`",
                        entry.source_id, entry.definition.name
                    )));
                }
            }
        }

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
        state.tools = next.into_entries();
        state.generation += 1;
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

    pub(crate) fn upsert_source(
        &self,
        source: Arc<dyn ToolSourceExecutor>,
    ) -> Result<u64, ReconfigureError> {
        let source_id = source.id().to_string();
        let advertised_tools = source.advertised_tools();
        validate_unique_definitions(&advertised_tools)?;

        let mut state = self
            .state
            .write()
            .expect("tool registry state lock poisoned");
        let previous_overrides = state
            .tools
            .iter()
            .map(|(name, entry)| (name.clone(), entry.definition.availability_override))
            .collect::<BTreeMap<_, _>>();
        let same_source_names = state
            .tools
            .iter()
            .filter_map(|(name, entry)| (entry.source_id == source_id).then_some(name.clone()))
            .collect::<BTreeSet<_>>();
        for def in &advertised_tools {
            if let Some(existing) = state.tools.get(&def.name)
                && existing.source_id != source_id
            {
                return Err(ReconfigureError::Validation(format!(
                    "duplicate tool name `{}` from source `{}` conflicts with source `{}`",
                    def.name, source_id, existing.source_id
                )));
            }
        }
        state.tools.retain(|name, entry| {
            entry.source_id != source_id || !same_source_names.contains(name)
        });

        for mut def in advertised_tools {
            let name = def.name.clone();
            def.availability_override = previous_overrides
                .get(&name)
                .copied()
                .flatten()
                .or(def.availability_override);
            state.tools.insert(
                name,
                ToolStateEntry {
                    definition: def,
                    source_id: source_id.clone(),
                },
            );
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
            .collect();
        let generation = snapshot.generation.max(1);
        Ok(Self {
            sources: Arc::new(RwLock::new(sources)),
            state: Arc::new(RwLock::new(ToolRegistryState {
                generation,
                tools: snapshot.into_entries(),
                next_live_source_id: 0,
            })),
        })
    }
}

#[async_trait::async_trait]
impl ToolProvider for ToolRegistry {
    fn definitions(&self) -> Vec<ToolDefinition> {
        self.export_state().definitions()
    }

    async fn execute(&self, call: ToolCall<'_>) -> ToolResult {
        let name = call.name;
        let source_id = {
            let state = self
                .state
                .read()
                .expect("tool registry state lock poisoned");
            state.tools.get(name).map(|entry| entry.source_id.clone())
        };
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

fn validate_unique_definitions(defs: &[ToolDefinition]) -> Result<(), ReconfigureError> {
    let mut names = BTreeSet::new();
    for def in defs {
        if def.name.trim().is_empty() {
            return Err(ReconfigureError::Validation(
                "tool name cannot be empty".to_string(),
            ));
        }
        if !names.insert(def.name.clone()) {
            return Err(ReconfigureError::Validation(format!(
                "duplicate tool name `{}` in source",
                def.name
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    struct MockTool;
    struct MixedEnabledTool;
    struct ExternalMockSource;

    fn test_tool(
        name: &str,
        description: &str,
        availability: crate::ToolAvailabilityConfig,
    ) -> ToolDefinition {
        ToolDefinition::raw(
            name,
            description,
            ToolDefinition::default_input_schema(),
            json!({ "type": "string" }),
        )
        .with_availability(availability)
    }

    #[async_trait::async_trait]
    impl ToolProvider for MockTool {
        fn definitions(&self) -> Vec<ToolDefinition> {
            vec![test_tool(
                "mock_tool",
                "mock",
                crate::ToolAvailabilityConfig::callable(),
            )]
        }

        async fn execute(&self, _call: ToolCall<'_>) -> ToolResult {
            ToolResult::ok(serde_json::json!("ok"))
        }
    }

    #[async_trait::async_trait]
    impl ToolProvider for MixedEnabledTool {
        fn definitions(&self) -> Vec<ToolDefinition> {
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
            ]
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

        fn advertised_tools(&self) -> Vec<ToolDefinition> {
            vec![ToolDefinition::raw(
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
            )]
        }

        async fn execute(
            &self,
            tool: &str,
            args: &serde_json::Value,
            _context: &ToolContext,
            _progress: Option<&ProgressSender>,
        ) -> ToolResult {
            ToolResult::ok(json!({
                "tool": tool,
                "args": args
            }))
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
                .definition()
                .effective_availability(&crate::ExecutionMode::standard()),
            crate::ToolAvailability::Callable
        );
        assert_eq!(
            snapshot
                .get("disabled_tool")
                .unwrap()
                .definition()
                .effective_availability(&crate::ExecutionMode::standard()),
            crate::ToolAvailability::Off
        );
    }

    #[test]
    fn apply_state_rejects_tools_not_advertised_by_source() {
        let registry = ToolRegistry::from_tool_provider(Arc::new(MockTool)).expect("registry");
        let mut snapshot = registry.export_state();
        snapshot.tools.insert(
            "missing".to_string(),
            ToolStateEntry {
                definition: test_tool(
                    "missing",
                    "missing",
                    crate::ToolAvailabilityConfig::callable(),
                ),
                source_id: PLUGIN_SOURCE_ID.to_string(),
            },
        );
        assert!(matches!(
            registry.apply_state(snapshot),
            Err(ReconfigureError::Validation(_))
        ));
    }

    #[tokio::test]
    async fn upsert_source_registers_and_executes_external_tools() {
        let registry = ToolRegistry::from_tool_provider(Arc::new(MockTool)).expect("registry");
        registry
            .upsert_source(Arc::new(ExternalMockSource))
            .expect("source registered");

        let defs = registry.definitions();
        assert!(defs.iter().any(|def| def.name == "mcp__demo__search"));

        let context = crate::ToolContext::new(
            "registry-test".to_string(),
            Arc::new(crate::testing::MockSessionManager::default()),
            crate::TurnContext::default(),
            None,
        );
        let args = json!({ "query": "hello" });
        let result = registry
            .execute(crate::ToolCall {
                name: "mcp__demo__search",
                args: &args,
                context: &context,
                progress: None,
            })
            .await;
        assert!(result.success);
        assert_eq!(result.result["tool"], json!("mcp__demo__search"));
        assert_eq!(result.result["args"]["query"], json!("hello"));
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
                .definition()
                .effective_availability(&crate::ExecutionMode::standard()),
            crate::ToolAvailability::Off
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
        let defs = registry.definitions();
        assert!(!defs.iter().any(|def| def.name == "mcp__demo__search"));
    }

    #[test]
    fn project_tool_catalog_keeps_searchable_tools_with_surface_metadata() {
        fn dummy_tool(name: &str) -> crate::ToolDefinition {
            crate::ToolDefinition::raw(
                name,
                format!("desc for {name}"),
                crate::ToolDefinition::default_input_schema(),
                serde_json::json!({}),
            )
        }
        let catalog = project_tool_catalog([
            crate::ToolSurfaceEntry {
                definition: dummy_tool("read_file"),
                availability: crate::ToolAvailability::Showcased,
            },
            crate::ToolSurfaceEntry {
                definition: dummy_tool("search_tools"),
                availability: crate::ToolAvailability::Callable,
            },
        ]);
        assert_eq!(catalog.len(), 2);
        assert_eq!(catalog[0]["name"], serde_json::json!("read_file"));
        assert_eq!(
            catalog[0]["signature"],
            serde_json::json!("read_file() -> any")
        );
        assert_eq!(catalog[0]["showcased"], serde_json::json!(true));
        assert_eq!(catalog[1]["callable"], serde_json::json!(true));
    }

    #[test]
    fn project_tool_catalog_preserves_dynamic_output_contracts() {
        fn dummy_tool(name: &str) -> crate::ToolDefinition {
            crate::ToolDefinition::raw(
                name,
                format!("desc for {name}"),
                crate::ToolDefinition::default_input_schema(),
                serde_json::json!({}),
            )
        }
        let catalog = project_tool_catalog([crate::ToolSurfaceEntry {
            definition: dummy_tool("llm_query").with_output_from_input_schema(
                "output",
                Some(serde_json::json!({ "type": "string" })),
            ),
            availability: crate::ToolAvailability::Searchable,
        }]);

        assert_eq!(
            catalog[0]["output_contract"],
            serde_json::json!({
                "kind": "from_input_schema",
                "input_field": "output",
                "default_schema": { "type": "string" }
            })
        );
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
            let definition = entry.definition;
            let availability = entry.availability;
            let signature = definition.compact_contract().render_signature();
            let loadable = matches!(definition.activation, crate::ToolActivation::Loadable);
            let activation_hint = if loadable && !availability.is_callable() {
                format!(
                    "Call `load_tools(names=[\"{}\"])` to make this tool callable in the current session.",
                    definition.name
                )
            } else if matches!(definition.activation, crate::ToolActivation::Internal) {
                "This tool is internal and cannot be activated directly.".to_string()
            } else {
                String::new()
            };
            let mut projected = serde_json::json!({
                "name": definition.name,
                "signature": signature,
                "namespace": definition.discovery.namespace,
                "description": definition.description,
                "params": definition.parameter_metadata(),
                "input_schema": definition.input_schema,
                "output_schema": definition.output_schema,
                "examples": definition.examples,
                "aliases": definition.discovery.aliases,
                "availability": availability,
                "callable": availability.is_callable(),
                "showcased": availability.is_showcased(),
                "searchable": availability.is_searchable(),
                "activation": definition.activation,
                "loadable": loadable,
                "activation_hint": activation_hint,
            });
            if !definition.output_contract.is_static()
                && let Some(object) = projected.as_object_mut()
            {
                object.insert(
                    "output_contract".to_string(),
                    serde_json::json!(definition.output_contract),
                );
            }
            projected
        })
        .collect()
}
