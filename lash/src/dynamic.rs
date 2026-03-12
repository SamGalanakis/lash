use std::collections::{BTreeMap, BTreeSet};
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, RwLock};

use serde::{Deserialize, Serialize};

use crate::capabilities::{AgentCapabilities, CAPABILITY_DEFINITIONS, CapabilityId};
use crate::{ProgressSender, ToolDefinition, ToolProvider, ToolResult};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DynamicCapabilityDef {
    pub id: String,
    pub name: String,
    pub description: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_section: Option<String>,
    #[serde(default)]
    pub helper_bindings: BTreeSet<String>,
    #[serde(default)]
    pub tool_names: BTreeSet<String>,
    #[serde(default)]
    pub enabled_by_default: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CapabilityProfile {
    #[serde(default)]
    pub enabled_capabilities: BTreeSet<String>,
    #[serde(default)]
    pub enabled_tools: BTreeSet<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResolvedProjection {
    #[serde(default)]
    pub enabled_capabilities: BTreeSet<String>,
    #[serde(default)]
    pub effective_tools: BTreeSet<String>,
    #[serde(default)]
    pub helper_bindings: BTreeSet<String>,
    #[serde(default)]
    pub prompt_sections: Vec<String>,
}

pub fn resolve_projection(
    defs: &BTreeMap<String, DynamicCapabilityDef>,
    profile: &CapabilityProfile,
    available_tools: &BTreeSet<String>,
) -> Result<ResolvedProjection, ReconfigureError> {
    let mut effective_tools = BTreeSet::new();
    let mut helper_bindings = BTreeSet::new();
    let mut prompt_sections = Vec::new();

    for id in &profile.enabled_capabilities {
        let Some(def) = defs.get(id) else {
            return Err(ReconfigureError::Validation(format!(
                "enabled capability not defined: {id}"
            )));
        };

        for tool in &def.tool_names {
            if available_tools.contains(tool) {
                effective_tools.insert(tool.clone());
            }
        }

        for helper in &def.helper_bindings {
            if available_tools.contains(helper) {
                helper_bindings.insert(helper.clone());
            }
        }

        if let Some(section) = &def.prompt_section
            && !section.trim().is_empty()
        {
            prompt_sections.push(section.clone());
        }
    }

    for tool in &profile.enabled_tools {
        if !available_tools.contains(tool) {
            return Err(ReconfigureError::Validation(format!(
                "explicitly enabled tool not registered: {tool}"
            )));
        }
        effective_tools.insert(tool.clone());
    }

    Ok(ResolvedProjection {
        enabled_capabilities: profile.enabled_capabilities.clone(),
        effective_tools,
        helper_bindings,
        prompt_sections,
    })
}

pub fn default_dynamic_capability_defs() -> BTreeMap<String, DynamicCapabilityDef> {
    let mut defs = BTreeMap::new();
    for def in CAPABILITY_DEFINITIONS {
        defs.insert(
            def.id.as_str().to_string(),
            DynamicCapabilityDef {
                id: def.id.as_str().to_string(),
                name: def.name.to_string(),
                description: def.description.to_string(),
                prompt_section: def.prompt_section.map(str::to_string),
                helper_bindings: def
                    .helper_bindings
                    .iter()
                    .map(|v| (*v).to_string())
                    .collect(),
                tool_names: def.tools.iter().map(|v| (*v).to_string()).collect(),
                enabled_by_default: def.enabled_by_default,
            },
        );
    }
    #[cfg(feature = "sqlite-store")]
    defs.extend(crate::plugin::builtin_dynamic_capability_defs());
    defs
}

pub fn resolve_capability_projection(
    defs: &BTreeMap<String, DynamicCapabilityDef>,
    caps: &AgentCapabilities,
    available_defs: &[ToolDefinition],
) -> Result<ResolvedProjection, ReconfigureError> {
    let available_tools: BTreeSet<String> =
        available_defs.iter().map(|def| def.name.clone()).collect();
    resolve_projection(
        defs,
        &profile_from_agent_capabilities(caps),
        &available_tools,
    )
}

pub fn profile_from_agent_capabilities(caps: &AgentCapabilities) -> CapabilityProfile {
    CapabilityProfile {
        enabled_capabilities: caps
            .enabled_capabilities
            .iter()
            .map(|c| c.as_str().to_string())
            .collect(),
        enabled_tools: caps.enabled_tools.clone(),
    }
}

pub fn agent_capabilities_from_profile(profile: &CapabilityProfile) -> AgentCapabilities {
    let mut enabled = BTreeSet::new();
    for id in &profile.enabled_capabilities {
        if let Some(parsed) = CapabilityId::parse(id) {
            enabled.insert(parsed);
        }
    }
    AgentCapabilities {
        enabled_capabilities: enabled,
        enabled_tools: profile.enabled_tools.clone(),
    }
}

pub type InProcessToolFuture = Pin<Box<dyn Future<Output = ToolResult> + Send>>;
pub type InProcessToolHandler =
    Arc<dyn Fn(serde_json::Value, Option<ProgressSender>) -> InProcessToolFuture + Send + Sync>;

#[async_trait::async_trait]
pub trait ToolExecutionAdapter: Send + Sync + 'static {
    fn id(&self) -> &str;
    fn advertised_tools(&self) -> Vec<ToolDefinition>;
    async fn execute(
        &self,
        tool: &str,
        args: &serde_json::Value,
        progress: Option<&ProgressSender>,
    ) -> ToolResult;
}

pub struct InProcessToolExecutionAdapter {
    id: String,
    defs: RwLock<BTreeMap<String, ToolDefinition>>,
    handlers: RwLock<BTreeMap<String, InProcessToolHandler>>,
}

impl InProcessToolExecutionAdapter {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            defs: RwLock::new(BTreeMap::new()),
            handlers: RwLock::new(BTreeMap::new()),
        }
    }

    pub fn register_tool(&self, def: ToolDefinition, handler: InProcessToolHandler) {
        let name = def.name.clone();
        self.defs
            .write()
            .expect("inprocess defs lock poisoned")
            .insert(name.clone(), def);
        self.handlers
            .write()
            .expect("inprocess handlers lock poisoned")
            .insert(name, handler);
    }

    pub fn remove_tool(&self, name: &str) {
        self.defs
            .write()
            .expect("inprocess defs lock poisoned")
            .remove(name);
        self.handlers
            .write()
            .expect("inprocess handlers lock poisoned")
            .remove(name);
    }

    pub fn snapshot_registry(
        &self,
    ) -> (
        BTreeMap<String, ToolDefinition>,
        BTreeMap<String, InProcessToolHandler>,
    ) {
        (
            self.defs
                .read()
                .expect("inprocess defs lock poisoned")
                .clone(),
            self.handlers
                .read()
                .expect("inprocess handlers lock poisoned")
                .clone(),
        )
    }

    pub fn from_registry(
        id: impl Into<String>,
        defs: BTreeMap<String, ToolDefinition>,
        handlers: BTreeMap<String, InProcessToolHandler>,
    ) -> Self {
        Self {
            id: id.into(),
            defs: RwLock::new(defs),
            handlers: RwLock::new(handlers),
        }
    }
}

#[async_trait::async_trait]
impl ToolExecutionAdapter for InProcessToolExecutionAdapter {
    fn id(&self) -> &str {
        &self.id
    }

    fn advertised_tools(&self) -> Vec<ToolDefinition> {
        self.defs
            .read()
            .expect("inprocess defs lock poisoned")
            .values()
            .cloned()
            .collect()
    }

    async fn execute(
        &self,
        tool: &str,
        args: &serde_json::Value,
        progress: Option<&ProgressSender>,
    ) -> ToolResult {
        let Some(handler) = self
            .handlers
            .read()
            .expect("inprocess handlers lock poisoned")
            .get(tool)
            .cloned()
        else {
            return ToolResult::err_fmt(format_args!("Unknown tool: {tool}"));
        };
        let progress = progress.cloned();
        handler(args.clone(), progress).await
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DynamicToolSpec {
    pub definition: ToolDefinition,
    pub adapter_id: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DynamicStateSnapshot {
    pub base_generation: u64,
    pub tools: BTreeMap<String, DynamicToolSpec>,
    pub capability_defs: BTreeMap<String, DynamicCapabilityDef>,
    pub profile: CapabilityProfile,
}

#[derive(Clone)]
struct DynamicRegistryState {
    generation: u64,
    tools: BTreeMap<String, DynamicToolSpec>,
    capability_defs: BTreeMap<String, DynamicCapabilityDef>,
    profile: CapabilityProfile,
    resolved: ResolvedProjection,
}

#[derive(Debug, thiserror::Error)]
pub enum ReconfigureError {
    #[error("validation error: {0}")]
    Validation(String),
    #[error("unknown adapter: {0}")]
    UnknownAdapter(String),
    #[error("generation mismatch: expected {expected}, actual {actual}")]
    GenerationMismatch { expected: u64, actual: u64 },
}

#[derive(Clone)]
pub struct DynamicToolProvider {
    adapters: Arc<RwLock<BTreeMap<String, Arc<dyn ToolExecutionAdapter>>>>,
    inprocess: Arc<InProcessToolExecutionAdapter>,
    state: Arc<RwLock<DynamicRegistryState>>,
}

impl DynamicToolProvider {
    pub fn from_tool_provider(
        provider: Arc<dyn ToolProvider>,
        capability_defs: BTreeMap<String, DynamicCapabilityDef>,
        mut profile: CapabilityProfile,
    ) -> Result<Self, ReconfigureError> {
        let inprocess = Arc::new(InProcessToolExecutionAdapter::new("inprocess"));

        let mut tools = BTreeMap::new();
        for def in provider.definitions() {
            let tool_name = def.name.clone();
            let delegate = Arc::clone(&provider);
            let delegate_name = tool_name.clone();
            let handler: InProcessToolHandler = Arc::new(move |args, progress| {
                let delegate = Arc::clone(&delegate);
                let delegate_name = delegate_name.clone();
                Box::pin(async move {
                    delegate
                        .execute_streaming(&delegate_name, &args, progress.as_ref())
                        .await
                })
            });
            inprocess.register_tool(def.clone(), handler);
            tools.insert(
                tool_name,
                DynamicToolSpec {
                    definition: def,
                    adapter_id: "inprocess".to_string(),
                },
            );
        }

        let mut adapters: BTreeMap<String, Arc<dyn ToolExecutionAdapter>> = BTreeMap::new();
        adapters.insert(
            "inprocess".to_string(),
            Arc::clone(&inprocess) as Arc<dyn ToolExecutionAdapter>,
        );

        let available: BTreeSet<String> = tools.keys().cloned().collect();
        let resolved = resolve_projection(&capability_defs, &profile, &available)?;
        profile.enabled_capabilities = resolved.enabled_capabilities.clone();

        Ok(Self {
            adapters: Arc::new(RwLock::new(adapters)),
            inprocess,
            state: Arc::new(RwLock::new(DynamicRegistryState {
                generation: 1,
                tools,
                capability_defs,
                profile,
                resolved,
            })),
        })
    }

    pub fn inprocess_adapter(&self) -> Arc<InProcessToolExecutionAdapter> {
        Arc::clone(&self.inprocess)
    }

    pub fn generation(&self) -> u64 {
        self.state
            .read()
            .expect("dynamic state lock poisoned")
            .generation
    }

    pub fn export_state(&self) -> DynamicStateSnapshot {
        let state = self.state.read().expect("dynamic state lock poisoned");
        DynamicStateSnapshot {
            base_generation: state.generation,
            tools: state.tools.clone(),
            capability_defs: state.capability_defs.clone(),
            profile: state.profile.clone(),
        }
    }

    pub fn profile(&self) -> CapabilityProfile {
        self.state
            .read()
            .expect("dynamic state lock poisoned")
            .profile
            .clone()
    }

    pub fn resolved_projection(&self) -> ResolvedProjection {
        self.state
            .read()
            .expect("dynamic state lock poisoned")
            .resolved
            .clone()
    }

    pub fn capabilities_payload_json(&self) -> String {
        let resolved = self.resolved_projection();
        let profile = self.profile();
        serde_json::json!({
            "enabled_capabilities": resolved.enabled_capabilities.into_iter().collect::<Vec<_>>(),
            "enabled_tools": resolved.effective_tools.into_iter().collect::<Vec<_>>(),
            "helper_bindings": resolved.helper_bindings.into_iter().collect::<Vec<_>>(),
            "explicit_enabled_tools": profile.enabled_tools.into_iter().collect::<Vec<_>>(),
        })
        .to_string()
    }

    pub fn apply_state(&self, next: DynamicStateSnapshot) -> Result<u64, ReconfigureError> {
        let current_generation = self.generation();
        if next.base_generation != current_generation {
            return Err(ReconfigureError::GenerationMismatch {
                expected: next.base_generation,
                actual: current_generation,
            });
        }

        {
            let adapters = self.adapters.read().expect("adapters lock poisoned");
            for spec in next.tools.values() {
                let Some(adapter) = adapters.get(&spec.adapter_id) else {
                    return Err(ReconfigureError::UnknownAdapter(spec.adapter_id.clone()));
                };
                let advertised: BTreeSet<String> = adapter
                    .advertised_tools()
                    .into_iter()
                    .map(|d| d.name)
                    .collect();
                if !advertised.contains(&spec.definition.name) {
                    return Err(ReconfigureError::Validation(format!(
                        "adapter `{}` does not advertise tool `{}`",
                        spec.adapter_id, spec.definition.name
                    )));
                }
            }
        }

        let available: BTreeSet<String> = next.tools.keys().cloned().collect();
        let resolved = resolve_projection(&next.capability_defs, &next.profile, &available)?;
        let mut normalized_profile = next.profile.clone();
        normalized_profile.enabled_capabilities = resolved.enabled_capabilities.clone();

        let mut state = self.state.write().expect("dynamic state lock poisoned");
        if state.generation != next.base_generation {
            return Err(ReconfigureError::GenerationMismatch {
                expected: next.base_generation,
                actual: state.generation,
            });
        }
        state.tools = next.tools;
        state.capability_defs = next.capability_defs;
        state.profile = normalized_profile;
        state.resolved = resolved;
        state.generation += 1;

        Ok(state.generation)
    }

    pub fn upsert_inprocess_handler(&self, def: ToolDefinition, handler: InProcessToolHandler) {
        self.inprocess.register_tool(def, handler);
    }

    pub fn remove_inprocess_handler(&self, tool: &str) {
        self.inprocess.remove_tool(tool);
    }

    pub fn fork_with_snapshot(
        &self,
        snapshot: DynamicStateSnapshot,
    ) -> Result<Self, ReconfigureError> {
        let (defs, handlers) = self.inprocess.snapshot_registry();
        let inprocess = Arc::new(InProcessToolExecutionAdapter::from_registry(
            self.inprocess.id().to_string(),
            defs,
            handlers,
        ));

        let mut adapters: BTreeMap<String, Arc<dyn ToolExecutionAdapter>> = self
            .adapters
            .read()
            .expect("adapters lock poisoned")
            .iter()
            .map(|(k, v)| (k.clone(), Arc::clone(v)))
            .collect();
        adapters.insert(
            inprocess.id().to_string(),
            Arc::clone(&inprocess) as Arc<dyn ToolExecutionAdapter>,
        );

        let available: BTreeSet<String> = snapshot.tools.keys().cloned().collect();
        let resolved =
            resolve_projection(&snapshot.capability_defs, &snapshot.profile, &available)?;
        let mut profile = snapshot.profile;
        profile.enabled_capabilities = resolved.enabled_capabilities.clone();

        let generation = snapshot.base_generation.max(1);
        Ok(Self {
            adapters: Arc::new(RwLock::new(adapters)),
            inprocess,
            state: Arc::new(RwLock::new(DynamicRegistryState {
                generation,
                tools: snapshot.tools,
                capability_defs: snapshot.capability_defs,
                profile,
                resolved,
            })),
        })
    }
}

#[async_trait::async_trait]
impl ToolProvider for DynamicToolProvider {
    fn definitions(&self) -> Vec<ToolDefinition> {
        let state = self.state.read().expect("dynamic state lock poisoned");
        state
            .tools
            .values()
            .filter(|spec| {
                state
                    .resolved
                    .effective_tools
                    .contains(&spec.definition.name)
            })
            .map(|spec| spec.definition.clone())
            .collect()
    }

    fn dynamic_projection(&self) -> Option<ResolvedProjection> {
        Some(self.resolved_projection())
    }

    fn dynamic_snapshot(&self) -> Option<DynamicStateSnapshot> {
        Some(self.export_state())
    }

    fn fork_dynamic_with_snapshot(
        &self,
        snapshot: DynamicStateSnapshot,
    ) -> Option<Arc<dyn ToolProvider>> {
        self.fork_with_snapshot(snapshot)
            .ok()
            .map(|fork| Arc::new(fork) as Arc<dyn ToolProvider>)
    }

    fn dynamic_capabilities_payload_json(&self) -> Option<String> {
        Some(self.capabilities_payload_json())
    }

    fn dynamic_generation(&self) -> Option<u64> {
        Some(self.generation())
    }
    async fn execute(&self, name: &str, args: &serde_json::Value) -> ToolResult {
        self.execute_streaming(name, args, None).await
    }

    async fn execute_streaming(
        &self,
        name: &str,
        args: &serde_json::Value,
        progress: Option<&ProgressSender>,
    ) -> ToolResult {
        let (adapter_id, allowed, catalog) = {
            let state = self.state.read().expect("dynamic state lock poisoned");
            let allowed = state.resolved.effective_tools.contains(name);
            let adapter_id = state.tools.get(name).map(|s| s.adapter_id.clone());
            let catalog = if matches!(name, "list_tools" | "search_tools")
                && args.get("catalog").is_none()
            {
                Some(
                    state
                        .tools
                        .values()
                        .filter(|spec| {
                            state
                                .resolved
                                .effective_tools
                                .contains(&spec.definition.name)
                        })
                        .map(|spec| {
                            let projected = spec.definition.project(crate::ExecutionMode::Standard);
                            serde_json::json!({
                                "name": projected.name,
                                "description": projected.description,
                                "params": projected.params,
                                "returns": projected.returns,
                                "examples": projected.examples,
                                "inject_into_prompt": projected.inject_into_prompt,
                                "hidden": projected.hidden,
                            })
                        })
                        .collect::<Vec<_>>(),
                )
            } else {
                None
            };
            (adapter_id, allowed, catalog)
        };

        if !allowed {
            return ToolResult::err_fmt(format_args!("Unknown tool: {name}"));
        }

        let Some(adapter_id) = adapter_id else {
            return ToolResult::err_fmt(format_args!("Unknown tool: {name}"));
        };

        let adapter = {
            self.adapters
                .read()
                .expect("adapters lock poisoned")
                .get(&adapter_id)
                .cloned()
        };

        let Some(adapter) = adapter else {
            return ToolResult::err_fmt(format_args!("Tool adapter missing for tool `{name}`"));
        };

        let payload = if let Some(catalog) = catalog {
            let mut object = args.as_object().cloned().unwrap_or_default();
            object.insert("catalog".to_string(), serde_json::Value::Array(catalog));
            serde_json::Value::Object(object)
        } else {
            args.clone()
        };

        adapter.execute(name, &payload, progress).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockTool;

    #[async_trait::async_trait]
    impl ToolProvider for MockTool {
        fn definitions(&self) -> Vec<ToolDefinition> {
            vec![ToolDefinition {
                name: "mock_tool".to_string(),
                description: vec![crate::ToolText::new(
                    "mock",
                    [crate::ExecutionMode::Repl, crate::ExecutionMode::Standard],
                )],
                params: vec![],
                returns: "str".to_string(),
                examples: vec![],
                hidden: false,
                inject_into_prompt: false,
            }]
        }

        async fn execute(&self, _name: &str, _args: &serde_json::Value) -> ToolResult {
            ToolResult::ok(serde_json::json!("ok"))
        }
    }

    #[tokio::test]
    async fn dynamic_provider_executes_enabled_tool() {
        let provider: Arc<dyn ToolProvider> = Arc::new(MockTool);
        let mut enabled_tools = BTreeSet::new();
        enabled_tools.insert("mock_tool".to_string());
        let registry = DynamicToolProvider::from_tool_provider(
            provider,
            BTreeMap::new(),
            CapabilityProfile {
                enabled_capabilities: BTreeSet::new(),
                enabled_tools,
            },
        )
        .expect("registry builds");

        let result = registry.execute("mock_tool", &serde_json::json!({})).await;
        assert!(result.success);
        assert_eq!(result.result, serde_json::json!("ok"));
    }

    #[tokio::test]
    async fn dynamic_provider_blocks_disabled_tool() {
        let provider: Arc<dyn ToolProvider> = Arc::new(MockTool);
        let registry = DynamicToolProvider::from_tool_provider(
            provider,
            BTreeMap::new(),
            CapabilityProfile {
                enabled_capabilities: BTreeSet::new(),
                enabled_tools: BTreeSet::new(),
            },
        )
        .expect("registry builds");

        let result = registry.execute("mock_tool", &serde_json::json!({})).await;
        assert!(!result.success);
    }

    #[test]
    fn apply_state_increments_generation() {
        let provider: Arc<dyn ToolProvider> = Arc::new(MockTool);
        let mut enabled_tools = BTreeSet::new();
        enabled_tools.insert("mock_tool".to_string());
        let registry = DynamicToolProvider::from_tool_provider(
            provider,
            BTreeMap::new(),
            CapabilityProfile {
                enabled_capabilities: BTreeSet::new(),
                enabled_tools,
            },
        )
        .expect("registry builds");

        let mut snapshot = registry.export_state();
        snapshot.profile.enabled_tools.clear();
        let generation = registry.apply_state(snapshot).expect("apply succeeds");
        assert_eq!(generation, 2);
    }

    #[test]
    fn resolve_projection_rejects_missing_capability() {
        let mut enabled_caps = BTreeSet::new();
        enabled_caps.insert("missing".to_string());
        let result = resolve_projection(
            &BTreeMap::new(),
            &CapabilityProfile {
                enabled_capabilities: enabled_caps,
                enabled_tools: BTreeSet::new(),
            },
            &BTreeSet::new(),
        );
        assert!(result.is_err());
    }

    #[test]
    fn resolve_projection_skips_unavailable_capability_tools() {
        let mut defs = BTreeMap::new();
        defs.insert(
            "planning".to_string(),
            DynamicCapabilityDef {
                id: "planning".to_string(),
                name: "Planning".to_string(),
                description: "Plan tracking tools".to_string(),
                prompt_section: Some("plan section".to_string()),
                helper_bindings: BTreeSet::new(),
                tool_names: BTreeSet::from(["update_plan".to_string()]),
                enabled_by_default: true,
            },
        );

        let profile = CapabilityProfile {
            enabled_capabilities: BTreeSet::from(["planning".to_string()]),
            enabled_tools: BTreeSet::new(),
        };
        let available_tools = BTreeSet::new();

        let result = resolve_projection(&defs, &profile, &available_tools);
        assert!(result.is_ok());
        assert!(result.unwrap().effective_tools.is_empty());
    }
}
