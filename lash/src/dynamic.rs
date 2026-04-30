use std::collections::{BTreeMap, BTreeSet};
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, RwLock};

use serde::{Deserialize, Serialize};

use crate::{ProgressSender, ToolDefinition, ToolExecutionContext, ToolProvider, ToolResult};

pub type InProcessToolFuture = Pin<Box<dyn Future<Output = ToolResult> + Send>>;
pub type InProcessToolHandler = Arc<
    dyn Fn(
            serde_json::Value,
            Option<ToolExecutionContext>,
            Option<ProgressSender>,
        ) -> InProcessToolFuture
        + Send
        + Sync,
>;

#[async_trait::async_trait]
pub trait ToolExecutionAdapter: Send + Sync + 'static {
    fn id(&self) -> &str;
    fn advertised_tools(&self) -> Vec<ToolDefinition>;
    async fn execute(
        &self,
        tool: &str,
        args: &serde_json::Value,
        context: Option<ToolExecutionContext>,
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
        context: Option<ToolExecutionContext>,
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
        handler(args.clone(), context, progress).await
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
}

#[derive(Clone)]
struct DynamicRegistryState {
    generation: u64,
    tools: BTreeMap<String, DynamicToolSpec>,
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
    pub fn from_tool_provider(provider: Arc<dyn ToolProvider>) -> Result<Self, ReconfigureError> {
        let inprocess = Arc::new(InProcessToolExecutionAdapter::new("inprocess"));

        let mut tools = BTreeMap::new();
        for def in provider.definitions() {
            let tool_name = def.name.clone();
            let delegate = Arc::clone(&provider);
            let delegate_name = tool_name.clone();
            let handler: InProcessToolHandler = Arc::new(move |args, context, progress| {
                let delegate = Arc::clone(&delegate);
                let delegate_name = delegate_name.clone();
                Box::pin(async move {
                    match context.as_ref() {
                        Some(context) => {
                            delegate
                                .execute_streaming_with_context(
                                    &delegate_name,
                                    &args,
                                    context,
                                    progress.as_ref(),
                                )
                                .await
                        }
                        None => {
                            delegate
                                .execute_streaming(&delegate_name, &args, progress.as_ref())
                                .await
                        }
                    }
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

        Ok(Self {
            adapters: Arc::new(RwLock::new(adapters)),
            inprocess,
            state: Arc::new(RwLock::new(DynamicRegistryState {
                generation: 1,
                tools,
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
        }
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

        let mut state = self.state.write().expect("dynamic state lock poisoned");
        if state.generation != next.base_generation {
            return Err(ReconfigureError::GenerationMismatch {
                expected: next.base_generation,
                actual: state.generation,
            });
        }
        state.tools = next.tools;
        state.generation += 1;

        Ok(state.generation)
    }

    pub fn upsert_adapter(
        &self,
        adapter: Arc<dyn ToolExecutionAdapter>,
    ) -> Result<u64, ReconfigureError> {
        let adapter_id = adapter.id().to_string();
        let advertised_tools = adapter.advertised_tools();
        {
            let mut adapters = self.adapters.write().expect("adapters lock poisoned");
            adapters.insert(adapter_id.clone(), adapter);
        }

        let mut state = self.state.write().expect("dynamic state lock poisoned");
        let previous_overrides = state
            .tools
            .iter()
            .map(|(name, spec)| (name.clone(), spec.definition.availability_override))
            .collect::<BTreeMap<_, _>>();
        state.tools.retain(|_, spec| spec.adapter_id != adapter_id);

        for mut def in advertised_tools {
            let name = def.name.clone();
            def.availability_override = previous_overrides
                .get(&name)
                .copied()
                .flatten()
                .or(def.availability_override);
            state.tools.insert(
                name.clone(),
                DynamicToolSpec {
                    definition: def,
                    adapter_id: adapter_id.clone(),
                },
            );
        }

        state.generation += 1;
        Ok(state.generation)
    }

    pub fn remove_adapter(&self, adapter_id: &str) -> Result<u64, ReconfigureError> {
        {
            let mut adapters = self.adapters.write().expect("adapters lock poisoned");
            if adapters.remove(adapter_id).is_none() {
                return Err(ReconfigureError::UnknownAdapter(adapter_id.to_string()));
            }
        }

        let mut state = self.state.write().expect("dynamic state lock poisoned");
        state.tools.retain(|_, spec| spec.adapter_id != adapter_id);
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

        let generation = snapshot.base_generation.max(1);
        Ok(Self {
            adapters: Arc::new(RwLock::new(adapters)),
            inprocess,
            state: Arc::new(RwLock::new(DynamicRegistryState {
                generation,
                tools: snapshot.tools,
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
            .map(|spec| spec.definition.clone())
            .collect()
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

    fn dynamic_generation(&self) -> Option<u64> {
        Some(self.generation())
    }

    async fn execute(&self, name: &str, args: &serde_json::Value) -> ToolResult {
        self.execute_streaming(name, args, None).await
    }

    async fn execute_with_context(
        &self,
        name: &str,
        args: &serde_json::Value,
        context: &ToolExecutionContext,
    ) -> ToolResult {
        self.execute_streaming_with_context(name, args, context, None)
            .await
    }

    async fn execute_streaming(
        &self,
        name: &str,
        args: &serde_json::Value,
        progress: Option<&ProgressSender>,
    ) -> ToolResult {
        let adapter_id = {
            let state = self.state.read().expect("dynamic state lock poisoned");
            state.tools.get(name).map(|spec| spec.adapter_id.clone())
        };

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

        adapter.execute(name, args, None, progress).await
    }

    async fn execute_streaming_with_context(
        &self,
        name: &str,
        args: &serde_json::Value,
        context: &ToolExecutionContext,
        progress: Option<&ProgressSender>,
    ) -> ToolResult {
        let adapter_id = {
            let state = self.state.read().expect("dynamic state lock poisoned");
            state.tools.get(name).map(|spec| spec.adapter_id.clone())
        };

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

        adapter
            .execute(name, args, Some(context.clone()), progress)
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use serde_json::json;

    use crate::ToolParam;

    struct MockTool;
    struct MixedEnabledTool;
    struct ExternalMockAdapter;
    struct DisabledExternalMockAdapter;

    #[async_trait::async_trait]
    impl ToolProvider for MockTool {
        fn definitions(&self) -> Vec<ToolDefinition> {
            vec![ToolDefinition {
                name: "mock_tool".to_string(),
                description: "mock".to_string(),
                params: vec![],
                returns: "str".to_string(),
                examples: vec![],
                availability: crate::ToolAvailabilityConfig::callable(),
                activation: crate::ToolActivation::Always,
                availability_override: None,
                input_schema_override: None,
                output_schema_override: None,
                discovery: Default::default(),
                execution_mode: crate::ToolExecutionMode::Parallel,
            }]
        }

        async fn execute(&self, _name: &str, _args: &serde_json::Value) -> ToolResult {
            ToolResult::ok(serde_json::json!("ok"))
        }
    }

    #[async_trait::async_trait]
    impl ToolProvider for MixedEnabledTool {
        fn definitions(&self) -> Vec<ToolDefinition> {
            vec![
                ToolDefinition {
                    name: "enabled_tool".to_string(),
                    description: "enabled".to_string(),
                    params: vec![],
                    returns: "str".to_string(),
                    examples: vec![],
                    availability: crate::ToolAvailabilityConfig::callable(),
                    activation: crate::ToolActivation::Always,
                    availability_override: None,
                    input_schema_override: None,
                    output_schema_override: None,
                    discovery: Default::default(),
                    execution_mode: crate::ToolExecutionMode::Parallel,
                },
                ToolDefinition {
                    name: "disabled_tool".to_string(),
                    description: "disabled".to_string(),
                    params: vec![],
                    returns: "str".to_string(),
                    examples: vec![],
                    availability: crate::ToolAvailabilityConfig::hidden(),
                    activation: crate::ToolActivation::Always,
                    availability_override: None,
                    input_schema_override: None,
                    output_schema_override: None,
                    discovery: Default::default(),
                    execution_mode: crate::ToolExecutionMode::Parallel,
                },
            ]
        }

        async fn execute(&self, _name: &str, _args: &serde_json::Value) -> ToolResult {
            ToolResult::ok(serde_json::json!("ok"))
        }
    }

    #[async_trait::async_trait]
    impl ToolExecutionAdapter for ExternalMockAdapter {
        fn id(&self) -> &str {
            "external"
        }

        fn advertised_tools(&self) -> Vec<ToolDefinition> {
            vec![ToolDefinition {
                name: "mcp__demo__search".to_string(),
                description: "search".to_string(),
                params: vec![ToolParam::typed("query", "str")],
                returns: "dict".to_string(),
                examples: vec![],
                availability: crate::ToolAvailabilityConfig::documented(),
                activation: crate::ToolActivation::Always,
                availability_override: None,
                input_schema_override: Some(json!({
                    "type": "object",
                    "properties": {
                        "query": { "type": "string" }
                    },
                    "required": ["query"],
                    "additionalProperties": false
                })),
                output_schema_override: None,
                discovery: Default::default(),
                execution_mode: crate::ToolExecutionMode::Parallel,
            }]
        }

        async fn execute(
            &self,
            tool: &str,
            args: &serde_json::Value,
            _context: Option<ToolExecutionContext>,
            _progress: Option<&ProgressSender>,
        ) -> ToolResult {
            ToolResult::ok(json!({
                "tool": tool,
                "args": args
            }))
        }
    }

    #[async_trait::async_trait]
    impl ToolExecutionAdapter for DisabledExternalMockAdapter {
        fn id(&self) -> &str {
            "external-disabled"
        }

        fn advertised_tools(&self) -> Vec<ToolDefinition> {
            vec![ToolDefinition {
                name: "mcp__demo__disabled".to_string(),
                description: "disabled".to_string(),
                params: vec![],
                returns: "dict".to_string(),
                examples: vec![],
                availability: crate::ToolAvailabilityConfig::hidden(),
                activation: crate::ToolActivation::Always,
                availability_override: None,
                input_schema_override: None,
                output_schema_override: None,
                discovery: Default::default(),
                execution_mode: crate::ToolExecutionMode::Parallel,
            }]
        }

        async fn execute(
            &self,
            tool: &str,
            args: &serde_json::Value,
            _context: Option<ToolExecutionContext>,
            _progress: Option<&ProgressSender>,
        ) -> ToolResult {
            ToolResult::ok(json!({
                "tool": tool,
                "args": args
            }))
        }
    }

    #[test]
    fn dynamic_registry_preserves_initial_availability_state() {
        let registry = DynamicToolProvider::from_tool_provider(Arc::new(MixedEnabledTool))
            .expect("dynamic registry");
        let defs = registry.definitions();
        assert_eq!(defs.len(), 2);
        assert!(defs.iter().any(|def| def.name == "enabled_tool"));
        assert!(defs.iter().any(|def| def.name == "disabled_tool"));
        let snapshot = registry.export_state();
        assert_eq!(
            snapshot.tools["enabled_tool"]
                .definition
                .effective_availability(&crate::ExecutionMode::standard()),
            crate::ToolAvailability::Callable
        );
        assert_eq!(
            snapshot.tools["disabled_tool"]
                .definition
                .effective_availability(&crate::ExecutionMode::standard()),
            crate::ToolAvailability::Hidden
        );
    }

    #[test]
    fn dynamic_registry_enables_all_tools_by_default_when_requested() {
        let registry =
            DynamicToolProvider::from_tool_provider(Arc::new(MockTool)).expect("dynamic registry");
        let defs = registry.definitions();
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].name, "mock_tool");
    }

    #[test]
    fn apply_state_rejects_tools_not_advertised_by_adapter() {
        let registry =
            DynamicToolProvider::from_tool_provider(Arc::new(MockTool)).expect("dynamic registry");
        let mut snapshot = registry.export_state();
        snapshot.tools.insert(
            "missing".to_string(),
            DynamicToolSpec {
                definition: ToolDefinition {
                    name: "missing".to_string(),
                    description: "missing".to_string(),
                    params: vec![],
                    returns: "str".to_string(),
                    examples: vec![],
                    availability: crate::ToolAvailabilityConfig::callable(),
                    activation: crate::ToolActivation::Always,
                    availability_override: None,
                    input_schema_override: None,
                    output_schema_override: None,
                    discovery: Default::default(),
                    execution_mode: crate::ToolExecutionMode::Parallel,
                },
                adapter_id: "inprocess".to_string(),
            },
        );
        assert!(matches!(
            registry.apply_state(snapshot),
            Err(ReconfigureError::Validation(_))
        ));
    }

    #[tokio::test]
    async fn upsert_adapter_registers_and_executes_external_tools() {
        let registry =
            DynamicToolProvider::from_tool_provider(Arc::new(MockTool)).expect("dynamic registry");
        registry
            .upsert_adapter(Arc::new(ExternalMockAdapter))
            .expect("adapter registered");

        let defs = registry.definitions();
        assert!(defs.iter().any(|def| def.name == "mcp__demo__search"));

        let result = registry
            .execute("mcp__demo__search", &json!({ "query": "hello" }))
            .await;
        assert!(result.success);
        assert_eq!(result.result["tool"], json!("mcp__demo__search"));
        assert_eq!(result.result["args"]["query"], json!("hello"));
    }

    #[test]
    fn upsert_adapter_preserves_hidden_advertised_tools() {
        let registry =
            DynamicToolProvider::from_tool_provider(Arc::new(MockTool)).expect("dynamic registry");
        registry
            .upsert_adapter(Arc::new(DisabledExternalMockAdapter))
            .expect("adapter registered");

        let defs = registry.definitions();
        assert!(defs.iter().any(|def| def.name == "mcp__demo__disabled"));
        let snapshot = registry.export_state();
        assert_eq!(
            snapshot.tools["mcp__demo__disabled"]
                .definition
                .effective_availability(&crate::ExecutionMode::standard()),
            crate::ToolAvailability::Hidden
        );
    }

    #[test]
    fn remove_adapter_removes_all_adapter_tools() {
        let registry =
            DynamicToolProvider::from_tool_provider(Arc::new(MockTool)).expect("dynamic registry");
        registry
            .upsert_adapter(Arc::new(ExternalMockAdapter))
            .expect("adapter registered");
        registry
            .remove_adapter("external")
            .expect("adapter removed");

        let defs = registry.definitions();
        assert!(!defs.iter().any(|def| def.name == "mcp__demo__search"));
    }
}
