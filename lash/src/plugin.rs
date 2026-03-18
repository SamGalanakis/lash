use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::llm::types::LlmResponse;
use crate::runtime::AssembledTurn;
use crate::{
    AgentStateEnvelope, ExecutionMode, MessageRole, SessionPolicy, ToolDefinition, ToolProvider,
    ToolResult, TurnInput,
};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

pub use lash_sansio::{CheckpointKind, PluginMessage, PluginSurfaceEvent, PromptContribution};

pub type PluginFuture<T> = Pin<Box<dyn Future<Output = Result<T, PluginError>> + Send>>;
pub type TurnCommittedHook = Arc<dyn Fn(AssembledTurn) -> PluginFuture<()> + Send + Sync>;
pub type SessionRestoredHook = Arc<dyn Fn(AgentStateEnvelope) -> PluginFuture<()> + Send + Sync>;
pub type SessionConfigChangedHook =
    Arc<dyn Fn(SessionConfigChangedContext) -> PluginFuture<()> + Send + Sync>;
pub type SessionConfigMutator = Arc<
    dyn Fn(SessionConfigChangedContext, AgentStateEnvelope) -> PluginFuture<AgentStateEnvelope>
        + Send
        + Sync,
>;
pub type ExternalInvokeFuture = Pin<Box<dyn Future<Output = ToolResult> + Send>>;
pub type ExternalInvokeHandler =
    Arc<dyn Fn(ExternalInvokeContext, serde_json::Value) -> ExternalInvokeFuture + Send + Sync>;
pub type BeforeTurnHook =
    Arc<dyn Fn(TurnHookContext) -> PluginFuture<Vec<PluginDirective>> + Send + Sync>;
pub type BeforeToolCallHook =
    Arc<dyn Fn(ToolCallHookContext) -> PluginFuture<Vec<PluginDirective>> + Send + Sync>;
pub type AfterToolCallHook =
    Arc<dyn Fn(ToolResultHookContext) -> PluginFuture<Vec<PluginDirective>> + Send + Sync>;
pub type ToolResultProjector =
    Arc<dyn Fn(ToolResultProjectionContext) -> PluginFuture<ToolResult> + Send + Sync>;
pub type AfterTurnHook =
    Arc<dyn Fn(TurnResultHookContext) -> PluginFuture<Vec<PluginDirective>> + Send + Sync>;
pub type CheckpointHook =
    Arc<dyn Fn(CheckpointHookContext) -> PluginFuture<Vec<PluginDirective>> + Send + Sync>;
pub type PromptContributor =
    Arc<dyn Fn(PromptHookContext) -> PluginFuture<Vec<PromptContribution>> + Send + Sync>;
pub type ToolSurfaceContributor =
    Arc<dyn Fn(ToolSurfaceContext) -> Result<ToolSurfaceContribution, PluginError> + Send + Sync>;
pub type AssistantStreamHook =
    Arc<dyn Fn(AssistantStreamHookContext) -> PluginFuture<AssistantStreamTransform> + Send + Sync>;
pub type AssistantResponseHook = Arc<
    dyn Fn(AssistantResponseHookContext) -> PluginFuture<AssistantResponseTransform> + Send + Sync,
>;

#[derive(Debug, thiserror::Error, Clone)]
pub enum PluginError {
    #[error("plugin registration error: {0}")]
    Registration(String),
    #[error("plugin snapshot error: {0}")]
    Snapshot(String),
    #[error("plugin invoke error: {0}")]
    Invoke(String),
    #[error("plugin session error: {0}")]
    Session(String),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SessionHandle {
    pub session_id: String,
    pub policy: SessionPolicy,
}

pub struct SessionTurnHandle {
    pub turn_id: String,
    pub session_id: String,
    pub policy: SessionPolicy,
    pub events: mpsc::Receiver<crate::AgentEvent>,
}

pub type SessionSnapshot = AgentStateEnvelope;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SessionStartPoint {
    Empty,
    CurrentSession,
    ExistingSession { session_id: String },
    Snapshot { snapshot: Box<SessionSnapshot> },
}

#[derive(Clone, Debug)]
pub struct PluginOwned<T> {
    pub plugin_id: String,
    pub value: T,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionPluginMode {
    Fresh,
    #[default]
    InheritCurrent,
}

#[derive(Clone)]
pub struct SessionContextSurface {
    pub include_base_tools: bool,
    pub tool_providers: Vec<Arc<dyn ToolProvider>>,
    pub prompt_contributions: Vec<PromptContribution>,
}

impl Default for SessionContextSurface {
    fn default() -> Self {
        Self {
            include_base_tools: true,
            tool_providers: Vec::new(),
            prompt_contributions: Vec::new(),
        }
    }
}

impl std::fmt::Debug for SessionContextSurface {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SessionContextSurface")
            .field("include_base_tools", &self.include_base_tools)
            .field("tool_provider_count", &self.tool_providers.len())
            .field(
                "prompt_contribution_count",
                &self.prompt_contributions.len(),
            )
            .finish()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SessionCreateRequest {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_id: Option<String>,
    pub start: SessionStartPoint,
    #[serde(default)]
    pub policy: Option<SessionPolicy>,
    #[serde(default)]
    pub plugin_mode: SessionPluginMode,
    #[serde(default)]
    pub initial_messages: Vec<PluginMessage>,
    #[serde(skip)]
    pub context_surface: SessionContextSurface,
}

#[derive(Clone, Debug)]
pub struct ToolSurfaceContext {
    pub session_id: String,
    pub mode: ExecutionMode,
    pub tools: Vec<ToolDefinition>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ToolSurfaceContribution {
    pub overrides: Vec<ToolSurfaceOverride>,
    pub tool_list_notes: Vec<String>,
}

impl ToolSurfaceContribution {
    pub fn is_empty(&self) -> bool {
        self.overrides.is_empty() && self.tool_list_notes.is_empty()
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ToolSurfaceOverride {
    pub tool_name: String,
    pub enabled: Option<bool>,
    pub injected: Option<bool>,
}

#[derive(Clone, Debug, Default)]
pub struct ExecutionSurface {
    pub tools: Vec<ToolDefinition>,
    pub tool_list_notes: Vec<String>,
}

impl ExecutionSurface {
    pub fn from_tools(tools: Vec<ToolDefinition>) -> Self {
        Self {
            tools,
            tool_list_notes: Vec::new(),
        }
    }

    pub fn enabled_tools(&self) -> Vec<ToolDefinition> {
        self.tools
            .iter()
            .filter(|tool| tool.enabled)
            .cloned()
            .collect()
    }

    pub fn prompt_tools(&self) -> Vec<ToolDefinition> {
        self.tools
            .iter()
            .filter(|tool| tool.enabled && tool.injected)
            .cloned()
            .collect()
    }
}

#[derive(Clone, Debug)]
pub struct PluginAbort {
    pub code: String,
    pub message: String,
}

#[derive(Clone, Debug, Default)]
pub struct TurnPreparation {
    pub messages: Vec<crate::Message>,
    pub events: Vec<crate::AgentEvent>,
    pub abort: Option<PluginAbort>,
}

#[derive(Clone)]
pub struct PrepareTurnRequest {
    pub session_id: String,
    pub state: AgentStateEnvelope,
    pub messages: Vec<crate::Message>,
    pub host: Arc<dyn SessionManager>,
}

#[derive(Clone, Debug, Default)]
pub struct CheckpointApplication {
    pub messages: Vec<PluginMessage>,
    pub events: Vec<crate::AgentEvent>,
    pub abort: Option<PluginAbort>,
}

#[derive(Clone, Debug)]
pub struct TurnFinalization {
    pub turn: AssembledTurn,
    pub events: Vec<crate::AgentEvent>,
}

pub(crate) async fn emit_plugin_surface_events(
    event_tx: &mpsc::Sender<crate::AgentEvent>,
    plugin_id: &str,
    events: Vec<PluginSurfaceEvent>,
) {
    for event in events {
        crate::agent::send_event(
            event_tx,
            crate::AgentEvent::PluginEvent {
                plugin_id: plugin_id.to_string(),
                event,
            },
        )
        .await;
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PluginDirective {
    AbortTurn {
        code: String,
        message: String,
    },
    EnqueueMessages {
        messages: Vec<PluginMessage>,
    },
    CreateSession {
        request: Box<SessionCreateRequest>,
    },
    ReplaceToolArgs {
        args: serde_json::Value,
    },
    ShortCircuitTool {
        result: serde_json::Value,
        success: bool,
    },
    EmitEvents {
        events: Vec<PluginSurfaceEvent>,
    },
}

impl PluginDirective {
    pub fn short_circuit(result: ToolResult) -> Self {
        Self::ShortCircuitTool {
            result: result.result,
            success: result.success,
        }
    }

    pub fn into_tool_result(self) -> Option<ToolResult> {
        match self {
            Self::ShortCircuitTool { result, success } => Some(ToolResult {
                success,
                result,
                images: Vec::new(),
            }),
            _ => None,
        }
    }

    pub fn emit_events(events: Vec<PluginSurfaceEvent>) -> Self {
        Self::EmitEvents { events }
    }
}

#[async_trait::async_trait]
pub trait SessionManager: Send + Sync {
    async fn snapshot_current(&self) -> Result<SessionSnapshot, PluginError>;
    async fn snapshot_session(&self, session_id: &str) -> Result<SessionSnapshot, PluginError>;
    async fn tool_catalog(&self, session_id: &str) -> Result<Vec<serde_json::Value>, PluginError>;
    async fn create_session(
        &self,
        request: SessionCreateRequest,
    ) -> Result<SessionHandle, PluginError>;
    async fn close_session(&self, session_id: &str) -> Result<(), PluginError>;
    async fn start_turn_stream(
        &self,
        session_id: &str,
        input: TurnInput,
    ) -> Result<SessionTurnHandle, PluginError>;
    async fn await_turn(&self, turn_id: &str) -> Result<AssembledTurn, PluginError>;
    async fn cancel_turn(&self, turn_id: &str) -> Result<(), PluginError>;
    async fn start_turn(
        &self,
        session_id: &str,
        input: TurnInput,
    ) -> Result<AssembledTurn, PluginError> {
        let handle = self.start_turn_stream(session_id, input).await?;
        drop(handle.events);
        self.await_turn(&handle.turn_id).await
    }
}

#[derive(Clone)]
pub struct PromptHookContext {
    pub session_id: String,
    pub host: Arc<dyn SessionManager>,
    pub prompt: crate::PromptContext,
    pub state: SessionSnapshot,
}

#[derive(Clone)]
pub struct TurnHookContext {
    pub session_id: String,
    pub state: SessionSnapshot,
    pub host: Arc<dyn SessionManager>,
}

#[derive(Clone)]
pub struct SessionConfigChangedContext {
    pub session_id: String,
    pub previous: SessionPolicy,
    pub current: SessionPolicy,
    pub host: Arc<dyn SessionManager>,
}

#[derive(Clone)]
pub struct ToolCallHookContext {
    pub session_id: String,
    pub tool_name: String,
    pub args: serde_json::Value,
    pub host: Arc<dyn SessionManager>,
}

#[derive(Clone)]
pub struct ToolResultHookContext {
    pub session_id: String,
    pub tool_name: String,
    pub args: serde_json::Value,
    pub result: ToolResult,
    pub duration_ms: u64,
    pub host: Arc<dyn SessionManager>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolResultProjectionHook {
    BeforeModel,
    BeforeHistory,
}

impl ToolResultProjectionHook {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::BeforeModel => "before_model",
            Self::BeforeHistory => "before_history",
        }
    }
}

#[derive(Clone)]
pub struct ToolResultProjectionContext {
    pub hook: ToolResultProjectionHook,
    pub session_id: String,
    pub tool_name: String,
    pub args: serde_json::Value,
    pub result: ToolResult,
    pub duration_ms: u64,
    pub host: Arc<dyn SessionManager>,
}

#[derive(Clone)]
pub struct TurnResultHookContext {
    pub session_id: String,
    pub turn: AssembledTurn,
    pub host: Arc<dyn SessionManager>,
}

#[derive(Clone)]
pub struct CheckpointHookContext {
    pub session_id: String,
    pub checkpoint: CheckpointKind,
    pub state: SessionSnapshot,
    pub host: Arc<dyn SessionManager>,
}

#[derive(Clone)]
pub struct AssistantStreamHookContext {
    pub session_id: String,
    pub chunk: String,
    pub host: Arc<dyn SessionManager>,
}

#[derive(Clone, Debug, Default)]
pub struct AssistantStreamTransform {
    pub chunk: String,
    pub events: Vec<PluginSurfaceEvent>,
}

#[derive(Clone)]
pub struct AssistantResponseHookContext {
    pub session_id: String,
    pub response: LlmResponse,
    pub host: Arc<dyn SessionManager>,
}

#[derive(Clone, Debug)]
pub struct AssistantResponseTransform {
    pub response: LlmResponse,
    pub events: Vec<PluginSurfaceEvent>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PluginSessionSnapshot {
    #[serde(default)]
    pub plugins: BTreeMap<String, PluginSnapshotEntry>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PluginSnapshotEntry {
    pub meta: PluginSnapshotMeta,
    #[serde(default)]
    pub artifacts: Vec<PluginSnapshotArtifact>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PluginSnapshotMeta {
    pub plugin_id: String,
    pub plugin_version: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state: Option<serde_json::Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PluginSnapshotArtifact {
    pub name: String,
    pub data: Vec<u8>,
}

pub trait SnapshotWriter {
    fn write_blob(&mut self, name: String, data: Vec<u8>);
}

pub trait SnapshotReader {
    fn read_blob(&self, name: &str) -> Option<&[u8]>;
}

#[derive(Default)]
struct InMemorySnapshotWriter {
    artifacts: Vec<PluginSnapshotArtifact>,
}

impl InMemorySnapshotWriter {
    fn finish(self) -> Vec<PluginSnapshotArtifact> {
        self.artifacts
    }
}

impl SnapshotWriter for InMemorySnapshotWriter {
    fn write_blob(&mut self, name: String, data: Vec<u8>) {
        self.artifacts.push(PluginSnapshotArtifact { name, data });
    }
}

struct InMemorySnapshotReader<'a> {
    entry: &'a PluginSnapshotEntry,
}

impl SnapshotReader for InMemorySnapshotReader<'_> {
    fn read_blob(&self, name: &str) -> Option<&[u8]> {
        self.entry
            .artifacts
            .iter()
            .find(|artifact| artifact.name == name)
            .map(|artifact| artifact.data.as_slice())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionParam {
    Required,
    Optional,
    Forbidden,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExternalOpKind {
    Query,
    Command,
    Task,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExternalOpDef {
    pub name: String,
    pub description: String,
    pub kind: ExternalOpKind,
    pub session_param: SessionParam,
    #[serde(default)]
    pub input_schema: serde_json::Value,
    #[serde(default)]
    pub output_schema: serde_json::Value,
}

#[derive(Clone)]
pub struct ExternalInvokeContext {
    pub session_id: Option<String>,
    pub host: Arc<dyn SessionManager>,
}

#[derive(Clone)]
struct RegisteredExternalOp {
    def: ExternalOpDef,
    handler: ExternalInvokeHandler,
}

#[derive(Clone, Default)]
pub struct PluginSpec {
    pub tool_providers: Vec<Arc<dyn ToolProvider>>,
    pub prompt_contributors: Vec<PromptContributor>,
    pub tool_surface_contributors: Vec<ToolSurfaceContributor>,
    pub before_turn_hooks: Vec<BeforeTurnHook>,
    pub before_tool_call_hooks: Vec<BeforeToolCallHook>,
    pub after_tool_call_hooks: Vec<AfterToolCallHook>,
    pub after_turn_hooks: Vec<AfterTurnHook>,
    pub checkpoint_hooks: Vec<CheckpointHook>,
    pub assistant_stream_hooks: Vec<AssistantStreamHook>,
    pub assistant_response_hooks: Vec<AssistantResponseHook>,
    pub tool_result_projectors: BTreeMap<ToolResultProjectionHook, ToolResultProjector>,
    pub turn_committed_hooks: Vec<TurnCommittedHook>,
    pub session_restored_hooks: Vec<SessionRestoredHook>,
    pub session_config_mutators: Vec<SessionConfigMutator>,
    pub session_config_changed_hooks: Vec<SessionConfigChangedHook>,
    pub external_ops: Vec<(ExternalOpDef, ExternalInvokeHandler)>,
}

impl PluginSpec {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_tool_provider(mut self, provider: Arc<dyn ToolProvider>) -> Self {
        self.tool_providers.push(provider);
        self
    }

    pub fn with_prompt_contributor(mut self, contributor: PromptContributor) -> Self {
        self.prompt_contributors.push(contributor);
        self
    }

    pub fn with_tool_surface_contributor(mut self, contributor: ToolSurfaceContributor) -> Self {
        self.tool_surface_contributors.push(contributor);
        self
    }

    pub fn with_before_turn(mut self, hook: BeforeTurnHook) -> Self {
        self.before_turn_hooks.push(hook);
        self
    }

    pub fn with_before_tool_call(mut self, hook: BeforeToolCallHook) -> Self {
        self.before_tool_call_hooks.push(hook);
        self
    }

    pub fn with_after_tool_call(mut self, hook: AfterToolCallHook) -> Self {
        self.after_tool_call_hooks.push(hook);
        self
    }

    pub fn with_after_turn(mut self, hook: AfterTurnHook) -> Self {
        self.after_turn_hooks.push(hook);
        self
    }

    pub fn with_checkpoint(mut self, hook: CheckpointHook) -> Self {
        self.checkpoint_hooks.push(hook);
        self
    }

    pub fn with_assistant_stream(mut self, hook: AssistantStreamHook) -> Self {
        self.assistant_stream_hooks.push(hook);
        self
    }

    pub fn with_assistant_response(mut self, hook: AssistantResponseHook) -> Self {
        self.assistant_response_hooks.push(hook);
        self
    }

    pub fn with_tool_result_projector(
        mut self,
        hook: ToolResultProjectionHook,
        projector: ToolResultProjector,
    ) -> Self {
        self.tool_result_projectors.insert(hook, projector);
        self
    }

    pub fn with_turn_committed(mut self, hook: TurnCommittedHook) -> Self {
        self.turn_committed_hooks.push(hook);
        self
    }

    pub fn with_session_restored(mut self, hook: SessionRestoredHook) -> Self {
        self.session_restored_hooks.push(hook);
        self
    }

    pub fn with_session_config_mutator(mut self, hook: SessionConfigMutator) -> Self {
        self.session_config_mutators.push(hook);
        self
    }

    pub fn with_session_config_changed(mut self, hook: SessionConfigChangedHook) -> Self {
        self.session_config_changed_hooks.push(hook);
        self
    }

    pub fn with_external_op(mut self, def: ExternalOpDef, handler: ExternalInvokeHandler) -> Self {
        self.external_ops.push((def, handler));
        self
    }
}

#[derive(Clone, Debug)]
pub struct PluginSessionContext {
    pub agent_id: String,
    pub execution_mode: ExecutionMode,
}

#[derive(Clone)]
pub struct SessionReadyContext {
    pub agent_id: String,
    pub execution_mode: ExecutionMode,
    pub host: PluginHost,
}

pub trait SessionPlugin: Send + Sync {
    fn id(&self) -> &'static str;

    fn version(&self) -> &'static str {
        "1"
    }

    fn register(&self, reg: &mut PluginRegistrar) -> Result<(), PluginError>;

    fn snapshot(
        &self,
        _writer: &mut dyn SnapshotWriter,
    ) -> Result<PluginSnapshotMeta, PluginError> {
        Ok(PluginSnapshotMeta {
            plugin_id: self.id().to_string(),
            plugin_version: self.version().to_string(),
            state: None,
        })
    }

    fn restore(
        &self,
        _meta: &PluginSnapshotMeta,
        _reader: &dyn SnapshotReader,
    ) -> Result<(), PluginError> {
        Ok(())
    }

    fn session_ready(&self, _ctx: SessionReadyContext) -> Result<(), PluginError> {
        Ok(())
    }
}

pub trait PluginFactory: Send + Sync {
    fn id(&self) -> &'static str;
    fn build(&self, ctx: &PluginSessionContext) -> Result<Arc<dyn SessionPlugin>, PluginError>;
}

pub type PluginSpecBuilder =
    Arc<dyn Fn(&PluginSessionContext) -> Result<PluginSpec, PluginError> + Send + Sync>;

pub struct PluginSpecFactory {
    id: &'static str,
    builder: PluginSpecBuilder,
}

impl PluginSpecFactory {
    pub fn new(id: &'static str, builder: PluginSpecBuilder) -> Self {
        Self { id, builder }
    }
}

pub struct StaticPluginFactory {
    id: &'static str,
    spec: PluginSpec,
}

impl StaticPluginFactory {
    pub fn new(id: &'static str, spec: PluginSpec) -> Self {
        Self { id, spec }
    }
}

struct SpecPlugin {
    id: &'static str,
    spec: PluginSpec,
}

impl PluginFactory for PluginSpecFactory {
    fn id(&self) -> &'static str {
        self.id
    }

    fn build(&self, ctx: &PluginSessionContext) -> Result<Arc<dyn SessionPlugin>, PluginError> {
        Ok(Arc::new(SpecPlugin {
            id: self.id,
            spec: (self.builder)(ctx)?,
        }))
    }
}

impl PluginFactory for StaticPluginFactory {
    fn id(&self) -> &'static str {
        self.id
    }

    fn build(&self, _ctx: &PluginSessionContext) -> Result<Arc<dyn SessionPlugin>, PluginError> {
        Ok(Arc::new(SpecPlugin {
            id: self.id,
            spec: self.spec.clone(),
        }))
    }
}

impl SessionPlugin for SpecPlugin {
    fn id(&self) -> &'static str {
        self.id
    }

    fn register(&self, reg: &mut PluginRegistrar) -> Result<(), PluginError> {
        for provider in &self.spec.tool_providers {
            reg.tools().provider(Arc::clone(provider))?;
        }
        for contributor in &self.spec.prompt_contributors {
            reg.prompt().contribute(Arc::clone(contributor));
        }
        for contributor in &self.spec.tool_surface_contributors {
            reg.surface().contribute(Arc::clone(contributor));
        }
        for hook in &self.spec.before_turn_hooks {
            reg.turn().before(Arc::clone(hook));
        }
        for hook in &self.spec.before_tool_call_hooks {
            reg.tool_calls().before(Arc::clone(hook));
        }
        for hook in &self.spec.after_tool_call_hooks {
            reg.tool_calls().after(Arc::clone(hook));
        }
        for hook in &self.spec.after_turn_hooks {
            reg.turn().after(Arc::clone(hook));
        }
        for hook in &self.spec.checkpoint_hooks {
            reg.turn().checkpoint(Arc::clone(hook));
        }
        for hook in &self.spec.assistant_stream_hooks {
            reg.output().stream(Arc::clone(hook));
        }
        for hook in &self.spec.assistant_response_hooks {
            reg.output().response(Arc::clone(hook));
        }
        for (hook, projector) in &self.spec.tool_result_projectors {
            reg.tool_results().projector(*hook, Arc::clone(projector))?;
        }
        for hook in &self.spec.turn_committed_hooks {
            reg.turn().committed(Arc::clone(hook));
        }
        for hook in &self.spec.session_restored_hooks {
            reg.session().restored(Arc::clone(hook));
        }
        for hook in &self.spec.session_config_mutators {
            reg.session().config_mutator(Arc::clone(hook));
        }
        for hook in &self.spec.session_config_changed_hooks {
            reg.session().config_changed(Arc::clone(hook));
        }
        for (def, handler) in &self.spec.external_ops {
            reg.external().op(def.clone(), Arc::clone(handler))?;
        }
        Ok(())
    }
}
mod runtime_impl;
mod tool_result_projection_builtin;

pub use runtime_impl::{
    ExternalInvokeError, ExternalRegistrations, OutputRegistrations, PluginHost, PluginRegistrar,
    PluginSession, PromptRegistrations, RuntimeServices, SessionRegistrations,
    SurfaceRegistrations, ToolCallRegistrations, ToolRegistrations, ToolResultRegistrations,
    TurnRegistrations,
};
pub use tool_result_projection_builtin::{
    BuiltinToolResultProjectionPluginFactory, ToolResultProjectionMode,
    ToolResultProjectionPluginConfig,
};

#[cfg(feature = "sqlite-store")]
#[path = "plugin_builtin.rs"]
mod builtin;

#[cfg(feature = "sqlite-store")]
pub(crate) use builtin::history;

#[cfg(feature = "sqlite-store")]
pub use builtin::{
    BuiltinPlanModePluginFactory, BuiltinPlanTrackerPluginFactory,
    BuiltinPromptContextPluginFactory, PromptContextPluginConfig,
};

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::{AgentStateEnvelope, ExecutionMode, ToolDefinition, ToolParam, TurnInput};

    struct MockToolProvider;

    #[async_trait::async_trait]
    impl ToolProvider for MockToolProvider {
        fn definitions(&self) -> Vec<ToolDefinition> {
            vec![ToolDefinition {
                name: "mock_tool".to_string(),
                description: String::new(),
                params: vec![ToolParam::typed("value", "str")],
                returns: "str".to_string(),
                examples: vec![],
                enabled: true,
                injected: false,
            }]
        }

        async fn execute(&self, _name: &str, args: &serde_json::Value) -> ToolResult {
            ToolResult::ok(args.clone())
        }
    }

    struct MockPluginFactory;

    impl PluginFactory for MockPluginFactory {
        fn id(&self) -> &'static str {
            "mock"
        }

        fn build(&self, ctx: &PluginSessionContext) -> Result<Arc<dyn SessionPlugin>, PluginError> {
            Ok(Arc::new(MockPlugin {
                agent_id: ctx.agent_id.clone(),
            }))
        }
    }

    struct MockPlugin {
        agent_id: String,
    }

    struct MockSessionManager;

    #[async_trait::async_trait]
    impl SessionManager for MockSessionManager {
        async fn snapshot_current(&self) -> Result<SessionSnapshot, PluginError> {
            Ok(AgentStateEnvelope::default())
        }

        async fn snapshot_session(
            &self,
            _session_id: &str,
        ) -> Result<SessionSnapshot, PluginError> {
            Ok(AgentStateEnvelope::default())
        }

        async fn tool_catalog(
            &self,
            _session_id: &str,
        ) -> Result<Vec<serde_json::Value>, PluginError> {
            Ok(Vec::new())
        }

        async fn create_session(
            &self,
            request: SessionCreateRequest,
        ) -> Result<SessionHandle, PluginError> {
            Ok(SessionHandle {
                session_id: request.agent_id.unwrap_or_else(|| "child".to_string()),
                policy: SessionPolicy {
                    provider: crate::Provider::OpenAiGeneric {
                        api_key: String::new(),
                        base_url: crate::provider::OPENAI_GENERIC_DEFAULT_BASE_URL.to_string(),
                        options: crate::ProviderOptions::default(),
                    },
                    model: "mock-model".to_string(),
                    execution_mode: ExecutionMode::Standard,
                    context_strategy: crate::default_context_strategy(),
                    ..Default::default()
                },
            })
        }

        async fn close_session(&self, _session_id: &str) -> Result<(), PluginError> {
            Ok(())
        }

        async fn start_turn_stream(
            &self,
            session_id: &str,
            _input: TurnInput,
        ) -> Result<SessionTurnHandle, PluginError> {
            let (tx, rx) = mpsc::channel(1);
            let turn_id = format!("{session_id}-turn");
            tokio::spawn(async move {
                drop(tx);
            });
            Ok(SessionTurnHandle {
                turn_id,
                session_id: session_id.to_string(),
                policy: SessionPolicy {
                    provider: crate::Provider::OpenAiGeneric {
                        api_key: String::new(),
                        base_url: crate::provider::OPENAI_GENERIC_DEFAULT_BASE_URL.to_string(),
                        options: crate::ProviderOptions::default(),
                    },
                    model: "mock-model".to_string(),
                    execution_mode: ExecutionMode::Standard,
                    context_strategy: crate::default_context_strategy(),
                    ..Default::default()
                },
                events: rx,
            })
        }

        async fn await_turn(&self, session_turn_id: &str) -> Result<AssembledTurn, PluginError> {
            let session_id = session_turn_id.trim_end_matches("-turn");
            Ok(AssembledTurn {
                state: AgentStateEnvelope {
                    agent_id: session_id.to_string(),
                    policy: SessionPolicy {
                        execution_mode: ExecutionMode::Standard,
                        context_strategy: crate::default_context_strategy(),
                        ..Default::default()
                    },
                    ..Default::default()
                },
                status: crate::TurnStatus::Completed,
                assistant_output: crate::AssistantOutput {
                    safe_text: String::new(),
                    raw_text: String::new(),
                    state: crate::OutputState::Usable,
                },
                done_reason: crate::DoneReason::ModelStop,
                execution: crate::ExecutionSummary {
                    mode: ExecutionMode::Standard,
                    had_tool_calls: false,
                    had_code_execution: false,
                },
                token_usage: crate::TokenUsage::default(),
                tool_calls: Vec::new(),
                code_outputs: Vec::new(),
                errors: Vec::new(),
            })
        }

        async fn cancel_turn(&self, _turn_id: &str) -> Result<(), PluginError> {
            Ok(())
        }
    }

    impl SessionPlugin for MockPlugin {
        fn id(&self) -> &'static str {
            "mock"
        }

        fn register(&self, reg: &mut PluginRegistrar) -> Result<(), PluginError> {
            reg.tools().provider(Arc::new(MockToolProvider))?;
            reg.prompt().contribute(Arc::new(|_ctx| {
                Box::pin(async move {
                    Ok(vec![
                        PromptContribution {
                            section: PromptSectionName::Guidance,
                            priority: 0,
                            content: "## Plugin Prompt".to_string(),
                        },
                        PromptContribution {
                            section: PromptSectionName::Guidance,
                            priority: 1,
                            content: "dynamic note".to_string(),
                        },
                    ])
                })
            }));
            let agent_id = self.agent_id.clone();
            reg.external().op(
                ExternalOpDef {
                    name: "mock.echo".to_string(),
                    description: "echo".to_string(),
                    kind: ExternalOpKind::Query,
                    session_param: SessionParam::Optional,
                    input_schema: json!({}),
                    output_schema: json!({}),
                },
                Arc::new(move |ctx, args| {
                    let agent_id = agent_id.clone();
                    Box::pin(async move {
                        ToolResult::ok(json!({
                            "session_id": ctx.session_id,
                            "plugin_agent_id": agent_id,
                            "args": args,
                        }))
                    })
                }),
            )?;
            Ok(())
        }

        fn snapshot(
            &self,
            _writer: &mut dyn SnapshotWriter,
        ) -> Result<PluginSnapshotMeta, PluginError> {
            Ok(PluginSnapshotMeta {
                plugin_id: self.id().to_string(),
                plugin_version: self.version().to_string(),
                state: Some(json!({"agent_id": self.agent_id})),
            })
        }
    }

    #[tokio::test]
    async fn session_collects_tools_and_prompts() {
        let host = PluginHost::new(vec![Arc::new(MockPluginFactory)]);
        let session = host.build_standard_session("root", None).expect("session");
        assert_eq!(session.tools().definitions().len(), 1);
        let contributions = session
            .collect_prompt_contributions(PromptHookContext {
                session_id: "root".to_string(),
                host: Arc::new(MockSessionManager),
                prompt: crate::PromptContext::default(),
                state: AgentStateEnvelope::default(),
            })
            .await
            .expect("prompt contributions");
        assert_eq!(
            contributions,
            vec![
                PromptContribution {
                    section: PromptSectionName::Guidance,
                    priority: 0,
                    content: "## Plugin Prompt".to_string(),
                },
                PromptContribution {
                    section: PromptSectionName::Guidance,
                    priority: 1,
                    content: "dynamic note".to_string(),
                },
            ]
        );
    }

    #[tokio::test]
    async fn external_invoke_defaults_to_current_session_when_requested() {
        let host = PluginHost::new(vec![Arc::new(MockPluginFactory)]);
        let session = host.build_standard_session("root", None).expect("session");
        let result = session
            .invoke_external(
                "mock.echo",
                json!({"ok":true}),
                None,
                true,
                Arc::new(MockSessionManager),
            )
            .await
            .expect("invoke");
        assert!(result.success);
        assert_eq!(
            result.result.get("session_id").and_then(|v| v.as_str()),
            Some("root")
        );
    }

    #[tokio::test]
    async fn plugin_host_can_invoke_external_for_registered_session() {
        let host = PluginHost::new(vec![Arc::new(MockPluginFactory)]);
        let _session = host.build_standard_session("root", None).expect("session");

        let result = host
            .invoke_external_for_session(
                "root",
                "mock.echo",
                json!({"ok":true}),
                Arc::new(MockSessionManager),
            )
            .await
            .expect("invoke");
        assert!(result.success);
        assert_eq!(
            result.result.get("session_id").and_then(|v| v.as_str()),
            Some("root")
        );
        assert_eq!(
            result
                .result
                .get("plugin_agent_id")
                .and_then(|v| v.as_str()),
            Some("root")
        );
    }

    #[tokio::test]
    async fn plugin_host_can_invoke_external_for_forked_session() {
        let host = PluginHost::new(vec![Arc::new(MockPluginFactory)]);
        let root = host.build_standard_session("root", None).expect("root");
        let child = root
            .fork_for_agent("child", ExecutionMode::Standard)
            .expect("child");

        let result = host
            .invoke_external_for_session(
                "child",
                "mock.echo",
                json!({"ok":true}),
                Arc::new(MockSessionManager),
            )
            .await
            .expect("invoke");
        assert!(result.success);
        assert_eq!(
            result.result.get("session_id").and_then(|v| v.as_str()),
            Some("child")
        );
        assert_eq!(
            result
                .result
                .get("plugin_agent_id")
                .and_then(|v| v.as_str()),
            Some("child")
        );

        drop(child);
    }

    #[test]
    fn plugin_host_unregisters_sessions() {
        let host = PluginHost::new(vec![Arc::new(MockPluginFactory)]);
        let _session = host.build_standard_session("root", None).expect("session");
        assert!(host.session("root").is_ok());
        host.unregister_session("root").expect("unregister");
        match host.session("root") {
            Err(ExternalInvokeError::UnknownSession(id)) => assert_eq!(id, "root"),
            Ok(_) => panic!("expected missing session"),
            Err(other) => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn snapshot_round_trip_preserves_plugin_entries() {
        let host = PluginHost::new(vec![Arc::new(MockPluginFactory)]);
        let session = host.build_standard_session("root", None).expect("session");
        let snapshot = session.snapshot().expect("snapshot");
        assert!(snapshot.plugins.contains_key("mock"));
        let restored = host
            .build_standard_session("child", Some(&snapshot))
            .expect("restored");
        let restored_snapshot = restored.snapshot().expect("snapshot");
        assert!(restored_snapshot.plugins.contains_key("mock"));
    }

    #[test]
    fn runtime_services_are_backed_by_plugin_sessions() {
        let host = PluginHost::new(vec![Arc::new(StaticPluginFactory::new(
            "mock_tool",
            PluginSpec::new()
                .with_tool_provider(Arc::new(MockToolProvider) as Arc<dyn ToolProvider>),
        ))]);
        let services =
            RuntimeServices::new(host.build_standard_session("root", None).expect("session"));
        assert_eq!(services.plugins.agent_id(), "root");
        assert!(
            services
                .plugins
                .tools()
                .definitions()
                .iter()
                .any(|tool| tool.name == "mock_tool")
        );
    }

    struct ProjectorPluginFactory {
        plugin_id: &'static str,
        hook: ToolResultProjectionHook,
    }

    impl PluginFactory for ProjectorPluginFactory {
        fn id(&self) -> &'static str {
            self.plugin_id
        }

        fn build(
            &self,
            _ctx: &PluginSessionContext,
        ) -> Result<Arc<dyn SessionPlugin>, PluginError> {
            Ok(Arc::new(ProjectorPlugin {
                plugin_id: self.plugin_id,
                hook: self.hook,
            }))
        }
    }

    struct ProjectorPlugin {
        plugin_id: &'static str,
        hook: ToolResultProjectionHook,
    }

    impl SessionPlugin for ProjectorPlugin {
        fn id(&self) -> &'static str {
            self.plugin_id
        }

        fn register(&self, reg: &mut PluginRegistrar) -> Result<(), PluginError> {
            reg.tool_results().projector(
                self.hook,
                Arc::new(|ctx| Box::pin(async move { Ok(ctx.result) })),
            )
        }
    }

    #[test]
    fn duplicate_model_tool_result_projectors_are_rejected() {
        let host = PluginHost::new(vec![
            Arc::new(ProjectorPluginFactory {
                plugin_id: "projector-a",
                hook: ToolResultProjectionHook::BeforeModel,
            }),
            Arc::new(ProjectorPluginFactory {
                plugin_id: "projector-b",
                hook: ToolResultProjectionHook::BeforeModel,
            }),
        ]);
        let err = match host.build_standard_session("root", None) {
            Ok(_) => panic!("duplicate projector"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("duplicate tool result projector"));
        assert!(err.to_string().contains("projector-a"));
        assert!(err.to_string().contains("projector-b"));
    }

    #[test]
    fn duplicate_history_tool_result_projectors_are_rejected() {
        let host = PluginHost::new(vec![
            Arc::new(ProjectorPluginFactory {
                plugin_id: "projector-a",
                hook: ToolResultProjectionHook::BeforeHistory,
            }),
            Arc::new(ProjectorPluginFactory {
                plugin_id: "projector-b",
                hook: ToolResultProjectionHook::BeforeHistory,
            }),
        ]);
        let err = match host.build_standard_session("root", None) {
            Ok(_) => panic!("duplicate projector"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("duplicate tool result projector"));
        assert!(err.to_string().contains("projector-a"));
        assert!(err.to_string().contains("projector-b"));
    }

    #[test]
    fn different_tool_result_projector_hooks_can_coexist() {
        let host = PluginHost::new(vec![
            Arc::new(ProjectorPluginFactory {
                plugin_id: "projector-model",
                hook: ToolResultProjectionHook::BeforeModel,
            }),
            Arc::new(ProjectorPluginFactory {
                plugin_id: "projector-history",
                hook: ToolResultProjectionHook::BeforeHistory,
            }),
        ]);
        host.build_standard_session("root", None).expect("session");
    }
}
