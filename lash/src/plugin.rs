use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::agent::PromptSectionName;
use crate::llm::types::LlmResponse;
use crate::runtime::{AssembledTurn, PromptUsage};
use crate::{
    AgentCapabilities, AgentStateEnvelope, ContextFoldingConfig, ExecutionMode, MessageRole,
    ToolDefinition, ToolProvider, ToolResult, TurnInput,
};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

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
pub type AfterTurnHook =
    Arc<dyn Fn(TurnResultHookContext) -> PluginFuture<Vec<PluginDirective>> + Send + Sync>;
pub type CheckpointHook =
    Arc<dyn Fn(CheckpointHookContext) -> PluginFuture<Vec<PluginDirective>> + Send + Sync>;
pub type PromptContributor =
    Arc<dyn Fn(PromptHookContext) -> Result<Vec<PromptContribution>, PluginError> + Send + Sync>;
pub type TurnPromptContributor =
    Arc<dyn Fn(TurnHookContext) -> PluginFuture<Vec<TurnPromptContribution>> + Send + Sync>;
pub type ToolSurfaceContributor =
    Arc<dyn Fn(ToolSurfaceContext) -> Result<ToolSurfaceContribution, PluginError> + Send + Sync>;
pub type MessageMutator = Arc<
    dyn Fn(MessageMutatorContext, Vec<crate::Message>) -> PluginFuture<Vec<crate::Message>>
        + Send
        + Sync,
>;
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
}

pub type SessionSnapshot = AgentStateEnvelope;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SessionConfigOverrides {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_variant: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_context_tokens: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execution_mode: Option<ExecutionMode>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub capabilities: Option<AgentCapabilities>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_folding: Option<ContextFoldingConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionConfigSnapshot {
    pub provider_kind: crate::provider::ProviderKind,
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_variant: Option<String>,
    pub execution_mode: ExecutionMode,
    pub context_folding: ContextFoldingConfig,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_window: Option<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SessionStartPoint {
    Empty,
    CurrentSession,
    ExistingSession { session_id: String },
    Snapshot { snapshot: Box<SessionSnapshot> },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PluginMessage {
    pub role: MessageRole,
    pub content: String,
}

#[derive(Clone, Debug)]
pub struct PluginOwned<T> {
    pub plugin_id: String,
    pub value: T,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SessionCreateRequest {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_id: Option<String>,
    pub start: SessionStartPoint,
    #[serde(default)]
    pub config_overrides: SessionConfigOverrides,
    #[serde(default)]
    pub initial_messages: Vec<PluginMessage>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromptContribution {
    pub section: PromptSectionName,
    #[serde(default)]
    pub priority: i32,
    pub content: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TurnPromptContribution {
    #[serde(default)]
    pub priority: i32,
    pub content: String,
}

#[derive(Clone, Debug)]
pub struct ToolSurfaceContext {
    pub session_id: String,
    pub mode: ExecutionMode,
    pub tools: Vec<ToolDefinition>,
}

#[derive(Clone, Debug, Default)]
pub struct ToolSurfaceContribution {
    pub overrides: Vec<ToolSurfaceOverride>,
    pub guide_sections: Vec<String>,
    pub tool_list_notes: Vec<String>,
}

#[derive(Clone, Debug, Default)]
pub struct ToolSurfaceOverride {
    pub tool_name: String,
    pub inject_into_prompt: Option<bool>,
    pub discoverable: Option<bool>,
}

#[derive(Clone, Debug, Default)]
pub struct ResolvedToolSurface {
    pub tools: Vec<ToolDefinition>,
    pub guide_sections: Vec<String>,
    pub tool_list_notes: Vec<String>,
}

impl ResolvedToolSurface {
    pub fn from_tools(mode: ExecutionMode, mut tools: Vec<ToolDefinition>) -> Self {
        for tool in &mut tools {
            if tool.description_for(mode).is_empty() {
                tool.inject_into_prompt = false;
                tool.hidden = true;
            }
        }
        Self {
            tools,
            guide_sections: Vec::new(),
            tool_list_notes: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PluginSurfaceEvent {
    ModeIndicatorUpsert {
        key: String,
        label: String,
    },
    ModeIndicatorClear {
        key: String,
    },
    PanelUpsert {
        key: String,
        title: String,
        content: String,
    },
    PanelAppend {
        key: String,
        content: String,
    },
    PanelClear {
        key: String,
    },
    Custom {
        name: String,
        payload: serde_json::Value,
    },
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
    async fn start_turn(
        &self,
        session_id: &str,
        input: TurnInput,
    ) -> Result<AssembledTurn, PluginError>;
}

#[derive(Clone)]
pub struct PromptHookContext {
    pub session_id: String,
    pub host: Arc<dyn SessionManager>,
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
    pub previous: SessionConfigSnapshot,
    pub current: SessionConfigSnapshot,
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

#[derive(Clone)]
pub struct TurnResultHookContext {
    pub session_id: String,
    pub turn: AssembledTurn,
    pub host: Arc<dyn SessionManager>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CheckpointKind {
    AfterWork,
    BeforeCompletion,
}

#[derive(Clone)]
pub struct CheckpointHookContext {
    pub session_id: String,
    pub checkpoint: CheckpointKind,
    pub state: SessionSnapshot,
    pub host: Arc<dyn SessionManager>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MessageMutatorHook {
    BeforeTurn,
    AfterTokenCount,
    AfterTurn,
}

impl MessageMutatorHook {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::BeforeTurn => "before_turn",
            Self::AfterTokenCount => "after_token_count",
            Self::AfterTurn => "after_turn",
        }
    }
}

#[derive(Clone)]
pub struct MessageMutatorContext {
    pub hook: MessageMutatorHook,
    pub session_id: String,
    pub state: SessionSnapshot,
    pub host: Arc<dyn SessionManager>,
    pub turn: Option<AssembledTurn>,
    pub prompt_usage: Option<PromptUsage>,
    pub max_context_tokens: Option<usize>,
    pub context_folding: Option<ContextFoldingConfig>,
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

#[derive(Clone, Debug)]
pub struct PluginSessionContext {
    pub agent_id: String,
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
}

pub trait PluginFactory: Send + Sync {
    fn id(&self) -> &'static str;
    fn build(&self, ctx: &PluginSessionContext) -> Result<Arc<dyn SessionPlugin>, PluginError>;
}
mod runtime_impl;

pub use runtime_impl::{
    ExternalInvokeError, PluginHost, PluginRegistrar, PluginSession, RuntimeServices,
};

#[cfg(feature = "sqlite-store")]
#[path = "plugin_builtin.rs"]
mod builtin;

#[cfg(feature = "sqlite-store")]
pub use builtin::{
    BuiltinHistoryPluginFactory, BuiltinMemoryPluginFactory, BuiltinPlanModePluginFactory,
    BuiltinPlanTrackerPluginFactory, BuiltinPromptContextPluginFactory,
    BuiltinToolSurfacePluginFactory, PromptContextPluginConfig, builtin_dynamic_capability_defs,
};

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use serde_json::json;

    use super::*;
    use crate::{
        AgentStateEnvelope, ContextFoldingConfig, DynamicCapabilityDef, ExecutionMode,
        ToolDefinition, ToolParam, TurnInput,
    };

    struct MockToolProvider;

    #[async_trait::async_trait]
    impl ToolProvider for MockToolProvider {
        fn definitions(&self) -> Vec<ToolDefinition> {
            vec![ToolDefinition {
                name: "mock_tool".to_string(),
                description: vec![],
                params: vec![ToolParam::typed("value", "str")],
                returns: "str".to_string(),
                examples: vec![],
                hidden: false,
                inject_into_prompt: false,
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
            })
        }

        async fn close_session(&self, _session_id: &str) -> Result<(), PluginError> {
            Ok(())
        }

        async fn start_turn(
            &self,
            session_id: &str,
            _input: TurnInput,
        ) -> Result<AssembledTurn, PluginError> {
            Ok(AssembledTurn {
                state: AgentStateEnvelope {
                    agent_id: session_id.to_string(),
                    execution_mode: ExecutionMode::Standard,
                    context_folding: ContextFoldingConfig::default(),
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
    }

    impl SessionPlugin for MockPlugin {
        fn id(&self) -> &'static str {
            "mock"
        }

        fn register(&self, reg: &mut PluginRegistrar) -> Result<(), PluginError> {
            reg.register_tool_provider(Arc::new(MockToolProvider))?;
            reg.register_capability(DynamicCapabilityDef {
                id: "mock_cap".to_string(),
                name: "Mock".to_string(),
                description: "mock".to_string(),
                prompt_section: Some("## Mock".to_string()),
                helper_bindings: BTreeSet::from(["mock_tool".to_string()]),
                tool_names: BTreeSet::from(["mock_tool".to_string()]),
                enabled_by_default: true,
            })?;
            reg.register_prompt_contributor(Arc::new(|_ctx| {
                Ok(vec![PromptContribution {
                    section: PromptSectionName::PluginExtensions,
                    priority: 0,
                    content: "## Plugin Prompt".to_string(),
                }])
            }));
            reg.register_turn_prompt_contributor(Arc::new(|_ctx| {
                Box::pin(async move {
                    Ok(vec![TurnPromptContribution {
                        priority: 0,
                        content: "dynamic note".to_string(),
                    }])
                })
            }));
            let agent_id = self.agent_id.clone();
            reg.register_external_op(
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

    #[test]
    fn session_collects_tools_capabilities_and_prompts() {
        let host = PluginHost::new(vec![Arc::new(MockPluginFactory)]);
        let session = host.build_session("root", None).expect("session");
        assert_eq!(session.tool_providers().len(), 1);
        assert!(session.capability_defs().contains_key("mock_cap"));
        let contributions = session
            .collect_prompt_contributions(PromptHookContext {
                session_id: "root".to_string(),
                host: Arc::new(MockSessionManager),
            })
            .expect("prompt contributions");
        assert_eq!(
            contributions,
            vec![PromptContribution {
                section: PromptSectionName::PluginExtensions,
                priority: 0,
                content: "## Plugin Prompt".to_string(),
            }]
        );
    }

    #[tokio::test]
    async fn session_collects_turn_prompt_contributions() {
        let host = PluginHost::new(vec![Arc::new(MockPluginFactory)]);
        let session = host.build_session("root", None).expect("session");
        let contributions = session
            .collect_turn_prompt_contributions(TurnHookContext {
                session_id: "root".to_string(),
                state: AgentStateEnvelope::default(),
                host: Arc::new(MockSessionManager),
            })
            .await
            .expect("turn prompt contributions");
        assert_eq!(
            contributions,
            vec![TurnPromptContribution {
                priority: 0,
                content: "dynamic note".to_string(),
            }]
        );
    }

    #[tokio::test]
    async fn external_invoke_defaults_to_current_session_when_requested() {
        let host = PluginHost::new(vec![Arc::new(MockPluginFactory)]);
        let session = host.build_session("root", None).expect("session");
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
        let _session = host.build_session("root", None).expect("session");

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
        let root = host.build_session("root", None).expect("root");
        let child = root.fork_for_agent("child").expect("child");

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
        let _session = host.build_session("root", None).expect("session");
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
        let session = host.build_session("root", None).expect("session");
        let snapshot = session.snapshot().expect("snapshot");
        assert!(snapshot.plugins.contains_key("mock"));
        let restored = host
            .build_session("child", Some(&snapshot))
            .expect("restored");
        let restored_snapshot = restored.snapshot().expect("snapshot");
        assert!(restored_snapshot.plugins.contains_key("mock"));
    }

    #[test]
    fn runtime_services_tools_only_builds_empty_plugin_session() {
        let tools: Arc<dyn ToolProvider> = Arc::new(MockToolProvider);
        let services = RuntimeServices::tools_only(tools, "root").expect("services");
        assert_eq!(services.plugins.agent_id(), "root");
        assert!(services.plugins.tool_providers().is_empty());
    }

    struct MutatorPluginFactory {
        plugin_id: &'static str,
        hook: MessageMutatorHook,
    }

    impl PluginFactory for MutatorPluginFactory {
        fn id(&self) -> &'static str {
            self.plugin_id
        }

        fn build(
            &self,
            _ctx: &PluginSessionContext,
        ) -> Result<Arc<dyn SessionPlugin>, PluginError> {
            Ok(Arc::new(MutatorPlugin {
                plugin_id: self.plugin_id,
                hook: self.hook,
            }))
        }
    }

    struct MutatorPlugin {
        plugin_id: &'static str,
        hook: MessageMutatorHook,
    }

    impl SessionPlugin for MutatorPlugin {
        fn id(&self) -> &'static str {
            self.plugin_id
        }

        fn register(&self, reg: &mut PluginRegistrar) -> Result<(), PluginError> {
            reg.register_message_mutator(
                self.hook,
                Arc::new(|_ctx, messages| Box::pin(async move { Ok(messages) })),
            )
        }
    }

    #[test]
    fn duplicate_message_mutator_hooks_are_rejected() {
        let host = PluginHost::new(vec![
            Arc::new(MutatorPluginFactory {
                plugin_id: "mutator-a",
                hook: MessageMutatorHook::AfterTokenCount,
            }),
            Arc::new(MutatorPluginFactory {
                plugin_id: "mutator-b",
                hook: MessageMutatorHook::AfterTokenCount,
            }),
        ]);
        let err = match host.build_session("root", None) {
            Ok(_) => panic!("duplicate mutator"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("duplicate message mutator"));
        assert!(err.to_string().contains("mutator-a"));
        assert!(err.to_string().contains("mutator-b"));
    }

    #[test]
    fn different_message_mutator_hooks_can_coexist() {
        let host = PluginHost::new(vec![
            Arc::new(MutatorPluginFactory {
                plugin_id: "mutator-before",
                hook: MessageMutatorHook::BeforeTurn,
            }),
            Arc::new(MutatorPluginFactory {
                plugin_id: "mutator-after",
                hook: MessageMutatorHook::AfterTurn,
            }),
        ]);
        host.build_session("root", None).expect("session");
    }
}
