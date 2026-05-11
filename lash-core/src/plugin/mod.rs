use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::llm::types::LlmResponse;
use crate::monitor::{MonitorSnapshot, MonitorSpec, MonitorUpdateBatch};
use crate::runtime::{AssembledTurn, PersistedSessionState};
use crate::{
    ExecutionMode, MessageRole, ModeTurnOptions, SessionPolicy, ToolAvailability, ToolDefinition,
    ToolProvider, ToolResult, TurnInput,
};
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use tokio::sync::mpsc;

pub use lash_sansio::{
    CheckpointKind, PluginMessage, PluginSurfaceEvent, PromptContribution, ToolSurfaceContribution,
    ToolSurfaceOverride,
};

pub type PluginFuture<T> = Pin<Box<dyn Future<Output = Result<T, PluginError>> + Send>>;
pub type PluginRuntimeEventHook = Arc<dyn Fn(PluginRuntimeEvent) -> PluginFuture<()> + Send + Sync>;
pub type PluginSessionTask = PluginFuture<()>;
pub type SessionConfigMutator = Arc<
    dyn Fn(SessionConfigChangedContext, SessionPolicy) -> PluginFuture<SessionPolicy> + Send + Sync,
>;
pub type PluginActionInvokeFuture = Pin<Box<dyn Future<Output = ToolResult> + Send>>;
pub type PluginActionHandler =
    Arc<dyn Fn(PluginActionContext, serde_json::Value) -> PluginActionInvokeFuture + Send + Sync>;
pub type PluginActionFuture<T> =
    Pin<Box<dyn Future<Output = Result<T, PluginActionFailure>> + Send>>;
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
pub type ToolDiscoveryContributor = Arc<
    dyn Fn(ToolDiscoveryContext) -> Result<ToolDiscoveryContribution, PluginError> + Send + Sync,
>;
pub type AssistantStreamHook =
    Arc<dyn Fn(AssistantStreamHookContext) -> PluginFuture<AssistantStreamTransform> + Send + Sync>;
pub type AssistantResponseHook = Arc<
    dyn Fn(AssistantResponseHookContext) -> PluginFuture<AssistantResponseTransform> + Send + Sync,
>;
mod mode;
pub use mode::{
    ModeBeforeLlmCallContext, ModeExtras, ModeLlmCallAction, ModeNativeToolsPlugin,
    ModeProtocolDriverPlugin, ModeRuntimeContext, ModeSessionContext, ModeSessionPlugin,
    StandardCreateExtras,
};

mod history;
pub use history::{
    HistoryError, HistoryRewriteMetadata, HistoryRewriter, HistoryState, RewriteContext,
    RewriteTrigger, SessionReadView, TurnContextTransform, TurnTransformContext,
};

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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_session_id: Option<String>,
    pub policy: SessionPolicy,
}

pub struct SessionTurnHandle {
    pub turn_id: String,
    pub session_id: String,
    pub policy: SessionPolicy,
    pub events: mpsc::Receiver<crate::SessionEvent>,
}

pub type SessionSnapshot = PersistedSessionState;

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

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SessionRelation {
    #[default]
    Root,
    Child {
        parent_session_id: String,
    },
    Handoff {
        parent_session_id: String,
        reason: String,
        #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
        metadata: serde_json::Map<String, serde_json::Value>,
    },
}

impl SessionRelation {
    pub fn parent_session_id(&self) -> Option<&str> {
        match self {
            Self::Root => None,
            Self::Child { parent_session_id }
            | Self::Handoff {
                parent_session_id, ..
            } => Some(parent_session_id),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SessionCreateRequest {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    #[serde(default)]
    pub relation: SessionRelation,
    pub start: SessionStartPoint,
    #[serde(default)]
    pub policy: Option<SessionPolicy>,
    #[serde(default)]
    pub plugin_mode: SessionPluginMode,
    #[serde(default)]
    pub initial_nodes: Vec<SessionAppendNode>,
    /// Optional seed message dispatched as the new session's first turn
    /// input. The runtime stashes it during `create_session`; any host
    /// that drives turns on the new session can claim it via
    /// `RuntimeSessionHost::take_first_turn_input`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_turn_input: Option<PluginMessage>,
    #[serde(default)]
    pub tool_access: SessionToolAccess,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subagent: Option<SubagentSessionAuthority>,
    #[serde(skip)]
    pub context_surface: SessionContextSurface,
    /// Per-execution-mode "extras" that configure mode-specific
    /// behavior at session-creation time. The base request stays
    /// mode-agnostic; each `ExecutionMode` defines its own struct.
    #[serde(default)]
    pub mode_extras: ModeExtras,
    /// Label for the token-cost ledger. When this session's turns
    /// complete, their token usage is accumulated under this label on
    /// the parent session's `token_ledger`. Examples: `"subagent"`,
    /// `"compaction"`. Defaults to `"child"` if unset.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage_source: Option<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SessionToolAccess {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ToolDefinition>,
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub hidden_tools: BTreeSet<String>,
}

impl SessionToolAccess {
    pub fn hides(&self, name: &str) -> bool {
        self.hidden_tools.contains(name)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SubagentSessionAuthority {
    pub agent_name: String,
    pub parent_session_id: String,
    pub capability: String,
    pub depth: u8,
    pub max_depth: u8,
}

/// Per-execution-mode configuration carried on a `SessionCreateRequest`.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SessionAppendNode {
    Message {
        message: PluginMessage,
    },
    Event {
        event: crate::SessionEventRecord,
    },
    Plugin {
        plugin_type: String,
        #[serde(default)]
        body: serde_json::Value,
    },
}

impl SessionAppendNode {
    pub fn message(message: PluginMessage) -> Self {
        Self::Message { message }
    }

    pub fn plugin(plugin_type: impl Into<String>, body: serde_json::Value) -> Self {
        Self::Plugin {
            plugin_type: plugin_type.into(),
            body,
        }
    }

    pub fn event(event: crate::SessionEventRecord) -> Self {
        Self::Event { event }
    }
}

#[derive(Clone, Debug)]
pub struct ToolSurfaceContext {
    pub session_id: String,
    pub mode: ExecutionMode,
    pub tools: Vec<ToolDefinition>,
    pub tool_access: SessionToolAccess,
    pub subagent: Option<SubagentSessionAuthority>,
}

#[derive(Clone, Debug)]
pub struct ToolDiscoveryContext {
    pub session_id: String,
    pub mode: ExecutionMode,
    pub catalog: Vec<serde_json::Value>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ToolDiscoveryContribution {
    pub tools: Vec<ToolDiscoveryToolContribution>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ToolDiscoveryToolContribution {
    pub tool_name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub namespace: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub aliases: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct PluginAbort {
    pub code: String,
    pub message: String,
}

#[derive(Clone, Debug, Default)]
pub struct TurnPreparation {
    pub messages: crate::MessageSequence,
    pub events: Vec<crate::SessionEvent>,
    pub abort: Option<PluginAbort>,
}

#[derive(Clone)]
pub struct PrepareTurnRequest {
    pub session_id: String,
    pub state: SessionReadView,
    pub messages: crate::MessageSequence,
    pub host: Arc<dyn TurnHookHost>,
    pub turn_context: crate::TurnContext,
}

#[derive(Clone, Debug, Default)]
pub struct CheckpointApplication {
    pub messages: Vec<PluginMessage>,
    pub events: Vec<crate::SessionEvent>,
    pub abort: Option<PluginAbort>,
}

#[derive(Clone, Debug)]
pub struct TurnFinalization {
    pub turn: AssembledTurn,
    pub events: Vec<crate::SessionEvent>,
}

pub(crate) async fn emit_plugin_surface_events(
    event_tx: &mpsc::Sender<crate::SessionEvent>,
    plugin_id: &str,
    events: Vec<PluginSurfaceEvent>,
) {
    for event in plugin_surface_session_events(plugin_id, events) {
        crate::session_model::send_event(event_tx, event).await;
    }
}

pub(crate) fn plugin_surface_session_events(
    plugin_id: &str,
    events: Vec<PluginSurfaceEvent>,
) -> Vec<crate::SessionEvent> {
    events
        .into_iter()
        .map(|event| crate::SessionEvent::PluginEvent {
            plugin_id: plugin_id.to_string(),
            event,
        })
        .collect()
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
    HandoffSession {
        session_id: String,
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
    EmitTrace {
        name: String,
        #[serde(default)]
        payload: serde_json::Value,
        #[serde(default)]
        context: Box<lash_trace::TraceContext>,
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
                control: None,
            }),
            _ => None,
        }
    }

    pub fn emit_events(events: Vec<PluginSurfaceEvent>) -> Self {
        Self::EmitEvents { events }
    }

    pub fn emit_trace(name: impl Into<String>, payload: serde_json::Value) -> Self {
        Self::EmitTrace {
            name: name.into(),
            payload,
            context: Box::new(lash_trace::TraceContext::default()),
        }
    }
}

#[async_trait::async_trait]
pub trait SessionSnapshotHost: Send + Sync {
    async fn snapshot_current(&self) -> Result<SessionSnapshot, PluginError>;
    async fn snapshot_session(&self, session_id: &str) -> Result<SessionSnapshot, PluginError>;
}

#[async_trait::async_trait]
pub trait ToolCatalogHost: Send + Sync {
    async fn tool_catalog(&self, session_id: &str) -> Result<Vec<serde_json::Value>, PluginError>;
}

#[async_trait::async_trait]
pub trait ToolStateHost: Send + Sync {
    async fn tool_state(&self, _session_id: &str) -> Result<crate::ToolState, PluginError> {
        Err(PluginError::Session(
            "tool state is unavailable in this session".to_string(),
        ))
    }
    async fn apply_tool_state(
        &self,
        _session_id: &str,
        _snapshot: crate::ToolState,
    ) -> Result<u64, PluginError> {
        Err(PluginError::Session(
            "tool state mutation is unavailable in this session".to_string(),
        ))
    }
    async fn set_tools_availability(
        &self,
        session_id: &str,
        tool_names: &[String],
        availability: Option<crate::ToolAvailability>,
    ) -> Result<u64, PluginError> {
        let mut snapshot = self.tool_state(session_id).await?;
        for name in tool_names {
            snapshot
                .set_availability(name, availability)
                .map_err(|err| PluginError::Session(err.to_string()))?;
        }
        self.apply_tool_state(session_id, snapshot).await
    }
    async fn set_tool_availability(
        &self,
        session_id: &str,
        tool_name: &str,
        availability: Option<ToolAvailability>,
    ) -> Result<u64, PluginError> {
        let mut snapshot = self.tool_state(session_id).await?;
        snapshot
            .set_availability(tool_name, availability)
            .map_err(|err| PluginError::Session(err.to_string()))?;
        self.apply_tool_state(session_id, snapshot).await
    }
}

#[async_trait::async_trait]
pub trait SessionLifecycleHost: Send + Sync {
    async fn create_session(
        &self,
        request: SessionCreateRequest,
    ) -> Result<SessionHandle, PluginError>;
    /// Pop the seed message that was queued for `session_id` via
    /// `SessionCreateRequest::first_turn_input`. Returns `None` if no
    /// seed was queued, or after a previous caller has already taken
    /// it. Hosts call this when starting the inaugural turn on a
    /// freshly created session.
    async fn take_first_turn_input(
        &self,
        _session_id: &str,
    ) -> Result<Option<PluginMessage>, PluginError> {
        Ok(None)
    }
    async fn close_session(&self, session_id: &str) -> Result<(), PluginError>;
}

#[async_trait::async_trait]
pub trait TurnHost: Send + Sync {
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

#[async_trait::async_trait]
pub trait TaskHost: Send + Sync {
    /// Push a user-visible message into the target session's turn-input
    /// injection bridge so it surfaces at the next iteration boundary of
    /// the current turn (or at the start of the next turn if the target
    /// is idle). Used by monitor and other wake-up flows where a note
    /// should land at the next available step rather than waiting for a
    /// brand-new task.
    async fn inject_turn_input(
        &self,
        _session_id: &str,
        _input: crate::InjectedTurnInput,
    ) -> Result<(), PluginError> {
        Err(PluginError::Session(
            "turn input injection is unavailable in this session".to_string(),
        ))
    }
    async fn spawn_hidden_task(
        &self,
        _session_id: &str,
        _label: &str,
        _task: PluginSessionTask,
    ) -> Result<(), PluginError> {
        Err(PluginError::Session(
            "session tasks are unavailable in this session".to_string(),
        ))
    }
    async fn await_hidden_tasks(&self, _session_id: &str) -> Result<(), PluginError> {
        Ok(())
    }
    async fn spawn_managed_task(
        &self,
        _session_id: &str,
        _spec: crate::ManagedTaskSpec,
        _task: PluginSessionTask,
    ) -> Result<(), PluginError> {
        Err(PluginError::Session(
            "managed session tasks are unavailable in this session".to_string(),
        ))
    }
    async fn cancel_managed_task(
        &self,
        _session_id: &str,
        _task_id: &str,
    ) -> Result<(), PluginError> {
        Err(PluginError::Session(
            "managed session tasks are unavailable in this session".to_string(),
        ))
    }
    async fn register_background_task(
        &self,
        _session_id: &str,
        _spec: crate::ManagedTaskSpec,
        _cancel: Option<crate::ManagedTaskCancel>,
    ) -> Result<(), PluginError> {
        Err(PluginError::Session(
            "background task registry is unavailable in this session".to_string(),
        ))
    }
    async fn unregister_background_task(&self, _session_id: &str, _task_id: &str) {}
    async fn complete_background_task(
        &self,
        _session_id: &str,
        _task_id: &str,
        _run_state: crate::ManagedRunState,
    ) {
    }
    /// Transition a still-live background task between the non-terminal
    /// `Running` and `Idle` run states. Used by subagent hosts to
    /// reflect whether the subagent is actively working or waiting for
    /// a follow-up task.
    async fn transition_background_task_live_state(
        &self,
        _session_id: &str,
        _task_id: &str,
        _run_state: crate::ManagedRunState,
    ) {
    }
    async fn list_background_tasks(
        &self,
        _session_id: &str,
    ) -> Result<Vec<crate::ManagedTaskStatus>, PluginError> {
        Err(PluginError::Session(
            "background task registry is unavailable in this session".to_string(),
        ))
    }
    /// Dispatch a kind-aware cancel for any registered background task.
    /// Monitor tasks terminate their process trees; subagent tasks close
    /// the agent subtree; other managed tasks are aborted.
    async fn cancel_background_task(
        &self,
        _session_id: &str,
        _task_id: &str,
    ) -> Result<crate::ManagedTaskStatus, PluginError> {
        Err(PluginError::Session(
            "background task registry is unavailable in this session".to_string(),
        ))
    }
}

#[async_trait::async_trait]
pub trait MonitorHost: Send + Sync {
    async fn monitor_snapshot(&self, _session_id: &str) -> Result<MonitorSnapshot, PluginError> {
        Err(PluginError::Session(
            "monitors are unavailable in this session".to_string(),
        ))
    }
    async fn take_monitor_updates(
        &self,
        _session_id: &str,
    ) -> Result<MonitorUpdateBatch, PluginError> {
        Err(PluginError::Session(
            "monitors are unavailable in this session".to_string(),
        ))
    }
    async fn start_monitor(
        &self,
        _session_id: &str,
        _spec: MonitorSpec,
    ) -> Result<MonitorSnapshot, PluginError> {
        Err(PluginError::Session(
            "monitors are unavailable in this session".to_string(),
        ))
    }
    async fn stop_monitor(
        &self,
        _session_id: &str,
        _monitor_id: &str,
    ) -> Result<MonitorSnapshot, PluginError> {
        Err(PluginError::Session(
            "monitors are unavailable in this session".to_string(),
        ))
    }
}

#[async_trait::async_trait]
pub trait SessionGraphHost: Send + Sync {
    async fn append_session_nodes(
        &self,
        _session_id: &str,
        _request: AppendSessionNodesRequest,
    ) -> Result<AppendSessionNodesResult, PluginError> {
        Err(PluginError::Session(
            "session graph mutation is unavailable in this session".to_string(),
        ))
    }
}

#[async_trait::async_trait]
pub trait DirectCompletionHost: Send + Sync {
    /// Make a single LLM call without creating a full session. Used by
    /// plugins for structured extraction, summarization, observation,
    /// and other one-shot calls that don't need tools, turn loops, or
    /// session state. The `usage_source` label tags the resulting
    /// token cost in the parent session's ledger.
    async fn direct_completion(
        &self,
        _request: crate::DirectRequest,
        _usage_source: &str,
    ) -> Result<DirectCompletion, PluginError> {
        Err(PluginError::Session(
            "direct completions are unavailable in this session".to_string(),
        ))
    }

    async fn direct_llm_completion(
        &self,
        _request: crate::LlmRequest,
        _usage_source: &str,
    ) -> Result<DirectLlmCompletion, PluginError> {
        Err(PluginError::Session(
            "direct LLM completions are unavailable in this session".to_string(),
        ))
    }
}

#[async_trait::async_trait]
pub trait TraceHost: Send + Sync {
    async fn emit_trace_event(
        &self,
        _context: lash_trace::TraceContext,
        _event: lash_trace::TraceEvent,
    ) -> Result<(), PluginError> {
        Ok(())
    }
}

pub trait PromptHookHost:
    SessionSnapshotHost + ToolCatalogHost + TaskHost + DirectCompletionHost
{
}
impl<T> PromptHookHost for T where
    T: SessionSnapshotHost + ToolCatalogHost + TaskHost + DirectCompletionHost + ?Sized
{
}

pub trait TurnHookHost:
    SessionSnapshotHost + ToolStateHost + SessionLifecycleHost + TraceHost
{
}
impl<T> TurnHookHost for T where
    T: SessionSnapshotHost + ToolStateHost + SessionLifecycleHost + TraceHost + ?Sized
{
}

pub trait ToolHookHost:
    SessionSnapshotHost
    + ToolCatalogHost
    + ToolStateHost
    + SessionLifecycleHost
    + TurnHost
    + TaskHost
    + MonitorHost
    + SessionGraphHost
    + DirectCompletionHost
    + TraceHost
    + TurnResultHookHost
    + CheckpointHookHost
{
}
impl<T> ToolHookHost for T where
    T: SessionSnapshotHost
        + ToolCatalogHost
        + ToolStateHost
        + SessionLifecycleHost
        + TurnHost
        + TaskHost
        + MonitorHost
        + SessionGraphHost
        + DirectCompletionHost
        + TraceHost
        + ?Sized
{
}

pub trait TurnResultHookHost: SessionLifecycleHost + TraceHost {}
impl<T> TurnResultHookHost for T where T: SessionLifecycleHost + TraceHost + ?Sized {}

pub trait CheckpointHookHost: SessionLifecycleHost + TraceHost {}
impl<T> CheckpointHookHost for T where T: SessionLifecycleHost + TraceHost + ?Sized {}

pub trait HistoryHost:
    SessionSnapshotHost
    + SessionLifecycleHost
    + TurnHost
    + TaskHost
    + SessionGraphHost
    + DirectCompletionHost
{
}
impl<T> HistoryHost for T where
    T: SessionSnapshotHost
        + SessionLifecycleHost
        + TurnHost
        + TaskHost
        + SessionGraphHost
        + DirectCompletionHost
        + ?Sized
{
}

pub trait PluginActionHost:
    SessionSnapshotHost
    + ToolCatalogHost
    + ToolStateHost
    + SessionLifecycleHost
    + TurnHost
    + TaskHost
    + MonitorHost
    + SessionGraphHost
    + DirectCompletionHost
    + TraceHost
{
}
impl<T> PluginActionHost for T where
    T: SessionSnapshotHost
        + ToolCatalogHost
        + ToolStateHost
        + SessionLifecycleHost
        + TurnHost
        + TaskHost
        + MonitorHost
        + SessionGraphHost
        + DirectCompletionHost
        + TraceHost
        + ?Sized
{
}

pub trait RuntimeSessionHost:
    PluginActionHost + ToolHookHost + HistoryHost + TurnHookHost + PromptHookHost
{
}
impl<T> RuntimeSessionHost for T where
    T: PluginActionHost + ToolHookHost + HistoryHost + TurnHookHost + PromptHookHost + ?Sized
{
}

/// Result of a single-shot LLM call via
/// [`DirectCompletionHost::direct_completion`].
#[derive(Clone, Debug)]
pub struct DirectCompletion {
    pub text: String,
    pub usage: crate::TokenUsage,
}

#[derive(Clone, Debug)]
pub struct DirectLlmCompletion {
    pub response: crate::LlmResponse,
    pub usage: crate::TokenUsage,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AppendSessionNodesRequest {
    pub nodes: Vec<SessionAppendNode>,
    #[serde(default)]
    pub requires_ancestor_node_id: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum AppendSessionNodesResult {
    Appended {
        node_ids: Vec<String>,
        leaf_node_id: String,
    },
    StaleBranch {
        current_leaf_node_id: Option<String>,
    },
}

#[derive(Clone)]
pub struct PromptHookContext {
    pub session_id: String,
    pub host: Arc<dyn PromptHookHost>,
    pub state: SessionReadView,
    pub mode_turn_options: ModeTurnOptions,
    pub turn_context: crate::TurnContext,
}

#[derive(Clone)]
pub struct TurnHookContext {
    pub session_id: String,
    pub state: SessionReadView,
    pub host: Arc<dyn TurnHookHost>,
    pub turn_context: crate::TurnContext,
}

#[derive(Clone)]
pub struct SessionConfigChangedContext {
    pub session_id: String,
    pub previous: SessionPolicy,
    pub current: SessionPolicy,
    pub host: Arc<dyn TaskHost>,
}

#[derive(Clone)]
pub struct SessionStateChangedContext {
    pub session_id: String,
    pub state: SessionReadView,
    pub host: Arc<dyn HistoryHost>,
}

#[derive(Clone)]
pub enum PluginRuntimeEvent {
    TurnCommitted(Arc<AssembledTurn>),
    TurnPersisted(SessionStateChangedContext),
    SessionRestored(SessionReadView),
    SessionConfigChanged(Box<SessionConfigChangedContext>),
}

#[derive(Clone, Debug)]
pub struct TurnResultSummary {
    pub outcome: crate::TurnOutcome,
    pub assistant_output: crate::runtime::AssistantOutput,
    pub execution: crate::runtime::ExecutionSummary,
    pub token_usage: crate::TokenUsage,
    pub tool_calls: Arc<Vec<crate::ToolCallRecord>>,
    pub errors: Arc<Vec<crate::runtime::TurnIssue>>,
}

impl TurnResultSummary {
    pub fn from_assembled(turn: &AssembledTurn) -> Self {
        Self {
            outcome: turn.outcome.clone(),
            assistant_output: turn.assistant_output.clone(),
            execution: turn.execution.clone(),
            token_usage: turn.token_usage.clone(),
            tool_calls: Arc::new(turn.tool_calls.clone()),
            errors: Arc::new(turn.errors.clone()),
        }
    }
}

#[derive(Clone)]
pub struct ToolCallHookContext {
    pub session_id: String,
    pub tool_name: String,
    pub args: serde_json::Value,
    pub turn_context: crate::TurnContext,
    pub(crate) host: Arc<dyn ToolHookHost>,
}

impl ToolCallHookContext {
    pub fn new(
        session_id: String,
        tool_name: String,
        args: serde_json::Value,
        turn_context: crate::TurnContext,
        host: Arc<dyn ToolHookHost>,
    ) -> Self {
        Self {
            session_id,
            tool_name,
            args,
            turn_context,
            host,
        }
    }

    pub async fn session_snapshot(&self) -> Result<SessionSnapshot, PluginError> {
        self.host.snapshot_session(&self.session_id).await
    }

    pub async fn set_tools_availability(
        &self,
        names: &[String],
        availability: Option<crate::ToolAvailability>,
    ) -> Result<u64, PluginError> {
        self.host
            .set_tools_availability(&self.session_id, names, availability)
            .await
    }
}

#[derive(Clone)]
pub struct ToolResultHookContext {
    pub session_id: String,
    pub tool_name: String,
    pub args: serde_json::Value,
    pub result: ToolResult,
    pub duration_ms: u64,
    pub turn_context: crate::TurnContext,
    pub(crate) host: Arc<dyn ToolHookHost>,
}

impl ToolResultHookContext {
    pub fn new(
        session_id: String,
        tool_name: String,
        args: serde_json::Value,
        result: ToolResult,
        duration_ms: u64,
        turn_context: crate::TurnContext,
        host: Arc<dyn ToolHookHost>,
    ) -> Self {
        Self {
            session_id,
            tool_name,
            args,
            result,
            duration_ms,
            turn_context,
            host,
        }
    }

    pub async fn session_snapshot(&self) -> Result<SessionSnapshot, PluginError> {
        self.host.snapshot_session(&self.session_id).await
    }

    pub async fn set_tools_availability(
        &self,
        names: &[String],
        availability: Option<crate::ToolAvailability>,
    ) -> Result<u64, PluginError> {
        self.host
            .set_tools_availability(&self.session_id, names, availability)
            .await
    }
}

#[derive(Clone)]
pub struct ToolResultProjectionContext {
    pub session_id: String,
    pub tool_name: String,
    pub args: serde_json::Value,
    pub result: ToolResult,
    pub duration_ms: u64,
}

#[derive(Clone)]
pub struct TurnResultHookContext {
    pub session_id: String,
    pub turn: Arc<TurnResultSummary>,
    pub host: Arc<dyn TurnResultHookHost>,
}

#[derive(Clone)]
pub struct CheckpointHookContext {
    pub session_id: String,
    pub checkpoint: CheckpointKind,
    pub state: SessionReadView,
    pub host: Arc<dyn CheckpointHookHost>,
}

#[derive(Clone)]
pub struct AssistantStreamHookContext {
    pub session_id: String,
    pub chunk: String,
}

#[derive(Clone, Debug, Default)]
pub struct AssistantStreamTransform {
    pub chunk: String,
    pub reasoning_deltas: Vec<String>,
    pub events: Vec<PluginSurfaceEvent>,
    /// When `true`, the runtime cancels the in-flight LLM call the
    /// moment this hook returns and finalizes the turn using whatever
    /// text has been streamed so far. Any plugin may set this — the
    /// first to raise it wins. Used by mode plugins to enforce
    /// one-block-per-turn contracts (e.g. the RLM stream mask aborts
    /// as soon as the first lashlang fence closes).
    pub abort_stream: bool,
}

#[derive(Clone)]
pub struct AssistantResponseHookContext {
    pub session_id: String,
    pub response: LlmResponse,
}

#[derive(Clone, Debug)]
pub struct AssistantResponseTransform {
    pub response: LlmResponse,
    pub events: Vec<PluginSurfaceEvent>,
}

mod snapshot;
pub(crate) use snapshot::{InMemorySnapshotReader, InMemorySnapshotWriter};
pub use snapshot::{
    PluginSessionSnapshot, PluginSnapshotArtifact, PluginSnapshotEntry, PluginSnapshotMeta,
    SnapshotReader, SnapshotWriter,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionParam {
    Required,
    Optional,
    Forbidden,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PluginActionKind {
    Query,
    Command,
    Task,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PluginActionDef {
    pub name: String,
    pub description: String,
    pub kind: PluginActionKind,
    pub session_param: SessionParam,
    #[serde(default)]
    pub input_schema: serde_json::Value,
    #[serde(default)]
    pub output_schema: serde_json::Value,
}

pub trait PluginAction: Send + Sync + 'static {
    const NAME: &'static str;
    const DESCRIPTION: &'static str;
    const KIND: PluginActionKind;
    const SESSION_PARAM: SessionParam;
    type Args: Serialize + DeserializeOwned + JsonSchema + Send + 'static;
    type Output: Serialize + DeserializeOwned + JsonSchema + Send + 'static;
}

#[derive(Clone, Debug, thiserror::Error)]
#[error("{message}")]
pub struct PluginActionFailure {
    message: String,
}

impl PluginActionFailure {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl From<String> for PluginActionFailure {
    fn from(value: String) -> Self {
        Self::new(value)
    }
}

impl From<&str> for PluginActionFailure {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

impl From<PluginError> for PluginActionFailure {
    fn from(value: PluginError) -> Self {
        Self::new(value.to_string())
    }
}

pub fn plugin_action_def<Op: PluginAction>() -> PluginActionDef {
    PluginActionDef {
        name: Op::NAME.to_string(),
        description: Op::DESCRIPTION.to_string(),
        kind: Op::KIND,
        session_param: Op::SESSION_PARAM,
        input_schema: serde_json::to_value(schemars::schema_for!(Op::Args))
            .unwrap_or_else(|_| serde_json::json!({})),
        output_schema: serde_json::to_value(schemars::schema_for!(Op::Output))
            .unwrap_or_else(|_| serde_json::json!({})),
    }
}

#[derive(Clone)]
pub struct PluginActionContext {
    pub session_id: Option<String>,
    pub host: Arc<dyn PluginActionHost>,
}

#[derive(Clone)]
pub(crate) struct RegisteredPluginAction {
    pub(crate) def: PluginActionDef,
    pub(crate) handler: PluginActionHandler,
}

mod registry;
pub use registry::{
    PluginFactory, PluginSessionContext, PluginSpec, PluginSpecBuilder, PluginSpecFactory,
    SessionPlugin, SessionReadyContext, StaticPluginFactory,
};

mod monitor;
pub use monitor::{
    AckWakeArgs, MonitorAckWakeOp, MonitorEmptyArgs, MonitorRegisterSpecsOp, MonitorStartOp,
    MonitorStatusOp, MonitorStopOp, MonitorTakeUpdatesOp, OwnedMonitorSpec, RegisterSpecsArgs,
    StartMonitorArgs, StopMonitorArgs,
};
mod registrar;
mod runtime_impl;
mod services;
mod session_obj;
mod tool_result_projection_builtin;

pub use registrar::{
    HistoryRegistrations, ModeRegistrations, MonitorRegistrations, OutputRegistrations,
    PluginActionRegistrations, PluginRegistrar, PromptRegistrations, SessionRegistrations,
    SurfaceRegistrations, ToolCallRegistrations, ToolRegistrations, ToolResultRegistrations,
    TurnRegistrations,
};
pub(crate) use registrar::{RegisteredExclusiveHook, RegisteredHook};
pub use runtime_impl::{PluginHost, SessionAuthorityContext};
pub(crate) use services::NoopSessionManager;
pub use services::{PersistentRuntimeServices, PluginActionInvokeError, RuntimeServices};
pub use session_obj::PluginSession;
pub use tool_result_projection_builtin::{
    DEFAULT_TOOL_OUTPUT_BUDGET_LIMIT_BYTES, DEFAULT_TOOL_OUTPUT_BUDGET_MAX_LINES,
    ToolOutputBudgetConfig, ToolOutputBudgetMode, ToolOutputBudgetPluginFactory,
    observation_projection_metadata, project_observation_text, truncate_observation_text,
};

pub(crate) fn builtin_plugin_factories() -> Vec<Arc<dyn PluginFactory>> {
    // Mode plugins (`lash-mode-standard`, `lash-mode-rlm`) must be
    // registered by the embedder before calling `PluginHost::build_session`.
    // lash's own test suite uses an in-tree fake (`testing::test_mode_factories()`)
    // to avoid a dev-dep cycle through the mode crates.
    let factories: Vec<Arc<dyn PluginFactory>> = vec![Arc::new(monitor::MonitorPluginFactory)];
    #[cfg(not(test))]
    return factories;

    #[cfg(test)]
    {
        factories
            .into_iter()
            .chain(crate::testing::test_mode_factories())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::{ExecutionMode, SessionStateEnvelope, ToolDefinition};

    struct MockToolProvider;

    #[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
    struct TypedEchoArgs {
        value: String,
    }

    #[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
    struct TypedEchoOutput {
        value: String,
        session_id: Option<String>,
    }

    struct TypedEchoOp;

    impl PluginAction for TypedEchoOp {
        const NAME: &'static str = "mock.typed_echo";
        const DESCRIPTION: &'static str = "typed echo";
        const KIND: PluginActionKind = PluginActionKind::Query;
        const SESSION_PARAM: SessionParam = SessionParam::Optional;
        type Args = TypedEchoArgs;
        type Output = TypedEchoOutput;
    }

    #[async_trait::async_trait]
    impl ToolProvider for MockToolProvider {
        fn definitions(&self) -> Vec<ToolDefinition> {
            vec![
                ToolDefinition::raw(
                    "mock_tool",
                    "",
                    json!({
                        "type": "object",
                        "properties": { "value": { "type": "string" } },
                        "required": ["value"],
                        "additionalProperties": false
                    }),
                    json!({ "type": "string" }),
                )
                .with_availability(crate::ToolAvailabilityConfig::callable()),
            ]
        }

        async fn execute(&self, call: crate::ToolCall<'_>) -> ToolResult {
            ToolResult::ok(call.args.clone())
        }
    }

    struct MockPluginFactory;

    impl PluginFactory for MockPluginFactory {
        fn id(&self) -> &'static str {
            "mock"
        }

        fn build(&self, ctx: &PluginSessionContext) -> Result<Arc<dyn SessionPlugin>, PluginError> {
            Ok(Arc::new(MockPlugin {
                session_id: ctx.session_id.clone(),
            }))
        }
    }

    struct MockPlugin {
        session_id: String,
    }

    use crate::testing::MockSessionManager;

    impl SessionPlugin for MockPlugin {
        fn id(&self) -> &'static str {
            "mock"
        }

        fn register(&self, reg: &mut PluginRegistrar) -> Result<(), PluginError> {
            reg.tools().provider(Arc::new(MockToolProvider))?;
            reg.prompt().contribute(Arc::new(|_ctx| {
                Box::pin(async move {
                    Ok(vec![
                        PromptContribution::guidance("Plugin Prompt", "Structured plugin prompt"),
                        PromptContribution::guidance("Dynamic Note", "dynamic note")
                            .with_priority(1),
                    ])
                })
            }));
            let session_id = self.session_id.clone();
            reg.actions().op(
                PluginActionDef {
                    name: "mock.echo".to_string(),
                    description: "echo".to_string(),
                    kind: PluginActionKind::Query,
                    session_param: SessionParam::Optional,
                    input_schema: json!({}),
                    output_schema: json!({}),
                },
                Arc::new(move |ctx, args| {
                    let session_id = session_id.clone();
                    Box::pin(async move {
                        ToolResult::ok(json!({
                            "session_id": ctx.session_id,
                            "plugin_session_id": session_id,
                            "args": args,
                        }))
                    })
                }),
            )?;
            reg.actions()
                .typed::<TypedEchoOp, _, _>(move |ctx, args| async move {
                    Ok(TypedEchoOutput {
                        value: args.value,
                        session_id: ctx.session_id,
                    })
                })?;
            Ok(())
        }

        fn snapshot(
            &self,
            _writer: &mut dyn SnapshotWriter,
        ) -> Result<PluginSnapshotMeta, PluginError> {
            Ok(PluginSnapshotMeta {
                plugin_id: self.id().to_string(),
                plugin_version: self.version().to_string(),
                revision: self.snapshot_revision(),
                state: Some(json!({"session_id": self.session_id})),
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
                host: Arc::new(MockSessionManager::default()),
                state: SessionReadView::from_exported_state(&SessionStateEnvelope::default()),
                mode_turn_options: ModeTurnOptions::default(),
                turn_context: crate::TurnContext::default(),
            })
            .await
            .expect("prompt contributions");
        assert_eq!(
            contributions,
            vec![
                PromptContribution::guidance("Plugin Prompt", "Structured plugin prompt"),
                PromptContribution::guidance("Dynamic Note", "dynamic note").with_priority(1),
            ]
        );
    }

    #[tokio::test]
    async fn external_invoke_defaults_to_current_session_when_requested() {
        let host = PluginHost::new(vec![Arc::new(MockPluginFactory)]);
        let session = host.build_standard_session("root", None).expect("session");
        let result = session
            .invoke_plugin_action(
                "mock.echo",
                json!({"ok":true}),
                None,
                true,
                Arc::new(MockSessionManager::default()),
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
    async fn plugin_action_generates_schema_and_invokes_typed_output() {
        let host = PluginHost::new(vec![Arc::new(MockPluginFactory)]);
        let session = host.build_standard_session("root", None).expect("session");

        let def = session
            .plugin_actions()
            .into_iter()
            .find(|def| def.name == TypedEchoOp::NAME)
            .expect("typed op definition");
        assert_eq!(def.kind, PluginActionKind::Query);
        assert_eq!(def.session_param, SessionParam::Optional);
        let value_type = def
            .input_schema
            .pointer("/schema/properties/value/type")
            .or_else(|| def.input_schema.pointer("/properties/value/type"))
            .and_then(serde_json::Value::as_str);
        assert_eq!(value_type, Some("string"));

        let output = session
            .call_plugin_action::<TypedEchoOp>(
                TypedEchoArgs {
                    value: "hello".to_string(),
                },
                None,
                true,
                Arc::new(MockSessionManager::default()),
            )
            .await
            .expect("typed invoke");
        assert_eq!(output.value, "hello");
        assert_eq!(output.session_id.as_deref(), Some("root"));
    }

    #[test]
    fn plugin_action_rejects_duplicate_names() {
        struct DuplicatePlugin;

        impl SessionPlugin for DuplicatePlugin {
            fn id(&self) -> &'static str {
                "duplicate"
            }

            fn register(&self, reg: &mut PluginRegistrar) -> Result<(), PluginError> {
                reg.actions()
                    .typed::<TypedEchoOp, _, _>(move |ctx, args| async move {
                        Ok(TypedEchoOutput {
                            value: args.value,
                            session_id: ctx.session_id,
                        })
                    })?;
                reg.actions()
                    .typed::<TypedEchoOp, _, _>(move |ctx, args| async move {
                        Ok(TypedEchoOutput {
                            value: args.value,
                            session_id: ctx.session_id,
                        })
                    })
            }
        }

        struct DuplicateFactory;
        impl PluginFactory for DuplicateFactory {
            fn id(&self) -> &'static str {
                "duplicate"
            }

            fn build(
                &self,
                _ctx: &PluginSessionContext,
            ) -> Result<Arc<dyn SessionPlugin>, PluginError> {
                Ok(Arc::new(DuplicatePlugin))
            }
        }

        let err = match PluginHost::new(vec![Arc::new(DuplicateFactory)])
            .build_standard_session("root", None)
        {
            Ok(_) => panic!("duplicate typed plugin action should fail"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("duplicate plugin action name"));
    }

    #[tokio::test]
    async fn typed_external_invoke_errors_on_failed_or_invalid_output() {
        struct BadOp;
        impl PluginAction for BadOp {
            const NAME: &'static str = "mock.echo";
            const DESCRIPTION: &'static str = "bad typed projection over raw op";
            const KIND: PluginActionKind = PluginActionKind::Query;
            const SESSION_PARAM: SessionParam = SessionParam::Optional;
            type Args = TypedEchoArgs;
            type Output = TypedEchoOutput;
        }

        let host = PluginHost::new(vec![Arc::new(MockPluginFactory)]);
        let session = host.build_standard_session("root", None).expect("session");
        let err = session
            .call_plugin_action::<BadOp>(
                TypedEchoArgs {
                    value: "hello".to_string(),
                },
                None,
                true,
                Arc::new(MockSessionManager::default()),
            )
            .await
            .expect_err("raw output shape should not match typed output");
        assert!(err.to_string().contains("invalid mock.echo output"));
    }

    #[tokio::test]
    async fn plugin_host_can_invoke_plugin_action_for_registered_session() {
        let host = PluginHost::new(vec![Arc::new(MockPluginFactory)]);
        let _session = host.build_standard_session("root", None).expect("session");

        let result = host
            .invoke_plugin_action_for_session(
                "root",
                "mock.echo",
                json!({"ok":true}),
                Arc::new(MockSessionManager::default()),
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
                .get("plugin_session_id")
                .and_then(|v| v.as_str()),
            Some("root")
        );
    }

    #[tokio::test]
    async fn plugin_host_can_invoke_plugin_action_for_forked_session() {
        let host = PluginHost::new(vec![Arc::new(MockPluginFactory)]);
        let root = host.build_standard_session("root", None).expect("root");
        let child = root
            .fork_for_session(
                "child",
                ExecutionMode::standard(),
                Some(crate::StandardContextApproach::default()),
            )
            .expect("child");

        let result = host
            .invoke_plugin_action_for_session(
                "child",
                "mock.echo",
                json!({"ok":true}),
                Arc::new(MockSessionManager::default()),
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
                .get("plugin_session_id")
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
            Err(PluginActionInvokeError::UnknownSession(id)) => assert_eq!(id, "root"),
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
        assert_eq!(services.plugins.session_id(), "root");
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
            }))
        }
    }

    struct ProjectorPlugin {
        plugin_id: &'static str,
    }

    impl SessionPlugin for ProjectorPlugin {
        fn id(&self) -> &'static str {
            self.plugin_id
        }

        fn register(&self, reg: &mut PluginRegistrar) -> Result<(), PluginError> {
            reg.tool_results()
                .projector(Arc::new(|ctx| Box::pin(async move { Ok(ctx.result) })))
        }
    }

    #[test]
    fn duplicate_tool_result_projectors_are_rejected() {
        let host = PluginHost::new(vec![
            Arc::new(ProjectorPluginFactory {
                plugin_id: "projector-a",
            }),
            Arc::new(ProjectorPluginFactory {
                plugin_id: "projector-b",
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
}
