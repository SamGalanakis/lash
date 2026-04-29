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
use serde::{Deserialize, Serialize};
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
pub type PromptRequestHook =
    Arc<dyn Fn(PromptRequestHookContext) -> PluginFuture<Vec<PluginSurfaceEvent>> + Send + Sync>;
pub type ToolSurfaceContributor =
    Arc<dyn Fn(ToolSurfaceContext) -> Result<ToolSurfaceContribution, PluginError> + Send + Sync>;
pub type AssistantStreamHook =
    Arc<dyn Fn(AssistantStreamHookContext) -> PluginFuture<AssistantStreamTransform> + Send + Sync>;
pub type AssistantResponseHook = Arc<
    dyn Fn(AssistantResponseHookContext) -> PluginFuture<AssistantResponseTransform> + Send + Sync,
>;
pub type CommandHandler =
    Arc<dyn Fn(CommandInvocation) -> PluginFuture<CommandOutcome> + Send + Sync>;

mod mode;
pub use mode::{
    ModeExtras, ModeNativeToolsPlugin, ModeProtocolDriverPlugin, ModeRuntimeContext,
    ModeSessionContext, ModeSessionPlugin, RlmCreateExtras, StandardCreateExtras,
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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SessionCreateRequest {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_session_id: Option<String>,
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
    /// `SessionManager::take_first_turn_input`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_turn_input: Option<PluginMessage>,
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
    pub host: Arc<dyn SessionManager>,
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

pub fn plugin_surface_event_renders_visible_output(event: &PluginSurfaceEvent) -> bool {
    match event {
        PluginSurfaceEvent::PanelUpsert { .. } => true,
        PluginSurfaceEvent::PanelAppend { content, .. } => !content.is_empty(),
        PluginSurfaceEvent::ModeIndicatorUpsert { .. }
        | PluginSurfaceEvent::ModeIndicatorClear { .. }
        | PluginSurfaceEvent::Status { .. }
        | PluginSurfaceEvent::PanelClear { .. }
        | PluginSurfaceEvent::Custom { .. } => false,
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
pub trait SessionManager: Send + Sync {
    async fn snapshot_current(&self) -> Result<SessionSnapshot, PluginError>;
    async fn snapshot_session(&self, session_id: &str) -> Result<SessionSnapshot, PluginError>;
    async fn tool_catalog(&self, session_id: &str) -> Result<Vec<serde_json::Value>, PluginError>;
    async fn dynamic_tool_state(
        &self,
        _session_id: &str,
    ) -> Result<crate::DynamicStateSnapshot, PluginError> {
        Err(PluginError::Session(
            "dynamic tool state is unavailable in this session".to_string(),
        ))
    }
    async fn apply_dynamic_tool_state(
        &self,
        _session_id: &str,
        _snapshot: crate::DynamicStateSnapshot,
    ) -> Result<u64, PluginError> {
        Err(PluginError::Session(
            "dynamic tool state mutation is unavailable in this session".to_string(),
        ))
    }
    async fn set_tools_availability(
        &self,
        session_id: &str,
        tool_names: &[String],
        availability: Option<crate::ToolAvailability>,
    ) -> Result<u64, PluginError> {
        let mut snapshot = self.dynamic_tool_state(session_id).await?;
        for name in tool_names {
            let Some(spec) = snapshot.tools.get_mut(name) else {
                return Err(PluginError::Session(format!(
                    "unknown dynamic tool `{name}`"
                )));
            };
            spec.definition.availability_override = availability;
        }
        self.apply_dynamic_tool_state(session_id, snapshot).await
    }
    async fn set_tool_availability(
        &self,
        session_id: &str,
        tool_name: &str,
        availability: Option<ToolAvailability>,
    ) -> Result<u64, PluginError> {
        let mut snapshot = self.dynamic_tool_state(session_id).await?;
        let Some(spec) = snapshot.tools.get_mut(tool_name) else {
            return Err(PluginError::Session(format!(
                "unknown dynamic tool `{tool_name}`"
            )));
        };
        spec.definition.availability_override = availability;
        self.apply_dynamic_tool_state(session_id, snapshot).await
    }
    async fn create_session(
        &self,
        request: SessionCreateRequest,
    ) -> Result<SessionHandle, PluginError>;
    async fn emit_trace_event(
        &self,
        _context: lash_trace::TraceContext,
        _event: lash_trace::TraceEvent,
    ) -> Result<(), PluginError> {
        Ok(())
    }
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
    async fn start_turn_stream(
        &self,
        session_id: &str,
        input: TurnInput,
    ) -> Result<SessionTurnHandle, PluginError>;
    async fn await_turn(&self, turn_id: &str) -> Result<AssembledTurn, PluginError>;
    async fn cancel_turn(&self, turn_id: &str) -> Result<(), PluginError>;
    /// Push a user-visible message into the target session's turn-input
    /// injection bridge so it surfaces at the next iteration boundary of
    /// the current turn (or at the start of the next turn if the target
    /// is idle). Used for cross-session "poke" flows like
    /// `send_message`, where an inbox note should land at the next
    /// available step rather than waiting for a brand-new task.
    async fn inject_turn_input(
        &self,
        _session_id: &str,
        _message: PluginMessage,
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
    async fn append_session_nodes(
        &self,
        _session_id: &str,
        _request: AppendSessionNodesRequest,
    ) -> Result<AppendSessionNodesResult, PluginError> {
        Err(PluginError::Session(
            "session graph mutation is unavailable in this session".to_string(),
        ))
    }
    async fn prompt_user(
        &self,
        _request: crate::PromptRequest,
    ) -> Result<crate::PromptResponse, PluginError> {
        Err(PluginError::Session(
            "user prompts are unavailable in this session".to_string(),
        ))
    }
    async fn start_turn(
        &self,
        session_id: &str,
        input: TurnInput,
    ) -> Result<AssembledTurn, PluginError> {
        let handle = self.start_turn_stream(session_id, input).await?;
        drop(handle.events);
        self.await_turn(&handle.turn_id).await
    }
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
}

/// Result of a single-shot LLM call via
/// [`SessionManager::direct_completion`].
#[derive(Clone, Debug)]
pub struct DirectCompletion {
    pub text: String,
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
    pub host: Arc<dyn SessionManager>,
    pub state: SessionReadView,
    pub mode_turn_options: ModeTurnOptions,
}

#[derive(Clone)]
pub struct PromptRequestHookContext {
    pub session_id: String,
    pub request: crate::PromptRequest,
    pub host: Arc<dyn SessionManager>,
}

#[derive(Clone)]
pub struct TurnHookContext {
    pub session_id: String,
    pub state: SessionReadView,
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
pub struct SessionStateChangedContext {
    pub session_id: String,
    pub state: SessionReadView,
    pub host: Arc<dyn SessionManager>,
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
    pub status: crate::TurnStatus,
    pub assistant_output: crate::runtime::AssistantOutput,
    pub has_plugin_visible_output: bool,
    pub done_reason: crate::runtime::DoneReason,
    pub execution: crate::runtime::ExecutionSummary,
    pub token_usage: crate::TokenUsage,
    pub tool_calls: Arc<Vec<crate::ToolCallRecord>>,
    pub errors: Arc<Vec<crate::runtime::TurnIssue>>,
    pub typed_finish: Option<serde_json::Value>,
}

impl TurnResultSummary {
    pub fn from_assembled(turn: &AssembledTurn) -> Self {
        Self {
            status: turn.status.clone(),
            assistant_output: turn.assistant_output.clone(),
            has_plugin_visible_output: turn.has_plugin_visible_output,
            done_reason: turn.done_reason.clone(),
            execution: turn.execution.clone(),
            token_usage: turn.token_usage.clone(),
            tool_calls: Arc::new(turn.tool_calls.clone()),
            errors: Arc::new(turn.errors.clone()),
            typed_finish: turn.typed_finish.clone(),
        }
    }
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
    BeforeState,
    BeforeModel,
    BeforeHistory,
}

impl ToolResultProjectionHook {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::BeforeState => "before_state",
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
    pub turn: Arc<TurnResultSummary>,
    pub host: Arc<dyn SessionManager>,
}

#[derive(Clone)]
pub struct CheckpointHookContext {
    pub session_id: String,
    pub checkpoint: CheckpointKind,
    pub state: SessionReadView,
    pub host: Arc<dyn SessionManager>,
}

/// Metadata about a slash command contributed by a plugin.
#[derive(Clone, Debug)]
pub struct CommandDef {
    pub name: &'static str,
    pub usage: &'static str,
    pub description: &'static str,
    pub argument_hint: Option<&'static str>,
    pub argument_options: &'static [&'static str],
    pub takes_argument: bool,
    /// When true, the host may invoke this command while a turn is
    /// streaming instead of queueing it behind the turn. Plugins should
    /// only opt in for handlers that are read-only with respect to the
    /// runtime / persisted state.
    pub runs_out_of_band: bool,
}

/// Runtime context passed to a plugin command handler.
#[derive(Clone)]
pub struct CommandInvocation {
    pub name: String,
    pub argument: Option<String>,
    pub session_id: String,
    pub host: Arc<dyn SessionManager>,
}

/// Result from a plugin command handler, surfaced into the host UI.
#[derive(Clone, Debug)]
pub enum CommandOutcome {
    /// Handler did its work and has no user-visible message.
    Handled,
    /// Push a user-visible system message.
    Message(String),
    /// Push a user-visible error message.
    Error(String),
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
    pub host: Arc<dyn SessionManager>,
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
pub(crate) struct RegisteredExternalOp {
    pub(crate) def: ExternalOpDef,
    pub(crate) handler: ExternalInvokeHandler,
}

mod registry;
pub use registry::{
    PluginFactory, PluginSessionContext, PluginSpec, PluginSpecBuilder, PluginSpecFactory,
    SessionPlugin, SessionReadyContext, StaticPluginFactory,
};

mod monitor;
mod registrar;
mod runtime_impl;
mod services;
mod session_obj;
mod tool_result_projection_builtin;

pub use registrar::{
    CommandRegistrations, ExternalRegistrations, HistoryRegistrations, ModeRegistrations,
    MonitorRegistrations, OutputRegistrations, PluginRegistrar, PromptRegistrations,
    SessionRegistrations, SurfaceRegistrations, ToolCallRegistrations, ToolRegistrations,
    ToolResultRegistrations, TurnRegistrations,
};
pub(crate) use registrar::{RegisteredCommand, RegisteredExclusiveHook, RegisteredHook};
pub use runtime_impl::PluginHost;
pub(crate) use services::NoopSessionManager;
pub use services::{ExternalInvokeError, PersistentRuntimeServices, RuntimeServices};
pub use session_obj::PluginSession;
pub use tool_result_projection_builtin::{
    BuiltinToolResultProjectionPluginFactory, DEFAULT_TOOL_RESULT_PROJECTION_LIMIT_BYTES,
    DEFAULT_TOOL_RESULT_PROJECTION_MAX_LINES, ToolResultProjectionMode,
    ToolResultProjectionPluginConfig, truncate_observation_text,
};

pub(crate) fn builtin_plugin_factories() -> Vec<Arc<dyn PluginFactory>> {
    // Mode plugins (`lash-mode-standard`, `lash-mode-rlm`) must be
    // registered by the embedder before calling `PluginHost::build_session`.
    // lash's own test suite uses an in-tree fake (`testing::test_mode_factories()`)
    // to avoid a dev-dep cycle through the mode crates.
    #[allow(unused_mut)]
    let mut factories: Vec<Arc<dyn PluginFactory>> = vec![Arc::new(monitor::MonitorPluginFactory)];
    #[cfg(test)]
    factories.extend(crate::testing::test_mode_factories());
    factories
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::{ExecutionMode, SessionStateEnvelope, ToolDefinition, ToolParam};

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
                availability: crate::ToolAvailabilityConfig::callable(),
                activation: crate::ToolActivation::Always,
                availability_override: None,
                input_schema_override: None,
                output_schema_override: None,
                execution_mode: crate::ToolExecutionMode::Parallel,
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
                state: SessionReadView::new(SessionStateEnvelope::default()),
                mode_turn_options: ModeTurnOptions::default(),
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
            .invoke_external(
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
    async fn plugin_host_can_invoke_external_for_registered_session() {
        let host = PluginHost::new(vec![Arc::new(MockPluginFactory)]);
        let _session = host.build_standard_session("root", None).expect("session");

        let result = host
            .invoke_external_for_session(
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
    async fn plugin_host_can_invoke_external_for_forked_session() {
        let host = PluginHost::new(vec![Arc::new(MockPluginFactory)]);
        let root = host.build_standard_session("root", None).expect("root");
        let child = root
            .fork_for_session(
                "child",
                ExecutionMode::standard(),
                crate::ContextApproach::default(),
            )
            .expect("child");

        let result = host
            .invoke_external_for_session(
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
    fn duplicate_state_tool_result_projectors_are_rejected() {
        let host = PluginHost::new(vec![
            Arc::new(ProjectorPluginFactory {
                plugin_id: "projector-a",
                hook: ToolResultProjectionHook::BeforeState,
            }),
            Arc::new(ProjectorPluginFactory {
                plugin_id: "projector-b",
                hook: ToolResultProjectionHook::BeforeState,
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
                plugin_id: "projector-state",
                hook: ToolResultProjectionHook::BeforeState,
            }),
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
