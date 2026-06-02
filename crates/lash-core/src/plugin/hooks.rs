use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use super::*;

pub type PluginFuture<T> = Pin<Box<dyn Future<Output = Result<T, PluginError>> + Send>>;
pub type PluginLifecycleEventHook =
    Arc<dyn Fn(PluginLifecycleEvent) -> PluginFuture<()> + Send + Sync>;
pub type PluginSessionTask = PluginFuture<()>;
pub type SessionConfigMutator = Arc<
    dyn Fn(SessionConfigChangedContext, SessionPolicy) -> PluginFuture<SessionPolicy> + Send + Sync,
>;
pub type BeforeTurnHook =
    Arc<dyn Fn(TurnHookContext) -> PluginFuture<Vec<PluginDirective>> + Send + Sync>;
pub type BeforeToolCallHook =
    Arc<dyn Fn(ToolCallHookContext) -> PluginFuture<Vec<PluginDirective>> + Send + Sync>;
pub type AfterToolCallHook =
    Arc<dyn Fn(ToolResultHookContext) -> PluginFuture<Vec<PluginDirective>> + Send + Sync>;
pub type ToolResultProjector =
    Arc<dyn Fn(ToolResultProjectionContext) -> PluginFuture<crate::ModelToolReturn> + Send + Sync>;
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

#[derive(Clone)]
pub struct PromptHookContext {
    pub session_id: String,
    pub sessions: Arc<dyn SessionStateService>,
    pub state: SessionReadView,
    pub protocol_turn_options: ProtocolTurnOptions,
    pub turn_context: crate::TurnContext,
}

#[derive(Clone)]
pub struct TurnHookContext {
    pub session_id: String,
    pub state: SessionReadView,
    pub sessions: Arc<dyn SessionStateService>,
    pub turn_context: crate::TurnContext,
}

#[derive(Clone)]
pub struct SessionConfigChangedContext {
    pub session_id: String,
    pub previous: SessionPolicy,
    pub current: SessionPolicy,
    pub sessions: Arc<dyn SessionStateService>,
}

#[derive(Clone)]
pub struct SessionStateChangedContext {
    pub session_id: String,
    pub state: SessionReadView,
    pub sessions: Arc<dyn SessionStateService>,
    pub session_graph: Arc<dyn SessionGraphService>,
    pub direct_completions: crate::DirectCompletionClient<'static>,
}

#[derive(Clone)]
pub enum PluginLifecycleEvent {
    TurnFinalized(Arc<AssembledTurn>),
    /// Best-effort observer hook emitted after durable session state advances.
    ///
    /// Hook failures are isolated from the foreground turn: the turn has already
    /// committed, and observers must not affect that commit.
    TurnPersisted(Box<SessionStateChangedContext>),
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
    pub argument_projection: crate::ToolArgumentProjectionPolicy,
    pub turn_context: crate::TurnContext,
    pub(crate) sessions: Arc<dyn SessionStateService>,
}

impl ToolCallHookContext {
    pub fn new(
        session_id: String,
        tool_name: String,
        args: serde_json::Value,
        argument_projection: crate::ToolArgumentProjectionPolicy,
        turn_context: crate::TurnContext,
        sessions: Arc<dyn SessionStateService>,
    ) -> Self {
        Self {
            session_id,
            tool_name,
            args,
            argument_projection,
            turn_context,
            sessions,
        }
    }

    pub async fn session_snapshot(&self) -> Result<SessionSnapshot, PluginError> {
        self.sessions.snapshot_session(&self.session_id).await
    }

    pub async fn set_tools_availability(
        &self,
        names: &[String],
        availability: Option<crate::ToolAvailability>,
    ) -> Result<u64, PluginError> {
        self.sessions
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
    pub(crate) sessions: Arc<dyn SessionStateService>,
}

impl ToolResultHookContext {
    pub fn new(
        session_id: String,
        tool_name: String,
        args: serde_json::Value,
        result: ToolResult,
        duration_ms: u64,
        turn_context: crate::TurnContext,
        sessions: Arc<dyn SessionStateService>,
    ) -> Self {
        Self {
            session_id,
            tool_name,
            args,
            result,
            duration_ms,
            turn_context,
            sessions,
        }
    }

    pub async fn session_snapshot(&self) -> Result<SessionSnapshot, PluginError> {
        self.sessions.snapshot_session(&self.session_id).await
    }

    pub async fn set_tools_availability(
        &self,
        names: &[String],
        availability: Option<crate::ToolAvailability>,
    ) -> Result<u64, PluginError> {
        self.sessions
            .set_tools_availability(&self.session_id, names, availability)
            .await
    }
}

#[derive(Clone)]
pub struct ToolResultProjectionContext {
    pub session_id: String,
    pub call_id: String,
    pub tool_name: String,
    pub args: serde_json::Value,
    pub output: crate::ToolCallOutput,
    pub duration_ms: u64,
}

#[derive(Clone)]
pub struct TurnResultHookContext {
    pub session_id: String,
    pub turn: Arc<TurnResultSummary>,
    pub sessions: Arc<dyn SessionStateService>,
}

#[derive(Clone)]
pub struct CheckpointHookContext {
    pub session_id: String,
    pub checkpoint: CheckpointKind,
    pub state: SessionReadView,
    pub sessions: Arc<dyn SessionStateService>,
    pub session_lifecycle: Arc<dyn SessionLifecycleService>,
    pub session_graph: Arc<dyn SessionGraphService>,
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
    pub events: Vec<PluginRuntimeEvent>,
    /// When `true`, the runtime cancels the in-flight LLM call the
    /// moment this hook returns and finalizes the turn using whatever
    /// text has been streamed so far. Any plugin may set this — the
    /// first to raise it wins. Used by protocol plugins to enforce
    /// one-block-per-turn contracts (e.g. the RLM stream mask aborts
    /// as soon as the first lashlang fence closes).
    pub abort_stream: bool,
}

#[derive(Clone)]
pub struct AssistantResponseHookContext {
    pub session_id: String,
    pub response: crate::LlmResponse,
}

#[derive(Clone, Debug)]
pub struct AssistantResponseTransform {
    pub response: crate::LlmResponse,
    pub events: Vec<PluginRuntimeEvent>,
}
