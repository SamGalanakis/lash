use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, OnceLock};

use crate::llm::types::LlmResponse;
use crate::runtime::AssembledTurn;
use crate::{
    ExecutionMode, MessageRole, SessionPolicy, SessionStateEnvelope, ToolDefinition, ToolProvider,
    ToolResult, TurnInput,
};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

pub use lash_sansio::{CheckpointKind, PluginMessage, PluginSurfaceEvent, PromptContribution};

pub type PluginFuture<T> = Pin<Box<dyn Future<Output = Result<T, PluginError>> + Send>>;
pub type PluginRuntimeEventHook = Arc<dyn Fn(PluginRuntimeEvent) -> PluginFuture<()> + Send + Sync>;
pub type PluginBackgroundJob = PluginFuture<()>;
pub type SessionConfigMutator = Arc<
    dyn Fn(SessionConfigChangedContext, SessionStateEnvelope) -> PluginFuture<SessionStateEnvelope>
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

/// Reason the history pipeline is being invoked.
#[derive(Clone, Debug)]
pub enum RewriteTrigger {
    /// User invoked `/compact` (or an equivalent plugin command).
    Manual { instructions: Option<String> },
    /// The previous turn overflowed the context window; retry with
    /// compacted history.
    OverflowRecovery,
    /// Session config changed to a smaller context window.
    WindowShrink {
        old_max: Option<usize>,
        new_max: Option<usize>,
    },
    /// Reserved for future scheduled compactors — not fired by any call
    /// site today.
    Periodic,
}

/// Metadata accumulated as a history rewrite pipeline runs.
#[derive(Clone, Debug, Default)]
pub struct HistoryRewriteMetadata {
    pub summarized_token_count: Option<u64>,
    pub pruned_message_count: u32,
    pub produced_summary: bool,
}

/// Mutable state passed through the history rewrite pipeline.
#[derive(Clone, Debug)]
pub struct HistoryState {
    pub messages: Vec<crate::Message>,
    pub tool_calls: Vec<crate::ToolCallRecord>,
    pub metadata: HistoryRewriteMetadata,
}

impl HistoryState {
    pub fn from_state(state: &SessionStateEnvelope) -> Self {
        Self {
            messages: state.project_messages(),
            tool_calls: state.project_tool_calls(),
            metadata: HistoryRewriteMetadata::default(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct SessionReadView(Arc<SessionReadState>);

#[derive(Debug)]
struct SessionReadState {
    meta: SessionReadMeta,
    graph: SessionReadGraph,
    messages: Arc<Vec<crate::Message>>,
    tool_calls: Arc<Vec<crate::ToolCallRecord>>,
    projected_rlm_globals: Arc<serde_json::Map<String, serde_json::Value>>,
}

#[derive(Clone, Debug)]
struct SessionReadMeta {
    session_id: String,
    policy: SessionPolicy,
    iteration: usize,
    token_usage: crate::TokenUsage,
    last_prompt_usage: Option<crate::runtime::PromptUsage>,
    dynamic_state_ref: Option<crate::store::BlobRef>,
    dynamic_state_generation: Option<u64>,
    checkpoint_ref: Option<crate::store::BlobRef>,
    token_ledger: Arc<Vec<crate::runtime::TokenLedgerEntry>>,
}

impl SessionReadMeta {
    fn from_state_ref(state: &SessionStateEnvelope) -> Self {
        Self {
            session_id: state.session_id.clone(),
            policy: state.policy.clone(),
            iteration: state.iteration,
            token_usage: state.token_usage.clone(),
            last_prompt_usage: state.last_prompt_usage.clone(),
            dynamic_state_ref: state.dynamic_state_ref.clone(),
            dynamic_state_generation: state.dynamic_state_generation,
            checkpoint_ref: state.checkpoint_ref.clone(),
            token_ledger: Arc::new(state.token_ledger.clone()),
        }
    }

    fn from_state_owned(state: SessionStateEnvelope) -> Self {
        Self {
            session_id: state.session_id,
            policy: state.policy,
            iteration: state.iteration,
            token_usage: state.token_usage,
            last_prompt_usage: state.last_prompt_usage,
            dynamic_state_ref: state.dynamic_state_ref,
            dynamic_state_generation: state.dynamic_state_generation,
            checkpoint_ref: state.checkpoint_ref,
            token_ledger: Arc::new(state.token_ledger),
        }
    }

    fn to_owned_state(&self, session_graph: crate::SessionGraph) -> SessionStateEnvelope {
        SessionStateEnvelope {
            session_id: self.session_id.clone(),
            policy: self.policy.clone(),
            session_graph,
            iteration: self.iteration,
            token_usage: self.token_usage.clone(),
            last_prompt_usage: self.last_prompt_usage.clone(),
            dynamic_state_ref: self.dynamic_state_ref.clone(),
            dynamic_state_generation: self.dynamic_state_generation,
            dynamic_state_snapshot: None,
            plugin_snapshot_ref: None,
            plugin_snapshot_revision: None,
            plugin_snapshot: None,
            execution_state_snapshot: None,
            token_ledger: self.token_ledger.as_ref().clone(),
            checkpoint_ref: self.checkpoint_ref.clone(),
            persisted_graph_node_count: 0,
            graph_replace_required: false,
        }
    }
}

#[derive(Debug)]
enum SessionReadGraph {
    Owned(Arc<crate::SessionGraph>),
    Derived {
        cache: OnceLock<Arc<crate::SessionGraph>>,
        base_graph: Option<Arc<crate::SessionGraph>>,
        messages: Arc<Vec<crate::Message>>,
        tool_calls: Arc<Vec<crate::ToolCallRecord>>,
    },
}

impl SessionReadView {
    pub fn new(state: SessionStateEnvelope) -> Self {
        let mut state = state;
        let graph = Arc::new(std::mem::take(&mut state.session_graph));
        let messages = graph.shared_projected_messages();
        let tool_calls = graph.shared_projected_tool_calls();
        Self(Arc::new(SessionReadState {
            meta: SessionReadMeta::from_state_owned(state),
            graph: SessionReadGraph::Owned(Arc::clone(&graph)),
            messages,
            tool_calls,
            projected_rlm_globals: graph.shared_projected_rlm_globals(),
        }))
    }

    pub fn from_projection_state(
        mut state: SessionStateEnvelope,
        messages: Arc<Vec<crate::Message>>,
        tool_calls: Arc<Vec<crate::ToolCallRecord>>,
    ) -> Self {
        Self(Arc::new(SessionReadState {
            meta: SessionReadMeta::from_state_owned({
                state.session_graph = crate::SessionGraph::default();
                state
            }),
            graph: SessionReadGraph::Derived {
                cache: OnceLock::new(),
                base_graph: None,
                messages: Arc::clone(&messages),
                tool_calls: Arc::clone(&tool_calls),
            },
            messages,
            tool_calls,
            projected_rlm_globals: Arc::new(serde_json::Map::new()),
        }))
    }

    pub fn from_graph_projection(
        state: &SessionStateEnvelope,
        base_graph: crate::SessionGraph,
        messages: Arc<Vec<crate::Message>>,
        tool_calls: Arc<Vec<crate::ToolCallRecord>>,
    ) -> Self {
        Self(Arc::new(SessionReadState {
            meta: SessionReadMeta::from_state_ref(state),
            graph: SessionReadGraph::Derived {
                cache: OnceLock::new(),
                base_graph: Some(Arc::new(base_graph.clone())),
                messages: Arc::clone(&messages),
                tool_calls: Arc::clone(&tool_calls),
            },
            messages,
            tool_calls,
            projected_rlm_globals: base_graph.shared_projected_rlm_globals(),
        }))
    }

    pub fn from_runtime_state(state: &SessionStateEnvelope) -> Self {
        Self(Arc::new(SessionReadState {
            meta: SessionReadMeta::from_state_ref(state),
            graph: SessionReadGraph::Owned(Arc::new(state.session_graph.clone())),
            messages: state.session_graph.shared_projected_messages(),
            tool_calls: state.session_graph.shared_projected_tool_calls(),
            projected_rlm_globals: state.session_graph.shared_projected_rlm_globals(),
        }))
    }

    fn graph_arc(&self) -> &Arc<crate::SessionGraph> {
        match &self.0.graph {
            SessionReadGraph::Owned(graph) => graph,
            SessionReadGraph::Derived {
                cache,
                base_graph,
                messages,
                tool_calls,
            } => cache.get_or_init(|| {
                let mut graph = base_graph
                    .as_ref()
                    .map(|graph| graph.as_ref().clone())
                    .unwrap_or_default();
                graph.merge_active_projection(messages.as_slice(), tool_calls.as_slice());
                Arc::new(graph)
            }),
        }
    }

    fn messages_arc(&self) -> &Arc<Vec<crate::Message>> {
        &self.0.messages
    }

    fn tool_calls_arc(&self) -> &Arc<Vec<crate::ToolCallRecord>> {
        &self.0.tool_calls
    }

    fn projected_rlm_globals_arc(&self) -> &Arc<serde_json::Map<String, serde_json::Value>> {
        &self.0.projected_rlm_globals
    }

    pub fn session_id(&self) -> &str {
        &self.0.meta.session_id
    }

    pub fn policy(&self) -> &SessionPolicy {
        &self.0.meta.policy
    }

    pub fn session_graph(&self) -> &crate::SessionGraph {
        self.graph_arc().as_ref()
    }

    pub fn messages(&self) -> &[crate::Message] {
        self.messages_arc().as_slice()
    }

    pub fn tool_calls(&self) -> &[crate::ToolCallRecord] {
        self.tool_calls_arc().as_slice()
    }

    pub fn projected_rlm_globals(&self) -> &serde_json::Map<String, serde_json::Value> {
        self.projected_rlm_globals_arc().as_ref()
    }

    pub fn iteration(&self) -> usize {
        self.0.meta.iteration
    }

    pub fn token_usage(&self) -> &crate::TokenUsage {
        &self.0.meta.token_usage
    }

    pub fn last_prompt_usage(&self) -> Option<&crate::runtime::PromptUsage> {
        self.0.meta.last_prompt_usage.as_ref()
    }

    pub fn dynamic_state_ref(&self) -> Option<&crate::store::BlobRef> {
        self.0.meta.dynamic_state_ref.as_ref()
    }

    pub fn dynamic_state_generation(&self) -> Option<u64> {
        self.0.meta.dynamic_state_generation
    }

    pub fn checkpoint_ref(&self) -> Option<&crate::store::BlobRef> {
        self.0.meta.checkpoint_ref.as_ref()
    }

    pub fn token_ledger(&self) -> &[crate::runtime::TokenLedgerEntry] {
        self.0.meta.token_ledger.as_slice()
    }

    pub fn usage_report(&self) -> crate::runtime::SessionUsageReport {
        crate::runtime::SessionUsageReport::from_entries(self.token_ledger())
    }

    pub fn to_owned_state(&self) -> SessionStateEnvelope {
        self.0.meta.to_owned_state(self.session_graph().clone())
    }
}

/// Context passed to a turn-context transform.
#[derive(Clone)]
pub struct TurnTransformContext {
    pub session_id: String,
    pub state: SessionReadView,
    pub prompt_usage: Option<crate::runtime::PromptUsage>,
    pub max_context_tokens: Option<usize>,
    pub host: Arc<dyn SessionManager>,
}

/// Context passed to a history rewriter.
#[derive(Clone)]
pub struct RewriteContext {
    pub session_id: String,
    pub trigger: RewriteTrigger,
    pub state: SessionReadView,
    pub host: Arc<dyn SessionManager>,
}

#[derive(Debug, thiserror::Error, Clone)]
pub enum HistoryError {
    #[error("history pipeline error: {0}")]
    Pipeline(String),
    #[error("history session error: {0}")]
    Session(String),
}

impl From<PluginError> for HistoryError {
    fn from(value: PluginError) -> Self {
        Self::Session(value.to_string())
    }
}

/// Prepares the ephemeral turn context presented to the model.
#[async_trait::async_trait]
pub trait TurnContextTransform: Send + Sync {
    fn id(&self) -> &'static str;
    async fn transform(
        &self,
        ctx: &TurnTransformContext,
        input: crate::session_model::context::PreparedContext,
    ) -> Result<crate::session_model::context::PreparedContext, HistoryError>;
}

/// Performs a permanent transform on persisted history (compaction,
/// overflow recovery, manual `/compact`, …).
#[async_trait::async_trait]
pub trait HistoryRewriter: Send + Sync {
    fn id(&self) -> &'static str;
    fn accepts(&self, _trigger: &RewriteTrigger) -> bool {
        true
    }
    async fn rewrite(
        &self,
        ctx: &RewriteContext,
        input: HistoryState,
    ) -> Result<HistoryState, HistoryError>;
}

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

pub type SessionSnapshot = SessionStateEnvelope;

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
    pub prompt_overrides: Vec<crate::PromptSectionOverride>,
}

impl Default for SessionContextSurface {
    fn default() -> Self {
        Self {
            include_base_tools: true,
            tool_providers: Vec::new(),
            prompt_contributions: Vec::new(),
            prompt_overrides: Vec::new(),
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
            .field("prompt_override_count", &self.prompt_overrides.len())
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
    #[serde(skip)]
    pub context_surface: SessionContextSurface,
    /// Per-execution-mode "extras" that configure mode-specific
    /// behavior at session-creation time. The base request stays
    /// mode-agnostic; each `ExecutionMode` defines its own struct.
    #[serde(default)]
    pub mode_extras: ModeExtras,
    /// Label for the token-cost ledger. When this session's turns
    /// complete, their token usage is accumulated under this label on
    /// the parent session's `token_ledger`. Examples: `"predict"`,
    /// `"agent_call"`. Defaults to `"child"` if unset.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage_source: Option<String>,
}

/// Per-execution-mode configuration carried on a `SessionCreateRequest`.
/// Each variant matches an `ExecutionMode` value and carries the
/// settings only that mode cares about. Adding a new mode means adding
/// a new variant with its own struct — no mode-specific fields ever
/// leak into the base request.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum ModeExtras {
    Standard(StandardCreateExtras),
    Rlm(RlmCreateExtras),
}

impl Default for ModeExtras {
    fn default() -> Self {
        Self::Standard(StandardCreateExtras::default())
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct StandardCreateExtras {}

/// RLM-mode session config. Carries the choice of how the model
/// terminates the session (prose vs `finish`-with-optional-schema).
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct RlmCreateExtras {
    #[serde(default)]
    pub termination: RlmTermination,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SessionAppendNode {
    Message {
        message: PluginMessage,
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
}

/// How a RLM session ends. Top-level chat sessions use
/// `ProseWithoutFence` (today's behavior); typed sub-sessions spawned
/// via `predict` use `Finish` so the captured value is the terminal
/// result.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[derive(Default)]
pub enum RlmTermination {
    /// Terminate when the model writes prose with no fenced lashlang
    /// block. The prose IS the assistant's final reply. lashlang
    /// `finish` inside a fenced block continues to be an error in this
    /// mode (the model shouldn't end the session from inside the
    /// language).
    #[default]
    ProseWithoutFence,
    /// Terminate when the model calls `finish <expr>` from inside a
    /// fenced lashlang block. The captured value is the terminal
    /// result. Prose-without-fence becomes a soft error that loops the
    /// model with a "you must call finish" reminder. When `schema` is
    /// `Some`, the captured value is validated against the JSON Schema
    /// before being accepted; mismatches loop with an explanation.
    Finish {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        schema: Option<serde_json::Value>,
    },
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
    async fn spawn_background_job(
        &self,
        _session_id: &str,
        _label: &str,
        _job: PluginBackgroundJob,
    ) -> Result<(), PluginError> {
        Err(PluginError::Session(
            "background jobs are unavailable in this session".to_string(),
        ))
    }
    async fn await_background_jobs(&self, _session_id: &str) -> Result<(), PluginError> {
        Ok(())
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
    pub prompt: crate::PromptContext,
    pub state: SessionReadView,
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
    SessionConfigChanged(SessionConfigChangedContext),
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
    #[serde(default)]
    pub revision: u64,
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
    pub prompt_request_hooks: Vec<PromptRequestHook>,
    pub tool_surface_contributors: Vec<ToolSurfaceContributor>,
    pub before_turn_hooks: Vec<BeforeTurnHook>,
    pub before_tool_call_hooks: Vec<BeforeToolCallHook>,
    pub after_tool_call_hooks: Vec<AfterToolCallHook>,
    pub after_turn_hooks: Vec<AfterTurnHook>,
    pub checkpoint_hooks: Vec<CheckpointHook>,
    pub assistant_stream_hooks: Vec<AssistantStreamHook>,
    pub assistant_response_hooks: Vec<AssistantResponseHook>,
    pub tool_result_projectors: BTreeMap<ToolResultProjectionHook, ToolResultProjector>,
    pub runtime_event_hooks: Vec<PluginRuntimeEventHook>,
    pub session_config_mutators: Vec<SessionConfigMutator>,
    pub external_ops: Vec<(ExternalOpDef, ExternalInvokeHandler)>,
    pub commands: Vec<(CommandDef, CommandHandler)>,
    pub turn_context_transforms: Vec<(i32, Arc<dyn TurnContextTransform>)>,
    pub history_rewriters: Vec<(i32, Arc<dyn HistoryRewriter>)>,
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

    pub fn with_prompt_request(mut self, hook: PromptRequestHook) -> Self {
        self.prompt_request_hooks.push(hook);
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

    pub fn with_runtime_event(mut self, hook: PluginRuntimeEventHook) -> Self {
        self.runtime_event_hooks.push(hook);
        self
    }

    pub fn with_session_config_mutator(mut self, hook: SessionConfigMutator) -> Self {
        self.session_config_mutators.push(hook);
        self
    }

    pub fn with_external_op(mut self, def: ExternalOpDef, handler: ExternalInvokeHandler) -> Self {
        self.external_ops.push((def, handler));
        self
    }

    pub fn with_command(mut self, def: CommandDef, handler: CommandHandler) -> Self {
        self.commands.push((def, handler));
        self
    }

    pub fn with_turn_context_transform(
        mut self,
        priority: i32,
        transform: Arc<dyn TurnContextTransform>,
    ) -> Self {
        self.turn_context_transforms.push((priority, transform));
        self
    }

    pub fn with_history_rewriter(
        mut self,
        priority: i32,
        rewriter: Arc<dyn HistoryRewriter>,
    ) -> Self {
        self.history_rewriters.push((priority, rewriter));
        self
    }
}

#[derive(Clone, Debug)]
pub struct PluginSessionContext {
    pub session_id: String,
    pub execution_mode: ExecutionMode,
    pub context_approach: crate::ContextApproach,
}

#[derive(Clone)]
pub struct SessionReadyContext {
    pub session_id: String,
    pub execution_mode: ExecutionMode,
    pub context_approach: crate::ContextApproach,
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
            revision: self.snapshot_revision(),
            state: None,
        })
    }

    fn snapshot_revision(&self) -> u64 {
        0
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
    fn supports_context_approach(&self, _approach: &crate::ContextApproach) -> bool {
        false
    }
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
        for hook in &self.spec.prompt_request_hooks {
            reg.prompt().on_request(Arc::clone(hook));
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
        for hook in &self.spec.runtime_event_hooks {
            reg.session().on_event(Arc::clone(hook));
        }
        for hook in &self.spec.session_config_mutators {
            reg.session().config_mutator(Arc::clone(hook));
        }
        for (def, handler) in &self.spec.external_ops {
            reg.external().op(def.clone(), Arc::clone(handler))?;
        }
        for (def, handler) in &self.spec.commands {
            reg.commands().register(def.clone(), Arc::clone(handler))?;
        }
        for (priority, transform) in &self.spec.turn_context_transforms {
            reg.history().prepare_turn(*priority, Arc::clone(transform));
        }
        for (priority, rewriter) in &self.spec.history_rewriters {
            reg.history().rewrite(*priority, Arc::clone(rewriter));
        }
        Ok(())
    }
}
mod runtime_impl;
mod tool_result_projection_builtin;

#[path = "plugin_builtin/observational_memory.rs"]
mod observational_memory;
#[path = "plugin_builtin/rolling_history.rs"]
mod rolling_history;

pub use observational_memory::ObservationalMemoryPluginFactory;
pub use rolling_history::RollingHistoryPluginFactory;

pub use runtime_impl::{
    CommandRegistrations, ExternalInvokeError, ExternalRegistrations, HistoryRegistrations,
    OutputRegistrations, PluginHost, PluginRegistrar, PluginSession, PromptRegistrations,
    RuntimeServices, SessionRegistrations, SurfaceRegistrations, ToolCallRegistrations,
    ToolRegistrations, ToolResultRegistrations, TurnRegistrations,
};
pub use tool_result_projection_builtin::{
    BuiltinToolResultProjectionPluginFactory, ToolResultProjectionMode,
    ToolResultProjectionPluginConfig,
};
pub(crate) use tool_result_projection_builtin::{
    DEFAULT_TOOL_RESULT_PROJECTION_LIMIT_BYTES, DEFAULT_TOOL_RESULT_PROJECTION_MAX_LINES,
};

#[cfg(feature = "sqlite-store")]
#[path = "plugin_builtin.rs"]
mod builtin;

#[cfg(feature = "sqlite-store")]
pub use builtin::{
    BuiltinPlanModePluginFactory, BuiltinPlanTrackerPluginFactory,
    BuiltinPromptContextPluginFactory, BuiltinUiActivityPluginFactory, PromptContextPluginConfig,
};

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::{
        ExecutionMode, PromptSectionName, SessionStateEnvelope, ToolDefinition, ToolParam,
    };

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
                input_schema_override: None,
                output_schema_override: None,
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

    use crate::test_support::MockSessionManager;

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
                            block: "plugin_prompt".to_string(),
                            title: Some("Plugin Prompt".to_string()),
                            priority: 0,
                            content: "Structured plugin prompt".to_string(),
                        },
                        PromptContribution {
                            section: PromptSectionName::Guidance,
                            block: "dynamic_note".to_string(),
                            title: Some("Dynamic Note".to_string()),
                            priority: 1,
                            content: "dynamic note".to_string(),
                        },
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
                prompt: crate::PromptContext::default(),
                state: SessionReadView::new(SessionStateEnvelope::default()),
            })
            .await
            .expect("prompt contributions");
        assert_eq!(
            contributions,
            vec![
                PromptContribution {
                    section: PromptSectionName::Guidance,
                    block: "plugin_prompt".to_string(),
                    title: Some("Plugin Prompt".to_string()),
                    priority: 0,
                    content: "Structured plugin prompt".to_string(),
                },
                PromptContribution {
                    section: PromptSectionName::Guidance,
                    block: "dynamic_note".to_string(),
                    title: Some("Dynamic Note".to_string()),
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
                ExecutionMode::Standard,
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
