mod assembly;
mod builder;
mod config_ops;
mod effect_host;
mod environment;
mod host;
mod io;
mod lifecycle;
mod observation;
mod session_api;
mod session_manager;
mod session_ops;
mod state;
#[cfg(test)]
mod tests;
mod turn_commit_draft;
mod turn_commit_pipeline;
mod turn_driver;
mod turn_graph_editor;
mod turn_loop;
mod usage;

use std::any::Any;
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::sync::Mutex as StdMutex;
use std::sync::atomic::{AtomicBool, Ordering};

use tokio::sync::{Mutex, mpsc};
use tokio_util::sync::CancellationToken;

use crate::llm::types::{
    LlmOutputPart, LlmProviderTraceEvent, LlmProviderTraceSender, LlmRequest, LlmResponse,
    LlmStreamEvent, LlmUsage,
};
use crate::plugin::runtime_host::RuntimeSessionHost;
use crate::plugin::{
    CheckpointHookContext, PluginMessage, PrepareTurnRequest, SessionConfigChangedContext,
    SessionRelation,
};
use crate::sansio::{LlmCallError, Response};
use crate::session_model::{
    Message, MessageRole, Part, PartKind, PruneState, SessionEvent, SessionPolicy, TokenUsage,
    fresh_message_id, make_error_event, reassign_part_ids, shared_parts, transport_stream_events,
};
use crate::{
    CheckpointKind, ExecutionMode, PersistentRuntimeServices, PluginActionInvokeError,
    PromptHookContext, RuntimeServices, SandboxMessage, Session, SessionCreateRequest,
    SessionError, SessionHandle, SessionSnapshot, SessionStartPoint, ToolCallRecord, TurnFinish,
    TurnOutcome, TurnStop,
};
use crate::{Effect, TurnMachine};

use host::*;
use session_manager::*;
use turn_commit_draft::*;
use turn_commit_pipeline::*;
use turn_driver::*;

// `PromptUsage` is re-exported below alongside the runtime's own types.
pub use lash_sansio::PromptUsage;

use assembly::{
    LlmDebugText, LlmDebugToolCall, LlmStreamDebugState, LlmStreamEventLog, LlmStreamSummary,
    StandardStreamAccumulator, StandardStreamState, TurnAssembler,
};
#[cfg(test)]
#[allow(unused_imports)]
use assembly::{classify_output_state, sanitize_assistant_output};
pub use builder::EmbeddedRuntimeBuilder;
pub use effect_host::{
    DirectEffectLocalExecutor, EffectInvocation, EffectInvocationMetadata, EffectOrigin,
    LocalRuntimeEffectHost, RuntimeEffectHost, RuntimeEffectKind, TurnEffectLocalExecutor,
};
pub use environment::{ParkedSession, Residency, RuntimeEnvironment, RuntimeEnvironmentBuilder};
pub use host::{
    BackgroundCancelPolicy, BackgroundClosePolicy, BackgroundRuntimeHost, BackgroundTaskAttempt,
    BackgroundTaskEvent, BackgroundTaskFilter, BackgroundTaskHost, BackgroundTaskId,
    BackgroundTaskKind, BackgroundTaskOutcome, BackgroundTaskRecord, BackgroundTaskRegistration,
    BackgroundTaskScope, BackgroundTaskState, EmbeddedRuntimeHost, LocalBackgroundTaskCancel,
    LocalBackgroundTaskHost, RuntimeCoreConfig,
};
use io::normalize_input_items;
pub use observation::{RuntimeHandle, RuntimeObservation};
pub use state::{PersistedSessionState, SessionStateEnvelope};
use state::{
    append_session_nodes_to_state, apply_residency_on_load, apply_session_checkpoint,
    apply_session_head, normalize_session_graph,
};
pub use usage::{
    SessionUsageReport, TokenLedgerEntry, UsageReportRow, UsageTotals, diff_token_ledger,
    diff_usage_reports,
};
use usage::{merge_ledger_entry, merge_usage_delta_entries, normalize_prompt_usage};

#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RuntimeTurnPhase {
    ContextTransform,
    BeforeTurnHooks,
    PromptBuild,
    EffectLoop,
    FinalizeTurn,
    PersistTurn,
}

#[doc(hidden)]
pub trait RuntimeTurnPhaseProbe: Send + Sync {
    fn begin(&self, phase: RuntimeTurnPhase);
    fn end(&self, phase: RuntimeTurnPhase);
}

/// Host-provided per-turn input.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InputItem {
    Text { text: String },
    ImageRef { id: String },
}

impl InputItem {
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text { text: text.into() }
    }

    pub fn image_ref(id: impl Into<String>) -> Self {
        Self::ImageRef { id: id.into() }
    }
}

/// Host-provided per-turn input.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TurnInput {
    pub items: Vec<InputItem>,
    #[serde(default)]
    pub image_blobs: HashMap<String, Vec<u8>>,
    /// Per-turn override for mode-owned turn options.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mode_turn_options: Option<crate::ModeTurnOptions>,
    /// Optional externally-stable trace turn id. Normal runtime callers leave
    /// this empty and the runtime generates one per outer turn.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trace_turn_id: Option<String>,
    #[serde(skip)]
    pub mode_extension: Option<ModeTurnExtensionHandle>,
    #[serde(skip)]
    pub turn_context: TurnContext,
}

impl TurnInput {
    pub fn empty() -> Self {
        Self::items(std::iter::empty())
    }

    pub fn text(text: impl Into<String>) -> Self {
        Self::items([InputItem::text(text)])
    }

    pub fn items(items: impl IntoIterator<Item = InputItem>) -> Self {
        Self {
            items: items.into_iter().collect(),
            image_blobs: HashMap::new(),
            mode_turn_options: None,
            trace_turn_id: None,
            mode_extension: None,
            turn_context: TurnContext::default(),
        }
    }

    pub fn with_image_blob(mut self, id: impl Into<String>, bytes: Vec<u8>) -> Self {
        self.image_blobs.insert(id.into(), bytes);
        self
    }

    pub fn with_image_blobs<I, K>(mut self, image_blobs: I) -> Self
    where
        I: IntoIterator<Item = (K, Vec<u8>)>,
        K: Into<String>,
    {
        self.image_blobs.extend(
            image_blobs
                .into_iter()
                .map(|(id, bytes)| (id.into(), bytes)),
        );
        self
    }

    pub fn with_image_ref(mut self, id: impl Into<String>, bytes: Vec<u8>) -> Self {
        let id = id.into();
        self.items.push(InputItem::image_ref(id.clone()));
        self.image_blobs.insert(id, bytes);
        self
    }

    pub fn with_mode_turn_options(mut self, options: crate::ModeTurnOptions) -> Self {
        self.mode_turn_options = Some(options);
        self
    }

    pub fn with_trace_turn_id(mut self, trace_turn_id: impl Into<String>) -> Self {
        self.trace_turn_id = Some(trace_turn_id.into());
        self
    }
}

#[derive(Clone, Default)]
pub struct TurnContext {
    plugin_inputs: HashMap<&'static str, Arc<dyn Any + Send + Sync>>,
    provider: Option<crate::ProviderHandle>,
    model: Option<String>,
    model_variant: Option<Option<String>>,
    prompt: crate::PromptLayer,
}

impl TurnContext {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert_plugin_input<T>(&mut self, plugin_id: &'static str, input: T)
    where
        T: Send + Sync + 'static,
    {
        self.plugin_inputs.insert(plugin_id, Arc::new(input));
    }

    pub fn set_provider(&mut self, provider: crate::ProviderHandle) {
        self.provider = Some(provider);
    }

    pub fn provider(&self) -> Option<&crate::ProviderHandle> {
        self.provider.as_ref()
    }

    pub fn set_model(&mut self, model: impl Into<String>, variant: Option<String>) {
        self.model = Some(model.into());
        self.model_variant = Some(variant);
    }

    pub fn model_selection(&self) -> Option<(&str, Option<&str>)> {
        self.model.as_deref().map(|model| {
            (
                model,
                self.model_variant
                    .as_ref()
                    .and_then(|variant| variant.as_deref()),
            )
        })
    }

    pub fn plugin_input<T>(&self, plugin_id: &'static str) -> Option<&T>
    where
        T: 'static,
    {
        self.plugin_inputs
            .get(plugin_id)
            .and_then(|input| input.downcast_ref::<T>())
    }

    pub fn has_plugin_input(&self, plugin_id: &'static str) -> bool {
        self.plugin_inputs.contains_key(plugin_id)
    }

    pub fn set_prompt_template(&mut self, template: crate::PromptTemplate) {
        self.prompt.template = Some(template);
    }

    pub fn add_prompt_contribution(&mut self, contribution: crate::PromptContribution) {
        self.prompt.add_contribution(contribution);
    }

    pub fn replace_prompt_slot(
        &mut self,
        slot: crate::PromptSlot,
        contributions: impl IntoIterator<Item = crate::PromptContribution>,
    ) {
        self.prompt.replace_slot(slot, contributions);
    }

    pub fn clear_prompt_slot(&mut self, slot: crate::PromptSlot) {
        self.prompt.clear_slot(slot);
    }

    pub fn set_prompt_layer(&mut self, prompt: crate::PromptLayer) {
        self.prompt = prompt;
    }

    pub fn prompt_layer(&self) -> &crate::PromptLayer {
        &self.prompt
    }
}

impl fmt::Debug for TurnContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TurnContext")
            .field(
                "plugin_inputs",
                &self.plugin_inputs.keys().collect::<Vec<_>>(),
            )
            .field("has_provider", &self.provider.is_some())
            .field("has_model", &self.model.is_some())
            .field("has_prompt_layer", &(!self.prompt.is_empty()))
            .finish()
    }
}

#[derive(Clone)]
pub struct ModeTurnExtensionHandle(Arc<dyn ModeTurnExtension>);

impl ModeTurnExtensionHandle {
    pub fn new(extension: impl ModeTurnExtension + 'static) -> Self {
        Self(Arc::new(extension))
    }

    pub fn as_any(&self) -> &dyn Any {
        self.0.as_any()
    }

    pub fn prompt_contributions(&self) -> Vec<crate::PromptContribution> {
        self.0.prompt_contributions()
    }
}

impl fmt::Debug for ModeTurnExtensionHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("ModeTurnExtensionHandle(..)")
    }
}

pub trait ModeTurnExtension: Send + Sync {
    fn as_any(&self) -> &dyn Any;

    fn prompt_contributions(&self) -> Vec<crate::PromptContribution> {
        Vec::new()
    }
}

#[derive(Clone)]
pub struct ModeSessionExtensionHandle(Arc<dyn ModeSessionExtension>);

impl ModeSessionExtensionHandle {
    pub fn new(extension: impl ModeSessionExtension + 'static) -> Self {
        Self(Arc::new(extension))
    }

    pub fn as_any(&self) -> &dyn Any {
        self.0.as_any()
    }
}

impl fmt::Debug for ModeSessionExtensionHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("ModeSessionExtensionHandle(..)")
    }
}

pub trait ModeSessionExtension: Send + Sync {
    fn as_any(&self) -> &dyn Any;
}

#[derive(Clone, Debug)]
pub(super) enum NormalizedItem {
    Text(String),
    Image(crate::AttachmentRef),
}

/// Canonical assistant output payload.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct AssistantOutput {
    pub safe_text: String,
    pub raw_text: String,
    pub state: OutputState,
}

/// Quality and usability of assembled terminal output.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OutputState {
    Usable,
    EmptyOutput,
    TracebackOnly,
    RecoveredFromError,
}

/// RLM code execution output observed during a turn.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CodeOutputRecord {
    pub output: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// High-level execution summary shared across execution modes.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ExecutionSummary {
    pub mode: ExecutionMode,
    #[serde(default)]
    pub had_tool_calls: bool,
    #[serde(default)]
    pub had_code_execution: bool,
}

/// Structured issue surfaced during turn execution.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TurnIssue {
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub terminal_reason: Option<crate::LlmTerminalReason>,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw: Option<String>,
}

/// Canonical high-level turn result returned to hosts.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct AssembledTurn {
    pub state: SessionStateEnvelope,
    pub outcome: crate::TurnOutcome,
    pub assistant_output: AssistantOutput,
    pub execution: ExecutionSummary,
    #[serde(default)]
    pub token_usage: TokenUsage,
    /// Per-(session, source, model) ledger entries for child sessions whose
    /// LLM calls completed during this turn. `token_usage` above is the
    /// parent's own LLM tokens; `total_usage` (on the embed-facing
    /// `TurnResult`) sums both.
    #[serde(default)]
    pub children_usage: Vec<TokenLedgerEntry>,
    #[serde(default)]
    pub tool_calls: Vec<ToolCallRecord>,
    #[serde(default)]
    pub errors: Vec<TurnIssue>,
}

/// Result of driving one logical host turn through any foreground handoffs.
///
/// A handoff is an internal runtime continuation, similar to compaction from a
/// host's perspective. Callers that need a final answer can use
/// [`LashRuntime::stream_turn_following_handoffs`] and inspect `final_turn()`.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct FollowedTurn {
    pub turns: Vec<AssembledTurn>,
}

impl FollowedTurn {
    pub fn final_turn(&self) -> Option<&AssembledTurn> {
        self.turns.last()
    }

    pub fn into_final_turn(mut self) -> Option<AssembledTurn> {
        self.turns.pop()
    }

    pub fn handoff_count(&self) -> usize {
        self.turns
            .iter()
            .filter(|turn| matches!(turn.outcome, crate::TurnOutcome::Handoff { .. }))
            .count()
    }
}

/// Runtime error for unexpected failures.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RuntimeError {
    pub code: String,
    pub message: String,
}

impl std::fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.code, self.message)
    }
}

impl std::error::Error for RuntimeError {}

/// Termination policy knobs.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TerminationPolicy {
    #[serde(default)]
    pub treat_missing_done_as_failure: bool,
}

impl Default for TerminationPolicy {
    fn default() -> Self {
        Self {
            treat_missing_done_as_failure: true,
        }
    }
}

/// Host event sink for low-level streaming runtime events.
/// `SessionEvent` is intentionally mode-specific and should be treated as preview/progress data.
#[async_trait::async_trait]
pub trait EventSink: Send + Sync {
    fn is_noop(&self) -> bool {
        false
    }

    async fn emit(&self, event: SessionEvent);
}

/// No-op sink useful for callers that only care about final state.
pub struct NoopEventSink;

#[async_trait::async_trait]
impl EventSink for NoopEventSink {
    fn is_noop(&self) -> bool {
        true
    }

    async fn emit(&self, _event: SessionEvent) {}
}

/// Stable identifier for a semantic turn activity.
#[derive(Clone, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(transparent)]
pub struct TurnActivityId(pub String);

impl TurnActivityId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    pub fn fresh() -> Self {
        Self(uuid::Uuid::new_v4().to_string())
    }
}

/// App-facing semantic activity emitted during a turn.
///
/// `id` is unique per emitted activity event. `correlation_id` groups related
/// events in the same logical activity, such as code start/completion, tool
/// start/completion, or text deltas from one output block.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TurnActivity {
    pub id: TurnActivityId,
    pub correlation_id: TurnActivityId,
    #[serde(flatten)]
    pub event: TurnEvent,
}

impl TurnActivity {
    pub fn new(correlation_id: TurnActivityId, event: TurnEvent) -> Self {
        Self {
            id: TurnActivityId::fresh(),
            correlation_id,
            event,
        }
    }

    pub fn independent(event: TurnEvent) -> Self {
        let correlation_id = TurnActivityId::fresh();
        Self::new(correlation_id, event)
    }
}

/// App-facing semantic event payload for a turn activity.
///
/// Unlike [`SessionEvent`], these events are stable application signals rather
/// than low-level runtime/debug events. Public streams carry these payloads
/// inside [`TurnActivity`] so every emitted item has identity.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[allow(clippy::large_enum_variant)]
pub enum TurnEvent {
    ModelRequestStarted {
        mode_iteration: usize,
    },
    AssistantProseDelta {
        text: String,
    },
    ReasoningDelta {
        text: String,
    },
    CodeBlockStarted {
        language: String,
        code: String,
    },
    CodeBlockCompleted {
        language: String,
        output: String,
        error: Option<String>,
        success: bool,
        duration_ms: u64,
        tool_call_ids: Vec<String>,
    },
    ToolCallStarted {
        call_id: Option<String>,
        name: String,
        args: serde_json::Value,
    },
    ToolCallCompleted {
        call_id: Option<String>,
        name: String,
        args: serde_json::Value,
        output: crate::ToolCallOutput,
        duration_ms: u64,
    },
    SubmittedValue {
        value: serde_json::Value,
    },
    ToolValue {
        tool_name: String,
        value: serde_json::Value,
    },
    Usage {
        mode_iteration: usize,
        usage: TokenUsage,
        cumulative: TokenUsage,
    },
    ChildUsage {
        session_id: String,
        source: String,
        model: String,
        mode_iteration: usize,
        usage: TokenUsage,
        cumulative: TokenUsage,
    },
    RetryStatus {
        wait_seconds: u64,
        attempt: usize,
        max_attempts: usize,
        reason: String,
    },
    PluginRuntime {
        plugin_id: String,
        event: crate::PluginRuntimeEvent,
    },
    QueuedInputAccepted {
        checkpoint: crate::CheckpointKind,
        inputs: Vec<crate::AcceptedInjectedTurnInput>,
    },
    QueuedMessagesCommitted {
        messages: Vec<crate::PluginMessage>,
        checkpoint: crate::CheckpointKind,
    },
    Error {
        message: String,
    },
}

#[async_trait::async_trait]
pub trait TurnActivitySink: Send + Sync {
    fn is_noop(&self) -> bool {
        false
    }

    async fn emit(&self, activity: TurnActivity);
}

pub struct NoopTurnActivitySink;

#[async_trait::async_trait]
impl TurnActivitySink for NoopTurnActivitySink {
    fn is_noop(&self) -> bool {
        true
    }

    async fn emit(&self, _activity: TurnActivity) {}
}

enum RuntimeStreamEvent {
    Session(SessionEvent),
    Turn(TurnActivity),
}

#[derive(Clone)]
pub struct SessionStoreCreateRequest {
    pub session_id: String,
    pub parent_session_id: Option<String>,
    pub relation: SessionRelation,
    pub policy: SessionPolicy,
}

pub trait SessionStoreFactory: Send + Sync {
    fn create_store(
        &self,
        request: &SessionStoreCreateRequest,
    ) -> Result<Arc<dyn crate::store::RuntimePersistence>, String>;
}

fn debug_rss_kb() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    status.lines().find_map(|line| {
        let value = line.strip_prefix("VmRSS:")?.trim();
        let kb = value.split_whitespace().next()?.parse::<u64>().ok()?;
        Some(kb)
    })
}

/// Generic runtime for CLI or programmatic embedding.
pub struct LashRuntime {
    pub(in crate::runtime) session: Option<Session>,
    pub(in crate::runtime) policy: SessionPolicy,
    pub(in crate::runtime) host: RuntimeHost,
    pub(in crate::runtime) services: RuntimeServices,
    pub(in crate::runtime) state: PersistedSessionState,
    pub(in crate::runtime) runtime_scope_id: Arc<str>,
    pub(in crate::runtime) managed_sessions: Arc<Mutex<HashMap<String, RuntimeHandle>>>,
    pub(in crate::runtime) active_handoff_continuations: Arc<Mutex<HashMap<String, String>>>,
    pub(in crate::runtime) managed_turns: Arc<Mutex<HashMap<String, ManagedSessionTurn>>>,
    /// Mode-owned turn options for this session.
    pub(in crate::runtime) mode_turn_options: crate::ModeTurnOptions,
    /// Session-scoped token cost ledger. Shared by ALL
    /// `RuntimeSessionManager` instances created from this runtime
    /// (both per-turn and async maintenance). Entries accumulate here
    /// and are drained into `state.token_ledger` at turn-commit time.
    pub(in crate::runtime) shared_token_ledger: Arc<std::sync::Mutex<Vec<TokenLedgerEntry>>>,
    pub(in crate::runtime) background_sync_needed: Arc<AtomicBool>,
    /// Seed `PluginMessage`s queued via
    /// `SessionCreateRequest::first_turn_input` for child sessions.
    /// Shared across `RuntimeSessionManager` instances built from this
    /// runtime so the seed remains visible after the parent turn that
    /// created the session has ended.
    pub(in crate::runtime) pending_first_turn_inputs:
        Arc<std::sync::Mutex<HashMap<String, crate::PluginMessage>>>,
    pub(in crate::runtime) turn_phase_probe: Option<Arc<dyn RuntimeTurnPhaseProbe>>,
}
