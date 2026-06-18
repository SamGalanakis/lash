mod assembly;
mod builder;
pub(crate) mod causal;
mod config_ops;
mod effect;
mod environment;
mod error;
mod host;
mod in_memory_store;
mod io;
mod lifecycle;
mod observation;
mod process;
mod process_work_runner;
mod process_worker;
mod queued_work_runner;
mod session_api;
mod session_manager;
mod session_ops;
mod state;
#[cfg(test)]
pub(crate) mod tests;
mod turn_boundary;
mod turn_commit_draft;
mod turn_driver;
mod turn_graph_editor;
mod turn_loop;
mod turn_queue;
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
use crate::plugin::{
    CheckpointHookContext, PrepareTurnRequest, SessionConfigChangedContext, SessionRelation,
};
use crate::sansio::{LlmCallError, Response};
use crate::session_model::{
    Message, MessageRole, Part, PartKind, PruneState, RuntimeSessionPolicy, SessionEvent,
    SessionPolicy, TokenUsage, fresh_message_id, make_error_event, reassign_part_ids, shared_parts,
    transport_stream_events,
};
use crate::{
    CheckpointKind, PersistentRuntimeServices, PluginActionInvokeError, PromptHookContext,
    RuntimeServices, SandboxMessage, Session, SessionCreateRequest, SessionError, SessionHandle,
    SessionSnapshot, SessionStartPoint, ToolCallRecord, TurnFinish, TurnOutcome, TurnStop,
};
use crate::{Effect, TurnMachine};

use host::*;
use session_manager::*;
use turn_boundary::*;
use turn_commit_draft::*;
use turn_driver::*;

pub(crate) const RUNTIME_TURN_LEASE_TTL_MS: u64 = 15 * 60 * 1000;

// `PromptUsage` is re-exported below alongside the runtime's own types.
pub use lash_sansio::PromptUsage;

use assembly::{
    LlmDebugText, LlmDebugToolCall, LlmStreamAccumulator, LlmStreamDebugState, LlmStreamEventLog,
    LlmStreamState, LlmStreamSummary, TurnAssembler,
};
#[cfg(test)]
#[allow(unused_imports)]
use assembly::{classify_output_state, sanitize_assistant_output};
pub use builder::EmbeddedRuntimeBuilder;
pub use causal::process_event_invocation;
pub(crate) use causal::tool_retry_sleep_invocation;
pub(crate) use effect::RuntimeEffectControllerHandle;
pub use effect::{
    AwaitEventKey, AwaitEventWaitIdentity, CausalRef, EffectHost, ExecutionScope,
    ExternalCompletionError, InlineEffectHost, InlineRuntimeEffectController, LlmAttachmentSpec,
    LlmRequestSpec, ProcessCommand, ProcessEffectOutcome, Resolution, ResolveOutcome,
    RuntimeEffectCommand, RuntimeEffectController, RuntimeEffectControllerError,
    RuntimeEffectEnvelope, RuntimeEffectKind, RuntimeEffectLocalExecutor, RuntimeEffectOutcome,
    RuntimeInvocation, RuntimeReplay, RuntimeScope, RuntimeSubject, ScopedEffectController,
    ToolCallLaunch,
};
pub use environment::{ParkedSession, Residency, RuntimeEnvironment, RuntimeEnvironmentBuilder};
pub use error::{DurableStoreFacet, RuntimeError, RuntimeErrorCode};
pub use host::{EmbeddedRuntimeHost, ProcessRuntimeHost, RuntimeHostConfig};
pub use in_memory_store::{InMemorySessionStore, InMemorySessionStoreFactory};
use io::normalize_input_items;
pub use observation::{
    InMemoryLiveReplayStore, InMemoryLiveReplayStoreConfig, LiveReplayGap, LiveReplayGapReason,
    LiveReplayResult, LiveReplayStore, LiveReplayStoreError, LiveReplaySubscribeResult,
    LiveReplaySubscription, RuntimeHandle, RuntimeObservation, SessionCursor, SessionCursorError,
    SessionObservation, SessionObservationEvent, SessionObservationEventPayload,
    SessionObservationSubscription, SessionProcessEventKind, SessionQueueEventKind, SessionResume,
    SessionRevision,
};
#[cfg(any(test, feature = "testing"))]
pub use process::TestLocalProcessRegistry;
pub use process::{
    DefaultProcessCancelAbility, InMemoryProcessExecutionEnvStore, ObservedProcess,
    ObservedProcessEvent, ObservedWorkItem, PROCESS_LEASE_SCHEMA_VERSION, ProcessAwaitOutput,
    ProcessCancelAbility, ProcessCancelAllRequest, ProcessCancelRequest, ProcessCancelSource,
    ProcessCancelSummary, ProcessEngine, ProcessEngineRegistry, ProcessEngineRunContext,
    ProcessEngineRunGuard, ProcessEngineRuntimeContext, ProcessEngineValidationContext,
    ProcessEvent, ProcessEventAppendPlan, ProcessEventAppendRequest, ProcessEventAppendResult,
    ProcessEventSemantics, ProcessEventSemanticsSpec, ProcessEventType, ProcessExecutionContext,
    ProcessExecutionEnvRef, ProcessExecutionEnvSpec, ProcessExecutionEnvStore, ProcessExternalRef,
    ProcessHandleDescriptor, ProcessHandleGrant, ProcessHandleGrantEntry, ProcessHandleSummary,
    ProcessId, ProcessIdentity, ProcessInput, ProcessLease, ProcessLeaseCompletion,
    ProcessLifecycleStatus, ProcessListFilter, ProcessListMode, ProcessOpScope, ProcessOriginator,
    ProcessProvenance, ProcessRecord, ProcessRegistration, ProcessRegistry, ProcessService,
    ProcessSessionDeleteReport, ProcessSpawnProvenance, ProcessStartGrant, ProcessStartOptions,
    ProcessStartRequest, ProcessStatus, ProcessStatusFilter, ProcessTerminalSemantics,
    ProcessTerminalSpec, ProcessTerminalState, ProcessValueSelector, ProcessWake,
    ProcessWakeDedupeKey, ProcessWakeDelivery, ProcessWakeSpec, ProcessWorkObserver,
    ProcessWorkSnapshot, SessionScope, SessionScopeId, UnavailableProcessService, WaitKind,
    WaitState, apply_process_status_projection, current_epoch_ms, epoch_ms_from_system_time,
    load_process_execution_env, materialize_process_event_semantics, persist_process_execution_env,
    prepare_process_event_append, prepare_process_registration, process_event_payload_hash,
    process_signal_event_type, process_signal_name_from_event_type, process_signal_wait_key,
    process_wake_delivery, process_wake_input_from_event_payload, process_wake_turn_cause,
    process_wake_turn_text, require_event_replay, system_time_from_epoch_ms,
    validate_process_signal_name,
};
pub use process_work_runner::{
    InlineProcessRunHandle, ProcessRunHandle, ProcessWorkDriver, ProcessWorkPoke, ProcessWorkRunner,
};
pub use process_worker::{DurableProcessWorker, DurableProcessWorkerConfig};
pub use queued_work_runner::{
    QueuedWorkPoke, QueuedWorkRunHandle, QueuedWorkRunOutcome, QueuedWorkRunRequest,
    QueuedWorkRunner,
};
pub use session_manager::DirectCompletionClient;
pub use state::RuntimeSessionState;
use state::{
    append_session_nodes_to_state, apply_residency_on_load, apply_session_checkpoint,
    apply_session_head, normalize_session_graph, open_agent_frame_in_state,
};
pub use turn_loop::ensure_durable_effect_input;
pub use turn_queue::{
    DeliveryPolicy, MergeKey, QUEUED_WORK_CLAIM_TTL_MS, QueuedCheckpointWork, QueuedTurnWork,
    QueuedWorkBatch, QueuedWorkBatchDraft, QueuedWorkClaim, QueuedWorkClaimBoundary,
    QueuedWorkCompletion, QueuedWorkItem, QueuedWorkPayload, SessionCommand, SessionCommandReceipt,
    SlotPolicy, process_wake_batch_draft,
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
    FinalCommit,
    PostPersistHooks,
}

#[doc(hidden)]
pub trait RuntimeTurnPhaseProbe: Send + Sync {
    fn begin(&self, phase: RuntimeTurnPhase);
    fn end(&self, phase: RuntimeTurnPhase);
    fn begin_named(&self, _phase: &str) {}
    fn end_named(&self, _phase: &str) {}
}

#[doc(hidden)]
#[derive(Clone, Default)]
pub struct RuntimeTurnPhaseProbeSlot {
    probes: Arc<StdMutex<HashMap<crate::SessionScopeId, Arc<dyn RuntimeTurnPhaseProbe>>>>,
}

impl RuntimeTurnPhaseProbeSlot {
    pub fn set_for_session(
        &self,
        session_id: impl Into<String>,
        probe: Arc<dyn RuntimeTurnPhaseProbe>,
    ) {
        self.set_for_scope(&crate::SessionScope::new(session_id), probe);
    }

    pub fn set_for_scope(
        &self,
        scope: &crate::SessionScope,
        probe: Arc<dyn RuntimeTurnPhaseProbe>,
    ) {
        self.probes
            .lock()
            .expect("runtime phase probe slot")
            .insert(scope.id(), probe);
    }

    pub fn get_for_scope(
        &self,
        scope: &crate::SessionScope,
    ) -> Option<Arc<dyn RuntimeTurnPhaseProbe>> {
        let probes = self.probes.lock().expect("runtime phase probe slot");
        probes.get(&scope.id()).cloned().or_else(|| {
            probes
                .get(&crate::SessionScope::new(&scope.session_id).id())
                .cloned()
        })
    }
}

#[doc(hidden)]
pub struct RuntimeNamedPhase {
    probe: Option<Arc<dyn RuntimeTurnPhaseProbe>>,
    phase: &'static str,
}

impl RuntimeNamedPhase {
    pub fn begin(
        probe: Option<Arc<dyn RuntimeTurnPhaseProbe>>,
        phase: &'static str,
    ) -> RuntimeNamedPhase {
        if let Some(probe) = probe.as_ref() {
            probe.begin_named(phase);
        }
        RuntimeNamedPhase { probe, phase }
    }
}

impl Drop for RuntimeNamedPhase {
    fn drop(&mut self) {
        if let Some(probe) = self.probe.as_ref() {
            probe.end_named(self.phase);
        }
    }
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
    /// Per-turn override for protocol-owned turn options.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub protocol_turn_options: Option<crate::ProtocolTurnOptions>,
    /// Optional externally-stable trace turn id. Normal runtime callers leave
    /// this empty and the runtime generates one per outer turn.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trace_turn_id: Option<String>,
    #[serde(skip)]
    pub protocol_extension: Option<ProtocolTurnExtensionHandle>,
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
            protocol_turn_options: None,
            trace_turn_id: None,
            protocol_extension: None,
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

    pub fn with_protocol_turn_options(mut self, options: crate::ProtocolTurnOptions) -> Self {
        self.protocol_turn_options = Some(options);
        self
    }

    pub fn with_trace_turn_id(mut self, trace_turn_id: impl Into<String>) -> Self {
        self.trace_turn_id = Some(trace_turn_id.into());
        self
    }
}

/// Per-turn, in-process side channel of typed plugin inputs.
///
/// This is an `Any`-keyed map of live Rust values handed to plugins for a
/// single turn. It is deliberately **not** serializable: the values never
/// survive a process boundary, so durable effect-host runs explicitly reject a
/// turn that carries any live inputs (see
/// [`LiveTurnInputs::durable_effect_rejection`]). Durable callers must instead
/// encode replayable data in
/// `protocol_turn_options` or persisted plugin state.
#[derive(Clone, Default)]
pub struct LiveTurnInputs {
    inputs: HashMap<&'static str, Arc<dyn Any + Send + Sync>>,
}

impl LiveTurnInputs {
    fn insert<T>(&mut self, plugin_id: &'static str, input: T)
    where
        T: Send + Sync + 'static,
    {
        self.inputs.insert(plugin_id, Arc::new(input));
    }

    fn get<T>(&self, plugin_id: &'static str) -> Option<&T>
    where
        T: 'static,
    {
        self.inputs
            .get(plugin_id)
            .and_then(|input| input.downcast_ref::<T>())
    }

    fn contains(&self, plugin_id: &'static str) -> bool {
        self.inputs.contains_key(plugin_id)
    }

    fn is_empty(&self) -> bool {
        self.inputs.is_empty()
    }

    pub fn plugin_ids(&self) -> Vec<&'static str> {
        self.inputs.keys().copied().collect()
    }

    /// Returns an error when live per-turn inputs would make a durable effect
    /// host replay depend on process-local values.
    pub(crate) fn durable_effect_rejection(&self) -> Result<(), RuntimeError> {
        if self.is_empty() {
            return Ok(());
        }
        Err(RuntimeError::new(
            RuntimeErrorCode::DurableEffectLivePluginInput,
            "durable effect hosts do not support live TurnContext plugin inputs; encode replayable data in protocol_turn_options or persisted plugin state",
        ))
    }
}

#[derive(Clone, Default)]
pub struct TurnContext {
    plugin_inputs: LiveTurnInputs,
    provider: Option<crate::ProviderHandle>,
    model: Option<crate::ModelSpec>,
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
        self.plugin_inputs.insert(plugin_id, input);
    }

    pub fn set_provider(&mut self, provider: crate::ProviderHandle) {
        self.provider = Some(provider);
    }

    pub fn provider(&self) -> Option<&crate::ProviderHandle> {
        self.provider.as_ref()
    }

    pub fn set_model(&mut self, model: crate::ModelSpec) {
        self.model = Some(model);
    }

    pub fn model_spec(&self) -> Option<&crate::ModelSpec> {
        self.model.as_ref()
    }

    pub fn plugin_input<T>(&self, plugin_id: &'static str) -> Option<&T>
    where
        T: 'static,
    {
        self.plugin_inputs.get(plugin_id)
    }

    pub fn has_plugin_input(&self, plugin_id: &'static str) -> bool {
        self.plugin_inputs.contains(plugin_id)
    }

    pub fn has_live_plugin_inputs(&self) -> bool {
        !self.plugin_inputs.is_empty()
    }

    pub fn live_plugin_input_ids(&self) -> Vec<&'static str> {
        self.plugin_inputs.plugin_ids()
    }

    /// Live plugin inputs for this turn. The durable boundary inspects this to
    /// reject turns carrying non-serializable live state.
    pub(crate) fn live_plugin_inputs(&self) -> &LiveTurnInputs {
        &self.plugin_inputs
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
            .field("plugin_inputs", &self.plugin_inputs.plugin_ids())
            .field("has_provider", &self.provider.is_some())
            .field("has_model", &self.model.is_some())
            .field("has_prompt_layer", &(!self.prompt.is_empty()))
            .finish()
    }
}

#[derive(Clone)]
pub struct ProtocolTurnExtensionHandle(Arc<dyn ProtocolTurnExtension>);

impl ProtocolTurnExtensionHandle {
    pub fn new(extension: impl ProtocolTurnExtension + 'static) -> Self {
        Self(Arc::new(extension))
    }

    pub fn as_any(&self) -> &dyn Any {
        self.0.as_any()
    }

    pub fn prompt_contributions(&self) -> Vec<crate::PromptContribution> {
        self.0.prompt_contributions()
    }
}

impl fmt::Debug for ProtocolTurnExtensionHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("ProtocolTurnExtensionHandle(..)")
    }
}

pub trait ProtocolTurnExtension: Send + Sync {
    fn as_any(&self) -> &dyn Any;

    fn prompt_contributions(&self) -> Vec<crate::PromptContribution> {
        Vec::new()
    }
}

#[derive(Clone)]
pub struct ProtocolSessionExtensionHandle(Arc<dyn ProtocolSessionExtension>);

impl ProtocolSessionExtensionHandle {
    pub fn new(extension: impl ProtocolSessionExtension + 'static) -> Self {
        Self(Arc::new(extension))
    }

    pub fn as_any(&self) -> &dyn Any {
        self.0.as_any()
    }
}

impl fmt::Debug for ProtocolSessionExtensionHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("ProtocolSessionExtensionHandle(..)")
    }
}

pub trait ProtocolSessionExtension: Send + Sync {
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

/// Code execution output observed during a turn.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CodeOutputRecord {
    pub output: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// High-level execution summary for a completed turn.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ExecutionSummary {
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
    pub state: SessionSnapshot,
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

/// Result of driving one logical host turn through any AgentFrame switches.
///
/// A frame switch is an internal runtime continuation, similar to compaction
/// from a host's perspective. Callers that need a final answer can use
/// [`LashRuntime::stream_turn_with_agent_frames`] and inspect `final_turn()`.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct AgentFrameRun {
    pub turns: Vec<AssembledTurn>,
}

impl AgentFrameRun {
    pub fn final_turn(&self) -> Option<&AssembledTurn> {
        self.turns.last()
    }

    pub fn into_final_turn(mut self) -> Option<AssembledTurn> {
        self.turns.pop()
    }

    pub fn frame_switch_count(&self) -> usize {
        self.turns
            .iter()
            .filter(|turn| matches!(turn.outcome, crate::TurnOutcome::AgentFrameSwitch { .. }))
            .count()
    }
}

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

/// Host application sink for low-level streaming runtime events.
/// `SessionEvent` is protocol-specific preview/progress data.
#[async_trait::async_trait]
pub trait EventSink: Send + Sync {
    fn is_noop(&self) -> bool {
        false
    }

    async fn emit(&self, event: SessionEvent);
}

/// No-op sink useful for callers that only care about final state.
pub struct NoopEventSink;

/// Static no-op event sink for callers that need a `&dyn EventSink` default.
pub static NOOP_EVENT_SINK: NoopEventSink = NoopEventSink;

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
    QueuedWorkStarted {
        boundary: crate::QueuedWorkClaimBoundary,
        batch_ids: Vec<String>,
        causes: Vec<crate::TurnCause>,
    },
    ModelRequestStarted {
        protocol_iteration: usize,
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
        #[serde(default, skip_serializing_if = "Option::is_none")]
        graph_key: Option<String>,
    },
    CodeBlockCompleted {
        language: String,
        output: String,
        error: Option<String>,
        success: bool,
        duration_ms: u64,
        tool_call_ids: Vec<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        graph_key: Option<String>,
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
        protocol_iteration: usize,
        usage: TokenUsage,
        cumulative: TokenUsage,
    },
    ChildUsage {
        session_id: String,
        source: String,
        model: String,
        protocol_iteration: usize,
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

/// Static no-op turn-activity sink for callers that need a `&dyn TurnActivitySink` default.
pub static NOOP_TURN_ACTIVITY_SINK: NoopTurnActivitySink = NoopTurnActivitySink;

#[async_trait::async_trait]
impl TurnActivitySink for NoopTurnActivitySink {
    fn is_noop(&self) -> bool {
        true
    }

    async fn emit(&self, _activity: TurnActivity) {}
}

/// Optional sinks and scoped effect controller passed to one of [`LashRuntime`]'s
/// turn-driving entry points (`stream_turn`,
/// `stream_turn_with_agent_frames`).
///
/// Construct via [`TurnOptions::new`] and chain `with_*` builders. Event sinks
/// default to no-op sinks. Execution scope is explicit and required at every
/// runtime boundary that can execute nondeterministic work.
pub struct TurnOptions<'a> {
    events: Option<&'a dyn EventSink>,
    turn_events: Option<&'a dyn TurnActivitySink>,
    scoped_effect_controller: ScopedEffectController<'a>,
    cancel: CancellationToken,
}

impl<'a> TurnOptions<'a> {
    pub fn new(
        cancel: CancellationToken,
        scoped_effect_controller: ScopedEffectController<'a>,
    ) -> Self {
        Self {
            events: None,
            turn_events: None,
            scoped_effect_controller,
            cancel,
        }
    }

    pub fn with_events(mut self, events: &'a dyn EventSink) -> Self {
        self.events = Some(events);
        self
    }

    pub fn with_turn_events(mut self, turn_events: &'a dyn TurnActivitySink) -> Self {
        self.turn_events = Some(turn_events);
        self
    }

    pub(crate) fn events_or_noop(&self) -> &'a dyn EventSink {
        self.events.unwrap_or(&NOOP_EVENT_SINK)
    }

    pub(crate) fn turn_events_or_noop(&self) -> &'a dyn TurnActivitySink {
        self.turn_events.unwrap_or(&NOOP_TURN_ACTIVITY_SINK)
    }

    pub(crate) fn execution_scope_id(&self) -> &str {
        self.scoped_effect_controller.scope_id()
    }

    pub(crate) fn scoped_effect_controller(&self) -> ScopedEffectController<'a> {
        self.scoped_effect_controller.clone()
    }
}

enum RuntimeStreamEvent {
    Session(SessionEvent),
    Turn(TurnActivity),
}

#[derive(Clone)]
pub struct SessionStoreCreateRequest {
    pub session_id: String,
    pub relation: SessionRelation,
    pub policy: SessionPolicy,
}

impl SessionStoreCreateRequest {
    pub fn parent_session_id(&self) -> Option<&str> {
        self.relation.parent_session_id()
    }
}

#[async_trait::async_trait]
pub trait SessionStoreFactory: Send + Sync {
    /// Durability tier the stores produced by this factory provide; defaults to
    /// [`DurabilityTier::Inline`].
    fn durability_tier(&self) -> crate::DurabilityTier {
        crate::DurabilityTier::Inline
    }

    async fn create_store(
        &self,
        request: &SessionStoreCreateRequest,
    ) -> Result<Arc<dyn crate::store::RuntimePersistence>, String>;

    async fn open_existing_store(
        &self,
        _request: &SessionStoreCreateRequest,
    ) -> Result<Option<Arc<dyn crate::store::RuntimePersistence>>, String> {
        Ok(None)
    }

    async fn delete_session(&self, session_id: &str) -> Result<(), String>;
}

/// Generic runtime for CLI or programmatic embedding.
pub struct LashRuntime {
    pub(in crate::runtime) session: Option<Session>,
    pub(in crate::runtime) policy: SessionPolicy,
    pub(in crate::runtime) host: RuntimeHost,
    pub(in crate::runtime) services: RuntimeServices,
    pub(in crate::runtime) state: RuntimeSessionState,
    pub(in crate::runtime) runtime_scope_id: Arc<str>,
    pub(in crate::runtime) managed_sessions: Arc<Mutex<HashMap<String, RuntimeHandle>>>,
    pub(in crate::runtime) managed_turns: Arc<Mutex<HashMap<String, ManagedSessionTurn>>>,
    /// Protocol-owned turn options for this session.
    pub(in crate::runtime) protocol_turn_options: crate::ProtocolTurnOptions,
    /// Session-scoped token cost ledger. Shared by ALL
    /// `RuntimeSessionServices` instances created from this runtime
    /// (both per-turn and async maintenance). Entries accumulate here
    /// and are drained into `state.token_ledger` at turn-commit time.
    pub(in crate::runtime) shared_token_ledger: Arc<std::sync::Mutex<Vec<TokenLedgerEntry>>>,
    pub(in crate::runtime) process_sync_needed: Arc<AtomicBool>,
    pub(in crate::runtime) turn_phase_probe: Option<Arc<dyn RuntimeTurnPhaseProbe>>,
    /// Resident-graph policy chosen by the host. Controls whether
    /// [`LashRuntime::refresh_session_graph_from_store`] reloads the full
    /// graph or just the active path, matching the trimming behavior set at
    /// load time via [`apply_residency_on_load`](crate::runtime::apply_residency_on_load).
    pub(in crate::runtime) residency: Residency,
}
