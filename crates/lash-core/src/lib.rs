pub mod attachments;
pub mod chronological;
pub mod direct;
pub mod lashlang_bridge;
pub mod llm;
mod mode;
mod model;
pub mod monitor;
pub mod plugin;
mod plugin_stack;
pub mod provider;
pub mod runtime;
pub mod runtime_controls;
pub mod search;
pub mod session;
pub mod session_graph;
pub mod session_model;
mod stable_hash;
pub mod standard_context_approach;
pub mod store;
#[cfg(any(test, feature = "testing"))]
pub mod testing;
pub mod tool_dispatch;
mod tool_provider;
pub mod tool_registry;
mod tool_result;
mod tool_schema;
mod trace;

pub use lash_sansio::sansio;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const SANSIO_VERSION: &str = lash_sansio::VERSION;

// Re-exports
pub use attachments::{
    AttachmentStore, AttachmentStoreError, AttachmentStorePersistence, FileAttachmentStore,
    InMemoryAttachmentStore, SessionScopedAttachmentStore, StoredAttachment,
};
pub use chronological::{
    BorrowedChronologicalEntry, BorrowedChronologicalMessage, BorrowedChronologicalPayload,
    ChronologicalEntry, ChronologicalPayload, ChronologicalProjection, visit_turn_view,
};
pub use direct::{
    DirectJsonSchema, DirectLlmClient, DirectLlmError, DirectMessage, DirectOutputSpec, DirectPart,
    DirectRequest, DirectRole,
};
pub use lash_sansio::llm::types::{
    GenerationOptions, LlmOutputPart, LlmRequest, LlmResponse, LlmTerminalReason,
};
pub use lash_sansio::{
    AcceptedInjectedTurnInput, AttachmentCreateMeta, AttachmentId, AttachmentMeta, AttachmentRef,
    BaseRenderCache, CheckpointKind, CompactToolContract, EffectId, ErrorEnvelope, ExecImage,
    ExecResponse, ExecutionMode, ImageMediaType, LlmCallError, MediaType, Message, MessageOrigin,
    MessageRole, MessageSequence, ModelToolReturn, ModelToolReturnPart, Part, PartKind,
    PluginMessage, PluginRuntimeEvent, PreparedPrompt, PromptBuildInput, PromptBuiltin,
    PromptContext, PromptContribution, PromptContributionGate, PromptContributionSet,
    PromptFingerprint, PromptLayer, PromptSlot, PromptSlotLayer, PromptTemplate,
    PromptTemplateEntry, PromptTemplateSection, PruneState, RenderedPrompt, ResolvedPromptLayer,
    Response, SchemaProjectionOverride, SessionEvent, TextProjectionMetadata, TokenUsage,
    ToolActivation, ToolArgumentProjectionPolicy, ToolAvailability, ToolAvailabilityConfig,
    ToolCallOutcome, ToolCallOutput, ToolCallRecord, ToolCallStatus, ToolCancellation,
    ToolContract, ToolControl, ToolDefinition, ToolDiscoveryMetadata, ToolExecutionMode,
    ToolFailure, ToolFailureClass, ToolFailureSource, ToolId, ToolManifest, ToolOutputContract,
    ToolRetryDisposition, ToolRetryPolicy, ToolSurface, ToolSurfaceBuildInput, ToolSurfaceEntry,
    ToolSurfaceOverride, ToolValue, TurnFinish, TurnLimitFinalMessage, TurnOutcome, TurnStop,
    append_assistant_text_part, build_prompt, build_tool_surface, build_turn,
    default_execution_mode, default_prompt_template, execution_mode_supported, head_tail_truncate,
    messages_are_prompt_resume_safe, normalized_response_parts, prompt_template_fingerprint,
    prompt_text_fingerprint, prompt_tool_names_fingerprint, reasoning_part, resolve_prompt_layers,
    shared_parts,
};
pub use mode::ModeBuildInput;
pub use standard_context_approach::{
    ObservationalMemoryConfig, RollingHistoryConfig, StandardContextApproach,
    StandardContextApproachKind,
};
pub use tool_registry::{
    ReconfigureError, ToolRegistry, ToolSourceHandle, ToolState, ToolStateEntry,
};
pub use tool_result::ToolResult;
pub use tool_schema::LashSchema;
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ModeTurnOptions {
    pub mode_id: ExecutionMode,
    #[serde(default)]
    pub payload: serde_json::Value,
}

impl Default for ModeTurnOptions {
    fn default() -> Self {
        Self::empty(ExecutionMode::standard())
    }
}

impl ModeTurnOptions {
    pub fn empty(mode_id: ExecutionMode) -> Self {
        Self {
            mode_id,
            payload: serde_json::Value::Object(serde_json::Map::new()),
        }
    }

    pub fn typed<T>(mode_id: ExecutionMode, value: T) -> Result<Self, serde_json::Error>
    where
        T: serde::Serialize,
    {
        Ok(Self {
            mode_id,
            payload: serde_json::to_value(value)?,
        })
    }

    pub fn decode<T>(&self, expected_mode: &ExecutionMode) -> Result<Option<T>, serde_json::Error>
    where
        T: serde::de::DeserializeOwned,
    {
        if &self.mode_id != expected_mode {
            return Ok(None);
        }
        serde_json::from_value(self.payload.clone()).map(Some)
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct HostModeProtocol;

impl lash_sansio::ModeProtocol for HostModeProtocol {
    type Event = crate::session_model::ModeEvent;
    type Termination = ModeTurnOptions;
    type DriverState = serde_json::Value;
}

pub type Effect = lash_sansio::Effect<HostModeProtocol>;
pub type DriverAction = lash_sansio::DriverAction<HostModeProtocol>;
pub type DriverContextView<'a> = lash_sansio::DriverContextView<'a, HostModeProtocol>;
pub type ModeConfig = lash_sansio::ModeConfig<HostModeProtocol>;
pub type ModePreamble = lash_sansio::ModePreamble<HostModeProtocol>;
pub type ProjectorContext<'a> = lash_sansio::ProjectorContext<'a, HostModeProtocol>;
pub type PreparedTurnMachine = lash_sansio::PreparedTurnMachine<HostModeProtocol>;
pub type SansIoTurnInput = lash_sansio::SansIoTurnInput<HostModeProtocol>;
pub type TurnMachine = lash_sansio::TurnMachine<HostModeProtocol>;
pub type TurnMachineConfig = lash_sansio::TurnMachineConfig<HostModeProtocol>;
#[cfg(feature = "otel-trace")]
pub use lash_trace::otel::{OtelTraceOptions, OtelTraceSink};
pub use lash_trace::{
    JsonlTraceSink, TraceAttachment, TraceContentBlock, TraceContext, TraceError, TraceEvent,
    TraceLevel, TraceLlmMessage, TraceLlmRequest, TraceLlmResponse, TracePromptComponent,
    TraceProviderStreamEvent, TraceRecord, TraceRuntimeStreamEvent, TraceSink, TraceSinkError,
    TraceTokenUsage, TraceToolSpec,
};
pub use llm::transport::{LlmTransportError, ProviderFailure, ProviderFailureKind};
pub use model::{ModelLimits, ModelSpec};
pub use monitor::{
    MAX_MONITOR_TIMEOUT_MS, MonitorArmOn, MonitorRunState, MonitorSnapshot, MonitorSpec,
    MonitorStatus, MonitorWakePolicy,
};
pub use plugin::{
    AppendSessionNodesRequest, AppendSessionNodesResult, AssistantResponseHookContext,
    AssistantResponseTransform, AssistantStreamHookContext, AssistantStreamTransform,
    CheckpointHookContext, DirectCompletion, DirectLlmCompletion, HistoryError,
    HistoryRegistrations, HistoryRewriteMetadata, HistoryRewriter, HistoryState,
    ModeBeforeLlmCallContext, ModeExtras, ModeLlmCallAction, MonitorEmptyArgs,
    MonitorRegisterSpecsOp, MonitorRegistrations, MonitorStartOp, MonitorStatusOp, MonitorStopOp,
    OwnedMonitorSpec, PersistentRuntimeServices, PluginAction, PluginActionContext,
    PluginActionDef, PluginActionFailure, PluginActionInvokeError, PluginActionKind,
    PluginDirective, PluginError, PluginFactory, PluginHost, PluginLifecycleEvent,
    PluginLifecycleEventHook, PluginOwned, PluginRegistrar, PluginSession, PluginSessionContext,
    PluginSessionSnapshot, PluginSnapshotArtifact, PluginSnapshotEntry, PluginSnapshotMeta,
    PluginSpec, PluginSpecFactory, PromptHookContext, RegisterSpecsArgs, RewriteContext,
    RewriteTrigger, RuntimeServices, SessionAppendNode, SessionConfigChangedContext,
    SessionContextSurface, SessionCreateRequest, SessionHandle, SessionParam, SessionPlugin,
    SessionPluginMode, SessionReadView, SessionRelation, SessionSnapshot, SessionStartPoint,
    SessionStateChangedContext, SessionToolAccess, SnapshotReader, SnapshotWriter,
    StandardCreateExtras, StartMonitorArgs, StopMonitorArgs, SubagentSessionContext,
    ToolDiscoveryContext, ToolDiscoveryContribution, ToolDiscoveryContributor,
    ToolDiscoveryToolContribution, ToolOutputBudgetConfig, ToolOutputBudgetMode,
    ToolOutputBudgetPluginFactory, ToolResultProjectionContext, ToolResultProjector,
    ToolSurfaceContribution, TurnContextTransform, TurnHookContext, TurnResultHookContext,
    TurnResultSummary, TurnTransformContext, plugin_action_def,
};
pub use plugin_stack::PluginStack;
pub use provider::{
    CacheRetention, LlmTimeouts, ProviderComponents, ProviderFactory, ProviderHandle,
    ProviderModelPolicy, ProviderOptions, ProviderRegistry, ProviderSpec, ProviderState,
    ProviderThinkingPolicy, ProviderTransport, RequestTimeout, StaticModelPolicy,
    VariantRequestConfig, build_provider, provider_factory, register_provider_factory,
};
pub use runtime::{
    AssembledTurn, AssistantOutput, CodeOutputRecord, DirectCompletionClient, DirectRequestSpec,
    DurableProcessWorker, DurableProcessWorkerConfig, EffectInvocationMetadata, EffectOrigin,
    EmbeddedRuntimeBuilder, EmbeddedRuntimeHost, EventSink, ExecutionSummary, FollowedTurn,
    InlineRuntimeEffectController, InputItem, LashRuntime, LlmAttachmentSpec, LlmRequestSpec,
    LocalProcessRegistry, ModeSessionExtension, ModeSessionExtensionHandle, ModeTurnExtension,
    ModeTurnExtensionHandle, NoopEventSink, NoopTurnActivitySink, OutputState, ParkedSession,
    PersistedSessionSnapshot, ProcessAwaitOutput, ProcessAwaitRequest, ProcessCancelRequest,
    ProcessCleanupRequest, ProcessCommand, ProcessEffectOutcome, ProcessEvent,
    ProcessEventAppendRequest, ProcessEventSemantics, ProcessEventSemanticsSpec, ProcessEventType,
    ProcessExecutionContext, ProcessExternalRef, ProcessHandleDescriptor, ProcessHandleGrant,
    ProcessHandleGrantEntry, ProcessId, ProcessInput, ProcessListRequest, ProcessRecord,
    ProcessRegistration, ProcessRegistry, ProcessRequestScope, ProcessRuntimeHost,
    ProcessStartGrant, ProcessStartRequest, ProcessTerminalSemantics, ProcessTerminalSpec,
    ProcessTerminalState, ProcessTransferRequest, ProcessValueSelector, ProcessWake,
    ProcessWakeDedupeKey, ProcessWakeDelivery, ProcessWakeSpec, PromptUsage, Residency,
    RuntimeCoreConfig, RuntimeEffectCommand, RuntimeEffectController, RuntimeEffectControllerError,
    RuntimeEffectControllerScope, RuntimeEffectEnvelope, RuntimeEffectKind,
    RuntimeEffectLocalExecutor, RuntimeEffectOutcome, RuntimeEnvironment,
    RuntimeEnvironmentBuilder, RuntimeError, RuntimeErrorCode, RuntimeHandle, RuntimeObservation,
    RuntimeSessionState, SessionStateEnvelope, SessionStoreCreateRequest, SessionStoreFactory,
    SessionUsageReport, TerminationPolicy, TokenLedgerEntry, TurnActivity, TurnActivityId,
    TurnActivitySink, TurnContext, TurnEvent, TurnInput, TurnIssue, TurnOptions, UsageReportRow,
    UsageTotals, current_epoch_ms, diff_token_ledger, diff_usage_reports,
    epoch_ms_from_system_time, lashlang_process_event_types, materialize_process_event_semantics,
    prepare_process_registration, process_event_payload_hash, process_wake_delivery,
    require_event_idempotency, system_time_from_epoch_ms,
};
pub use runtime_controls::BuiltinProcessControlsPluginFactory;
pub use schemars::JsonSchema;
pub use session::{
    ExecRequest, InjectedTurnInput, ModeExecutionContext, ModeToolBatchItem, ModeToolReply,
    Session, SessionError, TurnInjectionBridge, TurnInputInjectionBridge,
};
pub use session_graph::{
    PersistedSessionConfig, PersistedTurnState, SessionGraph, SessionMessageTreeNode,
    SessionNodePayload, SessionNodeRecord,
};
pub use session_model::context::PreparedContext;
pub use session_model::{ConversationRecord, ModeEvent, SessionEventRecord, ToolEvent};
pub use session_model::{SessionPolicy, SessionSpec};
pub use store::{
    AttachmentIntent, AttachmentManifest, AttachmentManifestEntry, BlobRef, GcReport,
    GraphCommitDelta, HydratedSessionCheckpoint, PersistedSessionRead,
    RUNTIME_EFFECT_JOURNAL_SCHEMA_VERSION, RUNTIME_TURN_CHECKPOINT_SCHEMA_VERSION,
    RUNTIME_TURN_LEASE_SCHEMA_VERSION, RuntimeCommit, RuntimeCommitResult,
    RuntimeEffectJournalRecord, RuntimePersistence, RuntimeTurnCheckpoint, RuntimeTurnCompletion,
    RuntimeTurnLease, RuntimeTurnMachineConfigSnapshot, SessionCheckpoint, SessionHead,
    SessionHeadMeta, SessionMeta, SessionPickerInfo, SessionReadScope, StoreError, VacuumReport,
    ensure_supported_schema_version, load_persisted_session_state,
    load_persisted_session_state_active_path, refresh_persisted_session_state,
    runtime_turn_checkpoint_hash,
};
pub use tool_provider::{
    PreparedToolCall, ProgressSender, SandboxMessage, ToolCall, ToolContext, ToolPrepareCall,
    ToolPrepareContext, ToolProcessControl, ToolProvider, ToolSessionControl, ToolSessionModel,
};
