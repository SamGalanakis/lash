pub mod attachments;
pub mod chronological;
pub mod direct;
pub mod host_events;
pub mod lashlang_bridge;
pub mod llm;
mod model;
pub mod plugin;
mod plugin_stack;
mod protocol_build;
pub mod provider;
pub mod runtime;
pub mod search;
pub mod session;
pub mod session_graph;
pub mod session_model;
mod stable_hash;
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
    AttachmentStore, AttachmentStoreError, AttachmentStorePersistence, InMemoryAttachmentStore,
    SessionScopedAttachmentStore, StoredAttachment,
};
pub use chronological::{
    BorrowedChronologicalEntry, BorrowedChronologicalMessage, BorrowedChronologicalPayload,
    ChronologicalEntry, ChronologicalPayload, ChronologicalProjection, visit_turn_view,
};
pub use direct::{
    DirectJsonSchema, DirectLlmClient, DirectLlmError, DirectMessage, DirectOutputSpec, DirectPart,
    DirectRequest, DirectRole,
};
pub use host_events::{
    HostEvent, HostEventCatalog, HostEventEmitReport, HostEventKey, SessionTriggerInstallReport,
};
pub use lash_sansio::llm::types::{
    GenerationOptions, LlmOutputPart, LlmRequest, LlmResponse, LlmTerminalReason,
};
pub use lash_sansio::{
    AcceptedInjectedTurnInput, AttachmentCreateMeta, AttachmentId, AttachmentMeta, AttachmentRef,
    BaseRenderCache, CheckpointDelivery, CheckpointKind, CompactToolContract, EffectId,
    ErrorEnvelope, ExecImage, ExecResponse, ImageMediaType, LlmCallError, MediaType, Message,
    MessageOrigin, MessageRole, MessageSequence, ModelToolReturn, ModelToolReturnPart, Part,
    PartKind, PluginMessage, PluginRuntimeEvent, PreparedPrompt, PromptBuildInput, PromptBuiltin,
    PromptContext, PromptContribution, PromptContributionGate, PromptContributionSet,
    PromptFingerprint, PromptLayer, PromptSlot, PromptSlotLayer, PromptTemplate,
    PromptTemplateEntry, PromptTemplateSection, PruneState, RenderedPrompt, ResolvedPromptLayer,
    Response, SchemaProjectionOverride, SessionEvent, TextProjectionMetadata, TokenUsage,
    ToolActivation, ToolArgumentProjectionPolicy, ToolAvailability, ToolAvailabilityConfig,
    ToolCallOutcome, ToolCallOutput, ToolCallRecord, ToolCallStatus, ToolCancellation,
    ToolContract, ToolControl, ToolDefinition, ToolDiscoveryMetadata, ToolFailure,
    ToolFailureClass, ToolFailureSource, ToolId, ToolManifest, ToolOutputContract,
    ToolRetryDisposition, ToolRetryPolicy, ToolScheduling, ToolSurface, ToolSurfaceBuildInput,
    ToolSurfaceEntry, ToolSurfaceOverride, ToolValue, TurnCause, TurnFinish, TurnLimitFinalMessage,
    TurnOutcome, TurnStop, append_assistant_text_part, build_prompt, build_tool_surface,
    build_turn, default_prompt_template, head_tail_truncate, messages_are_prompt_resume_safe,
    normalized_response_parts, prompt_template_fingerprint, prompt_text_fingerprint,
    prompt_tool_names_fingerprint, reasoning_part, render_turn_causes_prompt,
    resolve_prompt_layers, shared_parts,
};
pub use protocol_build::ProtocolBuildInput;
pub use tool_registry::{
    ReconfigureError, ToolRegistry, ToolSourceHandle, ToolState, ToolStateEntry,
};
pub use tool_result::ToolResult;
pub use tool_schema::LashSchema;
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ProtocolTurnOptions {
    #[serde(default = "empty_protocol_turn_payload")]
    pub payload: serde_json::Value,
}

fn empty_protocol_turn_payload() -> serde_json::Value {
    serde_json::Value::Object(serde_json::Map::new())
}

impl Default for ProtocolTurnOptions {
    fn default() -> Self {
        Self::empty()
    }
}

impl ProtocolTurnOptions {
    pub fn empty() -> Self {
        Self {
            payload: serde_json::Value::Object(serde_json::Map::new()),
        }
    }

    pub fn is_empty(&self) -> bool {
        match &self.payload {
            serde_json::Value::Object(map) => map.is_empty(),
            _ => false,
        }
    }

    pub fn typed<T>(value: T) -> Result<Self, serde_json::Error>
    where
        T: serde::Serialize,
    {
        Ok(Self {
            payload: serde_json::to_value(value)?,
        })
    }

    pub fn decode<T>(&self) -> Result<T, serde_json::Error>
    where
        T: serde::de::DeserializeOwned,
    {
        serde_json::from_value(self.payload.clone())
    }
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct ProtocolDriverState {
    pub plugin_id: String,
    pub payload: serde_json::Value,
}

impl ProtocolDriverState {
    pub fn new(plugin_id: impl Into<String>, payload: serde_json::Value) -> Self {
        Self {
            plugin_id: plugin_id.into(),
            payload,
        }
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct HostTurnProtocol;

impl lash_sansio::TurnProtocol for HostTurnProtocol {
    type Event = crate::session_model::ProtocolEvent;
    type Termination = ProtocolTurnOptions;
    type DriverState = ProtocolDriverState;
}

pub type Effect = lash_sansio::Effect<HostTurnProtocol>;
pub type DriverAction = lash_sansio::DriverAction<HostTurnProtocol>;
pub type DriverContextView<'a> = lash_sansio::DriverContextView<'a, HostTurnProtocol>;
pub type TurnDriverConfig = lash_sansio::TurnDriverConfig<HostTurnProtocol>;
pub type TurnDriverPreamble = lash_sansio::TurnDriverPreamble<HostTurnProtocol>;
pub type ProjectorContext<'a> = lash_sansio::ProjectorContext<'a, HostTurnProtocol>;
pub type PreparedTurnMachine = lash_sansio::PreparedTurnMachine<HostTurnProtocol>;
pub type SansIoTurnInput = lash_sansio::SansIoTurnInput<HostTurnProtocol>;
pub type TurnMachine = lash_sansio::TurnMachine<HostTurnProtocol>;
pub type TurnMachineConfig = lash_sansio::TurnMachineConfig<HostTurnProtocol>;
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
pub use plugin::{
    AppendSessionNodesRequest, AppendSessionNodesResult, AssistantResponseHookContext,
    AssistantResponseTransform, AssistantStreamHookContext, AssistantStreamTransform,
    CheckpointHookContext, DirectCompletion, DirectLlmCompletion, HistoryError,
    HistoryRegistrations, HistoryRewriteMetadata, HistoryRewriter, HistoryState,
    HostEventRegistrations, PersistentRuntimeServices, PluginAction, PluginActionContext,
    PluginActionDef, PluginActionFailure, PluginActionInvokeError, PluginActionKind,
    PluginDirective, PluginError, PluginFactory, PluginHost, PluginLifecycleEvent,
    PluginLifecycleEventHook, PluginOptions, PluginOwned, PluginRegistrar, PluginSession,
    PluginSessionContext, PluginSessionSnapshot, PluginSnapshotArtifact, PluginSnapshotEntry,
    PluginSnapshotMeta, PluginSpec, PluginSpecFactory, PromptHookContext,
    ProtocolBeforeLlmCallContext, ProtocolLlmCallAction, RewriteContext, RewriteTrigger,
    RuntimeServices, SessionAppendNode, SessionConfigChangedContext, SessionContextSurface,
    SessionCreateRequest, SessionHandle, SessionParam, SessionPlugin, SessionPluginSource,
    SessionReadView, SessionRelation, SessionSnapshot, SessionStartPoint,
    SessionStateChangedContext, SessionToolAccess, SnapshotReader, SnapshotWriter,
    SubagentSessionContext, ToolDiscoveryContext, ToolDiscoveryContribution,
    ToolDiscoveryContributor, ToolDiscoveryToolContribution, ToolResultProjectionContext,
    ToolResultProjector, ToolSurfaceContribution, TurnContextTransform, TurnHookContext,
    TurnResultHookContext, TurnResultSummary, TurnTransformContext, plugin_action_def,
};
pub use plugin_stack::PluginStack;
pub use provider::{
    CacheRetention, LlmTimeouts, ProviderComponents, ProviderFactory, ProviderHandle,
    ProviderModelPolicy, ProviderOptions, ProviderRegistry, ProviderSpec, ProviderState,
    ProviderThinkingPolicy, ProviderTransport, RequestTimeout, StaticModelPolicy, build_provider,
    provider_factory, register_provider_factory,
};
#[cfg(any(test, feature = "testing"))]
pub use runtime::TestLocalProcessRegistry;
pub use runtime::{
    AssembledTurn, AssistantOutput, CodeOutputRecord, DirectCompletionClient, DirectRequestSpec,
    DurableProcessWorker, DurableProcessWorkerConfig, EffectInvocationMetadata, EffectOrigin,
    EmbeddedRuntimeBuilder, EmbeddedRuntimeHost, EventSink, ExecutionSummary, FollowedTurn,
    InlineRuntimeEffectController, InputItem, LashRuntime, LlmAttachmentSpec, LlmRequestSpec,
    NoopEventSink, NoopTurnActivitySink, OutputState, ParkedSession, PersistedSessionSnapshot,
    ProcessAwaitOutput, ProcessCommand, ProcessEffectOutcome, ProcessEvent,
    ProcessEventAppendRequest, ProcessEventSemantics, ProcessEventSemanticsSpec, ProcessEventType,
    ProcessExecutionContext, ProcessExternalRef, ProcessHandleDescriptor, ProcessHandleGrant,
    ProcessHandleGrantEntry, ProcessId, ProcessInput, ProcessOpScope, ProcessRecord,
    ProcessRegistration, ProcessRegistry, ProcessRuntimeHost, ProcessScope, ProcessScopeId,
    ProcessService, ProcessSessionDeleteReport, ProcessStartGrant, ProcessStartOptions,
    ProcessTerminalSemantics, ProcessTerminalSpec, ProcessTerminalState, ProcessValueSelector,
    ProcessWake, ProcessWakeDedupeKey, ProcessWakeDelivery, ProcessWakeSpec, PromptUsage,
    ProtocolSessionExtension, ProtocolSessionExtensionHandle, ProtocolTurnExtension,
    ProtocolTurnExtensionHandle, Residency, RuntimeCoreConfig, RuntimeEffectCommand,
    RuntimeEffectController, RuntimeEffectControllerError, RuntimeEffectControllerScope,
    RuntimeEffectEnvelope, RuntimeEffectKind, RuntimeEffectLocalExecutor, RuntimeEffectOutcome,
    RuntimeEnvironment, RuntimeEnvironmentBuilder, RuntimeError, RuntimeErrorCode, RuntimeHandle,
    RuntimeObservation, RuntimeSessionState, SessionStateEnvelope, SessionStoreCreateRequest,
    SessionStoreFactory, SessionUsageReport, TerminationPolicy, TokenLedgerEntry, TurnActivity,
    TurnActivityId, TurnActivitySink, TurnContext, TurnEvent, TurnInput, TurnIssue, TurnOptions,
    UnavailableProcessService, UsageReportRow, UsageTotals, current_epoch_ms, diff_token_ledger,
    diff_usage_reports, epoch_ms_from_system_time, lashlang_process_event_types,
    materialize_process_event_semantics, prepare_process_registration, process_event_payload_hash,
    process_wake_delivery, process_wake_input_from_event_payload, process_wake_turn_cause,
    process_wake_turn_text, require_event_idempotency, system_time_from_epoch_ms,
};
pub use schemars::JsonSchema;
pub use session::{
    ExecRequest, InjectedTurnInput, RuntimeExecutionContext, Session, SessionError, ToolInvocation,
    ToolInvocationReply, TurnInjectionBridge, TurnInputInjectionBridge,
};
pub use session_graph::{
    PersistedSessionConfig, PersistedTurnState, SessionGraph, SessionMessageTreeNode,
    SessionNodePayload, SessionNodeRecord,
};
pub use session_model::context::PreparedContext;
pub use session_model::{ConversationRecord, ProtocolEvent, SessionEventRecord, ToolEvent};
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
    ToolPrepareContext, ToolProvider, ToolSessionControl, ToolSessionModel,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn protocol_turn_options_missing_payload_deserializes_to_empty_object() {
        let options: ProtocolTurnOptions =
            serde_json::from_value(serde_json::json!({})).expect("deserialize options");

        assert!(options.is_empty());
        assert_eq!(options.payload, serde_json::json!({}));
    }

    #[test]
    fn protocol_turn_options_explicit_null_is_not_empty() {
        let options: ProtocolTurnOptions =
            serde_json::from_value(serde_json::json!({ "payload": null }))
                .expect("deserialize options");

        assert!(!options.is_empty());
        assert_eq!(options.payload, serde_json::Value::Null);
    }
}
