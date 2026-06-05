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
mod trace;

pub use lash_sansio::sansio;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const SANSIO_VERSION: &str = lash_sansio::VERSION;

// Re-exports
pub use attachments::{
    AttachmentStore, AttachmentStoreError, AttachmentStorePersistence, InMemoryAttachmentStore,
    SessionScopedAttachmentStore, StoredAttachment,
};
// The Lashlang artifact store is a host-owned durability dependency of
// `RuntimeHostConfig`; re-export it so the `lash` facade can name it without a
// direct `lashlang` dependency.
pub use chronological::{
    BorrowedChronologicalEntry, BorrowedChronologicalMessage, BorrowedChronologicalPayload,
    ChronologicalEntry, ChronologicalPayload, ChronologicalProjection, visit_turn_view,
};
pub use direct::{
    DirectJsonSchema, DirectLlmClient, DirectLlmError, DirectMessage, DirectOutputSpec, DirectPart,
    DirectRequest, DirectRole,
};
pub use host_events::{
    HostEvent, HostEventCatalog, HostEventEmitReport, HostEventKey, host_event_source_type,
};
pub use lash_sansio::llm::types::{
    GenerationOptions, LlmOutputPart, LlmRequest, LlmResponse, LlmTerminalReason,
};
pub use lash_sansio::{
    AcceptedInjectedTurnInput, AttachmentCreateMeta, AttachmentId, AttachmentMeta, AttachmentRef,
    BaseRenderCache, CheckpointDelivery, CheckpointKind, CompactToolContract, EffectId,
    ErrorEnvelope, ExecImage, ExecResponse, ImageMediaType, LashSchema, LlmCallError, MediaType,
    Message, MessageOrigin, MessageRole, MessageSequence, ModelToolReturn, ModelToolReturnPart,
    Part, PartKind, PluginMessage, PluginRuntimeEvent, PreparedPrompt, PromptBuildInput,
    PromptBuiltin, PromptContext, PromptContribution, PromptContributionGate,
    PromptContributionSet, PromptFingerprint, PromptLayer, PromptSlot, PromptSlotLayer,
    PromptTemplate, PromptTemplateEntry, PromptTemplateSection, PruneState, RenderedPrompt,
    ResolvedPromptLayer, Response, SchemaProjectionOverride, SessionEvent, TextProjectionMetadata,
    TokenUsage, ToolActivation, ToolAgentExecutableSurface, ToolAgentSurface,
    ToolArgumentProjectionPolicy, ToolAvailability, ToolAvailabilityConfig, ToolCallOutcome,
    ToolCallOutput, ToolCallRecord, ToolCallStatus, ToolCancellation, ToolContract, ToolControl,
    ToolDefinition, ToolFailure, ToolFailureClass, ToolFailureSource, ToolId, ToolManifest,
    ToolOutputContract, ToolRetryDisposition, ToolRetryPolicy, ToolScheduling, ToolSurface,
    ToolSurfaceBuildInput, ToolSurfaceEntry, ToolSurfaceOverride, ToolValue, TurnCause, TurnFinish,
    TurnLimitFinalMessage, TurnOutcome, TurnStop, append_assistant_text_part, build_prompt,
    build_tool_surface, build_turn, default_prompt_template, head_tail_truncate,
    messages_are_prompt_resume_safe, normalized_response_parts, prompt_template_fingerprint,
    prompt_text_fingerprint, prompt_tool_names_fingerprint, reasoning_part,
    render_turn_causes_prompt, resolve_prompt_layers, shared_parts, validate_tool_input,
};
pub use lashlang::{DurabilityTier, InMemoryLashlangArtifactStore, LashlangArtifactStore};
pub use protocol_build::ProtocolBuildInput;
pub use session::triggers::TriggerActivationService;
pub use tool_registry::{
    ReconfigureError, ToolRegistry, ToolSourceHandle, ToolState, ToolStateEntry,
};
pub use tool_result::ToolResult;
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

    pub fn merged_with_override(&self, override_options: &Self) -> Self {
        match (&self.payload, &override_options.payload) {
            (serde_json::Value::Object(base), serde_json::Value::Object(overrides)) => {
                let mut payload = base.clone();
                payload.extend(overrides.clone());
                Self {
                    payload: serde_json::Value::Object(payload),
                }
            }
            _ => override_options.clone(),
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
    JsonlTraceSink, TraceAttachment, TraceBranchSelection, TraceContentBlock, TraceContext,
    TraceError, TraceEvent, TraceLabelMetadata, TraceLashlangChildExecution,
    TraceLashlangEdgeSelection, TraceLashlangExecutionEvent, TraceLashlangExecutionIdentity,
    TraceLashlangGraph, TraceLashlangGraphChildLink, TraceLashlangGraphEdge,
    TraceLashlangGraphNode, TraceLashlangGraphStore, TraceLashlangMap, TraceLashlangMapEdge,
    TraceLashlangMapNode, TraceLashlangNodeStatus, TraceLashlangStatus, TraceLevel,
    TraceLlmMessage, TraceLlmRequest, TraceLlmResponse, TracePromptComponent,
    TraceProviderStreamEvent, TraceRecord, TraceRuntimeScope, TraceRuntimeStreamEvent,
    TraceRuntimeSubject, TraceSink, TraceSinkError, TraceTokenUsage, TraceToolSpec,
};
pub use llm::transport::{LlmTransportError, ProviderFailure, ProviderFailureKind};
pub use model::{ModelLimits, ModelSpec};
pub use plugin::{
    AgentFrameAssignment, AgentFrameId, AgentFrameReason, AgentFrameRecord, AgentFrameStatus,
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
    SessionCreateRequest, SessionGraphService, SessionHandle, SessionLifecycleService,
    SessionParam, SessionPlugin, SessionPluginSource, SessionReadView, SessionRelation,
    SessionSnapshot, SessionStartPoint, SessionStateChangedContext, SessionStateService,
    SessionToolAccess, SessionTurnInput, SessionTurnRequest, SnapshotReader, SnapshotWriter,
    SubagentSessionContext, ToolDiscoveryContext, ToolDiscoveryContribution,
    ToolDiscoveryContributor, ToolDiscoveryToolContribution, ToolResultProjectionContext,
    ToolResultProjector, ToolSurfaceContribution, TriggerRegistration, TriggerSourceType,
    TriggerTargetSummary, TurnContextTransform, TurnHookContext, TurnResultHookContext,
    TurnResultSummary, TurnTransformContext, plugin_action_def,
};
pub use plugin_stack::PluginStack;
pub use provider::{
    CacheRetention, EmptyProviderResolver, LlmTimeouts, MapProviderResolver, Provider,
    ProviderBinding, ProviderComponents, ProviderFactory, ProviderHandle, ProviderModelPolicy,
    ProviderOptions, ProviderResolutionError, ProviderSpec, ProviderThinkingPolicy, RequestTimeout,
    RuntimeProviderResolver, SingleProviderResolver, StaticModelPolicy,
};
#[cfg(any(test, feature = "testing"))]
pub use runtime::TestLocalProcessRegistry;
pub use runtime::{
    AgentFrameRun, AssembledTurn, AssistantOutput, CausalRef, CodeOutputRecord,
    DefaultProcessCancelAbility, DeliveryPolicy, DirectCompletionClient, DurableProcessWorker,
    DurableProcessWorkerConfig, DurableStoreFacet, EffectHost, EffectScope, EmbeddedRuntimeBuilder,
    EmbeddedRuntimeHost, EventSink, ExecutionSummary, InMemorySessionStore,
    InMemorySessionStoreFactory, InlineEffectHost, InlineProcessRunHandle,
    InlineRuntimeEffectController, InputItem, LashRuntime, MergeKey, NoopEventSink,
    NoopTurnActivitySink, ObservedProcess, ObservedProcessEvent, ObservedWorkItem, OutputState,
    PROCESS_LEASE_SCHEMA_VERSION, ParkedSession, ProcessAwaitOutput, ProcessCancelAbility,
    ProcessCancelAllRequest, ProcessCancelRequest, ProcessCancelSource, ProcessCancelSummary,
    ProcessDefinitionSelector, ProcessDefinitionSummary, ProcessEvent, ProcessEventAppendRequest,
    ProcessEventAppendResult, ProcessEventType, ProcessExecutionContext, ProcessExternalRef,
    ProcessHandleDescriptor, ProcessHandleGrant, ProcessHandleSummary, ProcessId, ProcessInput,
    ProcessLease, ProcessLeaseCompletion, ProcessLifecycleStatus, ProcessListFilter,
    ProcessListMode, ProcessOpScope, ProcessProvenance, ProcessRecord, ProcessRegistration,
    ProcessRegistry, ProcessRunHandle, ProcessRuntimeHost, ProcessScope, ProcessScopeId,
    ProcessService, ProcessSessionDeleteReport, ProcessStartGrant, ProcessStartOptions,
    ProcessStartRequest, ProcessStatus, ProcessStatusFilter, ProcessTerminalSemantics,
    ProcessTerminalSpec, ProcessTerminalState, ProcessValueSelector, ProcessWake,
    ProcessWakeDedupeKey, ProcessWakeDelivery, ProcessWakeSpec, ProcessWorkDriver,
    ProcessWorkObserver, ProcessWorkPoke, ProcessWorkRunner, ProcessWorkSnapshot, PromptUsage,
    ProtocolSessionExtension, ProtocolSessionExtensionHandle, ProtocolTurnExtension,
    ProtocolTurnExtensionHandle, QueuedWorkPoke, QueuedWorkRunHandle, QueuedWorkRunOutcome,
    QueuedWorkRunRequest, QueuedWorkRunner, Residency, RuntimeEnvironment,
    RuntimeEnvironmentBuilder, RuntimeError, RuntimeErrorCode, RuntimeHandle, RuntimeHostConfig,
    RuntimeObservation, ScopedEffectController, SessionStoreCreateRequest, SessionStoreFactory,
    SessionUsageReport, SlotPolicy, TerminationPolicy, TokenLedgerEntry, TurnActivity,
    TurnActivityId, TurnActivitySink, TurnContext, TurnEvent, TurnInput, TurnIssue, TurnOptions,
    UnavailableProcessService, UsageReportRow, UsageTotals, current_epoch_ms, diff_token_ledger,
    diff_usage_reports, ensure_durable_effect_input, epoch_ms_from_system_time,
    lashlang_process_event_types, system_time_from_epoch_ms,
};
#[allow(unused_imports)]
pub(crate) use runtime::{
    LlmAttachmentSpec, PreparedProcessEventAppend, ProcessEventSemantics, QUEUED_WORK_CLAIM_TTL_MS,
    QueuedCheckpointWork, QueuedTurnWork, QueuedWorkBatch, QueuedWorkBatchDraft, QueuedWorkClaim,
    QueuedWorkClaimBoundary, QueuedWorkCompletion, QueuedWorkItem, QueuedWorkPayload,
    RuntimeReplay, RuntimeScope, RuntimeSubject, materialize_process_event_semantics,
    prepare_process_event_append, prepare_process_registration, process_event_invocation,
    process_event_payload_hash, process_wake_batch_draft, process_wake_delivery,
    process_wake_input_from_event_payload, process_wake_turn_cause, process_wake_turn_text,
    require_event_replay,
};
// Effect / process-control types consumed by external effect hosts (e.g.
// lash-restate's workflows) and their integration tests. Kept on the public
// surface; the rest of the runtime block above stays crate-internal.
pub use runtime::{
    LlmRequestSpec, ProcessCommand, ProcessEffectOutcome, ProcessEventSemanticsSpec,
    RuntimeEffectCommand, RuntimeEffectController, RuntimeEffectControllerError,
    RuntimeEffectEnvelope, RuntimeEffectKind, RuntimeEffectLocalExecutor, RuntimeEffectOutcome,
    RuntimeInvocation, RuntimeSessionState,
};
pub use schemars::JsonSchema;
pub use session::{
    ExecRequest, InjectedTurnInput, RuntimeExecutionContext, Session, SessionError, ToolInvocation,
    ToolInvocationReply,
};
pub use session_graph::{
    PersistedSessionConfig, PersistedTurnState, SessionGraph, SessionMessageTreeNode,
    SessionNodePayload, SessionNodeRecord,
};
pub use session_model::context::PreparedContext;
pub use session_model::{ConversationRecord, ProtocolEvent, SessionEventRecord, ToolEvent};
pub use session_model::{RuntimeSessionPolicy, SessionPolicy, SessionSpec};
pub use store::{
    AttachmentIntent, AttachmentManifest, AttachmentManifestEntry, BlobRef, GcReport,
    RuntimePersistence, SessionMeta, SessionPickerInfo, SessionReadScope, StoreError, VacuumReport,
};
#[allow(unused_imports)]
pub(crate) use store::{
    GraphCommitDelta, PersistedSessionRead, RuntimeCommitResult, SessionCheckpoint,
    SessionHeadMeta, ensure_supported_schema_version, load_persisted_session_state,
    load_persisted_session_state_active_path,
};
pub use store::{
    HydratedSessionCheckpoint, RuntimeCommit, RuntimeTurnCommitStamp, SessionHead,
    refresh_persisted_session_state,
};
pub use tool_provider::{
    PreparedToolCall, ProgressSender, SandboxMessage, ToolCall, ToolContext, ToolHostEventControl,
    ToolLashlangExecutionCallSite, ToolPrepareCall, ToolPrepareContext, ToolProvider,
    ToolSessionControl, ToolSessionModel,
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

    #[test]
    fn root_exports_do_not_reintroduce_removed_session_state_shapes() {
        let source = include_str!("lib.rs");
        let removed_envelope = ["SessionState", "Envelope"].concat();
        let removed_persisted = ["PersistedSession", "Snapshot"].concat();

        assert!(!source.contains(&removed_envelope));
        assert!(!source.contains(&removed_persisted));
    }

    fn public_reexport_block(source: &str, module: &str) -> String {
        let start = format!("pub use {module}::{{");
        let mut block = String::new();
        let mut collecting = false;
        for line in source.lines() {
            if line.trim_start().starts_with(&start) {
                collecting = true;
            }
            if collecting {
                block.push_str(line);
                block.push('\n');
                if line.trim_end() == "};" {
                    break;
                }
            }
        }
        assert!(!block.is_empty(), "missing public {module} re-export block");
        block
    }

    #[test]
    fn root_runtime_exports_exclude_internal_runtime_records() {
        let runtime_exports = public_reexport_block(include_str!("lib.rs"), "runtime");
        for removed in [
            "RuntimeEffectCommand",
            "RuntimeEffectEnvelope",
            "RuntimeEffectKind",
            "RuntimeEffectOutcome",
            "RuntimeInvocation",
            "RuntimeScope",
            "RuntimeSessionState",
            "QueuedWorkBatch",
            "QueuedWorkBatchDraft",
            "QueuedWorkPayload",
            "prepare_process_registration",
            "process_wake_batch_draft",
            "require_event_replay",
        ] {
            assert!(
                !runtime_exports.contains(removed),
                "runtime root export leaked {removed}"
            );
        }
    }

    #[test]
    fn root_store_exports_exclude_wire_records() {
        let store_exports = public_reexport_block(include_str!("lib.rs"), "store");
        for removed in [
            "SessionHead",
            "SessionCheckpoint",
            "RuntimeCommit",
            "HydratedSessionCheckpoint",
            "PersistedSessionRead",
            "GraphCommitDelta",
        ] {
            assert!(
                !store_exports.contains(removed),
                "store root export leaked {removed}"
            );
        }
    }

    #[test]
    fn removed_manager_and_host_trait_names_stay_removed() {
        let removed_manager = ["Runtime", "Session", "Manager"].concat();
        let removed_host = ["Runtime", "Session", "Host"].concat();
        let sources = [
            include_str!("runtime/session_manager/mod.rs"),
            include_str!("plugin/runtime_host.rs"),
            include_str!("tool_dispatch/context.rs"),
            include_str!("tool_provider.rs"),
        ];

        for source in sources {
            assert!(!source.contains(&removed_manager));
            assert!(!source.contains(&removed_host));
        }
    }
}
