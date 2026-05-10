pub mod attachments;
pub mod chronological;
pub mod direct;
pub mod instructions;
pub mod llm;
pub mod mcp;
pub mod model_info;
pub mod monitor;
pub mod plugin;
pub mod provider;
pub mod runtime;
pub mod runtime_controls;
pub mod search;
pub mod session;
pub mod session_graph;
pub mod session_model;
pub mod standard_context_approach;
pub mod store;
#[cfg(any(test, feature = "testing"))]
pub mod testing;
pub mod tool_dispatch;
mod tool_provider;
pub mod tool_registry;
mod tool_schema;
mod trace;

pub use lash_sansio::sansio;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const SANSIO_VERSION: &str = lash_sansio::VERSION;

// Re-exports
pub use attachments::{
    AttachmentStore, AttachmentStoreError, FileAttachmentStore, InMemoryAttachmentStore,
    StoredAttachment,
};
pub use chronological::{ChronologicalEntry, ChronologicalPayload, ChronologicalProjection};
pub use direct::{
    DirectJsonSchema, DirectLlmClient, DirectLlmError, DirectMessage, DirectOutputSpec, DirectPart,
    DirectRequest, DirectRole,
};
pub use instructions::InstructionLoaderConfig;
pub use instructions::{FsInstructionSource, InstructionLoader, InstructionSource};
pub use lash_sansio::llm::types::{LlmOutputPart, LlmRequest, LlmResponse};
pub use lash_sansio::{
    AcceptedInjectedTurnInput, AttachmentId, AttachmentMeta, AttachmentRef, BaseRenderCache,
    CheckpointKind, CompactToolContract, EffectId, ErrorEnvelope, ExecResponse, ExecutionMode,
    ImageMediaType, LlmCallError, MediaType, Message, MessageOrigin, MessageRole, MessageSequence,
    ModeBuildInput, Part, PartKind, PluginMessage, PluginSurfaceEvent, PreparedPrompt,
    PromptBuildInput, PromptBuiltin, PromptContext, PromptContribution, PromptContributionGate,
    PromptLayer, PromptPanel, PromptRequest, PromptResponse, PromptSelectionMode, PromptSlot,
    PromptSlotLayer, PromptTemplate, PromptTemplateEntry, PromptTemplateSection, PruneState,
    RenderedPrompt, ResolvedPromptLayer, Response, SchemaProjectionOverride, SessionEvent,
    TextProjectionMetadata, TokenUsage, ToolActivation, ToolAvailability, ToolAvailabilityConfig,
    ToolCallRecord, ToolControl, ToolDefinition, ToolDiscoveryMetadata, ToolExecutionMode,
    ToolImage, ToolOutputContract, ToolResult, ToolSurface, ToolSurfaceBuildInput,
    ToolSurfaceEntry, ToolSurfaceOverride, TurnFinish, TurnOutcome, TurnStop,
    append_assistant_text_part, build_prompt, build_tool_surface, build_turn,
    default_execution_mode, default_prompt_template, execution_mode_supported, head_tail_truncate,
    messages_are_prompt_resume_safe, normalized_response_parts, reasoning_part,
    resolve_prompt_layers, shared_parts, turn_limit_exhausted_message,
};
pub use standard_context_approach::{
    ObservationalMemoryConfig, RollingHistoryConfig, StandardContextApproach,
    StandardContextApproachKind,
};
pub use tool_registry::{
    ReconfigureError, ToolRegistry, ToolSourceHandle, ToolState, ToolStateEntry,
};
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

#[derive(Clone, Debug)]
pub struct HostModeProtocol;

impl lash_sansio::ModeProtocol for HostModeProtocol {
    type Event = crate::session_model::ModeEvent;
    type Termination = ModeTurnOptions;
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
pub use mcp::{McpError, McpServerConfig, attach_mcp_servers};
pub use model_info::{
    CachedModelCatalog, FileModelCatalogStore, MemoryModelCatalogStore, ModelCatalog,
    ModelCatalogSource, ModelCatalogStore, ModelInfo, ModelsDevHttpSource, ResolvedModelSpec,
};
pub use monitor::{
    MAX_MONITOR_TIMEOUT_MS, MonitorArmOn, MonitorEvent, MonitorRunState, MonitorSnapshot,
    MonitorSpec, MonitorStatus, MonitorUpdateBatch, MonitorWakePolicy,
};
pub use plugin::{
    AckWakeArgs, AppendSessionNodesRequest, AppendSessionNodesResult, AssistantResponseHookContext,
    AssistantResponseTransform, AssistantStreamHookContext, AssistantStreamTransform,
    BuiltinToolResultProjectionPluginFactory, CheckpointHookContext, CheckpointHookHost,
    DirectCompletion, DirectCompletionHost, DirectLlmCompletion, ExternalInvokeContext,
    ExternalInvokeError, ExternalInvokeHost, ExternalOpDef, ExternalOpKind, HistoryError,
    HistoryHost, HistoryRegistrations, HistoryRewriteMetadata, HistoryRewriter, HistoryState,
    ModeBeforeLlmCallContext, ModeExtras, ModeLlmCallAction, MonitorAckWakeOp, MonitorEmptyArgs,
    MonitorHost, MonitorRegisterSpecsOp, MonitorRegistrations, MonitorStartOp, MonitorStatusOp,
    MonitorStopOp, MonitorTakeUpdatesOp, OwnedMonitorSpec, PersistentRuntimeServices,
    PluginDirective, PluginError, PluginFactory, PluginHost, PluginOwned, PluginRegistrar,
    PluginRuntimeEvent, PluginRuntimeEventHook, PluginSession, PluginSessionContext,
    PluginSessionSnapshot, PluginSnapshotArtifact, PluginSnapshotEntry, PluginSnapshotMeta,
    PluginSpec, PluginSpecFactory, PromptHookContext, PromptHookHost, RegisterSpecsArgs,
    RewriteContext, RewriteTrigger, RuntimeServices, RuntimeSessionHost, SessionAppendNode,
    SessionConfigChangedContext, SessionContextSurface, SessionCreateRequest, SessionGraphHost,
    SessionHandle, SessionLifecycleHost, SessionParam, SessionPlugin, SessionPluginMode,
    SessionReadView, SessionRelation, SessionSnapshot, SessionSnapshotHost, SessionStartPoint,
    SessionStateChangedContext, SessionToolAccess, SessionTurnHandle, SnapshotReader,
    SnapshotWriter, StandardCreateExtras, StartMonitorArgs, StopMonitorArgs,
    SubagentSessionAuthority, TaskHost, ToolCatalogHost, ToolDiscoveryContext,
    ToolDiscoveryContribution, ToolDiscoveryContributor, ToolDiscoveryToolContribution,
    ToolHookHost, ToolResultProjectionContext, ToolResultProjectionHook, ToolResultProjectionMode,
    ToolResultProjectionPluginConfig, ToolResultProjector, ToolStateHost, ToolSurfaceContribution,
    TraceHost, TurnContextTransform, TurnHookContext, TurnHookHost, TurnHost,
    TurnResultHookContext, TurnResultHookHost, TurnResultSummary, TurnTransformContext,
    TypedExternalOp, TypedExternalOpError, typed_external_op_def,
};
pub use provider::{
    AgentModelSelection, LashConfig, LlmTimeouts, ProviderComponents, ProviderFactory,
    ProviderHandle, ProviderModelPolicy, ProviderOptions, ProviderRegistry, ProviderSpec,
    ProviderState, ProviderThinkingPolicy, ProviderTransport, RequestTimeout, StaticModelPolicy,
    VariantRequestConfig, build_provider, provider_factory, register_provider_factory,
};
pub use runtime::{
    AssembledTurn, AssistantOutput, BackgroundRuntimeHost, CodeOutputRecord,
    EmbeddedRuntimeBuilder, EmbeddedRuntimeHost, EventSink, ExecutionSummary, FollowedTurn,
    InputItem, LashRuntime, ManagedRunState, ManagedTaskCancel, ManagedTaskKind, ManagedTaskSpec,
    ManagedTaskStatus, ModeSessionExtension, ModeSessionExtensionHandle, ModeTurnExtension,
    ModeTurnExtensionHandle, NoopEventSink, NoopTurnActivitySink, OutputState, ParkedSession,
    PersistedSessionState, PromptUsage, Residency, RunMode, RuntimeCoreConfig, RuntimeEnvironment,
    RuntimeEnvironmentBuilder, RuntimeError, RuntimeHandle, RuntimeObservation,
    SessionStateEnvelope, SessionStoreCreateRequest, SessionStoreFactory, SessionTaskExecutor,
    SessionUsageReport, TerminationPolicy, TokenLedgerEntry, TokioSessionTaskExecutor,
    ToolResultView, TurnActivity, TurnActivityId, TurnActivitySink, TurnContext, TurnEvent,
    TurnInput, TurnIssue, UsageReportRow, UsageTotals, diff_token_ledger, diff_usage_reports,
};
pub use runtime_controls::{BuiltinMonitorToolPluginFactory, BuiltinTaskControlsPluginFactory};
pub use schemars::JsonSchema;
pub use session::{
    ExecRequest, InjectedTurnInput, ModeExecutionContext, ModeToolBatchItem, ModeToolReply,
    Session, SessionError, TurnInjectionBridge, TurnInputInjectionBridge,
};
pub use session_graph::{
    PersistedSessionConfig, PersistedTurnState, SessionGraph, SessionMessageTreeNode,
    SessionNodePayload, SessionNodeRecord,
};
pub use session_model::SessionPolicy;
pub use session_model::context::PreparedContext;
pub use session_model::{
    ConversationRecord, ModeEvent, SessionEventRecord, StateSnapshotEvent, ToolEvent,
};
pub use store::{
    BlobRef, GcReport, GraphCommitDelta, HydratedSessionCheckpoint, PersistedSessionRead,
    RuntimeCommit, RuntimeCommitResult, RuntimePersistence, SessionCheckpoint, SessionHead,
    SessionHeadMeta, SessionMeta, SessionPickerInfo, SessionReadScope, StoreError, VacuumReport,
    load_persisted_session_state, load_persisted_session_state_active_path,
    refresh_persisted_session_state,
};
pub use tool_provider::{ProgressSender, SandboxMessage, ToolExecutionContext, ToolProvider};
