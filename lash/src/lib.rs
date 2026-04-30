pub mod context_approach;
pub mod direct;
pub mod dynamic;
pub mod embedded;
pub mod instructions;
pub mod llm;
pub mod mcp;
pub mod model_info;
pub mod monitor;
pub mod oauth;
pub mod plugin;
pub mod provider;
pub mod runtime;
pub mod runtime_controls;
#[cfg(feature = "sqlite-store")]
pub mod search;
pub mod session;
pub mod session_graph;
pub mod session_model;
pub mod skill_catalog;
pub mod skill_prompt;
pub mod store;
#[cfg(any(test, feature = "testing"))]
pub mod testing;
pub mod tool_dispatch;
mod tool_provider;
pub mod tools;
mod trace;

pub use lash_sansio::sansio;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const SANSIO_VERSION: &str = lash_sansio::VERSION;

// Re-exports
pub use context_approach::{
    ContextApproach, ContextApproachKind, ObservationalMemoryConfig, RollingHistoryConfig,
};
pub use direct::{
    DirectJsonSchema, DirectLlmClient, DirectLlmError, DirectMessage, DirectOutputSpec, DirectPart,
    DirectRequest, DirectRole,
};
pub use dynamic::{
    DynamicStateSnapshot, DynamicToolProvider, DynamicToolSpec, InProcessToolExecutionAdapter,
    InProcessToolFuture, InProcessToolHandler, ReconfigureError, ToolExecutionAdapter,
};
pub use instructions::InstructionLoaderConfig;
pub use instructions::{FsInstructionSource, InstructionLoader, InstructionSource};
pub use lash_sansio::llm::types::{LlmOutputPart, LlmRequest, LlmResponse};
pub use lash_sansio::{
    BaseRenderCache, CheckpointKind, EffectId, ErrorEnvelope, ExecResponse, ExecutionMode,
    LlmCallError, Message, MessageOrigin, MessageRole, MessageSequence, ModeBuildInput, Part,
    PartKind, PluginMessage, PluginSurfaceEvent, PreparedPrompt, PromptBuildInput, PromptBuiltin,
    PromptContext, PromptContribution, PromptPanel, PromptRequest, PromptResponse,
    PromptSelectionMode, PromptSlot, PromptTemplate, PromptTemplateEntry, PromptTemplateSection,
    PruneState, RenderedPrompt, Response, SessionEvent, TokenUsage, ToolActivation,
    ToolAvailability, ToolAvailabilityConfig, ToolCallRecord, ToolDefinition, ToolExecutionMode,
    ToolImage, ToolParam, ToolResult, ToolSurface, ToolSurfaceBuildInput, ToolSurfaceEntry,
    UserInputProvenance, UserInputTransform, append_assistant_text_part, build_prompt,
    build_tool_surface, build_turn, default_execution_mode, default_prompt_template,
    execution_mode_supported, head_tail_truncate, messages_are_prompt_resume_safe,
    normalized_response_parts, reasoning_part, shared_parts, turn_limit_exhausted_message,
};
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
#[serde(tag = "mode", content = "options", rename_all = "snake_case")]
pub enum ModeTurnOptions {
    #[default]
    Unit,
    Rlm(lash_rlm_types::RlmTermination),
}

impl ModeTurnOptions {
    pub fn rlm(termination: lash_rlm_types::RlmTermination) -> Self {
        Self::Rlm(termination)
    }

    pub fn rlm_termination(&self) -> lash_rlm_types::RlmTermination {
        match self {
            Self::Unit => lash_rlm_types::RlmTermination::default(),
            Self::Rlm(termination) => termination.clone(),
        }
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
    TraceLlmMessage, TraceLlmRequest, TraceLlmResponse, TracePromptComponent, TraceRecord,
    TraceSink, TraceTokenUsage, TraceToolSpec,
};
pub use mcp::{McpError, McpServerConfig, McpToolExecutionAdapter, attach_mcp_servers};
pub use model_info::{
    CachedModelCatalog, FileModelCatalogStore, MemoryModelCatalogStore, ModelCatalog,
    ModelCatalogSource, ModelCatalogStore, ModelInfo, ModelsDevHttpSource, ResolvedModelSpec,
};
pub use monitor::{
    MAX_MONITOR_TIMEOUT_MS, MonitorArmOn, MonitorEvent, MonitorRunState, MonitorSnapshot,
    MonitorSpec, MonitorStatus, MonitorUpdateBatch, MonitorWakePolicy,
};
pub use plugin::{
    AppendSessionNodesRequest, AppendSessionNodesResult, AssistantResponseHookContext,
    AssistantResponseTransform, AssistantStreamHookContext, AssistantStreamTransform,
    BuiltinToolResultProjectionPluginFactory, CheckpointHookContext, CommandDef, CommandHandler,
    CommandInvocation, CommandOutcome, CommandRegistrations, DirectCompletion,
    ExternalInvokeContext, ExternalInvokeError, ExternalOpDef, ExternalOpKind, HistoryError,
    HistoryRegistrations, HistoryRewriteMetadata, HistoryRewriter, HistoryState, ModeExtras,
    MonitorRegistrations, PersistentRuntimeServices, PluginDirective, PluginError, PluginFactory,
    PluginHost, PluginOwned, PluginRegistrar, PluginRuntimeEvent, PluginRuntimeEventHook,
    PluginSession, PluginSessionContext, PluginSessionSnapshot, PluginSnapshotArtifact,
    PluginSnapshotEntry, PluginSnapshotMeta, PluginSpec, PluginSpecFactory, PromptHookContext,
    PromptRequestHookContext, RewriteContext, RewriteTrigger, RuntimeServices, SessionAppendNode,
    SessionConfigChangedContext, SessionContextSurface, SessionCreateRequest, SessionHandle,
    SessionManager, SessionParam, SessionPlugin, SessionPluginMode, SessionReadView,
    SessionSnapshot, SessionStartPoint, SessionStateChangedContext, SessionTurnHandle,
    SnapshotReader, SnapshotWriter, StandardCreateExtras, ToolResultProjectionContext,
    ToolResultProjectionHook, ToolResultProjectionMode, ToolResultProjectionPluginConfig,
    ToolResultProjector, ToolSurfaceContribution, TurnContextTransform, TurnHookContext,
    TurnResultHookContext, TurnResultSummary, TurnTransformContext,
    plugin_surface_event_renders_visible_output,
};
pub use provider::{
    AgentModelSelection, LashConfig, Provider, ProviderFactory, ProviderHandle, ProviderOptions,
    ProviderRegistry, ProviderSpec, RequestTimeout, VariantRequestConfig, build_provider,
    provider_cli_label, provider_factory, register_provider_factory,
};
pub use runtime::{
    AssembledTurn, AssistantOutput, BackgroundRuntimeHost, CodeOutputRecord, DefaultPathResolver,
    DoneReason, EmbeddedRuntimeBuilder, EmbeddedRuntimeHost, EventSink, ExecutionSummary,
    InputItem, LashRuntime, ManagedRunState, ManagedTaskCancel, ManagedTaskKind, ManagedTaskSpec,
    ManagedTaskStatus, NoopEventSink, OutputState, ParkedSession, PathResolver,
    PersistedSessionState, PromptUsage, Residency, RunMode, RuntimeCoreConfig, RuntimeEnvironment,
    RuntimeEnvironmentBuilder, RuntimeError, SanitizerPolicy, SessionStateEnvelope,
    SessionStoreCreateRequest, SessionStoreFactory, SessionTaskExecutor, SessionUsageReport,
    TerminationPolicy, TokenLedgerEntry, TokioSessionTaskExecutor, TurnInput, TurnIssue,
    TurnStatus, UsageReportRow, UsageTotals, diff_token_ledger, diff_usage_reports,
};
pub use runtime_controls::{BuiltinMonitorToolPluginFactory, BuiltinTaskControlsPluginFactory};
pub use session::{
    InjectedTurnInput, Session, SessionError, TurnInjectionBridge, TurnInputInjectionBridge,
};
pub use session_graph::{
    PersistedSessionConfig, PersistedTurnState, SessionGraph, SessionMessageTreeNode,
    SessionNodePayload, SessionNodeRecord, SessionProjection,
};
pub use session_model::SessionPolicy;
pub use session_model::context::PreparedContext;
pub use session_model::{
    ConversationRecord, ModeEvent, SessionEventRecord, StateSnapshotEvent, ToolEvent,
};
pub use skill_catalog::{LoadedSkill, SkillCatalog};
pub use skill_prompt::{
    append_skill_blocks, collect_skill_mentions, collect_skill_mentions_with_ranges,
};
pub use store::{
    BlobArtifactDescriptor, BlobCompression, BlobRef, BlobStorageHint, BlobStore, GcReport,
    HydratedSessionCheckpoint, PersistedArtifactKind, PersistedStateCommit,
    PersistedStateCommitResult, RetainedArtifactRef, RetentionStore, RuntimePersistence,
    SessionCheckpoint, SessionGraphCommit, SessionGraphStore, SessionHead, SessionHeadMeta,
    SessionHeadStore, SessionMeta, SessionPickerInfo, UsageLedgerStore, VacuumReport,
    apply_runtime_commit, head_copy_from_store, load_persisted_session_state,
    load_persisted_session_state_active_path, load_session_head, refresh_persisted_session_state,
    save_session_head,
};
#[cfg(feature = "sqlite-store")]
pub use store::{BuiltinBlobProfile, SqliteStore, Store, StoreGcPolicy, StoreOptions};
pub use tool_provider::{ProgressSender, SandboxMessage, ToolExecutionContext, ToolProvider};
