pub mod context_approach;
pub mod direct;
pub mod dynamic;
pub mod embedded;
pub mod instructions;
pub mod llm;
pub mod mcp;
pub mod model_info;
pub mod model_variant;
pub mod monitor;
pub mod oauth;
mod paths;
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
#[cfg(test)]
pub(crate) mod test_support;
pub mod tool_dispatch;
mod tool_provider;
pub mod tools;

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
pub use lash_sansio::llm::types::{LlmOutputPart, LlmResponse};
pub use lash_sansio::{
    CheckpointKind, Effect, EffectId, ErrorEnvelope, ExecResponse, ExecutionMode, LlmCallError,
    Message, MessageOrigin, MessageRole, MessageSequence, ModeBuildInput, ModeConfig, ModePreamble,
    Part, PartKind, PluginMessage, PluginSurfaceEvent, PreparedPrompt, PreparedTurnMachine,
    PromptBuildInput, PromptBuiltin, PromptContext, PromptContribution, PromptPanel, PromptRequest,
    PromptResponse, PromptSelectionMode, PromptSlot, PromptTemplate, PromptTemplateEntry,
    PromptTemplateSection, PruneState, RenderedPrompt, Response, SansIoTurnInput, SessionEvent,
    TokenUsage, ToolCallRecord, ToolDefinition, ToolExecutionMode, ToolImage, ToolParam,
    ToolResult, ToolSurface, ToolSurfaceBuildInput, TurnMachine, TurnMachineConfig,
    UserInputProvenance, UserInputTransform, append_assistant_text_part, build_prompt,
    build_tool_surface, build_turn, default_execution_mode, default_prompt_template,
    execution_mode_supported, messages_are_live_resume_safe, normalized_response_parts,
    reasoning_part, turn_limit_exhausted_message,
};
pub use mcp::{McpError, McpServerConfig, McpToolExecutionAdapter, attach_mcp_servers};
pub use model_info::{
    CachedModelCatalog, FileModelCatalogStore, MemoryModelCatalogStore, ModelCatalog,
    ModelCatalogSource, ModelCatalogStore, ModelInfo, ModelsDevHttpSource, ResolvedModelSpec,
};
pub use model_variant::VariantRequestConfig;
pub use monitor::{
    MAX_MONITOR_TIMEOUT_MS, MonitorArmOn, MonitorEvent, MonitorRunState, MonitorSnapshot,
    MonitorSpec, MonitorStatus, MonitorUpdateBatch, MonitorWakePolicy,
};
pub use paths::{
    default_skill_dirs, lash_cache_dir, lash_home, legacy_repo_local_lash_dir, repo_local_lash_dir,
};
pub use plugin::ObservationalMemoryPluginFactory as BuiltinObservationalMemoryPluginFactory;
pub use plugin::RollingHistoryPluginFactory as BuiltinRollingHistoryPluginFactory;
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
    PromptRequestHookContext, RewriteContext, RewriteTrigger, RlmCreateExtras, RlmTermination,
    RuntimeServices, SessionAppendNode, SessionConfigChangedContext, SessionContextSurface,
    SessionCreateRequest, SessionHandle, SessionManager, SessionParam, SessionPlugin,
    SessionPluginMode, SessionReadView, SessionSnapshot, SessionStartPoint,
    SessionStateChangedContext, SessionTurnHandle, SnapshotReader, SnapshotWriter,
    StandardCreateExtras, ToolResultProjectionContext, ToolResultProjectionHook,
    ToolResultProjectionMode, ToolResultProjectionPluginConfig, ToolResultProjector,
    ToolSurfaceContribution, TurnContextTransform, TurnHookContext, TurnResultHookContext,
    TurnResultSummary, TurnTransformContext, plugin_surface_event_renders_visible_output,
};
#[cfg(feature = "sqlite-store")]
pub use plugin::{
    BuiltinPlanModePluginFactory, BuiltinPromptContextPluginFactory,
    BuiltinUiActivityPluginFactory, BuiltinUpdatePlanPluginFactory, PromptContextPluginConfig,
    UpdatePlanItem, UpdatePlanSnapshot,
};
pub use provider::{LashConfig, Provider, ProviderOptions, RequestTimeout};
pub use runtime::{
    AssembledTurn, AssistantOutput, BackgroundRuntimeHost, CodeOutputRecord, DefaultPathResolver,
    DoneReason, EmbeddedRuntimeBuilder, EmbeddedRuntimeHost, EventSink, ExecutionSummary,
    InputItem, LashRuntime, ManagedRunState, ManagedTaskCancel, ManagedTaskKind, ManagedTaskSpec,
    ManagedTaskStatus, NoopEventSink, OutputState, PathResolver, PersistedSessionState,
    PromptUsage, RunMode, RuntimeCoreConfig, RuntimeError, SanitizerPolicy, SessionStateEnvelope,
    SessionStoreCreateRequest, SessionStoreFactory, SessionTaskExecutor, SessionUsageReport,
    TerminationPolicy, TokenLedgerEntry, TokioSessionTaskExecutor, TurnInput, TurnIssue,
    TurnStatus, UsageReportRow, UsageTotals, diff_token_ledger, diff_usage_reports,
};
pub use session::{
    InjectedTurnInput, Session, SessionError, TurnInjectionBridge, TurnInputInjectionBridge,
};
pub use session_graph::{
    INTERNAL_RLM_GLOBALS_PATCH_PLUGIN_TYPE, INTERNAL_TOOL_CALL_PLUGIN_TYPE, PersistedSessionConfig,
    PersistedTurnState, RlmGlobalsPatchPluginBody, SessionGraph, SessionMessageTreeNode,
    SessionNodePayload, SessionNodeRecord, ToolCallPluginBody,
};
pub use session_model::SessionPolicy;
pub use session_model::context::PreparedContext;
pub use skill_catalog::{LoadedSkill, SkillCatalog};
pub use skill_prompt::{
    append_skill_blocks, collect_skill_mentions, collect_skill_mentions_with_ranges,
};
pub use store::{
    BlobArtifactDescriptor, BlobCompression, BlobRef, BlobStorageHint, GcReport,
    HydratedSessionCheckpoint, LiveResumeCommit, LiveResumeDelta, LiveResumeSnapshot,
    PersistedArtifactKind, PersistedStateCommit, PersistedStateCommitResult, RetainedArtifactRef,
    RuntimeCommit, RuntimeCommitResult, RuntimeStore, SessionCheckpoint, SessionGraphCommit,
    SessionHead, SessionHeadMeta, SessionMeta, SessionPickerInfo, materialize_live_resume_graph,
};
#[cfg(feature = "sqlite-store")]
pub use store::{BuiltinBlobProfile, SqliteStore, Store, StoreGcPolicy, StoreOptions};
pub use tool_provider::{ProgressSender, SandboxMessage, ToolExecutionContext, ToolProvider};
