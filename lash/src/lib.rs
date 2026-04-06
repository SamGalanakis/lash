pub mod direct;
pub mod dynamic;
pub mod embedded;
pub mod instructions;
pub mod llm;
pub mod mcp;
pub mod model_info;
pub mod model_variant;
pub mod oauth;
pub mod plugin;
pub mod provider;
pub mod runtime;
#[cfg(feature = "sqlite-store")]
mod search;
pub mod session;
pub mod session_model;
pub mod skill_catalog;
pub mod skill_prompt;
pub mod store;
pub mod text;
mod tool_dispatch;
pub mod tools;

pub use lash_sansio::sansio;

use std::path::PathBuf;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const SANSIO_VERSION: &str = lash_sansio::VERSION;

/// Return the root data directory for lash.
///
/// Checks `LASH_HOME` env var first, falling back to `~/.lash/`.
pub fn lash_home() -> PathBuf {
    if let Ok(dir) = std::env::var("LASH_HOME") {
        PathBuf::from(dir)
    } else {
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".lash")
    }
}

/// Return the cache directory for lash.
///
/// When `LASH_HOME` is set: `$LASH_HOME/cache`.
/// Otherwise: `~/.cache/lash/` (via `dirs::cache_dir`).
pub fn lash_cache_dir() -> PathBuf {
    if std::env::var("LASH_HOME").is_ok() {
        lash_home().join("cache")
    } else {
        dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from(".cache"))
            .join("lash")
    }
}

/// Return the preferred repo-local directory for lash artifacts.
pub fn repo_local_lash_dir() -> PathBuf {
    PathBuf::from(".agents").join("lash")
}

/// Return the legacy repo-local directory for lash artifacts.
pub fn legacy_repo_local_lash_dir() -> PathBuf {
    PathBuf::from(".lash")
}

/// Return skill search directories in override order from lowest to highest priority.
pub fn default_skill_dirs() -> Vec<PathBuf> {
    vec![
        lash_home().join("skills"),
        legacy_repo_local_lash_dir().join("skills"),
        repo_local_lash_dir().join("skills"),
    ]
}

// Re-exports
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
pub use lash_sansio::{
    CheckpointKind, ContextStrategy, DefaultPromptRenderer, DurableTurnSnapshot, Effect, EffectId,
    ErrorEnvelope, ExecResponse, ExecutionMode, LlmCallError, Message, MessageOrigin, MessageRole,
    Part, PartKind, PluginMessage, PluginSurfaceEvent, PromptContext, PromptContribution,
    PromptOverrideMode, PromptRenderer, PromptRequest, PromptResponse, PromptSectionName,
    PromptSectionOverride, PromptSelectionMode, PruneState, Response, SessionEvent, TokenUsage,
    ToolCallRecord, ToolDefinition, ToolImage, ToolParam, ToolResult, TurnMachine,
    TurnMachineConfig, default_context_strategy, default_execution_mode, default_prompt_renderer,
    execution_mode_supported,
};
pub use mcp::{McpError, McpServerConfig, McpToolExecutionAdapter, attach_mcp_servers};
pub use model_info::{
    CachedModelCatalog, FileModelCatalogStore, MemoryModelCatalogStore, ModelCatalog,
    ModelCatalogSource, ModelCatalogStore, ModelInfo, ModelsDevHttpSource, ResolvedModelSpec,
};
pub use model_variant::VariantRequestConfig;
pub use plugin::{
    AssistantResponseHookContext, AssistantResponseTransform, AssistantStreamHookContext,
    AssistantStreamTransform, BuiltinToolResultProjectionPluginFactory, CheckpointHookContext,
    ExternalInvokeContext, ExternalInvokeError, ExternalOpDef, ExternalOpKind, PluginDirective,
    PluginError, PluginFactory, PluginHost, PluginOwned, PluginRegistrar, PluginSession,
    PluginSessionContext, PluginSessionSnapshot, PluginSnapshotArtifact, PluginSnapshotEntry,
    PluginSnapshotMeta, PluginSpec, PluginSpecFactory, PromptHookContext, RuntimeServices,
    SessionContextSurface, SessionCreateRequest, SessionHandle, SessionManager, SessionParam,
    SessionPlugin, SessionPluginMode, SessionSnapshot, SessionStartPoint, SessionTurnHandle,
    SnapshotReader, SnapshotWriter, ToolResultProjectionContext, ToolResultProjectionHook,
    ToolResultProjectionMode, ToolResultProjectionPluginConfig, ToolResultProjector,
    ToolSurfaceContribution, TurnHookContext, TurnResultHookContext,
    plugin_surface_event_renders_visible_output,
};
#[cfg(feature = "sqlite-store")]
pub use plugin::{
    BuiltinPlanModePluginFactory, BuiltinPlanTrackerPluginFactory,
    BuiltinPromptContextPluginFactory, PromptContextPluginConfig,
};
pub use provider::{LashConfig, Provider, ProviderOptions, RequestTimeout};
pub use runtime::{
    AssembledTurn, AssistantOutput, CodeOutputRecord, DoneReason, EventSink, ExecutionSummary,
    HostProfile, InputItem, LashRuntime, NoopEventSink, OutputState, PathResolver, PromptUsage,
    RunMode, RuntimeError, RuntimeHostConfig, SanitizerPolicy, SessionStateEnvelope,
    SessionStoreCreateRequest, SessionStoreFactory, TerminationPolicy, TurnInput, TurnIssue,
    TurnStatus,
};
pub use session::{Session, SessionError, TurnInjectionBridge};
pub use session_model::SessionPolicy;
pub use skill_catalog::{LoadedSkill, SkillCatalog};
pub use skill_prompt::{
    append_skill_blocks, collect_skill_mentions, collect_skill_mentions_with_ranges,
};
pub use store::{
    RuntimeStore, SessionMeta, SessionPickerInfo, SessionState, TranscriptEntry,
    TranscriptEntryPayload, TranscriptKeyspace, TurnCheckpoint, group_transcript_entries,
    semantic_transcript_keyspaces, transcript_messages, transcript_tool_calls,
};
#[cfg(feature = "sqlite-store")]
pub use store::{SqliteStore, Store};
pub use text::strip_repl_fragments;
pub use tools::{DefaultToolPluginDeps, default_tool_plugin_factories};

/// A message sent from the sandbox to the host during execution.
#[derive(Clone, Debug)]
pub struct SandboxMessage {
    pub text: String,
    /// "final", "tool_output", or other host-rendered progress events such as "delegate_start"
    pub kind: String,
}

/// Sender for streaming progress messages from tools (e.g. live bash output).
pub type ProgressSender = tokio::sync::mpsc::UnboundedSender<SandboxMessage>;

#[derive(Clone)]
pub struct ToolExecutionContext {
    pub session_id: String,
    pub host: std::sync::Arc<dyn crate::plugin::SessionManager>,
}

/// Trait for providing tools to the sandbox. Implement this per-project.
#[async_trait::async_trait]
pub trait ToolProvider: Send + Sync + 'static {
    fn definitions(&self) -> Vec<ToolDefinition>;
    fn dynamic_snapshot(&self) -> Option<crate::dynamic::DynamicStateSnapshot> {
        None
    }
    fn fork_dynamic_with_snapshot(
        &self,
        _snapshot: crate::dynamic::DynamicStateSnapshot,
    ) -> Option<std::sync::Arc<dyn ToolProvider>> {
        None
    }
    fn dynamic_generation(&self) -> Option<u64> {
        None
    }
    async fn execute(&self, name: &str, args: &serde_json::Value) -> ToolResult;

    async fn execute_with_context(
        &self,
        name: &str,
        args: &serde_json::Value,
        _context: &ToolExecutionContext,
    ) -> ToolResult {
        self.execute(name, args).await
    }

    /// Execute with progress streaming. Default: delegates to execute().
    async fn execute_streaming(
        &self,
        name: &str,
        args: &serde_json::Value,
        _progress: Option<&ProgressSender>,
    ) -> ToolResult {
        self.execute(name, args).await
    }

    /// Execute with progress streaming and session context. Default: delegates to
    /// `execute_streaming()`.
    async fn execute_streaming_with_context(
        &self,
        name: &str,
        args: &serde_json::Value,
        _context: &ToolExecutionContext,
        progress: Option<&ProgressSender>,
    ) -> ToolResult {
        self.execute_streaming(name, args, progress).await
    }
}
