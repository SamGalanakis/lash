//! App-facing embedding facade for Lash.
//!
//! `lash` is intentionally a small layer above the lower-level
//! `lash-core` runtime crate. Host applications own providers, persistence,
//! app state, HTTP protocols, auth, and frontend streaming; this crate
//! owns only the ergonomic core/session/turn API.

pub mod control;
mod core;
mod error;
mod mode;
mod plugin_binding;
mod session;
mod support;
#[cfg(test)]
mod tests;
pub mod turn;
pub mod usage;

pub use crate::control::{
    AdvancedToolsControl, BackgroundTasks, Handoffs, PluginActions, ToolsControl,
};
pub use crate::core::{AdvancedLashCoreBuilder, LashCore, LashCoreBuilder};
pub use crate::error::{EmbedError, Result};
pub use crate::mode::{ModeId, ModePreset, ModelSelection};
pub use crate::plugin_binding::PluginBinding;
pub use crate::session::{
    LashSession, ObservableSession, QueueInputBuilder, SessionBuilder, SessionConfigPatch,
};
pub use crate::turn::{
    FollowedTurnResult, TurnActivityFanout, TurnBuilder, TurnOutput, TurnResult, TurnStream,
    message_role, message_text,
};
pub use lash_core::{
    InputItem, PluginStack, SessionSpec, TurnActivity, TurnActivityId, TurnActivitySink, TurnEvent,
    TurnInput,
};

pub mod prelude {
    pub use crate::{
        AdvancedLashCoreBuilder, AdvancedToolsControl, BackgroundTasks, EmbedError,
        FollowedTurnResult, Handoffs, InputItem, LashCore, LashCoreBuilder, LashSession, ModeId,
        ModePreset, ObservableSession, PluginActions, PluginBinding, PluginStack, Result,
        SessionBuilder, SessionSpec, ToolsControl, TurnActivity, TurnBuilder, TurnEvent, TurnInput,
        TurnOutput, TurnResult, TurnStream,
    };
    pub use lash_core::TurnActivitySink;
}

pub mod tools {
    pub use crate::ToolState;
    pub use lash_core::{
        AckWakeArgs, MonitorAckWakeOp, MonitorEmptyArgs, MonitorRegisterSpecsOp, MonitorRunState,
        MonitorSnapshot, MonitorSpec, MonitorStartOp, MonitorStatus, MonitorStatusOp,
        MonitorStopOp, MonitorTakeUpdatesOp, MonitorUpdateBatch, RegisterSpecsArgs,
        StartMonitorArgs, StopMonitorArgs, ToolActivation, ToolArgumentProjectionPolicy,
        ToolAvailability, ToolAvailabilityConfig, ToolCall, ToolCallOutput, ToolCallRecord,
        ToolContext, ToolContract, ToolDefinition, ToolDiscoveryMetadata, ToolExecutionMode,
        ToolManifest, ToolOutputContract, ToolProvider, ToolResult, ToolSourceHandle,
    };
}

pub mod direct {
    pub use lash_core::llm::types::{
        LlmAttachment, LlmEventSender, LlmOutputPart, LlmTerminalReason, LlmUsage,
    };
    pub use lash_core::{
        DirectCompletion, DirectJsonSchema, DirectLlmClient, DirectLlmCompletion, DirectLlmError,
        DirectMessage, DirectOutputSpec, DirectPart, DirectRequest, DirectRole, LlmResponse,
        TokenUsage,
    };
}

pub mod persistence {
    pub use lash_core::{
        AttachmentStore, BlobRef, GcReport, GraphCommitDelta, HydratedSessionCheckpoint, ModeEvent,
        PersistedSessionConfig, PersistedSessionRead, PersistedSessionState, PersistedTurnState,
        RuntimeCommit, RuntimeCommitResult, RuntimePersistence, SessionCheckpoint,
        SessionEventRecord, SessionGraph, SessionHead, SessionHeadMeta, SessionMeta,
        SessionNodeRecord, SessionReadScope, SessionReadView, SessionStateEnvelope,
        SessionStoreCreateRequest, SessionStoreFactory, StoreError, TokenLedgerEntry, VacuumReport,
        load_persisted_session_state, load_persisted_session_state_active_path,
    };
}

pub mod plugins {
    pub use crate::plugin_binding::PluginBinding;
    pub use lash_core::PluginDirective;
    pub use lash_core::plugin::{
        AfterToolCallHook, AfterTurnHook, AssistantResponseHook, AssistantResponseHookContext,
        AssistantResponseTransform, AssistantStreamHook, AssistantStreamHookContext,
        AssistantStreamTransform, BeforeToolCallHook, BeforeTurnHook, CheckpointHook,
        CheckpointHookContext, PluginSpecBuilder, StaticPluginFactory, ToolCallHookContext,
        ToolResultHookContext,
    };
    pub use lash_core::{
        BuiltinMonitorToolPluginFactory, BuiltinTaskControlsPluginFactory, PluginError,
        PluginFactory, PluginHost, PluginMessage, PluginRegistrar, PluginRuntimeEvent,
        PluginSession, PluginSessionContext, PluginSpec, PluginSpecFactory, PromptHookContext,
        SessionPlugin, ToolOutputBudgetConfig, ToolOutputBudgetMode, ToolOutputBudgetPluginFactory,
        ToolSurfaceContribution, ToolSurfaceOverride, TurnHookContext, TurnResultHookContext,
    };
}

pub mod modes {
    pub use crate::mode::{ModeId, ModePreset};
}

pub mod messages {
    pub use lash_core::MessageRole;
}

pub mod advanced {
    pub use crate::AdvancedLashCoreBuilder;
    pub use lash_core::runtime::{RuntimeTurnPhase, RuntimeTurnPhaseProbe};
    // Benchmarks and diagnostics still need a semantic harness facade for
    // preloaded state, event capture, plugin-stack presets, and graph seeding.
    // Do not expose runtime bridge internals here to fill that gap.
    pub use lash_core::{
        AssembledTurn, BackgroundTaskRecord, BackgroundTaskRegistry, DirectCompletionClient,
        DirectRequestSpec, EffectInvocationMetadata, EffectOrigin, EmbeddedRuntimeHost, EventSink,
        ExecutionMode, InlineRuntimeEffectController, LashRuntime, LlmAttachmentSpec,
        LlmRequestSpec, LocalBackgroundTaskRegistry, ModeSessionExtensionHandle, ModeTurnOptions,
        NoopEventSink, NoopTurnActivitySink, PersistentRuntimeServices, PluginMessage, Residency,
        RewriteTrigger, RuntimeCoreConfig, RuntimeEffectCommand, RuntimeEffectController,
        RuntimeEffectControllerError, RuntimeEffectControllerScope, RuntimeEffectEnvelope,
        RuntimeEffectKind, RuntimeEffectLocalExecutor, RuntimeEffectOutcome, RuntimeEnvironment,
        RuntimeEnvironmentBuilder, RuntimeError, RuntimeHandle, RuntimeObservation, SessionEvent,
        SessionHandle, SessionPolicy, SessionStateEnvelope, SessionTurnHandle,
        StandardContextApproach, TerminationPolicy, TurnContext, TurnFinish, TurnOutcome, TurnStop,
    };
}

pub mod prompt {
    pub use lash_core::{
        PromptBuiltin, PromptContribution, PromptContributionGate, PromptLayer, PromptSlot,
        PromptTemplate, PromptTemplateEntry, PromptTemplateSection, default_prompt_template,
    };
}

pub mod tracing {
    pub use lash_core::{
        JsonlTraceSink, TraceAttachment, TraceContentBlock, TraceError, TraceEvent,
        TraceLlmMessage, TraceLlmRequest, TraceLlmResponse, TracePromptComponent,
        TraceProviderStreamEvent, TraceRecord, TraceRuntimeStreamEvent, TraceSinkError,
        TraceTokenUsage, TraceToolSpec,
    };
    pub use lash_trace::{TraceContext, TraceLevel, TraceSink};
}

pub mod provider {
    pub use lash_core::provider::{
        ProviderRateLimitPolicy, ProviderReliability, ProviderReliabilityBuilder,
        ProviderRetryPolicy, ProviderTimeoutPolicy,
    };
    pub use lash_core::{
        AgentModelSelection, LlmTimeouts, ProviderComponents, ProviderFactory, ProviderHandle,
        ProviderModelPolicy, ProviderOptions, ProviderRegistry, ProviderSpec, ProviderState,
        ProviderThinkingPolicy, ProviderTransport, RequestTimeout, StaticModelPolicy,
        VariantRequestConfig, build_provider, provider_factory, register_provider_factory,
    };
}

pub mod model_info {
    pub use lash_core::model_info::bundled_models_dev_snapshot;
    pub use lash_core::{ModelCatalog, ModelInfo, ResolvedModelSpec};
}

pub type ToolState = lash_core::ToolState;
