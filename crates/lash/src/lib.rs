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
mod prompt_layer;
mod session;
mod support;
#[cfg(test)]
mod tests;
pub mod turn;
pub mod usage;

pub use crate::control::{
    AdvancedToolsControl, HostEventsControl, PluginActions, ProcessControl, ToolsControl,
    TriggersControl,
};
pub use crate::core::{AdvancedLashCoreBuilder, LashCore, LashCoreBuilder, SessionDeleteReport};
pub use crate::error::{EmbedError, Result};
pub use crate::mode::{ModeId, ModePreset};
pub use crate::plugin_binding::PluginBinding;
pub use crate::prompt_layer::PromptLayerSink;
pub use crate::session::{
    LashSession, ObservableSession, QueueInputBuilder, SessionBuilder, SessionConfigPatch,
};
pub use crate::turn::{
    AdvancedTurn, QueuedTurnBuilder, TurnActivityFanout, TurnBuilder, TurnOutput, TurnResult,
    TurnStream, message_role, message_text,
};
pub use lash_core::{
    HostEvent, HostEventEmitReport, InputItem, ModelLimits, ModelSpec, PluginStack,
    ProcessCancelSummary, ProcessDefinitionSummary, ProcessHandleDescriptor, ProcessHandleSummary,
    ProcessInput, ProcessLifecycleStatus, ProcessListFilter, ProcessStartRequest,
    ProcessStatusFilter, SessionSpec, TriggerRegistration, TriggerSourceType, TriggerTargetSummary,
    TurnActivity, TurnActivityId, TurnActivitySink, TurnEvent, TurnInput,
    impl_unsupported_queued_work_methods,
};

pub mod prelude {
    pub use crate::{
        AdvancedLashCoreBuilder, AdvancedToolsControl, EmbedError, HostEvent, HostEventEmitReport,
        HostEventsControl, InputItem, LashCore, LashCoreBuilder, LashSession, ModeId, ModePreset,
        ModelSpec, ObservableSession, PluginActions, PluginBinding, PluginStack,
        ProcessCancelSummary, ProcessControl, ProcessDefinitionSummary, ProcessHandleDescriptor,
        ProcessHandleSummary, ProcessInput, ProcessLifecycleStatus, ProcessListFilter,
        ProcessStartRequest, ProcessStatusFilter, PromptLayerSink, QueuedTurnBuilder, Result,
        SessionBuilder, SessionSpec, ToolsControl, TriggerRegistration, TriggerSourceType,
        TriggerTargetSummary, TriggersControl, TurnActivity, TurnBuilder, TurnEvent, TurnInput,
        TurnOutput, TurnResult, TurnStream,
    };
    pub use lash_core::TurnActivitySink;
}

pub mod tools {
    pub use crate::ToolState;
    pub use lash_core::{
        PreparedToolCall, ToolActivation, ToolAgentSurface, ToolArgumentProjectionPolicy,
        ToolAvailability, ToolAvailabilityConfig, ToolCall, ToolCallOutput, ToolCallRecord,
        ToolContext, ToolContract, ToolDefinition, ToolManifest, ToolOutputContract,
        ToolPrepareCall, ToolPrepareContext, ToolProvider, ToolResult, ToolScheduling,
        ToolSourceHandle,
    };
    /// Author a fixed-tool provider without hand-rolling `tool_manifests` /
    /// `resolve_contract`: supply the [`ToolDefinition`]s once and an
    /// [`StaticToolExecute`] for behavior.
    pub use lash_tool_support::{StaticToolExecute, StaticToolProvider};
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
    pub use lash_core::LashlangArtifactStore;
    pub use lash_core::runtime::ProcessHandleGrantEntry;
    pub use lash_core::runtime::{
        DeliveryPolicy, InMemorySessionStore, InMemorySessionStoreFactory, MergeKey,
        QueuedWorkBatch, QueuedWorkBatchDraft, QueuedWorkClaim, QueuedWorkClaimBoundary,
        QueuedWorkCompletion, QueuedWorkItem, QueuedWorkPayload, RuntimeSessionState,
        SessionStoreCreateRequest, SessionStoreFactory, SlotPolicy,
    };
    pub use lash_core::store::{
        GraphCommitDelta, HydratedSessionCheckpoint, PersistedSessionRead, RuntimeCommit,
        RuntimeCommitResult, RuntimeTurnCommitStamp, SessionCheckpoint, SessionHead,
        SessionHeadMeta, load_persisted_session_state, load_persisted_session_state_active_path,
    };
    pub use lash_core::{
        AttachmentStore, BlobRef, GcReport, PersistedSessionConfig, PersistedTurnState,
        ProcessAwaitOutput, ProcessEvent, ProcessEventAppendRequest, ProcessEventAppendResult,
        ProcessEventType, ProcessExternalRef, ProcessHandleDescriptor, ProcessHandleGrant,
        ProcessLease, ProcessLeaseCompletion, ProcessRecord, ProcessRegistration, ProcessRegistry,
        ProcessScope, ProcessScopeId, ProcessSessionDeleteReport, ProcessTerminalState,
        ProcessWake, ProcessWakeDedupeKey, ProcessWakeDelivery, ProcessWakeSpec, ProtocolEvent,
        RuntimePersistence, SessionEventRecord, SessionGraph, SessionMeta, SessionNodeRecord,
        SessionReadScope, SessionReadView, SessionRelation, SessionSnapshot, StoreError,
        TokenLedgerEntry, VacuumReport,
    };
    pub use lash_local_store::FileAttachmentStore;
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
        HostEvent, PluginError, PluginFactory, PluginHost, PluginMessage, PluginRegistrar,
        PluginRuntimeEvent, PluginSession, PluginSessionContext, PluginSpec, PluginSpecFactory,
        PromptHookContext, SessionPlugin, ToolSurfaceContribution, ToolSurfaceOverride,
        TurnHookContext, TurnResultHookContext,
    };
    pub use lash_plugin_tool_output_budget::{
        ToolOutputBudgetConfig, ToolOutputBudgetMode, ToolOutputBudgetPluginFactory,
        tool_output_budget_stack as runtime_plugin_stack,
    };
}

pub mod modes {
    pub use lash_protocol_rlm::{
        LashlangAbilities, LashlangLanguageFeatures, LashlangSurface, NamedDataType,
        ResourceCatalog, RlmProtocolPluginConfig, TypeExpr, TypeField, format_type_expr,
    };

    pub use crate::mode::{ModeId, ModePreset};
}

pub mod messages {
    pub use lash_core::MessageRole;
}

pub mod advanced {
    pub use crate::AdvancedLashCoreBuilder;
    pub use lash_core::runtime::{
        AssembledTurn, DirectCompletionClient, DurableProcessWorker, DurableProcessWorkerConfig,
        EffectHost, EffectScope, EmbeddedRuntimeHost, EventSink, InlineEffectHost,
        InlineRuntimeEffectController, LashRuntime, LlmAttachmentSpec, LlmRequestSpec,
        NoopEventSink, NoopTurnActivitySink, ProcessCancelAbility, ProcessCancelAllRequest,
        ProcessCancelRequest, ProcessCancelSource, ProcessCancelSummary, ProcessCommand,
        ProcessDefinitionSummary, ProcessEffectOutcome, ProcessHandleDescriptor,
        ProcessHandleGrant, ProcessHandleSummary, ProcessInput, ProcessLifecycleStatus,
        ProcessListFilter, ProcessListMode, ProcessOpScope, ProcessRecord, ProcessRegistration,
        ProcessRunHandle, ProcessScope, ProcessScopeId, ProcessService, ProcessSessionDeleteReport,
        ProcessStartOptions, ProcessStartRequest, ProcessStatus, ProcessStatusFilter,
        ProcessWakeDelivery, ProcessWorkPoke, ProcessWorkRunner, QueuedWorkPoke,
        QueuedWorkRunHandle, QueuedWorkRunOutcome, QueuedWorkRunRequest, QueuedWorkRunner,
        Residency, RuntimeEffectCommand, RuntimeEffectController, RuntimeEffectControllerError,
        RuntimeEffectEnvelope, RuntimeEffectKind, RuntimeEffectLocalExecutor, RuntimeEffectOutcome,
        RuntimeEnvironment, RuntimeEnvironmentBuilder, RuntimeError, RuntimeErrorCode,
        RuntimeHandle, RuntimeHostConfig, RuntimeInvocation, RuntimeObservation, RuntimeScope,
        RuntimeTurnPhase, RuntimeTurnPhaseProbe, ScopedEffectController, TerminationPolicy,
        TurnContext, lashlang_process_event_types, process_wake_input_from_event_payload,
        process_wake_turn_cause, process_wake_turn_text,
    };
    // Benchmarks and diagnostics still need a semantic harness facade for
    // preloaded state, event capture, plugin-stack presets, and graph seeding.
    // Do not expose runtime bridge internals here to fill that gap.
    pub use lash_core::{
        PersistentRuntimeServices, PluginMessage, ProtocolSessionExtensionHandle,
        ProtocolTurnOptions, RewriteTrigger, SessionEvent, SessionHandle, SessionPolicy,
        SessionSnapshot, TurnCause, TurnFinish, TurnOutcome, TurnStop, render_turn_causes_prompt,
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
        JsonlTraceSink, TraceAttachment, TraceBranchSelection, TraceContentBlock, TraceError,
        TraceEvent, TraceLabelMetadata, TraceLashlangChildExecution, TraceLashlangEdgeSelection,
        TraceLashlangExecutionEvent, TraceLashlangExecutionIdentity, TraceLashlangGraph,
        TraceLashlangGraphChildLink, TraceLashlangGraphEdge, TraceLashlangGraphNode,
        TraceLashlangGraphStore, TraceLashlangMap, TraceLashlangMapEdge, TraceLashlangMapNode,
        TraceLashlangNodeStatus, TraceLashlangStatus, TraceLlmMessage, TraceLlmRequest,
        TraceLlmResponse, TracePromptComponent, TraceProviderStreamEvent, TraceRecord,
        TraceRuntimeScope, TraceRuntimeStreamEvent, TraceRuntimeSubject, TraceSinkError,
        TraceTokenUsage, TraceToolSpec,
    };
    pub use lash_trace::{StderrTraceSink, TeeTraceSink, TraceContext, TraceLevel, TraceSink};
}

/// Test helpers for embedders. Enable with `lash = { ..., features = ["testing"] }`
/// to script model responses in integration tests without a live provider.
#[cfg(any(test, feature = "testing"))]
pub mod testing;

pub mod provider {
    pub use lash_core::provider::{
        ProviderRateLimitPolicy, ProviderReliability, ProviderReliabilityBuilder,
        ProviderRetryPolicy, ProviderTimeoutPolicy,
    };
    pub use lash_core::{
        LlmTimeouts, Provider, ProviderComponents, ProviderFactory, ProviderHandle,
        ProviderModelPolicy, ProviderOptions, ProviderSpec, ProviderThinkingPolicy, RequestTimeout,
        StaticModelPolicy,
    };
}

pub type ToolState = lash_core::ToolState;
