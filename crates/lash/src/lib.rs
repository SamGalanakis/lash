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
    AdvancedToolsControl, HostEventsControl, PluginActions, SessionCommandsControl, ToolsControl,
    TriggersControl,
};
pub use crate::core::{LashCore, LashCoreBuilder, SessionDeleteReport};
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
    HostEvent, HostEventEmitReport, HostEventOccurrenceRequest, InputItem, ModelLimits, ModelSpec,
    PluginStack, SessionCommand, SessionCommandReceipt, SessionCursor, SessionObservation,
    SessionObservationEvent, SessionObservationEventPayload, SessionObservationSubscription,
    SessionProcessEventKind, SessionQueueEventKind, SessionResume, SessionRevision, SessionSpec,
    TriggerRegistration, TriggerSourceType, TriggerSubscriptionFilter, TriggerTargetSummary,
    TurnActivity, TurnActivityId, TurnActivitySink, TurnEvent, TurnInput,
    empty_host_event_source_key, impl_unsupported_queued_work_methods,
};

pub mod prelude {
    pub use crate::{
        AdvancedToolsControl, EmbedError, HostEvent, HostEventEmitReport,
        HostEventOccurrenceRequest, HostEventsControl, InputItem, LashCore, LashCoreBuilder,
        LashSession, ModeId, ModePreset, ModelSpec, ObservableSession, PluginActions,
        PluginBinding, PluginStack, PromptLayerSink, QueuedTurnBuilder, Result, SessionBuilder,
        SessionCommand, SessionCommandReceipt, SessionCommandsControl, SessionCursor,
        SessionObservation, SessionObservationEvent, SessionObservationEventPayload,
        SessionObservationSubscription, SessionProcessEventKind, SessionQueueEventKind,
        SessionResume, SessionRevision, SessionSpec, ToolsControl, TriggerRegistration,
        TriggerSourceType, TriggerSubscriptionFilter, TriggerTargetSummary, TriggersControl,
        TurnActivity, TurnBuilder, TurnEvent, TurnInput, TurnOutput, TurnResult, TurnStream,
        empty_host_event_source_key,
    };
    pub use lash_core::TurnActivitySink;
}

pub mod tools {
    pub use crate::ToolState;
    pub use lash_core::{
        PreparedToolCall, ToolActivation, ToolAgentSurface, ToolArgumentProjectionPolicy,
        ToolAvailability, ToolAvailabilityConfig, ToolCall, ToolCallOutput, ToolCallRecord,
        ToolContext, ToolContract, ToolDefinition, ToolHostEventControl, ToolManifest,
        ToolOutputContract, ToolPrepareCall, ToolPrepareContext, ToolProvider, ToolResult,
        ToolScheduling, ToolSourceHandle,
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
    pub use lash_core::runtime::{
        DeliveryPolicy, InMemorySessionStore, InMemorySessionStoreFactory, MergeKey,
        QueuedWorkBatch, QueuedWorkBatchDraft, QueuedWorkClaim, QueuedWorkClaimBoundary,
        QueuedWorkCompletion, QueuedWorkItem, QueuedWorkPayload, RuntimeSessionState,
        SessionCommand, SessionCommandReceipt, SessionStoreCreateRequest, SessionStoreFactory,
        SlotPolicy,
    };
    pub use lash_core::store::{
        GraphCommitDelta, HydratedSessionCheckpoint, PersistedSessionRead, RuntimeCommit,
        RuntimeCommitResult, RuntimeTurnCommitStamp, SessionCheckpoint, SessionHead,
        SessionHeadMeta, load_persisted_session_state, load_persisted_session_state_active_path,
    };
    pub use lash_core::{AttachmentStore, InMemoryAttachmentStore};
    pub use lash_core::{
        BlobRef, GcReport, PersistedSessionConfig, PersistedTurnState, ProtocolEvent,
        RuntimePersistence, SessionEventRecord, SessionGraph, SessionMeta, SessionNodeRecord,
        SessionReadScope, SessionReadView, SessionRelation, SessionSnapshot, StoreError,
        TokenLedgerEntry, VacuumReport,
    };
    pub use lash_core::{InMemoryLashlangArtifactStore, LashlangArtifactStore};
    pub use lash_local_store::FileAttachmentStore;
}

pub mod plugins {
    pub use crate::plugin_binding::PluginBinding;
    pub use lash_core::PluginDirective;
    pub use lash_core::plugin::{
        AfterToolCallHook, AfterTurnHook, AssistantResponseHook, AssistantResponseHookContext,
        AssistantResponseTransform, AssistantStreamHook, AssistantStreamHookContext,
        AssistantStreamTransform, BeforeToolCallHook, BeforeTurnHook, CheckpointHook,
        CheckpointHookContext, CompactionContext, ContextCompaction, ContextCompactor,
        ContextError, PluginSpecBuilder, StaticPluginFactory, ToolCallHookContext,
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
    pub use crate::mode::{RlmSessionBuilderExt, RlmTurnBuilderExt};
    pub use lash_protocol_rlm::{
        LashlangAbilities, LashlangLanguageFeatures, LashlangSurface, NamedDataType,
        ResourceCatalog, RlmProtocolPluginConfig, TypeExpr, TypeField, format_type_expr,
    };
    pub use lash_rlm_types::RlmFinalAnswerFormat;

    pub use crate::mode::{ModeId, ModePreset};
}

pub mod messages {
    pub use lash_core::MessageRole;
}

pub mod remote {
    pub use lash_remote_protocol::*;
}

pub mod process {
    pub use crate::control::ProcessControl;
    pub use lash_core::{
        ObservedProcess, ObservedProcessEvent, ObservedWorkItem, ProcessAwaitOutput,
        ProcessCancelAbility, ProcessCancelAllRequest, ProcessCancelRequest, ProcessCancelSource,
        ProcessCancelSummary, ProcessDefinitionSummary, ProcessEvent, ProcessEventAppendRequest,
        ProcessEventAppendResult, ProcessEventType, ProcessExecutionContext, ProcessExternalRef,
        ProcessHandleDescriptor, ProcessHandleGrant, ProcessHandleSummary, ProcessInput,
        ProcessLease, ProcessLeaseCompletion, ProcessLifecycleStatus, ProcessListFilter,
        ProcessListMode, ProcessOpScope, ProcessRecord, ProcessRegistration, ProcessRegistry,
        ProcessRunHandle, ProcessRuntimeHost, ProcessScope, ProcessScopeId, ProcessService,
        ProcessSessionDeleteReport, ProcessStartOptions, ProcessStartRequest, ProcessStatus,
        ProcessStatusFilter, ProcessTerminalState, ProcessWake, ProcessWakeDedupeKey,
        ProcessWakeDelivery, ProcessWakeSpec, ProcessWorkDriver, ProcessWorkObserver,
        ProcessWorkPoke, ProcessWorkRunner, ProcessWorkSnapshot, lashlang_process_event_types,
    };
}

pub mod durability {
    pub use lash_core::{
        DurableProcessWorker, DurableProcessWorkerConfig, EffectHost, InlineEffectHost, Residency,
        RuntimeEnvironment, RuntimeHostConfig, TerminationPolicy,
    };
}

pub mod runtime {
    pub use crate::core::AdvancedLashCoreBuilder;
    pub use lash_core::runtime::{
        AssembledTurn, DirectCompletionClient, EffectScope, EmbeddedRuntimeHost, EventSink,
        InlineRuntimeEffectController, LashRuntime, LlmAttachmentSpec, LlmRequestSpec,
        NoopEventSink, NoopTurnActivitySink, ProcessCommand, ProcessEffectOutcome, QueuedWorkPoke,
        QueuedWorkRunHandle, QueuedWorkRunOutcome, QueuedWorkRunRequest, QueuedWorkRunner,
        RuntimeEffectCommand, RuntimeEffectController, RuntimeEffectControllerError,
        RuntimeEffectEnvelope, RuntimeEffectKind, RuntimeEffectLocalExecutor, RuntimeEffectOutcome,
        RuntimeEnvironmentBuilder, RuntimeError, RuntimeErrorCode, RuntimeHandle,
        RuntimeInvocation, RuntimeObservation, RuntimeScope, RuntimeTurnPhase,
        RuntimeTurnPhaseProbe, ScopedEffectController, SessionCommand, SessionCommandReceipt,
        SessionCursor, SessionObservation, SessionObservationEvent, SessionObservationEventPayload,
        SessionObservationSubscription, SessionResume, SessionRevision, TurnContext,
    };
    pub use lash_core::{
        PersistentRuntimeServices, PluginMessage, ProtocolSessionExtensionHandle,
        ProtocolTurnOptions, SessionEvent, SessionHandle, SessionPolicy, SessionSnapshot,
        TurnCause, TurnFinish, TurnOutcome, TurnStop, render_turn_causes_prompt,
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
