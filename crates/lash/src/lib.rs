//! App-facing embedding facade for Lash.
//!
//! `lash` is intentionally a small layer above the lower-level
//! `lash-core` runtime crate. Host applications own providers, persistence,
//! app state, HTTP protocols, auth, and frontend streaming; this crate
//! owns only the ergonomic core/session/turn API.
//!
//! Every public name has exactly one home. The crate root carries the daily
//! core/session/turn path; each domain module ([`tools`], [`persistence`],
//! [`plugins`], [`observe`], [`triggers`], ...) carries its own
//! vocabulary. [`prelude`] mirrors the crate root exactly.

pub mod admin;
mod core;
mod error;
mod plugin_binding;
mod prompt_layer;
#[cfg(feature = "rlm")]
pub mod rlm;
pub mod scenario_contracts;
mod session;
mod support;
#[cfg(all(test, feature = "rlm"))]
mod tests;
pub mod turn;
pub mod usage;

pub use crate::admin::{
    AdvancedToolAdmin, Completions, CoreTriggerAdmin, PluginOperations, SessionCommandAdmin,
    SessionTriggerAdmin, ToolAdmin,
};
pub use crate::core::{LashCore, LashCoreBuilder, SessionDeleteReport};
pub use crate::error::{EmbedError, Result};
pub use crate::plugin_binding::PluginBinding;
pub use crate::prompt_layer::PromptLayerSink;
pub use crate::session::{
    EnqueueTurnBuilder, LashSession, ObservableSession, SessionBuilder, SessionConfigPatch,
};
pub use crate::turn::{
    QueuedTurnBuilder, TurnActivityFanout, TurnBuilder, TurnOutput, TurnResult, TurnStream,
    message_role, message_text,
};
pub use lash_core::{
    AwaitEventKey, AwaitEventWaitIdentity, ExecutionSummary, ExternalCompletionError, InputItem,
    ModelLimits, ModelSpec, PendingTurnInput, PendingTurnInputCancelOutcome,
    PendingTurnInputCancelResult, PendingTurnInputCancelTarget,
    PendingTurnInputSuffixCancelOutcome, PluginStack, Resolution, ResolveOutcome, SessionCommand,
    SessionCommandReceipt, SessionCreateRequest, SessionSpec, SessionStartPoint, TurnActivity,
    TurnActivityId, TurnActivitySink, TurnCause, TurnEvent, TurnFinish, TurnInput, TurnOutcome,
    TurnStop,
};
/// Cooperative cancellation handle accepted by
/// [`TurnBuilder::cancel`](crate::TurnBuilder::cancel); re-exported so
/// embedders cancel turns without depending on `tokio-util` themselves.
pub use tokio_util::sync::CancellationToken;

/// The prelude is exactly the crate root: `use lash::prelude::*;` brings in
/// the daily core/session/turn vocabulary and nothing from the domain
/// modules.
pub mod prelude {
    pub use crate::{
        AdvancedToolAdmin, CoreTriggerAdmin, EmbedError, EnqueueTurnBuilder, ExecutionSummary,
        InputItem, LashCore, LashCoreBuilder, LashSession, ModelLimits, ModelSpec,
        ObservableSession, PendingTurnInputCancelOutcome, PluginBinding, PluginOperations,
        PluginStack, PromptLayerSink, QueuedTurnBuilder, Result, SessionBuilder, SessionCommand,
        SessionCommandAdmin, SessionCommandReceipt, SessionConfigPatch, SessionCreateRequest,
        SessionDeleteReport, SessionSpec, SessionStartPoint, SessionTriggerAdmin, ToolAdmin,
        TurnActivity, TurnActivityFanout, TurnActivityId, TurnActivitySink, TurnBuilder, TurnCause,
        TurnEvent, TurnFinish, TurnInput, TurnOutcome, TurnOutput, TurnResult, TurnStop,
        TurnStream, message_role, message_text,
    };
}

/// Session observation: cursors, resumable event streams, and live replay
/// recovery for host frontends. Entry point: [`LashSession::observe`] /
/// [`ObservableSession`].
pub mod observe {
    pub use crate::session::{
        RemoteSessionObservationEventStream, RemoteSessionObservationStream,
        RemoteSessionObservationStreamItem, RemoteSessionObservationSubscription,
        SessionObservationStream, SessionObservationStreamItem,
    };
    pub use lash_core::{
        InMemoryLiveReplayStore, InMemoryLiveReplayStoreConfig, LiveReplayGap, LiveReplayGapReason,
        LiveReplayStore, SessionCursor, SessionObservation, SessionObservationEvent,
        SessionObservationEventPayload, SessionObservationSubscription, SessionProcessEventKind,
        SessionQueueEventKind, SessionResume, SessionRevision,
    };
}

/// Triggers and subscriptions: declaring event sources, emitting occurrences,
/// and inspecting trigger subscriptions. Entry points:
/// [`LashCore::triggers`] and [`LashSession::triggers`].
pub mod triggers {
    pub use lash_core::{
        LashSchema, TriggerEmitReport, TriggerEvent, TriggerEventType, TriggerOccurrenceRequest,
        TriggerRegistration, TriggerStore, TriggerSubscriptionDraft, TriggerSubscriptionFilter,
        TriggerSubscriptionRecord, TriggerTargetSummary, empty_trigger_source_key,
    };
}

pub mod tools {
    pub use lash_core::{
        CancelHint, PendingCompletion, PreparedToolCall, TimeoutBehavior, ToolActivation,
        ToolArgumentProjectionPolicy, ToolCall, ToolCallOutput, ToolCallRecord, ToolContext,
        ToolContract, ToolDefinition, ToolDurableEffects, ToolExecutionGrant, ToolManifest,
        ToolOutputContract, ToolPrepareCall, ToolPrepareContext, ToolProvider, ToolResult,
        ToolScheduling, ToolSourceHandle, ToolTriggerClient,
    };
    pub use lash_core::{
        PLUGIN_TOOL_SOURCE_ID, ToolId, ToolRestoreReport, ToolState, ToolStateEntry,
    };
    #[cfg(feature = "rlm")]
    pub use lash_lashlang_runtime::{
        CataloguePreviewEntry, CataloguePreviewOptions, DEFAULT_CATALOGUE_PREVIEW_CALL_NAME_LIMIT,
        DEFAULT_CATALOGUE_PREVIEW_MODULE_LIMIT, LASHLANG_TOOL_BINDING_KEY, LashlangToolBinding,
        RemoteToolGrantLashlangExt, ToolDefinitionLashlangExt, ToolManifestLashlangExt,
        catalogue_preview_contribution, catalogue_preview_contribution_for_entries,
        catalogue_preview_contribution_for_entries_with_options,
        catalogue_preview_contribution_for_manifests, catalogue_preview_contribution_with_options,
        catalogue_preview_entries_from_catalog_records, catalogue_preview_entries_from_manifests,
        catalogue_preview_entry_from_catalog_record, catalogue_preview_entry_from_manifest,
    };
    #[cfg(feature = "rlm")]
    pub use lash_lashlang_runtime::{
        DeferredResolutionRecord, DeferredToolResolver, Resolution as DeferredToolResolution,
        SharedDeferredToolResolver, ToolGrant as DeferredToolGrant,
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
    };
}

pub mod persistence {
    pub use lash_core::FileAttachmentStore;
    pub use lash_core::runtime::{
        DeliveryPolicy, InMemorySessionStore, InMemorySessionStoreFactory, MergeKey,
        PendingTurnInputClaimDiagnostics, PendingTurnInputDraft, QueuedWorkBatch,
        QueuedWorkBatchDraft, QueuedWorkClaim, QueuedWorkClaimBoundary, QueuedWorkClass,
        QueuedWorkCompletion, QueuedWorkItem, QueuedWorkPayload, RuntimeSessionState,
        SessionStoreCreateRequest, SessionStoreFactory, SlotPolicy, TurnInputClaim,
    };
    pub use lash_core::store::queued_work;
    pub use lash_core::store::{
        GraphCommitDelta, HydratedSessionCheckpoint, PersistedSessionRead, RuntimeCommit,
        RuntimeCommitResult, RuntimeTurnCommitStamp, SessionCheckpoint, SessionHead,
        SessionHeadMeta, load_persisted_session_state, load_persisted_session_state_active_path,
    };
    pub use lash_core::{
        AttachmentStore, InMemoryAttachmentStore, InMemoryProcessExecutionEnvStore,
        ProcessExecutionEnvStore,
    };
    pub use lash_core::{
        BlobRef, GcReport, LeaseOwnerIdentity, LeaseOwnerLiveness, PersistedSessionConfig,
        PersistedTurnState, ProtocolEvent, RuntimePersistence, SessionEventRecord,
        SessionExecutionLease, SessionExecutionLeaseClaimOutcome, SessionExecutionLeaseCompletion,
        SessionExecutionLeaseFence, SessionGraph, SessionMeta, SessionNodeRecord, SessionReadScope,
        SessionReadView, SessionRelation, StoreError, VacuumReport,
    };
    #[cfg(feature = "rlm")]
    pub use lash_lashlang_runtime::{InMemoryLashlangArtifactStore, LashlangArtifactStore};
}

pub mod plugins {
    pub use lash_core::PluginDirective;
    pub use lash_core::PluginOptions;
    pub use lash_core::plugin::{
        AfterToolCallHook, AfterTurnHook, AssistantResponseHook, AssistantResponseHookContext,
        AssistantResponseTransform, AssistantStreamHook, AssistantStreamHookContext,
        AssistantStreamTransform, BeforeToolCallHook, BeforeTurnHook, CheckpointHook,
        CheckpointHookContext, CompactionContext, ContextCompaction, ContextCompactor,
        ContextError, PluginExtensionContribution, PluginSpecBuilder, StaticPluginFactory,
        ToolCallHookContext, ToolResultHookContext,
    };
    pub use lash_core::{
        PluginError, PluginFactory, PluginHost, PluginMessage, PluginRegistrar, PluginRuntimeEvent,
        PluginSession, PluginSessionContext, PluginSpec, PluginSpecFactory, PromptHookContext,
        SessionPlugin, ToolCatalogContribution, TurnHookContext, TurnResultHookContext,
    };
    pub use lash_plugin_tool_output_budget::{
        ToolOutputBudgetConfig, ToolOutputBudgetMode, ToolOutputBudgetPluginFactory,
        tool_output_budget_stack as runtime_plugin_stack,
    };
}

pub mod messages {
    pub use lash_core::{Message, MessageRole};
}

pub mod remote {
    pub use lash_remote_protocol::{
        REMOTE_PROTOCOL_VERSION, RemoteAssistantOutput, RemoteAssistantOutputState,
        RemoteAttachmentRef, RemoteCausalRef, RemoteDiagnostic, RemoteExecutionSummary,
        RemoteGenerationOptions, RemoteInputItem, RemoteLiveReplayGap, RemoteLiveReplayGapReason,
        RemoteLlmAttachment, RemoteLlmContentBlock, RemoteLlmMessage, RemoteLlmOutputPart,
        RemoteLlmOutputSpec, RemoteLlmRequest, RemoteLlmRequestScope, RemoteLlmResponse,
        RemoteLlmRole, RemoteLlmTerminalReason, RemoteLlmToolChoice, RemoteLlmToolSpec,
        RemoteModelIntent, RemoteObservedProcess, RemoteObservedProcessEvent,
        RemotePersistProcessEnvRequest, RemotePersistProcessEnvResult, RemoteProcessAwaitOutput,
        RemoteProcessAwaitRequest, RemoteProcessAwaitResult, RemoteProcessCancelRequest,
        RemoteProcessCancelResult, RemoteProcessDefinitionIdentity, RemoteProcessEvent,
        RemoteProcessEventSemantics, RemoteProcessEventSemanticsSpec, RemoteProcessEventType,
        RemoteProcessEventsRequest, RemoteProcessEventsResponse, RemoteProcessExecutionEnvRef,
        RemoteProcessExecutionEnvSpec, RemoteProcessExecutionPolicy, RemoteProcessExternalRef,
        RemoteProcessHandleDescriptor, RemoteProcessInput, RemoteProcessLifecycleStatus,
        RemoteProcessListFilter, RemoteProcessListResponse, RemoteProcessModelLimits,
        RemoteProcessModelSpec, RemoteProcessOriginator, RemoteProcessPluginOptions,
        RemoteProcessProvenance, RemoteProcessSignalRequest, RemoteProcessSignalResult,
        RemoteProcessStartGrant, RemoteProcessStartRequest, RemoteProcessStartResult,
        RemoteProcessStatus, RemoteProcessStatusFilter, RemoteProcessSummary,
        RemoteProcessTerminalSemantics, RemoteProcessTerminalSpec, RemoteProcessTerminalState,
        RemoteProcessValueSelector, RemoteProcessWaitKind, RemoteProcessWaitState,
        RemoteProcessWake, RemoteProcessWakeDedupeKey, RemoteProcessWakeSpec,
        RemoteProcessWorkItem, RemoteProcessWorkSnapshot, RemotePromptBuiltin,
        RemotePromptContribution, RemotePromptContributionGate, RemotePromptLayer,
        RemotePromptSlot, RemotePromptSlotLayer, RemotePromptTemplate, RemotePromptTemplateEntry,
        RemotePromptTemplateSection, RemoteProtocolError, RemoteProtocolTurnOptions,
        RemoteProviderMetadata, RemoteProviderReasoningReplay, RemoteProviderReplayMeta,
        RemoteResponseTextMeta, RemoteRuntimeEffectKind, RemoteRuntimeInvocation,
        RemoteRuntimeReplay, RemoteRuntimeScope, RemoteRuntimeSubject,
        RemoteSchemaProjectionOverride, RemoteSessionCursor, RemoteSessionObservation,
        RemoteSessionObservationEvent, RemoteSessionObservationEventPayload,
        RemoteSessionProcessEventKind, RemoteSessionQueueEventKind, RemoteSessionScope,
        RemoteTokenLedgerEntry, RemoteToolActivation, RemoteToolArgumentProjectionPolicy,
        RemoteToolCallOutcome, RemoteToolCallSummary, RemoteToolFailureClass, RemoteToolGrant,
        RemoteToolOutputContract, RemoteToolRegistry, RemoteToolRetryPolicy, RemoteToolScheduling,
        RemoteTriggerCancelSubscriptionRequest, RemoteTriggerCancelSubscriptionResult,
        RemoteTriggerEmitReport, RemoteTriggerInputBinding, RemoteTriggerInputTemplate,
        RemoteTriggerListSubscriptionsResponse, RemoteTriggerOccurrenceRecord,
        RemoteTriggerOccurrenceRequest, RemoteTriggerRegisterSubscriptionRequest,
        RemoteTriggerRegisterSubscriptionResult, RemoteTriggerRegistration,
        RemoteTriggerSubscriptionDraft, RemoteTriggerSubscriptionFilter,
        RemoteTriggerSubscriptionRecord, RemoteTriggerTargetSummary, RemoteTurnActivity,
        RemoteTurnEvent, RemoteTurnFinish, RemoteTurnInput, RemoteTurnIssue, RemoteTurnOutcome,
        RemoteTurnRequest, RemoteTurnResult, RemoteTurnStatus, RemoteTurnStop,
        RemoteTurnUsageSummary, RemoteUsage, assert_remote_tool_registry_reopenable,
        ensure_protocol_version,
    };
}

pub mod process {
    pub use crate::admin::{Processes, SessionProcessAdmin};
    pub use lash_core::{
        ObservedProcess, ObservedProcessEvent, ObservedWorkItem, ProcessAwaitOutput,
        ProcessCancelAbility, ProcessCancelAllRequest, ProcessCancelRequest, ProcessCancelSource,
        ProcessCancelSummary, ProcessEvent, ProcessEventAppendRequest, ProcessEventAppendResult,
        ProcessEventType, ProcessExecutionContext, ProcessExecutionEnvRef, ProcessExecutionEnvSpec,
        ProcessExternalRef, ProcessHandleDescriptor, ProcessHandleGrant, ProcessHandleSummary,
        ProcessIdentity, ProcessInput, ProcessLease, ProcessLeaseClaimOutcome,
        ProcessLeaseCompletion, ProcessLifecycleStatus, ProcessListFilter, ProcessListMode,
        ProcessOpScope, ProcessProvenance, ProcessRecord, ProcessRegistration, ProcessRegistry,
        ProcessRunHandle, ProcessRuntimeHost, ProcessService, ProcessSessionDeleteReport,
        ProcessStartOptions, ProcessStartRequest, ProcessStatus, ProcessStatusFilter,
        ProcessTerminalState, ProcessWake, ProcessWakeDedupeKey, ProcessWakeDelivery,
        ProcessWakeSpec, ProcessWorkDriver, ProcessWorkObserver, ProcessWorkSnapshot, SessionScope,
        SessionScopeId,
    };
    #[cfg(feature = "rlm")]
    pub use lash_lashlang_runtime::{
        LASHLANG_ENGINE_KIND, LashlangProcessInput, lashlang_process_event_types,
        lashlang_process_signal_event_types,
    };
}

pub mod durability {
    pub use lash_core::{
        DurableProcessWorker, DurableProcessWorkerConfig, EffectHost, InlineEffectHost,
        LeaseTimings, LeaseTimingsError, Residency, RuntimeEnvironment, RuntimeHostConfig,
        TerminationPolicy,
    };
}

pub mod runtime {
    pub use crate::core::AdvancedLashCoreBuilder;
    pub use lash_core::runtime::{
        AssembledTurn, AwaitEventResolver, DirectCompletionClient, EmbeddedRuntimeHost, EventSink,
        ExecutionScope, InlineRuntimeEffectController, LashRuntime, LlmAttachmentSpec,
        LlmRequestSpec, NoopEventSink, NoopTurnActivitySink, ProcessCommand, ProcessEffectOutcome,
        QueuedWorkDriver, QueuedWorkRunHandle, QueuedWorkRunRequest, RuntimeEffectCommand,
        RuntimeEffectController, RuntimeEffectControllerError, RuntimeEffectEnvelope,
        RuntimeEffectKind, RuntimeEffectLocalExecutor, RuntimeEffectOutcome,
        RuntimeEnvironmentBuilder, RuntimeError, RuntimeErrorCode, RuntimeHandle,
        RuntimeInvocation, RuntimeObservation, RuntimeScope, RuntimeTurnPhase,
        RuntimeTurnPhaseProbe, ScopedEffectController, TurnContext,
    };
    pub use lash_core::{
        PersistentRuntimeServices, ProtocolSessionExtensionHandle, ProtocolTurnOptions,
        SessionHandle, SessionPolicy, SessionSnapshot, render_turn_causes_prompt,
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
        TraceEvent, TraceLabelMetadata, TraceLlmMessage, TraceLlmRequest, TraceLlmResponse,
        TracePromptComponent, TraceProviderStreamEvent, TraceRecord, TraceRuntimeScope,
        TraceRuntimeStreamEvent, TraceRuntimeSubject, TraceSinkError, TraceTokenUsage,
        TraceToolSpec,
    };
    #[cfg(feature = "otel-trace")]
    pub use lash_core::{OtelTraceOptions, OtelTraceSink};
    #[cfg(feature = "rlm")]
    pub use lash_lashlang_runtime::{
        TraceLashlangChildExecution, TraceLashlangEdgeSelection, TraceLashlangExecutionEvent,
        TraceLashlangExecutionIdentity, TraceLashlangGraph, TraceLashlangGraphChildLink,
        TraceLashlangGraphEdge, TraceLashlangGraphNode, TraceLashlangGraphStore, TraceLashlangMap,
        TraceLashlangMapEdge, TraceLashlangMapNode, TraceLashlangNodeStatus, TraceLashlangStatus,
    };
    pub use lash_trace::{StderrTraceSink, TeeTraceSink, TraceContext, TraceLevel, TraceSink};
}

/// Test helpers for embedders. Enable with `lash = { ..., features = ["testing"] }`
/// to script model responses in integration tests without a live provider.
#[cfg(all(any(test, feature = "testing"), feature = "rlm"))]
pub mod testing;

pub mod provider {
    pub use lash_core::provider::{
        ProviderRateLimitPolicy, ProviderReliability, ProviderRetryPolicy,
    };
    pub use lash_core::{
        LlmTimeouts, Provider, ProviderComponents, ProviderFactory, ProviderHandle,
        ProviderModelPolicy, ProviderOptions, ProviderSpec, RequestTimeout, StaticModelPolicy,
    };
}
