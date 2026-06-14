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
mod mode;
mod plugin_binding;
mod prompt_layer;
mod session;
mod support;
#[cfg(test)]
mod tests;
pub mod turn;
pub mod usage;

pub use crate::admin::{
    AdvancedToolAdmin, CoreTriggerAdmin, PluginActions, SessionCommandAdmin, SessionTriggerAdmin,
    ToolAdmin,
};
pub use crate::core::{LashCore, LashCoreBuilder, SessionDeleteReport};
pub use crate::error::{EmbedError, Result};
pub use crate::mode::{ModeId, ModePreset};
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
    InputItem, ModelLimits, ModelSpec, PluginStack, SessionCommand, SessionCommandReceipt,
    SessionSpec, TurnActivity, TurnActivityId, TurnActivitySink, TurnEvent, TurnInput,
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
        AdvancedToolAdmin, CoreTriggerAdmin, EmbedError, EnqueueTurnBuilder, InputItem, LashCore,
        LashCoreBuilder, LashSession, ModeId, ModePreset, ModelLimits, ModelSpec,
        ObservableSession, PluginActions, PluginBinding, PluginStack, PromptLayerSink,
        QueuedTurnBuilder, Result, SessionBuilder, SessionCommand, SessionCommandAdmin,
        SessionCommandReceipt, SessionConfigPatch, SessionDeleteReport, SessionSpec,
        SessionTriggerAdmin, ToolAdmin, TurnActivity, TurnActivityFanout, TurnActivityId,
        TurnActivitySink, TurnBuilder, TurnEvent, TurnInput, TurnOutput, TurnResult, TurnStream,
        message_role, message_text,
    };
}

/// Session observation: cursors, resumable event streams, and live replay
/// recovery for host frontends. Entry point: [`LashSession::observe`] /
/// [`ObservableSession`].
pub mod observe {
    pub use lash_core::{
        SessionCursor, SessionObservation, SessionObservationEvent, SessionObservationEventPayload,
        SessionObservationSubscription, SessionProcessEventKind, SessionQueueEventKind,
        SessionResume, SessionRevision,
    };
}

/// Triggers and triggers: declaring event sources, emitting occurrences,
/// and inspecting trigger subscriptions. Entry points:
/// [`LashCore::triggers`] and [`LashSession::triggers`].
pub mod triggers {
    pub use lash_core::{
        TriggerEmitReport, TriggerEvent, TriggerEventType, TriggerOccurrenceRequest,
        TriggerRegistration, TriggerSubscriptionFilter, TriggerTargetSummary,
        empty_trigger_source_key,
    };
}

pub mod tools {
    pub use lash_core::{
        LashlangToolBinding, PreparedToolCall, ToolActivation, ToolArgumentProjectionPolicy,
        ToolAvailability, ToolAvailabilityConfig, ToolCall, ToolCallOutput, ToolCallRecord,
        ToolContext, ToolContract, ToolDefinition, ToolManifest, ToolOutputContract,
        ToolPrepareCall, ToolPrepareContext, ToolProvider, ToolResult, ToolScheduling,
        ToolSourceHandle, ToolTriggerClient,
    };
    pub use lash_core::{ToolRestoreReport, ToolState, ToolStateEntry};
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
    pub use lash_core::FileAttachmentStore;
    pub use lash_core::runtime::{
        DeliveryPolicy, InMemorySessionStore, InMemorySessionStoreFactory, MergeKey,
        QueuedWorkBatch, QueuedWorkBatchDraft, QueuedWorkClaim, QueuedWorkClaimBoundary,
        QueuedWorkCompletion, QueuedWorkItem, QueuedWorkPayload, RuntimeSessionState,
        SessionStoreCreateRequest, SessionStoreFactory, SlotPolicy,
    };
    pub use lash_core::store::queued_work;
    pub use lash_core::store::{
        GraphCommitDelta, HydratedSessionCheckpoint, PersistedSessionRead, RuntimeCommit,
        RuntimeCommitResult, RuntimeTurnCommitStamp, SessionCheckpoint, SessionHead,
        SessionHeadMeta, load_persisted_session_state, load_persisted_session_state_active_path,
    };
    pub use lash_core::{AttachmentStore, InMemoryAttachmentStore};
    pub use lash_core::{
        BlobRef, GcReport, PersistedSessionConfig, PersistedTurnState, ProtocolEvent,
        RuntimePersistence, SessionEventRecord, SessionGraph, SessionMeta, SessionNodeRecord,
        SessionReadScope, SessionReadView, SessionRelation, StoreError, TokenLedgerEntry,
        VacuumReport,
    };
    pub use lash_core::{InMemoryLashlangArtifactStore, LashlangArtifactStore};
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
        PluginError, PluginFactory, PluginHost, PluginMessage, PluginRegistrar, PluginRuntimeEvent,
        PluginSession, PluginSessionContext, PluginSpec, PluginSpecFactory, PromptHookContext,
        SessionPlugin, ToolCatalogContribution, ToolCatalogOverride, TurnHookContext,
        TurnResultHookContext,
    };
    pub use lash_plugin_tool_output_budget::{
        ToolOutputBudgetConfig, ToolOutputBudgetMode, ToolOutputBudgetPluginFactory,
        tool_output_budget_stack as runtime_plugin_stack,
    };
}

pub mod modes {
    pub use crate::mode::{RlmSessionBuilderExt, RlmTurnBuilderExt};
    pub use lash_protocol_rlm::{
        LashlangAbilities, LashlangHostCatalog, LashlangHostEnvironment, LashlangLanguageFeatures,
        NamedDataType, RlmProtocolPluginConfig, TypeExpr, TypeField, format_type_expr,
    };
    pub use lash_rlm_types::RlmFinalAnswerFormat;
}

pub mod messages {
    pub use lash_core::MessageRole;
}

pub mod remote {
    pub use lash_remote_protocol::{
        REMOTE_PROTOCOL_VERSION, RemoteAssistantOutput, RemoteAssistantOutputState,
        RemoteAttachmentRef, RemoteCausalRef, RemoteDiagnostic, RemoteExecutionSummary,
        RemoteGenerationOptions, RemoteInputItem, RemoteLashlangToolBinding, RemoteLiveReplayGap,
        RemoteLiveReplayGapReason, RemoteLlmAttachment, RemoteLlmContentBlock, RemoteLlmMessage,
        RemoteLlmOutputPart, RemoteLlmOutputSpec, RemoteLlmRequest, RemoteLlmRequestMetadata,
        RemoteLlmResponse, RemoteLlmRole, RemoteLlmTerminalReason, RemoteLlmToolChoice,
        RemoteLlmToolSpec, RemoteModelIntent, RemotePromptBuiltin, RemotePromptContribution,
        RemotePromptContributionGate, RemotePromptLayer, RemotePromptSlot, RemotePromptSlotLayer,
        RemotePromptTemplate, RemotePromptTemplateEntry, RemotePromptTemplateSection,
        RemoteProtocolError, RemoteProtocolTurnOptions, RemoteProviderMetadata,
        RemoteProviderReasoningReplay, RemoteProviderReplayMeta, RemoteResponseTextMeta,
        RemoteSchemaProjectionOverride, RemoteSessionCursor, RemoteSessionObservationEvent,
        RemoteSessionObservationEventPayload, RemoteSessionProcessEventKind,
        RemoteSessionQueueEventKind, RemoteTokenLedgerEntry, RemoteToolActivation,
        RemoteToolArgumentProjectionPolicy, RemoteToolAvailability, RemoteToolCallOutcome,
        RemoteToolCallRequest, RemoteToolCallResponse, RemoteToolCallSummary, RemoteToolGrant,
        RemoteToolOutputContract, RemoteToolRegistry, RemoteToolRetryPolicy, RemoteToolScheduling,
        RemoteTriggerEmitReport, RemoteTriggerOccurrenceRecord, RemoteTriggerOccurrenceRequest,
        RemoteTriggerRegistration, RemoteTriggerSubscriptionFilter, RemoteTriggerTargetSummary,
        RemoteTurnActivity, RemoteTurnEvent, RemoteTurnFinish, RemoteTurnInput, RemoteTurnIssue,
        RemoteTurnOutcome, RemoteTurnRequest, RemoteTurnResult, RemoteTurnStatus, RemoteTurnStop,
        RemoteTurnUsageSummary, RemoteUsage, assert_remote_tool_registry_reopenable,
        ensure_protocol_version,
    };
}

pub mod process {
    pub use crate::admin::{Processes, SessionProcessAdmin};
    pub use lash_core::{
        ObservedProcess, ObservedProcessEvent, ObservedWorkItem, ProcessAwaitOutput,
        ProcessCancelAbility, ProcessCancelAllRequest, ProcessCancelRequest, ProcessCancelSource,
        ProcessCancelSummary, ProcessDefinitionSummary, ProcessEvent, ProcessEventAppendRequest,
        ProcessEventAppendResult, ProcessEventType, ProcessExecutionContext, ProcessExternalRef,
        ProcessHandleDescriptor, ProcessHandleGrant, ProcessHandleSummary, ProcessInput,
        ProcessLease, ProcessLeaseCompletion, ProcessLifecycleStatus, ProcessListFilter,
        ProcessListMode, ProcessOpScope, ProcessRecord, ProcessRegistration, ProcessRegistry,
        ProcessRunHandle, ProcessRuntimeHost, ProcessService, ProcessSessionDeleteReport,
        ProcessStartOptions, ProcessStartRequest, ProcessStatus, ProcessStatusFilter,
        ProcessTerminalState, ProcessWake, ProcessWakeDedupeKey, ProcessWakeDelivery,
        ProcessWakeSpec, ProcessWorkDriver, ProcessWorkObserver, ProcessWorkPoke,
        ProcessWorkRunner, ProcessWorkSnapshot, SessionScope, SessionScopeId,
        lashlang_process_event_types,
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
        RuntimeTurnPhaseProbe, ScopedEffectController, TurnContext,
    };
    pub use lash_core::{
        PersistentRuntimeServices, PluginMessage, ProtocolSessionExtensionHandle,
        ProtocolTurnOptions, SessionHandle, SessionPolicy, SessionSnapshot, TurnCause, TurnFinish,
        TurnOutcome, TurnStop, render_turn_causes_prompt,
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
        ProviderRateLimitPolicy, ProviderReliability, ProviderRetryPolicy,
    };
    pub use lash_core::{
        LlmTimeouts, Provider, ProviderComponents, ProviderFactory, ProviderHandle,
        ProviderModelPolicy, ProviderOptions, ProviderSpec, RequestTimeout, StaticModelPolicy,
    };
}
