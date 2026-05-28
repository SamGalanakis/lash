pub mod attachment;
pub mod llm;
pub mod plugin;
pub mod prompt;
pub mod sansio;
pub mod session;
pub mod session_model;
pub mod tool_contract;
pub mod tool_output;
pub mod tool_surface;
pub mod turn;
pub mod turn_driver;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

pub use attachment::{
    AttachmentCreateMeta, AttachmentId, AttachmentMeta, AttachmentRef, ImageMediaType, MediaType,
};
pub use llm::types::LlmTerminalReason;
pub use plugin::{
    CheckpointKind, PluginMessage, PluginRuntimeEvent, PromptContribution, PromptContributionGate,
};
pub use prompt::{
    PreparedPrompt, PromptBuildInput, PromptCache, PromptContext, PromptContributionSet,
    PromptFingerprint, build_prompt, build_prompt_cached, prompt_template_fingerprint,
    prompt_text_fingerprint, prompt_tool_names_fingerprint,
};
pub use sansio::{
    ChatContextProjector, CheckpointDelivery, CheckpointResumeAction, CompletedToolCall,
    ContextProjector, DriverAction, DriverContextView, Effect, EffectId, LlmCallError,
    PendingToolCall, ProjectorContext, ProtocolDriverHandle, Response, TurnCause, TurnCheckpoint,
    TurnMachine, TurnMachineConfig, TurnProtocol, UnitTurnProtocol, WaitingExecState,
    WaitingLlmState, render_turn_causes_prompt,
};
pub use session::{
    CompletedTurn, ExecImage, ExecResponse, PromptUsage, SansIoSessionState,
    TextProjectionMetadata, apply_completed_turn,
};
pub use session_model::message::MessageOrigin;
pub use session_model::{
    AcceptedInjectedTurnInput, BaseRenderCache, ConversationRecord, ErrorEnvelope,
    MAIN_AGENT_INTRO, Message, MessageRole, MessageSequence, Part, PartAttachment, PartKind,
    PromptBuiltin, PromptLayer, PromptSlot, PromptSlotLayer, PromptTemplate, PromptTemplateEntry,
    PromptTemplateSection, PruneState, RenderedPrompt, ResolvedPromptLayer, SessionEvent,
    SessionEventRecord, TokenUsage, ToolEvent, TurnFinish, TurnOutcome, TurnStop,
    default_prompt_template, messages_are_prompt_resume_safe, resolve_prompt_layers, shared_parts,
};
pub use tool_contract::{
    CompactToolContract, ModelTool, SchemaProjectionOverride, ToolActivation,
    ToolArgumentProjectionPolicy, ToolAvailability, ToolAvailabilityConfig, ToolContract,
    ToolDefinition, ToolDiscoveryMetadata, ToolId, ToolManifest, ToolOutputContract,
    ToolRetryPolicy, ToolScheduling, schema_for,
};
pub use tool_output::{
    ModelToolReturn, ModelToolReturnPart, ToolCallOutcome, ToolCallOutput, ToolCallRecord,
    ToolCallStatus, ToolCancellation, ToolControl, ToolFailure, ToolFailureClass,
    ToolFailureSource, ToolRetryDisposition, ToolValue, format_tool_output_content,
    model_parts_from_tool_output,
};
pub use tool_surface::{
    ToolContractResolver, ToolSurface, ToolSurfaceBuildInput, ToolSurfaceContribution,
    ToolSurfaceEntry, ToolSurfaceOverride, build_tool_surface,
};
pub use turn::{PreparedTurnMachine, SansIoTurnInput, build_turn};
pub use turn_driver::{
    TurnDriverConfig, TurnDriverPreamble, TurnLimitFinalMessage, append_assistant_text_part,
    normalized_response_parts, reasoning_part,
};

pub fn head_tail_truncate(value: &str, max_chars: usize) -> (String, usize) {
    let raw_len = value.chars().count();
    if max_chars == 0 || raw_len <= max_chars {
        return (value.to_string(), raw_len);
    }
    let head_len = max_chars / 2;
    let tail_len = max_chars.saturating_sub(head_len);
    let head = value.chars().take(head_len).collect::<String>();
    let tail = value
        .chars()
        .rev()
        .take(tail_len)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<String>();
    let omitted = raw_len.saturating_sub(head_len + tail_len);
    (
        format!("{head}\n\n... ({omitted} characters omitted) ...\n\n{tail}"),
        raw_len,
    )
}
