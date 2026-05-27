use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::CheckpointKind;
use crate::llm::types::{
    LlmAttachment, LlmEventSender, LlmMessage, LlmOutputSpec, LlmProviderTraceSender,
    LlmToolChoice, LlmToolSpec,
};
use crate::plugin::PluginMessage;
use crate::sansio::{CompletedToolCall, ExecutionSurfaceSync, LlmCallError};
use crate::{
    AttachmentCreateMeta, AttachmentRef, AttachmentStore, DirectMessage, DirectOutputSpec,
    DirectRequest, ExecResponse, LlmRequest as CoreLlmRequest, LlmResponse, MediaType,
    ProcessAwaitOutput, ProcessExecutionContext, ProcessHandleGrantEntry, ProcessRecord,
    ProcessRegistration, ProcessScope, ProcessStartGrant,
};

use super::executor::RuntimeEffectControllerError;

/// Where a runtime effect originated.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EffectOrigin {
    Turn,
    DirectCompletion { usage_source: String },
    DirectLlmCompletion { usage_source: String },
}

/// Durable category for a runtime effect.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeEffectKind {
    LlmCall,
    DirectCompletion,
    DirectLlmCompletion,
    ToolCall,
    Process,
    ExecCode,
    Checkpoint,
    SyncExecutionSurface,
    Sleep,
}

impl RuntimeEffectKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::LlmCall => "llm_call",
            Self::DirectCompletion => "direct_completion",
            Self::DirectLlmCompletion => "direct_llm_completion",
            Self::ToolCall => "tool_call",
            Self::Process => "process",
            Self::ExecCode => "exec_code",
            Self::Checkpoint => "checkpoint",
            Self::SyncExecutionSurface => "sync_execution_surface",
            Self::Sleep => "sleep",
        }
    }
}

/// Serializable metadata attached to every controller-run runtime effect.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EffectInvocationMetadata {
    pub session_id: String,
    pub origin: EffectOrigin,
    pub turn_id: Option<String>,
    pub turn_index: Option<usize>,
    pub protocol_iteration: Option<usize>,
    pub effect_id: String,
    pub effect_kind: RuntimeEffectKind,
    pub idempotency_key: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn_checkpoint_hash: Option<String>,
}

/// Fully serializable envelope emitted at Lash's nondeterministic boundary.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RuntimeEffectEnvelope {
    pub metadata: EffectInvocationMetadata,
    pub command: RuntimeEffectCommand,
}

impl RuntimeEffectEnvelope {
    pub fn new(metadata: EffectInvocationMetadata, command: RuntimeEffectCommand) -> Self {
        Self { metadata, command }
    }

    pub fn stable_hash(&self) -> Result<String, RuntimeEffectControllerError> {
        crate::stable_hash::stable_json_sha256_hex(self).map_err(|err| {
            RuntimeEffectControllerError::new(
                "runtime_effect_envelope_hash",
                format!("failed to serialize runtime effect envelope: {err}"),
            )
        })
    }
}

/// Serializable command emitted at Lash's nondeterministic runtime boundary.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RuntimeEffectCommand {
    LlmCall {
        request: Box<LlmRequestSpec>,
    },
    DirectCompletion {
        request: Box<DirectRequestSpec>,
        normalized_request: Box<LlmRequestSpec>,
        model: String,
        usage_source: String,
    },
    DirectLlmCompletion {
        request: Box<LlmRequestSpec>,
        usage_source: String,
    },
    ToolCall {
        call: crate::PreparedToolCall,
    },
    Process {
        command: ProcessCommand,
    },
    ExecCode {
        code: String,
    },
    Checkpoint {
        checkpoint: CheckpointKind,
    },
    SyncExecutionSurface {
        update_machine_config: bool,
    },
    Sleep {
        duration_ms: u64,
    },
}

impl RuntimeEffectCommand {
    pub fn kind(&self) -> RuntimeEffectKind {
        match self {
            Self::LlmCall { .. } => RuntimeEffectKind::LlmCall,
            Self::DirectCompletion { .. } => RuntimeEffectKind::DirectCompletion,
            Self::DirectLlmCompletion { .. } => RuntimeEffectKind::DirectLlmCompletion,
            Self::ToolCall { .. } => RuntimeEffectKind::ToolCall,
            Self::Process { .. } => RuntimeEffectKind::Process,
            Self::ExecCode { .. } => RuntimeEffectKind::ExecCode,
            Self::Checkpoint { .. } => RuntimeEffectKind::Checkpoint,
            Self::SyncExecutionSurface { .. } => RuntimeEffectKind::SyncExecutionSurface,
            Self::Sleep { .. } => RuntimeEffectKind::Sleep,
        }
    }
}

/// Serializable operation against the process control plane.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum ProcessCommand {
    Start {
        registration: ProcessRegistration,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        grant: Option<ProcessStartGrant>,
        #[serde(
            default,
            skip_serializing_if = "boxed_process_execution_context_is_empty"
        )]
        execution_context: Box<ProcessExecutionContext>,
    },
    List {
        owner_scope: ProcessScope,
    },
    Transfer {
        from_scope: ProcessScope,
        to_scope: ProcessScope,
        process_ids: Vec<String>,
    },
    DeleteSession {
        session_id: String,
    },
    Await {
        process_id: String,
    },
    Cancel {
        process_id: String,
        reason: Option<String>,
    },
}

fn boxed_process_execution_context_is_empty(context: &ProcessExecutionContext) -> bool {
    context.is_empty()
}

type CheckpointMessageDeltas = (Vec<PluginMessage>, Vec<PluginMessage>);
type CheckpointOutcome = Result<CheckpointMessageDeltas, RuntimeEffectControllerError>;

impl ProcessCommand {
    pub fn effect_id(&self) -> String {
        match self {
            Self::Start { registration, .. } => format!("process:start:{}", registration.id),
            Self::List { owner_scope } => format!("process:list:{}", owner_scope.id()),
            Self::Transfer {
                from_scope,
                to_scope,
                process_ids,
            } => {
                let digest = crate::stable_hash::stable_json_sha256_hex(process_ids)
                    .unwrap_or_else(|_| "unhashable".to_string());
                format!(
                    "process:transfer:{}:{}:{digest}",
                    from_scope.id(),
                    to_scope.id()
                )
            }
            Self::DeleteSession { session_id } => format!("process:delete-session:{session_id}"),
            Self::Await { process_id } => format!("process:await:{process_id}"),
            Self::Cancel { process_id, .. } => format!("process:cancel:{process_id}"),
        }
    }
}

/// Serializable result of a process operation.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum ProcessEffectOutcome {
    Start {
        record: ProcessRecord,
    },
    List {
        entries: Vec<ProcessHandleGrantEntry>,
    },
    Transfer,
    DeleteSession {
        report: crate::ProcessSessionDeleteReport,
    },
    Await {
        output: ProcessAwaitOutput,
    },
    Cancel {
        record: ProcessRecord,
    },
}

/// Serializable result of a runtime effect command.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RuntimeEffectOutcome {
    LlmCall {
        result: Result<LlmResponse, LlmCallError>,
        text_streamed: bool,
    },
    DirectCompletion {
        result: Result<LlmResponse, LlmCallError>,
    },
    DirectLlmCompletion {
        result: Result<LlmResponse, LlmCallError>,
    },
    ToolCall {
        result: CompletedToolCall,
    },
    Process {
        result: ProcessEffectOutcome,
    },
    ExecCode {
        result: Result<ExecResponse, String>,
    },
    Checkpoint {
        result: Result<(Vec<PluginMessage>, Vec<PluginMessage>), RuntimeEffectControllerError>,
    },
    SyncExecutionSurface {
        result: Result<Option<ExecutionSurfaceSync>, String>,
    },
    Sleep,
}

// =============================================================================
// Request specs (serializable forms of LLM/Direct requests)
// =============================================================================

/// Serializable attachment data for runtime effect envelopes.
///
/// Effect envelopes carry attachment references only. Local executors resolve
/// bytes from the configured attachment store when a provider request is
/// actually executed.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LlmAttachmentSpec {
    pub reference: AttachmentRef,
}

impl LlmAttachmentSpec {
    fn into_attachment(self) -> LlmAttachment {
        LlmAttachment::reference(self.reference)
    }
}

/// Serializable LLM request data. Live stream and provider-trace callbacks are
/// attached by the local executor, and attachment bytes are resolved locally
/// from refs rather than persisted in the effect envelope.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LlmRequestSpec {
    pub model: String,
    pub messages: Vec<LlmMessage>,
    pub attachments: Vec<LlmAttachmentSpec>,
    pub tools: Vec<LlmToolSpec>,
    pub tool_choice: LlmToolChoice,
    pub model_variant: Option<String>,
    #[serde(default)]
    pub generation: crate::GenerationOptions,
    pub session_id: Option<String>,
    pub output_spec: Option<LlmOutputSpec>,
}

impl LlmRequestSpec {
    pub fn from_request(
        request: &CoreLlmRequest,
        attachment_store: &dyn AttachmentStore,
    ) -> Result<Self, RuntimeEffectControllerError> {
        Ok(Self {
            model: request.model.clone(),
            messages: request.messages.clone(),
            attachments: attachment_specs_from_attachments(&request.attachments, attachment_store)?,
            tools: request.tools.iter().cloned().collect(),
            tool_choice: request.tool_choice.clone(),
            model_variant: request.model_variant.clone(),
            generation: request.generation.clone(),
            session_id: request.session_id.clone(),
            output_spec: request.output_spec.clone(),
        })
    }

    pub fn into_request(
        self,
        stream_events: Option<LlmEventSender>,
        provider_trace: Option<LlmProviderTraceSender>,
    ) -> CoreLlmRequest {
        CoreLlmRequest {
            model: self.model,
            messages: self.messages,
            attachments: self
                .attachments
                .into_iter()
                .map(LlmAttachmentSpec::into_attachment)
                .collect(),
            tools: Arc::new(self.tools),
            tool_choice: self.tool_choice,
            model_variant: self.model_variant,
            generation: self.generation,
            session_id: self.session_id,
            output_spec: self.output_spec,
            stream_events,
            provider_trace,
        }
    }
}

/// Serializable direct request data. Caller-provided stream callbacks remain
/// local process state and are reattached by local direct executors.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DirectRequestSpec {
    pub model: String,
    pub model_variant: Option<String>,
    #[serde(default)]
    pub generation: crate::GenerationOptions,
    pub messages: Vec<DirectMessage>,
    pub attachments: Vec<LlmAttachmentSpec>,
    pub output: DirectOutputSpec,
    pub session_id: Option<String>,
    pub originating_tool_call_id: Option<String>,
    pub idempotency_key: Option<String>,
}

impl DirectRequestSpec {
    pub fn from_request(
        request: &DirectRequest,
        attachment_store: &dyn AttachmentStore,
    ) -> Result<Self, RuntimeEffectControllerError> {
        Ok(Self {
            model: request.model.clone(),
            model_variant: request.model_variant.clone(),
            generation: request.generation.clone(),
            messages: request.messages.clone(),
            attachments: attachment_specs_from_attachments(&request.attachments, attachment_store)?,
            output: request.output.clone(),
            session_id: request.session_id.clone(),
            originating_tool_call_id: request.originating_tool_call_id.clone(),
            idempotency_key: request.idempotency_key.clone(),
        })
    }

    pub fn into_request(self, stream_events: Option<LlmEventSender>) -> DirectRequest {
        DirectRequest {
            model: self.model,
            model_variant: self.model_variant,
            generation: self.generation,
            messages: self.messages,
            attachments: self
                .attachments
                .into_iter()
                .map(LlmAttachmentSpec::into_attachment)
                .collect(),
            output: self.output,
            stream_events,
            session_id: self.session_id,
            originating_tool_call_id: self.originating_tool_call_id,
            idempotency_key: self.idempotency_key,
        }
    }
}

fn attachment_specs_from_attachments(
    attachments: &[LlmAttachment],
    attachment_store: &dyn AttachmentStore,
) -> Result<Vec<LlmAttachmentSpec>, RuntimeEffectControllerError> {
    attachments
        .iter()
        .map(|attachment| attachment_spec_from_attachment(attachment, attachment_store))
        .collect()
}

fn attachment_spec_from_attachment(
    attachment: &LlmAttachment,
    attachment_store: &dyn AttachmentStore,
) -> Result<LlmAttachmentSpec, RuntimeEffectControllerError> {
    if let Some(reference) = attachment.reference.as_ref() {
        return Ok(LlmAttachmentSpec {
            reference: reference.clone(),
        });
    }
    if attachment.data.is_empty() {
        return Err(RuntimeEffectControllerError::new(
            "runtime_effect_attachment_missing_reference",
            "runtime effect attachment has neither a durable reference nor inline bytes",
        ));
    }
    let media_type = MediaType::from_mime(&attachment.mime).ok_or_else(|| {
        RuntimeEffectControllerError::new(
            "runtime_effect_attachment_media_type",
            format!(
                "attachment media type `{}` cannot be represented durably",
                attachment.mime
            ),
        )
    })?;
    let reference = attachment_store
        .put(
            attachment.data.clone(),
            AttachmentCreateMeta::new(media_type, None, None, None),
        )
        .map_err(|err| {
            RuntimeEffectControllerError::new(
                "runtime_effect_attachment_store",
                format!("failed to store attachment before runtime effect invocation: {err}"),
            )
        })?;
    Ok(LlmAttachmentSpec { reference })
}

impl RuntimeEffectOutcome {
    pub fn into_llm_call(
        self,
    ) -> Result<(Result<LlmResponse, LlmCallError>, bool), RuntimeEffectControllerError> {
        match self {
            Self::LlmCall {
                result,
                text_streamed,
            } => Ok((result, text_streamed)),
            other => Err(RuntimeEffectControllerError::wrong_outcome(
                RuntimeEffectKind::LlmCall,
                other.kind(),
            )),
        }
    }

    pub fn into_direct_completion_response(
        self,
    ) -> Result<Result<LlmResponse, LlmCallError>, RuntimeEffectControllerError> {
        match self {
            Self::DirectCompletion { result } => Ok(result),
            other => Err(RuntimeEffectControllerError::wrong_outcome(
                RuntimeEffectKind::DirectCompletion,
                other.kind(),
            )),
        }
    }

    pub fn into_direct_llm_completion_response(
        self,
    ) -> Result<Result<LlmResponse, LlmCallError>, RuntimeEffectControllerError> {
        match self {
            Self::DirectLlmCompletion { result } => Ok(result),
            other => Err(RuntimeEffectControllerError::wrong_outcome(
                RuntimeEffectKind::DirectLlmCompletion,
                other.kind(),
            )),
        }
    }

    pub fn into_tool_call(self) -> Result<CompletedToolCall, RuntimeEffectControllerError> {
        match self {
            Self::ToolCall { result } => Ok(result),
            other => Err(RuntimeEffectControllerError::wrong_outcome(
                RuntimeEffectKind::ToolCall,
                other.kind(),
            )),
        }
    }

    pub fn into_process(self) -> Result<ProcessEffectOutcome, RuntimeEffectControllerError> {
        match self {
            Self::Process { result } => Ok(result),
            other => Err(RuntimeEffectControllerError::wrong_outcome(
                RuntimeEffectKind::Process,
                other.kind(),
            )),
        }
    }

    pub fn into_exec_code(
        self,
    ) -> Result<Result<ExecResponse, String>, RuntimeEffectControllerError> {
        match self {
            Self::ExecCode { result } => Ok(result),
            other => Err(RuntimeEffectControllerError::wrong_outcome(
                RuntimeEffectKind::ExecCode,
                other.kind(),
            )),
        }
    }

    pub(crate) fn into_checkpoint(self) -> Result<CheckpointOutcome, RuntimeEffectControllerError> {
        match self {
            Self::Checkpoint { result } => Ok(result),
            other => Err(RuntimeEffectControllerError::wrong_outcome(
                RuntimeEffectKind::Checkpoint,
                other.kind(),
            )),
        }
    }

    pub fn into_sync_execution_surface(
        self,
    ) -> Result<Result<Option<ExecutionSurfaceSync>, String>, RuntimeEffectControllerError> {
        match self {
            Self::SyncExecutionSurface { result } => Ok(result),
            other => Err(RuntimeEffectControllerError::wrong_outcome(
                RuntimeEffectKind::SyncExecutionSurface,
                other.kind(),
            )),
        }
    }

    pub fn kind(&self) -> RuntimeEffectKind {
        match self {
            Self::LlmCall { .. } => RuntimeEffectKind::LlmCall,
            Self::DirectCompletion { .. } => RuntimeEffectKind::DirectCompletion,
            Self::DirectLlmCompletion { .. } => RuntimeEffectKind::DirectLlmCompletion,
            Self::ToolCall { .. } => RuntimeEffectKind::ToolCall,
            Self::Process { .. } => RuntimeEffectKind::Process,
            Self::ExecCode { .. } => RuntimeEffectKind::ExecCode,
            Self::Checkpoint { .. } => RuntimeEffectKind::Checkpoint,
            Self::SyncExecutionSurface { .. } => RuntimeEffectKind::SyncExecutionSurface,
            Self::Sleep => RuntimeEffectKind::Sleep,
        }
    }
}
