use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::CheckpointKind;
use crate::llm::types::{
    LlmAttachment, LlmEventSender, LlmMessage, LlmOutputSpec, LlmProviderTraceSender,
    LlmToolChoice, LlmToolSpec,
};
use crate::runtime::ProcessHandleGrantEntry;
use crate::sansio::{CompletedToolCall, ExecutionEnvironmentSync, LlmCallError};
use crate::tool_dispatch::ToolTriggerEffectOutcome;
use crate::{
    AttachmentCreateMeta, AttachmentRef, AttachmentStore, CausalRef, CheckpointDelivery,
    ExecResponse, LlmRequest as CoreLlmRequest, LlmResponse, MediaType, ProcessAwaitOutput,
    ProcessExecutionContext, ProcessListMode, ProcessRecord, ProcessRegistration,
    ProcessStartGrant, SessionScope,
};

use super::executor::RuntimeEffectControllerError;

/// Durable category for a runtime effect.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeEffectKind {
    LlmCall,
    Direct,
    ToolAttempt,
    ToolBatch,
    Process,
    ExecCode,
    Checkpoint,
    SyncExecutionEnvironment,
    Sleep,
    AwaitEvent,
    DurableStep,
}

impl RuntimeEffectKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::LlmCall => "llm_call",
            Self::Direct => "direct",
            Self::ToolAttempt => "tool_attempt",
            Self::ToolBatch => "tool_batch",
            Self::Process => "process",
            Self::ExecCode => "exec_code",
            Self::Checkpoint => "checkpoint",
            Self::SyncExecutionEnvironment => "sync_execution_environment",
            Self::Sleep => "sleep",
            Self::AwaitEvent => "await_event",
            Self::DurableStep => "durable_step",
        }
    }
}

/// Canonical lineage for a runtime-side invocation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeInvocation {
    pub scope: RuntimeScope,
    pub subject: RuntimeSubject,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub caused_by: Option<CausalRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub replay: Option<RuntimeReplay>,
}

impl RuntimeInvocation {
    pub fn effect(
        scope: RuntimeScope,
        effect_id: impl Into<String>,
        kind: RuntimeEffectKind,
        replay_key: impl Into<String>,
    ) -> Self {
        Self {
            scope,
            subject: RuntimeSubject::Effect {
                effect_id: effect_id.into(),
                kind,
            },
            caused_by: None,
            replay: Some(RuntimeReplay {
                key: replay_key.into(),
            }),
        }
    }

    pub fn with_caused_by(mut self, caused_by: Option<CausalRef>) -> Self {
        self.caused_by = caused_by;
        self
    }

    pub fn effect_id(&self) -> Option<&str> {
        match &self.subject {
            RuntimeSubject::Effect { effect_id, .. } => Some(effect_id),
            _ => None,
        }
    }

    pub fn effect_kind(&self) -> Option<RuntimeEffectKind> {
        match &self.subject {
            RuntimeSubject::Effect { kind, .. } => Some(*kind),
            _ => None,
        }
    }

    pub fn replay_key(&self) -> Option<&str> {
        self.replay.as_ref().map(|replay| replay.key.as_str())
    }

    pub fn causal_ref(&self) -> Option<CausalRef> {
        match &self.subject {
            RuntimeSubject::Effect { effect_id, .. } => Some(CausalRef::Effect {
                session_id: self.scope.session_id.clone(),
                turn_id: self.scope.turn_id.clone(),
                effect_id: effect_id.clone(),
            }),
            RuntimeSubject::Process { process_id } => Some(CausalRef::Process {
                process_id: process_id.clone(),
            }),
            RuntimeSubject::ProcessEvent {
                process_id,
                sequence,
                ..
            } => Some(CausalRef::ProcessEvent {
                process_id: process_id.clone(),
                sequence: *sequence,
            }),
            RuntimeSubject::TriggerOccurrence { occurrence_id } => {
                Some(CausalRef::TriggerOccurrence {
                    occurrence_id: occurrence_id.clone(),
                })
            }
            RuntimeSubject::SessionNode { node_id } => Some(CausalRef::SessionNode {
                session_id: self.scope.session_id.clone(),
                node_id: node_id.clone(),
            }),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeScope {
    pub session_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn_index: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub protocol_iteration: Option<usize>,
}

impl RuntimeScope {
    pub fn new(session_id: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
            turn_id: None,
            turn_index: None,
            protocol_iteration: None,
        }
    }

    pub fn for_turn(
        session_id: impl Into<String>,
        turn_id: impl Into<String>,
        turn_index: usize,
        protocol_iteration: usize,
    ) -> Self {
        Self {
            session_id: session_id.into(),
            turn_id: Some(turn_id.into()),
            turn_index: Some(turn_index),
            protocol_iteration: Some(protocol_iteration),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeReplay {
    pub key: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RuntimeSubject {
    Effect {
        effect_id: String,
        kind: RuntimeEffectKind,
    },
    Process {
        process_id: String,
    },
    ProcessEvent {
        process_id: String,
        sequence: u64,
        event_type: String,
    },
    TriggerOccurrence {
        occurrence_id: String,
    },
    SessionNode {
        node_id: String,
    },
}

/// Fully serializable envelope emitted at Lash's nondeterministic boundary.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RuntimeEffectEnvelope {
    pub invocation: RuntimeInvocation,
    pub command: RuntimeEffectCommand,
}

impl RuntimeEffectEnvelope {
    pub fn new(invocation: RuntimeInvocation, command: RuntimeEffectCommand) -> Self {
        Self::try_new(invocation, command).expect("valid runtime effect invocation")
    }

    pub fn try_new(
        invocation: RuntimeInvocation,
        command: RuntimeEffectCommand,
    ) -> Result<Self, RuntimeEffectControllerError> {
        validate_effect_invocation(&invocation, command.kind())?;
        validate_effect_command(&command)?;
        Ok(Self {
            invocation,
            command,
        })
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

fn validate_effect_invocation(
    invocation: &RuntimeInvocation,
    command_kind: RuntimeEffectKind,
) -> Result<(), RuntimeEffectControllerError> {
    let RuntimeSubject::Effect { effect_id, kind } = &invocation.subject else {
        return Err(RuntimeEffectControllerError::new(
            "runtime_effect_invocation_subject",
            "runtime effect envelope subject must be an effect",
        ));
    };
    if effect_id.trim().is_empty() {
        return Err(RuntimeEffectControllerError::new(
            "runtime_effect_invocation_subject",
            "runtime effect envelope effect id must be non-empty",
        ));
    }
    if *kind != command_kind {
        return Err(RuntimeEffectControllerError::new(
            "runtime_effect_invocation_kind",
            format!(
                "runtime effect invocation kind {} does not match command kind {}",
                kind.as_str(),
                command_kind.as_str()
            ),
        ));
    }
    if invocation
        .replay
        .as_ref()
        .is_none_or(|replay| replay.key.is_empty())
    {
        return Err(RuntimeEffectControllerError::new(
            "runtime_effect_replay_required",
            "runtime effect envelope requires replay.key",
        ));
    }
    Ok(())
}

fn validate_effect_command(
    command: &RuntimeEffectCommand,
) -> Result<(), RuntimeEffectControllerError> {
    if let RuntimeEffectCommand::DurableStep { step_id, .. } = command
        && step_id.trim().is_empty()
    {
        return Err(RuntimeEffectControllerError::new(
            "runtime_effect_durable_step_id",
            "runtime effect durable step id must be non-empty",
        ));
    }
    if let RuntimeEffectCommand::ToolAttempt {
        call,
        execution_grant: _,
        attempt,
        max_attempts,
    } = command
    {
        if call.call_id.trim().is_empty() {
            return Err(RuntimeEffectControllerError::new(
                "runtime_effect_tool_attempt_call_id",
                "runtime effect tool attempt requires a non-empty call id",
            ));
        }
        if *attempt == 0 || *max_attempts == 0 || *attempt > *max_attempts {
            return Err(RuntimeEffectControllerError::new(
                "runtime_effect_tool_attempt_index",
                format!(
                    "runtime effect tool attempt must satisfy 1 <= attempt <= max_attempts, got {attempt}/{max_attempts}"
                ),
            ));
        }
    }
    if let RuntimeEffectCommand::ToolBatch { batch } = command {
        if batch.batch_id.trim().is_empty() {
            return Err(RuntimeEffectControllerError::new(
                "runtime_effect_tool_batch_id",
                "runtime effect tool batch id must be non-empty",
            ));
        }
        if batch.calls.is_empty() {
            return Err(RuntimeEffectControllerError::new(
                "runtime_effect_tool_batch_empty",
                "runtime effect tool batch must contain at least one prepared call",
            ));
        }
        for (index, call) in batch.calls.iter().enumerate() {
            if call.call.call_id.trim().is_empty() {
                return Err(RuntimeEffectControllerError::new(
                    "runtime_effect_tool_batch_call_id",
                    format!("runtime effect tool batch call {index} has an empty call id"),
                ));
            }
            if call.replay_suffix.trim().is_empty() {
                return Err(RuntimeEffectControllerError::new(
                    "runtime_effect_tool_batch_call_replay",
                    format!("runtime effect tool batch call {index} has an empty replay suffix"),
                ));
            }
        }
    }
    Ok(())
}

/// Serializable command emitted at Lash's nondeterministic runtime boundary.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RuntimeEffectCommand {
    LlmCall {
        request: Box<LlmRequestSpec>,
    },
    Direct {
        request: Box<LlmRequestSpec>,
        usage_source: String,
    },
    ToolAttempt {
        call: crate::PreparedToolCall,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        execution_grant: Option<Box<crate::ToolExecutionGrant>>,
        attempt: u32,
        max_attempts: u32,
    },
    ToolBatch {
        batch: crate::PreparedToolBatch,
    },
    Process {
        command: Box<ProcessCommand>,
    },
    ExecCode {
        language: String,
        code: String,
    },
    Checkpoint {
        checkpoint: CheckpointKind,
    },
    SyncExecutionEnvironment {
        update_machine_config: bool,
    },
    Sleep {
        duration_ms: u64,
    },
    AwaitEvent {
        key: crate::AwaitEventKey,
    },
    DurableStep {
        step_id: String,
        input: serde_json::Value,
    },
}

impl RuntimeEffectCommand {
    pub fn process(command: ProcessCommand) -> Self {
        Self::Process {
            command: Box::new(command),
        }
    }

    pub fn kind(&self) -> RuntimeEffectKind {
        match self {
            Self::LlmCall { .. } => RuntimeEffectKind::LlmCall,
            Self::Direct { .. } => RuntimeEffectKind::Direct,
            Self::ToolAttempt { .. } => RuntimeEffectKind::ToolAttempt,
            Self::ToolBatch { .. } => RuntimeEffectKind::ToolBatch,
            Self::Process { .. } => RuntimeEffectKind::Process,
            Self::ExecCode { .. } => RuntimeEffectKind::ExecCode,
            Self::Checkpoint { .. } => RuntimeEffectKind::Checkpoint,
            Self::SyncExecutionEnvironment { .. } => RuntimeEffectKind::SyncExecutionEnvironment,
            Self::Sleep { .. } => RuntimeEffectKind::Sleep,
            Self::AwaitEvent { .. } => RuntimeEffectKind::AwaitEvent,
            Self::DurableStep { .. } => RuntimeEffectKind::DurableStep,
        }
    }
}

/// Serializable operation against the process admin plane.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
#[allow(clippy::large_enum_variant)]
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
        session_scope: SessionScope,
        #[serde(default)]
        mode: ProcessListMode,
    },
    Transfer {
        from_scope: SessionScope,
        to_scope: SessionScope,
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
    Signal {
        process_id: String,
        signal_name: String,
        signal_id: String,
        request: crate::ProcessEventAppendRequest,
    },
}

fn boxed_process_execution_context_is_empty(context: &ProcessExecutionContext) -> bool {
    context.is_empty()
}

type CheckpointOutcome = Result<CheckpointDelivery, RuntimeEffectControllerError>;

impl ProcessCommand {
    pub fn effect_id(&self) -> String {
        match self {
            Self::Start { registration, .. } => format!("process:start:{}", registration.id),
            Self::List {
                session_scope,
                mode,
            } => {
                format!("process:list:{}:{}", session_scope.id(), mode.as_str())
            }
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
            Self::Signal {
                process_id,
                signal_name,
                signal_id,
                ..
            } => {
                format!("process:signal:{process_id}:signal.{signal_name}:{signal_id}")
            }
        }
    }
}

/// Serializable result of a process operation.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum ProcessEffectOutcome {
    Start {
        // Boxed so the fat durable record does not size the whole outcome enum
        // (and the runtime effect enum wrapping it) inline through the recursive
        // effect executor.
        record: Box<ProcessRecord>,
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
        record: Box<ProcessRecord>,
    },
    Signal {
        // Boxed for the same reason as the record variants: a fat event should
        // not size the outcome enum inline through the recursive executor.
        event: Box<crate::ProcessEvent>,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolAttemptEffectOutcome {
    pub launch: ToolAttemptLaunch,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub triggers: Vec<ToolTriggerEffectOutcome>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolBatchEffectOutcome {
    pub launches: Vec<ToolCallLaunch>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub triggers: Vec<ToolTriggerEffectOutcome>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
#[allow(clippy::large_enum_variant)]
pub enum ToolCallLaunch {
    Done {
        result: CompletedToolCall,
    },
    Pending {
        key: crate::AwaitEventKey,
        pending: crate::PendingCompletion,
        duration_ms: u64,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum ToolAttemptLaunch {
    Done {
        record: crate::ToolCallRecord,
    },
    Pending {
        key: crate::AwaitEventKey,
        pending: crate::PendingCompletion,
        duration_ms: u64,
    },
}

/// Serializable result of a runtime effect command.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[allow(clippy::large_enum_variant)]
pub enum RuntimeEffectOutcome {
    LlmCall {
        result: Result<LlmResponse, LlmCallError>,
        text_streamed: bool,
    },
    Direct {
        result: Result<LlmResponse, LlmCallError>,
    },
    ToolAttempt {
        launch: ToolAttemptLaunch,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        triggers: Vec<ToolTriggerEffectOutcome>,
    },
    ToolBatch {
        launches: Vec<ToolCallLaunch>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        triggers: Vec<ToolTriggerEffectOutcome>,
    },
    Process {
        result: ProcessEffectOutcome,
    },
    ExecCode {
        result: Result<ExecResponse, String>,
    },
    Checkpoint {
        result: CheckpointOutcome,
    },
    SyncExecutionEnvironment {
        result: Result<Option<ExecutionEnvironmentSync>, String>,
    },
    Sleep,
    AwaitEvent {
        resolution: crate::Resolution,
    },
    DurableStep {
        value: serde_json::Value,
    },
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
    pub tools: Arc<Vec<LlmToolSpec>>,
    pub tool_choice: LlmToolChoice,
    pub model_variant: Option<String>,
    #[serde(default)]
    pub generation: crate::GenerationOptions,
    pub scope: crate::LlmRequestScope,
    pub output_spec: Option<LlmOutputSpec>,
}

impl LlmRequestSpec {
    pub async fn from_request(
        request: &CoreLlmRequest,
        attachment_store: &dyn AttachmentStore,
    ) -> Result<Self, RuntimeEffectControllerError> {
        Ok(Self {
            model: request.model.clone(),
            messages: request.messages.clone(),
            attachments: attachment_specs_from_attachments(&request.attachments, attachment_store)
                .await?,
            tools: Arc::clone(&request.tools),
            tool_choice: request.tool_choice.clone(),
            model_variant: request.model_variant.clone(),
            generation: request.generation.clone(),
            scope: request.scope.clone(),
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
            tools: self.tools,
            tool_choice: self.tool_choice,
            model_variant: self.model_variant,
            generation: self.generation,
            scope: self.scope,
            output_spec: self.output_spec,
            stream_events,
            provider_trace,
        }
    }
}

async fn attachment_specs_from_attachments(
    attachments: &[LlmAttachment],
    attachment_store: &dyn AttachmentStore,
) -> Result<Vec<LlmAttachmentSpec>, RuntimeEffectControllerError> {
    let mut specs = Vec::with_capacity(attachments.len());
    for attachment in attachments {
        specs.push(attachment_spec_from_attachment(attachment, attachment_store).await?);
    }
    Ok(specs)
}

async fn attachment_spec_from_attachment(
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
        .await
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

    pub fn into_direct_response(
        self,
    ) -> Result<Result<LlmResponse, LlmCallError>, RuntimeEffectControllerError> {
        match self {
            Self::Direct { result } => Ok(result),
            other => Err(RuntimeEffectControllerError::wrong_outcome(
                RuntimeEffectKind::Direct,
                other.kind(),
            )),
        }
    }

    pub fn into_tool_attempt_effect(
        self,
    ) -> Result<ToolAttemptEffectOutcome, RuntimeEffectControllerError> {
        match self {
            Self::ToolAttempt { launch, triggers } => {
                Ok(ToolAttemptEffectOutcome { launch, triggers })
            }
            other => Err(RuntimeEffectControllerError::wrong_outcome(
                RuntimeEffectKind::ToolAttempt,
                other.kind(),
            )),
        }
    }

    pub fn into_tool_batch_effect(
        self,
    ) -> Result<ToolBatchEffectOutcome, RuntimeEffectControllerError> {
        match self {
            Self::ToolBatch { launches, triggers } => {
                Ok(ToolBatchEffectOutcome { launches, triggers })
            }
            other => Err(RuntimeEffectControllerError::wrong_outcome(
                RuntimeEffectKind::ToolBatch,
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

    pub fn into_sync_execution_environment(
        self,
    ) -> Result<Result<Option<ExecutionEnvironmentSync>, String>, RuntimeEffectControllerError>
    {
        match self {
            Self::SyncExecutionEnvironment { result } => Ok(result),
            other => Err(RuntimeEffectControllerError::wrong_outcome(
                RuntimeEffectKind::SyncExecutionEnvironment,
                other.kind(),
            )),
        }
    }

    pub fn into_await_event(self) -> Result<crate::Resolution, RuntimeEffectControllerError> {
        match self {
            Self::AwaitEvent { resolution } => Ok(resolution),
            other => Err(RuntimeEffectControllerError::wrong_outcome(
                RuntimeEffectKind::AwaitEvent,
                other.kind(),
            )),
        }
    }

    pub fn into_durable_step(self) -> Result<serde_json::Value, RuntimeEffectControllerError> {
        match self {
            Self::DurableStep { value } => Ok(value),
            other => Err(RuntimeEffectControllerError::wrong_outcome(
                RuntimeEffectKind::DurableStep,
                other.kind(),
            )),
        }
    }

    pub fn kind(&self) -> RuntimeEffectKind {
        match self {
            Self::LlmCall { .. } => RuntimeEffectKind::LlmCall,
            Self::Direct { .. } => RuntimeEffectKind::Direct,
            Self::ToolAttempt { .. } => RuntimeEffectKind::ToolAttempt,
            Self::ToolBatch { .. } => RuntimeEffectKind::ToolBatch,
            Self::Process { .. } => RuntimeEffectKind::Process,
            Self::ExecCode { .. } => RuntimeEffectKind::ExecCode,
            Self::Checkpoint { .. } => RuntimeEffectKind::Checkpoint,
            Self::SyncExecutionEnvironment { .. } => RuntimeEffectKind::SyncExecutionEnvironment,
            Self::Sleep => RuntimeEffectKind::Sleep,
            Self::AwaitEvent { .. } => RuntimeEffectKind::AwaitEvent,
            Self::DurableStep { .. } => RuntimeEffectKind::DurableStep,
        }
    }
}
