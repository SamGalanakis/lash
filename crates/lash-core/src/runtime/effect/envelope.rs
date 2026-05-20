use serde::{Deserialize, Serialize};

use crate::CheckpointKind;
use crate::plugin::PluginMessage;
use crate::sansio::{CompletedToolCall, ExecutionSurfaceSync, LlmCallError};
use crate::{
    ExecResponse, LlmResponse, ProcessAwaitOutput, ProcessExecutionContext,
    ProcessHandleGrantEntry, ProcessRecord, ProcessRegistration, ProcessStartGrant,
};

use super::controller::RuntimeEffectControllerError;
use super::spec::{DirectRequestSpec, LlmRequestSpec};

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
    pub mode_iteration: Option<usize>,
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
        request: LlmRequestSpec,
    },
    DirectCompletion {
        request: DirectRequestSpec,
        normalized_request: LlmRequestSpec,
        model: String,
        usage_source: String,
    },
    DirectLlmCompletion {
        request: LlmRequestSpec,
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
        #[serde(default, skip_serializing_if = "ProcessExecutionContext::is_empty")]
        execution_context: ProcessExecutionContext,
    },
    List {
        session_id: String,
    },
    Transfer {
        from_session_id: String,
        to_session_id: String,
        process_ids: Vec<String>,
    },
    Await {
        process_id: String,
    },
    Cancel {
        process_id: String,
        reason: Option<String>,
    },
}

impl ProcessCommand {
    pub fn effect_id(&self) -> String {
        match self {
            Self::Start { registration, .. } => format!("process:start:{}", registration.id),
            Self::List { session_id } => format!("process:list:{session_id}"),
            Self::Transfer {
                from_session_id,
                to_session_id,
                process_ids,
            } => {
                let digest = crate::stable_hash::stable_json_sha256_hex(process_ids)
                    .unwrap_or_else(|_| "unhashable".to_string());
                format!("process:transfer:{from_session_id}:{to_session_id}:{digest}")
            }
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

    pub fn into_checkpoint(
        self,
    ) -> Result<
        Result<(Vec<PluginMessage>, Vec<PluginMessage>), RuntimeEffectControllerError>,
        RuntimeEffectControllerError,
    > {
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
