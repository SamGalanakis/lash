use futures_util::FutureExt;
use serde::{Deserialize, Serialize};
use sha2::Digest;
use std::collections::HashMap;
use std::future::Future;
use std::panic::AssertUnwindSafe;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::{Mutex, mpsc};
use tokio_util::sync::CancellationToken;

use crate::llm::transport::LlmTransportError;
use crate::llm::types::{
    LlmAttachment, LlmEventSender, LlmMessage, LlmOutputSpec, LlmProviderTraceSender, LlmResponse,
    LlmToolChoice, LlmToolSpec,
};
use crate::plugin::{DirectCompletion, DirectLlmCompletion, PluginMessage};
use crate::provider::ProviderHandle;
use crate::sansio::{
    CompletedToolCall, EffectId, ExecutionSurfaceSync, LlmCallError, PendingToolCall,
};
use crate::session_model::TokenUsage;
use crate::{
    AttachmentCreateMeta, AttachmentRef, AttachmentStore, CheckpointKind, DirectMessage,
    DirectOutputSpec, DirectRequest, ExecResponse, LlmRequest as CoreLlmRequest, MediaType,
    PluginError, RuntimeError,
};

use super::host::{
    BackgroundCancelPolicy, BackgroundTaskCompletion, BackgroundTaskRecord,
    BackgroundTaskRegistration, BackgroundTaskRegistry, BackgroundTaskState,
};
use super::session_manager::{CurrentSessionCapability, UsageCapability};
use super::{RuntimeStreamEvent, RuntimeTurnDriver};

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
            Self::ExecCode => "exec_code",
            Self::Checkpoint => "checkpoint",
            Self::SyncExecutionSurface => "sync_execution_surface",
            Self::Sleep => "sleep",
        }
    }
}

/// Borrowed durable effect controller for one runtime execution.
///
/// Durable integrations create one scope per externally
/// identified run and pass it to the scoped runtime entrypoints, making the
/// turn identity part of the idempotency contract rather than a tracing-only
/// hint.
#[derive(Clone, Copy)]
pub struct RuntimeEffectControllerScope<'run> {
    controller: &'run dyn RuntimeEffectController,
    turn_id: &'run str,
}

impl<'run> RuntimeEffectControllerScope<'run> {
    pub fn new(
        controller: &'run dyn RuntimeEffectController,
        turn_id: &'run str,
    ) -> Result<Self, RuntimeError> {
        if turn_id.is_empty() {
            return Err(RuntimeError {
                code: "missing_effect_scope_turn_id".to_string(),
                message: "scoped durable runs require a non-empty stable turn_id".to_string(),
            });
        }
        Ok(Self {
            controller,
            turn_id,
        })
    }

    pub fn controller(&self) -> &'run dyn RuntimeEffectController {
        self.controller
    }

    pub fn turn_id(&self) -> &'run str {
        self.turn_id
    }
}

/// Runtime-internal handle for effect-controller references carried through
/// per-turn execution contexts.
#[derive(Clone)]
pub(crate) enum RuntimeEffectControllerHandle<'run> {
    Borrowed(&'run dyn RuntimeEffectController),
    Shared(Arc<dyn RuntimeEffectController>),
}

impl<'run> RuntimeEffectControllerHandle<'run> {
    pub(crate) fn borrowed(controller: &'run dyn RuntimeEffectController) -> Self {
        Self::Borrowed(controller)
    }

    pub(crate) fn shared(controller: Arc<dyn RuntimeEffectController>) -> Self {
        Self::Shared(controller)
    }

    pub(crate) fn as_controller(&self) -> &dyn RuntimeEffectController {
        match self {
            Self::Borrowed(controller) => *controller,
            Self::Shared(controller) => controller.as_ref(),
        }
    }

    pub(crate) fn clone_scoped(&self) -> RuntimeEffectControllerHandle<'run> {
        self.clone()
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
        let bytes = serde_json::to_vec(self).map_err(|err| {
            RuntimeEffectControllerError::new(
                "runtime_effect_envelope_hash",
                format!("failed to serialize runtime effect envelope: {err}"),
            )
        })?;
        Ok(format!("{:x}", sha2::Sha256::digest(&bytes)))
    }
}

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
        call: PendingToolCall,
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
            Self::ExecCode { .. } => RuntimeEffectKind::ExecCode,
            Self::Checkpoint { .. } => RuntimeEffectKind::Checkpoint,
            Self::SyncExecutionSurface { .. } => RuntimeEffectKind::SyncExecutionSurface,
            Self::Sleep { .. } => RuntimeEffectKind::Sleep,
        }
    }
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
            Self::ExecCode { .. } => RuntimeEffectKind::ExecCode,
            Self::Checkpoint { .. } => RuntimeEffectKind::Checkpoint,
            Self::SyncExecutionSurface { .. } => RuntimeEffectKind::SyncExecutionSurface,
            Self::Sleep => RuntimeEffectKind::Sleep,
        }
    }
}

#[derive(Clone, Debug, thiserror::Error, Serialize, Deserialize)]
#[error("{code}: {message}")]
pub struct RuntimeEffectControllerError {
    pub code: String,
    pub message: String,
}

impl RuntimeEffectControllerError {
    pub fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
        }
    }

    fn wrong_outcome(expected: RuntimeEffectKind, actual: RuntimeEffectKind) -> Self {
        Self::new(
            "runtime_effect_wrong_outcome",
            format!(
                "expected {} outcome, got {}",
                expected.as_str(),
                actual.as_str()
            ),
        )
    }

    pub(crate) fn into_runtime_error(self) -> RuntimeError {
        RuntimeError {
            code: self.code,
            message: self.message,
        }
    }
}

impl From<RuntimeError> for RuntimeEffectControllerError {
    fn from(err: RuntimeError) -> Self {
        Self::new(err.code, err.message)
    }
}

impl From<PluginError> for RuntimeEffectControllerError {
    fn from(err: PluginError) -> Self {
        Self::new("plugin", err.to_string())
    }
}

struct LocalTurnEffectRunner<'a, 'run> {
    driver: &'a mut RuntimeTurnDriver<'run>,
    machine: &'a mut crate::TurnMachine,
    event_tx: mpsc::Sender<RuntimeStreamEvent>,
    cancellation: CancellationToken,
}

struct LocalDirectEffectRunner {
    provider: ProviderHandle,
    attachment_store: Arc<dyn AttachmentStore>,
}

pub type BackgroundTaskFuture =
    Pin<Box<dyn Future<Output = BackgroundTaskCompletion> + Send + 'static>>;

/// One-shot local executor for a background run container.
pub struct BackgroundTaskLocalExecutor {
    run: Box<dyn FnOnce(CancellationToken) -> BackgroundTaskFuture + Send + 'static>,
}

impl BackgroundTaskLocalExecutor {
    pub fn new<F, Fut>(run: F) -> Self
    where
        F: FnOnce(CancellationToken) -> Fut + Send + 'static,
        Fut: Future<Output = BackgroundTaskCompletion> + Send + 'static,
    {
        Self {
            run: Box::new(move |cancellation| Box::pin(run(cancellation))),
        }
    }

    fn run(self, cancellation: CancellationToken) -> BackgroundTaskFuture {
        (self.run)(cancellation)
    }
}

#[async_trait::async_trait]
trait RuntimeEffectLocalRunner: Send {
    async fn execute(
        self: Box<Self>,
        envelope: RuntimeEffectEnvelope,
    ) -> Result<RuntimeEffectOutcome, RuntimeEffectControllerError>;
}

enum RuntimeEffectLocalExecutorState<'run> {
    Unavailable,
    SleepOnly { cancellation: CancellationToken },
    Runner(Box<dyn RuntimeEffectLocalRunner + Send + 'run>),
}

/// Scoped local executor provided to a [`RuntimeEffectController`] for one effect.
///
/// Durable controllers may ignore it and replay their own recorded result. The
/// default inline controller delegates to it, so local provider/tool/checkpoint
/// work still crosses the same `execute_effect` boundary as durable controllers.
pub struct RuntimeEffectLocalExecutor<'run> {
    state: RuntimeEffectLocalExecutorState<'run>,
}

impl<'run> RuntimeEffectLocalExecutor<'run> {
    pub fn unavailable() -> Self {
        Self {
            state: RuntimeEffectLocalExecutorState::Unavailable,
        }
    }

    pub fn sleep(cancellation: CancellationToken) -> Self {
        Self {
            state: RuntimeEffectLocalExecutorState::SleepOnly { cancellation },
        }
    }

    pub(in crate::runtime) fn turn<'scope>(
        driver: &'run mut RuntimeTurnDriver<'scope>,
        machine: &'run mut crate::TurnMachine,
        event_tx: mpsc::Sender<RuntimeStreamEvent>,
        cancellation: CancellationToken,
    ) -> Self
    where
        'scope: 'run,
    {
        Self {
            state: RuntimeEffectLocalExecutorState::Runner(Box::new(LocalTurnEffectRunner {
                driver,
                machine,
                event_tx,
                cancellation,
            })),
        }
    }

    pub(in crate::runtime) fn direct(
        provider: ProviderHandle,
        attachment_store: Arc<dyn AttachmentStore>,
    ) -> Self {
        Self {
            state: RuntimeEffectLocalExecutorState::Runner(Box::new(LocalDirectEffectRunner {
                provider,
                attachment_store,
            })),
        }
    }

    pub async fn execute(
        self,
        envelope: RuntimeEffectEnvelope,
    ) -> Result<RuntimeEffectOutcome, RuntimeEffectControllerError> {
        match self.state {
            RuntimeEffectLocalExecutorState::Runner(runner) => runner.execute(envelope).await,
            RuntimeEffectLocalExecutorState::SleepOnly { cancellation } => {
                execute_local_sleep(envelope, cancellation).await
            }
            RuntimeEffectLocalExecutorState::Unavailable => Err(RuntimeEffectControllerError::new(
                "runtime_effect_local_executor_unavailable",
                format!(
                    "no local executor is available for {}",
                    envelope.command.kind().as_str()
                ),
            )),
        }
    }
}

#[async_trait::async_trait]
impl RuntimeEffectLocalRunner for LocalTurnEffectRunner<'_, '_> {
    async fn execute(
        self: Box<Self>,
        envelope: RuntimeEffectEnvelope,
    ) -> Result<RuntimeEffectOutcome, RuntimeEffectControllerError> {
        let runner = *self;
        match envelope.command {
            RuntimeEffectCommand::LlmCall { request } => {
                let mode_iteration = runner.machine.mode_iteration();
                let (result, text_streamed) = runner
                    .driver
                    .run_standard_llm_call(
                        Arc::new(request.into_request(None, None)),
                        mode_iteration,
                        &runner.event_tx,
                        &runner.cancellation,
                    )
                    .await;
                Ok(RuntimeEffectOutcome::LlmCall {
                    result,
                    text_streamed,
                })
            }
            RuntimeEffectCommand::ToolCall { call } => {
                let tool_name = call.tool_name.clone();
                let mut results = runner
                    .driver
                    .run_tool_calls(
                        vec![(call, envelope.metadata)],
                        &runner.event_tx,
                        &runner.cancellation,
                    )
                    .await?;
                let result = results.pop().ok_or_else(|| {
                    RuntimeEffectControllerError::new(
                        "tool_result_missing",
                        format!("tool `{tool_name}` completed without a result"),
                    )
                })?;
                Ok(RuntimeEffectOutcome::ToolCall { result })
            }
            RuntimeEffectCommand::ExecCode { code } => {
                let mode_iteration = runner.machine.mode_iteration();
                let messages = runner.machine.message_sequence();
                Ok(RuntimeEffectOutcome::ExecCode {
                    result: runner
                        .driver
                        .run_exec_code(&code, messages, mode_iteration, &runner.event_tx)
                        .await,
                })
            }
            RuntimeEffectCommand::Checkpoint { checkpoint } => {
                Ok(RuntimeEffectOutcome::Checkpoint {
                    result: runner
                        .driver
                        .run_checkpoint(runner.machine, checkpoint, &runner.event_tx)
                        .await
                        .map_err(RuntimeEffectControllerError::from),
                })
            }
            RuntimeEffectCommand::SyncExecutionSurface {
                update_machine_config,
            } => Ok(RuntimeEffectOutcome::SyncExecutionSurface {
                result: runner
                    .driver
                    .refresh_execution_surface(runner.machine, update_machine_config)
                    .await
                    .map_err(|err| err.to_string()),
            }),
            RuntimeEffectCommand::Sleep { duration_ms } => {
                sleep_with_cancellation(duration_ms, &runner.cancellation).await?;
                Ok(RuntimeEffectOutcome::Sleep)
            }
            command => Err(RuntimeEffectControllerError::new(
                "runtime_effect_local_executor_mismatch",
                format!(
                    "local turn executor cannot execute {} command",
                    command.kind().as_str()
                ),
            )),
        }
    }
}

#[async_trait::async_trait]
impl RuntimeEffectLocalRunner for LocalDirectEffectRunner {
    async fn execute(
        mut self: Box<Self>,
        envelope: RuntimeEffectEnvelope,
    ) -> Result<RuntimeEffectOutcome, RuntimeEffectControllerError> {
        match envelope.command {
            RuntimeEffectCommand::DirectCompletion {
                normalized_request, ..
            } => Ok(RuntimeEffectOutcome::DirectCompletion {
                result: self
                    .run_direct_llm_request(normalized_request.into_request(None, None))
                    .await,
            }),
            RuntimeEffectCommand::DirectLlmCompletion { request, .. } => {
                Ok(RuntimeEffectOutcome::DirectLlmCompletion {
                    result: self
                        .run_direct_llm_request(request.into_request(None, None))
                        .await,
                })
            }
            RuntimeEffectCommand::Sleep { duration_ms } => {
                sleep_with_cancellation(duration_ms, &CancellationToken::new()).await?;
                Ok(RuntimeEffectOutcome::Sleep)
            }
            command => Err(RuntimeEffectControllerError::new(
                "runtime_effect_local_executor_mismatch",
                format!(
                    "local direct executor cannot execute {} command",
                    command.kind().as_str()
                ),
            )),
        }
    }
}

impl LocalDirectEffectRunner {
    async fn run_direct_llm_request(
        &mut self,
        request: CoreLlmRequest,
    ) -> Result<LlmResponse, LlmCallError> {
        let request = crate::attachments::resolve_llm_request_attachments(
            request,
            self.attachment_store.as_ref(),
        )
        .map_err(|err| LlmCallError {
            message: err.to_string(),
            retryable: false,
            raw: None,
            code: Some("attachment_resolution_failed".to_string()),
            terminal_reason: crate::LlmTerminalReason::ProviderError,
            request_body: None,
        })?;
        self.provider
            .complete(request)
            .await
            .map_err(llm_call_error_from_transport)
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

async fn execute_local_sleep(
    envelope: RuntimeEffectEnvelope,
    cancellation: CancellationToken,
) -> Result<RuntimeEffectOutcome, RuntimeEffectControllerError> {
    match envelope.command {
        RuntimeEffectCommand::Sleep { duration_ms } => {
            sleep_with_cancellation(duration_ms, &cancellation).await?;
            Ok(RuntimeEffectOutcome::Sleep)
        }
        command => Err(RuntimeEffectControllerError::new(
            "runtime_effect_local_executor_mismatch",
            format!(
                "local sleep executor cannot execute {} command",
                command.kind().as_str()
            ),
        )),
    }
}

async fn sleep_with_cancellation(
    duration_ms: u64,
    cancellation: &CancellationToken,
) -> Result<(), RuntimeEffectControllerError> {
    let sleep = tokio::time::sleep(std::time::Duration::from_millis(duration_ms));
    tokio::pin!(sleep);
    tokio::select! {
        _ = cancellation.cancelled() => Err(RuntimeEffectControllerError::new(
            "runtime_effect_sleep_cancelled",
            "runtime effect sleep was cancelled",
        )),
        _ = &mut sleep => Ok(()),
    }
}

pub(in crate::runtime) fn token_usage_from_llm(usage: &crate::llm::types::LlmUsage) -> TokenUsage {
    TokenUsage {
        input_tokens: usage.input_tokens,
        output_tokens: usage.output_tokens,
        cached_input_tokens: usage.cached_input_tokens,
        reasoning_tokens: usage.reasoning_tokens,
    }
}

fn llm_call_error_from_transport(err: LlmTransportError) -> LlmCallError {
    LlmCallError {
        message: err.message,
        retryable: err.retryable,
        raw: err.raw,
        code: err.code,
        terminal_reason: err.terminal_reason,
        request_body: err.request_body,
    }
}

pub(in crate::runtime) async fn apply_direct_completion_outcome(
    current: &CurrentSessionCapability,
    usage_capability: &UsageCapability,
    request: &DirectRequest,
    normalized_request: &CoreLlmRequest,
    model: &str,
    usage_source: &str,
    outcome: RuntimeEffectOutcome,
) -> Result<DirectCompletion, PluginError> {
    let result = outcome
        .into_direct_completion_response()
        .map_err(|err| PluginError::Session(err.to_string()))?;
    let (response, usage) = apply_direct_llm_result(
        current,
        usage_capability,
        normalized_request,
        usage_source,
        model,
        request.originating_tool_call_id.as_deref(),
        result,
    )
    .await?;
    Ok(DirectCompletion {
        text: response.full_text,
        usage,
    })
}

pub(in crate::runtime) async fn apply_direct_llm_completion_outcome(
    current: &CurrentSessionCapability,
    usage_capability: &UsageCapability,
    request: &CoreLlmRequest,
    usage_source: &str,
    outcome: RuntimeEffectOutcome,
) -> Result<DirectLlmCompletion, PluginError> {
    let result = outcome
        .into_direct_llm_completion_response()
        .map_err(|err| PluginError::Session(err.to_string()))?;
    let model = request.model.clone();
    let (response, usage) = apply_direct_llm_result(
        current,
        usage_capability,
        request,
        usage_source,
        &model,
        None,
        result,
    )
    .await?;
    Ok(DirectLlmCompletion { response, usage })
}

async fn apply_direct_llm_result(
    current: &CurrentSessionCapability,
    usage_capability: &UsageCapability,
    request: &CoreLlmRequest,
    usage_source: &str,
    usage_model: &str,
    originating_tool_call_id: Option<&str>,
    result: Result<LlmResponse, LlmCallError>,
) -> Result<(LlmResponse, TokenUsage), PluginError> {
    let llm_call_id = emit_direct_llm_trace_started(current, request, originating_tool_call_id);
    match result {
        Ok(response) => {
            emit_direct_llm_trace_completed(
                current,
                llm_call_id.as_deref(),
                originating_tool_call_id,
                &response,
            );
            let usage = token_usage_from_llm(&response.usage);
            usage_capability.record_token_usage(usage_source, usage_model, &usage);
            usage_capability
                .persist_current_usage_ledger(current)
                .await?;
            Ok((response, usage))
        }
        Err(err) => {
            emit_direct_llm_trace_failed(
                current,
                llm_call_id.as_deref(),
                originating_tool_call_id,
                &err,
            );
            Err(PluginError::Session(err.message))
        }
    }
}

fn emit_direct_llm_trace_started(
    current: &CurrentSessionCapability,
    request: &CoreLlmRequest,
    originating_tool_call_id: Option<&str>,
) -> Option<String> {
    current.host.core.trace_sink.as_ref()?;
    let llm_call_id = uuid::Uuid::new_v4().to_string();
    emit_llm_trace_started(
        &current.host.core.trace_sink,
        &current.host.core.trace_context,
        direct_trace_context(
            &current.session_id,
            Some(&llm_call_id),
            originating_tool_call_id,
        ),
        request,
    );
    Some(llm_call_id)
}

fn emit_direct_llm_trace_completed(
    current: &CurrentSessionCapability,
    llm_call_id: Option<&str>,
    originating_tool_call_id: Option<&str>,
    response: &LlmResponse,
) {
    let Some(llm_call_id) = llm_call_id else {
        return;
    };
    emit_llm_trace_completed(
        &current.host.core.trace_sink,
        &current.host.core.trace_context,
        direct_trace_context(
            &current.session_id,
            Some(llm_call_id),
            originating_tool_call_id,
        ),
        response,
        0,
        None,
    );
}

fn emit_direct_llm_trace_failed(
    current: &CurrentSessionCapability,
    llm_call_id: Option<&str>,
    originating_tool_call_id: Option<&str>,
    err: &LlmCallError,
) {
    let Some(llm_call_id) = llm_call_id else {
        return;
    };
    emit_llm_trace_failed(
        &current.host.core.trace_sink,
        &current.host.core.trace_context,
        direct_trace_context(
            &current.session_id,
            Some(llm_call_id),
            originating_tool_call_id,
        ),
        LlmTraceFailure::from(err),
        None,
    );
}

fn direct_trace_context(
    session_id: &str,
    llm_call_id: Option<&str>,
    originating_tool_call_id: Option<&str>,
) -> lash_trace::TraceContext {
    let mut context = lash_trace::TraceContext::default().for_session(session_id.to_string());
    if let Some(llm_call_id) = llm_call_id {
        context = context.for_llm_call(llm_call_id.to_string());
    }
    if let Some(originating_tool_call_id) = originating_tool_call_id {
        context = context.for_originating_tool_call(originating_tool_call_id.to_string());
    }
    context
}

pub(in crate::runtime) fn emit_llm_trace_started(
    trace_sink: &Option<Arc<dyn lash_trace::TraceSink>>,
    base_context: &lash_trace::TraceContext,
    context: lash_trace::TraceContext,
    request: &CoreLlmRequest,
) {
    crate::trace::emit_trace(
        trace_sink,
        base_context,
        context,
        lash_trace::TraceEvent::LlmCallStarted {
            request: crate::trace::trace_llm_request(request),
        },
    );
}

pub(in crate::runtime) fn emit_llm_trace_completed(
    trace_sink: &Option<Arc<dyn lash_trace::TraceSink>>,
    base_context: &lash_trace::TraceContext,
    context: lash_trace::TraceContext,
    response: &LlmResponse,
    duration_ms: u64,
    stream_summary: Option<serde_json::Value>,
) {
    crate::trace::emit_trace(
        trace_sink,
        base_context,
        context,
        lash_trace::TraceEvent::LlmCallCompleted {
            response: crate::trace::trace_llm_response(
                response.full_text.clone(),
                duration_ms,
                Some(response.terminal_reason),
                crate::trace::trace_output_parts(&response.parts),
            ),
            usage: Some(crate::trace::trace_usage_from_llm(&response.usage)),
            provider_usage: response.provider_usage.clone(),
            stream_summary,
        },
    );
}

pub(in crate::runtime) struct LlmTraceFailure {
    message: String,
    retryable: bool,
    terminal_reason: crate::LlmTerminalReason,
    code: Option<String>,
    raw: Option<String>,
}

impl From<&LlmTransportError> for LlmTraceFailure {
    fn from(err: &LlmTransportError) -> Self {
        Self {
            message: err.message.clone(),
            retryable: err.retryable,
            terminal_reason: err.terminal_reason,
            code: err.code.clone(),
            raw: err.raw.clone(),
        }
    }
}

impl From<&LlmCallError> for LlmTraceFailure {
    fn from(err: &LlmCallError) -> Self {
        Self {
            message: err.message.clone(),
            retryable: err.retryable,
            terminal_reason: err.terminal_reason,
            code: err.code.clone(),
            raw: err.raw.clone(),
        }
    }
}

pub(in crate::runtime) fn emit_llm_trace_failed(
    trace_sink: &Option<Arc<dyn lash_trace::TraceSink>>,
    base_context: &lash_trace::TraceContext,
    context: lash_trace::TraceContext,
    failure: LlmTraceFailure,
    stream_summary: Option<serde_json::Value>,
) {
    crate::trace::emit_trace(
        trace_sink,
        base_context,
        context,
        lash_trace::TraceEvent::LlmCallFailed {
            error: lash_trace::TraceError {
                message: failure.message,
                retryable: failure.retryable,
                terminal_reason: Some(failure.terminal_reason.code().to_string()),
                code: failure.code,
                raw: failure.raw,
            },
            stream_summary,
        },
    );
}

/// Boundary for nondeterministic runtime work.
#[async_trait::async_trait]
pub trait RuntimeEffectController: Send + Sync {
    fn requires_durable_attachment_store(&self) -> bool {
        false
    }

    async fn execute_effect(
        &self,
        envelope: RuntimeEffectEnvelope,
        local_executor: RuntimeEffectLocalExecutor<'_>,
    ) -> Result<RuntimeEffectOutcome, RuntimeEffectControllerError>;

    async fn start_background_task(
        &self,
        registry: Arc<dyn BackgroundTaskRegistry>,
        registration: BackgroundTaskRegistration,
        local_executor: BackgroundTaskLocalExecutor,
    ) -> Result<BackgroundTaskRecord, PluginError>;

    async fn request_background_task_cancel(
        &self,
        registry: Arc<dyn BackgroundTaskRegistry>,
        task_id: &str,
        reason: Option<String>,
    ) -> Result<BackgroundTaskRecord, PluginError>;
}

/// Default in-process effect controller.
#[derive(Clone, Default)]
pub struct InlineRuntimeEffectController {
    background_tasks: Arc<Mutex<HashMap<String, LocalBackgroundExecution>>>,
}

#[async_trait::async_trait]
impl RuntimeEffectController for InlineRuntimeEffectController {
    async fn execute_effect(
        &self,
        envelope: RuntimeEffectEnvelope,
        local_executor: RuntimeEffectLocalExecutor<'_>,
    ) -> Result<RuntimeEffectOutcome, RuntimeEffectControllerError> {
        local_executor.execute(envelope).await
    }

    async fn start_background_task(
        &self,
        registry: Arc<dyn BackgroundTaskRegistry>,
        registration: BackgroundTaskRegistration,
        local_executor: BackgroundTaskLocalExecutor,
    ) -> Result<BackgroundTaskRecord, PluginError> {
        let task_id = registration.id.clone();
        let running = registry.mark_running(&task_id).await?;
        let cancellation = CancellationToken::new();
        let task_cancellation = cancellation.clone();
        let registry_for_task = Arc::clone(&registry);
        let tasks = Arc::clone(&self.background_tasks);
        let task_id_for_task = task_id.clone();
        let handle = tokio::spawn(async move {
            let future = local_executor.run(task_cancellation);
            let completion = AssertUnwindSafe(future).catch_unwind().await;
            let completion = match completion {
                Ok(completion) if completion.state.is_terminal() => completion,
                Ok(completion) => BackgroundTaskCompletion {
                    state: BackgroundTaskState::Failed,
                    summary: Some(format!(
                        "background task returned non-terminal state {:?}",
                        completion.state
                    )),
                    output: completion.output,
                },
                Err(_) => BackgroundTaskCompletion {
                    state: BackgroundTaskState::Failed,
                    summary: Some("background task panicked".to_string()),
                    output: Some(crate::ToolCallOutput::failure(crate::ToolFailure::runtime(
                        crate::ToolFailureClass::Internal,
                        "background_task_panicked",
                        "background task panicked",
                    ))),
                },
            };
            let _ = registry_for_task
                .complete(&task_id_for_task, completion)
                .await;
            tasks.lock().await.remove(&task_id_for_task);
        });
        self.background_tasks.lock().await.insert(
            task_id,
            LocalBackgroundExecution {
                cancellation,
                abort: handle.abort_handle(),
            },
        );
        drop(handle);
        Ok(running)
    }

    async fn request_background_task_cancel(
        &self,
        registry: Arc<dyn BackgroundTaskRegistry>,
        task_id: &str,
        reason: Option<String>,
    ) -> Result<BackgroundTaskRecord, PluginError> {
        let record = registry.request_cancel(task_id, reason.clone()).await?;
        let execution = self.background_tasks.lock().await.get(task_id).cloned();
        if let Some(execution) = execution {
            execution.cancellation.cancel();
            if matches!(record.cancel_policy, BackgroundCancelPolicy::AbortLocal) {
                execution.abort.abort();
                self.background_tasks.lock().await.remove(task_id);
                let message = reason.unwrap_or_else(|| "background task was cancelled".to_string());
                return registry
                    .complete(
                        task_id,
                        BackgroundTaskCompletion {
                            state: BackgroundTaskState::Cancelled,
                            summary: Some(message.clone()),
                            output: Some(crate::ToolCallOutput::cancelled(
                                crate::ToolCancellation::runtime(message),
                            )),
                        },
                    )
                    .await;
            }
        }
        Ok(record)
    }
}

#[derive(Clone)]
struct LocalBackgroundExecution {
    cancellation: CancellationToken,
    abort: tokio::task::AbortHandle,
}

impl std::fmt::Debug for InlineRuntimeEffectController {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InlineRuntimeEffectController")
            .finish_non_exhaustive()
    }
}

pub(crate) fn turn_idempotency_key(
    session_id: &str,
    turn_id: &str,
    turn_index: usize,
    mode_iteration: usize,
    kind: RuntimeEffectKind,
    effect_id: EffectId,
) -> String {
    format!(
        "{session_id}:{turn_id}:{turn_index}:{mode_iteration}:{}:{}",
        kind.as_str(),
        effect_id.0
    )
}

pub(crate) fn direct_effect_metadata(
    session_id: &str,
    usage_source: &str,
    effect_kind: RuntimeEffectKind,
    idempotency_discriminator: String,
    turn_id: Option<&str>,
) -> EffectInvocationMetadata {
    let origin = match effect_kind {
        RuntimeEffectKind::DirectCompletion => EffectOrigin::DirectCompletion {
            usage_source: usage_source.to_string(),
        },
        RuntimeEffectKind::DirectLlmCompletion => EffectOrigin::DirectLlmCompletion {
            usage_source: usage_source.to_string(),
        },
        _ => unreachable!("direct invocation requires a direct effect kind"),
    };
    let idempotency_key = match turn_id.filter(|value| !value.is_empty()) {
        Some(turn_id) => format!(
            "{session_id}:{turn_id}:direct:{}:{usage_source}:{idempotency_discriminator}",
            effect_kind.as_str()
        ),
        None => format!(
            "{session_id}:direct:{}:{usage_source}:{idempotency_discriminator}",
            effect_kind.as_str()
        ),
    };
    EffectInvocationMetadata {
        session_id: session_id.to_string(),
        origin,
        turn_id: turn_id.map(str::to_string),
        turn_index: None,
        mode_iteration: None,
        effect_id: idempotency_discriminator,
        effect_kind,
        idempotency_key,
        turn_checkpoint_hash: None,
    }
}

pub(crate) fn tool_retry_sleep_metadata(
    parent: &EffectInvocationMetadata,
    tool_name: &str,
    attempt: u32,
) -> EffectInvocationMetadata {
    let effect_id = format!("{}:{tool_name}:attempt:{attempt}:sleep", parent.effect_id);
    let idempotency_key = format!(
        "{}:{tool_name}:attempt:{attempt}:sleep",
        parent.idempotency_key
    );
    EffectInvocationMetadata {
        session_id: parent.session_id.clone(),
        origin: parent.origin.clone(),
        turn_id: parent.turn_id.clone(),
        turn_index: parent.turn_index,
        mode_iteration: parent.mode_iteration,
        effect_id,
        effect_kind: RuntimeEffectKind::Sleep,
        idempotency_key,
        turn_checkpoint_hash: parent.turn_checkpoint_hash.clone(),
    }
}

pub(crate) fn direct_request_discriminator<T>(
    request: &T,
    explicit_key: Option<&str>,
    parent_tool_call_id: Option<&str>,
) -> Result<String, RuntimeEffectControllerError>
where
    T: Serialize,
{
    if let Some(explicit_key) = explicit_key.filter(|key| !key.is_empty()) {
        return Ok(match parent_tool_call_id.filter(|id| !id.is_empty()) {
            Some(parent) => format!("tool:{parent}:request:{explicit_key}"),
            None => format!("request:{explicit_key}"),
        });
    }
    let bytes = serde_json::to_vec(request).map_err(|err| {
        RuntimeEffectControllerError::new(
            "runtime_effect_discriminator",
            format!("failed to serialize runtime effect discriminator: {err}"),
        )
    })?;
    let digest = format!("{:x}", sha2::Sha256::digest(&bytes));
    Ok(match parent_tool_call_id.filter(|id| !id.is_empty()) {
        Some(parent) => format!("tool:{parent}:sha256:{digest}"),
        None => format!("sha256:{digest}"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_effect_metadata_preserves_metadata_shape() {
        let metadata = direct_effect_metadata(
            "s",
            "tool",
            RuntimeEffectKind::DirectCompletion,
            "request:k".to_string(),
            None,
        );

        assert_eq!(metadata.session_id, "s");
        assert_eq!(
            metadata.origin,
            EffectOrigin::DirectCompletion {
                usage_source: "tool".to_string()
            }
        );
        assert!(
            metadata
                .idempotency_key
                .starts_with("s:direct:direct_completion:tool:request:k")
        );
        assert!(metadata.turn_checkpoint_hash.is_none());
    }

    #[test]
    fn tool_retry_sleep_metadata_preserves_parent_checkpoint_digest() {
        let mut parent = direct_effect_metadata(
            "s",
            "tool",
            RuntimeEffectKind::DirectCompletion,
            "request:k".to_string(),
            Some("turn"),
        );
        parent.turn_checkpoint_hash = Some("a".repeat(64));

        let sleep = tool_retry_sleep_metadata(&parent, "probe", 2);

        assert_eq!(sleep.effect_kind, RuntimeEffectKind::Sleep);
        assert_eq!(sleep.turn_checkpoint_hash, parent.turn_checkpoint_hash);
        assert!(sleep.idempotency_key.ends_with(":probe:attempt:2:sleep"));
    }

    #[test]
    fn runtime_effect_envelope_and_request_specs_round_trip_without_live_fields() {
        let attachment_store = crate::InMemoryAttachmentStore::new();
        let llm_request = CoreLlmRequest {
            model: "model".to_string(),
            messages: vec![LlmMessage::text(crate::llm::types::LlmRole::User, "hello")],
            attachments: vec![LlmAttachment::bytes("image/png", vec![1, 2, 3, 4])],
            tools: Arc::new(Vec::new()),
            tool_choice: LlmToolChoice::None,
            model_variant: Some("fast".to_string()),
            session_id: Some("session".to_string()),
            output_spec: None,
            stream_events: Some(LlmEventSender::new(|_| {})),
            provider_trace: Some(LlmProviderTraceSender::new(|_| {})),
        };
        let spec = LlmRequestSpec::from_request(&llm_request, &attachment_store).expect("llm spec");
        let encoded = serde_json::to_string(&spec).expect("serialize llm spec");
        assert!(!encoded.contains("stream_events"));
        assert!(!encoded.contains("provider_trace"));
        assert!(!encoded.contains("\"data\""));
        assert!(encoded.contains(crate::attachments::content_id(&[1, 2, 3, 4]).as_str()));
        let decoded: LlmRequestSpec = serde_json::from_str(&encoded).expect("decode llm spec");
        let live = decoded.into_request(None, None);
        assert_eq!(live.model, "model");
        assert!(live.attachments[0].data.is_empty());
        assert!(live.attachments[0].reference.is_some());
        assert!(live.stream_events.is_none());
        assert!(live.provider_trace.is_none());

        let direct_request = DirectRequest::text("model", "direct");
        let direct_spec = DirectRequestSpec::from_request(&direct_request, &attachment_store)
            .expect("direct spec");
        let metadata = direct_effect_metadata(
            "session",
            "test",
            RuntimeEffectKind::DirectCompletion,
            "request:direct".to_string(),
            Some("turn"),
        );
        let envelope = RuntimeEffectEnvelope::new(
            metadata,
            RuntimeEffectCommand::DirectCompletion {
                request: direct_spec,
                normalized_request: LlmRequestSpec::from_request(&llm_request, &attachment_store)
                    .expect("normalized spec"),
                model: "model".to_string(),
                usage_source: "test".to_string(),
            },
        );
        let hash = envelope.stable_hash().expect("stable hash");
        assert!(!hash.is_empty());
        let encoded = serde_json::to_string(&envelope).expect("serialize envelope");
        let decoded: RuntimeEffectEnvelope =
            serde_json::from_str(&encoded).expect("decode envelope");
        assert_eq!(
            decoded.metadata.idempotency_key,
            envelope.metadata.idempotency_key
        );
        assert_eq!(decoded.command.kind(), RuntimeEffectKind::DirectCompletion);
    }
}
