use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::AttachmentStore;
use crate::LlmRequest as CoreLlmRequest;
use crate::LlmResponse;
use crate::provider::ProviderHandle;
use crate::runtime::host::BackgroundTaskCompletion;
use crate::runtime::{RuntimeStreamEvent, RuntimeTurnDriver};
use crate::sansio::LlmCallError;

use super::controller::RuntimeEffectControllerError;
use super::envelope::{RuntimeEffectCommand, RuntimeEffectEnvelope, RuntimeEffectOutcome};
use super::trace::llm_call_error_from_transport;

pub type BackgroundTaskFuture =
    Pin<Box<dyn Future<Output = BackgroundTaskCompletion> + Send + 'static>>;

/// Cancellation behavior available only to the inline/local background runner.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LocalBackgroundCancelPolicy {
    #[default]
    Cooperative,
    LocalAbort,
}

/// One-shot local executor for a background run container.
pub struct BackgroundTaskLocalExecutor {
    run: Box<dyn FnOnce(CancellationToken) -> BackgroundTaskFuture + Send + 'static>,
    cancel_policy: LocalBackgroundCancelPolicy,
}

impl BackgroundTaskLocalExecutor {
    pub fn new<F, Fut>(run: F) -> Self
    where
        F: FnOnce(CancellationToken) -> Fut + Send + 'static,
        Fut: Future<Output = BackgroundTaskCompletion> + Send + 'static,
    {
        Self {
            run: Box::new(move |cancellation| Box::pin(run(cancellation))),
            cancel_policy: LocalBackgroundCancelPolicy::Cooperative,
        }
    }

    pub fn with_cancel_policy(mut self, cancel_policy: LocalBackgroundCancelPolicy) -> Self {
        self.cancel_policy = cancel_policy;
        self
    }

    pub(super) fn cancel_policy(&self) -> LocalBackgroundCancelPolicy {
        self.cancel_policy
    }

    pub(super) fn run(self, cancellation: CancellationToken) -> BackgroundTaskFuture {
        (self.run)(cancellation)
    }
}

pub(super) struct LocalTurnEffectRunner<'a, 'run> {
    driver: &'a mut RuntimeTurnDriver<'run>,
    machine: &'a mut crate::TurnMachine,
    event_tx: mpsc::Sender<RuntimeStreamEvent>,
    cancellation: CancellationToken,
}

pub(super) struct LocalDirectEffectRunner {
    provider: ProviderHandle,
    attachment_store: Arc<dyn AttachmentStore>,
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
///
/// [`RuntimeEffectController`]: super::controller::RuntimeEffectController
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
