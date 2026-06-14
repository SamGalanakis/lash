#[cfg(any(test, feature = "testing"))]
use std::pin::Pin;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::AttachmentStore;
use crate::LlmRequest as CoreLlmRequest;
use crate::LlmResponse;
use crate::ProcessRecord;
use crate::ProcessRegistry;
use crate::provider::ProviderHandle;
use crate::runtime::{RuntimeStreamEvent, RuntimeTurnDriver};
use crate::sansio::LlmCallError;
use crate::{PluginError, RuntimeError, RuntimeErrorCode};

use super::envelope::{
    ProcessCommand, ProcessEffectOutcome, RuntimeEffectCommand, RuntimeEffectEnvelope,
    RuntimeEffectKind, RuntimeEffectOutcome,
};
use super::outcome::llm_call_error_from_transport;

// =============================================================================
// Effect host + controller trait + scope + error
// =============================================================================

/// Stable semantic identity for one effectful runtime operation.
///
/// The scope is chosen by the host boundary before any nondeterministic work is
/// planned. It is intentionally generic: Restate, an inline test host, or a
/// future durable effect host all receive the same Lash scope vocabulary.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EffectScope {
    Turn {
        session_id: String,
        turn_id: String,
    },
    Process {
        process_id: String,
    },
    QueueDrain {
        session_id: String,
        drain_id: String,
    },
    SessionDelete {
        session_id: String,
    },
    RuntimeOperation {
        operation_id: String,
    },
}

impl EffectScope {
    pub fn turn(session_id: impl Into<String>, turn_id: impl Into<String>) -> Self {
        Self::Turn {
            session_id: session_id.into(),
            turn_id: turn_id.into(),
        }
    }

    pub fn process(process_id: impl Into<String>) -> Self {
        Self::Process {
            process_id: process_id.into(),
        }
    }

    pub fn queue_drain(session_id: impl Into<String>, drain_id: impl Into<String>) -> Self {
        Self::QueueDrain {
            session_id: session_id.into(),
            drain_id: drain_id.into(),
        }
    }

    pub fn session_delete(session_id: impl Into<String>) -> Self {
        Self::SessionDelete {
            session_id: session_id.into(),
        }
    }

    pub fn runtime_operation(operation_id: impl Into<String>) -> Self {
        Self::RuntimeOperation {
            operation_id: operation_id.into(),
        }
    }

    pub fn id(&self) -> &str {
        match self {
            Self::Turn { turn_id, .. } => turn_id,
            Self::Process { process_id } => process_id,
            Self::QueueDrain { drain_id, .. } => drain_id,
            Self::SessionDelete { session_id } => session_id,
            Self::RuntimeOperation { operation_id } => operation_id,
        }
    }

    pub fn session_id(&self) -> Option<&str> {
        match self {
            Self::Turn { session_id, .. }
            | Self::QueueDrain { session_id, .. }
            | Self::SessionDelete { session_id } => Some(session_id),
            Self::Process { .. } | Self::RuntimeOperation { .. } => None,
        }
    }

    pub fn turn_id(&self) -> Option<&str> {
        match self {
            Self::Turn { turn_id, .. } => Some(turn_id),
            _ => None,
        }
    }

    pub fn validates_turn_trace_id(&self) -> bool {
        matches!(self, Self::Turn { .. })
    }

    fn validate(&self) -> Result<(), RuntimeError> {
        let missing = match self {
            Self::Turn {
                session_id,
                turn_id,
            } => session_id.trim().is_empty() || turn_id.trim().is_empty(),
            Self::Process { process_id } => process_id.trim().is_empty(),
            Self::QueueDrain {
                session_id,
                drain_id,
            } => session_id.trim().is_empty() || drain_id.trim().is_empty(),
            Self::SessionDelete { session_id } => session_id.trim().is_empty(),
            Self::RuntimeOperation { operation_id } => operation_id.trim().is_empty(),
        };
        if missing {
            return Err(RuntimeError::new(
                RuntimeErrorCode::MissingEffectScopeId,
                "effect scopes require non-empty stable ids",
            ));
        }
        Ok(())
    }
}

enum ScopedEffectControllerInner<'run> {
    Borrowed(&'run dyn RuntimeEffectController),
    Shared(Arc<dyn RuntimeEffectController>),
}

impl Clone for ScopedEffectControllerInner<'_> {
    fn clone(&self) -> Self {
        match self {
            Self::Borrowed(controller) => Self::Borrowed(*controller),
            Self::Shared(controller) => Self::Shared(Arc::clone(controller)),
        }
    }
}

/// Scoped low-level controller plus the semantic effect scope it is serving.
#[derive(Clone)]
pub struct ScopedEffectController<'run> {
    controller: ScopedEffectControllerInner<'run>,
    scope: EffectScope,
}

impl<'run> ScopedEffectController<'run> {
    pub fn borrowed(
        controller: &'run dyn RuntimeEffectController,
        scope: EffectScope,
    ) -> Result<Self, RuntimeError> {
        scope.validate()?;
        Ok(Self {
            controller: ScopedEffectControllerInner::Borrowed(controller),
            scope,
        })
    }

    pub fn shared(
        controller: Arc<dyn RuntimeEffectController>,
        scope: EffectScope,
    ) -> Result<Self, RuntimeError> {
        scope.validate()?;
        Ok(Self {
            controller: ScopedEffectControllerInner::Shared(controller),
            scope,
        })
    }

    pub fn controller(&self) -> &dyn RuntimeEffectController {
        match &self.controller {
            ScopedEffectControllerInner::Borrowed(controller) => *controller,
            ScopedEffectControllerInner::Shared(controller) => controller.as_ref(),
        }
    }

    pub fn effect_scope(&self) -> &EffectScope {
        &self.scope
    }

    pub fn scope_id(&self) -> &str {
        self.scope.id()
    }

    pub fn turn_id(&self) -> Option<&str> {
        self.scope.turn_id()
    }
}

/// Deployment-level factory for scoped effect controllers.
#[async_trait::async_trait]
pub trait EffectHost: Send + Sync {
    fn durability_tier(&self) -> crate::DurabilityTier {
        crate::DurabilityTier::Inline
    }

    fn requires_durable_attachment_store(&self) -> bool {
        false
    }

    fn scoped<'run>(
        &'run self,
        scope: EffectScope,
    ) -> Result<ScopedEffectController<'run>, RuntimeError>;

    fn scoped_static(
        &self,
        _scope: EffectScope,
    ) -> Result<Option<ScopedEffectController<'static>>, RuntimeError> {
        Ok(None)
    }
}

/// Boundary for nondeterministic runtime work.
#[async_trait::async_trait]
pub trait RuntimeEffectController: Send + Sync {
    /// Durability tier this controller provides; defaults to
    /// [`DurabilityTier::Inline`].
    fn durability_tier(&self) -> crate::DurabilityTier {
        crate::DurabilityTier::Inline
    }

    fn requires_durable_attachment_store(&self) -> bool {
        false
    }

    async fn execute_effect(
        &self,
        envelope: RuntimeEffectEnvelope,
        local_executor: RuntimeEffectLocalExecutor<'_>,
    ) -> Result<RuntimeEffectOutcome, RuntimeEffectControllerError>;
}

/// Runtime-internal handle for effect-controller references carried through
/// per-turn execution contexts.
#[derive(Clone)]
pub(crate) enum RuntimeEffectControllerHandle<'run> {
    Borrowed(ScopedEffectController<'run>),
    #[cfg(any(test, feature = "testing"))]
    Shared {
        controller: Arc<dyn RuntimeEffectController>,
        scope: EffectScope,
    },
}

impl<'run> RuntimeEffectControllerHandle<'run> {
    pub(crate) fn borrowed(scoped: ScopedEffectController<'run>) -> Self {
        Self::Borrowed(scoped)
    }

    #[cfg(any(test, feature = "testing"))]
    pub(crate) fn shared(controller: Arc<dyn RuntimeEffectController>) -> Self {
        Self::Shared {
            controller,
            scope: EffectScope::runtime_operation("test-runtime-effect-controller"),
        }
    }

    pub(crate) fn controller(&self) -> &dyn RuntimeEffectController {
        match self {
            Self::Borrowed(scoped) => scoped.controller(),
            #[cfg(any(test, feature = "testing"))]
            Self::Shared { controller, .. } => controller.as_ref(),
        }
    }

    pub(crate) fn scoped(&self) -> ScopedEffectController<'_> {
        match self {
            Self::Borrowed(scoped) => scoped.clone(),
            #[cfg(any(test, feature = "testing"))]
            Self::Shared { controller, scope } => {
                ScopedEffectController::shared(Arc::clone(controller), scope.clone())
                    .expect("runtime effect controller handle carries a valid scope")
            }
        }
    }

    pub(crate) fn clone_scoped(&self) -> RuntimeEffectControllerHandle<'run> {
        self.clone()
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

    pub(super) fn wrong_outcome(expected: RuntimeEffectKind, actual: RuntimeEffectKind) -> Self {
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
        RuntimeError::new(self.code, self.message)
    }
}

impl From<RuntimeError> for RuntimeEffectControllerError {
    fn from(err: RuntimeError) -> Self {
        Self::new(err.code.as_str(), err.message)
    }
}

impl From<PluginError> for RuntimeEffectControllerError {
    fn from(err: PluginError) -> Self {
        Self::new("plugin", err.to_string())
    }
}

impl From<crate::StoreError> for RuntimeEffectControllerError {
    fn from(err: crate::StoreError) -> Self {
        Self::new("runtime_store", err.to_string())
    }
}

// =============================================================================
// Local executor (per-effect borrowed runner state)
// =============================================================================

#[async_trait::async_trait]
pub(crate) trait ProcessRunner: Send + Sync {
    async fn run_process(
        &self,
        registration: crate::ProcessRegistration,
        execution_context: crate::ProcessExecutionContext,
        registry: Arc<dyn ProcessRegistry>,
        scoped_effect_controller: crate::ScopedEffectController<'_>,
        cancellation: CancellationToken,
    ) -> crate::ProcessAwaitOutput;
}

pub struct ProcessLocalExecution {
    pub registry: Arc<dyn ProcessRegistry>,
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

#[cfg(any(test, feature = "testing"))]
type TestingRuntimeEffectLocalRunnerFn<'run> = dyn FnOnce(
        RuntimeEffectEnvelope,
    ) -> Pin<
        Box<
            dyn Future<Output = Result<RuntimeEffectOutcome, RuntimeEffectControllerError>>
                + Send
                + 'run,
        >,
    > + Send
    + 'run;

#[cfg(any(test, feature = "testing"))]
struct TestingRuntimeEffectLocalRunner<'run> {
    run: Box<TestingRuntimeEffectLocalRunnerFn<'run>>,
}

enum RuntimeEffectLocalExecutorState<'run> {
    Unavailable,
    SleepOnly {
        cancellation: CancellationToken,
    },
    AwaitEvent {
        key: String,
        registry: Arc<dyn ProcessRegistry>,
        process_id: String,
        event_type: String,
        event_ordinal: u64,
        cancellation: CancellationToken,
    },
    Process(ProcessLocalExecution),
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

    pub fn await_process_event(
        key: impl Into<String>,
        registry: Arc<dyn ProcessRegistry>,
        process_id: impl Into<String>,
        event_type: impl Into<String>,
        event_ordinal: u64,
        cancellation: CancellationToken,
    ) -> Self {
        Self {
            state: RuntimeEffectLocalExecutorState::AwaitEvent {
                key: key.into(),
                registry,
                process_id: process_id.into(),
                event_type: event_type.into(),
                event_ordinal,
                cancellation,
            },
        }
    }

    pub fn processes(registry: Arc<dyn ProcessRegistry>) -> Self {
        Self {
            state: RuntimeEffectLocalExecutorState::Process(ProcessLocalExecution { registry }),
        }
    }

    #[cfg(any(test, feature = "testing"))]
    pub fn testing<F, Fut>(run: F) -> Self
    where
        F: FnOnce(RuntimeEffectEnvelope) -> Fut + Send + 'run,
        Fut: Future<Output = Result<RuntimeEffectOutcome, RuntimeEffectControllerError>>
            + Send
            + 'run,
    {
        Self {
            state: RuntimeEffectLocalExecutorState::Runner(Box::new(
                TestingRuntimeEffectLocalRunner {
                    run: Box::new(move |envelope| Box::pin(run(envelope))),
                },
            )),
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
            RuntimeEffectLocalExecutorState::AwaitEvent {
                key,
                registry,
                process_id,
                event_type,
                event_ordinal,
                cancellation,
            } => {
                execute_local_await_event(
                    envelope,
                    &key,
                    registry,
                    &process_id,
                    &event_type,
                    event_ordinal,
                    cancellation,
                )
                .await
            }
            RuntimeEffectLocalExecutorState::Unavailable => Err(RuntimeEffectControllerError::new(
                "runtime_effect_local_executor_unavailable",
                format!(
                    "no local executor is available for {}",
                    envelope.command.kind().as_str()
                ),
            )),
            RuntimeEffectLocalExecutorState::Process(_) => Err(RuntimeEffectControllerError::new(
                "runtime_effect_local_executor_mismatch",
                format!(
                    "process executor cannot execute {} command directly",
                    envelope.command.kind().as_str()
                ),
            )),
        }
    }

    pub fn into_process(self) -> Result<ProcessLocalExecution, RuntimeEffectControllerError> {
        match self.state {
            RuntimeEffectLocalExecutorState::Process(execution) => Ok(execution),
            _ => Err(RuntimeEffectControllerError::new(
                "runtime_effect_local_executor_unavailable",
                "no process executor is available for process command",
            )),
        }
    }
}

#[cfg(any(test, feature = "testing"))]
#[async_trait::async_trait]
impl RuntimeEffectLocalRunner for TestingRuntimeEffectLocalRunner<'_> {
    async fn execute(
        self: Box<Self>,
        envelope: RuntimeEffectEnvelope,
    ) -> Result<RuntimeEffectOutcome, RuntimeEffectControllerError> {
        (self.run)(envelope).await
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
                let protocol_iteration = runner.machine.protocol_iteration();
                let (result, text_streamed) = runner
                    .driver
                    .run_llm_call(
                        Arc::new((*request).into_request(None, None)),
                        protocol_iteration,
                        envelope.invocation,
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
                let mut outcome = runner
                    .driver
                    .run_tool_calls(
                        vec![(call, envelope.invocation)],
                        &runner.event_tx,
                        &runner.cancellation,
                    )
                    .await?;
                let result = outcome.completed.pop().ok_or_else(|| {
                    RuntimeEffectControllerError::new(
                        "tool_result_missing",
                        format!("tool `{tool_name}` completed without a result"),
                    )
                })?;
                Ok(RuntimeEffectOutcome::ToolCall {
                    result,
                    triggers: outcome.triggers,
                })
            }
            RuntimeEffectCommand::ExecCode { code } => {
                let protocol_iteration = runner.machine.protocol_iteration();
                let messages = runner.machine.message_sequence();
                Ok(RuntimeEffectOutcome::ExecCode {
                    result: runner
                        .driver
                        .run_exec_code(
                            &code,
                            messages,
                            protocol_iteration,
                            envelope.invocation,
                            &runner.event_tx,
                        )
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
            RuntimeEffectCommand::SyncExecutionEnvironment {
                update_machine_config,
            } => Ok(RuntimeEffectOutcome::SyncExecutionEnvironment {
                result: runner
                    .driver
                    .refresh_execution_environment(runner.machine, update_machine_config)
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
            RuntimeEffectCommand::Direct { request, .. } => Ok(RuntimeEffectOutcome::Direct {
                result: self
                    .run_direct_llm_request((*request).into_request(
                        crate::session_model::transport_stream_events(&self.provider, None),
                        None,
                    ))
                    .await,
            }),
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
        .await
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

async fn execute_local_await_event(
    envelope: RuntimeEffectEnvelope,
    expected_key: &str,
    registry: Arc<dyn ProcessRegistry>,
    process_id: &str,
    event_type: &str,
    event_ordinal: u64,
    cancellation: CancellationToken,
) -> Result<RuntimeEffectOutcome, RuntimeEffectControllerError> {
    match envelope.command {
        RuntimeEffectCommand::AwaitEvent { key } if key == expected_key => {
            if event_ordinal == 0 {
                return Err(RuntimeEffectControllerError::new(
                    "runtime_effect_await_event_invalid_ordinal",
                    "event ordinal must be one-based",
                ));
            }
            let mut after_sequence = 0;
            let mut matching_count = 0;
            let event = loop {
                let wait = registry.wait_event_after(process_id, event_type, after_sequence);
                tokio::pin!(wait);
                let event = tokio::select! {
                    _ = cancellation.cancelled() => {
                        return Err(RuntimeEffectControllerError::new(
                            "runtime_effect_await_event_cancelled",
                            "runtime effect event wait was cancelled",
                        ));
                    }
                    event = &mut wait => event?,
                };
                matching_count += 1;
                if matching_count == event_ordinal {
                    break event;
                }
                after_sequence = event.sequence;
            };
            Ok(RuntimeEffectOutcome::AwaitEvent {
                payload: event.payload,
            })
        }
        RuntimeEffectCommand::AwaitEvent { key } => Err(RuntimeEffectControllerError::new(
            "runtime_effect_await_event_key_mismatch",
            format!("local event wait expected `{expected_key}`, got `{key}`"),
        )),
        command => Err(RuntimeEffectControllerError::new(
            "runtime_effect_local_executor_mismatch",
            format!(
                "local event wait executor cannot execute {} command",
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

// =============================================================================
// Default in-process effect controller
// =============================================================================

/// Default in-process effect controller.
///
/// Stateless: the inline controller only registers process rows; the
/// lease-protected [`ProcessWorkRunner`](crate::ProcessWorkRunner) is the sole
/// executor.
#[derive(Clone, Default)]
pub struct InlineRuntimeEffectController;

#[async_trait::async_trait]
impl RuntimeEffectController for InlineRuntimeEffectController {
    async fn execute_effect(
        &self,
        envelope: RuntimeEffectEnvelope,
        local_executor: RuntimeEffectLocalExecutor<'_>,
    ) -> Result<RuntimeEffectOutcome, RuntimeEffectControllerError> {
        match envelope.command {
            RuntimeEffectCommand::Process { command } => {
                let execution = local_executor.into_process()?;
                let registry = execution.registry;
                let result = tokio::task::spawn(async move {
                    Self::execute_process_command(registry, *command).await
                })
                .await
                .map_err(|err| {
                    RuntimeEffectControllerError::new(
                        "runtime_effect_process_task_join",
                        format!("inline process effect task failed: {err}"),
                    )
                })??;
                Ok(RuntimeEffectOutcome::Process { result })
            }
            _ => local_executor.execute(envelope).await,
        }
    }
}

/// In-process deployment effect host.
#[derive(Clone)]
pub struct InlineEffectHost {
    controller: Arc<dyn RuntimeEffectController>,
}

impl InlineEffectHost {
    pub fn new(controller: Arc<dyn RuntimeEffectController>) -> Self {
        Self { controller }
    }
}

impl Default for InlineEffectHost {
    fn default() -> Self {
        Self::new(Arc::new(InlineRuntimeEffectController))
    }
}

impl EffectHost for InlineEffectHost {
    fn durability_tier(&self) -> crate::DurabilityTier {
        self.controller.durability_tier()
    }

    fn requires_durable_attachment_store(&self) -> bool {
        self.controller.requires_durable_attachment_store()
    }

    fn scoped<'run>(
        &'run self,
        scope: EffectScope,
    ) -> Result<ScopedEffectController<'run>, RuntimeError> {
        ScopedEffectController::shared(Arc::clone(&self.controller), scope)
    }

    fn scoped_static(
        &self,
        scope: EffectScope,
    ) -> Result<Option<ScopedEffectController<'static>>, RuntimeError> {
        Ok(Some(ScopedEffectController::shared(
            Arc::clone(&self.controller),
            scope,
        )?))
    }
}

impl InlineRuntimeEffectController {
    /// Register the process (and any handle grant) into the durable registry.
    ///
    /// The inline controller no longer runs the process here: the registry's
    /// non-terminal row *is* the durable work queue, and the lease-protected
    /// [`ProcessWorkRunner`](crate::ProcessWorkRunner) is the sole executor. The
    /// control seam pokes that runner after a successful start, so registering
    /// the row is all this path does.
    pub(crate) async fn start_process(
        registry: Arc<dyn crate::ProcessRegistry>,
        registration: crate::ProcessRegistration,
        grant: Option<crate::ProcessStartGrant>,
    ) -> Result<ProcessRecord, PluginError> {
        let registration_for_record = registration.clone();
        let record = registry.register_process(registration_for_record).await?;
        if let Some(grant) = grant {
            registry
                .grant_handle(&grant.session_scope, &registration.id, grant.descriptor)
                .await?;
        }
        Ok(record)
    }

    pub(crate) async fn request_process_cancel(
        &self,
        registry: Arc<dyn crate::ProcessRegistry>,
        process_id: &str,
        reason: Option<String>,
    ) -> Result<ProcessRecord, PluginError> {
        // Cancellation is a durable signal: the cancel event is what the
        // runner-run process observes, so the inline controller appends it and
        // no longer tracks an in-process cancellation token.
        registry
            .append_event(
                process_id,
                crate::ProcessEventAppendRequest::cancel_requested(process_id, reason.clone()),
            )
            .await?;
        registry
            .get_process(process_id)
            .await
            .ok_or_else(|| PluginError::Session(format!("unknown process `{process_id}`")))
    }

    async fn execute_process_command(
        registry: Arc<dyn crate::ProcessRegistry>,
        command: ProcessCommand,
    ) -> Result<ProcessEffectOutcome, RuntimeEffectControllerError> {
        match command {
            ProcessCommand::Start {
                registration,
                grant,
                execution_context: _,
            } => {
                let record = Self::start_process(registry, registration, grant).await?;
                Ok(ProcessEffectOutcome::Start { record })
            }
            ProcessCommand::List {
                session_scope,
                mode,
            } => {
                let entries = match mode {
                    crate::ProcessListMode::Live => {
                        registry.list_live_handle_grants(&session_scope).await?
                    }
                    crate::ProcessListMode::All => {
                        registry.list_handle_grants(&session_scope).await?
                    }
                };
                Ok(ProcessEffectOutcome::List { entries })
            }
            ProcessCommand::Transfer {
                from_scope,
                to_scope,
                process_ids,
            } => {
                registry
                    .transfer_handle_grants(&from_scope, &to_scope, &process_ids)
                    .await?;
                Ok(ProcessEffectOutcome::Transfer)
            }
            ProcessCommand::DeleteSession { session_id } => {
                let report = registry.delete_session_process_state(&session_id).await?;
                Ok(ProcessEffectOutcome::DeleteSession { report })
            }
            ProcessCommand::Await { process_id } => {
                let output = registry.await_process(&process_id).await?;
                Ok(ProcessEffectOutcome::Await { output })
            }
            ProcessCommand::Cancel { process_id, reason } => {
                let record = InlineRuntimeEffectController
                    .request_process_cancel(registry, &process_id, reason)
                    .await?;
                Ok(ProcessEffectOutcome::Cancel { record })
            }
            ProcessCommand::Signal {
                process_id,
                request,
                ..
            } => {
                let result = registry.append_event(&process_id, request).await?;
                Ok(ProcessEffectOutcome::Signal {
                    event: result.event,
                })
            }
        }
    }
}

impl std::fmt::Debug for InlineRuntimeEffectController {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InlineRuntimeEffectController").finish()
    }
}
