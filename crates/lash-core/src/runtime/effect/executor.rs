use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::LlmRequest as CoreLlmRequest;
use crate::ProcessRecord;
use crate::ProcessRegistry;
use crate::provider::ProviderHandle;
use crate::runtime::{RuntimeStreamEvent, RuntimeTurnDriver};
use crate::sansio::LlmCallError;
use crate::{PluginError, RuntimeError, RuntimeErrorCode};

use super::envelope::{
    ProcessCommand, ProcessEffectOutcome, RuntimeDirectLlmOutcome, RuntimeEffectCommand,
    RuntimeEffectEnvelope, RuntimeEffectKind, RuntimeEffectOutcome,
};
use super::outcome::llm_call_error_from_transport;

/// Host controls attached to one external event wait.
///
/// Durable effect controllers consume these controls and translate them to
/// their engine-native cancellation and timer primitives.
pub struct RuntimeAwaitEventOptions {
    pub cancellation: CancellationToken,
    pub deadline: Option<Instant>,
    pub clock: Arc<dyn crate::Clock>,
}

use super::await_events::inline_await_events;

// =============================================================================
// Effect host + controller trait + scope + error
// =============================================================================

/// Stable semantic identity for one effectful runtime operation.
///
/// The scope is chosen by the host boundary before any nondeterministic work is
/// planned. It is intentionally generic: Restate, an inline test host, or a
/// future durable effect host all receive the same Lash scope vocabulary.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ExecutionScope {
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

impl ExecutionScope {
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

    pub(crate) fn validate(&self) -> Result<(), RuntimeError> {
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
                RuntimeErrorCode::MissingExecutionScopeId,
                "execution scopes require non-empty stable ids",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AwaitEventWaitIdentity {
    ToolCompletion {
        tool_call_id: String,
    },
    ProcessSignal {
        process_id: String,
        signal_name: String,
        ordinal: u64,
    },
    /// Reserved first-writer-wins cancellation-versus-completion gate for a
    /// foreground turn.
    TurnCancelGate,
    /// Reserved terminal publication promise for a foreground turn.
    TurnTerminal,
    Custom {
        key: String,
    },
}

impl AwaitEventWaitIdentity {
    pub fn tool_completion(tool_call_id: impl Into<String>) -> Self {
        Self::ToolCompletion {
            tool_call_id: tool_call_id.into(),
        }
    }

    pub fn process_signal(
        process_id: impl Into<String>,
        signal_name: impl Into<String>,
        ordinal: u64,
    ) -> Self {
        Self::ProcessSignal {
            process_id: process_id.into(),
            signal_name: signal_name.into(),
            ordinal,
        }
    }

    pub(super) fn validate(&self) -> Result<(), RuntimeError> {
        let invalid = match self {
            Self::ToolCompletion { tool_call_id } => tool_call_id.trim().is_empty(),
            Self::ProcessSignal {
                process_id,
                signal_name,
                ordinal,
            } => process_id.trim().is_empty() || signal_name.trim().is_empty() || *ordinal == 0,
            Self::TurnCancelGate | Self::TurnTerminal => false,
            Self::Custom { key } => key.trim().is_empty(),
        };
        if invalid {
            return Err(RuntimeError::new(
                "invalid_await_event_wait_identity",
                "await-event wait identity requires non-empty stable ids",
            ));
        }
        Ok(())
    }

    pub fn is_turn_control(&self) -> bool {
        matches!(self, Self::TurnCancelGate | Self::TurnTerminal)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AwaitEventKey {
    pub scope: ExecutionScope,
    pub wait: AwaitEventWaitIdentity,
    pub key_id: String,
    pub signature: String,
}

impl AwaitEventKey {
    pub fn promise_key(&self) -> String {
        format!("lash-await-event:{}", self.key_id)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExternalCompletionError {
    pub code: String,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw: Option<serde_json::Value>,
}

impl ExternalCompletionError {
    pub fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
            raw: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", content = "payload", rename_all = "snake_case")]
pub enum Resolution {
    Ok(serde_json::Value),
    Err(ExternalCompletionError),
    Timeout,
    Cancelled,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum ResolveOutcome {
    Accepted,
    AlreadyResolved { terminal: Resolution },
    UnknownOrRevoked,
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

/// Scoped low-level controller plus the semantic execution scope it is serving.
#[derive(Clone)]
pub struct ScopedEffectController<'run> {
    controller: ScopedEffectControllerInner<'run>,
    scope: ExecutionScope,
}

impl<'run> ScopedEffectController<'run> {
    pub fn borrowed(
        controller: &'run dyn RuntimeEffectController,
        scope: ExecutionScope,
    ) -> Result<Self, RuntimeError> {
        scope.validate()?;
        Ok(Self {
            controller: ScopedEffectControllerInner::Borrowed(controller),
            scope,
        })
    }

    pub fn shared(
        controller: Arc<dyn RuntimeEffectController>,
        scope: ExecutionScope,
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

    pub(crate) fn shared_controller(&self) -> Option<Arc<dyn RuntimeEffectController>> {
        match &self.controller {
            ScopedEffectControllerInner::Borrowed(_) => None,
            ScopedEffectControllerInner::Shared(controller) => Some(Arc::clone(controller)),
        }
    }

    pub fn execution_scope(&self) -> &ExecutionScope {
        &self.scope
    }

    pub fn scope_id(&self) -> &str {
        self.scope.id()
    }

    pub fn turn_id(&self) -> Option<&str> {
        self.scope.turn_id()
    }
}

/// Shared durability and Durable Wait contract for effect boundaries.
///
/// Both the deployment-level [`EffectHost`] factory and the per-run
/// [`RuntimeEffectController`] resolve Durable Waits and describe their
/// durability; this supertrait is the single declaration of that contract.
#[async_trait::async_trait]
pub trait AwaitEventResolver: Send + Sync {
    fn durability_tier(&self) -> crate::DurabilityTier {
        crate::DurabilityTier::Inline
    }

    fn supports_durable_effects(&self) -> bool {
        false
    }

    async fn await_event_key(
        &self,
        scope: &ExecutionScope,
        wait: AwaitEventWaitIdentity,
    ) -> Result<AwaitEventKey, RuntimeError> {
        if wait.is_turn_control() {
            return super::await_events::inline_await_events().key_for(scope, wait);
        }
        Err(RuntimeError::new(
            "await_event_unsupported",
            "this effect boundary does not support await-event keys",
        ))
    }

    async fn resolve_await_event(
        &self,
        key: &AwaitEventKey,
        resolution: Resolution,
    ) -> Result<ResolveOutcome, RuntimeError> {
        if key.wait.is_turn_control() {
            return super::await_events::inline_await_events().resolve(key, resolution);
        }
        Ok(ResolveOutcome::UnknownOrRevoked)
    }

    async fn await_await_event(
        &self,
        key: &AwaitEventKey,
        cancel: CancellationToken,
        deadline: Option<Instant>,
    ) -> Result<Resolution, RuntimeError> {
        if key.wait.is_turn_control() {
            return super::await_events::inline_await_events()
                .await_resolution(key, cancel, deadline, &crate::SystemClock)
                .await;
        }
        Err(RuntimeError::new(
            "await_event_unsupported",
            "this effect boundary does not support await-event waits",
        ))
    }

    async fn revoke_await_events_for_session(&self, session_id: &str) -> Result<(), RuntimeError> {
        super::await_events::inline_await_events().revoke_session(session_id)
    }

    /// Cancel every *outstanding* durable wait for `session_id` without
    /// deleting the session: each waiter receives a terminal
    /// [`Resolution::Cancelled`] instead of hanging, late resolves observe
    /// that terminal, and waits registered afterwards behave normally — in
    /// contrast to [`revoke_await_events_for_session`](Self::revoke_await_events_for_session),
    /// which tombstones the session's waits forever.
    ///
    /// The default errors loudly: an effect boundary that tracks durable waits
    /// must implement this to honor the host lever, and one that cannot must
    /// not silently claim success.
    async fn cancel_await_events_for_session(&self, _session_id: &str) -> Result<(), RuntimeError> {
        Err(RuntimeError::new(
            "await_event_cancel_unsupported",
            "this effect boundary does not support cancelling durable waits",
        ))
    }
}

/// Deployment-level factory for scoped effect controllers.
#[async_trait::async_trait]
pub trait EffectHost: AwaitEventResolver {
    fn scoped<'run>(
        &'run self,
        scope: ExecutionScope,
    ) -> Result<ScopedEffectController<'run>, RuntimeError>;

    fn scoped_static(
        &self,
        _scope: ExecutionScope,
    ) -> Result<Option<ScopedEffectController<'static>>, RuntimeError> {
        Ok(None)
    }
}

/// Boundary for nondeterministic runtime work.
#[async_trait::async_trait]
pub trait RuntimeEffectController: AwaitEventResolver {
    /// Advises an engine to end the current in-process execution segment at a
    /// quiescent point. Engines may decline when live state is not capturable,
    /// but must make progress before returning another decline. In particular,
    /// an engine must not repeatedly return the same boundary and unchanged
    /// durable-wait state in one invocation; a host may bound and retry such a
    /// non-progressing invocation.
    fn wants_segment_boundary(&self, _progress: &SegmentProgress) -> Option<BoundaryReason> {
        None
    }

    /// Whether this controller can safely accept overlapping `execute_effect`
    /// calls from one runtime coordinator.
    ///
    /// Local and store-backed controllers can usually fan out independent
    /// effects. Some workflow substrates expose a single ordered journal
    /// context where native operations must be awaited immediately before the
    /// next context call is issued. Those controllers should return `false` so
    /// coordinators serialize child effects while still replaying each child by
    /// its own stable key.
    fn supports_concurrent_effects(&self) -> bool {
        true
    }

    async fn execute_effect(
        &self,
        envelope: RuntimeEffectEnvelope,
        local_executor: RuntimeEffectLocalExecutor<'_>,
    ) -> Result<RuntimeEffectOutcome, RuntimeEffectControllerError>;
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SegmentProgress {
    pub effects_executed: u64,
    pub journaled_bytes_estimate: Option<u64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BoundaryReason {
    JournalBudget,
    DurationCap,
}

/// Runtime-internal handle for effect-controller references carried through
/// per-turn execution contexts.
#[derive(Clone)]
pub(crate) enum RuntimeEffectControllerHandle<'run> {
    Borrowed(ScopedEffectController<'run>),
    #[cfg(any(test, feature = "testing"))]
    Shared {
        controller: Arc<dyn RuntimeEffectController>,
        scope: ExecutionScope,
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
            scope: ExecutionScope::runtime_operation("test-runtime-effect-controller"),
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
        handover: Option<crate::SegmentHandover>,
    ) -> crate::ProcessRunOutcome;
}

pub struct ProcessLocalExecution {
    pub registry: Arc<dyn ProcessRegistry>,
    pub process_work_driver: Option<crate::ProcessWorkDriver>,
}

impl ProcessLocalExecution {
    pub async fn execute(
        self,
        command: ProcessCommand,
    ) -> Result<ProcessEffectOutcome, RuntimeEffectControllerError> {
        let Self {
            registry,
            process_work_driver,
        } = self;
        match command {
            ProcessCommand::Start {
                registration,
                grant,
                execution_context: _,
            } => {
                let record =
                    InlineRuntimeEffectController::start_process(registry, registration, grant)
                        .await?;
                if let Some(driver) = process_work_driver.as_ref() {
                    driver.claim_and_run_pending("process_start").await?;
                }
                Ok(ProcessEffectOutcome::Start {
                    record: Box::new(record),
                })
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
                let output = if let Some(driver) = process_work_driver.as_ref() {
                    driver.await_terminal(&process_id).await?
                } else {
                    crate::ProcessAwaiter::polling(registry)
                        .await_terminal(&process_id)
                        .await?
                };
                Ok(ProcessEffectOutcome::Await { output })
            }
            ProcessCommand::Cancel { process_id, reason } => {
                let record = InlineRuntimeEffectController
                    .request_process_cancel(registry, &process_id, reason)
                    .await?;
                Ok(ProcessEffectOutcome::Cancel {
                    record: Box::new(record),
                })
            }
            ProcessCommand::Signal {
                process_id,
                request,
                ..
            } => {
                let result = registry.append_event(&process_id, request).await?;
                Ok(ProcessEffectOutcome::Signal {
                    event: Box::new(result.event),
                })
            }
        }
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
    attachment_store: Arc<crate::SessionAttachmentStore>,
}

struct LocalToolBatchEffectRunner<'run> {
    context: crate::RuntimeExecutionContext<'run>,
    child_trace_hooks: HashMap<String, crate::ToolChildExecutionTraceHook>,
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

type DurableStepLocalRunnerFn<'run> = dyn FnOnce(
        serde_json::Value,
    ) -> Pin<
        Box<
            dyn Future<Output = Result<serde_json::Value, RuntimeEffectControllerError>>
                + Send
                + 'run,
        >,
    > + Send
    + 'run;

struct DurableStepLocalRunner<'run> {
    run: Box<DurableStepLocalRunnerFn<'run>>,
}

enum RuntimeEffectLocalExecutorState<'run> {
    Unavailable,
    SleepOnly {
        cancellation: CancellationToken,
        clock: Arc<dyn crate::Clock>,
    },
    ExternalWaitOptions {
        cancellation: CancellationToken,
        deadline: Option<Instant>,
        clock: Arc<dyn crate::Clock>,
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
        Self::sleep_with_clock(cancellation, Arc::new(crate::SystemClock))
    }

    pub fn sleep_with_clock(cancellation: CancellationToken, clock: Arc<dyn crate::Clock>) -> Self {
        Self {
            state: RuntimeEffectLocalExecutorState::SleepOnly {
                cancellation,
                clock,
            },
        }
    }

    pub fn await_event(cancellation: CancellationToken, deadline: Option<Instant>) -> Self {
        Self::await_event_with_clock(cancellation, deadline, Arc::new(crate::SystemClock))
    }

    pub fn await_event_with_clock(
        cancellation: CancellationToken,
        deadline: Option<Instant>,
        clock: Arc<dyn crate::Clock>,
    ) -> Self {
        Self {
            state: RuntimeEffectLocalExecutorState::ExternalWaitOptions {
                cancellation,
                deadline,
                clock,
            },
        }
    }

    pub fn processes(
        registry: Arc<dyn ProcessRegistry>,
        process_work_driver: Option<crate::ProcessWorkDriver>,
    ) -> Self {
        Self {
            state: RuntimeEffectLocalExecutorState::Process(ProcessLocalExecution {
                registry,
                process_work_driver,
            }),
        }
    }

    pub fn durable_step<F, Fut>(run: F) -> Self
    where
        F: FnOnce(serde_json::Value) -> Fut + Send + 'run,
        Fut: Future<Output = Result<serde_json::Value, RuntimeError>> + Send + 'run,
    {
        Self {
            state: RuntimeEffectLocalExecutorState::Runner(Box::new(DurableStepLocalRunner {
                run: Box::new(move |input| {
                    Box::pin(async move { run(input).await.map_err(Into::into) })
                }),
            })),
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
        attachment_store: Arc<crate::SessionAttachmentStore>,
    ) -> Self {
        Self {
            state: RuntimeEffectLocalExecutorState::Runner(Box::new(LocalDirectEffectRunner {
                provider,
                attachment_store,
            })),
        }
    }

    pub(crate) fn tool_batch(
        context: crate::RuntimeExecutionContext<'run>,
        child_trace_hooks: HashMap<String, crate::ToolChildExecutionTraceHook>,
    ) -> Self {
        Self {
            state: RuntimeEffectLocalExecutorState::Runner(Box::new(LocalToolBatchEffectRunner {
                context,
                child_trace_hooks,
            })),
        }
    }

    pub async fn execute(
        self,
        envelope: RuntimeEffectEnvelope,
    ) -> Result<RuntimeEffectOutcome, RuntimeEffectControllerError> {
        match self.state {
            RuntimeEffectLocalExecutorState::Runner(runner) => runner.execute(envelope).await,
            RuntimeEffectLocalExecutorState::SleepOnly {
                cancellation,
                clock,
            } => execute_local_sleep(envelope, cancellation, clock.as_ref()).await,
            RuntimeEffectLocalExecutorState::ExternalWaitOptions { .. } => {
                Err(RuntimeEffectControllerError::new(
                    "runtime_effect_local_executor_mismatch",
                    format!(
                        "local await-event options cannot execute {} command directly",
                        envelope.command.kind().as_str()
                    ),
                ))
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

    pub fn into_await_event_options(
        self,
    ) -> Result<RuntimeAwaitEventOptions, RuntimeEffectControllerError> {
        match self.state {
            RuntimeEffectLocalExecutorState::ExternalWaitOptions {
                cancellation,
                deadline,
                clock,
            } => Ok(RuntimeAwaitEventOptions {
                cancellation,
                deadline,
                clock,
            }),
            _ => Ok(RuntimeAwaitEventOptions {
                cancellation: CancellationToken::new(),
                deadline: None,
                clock: Arc::new(crate::SystemClock),
            }),
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
impl RuntimeEffectLocalRunner for DurableStepLocalRunner<'_> {
    async fn execute(
        self: Box<Self>,
        envelope: RuntimeEffectEnvelope,
    ) -> Result<RuntimeEffectOutcome, RuntimeEffectControllerError> {
        match envelope.command {
            RuntimeEffectCommand::DurableStep { input, .. } => {
                let value = (self.run)(input).await?;
                Ok(RuntimeEffectOutcome::DurableStep { value })
            }
            command => Err(RuntimeEffectControllerError::new(
                "runtime_effect_local_executor_mismatch",
                format!(
                    "local durable step executor cannot execute {} command",
                    command.kind().as_str()
                ),
            )),
        }
    }
}

#[async_trait::async_trait]
impl RuntimeEffectLocalRunner for LocalToolBatchEffectRunner<'_> {
    async fn execute(
        self: Box<Self>,
        envelope: RuntimeEffectEnvelope,
    ) -> Result<RuntimeEffectOutcome, RuntimeEffectControllerError> {
        match envelope.command {
            RuntimeEffectCommand::ToolBatch { batch } => {
                let outcome = self
                    .context
                    .execute_prepared_tool_batch_launches(
                        batch,
                        envelope.invocation,
                        self.child_trace_hooks,
                    )
                    .await?;
                Ok(RuntimeEffectOutcome::ToolBatch {
                    launches: outcome.launches,
                    triggers: outcome.triggers,
                })
            }
            RuntimeEffectCommand::ToolAttempt {
                call,
                execution_grant,
                attempt,
                max_attempts,
            } => {
                let child_execution_trace_hook = self.child_trace_hooks.get(&call.call_id).cloned();
                let outcome = self
                    .context
                    .execute_prepared_tool_attempt_effect(
                        call,
                        execution_grant,
                        attempt,
                        max_attempts,
                        envelope.invocation,
                        child_execution_trace_hook,
                    )
                    .await?;
                Ok(RuntimeEffectOutcome::ToolAttempt {
                    launch: outcome.launch,
                    triggers: outcome.triggers,
                })
            }
            command => Err(RuntimeEffectControllerError::new(
                "runtime_effect_local_executor_mismatch",
                format!(
                    "local tool executor cannot execute {} command",
                    command.kind().as_str()
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
                let protocol_iteration = runner.machine.protocol_iteration();
                let (result, text_streamed, call_record) = runner
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
                    call_record,
                })
            }
            RuntimeEffectCommand::ToolBatch { batch } => {
                let outcome = runner
                    .driver
                    .run_tool_batch(
                        batch,
                        envelope.invocation,
                        &runner.event_tx,
                        &runner.cancellation,
                    )
                    .await?;
                Ok(RuntimeEffectOutcome::ToolBatch {
                    launches: outcome.launches,
                    triggers: outcome.triggers,
                })
            }
            RuntimeEffectCommand::ExecCode { language, code } => {
                let protocol_iteration = runner.machine.protocol_iteration();
                let messages = runner.machine.message_sequence();
                Ok(RuntimeEffectOutcome::ExecCode {
                    result: runner
                        .driver
                        .run_exec_code(
                            language,
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
                sleep_with_cancellation(
                    duration_ms,
                    &runner.cancellation,
                    runner.driver.host.core.clock.as_ref(),
                )
                .await?;
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
            RuntimeEffectCommand::Direct { request, .. } => {
                let (result, call_record) = self
                    .run_direct_llm_request((*request).into_request(
                        crate::session_model::transport_stream_events(&self.provider, None),
                        None,
                    ))
                    .await;
                Ok(RuntimeEffectOutcome::Direct {
                    result,
                    call_record,
                })
            }
            RuntimeEffectCommand::Sleep { duration_ms } => {
                sleep_with_cancellation(
                    duration_ms,
                    &CancellationToken::new(),
                    &crate::SystemClock,
                )
                .await?;
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
    async fn run_direct_llm_request(&mut self, request: CoreLlmRequest) -> RuntimeDirectLlmOutcome {
        let request = match crate::attachments::resolve_llm_request_attachments(
            request,
            self.attachment_store.as_ref(),
        )
        .await
        {
            Ok(request) => request,
            Err(err) => {
                return (
                    Err(LlmCallError {
                        message: err.to_string(),
                        retryable: false,
                        kind: crate::ProviderFailureKind::Unknown,
                        raw: None,
                        code: Some("attachment_resolution_failed".to_string()),
                        terminal_reason: crate::LlmTerminalReason::ProviderError,
                        request_body: None,
                        partial_response: None,
                    }),
                    None,
                );
            }
        };
        match self.provider.complete(request).await {
            Ok(completion) => (Ok(completion.response), Some(completion.call_record)),
            Err(failure) => (
                Err(llm_call_error_from_transport(failure.error)),
                Some(failure.call_record),
            ),
        }
    }
}

async fn execute_local_sleep(
    envelope: RuntimeEffectEnvelope,
    cancellation: CancellationToken,
    clock: &dyn crate::Clock,
) -> Result<RuntimeEffectOutcome, RuntimeEffectControllerError> {
    match envelope.command {
        RuntimeEffectCommand::Sleep { duration_ms } => {
            sleep_with_cancellation(duration_ms, &cancellation, clock).await?;
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
    clock: &dyn crate::Clock,
) -> Result<(), RuntimeEffectControllerError> {
    let sleep = clock.sleep(std::time::Duration::from_millis(duration_ms));
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
/// The inline controller executes local runners in process, provides in-memory
/// await-event resolution, and exposes durable-tool-effect semantics for local
/// runs. It does not make in-flight effects crash durable; workflow adapters
/// provide that by recording outcomes in their own history.
#[derive(Clone, Default)]
pub struct InlineRuntimeEffectController;

#[async_trait::async_trait]
impl AwaitEventResolver for InlineRuntimeEffectController {
    fn supports_durable_effects(&self) -> bool {
        true
    }

    async fn await_event_key(
        &self,
        scope: &ExecutionScope,
        wait: AwaitEventWaitIdentity,
    ) -> Result<AwaitEventKey, RuntimeError> {
        inline_await_events().key_for(scope, wait)
    }

    async fn resolve_await_event(
        &self,
        key: &AwaitEventKey,
        resolution: Resolution,
    ) -> Result<ResolveOutcome, RuntimeError> {
        inline_await_events().resolve(key, resolution)
    }

    async fn await_await_event(
        &self,
        key: &AwaitEventKey,
        cancel: CancellationToken,
        deadline: Option<Instant>,
    ) -> Result<Resolution, RuntimeError> {
        inline_await_events()
            .await_resolution(key, cancel, deadline, &crate::SystemClock)
            .await
    }

    async fn revoke_await_events_for_session(&self, session_id: &str) -> Result<(), RuntimeError> {
        inline_await_events().revoke_session(session_id)
    }

    async fn cancel_await_events_for_session(&self, session_id: &str) -> Result<(), RuntimeError> {
        inline_await_events().cancel_session(session_id)
    }
}

#[async_trait::async_trait]
impl RuntimeEffectController for InlineRuntimeEffectController {
    async fn execute_effect(
        &self,
        envelope: RuntimeEffectEnvelope,
        local_executor: RuntimeEffectLocalExecutor<'_>,
    ) -> Result<RuntimeEffectOutcome, RuntimeEffectControllerError> {
        match envelope.command {
            RuntimeEffectCommand::AwaitEvent { key } => {
                let RuntimeAwaitEventOptions {
                    cancellation,
                    deadline,
                    clock,
                } = local_executor.into_await_event_options()?;
                let resolution = inline_await_events()
                    .await_resolution(&key, cancellation, deadline, clock.as_ref())
                    .await
                    .map_err(RuntimeEffectControllerError::from)?;
                Ok(RuntimeEffectOutcome::AwaitEvent { resolution })
            }
            RuntimeEffectCommand::Process { command } => {
                let execution = local_executor.into_process()?;
                let result = tokio::task::spawn(async move { execution.execute(*command).await })
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

#[async_trait::async_trait]
impl AwaitEventResolver for InlineEffectHost {
    fn durability_tier(&self) -> crate::DurabilityTier {
        self.controller.durability_tier()
    }

    fn supports_durable_effects(&self) -> bool {
        self.controller.supports_durable_effects()
    }

    async fn await_event_key(
        &self,
        scope: &ExecutionScope,
        wait: AwaitEventWaitIdentity,
    ) -> Result<AwaitEventKey, RuntimeError> {
        self.controller.await_event_key(scope, wait).await
    }

    async fn resolve_await_event(
        &self,
        key: &AwaitEventKey,
        resolution: Resolution,
    ) -> Result<ResolveOutcome, RuntimeError> {
        self.controller.resolve_await_event(key, resolution).await
    }

    async fn await_await_event(
        &self,
        key: &AwaitEventKey,
        cancel: CancellationToken,
        deadline: Option<Instant>,
    ) -> Result<Resolution, RuntimeError> {
        self.controller
            .await_await_event(key, cancel, deadline)
            .await
    }

    async fn revoke_await_events_for_session(&self, session_id: &str) -> Result<(), RuntimeError> {
        self.controller
            .revoke_await_events_for_session(session_id)
            .await
    }

    async fn cancel_await_events_for_session(&self, session_id: &str) -> Result<(), RuntimeError> {
        self.controller
            .cancel_await_events_for_session(session_id)
            .await
    }
}

#[async_trait::async_trait]
impl EffectHost for InlineEffectHost {
    fn scoped<'run>(
        &'run self,
        scope: ExecutionScope,
    ) -> Result<ScopedEffectController<'run>, RuntimeError> {
        ScopedEffectController::shared(Arc::clone(&self.controller), scope)
    }

    fn scoped_static(
        &self,
        scope: ExecutionScope,
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
    /// non-terminal row *is* the durable work queue, and the host-owned
    /// [`ProcessWorkDriver`](crate::ProcessWorkDriver) is the sole executor.
    /// Registering the row is all this path does; the control seam drives the
    /// host driver after a successful start.
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
}

impl std::fmt::Debug for InlineRuntimeEffectController {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InlineRuntimeEffectController").finish()
    }
}
