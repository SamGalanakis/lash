use std::collections::{HashMap, HashSet};
#[cfg(any(test, feature = "testing"))]
use std::pin::Pin;
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use hmac::{Hmac, Mac};
use serde::{Deserialize, Serialize};
use tokio::sync::{Notify, mpsc};
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

type HmacSha256 = Hmac<sha2::Sha256>;

fn inline_await_events() -> &'static AwaitEventRegistry {
    static REGISTRY: OnceLock<AwaitEventRegistry> = OnceLock::new();
    REGISTRY.get_or_init(AwaitEventRegistry::new)
}

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

    fn validate(&self) -> Result<(), RuntimeError> {
        let invalid = match self {
            Self::ToolCompletion { tool_call_id } => tool_call_id.trim().is_empty(),
            Self::ProcessSignal {
                process_id,
                signal_name,
                ordinal,
            } => process_id.trim().is_empty() || signal_name.trim().is_empty() || *ordinal == 0,
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

#[derive(Debug)]
struct AwaitEventEntry {
    terminal: Option<Resolution>,
    notify: Arc<Notify>,
}

#[derive(Debug)]
struct AwaitEventRegistryState {
    entries: HashMap<String, AwaitEventEntry>,
    revoked_key_ids: HashSet<String>,
    revoked_session_ids: HashSet<String>,
}

#[derive(Debug)]
struct AwaitEventRegistry {
    secret: Vec<u8>,
    state: std::sync::Mutex<AwaitEventRegistryState>,
}

impl AwaitEventRegistry {
    fn new() -> Self {
        Self {
            secret: uuid::Uuid::new_v4().as_bytes().to_vec(),
            state: std::sync::Mutex::new(AwaitEventRegistryState {
                entries: HashMap::new(),
                revoked_key_ids: HashSet::new(),
                revoked_session_ids: HashSet::new(),
            }),
        }
    }

    fn key_for(
        &self,
        scope: &ExecutionScope,
        wait: AwaitEventWaitIdentity,
    ) -> Result<AwaitEventKey, RuntimeError> {
        scope.validate()?;
        wait.validate()?;
        let key_id =
            crate::stable_hash::stable_json_sha256_hex(&(scope, &wait)).map_err(|err| {
                RuntimeError::new(
                    "await_event_key_hash",
                    format!("failed to hash await-event identity: {err}"),
                )
            })?;
        let signature = self.signature(scope, &wait, &key_id)?;
        Ok(AwaitEventKey {
            scope: scope.clone(),
            wait,
            key_id,
            signature,
        })
    }

    fn signature(
        &self,
        scope: &ExecutionScope,
        wait: &AwaitEventWaitIdentity,
        key_id: &str,
    ) -> Result<String, RuntimeError> {
        let mut mac = HmacSha256::new_from_slice(&self.secret).map_err(|err| {
            RuntimeError::new(
                "await_event_key_sign",
                format!("failed to initialize await-event key signer: {err}"),
            )
        })?;
        let canonical = serde_json::to_vec(&(scope, wait, key_id)).map_err(|err| {
            RuntimeError::new(
                "await_event_key_sign",
                format!("failed to serialize await-event key identity: {err}"),
            )
        })?;
        mac.update(&canonical);
        Ok(format!("{:x}", mac.finalize().into_bytes()))
    }

    fn verify(&self, key: &AwaitEventKey) -> Result<bool, RuntimeError> {
        let expected = self.signature(&key.scope, &key.wait, &key.key_id)?;
        Ok(expected == key.signature)
    }

    fn resolve(
        &self,
        key: &AwaitEventKey,
        resolution: Resolution,
    ) -> Result<ResolveOutcome, RuntimeError> {
        if !self.verify(key)? {
            return Ok(ResolveOutcome::UnknownOrRevoked);
        }
        let mut state = self.state.lock().map_err(|_| {
            RuntimeError::new(
                "await_event_registry_poisoned",
                "await-event registry lock poisoned",
            )
        })?;
        if state.revoked_key_ids.contains(&key.key_id)
            || key
                .scope
                .session_id()
                .is_some_and(|session_id| state.revoked_session_ids.contains(session_id))
        {
            return Ok(ResolveOutcome::UnknownOrRevoked);
        }
        let entry = state
            .entries
            .entry(key.key_id.clone())
            .or_insert_with(|| AwaitEventEntry {
                terminal: None,
                notify: Arc::new(Notify::new()),
            });
        if let Some(terminal) = &entry.terminal {
            return Ok(ResolveOutcome::AlreadyResolved {
                terminal: terminal.clone(),
            });
        }
        entry.terminal = Some(resolution);
        entry.notify.notify_waiters();
        Ok(ResolveOutcome::Accepted)
    }

    async fn await_resolution(
        &self,
        key: &AwaitEventKey,
        cancel: CancellationToken,
        deadline: Option<Instant>,
    ) -> Result<Resolution, RuntimeError> {
        if !self.verify(key)? {
            return Err(RuntimeError::new(
                "await_event_unknown_or_revoked",
                "await-event key is invalid or revoked",
            ));
        }
        loop {
            let notify =
                {
                    let mut state = self.state.lock().map_err(|_| {
                        RuntimeError::new(
                            "await_event_registry_poisoned",
                            "await-event registry lock poisoned",
                        )
                    })?;
                    if state.revoked_key_ids.contains(&key.key_id)
                        || key.scope.session_id().is_some_and(|session_id| {
                            state.revoked_session_ids.contains(session_id)
                        })
                    {
                        return Err(RuntimeError::new(
                            "await_event_unknown_or_revoked",
                            "await-event key is invalid or revoked",
                        ));
                    }
                    let entry = state.entries.entry(key.key_id.clone()).or_insert_with(|| {
                        AwaitEventEntry {
                            terminal: None,
                            notify: Arc::new(Notify::new()),
                        }
                    });
                    if let Some(terminal) = entry.terminal.clone() {
                        return Ok(terminal);
                    }
                    Arc::clone(&entry.notify)
                };
            if let Some(deadline) = deadline {
                tokio::select! {
                    _ = cancel.cancelled() => {
                        let _ = self.resolve(key, Resolution::Cancelled)?;
                    }
                    _ = tokio::time::sleep_until(tokio::time::Instant::from_std(deadline)) => {
                        let _ = self.resolve(key, Resolution::Timeout)?;
                    }
                    _ = notify.notified() => {}
                }
            } else {
                tokio::select! {
                    _ = cancel.cancelled() => {
                        let _ = self.resolve(key, Resolution::Cancelled)?;
                    }
                    _ = notify.notified() => {}
                }
            }
        }
    }

    fn revoke_session(&self, session_id: &str) -> Result<(), RuntimeError> {
        let mut state = self.state.lock().map_err(|_| {
            RuntimeError::new(
                "await_event_registry_poisoned",
                "await-event registry lock poisoned",
            )
        })?;
        state.revoked_session_ids.insert(session_id.to_string());
        for entry in state.entries.values() {
            entry.notify.notify_waiters();
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
        scope: ExecutionScope,
    ) -> Result<ScopedEffectController<'run>, RuntimeError>;

    fn scoped_static(
        &self,
        _scope: ExecutionScope,
    ) -> Result<Option<ScopedEffectController<'static>>, RuntimeError> {
        Ok(None)
    }

    async fn await_event_key(
        &self,
        _scope: &ExecutionScope,
        _wait: AwaitEventWaitIdentity,
    ) -> Result<AwaitEventKey, RuntimeError> {
        Err(RuntimeError::new(
            "await_event_unsupported",
            "this effect host does not support await-event keys",
        ))
    }

    async fn resolve_await_event(
        &self,
        _key: &AwaitEventKey,
        _resolution: Resolution,
    ) -> Result<ResolveOutcome, RuntimeError> {
        Ok(ResolveOutcome::UnknownOrRevoked)
    }

    async fn await_await_event(
        &self,
        _key: &AwaitEventKey,
        _cancel: CancellationToken,
        _deadline: Option<Instant>,
    ) -> Result<Resolution, RuntimeError> {
        Err(RuntimeError::new(
            "await_event_unsupported",
            "this effect host does not support await-event waits",
        ))
    }

    async fn revoke_await_events_for_session(&self, _session_id: &str) -> Result<(), RuntimeError> {
        Ok(())
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

    async fn await_event_key(
        &self,
        _scope: &ExecutionScope,
        _wait: AwaitEventWaitIdentity,
    ) -> Result<AwaitEventKey, RuntimeError> {
        Err(RuntimeError::new(
            "await_event_unsupported",
            "this effect controller does not support await-event keys",
        ))
    }

    async fn resolve_await_event(
        &self,
        _key: &AwaitEventKey,
        _resolution: Resolution,
    ) -> Result<ResolveOutcome, RuntimeError> {
        Ok(ResolveOutcome::UnknownOrRevoked)
    }

    async fn await_await_event(
        &self,
        _key: &AwaitEventKey,
        _cancel: CancellationToken,
        _deadline: Option<Instant>,
    ) -> Result<Resolution, RuntimeError> {
        Err(RuntimeError::new(
            "await_event_unsupported",
            "this effect controller does not support await-event waits",
        ))
    }

    async fn revoke_await_events_for_session(&self, _session_id: &str) -> Result<(), RuntimeError> {
        Ok(())
    }
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
    ExternalWaitOptions {
        cancellation: CancellationToken,
        deadline: Option<Instant>,
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

    pub fn await_event(cancellation: CancellationToken, deadline: Option<Instant>) -> Self {
        Self {
            state: RuntimeEffectLocalExecutorState::ExternalWaitOptions {
                cancellation,
                deadline,
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

    fn into_await_event_options(
        self,
    ) -> Result<(CancellationToken, Option<Instant>), RuntimeEffectControllerError> {
        match self.state {
            RuntimeEffectLocalExecutorState::ExternalWaitOptions {
                cancellation,
                deadline,
            } => Ok((cancellation, deadline)),
            _ => Ok((CancellationToken::new(), None)),
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
                let launch = outcome.launches.pop().ok_or_else(|| {
                    RuntimeEffectControllerError::new(
                        "tool_result_missing",
                        format!("tool `{tool_name}` completed without a launch result"),
                    )
                })?;
                Ok(RuntimeEffectOutcome::ToolCall {
                    launch,
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
            RuntimeEffectCommand::AwaitEvent { key } => {
                let (cancellation, deadline) = local_executor.into_await_event_options()?;
                let resolution = self
                    .await_await_event(&key, cancellation, deadline)
                    .await
                    .map_err(RuntimeEffectControllerError::from)?;
                Ok(RuntimeEffectOutcome::AwaitEvent { resolution })
            }
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
            .await_resolution(key, cancel, deadline)
            .await
    }

    async fn revoke_await_events_for_session(&self, session_id: &str) -> Result<(), RuntimeError> {
        inline_await_events().revoke_session(session_id)
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
impl EffectHost for InlineEffectHost {
    fn durability_tier(&self) -> crate::DurabilityTier {
        self.controller.durability_tier()
    }

    fn requires_durable_attachment_store(&self) -> bool {
        self.controller.requires_durable_attachment_store()
    }

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
