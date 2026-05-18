//! Restate durable execution adapter for Lash runtime effects.
//!
//! The primary entrypoint is [`RestateRuntimeEffectController`]. Construct it inside
//! a Restate service, object, or workflow handler, derive a stable
//! [`RuntimeEffectControllerScope`](lash_core::RuntimeEffectControllerScope), and run the Lash turn
//! through the scoped API.
//!
//! ```rust,ignore
//! use lash_restate::RestateRuntimeEffectController;
//! use restate_sdk::prelude::*;
//!
//! # #[derive(serde::Serialize, serde::Deserialize)]
//! # struct TurnRequest { turn_id: String }
//! # #[derive(serde::Serialize, serde::Deserialize)]
//! # struct TurnResponse;
//! # async fn run_lash_turn(
//! #     _scope: lash_core::RuntimeEffectControllerScope<'_>,
//! #     _req: TurnRequest,
//! # ) -> Result<TurnResponse, std::io::Error> {
//! #     Ok(TurnResponse)
//! # }
//! #[restate_sdk::workflow]
//! pub trait AgentTurnWorkflow {
//!     async fn run(req: Json<TurnRequest>) -> HandlerResult<Json<TurnResponse>>;
//! }
//!
//! pub struct AgentTurnWorkflowImpl;
//!
//! impl AgentTurnWorkflow for AgentTurnWorkflowImpl {
//!     async fn run(
//!         &self,
//!         ctx: WorkflowContext<'_>,
//!         Json(req): Json<TurnRequest>,
//!     ) -> HandlerResult<Json<TurnResponse>> {
//!         # let hooks = todo!();
//!         let effect_controller = RestateRuntimeEffectController::new(ctx, hooks);
//!         let turn_id = req.turn_id.clone();
//!         let effect_scope = effect_controller
//!             .effect_scope(&turn_id)
//!             .map_err(TerminalError::from_error)?;
//!         let response = run_lash_turn(effect_scope, req)
//!             .await
//!             .map_err(TerminalError::from_error)?;
//!         Ok(Json(response))
//!     }
//! }
//! ```
//!
//! Restate's Rust SDK requires journaled closures to be awaited immediately and
//! not to call the Restate context from inside the closure. This adapter follows
//! that rule: every Lash effect is wrapped as one immediately awaited
//! `ctx.run(...).name(envelope.metadata.idempotency_key)` call, and
//! sleep commands map to Restate's durable timer.

use std::fmt;
use std::future::Future;
use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use lash_core::{
    BackgroundTaskLocalExecutor, BackgroundTaskRecord, BackgroundTaskRegistration,
    BackgroundTaskRegistry, BackgroundTaskStartReceipt, EffectInvocationMetadata, PluginError,
    RuntimeEffectCommand, RuntimeEffectController, RuntimeEffectControllerError,
    RuntimeEffectControllerScope, RuntimeEffectEnvelope, RuntimeEffectKind, RuntimeEffectOutcome,
    RuntimeError,
};
use restate_sdk::context::{ContextSideEffects, ContextTimers, RunFuture, RunRetryPolicy};
use restate_sdk::errors::{HandlerResult, TerminalError};
use restate_sdk::serde::Json;
use serde::{Serialize, de::DeserializeOwned};

pub use restate_sdk;

type RestateHandlerFuture<'a, T> =
    Pin<Box<dyn Future<Output = HandlerResult<Json<T>>> + Send + 'a>>;

pub type RestateEffectFuture = Pin<
    Box<
        dyn Future<Output = Result<RuntimeEffectOutcome, RuntimeEffectControllerError>>
            + Send
            + 'static,
    >,
>;

#[derive(Clone, Debug, Serialize, serde::Deserialize)]
struct JournaledRuntimeEffect {
    envelope_hash: String,
    outcome: Result<RuntimeEffectOutcome, RuntimeEffectControllerError>,
}

/// Error raised while bridging a Lash effect to Restate.
#[derive(Debug, thiserror::Error)]
pub enum RestateEffectError {
    #[error("Restate terminal error while running `{effect}`: {terminal}")]
    Terminal {
        effect: String,
        terminal: TerminalError,
    },
    #[error("Restate background scheduler error: {0}")]
    BackgroundScheduler(String),
}

impl RestateEffectError {
    fn into_plugin_error(self) -> PluginError {
        PluginError::Session(self.to_string())
    }
}

/// Application hooks for executing Lash's serialized durable requests.
///
/// Background scheduling should submit or signal a Restate service/workflow
/// using the stable `registration.id` as the Restate workflow/object key.
/// Durable task runners reconstruct work from
/// [`BackgroundTaskRegistration::input`](lash_core::BackgroundTaskRegistration::input).
#[async_trait::async_trait]
pub trait RestateRuntimeHooks: Send + Sync {
    fn execute_effect(&self, envelope: RuntimeEffectEnvelope) -> RestateEffectFuture;

    async fn start_background_task(
        &self,
        registration: lash_core::BackgroundTaskRegistration,
    ) -> Result<BackgroundTaskStartReceipt, RestateEffectError>;

    async fn request_background_task_cancel(
        &self,
        task_id: &str,
        reason: Option<String>,
    ) -> Result<(), RestateEffectError>;
}

#[async_trait::async_trait]
impl<T> RestateRuntimeHooks for std::sync::Arc<T>
where
    T: RestateRuntimeHooks + ?Sized + 'static,
{
    fn execute_effect(&self, envelope: RuntimeEffectEnvelope) -> RestateEffectFuture {
        self.as_ref().execute_effect(envelope)
    }

    async fn start_background_task(
        &self,
        registration: lash_core::BackgroundTaskRegistration,
    ) -> Result<BackgroundTaskStartReceipt, RestateEffectError> {
        self.as_ref().start_background_task(registration).await
    }

    async fn request_background_task_cancel(
        &self,
        task_id: &str,
        reason: Option<String>,
    ) -> Result<(), RestateEffectError> {
        self.as_ref()
            .request_background_task_cancel(task_id, reason)
            .await
    }
}

/// Configuration for [`RestateRuntimeEffectController`].
#[derive(Clone, Default)]
pub struct RestateEffectControllerOptions {
    run_retry_policy: Option<RunRetryPolicy>,
}

impl RestateEffectControllerOptions {
    pub fn new() -> Self {
        Self::default()
    }

    /// Set a Restate retry policy for journaled `ctx.run` effects.
    ///
    /// Lash provider/tool errors are recorded as Lash data, so this policy is
    /// used only when the journaled closure itself fails before producing a
    /// serializable effect result.
    pub fn run_retry_policy(mut self, policy: RunRetryPolicy) -> Self {
        self.run_retry_policy = Some(policy);
        self
    }
}

impl fmt::Debug for RestateEffectControllerOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RestateEffectControllerOptions")
            .field("run_retry_policy", &self.run_retry_policy)
            .finish()
    }
}

/// Lash [`RuntimeEffectController`] backed by a Restate handler context.
///
/// This type is intentionally handler-scoped. Create one inside the Restate
/// handler that owns the Lash turn, then pass [`RestateRuntimeEffectController::effect_scope`]
/// to Lash's scoped turn API with a stable turn ID.
pub struct RestateRuntimeEffectController<'ctx, C, H> {
    context: C,
    options: RestateEffectControllerOptions,
    hooks: H,
    _ctx: PhantomData<&'ctx ()>,
}

impl<'ctx, C, H> RestateRuntimeEffectController<'ctx, C, H> {
    pub fn new(context: C, hooks: H) -> Self {
        Self::with_options(context, hooks, RestateEffectControllerOptions::default())
    }

    pub fn with_options(context: C, hooks: H, options: RestateEffectControllerOptions) -> Self {
        Self {
            context,
            options,
            hooks,
            _ctx: PhantomData,
        }
    }

    pub fn context(&self) -> &C {
        &self.context
    }

    pub fn options(&self) -> &RestateEffectControllerOptions {
        &self.options
    }
}

impl<'ctx, C, H> RestateRuntimeEffectController<'ctx, C, H>
where
    C: ContextSideEffects<'ctx> + ContextTimers<'ctx> + Send + Sync + 'ctx,
    H: RestateRuntimeHooks + Send + Sync + 'static,
{
    pub fn effect_scope<'run>(
        &'run self,
        turn_id: &'run str,
    ) -> Result<RuntimeEffectControllerScope<'run>, RuntimeError> {
        RuntimeEffectControllerScope::new(self, turn_id)
    }

    async fn journal_effect<'run, T, Fut>(
        &'run self,
        metadata: EffectInvocationMetadata,
        future: Fut,
    ) -> Result<T, RestateEffectError>
    where
        'ctx: 'run,
        T: Serialize + DeserializeOwned + Send + 'static,
        Fut: Future<Output = T> + Send + 'static,
    {
        let effect_name = restate_effect_name(&metadata);
        let run_retry_policy = self.options.run_retry_policy.clone();
        let future: RestateHandlerFuture<'ctx, T> = Box::pin(async move { Ok(Json(future.await)) });
        let run = self.context.run(move || future).name(effect_name.clone());
        let run = match run_retry_policy {
            Some(policy) => run.retry_policy(policy),
            None => run,
        };
        let run = unsafe { send_restate_run(run) };
        let Json(value) = run.await.map_err(|source| RestateEffectError::Terminal {
            effect: effect_name,
            terminal: source,
        })?;
        Ok(value)
    }
}

impl<C, H> fmt::Debug for RestateRuntimeEffectController<'_, C, H>
where
    H: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RestateRuntimeEffectController")
            .field("options", &self.options)
            .field("hooks", &self.hooks)
            .finish_non_exhaustive()
    }
}

#[async_trait::async_trait]
impl<'ctx, C, H> RuntimeEffectController for RestateRuntimeEffectController<'ctx, C, H>
where
    C: ContextSideEffects<'ctx> + ContextTimers<'ctx> + Send + Sync + 'ctx,
    H: RestateRuntimeHooks + Send + Sync + 'static,
{
    fn requires_durable_attachment_store(&self) -> bool {
        true
    }

    async fn execute_effect(
        &self,
        envelope: RuntimeEffectEnvelope,
        _local_executor: lash_core::RuntimeEffectLocalExecutor<'_>,
    ) -> Result<RuntimeEffectOutcome, RuntimeEffectControllerError> {
        match restate_effect_execution(&envelope.command) {
            RestateEffectExecution::Timer => {
                let RuntimeEffectCommand::Sleep { duration_ms } = &envelope.command else {
                    unreachable!("timer execution is only selected for sleep effects");
                };
                let duration = Duration::from_millis(*duration_ms);
                let sleep = unsafe { send_restate_sleep(self.context.sleep(duration)) };
                if let Err(err) = sleep.await {
                    tracing_sleep_error(&envelope.metadata, &err);
                    return Err(RuntimeEffectControllerError::new(
                        "restate_effect_controller",
                        err.to_string(),
                    ));
                }
                Ok(RuntimeEffectOutcome::Sleep)
            }
            RestateEffectExecution::JournaledRun => {
                let current_hash = envelope.stable_hash()?;
                let metadata = envelope.metadata.clone();
                let journal_hash = current_hash.clone();
                let effect = self.hooks.execute_effect(envelope);
                let journaled = self
                    .journal_effect(metadata, async move {
                        JournaledRuntimeEffect {
                            envelope_hash: journal_hash,
                            outcome: effect.await,
                        }
                    })
                    .await
                    .map_err(|err| {
                        RuntimeEffectControllerError::new(
                            "restate_effect_controller",
                            err.to_string(),
                        )
                    })?;
                validate_journaled_effect_hash(journaled, &current_hash)?
            }
        }
    }

    async fn start_background_task(
        &self,
        registry: Arc<dyn BackgroundTaskRegistry>,
        registration: BackgroundTaskRegistration,
        _local_executor: BackgroundTaskLocalExecutor,
    ) -> Result<BackgroundTaskRecord, PluginError> {
        schedule_restate_background_task(registry, registration, &self.hooks).await
    }

    async fn request_background_task_cancel(
        &self,
        registry: Arc<dyn BackgroundTaskRegistry>,
        task_id: &str,
        reason: Option<String>,
    ) -> Result<BackgroundTaskRecord, PluginError> {
        let record = registry.request_cancel(task_id, reason.clone()).await?;
        self.hooks
            .request_background_task_cancel(task_id, reason)
            .await
            .map_err(RestateEffectError::into_plugin_error)?;
        Ok(record)
    }
}

async fn schedule_restate_background_task<H>(
    registry: Arc<dyn BackgroundTaskRegistry>,
    registration: lash_core::BackgroundTaskRegistration,
    hooks: &H,
) -> Result<BackgroundTaskRecord, PluginError>
where
    H: RestateRuntimeHooks + ?Sized,
{
    let receipt = hooks
        .start_background_task(registration)
        .await
        .map_err(RestateEffectError::into_plugin_error)?;
    registry.mark_scheduled(receipt).await
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RestateEffectExecution {
    Timer,
    JournaledRun,
}

fn restate_effect_execution(command: &RuntimeEffectCommand) -> RestateEffectExecution {
    match command {
        RuntimeEffectCommand::Sleep { .. } => RestateEffectExecution::Timer,
        RuntimeEffectCommand::LlmCall { .. }
        | RuntimeEffectCommand::DirectCompletion { .. }
        | RuntimeEffectCommand::DirectLlmCompletion { .. }
        | RuntimeEffectCommand::ToolCall { .. }
        | RuntimeEffectCommand::ExecCode { .. }
        | RuntimeEffectCommand::Checkpoint { .. }
        | RuntimeEffectCommand::SyncExecutionSurface { .. } => RestateEffectExecution::JournaledRun,
    }
}

fn restate_effect_name(metadata: &EffectInvocationMetadata) -> String {
    if metadata.idempotency_key.is_empty() {
        format!(
            "lash:{}:{}",
            metadata.effect_kind.as_str(),
            metadata.effect_id
        )
    } else {
        format!("lash:{}", metadata.idempotency_key)
    }
}

fn validate_journaled_effect_hash(
    journaled: JournaledRuntimeEffect,
    current_hash: &str,
) -> Result<Result<RuntimeEffectOutcome, RuntimeEffectControllerError>, RuntimeEffectControllerError>
{
    if journaled.envelope_hash != current_hash {
        return Err(RuntimeEffectControllerError::new(
            "restate_effect_hash_mismatch",
            format!(
                "journaled runtime effect hash {} did not match current envelope hash {}",
                journaled.envelope_hash, current_hash
            ),
        ));
    }
    Ok(journaled.outcome)
}

fn tracing_sleep_error(metadata: &EffectInvocationMetadata, err: &TerminalError) {
    tracing::warn!(
        session_id = %metadata.session_id,
        effect_id = %metadata.effect_id,
        effect_kind = %RuntimeEffectKind::Sleep.as_str(),
        error = %err,
        "Restate durable sleep failed"
    );
}

unsafe fn send_restate_sleep<'a, F>(
    future: F,
) -> Pin<Box<dyn Future<Output = Result<(), TerminalError>> + Send + 'a>>
where
    F: Future<Output = Result<(), TerminalError>> + 'a,
{
    // SAFETY: Restate's handler context futures are tied to one handler and are
    // immediately awaited by this controller. The SDK's public ContextTimers
    // trait does not expose `Send`, while RuntimeEffectController's async trait
    // requires the returned future to be `Send`.
    let boxed: Pin<Box<dyn Future<Output = Result<(), TerminalError>> + 'a>> = Box::pin(future);
    unsafe {
        std::mem::transmute::<
            Pin<Box<dyn Future<Output = Result<(), TerminalError>> + 'a>>,
            Pin<Box<dyn Future<Output = Result<(), TerminalError>> + Send + 'a>>,
        >(boxed)
    }
}

unsafe fn send_restate_run<'a, T, F>(
    future: F,
) -> Pin<Box<dyn Future<Output = Result<Json<T>, TerminalError>> + Send + 'a>>
where
    T: 'static,
    F: Future<Output = Result<Json<T>, TerminalError>> + 'a,
{
    // SAFETY: Restate's journaled run future is created and awaited inside the
    // same handler-scoped controller method. The SDK's public
    // ContextSideEffects trait omits `Send`, while RuntimeEffectController's
    // async trait requires it.
    let boxed: Pin<Box<dyn Future<Output = Result<Json<T>, TerminalError>> + 'a>> =
        Box::pin(future);
    unsafe {
        std::mem::transmute::<
            Pin<Box<dyn Future<Output = Result<Json<T>, TerminalError>> + 'a>>,
            Pin<Box<dyn Future<Output = Result<Json<T>, TerminalError>> + Send + 'a>>,
        >(boxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::{StreamExt, stream};
    use lash_core::{
        BackgroundCancelPolicy, BackgroundTaskInput, BackgroundTaskKind, BackgroundTaskScope,
        BackgroundTaskState,
    };
    use std::sync::Mutex;
    use std::time::SystemTime;

    #[test]
    fn restate_effect_name_uses_lash_idempotency_key() {
        let metadata = EffectInvocationMetadata {
            session_id: "session".to_string(),
            origin: lash_core::EffectOrigin::Turn,
            turn_id: Some("turn".to_string()),
            turn_index: Some(1),
            mode_iteration: Some(2),
            effect_id: "effect".to_string(),
            effect_kind: RuntimeEffectKind::ToolCall,
            idempotency_key: "session:turn:1:2:tool_call:effect".to_string(),
            turn_checkpoint_hash: None,
        };

        assert_eq!(
            restate_effect_name(&metadata),
            "lash:session:turn:1:2:tool_call:effect"
        );
    }

    #[test]
    fn journaled_runtime_effect_hash_mismatch_fails_explicitly() {
        let journaled = JournaledRuntimeEffect {
            envelope_hash: "old".to_string(),
            outcome: Ok(RuntimeEffectOutcome::Sleep),
        };

        let err = validate_journaled_effect_hash(journaled, "new").expect_err("hash mismatch");

        assert_eq!(err.code, "restate_effect_hash_mismatch");
    }

    #[test]
    fn journaled_runtime_effect_hash_match_returns_recorded_outcome() {
        let journaled = JournaledRuntimeEffect {
            envelope_hash: "same".to_string(),
            outcome: Ok(RuntimeEffectOutcome::Sleep),
        };

        let outcome = validate_journaled_effect_hash(journaled, "same")
            .expect("hash match")
            .expect("recorded outcome");

        assert!(matches!(outcome, RuntimeEffectOutcome::Sleep));
    }

    fn llm_spec() -> lash_core::LlmRequestSpec {
        lash_core::LlmRequestSpec {
            model: "model".to_string(),
            messages: Vec::new(),
            attachments: Vec::new(),
            tools: Vec::new(),
            tool_choice: Default::default(),
            model_variant: None,
            session_id: Some("session".to_string()),
            output_spec: None,
        }
    }

    fn direct_spec() -> lash_core::DirectRequestSpec {
        lash_core::DirectRequestSpec {
            model: "model".to_string(),
            model_variant: None,
            messages: Vec::new(),
            attachments: Vec::new(),
            output: Default::default(),
            session_id: Some("session".to_string()),
            originating_tool_call_id: None,
            idempotency_key: None,
        }
    }

    fn pending_tool_call() -> lash_core::sansio::PendingToolCall {
        lash_core::sansio::PendingToolCall {
            call_id: "call-1".to_string(),
            tool_name: "tool".to_string(),
            args: serde_json::json!({}),
            replay: None,
        }
    }

    #[test]
    fn restate_command_execution_plan_is_explicit_for_every_command() {
        let cases = vec![
            (
                RuntimeEffectCommand::Sleep { duration_ms: 1 },
                RestateEffectExecution::Timer,
            ),
            (
                RuntimeEffectCommand::LlmCall {
                    request: llm_spec(),
                },
                RestateEffectExecution::JournaledRun,
            ),
            (
                RuntimeEffectCommand::DirectCompletion {
                    request: direct_spec(),
                    normalized_request: llm_spec(),
                    model: "model".to_string(),
                    usage_source: "test".to_string(),
                },
                RestateEffectExecution::JournaledRun,
            ),
            (
                RuntimeEffectCommand::DirectLlmCompletion {
                    request: llm_spec(),
                    usage_source: "test".to_string(),
                },
                RestateEffectExecution::JournaledRun,
            ),
            (
                RuntimeEffectCommand::ToolCall {
                    call: pending_tool_call(),
                },
                RestateEffectExecution::JournaledRun,
            ),
            (
                RuntimeEffectCommand::ExecCode {
                    code: "1 + 1".to_string(),
                },
                RestateEffectExecution::JournaledRun,
            ),
            (
                RuntimeEffectCommand::Checkpoint {
                    checkpoint: lash_core::CheckpointKind::AfterWork,
                },
                RestateEffectExecution::JournaledRun,
            ),
            (
                RuntimeEffectCommand::SyncExecutionSurface {
                    update_machine_config: true,
                },
                RestateEffectExecution::JournaledRun,
            ),
        ];

        for (command, expected) in cases {
            assert_eq!(restate_effect_execution(&command), expected);
        }
    }

    #[test]
    fn private_send_helpers_return_send_futures() {
        fn assert_send<T: Send>(_: T) {}

        let sleep = unsafe { send_restate_sleep(async { Ok(()) }) };
        assert_send(sleep);
        let run = unsafe { send_restate_run(async { Ok(Json(())) }) };
        assert_send(run);
    }

    #[derive(Default)]
    struct RecordingHooks {
        started: Mutex<Vec<String>>,
        cancelled: Mutex<Vec<(String, Option<String>)>>,
    }

    #[async_trait::async_trait]
    impl RestateRuntimeHooks for RecordingHooks {
        fn execute_effect(&self, envelope: RuntimeEffectEnvelope) -> RestateEffectFuture {
            Box::pin(async move {
                Err(RuntimeEffectControllerError::new(
                    "test_restate_hooks_unsupported",
                    format!(
                        "test hooks cannot execute {}",
                        envelope.command.kind().as_str()
                    ),
                ))
            })
        }

        async fn start_background_task(
            &self,
            registration: lash_core::BackgroundTaskRegistration,
        ) -> Result<BackgroundTaskStartReceipt, RestateEffectError> {
            self.started
                .lock()
                .expect("started lock")
                .push(registration.id.clone());
            Ok(BackgroundTaskStartReceipt {
                task_id: registration.id,
                state: BackgroundTaskState::Scheduled,
                external_ref: Some(lash_core::BackgroundTaskExternalRef {
                    backend: "restate".to_string(),
                    id: "workflow/task-1".to_string(),
                    metadata: Some(serde_json::json!({"service": "tasks"})),
                }),
                message: Some("submitted to Restate".to_string()),
            })
        }

        async fn request_background_task_cancel(
            &self,
            task_id: &str,
            reason: Option<String>,
        ) -> Result<(), RestateEffectError> {
            self.cancelled
                .lock()
                .expect("cancelled lock")
                .push((task_id.to_string(), reason));
            Ok(())
        }
    }

    #[tokio::test]
    async fn restate_hooks_schedule_serialized_registration_without_running_executor() {
        let hooks = Arc::new(RecordingHooks::default());
        let host = TestBackgroundHost {
            hooks: hooks.clone(),
        };
        let registry = Arc::new(MemoryRegistry::default());
        let registration = lash_core::BackgroundTaskRegistration {
            id: "task-1".to_string(),
            kind: BackgroundTaskKind::External,
            producer: "test".to_string(),
            scope: BackgroundTaskScope {
                session_id: "session".to_string(),
            },
            child_session_id: None,
            parent_task_id: None,
            input: BackgroundTaskInput::External {
                metadata: serde_json::Value::Null,
            },
            attempt: Default::default(),
            cancel_policy: BackgroundCancelPolicy::Cooperative,
            close_policy: Default::default(),
        };
        registry
            .register(registration.clone())
            .await
            .expect("register");
        let record = host
            .start_background_task(
                registry.clone(),
                registration,
                BackgroundTaskLocalExecutor::new(|_| async {
                    panic!("Restate test controller should not run the local executor")
                }),
            )
            .await
            .expect("start");

        assert_eq!(record.state, BackgroundTaskState::Scheduled);
        assert_eq!(record.progress.as_deref(), Some("submitted to Restate"));
        assert_eq!(
            record
                .external_ref
                .as_ref()
                .map(|external| external.id.as_str()),
            Some("workflow/task-1")
        );
        assert_eq!(
            registry
                .get("task-1")
                .await
                .expect("get")
                .progress
                .as_deref(),
            Some("submitted to Restate")
        );
        assert_eq!(
            registry
                .list(Default::default())
                .await
                .into_iter()
                .next()
                .and_then(|record| record.external_ref.map(|external| external.backend)),
            Some("restate".to_string())
        );
        assert_eq!(
            hooks.started.lock().expect("started lock").as_slice(),
            &["task-1".to_string()]
        );
    }

    struct TestBackgroundHost {
        hooks: Arc<RecordingHooks>,
    }

    #[async_trait::async_trait]
    impl RuntimeEffectController for TestBackgroundHost {
        async fn execute_effect(
            &self,
            envelope: RuntimeEffectEnvelope,
            _local_executor: lash_core::RuntimeEffectLocalExecutor<'_>,
        ) -> Result<RuntimeEffectOutcome, RuntimeEffectControllerError> {
            Err(RuntimeEffectControllerError::new(
                "test_background_host_unsupported",
                format!(
                    "test background controller cannot execute {}",
                    envelope.command.kind().as_str()
                ),
            ))
        }

        async fn start_background_task(
            &self,
            registry: Arc<dyn BackgroundTaskRegistry>,
            registration: BackgroundTaskRegistration,
            _local_executor: BackgroundTaskLocalExecutor,
        ) -> Result<BackgroundTaskRecord, PluginError> {
            schedule_restate_background_task(registry, registration, self.hooks.as_ref()).await
        }

        async fn request_background_task_cancel(
            &self,
            registry: Arc<dyn BackgroundTaskRegistry>,
            task_id: &str,
            reason: Option<String>,
        ) -> Result<BackgroundTaskRecord, PluginError> {
            let record = registry.request_cancel(task_id, reason.clone()).await?;
            self.hooks
                .request_background_task_cancel(task_id, reason)
                .await
                .map_err(RestateEffectError::into_plugin_error)?;
            Ok(record)
        }
    }

    #[derive(Default)]
    struct MemoryRegistry {
        record: Mutex<Option<BackgroundTaskRecord>>,
    }

    fn record_from_registration(
        registration: lash_core::BackgroundTaskRegistration,
        state: BackgroundTaskState,
    ) -> BackgroundTaskRecord {
        let now = SystemTime::now();
        BackgroundTaskRecord {
            id: registration.id,
            kind: registration.kind,
            producer: registration.producer,
            scope: registration.scope,
            parent_task_id: registration.parent_task_id,
            child_session_id: registration.child_session_id,
            input: registration.input,
            state,
            cancel_policy: registration.cancel_policy,
            close_policy: registration.close_policy,
            attempt: registration.attempt,
            external_ref: None,
            progress: None,
            result: None,
            failure: None,
            created_at: now,
            updated_at: now,
            completed_at: None,
        }
    }

    #[async_trait::async_trait]
    impl BackgroundTaskRegistry for MemoryRegistry {
        async fn register(
            &self,
            registration: lash_core::BackgroundTaskRegistration,
        ) -> Result<BackgroundTaskRecord, PluginError> {
            let record = record_from_registration(registration, BackgroundTaskState::Pending);
            *self.record.lock().expect("record lock") = Some(record.clone());
            Ok(record)
        }

        async fn await_completion(
            &self,
            task_id: &str,
        ) -> Result<lash_core::BackgroundTaskCompletion, PluginError> {
            Err(PluginError::Session(format!(
                "test registry has no completion for `{task_id}`"
            )))
        }

        async fn update(
            &self,
            task_id: &str,
            update: lash_core::BackgroundTaskUpdate,
        ) -> Result<BackgroundTaskRecord, PluginError> {
            let mut guard = self.record.lock().expect("record lock");
            let Some(record) = guard.as_mut() else {
                return Err(PluginError::Session(format!("unknown task `{task_id}`")));
            };
            if record.id != task_id {
                return Err(PluginError::Session(format!("unknown task `{task_id}`")));
            }
            if let Some(state) = update.state {
                record.state = state;
            }
            if let Some(progress) = update.progress {
                record.progress = Some(progress);
            }
            record.updated_at = SystemTime::now();
            Ok(record.clone())
        }

        async fn mark_scheduled(
            &self,
            receipt: BackgroundTaskStartReceipt,
        ) -> Result<BackgroundTaskRecord, PluginError> {
            let mut guard = self.record.lock().expect("record lock");
            let Some(record) = guard.as_mut() else {
                return Err(PluginError::Session(format!(
                    "unknown task `{}`",
                    receipt.task_id
                )));
            };
            if record.id != receipt.task_id {
                return Err(PluginError::Session(format!(
                    "unknown task `{}`",
                    receipt.task_id
                )));
            }
            record.state = if receipt.state == BackgroundTaskState::Pending {
                BackgroundTaskState::Scheduled
            } else {
                receipt.state
            };
            record.external_ref = receipt.external_ref;
            record.progress = receipt.message;
            record.updated_at = SystemTime::now();
            Ok(record.clone())
        }

        async fn complete(
            &self,
            task_id: &str,
            _outcome: lash_core::BackgroundTaskCompletion,
        ) -> Result<BackgroundTaskRecord, PluginError> {
            self.update(
                task_id,
                lash_core::BackgroundTaskUpdate {
                    state: Some(BackgroundTaskState::Completed),
                    progress: None,
                },
            )
            .await
        }

        async fn request_cancel(
            &self,
            task_id: &str,
            _reason: Option<String>,
        ) -> Result<BackgroundTaskRecord, PluginError> {
            self.update(
                task_id,
                lash_core::BackgroundTaskUpdate {
                    state: Some(BackgroundTaskState::CancelRequested),
                    progress: None,
                },
            )
            .await
        }

        async fn get(&self, _task_id: &str) -> Option<BackgroundTaskRecord> {
            self.record.lock().expect("record lock").clone()
        }

        async fn list(
            &self,
            _filter: lash_core::BackgroundTaskFilter,
        ) -> Vec<BackgroundTaskRecord> {
            self.record
                .lock()
                .expect("record lock")
                .clone()
                .into_iter()
                .collect()
        }

        async fn transfer(
            &self,
            task_id: &str,
            scope: BackgroundTaskScope,
        ) -> Result<BackgroundTaskRecord, PluginError> {
            let mut guard = self.record.lock().expect("record lock");
            let Some(record) = guard.as_mut() else {
                return Err(PluginError::Session(format!("unknown task `{task_id}`")));
            };
            record.scope = scope;
            Ok(record.clone())
        }

        fn subscribe(
            &self,
            _filter: lash_core::BackgroundTaskFilter,
        ) -> futures_util::stream::BoxStream<'static, lash_core::BackgroundTaskEvent> {
            stream::empty().boxed()
        }
    }
}
