//! Restate durable execution adapter for Lash runtime effects.
//!
//! The primary entrypoint is [`RestateRuntimeEffectController`]. Construct it inside
//! a Restate service, object, or workflow handler, derive a stable
//! [`RuntimeEffectControllerScope`](lash_core::RuntimeEffectControllerScope), and run or resume
//! the Lash turn through the scoped API. Fresh durable turns should use a stable
//! `TurnInput::with_trace_turn_id(turn_id)`; resumed turns should use the
//! facade `session.resume_turn(turn_id).run_with_effect_scope(scope)` handle so
//! Lash reloads its `RuntimeTurnCheckpoint` and runtime effect journal.
//!
//! ```rust,ignore
//! use lash_restate::RestateRuntimeEffectController;
//! use restate_sdk::prelude::*;
//!
//! # #[derive(serde::Serialize, serde::Deserialize)]
//! # struct TurnRequest { turn_id: String }
//! # #[derive(serde::Serialize, serde::Deserialize)]
//! # struct TurnResponse;
//! # async fn run_or_resume_lash_turn(
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
//!         let effect_controller = RestateRuntimeEffectController::new(ctx);
//!         let turn_id = req.turn_id.clone();
//!         let effect_scope = effect_controller
//!             .effect_scope(&turn_id)
//!             .map_err(TerminalError::from_error)?;
//!         let response = run_or_resume_lash_turn(effect_scope, req)
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
//! sleep commands map to Restate's durable timer. Lash's own runtime journal
//! stores the same completed effect outcome by idempotency key and envelope
//! hash before the restored `TurnMachine` consumes it.

use std::fmt;
use std::future::Future;
use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use lash_core::{
    EffectInvocationMetadata, PluginError, ProcessAwaitOutput, ProcessCommand,
    ProcessEffectOutcome, ProcessExternalRef, ProcessRecord, ProcessRegistration, ProcessRegistry,
    ProcessStartReceipt, ProcessState, RuntimeEffectCommand, RuntimeEffectController,
    RuntimeEffectControllerError, RuntimeEffectControllerScope, RuntimeEffectEnvelope,
    RuntimeEffectKind, RuntimeEffectLocalExecutor, RuntimeEffectOutcome, RuntimeError,
};
use restate_sdk::context::{
    Context as RestateContext, ObjectContext, RunRetryPolicy, SharedObjectContext,
    SharedWorkflowContext, WorkflowContext,
};
use restate_sdk::context::{ContextClient, InvocationHandle, RequestTarget};
use restate_sdk::errors::{HandlerResult, TerminalError};
use restate_sdk::serde::Json;
use serde::{Serialize, de::DeserializeOwned};

pub use restate_sdk;

type RestateHandlerFuture<'a, T> =
    Pin<Box<dyn Future<Output = HandlerResult<Json<T>>> + Send + 'a>>;

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

#[derive(Clone, Debug, PartialEq, Eq, Serialize, serde::Deserialize)]
pub struct RestateProcessCancelRequest {
    pub process_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

#[async_trait::async_trait]
pub trait RestateProcessRunner: Send + Sync + 'static {
    async fn run_process(
        &self,
        registration: ProcessRegistration,
    ) -> Result<ProcessAwaitOutput, PluginError>;

    async fn request_process_cancel(
        &self,
        request: RestateProcessCancelRequest,
    ) -> Result<(), PluginError>;
}

#[restate_sdk::workflow]
pub trait LashProcessWorkflow {
    async fn run(
        registration: Json<ProcessRegistration>,
    ) -> HandlerResult<Json<ProcessAwaitOutput>>;

    #[shared]
    async fn cancel(request: Json<RestateProcessCancelRequest>) -> HandlerResult<Json<()>>;
}

pub struct LashProcessWorkflowImpl<R> {
    runner: Arc<R>,
}

impl<R> LashProcessWorkflowImpl<R> {
    pub fn new(runner: Arc<R>) -> Self {
        Self { runner }
    }
}

impl<R> LashProcessWorkflowImpl<R>
where
    R: RestateProcessRunner,
{
    async fn run_registration(
        &self,
        registration: ProcessRegistration,
    ) -> Result<ProcessAwaitOutput, PluginError> {
        self.runner.run_process(registration).await
    }

    async fn cancel_registration(
        &self,
        request: RestateProcessCancelRequest,
    ) -> Result<(), PluginError> {
        self.runner.request_process_cancel(request).await
    }
}

impl<R> LashProcessWorkflow for LashProcessWorkflowImpl<R>
where
    R: RestateProcessRunner,
{
    async fn run(
        &self,
        _ctx: WorkflowContext<'_>,
        Json(registration): Json<ProcessRegistration>,
    ) -> HandlerResult<Json<ProcessAwaitOutput>> {
        self.run_registration(registration)
            .await
            .map(Json)
            .map_err(|err| TerminalError::from_error(err).into())
    }

    async fn cancel(
        &self,
        _ctx: SharedWorkflowContext<'_>,
        Json(request): Json<RestateProcessCancelRequest>,
    ) -> HandlerResult<Json<()>> {
        self.cancel_registration(request)
            .await
            .map(Json)
            .map_err(|err| TerminalError::from_error(err).into())
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

#[doc(hidden)]
pub trait RestateControllerContext<'ctx>: Send + Sync + 'ctx {
    fn sleep_send<'run>(
        &'run self,
        duration: Duration,
    ) -> Pin<Box<dyn Future<Output = Result<(), TerminalError>> + Send + 'run>>
    where
        'ctx: 'run;

    fn run_json_send<'run, T, Fut>(
        &'run self,
        effect_name: String,
        retry_policy: Option<RunRetryPolicy>,
        future: Fut,
    ) -> Pin<Box<dyn Future<Output = Result<Json<T>, TerminalError>> + Send + 'run>>
    where
        'ctx: 'run,
        T: Serialize + DeserializeOwned + Send + 'static,
        Fut: Future<Output = T> + Send + 'run;

    fn start_process_workflow<'run>(
        &'run self,
        registration: ProcessRegistration,
    ) -> Pin<Box<dyn Future<Output = Result<String, TerminalError>> + Send + 'run>>
    where
        'ctx: 'run;

    fn request_process_workflow_cancel<'run>(
        &'run self,
        request: RestateProcessCancelRequest,
    ) -> Pin<Box<dyn Future<Output = Result<(), TerminalError>> + Send + 'run>>
    where
        'ctx: 'run;
}

macro_rules! impl_restate_controller_context {
    ($($context:ident),+ $(,)?) => {
        $(
            impl<'ctx> RestateControllerContext<'ctx> for $context<'ctx> {
                fn sleep_send<'run>(
                    &'run self,
                    duration: Duration,
                ) -> Pin<Box<dyn Future<Output = Result<(), TerminalError>> + Send + 'run>>
                where
                    'ctx: 'run,
                {
                    Box::pin(async move {
                        restate_sdk::context::ContextTimers::sleep(self, duration).await
                    })
                }

                fn run_json_send<'run, T, Fut>(
                    &'run self,
                    effect_name: String,
                    retry_policy: Option<RunRetryPolicy>,
                    future: Fut,
                ) -> Pin<Box<dyn Future<Output = Result<Json<T>, TerminalError>> + Send + 'run>>
                where
                    'ctx: 'run,
                    T: Serialize + DeserializeOwned + Send + 'static,
                    Fut: Future<Output = T> + Send + 'run,
                {
                    let future: RestateHandlerFuture<'run, T> =
                        Box::pin(async move { Ok(Json(future.await)) });
                    let run = restate_sdk::context::ContextSideEffects::run(self, move || future);
                    let run = restate_sdk::context::RunFuture::name(run, effect_name);
                    let run = match retry_policy {
                        Some(policy) => restate_sdk::context::RunFuture::retry_policy(run, policy),
                        None => run,
                    };
                    Box::pin(run)
                }

                fn start_process_workflow<'run>(
                    &'run self,
                    registration: ProcessRegistration,
                ) -> Pin<Box<dyn Future<Output = Result<String, TerminalError>> + Send + 'run>>
                where
                    'ctx: 'run,
                {
                    let workflow_key = registration.id.clone();
                    let request: restate_sdk::context::Request<
                        '_,
                        Json<ProcessRegistration>,
                        Json<ProcessAwaitOutput>,
                    > = ContextClient::request(
                        self,
                        RequestTarget::workflow(
                            "LashProcessWorkflow",
                            workflow_key.clone(),
                            "run",
                        ),
                        Json(registration),
                    )
                    .idempotency_key(format!("lash-process:{workflow_key}:run"));
                    let handle = request.send();
                    Box::pin(async move { handle.invocation_id().await })
                }

                fn request_process_workflow_cancel<'run>(
                    &'run self,
                    request: RestateProcessCancelRequest,
                ) -> Pin<Box<dyn Future<Output = Result<(), TerminalError>> + Send + 'run>>
                where
                    'ctx: 'run,
                {
                    let workflow_key = request.process_id.clone();
                    let request: restate_sdk::context::Request<
                        '_,
                        Json<RestateProcessCancelRequest>,
                        Json<()>,
                    > = ContextClient::request(
                        self,
                        RequestTarget::workflow(
                            "LashProcessWorkflow",
                            workflow_key.clone(),
                            "cancel",
                        ),
                        Json(request),
                    )
                    .idempotency_key(format!("lash-process:{workflow_key}:cancel"));
                    let call = request.call();
                    Box::pin(async move {
                        let Json(()) = call.await?;
                        Ok(())
                    })
                }
            }
        )+
    };
}

impl_restate_controller_context!(
    RestateContext,
    SharedObjectContext,
    ObjectContext,
    SharedWorkflowContext,
    WorkflowContext,
);

/// Lash [`RuntimeEffectController`] backed by a Restate handler context.
///
/// This type is intentionally handler-scoped. Create one inside the Restate
/// handler that owns the Lash turn, then pass [`RestateRuntimeEffectController::effect_scope`]
/// to Lash's scoped turn API with a stable turn ID.
pub struct RestateRuntimeEffectController<'ctx, C> {
    context: C,
    options: RestateEffectControllerOptions,
    _ctx: PhantomData<&'ctx ()>,
}

impl<'ctx, C> RestateRuntimeEffectController<'ctx, C> {
    pub fn new(context: C) -> Self {
        Self::with_options(context, RestateEffectControllerOptions::default())
    }

    pub fn with_options(context: C, options: RestateEffectControllerOptions) -> Self {
        Self {
            context,
            options,
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

impl<'ctx, C> RestateRuntimeEffectController<'ctx, C>
where
    C: RestateControllerContext<'ctx>,
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
        Fut: Future<Output = T> + Send + 'run,
    {
        let effect_name = restate_effect_name(&metadata);
        let run_retry_policy = self.options.run_retry_policy.clone();
        let Json(value) = self
            .context
            .run_json_send(effect_name.clone(), run_retry_policy, future)
            .await
            .map_err(|source| RestateEffectError::Terminal {
                effect: effect_name,
                terminal: source,
            })?;
        Ok(value)
    }
}

impl<C> fmt::Debug for RestateRuntimeEffectController<'_, C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RestateRuntimeEffectController")
            .field("options", &self.options)
            .finish_non_exhaustive()
    }
}

#[async_trait::async_trait]
impl<'ctx, C> RuntimeEffectController for RestateRuntimeEffectController<'ctx, C>
where
    C: RestateControllerContext<'ctx>,
{
    fn requires_durable_attachment_store(&self) -> bool {
        true
    }

    async fn execute_effect(
        &self,
        envelope: RuntimeEffectEnvelope,
        local_executor: RuntimeEffectLocalExecutor<'_>,
    ) -> Result<RuntimeEffectOutcome, RuntimeEffectControllerError> {
        if let RuntimeEffectCommand::Process { command } = envelope.command.clone() {
            let current_hash = envelope.stable_hash()?;
            let metadata = envelope.metadata.clone();
            let journal_hash = current_hash.clone();
            let context = &self.context;
            let journaled = self
                .journal_effect(metadata, async move {
                    let outcome = execute_restate_process_command(context, command, local_executor)
                        .await
                        .map(|result| RuntimeEffectOutcome::Process { result });
                    JournaledRuntimeEffect {
                        envelope_hash: journal_hash,
                        outcome,
                    }
                })
                .await
                .map_err(|err| {
                    RuntimeEffectControllerError::new("restate_effect_controller", err.to_string())
                })?;
            return validate_journaled_effect_hash(journaled, &current_hash)?;
        }
        match restate_effect_execution(&envelope.command) {
            RestateEffectExecution::Timer => {
                let RuntimeEffectCommand::Sleep { duration_ms } = &envelope.command else {
                    unreachable!("timer execution is only selected for sleep effects");
                };
                let duration = Duration::from_millis(*duration_ms);
                if let Err(err) = self.context.sleep_send(duration).await {
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
                let journaled = self
                    .journal_effect(metadata, async move {
                        let outcome = local_executor.execute(envelope).await;
                        JournaledRuntimeEffect {
                            envelope_hash: journal_hash,
                            outcome,
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
}

async fn execute_restate_process_command<'ctx, C>(
    context: &C,
    command: ProcessCommand,
    local_executor: RuntimeEffectLocalExecutor<'_>,
) -> Result<ProcessEffectOutcome, RuntimeEffectControllerError>
where
    C: RestateControllerContext<'ctx> + ?Sized,
{
    let execution = local_executor.into_process()?;
    let registry = execution.registry;
    match command {
        ProcessCommand::Start { registration } => {
            let record = schedule_restate_process(registry, registration, context).await?;
            Ok(ProcessEffectOutcome::Start { record })
        }
        ProcessCommand::Get { process_id } => {
            let record = registry.get(&process_id).await;
            Ok(ProcessEffectOutcome::Get { record })
        }
        ProcessCommand::List { filter } => {
            let records = registry.list(filter).await;
            Ok(ProcessEffectOutcome::List { records })
        }
        ProcessCommand::Await { process_id } => {
            let output = registry.await_process(&process_id).await?;
            Ok(ProcessEffectOutcome::Await { output })
        }
        ProcessCommand::Cancel { process_id, reason } => {
            let record = registry.request_cancel(&process_id, reason.clone()).await?;
            context
                .request_process_workflow_cancel(RestateProcessCancelRequest { process_id, reason })
                .await
                .map_err(|err| {
                    RestateEffectError::BackgroundScheduler(err.to_string()).into_plugin_error()
                })?;
            Ok(ProcessEffectOutcome::Cancel { record })
        }
    }
}

async fn schedule_restate_process<'ctx, C>(
    registry: Arc<dyn ProcessRegistry>,
    registration: lash_core::ProcessRegistration,
    context: &C,
) -> Result<ProcessRecord, PluginError>
where
    C: RestateControllerContext<'ctx> + ?Sized,
{
    let process_id = registration.id.clone();
    registry.register(registration.clone()).await?;
    let invocation_id = context
        .start_process_workflow(registration)
        .await
        .map_err(|err| {
            RestateEffectError::BackgroundScheduler(err.to_string()).into_plugin_error()
        })?;
    let receipt = ProcessStartReceipt {
        process_id: process_id.clone(),
        state: ProcessState::Scheduled,
        external_ref: Some(ProcessExternalRef {
            backend: "restate".to_string(),
            id: format!("LashProcessWorkflow/{process_id}"),
            metadata: Some(serde_json::json!({ "invocation_id": invocation_id })),
        }),
        message: Some("submitted to Restate".to_string()),
    };
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
        | RuntimeEffectCommand::Process { .. }
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

#[cfg(test)]
mod tests {
    use super::*;
    use http_body_util::{BodyExt, Empty};
    use lash_core::{ProcessInput, ProcessRegistration, ProcessScope, ProcessState};
    use restate_sdk::prelude::Endpoint;
    use restate_sdk::service::Discoverable;
    use std::sync::Mutex;

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

    fn prepared_tool_call() -> lash_core::PreparedToolCall {
        lash_core::PreparedToolCall {
            call_id: "call-1".to_string(),
            tool_name: "tool".to_string(),
            args: serde_json::json!({}),
            replay: None,
            prepared_payload: serde_json::Value::Null,
        }
    }

    fn external_registration(id: &str) -> ProcessRegistration {
        ProcessRegistration::new(
            id,
            "test",
            ProcessScope {
                session_id: "session".to_string(),
            },
            ProcessInput::External {
                metadata: serde_json::Value::Null,
            },
        )
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
                    call: prepared_tool_call(),
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

    #[derive(Default)]
    struct RecordingContext {
        sleeps: Mutex<Vec<u64>>,
        runs: Mutex<Vec<String>>,
        started: Mutex<Vec<String>>,
        cancelled: Mutex<Vec<(String, Option<String>)>>,
    }

    impl<'ctx> RestateControllerContext<'ctx> for Arc<RecordingContext> {
        fn sleep_send<'run>(
            &'run self,
            duration: Duration,
        ) -> Pin<Box<dyn Future<Output = Result<(), TerminalError>> + Send + 'run>>
        where
            'ctx: 'run,
        {
            self.sleeps
                .lock()
                .expect("sleeps lock")
                .push(duration.as_millis() as u64);
            Box::pin(async { Ok(()) })
        }

        fn run_json_send<'run, T, Fut>(
            &'run self,
            _effect_name: String,
            _retry_policy: Option<RunRetryPolicy>,
            future: Fut,
        ) -> Pin<Box<dyn Future<Output = Result<Json<T>, TerminalError>> + Send + 'run>>
        where
            'ctx: 'run,
            T: Serialize + DeserializeOwned + Send + 'static,
            Fut: Future<Output = T> + Send + 'run,
        {
            self.runs.lock().expect("runs lock").push(_effect_name);
            Box::pin(async move { Ok(Json(future.await)) })
        }

        fn start_process_workflow<'run>(
            &'run self,
            registration: ProcessRegistration,
        ) -> Pin<Box<dyn Future<Output = Result<String, TerminalError>> + Send + 'run>>
        where
            'ctx: 'run,
        {
            self.started
                .lock()
                .expect("started lock")
                .push(registration.id.clone());
            Box::pin(async move { Ok(format!("invocation-{}", registration.id)) })
        }

        fn request_process_workflow_cancel<'run>(
            &'run self,
            request: RestateProcessCancelRequest,
        ) -> Pin<Box<dyn Future<Output = Result<(), TerminalError>> + Send + 'run>>
        where
            'ctx: 'run,
        {
            self.cancelled
                .lock()
                .expect("cancelled lock")
                .push((request.process_id, request.reason));
            Box::pin(async { Ok(()) })
        }
    }

    fn effect_metadata(kind: RuntimeEffectKind, effect_id: &str) -> EffectInvocationMetadata {
        EffectInvocationMetadata {
            session_id: "session".to_string(),
            origin: lash_core::EffectOrigin::Turn,
            turn_id: Some("turn".to_string()),
            turn_index: Some(1),
            mode_iteration: Some(0),
            effect_id: effect_id.to_string(),
            effect_kind: kind,
            idempotency_key: format!("session:turn:1:0:{}:{effect_id}", kind.as_str()),
            turn_checkpoint_hash: Some("0".repeat(64)),
        }
    }

    #[tokio::test]
    async fn restate_controller_executes_non_sleep_effect_inside_run() {
        let context = Arc::new(RecordingContext::default());
        let host = RestateRuntimeEffectController::new(context.clone());
        let err = host
            .execute_effect(
                RuntimeEffectEnvelope::new(
                    effect_metadata(RuntimeEffectKind::ExecCode, "exec"),
                    RuntimeEffectCommand::ExecCode {
                        code: "1 + 1".to_string(),
                    },
                ),
                RuntimeEffectLocalExecutor::unavailable(),
            )
            .await
            .expect_err("unavailable local executor should be returned from ctx.run");

        assert_eq!(err.code, "runtime_effect_local_executor_unavailable");
        assert_eq!(
            context.runs.lock().expect("runs lock").as_slice(),
            &["lash:session:turn:1:0:exec_code:exec".to_string()]
        );
        assert!(context.sleeps.lock().expect("sleeps lock").is_empty());
    }

    #[tokio::test]
    async fn restate_controller_routes_sleep_only_through_timer() {
        let context = Arc::new(RecordingContext::default());
        let host = RestateRuntimeEffectController::new(context.clone());
        let outcome = host
            .execute_effect(
                RuntimeEffectEnvelope::new(
                    effect_metadata(RuntimeEffectKind::Sleep, "sleep"),
                    RuntimeEffectCommand::Sleep { duration_ms: 42 },
                ),
                RuntimeEffectLocalExecutor::unavailable(),
            )
            .await
            .expect("sleep");

        assert!(matches!(outcome, RuntimeEffectOutcome::Sleep));
        assert_eq!(
            context.sleeps.lock().expect("sleeps lock").as_slice(),
            &[42]
        );
        assert!(context.runs.lock().expect("runs lock").is_empty());
    }

    #[tokio::test]
    async fn restate_controller_schedules_process_workflow_without_running_executor() {
        let context = Arc::new(RecordingContext::default());
        let host = RestateRuntimeEffectController::new(context.clone());
        let registry = Arc::new(lash_core::LocalProcessRegistry::default());
        let registration = external_registration("task-1");
        let outcome = host
            .execute_effect(
                RuntimeEffectEnvelope::new(
                    effect_metadata(RuntimeEffectKind::Process, "background-start"),
                    RuntimeEffectCommand::Process {
                        command: ProcessCommand::Start { registration },
                    },
                ),
                RuntimeEffectLocalExecutor::process_control(registry.clone()),
            )
            .await
            .expect("start");
        let RuntimeEffectOutcome::Process {
            result: ProcessEffectOutcome::Start { record },
        } = outcome
        else {
            panic!("wrong outcome");
        };

        assert_eq!(record.state, ProcessState::Scheduled);
        assert_eq!(record.progress.as_deref(), Some("submitted to Restate"));
        assert_eq!(
            record
                .external_ref
                .as_ref()
                .map(|external| external.id.as_str()),
            Some("LashProcessWorkflow/task-1")
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
                .and_then(|record| record.external_ref)
                .map(|external| (
                    external.backend,
                    external
                        .metadata
                        .and_then(|metadata| metadata.get("invocation_id").cloned())
                )),
            Some((
                "restate".to_string(),
                Some(serde_json::json!("invocation-task-1"))
            ))
        );
        assert_eq!(
            context.started.lock().expect("started lock").as_slice(),
            &["task-1".to_string()]
        );
    }

    #[tokio::test]
    async fn restate_controller_cancel_requests_call_workflow_cancel() {
        let context = Arc::new(RecordingContext::default());
        let host = RestateRuntimeEffectController::new(context.clone());
        let registry = Arc::new(lash_core::LocalProcessRegistry::default());
        let registration = external_registration("task-cancel");
        registry.register(registration).await.expect("register");

        let outcome = host
            .execute_effect(
                RuntimeEffectEnvelope::new(
                    effect_metadata(RuntimeEffectKind::Process, "background-cancel"),
                    RuntimeEffectCommand::Process {
                        command: ProcessCommand::Cancel {
                            process_id: "task-cancel".to_string(),
                            reason: Some("user requested".to_string()),
                        },
                    },
                ),
                RuntimeEffectLocalExecutor::process_control(registry),
            )
            .await
            .expect("cancel");
        let RuntimeEffectOutcome::Process {
            result: ProcessEffectOutcome::Cancel { record },
        } = outcome
        else {
            panic!("wrong outcome");
        };

        assert_eq!(record.state, ProcessState::CancelRequested);
        assert_eq!(
            context.cancelled.lock().expect("cancelled lock").as_slice(),
            &[(
                "task-cancel".to_string(),
                Some("user requested".to_string())
            )]
        );
    }

    #[derive(Default)]
    struct RecordingRunner {
        ran: Mutex<Vec<String>>,
        cancelled: Mutex<Vec<RestateProcessCancelRequest>>,
    }

    #[async_trait::async_trait]
    impl RestateProcessRunner for RecordingRunner {
        async fn run_process(
            &self,
            registration: ProcessRegistration,
        ) -> Result<ProcessAwaitOutput, PluginError> {
            self.ran
                .lock()
                .expect("runner ran lock")
                .push(registration.id);
            Ok(ProcessAwaitOutput::Success {
                value: serde_json::json!({"ok": true}),
                control: None,
            })
        }

        async fn request_process_cancel(
            &self,
            request: RestateProcessCancelRequest,
        ) -> Result<(), PluginError> {
            self.cancelled
                .lock()
                .expect("runner cancelled lock")
                .push(request);
            Ok(())
        }
    }

    fn discover_service<S: Discoverable>(_: &S) -> restate_sdk::discovery::Service {
        S::discover()
    }

    #[tokio::test]
    async fn process_workflow_binds_to_restate_endpoint_and_discovers_handlers() {
        let runner = Arc::new(RecordingRunner::default());
        let service = LashProcessWorkflowImpl::new(runner).serve();
        let discovery = discover_service(&service);
        let endpoint = Endpoint::builder().bind(service).build();

        assert_eq!(discovery.name.to_string(), "LashProcessWorkflow");
        assert_eq!(
            discovery.ty.to_string(),
            restate_sdk::discovery::ServiceType::Workflow.to_string()
        );
        assert_eq!(discovery.handlers.len(), 2);

        let run = discovery
            .handlers
            .iter()
            .find(|handler| handler.name.to_string() == "run")
            .expect("run handler discovery");
        let cancel = discovery
            .handlers
            .iter()
            .find(|handler| handler.name.to_string() == "cancel")
            .expect("cancel handler discovery");

        assert_eq!(
            run.ty.as_ref().map(ToString::to_string).as_deref(),
            Some("WORKFLOW")
        );
        assert_eq!(
            cancel.ty.as_ref().map(ToString::to_string).as_deref(),
            Some("SHARED")
        );

        let response = endpoint.handle(
            http::Request::builder()
                .uri("/discover")
                .header("accept", "application/vnd.restate.endpointmanifest.v3+json")
                .body(Empty::<bytes::Bytes>::new())
                .expect("discover request"),
        );
        assert_eq!(response.status(), http::StatusCode::OK);
        assert_eq!(
            response
                .headers()
                .get(http::header::CONTENT_TYPE)
                .and_then(|value| value.to_str().ok()),
            Some("application/vnd.restate.endpointmanifest.v3+json")
        );
        let body = response
            .into_body()
            .collect()
            .await
            .expect("discover response body")
            .to_bytes();
        let manifest: serde_json::Value =
            serde_json::from_slice(&body).expect("discover response json");
        let workflow = manifest["services"]
            .as_array()
            .expect("services array")
            .iter()
            .find(|service| service["name"] == "LashProcessWorkflow")
            .expect("workflow service");
        let handlers = workflow["handlers"].as_array().expect("handlers array");
        assert!(
            handlers
                .iter()
                .any(|handler| handler["name"] == "run" && handler["ty"] == "WORKFLOW")
        );
        assert!(
            handlers
                .iter()
                .any(|handler| handler["name"] == "cancel" && handler["ty"] == "SHARED")
        );
    }

    #[tokio::test]
    async fn process_workflow_impl_runs_and_cancels_through_runner() {
        let runner = Arc::new(RecordingRunner::default());
        let workflow = LashProcessWorkflowImpl::new(runner.clone());
        let registration = external_registration("task-workflow");

        let output = workflow
            .run_registration(registration)
            .await
            .expect("workflow run");
        workflow
            .cancel_registration(RestateProcessCancelRequest {
                process_id: "task-workflow".to_string(),
                reason: Some("stop".to_string()),
            })
            .await
            .expect("workflow cancel");

        assert!(matches!(output, ProcessAwaitOutput::Success { .. }));
        assert_eq!(
            runner.ran.lock().expect("runner ran lock").as_slice(),
            &["task-workflow".to_string()]
        );
        assert_eq!(
            runner
                .cancelled
                .lock()
                .expect("runner cancelled lock")
                .as_slice(),
            &[RestateProcessCancelRequest {
                process_id: "task-workflow".to_string(),
                reason: Some("stop".to_string()),
            }]
        );
    }
}
