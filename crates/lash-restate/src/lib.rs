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
//! `ctx.run(...).name(envelope.metadata.idempotency_key)` call, sleep commands
//! map to Restate's durable timer, and process commands call Restate workflow
//! scheduling directly through idempotent registry/workflow operations. Lash's
//! own runtime journal stores the same completed effect outcome by idempotency
//! key and envelope hash before the restored `TurnMachine` consumes it.

use std::fmt;
use std::future::Future;
use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use lash_core::{
    DurableProcessWorker, EffectInvocationMetadata, PluginError, ProcessAwaitOutput,
    ProcessCommand, ProcessEffectOutcome, ProcessExecutionContext, ProcessExternalRef,
    ProcessRecord, ProcessRegistration, ProcessRegistry, RuntimeEffectCommand,
    RuntimeEffectController, RuntimeEffectControllerError, RuntimeEffectControllerScope,
    RuntimeEffectEnvelope, RuntimeEffectKind, RuntimeEffectLocalExecutor, RuntimeEffectOutcome,
    RuntimeError,
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
        execution_context: ProcessExecutionContext,
    ) -> Result<ProcessAwaitOutput, PluginError>;

    async fn request_process_cancel(
        &self,
        request: RestateProcessCancelRequest,
    ) -> Result<(), PluginError>;
}

#[derive(Clone)]
pub struct RestateCoreProcessRunner {
    worker: DurableProcessWorker,
}

impl RestateCoreProcessRunner {
    pub fn new(worker: DurableProcessWorker) -> Self {
        Self { worker }
    }

    pub fn worker(&self) -> &DurableProcessWorker {
        &self.worker
    }
}

#[async_trait::async_trait]
impl RestateProcessRunner for RestateCoreProcessRunner {
    async fn run_process(
        &self,
        registration: ProcessRegistration,
        execution_context: ProcessExecutionContext,
    ) -> Result<ProcessAwaitOutput, PluginError> {
        self.worker
            .run_process(
                registration,
                execution_context,
                tokio_util::sync::CancellationToken::new(),
            )
            .await
    }

    async fn request_process_cancel(
        &self,
        request: RestateProcessCancelRequest,
    ) -> Result<(), PluginError> {
        self.worker
            .request_process_cancel(&request.process_id, request.reason)
            .await
    }
}

#[restate_sdk::workflow]
pub trait LashProcessWorkflow {
    async fn run(
        input: Json<RestateProcessWorkflowInput>,
    ) -> HandlerResult<Json<ProcessAwaitOutput>>;

    #[shared]
    async fn cancel(request: Json<RestateProcessCancelRequest>) -> HandlerResult<Json<()>>;
}

#[derive(Clone, Debug, Serialize, serde::Deserialize)]
pub struct RestateProcessWorkflowInput {
    pub registration: ProcessRegistration,
    #[serde(default, skip_serializing_if = "ProcessExecutionContext::is_empty")]
    pub execution_context: ProcessExecutionContext,
}

pub struct LashProcessWorkflowImpl<R> {
    runner: Arc<R>,
    registry: Arc<dyn ProcessRegistry>,
}

impl<R> LashProcessWorkflowImpl<R> {
    pub fn new(runner: Arc<R>, registry: Arc<dyn ProcessRegistry>) -> Self {
        Self { runner, registry }
    }
}

impl<R> LashProcessWorkflowImpl<R>
where
    R: RestateProcessRunner,
{
    async fn run_registration(
        &self,
        registration: ProcessRegistration,
        execution_context: ProcessExecutionContext,
    ) -> Result<ProcessAwaitOutput, PluginError> {
        let process_id = registration.id.clone();
        let output = self
            .runner
            .run_process(registration, execution_context)
            .await;
        match output {
            Ok(output) => {
                self.registry
                    .complete_process(&process_id, output.clone())
                    .await?;
                Ok(output)
            }
            Err(err) => {
                let output = ProcessAwaitOutput::Failure {
                    class: lash_core::ToolFailureClass::Execution,
                    code: "restate_process_runner_failed".to_string(),
                    message: err.to_string(),
                    raw: None,
                    control: None,
                };
                let _ = self.registry.complete_process(&process_id, output).await;
                Err(err)
            }
        }
    }

    async fn cancel_registration(
        &self,
        request: RestateProcessCancelRequest,
    ) -> Result<(), PluginError> {
        self.registry
            .append_event(
                &request.process_id,
                lash_core::ProcessEventAppendRequest::cancel_requested(
                    &request.process_id,
                    request.reason.clone(),
                ),
            )
            .await?;
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
        Json(input): Json<RestateProcessWorkflowInput>,
    ) -> HandlerResult<Json<ProcessAwaitOutput>> {
        self.run_registration(input.registration, input.execution_context)
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
        execution_context: ProcessExecutionContext,
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
                    execution_context: ProcessExecutionContext,
                ) -> Pin<Box<dyn Future<Output = Result<String, TerminalError>> + Send + 'run>>
                where
                    'ctx: 'run,
                {
                    let workflow_key = registration.id.clone();
                    let request: restate_sdk::context::Request<
                        '_,
                        Json<RestateProcessWorkflowInput>,
                        Json<ProcessAwaitOutput>,
                    > = ContextClient::request(
                        self,
                        RequestTarget::workflow(
                            "LashProcessWorkflow",
                            workflow_key.clone(),
                            "run",
                        ),
                        Json(RestateProcessWorkflowInput {
                            registration,
                            execution_context,
                        }),
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
        match restate_effect_execution(&envelope.command) {
            RestateEffectExecution::DirectProcess => {
                let RuntimeEffectCommand::Process { command } = envelope.command else {
                    unreachable!("direct process execution is only selected for process effects");
                };
                execute_restate_process_command(&self.context, command, local_executor)
                    .await
                    .map(|result| RuntimeEffectOutcome::Process { result })
            }
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
        ProcessCommand::Start {
            registration,
            grant,
            execution_context,
        } => {
            let record = schedule_restate_process(
                registry,
                registration,
                grant,
                *execution_context,
                context,
            )
            .await?;
            Ok(ProcessEffectOutcome::Start { record })
        }
        ProcessCommand::List { owner_scope } => {
            let entries = registry.list_handle_grants(&owner_scope).await?;
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
            for process_id in &report.cancel_process_ids {
                registry
                    .append_event(
                        process_id,
                        lash_core::ProcessEventAppendRequest::cancel_requested(
                            process_id,
                            Some("session deleted".to_string()),
                        ),
                    )
                    .await?;
                context
                    .request_process_workflow_cancel(RestateProcessCancelRequest {
                        process_id: process_id.clone(),
                        reason: Some("session deleted".to_string()),
                    })
                    .await
                    .map_err(|err| {
                        RestateEffectError::BackgroundScheduler(err.to_string()).into_plugin_error()
                    })?;
            }
            Ok(ProcessEffectOutcome::DeleteSession { report })
        }
        ProcessCommand::Await { process_id } => {
            let output = registry.await_process(&process_id).await?;
            Ok(ProcessEffectOutcome::Await { output })
        }
        ProcessCommand::Cancel { process_id, reason } => {
            let record = registry
                .get_process(&process_id)
                .await
                .ok_or_else(|| PluginError::Session(format!("unknown process `{process_id}`")))?;
            registry
                .append_event(
                    &process_id,
                    lash_core::ProcessEventAppendRequest::cancel_requested(
                        &process_id,
                        reason.clone(),
                    ),
                )
                .await?;
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
    grant: Option<lash_core::ProcessStartGrant>,
    execution_context: lash_core::ProcessExecutionContext,
    context: &C,
) -> Result<ProcessRecord, PluginError>
where
    C: RestateControllerContext<'ctx> + ?Sized,
{
    let process_id = registration.id.clone();
    let record = registry.register_process(registration.clone()).await?;
    if let Some(grant) = grant {
        registry
            .grant_handle(&grant.owner_scope, &process_id, grant.descriptor)
            .await?;
    }
    if record.external_ref.is_some() {
        return Ok(record);
    }
    let invocation_id = context
        .start_process_workflow(registration, execution_context)
        .await
        .map_err(|err| {
            RestateEffectError::BackgroundScheduler(err.to_string()).into_plugin_error()
        })?;
    registry
        .set_external_ref(
            &process_id,
            ProcessExternalRef {
                backend: "restate".to_string(),
                id: format!("LashProcessWorkflow/{process_id}"),
                metadata: Some(serde_json::json!({ "invocation_id": invocation_id })),
            },
        )
        .await
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RestateEffectExecution {
    DirectProcess,
    Timer,
    JournaledRun,
}

fn restate_effect_execution(command: &RuntimeEffectCommand) -> RestateEffectExecution {
    match command {
        RuntimeEffectCommand::Process { .. } => RestateEffectExecution::DirectProcess,
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

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::{BufMut, Bytes, BytesMut};
    use http_body_util::{BodyExt, Empty, Full};
    use lash_core::{ProcessInput, ProcessRegistration};
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
            protocol_iteration: Some(2),
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
            generation: lash_core::GenerationOptions::default(),
            session_id: Some("session".to_string()),
            output_spec: None,
        }
    }

    fn direct_spec() -> lash_core::DirectRequestSpec {
        lash_core::DirectRequestSpec {
            model: "model".to_string(),
            model_variant: None,
            generation: lash_core::GenerationOptions::default(),
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
            ProcessInput::External {
                metadata: serde_json::Value::Null,
            },
        )
    }

    fn process_registry() -> Arc<dyn ProcessRegistry> {
        Arc::new(lash_sqlite_store::SqliteProcessRegistry::memory().expect("sqlite registry"))
    }

    const RESTATE_INVOCATION_CONTENT_TYPE: &str = "application/vnd.restate.invocation.v6";

    fn encode_restate_message(message_type: u16, payload: Vec<u8>) -> Bytes {
        let mut encoded = BytesMut::with_capacity(8 + payload.len());
        let header = ((message_type as u64) << 48) | payload.len() as u64;
        encoded.put_u64(header);
        encoded.extend_from_slice(&payload);
        encoded.freeze()
    }

    fn put_varint(buf: &mut BytesMut, mut value: u64) {
        while value >= 0x80 {
            buf.put_u8(((value as u8) & 0x7f) | 0x80);
            value >>= 7;
        }
        buf.put_u8(value as u8);
    }

    fn put_field_key(buf: &mut BytesMut, field_number: u32, wire_type: u8) {
        put_varint(buf, ((field_number as u64) << 3) | wire_type as u64);
    }

    fn put_varint_field(buf: &mut BytesMut, field_number: u32, value: u64) {
        put_field_key(buf, field_number, 0);
        put_varint(buf, value);
    }

    fn put_len_field(buf: &mut BytesMut, field_number: u32, value: &[u8]) {
        put_field_key(buf, field_number, 2);
        put_varint(buf, value.len() as u64);
        buf.extend_from_slice(value);
    }

    fn encode_start_message(workflow_key: &str) -> Bytes {
        let mut payload = BytesMut::new();
        put_len_field(&mut payload, 1, workflow_key.as_bytes());
        put_len_field(&mut payload, 2, workflow_key.as_bytes());
        put_varint_field(&mut payload, 3, 1);
        put_len_field(&mut payload, 6, workflow_key.as_bytes());
        encode_restate_message(0x0000, payload.to_vec())
    }

    fn encode_input_command(payload: &[u8]) -> Bytes {
        let mut value = BytesMut::new();
        put_len_field(&mut value, 1, payload);

        let mut command = BytesMut::new();
        put_len_field(&mut command, 14, &value);
        encode_restate_message(0x0400, command.to_vec())
    }

    fn encode_invocation_body<T: serde::Serialize>(
        workflow_key: &str,
        input: &T,
    ) -> Result<Bytes, TerminalError> {
        let input = serde_json::to_vec(input).map_err(TerminalError::from_error)?;
        let start = encode_start_message(workflow_key);
        let input = encode_input_command(&input);
        let mut body = BytesMut::with_capacity(start.len() + input.len());
        body.extend_from_slice(&start);
        body.extend_from_slice(&input);
        Ok(body.freeze())
    }

    async fn invoke_process_workflow_endpoint<T: serde::Serialize>(
        endpoint: &Endpoint,
        handler: &str,
        workflow_key: &str,
        input: &T,
    ) -> Result<Bytes, TerminalError> {
        let response = endpoint.handle(
            http::Request::builder()
                .uri(format!("/invoke/LashProcessWorkflow/{handler}"))
                .header(http::header::CONTENT_TYPE, RESTATE_INVOCATION_CONTENT_TYPE)
                .body(Full::new(encode_invocation_body(workflow_key, input)?))
                .expect("workflow invocation request"),
        );
        let status = response.status();
        if !status.is_success() {
            return Err(TerminalError::new_with_code(
                status.as_u16(),
                format!("workflow endpoint invocation returned status {status}"),
            ));
        }
        response
            .into_body()
            .collect()
            .await
            .map(|body| body.to_bytes())
            .map_err(|err| TerminalError::new(format!("workflow endpoint body failed: {err}")))
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
                    request: Box::new(llm_spec()),
                },
                RestateEffectExecution::JournaledRun,
            ),
            (
                RuntimeEffectCommand::DirectCompletion {
                    request: Box::new(direct_spec()),
                    normalized_request: Box::new(llm_spec()),
                    model: "model".to_string(),
                    usage_source: "test".to_string(),
                },
                RestateEffectExecution::JournaledRun,
            ),
            (
                RuntimeEffectCommand::DirectLlmCompletion {
                    request: Box::new(llm_spec()),
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
        endpoint: Option<Endpoint>,
        sleeps: Mutex<Vec<u64>>,
        runs: Mutex<Vec<String>>,
        started: Mutex<Vec<ProcessRegistration>>,
        started_execution_contexts: Mutex<Vec<ProcessExecutionContext>>,
        cancelled: Mutex<Vec<(String, Option<String>)>>,
    }

    impl RecordingContext {
        fn with_endpoint(endpoint: Endpoint) -> Self {
            Self {
                endpoint: Some(endpoint),
                ..Default::default()
            }
        }
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
            execution_context: ProcessExecutionContext,
        ) -> Pin<Box<dyn Future<Output = Result<String, TerminalError>> + Send + 'run>>
        where
            'ctx: 'run,
        {
            let process_id = registration.id.clone();
            let endpoint = self.endpoint.clone();
            self.started
                .lock()
                .expect("started lock")
                .push(registration.clone());
            self.started_execution_contexts
                .lock()
                .expect("started execution contexts lock")
                .push(execution_context.clone());
            Box::pin(async move {
                if let Some(endpoint) = endpoint {
                    invoke_process_workflow_endpoint(
                        &endpoint,
                        "run",
                        &process_id,
                        &RestateProcessWorkflowInput {
                            registration,
                            execution_context,
                        },
                    )
                    .await?;
                }
                Ok(format!("invocation-{process_id}"))
            })
        }

        fn request_process_workflow_cancel<'run>(
            &'run self,
            request: RestateProcessCancelRequest,
        ) -> Pin<Box<dyn Future<Output = Result<(), TerminalError>> + Send + 'run>>
        where
            'ctx: 'run,
        {
            let endpoint = self.endpoint.clone();
            let process_id = request.process_id.clone();
            self.cancelled
                .lock()
                .expect("cancelled lock")
                .push((request.process_id.clone(), request.reason.clone()));
            Box::pin(async move {
                if let Some(endpoint) = endpoint {
                    invoke_process_workflow_endpoint(&endpoint, "cancel", &process_id, &request)
                        .await?;
                }
                Ok(())
            })
        }
    }

    fn effect_metadata(kind: RuntimeEffectKind, effect_id: &str) -> EffectInvocationMetadata {
        EffectInvocationMetadata {
            session_id: "session".to_string(),
            origin: lash_core::EffectOrigin::Turn,
            turn_id: Some("turn".to_string()),
            turn_index: Some(1),
            protocol_iteration: Some(0),
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
        let registry = process_registry();
        let registration = external_registration("task-1");
        let outcome = host
            .execute_effect(
                RuntimeEffectEnvelope::new(
                    effect_metadata(RuntimeEffectKind::Process, "background-start"),
                    RuntimeEffectCommand::Process {
                        command: ProcessCommand::Start {
                            registration,
                            grant: Some(lash_core::ProcessStartGrant {
                                owner_scope: lash_core::ProcessScope::new("runtime", "session"),
                                descriptor: lash_core::ProcessHandleDescriptor::new(
                                    Some("tool"),
                                    Some("task"),
                                ),
                            }),
                            execution_context: Box::new(ProcessExecutionContext::default()),
                        },
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

        assert_eq!(
            record
                .external_ref
                .as_ref()
                .map(|external| external.id.as_str()),
            Some("LashProcessWorkflow/task-1")
        );
        assert_eq!(
            registry
                .get_process("task-1")
                .await
                .expect("get")
                .external_ref
                .as_ref()
                .map(|external| external.id.as_str()),
            Some("LashProcessWorkflow/task-1")
        );
        let owner_scope = lash_core::ProcessScope::new("runtime", "session");
        assert_eq!(
            registry
                .list_handle_grants(&owner_scope)
                .await
                .expect("grants")
                .into_iter()
                .next()
                .and_then(|(_, record)| record.external_ref)
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
            context
                .started
                .lock()
                .expect("started lock")
                .iter()
                .map(|registration| registration.id.as_str())
                .collect::<Vec<_>>(),
            vec!["task-1"]
        );
        assert!(
            context.runs.lock().expect("runs lock").is_empty(),
            "process workflow scheduling must not call Restate context from inside ctx.run"
        );
    }

    #[tokio::test]
    async fn restate_controller_schedules_lashlang_process_with_serializable_input() {
        let context = Arc::new(RecordingContext::default());
        let host = RestateRuntimeEffectController::new(context.clone());
        let registry = process_registry();
        let module = lashlang::parse("process scan(root: str) { finish root }")
            .expect("lashlang process module");
        let catalog = lashlang::ResourceCatalog::new();
        let linked_module = lashlang::LinkedModule::link(
            module.clone(),
            lashlang::LashlangSurface::new(catalog, lashlang::LashlangAbilities::all()),
        )
        .expect("link lashlang module");
        let process_ref = linked_module
            .artifact
            .process_ref("scan")
            .expect("scan process ref")
            .clone();
        let mut args = serde_json::Map::new();
        args.insert("root".to_string(), serde_json::json!("."));
        let registration = ProcessRegistration::new(
            "process-1",
            ProcessInput::LashlangProcess {
                module_ref: linked_module.module_ref.clone(),
                process_ref: process_ref.clone(),
                required_surface_ref: linked_module.required_surface_ref.clone(),
                process_name: "scan".to_string(),
                args: args.clone(),
            },
        )
        .with_extra_event_types(lash_core::lashlang_process_event_types());

        let outcome = host
            .execute_effect(
                RuntimeEffectEnvelope::new(
                    effect_metadata(RuntimeEffectKind::Process, "lashlang-process-start"),
                    RuntimeEffectCommand::Process {
                        command: ProcessCommand::Start {
                            registration,
                            grant: None,
                            execution_context: Box::new(
                                ProcessExecutionContext::default().with_wake_target_scope(
                                    lash_core::ProcessScope::new("runtime", "session"),
                                ),
                            ),
                        },
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

        assert_eq!(
            record
                .external_ref
                .as_ref()
                .map(|external| external.backend.as_str()),
            Some("restate")
        );
        assert_eq!(
            registry
                .get_process("process-1")
                .await
                .expect("registered process")
                .external_ref
                .as_ref()
                .map(|external| external.backend.as_str()),
            Some("restate")
        );
        let started = context.started.lock().expect("started lock");
        assert_eq!(started.len(), 1);
        let ProcessInput::LashlangProcess {
            module_ref,
            process_ref: sent_process_ref,
            required_surface_ref,
            process_name,
            args: sent_args,
            ..
        } = started[0].input.as_ref()
        else {
            panic!("expected lashlang process input");
        };
        assert_eq!(module_ref, &linked_module.module_ref);
        assert_eq!(sent_process_ref, &process_ref);
        assert_eq!(required_surface_ref, &linked_module.required_surface_ref);
        assert_eq!(process_name, "scan");
        assert_eq!(sent_args, &args);
        assert_eq!(
            context
                .started_execution_contexts
                .lock()
                .expect("started execution contexts lock")
                .iter()
                .map(|context| {
                    context
                        .wake_target_scope
                        .as_ref()
                        .map(|scope| scope.session_id.as_str())
                })
                .collect::<Vec<_>>(),
            vec![Some("session")]
        );
    }

    #[tokio::test]
    async fn restate_controller_lists_and_transfers_grants_through_process_effects() {
        let context = Arc::new(RecordingContext::default());
        let host = RestateRuntimeEffectController::new(context.clone());
        let registry = process_registry();
        let s1 = lash_core::ProcessScope::new("runtime", "s1");
        let s2 = lash_core::ProcessScope::new("runtime", "s2");
        registry
            .register_process(external_registration("task-list"))
            .await
            .expect("register");
        registry
            .grant_handle(
                &s1,
                "task-list",
                lash_core::ProcessHandleDescriptor::new(Some("tool"), Some("task")),
            )
            .await
            .expect("grant");

        let outcome = host
            .execute_effect(
                RuntimeEffectEnvelope::new(
                    effect_metadata(RuntimeEffectKind::Process, "process-list-s1"),
                    RuntimeEffectCommand::Process {
                        command: ProcessCommand::List {
                            owner_scope: s1.clone(),
                        },
                    },
                ),
                RuntimeEffectLocalExecutor::process_control(registry.clone()),
            )
            .await
            .expect("list");
        let RuntimeEffectOutcome::Process {
            result: ProcessEffectOutcome::List { entries },
        } = outcome
        else {
            panic!("wrong list outcome");
        };
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].0.process_id, "task-list");

        let outcome = host
            .execute_effect(
                RuntimeEffectEnvelope::new(
                    effect_metadata(RuntimeEffectKind::Process, "process-transfer"),
                    RuntimeEffectCommand::Process {
                        command: ProcessCommand::Transfer {
                            from_scope: s1.clone(),
                            to_scope: s2.clone(),
                            process_ids: vec!["task-list".to_string()],
                        },
                    },
                ),
                RuntimeEffectLocalExecutor::process_control(registry.clone()),
            )
            .await
            .expect("transfer");
        assert!(matches!(
            outcome,
            RuntimeEffectOutcome::Process {
                result: ProcessEffectOutcome::Transfer
            }
        ));

        let entries = registry.list_handle_grants(&s2).await.expect("s2 grants");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].0.process_id, "task-list");
        assert!(
            registry
                .list_handle_grants(&s1)
                .await
                .expect("s1")
                .is_empty()
        );
        assert!(context.started.lock().expect("started lock").is_empty());
    }

    #[tokio::test]
    async fn restate_controller_cancel_requests_call_workflow_cancel() {
        let context = Arc::new(RecordingContext::default());
        let host = RestateRuntimeEffectController::new(context.clone());
        let registry = process_registry();
        let registration = external_registration("task-cancel");
        registry
            .register_process(registration)
            .await
            .expect("register");

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

        assert!(!record.is_terminal());
        assert_eq!(
            context.cancelled.lock().expect("cancelled lock").as_slice(),
            &[(
                "task-cancel".to_string(),
                Some("user requested".to_string())
            )]
        );
    }

    #[derive(Debug, PartialEq, Eq)]
    struct RecordedProcessRun {
        process_id: String,
        wake_target_session_id: Option<String>,
        tool_effect_id: Option<String>,
    }

    #[derive(Default)]
    struct RecordingRunner {
        ran: Mutex<Vec<RecordedProcessRun>>,
        cancelled: Mutex<Vec<RestateProcessCancelRequest>>,
    }

    #[async_trait::async_trait]
    impl RestateProcessRunner for RecordingRunner {
        async fn run_process(
            &self,
            registration: ProcessRegistration,
            execution_context: ProcessExecutionContext,
        ) -> Result<ProcessAwaitOutput, PluginError> {
            self.ran
                .lock()
                .expect("runner ran lock")
                .push(RecordedProcessRun {
                    process_id: registration.id,
                    wake_target_session_id: execution_context
                        .wake_target_scope
                        .map(|scope| scope.session_id),
                    tool_effect_id: execution_context
                        .tool_effect_metadata
                        .map(|metadata| metadata.effect_id),
                });
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

    #[tokio::test]
    async fn process_workflow_endpoint_smoke_schedules_runs_and_cancels_process() {
        let runner = Arc::new(RecordingRunner::default());
        let registry = process_registry();
        let endpoint = Endpoint::builder()
            .bind(LashProcessWorkflowImpl::new(runner.clone(), registry.clone()).serve())
            .build();
        let context = Arc::new(RecordingContext::with_endpoint(endpoint));
        let host = RestateRuntimeEffectController::new(context.clone());
        let registration = external_registration("task-smoke");
        let execution_context = ProcessExecutionContext::default()
            .with_wake_target_scope(lash_core::ProcessScope::new("runtime", "wake-smoke"))
            .with_tool_effect_metadata(Some(effect_metadata(
                RuntimeEffectKind::ToolCall,
                "tool-smoke",
            )));

        let outcome = host
            .execute_effect(
                RuntimeEffectEnvelope::new(
                    effect_metadata(RuntimeEffectKind::Process, "background-smoke-start"),
                    RuntimeEffectCommand::Process {
                        command: ProcessCommand::Start {
                            registration,
                            grant: Some(lash_core::ProcessStartGrant {
                                owner_scope: lash_core::ProcessScope::new("runtime", "session"),
                                descriptor: lash_core::ProcessHandleDescriptor::new(
                                    Some("tool"),
                                    Some("task-smoke"),
                                ),
                            }),
                            execution_context: Box::new(execution_context),
                        },
                    },
                ),
                RuntimeEffectLocalExecutor::process_control(registry.clone()),
            )
            .await
            .expect("start through endpoint smoke");
        let RuntimeEffectOutcome::Process {
            result: ProcessEffectOutcome::Start { record },
        } = outcome
        else {
            panic!("wrong start outcome");
        };

        let external_ref = record.external_ref.as_ref().expect("external ref");
        assert_eq!(external_ref.backend, "restate");
        assert_eq!(external_ref.id, "LashProcessWorkflow/task-smoke");
        assert_eq!(
            external_ref
                .metadata
                .as_ref()
                .and_then(|metadata| metadata.get("invocation_id")),
            Some(&serde_json::json!("invocation-task-smoke"))
        );

        let owner_scope = lash_core::ProcessScope::new("runtime", "session");
        let grants = registry
            .list_handle_grants(&owner_scope)
            .await
            .expect("session grants");
        assert_eq!(grants.len(), 1);
        assert_eq!(grants[0].0.process_id, "task-smoke");
        let granted_external_ref = grants[0].1.external_ref.as_ref().expect("grant ref");
        assert_eq!(granted_external_ref.backend, "restate");
        assert_eq!(granted_external_ref.id, "LashProcessWorkflow/task-smoke");

        assert_eq!(
            context
                .started
                .lock()
                .expect("started lock")
                .iter()
                .map(|registration| registration.id.as_str())
                .collect::<Vec<_>>(),
            vec!["task-smoke"]
        );
        assert_eq!(
            context
                .started_execution_contexts
                .lock()
                .expect("started execution contexts lock")
                .iter()
                .map(|context| {
                    context
                        .wake_target_scope
                        .as_ref()
                        .map(|scope| scope.session_id.as_str())
                })
                .collect::<Vec<_>>(),
            vec![Some("wake-smoke")]
        );
        assert_eq!(
            runner.ran.lock().expect("runner ran lock").as_slice(),
            &[RecordedProcessRun {
                process_id: "task-smoke".to_string(),
                wake_target_session_id: Some("wake-smoke".to_string()),
                tool_effect_id: Some("tool-smoke".to_string()),
            }]
        );

        let outcome = host
            .execute_effect(
                RuntimeEffectEnvelope::new(
                    effect_metadata(RuntimeEffectKind::Process, "background-smoke-cancel"),
                    RuntimeEffectCommand::Process {
                        command: ProcessCommand::Cancel {
                            process_id: "task-smoke".to_string(),
                            reason: Some("stop-smoke".to_string()),
                        },
                    },
                ),
                RuntimeEffectLocalExecutor::process_control(registry),
            )
            .await
            .expect("cancel through endpoint smoke");
        assert!(matches!(
            outcome,
            RuntimeEffectOutcome::Process {
                result: ProcessEffectOutcome::Cancel { .. }
            }
        ));
        assert_eq!(
            context.cancelled.lock().expect("cancelled lock").as_slice(),
            &[("task-smoke".to_string(), Some("stop-smoke".to_string()))]
        );
        assert_eq!(
            runner
                .cancelled
                .lock()
                .expect("runner cancelled lock")
                .as_slice(),
            &[RestateProcessCancelRequest {
                process_id: "task-smoke".to_string(),
                reason: Some("stop-smoke".to_string()),
            }]
        );
    }

    struct RecoveryProcessTool;

    impl RecoveryProcessTool {
        fn definition() -> lash_core::ToolDefinition {
            lash_core::ToolDefinition::raw(
                "tool:recovery_echo",
                "recovery_echo",
                "Echo a line and emit a durable process wake.",
                serde_json::json!({
                    "type": "object",
                    "properties": { "line": { "type": "string" } },
                    "required": ["line"],
                    "additionalProperties": false
                }),
                serde_json::json!({ "type": "object" }),
            )
        }
    }

    #[async_trait::async_trait]
    impl lash_core::ToolProvider for RecoveryProcessTool {
        fn tool_manifests(&self) -> Vec<lash_core::ToolManifest> {
            vec![Self::definition().manifest()]
        }

        fn resolve_contract(&self, name: &str) -> Option<Arc<lash_core::ToolContract>> {
            (name == "recovery_echo").then(|| Arc::new(Self::definition().contract()))
        }

        async fn execute(&self, call: lash_core::ToolCall<'_>) -> lash_core::ToolResult {
            let line = call
                .args
                .get("line")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default()
                .to_string();
            let event = lash_core::ProcessEventAppendRequest::new(
                "process.wake",
                serde_json::json!({ "message": line, "wake_input": line }),
            )
            .with_idempotency_key(format!("process.wake:{line}"));
            if let Err(err) = call.context.process_events().emit_request(event).await {
                return lash_core::ToolResult::err_fmt(err);
            }
            lash_core::ToolResult::ok(serde_json::json!({ "echo": line }))
        }
    }

    fn recovery_worker(
        registry: Arc<dyn ProcessRegistry>,
        store_factory: Arc<dyn lash_core::SessionStoreFactory>,
    ) -> DurableProcessWorker {
        let tools: Arc<dyn lash_core::ToolProvider> = Arc::new(RecoveryProcessTool);
        let plugin_host = lash_core::PluginHost::new(vec![
            Arc::new(lash_protocol_standard::StandardProtocolPluginFactory::new())
                as Arc<dyn lash_core::PluginFactory>,
            Arc::new(lash_core::plugin::StaticPluginFactory::new(
                "recovery-tool",
                lash_core::PluginSpec::new().with_tool_provider(tools),
            )),
        ]);
        DurableProcessWorker::new(
            lash_core::DurableProcessWorkerConfig::new(
                Arc::new(plugin_host),
                lash_core::RuntimeCoreConfig::default().with_host_profile_id("recovery-host"),
                store_factory,
                registry,
            )
            .with_session_policy(lash_core::SessionPolicy {
                provider: lash_core::ProviderHandle::default(),
                model: lash_core::ModelSpec::from_token_limits(
                    "mock-model",
                    None,
                    200_000,
                    None,
                    None,
                )
                .expect("model spec"),
                ..lash_core::SessionPolicy::default()
            }),
        )
    }

    fn process_wake_event_type() -> lash_core::ProcessEventType {
        lash_core::ProcessEventType {
            name: "process.wake".to_string(),
            payload_schema: lash_core::LashSchema::any(),
            semantics: lash_core::ProcessEventSemanticsSpec {
                wake: Some(lash_core::ProcessWakeSpec {
                    when: Some(lash_core::ProcessValueSelector::Present(
                        "/wake_input".to_string(),
                    )),
                    input: lash_core::ProcessValueSelector::Pointer("/wake_input".to_string()),
                    dedupe_key: lash_core::ProcessWakeDedupeKey::Selector(
                        lash_core::ProcessValueSelector::Pointer("/message".to_string()),
                    ),
                }),
                ..lash_core::ProcessEventSemanticsSpec::default()
            },
        }
    }

    #[tokio::test]
    async fn sqlite_process_recovery_reopens_registry_worker_grants_wakes_and_cancel() {
        let temp = tempfile::tempdir().expect("tempdir");
        let process_db = temp.path().join("processes.db");
        let store_factory = Arc::new(lash_sqlite_store::SqliteSessionStoreFactory::new(
            temp.path().join("sessions"),
        )) as Arc<dyn lash_core::SessionStoreFactory>;
        let registry_a = Arc::new(
            lash_sqlite_store::SqliteProcessRegistry::open(&process_db).expect("open registry"),
        ) as Arc<dyn ProcessRegistry>;
        let worker_a = recovery_worker(Arc::clone(&registry_a), Arc::clone(&store_factory));
        let endpoint_a = Endpoint::builder()
            .bind(
                LashProcessWorkflowImpl::new(
                    Arc::new(RestateCoreProcessRunner::new(worker_a)),
                    Arc::clone(&registry_a),
                )
                .serve(),
            )
            .build();
        let context_a = Arc::new(RecordingContext::with_endpoint(endpoint_a));
        let host_a = RestateRuntimeEffectController::new(context_a);
        let creator_scope = lash_core::ProcessScope::new("recovery-runtime", "root");
        let scope_id = creator_scope.id();
        let registration = ProcessRegistration::new(
            "recover-tool",
            ProcessInput::ToolCall {
                call: lash_core::PreparedToolCall {
                    call_id: "recover-call".to_string(),
                    tool_name: "recovery_echo".to_string(),
                    args: serde_json::json!({ "line": "wake-after-rebuild" }),
                    replay: None,
                    prepared_payload: serde_json::Value::Null,
                },
            },
        )
        .with_provenance(creator_scope.clone(), "recovery-host")
        .with_extra_event_types([process_wake_event_type()]);

        host_a
            .execute_effect(
                RuntimeEffectEnvelope::new(
                    effect_metadata(RuntimeEffectKind::Process, "recovery-start"),
                    RuntimeEffectCommand::Process {
                        command: ProcessCommand::Start {
                            registration,
                            grant: Some(lash_core::ProcessStartGrant {
                                owner_scope: creator_scope.clone(),
                                descriptor: lash_core::ProcessHandleDescriptor::new(
                                    Some("tool"),
                                    Some("recover-tool"),
                                ),
                            }),
                            execution_context: Box::new(
                                ProcessExecutionContext::default()
                                    .with_wake_target_scope(creator_scope.clone()),
                            ),
                        },
                    },
                ),
                RuntimeEffectLocalExecutor::process_control(Arc::clone(&registry_a)),
            )
            .await
            .expect("schedule and run process through Restate endpoint");
        drop(host_a);
        drop(registry_a);

        let registry_b = Arc::new(
            lash_sqlite_store::SqliteProcessRegistry::open(&process_db).expect("reopen registry"),
        ) as Arc<dyn ProcessRegistry>;
        let grants = registry_b
            .list_handle_grants(&creator_scope)
            .await
            .expect("list reopened grants");
        assert_eq!(grants.len(), 1);
        assert_eq!(grants[0].0.process_id, "recover-tool");
        assert_eq!(
            registry_b
                .await_process("recover-tool")
                .await
                .expect("await recovered terminal process"),
            ProcessAwaitOutput::Success {
                value: serde_json::json!({ "echo": "wake-after-rebuild" }),
                control: None,
            }
        );
        let wakes = registry_b
            .drain_wake_inputs(&scope_id, 10)
            .await
            .expect("drain reopened wakes");
        assert_eq!(wakes.len(), 1);
        assert_eq!(wakes[0].input, "wake-after-rebuild");
        registry_b
            .ack_wake_input(&wakes[0].wake_id)
            .await
            .expect("ack wake");
        assert!(
            registry_b
                .drain_wake_inputs(&scope_id, 10)
                .await
                .expect("drain after ack")
                .is_empty()
        );

        let worker_b = recovery_worker(Arc::clone(&registry_b), store_factory);
        let endpoint_b = Endpoint::builder()
            .bind(
                LashProcessWorkflowImpl::new(
                    Arc::new(RestateCoreProcessRunner::new(worker_b)),
                    Arc::clone(&registry_b),
                )
                .serve(),
            )
            .build();
        let context_b = Arc::new(RecordingContext::with_endpoint(endpoint_b));
        let host_b = RestateRuntimeEffectController::new(context_b);
        host_b
            .execute_effect(
                RuntimeEffectEnvelope::new(
                    effect_metadata(RuntimeEffectKind::Process, "recovery-cancel"),
                    RuntimeEffectCommand::Process {
                        command: ProcessCommand::Cancel {
                            process_id: "recover-tool".to_string(),
                            reason: Some("post-rebuild cancel probe".to_string()),
                        },
                    },
                ),
                RuntimeEffectLocalExecutor::process_control(Arc::clone(&registry_b)),
            )
            .await
            .expect("cancel through reopened process workflow");
        assert!(
            registry_b
                .events_after("recover-tool", 0)
                .await
                .expect("events after cancel")
                .iter()
                .any(|event| event.event_type == "process.cancel_requested")
        );
    }

    fn discover_service<S: Discoverable>(_: &S) -> restate_sdk::discovery::Service {
        S::discover()
    }

    #[tokio::test]
    async fn process_workflow_binds_to_restate_endpoint_and_discovers_handlers() {
        let runner = Arc::new(RecordingRunner::default());
        let registry = process_registry();
        let service = LashProcessWorkflowImpl::new(runner, registry).serve();
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
        let registry = process_registry();
        let workflow = LashProcessWorkflowImpl::new(runner.clone(), registry.clone());
        let registration = external_registration("task-workflow");
        registry
            .register_process(registration.clone())
            .await
            .expect("register workflow process");
        let execution_context = ProcessExecutionContext::default()
            .with_wake_target_scope(lash_core::ProcessScope::new("runtime", "wake-session"))
            .with_tool_effect_metadata(Some(effect_metadata(
                RuntimeEffectKind::ToolCall,
                "tool-effect",
            )));

        let output = workflow
            .run_registration(registration, execution_context)
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
            &[RecordedProcessRun {
                process_id: "task-workflow".to_string(),
                wake_target_session_id: Some("wake-session".to_string()),
                tool_effect_id: Some("tool-effect".to_string()),
            }]
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
