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
//! `ctx.run(...).name(envelope.invocation.replay.key)` call, sleep commands
//! map to Restate's durable timer, and process commands call Restate workflow
//! scheduling directly through idempotent registry/workflow operations. Lash's
//! own runtime journal stores the same completed effect outcome by replay key
//! and envelope hash before the restored `TurnMachine` consumes it.

use std::fmt;
use std::future::Future;
use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use lash_core::{
    DurabilityTier, DurableProcessWorker, PluginError, ProcessAwaitOutput, ProcessCommand,
    ProcessEffectOutcome, ProcessExecutionContext, ProcessExternalRef, ProcessLease,
    ProcessLeaseCompletion, ProcessRecord, ProcessRegistration, ProcessRegistry, ProcessRunHandle,
    RuntimeEffectCommand, RuntimeEffectController, RuntimeEffectControllerError,
    RuntimeEffectControllerScope, RuntimeEffectEnvelope, RuntimeEffectKind,
    RuntimeEffectLocalExecutor, RuntimeEffectOutcome, RuntimeError, RuntimeInvocation,
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

/// Short TTL for the fence lease the ingress runner holds across a single
/// `LashProcessWorkflow` submit.
///
/// The lease only fences *the submit* — Restate owns the durable execution once
/// the workflow is keyed and accepted — so it is released immediately after the
/// ingress POST. A short window keeps a runner that crashed mid-submit from
/// fencing the record for long.
const INGRESS_SUBMIT_FENCE_TTL_MS: u64 = 30_000;

/// [`ProcessRunHandle`] that drives pending processes by submitting their
/// `LashProcessWorkflow` through the Restate ingress instead of running them
/// in-process.
///
/// This is the durable tier's run handle: a [`ProcessWorkRunner`](lash_core::ProcessWorkRunner)
/// over this handle pokes/polls the registry's non-terminal rows and, per row,
/// claims a short single-owner [`ProcessLease`] to fence the submit, POSTs
/// `LashProcessWorkflow/{process_id}/run/send` to the ingress (idempotent by the
/// `lash-process:{process_id}:run` key, mirroring the in-handler
/// [`schedule_restate_process`] submit), records the durable `external_ref`, and
/// releases the lease. The submitted workflow runs on the Restate-bound
/// [`RestateCoreProcessRunner`], which claims the run-time lease and writes the
/// terminal outcome — so the single coordination point stays the
/// [`ProcessLease`] and a process runs exactly once.
pub struct RestateProcessIngressRunner {
    http: reqwest::Client,
    ingress_url: String,
    registry: Arc<dyn ProcessRegistry>,
}

impl RestateProcessIngressRunner {
    /// Build an ingress-client run handle over the given ingress base URL and
    /// process registry.
    pub fn new(ingress_url: impl Into<String>, registry: Arc<dyn ProcessRegistry>) -> Self {
        Self {
            http: reqwest::Client::new(),
            ingress_url: ingress_url.into(),
            registry,
        }
    }

    async fn submit_record(
        &self,
        owner_id: &str,
        record: ProcessRecord,
    ) -> Result<(), PluginError> {
        let process_id = record.id.clone();
        // Fence the submit: a claim conflict means another runner is already
        // submitting this process, so skip it. Treat any claim failure as
        // "fenced elsewhere" — the keyed-idempotent submit makes a missed one a
        // no-op on the next poll tick anyway.
        let Ok(lease) = self
            .registry
            .claim_process_lease(&process_id, owner_id, INGRESS_SUBMIT_FENCE_TTL_MS)
            .await
        else {
            return Ok(());
        };
        let outcome = self.submit_under_lease(record).await;
        // Release the fence lease before propagating any submit error so a
        // transient ingress failure does not pin the record until the lease
        // expires; the next poll/poke re-submits (keyed-idempotent).
        self.release_lease(&lease).await?;
        outcome
    }

    async fn submit_under_lease(&self, record: ProcessRecord) -> Result<(), PluginError> {
        let process_id = record.id.clone();
        // The record may have reached a terminal state between the list and the
        // claim. Idempotent by process_id: never re-submit a finished process.
        if self
            .registry
            .get_process(&process_id)
            .await
            .is_some_and(|current| current.is_terminal())
        {
            return Ok(());
        }
        let registration = ProcessRegistration {
            id: record.id,
            input: record.input,
            event_types: record.event_types,
            provenance: record.provenance.clone(),
        };
        // Wakes route to the creator scope; the owner scope persisted in
        // provenance is that creator scope, mirroring the inline worker sweep.
        let execution_context = ProcessExecutionContext::default()
            .with_wake_target_scope(record.provenance.owner_scope);
        let url = format!(
            "{}/LashProcessWorkflow/{}/run/send",
            self.ingress_url.trim_end_matches('/'),
            process_id
        );
        let response = self
            .http
            .post(url)
            // Idempotency key matches the in-handler submit so a re-submit (from
            // a poll racing the original poke, or another runner) coalesces.
            .header("idempotency-key", format!("lash-process:{process_id}:run"))
            .json(&RestateProcessWorkflowInput {
                registration,
                execution_context,
            })
            .send()
            .await
            .map_err(|err| {
                RestateEffectError::BackgroundScheduler(format!(
                    "ingress submit for process `{process_id}` failed: {err}"
                ))
                .into_plugin_error()
            })?;
        if !response.status().is_success() {
            return Err(RestateEffectError::BackgroundScheduler(format!(
                "ingress submit for process `{process_id}` returned status {}",
                response.status()
            ))
            .into_plugin_error());
        }
        // Record the durable backend reference so the process is observably
        // owned by Restate, mirroring `schedule_restate_process`.
        self.registry
            .set_external_ref(
                &process_id,
                ProcessExternalRef {
                    backend: "restate".to_string(),
                    id: format!("LashProcessWorkflow/{process_id}"),
                    metadata: None,
                },
            )
            .await
            .map(|_| ())
    }

    async fn release_lease(&self, lease: &ProcessLease) -> Result<(), PluginError> {
        self.registry
            .complete_process_lease(&ProcessLeaseCompletion::from_lease(lease))
            .await
    }
}

#[async_trait::async_trait]
impl ProcessRunHandle for RestateProcessIngressRunner {
    async fn claim_and_run_pending(&self) -> Result<(), PluginError> {
        let owner_id = format!("restate-ingress-{}", uuid::Uuid::new_v4());
        for record in self.registry.list_non_terminal().await? {
            self.submit_record(&owner_id, record).await?;
        }
        Ok(())
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
        metadata: RuntimeInvocation,
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
    fn durability_tier(&self) -> DurabilityTier {
        DurabilityTier::Durable
    }

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
                    tracing_sleep_error(&envelope.invocation, &err);
                    return Err(RuntimeEffectControllerError::new(
                        "restate_effect_controller",
                        err.to_string(),
                    ));
                }
                Ok(RuntimeEffectOutcome::Sleep)
            }
            RestateEffectExecution::JournaledRun => {
                let current_hash = envelope.stable_hash()?;
                let invocation = envelope.invocation.clone();
                let journal_hash = current_hash.clone();
                let journaled = self
                    .journal_effect(invocation, async move {
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
        | RuntimeEffectCommand::Direct { .. }
        | RuntimeEffectCommand::ToolCall { .. }
        | RuntimeEffectCommand::ExecCode { .. }
        | RuntimeEffectCommand::Checkpoint { .. }
        | RuntimeEffectCommand::SyncExecutionSurface { .. } => RestateEffectExecution::JournaledRun,
    }
}

fn restate_effect_name(invocation: &RuntimeInvocation) -> String {
    if let Some(replay_key) = invocation.replay_key() {
        format!("lash:{replay_key}")
    } else if let (Some(kind), Some(effect_id)) = (invocation.effect_kind(), invocation.effect_id())
    {
        format!("lash:{}:{effect_id}", kind.as_str())
    } else {
        "lash:runtime-invocation".to_string()
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

fn tracing_sleep_error(invocation: &RuntimeInvocation, err: &TerminalError) {
    tracing::warn!(
        session_id = %invocation.scope.session_id,
        effect_id = invocation.effect_id().unwrap_or(""),
        effect_kind = %RuntimeEffectKind::Sleep.as_str(),
        error = %err,
        "Restate durable sleep failed"
    );
}

#[cfg(test)]
mod tests;
