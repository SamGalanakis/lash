//! Restate durable execution adapter for Lash runtime effects.
//!
//! The primary entrypoint is [`RestateRuntimeEffectController`]. Construct it inside
//! a Restate service, object, or workflow handler, derive a stable
//! [`ScopedEffectController`](lash_core::ScopedEffectController) from an
//! [`EffectScope`](lash_core::EffectScope), and run Lash through the scoped API.
//! Restate recovery is handler replay with the same scope id and request data,
//! not Lash checkpoint reload.
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
//! #     _scope: lash_core::ScopedEffectController<'_>,
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
//!         let scoped_effect_controller = effect_controller
//!             .scoped_effect_controller(lash_core::EffectScope::turn("session", &turn_id))
//!             .map_err(TerminalError::from_error)?;
//!         let response = run_lash_turn(scoped_effect_controller, req)
//!             .await
//!             .map_err(TerminalError::from_error)?;
//!         Ok(Json(response))
//!     }
//! }
//! ```
//!
//! Restate's Rust SDK requires `ctx.run` closures to be awaited immediately and
//! not to call the Restate context from inside the closure. This adapter follows
//! that rule: every Lash effect is wrapped as one immediately awaited
//! `ctx.run(...).name(lash:<replay_key>)` call, sleep commands
//! map to Restate's durable timer, and process commands call Restate workflow
//! scheduling directly through idempotent registry/workflow operations.
//! Substrate-native Restate turns do not use store-side in-flight replay rows;
//! Lash only commits final session state through turn-commit idempotency.

use std::fmt;
use std::future::Future;
use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use lash_core::{
    DurabilityTier, DurableProcessWorker, EffectHost, EffectScope, PluginError, ProcessAwaitOutput,
    ProcessCommand, ProcessEffectOutcome, ProcessExecutionContext, ProcessExternalRef,
    ProcessLease, ProcessLeaseCompletion, ProcessRecord, ProcessRegistration, ProcessRegistry,
    ProcessRunHandle, ProcessWorkDriver, ProcessWorkPoke, ProcessWorkRunner, RuntimeEffectCommand,
    RuntimeEffectController, RuntimeEffectControllerError, RuntimeEffectEnvelope,
    RuntimeEffectKind, RuntimeEffectLocalExecutor, RuntimeEffectOutcome, RuntimeError,
    RuntimeInvocation, ScopedEffectController,
};
use restate_sdk::context::{
    Context as RestateContext, ObjectContext, RunRetryPolicy, SharedObjectContext,
    SharedWorkflowContext, WorkflowContext,
};
use restate_sdk::context::{ContextClient, ContextPromises, InvocationHandle, RequestTarget};
use restate_sdk::errors::{HandlerError, HandlerResult, TerminalError};
use restate_sdk::serde::Json;
use serde::{Serialize, de::DeserializeOwned};

pub use restate_sdk;

type RestateHandlerFuture<'a, T> =
    Pin<Box<dyn Future<Output = HandlerResult<Json<T>>> + Send + 'a>>;

#[derive(Clone, Debug, Serialize, serde::Deserialize)]
struct RecordedRuntimeEffect {
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
        scoped_effect_controller: ScopedEffectController<'_>,
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
        scoped_effect_controller: ScopedEffectController<'_>,
    ) -> Result<ProcessAwaitOutput, PluginError> {
        self.worker
            .run_process_with_scoped_effect_controller(
                registration,
                execution_context,
                scoped_effect_controller,
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

/// Deployment-level Restate effect host for long-lived Lash cores.
///
/// Restate's real effect execution requires a handler context, so this host is
/// intentionally a durable boundary, not an executor. HTTP/API code should
/// enter a Restate workflow/object first and then pass
/// [`RestateRuntimeEffectController::scoped_effect_controller`] into Lash. If a
/// caller tries to execute through this deployment host directly, it fails
/// loudly instead of falling back to inline execution.
#[derive(Clone, Default)]
pub struct RestateEffectHost {
    controller: Arc<RestateEffectHostController>,
}

impl RestateEffectHost {
    pub fn new() -> Self {
        Self::default()
    }
}

impl EffectHost for RestateEffectHost {
    fn durability_tier(&self) -> DurabilityTier {
        DurabilityTier::Durable
    }

    fn requires_durable_attachment_store(&self) -> bool {
        true
    }

    fn scoped<'run>(
        &'run self,
        scope: EffectScope,
    ) -> Result<ScopedEffectController<'run>, RuntimeError> {
        ScopedEffectController::shared(self.controller.clone(), scope)
    }

    fn scoped_static(
        &self,
        scope: EffectScope,
    ) -> Result<Option<ScopedEffectController<'static>>, RuntimeError> {
        Ok(Some(ScopedEffectController::shared(
            self.controller.clone(),
            scope,
        )?))
    }
}

#[derive(Default)]
struct RestateEffectHostController;

#[async_trait::async_trait]
impl RuntimeEffectController for RestateEffectHostController {
    fn durability_tier(&self) -> DurabilityTier {
        DurabilityTier::Durable
    }

    fn requires_durable_attachment_store(&self) -> bool {
        true
    }

    async fn execute_effect(
        &self,
        envelope: RuntimeEffectEnvelope,
        _local_executor: RuntimeEffectLocalExecutor<'_>,
    ) -> Result<RuntimeEffectOutcome, RuntimeEffectControllerError> {
        Err(RuntimeEffectControllerError::new(
            "restate_effect_host_requires_handler_scope",
            format!(
                "effect `{}` must enter a Restate handler and use RestateRuntimeEffectController::scoped_effect_controller",
                envelope
                    .invocation
                    .effect_id()
                    .unwrap_or_else(|| envelope.command.kind().as_str())
            ),
        ))
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
            env_ref: record.env_ref,
            wake_target: record.wake_target,
        };
        let execution_context = ProcessExecutionContext::default();
        let url = format!(
            "{}/LashProcessWorkflow/{}/run/send",
            self.ingress_url.trim_end_matches('/'),
            process_id
        );
        let response = self
            .http
            .post(url)
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
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(RestateEffectError::BackgroundScheduler(format!(
                "ingress submit for process `{process_id}` returned status {status}: {body}"
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

/// Bundled Restate process deployment wiring for a Lash core.
///
/// Construct this once per deployment, pass [`process_work_driver`](Self::process_work_driver)
/// into `LashCoreBuilder::process_work_driver`, bind
/// [`workflow`](Self::workflow) on the Restate endpoint, then call
/// [`spawn`](Self::spawn) after the endpoint is listening.
pub struct RestateProcessDeployment {
    process_work_runner: ProcessWorkRunner,
    driver: ProcessWorkDriver,
}

impl RestateProcessDeployment {
    pub fn new(ingress_url: impl Into<String>, registry: Arc<dyn ProcessRegistry>) -> Self {
        let ingress_runner = RestateProcessIngressRunner::new(ingress_url, Arc::clone(&registry));
        let process_work_runner = ProcessWorkRunner::new(Arc::new(ingress_runner));
        let driver = ProcessWorkDriver::new(registry, process_work_runner.poke_handle());
        Self {
            process_work_runner,
            driver,
        }
    }

    pub fn process_work_driver(&self) -> ProcessWorkDriver {
        self.driver.clone()
    }

    pub fn workflow(
        &self,
        worker: DurableProcessWorker,
    ) -> LashProcessWorkflowImpl<RestateCoreProcessRunner> {
        LashProcessWorkflowImpl::new(
            Arc::new(RestateCoreProcessRunner::new(worker)),
            self.driver.process_registry(),
        )
    }

    pub fn spawn(self) -> ProcessWorkPoke {
        self.process_work_runner.spawn()
    }
}

#[restate_sdk::workflow]
pub trait LashProcessWorkflow {
    async fn run(
        input: Json<RestateProcessWorkflowInput>,
    ) -> HandlerResult<Json<ProcessAwaitOutput>>;

    #[shared]
    async fn cancel(request: Json<RestateProcessCancelRequest>) -> HandlerResult<Json<()>>;

    #[shared]
    async fn resolve_event(
        request: Json<RestateProcessEventResolveRequest>,
    ) -> HandlerResult<Json<()>>;
}

#[derive(Clone, Debug, Serialize, serde::Deserialize)]
pub struct RestateProcessWorkflowInput {
    pub registration: ProcessRegistration,
    #[serde(default, skip_serializing_if = "ProcessExecutionContext::is_empty")]
    pub execution_context: ProcessExecutionContext,
}

#[derive(Clone, Debug, Serialize, serde::Deserialize)]
pub struct RestateProcessEventResolveRequest {
    pub process_id: String,
    pub key: String,
    pub payload: serde_json::Value,
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
        scoped_effect_controller: ScopedEffectController<'_>,
    ) -> Result<ProcessAwaitOutput, PluginError> {
        let process_id = registration.id.clone();
        let output = self
            .runner
            .run_process(registration, execution_context, scoped_effect_controller)
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
        ctx: WorkflowContext<'_>,
        Json(input): Json<RestateProcessWorkflowInput>,
    ) -> HandlerResult<Json<ProcessAwaitOutput>> {
        let process_id = input.registration.id.clone();
        let controller = RestateRuntimeEffectController::new(ctx);
        let scoped_effect_controller = controller
            .scoped_effect_controller(EffectScope::process(process_id))
            .map_err(|err| HandlerError::from(TerminalError::from_error(err)))?;
        self.run_registration(
            input.registration,
            input.execution_context,
            scoped_effect_controller,
        )
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

    async fn resolve_event(
        &self,
        ctx: SharedWorkflowContext<'_>,
        Json(request): Json<RestateProcessEventResolveRequest>,
    ) -> HandlerResult<Json<()>> {
        let payload = serde_json::to_string(&request.payload)
            .map_err(|err| HandlerError::from(TerminalError::from_error(err)))?;
        ctx.resolve_promise(&request.key, payload);
        Ok(Json(()))
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

    /// Set a Restate retry policy for recorded `ctx.run` effects.
    ///
    /// Lash provider/tool errors are recorded as Lash data, so this policy is
    /// used only when the recorded closure itself fails before producing a
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

    fn await_event<'run>(
        &'run self,
        key: String,
    ) -> Pin<Box<dyn Future<Output = Result<serde_json::Value, TerminalError>> + Send + 'run>>
    where
        'ctx: 'run;

    fn resolve_event<'run>(
        &'run self,
        request: RestateProcessEventResolveRequest,
    ) -> Pin<Box<dyn Future<Output = Result<(), TerminalError>> + Send + 'run>>
    where
        'ctx: 'run;
}

trait RestateAwaitEventContext {
    fn await_event_json<'run>(
        &'run self,
        key: String,
    ) -> Pin<Box<dyn Future<Output = Result<serde_json::Value, TerminalError>> + Send + 'run>>;
}

macro_rules! impl_unsupported_await_event_context {
    ($($context:ident),+ $(,)?) => {
        $(
            impl<'ctx> RestateAwaitEventContext for $context<'ctx> {
                fn await_event_json<'run>(
                    &'run self,
                    _key: String,
                ) -> Pin<Box<dyn Future<Output = Result<serde_json::Value, TerminalError>> + Send + 'run>> {
                    Box::pin(async move {
                        Err(TerminalError::from_error(
                            RestateEffectError::BackgroundScheduler(
                                "AwaitEvent requires a Restate workflow context".to_string(),
                            ),
                        ))
                    })
                }
            }
        )+
    };
}

macro_rules! impl_workflow_await_event_context {
    ($($context:ident),+ $(,)?) => {
        $(
            impl<'ctx> RestateAwaitEventContext for $context<'ctx> {
                fn await_event_json<'run>(
                    &'run self,
                    key: String,
                ) -> Pin<Box<dyn Future<Output = Result<serde_json::Value, TerminalError>> + Send + 'run>> {
                    Box::pin(async move {
                        let payload =
                            restate_sdk::context::ContextPromises::promise::<String>(self, &key)
                                .await?;
                        serde_json::from_str(&payload)
                            .map_err(|err| TerminalError::from_error(err))
                    })
                }
            }
        )+
    };
}

impl_unsupported_await_event_context!(RestateContext, SharedObjectContext, ObjectContext);
impl_workflow_await_event_context!(SharedWorkflowContext, WorkflowContext);

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
                    );
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
                    );
                    let call = request.call();
                    Box::pin(async move {
                        let Json(()) = call.await?;
                        Ok(())
                    })
                }

                fn await_event<'run>(
                    &'run self,
                    key: String,
                ) -> Pin<Box<dyn Future<Output = Result<serde_json::Value, TerminalError>> + Send + 'run>>
                where
                    'ctx: 'run,
                {
                    RestateAwaitEventContext::await_event_json(self, key)
                }

                fn resolve_event<'run>(
                    &'run self,
                    request: RestateProcessEventResolveRequest,
                ) -> Pin<Box<dyn Future<Output = Result<(), TerminalError>> + Send + 'run>>
                where
                    'ctx: 'run,
                {
                    let workflow_key = request.process_id.clone();
                    let request: restate_sdk::context::Request<
                        '_,
                        Json<RestateProcessEventResolveRequest>,
                        Json<()>,
                    > = ContextClient::request(
                        self,
                        RequestTarget::workflow(
                            "LashProcessWorkflow",
                            workflow_key,
                            "resolve_event",
                        ),
                        Json(request),
                    );
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

/// Lash [`RuntimeEffectController`] and [`EffectHost`] backed by a Restate handler context.
///
/// This type is intentionally handler-scoped. Create one inside the Restate
/// handler that owns the Lash operation, then pass
/// [`RestateRuntimeEffectController::scoped_effect_controller`] to Lash's
/// scoped API with a stable [`EffectScope`].
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
    pub fn scoped_effect_controller<'run>(
        &'run self,
        scope: EffectScope,
    ) -> Result<ScopedEffectController<'run>, RuntimeError> {
        ScopedEffectController::borrowed(self, scope)
    }

    async fn record_effect<'run, T, Fut>(
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

impl<'ctx, C> EffectHost for RestateRuntimeEffectController<'ctx, C>
where
    C: RestateControllerContext<'ctx>,
{
    fn durability_tier(&self) -> DurabilityTier {
        DurabilityTier::Durable
    }

    fn requires_durable_attachment_store(&self) -> bool {
        true
    }

    fn scoped<'run>(
        &'run self,
        scope: EffectScope,
    ) -> Result<ScopedEffectController<'run>, RuntimeError> {
        self.scoped_effect_controller(scope)
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
                execute_restate_process_command(&self.context, *command, local_executor)
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
            RestateEffectExecution::AwaitEvent => {
                let RuntimeEffectCommand::AwaitEvent { key } = envelope.command else {
                    unreachable!("await-event execution is only selected for event waits");
                };
                self.context
                    .await_event(key)
                    .await
                    .map(|payload| RuntimeEffectOutcome::AwaitEvent { payload })
                    .map_err(|err| {
                        RuntimeEffectControllerError::new(
                            "restate_effect_controller",
                            err.to_string(),
                        )
                    })
            }
            RestateEffectExecution::JournaledRun => {
                let current_hash = envelope.stable_hash()?;
                let invocation = envelope.invocation.clone();
                let recorded_hash = current_hash.clone();
                let recorded = self
                    .record_effect(invocation, async move {
                        let outcome = local_executor.execute(envelope).await;
                        RecordedRuntimeEffect {
                            envelope_hash: recorded_hash,
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
                validate_recorded_effect_hash(recorded, &current_hash)?
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
        ProcessCommand::List {
            session_scope,
            mode,
        } => {
            let entries = match mode {
                lash_core::ProcessListMode::Live => {
                    registry.list_live_handle_grants(&session_scope).await?
                }
                lash_core::ProcessListMode::All => {
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
        ProcessCommand::Signal {
            process_id,
            signal_name,
            request,
            ..
        } => {
            let result = registry.append_event(&process_id, request).await?;
            let ordinal = signal_ordinal_for_event(
                registry.as_ref(),
                &process_id,
                result.event.event_type.as_str(),
                result.event.sequence,
            )
            .await?;
            let key = lash_core::process_signal_wait_key(&process_id, &signal_name, ordinal);
            context
                .resolve_event(RestateProcessEventResolveRequest {
                    process_id: process_id.clone(),
                    key,
                    payload: result.event.payload.clone(),
                })
                .await
                .map_err(|err| {
                    RestateEffectError::BackgroundScheduler(err.to_string()).into_plugin_error()
                })?;
            Ok(ProcessEffectOutcome::Signal {
                event: result.event,
            })
        }
    }
}

async fn signal_ordinal_for_event(
    registry: &dyn ProcessRegistry,
    process_id: &str,
    event_type: &str,
    sequence: u64,
) -> Result<u64, PluginError> {
    // COUNT at the store, not a full log fetch: per-signal cost must stay
    // flat for long-lived processes that accumulate large event histories.
    registry
        .count_events_through(process_id, event_type, sequence)
        .await
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
            .grant_handle(&grant.session_scope, &process_id, grant.descriptor)
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
    AwaitEvent,
    JournaledRun,
}

fn restate_effect_execution(command: &RuntimeEffectCommand) -> RestateEffectExecution {
    match command {
        RuntimeEffectCommand::Process { .. } => RestateEffectExecution::DirectProcess,
        RuntimeEffectCommand::Sleep { .. } => RestateEffectExecution::Timer,
        RuntimeEffectCommand::AwaitEvent { .. } => RestateEffectExecution::AwaitEvent,
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

fn validate_recorded_effect_hash(
    recorded: RecordedRuntimeEffect,
    current_hash: &str,
) -> Result<Result<RuntimeEffectOutcome, RuntimeEffectControllerError>, RuntimeEffectControllerError>
{
    if recorded.envelope_hash != current_hash {
        return Err(RuntimeEffectControllerError::new(
            "restate_effect_hash_mismatch",
            format!(
                "recorded runtime effect hash {} did not match current envelope hash {}",
                recorded.envelope_hash, current_hash
            ),
        ));
    }
    Ok(recorded.outcome)
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
