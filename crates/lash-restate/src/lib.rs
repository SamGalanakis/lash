//! Restate durable execution adapter for Lash runtime effects.
//!
//! The primary entrypoint is [`RestateRuntimeEffectController`]. Construct it inside
//! a Restate service, object, or workflow handler, derive a stable
//! [`ScopedEffectController`](lash_core::ScopedEffectController) from an
//! [`ExecutionScope`](lash_core::ExecutionScope), and run Lash through the scoped API.
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
//!             .scoped_effect_controller(lash_core::ExecutionScope::turn("session", &turn_id))
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
    AwaitEventKey, AwaitEventWaitIdentity, DurabilityTier, DurableProcessWorker, EffectHost,
    ExecutionScope, PluginError, ProcessAwaitOutput, ProcessCommand, ProcessEffectOutcome,
    ProcessExecutionContext, ProcessExternalRef, ProcessRecord, ProcessRegistration,
    ProcessRegistry, ProcessRunHandle, ProcessWorkDriver, Resolution, ResolveOutcome,
    RuntimeEffectCommand, RuntimeEffectController, RuntimeEffectControllerError,
    RuntimeEffectEnvelope, RuntimeEffectKind, RuntimeEffectLocalExecutor, RuntimeEffectOutcome,
    RuntimeError, RuntimeInvocation, ScopedEffectController,
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

fn restate_await_event_key(
    scope: &ExecutionScope,
    wait: AwaitEventWaitIdentity,
) -> Result<AwaitEventKey, RuntimeError> {
    let key_id = serde_json::to_string(&(scope, &wait)).map_err(|err| {
        RuntimeError::new(
            "await_event_key_hash",
            format!("failed to encode Restate await-event identity: {err}"),
        )
    })?;
    Ok(AwaitEventKey {
        scope: scope.clone(),
        wait,
        key_id,
        signature: "restate-handler".to_string(),
    })
}

fn restate_await_event_process_id(key: &AwaitEventKey) -> Option<&str> {
    match &key.scope {
        ExecutionScope::Process { process_id } => Some(process_id.as_str()),
        _ => None,
    }
}

fn restate_process_terminal_await_key(process_id: &str) -> Result<AwaitEventKey, RuntimeError> {
    restate_await_event_key(
        &ExecutionScope::process(process_id.to_string()),
        AwaitEventWaitIdentity::Custom {
            key: "process_terminal".to_string(),
        },
    )
}

fn restate_process_terminal_resolution(
    output: &ProcessAwaitOutput,
) -> Result<Resolution, RuntimeError> {
    serde_json::to_value(output)
        .map(Resolution::Ok)
        .map_err(|err| RuntimeError::new("restate_process_terminal_encode", err.to_string()))
}

fn restate_process_terminal_output(
    process_id: &str,
    resolution: Resolution,
) -> Result<ProcessAwaitOutput, PluginError> {
    match resolution {
        Resolution::Ok(value) => serde_json::from_value(value).map_err(|err| {
            PluginError::Session(format!(
                "invalid terminal output for process `{process_id}`: {err}"
            ))
        }),
        Resolution::Err(err) => Ok(ProcessAwaitOutput::Failure {
            class: lash_core::ToolFailureClass::Execution,
            code: err.code,
            message: err.message,
            raw: None,
            control: None,
        }),
        Resolution::Timeout => Ok(ProcessAwaitOutput::Failure {
            class: lash_core::ToolFailureClass::Execution,
            code: "process_await_timeout".to_string(),
            message: format!("awaiting process `{process_id}` timed out"),
            raw: None,
            control: None,
        }),
        Resolution::Cancelled => Ok(ProcessAwaitOutput::Failure {
            class: lash_core::ToolFailureClass::Execution,
            code: "process_await_cancelled".to_string(),
            message: format!("awaiting process `{process_id}` was cancelled"),
            raw: None,
            control: None,
        }),
    }
}

async fn resolve_restate_process_await_event<'ctx, C>(
    context: &C,
    key: &AwaitEventKey,
    resolution: Resolution,
) -> Result<ResolveOutcome, RuntimeError>
where
    C: RestateControllerContext<'ctx> + ?Sized,
{
    let Some(process_id) = restate_await_event_process_id(key) else {
        return Ok(ResolveOutcome::UnknownOrRevoked);
    };
    context
        .resolve_event(RestateProcessEventResolveRequest {
            process_id: process_id.to_string(),
            key: key.promise_key(),
            resolution,
        })
        .await
        .map_err(|err| RuntimeError::new("restate_await_event_resolve", err.to_string()))?;
    Ok(ResolveOutcome::Accepted)
}

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

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, serde::Deserialize)]
pub struct RestateInvocationId(String);

impl RestateInvocationId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn into_string(self) -> String {
        self.0
    }
}

impl fmt::Display for RestateInvocationId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl From<String> for RestateInvocationId {
    fn from(value: String) -> Self {
        Self::new(value)
    }
}

impl From<&str> for RestateInvocationId {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum RestateHttpError {
    #[error("{operation} request failed for {url}: {source}")]
    Request {
        operation: &'static str,
        url: String,
        source: reqwest::Error,
    },
    #[error("{operation} returned status {status} for {url}: {body}")]
    Status {
        operation: &'static str,
        url: String,
        status: reqwest::StatusCode,
        body: String,
    },
    #[error("{operation} response decode failed for {url}: {source}")]
    Decode {
        operation: &'static str,
        url: String,
        source: reqwest::Error,
    },
    #[error("Restate /send returned unexpected status `{status}` for {url}")]
    UnexpectedSendStatus { url: String, status: String },
}

#[derive(Clone, Debug)]
pub struct RestateIngressClient {
    http: reqwest::Client,
    ingress_url: String,
}

impl RestateIngressClient {
    pub fn new(ingress_url: impl Into<String>) -> Self {
        Self::with_client(reqwest::Client::new(), ingress_url)
    }

    pub fn with_client(http: reqwest::Client, ingress_url: impl Into<String>) -> Self {
        Self {
            http,
            ingress_url: ingress_url.into(),
        }
    }

    pub fn ingress_url(&self) -> &str {
        &self.ingress_url
    }

    pub async fn send_json_path<T: Serialize + ?Sized>(
        &self,
        path: impl AsRef<str>,
        body: &T,
    ) -> Result<RestateInvocationId, RestateHttpError> {
        let url = format_restate_url(&self.ingress_url, path.as_ref());
        let response = self
            .http
            .post(&url)
            .json(body)
            .send()
            .await
            .map_err(|source| RestateHttpError::Request {
                operation: "Restate /send",
                url: url.clone(),
                source,
            })?;
        if !response.status().is_success() {
            return Err(status_error("Restate /send", url, response).await);
        }
        let accepted: RestateSendResponse =
            response
                .json()
                .await
                .map_err(|source| RestateHttpError::Decode {
                    operation: "Restate /send",
                    url: url.clone(),
                    source,
                })?;
        if !matches_restate_accepted_status(&accepted.status) {
            return Err(RestateHttpError::UnexpectedSendStatus {
                url,
                status: accepted.status,
            });
        }
        Ok(RestateInvocationId::new(accepted.invocation_id))
    }

    pub async fn send_workflow_json<T: Serialize + ?Sized>(
        &self,
        workflow: &str,
        workflow_key: &str,
        handler: &str,
        body: &T,
    ) -> Result<RestateInvocationId, RestateHttpError> {
        self.send_json_path(format!("{workflow}/{workflow_key}/{handler}/send"), body)
            .await
    }

    pub async fn send_service_json<T: Serialize + ?Sized>(
        &self,
        service: &str,
        handler: &str,
        body: &T,
    ) -> Result<RestateInvocationId, RestateHttpError> {
        self.send_json_path(format!("{service}/{handler}/send"), body)
            .await
    }

    pub async fn send_object_json<T: Serialize + ?Sized>(
        &self,
        object: &str,
        key: &str,
        handler: &str,
        body: &T,
    ) -> Result<RestateInvocationId, RestateHttpError> {
        self.send_json_path(format!("{object}/{key}/{handler}/send"), body)
            .await
    }
}

#[derive(Debug, serde::Deserialize)]
struct RestateSendResponse {
    #[serde(rename = "invocationId")]
    invocation_id: String,
    status: String,
}

#[derive(Clone, Debug)]
pub struct RestateAdminClient {
    http: reqwest::Client,
    admin_url: String,
}

impl RestateAdminClient {
    pub fn new(admin_url: impl Into<String>) -> Self {
        Self::with_client(reqwest::Client::new(), admin_url)
    }

    pub fn with_client(http: reqwest::Client, admin_url: impl Into<String>) -> Self {
        Self {
            http,
            admin_url: admin_url.into(),
        }
    }

    pub fn admin_url(&self) -> &str {
        &self.admin_url
    }

    pub async fn cancel_invocation(
        &self,
        invocation_id: &RestateInvocationId,
    ) -> Result<(), RestateHttpError> {
        self.patch_invocation(invocation_id, "cancel", "Restate invocation cancel")
            .await
    }

    /// Forcefully kill an invocation. This is intended for test/dev cleanup
    /// after graceful cancellation has failed.
    pub async fn kill_invocation_for_test_cleanup(
        &self,
        invocation_id: &RestateInvocationId,
    ) -> Result<(), RestateHttpError> {
        self.patch_invocation(invocation_id, "kill", "Restate invocation kill")
            .await
    }

    pub async fn invocation_status(
        &self,
        invocation_id: &RestateInvocationId,
    ) -> Result<Option<RestateInvocationStatus>, RestateHttpError> {
        let id = sql_string_literal(invocation_id.as_str());
        let mut rows = self
            .query_json::<RestateInvocationStatus>(&format!(
                "SELECT id, target, target_service_name, target_service_key, target_handler_name, status, completion_result, completion_failure FROM sys_invocation WHERE id = {id}"
            ))
            .await?;
        Ok(rows.pop())
    }

    pub async fn unfinished_invocations_for_service_prefixes(
        &self,
        prefixes: &[&str],
    ) -> Result<Vec<RestateInvocationStatus>, RestateHttpError> {
        if prefixes.is_empty() {
            return Ok(Vec::new());
        }
        let service_filter = prefixes
            .iter()
            .map(|prefix| {
                format!(
                    "target_service_name LIKE {}",
                    sql_string_literal(&format!("{prefix}%"))
                )
            })
            .collect::<Vec<_>>()
            .join(" OR ");
        self.query_json(&format!(
            "SELECT id, target, target_service_name, target_service_key, target_handler_name, status, completion_result, completion_failure FROM sys_invocation WHERE status IN ('pending', 'ready', 'running', 'backing-off', 'suspended') AND ({service_filter}) ORDER BY modified_at DESC"
        ))
        .await
    }

    pub async fn query_json<T: DeserializeOwned>(
        &self,
        query: &str,
    ) -> Result<Vec<T>, RestateHttpError> {
        #[derive(Serialize)]
        struct QueryRequest<'a> {
            query: &'a str,
        }

        #[derive(serde::Deserialize)]
        struct QueryResponse<T> {
            rows: Vec<T>,
        }

        let url = format_restate_url(&self.admin_url, "query");
        let response = self
            .http
            .post(&url)
            .header(reqwest::header::ACCEPT, "application/json")
            .json(&QueryRequest { query })
            .send()
            .await
            .map_err(|source| RestateHttpError::Request {
                operation: "Restate SQL query",
                url: url.clone(),
                source,
            })?;
        if !response.status().is_success() {
            return Err(status_error("Restate SQL query", url, response).await);
        }
        response
            .json::<QueryResponse<T>>()
            .await
            .map(|response| response.rows)
            .map_err(|source| RestateHttpError::Decode {
                operation: "Restate SQL query",
                url,
                source,
            })
    }

    async fn patch_invocation(
        &self,
        invocation_id: &RestateInvocationId,
        action: &str,
        operation: &'static str,
    ) -> Result<(), RestateHttpError> {
        let url = format_restate_url(
            &self.admin_url,
            &format!("invocations/{invocation_id}/{action}"),
        );
        let response =
            self.http
                .patch(&url)
                .send()
                .await
                .map_err(|source| RestateHttpError::Request {
                    operation,
                    url: url.clone(),
                    source,
                })?;
        if response.status().is_success() {
            Ok(())
        } else {
            Err(status_error(operation, url, response).await)
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, serde::Deserialize)]
pub struct RestateInvocationStatus {
    pub id: String,
    pub target: String,
    pub target_service_name: String,
    #[serde(default)]
    pub target_service_key: Option<String>,
    pub target_handler_name: String,
    pub status: String,
    #[serde(default)]
    pub completion_result: Option<String>,
    #[serde(default)]
    pub completion_failure: Option<String>,
}

impl RestateInvocationStatus {
    pub fn invocation_id(&self) -> RestateInvocationId {
        RestateInvocationId::new(self.id.clone())
    }

    pub fn is_still_active(&self) -> bool {
        matches!(
            self.status.as_str(),
            "pending" | "ready" | "running" | "backing-off" | "suspended"
        )
    }

    pub fn completed_successfully(&self) -> bool {
        self.status == "completed" && self.completion_result.as_deref() == Some("success")
    }
}

fn format_restate_url(base_url: &str, path: &str) -> String {
    format!(
        "{}/{}",
        base_url.trim_end_matches('/'),
        path.trim_start_matches('/')
    )
}

async fn status_error(
    operation: &'static str,
    url: String,
    response: reqwest::Response,
) -> RestateHttpError {
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    RestateHttpError::Status {
        operation,
        url,
        status,
        body,
    }
}

fn sql_string_literal(value: &str) -> String {
    format!("'{}'", value.replace('\'', "''"))
}

fn matches_restate_accepted_status(status: &str) -> bool {
    status.eq_ignore_ascii_case("accepted") || status.eq_ignore_ascii_case("previouslyaccepted")
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

    pub fn with_ingress_url(ingress_url: impl Into<String>) -> Self {
        Self {
            controller: Arc::new(RestateEffectHostController {
                await_event_ingress: Some(RestateAwaitEventIngress {
                    http: reqwest::Client::new(),
                    ingress_url: ingress_url.into(),
                }),
            }),
        }
    }
}

#[async_trait::async_trait]
impl EffectHost for RestateEffectHost {
    fn durability_tier(&self) -> DurabilityTier {
        DurabilityTier::Durable
    }

    fn requires_durable_attachment_store(&self) -> bool {
        true
    }

    fn supports_durable_effects(&self) -> bool {
        true
    }

    fn scoped<'run>(
        &'run self,
        scope: ExecutionScope,
    ) -> Result<ScopedEffectController<'run>, RuntimeError> {
        ScopedEffectController::shared(self.controller.clone(), scope)
    }

    fn scoped_static(
        &self,
        scope: ExecutionScope,
    ) -> Result<Option<ScopedEffectController<'static>>, RuntimeError> {
        Ok(Some(ScopedEffectController::shared(
            self.controller.clone(),
            scope,
        )?))
    }

    async fn await_event_key(
        &self,
        scope: &ExecutionScope,
        wait: AwaitEventWaitIdentity,
    ) -> Result<AwaitEventKey, RuntimeError> {
        restate_await_event_key(scope, wait)
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
        _key: &AwaitEventKey,
        _cancel: tokio_util::sync::CancellationToken,
        _deadline: Option<std::time::Instant>,
    ) -> Result<Resolution, RuntimeError> {
        Err(RuntimeError::new(
            "restate_await_event_requires_handler",
            "Restate await events require a workflow handler context",
        ))
    }

    async fn revoke_await_events_for_session(&self, _session_id: &str) -> Result<(), RuntimeError> {
        Ok(())
    }
}

#[derive(Clone)]
struct RestateAwaitEventIngress {
    http: reqwest::Client,
    ingress_url: String,
}

async fn resolve_restate_process_await_event_via_ingress(
    ingress: &RestateAwaitEventIngress,
    key: &AwaitEventKey,
    resolution: Resolution,
) -> Result<ResolveOutcome, RuntimeError> {
    let Some(process_id) = restate_await_event_process_id(key) else {
        return Ok(ResolveOutcome::UnknownOrRevoked);
    };
    let url = format!(
        "{}/LashProcessWorkflow/{}/resolve_event",
        ingress.ingress_url.trim_end_matches('/'),
        process_id
    );
    let response = ingress
        .http
        .post(url)
        .json(&RestateProcessEventResolveRequest {
            process_id: process_id.to_string(),
            key: key.promise_key(),
            resolution,
        })
        .send()
        .await
        .map_err(|err| RuntimeError::new("restate_await_event_resolve", err.to_string()))?;
    if response.status().is_success() {
        return Ok(ResolveOutcome::Accepted);
    }
    if response.status() == reqwest::StatusCode::NOT_FOUND
        || response.status() == reqwest::StatusCode::GONE
    {
        return Ok(ResolveOutcome::UnknownOrRevoked);
    }
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    Err(RuntimeError::new(
        "restate_await_event_resolve",
        format!("Restate await-event resolve returned status {status}: {body}"),
    ))
}

#[derive(Default)]
struct RestateEffectHostController {
    await_event_ingress: Option<RestateAwaitEventIngress>,
}

#[async_trait::async_trait]
impl RuntimeEffectController for RestateEffectHostController {
    fn durability_tier(&self) -> DurabilityTier {
        DurabilityTier::Durable
    }

    fn requires_durable_attachment_store(&self) -> bool {
        true
    }

    fn supports_concurrent_effects(&self) -> bool {
        false
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

    async fn resolve_await_event(
        &self,
        key: &AwaitEventKey,
        resolution: Resolution,
    ) -> Result<ResolveOutcome, RuntimeError> {
        match &self.await_event_ingress {
            Some(ingress) => {
                resolve_restate_process_await_event_via_ingress(ingress, key, resolution).await
            }
            None => Ok(ResolveOutcome::UnknownOrRevoked),
        }
    }
}

/// [`ProcessRunHandle`] that drives pending processes by submitting their
/// `LashProcessWorkflow` through the Restate ingress instead of running them
/// in-process.
///
/// This is the durable tier's process work handle: a host-owned
/// [`ProcessWorkDriver`] calls it on ingress-relevant events. Per row, it POSTs
/// `LashProcessWorkflow/{process_id}/run/send` to the ingress. Restate
/// coalesces by workflow key, so duplicate submits are idempotent and no Lash
/// registry lease is needed at the Restate tier.
pub struct RestateProcessIngressRunner {
    ingress: RestateIngressClient,
    registry: Arc<dyn ProcessRegistry>,
}

impl RestateProcessIngressRunner {
    /// Build an ingress-client run handle over the given ingress base URL and
    /// process registry.
    pub fn new(ingress_url: impl Into<String>, registry: Arc<dyn ProcessRegistry>) -> Self {
        Self {
            ingress: RestateIngressClient::new(ingress_url),
            registry,
        }
    }

    async fn submit_record(&self, record: ProcessRecord) -> Result<(), PluginError> {
        let process_id = record.id.clone();
        // The record may have reached a terminal state between the list and the submit.
        // Idempotent by process_id: never re-submit a finished process.
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
            identity: record.identity,
            event_types: record.event_types,
            provenance: record.provenance.clone(),
            env_ref: record.env_ref,
            wake_target: record.wake_target,
        };
        let execution_context = ProcessExecutionContext::default();
        let invocation_id = self
            .ingress
            .send_workflow_json(
                "LashProcessWorkflow",
                &process_id,
                "run",
                &RestateProcessWorkflowInput {
                    registration,
                    execution_context,
                },
            )
            .await
            .map_err(|err| {
                RestateEffectError::BackgroundScheduler(format!(
                    "ingress submit for process `{process_id}` failed: {err}"
                ))
                .into_plugin_error()
            })?;
        // Record the durable backend reference so the process is observably
        // owned by Restate, mirroring `schedule_restate_process`.
        self.registry
            .set_external_ref(
                &process_id,
                ProcessExternalRef {
                    backend: "restate".to_string(),
                    id: format!("LashProcessWorkflow/{process_id}"),
                    metadata: Some(serde_json::json!({ "invocation_id": invocation_id })),
                },
            )
            .await
            .map(|_| ())
    }
}

#[async_trait::async_trait]
impl ProcessRunHandle for RestateProcessIngressRunner {
    async fn claim_and_run_pending(&self) -> Result<(), PluginError> {
        for record in self.registry.list_non_terminal().await? {
            self.submit_record(record).await?;
        }
        Ok(())
    }
}

/// Bundled Restate process deployment wiring for a Lash core.
///
/// Construct this once per deployment, pass [`process_work_driver`](Self::process_work_driver)
/// into `LashCoreBuilder::process_work_driver`, and bind
/// [`workflow`](Self::workflow) on the Restate endpoint.
pub struct RestateProcessDeployment {
    driver: ProcessWorkDriver,
}

impl RestateProcessDeployment {
    pub fn new(ingress_url: impl Into<String>, registry: Arc<dyn ProcessRegistry>) -> Self {
        let ingress_runner = RestateProcessIngressRunner::new(ingress_url, Arc::clone(&registry));
        let driver = ProcessWorkDriver::new(registry, Arc::new(ingress_runner));
        Self { driver }
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
}

#[restate_sdk::workflow]
pub trait LashProcessWorkflow {
    async fn run(
        input: Json<RestateProcessWorkflowInput>,
    ) -> HandlerResult<Json<ProcessAwaitOutput>>;

    #[shared]
    async fn await_terminal(
        request: Json<RestateProcessAwaitRequest>,
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
    pub resolution: Resolution,
}

#[derive(Clone, Debug, Serialize, serde::Deserialize)]
pub struct RestateProcessAwaitRequest {
    pub process_id: String,
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
                let _ = self
                    .registry
                    .complete_process(&process_id, output.clone())
                    .await;
                tracing::warn!(
                    process_id = %process_id,
                    error = %err,
                    "Restate process runner failed; completed process with failure output",
                );
                Ok(output)
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
            .scoped_effect_controller(ExecutionScope::process(process_id.clone()))
            .map_err(|err| HandlerError::from(TerminalError::from_error(err)))?;
        let output = self
            .run_registration(
                input.registration,
                input.execution_context,
                scoped_effect_controller,
            )
            .await
            .map_err(|err| HandlerError::from(TerminalError::from_error(err)))?;
        let key = restate_process_terminal_await_key(&process_id)
            .map_err(|err| HandlerError::from(TerminalError::from_error(err)))?;
        let resolution = restate_process_terminal_resolution(&output)
            .map_err(|err| HandlerError::from(TerminalError::from_error(err)))?;
        let payload = serde_json::to_string(&resolution)
            .map_err(|err| HandlerError::from(TerminalError::from_error(err)))?;
        let promise_key = key.promise_key();
        controller.context().resolve_promise(&promise_key, payload);
        Ok(Json(output))
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

    async fn await_terminal(
        &self,
        ctx: SharedWorkflowContext<'_>,
        Json(request): Json<RestateProcessAwaitRequest>,
    ) -> HandlerResult<Json<ProcessAwaitOutput>> {
        let key = restate_process_terminal_await_key(&request.process_id)
            .map_err(|err| HandlerError::from(TerminalError::from_error(err)))?;
        let promise_key = key.promise_key();
        let payload = ctx.promise::<String>(&promise_key).await?;
        let resolution = serde_json::from_str(&payload)
            .map_err(|err| HandlerError::from(TerminalError::from_error(err)))?;
        let output = restate_process_terminal_output(&request.process_id, resolution)
            .map_err(|err| HandlerError::from(TerminalError::from_error(err)))?;
        Ok(Json(output))
    }

    async fn resolve_event(
        &self,
        ctx: SharedWorkflowContext<'_>,
        Json(request): Json<RestateProcessEventResolveRequest>,
    ) -> HandlerResult<Json<()>> {
        let payload = serde_json::to_string(&request.resolution)
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
    ) -> Pin<Box<dyn Future<Output = Result<Resolution, TerminalError>> + Send + 'run>>
    where
        'ctx: 'run;

    fn await_process_terminal<'run>(
        &'run self,
        process_id: String,
    ) -> Pin<Box<dyn Future<Output = Result<ProcessAwaitOutput, TerminalError>> + Send + 'run>>
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
    ) -> Pin<Box<dyn Future<Output = Result<Resolution, TerminalError>> + Send + 'run>>;
}

macro_rules! impl_unsupported_await_event_context {
    ($($context:ident),+ $(,)?) => {
        $(
            impl<'ctx> RestateAwaitEventContext for $context<'ctx> {
                fn await_event_json<'run>(
                    &'run self,
                    _key: String,
                ) -> Pin<Box<dyn Future<Output = Result<Resolution, TerminalError>> + Send + 'run>> {
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
                ) -> Pin<Box<dyn Future<Output = Result<Resolution, TerminalError>> + Send + 'run>> {
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
                    Box::pin(async move {
                        let run = restate_sdk::context::ContextSideEffects::run(self, move || async move {
                            Ok::<Json<T>, HandlerError>(Json(future.await))
                        });
                        let run = restate_sdk::context::RunFuture::name(run, effect_name);
                        let run = match retry_policy {
                            Some(policy) => restate_sdk::context::RunFuture::retry_policy(run, policy),
                            None => run,
                        };
                        run.await
                    })
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
                ) -> Pin<Box<dyn Future<Output = Result<Resolution, TerminalError>> + Send + 'run>>
                where
                    'ctx: 'run,
                {
                    RestateAwaitEventContext::await_event_json(self, key)
                }

                fn await_process_terminal<'run>(
                    &'run self,
                    process_id: String,
                ) -> Pin<Box<dyn Future<Output = Result<ProcessAwaitOutput, TerminalError>> + Send + 'run>>
                where
                    'ctx: 'run,
                {
                    let request: restate_sdk::context::Request<
                        '_,
                        Json<RestateProcessAwaitRequest>,
                        Json<ProcessAwaitOutput>,
                    > = ContextClient::request(
                        self,
                        RequestTarget::workflow(
                            "LashProcessWorkflow",
                            process_id.clone(),
                            "await_terminal",
                        ),
                        Json(RestateProcessAwaitRequest { process_id }),
                    );
                    let call = request.call();
                    Box::pin(async move {
                        let Json(output) = call.await?;
                        Ok(output)
                    })
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
/// scoped API with a stable [`ExecutionScope`].
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
        scope: ExecutionScope,
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

#[async_trait::async_trait]
impl<'ctx, C> EffectHost for RestateRuntimeEffectController<'ctx, C>
where
    C: RestateControllerContext<'ctx> + Sync,
{
    fn durability_tier(&self) -> DurabilityTier {
        DurabilityTier::Durable
    }

    fn requires_durable_attachment_store(&self) -> bool {
        true
    }

    fn scoped<'run>(
        &'run self,
        scope: ExecutionScope,
    ) -> Result<ScopedEffectController<'run>, RuntimeError> {
        self.scoped_effect_controller(scope)
    }

    async fn await_event_key(
        &self,
        scope: &ExecutionScope,
        wait: AwaitEventWaitIdentity,
    ) -> Result<AwaitEventKey, RuntimeError> {
        restate_await_event_key(scope, wait)
    }

    async fn resolve_await_event(
        &self,
        key: &AwaitEventKey,
        resolution: Resolution,
    ) -> Result<ResolveOutcome, RuntimeError> {
        resolve_restate_process_await_event(&self.context, key, resolution).await
    }

    async fn await_await_event(
        &self,
        key: &AwaitEventKey,
        _cancel: tokio_util::sync::CancellationToken,
        _deadline: Option<std::time::Instant>,
    ) -> Result<Resolution, RuntimeError> {
        self.context
            .await_event(key.promise_key())
            .await
            .map_err(|err| RuntimeError::new("restate_effect_controller", err.to_string()))
    }

    async fn revoke_await_events_for_session(&self, _session_id: &str) -> Result<(), RuntimeError> {
        Ok(())
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

    fn supports_durable_effects(&self) -> bool {
        true
    }

    fn supports_concurrent_effects(&self) -> bool {
        false
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
            RestateEffectExecution::DirectLocal => local_executor.execute(envelope).await,
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
                    .await_event(key.promise_key())
                    .await
                    .map(|resolution| RuntimeEffectOutcome::AwaitEvent { resolution })
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

    async fn await_event_key(
        &self,
        scope: &ExecutionScope,
        wait: AwaitEventWaitIdentity,
    ) -> Result<AwaitEventKey, RuntimeError> {
        restate_await_event_key(scope, wait)
    }

    async fn resolve_await_event(
        &self,
        key: &AwaitEventKey,
        resolution: Resolution,
    ) -> Result<ResolveOutcome, RuntimeError> {
        resolve_restate_process_await_event(&self.context, key, resolution).await
    }

    async fn await_await_event(
        &self,
        key: &AwaitEventKey,
        _cancel: tokio_util::sync::CancellationToken,
        _deadline: Option<std::time::Instant>,
    ) -> Result<Resolution, RuntimeError> {
        self.context
            .await_event(key.promise_key())
            .await
            .map_err(|err| RuntimeError::new("restate_effect_controller", err.to_string()))
    }

    async fn revoke_await_events_for_session(&self, _session_id: &str) -> Result<(), RuntimeError> {
        Ok(())
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
            if registry.get_process(&process_id).await.is_none() {
                return Err(PluginError::Session(format!("unknown process `{process_id}`")).into());
            }
            let output = context
                .await_process_terminal(process_id.clone())
                .await
                .map_err(|err| {
                    RuntimeEffectControllerError::new("restate_process_await", err.to_string())
                })?;
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
                    resolution: Resolution::Ok(result.event.payload.clone()),
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
    DirectLocal,
    Timer,
    AwaitEvent,
    JournaledRun,
}

fn restate_effect_execution(command: &RuntimeEffectCommand) -> RestateEffectExecution {
    match command {
        RuntimeEffectCommand::Process { .. } => RestateEffectExecution::DirectProcess,
        RuntimeEffectCommand::ToolBatch { .. } => RestateEffectExecution::DirectLocal,
        RuntimeEffectCommand::Sleep { .. } => RestateEffectExecution::Timer,
        RuntimeEffectCommand::AwaitEvent { .. } => RestateEffectExecution::AwaitEvent,
        RuntimeEffectCommand::LlmCall { .. }
        | RuntimeEffectCommand::Direct { .. }
        | RuntimeEffectCommand::ToolAttempt { .. }
        | RuntimeEffectCommand::ExecCode { .. }
        | RuntimeEffectCommand::Checkpoint { .. }
        | RuntimeEffectCommand::SyncExecutionEnvironment { .. }
        | RuntimeEffectCommand::DurableStep { .. } => RestateEffectExecution::JournaledRun,
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
