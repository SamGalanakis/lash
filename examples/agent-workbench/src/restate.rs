use std::collections::BTreeSet;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use chrono::{DateTime, Utc};
use chrono_tz::Tz;
use croner::parser::{CronParser, Seconds};
use lash::TurnInput;
use lash_restate::LashProcessWorkflow;
use restate_sdk::context::{
    ContextClient, ContextReadState, ContextSideEffects, ContextWriteState, InvocationHandle,
    RunFuture,
};
use restate_sdk::errors::{HandlerError, HandlerResult, TerminalError};
use restate_sdk::prelude::{Endpoint, ObjectContext, SharedObjectContext, WorkflowContext};
use restate_sdk::serde::Json;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::{
    AppError, AppState, CRON_SCHEDULE_SOURCE_TYPE, ChannelTurnEvents, ModelSelection,
    TurnStreamState, apply_model_selection_to_session, assistant_text_for_display,
    model_spec_from_selection,
};

const CRON_STATE_KEY: &str = "state";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct WorkbenchTurnWorkflowRequest {
    pub turn_id: String,
    pub session_id: String,
    pub text: String,
    pub model: ModelSelection,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct WorkbenchQueuedTurnWorkflowRequest {
    pub turn_id: String,
    pub session_id: String,
    pub reason: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct WorkbenchCronRequest {
    session_id: String,
    trigger_handle: String,
    expr: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tz: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    name: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct WorkbenchCronState {
    request: WorkbenchCronRequest,
    next_execution_time: String,
    next_execution_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    last_fired_at: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct WorkbenchCronInfo {
    trigger_handle: String,
    expr: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tz: Option<String>,
    next_execution_time: String,
    next_execution_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    last_fired_at: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct CronActivationReport {
    started_process_ids: Vec<String>,
}

impl From<&WorkbenchCronState> for WorkbenchCronInfo {
    fn from(state: &WorkbenchCronState) -> Self {
        Self {
            trigger_handle: state.request.trigger_handle.clone(),
            expr: state.request.expr.clone(),
            tz: state.request.tz.clone(),
            next_execution_time: state.next_execution_time.clone(),
            next_execution_id: state.next_execution_id.clone(),
            last_fired_at: state.last_fired_at.clone(),
        }
    }
}

#[restate_sdk::workflow]
pub(crate) trait WorkbenchTurnWorkflow {
    async fn run(request: Json<WorkbenchTurnWorkflowRequest>) -> HandlerResult<Json<()>>;
}

pub(crate) struct WorkbenchTurnWorkflowImpl {
    state: AppState,
}

impl WorkbenchTurnWorkflowImpl {
    pub(crate) fn new(state: AppState) -> Self {
        Self { state }
    }
}

impl WorkbenchTurnWorkflow for WorkbenchTurnWorkflowImpl {
    async fn run(
        &self,
        ctx: WorkflowContext<'_>,
        Json(request): Json<WorkbenchTurnWorkflowRequest>,
    ) -> HandlerResult<Json<()>> {
        let controller = lash_restate::RestateRuntimeEffectController::new(ctx);
        run_user_turn(self.state.clone(), request, &controller)
            .await
            .map_err(terminal_handler_error)?;
        sync_cron_jobs_with_context(&self.state, controller.context(), "user_turn").await?;
        self.state.queue_runner.poke("user_turn_completed");
        Ok(Json(()))
    }
}

#[restate_sdk::workflow]
pub(crate) trait WorkbenchQueuedTurnWorkflow {
    async fn run(request: Json<WorkbenchQueuedTurnWorkflowRequest>) -> HandlerResult<Json<()>>;
}

pub(crate) struct WorkbenchQueuedTurnWorkflowImpl {
    state: AppState,
}

impl WorkbenchQueuedTurnWorkflowImpl {
    pub(crate) fn new(state: AppState) -> Self {
        Self { state }
    }
}

impl WorkbenchQueuedTurnWorkflow for WorkbenchQueuedTurnWorkflowImpl {
    async fn run(
        &self,
        ctx: WorkflowContext<'_>,
        Json(request): Json<WorkbenchQueuedTurnWorkflowRequest>,
    ) -> HandlerResult<Json<()>> {
        let controller = lash_restate::RestateRuntimeEffectController::new(ctx);
        let outcome = run_queued_turn(self.state.clone(), request, &controller)
            .await
            .map_err(terminal_handler_error);
        self.state.set_queued_turn_inflight(false);
        outcome?;
        sync_cron_jobs_with_context(&self.state, controller.context(), "queued_turn").await?;
        self.state.queue_runner.poke("queued_turn_completed");
        Ok(Json(()))
    }
}

#[restate_sdk::object]
trait WorkbenchCronJob {
    async fn upsert(request: Json<WorkbenchCronRequest>) -> HandlerResult<Json<WorkbenchCronInfo>>;
    async fn run() -> HandlerResult<Json<()>>;
    async fn cancel() -> HandlerResult<Json<()>>;
    #[shared]
    async fn info() -> HandlerResult<Json<Option<WorkbenchCronInfo>>>;
}

pub(crate) struct WorkbenchCronJobImpl {
    state: AppState,
}

impl WorkbenchCronJobImpl {
    pub(crate) fn new(state: AppState) -> Self {
        Self { state }
    }
}

impl WorkbenchCronJob for WorkbenchCronJobImpl {
    async fn upsert(
        &self,
        ctx: ObjectContext<'_>,
        Json(request): Json<WorkbenchCronRequest>,
    ) -> HandlerResult<Json<WorkbenchCronInfo>> {
        if let Some(Json(existing)) = ctx.get::<Json<WorkbenchCronState>>(CRON_STATE_KEY).await?
            && existing.request == request
        {
            return Ok(Json(WorkbenchCronInfo::from(&existing)));
        }
        cancel_stored_execution(&ctx).await?;
        let now = journaled_now(&ctx, "workbench-cron:upsert-now").await?;
        let state = schedule_next(&ctx, request, now, None).await?;
        Ok(Json(WorkbenchCronInfo::from(&state)))
    }

    async fn run(&self, ctx: ObjectContext<'_>) -> HandlerResult<Json<()>> {
        let Some(Json(state)) = ctx.get::<Json<WorkbenchCronState>>(CRON_STATE_KEY).await? else {
            return Ok(Json(()));
        };
        let fired_at = journaled_now(&ctx, "workbench-cron:fired-at").await?;
        let request = state.request.clone();
        let app_state = self.state.clone();
        let fired_at_text = fired_at.to_rfc3339();
        let Json(activation) =
            ctx.run(move || async move {
                activate_cron_trigger(app_state, request, fired_at_text).await
            })
            .name("workbench-cron:activate")
            .await?;
        self.state.trace(
            "cron.restate.run",
            json!({
                "job_key": ctx.key(),
                "trigger_handle": state.request.trigger_handle,
                "expr": state.request.expr,
                "tz": state.request.tz,
                "fired_at": fired_at.to_rfc3339(),
                "started_process_ids": activation.started_process_ids,
            }),
        );
        self.state.queue_runner.poke("cron_tick");
        schedule_next(&ctx, state.request, fired_at, Some(fired_at.to_rfc3339())).await?;
        Ok(Json(()))
    }

    async fn cancel(&self, ctx: ObjectContext<'_>) -> HandlerResult<Json<()>> {
        cancel_stored_execution(&ctx).await?;
        ctx.clear(CRON_STATE_KEY);
        Ok(Json(()))
    }

    async fn info(
        &self,
        ctx: SharedObjectContext<'_>,
    ) -> HandlerResult<Json<Option<WorkbenchCronInfo>>> {
        Ok(Json(
            ctx.get::<Json<WorkbenchCronState>>(CRON_STATE_KEY)
                .await?
                .as_ref()
                .map(|Json(state)| WorkbenchCronInfo::from(state)),
        ))
    }
}

pub(crate) fn spawn_restate_endpoint(
    addr: SocketAddr,
    state: AppState,
    process_registry: Arc<dyn lash::advanced::ProcessRegistry>,
    process_worker: lash::advanced::DurableProcessWorker,
) {
    let process_runner = Arc::new(lash_restate::RestateCoreProcessRunner::new(process_worker));
    let endpoint = Endpoint::builder()
        .bind(WorkbenchTurnWorkflowImpl::new(state.clone()).serve())
        .bind(WorkbenchQueuedTurnWorkflowImpl::new(state.clone()).serve())
        .bind(WorkbenchCronJobImpl::new(state).serve())
        .bind(lash_restate::LashProcessWorkflowImpl::new(process_runner, process_registry).serve())
        .build();
    tokio::spawn(async move {
        restate_sdk::http_server::HttpServer::new(endpoint)
            .listen_and_serve(addr)
            .await;
    });
}

pub(crate) async fn submit_user_turn(
    state: &AppState,
    request: WorkbenchTurnWorkflowRequest,
) -> Result<(), AppError> {
    let url = format!(
        "{}/WorkbenchTurnWorkflow/{}/run/send",
        state.restate_ingress_url.trim_end_matches('/'),
        request.turn_id
    );
    submit_restate_json(state, url, &request).await
}

pub(crate) async fn submit_queued_turn(state: &AppState, reason: &str) -> Result<(), AppError> {
    let turn_id = format!("workbench-queued-{}", uuid::Uuid::new_v4());
    let request = WorkbenchQueuedTurnWorkflowRequest {
        turn_id: turn_id.clone(),
        session_id: state.current_session_id(),
        reason: reason.to_string(),
    };
    let url = format!(
        "{}/WorkbenchQueuedTurnWorkflow/{turn_id}/run/send",
        state.restate_ingress_url.trim_end_matches('/')
    );
    submit_restate_json(state, url, &request).await
}

pub(crate) async fn cancel_known_cron_jobs(state: &AppState, reason: &str) -> Result<(), AppError> {
    let known = {
        let mut guard = state
            .restate_cron_job_keys
            .lock()
            .expect("restate cron job key lock");
        let known = guard.clone();
        guard.clear();
        known
    };
    for job_key in known {
        let url = format!(
            "{}/WorkbenchCronJob/{job_key}/cancel",
            state.restate_ingress_url.trim_end_matches('/')
        );
        state.trace(
            "cron.restate.cancel",
            json!({
                "job_key": job_key,
                "reason": reason,
            }),
        );
        submit_restate_empty(state, url).await?;
    }
    Ok(())
}

async fn run_user_turn(
    state: AppState,
    request: WorkbenchTurnWorkflowRequest,
    controller: &lash_restate::RestateRuntimeEffectController<'_, WorkflowContext<'_>>,
) -> Result<(), AppError> {
    let turn_model = model_spec_from_selection(request.model);
    let session = state
        .core
        .session(request.session_id.clone())
        .rlm()
        .open()
        .await
        .map_err(AppError::internal)?;
    apply_model_selection_to_session(&state, &session, turn_model.clone(), "restate_user_turn")
        .await?;
    let turn_state = Arc::new(Mutex::new(TurnStreamState::default()));
    let ui_events = ChannelTurnEvents {
        state: state.clone(),
        turn_state: Arc::clone(&turn_state),
    };
    let durable_turn_scope = controller
        .durable_turn_scope(&request.turn_id)
        .map_err(AppError::internal)?;
    let mut input = TurnInput::text(request.text.clone());
    input.trace_turn_id = Some(request.turn_id.clone());
    let output = session
        .turn(input)
        .model(turn_model)
        .require_submit()
        .map_err(AppError::internal)?
        .stream_with_durable_turn(&ui_events, durable_turn_scope)
        .await
        .map_err(AppError::internal)?;
    record_turn_output(&state, output, turn_state, "restate_user_turn.completed");
    Ok(())
}

async fn run_queued_turn(
    state: AppState,
    request: WorkbenchQueuedTurnWorkflowRequest,
    controller: &lash_restate::RestateRuntimeEffectController<'_, WorkflowContext<'_>>,
) -> Result<(), AppError> {
    let session = state
        .core
        .session(request.session_id.clone())
        .rlm()
        .open()
        .await
        .map_err(AppError::internal)?;
    let selected_model = model_spec_from_selection(state.selected_model());
    session
        .configure(lash::SessionConfigPatch {
            model: Some(selected_model.clone()),
            ..lash::SessionConfigPatch::default()
        })
        .await
        .map_err(AppError::internal)?;
    let turn_state = Arc::new(Mutex::new(TurnStreamState::default()));
    let ui_events = ChannelTurnEvents {
        state: state.clone(),
        turn_state: Arc::clone(&turn_state),
    };
    state.trace(
        "queue_runner.restate.start",
        json!({
            "reason": request.reason,
            "session_id": request.session_id,
            "turn_id": request.turn_id,
            "model": serde_json::to_value(&selected_model).unwrap_or(Value::Null),
        }),
    );
    let durable_turn_scope = controller
        .durable_turn_scope(&request.turn_id)
        .map_err(AppError::internal)?;
    let Some(output) = session
        .next_queued_turn()
        .stream_with_durable_turn(&ui_events, durable_turn_scope)
        .await
        .map_err(AppError::internal)?
    else {
        state.publish(crate::StreamItem::Done);
        return Ok(());
    };
    record_turn_output(&state, output, turn_state, "restate_queued_turn.completed");
    Ok(())
}

fn record_turn_output(
    state: &AppState,
    output: lash::TurnResult,
    turn_state: Arc<Mutex<TurnStreamState>>,
    trace_name: &str,
) {
    let streamed_prose = turn_state
        .lock()
        .expect("turn state lock")
        .assistant_prose
        .clone();
    let assistant_text = assistant_text_for_display(&output, &streamed_prose);
    state.trace(
        trace_name,
        json!({
            "assistant_text": assistant_text.clone(),
            "streamed_prose": streamed_prose,
            "submitted_value": output.submitted_value().cloned(),
            "tool_value": output.tool_value().map(|(tool_name, value)| {
                json!({
                    "tool_name": tool_name,
                    "value": value,
                })
            }),
        }),
    );
    state.push_message("assistant", assistant_text);
    state.publish(crate::StreamItem::Done);
}

async fn sync_cron_jobs_with_context(
    state: &AppState,
    ctx: &WorkflowContext<'_>,
    reason: &str,
) -> HandlerResult<()> {
    let session = state
        .core
        .session(state.current_session_id())
        .rlm()
        .open()
        .await
        .map_err(|err| TerminalError::new(err.to_string()))?;
    let registrations = session
        .triggers()
        .by_source_type(CRON_SCHEDULE_SOURCE_TYPE)
        .await
        .map_err(|err| TerminalError::new(err.to_string()))?;
    let previous = state
        .restate_cron_job_keys
        .lock()
        .expect("restate cron job key lock")
        .clone();
    let mut active = BTreeSet::new();
    for registration in registrations {
        if !registration.enabled {
            continue;
        }
        let (job_key, request) =
            match cron_request_from_registration(&state.current_session_id(), &registration) {
                Ok(value) => value,
                Err(err) => {
                    state.trace(
                        "cron.restate.sync_invalid",
                        json!({
                            "reason": reason,
                            "handle": registration.handle,
                            "error": err,
                        }),
                    );
                    continue;
                }
            };
        let Json(info) = ctx
            .object_client::<WorkbenchCronJobClient>(job_key.clone())
            .upsert(Json(request))
            .call()
            .await?;
        state.trace(
            "cron.restate.sync_upserted",
            json!({
                "reason": reason,
                "job_key": job_key,
                "next_execution_time": info.next_execution_time,
                "next_execution_id": info.next_execution_id,
            }),
        );
        active.insert(job_key);
    }
    for stale in previous.difference(&active) {
        ctx.object_client::<WorkbenchCronJobClient>(stale.clone())
            .cancel()
            .call()
            .await?;
        state.trace(
            "cron.restate.sync_cancelled",
            json!({
                "reason": reason,
                "job_key": stale,
            }),
        );
    }
    *state
        .restate_cron_job_keys
        .lock()
        .expect("restate cron job key lock") = active;
    Ok(())
}

async fn activate_cron_trigger(
    state: AppState,
    request: WorkbenchCronRequest,
    fired_at: String,
) -> HandlerResult<Json<CronActivationReport>> {
    let session = state
        .core
        .session(request.session_id.clone())
        .rlm()
        .open()
        .await
        .map_err(|err| TerminalError::new(err.to_string()))?;
    let report = session
        .triggers()
        .activate(
            request.trigger_handle.clone(),
            json!({
                "fired_at": fired_at,
            }),
        )
        .await
        .map_err(|err| HandlerError::from(TerminalError::new(err.to_string())))?;
    Ok(Json(CronActivationReport {
        started_process_ids: report.started_process_ids,
    }))
}

async fn schedule_next(
    ctx: &ObjectContext<'_>,
    request: WorkbenchCronRequest,
    now: DateTime<Utc>,
    last_fired_at: Option<String>,
) -> HandlerResult<WorkbenchCronState> {
    let next = next_cron_time(&request.expr, request.tz.as_deref(), now)
        .map_err(|err| HandlerError::from(TerminalError::new(err)))?;
    let delay = next
        .signed_duration_since(now)
        .to_std()
        .unwrap_or_else(|_| Duration::from_secs(0));
    let handle = ctx
        .object_client::<WorkbenchCronJobClient>(ctx.key())
        .run()
        .send_after(delay);
    let next_execution_id = handle.invocation_id().await?;
    let state = WorkbenchCronState {
        request,
        next_execution_time: next.to_rfc3339(),
        next_execution_id,
        last_fired_at,
    };
    ctx.set(CRON_STATE_KEY, Json(state.clone()));
    Ok(state)
}

async fn cancel_stored_execution(ctx: &ObjectContext<'_>) -> HandlerResult<()> {
    if let Some(Json(existing)) = ctx.get::<Json<WorkbenchCronState>>(CRON_STATE_KEY).await? {
        let _ = ctx
            .invocation_handle(existing.next_execution_id)
            .cancel()
            .await;
    }
    Ok(())
}

async fn journaled_now(
    ctx: &ObjectContext<'_>,
    name: &'static str,
) -> HandlerResult<DateTime<Utc>> {
    let now = ctx
        .run(|| async { Ok::<_, HandlerError>(Utc::now().to_rfc3339()) })
        .name(name)
        .await?;
    DateTime::parse_from_rfc3339(&now)
        .map(|value| value.with_timezone(&Utc))
        .map_err(|err| TerminalError::new(err.to_string()).into())
}

fn next_cron_time(
    expr: &str,
    tz: Option<&str>,
    now: DateTime<Utc>,
) -> Result<DateTime<Utc>, String> {
    let timezone: Tz = tz
        .filter(|value| !value.trim().is_empty())
        .unwrap_or("UTC")
        .parse()
        .map_err(|err| format!("invalid timezone: {err}"))?;
    let cron = CronParser::builder()
        .seconds(Seconds::Optional)
        .build()
        .parse(expr)
        .map_err(|err| format!("invalid cron expression `{expr}`: {err}"))?;
    let zoned_now = now.with_timezone(&timezone);
    cron.find_next_occurrence(&zoned_now, false)
        .map(|value| value.with_timezone(&Utc))
        .map_err(|err| format!("cron expression `{expr}` has no next occurrence: {err}"))
}

fn cron_request_from_registration(
    session_id: &str,
    registration: &lash::TriggerRegistration,
) -> Result<(String, WorkbenchCronRequest), String> {
    let source_type = registration.source_type.as_str();
    if source_type != CRON_SCHEDULE_SOURCE_TYPE {
        return Err(format!("unexpected source type `{source_type}`"));
    }
    let value = registration
        .source
        .get(lashlang::LASH_HOST_VALUE_KEY)
        .ok_or_else(|| "missing host value payload".to_string())?;
    let expr = value
        .get("expr")
        .and_then(Value::as_str)
        .ok_or_else(|| "cron.Schedule requires string field `expr`".to_string())?
        .to_string();
    let tz = value
        .get("tz")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned);
    let request = WorkbenchCronRequest {
        session_id: session_id.to_string(),
        trigger_handle: registration.handle.clone(),
        expr,
        tz,
        name: registration.name.clone(),
    };
    Ok((cron_job_key(session_id, &registration.handle), request))
}

fn cron_job_key(session_id: &str, handle: &str) -> String {
    format!("{session_id}:{handle}")
}

async fn submit_restate_json<T: Serialize>(
    state: &AppState,
    url: String,
    body: &T,
) -> Result<(), AppError> {
    let response = state
        .restate_http
        .post(&url)
        .json(body)
        .send()
        .await
        .map_err(|err| AppError::internal(format!("Restate submit failed: {err}")))?;
    if !response.status().is_success() {
        return Err(AppError::internal(format!(
            "Restate submit failed with status {} for {url}",
            response.status()
        )));
    }
    Ok(())
}

async fn submit_restate_empty(state: &AppState, url: String) -> Result<(), AppError> {
    let response = state
        .restate_http
        .post(&url)
        .send()
        .await
        .map_err(|err| AppError::internal(format!("Restate submit failed: {err}")))?;
    if !response.status().is_success() {
        return Err(AppError::internal(format!(
            "Restate submit failed with status {} for {url}",
            response.status()
        )));
    }
    Ok(())
}

fn terminal_handler_error(err: AppError) -> HandlerError {
    TerminalError::new(err.message).into()
}
