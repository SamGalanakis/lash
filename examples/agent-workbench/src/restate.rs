use std::collections::{BTreeMap, BTreeSet};
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use chrono::{DateTime, Utc};
use chrono_tz::Tz;
use croner::parser::{CronParser, Seconds};
use lash::TurnInput;
use lash::modes::RlmTurnBuilderExt as _;
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
    AppError, AppState, ButtonChoice, CRON_SCHEDULE_SOURCE_TYPE, ChannelTurnEvents, ModelSelection,
    TurnStreamState, apply_model_selection_to_session, assistant_text_for_display,
    enqueue_button_trigger_command, enqueue_mail_received_trigger_command,
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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct WorkbenchButtonTriggerWorkflowRequest {
    pub operation_id: String,
    pub session_id: String,
    pub button: ButtonChoice,
    pub model: ModelSelection,
    pub pressed_at: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct WorkbenchSessionDeleteWorkflowRequest {
    pub operation_id: String,
    pub session_id: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct WorkbenchMailReceivedWorkflowRequest {
    pub operation_id: String,
    pub session_id: String,
    pub model: ModelSelection,
    pub delivery: crate::mail::MailDelivery,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct WorkbenchCronRequest {
    session_id: String,
    source_key: String,
    expr: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tz: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    name: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
struct CronScheduleSource {
    expr: String,
    #[serde(default)]
    tz: Option<String>,
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
    source_key: String,
    expr: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tz: Option<String>,
    next_execution_time: String,
    next_execution_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    last_fired_at: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct CronEmitReport {
    started_process_ids: Vec<String>,
}

impl From<&WorkbenchCronState> for WorkbenchCronInfo {
    fn from(state: &WorkbenchCronState) -> Self {
        Self {
            source_key: state.request.source_key.clone(),
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
        let session_id = request.session_id.clone();
        let controller = lash_restate::RestateRuntimeEffectController::new(ctx);
        run_user_turn(self.state.clone(), request, &controller)
            .await
            .map_err(terminal_handler_error)?;
        sync_cron_jobs_with_context(&self.state, controller.context(), "user_turn").await?;
        self.state
            .queued_work_poke
            .poke_session(session_id, "user_turn_completed");
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
        let session_id = request.session_id.clone();
        let controller = lash_restate::RestateRuntimeEffectController::new(ctx);
        let outcome = run_queued_turn(self.state.clone(), request, &controller)
            .await
            .map_err(terminal_handler_error);
        self.state
            .queued_work_poke
            .complete_session(session_id.clone(), "queued_turn_completed");
        outcome?;
        sync_cron_jobs_with_context(&self.state, controller.context(), "queued_turn").await?;
        Ok(Json(()))
    }
}

#[restate_sdk::workflow]
pub(crate) trait WorkbenchButtonTriggerWorkflow {
    async fn run(request: Json<WorkbenchButtonTriggerWorkflowRequest>) -> HandlerResult<Json<()>>;
}

pub(crate) struct WorkbenchButtonTriggerWorkflowImpl {
    state: AppState,
}

impl WorkbenchButtonTriggerWorkflowImpl {
    pub(crate) fn new(state: AppState) -> Self {
        Self { state }
    }
}

impl WorkbenchButtonTriggerWorkflow for WorkbenchButtonTriggerWorkflowImpl {
    async fn run(
        &self,
        ctx: WorkflowContext<'_>,
        Json(request): Json<WorkbenchButtonTriggerWorkflowRequest>,
    ) -> HandlerResult<Json<()>> {
        let session_id = request.session_id.clone();
        let controller = lash_restate::RestateRuntimeEffectController::new(ctx);
        run_button_trigger(self.state.clone(), request, &controller)
            .await
            .map_err(terminal_handler_error)?;
        self.state
            .queued_work_poke
            .poke_session(session_id, "button_trigger");
        Ok(Json(()))
    }
}

#[restate_sdk::workflow]
pub(crate) trait WorkbenchMailReceivedWorkflow {
    async fn run(request: Json<WorkbenchMailReceivedWorkflowRequest>) -> HandlerResult<Json<()>>;
}

pub(crate) struct WorkbenchMailReceivedWorkflowImpl {
    state: AppState,
}

impl WorkbenchMailReceivedWorkflowImpl {
    pub(crate) fn new(state: AppState) -> Self {
        Self { state }
    }
}

impl WorkbenchMailReceivedWorkflow for WorkbenchMailReceivedWorkflowImpl {
    async fn run(
        &self,
        ctx: WorkflowContext<'_>,
        Json(request): Json<WorkbenchMailReceivedWorkflowRequest>,
    ) -> HandlerResult<Json<()>> {
        let session_id = request.session_id.clone();
        let controller = lash_restate::RestateRuntimeEffectController::new(ctx);
        run_mail_received(self.state.clone(), request, &controller)
            .await
            .map_err(terminal_handler_error)?;
        self.state
            .queued_work_poke
            .poke_session(session_id, "mail_received");
        Ok(Json(()))
    }
}

#[restate_sdk::workflow]
pub(crate) trait WorkbenchSessionDeleteWorkflow {
    async fn run(request: Json<WorkbenchSessionDeleteWorkflowRequest>) -> HandlerResult<Json<()>>;
}

pub(crate) struct WorkbenchSessionDeleteWorkflowImpl {
    state: AppState,
}

impl WorkbenchSessionDeleteWorkflowImpl {
    pub(crate) fn new(state: AppState) -> Self {
        Self { state }
    }
}

impl WorkbenchSessionDeleteWorkflow for WorkbenchSessionDeleteWorkflowImpl {
    async fn run(
        &self,
        ctx: WorkflowContext<'_>,
        Json(request): Json<WorkbenchSessionDeleteWorkflowRequest>,
    ) -> HandlerResult<Json<()>> {
        let controller = lash_restate::RestateRuntimeEffectController::new(ctx);
        run_session_delete(self.state.clone(), request, &controller)
            .await
            .map_err(terminal_handler_error)?;
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
        let now = journaled_now(&ctx, "workbench-cron:upsert-now").await?;
        if let Some(Json(existing)) = ctx.get::<Json<WorkbenchCronState>>(CRON_STATE_KEY).await?
            && existing.request == request
        {
            // Only short-circuit while the stored chain is still alive. If the
            // recorded next execution is in the past the chain died (e.g. a
            // crash between fire and re-arm) and an equal-request upsert is
            // exactly the sync pass that should revive it.
            let chain_alive = DateTime::parse_from_rfc3339(&existing.next_execution_time)
                .map(|next| next.with_timezone(&Utc) > now)
                .unwrap_or(false);
            if chain_alive {
                return Ok(Json(WorkbenchCronInfo::from(&existing)));
            }
        }
        cancel_stored_execution(&ctx).await?;
        let state = schedule_next(&ctx, request, now, None).await?;
        Ok(Json(WorkbenchCronInfo::from(&state)))
    }

    async fn run(&self, ctx: ObjectContext<'_>) -> HandlerResult<Json<()>> {
        let Some(Json(state)) = ctx.get::<Json<WorkbenchCronState>>(CRON_STATE_KEY).await? else {
            return Ok(Json(()));
        };
        // Zombie guard: a job whose session is no longer the live workbench
        // session terminates itself instead of firing into a deleted session.
        // (Jobs armed by a previous process run are invisible to the
        // in-memory cancel bookkeeping, so reset alone cannot reach them.)
        let current_session = self.state.session_ids.current();
        if state.request.session_id != current_session {
            self.state.trace(
                "cron.restate.zombie_cancelled",
                json!({
                    "job_key": ctx.key(),
                    "job_session_id": state.request.session_id,
                    "current_session_id": current_session,
                }),
            );
            ctx.clear(CRON_STATE_KEY);
            return Ok(Json(()));
        }
        let controller = lash_restate::RestateRuntimeEffectController::new(ctx);
        let fired_at = journaled_now(controller.context(), "workbench-cron:fired-at").await?;
        let request = state.request.clone();
        let fired_at_text = fired_at.to_rfc3339();
        // Re-arm before emitting: a tick whose emission fails terminally must
        // not take the whole schedule down with it.
        schedule_next(
            controller.context(),
            state.request.clone(),
            fired_at,
            Some(fired_at.to_rfc3339()),
        )
        .await?;
        let Json(emit_report) =
            emit_cron_occurrence(self.state.clone(), request, fired_at_text, &controller).await?;
        self.state.trace(
            "cron.restate.run",
            json!({
                "job_key": controller.context().key(),
                "source_key": state.request.source_key,
                "expr": state.request.expr,
                "tz": state.request.tz,
                "fired_at": fired_at.to_rfc3339(),
                "started_process_ids": emit_report.started_process_ids,
            }),
        );
        self.state
            .queued_work_poke
            .poke_session(state.request.session_id.clone(), "cron_tick");
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
    process_deployment: lash_restate::RestateProcessDeployment,
    process_worker: lash::durability::DurableProcessWorker,
) {
    let endpoint = Endpoint::builder()
        .bind(WorkbenchTurnWorkflowImpl::new(state.clone()).serve())
        .bind(WorkbenchQueuedTurnWorkflowImpl::new(state.clone()).serve())
        .bind(WorkbenchButtonTriggerWorkflowImpl::new(state.clone()).serve())
        .bind(WorkbenchMailReceivedWorkflowImpl::new(state.clone()).serve())
        .bind(WorkbenchSessionDeleteWorkflowImpl::new(state.clone()).serve())
        .bind(WorkbenchCronJobImpl::new(state).serve())
        .bind(process_deployment.workflow(process_worker).serve())
        .build();
    tokio::spawn(async move {
        restate_sdk::http_server::HttpServer::new(endpoint)
            .listen_and_serve(addr)
            .await;
    });
    process_deployment.spawn();
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

pub(crate) async fn submit_queued_turn_request(
    restate_http: &reqwest::Client,
    restate_ingress_url: &str,
    request: &WorkbenchQueuedTurnWorkflowRequest,
) -> Result<(), AppError> {
    let url = format!(
        "{}/WorkbenchQueuedTurnWorkflow/{}/run/send",
        restate_ingress_url.trim_end_matches('/'),
        request.turn_id
    );
    submit_restate_json_with_client(restate_http, url, request).await
}

pub(crate) async fn submit_button_trigger(
    state: &AppState,
    request: WorkbenchButtonTriggerWorkflowRequest,
) -> Result<(), AppError> {
    let url = format!(
        "{}/WorkbenchButtonTriggerWorkflow/{}/run/send",
        state.restate_ingress_url.trim_end_matches('/'),
        request.operation_id
    );
    submit_restate_json(state, url, &request).await
}

pub(crate) async fn submit_mail_received(
    state: &AppState,
    request: WorkbenchMailReceivedWorkflowRequest,
) -> Result<(), AppError> {
    submit_mail_received_with_client(&state.restate_http, &state.restate_ingress_url, request).await
}

pub(crate) async fn submit_mail_received_with_client(
    restate_http: &reqwest::Client,
    restate_ingress_url: &str,
    request: WorkbenchMailReceivedWorkflowRequest,
) -> Result<(), AppError> {
    let url = format!(
        "{}/WorkbenchMailReceivedWorkflow/{}/run/send",
        restate_ingress_url.trim_end_matches('/'),
        request.operation_id
    );
    submit_restate_json_with_client(restate_http, url, &request).await
}

pub(crate) async fn submit_session_delete(
    state: &AppState,
    request: WorkbenchSessionDeleteWorkflowRequest,
) -> Result<(), AppError> {
    let url = format!(
        "{}/WorkbenchSessionDeleteWorkflow/{}/run/send",
        state.restate_ingress_url.trim_end_matches('/'),
        request.operation_id
    );
    submit_restate_json(state, url, &request).await
}

/// Cancel every cron job belonging to `session_id`, derived from the durable
/// trigger registrations (the same source `sync_cron_jobs_with_context`
/// schedules from), plus anything this process armed. The in-memory key set
/// alone is not enough: jobs armed by a previous process run are invisible to
/// it and would keep firing into a deleted session forever.
pub(crate) async fn cancel_cron_jobs_for_session(
    state: &AppState,
    session_id: &str,
    reason: &str,
) -> Result<(), AppError> {
    let session = state
        .core
        .session(session_id.to_string())
        .rlm()
        .open()
        .await
        .map_err(AppError::internal)?;
    let registrations = session
        .triggers()
        .by_source_type(CRON_SCHEDULE_SOURCE_TYPE)
        .await
        .map_err(AppError::internal)?;
    let mut job_keys: BTreeSet<String> = registrations
        .iter()
        .map(|registration| cron_job_key(session_id, &registration.source_key))
        .collect();
    session.close().await.map_err(AppError::internal)?;
    job_keys.extend({
        let mut guard = state
            .restate_cron_job_keys
            .lock()
            .expect("restate cron job key lock");
        let known = guard.clone();
        guard.clear();
        known
    });
    for job_key in job_keys {
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
    let mut input = TurnInput::text(request.text.clone());
    input.trace_turn_id = Some(request.turn_id.clone());
    // Registered for /api/turn/cancel (the UI's stop button / Esc); the guard
    // drops the registry entry when the turn finishes either way, and the
    // cancel endpoint calls `cancel_running_turns()` on this session handle.
    let _cancel_guard = state.register_turn_cancel(&session, &request.turn_id);
    let output = session
        .turn(input)
        .turn_id(request.turn_id.clone())
        .model(turn_model)
        .require_submit()
        .map_err(AppError::internal)?
        .effects(controller)
        .stream_to(&ui_events)
        .await
        .map_err(AppError::internal)?;
    record_turn_output(&state, output, turn_state, "restate_user_turn.completed");
    Ok(())
}

async fn run_button_trigger(
    state: AppState,
    request: WorkbenchButtonTriggerWorkflowRequest,
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
    apply_model_selection_to_session(
        &state,
        &session,
        turn_model.clone(),
        "restate_button_trigger",
    )
    .await?;
    let scoped_effect_controller = controller
        .scoped_effect_controller(lash::runtime::EffectScope::runtime_operation(format!(
            "button-trigger:{}",
            request.operation_id
        )))
        .map_err(AppError::internal)?;
    let receipt = enqueue_button_trigger_command(
        &state,
        request.button,
        &request.pressed_at,
        &request.operation_id,
        scoped_effect_controller,
    )
    .await
    .map_err(AppError::internal)?;
    state.trace(
        "button_trigger.restate.trigger_occurrence",
        json!({
            "button": request.button,
            "occurrence_id": receipt.occurrence_id,
            "started_process_ids": receipt.started_process_ids,
        }),
    );
    state.push_message("event", "button trigger occurrence emitted");
    // Host-event dispatch is the end of this client-initiated request. Emit a
    // terminal Done so the UI clears its busy state even when no trigger
    // matched (any process the event started streams its own turn separately).
    state.publish(crate::StreamItem::Done);
    Ok(())
}

async fn run_mail_received(
    state: AppState,
    request: WorkbenchMailReceivedWorkflowRequest,
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
    apply_model_selection_to_session(&state, &session, turn_model, "restate_mail_received").await?;
    let scoped_effect_controller = controller
        .scoped_effect_controller(lash::runtime::EffectScope::runtime_operation(format!(
            "mail-received:{}",
            request.operation_id
        )))
        .map_err(AppError::internal)?;
    let receipt = enqueue_mail_received_trigger_command(
        &state,
        &request.delivery,
        &request.operation_id,
        scoped_effect_controller,
    )
    .await
    .map_err(AppError::internal)?;
    state.trace(
        "mail_received.restate.trigger_occurrence",
        json!({
            "account": request.delivery.account,
            "title": request.delivery.title,
            "occurrence_id": receipt.occurrence_id,
            "started_process_ids": receipt.started_process_ids,
        }),
    );
    state.push_message("event", "mail received trigger occurrence queued");
    // Host-event dispatch is the end of this client-initiated request. Emit a
    // terminal Done so the UI clears its busy state even when no trigger
    // matched (any process the event started streams its own turn separately).
    state.publish(crate::StreamItem::Done);
    Ok(())
}

async fn run_session_delete(
    state: AppState,
    request: WorkbenchSessionDeleteWorkflowRequest,
    controller: &lash_restate::RestateRuntimeEffectController<'_, WorkflowContext<'_>>,
) -> Result<(), AppError> {
    let scoped_effect_controller = controller
        .scoped_effect_controller(lash::runtime::EffectScope::session_delete(
            &request.session_id,
        ))
        .map_err(AppError::internal)?;
    state
        .core
        .delete_session(&request.session_id, scoped_effect_controller)
        .await
        .map_err(AppError::internal)?;
    state.trace(
        "reset.restate.session_deleted",
        json!({
            "session_id": request.session_id,
        }),
    );
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
        "queued_work.restate.start",
        json!({
            "reason": request.reason,
            "session_id": request.session_id,
            "turn_id": request.turn_id,
            "model": serde_json::to_value(&selected_model).unwrap_or(Value::Null),
        }),
    );
    let Some(output) = session
        .queued_turn()
        .drain_id(request.turn_id.clone())
        .effects(controller)
        .stream_to(&ui_events)
        .await
        .map_err(AppError::internal)?
    else {
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
    if matches!(
        output.outcome,
        lash::runtime::TurnOutcome::Stopped(lash::runtime::TurnStop::Cancelled)
    ) {
        state.push_message("event", "turn cancelled");
    } else {
        state.push_message("assistant", assistant_text);
    }
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
    let mut scheduled = BTreeMap::new();
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
        scheduled.entry(job_key).or_insert(request);
    }
    let mut active = BTreeSet::new();
    for (job_key, request) in scheduled {
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

/// Idempotency key for one cron tick's trigger occurrence. Must be unique
/// per (job, tick): `fired_at` is the journaled fire time, so retries of the
/// same tick dedupe while the next tick gets a fresh occurrence. (A key
/// without the tick component kills the schedule: the second tick conflicts,
/// the handler fails before re-arming, and the chain stops.)
fn cron_occurrence_key(job_key: &str, fired_at: &str) -> String {
    format!("workbench-cron:{job_key}:{fired_at}")
}

async fn emit_cron_occurrence(
    state: AppState,
    request: WorkbenchCronRequest,
    fired_at: String,
    controller: &lash_restate::RestateRuntimeEffectController<'_, ObjectContext<'_>>,
) -> HandlerResult<Json<CronEmitReport>> {
    let scoped_effect_controller = controller
        .scoped_effect_controller(lash::runtime::EffectScope::runtime_operation(format!(
            "cron:{}:{fired_at}",
            controller.context().key()
        )))
        .map_err(|err| HandlerError::from(TerminalError::new(err.to_string())))?;
    let report = state
        .core
        .triggers()
        .emit(
            lash::triggers::TriggerOccurrenceRequest::new(
                CRON_SCHEDULE_SOURCE_TYPE,
                request.source_key.clone(),
                json!({
                    "fired_at": fired_at,
                }),
                cron_occurrence_key(controller.context().key(), &fired_at),
            )
            .with_source(json!({
                "expr": request.expr,
                "tz": request.tz,
            })),
            scoped_effect_controller,
        )
        .await
        .map_err(|err| HandlerError::from(TerminalError::new(err.to_string())))?;
    Ok(Json(CronEmitReport {
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
    registration: &lash::triggers::TriggerRegistration,
) -> Result<(String, WorkbenchCronRequest), String> {
    let source_type = registration.source_type.as_str();
    if source_type != CRON_SCHEDULE_SOURCE_TYPE {
        return Err(format!("unexpected source type `{source_type}`"));
    }
    let source =
        lashlang::HostDescriptor::decode(&registration.source).map_err(|err| err.to_string())?;
    if source.source_type != source_type {
        return Err(format!(
            "registration source type `{source_type}` does not match host descriptor `{}`",
            source.source_type
        ));
    }
    let payload: CronScheduleSource = source
        .decode_as(&crate::workbench_lashlang_resources())
        .map_err(|err| err.to_string())?;
    let request = WorkbenchCronRequest {
        session_id: session_id.to_string(),
        source_key: registration.source_key.clone(),
        expr: payload.expr,
        tz: payload.tz,
        name: registration.name.clone(),
    };
    Ok((cron_job_key(session_id, &registration.source_key), request))
}

fn cron_job_key(session_id: &str, source_key: &str) -> String {
    format!("{session_id}:{source_key}")
}

async fn submit_restate_json<T: Serialize>(
    state: &AppState,
    url: String,
    body: &T,
) -> Result<(), AppError> {
    submit_restate_json_with_client(&state.restate_http, url, body).await
}

async fn submit_restate_json_with_client<T: Serialize>(
    restate_http: &reqwest::Client,
    url: String,
    body: &T,
) -> Result<(), AppError> {
    let response = restate_http
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

#[cfg(test)]
mod tests {
    use super::cron_occurrence_key;

    #[test]
    fn cron_occurrence_key_is_unique_per_tick() {
        let job = "session:source:cron.Schedule:sha256:abc";
        let first = cron_occurrence_key(job, "2026-06-09T22:30:30+00:00");
        let second = cron_occurrence_key(job, "2026-06-09T22:31:00+00:00");
        // Two ticks of one job must not collide: a constant key makes the
        // second tick fail its trigger emit with an idempotency conflict
        // before re-arming, killing the schedule after exactly one fire.
        assert_ne!(first, second);
        // A retried tick (same journaled fired_at) must dedupe.
        assert_eq!(first, cron_occurrence_key(job, "2026-06-09T22:30:30+00:00"));
        // Distinct jobs never collide on the same tick time.
        assert_ne!(
            first,
            cron_occurrence_key("other-job", "2026-06-09T22:30:30+00:00")
        );
    }
}
