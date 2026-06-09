use anyhow::{Context, Result};
use lash::durability::DurableProcessWorker;
use lash::{SessionResume, TurnActivity, TurnActivitySink, TurnEvent, TurnInput};
use lash_postgres_store::PostgresStorage;
use lash_restate::{LashProcessWorkflow, RestateProcessDeployment, RestateRuntimeEffectController};
use restate_sdk::errors::{HandlerResult, TerminalError};
use restate_sdk::prelude::{Endpoint, WorkflowContext};
use restate_sdk::serde::Json;
use serde_json::json;
use std::fmt::Display;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use lash_restate_postgres_workers_e2e::{
    DEFAULT_SESSION_ID, EXPECTED_FINAL_TEXT, HealthResponse, TurnRequest, TurnResponse,
    TurnScenario, build_e2e_core, default_session_child_owner_scope_pattern,
    default_session_owner_scope_id, ensure_e2e_schema, env, process_registry_from_storage,
    record_terminal_result, record_turn_activity, record_worker_event, required_env,
    s3_store_from_env,
};

fn terminal_error(err: impl Display) -> TerminalError {
    TerminalError::new(err.to_string())
}

#[restate_sdk::workflow]
trait E2eTurnWorkflow {
    async fn run(request: Json<TurnRequest>) -> HandlerResult<Json<TurnResponse>>;

    #[shared]
    async fn health() -> HandlerResult<Json<HealthResponse>>;
}

#[derive(Clone)]
struct AppState {
    worker_id: String,
    storage: PostgresStorage,
    attachment_store: Arc<dyn lash::persistence::AttachmentStore>,
    process_work_driver: lash::process::ProcessWorkDriver,
    mock_provider_base_url: String,
    trace_dir: Option<PathBuf>,
    fail_once: bool,
}

impl AppState {
    async fn connect(process_work_driver: lash::process::ProcessWorkDriver) -> Result<Self> {
        let worker_id = env("WORKER_INSTANCE_ID", "worker-local");
        let database_url = required_env("DATABASE_URL")?;
        let storage = PostgresStorage::connect(&database_url)
            .await
            .context("connect Postgres storage")?;
        ensure_e2e_schema(storage.pool()).await?;
        let attachment_store =
            Arc::new(s3_store_from_env()?) as Arc<dyn lash::persistence::AttachmentStore>;
        let mock_provider_base_url = env("MOCK_PROVIDER_BASE_URL", "http://mock-provider:18001");
        let trace_dir = std::env::var("LASH_E2E_TRACE_DIR").ok().map(PathBuf::from);
        if let Some(dir) = &trace_dir {
            std::fs::create_dir_all(dir)
                .with_context(|| format!("create trace dir `{}`", dir.display()))?;
        }
        let fail_once = env("LASH_E2E_FAIL_ONCE", "0") == "1";
        Ok(Self {
            worker_id,
            storage,
            attachment_store,
            process_work_driver,
            mock_provider_base_url,
            trace_dir,
            fail_once,
        })
    }

    fn build_core(&self) -> Result<lash::LashCore> {
        build_e2e_core(lash_restate_postgres_workers_e2e::E2eCoreConfig {
            worker_id: self.worker_id.clone(),
            storage: self.storage.clone(),
            attachment_store: Arc::clone(&self.attachment_store),
            process_work_driver: self.process_work_driver.clone(),
            mock_provider_base_url: self.mock_provider_base_url.clone(),
            trace_dir: self.trace_dir.clone(),
            fail_once: self.fail_once,
        })
    }

    async fn run_turn_with_restate(
        &self,
        ctx: WorkflowContext<'_>,
        request: TurnRequest,
    ) -> HandlerResult<Json<TurnResponse>> {
        self.record(
            &request.workflow_id,
            "turn_started",
            json!({
                "scenario": request.scenario,
                "fail_once": request.fail_once,
            }),
        )
        .await?;

        let controller = RestateRuntimeEffectController::new(ctx);
        let core = self.build_core().map_err(terminal_error)?;
        let session = core
            .session(DEFAULT_SESSION_ID)
            .rlm()
            .open()
            .await
            .map_err(terminal_error)?;

        let cursor = session.observe().current_observation().cursor;
        let cursor_text = cursor.as_str().to_string();
        if request.scenario == TurnScenario::DrainQueued {
            let sink = RecordingTurnSink::new(
                self.storage.pool().clone(),
                request.workflow_id.clone(),
                self.worker_id.clone(),
                "queued",
                Some(cursor_text.clone()),
            );
            let turn = session
                .queued_turn()
                .drain_id(request.workflow_id.clone())
                .effects(&controller)
                .stream_to(&sink)
                .await
                .map_err(terminal_error)?;
            let submitted_value = turn
                .as_ref()
                .and_then(|turn| turn.submitted_value().cloned())
                .unwrap_or(serde_json::Value::Null);
            return self
                .finish_response(
                    &request,
                    submitted_value,
                    sink.count().await,
                    Some(cursor_text),
                    turn.is_some(),
                )
                .await
                .map(Json);
        }

        let sink = RecordingTurnSink::new(
            self.storage.pool().clone(),
            request.workflow_id.clone(),
            self.worker_id.clone(),
            "main",
            Some(cursor_text.clone()),
        );
        let input = TurnInput::text(prompt_for_request(&request))
            .with_trace_turn_id(request.workflow_id.clone());
        let turn = session
            .turn(input)
            .turn_id(request.workflow_id.clone())
            .effects(&controller)
            .stream_to(&sink)
            .await
            .map_err(terminal_error)?;
        let submitted_value = turn
            .submitted_value()
            .cloned()
            .unwrap_or(serde_json::Value::Null);
        let submitted_events = sink.submitted_values().await;
        self.record(
            &request.workflow_id,
            "main_submitted_values",
            json!({
                "turn_result": submitted_value.clone(),
                "stream_submitted_values": submitted_events,
            }),
        )
        .await?;

        let replay_count = match session.observe().resume_from_cursor(&cursor) {
            Ok(SessionResume::Replayed { events }) => events.len(),
            Ok(SessionResume::Gap { .. }) => 0,
            Err(err) => {
                self.record(
                    &request.workflow_id,
                    "live_replay_failed",
                    json!({"error": err.to_string()}),
                )
                .await?;
                0
            }
        };
        self.record(
            &request.workflow_id,
            "live_replay_checked",
            json!({"events": replay_count, "cursor": cursor_text}),
        )
        .await?;

        self.finish_response(
            &request,
            submitted_value,
            sink.count().await,
            Some(cursor_text),
            false,
        )
        .await
        .map(Json)
    }

    async fn finish_response(
        &self,
        request: &TurnRequest,
        submitted_value: serde_json::Value,
        streamed_event_count: usize,
        replay_cursor: Option<String>,
        queued_turn_ran: bool,
    ) -> HandlerResult<TurnResponse> {
        let process_ids = self.load_session_process_ids().await?;
        let attachment_id = submitted_value
            .get("attachment_id")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default()
            .to_string();
        let final_text = submitted_value
            .get("final")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_else(|| match request.scenario {
                TurnScenario::KitchenSink => EXPECTED_FINAL_TEXT,
                TurnScenario::TriggerSetup => "trigger-registered",
                TurnScenario::DrainQueued => "wake-consumed",
            })
            .to_string();
        let response = TurnResponse {
            workflow_id: request.workflow_id.clone(),
            worker_id: self.worker_id.clone(),
            process_id: process_ids.first().cloned().unwrap_or_default(),
            process_ids,
            attachment_id,
            final_text,
            submitted_value,
            streamed_event_count,
            replay_cursor,
            queued_turn_ran,
        };
        record_terminal_result(self.storage.pool(), &response)
            .await
            .map_err(terminal_error)?;
        self.record(
            &request.workflow_id,
            "turn_completed",
            json!({
                "attachment_id": response.attachment_id,
                "final_text": response.final_text,
                "queued_turn_ran": response.queued_turn_ran,
                "streamed_event_count": response.streamed_event_count,
            }),
        )
        .await?;
        Ok(response)
    }

    async fn load_session_process_ids(&self) -> HandlerResult<Vec<String>> {
        Ok(sqlx::query_scalar::<_, String>(
            "SELECT process_id
             FROM lash_processes
             WHERE owner_scope_id = $1 OR owner_scope_id LIKE $2
             ORDER BY created_at_ms, process_id",
        )
        .bind(default_session_owner_scope_id())
        .bind(default_session_child_owner_scope_pattern())
        .fetch_all(self.storage.pool())
        .await
        .map_err(terminal_error)?)
    }

    async fn record(
        &self,
        workflow_id: &str,
        event_type: &str,
        detail: serde_json::Value,
    ) -> HandlerResult<()> {
        record_worker_event(
            self.storage.pool(),
            workflow_id,
            &self.worker_id,
            event_type,
            detail,
        )
        .await
        .map_err(terminal_error)?;
        Ok(())
    }
}

fn prompt_for_request(request: &TurnRequest) -> String {
    match request.scenario {
        TurnScenario::KitchenSink => format!(
            "Run the canonical Lash Restate/Postgres/S3 kitchen sink workflow. workflow_id={} fail_once={}",
            request.workflow_id, request.fail_once
        ),
        TurnScenario::TriggerSetup => format!(
            "Register the E2E host event trigger through Lashlang. workflow_id={} trigger_setup=true",
            request.workflow_id
        ),
        TurnScenario::DrainQueued => format!(
            "Drain the next queued E2E wake turn. workflow_id={} drain_queued=true",
            request.workflow_id
        ),
    }
}

#[derive(Clone)]
struct RecordingTurnSink {
    pool: sqlx::PgPool,
    workflow_id: String,
    worker_id: String,
    stream_name: String,
    cursor: Option<String>,
    activities: Arc<tokio::sync::Mutex<Vec<TurnActivity>>>,
}

impl RecordingTurnSink {
    fn new(
        pool: sqlx::PgPool,
        workflow_id: String,
        worker_id: String,
        stream_name: impl Into<String>,
        cursor: Option<String>,
    ) -> Self {
        Self {
            pool,
            workflow_id,
            worker_id,
            stream_name: stream_name.into(),
            cursor,
            activities: Arc::new(tokio::sync::Mutex::new(Vec::new())),
        }
    }

    async fn count(&self) -> usize {
        self.activities.lock().await.len()
    }

    async fn submitted_values(&self) -> Vec<serde_json::Value> {
        self.activities
            .lock()
            .await
            .iter()
            .filter_map(|activity| match &activity.event {
                TurnEvent::SubmittedValue { value } => Some(value.clone()),
                _ => None,
            })
            .collect()
    }
}

#[async_trait::async_trait]
impl TurnActivitySink for RecordingTurnSink {
    async fn emit(&self, activity: TurnActivity) {
        if let Err(err) = record_turn_activity(
            &self.pool,
            &self.workflow_id,
            &self.worker_id,
            &self.stream_name,
            self.cursor.as_deref(),
            &activity,
        )
        .await
        {
            tracing::error!(
                workflow_id = %self.workflow_id,
                worker_id = %self.worker_id,
                error = %err,
                "failed to record streamed turn activity"
            );
        }
        self.activities.lock().await.push(activity);
    }
}

struct E2eTurnWorkflowImpl {
    state: AppState,
}

impl E2eTurnWorkflowImpl {
    fn new(state: AppState) -> Self {
        Self { state }
    }
}

impl E2eTurnWorkflow for E2eTurnWorkflowImpl {
    async fn run(
        &self,
        ctx: WorkflowContext<'_>,
        Json(request): Json<TurnRequest>,
    ) -> HandlerResult<Json<TurnResponse>> {
        self.state.run_turn_with_restate(ctx, request).await
    }

    async fn health(
        &self,
        _ctx: restate_sdk::context::SharedWorkflowContext<'_>,
    ) -> HandlerResult<Json<HealthResponse>> {
        Ok(Json(HealthResponse {
            worker_id: self.state.worker_id.clone(),
            ok: true,
        }))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let database_url = required_env("DATABASE_URL")?;
    let storage = PostgresStorage::connect(&database_url)
        .await
        .context("connect Postgres storage for process deployment")?;
    ensure_e2e_schema(storage.pool()).await?;
    let registry = process_registry_from_storage(&storage);
    let deployment =
        RestateProcessDeployment::new(env("RESTATE_INGRESS_URL", "http://restate:8080"), registry);
    let process_work_driver = deployment.process_work_driver();
    let state = AppState::connect(process_work_driver.clone()).await?;
    if state.fail_once {
        tracing::warn!(worker_id = %state.worker_id, "worker can exit once from crash_once tool");
    }

    let core = state.build_core()?;
    let process_worker = DurableProcessWorker::new(core.durable_process_worker_config()?);
    let process_workflow = deployment.workflow(process_worker);
    let _process_poke = deployment.spawn();

    let port = env("WORKER_PORT", "18100");
    let addr: SocketAddr = format!("0.0.0.0:{port}")
        .parse()
        .context("parse worker addr")?;
    let endpoint = Endpoint::builder()
        .bind(E2eTurnWorkflowImpl::new(state).serve())
        .bind(process_workflow.serve())
        .build();
    restate_sdk::http_server::HttpServer::new(endpoint)
        .listen_and_serve(addr)
        .await;
    Ok(())
}
