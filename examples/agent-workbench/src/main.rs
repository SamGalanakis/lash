mod execution_graphs;
mod mail;
mod restate;
mod ui;

use std::collections::{BTreeMap, BTreeSet};
use std::convert::Infallible;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
#[cfg(test)]
use std::time::Duration;

use anyhow::{Context, Result as AnyhowResult, anyhow};
use async_trait::async_trait;
use axum::body::Body;
use axum::extract::{Path as AxumPath, State};
use axum::http::{StatusCode, header};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use bytes::Bytes;
use chrono::Utc;
use lash::host_events::HostEvent;
use lash::plugins::{
    PluginError, PluginFactory, PluginRegistrar, PluginSessionContext, SessionPlugin,
};
use lash::prompt::PromptContribution;
use lash::provider::{ProviderHandle, ProviderOptions, ProviderThinkingPolicy};
use lash::{
    LashCore, ModeId, ModePreset, SessionSpec, TurnActivity, TurnActivitySink, TurnEvent,
    TurnResult,
    tracing::{
        JsonlTraceSink, StderrTraceSink, TeeTraceSink, TraceContext, TraceEvent,
        TraceLashlangGraph, TraceLashlangGraphStore, TraceLevel, TraceRecord, TraceSink,
    },
};
use lash_provider_openai::{OPENROUTER_BASE_URL, OpenAiCompatibleProvider};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::sync::{broadcast, mpsc};
use tokio_stream::StreamExt;
use tokio_stream::wrappers::ReceiverStream;

const SESSION_ID_PREFIX: &str = "workbench";
const DEFAULT_CONTEXT_WINDOW_TOKENS: usize = 200_000;
pub(crate) const BUTTON_TRIGGER_RESOURCE: &str = "Button";
pub(crate) const BUTTON_TRIGGER_ALIAS: &str = "ui.button";
pub(crate) const BUTTON_TRIGGER_EVENT: &str = "pressed";
pub(crate) const BUTTON_TRIGGER_SOURCE_TYPE: &str = "ui.button.pressed";
pub(crate) const CRON_SCHEDULE_SOURCE_TYPE: &str = "cron.Schedule";
pub(crate) const MAIL_EVENT_RESOURCE: &str = "Mail";
pub(crate) const MAIL_EVENT_ALIAS: &str = "mail";
pub(crate) const MAIL_EVENT_EVENT: &str = "received";
pub(crate) const MAIL_RECEIVED_SOURCE_TYPE: &str = "mail.received";
const DEFAULT_TOKIO_THREAD_STACK_BYTES: usize = 2 * 1024 * 1024;

fn main() -> AnyhowResult<()> {
    let stack_bytes = std::env::var("AGENT_WORKBENCH_TOKIO_STACK_BYTES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(DEFAULT_TOKIO_THREAD_STACK_BYTES);
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .thread_stack_size(stack_bytes)
        .build()
        .context("build agent-workbench tokio runtime")?
        .block_on(async_main())
}

async fn async_main() -> AnyhowResult<()> {
    let _ = dotenvy::dotenv();

    let addr: SocketAddr = std::env::var("AGENT_WORKBENCH_ADDR")
        .unwrap_or_else(|_| "127.0.0.1:3030".to_string())
        .parse()
        .context("invalid AGENT_WORKBENCH_ADDR")?;
    let restate_endpoint_addr: SocketAddr = std::env::var("AGENT_WORKBENCH_RESTATE_ADDR")
        .unwrap_or_else(|_| "127.0.0.1:9081".to_string())
        .parse()
        .context("invalid AGENT_WORKBENCH_RESTATE_ADDR")?;
    let restate_ingress_url = std::env::var("RESTATE_INGRESS_URL")
        .unwrap_or_else(|_| "http://127.0.0.1:8080".to_string());
    let data_dir = std::env::var("AGENT_WORKBENCH_DATA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(".agent-workbench"));
    std::fs::create_dir_all(&data_dir).with_context(|| format!("create {}", data_dir.display()))?;
    let trace_path = std::env::var("AGENT_WORKBENCH_TRACE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| data_dir.join("trace.jsonl"));
    eprintln!("agent-workbench trace: {}", trace_path.display());
    let trace_path_display = trace_path.display().to_string();
    let trace_sink = Arc::new(TeeTraceSink::new([
        Arc::new(StderrTraceSink::default()) as Arc<dyn TraceSink>,
        Arc::new(JsonlTraceSink::new(trace_path)),
    ])) as Arc<dyn TraceSink>;
    let lashlang_execution_path = std::env::var("AGENT_WORKBENCH_LASHLANG_EXECUTION_TRACE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| data_dir.join("lashlang-execution.jsonl"));
    eprintln!(
        "agent-workbench Lashlang execution trace: {}",
        lashlang_execution_path.display()
    );
    let lashlang_execution = Arc::new(TraceLashlangGraphStore::default());
    let lashlang_execution_sink = Arc::new(TeeTraceSink::new([
        Arc::clone(&lashlang_execution) as Arc<dyn TraceSink>,
        Arc::new(JsonlTraceSink::new(lashlang_execution_path.clone())) as Arc<dyn TraceSink>,
    ])) as Arc<dyn TraceSink>;

    let api_key = std::env::var("OPENROUTER_API_KEY").unwrap_or_default();
    if api_key.trim().is_empty() {
        eprintln!("warning: OPENROUTER_API_KEY is empty; turns will fail until it is set");
    }
    let tavily_api_key = std::env::var("TAVILY_API_KEY").unwrap_or_default();
    if tavily_api_key.trim().is_empty() {
        eprintln!("warning: TAVILY_API_KEY is empty; web tools will return configuration errors");
    }
    let model = std::env::var("OPENROUTER_MODEL")
        .unwrap_or_else(|_| "anthropic/claude-sonnet-4.6".to_string());
    let model_variant =
        std::env::var("OPENROUTER_MODEL_VARIANT").unwrap_or_else(|_| "high".to_string());

    let provider = ProviderHandle::new(
        OpenAiCompatibleProvider::new(api_key, OPENROUTER_BASE_URL)
            .with_options(ProviderOptions {
                thinking: ProviderThinkingPolicy { expose: true },
                ..ProviderOptions::default()
            })
            .into_components(),
    );
    let model_spec = lash::ModelSpec::from_token_limits(
        model.clone(),
        Some(model_variant.clone()),
        DEFAULT_CONTEXT_WINDOW_TOKENS,
        None,
    )
    .map_err(|err| anyhow!("invalid OPENROUTER_MODEL metadata: {err}"))?;
    let session_store_factory = Arc::new(lash_sqlite_store::SqliteSessionStoreFactory::new(
        data_dir.join("lash-sessions"),
    ));
    let core_store_factory: Arc<dyn lash::persistence::SessionStoreFactory> =
        session_store_factory.clone();
    let process_registry = Arc::new(
        lash_sqlite_store::SqliteProcessRegistry::open(&data_dir.join("processes.db"))
            .await
            .context("open process registry")?,
    ) as Arc<dyn lash::process::ProcessRegistry>;
    let host_event_store = Arc::new(
        lash_sqlite_store::SqliteHostEventStore::open(&data_dir.join("host-events.db"))
            .await
            .context("open host event store")?,
    );
    // Deployment-level Lashlang artifact store (compiled trigger/process
    // modules), shared across the session tree. SQLite keeps installed triggers
    // durable across restarts.
    let artifact_store = Arc::new(
        lash_sqlite_store::Store::open(&data_dir.join("artifacts.db"))
            .await
            .context("open lashlang artifact store")?,
    ) as Arc<dyn lash::persistence::LashlangArtifactStore>;
    let subagent_registry = Arc::new(lash_subagents::default_registry(&BTreeMap::new()));
    let mail_world = mail::MailWorld::new();
    let session_ids = WorkbenchSessionIds::fresh();
    let (event_tx, _) = broadcast::channel(1024);
    let restate_http = reqwest::Client::new();
    let process_deployment =
        lash_restate::RestateProcessDeployment::new(restate_ingress_url.clone(), process_registry);
    let queued_work_runner =
        lash::runtime::QueuedWorkRunner::new(Arc::new(WorkbenchQueuedWorkSubmitter {
            session_ids: session_ids.clone(),
            store_factory: Arc::clone(&core_store_factory),
            restate_ingress_url: restate_ingress_url.clone(),
            restate_http: restate_http.clone(),
        }));
    let queued_work_poke = queued_work_runner.poke_handle();

    let core = LashCore::builder()
        .install_mode(ModePreset::rlm_with_config(
            lash::modes::RlmProtocolPluginConfig::default()
                .with_lashlang_abilities(workbench_lashlang_abilities()),
        ))
        .default_mode(ModeId::rlm())
        .provider(provider)
        .model(model_spec)
        .store_factory(Arc::clone(&core_store_factory))
        .attachment_store(Arc::new(lash::persistence::FileAttachmentStore::new(
            data_dir.join("attachments"),
        )))
        .lashlang_artifact_store(artifact_store)
        .host_event_store(host_event_store)
        .trace_sink(Arc::clone(&trace_sink))
        .lashlang_execution_sink(Arc::clone(&lashlang_execution_sink))
        .trace_level(TraceLevel::Extended)
        .configure_plugins(|plugins| {
            plugins.push(Arc::new(
                WorkbenchPluginFactory::new(tavily_api_key.clone())
                    .with_mail_world(mail_world.clone()),
            ));
            plugins.push(Arc::new(
                lash_plugin_process_controls::ProcessControlsPluginFactory::new(),
            ));
            plugins.push(Arc::new(
                lash_subagents::SubagentsPluginFactory::new(subagent_registry)
                    .with_session_spec(SessionSpec::inherit()),
            ));
        })
        .effect_host(Arc::new(lash_restate::RestateEffectHost::new()))
        .process_work_driver(process_deployment.process_work_driver())
        .queued_work_poke(queued_work_poke.clone())
        .build()
        .context("build Lash core")?;
    let process_worker = lash::durability::DurableProcessWorker::new(
        core.durable_process_worker_config()
            .context("build Restate process worker config")?,
    );
    let process_observer = core
        .process_observer()
        .expect("process observer configured")
        .clone();

    let state = AppState {
        core,
        process_observer,
        session_store_factory: Arc::clone(&core_store_factory),
        session_ids,
        messages: Arc::new(Mutex::new(Vec::new())),
        timeline: Arc::new(Mutex::new(Vec::new())),
        selected_model: Arc::new(Mutex::new(ModelSelection {
            model,
            model_variant: Some(model_variant),
        })),
        web_configured: !tavily_api_key.trim().is_empty(),
        trace_sink: Some(Arc::clone(&trace_sink)),
        lashlang_execution,
        event_tx,
        queued_work_poke,
        restate_ingress_url,
        restate_http,
        restate_cron_job_keys: Arc::new(Mutex::new(BTreeSet::new())),
        mail_world,
    };
    restate::spawn_restate_endpoint(
        restate_endpoint_addr,
        state.clone(),
        process_deployment,
        process_worker,
    );
    queued_work_runner.spawn();
    emit_workbench_trace(
        &state.trace_sink,
        None,
        "startup",
        json!({
            "addr": addr.to_string(),
            "data_dir": data_dir.display().to_string(),
            "trace_path": trace_path_display,
            "lashlang_execution_path": lashlang_execution_path.display().to_string(),
            "model": serde_json::to_value(state.selected_model()).unwrap_or(Value::Null),
            "web_configured": state.web_configured,
            "restate_endpoint_addr": restate_endpoint_addr.to_string(),
            "restate_ingress_url": state.restate_ingress_url,
        }),
    );

    let app = Router::new()
        .route("/", get(index))
        .route("/api/state", get(app_state))
        .route("/api/events", get(session_events))
        .route("/api/turn", post(send_turn))
        .route("/api/reset", post(reset_chat))
        .route("/api/button-trigger", post(button_trigger))
        .route("/api/accounts", get(list_accounts).post(add_account))
        .route("/api/accounts/{slug}", delete(delete_account))
        .route("/api/accounts/{slug}/messages", post(inject_message))
        .route("/api/accounts/{slug}/messages/{id}", delete(delete_message))
        .route("/api/accounts/{slug}/inbox", get(account_inbox))
        .route("/api/work", get(list_work))
        .route("/api/lashlang-graphs", get(list_lashlang_graphs))
        .route("/api/lashlang-graph/{graph_key}", get(lashlang_graph))
        .with_state(state);

    println!("agent-workbench listening on http://{addr}");
    println!("agent-workbench Restate endpoint listening on http://{restate_endpoint_addr}");
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .context("bind listener")?;
    axum::serve(listener, app).await.context("serve")?;
    Ok(())
}

#[derive(Clone)]
struct AppState {
    core: LashCore,
    process_observer: lash::process::ProcessWorkObserver,
    session_store_factory: Arc<dyn lash::persistence::SessionStoreFactory>,
    session_ids: WorkbenchSessionIds,
    messages: Arc<Mutex<Vec<ChatMessage>>>,
    timeline: Arc<Mutex<Vec<StreamItem>>>,
    selected_model: Arc<Mutex<ModelSelection>>,
    web_configured: bool,
    trace_sink: Option<Arc<dyn TraceSink>>,
    lashlang_execution: Arc<TraceLashlangGraphStore>,
    event_tx: broadcast::Sender<StreamItem>,
    queued_work_poke: lash::runtime::QueuedWorkPoke,
    restate_ingress_url: String,
    restate_http: reqwest::Client,
    restate_cron_job_keys: Arc<Mutex<BTreeSet<String>>>,
    mail_world: mail::MailWorld,
}

#[derive(Clone, Debug, Serialize)]
struct Settings {
    model: String,
    model_variant: Option<String>,
    web_configured: bool,
    model_variants: Vec<&'static str>,
    session_id: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ModelSelection {
    model: String,
    model_variant: Option<String>,
}

impl ModelSelection {
    fn from_spec(model: &lash::ModelSpec) -> Self {
        Self {
            model: model.id.clone(),
            model_variant: model.variant.clone(),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
struct StateSnapshot {
    settings: Settings,
    messages: Vec<ChatMessage>,
    timeline: Vec<StreamItem>,
}

#[derive(Clone, Debug, Serialize)]
struct ChatMessage {
    id: String,
    role: String,
    text: String,
    at: String,
}

#[derive(Debug, Deserialize)]
struct TurnRequest {
    text: String,
    model: Option<String>,
    model_variant: Option<String>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub(crate) enum ButtonChoice {
    Red,
    Blue,
}

impl ButtonChoice {
    fn as_str(self) -> &'static str {
        match self {
            Self::Red => "Red",
            Self::Blue => "Blue",
        }
    }

    fn lower(self) -> &'static str {
        match self {
            Self::Red => "red",
            Self::Blue => "blue",
        }
    }
}

#[derive(Debug, Deserialize)]
struct ButtonEventRequest {
    button: ButtonChoice,
    model: Option<String>,
    model_variant: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AddAccountRequest {
    name: String,
}

#[derive(Debug, Deserialize)]
struct InjectMessageRequest {
    title: Option<String>,
    text: Option<String>,
    model: Option<String>,
    model_variant: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum StreamItem {
    Event { event: Box<TurnActivity> },
    Message { message: ChatMessage },
    Error { message: String },
    Done,
}

impl StreamItem {
    fn event(event: TurnActivity) -> Self {
        Self::Event {
            event: Box::new(event),
        }
    }
}

#[derive(Debug, Serialize)]
struct CommandAccepted {
    accepted: bool,
}

#[derive(Clone)]
struct WorkbenchQueuedWorkSubmitter {
    session_ids: WorkbenchSessionIds,
    store_factory: Arc<dyn lash::persistence::SessionStoreFactory>,
    restate_ingress_url: String,
    restate_http: reqwest::Client,
}

#[async_trait]
impl lash::runtime::QueuedWorkRunHandle for WorkbenchQueuedWorkSubmitter {
    async fn run_queued_work(
        &self,
        request: lash::runtime::QueuedWorkRunRequest,
    ) -> std::result::Result<lash::runtime::QueuedWorkRunOutcome, PluginError> {
        let session_id = request
            .session_id
            .unwrap_or_else(|| self.session_ids.current());
        if !self.has_queued_work(&session_id).await? {
            return Ok(lash::runtime::QueuedWorkRunOutcome::Idle);
        }
        let workflow_request = restate::WorkbenchQueuedTurnWorkflowRequest {
            turn_id: format!("workbench-queued-{}", uuid::Uuid::new_v4()),
            session_id: session_id.clone(),
            reason: request.reason,
        };
        restate::submit_queued_turn_request(
            &self.restate_http,
            &self.restate_ingress_url,
            &workflow_request,
        )
        .await
        .map_err(|err| PluginError::Session(err.to_string()))?;
        Ok(lash::runtime::QueuedWorkRunOutcome::Submitted { session_id })
    }
}

impl WorkbenchQueuedWorkSubmitter {
    async fn has_queued_work(&self, session_id: &str) -> std::result::Result<bool, PluginError> {
        let store = self
            .store_factory
            .create_store(&lash::persistence::SessionStoreCreateRequest {
                session_id: session_id.to_string(),
                relation: lash::persistence::SessionRelation::default(),
                policy: lash::runtime::SessionPolicy::default(),
            })
            .await
            .map_err(PluginError::Session)?;
        let queued = store
            .list_queued_work(session_id)
            .await
            .map_err(|err| PluginError::Session(err.to_string()))?;
        Ok(!queued.is_empty())
    }
}

#[cfg(test)]
struct NoopQueuedWorkRunHandle;

#[cfg(test)]
#[async_trait]
impl lash::runtime::QueuedWorkRunHandle for NoopQueuedWorkRunHandle {
    async fn run_queued_work(
        &self,
        _request: lash::runtime::QueuedWorkRunRequest,
    ) -> std::result::Result<lash::runtime::QueuedWorkRunOutcome, PluginError> {
        Ok(lash::runtime::QueuedWorkRunOutcome::Idle)
    }
}

#[cfg(test)]
fn inert_queued_work_poke() -> lash::runtime::QueuedWorkPoke {
    lash::runtime::QueuedWorkRunner::new(Arc::new(NoopQueuedWorkRunHandle)).poke_handle()
}

#[derive(Debug, Serialize)]
struct WorkItem {
    process: WorkProcess,
    descriptor: lash::process::ProcessHandleDescriptor,
    events: Vec<WorkEvent>,
    kind: String,
    label: String,
}

#[derive(Debug, Serialize)]
struct WorkProcess {
    process_id: String,
    graph_key: String,
    lifecycle: lash::process::ProcessLifecycleStatus,
    status_label: String,
    terminal: bool,
    error: Option<String>,
    created_at_ms: u64,
    updated_at_ms: u64,
    input: Value,
    external_ref: Option<Value>,
    child_session_id: Option<String>,
    label: String,
}

#[derive(Debug, Serialize)]
struct WorkEvent {
    sequence: u64,
    event_type: String,
    occurred_at_ms: u64,
    payload: Value,
}

async fn index() -> Html<&'static str> {
    Html(ui::INDEX_HTML)
}

async fn app_state(State(state): State<AppState>) -> Json<StateSnapshot> {
    Json(StateSnapshot {
        settings: state.settings(),
        messages: state.messages_snapshot(),
        timeline: state.timeline_snapshot(),
    })
}

async fn session_events(State(state): State<AppState>) -> Response {
    let mut events = state.event_tx.subscribe();
    let (tx, rx) = mpsc::channel::<StreamItem>(64);
    tokio::spawn(async move {
        loop {
            match events.recv().await {
                Ok(item) => {
                    if tx.send(item).await.is_err() {
                        break;
                    }
                }
                Err(broadcast::error::RecvError::Lagged(count)) => {
                    let _ = tx
                        .send(StreamItem::Error {
                            message: format!("event stream skipped {count} updates"),
                        })
                        .await;
                }
                Err(broadcast::error::RecvError::Closed) => break,
            }
        }
    });
    ndjson_response(rx)
}

async fn send_turn(
    State(state): State<AppState>,
    Json(request): Json<TurnRequest>,
) -> Result<Json<CommandAccepted>, AppError> {
    let text = request.text.trim().to_string();
    if text.is_empty() {
        return Err(AppError::bad_request("message text is required"));
    }
    let turn_model = model_spec_for_request(
        &state,
        request.model.as_deref(),
        request.model_variant.as_deref(),
    )?;
    state.trace(
        "api.turn.request",
        json!({
            "text": text.clone(),
            "model": serde_json::to_value(&turn_model).unwrap_or(Value::Null),
        }),
    );
    state.push_message("user", text.clone());
    state.set_selected_model(ModelSelection::from_spec(&turn_model));
    let turn_id = format!("workbench-turn-{}", uuid::Uuid::new_v4());
    restate::submit_user_turn(
        &state,
        restate::WorkbenchTurnWorkflowRequest {
            turn_id,
            session_id: state.current_session_id(),
            text,
            model: ModelSelection::from_spec(&turn_model),
        },
    )
    .await?;
    Ok(Json(CommandAccepted { accepted: true }))
}

async fn button_trigger(
    State(state): State<AppState>,
    Json(request): Json<ButtonEventRequest>,
) -> Result<Json<CommandAccepted>, AppError> {
    let turn_model = model_spec_for_request(
        &state,
        request.model.as_deref(),
        request.model_variant.as_deref(),
    )?;
    let model = ModelSelection::from_spec(&turn_model);
    state.set_selected_model(model.clone());
    state.trace(
        "api.button_trigger.request",
        json!({
            "button": request.button,
            "model": serde_json::to_value(&turn_model).unwrap_or(Value::Null),
        }),
    );
    let pressed_at = Utc::now().to_rfc3339();
    state.push_message("event", format!("{} button event", request.button.lower()));
    restate::submit_button_trigger(
        &state,
        restate::WorkbenchButtonTriggerWorkflowRequest {
            operation_id: format!("workbench-button-{}", uuid::Uuid::new_v4()),
            session_id: state.current_session_id(),
            button: request.button,
            model,
            pressed_at,
        },
    )
    .await?;
    Ok(Json(CommandAccepted { accepted: true }))
}

async fn list_accounts(State(state): State<AppState>) -> Json<Vec<mail::AccountSummary>> {
    Json(state.mail_world.account_summaries())
}

async fn add_account(
    State(state): State<AppState>,
    Json(request): Json<AddAccountRequest>,
) -> Result<Json<mail::AccountSummary>, AppError> {
    let summary = state
        .mail_world
        .add_account(&request.name)
        .map_err(AppError::bad_request)?;
    state.trace(
        "api.accounts.add",
        json!({ "slug": summary.slug, "authority": summary.authority }),
    );
    refresh_persisted_tool_surface(&state, "account_added").await?;
    state.push_message(
        "event",
        format!("connected mock account `{}`", summary.authority),
    );
    Ok(Json(summary))
}

async fn delete_account(
    AxumPath(slug): AxumPath<String>,
    State(state): State<AppState>,
) -> Result<Json<CommandAccepted>, AppError> {
    state
        .mail_world
        .remove_account(&slug)
        .map_err(AppError::not_found)?;
    state.trace("api.accounts.remove", json!({ "slug": slug }));
    refresh_persisted_tool_surface(&state, "account_removed").await?;
    state.push_message("event", format!("removed mock account `inbox.{slug}`"));
    Ok(Json(CommandAccepted { accepted: true }))
}

async fn delete_message(
    AxumPath((slug, id)): AxumPath<(String, String)>,
    State(state): State<AppState>,
) -> Result<Json<CommandAccepted>, AppError> {
    state
        .mail_world
        .remove_message(&slug, &id)
        .map_err(AppError::not_found)?;
    state.trace(
        "api.accounts.message.delete",
        json!({ "account": slug, "id": id }),
    );
    Ok(Json(CommandAccepted { accepted: true }))
}

async fn account_inbox(
    AxumPath(slug): AxumPath<String>,
    State(state): State<AppState>,
) -> Result<Json<Vec<mail::MailMessage>>, AppError> {
    let inbox = state.mail_world.inbox(&slug).map_err(AppError::not_found)?;
    Ok(Json(inbox))
}

async fn refresh_persisted_tool_surface(state: &AppState, reason: &str) -> Result<(), AppError> {
    let session_id = state.current_session_id();
    let session = state
        .core
        .session(session_id.clone())
        .rlm()
        .open()
        .await
        .map_err(AppError::internal)?;
    let expected_generation = session
        .tools()
        .state()
        .await
        .map_err(AppError::internal)?
        .generation();
    let receipt = session
        .commands()
        .refresh_tool_surface(
            reason,
            Some(expected_generation),
            format!(
                "workbench-refresh-tool-surface:{}:{}:{}",
                session_id,
                reason,
                uuid::Uuid::new_v4()
            ),
        )
        .await
        .map_err(AppError::internal)?;
    let _ = session
        .queued_turn()
        .drain_id(receipt.batch_id.clone())
        .run()
        .await
        .map_err(AppError::internal)?;
    let persisted = session
        .control()
        .state()
        .persist_current()
        .await
        .map_err(AppError::internal)?;
    let tool_count = persisted
        .tool_state_snapshot
        .as_ref()
        .map(lash::tools::ToolState::len)
        .unwrap_or_default();
    let store = state
        .session_store_factory
        .create_store(&lash::persistence::SessionStoreCreateRequest {
            session_id: session_id.clone(),
            relation: lash::persistence::SessionRelation::default(),
            policy: persisted.policy.clone(),
        })
        .await
        .map_err(AppError::internal)?;
    let result = store
        .commit_runtime_state(lash::persistence::RuntimeCommit::persisted_state(
            &persisted,
            &[],
        ))
        .await
        .map_err(AppError::internal)?;
    session.close().await.map_err(AppError::internal)?;
    state.trace(
        "mail.tool_surface.refresh",
        json!({
            "reason": reason,
            "session_id": session_id,
            "command_batch_id": receipt.batch_id,
            "command_source_key": receipt.source_key,
            "tool_state_generation": persisted.tool_state_generation,
            "tool_count": tool_count,
            "head_revision": result.head_revision,
        }),
    );
    Ok(())
}

async fn inject_message(
    AxumPath(slug): AxumPath<String>,
    State(state): State<AppState>,
    Json(request): Json<InjectMessageRequest>,
) -> Result<Json<CommandAccepted>, AppError> {
    let turn_model = model_spec_for_request(
        &state,
        request.model.as_deref(),
        request.model_variant.as_deref(),
    )?;
    let model = ModelSelection::from_spec(&turn_model);
    state.set_selected_model(model.clone());
    let delivered = state
        .mail_world
        .deliver(
            &slug,
            request.title.as_deref().unwrap_or_default(),
            request.text.as_deref().unwrap_or_default(),
        )
        .map_err(AppError::not_found)?;
    let message = delivered.message;
    let delivery = delivered.delivery;
    state.trace(
        "api.accounts.inject",
        json!({ "account": slug, "title": message.title }),
    );
    state.push_message(
        "event",
        format!("message delivered to `inbox.{}`: {}", slug, message.title),
    );
    restate::submit_mail_received(
        &state,
        restate::WorkbenchMailReceivedWorkflowRequest {
            operation_id: format!("workbench-mail-{}", uuid::Uuid::new_v4()),
            session_id: state.current_session_id(),
            model,
            delivery,
        },
    )
    .await?;
    Ok(Json(CommandAccepted { accepted: true }))
}

async fn reset_chat(State(state): State<AppState>) -> Result<Json<StateSnapshot>, AppError> {
    let old_session_id = state.current_session_id();
    restate::cancel_known_cron_jobs(&state, "reset").await?;
    restate::submit_session_delete(
        &state,
        restate::WorkbenchSessionDeleteWorkflowRequest {
            operation_id: format!("workbench-delete-{}", uuid::Uuid::new_v4()),
            session_id: old_session_id.clone(),
        },
    )
    .await?;
    let (rotated_old, _) = state.session_ids.rotate();
    if rotated_old != old_session_id {
        eprintln!(
            "warning: workbench session changed during reset; deleted {old_session_id}, rotated {rotated_old}"
        );
    }
    let new_session_id = state.current_session_id();
    state.trace(
        "api.reset",
        json!({
            "old_session_id": old_session_id,
            "new_session_id": new_session_id.clone(),
        }),
    );
    let session = state
        .core
        .session(new_session_id)
        .rlm()
        .open()
        .await
        .map_err(AppError::internal)?;
    let selected_model = model_spec_from_selection(state.selected_model());
    session
        .configure(lash::SessionConfigPatch {
            model: Some(selected_model),
            ..lash::SessionConfigPatch::default()
        })
        .await
        .map_err(AppError::internal)?;
    state.messages.lock().expect("messages lock").clear();
    state.timeline.lock().expect("timeline lock").clear();
    state.lashlang_execution.clear();
    Ok(Json(StateSnapshot {
        settings: state.settings(),
        messages: Vec::new(),
        timeline: Vec::new(),
    }))
}

async fn list_work(State(state): State<AppState>) -> Result<Json<Vec<WorkItem>>, AppError> {
    let session_id = state.current_session_id();
    let snapshot = state
        .process_observer
        .snapshot_for_session(session_id)
        .await
        .map_err(AppError::internal)?;
    let work = snapshot
        .items
        .into_iter()
        .map(work_item_from_observed)
        .collect::<Vec<_>>();
    state.trace(
        "api.work.response",
        json!({
            "count": work.len(),
            "items": work.iter().map(trace_work_item).collect::<Vec<_>>(),
        }),
    );
    Ok(Json(work))
}

async fn list_lashlang_graphs(
    State(state): State<AppState>,
) -> Result<Json<execution_graphs::LashlangGraphIndex>, AppError> {
    let index = execution_graphs::index_for_session(
        &state.process_observer,
        &state.current_session_id(),
        state.lashlang_execution.graphs(),
    )
    .await?;
    Ok(Json(index))
}

async fn lashlang_graph(
    AxumPath(graph_key): AxumPath<String>,
    State(state): State<AppState>,
) -> Result<Json<TraceLashlangGraph>, AppError> {
    let graph = execution_graphs::visible_graph_by_key(
        &state.process_observer,
        &state.current_session_id(),
        state.lashlang_execution.graphs(),
        &graph_key,
    )
    .await?;
    Ok(Json(graph))
}

fn ndjson_response(rx: mpsc::Receiver<StreamItem>) -> Response {
    let stream = ReceiverStream::new(rx).map(|item| {
        let mut line = serde_json::to_string(&item).unwrap_or_else(|err| {
            json!({
                "type": "error",
                "message": err.to_string(),
            })
            .to_string()
        });
        line.push('\n');
        Ok::<Bytes, Infallible>(Bytes::from(line))
    });

    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/x-ndjson; charset=utf-8")
        .header(header::CACHE_CONTROL, "no-store")
        .body(Body::from_stream(stream))
        .expect("valid streaming response")
}

#[derive(Default)]
struct TurnStreamState {
    assistant_prose: String,
}

struct ChannelTurnEvents {
    state: AppState,
    turn_state: Arc<Mutex<TurnStreamState>>,
}

#[async_trait]
impl TurnActivitySink for ChannelTurnEvents {
    async fn emit(&self, activity: TurnActivity) {
        if let TurnEvent::AssistantProseDelta { text } = &activity.event {
            self.turn_state
                .lock()
                .expect("turn state lock")
                .assistant_prose
                .push_str(text);
        }
        self.state.publish(StreamItem::event(activity));
    }
}

pub(crate) async fn enqueue_button_host_event_command(
    state: &AppState,
    button: ButtonChoice,
    pressed_at: &str,
    operation_id: &str,
    scoped_effect_controller: lash::runtime::ScopedEffectController<'_>,
) -> AnyhowResult<lash::host_events::HostEventEmitReport> {
    let payload = json!({
        "pressed_at": pressed_at,
        "button": button.as_str(),
        "message": format!("user pressed the {} button", button.lower()),
    });
    let source_key = lash::host_events::empty_host_event_source_key(BUTTON_TRIGGER_SOURCE_TYPE)
        .context("button source key")?;
    state.trace(
        "host_event.emit",
        json!({
            "resource_type": BUTTON_TRIGGER_RESOURCE,
            "alias": BUTTON_TRIGGER_ALIAS,
            "event": BUTTON_TRIGGER_EVENT,
            "source_type": BUTTON_TRIGGER_SOURCE_TYPE,
            "source_key": source_key,
            "payload": payload.clone(),
        }),
    );
    state
        .core
        .host_events()
        .emit(
            lash::host_events::HostEventOccurrenceRequest::new(
                BUTTON_TRIGGER_SOURCE_TYPE,
                source_key,
                payload,
                format!("workbench-button-host-event:{operation_id}"),
            )
            .with_source(json!({})),
            scoped_effect_controller,
        )
        .await
        .context("emit button host event occurrence")
}

pub(crate) async fn enqueue_mail_received_host_event_command(
    state: &AppState,
    message: &mail::MailDelivery,
    operation_id: &str,
    scoped_effect_controller: lash::runtime::ScopedEffectController<'_>,
) -> AnyhowResult<lash::host_events::HostEventEmitReport> {
    let payload = json!({
        "account": message.account,
        "title": message.title,
        "text": message.text,
    });
    let source_key = lash::host_events::empty_host_event_source_key(MAIL_RECEIVED_SOURCE_TYPE)
        .context("mail source key")?;
    state.trace(
        "host_event.emit",
        json!({
            "resource_type": MAIL_EVENT_RESOURCE,
            "alias": MAIL_EVENT_ALIAS,
            "event": MAIL_EVENT_EVENT,
            "source_type": MAIL_RECEIVED_SOURCE_TYPE,
            "source_key": source_key,
            "payload": payload.clone(),
        }),
    );
    state
        .core
        .host_events()
        .emit(
            lash::host_events::HostEventOccurrenceRequest::new(
                MAIL_RECEIVED_SOURCE_TYPE,
                source_key,
                payload,
                format!("workbench-mail-host-event:{operation_id}"),
            )
            .with_source(json!({})),
            scoped_effect_controller,
        )
        .await
        .context("emit mail received host event occurrence")
}

fn button_trigger_event_type() -> lashlang::NamedDataType {
    lashlang::NamedDataType::object(
        "ui.button.Pressed",
        vec![
            lashlang::TypeField {
                name: "button".into(),
                ty: lashlang::TypeExpr::Union(vec![
                    lashlang::TypeExpr::Enum(vec!["Red".into()]),
                    lashlang::TypeExpr::Enum(vec!["Blue".into()]),
                ]),
                optional: false,
            },
            lashlang::TypeField {
                name: "message".into(),
                ty: lashlang::TypeExpr::Str,
                optional: false,
            },
            lashlang::TypeField {
                name: "pressed_at".into(),
                ty: lashlang::TypeExpr::Str,
                optional: false,
            },
        ],
    )
    .expect("valid button trigger event type")
}

fn workbench_lashlang_abilities() -> lashlang::LashlangAbilities {
    lashlang::LashlangAbilities::default()
        .with_processes()
        .with_sleep()
        .with_process_signals()
        .with_triggers()
}

impl AppState {
    fn current_session_id(&self) -> String {
        self.session_ids.current()
    }

    fn selected_model(&self) -> ModelSelection {
        self.selected_model
            .lock()
            .expect("selected model lock")
            .clone()
    }

    fn set_selected_model(&self, model: ModelSelection) {
        *self.selected_model.lock().expect("selected model lock") = model;
    }

    fn settings(&self) -> Settings {
        let selected_model = self.selected_model();
        Settings {
            model: selected_model.model,
            model_variant: selected_model.model_variant,
            web_configured: self.web_configured,
            model_variants: vec!["", "low", "medium", "high"],
            session_id: self.current_session_id(),
        }
    }

    fn messages_snapshot(&self) -> Vec<ChatMessage> {
        self.messages.lock().expect("messages lock").clone()
    }

    fn timeline_snapshot(&self) -> Vec<StreamItem> {
        self.timeline.lock().expect("timeline lock").clone()
    }

    fn trace(&self, name: &str, payload: Value) {
        emit_workbench_trace(
            &self.trace_sink,
            Some(self.current_session_id()),
            name,
            payload,
        );
    }

    fn publish(&self, item: StreamItem) {
        if !matches!(item, StreamItem::Done) {
            self.timeline
                .lock()
                .expect("timeline lock")
                .push(item.clone());
        }
        let _ = self.event_tx.send(item);
    }

    fn push_message(&self, role: impl Into<String>, text: impl Into<String>) -> ChatMessage {
        let message = ChatMessage {
            id: uuid::Uuid::new_v4().to_string(),
            role: role.into(),
            text: text.into(),
            at: Utc::now().to_rfc3339(),
        };
        self.messages
            .lock()
            .expect("messages lock")
            .push(message.clone());
        self.publish(StreamItem::Message {
            message: message.clone(),
        });
        message
    }
}

fn emit_workbench_trace(
    sink: &Option<Arc<dyn TraceSink>>,
    session_id: Option<String>,
    name: &str,
    payload: Value,
) {
    let Some(sink) = sink else {
        return;
    };
    let context = session_id
        .map(|session_id| TraceContext::default().for_session(session_id))
        .unwrap_or_default();
    let record = TraceRecord::new(
        context,
        TraceEvent::Custom {
            name: format!("agent_workbench.{name}"),
            payload,
        },
    );
    if let Err(err) = sink.append(&record) {
        eprintln!("warning: failed to append agent-workbench trace event `{name}`: {err}");
    }
}

fn trace_work_item(item: &WorkItem) -> Value {
    json!({
        "process_id": item.process.process_id.clone(),
        "graph_key": item.process.graph_key.clone(),
        "kind": item.kind.clone(),
        "label": item.label.clone(),
        "status_label": item.process.status_label.clone(),
        "terminal": item.process.terminal,
        "created_at_ms": item.process.created_at_ms,
        "updated_at_ms": item.process.updated_at_ms,
        "input": item.process.input.clone(),
        "events": item.events.iter().map(|event| {
            json!({
                "sequence": event.sequence,
                "event_type": event.event_type.clone(),
                "occurred_at_ms": event.occurred_at_ms,
                "payload": event.payload.clone(),
            })
        }).collect::<Vec<_>>(),
    })
}

#[derive(Clone, Debug)]
struct WorkbenchSessionIds {
    current: Arc<Mutex<String>>,
}

impl WorkbenchSessionIds {
    fn fresh() -> Self {
        Self {
            current: Arc::new(Mutex::new(new_session_id())),
        }
    }

    fn current(&self) -> String {
        self.current.lock().expect("session id lock").clone()
    }

    fn rotate(&self) -> (String, String) {
        let mut current = self.current.lock().expect("session id lock");
        let old = current.clone();
        let new = new_session_id();
        *current = new.clone();
        (old, new)
    }
}

fn new_session_id() -> String {
    format!("{SESSION_ID_PREFIX}-{}", uuid::Uuid::new_v4().simple())
}

fn model_spec_for_request(
    state: &AppState,
    model: Option<&str>,
    model_variant: Option<&str>,
) -> Result<lash::ModelSpec, AppError> {
    let selected_model = state.selected_model();
    let model = model
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(selected_model.model.as_str())
        .to_string();
    let model_variant = model_variant_for_request(&selected_model, model_variant);
    lash::ModelSpec::from_token_limits(model, model_variant, DEFAULT_CONTEXT_WINDOW_TOKENS, None)
        .map_err(AppError::bad_request)
}

fn model_variant_for_request(
    selected_model: &ModelSelection,
    model_variant: Option<&str>,
) -> Option<String> {
    match model_variant {
        Some(value) => {
            let value = value.trim();
            if value.is_empty() {
                None
            } else {
                Some(value.to_string())
            }
        }
        None => selected_model.model_variant.clone(),
    }
}

fn model_spec_from_selection(selection: ModelSelection) -> lash::ModelSpec {
    lash::ModelSpec::from_token_limits(
        selection.model,
        selection.model_variant,
        DEFAULT_CONTEXT_WINDOW_TOKENS,
        None,
    )
    .expect("workbench model selection should use a valid token limit")
}

async fn apply_model_selection_to_session(
    state: &AppState,
    session: &lash::LashSession,
    model: lash::ModelSpec,
    reason: &str,
) -> Result<(), AppError> {
    state.set_selected_model(ModelSelection::from_spec(&model));
    session
        .configure(lash::SessionConfigPatch {
            model: Some(model.clone()),
            ..lash::SessionConfigPatch::default()
        })
        .await
        .map_err(AppError::internal)?;
    state.trace(
        "model_selection.applied",
        json!({
            "reason": reason,
            "model": serde_json::to_value(&model).unwrap_or(Value::Null),
        }),
    );
    Ok(())
}

fn assistant_text_for_display(output: &TurnResult, streamed_prose: &str) -> String {
    let terminal = output
        .submitted_value()
        .map(terminal_value_text)
        .or_else(|| {
            output
                .tool_value()
                .map(|(_tool_name, value)| terminal_value_text(value))
        });
    let assistant = (!streamed_prose.trim().is_empty())
        .then(|| streamed_prose.to_string())
        .or_else(|| {
            output
                .assistant_message()
                .filter(|text| !text.trim().is_empty())
                .map(str::to_string)
        });
    combine_assistant_display_parts(assistant, terminal)
}

fn combine_assistant_display_parts(assistant: Option<String>, terminal: Option<String>) -> String {
    let assistant = assistant.filter(|text| !text.trim().is_empty());
    let terminal = terminal.filter(|text| !text.trim().is_empty());
    match (assistant, terminal) {
        (Some(assistant), Some(terminal)) if assistant.trim() == terminal.trim() => assistant,
        (Some(assistant), Some(terminal)) => format!("{}\n\n{}", assistant.trim_end(), terminal),
        (Some(assistant), None) => assistant,
        (None, Some(terminal)) => terminal,
        (None, None) => String::new(),
    }
}

fn terminal_value_text(value: &Value) -> String {
    value
        .as_str()
        .map(str::to_string)
        .unwrap_or_else(|| value.to_string())
}

fn compact_payload(value: Value) -> Value {
    match value {
        Value::String(text) if text.len() > 1_200 => Value::String(truncate_chars(&text, 1_200)),
        Value::Array(items) => {
            Value::Array(items.into_iter().take(12).map(compact_payload).collect())
        }
        Value::Object(map) => Value::Object(
            map.into_iter()
                .map(|(key, value)| (key, compact_payload(value)))
                .collect(),
        ),
        other => other,
    }
}

fn truncate_chars(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    format!("{}...", text.chars().take(max_chars).collect::<String>())
}

fn work_item_from_observed(item: lash::process::ObservedWorkItem) -> WorkItem {
    WorkItem {
        process: work_process_from_observed(item.process),
        descriptor: item.descriptor,
        events: item
            .events
            .into_iter()
            .map(work_event_from_observed)
            .collect(),
        kind: item.kind,
        label: item.label,
    }
}

fn work_process_from_observed(process: lash::process::ObservedProcess) -> WorkProcess {
    WorkProcess {
        process_id: process.process_id,
        graph_key: process.graph_key,
        lifecycle: process.lifecycle,
        status_label: process.status_label,
        terminal: process.terminal,
        error: process.error,
        created_at_ms: process.created_at_ms,
        updated_at_ms: process.updated_at_ms,
        input: compact_payload(serde_json::to_value(process.input).unwrap_or(Value::Null)),
        external_ref: process
            .external_ref
            .and_then(|value| serde_json::to_value(value).ok())
            .map(compact_payload),
        child_session_id: process.child_session_id,
        label: process.label,
    }
}

fn work_event_from_observed(event: lash::process::ObservedProcessEvent) -> WorkEvent {
    WorkEvent {
        sequence: event.sequence,
        event_type: event.event_type,
        occurred_at_ms: event.occurred_at_ms,
        payload: compact_payload(event.payload),
    }
}

#[derive(Debug)]
struct AppError {
    status: StatusCode,
    message: String,
}

impl AppError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
        }
    }

    fn not_found(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            message: message.into(),
        }
    }

    fn internal(message: impl std::fmt::Display) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: message.to_string(),
        }
    }
}

impl std::fmt::Display for AppError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for AppError {}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(json!({
                "error": self.message,
            })),
        )
            .into_response()
    }
}

struct WorkbenchPluginFactory {
    tavily_api_key: String,
    mail_world: mail::MailWorld,
}

impl WorkbenchPluginFactory {
    fn new(tavily_api_key: impl Into<String>) -> Self {
        Self {
            tavily_api_key: tavily_api_key.into(),
            mail_world: mail::MailWorld::new(),
        }
    }

    fn with_mail_world(mut self, mail_world: mail::MailWorld) -> Self {
        self.mail_world = mail_world;
        self
    }
}

impl PluginFactory for WorkbenchPluginFactory {
    fn id(&self) -> &'static str {
        "agent_workbench"
    }

    fn lashlang_resources(&self) -> lashlang::ResourceCatalog {
        workbench_lashlang_resources()
    }

    fn build(&self, _ctx: &PluginSessionContext) -> Result<Arc<dyn SessionPlugin>, PluginError> {
        Ok(Arc::new(WorkbenchSessionPlugin {
            tavily_api_key: self.tavily_api_key.clone(),
            mail_world: self.mail_world.clone(),
        }))
    }
}

struct WorkbenchSessionPlugin {
    tavily_api_key: String,
    mail_world: mail::MailWorld,
}

impl SessionPlugin for WorkbenchSessionPlugin {
    fn id(&self) -> &'static str {
        "agent_workbench"
    }

    fn register(&self, reg: &mut PluginRegistrar) -> Result<(), PluginError> {
        let mail_world = self.mail_world.clone();
        reg.prompt().contribute(Arc::new(move |_ctx| {
            let mail_world = mail_world.clone();
            Box::pin(async move {
                Ok(vec![PromptContribution::environment(
                    "Agent Workbench",
                    format!(
                        "{WORKBENCH_PROMPT}\n\n{}",
                        connected_accounts_prompt(&mail_world)
                    ),
                )])
            })
        }));
        reg.host_events().declare(HostEvent::new(
            BUTTON_TRIGGER_RESOURCE,
            BUTTON_TRIGGER_ALIAS,
            BUTTON_TRIGGER_EVENT,
            button_trigger_event_type(),
        ))?;
        reg.host_events().declare(HostEvent::new(
            MAIL_EVENT_RESOURCE,
            MAIL_EVENT_ALIAS,
            MAIL_EVENT_EVENT,
            mail_received_event_type(),
        ))?;
        reg.tools()
            .provider(Arc::new(lash_tool_web::web_search_provider(
                self.tavily_api_key.clone(),
            )))?;
        reg.tools()
            .provider(Arc::new(lash_tool_web::fetch_url_provider(
                self.tavily_api_key.clone(),
            )))?;
        reg.tools().provider(Arc::new(mail::MockMailProvider::new(
            self.mail_world.clone(),
        )))?;
        Ok(())
    }
}

fn workbench_lashlang_resources() -> lashlang::ResourceCatalog {
    let mut resources = lashlang::ResourceCatalog::new();
    resources
        .add_trigger_source_constructor(
            CRON_SCHEDULE_SOURCE_TYPE.split('.'),
            lashlang::TypeExpr::Object(vec![
                lashlang::TypeField {
                    name: "expr".into(),
                    ty: lashlang::TypeExpr::Str,
                    optional: false,
                },
                lashlang::TypeField {
                    name: "tz".into(),
                    ty: lashlang::TypeExpr::Str,
                    optional: true,
                },
            ]),
            cron_tick_event_type(),
        )
        .expect("valid cron trigger source");
    resources
        .add_trigger_source_constructor(
            MAIL_RECEIVED_SOURCE_TYPE.split('.'),
            lashlang::TypeExpr::Object(vec![]),
            mail_received_event_type(),
        )
        .expect("valid mail trigger source");
    resources
}

fn mail_received_event_type() -> lashlang::NamedDataType {
    lashlang::NamedDataType::object(
        "mail.Received",
        vec![
            field("account", lashlang::TypeExpr::Str),
            field("title", lashlang::TypeExpr::Str),
            field("text", lashlang::TypeExpr::Str),
        ],
    )
    .expect("valid mail received event type")
}

fn field(name: &str, ty: lashlang::TypeExpr) -> lashlang::TypeField {
    lashlang::TypeField {
        name: name.into(),
        ty,
        optional: false,
    }
}

/// Live, per-turn prompt line naming the inbox authorities that actually exist,
/// so the agent never assumes the illustrative `inbox.work`/`inbox.personal`
/// names from the static guidance are real.
fn connected_accounts_prompt(mail_world: &mail::MailWorld) -> String {
    let accounts = mail_world.account_summaries();
    if accounts.is_empty() {
        return "Connected inbox accounts: none yet. The `inbox` namespace is empty until the \
            user adds an account from the Accounts tab, so `inbox.<anything>` will not resolve. \
            If asked to use an inbox, tell the user to add one first instead of guessing a name."
            .to_string();
    }
    let list = accounts
        .iter()
        .map(|account| account.authority.as_str())
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "Connected inbox authorities right now: {list}. These are the ONLY inbox accounts that \
        exist — use these exact paths and never reference any other `inbox.<name>`. The \
        `inbox.work` / `inbox.personal` names used in the examples above are illustrative only; \
        substitute the real authorities listed here."
    )
}

fn cron_tick_event_type() -> lashlang::NamedDataType {
    lashlang::NamedDataType::object(
        "cron.Tick",
        vec![lashlang::TypeField {
            name: "fired_at".into(),
            ty: lashlang::TypeExpr::Str,
            optional: false,
        }],
    )
    .expect("valid cron tick type")
}

const WORKBENCH_PROMPT: &str = r###"You are running inside the Agent Workbench demo.

Available host features:
- Web access is limited to `web.search(...)` and `web.fetch(...)`, both backed by the same Tavily tools the CLI uses.
- You may call `agents.spawn(...)` for independent investigation.
- You may use Lashlang process definitions for work that should run independently. A `start` creates a process run immediately; a trigger registration is the durable rule that creates future runs when the host emits a matching event.
- When you start a process and need its `finish` value, write `result = (await handle)?`. Bare `await handle` waits, but returns the result wrapper, so `result.field` will not read fields from the finished value.
- To run subagents or slow tool branches in parallel, define one branch process, start every process handle first, then join the handles. Do not write several `x = await agents.spawn(...)` lines and call that parallel:

```lashlang
process research(task: str) {
  result = await agents.spawn({
    task: task,
    capability: "explore",
    output: { summary: "str", key_metrics: "list[str]" }
  })?
  finish result
}

handles = {
  first: start research(task: "Research the first topic"),
  second: start research(task: "Research the second topic")
}
results = await handles
first = results.first?
second = results.second?
submit format("## Results\n\n### First topic\n{}\n\nKey metrics:\n- {}\n\n### Second topic\n{}\n\nKey metrics:\n- {}", first.summary, join(first.key_metrics, "\n- "), second.summary, join(second.key_metrics, "\n- "))
```

- The red and blue UI buttons emit `ui.button.pressed`. Register `ui.button.pressed({})`; the selected button arrives in the event payload, not in the source config:

```lashlang
process on_button(event: ui.button.Pressed) {
  wake { kind: "button_pressed", button: event.button, message: event.message }
  finish true
}

handle = await triggers.register({
  source: ui.button.pressed({}),
  target: on_button,
  inputs: { event: trigger.event },
  name: "button watcher"
})?
registrations = await triggers.list({ name: "button watcher" })?
submit format("Registered button watcher `{}`. Active matching registrations: {}.", handle, len(registrations))
```

- For schedule requests, build `cron.Schedule(...)` values and register a process definition with explicit `inputs`. Use `trigger.event` directly for the `cron.Tick` param, for example `inputs: { tick: trigger.event }`. The workbench syncs enabled `cron.Schedule` registrations to Restate cron objects by stored source key, then emits host-event occurrences with `cron.Tick { fired_at: str }`; use a seconds expression such as `*/10 * * * * *` when the user wants a quick smoke test. Use `await triggers.list({})?` to discover registrations and `await triggers.cancel({ handle: handle })?` to disable future occurrence delivery.

- Mock email accounts the user has connected appear as typed `Inbox` authorities at `inbox.<account>` (for example `inbox.work`, `inbox.personal`). Every account exposes the same three operations:
  - `await inbox.work.send({ title: t, text: b })?` adds a message to that inbox and returns `{ account, id }`. There is no recipient address — a message is just a title and text.
  - `await inbox.work.list({})?` returns `{ account, messages: [{ id, title, text }] }`.
  - `await inbox.work.delete({ id: id })?` removes a message.
  Because they all share the `Inbox` authority type, write account-parametric processes once and start them per account: `process triage(box: Inbox) { items = await box.list({})? wake { kind: "triage", account: items.account, count: len(items.messages) } finish true }` then `start triage(box: inbox.work)`. To sweep several inboxes in parallel, start one handle per account before awaiting any of them.

- When a message is delivered from the Accounts tab or sent with `inbox.<account>.send(...)`, the host emits `mail.received` with payload `mail.Received { account: str, title: str, text: str }`. Register an inbox concierge once and it will fire on every delivery:

```lashlang
process on_mail(event: mail.Received) {
  work = start inbox.work.list({})
  personal = start inbox.personal.list({})
  inboxes = await { work: work, personal: personal }
  wake { kind: "mail_brief", arrived_in: event.account, title: event.title }
  finish true
}

handle = await triggers.register({
  source: mail.received({}),
  target: on_mail,
  inputs: { event: trigger.event },
  name: "inbox concierge"
})?
submit format("Inbox concierge registered as `{}`.", handle)
```

Reference only the `inbox.<account>` authorities that actually exist; if the user has not connected an account yet, ask them to add one from the Accounts tab first.

Use background processes or subagents only when they clarify the user's request or make parallel progress. Keep the visible answer concise and mention any background work you started."###;

#[cfg(test)]
mod tests {
    use super::*;
    use lash::persistence::RuntimePersistence;
    use lash::tracing::{
        TraceBranchSelection, TraceLashlangChildExecution, TraceLashlangEdgeSelection,
        TraceLashlangExecutionEvent, TraceLashlangExecutionIdentity, TraceLashlangGraphChildLink,
        TraceLashlangMap, TraceLashlangMapEdge, TraceLashlangMapNode, TraceLashlangStatus,
        TraceRuntimeScope, TraceRuntimeSubject,
    };
    use std::future::Future;

    fn sync_await<T, F>(future: F) -> T
    where
        T: Send + 'static,
        F: Future<Output = T> + Send + 'static,
    {
        std::thread::spawn(move || {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("tokio runtime")
                .block_on(future)
        })
        .join()
        .expect("runtime thread")
    }

    fn explicit_durable_test_facets(
        builder: lash::LashCoreBuilder,
        data_dir: &std::path::Path,
    ) -> lash::LashCoreBuilder {
        builder
            .effect_host(Arc::new(lash::durability::InlineEffectHost::default()))
            .attachment_store(Arc::new(lash::persistence::FileAttachmentStore::new(
                data_dir.join("attachments"),
            )))
            .lashlang_artifact_store(Arc::new(sync_await({
                let path = data_dir.join("artifacts.db");
                async move {
                    lash_sqlite_store::Store::open(&path)
                        .await
                        .expect("open artifact store")
                }
            })))
            .host_event_store(Arc::new(sync_await({
                let path = data_dir.join("host-events.db");
                async move {
                    lash_sqlite_store::SqliteHostEventStore::open(&path)
                        .await
                        .expect("open host event store")
                }
            })))
    }

    const STACK_BUDGET_BYTES: usize = 2 * 1024 * 1024;

    fn run_async_test_on_stack_budget<F, Fut>(name: &str, test: F)
    where
        F: FnOnce() -> Fut + Send + 'static,
        Fut: Future<Output = ()> + 'static,
    {
        std::thread::Builder::new()
            .name(name.to_string())
            .stack_size(STACK_BUDGET_BYTES)
            .spawn(|| {
                let test = Box::pin(test());
                tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("tokio runtime")
                    .block_on(test)
            })
            .expect("spawn stack-budget test thread")
            .join()
            .expect("stack-budget test thread");
    }

    fn run_async_test_on_stack_budget_multi_thread<F, Fut>(
        name: &str,
        worker_threads: usize,
        test: F,
    ) where
        F: FnOnce() -> Fut + Send + 'static,
        Fut: Future<Output = ()> + 'static,
    {
        std::thread::Builder::new()
            .name(name.to_string())
            .stack_size(STACK_BUDGET_BYTES)
            .spawn(move || {
                let test = Box::pin(test());
                tokio::runtime::Builder::new_multi_thread()
                    .worker_threads(worker_threads)
                    .thread_stack_size(STACK_BUDGET_BYTES)
                    .enable_all()
                    .build()
                    .expect("tokio runtime")
                    .block_on(test)
            })
            .expect("spawn stack-budget multi-thread test thread")
            .join()
            .expect("stack-budget multi-thread test thread");
    }

    fn test_graph(
        graph_key: &str,
        session_id: &str,
        subject: TraceRuntimeSubject,
        children: Vec<TraceLashlangGraphChildLink>,
    ) -> TraceLashlangGraph {
        TraceLashlangGraph {
            graph_key: graph_key.to_string(),
            scope: TraceRuntimeScope::new(session_id),
            subject,
            module_ref: format!("{graph_key}:module"),
            entry_kind: "main".to_string(),
            entry_ref: None,
            entry_name: "main".to_string(),
            status: TraceLashlangStatus::Running,
            nodes: Vec::new(),
            edges: Vec::new(),
            children,
        }
    }

    fn append_started_graph(store: &TraceLashlangGraphStore, graph: &TraceLashlangGraph) {
        let identity = TraceLashlangExecutionIdentity {
            scope: graph.scope.clone(),
            subject: graph.subject.clone(),
            module_ref: graph.module_ref.clone(),
            entry_kind: graph.entry_kind.clone(),
            entry_ref: graph.entry_ref.clone(),
            entry_name: graph.entry_name.clone(),
        };
        store
            .append(&TraceRecord::new(
                TraceContext::default().for_session(graph.scope.session_id.clone()),
                TraceEvent::LashlangExecution {
                    event: TraceLashlangExecutionEvent::ExecutionStarted {
                        event_key: format!("{}:start", graph.graph_key),
                        identity,
                        execution_map: TraceLashlangMap {
                            module_ref: graph.module_ref.clone(),
                            entry_kind: graph.entry_kind.clone(),
                            entry_ref: graph.entry_ref.clone(),
                            entry_name: graph.entry_name.clone(),
                            nodes: Vec::new(),
                            edges: Vec::new(),
                        },
                    },
                },
            ))
            .expect("append test graph");
    }

    #[test]
    fn reset_session_rotation_replaces_workbench_session_id() {
        let ids = WorkbenchSessionIds::fresh();
        let original = ids.current();

        let (old, new) = ids.rotate();

        assert_eq!(old, original);
        assert_eq!(ids.current(), new);
        assert_ne!(old, new);
        assert!(old.starts_with(SESSION_ID_PREFIX));
        assert!(new.starts_with(SESSION_ID_PREFIX));
    }

    #[test]
    fn assistant_display_keeps_streamed_prose_with_terminal_value() {
        assert_eq!(
            combine_assistant_display_parts(
                Some("I started the background checks.".to_string()),
                Some("summary ready".to_string()),
            ),
            "I started the background checks.\n\nsummary ready"
        );
    }

    #[test]
    fn assistant_display_does_not_duplicate_matching_terminal_value() {
        assert_eq!(
            combine_assistant_display_parts(
                Some("summary ready".to_string()),
                Some("summary ready".to_string())
            ),
            "summary ready"
        );
    }

    #[test]
    fn workbench_ui_renders_assistant_markdown() {
        assert!(ui::INDEX_HTML.contains("function renderMarkdownBlocks(markdown)"));
        assert!(ui::INDEX_HTML.contains("setMessageBody(body, message.role, message.text)"));
        assert!(
            ui::INDEX_HTML.contains("draft.innerHTML = renderMarkdownBlocks(assistantDraftText)")
        );
        assert!(ui::INDEX_HTML.contains(".message.assistant .msg-body h1"));
    }

    #[test]
    fn workbench_ui_renders_accounts_panel() {
        assert!(ui::INDEX_HTML.contains("id=\"accountsView\""));
        assert!(ui::INDEX_HTML.contains("data-view=\"accounts\""));
        assert!(ui::INDEX_HTML.contains("id=\"accountAddForm\""));
        assert!(ui::INDEX_HTML.contains("async function loadAccounts"));
        assert!(ui::INDEX_HTML.contains("async function deleteAccount"));
    }

    #[test]
    fn mail_received_event_type_matches_source_type() {
        let resources = workbench_lashlang_resources();
        let binding = resources
            .resolve_trigger_source(MAIL_RECEIVED_SOURCE_TYPE)
            .expect("mail.received source registered");
        assert_eq!(binding.event_type_name(), "mail.Received");
    }

    #[test]
    fn lashlang_graph_store_builds_graph_state() {
        let store = TraceLashlangGraphStore::default();
        let context = TraceContext::default().for_session("s1");
        let identity = TraceLashlangExecutionIdentity {
            scope: TraceRuntimeScope::new("s1"),
            subject: TraceRuntimeSubject::Process {
                process_id: "p1".to_string(),
            },
            module_ref: "m1".to_string(),
            entry_kind: "process".to_string(),
            entry_ref: Some("r1:0".to_string()),
            entry_name: "main".to_string(),
        };
        let append = |event: TraceLashlangExecutionEvent| {
            store
                .append(&TraceRecord::new(
                    context.clone(),
                    TraceEvent::LashlangExecution { event },
                ))
                .expect("append tracking event");
        };

        append(TraceLashlangExecutionEvent::ExecutionStarted {
            event_key: "p1:start".to_string(),
            identity: identity.clone(),
            execution_map: TraceLashlangMap {
                module_ref: "m1".to_string(),
                entry_kind: "process".to_string(),
                entry_ref: Some("r1:0".to_string()),
                entry_name: "main".to_string(),
                nodes: vec![TraceLashlangMapNode {
                    id: "branch".to_string(),
                    kind: "branch".to_string(),
                    label: "if".to_string(),
                    label_metadata: None,
                }],
                edges: vec![
                    TraceLashlangMapEdge {
                        id: "then-edge".to_string(),
                        from: "branch".to_string(),
                        to: "then".to_string(),
                        label: "then".to_string(),
                    },
                    TraceLashlangMapEdge {
                        id: "else-edge".to_string(),
                        from: "branch".to_string(),
                        to: "else".to_string(),
                        label: "else".to_string(),
                    },
                ],
            },
        });
        append(TraceLashlangExecutionEvent::BranchSelected {
            event_key: "p1:branch".to_string(),
            identity: identity.clone(),
            node_id: "branch".to_string(),
            occurrence: 1,
            edge_id: "then-edge".to_string(),
            selected: TraceBranchSelection::Then,
        });
        append(TraceLashlangExecutionEvent::ChildStarted {
            event_key: "p1:child".to_string(),
            identity,
            parent_node_id: "branch".to_string(),
            occurrence: 1,
            child: TraceLashlangChildExecution {
                scope: TraceRuntimeScope::new("s1"),
                subject: TraceRuntimeSubject::Process {
                    process_id: "p2".to_string(),
                },
                module_ref: Some("m1".to_string()),
                entry_ref: Some("r2:1".to_string()),
                entry_name: Some("child".to_string()),
            },
        });

        let graph = store.graph("process:p1").expect("graph");
        assert_eq!(graph.status, TraceLashlangStatus::Running);
        assert_eq!(graph.children.len(), 1);
        assert_eq!(graph.children[0].child_graph_key, "process:p2");
        assert_eq!(
            graph
                .edges
                .iter()
                .find(|edge| edge.id == "then-edge")
                .map(|edge| edge.selection),
            Some(TraceLashlangEdgeSelection::Selected)
        );
        assert_eq!(
            graph
                .edges
                .iter()
                .find(|edge| edge.id == "else-edge")
                .map(|edge| edge.selection),
            Some(TraceLashlangEdgeSelection::Rejected)
        );
    }

    #[test]
    fn empty_model_variant_request_clears_selected_variant() {
        let selected_model = ModelSelection {
            model: "x-ai/grok-build-0.1".to_string(),
            model_variant: Some("medium".to_string()),
        };

        assert_eq!(
            model_variant_for_request(&selected_model, None),
            Some("medium".to_string())
        );
        assert_eq!(
            model_variant_for_request(&selected_model, Some(" high ")),
            Some("high".to_string())
        );
        assert_eq!(model_variant_for_request(&selected_model, Some("")), None);
        assert_eq!(
            model_variant_for_request(&selected_model, Some("   ")),
            None
        );
    }

    #[test]
    fn done_stream_items_are_transient_and_not_snapshotted() {
        let data_dir = std::env::temp_dir().join(format!(
            "agent-workbench-transient-done-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&data_dir).expect("create temp workbench dir");
        let process_registry = Arc::new(sync_await({
            let path = data_dir.join("processes.db");
            async move {
                lash_sqlite_store::SqliteProcessRegistry::open(&path)
                    .await
                    .expect("open registry")
            }
        })) as Arc<dyn lash::process::ProcessRegistry>;
        let session_store_factory = Arc::new(lash_sqlite_store::SqliteSessionStoreFactory::new(
            data_dir.join("lash-sessions"),
        ));
        let core_store_factory: Arc<dyn lash::persistence::SessionStoreFactory> =
            session_store_factory;
        let provider = lash::testing::TestProvider::builder()
            .kind("workbench-test")
            .complete_error("transient done test should not call the provider")
            .build()
            .into_handle();
        let model =
            lash::ModelSpec::from_token_limits("test-model", None, 4096, None).expect("model spec");
        let (event_tx, _) = broadcast::channel(16);
        let mut events = event_tx.subscribe();
        let core = explicit_durable_test_facets(LashCore::builder(), &data_dir)
            .install_mode(ModePreset::rlm_with_config(
                lash::modes::RlmProtocolPluginConfig::default()
                    .with_lashlang_abilities(workbench_lashlang_abilities()),
            ))
            .default_mode(ModeId::rlm())
            .provider(provider)
            .model(model)
            .store_factory(Arc::clone(&core_store_factory))
            .process_registry(Arc::clone(&process_registry))
            .build()
            .expect("build core");
        let process_observer = core
            .process_observer()
            .expect("process observer configured")
            .clone();
        let state = AppState {
            core,
            process_observer,
            session_store_factory: Arc::clone(&core_store_factory),
            session_ids: WorkbenchSessionIds::fresh(),
            messages: Arc::new(Mutex::new(Vec::new())),
            timeline: Arc::new(Mutex::new(Vec::new())),
            selected_model: Arc::new(Mutex::new(ModelSelection {
                model: "test-model".to_string(),
                model_variant: None,
            })),
            web_configured: false,
            trace_sink: None,
            lashlang_execution: Arc::new(TraceLashlangGraphStore::default()),
            event_tx,
            queued_work_poke: inert_queued_work_poke(),
            restate_ingress_url: "http://127.0.0.1:8080".to_string(),
            restate_http: reqwest::Client::new(),
            restate_cron_job_keys: Arc::new(Mutex::new(BTreeSet::new())),
            mail_world: mail::MailWorld::new(),
        };

        state.publish(StreamItem::Done);

        assert!(state.timeline_snapshot().is_empty());
        assert!(matches!(events.try_recv(), Ok(StreamItem::Done)));
        let _ = std::fs::remove_dir_all(data_dir);
    }

    #[test]
    fn inbox_authority_resolves_for_any_account_name() {
        run_async_test_on_stack_budget("workbench-inbox-authority-test", || {
            inbox_authority_resolves_for_any_account_name_inner()
        });
    }

    // Names other than the prompt's illustrative work/personal must resolve too:
    // an account called "test" should yield a usable `inbox.test` authority.
    async fn inbox_authority_resolves_for_any_account_name_inner() {
        let data_dir =
            std::env::temp_dir().join(format!("agent-workbench-inbox-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&data_dir).expect("create temp workbench dir");
        let session_store_factory = Arc::new(lash_sqlite_store::SqliteSessionStoreFactory::new(
            data_dir.join("lash-sessions"),
        ));
        let core_store_factory: Arc<dyn lash::persistence::SessionStoreFactory> =
            session_store_factory;
        let process_registry = Arc::new(
            lash_sqlite_store::SqliteProcessRegistry::open(&data_dir.join("processes.db"))
                .await
                .expect("open registry"),
        ) as Arc<dyn lash::process::ProcessRegistry>;
        let mail_world = mail::MailWorld::new();
        mail_world.add_account("test").expect("add test");
        let provider = lash::testing::TestProvider::builder()
            .kind("workbench-test")
            .complete(|_| async {
                Ok(text_response(
                    "```lashlang\nresult = await inbox.test.send({ title: \"Hi\", text: \"Yo\" })?\nsubmit result.id\n```",
                ))
            })
            .build()
            .into_handle();
        let model =
            lash::ModelSpec::from_token_limits("test-model", None, 4096, None).expect("model spec");
        let session_id = WorkbenchSessionIds::fresh().current();
        let core = explicit_durable_test_facets(LashCore::builder(), &data_dir)
            .install_mode(ModePreset::rlm_with_config(
                lash::modes::RlmProtocolPluginConfig::default()
                    .with_lashlang_abilities(workbench_lashlang_abilities()),
            ))
            .default_mode(ModeId::rlm())
            .provider(provider)
            .model(model)
            .store_factory(Arc::clone(&core_store_factory))
            .plugin(Arc::new(
                WorkbenchPluginFactory::new("").with_mail_world(mail_world.clone()),
            ))
            .process_registry(Arc::clone(&process_registry))
            .build()
            .expect("build core");
        let session = core
            .session(session_id)
            .rlm()
            .open()
            .await
            .expect("open session");

        let tool_names = session
            .tools()
            .active_manifests()
            .await
            .expect("active tools")
            .into_iter()
            .map(|tool| tool.name)
            .collect::<Vec<_>>();
        assert!(
            tool_names.iter().any(|name| name == "inbox__test__send"),
            "inbox.test send tool should be active: {tool_names:?}"
        );

        let output = session
            .turn(lash::TurnInput::text("send a message"))
            .turn_id(format!("workbench-test-turn:{}", uuid::Uuid::new_v4()))
            .run()
            .await
            .expect("turn should resolve inbox.test.send, not fail with unknown name");
        assert_eq!(output.submitted_value(), Some(&serde_json::json!("test-1")));
        assert_eq!(mail_world.inbox("test").expect("inbox").len(), 1);
        let _ = std::fs::remove_dir_all(data_dir);
    }

    #[test]
    fn inbox_added_after_session_open_updates_persisted_tool_surface() {
        run_async_test_on_stack_budget("workbench-dynamic-inbox-surface-test", || {
            inbox_added_after_session_open_updates_persisted_tool_surface_inner()
        });
    }

    async fn inbox_added_after_session_open_updates_persisted_tool_surface_inner() {
        let data_dir = std::env::temp_dir().join(format!(
            "agent-workbench-dynamic-inbox-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&data_dir).expect("create temp workbench dir");
        let session_store_factory = Arc::new(lash_sqlite_store::SqliteSessionStoreFactory::new(
            data_dir.join("lash-sessions"),
        ));
        let core_store_factory: Arc<dyn lash::persistence::SessionStoreFactory> =
            session_store_factory;
        let process_registry = Arc::new(
            lash_sqlite_store::SqliteProcessRegistry::open(&data_dir.join("processes.db"))
                .await
                .expect("open registry"),
        ) as Arc<dyn lash::process::ProcessRegistry>;
        let mail_world = mail::MailWorld::new();
        let provider = lash::testing::TestProvider::builder()
            .kind("workbench-test")
            .complete_error("dynamic inbox surface test should not call the provider")
            .build()
            .into_handle();
        let model =
            lash::ModelSpec::from_token_limits("test-model", None, 4096, None).expect("model spec");
        let session_ids = WorkbenchSessionIds::fresh();
        let core = explicit_durable_test_facets(LashCore::builder(), &data_dir)
            .install_mode(ModePreset::rlm_with_config(
                lash::modes::RlmProtocolPluginConfig::default()
                    .with_lashlang_abilities(workbench_lashlang_abilities()),
            ))
            .default_mode(ModeId::rlm())
            .provider(provider)
            .model(model)
            .store_factory(Arc::clone(&core_store_factory))
            .plugin(Arc::new(
                WorkbenchPluginFactory::new("").with_mail_world(mail_world.clone()),
            ))
            .process_registry(Arc::clone(&process_registry))
            .build()
            .expect("build core");
        let process_observer = core
            .process_observer()
            .expect("process observer configured")
            .clone();
        let state = AppState {
            core,
            process_observer,
            session_store_factory: Arc::clone(&core_store_factory),
            session_ids,
            messages: Arc::new(Mutex::new(Vec::new())),
            timeline: Arc::new(Mutex::new(Vec::new())),
            selected_model: Arc::new(Mutex::new(ModelSelection {
                model: "test-model".to_string(),
                model_variant: None,
            })),
            web_configured: false,
            trace_sink: None,
            lashlang_execution: Arc::new(TraceLashlangGraphStore::default()),
            event_tx: broadcast::channel(1024).0,
            queued_work_poke: inert_queued_work_poke(),
            restate_ingress_url: "http://127.0.0.1:8080".to_string(),
            restate_http: reqwest::Client::new(),
            restate_cron_job_keys: Arc::new(Mutex::new(BTreeSet::new())),
            mail_world: mail_world.clone(),
        };

        refresh_persisted_tool_surface(&state, "initial_empty")
            .await
            .expect("persist initial empty surface");
        mail_world.add_account("Late Account").expect("add account");
        refresh_persisted_tool_surface(&state, "account_added")
            .await
            .expect("refresh account surface");

        let reopened = state
            .core
            .session(state.current_session_id())
            .rlm()
            .open()
            .await
            .expect("reopen session");
        let tool_names = reopened
            .tools()
            .active_manifests()
            .await
            .expect("active tools")
            .into_iter()
            .map(|tool| tool.name)
            .collect::<Vec<_>>();
        assert!(
            tool_names
                .iter()
                .any(|name| name == "inbox__late_account__send"),
            "late account send tool should be active after persisted refresh: {tool_names:?}"
        );
        reopened.close().await.expect("close session");
        let _ = std::fs::remove_dir_all(data_dir);
    }

    #[test]
    fn button_host_event_occurrence_starts_visible_lashlang_process() {
        run_async_test_on_stack_budget("workbench-button-trigger-test", || {
            button_host_event_occurrence_starts_visible_lashlang_process_inner()
        });
    }

    async fn button_host_event_occurrence_starts_visible_lashlang_process_inner() {
        let data_dir = std::env::temp_dir().join(format!(
            "agent-workbench-processes-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&data_dir).expect("create temp workbench dir");
        let db_path = data_dir.join("processes.db");
        let session_store_factory = Arc::new(lash_sqlite_store::SqliteSessionStoreFactory::new(
            data_dir.join("lash-sessions"),
        ));
        let core_store_factory: Arc<dyn lash::persistence::SessionStoreFactory> =
            session_store_factory.clone();
        let process_registry = Arc::new(
            lash_sqlite_store::SqliteProcessRegistry::open(&db_path)
                .await
                .expect("open registry"),
        ) as Arc<dyn lash::process::ProcessRegistry>;
        let provider = trigger_registration_provider();
        let model =
            lash::ModelSpec::from_token_limits("test-model", None, 4096, None).expect("model spec");
        let session_ids = WorkbenchSessionIds::fresh();
        let session_id = session_ids.current();
        let core = explicit_durable_test_facets(LashCore::builder(), &data_dir)
            .install_mode(ModePreset::rlm_with_config(
                lash::modes::RlmProtocolPluginConfig::default()
                    .with_lashlang_abilities(workbench_lashlang_abilities()),
            ))
            .default_mode(ModeId::rlm())
            .provider(provider)
            .model(model)
            .store_factory(Arc::clone(&core_store_factory))
            .plugin(Arc::new(WorkbenchPluginFactory::new("")))
            .process_registry(Arc::clone(&process_registry))
            .build()
            .expect("build core");
        let session = core
            .session(session_id.clone())
            .rlm()
            .open()
            .await
            .expect("open session");
        register_test_trigger(&session).await;
        let tool_names = session
            .tools()
            .active_manifests()
            .await
            .expect("active tools")
            .into_iter()
            .map(|tool| tool.name)
            .collect::<Vec<_>>();
        let removed_tool_name = ["attach", "button", "trigger"].join("_");
        assert!(!tool_names.iter().any(|name| name == &removed_tool_name));

        let report = emit_test_button_trigger(&core, ButtonChoice::Red).await;

        assert_eq!(report.started_process_ids.len(), 1);
        process_registry
            .await_process(&report.started_process_ids[0])
            .await
            .expect("trigger process should finish");
        let handles = session
            .process_control()
            .list_all()
            .await
            .expect("list handles");
        assert_eq!(handles.len(), 1);
        assert_eq!(handles[0].descriptor.kind.as_deref(), Some("lashlang"));
        assert_eq!(handles[0].descriptor.label.as_deref(), Some("remember"));
        session.close().await.expect("close session");

        let reopened = core
            .session(session_id.clone())
            .rlm()
            .open()
            .await
            .expect("reopen session");
        let reopened_handles = reopened
            .process_control()
            .list_all()
            .await
            .expect("list handles after reopen");
        assert_eq!(reopened_handles.len(), 1);
        assert_eq!(
            reopened_handles[0].status.label(),
            "completed",
            "{:?}",
            reopened_handles[0].status
        );
        drop(reopened);

        let process_observer = core
            .process_observer()
            .expect("process observer configured")
            .clone();
        let state = AppState {
            core,
            process_observer,
            session_store_factory: Arc::clone(&core_store_factory),
            session_ids,
            messages: Arc::new(Mutex::new(Vec::new())),
            timeline: Arc::new(Mutex::new(Vec::new())),
            selected_model: Arc::new(Mutex::new(ModelSelection {
                model: "test-model".to_string(),
                model_variant: None,
            })),
            web_configured: false,
            trace_sink: None,
            lashlang_execution: Arc::new(TraceLashlangGraphStore::default()),
            event_tx: broadcast::channel(1024).0,
            queued_work_poke: inert_queued_work_poke(),
            restate_ingress_url: "http://127.0.0.1:8080".to_string(),
            restate_http: reqwest::Client::new(),
            restate_cron_job_keys: Arc::new(Mutex::new(BTreeSet::new())),
            mail_world: mail::MailWorld::new(),
        };
        let target_scope_id = lash::process::ProcessScope::new(state.current_session_id()).id();
        let session_store =
            lash_sqlite_store::Store::open(&session_store_factory.path_for_session(&session_id))
                .await
                .expect("open session store");
        let queued = session_store
            .list_queued_work(&session_id)
            .await
            .expect("list queued work");
        assert_eq!(queued.len(), 1);
        assert_eq!(queued[0].items.len(), 1);
        let lash::persistence::QueuedWorkPayload::ProcessWake { wake } =
            &queued[0].items[0].payload
        else {
            panic!("expected process wake queue payload");
        };
        assert!(wake.input.contains("button_pressed"));
        assert!(wake.input.contains("Red"));
        assert_eq!(wake.target_scope_id, target_scope_id);
        let Json(work) = list_work(State(state)).await.expect("list work");
        assert_eq!(work.len(), 1);
        assert_eq!(work[0].process.status_label, "completed");
        assert!(
            work[0]
                .events
                .iter()
                .any(|event| event.event_type == "process.completed")
        );
        assert!(
            work[0]
                .events
                .iter()
                .any(|event| event.event_type == "process.wake")
        );
        let _ = std::fs::remove_dir_all(data_dir);
    }

    #[test]
    fn button_host_event_occurrence_is_submitted_to_restate_workflow() {
        run_async_test_on_stack_budget("workbench-trigger-restate-test", || {
            button_host_event_occurrence_is_submitted_to_restate_workflow_inner()
        });
    }

    async fn button_host_event_occurrence_is_submitted_to_restate_workflow_inner() {
        let data_dir = std::env::temp_dir().join(format!(
            "agent-workbench-queue-runner-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&data_dir).expect("create temp workbench dir");
        let session_store_factory = Arc::new(lash_sqlite_store::SqliteSessionStoreFactory::new(
            data_dir.join("lash-sessions"),
        ));
        let core_store_factory: Arc<dyn lash::persistence::SessionStoreFactory> =
            session_store_factory;
        let process_registry = Arc::new(
            lash_sqlite_store::SqliteProcessRegistry::open(&data_dir.join("processes.db"))
                .await
                .expect("open registry"),
        ) as Arc<dyn lash::process::ProcessRegistry>;
        let host_event_store = Arc::new(
            lash_sqlite_store::SqliteHostEventStore::open(&data_dir.join("host-events.db"))
                .await
                .expect("open host event store"),
        );
        let artifact_store = Arc::new(
            lash_sqlite_store::Store::open(&data_dir.join("artifacts.db"))
                .await
                .expect("open artifact store"),
        );
        let artifact_store_for_core: Arc<dyn lashlang::LashlangArtifactStore> =
            artifact_store.clone();
        let provider = lash::testing::TestProvider::builder()
            .kind("workbench-test")
            .supported_variants(|_| &["low", "medium", "high"])
            .complete(|_| async { Ok(trigger_registration_response()) })
            .build()
            .into_handle();
        let model =
            lash::ModelSpec::from_token_limits("test-model", None, 4096, None).expect("model spec");
        let (restate_ingress_url, mut restate_requests) = spawn_restate_ingress_capture().await;
        let (event_tx, _) = broadcast::channel(1024);
        let core = LashCore::builder()
            .install_mode(ModePreset::rlm_with_config(
                lash::modes::RlmProtocolPluginConfig::default()
                    .with_lashlang_abilities(workbench_lashlang_abilities()),
            ))
            .default_mode(ModeId::rlm())
            .provider(provider)
            .model(model)
            .store_factory(Arc::clone(&core_store_factory))
            .plugin(Arc::new(WorkbenchPluginFactory::new("")))
            .process_registry(Arc::clone(&process_registry))
            .host_event_store(host_event_store)
            .advanced()
            .runtime_host_config({
                let mut config = lash::durability::RuntimeHostConfig::in_memory();
                config.durability.lashlang_artifact_store = artifact_store_for_core;
                config.control.effect_host =
                    Arc::new(lash::durability::InlineEffectHost::default());
                config
            })
            .build()
            .expect("build core");
        let process_observer = core
            .process_observer()
            .expect("process observer configured")
            .clone();
        let state = AppState {
            core,
            process_observer,
            session_store_factory: Arc::clone(&core_store_factory),
            session_ids: WorkbenchSessionIds::fresh(),
            messages: Arc::new(Mutex::new(Vec::new())),
            timeline: Arc::new(Mutex::new(Vec::new())),
            selected_model: Arc::new(Mutex::new(ModelSelection {
                model: "test-model".to_string(),
                model_variant: None,
            })),
            web_configured: false,
            trace_sink: None,
            lashlang_execution: Arc::new(TraceLashlangGraphStore::default()),
            event_tx,
            queued_work_poke: inert_queued_work_poke(),
            restate_ingress_url,
            restate_http: reqwest::Client::new(),
            restate_cron_job_keys: Arc::new(Mutex::new(BTreeSet::new())),
            mail_world: mail::MailWorld::new(),
        };
        let session = state
            .core
            .session(state.current_session_id())
            .rlm()
            .open()
            .await
            .expect("open session");
        register_test_trigger(&session).await;
        drop(session);

        let _accepted = button_trigger(
            State(state.clone()),
            Json(ButtonEventRequest {
                button: ButtonChoice::Blue,
                model: Some("button-model".to_string()),
                model_variant: Some("high".to_string()),
            }),
        )
        .await
        .expect("button command");
        let selected_model = state.selected_model();
        assert_eq!(selected_model.model, "button-model");
        assert_eq!(selected_model.model_variant.as_deref(), Some("high"));
        assert!(
            state
                .messages_snapshot()
                .iter()
                .any(|message| message.role == "event" && message.text == "blue button event"),
            "button click should publish the local accepted event"
        );

        let request = tokio::time::timeout(Duration::from_secs(2), restate_requests.recv())
            .await
            .expect("Restate request")
            .expect("Restate request payload");
        let path = request
            .get("path")
            .and_then(Value::as_str)
            .expect("request path");
        assert!(
            path.starts_with("WorkbenchButtonTriggerWorkflow/workbench-button-"),
            "unexpected Restate path: {path}"
        );
        assert!(
            path.ends_with("/run/send"),
            "unexpected Restate path: {path}"
        );
        assert_eq!(
            request.pointer("/body/session_id").and_then(Value::as_str),
            Some(state.current_session_id().as_str())
        );
        assert_eq!(
            request.pointer("/body/button").and_then(Value::as_str),
            Some("Blue")
        );
        assert_eq!(
            request.pointer("/body/model/model").and_then(Value::as_str),
            Some("button-model")
        );
        assert_eq!(
            request
                .pointer("/body/model/model_variant")
                .and_then(Value::as_str),
            Some("high")
        );
        let _ = std::fs::remove_dir_all(data_dir);
    }

    async fn spawn_restate_ingress_capture() -> (String, mpsc::UnboundedReceiver<Value>) {
        let (tx, rx) = mpsc::unbounded_channel();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind mock Restate ingress");
        let addr = listener.local_addr().expect("mock Restate ingress addr");
        let app = Router::new()
            .route("/{*path}", post(capture_restate_send))
            .with_state(tx);
        tokio::spawn(async move {
            if let Err(err) = axum::serve(listener, app).await {
                eprintln!("mock Restate ingress stopped: {err}");
            }
        });
        (format!("http://{addr}"), rx)
    }

    async fn capture_restate_send(
        AxumPath(path): AxumPath<String>,
        State(tx): State<mpsc::UnboundedSender<Value>>,
        Json(body): Json<Value>,
    ) -> StatusCode {
        let _ = tx.send(json!({
            "path": path,
            "body": body,
        }));
        StatusCode::ACCEPTED
    }

    #[test]
    fn reset_chat_deletes_old_session_and_clears_trigger_started_work() {
        run_async_test_on_stack_budget("workbench-reset-chat-test", || {
            reset_chat_deletes_old_session_and_clears_trigger_started_work_inner()
        });
    }

    async fn reset_chat_deletes_old_session_and_clears_trigger_started_work_inner() {
        let data_dir =
            std::env::temp_dir().join(format!("agent-workbench-reset-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&data_dir).expect("create temp workbench dir");
        let session_store_factory = Arc::new(lash_sqlite_store::SqliteSessionStoreFactory::new(
            data_dir.join("lash-sessions"),
        ));
        let core_store_factory: Arc<dyn lash::persistence::SessionStoreFactory> =
            session_store_factory;
        let process_registry = Arc::new(
            lash_sqlite_store::SqliteProcessRegistry::open(&data_dir.join("processes.db"))
                .await
                .expect("open registry"),
        ) as Arc<dyn lash::process::ProcessRegistry>;
        let provider = trigger_registration_provider();
        let model =
            lash::ModelSpec::from_token_limits("test-model", None, 4096, None).expect("model spec");
        let (restate_ingress_url, mut restate_requests) = spawn_restate_ingress_capture().await;
        let core = explicit_durable_test_facets(LashCore::builder(), &data_dir)
            .install_mode(ModePreset::rlm_with_config(
                lash::modes::RlmProtocolPluginConfig::default()
                    .with_lashlang_abilities(workbench_lashlang_abilities()),
            ))
            .default_mode(ModeId::rlm())
            .provider(provider)
            .model(model)
            .store_factory(Arc::clone(&core_store_factory))
            .plugin(Arc::new(WorkbenchPluginFactory::new("")))
            .process_registry(Arc::clone(&process_registry))
            .build()
            .expect("build core");
        let process_observer = core
            .process_observer()
            .expect("process observer configured")
            .clone();
        let state = AppState {
            core,
            process_observer,
            session_store_factory: Arc::clone(&core_store_factory),
            session_ids: WorkbenchSessionIds::fresh(),
            messages: Arc::new(Mutex::new(vec![ChatMessage {
                id: "message".to_string(),
                role: "user".to_string(),
                text: "before reset".to_string(),
                at: "2026-05-27T00:00:00Z".to_string(),
            }])),
            timeline: Arc::new(Mutex::new(vec![StreamItem::event(
                TurnActivity::independent(TurnEvent::CodeBlockStarted {
                    language: "lashlang".to_string(),
                    code: "finish true".to_string(),
                    graph_key: None,
                }),
            )])),
            selected_model: Arc::new(Mutex::new(ModelSelection {
                model: "test-model".to_string(),
                model_variant: None,
            })),
            web_configured: false,
            trace_sink: None,
            lashlang_execution: Arc::new(TraceLashlangGraphStore::default()),
            event_tx: broadcast::channel(1024).0,
            queued_work_poke: inert_queued_work_poke(),
            restate_ingress_url,
            restate_http: reqwest::Client::new(),
            restate_cron_job_keys: Arc::new(Mutex::new(BTreeSet::new())),
            mail_world: mail::MailWorld::new(),
        };
        let old_session_id = state.current_session_id();
        let session = state
            .core
            .session(old_session_id.clone())
            .rlm()
            .open()
            .await
            .expect("open old session");
        register_test_trigger(&session).await;
        let started = emit_test_button_trigger(&state.core, ButtonChoice::Red).await;
        assert_eq!(started.started_process_ids.len(), 1);
        let old_process_scope = session.observe().process_scope();
        assert_eq!(
            process_registry
                .list_handle_grants(&old_process_scope)
                .await
                .expect("old grants before reset")
                .len(),
            1
        );
        append_started_graph(
            &state.lashlang_execution,
            &test_graph(
                "process:old-reset-process",
                &old_session_id,
                TraceRuntimeSubject::Process {
                    process_id: "old-reset-process".to_string(),
                },
                Vec::new(),
            ),
        );
        assert_eq!(state.lashlang_execution.graphs().len(), 1);
        drop(session);

        let Json(snapshot) = reset_chat(State(state.clone())).await.expect("reset");

        assert_ne!(snapshot.settings.session_id, old_session_id);
        assert!(snapshot.messages.is_empty());
        assert!(snapshot.timeline.is_empty());
        assert!(state.messages_snapshot().is_empty());
        assert!(state.timeline_snapshot().is_empty());
        let request = tokio::time::timeout(Duration::from_secs(2), restate_requests.recv())
            .await
            .expect("Restate request")
            .expect("Restate request payload");
        let path = request
            .get("path")
            .and_then(Value::as_str)
            .expect("request path");
        assert!(
            path.starts_with("WorkbenchSessionDeleteWorkflow/workbench-delete-"),
            "unexpected Restate path: {path}"
        );
        assert!(
            path.ends_with("/run/send"),
            "unexpected Restate path: {path}"
        );
        assert_eq!(
            request.pointer("/body/session_id").and_then(Value::as_str),
            Some(old_session_id.as_str())
        );
        assert!(
            process_registry
                .list_handle_grants(&old_process_scope)
                .await
                .expect("old grants after reset submission")
                .len()
                == 1,
            "mock Restate ingress must not consume deletion work inline"
        );
        assert!(
            state
                .core
                .session(snapshot.settings.session_id)
                .rlm()
                .open()
                .await
                .expect("open new session")
                .process_control()
                .list()
                .await
                .expect("new work")
                .is_empty()
        );
        let Json(graph_index) = list_lashlang_graphs(State(state.clone()))
            .await
            .expect("list graphs after reset");
        assert!(
            graph_index.graphs.is_empty(),
            "new session graph index should be empty after reset: {graph_index:#?}"
        );
        let _ = std::fs::remove_dir_all(data_dir);
    }

    #[test]
    #[ignore = "requires a running Restate server; use `just agent-workbench-restate-e2e`"]
    fn live_restate_cron_runs_trigger_and_queued_turn_end_to_end() {
        run_async_test_on_stack_budget_multi_thread("workbench-restate-cron-e2e", 4, || {
            live_restate_cron_runs_trigger_and_queued_turn_end_to_end_inner()
        });
    }

    async fn live_restate_cron_runs_trigger_and_queued_turn_end_to_end_inner() {
        let ingress_url = match std::env::var("RESTATE_INGRESS_URL") {
            Ok(value) => value,
            Err(_) => {
                eprintln!("skipping live Restate E2E: RESTATE_INGRESS_URL is not set");
                return;
            }
        };
        let admin_url = std::env::var("RESTATE_ADMIN_URL")
            .unwrap_or_else(|_| "http://127.0.0.1:19071".to_string());
        let endpoint_bind: SocketAddr = std::env::var("AGENT_WORKBENCH_E2E_ENDPOINT_BIND")
            .unwrap_or_else(|_| "127.0.0.1:19081".to_string())
            .parse()
            .expect("valid workbench E2E endpoint bind");
        let endpoint_url = std::env::var("AGENT_WORKBENCH_E2E_ENDPOINT_URL")
            .unwrap_or_else(|_| format!("http://{endpoint_bind}"));
        let data_dir = std::env::temp_dir().join(format!(
            "agent-workbench-restate-e2e-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&data_dir).expect("create temp workbench dir");
        let harness = live_workbench_restate_state(&data_dir, ingress_url).await;
        restate::spawn_restate_endpoint(
            endpoint_bind,
            harness.state.clone(),
            harness.process_deployment,
            harness.process_worker,
        );
        wait_for_endpoint_socket(endpoint_bind).await;
        register_restate_deployment(&admin_url, &endpoint_url).await;
        harness.queued_work_runner.spawn();

        run_workbench_turn_via_restate(
            &harness.state,
            "Register a cron trigger that runs every two seconds and reports the tick.",
        )
        .await;
        wait_for_workbench_message(&harness.state, "cron registered", Duration::from_secs(60))
            .await;
        wait_for_restate_cron_sync(&harness.state, &harness.trace_path, Duration::from_secs(30))
            .await;
        wait_for_workbench_message(
            &harness.state,
            "cron tick observed",
            Duration::from_secs(60),
        )
        .await;
        let trace_text =
            std::fs::read_to_string(&harness.trace_path).expect("read workbench trace jsonl");
        assert!(
            trace_text.contains("agent_workbench.cron.restate.sync_upserted"),
            "trace should include cron sync; trace at {}",
            harness.trace_path.display()
        );
        assert!(
            trace_text.contains("agent_workbench.cron.restate.run"),
            "trace should include cron run; trace at {}",
            harness.trace_path.display()
        );
        let _ = restate::cancel_known_cron_jobs(&harness.state, "live_e2e_cleanup").await;
        let _ = std::fs::remove_dir_all(data_dir);
    }

    async fn run_workbench_turn_via_restate(state: &AppState, text: &str) {
        state.push_message("user", text);
        let turn_id = format!("workbench-turn-{}", uuid::Uuid::new_v4());
        let request = restate::WorkbenchTurnWorkflowRequest {
            turn_id: turn_id.clone(),
            session_id: state.current_session_id(),
            text: text.to_string(),
            model: state.selected_model(),
        };
        let url = format!(
            "{}/WorkbenchTurnWorkflow/{}/run",
            state.restate_ingress_url.trim_end_matches('/'),
            turn_id,
        );
        let response = tokio::time::timeout(
            Duration::from_secs(60),
            state.restate_http.post(&url).json(&request).send(),
        )
        .await
        .expect("Restate-backed workbench turn timed out")
        .expect("submit Restate-backed workbench turn");
        assert!(
            response.status().is_success(),
            "Restate-backed workbench turn failed: {} {}",
            response.status(),
            response.text().await.unwrap_or_default(),
        );
    }

    struct LiveWorkbenchRestateHarness {
        state: AppState,
        process_worker: lash::durability::DurableProcessWorker,
        process_deployment: lash_restate::RestateProcessDeployment,
        queued_work_runner: lash::runtime::QueuedWorkRunner,
        trace_path: PathBuf,
    }

    async fn live_workbench_restate_state(
        data_dir: &std::path::Path,
        restate_ingress_url: String,
    ) -> LiveWorkbenchRestateHarness {
        let session_store_factory = Arc::new(lash_sqlite_store::SqliteSessionStoreFactory::new(
            data_dir.join("lash-sessions"),
        ));
        let core_store_factory: Arc<dyn lash::persistence::SessionStoreFactory> =
            session_store_factory.clone();
        let process_registry = Arc::new(
            lash_sqlite_store::SqliteProcessRegistry::open(&data_dir.join("processes.db"))
                .await
                .expect("open process registry"),
        ) as Arc<dyn lash::process::ProcessRegistry>;
        let host_event_store = Arc::new(
            lash_sqlite_store::SqliteHostEventStore::open(&data_dir.join("host-events.db"))
                .await
                .expect("open host event store"),
        );
        let artifact_store = Arc::new(
            lash_sqlite_store::Store::open(&data_dir.join("artifacts.db"))
                .await
                .expect("open artifact store"),
        ) as Arc<dyn lash::persistence::LashlangArtifactStore>;
        let trace_path = data_dir.join("trace.jsonl");
        let lashlang_execution_path = data_dir.join("lashlang-execution.jsonl");
        let trace_sink = Arc::new(JsonlTraceSink::new(trace_path.clone())) as Arc<dyn TraceSink>;
        let lashlang_execution = Arc::new(TraceLashlangGraphStore::default());
        let lashlang_execution_sink = Arc::new(TeeTraceSink::new([
            Arc::clone(&lashlang_execution) as Arc<dyn TraceSink>,
            Arc::new(JsonlTraceSink::new(lashlang_execution_path)) as Arc<dyn TraceSink>,
        ])) as Arc<dyn TraceSink>;
        let response_index = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let response_index_for_provider = Arc::clone(&response_index);
        let provider = lash::testing::TestProvider::builder()
            .kind("workbench-restate-e2e")
            .supported_variants(|_| &["high"])
            .complete(move |_| {
                let response_index = Arc::clone(&response_index_for_provider);
                async move {
                    if response_index.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 0 {
                        Ok(text_response(&format!(
                            "```lashlang\n{}\n```",
                            test_cron_trigger_source().trim()
                        )))
                    } else {
                        Ok(text_response(
                            "```lashlang\nsubmit \"cron tick observed\"\n```",
                        ))
                    }
                }
            })
            .build()
            .into_handle();
        let model =
            lash::ModelSpec::from_token_limits("mock-model", Some("high".to_string()), 4096, None)
                .expect("model spec");
        let process_deployment = lash_restate::RestateProcessDeployment::new(
            restate_ingress_url.clone(),
            Arc::clone(&process_registry),
        );
        let session_ids = WorkbenchSessionIds::fresh();
        let restate_http = reqwest::Client::new();
        let queued_work_runner =
            lash::runtime::QueuedWorkRunner::new(Arc::new(WorkbenchQueuedWorkSubmitter {
                session_ids: session_ids.clone(),
                store_factory: Arc::clone(&core_store_factory),
                restate_ingress_url: restate_ingress_url.clone(),
                restate_http: restate_http.clone(),
            }));
        let queued_work_poke = queued_work_runner.poke_handle();
        let core = LashCore::builder()
            .install_mode(ModePreset::rlm_with_config(
                lash::modes::RlmProtocolPluginConfig::default()
                    .with_lashlang_abilities(workbench_lashlang_abilities()),
            ))
            .default_mode(ModeId::rlm())
            .provider(provider)
            .model(model)
            .store_factory(Arc::clone(&core_store_factory))
            .attachment_store(Arc::new(lash::persistence::FileAttachmentStore::new(
                data_dir.join("attachments"),
            )))
            .lashlang_artifact_store(artifact_store)
            .host_event_store(host_event_store)
            .trace_sink(Arc::clone(&trace_sink))
            .lashlang_execution_sink(lashlang_execution_sink)
            .trace_level(TraceLevel::Extended)
            .plugin(Arc::new(WorkbenchPluginFactory::new("")))
            .effect_host(Arc::new(lash::durability::InlineEffectHost::default()))
            .process_work_driver(process_deployment.process_work_driver())
            .queued_work_poke(queued_work_poke.clone())
            .build()
            .expect("build core");
        let process_worker = lash::durability::DurableProcessWorker::new(
            core.durable_process_worker_config()
                .expect("build process worker config"),
        );
        let process_observer = core
            .process_observer()
            .expect("process observer configured")
            .clone();
        let (event_tx, _) = broadcast::channel(1024);
        let state = AppState {
            core,
            process_observer,
            session_store_factory: Arc::clone(&core_store_factory),
            session_ids,
            messages: Arc::new(Mutex::new(Vec::new())),
            timeline: Arc::new(Mutex::new(Vec::new())),
            selected_model: Arc::new(Mutex::new(ModelSelection {
                model: "mock-model".to_string(),
                model_variant: Some("high".to_string()),
            })),
            web_configured: false,
            trace_sink: Some(trace_sink),
            lashlang_execution,
            event_tx,
            queued_work_poke,
            restate_ingress_url,
            restate_http,
            restate_cron_job_keys: Arc::new(Mutex::new(BTreeSet::new())),
            mail_world: mail::MailWorld::new(),
        };
        LiveWorkbenchRestateHarness {
            state,
            process_worker,
            process_deployment,
            queued_work_runner,
            trace_path,
        }
    }

    async fn wait_for_workbench_message(state: &AppState, needle: &str, timeout: Duration) {
        let deadline = std::time::Instant::now() + timeout;
        loop {
            let messages = state.messages_snapshot();
            if messages
                .iter()
                .any(|message| message.role == "assistant" && message.text.contains(needle))
            {
                return;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "timed out waiting for workbench message containing `{needle}`; messages={messages:#?}"
            );
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
    }

    async fn wait_for_restate_cron_sync(
        state: &AppState,
        trace_path: &std::path::Path,
        timeout: Duration,
    ) {
        let deadline = std::time::Instant::now() + timeout;
        loop {
            let known_jobs = state
                .restate_cron_job_keys
                .lock()
                .expect("cron job key lock")
                .clone();
            let trace_text = std::fs::read_to_string(trace_path).unwrap_or_default();
            if !known_jobs.is_empty()
                && trace_text.contains("agent_workbench.cron.restate.sync_upserted")
            {
                return;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "timed out waiting for Restate cron sync; known_jobs={known_jobs:#?}; messages={:#?}; trace_tail={}",
                state.messages_snapshot(),
                trace_tail(trace_path),
            );
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
    }

    fn trace_tail(path: &std::path::Path) -> String {
        let Ok(text) = std::fs::read_to_string(path) else {
            return format!("<unreadable {}>", path.display());
        };
        let mut lines = text.lines().rev().take(20).collect::<Vec<_>>();
        lines.reverse();
        lines.join("\n")
    }

    async fn wait_for_endpoint_socket(addr: SocketAddr) {
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        loop {
            if tokio::net::TcpStream::connect(addr).await.is_ok() {
                return;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "Restate endpoint did not open a TCP listener at {addr}"
            );
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }

    async fn register_restate_deployment(admin_url: &str, endpoint_url: &str) {
        let client = reqwest::Client::builder()
            .http2_prior_knowledge()
            .build()
            .expect("build Restate admin client");
        let response = client
            .post(format!("{}/deployments", admin_url.trim_end_matches('/')))
            .json(&json!({
                "uri": endpoint_url,
                "force": true,
                "breaking": true,
            }))
            .send()
            .await
            .expect("register deployment with Restate admin API");
        assert!(
            response.status().is_success(),
            "Restate deployment registration failed: {} {}",
            response.status(),
            response.text().await.unwrap_or_default()
        );
    }

    #[test]
    fn persisted_trigger_route_fires_after_reopening_sqlite_artifact_store() {
        run_async_test_on_stack_budget("workbench-persisted-trigger-test", || {
            persisted_trigger_route_fires_after_reopening_sqlite_artifact_store_inner()
        });
    }

    async fn persisted_trigger_route_fires_after_reopening_sqlite_artifact_store_inner() {
        let data_dir =
            std::env::temp_dir().join(format!("agent-workbench-trigger-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&data_dir).expect("create temp workbench dir");
        let session_store_factory = Arc::new(lash_sqlite_store::SqliteSessionStoreFactory::new(
            data_dir.join("lash-sessions"),
        ));
        let process_registry_path = data_dir.join("processes.db");
        let host_event_store_path = data_dir.join("host-events.db");
        let artifact_store_path = data_dir.join("artifacts.db");
        let session_id = WorkbenchSessionIds::fresh().current();

        {
            let artifact_store = Arc::new(
                lash_sqlite_store::Store::open(&artifact_store_path)
                    .await
                    .expect("open artifacts"),
            );
            let artifact_store_for_core: Arc<dyn lashlang::LashlangArtifactStore> =
                artifact_store.clone();
            let process_registry = Arc::new(
                lash_sqlite_store::SqliteProcessRegistry::open(&process_registry_path)
                    .await
                    .expect("open registry"),
            ) as Arc<dyn lash::process::ProcessRegistry>;
            let host_event_store = Arc::new(
                lash_sqlite_store::SqliteHostEventStore::open(&host_event_store_path)
                    .await
                    .expect("open host event store"),
            );
            let core = test_workbench_core(
                session_store_factory.clone(),
                process_registry,
                host_event_store,
                artifact_store_for_core,
            );
            let session = core
                .session(session_id.clone())
                .rlm()
                .open()
                .await
                .expect("open session");
            register_test_trigger(&session).await;
            drop(session);
            drop(core);
        }

        let artifact_store = Arc::new(
            lash_sqlite_store::Store::open(&artifact_store_path)
                .await
                .expect("reopen artifacts"),
        );
        let artifact_store_for_core: Arc<dyn lashlang::LashlangArtifactStore> =
            artifact_store.clone();
        let process_registry = Arc::new(
            lash_sqlite_store::SqliteProcessRegistry::open(&process_registry_path)
                .await
                .expect("reopen registry"),
        ) as Arc<dyn lash::process::ProcessRegistry>;
        let host_event_store = Arc::new(
            lash_sqlite_store::SqliteHostEventStore::open(&host_event_store_path)
                .await
                .expect("reopen host event store"),
        );
        let core = test_workbench_core(
            session_store_factory,
            Arc::clone(&process_registry),
            host_event_store,
            artifact_store_for_core,
        );
        let _reopened = core
            .session(session_id)
            .rlm()
            .open()
            .await
            .expect("reopen session");
        let report = emit_test_button_trigger(&core, ButtonChoice::Blue).await;
        assert_eq!(report.started_process_ids.len(), 1);
        process_registry
            .await_process(&report.started_process_ids[0])
            .await
            .expect("trigger process should finish");

        let _ = std::fs::remove_dir_all(data_dir);
    }

    fn test_workbench_core(
        session_store_factory: Arc<dyn lash::persistence::SessionStoreFactory>,
        process_registry: Arc<dyn lash::process::ProcessRegistry>,
        host_event_store: Arc<lash_sqlite_store::SqliteHostEventStore>,
        artifact_store: Arc<dyn lashlang::LashlangArtifactStore>,
    ) -> LashCore {
        let provider = trigger_registration_provider();
        let model =
            lash::ModelSpec::from_token_limits("test-model", None, 4096, None).expect("model spec");
        LashCore::builder()
            .install_mode(ModePreset::rlm_with_config(
                lash::modes::RlmProtocolPluginConfig::default()
                    .with_lashlang_abilities(workbench_lashlang_abilities()),
            ))
            .default_mode(ModeId::rlm())
            .provider(provider)
            .model(model)
            .store_factory(session_store_factory)
            .plugin(Arc::new(WorkbenchPluginFactory::new("")))
            .process_registry(process_registry)
            .host_event_store(host_event_store)
            .advanced()
            .runtime_host_config({
                let mut config = lash::durability::RuntimeHostConfig::in_memory();
                config.durability.lashlang_artifact_store = artifact_store;
                config.control.effect_host =
                    Arc::new(lash::durability::InlineEffectHost::default());
                config
            })
            .build()
            .expect("build core")
    }

    fn text_response(text: &str) -> lash::direct::LlmResponse {
        lash::direct::LlmResponse {
            full_text: text.to_string(),
            parts: vec![lash::direct::LlmOutputPart::Text {
                text: text.to_string(),
                response_meta: None,
            }],
            ..lash::direct::LlmResponse::default()
        }
    }

    fn trigger_registration_response() -> lash::direct::LlmResponse {
        text_response(&format!(
            "```lashlang\n{}\n```",
            test_button_trigger_source().trim()
        ))
    }

    async fn register_test_trigger(session: &lash::LashSession) {
        let output = session
            .turn(lash::TurnInput::text("register trigger"))
            .turn_id(format!("workbench-test-register:{}", uuid::Uuid::new_v4()))
            .run()
            .await
            .expect("register trigger route");
        assert_eq!(
            output.submitted_value(),
            Some(&serde_json::json!("registered"))
        );
    }

    async fn emit_test_button_trigger(
        core: &LashCore,
        button: ButtonChoice,
    ) -> lash::host_events::HostEventEmitReport {
        let source_key = lash::host_events::empty_host_event_source_key(BUTTON_TRIGGER_SOURCE_TYPE)
            .expect("source key");
        let idempotency_key = format!(
            "workbench-test-button-host-event:{}:{}",
            button.as_str(),
            uuid::Uuid::new_v4()
        );
        let scoped_effect_controller = lash::runtime::ScopedEffectController::shared(
            Arc::new(lash::runtime::InlineRuntimeEffectController),
            lash::runtime::EffectScope::runtime_operation(format!("host-event:{idempotency_key}")),
        )
        .expect("inline host event effect scope");
        core.host_events()
            .emit(
                lash::host_events::HostEventOccurrenceRequest::new(
                    BUTTON_TRIGGER_SOURCE_TYPE,
                    source_key,
                    json!({
                        "button": button.as_str(),
                        "message": format!("user pressed the {} button", button.lower()),
                        "pressed_at": "2026-06-02T12:00:00Z"
                    }),
                    idempotency_key,
                )
                .with_source(json!({})),
                scoped_effect_controller,
            )
            .await
            .expect("emit button trigger occurrence")
    }

    fn trigger_registration_provider() -> ProviderHandle {
        lash::testing::TestProvider::builder()
            .kind("workbench-test")
            .complete(|_| async { Ok(trigger_registration_response()) })
            .build()
            .into_handle()
    }

    fn test_button_trigger_source() -> &'static str {
        r#"
        process remember(event: ui.button.Pressed) {
          wake { kind: "button_pressed", button: event.button, message: event.message }
          finish { button: event.button, ok: true }
        }

        handle = await triggers.register({
          source: ui.button.pressed({}),
          target: remember,
          inputs: { event: trigger.event },
          name: "remembered"
        })?
        submit "registered"
        "#
    }

    fn test_cron_trigger_source() -> &'static str {
        r#"
        process remember_tick(tick: cron.Tick) {
          wake { kind: "cron_tick", fired_at: tick.fired_at }
          finish { fired_at: tick.fired_at }
        }

        handle = await triggers.register({
          source: cron.Schedule({ expr: "*/2 * * * * *", tz: "UTC" }),
          target: remember_tick,
          inputs: { tick: trigger.event },
          name: "cron smoke"
        })?
        submit "cron registered"
        "#
    }
}
