mod restate;
mod ui;

use std::collections::{BTreeMap, BTreeSet};
use std::convert::Infallible;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
#[cfg(test)]
use std::time::Duration;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result as AnyhowResult, anyhow};
use async_trait::async_trait;
use axum::body::Body;
use axum::extract::{Path as AxumPath, State};
use axum::http::{StatusCode, header};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use bytes::Bytes;
use chrono::Utc;
use lash::plugins::{
    HostEvent, PluginError, PluginFactory, PluginRegistrar, PluginSessionContext, SessionPlugin,
};
use lash::prompt::PromptContribution;
use lash::provider::{ProviderHandle, ProviderOptions, ProviderThinkingPolicy};
use lash::{
    LashCore, ModeId, ModePreset, SessionSpec, TurnActivity, TurnActivitySink, TurnEvent,
    TurnResult,
    tracing::{
        JsonlTraceSink, StderrTraceSink, TeeTraceSink, TraceContext, TraceEvent,
        TraceLashlangGraph, TraceLashlangGraphStore, TraceLevel, TraceRecord, TraceRuntimeScope,
        TraceRuntimeSubject, TraceSink,
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
const DEFAULT_TOKIO_THREAD_STACK_BYTES: usize = 16 * 1024 * 1024;

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
            .context("open process registry")?,
    ) as Arc<dyn lash::persistence::ProcessRegistry>;
    // Deployment-level Lashlang artifact store (compiled trigger/process
    // modules), shared across the session tree. SQLite keeps installed triggers
    // durable across restarts.
    let artifact_store = Arc::new(
        lash_sqlite_store::Store::open(&data_dir.join("artifacts.db"))
            .context("open lashlang artifact store")?,
    ) as Arc<dyn lash::persistence::LashlangArtifactStore>;
    let subagent_registry = Arc::new(lash_subagents::default_registry(&BTreeMap::new()));
    let session_ids = WorkbenchSessionIds::fresh();
    let (event_tx, _) = broadcast::channel(1024);
    let restate_http = reqwest::Client::new();
    let process_work_runner = lash::advanced::ProcessWorkRunner::new(Arc::new(
        lash_restate::RestateProcessIngressRunner::new(
            restate_ingress_url.clone(),
            Arc::clone(&process_registry),
        ),
    ));
    let process_work_poke = process_work_runner.poke_handle();
    let queued_work_runner =
        lash::advanced::QueuedWorkRunner::new(Arc::new(WorkbenchQueuedWorkSubmitter {
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
        .store_factory(core_store_factory)
        .attachment_store(Arc::new(lash::persistence::FileAttachmentStore::new(
            data_dir.join("attachments"),
        )))
        .lashlang_artifact_store(artifact_store)
        .trace_sink(Some(Arc::clone(&trace_sink)))
        .lashlang_execution_sink(Some(Arc::clone(&lashlang_execution_sink)))
        .trace_level(TraceLevel::Extended)
        .configure_plugins(|plugins| {
            plugins.push(Arc::new(WorkbenchPluginFactory::new(
                tavily_api_key.clone(),
            )));
            plugins.push(Arc::new(
                lash_plugin_process_controls::ProcessControlsPluginFactory::new(),
            ));
            plugins.push(Arc::new(
                lash_subagents::SubagentsPluginFactory::new(subagent_registry)
                    .with_session_spec(SessionSpec::inherit()),
            ));
        })
        .advanced()
        .effect_host(Arc::new(lash_restate::RestateEffectHost::new()))
        .process_registry(Arc::clone(&process_registry))
        .disable_default_process_work_runner()
        .with_process_work_runner(process_work_poke)
        .with_queued_work_runner(queued_work_poke.clone())
        .build()
        .context("build Lash core")?;
    let process_worker = lash::advanced::DurableProcessWorker::new(
        core.durable_process_worker_config(Arc::clone(&process_registry))
            .context("build Restate process worker config")?,
    );

    let state = AppState {
        core,
        process_registry: Arc::clone(&process_registry),
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
    };
    restate::spawn_restate_endpoint(
        restate_endpoint_addr,
        state.clone(),
        Arc::clone(&process_registry),
        process_worker,
    );
    process_work_runner.spawn();
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
    process_registry: Arc<dyn lash::persistence::ProcessRegistry>,
    session_ids: WorkbenchSessionIds,
    messages: Arc<Mutex<Vec<ChatMessage>>>,
    timeline: Arc<Mutex<Vec<StreamItem>>>,
    selected_model: Arc<Mutex<ModelSelection>>,
    web_configured: bool,
    trace_sink: Option<Arc<dyn TraceSink>>,
    lashlang_execution: Arc<TraceLashlangGraphStore>,
    event_tx: broadcast::Sender<StreamItem>,
    queued_work_poke: lash::advanced::QueuedWorkPoke,
    restate_ingress_url: String,
    restate_http: reqwest::Client,
    restate_cron_job_keys: Arc<Mutex<BTreeSet<String>>>,
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
impl lash::advanced::QueuedWorkRunHandle for WorkbenchQueuedWorkSubmitter {
    async fn run_queued_work(
        &self,
        request: lash::advanced::QueuedWorkRunRequest,
    ) -> std::result::Result<lash::advanced::QueuedWorkRunOutcome, PluginError> {
        let session_id = request
            .session_id
            .unwrap_or_else(|| self.session_ids.current());
        if !self.has_queued_work(&session_id).await? {
            return Ok(lash::advanced::QueuedWorkRunOutcome::Idle);
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
        Ok(lash::advanced::QueuedWorkRunOutcome::Submitted { session_id })
    }
}

impl WorkbenchQueuedWorkSubmitter {
    async fn has_queued_work(&self, session_id: &str) -> std::result::Result<bool, PluginError> {
        let store = self
            .store_factory
            .create_store(&lash::persistence::SessionStoreCreateRequest {
                session_id: session_id.to_string(),
                relation: lash::persistence::SessionRelation::default(),
                policy: lash::advanced::SessionPolicy::default(),
            })
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
impl lash::advanced::QueuedWorkRunHandle for NoopQueuedWorkRunHandle {
    async fn run_queued_work(
        &self,
        _request: lash::advanced::QueuedWorkRunRequest,
    ) -> std::result::Result<lash::advanced::QueuedWorkRunOutcome, PluginError> {
        Ok(lash::advanced::QueuedWorkRunOutcome::Idle)
    }
}

#[cfg(test)]
fn inert_queued_work_poke() -> lash::advanced::QueuedWorkPoke {
    lash::advanced::QueuedWorkRunner::new(Arc::new(NoopQueuedWorkRunHandle)).poke_handle()
}

#[derive(Debug, Serialize)]
struct WorkItem {
    process_id: String,
    graph_key: String,
    kind: String,
    label: String,
    terminal: String,
    created_at_ms: u64,
    updated_at_ms: u64,
    input: Value,
    external_ref: Option<Value>,
    events: Vec<WorkEvent>,
}

#[derive(Debug, Serialize)]
struct WorkEvent {
    sequence: u64,
    event_type: String,
    occurred_at_ms: u64,
    payload: Value,
}

#[derive(Debug, Serialize)]
struct LashlangGraphIndex {
    graphs: Vec<LashlangGraphSummary>,
    lineage_edges: Vec<LashlangGraphLineageEdge>,
}

#[derive(Debug, Serialize)]
struct LashlangGraphSummary {
    graph_key: String,
    title: String,
    status: String,
    kind: String,
    scope: TraceRuntimeScope,
    subject: TraceRuntimeSubject,
    module_ref: String,
    entry_kind: String,
    entry_ref: Option<String>,
    entry_name: String,
    node_count: usize,
    edge_count: usize,
    child_count: usize,
    process: Option<LashlangGraphProcessSummary>,
}

#[derive(Debug, Serialize)]
struct LashlangGraphProcessSummary {
    process_id: String,
    status: String,
    label: Option<String>,
    created_at_ms: u64,
    updated_at_ms: u64,
    input: Value,
    error: Option<String>,
}

#[derive(Debug, Serialize)]
struct LashlangGraphLineageEdge {
    parent_graph_key: String,
    parent_node_id: String,
    bridge_graph_key: String,
    bridge_process_id: Option<String>,
    bridge_status: String,
    bridge_title: String,
    child_graph_key: Option<String>,
    child_session_id: Option<String>,
    pending: bool,
    terminal: bool,
    error: Option<String>,
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
    Ok(Json(StateSnapshot {
        settings: state.settings(),
        messages: Vec::new(),
        timeline: Vec::new(),
    }))
}

async fn list_work(State(state): State<AppState>) -> Result<Json<Vec<WorkItem>>, AppError> {
    let session_id = state.current_session_id();
    let owner_scope = lash::advanced::ProcessScope::new(session_id);
    let entries = state
        .process_registry
        .list_handle_grants(&owner_scope)
        .await
        .map_err(AppError::internal)?;
    let mut work = Vec::with_capacity(entries.len());
    for (grant, record) in entries {
        let events = state
            .process_registry
            .events_after(&record.id, 0)
            .await
            .unwrap_or_default()
            .into_iter()
            .map(|event| WorkEvent {
                sequence: event.sequence,
                event_type: event.event_type,
                occurred_at_ms: system_time_ms(event.occurred_at),
                payload: compact_payload(event.payload),
            })
            .collect::<Vec<_>>();
        let input = serde_json::to_value(&record.input).unwrap_or(Value::Null);
        let inferred_kind = input
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("process")
            .replace('_', " ");
        let kind = grant
            .descriptor
            .kind
            .clone()
            .unwrap_or_else(|| inferred_kind.clone());
        let label = grant
            .descriptor
            .label
            .clone()
            .or_else(|| label_from_input(&input))
            .unwrap_or(inferred_kind);
        let terminal = terminal_label(&record);
        work.push(WorkItem {
            process_id: record.id.clone(),
            graph_key: format!("process:{}", record.id),
            kind,
            label,
            terminal,
            created_at_ms: record.created_at_ms,
            updated_at_ms: record.updated_at_ms,
            input: compact_payload(input),
            external_ref: record
                .external_ref
                .as_ref()
                .and_then(|value| serde_json::to_value(value).ok()),
            events,
        });
    }
    work.sort_by(|left, right| {
        right
            .updated_at_ms
            .cmp(&left.updated_at_ms)
            .then_with(|| right.created_at_ms.cmp(&left.created_at_ms))
    });
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
) -> Result<Json<LashlangGraphIndex>, AppError> {
    let graphs = state.lashlang_execution.graphs();
    let mut graph_summaries = Vec::with_capacity(graphs.len());
    for graph in &graphs {
        let process = graph_process_summary(state.process_registry.as_ref(), graph).await;
        graph_summaries.push(LashlangGraphSummary {
            graph_key: graph.graph_key.clone(),
            title: graph_title(graph),
            status: format!("{:?}", graph.status).to_ascii_lowercase(),
            kind: graph_kind(graph),
            scope: graph.scope.clone(),
            subject: graph.subject.clone(),
            module_ref: graph.module_ref.clone(),
            entry_kind: graph.entry_kind.clone(),
            entry_ref: graph.entry_ref.clone(),
            entry_name: graph.entry_name.clone(),
            node_count: graph.nodes.len(),
            edge_count: graph.edges.len(),
            child_count: graph.children.len(),
            process,
        });
    }
    graph_summaries.sort_by(|left, right| {
        graph_sort_key(&right.scope, &right.graph_key)
            .cmp(&graph_sort_key(&left.scope, &left.graph_key))
    });

    let mut lineage_edges = Vec::new();
    for graph in &graphs {
        for child in &graph.children {
            append_lineage_edges(
                state.process_registry.as_ref(),
                &graphs,
                child,
                &mut lineage_edges,
            )
            .await;
        }
    }
    lineage_edges.sort_by(|left, right| {
        left.parent_graph_key
            .cmp(&right.parent_graph_key)
            .then_with(|| left.parent_node_id.cmp(&right.parent_node_id))
            .then_with(|| left.bridge_graph_key.cmp(&right.bridge_graph_key))
            .then_with(|| left.child_graph_key.cmp(&right.child_graph_key))
    });

    Ok(Json(LashlangGraphIndex {
        graphs: graph_summaries,
        lineage_edges,
    }))
}

async fn lashlang_graph(
    AxumPath(graph_key): AxumPath<String>,
    State(state): State<AppState>,
) -> Result<Json<TraceLashlangGraph>, AppError> {
    state
        .lashlang_execution
        .graph(&graph_key)
        .map(Json)
        .ok_or_else(|| AppError::not_found(format!("no Lashlang graph for `{graph_key}`")))
}

fn started_process_text(count: usize) -> String {
    match count {
        0 => "no trigger handled the event".to_string(),
        1 => "started 1 background process".to_string(),
        count => format!("started {count} background processes"),
    }
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

pub(crate) async fn emit_button_host_event_with_effect_host(
    state: &AppState,
    session: &lash::LashSession,
    button: ButtonChoice,
    pressed_at: &str,
    effect_host: &dyn lash::advanced::EffectHost,
) -> AnyhowResult<lash::HostEventEmitReport> {
    let payload = json!({
        "pressed_at": pressed_at,
        "button": button.as_str(),
        "message": format!("user pressed the {} button", button.lower()),
    });
    state.trace(
        "host_event.emit",
        json!({
            "resource_type": BUTTON_TRIGGER_RESOURCE,
            "alias": BUTTON_TRIGGER_ALIAS,
            "event": BUTTON_TRIGGER_EVENT,
            "source_type": BUTTON_TRIGGER_SOURCE_TYPE,
            "payload": payload.clone(),
        }),
    );
    session
        .host_events()
        .emit_with_effect_host(
            BUTTON_TRIGGER_RESOURCE,
            BUTTON_TRIGGER_ALIAS,
            BUTTON_TRIGGER_EVENT,
            payload,
            effect_host,
        )
        .await
        .context("emit button host event")
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
        self.timeline
            .lock()
            .expect("timeline lock")
            .push(item.clone());
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
        "process_id": item.process_id.clone(),
        "graph_key": item.graph_key.clone(),
        "kind": item.kind.clone(),
        "label": item.label.clone(),
        "terminal": item.terminal.clone(),
        "created_at_ms": item.created_at_ms,
        "updated_at_ms": item.updated_at_ms,
        "input": item.input.clone(),
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

async fn graph_process_summary(
    process_registry: &dyn lash::persistence::ProcessRegistry,
    graph: &TraceLashlangGraph,
) -> Option<LashlangGraphProcessSummary> {
    let TraceRuntimeSubject::Process { process_id } = &graph.subject else {
        return None;
    };
    process_registry
        .get_process(process_id)
        .await
        .map(|record| process_summary_from_record(&record))
}

fn process_summary_from_record(
    record: &lash::advanced::ProcessRecord,
) -> LashlangGraphProcessSummary {
    let input = serde_json::to_value(record.input.as_ref()).unwrap_or(Value::Null);
    LashlangGraphProcessSummary {
        process_id: record.id.clone(),
        status: terminal_label(record),
        label: label_from_input(&input),
        created_at_ms: record.created_at_ms,
        updated_at_ms: record.updated_at_ms,
        input: compact_payload(input),
        error: process_error(record),
    }
}

async fn append_lineage_edges(
    process_registry: &dyn lash::persistence::ProcessRegistry,
    graphs: &[TraceLashlangGraph],
    child: &lash::tracing::TraceLashlangGraphChildLink,
    out: &mut Vec<LashlangGraphLineageEdge>,
) {
    let Some(process_id) = process_id_from_graph_key(&child.child_graph_key) else {
        out.push(LashlangGraphLineageEdge {
            parent_graph_key: child.parent_graph_key.clone(),
            parent_node_id: child.parent_node_id.clone(),
            bridge_graph_key: child.child_graph_key.clone(),
            bridge_process_id: None,
            bridge_status: graph_presence_status(graphs, &child.child_graph_key),
            bridge_title: child
                .child_entry_name
                .clone()
                .unwrap_or_else(|| short_graph_title(&child.child_graph_key)),
            child_graph_key: Some(child.child_graph_key.clone()),
            child_session_id: None,
            pending: !graphs
                .iter()
                .any(|graph| graph.graph_key == child.child_graph_key),
            terminal: false,
            error: None,
        });
        return;
    };

    let record = process_registry.get_process(process_id).await;
    if let Some(record) = record.as_ref()
        && let lash::advanced::ProcessInput::SessionTurn { create_request, .. } =
            record.input.as_ref()
    {
        let child_session_id = create_request.session_id.clone();
        let child_graphs = child_session_id
            .as_deref()
            .map(|session_id| child_session_effect_graphs(graphs, session_id))
            .unwrap_or_default();
        if child_graphs.is_empty() {
            out.push(LashlangGraphLineageEdge {
                parent_graph_key: child.parent_graph_key.clone(),
                parent_node_id: child.parent_node_id.clone(),
                bridge_graph_key: child.child_graph_key.clone(),
                bridge_process_id: Some(process_id.to_string()),
                bridge_status: terminal_label(record),
                bridge_title: lineage_bridge_title(child, Some(record), process_id),
                child_graph_key: None,
                child_session_id,
                pending: !record.is_terminal(),
                terminal: record.is_terminal(),
                error: process_error(record),
            });
        } else {
            for graph in child_graphs {
                out.push(LashlangGraphLineageEdge {
                    parent_graph_key: child.parent_graph_key.clone(),
                    parent_node_id: child.parent_node_id.clone(),
                    bridge_graph_key: child.child_graph_key.clone(),
                    bridge_process_id: Some(process_id.to_string()),
                    bridge_status: terminal_label(record),
                    bridge_title: lineage_bridge_title(child, Some(record), process_id),
                    child_graph_key: Some(graph.graph_key.clone()),
                    child_session_id: child_session_id.clone(),
                    pending: false,
                    terminal: record.is_terminal(),
                    error: process_error(record),
                });
            }
        }
        return;
    }

    let child_graph_observed = graphs
        .iter()
        .any(|graph| graph.graph_key == child.child_graph_key);
    let terminal = record
        .as_ref()
        .map(lash::advanced::ProcessRecord::is_terminal)
        .unwrap_or(false);
    out.push(LashlangGraphLineageEdge {
        parent_graph_key: child.parent_graph_key.clone(),
        parent_node_id: child.parent_node_id.clone(),
        bridge_graph_key: child.child_graph_key.clone(),
        bridge_process_id: Some(process_id.to_string()),
        bridge_status: record
            .as_ref()
            .map(terminal_label)
            .unwrap_or_else(|| graph_presence_status(graphs, &child.child_graph_key)),
        bridge_title: lineage_bridge_title(child, record.as_ref(), process_id),
        child_graph_key: Some(child.child_graph_key.clone()),
        child_session_id: None,
        pending: !child_graph_observed && !terminal,
        terminal,
        error: record.as_ref().and_then(process_error),
    });
}

fn child_session_effect_graphs<'graph>(
    graphs: &'graph [TraceLashlangGraph],
    session_id: &str,
) -> Vec<&'graph TraceLashlangGraph> {
    let mut matches = graphs
        .iter()
        .filter(|graph| {
            graph.scope.session_id == session_id
                && matches!(
                    &graph.subject,
                    TraceRuntimeSubject::Effect { kind, .. } if kind == "exec_code"
                )
        })
        .collect::<Vec<_>>();
    matches.sort_by(|left, right| {
        graph_sort_key(&left.scope, &left.graph_key)
            .cmp(&graph_sort_key(&right.scope, &right.graph_key))
    });
    matches
}

fn graph_sort_key(scope: &TraceRuntimeScope, graph_key: &str) -> (usize, usize, String) {
    (
        scope.turn_index.unwrap_or_default(),
        scope.protocol_iteration.unwrap_or_default(),
        graph_key.to_string(),
    )
}

fn graph_kind(graph: &TraceLashlangGraph) -> String {
    match &graph.subject {
        TraceRuntimeSubject::Effect { kind, .. } if kind == "exec_code" => "foreground".to_string(),
        TraceRuntimeSubject::Effect { kind, .. } => kind.clone(),
        TraceRuntimeSubject::Process { .. } => "process".to_string(),
    }
}

fn graph_title(graph: &TraceLashlangGraph) -> String {
    match &graph.subject {
        TraceRuntimeSubject::Effect { .. } if graph.entry_name == "main" => {
            "foreground Lashlang".to_string()
        }
        TraceRuntimeSubject::Effect { .. } => graph.entry_name.clone(),
        TraceRuntimeSubject::Process { process_id } => {
            if graph.entry_name.trim().is_empty() {
                process_id.clone()
            } else {
                graph.entry_name.clone()
            }
        }
    }
}

fn process_id_from_graph_key(graph_key: &str) -> Option<&str> {
    graph_key
        .strip_prefix("process:")
        .filter(|id| !id.is_empty())
}

fn graph_presence_status(graphs: &[TraceLashlangGraph], graph_key: &str) -> String {
    if graphs.iter().any(|graph| graph.graph_key == graph_key) {
        "observed".to_string()
    } else {
        "pending".to_string()
    }
}

fn lineage_bridge_title(
    child: &lash::tracing::TraceLashlangGraphChildLink,
    record: Option<&lash::advanced::ProcessRecord>,
    process_id: &str,
) -> String {
    record
        .and_then(|record| {
            let input = serde_json::to_value(record.input.as_ref()).ok()?;
            label_from_input(&input)
        })
        .or_else(|| child.child_entry_name.clone())
        .unwrap_or_else(|| process_id.to_string())
}

fn short_graph_title(graph_key: &str) -> String {
    if let Some(process_id) = process_id_from_graph_key(graph_key) {
        return process_id.to_string();
    }
    graph_key.to_string()
}

fn process_error(record: &lash::advanced::ProcessRecord) -> Option<String> {
    match &record.status {
        lash::advanced::ProcessStatus::Failed { await_output }
        | lash::advanced::ProcessStatus::Cancelled { await_output } => {
            serde_json::to_value(await_output).ok().and_then(|value| {
                value
                    .get("message")
                    .and_then(Value::as_str)
                    .map(str::to_string)
            })
        }
        lash::advanced::ProcessStatus::Running
        | lash::advanced::ProcessStatus::Completed { .. } => None,
    }
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
    lash::ModelSpec::from_token_limits(
        model,
        model_variant,
        DEFAULT_CONTEXT_WINDOW_TOKENS,
        None,
        None,
    )
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

fn terminal_label(record: &lash::advanced::ProcessRecord) -> String {
    record
        .status
        .terminal_semantics()
        .map(|terminal| format!("{:?}", terminal.state).to_ascii_lowercase())
        .unwrap_or_else(|| "running".to_string())
}

fn label_from_input(input: &Value) -> Option<String> {
    input
        .pointer("/call/name")
        .or_else(|| input.get("process_name"))
        .and_then(Value::as_str)
        .map(str::to_string)
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

fn system_time_ms(time: SystemTime) -> u64 {
    time.duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or_default()
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
}

impl WorkbenchPluginFactory {
    fn new(tavily_api_key: impl Into<String>) -> Self {
        Self {
            tavily_api_key: tavily_api_key.into(),
        }
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
        }))
    }
}

struct WorkbenchSessionPlugin {
    tavily_api_key: String,
}

impl SessionPlugin for WorkbenchSessionPlugin {
    fn id(&self) -> &'static str {
        "agent_workbench"
    }

    fn register(&self, reg: &mut PluginRegistrar) -> Result<(), PluginError> {
        reg.prompt().contribute(Arc::new(|_ctx| {
            Box::pin(async {
                Ok(vec![PromptContribution::environment(
                    "Agent Workbench",
                    WORKBENCH_PROMPT.to_string(),
                )])
            })
        }));
        reg.host_events().declare(HostEvent::new(
            BUTTON_TRIGGER_RESOURCE,
            BUTTON_TRIGGER_ALIAS,
            BUTTON_TRIGGER_EVENT,
            button_trigger_event_type(),
        ))?;
        reg.tools()
            .provider(Arc::new(lash_tool_web::web_search_provider(
                self.tavily_api_key.clone(),
            )))?;
        reg.tools()
            .provider(Arc::new(lash_tool_web::fetch_url_provider(
                self.tavily_api_key.clone(),
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
            ["ui", "button", "pressed"],
            lashlang::TypeExpr::Object(vec![]),
            button_trigger_event_type(),
        )
        .expect("valid button trigger source");
    resources
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

const WORKBENCH_PROMPT: &str = r#"You are running inside the Agent Workbench demo.

Available host features:
- Web access is limited to `web.search(...)` and `web.fetch(...)`, both backed by the same Tavily tools the CLI uses.
- You may call `agents.spawn(...)` for independent investigation.
- You may use Lashlang process definitions for work that should run independently. A `start` or trigger activation creates a process run; a trigger registration is the durable rule that creates future runs.
- When you start a process and need its `finish` value, write `result = (await handle)?`. Bare `await handle` waits, but returns the result wrapper, so `result.field` will not read fields from the finished value.
- The red and blue UI buttons emit `ui.button.pressed`. Register `ui.button.pressed({})`; the selected button arrives in the event payload, not in the source config:

```lash
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
submit {
  handle: handle,
  registrations: await triggers.list({ name: "button watcher" })?
}
```

- For schedule requests, build `cron.Schedule(...)` values and register a process definition with explicit `inputs`. Use `trigger.event` directly for the `cron.Tick` param, for example `inputs: { tick: trigger.event }`. The workbench syncs enabled `cron.Schedule` registrations to Restate cron objects, which activate the trigger with `cron.Tick { fired_at: str }`; use a seconds expression such as `*/10 * * * * *` when the user wants a quick smoke test. Use `await triggers.list({})?` to discover registrations and `await triggers.cancel({ handle: handle })?` to disable future activations.

Use background processes or subagents only when they clarify the user's request or make parallel progress. Keep the visible answer concise and mention any background work you started."#;

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

    fn run_async_test_on_large_stack<F, Fut>(name: &str, test: F)
    where
        F: FnOnce() -> Fut + Send + 'static,
        Fut: Future<Output = ()> + 'static,
    {
        std::thread::Builder::new()
            .name(name.to_string())
            .stack_size(16 * 1024 * 1024)
            .spawn(|| {
                tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("tokio runtime")
                    .block_on(test())
            })
            .expect("spawn large-stack test thread")
            .join()
            .expect("large-stack test thread");
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
    fn graph_index_resolves_subagent_bridge_to_child_session_effect_graph() {
        run_async_test_on_large_stack("workbench-graph-index-subagent-bridge", || async {
            use lash::persistence::ProcessRegistry as _;

            let registry = lash_core::TestLocalProcessRegistry::default();
            let child_session_id = "child-session";
            let create_request = lash_core::SessionCreateRequest::child_session(
                "root",
                lash_core::SessionStartPoint::Empty,
                lash_core::PluginOptions::default(),
            )
            .with_session_id(child_session_id);
            registry
                .register_process(lash::advanced::ProcessRegistration::new(
                    "subagent-process",
                    lash::ProcessInput::SessionTurn {
                        create_request: Box::new(create_request),
                        turn_input: Box::new(lash::TurnInput::text("run child")),
                        output_contract: lash::tools::ToolOutputContract::Static,
                    },
                ))
                .await
                .expect("register subagent process");

            let parent_graph = TraceLashlangGraph {
                graph_key: "effect:root:turn-1:exec-1".to_string(),
                scope: TraceRuntimeScope {
                    session_id: "root".to_string(),
                    turn_id: Some("turn-1".to_string()),
                    turn_index: Some(0),
                    protocol_iteration: Some(0),
                },
                subject: TraceRuntimeSubject::Effect {
                    effect_id: "exec-1".to_string(),
                    kind: "exec_code".to_string(),
                },
                module_ref: "parent-module".to_string(),
                entry_kind: "main".to_string(),
                entry_ref: None,
                entry_name: "main".to_string(),
                status: TraceLashlangStatus::Running,
                nodes: Vec::new(),
                edges: Vec::new(),
                children: vec![TraceLashlangGraphChildLink {
                    parent_graph_key: "effect:root:turn-1:exec-1".to_string(),
                    parent_node_id: "spawn".to_string(),
                    child_graph_key: "process:subagent-process".to_string(),
                    child_module_ref: None,
                    child_entry_ref: None,
                    child_entry_name: Some("subagent".to_string()),
                }],
            };
            let child_graph = TraceLashlangGraph {
                graph_key: "effect:child-session:turn-1:exec-1".to_string(),
                scope: TraceRuntimeScope {
                    session_id: child_session_id.to_string(),
                    turn_id: Some("turn-1".to_string()),
                    turn_index: Some(0),
                    protocol_iteration: Some(0),
                },
                subject: TraceRuntimeSubject::Effect {
                    effect_id: "exec-1".to_string(),
                    kind: "exec_code".to_string(),
                },
                module_ref: "child-module".to_string(),
                entry_kind: "main".to_string(),
                entry_ref: None,
                entry_name: "main".to_string(),
                status: TraceLashlangStatus::Completed,
                nodes: Vec::new(),
                edges: Vec::new(),
                children: Vec::new(),
            };
            let graphs = vec![parent_graph.clone(), child_graph.clone()];
            let mut lineage_edges = Vec::new();

            append_lineage_edges(
                &registry,
                &graphs,
                &parent_graph.children[0],
                &mut lineage_edges,
            )
            .await;

            assert_eq!(lineage_edges.len(), 1);
            let edge = &lineage_edges[0];
            assert_eq!(edge.bridge_process_id.as_deref(), Some("subagent-process"));
            assert_eq!(edge.bridge_graph_key, "process:subagent-process");
            assert_eq!(edge.child_session_id.as_deref(), Some(child_session_id));
            assert_eq!(
                edge.child_graph_key.as_deref(),
                Some(child_graph.graph_key.as_str())
            );
            assert!(!edge.pending);
        });
    }

    #[test]
    fn workbench_does_not_define_a_local_lashlang_execution_reducer() {
        let source = include_str!("main.rs");
        let old_reducer_decl = format!("struct {}{}", "LashlangExecution", "Reducer");

        assert!(!source.contains(&old_reducer_decl));
    }

    #[test]
    fn workbench_execution_explorer_cutover_removed_process_modal_names() {
        let main_source = include_str!("main.rs");
        let ui_source = include_str!("ui.rs");
        let removed = [
            format!("/api/{}-graph", "process"),
            format!("open{}Graph", "Process"),
            format!("render{}Graph", "Process"),
            format!("refresh{}Graph", "Process"),
            format!("graph{}ProcessId", "Overlay"),
            format!("graph{}", "Overlay"),
        ];
        for removed in removed {
            assert!(
                !main_source.contains(&removed),
                "`{removed}` remains in main.rs"
            );
            assert!(
                !ui_source.contains(&removed),
                "`{removed}` remains in ui.rs"
            );
        }
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
    fn button_trigger_activation_starts_visible_lashlang_process() {
        run_async_test_on_large_stack("workbench-button-trigger-test", || {
            button_trigger_activation_starts_visible_lashlang_process_inner()
        });
    }

    async fn button_trigger_activation_starts_visible_lashlang_process_inner() {
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
            lash_sqlite_store::SqliteProcessRegistry::open(&db_path).expect("open registry"),
        ) as Arc<dyn lash::persistence::ProcessRegistry>;
        let provider = trigger_registration_provider();
        let model = lash::ModelSpec::from_token_limits("test-model", None, 4096, None, None)
            .expect("model spec");
        let session_ids = WorkbenchSessionIds::fresh();
        let session_id = session_ids.current();
        let core = LashCore::builder()
            .install_mode(ModePreset::rlm_with_config(
                lash::modes::RlmProtocolPluginConfig::default()
                    .with_lashlang_abilities(workbench_lashlang_abilities()),
            ))
            .default_mode(ModeId::rlm())
            .provider(provider)
            .model(model)
            .store_factory(core_store_factory)
            .in_memory_stores()
            .plugin(Arc::new(WorkbenchPluginFactory::new("")))
            .advanced()
            .effect_host(Arc::new(lash::advanced::InlineEffectHost::default()))
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
            .active_definitions()
            .await
            .expect("active tools")
            .into_iter()
            .map(|tool| tool.name)
            .collect::<Vec<_>>();
        let removed_tool_name = ["attach", "button", "trigger"].join("_");
        assert!(!tool_names.iter().any(|name| name == &removed_tool_name));

        let report = emit_test_button_trigger(&session, ButtonChoice::Red).await;

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

        let state = AppState {
            core,
            process_registry: Arc::clone(&process_registry),
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
        };
        let target_scope_id = lash::advanced::ProcessScope::new(state.current_session_id()).id();
        let session_store =
            lash_sqlite_store::Store::open(&session_store_factory.path_for_session(&session_id))
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
        assert_eq!(work[0].terminal, "completed");
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
    fn button_trigger_activation_is_submitted_to_restate_workflow() {
        run_async_test_on_large_stack("workbench-trigger-restate-test", || {
            button_trigger_activation_is_submitted_to_restate_workflow_inner()
        });
    }

    async fn button_trigger_activation_is_submitted_to_restate_workflow_inner() {
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
                .expect("open registry"),
        ) as Arc<dyn lash::persistence::ProcessRegistry>;
        let artifact_store = Arc::new(
            lash_sqlite_store::Store::open(&data_dir.join("artifacts.db"))
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
        let model = lash::ModelSpec::from_token_limits("test-model", None, 4096, None, None)
            .expect("model spec");
        let (restate_ingress_url, mut restate_requests) = spawn_restate_ingress_capture().await;
        let (event_tx, _) = broadcast::channel(1024);
        let state = AppState {
            core: LashCore::builder()
                .install_mode(ModePreset::rlm_with_config(
                    lash::modes::RlmProtocolPluginConfig::default()
                        .with_lashlang_abilities(workbench_lashlang_abilities()),
                ))
                .default_mode(ModeId::rlm())
                .provider(provider)
                .model(model)
                .store_factory(core_store_factory)
                .plugin(Arc::new(WorkbenchPluginFactory::new("")))
                .advanced()
                .runtime_host_config({
                    let mut config = lash::advanced::RuntimeHostConfig::in_memory();
                    config.durability.lashlang_artifact_store = artifact_store_for_core;
                    config.control.effect_host =
                        Arc::new(lash::advanced::InlineEffectHost::default());
                    config
                })
                .process_registry(Arc::clone(&process_registry))
                .build()
                .expect("build core"),
            process_registry,
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
        run_async_test_on_large_stack("workbench-reset-chat-test", || {
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
                .expect("open registry"),
        ) as Arc<dyn lash::persistence::ProcessRegistry>;
        let provider = trigger_registration_provider();
        let model = lash::ModelSpec::from_token_limits("test-model", None, 4096, None, None)
            .expect("model spec");
        let (restate_ingress_url, mut restate_requests) = spawn_restate_ingress_capture().await;
        let core = LashCore::builder()
            .install_mode(ModePreset::rlm_with_config(
                lash::modes::RlmProtocolPluginConfig::default()
                    .with_lashlang_abilities(workbench_lashlang_abilities()),
            ))
            .default_mode(ModeId::rlm())
            .provider(provider)
            .model(model)
            .store_factory(core_store_factory)
            .in_memory_stores()
            .plugin(Arc::new(WorkbenchPluginFactory::new("")))
            .advanced()
            .effect_host(Arc::new(lash::advanced::InlineEffectHost::default()))
            .process_registry(Arc::clone(&process_registry))
            .build()
            .expect("build core");
        let state = AppState {
            core,
            process_registry: Arc::clone(&process_registry),
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
        let started = emit_test_button_trigger(&session, ButtonChoice::Red).await;
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
        let _ = std::fs::remove_dir_all(data_dir);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    #[ignore = "requires a running Restate server; use `just agent-workbench-restate-e2e`"]
    async fn live_restate_cron_runs_trigger_and_queued_turn_end_to_end() {
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
        let harness = live_workbench_restate_state(&data_dir, ingress_url);
        restate::spawn_restate_endpoint(
            endpoint_bind,
            harness.state.clone(),
            Arc::clone(&harness.process_registry),
            harness.process_worker,
        );
        wait_for_endpoint_socket(endpoint_bind).await;
        register_restate_deployment(&admin_url, &endpoint_url).await;
        harness.process_work_runner.spawn();
        harness.queued_work_runner.spawn();

        let Json(accepted) = send_turn(
            State(harness.state.clone()),
            Json(TurnRequest {
                text: "Register a cron trigger that runs every two seconds and reports the tick."
                    .to_string(),
                model: None,
                model_variant: None,
            }),
        )
        .await
        .expect("submit Restate-backed workbench turn");
        assert!(accepted.accepted);
        wait_for_workbench_message(
            &harness.state,
            "cron tick observed",
            Duration::from_secs(30),
        )
        .await;
        assert!(
            !harness
                .state
                .restate_cron_job_keys
                .lock()
                .expect("cron job key lock")
                .is_empty(),
            "cron trigger registration should sync to a Restate cron object"
        );
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

    struct LiveWorkbenchRestateHarness {
        state: AppState,
        process_registry: Arc<dyn lash::persistence::ProcessRegistry>,
        process_worker: lash::advanced::DurableProcessWorker,
        process_work_runner: lash::advanced::ProcessWorkRunner,
        queued_work_runner: lash::advanced::QueuedWorkRunner,
        trace_path: PathBuf,
    }

    fn live_workbench_restate_state(
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
                .expect("open process registry"),
        ) as Arc<dyn lash::persistence::ProcessRegistry>;
        let artifact_store = Arc::new(
            lash_sqlite_store::Store::open(&data_dir.join("artifacts.db"))
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
        let model = lash::ModelSpec::from_token_limits(
            "mock-model",
            Some("high".to_string()),
            4096,
            None,
            None,
        )
        .expect("model spec");
        let process_work_runner = lash::advanced::ProcessWorkRunner::new(Arc::new(
            lash_restate::RestateProcessIngressRunner::new(
                restate_ingress_url.clone(),
                Arc::clone(&process_registry),
            ),
        ));
        let process_work_poke = process_work_runner.poke_handle();
        let session_ids = WorkbenchSessionIds::fresh();
        let restate_http = reqwest::Client::new();
        let queued_work_runner =
            lash::advanced::QueuedWorkRunner::new(Arc::new(WorkbenchQueuedWorkSubmitter {
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
            .store_factory(core_store_factory)
            .attachment_store(Arc::new(lash::persistence::FileAttachmentStore::new(
                data_dir.join("attachments"),
            )))
            .lashlang_artifact_store(artifact_store)
            .trace_sink(Some(Arc::clone(&trace_sink)))
            .lashlang_execution_sink(Some(lashlang_execution_sink))
            .trace_level(TraceLevel::Extended)
            .plugin(Arc::new(WorkbenchPluginFactory::new("")))
            .advanced()
            .effect_host(Arc::new(lash::advanced::InlineEffectHost::default()))
            .process_registry(Arc::clone(&process_registry))
            .disable_default_process_work_runner()
            .with_process_work_runner(process_work_poke)
            .with_queued_work_runner(queued_work_poke.clone())
            .build()
            .expect("build core");
        let process_worker = lash::advanced::DurableProcessWorker::new(
            core.durable_process_worker_config(Arc::clone(&process_registry))
                .expect("build process worker config"),
        );
        let (event_tx, _) = broadcast::channel(1024);
        let state = AppState {
            core,
            process_registry: Arc::clone(&process_registry),
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
        };
        LiveWorkbenchRestateHarness {
            state,
            process_registry,
            process_worker,
            process_work_runner,
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
        run_async_test_on_large_stack("workbench-persisted-trigger-test", || {
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
        let artifact_store_path = data_dir.join("artifacts.db");
        let session_id = WorkbenchSessionIds::fresh().current();

        {
            let artifact_store = Arc::new(
                lash_sqlite_store::Store::open(&artifact_store_path).expect("open artifacts"),
            );
            let artifact_store_for_core: Arc<dyn lashlang::LashlangArtifactStore> =
                artifact_store.clone();
            let process_registry = Arc::new(
                lash_sqlite_store::SqliteProcessRegistry::open(&process_registry_path)
                    .expect("open registry"),
            ) as Arc<dyn lash::persistence::ProcessRegistry>;
            let core = test_workbench_core(
                session_store_factory.clone(),
                process_registry,
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
            lash_sqlite_store::Store::open(&artifact_store_path).expect("reopen artifacts"),
        );
        let artifact_store_for_core: Arc<dyn lashlang::LashlangArtifactStore> =
            artifact_store.clone();
        let process_registry = Arc::new(
            lash_sqlite_store::SqliteProcessRegistry::open(&process_registry_path)
                .expect("reopen registry"),
        ) as Arc<dyn lash::persistence::ProcessRegistry>;
        let core = test_workbench_core(
            session_store_factory,
            Arc::clone(&process_registry),
            artifact_store_for_core,
        );
        let reopened = core
            .session(session_id)
            .rlm()
            .open()
            .await
            .expect("reopen session");
        let report = emit_test_button_trigger(&reopened, ButtonChoice::Blue).await;
        assert_eq!(report.started_process_ids.len(), 1);
        process_registry
            .await_process(&report.started_process_ids[0])
            .await
            .expect("trigger process should finish");

        let _ = std::fs::remove_dir_all(data_dir);
    }

    fn test_workbench_core(
        session_store_factory: Arc<dyn lash::persistence::SessionStoreFactory>,
        process_registry: Arc<dyn lash::persistence::ProcessRegistry>,
        artifact_store: Arc<dyn lashlang::LashlangArtifactStore>,
    ) -> LashCore {
        let provider = trigger_registration_provider();
        let model = lash::ModelSpec::from_token_limits("test-model", None, 4096, None, None)
            .expect("model spec");
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
            .advanced()
            .runtime_host_config({
                let mut config = lash::advanced::RuntimeHostConfig::in_memory();
                config.durability.lashlang_artifact_store = artifact_store;
                config.control.effect_host = Arc::new(lash::advanced::InlineEffectHost::default());
                config
            })
            .process_registry(process_registry)
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
            .run()
            .await
            .expect("register trigger route");
        assert_eq!(
            output.submitted_value(),
            Some(&serde_json::json!("registered"))
        );
    }

    async fn emit_test_button_trigger(
        session: &lash::LashSession,
        button: ButtonChoice,
    ) -> lash::HostEventEmitReport {
        session
            .host_events()
            .emit(
                BUTTON_TRIGGER_RESOURCE,
                BUTTON_TRIGGER_ALIAS,
                BUTTON_TRIGGER_EVENT,
                json!({
                    "button": button.as_str(),
                    "message": format!("user pressed the {} button", button.lower()),
                    "pressed_at": "2026-06-02T12:00:00Z"
                }),
            )
            .await
            .expect("emit button trigger")
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
