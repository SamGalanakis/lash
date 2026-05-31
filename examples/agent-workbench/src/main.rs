mod ui;

use std::collections::BTreeMap;
use std::convert::Infallible;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result as AnyhowResult, anyhow};
use async_trait::async_trait;
use axum::body::Body;
use axum::extract::State;
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
    TurnInput, TurnResult,
    tracing::{
        JsonlTraceSink, StderrTraceSink, TeeTraceSink, TraceContext, TraceEvent, TraceLevel,
        TraceRecord, TraceSink,
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
const BUTTON_TRIGGER_RESOURCE: &str = "Button";
const BUTTON_TRIGGER_ALIAS: &str = "ui.button";
const BUTTON_TRIGGER_EVENT: &str = "pressed";

#[tokio::main]
async fn main() -> AnyhowResult<()> {
    let _ = dotenvy::dotenv();

    let addr: SocketAddr = std::env::var("AGENT_WORKBENCH_ADDR")
        .unwrap_or_else(|_| "127.0.0.1:3030".to_string())
        .parse()
        .context("invalid AGENT_WORKBENCH_ADDR")?;
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

    let api_key = std::env::var("OPENROUTER_API_KEY").unwrap_or_default();
    if api_key.trim().is_empty() {
        eprintln!("warning: OPENROUTER_API_KEY is empty; turns will fail until it is set");
    }
    let tavily_api_key = std::env::var("TAVILY_API_KEY").unwrap_or_default();
    if tavily_api_key.trim().is_empty() {
        eprintln!("warning: TAVILY_API_KEY is empty; web tools will return configuration errors");
    }
    let model = std::env::var("OPENROUTER_MODEL").unwrap_or_else(|_| "openai/gpt-5.5".to_string());
    let model_variant =
        std::env::var("OPENROUTER_MODEL_VARIANT").unwrap_or_else(|_| "medium".to_string());

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
    ) as Arc<dyn lash::advanced::ProcessRegistry>;
    // Deployment-level Lashlang artifact store (compiled trigger/process
    // modules), shared across the session tree. SQLite keeps installed triggers
    // durable across restarts.
    let artifact_store = Arc::new(
        lash_sqlite_store::Store::open(&data_dir.join("artifacts.db"))
            .context("open lashlang artifact store")?,
    ) as Arc<dyn lash::persistence::LashlangArtifactStore>;
    let subagent_registry = Arc::new(lash_subagents::default_registry(&BTreeMap::new()));
    let session_ids = WorkbenchSessionIds::fresh();
    let (queue_runner, runner_rx) = SessionQueueRunner::channel();
    let (event_tx, _) = broadcast::channel(1024);

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
        .effect_controller(Arc::new(
            lash::advanced::InlineRuntimeEffectController::default(),
        ))
        .process_registry(Arc::clone(&process_registry))
        .build()
        .context("build Lash core")?;

    let state = AppState {
        core,
        process_registry,
        session_ids,
        messages: Arc::new(Mutex::new(Vec::new())),
        default_model: model,
        default_model_variant: Some(model_variant),
        web_configured: !tavily_api_key.trim().is_empty(),
        trace_sink: Some(Arc::clone(&trace_sink)),
        event_tx,
        queue_runner,
    };
    spawn_session_queue_runner(state.clone(), runner_rx);
    emit_workbench_trace(
        &state.trace_sink,
        None,
        "startup",
        json!({
            "addr": addr.to_string(),
            "data_dir": data_dir.display().to_string(),
            "trace_path": trace_path_display,
            "model": state.default_model.clone(),
            "model_variant": state.default_model_variant.clone(),
            "web_configured": state.web_configured,
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
        .with_state(state);

    println!("agent-workbench listening on http://{addr}");
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .context("bind listener")?;
    axum::serve(listener, app).await.context("serve")?;
    Ok(())
}

#[derive(Clone)]
struct AppState {
    core: LashCore,
    process_registry: Arc<dyn lash::advanced::ProcessRegistry>,
    session_ids: WorkbenchSessionIds,
    messages: Arc<Mutex<Vec<ChatMessage>>>,
    default_model: String,
    default_model_variant: Option<String>,
    web_configured: bool,
    trace_sink: Option<Arc<dyn TraceSink>>,
    event_tx: broadcast::Sender<StreamItem>,
    queue_runner: SessionQueueRunner,
}

#[derive(Clone, Debug, Serialize)]
struct Settings {
    default_model: String,
    default_model_variant: Option<String>,
    web_configured: bool,
    model_variants: Vec<&'static str>,
    session_id: String,
}

#[derive(Clone, Debug, Serialize)]
struct StateSnapshot {
    settings: Settings,
    messages: Vec<ChatMessage>,
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
enum ButtonChoice {
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
struct SessionQueueRunner {
    tx: mpsc::UnboundedSender<QueueRunnerCommand>,
}

#[derive(Debug)]
struct QueueRunnerCommand {
    reason: String,
}

impl SessionQueueRunner {
    fn channel() -> (Self, mpsc::UnboundedReceiver<QueueRunnerCommand>) {
        let (tx, rx) = mpsc::unbounded_channel();
        (Self { tx }, rx)
    }

    #[cfg(test)]
    fn inert() -> Self {
        let (runner, _rx) = Self::channel();
        runner
    }

    fn poke(&self, reason: impl Into<String>) {
        let _ = self.tx.send(QueueRunnerCommand {
            reason: reason.into(),
        });
    }
}

#[derive(Debug, Serialize)]
struct WorkItem {
    process_id: String,
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

async fn index() -> Html<&'static str> {
    Html(ui::INDEX_HTML)
}

async fn app_state(State(state): State<AppState>) -> Json<StateSnapshot> {
    Json(StateSnapshot {
        settings: state.settings(),
        messages: state.messages_snapshot(),
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
    queue_user_turn(&state, text, turn_model).await?;
    state.queue_runner.poke("user_turn");
    Ok(Json(CommandAccepted { accepted: true }))
}

async fn button_trigger(
    State(state): State<AppState>,
    Json(request): Json<ButtonEventRequest>,
) -> Result<Json<CommandAccepted>, AppError> {
    state.trace(
        "api.button_trigger.request",
        json!({
            "button": request.button,
        }),
    );
    let pressed_at = Utc::now().to_rfc3339();
    state.push_message("event", format!("{} button event", request.button.lower()));
    match emit_button_host_event(&state, request.button, &pressed_at).await {
        Ok(report) => {
            state.trace(
                "button_trigger.host_event_report",
                json!({
                    "button": request.button,
                    "started_process_ids": report.started_process_ids.clone(),
                }),
            );
            state.push_message(
                "event",
                started_process_text(report.started_process_ids.len()),
            );
            state.queue_runner.poke("host_event");
            Ok(Json(CommandAccepted { accepted: true }))
        }
        Err(err) => {
            state.trace(
                "button_trigger.host_event_failed",
                json!({
                    "button": request.button,
                    "error": err.to_string(),
                }),
            );
            Err(AppError::internal(err))
        }
    }
}

async fn reset_chat(State(state): State<AppState>) -> Result<Json<StateSnapshot>, AppError> {
    let old_session_id = state.current_session_id();
    state
        .core
        .delete_session(&old_session_id)
        .await
        .map_err(AppError::internal)?;
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
    state
        .core
        .session(new_session_id)
        .rlm()
        .open()
        .await
        .map_err(AppError::internal)?;
    state.messages.lock().expect("messages lock").clear();
    Ok(Json(StateSnapshot {
        settings: state.settings(),
        messages: Vec::new(),
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

async fn queue_user_turn(
    state: &AppState,
    turn_text: String,
    turn_model: lash::ModelSpec,
) -> Result<(), AppError> {
    let session = state
        .core
        .session(state.current_session_id())
        .rlm()
        .open()
        .await
        .map_err(AppError::internal)?;
    let mut input = TurnInput::text(turn_text.clone());
    input.turn_context.set_model(turn_model.clone());
    input.protocol_turn_options = Some(lash::advanced::ProtocolTurnOptions {
        payload: json!({
            "kind": "submit_required",
            "schema": null,
        }),
    });
    session
        .queue(input)
        .send()
        .await
        .map_err(AppError::internal)?;
    state.trace(
        "turn.queued",
        json!({
            "turn_text": turn_text,
            "model": serde_json::to_value(&turn_model).unwrap_or(Value::Null),
        }),
    );
    Ok(())
}

fn spawn_session_queue_runner(
    state: AppState,
    mut rx: mpsc::UnboundedReceiver<QueueRunnerCommand>,
) {
    tokio::spawn(async move {
        let mut poll = tokio::time::interval(Duration::from_millis(400));
        poll.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            tokio::select! {
                command = rx.recv() => {
                    let Some(command) = command else {
                        break;
                    };
                    drain_session_queue(&state, &command.reason, true).await;
                }
                _ = poll.tick() => {
                    drain_session_queue(&state, "poll", false).await;
                }
            }
        }
    });
}

async fn drain_session_queue(state: &AppState, reason: &str, trace_idle: bool) {
    let mut ran_work = false;
    loop {
        match run_next_queued_turn(state, reason).await {
            Ok(true) => {
                ran_work = true;
            }
            Ok(false) => {
                if ran_work || trace_idle {
                    state.trace(
                        "queue_runner.idle",
                        json!({
                            "reason": reason,
                            "ran_work": ran_work,
                        }),
                    );
                    state.publish(StreamItem::Done);
                }
                break;
            }
            Err(err) => {
                state.trace(
                    "queue_runner.error",
                    json!({
                        "reason": reason,
                        "error": err.to_string(),
                    }),
                );
                state.publish(StreamItem::Error {
                    message: err.to_string(),
                });
                state.publish(StreamItem::Done);
                break;
            }
        }
    }
}

async fn run_next_queued_turn(state: &AppState, reason: &str) -> Result<bool, lash::EmbedError> {
    let session = state
        .core
        .session(state.current_session_id())
        .rlm()
        .open()
        .await?;
    let turn_state = Arc::new(Mutex::new(TurnStreamState::default()));
    let ui_events = ChannelTurnEvents {
        state: state.clone(),
        turn_state: Arc::clone(&turn_state),
    };
    if session.queued_work().await?.is_empty() {
        return Ok(false);
    }
    state.trace(
        "queue_runner.start",
        json!({
            "reason": reason,
            "session_id": state.current_session_id(),
        }),
    );
    let Some(output) = session.next_queued_turn().stream(&ui_events).await? else {
        return Ok(false);
    };
    let streamed_prose = turn_state
        .lock()
        .expect("turn state lock")
        .assistant_prose
        .clone();
    let assistant_text = assistant_text_for_display(&output, &streamed_prose);
    state.trace(
        "queued_turn.completed",
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
    Ok(true)
}

fn started_process_text(count: usize) -> String {
    match count {
        0 => "no trigger handled the button event".to_string(),
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

async fn emit_button_host_event(
    state: &AppState,
    button: ButtonChoice,
    pressed_at: &str,
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
            "payload": payload.clone(),
        }),
    );
    let session = state
        .core
        .session(state.current_session_id())
        .rlm()
        .open()
        .await
        .context("open workbench session")?;
    session
        .host_events()
        .emit(
            BUTTON_TRIGGER_RESOURCE,
            BUTTON_TRIGGER_ALIAS,
            BUTTON_TRIGGER_EVENT,
            payload,
        )
        .await
        .context("emit button host event")
}

fn button_trigger_event_type() -> lashlang::TypeExpr {
    lashlang::TypeExpr::Object(vec![
        lashlang::TypeField {
            name: "button".into(),
            ty: lashlang::TypeExpr::Enum(vec!["Red".into(), "Blue".into()]),
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
    ])
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

    fn settings(&self) -> Settings {
        Settings {
            default_model: self.default_model.clone(),
            default_model_variant: self.default_model_variant.clone(),
            web_configured: self.web_configured,
            model_variants: vec!["low", "medium", "high"],
            session_id: self.current_session_id(),
        }
    }

    fn messages_snapshot(&self) -> Vec<ChatMessage> {
        self.messages.lock().expect("messages lock").clone()
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
    let model = model
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(&state.default_model)
        .to_string();
    let model_variant = model_variant
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .or_else(|| state.default_model_variant.clone());
    lash::ModelSpec::from_token_limits(
        model,
        model_variant,
        DEFAULT_CONTEXT_WINDOW_TOKENS,
        None,
        None,
    )
    .map_err(AppError::bad_request)
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

    fn internal(message: impl std::fmt::Display) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: message.to_string(),
        }
    }
}

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
        button_trigger_lashlang_resources()
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
        reg.host_events().declare(
            HostEvent::new(
                BUTTON_TRIGGER_RESOURCE,
                BUTTON_TRIGGER_ALIAS,
                BUTTON_TRIGGER_EVENT,
            )
            .payload(button_trigger_event_type()),
        )?;
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

fn button_trigger_lashlang_resources() -> lashlang::ResourceCatalog {
    let mut resources = lashlang::ResourceCatalog::new();
    resources.add_module_instance(["ui", "button"], BUTTON_TRIGGER_RESOURCE);
    resources.add_trigger_event(
        BUTTON_TRIGGER_RESOURCE,
        BUTTON_TRIGGER_EVENT,
        button_trigger_event_type(),
    );
    resources
}

const WORKBENCH_PROMPT: &str = r#"You are running inside the Agent Workbench demo.

Available host features:
- Web access is limited to `web.search(...)` and `web.fetch(...)`, both backed by the same Tavily tools the CLI uses.
- You may call `agents.spawn(...)` for independent investigation.
- You may use Lashlang background processes or subagents for work that should continue independently.
- The red and blue UI buttons emit the host event listed under Host Events.

Use background processes or subagents only when they clarify the user's request or make parallel progress. Keep the visible answer concise and mention any background work you started."#;

#[cfg(test)]
mod tests {
    use super::*;
    use lash::persistence::RuntimePersistence;
    use lashlang::LashlangArtifactStore;

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

    #[tokio::test(flavor = "current_thread")]
    async fn button_host_event_starts_visible_lashlang_process() {
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
        ) as Arc<dyn lash::advanced::ProcessRegistry>;
        let provider = ProviderHandle::new(
            OpenAiCompatibleProvider::new(String::new(), OPENROUTER_BASE_URL).into_components(),
        );
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
            .effect_controller(Arc::new(
                lash::advanced::InlineRuntimeEffectController::default(),
            ))
            .process_registry(Arc::clone(&process_registry))
            .build()
            .expect("build core");
        let session = core
            .session(session_id.clone())
            .rlm()
            .open()
            .await
            .expect("open session");
        let install = session
            .triggers()
            .install_lashlang_source(test_button_trigger_source())
            .await
            .expect("install trigger source");
        assert_eq!(install.installed, vec!["remembered"]);
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

        let report = session
            .host_events()
            .emit(
                BUTTON_TRIGGER_RESOURCE,
                BUTTON_TRIGGER_ALIAS,
                BUTTON_TRIGGER_EVENT,
                json!({
                    "button": "Blue",
                    "message": "user pressed the blue button",
                    "pressed_at": "2026-05-27T00:00:00Z"
                }),
            )
            .await
            .expect("emit host event");

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
        assert_eq!(handles[0].0.descriptor.kind.as_deref(), Some("lashlang"));
        assert_eq!(handles[0].0.descriptor.label.as_deref(), Some("remember"));
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
        assert_eq!(terminal_label(&reopened_handles[0].1), "completed");
        drop(reopened);

        let state = AppState {
            core,
            process_registry: Arc::clone(&process_registry),
            session_ids,
            messages: Arc::new(Mutex::new(Vec::new())),
            default_model: "test-model".to_string(),
            default_model_variant: None,
            web_configured: false,
            trace_sink: None,
            event_tx: broadcast::channel(1024).0,
            queue_runner: SessionQueueRunner::inert(),
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
        assert!(wake.input.contains("user pressed the blue button"));
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

    #[tokio::test(flavor = "current_thread")]
    async fn button_host_event_wake_is_consumed_by_session_queue_runner() {
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
        ) as Arc<dyn lash::advanced::ProcessRegistry>;
        let artifact_store = Arc::new(
            lash_sqlite_store::Store::open(&data_dir.join("artifacts.db"))
                .expect("open artifact store"),
        );
        let artifact_store_for_core: Arc<dyn lashlang::LashlangArtifactStore> =
            artifact_store.clone();
        let provider = lash::testing::TestProvider::builder()
            .kind("workbench-test")
            .complete(|_| async {
                Ok(text_response(
                    "```lashlang\nsubmit \"blue button joke delivered\"\n```",
                ))
            })
            .build()
            .into_handle();
        let model = lash::ModelSpec::from_token_limits("test-model", None, 4096, None, None)
            .expect("model spec");
        let (queue_runner, runner_rx) = SessionQueueRunner::channel();
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
                    config.control.effect_controller =
                        Arc::new(lash::advanced::InlineRuntimeEffectController::default());
                    config
                })
                .process_registry(Arc::clone(&process_registry))
                .build()
                .expect("build core"),
            process_registry,
            session_ids: WorkbenchSessionIds::fresh(),
            messages: Arc::new(Mutex::new(Vec::new())),
            default_model: "test-model".to_string(),
            default_model_variant: None,
            web_configured: false,
            trace_sink: None,
            event_tx,
            queue_runner,
        };
        spawn_session_queue_runner(state.clone(), runner_rx);
        let session = state
            .core
            .session(state.current_session_id())
            .rlm()
            .open()
            .await
            .expect("open session");
        let install = session
            .triggers()
            .install_lashlang_source(test_button_trigger_source())
            .await
            .expect("install trigger source");
        assert_eq!(install.installed, vec!["remembered"]);
        drop(session);

        let mut events = state.event_tx.subscribe();
        let _accepted = button_trigger(
            State(state.clone()),
            Json(ButtonEventRequest {
                button: ButtonChoice::Blue,
            }),
        )
        .await
        .expect("button command");

        let mut saw_wake = false;
        let mut seen_events = Vec::new();
        let delivered = tokio::time::timeout(Duration::from_secs(10), async {
            loop {
                while let Ok(item) = events.try_recv() {
                    if let StreamItem::Event { event } = item {
                        seen_events.push(format!("{:?}", event.event));
                        if matches!(
                            event.event,
                            TurnEvent::QueuedWorkStarted { ref causes, .. }
                                if causes.iter().any(|cause| cause.event_type == "process.wake")
                        ) {
                            saw_wake = true;
                        }
                    }
                }
                let assistant_delivered = state.messages_snapshot().iter().any(|message| {
                    message.role == "assistant"
                        && message.text.contains("blue button joke delivered")
                });
                if assistant_delivered && saw_wake {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(25)).await;
            }
        })
        .await;
        if delivered.is_err() {
            eprintln!("messages: {:?}", state.messages_snapshot());
            if let Ok(session) = state
                .core
                .session(state.current_session_id())
                .rlm()
                .open()
                .await
            {
                eprintln!("queued: {:?}", session.queued_work().await);
            }
        }
        delivered.expect("runner should deliver assistant response and wake event");
        assert!(
            saw_wake,
            "runner should publish the queued wake start event; saw {seen_events:?}"
        );
        let _ = std::fs::remove_dir_all(data_dir);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn reset_chat_deletes_old_session_and_clears_trigger_started_work() {
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
        ) as Arc<dyn lash::advanced::ProcessRegistry>;
        let provider = ProviderHandle::new(
            OpenAiCompatibleProvider::new(String::new(), OPENROUTER_BASE_URL).into_components(),
        );
        let model = lash::ModelSpec::from_token_limits("test-model", None, 4096, None, None)
            .expect("model spec");
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
            .effect_controller(Arc::new(
                lash::advanced::InlineRuntimeEffectController::default(),
            ))
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
            default_model: "test-model".to_string(),
            default_model_variant: None,
            web_configured: false,
            trace_sink: None,
            event_tx: broadcast::channel(1024).0,
            queue_runner: SessionQueueRunner::inert(),
        };
        let old_session_id = state.current_session_id();
        let session = state
            .core
            .session(old_session_id.clone())
            .rlm()
            .open()
            .await
            .expect("open old session");
        session
            .triggers()
            .install_lashlang_source(test_button_trigger_source())
            .await
            .expect("install trigger source");
        let started = session
            .host_events()
            .emit(
                BUTTON_TRIGGER_RESOURCE,
                BUTTON_TRIGGER_ALIAS,
                BUTTON_TRIGGER_EVENT,
                json!({
                    "button": "Red",
                    "message": "user pressed the red button",
                    "pressed_at": "2026-05-27T00:00:00Z"
                }),
            )
            .await
            .expect("emit host event");
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
        assert!(state.messages_snapshot().is_empty());
        assert!(
            process_registry
                .list_handle_grants(&old_process_scope)
                .await
                .expect("old grants after reset")
                .is_empty()
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

    #[tokio::test(flavor = "current_thread")]
    async fn persisted_trigger_route_fires_after_reopening_sqlite_artifact_store() {
        let data_dir =
            std::env::temp_dir().join(format!("agent-workbench-trigger-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&data_dir).expect("create temp workbench dir");
        let session_store_factory = Arc::new(lash_sqlite_store::SqliteSessionStoreFactory::new(
            data_dir.join("lash-sessions"),
        ));
        let process_registry_path = data_dir.join("processes.db");
        let artifact_store_path = data_dir.join("artifacts.db");
        let session_id = WorkbenchSessionIds::fresh().current();

        let linked = lashlang::LinkedModule::link(
            lashlang::parse(test_button_trigger_source()).expect("parse trigger source"),
            lashlang::LashlangSurface::new(
                button_trigger_lashlang_resources(),
                workbench_lashlang_abilities(),
            ),
        )
        .expect("link trigger source");

        {
            let artifact_store = Arc::new(
                lash_sqlite_store::Store::open(&artifact_store_path).expect("open artifacts"),
            );
            let artifact_store_for_core: Arc<dyn lashlang::LashlangArtifactStore> =
                artifact_store.clone();
            let process_registry = Arc::new(
                lash_sqlite_store::SqliteProcessRegistry::open(&process_registry_path)
                    .expect("open registry"),
            ) as Arc<dyn lash::advanced::ProcessRegistry>;
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
            let install = session
                .triggers()
                .install_lashlang_source(test_button_trigger_source())
                .await
                .expect("install trigger source");
            assert_eq!(install.installed, vec!["remembered"]);
            assert!(
                artifact_store
                    .get_module_artifact(&linked.module_ref)
                    .expect("load stored module artifact")
                    .is_some()
            );
            drop(session);
            drop(core);
        }

        let artifact_store = Arc::new(
            lash_sqlite_store::Store::open(&artifact_store_path).expect("reopen artifacts"),
        );
        assert!(
            artifact_store
                .get_module_artifact(&linked.module_ref)
                .expect("load reopened module artifact")
                .is_some()
        );
        let artifact_store_for_core: Arc<dyn lashlang::LashlangArtifactStore> =
            artifact_store.clone();
        let process_registry = Arc::new(
            lash_sqlite_store::SqliteProcessRegistry::open(&process_registry_path)
                .expect("reopen registry"),
        ) as Arc<dyn lash::advanced::ProcessRegistry>;
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
        let report = reopened
            .host_events()
            .emit(
                BUTTON_TRIGGER_RESOURCE,
                BUTTON_TRIGGER_ALIAS,
                BUTTON_TRIGGER_EVENT,
                json!({
                    "button": "Red",
                    "message": "user pressed the red button",
                    "pressed_at": "2026-05-27T00:00:00Z"
                }),
            )
            .await
            .expect("emit reopened host event");
        assert_eq!(report.started_process_ids.len(), 1);
        process_registry
            .await_process(&report.started_process_ids[0])
            .await
            .expect("trigger process should finish");

        let _ = std::fs::remove_dir_all(data_dir);
    }

    fn test_workbench_core(
        session_store_factory: Arc<dyn lash::persistence::SessionStoreFactory>,
        process_registry: Arc<dyn lash::advanced::ProcessRegistry>,
        artifact_store: Arc<dyn lashlang::LashlangArtifactStore>,
    ) -> LashCore {
        let provider = ProviderHandle::new(
            OpenAiCompatibleProvider::new(String::new(), OPENROUTER_BASE_URL).into_components(),
        );
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
                config.control.effect_controller =
                    Arc::new(lash::advanced::InlineRuntimeEffectController::default());
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

    fn test_button_trigger_source() -> &'static str {
        r#"
        type ButtonChoice = enum["Red", "Blue"]
        type ButtonPressed = { button: ButtonChoice, message: str, pressed_at: str }

        process remember(event: ButtonPressed) {
          checked = validate(event, Type {
            button: enum["Red", "Blue"],
            message: str,
            pressed_at: str
          })
          wake { button: checked.button, message: checked.message }
          finish { button: checked.button, ok: true }
        }

        trigger remembered on ui.button.pressed as event
          -> remember(event: event)
        "#
    }
}
