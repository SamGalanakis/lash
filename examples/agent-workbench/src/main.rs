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
    TurnInput, TurnOutput,
};
use lash_provider_openai::{OPENROUTER_BASE_URL, OpenAiCompatibleProvider};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tokio::time::timeout;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::ReceiverStream;

const SESSION_ID_PREFIX: &str = "workbench";
const DEFAULT_CONTEXT_WINDOW_TOKENS: usize = 200_000;
const BUTTON_TRIGGER_RESOURCE: &str = "TRIGGER";
const BUTTON_TRIGGER_ALIAS: &str = "button";
const BUTTON_TRIGGER_EVENT: &str = "pressed";
const PROCESS_FOLLOWUP_WAIT: Duration = Duration::from_secs(12);

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
    let subagent_registry = Arc::new(lash_subagents::default_registry(&BTreeMap::new()));
    let session_ids = WorkbenchSessionIds::fresh();

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
    };

    let app = Router::new()
        .route("/", get(index))
        .route("/api/state", get(app_state))
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

async fn send_turn(
    State(state): State<AppState>,
    Json(request): Json<TurnRequest>,
) -> Result<Response, AppError> {
    let text = request.text.trim().to_string();
    if text.is_empty() {
        return Err(AppError::bad_request("message text is required"));
    }
    let turn_model = model_spec_for_request(
        &state,
        request.model.as_deref(),
        request.model_variant.as_deref(),
    )?;
    Ok(stream_turn(state, "user", text.clone(), text, turn_model).await)
}

async fn button_trigger(
    State(state): State<AppState>,
    Json(request): Json<ButtonEventRequest>,
) -> Result<Response, AppError> {
    Ok(stream_button_trigger(state, request.button).await)
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
    Ok(Json(work))
}

async fn stream_turn(
    state: AppState,
    role: &'static str,
    display_text: String,
    turn_text: String,
    turn_model: lash::ModelSpec,
) -> Response {
    let user_message = state.push_message(role, display_text);
    let (tx, rx) = mpsc::channel::<StreamItem>(64);
    tokio::spawn(async move {
        let _ = tx
            .send(StreamItem::Message {
                message: user_message,
            })
            .await;

        let session_id = state.current_session_id();
        let session = match state.core.session(session_id).rlm().open().await {
            Ok(session) => session,
            Err(err) => {
                let _ = tx
                    .send(StreamItem::Error {
                        message: err.to_string(),
                    })
                    .await;
                let _ = tx.send(StreamItem::Done).await;
                return;
            }
        };
        if let Err(err) =
            run_agent_turn(&state, &tx, session, TurnInput::text(turn_text), turn_model).await
        {
            let _ = tx
                .send(StreamItem::Error {
                    message: err.to_string(),
                })
                .await;
        }
        let _ = tx.send(StreamItem::Done).await;
    });

    ndjson_response(rx)
}

async fn stream_button_trigger(state: AppState, button: ButtonChoice) -> Response {
    let (tx, rx) = mpsc::channel::<StreamItem>(64);
    tokio::spawn(async move {
        let pressed_at = Utc::now().to_rfc3339();
        let trigger_message =
            state.push_message("event", format!("{} button event", button.lower()));
        let _ = tx
            .send(StreamItem::Message {
                message: trigger_message,
            })
            .await;

        match emit_button_host_event(&state, button, &pressed_at).await {
            Ok(report) => {
                if report.started_process_ids.is_empty() {
                    let message =
                        state.push_message("event", "no trigger handled the button event");
                    let _ = tx.send(StreamItem::Message { message }).await;
                } else {
                    let message = state.push_message(
                        "event",
                        started_process_text(report.started_process_ids.len()),
                    );
                    let _ = tx.send(StreamItem::Message { message }).await;

                    match wait_for_started_processes(
                        &state,
                        &report.started_process_ids,
                        PROCESS_FOLLOWUP_WAIT,
                    )
                    .await
                    {
                        Ok(true) => {
                            run_button_followup_turn(&state, &tx).await;
                        }
                        Ok(false) => {
                            let message = state.push_message(
                                "event",
                                "process still running; the next turn will pick up its wake",
                            );
                            let _ = tx.send(StreamItem::Message { message }).await;
                        }
                        Err(err) => {
                            let _ = tx
                                .send(StreamItem::Error {
                                    message: err.to_string(),
                                })
                                .await;
                        }
                    }
                }
            }
            Err(err) => {
                let _ = tx
                    .send(StreamItem::Error {
                        message: err.to_string(),
                    })
                    .await;
            }
        }
        let _ = tx.send(StreamItem::Done).await;
    });

    ndjson_response(rx)
}

async fn run_agent_turn(
    state: &AppState,
    tx: &mpsc::Sender<StreamItem>,
    session: lash::LashSession,
    input: TurnInput,
    turn_model: lash::ModelSpec,
) -> Result<(), lash::EmbedError> {
    let turn_state = Arc::new(Mutex::new(TurnStreamState::default()));
    let ui_events = ChannelTurnEvents {
        tx: tx.clone(),
        turn_state: Arc::clone(&turn_state),
    };
    let turn = session.turn(input).model(turn_model).require_submit()?;
    let output = turn.collect_with(&ui_events).await?;
    let streamed_prose = turn_state
        .lock()
        .expect("turn state lock")
        .assistant_prose
        .clone();
    let assistant_text = assistant_text_for_display(&output, &streamed_prose);
    let message = state.push_message("assistant", assistant_text);
    let _ = tx.send(StreamItem::Message { message }).await;
    Ok(())
}

fn started_process_text(count: usize) -> String {
    match count {
        0 => "no trigger handled the button event".to_string(),
        1 => "started 1 background process".to_string(),
        count => format!("started {count} background processes"),
    }
}

async fn wait_for_started_processes(
    state: &AppState,
    process_ids: &[String],
    wait: Duration,
) -> AnyhowResult<bool> {
    let result = timeout(wait, async {
        for process_id in process_ids {
            state
                .process_registry
                .await_process(process_id)
                .await
                .with_context(|| format!("await process {process_id}"))?;
        }
        Ok::<(), anyhow::Error>(())
    })
    .await;
    match result {
        Ok(Ok(())) => Ok(true),
        Ok(Err(err)) => Err(err),
        Err(_) => Ok(false),
    }
}

async fn run_button_followup_turn(state: &AppState, tx: &mpsc::Sender<StreamItem>) {
    match state
        .core
        .session(state.current_session_id())
        .rlm()
        .open()
        .await
    {
        Ok(session) => match model_spec_for_request(state, None, None) {
            Ok(model) => {
                if let Err(err) =
                    run_agent_turn(state, tx, session, TurnInput::empty(), model).await
                {
                    let _ = tx
                        .send(StreamItem::Error {
                            message: err.to_string(),
                        })
                        .await;
                }
            }
            Err(err) => {
                let _ = tx
                    .send(StreamItem::Error {
                        message: err.message,
                    })
                    .await;
            }
        },
        Err(err) => {
            let _ = tx
                .send(StreamItem::Error {
                    message: err.to_string(),
                })
                .await;
        }
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
    tx: mpsc::Sender<StreamItem>,
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
        let _ = self.tx.send(StreamItem::event(activity)).await;
    }
}

async fn emit_button_host_event(
    state: &AppState,
    button: ButtonChoice,
    pressed_at: &str,
) -> AnyhowResult<lash::HostEventEmitReport> {
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
            json!({
                "pressed_at": pressed_at,
                "button": button.as_str(),
                "message": format!("user pressed the {} button", button.lower()),
            }),
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
        .with_process_lifecycle()
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
        message
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

fn assistant_text_for_display(output: &TurnOutput, streamed_prose: &str) -> String {
    if let Some(value) = output.submitted_value() {
        return terminal_value_text(value);
    }
    if let Some((_tool_name, value)) = output.tool_value() {
        return terminal_value_text(value);
    }
    output
        .assistant_message()
        .filter(|text| !text.trim().is_empty())
        .unwrap_or(streamed_prose)
        .to_string()
}

fn terminal_value_text(value: &Value) -> String {
    value
        .as_str()
        .map(str::to_string)
        .unwrap_or_else(|| value.to_string())
}

fn terminal_label(record: &lash::advanced::ProcessRecord) -> String {
    record
        .terminal
        .as_ref()
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
            .provider(Arc::new(lash_tool_web::WebSearch::new(
                self.tavily_api_key.clone(),
            )))?;
        reg.tools().provider(Arc::new(lash_tool_web::FetchUrl::new(
            self.tavily_api_key.clone(),
        )))?;
        Ok(())
    }
}

fn button_trigger_lashlang_resources() -> lashlang::ResourceCatalog {
    let mut resources = lashlang::ResourceCatalog::new();
    resources.add_alias(BUTTON_TRIGGER_RESOURCE, BUTTON_TRIGGER_ALIAS);
    resources.add_trigger_event(
        BUTTON_TRIGGER_RESOURCE,
        BUTTON_TRIGGER_EVENT,
        button_trigger_event_type(),
    );
    resources
}

const WORKBENCH_PROMPT: &str = r#"You are running inside the Agent Workbench demo.

Available host features:
- Web access is limited to `search_web` and `fetch_url`, both backed by the same Tavily tools the CLI uses.
- You may spawn subagents for independent investigation.
- You may use Lashlang background processes or subagents for work that should continue independently.
- The red and blue UI buttons emit the host event listed under Host Events.

Use background processes or subagents only when they clarify the user's request or make parallel progress. Keep the visible answer concise and mention any background work you started."#;

#[cfg(test)]
mod tests {
    use super::*;

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

    #[tokio::test(flavor = "current_thread")]
    async fn button_host_event_starts_visible_lashlang_process() {
        let db_path = std::env::temp_dir().join(format!(
            "agent-workbench-processes-{}.db",
            uuid::Uuid::new_v4()
        ));
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
            .list()
            .await
            .expect("list handles");
        assert_eq!(handles.len(), 1);
        assert_eq!(handles[0].0.descriptor.kind.as_deref(), Some("lashlang"));
        assert_eq!(handles[0].0.descriptor.label.as_deref(), Some("remember"));
        drop(session);

        let reopened = core
            .session(session_id.clone())
            .rlm()
            .open()
            .await
            .expect("reopen session");
        let reopened_handles = reopened
            .process_control()
            .list()
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
        };
        let target_scope_id = lash::advanced::ProcessScope::new(state.current_session_id()).id();
        let wakes = process_registry
            .drain_wake_inputs(&target_scope_id, 10)
            .await
            .expect("list pending wake inputs");
        assert_eq!(wakes.len(), 1);
        assert!(wakes[0].input.contains("user pressed the blue button"));
        assert_eq!(wakes[0].target_scope_id, target_scope_id);
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
        let _ = std::fs::remove_file(db_path);
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

        trigger remembered on TRIGGER.button.pressed as event
          -> remember(event: event)
        "#
    }
}
