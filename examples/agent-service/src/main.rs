use std::collections::HashMap;
use std::convert::Infallible;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use axum::body::Body;
use axum::extract::{Path as AxumPath, State};
use axum::http::{StatusCode, header};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use bytes::Bytes;
use lash::{
    LashCore, LashSession, ModeId, ModePreset, PluginBinding, TurnActivity, TurnActivitySink,
    TurnBuilder, TurnEvent, TurnInput, TurnOutput,
    advanced::{LocalBackgroundTaskRegistry, LocalRuntimeEffectHost},
    plugins::{PluginError, PluginFactory, PluginRegistrar, PluginSessionContext, SessionPlugin},
    prompt::PromptContribution,
    provider::{ProviderHandle, ProviderOptions, ProviderThinkingPolicy},
    tools::{ToolCall, ToolContract, ToolDefinition, ToolManifest, ToolProvider, ToolResult},
    tracing::{JsonlTraceSink, TraceLevel, TraceRecord, TraceSink, TraceSinkError},
};
use lash_provider_openai::{OPENROUTER_BASE_URL, OpenAiCompatibleProvider};
use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::sync::mpsc;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::ReceiverStream;

type AppResult<T> = Result<T, AppError>;

#[derive(Clone)]
struct AppStateData {
    core: LashCore,
    db: Arc<Mutex<AppDb>>,
    default_model: String,
    default_model_variant: Option<String>,
}

impl AppStateData {
    async fn open_session(&self, chat_id: &str) -> AppResult<LashSession> {
        Ok(self
            .core
            .session(chat_id)
            .rlm()
            .plugin::<DemoPlugin>(DemoPluginConfig)
            .open()
            .await?)
    }

    async fn with_db<T, F>(&self, f: F) -> AppResult<T>
    where
        T: Send + 'static,
        F: FnOnce(&mut AppDb) -> AppResult<T> + Send + 'static,
    {
        let db = Arc::clone(&self.db);
        tokio::task::spawn_blocking(move || {
            let mut db = db
                .lock()
                .map_err(|_| AppError::internal("database lock poisoned"))?;
            f(&mut db)
        })
        .await
        .map_err(|err| AppError::internal(format!("database task failed: {err}")))?
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

    fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: message.into(),
        }
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        (self.status, Json(json!({ "error": self.message }))).into_response()
    }
}

impl From<rusqlite::Error> for AppError {
    fn from(err: rusqlite::Error) -> Self {
        Self::internal(err.to_string())
    }
}

impl From<lash::EmbedError> for AppError {
    fn from(err: lash::EmbedError) -> Self {
        Self::internal(err.to_string())
    }
}

#[derive(Clone, Debug, Serialize)]
struct ChatSummary {
    id: String,
    title: String,
    created_at: String,
    updated_at: String,
    model: String,
    model_variant: Option<String>,
    model_label: String,
}

#[derive(Clone, Debug, Serialize)]
struct ChatMessage {
    id: i64,
    chat_id: String,
    kind: String,
    role: String,
    text: String,
    payload: Option<serde_json::Value>,
    created_at: String,
}

#[derive(Debug, Deserialize)]
struct CreateChatRequest {
    title: Option<String>,
    model: Option<String>,
    model_variant: Option<String>,
}

#[derive(Debug, Deserialize)]
struct UpdateChatModelRequest {
    model: String,
    model_variant: Option<String>,
}

#[derive(Debug, Deserialize)]
struct SendMessageRequest {
    text: String,
    board: BoardState,
    model: Option<String>,
    model_variant: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
struct AppSettings {
    default_model: String,
    default_model_variant: Option<String>,
    model_variants: Vec<&'static str>,
}

#[derive(Clone, Debug)]
struct ChatModelSelection {
    model: String,
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

#[derive(Default)]
struct StderrTraceSink {
    lock: Mutex<()>,
}

impl TraceSink for StderrTraceSink {
    fn append(&self, record: &TraceRecord) -> Result<(), TraceSinkError> {
        let line = serde_json::to_string(record)?;
        let _guard = self.lock.lock().map_err(|_| TraceSinkError::LockPoisoned)?;
        eprintln!("{line}");
        Ok(())
    }
}

struct FanoutTraceSink {
    sinks: Vec<Arc<dyn TraceSink>>,
}

impl TraceSink for FanoutTraceSink {
    fn append(&self, record: &TraceRecord) -> Result<(), TraceSinkError> {
        for sink in &self.sinks {
            sink.append(record)?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AgentServiceDurability {
    Local,
    Restate,
}

impl AgentServiceDurability {
    fn configured() -> anyhow_like::Result<Self> {
        let mut args = std::env::args().skip(1);
        let mut from_args = None;
        while let Some(arg) = args.next() {
            if let Some(value) = arg.strip_prefix("--durability=") {
                from_args = Some(value.to_string());
                continue;
            }
            if arg == "--durability" {
                let value = args
                    .next()
                    .ok_or_else(|| "--durability requires local or restate".to_string())?;
                from_args = Some(value);
                continue;
            }
            return Err(format!("unknown argument `{arg}`"));
        }

        let raw = from_args
            .or_else(|| std::env::var("AGENT_SERVICE_DURABILITY").ok())
            .unwrap_or_else(|| "local".to_string());
        Self::parse(&raw)
    }

    fn parse(value: &str) -> anyhow_like::Result<Self> {
        match value {
            "local" => Ok(Self::Local),
            "restate" => Ok(Self::Restate),
            other => Err(format!(
                "invalid durability `{other}`; expected `local` or `restate`"
            )),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Local => "local",
            Self::Restate => "restate",
        }
    }
}

#[tokio::main]
async fn main() -> anyhow_like::Result<()> {
    let durability = AgentServiceDurability::configured()?;
    let api_key = std::env::var("OPENROUTER_API_KEY")
        .map_err(|_| "OPENROUTER_API_KEY is required".to_string())?;
    let model = std::env::var("OPENROUTER_MODEL").unwrap_or_else(|_| "openai/gpt-5.5".to_string());
    let model_variant =
        std::env::var("OPENROUTER_MODEL_VARIANT").unwrap_or_else(|_| "medium".to_string());
    let addr: SocketAddr = std::env::var("AGENT_SERVICE_ADDR")
        .unwrap_or_else(|_| "127.0.0.1:3000".to_string())
        .parse()
        .map_err(|err| format!("invalid AGENT_SERVICE_ADDR: {err}"))?;
    let data_dir = std::env::var("AGENT_SERVICE_DATA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(".agent-service"));
    std::fs::create_dir_all(&data_dir).map_err(|err| err.to_string())?;
    let trace_path = std::env::var("AGENT_SERVICE_TRACE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| data_dir.join("trace.jsonl"));
    eprintln!("agent-service trace: {}", trace_path.display());

    let provider = ProviderHandle::new(
        OpenAiCompatibleProvider::new(api_key, OPENROUTER_BASE_URL)
            .with_options(ProviderOptions {
                thinking: ProviderThinkingPolicy { expose: true },
                ..ProviderOptions::default()
            })
            .into_components(),
    );
    let store_factory = Arc::new(lash_sqlite_store::SqliteSessionStoreFactory::new(
        data_dir.join("lash-sessions"),
    ));
    let core_builder = LashCore::builder()
        .install_mode(ModePreset::rlm())
        .default_mode(ModeId::rlm())
        .provider(provider)
        .model(model.clone(), Some(model_variant.clone()))
        .max_context_tokens(200_000)
        .store_factory(store_factory)
        .trace_sink(Some(Arc::new(FanoutTraceSink {
            sinks: vec![
                Arc::new(StderrTraceSink::default()),
                Arc::new(JsonlTraceSink::new(trace_path)),
            ],
        })))
        .trace_level(TraceLevel::Extended);
    let core = match durability {
        AgentServiceDurability::Local => core_builder
            .advanced()
            .effect_host(Arc::new(LocalRuntimeEffectHost::default()))
            .background_task_registry(Arc::new(LocalBackgroundTaskRegistry::default()))
            .build()
            .map_err(|err| err.to_string())?,
        AgentServiceDurability::Restate => {
            return Err(
                "AGENT_SERVICE_DURABILITY=restate requires a Restate RuntimeEffectHost adapter"
                    .to_string(),
            );
        }
    };

    let app_db = AppDb::open(&data_dir.join("app.db")).map_err(|err| err.to_string())?;
    let state = AppStateData {
        core,
        db: Arc::new(Mutex::new(app_db)),
        default_model: model,
        default_model_variant: Some(model_variant),
    };

    let app = Router::new()
        .route("/", get(index))
        .route("/api/settings", get(settings))
        .route("/api/chats", get(list_chats).post(create_chat))
        .route(
            "/api/chats/{chat_id}/model",
            axum::routing::post(update_chat_model),
        )
        .route(
            "/api/chats/{chat_id}/messages",
            get(list_messages).post(send_message),
        )
        .with_state(state);

    println!(
        "agent-service listening on http://{addr} (durability: {})",
        durability.as_str()
    );
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .map_err(|err| err.to_string())?;
    axum::serve(listener, app)
        .await
        .map_err(|err| err.to_string())?;
    Ok(())
}

async fn index() -> Html<&'static str> {
    Html(INDEX_HTML)
}

async fn list_chats(State(state): State<AppStateData>) -> AppResult<Json<Vec<ChatSummary>>> {
    state.with_db(|db| db.list_chats()).await.map(Json)
}

async fn settings(State(state): State<AppStateData>) -> Json<AppSettings> {
    Json(AppSettings {
        default_model: state.default_model,
        default_model_variant: state.default_model_variant,
        model_variants: vec!["low", "medium", "high"],
    })
}

async fn create_chat(
    State(state): State<AppStateData>,
    Json(request): Json<CreateChatRequest>,
) -> AppResult<Json<ChatSummary>> {
    let title = request
        .title
        .as_deref()
        .map(str::trim)
        .filter(|title| !title.is_empty())
        .unwrap_or("New chat");
    let title = title.to_string();
    let selection = normalize_model_selection(
        request.model.as_deref(),
        request.model_variant.as_deref(),
        &state.default_model,
        state.default_model_variant.as_deref(),
    )?;
    state
        .with_db(move |db| {
            db.create_chat(&title, &selection.model, selection.model_variant.as_deref())
        })
        .await
        .map(Json)
}

async fn update_chat_model(
    State(state): State<AppStateData>,
    AxumPath(chat_id): AxumPath<String>,
    Json(request): Json<UpdateChatModelRequest>,
) -> AppResult<Json<ChatSummary>> {
    let selection =
        normalize_optional_model_selection(Some(&request.model), request.model_variant.as_deref())?
            .ok_or_else(|| AppError::bad_request("model is required"))?;
    state
        .with_db(move |db| {
            db.update_chat_model(
                &chat_id,
                &selection.model,
                selection.model_variant.as_deref(),
            )
        })
        .await
        .map(Json)
}

async fn list_messages(
    State(state): State<AppStateData>,
    AxumPath(chat_id): AxumPath<String>,
) -> AppResult<Json<Vec<ChatMessage>>> {
    state
        .with_db(move |db| {
            db.require_chat(&chat_id)?;
            db.list_messages(&chat_id)
        })
        .await
        .map(Json)
}

async fn send_message(
    State(state): State<AppStateData>,
    AxumPath(chat_id): AxumPath<String>,
    Json(request): Json<SendMessageRequest>,
) -> AppResult<Response> {
    let text = request.text.trim().to_string();
    if text.is_empty() {
        return Err(AppError::bad_request("message text is required"));
    }
    let request_model = normalize_optional_model_selection(
        request.model.as_deref(),
        request.model_variant.as_deref(),
    )?;

    // Persist the submitted board with the user message so reload can replay
    // human-only terminal states, including a final X move that ends in draw.
    let user_board = request.board.clone();
    let (user_message, model_selection) = state
        .with_db({
            let chat_id = chat_id.clone();
            let text = text.clone();
            move |db| {
                db.require_chat(&chat_id)?;
                if let Some(selection) = request_model {
                    db.update_chat_model(
                        &chat_id,
                        &selection.model,
                        selection.model_variant.as_deref(),
                    )?;
                }
                let model_selection = db.chat_model_selection(&chat_id)?;
                db.maybe_title_from_first_message(&chat_id, &text)?;
                let message = db.insert_message_with_payload(
                    &chat_id,
                    "user",
                    &text,
                    Some(json!({ "board": user_board })),
                )?;
                Ok((message, model_selection))
            }
        })
        .await?;

    let session = state.open_session(&chat_id).await?;
    let (tx, rx) = mpsc::channel::<StreamItem>(64);
    let run_state = state.clone();
    tokio::spawn(async move {
        let _ = tx
            .send(StreamItem::Message {
                message: user_message,
            })
            .await;
        let turn_state = Arc::new(Mutex::new(TurnPersistenceState::default()));
        let ui_events = ChannelTurnEvents {
            tx: tx.clone(),
            state: run_state.clone(),
            chat_id: chat_id.clone(),
            turn_state: Arc::clone(&turn_state),
        };
        let turn = session
            .turn(TurnInput::text(text))
            .model(model_selection.model, model_selection.model_variant)
            .with_board(request.board)
            .require_submit();
        let turn = match turn {
            Ok(turn) => turn.collect_with(&ui_events).await,
            Err(err) => Err(err),
        };
        match turn {
            Ok(output) => {
                let assistant_text = assistant_text_for_persistence(
                    &output,
                    &turn_state.lock().expect("turn state lock").assistant_prose,
                );
                let inserted = run_state
                    .with_db({
                        let chat_id = chat_id.clone();
                        move |db| db.insert_message(&chat_id, "assistant", &assistant_text)
                    })
                    .await;
                match inserted {
                    Ok(message) => {
                        let _ = tx.send(StreamItem::Message { message }).await;
                    }
                    Err(err) => {
                        let _ = tx
                            .send(StreamItem::Error {
                                message: err.message,
                            })
                            .await;
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

    Ok(Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/x-ndjson; charset=utf-8")
        .header(header::CACHE_CONTROL, "no-store")
        .body(Body::from_stream(stream))
        .expect("valid streaming response"))
}

struct ChannelTurnEvents {
    tx: mpsc::Sender<StreamItem>,
    state: AppStateData,
    chat_id: String,
    turn_state: Arc<Mutex<TurnPersistenceState>>,
}

#[derive(Default)]
struct TurnPersistenceState {
    reasoning: Option<(i64, String)>,
    assistant_prose: String,
    code: Option<String>,
    code_message: Option<i64>,
    tools: HashMap<String, i64>,
}

impl ChannelTurnEvents {
    async fn handle(&self, activity: TurnActivity) {
        let event = &activity.event;
        if let TurnEvent::AssistantProseDelta { text } = &event {
            self.turn_state
                .lock()
                .expect("turn state lock")
                .assistant_prose
                .push_str(text);
            let _ = self.tx.send(StreamItem::event(activity)).await;
            return;
        }
        // Keep persisted message order tied to event start order. The browser
        // only renders completed code/tool rows, but reload should still
        // reconstruct "thinking -> lashlang -> tools -> assistant".
        if let TurnEvent::ReasoningDelta { text } = &event {
            let update = {
                let mut state = self.turn_state.lock().expect("turn state lock");
                match state.reasoning.as_mut() {
                    Some((id, existing)) => {
                        existing.push_str(text);
                        Some((*id, existing.clone(), false))
                    }
                    None => Some((0, text.clone(), true)),
                }
            };
            if let Some((id, reasoning, insert)) = update {
                let result = if insert {
                    self.state
                        .with_db({
                            let chat_id = self.chat_id.clone();
                            let reasoning = reasoning.clone();
                            move |db| db.insert_reasoning(&chat_id, &reasoning)
                        })
                        .await
                        .map(|message| {
                            self.turn_state.lock().expect("turn state lock").reasoning =
                                Some((message.id, reasoning));
                        })
                } else {
                    self.state
                        .with_db({
                            let reasoning = reasoning.clone();
                            move |db| db.update_reasoning(id, &reasoning)
                        })
                        .await
                        .map(|_| ())
                };
                if let Err(err) = result {
                    let _ = self
                        .tx
                        .send(StreamItem::Error {
                            message: err.message,
                        })
                        .await;
                }
            }
            let _ = self.tx.send(StreamItem::event(activity)).await;
            return;
        }
        if let TurnEvent::CodeBlockStarted { code, .. } = &event {
            self.turn_state.lock().expect("turn state lock").code = Some(code.clone());
            match self
                .state
                .with_db({
                    let chat_id = self.chat_id.clone();
                    let event = event.clone();
                    let code = code.clone();
                    move |db| db.insert_code_block(&chat_id, event, Some(code))
                })
                .await
            {
                Ok(message) => {
                    self.turn_state
                        .lock()
                        .expect("turn state lock")
                        .code_message = Some(message.id);
                }
                Err(err) => {
                    let _ = self
                        .tx
                        .send(StreamItem::Error {
                            message: err.message,
                        })
                        .await;
                }
            }
            let _ = self.tx.send(StreamItem::event(activity)).await;
            return;
        }
        if matches!(&event, TurnEvent::ToolCallStarted { .. }) {
            match self
                .state
                .with_db({
                    let chat_id = self.chat_id.clone();
                    let event = event.clone();
                    move |db| db.insert_tool_call(&chat_id, event)
                })
                .await
            {
                Ok(message) => {
                    self.turn_state
                        .lock()
                        .expect("turn state lock")
                        .tools
                        .insert(activity.correlation_id.0.clone(), message.id);
                }
                Err(err) => {
                    let _ = self
                        .tx
                        .send(StreamItem::Error {
                            message: err.message,
                        })
                        .await;
                }
            }
            let _ = self.tx.send(StreamItem::event(activity)).await;
            return;
        }
        if matches!(&event, TurnEvent::ToolCallCompleted { .. }) {
            let existing = self
                .turn_state
                .lock()
                .expect("turn state lock")
                .tools
                .remove(&activity.correlation_id.0);
            let result = self
                .state
                .with_db({
                    let chat_id = self.chat_id.clone();
                    let event = event.clone();
                    move |db| {
                        if let Some(id) = existing {
                            db.update_tool_call(id, event)
                        } else {
                            db.insert_tool_call(&chat_id, event)
                        }
                    }
                })
                .await;
            if let Err(err) = result {
                let _ = self
                    .tx
                    .send(StreamItem::Error {
                        message: err.message,
                    })
                    .await;
            }
            let _ = self.tx.send(StreamItem::event(activity)).await;
            return;
        }
        if matches!(&event, TurnEvent::CodeBlockCompleted { .. }) {
            let (code, existing) = {
                let mut state = self.turn_state.lock().expect("turn state lock");
                (state.code.take(), state.code_message.take())
            };
            let result = self
                .state
                .with_db({
                    let chat_id = self.chat_id.clone();
                    let event = event.clone();
                    move |db| {
                        if let Some(id) = existing {
                            db.update_code_block(id, event, code)
                        } else {
                            db.insert_code_block(&chat_id, event, code)
                        }
                    }
                })
                .await;
            if let Err(err) = result {
                let _ = self
                    .tx
                    .send(StreamItem::Error {
                        message: err.message,
                    })
                    .await;
            }
            let _ = self.tx.send(StreamItem::event(activity)).await;
            return;
        }
        if matches!(
            &event,
            TurnEvent::SubmittedValue { .. } | TurnEvent::ToolValue { .. }
        ) {
            let _ = self.tx.send(StreamItem::event(activity)).await;
            return;
        }
        let _ = self.tx.send(StreamItem::event(activity)).await;
    }
}

#[async_trait]
impl TurnActivitySink for ChannelTurnEvents {
    async fn emit(&self, activity: TurnActivity) {
        self.handle(activity).await;
    }
}

#[derive(Clone, Debug)]
struct DemoPlugin;

#[derive(Clone, Debug)]
struct DemoPluginConfig;

#[derive(Clone, Debug, Serialize, Deserialize)]
struct BoardState {
    cells: Vec<Option<String>>,
    turn: String,
}

#[derive(Clone, Debug)]
struct DemoTurnInput {
    board: BoardState,
}

trait BoardTurnExt {
    fn with_board(self, board: BoardState) -> Self;
}

impl BoardTurnExt for TurnBuilder {
    fn with_board(self, board: BoardState) -> Self {
        self.with_plugin_input::<DemoPlugin>(DemoTurnInput { board })
    }
}

impl PluginBinding for DemoPlugin {
    const ID: &'static str = "demo_tic_tac_toe";
    type SessionConfig = DemoPluginConfig;
    type Input = DemoTurnInput;

    fn factory(_config: &Self::SessionConfig) -> Arc<dyn PluginFactory> {
        Arc::new(DemoPluginFactory)
    }

    fn requires_turn_input(_config: &Self::SessionConfig) -> bool {
        true
    }
}

struct DemoPluginFactory;

impl PluginFactory for DemoPluginFactory {
    fn id(&self) -> &'static str {
        DemoPlugin::ID
    }

    fn build(&self, _ctx: &PluginSessionContext) -> Result<Arc<dyn SessionPlugin>, PluginError> {
        Ok(Arc::new(DemoSessionPlugin))
    }
}

struct DemoSessionPlugin;

impl SessionPlugin for DemoSessionPlugin {
    fn id(&self) -> &'static str {
        DemoPlugin::ID
    }

    fn register(&self, reg: &mut PluginRegistrar) -> Result<(), PluginError> {
        reg.prompt().contribute(Arc::new(|ctx| {
            Box::pin(async move {
                let Some(input) = ctx
                    .turn_context
                    .plugin_input::<DemoTurnInput>(DemoPlugin::ID)
                else {
                    return Ok(Vec::new());
                };
                let context = board_prompt(&input.board);
                Ok(vec![PromptContribution::environment(
                    "Tic Tac Toe Board",
                    context,
                )])
            })
        }));
        reg.tools().provider(Arc::new(DemoTools))?;
        Ok(())
    }
}

struct DemoTools;

#[async_trait]
impl ToolProvider for DemoTools {
    fn tool_manifests(&self) -> Vec<ToolManifest> {
        demo_tool_definitions()
            .into_iter()
            .map(|tool| tool.manifest())
            .collect()
    }

    fn resolve_contract(&self, name: &str) -> Option<Arc<ToolContract>> {
        demo_tool_definitions()
            .into_iter()
            .find(|tool| tool.name == name)
            .map(|tool| Arc::new(tool.contract()))
    }

    async fn execute(&self, call: ToolCall<'_>) -> ToolResult {
        let Some(input) = call.context.plugin_input::<DemoTurnInput>(DemoPlugin::ID) else {
            return ToolResult::err_fmt("missing board turn input");
        };

        match call.name {
            "read_board" => ToolResult::ok(board_snapshot(&input.board)),
            "play_move" => {
                let Some(cell) = call.args.get("cell").and_then(|value| value.as_u64()) else {
                    return ToolResult::err_fmt("missing integer cell");
                };
                ToolResult::ok(apply_agent_move(&input.board, cell as usize))
            }
            other => ToolResult::err_fmt(format!("unknown demo tool `{other}`")),
        }
    }
}

fn demo_tool_definitions() -> Vec<ToolDefinition> {
    vec![read_board_tool(), play_move_tool()]
}

fn read_board_tool() -> ToolDefinition {
    ToolDefinition::raw(
        "read_board",
        "Read the app-owned Tic Tac Toe board. Returns the 0..8 index map, current marks by index, legal moves, winner, and whose turn it is.",
        json!({
            "type": "object",
            "properties": {},
            "additionalProperties": false
        }),
        json!({ "type": "object" }),
    )
}

fn play_move_tool() -> ToolDefinition {
    ToolDefinition::raw(
        "play_move",
        "Play one O move for the agent when it is O's turn. The move is a zero-based cell index: 0 top-left, 1 top-middle, 2 top-right, 3 middle-left, 4 center, 5 middle-right, 6 bottom-left, 7 bottom-middle, 8 bottom-right.",
        json!({
            "type": "object",
            "properties": { "cell": { "type": "integer", "minimum": 0, "maximum": 8 } },
            "required": ["cell"],
            "additionalProperties": false
        }),
        json!({ "type": "object" }),
    )
}

fn board_prompt(board: &BoardState) -> String {
    let status = board_status(board);
    format!(
        "You are O. The human is X.\nCurrent turn: {}.\nIndex map:\n0 top-left | 1 top-middle | 2 top-right\n3 middle-left | 4 center | 5 middle-right\n6 bottom-left | 7 bottom-middle | 8 bottom-right\nCurrent marks by index:\n{}\nVisual board:\n{}\nLegal moves: {:?}\nStatus: {}.\nIf it is O's turn and the game is not over, call `play_move` exactly once before answering. Only choose one of the legal move indexes. Use `read_board` only when needed. Finish with `submit \"<one short user-facing sentence>\"`; do not repeat that sentence as prose outside the lashlang block. If your move ended the game, clearly say that you won or that the game ended in a draw; otherwise say it is the human's turn. Do not explain your strategy, do not describe threats, do not print an ASCII board, do not narrate every cell, and do not return JSON to the user.",
        board.turn,
        indexed_marks(board),
        board_rows(board),
        legal_moves(board),
        status
    )
}

fn indexed_marks(board: &BoardState) -> String {
    (0..9)
        .map(|index| {
            let mark = board
                .cells
                .get(index)
                .and_then(|cell| cell.as_deref())
                .unwrap_or("empty");
            format!("{index}: {mark}")
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn board_rows(board: &BoardState) -> String {
    (0..3)
        .map(|row| {
            (0..3)
                .map(|col| {
                    let index = row * 3 + col;
                    board
                        .cells
                        .get(index)
                        .and_then(|cell| cell.as_deref())
                        .unwrap_or(".")
                        .to_string()
                })
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn legal_moves(board: &BoardState) -> Vec<usize> {
    if winner(&board.cells).is_some() {
        return Vec::new();
    }
    board
        .cells
        .iter()
        .enumerate()
        .filter_map(|(index, cell)| cell.is_none().then_some(index))
        .collect()
}

fn board_status(board: &BoardState) -> String {
    if let Some(winner) = winner(&board.cells) {
        return format!("{winner} won");
    }
    if legal_moves(board).is_empty() {
        return "draw".to_string();
    }
    format!("{} to move", board.turn)
}

fn board_snapshot(board: &BoardState) -> serde_json::Value {
    json!({
        "cells": board.cells,
        "index_map": [
            "0 top-left", "1 top-middle", "2 top-right",
            "3 middle-left", "4 center", "5 middle-right",
            "6 bottom-left", "7 bottom-middle", "8 bottom-right"
        ],
        "marks_by_index": indexed_marks(board),
        "turn": board.turn,
        "legal_moves": legal_moves(board),
        "status": board_status(board),
        "winner": winner(&board.cells),
    })
}

fn apply_agent_move(board: &BoardState, cell: usize) -> serde_json::Value {
    if board.turn != "O" {
        return json!({
            "accepted": false,
            "reason": "It is not O's turn.",
            "board": board_snapshot(board),
        });
    }
    if cell >= 9
        || board
            .cells
            .get(cell)
            .and_then(|value| value.as_ref())
            .is_some()
    {
        return json!({
            "accepted": false,
            "reason": "Cell is not legal.",
            "board": board_snapshot(board),
        });
    }
    let mut next = board.clone();
    next.cells[cell] = Some("O".to_string());
    next.turn = "X".to_string();
    json!({
        "accepted": true,
        "move": { "mark": "O", "cell": cell },
        "board": board_snapshot(&next),
    })
}

fn winner(cells: &[Option<String>]) -> Option<&'static str> {
    const LINES: [[usize; 3]; 8] = [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
        [0, 3, 6],
        [1, 4, 7],
        [2, 5, 8],
        [0, 4, 8],
        [2, 4, 6],
    ];
    for [a, b, c] in LINES {
        let Some(mark) = cells.get(a).and_then(|cell| cell.as_deref()) else {
            continue;
        };
        if cells.get(b).and_then(|cell| cell.as_deref()) == Some(mark)
            && cells.get(c).and_then(|cell| cell.as_deref()) == Some(mark)
        {
            return match mark {
                "X" => Some("X"),
                "O" => Some("O"),
                _ => None,
            };
        }
    }
    None
}

struct AppDb {
    conn: Connection,
}

impl AppDb {
    fn open(path: &Path) -> rusqlite::Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|err| rusqlite::Error::ToSqlConversionFailure(Box::new(err)))?;
        }
        let conn = Connection::open(path)?;
        conn.execute_batch(
            "
            PRAGMA foreign_keys = ON;
            CREATE TABLE IF NOT EXISTS chats (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                model TEXT NOT NULL,
                model_variant TEXT
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
                kind TEXT NOT NULL DEFAULT 'message',
                role TEXT NOT NULL,
                text TEXT NOT NULL,
                payload TEXT,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_messages_chat_id_id
                ON messages(chat_id, id);
            CREATE TABLE IF NOT EXISTS code_block_tool_calls (
                code_block_message_id INTEGER NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
                tool_call_message_id INTEGER REFERENCES messages(id) ON DELETE SET NULL,
                call_id TEXT NOT NULL,
                PRIMARY KEY (code_block_message_id, call_id)
            );
            CREATE INDEX IF NOT EXISTS idx_code_block_tool_calls_tool_message
                ON code_block_tool_calls(tool_call_message_id);
            ",
        )?;
        add_column_if_missing(&conn, "chats", "model_variant", "TEXT")?;
        add_column_if_missing(&conn, "messages", "kind", "TEXT NOT NULL DEFAULT 'message'")?;
        add_column_if_missing(&conn, "messages", "payload", "TEXT")?;
        let mut db = Self { conn };
        db.migrate_legacy_chat_model_labels()?;
        Ok(db)
    }

    fn list_chats(&mut self) -> AppResult<Vec<ChatSummary>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, title, created_at, updated_at, model, model_variant
             FROM chats ORDER BY updated_at DESC",
        )?;
        let rows = stmt.query_map([], chat_summary_from_row)?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(AppError::from)
    }

    fn create_chat(
        &mut self,
        title: &str,
        model: &str,
        model_variant: Option<&str>,
    ) -> AppResult<ChatSummary> {
        let id = uuid::Uuid::new_v4().to_string();
        let now = now();
        self.conn.execute(
            "INSERT INTO chats (id, title, created_at, updated_at, model, model_variant)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![id, title, now, now, model, model_variant],
        )?;
        Ok(self.chat(&id)?)
    }

    fn require_chat(&mut self, chat_id: &str) -> AppResult<()> {
        if self.chat(chat_id).optional()?.is_some() {
            return Ok(());
        }
        Err(AppError {
            status: StatusCode::NOT_FOUND,
            message: format!("chat `{chat_id}` not found"),
        })
    }

    fn chat(&mut self, chat_id: &str) -> rusqlite::Result<ChatSummary> {
        self.conn.query_row(
            "SELECT id, title, created_at, updated_at, model, model_variant FROM chats WHERE id = ?1",
            params![chat_id],
            chat_summary_from_row,
        )
    }

    fn update_chat_model(
        &mut self,
        chat_id: &str,
        model: &str,
        model_variant: Option<&str>,
    ) -> AppResult<ChatSummary> {
        let changed = self.conn.execute(
            "UPDATE chats SET model = ?1, model_variant = ?2, updated_at = ?3 WHERE id = ?4",
            params![model, model_variant, now(), chat_id],
        )?;
        if changed == 0 {
            return Err(AppError {
                status: StatusCode::NOT_FOUND,
                message: format!("chat `{chat_id}` not found"),
            });
        }
        Ok(self.chat(chat_id)?)
    }

    fn chat_model_selection(&mut self, chat_id: &str) -> AppResult<ChatModelSelection> {
        self.conn
            .query_row(
                "SELECT model, model_variant FROM chats WHERE id = ?1",
                params![chat_id],
                |row| {
                    Ok(ChatModelSelection {
                        model: row.get(0)?,
                        model_variant: row.get(1)?,
                    })
                },
            )
            .map_err(AppError::from)
    }

    fn migrate_legacy_chat_model_labels(&mut self) -> rusqlite::Result<()> {
        let mut stmt = self
            .conn
            .prepare("SELECT id, model, model_variant FROM chats")?;
        let rows = stmt.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, Option<String>>(2)?,
            ))
        })?;
        let updates = rows.collect::<rusqlite::Result<Vec<_>>>()?;
        drop(stmt);
        for (id, model, variant) in updates {
            if variant.is_some() {
                continue;
            }
            if let Some((model, variant)) = split_legacy_model_label(&model) {
                self.conn.execute(
                    "UPDATE chats SET model = ?1, model_variant = ?2 WHERE id = ?3",
                    params![model, variant, id],
                )?;
            }
        }
        Ok(())
    }

    fn maybe_title_from_first_message(&mut self, chat_id: &str, text: &str) -> AppResult<()> {
        let title: String = self.conn.query_row(
            "SELECT title FROM chats WHERE id = ?1",
            params![chat_id],
            |row| row.get(0),
        )?;
        if title != "New chat" {
            return Ok(());
        }
        let message_count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM messages WHERE chat_id = ?1",
            params![chat_id],
            |row| row.get(0),
        )?;
        if message_count == 0 {
            self.conn.execute(
                "UPDATE chats SET title = ?1, updated_at = ?2 WHERE id = ?3",
                params![compact_title(text), now(), chat_id],
            )?;
        }
        Ok(())
    }

    fn insert_message(&mut self, chat_id: &str, role: &str, text: &str) -> AppResult<ChatMessage> {
        self.insert_message_with_payload(chat_id, role, text, None)
    }

    fn insert_message_with_payload(
        &mut self,
        chat_id: &str,
        role: &str,
        text: &str,
        payload: Option<serde_json::Value>,
    ) -> AppResult<ChatMessage> {
        let created_at = now();
        let payload_text = payload.as_ref().map(serde_json::Value::to_string);
        self.conn.execute(
            "INSERT INTO messages (chat_id, kind, role, text, payload, created_at)
             VALUES (?1, 'message', ?2, ?3, ?4, ?5)",
            params![chat_id, role, text, payload_text, created_at],
        )?;
        self.conn.execute(
            "UPDATE chats SET updated_at = ?1 WHERE id = ?2",
            params![now(), chat_id],
        )?;
        let id = self.conn.last_insert_rowid();
        Ok(ChatMessage {
            id,
            chat_id: chat_id.to_string(),
            kind: "message".to_string(),
            role: role.to_string(),
            text: text.to_string(),
            payload,
            created_at,
        })
    }

    fn insert_reasoning(&mut self, chat_id: &str, text: &str) -> AppResult<ChatMessage> {
        let created_at = now();
        self.conn.execute(
            "INSERT INTO messages (chat_id, kind, role, text, payload, created_at)
             VALUES (?1, 'reasoning', 'assistant', ?2, NULL, ?3)",
            params![chat_id, text, created_at],
        )?;
        self.conn.execute(
            "UPDATE chats SET updated_at = ?1 WHERE id = ?2",
            params![now(), chat_id],
        )?;
        let id = self.conn.last_insert_rowid();
        Ok(ChatMessage {
            id,
            chat_id: chat_id.to_string(),
            kind: "reasoning".to_string(),
            role: "assistant".to_string(),
            text: text.to_string(),
            payload: None,
            created_at,
        })
    }

    fn update_reasoning(&mut self, id: i64, text: &str) -> AppResult<ChatMessage> {
        self.conn.execute(
            "UPDATE messages SET text = ?1 WHERE id = ?2 AND kind = 'reasoning'",
            params![text, id],
        )?;
        self.message_by_id(id)
    }

    fn insert_tool_call(&mut self, chat_id: &str, event: TurnEvent) -> AppResult<ChatMessage> {
        let payload = tool_payload(event)?;
        let created_at = now();
        let tool_name = payload["name"].as_str().unwrap_or("tool").to_string();
        self.conn.execute(
            "INSERT INTO messages (chat_id, kind, role, text, payload, created_at)
             VALUES (?1, 'tool_call', 'tool', ?2, ?3, ?4)",
            params![chat_id, tool_name, payload.to_string(), created_at],
        )?;
        self.conn.execute(
            "UPDATE chats SET updated_at = ?1 WHERE id = ?2",
            params![now(), chat_id],
        )?;
        let id = self.conn.last_insert_rowid();
        Ok(ChatMessage {
            id,
            chat_id: chat_id.to_string(),
            kind: "tool_call".to_string(),
            role: "tool".to_string(),
            text: tool_name,
            payload: Some(payload),
            created_at,
        })
    }

    fn update_tool_call(&mut self, id: i64, event: TurnEvent) -> AppResult<ChatMessage> {
        let payload = tool_payload(event)?;
        let tool_name = payload["name"].as_str().unwrap_or("tool").to_string();
        self.conn.execute(
            "UPDATE messages SET text = ?1, payload = ?2
             WHERE id = ?3 AND kind = 'tool_call'",
            params![tool_name, payload.to_string(), id],
        )?;
        self.message_by_id(id)
    }

    fn insert_code_block(
        &mut self,
        chat_id: &str,
        event: TurnEvent,
        code: Option<String>,
    ) -> AppResult<ChatMessage> {
        let (payload, tool_call_ids) = match event {
            TurnEvent::CodeBlockStarted { language, code } => (
                json!({
                    "phase": "started",
                    "language": language,
                    "code": code,
                }),
                Vec::new(),
            ),
            TurnEvent::CodeBlockCompleted {
                language,
                output,
                error,
                success,
                duration_ms,
                tool_call_ids,
            } => (
                json!({
                    "phase": "completed",
                    "language": language,
                    "output": output,
                    "error": error,
                    "success": success,
                    "duration_ms": duration_ms,
                    "tool_call_ids": tool_call_ids,
                    "code": code,
                }),
                tool_call_ids,
            ),
            _ => return Err(AppError::internal("expected code-block event")),
        };
        let created_at = now();
        let language = payload["language"].as_str().unwrap_or("code").to_string();
        self.conn.execute(
            "INSERT INTO messages (chat_id, kind, role, text, payload, created_at)
             VALUES (?1, 'code_block', 'assistant', ?2, ?3, ?4)",
            params![chat_id, language, payload.to_string(), created_at],
        )?;
        self.conn.execute(
            "UPDATE chats SET updated_at = ?1 WHERE id = ?2",
            params![now(), chat_id],
        )?;
        let id = self.conn.last_insert_rowid();
        self.replace_code_block_tool_links(chat_id, id, &tool_call_ids)?;
        Ok(ChatMessage {
            id,
            chat_id: chat_id.to_string(),
            kind: "code_block".to_string(),
            role: "assistant".to_string(),
            text: language,
            payload: Some(payload),
            created_at,
        })
    }

    fn update_code_block(
        &mut self,
        id: i64,
        event: TurnEvent,
        code: Option<String>,
    ) -> AppResult<ChatMessage> {
        let (payload, tool_call_ids) = match event {
            TurnEvent::CodeBlockCompleted {
                language,
                output,
                error,
                success,
                duration_ms,
                tool_call_ids,
            } => (
                json!({
                    "phase": "completed",
                    "language": language,
                    "output": output,
                    "error": error,
                    "success": success,
                    "duration_ms": duration_ms,
                    "tool_call_ids": tool_call_ids,
                    "code": code,
                }),
                tool_call_ids,
            ),
            _ => return Err(AppError::internal("expected code-block event")),
        };
        let language = payload["language"].as_str().unwrap_or("code").to_string();
        self.conn.execute(
            "UPDATE messages SET text = ?1, payload = ?2
             WHERE id = ?3 AND kind = 'code_block'",
            params![language, payload.to_string(), id],
        )?;
        let chat_id = self
            .conn
            .query_row(
                "SELECT chat_id FROM messages WHERE id = ?1 AND kind = 'code_block'",
                params![id],
                |row| row.get::<_, String>(0),
            )
            .optional()?
            .ok_or_else(|| AppError::internal("code block message not found"))?;
        self.replace_code_block_tool_links(&chat_id, id, &tool_call_ids)?;
        self.message_by_id(id)
    }

    fn list_messages(&mut self, chat_id: &str) -> AppResult<Vec<ChatMessage>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, chat_id, kind, role, text, payload, created_at
             FROM messages WHERE chat_id = ?1 ORDER BY id ASC",
        )?;
        let rows = stmt.query_map(params![chat_id], chat_message_from_row)?;
        let mut messages = rows
            .collect::<rusqlite::Result<Vec<_>>>()
            .map_err(AppError::from)?;
        drop(stmt);
        self.hydrate_code_block_tool_links(&mut messages)?;
        Ok(messages)
    }

    fn message_by_id(&mut self, id: i64) -> AppResult<ChatMessage> {
        self.conn
            .query_row(
                "SELECT id, chat_id, kind, role, text, payload, created_at
                 FROM messages WHERE id = ?1",
                params![id],
                chat_message_from_row,
            )
            .map_err(AppError::from)
    }

    fn replace_code_block_tool_links(
        &mut self,
        chat_id: &str,
        code_block_message_id: i64,
        tool_call_ids: &[String],
    ) -> AppResult<()> {
        self.conn.execute(
            "DELETE FROM code_block_tool_calls WHERE code_block_message_id = ?1",
            params![code_block_message_id],
        )?;
        for call_id in tool_call_ids {
            let tool_message_id = self
                .find_tool_call_message_id(chat_id, call_id)
                .map_err(AppError::from)?;
            self.conn.execute(
                "INSERT OR REPLACE INTO code_block_tool_calls
                 (code_block_message_id, tool_call_message_id, call_id)
                 VALUES (?1, ?2, ?3)",
                params![code_block_message_id, tool_message_id, call_id],
            )?;
        }
        Ok(())
    }

    fn find_tool_call_message_id(
        &mut self,
        chat_id: &str,
        call_id: &str,
    ) -> rusqlite::Result<Option<i64>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, payload FROM messages
             WHERE chat_id = ?1 AND kind = 'tool_call'
             ORDER BY id ASC",
        )?;
        let rows = stmt.query_map(params![chat_id], |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, Option<String>>(1)?))
        })?;
        for row in rows {
            let (id, payload) = row?;
            let Some(payload) = payload else {
                continue;
            };
            let Ok(payload) = serde_json::from_str::<serde_json::Value>(&payload) else {
                continue;
            };
            if payload.get("call_id").and_then(|value| value.as_str()) == Some(call_id) {
                return Ok(Some(id));
            }
        }
        Ok(None)
    }

    fn hydrate_code_block_tool_links(&mut self, messages: &mut [ChatMessage]) -> AppResult<()> {
        for message in messages.iter_mut() {
            if message.kind != "code_block" {
                continue;
            }
            let Some(payload) = message.payload.as_mut() else {
                continue;
            };
            let mut stmt = self.conn.prepare(
                "SELECT call_id FROM code_block_tool_calls
                 WHERE code_block_message_id = ?1
                 ORDER BY rowid ASC",
            )?;
            let call_ids = stmt
                .query_map(params![message.id], |row| row.get::<_, String>(0))?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            if !call_ids.is_empty()
                && let Some(object) = payload.as_object_mut()
            {
                object.insert("tool_call_ids".to_string(), json!(call_ids));
            }
        }
        Ok(())
    }
}

fn tool_payload(event: TurnEvent) -> AppResult<serde_json::Value> {
    match event {
        TurnEvent::ToolCallStarted {
            call_id,
            name,
            args,
        } => Ok(json!({
            "phase": "started",
            "call_id": call_id,
            "name": name,
            "args": args,
        })),
        TurnEvent::ToolCallCompleted {
            call_id,
            name,
            args,
            output,
            duration_ms,
        } => Ok(json!({
            "phase": "completed",
            "call_id": call_id,
            "name": name,
            "args": args,
            "output": output,
            "duration_ms": duration_ms,
        })),
        _ => Err(AppError::internal("expected tool-call event")),
    }
}

fn chat_summary_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<ChatSummary> {
    let model: String = row.get(4)?;
    let model_variant: Option<String> = row.get(5)?;
    let model_label = model_label(&model, model_variant.as_deref());
    Ok(ChatSummary {
        id: row.get(0)?,
        title: row.get(1)?,
        created_at: row.get(2)?,
        updated_at: row.get(3)?,
        model,
        model_variant,
        model_label,
    })
}

fn chat_message_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<ChatMessage> {
    let payload: Option<String> = row.get(5)?;
    Ok(ChatMessage {
        id: row.get(0)?,
        chat_id: row.get(1)?,
        kind: row.get(2)?,
        role: row.get(3)?,
        text: row.get(4)?,
        payload: payload.and_then(|value| serde_json::from_str(&value).ok()),
        created_at: row.get(6)?,
    })
}

fn add_column_if_missing(
    conn: &Connection,
    table: &str,
    column: &str,
    definition: &str,
) -> rusqlite::Result<()> {
    let mut stmt = conn.prepare(&format!("PRAGMA table_info({table})"))?;
    let columns = stmt.query_map([], |row| row.get::<_, String>(1))?;
    for existing in columns {
        if existing? == column {
            return Ok(());
        }
    }
    conn.execute(
        &format!("ALTER TABLE {table} ADD COLUMN {column} {definition}"),
        [],
    )?;
    Ok(())
}

fn now() -> String {
    chrono::Utc::now().to_rfc3339()
}

fn normalize_model_selection(
    model: Option<&str>,
    model_variant: Option<&str>,
    default_model: &str,
    default_model_variant: Option<&str>,
) -> AppResult<ChatModelSelection> {
    let model = model
        .map(str::trim)
        .filter(|model| !model.is_empty())
        .unwrap_or(default_model)
        .to_string();
    let model_variant = normalize_model_variant(model_variant).or_else(|| {
        default_model_variant
            .map(str::trim)
            .filter(|variant| !variant.is_empty())
            .map(str::to_string)
    });
    if model.trim().is_empty() {
        return Err(AppError::bad_request("model is required"));
    }
    Ok(ChatModelSelection {
        model,
        model_variant,
    })
}

fn normalize_optional_model_selection(
    model: Option<&str>,
    model_variant: Option<&str>,
) -> AppResult<Option<ChatModelSelection>> {
    let Some(model) = model.map(str::trim).filter(|model| !model.is_empty()) else {
        return Ok(None);
    };
    Ok(Some(ChatModelSelection {
        model: model.to_string(),
        model_variant: normalize_model_variant(model_variant),
    }))
}

fn normalize_model_variant(model_variant: Option<&str>) -> Option<String> {
    model_variant
        .map(str::trim)
        .filter(|variant| !variant.is_empty())
        .map(str::to_string)
}

fn model_label(model: &str, model_variant: Option<&str>) -> String {
    match model_variant
        .map(str::trim)
        .filter(|variant| !variant.is_empty())
    {
        Some(variant) => format!("{model} ({variant})"),
        None => model.to_string(),
    }
}

fn split_legacy_model_label(label: &str) -> Option<(String, String)> {
    let label = label.trim();
    let (model, variant) = label.rsplit_once(" (")?;
    let variant = variant.strip_suffix(')')?.trim();
    let model = model.trim();
    if model.is_empty() || variant.is_empty() {
        return None;
    }
    Some((model.to_string(), variant.to_string()))
}

fn assistant_text_for_persistence(output: &TurnOutput, streamed_prose: &str) -> String {
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

fn terminal_value_text(value: &serde_json::Value) -> String {
    value
        .as_str()
        .map(str::to_string)
        .unwrap_or_else(|| value.to_string())
}

fn compact_title(text: &str) -> String {
    let text = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut title = text.chars().take(54).collect::<String>();
    if text.chars().count() > 54 {
        title.push_str("...");
    }
    if title.is_empty() {
        "New chat".to_string()
    } else {
        title
    }
}

mod anyhow_like {
    pub type Result<T> = std::result::Result<T, String>;
}

const INDEX_HTML: &str = r#"<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Tic Tac Toe Agent</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Big+Shoulders+Display:wght@600;760&family=Chivo+Mono:wght@400;600&family=Spectral:wght@400;600&display=swap');
    :root {
      color-scheme: dark;
      --ink: oklch(0.91 0.018 78);
      --muted: oklch(0.68 0.025 78);
      --dim: oklch(0.48 0.023 78);
      --line: oklch(0.31 0.027 78);
      --paper: oklch(0.15 0.018 78);
      --panel: oklch(0.19 0.019 78);
      --panel-2: oklch(0.23 0.021 78);
      --sun: oklch(0.74 0.155 65);
      --human: oklch(0.73 0.112 165);
      --agent: oklch(0.68 0.12 48);
      --bad: oklch(0.62 0.19 32);
      font-family: Spectral, Georgia, serif;
    }
    * { box-sizing: border-box; }
    html { width:100%; height:100%; min-height:100%; overflow:hidden; background:var(--paper); }
    body { width:100%; height:100%; min-height:100%; overflow:hidden; margin:0; background:var(--paper); color:var(--ink); }
    body::before { content:""; position:fixed; inset:0; pointer-events:none; opacity:.22; background:repeating-linear-gradient(135deg, transparent 0 10px, oklch(0.24 0.025 78) 10px 11px); }
    .app { display:grid; grid-template-columns:312px minmax(0, 1fr); width:100%; height:100vh; height:100dvh; min-height:100vh; min-height:100dvh; overflow:hidden; position:relative; background:var(--panel); }
    aside { border-right:1px solid var(--line); background:oklch(0.17 0.019 78); display:flex; flex-direction:column; min-width:0; }
    header { padding:20px; border-bottom:1px solid var(--line); display:grid; gap:12px; }
    h1 { margin:0; font-family:"Big Shoulders Display", sans-serif; font-size:32px; line-height:.92; letter-spacing:.01em; text-transform:uppercase; }
    .subhead { color:var(--muted); font-size:13px; line-height:1.35; }
    button, select, input, textarea { font:inherit; }
    button { border:1px solid var(--sun); background:var(--sun); color:oklch(0.15 0.018 78); border-radius:3px; padding:9px 12px; cursor:pointer; font-family:"Chivo Mono", monospace; font-size:13px; font-weight:600; }
    button.secondary { background:transparent; color:var(--ink); border-color:var(--line); }
    button:disabled { opacity:.55; cursor:default; }
    .chat-list { overflow:auto; padding:12px; display:grid; gap:8px; }
    .chat-row { width:100%; text-align:left; background:transparent; color:var(--ink); border-color:transparent; display:grid; gap:3px; padding:10px; }
    .chat-row.active { border-color:var(--sun); background:var(--panel); }
    .chat-title, .chat-model { overflow:hidden; white-space:nowrap; text-overflow:ellipsis; }
    .chat-model { color:var(--muted); font:12px "Chivo Mono", monospace; }
    main { min-width:0; min-height:0; display:grid; grid-template-rows:auto auto minmax(0,1fr) auto; background:var(--panel); overflow:hidden; }
    .topbar { min-height:66px; padding:12px 20px; border-bottom:1px solid var(--line); display:flex; justify-content:space-between; gap:16px; align-items:center; background:oklch(0.18 0.019 78); }
    .topbar-title { display:grid; gap:2px; min-width:0; }
    .topbar-title strong { font-family:"Chivo Mono", monospace; font-size:13px; color:var(--sun); }
    .topbar-title span { color:var(--muted); font-size:12px; }
    .model-controls { display:grid; grid-template-columns:minmax(210px, 340px) 120px; gap:8px; align-items:end; margin-left:auto; }
    .model-controls .field { gap:4px; }
    .model-controls input, .model-controls select { font-family:"Chivo Mono", monospace; font-size:12px; padding:8px; }
    .game { border-bottom:1px solid var(--line); padding:22px 26px; display:grid; grid-template-columns:auto minmax(260px,380px); grid-template-areas:"board status" "board actions"; gap:14px 26px; align-items:start; justify-content:start; background:var(--panel-2); }
    .board { display:grid; grid-template-columns:repeat(3,64px); grid-template-rows:repeat(3,64px); gap:7px; }
    .game .board { grid-area:board; }
    .cell { width:64px; height:64px; display:grid; place-items:center; border:1px solid var(--line); border-radius:5px; background:oklch(0.16 0.018 78); color:var(--ink); font:34px/1 "Big Shoulders Display", sans-serif; padding:0; }
    .cell:not(:disabled):hover { border-color:var(--sun); background:oklch(0.21 0.025 78); }
    .cell.x { color:var(--human); }
    .cell.o { color:var(--agent); }
    .cell.win { border-color:var(--sun); background:oklch(0.22 0.045 92); box-shadow:inset 0 0 0 1px color-mix(in oklch, var(--sun), transparent 45%); }
    .cell:disabled { opacity:1; cursor:default; }
    .game-status { grid-area:status; display:grid; gap:7px; min-width:0; align-self:end; }
    .game-status strong { font:24px/1 "Big Shoulders Display", sans-serif; letter-spacing:0; text-transform:uppercase; }
    .game-status span { color:var(--muted); font-size:13px; }
    .game-status.done strong { color:var(--sun); }
    .game-status.done span { color:var(--ink); }
    #resetBoard { grid-area:actions; justify-self:start; align-self:start; min-height:42px; }
    .players { display:flex; gap:8px; flex-wrap:wrap; }
    .player { border:1px solid var(--line); border-radius:5px; padding:8px 10px; background:oklch(0.18 0.018 78); display:grid; gap:2px; min-width:108px; }
    .player b { font:22px/1 "Big Shoulders Display", sans-serif; }
    .player span { color:var(--muted); font:12px "Chivo Mono", monospace; }
    .player.you { border-color:color-mix(in oklch, var(--human), var(--line)); }
    .player.agent { border-color:color-mix(in oklch, var(--agent), var(--line)); }
    .messages { min-height:0; overflow:auto; padding:20px; display:grid; align-content:start; gap:14px; overscroll-behavior:contain; }
    .msg { max-width:min(820px, 78%); white-space:pre-wrap; line-height:1.45; border:1px solid var(--line); border-radius:5px; padding:10px 12px; background:oklch(0.18 0.018 78); }
    .msg.user { justify-self:end; border-color:color-mix(in oklch, var(--human), var(--line)); background:oklch(0.18 0.024 150); }
    .meta { color:var(--muted); font:12px "Chivo Mono", monospace; margin-bottom:4px; }
    .tool { max-width:min(820px, 82%); border:1px solid var(--line); border-radius:5px; padding:11px; background:oklch(0.17 0.02 78); font-size:13px; display:grid; gap:8px; }
    .tool-head { display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
    .tool strong { color:var(--sun); font-family:"Chivo Mono", monospace; }
    .tool.fail strong { color:var(--bad); }
    .badge { border:1px solid var(--line); border-radius:999px; padding:2px 7px; color:var(--muted); background:oklch(0.16 0.018 78); font:12px "Chivo Mono", monospace; }
    .tool-summary { color:var(--ink); }
    .reasoning, .code-block { max-width:min(820px, 82%); border:1px dashed var(--line); border-radius:5px; padding:9px 11px; background:oklch(0.16 0.018 78); color:var(--muted); }
    .reasoning { border-color:color-mix(in oklch, var(--sun), var(--line)); }
    .reasoning summary, .code-block summary { cursor:pointer; font-family:"Chivo Mono", monospace; font-size:12px; color:var(--sun); }
    .reasoning pre, .code-block pre { white-space:pre-wrap; margin:8px 0 0; font-size:12px; line-height:1.45; overflow:auto; }
    .code-block.fail summary { color:var(--bad); }
    .tool details { border-top:1px solid var(--line); padding-top:7px; }
    .tool summary { color:var(--muted); cursor:pointer; }
    .tool pre { overflow:auto; margin:8px 0 0; font-size:12px; }
    form { border-top:1px solid var(--line); padding:14px; display:grid; grid-template-columns:1fr auto; gap:10px; align-items:end; background:oklch(0.18 0.019 78); }
    .field { display:grid; gap:5px; min-width:0; }
    label { font-size:12px; color:var(--muted); }
    textarea, input, select { width:100%; border:1px solid var(--line); border-radius:3px; padding:9px; background:oklch(0.14 0.018 78); color:var(--ink); }
    textarea { min-height:76px; resize:vertical; }
    @media (max-width: 760px) {
      .app { grid-template-columns:1fr; }
      aside { display:none; }
      .topbar { padding:11px 14px; display:grid; grid-template-columns:1fr auto; }
      .model-controls { grid-column:1 / -1; grid-template-columns:1fr 118px; width:100%; margin-left:0; }
      .game { grid-template-columns:1fr; grid-template-areas:"board" "status" "actions"; padding:18px 14px; justify-items:start; }
      .game-status { align-self:start; }
      form { grid-template-columns:1fr; }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside>
      <header>
        <h1>Tic Tac Toe Agent</h1>
        <div class="subhead">You play X. The RLM agent plays O through app-owned board tools.</div>
        <button id="newChat">New chat</button>
      </header>
      <div id="chats" class="chat-list"></div>
    </aside>
    <main>
      <div class="topbar">
        <div class="topbar-title">
          <strong id="activeTitle">No chat</strong>
          <span>Board state is sent to the agent each turn</span>
        </div>
        <div class="model-controls">
          <label class="field">Model
            <input id="modelInput" autocomplete="off" spellcheck="false" />
          </label>
          <label class="field">Variant
            <select id="variantInput">
              <option value="">default</option>
            </select>
          </label>
        </div>
        <button id="mobileNew" class="secondary">New chat</button>
      </div>
      <section class="game">
        <div id="board" class="board"></div>
        <div class="game-status">
          <div class="players">
            <div class="player you"><b>X</b><span>You</span></div>
            <div class="player agent"><b>O</b><span>Agent</span></div>
          </div>
          <strong id="gameStatus">X to move</strong>
          <span id="gameHint">Click any empty square. The agent replies automatically as O.</span>
        </div>
        <button id="resetBoard" class="secondary" type="button">Reset board</button>
      </section>
      <div id="messages" class="messages"></div>
      <form id="composer">
        <textarea id="text" placeholder="Optional: add a note for the agent. Board clicks send turns automatically."></textarea>
        <button id="send" type="submit">Send</button>
      </form>
    </main>
  </div>
  <script>
    const chatsEl = document.querySelector('#chats');
    const messagesEl = document.querySelector('#messages');
    const titleEl = document.querySelector('#activeTitle');
    const form = document.querySelector('#composer');
    const modelInput = document.querySelector('#modelInput');
    const variantInput = document.querySelector('#variantInput');
    const boardEl = document.querySelector('#board');
    const gameStatusEl = document.querySelector('#gameStatus');
    const gameHintEl = document.querySelector('#gameHint');
    let chats = [];
    let activeChat = null;
    let settings = { default_model:'openai/gpt-5.5', default_model_variant:'medium', model_variants:['low','medium','high'] };
    let streaming = null;
    let reasoning = null;
    let pendingCodeBlock = null;
    let pendingTools = [];
    let busy = false;
    const boards = new Map();

    function emptyBoard() {
      return { cells:Array(9).fill(null), turn:'X' };
    }
    function currentBoard() {
      if (!activeChat) return emptyBoard();
      if (!boards.has(activeChat)) boards.set(activeChat, emptyBoard());
      return boards.get(activeChat);
    }
    function winner(cells) {
      const lines = [[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]];
      for (const [a,b,c] of lines) if (cells[a] && cells[a] === cells[b] && cells[a] === cells[c]) return cells[a];
      return null;
    }
    function winningLine(cells) {
      const lines = [[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]];
      return lines.find(([a,b,c]) => cells[a] && cells[a] === cells[b] && cells[a] === cells[c]) || [];
    }
    function boardStatus(board) {
      const win = winner(board.cells);
      if (win === 'X') return 'You won';
      if (win === 'O') return 'Agent won';
      if (board.cells.every(Boolean)) return 'Draw';
      return `${board.turn} to move`;
    }
    function terminalHint(board) {
      const win = winner(board.cells);
      if (win === 'X') return 'You won this round.';
      if (win === 'O') return 'Agent won this round.';
      if (board.cells.every(Boolean)) return 'The round ended in a draw.';
      return null;
    }
    function cellName(index) {
      return ['top left','top middle','top right','middle left','center','middle right','bottom left','bottom middle','bottom right'][index] || `cell ${index}`;
    }
    function renderBoard() {
      const board = currentBoard();
      boardEl.innerHTML = '';
      const done = Boolean(winner(board.cells)) || board.cells.every(Boolean);
      const winCells = new Set(winningLine(board.cells));
      board.cells.forEach((mark, index) => {
        const cell = document.createElement('button');
        cell.type = 'button';
        cell.className = `cell ${mark ? mark.toLowerCase() : ''}${winCells.has(index) ? ' win' : ''}`;
        cell.textContent = mark || '';
        cell.ariaLabel = `cell ${index}`;
        cell.disabled = busy || Boolean(mark) || board.turn !== 'X' || done;
        cell.onclick = () => playHuman(index);
        boardEl.appendChild(cell);
      });
      gameStatusEl.parentElement.classList.toggle('done', done);
      gameStatusEl.textContent = boardStatus(board);
      gameHintEl.textContent = done
        ? `${terminalHint(board)} Reset the board to start another round.`
        : busy
          ? 'Agent is thinking and may call board tools.'
        : board.turn === 'X'
          ? 'Your turn: click any empty square.'
          : 'Agent turn: waiting for O to play.';
    }
    function selectedModel() {
      return {
        model: modelInput.value.trim() || settings.default_model,
        model_variant: variantInput.value || null
      };
    }
    function setModelControls(chat) {
      const model = chat?.model || settings.default_model;
      const variant = chat?.model_variant ?? settings.default_model_variant ?? '';
      modelInput.value = model;
      variantInput.value = variant || '';
    }
    async function loadSettings() {
      settings = await (await api('/api/settings')).json();
      variantInput.innerHTML = '<option value="">default</option>';
      for (const variant of settings.model_variants || []) {
        const option = document.createElement('option');
        option.value = variant;
        option.textContent = variant;
        variantInput.appendChild(option);
      }
      setModelControls(null);
    }
    async function saveActiveModel() {
      if (!activeChat || busy) return;
      const selection = selectedModel();
      const chat = await (await api(`/api/chats/${activeChat}/model`, {
        method:'POST',
        body: JSON.stringify(selection)
      })).json();
      const index = chats.findIndex(item => item.id === chat.id);
      if (index >= 0) chats[index] = chat;
      renderChats();
    }
    function setBoard(board) {
      if (!activeChat || !board || !Array.isArray(board.cells)) return;
      const cells = board.cells.slice(0, 9).map(cell => cell === 'X' || cell === 'O' ? cell : null);
      const terminal = Boolean(winner(cells)) || cells.every(Boolean);
      boards.set(activeChat, {
        cells,
        turn: terminal ? 'X' : board.turn === 'O' ? 'O' : 'X'
      });
      renderBoard();
    }
    function resetBoard() {
      if (!activeChat) return;
      boards.set(activeChat, emptyBoard());
      renderBoard();
    }
    function playHuman(index) {
      const board = currentBoard();
      if (board.turn !== 'X' || board.cells[index] || winner(board.cells)) return;
      board.cells[index] = 'X';
      board.turn = 'O';
      renderBoard();
      sendText(`I played X in the ${cellName(index)}.`);
    }
    function applyToolBoard(event) {
      const raw = toolResult(event);
      const board = raw?.board?.cells ? raw.board : raw;
      if (board?.cells) setBoard(board);
    }
    function terminalToolSummary(board) {
      if (!board?.cells) return null;
      const win = winner(board.cells);
      if (win === 'O') return 'Agent won this round.';
      if (win === 'X') return 'You won this round.';
      if (board.cells.every(Boolean)) return 'The round ended in a draw.';
      return null;
    }
    function cleanArgs(args) {
      const out = { ...(args || {}) };
      delete out.__session_id__;
      return out;
    }
    function toolResult(event) {
      return event?.result
        ?? event?.output?.outcome?.payload
        ?? event?.output?.value
        ?? null;
    }
    function toolSucceeded(event) {
      if (typeof event?.success === 'boolean') return event.success;
      return event?.output?.outcome?.status === 'success';
    }
    function compactToolPayload(event) {
      const raw = toolResult(event);
      if (event.name === 'play_move') {
        return {
          args: cleanArgs(event.args),
          accepted: raw?.accepted,
          move: raw?.move,
          status: raw?.board?.status,
          turn: raw?.board?.turn,
          winner: raw?.board?.winner
        };
      }
      if (event.name === 'read_board') {
        return {
          args: cleanArgs(event.args),
          status: raw?.status,
          turn: raw?.turn,
          legal_moves: raw?.legal_moves,
          winner: raw?.winner,
          marks_by_index: raw?.marks_by_index
        };
      }
      return { args: cleanArgs(event.args), result: raw };
    }
    function renderTerminalValue(value) {
      if (value === null || value === undefined) return '';
      if (typeof value === 'string') return value;
      return JSON.stringify(value, null, 2);
    }

    async function api(url, options = {}) {
      const res = await fetch(url, { headers: { 'content-type': 'application/json' }, ...options });
      if (!res.ok) throw new Error((await res.json()).error || res.statusText);
      return res;
    }
    async function loadChats() {
      chats = await (await api('/api/chats')).json();
      if (!activeChat && chats[0]) activeChat = chats[0].id;
      renderChats();
      if (activeChat) await loadMessages(activeChat);
    }
    function renderChats() {
      chatsEl.innerHTML = '';
      for (const chat of chats) {
        const b = document.createElement('button');
        b.className = 'chat-row' + (chat.id === activeChat ? ' active' : '');
        b.innerHTML = `<span class="chat-title"></span><span class="chat-model"></span>`;
        b.querySelector('.chat-title').textContent = chat.title;
        b.querySelector('.chat-model').textContent = chat.model_label;
        b.onclick = async () => { activeChat = chat.id; renderChats(); await loadMessages(chat.id); };
        chatsEl.appendChild(b);
      }
      const current = chats.find(c => c.id === activeChat);
      titleEl.textContent = current ? current.title : 'No chat';
      if (current) setModelControls(current);
    }
    async function newChat() {
      const chat = await (await api('/api/chats', { method:'POST', body: JSON.stringify(selectedModel()) })).json();
      chats.unshift(chat); activeChat = chat.id; boards.set(chat.id, emptyBoard()); renderChats(); renderBoard(); messagesEl.innerHTML = '';
    }
    async function loadMessages(id) {
      boards.set(id, emptyBoard());
      const messages = await (await api(`/api/chats/${id}/messages`)).json();
      messagesEl.innerHTML = '';
      // Replay persisted board snapshots in message order. User messages carry
      // the human move; tool rows carry accepted agent moves.
      const toolsByCallId = new Map();
      const codeLinkedToolIds = new Set();
      for (const message of messages) {
        if (message.kind === 'tool_call' && message.payload?.phase === 'completed' && message.payload.call_id) {
          toolsByCallId.set(message.payload.call_id, message.payload);
        }
        if (message.kind === 'code_block' && message.payload?.phase === 'completed') {
          for (const callId of message.payload.tool_call_ids || []) codeLinkedToolIds.add(callId);
        }
      }
      for (const message of messages) appendMessage(message, { toolsByCallId, codeLinkedToolIds });
      renderBoard();
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }
    function appendMessage(message, replay = {}) {
      if (message.kind === 'reasoning') {
        appendReasoningMessage(message.text);
        return;
      }
      if (message.kind === 'tool_call' && message.payload) {
        if (message.payload.phase !== 'completed') return;
        if (message.payload.call_id && replay.codeLinkedToolIds?.has(message.payload.call_id)) return;
        appendTool(message.payload);
        return;
      }
      if (message.kind === 'code_block' && message.payload) {
        if (message.payload.phase !== 'completed') return;
        const linkedTools = (message.payload.tool_call_ids || [])
          .map((callId) => replay.toolsByCallId?.get(callId))
          .filter(Boolean);
        appendCodeBlock(message.payload, linkedTools);
        return;
      }
      if (message.payload?.board?.cells) setBoard(message.payload.board);
      const el = document.createElement('div');
      el.className = `msg ${message.role}`;
      el.innerHTML = `<div class="meta"></div><div></div>`;
      el.querySelector('.meta').textContent = message.role;
      el.lastElementChild.textContent = message.text;
      messagesEl.appendChild(el);
    }
    function appendTool(event, parent = messagesEl) {
      if (event.phase !== 'completed') return;
      applyToolBoard(event);
      const ok = toolSucceeded(event);
      const el = document.createElement('div');
      el.className = 'tool' + (ok ? '' : ' fail');
      el.innerHTML = `<div class="tool-head"><strong></strong><span class="badge"></span><span></span></div><div class="tool-summary"></div><details><summary>JSON payload</summary><pre></pre></details>`;
      el.querySelector('strong').textContent = event.name;
      el.querySelector('.badge').textContent = 'completed';
      el.querySelector('.tool-head span:last-child').textContent = `${ok ? 'ok' : 'failed'} in ${event.duration_ms}ms`;
      const summary = el.querySelector('.tool-summary');
      const raw = toolResult(event);
      if (event.name === 'play_move') {
        const terminal = terminalToolSummary(raw?.board);
        summary.textContent = raw?.accepted
          ? `Agent played O in ${cellName(raw.move?.cell)}. ${terminal || raw.board?.status || ''}`
          : `Move rejected: ${raw?.reason || 'unknown reason'}`;
      } else if (event.name === 'read_board') {
        summary.textContent = `${raw?.status || 'Board read'} · legal moves: ${(raw?.legal_moves || []).join(', ') || 'none'}`;
      } else {
        summary.textContent = 'Tool completed';
      }
      el.querySelector('pre').textContent = JSON.stringify(
        compactToolPayload(event),
        null,
        2
      );
      parent.appendChild(el);
    }
    function appendCodeBlock(event, linkedTools = []) {
      if (event.phase !== 'completed') return;
      const el = document.createElement('details');
      el.className = 'code-block' + (event.success === false ? ' fail' : '');
      el.open = false;
      el.innerHTML = '<summary></summary><pre></pre>';
      const toolCount = linkedTools.length || (event.tool_call_ids || []).length;
      const toolLabel = toolCount ? ` · ${toolCount} tool${toolCount === 1 ? '' : 's'}` : '';
      const label = `${event.language || 'code'} ${event.success ? 'completed' : 'failed'} in ${event.duration_ms || 0}ms${toolLabel}`;
      el.querySelector('summary').textContent = label;
      const code = event.code || el.querySelector('pre').textContent || '';
      el.querySelector('pre').textContent = code;
      for (const tool of linkedTools) appendTool(tool, el);
      messagesEl.appendChild(el);
      return el;
    }
    function appendCompletedTool(event) {
      if (pendingCodeBlock) {
        pendingTools.push(event);
      } else {
        appendTool(event);
      }
    }
    function completeCodeBlock(event) {
      const linkedIds = new Set(event.tool_call_ids || []);
      const linkedTools = pendingTools.filter((tool) => tool.call_id && linkedIds.has(tool.call_id));
      const unlinkedTools = pendingTools.filter((tool) => !tool.call_id || !linkedIds.has(tool.call_id));
      appendCodeBlock(
        { ...event, phase:'completed', code: event.code || pendingCodeBlock?.code || '' },
        linkedTools,
      );
      pendingCodeBlock = null;
      for (const tool of unlinkedTools) appendTool(tool);
      pendingTools = [];
    }
    function thinkingPanel(label) {
      const el = document.createElement('details');
      el.className = 'reasoning';
      el.open = true;
      el.innerHTML = '<summary></summary><pre></pre>';
      el.querySelector('summary').textContent = label;
      return el;
    }
    function appendReasoning(delta) {
      if (!reasoning) {
        reasoning = thinkingPanel('thinking');
        messagesEl.appendChild(reasoning);
      }
      reasoning.querySelector('pre').textContent += delta;
    }
    function appendReasoningMessage(text) {
      if (!text) return;
      if (reasoning) {
        reasoning.open = true;
        reasoning.querySelector('summary').textContent = 'thinking';
        reasoning.querySelector('pre').textContent = text;
        reasoning.dataset.persisted = 'true';
        reasoning = null;
        return;
      }
      const el = thinkingPanel('thinking');
      el.querySelector('pre').textContent = text;
      messagesEl.appendChild(el);
    }
    function finishReasoning() {
      reasoning = null;
    }
    function appendStreamText(delta) {
      if (!streaming) {
        streaming = document.createElement('div');
        streaming.className = 'msg assistant';
        streaming.innerHTML = '<div class="meta">assistant</div><div></div>';
        messagesEl.appendChild(streaming);
      }
      streaming.lastElementChild.textContent += delta;
    }
    async function sendText(text) {
      if (!activeChat) await newChat();
      if (!text) return;
      if (busy) return;
      document.querySelector('#text').value = '';
      streaming = null;
      reasoning = null;
      pendingCodeBlock = null;
      pendingTools = [];
      busy = true;
      renderBoard();
      const res = await api(`/api/chats/${activeChat}/messages`, {
        method:'POST',
        body: JSON.stringify({
          text,
          board: currentBoard(),
          ...selectedModel()
        })
      });
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream:true });
        const lines = buffer.split('\n');
        buffer = lines.pop();
        for (const line of lines) {
          if (!line.trim()) continue;
          const item = JSON.parse(line);
          if (item.type === 'message') appendMessage(item.message);
          if (item.type === 'event' && item.event.type === 'assistant_prose_delta') appendStreamText(item.event.text);
          if (item.type === 'event' && item.event.type === 'reasoning_delta') appendReasoning(item.event.text);
          if (item.type === 'event' && item.event.type === 'code_block_started') pendingCodeBlock = item.event;
          if (item.type === 'event' && item.event.type === 'code_block_completed') completeCodeBlock(item.event);
          if (item.type === 'event' && item.event.type === 'tool_call_completed') appendCompletedTool({ ...item.event, phase:'completed' });
          if (item.type === 'event' && item.event.type === 'submitted_value') appendStreamText(renderTerminalValue(item.event.value));
          if (item.type === 'event' && item.event.type === 'tool_value') appendStreamText(renderTerminalValue(item.event.value));
          if (item.type === 'error') alert(item.message);
          messagesEl.scrollTop = messagesEl.scrollHeight;
        }
      }
      streaming = null;
      for (const tool of pendingTools) appendTool(tool);
      pendingCodeBlock = null;
      pendingTools = [];
      finishReasoning();
      busy = false;
      renderBoard();
      await loadChats();
    }
    async function send(e) {
      e.preventDefault();
      await sendText(document.querySelector('#text').value.trim());
    }
    document.querySelector('#newChat').onclick = newChat;
    document.querySelector('#mobileNew').onclick = newChat;
    document.querySelector('#resetBoard').onclick = resetBoard;
    modelInput.addEventListener('change', saveActiveModel);
    variantInput.addEventListener('change', saveActiveModel);
    document.querySelector('#text').addEventListener('keydown', (event) => {
      if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
        event.preventDefault();
        form.requestSubmit();
      }
    });
    form.onsubmit = send;
    renderBoard();
    loadSettings().then(() => loadChats()).then(() => { if (!activeChat) newChat(); else renderBoard(); });
  </script>
</body>
</html>"#;
