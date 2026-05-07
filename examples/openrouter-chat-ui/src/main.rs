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
use lash::{ExecutionMode, ProviderHandle, ToolDefinition, ToolExecutionContext, ToolProvider};
use lash_embed::{
    EmbedPlugin, Input, LashCore, LashSession, ModeId, ModePreset, TurnBuilder, TurnEvent,
    TurnEventSink,
};
use lash_provider_openai::{OPENROUTER_BASE_URL, OpenAiGenericProvider};
use lash_rlm_types::RlmTermination;
use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::sync::{Mutex as AsyncMutex, mpsc};
use tokio_stream::StreamExt;
use tokio_stream::wrappers::ReceiverStream;

type AppResult<T> = Result<T, AppError>;

#[derive(Clone)]
struct AppStateData {
    core: LashCore,
    db: Arc<Mutex<AppDb>>,
    sessions: Arc<AsyncMutex<HashMap<String, LashSession>>>,
}

impl AppStateData {
    async fn session_for(&self, chat_id: &str) -> AppResult<LashSession> {
        if let Some(session) = self.sessions.lock().await.get(chat_id).cloned() {
            return Ok(session);
        }
        let session = self
            .core
            .session(chat_id)
            .rlm()
            .use_plugin::<DemoPlugin>(DemoPluginConfig)?
            .open()
            .await?;
        self.sessions
            .lock()
            .await
            .insert(chat_id.to_string(), session.clone());
        Ok(session)
    }

    fn with_db<T>(&self, f: impl FnOnce(&mut AppDb) -> AppResult<T>) -> AppResult<T> {
        let mut db = self
            .db
            .lock()
            .map_err(|_| AppError::internal("database lock poisoned"))?;
        f(&mut db)
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

impl From<lash_embed::EmbedError> for AppError {
    fn from(err: lash_embed::EmbedError) -> Self {
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
}

#[derive(Debug, Deserialize)]
struct SendMessageRequest {
    text: String,
    tone: Option<Tone>,
    page: DemoPageContext,
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum StreamItem {
    Event { event: TurnEvent },
    Message { message: ChatMessage },
    Error { message: String },
    Done,
}

#[tokio::main]
async fn main() -> anyhow_like::Result<()> {
    let api_key = std::env::var("OPENROUTER_API_KEY")
        .map_err(|_| "OPENROUTER_API_KEY is required".to_string())?;
    let model = std::env::var("OPENROUTER_MODEL")
        .unwrap_or_else(|_| "anthropic/claude-sonnet-4.6".to_string());
    let addr: SocketAddr = std::env::var("OPENROUTER_CHAT_ADDR")
        .unwrap_or_else(|_| "127.0.0.1:3000".to_string())
        .parse()
        .map_err(|err| format!("invalid OPENROUTER_CHAT_ADDR: {err}"))?;
    let data_dir = std::env::var("OPENROUTER_CHAT_DATA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(".openrouter-chat-ui"));
    std::fs::create_dir_all(&data_dir).map_err(|err| err.to_string())?;

    let provider = ProviderHandle::new(
        OpenAiGenericProvider::new(api_key, OPENROUTER_BASE_URL).into_components(),
    );
    let store_factory = Arc::new(ChatStoreFactory::new(data_dir.join("lash-sessions")));
    let core = LashCore::builder()
        .install_mode(ModePreset::rlm())
        .default_mode(ModeId::rlm())
        .register_plugin::<DemoPlugin>()
        .provider(provider)
        .model(model.clone())
        .max_context_tokens(200_000)
        .store_factory(store_factory)
        .build()
        .map_err(|err| err.to_string())?;

    let app_db = AppDb::open(&data_dir.join("app.db"), model).map_err(|err| err.to_string())?;
    let state = AppStateData {
        core,
        db: Arc::new(Mutex::new(app_db)),
        sessions: Arc::new(AsyncMutex::new(HashMap::new())),
    };

    let app = Router::new()
        .route("/", get(index))
        .route("/api/chats", get(list_chats).post(create_chat))
        .route(
            "/api/chats/{chat_id}/messages",
            get(list_messages).post(send_message),
        )
        .with_state(state);

    println!("rlm-plugin-chat listening on http://{addr}");
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
    state.with_db(|db| db.list_chats()).map(Json)
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
    state.with_db(|db| db.create_chat(title)).map(Json)
}

async fn list_messages(
    State(state): State<AppStateData>,
    AxumPath(chat_id): AxumPath<String>,
) -> AppResult<Json<Vec<ChatMessage>>> {
    state
        .with_db(|db| {
            db.require_chat(&chat_id)?;
            db.list_messages(&chat_id)
        })
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

    let user_message = state.with_db(|db| {
        db.require_chat(&chat_id)?;
        db.maybe_title_from_first_message(&chat_id, &text)?;
        db.insert_message(&chat_id, "user", &text)
    })?;

    let session = state.session_for(&chat_id).await?;
    let (tx, rx) = mpsc::channel::<StreamItem>(64);
    let run_state = state.clone();
    tokio::spawn(async move {
        let _ = tx
            .send(StreamItem::Message {
                message: user_message,
            })
            .await;
        let sink = ChannelTurnEvents {
            tx: tx.clone(),
            state: run_state.clone(),
            chat_id: chat_id.clone(),
        };
        let turn = session
            .demo_turn(Input::text(text))
            .tone(request.tone.unwrap_or(Tone::Plain))
            .page_context(request.page)
            .events(&sink)
            .mode_turn_options(
                lash::ModeTurnOptions::typed(
                    ExecutionMode::new("rlm"),
                    RlmTermination::Finish {
                        schema: None,
                        include_submit_prompt: true,
                    },
                )
                .expect("RLM termination options serialize"),
            )
            .run()
            .await;
        match turn {
            Ok(result) => {
                let inserted = run_state
                    .with_db(|db| db.insert_message(&chat_id, "assistant", &result.final_text));
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
}

#[async_trait]
impl TurnEventSink for ChannelTurnEvents {
    async fn emit(&self, event: TurnEvent) {
        if let TurnEvent::ToolCall { .. } = &event {
            match self
                .state
                .with_db(|db| db.insert_tool_call(&self.chat_id, event.clone()))
            {
                Ok(message) => {
                    let _ = self.tx.send(StreamItem::Message { message }).await;
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
            return;
        }
        let _ = self.tx.send(StreamItem::Event { event }).await;
    }
}

#[derive(Clone, Debug)]
struct DemoPlugin;

#[derive(Clone, Debug)]
struct DemoPluginConfig;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Tone {
    Plain,
    Warm,
    Precise,
}

impl Tone {
    fn as_prompt(&self) -> &'static str {
        match self {
            Self::Plain => "plain and direct",
            Self::Warm => "warm and conversational",
            Self::Precise => "precise and terse",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct DemoPageContext {
    title: String,
    url: String,
    viewport: String,
}

#[derive(Clone, Debug)]
struct DemoTurnInput {
    tone: Option<Tone>,
    page: DemoPageContext,
}

impl EmbedPlugin for DemoPlugin {
    const ID: &'static str = "demo_page";
    type SessionConfig = DemoPluginConfig;
    type TurnInput = DemoTurnInput;

    fn factory(_config: &Self::SessionConfig) -> Arc<dyn lash::PluginFactory> {
        Arc::new(DemoPluginFactory)
    }

    fn requires_turn_input(_config: &Self::SessionConfig) -> bool {
        true
    }
}

struct DemoPluginFactory;

impl lash::PluginFactory for DemoPluginFactory {
    fn id(&self) -> &'static str {
        DemoPlugin::ID
    }

    fn build(
        &self,
        _ctx: &lash::PluginSessionContext,
    ) -> Result<Arc<dyn lash::SessionPlugin>, lash::PluginError> {
        Ok(Arc::new(DemoSessionPlugin))
    }
}

struct DemoSessionPlugin;

impl lash::SessionPlugin for DemoSessionPlugin {
    fn id(&self) -> &'static str {
        DemoPlugin::ID
    }

    fn register(&self, reg: &mut lash::PluginRegistrar) -> Result<(), lash::PluginError> {
        reg.prompt().contribute(Arc::new(|ctx| {
            Box::pin(async move {
                let Some(input) = ctx
                    .turn_context
                    .plugin_input::<DemoTurnInput>(DemoPlugin::ID)
                else {
                    return Ok(Vec::new());
                };
                let tone = input.tone.as_ref().unwrap_or(&Tone::Plain).as_prompt();
                Ok(vec![lash::PromptContribution::environment(
                    "Demo Page Context",
                    format!(
                        "Tone: {tone}\nPage title: {}\nPage URL: {}\nViewport: {}\nUse `call demo_lookup {{ \"query\": \"...\" }}` from lashlang when page-specific lookup helps.",
                        input.page.title, input.page.url, input.page.viewport
                    ),
                )])
            })
        }));
        reg.tools().provider(Arc::new(DemoTools))
    }
}

struct DemoTools;

#[async_trait]
impl ToolProvider for DemoTools {
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![ToolDefinition::new(
            "demo_lookup",
            "Return app-owned page context for the current RLM turn.",
            json!({
                "type": "object",
                "properties": { "query": { "type": "string" } },
                "required": ["query"],
                "additionalProperties": false
            }),
            json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string" },
                    "page_title": { "type": "string" },
                    "page_url": { "type": "string" },
                    "tone": { "type": "string" }
                }
            }),
        )]
    }

    async fn execute(&self, _name: &str, _args: &serde_json::Value) -> lash::ToolResult {
        lash::ToolResult::err_fmt("demo_lookup requires turn context")
    }

    async fn execute_with_context(
        &self,
        name: &str,
        args: &serde_json::Value,
        context: &ToolExecutionContext,
    ) -> lash::ToolResult {
        if name != "demo_lookup" {
            return lash::ToolResult::err_fmt(format!("unknown tool `{name}`"));
        }
        let Some(input) = context
            .turn_context
            .plugin_input::<DemoTurnInput>(DemoPlugin::ID)
        else {
            return lash::ToolResult::err_fmt("missing DemoPageContext turn input");
        };
        lash::ToolResult::ok(json!({
            "query": args.get("query").and_then(|value| value.as_str()).unwrap_or(""),
            "page_title": input.page.title,
            "page_url": input.page.url,
            "viewport": input.page.viewport,
            "tone": input.tone.as_ref().unwrap_or(&Tone::Plain).as_prompt(),
        }))
    }
}

trait DemoTurnExt<'a> {
    fn demo_turn(&'a self, input: Input) -> DemoTurnBuilder<'a>;
}

impl<'a> DemoTurnExt<'a> for LashSession {
    fn demo_turn(&'a self, input: Input) -> DemoTurnBuilder<'a> {
        DemoTurnBuilder {
            builder: self.turn(input),
            tone: None,
        }
    }
}

struct DemoTurnBuilder<'a> {
    builder: TurnBuilder<'a>,
    tone: Option<Tone>,
}

impl<'a> DemoTurnBuilder<'a> {
    fn tone(mut self, tone: Tone) -> Self {
        self.tone = Some(tone);
        self
    }

    fn page_context(self, page: DemoPageContext) -> DemoReadyTurnBuilder<'a> {
        DemoReadyTurnBuilder {
            builder: self.builder.with_plugin_input::<DemoPlugin>(DemoTurnInput {
                tone: self.tone,
                page,
            }),
        }
    }
}

struct DemoReadyTurnBuilder<'a> {
    builder: TurnBuilder<'a>,
}

impl<'a> DemoReadyTurnBuilder<'a> {
    fn events(mut self, events: &'a dyn TurnEventSink) -> Self {
        self.builder = self.builder.events(events);
        self
    }

    fn mode_turn_options(mut self, options: lash::ModeTurnOptions) -> Self {
        self.builder = self.builder.mode_turn_options(options);
        self
    }

    async fn run(self) -> lash_embed::Result<lash_embed::TurnResult> {
        self.builder.run().await
    }
}

struct AppDb {
    conn: Connection,
    model: String,
}

impl AppDb {
    fn open(path: &Path, model: String) -> rusqlite::Result<Self> {
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
                model TEXT NOT NULL
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
            ",
        )?;
        add_column_if_missing(&conn, "messages", "kind", "TEXT NOT NULL DEFAULT 'message'")?;
        add_column_if_missing(&conn, "messages", "payload", "TEXT")?;
        Ok(Self { conn, model })
    }

    fn list_chats(&mut self) -> AppResult<Vec<ChatSummary>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, title, created_at, updated_at, model
             FROM chats ORDER BY updated_at DESC",
        )?;
        let rows = stmt.query_map([], chat_summary_from_row)?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(AppError::from)
    }

    fn create_chat(&mut self, title: &str) -> AppResult<ChatSummary> {
        let id = uuid::Uuid::new_v4().to_string();
        let now = now();
        self.conn.execute(
            "INSERT INTO chats (id, title, created_at, updated_at, model)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![id, title, now, now, self.model],
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
            "SELECT id, title, created_at, updated_at, model FROM chats WHERE id = ?1",
            params![chat_id],
            chat_summary_from_row,
        )
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
        let created_at = now();
        self.conn.execute(
            "INSERT INTO messages (chat_id, kind, role, text, payload, created_at)
             VALUES (?1, 'message', ?2, ?3, NULL, ?4)",
            params![chat_id, role, text, created_at],
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
            payload: None,
            created_at,
        })
    }

    fn insert_tool_call(&mut self, chat_id: &str, event: TurnEvent) -> AppResult<ChatMessage> {
        let TurnEvent::ToolCall {
            call_id,
            name,
            args,
            result,
            success,
            duration_ms,
        } = event
        else {
            return Err(AppError::internal("expected tool-call event"));
        };
        let created_at = now();
        let payload = json!({
            "call_id": call_id,
            "name": name,
            "args": args,
            "result": result,
            "success": success,
            "duration_ms": duration_ms,
        });
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

    fn list_messages(&mut self, chat_id: &str) -> AppResult<Vec<ChatMessage>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, chat_id, kind, role, text, payload, created_at
             FROM messages WHERE chat_id = ?1 ORDER BY id ASC",
        )?;
        let rows = stmt.query_map(params![chat_id], chat_message_from_row)?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(AppError::from)
    }
}

fn chat_summary_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<ChatSummary> {
    Ok(ChatSummary {
        id: row.get(0)?,
        title: row.get(1)?,
        created_at: row.get(2)?,
        updated_at: row.get(3)?,
        model: row.get(4)?,
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

#[derive(Clone)]
struct ChatStoreFactory {
    sessions_dir: PathBuf,
}

impl ChatStoreFactory {
    fn new(sessions_dir: PathBuf) -> Self {
        Self { sessions_dir }
    }
}

impl lash::SessionStoreFactory for ChatStoreFactory {
    fn create_store(
        &self,
        request: &lash::SessionStoreCreateRequest,
    ) -> Result<Arc<dyn lash::RuntimePersistence>, String> {
        std::fs::create_dir_all(&self.sessions_dir).map_err(|err| err.to_string())?;
        let path = self.sessions_dir.join(format!("{}.db", request.session_id));
        let store = Arc::new(lash_sqlite_store::Store::open(&path).map_err(|err| err.to_string())?);
        store.save_session_meta(lash::SessionMeta {
            session_id: request.session_id.clone(),
            session_name: request.session_id.clone(),
            created_at: now(),
            model: request.policy.model.clone(),
            cwd: std::env::current_dir()
                .ok()
                .and_then(|path| path.to_str().map(str::to_string)),
            parent_session_id: request.parent_session_id.clone(),
        });
        Ok(store as Arc<dyn lash::RuntimePersistence>)
    }
}

fn now() -> String {
    chrono::Utc::now().to_rfc3339()
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
  <title>RLM Plugin Chat</title>
  <style>
    :root { color-scheme: light; --ink:#151515; --muted:#667085; --line:#d0d5dd; --paper:#f6f7f9; --panel:#ffffff; --accent:#0f766e; --bad:#b42318; font-family: ui-sans-serif, system-ui, sans-serif; }
    * { box-sizing: border-box; }
    body { margin:0; min-height:100vh; background:var(--paper); color:var(--ink); }
    .app { display:grid; grid-template-columns:300px minmax(0, 1fr); height:100vh; }
    aside { border-right:1px solid var(--line); background:#eceff3; display:flex; flex-direction:column; min-width:0; }
    header { padding:16px; border-bottom:1px solid var(--line); }
    h1 { margin:0 0 12px; font-size:20px; letter-spacing:0; }
    button, select, input, textarea { font:inherit; }
    button { border:1px solid var(--ink); background:var(--ink); color:white; border-radius:4px; padding:9px 11px; cursor:pointer; }
    button.secondary { background:transparent; color:var(--ink); }
    button:disabled { opacity:.55; cursor:default; }
    .chat-list { overflow:auto; padding:10px; display:grid; gap:8px; }
    .chat-row { width:100%; text-align:left; background:transparent; color:var(--ink); border-color:transparent; display:grid; gap:3px; }
    .chat-row.active { border-color:var(--accent); background:var(--panel); }
    .chat-title, .chat-model { overflow:hidden; white-space:nowrap; text-overflow:ellipsis; }
    .chat-model { color:var(--muted); font-size:12px; }
    main { min-width:0; display:grid; grid-template-rows:auto minmax(0,1fr) auto; background:var(--panel); }
    .topbar { padding:14px 20px; border-bottom:1px solid var(--line); display:flex; justify-content:space-between; gap:12px; align-items:center; }
    .messages { overflow:auto; padding:20px; display:grid; align-content:start; gap:14px; }
    .msg { max-width:820px; white-space:pre-wrap; line-height:1.45; border-left:3px solid var(--line); padding-left:12px; }
    .msg.user { justify-self:end; border-left:0; border-right:3px solid var(--accent); padding-left:0; padding-right:12px; }
    .meta { color:var(--muted); font-size:12px; margin-bottom:4px; }
    .tool { max-width:820px; border:1px solid var(--line); border-radius:6px; padding:10px; background:#f8fafc; font-size:13px; }
    .tool strong { color:var(--accent); }
    .tool.fail strong { color:var(--bad); }
    .tool pre { overflow:auto; margin:8px 0 0; font-size:12px; }
    form { border-top:1px solid var(--line); padding:14px; display:grid; grid-template-columns:170px 1fr auto; gap:10px; align-items:end; }
    .field { display:grid; gap:5px; min-width:0; }
    label { font-size:12px; color:var(--muted); }
    textarea, input, select { width:100%; border:1px solid var(--line); border-radius:4px; padding:9px; background:white; color:var(--ink); }
    textarea { min-height:76px; resize:vertical; grid-column:1 / 3; }
    .page-row { display:grid; grid-template-columns:1fr 1fr 130px; gap:10px; grid-column:1 / 3; }
    @media (max-width: 760px) {
      .app { grid-template-columns:1fr; }
      aside { display:none; }
      form, .page-row { grid-template-columns:1fr; }
      textarea, .page-row { grid-column:1; }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside>
      <header>
        <h1>RLM Plugin Chat</h1>
        <button id="newChat">New chat</button>
      </header>
      <div id="chats" class="chat-list"></div>
    </aside>
    <main>
      <div class="topbar">
        <strong id="activeTitle">No chat</strong>
        <button id="mobileNew" class="secondary">New chat</button>
      </div>
      <div id="messages" class="messages"></div>
      <form id="composer">
        <div class="field">
          <label for="tone">Tone</label>
          <select id="tone">
            <option value="plain">Plain</option>
            <option value="warm">Warm</option>
            <option value="precise">Precise</option>
          </select>
        </div>
        <div></div>
        <div class="page-row">
          <div class="field"><label for="pageTitle">Page title</label><input id="pageTitle" value="Demo dashboard" /></div>
          <div class="field"><label for="pageUrl">Page URL</label><input id="pageUrl" value="https://app.example.local/dashboard" /></div>
          <div class="field"><label for="viewport">Viewport</label><input id="viewport" value="desktop" /></div>
        </div>
        <textarea id="text" placeholder="Ask the RLM agent to inspect the page context or call demo_lookup"></textarea>
        <button id="send" type="submit">Send</button>
      </form>
    </main>
  </div>
  <script>
    const chatsEl = document.querySelector('#chats');
    const messagesEl = document.querySelector('#messages');
    const titleEl = document.querySelector('#activeTitle');
    const form = document.querySelector('#composer');
    let chats = [];
    let activeChat = null;
    let streaming = null;

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
        b.querySelector('.chat-model').textContent = chat.model;
        b.onclick = async () => { activeChat = chat.id; renderChats(); await loadMessages(chat.id); };
        chatsEl.appendChild(b);
      }
      const current = chats.find(c => c.id === activeChat);
      titleEl.textContent = current ? current.title : 'No chat';
    }
    async function newChat() {
      const chat = await (await api('/api/chats', { method:'POST', body: JSON.stringify({}) })).json();
      chats.unshift(chat); activeChat = chat.id; renderChats(); messagesEl.innerHTML = '';
    }
    async function loadMessages(id) {
      const messages = await (await api(`/api/chats/${id}/messages`)).json();
      messagesEl.innerHTML = '';
      for (const message of messages) appendMessage(message);
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }
    function appendMessage(message) {
      if (message.kind === 'tool_call' && message.payload) {
        appendTool(message.payload);
        return;
      }
      const el = document.createElement('div');
      el.className = `msg ${message.role}`;
      el.innerHTML = `<div class="meta"></div><div></div>`;
      el.querySelector('.meta').textContent = message.role;
      el.lastElementChild.textContent = message.text;
      messagesEl.appendChild(el);
    }
    function appendTool(event) {
      const el = document.createElement('div');
      el.className = 'tool' + (event.success ? '' : ' fail');
      el.innerHTML = `<strong></strong> <span></span><pre></pre>`;
      el.querySelector('strong').textContent = event.name;
      el.querySelector('span').textContent = `${event.success ? 'ok' : 'failed'} in ${event.duration_ms}ms`;
      el.querySelector('pre').textContent = JSON.stringify({ args: event.args, result: event.result }, null, 2);
      messagesEl.appendChild(el);
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
    async function send(e) {
      e.preventDefault();
      if (!activeChat) await newChat();
      const text = document.querySelector('#text').value.trim();
      if (!text) return;
      document.querySelector('#text').value = '';
      streaming = null;
      const res = await api(`/api/chats/${activeChat}/messages`, {
        method:'POST',
        body: JSON.stringify({
          text,
          tone: document.querySelector('#tone').value,
          page: {
            title: document.querySelector('#pageTitle').value,
            url: document.querySelector('#pageUrl').value,
            viewport: document.querySelector('#viewport').value
          }
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
          if (item.type === 'event' && item.event.type === 'text_delta') appendStreamText(item.event.content);
          if (item.type === 'error') alert(item.message);
          messagesEl.scrollTop = messagesEl.scrollHeight;
        }
      }
      streaming = null;
      await loadChats();
    }
    document.querySelector('#newChat').onclick = newChat;
    document.querySelector('#mobileNew').onclick = newChat;
    document.querySelector('#text').addEventListener('keydown', (event) => {
      if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
        event.preventDefault();
        form.requestSubmit();
      }
    });
    form.onsubmit = send;
    loadChats().then(() => { if (!activeChat) newChat(); });
  </script>
</body>
</html>"#;
