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
use lash::ProviderHandle;
use lash_embed::{Input, LashCore, LashSession, TurnEvent, TurnEventSink};
use lash_provider_openai::{OPENROUTER_BASE_URL, OpenAiGenericProvider};
use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};
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
        let session = self.core.session(chat_id).standard().open().await?;
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
        (
            self.status,
            Json(serde_json::json!({
                "error": self.message,
            })),
        )
            .into_response()
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
    role: String,
    text: String,
    created_at: String,
}

#[derive(Debug, Deserialize)]
struct CreateChatRequest {
    title: Option<String>,
}

#[derive(Debug, Deserialize)]
struct SendMessageRequest {
    text: String,
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
    let core = LashCore::standard()
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

    println!("openrouter-chat-ui listening on http://{addr}");
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
        let sink = ChannelTurnEvents { tx: tx.clone() };
        match session.turn(Input::text(text)).events(&sink).run().await {
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
            serde_json::json!({
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
}

#[async_trait]
impl TurnEventSink for ChannelTurnEvents {
    async fn emit(&self, event: TurnEvent) {
        let _ = self.tx.send(StreamItem::Event { event }).await;
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
                role TEXT NOT NULL,
                text TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_messages_chat_id_id
                ON messages(chat_id, id);
            ",
        )?;
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
        if message_count != 0 {
            return Ok(());
        }
        let new_title = compact_title(text);
        self.conn.execute(
            "UPDATE chats SET title = ?1, updated_at = ?2 WHERE id = ?3",
            params![new_title, now(), chat_id],
        )?;
        Ok(())
    }

    fn insert_message(&mut self, chat_id: &str, role: &str, text: &str) -> AppResult<ChatMessage> {
        let created_at = now();
        self.conn.execute(
            "INSERT INTO messages (chat_id, role, text, created_at)
             VALUES (?1, ?2, ?3, ?4)",
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
            role: role.to_string(),
            text: text.to_string(),
            created_at,
        })
    }

    fn list_messages(&mut self, chat_id: &str) -> AppResult<Vec<ChatMessage>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, chat_id, role, text, created_at
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
    Ok(ChatMessage {
        id: row.get(0)?,
        chat_id: row.get(1)?,
        role: row.get(2)?,
        text: row.get(3)?,
        created_at: row.get(4)?,
    })
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
  <title>Lash OpenRouter Chat</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #171614;
      --muted: #6e675d;
      --line: #d8d1c7;
      --paper: #f7f4ee;
      --panel: #fffdf8;
      --accent: #0d6b57;
      --accent-2: #b6462f;
      --soft: #ebe5d9;
      font-family: ui-serif, Georgia, Cambria, "Times New Roman", serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--paper);
      color: var(--ink);
    }
    .app {
      display: grid;
      grid-template-columns: 310px minmax(0, 1fr);
      height: 100vh;
    }
    aside {
      border-right: 1px solid var(--line);
      background: var(--soft);
      display: flex;
      flex-direction: column;
      min-width: 0;
    }
    header {
      padding: 20px;
      border-bottom: 1px solid var(--line);
    }
    h1 {
      margin: 0 0 14px;
      font-size: 22px;
      font-weight: 700;
      letter-spacing: 0;
    }
    button {
      border: 1px solid var(--ink);
      background: var(--ink);
      color: var(--panel);
      padding: 10px 12px;
      font: inherit;
      cursor: pointer;
      border-radius: 4px;
    }
    button.secondary {
      background: transparent;
      color: var(--ink);
    }
    button:disabled { opacity: .55; cursor: default; }
    .chat-list {
      overflow: auto;
      padding: 10px;
      display: grid;
      gap: 8px;
    }
    .chat-row {
      width: 100%;
      text-align: left;
      background: transparent;
      color: var(--ink);
      border-color: transparent;
      display: grid;
      gap: 4px;
    }
    .chat-row.active {
      border-color: var(--accent);
      background: var(--panel);
    }
    .chat-title {
      overflow: hidden;
      white-space: nowrap;
      text-overflow: ellipsis;
    }
    .chat-model {
      color: var(--muted);
      font-size: 12px;
      overflow: hidden;
      white-space: nowrap;
      text-overflow: ellipsis;
    }
    main {
      min-width: 0;
      display: grid;
      grid-template-rows: auto minmax(0, 1fr) auto;
      background: var(--panel);
    }
    .topbar {
      padding: 18px 24px;
      border-bottom: 1px solid var(--line);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
    }
    .status {
      color: var(--muted);
      font-size: 14px;
      min-height: 20px;
    }
    .messages {
      overflow: auto;
      padding: 26px clamp(18px, 4vw, 54px);
      display: flex;
      flex-direction: column;
      gap: 16px;
    }
    .message {
      max-width: 780px;
      border-left: 3px solid var(--line);
      padding: 2px 0 2px 14px;
      white-space: pre-wrap;
      line-height: 1.5;
      font-size: 17px;
    }
    .message.user {
      border-left-color: var(--accent);
      align-self: flex-end;
      text-align: right;
      padding: 2px 14px 2px 0;
      border-left: 0;
      border-right: 3px solid var(--accent);
    }
    .message.assistant { border-left-color: var(--accent-2); }
    .composer {
      border-top: 1px solid var(--line);
      padding: 16px 24px 20px;
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 12px;
      align-items: end;
      background: var(--paper);
    }
    textarea {
      width: 100%;
      min-height: 72px;
      max-height: 190px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 4px;
      padding: 12px;
      font: 16px/1.4 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      background: var(--panel);
      color: var(--ink);
    }
    .empty {
      margin: auto;
      color: var(--muted);
      text-align: center;
      max-width: 420px;
      line-height: 1.5;
    }
    @media (max-width: 760px) {
      .app { grid-template-columns: 1fr; }
      aside { height: 220px; border-right: 0; border-bottom: 1px solid var(--line); }
      main { height: calc(100vh - 220px); }
      .composer { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside>
      <header>
        <h1>Lash OpenRouter</h1>
        <button id="new-chat">New chat</button>
      </header>
      <div id="chat-list" class="chat-list"></div>
    </aside>
    <main>
      <div class="topbar">
        <strong id="active-title">No chat selected</strong>
        <span id="status" class="status"></span>
      </div>
      <section id="messages" class="messages">
        <div class="empty">Create a chat or select one from the list.</div>
      </section>
      <form id="composer" class="composer">
        <textarea id="input" placeholder="Ask something..." disabled></textarea>
        <button id="send" type="submit" disabled>Send</button>
      </form>
    </main>
  </div>
  <script>
    const state = { chats: [], activeChatId: null, sending: false };
    const els = {
      list: document.getElementById('chat-list'),
      messages: document.getElementById('messages'),
      title: document.getElementById('active-title'),
      status: document.getElementById('status'),
      input: document.getElementById('input'),
      send: document.getElementById('send'),
      newChat: document.getElementById('new-chat'),
      composer: document.getElementById('composer'),
    };

    async function api(path, options = {}) {
      const res = await fetch(path, {
        headers: { 'content-type': 'application/json', ...(options.headers || {}) },
        ...options,
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.error || res.statusText);
      }
      return res;
    }

    async function loadChats(selectFirst = true) {
      const res = await api('/api/chats');
      state.chats = await res.json();
      renderChats();
      if (!state.activeChatId && selectFirst && state.chats[0]) {
        await selectChat(state.chats[0].id);
      }
    }

    function renderChats() {
      els.list.innerHTML = '';
      for (const chat of state.chats) {
        const button = document.createElement('button');
        button.className = 'chat-row' + (chat.id === state.activeChatId ? ' active' : '');
        button.innerHTML = `<span class="chat-title"></span><span class="chat-model"></span>`;
        button.querySelector('.chat-title').textContent = chat.title;
        button.querySelector('.chat-model').textContent = chat.model;
        button.onclick = () => selectChat(chat.id);
        els.list.appendChild(button);
      }
    }

    async function createChat() {
      const res = await api('/api/chats', {
        method: 'POST',
        body: JSON.stringify({ title: 'New chat' }),
      });
      const chat = await res.json();
      state.chats.unshift(chat);
      await selectChat(chat.id);
      renderChats();
      els.input.focus();
    }

    async function selectChat(id) {
      state.activeChatId = id;
      const chat = state.chats.find(chat => chat.id === id);
      els.title.textContent = chat ? chat.title : 'Chat';
      els.input.disabled = false;
      els.send.disabled = state.sending;
      renderChats();
      const res = await api(`/api/chats/${id}/messages`);
      renderMessages(await res.json());
    }

    function renderMessages(messages) {
      els.messages.innerHTML = '';
      if (!messages.length) {
        const empty = document.createElement('div');
        empty.className = 'empty';
        empty.textContent = 'Start the conversation.';
        els.messages.appendChild(empty);
        return;
      }
      for (const message of messages) appendMessage(message);
      scrollBottom();
    }

    function appendMessage(message) {
      const previousEmpty = els.messages.querySelector('.empty');
      if (previousEmpty) previousEmpty.remove();
      const div = document.createElement('div');
      div.className = `message ${message.role}`;
      div.textContent = message.text;
      els.messages.appendChild(div);
      scrollBottom();
      return div;
    }

    function appendAssistantDraft() {
      return appendMessage({ role: 'assistant', text: '' });
    }

    async function sendMessage(text) {
      state.sending = true;
      els.send.disabled = true;
      els.input.disabled = true;
      els.status.textContent = 'running';
      let draft = null;
      const res = await api(`/api/chats/${state.activeChatId}/messages`, {
        method: 'POST',
        body: JSON.stringify({ text }),
      });
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop();
        for (const line of lines) {
          if (!line.trim()) continue;
          const item = JSON.parse(line);
          if (item.type === 'message') {
            if (item.message.role === 'user') appendMessage(item.message);
            if (item.message.role === 'assistant') {
              if (draft) draft.textContent = item.message.text;
              else appendMessage(item.message);
            }
          } else if (item.type === 'event' && item.event.type === 'text_delta') {
            if (!draft) draft = appendAssistantDraft();
            draft.textContent += item.event.content;
            scrollBottom();
          } else if (item.type === 'error') {
            els.status.textContent = item.message;
          } else if (item.type === 'done') {
            els.status.textContent = '';
          }
        }
      }
      await loadChats(false);
      state.sending = false;
      els.input.disabled = false;
      els.send.disabled = false;
      els.input.focus();
    }

    function scrollBottom() {
      els.messages.scrollTop = els.messages.scrollHeight;
    }

    els.newChat.onclick = createChat;
    els.composer.onsubmit = async (event) => {
      event.preventDefault();
      const text = els.input.value.trim();
      if (!text || !state.activeChatId || state.sending) return;
      els.input.value = '';
      try {
        await sendMessage(text);
      } catch (err) {
        els.status.textContent = err.message;
        state.sending = false;
        els.input.disabled = false;
        els.send.disabled = false;
      }
    };
    els.input.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        els.composer.requestSubmit();
      }
    });

    loadChats(false).catch(err => { els.status.textContent = err.message; });
  </script>
</body>
</html>"#;
