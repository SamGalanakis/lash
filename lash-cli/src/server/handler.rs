//! Request handler: dispatches JSON-RPC methods to implementations.

use std::collections::{BTreeSet, HashMap};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::SystemTime;

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use lash_core::provider::Provider;
use lash_core::tools::{FilteredTools, ToolSet, ToolSetDeps};
use lash_core::*;

use super::protocol::{self, *};
use protocol::TurnStatus;
use super::transport::MessageWriter;
use super::{ActiveTurn, TurnReturn};
use crate::session_log;
use crate::skill::SkillRegistry;

// ─── Loaded thread state ───

struct LoadedThread {
    runtime: Option<RuntimeEngine>,
    state: AgentStateEnvelope,
    thread_meta: Thread,
    cancel_token: CancellationToken,
    active_turn_id: Option<String>,
}

// ─── Server handler ───

pub struct ServerHandler {
    initialized: bool,
    client_info: Option<ClientInfo>,
    notification_filter: NotificationFilter,
    writer: MessageWriter,
    threads: HashMap<String, LoadedThread>,
    config: RuntimeConfig,
    provider: Provider,
    model: String,
    lash_config: LashConfig,
    skills: SkillRegistry,
    item_counter: AtomicU64,
}

impl ServerHandler {
    pub fn new(
        writer: MessageWriter,
        config: RuntimeConfig,
        provider: Provider,
        model: String,
        lash_config: LashConfig,
    ) -> Self {
        let skills = SkillRegistry::load();
        Self {
            initialized: false,
            client_info: None,
            notification_filter: NotificationFilter::empty(),
            writer,
            threads: HashMap::new(),
            config,
            provider,
            model,
            lash_config,
            skills,
            item_counter: AtomicU64::new(0),
        }
    }

    pub fn next_item_id(&self) -> String {
        let n = self.item_counter.fetch_add(1, Ordering::Relaxed);
        format!("item_{n}")
    }

    pub fn should_send_notification(&self, method: &str) -> bool {
        self.notification_filter.should_send(method)
    }

    /// Return the runtime to a thread after a turn completes.
    pub fn complete_turn(
        &mut self,
        thread_id: &str,
        runtime: RuntimeEngine,
        state: AgentStateEnvelope,
    ) {
        if let Some(lt) = self.threads.get_mut(thread_id) {
            lt.runtime = Some(runtime);
            lt.state = state;
            lt.active_turn_id = None;
        }
    }

    /// Clear active turn marker (e.g. on panic).
    pub fn clear_active_turn(&mut self, thread_id: &str) {
        if let Some(lt) = self.threads.get_mut(thread_id) {
            lt.active_turn_id = None;
        }
    }

    /// Handle a request. Returns Some(ActiveTurn) if a turn was started.
    pub async fn handle_request(
        &mut self,
        method: &str,
        id: serde_json::Value,
        params: Option<serde_json::Value>,
    ) -> Option<ActiveTurn> {
        if method == "initialize" {
            self.handle_initialize(id, params).await;
            return None;
        }

        if !self.initialized {
            self.writer
                .send_response(&JsonRpcResponse::err(id, NOT_INITIALIZED, "Not initialized"))
                .await;
            return None;
        }

        match method {
            "thread/start" => {
                self.handle_thread_start(id, params).await;
                None
            }
            "thread/resume" => {
                self.handle_thread_resume(id, params).await;
                None
            }
            "thread/list" => {
                self.handle_thread_list(id, params).await;
                None
            }
            "thread/read" => {
                self.handle_thread_read(id, params).await;
                None
            }
            "thread/archive" => {
                self.handle_thread_archive(id, params).await;
                None
            }
            "thread/unsubscribe" => {
                self.handle_thread_unsubscribe(id, params).await;
                None
            }
            "thread/loaded/list" => {
                self.handle_thread_loaded_list(id).await;
                None
            }
            "turn/start" => self.handle_turn_start(id, params).await,
            "turn/interrupt" => {
                self.handle_turn_interrupt(id, params).await;
                None
            }
            "skills/list" => {
                self.handle_skills_list(id).await;
                None
            }
            "model/list" => {
                self.handle_model_list(id).await;
                None
            }
            _ => {
                self.writer
                    .send_response(&JsonRpcResponse::err(
                        id,
                        METHOD_NOT_FOUND,
                        format!("Unknown method: {method}"),
                    ))
                    .await;
                None
            }
        }
    }

    pub async fn handle_notification(
        &mut self,
        method: &str,
        _params: Option<serde_json::Value>,
    ) {
        match method {
            "initialized" => {
                tracing::info!("client acknowledged initialization");
            }
            _ => {
                tracing::debug!("ignoring unknown notification: {method}");
            }
        }
    }

    // ─── Initialize ───

    async fn handle_initialize(
        &mut self,
        id: serde_json::Value,
        params: Option<serde_json::Value>,
    ) {
        if self.initialized {
            self.writer
                .send_response(&JsonRpcResponse::err(
                    id,
                    ALREADY_INITIALIZED,
                    "Already initialized",
                ))
                .await;
            return;
        }

        let init_params: InitializeParams = match params
            .and_then(|p| serde_json::from_value(p).ok())
        {
            Some(v) => v,
            None => {
                self.writer
                    .send_response(&JsonRpcResponse::err(
                        id,
                        INVALID_PARAMS,
                        "Missing or invalid params for initialize",
                    ))
                    .await;
                return;
            }
        };

        let opt_out = init_params
            .capabilities
            .as_ref()
            .map(|c| c.opt_out_notification_methods.clone())
            .unwrap_or_default();

        self.client_info = Some(init_params.client_info);
        self.notification_filter = NotificationFilter::new(opt_out);
        self.initialized = true;

        tracing::info!(
            "initialized: client={:?}",
            self.client_info.as_ref().map(|c| &c.name)
        );

        let result = InitializeResult {
            server_info: ServerInfo {
                name: "lash-app-server".to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
            },
        };

        self.writer
            .send_response(&JsonRpcResponse::ok(
                id,
                serde_json::to_value(result).unwrap(),
            ))
            .await;
    }

    // ─── Thread management ───

    async fn handle_thread_start(
        &mut self,
        id: serde_json::Value,
        params: Option<serde_json::Value>,
    ) {
        let start_params: ThreadStartParams = params
            .and_then(|p| serde_json::from_value(p).ok())
            .unwrap_or_default();

        if let Some(ref cwd) = start_params.cwd {
            if let Err(e) = std::env::set_current_dir(cwd) {
                self.writer
                    .send_response(&JsonRpcResponse::err(
                        id,
                        INVALID_PARAMS,
                        format!("Failed to set cwd: {e}"),
                    ))
                    .await;
                return;
            }
        }

        let model = start_params.model.unwrap_or_else(|| self.model.clone());
        let thread_id = short_uuid("thr");
        let now = chrono::Utc::now().timestamp();

        let mut config = self.config.clone();
        config.model = model.clone();
        config.provider = self.provider.clone();
        config.headless = true;

        let tools = build_tools(&self.lash_config, &config);
        let state = AgentStateEnvelope::default();

        let runtime = match RuntimeEngine::from_state(config, tools, state.clone()).await {
            Ok(rt) => rt,
            Err(e) => {
                self.writer
                    .send_response(&JsonRpcResponse::err(
                        id,
                        INTERNAL_ERROR,
                        format!("Failed to create runtime: {e}"),
                    ))
                    .await;
                return;
            }
        };

        let thread = Thread {
            id: thread_id.clone(),
            preview: String::new(),
            model: model.clone(),
            provider: provider_name(&self.provider).to_string(),
            created_at: now,
            updated_at: Some(now),
            status: Some(ThreadStatus::Idle),
            turns: None,
        };

        self.threads.insert(
            thread_id.clone(),
            LoadedThread {
                runtime: Some(runtime),
                state,
                thread_meta: thread.clone(),
                cancel_token: CancellationToken::new(),
                active_turn_id: None,
            },
        );

        self.notify(
            "thread/started",
            serde_json::json!({ "thread": thread }),
        )
        .await;

        self.writer
            .send_response(&JsonRpcResponse::ok(
                id,
                serde_json::json!({ "thread": thread }),
            ))
            .await;
    }

    async fn handle_thread_resume(
        &mut self,
        id: serde_json::Value,
        params: Option<serde_json::Value>,
    ) {
        let resume_params: ThreadResumeParams =
            match params.and_then(|p| serde_json::from_value(p).ok()) {
                Some(p) => p,
                None => {
                    self.writer
                        .send_response(&JsonRpcResponse::err(
                            id,
                            INVALID_PARAMS,
                            "Missing or invalid threadId",
                        ))
                        .await;
                    return;
                }
            };

        // Already loaded
        if let Some(lt) = self.threads.get(&resume_params.thread_id) {
            self.writer
                .send_response(&JsonRpcResponse::ok(
                    id,
                    serde_json::json!({ "thread": lt.thread_meta }),
                ))
                .await;
            return;
        }

        // Load from disk
        let sessions = session_log::list_sessions();
        let Some(session_info) = sessions
            .iter()
            .find(|s| s.session_id == resume_params.thread_id)
        else {
            self.writer
                .send_response(&JsonRpcResponse::err(
                    id,
                    INVALID_PARAMS,
                    format!("Thread not found: {}", resume_params.thread_id),
                ))
                .await;
            return;
        };

        let Some((messages, _)) = session_log::load_session(&session_info.filename) else {
            self.writer
                .send_response(&JsonRpcResponse::err(
                    id,
                    INTERNAL_ERROR,
                    "Failed to load session data",
                ))
                .await;
            return;
        };

        let mut config = self.config.clone();
        config.provider = self.provider.clone();
        config.headless = true;

        let tools = build_tools(&self.lash_config, &config);
        let state = AgentStateEnvelope {
            messages,
            ..AgentStateEnvelope::default()
        };

        let runtime = match RuntimeEngine::from_state(config, tools, state.clone()).await {
            Ok(rt) => rt,
            Err(e) => {
                self.writer
                    .send_response(&JsonRpcResponse::err(
                        id,
                        INTERNAL_ERROR,
                        format!("Failed to resume runtime: {e}"),
                    ))
                    .await;
                return;
            }
        };

        let now = chrono::Utc::now().timestamp();
        let thread = Thread {
            id: resume_params.thread_id.clone(),
            preview: truncate_preview(&session_info.first_message, 120),
            model: self.model.clone(),
            provider: provider_name(&self.provider).to_string(),
            created_at: unix_timestamp(&session_info.modified).unwrap_or(now),
            updated_at: Some(now),
            status: Some(ThreadStatus::Idle),
            turns: None,
        };

        self.threads.insert(
            resume_params.thread_id.clone(),
            LoadedThread {
                runtime: Some(runtime),
                state,
                thread_meta: thread.clone(),
                cancel_token: CancellationToken::new(),
                active_turn_id: None,
            },
        );

        self.writer
            .send_response(&JsonRpcResponse::ok(
                id,
                serde_json::json!({ "thread": thread }),
            ))
            .await;
    }

    async fn handle_thread_list(
        &mut self,
        id: serde_json::Value,
        params: Option<serde_json::Value>,
    ) {
        let list_params: ThreadListParams = params
            .and_then(|p| serde_json::from_value(p).ok())
            .unwrap_or_default();

        let sessions = session_log::list_sessions();
        let limit = list_params.limit.unwrap_or(50);

        let data: Vec<Thread> = sessions
            .iter()
            .take(limit)
            .map(|s| {
                let status = self
                    .threads
                    .get(&s.session_id)
                    .map(|lt| {
                        if lt.active_turn_id.is_some() {
                            ThreadStatus::Active {
                                active_flags: vec![],
                            }
                        } else {
                            ThreadStatus::Idle
                        }
                    })
                    .unwrap_or(ThreadStatus::NotLoaded);

                Thread {
                    id: s.session_id.clone(),
                    preview: truncate_preview(&s.first_message, 120),
                    model: self.model.clone(),
                    provider: provider_name(&self.provider).to_string(),
                    created_at: unix_timestamp(&s.modified).unwrap_or(0),
                    updated_at: None,
                    status: Some(status),
                    turns: None,
                }
            })
            .collect();

        self.writer
            .send_response(&JsonRpcResponse::ok(
                id,
                serde_json::json!({ "data": data, "nextCursor": null }),
            ))
            .await;
    }

    async fn handle_thread_read(
        &mut self,
        id: serde_json::Value,
        params: Option<serde_json::Value>,
    ) {
        let read_params: ThreadReadParams =
            match params.and_then(|p| serde_json::from_value(p).ok()) {
                Some(p) => p,
                None => {
                    self.writer
                        .send_response(&JsonRpcResponse::err(
                            id,
                            INVALID_PARAMS,
                            "Missing or invalid threadId",
                        ))
                        .await;
                    return;
                }
            };

        if let Some(lt) = self.threads.get(&read_params.thread_id) {
            let mut thread = lt.thread_meta.clone();
            if read_params.include_turns {
                thread.turns = Some(vec![]);
            }
            self.writer
                .send_response(&JsonRpcResponse::ok(
                    id,
                    serde_json::json!({ "thread": thread }),
                ))
                .await;
            return;
        }

        let sessions = session_log::list_sessions();
        let Some(info) = sessions
            .iter()
            .find(|s| s.session_id == read_params.thread_id)
        else {
            self.writer
                .send_response(&JsonRpcResponse::err(
                    id,
                    INVALID_PARAMS,
                    format!("Thread not found: {}", read_params.thread_id),
                ))
                .await;
            return;
        };

        let thread = Thread {
            id: info.session_id.clone(),
            preview: truncate_preview(&info.first_message, 120),
            model: self.model.clone(),
            provider: provider_name(&self.provider).to_string(),
            created_at: unix_timestamp(&info.modified).unwrap_or(0),
            updated_at: None,
            status: Some(ThreadStatus::NotLoaded),
            turns: if read_params.include_turns {
                Some(vec![])
            } else {
                None
            },
        };

        self.writer
            .send_response(&JsonRpcResponse::ok(
                id,
                serde_json::json!({ "thread": thread }),
            ))
            .await;
    }

    async fn handle_thread_archive(
        &mut self,
        id: serde_json::Value,
        params: Option<serde_json::Value>,
    ) {
        let archive_params: ThreadArchiveParams =
            match params.and_then(|p| serde_json::from_value(p).ok()) {
                Some(p) => p,
                None => {
                    self.writer
                        .send_response(&JsonRpcResponse::err(
                            id,
                            INVALID_PARAMS,
                            "Missing or invalid threadId",
                        ))
                        .await;
                    return;
                }
            };

        self.threads.remove(&archive_params.thread_id);

        let sessions_dir = session_log::sessions_dir();
        let archive_dir = sessions_dir.join("archived");
        std::fs::create_dir_all(&archive_dir).ok();

        let sessions = session_log::list_sessions();
        if let Some(info) = sessions
            .iter()
            .find(|s| s.session_id == archive_params.thread_id)
        {
            let src = sessions_dir.join(&info.filename);
            let dst = archive_dir.join(&info.filename);
            if let Err(e) = std::fs::rename(&src, &dst) {
                self.writer
                    .send_response(&JsonRpcResponse::err(
                        id,
                        INTERNAL_ERROR,
                        format!("Failed to archive: {e}"),
                    ))
                    .await;
                return;
            }
        }

        self.notify(
            "thread/archived",
            serde_json::json!({ "threadId": archive_params.thread_id }),
        )
        .await;

        self.writer
            .send_response(&JsonRpcResponse::ok(id, serde_json::json!({})))
            .await;
    }

    async fn handle_thread_unsubscribe(
        &mut self,
        id: serde_json::Value,
        params: Option<serde_json::Value>,
    ) {
        let unsub_params: ThreadUnsubscribeParams =
            match params.and_then(|p| serde_json::from_value(p).ok()) {
                Some(p) => p,
                None => {
                    self.writer
                        .send_response(&JsonRpcResponse::err(
                            id,
                            INVALID_PARAMS,
                            "Missing or invalid threadId",
                        ))
                        .await;
                    return;
                }
            };

        let status = if self.threads.remove(&unsub_params.thread_id).is_some() {
            self.notify(
                "thread/status/changed",
                serde_json::json!({
                    "threadId": unsub_params.thread_id,
                    "status": { "type": "notLoaded" }
                }),
            )
            .await;
            self.notify(
                "thread/closed",
                serde_json::json!({ "threadId": unsub_params.thread_id }),
            )
            .await;
            "unsubscribed"
        } else {
            "notSubscribed"
        };

        self.writer
            .send_response(&JsonRpcResponse::ok(
                id,
                serde_json::json!({ "status": status }),
            ))
            .await;
    }

    async fn handle_thread_loaded_list(&mut self, id: serde_json::Value) {
        let data: Vec<&String> = self.threads.keys().collect();
        self.writer
            .send_response(&JsonRpcResponse::ok(id, serde_json::json!({ "data": data })))
            .await;
    }

    // ─── Turn execution ───

    async fn handle_turn_start(
        &mut self,
        id: serde_json::Value,
        params: Option<serde_json::Value>,
    ) -> Option<ActiveTurn> {
        let turn_params: TurnStartParams =
            match params.and_then(|p| serde_json::from_value(p).ok()) {
                Some(p) => p,
                None => {
                    self.writer
                        .send_response(&JsonRpcResponse::err(
                            id,
                            INVALID_PARAMS,
                            "Missing or invalid turn/start params",
                        ))
                        .await;
                    return None;
                }
            };

        let thread_id = turn_params.thread_id.clone();

        let Some(lt) = self.threads.get_mut(&thread_id) else {
            self.writer
                .send_response(&JsonRpcResponse::err(
                    id,
                    INVALID_PARAMS,
                    format!("Thread not loaded: {thread_id}"),
                ))
                .await;
            return None;
        };

        if lt.active_turn_id.is_some() {
            self.writer
                .send_response(&JsonRpcResponse::err(
                    id,
                    INVALID_REQUEST,
                    "Thread already has an active turn",
                ))
                .await;
            return None;
        }

        let Some(mut runtime) = lt.runtime.take() else {
            self.writer
                .send_response(&JsonRpcResponse::err(
                    id,
                    INTERNAL_ERROR,
                    "Runtime not available",
                ))
                .await;
            return None;
        };

        // Apply per-turn overrides
        if let Some(ref model) = turn_params.model {
            runtime.set_model(model.clone());
        }

        let (items, image_blobs) = convert_turn_input(&turn_params.input);
        let mode = turn_params
            .mode
            .as_deref()
            .map(|m| match m {
                "plan" => RunMode::Plan,
                _ => RunMode::Normal,
            })
            .unwrap_or(RunMode::Normal);

        let turn_id = short_uuid("turn");

        let cancel = CancellationToken::new();
        lt.cancel_token = cancel.clone();
        lt.active_turn_id = Some(turn_id.clone());

        let turn = Turn {
            id: turn_id.clone(),
            status: TurnStatus::InProgress,
            items: vec![],
            error: None,
        };

        // Emit status change: idle → active
        self.notify(
            "thread/status/changed",
            serde_json::json!({
                "threadId": thread_id,
                "status": { "type": "active", "activeFlags": [] }
            }),
        )
        .await;

        self.notify(
            "turn/started",
            serde_json::json!({ "turn": turn }),
        )
        .await;

        self.writer
            .send_response(&JsonRpcResponse::ok(
                id,
                serde_json::json!({ "turn": turn }),
            ))
            .await;

        // Spawn the turn task
        let (event_tx, event_rx) = mpsc::unbounded_channel();
        let (return_tx, return_rx) = tokio::sync::oneshot::channel();

        let turn_input = TurnInput {
            items,
            image_blobs,
            mode: Some(mode),
            plan_file: None,
        };

        tokio::spawn(async move {
            let sink = ChannelEventSink { tx: event_tx };
            let result = runtime.stream_turn(turn_input, &sink, cancel).await;
            let _ = return_tx.send(TurnReturn { runtime, result });
        });

        Some(ActiveTurn {
            thread_id,
            turn_id,
            event_rx,
            return_rx,
            accumulated_text: String::new(),
            items: vec![],
            error: None,
            agent_message_item_id: None,
        })
    }

    async fn handle_turn_interrupt(
        &mut self,
        id: serde_json::Value,
        params: Option<serde_json::Value>,
    ) {
        let interrupt_params: TurnInterruptParams =
            match params.and_then(|p| serde_json::from_value(p).ok()) {
                Some(p) => p,
                None => {
                    self.writer
                        .send_response(&JsonRpcResponse::err(
                            id,
                            INVALID_PARAMS,
                            "Missing or invalid turn/interrupt params",
                        ))
                        .await;
                    return;
                }
            };

        if let Some(lt) = self.threads.get(&interrupt_params.thread_id) {
            if lt.active_turn_id.as_deref() == Some(&interrupt_params.turn_id) {
                lt.cancel_token.cancel();
                self.writer
                    .send_response(&JsonRpcResponse::ok(id, serde_json::json!({})))
                    .await;
                return;
            }
        }

        self.writer
            .send_response(&JsonRpcResponse::err(
                id,
                INVALID_REQUEST,
                "No matching active turn to interrupt",
            ))
            .await;
    }

    // ─── Skills ───

    async fn handle_skills_list(&mut self, id: serde_json::Value) {
        let skills: Vec<SkillInfo> = self
            .skills
            .iter()
            .map(|s| SkillInfo {
                name: s.name.clone(),
                description: s.description.clone(),
                enabled: true,
            })
            .collect();

        self.writer
            .send_response(&JsonRpcResponse::ok(
                id,
                serde_json::json!({ "data": skills }),
            ))
            .await;
    }

    // ─── Models ───

    async fn handle_model_list(&mut self, id: serde_json::Value) {
        let default_model = self.provider.default_model().to_string();
        let context_window = self.provider.context_window(&default_model);
        let effort = self
            .provider
            .reasoning_effort_for_model(&default_model)
            .map(str::to_string);

        let data = vec![ModelInfo {
            id: default_model,
            context_window,
            reasoning_effort: effort,
        }];

        self.writer
            .send_response(&JsonRpcResponse::ok(
                id,
                serde_json::json!({ "data": data }),
            ))
            .await;
    }

    // ─── Notification helper ───

    async fn notify(&self, method: &str, params: serde_json::Value) {
        if !self.notification_filter.should_send(method) {
            return;
        }
        self.writer
            .send_notification(&JsonRpcNotification {
                method: method.to_string(),
                params: Some(params),
            })
            .await;
    }
}

// ─── Event sink adapter ───

struct ChannelEventSink {
    tx: mpsc::UnboundedSender<AgentEvent>,
}

#[async_trait::async_trait]
impl EventSink for ChannelEventSink {
    async fn emit(&self, event: AgentEvent) {
        let _ = self.tx.send(event);
    }
}

// ─── Helpers ───

fn convert_turn_input(items: &[TurnInputItem]) -> (Vec<InputItem>, HashMap<String, Vec<u8>>) {
    let mut input_items = Vec::new();
    let image_blobs = HashMap::new();

    for item in items {
        match item {
            TurnInputItem::Text { text } => {
                input_items.push(InputItem::Text { text: text.clone() });
            }
            TurnInputItem::FileRef { path } => {
                input_items.push(InputItem::FileRef { path: path.clone() });
            }
            TurnInputItem::DirRef { path } => {
                input_items.push(InputItem::DirRef { path: path.clone() });
            }
            TurnInputItem::Skill { name, .. } => {
                input_items.push(InputItem::SkillRef {
                    name: name.clone(),
                    args: None,
                });
            }
            TurnInputItem::Image { .. } | TurnInputItem::LocalImage { .. } => {
                // Image loading from URL/path is not yet implemented
            }
        }
    }

    (input_items, image_blobs)
}

fn build_tools(lash_config: &LashConfig, config: &RuntimeConfig) -> Arc<dyn ToolProvider> {
    let skill_dirs = vec![
        lash_core::lash_home().join("skills"),
        PathBuf::from(".lash").join("skills"),
    ];
    let tavily_key = lash_config
        .tavily_api_key()
        .unwrap_or_default()
        .to_string();
    let base = ToolSet::defaults(ToolSetDeps {
        store: None,
        tavily_api_key: if tavily_key.is_empty() {
            None
        } else {
            Some(tavily_key)
        },
        skill_dirs: Some(skill_dirs),
    });
    let all_tools: Arc<dyn ToolProvider> = Arc::new(base);
    let defs = all_tools.definitions();
    let resolved = resolve_features(&config.capabilities, &defs);
    let allowed: BTreeSet<String> = defs
        .into_iter()
        .map(|d| d.name)
        .filter(|n| resolved.effective_tools.contains(n))
        .collect();
    Arc::new(FilteredTools::new(all_tools, allowed))
}

fn provider_name(provider: &Provider) -> &'static str {
    match provider {
        Provider::OpenRouter { .. } => "openrouter",
        Provider::Claude { .. } => "claude",
        Provider::Codex { .. } => "codex",
        Provider::GoogleOAuth { .. } => "google",
    }
}

fn truncate_preview(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}…", &s[..max])
    }
}

fn short_uuid(prefix: &str) -> String {
    let uuid = uuid::Uuid::new_v4().to_string().replace('-', "");
    format!("{}_{}", prefix, &uuid[..12])
}

fn unix_timestamp(time: &SystemTime) -> Option<i64> {
    time.duration_since(SystemTime::UNIX_EPOCH)
        .ok()
        .map(|d| d.as_secs() as i64)
}
