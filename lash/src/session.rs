use std::collections::{HashMap, VecDeque};
use std::path::Path;
use std::sync::{Arc, Mutex as StdMutex, OnceLock};

use serde_json::json;
use tokio::sync::{Notify, mpsc::UnboundedSender};
use tokio_util::sync::CancellationToken;

use crate::embedded::{LashlangRequest, LashlangResponse, LashlangRuntime};
use crate::tool_dispatch::{
    ToolDispatchContext, ToolDispatchOutcome, dispatch_tool_call,
    dispatch_tool_call_with_execution_context,
};
use crate::{
    ExecResponse, PluginMessage, PromptContribution, RuntimeServices, SandboxMessage, SessionEvent,
    SessionManager, ToolCallRecord, ToolDefinition, ToolExecutionContext, ToolImage, ToolProvider,
};

const REPL_SNAPSHOT_VERSION: u32 = 3;

#[derive(Clone, Debug, PartialEq, Eq)]
struct ToolSurfaceCacheKey {
    mode: crate::ExecutionMode,
    include_base_tools: bool,
    context_surface_revision: u64,
    dynamic_generation: u64,
    plugin_revision: u64,
}

#[derive(Debug, Default)]
struct ToolSurfaceDerived {
    enabled_tools: OnceLock<Arc<Vec<ToolDefinition>>>,
    catalog: OnceLock<Arc<Vec<serde_json::Value>>>,
    rlm_tools_json: OnceLock<Arc<String>>,
}

struct ToolSurfaceArtifact {
    surface: Arc<crate::ToolSurface>,
    preamble: Arc<crate::ModePreamble>,
    derived: ToolSurfaceDerived,
}

#[derive(Clone)]
pub(crate) struct ToolSurfaceHandle(Arc<ToolSurfaceArtifact>);

impl ToolSurfaceHandle {
    fn surface(&self) -> Arc<crate::ToolSurface> {
        Arc::clone(&self.0.surface)
    }

    fn preamble(&self) -> Arc<crate::ModePreamble> {
        Arc::clone(&self.0.preamble)
    }

    fn enabled_tools(&self) -> Arc<Vec<ToolDefinition>> {
        Arc::clone(
            self.0
                .derived
                .enabled_tools
                .get_or_init(|| Arc::new(self.0.surface.enabled_tools())),
        )
    }

    fn catalog(&self) -> Arc<Vec<serde_json::Value>> {
        Arc::clone(self.0.derived.catalog.get_or_init(|| {
            Arc::new(crate::tools::project_tool_catalog(
                self.enabled_tools().iter().cloned(),
            ))
        }))
    }

    fn rlm_tools_json(&self) -> Arc<String> {
        Arc::clone(self.0.derived.rlm_tools_json.get_or_init(|| {
            Arc::new(
                serde_json::to_string(self.catalog().as_ref()).unwrap_or_else(|_| "[]".to_string()),
            )
        }))
    }
}

#[derive(Clone, Default)]
pub struct TurnInjectionBridge {
    queue: std::sync::Arc<std::sync::Mutex<VecDeque<PluginMessage>>>,
}

#[derive(Clone, Debug)]
pub struct InjectedTurnInput {
    pub message: PluginMessage,
}

#[derive(Clone, Default)]
pub struct TurnInputInjectionBridge {
    queue: std::sync::Arc<std::sync::Mutex<VecDeque<InjectedTurnInput>>>,
}

impl TurnInjectionBridge {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn enqueue(&self, messages: Vec<PluginMessage>) -> Result<(), String> {
        let mut queue = self
            .queue
            .lock()
            .map_err(|_| "turn injection bridge poisoned".to_string())?;
        queue.extend(messages);
        Ok(())
    }

    pub fn drain(&self) -> Result<Vec<PluginMessage>, String> {
        let mut queue = self
            .queue
            .lock()
            .map_err(|_| "turn injection bridge poisoned".to_string())?;
        Ok(queue.drain(..).collect())
    }
}

impl TurnInputInjectionBridge {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn enqueue(&self, messages: Vec<InjectedTurnInput>) -> Result<(), String> {
        let mut queue = self
            .queue
            .lock()
            .map_err(|_| "turn injection bridge poisoned".to_string())?;
        queue.extend(messages);
        Ok(())
    }

    pub fn drain(&self) -> Result<Vec<InjectedTurnInput>, String> {
        let mut queue = self
            .queue
            .lock()
            .map_err(|_| "turn input injection bridge poisoned".to_string())?;
        Ok(queue.drain(..).collect())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SessionError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("rlm execution mode is not available in this build or session")]
    RlmUnavailable,
    #[error("rlm runtime exited unexpectedly")]
    RuntimeExited,
    #[error("protocol error: {0}")]
    Protocol(String),
}

const ASYNC_TOOL_HANDLE_KIND: &str = "task";

#[derive(Clone)]
struct AsyncToolHandleEntry {
    tool_name: String,
    state: Arc<StdMutex<AsyncToolHandleState>>,
    done_notify: Arc<Notify>,
    progress_notify: Arc<Notify>,
    cancellation: CancellationToken,
}

struct AsyncToolHandleState {
    join_handle: Option<tokio::task::JoinHandle<()>>,
    buffered_messages: Vec<SandboxMessage>,
    terminal: Option<AsyncToolTerminal>,
}

#[derive(Clone)]
enum AsyncToolTerminal {
    Completed(ToolDispatchOutcome),
    Cancelled,
    Failed(String),
}

struct AsyncToolReply {
    success: bool,
    value: serde_json::Value,
}

impl AsyncToolReply {
    fn success(value: serde_json::Value) -> Self {
        Self {
            success: true,
            value,
        }
    }

    fn error(value: serde_json::Value) -> Self {
        Self {
            success: false,
            value,
        }
    }

    fn into_wire(self) -> String {
        json!({
            "success": self.success,
            "result": serde_json::to_string(&self.value).unwrap_or_else(|_| "null".to_string()),
        })
        .to_string()
    }
}

pub struct Session {
    session_id: String,
    rlm_runtime: Option<LashlangRuntime>,
    last_repl_tools_json: Option<String>,
    services: RuntimeServices,
    include_base_tools: bool,
    context_surface_revision: u64,
    context_tools: Vec<Arc<dyn ToolProvider>>,
    context_prompt_contributions: Vec<PromptContribution>,
    tool_calls: Vec<ToolCallRecord>,
    tool_images: Vec<ToolImage>,
    message_tx: Option<UnboundedSender<SandboxMessage>>,
    scratch_dir: tempfile::TempDir,
    rlm_observe_projection_config: crate::ToolResultProjectionPluginConfig,
    tool_surface_cache: std::sync::Mutex<Vec<(ToolSurfaceCacheKey, ToolSurfaceHandle)>>,
    async_tool_handles: Arc<StdMutex<HashMap<String, AsyncToolHandleEntry>>>,
}

impl Session {
    pub async fn new(
        services: RuntimeServices,
        session_id: &str,
        _execution_mode: crate::ExecutionMode,
    ) -> Result<Self, SessionError> {
        let scratch_dir = tempfile::TempDir::new()?;

        let mut session = Self {
            session_id: session_id.to_string(),
            rlm_runtime: None,
            last_repl_tools_json: None,
            services,
            include_base_tools: true,
            context_surface_revision: 0,
            context_tools: Vec::new(),
            context_prompt_contributions: Vec::new(),
            tool_calls: Vec::new(),
            tool_images: Vec::new(),
            message_tx: None,
            scratch_dir,
            rlm_observe_projection_config: crate::ToolResultProjectionPluginConfig::default(),
            tool_surface_cache: std::sync::Mutex::new(Vec::new()),
            async_tool_handles: Arc::new(StdMutex::new(HashMap::new())),
        };

        let mode_session = Arc::clone(session.plugins().mode_session());
        mode_session
            .initialize_session(&mut session, session_id)
            .await?;

        Ok(session)
    }

    fn runtime(&self) -> Result<&LashlangRuntime, SessionError> {
        self.rlm_runtime
            .as_ref()
            .ok_or(SessionError::RlmUnavailable)
    }

    pub fn supports_repl(&self) -> bool {
        self.rlm_runtime.is_some()
    }

    pub(crate) fn set_rlm_observe_projection_config(
        &mut self,
        config: crate::ToolResultProjectionPluginConfig,
    ) {
        self.rlm_observe_projection_config = config;
    }

    pub(crate) fn mode_extra_prompt_contributions(
        &self,
        mode: crate::ExecutionMode,
    ) -> Vec<PromptContribution> {
        match mode {
            crate::ExecutionMode::Standard => Vec::new(),
            crate::ExecutionMode::Rlm => vec![PromptContribution::execution(
                "Observe Output",
                format!(
                    "Observe output is capped before reinjection using the current RLM observe limit (mode: `{}`, limit: {}, max_lines: {}). If you see a cap/truncation note, narrow the expression and inspect specific fields or slices instead of dumping the whole value.",
                    match self.rlm_observe_projection_config.mode {
                        crate::ToolResultProjectionMode::Bytes => "bytes",
                        crate::ToolResultProjectionMode::Tokens => "tokens",
                    },
                    self.rlm_observe_projection_config.limit,
                    self.rlm_observe_projection_config.max_lines,
                ),
            )],
        }
    }

    pub(crate) async fn start_rlm_runtime(&mut self, session_id: &str) -> Result<(), SessionError> {
        if self.rlm_runtime.is_some() {
            return Ok(());
        }
        let runtime = LashlangRuntime::start()?;
        self.rlm_runtime = Some(runtime);
        self.initialize_tool_surface(session_id).await
    }

    pub fn tools(&self) -> Arc<dyn ToolProvider> {
        if self.include_base_tools && self.context_tools.is_empty() {
            return self.services.plugins.tools();
        }

        let mut providers = Vec::new();
        if self.include_base_tools {
            providers.push(self.services.plugins.tools());
        }
        providers.extend(self.context_tools.iter().cloned());
        Arc::new(crate::tools::CompositeToolProvider::from_providers(
            providers,
        ))
    }

    pub fn plugins(&self) -> &Arc<crate::PluginSession> {
        &self.services.plugins
    }

    pub fn set_context_surface(
        &mut self,
        tool_providers: Vec<Arc<dyn ToolProvider>>,
        prompt_contributions: Vec<PromptContribution>,
        include_base_tools: bool,
    ) {
        self.include_base_tools = include_base_tools;
        self.context_surface_revision = self.context_surface_revision.wrapping_add(1);
        self.context_tools = tool_providers;
        self.context_prompt_contributions = prompt_contributions;
        self.tool_surface_cache
            .lock()
            .expect("tool surface cache lock")
            .clear();
    }

    pub fn context_prompt_contributions(&self) -> &[PromptContribution] {
        &self.context_prompt_contributions
    }

    pub fn history_store(&self) -> Option<Arc<dyn crate::store::RuntimeStore>> {
        self.services.store.clone()
    }

    fn tool_surface_cache_key(&self, mode: crate::ExecutionMode) -> ToolSurfaceCacheKey {
        ToolSurfaceCacheKey {
            mode,
            include_base_tools: self.include_base_tools,
            context_surface_revision: self.context_surface_revision,
            dynamic_generation: self.tools().dynamic_generation().unwrap_or(0),
            plugin_revision: self.plugins().snapshot_revision_fingerprint(),
        }
    }

    fn build_tool_surface_entry(
        &self,
        session_id: &str,
        mode: crate::ExecutionMode,
    ) -> ToolSurfaceHandle {
        let mut tools = self.tools().definitions();
        if self.include_base_tools && mode == self.plugins().execution_mode() {
            tools.extend(self.plugins().mode_native_tools().definitions());
        }
        let fallback_tools = tools.clone();
        let surface = self
            .plugins()
            .resolve_tool_surface(crate::plugin::ToolSurfaceContext {
                session_id: session_id.to_string(),
                mode,
                tools,
            })
            .unwrap_or_else(|err| {
                tracing::warn!("failed to resolve tool surface: {err}");
                crate::ToolSurface::from_tools(fallback_tools)
            });
        let preamble = crate::build_mode_preamble(crate::ModeBuildInput {
            mode,
            tool_surface: surface.clone(),
            extra_prompt_contributions: self.mode_extra_prompt_contributions(mode),
        });
        let surface = Arc::new(surface);
        ToolSurfaceHandle(Arc::new(ToolSurfaceArtifact {
            surface,
            preamble: Arc::new(preamble),
            derived: ToolSurfaceDerived::default(),
        }))
    }

    fn tool_surface_cache_entry(
        &self,
        session_id: &str,
        mode: crate::ExecutionMode,
    ) -> ToolSurfaceHandle {
        let key = self.tool_surface_cache_key(mode);
        let mut cache = self
            .tool_surface_cache
            .lock()
            .expect("tool surface cache lock");
        if let Some((_, entry)) = cache.iter().find(|(entry_key, _)| *entry_key == key) {
            return entry.clone();
        }
        let entry = self.build_tool_surface_entry(session_id, mode);
        cache.push((key, entry.clone()));
        entry
    }

    pub fn tool_surface(&self, session_id: &str, mode: crate::ExecutionMode) -> crate::ToolSurface {
        self.tool_surface_cache_entry(session_id, mode)
            .surface()
            .as_ref()
            .clone()
    }

    pub(crate) fn mode_preamble(
        &self,
        session_id: &str,
        mode: crate::ExecutionMode,
    ) -> Arc<crate::ModePreamble> {
        self.tool_surface_cache_entry(session_id, mode).preamble()
    }

    pub(crate) fn shared_tool_catalog(
        &self,
        session_id: &str,
        mode: crate::ExecutionMode,
    ) -> Arc<Vec<serde_json::Value>> {
        self.tool_surface_cache_entry(session_id, mode).catalog()
    }

    pub fn tool_catalog(
        &self,
        session_id: &str,
        mode: crate::ExecutionMode,
    ) -> Vec<serde_json::Value> {
        self.shared_tool_catalog(session_id, mode).as_ref().clone()
    }

    fn rlm_tools_json(&self, session_id: &str) -> String {
        let entry = self.tool_surface_cache_entry(session_id, crate::ExecutionMode::Rlm);
        let catalog = entry.catalog();
        tracing::debug!(
            session_id,
            tool_count = catalog.len(),
            tool_names = ?catalog
                .iter()
                .filter_map(|tool| tool.get("name").and_then(|value| value.as_str()))
                .collect::<Vec<_>>(),
            "serializing RLM tool catalog"
        );
        entry.rlm_tools_json().as_ref().clone()
    }

    fn async_tool_handle_value(id: &str, tool_name: &str) -> serde_json::Value {
        json!({
            "__handle__": ASYNC_TOOL_HANDLE_KIND,
            "id": id,
            "tool": tool_name,
        })
    }

    fn parse_async_tool_handle(handle: &str) -> Result<(String, Option<String>), String> {
        let value: serde_json::Value =
            serde_json::from_str(handle).map_err(|err| format!("Invalid async handle: {err}"))?;
        let kind = value
            .get("__handle__")
            .and_then(|value| value.as_str())
            .ok_or_else(|| "Invalid async handle: missing `__handle__`".to_string())?;
        if kind != ASYNC_TOOL_HANDLE_KIND {
            return Err(format!("Invalid async handle kind: {kind}"));
        }
        let id = value
            .get("id")
            .and_then(|value| value.as_str())
            .filter(|value| !value.is_empty())
            .ok_or_else(|| "Invalid async handle: missing `id`".to_string())?;
        let tool_name = value
            .get("tool")
            .and_then(|value| value.as_str())
            .map(str::to_string);
        Ok((id.to_string(), tool_name))
    }

    fn async_tool_handle_entry(&self, id: &str) -> Option<AsyncToolHandleEntry> {
        self.async_tool_handles.lock().ok()?.get(id).cloned()
    }

    fn flush_async_tool_messages(&self, entry: &AsyncToolHandleEntry) {
        let Some(message_tx) = self.message_tx.as_ref() else {
            return;
        };
        let pending = {
            let mut state = entry.state.lock().expect("async tool state lock");
            std::mem::take(&mut state.buffered_messages)
        };
        for message in pending {
            let _ = message_tx.send(message);
        }
    }

    fn synthesize_delegate_result_record(
        &mut self,
        call_id: String,
        handle_id: &str,
        outcome: &ToolDispatchOutcome,
    ) {
        let record_result = match outcome.record.result.as_object() {
            Some(_) => outcome.record.result.clone(),
            None => json!({
                "result": outcome.record.result.clone(),
                "status": if outcome.record.success { "completed" } else { "failed" },
            }),
        };
        self.tool_calls.push(ToolCallRecord {
            call_id: Some(call_id),
            tool: "delegate_result".into(),
            args: json!({ "id": handle_id }),
            result: record_result,
            success: outcome.record.success,
            duration_ms: outcome.record.duration_ms,
        });
    }

    fn synthesize_delegate_kill_record(&mut self, call_id: String, handle_id: &str) {
        self.tool_calls.push(ToolCallRecord {
            call_id: Some(call_id),
            tool: "delegate_kill".into(),
            args: json!({ "id": handle_id }),
            result: serde_json::Value::Null,
            success: true,
            duration_ms: 0,
        });
    }

    async fn start_async_tool_call(
        &mut self,
        call_id: String,
        dispatch: Arc<ToolDispatchContext>,
        tool_name: String,
        args: serde_json::Value,
    ) -> String {
        let handle_id = uuid::Uuid::new_v4().to_string();
        let state = Arc::new(StdMutex::new(AsyncToolHandleState {
            join_handle: None,
            buffered_messages: Vec::new(),
            terminal: None,
        }));
        let done_notify = Arc::new(Notify::new());
        let progress_notify = Arc::new(Notify::new());
        let cancellation = CancellationToken::new();
        let entry = AsyncToolHandleEntry {
            tool_name: tool_name.clone(),
            state: Arc::clone(&state),
            done_notify: Arc::clone(&done_notify),
            progress_notify: Arc::clone(&progress_notify),
            cancellation: cancellation.clone(),
        };
        self.async_tool_handles
            .lock()
            .expect("async tool handle map lock")
            .insert(handle_id.clone(), entry);

        let (progress_tx, mut progress_rx) = tokio::sync::mpsc::unbounded_channel();
        let progress_state = Arc::clone(&state);
        let progress_notify_task = Arc::clone(&progress_notify);
        tokio::spawn(async move {
            while let Some(message) = progress_rx.recv().await {
                {
                    let mut guard = progress_state.lock().expect("async tool state lock");
                    guard.buffered_messages.push(message);
                }
                progress_notify_task.notify_waiters();
            }
            progress_notify_task.notify_waiters();
        });

        let task_state = Arc::clone(&state);
        let task_done_notify = Arc::clone(&done_notify);
        let task_progress_notify = Arc::clone(&progress_notify);
        let task_handle_id = handle_id.clone();
        let task_tool_name = tool_name.clone();
        let task_args = args.clone();
        let join_handle = tokio::spawn(async move {
            let tool_context = ToolExecutionContext {
                session_id: dispatch.session_id.clone(),
                host: Arc::clone(&dispatch.host),
                cancellation_token: None,
                async_task_id: None,
            }
            .with_async_task(task_handle_id.clone(), cancellation.clone());
            let outcome = dispatch_tool_call_with_execution_context(
                &dispatch,
                task_tool_name,
                task_args,
                Some(&progress_tx),
                tool_context,
            )
            .await;
            drop(progress_tx);
            let mut guard = task_state.lock().expect("async tool state lock");
            if guard.terminal.is_none() {
                guard.terminal = Some(AsyncToolTerminal::Completed(outcome));
            }
            drop(guard);
            task_progress_notify.notify_waiters();
            task_done_notify.notify_waiters();
        });

        state.lock().expect("async tool state lock").join_handle = Some(join_handle);

        let handle_value = Self::async_tool_handle_value(&handle_id, &tool_name);
        self.tool_calls.push(ToolCallRecord {
            call_id: Some(call_id),
            tool: tool_name,
            args,
            result: handle_value.clone(),
            success: true,
            duration_ms: 0,
        });
        AsyncToolReply::success(handle_value).into_wire()
    }

    async fn await_async_tool_handle(&mut self, call_id: String, handle: String) -> String {
        let (handle_id, hinted_tool_name) = match Self::parse_async_tool_handle(&handle) {
            Ok(parsed) => parsed,
            Err(err) => return AsyncToolReply::error(json!(err)).into_wire(),
        };
        let Some(entry) = self.async_tool_handle_entry(&handle_id) else {
            return AsyncToolReply::error(json!(format!("Unknown async handle: {handle_id}")))
                .into_wire();
        };

        loop {
            self.flush_async_tool_messages(&entry);
            let is_done = {
                let guard = entry.state.lock().expect("async tool state lock");
                guard.terminal.is_some()
            };
            if is_done {
                break;
            }
            tokio::select! {
                _ = entry.done_notify.notified() => {}
                _ = entry.progress_notify.notified() => {}
            }
        }
        self.flush_async_tool_messages(&entry);

        let join_handle = {
            let mut guard = entry.state.lock().expect("async tool state lock");
            guard.join_handle.take()
        };
        if let Some(handle) = join_handle
            && let Err(err) = handle.await
            && !err.is_cancelled()
        {
            let mut guard = entry.state.lock().expect("async tool state lock");
            if guard.terminal.is_none() {
                guard.terminal = Some(AsyncToolTerminal::Failed(format!(
                    "async tool task failed: {err}"
                )));
            }
        }

        let terminal = {
            let guard = entry.state.lock().expect("async tool state lock");
            guard.terminal.clone()
        };

        match terminal {
            Some(AsyncToolTerminal::Completed(outcome)) => {
                self.tool_images.extend(outcome.images.clone());
                let tool_name = hinted_tool_name.unwrap_or_else(|| entry.tool_name.clone());
                if tool_name == "delegate" {
                    self.synthesize_delegate_result_record(call_id, &handle_id, &outcome);
                }
                if outcome.record.success {
                    AsyncToolReply::success(outcome.record.result).into_wire()
                } else {
                    AsyncToolReply::error(outcome.record.result).into_wire()
                }
            }
            Some(AsyncToolTerminal::Cancelled) => {
                AsyncToolReply::error(json!("async task was cancelled")).into_wire()
            }
            Some(AsyncToolTerminal::Failed(err)) => AsyncToolReply::error(json!(err)).into_wire(),
            None => AsyncToolReply::error(json!("async task did not produce a result")).into_wire(),
        }
    }

    async fn cancel_async_tool_handle(&mut self, call_id: String, handle: String) -> String {
        let (handle_id, hinted_tool_name) = match Self::parse_async_tool_handle(&handle) {
            Ok(parsed) => parsed,
            Err(err) => return AsyncToolReply::error(json!(err)).into_wire(),
        };
        let Some(entry) = self.async_tool_handle_entry(&handle_id) else {
            return AsyncToolReply::error(json!(format!("Unknown async handle: {handle_id}")))
                .into_wire();
        };

        entry.cancellation.cancel();
        let _ = tokio::time::timeout(
            std::time::Duration::from_millis(100),
            entry.done_notify.notified(),
        )
        .await;
        let join_handle = {
            let mut guard = entry.state.lock().expect("async tool state lock");
            if guard.terminal.is_none() {
                guard.join_handle.take()
            } else {
                None
            }
        };
        if let Some(handle) = join_handle {
            handle.abort();
            let _ = handle.await;
        }

        {
            let mut guard = entry.state.lock().expect("async tool state lock");
            if guard.terminal.is_none() {
                guard.terminal = Some(AsyncToolTerminal::Cancelled);
            }
        }
        entry.progress_notify.notify_waiters();
        entry.done_notify.notify_waiters();
        self.flush_async_tool_messages(&entry);

        let tool_name = hinted_tool_name.unwrap_or_else(|| entry.tool_name.clone());
        if tool_name == "delegate" {
            self.synthesize_delegate_kill_record(call_id, &handle_id);
        }
        AsyncToolReply::success(serde_json::Value::Null).into_wire()
    }

    async fn cancel_all_async_tool_handles(&self) {
        let entries = self
            .async_tool_handles
            .lock()
            .expect("async tool handle map lock")
            .values()
            .cloned()
            .collect::<Vec<_>>();
        for entry in entries {
            entry.cancellation.cancel();
            let join_handle = {
                let mut guard = entry.state.lock().expect("async tool state lock");
                guard.join_handle.take()
            };
            if let Some(handle) = join_handle {
                handle.abort();
                let _ = handle.await;
            }
        }
        self.async_tool_handles
            .lock()
            .expect("async tool handle map lock")
            .clear();
    }

    pub fn turn_injection_bridge(&self) -> &TurnInjectionBridge {
        &self.services.turn_injection_bridge
    }

    pub fn turn_input_injection_bridge(&self) -> &TurnInputInjectionBridge {
        &self.services.turn_input_injection_bridge
    }

    /// Set the message sender for streaming messages during execution.
    pub fn set_message_sender(&mut self, tx: UnboundedSender<SandboxMessage>) {
        self.message_tx = Some(tx);
    }

    /// Clear the message sender (drops the sender, causing receivers to terminate).
    pub fn clear_message_sender(&mut self) {
        self.message_tx = None;
    }

    /// Execute code in the persistent lashlang RLM.
    ///
    /// `accept_finish` controls how `lashlang::ExecutionOutcome::Finished`
    /// is treated: when true, the captured value flows back through
    /// `ExecResponse::terminal_finish`; when false (today's chat-style
    /// RLM contract), it surfaces as an error telling the model to
    /// terminate via prose instead.
    pub async fn run_code(
        &mut self,
        session_id: &str,
        host: Arc<dyn SessionManager>,
        event_tx: &tokio::sync::mpsc::Sender<SessionEvent>,
        code: &str,
        accept_finish: bool,
    ) -> Result<ExecResponse, SessionError> {
        self.tool_calls.clear();
        self.tool_images.clear();
        let start = std::time::Instant::now();
        let id = uuid::Uuid::new_v4().to_string();

        // Strip markdown separator lines (all dashes + whitespace) the LLM
        // sometimes emits between code blocks.
        let clean_code: String = code
            .lines()
            .filter(|line| {
                let trimmed = line.trim();
                trimmed.is_empty()
                    || trimmed
                        .trim_matches('-')
                        .chars()
                        .any(|c| !c.is_whitespace())
            })
            .collect::<Vec<_>>()
            .join("\n");

        self.runtime()?.send(LashlangRequest::Exec {
            id: id.clone(),
            code: clean_code,
            accept_finish,
        })?;

        // Read messages until we get exec_result.
        // Tool calls are spawned as concurrent tokio tasks so RLM parallel branches
        // can dispatch multiple tools at the same time.
        let dispatch = Arc::new(ToolDispatchContext {
            plugins: Arc::clone(self.plugins()),
            tools: self.tools(),
            surface: self.tool_surface(session_id, crate::ExecutionMode::Rlm),
            host,
            session_id: session_id.to_string(),
            event_tx: event_tx.clone(),
            turn_injection_bridge: self.turn_injection_bridge().clone(),
        });
        let mut tool_handles: Vec<tokio::task::JoinHandle<(ToolCallRecord, Vec<ToolImage>)>> =
            Vec::new();

        // Use block_in_place so tokio knows this thread is blocked and can
        // schedule drain tasks (prompt forwarding, message forwarding) on other threads.
        loop {
            let runtime = self.runtime()?;
            let response = tokio::task::block_in_place(|| runtime.recv())
                .map_err(|_| SessionError::RuntimeExited)?;
            match response {
                LashlangResponse::ToolCall {
                    id: _call_id,
                    name,
                    args,
                    result_tx,
                } => {
                    let tc_num = tool_handles.len();
                    tracing::info!(
                        "PARALLEL: ToolCall #{tc_num} '{name}' received at t+{:.3}s",
                        start.elapsed().as_secs_f64()
                    );
                    let parsed_args: serde_json::Value =
                        serde_json::from_str(&args).unwrap_or(json!({}));
                    let msg_tx = self.message_tx.clone();
                    let run_start = start;
                    let dispatch = Arc::clone(&dispatch);

                    let handle = tokio::spawn(async move {
                        let tool_name_for_log = name.clone();
                        tracing::info!(
                            "PARALLEL: task #{tc_num} '{name}' executing at t+{:.3}s",
                            run_start.elapsed().as_secs_f64()
                        );
                        let outcome =
                            dispatch_tool_call(&dispatch, name, parsed_args, msg_tx.as_ref()).await;

                        // Send the tool result back to the embedded lashlang runtime.
                        let result_json = json!({
                            "success": outcome.record.success,
                            "result": serde_json::to_string(&outcome.record.result)
                                .unwrap_or_else(|_| "null".to_string()),
                        });
                        let _ = result_tx.send(result_json.to_string());
                        tracing::info!(
                            "PARALLEL: task #{tc_num} '{tool_name_for_log}' done at t+{:.3}s",
                            run_start.elapsed().as_secs_f64()
                        );

                        (outcome.record, outcome.images)
                    });
                    tool_handles.push(handle);
                }
                LashlangResponse::StartToolCall {
                    id: call_id,
                    name,
                    args,
                    result_tx,
                } => {
                    let parsed_args: serde_json::Value =
                        serde_json::from_str(&args).unwrap_or(json!({}));
                    let reply = self
                        .start_async_tool_call(call_id, Arc::clone(&dispatch), name, parsed_args)
                        .await;
                    let _ = result_tx.send(reply);
                }
                LashlangResponse::AwaitToolHandle {
                    id: call_id,
                    handle,
                    result_tx,
                } => {
                    let reply = self.await_async_tool_handle(call_id, handle).await;
                    let _ = result_tx.send(reply);
                }
                LashlangResponse::CancelToolHandle {
                    id: call_id,
                    handle,
                    result_tx,
                } => {
                    let reply = self.cancel_async_tool_handle(call_id, handle).await;
                    let _ = result_tx.send(reply);
                }
                LashlangResponse::ExecResult {
                    id: _,
                    output,
                    observations,
                    error,
                    terminal_finish,
                } => {
                    tracing::info!(
                        "PARALLEL: ExecResult received at t+{:.3}s ({} handles)",
                        start.elapsed().as_secs_f64(),
                        tool_handles.len()
                    );
                    // Collect results from all concurrent tool calls.
                    // By the time the runtime sends ExecResult, all tool futures have
                    // resolved, so these awaits are instant.
                    for handle in tool_handles {
                        match handle.await {
                            Ok((record, images)) => {
                                self.tool_calls.push(record);
                                self.tool_images.extend(images);
                            }
                            Err(e) => {
                                self.tool_calls.push(ToolCallRecord {
                                    call_id: None,
                                    tool: "unknown".into(),
                                    args: json!({}),
                                    result: json!({"error": format!("task panicked: {e}")}),
                                    success: false,
                                    duration_ms: 0,
                                });
                            }
                        }
                    }
                    let error = error.filter(|value| !value.trim().is_empty());
                    return Ok(ExecResponse {
                        output,
                        observations,
                        tool_calls: self.tool_calls.clone(),
                        images: std::mem::take(&mut self.tool_images),
                        error,
                        duration_ms: start.elapsed().as_millis() as u64,
                        terminal_finish,
                    });
                }
                LashlangResponse::Ready => {
                    // Unexpected but harmless
                }
                LashlangResponse::SnapshotResult { .. }
                | LashlangResponse::PatchGlobalsResult { .. }
                | LashlangResponse::ResetResult { .. }
                | LashlangResponse::ReconfigureResult { .. }
                | LashlangResponse::CheckCompleteResult { .. } => {
                    return Err(SessionError::Protocol(
                        "unexpected response during exec".to_string(),
                    ));
                }
            }
        }
    }

    /// Check if a code string is syntactically complete for the lashlang RLM.
    pub fn check_complete(&self, code: &str) -> Result<bool, SessionError> {
        tracing::debug!(
            code_preview = %code.chars().take(300).collect::<String>(),
            "checking RLM completeness"
        );
        self.runtime()?.send(LashlangRequest::CheckComplete {
            code: code.to_string(),
        })?;
        let runtime = self.runtime()?;
        let response = tokio::task::block_in_place(|| runtime.recv())
            .map_err(|_| SessionError::RuntimeExited)?;
        match response {
            LashlangResponse::CheckCompleteResult { is_complete } => {
                tracing::debug!(is_complete, "received RLM completeness result");
                Ok(is_complete)
            }
            _ => Ok(false),
        }
    }

    /// Reset the lashlang RLM state and re-register tools.
    pub async fn reset(&mut self) -> Result<(), SessionError> {
        self.cancel_all_async_tool_handles().await;
        let id = uuid::Uuid::new_v4().to_string();
        self.runtime()?
            .send(LashlangRequest::Reset { id: id.clone() })?;

        loop {
            match self.runtime()?.recv()? {
                LashlangResponse::ResetResult { .. } => break,
                _ => continue,
            }
        }

        self.last_repl_tools_json = None;
        self.tool_surface_cache
            .lock()
            .expect("tool surface cache lock")
            .clear();
        Ok(())
    }

    /// Apply an in-place patch to the lashlang `FlowState.globals`
    /// without replacing the rest of the execution snapshot.
    pub async fn apply_rlm_globals_patch(
        &mut self,
        patch: &crate::RlmGlobalsPatchPluginBody,
    ) -> Result<(), SessionError> {
        if !self.supports_repl() || patch.is_empty() {
            return Ok(());
        }
        let id = uuid::Uuid::new_v4().to_string();
        self.runtime()?.send(LashlangRequest::PatchGlobals {
            id: id.clone(),
            set: patch.set.clone(),
            unset: patch.unset.clone(),
        })?;

        loop {
            match self.runtime()?.recv()? {
                LashlangResponse::PatchGlobalsResult { id: got_id, error } if got_id == id => {
                    if let Some(err) = error {
                        return Err(SessionError::Protocol(format!(
                            "failed to patch RLM globals: {err}"
                        )));
                    }
                    break;
                }
                _ => continue,
            }
        }

        Ok(())
    }

    /// Re-register the current tool definitions in the live RLM.
    /// This is intended for turn-boundary runtime reconfiguration.
    pub async fn refresh_tool_surface(&mut self) -> Result<(), SessionError> {
        if !self.supports_repl() {
            return Ok(());
        }

        let tools_json = self.rlm_tools_json(&self.session_id);
        tracing::debug!(
            session_id = self.session_id,
            generation = self.tools().dynamic_generation().unwrap_or(0),
            tools_json_preview = %tools_json.chars().take(400).collect::<String>(),
            "refreshing RLM tool surface"
        );
        if self.last_repl_tools_json.as_deref() == Some(tools_json.as_str()) {
            return Ok(());
        }
        let generation = self.tools().dynamic_generation().unwrap_or(0);
        self.runtime()?.send(LashlangRequest::Reconfigure {
            tools_json: tools_json.clone(),
            generation,
            observe_projection: self.rlm_observe_projection_config.clone(),
        })?;

        loop {
            match self.runtime()?.recv()? {
                LashlangResponse::ReconfigureResult {
                    generation: got_generation,
                    error,
                } => {
                    if got_generation != generation {
                        return Err(SessionError::Protocol(format!(
                            "reconfigure generation mismatch: expected {generation}, got {got_generation}"
                        )));
                    }
                    if let Some(err) = error {
                        return Err(SessionError::Protocol(format!("reconfigure failed: {err}")));
                    }
                    self.last_repl_tools_json = Some(tools_json.clone());
                    break;
                }
                _ => continue,
            }
        }

        Ok(())
    }

    async fn initialize_tool_surface(&mut self, session_id: &str) -> Result<(), SessionError> {
        let tools_json = self.rlm_tools_json(session_id);
        tracing::debug!(
            session_id,
            tools_json_preview = %tools_json.chars().take(400).collect::<String>(),
            "initializing RLM tool surface"
        );
        self.runtime()?.send(LashlangRequest::Init {
            tools_json: tools_json.clone(),
            session_id: session_id.to_string(),
            observe_projection: self.rlm_observe_projection_config.clone(),
        })?;

        match self.runtime()?.recv()? {
            LashlangResponse::Ready => {
                self.last_repl_tools_json = Some(tools_json);
                Ok(())
            }
            other => Err(SessionError::Protocol(format!(
                "expected ready, got: {:?}",
                std::mem::discriminant(&other)
            ))),
        }
    }

    /// Snapshot execution-mode-local state, if any.
    pub async fn snapshot_execution_state(&mut self) -> Result<Option<Vec<u8>>, SessionError> {
        if !self.supports_repl() {
            return Ok(None);
        }
        let id = uuid::Uuid::new_v4().to_string();
        self.runtime()?
            .send(LashlangRequest::Snapshot { id: id.clone() })?;

        let data = loop {
            match self.runtime()?.recv()? {
                LashlangResponse::SnapshotResult { id: _, data } => break data,
                _ => continue,
            }
        };

        // Collect scratch files
        let files = collect_files(self.scratch_dir.path()).unwrap_or_default();

        let combined = json!({
            "version": REPL_SNAPSHOT_VERSION,
            "engine": "lashlang",
            "vars": data,
            "files": files,
        });
        Ok(Some(serde_json::to_vec(&combined).unwrap()))
    }

    /// Restore execution-mode-local state from a snapshot blob.
    pub async fn restore_execution_state(&mut self, data: &[u8]) -> Result<(), SessionError> {
        if !self.supports_repl() {
            return Ok(());
        }
        let parsed: serde_json::Value = serde_json::from_slice(data).unwrap_or(json!({}));

        if parsed.get("version").is_none() || parsed.get("engine").is_none() {
            return Err(SessionError::Protocol(
                "unsupported RLM snapshot format".to_string(),
            ));
        }
        if parsed.get("version").and_then(|v| v.as_u64()) != Some(REPL_SNAPSHOT_VERSION as u64) {
            return Err(SessionError::Protocol(
                "unsupported RLM snapshot version".to_string(),
            ));
        }
        if parsed.get("engine").and_then(|v| v.as_str()) != Some("lashlang") {
            return Err(SessionError::Protocol(
                "unsupported RLM snapshot engine".to_string(),
            ));
        }

        // `vars` is an opaque lashlang snapshot string from the executor thread.
        let vars_str = parsed
            .get("vars")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let id = uuid::Uuid::new_v4().to_string();
        self.runtime()?
            .send(LashlangRequest::Restore { id, data: vars_str })?;

        // Wait for acknowledgment (exec_result with optional restore error)
        loop {
            match self.runtime()?.recv()? {
                LashlangResponse::ExecResult { error, .. } => {
                    if let Some(err) = error {
                        return Err(SessionError::Protocol(format!(
                            "executor restore failed: {err}"
                        )));
                    }
                    break;
                }
                _ => continue,
            }
        }

        // Restore scratch files
        if let Some(files_val) = parsed.get("files")
            && let Ok(files) = serde_json::from_value::<HashMap<String, String>>(files_val.clone())
        {
            // Clear existing scratch contents
            if let Ok(entries) = std::fs::read_dir(self.scratch_dir.path()) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_dir() {
                        let _ = std::fs::remove_dir_all(&path);
                    } else {
                        let _ = std::fs::remove_file(&path);
                    }
                }
            }
            let _ = restore_files(self.scratch_dir.path(), &files);
        }

        Ok(())
    }
}

/// Walk a directory recursively and collect all files as relative_path -> contents.
fn collect_files(root: &Path) -> std::io::Result<HashMap<String, String>> {
    let mut files = HashMap::new();
    walk_dir(root, root, &mut files)?;
    Ok(files)
}

fn walk_dir(root: &Path, dir: &Path, files: &mut HashMap<String, String>) -> std::io::Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            walk_dir(root, &path, files)?;
        } else {
            let rel = path
                .strip_prefix(root)
                .unwrap_or(&path)
                .to_string_lossy()
                .to_string();
            // Best-effort: skip binary files that aren't valid UTF-8
            if let Ok(content) = std::fs::read_to_string(&path) {
                files.insert(rel, content);
            }
        }
    }
    Ok(())
}

/// Recreate files in a directory from a path -> content map.
fn restore_files(root: &Path, files: &HashMap<String, String>) -> std::io::Result<()> {
    for (rel_path, content) in files {
        let full = root.join(rel_path);
        if let Some(parent) = full.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&full, content)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plugin::StaticPluginFactory;
    use crate::tools::UpdatePlanTool;
    use crate::{
        PluginError, PluginHost, PluginSpec, SessionHandle, SessionSnapshot, ToolResult, TurnInput,
    };
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;

    #[cfg(feature = "tool-impls")]
    struct CurrentDirGuard {
        original: std::path::PathBuf,
    }

    #[cfg(feature = "tool-impls")]
    impl CurrentDirGuard {
        fn set(path: &std::path::Path) -> Self {
            let original = std::env::current_dir().expect("current dir");
            std::env::set_current_dir(path).expect("set current dir");
            Self { original }
        }
    }

    #[cfg(feature = "tool-impls")]
    impl Drop for CurrentDirGuard {
        fn drop(&mut self) {
            std::env::set_current_dir(&self.original).expect("restore current dir");
        }
    }

    struct NoopManager;

    #[async_trait::async_trait]
    impl SessionManager for NoopManager {
        async fn snapshot_current(&self) -> Result<SessionSnapshot, PluginError> {
            Err(PluginError::Session("snapshot unavailable".to_string()))
        }

        async fn snapshot_session(
            &self,
            _session_id: &str,
        ) -> Result<SessionSnapshot, PluginError> {
            Err(PluginError::Session("snapshot unavailable".to_string()))
        }

        async fn tool_catalog(
            &self,
            _session_id: &str,
        ) -> Result<Vec<serde_json::Value>, PluginError> {
            Err(PluginError::Session("tool catalog unavailable".to_string()))
        }

        async fn create_session(
            &self,
            _request: crate::SessionCreateRequest,
        ) -> Result<SessionHandle, PluginError> {
            Err(PluginError::Session(
                "session creation unavailable".to_string(),
            ))
        }

        async fn close_session(&self, _session_id: &str) -> Result<(), PluginError> {
            Err(PluginError::Session(
                "session close unavailable".to_string(),
            ))
        }

        async fn start_turn_stream(
            &self,
            _session_id: &str,
            _input: TurnInput,
        ) -> Result<crate::plugin::SessionTurnHandle, PluginError> {
            Err(PluginError::Session(
                "turn streaming unavailable".to_string(),
            ))
        }

        async fn await_turn(&self, _turn_id: &str) -> Result<crate::AssembledTurn, PluginError> {
            Err(PluginError::Session("await turn unavailable".to_string()))
        }

        async fn cancel_turn(&self, _turn_id: &str) -> Result<(), PluginError> {
            Err(PluginError::Session("cancel turn unavailable".to_string()))
        }
    }

    #[derive(Clone, Default)]
    struct AsyncToolState {
        cancelled: Arc<AtomicBool>,
    }

    struct AsyncTestToolProvider {
        state: AsyncToolState,
    }

    #[async_trait::async_trait]
    impl crate::ToolProvider for AsyncTestToolProvider {
        fn definitions(&self) -> Vec<ToolDefinition> {
            vec![
                ToolDefinition {
                    name: "echo_async".into(),
                    description: "Return an echo result after a short delay.".into(),
                    params: vec![crate::ToolParam::typed("text", "str")],
                    returns: "dict".into(),
                    examples: vec![],
                    enabled: true,
                    injected: true,
                    input_schema_override: None,
                    output_schema_override: None,
                },
                ToolDefinition {
                    name: "wait_for_cancel".into(),
                    description: "Wait until the async task is cancelled.".into(),
                    params: vec![],
                    returns: "dict".into(),
                    examples: vec![],
                    enabled: true,
                    injected: true,
                    input_schema_override: None,
                    output_schema_override: None,
                },
            ]
        }

        async fn execute(&self, _name: &str, _args: &serde_json::Value) -> ToolResult {
            ToolResult::err_fmt("async test tool requires session context")
        }

        async fn execute_streaming_with_context(
            &self,
            name: &str,
            args: &serde_json::Value,
            context: &ToolExecutionContext,
            _progress: Option<&crate::ProgressSender>,
        ) -> ToolResult {
            match name {
                "echo_async" => {
                    let text = args
                        .get("text")
                        .and_then(|value| value.as_str())
                        .unwrap_or_default();
                    tokio::time::sleep(Duration::from_millis(10)).await;
                    ToolResult::ok(json!({ "echo": text }))
                }
                "wait_for_cancel" => {
                    let Some(token) = context.cancellation_token.clone() else {
                        return ToolResult::err_fmt("missing cancellation token");
                    };
                    token.cancelled().await;
                    self.state.cancelled.store(true, Ordering::SeqCst);
                    ToolResult::ok(json!({ "cancelled": true }))
                }
                _ => ToolResult::err_fmt(format_args!("Unknown tool: {name}")),
            }
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn run_code_reports_finish_as_an_error_after_tool_call() {
        let tools: Arc<dyn crate::ToolProvider> = Arc::new(UpdatePlanTool::new());
        let plugin_host = PluginHost::new(vec![Arc::new(StaticPluginFactory::new(
            "test_tools",
            PluginSpec::new().with_tool_provider(Arc::clone(&tools)),
        ))]);
        let plugin_session = plugin_host
            .build_session(
                "root",
                crate::ExecutionMode::Rlm,
                crate::ContextApproach::default(),
                None,
            )
            .expect("plugin session");
        let mut session = Session::new(
            crate::RuntimeServices::new(plugin_session),
            "root",
            crate::ExecutionMode::Rlm,
        )
        .await
        .expect("session");

        let (event_tx, _event_rx) = tokio::sync::mpsc::channel(16);
        let manager: Arc<dyn SessionManager> = Arc::new(NoopManager);

        let response = session
            .run_code(
                "root",
                manager,
                &event_tx,
                r#"
call update_plan {
  explanation: "done",
  plan: [{ step: "ship it", status: "completed" }]
}
finish "ok"
"#,
                false,
            )
            .await
            .expect("exec response");

        assert_eq!(
            response.error,
            Some(
                "This lashlang step tried to terminate the task directly. End the task by replying in plain prose instead of calling `execute_lashlang` again.".to_string()
            )
        );
        assert_eq!(response.tool_calls.len(), 1);
        assert_eq!(response.tool_calls[0].tool, "update_plan");
        assert!(response.tool_calls[0].success);
    }

    #[cfg(feature = "tool-impls")]
    #[tokio::test(flavor = "multi_thread")]
    async fn rlm_run_code_can_call_common_repo_tools_without_model() {
        let temp = tempfile::tempdir().expect("tempdir");
        std::fs::write(temp.path().join("alpha.txt"), "hello\n").expect("write alpha");
        std::fs::write(temp.path().join("beta.rs"), "fn main() {}\n").expect("write beta");
        let _cwd = CurrentDirGuard::set(temp.path());

        let plugin_host = PluginHost::new(vec![
            Arc::new(StaticPluginFactory::new(
                "ls",
                PluginSpec::new().with_tool_provider(Arc::new(crate::tools::Ls)),
            )),
            Arc::new(StaticPluginFactory::new(
                "grep",
                PluginSpec::new().with_tool_provider(Arc::new(crate::tools::Grep::new())),
            )),
        ]);
        let plugin_session = plugin_host
            .build_session(
                "root",
                crate::ExecutionMode::Rlm,
                crate::ContextApproach::default(),
                None,
            )
            .expect("plugin session");
        let mut session = Session::new(
            crate::RuntimeServices::new(plugin_session),
            "root",
            crate::ExecutionMode::Rlm,
        )
        .await
        .expect("session");

        let (event_tx, _event_rx) = tokio::sync::mpsc::channel(16);
        let manager: Arc<dyn SessionManager> = Arc::new(NoopManager);

        let response = session
            .run_code(
                "root",
                manager,
                &event_tx,
                r#"
files = call ls { path: "." }
match = call grep { query: "fn main" }
observe files
observe match
"#,
                false,
            )
            .await
            .expect("exec response");

        assert!(
            response.error.is_none(),
            "unexpected rlm error: {:?}",
            response.error
        );
        assert_eq!(response.tool_calls.len(), 2);
        assert_eq!(response.tool_calls[0].tool, "ls");
        assert!(response.tool_calls[0].success);
        assert_eq!(response.tool_calls[1].tool, "grep");
        assert!(response.tool_calls[1].success);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn rlm_run_code_can_start_and_await_async_tool_handles() {
        let async_tools: Arc<dyn crate::ToolProvider> = Arc::new(AsyncTestToolProvider {
            state: AsyncToolState::default(),
        });
        let plugin_host = PluginHost::new(vec![Arc::new(StaticPluginFactory::new(
            "async_tools",
            PluginSpec::new().with_tool_provider(Arc::clone(&async_tools)),
        ))]);
        let plugin_session = plugin_host
            .build_session(
                "root",
                crate::ExecutionMode::Rlm,
                crate::ContextApproach::default(),
                None,
            )
            .expect("plugin session");
        let mut session = Session::new(
            crate::RuntimeServices::new(plugin_session),
            "root",
            crate::ExecutionMode::Rlm,
        )
        .await
        .expect("session");

        let (event_tx, _event_rx) = tokio::sync::mpsc::channel(16);
        let manager: Arc<dyn SessionManager> = Arc::new(NoopManager);

        let response = session
            .run_code(
                "root",
                manager,
                &event_tx,
                r#"
handle = start call echo_async { text: "hello" }
result = await handle
observe result
"#,
                false,
            )
            .await
            .expect("exec response");

        assert!(
            response.error.is_none(),
            "unexpected rlm error: {:?}",
            response.error
        );
        assert_eq!(response.tool_calls.len(), 1);
        assert_eq!(response.tool_calls[0].tool, "echo_async");
        assert_eq!(
            response.tool_calls[0]
                .result
                .get("__handle__")
                .and_then(|value| value.as_str()),
            Some(ASYNC_TOOL_HANDLE_KIND)
        );
        assert!(
            response
                .observations
                .iter()
                .any(|text| text.contains("\"value\":{\"echo\":\"hello\"}")),
            "observations were: {:?}",
            response.observations
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn rlm_run_code_can_cancel_async_tool_handles() {
        let async_state = AsyncToolState::default();
        let async_tools: Arc<dyn crate::ToolProvider> = Arc::new(AsyncTestToolProvider {
            state: async_state.clone(),
        });
        let plugin_host = PluginHost::new(vec![Arc::new(StaticPluginFactory::new(
            "async_tools",
            PluginSpec::new().with_tool_provider(Arc::clone(&async_tools)),
        ))]);
        let plugin_session = plugin_host
            .build_session(
                "root",
                crate::ExecutionMode::Rlm,
                crate::ContextApproach::default(),
                None,
            )
            .expect("plugin session");
        let mut session = Session::new(
            crate::RuntimeServices::new(plugin_session),
            "root",
            crate::ExecutionMode::Rlm,
        )
        .await
        .expect("session");

        let (event_tx, _event_rx) = tokio::sync::mpsc::channel(16);
        let manager: Arc<dyn SessionManager> = Arc::new(NoopManager);

        let response = session
            .run_code(
                "root",
                manager,
                &event_tx,
                r#"
handle = start call wait_for_cancel {}
cancel handle
observe "cancelled"
"#,
                false,
            )
            .await
            .expect("exec response");

        assert!(
            response.error.is_none(),
            "unexpected rlm error: {:?}",
            response.error
        );
        assert_eq!(response.tool_calls.len(), 1);
        assert_eq!(response.tool_calls[0].tool, "wait_for_cancel");
        assert!(async_state.cancelled.load(Ordering::SeqCst));
        assert!(
            response
                .observations
                .iter()
                .any(|text| text.contains("cancelled")),
            "observations were: {:?}",
            response.observations
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn rlm_run_code_caps_observe_output_via_mode_plugin_config() {
        let plugin_host = PluginHost::new(vec![Arc::new(crate::BuiltinRlmModePluginFactory::new(
            crate::RlmModePluginConfig {
                observe_projection: crate::ToolResultProjectionPluginConfig {
                    mode: crate::ToolResultProjectionMode::Bytes,
                    limit: 24,
                    max_lines: 2,
                },
            },
        ))]);
        let plugin_session = plugin_host
            .build_session(
                "root",
                crate::ExecutionMode::Rlm,
                crate::ContextApproach::default(),
                None,
            )
            .expect("plugin session");
        let mut session = Session::new(
            crate::RuntimeServices::new(plugin_session),
            "root",
            crate::ExecutionMode::Rlm,
        )
        .await
        .expect("session");

        let (event_tx, _event_rx) = tokio::sync::mpsc::channel(16);
        let manager: Arc<dyn SessionManager> = Arc::new(NoopManager);

        let response = session
            .run_code(
                "root",
                manager,
                &event_tx,
                r#"
observe "abcdefghijklmnopqrstuvwxyz0123456789"
"#,
                false,
            )
            .await
            .expect("exec response");

        assert!(
            response.error.is_none(),
            "unexpected rlm error: {:?}",
            response.error
        );
        let observation = response
            .observations
            .first()
            .expect("expected one observation");
        assert!(observation.contains("truncated"));
        assert!(observation.contains("observe output was capped at 24 bytes and 2 lines max"));
        assert!(observation.contains("Use a narrower `observe` expression"));
    }
}
