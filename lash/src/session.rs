use std::collections::{HashMap, VecDeque};
use std::path::Path;
use std::sync::{Arc, Mutex as StdMutex, OnceLock};

use futures_util::stream::{FuturesUnordered, StreamExt};
use serde_json::json;
use tokio::sync::{Notify, mpsc::UnboundedSender};
use tokio_util::sync::CancellationToken;

use crate::embedded::{LashlangRequest, LashlangResponse, LashlangRuntime, LashlangToolReply};
use crate::tool_dispatch::{
    ParallelToolCallSpec, ToolDispatchContext, ToolDispatchOutcome, dispatch_parallel_tool_call,
    dispatch_tool_call_with_execution_context,
};
use crate::{
    ExecResponse, PluginMessage, PromptContribution, RuntimeServices, RuntimeSessionHost,
    SandboxMessage, SessionEvent, ToolCallRecord, ToolExecutionContext, ToolImage, ToolProvider,
};

const REPL_SNAPSHOT_VERSION: u32 = 3;

type RlmToolTaskOutput = Vec<(ToolCallRecord, Vec<ToolImage>)>;
type RlmToolTaskHandle = tokio::task::JoinHandle<RlmToolTaskOutput>;

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

    fn catalog(&self) -> Arc<Vec<serde_json::Value>> {
        Arc::clone(self.0.derived.catalog.get_or_init(|| {
            Arc::new(crate::tools::project_tool_catalog(
                self.0.surface.discoverable_tools_iter().cloned(),
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
    state: Arc<StdMutex<AsyncToolHandleState>>,
    done_notify: Arc<Notify>,
    progress_notify: Arc<Notify>,
    cancellation: CancellationToken,
    metadata: AsyncToolHandleMetadata,
}

#[derive(Clone)]
struct AsyncToolHandleMetadata {
    tool_name: String,
    namespace: AsyncToolHandleNamespace,
    identifier: String,
}

#[derive(Clone, PartialEq, Eq)]
enum AsyncToolHandleNamespace {
    Subagent,
    Tool,
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
    images: Vec<ToolImage>,
}

impl AsyncToolReply {
    fn success(value: serde_json::Value) -> Self {
        Self {
            success: true,
            value,
            images: Vec::new(),
        }
    }

    fn success_with_images(value: serde_json::Value, images: Vec<ToolImage>) -> Self {
        Self {
            success: true,
            value,
            images,
        }
    }

    fn error(value: serde_json::Value) -> Self {
        Self {
            success: false,
            value,
            images: Vec::new(),
        }
    }

    fn into_lashlang_reply(self) -> LashlangToolReply {
        if self.success {
            LashlangToolReply::success_with_images(self.value, self.images)
        } else {
            LashlangToolReply::error(self.value)
        }
    }
}

pub struct Session {
    session_id: String,
    execution_mode: crate::ExecutionMode,
    rlm_runtime: Option<LashlangRuntime>,
    last_repl_tools_json: Option<Arc<String>>,
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
    /// Memoizes the rendered system prompt across turns. Most consecutive
    /// turns reuse the same template + context surface, so the cache hits
    /// and we skip the section/Vec-join work in
    /// `lash_sansio::PromptTemplate::render`.
    prompt_cache: Arc<lash_sansio::PromptCache>,
    async_tool_handles: Arc<StdMutex<HashMap<String, AsyncToolHandleEntry>>>,
    /// Tracks whether the lashlang VM (and the scratch dir it owns) has
    /// changed since the last successful `snapshot_execution_state` call.
    /// Bumped to `true` whenever a `LashlangRequest` that can mutate state
    /// is sent (Exec, Reset, PatchGlobals, Reconfigure, Init, Restore).
    /// Callers gate the per-iteration snapshot on
    /// [`Session::execution_state_dirty`] so iterations that don't run any
    /// lashlang code skip the round-trip + JSON serialization.
    lashlang_state_dirty: bool,
}

impl Session {
    pub async fn new(
        services: RuntimeServices,
        session_id: &str,
        execution_mode: crate::ExecutionMode,
    ) -> Result<Self, SessionError> {
        let scratch_dir = tempfile::TempDir::new()?;

        let mut session = Self {
            session_id: session_id.to_string(),
            execution_mode,
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
            prompt_cache: Arc::new(lash_sansio::PromptCache::new()),
            async_tool_handles: Arc::new(StdMutex::new(HashMap::new())),
            lashlang_state_dirty: true,
        };

        let mode_session = Arc::clone(session.plugins().mode_session());
        mode_session
            .initialize_session(crate::plugin::ModeSessionContext::new(
                &mut session,
                session_id,
            ))
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

    pub(crate) fn set_execution_output_projection(
        &mut self,
        config: crate::ToolResultProjectionPluginConfig,
    ) {
        self.rlm_observe_projection_config = config;
    }

    pub(crate) fn mode_extra_prompt_contributions(
        &self,
        _mode: &crate::ExecutionMode,
    ) -> Vec<PromptContribution> {
        // Mode-specific prompt contributions are owned by the mode
        // plugins (`lash-mode-standard`, `lash-mode-rlm`) via their
        // `reg.prompt().contribute(...)` hooks. Nothing to add here.
        Vec::new()
    }

    pub(crate) async fn start_lashlang_runtime(
        &mut self,
        session_id: &str,
    ) -> Result<(), SessionError> {
        if self.rlm_runtime.is_some() {
            return Ok(());
        }
        let runtime = LashlangRuntime::start_with_attachment_store(Arc::clone(
            &self.services.attachment_store,
        ))?;
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
        let tool_providers_unchanged = self.context_tools.len() == tool_providers.len()
            && self
                .context_tools
                .iter()
                .zip(&tool_providers)
                .all(|(current, next)| Arc::ptr_eq(current, next));
        if self.include_base_tools == include_base_tools
            && self.context_prompt_contributions == prompt_contributions
            && tool_providers_unchanged
        {
            return;
        }
        self.include_base_tools = include_base_tools;
        self.context_surface_revision = self.context_surface_revision.wrapping_add(1);
        self.context_tools = tool_providers;
        self.context_prompt_contributions = prompt_contributions;
        self.tool_surface_cache
            .lock()
            .expect("tool surface cache lock")
            .clear();
    }

    pub fn prompt_cache(&self) -> Arc<lash_sansio::PromptCache> {
        Arc::clone(&self.prompt_cache)
    }

    pub fn context_prompt_contributions(&self) -> &[PromptContribution] {
        &self.context_prompt_contributions
    }

    pub fn history_store(&self) -> Option<Arc<dyn crate::store::RuntimePersistence>> {
        self.services.store.clone()
    }

    fn tool_surface_cache_key(&self, mode: &crate::ExecutionMode) -> ToolSurfaceCacheKey {
        ToolSurfaceCacheKey {
            mode: mode.clone(),
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
            tools.extend(self.plugins().mode_native_tool_definitions());
        }
        let fallback_tools = tools.clone();
        let surface = Arc::new(
            self.plugins()
                .resolve_tool_surface(crate::plugin::ToolSurfaceContext {
                    session_id: session_id.to_string(),
                    mode: mode.clone(),
                    tools,
                    tool_access: self.plugins().tool_access().clone(),
                    subagent: self.plugins().subagent_authority().cloned(),
                })
                .unwrap_or_else(|err| {
                    tracing::warn!("failed to resolve tool surface: {err}");
                    crate::ToolSurface::from_tools(fallback_tools, mode.clone())
                }),
        );
        let input = crate::ModeBuildInput {
            mode: mode.clone(),
            tool_surface: Arc::clone(&surface),
            extra_prompt_contributions: self.mode_extra_prompt_contributions(&mode),
        };
        let driver = self.plugins().mode_protocol_driver().unwrap_or_else(|| {
            panic!(
                "no protocol driver registered for execution mode `{}` — \
                 did you forget to register the mode plugin (e.g. \
                 `lash_mode_standard::BuiltinStandardModePluginFactory` or \
                 `lash_mode_rlm::BuiltinRlmModePluginFactory`)?",
                mode.plugin_id()
            )
        });
        assert_eq!(
            driver.mode_id(),
            mode.plugin_id(),
            "protocol driver `{}` does not match session mode `{}`",
            driver.mode_id(),
            mode.plugin_id(),
        );
        let preamble = driver.build_preamble(input);
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
        let key = self.tool_surface_cache_key(&mode);
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

    pub fn tool_surface(
        &self,
        session_id: &str,
        mode: crate::ExecutionMode,
    ) -> Arc<crate::ToolSurface> {
        self.tool_surface_cache_entry(session_id, mode).surface()
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

    fn rlm_tools_json(&self, session_id: &str) -> Arc<String> {
        let entry = self.tool_surface_cache_entry(session_id, self.execution_mode.clone());
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
        entry.rlm_tools_json()
    }

    fn async_tool_handle_value(id: &str, tool_name: &str) -> serde_json::Value {
        json!({
            "__handle__": ASYNC_TOOL_HANDLE_KIND,
            "id": id,
            "tool": tool_name,
        })
    }

    fn async_tool_handle_metadata(
        id: &str,
        tool_name: &str,
        args: &serde_json::Value,
    ) -> AsyncToolHandleMetadata {
        if tool_name == "spawn_agent"
            && let Some(agent_name) = args.get("agent_name").and_then(|value| value.as_str())
            && let Some(normalized) = Self::normalize_async_subagent_name(agent_name)
        {
            return AsyncToolHandleMetadata {
                tool_name: tool_name.to_string(),
                namespace: AsyncToolHandleNamespace::Subagent,
                identifier: normalized,
            };
        }
        AsyncToolHandleMetadata {
            tool_name: tool_name.to_string(),
            namespace: AsyncToolHandleNamespace::Tool,
            identifier: id.to_string(),
        }
    }

    fn normalize_async_subagent_name(agent_name: &str) -> Option<String> {
        let mut out = String::new();
        let mut last_was_sep = false;
        for ch in agent_name.chars().flat_map(char::to_lowercase) {
            if ch.is_ascii_alphanumeric() {
                out.push(ch);
                last_was_sep = false;
            } else if !last_was_sep && !out.is_empty() {
                out.push('_');
                last_was_sep = true;
            }
        }
        while out.ends_with('_') {
            out.pop();
        }
        (!out.is_empty()).then_some(out)
    }

    fn parse_async_tool_handle(
        handle: &serde_json::Value,
    ) -> Result<(String, Option<String>), String> {
        let kind = handle
            .get("__handle__")
            .and_then(|value| value.as_str())
            .ok_or_else(|| "Invalid async handle: missing `__handle__`".to_string())?;
        if kind != ASYNC_TOOL_HANDLE_KIND {
            return Err(format!("Invalid async handle kind: {kind}"));
        }
        let id = handle
            .get("id")
            .and_then(|value| value.as_str())
            .filter(|value| !value.is_empty())
            .ok_or_else(|| "Invalid async handle: missing `id`".to_string())?;
        let tool_name = handle
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

    async fn start_async_tool_call(
        &mut self,
        call_id: String,
        dispatch: Arc<ToolDispatchContext>,
        tool_name: String,
        args: serde_json::Value,
    ) -> LashlangToolReply {
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
            state: Arc::clone(&state),
            done_notify: Arc::clone(&done_notify),
            progress_notify: Arc::clone(&progress_notify),
            cancellation: cancellation.clone(),
            metadata: Self::async_tool_handle_metadata(&handle_id, &tool_name, &args),
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
        AsyncToolReply::success(handle_value).into_lashlang_reply()
    }

    fn list_async_handles(&self) -> LashlangToolReply {
        let entries = self
            .async_tool_handles
            .lock()
            .expect("async tool handle map lock")
            .iter()
            .filter_map(|(id, entry)| {
                let is_terminal = entry
                    .state
                    .lock()
                    .expect("async tool state lock")
                    .terminal
                    .is_some();
                (!is_terminal).then(|| (id.clone(), entry.metadata.clone()))
            })
            .collect::<Vec<_>>();

        let mut subagent = serde_json::Map::new();
        let mut tool = serde_json::Map::new();
        for (id, metadata) in entries {
            let value = Self::async_tool_handle_value(&id, &metadata.tool_name);
            match metadata.namespace {
                AsyncToolHandleNamespace::Subagent => {
                    subagent.insert(metadata.identifier, value);
                }
                AsyncToolHandleNamespace::Tool => {
                    tool.insert(metadata.identifier, value);
                }
            }
        }
        AsyncToolReply::success(json!({
            "subagent": subagent,
            "tool": tool,
        }))
        .into_lashlang_reply()
    }

    async fn await_async_tool_handle(
        &mut self,
        _call_id: String,
        handle: serde_json::Value,
    ) -> LashlangToolReply {
        let (handle_id, _hinted_tool_name) = match Self::parse_async_tool_handle(&handle) {
            Ok(parsed) => parsed,
            Err(err) => return AsyncToolReply::error(json!(err)).into_lashlang_reply(),
        };
        let Some(entry) = self.async_tool_handle_entry(&handle_id) else {
            return AsyncToolReply::error(json!(format!("Unknown async handle: {handle_id}")))
                .into_lashlang_reply();
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
                if outcome.record.success {
                    AsyncToolReply::success_with_images(outcome.record.result, outcome.images)
                        .into_lashlang_reply()
                } else {
                    AsyncToolReply::error(outcome.record.result).into_lashlang_reply()
                }
            }
            Some(AsyncToolTerminal::Cancelled) => {
                AsyncToolReply::error(json!("async task was cancelled")).into_lashlang_reply()
            }
            Some(AsyncToolTerminal::Failed(err)) => {
                AsyncToolReply::error(json!(err)).into_lashlang_reply()
            }
            None => AsyncToolReply::error(json!("async task did not produce a result"))
                .into_lashlang_reply(),
        }
    }

    async fn cancel_async_tool_handle(
        &mut self,
        _call_id: String,
        handle: serde_json::Value,
    ) -> LashlangToolReply {
        let (handle_id, _hinted_tool_name) = match Self::parse_async_tool_handle(&handle) {
            Ok(parsed) => parsed,
            Err(err) => return AsyncToolReply::error(json!(err)).into_lashlang_reply(),
        };
        let Some(entry) = self.async_tool_handle_entry(&handle_id) else {
            return AsyncToolReply::error(json!(format!("Unknown async handle: {handle_id}")))
                .into_lashlang_reply();
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

        AsyncToolReply::success(serde_json::Value::Null).into_lashlang_reply()
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
        host: Arc<dyn RuntimeSessionHost>,
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
        self.lashlang_state_dirty = true;

        // Read messages until we get exec_result.
        // Tool calls are spawned as concurrent tokio tasks so RLM parallel branches
        // can dispatch multiple tools at the same time.
        let dispatch = Arc::new(ToolDispatchContext {
            plugins: Arc::clone(self.plugins()),
            tools: self.tools(),
            surface: self.tool_surface(session_id, self.execution_mode.clone()),
            host,
            session_id: session_id.to_string(),
            event_tx: event_tx.clone(),
            turn_injection_bridge: self.turn_injection_bridge().clone(),
            attachment_store: Arc::clone(&self.services.attachment_store),
        });
        let mut tool_handles: Vec<RlmToolTaskHandle> = Vec::new();
        let mut tool_call_count = 0usize;

        // Use block_in_place so tokio knows this thread is blocked and can
        // schedule drain tasks (prompt forwarding, message forwarding) on other threads.
        loop {
            let runtime = self.runtime()?;
            let response = tokio::task::block_in_place(|| runtime.recv())
                .map_err(|_| SessionError::RuntimeExited)?;
            match response {
                LashlangResponse::ToolCall {
                    id: call_id,
                    name,
                    args,
                    result_tx,
                } => {
                    if name == "list_async_handles" {
                        let reply = self.list_async_handles();
                        let _ = result_tx.send(reply);
                        continue;
                    }
                    let tc_num = tool_call_count;
                    tool_call_count += 1;
                    tracing::info!(
                        "PARALLEL: ToolCall #{tc_num} '{name}' received at t+{:.3}s",
                        start.elapsed().as_secs_f64()
                    );
                    let msg_tx = self.message_tx.clone();
                    let run_start = start;
                    let dispatch = Arc::clone(&dispatch);

                    let handle = tokio::spawn(async move {
                        let tool_name_for_log = name.clone();
                        tracing::info!(
                            "PARALLEL: task #{tc_num} '{name}' executing at t+{:.3}s",
                            run_start.elapsed().as_secs_f64()
                        );
                        let mut outcome = dispatch_parallel_tool_call(
                            Arc::clone(&dispatch),
                            ParallelToolCallSpec {
                                index: tc_num,
                                tool_name: name,
                                args,
                            },
                            msg_tx,
                        )
                        .await;
                        outcome.record.call_id = Some(call_id);

                        // Send the tool result back to the embedded lashlang runtime.
                        let reply = if outcome.record.success {
                            LashlangToolReply::success_with_images(
                                outcome.record.result.clone(),
                                outcome.images.clone(),
                            )
                        } else {
                            LashlangToolReply::error(outcome.record.result.clone())
                        };
                        let _ = result_tx.send(reply);
                        tracing::info!(
                            "PARALLEL: task #{tc_num} '{tool_name_for_log}' done at t+{:.3}s",
                            run_start.elapsed().as_secs_f64()
                        );

                        vec![(outcome.record, outcome.images)]
                    });
                    tool_handles.push(handle);
                }
                LashlangResponse::ToolBatchCall { calls, result_tx } => {
                    let call_count = calls.len();
                    let base_index = tool_call_count;
                    tool_call_count += call_count;
                    tracing::info!(
                        "PARALLEL: ToolBatchCall {} calls received at t+{:.3}s",
                        call_count,
                        start.elapsed().as_secs_f64()
                    );
                    let msg_tx = self.message_tx.clone();
                    let run_start = start;
                    let dispatch = Arc::clone(&dispatch);

                    let handle = tokio::spawn(async move {
                        let mut pending = FuturesUnordered::new();
                        for (offset, call) in calls.into_iter().enumerate() {
                            let dispatch = Arc::clone(&dispatch);
                            let msg_tx = msg_tx.clone();
                            pending.push(async move {
                                let tc_num = base_index + offset;
                                let call_id = call.id;
                                let tool_name_for_log = call.name.clone();
                                tracing::info!(
                                    "PARALLEL: batch task #{tc_num} '{}' executing at t+{:.3}s",
                                    call.name,
                                    run_start.elapsed().as_secs_f64()
                                );
                                let mut outcome = dispatch_parallel_tool_call(
                                    Arc::clone(&dispatch),
                                    ParallelToolCallSpec {
                                        index: tc_num,
                                        tool_name: call.name,
                                        args: call.args,
                                    },
                                    msg_tx,
                                )
                                .await;
                                outcome.record.call_id = Some(call_id);
                                tracing::info!(
                                    "PARALLEL: batch task #{tc_num} '{tool_name_for_log}' done at t+{:.3}s",
                                    run_start.elapsed().as_secs_f64()
                                );
                                (offset, outcome)
                            });
                        }

                        let mut outcomes = Vec::with_capacity(call_count);
                        while let Some(outcome) = pending.next().await {
                            outcomes.push(outcome);
                        }
                        outcomes.sort_by_key(|(offset, _)| *offset);

                        let replies = outcomes
                            .iter()
                            .map(|(_, outcome)| {
                                if outcome.record.success {
                                    LashlangToolReply::success_with_images(
                                        outcome.record.result.clone(),
                                        outcome.images.clone(),
                                    )
                                } else {
                                    LashlangToolReply::error(outcome.record.result.clone())
                                }
                            })
                            .collect::<Vec<_>>();
                        let _ = result_tx.send(replies);

                        outcomes
                            .into_iter()
                            .map(|(_, outcome)| (outcome.record, outcome.images))
                            .collect::<Vec<_>>()
                    });
                    tool_handles.push(handle);
                }
                LashlangResponse::StartToolCall {
                    id: call_id,
                    name,
                    args,
                    result_tx,
                } => {
                    let reply = self
                        .start_async_tool_call(call_id, Arc::clone(&dispatch), name, args)
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
                    observation_truncation,
                    images,
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
                            Ok(results) => {
                                for (record, images) in results {
                                    self.tool_calls.push(record);
                                    self.tool_images.extend(images);
                                }
                            }
                            Err(e) => {
                                self.tool_calls.push(ToolCallRecord {
                                    call_id: Some(uuid::Uuid::new_v4().to_string()),
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
                        observation_truncation,
                        tool_calls: self.tool_calls.clone(),
                        images: std::mem::take(&mut self.tool_images),
                        printed_images: images,
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
        self.lashlang_state_dirty = true;

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
    pub async fn apply_mode_globals_patch(
        &mut self,
        patch: &lash_rlm_types::RlmGlobalsPatchPluginBody,
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
        self.lashlang_state_dirty = true;

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
        // Fast-path: same Arc as last time → definitely identical.
        // Fallback: same contents → also skip. The tool-surface cache
        // reuses the inner Arc across turns when inputs are stable, so
        // the fast path fires most often.
        if let Some(last) = self.last_repl_tools_json.as_ref()
            && (Arc::ptr_eq(last, &tools_json) || last.as_str() == tools_json.as_str())
        {
            return Ok(());
        }
        let generation = self.tools().dynamic_generation().unwrap_or(0);
        self.runtime()?.send(LashlangRequest::Reconfigure {
            tools_json: (*tools_json).clone(),
            generation,
            observe_projection: self.rlm_observe_projection_config.clone(),
        })?;
        self.lashlang_state_dirty = true;

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
                    self.last_repl_tools_json = Some(tools_json);
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
            tools_json: (*tools_json).clone(),
            session_id: session_id.to_string(),
            observe_projection: self.rlm_observe_projection_config.clone(),
        })?;
        self.lashlang_state_dirty = true;

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

    /// Returns `true` if the lashlang VM (or its scratch dir) has been
    /// mutated since the last successful `snapshot_execution_state` call.
    /// Callers in the per-iteration progress-boundary path should gate
    /// the snapshot on this so chat-only iterations skip the round-trip.
    pub fn execution_state_dirty(&self) -> bool {
        self.lashlang_state_dirty
    }

    /// Snapshot execution-mode-local state, if any.
    pub async fn snapshot_execution_state(&mut self) -> Result<Option<Vec<u8>>, SessionError> {
        if !self.supports_repl() {
            self.lashlang_state_dirty = false;
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
        self.lashlang_state_dirty = false;
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
        self.lashlang_state_dirty = true;

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
    use crate::plugin::{
        DirectCompletionHost, DynamicToolHost, MonitorHost, PromptHost, SessionGraphHost,
        SessionLifecycleHost, SessionSnapshotHost, StaticPluginFactory, TaskHost, ToolCatalogHost,
        TraceHost, TurnHost,
    };
    use crate::{
        PluginError, PluginHost, PluginSpec, SessionHandle, SessionSnapshot, ToolDefinition,
        ToolResult, TurnInput,
    };
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;

    struct NoopManager;

    #[async_trait::async_trait]
    impl SessionSnapshotHost for NoopManager {
        async fn snapshot_current(&self) -> Result<SessionSnapshot, PluginError> {
            Err(PluginError::Session("snapshot unavailable".to_string()))
        }

        async fn snapshot_session(
            &self,
            _session_id: &str,
        ) -> Result<SessionSnapshot, PluginError> {
            Err(PluginError::Session("snapshot unavailable".to_string()))
        }
    }

    #[async_trait::async_trait]
    impl ToolCatalogHost for NoopManager {
        async fn tool_catalog(
            &self,
            _session_id: &str,
        ) -> Result<Vec<serde_json::Value>, PluginError> {
            Err(PluginError::Session("tool catalog unavailable".to_string()))
        }
    }

    impl DynamicToolHost for NoopManager {}

    #[async_trait::async_trait]
    impl SessionLifecycleHost for NoopManager {
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
    }

    #[async_trait::async_trait]
    impl TurnHost for NoopManager {
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

    impl TaskHost for NoopManager {}
    impl MonitorHost for NoopManager {}
    impl SessionGraphHost for NoopManager {}
    impl PromptHost for NoopManager {}
    impl DirectCompletionHost for NoopManager {}
    impl TraceHost for NoopManager {}

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
                ToolDefinition::new(
                    "echo_async",
                    "Return an echo result after a short delay.",
                    serde_json::json!({
                        "type": "object",
                        "properties": { "text": { "type": "string" } },
                        "required": ["text"],
                        "additionalProperties": false
                    }),
                    serde_json::json!({ "type": "object", "additionalProperties": true }),
                ),
                ToolDefinition::new(
                    "wait_for_cancel",
                    "Wait until the async task is cancelled.",
                    ToolDefinition::default_input_schema(),
                    serde_json::json!({ "type": "object", "additionalProperties": true }),
                ),
                ToolDefinition::new(
                    "spawn_agent",
                    "Test-only spawn stand-in.",
                    serde_json::json!({
                        "type": "object",
                        "properties": { "agent_name": { "type": "string" } },
                        "required": ["agent_name"],
                        "additionalProperties": true
                    }),
                    serde_json::json!({ "type": "object", "additionalProperties": true }),
                ),
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
                "spawn_agent" => {
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

    #[tokio::test]
    async fn set_context_surface_is_noop_for_unchanged_surface() {
        let plugin_host = PluginHost::new(Vec::new());
        let plugin_session = plugin_host
            .build_session(
                "root",
                crate::ExecutionMode::standard(),
                Some(crate::StandardContextApproach::default()),
                None,
            )
            .expect("plugin session");
        let mut session = Session::new(
            crate::RuntimeServices::new(plugin_session),
            "root",
            crate::ExecutionMode::standard(),
        )
        .await
        .expect("session");

        session.set_context_surface(Vec::new(), Vec::new(), true);
        assert_eq!(session.context_surface_revision, 0);

        let prompt_contributions = vec![crate::PromptContribution::guidance(
            "Memory Context",
            "remember",
        )];
        session.set_context_surface(Vec::new(), prompt_contributions.clone(), true);
        assert_eq!(session.context_surface_revision, 1);

        session.set_context_surface(Vec::new(), prompt_contributions, true);
        assert_eq!(session.context_surface_revision, 1);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn run_code_returns_terminal_finish_after_tool_call() {
        let tools: Arc<dyn crate::ToolProvider> = Arc::new(AsyncTestToolProvider {
            state: AsyncToolState::default(),
        });
        let plugin_host = PluginHost::new(vec![Arc::new(StaticPluginFactory::new(
            "test_tools",
            PluginSpec::new().with_tool_provider(Arc::clone(&tools)),
        ))]);
        let plugin_session = plugin_host
            .build_session("root", crate::ExecutionMode::new("rlm"), None, None)
            .expect("plugin session");
        let mut session = Session::new(
            crate::RuntimeServices::new(plugin_session),
            "root",
            crate::ExecutionMode::new("rlm"),
        )
        .await
        .expect("session");

        let (event_tx, _event_rx) = tokio::sync::mpsc::channel(16);
        let manager: Arc<dyn RuntimeSessionHost> = Arc::new(NoopManager);

        let response = session
            .run_code(
                "root",
                manager,
                &event_tx,
                r#"
call echo_async { text: "before finish" }
submit "ok"
"#,
                false,
            )
            .await
            .expect("exec response");

        // `submit "ok"` terminates the lashlang program and delivers the
        // value through `terminal_finish` regardless of mode. The prior
        // tool call stays successful; no error is surfaced.
        assert_eq!(response.error, None);
        assert_eq!(response.tool_calls.len(), 1);
        assert_eq!(response.tool_calls[0].tool, "echo_async");
        assert!(response.tool_calls[0].success);
        assert_eq!(response.terminal_finish, Some(serde_json::json!("ok")));
    }

    #[cfg(feature = "tool-impls")]
    #[tokio::test(flavor = "multi_thread")]
    async fn rlm_run_code_can_call_common_repo_tools_without_model() {
        let temp = tempfile::tempdir().expect("tempdir");
        std::fs::write(temp.path().join("alpha.txt"), "hello\n").expect("write alpha");
        std::fs::write(temp.path().join("beta.rs"), "fn main() {}\n").expect("write beta");
        let temp_path = temp.path().display().to_string();

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
            .build_session("root", crate::ExecutionMode::new("rlm"), None, None)
            .expect("plugin session");
        let mut session = Session::new(
            crate::RuntimeServices::new(plugin_session),
            "root",
            crate::ExecutionMode::new("rlm"),
        )
        .await
        .expect("session");

        let (event_tx, _event_rx) = tokio::sync::mpsc::channel(16);
        let manager: Arc<dyn RuntimeSessionHost> = Arc::new(NoopManager);

        let response = session
            .run_code(
                "root",
                manager,
                &event_tx,
                &format!(
                    r#"
files = call ls {{ path: "{}" }}
match = call grep {{ query: "fn main", path: "{}" }}
print files
print match
"#,
                    temp_path, temp_path
                ),
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
            .build_session("root", crate::ExecutionMode::new("rlm"), None, None)
            .expect("plugin session");
        let mut session = Session::new(
            crate::RuntimeServices::new(plugin_session),
            "root",
            crate::ExecutionMode::new("rlm"),
        )
        .await
        .expect("session");

        let (event_tx, _event_rx) = tokio::sync::mpsc::channel(16);
        let manager: Arc<dyn RuntimeSessionHost> = Arc::new(NoopManager);

        let response = session
            .run_code(
                "root",
                manager,
                &event_tx,
                r#"
handle = start call echo_async { text: "hello" }
result = await handle
print result
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
            .build_session("root", crate::ExecutionMode::new("rlm"), None, None)
            .expect("plugin session");
        let mut session = Session::new(
            crate::RuntimeServices::new(plugin_session),
            "root",
            crate::ExecutionMode::new("rlm"),
        )
        .await
        .expect("session");

        let (event_tx, _event_rx) = tokio::sync::mpsc::channel(16);
        let manager: Arc<dyn RuntimeSessionHost> = Arc::new(NoopManager);

        let response = session
            .run_code(
                "root",
                manager,
                &event_tx,
                r#"
handle = start call wait_for_cancel {}
cancel handle
print "cancelled"
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
    async fn rlm_run_code_lists_live_async_handles_by_namespace() {
        let async_state = AsyncToolState::default();
        let async_tools: Arc<dyn crate::ToolProvider> = Arc::new(AsyncTestToolProvider {
            state: async_state.clone(),
        });
        let plugin_host = PluginHost::new(vec![Arc::new(StaticPluginFactory::new(
            "async_tools",
            PluginSpec::new().with_tool_provider(Arc::clone(&async_tools)),
        ))]);
        let plugin_session = plugin_host
            .build_session("root", crate::ExecutionMode::new("rlm"), None, None)
            .expect("plugin session");
        let mut session = Session::new(
            crate::RuntimeServices::new(plugin_session),
            "root",
            crate::ExecutionMode::new("rlm"),
        )
        .await
        .expect("session");

        let (event_tx, _event_rx) = tokio::sync::mpsc::channel(16);
        let manager: Arc<dyn RuntimeSessionHost> = Arc::new(NoopManager);

        let response = session
            .run_code(
                "root",
                manager,
                &event_tx,
                r#"
agent = start call spawn_agent { agent_name: "Auth Worker" }
tool = start call wait_for_cancel {}
handles = (call list_async_handles {})?
print handles
cancel handles.subagent.auth_worker
cancel handles.tool[tool.id]
after = (call list_async_handles {})?
print after
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
        assert!(async_state.cancelled.load(Ordering::SeqCst));
        assert!(
            response
                .observations
                .iter()
                .any(|text| text.contains("\"auth_worker\"") && text.contains("\"tool\"")),
            "observations were: {:?}",
            response.observations
        );
        assert!(
            response
                .observations
                .iter()
                .any(|text| text.contains("\"subagent\":{}") && text.contains("\"tool\":{}")),
            "observations were: {:?}",
            response.observations
        );
    }

    // `rlm_run_code_caps_observe_output_via_mode_plugin_config` moved
    // to `lash-mode-rlm` integration tests (lash itself no longer
    // knows about the RLM plugin factory after the crate split).
}
