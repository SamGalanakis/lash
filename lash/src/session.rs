use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex as StdMutex, OnceLock};

use futures_util::stream::{FuturesUnordered, StreamExt};
use serde_json::json;
use tokio::sync::{Notify, mpsc::UnboundedSender};
use tokio_util::sync::CancellationToken;

use crate::tool_dispatch::{
    ParallelToolCallSpec, ToolDispatchContext, ToolDispatchOutcome, dispatch_parallel_tool_call,
    dispatch_tool_call_with_execution_context,
};
use crate::{
    PluginMessage, PromptContribution, RuntimeServices, RuntimeSessionHost, SandboxMessage,
    SessionEvent, ToolCallRecord, ToolExecutionContext, ToolImage, ToolProvider,
};

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
    Monitor,
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

#[derive(Clone, Debug)]
pub struct ExecRequest {
    pub code: String,
    pub accept_finish: bool,
}

#[derive(Clone, Debug)]
pub struct ModeToolBatchItem {
    pub id: String,
    pub name: String,
    pub args: serde_json::Value,
}

#[derive(Clone, Debug)]
pub struct ModeToolReply {
    pub success: bool,
    pub value: serde_json::Value,
    pub images: Vec<ToolImage>,
    pub record: Option<ToolCallRecord>,
}

impl ModeToolReply {
    pub fn success(value: serde_json::Value) -> Self {
        Self {
            success: true,
            value,
            images: Vec::new(),
            record: None,
        }
    }

    pub fn success_with_images(value: serde_json::Value, images: Vec<ToolImage>) -> Self {
        Self {
            success: true,
            value,
            images,
            record: None,
        }
    }

    pub fn error(value: serde_json::Value) -> Self {
        Self {
            success: false,
            value,
            images: Vec::new(),
            record: None,
        }
    }

    fn with_record(mut self, record: ToolCallRecord) -> Self {
        self.record = Some(record);
        self
    }
}

#[derive(Clone)]
pub struct ModeExecutionContext {
    session_id: String,
    execution_mode: crate::ExecutionMode,
    dispatch: Arc<ToolDispatchContext>,
    async_tool_handles: Arc<StdMutex<HashMap<String, AsyncToolHandleEntry>>>,
    message_tx: Option<UnboundedSender<SandboxMessage>>,
    attachment_store: Arc<dyn crate::AttachmentStore>,
    rlm_globals: Arc<serde_json::Map<String, serde_json::Value>>,
    rlm_chronological_projection: Arc<crate::ChronologicalProjection>,
    mode_extension: Option<crate::ModeTurnExtensionHandle>,
}

impl ModeExecutionContext {
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    pub fn execution_mode(&self) -> &crate::ExecutionMode {
        &self.execution_mode
    }

    pub fn attachment_store(&self) -> Arc<dyn crate::AttachmentStore> {
        Arc::clone(&self.attachment_store)
    }

    pub fn rlm_globals(&self) -> Arc<serde_json::Map<String, serde_json::Value>> {
        Arc::clone(&self.rlm_globals)
    }

    pub fn rlm_chronological_projection(&self) -> Arc<crate::ChronologicalProjection> {
        Arc::clone(&self.rlm_chronological_projection)
    }

    pub fn mode_extension<T: 'static>(&self) -> Option<&T> {
        self.mode_extension
            .as_ref()
            .and_then(|extension| extension.as_any().downcast_ref::<T>())
    }

    pub async fn call_tool(
        &self,
        call_id: String,
        name: String,
        args: serde_json::Value,
        index: usize,
    ) -> ModeToolReply {
        if name == "list_async_handles" {
            let live_monitor_tasks = self.live_monitor_tasks().await;
            return self.list_async_handles(live_monitor_tasks);
        }
        if name == "monitor" {
            return self
                .start_monitor_handle_call(call_id, args, index, self.message_tx.clone())
                .await;
        }
        let mut outcome = dispatch_parallel_tool_call(
            Arc::clone(&self.dispatch),
            ParallelToolCallSpec {
                index,
                tool_name: name,
                args,
            },
            self.message_tx.clone(),
        )
        .await;
        outcome.record.call_id = Some(call_id);
        let reply = if outcome.record.success {
            ModeToolReply::success_with_images(
                outcome.record.result.clone(),
                outcome.images.clone(),
            )
        } else {
            ModeToolReply::error(outcome.record.result.clone())
        };
        reply.with_record(outcome.record)
    }

    pub async fn call_tool_batch(&self, calls: Vec<ModeToolBatchItem>) -> Vec<ModeToolReply> {
        let mut pending = FuturesUnordered::new();
        for (offset, call) in calls.into_iter().enumerate() {
            let ctx = self.clone();
            pending.push(async move {
                let reply = ctx.call_tool(call.id, call.name, call.args, offset).await;
                (offset, reply)
            });
        }
        let mut replies = Vec::new();
        while let Some(reply) = pending.next().await {
            replies.push(reply);
        }
        replies.sort_by_key(|(offset, _)| *offset);
        replies.into_iter().map(|(_, reply)| reply).collect()
    }

    pub async fn start_tool_call(
        &self,
        call_id: String,
        name: String,
        args: serde_json::Value,
    ) -> ModeToolReply {
        if name == "monitor" {
            return self
                .start_monitor_handle_call(call_id, args, 0, self.message_tx.clone())
                .await;
        }
        self.start_async_tool_call(call_id, name, args).await
    }

    pub async fn await_tool_handle(
        &self,
        _call_id: String,
        handle: serde_json::Value,
    ) -> ModeToolReply {
        self.await_async_tool_handle(handle).await
    }

    pub async fn cancel_tool_handle(
        &self,
        _call_id: String,
        handle: serde_json::Value,
    ) -> ModeToolReply {
        self.cancel_async_tool_handle(handle).await
    }

    async fn live_monitor_tasks(&self) -> Vec<crate::ManagedTaskStatus> {
        self.dispatch
            .host
            .list_background_tasks(&self.session_id)
            .await
            .unwrap_or_default()
            .into_iter()
            .filter(|task| {
                task.kind == crate::ManagedTaskKind::Monitor && !task.run_state.is_terminal()
            })
            .collect()
    }

    fn async_tool_handle_value(id: &str, tool_name: &str) -> serde_json::Value {
        json!({
            "__handle__": ASYNC_TOOL_HANDLE_KIND,
            "id": id,
            "tool": tool_name,
        })
    }

    fn background_task_status_value(status: &crate::ManagedTaskStatus) -> serde_json::Value {
        json!({
            "task_id": status.id,
            "label": status.label,
            "kind": status.kind.as_str(),
            "producer": status.producer,
            "run_state": match status.run_state {
                crate::ManagedRunState::Running => "running",
                crate::ManagedRunState::Idle => "idle",
                crate::ManagedRunState::Completed => "completed",
                crate::ManagedRunState::Failed => "failed",
                crate::ManagedRunState::Cancelled => "cancelled",
            },
        })
    }

    fn monitor_handle_identifier(task_id: &str) -> String {
        task_id
            .strip_prefix("monitor:")
            .unwrap_or(task_id)
            .to_string()
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

    fn ensure_monitor_async_handle(&self, status: &crate::ManagedTaskStatus) -> serde_json::Value {
        let mut handles = self
            .async_tool_handles
            .lock()
            .expect("async tool handle map lock");
        handles
            .entry(status.id.clone())
            .or_insert_with(|| AsyncToolHandleEntry {
                state: Arc::new(StdMutex::new(AsyncToolHandleState {
                    join_handle: None,
                    buffered_messages: Vec::new(),
                    terminal: None,
                })),
                done_notify: Arc::new(Notify::new()),
                progress_notify: Arc::new(Notify::new()),
                cancellation: CancellationToken::new(),
                metadata: AsyncToolHandleMetadata {
                    tool_name: "monitor".to_string(),
                    namespace: AsyncToolHandleNamespace::Monitor,
                    identifier: Self::monitor_handle_identifier(&status.id),
                },
            });
        Self::async_tool_handle_value(&status.id, "monitor")
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
        &self,
        call_id: String,
        tool_name: String,
        args: serde_json::Value,
    ) -> ModeToolReply {
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
        let dispatch = Arc::clone(&self.dispatch);
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
        let record = ToolCallRecord {
            call_id: Some(call_id),
            tool: tool_name,
            args,
            result: handle_value.clone(),
            success: true,
            duration_ms: 0,
            control: None,
        };
        ModeToolReply::success(handle_value).with_record(record)
    }

    async fn start_monitor_handle_call(
        &self,
        call_id: String,
        args: serde_json::Value,
        tc_num: usize,
        msg_tx: Option<UnboundedSender<SandboxMessage>>,
    ) -> ModeToolReply {
        let mut outcome = dispatch_parallel_tool_call(
            Arc::clone(&self.dispatch),
            ParallelToolCallSpec {
                index: tc_num,
                tool_name: "monitor".to_string(),
                args,
            },
            msg_tx,
        )
        .await;
        outcome.record.call_id = Some(call_id);

        let reply = if outcome.record.success {
            let task_id = outcome
                .record
                .result
                .get("task_id")
                .and_then(|value| value.as_str())
                .map(str::to_string);
            match task_id {
                Some(task_id) => {
                    let status = crate::ManagedTaskStatus {
                        id: task_id.clone(),
                        label: outcome
                            .record
                            .result
                            .get("description")
                            .and_then(|value| value.as_str())
                            .unwrap_or("monitor")
                            .to_string(),
                        kind: crate::ManagedTaskKind::Monitor,
                        producer: "monitor".to_string(),
                        run_state: crate::ManagedRunState::Running,
                        started_at: std::time::SystemTime::now(),
                    };
                    ModeToolReply::success(self.ensure_monitor_async_handle(&status))
                }
                None => ModeToolReply::error(json!("monitor started but did not return a task_id")),
            }
        } else {
            ModeToolReply::error(outcome.record.result.clone())
        };

        ModeToolReply {
            record: Some(outcome.record),
            images: outcome.images,
            ..reply
        }
    }

    fn list_async_handles(
        &self,
        live_monitor_tasks: Vec<crate::ManagedTaskStatus>,
    ) -> ModeToolReply {
        for task in &live_monitor_tasks {
            self.ensure_monitor_async_handle(task);
        }

        let entries = self
            .async_tool_handles
            .lock()
            .expect("async tool handle map lock")
            .iter()
            .filter_map(|(id, entry)| {
                if entry.metadata.namespace == AsyncToolHandleNamespace::Monitor {
                    return None;
                }
                let is_terminal = entry
                    .state
                    .lock()
                    .expect("async tool state lock")
                    .terminal
                    .is_some();
                (!is_terminal).then(|| (id.clone(), entry.metadata.clone()))
            })
            .collect::<Vec<_>>();

        let mut monitor = serde_json::Map::new();
        let mut subagent = serde_json::Map::new();
        let mut tool = serde_json::Map::new();
        for (id, metadata) in entries {
            let value = Self::async_tool_handle_value(&id, &metadata.tool_name);
            match metadata.namespace {
                AsyncToolHandleNamespace::Monitor => {
                    monitor.insert(metadata.identifier, value);
                }
                AsyncToolHandleNamespace::Subagent => {
                    subagent.insert(metadata.identifier, value);
                }
                AsyncToolHandleNamespace::Tool => {
                    tool.insert(metadata.identifier, value);
                }
            }
        }
        for task in live_monitor_tasks {
            monitor.insert(
                Self::monitor_handle_identifier(&task.id),
                Self::async_tool_handle_value(&task.id, "monitor"),
            );
        }
        ModeToolReply::success(json!({
            "monitor": monitor,
            "subagent": subagent,
            "tool": tool,
        }))
    }

    async fn await_async_tool_handle(&self, handle: serde_json::Value) -> ModeToolReply {
        let (handle_id, hinted_tool_name) = match Self::parse_async_tool_handle(&handle) {
            Ok(parsed) => parsed,
            Err(err) => return ModeToolReply::error(json!(err)),
        };
        let Some(entry) = self.async_tool_handle_entry(&handle_id) else {
            if hinted_tool_name.as_deref() == Some("monitor") || handle_id.starts_with("monitor:") {
                return self.await_monitor_handle(&handle_id).await;
            }
            return ModeToolReply::error(json!(format!("Unknown async handle: {handle_id}")));
        };
        if entry.metadata.namespace == AsyncToolHandleNamespace::Monitor {
            return self.await_monitor_handle(&handle_id).await;
        }

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
                if outcome.record.success {
                    ModeToolReply::success_with_images(outcome.record.result, outcome.images)
                } else {
                    ModeToolReply::error(outcome.record.result)
                }
            }
            Some(AsyncToolTerminal::Cancelled) => {
                ModeToolReply::error(json!("async task was cancelled"))
            }
            Some(AsyncToolTerminal::Failed(err)) => ModeToolReply::error(json!(err)),
            None => ModeToolReply::error(json!("async task did not produce a result")),
        }
    }

    async fn await_monitor_handle(&self, task_id: &str) -> ModeToolReply {
        loop {
            let tasks = match self
                .dispatch
                .host
                .list_background_tasks(&self.session_id)
                .await
            {
                Ok(tasks) => tasks,
                Err(err) => return ModeToolReply::error(json!(err.to_string())),
            };
            let Some(status) = tasks.into_iter().find(|task| task.id == task_id) else {
                return ModeToolReply::error(json!(format!("Unknown monitor handle: {task_id}")));
            };
            if status.run_state.is_terminal() {
                if let Some(entry) = self
                    .async_tool_handles
                    .lock()
                    .ok()
                    .and_then(|handles| handles.get(task_id).cloned())
                {
                    let mut guard = entry.state.lock().expect("async tool state lock");
                    if guard.terminal.is_none() {
                        guard.terminal = Some(match status.run_state {
                            crate::ManagedRunState::Cancelled => AsyncToolTerminal::Cancelled,
                            crate::ManagedRunState::Failed => {
                                AsyncToolTerminal::Failed("monitor failed".to_string())
                            }
                            _ => AsyncToolTerminal::Completed(ToolDispatchOutcome {
                                record: ToolCallRecord {
                                    call_id: None,
                                    tool: "monitor".into(),
                                    args: json!({}),
                                    result: Self::background_task_status_value(&status),
                                    success: true,
                                    duration_ms: 0,
                                    control: None,
                                },
                                images: Vec::new(),
                            }),
                        });
                    }
                }
                let value = Self::background_task_status_value(&status);
                return match status.run_state {
                    crate::ManagedRunState::Failed | crate::ManagedRunState::Cancelled => {
                        ModeToolReply::error(value)
                    }
                    _ => ModeToolReply::success(value),
                };
            }
            tokio::time::sleep(std::time::Duration::from_millis(250)).await;
        }
    }

    async fn cancel_async_tool_handle(&self, handle: serde_json::Value) -> ModeToolReply {
        let (handle_id, hinted_tool_name) = match Self::parse_async_tool_handle(&handle) {
            Ok(parsed) => parsed,
            Err(err) => return ModeToolReply::error(json!(err)),
        };
        let Some(entry) = self.async_tool_handle_entry(&handle_id) else {
            if hinted_tool_name.as_deref() == Some("monitor") || handle_id.starts_with("monitor:") {
                return self.cancel_monitor_handle(&handle_id).await;
            }
            return ModeToolReply::error(json!(format!("Unknown async handle: {handle_id}")));
        };
        if entry.metadata.namespace == AsyncToolHandleNamespace::Monitor {
            return self.cancel_monitor_handle(&handle_id).await;
        }

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

        ModeToolReply::success(serde_json::Value::Null)
    }

    async fn cancel_monitor_handle(&self, task_id: &str) -> ModeToolReply {
        match self
            .dispatch
            .host
            .cancel_background_task(&self.session_id, task_id)
            .await
        {
            Ok(status) => {
                if let Some(entry) = self
                    .async_tool_handles
                    .lock()
                    .ok()
                    .and_then(|handles| handles.get(task_id).cloned())
                {
                    let mut guard = entry.state.lock().expect("async tool state lock");
                    guard.terminal = Some(AsyncToolTerminal::Cancelled);
                    entry.progress_notify.notify_waiters();
                    entry.done_notify.notify_waiters();
                }
                ModeToolReply::success(Self::background_task_status_value(&status))
            }
            Err(err) => ModeToolReply::error(json!(err.to_string())),
        }
    }
}

pub struct Session {
    session_id: String,
    execution_mode: crate::ExecutionMode,
    services: RuntimeServices,
    include_base_tools: bool,
    context_surface_revision: u64,
    context_tools: Vec<Arc<dyn ToolProvider>>,
    context_prompt_contributions: Vec<PromptContribution>,
    message_tx: Option<UnboundedSender<SandboxMessage>>,
    tool_surface_cache: std::sync::Mutex<Vec<(ToolSurfaceCacheKey, ToolSurfaceHandle)>>,
    /// Memoizes the rendered system prompt across turns. Most consecutive
    /// turns reuse the same template + context surface, so the cache hits
    /// and we skip the section/Vec-join work in
    /// `lash_sansio::PromptTemplate::render`.
    prompt_cache: Arc<lash_sansio::PromptCache>,
    async_tool_handles: Arc<StdMutex<HashMap<String, AsyncToolHandleEntry>>>,
}

impl Session {
    pub async fn new(
        services: RuntimeServices,
        session_id: &str,
        execution_mode: crate::ExecutionMode,
    ) -> Result<Self, SessionError> {
        let mut session = Self {
            session_id: session_id.to_string(),
            execution_mode,
            services,
            include_base_tools: true,
            context_surface_revision: 0,
            context_tools: Vec::new(),
            context_prompt_contributions: Vec::new(),
            message_tx: None,
            tool_surface_cache: std::sync::Mutex::new(Vec::new()),
            prompt_cache: Arc::new(lash_sansio::PromptCache::new()),
            async_tool_handles: Arc::new(StdMutex::new(HashMap::new())),
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

    pub fn session_id(&self) -> &str {
        &self.session_id
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

    pub(crate) fn mode_execution_context(
        &self,
        session_id: &str,
        host: Arc<dyn RuntimeSessionHost>,
        event_tx: tokio::sync::mpsc::Sender<SessionEvent>,
        rlm_globals: Arc<serde_json::Map<String, serde_json::Value>>,
        rlm_chronological_projection: Arc<crate::ChronologicalProjection>,
        mode_extension: Option<crate::ModeTurnExtensionHandle>,
    ) -> ModeExecutionContext {
        let dispatch = Arc::new(ToolDispatchContext {
            plugins: Arc::clone(self.plugins()),
            tools: self.tools(),
            surface: self.tool_surface(session_id, self.execution_mode.clone()),
            host,
            session_id: session_id.to_string(),
            event_tx,
            turn_injection_bridge: self.turn_injection_bridge().clone(),
            attachment_store: Arc::clone(&self.services.attachment_store),
        });
        ModeExecutionContext {
            session_id: session_id.to_string(),
            execution_mode: self.execution_mode.clone(),
            dispatch,
            async_tool_handles: Arc::clone(&self.async_tool_handles),
            message_tx: self.message_tx.clone(),
            attachment_store: Arc::clone(&self.services.attachment_store),
            rlm_globals,
            rlm_chronological_projection,
            mode_extension,
        }
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

    pub async fn reset(&mut self) -> Result<(), SessionError> {
        self.async_tool_handles
            .lock()
            .expect("async tool handle map lock")
            .clear();
        self.tool_surface_cache
            .lock()
            .expect("tool surface cache lock")
            .clear();
        Ok(())
    }

    pub async fn refresh_tool_surface(&mut self) -> Result<(), SessionError> {
        self.tool_surface_cache
            .lock()
            .expect("tool surface cache lock")
            .clear();
        Ok(())
    }
}
