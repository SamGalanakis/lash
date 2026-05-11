use std::collections::HashMap;
use std::sync::{Arc, Mutex as StdMutex};

use serde_json::json;
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;

use super::execution_context::ModeExecutionContext;
use super::tool_execution::ModeToolReply;
use crate::tool_dispatch::{ToolDispatchOutcome, dispatch_tool_call_with_execution_context};
use crate::{SandboxMessage, ToolCallRecord, ToolContext};

const ASYNC_TOOL_HANDLE_KIND: &str = "task";

pub(super) type AsyncToolHandleMap = Arc<StdMutex<HashMap<String, AsyncToolHandleEntry>>>;

#[derive(Clone)]
pub(super) struct AsyncToolHandleEntry {
    pub(super) state: Arc<StdMutex<AsyncToolHandleState>>,
    pub(super) done_notify: Arc<Notify>,
    pub(super) progress_notify: Arc<Notify>,
    pub(super) cancellation: CancellationToken,
    pub(super) metadata: AsyncToolHandleMetadata,
}

#[derive(Clone)]
pub(super) struct AsyncToolHandleMetadata {
    pub(super) tool_name: String,
    pub(super) namespace: AsyncToolHandleNamespace,
    pub(super) identifier: String,
}

#[derive(Clone, PartialEq, Eq)]
pub(super) enum AsyncToolHandleNamespace {
    Monitor,
    Subagent,
    Tool,
}

pub(super) struct AsyncToolHandleState {
    pub(super) join_handle: Option<tokio::task::JoinHandle<()>>,
    pub(super) buffered_messages: Vec<SandboxMessage>,
    pub(super) terminal: Option<AsyncToolTerminal>,
}

#[derive(Clone)]
pub(super) enum AsyncToolTerminal {
    Completed(ToolDispatchOutcome),
    Cancelled,
    Failed(String),
}

impl AsyncToolHandleEntry {
    pub(super) fn empty_monitor(metadata: AsyncToolHandleMetadata) -> Self {
        Self {
            state: Arc::new(StdMutex::new(AsyncToolHandleState {
                join_handle: None,
                buffered_messages: Vec::new(),
                terminal: None,
            })),
            done_notify: Arc::new(Notify::new()),
            progress_notify: Arc::new(Notify::new()),
            cancellation: CancellationToken::new(),
            metadata,
        }
    }
}

impl ModeExecutionContext {
    pub(super) fn async_tool_handle_value(id: &str, tool_name: &str) -> serde_json::Value {
        json!({
            "__handle__": ASYNC_TOOL_HANDLE_KIND,
            "id": id,
            "tool": tool_name,
        })
    }

    pub(super) fn normalize_async_subagent_name(agent_name: &str) -> Option<String> {
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

    pub(super) fn async_tool_handle_metadata(
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

    pub(super) fn parse_async_tool_handle(
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

    pub(super) fn async_tool_handle_entry(&self, id: &str) -> Option<AsyncToolHandleEntry> {
        self.async_tool_handles.lock().ok()?.get(id).cloned()
    }

    pub(super) fn flush_async_tool_messages(&self, entry: &AsyncToolHandleEntry) {
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

    pub(super) async fn start_async_tool_call(
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
        let async_call_id = handle_id.clone();
        let join_handle = tokio::spawn(async move {
            let tool_context = ToolContext::new(
                dispatch.session_id.clone(),
                Arc::clone(&dispatch.host),
                dispatch.turn_context.clone(),
                Some(async_call_id),
            )
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

    pub(super) async fn await_async_tool_handle(&self, handle: serde_json::Value) -> ModeToolReply {
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

    pub(super) async fn cancel_async_tool_handle(
        &self,
        handle: serde_json::Value,
    ) -> ModeToolReply {
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
}
