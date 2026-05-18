use serde_json::json;

use super::async_handles::{
    AsyncToolHandleEntry, AsyncToolHandleMetadata, AsyncToolHandleNamespace, AsyncToolTerminal,
};
use super::execution_context::ModeExecutionContext;
use super::tool_execution::ModeToolReply;
use crate::ToolCallRecord;
use crate::tool_dispatch::ToolDispatchOutcome;

impl ModeExecutionContext {
    pub(super) async fn live_monitor_tasks(&self) -> Vec<crate::BackgroundTaskRecord> {
        self.dispatch
            .host
            .list_background_tasks(&self.session_id)
            .await
            .unwrap_or_default()
            .into_iter()
            .filter(|task| {
                task.kind == crate::BackgroundTaskKind::Monitor && !task.state.is_terminal()
            })
            .collect()
    }

    pub(super) fn background_task_status_value(
        status: &crate::BackgroundTaskRecord,
    ) -> serde_json::Value {
        json!({
            "task_id": status.id,
            "kind": status.kind.as_str(),
            "producer": status.producer,
            "state": match status.state {
                crate::BackgroundTaskState::Pending => "pending",
                crate::BackgroundTaskState::Running => "running",
                crate::BackgroundTaskState::Waiting => "idle",
                crate::BackgroundTaskState::Completed => "completed",
                crate::BackgroundTaskState::Failed => "failed",
                crate::BackgroundTaskState::CancelRequested => "cancel_requested",
                crate::BackgroundTaskState::Cancelled => "cancelled",
            },
        })
    }

    pub(super) fn monitor_handle_identifier(task_id: &str) -> String {
        task_id
            .strip_prefix("monitor:")
            .unwrap_or(task_id)
            .to_string()
    }

    pub(super) fn ensure_monitor_async_handle(
        &self,
        status: &crate::BackgroundTaskRecord,
    ) -> serde_json::Value {
        let mut handles = self
            .async_tool_handles
            .lock()
            .expect("async tool handle map lock");
        handles.entry(status.id.clone()).or_insert_with(|| {
            AsyncToolHandleEntry::empty_monitor(AsyncToolHandleMetadata {
                tool_name: "monitor".to_string(),
                namespace: AsyncToolHandleNamespace::Monitor,
                identifier: Self::monitor_handle_identifier(&status.id),
            })
        });
        Self::async_tool_handle_value(&status.id, "monitor")
    }

    pub(super) async fn start_monitor_handle_call(
        &self,
        call_id: String,
        args: serde_json::Value,
        tc_num: usize,
    ) -> ModeToolReply {
        let executed = self
            .execute_tool_call(call_id, "monitor".to_string(), args, tc_num, None)
            .await;

        let reply = if executed.completed.output.is_success() {
            let task_id = executed
                .completed
                .output
                .value_for_projection()
                .get("task_id")
                .and_then(|value| value.as_str())
                .map(str::to_string);
            match task_id {
                Some(task_id) => {
                    let status = crate::BackgroundTaskRecord::local_session(
                        self.session_id.clone(),
                        task_id.clone(),
                        crate::BackgroundTaskKind::Monitor,
                        "monitor",
                        crate::BackgroundTaskState::Running,
                    );
                    ModeToolReply::success(self.ensure_monitor_async_handle(&status))
                }
                None => ModeToolReply::error(json!("monitor started but did not return a task_id")),
            }
        } else {
            ModeToolReply::from_output(executed.completed.output.clone())
        };

        reply.with_record(executed.record)
    }

    pub(super) fn list_async_handles(
        &self,
        live_monitor_tasks: Vec<crate::BackgroundTaskRecord>,
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
        let mut tool = serde_json::Map::new();
        for (id, metadata) in entries {
            let value = Self::async_tool_handle_value(&id, &metadata.tool_name);
            match metadata.namespace {
                AsyncToolHandleNamespace::Monitor => {
                    monitor.insert(metadata.identifier, value);
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
            "tool": tool,
        }))
    }

    pub(super) async fn await_monitor_handle(&self, task_id: &str) -> ModeToolReply {
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
            if status.state.is_terminal() {
                if let Some(entry) = self
                    .async_tool_handles
                    .lock()
                    .ok()
                    .and_then(|handles| handles.get(task_id).cloned())
                {
                    let mut guard = entry.state.lock().expect("async tool state lock");
                    if guard.terminal.is_none() {
                        guard.terminal = Some(match status.state {
                            crate::BackgroundTaskState::Cancelled => AsyncToolTerminal::Cancelled,
                            crate::BackgroundTaskState::Failed => {
                                AsyncToolTerminal::Failed("monitor failed".to_string())
                            }
                            _ => AsyncToolTerminal::Completed(ToolDispatchOutcome {
                                record: ToolCallRecord {
                                    call_id: None,
                                    tool: "monitor".into(),
                                    args: json!({}),
                                    output: crate::ToolCallOutput::success(
                                        Self::background_task_status_value(&status),
                                    ),
                                    duration_ms: 0,
                                },
                            }),
                        });
                    }
                }
                let value = Self::background_task_status_value(&status);
                return match status.state {
                    crate::BackgroundTaskState::Failed => ModeToolReply::error(value),
                    crate::BackgroundTaskState::Cancelled => {
                        ModeToolReply::cancelled("monitor was cancelled")
                    }
                    _ => ModeToolReply::success(value),
                };
            }
            tokio::time::sleep(std::time::Duration::from_millis(250)).await;
        }
    }

    pub(super) async fn cancel_monitor_handle(&self, task_id: &str) -> ModeToolReply {
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
