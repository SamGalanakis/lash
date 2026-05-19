use serde_json::json;

use super::execution_context::ModeExecutionContext;
use super::tool_execution::ModeToolReply;

impl ModeExecutionContext<'_> {
    pub(super) async fn live_background_tasks(&self) -> Vec<crate::BackgroundTaskRecord> {
        self.dispatch
            .host
            .list_background_tasks(&self.session_id)
            .await
            .unwrap_or_default()
            .into_iter()
            .filter(|task| !task.state.is_terminal())
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
                crate::BackgroundTaskState::Scheduled => "scheduled",
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

    pub(super) async fn start_monitor_handle_call(
        &self,
        call_id: String,
        args: serde_json::Value,
        tc_num: usize,
    ) -> ModeToolReply {
        let executed = self
            .execute_tool_call(call_id, "monitor".to_string(), args, tc_num, None, None)
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
                    ModeToolReply::success(Self::async_tool_handle_value(&task_id, "monitor"))
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
        live_tasks: Vec<crate::BackgroundTaskRecord>,
    ) -> ModeToolReply {
        let mut monitor = serde_json::Map::new();
        let mut tool = serde_json::Map::new();
        let mut subagent = serde_json::Map::new();
        let mut task = serde_json::Map::new();

        for record in live_tasks {
            let handle_tool = match record.kind {
                crate::BackgroundTaskKind::Monitor => "monitor",
                crate::BackgroundTaskKind::Tool => record.producer.as_str(),
                crate::BackgroundTaskKind::SessionTurn => "spawn_agent",
                crate::BackgroundTaskKind::Observer
                | crate::BackgroundTaskKind::External
                | crate::BackgroundTaskKind::Other => record.producer.as_str(),
            };
            let value = Self::async_tool_handle_value(&record.id, handle_tool);
            match record.kind {
                crate::BackgroundTaskKind::Monitor => {
                    monitor.insert(Self::monitor_handle_identifier(&record.id), value);
                }
                crate::BackgroundTaskKind::Tool => {
                    tool.insert(record.id, value);
                }
                crate::BackgroundTaskKind::SessionTurn => {
                    subagent.insert(record.id, value);
                }
                crate::BackgroundTaskKind::Observer
                | crate::BackgroundTaskKind::External
                | crate::BackgroundTaskKind::Other => {
                    task.insert(record.id, value);
                }
            }
        }

        ModeToolReply::success(json!({
            "monitor": monitor,
            "tool": tool,
            "subagent": subagent,
            "task": task,
        }))
    }
}
