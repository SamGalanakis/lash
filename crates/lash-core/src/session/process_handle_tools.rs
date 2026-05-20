use serde_json::json;

use super::execution_context::ModeExecutionContext;
use super::tool_execution::ModeToolReply;

impl ModeExecutionContext<'_> {
    pub(super) async fn live_processes(&self) -> Vec<crate::ProcessRecord> {
        self.dispatch
            .host
            .list_processes(&self.session_id)
            .await
            .unwrap_or_default()
            .into_iter()
            .filter(|process| !process.state.is_terminal() && process.handle_visible)
            .collect()
    }

    pub(super) fn process_status_value(status: &crate::ProcessRecord) -> serde_json::Value {
        json!({
            "process_id": status.id,
            "producer": status.producer,
            "tags": status.tags,
            "state": match status.state {
                crate::ProcessState::Pending => "pending",
                crate::ProcessState::Scheduled => "scheduled",
                crate::ProcessState::Running => "running",
                crate::ProcessState::Waiting => "idle",
                crate::ProcessState::Completed => "completed",
                crate::ProcessState::Failed => "failed",
                crate::ProcessState::CancelRequested => "cancel_requested",
                crate::ProcessState::Cancelled => "cancelled",
            },
        })
    }

    pub(super) fn monitor_handle_identifier(process_id: &str) -> String {
        process_id
            .strip_prefix("monitor:")
            .unwrap_or(process_id)
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
            let value = executed.completed.output.value_for_projection();
            let process_id = value
                .get("process_id")
                .and_then(|value| value.as_str())
                .map(str::to_string);
            match process_id {
                Some(process_id) => {
                    ModeToolReply::success(Self::process_handle_value(&process_id, "monitor"))
                }
                None => {
                    ModeToolReply::error(json!("monitor started but did not return a process id"))
                }
            }
        } else {
            ModeToolReply::from_output(executed.completed.output.clone())
        };

        reply.with_record(executed.record)
    }

    pub(super) fn list_process_handles(
        &self,
        live_processes: Vec<crate::ProcessRecord>,
    ) -> ModeToolReply {
        let mut monitor = serde_json::Map::new();
        let mut tool = serde_json::Map::new();
        let mut subagent = serde_json::Map::new();
        let mut process = serde_json::Map::new();

        for record in live_processes {
            let handle_tool = if record.tags.iter().any(|tag| tag == "monitor") {
                "monitor"
            } else if record.tags.iter().any(|tag| tag == "subagent") {
                "spawn_agent"
            } else {
                record.producer.as_str()
            };
            let value = Self::process_handle_value(&record.id, handle_tool);
            if record.tags.iter().any(|tag| tag == "monitor") {
                monitor.insert(Self::monitor_handle_identifier(&record.id), value);
            } else if record.tags.iter().any(|tag| tag == "tool") {
                tool.insert(record.id, value);
            } else if record.tags.iter().any(|tag| tag == "subagent") {
                subagent.insert(record.id, value);
            } else {
                process.insert(record.id, value);
            }
        }

        ModeToolReply::success(json!({
            "monitor": monitor,
            "tool": tool,
            "subagent": subagent,
            "process": process,
        }))
    }
}
