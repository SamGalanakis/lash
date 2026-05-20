use serde_json::json;

use super::execution_context::ModeExecutionContext;
use super::tool_execution::ModeToolReply;

impl ModeExecutionContext<'_> {
    pub(super) async fn live_processes(&self) -> Vec<crate::ProcessHandleGrantEntry> {
        self.dispatch
            .host
            .list_process_handles(crate::ProcessListRequest::new(&self.session_id))
            .await
            .unwrap_or_default()
            .into_iter()
            .filter(|(_, process)| !process.is_terminal())
            .collect()
    }

    pub(super) fn process_status_value(status: &crate::ProcessRecord) -> serde_json::Value {
        json!({
            "process_id": status.id,
            "terminal": terminal_status_label(status.terminal.as_ref()),
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
        live_processes: Vec<crate::ProcessHandleGrantEntry>,
    ) -> ModeToolReply {
        let mut monitor = serde_json::Map::new();
        let mut tool = serde_json::Map::new();
        let mut subagent = serde_json::Map::new();
        let mut process = serde_json::Map::new();

        for (grant, record) in live_processes {
            let kind = grant.descriptor.kind.as_deref().unwrap_or("process");
            let value = Self::process_handle_value(&record.id, kind);
            match kind {
                "monitor" => {
                    monitor.insert(Self::monitor_handle_identifier(&record.id), value);
                }
                "tool" => {
                    tool.insert(record.id, value);
                }
                "subagent" => {
                    subagent.insert(record.id, value);
                }
                _ => {
                    process.insert(record.id, value);
                }
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

fn terminal_status_label(terminal: Option<&crate::ProcessTerminalSemantics>) -> &'static str {
    match terminal.map(|terminal| terminal.state) {
        None => "running",
        Some(crate::ProcessTerminalState::Completed) => "completed",
        Some(crate::ProcessTerminalState::Failed) => "failed",
        Some(crate::ProcessTerminalState::Cancelled) => "cancelled",
    }
}
