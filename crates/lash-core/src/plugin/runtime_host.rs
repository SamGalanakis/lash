use serde::{Deserialize, Serialize};

use super::*;

#[async_trait::async_trait]
pub trait RuntimeSessionHost: Send + Sync {
    async fn snapshot_current(&self) -> Result<SessionSnapshot, PluginError> {
        Err(PluginError::Session(
            "session snapshots are unavailable in this runtime".to_string(),
        ))
    }

    async fn snapshot_session(&self, _session_id: &str) -> Result<SessionSnapshot, PluginError> {
        Err(PluginError::Session(
            "session lookup is unavailable in this runtime".to_string(),
        ))
    }

    async fn tool_catalog(&self, _session_id: &str) -> Result<Vec<serde_json::Value>, PluginError> {
        Err(PluginError::Session(
            "tool catalogs are unavailable in this runtime".to_string(),
        ))
    }

    async fn tool_state(&self, _session_id: &str) -> Result<crate::ToolState, PluginError> {
        Err(PluginError::Session(
            "tool state is unavailable in this session".to_string(),
        ))
    }

    async fn apply_tool_state(
        &self,
        _session_id: &str,
        _snapshot: crate::ToolState,
    ) -> Result<u64, PluginError> {
        Err(PluginError::Session(
            "tool state mutation is unavailable in this session".to_string(),
        ))
    }

    async fn set_tools_availability(
        &self,
        session_id: &str,
        tool_names: &[String],
        availability: Option<crate::ToolAvailability>,
    ) -> Result<u64, PluginError> {
        let mut snapshot = self.tool_state(session_id).await?;
        for name in tool_names {
            snapshot
                .set_availability(name, availability)
                .map_err(|err| PluginError::Session(err.to_string()))?;
        }
        self.apply_tool_state(session_id, snapshot).await
    }

    async fn set_tool_availability(
        &self,
        session_id: &str,
        tool_name: &str,
        availability: Option<ToolAvailability>,
    ) -> Result<u64, PluginError> {
        let mut snapshot = self.tool_state(session_id).await?;
        snapshot
            .set_availability(tool_name, availability)
            .map_err(|err| PluginError::Session(err.to_string()))?;
        self.apply_tool_state(session_id, snapshot).await
    }

    async fn create_session(
        &self,
        _request: SessionCreateRequest,
    ) -> Result<SessionHandle, PluginError> {
        Err(PluginError::Session(
            "session creation is unavailable in this runtime".to_string(),
        ))
    }

    /// Pop the seed message that was queued for `session_id` via
    /// `SessionCreateRequest::first_turn_input`. Returns `None` if no
    /// seed was queued, or after a previous caller has already taken
    /// it. Hosts call this when starting the inaugural turn on a
    /// freshly created session.
    async fn take_first_turn_input(
        &self,
        _session_id: &str,
    ) -> Result<Option<PluginMessage>, PluginError> {
        Ok(None)
    }

    async fn close_session(&self, _session_id: &str) -> Result<(), PluginError> {
        Err(PluginError::Session(
            "session closing is unavailable in this runtime".to_string(),
        ))
    }

    async fn start_turn(
        &self,
        _session_id: &str,
        _input: TurnInput,
    ) -> Result<AssembledTurn, PluginError> {
        Err(PluginError::Session(
            "session execution is unavailable in this runtime".to_string(),
        ))
    }

    /// Push a user-visible message into the target session's turn-input
    /// injection bridge so it surfaces at the next iteration boundary of
    /// the current turn (or at the start of the next turn if the target
    /// is idle).
    async fn inject_turn_input(
        &self,
        _session_id: &str,
        _input: crate::InjectedTurnInput,
    ) -> Result<(), PluginError> {
        Err(PluginError::Session(
            "turn input injection is unavailable in this session".to_string(),
        ))
    }

    async fn start_process(
        &self,
        _request: crate::ProcessStartRequest<'_>,
    ) -> Result<crate::ProcessRecord, PluginError> {
        Err(PluginError::Session(
            "processes are unavailable in this session".to_string(),
        ))
    }

    async fn await_process(
        &self,
        _request: crate::ProcessAwaitRequest<'_>,
    ) -> Result<crate::ProcessAwaitOutput, PluginError> {
        Err(PluginError::Session(
            "process awaiting is unavailable in this session".to_string(),
        ))
    }

    async fn list_process_handles(
        &self,
        _request: crate::ProcessListRequest<'_>,
    ) -> Result<Vec<crate::ProcessHandleGrantEntry>, PluginError> {
        Err(PluginError::Session(
            "process registry is unavailable in this session".to_string(),
        ))
    }

    async fn cancel_process(
        &self,
        _request: crate::ProcessCancelRequest<'_>,
    ) -> Result<crate::ProcessRecord, PluginError> {
        Err(PluginError::Session(
            "process registry is unavailable in this session".to_string(),
        ))
    }

    async fn cancel_all_processes(
        &self,
        session_id: &str,
    ) -> Result<Vec<crate::ProcessRecord>, PluginError> {
        let tasks = self
            .list_process_handles(crate::ProcessListRequest::new(session_id))
            .await?;
        let mut cancelled = Vec::new();
        for (grant, record) in tasks {
            if record.is_terminal() {
                continue;
            }
            cancelled.push(
                self.cancel_process(crate::ProcessCancelRequest::new(
                    session_id,
                    grant.process_id,
                ))
                .await?,
            );
        }
        Ok(cancelled)
    }

    async fn validate_process_handles_visible(
        &self,
        _session_id: &str,
        _handle_ids: &[String],
    ) -> Result<(), PluginError> {
        Err(PluginError::Session(
            "process handle validation is unavailable in this runtime".to_string(),
        ))
    }

    async fn transfer_process_handles(
        &self,
        request: crate::ProcessTransferRequest<'_>,
    ) -> Result<(), PluginError> {
        if request.process_ids.is_empty() {
            return Ok(());
        }
        Err(PluginError::Session(
            "process handle transfer is unavailable in this runtime".to_string(),
        ))
    }

    async fn cancel_unreferenced_process_handles(
        &self,
        _request: crate::ProcessCleanupRequest<'_>,
    ) -> Result<Vec<crate::ProcessRecord>, PluginError> {
        Err(PluginError::Session(
            "process handle cleanup is unavailable in this runtime".to_string(),
        ))
    }

    async fn monitor_snapshot(&self, _session_id: &str) -> Result<MonitorSnapshot, PluginError> {
        Err(PluginError::Session(
            "monitors are unavailable in this session".to_string(),
        ))
    }

    async fn start_monitor(
        &self,
        _session_id: &str,
        _spec: MonitorSpec,
    ) -> Result<MonitorSnapshot, PluginError> {
        Err(PluginError::Session(
            "monitors are unavailable in this session".to_string(),
        ))
    }

    async fn stop_monitor(
        &self,
        _session_id: &str,
        _monitor_id: &str,
    ) -> Result<MonitorSnapshot, PluginError> {
        Err(PluginError::Session(
            "monitors are unavailable in this session".to_string(),
        ))
    }

    async fn append_session_nodes(
        &self,
        _session_id: &str,
        _request: AppendSessionNodesRequest,
    ) -> Result<AppendSessionNodesResult, PluginError> {
        Err(PluginError::Session(
            "session graph mutation is unavailable in this session".to_string(),
        ))
    }

    async fn emit_trace_event(
        &self,
        _context: lash_trace::TraceContext,
        _event: lash_trace::TraceEvent,
    ) -> Result<(), PluginError> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DefaultHost;

    impl RuntimeSessionHost for DefaultHost {}

    fn assert_session_error(err: PluginError, expected: &str) {
        match err {
            PluginError::Session(message) => assert!(
                message.contains(expected),
                "expected `{message}` to contain `{expected}`"
            ),
            other => panic!("expected session error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn default_process_handle_validation_fails_loudly() {
        let err = DefaultHost
            .validate_process_handles_visible("session", &["process".to_string()])
            .await
            .expect_err("default host must reject process grant validation");

        assert_session_error(err, "process handle validation is unavailable");
    }

    #[tokio::test]
    async fn default_process_handle_transfer_fails_loudly() {
        let err = DefaultHost
            .transfer_process_handles(crate::ProcessTransferRequest::new(
                "source",
                "target",
                vec!["process".to_string()],
            ))
            .await
            .expect_err("default host must reject process grant transfer");

        assert_session_error(err, "process handle transfer is unavailable");
    }

    #[tokio::test]
    async fn default_process_handle_cleanup_fails_loudly() {
        let err = DefaultHost
            .cancel_unreferenced_process_handles(crate::ProcessCleanupRequest::new(
                "session",
                vec!["process".to_string()],
            ))
            .await
            .expect_err("default host must reject process grant cleanup");

        assert_session_error(err, "process handle cleanup is unavailable");
    }
}

/// Result of a single-shot direct LLM call.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DirectCompletion {
    pub text: String,
    pub usage: crate::TokenUsage,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DirectLlmCompletion {
    pub response: crate::LlmResponse,
    pub usage: crate::TokenUsage,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AppendSessionNodesRequest {
    pub nodes: Vec<SessionAppendNode>,
    #[serde(default)]
    pub requires_ancestor_node_id: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum AppendSessionNodesResult {
    Appended {
        node_ids: Vec<String>,
        leaf_node_id: String,
    },
    StaleBranch {
        current_leaf_node_id: Option<String>,
    },
}
