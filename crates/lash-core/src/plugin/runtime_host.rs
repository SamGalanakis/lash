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

    async fn shared_tool_catalog(
        &self,
        session_id: &str,
    ) -> Result<std::sync::Arc<Vec<serde_json::Value>>, PluginError> {
        Ok(std::sync::Arc::new(self.tool_catalog(session_id).await?))
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
