use serde::{Deserialize, Serialize};

use super::*;

#[async_trait::async_trait]
pub trait SessionSnapshotHost: Send + Sync {
    async fn snapshot_current(&self) -> Result<SessionSnapshot, PluginError>;
    async fn snapshot_session(&self, session_id: &str) -> Result<SessionSnapshot, PluginError>;
}

#[async_trait::async_trait]
pub trait ToolCatalogHost: Send + Sync {
    async fn tool_catalog(&self, session_id: &str) -> Result<Vec<serde_json::Value>, PluginError>;
}

#[async_trait::async_trait]
pub trait ToolStateHost: Send + Sync {
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
}

#[async_trait::async_trait]
pub trait SessionLifecycleHost: Send + Sync {
    async fn create_session(
        &self,
        request: SessionCreateRequest,
    ) -> Result<SessionHandle, PluginError>;
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
    async fn close_session(&self, session_id: &str) -> Result<(), PluginError>;
}

#[async_trait::async_trait]
pub trait TurnHost: Send + Sync {
    async fn start_turn_stream(
        &self,
        session_id: &str,
        input: TurnInput,
    ) -> Result<SessionTurnHandle, PluginError>;
    async fn await_turn(&self, turn_id: &str) -> Result<AssembledTurn, PluginError>;
    async fn cancel_turn(&self, turn_id: &str) -> Result<(), PluginError>;
    async fn start_turn(
        &self,
        session_id: &str,
        input: TurnInput,
    ) -> Result<AssembledTurn, PluginError> {
        let handle = self.start_turn_stream(session_id, input).await?;
        drop(handle.events);
        self.await_turn(&handle.turn_id).await
    }
}

#[async_trait::async_trait]
pub trait TaskHost: Send + Sync {
    /// Push a user-visible message into the target session's turn-input
    /// injection bridge so it surfaces at the next iteration boundary of
    /// the current turn (or at the start of the next turn if the target
    /// is idle). Used by monitor and other wake-up flows where a note
    /// should land at the next available step rather than waiting for a
    /// brand-new task.
    async fn inject_turn_input(
        &self,
        _session_id: &str,
        _input: crate::InjectedTurnInput,
    ) -> Result<(), PluginError> {
        Err(PluginError::Session(
            "turn input injection is unavailable in this session".to_string(),
        ))
    }
    async fn spawn_hidden_task(
        &self,
        _session_id: &str,
        _label: &str,
        _task: PluginSessionTask,
    ) -> Result<(), PluginError> {
        Err(PluginError::Session(
            "background tasks are unavailable in this session".to_string(),
        ))
    }
    async fn await_hidden_tasks(&self, _session_id: &str) -> Result<(), PluginError> {
        Ok(())
    }
    async fn spawn_managed_task(
        &self,
        _session_id: &str,
        _spec: crate::BackgroundTaskRegistration,
        _task: PluginSessionTask,
    ) -> Result<(), PluginError> {
        Err(PluginError::Session(
            "managed background tasks are unavailable in this session".to_string(),
        ))
    }
    async fn cancel_managed_task(
        &self,
        _session_id: &str,
        _task_id: &str,
    ) -> Result<(), PluginError> {
        Err(PluginError::Session(
            "managed background tasks are unavailable in this session".to_string(),
        ))
    }
    async fn register_background_task(
        &self,
        _session_id: &str,
        _spec: crate::BackgroundTaskRegistration,
        _cancel: Option<crate::LocalBackgroundTaskCancel>,
    ) -> Result<(), PluginError> {
        Err(PluginError::Session(
            "background task registry is unavailable in this session".to_string(),
        ))
    }
    async fn unregister_background_task(&self, _session_id: &str, _task_id: &str) {}
    async fn complete_background_task(
        &self,
        _session_id: &str,
        _task_id: &str,
        _state: crate::BackgroundTaskState,
    ) {
    }
    /// Transition a still-live background task between the non-terminal
    /// `Running` and `Idle` run states. Used by subagent hosts to
    /// reflect whether the subagent is actively working or waiting for
    /// a follow-up task.
    async fn transition_background_task_live_state(
        &self,
        _session_id: &str,
        _task_id: &str,
        _state: crate::BackgroundTaskState,
    ) {
    }
    async fn list_background_tasks(
        &self,
        _session_id: &str,
    ) -> Result<Vec<crate::BackgroundTaskRecord>, PluginError> {
        Err(PluginError::Session(
            "background task registry is unavailable in this session".to_string(),
        ))
    }
    /// Dispatch a kind-aware cancel for any registered background task.
    /// Monitor tasks terminate their process trees; subagent tasks close
    /// the agent subtree; other managed tasks are aborted.
    async fn cancel_background_task(
        &self,
        _session_id: &str,
        _task_id: &str,
    ) -> Result<crate::BackgroundTaskRecord, PluginError> {
        Err(PluginError::Session(
            "background task registry is unavailable in this session".to_string(),
        ))
    }
    async fn cancel_all_background_tasks(
        &self,
        session_id: &str,
    ) -> Result<Vec<crate::BackgroundTaskRecord>, PluginError> {
        let tasks = self.list_background_tasks(session_id).await?;
        let mut cancelled = Vec::new();
        for task in tasks {
            if task.state.is_terminal() {
                continue;
            }
            cancelled.push(self.cancel_background_task(session_id, &task.id).await?);
        }
        Ok(cancelled)
    }
    async fn validate_async_handles_visible(
        &self,
        _session_id: &str,
        _handle_ids: &[String],
    ) -> Result<(), PluginError> {
        Ok(())
    }
    async fn transfer_async_handles(
        &self,
        _from_session_id: &str,
        _to_session_id: &str,
        _handle_ids: &[String],
    ) -> Result<(), PluginError> {
        Ok(())
    }
    async fn cancel_unreferenced_async_handles(
        &self,
        _session_id: &str,
        _keep_handle_ids: &[String],
    ) -> Result<Vec<crate::BackgroundTaskRecord>, PluginError> {
        Ok(Vec::new())
    }
}

#[async_trait::async_trait]
pub trait MonitorHost: Send + Sync {
    async fn monitor_snapshot(&self, _session_id: &str) -> Result<MonitorSnapshot, PluginError> {
        Err(PluginError::Session(
            "monitors are unavailable in this session".to_string(),
        ))
    }
    async fn take_monitor_updates(
        &self,
        _session_id: &str,
    ) -> Result<MonitorUpdateBatch, PluginError> {
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
}

#[async_trait::async_trait]
pub trait SessionGraphHost: Send + Sync {
    async fn append_session_nodes(
        &self,
        _session_id: &str,
        _request: AppendSessionNodesRequest,
    ) -> Result<AppendSessionNodesResult, PluginError> {
        Err(PluginError::Session(
            "session graph mutation is unavailable in this session".to_string(),
        ))
    }
}

#[async_trait::async_trait]
pub trait DirectCompletionHost: Send + Sync {
    /// Make a single LLM call without creating a full session. Used by
    /// plugins for structured extraction, summarization, observation,
    /// and other one-shot calls that don't need tools, turn loops, or
    /// session state. The `usage_source` label tags the resulting
    /// token cost in the parent session's ledger.
    async fn direct_completion(
        &self,
        _request: crate::DirectRequest,
        _usage_source: &str,
    ) -> Result<DirectCompletion, PluginError> {
        Err(PluginError::Session(
            "direct completions are unavailable in this session".to_string(),
        ))
    }

    async fn direct_llm_completion(
        &self,
        _request: crate::LlmRequest,
        _usage_source: &str,
    ) -> Result<DirectLlmCompletion, PluginError> {
        Err(PluginError::Session(
            "direct LLM completions are unavailable in this session".to_string(),
        ))
    }
}

#[async_trait::async_trait]
pub trait TraceHost: Send + Sync {
    async fn emit_trace_event(
        &self,
        _context: lash_trace::TraceContext,
        _event: lash_trace::TraceEvent,
    ) -> Result<(), PluginError> {
        Ok(())
    }
}

pub trait PromptHookHost:
    SessionSnapshotHost + ToolCatalogHost + TaskHost + DirectCompletionHost
{
}
impl<T> PromptHookHost for T where
    T: SessionSnapshotHost + ToolCatalogHost + TaskHost + DirectCompletionHost + ?Sized
{
}

pub trait TurnHookHost:
    SessionSnapshotHost + ToolStateHost + SessionLifecycleHost + TraceHost
{
}
impl<T> TurnHookHost for T where
    T: SessionSnapshotHost + ToolStateHost + SessionLifecycleHost + TraceHost + ?Sized
{
}

pub trait ToolHookHost:
    SessionSnapshotHost
    + ToolCatalogHost
    + ToolStateHost
    + SessionLifecycleHost
    + TurnHost
    + TaskHost
    + MonitorHost
    + SessionGraphHost
    + DirectCompletionHost
    + TraceHost
    + TurnResultHookHost
    + CheckpointHookHost
{
}
impl<T> ToolHookHost for T where
    T: SessionSnapshotHost
        + ToolCatalogHost
        + ToolStateHost
        + SessionLifecycleHost
        + TurnHost
        + TaskHost
        + MonitorHost
        + SessionGraphHost
        + DirectCompletionHost
        + TraceHost
        + ?Sized
{
}

pub trait TurnResultHookHost: SessionLifecycleHost + TraceHost {}
impl<T> TurnResultHookHost for T where T: SessionLifecycleHost + TraceHost + ?Sized {}

pub trait CheckpointHookHost: SessionLifecycleHost + TraceHost {}
impl<T> CheckpointHookHost for T where T: SessionLifecycleHost + TraceHost + ?Sized {}

pub trait HistoryHost:
    SessionSnapshotHost
    + SessionLifecycleHost
    + TurnHost
    + TaskHost
    + SessionGraphHost
    + DirectCompletionHost
{
}
impl<T> HistoryHost for T where
    T: SessionSnapshotHost
        + SessionLifecycleHost
        + TurnHost
        + TaskHost
        + SessionGraphHost
        + DirectCompletionHost
        + ?Sized
{
}

pub trait PluginActionHost:
    SessionSnapshotHost
    + ToolCatalogHost
    + ToolStateHost
    + SessionLifecycleHost
    + TurnHost
    + TaskHost
    + MonitorHost
    + SessionGraphHost
    + DirectCompletionHost
    + TraceHost
{
}
impl<T> PluginActionHost for T where
    T: SessionSnapshotHost
        + ToolCatalogHost
        + ToolStateHost
        + SessionLifecycleHost
        + TurnHost
        + TaskHost
        + MonitorHost
        + SessionGraphHost
        + DirectCompletionHost
        + TraceHost
        + ?Sized
{
}

pub trait RuntimeSessionHost:
    PluginActionHost + ToolHookHost + HistoryHost + TurnHookHost + PromptHookHost
{
}
impl<T> RuntimeSessionHost for T where
    T: PluginActionHost + ToolHookHost + HistoryHost + TurnHookHost + PromptHookHost + ?Sized
{
}

/// Result of a single-shot LLM call via
/// [`DirectCompletionHost::direct_completion`].
#[derive(Clone, Debug)]
pub struct DirectCompletion {
    pub text: String,
    pub usage: crate::TokenUsage,
}

#[derive(Clone, Debug)]
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
