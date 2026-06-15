use std::sync::{Arc, Mutex};

use tokio::sync::mpsc;

use crate::plugin::{
    PluginSession, SessionGraphService, SessionLifecycleService, SessionStateService,
};
use crate::{
    PreparedToolCall, SessionEvent, ToolCallRecord, ToolCatalog, ToolFailure, ToolFailureClass,
    ToolProvider, ToolResult,
};

#[derive(Clone, Default)]
pub(crate) struct CheckpointMessageBuffer {
    queue: Arc<Mutex<Vec<crate::PluginMessage>>>,
}

impl CheckpointMessageBuffer {
    pub(crate) fn enqueue(&self, messages: Vec<crate::PluginMessage>) -> Result<(), String> {
        let mut queue = self
            .queue
            .lock()
            .map_err(|_| "checkpoint message buffer poisoned".to_string())?;
        queue.extend(messages);
        Ok(())
    }

    pub(crate) fn drain(&self) -> Result<Vec<crate::PluginMessage>, String> {
        let mut queue = self
            .queue
            .lock()
            .map_err(|_| "checkpoint message buffer poisoned".to_string())?;
        Ok(queue.drain(..).collect())
    }
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ToolTriggerEffectOutcome {
    pub source_type: String,
    pub source_key: String,
    pub occurrence_id: String,
    #[serde(default)]
    pub payload: serde_json::Value,
    pub idempotency_key: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<serde_json::Value>,
    pub started_process_ids: Vec<String>,
}

#[derive(Clone, Default)]
pub(crate) struct ToolTriggerOutcomeBuffer {
    queue: Arc<Mutex<Vec<ToolTriggerEffectOutcome>>>,
}

impl ToolTriggerOutcomeBuffer {
    pub(crate) fn enqueue(&self, outcome: ToolTriggerEffectOutcome) -> Result<(), String> {
        let mut queue = self
            .queue
            .lock()
            .map_err(|_| "tool trigger outcome buffer poisoned".to_string())?;
        queue.push(outcome);
        Ok(())
    }

    pub(crate) fn drain(&self) -> Result<Vec<ToolTriggerEffectOutcome>, String> {
        let mut queue = self
            .queue
            .lock()
            .map_err(|_| "tool trigger outcome buffer poisoned".to_string())?;
        Ok(queue.drain(..).collect())
    }
}

#[derive(Clone)]
pub struct ToolDispatchContext<'run> {
    pub plugins: Arc<PluginSession>,
    pub tools: Arc<dyn ToolProvider>,
    pub tool_catalog: Arc<ToolCatalog>,
    pub sessions: Arc<dyn SessionStateService>,
    pub session_lifecycle: Arc<dyn SessionLifecycleService>,
    pub session_graph: Arc<dyn SessionGraphService>,
    pub processes: Arc<dyn crate::ProcessService>,
    pub process_cancel_ability: Arc<dyn crate::ProcessCancelAbility>,
    pub trigger_router: Option<crate::TriggerRouter>,
    pub(crate) effect_controller: crate::runtime::RuntimeEffectControllerHandle<'run>,
    pub(crate) direct_completions: crate::DirectCompletionClient<'run>,
    pub(crate) parent_invocation: Option<crate::RuntimeInvocation>,
    pub(crate) execution_env_spec: crate::ProcessExecutionEnvSpec,
    pub session_id: String,
    pub agent_frame_id: crate::AgentFrameId,
    pub event_tx: mpsc::Sender<SessionEvent>,
    pub(crate) checkpoint_messages: CheckpointMessageBuffer,
    pub(crate) trigger_outcomes: ToolTriggerOutcomeBuffer,
    pub attachment_store: Arc<dyn crate::AttachmentStore>,
    pub turn_context: crate::TurnContext,
}

impl<'run> ToolDispatchContext<'run> {
    pub fn process_scope(&self) -> crate::ProcessOpScope<'_> {
        crate::ProcessOpScope::new(self.effect_controller.scoped())
            .with_parent_invocation(self.parent_invocation.clone())
            .with_agent_frame_id(Some(self.agent_frame_id.clone()))
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub(crate) struct ToolDispatchOutcome {
    pub record: ToolCallRecord,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub(crate) struct PendingToolDispatchOutcome {
    pub tool_name: String,
    pub args: serde_json::Value,
    pub key: crate::AwaitEventKey,
    pub pending: crate::PendingCompletion,
    pub duration_ms: u64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub(crate) enum ToolCallLaunch {
    Done(ToolDispatchOutcome),
    Pending(PendingToolDispatchOutcome),
}

pub(crate) enum ToolPreparationOutcome {
    Prepared(PreparedToolCall),
    Completed(Box<ToolDispatchOutcome>),
}

pub(super) fn completed_preparation(outcome: ToolDispatchOutcome) -> ToolPreparationOutcome {
    ToolPreparationOutcome::Completed(Box::new(outcome))
}
pub(super) fn outcome(
    tool_name: String,
    args: serde_json::Value,
    result: ToolResult,
    duration_ms: u64,
) -> ToolDispatchOutcome {
    let record = ToolCallRecord {
        call_id: None,
        tool: tool_name,
        args,
        output: result.into_done_output().unwrap_or_else(|_| {
            crate::ToolCallOutput::failure(crate::ToolFailure::runtime(
                crate::ToolFailureClass::Internal,
                "pending_tool_not_finalized",
                "pending tool result reached a completed-output projection path",
            ))
        }),
        duration_ms,
    };
    ToolDispatchOutcome { record }
}

pub(super) fn launch_done(outcome: ToolDispatchOutcome) -> ToolCallLaunch {
    ToolCallLaunch::Done(outcome)
}

pub(super) fn runtime_failure(
    class: ToolFailureClass,
    code: impl Into<String>,
    message: impl Into<String>,
) -> ToolResult {
    ToolResult::failure(ToolFailure::runtime(class, code, message))
}
