use std::sync::{Arc, Mutex};

use tokio::sync::mpsc;

use crate::plugin::{
    PluginSession, SessionGraphService, SessionLifecycleService, SessionStateService,
};
use crate::{
    PreparedToolCall, SessionEvent, ToolCallRecord, ToolFailure, ToolFailureClass, ToolProvider,
    ToolResult, ToolSurface,
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

#[derive(Clone)]
pub struct ToolDispatchContext<'run> {
    pub plugins: Arc<PluginSession>,
    pub tools: Arc<dyn ToolProvider>,
    pub surface: Arc<ToolSurface>,
    pub sessions: Arc<dyn SessionStateService>,
    pub session_lifecycle: Arc<dyn SessionLifecycleService>,
    pub session_graph: Arc<dyn SessionGraphService>,
    pub processes: Arc<dyn crate::ProcessService>,
    pub(crate) effect_controller: crate::runtime::RuntimeEffectControllerHandle<'run>,
    pub(crate) direct_completions: crate::DirectCompletionClient<'run>,
    pub(crate) parent_invocation: Option<crate::RuntimeInvocation>,
    pub session_id: String,
    pub agent_frame_id: crate::AgentFrameId,
    pub event_tx: mpsc::Sender<SessionEvent>,
    pub(crate) checkpoint_messages: CheckpointMessageBuffer,
    pub attachment_store: Arc<dyn crate::AttachmentStore>,
    pub turn_context: crate::TurnContext,
}

impl<'run> ToolDispatchContext<'run> {
    pub fn process_scope(&self) -> crate::ProcessOpScope<'_> {
        crate::ProcessOpScope::new()
            .with_parent_invocation(self.parent_invocation.clone())
            .with_effect_controller(self.effect_controller.as_controller())
            .with_agent_frame_id(Some(self.agent_frame_id.clone()))
    }
}

#[derive(Clone)]
pub(crate) struct ToolDispatchOutcome {
    pub record: ToolCallRecord,
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
        output: result.into_output(),
        duration_ms,
    };
    ToolDispatchOutcome { record }
}

pub(super) fn runtime_failure(
    class: ToolFailureClass,
    code: impl Into<String>,
    message: impl Into<String>,
) -> ToolResult {
    ToolResult::failure(ToolFailure::runtime(class, code, message))
}
