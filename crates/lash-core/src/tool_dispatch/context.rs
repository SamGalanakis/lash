use std::sync::Arc;

use tokio::sync::mpsc;

use crate::plugin::{PluginSession, RuntimeSessionHost};
use crate::{
    PreparedToolCall, SessionEvent, ToolCallRecord, ToolFailure, ToolFailureClass, ToolProvider,
    ToolResult, ToolSurface, TurnInjectionBridge,
};

#[derive(Clone)]
pub struct ToolDispatchContext<'run> {
    pub plugins: Arc<PluginSession>,
    pub tools: Arc<dyn ToolProvider>,
    pub surface: Arc<ToolSurface>,
    pub host: Arc<dyn RuntimeSessionHost>,
    pub processes: Arc<dyn crate::ProcessService>,
    pub(crate) effect_controller: crate::runtime::RuntimeEffectControllerHandle<'run>,
    pub(crate) direct_completions: crate::DirectCompletionClient<'run>,
    pub(crate) tool_effect_metadata: Option<crate::EffectInvocationMetadata>,
    pub session_id: String,
    pub event_tx: mpsc::Sender<SessionEvent>,
    pub turn_injection_bridge: TurnInjectionBridge,
    pub attachment_store: Arc<dyn crate::AttachmentStore>,
    pub turn_context: crate::TurnContext,
}

impl<'run> ToolDispatchContext<'run> {
    pub fn process_scope(&self) -> crate::ProcessOpScope<'_> {
        crate::ProcessOpScope::new()
            .with_effect_metadata(self.tool_effect_metadata.clone())
            .with_effect_controller(self.effect_controller.as_controller())
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
