use std::sync::Arc;

use crate::plugin::PluginError;

#[derive(Clone)]
pub struct ToolProcessControl<'run> {
    pub(super) session_id: String,
    pub(super) agent_frame_id: crate::AgentFrameId,
    pub(super) processes: Arc<dyn crate::ProcessService>,
    pub(super) process_cancel_ability: Arc<dyn crate::ProcessCancelAbility>,
    pub(super) effect_controller: crate::runtime::RuntimeEffectControllerHandle<'run>,
    pub(super) parent_invocation: Option<crate::RuntimeInvocation>,
    pub(super) tool_call_id: Option<String>,
}

impl ToolProcessControl<'_> {
    fn process_scope(&self) -> crate::ProcessOpScope<'_> {
        crate::ProcessOpScope::new()
            .with_parent_invocation(self.parent_invocation.clone())
            .with_effect_controller(self.effect_controller.as_controller())
            .with_agent_frame_id(Some(self.agent_frame_id.clone()))
    }

    /// Start a process owned by this session and registered to wake it,
    /// returning its public handle summary. Routes through the same
    /// [`crate::ProcessService::start_from_request`] path the runtime uses for
    /// every request-shaped process start, so the child is provider-re-supplied,
    /// durable, and recoverable through the worker.
    pub async fn start(
        &self,
        request: crate::ProcessStartRequest,
    ) -> Result<crate::ProcessHandleSummary, PluginError> {
        self.processes
            .start_from_request(&self.session_id, request, self.process_scope())
            .await
    }

    /// Await a process started from this session to its terminal output.
    pub async fn await_process(
        &self,
        process_id: &str,
    ) -> Result<crate::ProcessAwaitOutput, PluginError> {
        self.processes
            .await_process(process_id, self.process_scope())
            .await
    }

    pub async fn list_handles(&self) -> Result<Vec<crate::ProcessHandleSummary>, PluginError> {
        Ok(self
            .processes
            .list_visible(
                &self.session_id,
                crate::ProcessListMode::Live,
                self.process_scope(),
            )
            .await?
            .into_iter()
            .map(crate::ProcessHandleSummary::from)
            .collect())
    }

    pub async fn list_all_handles(&self) -> Result<Vec<crate::ProcessHandleSummary>, PluginError> {
        Ok(self
            .processes
            .list_visible(
                &self.session_id,
                crate::ProcessListMode::All,
                self.process_scope(),
            )
            .await?
            .into_iter()
            .map(crate::ProcessHandleSummary::from)
            .collect())
    }

    pub async fn list_handles_filtered(
        &self,
        filter: &crate::ProcessListFilter,
    ) -> Result<Vec<crate::ProcessHandleSummary>, PluginError> {
        Ok(self
            .processes
            .list_visible(&self.session_id, filter.list_mode(), self.process_scope())
            .await?
            .into_iter()
            .filter(|entry| filter.matches_entry(entry))
            .map(crate::ProcessHandleSummary::from)
            .collect())
    }

    pub async fn validate_handles(&self, handle_ids: &[String]) -> Result<(), PluginError> {
        self.processes
            .validate_visible(&self.session_id, handle_ids, self.process_scope())
            .await
    }

    pub async fn cancel(
        &self,
        process_id: &str,
    ) -> Result<crate::ProcessCancelSummary, PluginError> {
        let request = crate::ProcessCancelRequest::new(
            &self.session_id,
            process_id,
            self.process_scope(),
            crate::ProcessCancelSource::Tool,
        )
        .with_reason("requested by tool");
        self.process_cancel_ability
            .cancel_summary(self.processes.as_ref(), request)
            .await
    }

    pub async fn signal(
        &self,
        process_id: &str,
        payload: serde_json::Value,
    ) -> Result<crate::ProcessEvent, PluginError> {
        let signal_id = self
            .tool_call_id
            .clone()
            .unwrap_or_else(|| format!("adhoc-{}", uuid::Uuid::new_v4()));
        self.processes
            .signal(
                &self.session_id,
                process_id,
                signal_id,
                payload,
                self.process_scope(),
            )
            .await
    }

    pub async fn cancel_all(&self) -> Result<Vec<crate::ProcessCancelSummary>, PluginError> {
        self.process_cancel_ability
            .cancel_all_visible(
                self.processes.as_ref(),
                crate::ProcessCancelAllRequest::new(
                    &self.session_id,
                    self.process_scope(),
                    crate::ProcessCancelSource::Tool,
                )
                .with_reason("requested by tool"),
            )
            .await
    }

    pub async fn transfer_handles(
        &self,
        to_session_id: &str,
        process_ids: Vec<String>,
    ) -> Result<(), PluginError> {
        self.processes
            .transfer(
                &self.session_id,
                to_session_id,
                process_ids,
                self.process_scope(),
            )
            .await
    }

    pub async fn transfer_handles_to_frame(
        &self,
        to_agent_frame_id: &str,
        process_ids: Vec<String>,
    ) -> Result<(), PluginError> {
        self.processes
            .transfer(
                &self.session_id,
                &self.session_id,
                process_ids,
                self.process_scope()
                    .with_target_agent_frame_id(Some(to_agent_frame_id.to_string())),
            )
            .await
    }

    pub async fn cancel_unreferenced_handles(
        &self,
        keep_process_ids: Vec<String>,
    ) -> Result<Vec<crate::ProcessCancelSummary>, PluginError> {
        Ok(self
            .processes
            .cancel_unreferenced(&self.session_id, keep_process_ids, self.process_scope())
            .await?
            .into_iter()
            .map(crate::ProcessCancelSummary::from_record)
            .collect())
    }
}
