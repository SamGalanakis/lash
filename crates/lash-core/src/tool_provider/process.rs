use std::sync::Arc;

use crate::ProcessRecord;
use crate::plugin::PluginError;

#[derive(Clone)]
pub struct ToolProcessControl<'run> {
    pub(super) session_id: String,
    pub(super) agent_frame_id: crate::AgentFrameId,
    pub(super) processes: Arc<dyn crate::ProcessService>,
    pub(super) effect_controller: crate::runtime::RuntimeEffectControllerHandle<'run>,
    pub(super) parent_invocation: Option<crate::RuntimeInvocation>,
}

impl ToolProcessControl<'_> {
    fn process_scope(&self) -> crate::ProcessOpScope<'_> {
        crate::ProcessOpScope::new()
            .with_parent_invocation(self.parent_invocation.clone())
            .with_effect_controller(self.effect_controller.as_controller())
            .with_agent_frame_id(Some(self.agent_frame_id.clone()))
    }

    /// Start a process owned by this session and registered to wake it,
    /// returning its durable record. Routes through the same
    /// [`crate::ProcessService::start`] path the runtime uses for every other
    /// process start, so the child is provider-re-supplied, durable, and
    /// recoverable through the worker.
    pub async fn start(
        &self,
        registration: crate::ProcessRegistration,
        descriptor: crate::ProcessHandleDescriptor,
    ) -> Result<crate::ProcessRecord, PluginError> {
        self.processes
            .start(
                &self.session_id,
                registration,
                crate::ProcessStartOptions::new()
                    .with_descriptor(descriptor),
                self.process_scope(),
            )
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

    pub async fn list_handles(&self) -> Result<Vec<crate::ProcessHandleGrantEntry>, PluginError> {
        self.processes
            .list_visible(&self.session_id, self.process_scope())
            .await
    }

    pub async fn validate_handles(&self, handle_ids: &[String]) -> Result<(), PluginError> {
        self.processes
            .validate_visible(&self.session_id, handle_ids, self.process_scope())
            .await
    }

    pub async fn cancel(&self, process_id: &str) -> Result<ProcessRecord, PluginError> {
        self.processes
            .cancel(&self.session_id, process_id, self.process_scope())
            .await
    }

    pub async fn cancel_all(&self) -> Result<Vec<ProcessRecord>, PluginError> {
        self.processes
            .cancel_all(&self.session_id, self.process_scope())
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
    ) -> Result<Vec<ProcessRecord>, PluginError> {
        self.processes
            .cancel_unreferenced(&self.session_id, keep_process_ids, self.process_scope())
            .await
    }
}
