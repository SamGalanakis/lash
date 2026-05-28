use std::sync::Arc;

use crate::ProcessRecord;
use crate::plugin::PluginError;

#[derive(Clone)]
pub struct ToolProcessControl<'run> {
    pub(super) session_id: String,
    pub(super) processes: Arc<dyn crate::ProcessService>,
    pub(super) effect_controller: crate::runtime::RuntimeEffectControllerHandle<'run>,
    pub(super) parent_invocation: Option<crate::RuntimeInvocation>,
}

impl ToolProcessControl<'_> {
    fn process_scope(&self) -> crate::ProcessOpScope<'_> {
        crate::ProcessOpScope::new()
            .with_parent_invocation(self.parent_invocation.clone())
            .with_effect_controller(self.effect_controller.as_controller())
    }

    pub async fn list_handles(&self) -> Result<Vec<crate::ProcessHandleGrantEntry>, PluginError> {
        self.processes
            .list_visible(&self.session_id, self.process_scope())
            .await
    }

    pub async fn validate_handles(&self, handle_ids: &[String]) -> Result<(), PluginError> {
        self.processes
            .validate_visible(&self.session_id, handle_ids)
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

    pub async fn cancel_unreferenced_handles(
        &self,
        keep_process_ids: Vec<String>,
    ) -> Result<Vec<ProcessRecord>, PluginError> {
        self.processes
            .cancel_unreferenced(&self.session_id, keep_process_ids, self.process_scope())
            .await
    }
}
