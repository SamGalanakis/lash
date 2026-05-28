use super::*;
use crate::PluginError;

impl<'run> RuntimeTurnDriver<'run> {
    pub(super) fn effect_controller_handle(&self) -> RuntimeEffectControllerHandle<'run> {
        RuntimeEffectControllerHandle::borrowed(self.effect_scope.controller())
    }

    pub(super) fn execution_context(
        &self,
        event_tx: mpsc::Sender<SessionEvent>,
        chronological_projection: Arc<crate::ChronologicalProjection>,
    ) -> Result<crate::RuntimeExecutionContext<'run>, PluginError> {
        let manager = self.session_manager.clone();
        let effect_controller = self.effect_controller_handle();
        let direct_completions = manager.direct_completion_client(
            effect_controller.clone_scoped(),
            Some(self.turn_id.clone()),
            self.turn_lease.clone(),
        );
        self.session.code_execution_context(
            &self.session_id,
            &self.turn_pipeline.state().current_agent_frame_id,
            manager.clone() as Arc<dyn crate::plugin::RuntimeSessionHost>,
            manager as Arc<dyn crate::ProcessService>,
            effect_controller,
            direct_completions,
            event_tx,
            chronological_projection,
            self.protocol_extension.clone(),
            self.turn_context.clone(),
            self.checkpoint_messages.clone(),
        )
    }
}
