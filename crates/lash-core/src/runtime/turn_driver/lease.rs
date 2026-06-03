use super::*;

impl<'run> RuntimeTurnDriver<'run> {
    pub(super) fn turn_effect_invocation(
        &self,
        machine: &TurnMachine,
        effect_id: crate::sansio::EffectId,
        effect_kind: RuntimeEffectKind,
    ) -> Result<RuntimeInvocation, RuntimeEffectControllerError> {
        Ok(crate::runtime::causal::turn_effect_invocation(
            &self.session_id,
            &self.turn_id,
            self.turn_index,
            machine.protocol_iteration(),
            effect_id,
            effect_kind,
        ))
    }

    pub(super) async fn execute_typed_turn_effect<T>(
        &mut self,
        machine: &mut TurnMachine,
        event_tx: &mpsc::Sender<RuntimeStreamEvent>,
        cancel: &CancellationToken,
        envelope: RuntimeEffectEnvelope,
        decode: impl FnOnce(RuntimeEffectOutcome) -> Result<T, RuntimeEffectControllerError>,
    ) -> Result<T, RuntimeEffectControllerError> {
        let scoped_effect_controller = self.scoped_effect_controller.clone();
        let local_executor = crate::RuntimeEffectLocalExecutor::turn(
            self,
            machine,
            event_tx.clone(),
            cancel.clone(),
        );
        let outcome = scoped_effect_controller
            .controller()
            .execute_effect(envelope, local_executor)
            .await?;
        decode(outcome)
    }
}
