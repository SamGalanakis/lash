use std::time::{SystemTime, UNIX_EPOCH};

use super::*;

fn current_epoch_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

impl<'run> RuntimeTurnDriver<'run> {
    pub(super) fn turn_effect_invocation(
        &self,
        machine: &TurnMachine,
        effect_id: crate::sansio::EffectId,
        effect_kind: RuntimeEffectKind,
    ) -> Result<RuntimeInvocation, RuntimeEffectControllerError> {
        let checkpoint_hash = crate::runtime_turn_checkpoint_hash(&machine.checkpoint())
            .map_err(RuntimeEffectControllerError::from)?;
        Ok(crate::runtime::causal::turn_effect_invocation(
            &self.session_id,
            &self.turn_id,
            self.turn_index,
            machine.protocol_iteration(),
            effect_id,
            effect_kind,
            Some(checkpoint_hash),
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
        self.persist_turn_checkpoint_for_effect(machine, &envelope.invocation)
            .await?;
        let effect_scope = self.effect_scope;
        let controller = effect_scope.controller();
        let store = self.session.history_store();
        let turn_lease = self.turn_lease.clone();
        let local_executor = crate::RuntimeEffectLocalExecutor::turn(
            self,
            machine,
            event_tx.clone(),
            cancel.clone(),
        );
        crate::runtime::effect::invoke_journaled_effect(
            crate::runtime::effect::JournaledEffectInvocation::new(
                store.as_ref().map(|store| store.as_ref()),
                turn_lease.as_ref(),
                controller,
                envelope,
                local_executor,
            ),
            |outcome| async move { decode(outcome) },
        )
        .await
    }

    async fn persist_turn_checkpoint_for_effect(
        &mut self,
        machine: &TurnMachine,
        invocation: &RuntimeInvocation,
    ) -> Result<(), RuntimeEffectControllerError> {
        let Some(store) = self.session.history_store() else {
            return Ok(());
        };
        let Some(lease) = self.turn_lease.clone() else {
            return Err(RuntimeEffectControllerError::new(
                "runtime_turn_lease_required",
                format!(
                    "runtime effect `{}` for turn `{}` requires a runtime turn lease",
                    invocation.replay_key().unwrap_or("<missing replay>"),
                    self.turn_id
                ),
            ));
        };
        let renewed_lease = crate::runtime::effect::renew_runtime_turn_lease_for_effect(
            store.as_ref(),
            &lease,
            invocation,
        )
        .await?;
        let checkpoint = machine.checkpoint();
        let checkpoint_hash = crate::runtime_turn_checkpoint_hash(&checkpoint)
            .map_err(RuntimeEffectControllerError::from)?;
        if invocation.checkpoint_hash.as_deref() != Some(checkpoint_hash.as_str()) {
            return Err(RuntimeEffectControllerError::new(
                "runtime_turn_checkpoint_hash_mismatch",
                format!(
                    "effect `{}` expected checkpoint hash {:?}, computed `{}`",
                    invocation.effect_id().unwrap_or("<unknown>"),
                    invocation.checkpoint_hash,
                    checkpoint_hash
                ),
            ));
        }
        let Some(machine_config) = self.machine_config_snapshot.clone() else {
            return Err(RuntimeEffectControllerError::new(
                "runtime_turn_checkpoint_config_missing",
                "cannot persist runtime turn checkpoint without machine config snapshot",
            ));
        };
        let record = crate::RuntimeTurnCheckpoint {
            schema_version: crate::RUNTIME_TURN_CHECKPOINT_SCHEMA_VERSION,
            session_id: self.session_id.clone(),
            turn_id: self.turn_id.clone(),
            turn_index: self.turn_index,
            protocol_iteration: machine.protocol_iteration(),
            checkpoint_hash,
            machine_config,
            checkpoint,
            protocol_turn_options: self.protocol_turn_options.clone(),
            turn_prompt_layer: self.turn_context.prompt_layer().clone(),
            provider_id: self.policy.recorded_provider_id().to_string(),
            model: self.policy.model.clone(),
            updated_at_epoch_ms: current_epoch_ms(),
        };
        store
            .save_runtime_turn_checkpoint(&renewed_lease, record)
            .await
            .map_err(RuntimeEffectControllerError::from)?;
        self.turn_lease = Some(renewed_lease);
        Ok(())
    }
}
