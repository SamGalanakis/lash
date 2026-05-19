use std::time::{SystemTime, UNIX_EPOCH};

use super::controller::{RuntimeEffectController, RuntimeEffectControllerError};
use super::envelope::{
    EffectInvocationMetadata, EffectOrigin, RuntimeEffectEnvelope, RuntimeEffectOutcome,
};
use super::local::RuntimeEffectLocalExecutor;

pub(crate) async fn renew_runtime_turn_lease_for_effect(
    store: &(dyn crate::RuntimePersistence + '_),
    lease: &crate::RuntimeTurnLease,
    metadata: &EffectInvocationMetadata,
) -> Result<crate::RuntimeTurnLease, RuntimeEffectControllerError> {
    require_matching_turn_lease(Some(lease), metadata)?;
    store
        .renew_runtime_turn_lease(lease, crate::runtime::RUNTIME_TURN_LEASE_TTL_MS)
        .await
        .map_err(RuntimeEffectControllerError::from)
}

pub(crate) async fn execute_effect_with_journal(
    store: Option<&(dyn crate::RuntimePersistence + '_)>,
    lease: Option<&crate::RuntimeTurnLease>,
    controller: &dyn RuntimeEffectController,
    envelope: RuntimeEffectEnvelope,
    local_executor: RuntimeEffectLocalExecutor<'_>,
) -> Result<RuntimeEffectOutcome, RuntimeEffectControllerError> {
    let Some(turn_id) = envelope.metadata.turn_id.clone() else {
        return controller.execute_effect(envelope, local_executor).await;
    };
    let Some(store) = store else {
        return controller.execute_effect(envelope, local_executor).await;
    };
    let mut active_lease = require_matching_turn_lease(lease, &envelope.metadata)?;
    if !matches!(envelope.metadata.origin, EffectOrigin::Turn) {
        active_lease =
            renew_runtime_turn_lease_for_effect(store, &active_lease, &envelope.metadata).await?;
    }
    let envelope_hash = envelope.stable_hash()?;
    if let Some(record) = store
        .load_runtime_effect_outcome(
            &envelope.metadata.session_id,
            &turn_id,
            &envelope.metadata.idempotency_key,
        )
        .await
        .map_err(RuntimeEffectControllerError::from)?
    {
        if record.envelope_hash != envelope_hash {
            return Err(RuntimeEffectControllerError::new(
                "runtime_effect_journal_hash_mismatch",
                format!(
                    "recorded runtime effect `{}` has envelope hash `{}` but replay requested `{}`",
                    envelope.metadata.idempotency_key, record.envelope_hash, envelope_hash
                ),
            ));
        }
        return Ok(record.outcome);
    }

    let metadata = envelope.metadata.clone();
    let effect_kind = metadata.effect_kind;
    let outcome = controller.execute_effect(envelope, local_executor).await?;
    active_lease = renew_runtime_turn_lease_for_effect(store, &active_lease, &metadata).await?;
    store
        .save_runtime_effect_outcome(
            &active_lease,
            crate::RuntimeEffectJournalRecord {
                schema_version: crate::RUNTIME_EFFECT_JOURNAL_SCHEMA_VERSION,
                session_id: metadata.session_id,
                turn_id,
                idempotency_key: metadata.idempotency_key,
                envelope_hash,
                effect_kind,
                outcome: outcome.clone(),
                created_at_epoch_ms: current_epoch_ms(),
            },
        )
        .await
        .map_err(RuntimeEffectControllerError::from)?;
    Ok(outcome)
}

fn require_matching_turn_lease(
    lease: Option<&crate::RuntimeTurnLease>,
    metadata: &EffectInvocationMetadata,
) -> Result<crate::RuntimeTurnLease, RuntimeEffectControllerError> {
    let Some(turn_id) = metadata.turn_id.as_deref() else {
        return Err(RuntimeEffectControllerError::new(
            "runtime_turn_lease_required",
            format!(
                "runtime effect `{}` does not carry a turn id for lease validation",
                metadata.idempotency_key
            ),
        ));
    };
    let Some(lease) = lease else {
        return Err(RuntimeEffectControllerError::new(
            "runtime_turn_lease_required",
            format!(
                "runtime effect `{}` for turn `{}` requires a runtime turn lease",
                metadata.idempotency_key, turn_id
            ),
        ));
    };
    if lease.session_id != metadata.session_id || lease.turn_id != turn_id {
        return Err(RuntimeEffectControllerError::new(
            "runtime_turn_lease_required",
            format!(
                "runtime effect `{}` lease targets `{}`/`{}` but metadata targets `{}`/`{}`",
                metadata.idempotency_key,
                lease.session_id,
                lease.turn_id,
                metadata.session_id,
                turn_id
            ),
        ));
    }
    Ok(lease.clone())
}

fn current_epoch_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}
