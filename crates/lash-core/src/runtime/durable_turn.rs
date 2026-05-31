use std::future::Future;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use super::{
    RUNTIME_TURN_LEASE_TTL_MS, RuntimeEffectController, RuntimeEffectControllerError,
    RuntimeEffectEnvelope, RuntimeEffectLocalExecutor, RuntimeEffectOutcome, RuntimeError,
    RuntimeErrorCode, RuntimeInvocation,
};
use crate::store::{
    RuntimeCommit, RuntimeCommitResult, RuntimeEffectJournalRecord, RuntimePersistence,
    RuntimeTurnCheckpoint, RuntimeTurnCompletion, RuntimeTurnLease, StoreError,
};

fn current_epoch_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[async_trait::async_trait]
pub trait EmbeddedDurableTurnStore: Send + Sync {
    async fn claim_runtime_turn_lease(
        &self,
        session_id: &str,
        turn_id: &str,
        owner_id: &str,
        lease_ttl_ms: u64,
    ) -> Result<RuntimeTurnLease, StoreError>;

    async fn renew_runtime_turn_lease(
        &self,
        lease: &RuntimeTurnLease,
        lease_ttl_ms: u64,
    ) -> Result<RuntimeTurnLease, StoreError>;

    async fn abandon_runtime_turn_lease(&self, lease: &RuntimeTurnLease) -> Result<(), StoreError>;

    async fn save_runtime_turn_checkpoint(
        &self,
        lease: &RuntimeTurnLease,
        checkpoint: RuntimeTurnCheckpoint,
    ) -> Result<(), StoreError>;

    async fn load_runtime_turn_checkpoint(
        &self,
        session_id: &str,
        turn_id: &str,
    ) -> Result<Option<RuntimeTurnCheckpoint>, StoreError>;

    async fn save_runtime_effect_outcome(
        &self,
        lease: &RuntimeTurnLease,
        record: RuntimeEffectJournalRecord,
    ) -> Result<(), StoreError>;

    async fn load_runtime_effect_outcome(
        &self,
        session_id: &str,
        turn_id: &str,
        replay_key: &str,
    ) -> Result<Option<RuntimeEffectJournalRecord>, StoreError>;
}

pub struct BeginDurableTurnRequest {
    pub session_id: String,
    pub turn_id: String,
    pub owner_id: String,
    pub store: Option<Arc<dyn RuntimePersistence>>,
}

pub struct ResumeDurableTurnRequest {
    pub session_id: String,
    pub turn_id: String,
    pub owner_id: String,
    pub expected_provider_id: String,
    pub store: Arc<dyn RuntimePersistence>,
}

pub struct DurableTurnCheckpointSnapshot {
    record: Option<RuntimeTurnCheckpoint>,
}

impl DurableTurnCheckpointSnapshot {
    pub fn none() -> Self {
        Self { record: None }
    }

    pub fn persisted(record: RuntimeTurnCheckpoint) -> Self {
        Self {
            record: Some(record),
        }
    }
}

pub struct DurableTurnRun {
    session_id: String,
    turn_id: String,
    store: Option<Arc<dyn RuntimePersistence>>,
    lease: Option<RuntimeTurnLease>,
    checkpoint: Option<RuntimeTurnCheckpoint>,
    substrate_native: bool,
}

impl DurableTurnRun {
    pub fn substrate_native(session_id: impl Into<String>, turn_id: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
            turn_id: turn_id.into(),
            store: None,
            lease: None,
            checkpoint: None,
            substrate_native: true,
        }
    }

    fn embedded(
        session_id: String,
        turn_id: String,
        store: Option<Arc<dyn RuntimePersistence>>,
        lease: Option<RuntimeTurnLease>,
        checkpoint: Option<RuntimeTurnCheckpoint>,
    ) -> Self {
        Self {
            session_id,
            turn_id,
            store,
            lease,
            checkpoint,
            substrate_native: false,
        }
    }

    pub(crate) fn placeholder() -> Self {
        Self {
            session_id: String::new(),
            turn_id: String::new(),
            store: None,
            lease: None,
            checkpoint: None,
            substrate_native: false,
        }
    }

    pub fn turn_id(&self) -> &str {
        &self.turn_id
    }

    pub fn lease(&self) -> Option<&RuntimeTurnLease> {
        self.lease.as_ref()
    }

    pub fn take_checkpoint(&mut self) -> Option<RuntimeTurnCheckpoint> {
        self.checkpoint.take()
    }
}

#[async_trait::async_trait]
pub trait DurableTurnProvider: Send + Sync {
    fn durability_tier(&self) -> crate::DurabilityTier;

    fn requires_durable_attachment_store(&self) -> bool {
        false
    }

    async fn begin_turn(
        &self,
        request: BeginDurableTurnRequest,
    ) -> Result<DurableTurnRun, RuntimeError>;

    async fn resume_turn(
        &self,
        request: ResumeDurableTurnRequest,
    ) -> Result<DurableTurnRun, RuntimeError>;

    async fn execute_effect(
        &self,
        run: &mut DurableTurnRun,
        checkpoint: DurableTurnCheckpointSnapshot,
        envelope: RuntimeEffectEnvelope,
        local_executor: RuntimeEffectLocalExecutor<'_>,
    ) -> Result<RuntimeEffectOutcome, RuntimeEffectControllerError>;

    async fn finalize_turn(
        &self,
        run: DurableTurnRun,
        mut commit: RuntimeCommit,
        store: &dyn RuntimePersistence,
    ) -> Result<RuntimeCommitResult, StoreError> {
        let turn_commit_hash = commit.turn_commit_hash()?;
        let lease = run.lease.clone();
        commit.completed_turn = if let Some(lease) = run.lease.as_ref() {
            Some(RuntimeTurnCompletion::from_lease(lease, turn_commit_hash))
        } else if run.substrate_native {
            Some(RuntimeTurnCompletion::substrate_native(
                run.session_id,
                run.turn_id,
                turn_commit_hash,
            ))
        } else {
            None
        };
        let result = store.commit_runtime_state(commit).await;
        if result.is_err()
            && let Some(lease) = lease.as_ref()
            && let Some(embedded) = store.embedded_durable_turn_store()
        {
            let _ = embedded.abandon_runtime_turn_lease(lease).await;
        }
        result
    }

    async fn abandon_turn(&self, run: DurableTurnRun, reason: &str);
}

pub struct EmbeddedDurableTurnProvider<'run> {
    controller: &'run dyn RuntimeEffectController,
}

impl<'run> EmbeddedDurableTurnProvider<'run> {
    pub fn new(controller: &'run dyn RuntimeEffectController) -> Self {
        Self { controller }
    }

    fn embedded_store<'a>(
        store: &'a dyn RuntimePersistence,
    ) -> Result<&'a dyn EmbeddedDurableTurnStore, RuntimeEffectControllerError> {
        store.embedded_durable_turn_store().ok_or_else(|| {
            RuntimeEffectControllerError::new(
                "embedded_durable_turn_store_unavailable",
                "runtime store does not expose embedded durable turn",
            )
        })
    }

    fn embedded_store_runtime<'a>(
        store: &'a dyn RuntimePersistence,
    ) -> Result<&'a dyn EmbeddedDurableTurnStore, RuntimeError> {
        store.embedded_durable_turn_store().ok_or_else(|| {
            RuntimeError::new(
                RuntimeErrorCode::RuntimeTurnLease,
                "runtime store does not expose embedded durable turn",
            )
        })
    }
}

#[async_trait::async_trait]
impl DurableTurnProvider for EmbeddedDurableTurnProvider<'_> {
    fn durability_tier(&self) -> crate::DurabilityTier {
        self.controller.durability_tier()
    }

    fn requires_durable_attachment_store(&self) -> bool {
        self.controller.requires_durable_attachment_store()
    }

    async fn begin_turn(
        &self,
        request: BeginDurableTurnRequest,
    ) -> Result<DurableTurnRun, RuntimeError> {
        let lease = if let Some(store) = request.store.as_ref() {
            let embedded = Self::embedded_store_runtime(store.as_ref())?;
            Some(
                embedded
                    .claim_runtime_turn_lease(
                        &request.session_id,
                        &request.turn_id,
                        &request.owner_id,
                        RUNTIME_TURN_LEASE_TTL_MS,
                    )
                    .await
                    .map_err(|err| {
                        RuntimeError::new(RuntimeErrorCode::RuntimeTurnLease, err.to_string())
                    })?,
            )
        } else {
            None
        };
        Ok(DurableTurnRun::embedded(
            request.session_id,
            request.turn_id,
            request.store,
            lease,
            None,
        ))
    }

    async fn resume_turn(
        &self,
        request: ResumeDurableTurnRequest,
    ) -> Result<DurableTurnRun, RuntimeError> {
        let embedded = Self::embedded_store_runtime(request.store.as_ref())?;
        let checkpoint = embedded
            .load_runtime_turn_checkpoint(&request.session_id, &request.turn_id)
            .await
            .map_err(|err| {
                RuntimeError::new(RuntimeErrorCode::RuntimeTurnCheckpointLoad, err.to_string())
            })?
            .ok_or_else(|| {
                RuntimeError::new(
                    RuntimeErrorCode::RuntimeTurnCheckpointMissing,
                    format!(
                        "no in-flight runtime turn checkpoint found for `{}`",
                        request.turn_id
                    ),
                )
            })?;
        if checkpoint.provider_id != request.expected_provider_id {
            return Err(RuntimeError::new(
                RuntimeErrorCode::RuntimeTurnResumeProviderMismatch,
                format!(
                    "checkpoint requires provider `{}`, current runtime has `{}`",
                    checkpoint.provider_id, request.expected_provider_id
                ),
            ));
        }
        let lease = embedded
            .claim_runtime_turn_lease(
                &request.session_id,
                &request.turn_id,
                &request.owner_id,
                RUNTIME_TURN_LEASE_TTL_MS,
            )
            .await
            .map_err(|err| {
                RuntimeError::new(RuntimeErrorCode::RuntimeTurnLease, err.to_string())
            })?;
        Ok(DurableTurnRun::embedded(
            request.session_id,
            request.turn_id,
            Some(request.store),
            Some(lease),
            Some(checkpoint),
        ))
    }

    async fn execute_effect(
        &self,
        run: &mut DurableTurnRun,
        checkpoint: DurableTurnCheckpointSnapshot,
        envelope: RuntimeEffectEnvelope,
        local_executor: RuntimeEffectLocalExecutor<'_>,
    ) -> Result<RuntimeEffectOutcome, RuntimeEffectControllerError> {
        let Some(turn_id) = envelope.invocation.scope.turn_id.clone() else {
            return self
                .controller
                .execute_effect(envelope, local_executor)
                .await;
        };
        let Some(store) = run.store.as_ref() else {
            return self
                .controller
                .execute_effect(envelope, local_executor)
                .await;
        };
        if run.session_id != envelope.invocation.scope.session_id || run.turn_id != turn_id {
            return Err(RuntimeEffectControllerError::new(
                "runtime_durable_turn_mismatch",
                format!(
                    "effect targets `{}`/`{turn_id}` but durability run targets `{}`/`{}`",
                    envelope.invocation.scope.session_id, run.session_id, run.turn_id
                ),
            ));
        }
        let embedded = Self::embedded_store(store.as_ref())?;
        let mut lease = require_matching_turn_lease(run.lease.as_ref(), &envelope.invocation)?;

        if let Some(record) = checkpoint.record {
            lease =
                renew_runtime_turn_lease_for_effect(embedded, &lease, &envelope.invocation).await?;
            validate_checkpoint_hash(&record, &envelope.invocation)?;
            embedded
                .save_runtime_turn_checkpoint(&lease, record)
                .await
                .map_err(RuntimeEffectControllerError::from)?;
        }

        let outcome = execute_embedded_journaled_effect(
            embedded,
            &mut lease,
            self.controller,
            envelope,
            local_executor,
        )
        .await?;
        run.lease = Some(lease);
        Ok(outcome)
    }

    async fn abandon_turn(&self, run: DurableTurnRun, reason: &str) {
        let (Some(store), Some(lease)) = (run.store.as_ref(), run.lease.as_ref()) else {
            return;
        };
        let Some(embedded) = store.embedded_durable_turn_store() else {
            return;
        };
        if let Err(err) = embedded.abandon_runtime_turn_lease(lease).await {
            tracing::warn!(
                session_id = %lease.session_id,
                turn_id = %lease.turn_id,
                reason,
                "failed to abandon runtime turn lease: {err}"
            );
        }
    }
}

pub struct SubstrateDurableTurnProvider<'run> {
    controller: &'run dyn RuntimeEffectController,
}

impl<'run> SubstrateDurableTurnProvider<'run> {
    pub fn new(controller: &'run dyn RuntimeEffectController) -> Self {
        Self { controller }
    }
}

#[async_trait::async_trait]
impl DurableTurnProvider for SubstrateDurableTurnProvider<'_> {
    fn durability_tier(&self) -> crate::DurabilityTier {
        self.controller.durability_tier()
    }

    fn requires_durable_attachment_store(&self) -> bool {
        self.controller.requires_durable_attachment_store()
    }

    async fn begin_turn(
        &self,
        request: BeginDurableTurnRequest,
    ) -> Result<DurableTurnRun, RuntimeError> {
        Ok(DurableTurnRun::substrate_native(
            request.session_id,
            request.turn_id,
        ))
    }

    async fn resume_turn(
        &self,
        _request: ResumeDurableTurnRequest,
    ) -> Result<DurableTurnRun, RuntimeError> {
        Err(RuntimeError::new(
            RuntimeErrorCode::RuntimeTurnResumeStoreRequired,
            "substrate-native durable turn is resumed by replaying the provider handler, not by loading a Lash checkpoint",
        ))
    }

    async fn execute_effect(
        &self,
        _run: &mut DurableTurnRun,
        _checkpoint: DurableTurnCheckpointSnapshot,
        envelope: RuntimeEffectEnvelope,
        local_executor: RuntimeEffectLocalExecutor<'_>,
    ) -> Result<RuntimeEffectOutcome, RuntimeEffectControllerError> {
        self.controller
            .execute_effect(envelope, local_executor)
            .await
    }

    async fn abandon_turn(&self, _run: DurableTurnRun, _reason: &str) {}
}

pub(crate) async fn execute_embedded_journaled_effect(
    store: &(dyn EmbeddedDurableTurnStore + '_),
    active_lease: &mut RuntimeTurnLease,
    controller: &dyn RuntimeEffectController,
    envelope: RuntimeEffectEnvelope,
    local_executor: RuntimeEffectLocalExecutor<'_>,
) -> Result<RuntimeEffectOutcome, RuntimeEffectControllerError> {
    let turn_id = envelope.invocation.scope.turn_id.clone().ok_or_else(|| {
        RuntimeEffectControllerError::new(
            "runtime_turn_lease_required",
            "runtime effect envelope requires a turn id for embedded durability",
        )
    })?;
    require_matching_turn_lease(Some(active_lease), &envelope.invocation)?;
    let envelope_hash = envelope.stable_hash()?;
    let replay_key = envelope
        .invocation
        .replay_key()
        .ok_or_else(|| {
            RuntimeEffectControllerError::new(
                "runtime_effect_replay_required",
                "runtime effect envelope requires replay.key",
            )
        })?
        .to_string();
    if let Some(record) = store
        .load_runtime_effect_outcome(&envelope.invocation.scope.session_id, &turn_id, &replay_key)
        .await
        .map_err(RuntimeEffectControllerError::from)?
    {
        if record.envelope_hash != envelope_hash {
            return Err(RuntimeEffectControllerError::new(
                "runtime_effect_journal_hash_mismatch",
                format!(
                    "recorded runtime effect `{}` has envelope hash `{}` but replay requested `{}`",
                    replay_key, record.envelope_hash, envelope_hash
                ),
            ));
        }
        return Ok(record.outcome);
    }

    let invocation = envelope.invocation.clone();
    let effect_kind = invocation.effect_kind().ok_or_else(|| {
        RuntimeEffectControllerError::new(
            "runtime_effect_invocation_subject",
            "runtime effect envelope subject must be an effect",
        )
    })?;
    let outcome = execute_pending_effect_with_lease_renewal(
        store,
        active_lease,
        &invocation,
        controller,
        envelope,
        local_executor,
    )
    .await?;
    *active_lease = renew_runtime_turn_lease_for_effect(store, active_lease, &invocation).await?;
    store
        .save_runtime_effect_outcome(
            active_lease,
            RuntimeEffectJournalRecord {
                schema_version: crate::RUNTIME_EFFECT_JOURNAL_SCHEMA_VERSION,
                session_id: invocation.scope.session_id,
                turn_id,
                replay_key,
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

pub(crate) async fn invoke_embedded_journaled_effect<T, E, F, Fut>(
    store: Option<&(dyn EmbeddedDurableTurnStore + '_)>,
    lease: Option<&RuntimeTurnLease>,
    controller: &dyn RuntimeEffectController,
    envelope: RuntimeEffectEnvelope,
    local_executor: RuntimeEffectLocalExecutor<'_>,
    apply_outcome: F,
) -> Result<T, E>
where
    E: From<RuntimeEffectControllerError>,
    F: FnOnce(RuntimeEffectOutcome) -> Fut,
    Fut: Future<Output = Result<T, E>>,
{
    let outcome = if let (Some(store), Some(lease), Some(_)) =
        (store, lease, envelope.invocation.scope.turn_id.as_ref())
    {
        let mut lease = lease.clone();
        execute_embedded_journaled_effect(store, &mut lease, controller, envelope, local_executor)
            .await
            .map_err(E::from)?
    } else {
        controller
            .execute_effect(envelope, local_executor)
            .await
            .map_err(E::from)?
    };
    apply_outcome(outcome).await
}

pub(crate) async fn renew_runtime_turn_lease_for_effect(
    store: &(dyn EmbeddedDurableTurnStore + '_),
    lease: &RuntimeTurnLease,
    invocation: &RuntimeInvocation,
) -> Result<RuntimeTurnLease, RuntimeEffectControllerError> {
    require_matching_turn_lease(Some(lease), invocation)?;
    store
        .renew_runtime_turn_lease(lease, RUNTIME_TURN_LEASE_TTL_MS)
        .await
        .map_err(RuntimeEffectControllerError::from)
}

async fn execute_pending_effect_with_lease_renewal(
    store: &(dyn EmbeddedDurableTurnStore + '_),
    active_lease: &mut RuntimeTurnLease,
    invocation: &RuntimeInvocation,
    controller: &dyn RuntimeEffectController,
    envelope: RuntimeEffectEnvelope,
    local_executor: RuntimeEffectLocalExecutor<'_>,
) -> Result<RuntimeEffectOutcome, RuntimeEffectControllerError> {
    let pending = controller.execute_effect(envelope, local_executor);
    tokio::pin!(pending);
    loop {
        tokio::select! {
            outcome = &mut pending => return outcome,
            _ = tokio::time::sleep(pending_effect_lease_renew_interval()) => {
                *active_lease =
                    renew_runtime_turn_lease_for_effect(store, active_lease, invocation).await?;
            }
        }
    }
}

fn pending_effect_lease_renew_interval() -> Duration {
    Duration::from_millis(pending_effect_lease_renew_interval_ms())
}

#[cfg(test)]
fn pending_effect_lease_renew_interval_ms() -> u64 {
    25
}

#[cfg(not(test))]
fn pending_effect_lease_renew_interval_ms() -> u64 {
    30_000
}

fn validate_checkpoint_hash(
    checkpoint: &RuntimeTurnCheckpoint,
    invocation: &RuntimeInvocation,
) -> Result<(), RuntimeEffectControllerError> {
    let expected = invocation.checkpoint_hash.as_deref();
    if expected != Some(checkpoint.checkpoint_hash.as_str()) {
        return Err(RuntimeEffectControllerError::new(
            "runtime_turn_checkpoint_hash_mismatch",
            format!(
                "effect `{}` expected checkpoint hash {:?}, computed `{}`",
                invocation.effect_id().unwrap_or("<unknown>"),
                invocation.checkpoint_hash,
                checkpoint.checkpoint_hash
            ),
        ));
    }
    Ok(())
}

fn require_matching_turn_lease(
    lease: Option<&RuntimeTurnLease>,
    invocation: &RuntimeInvocation,
) -> Result<RuntimeTurnLease, RuntimeEffectControllerError> {
    let replay_key = invocation.replay_key().unwrap_or("<missing replay>");
    let Some(turn_id) = invocation.scope.turn_id.as_deref() else {
        return Err(RuntimeEffectControllerError::new(
            "runtime_turn_lease_required",
            format!(
                "runtime effect `{}` does not carry a turn id for lease validation",
                replay_key
            ),
        ));
    };
    let Some(lease) = lease else {
        return Err(RuntimeEffectControllerError::new(
            "runtime_turn_lease_required",
            format!(
                "runtime effect `{}` for turn `{}` requires a runtime turn lease",
                replay_key, turn_id
            ),
        ));
    };
    if lease.session_id != invocation.scope.session_id || lease.turn_id != turn_id {
        return Err(RuntimeEffectControllerError::new(
            "runtime_turn_lease_required",
            format!(
                "runtime effect `{}` lease targets `{}`/`{}` but metadata targets `{}`/`{}`",
                replay_key, lease.session_id, lease.turn_id, invocation.scope.session_id, turn_id
            ),
        ));
    }
    Ok(lease.clone())
}
