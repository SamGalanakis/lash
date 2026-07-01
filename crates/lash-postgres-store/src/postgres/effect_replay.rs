use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

const POSTGRES_EFFECT_STATUS_IN_PROGRESS: &str = "in_progress";
const POSTGRES_EFFECT_STATUS_COMPLETED: &str = "completed";
const POSTGRES_EFFECT_STATUS_FAILED: &str = "failed";
const POSTGRES_EFFECT_DEFAULT_LEASE_TTL: Duration = Duration::from_secs(30);
const POSTGRES_EFFECT_BUSY_POLL: Duration = Duration::from_millis(25);

static POSTGRES_EFFECT_OWNER_COUNTER: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Debug)]
pub struct PostgresEffectReplayOptions {
    pub lease_ttl: Duration,
}

impl Default for PostgresEffectReplayOptions {
    fn default() -> Self {
        Self {
            lease_ttl: POSTGRES_EFFECT_DEFAULT_LEASE_TTL,
        }
    }
}

struct PostgresEffectReplayInner {
    pool: PgPool,
    owner_id: String,
    lease_counter: AtomicU64,
    replay_mode: AtomicBool,
    lease_ttl_ms: u64,
}

#[derive(Clone)]
pub struct PostgresEffectHost {
    inner: Arc<PostgresEffectReplayInner>,
}

#[derive(Clone)]
pub struct PostgresRuntimeEffectController {
    inner: Arc<PostgresEffectReplayInner>,
    scope: ExecutionScope,
}

struct PostgresClaimedEffect {
    scope_id: String,
    replay_key: String,
    envelope_hash: String,
    lease_token: String,
    due_at_ms: Option<u64>,
}

enum PostgresPreparedEffect {
    ReplayOutcome {
        outcome: Box<RuntimeEffectOutcome>,
        due_at_ms: Option<u64>,
    },
    ReplayError(RuntimeEffectControllerError),
    Claimed(PostgresClaimedEffect),
    Busy {
        retry_at_ms: u64,
    },
}

struct PostgresEffectRow {
    envelope_hash: String,
    status: String,
    outcome_json: Option<String>,
    error_json: Option<String>,
    lease_expires_at_ms: i64,
    due_at_ms: Option<i64>,
}

impl PostgresEffectHost {
    pub fn new(storage: &PostgresStorage) -> Self {
        Self::with_options(storage, PostgresEffectReplayOptions::default())
    }

    pub fn with_options(storage: &PostgresStorage, options: PostgresEffectReplayOptions) -> Self {
        Self {
            inner: Arc::new(PostgresEffectReplayInner::new(storage.pool.clone(), options)),
        }
    }

    pub fn start_replay(&self) {
        self.inner.replay_mode.store(true, Ordering::SeqCst);
    }
}

impl AwaitEventResolver for PostgresEffectHost {
    fn durability_tier(&self) -> DurabilityTier {
        DurabilityTier::Durable
    }

    fn requires_durable_attachment_store(&self) -> bool {
        true
    }

    fn supports_durable_effects(&self) -> bool {
        true
    }
}

impl EffectHost for PostgresEffectHost {
    fn scoped<'run>(
        &'run self,
        scope: ExecutionScope,
    ) -> Result<ScopedEffectController<'run>, RuntimeError> {
        let controller = PostgresRuntimeEffectController {
            inner: Arc::clone(&self.inner),
            scope: scope.clone(),
        };
        ScopedEffectController::shared(Arc::new(controller), scope)
    }

    fn scoped_static(
        &self,
        scope: ExecutionScope,
    ) -> Result<Option<ScopedEffectController<'static>>, RuntimeError> {
        let controller = PostgresRuntimeEffectController {
            inner: Arc::clone(&self.inner),
            scope: scope.clone(),
        };
        Ok(Some(ScopedEffectController::shared(
            Arc::new(controller),
            scope,
        )?))
    }
}

impl PostgresRuntimeEffectController {
    pub fn new(storage: &PostgresStorage, scope: ExecutionScope) -> Self {
        Self::with_options(storage, scope, PostgresEffectReplayOptions::default())
    }

    pub fn with_options(
        storage: &PostgresStorage,
        scope: ExecutionScope,
        options: PostgresEffectReplayOptions,
    ) -> Self {
        Self {
            inner: Arc::new(PostgresEffectReplayInner::new(storage.pool.clone(), options)),
            scope,
        }
    }

    pub fn start_replay(&self) {
        self.inner.replay_mode.store(true, Ordering::SeqCst);
    }

    async fn prepare_effect(
        &self,
        envelope: &RuntimeEffectEnvelope,
    ) -> Result<PostgresPreparedEffect, RuntimeEffectControllerError> {
        let replay_key = envelope
            .invocation
            .replay_key()
            .ok_or_else(|| {
                RuntimeEffectControllerError::new(
                    "postgres_effect_replay_key_missing",
                    "runtime effect envelope requires replay.key",
                )
            })?
            .to_string();
        let envelope_hash = envelope.stable_hash()?;
        let scope_id = self.scope.id().to_string();
        let now = current_epoch_ms();
        let lease_token = self.inner.next_lease_token();
        let due_at_ms = postgres_effect_sleep_due_at_ms(envelope, now);
        let lease_expires_at_ms = now.saturating_add(self.inner.lease_ttl_ms);
        let replay_mode = self.inner.replay_mode.load(Ordering::SeqCst);
        let owner_id = self.inner.owner_id.clone();

        let mut tx = self
            .inner
            .pool
            .begin()
            .await
            .map_err(postgres_effect_store_error)?;
        let outcome = match postgres_select_effect_row_for_update(&mut tx, &scope_id, &replay_key)
            .await?
        {
            Some(row) => {
                self.prepare_locked_effect_row(
                    &mut tx,
                    PostgresPrepareInputs {
                        row,
                        scope_id,
                        replay_key,
                        envelope_hash,
                        owner_id,
                        lease_token,
                        due_at_ms,
                        now,
                    },
                )
                .await
            }
            None if replay_mode => Err(RuntimeEffectControllerError::new(
                "postgres_effect_replay_missing",
                format!(
                    "no recorded runtime effect for scope `{scope_id}` and replay key `{replay_key}`"
                ),
            )),
            None => {
                let due_at_param = due_at_ms.map(|value| value as i64);
                let inserted = sqlx::query(
                    "INSERT INTO lash_runtime_effect_replay (
                        scope_id, replay_key, envelope_hash, status, outcome_json,
                        error_json, lease_owner_id, lease_token, lease_expires_at_ms,
                        due_at_ms, created_at_ms, updated_at_ms
                     )
                     VALUES ($1, $2, $3, $4, NULL, NULL, $5, $6, $7, $8, $9, $10)
                     ON CONFLICT (scope_id, replay_key) DO NOTHING",
                )
                .bind(&scope_id)
                .bind(&replay_key)
                .bind(&envelope_hash)
                .bind(POSTGRES_EFFECT_STATUS_IN_PROGRESS)
                .bind(&owner_id)
                .bind(&lease_token)
                .bind(lease_expires_at_ms as i64)
                .bind(due_at_param)
                .bind(now as i64)
                .bind(now as i64)
                .execute(&mut *tx)
                .await
                .map_err(postgres_effect_store_error)?
                .rows_affected();
                if inserted == 1 {
                    Ok(PostgresPreparedEffect::Claimed(PostgresClaimedEffect {
                        scope_id,
                        replay_key,
                        envelope_hash,
                        lease_token,
                        due_at_ms,
                    }))
                } else {
                    let row =
                        postgres_select_effect_row_for_update(&mut tx, &scope_id, &replay_key)
                            .await?
                            .ok_or_else(|| {
                                RuntimeEffectControllerError::new(
                                    "postgres_effect_replay_corrupt_row",
                                    "effect replay insert conflicted but no row could be selected",
                                )
                            })?;
                    self.prepare_locked_effect_row(
                        &mut tx,
                        PostgresPrepareInputs {
                            row,
                            scope_id,
                            replay_key,
                            envelope_hash,
                            owner_id,
                            lease_token,
                            due_at_ms,
                            now,
                        },
                    )
                    .await
                }
            }
        };
        tx.commit().await.map_err(postgres_effect_store_error)?;
        outcome
    }

    async fn prepare_locked_effect_row(
        &self,
        tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
        inputs: PostgresPrepareInputs,
    ) -> Result<PostgresPreparedEffect, RuntimeEffectControllerError> {
        let PostgresPrepareInputs {
            row,
            scope_id,
            replay_key,
            envelope_hash,
            owner_id,
            lease_token,
            due_at_ms,
            now,
        } = inputs;
        if row.envelope_hash != envelope_hash {
            return Err(RuntimeEffectControllerError::new(
                "postgres_effect_replay_hash_conflict",
                format!(
                    "runtime effect replay key `{replay_key}` in scope `{scope_id}` was reused with a different envelope hash"
                ),
            ));
        }
        let existing_due_at_ms = row.due_at_ms.map(|value| value as u64);
        match row.status.as_str() {
            POSTGRES_EFFECT_STATUS_COMPLETED => {
                let Some(json) = row.outcome_json else {
                    return Err(RuntimeEffectControllerError::new(
                        "postgres_effect_replay_corrupt_row",
                        "completed runtime effect row is missing outcome_json",
                    ));
                };
                let outcome = serde_json::from_str(&json).map_err(postgres_effect_decode_error)?;
                Ok(PostgresPreparedEffect::ReplayOutcome {
                    outcome: Box::new(outcome),
                    due_at_ms: existing_due_at_ms,
                })
            }
            POSTGRES_EFFECT_STATUS_FAILED => {
                let Some(json) = row.error_json else {
                    return Err(RuntimeEffectControllerError::new(
                        "postgres_effect_replay_corrupt_row",
                        "failed runtime effect row is missing error_json",
                    ));
                };
                let err = serde_json::from_str(&json).map_err(postgres_effect_decode_error)?;
                Ok(PostgresPreparedEffect::ReplayError(err))
            }
            POSTGRES_EFFECT_STATUS_IN_PROGRESS if row.lease_expires_at_ms as u64 > now => {
                Ok(PostgresPreparedEffect::Busy {
                    retry_at_ms: row.lease_expires_at_ms as u64,
                })
            }
            POSTGRES_EFFECT_STATUS_IN_PROGRESS => {
                let due_at_ms = existing_due_at_ms.or(due_at_ms);
                let due_at_param = due_at_ms.map(|value| value as i64);
                let updated_at_ms = current_epoch_ms();
                sqlx::query(
                    "UPDATE lash_runtime_effect_replay
                     SET lease_owner_id = $3,
                         lease_token = $4,
                         lease_expires_at_ms = $5,
                         due_at_ms = $6,
                         updated_at_ms = $7
                     WHERE scope_id = $1 AND replay_key = $2",
                )
                .bind(&scope_id)
                .bind(&replay_key)
                .bind(&owner_id)
                .bind(&lease_token)
                .bind(updated_at_ms.saturating_add(self.inner.lease_ttl_ms) as i64)
                .bind(due_at_param)
                .bind(updated_at_ms as i64)
                .execute(&mut **tx)
                .await
                .map_err(postgres_effect_store_error)?;
                Ok(PostgresPreparedEffect::Claimed(PostgresClaimedEffect {
                    scope_id,
                    replay_key,
                    envelope_hash,
                    lease_token,
                    due_at_ms,
                }))
            }
            other => Err(RuntimeEffectControllerError::new(
                "postgres_effect_replay_corrupt_row",
                format!("unknown runtime effect replay status `{other}`"),
            )),
        }
    }

    async fn finalize_effect(
        &self,
        claim: &PostgresClaimedEffect,
        outcome: &Result<RuntimeEffectOutcome, RuntimeEffectControllerError>,
    ) -> Result<(), RuntimeEffectControllerError> {
        let (status, outcome_json, error_json) = match outcome {
            Ok(outcome) => (
                POSTGRES_EFFECT_STATUS_COMPLETED,
                Some(serde_json::to_string(outcome).map_err(postgres_effect_encode_error)?),
                None,
            ),
            Err(err) => (
                POSTGRES_EFFECT_STATUS_FAILED,
                None,
                Some(serde_json::to_string(err).map_err(postgres_effect_encode_error)?),
            ),
        };
        let now = current_epoch_ms();
        let changed = sqlx::query(
            "UPDATE lash_runtime_effect_replay
             SET status = $6,
                 outcome_json = $7,
                 error_json = $8,
                 lease_owner_id = NULL,
                 lease_token = NULL,
                 lease_expires_at_ms = 0,
                 updated_at_ms = $9
             WHERE scope_id = $1
               AND replay_key = $2
               AND envelope_hash = $3
               AND lease_owner_id = $4
               AND lease_token = $5
               AND status = 'in_progress'
               AND lease_expires_at_ms > $10",
        )
        .bind(&claim.scope_id)
        .bind(&claim.replay_key)
        .bind(&claim.envelope_hash)
        .bind(&self.inner.owner_id)
        .bind(&claim.lease_token)
        .bind(status)
        .bind(outcome_json)
        .bind(error_json)
        .bind(now as i64)
        .bind(now as i64)
        .execute(&self.inner.pool)
        .await
        .map_err(postgres_effect_store_error)?
        .rows_affected();
        if changed != 1 {
            return Err(RuntimeEffectControllerError::new(
                "postgres_effect_replay_lease_lost",
                format!(
                    "runtime effect replay lease was lost before finalizing scope `{}` replay key `{}`",
                    claim.scope_id, claim.replay_key
                ),
            ));
        }
        Ok(())
    }

    async fn renew_effect_lease(
        &self,
        claim: &PostgresClaimedEffect,
    ) -> Result<(), RuntimeEffectControllerError> {
        let now = current_epoch_ms();
        let renewed_expires_at = now.saturating_add(self.inner.lease_ttl_ms);
        let changed = sqlx::query(
            "UPDATE lash_runtime_effect_replay
             SET lease_expires_at_ms = $6,
                 updated_at_ms = $7
             WHERE scope_id = $1
               AND replay_key = $2
               AND envelope_hash = $3
               AND lease_owner_id = $4
               AND lease_token = $5
               AND status = 'in_progress'
               AND lease_expires_at_ms > $8",
        )
        .bind(&claim.scope_id)
        .bind(&claim.replay_key)
        .bind(&claim.envelope_hash)
        .bind(&self.inner.owner_id)
        .bind(&claim.lease_token)
        .bind(renewed_expires_at as i64)
        .bind(now as i64)
        .bind(now as i64)
        .execute(&self.inner.pool)
        .await
        .map_err(postgres_effect_store_error)?
        .rows_affected();
        if changed != 1 {
            return Err(RuntimeEffectControllerError::new(
                "postgres_effect_replay_lease_lost",
                format!(
                    "runtime effect replay lease was lost while executing scope `{}` replay key `{}`",
                    claim.scope_id, claim.replay_key
                ),
            ));
        }
        Ok(())
    }

    async fn execute_claimed_effect_with_renewal(
        &self,
        claim: &PostgresClaimedEffect,
        envelope: RuntimeEffectEnvelope,
        local_executor: RuntimeEffectLocalExecutor<'_>,
    ) -> Result<RuntimeEffectOutcome, RuntimeEffectControllerError> {
        let renew_every = Duration::from_millis((self.inner.lease_ttl_ms / 3).max(1));
        let effect = self.execute_claimed_effect(claim, envelope, local_executor);
        tokio::pin!(effect);
        let renew_sleep = tokio::time::sleep(renew_every);
        tokio::pin!(renew_sleep);

        loop {
            tokio::select! {
                result = &mut effect => return result,
                _ = &mut renew_sleep => {
                    self.renew_effect_lease(claim).await?;
                    renew_sleep.as_mut().reset(tokio::time::Instant::now() + renew_every);
                }
            }
        }
    }

    async fn execute_claimed_effect(
        &self,
        claim: &PostgresClaimedEffect,
        envelope: RuntimeEffectEnvelope,
        local_executor: RuntimeEffectLocalExecutor<'_>,
    ) -> Result<RuntimeEffectOutcome, RuntimeEffectControllerError> {
        if matches!(envelope.command, RuntimeEffectCommand::Sleep { .. }) {
            postgres_effect_sleep_until_due(claim.due_at_ms).await;
            return Ok(RuntimeEffectOutcome::Sleep);
        }
        match envelope.command {
            RuntimeEffectCommand::Process { command } => {
                let result = self
                    .execute_process_command(*command, local_executor)
                    .await?;
                Ok(RuntimeEffectOutcome::Process { result })
            }
            _ => local_executor.execute(envelope).await,
        }
    }

    async fn execute_process_command(
        &self,
        command: ProcessCommand,
        local_executor: RuntimeEffectLocalExecutor<'_>,
    ) -> Result<ProcessEffectOutcome, RuntimeEffectControllerError> {
        let execution = local_executor.into_process()?;
        let registry = execution.registry;
        match command {
            ProcessCommand::Start {
                registration,
                grant,
                execution_context: _,
            } => {
                let registration_id = registration.id.clone();
                let record = registry.register_process(registration).await?;
                if let Some(grant) = grant {
                    registry
                        .grant_handle(&grant.session_scope, &registration_id, grant.descriptor)
                        .await?;
                }
                Ok(ProcessEffectOutcome::Start { record })
            }
            ProcessCommand::List {
                session_scope,
                mode,
            } => {
                let entries = match mode {
                    lash_core::ProcessListMode::Live => {
                        registry.list_live_handle_grants(&session_scope).await?
                    }
                    lash_core::ProcessListMode::All => {
                        registry.list_handle_grants(&session_scope).await?
                    }
                };
                Ok(ProcessEffectOutcome::List { entries })
            }
            ProcessCommand::Transfer {
                from_scope,
                to_scope,
                process_ids,
            } => {
                registry
                    .transfer_handle_grants(&from_scope, &to_scope, &process_ids)
                    .await?;
                Ok(ProcessEffectOutcome::Transfer)
            }
            ProcessCommand::DeleteSession { session_id } => {
                let report = registry.delete_session_process_state(&session_id).await?;
                Ok(ProcessEffectOutcome::DeleteSession { report })
            }
            ProcessCommand::Await { process_id } => {
                let output = registry.await_process(&process_id).await?;
                Ok(ProcessEffectOutcome::Await { output })
            }
            ProcessCommand::Cancel { process_id, reason } => {
                registry
                    .append_event(
                        &process_id,
                        lash_core::ProcessEventAppendRequest::cancel_requested(&process_id, reason),
                    )
                    .await?;
                let record = registry.get_process(&process_id).await.ok_or_else(|| {
                    PluginError::Session(format!("unknown process `{process_id}`"))
                })?;
                Ok(ProcessEffectOutcome::Cancel { record })
            }
            ProcessCommand::Signal {
                process_id,
                request,
                ..
            } => {
                let result = registry.append_event(&process_id, request).await?;
                Ok(ProcessEffectOutcome::Signal {
                    event: result.event,
                })
            }
        }
    }
}

impl AwaitEventResolver for PostgresRuntimeEffectController {
    fn durability_tier(&self) -> DurabilityTier {
        DurabilityTier::Durable
    }

    fn requires_durable_attachment_store(&self) -> bool {
        true
    }

    fn supports_durable_effects(&self) -> bool {
        true
    }
}

#[async_trait::async_trait]
impl RuntimeEffectController for PostgresRuntimeEffectController {
    async fn execute_effect(
        &self,
        envelope: RuntimeEffectEnvelope,
        local_executor: RuntimeEffectLocalExecutor<'_>,
    ) -> Result<RuntimeEffectOutcome, RuntimeEffectControllerError> {
        loop {
            match self.prepare_effect(&envelope).await? {
                PostgresPreparedEffect::ReplayOutcome { outcome, due_at_ms } => {
                    postgres_effect_sleep_until_due(due_at_ms).await;
                    return Ok(*outcome);
                }
                PostgresPreparedEffect::ReplayError(err) => return Err(err),
                PostgresPreparedEffect::Claimed(claim) => {
                    let result = self
                        .execute_claimed_effect_with_renewal(&claim, envelope, local_executor)
                        .await;
                    let finalize = self.finalize_effect(&claim, &result).await;
                    return match (result, finalize) {
                        (Ok(outcome), Ok(())) => Ok(outcome),
                        (Err(err), Ok(())) => Err(err),
                        (_, Err(err)) => Err(err),
                    };
                }
                PostgresPreparedEffect::Busy { retry_at_ms } => {
                    postgres_effect_sleep_until_retry(retry_at_ms).await;
                }
            }
        }
    }
}

struct PostgresPrepareInputs {
    row: PostgresEffectRow,
    scope_id: String,
    replay_key: String,
    envelope_hash: String,
    owner_id: String,
    lease_token: String,
    due_at_ms: Option<u64>,
    now: u64,
}

impl PostgresEffectReplayInner {
    fn new(pool: PgPool, options: PostgresEffectReplayOptions) -> Self {
        let sequence = POSTGRES_EFFECT_OWNER_COUNTER.fetch_add(1, Ordering::SeqCst);
        Self {
            pool,
            owner_id: format!(
                "pid{}-{sequence}-{}",
                std::process::id(),
                current_epoch_ms()
            ),
            lease_counter: AtomicU64::new(1),
            replay_mode: AtomicBool::new(false),
            lease_ttl_ms: postgres_effect_duration_ms(options.lease_ttl),
        }
    }

    fn next_lease_token(&self) -> String {
        let sequence = self.lease_counter.fetch_add(1, Ordering::SeqCst);
        format!("{}:{sequence}", self.owner_id)
    }
}

async fn postgres_select_effect_row_for_update(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    scope_id: &str,
    replay_key: &str,
) -> Result<Option<PostgresEffectRow>, RuntimeEffectControllerError> {
    let row = sqlx::query(
        "SELECT envelope_hash, status, outcome_json, error_json, lease_expires_at_ms, due_at_ms
         FROM lash_runtime_effect_replay
         WHERE scope_id = $1 AND replay_key = $2
         FOR UPDATE",
    )
    .bind(scope_id)
    .bind(replay_key)
    .fetch_optional(&mut **tx)
    .await
    .map_err(postgres_effect_store_error)?;
    Ok(row.map(postgres_effect_row))
}

fn postgres_effect_row(row: PgRow) -> PostgresEffectRow {
    PostgresEffectRow {
        envelope_hash: row.get("envelope_hash"),
        status: row.get("status"),
        outcome_json: row.get("outcome_json"),
        error_json: row.get("error_json"),
        lease_expires_at_ms: row.get("lease_expires_at_ms"),
        due_at_ms: row.get("due_at_ms"),
    }
}

fn postgres_effect_duration_ms(duration: Duration) -> u64 {
    let millis = duration.as_millis();
    if millis == 0 {
        1
    } else {
        millis.min(u128::from(u64::MAX)) as u64
    }
}

fn postgres_effect_sleep_due_at_ms(envelope: &RuntimeEffectEnvelope, now: u64) -> Option<u64> {
    match envelope.command {
        RuntimeEffectCommand::Sleep { duration_ms } => Some(now.saturating_add(duration_ms)),
        _ => None,
    }
}

async fn postgres_effect_sleep_until_due(due_at_ms: Option<u64>) {
    let Some(due_at_ms) = due_at_ms else {
        return;
    };
    let now = current_epoch_ms();
    if due_at_ms > now {
        tokio::time::sleep(Duration::from_millis(due_at_ms - now)).await;
    }
}

async fn postgres_effect_sleep_until_retry(retry_at_ms: u64) {
    let now = current_epoch_ms();
    let delay = if retry_at_ms > now {
        Duration::from_millis(retry_at_ms - now).min(POSTGRES_EFFECT_BUSY_POLL)
    } else {
        POSTGRES_EFFECT_BUSY_POLL
    };
    tokio::time::sleep(delay).await;
}

fn postgres_effect_store_error(err: sqlx::Error) -> RuntimeEffectControllerError {
    RuntimeEffectControllerError::new("postgres_effect_replay_store", err.to_string())
}

fn postgres_effect_encode_error(err: serde_json::Error) -> RuntimeEffectControllerError {
    RuntimeEffectControllerError::new(
        "postgres_effect_replay_encode",
        format!("failed to encode runtime effect replay row: {err}"),
    )
}

fn postgres_effect_decode_error(err: serde_json::Error) -> RuntimeEffectControllerError {
    RuntimeEffectControllerError::new(
        "postgres_effect_replay_decode",
        format!("failed to decode runtime effect replay row: {err}"),
    )
}
