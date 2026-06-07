//! SQLite-backed runtime effect replay host (tokio-rusqlite port of the the prior store
//! store's `effect_replay`).
//!
//! The public surface is identical to the prior implementation — same type names
//! (`SqliteEffectHost`, `SqliteRuntimeEffectController`, `SqliteEffectReplayOptions`),
//! same async signatures — so consumers swap backends with a path rename only.
//! The only mechanical change is the database layer: the prior store ran every op directly
//! on `&Connection` with `.await`; here every database body is a *synchronous*
//! rusqlite closure handed to the shared [`SqliteConnection`] wrapper. The
//! claim/finalize paths run inside `conn.write` (`BEGIN IMMEDIATE`) so the
//! cross-process write lock is taken up front — the lease fence WAL gives us.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use lash_core::{
    DurabilityTier, EffectHost, EffectScope, PluginError, ProcessCommand, ProcessEffectOutcome,
    RuntimeEffectCommand, RuntimeEffectController, RuntimeEffectControllerError,
    RuntimeEffectEnvelope, RuntimeEffectLocalExecutor, RuntimeEffectOutcome, RuntimeError,
    ScopedEffectController,
};

use super::*;

const STATUS_IN_PROGRESS: &str = "in_progress";
const STATUS_COMPLETED: &str = "completed";
const STATUS_FAILED: &str = "failed";
const DEFAULT_LEASE_TTL: Duration = Duration::from_secs(30);
const BUSY_POLL: Duration = Duration::from_millis(25);

static EFFECT_OWNER_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Options for SQLite-backed runtime effect replay.
#[derive(Clone, Debug)]
pub struct SqliteEffectReplayOptions {
    pub lease_ttl: Duration,
}

impl Default for SqliteEffectReplayOptions {
    fn default() -> Self {
        Self {
            lease_ttl: DEFAULT_LEASE_TTL,
        }
    }
}

struct SqliteEffectReplayInner {
    conn: SqliteConnection,
    owner_id: String,
    lease_counter: AtomicU64,
    replay_mode: AtomicBool,
    lease_ttl_ms: u64,
}

/// Deployment-level SQLite effect host.
///
/// This host persists runtime effect history in a local SQLite database and
/// returns scoped controllers that replay completed outcomes by
/// `(scope_id, replay_key)`.
#[derive(Clone)]
pub struct SqliteEffectHost {
    inner: Arc<SqliteEffectReplayInner>,
}

/// Scoped SQLite-backed runtime effect controller.
#[derive(Clone)]
pub struct SqliteRuntimeEffectController {
    inner: Arc<SqliteEffectReplayInner>,
    scope: EffectScope,
}

struct ClaimedEffect {
    scope_id: String,
    replay_key: String,
    envelope_hash: String,
    lease_token: String,
    due_at_ms: Option<u64>,
}

enum PreparedEffect {
    ReplayOutcome {
        outcome: Box<RuntimeEffectOutcome>,
        due_at_ms: Option<u64>,
    },
    ReplayError(RuntimeEffectControllerError),
    Claimed(ClaimedEffect),
    Busy {
        retry_at_ms: u64,
    },
}

impl SqliteEffectHost {
    pub async fn open(path: &Path) -> tokio_rusqlite::Result<Self> {
        Self::open_with_options(path, SqliteEffectReplayOptions::default()).await
    }

    pub async fn open_with_options(
        path: &Path,
        options: SqliteEffectReplayOptions,
    ) -> tokio_rusqlite::Result<Self> {
        Ok(Self {
            inner: open_effect_replay_inner(path, StoreBacking::File, options).await?,
        })
    }

    pub async fn memory() -> tokio_rusqlite::Result<Self> {
        Self::memory_with_options(SqliteEffectReplayOptions::default()).await
    }

    pub async fn memory_with_options(
        options: SqliteEffectReplayOptions,
    ) -> tokio_rusqlite::Result<Self> {
        Ok(Self {
            inner: open_effect_replay_memory_inner(options).await?,
        })
    }

    /// Force strict replay mode: missing effect history fails instead of
    /// executing locally. Normal operation still replays any completed row.
    pub fn start_replay(&self) {
        self.inner.replay_mode.store(true, Ordering::SeqCst);
    }
}

impl EffectHost for SqliteEffectHost {
    fn durability_tier(&self) -> DurabilityTier {
        DurabilityTier::Durable
    }

    fn requires_durable_attachment_store(&self) -> bool {
        true
    }

    fn scoped<'run>(
        &'run self,
        scope: EffectScope,
    ) -> Result<ScopedEffectController<'run>, RuntimeError> {
        let controller = SqliteRuntimeEffectController {
            inner: Arc::clone(&self.inner),
            scope: scope.clone(),
        };
        ScopedEffectController::shared(Arc::new(controller), scope)
    }

    fn scoped_static(
        &self,
        scope: EffectScope,
    ) -> Result<Option<ScopedEffectController<'static>>, RuntimeError> {
        let controller = SqliteRuntimeEffectController {
            inner: Arc::clone(&self.inner),
            scope: scope.clone(),
        };
        Ok(Some(ScopedEffectController::shared(
            Arc::new(controller),
            scope,
        )?))
    }
}

impl SqliteRuntimeEffectController {
    pub async fn open(path: &Path, scope: EffectScope) -> tokio_rusqlite::Result<Self> {
        Self::open_with_options(path, scope, SqliteEffectReplayOptions::default()).await
    }

    pub async fn open_with_options(
        path: &Path,
        scope: EffectScope,
        options: SqliteEffectReplayOptions,
    ) -> tokio_rusqlite::Result<Self> {
        Ok(Self {
            inner: open_effect_replay_inner(path, StoreBacking::File, options).await?,
            scope,
        })
    }

    pub async fn memory(scope: EffectScope) -> tokio_rusqlite::Result<Self> {
        Self::memory_with_options(scope, SqliteEffectReplayOptions::default()).await
    }

    pub async fn memory_with_options(
        scope: EffectScope,
        options: SqliteEffectReplayOptions,
    ) -> tokio_rusqlite::Result<Self> {
        Ok(Self {
            inner: open_effect_replay_memory_inner(options).await?,
            scope,
        })
    }

    /// Force strict replay mode: missing effect history fails instead of
    /// executing locally. Normal operation still replays any completed row.
    pub fn start_replay(&self) {
        self.inner.replay_mode.store(true, Ordering::SeqCst);
    }

    async fn prepare_effect(
        &self,
        envelope: &RuntimeEffectEnvelope,
    ) -> Result<PreparedEffect, RuntimeEffectControllerError> {
        let replay_key = envelope
            .invocation
            .replay_key()
            .ok_or_else(|| {
                RuntimeEffectControllerError::new(
                    "sqlite_effect_replay_key_missing",
                    "runtime effect envelope requires replay.key",
                )
            })?
            .to_string();
        let envelope_hash = envelope.stable_hash()?;
        let scope_id = self.scope.id().to_string();
        let now = current_epoch_ms();
        let lease_token = self.inner.next_lease_token();
        let due_at_ms = sleep_due_at_ms(envelope, now);
        let lease_expires_at_ms = now.saturating_add(self.inner.lease_ttl_ms);
        let replay_mode = self.inner.replay_mode.load(Ordering::SeqCst);
        let owner_id = self.inner.owner_id.clone();
        let lease_ttl_ms = self.inner.lease_ttl_ms;

        // The `BEGIN IMMEDIATE` transaction is run on the connection thread via
        // `conn.write`. The closure returns a `rusqlite::Result` carrying our
        // own outcome `Result<PreparedEffect, RuntimeEffectControllerError>`:
        // the SQL committing is independent of whether the recorded effect was
        // a success, a failure, or a fresh claim — exactly the prior behaviour
        // (commit on `Ok(_)` for any `PreparedEffect` variant). Only a real
        // SQLite error rolls back.
        let outcome: Result<PreparedEffect, RuntimeEffectControllerError> = self
            .inner
            .conn
            .write(move |tx| {
                let row = tx
                    .query_row(
                        "SELECT envelope_hash, status, outcome_json, error_json,
                                lease_owner_id, lease_token, lease_expires_at_ms, due_at_ms
                         FROM runtime_effect_replay
                         WHERE scope_id = ?1 AND replay_key = ?2",
                        params![scope_id.as_str(), replay_key.as_str()],
                        |row| {
                            Ok((
                                row.get::<_, String>(0)?,
                                row.get::<_, String>(1)?,
                                row.get::<_, Option<String>>(2)?,
                                row.get::<_, Option<String>>(3)?,
                                row.get::<_, i64>(6)?,
                                row.get::<_, Option<i64>>(7)?,
                            ))
                        },
                    )
                    .optional()?;

                let Some((
                    existing_hash,
                    status,
                    outcome_json,
                    error_json,
                    lease_expires_row,
                    existing_due_row,
                )) = row
                else {
                    if replay_mode {
                        return Ok(Err(RuntimeEffectControllerError::new(
                            "sqlite_effect_replay_missing",
                            format!(
                                "no recorded runtime effect for scope `{scope_id}` and replay key `{replay_key}`"
                            ),
                        )));
                    }
                    let due_at_param = due_at_ms.map(|value| value as i64);
                    tx.execute(
                        "INSERT INTO runtime_effect_replay (
                            scope_id, replay_key, envelope_hash, status, outcome_json,
                            error_json, lease_owner_id, lease_token, lease_expires_at_ms,
                            due_at_ms, created_at_ms, updated_at_ms
                         )
                         VALUES (?1, ?2, ?3, ?4, NULL, NULL, ?5, ?6, ?7, ?8, ?9, ?10)",
                        params![
                            scope_id.as_str(),
                            replay_key.as_str(),
                            envelope_hash.as_str(),
                            STATUS_IN_PROGRESS,
                            owner_id.as_str(),
                            lease_token.as_str(),
                            lease_expires_at_ms as i64,
                            due_at_param,
                            now as i64,
                            now as i64,
                        ],
                    )?;
                    return Ok(Ok(PreparedEffect::Claimed(ClaimedEffect {
                        scope_id,
                        replay_key,
                        envelope_hash,
                        lease_token,
                        due_at_ms,
                    })));
                };

                if existing_hash != envelope_hash {
                    return Ok(Err(RuntimeEffectControllerError::new(
                        "sqlite_effect_replay_hash_conflict",
                        format!(
                            "runtime effect replay key `{replay_key}` in scope `{scope_id}` was reused with a different envelope hash"
                        ),
                    )));
                }

                let lease_expires_at_ms = lease_expires_row as u64;
                let existing_due_at_ms = existing_due_row.map(|value| value as u64);

                match status.as_str() {
                    STATUS_COMPLETED => {
                        let Some(json) = outcome_json else {
                            return Ok(Err(RuntimeEffectControllerError::new(
                                "sqlite_effect_replay_corrupt_row",
                                "completed runtime effect row is missing outcome_json",
                            )));
                        };
                        let outcome = match serde_json::from_str(&json) {
                            Ok(outcome) => outcome,
                            Err(err) => return Ok(Err(effect_decode_error(err))),
                        };
                        Ok(Ok(PreparedEffect::ReplayOutcome {
                            outcome: Box::new(outcome),
                            due_at_ms: existing_due_at_ms,
                        }))
                    }
                    STATUS_FAILED => {
                        let Some(json) = error_json else {
                            return Ok(Err(RuntimeEffectControllerError::new(
                                "sqlite_effect_replay_corrupt_row",
                                "failed runtime effect row is missing error_json",
                            )));
                        };
                        let err = match serde_json::from_str(&json) {
                            Ok(err) => err,
                            Err(err) => return Ok(Err(effect_decode_error(err))),
                        };
                        Ok(Ok(PreparedEffect::ReplayError(err)))
                    }
                    STATUS_IN_PROGRESS if lease_expires_at_ms > now => Ok(Ok(PreparedEffect::Busy {
                        retry_at_ms: lease_expires_at_ms,
                    })),
                    STATUS_IN_PROGRESS => {
                        let due_at_ms = existing_due_at_ms.or(due_at_ms);
                        let due_at_param = due_at_ms.map(|value| value as i64);
                        tx.execute(
                            "UPDATE runtime_effect_replay
                             SET lease_owner_id = ?3,
                                 lease_token = ?4,
                                 lease_expires_at_ms = ?5,
                                 due_at_ms = ?6,
                                 updated_at_ms = ?7
                             WHERE scope_id = ?1 AND replay_key = ?2",
                            params![
                                scope_id.as_str(),
                                replay_key.as_str(),
                                owner_id.as_str(),
                                lease_token.as_str(),
                                current_epoch_ms().saturating_add(lease_ttl_ms) as i64,
                                due_at_param,
                                current_epoch_ms() as i64,
                            ],
                        )?;
                        Ok(Ok(PreparedEffect::Claimed(ClaimedEffect {
                            scope_id,
                            replay_key,
                            envelope_hash,
                            lease_token,
                            due_at_ms,
                        })))
                    }
                    other => Ok(Err(RuntimeEffectControllerError::new(
                        "sqlite_effect_replay_corrupt_row",
                        format!("unknown runtime effect replay status `{other}`"),
                    ))),
                }
            })
            .await
            .map_err(effect_sqlite_error)?;
        outcome
    }

    async fn finalize_effect(
        &self,
        claim: &ClaimedEffect,
        outcome: &Result<RuntimeEffectOutcome, RuntimeEffectControllerError>,
    ) -> Result<(), RuntimeEffectControllerError> {
        let (status, outcome_json, error_json) = match outcome {
            Ok(outcome) => (
                STATUS_COMPLETED,
                Some(serde_json::to_string(outcome).map_err(effect_encode_error)?),
                None,
            ),
            Err(err) => (
                STATUS_FAILED,
                None,
                Some(serde_json::to_string(err).map_err(effect_encode_error)?),
            ),
        };
        let now = current_epoch_ms();
        let scope_id = claim.scope_id.clone();
        let replay_key = claim.replay_key.clone();
        let envelope_hash = claim.envelope_hash.clone();
        let lease_token = claim.lease_token.clone();
        let status = status.to_string();

        let result: Result<(), RuntimeEffectControllerError> = self
            .inner
            .conn
            .write(move |tx| {
                let changed = tx.execute(
                    "UPDATE runtime_effect_replay
                     SET status = ?5,
                         outcome_json = ?6,
                         error_json = ?7,
                         lease_owner_id = NULL,
                         lease_token = NULL,
                         lease_expires_at_ms = 0,
                         updated_at_ms = ?8
                     WHERE scope_id = ?1
                       AND replay_key = ?2
                       AND envelope_hash = ?3
                       AND lease_token = ?4
                       AND status = 'in_progress'",
                    params![
                        scope_id.as_str(),
                        replay_key.as_str(),
                        envelope_hash.as_str(),
                        lease_token.as_str(),
                        status.as_str(),
                        outcome_json,
                        error_json,
                        now as i64,
                    ],
                )?;
                if changed != 1 {
                    return Ok(Err(RuntimeEffectControllerError::new(
                        "sqlite_effect_replay_lease_lost",
                        format!(
                            "runtime effect replay lease was lost before finalizing scope `{scope_id}` replay key `{replay_key}`"
                        ),
                    )));
                }
                Ok(Ok(()))
            })
            .await
            .map_err(effect_sqlite_error)?;
        result
    }

    async fn execute_claimed_effect(
        &self,
        claim: &ClaimedEffect,
        envelope: RuntimeEffectEnvelope,
        local_executor: RuntimeEffectLocalExecutor<'_>,
    ) -> Result<RuntimeEffectOutcome, RuntimeEffectControllerError> {
        if matches!(envelope.command, RuntimeEffectCommand::Sleep { .. }) {
            sleep_until_due(claim.due_at_ms).await;
            return Ok(RuntimeEffectOutcome::Sleep);
        }
        match envelope.command {
            RuntimeEffectCommand::Process { command } => {
                let result = self
                    .execute_process_command(command, local_executor)
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
                        .grant_handle(&grant.owner_scope, &registration_id, grant.descriptor)
                        .await?;
                }
                Ok(ProcessEffectOutcome::Start { record })
            }
            ProcessCommand::List { owner_scope, mode } => {
                let entries = match mode {
                    lash_core::ProcessListMode::Live => {
                        registry.list_live_handle_grants(&owner_scope).await?
                    }
                    lash_core::ProcessListMode::All => {
                        registry.list_handle_grants(&owner_scope).await?
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
                for process_id in &report.cancel_process_ids {
                    registry
                        .append_event(
                            process_id,
                            lash_core::ProcessEventAppendRequest::cancel_requested(
                                process_id,
                                Some("session deleted".to_string()),
                            ),
                        )
                        .await?;
                }
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

#[async_trait::async_trait]
impl RuntimeEffectController for SqliteRuntimeEffectController {
    fn durability_tier(&self) -> DurabilityTier {
        DurabilityTier::Durable
    }

    fn requires_durable_attachment_store(&self) -> bool {
        true
    }

    async fn execute_effect(
        &self,
        envelope: RuntimeEffectEnvelope,
        local_executor: RuntimeEffectLocalExecutor<'_>,
    ) -> Result<RuntimeEffectOutcome, RuntimeEffectControllerError> {
        loop {
            match self.prepare_effect(&envelope).await? {
                PreparedEffect::ReplayOutcome { outcome, due_at_ms } => {
                    sleep_until_due(due_at_ms).await;
                    return Ok(*outcome);
                }
                PreparedEffect::ReplayError(err) => return Err(err),
                PreparedEffect::Claimed(claim) => {
                    let result = self
                        .execute_claimed_effect(&claim, envelope, local_executor)
                        .await;
                    let finalize = self.finalize_effect(&claim, &result).await;
                    return match (result, finalize) {
                        (Ok(outcome), Ok(())) => Ok(outcome),
                        (Err(err), Ok(())) => Err(err),
                        (_, Err(err)) => Err(err),
                    };
                }
                PreparedEffect::Busy { retry_at_ms } => {
                    sleep_until_retry(retry_at_ms).await;
                }
            }
        }
    }
}

async fn open_effect_replay_inner(
    path: &Path,
    backing: StoreBacking,
    options: SqliteEffectReplayOptions,
) -> tokio_rusqlite::Result<Arc<SqliteEffectReplayInner>> {
    let conn = SqliteConnection::open(path).await?;
    ensure_effect_schema(&conn).await?;
    apply_pragmas(&conn, backing).await?;
    Ok(Arc::new(SqliteEffectReplayInner::new(conn, options)))
}

async fn open_effect_replay_memory_inner(
    options: SqliteEffectReplayOptions,
) -> tokio_rusqlite::Result<Arc<SqliteEffectReplayInner>> {
    let conn = SqliteConnection::open_in_memory().await?;
    ensure_effect_schema(&conn).await?;
    apply_pragmas(&conn, StoreBacking::Memory).await?;
    Ok(Arc::new(SqliteEffectReplayInner::new(conn, options)))
}

impl SqliteEffectReplayInner {
    fn new(conn: SqliteConnection, options: SqliteEffectReplayOptions) -> Self {
        let sequence = EFFECT_OWNER_COUNTER.fetch_add(1, Ordering::SeqCst);
        Self {
            conn,
            owner_id: format!(
                "pid{}-{sequence}-{}",
                std::process::id(),
                current_epoch_ms()
            ),
            lease_counter: AtomicU64::new(1),
            replay_mode: AtomicBool::new(false),
            lease_ttl_ms: duration_ms(options.lease_ttl),
        }
    }

    fn next_lease_token(&self) -> String {
        let sequence = self.lease_counter.fetch_add(1, Ordering::SeqCst);
        format!("{}:{sequence}", self.owner_id)
    }
}

fn duration_ms(duration: Duration) -> u64 {
    let millis = duration.as_millis();
    if millis == 0 {
        1
    } else {
        millis.min(u128::from(u64::MAX)) as u64
    }
}

fn sleep_due_at_ms(envelope: &RuntimeEffectEnvelope, now: u64) -> Option<u64> {
    match envelope.command {
        RuntimeEffectCommand::Sleep { duration_ms } => Some(now.saturating_add(duration_ms)),
        _ => None,
    }
}

async fn sleep_until_due(due_at_ms: Option<u64>) {
    let Some(due_at_ms) = due_at_ms else {
        return;
    };
    let now = current_epoch_ms();
    if due_at_ms > now {
        tokio::time::sleep(Duration::from_millis(due_at_ms - now)).await;
    }
}

async fn sleep_until_retry(retry_at_ms: u64) {
    let now = current_epoch_ms();
    let delay = if retry_at_ms > now {
        Duration::from_millis(retry_at_ms - now).min(BUSY_POLL)
    } else {
        BUSY_POLL
    };
    tokio::time::sleep(delay).await;
}

fn effect_sqlite_error(err: rusqlite::Error) -> RuntimeEffectControllerError {
    RuntimeEffectControllerError::new("sqlite_effect_replay_store", err.to_string())
}

fn effect_encode_error(err: serde_json::Error) -> RuntimeEffectControllerError {
    RuntimeEffectControllerError::new(
        "sqlite_effect_replay_encode",
        format!("failed to encode runtime effect replay row: {err}"),
    )
}

fn effect_decode_error(err: serde_json::Error) -> RuntimeEffectControllerError {
    RuntimeEffectControllerError::new(
        "sqlite_effect_replay_decode",
        format!("failed to decode runtime effect replay row: {err}"),
    )
}
