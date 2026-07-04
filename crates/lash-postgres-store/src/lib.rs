//! PostgreSQL durable storage for Lash.
//!
//! One [`PostgresStorage`] owns a shared [`sqlx::PgPool`] and creates durable
//! implementations for the runtime session store, process registry, trigger
//! store, Lashlang artifact store, process execution environment store, and
//! attachment manifest.

use std::collections::HashSet;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use lash_core::runtime::{
    ProcessHandleGrantEntry, QueuedWorkBatch, QueuedWorkBatchDraft, QueuedWorkClaim,
    QueuedWorkClaimBoundary, QueuedWorkCompletion, QueuedWorkItem,
};
use lash_core::store::queued_work::{
    ClaimCandidate, QueuedWorkClaimLease, claim_scan_limit, derive_batch_id, renewed_claim,
    select_leading_session_command, select_turn_work_claim_prefix,
};
use lash_core::store::{
    GraphCommitDelta, HydratedSessionCheckpoint, PersistedSessionRead, RuntimeCommit,
    RuntimeCommitResult, SessionCheckpoint, SessionHeadMeta,
};
use lash_core::{
    AbandonRequest, AttachmentId, AttachmentIntent, AttachmentManifest, AttachmentManifestEntry,
    AwaitEventResolver, BlobRef, DeliveryPolicy, DurabilityTier, EffectHost, ExecutionScope,
    GcReport, LeaseOwnerIdentity, LeaseOwnerLiveness, MergeKey, ProcessAwaitOutput, ProcessEvent,
    ProcessEventAppendRequest, ProcessEventAppendResult, ProcessExternalRef,
    ProcessHandleDescriptor, ProcessHandleGrant, ProcessLease, ProcessLeaseCompletion,
    ProcessPruneReport, ProcessRecord, ProcessRegistration, ProcessRegistry, ProcessStarted,
    QueuedWorkStore, RuntimeEffectCommand, RuntimeEffectController, RuntimeEffectControllerError,
    RuntimeEffectEnvelope, RuntimeEffectLocalExecutor, RuntimeEffectOutcome, RuntimeError,
    RuntimePersistence, ScopedEffectController, SessionCommitStore, SessionExecutionLease,
    SessionExecutionLeaseClaimOutcome, SessionExecutionLeaseCompletion, SessionExecutionLeaseFence,
    SessionExecutionLeaseStore, SessionMeta, SessionNodeRecord, SessionReadScope, SessionScope,
    SessionStoreCreateRequest, SessionStoreFactory, SlotPolicy, StoreError, StoreMaintenance,
    TokenLedgerEntry, TurnInputStore, VacuumReport,
};
use lash_core::{
    PluginError, TriggerDeliveryReservation, TriggerOccurrenceRecord, TriggerOccurrenceRequest,
    TriggerStore, TriggerSubscriptionDraft, TriggerSubscriptionFilter, TriggerSubscriptionRecord,
};
use sha2::{Digest, Sha256};
use sqlx::postgres::{PgPool, PgPoolOptions, PgRow};
use sqlx::{Executor, Row};

const SCHEMA_COMPONENT: &str = "lash-postgres-store";
// Bumped to 7: `ProcessRecord` gained a required `disposition` field (plus
// optional `first_started`/`abandon_request`) inside `record_json` (ADR 0019).
// The schema is a reject-and-recreate boundary — a pre-7 database's rows predate
// the column and cannot deserialize, so the version mismatch is rejected at open
// rather than backfilled; new rows always carry the disposition.
const SCHEMA_VERSION: i32 = 7;
const PROCESS_LEASE_SCHEMA_VERSION: u32 = lash_core::PROCESS_LEASE_SCHEMA_VERSION;

#[derive(Clone)]
pub struct PostgresStorage {
    pool: PgPool,
}

#[derive(Clone)]
pub struct PostgresSessionStoreFactory {
    pool: PgPool,
}

#[derive(Clone)]
pub struct PostgresSessionStore {
    pool: PgPool,
    /// Explicit session binding for handles created via the factory.
    session_id: Option<String>,
    /// In-memory bind-on-first-commit for an *unbound* handle. A session-store
    /// handle commits to exactly one session; an unbound handle latches the first
    /// session it commits and rejects others (Postgres is multi-session per
    /// database, so this can't be inferred from a singleton head row the way the
    /// single-file SQLite store does). Shared across clones via `Arc`.
    bound_session: Arc<OnceLock<String>>,
}

#[derive(Clone)]
pub struct PostgresProcessRegistry {
    pool: PgPool,
}

#[derive(Clone)]
pub struct PostgresTriggerStore {
    pool: PgPool,
}

#[derive(Clone)]
pub struct PostgresLashlangArtifactStore {
    pool: PgPool,
}

/// Connection-pool and per-connection timeout knobs for [`PostgresStorage`].
///
/// Mutating session work first claims the durable session execution lease.
/// Session commits then verify that lease fence and use **optimistic CAS** on
/// the head (`UPDATE … WHERE head_revision = expected`) as a stale-writer
/// backstop, not as the normal cross-worker concurrency primitive.
/// `lock_timeout` is defense in depth: it caps how long the fenced CAS write
/// may wait on the head row's lock before erroring (surfaced as a retryable
/// conflict), so a pathological burst can never starve the pool.
#[derive(Clone, Debug)]
pub struct PostgresStoreConfig {
    /// Maximum pooled connections. Default 16.
    pub max_connections: u32,
    /// Minimum idle connections kept warm. Default 0.
    pub min_connections: u32,
    /// How long `acquire` waits for a free connection before erroring. Default 30s.
    pub acquire_timeout: Duration,
    /// Close a connection after this idle period. Default 10m.
    pub idle_timeout: Option<Duration>,
    /// Recycle a connection after this lifetime. Default 30m.
    pub max_lifetime: Option<Duration>,
    /// Postgres `lock_timeout` applied to every connection. Default 10s.
    pub lock_timeout: Option<Duration>,
    /// Postgres `statement_timeout` applied to every connection. Default 30s — a
    /// backstop so a wedged query can never hold a connection indefinitely.
    pub statement_timeout: Option<Duration>,
}

impl Default for PostgresStoreConfig {
    fn default() -> Self {
        Self {
            max_connections: 16,
            min_connections: 0,
            acquire_timeout: Duration::from_secs(30),
            idle_timeout: Some(Duration::from_secs(600)),
            max_lifetime: Some(Duration::from_secs(1800)),
            lock_timeout: Some(Duration::from_secs(10)),
            statement_timeout: Some(Duration::from_secs(30)),
        }
    }
}

impl PostgresStorage {
    /// Connect with [`PostgresStoreConfig::default`] pool/timeout settings.
    pub async fn connect(database_url: &str) -> Result<Self, StoreError> {
        Self::connect_with(database_url, PostgresStoreConfig::default()).await
    }

    /// Connect with explicit pool sizing and per-connection timeouts.
    pub async fn connect_with(
        database_url: &str,
        config: PostgresStoreConfig,
    ) -> Result<Self, StoreError> {
        let lock_ms = config.lock_timeout.map(|d| d.as_millis().max(1) as u64);
        let statement_ms = config
            .statement_timeout
            .map(|d| d.as_millis().max(1) as u64);
        let mut options = PgPoolOptions::new()
            .max_connections(config.max_connections)
            .min_connections(config.min_connections)
            .acquire_timeout(config.acquire_timeout);
        if let Some(timeout) = config.idle_timeout {
            options = options.idle_timeout(timeout);
        }
        if let Some(timeout) = config.max_lifetime {
            options = options.max_lifetime(timeout);
        }
        let pool = options
            .after_connect(move |conn, _meta| {
                Box::pin(async move {
                    if let Some(ms) = lock_ms {
                        conn.execute(format!("SET lock_timeout = {ms}").as_str())
                            .await?;
                    }
                    if let Some(ms) = statement_ms {
                        conn.execute(format!("SET statement_timeout = {ms}").as_str())
                            .await?;
                    }
                    Ok(())
                })
            })
            .connect(database_url)
            .await
            .map_err(store_sqlx_error)?;
        ensure_schema(&pool).await?;
        Ok(Self { pool })
    }

    pub fn from_pool(pool: PgPool) -> Self {
        Self { pool }
    }

    pub fn pool(&self) -> &PgPool {
        &self.pool
    }

    pub fn session_store_factory(&self) -> PostgresSessionStoreFactory {
        PostgresSessionStoreFactory {
            pool: self.pool.clone(),
        }
    }

    pub fn session_store(&self, session_id: impl Into<String>) -> PostgresSessionStore {
        PostgresSessionStore {
            pool: self.pool.clone(),
            session_id: Some(session_id.into()),
            bound_session: Arc::new(OnceLock::new()),
        }
    }

    pub fn unbound_session_store(&self) -> PostgresSessionStore {
        PostgresSessionStore {
            pool: self.pool.clone(),
            session_id: None,
            bound_session: Arc::new(OnceLock::new()),
        }
    }

    pub fn process_registry(&self) -> PostgresProcessRegistry {
        PostgresProcessRegistry {
            pool: self.pool.clone(),
        }
    }

    pub fn trigger_store(&self) -> PostgresTriggerStore {
        PostgresTriggerStore {
            pool: self.pool.clone(),
        }
    }

    pub fn lashlang_artifact_store(&self) -> PostgresLashlangArtifactStore {
        PostgresLashlangArtifactStore {
            pool: self.pool.clone(),
        }
    }

    pub fn process_env_store(&self) -> PostgresLashlangArtifactStore {
        PostgresLashlangArtifactStore {
            pool: self.pool.clone(),
        }
    }

    pub fn effect_host(&self) -> PostgresEffectHost {
        PostgresEffectHost::new(self)
    }

    pub fn runtime_effect_controller(
        &self,
        scope: ExecutionScope,
    ) -> PostgresRuntimeEffectController {
        PostgresRuntimeEffectController::new(self, scope)
    }
}

impl PostgresSessionStoreFactory {
    pub fn new(storage: &PostgresStorage) -> Self {
        storage.session_store_factory()
    }
}

impl PostgresSessionStore {
    pub fn unbound(storage: &PostgresStorage) -> Self {
        storage.unbound_session_store()
    }

    async fn selected_session_id(&self) -> Result<Option<String>, StoreError> {
        if let Some(session_id) = &self.session_id {
            return Ok(Some(session_id.clone()));
        }
        sqlx::query_scalar("SELECT session_id FROM lash_sessions ORDER BY session_id ASC LIMIT 1")
            .fetch_optional(&self.pool)
            .await
            .map_err(store_sqlx_error)
    }
}

include!("postgres/schema.rs");
include!("postgres/support.rs");
include!("postgres/attachments.rs");
include!("postgres/effect_replay.rs");
include!("postgres/session_factory.rs");
include!("postgres/runtime_persistence.rs");
include!("postgres/process_helpers.rs");
include!("postgres/process_registry.rs");
include!("postgres/trigger_store.rs");
include!("postgres/artifact_store.rs");
