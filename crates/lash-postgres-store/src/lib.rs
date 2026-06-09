//! PostgreSQL durable storage for Lash.
//!
//! One [`PostgresStorage`] owns a shared [`sqlx::PgPool`] and creates durable
//! implementations for the runtime session store, process registry, host-event
//! store, Lashlang artifact store, and attachment manifest.

use std::collections::HashSet;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use lash_core::runtime::{
    ProcessHandleGrantEntry, QueuedWorkBatch, QueuedWorkBatchDraft, QueuedWorkClaim,
    QueuedWorkClaimBoundary, QueuedWorkCompletion, QueuedWorkItem,
};
use lash_core::store::queued_work::{
    ClaimCandidate, QueuedWorkClaimLease, claim_scan_limit, derive_batch_id, renewed_claim,
    select_claim_prefix,
};
use lash_core::store::{
    GraphCommitDelta, HydratedSessionCheckpoint, PersistedSessionRead, RuntimeCommit,
    RuntimeCommitResult, SessionCheckpoint, SessionHeadMeta,
};
use lash_core::{
    AttachmentId, AttachmentIntent, AttachmentManifest, AttachmentManifestEntry, BlobRef,
    DeliveryPolicy, DurabilityTier, GcReport, MergeKey, ProcessAwaitOutput, ProcessEvent,
    ProcessEventAppendRequest, ProcessEventAppendResult, ProcessExternalRef,
    ProcessHandleDescriptor, ProcessHandleGrant, ProcessLease, ProcessLeaseCompletion,
    ProcessRecord, ProcessRegistration, ProcessRegistry, ProcessScope, RuntimePersistence,
    SessionMeta, SessionNodeRecord, SessionReadScope, SessionStoreCreateRequest,
    SessionStoreFactory, SlotPolicy, StoreError, TokenLedgerEntry, VacuumReport,
};
use lash_core::{
    HostEventOccurrenceRecord, HostEventOccurrenceRequest, HostEventStore, PluginError,
    TriggerDeliveryReservation, TriggerSubscriptionDraft, TriggerSubscriptionFilter,
    TriggerSubscriptionRecord,
};
use sha2::{Digest, Sha256};
use sqlx::postgres::{PgPool, PgPoolOptions, PgRow};
use sqlx::{Executor, Row};

const SCHEMA_COMPONENT: &str = "lash-postgres-store";
const SCHEMA_VERSION: i32 = 1;
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
    notify: Arc<tokio::sync::Notify>,
}

#[derive(Clone)]
pub struct PostgresHostEventStore {
    pool: PgPool,
}

#[derive(Clone)]
pub struct PostgresLashlangArtifactStore {
    pool: PgPool,
}

/// Connection-pool and per-connection timeout knobs for [`PostgresStorage`].
///
/// Session commits use **optimistic CAS** on the head (`UPDATE … WHERE
/// head_revision = expected`), not a held `SELECT … FOR UPDATE`, so concurrent
/// writers never pin a pool connection while blocked on a lock. `lock_timeout` is
/// defense in depth: it caps how long the single CAS write may wait on the head
/// row's lock before erroring (surfaced as a retryable conflict), so a pathological
/// burst can never starve the pool.
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
            notify: Arc::new(tokio::sync::Notify::new()),
        }
    }

    pub fn host_event_store(&self) -> PostgresHostEventStore {
        PostgresHostEventStore {
            pool: self.pool.clone(),
        }
    }

    pub fn lashlang_artifact_store(&self) -> PostgresLashlangArtifactStore {
        PostgresLashlangArtifactStore {
            pool: self.pool.clone(),
        }
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

async fn ensure_schema(pool: &PgPool) -> Result<(), StoreError> {
    let mut tx = pool.begin().await.map_err(store_sqlx_error)?;
    tx.execute("SELECT pg_advisory_xact_lock(715421, 907001)")
        .await
        .map_err(store_sqlx_error)?;
    tx.execute(
        r#"
        CREATE TABLE IF NOT EXISTS lash_schema_versions (
            component TEXT PRIMARY KEY,
            version INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS lash_blobs (
            hash TEXT PRIMARY KEY,
            content BYTEA NOT NULL
        );

        CREATE TABLE IF NOT EXISTS lash_sessions (
            session_id TEXT PRIMARY KEY,
            head_revision BIGINT NOT NULL DEFAULT 0,
            head_json TEXT NOT NULL,
            checkpoint_ref TEXT
        );

        CREATE TABLE IF NOT EXISTS lash_graph_nodes (
            session_id TEXT NOT NULL,
            seq BIGSERIAL,
            node_id TEXT NOT NULL,
            node_json TEXT NOT NULL,
            tombstoned BOOLEAN NOT NULL DEFAULT FALSE,
            PRIMARY KEY (session_id, node_id)
        );
        CREATE INDEX IF NOT EXISTS idx_lash_graph_nodes_seq
            ON lash_graph_nodes(session_id, seq);

        CREATE TABLE IF NOT EXISTS lash_usage_deltas (
            seq BIGSERIAL PRIMARY KEY,
            session_id TEXT NOT NULL,
            entry_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS lash_session_meta (
            session_id TEXT PRIMARY KEY,
            meta_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS lash_runtime_turn_commits (
            session_id TEXT NOT NULL,
            turn_id TEXT NOT NULL,
            turn_commit_hash TEXT NOT NULL,
            result_json TEXT NOT NULL,
            committed_at_ms BIGINT NOT NULL,
            PRIMARY KEY (session_id, turn_id)
        );

        CREATE TABLE IF NOT EXISTS lash_queued_work_batches (
            enqueue_seq BIGSERIAL PRIMARY KEY,
            batch_id TEXT NOT NULL UNIQUE,
            session_id TEXT NOT NULL,
            source_key TEXT,
            delivery_policy TEXT NOT NULL,
            slot_policy TEXT NOT NULL,
            merge_key_json TEXT NOT NULL,
            available_at_ms BIGINT NOT NULL,
            enqueued_at_ms BIGINT NOT NULL,
            claim_id TEXT,
            claim_owner_id TEXT,
            claim_token TEXT,
            claim_fencing_token BIGINT NOT NULL DEFAULT 0,
            claim_claimed_at_ms BIGINT NOT NULL DEFAULT 0,
            claim_expires_at_ms BIGINT NOT NULL DEFAULT 0,
            UNIQUE (session_id, source_key)
        );
        CREATE INDEX IF NOT EXISTS idx_lash_queued_work_ready
            ON lash_queued_work_batches(session_id, available_at_ms, enqueue_seq);

        CREATE TABLE IF NOT EXISTS lash_queued_work_items (
            batch_id TEXT NOT NULL REFERENCES lash_queued_work_batches(batch_id) ON DELETE CASCADE,
            item_index INTEGER NOT NULL,
            item_id TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            PRIMARY KEY (batch_id, item_index)
        );

        CREATE TABLE IF NOT EXISTS lash_attachment_manifest (
            attachment_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            canonical_uri TEXT NOT NULL,
            intent_at_ms BIGINT NOT NULL,
            committed_at_ms BIGINT
        );
        CREATE INDEX IF NOT EXISTS idx_lash_attachment_manifest_uncommitted
            ON lash_attachment_manifest(committed_at_ms)
            WHERE committed_at_ms IS NULL;

        CREATE TABLE IF NOT EXISTS lash_processes (
            process_id TEXT PRIMARY KEY,
            registration_hash TEXT NOT NULL,
            owner_scope_id TEXT NOT NULL,
            host_profile_id TEXT NOT NULL,
            created_at_ms BIGINT NOT NULL,
            updated_at_ms BIGINT NOT NULL,
            status TEXT NOT NULL,
            record_json TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_lash_processes_status
            ON lash_processes(status);

        CREATE TABLE IF NOT EXISTS lash_process_events (
            process_id TEXT NOT NULL REFERENCES lash_processes(process_id) ON DELETE CASCADE,
            sequence BIGINT NOT NULL,
            event_type TEXT NOT NULL,
            payload_hash TEXT NOT NULL,
            idempotency_key TEXT,
            occurred_at_ms BIGINT NOT NULL,
            event_json TEXT NOT NULL,
            PRIMARY KEY (process_id, sequence)
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_lash_process_events_key
            ON lash_process_events(process_id, idempotency_key)
            WHERE idempotency_key IS NOT NULL;

        CREATE TABLE IF NOT EXISTS lash_process_wake_acks (
            process_id TEXT NOT NULL REFERENCES lash_processes(process_id) ON DELETE CASCADE,
            sequence BIGINT NOT NULL,
            PRIMARY KEY (process_id, sequence)
        );

        CREATE TABLE IF NOT EXISTS lash_process_handle_grants (
            session_id TEXT NOT NULL,
            scope_id TEXT NOT NULL,
            process_id TEXT NOT NULL REFERENCES lash_processes(process_id) ON DELETE CASCADE,
            descriptor_json TEXT NOT NULL,
            PRIMARY KEY (scope_id, process_id)
        );
        CREATE INDEX IF NOT EXISTS idx_lash_process_handle_grants_session
            ON lash_process_handle_grants(session_id);
        CREATE INDEX IF NOT EXISTS idx_lash_process_handle_grants_process
            ON lash_process_handle_grants(process_id);

        CREATE TABLE IF NOT EXISTS lash_process_leases (
            process_id TEXT PRIMARY KEY REFERENCES lash_processes(process_id) ON DELETE CASCADE,
            lease_owner_id TEXT,
            lease_token TEXT,
            lease_fencing_token BIGINT NOT NULL DEFAULT 0,
            lease_claimed_at_ms BIGINT NOT NULL DEFAULT 0,
            lease_expires_at_ms BIGINT NOT NULL DEFAULT 0
        );

        CREATE SEQUENCE IF NOT EXISTS lash_host_event_subscription_seq;
        CREATE TABLE IF NOT EXISTS lash_host_event_trigger_subscriptions (
            subscription_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            handle TEXT NOT NULL,
            source_type TEXT NOT NULL,
            source_key TEXT NOT NULL,
            enabled BOOLEAN NOT NULL,
            created_at_ms BIGINT NOT NULL,
            updated_at_ms BIGINT NOT NULL,
            record_json TEXT NOT NULL,
            UNIQUE(session_id, handle)
        );
        CREATE INDEX IF NOT EXISTS idx_lash_host_event_subscriptions_source
            ON lash_host_event_trigger_subscriptions(source_type, source_key, enabled);

        CREATE TABLE IF NOT EXISTS lash_host_event_occurrences (
            occurrence_id TEXT PRIMARY KEY,
            idempotency_key TEXT NOT NULL UNIQUE,
            request_hash TEXT NOT NULL,
            source_type TEXT NOT NULL,
            source_key TEXT NOT NULL,
            occurred_at_ms BIGINT NOT NULL,
            record_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS lash_host_event_deliveries (
            occurrence_id TEXT NOT NULL REFERENCES lash_host_event_occurrences(occurrence_id) ON DELETE CASCADE,
            subscription_id TEXT NOT NULL REFERENCES lash_host_event_trigger_subscriptions(subscription_id) ON DELETE CASCADE,
            process_id TEXT NOT NULL,
            created_at_ms BIGINT NOT NULL,
            PRIMARY KEY (occurrence_id, subscription_id)
        );

        CREATE TABLE IF NOT EXISTS lash_lashlang_artifacts (
            module_ref TEXT PRIMARY KEY,
            artifact_bytes BYTEA NOT NULL
        );
        "#,
    )
    .await
    .map_err(store_sqlx_error)?;

    let existing: Option<i32> =
        sqlx::query_scalar("SELECT version FROM lash_schema_versions WHERE component = $1")
            .bind(SCHEMA_COMPONENT)
            .fetch_optional(&mut *tx)
            .await
            .map_err(store_sqlx_error)?;
    match existing {
        Some(version) if version == SCHEMA_VERSION => {}
        Some(version) => {
            return Err(StoreError::Backend(format!(
                "Postgres schema component `{SCHEMA_COMPONENT}` has version {version}, expected {SCHEMA_VERSION}"
            )));
        }
        None => {
            sqlx::query("INSERT INTO lash_schema_versions (component, version) VALUES ($1, $2)")
                .bind(SCHEMA_COMPONENT)
                .bind(SCHEMA_VERSION)
                .execute(&mut *tx)
                .await
                .map_err(store_sqlx_error)?;
        }
    }
    tx.commit().await.map_err(store_sqlx_error)
}

fn current_epoch_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn current_timestamp_string() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format!("unix:{}", now.as_secs())
}

fn store_sqlx_error(err: sqlx::Error) -> StoreError {
    StoreError::Backend(err.to_string())
}

/// Postgres SQLSTATEs that signal transient write contention rather than a hard
/// failure: serialization failure, deadlock, and lock-acquisition timeout. On the
/// session head these all mean "a concurrent committer got there first" — i.e. a
/// revision conflict the caller should reload-and-retry, not a backend error.
fn is_contention_error(err: &sqlx::Error) -> bool {
    matches!(
        err.as_database_error().and_then(|db| db.code()).as_deref(),
        Some("40001" | "40P01" | "55P03")
    )
}

fn plugin_sqlx_error(err: sqlx::Error) -> PluginError {
    PluginError::Session(err.to_string())
}

fn process_decode_error(err: serde_json::Error) -> PluginError {
    PluginError::Session(format!("failed to decode process registry row: {err}"))
}

fn store_decode_json<T: serde::de::DeserializeOwned>(
    json: &str,
    what: &str,
) -> Result<T, StoreError> {
    serde_json::from_str(json)
        .map_err(|err| StoreError::Backend(format!("failed to decode {what}: {err}")))
}

fn encode_json<T: serde::Serialize>(value: &T) -> String {
    serde_json::to_string(value).expect("persisted state should serialize")
}

fn encode_msgpack<T: serde::Serialize>(value: &T) -> Vec<u8> {
    let mut buf = Vec::with_capacity(1024);
    rmp_serde::encode::write_named(&mut buf, value).expect("value should serialize");
    buf
}

fn decode_msgpack<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> Option<T> {
    rmp_serde::from_slice(bytes).ok()
}

fn block_on_detached<T: Send + 'static>(
    future: impl std::future::Future<Output = T> + Send + 'static,
) -> T {
    std::thread::spawn(move || {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("postgres manifest runtime")
            .block_on(future)
    })
    .join()
    .expect("postgres manifest thread")
}

fn merge_token_ledger_entries(entries: Vec<TokenLedgerEntry>) -> Vec<TokenLedgerEntry> {
    let mut merged = Vec::<TokenLedgerEntry>::new();
    for entry in entries {
        if entry.usage.total() == 0 {
            continue;
        }
        if let Some(existing) = merged
            .iter_mut()
            .find(|existing| existing.source == entry.source && existing.model == entry.model)
        {
            existing.usage.input_tokens += entry.usage.input_tokens;
            existing.usage.output_tokens += entry.usage.output_tokens;
            existing.usage.cached_input_tokens += entry.usage.cached_input_tokens;
            existing.usage.reasoning_tokens += entry.usage.reasoning_tokens;
        } else {
            merged.push(entry);
        }
    }
    merged
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct SessionCheckpointEnvelope {
    manifest: SessionCheckpoint,
    tool_state: Option<lash_core::ToolState>,
    plugin_snapshot: Option<lash_core::PluginSessionSnapshot>,
    execution_state: Option<Vec<u8>>,
}

async fn put_blob_tx(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    content: &[u8],
) -> Result<BlobRef, StoreError> {
    let hash = format!("{:x}", Sha256::digest(content));
    sqlx::query(
        "INSERT INTO lash_blobs (hash, content)
         VALUES ($1, $2)
         ON CONFLICT (hash) DO NOTHING",
    )
    .bind(&hash)
    .bind(content)
    .execute(&mut **tx)
    .await
    .map_err(store_sqlx_error)?;
    Ok(BlobRef(hash))
}

async fn get_blob(pool: &PgPool, blob_ref: &BlobRef) -> Option<Vec<u8>> {
    sqlx::query_scalar::<_, Vec<u8>>("SELECT content FROM lash_blobs WHERE hash = $1")
        .bind(blob_ref.as_str())
        .fetch_optional(pool)
        .await
        .ok()
        .flatten()
}

async fn put_checkpoint_tx(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    checkpoint: &HydratedSessionCheckpoint,
) -> Result<(BlobRef, SessionCheckpoint), StoreError> {
    let manifest = SessionCheckpoint {
        turn_state: checkpoint.turn_state.clone(),
        tool_state_ref: checkpoint.tool_state_ref.clone(),
        plugin_snapshot_ref: checkpoint.plugin_snapshot_ref.clone(),
        plugin_snapshot_revision: checkpoint.plugin_snapshot_revision,
        execution_state_ref: checkpoint.execution_state_ref.clone(),
    };
    let envelope = SessionCheckpointEnvelope {
        manifest: manifest.clone(),
        tool_state: checkpoint.tool_state.clone(),
        plugin_snapshot: checkpoint.plugin_snapshot.clone(),
        execution_state: checkpoint.execution_state.clone(),
    };
    let bytes = encode_msgpack(&envelope);
    let checkpoint_ref = put_blob_tx(tx, &bytes).await?;
    Ok((checkpoint_ref, manifest))
}

async fn get_checkpoint(pool: &PgPool, blob_ref: &BlobRef) -> Option<HydratedSessionCheckpoint> {
    let bytes = get_blob(pool, blob_ref).await?;
    let envelope: SessionCheckpointEnvelope = decode_msgpack(&bytes)?;
    Some(HydratedSessionCheckpoint {
        turn_state: envelope.manifest.turn_state,
        tool_state_ref: envelope.manifest.tool_state_ref,
        tool_state: envelope.tool_state,
        plugin_snapshot_ref: envelope.manifest.plugin_snapshot_ref,
        plugin_snapshot: envelope.plugin_snapshot,
        plugin_snapshot_revision: envelope.manifest.plugin_snapshot_revision,
        execution_state_ref: envelope.manifest.execution_state_ref,
        execution_state: envelope.execution_state,
    })
}

async fn load_session_head_meta_tx(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    session_id: &str,
    for_update: bool,
) -> Result<Option<SessionHeadMeta>, StoreError> {
    let sql = if for_update {
        "SELECT head_json, head_revision FROM lash_sessions WHERE session_id = $1 FOR UPDATE"
    } else {
        "SELECT head_json, head_revision FROM lash_sessions WHERE session_id = $1"
    };
    let row = sqlx::query(sql)
        .bind(session_id)
        .fetch_optional(&mut **tx)
        .await
        .map_err(store_sqlx_error)?;
    let Some(row) = row else {
        return Ok(None);
    };
    let head_json: String = row.get(0);
    let head_revision: i64 = row.get(1);
    let mut meta: SessionHeadMeta = store_decode_json(&head_json, "session head")?;
    meta.head_revision = head_revision as u64;
    Ok(Some(meta))
}

async fn load_usage_deltas(pool: &PgPool, session_id: &str) -> Vec<TokenLedgerEntry> {
    let rows = sqlx::query(
        "SELECT entry_json FROM lash_usage_deltas WHERE session_id = $1 ORDER BY seq ASC",
    )
    .bind(session_id)
    .fetch_all(pool)
    .await
    .unwrap_or_default();
    rows.into_iter()
        .filter_map(|row| {
            let json: String = row.get(0);
            serde_json::from_str(&json).ok()
        })
        .collect()
}

async fn load_graph(
    pool: &PgPool,
    session_id: &str,
    leaf_node_id: Option<String>,
    active_path: bool,
) -> Result<lash_core::SessionGraph, StoreError> {
    let rows = sqlx::query(
        "SELECT node_json FROM lash_graph_nodes
         WHERE session_id = $1 AND tombstoned = FALSE
         ORDER BY seq ASC",
    )
    .bind(session_id)
    .fetch_all(pool)
    .await
    .map_err(store_sqlx_error)?;
    let mut nodes = Vec::<SessionNodeRecord>::new();
    for row in rows {
        let json: String = row.get(0);
        nodes.push(store_decode_json(&json, "session graph node")?);
    }
    if active_path && let Some(leaf) = leaf_node_id.clone() {
        let wanted = active_path_node_ids(&nodes, &leaf);
        nodes.retain(|node| wanted.contains(&node.node_id));
    }
    Ok(lash_core::SessionGraph::from_nodes(nodes, leaf_node_id))
}

fn active_path_node_ids(nodes: &[SessionNodeRecord], leaf_node_id: &str) -> HashSet<String> {
    let mut parent_by_id = std::collections::BTreeMap::new();
    for node in nodes {
        parent_by_id.insert(node.node_id.clone(), node.parent_node_id.clone());
    }
    let mut wanted = HashSet::new();
    let mut cursor = Some(leaf_node_id.to_string());
    while let Some(node_id) = cursor {
        if !wanted.insert(node_id.clone()) {
            break;
        }
        cursor = parent_by_id.get(&node_id).cloned().flatten();
    }
    wanted
}

async fn commit_attachment_refs_tx(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    session_id: &str,
    attachment_ids: &[AttachmentId],
) -> Result<(), StoreError> {
    if attachment_ids.is_empty() {
        return Ok(());
    }
    let now = current_epoch_ms() as i64;
    for id in attachment_ids {
        sqlx::query(
            "UPDATE lash_attachment_manifest
             SET committed_at_ms = COALESCE(committed_at_ms, $1)
             WHERE attachment_id = $2 AND session_id = $3",
        )
        .bind(now)
        .bind(id.as_str())
        .bind(session_id)
        .execute(&mut **tx)
        .await
        .map_err(store_sqlx_error)?;
    }
    Ok(())
}

impl AttachmentManifest for PostgresSessionStore {
    fn record_intent(&self, intent: AttachmentIntent) -> Result<(), StoreError> {
        let pool = self.pool.clone();
        block_on_detached(async move {
            sqlx::query(
                "INSERT INTO lash_attachment_manifest (
                    attachment_id, session_id, canonical_uri, intent_at_ms, committed_at_ms
                 )
                 VALUES ($1, $2, $3, $4, NULL)
                 ON CONFLICT (attachment_id) DO UPDATE SET
                    session_id = EXCLUDED.session_id,
                    canonical_uri = EXCLUDED.canonical_uri,
                    intent_at_ms = EXCLUDED.intent_at_ms",
            )
            .bind(intent.attachment_id.as_str())
            .bind(intent.session_id)
            .bind(intent.canonical_uri)
            .bind(intent.intent_at_epoch_ms as i64)
            .execute(&pool)
            .await
            .map(|_| ())
            .map_err(store_sqlx_error)
        })
    }

    fn commit_refs(
        &self,
        session_id: &str,
        attachment_ids: &[AttachmentId],
    ) -> Result<(), StoreError> {
        let pool = self.pool.clone();
        let session_id = session_id.to_string();
        let attachment_ids = attachment_ids.to_vec();
        block_on_detached(async move {
            let mut tx = pool.begin().await.map_err(store_sqlx_error)?;
            commit_attachment_refs_tx(&mut tx, &session_id, &attachment_ids).await?;
            tx.commit().await.map_err(store_sqlx_error)
        })
    }

    fn list_uncommitted(
        &self,
        older_than_epoch_ms: u64,
    ) -> Result<Vec<AttachmentManifestEntry>, StoreError> {
        let pool = self.pool.clone();
        block_on_detached(async move {
            let rows = sqlx::query(
                "SELECT attachment_id, session_id, canonical_uri, intent_at_ms, committed_at_ms
                 FROM lash_attachment_manifest
                 WHERE committed_at_ms IS NULL AND intent_at_ms <= $1
                 ORDER BY attachment_id ASC",
            )
            .bind(older_than_epoch_ms as i64)
            .fetch_all(&pool)
            .await
            .map_err(store_sqlx_error)?;
            Ok(rows
                .into_iter()
                .map(|row| AttachmentManifestEntry {
                    attachment_id: AttachmentId::new(row.get::<String, _>(0)),
                    session_id: row.get(1),
                    canonical_uri: row.get(2),
                    intent_at_epoch_ms: row.get::<i64, _>(3) as u64,
                    committed_at_epoch_ms: row.get::<Option<i64>, _>(4).map(|value| value as u64),
                })
                .collect())
        })
    }

    fn forget(&self, attachment_id: &AttachmentId) -> Result<(), StoreError> {
        let pool = self.pool.clone();
        let attachment_id = attachment_id.to_string();
        block_on_detached(async move {
            sqlx::query("DELETE FROM lash_attachment_manifest WHERE attachment_id = $1")
                .bind(attachment_id)
                .execute(&pool)
                .await
                .map(|_| ())
                .map_err(store_sqlx_error)
        })
    }
}

#[async_trait::async_trait]
impl SessionStoreFactory for PostgresSessionStoreFactory {
    fn durability_tier(&self) -> DurabilityTier {
        DurabilityTier::Durable
    }

    async fn create_store(
        &self,
        request: &SessionStoreCreateRequest,
    ) -> Result<Arc<dyn RuntimePersistence>, String> {
        let store = PostgresSessionStore {
            pool: self.pool.clone(),
            session_id: Some(request.session_id.clone()),
            bound_session: Arc::new(OnceLock::new()),
        };
        if store
            .load_session_meta()
            .await
            .map_err(|err| err.to_string())?
            .is_none()
        {
            store
                .save_session_meta(SessionMeta {
                    session_id: request.session_id.clone(),
                    session_name: request.session_id.clone(),
                    created_at: current_timestamp_string(),
                    model: request.policy.model.id.clone(),
                    cwd: std::env::current_dir()
                        .ok()
                        .and_then(|path| path.to_str().map(str::to_string)),
                    relation: request.relation.clone(),
                })
                .await
                .map_err(|err| err.to_string())?;
        }
        Ok(Arc::new(store))
    }

    async fn delete_session(&self, session_id: &str) -> Result<(), String> {
        let mut tx = self.pool.begin().await.map_err(|err| err.to_string())?;
        for sql in [
            "DELETE FROM lash_queued_work_items WHERE batch_id IN (SELECT batch_id FROM lash_queued_work_batches WHERE session_id = $1)",
            "DELETE FROM lash_queued_work_batches WHERE session_id = $1",
            "DELETE FROM lash_usage_deltas WHERE session_id = $1",
            "DELETE FROM lash_graph_nodes WHERE session_id = $1",
            "DELETE FROM lash_runtime_turn_commits WHERE session_id = $1",
            "DELETE FROM lash_session_meta WHERE session_id = $1",
            "DELETE FROM lash_sessions WHERE session_id = $1",
            "DELETE FROM lash_attachment_manifest WHERE session_id = $1",
        ] {
            sqlx::query(sql)
                .bind(session_id)
                .execute(&mut *tx)
                .await
                .map_err(|err| err.to_string())?;
        }
        tx.commit().await.map_err(|err| err.to_string())
    }
}

#[derive(Clone, Debug)]
struct QueuedBatchRow {
    enqueue_seq: u64,
    batch_id: String,
    session_id: String,
    source_key: Option<String>,
    delivery_policy: DeliveryPolicy,
    slot_policy: SlotPolicy,
    merge_key: MergeKey,
    available_at_ms: u64,
    enqueued_at_ms: u64,
    claim_fencing_token: u64,
}

fn queued_batch_row(row: PgRow) -> Result<QueuedBatchRow, StoreError> {
    let delivery_policy =
        DeliveryPolicy::from_wire_str(row.get::<String, _>("delivery_policy").as_str())
            .ok_or_else(|| {
                StoreError::Backend("invalid queued work delivery policy".to_string())
            })?;
    let slot_policy = SlotPolicy::from_wire_str(row.get::<String, _>("slot_policy").as_str())
        .ok_or_else(|| StoreError::Backend("invalid queued work slot policy".to_string()))?;
    let merge_json: String = row.get("merge_key_json");
    Ok(QueuedBatchRow {
        enqueue_seq: row.get::<i64, _>("enqueue_seq") as u64,
        batch_id: row.get("batch_id"),
        session_id: row.get("session_id"),
        source_key: row.get("source_key"),
        delivery_policy,
        slot_policy,
        merge_key: store_decode_json(&merge_json, "queued work merge key")?,
        available_at_ms: row.get::<i64, _>("available_at_ms") as u64,
        enqueued_at_ms: row.get::<i64, _>("enqueued_at_ms") as u64,
        claim_fencing_token: row.get::<i64, _>("claim_fencing_token") as u64,
    })
}

async fn load_queued_batch(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    batch_id: &str,
) -> Result<Option<QueuedWorkBatch>, StoreError> {
    let row = sqlx::query(
        "SELECT enqueue_seq, batch_id, session_id, source_key, delivery_policy,
                slot_policy, merge_key_json, available_at_ms, enqueued_at_ms,
                claim_fencing_token
         FROM lash_queued_work_batches
         WHERE batch_id = $1",
    )
    .bind(batch_id)
    .fetch_optional(&mut **tx)
    .await
    .map_err(store_sqlx_error)?;
    let Some(row) = row else {
        return Ok(None);
    };
    let row = queued_batch_row(row)?;
    queued_work_batch_from_row(tx, row).await.map(Some)
}

async fn queued_work_batch_from_row(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    row: QueuedBatchRow,
) -> Result<QueuedWorkBatch, StoreError> {
    let item_rows = sqlx::query(
        "SELECT item_id, payload_json
         FROM lash_queued_work_items
         WHERE batch_id = $1
         ORDER BY item_index ASC",
    )
    .bind(&row.batch_id)
    .fetch_all(&mut **tx)
    .await
    .map_err(store_sqlx_error)?;
    let mut items = Vec::new();
    for item in item_rows {
        let payload_json: String = item.get(1);
        items.push(QueuedWorkItem {
            item_id: item.get(0),
            payload: store_decode_json(&payload_json, "queued work payload")?,
        });
    }
    Ok(QueuedWorkBatch {
        batch_id: row.batch_id,
        session_id: row.session_id,
        enqueue_seq: row.enqueue_seq,
        source_key: row.source_key,
        delivery_policy: row.delivery_policy,
        slot_policy: row.slot_policy,
        merge_key: row.merge_key,
        available_at_ms: row.available_at_ms,
        enqueued_at_ms: row.enqueued_at_ms,
        items,
    })
}

async fn ensure_queued_work_completion_tx(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    completed: &QueuedWorkCompletion,
) -> Result<(), StoreError> {
    for batch_id in &completed.batch_ids {
        let exists: Option<i64> = sqlx::query_scalar(
            "SELECT 1::BIGINT FROM lash_queued_work_batches
             WHERE session_id = $1
               AND batch_id = $2
               AND claim_id = $3
               AND claim_token = $4
             LIMIT 1",
        )
        .bind(&completed.session_id)
        .bind(batch_id)
        .bind(&completed.claim_id)
        .bind(&completed.lease_token)
        .fetch_optional(&mut **tx)
        .await
        .map_err(store_sqlx_error)?;
        if exists.is_none() {
            return Err(StoreError::QueuedWorkClaimExpired {
                session_id: completed.session_id.clone(),
                claim_id: completed.claim_id.clone(),
            });
        }
    }
    Ok(())
}

#[async_trait::async_trait]
impl RuntimePersistence for PostgresSessionStore {
    fn durability_tier(&self) -> DurabilityTier {
        DurabilityTier::Durable
    }

    async fn load_session(
        &self,
        scope: SessionReadScope,
    ) -> Result<Option<PersistedSessionRead>, StoreError> {
        let Some(session_id) = self.selected_session_id().await? else {
            return Ok(None);
        };
        let mut tx = self.pool.begin().await.map_err(store_sqlx_error)?;
        let Some(meta) = load_session_head_meta_tx(&mut tx, &session_id, false).await? else {
            return Ok(None);
        };
        tx.commit().await.map_err(store_sqlx_error)?;
        let leaf_node_id = match &scope {
            SessionReadScope::FullGraph => meta.leaf_node_id.clone(),
            SessionReadScope::ActivePath { leaf_node_id } => {
                leaf_node_id.clone().or_else(|| meta.leaf_node_id.clone())
            }
        };
        let graph = load_graph(
            &self.pool,
            &session_id,
            leaf_node_id.clone(),
            matches!(scope, SessionReadScope::ActivePath { .. }),
        )
        .await?;
        let checkpoint = match meta.checkpoint_ref.as_ref() {
            Some(blob_ref) => get_checkpoint(&self.pool, blob_ref).await,
            None => None,
        };
        Ok(Some(PersistedSessionRead {
            session_id: meta.session_id,
            head_revision: meta.head_revision,
            config: meta.config,
            agent_frames: meta.agent_frames,
            current_agent_frame_id: meta.current_agent_frame_id,
            graph,
            checkpoint_ref: meta.checkpoint_ref,
            checkpoint,
            token_ledger: merge_token_ledger_entries(
                load_usage_deltas(&self.pool, &session_id).await,
            ),
        }))
    }

    async fn load_node(&self, node_id: &str) -> Result<Option<SessionNodeRecord>, StoreError> {
        let json: Option<String> = if let Some(session_id) = &self.session_id {
            sqlx::query_scalar(
                "SELECT node_json FROM lash_graph_nodes
                 WHERE session_id = $1 AND node_id = $2 AND tombstoned = FALSE",
            )
            .bind(session_id)
            .bind(node_id)
            .fetch_optional(&self.pool)
            .await
            .map_err(store_sqlx_error)?
        } else {
            sqlx::query_scalar(
                "SELECT node_json FROM lash_graph_nodes
                 WHERE node_id = $1 AND tombstoned = FALSE
                 ORDER BY session_id ASC
                 LIMIT 1",
            )
            .bind(node_id)
            .fetch_optional(&self.pool)
            .await
            .map_err(store_sqlx_error)?
        };
        json.map(|json| store_decode_json(&json, "session graph node"))
            .transpose()
    }

    async fn commit_runtime_state(
        &self,
        commit: RuntimeCommit,
    ) -> Result<RuntimeCommitResult, StoreError> {
        let mut tx = self.pool.begin().await.map_err(store_sqlx_error)?;
        // Read the head WITHOUT a lock; the conditional CAS write below is the
        // serialization point (optimistic concurrency), so no pessimistic
        // `FOR UPDATE` lock is held across the rest of this transaction.
        let existing = load_session_head_meta_tx(&mut tx, &commit.session_id, false).await?;
        if let Some(bound_session_id) = existing.as_ref().map(|meta| meta.session_id.as_str())
            && bound_session_id != commit.session_id
        {
            return Err(StoreError::SessionBindingMismatch {
                bound_session_id: bound_session_id.to_string(),
                attempted_session_id: commit.session_id,
            });
        }
        // A session-store handle commits to exactly one session. An explicit
        // binding (`self.session_id`) is authoritative; otherwise the handle binds
        // to the first session it commits and rejects any other thereafter.
        let effective_binding = self
            .session_id
            .clone()
            .or_else(|| self.bound_session.get().cloned());
        if let Some(bound_session_id) = &effective_binding
            && commit.session_id != *bound_session_id
        {
            return Err(StoreError::SessionBindingMismatch {
                bound_session_id: bound_session_id.clone(),
                attempted_session_id: commit.session_id,
            });
        }
        if self.session_id.is_none() {
            let _ = self.bound_session.set(commit.session_id.clone());
        }
        if let Some(completed) = &commit.turn_commit {
            if completed.session_id != commit.session_id {
                return Err(StoreError::RuntimeTurnCommitConflict {
                    session_id: completed.session_id.clone(),
                    turn_id: completed.turn_id.clone(),
                });
            }
            let prior = sqlx::query(
                "SELECT turn_commit_hash, result_json
                 FROM lash_runtime_turn_commits
                 WHERE session_id = $1 AND turn_id = $2",
            )
            .bind(&completed.session_id)
            .bind(&completed.turn_id)
            .fetch_optional(&mut *tx)
            .await
            .map_err(store_sqlx_error)?;
            if let Some(row) = prior {
                let hash: String = row.get(0);
                let result_json: String = row.get(1);
                if hash == completed.turn_commit_hash {
                    return store_decode_json(&result_json, "runtime turn commit result");
                }
                return Err(StoreError::RuntimeTurnCommitConflict {
                    session_id: completed.session_id.clone(),
                    turn_id: completed.turn_id.clone(),
                });
            }
        }
        let actual_revision = existing.as_ref().map_or(0, |meta| meta.head_revision);
        if commit.expected_head_revision.is_some()
            && commit.expected_head_revision != Some(actual_revision)
        {
            return Err(StoreError::HeadRevisionConflict {
                expected: commit.expected_head_revision,
                actual: actual_revision,
            });
        }
        for completed in &commit.completed_queue_claims {
            if completed.session_id != commit.session_id {
                return Err(StoreError::QueuedWorkClaimExpired {
                    session_id: completed.session_id.clone(),
                    claim_id: completed.claim_id.clone(),
                });
            }
            ensure_queued_work_completion_tx(&mut tx, completed).await?;
        }
        let (checkpoint_ref, manifest) = put_checkpoint_tx(&mut tx, &commit.checkpoint).await?;
        for entry in &commit.usage_deltas {
            sqlx::query("INSERT INTO lash_usage_deltas (session_id, entry_json) VALUES ($1, $2)")
                .bind(&commit.session_id)
                .bind(encode_json(entry))
                .execute(&mut *tx)
                .await
                .map_err(store_sqlx_error)?;
        }
        let leaf_node_id = match &commit.graph {
            GraphCommitDelta::Unchanged { leaf_node_id } => leaf_node_id.clone(),
            GraphCommitDelta::Append {
                nodes,
                leaf_node_id,
            } => {
                for node in nodes {
                    sqlx::query(
                        "INSERT INTO lash_graph_nodes (session_id, node_id, node_json)
                         VALUES ($1, $2, $3)
                         ON CONFLICT (session_id, node_id) DO UPDATE SET
                            node_json = EXCLUDED.node_json,
                            tombstoned = FALSE",
                    )
                    .bind(&commit.session_id)
                    .bind(&node.node_id)
                    .bind(encode_json(node))
                    .execute(&mut *tx)
                    .await
                    .map_err(store_sqlx_error)?;
                }
                leaf_node_id.clone()
            }
            GraphCommitDelta::ReplaceFull(graph) => {
                sqlx::query("DELETE FROM lash_graph_nodes WHERE session_id = $1")
                    .bind(&commit.session_id)
                    .execute(&mut *tx)
                    .await
                    .map_err(store_sqlx_error)?;
                for node in &graph.nodes {
                    sqlx::query(
                        "INSERT INTO lash_graph_nodes (session_id, node_id, node_json)
                         VALUES ($1, $2, $3)",
                    )
                    .bind(&commit.session_id)
                    .bind(&node.node_id)
                    .bind(encode_json(node))
                    .execute(&mut *tx)
                    .await
                    .map_err(store_sqlx_error)?;
                }
                graph.leaf_node_id.clone()
            }
        };
        let graph_node_count: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM lash_graph_nodes WHERE session_id = $1 AND tombstoned = FALSE",
        )
        .bind(&commit.session_id)
        .fetch_one(&mut *tx)
        .await
        .map_err(store_sqlx_error)?;
        let next_revision = actual_revision + 1;
        let meta = SessionHeadMeta {
            session_id: commit.session_id.clone(),
            head_revision: next_revision,
            config: commit.config.clone(),
            agent_frames: commit.agent_frames.clone(),
            current_agent_frame_id: commit.current_agent_frame_id.clone(),
            checkpoint_ref: Some(checkpoint_ref.clone()),
            leaf_node_id,
            graph_node_count: graph_node_count as usize,
            token_ledger: Vec::new(),
        };
        // Optimistic CAS on the head revision. The `WHERE head_revision = $5`
        // guard makes the write succeed only if no concurrent committer moved the
        // head since our unlocked read above. A brand-new session inserts (no
        // conflict); an existing one updates only when the revision still matches.
        let head_write = sqlx::query(
            "INSERT INTO lash_sessions (session_id, head_revision, head_json, checkpoint_ref)
             VALUES ($1, $2, $3, $4)
             ON CONFLICT (session_id) DO UPDATE SET
                head_revision = EXCLUDED.head_revision,
                head_json = EXCLUDED.head_json,
                checkpoint_ref = EXCLUDED.checkpoint_ref
             WHERE lash_sessions.head_revision = $5",
        )
        .bind(&commit.session_id)
        .bind(next_revision as i64)
        .bind(encode_json(&meta))
        .bind(checkpoint_ref.as_str())
        .bind(actual_revision as i64)
        .execute(&mut *tx)
        .await;
        let head_write = match head_write {
            Ok(result) => result,
            Err(err) if is_contention_error(&err) => {
                // The head row is contended by a concurrent committer (lock
                // timeout / serialization failure / deadlock). That is a conflict,
                // not an opaque backend error: surface it so the caller reloads
                // and retries. The tx is now aborted; returning drops it.
                return Err(StoreError::HeadRevisionConflict {
                    expected: commit.expected_head_revision.or(Some(actual_revision)),
                    actual: actual_revision,
                });
            }
            Err(err) => return Err(store_sqlx_error(err)),
        };
        if head_write.rows_affected() == 0 {
            // A concurrent commit won the race: the head no longer matches the
            // revision we read. Re-read the now-current revision for an accurate
            // report, then drop `tx` (auto-rollback), discarding this attempt's
            // node/usage writes; the caller reloads and retries.
            let actual_now = sqlx::query_scalar::<_, i64>(
                "SELECT head_revision FROM lash_sessions WHERE session_id = $1",
            )
            .bind(&commit.session_id)
            .fetch_optional(&mut *tx)
            .await
            .map_err(store_sqlx_error)?
            .map_or(actual_revision, |revision| revision as u64);
            return Err(StoreError::HeadRevisionConflict {
                expected: commit.expected_head_revision.or(Some(actual_revision)),
                actual: actual_now,
            });
        }
        for completed in &commit.completed_queue_claims {
            for batch_id in &completed.batch_ids {
                sqlx::query(
                    "DELETE FROM lash_queued_work_batches
                     WHERE session_id = $1 AND batch_id = $2 AND claim_id = $3 AND claim_token = $4",
                )
                .bind(&completed.session_id)
                .bind(batch_id)
                .bind(&completed.claim_id)
                .bind(&completed.lease_token)
                .execute(&mut *tx)
                .await
                .map_err(store_sqlx_error)?;
            }
        }
        commit_attachment_refs_tx(
            &mut tx,
            &commit.session_id,
            &commit.committed_attachment_ids,
        )
        .await?;
        let result = RuntimeCommitResult {
            head_revision: next_revision,
            checkpoint_ref,
            manifest,
        };
        if let Some(completed) = &commit.turn_commit {
            sqlx::query(
                "INSERT INTO lash_runtime_turn_commits (
                    session_id, turn_id, turn_commit_hash, result_json, committed_at_ms
                 )
                 VALUES ($1, $2, $3, $4, $5)",
            )
            .bind(&completed.session_id)
            .bind(&completed.turn_id)
            .bind(&completed.turn_commit_hash)
            .bind(encode_json(&result))
            .bind(current_epoch_ms() as i64)
            .execute(&mut *tx)
            .await
            .map_err(store_sqlx_error)?;
        }
        tx.commit().await.map_err(store_sqlx_error)?;
        Ok(result)
    }

    async fn enqueue_queued_work(
        &self,
        batch: QueuedWorkBatchDraft,
    ) -> Result<QueuedWorkBatch, StoreError> {
        let mut tx = self.pool.begin().await.map_err(store_sqlx_error)?;
        if let Some(source_key) = batch.source_key.as_deref() {
            let existing_id: Option<String> = sqlx::query_scalar(
                "SELECT batch_id FROM lash_queued_work_batches
                 WHERE session_id = $1 AND source_key = $2",
            )
            .bind(&batch.session_id)
            .bind(source_key)
            .fetch_optional(&mut *tx)
            .await
            .map_err(store_sqlx_error)?;
            if let Some(batch_id) = existing_id {
                let existing = load_queued_batch(&mut tx, &batch_id)
                    .await?
                    .ok_or_else(|| {
                        StoreError::Backend("queued work source row disappeared".to_string())
                    })?;
                tx.commit().await.map_err(store_sqlx_error)?;
                return Ok(existing);
            }
        }
        let now = current_epoch_ms();
        let batch_id = derive_batch_id(&batch.session_id, batch.source_key.as_deref(), now, None);
        let row = sqlx::query_scalar::<_, i64>(
            "INSERT INTO lash_queued_work_batches (
                batch_id, session_id, source_key, delivery_policy, slot_policy,
                merge_key_json, available_at_ms, enqueued_at_ms
             )
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
             RETURNING enqueue_seq",
        )
        .bind(&batch_id)
        .bind(&batch.session_id)
        .bind(&batch.source_key)
        .bind(batch.delivery_policy.as_str())
        .bind(batch.slot_policy.as_str())
        .bind(encode_json(&batch.merge_key))
        .bind(batch.available_at_ms as i64)
        .bind(now as i64)
        .fetch_one(&mut *tx)
        .await
        .map_err(store_sqlx_error)?;
        for (index, payload) in batch.payloads.iter().enumerate() {
            let item_id = format!("{batch_id}:item:{index}");
            sqlx::query(
                "INSERT INTO lash_queued_work_items (batch_id, item_index, item_id, payload_json)
                 VALUES ($1, $2, $3, $4)",
            )
            .bind(&batch_id)
            .bind(index as i32)
            .bind(item_id)
            .bind(encode_json(payload))
            .execute(&mut *tx)
            .await
            .map_err(store_sqlx_error)?;
        }
        let queued = load_queued_batch(&mut tx, &batch_id)
            .await?
            .ok_or_else(|| StoreError::Backend("queued work insert disappeared".to_string()))?;
        debug_assert_eq!(queued.enqueue_seq, row as u64);
        tx.commit().await.map_err(store_sqlx_error)?;
        Ok(queued)
    }

    async fn claim_ready_queued_work(
        &self,
        session_id: &str,
        owner_id: &str,
        boundary: QueuedWorkClaimBoundary,
        lease_ttl_ms: u64,
        max_batches: usize,
    ) -> Result<Option<QueuedWorkClaim>, StoreError> {
        if max_batches == 0 {
            return Ok(None);
        }
        let mut tx = self.pool.begin().await.map_err(store_sqlx_error)?;
        let now = current_epoch_ms();
        let rows = sqlx::query(
            "SELECT enqueue_seq, batch_id, session_id, source_key, delivery_policy,
                    slot_policy, merge_key_json, available_at_ms, enqueued_at_ms,
                    claim_fencing_token
             FROM lash_queued_work_batches
             WHERE session_id = $1
               AND available_at_ms <= $2
               AND (claim_token IS NULL OR claim_expires_at_ms <= $2)
             ORDER BY enqueue_seq ASC
             LIMIT $3
             FOR UPDATE SKIP LOCKED",
        )
        .bind(session_id)
        .bind(now as i64)
        .bind(claim_scan_limit(max_batches))
        .fetch_all(&mut *tx)
        .await
        .map_err(store_sqlx_error)?;
        let mut selected = Vec::new();
        for row in rows {
            selected.push(queued_batch_row(row)?);
        }
        let candidates = selected
            .iter()
            .map(|row| ClaimCandidate {
                enqueue_seq: row.enqueue_seq,
                claim_fencing_token: row.claim_fencing_token,
                delivery_policy: row.delivery_policy,
                slot_policy: row.slot_policy,
                merge_key: row.merge_key.clone(),
            })
            .collect::<Vec<_>>();
        let selected_len = select_claim_prefix(&candidates, boundary, max_batches);
        if selected_len == 0 {
            tx.commit().await.map_err(store_sqlx_error)?;
            return Ok(None);
        }
        selected.truncate(selected_len);
        let lease =
            QueuedWorkClaimLease::derive(&candidates[0], session_id, owner_id, now, lease_ttl_ms);
        for row in &selected {
            let changed = sqlx::query(
                "UPDATE lash_queued_work_batches
                 SET claim_id = $3,
                     claim_owner_id = $4,
                     claim_token = $5,
                     claim_fencing_token = claim_fencing_token + 1,
                     claim_claimed_at_ms = $6,
                     claim_expires_at_ms = $7
                 WHERE session_id = $1
                   AND batch_id = $2
                   AND (claim_token IS NULL OR claim_expires_at_ms <= $6)",
            )
            .bind(session_id)
            .bind(&row.batch_id)
            .bind(&lease.claim_id)
            .bind(owner_id)
            .bind(&lease.lease_token)
            .bind(now as i64)
            .bind(lease.expires_at_epoch_ms as i64)
            .execute(&mut *tx)
            .await
            .map_err(store_sqlx_error)?
            .rows_affected();
            if changed == 0 {
                tx.rollback().await.map_err(store_sqlx_error)?;
                return Ok(None);
            }
        }
        let mut batches = Vec::new();
        for row in selected {
            batches.push(queued_work_batch_from_row(&mut tx, row).await?);
        }
        tx.commit().await.map_err(store_sqlx_error)?;
        Ok(Some(QueuedWorkClaim {
            session_id: session_id.to_string(),
            claim_id: lease.claim_id,
            owner_id: owner_id.to_string(),
            lease_token: lease.lease_token,
            fencing_token: lease.fencing_token,
            claimed_at_epoch_ms: lease.claimed_at_epoch_ms,
            expires_at_epoch_ms: lease.expires_at_epoch_ms,
            batches,
        }))
    }

    async fn renew_queued_work_claim(
        &self,
        claim: &QueuedWorkClaim,
        lease_ttl_ms: u64,
    ) -> Result<QueuedWorkClaim, StoreError> {
        let expires_at = current_epoch_ms().saturating_add(lease_ttl_ms);
        let changed = sqlx::query(
            "UPDATE lash_queued_work_batches
             SET claim_expires_at_ms = $4
             WHERE session_id = $1 AND claim_id = $2 AND claim_token = $3",
        )
        .bind(&claim.session_id)
        .bind(&claim.claim_id)
        .bind(&claim.lease_token)
        .bind(expires_at as i64)
        .execute(&self.pool)
        .await
        .map_err(store_sqlx_error)?
        .rows_affected();
        renewed_claim(claim, changed as usize, expires_at)
    }

    async fn abandon_queued_work_claim(&self, claim: &QueuedWorkClaim) -> Result<(), StoreError> {
        sqlx::query(
            "UPDATE lash_queued_work_batches
             SET claim_id = NULL,
                 claim_owner_id = NULL,
                 claim_token = NULL,
                 claim_claimed_at_ms = 0,
                 claim_expires_at_ms = 0
             WHERE session_id = $1 AND claim_id = $2 AND claim_token = $3",
        )
        .bind(&claim.session_id)
        .bind(&claim.claim_id)
        .bind(&claim.lease_token)
        .execute(&self.pool)
        .await
        .map_err(store_sqlx_error)?;
        Ok(())
    }

    async fn cancel_queued_work_batch(
        &self,
        session_id: &str,
        batch_id: &str,
    ) -> Result<Option<QueuedWorkBatch>, StoreError> {
        let mut tx = self.pool.begin().await.map_err(store_sqlx_error)?;
        let now = current_epoch_ms();
        let row = sqlx::query(
            "SELECT enqueue_seq, batch_id, session_id, source_key, delivery_policy,
                    slot_policy, merge_key_json, available_at_ms, enqueued_at_ms,
                    claim_fencing_token
             FROM lash_queued_work_batches
             WHERE session_id = $1
               AND batch_id = $2
               AND (claim_token IS NULL OR claim_expires_at_ms <= $3)
             FOR UPDATE",
        )
        .bind(session_id)
        .bind(batch_id)
        .bind(now as i64)
        .fetch_optional(&mut *tx)
        .await
        .map_err(store_sqlx_error)?;
        let Some(row) = row else {
            tx.commit().await.map_err(store_sqlx_error)?;
            return Ok(None);
        };
        let batch = queued_work_batch_from_row(&mut tx, queued_batch_row(row)?).await?;
        sqlx::query("DELETE FROM lash_queued_work_batches WHERE batch_id = $1")
            .bind(batch_id)
            .execute(&mut *tx)
            .await
            .map_err(store_sqlx_error)?;
        tx.commit().await.map_err(store_sqlx_error)?;
        Ok(Some(batch))
    }

    async fn list_queued_work(&self, session_id: &str) -> Result<Vec<QueuedWorkBatch>, StoreError> {
        let mut tx = self.pool.begin().await.map_err(store_sqlx_error)?;
        let rows = sqlx::query(
            "SELECT enqueue_seq, batch_id, session_id, source_key, delivery_policy,
                    slot_policy, merge_key_json, available_at_ms, enqueued_at_ms,
                    claim_fencing_token
             FROM lash_queued_work_batches
             WHERE session_id = $1
             ORDER BY enqueue_seq ASC",
        )
        .bind(session_id)
        .fetch_all(&mut *tx)
        .await
        .map_err(store_sqlx_error)?;
        let mut batches = Vec::new();
        for row in rows {
            batches.push(queued_work_batch_from_row(&mut tx, queued_batch_row(row)?).await?);
        }
        tx.commit().await.map_err(store_sqlx_error)?;
        Ok(batches)
    }

    async fn list_pending_queued_work(
        &self,
        session_id: &str,
    ) -> Result<Vec<QueuedWorkBatch>, StoreError> {
        let mut tx = self.pool.begin().await.map_err(store_sqlx_error)?;
        let now = current_epoch_ms();
        let rows = sqlx::query(
            "SELECT enqueue_seq, batch_id, session_id, source_key, delivery_policy,
                    slot_policy, merge_key_json, available_at_ms, enqueued_at_ms,
                    claim_fencing_token
             FROM lash_queued_work_batches
             WHERE session_id = $1
               AND (claim_token IS NULL OR claim_expires_at_ms <= $2)
             ORDER BY enqueue_seq ASC",
        )
        .bind(session_id)
        .bind(now as i64)
        .fetch_all(&mut *tx)
        .await
        .map_err(store_sqlx_error)?;
        let mut batches = Vec::new();
        for row in rows {
            batches.push(queued_work_batch_from_row(&mut tx, queued_batch_row(row)?).await?);
        }
        tx.commit().await.map_err(store_sqlx_error)?;
        Ok(batches)
    }

    async fn save_session_meta(&self, meta: SessionMeta) -> Result<(), StoreError> {
        sqlx::query(
            "INSERT INTO lash_session_meta (session_id, meta_json)
             VALUES ($1, $2)
             ON CONFLICT (session_id) DO UPDATE SET meta_json = EXCLUDED.meta_json",
        )
        .bind(&meta.session_id)
        .bind(encode_json(&meta))
        .execute(&self.pool)
        .await
        .map_err(store_sqlx_error)?;
        Ok(())
    }

    async fn load_session_meta(&self) -> Result<Option<SessionMeta>, StoreError> {
        let json: Option<String> = if let Some(session_id) = &self.session_id {
            sqlx::query_scalar("SELECT meta_json FROM lash_session_meta WHERE session_id = $1")
                .bind(session_id)
                .fetch_optional(&self.pool)
                .await
                .map_err(store_sqlx_error)?
        } else {
            sqlx::query_scalar(
                "SELECT meta_json FROM lash_session_meta ORDER BY session_id ASC LIMIT 1",
            )
            .fetch_optional(&self.pool)
            .await
            .map_err(store_sqlx_error)?
        };
        json.map(|json| store_decode_json(&json, "session meta"))
            .transpose()
    }

    async fn tombstone_nodes(&self, ids: &[String]) -> Result<(), StoreError> {
        for id in ids {
            if let Some(session_id) = &self.session_id {
                sqlx::query(
                    "UPDATE lash_graph_nodes
                     SET tombstoned = TRUE
                     WHERE session_id = $1 AND node_id = $2",
                )
                .bind(session_id)
                .bind(id)
                .execute(&self.pool)
                .await
                .map_err(store_sqlx_error)?;
            } else {
                sqlx::query(
                    "UPDATE lash_graph_nodes
                     SET tombstoned = TRUE
                     WHERE node_id = $1",
                )
                .bind(id)
                .execute(&self.pool)
                .await
                .map_err(store_sqlx_error)?;
            }
        }
        Ok(())
    }

    async fn vacuum(&self) -> Result<VacuumReport, StoreError> {
        let removed = if let Some(session_id) = &self.session_id {
            sqlx::query("DELETE FROM lash_graph_nodes WHERE session_id = $1 AND tombstoned = TRUE")
                .bind(session_id)
                .execute(&self.pool)
                .await
                .map_err(store_sqlx_error)?
                .rows_affected()
        } else {
            sqlx::query("DELETE FROM lash_graph_nodes WHERE tombstoned = TRUE")
                .execute(&self.pool)
                .await
                .map_err(store_sqlx_error)?
                .rows_affected()
        };
        Ok(VacuumReport {
            removed_node_count: removed as usize,
        })
    }

    async fn gc_unreachable(&self) -> Result<GcReport, StoreError> {
        Ok(GcReport::default())
    }
}

fn process_status_label(record: &ProcessRecord) -> &'static str {
    record.status.label()
}

async fn load_process_tx(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    process_id: &str,
) -> Result<Option<ProcessRecord>, PluginError> {
    let json: Option<String> = sqlx::query_scalar(
        "SELECT record_json
             FROM lash_processes
             WHERE process_id = $1
             FOR UPDATE",
    )
    .bind(process_id)
    .fetch_optional(&mut **tx)
    .await
    .map_err(plugin_sqlx_error)?;
    json.map(|json| serde_json::from_str(&json).map_err(process_decode_error))
        .transpose()
}

async fn load_process(
    pool: &PgPool,
    process_id: &str,
) -> Result<Option<ProcessRecord>, PluginError> {
    let json: Option<String> =
        sqlx::query_scalar("SELECT record_json FROM lash_processes WHERE process_id = $1")
            .bind(process_id)
            .fetch_optional(pool)
            .await
            .map_err(plugin_sqlx_error)?;
    json.map(|json| serde_json::from_str(&json).map_err(process_decode_error))
        .transpose()
}

async fn save_process_tx(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    record: &ProcessRecord,
) -> Result<(), PluginError> {
    sqlx::query(
        "UPDATE lash_processes
         SET updated_at_ms = $2, status = $3, record_json = $4
         WHERE process_id = $1",
    )
    .bind(&record.id)
    .bind(record.updated_at_ms as i64)
    .bind(process_status_label(record))
    .bind(serde_json::to_string(record).map_err(process_decode_error)?)
    .execute(&mut **tx)
    .await
    .map_err(plugin_sqlx_error)?;
    Ok(())
}

async fn load_event_by_key_tx(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    process_id: &str,
    replay_key: &str,
) -> Result<Option<(String, ProcessEvent)>, PluginError> {
    let row = sqlx::query(
        "SELECT payload_hash, event_json
         FROM lash_process_events
         WHERE process_id = $1 AND idempotency_key = $2",
    )
    .bind(process_id)
    .bind(replay_key)
    .fetch_optional(&mut **tx)
    .await
    .map_err(plugin_sqlx_error)?;
    row.map(|row| {
        let hash: String = row.get(0);
        let json: String = row.get(1);
        serde_json::from_str(&json)
            .map(|event| (hash, event))
            .map_err(process_decode_error)
    })
    .transpose()
}

async fn load_process_lease_tx(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    process_id: &str,
) -> Result<Option<ProcessLease>, PluginError> {
    let row = sqlx::query(
        "SELECT lease_owner_id, lease_token, lease_fencing_token,
                lease_claimed_at_ms, lease_expires_at_ms
         FROM lash_process_leases
         WHERE process_id = $1",
    )
    .bind(process_id)
    .fetch_optional(&mut **tx)
    .await
    .map_err(plugin_sqlx_error)?;
    let Some(row) = row else {
        return Ok(None);
    };
    let owner_id: Option<String> = row.get(0);
    let lease_token: Option<String> = row.get(1);
    let (Some(owner_id), Some(lease_token)) = (owner_id, lease_token) else {
        return Ok(None);
    };
    Ok(Some(ProcessLease {
        schema_version: PROCESS_LEASE_SCHEMA_VERSION,
        process_id: process_id.to_string(),
        owner_id,
        lease_token,
        fencing_token: row.get::<i64, _>(2) as u64,
        claimed_at_epoch_ms: row.get::<i64, _>(3) as u64,
        expires_at_epoch_ms: row.get::<i64, _>(4) as u64,
    }))
}

fn process_lease_conflict(process_id: &str, current: &ProcessLease) -> PluginError {
    PluginError::Session(format!(
        "process `{process_id}` is already leased by `{}` until {}",
        current.owner_id, current.expires_at_epoch_ms
    ))
}

fn process_lease_expired(process_id: &str) -> PluginError {
    PluginError::Session(format!(
        "process lease for `{process_id}` is missing or expired"
    ))
}

fn guard_lease(current: Option<&ProcessLease>, lease_token: &str, now: u64) -> bool {
    current
        .map(|current| current.lease_token == lease_token && current.expires_at_epoch_ms > now)
        .unwrap_or(false)
}

async fn list_grants_for_scope(
    pool: &PgPool,
    owner_scope: &ProcessScope,
    live_only: bool,
) -> Result<Vec<ProcessHandleGrantEntry>, PluginError> {
    let status_clause = if live_only {
        "AND p.status = 'running'"
    } else {
        ""
    };
    let sql = format!(
        "SELECT g.process_id, g.descriptor_json, p.record_json
         FROM lash_process_handle_grants g
         JOIN lash_processes p ON p.process_id = g.process_id
         WHERE g.scope_id = $1 {status_clause}
         ORDER BY g.process_id ASC"
    );
    let rows = sqlx::query(&sql)
        .bind(owner_scope.id().as_str())
        .fetch_all(pool)
        .await
        .map_err(plugin_sqlx_error)?;
    let mut entries = Vec::new();
    for row in rows {
        let process_id: String = row.get(0);
        let descriptor_json: String = row.get(1);
        let record_json: String = row.get(2);
        let descriptor: ProcessHandleDescriptor =
            serde_json::from_str(&descriptor_json).map_err(process_decode_error)?;
        let record: ProcessRecord =
            serde_json::from_str(&record_json).map_err(process_decode_error)?;
        entries.push((
            ProcessHandleGrant {
                session_id: owner_scope.session_id.clone(),
                process_id,
                descriptor,
            },
            record,
        ));
    }
    Ok(entries)
}

#[async_trait::async_trait]
impl ProcessRegistry for PostgresProcessRegistry {
    fn durability_tier(&self) -> DurabilityTier {
        DurabilityTier::Durable
    }

    async fn register_process(
        &self,
        registration: ProcessRegistration,
    ) -> Result<ProcessRecord, PluginError> {
        let (registration, registration_hash) =
            lash_core::runtime::prepare_process_registration(registration)?;
        let mut tx = self.pool.begin().await.map_err(plugin_sqlx_error)?;
        if let Some(existing) = load_process_tx(&mut tx, &registration.id).await? {
            if existing.registration_hash == registration_hash {
                tx.commit().await.map_err(plugin_sqlx_error)?;
                return Ok(existing);
            }
            return Err(PluginError::Session(format!(
                "process `{}` registration hash conflict: existing {}, new {}",
                registration.id, existing.registration_hash, registration_hash
            )));
        }
        let now = current_epoch_ms();
        let record =
            ProcessRecord::from_prepared_registration(registration, registration_hash, now);
        let record_json = serde_json::to_string(&record).map_err(process_decode_error)?;
        sqlx::query(
            "INSERT INTO lash_processes (
                process_id, registration_hash, owner_scope_id, host_profile_id,
                created_at_ms, updated_at_ms, status, record_json
             )
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8)",
        )
        .bind(&record.id)
        .bind(&record.registration_hash)
        .bind(record.owner_scope_id().as_str())
        .bind(record.host_profile_id())
        .bind(record.created_at_ms as i64)
        .bind(record.updated_at_ms as i64)
        .bind(process_status_label(&record))
        .bind(record_json)
        .execute(&mut *tx)
        .await
        .map_err(plugin_sqlx_error)?;
        tx.commit().await.map_err(plugin_sqlx_error)?;
        Ok(record)
    }

    async fn set_external_ref(
        &self,
        process_id: &str,
        external_ref: ProcessExternalRef,
    ) -> Result<ProcessRecord, PluginError> {
        let mut tx = self.pool.begin().await.map_err(plugin_sqlx_error)?;
        let mut record = load_process_tx(&mut tx, process_id)
            .await?
            .ok_or_else(|| PluginError::Session(format!("unknown process `{process_id}`")))?;
        record.external_ref = Some(external_ref);
        record.updated_at_ms = current_epoch_ms();
        save_process_tx(&mut tx, &record).await?;
        tx.commit().await.map_err(plugin_sqlx_error)?;
        Ok(record)
    }

    async fn grant_handle(
        &self,
        owner_scope: &ProcessScope,
        process_id: &str,
        descriptor: ProcessHandleDescriptor,
    ) -> Result<ProcessHandleGrant, PluginError> {
        let mut tx = self.pool.begin().await.map_err(plugin_sqlx_error)?;
        if load_process_tx(&mut tx, process_id).await?.is_none() {
            return Err(PluginError::Session(format!(
                "unknown process `{process_id}`"
            )));
        }
        sqlx::query(
            "INSERT INTO lash_process_handle_grants (session_id, scope_id, process_id, descriptor_json)
             VALUES ($1, $2, $3, $4)
             ON CONFLICT (scope_id, process_id) DO UPDATE SET
                session_id = EXCLUDED.session_id,
                descriptor_json = EXCLUDED.descriptor_json",
        )
        .bind(&owner_scope.session_id)
        .bind(owner_scope.id().as_str())
        .bind(process_id)
        .bind(serde_json::to_string(&descriptor).map_err(process_decode_error)?)
        .execute(&mut *tx)
        .await
        .map_err(plugin_sqlx_error)?;
        tx.commit().await.map_err(plugin_sqlx_error)?;
        Ok(ProcessHandleGrant {
            session_id: owner_scope.session_id.clone(),
            process_id: process_id.to_string(),
            descriptor,
        })
    }

    async fn revoke_handle(
        &self,
        owner_scope: &ProcessScope,
        process_id: &str,
    ) -> Result<(), PluginError> {
        sqlx::query(
            "DELETE FROM lash_process_handle_grants WHERE scope_id = $1 AND process_id = $2",
        )
        .bind(owner_scope.id().as_str())
        .bind(process_id)
        .execute(&self.pool)
        .await
        .map_err(plugin_sqlx_error)?;
        Ok(())
    }

    async fn transfer_handle_grants(
        &self,
        from_scope: &ProcessScope,
        to_scope: &ProcessScope,
        process_ids: &[String],
    ) -> Result<(), PluginError> {
        let mut tx = self.pool.begin().await.map_err(plugin_sqlx_error)?;
        for process_id in process_ids {
            let descriptor_json: Option<String> = sqlx::query_scalar(
                "SELECT descriptor_json FROM lash_process_handle_grants
                 WHERE scope_id = $1 AND process_id = $2",
            )
            .bind(from_scope.id().as_str())
            .bind(process_id)
            .fetch_optional(&mut *tx)
            .await
            .map_err(plugin_sqlx_error)?;
            let Some(descriptor_json) = descriptor_json else {
                return Err(PluginError::Session(format!(
                    "process handle `{process_id}` is not granted to session `{}`",
                    from_scope.session_id
                )));
            };
            sqlx::query(
                "DELETE FROM lash_process_handle_grants WHERE scope_id = $1 AND process_id = $2",
            )
            .bind(from_scope.id().as_str())
            .bind(process_id)
            .execute(&mut *tx)
            .await
            .map_err(plugin_sqlx_error)?;
            sqlx::query(
                "INSERT INTO lash_process_handle_grants (session_id, scope_id, process_id, descriptor_json)
                 VALUES ($1, $2, $3, $4)
                 ON CONFLICT (scope_id, process_id) DO UPDATE SET
                    session_id = EXCLUDED.session_id,
                    descriptor_json = EXCLUDED.descriptor_json",
            )
            .bind(&to_scope.session_id)
            .bind(to_scope.id().as_str())
            .bind(process_id)
            .bind(descriptor_json)
            .execute(&mut *tx)
            .await
            .map_err(plugin_sqlx_error)?;
        }
        tx.commit().await.map_err(plugin_sqlx_error)
    }

    async fn list_handle_grants(
        &self,
        owner_scope: &ProcessScope,
    ) -> Result<Vec<ProcessHandleGrantEntry>, PluginError> {
        list_grants_for_scope(&self.pool, owner_scope, false).await
    }

    async fn list_live_handle_grants(
        &self,
        owner_scope: &ProcessScope,
    ) -> Result<Vec<ProcessHandleGrantEntry>, PluginError> {
        list_grants_for_scope(&self.pool, owner_scope, true).await
    }

    async fn has_handle_grant(
        &self,
        owner_scope: &ProcessScope,
        process_id: &str,
    ) -> Result<bool, PluginError> {
        let exists: Option<i64> = sqlx::query_scalar(
            "SELECT 1::BIGINT FROM lash_process_handle_grants
             WHERE scope_id = $1 AND process_id = $2
             LIMIT 1",
        )
        .bind(owner_scope.id().as_str())
        .bind(process_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(plugin_sqlx_error)?;
        Ok(exists.is_some())
    }

    async fn handle_grants_for_process(
        &self,
        process_id: &str,
    ) -> Result<Vec<ProcessHandleGrant>, PluginError> {
        if load_process(&self.pool, process_id).await?.is_none() {
            return Err(PluginError::Session(format!(
                "unknown process `{process_id}`"
            )));
        }
        let rows = sqlx::query(
            "SELECT session_id, descriptor_json
             FROM lash_process_handle_grants
             WHERE process_id = $1
             ORDER BY session_id ASC, scope_id ASC",
        )
        .bind(process_id)
        .fetch_all(&self.pool)
        .await
        .map_err(plugin_sqlx_error)?;
        let mut grants = Vec::new();
        for row in rows {
            let descriptor_json: String = row.get(1);
            grants.push(ProcessHandleGrant {
                session_id: row.get(0),
                process_id: process_id.to_string(),
                descriptor: serde_json::from_str(&descriptor_json).map_err(process_decode_error)?,
            });
        }
        Ok(grants)
    }

    async fn delete_session_process_state(
        &self,
        session_id: &str,
    ) -> Result<lash_core::ProcessSessionDeleteReport, PluginError> {
        let mut tx = self.pool.begin().await.map_err(plugin_sqlx_error)?;
        let rows = sqlx::query(
            "SELECT g.process_id, p.record_json
             FROM lash_process_handle_grants g
             JOIN lash_processes p ON p.process_id = g.process_id
             WHERE g.session_id = $1
             ORDER BY g.process_id ASC",
        )
        .bind(session_id)
        .fetch_all(&mut *tx)
        .await
        .map_err(plugin_sqlx_error)?;
        let mut removed = Vec::new();
        for row in rows {
            let process_id: String = row.get(0);
            let record_json: String = row.get(1);
            let record: ProcessRecord =
                serde_json::from_str(&record_json).map_err(process_decode_error)?;
            removed.push((process_id, record));
        }
        let revoked = sqlx::query("DELETE FROM lash_process_handle_grants WHERE session_id = $1")
            .bind(session_id)
            .execute(&mut *tx)
            .await
            .map_err(plugin_sqlx_error)?
            .rows_affected() as usize;
        let mut cancel_process_ids = Vec::new();
        let mut preserved_process_ids = Vec::new();
        for (process_id, record) in removed {
            if record.is_terminal() {
                continue;
            }
            let remaining: i64 = sqlx::query_scalar(
                "SELECT COUNT(*) FROM lash_process_handle_grants WHERE process_id = $1",
            )
            .bind(&process_id)
            .fetch_one(&mut *tx)
            .await
            .map_err(plugin_sqlx_error)?;
            if remaining == 0 {
                cancel_process_ids.push(process_id);
            } else {
                preserved_process_ids.push(process_id);
            }
        }
        tx.commit().await.map_err(plugin_sqlx_error)?;
        cancel_process_ids.sort();
        cancel_process_ids.dedup();
        preserved_process_ids.sort();
        preserved_process_ids.dedup();
        Ok(lash_core::ProcessSessionDeleteReport {
            session_id: session_id.to_string(),
            revoked_handle_count: revoked,
            deleted_wake_count: 0,
            cancel_process_ids,
            preserved_process_ids,
        })
    }

    async fn append_event(
        &self,
        process_id: &str,
        request: ProcessEventAppendRequest,
    ) -> Result<ProcessEventAppendResult, PluginError> {
        let mut tx = self.pool.begin().await.map_err(plugin_sqlx_error)?;
        let mut record = load_process_tx(&mut tx, process_id)
            .await?
            .ok_or_else(|| PluginError::Session(format!("unknown process `{process_id}`")))?;
        let replay_lookup =
            if let Some(replay_key) = request.replay.as_ref().map(|r| r.key.as_str()) {
                load_event_by_key_tx(&mut tx, process_id, replay_key).await?
            } else {
                None
            };
        let sequence: i64 = sqlx::query_scalar(
            "SELECT COALESCE(MAX(sequence), 0) + 1 FROM lash_process_events WHERE process_id = $1",
        )
        .bind(process_id)
        .fetch_one(&mut *tx)
        .await
        .map_err(plugin_sqlx_error)?;
        let occurred_at_ms = current_epoch_ms();
        let prepared = lash_core::runtime::prepare_process_event_append(
            &record,
            request,
            sequence as u64,
            replay_lookup,
            occurred_at_ms,
        )?;
        if prepared.replayed {
            let repaired = if let Some(status) = prepared.status_update.clone() {
                record.status = status;
                record.updated_at_ms = prepared.occurred_at_ms;
                save_process_tx(&mut tx, &record).await?;
                true
            } else {
                false
            };
            tx.commit().await.map_err(plugin_sqlx_error)?;
            if repaired {
                self.notify.notify_waiters();
            }
            return Ok(ProcessEventAppendResult {
                event: prepared.event,
                wake_delivery: prepared.wake_delivery,
            });
        }
        let event = prepared.event;
        sqlx::query(
            "INSERT INTO lash_process_events (
                process_id, sequence, event_type, payload_hash, idempotency_key,
                occurred_at_ms, event_json
             )
             VALUES ($1, $2, $3, $4, $5, $6, $7)",
        )
        .bind(process_id)
        .bind(sequence)
        .bind(event.event_type.as_str())
        .bind(&prepared.payload_hash)
        .bind(event.invocation.replay_key())
        .bind(prepared.occurred_at_ms as i64)
        .bind(serde_json::to_string(&event).map_err(process_decode_error)?)
        .execute(&mut *tx)
        .await
        .map_err(plugin_sqlx_error)?;
        if let Some(status) = prepared.status_update.clone() {
            record.status = status;
        }
        record.updated_at_ms = prepared.occurred_at_ms;
        save_process_tx(&mut tx, &record).await?;
        tx.commit().await.map_err(plugin_sqlx_error)?;
        self.notify.notify_waiters();
        Ok(ProcessEventAppendResult {
            event,
            wake_delivery: prepared.wake_delivery,
        })
    }

    async fn events_after(
        &self,
        process_id: &str,
        after_sequence: u64,
    ) -> Result<Vec<ProcessEvent>, PluginError> {
        if load_process(&self.pool, process_id).await?.is_none() {
            return Err(PluginError::Session(format!(
                "unknown process `{process_id}`"
            )));
        }
        let rows = sqlx::query(
            "SELECT event_json FROM lash_process_events
             WHERE process_id = $1 AND sequence > $2
             ORDER BY sequence ASC",
        )
        .bind(process_id)
        .bind(after_sequence as i64)
        .fetch_all(&self.pool)
        .await
        .map_err(plugin_sqlx_error)?;
        let mut events = Vec::new();
        for row in rows {
            let json: String = row.get(0);
            events.push(serde_json::from_str(&json).map_err(process_decode_error)?);
        }
        Ok(events)
    }

    async fn wake_events_after(
        &self,
        process_id: &str,
        after_sequence: u64,
    ) -> Result<Vec<ProcessEvent>, PluginError> {
        let rows = sqlx::query("SELECT sequence FROM lash_process_wake_acks WHERE process_id = $1")
            .bind(process_id)
            .fetch_all(&self.pool)
            .await
            .map_err(plugin_sqlx_error)?;
        let acked = rows
            .into_iter()
            .map(|row| row.get::<i64, _>(0) as u64)
            .collect::<HashSet<_>>();
        Ok(self
            .events_after(process_id, after_sequence)
            .await?
            .into_iter()
            .filter(|event| event.semantics.wake.is_some() && !acked.contains(&event.sequence))
            .collect())
    }

    async fn wait_event_after(
        &self,
        process_id: &str,
        event_type: &str,
        after_sequence: u64,
    ) -> Result<ProcessEvent, PluginError> {
        loop {
            if let Some(event) = self
                .events_after(process_id, after_sequence)
                .await?
                .into_iter()
                .find(|event| event.event_type == event_type)
            {
                return Ok(event);
            }
            tokio::select! {
                _ = self.notify.notified() => {}
                _ = tokio::time::sleep(Duration::from_millis(50)) => {}
            }
        }
    }

    async fn await_process(&self, process_id: &str) -> Result<ProcessAwaitOutput, PluginError> {
        loop {
            let record = load_process(&self.pool, process_id)
                .await?
                .ok_or_else(|| PluginError::Session(format!("unknown process `{process_id}`")))?;
            if let Some(await_output) = record.status.await_output() {
                return Ok(await_output.clone());
            }
            tokio::select! {
                _ = self.notify.notified() => {}
                _ = tokio::time::sleep(Duration::from_millis(50)) => {}
            }
        }
    }

    async fn complete_process(
        &self,
        process_id: &str,
        await_output: ProcessAwaitOutput,
    ) -> Result<ProcessRecord, PluginError> {
        let event_type = match await_output.terminal_state() {
            lash_core::ProcessTerminalState::Completed => "process.completed",
            lash_core::ProcessTerminalState::Failed => "process.failed",
            lash_core::ProcessTerminalState::Cancelled => "process.cancelled",
        };
        self.append_event(
            process_id,
            ProcessEventAppendRequest::new(
                event_type,
                serde_json::json!({ "await_output": await_output }),
            )
            .with_replay_key(format!("process:{process_id}:terminal:{event_type}")),
        )
        .await?;
        load_process(&self.pool, process_id).await?.ok_or_else(|| {
            PluginError::Session(format!(
                "unknown process `{process_id}` after terminal event"
            ))
        })
    }

    async fn get_process(&self, process_id: &str) -> Option<ProcessRecord> {
        load_process(&self.pool, process_id).await.ok().flatten()
    }

    async fn ack_wake(&self, process_id: &str, sequence: u64) -> Result<(), PluginError> {
        if load_process(&self.pool, process_id).await?.is_none() {
            return Err(PluginError::Session(format!(
                "unknown process `{process_id}`"
            )));
        }
        sqlx::query(
            "INSERT INTO lash_process_wake_acks (process_id, sequence)
             VALUES ($1, $2)
             ON CONFLICT DO NOTHING",
        )
        .bind(process_id)
        .bind(sequence as i64)
        .execute(&self.pool)
        .await
        .map_err(plugin_sqlx_error)?;
        Ok(())
    }

    async fn list_non_terminal(&self) -> Result<Vec<ProcessRecord>, PluginError> {
        let rows = sqlx::query(
            "SELECT record_json FROM lash_processes
             WHERE status = 'running'
             ORDER BY process_id ASC",
        )
        .fetch_all(&self.pool)
        .await
        .map_err(plugin_sqlx_error)?;
        let mut records = Vec::new();
        for row in rows {
            let json: String = row.get(0);
            records.push(serde_json::from_str(&json).map_err(process_decode_error)?);
        }
        Ok(records)
    }

    async fn claim_process_lease(
        &self,
        process_id: &str,
        owner_id: &str,
        lease_ttl_ms: u64,
    ) -> Result<ProcessLease, PluginError> {
        let mut tx = self.pool.begin().await.map_err(plugin_sqlx_error)?;
        if load_process_tx(&mut tx, process_id).await?.is_none() {
            return Err(PluginError::Session(format!(
                "unknown process `{process_id}`"
            )));
        }
        let now = current_epoch_ms();
        let current = load_process_lease_tx(&mut tx, process_id).await?;
        if let Some(current) = current.as_ref()
            && current.expires_at_epoch_ms > now
            && current.owner_id != owner_id
        {
            return Err(process_lease_conflict(process_id, current));
        }
        let existing_fence: Option<i64> = sqlx::query_scalar(
            "SELECT lease_fencing_token FROM lash_process_leases WHERE process_id = $1 FOR UPDATE",
        )
        .bind(process_id)
        .fetch_optional(&mut *tx)
        .await
        .map_err(plugin_sqlx_error)?;
        let fencing_token = existing_fence.unwrap_or(0) as u64 + 1;
        let lease = ProcessLease {
            schema_version: PROCESS_LEASE_SCHEMA_VERSION,
            process_id: process_id.to_string(),
            owner_id: owner_id.to_string(),
            lease_token: format!(
                "{:x}",
                Sha256::digest(format!("{process_id}:{owner_id}:{now}:{fencing_token}").as_bytes())
            ),
            fencing_token,
            claimed_at_epoch_ms: now,
            expires_at_epoch_ms: now.saturating_add(lease_ttl_ms),
        };
        sqlx::query(
            "INSERT INTO lash_process_leases (
                process_id, lease_owner_id, lease_token, lease_fencing_token,
                lease_claimed_at_ms, lease_expires_at_ms
             )
             VALUES ($1, $2, $3, $4, $5, $6)
             ON CONFLICT (process_id) DO UPDATE SET
                lease_owner_id = EXCLUDED.lease_owner_id,
                lease_token = EXCLUDED.lease_token,
                lease_fencing_token = EXCLUDED.lease_fencing_token,
                lease_claimed_at_ms = EXCLUDED.lease_claimed_at_ms,
                lease_expires_at_ms = EXCLUDED.lease_expires_at_ms",
        )
        .bind(&lease.process_id)
        .bind(&lease.owner_id)
        .bind(&lease.lease_token)
        .bind(lease.fencing_token as i64)
        .bind(lease.claimed_at_epoch_ms as i64)
        .bind(lease.expires_at_epoch_ms as i64)
        .execute(&mut *tx)
        .await
        .map_err(plugin_sqlx_error)?;
        tx.commit().await.map_err(plugin_sqlx_error)?;
        Ok(lease)
    }

    async fn renew_process_lease(
        &self,
        lease: &ProcessLease,
        lease_ttl_ms: u64,
    ) -> Result<ProcessLease, PluginError> {
        let mut tx = self.pool.begin().await.map_err(plugin_sqlx_error)?;
        let now = current_epoch_ms();
        let current = load_process_lease_tx(&mut tx, &lease.process_id).await?;
        if !guard_lease(current.as_ref(), &lease.lease_token, now) {
            return Err(process_lease_expired(&lease.process_id));
        }
        let renewed = ProcessLease {
            expires_at_epoch_ms: now.saturating_add(lease_ttl_ms),
            ..lease.clone()
        };
        sqlx::query(
            "UPDATE lash_process_leases
             SET lease_expires_at_ms = $2
             WHERE process_id = $1 AND lease_token = $3",
        )
        .bind(&renewed.process_id)
        .bind(renewed.expires_at_epoch_ms as i64)
        .bind(&renewed.lease_token)
        .execute(&mut *tx)
        .await
        .map_err(plugin_sqlx_error)?;
        tx.commit().await.map_err(plugin_sqlx_error)?;
        Ok(renewed)
    }

    async fn complete_process_lease(
        &self,
        completion: &ProcessLeaseCompletion,
    ) -> Result<(), PluginError> {
        sqlx::query(
            "UPDATE lash_process_leases
             SET lease_owner_id = NULL,
                 lease_token = NULL,
                 lease_claimed_at_ms = 0,
                 lease_expires_at_ms = 0
             WHERE process_id = $1 AND lease_token = $2",
        )
        .bind(&completion.process_id)
        .bind(&completion.lease_token)
        .execute(&self.pool)
        .await
        .map_err(plugin_sqlx_error)?;
        Ok(())
    }
}

#[async_trait::async_trait]
impl HostEventStore for PostgresHostEventStore {
    fn durability_tier(&self) -> DurabilityTier {
        DurabilityTier::Durable
    }

    async fn register_subscription(
        &self,
        draft: TriggerSubscriptionDraft,
    ) -> Result<TriggerSubscriptionRecord, PluginError> {
        let mut tx = self.pool.begin().await.map_err(plugin_sqlx_error)?;
        let seq: i64 = sqlx::query_scalar("SELECT nextval('lash_host_event_subscription_seq')")
            .fetch_one(&mut *tx)
            .await
            .map_err(plugin_sqlx_error)?;
        let handle = format!("trigger:{seq}");
        let subscription_id = format!("subscription:{seq}");
        let now = current_epoch_ms();
        let record = TriggerSubscriptionRecord {
            subscription_id: subscription_id.clone(),
            session_id: draft.session_id,
            handle,
            name: draft.name,
            source_type: draft.source_type,
            source_key: draft.source_key,
            source: draft.source,
            event_ty: draft.event_ty,
            module_ref: draft.module_ref,
            required_surface_ref: draft.required_surface_ref,
            process_ref: draft.process_ref,
            process_name: draft.process_name,
            input_template: draft.input_template,
            enabled: true,
            created_at_ms: now,
            updated_at_ms: now,
        };
        sqlx::query(
            "INSERT INTO lash_host_event_trigger_subscriptions (
                subscription_id, session_id, handle, source_type, source_key,
                enabled, created_at_ms, updated_at_ms, record_json
             )
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)",
        )
        .bind(&record.subscription_id)
        .bind(&record.session_id)
        .bind(&record.handle)
        .bind(&record.source_type)
        .bind(&record.source_key)
        .bind(record.enabled)
        .bind(record.created_at_ms as i64)
        .bind(record.updated_at_ms as i64)
        .bind(serde_json::to_string(&record).map_err(process_decode_error)?)
        .execute(&mut *tx)
        .await
        .map_err(plugin_sqlx_error)?;
        tx.commit().await.map_err(plugin_sqlx_error)?;
        Ok(record)
    }

    async fn list_subscriptions(
        &self,
        filter: TriggerSubscriptionFilter,
    ) -> Result<Vec<TriggerSubscriptionRecord>, PluginError> {
        let rows = sqlx::query(
            "SELECT record_json FROM lash_host_event_trigger_subscriptions
             ORDER BY session_id ASC, handle ASC",
        )
        .fetch_all(&self.pool)
        .await
        .map_err(plugin_sqlx_error)?;
        let mut records = Vec::new();
        for row in rows {
            let json: String = row.get(0);
            let record: TriggerSubscriptionRecord =
                serde_json::from_str(&json).map_err(process_decode_error)?;
            if filter.matches(&record) {
                records.push(record);
            }
        }
        Ok(records)
    }

    async fn cancel_subscription(
        &self,
        session_id: &str,
        handle: &str,
    ) -> Result<bool, PluginError> {
        let mut tx = self.pool.begin().await.map_err(plugin_sqlx_error)?;
        let json: Option<String> = sqlx::query_scalar(
            "SELECT record_json FROM lash_host_event_trigger_subscriptions
             WHERE session_id = $1 AND handle = $2
             FOR UPDATE",
        )
        .bind(session_id)
        .bind(handle)
        .fetch_optional(&mut *tx)
        .await
        .map_err(plugin_sqlx_error)?;
        let Some(json) = json else {
            tx.commit().await.map_err(plugin_sqlx_error)?;
            return Ok(false);
        };
        let mut record: TriggerSubscriptionRecord =
            serde_json::from_str(&json).map_err(process_decode_error)?;
        let changed = record.enabled;
        record.enabled = false;
        record.updated_at_ms = current_epoch_ms();
        sqlx::query(
            "UPDATE lash_host_event_trigger_subscriptions
             SET enabled = $3, updated_at_ms = $4, record_json = $5
             WHERE session_id = $1 AND handle = $2",
        )
        .bind(session_id)
        .bind(handle)
        .bind(record.enabled)
        .bind(record.updated_at_ms as i64)
        .bind(serde_json::to_string(&record).map_err(process_decode_error)?)
        .execute(&mut *tx)
        .await
        .map_err(plugin_sqlx_error)?;
        tx.commit().await.map_err(plugin_sqlx_error)?;
        Ok(changed)
    }

    async fn record_occurrence(
        &self,
        request: HostEventOccurrenceRequest,
    ) -> Result<HostEventOccurrenceRecord, PluginError> {
        lash_core::validate_host_event_occurrence_request(&request)?;
        let request_hash = lash_core::host_event_occurrence_request_hash(&request)?;
        let occurrence_id = lash_core::deterministic_occurrence_id(&request)?;
        let mut tx = self.pool.begin().await.map_err(plugin_sqlx_error)?;
        let existing = sqlx::query(
            "SELECT request_hash, record_json
             FROM lash_host_event_occurrences
             WHERE idempotency_key = $1
             FOR UPDATE",
        )
        .bind(&request.idempotency_key)
        .fetch_optional(&mut *tx)
        .await
        .map_err(plugin_sqlx_error)?;
        if let Some(row) = existing {
            let existing_hash: String = row.get(0);
            let existing_json: String = row.get(1);
            if existing_hash != request_hash {
                return Err(PluginError::Session(format!(
                    "host event occurrence idempotency conflict for `{}`",
                    request.idempotency_key
                )));
            }
            let record = serde_json::from_str(&existing_json).map_err(process_decode_error)?;
            tx.commit().await.map_err(plugin_sqlx_error)?;
            return Ok(record);
        }
        let record = HostEventOccurrenceRecord {
            occurrence_id: occurrence_id.clone(),
            source_type: request.source_type,
            source_key: request.source_key,
            payload: request.payload,
            idempotency_key: request.idempotency_key.clone(),
            source: request.source,
            occurred_at_ms: current_epoch_ms(),
        };
        sqlx::query(
            "INSERT INTO lash_host_event_occurrences (
                occurrence_id, idempotency_key, request_hash, source_type, source_key,
                occurred_at_ms, record_json
             )
             VALUES ($1, $2, $3, $4, $5, $6, $7)",
        )
        .bind(&record.occurrence_id)
        .bind(&record.idempotency_key)
        .bind(&request_hash)
        .bind(&record.source_type)
        .bind(&record.source_key)
        .bind(record.occurred_at_ms as i64)
        .bind(serde_json::to_string(&record).map_err(process_decode_error)?)
        .execute(&mut *tx)
        .await
        .map_err(plugin_sqlx_error)?;
        tx.commit().await.map_err(plugin_sqlx_error)?;
        Ok(record)
    }

    async fn reserve_matching_deliveries(
        &self,
        occurrence_id: &str,
    ) -> Result<Vec<TriggerDeliveryReservation>, PluginError> {
        let mut tx = self.pool.begin().await.map_err(plugin_sqlx_error)?;
        let occurrence_json: Option<String> = sqlx::query_scalar(
            "SELECT record_json FROM lash_host_event_occurrences WHERE occurrence_id = $1",
        )
        .bind(occurrence_id)
        .fetch_optional(&mut *tx)
        .await
        .map_err(plugin_sqlx_error)?;
        let Some(occurrence_json) = occurrence_json else {
            return Err(PluginError::Session(format!(
                "unknown host event occurrence `{occurrence_id}`"
            )));
        };
        let occurrence: HostEventOccurrenceRecord =
            serde_json::from_str(&occurrence_json).map_err(process_decode_error)?;
        let rows = sqlx::query(
            "SELECT record_json FROM lash_host_event_trigger_subscriptions
             WHERE enabled = TRUE AND source_type = $1 AND source_key = $2
             ORDER BY session_id ASC, handle ASC",
        )
        .bind(&occurrence.source_type)
        .bind(&occurrence.source_key)
        .fetch_all(&mut *tx)
        .await
        .map_err(plugin_sqlx_error)?;
        let mut deliveries = Vec::new();
        for row in rows {
            let json: String = row.get(0);
            let subscription: TriggerSubscriptionRecord =
                serde_json::from_str(&json).map_err(process_decode_error)?;
            let process_id = lash_core::deterministic_delivery_process_id(
                &occurrence.occurrence_id,
                &subscription.subscription_id,
            )?;
            let inserted = sqlx::query(
                "INSERT INTO lash_host_event_deliveries (
                    occurrence_id, subscription_id, process_id, created_at_ms
                 )
                 VALUES ($1, $2, $3, $4)
                 ON CONFLICT DO NOTHING",
            )
            .bind(&occurrence.occurrence_id)
            .bind(&subscription.subscription_id)
            .bind(&process_id)
            .bind(current_epoch_ms() as i64)
            .execute(&mut *tx)
            .await
            .map_err(plugin_sqlx_error)?
            .rows_affected();
            if inserted == 0 {
                continue;
            }
            deliveries.push(TriggerDeliveryReservation {
                occurrence: occurrence.clone(),
                subscription,
                process_id,
            });
        }
        tx.commit().await.map_err(plugin_sqlx_error)?;
        Ok(deliveries)
    }
}

#[async_trait::async_trait]
impl lashlang::LashlangArtifactStore for PostgresLashlangArtifactStore {
    fn durability_tier(&self) -> DurabilityTier {
        DurabilityTier::Durable
    }

    async fn put_module_artifact(
        &self,
        artifact: &lashlang::ModuleArtifact,
    ) -> Result<(), lashlang::ArtifactStoreError> {
        let bytes = artifact
            .to_store_bytes()
            .map_err(lashlang::ArtifactStoreError::from)?;
        sqlx::query(
            "INSERT INTO lash_lashlang_artifacts (module_ref, artifact_bytes)
             VALUES ($1, $2)
             ON CONFLICT (module_ref) DO UPDATE SET artifact_bytes = EXCLUDED.artifact_bytes",
        )
        .bind(artifact.module_ref.as_str())
        .bind(bytes)
        .execute(&self.pool)
        .await
        .map_err(|err| lashlang::ArtifactStoreError::Backend(err.to_string()))?;
        Ok(())
    }

    async fn get_module_artifact(
        &self,
        module_ref: &lashlang::ModuleRef,
    ) -> Result<Option<Arc<lashlang::ModuleArtifact>>, lashlang::ArtifactStoreError> {
        let bytes: Option<Vec<u8>> = sqlx::query_scalar(
            "SELECT artifact_bytes FROM lash_lashlang_artifacts WHERE module_ref = $1",
        )
        .bind(module_ref.as_str())
        .fetch_optional(&self.pool)
        .await
        .map_err(|err| lashlang::ArtifactStoreError::Backend(err.to_string()))?;
        bytes
            .map(|bytes| {
                lashlang::ModuleArtifact::from_store_bytes(&bytes)
                    .map(Arc::new)
                    .map_err(lashlang::ArtifactStoreError::from)
            })
            .transpose()
    }
}
