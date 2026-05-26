//! # lash-sqlite-store
//!
//! The high-performance local **durable** persistence backend for the lash
//! agent runtime. One SQLite database per session, opened with WAL journal
//! mode and a 15-second busy timeout, satisfying the full
//! [`RuntimePersistence`] + [`AttachmentManifest`] contract from `lash-core`.
//!
//! ## Why this is "the durable backend" not just "an option"
//!
//! Lash's runtime layer treats persistence as a first-class boundary, not a
//! debug-only convenience. Every primitive that lets the runtime survive a
//! crash — head-revision CAS, runtime turn leases with fencing tokens,
//! effect-journal idempotency, attachment write-ahead manifests, blob
//! content-addressing with optional compression — is implemented in this
//! crate against SQLite for one reason: SQLite is the simplest backend that
//! gives us *atomic multi-statement transactions on a single file* with
//! durability guarantees we can reason about. Everything else in the
//! `RuntimePersistence` contract (lease/fencing, optimistic concurrency,
//! per-record `schema_version` stamps, the attachment manifest) is shaped
//! for distributed durable backends — Restate, Postgres, hosted KV — and
//! the SQLite impl is the reference implementation that proves those
//! shapes work.
//!
//! In other words: SQLite is the local case. The lease/fencing machinery,
//! the per-record version stamps, the `RuntimeCommit::committed_attachment_ids`
//! plumbing — none of it is overkill for "single-process sqlite," it's the
//! contract that *also* has to hold when a second runtime carries the same
//! session over to Restate replay or to a different process. Treat any
//! simplification as "would this still work over Restate?" before
//! shipping.
//!
//! ## Schema cutover, not migrations
//!
//! There is exactly one supported schema (see [`SCHEMA`] below). Older
//! databases must be deleted before opening — we do not carry migration
//! code. The per-record `schema_version` stamps on
//! [`RuntimeTurnCheckpoint`], [`RuntimeTurnLease`], and
//! [`RuntimeEffectJournalRecord`] are the upgrade contract for the
//! *records that cross durable boundaries* (Restate replay, cross-version
//! workers). The SQLite schema itself is a snapshot.
//!
//! [`RuntimePersistence`]: lash_core::RuntimePersistence
//! [`AttachmentManifest`]: lash_core::AttachmentManifest
//! [`RuntimeTurnCheckpoint`]: lash_core::RuntimeTurnCheckpoint
//! [`RuntimeTurnLease`]: lash_core::RuntimeTurnLease
//! [`RuntimeEffectJournalRecord`]: lash_core::RuntimeEffectJournalRecord

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use flate2::Compression;
use flate2::read::ZlibDecoder;
use flate2::write::ZlibEncoder;
use lash_core::{
    AttachmentId, AttachmentIntent, AttachmentManifest, AttachmentManifestEntry, BlobRef, GcReport,
    GraphCommitDelta, HydratedSessionCheckpoint, PersistedSessionRead, ProcessAwaitOutput,
    ProcessEvent, ProcessEventAppendRequest, ProcessExternalRef, ProcessHandleDescriptor,
    ProcessHandleGrant, ProcessHandleGrantEntry, ProcessRecord, ProcessRegistration,
    ProcessRegistry, ProcessWakeDelivery, RUNTIME_EFFECT_JOURNAL_SCHEMA_VERSION,
    RUNTIME_TURN_CHECKPOINT_SCHEMA_VERSION, RUNTIME_TURN_LEASE_SCHEMA_VERSION, RuntimeCommit,
    RuntimeCommitResult, RuntimeEffectJournalRecord, RuntimePersistence, RuntimeTurnCheckpoint,
    RuntimeTurnLease, SessionCheckpoint, SessionHead, SessionHeadMeta, SessionMeta,
    SessionPickerInfo, SessionReadScope, SessionStoreCreateRequest, SessionStoreFactory,
    StoreError, VacuumReport, ensure_supported_schema_version, materialize_process_event_semantics,
    prepare_process_registration, process_event_payload_hash, process_wake_delivery,
    require_event_idempotency, system_time_from_epoch_ms,
};
use rusqlite::{Connection, OpenFlags, OptionalExtension, params};
use sha2::{Digest, Sha256};

/// SQLite-backed store for checkpoint blobs and the canonical session head.
pub struct Store {
    conn: Mutex<Connection>,
    options: StoreOptions,
    commit_count: AtomicU64,
}

/// SQLite-backed process registry for one configured runtime deployment.
///
/// This is intentionally separate from [`Store`]: session databases persist
/// one conversation, while this registry persists background process state and
/// handle visibility across all sessions in the same host profile.
pub struct SqliteProcessRegistry {
    conn: Mutex<Connection>,
    notify: tokio::sync::Notify,
}

fn sqlite_error(err: rusqlite::Error) -> StoreError {
    StoreError::Backend(err.to_string())
}

fn process_sqlite_error(err: rusqlite::Error) -> lash_core::PluginError {
    lash_core::PluginError::Session(err.to_string())
}

fn process_decode_error(err: serde_json::Error) -> lash_core::PluginError {
    lash_core::PluginError::Session(format!("failed to decode process registry row: {err}"))
}

fn process_encode_json<T: serde::Serialize>(value: &T) -> Result<String, lash_core::PluginError> {
    serde_json::to_string(value).map_err(|err| {
        lash_core::PluginError::Session(format!("failed to encode process row: {err}"))
    })
}

fn ensure_process_schema(conn: &Connection) -> rusqlite::Result<()> {
    let user_version: i32 = conn.query_row("PRAGMA user_version", [], |row| row.get(0))?;
    if user_version == PROCESS_SCHEMA_VERSION {
        conn.execute_batch(PROCESS_SCHEMA)?;
        return Ok(());
    }
    if user_version == 0 && !has_user_schema_objects(conn)? {
        conn.execute_batch(PROCESS_SCHEMA)?;
        conn.pragma_update(None, "user_version", PROCESS_SCHEMA_VERSION)?;
        return Ok(());
    }
    Err(rusqlite::Error::InvalidParameterName(
        "Unsupported lash process registry schema. Delete the runtime process registry database and start fresh."
            .to_string(),
    ))
}

/// Canonical SQLite schema for a lash session database.
///
/// This is the *only* schema the store supports. Older session databases —
/// including any rolled forward through prior migration chains — must be
/// deleted before opening with this binary; [`ensure_schema`] rejects any
/// `PRAGMA user_version` that does not match [`SCHEMA_VERSION`] exactly. We
/// run with no on-the-fly migrations on purpose: lash's durable contract
/// lives one level up in the per-record `schema_version` stamps, not in
/// SQL DDL juggling.
const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS blobs (
    hash    TEXT PRIMARY KEY,
    content BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS session_head (
    singleton      INTEGER PRIMARY KEY CHECK (singleton = 1),
    session_id     TEXT NOT NULL DEFAULT 'root',
    head_json      TEXT NOT NULL DEFAULT '{}',
    head_revision  INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS graph_nodes (
    seq        INTEGER PRIMARY KEY AUTOINCREMENT,
    node_id    TEXT NOT NULL UNIQUE,
    node_json  TEXT NOT NULL,
    tombstoned INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS usage_deltas (
    seq                  INTEGER PRIMARY KEY AUTOINCREMENT,
    source               TEXT NOT NULL,
    model                TEXT NOT NULL,
    input_tokens         INTEGER NOT NULL,
    output_tokens        INTEGER NOT NULL,
    cached_input_tokens  INTEGER NOT NULL,
    reasoning_tokens     INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS session_meta (
    singleton     INTEGER PRIMARY KEY CHECK (singleton = 1),
    session_id    TEXT NOT NULL,
    session_name  TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    model         TEXT NOT NULL,
    cwd           TEXT,
    relation_json TEXT
);

CREATE TABLE IF NOT EXISTS runtime_turn_checkpoints (
    session_id          TEXT NOT NULL,
    turn_id             TEXT NOT NULL,
    checkpoint_json     TEXT,
    checkpoint_hash     TEXT,
    lease_owner_id      TEXT,
    lease_token         TEXT,
    lease_fencing_token INTEGER NOT NULL DEFAULT 0,
    lease_claimed_at_ms INTEGER NOT NULL DEFAULT 0,
    lease_expires_at_ms INTEGER NOT NULL DEFAULT 0,
    updated_at_ms       INTEGER NOT NULL,
    PRIMARY KEY (session_id, turn_id)
);

CREATE TABLE IF NOT EXISTS runtime_effect_journal (
    session_id       TEXT NOT NULL,
    turn_id          TEXT NOT NULL,
    idempotency_key  TEXT NOT NULL,
    envelope_hash    TEXT NOT NULL,
    effect_kind      TEXT NOT NULL,
    outcome_json     TEXT NOT NULL,
    created_at_ms    INTEGER NOT NULL,
    PRIMARY KEY (session_id, turn_id, idempotency_key)
);

CREATE TABLE IF NOT EXISTS attachment_manifest (
    attachment_id    TEXT PRIMARY KEY,
    session_id       TEXT NOT NULL,
    canonical_uri    TEXT NOT NULL,
    intent_at_ms     INTEGER NOT NULL,
    committed_at_ms  INTEGER
);

CREATE INDEX IF NOT EXISTS idx_attachment_manifest_session
    ON attachment_manifest(session_id, committed_at_ms);
CREATE INDEX IF NOT EXISTS idx_attachment_manifest_uncommitted
    ON attachment_manifest(committed_at_ms)
    WHERE committed_at_ms IS NULL;
";

/// Canonical schema version. There is no migration chain — older databases
/// must be deleted before opening. See the [`SCHEMA`] doc comment for the
/// rationale.
const SCHEMA_VERSION: i32 = 1;

const SQLITE_BUSY_TIMEOUT: Duration = Duration::from_secs(15);
const SQLITE_WAL_AUTOCHECKPOINT_PAGES: i64 = 1_000;

const PROCESS_SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS processes (
    process_id            TEXT PRIMARY KEY,
    registration_hash     TEXT NOT NULL,
    created_by_scope_key  TEXT NOT NULL,
    host_profile_id       TEXT NOT NULL,
    created_at_ms         INTEGER NOT NULL,
    updated_at_ms         INTEGER NOT NULL,
    record_json           TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS process_events (
    process_id        TEXT NOT NULL,
    sequence          INTEGER NOT NULL,
    event_type        TEXT NOT NULL,
    payload_hash      TEXT NOT NULL,
    idempotency_key   TEXT,
    occurred_at_ms    INTEGER NOT NULL,
    event_json        TEXT NOT NULL,
    PRIMARY KEY (process_id, sequence),
    FOREIGN KEY (process_id) REFERENCES processes(process_id) ON DELETE CASCADE
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_process_events_key
    ON process_events(process_id, idempotency_key)
    WHERE idempotency_key IS NOT NULL;

CREATE TABLE IF NOT EXISTS process_handle_grants (
    session_id       TEXT NOT NULL,
    process_id       TEXT NOT NULL,
    descriptor_json  TEXT NOT NULL,
    PRIMARY KEY (session_id, process_id),
    FOREIGN KEY (process_id) REFERENCES processes(process_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_process_handle_grants_process
    ON process_handle_grants(process_id);

CREATE TABLE IF NOT EXISTS process_wake_inbox (
    wake_id           TEXT PRIMARY KEY,
    target_scope_key  TEXT NOT NULL,
    process_id        TEXT NOT NULL,
    sequence          INTEGER NOT NULL,
    dedupe_key        TEXT NOT NULL,
    input             TEXT NOT NULL,
    created_at_ms     INTEGER NOT NULL,
    acknowledged_at_ms INTEGER,
    UNIQUE (target_scope_key, dedupe_key),
    FOREIGN KEY (process_id) REFERENCES processes(process_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_process_wake_inbox_target
    ON process_wake_inbox(target_scope_key, acknowledged_at_ms, created_at_ms);
";

const PROCESS_SCHEMA_VERSION: i32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StoreBacking {
    File,
    Memory,
}

fn apply_pragmas(conn: &Connection, backing: StoreBacking) -> rusqlite::Result<()> {
    conn.busy_timeout(SQLITE_BUSY_TIMEOUT)?;
    conn.execute_batch(
        "PRAGMA synchronous = NORMAL;
         PRAGMA foreign_keys = ON;
         PRAGMA cache_size = -2000;",
    )?;
    if matches!(backing, StoreBacking::File) {
        conn.execute_batch(&format!(
            "PRAGMA journal_mode = WAL;
             PRAGMA wal_autocheckpoint = {SQLITE_WAL_AUTOCHECKPOINT_PAGES};"
        ))?;
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum PersistedArtifactKind {
    GenericBlob,
    CheckpointManifest,
    ToolState,
    PluginSessionSnapshot,
    ExecutionStateSnapshot,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum BlobStorageHint {
    Compressible,
    InlinePreferred,
    LargePayload,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
enum BlobCompression {
    None,
    Zlib,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct BlobArtifactDescriptor {
    pub kind: PersistedArtifactKind,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub hints: Vec<BlobStorageHint>,
}

impl BlobArtifactDescriptor {
    pub fn new(kind: PersistedArtifactKind, hints: impl Into<Vec<BlobStorageHint>>) -> Self {
        Self {
            kind,
            hints: hints.into(),
        }
    }

    pub fn checkpoint_manifest() -> Self {
        Self::new(
            PersistedArtifactKind::CheckpointManifest,
            vec![BlobStorageHint::Compressible],
        )
    }

    pub fn tool_state_snapshot() -> Self {
        Self::new(
            PersistedArtifactKind::ToolState,
            vec![BlobStorageHint::Compressible, BlobStorageHint::LargePayload],
        )
    }

    pub fn plugin_session_snapshot() -> Self {
        Self::new(
            PersistedArtifactKind::PluginSessionSnapshot,
            vec![BlobStorageHint::Compressible, BlobStorageHint::LargePayload],
        )
    }

    pub fn execution_state_snapshot() -> Self {
        Self::new(
            PersistedArtifactKind::ExecutionStateSnapshot,
            vec![BlobStorageHint::Compressible, BlobStorageHint::LargePayload],
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct RetainedArtifactRef {
    pub blob_ref: BlobRef,
    pub kind: PersistedArtifactKind,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BuiltinBlobProfile {
    LowLatency,
    #[default]
    Balanced,
    Compact,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StoreGcPolicy {
    pub auto_run_every_commits: Option<u64>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StoreOptions {
    pub blob_profile: BuiltinBlobProfile,
    pub gc_policy: StoreGcPolicy,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct StoredBlobEnvelope {
    descriptor: BlobArtifactDescriptor,
    compression: BlobCompression,
    content: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct StoredSessionCheckpoint {
    pub checkpoint_ref: BlobRef,
    pub manifest: SessionCheckpoint,
}

/// Explicit first-party factory for one SQLite session database per Lash session.
///
/// Hosts opt into this by passing it to `lash::LashCoreBuilder::store_factory`.
/// The factory never becomes a default: app storage and runtime storage remain
/// host-owned decisions.
#[derive(Clone, Debug)]
pub struct SqliteSessionStoreFactory {
    root: PathBuf,
    options: StoreOptions,
}

impl SqliteSessionStoreFactory {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            options: StoreOptions::default(),
        }
    }

    pub fn with_options(root: impl Into<PathBuf>, options: StoreOptions) -> Self {
        Self {
            root: root.into(),
            options,
        }
    }

    pub fn path_for_session(&self, session_id: &str) -> PathBuf {
        self.root.join(safe_session_db_file_name(session_id))
    }
}

impl SessionStoreFactory for SqliteSessionStoreFactory {
    fn create_store(
        &self,
        request: &SessionStoreCreateRequest,
    ) -> Result<Arc<dyn RuntimePersistence>, String> {
        std::fs::create_dir_all(&self.root).map_err(|err| err.to_string())?;
        let path = self.path_for_session(&request.session_id);
        let store =
            Arc::new(Store::open_with_options(&path, self.options).map_err(|err| err.to_string())?);
        if store.load_session_meta().is_none() {
            store.save_session_meta(SessionMeta {
                session_id: request.session_id.clone(),
                session_name: request.session_id.clone(),
                created_at: current_timestamp_string(),
                model: request.policy.model.id.clone(),
                cwd: std::env::current_dir()
                    .ok()
                    .and_then(|path| path.to_str().map(str::to_string)),
                relation: request.relation.clone(),
            });
        }
        Ok(store as Arc<dyn RuntimePersistence>)
    }
}

fn safe_session_db_file_name(session_id: &str) -> String {
    let mut safe = session_id
        .chars()
        .map(|ch| match ch {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '-' | '_' => ch,
            _ => '_',
        })
        .collect::<String>();
    safe = safe.trim_matches('_').to_string();
    if safe.is_empty() {
        safe.push_str("session");
    }
    safe.truncate(80);
    let hash = format!("{:x}", Sha256::digest(session_id.as_bytes()));
    format!("{safe}-{}.db", &hash[..16])
}

fn current_timestamp_string() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format!("unix:{}", now.as_secs())
}

fn current_epoch_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn retained_artifact_refs(checkpoint: &SessionCheckpoint) -> Vec<RetainedArtifactRef> {
    let mut refs = Vec::new();
    if let Some(blob_ref) = &checkpoint.tool_state_ref {
        refs.push(RetainedArtifactRef {
            blob_ref: blob_ref.clone(),
            kind: PersistedArtifactKind::ToolState,
        });
    }
    if let Some(blob_ref) = &checkpoint.plugin_snapshot_ref {
        refs.push(RetainedArtifactRef {
            blob_ref: blob_ref.clone(),
            kind: PersistedArtifactKind::PluginSessionSnapshot,
        });
    }
    if let Some(blob_ref) = &checkpoint.execution_state_ref {
        refs.push(RetainedArtifactRef {
            blob_ref: blob_ref.clone(),
            kind: PersistedArtifactKind::ExecutionStateSnapshot,
        });
    }
    refs
}

fn session_head_meta(head: &SessionHead) -> SessionHeadMeta {
    SessionHeadMeta {
        session_id: head.session_id.clone(),
        head_revision: 0,
        config: head.config.clone(),
        checkpoint_ref: head.checkpoint_ref.clone(),
        leaf_node_id: head.graph.leaf_node_id.clone(),
        graph_node_count: head.graph.nodes.len(),
        token_ledger: Vec::new(),
    }
}

fn encode_json<T: serde::Serialize>(value: &T) -> String {
    serde_json::to_string(value).expect("persisted state should serialize")
}

fn should_compress_blob(
    profile: BuiltinBlobProfile,
    descriptor: &BlobArtifactDescriptor,
    len: usize,
) -> bool {
    if !descriptor.hints.contains(&BlobStorageHint::Compressible) {
        return false;
    }
    match profile {
        BuiltinBlobProfile::LowLatency => false,
        BuiltinBlobProfile::Balanced => len >= 4 * 1024,
        BuiltinBlobProfile::Compact => len >= 1024,
    }
}

fn compress_blob(content: &[u8]) -> Vec<u8> {
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
    std::io::Write::write_all(&mut encoder, content).expect("compress blob");
    encoder.finish().expect("submit blob compression")
}

fn decompress_blob(content: &[u8]) -> Option<Vec<u8>> {
    let mut decoder = ZlibDecoder::new(content);
    let mut out = Vec::new();
    std::io::Read::read_to_end(&mut decoder, &mut out).ok()?;
    Some(out)
}

fn encode_artifact_blob(
    descriptor: &BlobArtifactDescriptor,
    profile: BuiltinBlobProfile,
    content: &[u8],
) -> Vec<u8> {
    let (compression, stored_content) = if should_compress_blob(profile, descriptor, content.len())
    {
        (BlobCompression::Zlib, compress_blob(content))
    } else {
        (BlobCompression::None, content.to_vec())
    };
    encode_msgpack(&StoredBlobEnvelope {
        descriptor: descriptor.clone(),
        compression,
        content: stored_content,
    })
}

fn decode_artifact_blob(bytes: &[u8]) -> Option<Vec<u8>> {
    let envelope = decode_msgpack::<StoredBlobEnvelope>(bytes)?;
    match envelope.compression {
        BlobCompression::None => Some(envelope.content),
        BlobCompression::Zlib => decompress_blob(&envelope.content),
    }
}

fn load_session_head_meta_from_conn(conn: &Connection) -> Option<SessionHeadMeta> {
    let (head_json, head_revision): (String, u64) = conn
        .query_row(
            "SELECT head_json, head_revision FROM session_head WHERE singleton = 1",
            [],
            |row| Ok((row.get(0)?, row.get::<_, i64>(1)? as u64)),
        )
        .ok()?;
    let mut meta: SessionHeadMeta = serde_json::from_str(&head_json).ok()?;
    meta.head_revision = head_revision;
    Some(meta)
}

fn load_session_meta_from_conn(conn: &Connection) -> Option<SessionMeta> {
    conn.query_row(
        "SELECT session_id, session_name, created_at, model, cwd, relation_json
         FROM session_meta WHERE singleton = 1",
        [],
        |row| {
            let relation_json: Option<String> = row.get(5)?;
            let relation = relation_json
                .and_then(|json| serde_json::from_str(&json).ok())
                .unwrap_or_default();
            Ok(SessionMeta {
                session_id: row.get(0)?,
                session_name: row.get(1)?,
                created_at: row.get(2)?,
                model: row.get(3)?,
                cwd: row.get(4)?,
                relation,
            })
        },
    )
    .ok()
}

fn load_runtime_turn_lease_from_conn(
    conn: &Connection,
    session_id: &str,
    turn_id: &str,
) -> rusqlite::Result<Option<RuntimeTurnLease>> {
    conn.query_row(
        "SELECT lease_owner_id, lease_token, lease_fencing_token, lease_claimed_at_ms, lease_expires_at_ms
         FROM runtime_turn_checkpoints
         WHERE session_id = ?1 AND turn_id = ?2",
        params![session_id, turn_id],
        |row| {
            let owner_id: Option<String> = row.get(0)?;
            let lease_token: Option<String> = row.get(1)?;
            let Some(owner_id) = owner_id else {
                return Ok(None);
            };
            let Some(lease_token) = lease_token else {
                return Ok(None);
            };
            Ok(Some(RuntimeTurnLease {
                schema_version: RUNTIME_TURN_LEASE_SCHEMA_VERSION,
                session_id: session_id.to_string(),
                turn_id: turn_id.to_string(),
                owner_id,
                lease_token,
                fencing_token: row.get::<_, i64>(2)? as u64,
                claimed_at_epoch_ms: row.get::<_, i64>(3)? as u64,
                expires_at_epoch_ms: row.get::<_, i64>(4)? as u64,
            }))
        },
    )
    .optional()
    .map(|lease| lease.flatten())
}

fn ensure_runtime_turn_lease_conn(
    conn: &Connection,
    lease: &RuntimeTurnLease,
) -> Result<(), StoreError> {
    let now = current_epoch_ms();
    let Some(current) = load_runtime_turn_lease_from_conn(conn, &lease.session_id, &lease.turn_id)
        .map_err(sqlite_error)?
    else {
        return Err(StoreError::RuntimeTurnLeaseExpired {
            session_id: lease.session_id.clone(),
            turn_id: lease.turn_id.clone(),
        });
    };
    if current.lease_token != lease.lease_token || current.expires_at_epoch_ms <= now {
        return Err(StoreError::RuntimeTurnLeaseExpired {
            session_id: lease.session_id.clone(),
            turn_id: lease.turn_id.clone(),
        });
    }
    Ok(())
}

fn ensure_runtime_turn_completion_conn(
    conn: &Connection,
    completed: &lash_core::RuntimeTurnCompletion,
) -> Result<(), StoreError> {
    let now = current_epoch_ms();
    let Some(current) =
        load_runtime_turn_lease_from_conn(conn, &completed.session_id, &completed.turn_id)
            .map_err(sqlite_error)?
    else {
        return Err(StoreError::RuntimeTurnLeaseExpired {
            session_id: completed.session_id.clone(),
            turn_id: completed.turn_id.clone(),
        });
    };
    if current.lease_token != completed.lease_token || current.expires_at_epoch_ms <= now {
        return Err(StoreError::RuntimeTurnLeaseExpired {
            session_id: completed.session_id.clone(),
            turn_id: completed.turn_id.clone(),
        });
    }
    Ok(())
}

fn decode_checkpoint(bytes: &[u8]) -> Option<SessionCheckpoint> {
    rmp_serde::from_slice(bytes).ok()
}

fn encode_msgpack<T: serde::Serialize>(value: &T) -> Vec<u8> {
    // Pre-size the buffer so the per-byte writes inside rmp_serde don't
    // walk the Vec through 0→4→8→16→32… reallocations on every call.
    // 1 KiB covers most small records (head meta, ledger entries) without
    // wasting much for the rare large blob.
    let mut buf = Vec::with_capacity(1024);
    rmp_serde::encode::write_named(&mut buf, value).expect("value should serialize");
    buf
}

fn decode_msgpack<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> Option<T> {
    rmp_serde::from_slice(bytes).ok()
}

fn merge_token_ledger_entries(
    entries: Vec<lash_core::TokenLedgerEntry>,
) -> Vec<lash_core::TokenLedgerEntry> {
    let mut merged: Vec<lash_core::TokenLedgerEntry> = Vec::new();
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

impl SqliteProcessRegistry {
    pub fn open(path: &Path) -> rusqlite::Result<Self> {
        let conn = Connection::open(path)?;
        apply_pragmas(&conn, StoreBacking::File)?;
        ensure_process_schema(&conn)?;
        Ok(Self {
            conn: Mutex::new(conn),
            notify: tokio::sync::Notify::new(),
        })
    }

    pub fn memory() -> rusqlite::Result<Self> {
        let conn = Connection::open_in_memory()?;
        apply_pragmas(&conn, StoreBacking::Memory)?;
        ensure_process_schema(&conn)?;
        Ok(Self {
            conn: Mutex::new(conn),
            notify: tokio::sync::Notify::new(),
        })
    }

    fn load_process_conn(
        conn: &Connection,
        process_id: &str,
    ) -> Result<Option<ProcessRecord>, lash_core::PluginError> {
        let json: Option<String> = conn
            .query_row(
                "SELECT record_json FROM processes WHERE process_id = ?1",
                params![process_id],
                |row| row.get(0),
            )
            .optional()
            .map_err(process_sqlite_error)?;
        json.map(|json| serde_json::from_str(&json).map_err(process_decode_error))
            .transpose()
    }

    fn save_process_conn(
        conn: &Connection,
        record: &ProcessRecord,
    ) -> Result<(), lash_core::PluginError> {
        conn.execute(
            "UPDATE processes
             SET updated_at_ms = ?2, record_json = ?3
             WHERE process_id = ?1",
            params![
                &record.id,
                record.updated_at_ms as i64,
                process_encode_json(record)?
            ],
        )
        .map_err(process_sqlite_error)?;
        Ok(())
    }

    fn load_event_by_key_conn(
        conn: &Connection,
        process_id: &str,
        idempotency_key: &str,
    ) -> Result<Option<(String, ProcessEvent)>, lash_core::PluginError> {
        let row: Option<(String, String)> = conn
            .query_row(
                "SELECT payload_hash, event_json
                 FROM process_events
                 WHERE process_id = ?1 AND idempotency_key = ?2",
                params![process_id, idempotency_key],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()
            .map_err(process_sqlite_error)?;
        row.map(|(hash, json)| {
            serde_json::from_str(&json)
                .map(|event| (hash, event))
                .map_err(process_decode_error)
        })
        .transpose()
    }

    fn insert_wake_conn(
        conn: &Connection,
        delivery: ProcessWakeDelivery,
    ) -> Result<(), lash_core::PluginError> {
        conn.execute(
            "INSERT OR IGNORE INTO process_wake_inbox (
                wake_id, target_scope_key, process_id, sequence, dedupe_key, input, created_at_ms
             )
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                delivery.wake_id,
                delivery.target_scope_key,
                delivery.process_id,
                delivery.sequence as i64,
                delivery.dedupe_key,
                delivery.input,
                delivery.created_at_ms as i64,
            ],
        )
        .map_err(process_sqlite_error)?;
        Ok(())
    }
}

#[async_trait::async_trait]
impl ProcessRegistry for SqliteProcessRegistry {
    async fn register_process(
        &self,
        registration: ProcessRegistration,
    ) -> Result<ProcessRecord, lash_core::PluginError> {
        let (registration, registration_hash) = prepare_process_registration(registration)?;
        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction().map_err(process_sqlite_error)?;
        if let Some(existing) = Self::load_process_conn(&tx, &registration.id)? {
            if existing.registration_hash == registration_hash {
                return Ok(existing);
            }
            return Err(lash_core::PluginError::Session(format!(
                "process `{}` registration hash conflict: existing {}, new {}",
                registration.id, existing.registration_hash, registration_hash
            )));
        }
        let now = current_epoch_ms();
        let record =
            ProcessRecord::from_prepared_registration(registration, registration_hash, now);
        tx.execute(
            "INSERT INTO processes (
                process_id, registration_hash, created_by_scope_key, host_profile_id,
                created_at_ms, updated_at_ms, record_json
             )
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                &record.id,
                &record.registration_hash,
                &record.created_by_scope_key,
                &record.host_profile_id,
                record.created_at_ms as i64,
                record.updated_at_ms as i64,
                process_encode_json(&record)?,
            ],
        )
        .map_err(process_sqlite_error)?;
        tx.commit().map_err(process_sqlite_error)?;
        Ok(record)
    }

    async fn set_external_ref(
        &self,
        process_id: &str,
        external_ref: ProcessExternalRef,
    ) -> Result<ProcessRecord, lash_core::PluginError> {
        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction().map_err(process_sqlite_error)?;
        let mut record = Self::load_process_conn(&tx, process_id)?.ok_or_else(|| {
            lash_core::PluginError::Session(format!("unknown process `{process_id}`"))
        })?;
        record.external_ref = Some(external_ref);
        record.updated_at_ms = current_epoch_ms();
        Self::save_process_conn(&tx, &record)?;
        tx.commit().map_err(process_sqlite_error)?;
        Ok(record)
    }

    async fn grant_handle(
        &self,
        session_id: &str,
        process_id: &str,
        descriptor: ProcessHandleDescriptor,
    ) -> Result<ProcessHandleGrant, lash_core::PluginError> {
        let conn = self.conn.lock().unwrap();
        if Self::load_process_conn(&conn, process_id)?.is_none() {
            return Err(lash_core::PluginError::Session(format!(
                "unknown process `{process_id}`"
            )));
        }
        conn.execute(
            "INSERT INTO process_handle_grants (session_id, process_id, descriptor_json)
             VALUES (?1, ?2, ?3)
             ON CONFLICT(session_id, process_id) DO UPDATE SET
                descriptor_json = excluded.descriptor_json",
            params![session_id, process_id, process_encode_json(&descriptor)?],
        )
        .map_err(process_sqlite_error)?;
        Ok(ProcessHandleGrant {
            session_id: session_id.to_string(),
            process_id: process_id.to_string(),
            descriptor,
        })
    }

    async fn revoke_handle(
        &self,
        session_id: &str,
        process_id: &str,
    ) -> Result<(), lash_core::PluginError> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "DELETE FROM process_handle_grants WHERE session_id = ?1 AND process_id = ?2",
            params![session_id, process_id],
        )
        .map_err(process_sqlite_error)?;
        Ok(())
    }

    async fn transfer_handle_grants(
        &self,
        from_session_id: &str,
        to_session_id: &str,
        process_ids: &[String],
    ) -> Result<(), lash_core::PluginError> {
        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction().map_err(process_sqlite_error)?;
        for process_id in process_ids {
            let descriptor_json: Option<String> = tx
                .query_row(
                    "SELECT descriptor_json
                     FROM process_handle_grants
                     WHERE session_id = ?1 AND process_id = ?2",
                    params![from_session_id, process_id],
                    |row| row.get(0),
                )
                .optional()
                .map_err(process_sqlite_error)?;
            let Some(descriptor_json) = descriptor_json else {
                return Err(lash_core::PluginError::Session(format!(
                    "process handle `{process_id}` is not granted to session `{from_session_id}`"
                )));
            };
            tx.execute(
                "DELETE FROM process_handle_grants
                 WHERE session_id = ?1 AND process_id = ?2",
                params![from_session_id, process_id],
            )
            .map_err(process_sqlite_error)?;
            tx.execute(
                "INSERT INTO process_handle_grants (session_id, process_id, descriptor_json)
                 VALUES (?1, ?2, ?3)
                 ON CONFLICT(session_id, process_id) DO UPDATE SET
                    descriptor_json = excluded.descriptor_json",
                params![to_session_id, process_id, descriptor_json],
            )
            .map_err(process_sqlite_error)?;
        }
        tx.commit().map_err(process_sqlite_error)?;
        Ok(())
    }

    async fn list_handle_grants(
        &self,
        session_id: &str,
    ) -> Result<Vec<ProcessHandleGrantEntry>, lash_core::PluginError> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn
            .prepare(
                "SELECT g.process_id, g.descriptor_json, p.record_json
                 FROM process_handle_grants g
                 JOIN processes p ON p.process_id = g.process_id
                 WHERE g.session_id = ?1
                 ORDER BY g.process_id ASC",
            )
            .map_err(process_sqlite_error)?;
        let rows = stmt
            .query_map(params![session_id], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                ))
            })
            .map_err(process_sqlite_error)?;
        let mut entries = Vec::new();
        for row in rows {
            let (process_id, descriptor_json, record_json) = row.map_err(process_sqlite_error)?;
            let descriptor: ProcessHandleDescriptor =
                serde_json::from_str(&descriptor_json).map_err(process_decode_error)?;
            let record: ProcessRecord =
                serde_json::from_str(&record_json).map_err(process_decode_error)?;
            entries.push((
                ProcessHandleGrant {
                    session_id: session_id.to_string(),
                    process_id,
                    descriptor,
                },
                record,
            ));
        }
        Ok(entries)
    }

    async fn handle_grants_for_process(
        &self,
        process_id: &str,
    ) -> Result<Vec<ProcessHandleGrant>, lash_core::PluginError> {
        let conn = self.conn.lock().unwrap();
        if Self::load_process_conn(&conn, process_id)?.is_none() {
            return Err(lash_core::PluginError::Session(format!(
                "unknown process `{process_id}`"
            )));
        }
        let mut stmt = conn
            .prepare(
                "SELECT session_id, descriptor_json
                 FROM process_handle_grants
                 WHERE process_id = ?1
                 ORDER BY session_id ASC",
            )
            .map_err(process_sqlite_error)?;
        let rows = stmt
            .query_map(params![process_id], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })
            .map_err(process_sqlite_error)?;
        let mut grants = Vec::new();
        for row in rows {
            let (session_id, descriptor_json) = row.map_err(process_sqlite_error)?;
            let descriptor: ProcessHandleDescriptor =
                serde_json::from_str(&descriptor_json).map_err(process_decode_error)?;
            grants.push(ProcessHandleGrant {
                session_id,
                process_id: process_id.to_string(),
                descriptor,
            });
        }
        Ok(grants)
    }

    async fn append_event(
        &self,
        process_id: &str,
        request: ProcessEventAppendRequest,
    ) -> Result<ProcessEvent, lash_core::PluginError> {
        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction().map_err(process_sqlite_error)?;
        let mut record = Self::load_process_conn(&tx, process_id)?.ok_or_else(|| {
            lash_core::PluginError::Session(format!("unknown process `{process_id}`"))
        })?;
        let payload_hash = process_event_payload_hash(&request.event_type, &request.payload)?;
        if let Some(idempotency_key) = request.idempotency_key.as_deref()
            && let Some((existing_hash, existing)) =
                Self::load_event_by_key_conn(&tx, process_id, idempotency_key)?
        {
            if existing_hash == payload_hash {
                return Ok(existing);
            }
            return Err(lash_core::PluginError::Session(format!(
                "process `{process_id}` event idempotency key `{idempotency_key}` conflicts with an existing event"
            )));
        }
        let declared = record
            .event_types
            .iter()
            .find(|declared| declared.name == request.event_type)
            .ok_or_else(|| {
                lash_core::PluginError::Session(format!(
                    "process `{process_id}` emitted undeclared event type `{}`",
                    request.event_type
                ))
            })?;
        require_event_idempotency(process_id, &request, &declared.semantics)?;
        declared
            .payload_schema
            .validate(&request.payload)
            .map_err(|err| {
                lash_core::PluginError::Session(format!(
                    "invalid `{}` payload: {err}",
                    request.event_type
                ))
            })?;
        let sequence = tx
            .query_row(
                "SELECT COALESCE(MAX(sequence), 0) + 1 FROM process_events WHERE process_id = ?1",
                params![process_id],
                |row| row.get::<_, i64>(0),
            )
            .map_err(process_sqlite_error)? as u64;
        let semantics = materialize_process_event_semantics(
            process_id,
            sequence,
            &request.payload,
            &declared.semantics,
        )?;
        if semantics.terminal.is_some() && record.terminal.is_some() {
            return Err(lash_core::PluginError::Session(format!(
                "process `{process_id}` is already terminal"
            )));
        }
        let occurred_at_ms = current_epoch_ms();
        let event = ProcessEvent {
            process_id: process_id.to_string(),
            sequence,
            event_type: request.event_type,
            payload: request.payload,
            idempotency_key: request.idempotency_key.clone(),
            semantics: semantics.clone(),
            occurred_at: system_time_from_epoch_ms(occurred_at_ms),
        };
        tx.execute(
            "INSERT INTO process_events (
                process_id, sequence, event_type, payload_hash, idempotency_key,
                occurred_at_ms, event_json
             )
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                process_id,
                sequence as i64,
                &event.event_type,
                &payload_hash,
                event.idempotency_key.as_deref(),
                occurred_at_ms as i64,
                process_encode_json(&event)?,
            ],
        )
        .map_err(process_sqlite_error)?;
        if let Some(terminal) = event.semantics.terminal.clone() {
            record.terminal = Some(terminal);
        }
        record.updated_at_ms = occurred_at_ms;
        Self::save_process_conn(&tx, &record)?;
        if let Some(wake) = semantics.wake
            && let Some(target_scope_key) = request.wake_target_scope_key
        {
            let delivery = process_wake_delivery(
                target_scope_key,
                process_id.to_string(),
                sequence,
                wake,
                event.occurred_at,
            )?;
            Self::insert_wake_conn(&tx, delivery)?;
        }
        tx.commit().map_err(process_sqlite_error)?;
        self.notify.notify_waiters();
        Ok(event)
    }

    async fn events_after(
        &self,
        process_id: &str,
        after_sequence: u64,
    ) -> Result<Vec<ProcessEvent>, lash_core::PluginError> {
        let conn = self.conn.lock().unwrap();
        if Self::load_process_conn(&conn, process_id)?.is_none() {
            return Err(lash_core::PluginError::Session(format!(
                "unknown process `{process_id}`"
            )));
        }
        let mut stmt = conn
            .prepare(
                "SELECT event_json FROM process_events
                 WHERE process_id = ?1 AND sequence > ?2
                 ORDER BY sequence ASC",
            )
            .map_err(process_sqlite_error)?;
        let rows = stmt
            .query_map(params![process_id, after_sequence as i64], |row| {
                row.get::<_, String>(0)
            })
            .map_err(process_sqlite_error)?;
        let mut events = Vec::new();
        for row in rows {
            events.push(
                serde_json::from_str(&row.map_err(process_sqlite_error)?)
                    .map_err(process_decode_error)?,
            );
        }
        Ok(events)
    }

    async fn wake_events_after(
        &self,
        process_id: &str,
        after_sequence: u64,
    ) -> Result<Vec<ProcessEvent>, lash_core::PluginError> {
        Ok(self
            .events_after(process_id, after_sequence)
            .await?
            .into_iter()
            .filter(|event| event.semantics.wake.is_some())
            .collect())
    }

    async fn wait_event_after(
        &self,
        process_id: &str,
        event_type: &str,
        after_sequence: u64,
    ) -> Result<ProcessEvent, lash_core::PluginError> {
        loop {
            if let Some(event) = self
                .events_after(process_id, after_sequence)
                .await?
                .into_iter()
                .find(|event| event.event_type == event_type)
            {
                return Ok(event);
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    async fn await_process(
        &self,
        process_id: &str,
    ) -> Result<ProcessAwaitOutput, lash_core::PluginError> {
        loop {
            let record = self.get_process(process_id).await.ok_or_else(|| {
                lash_core::PluginError::Session(format!("unknown process `{process_id}`"))
            })?;
            if let Some(terminal) = record.terminal {
                return Ok(terminal.await_output);
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    async fn complete_process(
        &self,
        process_id: &str,
        await_output: ProcessAwaitOutput,
    ) -> Result<ProcessRecord, lash_core::PluginError> {
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
            .with_idempotency_key(format!("process:{process_id}:terminal:{event_type}")),
        )
        .await?;
        self.get_process(process_id).await.ok_or_else(|| {
            lash_core::PluginError::Session(format!(
                "unknown process `{process_id}` after terminal event"
            ))
        })
    }

    async fn get_process(&self, process_id: &str) -> Option<ProcessRecord> {
        let conn = self.conn.lock().ok()?;
        Self::load_process_conn(&conn, process_id).ok().flatten()
    }

    async fn drain_wake_inputs(
        &self,
        target_scope_key: &str,
        limit: usize,
    ) -> Result<Vec<ProcessWakeDelivery>, lash_core::PluginError> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn
            .prepare(
                "SELECT wake_id, target_scope_key, process_id, sequence, dedupe_key, input, created_at_ms
                 FROM process_wake_inbox
                 WHERE target_scope_key = ?1 AND acknowledged_at_ms IS NULL
                 ORDER BY created_at_ms ASC, wake_id ASC
                 LIMIT ?2",
            )
            .map_err(process_sqlite_error)?;
        let rows = stmt
            .query_map(params![target_scope_key, limit as i64], |row| {
                Ok(ProcessWakeDelivery {
                    wake_id: row.get(0)?,
                    target_scope_key: row.get(1)?,
                    process_id: row.get(2)?,
                    sequence: row.get::<_, i64>(3)? as u64,
                    dedupe_key: row.get(4)?,
                    input: row.get(5)?,
                    created_at_ms: row.get::<_, i64>(6)? as u64,
                })
            })
            .map_err(process_sqlite_error)?;
        rows.collect::<Result<Vec<_>, _>>()
            .map_err(process_sqlite_error)
    }

    async fn ack_wake_input(&self, wake_id: &str) -> Result<(), lash_core::PluginError> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "UPDATE process_wake_inbox
             SET acknowledged_at_ms = COALESCE(acknowledged_at_ms, ?2)
             WHERE wake_id = ?1",
            params![wake_id, current_epoch_ms() as i64],
        )
        .map_err(process_sqlite_error)?;
        Ok(())
    }

    async fn ack_wake(
        &self,
        process_id: &str,
        sequence: u64,
    ) -> Result<(), lash_core::PluginError> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "UPDATE process_wake_inbox
             SET acknowledged_at_ms = COALESCE(acknowledged_at_ms, ?3)
             WHERE process_id = ?1 AND sequence = ?2",
            params![process_id, sequence as i64, current_epoch_ms() as i64],
        )
        .map_err(process_sqlite_error)?;
        Ok(())
    }
}

impl Store {
    fn load_session_graph_from_conn(
        conn: &Connection,
        leaf_node_id: Option<String>,
    ) -> lash_core::SessionGraph {
        // Tombstoned rows are physically still present until `vacuum()` is
        // called; the runtime view should never see them.
        let mut stmt = match conn
            .prepare("SELECT node_json FROM graph_nodes WHERE tombstoned = 0 ORDER BY seq ASC")
        {
            Ok(stmt) => stmt,
            Err(err) => {
                tracing::warn!(error = %err, "failed to prepare graph load statement");
                return lash_core::SessionGraph::from_nodes(Vec::new(), leaf_node_id);
            }
        };
        let rows = match stmt.query_map([], |row| row.get::<_, String>(0)) {
            Ok(rows) => rows,
            Err(err) => {
                tracing::warn!(error = %err, "failed to query graph rows");
                return lash_core::SessionGraph::from_nodes(Vec::new(), leaf_node_id);
            }
        };
        let mut nodes = Vec::new();
        for row in rows {
            let Ok(node_json) = row else {
                continue;
            };
            let Ok(node) = serde_json::from_str::<lash_core::SessionNodeRecord>(&node_json) else {
                continue;
            };
            nodes.push(node);
        }
        lash_core::SessionGraph::from_nodes(nodes, leaf_node_id)
    }

    fn load_active_path_session_graph_from_conn(
        conn: &Connection,
        leaf_node_id: Option<String>,
    ) -> rusqlite::Result<lash_core::SessionGraph> {
        let Some(leaf_node_id) = leaf_node_id else {
            return Ok(lash_core::SessionGraph::default());
        };
        let mut stmt = conn.prepare(
            "WITH RECURSIVE active(node_id, node_json, parent_node_id, depth) AS (
                SELECT
                    node_id,
                    node_json,
                    json_extract(node_json, '$.parent_node_id'),
                    0
                FROM graph_nodes
                WHERE node_id = ?1 AND tombstoned = 0
              UNION ALL
                SELECT
                    g.node_id,
                    g.node_json,
                    json_extract(g.node_json, '$.parent_node_id'),
                    active.depth + 1
                FROM graph_nodes g
                JOIN active ON g.node_id = active.parent_node_id
                WHERE g.tombstoned = 0
            )
            SELECT node_json FROM active ORDER BY depth DESC",
        )?;
        let rows = stmt.query_map(params![leaf_node_id.as_str()], |row| {
            row.get::<_, String>(0)
        })?;
        let mut nodes = Vec::new();
        for row in rows {
            let node_json = row?;
            if let Ok(node) = serde_json::from_str::<lash_core::SessionNodeRecord>(&node_json) {
                nodes.push(node);
            }
        }
        Ok(lash_core::SessionGraph::from_nodes(
            nodes,
            Some(leaf_node_id),
        ))
    }

    fn insert_artifact_blob_conn(
        conn: &Connection,
        descriptor: BlobArtifactDescriptor,
        content: &[u8],
        profile: BuiltinBlobProfile,
    ) -> rusqlite::Result<BlobRef> {
        let hash = format!("{:x}", Sha256::digest(content));
        let stored = encode_artifact_blob(&descriptor, profile, content);
        conn.execute(
            "INSERT OR IGNORE INTO blobs (hash, content) VALUES (?1, ?2)",
            params![hash, stored],
        )?;
        Ok(BlobRef(hash))
    }

    fn put_typed_artifact_blob_conn<T: serde::Serialize>(
        conn: &Connection,
        descriptor: BlobArtifactDescriptor,
        value: &T,
        profile: BuiltinBlobProfile,
    ) -> rusqlite::Result<BlobRef> {
        let bytes = encode_msgpack(value);
        Self::insert_artifact_blob_conn(conn, descriptor, &bytes, profile)
    }

    fn put_checkpoint_conn(
        conn: &Connection,
        checkpoint: &HydratedSessionCheckpoint,
        profile: BuiltinBlobProfile,
    ) -> rusqlite::Result<StoredSessionCheckpoint> {
        let tool_state_ref = checkpoint
            .tool_state
            .as_ref()
            .map(|snapshot| {
                Self::put_typed_artifact_blob_conn(
                    conn,
                    BlobArtifactDescriptor::tool_state_snapshot(),
                    snapshot,
                    profile,
                )
            })
            .transpose()?
            .or_else(|| checkpoint.tool_state_ref.clone());
        let plugin_snapshot_ref = checkpoint
            .plugin_snapshot
            .as_ref()
            .map(|snapshot| {
                Self::put_typed_artifact_blob_conn(
                    conn,
                    BlobArtifactDescriptor::plugin_session_snapshot(),
                    snapshot,
                    profile,
                )
            })
            .transpose()?
            .or_else(|| checkpoint.plugin_snapshot_ref.clone());
        let execution_state_ref = checkpoint
            .execution_state
            .as_ref()
            .map(|snapshot| {
                Self::put_typed_artifact_blob_conn(
                    conn,
                    BlobArtifactDescriptor::execution_state_snapshot(),
                    snapshot,
                    profile,
                )
            })
            .transpose()?
            .or_else(|| checkpoint.execution_state_ref.clone());
        let manifest = SessionCheckpoint {
            turn_state: checkpoint.turn_state.clone(),
            tool_state_ref,
            plugin_snapshot_ref,
            plugin_snapshot_revision: checkpoint.plugin_snapshot_revision,
            execution_state_ref,
        };
        let checkpoint_ref = Self::put_typed_artifact_blob_conn(
            conn,
            BlobArtifactDescriptor::checkpoint_manifest(),
            &manifest,
            profile,
        )?;
        Ok(StoredSessionCheckpoint {
            checkpoint_ref,
            manifest,
        })
    }

    fn get_blob_conn(conn: &Connection, blob_ref: &BlobRef) -> Option<Vec<u8>> {
        let bytes: Vec<u8> = conn
            .query_row(
                "SELECT content FROM blobs WHERE hash = ?1",
                params![blob_ref.as_str()],
                |row| row.get(0),
            )
            .ok()?;
        decode_artifact_blob(&bytes).or(Some(bytes))
    }

    fn get_typed_blob_conn<T: serde::de::DeserializeOwned>(
        conn: &Connection,
        blob_ref: &BlobRef,
    ) -> Option<T> {
        let bytes = Self::get_blob_conn(conn, blob_ref)?;
        decode_msgpack(&bytes)
    }

    fn get_checkpoint_conn(
        conn: &Connection,
        blob_ref: &BlobRef,
    ) -> Option<HydratedSessionCheckpoint> {
        let record: SessionCheckpoint = Self::get_typed_blob_conn(conn, blob_ref)?;
        Some(HydratedSessionCheckpoint {
            turn_state: record.turn_state,
            tool_state_ref: record.tool_state_ref.clone(),
            tool_state: record
                .tool_state_ref
                .as_ref()
                .and_then(|blob_ref| Self::get_typed_blob_conn(conn, blob_ref)),
            plugin_snapshot_ref: record.plugin_snapshot_ref.clone(),
            plugin_snapshot: record
                .plugin_snapshot_ref
                .as_ref()
                .and_then(|blob_ref| Self::get_typed_blob_conn(conn, blob_ref)),
            plugin_snapshot_revision: record.plugin_snapshot_revision,
            execution_state_ref: record.execution_state_ref.clone(),
            execution_state: record
                .execution_state_ref
                .as_ref()
                .and_then(|blob_ref| Self::get_typed_blob_conn(conn, blob_ref)),
        })
    }

    fn load_usage_deltas_conn(conn: &Connection) -> Vec<lash_core::TokenLedgerEntry> {
        let mut stmt = match conn.prepare(
            "SELECT source, model, input_tokens, output_tokens, cached_input_tokens, reasoning_tokens
             FROM usage_deltas ORDER BY seq ASC",
        ) {
            Ok(stmt) => stmt,
            Err(_) => return Vec::new(),
        };
        let rows = match stmt.query_map([], |row| {
            Ok(lash_core::TokenLedgerEntry {
                source: row.get(0)?,
                model: row.get(1)?,
                usage: lash_core::TokenUsage {
                    input_tokens: row.get(2)?,
                    output_tokens: row.get(3)?,
                    cached_input_tokens: row.get(4)?,
                    reasoning_tokens: row.get(5)?,
                },
            })
        }) {
            Ok(rows) => rows,
            Err(_) => return Vec::new(),
        };
        rows.filter_map(Result::ok).collect()
    }

    fn maybe_auto_gc(&self) {
        let Some(interval) = self.options.gc_policy.auto_run_every_commits else {
            return;
        };
        let commits = self.commit_count.fetch_add(1, AtomicOrdering::Relaxed) + 1;
        if interval != 0 && commits % interval == 0 {
            let _ = self.gc_unreachable();
        }
    }

    /// Open (or create) a SQLite database at `path`.
    pub fn open(path: &Path) -> rusqlite::Result<Self> {
        Self::open_with_options(path, StoreOptions::default())
    }

    /// Open (or create) a SQLite database at `path` with explicit store options.
    pub fn open_with_options(path: &Path, options: StoreOptions) -> rusqlite::Result<Self> {
        let conn = Connection::open(path)?;
        apply_pragmas(&conn, StoreBacking::File)?;
        ensure_schema(&conn)?;
        Ok(Self {
            conn: Mutex::new(conn),
            options,
            commit_count: AtomicU64::new(0),
        })
    }

    /// Open a SQLite database read-only with minimal setup (no schema check).
    /// Used for fast metadata reads like the session picker.
    pub fn open_readonly(path: &Path) -> rusqlite::Result<Self> {
        let flags = OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX;
        let conn = Connection::open_with_flags(path, flags)?;
        conn.busy_timeout(Duration::from_secs(1))?;
        conn.execute_batch("PRAGMA cache_size = -500;")?;
        Ok(Self {
            conn: Mutex::new(conn),
            options: StoreOptions::default(),
            commit_count: AtomicU64::new(0),
        })
    }

    /// Fast picker info: session_meta + first user prompt + user turn count from the persisted graph.
    pub fn load_picker_info(&self) -> Option<SessionPickerInfo> {
        let conn = self.conn.lock().unwrap();
        let meta = conn
            .query_row(
                "SELECT session_id, cwd, relation_json
                 FROM session_meta WHERE singleton = 1",
                [],
                |row| {
                    let relation_json: Option<String> = row.get(2)?;
                    let relation = relation_json
                        .and_then(|json| serde_json::from_str(&json).ok())
                        .unwrap_or_default();
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, Option<String>>(1)?,
                        relation,
                    ))
                },
            )
            .ok()?;

        let head_json: String = conn
            .query_row(
                "SELECT head_json FROM session_head WHERE singleton = 1",
                [],
                |row| row.get(0),
            )
            .unwrap_or_else(|_| "{}".to_string());
        let head_meta = serde_json::from_str::<SessionHeadMeta>(&head_json).unwrap_or_default();
        let graph = Self::load_session_graph_from_conn(&conn, head_meta.leaf_node_id);

        Some(SessionPickerInfo {
            session_id: meta.0,
            cwd: meta.1,
            relation: meta.2,
            first_user_message: graph.first_user_message(),
            user_message_count: graph.user_message_count(),
        })
    }

    /// In-memory database (for child-session flows / tests).
    pub fn memory() -> rusqlite::Result<Self> {
        Self::memory_with_options(StoreOptions {
            blob_profile: BuiltinBlobProfile::LowLatency,
            gc_policy: StoreGcPolicy::default(),
        })
    }

    /// In-memory database (for child-session flows / tests) with explicit store options.
    pub fn memory_with_options(options: StoreOptions) -> rusqlite::Result<Self> {
        let conn = Connection::open_in_memory()?;
        apply_pragmas(&conn, StoreBacking::Memory)?;
        ensure_schema(&conn)?;
        Ok(Self {
            conn: Mutex::new(conn),
            options,
            commit_count: AtomicU64::new(0),
        })
    }

    pub fn put_blob(&self, content: &[u8]) -> BlobRef {
        let hash = format!("{:x}", Sha256::digest(content));
        let conn = self.conn.lock().unwrap();
        if let Err(err) = conn.execute(
            "INSERT OR IGNORE INTO blobs (hash, content) VALUES (?1, ?2)",
            params![hash, content],
        ) {
            tracing::warn!(error = %err, hash, "failed to persist checkpoint blob");
        }
        BlobRef(hash)
    }

    pub fn put_artifact_blob(&self, descriptor: BlobArtifactDescriptor, content: &[u8]) -> BlobRef {
        let hash = format!("{:x}", Sha256::digest(content));
        let stored = encode_artifact_blob(&descriptor, self.options.blob_profile, content);
        let conn = self.conn.lock().unwrap();
        if let Err(err) = conn.execute(
            "INSERT OR IGNORE INTO blobs (hash, content) VALUES (?1, ?2)",
            params![hash, stored],
        ) {
            tracing::warn!(error = %err, hash, "failed to persist artifact blob");
        }
        BlobRef(hash)
    }

    pub fn get_blob(&self, blob_ref: &BlobRef) -> Option<Vec<u8>> {
        let conn = self.conn.lock().unwrap();
        Self::get_blob_conn(&conn, blob_ref)
    }

    pub fn put_typed_blob<T: serde::Serialize>(&self, value: &T) -> BlobRef {
        let bytes = encode_msgpack(value);
        self.put_blob(&bytes)
    }

    pub fn put_typed_artifact_blob<T: serde::Serialize>(
        &self,
        descriptor: BlobArtifactDescriptor,
        value: &T,
    ) -> BlobRef {
        let bytes = encode_msgpack(value);
        self.put_artifact_blob(descriptor, &bytes)
    }

    pub fn get_typed_blob<T: serde::de::DeserializeOwned>(&self, blob_ref: &BlobRef) -> Option<T> {
        let conn = self.conn.lock().unwrap();
        Self::get_typed_blob_conn(&conn, blob_ref)
    }

    pub fn put_checkpoint(
        &self,
        checkpoint: &HydratedSessionCheckpoint,
    ) -> StoredSessionCheckpoint {
        let tool_state_ref = checkpoint
            .tool_state
            .as_ref()
            .map(|snapshot| {
                self.put_typed_artifact_blob(
                    BlobArtifactDescriptor::tool_state_snapshot(),
                    snapshot,
                )
            })
            .or_else(|| checkpoint.tool_state_ref.clone());
        let plugin_snapshot_ref = checkpoint
            .plugin_snapshot
            .as_ref()
            .map(|snapshot| {
                self.put_typed_artifact_blob(
                    BlobArtifactDescriptor::plugin_session_snapshot(),
                    snapshot,
                )
            })
            .or_else(|| checkpoint.plugin_snapshot_ref.clone());
        let execution_state_ref = checkpoint
            .execution_state
            .as_ref()
            .map(|snapshot| {
                self.put_typed_artifact_blob(
                    BlobArtifactDescriptor::execution_state_snapshot(),
                    snapshot,
                )
            })
            .or_else(|| checkpoint.execution_state_ref.clone());
        let manifest = SessionCheckpoint {
            turn_state: checkpoint.turn_state.clone(),
            tool_state_ref,
            plugin_snapshot_ref,
            plugin_snapshot_revision: checkpoint.plugin_snapshot_revision,
            execution_state_ref,
        };
        let checkpoint_ref =
            self.put_typed_artifact_blob(BlobArtifactDescriptor::checkpoint_manifest(), &manifest);
        StoredSessionCheckpoint {
            checkpoint_ref,
            manifest,
        }
    }

    pub fn get_checkpoint(&self, blob_ref: &BlobRef) -> Option<HydratedSessionCheckpoint> {
        let conn = self.conn.lock().unwrap();
        Self::get_checkpoint_conn(&conn, blob_ref)
    }

    pub fn append_usage_deltas(&self, entries: &[lash_core::TokenLedgerEntry]) {
        if entries.is_empty() {
            return;
        }
        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction().expect("usage delta transaction");
        {
            let mut stmt = tx
                .prepare(
                    "INSERT INTO usage_deltas (
                        source, model, input_tokens, output_tokens, cached_input_tokens, reasoning_tokens
                    ) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                )
                .expect("usage delta statement");
            for entry in entries {
                stmt.execute(params![
                    entry.source,
                    entry.model,
                    entry.usage.input_tokens,
                    entry.usage.output_tokens,
                    entry.usage.cached_input_tokens,
                    entry.usage.reasoning_tokens,
                ])
                .expect("usage delta insert");
            }
        }
        tx.commit().expect("usage delta commit");
    }

    pub fn load_usage_deltas(&self) -> Vec<lash_core::TokenLedgerEntry> {
        let conn = self.conn.lock().unwrap();
        Self::load_usage_deltas_conn(&conn)
    }

    pub fn save_session_head_meta(&self, meta: SessionHeadMeta) {
        let conn = self.conn.lock().unwrap();
        let head_json = encode_json(&meta);
        if let Err(err) = conn.execute(
            "INSERT OR REPLACE INTO session_head (singleton, session_id, head_json, head_revision)
             VALUES (1, ?1, ?2, ?3)",
            params![meta.session_id, head_json, meta.head_revision as i64],
        ) {
            tracing::warn!(error = %err, "failed to persist session head");
        }
    }

    pub fn load_session_head_meta(&self) -> Option<SessionHeadMeta> {
        let conn = self.conn.lock().unwrap();
        load_session_head_meta_from_conn(&conn)
    }

    pub fn replace_session_graph(&self, graph: &lash_core::SessionGraph) {
        let mut conn = self.conn.lock().unwrap();
        let tx = match conn.transaction() {
            Ok(tx) => tx,
            Err(err) => {
                tracing::warn!(error = %err, "failed to begin graph replace transaction");
                return;
            }
        };
        if let Err(err) = tx.execute("DELETE FROM graph_nodes", []) {
            tracing::warn!(error = %err, "failed to clear graph rows");
            return;
        }
        for node in &graph.nodes {
            let node_json = encode_json(node);
            if let Err(err) = tx.execute(
                "INSERT INTO graph_nodes (node_id, node_json) VALUES (?1, ?2)",
                params![node.node_id, node_json],
            ) {
                tracing::warn!(error = %err, node_id = %node.node_id, "failed to persist graph node");
                return;
            }
        }
        if let Err(err) = tx.commit() {
            tracing::warn!(error = %err, "failed to commit graph replace");
        }
    }

    pub fn append_session_graph_nodes(&self, nodes: &[lash_core::SessionNodeRecord]) {
        if nodes.is_empty() {
            return;
        }
        let mut conn = self.conn.lock().unwrap();
        let tx = match conn.transaction() {
            Ok(tx) => tx,
            Err(err) => {
                tracing::warn!(error = %err, "failed to begin graph append transaction");
                return;
            }
        };
        for node in nodes {
            let node_json = encode_json(node);
            if let Err(err) = tx.execute(
                "INSERT INTO graph_nodes (node_id, node_json) VALUES (?1, ?2)",
                params![node.node_id, node_json],
            ) {
                tracing::warn!(error = %err, node_id = %node.node_id, "failed to append graph node");
                return;
            }
        }
        if let Err(err) = tx.commit() {
            tracing::warn!(error = %err, "failed to commit graph append");
        }
    }

    pub fn load_session_graph(&self) -> lash_core::SessionGraph {
        let conn = self.conn.lock().unwrap();
        Self::load_session_graph_from_conn(&conn, None)
    }

    pub fn gc_unreachable(&self) -> GcReport {
        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction().expect("gc transaction");
        let head_meta = load_session_head_meta_from_conn(&tx);
        let mut roots = Vec::new();
        if let Some(checkpoint_ref) = head_meta
            .as_ref()
            .and_then(|meta| meta.checkpoint_ref.as_ref())
            .cloned()
        {
            roots.push(RetainedArtifactRef {
                blob_ref: checkpoint_ref,
                kind: PersistedArtifactKind::CheckpointManifest,
            });
        }
        let mut retained = std::collections::BTreeMap::<String, PersistedArtifactKind>::new();
        let mut stack = roots.clone();
        while let Some(current) = stack.pop() {
            if retained
                .insert(current.blob_ref.0.clone(), current.kind)
                .is_some()
            {
                continue;
            }
            if current.kind != PersistedArtifactKind::CheckpointManifest {
                continue;
            }
            let Some(bytes) = tx
                .query_row(
                    "SELECT content FROM blobs WHERE hash = ?1",
                    params![current.blob_ref.as_str()],
                    |row| row.get::<_, Vec<u8>>(0),
                )
                .ok()
            else {
                continue;
            };
            let Some(content) = decode_artifact_blob(&bytes).or(Some(bytes)) else {
                continue;
            };
            let Some(checkpoint) = decode_checkpoint(&content) else {
                continue;
            };
            stack.extend(retained_artifact_refs(&checkpoint));
        }
        let all_hashes = {
            let mut stmt = tx
                .prepare("SELECT hash FROM blobs ORDER BY hash ASC")
                .expect("prepare blob scan");
            let rows = stmt
                .query_map([], |row| row.get::<_, String>(0))
                .expect("query blob scan");
            rows.filter_map(Result::ok).collect::<Vec<_>>()
        };
        let mut deleted_blob_count = 0usize;
        for hash in &all_hashes {
            if retained.contains_key(hash) {
                continue;
            }
            tx.execute("DELETE FROM blobs WHERE hash = ?1", params![hash])
                .expect("delete unreachable blob");
            deleted_blob_count += 1;
        }
        tx.commit().expect("commit gc transaction");
        GcReport {
            root_count: roots.len(),
            retained_blob_count: retained.len(),
            deleted_blob_count,
        }
    }

    pub fn save_session_head(&self, head: SessionHead) {
        self.replace_session_graph(&head.graph);
        self.save_session_head_meta(session_head_meta(&head));
    }

    pub fn load_session_head(&self) -> Option<SessionHead> {
        let meta = self.load_session_head_meta()?;
        let mut graph = self.load_session_graph();
        graph.set_leaf_node_id(meta.leaf_node_id.clone());
        Some(SessionHead {
            session_id: meta.session_id,
            head_revision: meta.head_revision,
            graph,
            config: meta.config,
            checkpoint_ref: meta.checkpoint_ref,
            token_ledger: merge_token_ledger_entries(self.load_usage_deltas()),
        })
    }

    pub fn head_copy_from_store(&self, source: &Store) {
        if let Some(head) = source.load_session_head() {
            if let Some(checkpoint_ref) = &head.checkpoint_ref
                && let Some(record) = source.get_typed_blob::<SessionCheckpoint>(checkpoint_ref)
            {
                for blob_ref in [
                    record.tool_state_ref.as_ref(),
                    record.plugin_snapshot_ref.as_ref(),
                ]
                .into_iter()
                .flatten()
                {
                    if let Some(blob) = source.get_blob(blob_ref) {
                        let descriptor = match record
                            .tool_state_ref
                            .as_ref()
                            .filter(|candidate| *candidate == blob_ref)
                        {
                            Some(_) => BlobArtifactDescriptor::tool_state_snapshot(),
                            None => BlobArtifactDescriptor::plugin_session_snapshot(),
                        };
                        let _ = self.put_artifact_blob(descriptor, &blob);
                    }
                }
                if let Some(blob) = source.get_blob(checkpoint_ref) {
                    let _ = self
                        .put_artifact_blob(BlobArtifactDescriptor::checkpoint_manifest(), &blob);
                }
            }
            self.replace_session_graph(&head.graph);
            self.save_session_head_meta(session_head_meta(&head));
        }
    }

    pub fn save_session_meta(&self, meta: SessionMeta) {
        let conn = self.conn.lock().unwrap();
        let relation_json = serde_json::to_string(&meta.relation).ok();
        if let Err(err) = conn.execute(
            "INSERT OR REPLACE INTO session_meta
             (singleton, session_id, session_name, created_at, model, cwd, relation_json)
             VALUES (1, ?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                meta.session_id,
                meta.session_name,
                meta.created_at,
                meta.model,
                meta.cwd,
                relation_json
            ],
        ) {
            tracing::warn!(
                error = %err,
                session_id = meta.session_id,
                "failed to persist session metadata"
            );
        }
    }

    pub fn load_session_meta(&self) -> Option<SessionMeta> {
        let conn = self.conn.lock().unwrap();
        load_session_meta_from_conn(&conn)
    }
}

#[async_trait::async_trait]
impl RuntimePersistence for Store {
    async fn load_session(
        &self,
        scope: SessionReadScope,
    ) -> Result<Option<PersistedSessionRead>, StoreError> {
        let conn = self.conn.lock().unwrap();
        let Some(meta) = load_session_head_meta_from_conn(&conn) else {
            return Ok(None);
        };
        let leaf_node_id = match &scope {
            SessionReadScope::FullGraph => meta.leaf_node_id.clone(),
            SessionReadScope::ActivePath { leaf_node_id } => {
                leaf_node_id.clone().or_else(|| meta.leaf_node_id.clone())
            }
        };
        let mut graph = match scope {
            SessionReadScope::FullGraph => {
                Self::load_session_graph_from_conn(&conn, meta.leaf_node_id.clone())
            }
            SessionReadScope::ActivePath { .. } => {
                Self::load_active_path_session_graph_from_conn(&conn, leaf_node_id.clone())
                    .map_err(sqlite_error)?
            }
        };
        graph.set_leaf_node_id(leaf_node_id);
        let checkpoint = meta
            .checkpoint_ref
            .as_ref()
            .and_then(|blob_ref| Self::get_checkpoint_conn(&conn, blob_ref));
        Ok(Some(PersistedSessionRead {
            session_id: meta.session_id,
            head_revision: meta.head_revision,
            config: meta.config,
            graph,
            checkpoint_ref: meta.checkpoint_ref,
            checkpoint,
            token_ledger: merge_token_ledger_entries(Self::load_usage_deltas_conn(&conn)),
        }))
    }

    async fn load_node(
        &self,
        node_id: &str,
    ) -> Result<Option<lash_core::SessionNodeRecord>, StoreError> {
        let conn = self.conn.lock().unwrap();
        let row: Option<String> = conn
            .query_row(
                "SELECT node_json FROM graph_nodes WHERE node_id = ?1 AND tombstoned = 0",
                params![node_id],
                |row| row.get(0),
            )
            .optional()
            .map_err(sqlite_error)?;
        Ok(row.and_then(|json| serde_json::from_str(&json).ok()))
    }

    async fn commit_runtime_state(
        &self,
        commit: RuntimeCommit,
    ) -> Result<RuntimeCommitResult, StoreError> {
        let result = {
            let mut conn = self.conn.lock().unwrap();
            let tx = conn.transaction().map_err(sqlite_error)?;
            let existing = load_session_head_meta_from_conn(&tx);
            if let Some(bound_session_id) = existing.as_ref().map(|meta| meta.session_id.as_str())
                && bound_session_id != commit.session_id
            {
                return Err(StoreError::SessionBindingMismatch {
                    bound_session_id: bound_session_id.to_string(),
                    attempted_session_id: commit.session_id,
                });
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
            if let Some(completed) = &commit.completed_turn {
                if completed.session_id != commit.session_id {
                    return Err(StoreError::RuntimeTurnLeaseExpired {
                        session_id: completed.session_id.clone(),
                        turn_id: completed.turn_id.clone(),
                    });
                }
                ensure_runtime_turn_completion_conn(&tx, completed)?;
            }

            let stored_checkpoint =
                Self::put_checkpoint_conn(&tx, &commit.checkpoint, self.options.blob_profile)
                    .map_err(sqlite_error)?;

            if !commit.usage_deltas.is_empty() {
                {
                    let mut stmt = tx
                        .prepare(
                            "INSERT INTO usage_deltas (
                                source, model, input_tokens, output_tokens, cached_input_tokens, reasoning_tokens
                            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                        )
                        .map_err(sqlite_error)?;
                    for entry in &commit.usage_deltas {
                        stmt.execute(params![
                            entry.source,
                            entry.model,
                            entry.usage.input_tokens,
                            entry.usage.output_tokens,
                            entry.usage.cached_input_tokens,
                            entry.usage.reasoning_tokens,
                        ])
                        .map_err(sqlite_error)?;
                    }
                }
            }

            let leaf_node_id = match &commit.graph {
                GraphCommitDelta::Unchanged { leaf_node_id } => leaf_node_id.clone(),
                GraphCommitDelta::Append {
                    nodes,
                    leaf_node_id,
                } => {
                    for node in nodes {
                        let node_json = encode_json(node);
                        tx.execute(
                            "INSERT INTO graph_nodes (node_id, node_json) VALUES (?1, ?2)",
                            params![node.node_id, node_json],
                        )
                        .map_err(sqlite_error)?;
                    }
                    leaf_node_id.clone()
                }
                GraphCommitDelta::ReplaceFull(graph) => {
                    tx.execute("DELETE FROM graph_nodes", [])
                        .map_err(sqlite_error)?;
                    for node in &graph.nodes {
                        let node_json = encode_json(node);
                        tx.execute(
                            "INSERT INTO graph_nodes (node_id, node_json) VALUES (?1, ?2)",
                            params![node.node_id, node_json],
                        )
                        .map_err(sqlite_error)?;
                    }
                    graph.leaf_node_id.clone()
                }
            };
            let graph_node_count: usize = tx
                .query_row(
                    "SELECT COUNT(*) FROM graph_nodes WHERE tombstoned = 0",
                    [],
                    |row| row.get::<_, i64>(0),
                )
                .map_err(sqlite_error)? as usize;
            let next_revision = actual_revision + 1;
            let meta = SessionHeadMeta {
                session_id: commit.session_id.clone(),
                head_revision: next_revision,
                config: commit.config,
                checkpoint_ref: Some(stored_checkpoint.checkpoint_ref.clone()),
                leaf_node_id,
                graph_node_count,
                token_ledger: Vec::new(),
            };
            tx.execute(
                "INSERT OR REPLACE INTO session_head (singleton, session_id, head_json, head_revision)
                 VALUES (1, ?1, ?2, ?3)",
                params![
                    meta.session_id,
                    encode_json(&meta),
                    meta.head_revision as i64
                ],
            )
            .map_err(sqlite_error)?;
            if let Some(completed) = &commit.completed_turn {
                tx.execute(
                    "DELETE FROM runtime_effect_journal
                     WHERE session_id = ?1 AND turn_id = ?2
                       AND EXISTS (
                         SELECT 1 FROM runtime_turn_checkpoints
                         WHERE session_id = ?1 AND turn_id = ?2 AND lease_token = ?3
                       )",
                    params![
                        completed.session_id,
                        completed.turn_id,
                        completed.lease_token
                    ],
                )
                .map_err(sqlite_error)?;
                tx.execute(
                    "DELETE FROM runtime_turn_checkpoints
                     WHERE session_id = ?1 AND turn_id = ?2 AND lease_token = ?3",
                    params![
                        completed.session_id,
                        completed.turn_id,
                        completed.lease_token
                    ],
                )
                .map_err(sqlite_error)?;
            }
            if !commit.committed_attachment_ids.is_empty() {
                let now = current_epoch_ms() as i64;
                let mut stmt = tx
                    .prepare(
                        "UPDATE attachment_manifest
                         SET committed_at_ms = COALESCE(committed_at_ms, ?1)
                         WHERE attachment_id = ?2 AND session_id = ?3",
                    )
                    .map_err(sqlite_error)?;
                for id in &commit.committed_attachment_ids {
                    stmt.execute(params![now, id.as_str(), commit.session_id])
                        .map_err(sqlite_error)?;
                }
            }
            tx.commit().map_err(sqlite_error)?;
            RuntimeCommitResult {
                head_revision: next_revision,
                checkpoint_ref: stored_checkpoint.checkpoint_ref,
                manifest: stored_checkpoint.manifest,
            }
        };
        self.maybe_auto_gc();
        Ok(result)
    }

    async fn claim_runtime_turn_lease(
        &self,
        session_id: &str,
        turn_id: &str,
        owner_id: &str,
        lease_ttl_ms: u64,
    ) -> Result<RuntimeTurnLease, StoreError> {
        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction().map_err(sqlite_error)?;
        let now = current_epoch_ms();
        let current =
            load_runtime_turn_lease_from_conn(&tx, session_id, turn_id).map_err(sqlite_error)?;
        if let Some(current) = current
            && current.expires_at_epoch_ms > now
            && current.owner_id != owner_id
        {
            return Err(StoreError::RuntimeTurnLeaseConflict {
                session_id: session_id.to_string(),
                turn_id: turn_id.to_string(),
                owner_id: current.owner_id,
                expires_at_epoch_ms: current.expires_at_epoch_ms,
            });
        }
        let fencing_token: u64 = tx
            .query_row(
                "SELECT lease_fencing_token FROM runtime_turn_checkpoints
                 WHERE session_id = ?1 AND turn_id = ?2",
                params![session_id, turn_id],
                |row| row.get::<_, i64>(0),
            )
            .optional()
            .map_err(sqlite_error)?
            .unwrap_or(0) as u64
            + 1;
        let lease = RuntimeTurnLease {
            schema_version: RUNTIME_TURN_LEASE_SCHEMA_VERSION,
            session_id: session_id.to_string(),
            turn_id: turn_id.to_string(),
            owner_id: owner_id.to_string(),
            lease_token: format!(
                "{:x}",
                Sha256::digest(
                    format!("{session_id}:{turn_id}:{owner_id}:{now}:{fencing_token}").as_bytes()
                )
            ),
            fencing_token,
            claimed_at_epoch_ms: now,
            expires_at_epoch_ms: now.saturating_add(lease_ttl_ms),
        };
        tx.execute(
            "INSERT INTO runtime_turn_checkpoints (
                session_id, turn_id, lease_owner_id, lease_token, lease_fencing_token,
                lease_claimed_at_ms, lease_expires_at_ms, updated_at_ms
             )
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
             ON CONFLICT(session_id, turn_id) DO UPDATE SET
                lease_owner_id = excluded.lease_owner_id,
                lease_token = excluded.lease_token,
                lease_fencing_token = excluded.lease_fencing_token,
                lease_claimed_at_ms = excluded.lease_claimed_at_ms,
                lease_expires_at_ms = excluded.lease_expires_at_ms,
                updated_at_ms = excluded.updated_at_ms",
            params![
                lease.session_id,
                lease.turn_id,
                lease.owner_id,
                lease.lease_token,
                lease.fencing_token as i64,
                lease.claimed_at_epoch_ms as i64,
                lease.expires_at_epoch_ms as i64,
                now as i64
            ],
        )
        .map_err(sqlite_error)?;
        tx.commit().map_err(sqlite_error)?;
        Ok(lease)
    }

    async fn renew_runtime_turn_lease(
        &self,
        lease: &RuntimeTurnLease,
        lease_ttl_ms: u64,
    ) -> Result<RuntimeTurnLease, StoreError> {
        let conn = self.conn.lock().unwrap();
        ensure_runtime_turn_lease_conn(&conn, lease)?;
        let now = current_epoch_ms();
        let renewed = RuntimeTurnLease {
            expires_at_epoch_ms: now.saturating_add(lease_ttl_ms),
            ..lease.clone()
        };
        conn.execute(
            "UPDATE runtime_turn_checkpoints
             SET lease_expires_at_ms = ?3
             WHERE session_id = ?1 AND turn_id = ?2 AND lease_token = ?4",
            params![
                renewed.session_id,
                renewed.turn_id,
                renewed.expires_at_epoch_ms as i64,
                renewed.lease_token
            ],
        )
        .map_err(sqlite_error)?;
        Ok(renewed)
    }

    async fn abandon_runtime_turn_lease(&self, lease: &RuntimeTurnLease) -> Result<(), StoreError> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "UPDATE runtime_turn_checkpoints
             SET lease_owner_id = NULL,
                 lease_token = NULL,
                 lease_claimed_at_ms = 0,
                 lease_expires_at_ms = 0,
                 updated_at_ms = ?6
             WHERE session_id = ?1
               AND turn_id = ?2
               AND lease_owner_id = ?3
               AND lease_token = ?4
               AND lease_fencing_token = ?5",
            params![
                lease.session_id,
                lease.turn_id,
                lease.owner_id,
                lease.lease_token,
                lease.fencing_token as i64,
                current_epoch_ms() as i64
            ],
        )
        .map_err(sqlite_error)?;
        Ok(())
    }

    async fn save_runtime_turn_checkpoint(
        &self,
        lease: &RuntimeTurnLease,
        checkpoint: RuntimeTurnCheckpoint,
    ) -> Result<(), StoreError> {
        if checkpoint.session_id != lease.session_id || checkpoint.turn_id != lease.turn_id {
            return Err(StoreError::RuntimeTurnLeaseExpired {
                session_id: checkpoint.session_id,
                turn_id: checkpoint.turn_id,
            });
        }
        let conn = self.conn.lock().unwrap();
        ensure_runtime_turn_lease_conn(&conn, lease)?;
        let actual_hash = lash_core::runtime_turn_checkpoint_hash(&checkpoint.checkpoint)?;
        if actual_hash != checkpoint.checkpoint_hash {
            return Err(StoreError::RuntimeTurnCheckpointHashMismatch {
                session_id: checkpoint.session_id,
                turn_id: checkpoint.turn_id,
            });
        }
        let checkpoint_json = encode_json(&checkpoint);
        conn.execute(
            "UPDATE runtime_turn_checkpoints
             SET checkpoint_json = ?3, checkpoint_hash = ?4, updated_at_ms = ?5
             WHERE session_id = ?1 AND turn_id = ?2 AND lease_token = ?6",
            params![
                checkpoint.session_id,
                checkpoint.turn_id,
                checkpoint_json,
                checkpoint.checkpoint_hash,
                checkpoint.updated_at_epoch_ms as i64,
                lease.lease_token
            ],
        )
        .map_err(sqlite_error)?;
        Ok(())
    }

    async fn load_runtime_turn_checkpoint(
        &self,
        session_id: &str,
        turn_id: &str,
    ) -> Result<Option<RuntimeTurnCheckpoint>, StoreError> {
        let conn = self.conn.lock().unwrap();
        let checkpoint_json: Option<String> = conn
            .query_row(
                "SELECT checkpoint_json FROM runtime_turn_checkpoints
                 WHERE session_id = ?1 AND turn_id = ?2 AND checkpoint_json IS NOT NULL",
                params![session_id, turn_id],
                |row| row.get(0),
            )
            .optional()
            .map_err(sqlite_error)?;
        let Some(checkpoint_json) = checkpoint_json else {
            return Ok(None);
        };
        let checkpoint: RuntimeTurnCheckpoint =
            serde_json::from_str(&checkpoint_json).map_err(|err| {
                StoreError::Backend(format!("failed to decode runtime turn checkpoint: {err}"))
            })?;
        ensure_supported_schema_version(
            "RuntimeTurnCheckpoint",
            checkpoint.schema_version,
            RUNTIME_TURN_CHECKPOINT_SCHEMA_VERSION,
        )?;
        let actual_hash = lash_core::runtime_turn_checkpoint_hash(&checkpoint.checkpoint)?;
        if checkpoint.checkpoint_hash != actual_hash {
            return Err(StoreError::RuntimeTurnCheckpointHashMismatch {
                session_id: session_id.to_string(),
                turn_id: turn_id.to_string(),
            });
        }
        Ok(Some(checkpoint))
    }

    async fn save_runtime_effect_outcome(
        &self,
        lease: &RuntimeTurnLease,
        record: RuntimeEffectJournalRecord,
    ) -> Result<(), StoreError> {
        if record.session_id != lease.session_id || record.turn_id != lease.turn_id {
            return Err(StoreError::RuntimeTurnLeaseExpired {
                session_id: record.session_id,
                turn_id: record.turn_id,
            });
        }
        let conn = self.conn.lock().unwrap();
        ensure_runtime_turn_lease_conn(&conn, lease)?;
        conn.execute(
            "INSERT INTO runtime_effect_journal (
                session_id, turn_id, idempotency_key, envelope_hash, effect_kind,
                outcome_json, created_at_ms
             )
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
             ON CONFLICT(session_id, turn_id, idempotency_key) DO UPDATE SET
                envelope_hash = excluded.envelope_hash,
                effect_kind = excluded.effect_kind,
                outcome_json = excluded.outcome_json,
                created_at_ms = excluded.created_at_ms",
            params![
                record.session_id,
                record.turn_id,
                record.idempotency_key,
                record.envelope_hash,
                record.effect_kind.as_str(),
                encode_json(&record),
                record.created_at_epoch_ms as i64
            ],
        )
        .map_err(sqlite_error)?;
        Ok(())
    }

    async fn load_runtime_effect_outcome(
        &self,
        session_id: &str,
        turn_id: &str,
        idempotency_key: &str,
    ) -> Result<Option<RuntimeEffectJournalRecord>, StoreError> {
        let conn = self.conn.lock().unwrap();
        let row: Option<String> = conn
            .query_row(
                "SELECT outcome_json FROM runtime_effect_journal
                 WHERE session_id = ?1 AND turn_id = ?2 AND idempotency_key = ?3",
                params![session_id, turn_id, idempotency_key],
                |row| row.get(0),
            )
            .optional()
            .map_err(sqlite_error)?;
        let Some(json) = row else {
            return Ok(None);
        };
        let record: RuntimeEffectJournalRecord = serde_json::from_str(&json).map_err(|err| {
            StoreError::Backend(format!("failed to decode runtime effect journal: {err}"))
        })?;
        ensure_supported_schema_version(
            "RuntimeEffectJournalRecord",
            record.schema_version,
            RUNTIME_EFFECT_JOURNAL_SCHEMA_VERSION,
        )?;
        Ok(Some(record))
    }

    async fn save_session_meta(&self, meta: SessionMeta) -> Result<(), StoreError> {
        let conn = self.conn.lock().unwrap();
        let relation_json = serde_json::to_string(&meta.relation)
            .map_err(|err| StoreError::Backend(err.to_string()))?;
        conn.execute(
            "INSERT OR REPLACE INTO session_meta
             (singleton, session_id, session_name, created_at, model, cwd, relation_json)
             VALUES (1, ?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                meta.session_id,
                meta.session_name,
                meta.created_at,
                meta.model,
                meta.cwd,
                relation_json
            ],
        )
        .map_err(sqlite_error)?;
        Ok(())
    }

    async fn load_session_meta(&self) -> Result<Option<SessionMeta>, StoreError> {
        let conn = self.conn.lock().unwrap();
        Ok(load_session_meta_from_conn(&conn))
    }

    async fn tombstone_nodes(&self, ids: &[String]) -> Result<(), StoreError> {
        if ids.is_empty() {
            return Ok(());
        }
        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction().map_err(sqlite_error)?;
        for id in ids {
            tx.execute(
                "UPDATE graph_nodes SET tombstoned = 1 WHERE node_id = ?1",
                params![id],
            )
            .map_err(sqlite_error)?;
        }
        tx.commit().map_err(sqlite_error)?;
        Ok(())
    }

    async fn vacuum(&self) -> Result<VacuumReport, StoreError> {
        let conn = self.conn.lock().unwrap();
        let removed = conn
            .execute("DELETE FROM graph_nodes WHERE tombstoned = 1", [])
            .map_err(sqlite_error)?;
        Ok(VacuumReport {
            removed_node_count: removed,
        })
    }

    async fn gc_unreachable(&self) -> Result<GcReport, StoreError> {
        Ok(Self::gc_unreachable(self))
    }
}

impl AttachmentManifest for Store {
    fn record_intent(&self, intent: AttachmentIntent) -> Result<(), StoreError> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO attachment_manifest
                (attachment_id, session_id, canonical_uri, intent_at_ms, committed_at_ms)
             VALUES (?1, ?2, ?3, ?4, NULL)
             ON CONFLICT(attachment_id) DO NOTHING",
            params![
                intent.attachment_id.as_str(),
                intent.session_id,
                intent.canonical_uri,
                intent.intent_at_epoch_ms as i64,
            ],
        )
        .map_err(sqlite_error)?;
        Ok(())
    }

    fn commit_refs(
        &self,
        session_id: &str,
        attachment_ids: &[AttachmentId],
    ) -> Result<(), StoreError> {
        if attachment_ids.is_empty() {
            return Ok(());
        }
        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction().map_err(sqlite_error)?;
        let now = current_epoch_ms() as i64;
        {
            let mut stmt = tx
                .prepare(
                    "UPDATE attachment_manifest
                     SET committed_at_ms = COALESCE(committed_at_ms, ?1)
                     WHERE attachment_id = ?2 AND session_id = ?3",
                )
                .map_err(sqlite_error)?;
            for id in attachment_ids {
                stmt.execute(params![now, id.as_str(), session_id])
                    .map_err(sqlite_error)?;
            }
        }
        tx.commit().map_err(sqlite_error)?;
        Ok(())
    }

    fn list_uncommitted(
        &self,
        older_than_epoch_ms: u64,
    ) -> Result<Vec<AttachmentManifestEntry>, StoreError> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn
            .prepare(
                "SELECT attachment_id, session_id, canonical_uri, intent_at_ms, committed_at_ms
                 FROM attachment_manifest
                 WHERE committed_at_ms IS NULL AND intent_at_ms <= ?1
                 ORDER BY intent_at_ms ASC",
            )
            .map_err(sqlite_error)?;
        let rows = stmt
            .query_map(params![older_than_epoch_ms as i64], |row| {
                let id: String = row.get(0)?;
                let session_id: String = row.get(1)?;
                let canonical_uri: String = row.get(2)?;
                let intent_at_ms: i64 = row.get(3)?;
                let committed_at_ms: Option<i64> = row.get(4)?;
                Ok(AttachmentManifestEntry {
                    attachment_id: AttachmentId::new(id),
                    session_id,
                    canonical_uri,
                    intent_at_epoch_ms: intent_at_ms as u64,
                    committed_at_epoch_ms: committed_at_ms.map(|v| v as u64),
                })
            })
            .map_err(sqlite_error)?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row.map_err(sqlite_error)?);
        }
        Ok(out)
    }

    fn forget(&self, attachment_id: &AttachmentId) -> Result<(), StoreError> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "DELETE FROM attachment_manifest WHERE attachment_id = ?1",
            params![attachment_id.as_str()],
        )
        .map_err(sqlite_error)?;
        Ok(())
    }
}

/// Initialize or verify the SQLite schema.
///
/// Lash's session store is a clean-cutover backend: there is exactly one
/// supported schema, and there is no in-flight migration code.
///
/// - **Fresh database** (`user_version = 0` with no user-defined objects):
///   apply [`SCHEMA`] and stamp `user_version` to [`SCHEMA_VERSION`].
/// - **Already-current database**: ensure idempotent `CREATE TABLE IF NOT EXISTS`
///   statements have run, then return.
/// - **Anything else** (a database from a prior schema version or a foreign
///   schema): fail with a clear error directing the host to delete the file.
fn ensure_schema(conn: &Connection) -> rusqlite::Result<()> {
    let user_version: i32 = conn.query_row("PRAGMA user_version", [], |row| row.get(0))?;
    if user_version == SCHEMA_VERSION {
        conn.execute_batch(SCHEMA)?;
        return Ok(());
    }

    if user_version == 0 && !has_user_schema_objects(conn)? {
        conn.execute_batch(SCHEMA)?;
        conn.pragma_update(None, "user_version", SCHEMA_VERSION)?;
        return Ok(());
    }

    Err(rusqlite::Error::InvalidParameterName(
        unsupported_schema_message(),
    ))
}

fn has_user_schema_objects(conn: &Connection) -> rusqlite::Result<bool> {
    let count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM sqlite_master
         WHERE name NOT LIKE 'sqlite_%'
           AND type IN ('table', 'index', 'trigger', 'view')",
        [],
        |row| row.get(0),
    )?;
    Ok(count > 0)
}

fn unsupported_schema_message() -> String {
    "Unsupported lash session schema. This binary supports schema version 1 only; \
     older databases must be deleted before opening. Delete the session database \
     and start fresh."
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use lash_core::{
        ProcessEventSemanticsSpec, ProcessInput, ProcessTerminalState, ProcessValueSelector,
        ProcessWakeDedupeKey, ProcessWakeSpec,
    };

    fn registration(id: &str) -> ProcessRegistration {
        ProcessRegistration::new(
            id,
            ProcessInput::External {
                metadata: serde_json::Value::Null,
            },
        )
        .with_provenance(
            lash_core::ProcessCreatorScope::new("runtime", "session"),
            "test-host",
        )
    }

    fn monitor_line_event_type() -> lash_core::ProcessEventType {
        let mut properties = serde_json::Map::new();
        properties.insert("line".to_string(), serde_json::json!({ "type": "string" }));
        properties.insert(
            "wake_input".to_string(),
            serde_json::json!({ "type": "string" }),
        );
        lash_core::ProcessEventType {
            name: "monitor.line".to_string(),
            payload_schema: lash_core::LashSchema::object(properties, vec!["line".to_string()]),
            semantics: ProcessEventSemanticsSpec {
                wake: Some(ProcessWakeSpec {
                    when: Some(ProcessValueSelector::Present("/wake_input".to_string())),
                    input: ProcessValueSelector::Pointer("/wake_input".to_string()),
                    dedupe_key: ProcessWakeDedupeKey::Selector(ProcessValueSelector::Pointer(
                        "/line".to_string(),
                    )),
                }),
                ..ProcessEventSemanticsSpec::default()
            },
        }
    }

    #[tokio::test]
    async fn sqlite_process_registry_persists_rows_after_reopen() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("processes.db");
        {
            let registry = SqliteProcessRegistry::open(&path).expect("open registry");
            registry
                .register_process(registration("proc-persist"))
                .await
                .expect("register");
            registry
                .grant_handle(
                    "runtime:session",
                    "proc-persist",
                    ProcessHandleDescriptor::new(Some("tool"), Some("demo")),
                )
                .await
                .expect("grant");
            registry
                .complete_process(
                    "proc-persist",
                    ProcessAwaitOutput::Success {
                        value: serde_json::json!({"ok": true}),
                        control: None,
                    },
                )
                .await
                .expect("complete");
        }

        let registry = SqliteProcessRegistry::open(&path).expect("reopen registry");
        let record = registry
            .get_process("proc-persist")
            .await
            .expect("persisted process");

        assert_eq!(record.created_by_scope_key, "runtime:session");
        assert_eq!(
            record
                .created_by_scope
                .as_ref()
                .map(|scope| scope.session_id.as_str()),
            Some("session")
        );
        assert_eq!(
            registry
                .await_process("proc-persist")
                .await
                .expect("await persisted"),
            ProcessAwaitOutput::Success {
                value: serde_json::json!({"ok": true}),
                control: None,
            }
        );
        assert_eq!(
            registry
                .list_handle_grants("runtime:session")
                .await
                .expect("grants")
                .len(),
            1
        );
    }

    #[tokio::test]
    async fn sqlite_process_registration_is_idempotent_and_hash_conflicts_fail() {
        let registry = SqliteProcessRegistry::memory().expect("registry");
        let first = registry
            .register_process(registration("proc-idempotent"))
            .await
            .expect("first register");
        let second = registry
            .register_process(registration("proc-idempotent"))
            .await
            .expect("replay register");

        assert_eq!(first.registration_hash, second.registration_hash);
        assert!(
            registry
                .register_process(
                    registration("proc-idempotent")
                        .with_extra_event_types([monitor_line_event_type()]),
                )
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn sqlite_process_keyed_events_and_wake_inbox_are_durable() {
        let registry = SqliteProcessRegistry::memory().expect("registry");
        registry
            .register_process(
                registration("proc-wake").with_extra_event_types([monitor_line_event_type()]),
            )
            .await
            .expect("register");
        let request = ProcessEventAppendRequest::new(
            "monitor.line",
            serde_json::json!({
                "line": "deploy failed",
                "wake_input": "Monitor event: deploy failed",
            }),
        )
        .with_idempotency_key("line:deploy failed")
        .with_wake_target_scope_key("runtime:session");

        let first = registry
            .append_event("proc-wake", request.clone())
            .await
            .expect("append");
        let second = registry
            .append_event("proc-wake", request)
            .await
            .expect("replay append");

        assert_eq!(first.sequence, second.sequence);
        assert!(
            registry
                .append_event(
                    "proc-wake",
                    ProcessEventAppendRequest::new(
                        "monitor.line",
                        serde_json::json!({
                            "line": "other",
                            "wake_input": "Monitor event: other",
                        }),
                    )
                    .with_idempotency_key("line:deploy failed"),
                )
                .await
                .is_err()
        );

        let wakes = registry
            .drain_wake_inputs("runtime:session", 10)
            .await
            .expect("drain wakes");
        assert_eq!(wakes.len(), 1);
        assert_eq!(wakes[0].input, "Monitor event: deploy failed");
        registry
            .ack_wake_input(&wakes[0].wake_id)
            .await
            .expect("ack wake");
        assert!(
            registry
                .drain_wake_inputs("runtime:session", 10)
                .await
                .expect("drain after ack")
                .is_empty()
        );
    }

    #[tokio::test]
    async fn sqlite_process_terminal_and_cancel_events_require_keys() {
        let registry = SqliteProcessRegistry::memory().expect("registry");
        registry
            .register_process(registration("proc-terminal"))
            .await
            .expect("register");

        assert!(
            registry
                .append_event(
                    "proc-terminal",
                    ProcessEventAppendRequest::new(
                        "process.cancel_requested",
                        serde_json::json!({"reason": "stop"}),
                    ),
                )
                .await
                .is_err()
        );
        registry
            .append_event(
                "proc-terminal",
                ProcessEventAppendRequest::cancel_requested(
                    "proc-terminal",
                    Some("stop".to_string()),
                ),
            )
            .await
            .expect("cancel intent");
        registry
            .complete_process(
                "proc-terminal",
                ProcessAwaitOutput::Cancelled {
                    message: "stopped".to_string(),
                    raw: None,
                    control: None,
                },
            )
            .await
            .expect("complete cancelled");
        assert_eq!(
            registry
                .get_process("proc-terminal")
                .await
                .and_then(|record| record.terminal.map(|terminal| terminal.state)),
            Some(ProcessTerminalState::Cancelled)
        );
    }
}
