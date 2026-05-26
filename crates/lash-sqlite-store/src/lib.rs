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
    GraphCommitDelta, HydratedSessionCheckpoint, PersistedSessionRead,
    RUNTIME_EFFECT_JOURNAL_SCHEMA_VERSION, RUNTIME_TURN_CHECKPOINT_SCHEMA_VERSION,
    RUNTIME_TURN_LEASE_SCHEMA_VERSION, RuntimeCommit, RuntimeCommitResult,
    RuntimeEffectJournalRecord, RuntimePersistence, RuntimeTurnCheckpoint, RuntimeTurnLease,
    SessionCheckpoint, SessionHead, SessionHeadMeta, SessionMeta, SessionPickerInfo,
    SessionReadScope, SessionStoreCreateRequest, SessionStoreFactory, StoreError, VacuumReport,
    ensure_supported_schema_version,
};
use rusqlite::{Connection, OpenFlags, OptionalExtension, params};
use sha2::{Digest, Sha256};

/// SQLite-backed store for checkpoint blobs and the canonical session head.
pub struct Store {
    conn: Mutex<Connection>,
    options: StoreOptions,
    commit_count: AtomicU64,
}

fn sqlite_error(err: rusqlite::Error) -> StoreError {
    StoreError::Backend(err.to_string())
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
