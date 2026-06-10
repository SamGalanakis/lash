//! # lash-sqlite-store
//!
//! The high-performance local **durable** persistence backend for the lash
//! agent runtime. One SQLite database per session, opened in WAL journal mode
//! with a 15-second busy timeout, satisfying the full [`RuntimePersistence`] +
//! [`AttachmentManifest`] contract from `lash-core`.
//!
//! This crate is a drop-in replacement for `lash-sqlite-store`: it exposes the
//! same public surface (`Store`, `SqliteProcessRegistry`,
//! `SqliteSessionStoreFactory`, `SqliteEffectHost`, the option/descriptor types)
//! with identical async signatures, so a consumer swaps backends by renaming
//! the crate path only. The difference is the engine underneath: tokio-rusqlite
//! over a statically-linked SQLite with real WAL (`-wal`/`-shm` sidecars,
//! multi-process readers + single writer) instead of the prior store's experimental mvcc.
//!
//! ## Why this is "the durable backend" not just "an option"
//!
//! Lash's runtime layer treats persistence as a first-class boundary, not a
//! debug-only convenience. Every primitive that lets the runtime survive a
//! crash — head-revision CAS, final turn-commit idempotency, attachment
//! write-ahead manifests, blob content-addressing with optional compression —
//! is implemented in this crate against SQLite for one reason: SQLite is the
//! simplest backend that gives us *atomic multi-statement transactions on a
//! single file* with durability guarantees we can reason about.
//!
//! ## Schema cutover, not migrations
//!
//! There is exactly one supported schema (see [`schema::SCHEMA`]). Older
//! databases must be deleted before opening — we do not carry migration code.
//!
//! [`RuntimePersistence`]: lash_core::RuntimePersistence
//! [`AttachmentManifest`]: lash_core::AttachmentManifest

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use flate2::Compression;
use flate2::read::ZlibDecoder;
use flate2::write::ZlibEncoder;
use lash_core::runtime::ProcessHandleGrantEntry;
use lash_core::runtime::{
    QueuedWorkBatch, QueuedWorkBatchDraft, QueuedWorkClaim, QueuedWorkClaimBoundary,
    QueuedWorkCompletion, QueuedWorkItem, QueuedWorkPayload, prepare_process_event_append,
    prepare_process_registration,
};
use lash_core::store::queued_work::{
    ClaimCandidate, QueuedWorkClaimLease, claim_scan_limit, derive_batch_id,
    ensure_completion_owns_all_batches, renewed_claim, select_claim_prefix,
};
use lash_core::store::{
    GraphCommitDelta, HydratedSessionCheckpoint, PersistedSessionRead, RuntimeCommit,
    RuntimeCommitResult, SessionCheckpoint, SessionHead, SessionHeadMeta,
};
use lash_core::{
    AttachmentId, AttachmentIntent, AttachmentManifest, AttachmentManifestEntry, BlobRef,
    DeliveryPolicy, DurabilityTier, GcReport, MergeKey, PROCESS_LEASE_SCHEMA_VERSION,
    ProcessAwaitOutput, ProcessEvent, ProcessEventAppendRequest, ProcessEventAppendResult,
    ProcessExternalRef, ProcessHandleDescriptor, ProcessHandleGrant, ProcessLease,
    ProcessLeaseCompletion, ProcessRecord, ProcessRegistration, ProcessRegistry,
    RuntimePersistence, SessionMeta, SessionPickerInfo, SessionReadScope, SessionScope,
    SessionStoreCreateRequest, SessionStoreFactory, SlotPolicy, StoreError, VacuumReport,
};
use rusqlite::{Connection, OptionalExtension, Transaction, params};
use sha2::{Digest, Sha256};

use conn::SqliteConnection;

mod attachments;
mod blobs;
mod conn;
mod effect_replay;
mod graph;
mod host_events;
mod leases;
mod lifecycle;
mod persistence;
mod process_registry;
mod queued_work;
mod schema;

use conn::TxOutcome;
pub use effect_replay::{
    SqliteEffectHost, SqliteEffectReplayOptions, SqliteRuntimeEffectController,
};
pub use host_events::SqliteHostEventStore;
use leases::*;
use queued_work::*;
use schema::{
    StoreBacking, apply_pragmas, ensure_effect_schema, ensure_host_event_schema,
    ensure_process_schema, ensure_schema,
};

/// SQLite-backed store for checkpoint blobs and the canonical session head.
///
/// The struct name and every public method match `lash_sqlite_store::Store`
/// exactly so consumers swap backends with a path rename. Internally it holds a
/// single cloneable [`SqliteConnection`] (a tokio-rusqlite handle to one
/// database thread) rather than the prior store's `tokio::sync::Mutex<rusqlite::Connection>`.
pub struct Store {
    conn: SqliteConnection,
    artifact_cache: Mutex<BTreeMap<lashlang::ModuleRef, Arc<lashlang::ModuleArtifact>>>,
    options: StoreOptions,
    commit_count: AtomicU64,
}

/// SQLite-backed process registry for one configured runtime deployment.
///
/// Named `SqliteProcessRegistry` so the path-rename swap keeps compiling; this is
/// the SQLite implementation. It is intentionally separate from [`Store`]:
/// session databases persist one conversation, while this registry persists
/// background process state and handle visibility across all sessions in the
/// same host profile.
pub struct SqliteProcessRegistry {
    conn: SqliteConnection,
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

fn block_on_store<T>(future: impl std::future::Future<Output = T>) -> T {
    futures_executor::block_on(future)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum PersistedArtifactKind {
    GenericBlob,
    CheckpointManifest,
    ToolState,
    PluginSessionSnapshot,
    ExecutionStateSnapshot,
    LashlangModule,
    ProcessExecutionEnv,
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

    pub fn lashlang_module() -> Self {
        Self::new(
            PersistedArtifactKind::LashlangModule,
            vec![BlobStorageHint::Compressible, BlobStorageHint::LargePayload],
        )
    }

    pub fn process_execution_env() -> Self {
        Self::new(
            PersistedArtifactKind::ProcessExecutionEnv,
            vec![BlobStorageHint::Compressible],
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

/// Explicit first-party factory for one SQLite session database per Lash
/// session.
///
/// Named `SqliteSessionStoreFactory` so the path-rename swap keeps compiling.
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

#[async_trait::async_trait]
impl SessionStoreFactory for SqliteSessionStoreFactory {
    fn durability_tier(&self) -> DurabilityTier {
        DurabilityTier::Durable
    }

    async fn create_store(
        &self,
        request: &SessionStoreCreateRequest,
    ) -> Result<Arc<dyn RuntimePersistence>, String> {
        std::fs::create_dir_all(&self.root).map_err(|err| err.to_string())?;
        let path = self.path_for_session(&request.session_id);
        let store = Arc::new(
            Store::open_with_options(&path, self.options)
                .await
                .map_err(|err| err.to_string())?,
        );
        if store.load_session_meta().await.is_none() {
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
                .await;
        }
        Ok(store as Arc<dyn RuntimePersistence>)
    }

    async fn open_existing_store(
        &self,
        request: &SessionStoreCreateRequest,
    ) -> Result<Option<Arc<dyn RuntimePersistence>>, String> {
        let path = self.path_for_session(&request.session_id);
        if !path.exists() {
            return Ok(None);
        }
        self.create_store(request).await.map(Some)
    }

    async fn delete_session(&self, session_id: &str) -> Result<(), String> {
        let db_path = self.path_for_session(session_id);
        for path in [
            db_path.clone(),
            sqlite_sidecar_path(&db_path, "-wal"),
            sqlite_sidecar_path(&db_path, "-shm"),
        ] {
            match std::fs::remove_file(&path) {
                Ok(()) => {}
                Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
                Err(err) => {
                    return Err(format!("remove session store {}: {err}", path.display()));
                }
            }
        }
        Ok(())
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

fn sqlite_sidecar_path(path: &Path, suffix: &str) -> PathBuf {
    let mut sidecar = path.as_os_str().to_os_string();
    sidecar.push(suffix);
    PathBuf::from(sidecar)
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
        agent_frames: head.agent_frames.clone(),
        current_agent_frame_id: head.current_agent_frame_id.clone(),
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

/// Read the session head meta off a raw connection. Synchronous because it runs
/// inside a `conn.call`/`conn.write` closure on the connection thread.
fn try_load_session_head_meta_from_conn(
    conn: &Connection,
) -> Result<Option<SessionHeadMeta>, StoreError> {
    let row = conn
        .query_row(
            "SELECT head_json, head_revision FROM session_head WHERE singleton = 1",
            [],
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?)),
        )
        .optional()
        .map_err(sqlite_error)?;
    let Some((head_json, head_revision)) = row else {
        return Ok(None);
    };
    let mut meta: SessionHeadMeta = serde_json::from_str(&head_json)
        .map_err(|err| StoreError::Backend(format!("decode session head: {err}")))?;
    meta.head_revision = head_revision as u64;
    Ok(Some(meta))
}

fn load_session_head_meta_from_conn(conn: &Connection) -> Option<SessionHeadMeta> {
    try_load_session_head_meta_from_conn(conn).ok().flatten()
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
    .optional()
    .ok()
    .flatten()
}

fn decode_checkpoint(bytes: &[u8]) -> Option<SessionCheckpoint> {
    rmp_serde::from_slice(bytes).ok()
}

fn encode_msgpack<T: serde::Serialize>(value: &T) -> Vec<u8> {
    // Pre-size the buffer so the per-byte writes inside rmp_serde don't
    // walk the Vec through 0→4→8→16→32… reallocations on every call.
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

#[cfg(test)]
mod tests {
    use super::*;
    use lash_core::ProcessInput;
    use lashlang::LashlangArtifactStore;

    fn registration(id: &str) -> ProcessRegistration {
        ProcessRegistration::new(
            id,
            ProcessInput::External {
                metadata: serde_json::Value::Null,
            },
            lash_core::ProcessProvenance::session(
                lash_core::SessionScope::new("session"),
                "test-host",
            ),
        )
    }

    #[tokio::test]
    async fn sqlite_lashlang_artifact_store_round_trips_verified_module_artifacts() {
        let store = Store::memory().await.expect("memory store");
        let module =
            lashlang::parse("process scan(root: str) { finish root }").expect("parse module");
        let linked = lashlang::LinkedModule::link(
            module,
            lashlang::LashlangSurface::new(
                lashlang::ResourceCatalog::new(),
                lashlang::LashlangAbilities::all(),
            ),
        )
        .expect("link module");

        store
            .put_module_artifact(&linked.artifact)
            .await
            .expect("put artifact");
        let restored = store
            .get_module_artifact(&linked.module_ref)
            .await
            .expect("get artifact")
            .expect("artifact exists");

        assert_eq!(restored.module_ref, linked.module_ref);
        assert_eq!(
            restored.process_ref("scan"),
            linked.artifact.process_ref("scan")
        );
    }

    #[tokio::test]
    async fn sqlite_process_registry_persists_rows_after_reopen() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("processes.db");
        {
            let registry = SqliteProcessRegistry::open(&path)
                .await
                .expect("open registry");
            let session_scope = lash_core::SessionScope::new("session");
            registry
                .register_process(registration("proc-persist"))
                .await
                .expect("register");
            registry
                .grant_handle(
                    &session_scope,
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

        let registry = SqliteProcessRegistry::open(&path)
            .await
            .expect("reopen registry");
        let session_scope = lash_core::SessionScope::new("session");
        let record = registry
            .get_process("proc-persist")
            .await
            .expect("persisted process");

        assert_eq!(record.originator_scope_id(), session_scope.id().as_str());
        assert_eq!(
            record.provenance.originator,
            lash_core::ProcessOriginator::session(session_scope.clone())
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
                .list_handle_grants(&session_scope)
                .await
                .expect("grants")
                .len(),
            1
        );
    }
}
