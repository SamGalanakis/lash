#[cfg(feature = "sqlite-store")]
use std::path::Path;
#[cfg(feature = "sqlite-store")]
use std::sync::Mutex;
#[cfg(feature = "sqlite-store")]
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
#[cfg(feature = "sqlite-store")]
use std::time::Duration;

#[cfg(feature = "sqlite-store")]
use flate2::Compression;
#[cfg(feature = "sqlite-store")]
use flate2::read::ZlibDecoder;
#[cfg(feature = "sqlite-store")]
use flate2::write::ZlibEncoder;
#[cfg(feature = "sqlite-store")]
use rusqlite::{Connection, OpenFlags, params};
#[cfg(feature = "sqlite-store")]
use sha2::{Digest, Sha256};

fn default_root_session_id() -> String {
    "root".to_string()
}

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error(
        "store is already bound to session `{bound_session_id}` and cannot be reused for `{attempted_session_id}`"
    )]
    SessionBindingMismatch {
        bound_session_id: String,
        attempted_session_id: String,
    },
    #[error("store has inconsistent session bindings: {session_ids:?}")]
    InconsistentSessionBindings { session_ids: Vec<String> },
}

/// SQLite-backed store for checkpoint blobs, live resume state, and the canonical session head.
#[cfg(feature = "sqlite-store")]
pub struct Store {
    conn: Mutex<Connection>,
    options: StoreOptions,
    commit_count: AtomicU64,
}

#[cfg(feature = "sqlite-store")]
pub type SqliteStore = Store;

#[cfg(feature = "sqlite-store")]
const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS blobs (
    hash    TEXT PRIMARY KEY,
    content BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS live_resume (
    singleton      INTEGER PRIMARY KEY CHECK (singleton = 1),
    session_id     TEXT NOT NULL DEFAULT 'root',
    snapshot_json  TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS ui_resume_state (
    singleton      INTEGER PRIMARY KEY CHECK (singleton = 1),
    state_json     TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS session_head (
    singleton      INTEGER PRIMARY KEY CHECK (singleton = 1),
    session_id     TEXT NOT NULL DEFAULT 'root',
    head_json      TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS graph_nodes (
    seq       INTEGER PRIMARY KEY AUTOINCREMENT,
    node_id   TEXT NOT NULL UNIQUE,
    node_json TEXT NOT NULL
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
    singleton        INTEGER PRIMARY KEY CHECK (singleton = 1),
    session_id       TEXT NOT NULL,
    session_name     TEXT NOT NULL,
    created_at       TEXT NOT NULL,
    model            TEXT NOT NULL,
    cwd              TEXT,
    parent_session_id TEXT
);
";

#[cfg(feature = "sqlite-store")]
const SCHEMA_VERSION: i32 = 12;

#[cfg(feature = "sqlite-store")]
const SQLITE_BUSY_TIMEOUT: Duration = Duration::from_secs(15);
#[cfg(feature = "sqlite-store")]
const SQLITE_WAL_AUTOCHECKPOINT_PAGES: i64 = 1_000;

#[cfg(feature = "sqlite-store")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StoreBacking {
    File,
    Memory,
}

#[cfg(feature = "sqlite-store")]
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

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SessionMeta {
    pub session_id: String,
    pub session_name: String,
    pub created_at: String,
    pub model: String,
    pub cwd: Option<String>,
    pub parent_session_id: Option<String>,
}

/// Lightweight session info for the resume picker.
#[derive(Clone, Debug)]
pub struct SessionPickerInfo {
    pub session_id: String,
    pub cwd: Option<String>,
    pub parent_session_id: Option<String>,
    pub first_user_message: String,
    pub user_message_count: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(transparent)]
pub struct BlobRef(pub String);

impl BlobRef {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for BlobRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<String> for BlobRef {
    fn from(value: String) -> Self {
        Self(value)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum PersistedArtifactKind {
    GenericBlob,
    CheckpointManifest,
    DynamicStateSnapshot,
    PluginSessionSnapshot,
    LiveResumeDelta,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum BlobStorageHint {
    Compressible,
    InlinePreferred,
    LargePayload,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum BlobCompression {
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

    pub fn dynamic_state_snapshot() -> Self {
        Self::new(
            PersistedArtifactKind::DynamicStateSnapshot,
            vec![BlobStorageHint::Compressible, BlobStorageHint::LargePayload],
        )
    }

    pub fn plugin_session_snapshot() -> Self {
        Self::new(
            PersistedArtifactKind::PluginSessionSnapshot,
            vec![BlobStorageHint::Compressible, BlobStorageHint::LargePayload],
        )
    }

    pub fn live_resume_delta() -> Self {
        Self::new(
            PersistedArtifactKind::LiveResumeDelta,
            vec![BlobStorageHint::Compressible, BlobStorageHint::LargePayload],
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RetainedArtifactRef {
    pub blob_ref: BlobRef,
    pub kind: PersistedArtifactKind,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GcReport {
    pub root_count: usize,
    pub retained_blob_count: usize,
    pub deleted_blob_count: usize,
}

#[cfg(feature = "sqlite-store")]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BuiltinBlobProfile {
    LowLatency,
    #[default]
    Balanced,
    Compact,
}

#[cfg(feature = "sqlite-store")]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StoreGcPolicy {
    pub auto_run_every_commits: Option<u64>,
}

#[cfg(feature = "sqlite-store")]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StoreOptions {
    pub blob_profile: BuiltinBlobProfile,
    pub gc_policy: StoreGcPolicy,
}

#[cfg(feature = "sqlite-store")]
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct StoredBlobEnvelope {
    descriptor: BlobArtifactDescriptor,
    compression: BlobCompression,
    content: Vec<u8>,
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct SessionCheckpoint {
    #[serde(default)]
    pub turn_state: crate::PersistedTurnState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dynamic_state_ref: Option<BlobRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plugin_snapshot_ref: Option<BlobRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plugin_snapshot_revision: Option<u64>,
}

impl SessionCheckpoint {
    pub fn retained_artifact_refs(&self) -> Vec<RetainedArtifactRef> {
        let mut refs = Vec::new();
        if let Some(blob_ref) = &self.dynamic_state_ref {
            refs.push(RetainedArtifactRef {
                blob_ref: blob_ref.clone(),
                kind: PersistedArtifactKind::DynamicStateSnapshot,
            });
        }
        if let Some(blob_ref) = &self.plugin_snapshot_ref {
            refs.push(RetainedArtifactRef {
                blob_ref: blob_ref.clone(),
                kind: PersistedArtifactKind::PluginSessionSnapshot,
            });
        }
        refs
    }
}

#[derive(Clone, Debug, Default)]
pub struct HydratedSessionCheckpoint {
    pub turn_state: crate::PersistedTurnState,
    pub dynamic_state_ref: Option<BlobRef>,
    pub dynamic_state: Option<crate::DynamicStateSnapshot>,
    pub plugin_snapshot_ref: Option<BlobRef>,
    pub plugin_snapshot: Option<crate::PluginSessionSnapshot>,
    pub plugin_snapshot_revision: Option<u64>,
}

#[derive(Clone, Debug)]
pub struct StoredSessionCheckpoint {
    pub checkpoint_ref: BlobRef,
    pub manifest: SessionCheckpoint,
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct LiveResumeDelta {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub appended_graph_nodes: Vec<crate::SessionNodeRecord>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub leaf_node_id: Option<String>,
    #[serde(default)]
    pub turn_state: crate::PersistedTurnState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dynamic_state: Option<crate::DynamicStateSnapshot>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plugin_snapshot: Option<crate::PluginSessionSnapshot>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execution_state_snapshot: Option<Vec<u8>>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub token_ledger: Vec<crate::TokenLedgerEntry>,
}

impl LiveResumeDelta {
    pub fn apply_to_graph(&self, base: &crate::SessionGraph) -> crate::SessionGraph {
        let mut graph = base.clone();
        graph.extend_node_records(self.appended_graph_nodes.iter().cloned());
        if self.leaf_node_id.is_some() {
            graph.set_leaf_node_id(self.leaf_node_id.clone());
        }
        graph
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SessionHead {
    #[serde(default = "default_root_session_id")]
    pub session_id: String,
    pub graph: crate::SessionGraph,
    pub config: crate::PersistedSessionConfig,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checkpoint_ref: Option<BlobRef>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub token_ledger: Vec<crate::TokenLedgerEntry>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SessionHeadMeta {
    #[serde(default = "default_root_session_id")]
    pub session_id: String,
    pub config: crate::PersistedSessionConfig,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checkpoint_ref: Option<BlobRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub leaf_node_id: Option<String>,
    #[serde(default)]
    pub graph_node_count: usize,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub token_ledger: Vec<crate::TokenLedgerEntry>,
}

fn session_head_meta(head: &SessionHead) -> SessionHeadMeta {
    SessionHeadMeta {
        session_id: head.session_id.clone(),
        config: head.config.clone(),
        checkpoint_ref: head.checkpoint_ref.clone(),
        leaf_node_id: head.graph.leaf_node_id.clone(),
        graph_node_count: head.graph.nodes.len(),
        token_ledger: Vec::new(),
    }
}

fn persisted_session_config_from_state(
    state: &crate::PersistedSessionState,
) -> crate::PersistedSessionConfig {
    crate::PersistedSessionConfig {
        provider_id: state.policy.provider.id().to_string(),
        configured_model: state.policy.model.clone(),
        context_window: state.policy.max_context_tokens.unwrap_or_default() as u64,
        execution_mode: state.policy.execution_mode,
        context_approach: state.policy.context_approach.clone(),
        model_variant: state.policy.model_variant.clone(),
    }
}

#[derive(Clone, Debug)]
pub enum SessionGraphCommit {
    Replace(crate::SessionGraph),
    Append {
        nodes: Vec<crate::SessionNodeRecord>,
        leaf_node_id: Option<String>,
        graph_node_count: usize,
    },
}

impl SessionGraphCommit {
    pub fn graph_node_count(&self) -> usize {
        match self {
            Self::Replace(graph) => graph.nodes.len(),
            Self::Append {
                graph_node_count, ..
            } => *graph_node_count,
        }
    }

    pub fn leaf_node_id(&self) -> Option<&String> {
        match self {
            Self::Replace(graph) => graph.leaf_node_id.as_ref(),
            Self::Append { leaf_node_id, .. } => leaf_node_id.as_ref(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct PersistedStateCommit {
    pub session_id: String,
    pub config: crate::PersistedSessionConfig,
    pub graph: SessionGraphCommit,
    pub checkpoint: HydratedSessionCheckpoint,
    pub usage_deltas: Vec<crate::TokenLedgerEntry>,
}

#[derive(Clone, Debug)]
pub struct LiveResumeCommit {
    pub session_id: String,
    pub seed_graph: Option<crate::SessionGraph>,
    pub live_graph: crate::SessionGraph,
    pub config: crate::PersistedSessionConfig,
    pub turn_state: crate::PersistedTurnState,
    pub dynamic_state: Option<crate::DynamicStateSnapshot>,
    pub plugin_snapshot: Option<crate::PluginSessionSnapshot>,
    pub execution_state_snapshot: Option<Vec<u8>>,
    pub token_ledger: Vec<crate::TokenLedgerEntry>,
}

#[derive(Clone, Debug)]
pub enum RuntimeCommit {
    PersistedState(PersistedStateCommit),
    LiveResume(LiveResumeCommit),
}

#[derive(Clone, Debug)]
pub struct PersistedStateCommitResult {
    pub checkpoint_ref: BlobRef,
    pub manifest: SessionCheckpoint,
    pub persisted_graph_node_count: usize,
}

#[derive(Clone, Debug)]
pub enum RuntimeCommitResult {
    PersistedState(PersistedStateCommitResult),
    LiveResumeSaved,
    LiveResumeSkipped,
}

fn build_persisted_turn_state(state: &crate::PersistedSessionState) -> crate::PersistedTurnState {
    crate::PersistedTurnState {
        iteration: state.iteration,
        token_usage: state.token_usage.clone(),
        last_prompt_usage: state.last_prompt_usage.clone(),
    }
}

fn build_checkpoint_from_persisted_state(
    state: &crate::PersistedSessionState,
) -> HydratedSessionCheckpoint {
    HydratedSessionCheckpoint {
        turn_state: build_persisted_turn_state(state),
        dynamic_state_ref: state.dynamic_state_ref.clone(),
        dynamic_state: state.dynamic_state_snapshot.clone(),
        plugin_snapshot_ref: state.plugin_snapshot_ref.clone(),
        plugin_snapshot_revision: state.plugin_snapshot_revision,
        plugin_snapshot: state.plugin_snapshot.clone(),
    }
}

fn graph_commit_from_state(state: &crate::PersistedSessionState) -> SessionGraphCommit {
    let nodes_len = state.session_graph.nodes.len();
    if state.graph_replace_required || state.persisted_graph_node_count > nodes_len {
        SessionGraphCommit::Replace(state.session_graph.clone())
    } else {
        SessionGraphCommit::Append {
            nodes: state.session_graph.nodes[state.persisted_graph_node_count..].to_vec(),
            leaf_node_id: state.session_graph.leaf_node_id.clone(),
            graph_node_count: nodes_len,
        }
    }
}

impl RuntimeCommit {
    pub fn persisted_state(
        state: &crate::PersistedSessionState,
        usage_deltas: &[crate::TokenLedgerEntry],
    ) -> Self {
        Self::PersistedState(PersistedStateCommit {
            session_id: state.session_id.clone(),
            config: persisted_session_config_from_state(state),
            graph: graph_commit_from_state(state),
            checkpoint: build_checkpoint_from_persisted_state(state),
            usage_deltas: usage_deltas.to_vec(),
        })
    }

    pub fn live_resume(
        seed_graph: Option<crate::SessionGraph>,
        live_graph: crate::SessionGraph,
        state: &crate::PersistedSessionState,
    ) -> Option<Self> {
        if !crate::messages_are_live_resume_safe(&live_graph.project_messages()) {
            return None;
        }
        Some(Self::LiveResume(LiveResumeCommit {
            session_id: state.session_id.clone(),
            seed_graph,
            live_graph,
            config: persisted_session_config_from_state(state),
            turn_state: build_persisted_turn_state(state),
            dynamic_state: state.dynamic_state_snapshot.clone(),
            plugin_snapshot: state.plugin_snapshot.clone(),
            execution_state_snapshot: state.execution_state_snapshot.clone(),
            token_ledger: state.token_ledger.clone(),
        }))
    }
}

fn persisted_session_state_from_head(
    head: SessionHead,
    checkpoint: Option<HydratedSessionCheckpoint>,
) -> crate::PersistedSessionState {
    let mut state = crate::PersistedSessionState {
        session_id: head.session_id,
        policy: crate::SessionPolicy::default(),
        session_graph: head.graph,
        iteration: 0,
        token_usage: crate::TokenUsage::default(),
        last_prompt_usage: None,
        dynamic_state_ref: None,
        dynamic_state_generation: None,
        dynamic_state_snapshot: None,
        plugin_snapshot_ref: None,
        plugin_snapshot_revision: None,
        plugin_snapshot: None,
        execution_state_snapshot: None,
        token_ledger: head.token_ledger,
        checkpoint_ref: head.checkpoint_ref.clone(),
        persisted_graph_node_count: 0,
        graph_replace_required: false,
    };
    state.persisted_graph_node_count = state.session_graph.nodes.len();
    state.policy.model = head.config.configured_model.clone();
    if head.config.context_window > 0 {
        state.policy.max_context_tokens = Some(head.config.context_window as usize);
    }
    state.policy.execution_mode = head.config.execution_mode;
    state.policy.context_approach = head.config.context_approach.clone();
    state.policy.model_variant = head.config.model_variant.clone();
    if let Some(checkpoint) = checkpoint {
        state.iteration = checkpoint.turn_state.iteration;
        state.token_usage = checkpoint.turn_state.token_usage;
        state.last_prompt_usage = checkpoint.turn_state.last_prompt_usage;
        state.dynamic_state_ref = checkpoint.dynamic_state_ref.clone();
        state.dynamic_state_generation = checkpoint
            .dynamic_state
            .as_ref()
            .map(|snapshot| snapshot.base_generation);
        state.dynamic_state_snapshot = checkpoint.dynamic_state;
        state.plugin_snapshot_ref = checkpoint.plugin_snapshot_ref.clone();
        state.plugin_snapshot_revision = checkpoint.plugin_snapshot_revision;
        state.plugin_snapshot = checkpoint.plugin_snapshot;
    }
    state
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct LiveResumeSnapshot {
    #[serde(default = "default_root_session_id")]
    pub session_id: String,
    pub graph: crate::SessionGraph,
    pub config: crate::PersistedSessionConfig,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checkpoint_ref: Option<BlobRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub delta_ref: Option<BlobRef>,
}

impl LiveResumeSnapshot {
    pub fn retained_artifact_refs(&self) -> Vec<RetainedArtifactRef> {
        let mut refs = Vec::new();
        if let Some(blob_ref) = &self.checkpoint_ref {
            refs.push(RetainedArtifactRef {
                blob_ref: blob_ref.clone(),
                kind: PersistedArtifactKind::CheckpointManifest,
            });
        }
        if let Some(blob_ref) = &self.delta_ref {
            refs.push(RetainedArtifactRef {
                blob_ref: blob_ref.clone(),
                kind: PersistedArtifactKind::LiveResumeDelta,
            });
        }
        refs
    }
}

impl Default for SessionHead {
    fn default() -> Self {
        Self {
            session_id: default_root_session_id(),
            graph: crate::SessionGraph::default(),
            config: crate::PersistedSessionConfig::default(),
            checkpoint_ref: None,
            token_ledger: Vec::new(),
        }
    }
}

impl Default for SessionHeadMeta {
    fn default() -> Self {
        Self {
            session_id: default_root_session_id(),
            config: crate::PersistedSessionConfig::default(),
            checkpoint_ref: None,
            leaf_node_id: None,
            graph_node_count: 0,
            token_ledger: Vec::new(),
        }
    }
}

impl Default for LiveResumeSnapshot {
    fn default() -> Self {
        Self {
            session_id: default_root_session_id(),
            graph: crate::SessionGraph::default(),
            config: crate::PersistedSessionConfig::default(),
            checkpoint_ref: None,
            delta_ref: None,
        }
    }
}

pub fn materialize_live_resume_graph(
    snapshot: &LiveResumeSnapshot,
    delta: Option<&LiveResumeDelta>,
) -> crate::SessionGraph {
    delta
        .map(|delta| delta.apply_to_graph(&snapshot.graph))
        .unwrap_or_else(|| snapshot.graph.clone())
}

/// Persistence backend for checkpoint blobs, committed session heads, and live resume snapshots.
#[async_trait::async_trait]
pub trait RuntimeStore: Send + Sync {
    #[doc(hidden)]
    async fn put_blob(&self, content: &[u8]) -> BlobRef;
    async fn put_artifact_blob(
        &self,
        descriptor: BlobArtifactDescriptor,
        content: &[u8],
    ) -> BlobRef {
        let _ = descriptor;
        self.put_blob(content).await
    }
    #[doc(hidden)]
    async fn get_blob(&self, blob_ref: &BlobRef) -> Option<Vec<u8>>;
    #[doc(hidden)]
    async fn append_usage_deltas(&self, entries: &[crate::TokenLedgerEntry]);
    #[doc(hidden)]
    async fn load_usage_deltas(&self) -> Vec<crate::TokenLedgerEntry>;
    #[doc(hidden)]
    async fn save_session_head_meta(&self, meta: SessionHeadMeta);
    #[doc(hidden)]
    async fn load_session_head_meta(&self) -> Option<SessionHeadMeta>;
    #[doc(hidden)]
    async fn replace_session_graph(&self, graph: &crate::SessionGraph);
    #[doc(hidden)]
    async fn append_session_graph_nodes(&self, nodes: &[crate::SessionNodeRecord]);
    #[doc(hidden)]
    async fn load_session_graph(&self) -> crate::SessionGraph;
    #[doc(hidden)]
    async fn save_live_resume(&self, snapshot: LiveResumeSnapshot);
    #[doc(hidden)]
    async fn load_live_resume(&self) -> Option<LiveResumeSnapshot>;
    #[doc(hidden)]
    async fn clear_live_resume(&self);
    async fn save_session_meta(&self, meta: SessionMeta);
    async fn load_session_meta(&self) -> Option<SessionMeta>;
    async fn gc_unreachable(&self) -> GcReport {
        GcReport::default()
    }

    async fn load_persisted_session_state(&self) -> Option<crate::PersistedSessionState> {
        let head = self.load_session_head().await?;
        let checkpoint = match head.checkpoint_ref.as_ref() {
            Some(blob_ref) => get_checkpoint(self, blob_ref).await,
            None => None,
        };
        Some(persisted_session_state_from_head(head, checkpoint))
    }

    async fn refresh_persisted_session_state(&self, state: &mut crate::PersistedSessionState) {
        if let Some(mut fresh) = self.load_persisted_session_state().await {
            // The store owns persisted graph/checkpoint/config state, but not
            // live provider credentials or other runtime-only policy fields.
            fresh.policy.provider = state.policy.provider.clone();
            fresh.policy.session_id = state.policy.session_id.clone();
            fresh.policy.max_turns = state.policy.max_turns;
            *state = fresh;
        }
    }

    async fn apply_runtime_commit(
        &self,
        commit: RuntimeCommit,
    ) -> Result<RuntimeCommitResult, StoreError> {
        Ok(match commit {
            RuntimeCommit::PersistedState(commit) => {
                let stored_checkpoint = put_checkpoint(self, &commit.checkpoint).await;
                if !commit.usage_deltas.is_empty() {
                    self.append_usage_deltas(&commit.usage_deltas).await;
                }
                match &commit.graph {
                    SessionGraphCommit::Replace(graph) => {
                        self.replace_session_graph(graph).await;
                    }
                    SessionGraphCommit::Append { nodes, .. } if !nodes.is_empty() => {
                        self.append_session_graph_nodes(nodes).await;
                    }
                    SessionGraphCommit::Append { .. } => {}
                }
                self.save_session_head_meta(SessionHeadMeta {
                    session_id: commit.session_id,
                    config: commit.config,
                    checkpoint_ref: Some(stored_checkpoint.checkpoint_ref.clone()),
                    leaf_node_id: commit.graph.leaf_node_id().cloned(),
                    graph_node_count: commit.graph.graph_node_count(),
                    token_ledger: Vec::new(),
                })
                .await;
                self.clear_live_resume().await;
                RuntimeCommitResult::PersistedState(PersistedStateCommitResult {
                    checkpoint_ref: stored_checkpoint.checkpoint_ref,
                    manifest: stored_checkpoint.manifest,
                    persisted_graph_node_count: commit.graph.graph_node_count(),
                })
            }
            RuntimeCommit::LiveResume(commit) => {
                let base = if let Some(snapshot) = self.load_live_resume().await {
                    Some((snapshot.graph, snapshot.checkpoint_ref))
                } else {
                    self.load_session_head()
                        .await
                        .map(|head| (head.graph, head.checkpoint_ref))
                };
                let (graph, checkpoint_ref) =
                    base.unwrap_or_else(|| (commit.seed_graph.unwrap_or_default(), None));
                if !crate::messages_are_live_resume_safe(&commit.live_graph.project_messages()) {
                    return Ok(RuntimeCommitResult::LiveResumeSkipped);
                }
                let delta_ref = put_typed_artifact_blob(
                    self,
                    BlobArtifactDescriptor::live_resume_delta(),
                    &crate::LiveResumeDelta {
                        appended_graph_nodes: commit.live_graph.nodes[graph.nodes.len()..].to_vec(),
                        leaf_node_id: commit.live_graph.leaf_node_id.clone(),
                        turn_state: commit.turn_state,
                        dynamic_state: commit.dynamic_state,
                        plugin_snapshot: commit.plugin_snapshot,
                        execution_state_snapshot: commit.execution_state_snapshot,
                        token_ledger: commit.token_ledger,
                    },
                )
                .await;
                self.save_live_resume(crate::LiveResumeSnapshot {
                    session_id: commit.session_id,
                    graph,
                    config: commit.config,
                    checkpoint_ref,
                    delta_ref: Some(delta_ref),
                })
                .await;
                RuntimeCommitResult::LiveResumeSaved
            }
        })
    }

    async fn save_session_head(&self, head: SessionHead) {
        self.replace_session_graph(&head.graph).await;
        self.save_session_head_meta(session_head_meta(&head)).await;
    }

    async fn load_session_head(&self) -> Option<SessionHead> {
        let meta = self.load_session_head_meta().await?;
        let mut graph = self.load_session_graph().await;
        graph.set_leaf_node_id(meta.leaf_node_id.clone());
        Some(SessionHead {
            session_id: meta.session_id,
            graph,
            config: meta.config,
            checkpoint_ref: meta.checkpoint_ref,
            token_ledger: merge_token_ledger_entries(self.load_usage_deltas().await),
        })
    }

    async fn save_turn_checkpoint(&self, head: SessionHead) {
        self.save_session_head(head).await;
        self.clear_live_resume().await;
    }

    async fn head_copy_from_store(&self, source: &(dyn RuntimeStore + '_))
    where
        Self: Sized,
    {
        if let Some(head) = source.load_session_head().await {
            if let Some(checkpoint_ref) = &head.checkpoint_ref {
                let _ = copy_checkpoint_blobs_from_store(self, source, checkpoint_ref).await;
            }
            self.replace_session_graph(&head.graph).await;
            self.save_session_head_meta(session_head_meta(&head)).await;
        }
    }
}

#[cfg(feature = "sqlite-store")]
fn encode_json<T: serde::Serialize>(value: &T) -> String {
    serde_json::to_string(value).expect("persisted state should serialize")
}

#[cfg(feature = "sqlite-store")]
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

#[cfg(feature = "sqlite-store")]
fn compress_blob(content: &[u8]) -> Vec<u8> {
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
    std::io::Write::write_all(&mut encoder, content).expect("compress blob");
    encoder.finish().expect("submit blob compression")
}

#[cfg(feature = "sqlite-store")]
fn decompress_blob(content: &[u8]) -> Option<Vec<u8>> {
    let mut decoder = ZlibDecoder::new(content);
    let mut out = Vec::new();
    std::io::Read::read_to_end(&mut decoder, &mut out).ok()?;
    Some(out)
}

#[cfg(feature = "sqlite-store")]
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

#[cfg(feature = "sqlite-store")]
fn decode_artifact_blob(bytes: &[u8]) -> Option<Vec<u8>> {
    let envelope = decode_msgpack::<StoredBlobEnvelope>(bytes)?;
    match envelope.compression {
        BlobCompression::None => Some(envelope.content),
        BlobCompression::Zlib => decompress_blob(&envelope.content),
    }
}

#[cfg(feature = "sqlite-store")]
fn put_artifact_blob_tx(
    tx: &rusqlite::Transaction<'_>,
    profile: BuiltinBlobProfile,
    descriptor: &BlobArtifactDescriptor,
    content: &[u8],
) -> BlobRef {
    let stored = encode_artifact_blob(descriptor, profile, content);
    let hash = format!("{:x}", Sha256::digest(content));
    tx.execute(
        "INSERT OR IGNORE INTO blobs (hash, content) VALUES (?1, ?2)",
        params![hash, stored],
    )
    .expect("persist artifact blob");
    BlobRef(hash)
}

#[cfg(feature = "sqlite-store")]
fn put_typed_artifact_blob_tx<T: serde::Serialize>(
    tx: &rusqlite::Transaction<'_>,
    profile: BuiltinBlobProfile,
    descriptor: &BlobArtifactDescriptor,
    value: &T,
) -> BlobRef {
    let bytes = encode_msgpack(value);
    put_artifact_blob_tx(tx, profile, descriptor, &bytes)
}

#[cfg(feature = "sqlite-store")]
fn put_checkpoint_tx(
    tx: &rusqlite::Transaction<'_>,
    profile: BuiltinBlobProfile,
    checkpoint: &HydratedSessionCheckpoint,
) -> StoredSessionCheckpoint {
    let dynamic_state_ref = checkpoint
        .dynamic_state
        .as_ref()
        .map(|snapshot| {
            put_typed_artifact_blob_tx(
                tx,
                profile,
                &BlobArtifactDescriptor::dynamic_state_snapshot(),
                snapshot,
            )
        })
        .or_else(|| checkpoint.dynamic_state_ref.clone());
    let plugin_snapshot_ref = checkpoint
        .plugin_snapshot
        .as_ref()
        .map(|snapshot| {
            put_typed_artifact_blob_tx(
                tx,
                profile,
                &BlobArtifactDescriptor::plugin_session_snapshot(),
                snapshot,
            )
        })
        .or_else(|| checkpoint.plugin_snapshot_ref.clone());
    let manifest = SessionCheckpoint {
        turn_state: checkpoint.turn_state.clone(),
        dynamic_state_ref,
        plugin_snapshot_ref,
        plugin_snapshot_revision: checkpoint.plugin_snapshot_revision,
    };
    let checkpoint_ref = put_typed_artifact_blob_tx(
        tx,
        profile,
        &BlobArtifactDescriptor::checkpoint_manifest(),
        &manifest,
    );
    StoredSessionCheckpoint {
        checkpoint_ref,
        manifest,
    }
}

#[cfg(feature = "sqlite-store")]
fn append_usage_deltas_tx(tx: &rusqlite::Transaction<'_>, entries: &[crate::TokenLedgerEntry]) {
    if entries.is_empty() {
        return;
    }
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

#[cfg(feature = "sqlite-store")]
fn save_session_head_meta_tx(tx: &rusqlite::Transaction<'_>, meta: SessionHeadMeta) {
    let head_json = encode_json(&meta);
    tx.execute(
        "INSERT OR REPLACE INTO session_head (singleton, session_id, head_json)
         VALUES (1, ?1, ?2)",
        params![meta.session_id, head_json],
    )
    .expect("persist session head");
}

#[cfg(feature = "sqlite-store")]
fn replace_session_graph_tx(tx: &rusqlite::Transaction<'_>, graph: &crate::SessionGraph) {
    tx.execute("DELETE FROM graph_nodes", [])
        .expect("clear graph rows");
    append_session_graph_nodes_tx(tx, &graph.nodes);
}

#[cfg(feature = "sqlite-store")]
fn append_session_graph_nodes_tx(
    tx: &rusqlite::Transaction<'_>,
    nodes: &[crate::SessionNodeRecord],
) {
    if nodes.is_empty() {
        return;
    }
    for node in nodes {
        let node_json = encode_json(node);
        tx.execute(
            "INSERT INTO graph_nodes (node_id, node_json) VALUES (?1, ?2)",
            params![node.node_id, node_json],
        )
        .expect("persist graph node");
    }
}

#[cfg(feature = "sqlite-store")]
fn save_live_resume_tx(tx: &rusqlite::Transaction<'_>, snapshot: LiveResumeSnapshot) {
    let snapshot_json = encode_json(&snapshot);
    tx.execute(
        "INSERT OR REPLACE INTO live_resume (singleton, session_id, snapshot_json)
         VALUES (1, ?1, ?2)",
        params![snapshot.session_id, snapshot_json],
    )
    .expect("persist live resume");
}

#[cfg(feature = "sqlite-store")]
fn clear_live_resume_tx(tx: &rusqlite::Transaction<'_>) {
    tx.execute("DELETE FROM live_resume WHERE singleton = 1", [])
        .expect("clear live resume");
}

#[cfg(feature = "sqlite-store")]
fn load_live_resume_from_conn(conn: &Connection) -> Option<LiveResumeSnapshot> {
    let snapshot_json: String = conn
        .query_row(
            "SELECT snapshot_json FROM live_resume WHERE singleton = 1",
            [],
            |row| row.get(0),
        )
        .ok()?;
    serde_json::from_str(&snapshot_json).ok()
}

#[cfg(feature = "sqlite-store")]
fn load_session_head_meta_from_conn(conn: &Connection) -> Option<SessionHeadMeta> {
    let head_json: String = conn
        .query_row(
            "SELECT head_json FROM session_head WHERE singleton = 1",
            [],
            |row| row.get(0),
        )
        .ok()?;
    serde_json::from_str(&head_json).ok()
}

#[cfg(feature = "sqlite-store")]
fn load_session_meta_from_conn(conn: &Connection) -> Option<SessionMeta> {
    conn.query_row(
        "SELECT session_id, session_name, created_at, model, cwd, parent_session_id
         FROM session_meta WHERE singleton = 1",
        [],
        |row| {
            Ok(SessionMeta {
                session_id: row.get(0)?,
                session_name: row.get(1)?,
                created_at: row.get(2)?,
                model: row.get(3)?,
                cwd: row.get(4)?,
                parent_session_id: row.get(5)?,
            })
        },
    )
    .ok()
}

#[cfg(feature = "sqlite-store")]
fn load_bound_session_id_from_head_conn(conn: &Connection) -> Option<String> {
    conn.query_row(
        "SELECT session_id FROM session_head WHERE singleton = 1",
        [],
        |row| row.get(0),
    )
    .ok()
}

#[cfg(feature = "sqlite-store")]
fn load_bound_session_id_from_live_resume_conn(conn: &Connection) -> Option<String> {
    conn.query_row(
        "SELECT session_id FROM live_resume WHERE singleton = 1",
        [],
        |row| row.get(0),
    )
    .ok()
}

#[cfg(feature = "sqlite-store")]
fn bound_session_id_from_conn(conn: &Connection) -> Result<Option<String>, StoreError> {
    let mut session_ids = std::collections::BTreeSet::new();
    if let Some(session_id) = load_bound_session_id_from_head_conn(conn) {
        session_ids.insert(session_id);
    }
    if let Some(session_id) = load_bound_session_id_from_live_resume_conn(conn) {
        session_ids.insert(session_id);
    }
    match session_ids.len() {
        0 => Ok(None),
        1 => Ok(session_ids.into_iter().next()),
        _ => Err(StoreError::InconsistentSessionBindings {
            session_ids: session_ids.into_iter().collect(),
        }),
    }
}

#[cfg(feature = "sqlite-store")]
fn validate_session_binding_conn(conn: &Connection, session_id: &str) -> Result<(), StoreError> {
    if let Some(bound_session_id) = bound_session_id_from_conn(conn)?
        && bound_session_id != session_id
    {
        return Err(StoreError::SessionBindingMismatch {
            bound_session_id,
            attempted_session_id: session_id.to_string(),
        });
    }
    Ok(())
}

pub fn encode_checkpoint(checkpoint: &SessionCheckpoint) -> Vec<u8> {
    encode_msgpack(checkpoint)
}

pub fn decode_checkpoint(bytes: &[u8]) -> Option<SessionCheckpoint> {
    rmp_serde::from_slice(bytes).ok()
}

fn encode_msgpack<T: serde::Serialize>(value: &T) -> Vec<u8> {
    rmp_serde::to_vec_named(value).expect("value should serialize")
}

fn decode_msgpack<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> Option<T> {
    rmp_serde::from_slice(bytes).ok()
}

fn merge_token_ledger_entries(
    entries: Vec<crate::TokenLedgerEntry>,
) -> Vec<crate::TokenLedgerEntry> {
    let mut merged: Vec<crate::TokenLedgerEntry> = Vec::new();
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

pub async fn put_typed_blob<T: serde::Serialize, S: RuntimeStore + ?Sized>(
    store: &S,
    value: &T,
) -> BlobRef {
    put_typed_artifact_blob(
        store,
        BlobArtifactDescriptor::new(PersistedArtifactKind::GenericBlob, Vec::new()),
        value,
    )
    .await
}

pub async fn put_typed_artifact_blob<T: serde::Serialize, S: RuntimeStore + ?Sized>(
    store: &S,
    descriptor: BlobArtifactDescriptor,
    value: &T,
) -> BlobRef {
    let bytes = encode_msgpack(value);
    store.put_artifact_blob(descriptor, &bytes).await
}

pub async fn get_typed_blob<T: serde::de::DeserializeOwned, S: RuntimeStore + ?Sized>(
    store: &S,
    blob_ref: &BlobRef,
) -> Option<T> {
    let bytes = store.get_blob(blob_ref).await?;
    decode_msgpack(&bytes)
}

pub async fn put_checkpoint<S: RuntimeStore + ?Sized>(
    store: &S,
    checkpoint: &HydratedSessionCheckpoint,
) -> StoredSessionCheckpoint {
    let dynamic_state_ref = match checkpoint.dynamic_state.as_ref() {
        Some(snapshot) => Some(
            put_typed_artifact_blob(
                store,
                BlobArtifactDescriptor::dynamic_state_snapshot(),
                snapshot,
            )
            .await,
        ),
        None => checkpoint.dynamic_state_ref.clone(),
    };
    let plugin_snapshot_ref = match checkpoint.plugin_snapshot.as_ref() {
        Some(snapshot) => Some(
            put_typed_artifact_blob(
                store,
                BlobArtifactDescriptor::plugin_session_snapshot(),
                snapshot,
            )
            .await,
        ),
        None => checkpoint.plugin_snapshot_ref.clone(),
    };
    let record = SessionCheckpoint {
        turn_state: checkpoint.turn_state.clone(),
        dynamic_state_ref,
        plugin_snapshot_ref,
        plugin_snapshot_revision: checkpoint.plugin_snapshot_revision,
    };
    let checkpoint_ref = put_typed_artifact_blob(
        store,
        BlobArtifactDescriptor::checkpoint_manifest(),
        &record,
    )
    .await;
    StoredSessionCheckpoint {
        checkpoint_ref,
        manifest: record,
    }
}

pub async fn append_usage_deltas<S: RuntimeStore + ?Sized>(
    store: &S,
    entries: &[crate::TokenLedgerEntry],
) {
    store.append_usage_deltas(entries).await;
}

pub async fn load_usage_deltas<S: RuntimeStore + ?Sized>(
    store: &S,
) -> Vec<crate::TokenLedgerEntry> {
    store.load_usage_deltas().await
}

pub async fn get_checkpoint<S: RuntimeStore + ?Sized>(
    store: &S,
    checkpoint_ref: &BlobRef,
) -> Option<HydratedSessionCheckpoint> {
    let record: SessionCheckpoint = get_typed_blob(store, checkpoint_ref).await?;
    let dynamic_state = match record.dynamic_state_ref.as_ref() {
        Some(blob_ref) => get_typed_blob(store, blob_ref).await,
        None => None,
    };
    let plugin_snapshot = match record.plugin_snapshot_ref.as_ref() {
        Some(blob_ref) => get_typed_blob(store, blob_ref).await,
        None => None,
    };
    Some(HydratedSessionCheckpoint {
        turn_state: record.turn_state,
        dynamic_state_ref: record.dynamic_state_ref,
        dynamic_state,
        plugin_snapshot_ref: record.plugin_snapshot_ref,
        plugin_snapshot,
        plugin_snapshot_revision: record.plugin_snapshot_revision,
    })
}

pub async fn copy_checkpoint_blobs_from_store<
    T: RuntimeStore + ?Sized,
    S: RuntimeStore + ?Sized,
>(
    target: &T,
    source: &S,
    checkpoint_ref: &BlobRef,
) -> Option<BlobRef> {
    let record: SessionCheckpoint = get_typed_blob(source, checkpoint_ref).await?;
    for blob_ref in [
        record.dynamic_state_ref.as_ref(),
        record.plugin_snapshot_ref.as_ref(),
    ]
    .into_iter()
    .enumerate()
    {
        let (index, blob_ref) = blob_ref;
        let Some(blob_ref) = blob_ref else {
            continue;
        };
        if let Some(bytes) = source.get_blob(blob_ref).await {
            let descriptor = match index {
                0 => BlobArtifactDescriptor::dynamic_state_snapshot(),
                1 => BlobArtifactDescriptor::plugin_session_snapshot(),
                _ => BlobArtifactDescriptor::new(PersistedArtifactKind::GenericBlob, Vec::new()),
            };
            let _ = target.put_artifact_blob(descriptor, &bytes).await;
        }
    }
    let checkpoint_bytes = source.get_blob(checkpoint_ref).await?;
    Some(
        target
            .put_artifact_blob(
                BlobArtifactDescriptor::checkpoint_manifest(),
                &checkpoint_bytes,
            )
            .await,
    )
}

#[cfg(feature = "sqlite-store")]
impl Store {
    fn load_session_graph_from_conn(
        conn: &Connection,
        leaf_node_id: Option<String>,
    ) -> crate::SessionGraph {
        let mut stmt = match conn.prepare("SELECT node_json FROM graph_nodes ORDER BY seq ASC") {
            Ok(stmt) => stmt,
            Err(err) => {
                tracing::warn!(error = %err, "failed to prepare graph load statement");
                return crate::SessionGraph::from_nodes(Vec::new(), leaf_node_id);
            }
        };
        let rows = match stmt.query_map([], |row| row.get::<_, String>(0)) {
            Ok(rows) => rows,
            Err(err) => {
                tracing::warn!(error = %err, "failed to query graph rows");
                return crate::SessionGraph::from_nodes(Vec::new(), leaf_node_id);
            }
        };
        let mut nodes = Vec::new();
        for row in rows {
            let Ok(node_json) = row else {
                continue;
            };
            let Ok(node) = serde_json::from_str::<crate::SessionNodeRecord>(&node_json) else {
                continue;
            };
            nodes.push(node);
        }
        crate::SessionGraph::from_nodes(nodes, leaf_node_id)
    }

    fn maybe_auto_gc(&self) {
        let Some(interval) = self.options.gc_policy.auto_run_every_commits else {
            return;
        };
        let commits = self.commit_count.fetch_add(1, AtomicOrdering::Relaxed) + 1;
        if commits.is_multiple_of(interval) {
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
                "SELECT session_id, cwd, parent_session_id
                 FROM session_meta WHERE singleton = 1",
                [],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, Option<String>>(1)?,
                        row.get::<_, Option<String>>(2)?,
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
            parent_session_id: meta.2,
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
        let bytes: Vec<u8> = conn
            .query_row(
                "SELECT content FROM blobs WHERE hash = ?1",
                params![blob_ref.as_str()],
                |row| row.get(0),
            )
            .ok()?;
        decode_artifact_blob(&bytes).or(Some(bytes))
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
        let bytes = self.get_blob(blob_ref)?;
        decode_msgpack(&bytes)
    }

    pub fn put_checkpoint(
        &self,
        checkpoint: &HydratedSessionCheckpoint,
    ) -> StoredSessionCheckpoint {
        let dynamic_state_ref = checkpoint
            .dynamic_state
            .as_ref()
            .map(|snapshot| {
                self.put_typed_artifact_blob(
                    BlobArtifactDescriptor::dynamic_state_snapshot(),
                    snapshot,
                )
            })
            .or_else(|| checkpoint.dynamic_state_ref.clone());
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
        let manifest = SessionCheckpoint {
            turn_state: checkpoint.turn_state.clone(),
            dynamic_state_ref,
            plugin_snapshot_ref,
            plugin_snapshot_revision: checkpoint.plugin_snapshot_revision,
        };
        let checkpoint_ref =
            self.put_typed_artifact_blob(BlobArtifactDescriptor::checkpoint_manifest(), &manifest);
        StoredSessionCheckpoint {
            checkpoint_ref,
            manifest,
        }
    }

    pub fn get_checkpoint(&self, blob_ref: &BlobRef) -> Option<HydratedSessionCheckpoint> {
        let record: SessionCheckpoint = self.get_typed_blob(blob_ref)?;
        Some(HydratedSessionCheckpoint {
            turn_state: record.turn_state,
            dynamic_state_ref: record.dynamic_state_ref.clone(),
            dynamic_state: record
                .dynamic_state_ref
                .as_ref()
                .and_then(|blob_ref| self.get_typed_blob(blob_ref)),
            plugin_snapshot_ref: record.plugin_snapshot_ref.clone(),
            plugin_snapshot: record
                .plugin_snapshot_ref
                .as_ref()
                .and_then(|blob_ref| self.get_typed_blob(blob_ref)),
            plugin_snapshot_revision: record.plugin_snapshot_revision,
        })
    }

    pub fn append_usage_deltas(&self, entries: &[crate::TokenLedgerEntry]) {
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

    pub fn load_usage_deltas(&self) -> Vec<crate::TokenLedgerEntry> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = match conn.prepare(
            "SELECT source, model, input_tokens, output_tokens, cached_input_tokens, reasoning_tokens
             FROM usage_deltas ORDER BY seq ASC",
        ) {
            Ok(stmt) => stmt,
            Err(_) => return Vec::new(),
        };
        let rows = match stmt.query_map([], |row| {
            Ok(crate::TokenLedgerEntry {
                source: row.get(0)?,
                model: row.get(1)?,
                usage: crate::TokenUsage {
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

    pub fn save_live_resume(&self, snapshot: LiveResumeSnapshot) {
        let conn = self.conn.lock().unwrap();
        let snapshot_json = encode_json(&snapshot);
        conn.execute(
            "INSERT OR REPLACE INTO live_resume (singleton, snapshot_json)
             VALUES (1, ?1)",
            params![snapshot_json],
        )
        .unwrap();
    }

    pub fn load_live_resume(&self) -> Option<LiveResumeSnapshot> {
        let conn = self.conn.lock().unwrap();
        load_live_resume_from_conn(&conn)
    }

    pub fn clear_live_resume(&self) {
        let conn = self.conn.lock().unwrap();
        if let Err(err) = conn.execute("DELETE FROM live_resume WHERE singleton = 1", []) {
            tracing::warn!(error = %err, "failed to clear live resume snapshot");
        }
    }

    pub fn save_ui_resume_state<T: serde::Serialize>(&self, state: &T) {
        let state_json = match serde_json::to_string(state) {
            Ok(state_json) => state_json,
            Err(err) => {
                tracing::warn!(error = %err, "failed to serialize UI resume state");
                return;
            }
        };
        let conn = self.conn.lock().unwrap();
        if let Err(err) = conn.execute(
            "INSERT OR REPLACE INTO ui_resume_state (singleton, state_json)
             VALUES (1, ?1)",
            params![state_json],
        ) {
            tracing::warn!(error = %err, "failed to persist UI resume state");
        }
    }

    pub fn load_ui_resume_state<T: serde::de::DeserializeOwned>(&self) -> Option<T> {
        let conn = self.conn.lock().unwrap();
        let state_json: String = conn
            .query_row(
                "SELECT state_json FROM ui_resume_state WHERE singleton = 1",
                [],
                |row| row.get(0),
            )
            .ok()?;
        serde_json::from_str(&state_json).ok()
    }

    pub fn save_session_head_meta(&self, meta: SessionHeadMeta) {
        let conn = self.conn.lock().unwrap();
        let head_json = encode_json(&meta);
        if let Err(err) = conn.execute(
            "INSERT OR REPLACE INTO session_head (singleton, head_json)
             VALUES (1, ?1)",
            params![head_json],
        ) {
            tracing::warn!(error = %err, "failed to persist session head");
        }
    }

    pub fn load_session_head_meta(&self) -> Option<SessionHeadMeta> {
        let conn = self.conn.lock().unwrap();
        load_session_head_meta_from_conn(&conn)
    }

    pub fn replace_session_graph(&self, graph: &crate::SessionGraph) {
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

    pub fn append_session_graph_nodes(&self, nodes: &[crate::SessionNodeRecord]) {
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

    pub fn load_session_graph(&self) -> crate::SessionGraph {
        let conn = self.conn.lock().unwrap();
        Self::load_session_graph_from_conn(&conn, None)
    }

    pub fn apply_runtime_commit(
        &self,
        commit: RuntimeCommit,
    ) -> Result<RuntimeCommitResult, StoreError> {
        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction().expect("runtime commit transaction");
        let result = match commit {
            RuntimeCommit::PersistedState(commit) => {
                validate_session_binding_conn(&tx, &commit.session_id)?;
                let stored_checkpoint =
                    put_checkpoint_tx(&tx, self.options.blob_profile, &commit.checkpoint);
                append_usage_deltas_tx(&tx, &commit.usage_deltas);
                match &commit.graph {
                    SessionGraphCommit::Replace(graph) => replace_session_graph_tx(&tx, graph),
                    SessionGraphCommit::Append { nodes, .. } => {
                        append_session_graph_nodes_tx(&tx, nodes)
                    }
                }
                save_session_head_meta_tx(
                    &tx,
                    SessionHeadMeta {
                        session_id: commit.session_id,
                        config: commit.config,
                        checkpoint_ref: Some(stored_checkpoint.checkpoint_ref.clone()),
                        leaf_node_id: commit.graph.leaf_node_id().cloned(),
                        graph_node_count: commit.graph.graph_node_count(),
                        token_ledger: Vec::new(),
                    },
                );
                clear_live_resume_tx(&tx);
                RuntimeCommitResult::PersistedState(PersistedStateCommitResult {
                    checkpoint_ref: stored_checkpoint.checkpoint_ref,
                    manifest: stored_checkpoint.manifest,
                    persisted_graph_node_count: commit.graph.graph_node_count(),
                })
            }
            RuntimeCommit::LiveResume(commit) => {
                validate_session_binding_conn(&tx, &commit.session_id)?;
                let base = load_live_resume_from_conn(&tx)
                    .map(|snapshot| (snapshot.graph, snapshot.checkpoint_ref))
                    .or_else(|| {
                        load_session_head_meta_from_conn(&tx).map(|meta| {
                            (
                                Self::load_session_graph_from_conn(&tx, meta.leaf_node_id.clone()),
                                meta.checkpoint_ref,
                            )
                        })
                    });
                let (graph, checkpoint_ref) =
                    base.unwrap_or_else(|| (commit.seed_graph.unwrap_or_default(), None));
                if !crate::messages_are_live_resume_safe(&commit.live_graph.project_messages()) {
                    RuntimeCommitResult::LiveResumeSkipped
                } else {
                    let delta_ref = put_typed_artifact_blob_tx(
                        &tx,
                        self.options.blob_profile,
                        &BlobArtifactDescriptor::live_resume_delta(),
                        &crate::LiveResumeDelta {
                            appended_graph_nodes: commit.live_graph.nodes[graph.nodes.len()..]
                                .to_vec(),
                            leaf_node_id: commit.live_graph.leaf_node_id.clone(),
                            turn_state: commit.turn_state,
                            dynamic_state: commit.dynamic_state,
                            plugin_snapshot: commit.plugin_snapshot,
                            execution_state_snapshot: commit.execution_state_snapshot,
                            token_ledger: commit.token_ledger,
                        },
                    );
                    save_live_resume_tx(
                        &tx,
                        LiveResumeSnapshot {
                            session_id: commit.session_id,
                            graph,
                            config: commit.config,
                            checkpoint_ref,
                            delta_ref: Some(delta_ref),
                        },
                    );
                    RuntimeCommitResult::LiveResumeSaved
                }
            }
        };
        tx.commit().expect("runtime commit commit");
        drop(conn);
        self.maybe_auto_gc();
        Ok(result)
    }

    pub fn gc_unreachable(&self) -> GcReport {
        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction().expect("gc transaction");
        let head_meta = load_session_head_meta_from_conn(&tx);
        let live_resume = load_live_resume_from_conn(&tx);
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
        if let Some(snapshot) = &live_resume {
            roots.extend(snapshot.retained_artifact_refs());
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
            stack.extend(checkpoint.retained_artifact_refs());
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
                    record.dynamic_state_ref.as_ref(),
                    record.plugin_snapshot_ref.as_ref(),
                ]
                .into_iter()
                .flatten()
                {
                    if let Some(blob) = source.get_blob(blob_ref) {
                        let descriptor = match record
                            .dynamic_state_ref
                            .as_ref()
                            .filter(|candidate| *candidate == blob_ref)
                        {
                            Some(_) => BlobArtifactDescriptor::dynamic_state_snapshot(),
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
        if let Err(err) = conn.execute(
            "INSERT OR REPLACE INTO session_meta
             (singleton, session_id, session_name, created_at, model, cwd, parent_session_id)
             VALUES (1, ?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                meta.session_id,
                meta.session_name,
                meta.created_at,
                meta.model,
                meta.cwd,
                meta.parent_session_id
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

#[cfg(feature = "sqlite-store")]
#[async_trait::async_trait]
impl RuntimeStore for Store {
    async fn put_blob(&self, content: &[u8]) -> BlobRef {
        Self::put_blob(self, content)
    }

    async fn put_artifact_blob(
        &self,
        descriptor: BlobArtifactDescriptor,
        content: &[u8],
    ) -> BlobRef {
        Self::put_artifact_blob(self, descriptor, content)
    }

    async fn get_blob(&self, blob_ref: &BlobRef) -> Option<Vec<u8>> {
        Self::get_blob(self, blob_ref)
    }

    async fn append_usage_deltas(&self, entries: &[crate::TokenLedgerEntry]) {
        Self::append_usage_deltas(self, entries);
    }

    async fn load_usage_deltas(&self) -> Vec<crate::TokenLedgerEntry> {
        Self::load_usage_deltas(self)
    }

    async fn save_session_head_meta(&self, meta: SessionHeadMeta) {
        Self::save_session_head_meta(self, meta);
    }

    async fn load_session_head_meta(&self) -> Option<SessionHeadMeta> {
        Self::load_session_head_meta(self)
    }

    async fn replace_session_graph(&self, graph: &crate::SessionGraph) {
        Self::replace_session_graph(self, graph);
    }

    async fn append_session_graph_nodes(&self, nodes: &[crate::SessionNodeRecord]) {
        Self::append_session_graph_nodes(self, nodes);
    }

    async fn load_session_graph(&self) -> crate::SessionGraph {
        Self::load_session_graph(self)
    }

    async fn save_live_resume(&self, snapshot: LiveResumeSnapshot) {
        Self::save_live_resume(self, snapshot);
    }

    async fn load_live_resume(&self) -> Option<LiveResumeSnapshot> {
        Self::load_live_resume(self)
    }

    async fn clear_live_resume(&self) {
        Self::clear_live_resume(self);
    }

    async fn save_session_meta(&self, meta: SessionMeta) {
        Self::save_session_meta(self, meta);
    }

    async fn load_session_meta(&self) -> Option<SessionMeta> {
        Self::load_session_meta(self)
    }

    async fn gc_unreachable(&self) -> GcReport {
        Self::gc_unreachable(self)
    }

    async fn apply_runtime_commit(
        &self,
        commit: RuntimeCommit,
    ) -> Result<RuntimeCommitResult, StoreError> {
        Self::apply_runtime_commit(self, commit)
    }
}

#[cfg(feature = "sqlite-store")]
fn ensure_schema(conn: &Connection) -> rusqlite::Result<()> {
    let user_version: i32 = conn.query_row("PRAGMA user_version", [], |row| row.get(0))?;
    if user_version == SCHEMA_VERSION {
        conn.execute_batch(SCHEMA)?;
        return Ok(());
    }

    if user_version == 11 {
        migrate_schema_v11_to_v12(conn)?;
        conn.execute_batch(SCHEMA)?;
        conn.pragma_update(None, "user_version", SCHEMA_VERSION)?;
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

#[cfg(feature = "sqlite-store")]
fn migrate_schema_v11_to_v12(conn: &Connection) -> rusqlite::Result<()> {
    conn.execute(
        "ALTER TABLE session_head ADD COLUMN session_id TEXT NOT NULL DEFAULT 'root'",
        [],
    )?;
    conn.execute(
        "ALTER TABLE live_resume ADD COLUMN session_id TEXT NOT NULL DEFAULT 'root'",
        [],
    )?;

    let head_rows: Vec<(i64, String)> = {
        let mut stmt = conn.prepare("SELECT singleton, head_json FROM session_head")?;
        let rows = stmt.query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?;
        rows.collect::<rusqlite::Result<Vec<_>>>()?
    };
    for (singleton, head_json) in head_rows {
        let session_id = serde_json::from_str::<SessionHeadMeta>(&head_json)
            .map(|meta| meta.session_id)
            .unwrap_or_else(|_| default_root_session_id());
        conn.execute(
            "UPDATE session_head SET session_id = ?1 WHERE singleton = ?2",
            params![session_id, singleton],
        )?;
    }

    let live_rows: Vec<(i64, String)> = {
        let mut stmt = conn.prepare("SELECT singleton, snapshot_json FROM live_resume")?;
        let rows = stmt.query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?;
        rows.collect::<rusqlite::Result<Vec<_>>>()?
    };
    for (singleton, snapshot_json) in live_rows {
        let session_id = serde_json::from_str::<LiveResumeSnapshot>(&snapshot_json)
            .map(|snapshot| snapshot.session_id)
            .unwrap_or_else(|_| default_root_session_id());
        conn.execute(
            "UPDATE live_resume SET session_id = ?1 WHERE singleton = ?2",
            params![session_id, singleton],
        )?;
    }

    Ok(())
}

#[cfg(feature = "sqlite-store")]
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

#[cfg(feature = "sqlite-store")]
fn unsupported_schema_message() -> String {
    format!(
        "Unsupported lash session schema. Delete {} and try again.",
        crate::lash_home().join("sessions").display()
    )
}

#[cfg(all(test, feature = "sqlite-store"))]
mod tests {
    use super::*;
    use crate::session_model::{Message, MessageRole, Part, PartKind, PruneState};
    use rusqlite::Connection;
    fn mem() -> Store {
        Store::memory().unwrap()
    }

    fn text_message(id: &str, role: MessageRole, content: &str) -> Message {
        Message {
            id: id.to_string(),
            role,
            parts: vec![Part {
                id: format!("{id}.p0"),
                kind: PartKind::Text,
                content: content.to_string(),
                attachment: None,
                tool_call_id: None,
                tool_name: None,
                tool_item_id: None,
                tool_signature: None,
                prune_state: PruneState::Intact,
                reasoning_meta: None,
            }],
            user_input: None,
            origin: None,
        }
    }

    #[test]
    fn open_rejects_legacy_session_schema() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("legacy.db");
        let conn = Connection::open(&path).unwrap();
        conn.execute_batch(
            "
            CREATE TABLE agents (
                agent_id TEXT PRIMARY KEY,
                messages TEXT NOT NULL DEFAULT '[]'
            );
            INSERT INTO agents (agent_id, messages) VALUES ('root', '[]');
            ",
        )
        .unwrap();
        drop(conn);

        let err = match Store::open(&path) {
            Ok(_) => panic!("legacy schema should be rejected"),
            Err(err) => err.to_string(),
        };
        assert!(err.contains("Unsupported lash session schema"));
    }

    #[test]
    fn open_uses_wal_journal_mode_and_busy_timeout() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("store.db");
        let store = Store::open(&path).unwrap();
        let conn = store.conn.lock().unwrap();
        let journal_mode: String = conn
            .query_row("PRAGMA journal_mode", [], |row| row.get(0))
            .unwrap();
        let busy_timeout_ms: i64 = conn
            .query_row("PRAGMA busy_timeout", [], |row| row.get(0))
            .unwrap();
        assert_eq!(journal_mode.to_ascii_lowercase(), "wal");
        assert_eq!(busy_timeout_ms, SQLITE_BUSY_TIMEOUT.as_millis() as i64);
    }

    #[test]
    fn open_migrates_v11_binding_columns_without_full_snapshot_decode_on_hot_path() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("v11.db");
        let conn = Connection::open(&path).unwrap();
        conn.execute_batch(
            "
            CREATE TABLE live_resume (
                singleton      INTEGER PRIMARY KEY CHECK (singleton = 1),
                snapshot_json  TEXT NOT NULL DEFAULT '{}'
            );
            CREATE TABLE session_head (
                singleton      INTEGER PRIMARY KEY CHECK (singleton = 1),
                head_json      TEXT NOT NULL DEFAULT '{}'
            );
            ",
        )
        .unwrap();
        conn.pragma_update(None, "user_version", 11).unwrap();
        let head_json = serde_json::to_string(&SessionHeadMeta {
            session_id: "head-session".to_string(),
            config: crate::PersistedSessionConfig::default(),
            checkpoint_ref: None,
            leaf_node_id: None,
            graph_node_count: 0,
            token_ledger: Vec::new(),
        })
        .unwrap();
        let snapshot_json = serde_json::to_string(&LiveResumeSnapshot {
            session_id: "resume-session".to_string(),
            graph: crate::SessionGraph::default(),
            config: crate::PersistedSessionConfig::default(),
            checkpoint_ref: None,
            delta_ref: None,
        })
        .unwrap();
        conn.execute(
            "INSERT INTO session_head (singleton, head_json) VALUES (1, ?1)",
            params![head_json],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO live_resume (singleton, snapshot_json) VALUES (1, ?1)",
            params![snapshot_json],
        )
        .unwrap();
        drop(conn);

        let store = Store::open(&path).unwrap();
        let conn = store.conn.lock().unwrap();
        let err = bound_session_id_from_conn(&conn).unwrap_err().to_string();
        drop(conn);
        assert!(err.contains("inconsistent session bindings"));

        let conn = Connection::open(&path).unwrap();
        let version: i32 = conn
            .query_row("PRAGMA user_version", [], |row| row.get(0))
            .unwrap();
        assert_eq!(version, SCHEMA_VERSION);
        let head_binding: String = conn
            .query_row(
                "SELECT session_id FROM session_head WHERE singleton = 1",
                [],
                |row| row.get(0),
            )
            .unwrap();
        let resume_binding: String = conn
            .query_row(
                "SELECT session_id FROM live_resume WHERE singleton = 1",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(head_binding, "head-session");
        assert_eq!(resume_binding, "resume-session");
    }

    #[test]
    fn graph_copy_from_store_round_trip() {
        let source = mem();
        source.save_session_head(SessionHead {
            session_id: "root".to_string(),
            graph: crate::SessionGraph::from_projection(
                &[
                    text_message("u0", MessageRole::User, "hello"),
                    text_message("a0", MessageRole::Assistant, "world"),
                ],
                &[],
            ),
            config: crate::PersistedSessionConfig::default(),
            checkpoint_ref: None,
            token_ledger: Vec::new(),
        });

        let target = mem();
        target.head_copy_from_store(&source);

        let graph = target.load_session_head().expect("session head").graph;
        let messages = graph.project_messages();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].parts[0].content, "hello");
        assert_eq!(messages[1].parts[0].content, "world");
    }

    #[test]
    fn save_session_head_rewrites_existing_snapshot() {
        let store = mem();
        store.save_session_head(SessionHead {
            session_id: "root".to_string(),
            graph: crate::SessionGraph::from_projection(
                &[text_message("u0", MessageRole::User, "old")],
                &[],
            ),
            config: crate::PersistedSessionConfig::default(),
            checkpoint_ref: None,
            token_ledger: Vec::new(),
        });

        store.save_session_head(SessionHead {
            session_id: "root".to_string(),
            graph: crate::SessionGraph::from_projection(
                &[text_message("u1", MessageRole::User, "updated")],
                &[],
            ),
            config: crate::PersistedSessionConfig::default(),
            checkpoint_ref: None,
            token_ledger: Vec::new(),
        });

        let graph = store.load_session_head().expect("session head").graph;
        let messages = graph.project_messages();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].parts[0].content, "updated");
    }

    #[test]
    fn live_resume_rewrites_and_clears() {
        let store = mem();
        store.save_live_resume(LiveResumeSnapshot {
            graph: crate::SessionGraph::from_projection(
                &[text_message("u0", MessageRole::User, "first")],
                &[],
            ),
            ..LiveResumeSnapshot::default()
        });
        store.save_live_resume(LiveResumeSnapshot {
            graph: crate::SessionGraph::from_projection(
                &[text_message("u1", MessageRole::User, "second")],
                &[],
            ),
            ..LiveResumeSnapshot::default()
        });

        let graph = store.load_live_resume().expect("live resume").graph;
        let messages = graph.project_messages();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].parts[0].content, "second");

        store.clear_live_resume();
        assert!(store.load_live_resume().is_none());
    }

    #[test]
    fn load_picker_info_reads_message_graph() {
        let store = mem();
        store.save_session_meta(SessionMeta {
            session_id: "s1".to_string(),
            session_name: "demo".to_string(),
            created_at: "2026-04-05T12:00:00Z".to_string(),
            model: "gpt-5".to_string(),
            cwd: Some("/tmp/demo".to_string()),
            parent_session_id: None,
        });
        store.save_session_head(SessionHead {
            session_id: "s1".to_string(),
            graph: crate::SessionGraph::from_projection(
                &[
                    text_message("u0", MessageRole::User, "hello there"),
                    text_message("a0", MessageRole::Assistant, "response"),
                    text_message("u1", MessageRole::User, "follow up"),
                ],
                &[],
            ),
            config: crate::PersistedSessionConfig::default(),
            checkpoint_ref: None,
            token_ledger: Vec::new(),
        });

        let info = store.load_picker_info().expect("picker info");
        assert_eq!(info.session_id, "s1");
        assert_eq!(info.first_user_message, "hello there");
        assert_eq!(info.user_message_count, 2);
    }

    #[test]
    fn checkpoint_round_trips_through_blob_store() {
        let store = mem();
        let checkpoint = HydratedSessionCheckpoint {
            turn_state: crate::PersistedTurnState {
                iteration: 7,
                token_usage: crate::TokenUsage {
                    input_tokens: 12,
                    output_tokens: 3,
                    cached_input_tokens: 1,
                    reasoning_tokens: 2,
                },
                last_prompt_usage: None,
            },
            dynamic_state_ref: None,
            dynamic_state: None,
            plugin_snapshot_ref: None,
            plugin_snapshot_revision: None,
            plugin_snapshot: None,
        };
        let stored = store.put_checkpoint(&checkpoint);
        let checkpoint_record = store
            .get_blob(&stored.checkpoint_ref)
            .and_then(|bytes| decode_checkpoint(&bytes))
            .expect("checkpoint record");
        assert_eq!(checkpoint_record.turn_state.iteration, 7);
        assert!(checkpoint_record.dynamic_state_ref.is_none());
        assert!(checkpoint_record.plugin_snapshot_ref.is_none());
        let loaded = store
            .get_checkpoint(&stored.checkpoint_ref)
            .expect("checkpoint");
        assert_eq!(loaded.turn_state.iteration, 7);
        assert!(loaded.dynamic_state.is_none());
        assert!(loaded.plugin_snapshot.is_none());
    }
}
