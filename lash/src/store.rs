#[cfg(feature = "sqlite-store")]
use std::path::Path;
#[cfg(feature = "sqlite-store")]
use std::sync::Mutex;
#[cfg(feature = "sqlite-store")]
use std::time::Duration;

#[cfg(feature = "sqlite-store")]
use rusqlite::{Connection, OpenFlags, params};
#[cfg(feature = "sqlite-store")]
use sha2::{Digest, Sha256};

/// SQLite-backed store for checkpoint blobs, live resume state, and the canonical session head.
#[cfg(feature = "sqlite-store")]
pub struct Store {
    conn: Mutex<Connection>,
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
    snapshot_json  TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS ui_resume_state (
    singleton      INTEGER PRIMARY KEY CHECK (singleton = 1),
    state_json     TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS session_head (
    singleton      INTEGER PRIMARY KEY CHECK (singleton = 1),
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
const SCHEMA_VERSION: i32 = 11;

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

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, Default)]
pub struct SessionHead {
    pub graph: crate::SessionGraph,
    pub config: crate::PersistedSessionConfig,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checkpoint_ref: Option<BlobRef>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub token_ledger: Vec<crate::TokenLedgerEntry>,
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct SessionHeadMeta {
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
        config: head.config.clone(),
        checkpoint_ref: head.checkpoint_ref.clone(),
        leaf_node_id: head.graph.leaf_node_id.clone(),
        graph_node_count: head.graph.nodes.len(),
        token_ledger: head.token_ledger.clone(),
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, Default)]
pub struct LiveResumeSnapshot {
    pub graph: crate::SessionGraph,
    pub config: crate::PersistedSessionConfig,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checkpoint_ref: Option<BlobRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub delta_ref: Option<BlobRef>,
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
    async fn put_blob(&self, content: &[u8]) -> BlobRef;
    async fn get_blob(&self, blob_ref: &BlobRef) -> Option<Vec<u8>>;
    async fn append_usage_deltas(&self, entries: &[crate::TokenLedgerEntry]);
    async fn load_usage_deltas(&self) -> Vec<crate::TokenLedgerEntry>;
    async fn save_session_head_meta(&self, meta: SessionHeadMeta);
    async fn load_session_head_meta(&self) -> Option<SessionHeadMeta>;
    async fn replace_session_graph(&self, graph: &crate::SessionGraph);
    async fn append_session_graph_nodes(&self, nodes: &[crate::SessionNodeRecord]);
    async fn load_session_graph(&self) -> crate::SessionGraph;
    async fn save_live_resume(&self, snapshot: LiveResumeSnapshot);
    async fn load_live_resume(&self) -> Option<LiveResumeSnapshot>;
    async fn clear_live_resume(&self);
    async fn save_session_meta(&self, meta: SessionMeta);
    async fn load_session_meta(&self) -> Option<SessionMeta>;

    async fn save_session_head(&self, head: SessionHead) {
        self.replace_session_graph(&head.graph).await;
        self.save_session_head_meta(session_head_meta(&head)).await;
    }

    async fn load_session_head(&self) -> Option<SessionHead> {
        let meta = self.load_session_head_meta().await?;
        let mut graph = self.load_session_graph().await;
        graph.set_leaf_node_id(meta.leaf_node_id.clone());
        Some(SessionHead {
            graph,
            config: meta.config,
            checkpoint_ref: meta.checkpoint_ref,
            token_ledger: meta.token_ledger,
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

pub async fn put_typed_blob<T: serde::Serialize>(
    store: &(dyn RuntimeStore + '_),
    value: &T,
) -> BlobRef {
    let bytes = encode_msgpack(value);
    store.put_blob(&bytes).await
}

pub async fn get_typed_blob<T: serde::de::DeserializeOwned>(
    store: &(dyn RuntimeStore + '_),
    blob_ref: &BlobRef,
) -> Option<T> {
    let bytes = store.get_blob(blob_ref).await?;
    decode_msgpack(&bytes)
}

pub async fn put_checkpoint(
    store: &(dyn RuntimeStore + '_),
    checkpoint: &HydratedSessionCheckpoint,
) -> StoredSessionCheckpoint {
    let dynamic_state_ref = match checkpoint.dynamic_state.as_ref() {
        Some(snapshot) => Some(put_typed_blob(store, snapshot).await),
        None => checkpoint.dynamic_state_ref.clone(),
    };
    let plugin_snapshot_ref = match checkpoint.plugin_snapshot.as_ref() {
        Some(snapshot) => Some(put_typed_blob(store, snapshot).await),
        None => checkpoint.plugin_snapshot_ref.clone(),
    };
    let record = SessionCheckpoint {
        turn_state: checkpoint.turn_state.clone(),
        dynamic_state_ref,
        plugin_snapshot_ref,
        plugin_snapshot_revision: checkpoint.plugin_snapshot_revision,
    };
    let checkpoint_ref = put_typed_blob(store, &record).await;
    StoredSessionCheckpoint {
        checkpoint_ref,
        manifest: record,
    }
}

pub async fn append_usage_deltas(
    store: &(dyn RuntimeStore + '_),
    entries: &[crate::TokenLedgerEntry],
) {
    store.append_usage_deltas(entries).await;
}

pub async fn load_usage_deltas(store: &(dyn RuntimeStore + '_)) -> Vec<crate::TokenLedgerEntry> {
    store.load_usage_deltas().await
}

pub async fn get_checkpoint(
    store: &(dyn RuntimeStore + '_),
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

pub async fn copy_checkpoint_blobs_from_store(
    target: &(dyn RuntimeStore + '_),
    source: &(dyn RuntimeStore + '_),
    checkpoint_ref: &BlobRef,
) -> Option<BlobRef> {
    let record: SessionCheckpoint = get_typed_blob(source, checkpoint_ref).await?;
    for blob_ref in [
        record.dynamic_state_ref.as_ref(),
        record.plugin_snapshot_ref.as_ref(),
    ]
    .into_iter()
    .flatten()
    {
        if let Some(bytes) = source.get_blob(blob_ref).await {
            let _ = target.put_blob(&bytes).await;
        }
    }
    let checkpoint_bytes = source.get_blob(checkpoint_ref).await?;
    Some(target.put_blob(&checkpoint_bytes).await)
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

    /// Open (or create) a SQLite database at `path`.
    pub fn open(path: &Path) -> rusqlite::Result<Self> {
        let conn = Connection::open(path)?;
        apply_pragmas(&conn, StoreBacking::File)?;
        ensure_schema(&conn)?;
        Ok(Self {
            conn: Mutex::new(conn),
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
        let conn = Connection::open_in_memory()?;
        apply_pragmas(&conn, StoreBacking::Memory)?;
        ensure_schema(&conn)?;
        Ok(Self {
            conn: Mutex::new(conn),
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

    pub fn get_blob(&self, blob_ref: &BlobRef) -> Option<Vec<u8>> {
        let conn = self.conn.lock().unwrap();
        conn.query_row(
            "SELECT content FROM blobs WHERE hash = ?1",
            params![blob_ref.as_str()],
            |row| row.get(0),
        )
        .ok()
    }

    pub fn put_typed_blob<T: serde::Serialize>(&self, value: &T) -> BlobRef {
        let bytes = encode_msgpack(value);
        self.put_blob(&bytes)
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
            .map(|snapshot| self.put_typed_blob(snapshot))
            .or_else(|| checkpoint.dynamic_state_ref.clone());
        let plugin_snapshot_ref = checkpoint
            .plugin_snapshot
            .as_ref()
            .map(|snapshot| self.put_typed_blob(snapshot))
            .or_else(|| checkpoint.plugin_snapshot_ref.clone());
        let manifest = SessionCheckpoint {
            turn_state: checkpoint.turn_state.clone(),
            dynamic_state_ref,
            plugin_snapshot_ref,
            plugin_snapshot_revision: checkpoint.plugin_snapshot_revision,
        };
        let checkpoint_ref = self.put_typed_blob(&manifest);
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
        let snapshot_json: String = conn
            .query_row(
                "SELECT snapshot_json FROM live_resume WHERE singleton = 1",
                [],
                |row| row.get(0),
            )
            .ok()?;
        serde_json::from_str(&snapshot_json).ok()
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
        let head_json: String = conn
            .query_row(
                "SELECT head_json FROM session_head WHERE singleton = 1",
                [],
                |row| row.get(0),
            )
            .ok()?;
        serde_json::from_str(&head_json).ok()
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

    pub fn save_session_head(&self, head: SessionHead) {
        self.replace_session_graph(&head.graph);
        self.save_session_head_meta(session_head_meta(&head));
    }

    pub fn load_session_head(&self) -> Option<SessionHead> {
        let meta = self.load_session_head_meta()?;
        let mut graph = self.load_session_graph();
        graph.set_leaf_node_id(meta.leaf_node_id.clone());
        Some(SessionHead {
            graph,
            config: meta.config,
            checkpoint_ref: meta.checkpoint_ref,
            token_ledger: meta.token_ledger,
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
                        let _ = self.put_blob(&blob);
                    }
                }
                if let Some(blob) = source.get_blob(checkpoint_ref) {
                    let _ = self.put_blob(&blob);
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
}

#[cfg(feature = "sqlite-store")]
#[async_trait::async_trait]
impl RuntimeStore for Store {
    async fn put_blob(&self, content: &[u8]) -> BlobRef {
        Self::put_blob(self, content)
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
}

#[cfg(feature = "sqlite-store")]
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
                prune_state: PruneState::Intact,
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
    fn graph_copy_from_store_round_trip() {
        let source = mem();
        source.save_session_head(SessionHead {
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
            graph: crate::SessionGraph::from_projection(
                &[text_message("u0", MessageRole::User, "old")],
                &[],
            ),
            config: crate::PersistedSessionConfig::default(),
            checkpoint_ref: None,
            token_ledger: Vec::new(),
        });

        store.save_session_head(SessionHead {
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
