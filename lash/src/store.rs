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

/// SQLite-backed store for archive, runtime state, and the canonical session graph.
#[cfg(feature = "sqlite-store")]
pub struct Store {
    conn: Mutex<Connection>,
}

#[cfg(feature = "sqlite-store")]
pub type SqliteStore = Store;

#[cfg(feature = "sqlite-store")]
const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS archive (
    hash    TEXT PRIMARY KEY,
    content TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS live_session_graph (
    singleton      INTEGER PRIMARY KEY CHECK (singleton = 1),
    graph_json     TEXT NOT NULL DEFAULT '{\"nodes\":[],\"leaf_node_id\":null}'
);

CREATE TABLE IF NOT EXISTS ui_resume_state (
    singleton      INTEGER PRIMARY KEY CHECK (singleton = 1),
    state_json     TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS session_graph (
    singleton      INTEGER PRIMARY KEY CHECK (singleton = 1),
    graph_json     TEXT NOT NULL DEFAULT '{\"nodes\":[],\"leaf_node_id\":null}'
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
const SCHEMA_VERSION: i32 = 8;

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

/// Persistence backend for archived content, committed session graphs, and live graphs.
#[async_trait::async_trait]
pub trait RuntimeStore: Send + Sync {
    async fn store_archive(&self, content: &str) -> String;
    async fn get_archive(&self, hash: &str) -> Option<String>;
    async fn save_session_graph(&self, graph: crate::SessionGraph);
    async fn load_session_graph(&self) -> Option<crate::SessionGraph>;
    async fn save_live_session_graph(&self, graph: crate::SessionGraph);
    async fn load_live_session_graph(&self) -> Option<crate::SessionGraph>;
    async fn clear_live_session_graph(&self);
    async fn save_session_meta(&self, meta: SessionMeta);
    async fn load_session_meta(&self) -> Option<SessionMeta>;

    async fn save_turn_checkpoint(&self, graph: crate::SessionGraph) {
        self.save_session_graph(graph).await;
        self.clear_live_session_graph().await;
    }

    async fn graph_copy_from_store(&self, source: &(dyn RuntimeStore + '_)) {
        if let Some(graph) = source.load_session_graph().await {
            self.save_session_graph(graph).await;
        }
    }
}

#[cfg(feature = "sqlite-store")]
fn encode_json<T: serde::Serialize>(value: &T) -> String {
    serde_json::to_string(value).expect("persisted state should serialize")
}

#[cfg(feature = "sqlite-store")]
impl Store {
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

        let graph_json: String = conn
            .query_row(
                "SELECT graph_json FROM session_graph WHERE singleton = 1",
                [],
                |row| row.get(0),
            )
            .unwrap_or_else(|_| "{\"nodes\":[],\"leaf_node_id\":null}".to_string());
        let graph = serde_json::from_str::<crate::SessionGraph>(&graph_json).unwrap_or_default();

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

    /// Store content, return its 12-char hex SHA-256 hash.
    pub fn store_archive(&self, content: &str) -> String {
        let hash = format!("{:x}", Sha256::digest(content.as_bytes()));
        let short = hash[..12].to_string();
        let conn = self.conn.lock().unwrap();
        if let Err(err) = conn.execute(
            "INSERT OR IGNORE INTO archive (hash, content) VALUES (?1, ?2)",
            params![short, content],
        ) {
            tracing::warn!(error = %err, hash = %short, "failed to persist archive content");
        }
        short
    }

    /// Retrieve archived content by short hash.
    pub fn get_archive(&self, hash: &str) -> Option<String> {
        let conn = self.conn.lock().unwrap();
        conn.query_row(
            "SELECT content FROM archive WHERE hash = ?1",
            params![hash],
            |row| row.get(0),
        )
        .ok()
    }

    /// Save or update the session runtime state in the store.
    pub fn save_live_session_graph(&self, graph: crate::SessionGraph) {
        let conn = self.conn.lock().unwrap();
        let graph_json = encode_json(&graph);
        conn.execute(
            "INSERT OR REPLACE INTO live_session_graph (singleton, graph_json)
             VALUES (1, ?1)",
            params![graph_json],
        )
        .unwrap();
    }

    pub fn load_live_session_graph(&self) -> Option<crate::SessionGraph> {
        let conn = self.conn.lock().unwrap();
        let graph_json: String = conn
            .query_row(
                "SELECT graph_json FROM live_session_graph WHERE singleton = 1",
                [],
                |row| row.get(0),
            )
            .ok()?;
        serde_json::from_str(&graph_json).ok()
    }

    pub fn clear_live_session_graph(&self) {
        let conn = self.conn.lock().unwrap();
        if let Err(err) = conn.execute("DELETE FROM live_session_graph WHERE singleton = 1", []) {
            tracing::warn!(error = %err, "failed to clear live session graph");
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

    pub fn save_session_graph(&self, graph: crate::SessionGraph) {
        let conn = self.conn.lock().unwrap();
        let graph_json = encode_json(&graph);
        if let Err(err) = conn.execute(
            "INSERT OR REPLACE INTO session_graph (singleton, graph_json)
             VALUES (1, ?1)",
            params![graph_json],
        ) {
            tracing::warn!(error = %err, "failed to persist session graph");
        }
    }

    pub fn load_session_graph(&self) -> Option<crate::SessionGraph> {
        let conn = self.conn.lock().unwrap();
        let graph_json: String = conn
            .query_row(
                "SELECT graph_json FROM session_graph WHERE singleton = 1",
                [],
                |row| row.get(0),
            )
            .ok()?;
        serde_json::from_str(&graph_json).ok()
    }

    pub fn graph_copy_from_store(&self, source: &Store) {
        if let Some(graph) = source.load_session_graph() {
            self.save_session_graph(graph);
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
    async fn store_archive(&self, content: &str) -> String {
        Self::store_archive(self, content)
    }

    async fn get_archive(&self, hash: &str) -> Option<String> {
        Self::get_archive(self, hash)
    }

    async fn save_session_graph(&self, graph: crate::SessionGraph) {
        Self::save_session_graph(self, graph);
    }

    async fn load_session_graph(&self) -> Option<crate::SessionGraph> {
        Self::load_session_graph(self)
    }

    async fn save_live_session_graph(&self, graph: crate::SessionGraph) {
        Self::save_live_session_graph(self, graph);
    }

    async fn load_live_session_graph(&self) -> Option<crate::SessionGraph> {
        Self::load_live_session_graph(self)
    }

    async fn clear_live_session_graph(&self) {
        Self::clear_live_session_graph(self);
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
        source.save_session_graph(crate::SessionGraph::from_projection(
            &[
                text_message("u0", MessageRole::User, "hello"),
                text_message("a0", MessageRole::Assistant, "world"),
            ],
            &[],
        ));

        let target = mem();
        target.graph_copy_from_store(&source);

        let graph = target.load_session_graph().expect("session graph");
        let messages = graph.project_messages();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].parts[0].content, "hello");
        assert_eq!(messages[1].parts[0].content, "world");
    }

    #[test]
    fn save_session_graph_rewrites_existing_snapshot() {
        let store = mem();
        store.save_session_graph(crate::SessionGraph::from_projection(
            &[text_message("u0", MessageRole::User, "old")],
            &[],
        ));

        store.save_session_graph(crate::SessionGraph::from_projection(
            &[text_message("u1", MessageRole::User, "updated")],
            &[],
        ));

        let graph = store.load_session_graph().expect("session graph");
        let messages = graph.project_messages();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].parts[0].content, "updated");
    }

    #[test]
    fn live_session_graph_rewrites_and_clears() {
        let store = mem();
        store.save_live_session_graph(crate::SessionGraph::from_projection(
            &[text_message("u0", MessageRole::User, "first")],
            &[],
        ));
        store.save_live_session_graph(crate::SessionGraph::from_projection(
            &[text_message("u1", MessageRole::User, "second")],
            &[],
        ));

        let graph = store.load_live_session_graph().expect("live session graph");
        let messages = graph.project_messages();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].parts[0].content, "second");

        store.clear_live_session_graph();
        assert!(store.load_live_session_graph().is_none());
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
        store.save_session_graph(crate::SessionGraph::from_projection(
            &[
                text_message("u0", MessageRole::User, "hello there"),
                text_message("a0", MessageRole::Assistant, "response"),
                text_message("u1", MessageRole::User, "follow up"),
            ],
            &[],
        ));

        let info = store.load_picker_info().expect("picker info");
        assert_eq!(info.session_id, "s1");
        assert_eq!(info.first_user_message, "hello there");
        assert_eq!(info.user_message_count, 2);
    }
}
