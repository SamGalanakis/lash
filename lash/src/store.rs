#[cfg(feature = "sqlite-store")]
use std::collections::BTreeSet;
#[cfg(feature = "sqlite-store")]
use std::path::Path;
#[cfg(feature = "sqlite-store")]
use std::sync::Mutex;
#[cfg(feature = "sqlite-store")]
use std::time::Duration;

#[cfg(feature = "sqlite-store")]
use rusqlite::{Connection, params};
#[cfg(feature = "sqlite-store")]
use sha2::{Digest, Sha256};

/// SQLite-backed store for archive, agent state, and history.
/// Single `Mutex<Connection>` serializes all access (same as the old `Mutex<HashMap>`).
#[cfg(feature = "sqlite-store")]
pub struct Store {
    conn: Mutex<Connection>,
}

#[cfg(feature = "sqlite-store")]
pub type SqliteStore = Store;

// ─── Schema ───

#[cfg(feature = "sqlite-store")]
const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS archive (
    hash    TEXT PRIMARY KEY,
    content TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS agents (
    agent_id              TEXT PRIMARY KEY,
    messages              TEXT NOT NULL DEFAULT '[]',
    tool_calls_json       TEXT NOT NULL DEFAULT '[]',
    ui_json               TEXT NOT NULL DEFAULT '{}',
    iteration             INTEGER NOT NULL DEFAULT 0,
    config_json           TEXT NOT NULL DEFAULT '{}',
    repl_snapshot         BLOB,
    input_tokens          INTEGER NOT NULL DEFAULT 0,
    output_tokens         INTEGER NOT NULL DEFAULT 0,
    cached_input_tokens   INTEGER NOT NULL DEFAULT 0,
    reasoning_tokens      INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS history_turns (
    agent_id         TEXT NOT NULL,
    turn_index       INTEGER NOT NULL,
    user_message     TEXT NOT NULL DEFAULT '',
    prose            TEXT NOT NULL DEFAULT '',
    code             TEXT NOT NULL DEFAULT '',
    output           TEXT NOT NULL DEFAULT '',
    error            TEXT,
    tool_calls_json  TEXT NOT NULL DEFAULT '[]',
    files_read_json  TEXT NOT NULL DEFAULT '[]',
    files_written_json TEXT NOT NULL DEFAULT '[]',
    PRIMARY KEY (agent_id, turn_index)
);
CREATE INDEX IF NOT EXISTS idx_history_turns_agent_turn
ON history_turns(agent_id, turn_index);

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
const SCHEMA_VERSION: i32 = 1;

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

/// Persisted agent state for snapshot/resume.
#[derive(Clone, Debug)]
pub struct AgentState {
    pub agent_id: String,
    pub messages_json: String,
    pub tool_calls_json: String,
    pub ui_json: String,
    pub iteration: i64,
    pub config_json: String,
    pub repl_snapshot: Option<Vec<u8>>,
    pub input_tokens: i64,
    pub output_tokens: i64,
    pub cached_input_tokens: i64,
    pub reasoning_tokens: i64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct HistoryTurnRecord {
    pub index: i64,
    pub user_message: String,
    pub prose: String,
    pub code: String,
    pub output: String,
    pub error: Option<String>,
    pub tool_calls: Vec<serde_json::Value>,
    pub files_read: Vec<String>,
    pub files_written: Vec<String>,
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

#[derive(Clone, Debug)]
pub struct TurnCheckpoint {
    pub agent_state: AgentState,
    pub latest_history_turn: Option<HistoryTurnRecord>,
}

/// Persistence backend for archived content, session snapshots, and turn history.
///
/// Implement this trait to back lash with a custom store such as Postgres while
/// keeping the built-in SQLite `Store` as the default first-party implementation.
#[async_trait::async_trait]
pub trait RuntimeStore: Send + Sync {
    async fn store_archive(&self, content: &str) -> String;
    async fn get_archive(&self, hash: &str) -> Option<String>;
    async fn save_agent_state(&self, state: AgentState);
    async fn load_agent_state(&self, agent_id: &str) -> Option<AgentState>;
    async fn history_upsert_turn(&self, agent_id: &str, turn: HistoryTurnRecord);
    async fn history_export(&self, agent_id: &str) -> Vec<HistoryTurnRecord>;
    async fn history_replace(&self, agent_id: &str, turns: Vec<HistoryTurnRecord>);
    async fn save_session_meta(&self, meta: SessionMeta);
    async fn update_agent_ui_state(&self, agent_id: &str, ui_json: &str);
    async fn load_session_meta(&self) -> Option<SessionMeta>;

    async fn save_turn_checkpoint(&self, checkpoint: TurnCheckpoint) {
        let TurnCheckpoint {
            agent_state,
            latest_history_turn,
        } = checkpoint;
        let agent_id = agent_state.agent_id.clone();
        self.save_agent_state(agent_state).await;
        if let Some(turn) = latest_history_turn {
            self.history_upsert_turn(&agent_id, turn).await;
        }
    }

    async fn history_copy(&self, source_agent_id: &str, target_agent_id: &str) {
        let turns = self.history_export(source_agent_id).await;
        self.history_replace(target_agent_id, turns).await;
    }

    async fn history_copy_from_store(
        &self,
        source: &(dyn RuntimeStore + '_),
        source_agent_id: &str,
        target_agent_id: &str,
    ) {
        let turns = source.history_export(source_agent_id).await;
        self.history_replace(target_agent_id, turns).await;
    }
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

    /// In-memory database (for child-session flows / tests).
    pub fn memory() -> rusqlite::Result<Self> {
        let conn = Connection::open_in_memory()?;
        apply_pragmas(&conn, StoreBacking::Memory)?;
        ensure_schema(&conn)?;
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    // ─── Archive operations ───

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

    // ─── Agent state operations (snapshot/resume) ───

    /// Save or update an agent's state in the store.
    pub fn save_agent_state(&self, state: AgentState) {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT OR REPLACE INTO agents (
                agent_id, messages, tool_calls_json, ui_json, iteration, config_json,
                repl_snapshot, input_tokens, output_tokens, cached_input_tokens,
                reasoning_tokens
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
            params![
                state.agent_id,
                state.messages_json,
                state.tool_calls_json,
                state.ui_json,
                state.iteration,
                state.config_json,
                state.repl_snapshot,
                state.input_tokens,
                state.output_tokens,
                state.cached_input_tokens,
                state.reasoning_tokens
            ],
        )
        .unwrap();
    }

    /// Load an agent's state by ID.
    pub fn load_agent_state(&self, agent_id: &str) -> Option<AgentState> {
        let conn = self.conn.lock().unwrap();
        conn.query_row(
            "SELECT agent_id, messages, tool_calls_json, ui_json, iteration, config_json, repl_snapshot, input_tokens, output_tokens, cached_input_tokens, reasoning_tokens
             FROM agents WHERE agent_id = ?1",
            params![agent_id],
            |row| {
                Ok(AgentState {
                    agent_id: row.get(0)?,
                    messages_json: row.get(1)?,
                    tool_calls_json: row.get(2)?,
                    ui_json: row.get(3)?,
                    iteration: row.get(4)?,
                    config_json: row.get(5)?,
                    repl_snapshot: row.get(6)?,
                    input_tokens: row.get(7)?,
                    output_tokens: row.get(8)?,
                    cached_input_tokens: row.get(9)?,
                    reasoning_tokens: row.get(10)?,
                })
            },
        )
        .ok()
    }

    // ─── History operations ───

    pub fn history_add_turn(&self, agent_id: &str, turn: &serde_json::Value) {
        let record = history_record_from_value(turn);
        self.history_upsert_turn(agent_id, record);
    }

    pub fn history_upsert_turn(&self, agent_id: &str, turn: HistoryTurnRecord) {
        let conn = self.conn.lock().unwrap();
        if let Err(err) = Self::history_upsert_turn_locked(&conn, agent_id, &turn) {
            tracing::warn!(
                error = %err,
                agent_id = agent_id,
                turn_index = turn.index,
                "failed to persist history turn"
            );
        }
    }

    pub fn history_export(&self, agent_id: &str) -> Vec<HistoryTurnRecord> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = match conn.prepare(
            "SELECT turn_index, user_message, prose, code, output, error, tool_calls_json, files_read_json, files_written_json
             FROM history_turns
             WHERE agent_id = ?1
             ORDER BY turn_index",
        ) {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };
        let rows = match stmt.query_map(params![agent_id], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, String>(4)?,
                row.get::<_, Option<String>>(5)?,
                row.get::<_, String>(6)?,
                row.get::<_, String>(7)?,
                row.get::<_, String>(8)?,
            ))
        }) {
            Ok(r) => r,
            Err(_) => return Vec::new(),
        };

        rows.filter_map(|r| r.ok())
            .map(
                |(
                    index,
                    user_message,
                    prose,
                    code,
                    output,
                    error,
                    tool_calls_json,
                    files_read_json,
                    files_written_json,
                )| {
                    let tool_calls = serde_json::from_str(&tool_calls_json).unwrap_or_default();
                    let files_read = serde_json::from_str(&files_read_json).unwrap_or_default();
                    let files_written =
                        serde_json::from_str(&files_written_json).unwrap_or_default();
                    HistoryTurnRecord {
                        index,
                        user_message,
                        prose,
                        code,
                        output,
                        error,
                        tool_calls,
                        files_read,
                        files_written,
                    }
                },
            )
            .collect()
    }

    pub fn history_replace(&self, agent_id: &str, turns: &[HistoryTurnRecord]) {
        let mut conn = self.conn.lock().unwrap();
        let tx = match conn.transaction() {
            Ok(tx) => tx,
            Err(err) => {
                tracing::warn!(
                    error = %err,
                    agent_id = agent_id,
                    "failed to open history transaction"
                );
                return;
            }
        };
        if let Err(err) = tx.execute(
            "DELETE FROM history_turns WHERE agent_id = ?1",
            params![agent_id],
        ) {
            tracing::warn!(
                error = %err,
                agent_id = agent_id,
                "failed to clear history before reload"
            );
        }
        for turn in turns {
            if let Err(err) = Self::history_upsert_turn_locked(&tx, agent_id, turn) {
                tracing::warn!(
                    error = %err,
                    agent_id = agent_id,
                    turn_index = turn.index,
                    "failed to reload history turn"
                );
            }
        }
        if let Err(err) = tx.commit() {
            tracing::warn!(
                error = %err,
                agent_id = agent_id,
                "failed to commit reloaded history"
            );
        }
    }

    pub fn history_load(&self, agent_id: &str, turns: &[serde_json::Value]) {
        let records = turns
            .iter()
            .map(history_record_from_value)
            .collect::<Vec<_>>();
        self.history_replace(agent_id, &records);
    }

    pub fn history_copy(&self, source_agent_id: &str, target_agent_id: &str) {
        let turns = self.history_export(source_agent_id);
        self.history_replace(target_agent_id, &turns);
    }

    pub fn history_copy_from_store(
        &self,
        source: &Store,
        source_agent_id: &str,
        target_agent_id: &str,
    ) {
        let turns = source.history_export(source_agent_id);
        self.history_replace(target_agent_id, &turns);
    }

    // ─── Session metadata ───

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

    pub fn update_agent_ui_state(&self, agent_id: &str, ui_json: &str) {
        let conn = self.conn.lock().unwrap();
        match conn.execute(
            "UPDATE agents SET ui_json = ?2 WHERE agent_id = ?1",
            params![agent_id, ui_json],
        ) {
            Ok(0) => {
                tracing::warn!(
                    agent_id = agent_id,
                    "skipped UI snapshot update because no agent row exists"
                );
            }
            Ok(_) => {}
            Err(err) => {
                tracing::warn!(
                    error = %err,
                    agent_id = agent_id,
                    "failed to persist UI snapshot update"
                );
            }
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

    // ─── Internal helpers (require caller to already hold the lock) ───

    fn history_upsert_turn_locked(
        conn: &Connection,
        agent_id: &str,
        turn: &HistoryTurnRecord,
    ) -> rusqlite::Result<()> {
        let (files_read, files_written) = derive_history_files(&turn.tool_calls);
        let tool_calls_json =
            serde_json::to_string(&turn.tool_calls).unwrap_or_else(|_| "[]".into());
        let files_read_json = serde_json::to_string(&files_read).unwrap_or_else(|_| "[]".into());
        let files_written_json =
            serde_json::to_string(&files_written).unwrap_or_else(|_| "[]".into());

        conn.execute(
            "INSERT OR REPLACE INTO history_turns
             (agent_id, turn_index, user_message, prose, code, output, error, tool_calls_json, files_read_json, files_written_json)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                agent_id,
                turn.index,
                turn.user_message,
                turn.prose,
                turn.code,
                turn.output,
                turn.error,
                tool_calls_json,
                files_read_json,
                files_written_json
            ],
        )?;
        Ok(())
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

    async fn save_agent_state(&self, state: AgentState) {
        Self::save_agent_state(self, state);
    }

    async fn load_agent_state(&self, agent_id: &str) -> Option<AgentState> {
        Self::load_agent_state(self, agent_id)
    }

    async fn history_upsert_turn(&self, agent_id: &str, turn: HistoryTurnRecord) {
        Self::history_upsert_turn(self, agent_id, turn);
    }

    async fn history_export(&self, agent_id: &str) -> Vec<HistoryTurnRecord> {
        Self::history_export(self, agent_id)
    }

    async fn history_replace(&self, agent_id: &str, turns: Vec<HistoryTurnRecord>) {
        Self::history_replace(self, agent_id, &turns);
    }

    async fn save_session_meta(&self, meta: SessionMeta) {
        Self::save_session_meta(self, meta);
    }

    async fn update_agent_ui_state(&self, agent_id: &str, ui_json: &str) {
        Self::update_agent_ui_state(self, agent_id, ui_json);
    }

    async fn load_session_meta(&self) -> Option<SessionMeta> {
        Self::load_session_meta(self)
    }
}

#[cfg(feature = "sqlite-store")]
fn derive_history_files(tool_calls: &[serde_json::Value]) -> (Vec<String>, Vec<String>) {
    let mut files_read = BTreeSet::new();
    let mut files_written = BTreeSet::new();
    for tc in tool_calls {
        let tool = tc.get("tool").and_then(|v| v.as_str()).unwrap_or_default();
        let path = tc
            .get("args")
            .and_then(|v| v.as_object())
            .and_then(|obj| obj.get("path"))
            .and_then(|v| v.as_str());
        if let Some(path) = path
            && matches!(tool, "read_file" | "glob" | "grep")
        {
            files_read.insert(path.to_string());
        }
        if tool == "apply_patch"
            && let Some(files) = tc
                .get("result")
                .and_then(|v| v.get("files"))
                .and_then(|v| v.as_array())
        {
            for file in files {
                if let Some(path) = file.get("path").and_then(|v| v.as_str()) {
                    files_written.insert(path.to_string());
                }
            }
        }
    }

    (
        files_read.into_iter().collect(),
        files_written.into_iter().collect(),
    )
}

#[cfg(feature = "sqlite-store")]
fn history_record_from_value(turn: &serde_json::Value) -> HistoryTurnRecord {
    HistoryTurnRecord {
        index: turn.get("index").and_then(|v| v.as_i64()).unwrap_or(0),
        user_message: turn
            .get("user_message")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string(),
        prose: turn
            .get("prose")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string(),
        code: turn
            .get("code")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string(),
        output: turn
            .get("output")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string(),
        error: turn
            .get("error")
            .and_then(|v| v.as_str())
            .map(str::to_string),
        tool_calls: turn
            .get("tool_calls")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default(),
        files_read: Vec::new(),
        files_written: Vec::new(),
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
    use rusqlite::Connection;

    fn mem() -> Store {
        Store::memory().unwrap()
    }

    // ── Archive ──

    #[test]
    fn archive_round_trip() {
        let s = mem();
        let hash = s.store_archive("hello world");
        assert_eq!(hash.len(), 12);
        assert_eq!(s.get_archive(&hash).unwrap(), "hello world");
    }

    #[test]
    fn archive_dedup() {
        let s = mem();
        let h1 = s.store_archive("same content");
        let h2 = s.store_archive("same content");
        assert_eq!(h1, h2);
    }

    #[test]
    fn archive_missing_hash() {
        let s = mem();
        assert!(s.get_archive("000000000000").is_none());
    }

    // ── Agent state ──

    #[test]
    fn agent_state_round_trip() {
        let s = mem();
        s.save_agent_state(AgentState {
            agent_id: "ag1".into(),
            messages_json: "[]".into(),
            tool_calls_json: "[]".into(),
            ui_json: "{}".into(),
            iteration: 5,
            config_json: "{}".into(),
            repl_snapshot: None,
            input_tokens: 100,
            output_tokens: 50,
            cached_input_tokens: 10,
            reasoning_tokens: 7,
        });
        let a = s.load_agent_state("ag1").unwrap();
        assert_eq!(a.agent_id, "ag1");
        assert_eq!(a.iteration, 5);
        assert_eq!(a.input_tokens, 100);
        assert_eq!(a.reasoning_tokens, 7);
        assert_eq!(a.ui_json, "{}");
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
    fn session_meta_round_trip() {
        let s = mem();
        s.save_session_meta(SessionMeta {
            session_id: "s1".into(),
            session_name: "demo".into(),
            created_at: "2026-03-25T10:00:00Z".into(),
            model: "gpt-5".into(),
            cwd: Some("/tmp/demo".into()),
            parent_session_id: None,
        });
        let meta = s.load_session_meta().expect("meta");
        assert_eq!(meta.session_id, "s1");
        assert_eq!(meta.session_name, "demo");
        assert_eq!(meta.model, "gpt-5");
        assert_eq!(meta.cwd.as_deref(), Some("/tmp/demo"));
    }

    #[test]
    fn history_copy_from_store_round_trip() {
        let source = mem();
        source.history_upsert_turn(
            "root",
            HistoryTurnRecord {
                index: 0,
                user_message: "hello".into(),
                prose: "world".into(),
                code: String::new(),
                output: String::new(),
                error: None,
                tool_calls: Vec::new(),
                files_read: Vec::new(),
                files_written: Vec::new(),
            },
        );

        let target = mem();
        target.history_copy_from_store(&source, "root", "root-copy");

        let turns = target.history_export("root-copy");
        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0].user_message, "hello");
        assert_eq!(turns[0].prose, "world");
    }

    #[test]
    fn update_agent_ui_state_rewrites_existing_snapshot() {
        let s = mem();
        s.save_agent_state(AgentState {
            agent_id: "ag1".into(),
            messages_json: "[]".into(),
            tool_calls_json: "[]".into(),
            ui_json: "{}".into(),
            iteration: 0,
            config_json: "{}".into(),
            repl_snapshot: None,
            input_tokens: 0,
            output_tokens: 0,
            cached_input_tokens: 0,
            reasoning_tokens: 0,
        });

        s.update_agent_ui_state("ag1", r#"{"blocks":["updated"]}"#);

        let state = s.load_agent_state("ag1").expect("agent state");
        assert_eq!(state.ui_json, r#"{"blocks":["updated"]}"#);
    }
}
