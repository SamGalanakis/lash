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

/// SQLite-backed store for archive, runtime state, and canonical transcript rows.
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

CREATE TABLE IF NOT EXISTS session_state (
    singleton             INTEGER PRIMARY KEY CHECK (singleton = 1),
    iteration             INTEGER NOT NULL DEFAULT 0,
    config_json           TEXT NOT NULL DEFAULT '{}',
    repl_snapshot         BLOB,
    input_tokens          INTEGER NOT NULL DEFAULT 0,
    output_tokens         INTEGER NOT NULL DEFAULT 0,
    cached_input_tokens   INTEGER NOT NULL DEFAULT 0,
    reasoning_tokens      INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS live_session_snapshot (
    singleton      INTEGER PRIMARY KEY CHECK (singleton = 1),
    snapshot_json  TEXT NOT NULL DEFAULT '{}',
    repl_snapshot  BLOB
);

CREATE TABLE IF NOT EXISTS transcript_entries (
    keyspace       TEXT NOT NULL,
    stable_key     TEXT NOT NULL,
    sort_index     INTEGER NOT NULL DEFAULT 0,
    message_role   TEXT,
    payload_json   TEXT NOT NULL,
    search_text    TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (keyspace, stable_key)
);
CREATE INDEX IF NOT EXISTS idx_transcript_entries_keyspace_sort
ON transcript_entries(keyspace, sort_index, stable_key);
CREATE INDEX IF NOT EXISTS idx_transcript_entries_picker
ON transcript_entries(keyspace, message_role, sort_index);

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
const SCHEMA_VERSION: i32 = 5;

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

/// Persisted session runtime state for snapshot/resume.
#[derive(Clone, Debug)]
pub struct SessionState {
    pub iteration: i64,
    pub config_json: String,
    pub repl_snapshot: Option<Vec<u8>>,
    pub input_tokens: i64,
    pub output_tokens: i64,
    pub cached_input_tokens: i64,
    pub reasoning_tokens: i64,
}

#[derive(Clone, Debug)]
pub struct LiveSessionSnapshot {
    pub snapshot_json: String,
    pub repl_snapshot: Option<Vec<u8>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TranscriptEntry {
    pub keyspace: String,
    pub stable_key: String,
    pub sort_index: i64,
    pub message_role: Option<String>,
    pub payload_json: String,
    pub search_text: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TranscriptEntryPayload {
    pub stable_key: String,
    pub sort_index: i64,
    pub message_role: Option<String>,
    pub payload_json: String,
    pub search_text: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TranscriptKeyspace {
    pub keyspace: String,
    pub entries: Vec<TranscriptEntryPayload>,
}

impl TranscriptEntryPayload {
    pub fn new(stable_key: impl Into<String>, sort_index: i64, payload_json: String) -> Self {
        Self {
            stable_key: stable_key.into(),
            sort_index,
            message_role: None,
            payload_json,
            search_text: String::new(),
        }
    }

    pub fn with_message_role(mut self, message_role: Option<String>) -> Self {
        self.message_role = message_role;
        self
    }

    pub fn with_search_text(mut self, search_text: impl Into<String>) -> Self {
        self.search_text = search_text.into();
        self
    }
}

impl TranscriptKeyspace {
    pub fn new(keyspace: impl Into<String>, entries: Vec<TranscriptEntryPayload>) -> Self {
        Self {
            keyspace: keyspace.into(),
            entries,
        }
    }
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

#[derive(Clone, Debug)]
pub struct TurnCheckpoint {
    pub session_state: SessionState,
    pub transcript_keyspaces: Vec<TranscriptKeyspace>,
}

pub const TRANSCRIPT_KEYSPACE_MESSAGES: &str = "message";
pub const TRANSCRIPT_KEYSPACE_TOOL_CALLS: &str = "tool_call";

/// Persistence backend for archived content, runtime snapshots, and transcript rows.
#[async_trait::async_trait]
pub trait RuntimeStore: Send + Sync {
    async fn store_archive(&self, content: &str) -> String;
    async fn get_archive(&self, hash: &str) -> Option<String>;
    async fn save_session_state(&self, state: SessionState);
    async fn load_session_state(&self) -> Option<SessionState>;
    async fn save_live_session_snapshot(&self, snapshot: LiveSessionSnapshot);
    async fn load_live_session_snapshot(&self) -> Option<LiveSessionSnapshot>;
    async fn clear_live_session_snapshot(&self);
    async fn transcript_clear(&self);
    async fn transcript_load(&self) -> Vec<TranscriptEntry>;
    async fn transcript_replace_keyspaces(&self, keyspaces: Vec<TranscriptKeyspace>);
    async fn save_session_meta(&self, meta: SessionMeta);
    async fn load_session_meta(&self) -> Option<SessionMeta>;

    async fn save_turn_checkpoint(&self, checkpoint: TurnCheckpoint) {
        let TurnCheckpoint {
            session_state,
            transcript_keyspaces,
        } = checkpoint;
        self.save_session_state(session_state).await;
        self.transcript_replace_keyspaces(transcript_keyspaces)
            .await;
        self.clear_live_session_snapshot().await;
    }

    async fn transcript_replace_all(&self, entries: Vec<TranscriptEntry>) {
        self.transcript_clear().await;
        self.transcript_replace_keyspaces(group_transcript_entries(entries))
            .await;
    }

    async fn transcript_copy_from_store(&self, source: &(dyn RuntimeStore + '_)) {
        let entries = source.transcript_load().await;
        self.transcript_replace_all(entries).await;
    }
}

pub fn group_transcript_entries(entries: Vec<TranscriptEntry>) -> Vec<TranscriptKeyspace> {
    let mut grouped = std::collections::BTreeMap::<String, Vec<TranscriptEntryPayload>>::new();
    for entry in entries {
        grouped
            .entry(entry.keyspace)
            .or_default()
            .push(TranscriptEntryPayload {
                stable_key: entry.stable_key,
                sort_index: entry.sort_index,
                message_role: entry.message_role,
                payload_json: entry.payload_json,
                search_text: entry.search_text,
            });
    }
    grouped
        .into_iter()
        .map(|(keyspace, mut entries)| {
            entries.sort_by(|a, b| {
                a.sort_index
                    .cmp(&b.sort_index)
                    .then_with(|| a.stable_key.cmp(&b.stable_key))
            });
            TranscriptKeyspace { keyspace, entries }
        })
        .collect()
}

pub fn semantic_transcript_keyspaces(
    messages: &[crate::Message],
    tool_calls: &[crate::ToolCallRecord],
) -> Vec<TranscriptKeyspace> {
    let message_entries = messages
        .iter()
        .enumerate()
        .map(|(idx, message)| {
            let payload_json =
                serde_json::to_string(message).unwrap_or_else(|_| "null".to_string());
            TranscriptEntryPayload::new(idx.to_string(), idx as i64, payload_json)
                .with_message_role(Some(transcript_message_role(message)))
                .with_search_text(transcript_message_search_text(message))
        })
        .collect();
    let tool_call_entries = tool_calls
        .iter()
        .enumerate()
        .map(|(idx, record)| {
            let stable_key = record
                .call_id
                .clone()
                .filter(|call_id| !call_id.is_empty())
                .unwrap_or_else(|| idx.to_string());
            let payload_json = serde_json::to_string(record).unwrap_or_else(|_| "null".to_string());
            TranscriptEntryPayload::new(stable_key, idx as i64, payload_json)
        })
        .collect();
    vec![
        TranscriptKeyspace::new(TRANSCRIPT_KEYSPACE_MESSAGES, message_entries),
        TranscriptKeyspace::new(TRANSCRIPT_KEYSPACE_TOOL_CALLS, tool_call_entries),
    ]
}

pub fn transcript_messages(entries: &[TranscriptEntry]) -> Vec<crate::Message> {
    entries
        .iter()
        .filter(|entry| entry.keyspace == TRANSCRIPT_KEYSPACE_MESSAGES)
        .filter_map(|entry| serde_json::from_str(&entry.payload_json).ok())
        .collect()
}

pub fn transcript_tool_calls(entries: &[TranscriptEntry]) -> Vec<crate::ToolCallRecord> {
    entries
        .iter()
        .filter(|entry| entry.keyspace == TRANSCRIPT_KEYSPACE_TOOL_CALLS)
        .filter_map(|entry| serde_json::from_str(&entry.payload_json).ok())
        .collect()
}

fn transcript_message_role(message: &crate::Message) -> String {
    match message.role {
        crate::MessageRole::User => "user".to_string(),
        crate::MessageRole::Assistant => "assistant".to_string(),
        crate::MessageRole::System => "system".to_string(),
    }
}

fn transcript_message_search_text(message: &crate::Message) -> String {
    message
        .parts
        .iter()
        .filter_map(|part| match part.kind {
            crate::PartKind::ToolCall | crate::PartKind::ToolResult => None,
            crate::PartKind::Image => Some("[Image attached]".to_string()),
            _ => (!part.content.trim().is_empty()).then(|| part.content.clone()),
        })
        .collect::<Vec<_>>()
        .join("\n\n")
        .trim()
        .to_string()
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

    /// Fast picker info: session_meta + first user prompt + user turn count from transcript rows.
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

        let turn_count: usize = conn
            .query_row(
                "SELECT COUNT(*) FROM transcript_entries
                 WHERE keyspace = 'message'
                   AND message_role = 'user'",
                [],
                |row| row.get(0),
            )
            .unwrap_or(0);

        let first_msg: String = conn
            .query_row(
                "SELECT search_text FROM transcript_entries
                 WHERE keyspace = 'message'
                   AND message_role = 'user'
                 ORDER BY sort_index ASC, stable_key ASC
                 LIMIT 1",
                [],
                |row| row.get(0),
            )
            .unwrap_or_default();

        Some(SessionPickerInfo {
            session_id: meta.0,
            cwd: meta.1,
            parent_session_id: meta.2,
            first_user_message: first_msg,
            user_message_count: turn_count,
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
    pub fn save_session_state(&self, state: SessionState) {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT OR REPLACE INTO session_state (
                singleton, iteration, config_json, repl_snapshot,
                input_tokens, output_tokens, cached_input_tokens, reasoning_tokens
             ) VALUES (1, ?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
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

    /// Load the persisted session runtime state.
    pub fn load_session_state(&self) -> Option<SessionState> {
        let conn = self.conn.lock().unwrap();
        conn.query_row(
            "SELECT iteration, config_json, repl_snapshot, input_tokens, output_tokens, cached_input_tokens, reasoning_tokens
             FROM session_state WHERE singleton = 1",
            [],
            |row| {
                Ok(SessionState {
                    iteration: row.get(0)?,
                    config_json: row.get(1)?,
                    repl_snapshot: row.get(2)?,
                    input_tokens: row.get(3)?,
                    output_tokens: row.get(4)?,
                    cached_input_tokens: row.get(5)?,
                    reasoning_tokens: row.get(6)?,
                })
            },
        )
        .ok()
    }

    pub fn save_live_session_snapshot(&self, snapshot: LiveSessionSnapshot) {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT OR REPLACE INTO live_session_snapshot (
                singleton, snapshot_json, repl_snapshot
             ) VALUES (1, ?1, ?2)",
            params![snapshot.snapshot_json, snapshot.repl_snapshot],
        )
        .unwrap();
    }

    pub fn load_live_session_snapshot(&self) -> Option<LiveSessionSnapshot> {
        let conn = self.conn.lock().unwrap();
        conn.query_row(
            "SELECT snapshot_json, repl_snapshot
             FROM live_session_snapshot WHERE singleton = 1",
            [],
            |row| {
                Ok(LiveSessionSnapshot {
                    snapshot_json: row.get(0)?,
                    repl_snapshot: row.get(1)?,
                })
            },
        )
        .ok()
    }

    pub fn clear_live_session_snapshot(&self) {
        let conn = self.conn.lock().unwrap();
        if let Err(err) = conn.execute("DELETE FROM live_session_snapshot WHERE singleton = 1", [])
        {
            tracing::warn!(error = %err, "failed to clear live session snapshot");
        }
    }

    pub fn transcript_clear(&self) {
        let conn = self.conn.lock().unwrap();
        if let Err(err) = conn.execute("DELETE FROM transcript_entries", []) {
            tracing::warn!(error = %err, "failed to clear transcript rows");
        }
    }

    pub fn transcript_load(&self) -> Vec<TranscriptEntry> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = match conn.prepare(
            "SELECT keyspace, stable_key, sort_index, message_role, payload_json, search_text
             FROM transcript_entries
             ORDER BY keyspace, sort_index, stable_key",
        ) {
            Ok(stmt) => stmt,
            Err(_) => return Vec::new(),
        };
        let rows = match stmt.query_map([], |row| {
            Ok(TranscriptEntry {
                keyspace: row.get(0)?,
                stable_key: row.get(1)?,
                sort_index: row.get(2)?,
                message_role: row.get(3)?,
                payload_json: row.get(4)?,
                search_text: row.get(5)?,
            })
        }) {
            Ok(rows) => rows,
            Err(_) => return Vec::new(),
        };

        rows.filter_map(|row| row.ok()).collect()
    }

    pub fn transcript_replace_keyspaces(&self, keyspaces: &[TranscriptKeyspace]) {
        let mut conn = self.conn.lock().unwrap();
        let tx = match conn.transaction() {
            Ok(tx) => tx,
            Err(err) => {
                tracing::warn!(error = %err, "failed to open transcript transaction");
                return;
            }
        };

        for keyspace in keyspaces {
            if let Err(err) = tx.execute(
                "DELETE FROM transcript_entries
                 WHERE keyspace = ?1",
                params![keyspace.keyspace],
            ) {
                tracing::warn!(
                    error = %err,
                    keyspace = keyspace.keyspace,
                    "failed to clear transcript keyspace"
                );
                continue;
            }

            for entry in &keyspace.entries {
                if let Err(err) = tx.execute(
                    "INSERT INTO transcript_entries (
                        keyspace, stable_key, sort_index, message_role, payload_json, search_text
                     ) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                    params![
                        keyspace.keyspace,
                        entry.stable_key,
                        entry.sort_index,
                        entry.message_role,
                        entry.payload_json,
                        entry.search_text
                    ],
                ) {
                    tracing::warn!(
                        error = %err,
                        keyspace = keyspace.keyspace,
                        stable_key = entry.stable_key,
                        "failed to persist transcript entry"
                    );
                }
            }
        }

        if let Err(err) = tx.commit() {
            tracing::warn!(error = %err, "failed to commit transcript transaction");
        }
    }

    pub fn transcript_copy_from_store(&self, source: &Store) {
        let entries = source.transcript_load();
        self.transcript_clear();
        self.transcript_replace_keyspaces(&group_transcript_entries(entries));
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

    async fn save_session_state(&self, state: SessionState) {
        Self::save_session_state(self, state);
    }

    async fn load_session_state(&self) -> Option<SessionState> {
        Self::load_session_state(self)
    }

    async fn save_live_session_snapshot(&self, snapshot: LiveSessionSnapshot) {
        Self::save_live_session_snapshot(self, snapshot);
    }

    async fn load_live_session_snapshot(&self) -> Option<LiveSessionSnapshot> {
        Self::load_live_session_snapshot(self)
    }

    async fn clear_live_session_snapshot(&self) {
        Self::clear_live_session_snapshot(self);
    }

    async fn transcript_clear(&self) {
        Self::transcript_clear(self);
    }

    async fn transcript_load(&self) -> Vec<TranscriptEntry> {
        Self::transcript_load(self)
    }

    async fn transcript_replace_keyspaces(&self, keyspaces: Vec<TranscriptKeyspace>) {
        Self::transcript_replace_keyspaces(self, &keyspaces);
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
    use rusqlite::Connection;

    fn mem() -> Store {
        Store::memory().unwrap()
    }

    fn transcript_entry(
        keyspace: &str,
        stable_key: &str,
        sort_index: i64,
        payload_json: &str,
    ) -> TranscriptEntry {
        TranscriptEntry {
            keyspace: keyspace.to_string(),
            stable_key: stable_key.to_string(),
            sort_index,
            message_role: None,
            payload_json: payload_json.to_string(),
            search_text: String::new(),
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
    fn transcript_copy_from_store_round_trip() {
        let source = mem();
        source.transcript_replace_keyspaces(&[TranscriptKeyspace::new(
            "message",
            vec![
                TranscriptEntryPayload::new("0", 0, r#"{"text":"hello"}"#.to_string())
                    .with_message_role(Some("user".to_string()))
                    .with_search_text("hello"),
                TranscriptEntryPayload::new("1", 1, r#"{"text":"world"}"#.to_string())
                    .with_message_role(Some("assistant".to_string())),
            ],
        )]);

        let target = mem();
        target.transcript_copy_from_store(&source);

        let entries = target.transcript_load();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].payload_json, r#"{"text":"hello"}"#);
        assert_eq!(entries[1].payload_json, r#"{"text":"world"}"#);
    }

    #[test]
    fn transcript_replace_keyspace_rewrites_existing_rows() {
        let store = mem();
        store.transcript_replace_keyspaces(&[TranscriptKeyspace::new(
            "runtime_state",
            vec![TranscriptEntryPayload::new(
                "state",
                0,
                r#"{"blocks":["old"]}"#.to_string(),
            )],
        )]);

        store.transcript_replace_keyspaces(&[TranscriptKeyspace::new(
            "runtime_state",
            vec![TranscriptEntryPayload::new(
                "state",
                0,
                r#"{"blocks":["updated"]}"#.to_string(),
            )],
        )]);

        let entries = store.transcript_load();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].payload_json, r#"{"blocks":["updated"]}"#);
    }

    #[test]
    fn save_session_state_rewrites_existing_snapshot() {
        let store = mem();
        store.save_session_state(SessionState {
            iteration: 0,
            config_json: "{}".into(),
            repl_snapshot: None,
            input_tokens: 0,
            output_tokens: 0,
            cached_input_tokens: 0,
            reasoning_tokens: 0,
        });

        store.save_session_state(SessionState {
            iteration: 7,
            config_json: r#"{"mode":"updated"}"#.into(),
            repl_snapshot: None,
            input_tokens: 5,
            output_tokens: 2,
            cached_input_tokens: 1,
            reasoning_tokens: 0,
        });

        let state = store.load_session_state().expect("session state");
        assert_eq!(state.iteration, 7);
        assert_eq!(state.config_json, r#"{"mode":"updated"}"#);
        assert_eq!(state.input_tokens, 5);
    }

    #[test]
    fn live_session_snapshot_rewrites_and_clears() {
        let store = mem();
        store.save_live_session_snapshot(LiveSessionSnapshot {
            snapshot_json: r#"{"iteration":1}"#.into(),
            repl_snapshot: Some(vec![1, 2, 3]),
        });
        store.save_live_session_snapshot(LiveSessionSnapshot {
            snapshot_json: r#"{"iteration":2}"#.into(),
            repl_snapshot: None,
        });

        let snapshot = store.load_live_session_snapshot().expect("live snapshot");
        assert_eq!(snapshot.snapshot_json, r#"{"iteration":2}"#);
        assert_eq!(snapshot.repl_snapshot, None);

        store.clear_live_session_snapshot();
        assert!(store.load_live_session_snapshot().is_none());
    }

    #[test]
    fn load_picker_info_reads_message_transcript() {
        let store = mem();
        store.save_session_meta(SessionMeta {
            session_id: "s1".to_string(),
            session_name: "demo".to_string(),
            created_at: "2026-04-05T12:00:00Z".to_string(),
            model: "gpt-5".to_string(),
            cwd: Some("/tmp/demo".to_string()),
            parent_session_id: None,
        });
        store.transcript_replace_keyspaces(&[TranscriptKeyspace::new(
            "message",
            vec![
                TranscriptEntryPayload::new("0", 0, r#"{"id":"u0"}"#.to_string())
                    .with_message_role(Some("user".to_string()))
                    .with_search_text("hello there"),
                TranscriptEntryPayload::new("1", 1, r#"{"id":"a0"}"#.to_string())
                    .with_message_role(Some("assistant".to_string())),
                TranscriptEntryPayload::new("2", 2, r#"{"id":"u1"}"#.to_string())
                    .with_message_role(Some("user".to_string()))
                    .with_search_text("follow up"),
            ],
        )]);

        let info = store.load_picker_info().expect("picker info");
        assert_eq!(info.session_id, "s1");
        assert_eq!(info.first_user_message, "hello there");
        assert_eq!(info.user_message_count, 2);
    }

    #[test]
    fn group_transcript_entries_groups_by_keyspace_and_preserves_sort_order() {
        let grouped = group_transcript_entries(vec![
            transcript_entry("panel", "1", 1, r#"{"kind":"assistant"}"#),
            transcript_entry("message", "0", 0, r#"{"role":"user"}"#),
            transcript_entry("panel", "0", 0, r#"{"kind":"user"}"#),
        ]);

        assert_eq!(grouped.len(), 2);
        assert_eq!(grouped[0].keyspace, "message");
        assert_eq!(grouped[1].keyspace, "panel");
        assert_eq!(grouped[1].entries[0].stable_key, "0");
        assert_eq!(grouped[1].entries[1].stable_key, "1");
    }
}
