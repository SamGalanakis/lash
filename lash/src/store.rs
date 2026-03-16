use std::path::Path;
use std::sync::Mutex;

use rusqlite::{Connection, params};
use sha2::{Digest, Sha256};

/// SQLite-backed store for archive, agent state, and history.
/// Single `Mutex<Connection>` serializes all access (same as the old `Mutex<HashMap>`).
pub struct Store {
    conn: Mutex<Connection>,
}

// ─── Schema ───

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS archive (
    hash    TEXT PRIMARY KEY,
    content TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS agents (
    agent_id            TEXT PRIMARY KEY,
    parent_id           TEXT,
    status              TEXT NOT NULL DEFAULT 'active',
    messages            TEXT NOT NULL DEFAULT '[]',
    iteration           INTEGER NOT NULL DEFAULT 0,
    config_json         TEXT NOT NULL DEFAULT '{}',
    repl_snapshot       BLOB,
    input_tokens        INTEGER NOT NULL DEFAULT 0,
    output_tokens       INTEGER NOT NULL DEFAULT 0,
    cached_input_tokens INTEGER NOT NULL DEFAULT 0,
    reasoning_tokens    INTEGER NOT NULL DEFAULT 0
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

";

fn apply_pragmas(conn: &Connection) -> rusqlite::Result<()> {
    conn.execute_batch(
        "PRAGMA journal_mode = WAL;
         PRAGMA synchronous = NORMAL;
         PRAGMA busy_timeout = 5000;
         PRAGMA foreign_keys = ON;
         PRAGMA cache_size = -2000;",
    )
}

/// Persisted agent state for snapshot/resume.
#[derive(Clone, Debug)]
pub struct AgentState {
    pub agent_id: String,
    pub messages_json: String,
    pub iteration: i64,
    pub config_json: String,
    pub repl_snapshot: Option<Vec<u8>>,
    pub input_tokens: i64,
    pub output_tokens: i64,
    pub cached_input_tokens: i64,
    pub reasoning_tokens: i64,
}

#[derive(Clone, Debug)]
pub struct AgentStateSave<'a> {
    pub agent_id: &'a str,
    pub messages_json: &'a str,
    pub iteration: i64,
    pub config_json: &'a str,
    pub repl_snapshot: Option<&'a [u8]>,
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

impl Store {
    /// Open (or create) a SQLite database at `path`.
    pub fn open(path: &Path) -> rusqlite::Result<Self> {
        let conn = Connection::open(path)?;
        apply_pragmas(&conn)?;
        conn.execute_batch(SCHEMA)?;
        apply_migrations(&conn)?;
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    /// In-memory database (for child-session flows / tests).
    pub fn memory() -> rusqlite::Result<Self> {
        let conn = Connection::open_in_memory()?;
        apply_pragmas(&conn)?;
        conn.execute_batch(SCHEMA)?;
        apply_migrations(&conn)?;
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
        let _ = conn.execute(
            "INSERT OR IGNORE INTO archive (hash, content) VALUES (?1, ?2)",
            params![short, content],
        );
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
    pub fn save_agent_state(&self, state: AgentStateSave<'_>) {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT OR REPLACE INTO agents (
                agent_id, messages, iteration, config_json, repl_snapshot,
                input_tokens, output_tokens, cached_input_tokens, reasoning_tokens
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                state.agent_id,
                state.messages_json,
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
            "SELECT agent_id, messages, iteration, config_json, repl_snapshot, input_tokens, output_tokens, cached_input_tokens, reasoning_tokens
             FROM agents WHERE agent_id = ?1",
            params![agent_id],
            |row| {
                Ok(AgentState {
                    agent_id: row.get(0)?,
                    messages_json: row.get(1)?,
                    iteration: row.get(2)?,
                    config_json: row.get(3)?,
                    repl_snapshot: row.get(4)?,
                    input_tokens: row.get(5)?,
                    output_tokens: row.get(6)?,
                    cached_input_tokens: row.get(7)?,
                    reasoning_tokens: row.get(8)?,
                })
            },
        )
        .ok()
    }

    // ─── History operations ───

    pub fn history_add_turn(&self, agent_id: &str, turn: &serde_json::Value) {
        let conn = self.conn.lock().unwrap();
        Self::history_add_turn_locked(&conn, agent_id, turn);
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

    pub fn history_load(&self, agent_id: &str, turns: &[serde_json::Value]) {
        let mut conn = self.conn.lock().unwrap();
        let tx = match conn.transaction() {
            Ok(tx) => tx,
            Err(_) => return,
        };
        let _ = tx.execute(
            "DELETE FROM history_turns WHERE agent_id = ?1",
            params![agent_id],
        );
        for turn in turns {
            Self::history_add_turn_locked(&tx, agent_id, turn);
        }
        let _ = tx.commit();
    }

    // ─── Internal helpers (require caller to already hold the lock) ───

    fn history_add_turn_locked(conn: &Connection, agent_id: &str, turn: &serde_json::Value) {
        let index = turn.get("index").and_then(|v| v.as_i64()).unwrap_or(0);
        let user_message = turn
            .get("user_message")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        let prose = turn
            .get("prose")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        let code = turn
            .get("code")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        let output = turn
            .get("output")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        let error = turn
            .get("error")
            .and_then(|v| v.as_str())
            .map(str::to_string);
        let tool_calls = turn
            .get("tool_calls")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();

        let mut files_read = std::collections::BTreeSet::new();
        let mut files_written = std::collections::BTreeSet::new();
        for tc in &tool_calls {
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

        let tool_calls_json = serde_json::to_string(&tool_calls).unwrap_or_else(|_| "[]".into());
        let files_read_json = serde_json::to_string(&files_read.into_iter().collect::<Vec<_>>())
            .unwrap_or_else(|_| "[]".into());
        let files_written_json =
            serde_json::to_string(&files_written.into_iter().collect::<Vec<_>>())
                .unwrap_or_else(|_| "[]".into());

        let _ = conn.execute(
            "INSERT OR REPLACE INTO history_turns
             (agent_id, turn_index, user_message, prose, code, output, error, tool_calls_json, files_read_json, files_written_json)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                agent_id,
                index,
                user_message,
                prose,
                code,
                output,
                error,
                tool_calls_json,
                files_read_json,
                files_written_json
            ],
        );
    }
}

fn apply_migrations(conn: &Connection) -> rusqlite::Result<()> {
    match conn.execute(
        "ALTER TABLE agents ADD COLUMN reasoning_tokens INTEGER NOT NULL DEFAULT 0",
        [],
    ) {
        Ok(_) => {}
        Err(rusqlite::Error::SqliteFailure(_, Some(message)))
            if message.contains("duplicate column name") => {}
        Err(err) => return Err(err),
    }
    Ok(())
}

#[cfg(test)]
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
        s.save_agent_state(AgentStateSave {
            agent_id: "ag1",
            messages_json: "[]",
            iteration: 5,
            config_json: "{}",
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
    }

    #[test]
    fn open_migrates_old_agents_schema_with_missing_reasoning_tokens() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("legacy.db");
        let conn = Connection::open(&path).unwrap();
        conn.execute_batch(
            "
            CREATE TABLE agents (
                agent_id            TEXT PRIMARY KEY,
                parent_id           TEXT,
                status              TEXT NOT NULL DEFAULT 'active',
                messages            TEXT NOT NULL DEFAULT '[]',
                iteration           INTEGER NOT NULL DEFAULT 0,
                config_json         TEXT NOT NULL DEFAULT '{}',
                repl_snapshot       BLOB,
                input_tokens        INTEGER NOT NULL DEFAULT 0,
                output_tokens       INTEGER NOT NULL DEFAULT 0,
                cached_input_tokens INTEGER NOT NULL DEFAULT 0
            );
            INSERT INTO agents (
                agent_id, parent_id, status, messages, iteration, config_json, repl_snapshot,
                input_tokens, output_tokens, cached_input_tokens
            ) VALUES ('root', NULL, 'active', '[]', 4, '{}', NULL, 100, 25, 5);
            ",
        )
        .unwrap();
        drop(conn);

        let store = Store::open(&path).unwrap();
        let state = store.load_agent_state("root").expect("agent");
        assert_eq!(state.input_tokens, 100);
        assert_eq!(state.output_tokens, 25);
        assert_eq!(state.cached_input_tokens, 5);
        assert_eq!(state.reasoning_tokens, 0);
    }
}
