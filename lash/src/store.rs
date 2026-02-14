use std::path::Path;
use std::sync::Mutex;

use rusqlite::{Connection, params};
use sha2::{Digest, Sha256};

/// SQLite-backed store for archive (pruned message content) and tasks.
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

CREATE TABLE IF NOT EXISTS tasks (
    id          TEXT PRIMARY KEY,
    subject     TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    status      TEXT NOT NULL DEFAULT 'pending',
    priority    TEXT NOT NULL DEFAULT 'medium',
    active_form TEXT NOT NULL DEFAULT '',
    metadata    TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS task_deps (
    blocker_id  TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
    blocked_id  TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
    PRIMARY KEY (blocker_id, blocked_id)
);

CREATE TABLE IF NOT EXISTS counters (
    name  TEXT PRIMARY KEY,
    value INTEGER NOT NULL DEFAULT 0
);
INSERT OR IGNORE INTO counters (name, value) VALUES ('task_id', 0);

CREATE TABLE IF NOT EXISTS agents (
    agent_id            TEXT PRIMARY KEY,
    parent_id           TEXT,
    status              TEXT NOT NULL DEFAULT 'active',
    messages            TEXT NOT NULL DEFAULT '[]',
    iteration           INTEGER NOT NULL DEFAULT 0,
    config_json         TEXT NOT NULL DEFAULT '{}',
    dill_blob           BLOB,
    input_tokens        INTEGER NOT NULL DEFAULT 0,
    output_tokens       INTEGER NOT NULL DEFAULT 0,
    cached_input_tokens INTEGER NOT NULL DEFAULT 0
);
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

// ─── TaskEntry (public, used by tasks.rs tool) ───

#[derive(Clone, Debug, serde::Serialize)]
pub struct TaskEntry {
    pub id: String,
    pub subject: String,
    pub description: String,
    pub status: String,
    pub priority: String,
    pub active_form: String,
    pub blocks: Vec<String>,
    pub blocked_by: Vec<String>,
    pub metadata: serde_json::Value,
}

/// Persisted agent state for snapshot/resume.
#[derive(Clone, Debug)]
pub struct AgentState {
    pub agent_id: String,
    pub parent_id: Option<String>,
    pub status: String,
    pub messages_json: String,
    pub iteration: i64,
    pub config_json: String,
    pub dill_blob: Option<Vec<u8>>,
    pub input_tokens: i64,
    pub output_tokens: i64,
    pub cached_input_tokens: i64,
}

impl Store {
    /// Open (or create) a SQLite database at `path`.
    pub fn open(path: &Path) -> rusqlite::Result<Self> {
        let conn = Connection::open(path)?;
        apply_pragmas(&conn)?;
        conn.execute_batch(SCHEMA)?;
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    /// In-memory database (for sub-agents / tests).
    pub fn memory() -> rusqlite::Result<Self> {
        let conn = Connection::open_in_memory()?;
        apply_pragmas(&conn)?;
        conn.execute_batch(SCHEMA)?;
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

    // ─── Task operations ───

    /// Atomically increment the task counter and return the next hex ID.
    pub fn next_task_id(&self) -> String {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "UPDATE counters SET value = value + 1 WHERE name = 'task_id'",
            [],
        )
        .unwrap();
        let val: i64 = conn
            .query_row(
                "SELECT value FROM counters WHERE name = 'task_id'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        format!("{:04x}", val)
    }

    /// Create a new task. Returns the created TaskEntry.
    pub fn create_task(
        &self,
        id: &str,
        subject: &str,
        description: &str,
        priority: &str,
        active_form: &str,
        metadata: &serde_json::Value,
    ) -> TaskEntry {
        let meta_str = serde_json::to_string(metadata).unwrap_or_else(|_| "{}".into());
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO tasks (id, subject, description, status, priority, active_form, metadata)
             VALUES (?1, ?2, ?3, 'pending', ?4, ?5, ?6)",
            params![id, subject, description, priority, active_form, meta_str],
        )
        .unwrap();
        drop(conn);
        self.get_task(id).unwrap()
    }

    /// Get a single task by ID, including its dependency edges.
    pub fn get_task(&self, id: &str) -> Option<TaskEntry> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn
            .prepare(
                "SELECT id, subject, description, status, priority, active_form, metadata
                 FROM tasks WHERE id = ?1",
            )
            .ok()?;
        let entry = stmt
            .query_row(params![id], |row| {
                Ok(TaskEntryRow {
                    id: row.get(0)?,
                    subject: row.get(1)?,
                    description: row.get(2)?,
                    status: row.get(3)?,
                    priority: row.get(4)?,
                    active_form: row.get(5)?,
                    metadata_str: row.get(6)?,
                })
            })
            .ok()?;

        let blocks = self.get_blocks_locked(&conn, &entry.id);
        let blocked_by = self.get_blocked_by_locked(&conn, &entry.id);

        Some(TaskEntry {
            id: entry.id,
            subject: entry.subject,
            description: entry.description,
            status: entry.status,
            priority: entry.priority,
            active_form: entry.active_form,
            blocks,
            blocked_by,
            metadata: serde_json::from_str(&entry.metadata_str).unwrap_or_default(),
        })
    }

    /// List tasks with optional status and blocked filters.
    pub fn list_tasks(
        &self,
        status_filter: Option<&str>,
        blocked_filter: Option<bool>,
    ) -> Vec<TaskEntry> {
        let conn = self.conn.lock().unwrap();
        let mut query = "SELECT id, subject, description, status, priority, active_form, metadata FROM tasks".to_string();
        let mut conditions = Vec::new();
        if let Some(s) = status_filter {
            conditions.push(format!("status = '{}'", s.replace('\'', "''")));
        }
        if !conditions.is_empty() {
            query.push_str(" WHERE ");
            query.push_str(&conditions.join(" AND "));
        }
        query.push_str(" ORDER BY id");

        let mut stmt = match conn.prepare(&query) {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };

        let rows: Vec<TaskEntryRow> = stmt
            .query_map([], |row| {
                Ok(TaskEntryRow {
                    id: row.get(0)?,
                    subject: row.get(1)?,
                    description: row.get(2)?,
                    status: row.get(3)?,
                    priority: row.get(4)?,
                    active_form: row.get(5)?,
                    metadata_str: row.get(6)?,
                })
            })
            .ok()
            .map(|iter| iter.filter_map(|r| r.ok()).collect())
            .unwrap_or_default();

        let mut result = Vec::new();
        for entry in rows {
            let blocks = self.get_blocks_locked(&conn, &entry.id);
            let blocked_by = self.get_blocked_by_locked(&conn, &entry.id);
            let is_blocked = self.is_blocked_locked(&conn, &entry.id);

            if let Some(want_blocked) = blocked_filter {
                if want_blocked != is_blocked {
                    continue;
                }
            }

            result.push(TaskEntry {
                id: entry.id,
                subject: entry.subject,
                description: entry.description,
                status: entry.status,
                priority: entry.priority,
                active_form: entry.active_form,
                blocks,
                blocked_by,
                metadata: serde_json::from_str(&entry.metadata_str).unwrap_or_default(),
            });
        }
        result
    }

    /// Update fields on a task. Only non-None fields are changed.
    pub fn update_task(
        &self,
        id: &str,
        subject: Option<&str>,
        description: Option<&str>,
        status: Option<&str>,
        priority: Option<&str>,
        active_form: Option<&str>,
        metadata_merge: Option<&serde_json::Map<String, serde_json::Value>>,
    ) -> Option<TaskEntry> {
        let conn = self.conn.lock().unwrap();

        // Check exists
        let exists: bool = conn
            .query_row("SELECT 1 FROM tasks WHERE id = ?1", params![id], |_| Ok(()))
            .is_ok();
        if !exists {
            return None;
        }

        if let Some(v) = subject {
            conn.execute("UPDATE tasks SET subject = ?1 WHERE id = ?2", params![v, id]).unwrap();
        }
        if let Some(v) = description {
            conn.execute("UPDATE tasks SET description = ?1 WHERE id = ?2", params![v, id]).unwrap();
        }
        if let Some(v) = status {
            conn.execute("UPDATE tasks SET status = ?1 WHERE id = ?2", params![v, id]).unwrap();
        }
        if let Some(v) = priority {
            conn.execute("UPDATE tasks SET priority = ?1 WHERE id = ?2", params![v, id]).unwrap();
        }
        if let Some(v) = active_form {
            conn.execute("UPDATE tasks SET active_form = ?1 WHERE id = ?2", params![v, id]).unwrap();
        }
        if let Some(merge) = metadata_merge {
            let current_str: String = conn
                .query_row("SELECT metadata FROM tasks WHERE id = ?1", params![id], |row| row.get(0))
                .unwrap_or_else(|_| "{}".into());
            let mut current: serde_json::Map<String, serde_json::Value> =
                serde_json::from_str(&current_str).unwrap_or_default();
            for (k, v) in merge {
                if v.is_null() {
                    current.remove(k);
                } else {
                    current.insert(k.clone(), v.clone());
                }
            }
            let new_str = serde_json::to_string(&current).unwrap_or_else(|_| "{}".into());
            conn.execute("UPDATE tasks SET metadata = ?1 WHERE id = ?2", params![new_str, id]).unwrap();
        }

        drop(conn);
        self.get_task(id)
    }

    /// Delete a task and clean up its dependency edges.
    pub fn delete_task(&self, id: &str) -> bool {
        let conn = self.conn.lock().unwrap();
        // CASCADE on task_deps handles edge cleanup
        let changed = conn
            .execute("DELETE FROM tasks WHERE id = ?1", params![id])
            .unwrap_or(0);
        changed > 0
    }

    /// Add a dependency edge: `blocker_id` blocks `blocked_id`.
    pub fn add_dep(&self, blocker_id: &str, blocked_id: &str) {
        let conn = self.conn.lock().unwrap();
        let _ = conn.execute(
            "INSERT OR IGNORE INTO task_deps (blocker_id, blocked_id) VALUES (?1, ?2)",
            params![blocker_id, blocked_id],
        );
    }

    /// Remove a dependency edge.
    pub fn remove_dep(&self, blocker_id: &str, blocked_id: &str) {
        let conn = self.conn.lock().unwrap();
        let _ = conn.execute(
            "DELETE FROM task_deps WHERE blocker_id = ?1 AND blocked_id = ?2",
            params![blocker_id, blocked_id],
        );
    }

    /// Get a task summary (counts, high-priority, blocked).
    pub fn task_summary(&self) -> String {
        let conn = self.conn.lock().unwrap();

        let total: i64 = conn
            .query_row("SELECT COUNT(*) FROM tasks", [], |row| row.get(0))
            .unwrap_or(0);
        if total == 0 {
            return "No tasks.".into();
        }

        let pending: i64 = conn.query_row("SELECT COUNT(*) FROM tasks WHERE status = 'pending'", [], |row| row.get(0)).unwrap_or(0);
        let in_progress: i64 = conn.query_row("SELECT COUNT(*) FROM tasks WHERE status = 'in_progress'", [], |row| row.get(0)).unwrap_or(0);
        let completed: i64 = conn.query_row("SELECT COUNT(*) FROM tasks WHERE status = 'completed'", [], |row| row.get(0)).unwrap_or(0);
        let cancelled: i64 = conn.query_row("SELECT COUNT(*) FROM tasks WHERE status = 'cancelled'", [], |row| row.get(0)).unwrap_or(0);

        let mut lines = vec![format!(
            "Tasks: {} total  |  {} pending  {} in_progress  {} completed  {} cancelled",
            total, pending, in_progress, completed, cancelled
        )];

        // High priority (non-completed, non-cancelled)
        let mut stmt = conn
            .prepare("SELECT id, subject, status FROM tasks WHERE priority = 'high' AND status NOT IN ('completed', 'cancelled') ORDER BY id")
            .unwrap();
        let high: Vec<(String, String, String)> = stmt
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))
            .unwrap()
            .filter_map(|r| r.ok())
            .collect();
        if !high.is_empty() {
            lines.push(String::new());
            lines.push("High priority:".into());
            for (id, subject, status) in &high {
                let is_blocked = self.is_blocked_locked(&conn, id);
                let symbol = match status.as_str() {
                    "in_progress" => "~",
                    _ if is_blocked => "!",
                    _ => "+",
                };
                lines.push(format!("  {} {} '{}'", symbol, id, subject));
            }
        }

        // Blocked tasks
        let mut stmt = conn
            .prepare("SELECT id, subject FROM tasks WHERE status NOT IN ('completed', 'cancelled') ORDER BY id")
            .unwrap();
        let active: Vec<(String, String)> = stmt
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
            .unwrap()
            .filter_map(|r| r.ok())
            .collect();
        let mut blocked_lines = Vec::new();
        for (id, subject) in &active {
            let blocked_by = self.get_blocked_by_locked(&conn, id);
            // Filter to only non-completed/non-cancelled blockers
            let active_blockers: Vec<&String> = blocked_by.iter().filter(|bid| {
                conn.query_row(
                    "SELECT status FROM tasks WHERE id = ?1",
                    params![bid],
                    |row| row.get::<_, String>(0),
                )
                .map(|s| s != "completed" && s != "cancelled")
                .unwrap_or(false)
            }).collect();
            if !active_blockers.is_empty() {
                let blocker_ids: Vec<&str> = active_blockers.iter().map(|s| s.as_str()).collect();
                blocked_lines.push(format!("  {} '{}'  blocked by: {}", id, subject, blocker_ids.join(", ")));
            }
        }
        if !blocked_lines.is_empty() {
            lines.push(String::new());
            lines.push("Blocked:".into());
            lines.extend(blocked_lines);
        }

        lines.join("\n")
    }

    // ─── Agent state operations (snapshot/resume) ───

    /// Save or update an agent's state in the store.
    #[allow(clippy::too_many_arguments)]
    pub fn save_agent_state(
        &self,
        agent_id: &str,
        parent_id: Option<&str>,
        status: &str,
        messages_json: &str,
        iteration: i64,
        config_json: &str,
        dill_blob: Option<&[u8]>,
        input_tokens: i64,
        output_tokens: i64,
        cached_input_tokens: i64,
    ) {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT OR REPLACE INTO agents (agent_id, parent_id, status, messages, iteration, config_json, dill_blob, input_tokens, output_tokens, cached_input_tokens)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![agent_id, parent_id, status, messages_json, iteration, config_json, dill_blob, input_tokens, output_tokens, cached_input_tokens],
        )
        .unwrap();
    }

    /// Load an agent's state by ID.
    pub fn load_agent_state(&self, agent_id: &str) -> Option<AgentState> {
        let conn = self.conn.lock().unwrap();
        conn.query_row(
            "SELECT agent_id, parent_id, status, messages, iteration, config_json, dill_blob, input_tokens, output_tokens, cached_input_tokens
             FROM agents WHERE agent_id = ?1",
            params![agent_id],
            |row| {
                Ok(AgentState {
                    agent_id: row.get(0)?,
                    parent_id: row.get(1)?,
                    status: row.get(2)?,
                    messages_json: row.get(3)?,
                    iteration: row.get(4)?,
                    config_json: row.get(5)?,
                    dill_blob: row.get(6)?,
                    input_tokens: row.get(7)?,
                    output_tokens: row.get(8)?,
                    cached_input_tokens: row.get(9)?,
                })
            },
        )
        .ok()
    }

    /// List all active agents, optionally filtered by parent_id.
    pub fn list_active_agents(&self, parent_id: Option<&str>) -> Vec<AgentState> {
        let conn = self.conn.lock().unwrap();
        let mut results = Vec::new();
        let query = match parent_id {
            Some(_) => {
                "SELECT agent_id, parent_id, status, messages, iteration, config_json, dill_blob, input_tokens, output_tokens, cached_input_tokens
                 FROM agents WHERE status = 'active' AND parent_id = ?1 ORDER BY agent_id"
            }
            None => {
                "SELECT agent_id, parent_id, status, messages, iteration, config_json, dill_blob, input_tokens, output_tokens, cached_input_tokens
                 FROM agents WHERE status = 'active' AND parent_id IS NULL ORDER BY agent_id"
            }
        };
        let mut stmt = match conn.prepare(query) {
            Ok(s) => s,
            Err(_) => return results,
        };
        let row_mapper = |row: &rusqlite::Row| -> rusqlite::Result<AgentState> {
            Ok(AgentState {
                agent_id: row.get(0)?,
                parent_id: row.get(1)?,
                status: row.get(2)?,
                messages_json: row.get(3)?,
                iteration: row.get(4)?,
                config_json: row.get(5)?,
                dill_blob: row.get(6)?,
                input_tokens: row.get(7)?,
                output_tokens: row.get(8)?,
                cached_input_tokens: row.get(9)?,
            })
        };
        let iter = if let Some(pid) = parent_id {
            stmt.query_map(params![pid], row_mapper)
        } else {
            stmt.query_map([], row_mapper)
        };
        if let Ok(rows) = iter {
            for row in rows.flatten() {
                results.push(row);
            }
        }
        results
    }

    /// Mark an agent as done.
    pub fn mark_agent_done(&self, agent_id: &str) {
        let conn = self.conn.lock().unwrap();
        let _ = conn.execute(
            "UPDATE agents SET status = 'done' WHERE agent_id = ?1",
            params![agent_id],
        );
    }

    // ─── Internal helpers (require caller to already hold the lock) ───

    /// Tasks that `id` blocks (id is the blocker).
    fn get_blocks_locked(&self, conn: &Connection, id: &str) -> Vec<String> {
        let mut stmt = conn
            .prepare("SELECT blocked_id FROM task_deps WHERE blocker_id = ?1 ORDER BY blocked_id")
            .unwrap();
        stmt.query_map(params![id], |row| row.get(0))
            .unwrap()
            .filter_map(|r| r.ok())
            .collect()
    }

    /// Tasks that block `id` (id is the blocked).
    fn get_blocked_by_locked(&self, conn: &Connection, id: &str) -> Vec<String> {
        let mut stmt = conn
            .prepare("SELECT blocker_id FROM task_deps WHERE blocked_id = ?1 ORDER BY blocker_id")
            .unwrap();
        stmt.query_map(params![id], |row| row.get(0))
            .unwrap()
            .filter_map(|r| r.ok())
            .collect()
    }

    /// Is this task blocked by any non-completed/non-cancelled task?
    fn is_blocked_locked(&self, conn: &Connection, id: &str) -> bool {
        conn.query_row(
            "SELECT 1 FROM task_deps d
             JOIN tasks t ON t.id = d.blocker_id
             WHERE d.blocked_id = ?1
               AND t.status NOT IN ('completed', 'cancelled')
             LIMIT 1",
            params![id],
            |_| Ok(()),
        )
        .is_ok()
    }
}

/// Internal row type for reading from SQLite before adding dep info.
struct TaskEntryRow {
    id: String,
    subject: String,
    description: String,
    status: String,
    priority: String,
    active_form: String,
    metadata_str: String,
}
