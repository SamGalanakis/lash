use super::*;

const SCHEMA_CHANGED_RETRY_ATTEMPTS: usize = 8;
const SCHEMA_CHANGED_RETRY_BASE_DELAY: Duration = Duration::from_millis(5);

pub(crate) async fn ensure_process_schema(conn: &Connection) -> turso::Result<()> {
    for attempt in 0..SCHEMA_CHANGED_RETRY_ATTEMPTS {
        match ensure_process_schema_once(conn).await {
            Err(err)
                if is_database_schema_changed(&err)
                    && attempt + 1 < SCHEMA_CHANGED_RETRY_ATTEMPTS =>
            {
                schema_changed_backoff(attempt);
            }
            result => return result,
        }
    }
    unreachable!("schema retry loop always returns")
}

async fn ensure_process_schema_once(conn: &Connection) -> turso::Result<()> {
    let user_version = pragma_user_version(conn).await?;
    if user_version == PROCESS_SCHEMA_VERSION {
        conn.execute_batch(PROCESS_SCHEMA).await?;
        return Ok(());
    }
    if user_version == 0 && !has_user_schema_objects(conn).await? {
        initialize_schema(conn, PROCESS_SCHEMA, PROCESS_SCHEMA_VERSION).await?;
        return Ok(());
    }
    Err(turso::Error::Misuse(unsupported_schema_message(
        "process registry",
        PROCESS_SCHEMA_VERSION,
        user_version,
    )))
}

pub(crate) async fn ensure_effect_schema(conn: &Connection) -> turso::Result<()> {
    for attempt in 0..SCHEMA_CHANGED_RETRY_ATTEMPTS {
        match ensure_effect_schema_once(conn).await {
            Err(err)
                if is_database_schema_changed(&err)
                    && attempt + 1 < SCHEMA_CHANGED_RETRY_ATTEMPTS =>
            {
                schema_changed_backoff(attempt);
            }
            result => return result,
        }
    }
    unreachable!("schema retry loop always returns")
}

async fn ensure_effect_schema_once(conn: &Connection) -> turso::Result<()> {
    let user_version = pragma_user_version(conn).await?;
    if user_version == EFFECT_SCHEMA_VERSION {
        conn.execute_batch(EFFECT_SCHEMA).await?;
        return Ok(());
    }
    if user_version == 0 && !has_user_schema_objects(conn).await? {
        initialize_schema(conn, EFFECT_SCHEMA, EFFECT_SCHEMA_VERSION).await?;
        return Ok(());
    }
    Err(turso::Error::Misuse(unsupported_schema_message(
        "effect replay",
        EFFECT_SCHEMA_VERSION,
        user_version,
    )))
}

/// Canonical Turso schema for a lash session database.
///
/// This is the *only* schema the store supports. Older session databases —
/// including any rolled forward through prior migration chains — must be
/// deleted before opening with this binary; [`ensure_schema`] rejects any
/// `PRAGMA user_version` that does not match [`SCHEMA_VERSION`] exactly. We
/// run with no on-the-fly migrations on purpose: lash's durable contract
/// lives one level up in the per-record `schema_version` stamps, not in
/// SQL DDL juggling.
pub(crate) const SCHEMA: &str = "
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
    seq        INTEGER PRIMARY KEY,
    node_id    TEXT NOT NULL UNIQUE,
    node_json  TEXT NOT NULL,
    tombstoned INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS usage_deltas (
    seq                  INTEGER PRIMARY KEY,
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

CREATE TABLE IF NOT EXISTS runtime_turn_commits (
    session_id        TEXT NOT NULL,
    turn_id           TEXT NOT NULL,
    turn_commit_hash  TEXT NOT NULL,
    result_json       TEXT NOT NULL,
    committed_at_ms   INTEGER NOT NULL,
    PRIMARY KEY (session_id, turn_id)
);

CREATE TABLE IF NOT EXISTS queued_work_batches (
    enqueue_seq       INTEGER PRIMARY KEY,
    batch_id          TEXT NOT NULL UNIQUE,
    session_id        TEXT NOT NULL,
    source_key        TEXT,
    delivery_policy   TEXT NOT NULL,
    slot_policy       TEXT NOT NULL,
    merge_key_json    TEXT NOT NULL,
    available_at_ms   INTEGER NOT NULL,
    enqueued_at_ms    INTEGER NOT NULL,
    claim_id          TEXT,
    claim_owner_id    TEXT,
    claim_token       TEXT,
    claim_fencing_token INTEGER NOT NULL DEFAULT 0,
    claim_claimed_at_ms INTEGER NOT NULL DEFAULT 0,
    claim_expires_at_ms INTEGER NOT NULL DEFAULT 0,
    UNIQUE (session_id, source_key)
        ON CONFLICT IGNORE
);

CREATE TABLE IF NOT EXISTS queued_work_items (
    batch_id      TEXT NOT NULL,
    item_index    INTEGER NOT NULL,
    item_id       TEXT NOT NULL,
    payload_json  TEXT NOT NULL,
    PRIMARY KEY (batch_id, item_index),
    FOREIGN KEY (batch_id) REFERENCES queued_work_batches(batch_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_queued_work_ready
    ON queued_work_batches(session_id, available_at_ms, enqueue_seq);

CREATE INDEX IF NOT EXISTS idx_queued_work_claim
    ON queued_work_batches(session_id, claim_id, claim_token);

CREATE TABLE IF NOT EXISTS attachment_manifest (
    attachment_id    TEXT PRIMARY KEY,
    session_id       TEXT NOT NULL,
    canonical_uri    TEXT NOT NULL,
    intent_at_ms     INTEGER NOT NULL,
    committed_at_ms  INTEGER
);

CREATE TABLE IF NOT EXISTS artifact_refs (
    artifact_ref TEXT PRIMARY KEY,
    blob_ref     TEXT NOT NULL
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
pub(crate) const SCHEMA_VERSION: i32 = 3;

pub(crate) const TURSO_BUSY_TIMEOUT: Duration = Duration::from_secs(15);

pub(crate) async fn file_schema_open_guard() -> tokio::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::OnceLock<tokio::sync::Mutex<()>> = std::sync::OnceLock::new();
    LOCK.get_or_init(|| tokio::sync::Mutex::new(()))
        .lock()
        .await
}

pub(crate) const PROCESS_SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS processes (
    process_id            TEXT PRIMARY KEY,
    registration_hash     TEXT NOT NULL,
    owner_scope_id       TEXT NOT NULL,
    host_profile_id       TEXT NOT NULL,
    created_at_ms         INTEGER NOT NULL,
    updated_at_ms         INTEGER NOT NULL,
    status                TEXT NOT NULL,
    record_json           TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_processes_status
    ON processes(status);

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

CREATE TABLE IF NOT EXISTS process_wake_acks (
    process_id  TEXT NOT NULL,
    sequence    INTEGER NOT NULL,
    PRIMARY KEY (process_id, sequence),
    FOREIGN KEY (process_id) REFERENCES processes(process_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS process_handle_grants (
    session_id       TEXT NOT NULL,
    scope_id        TEXT NOT NULL,
    process_id       TEXT NOT NULL,
    descriptor_json  TEXT NOT NULL,
    PRIMARY KEY (scope_id, process_id),
    FOREIGN KEY (process_id) REFERENCES processes(process_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_process_handle_grants_session
    ON process_handle_grants(session_id);

CREATE INDEX IF NOT EXISTS idx_process_handle_grants_process
    ON process_handle_grants(process_id);

CREATE TABLE IF NOT EXISTS process_leases (
    process_id       TEXT PRIMARY KEY,
    lease_owner_id   TEXT,
    lease_token      TEXT,
    lease_fencing_token  INTEGER NOT NULL DEFAULT 0,
    lease_claimed_at_ms  INTEGER NOT NULL DEFAULT 0,
    lease_expires_at_ms  INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (process_id) REFERENCES processes(process_id) ON DELETE CASCADE
);

";

pub(crate) const PROCESS_SCHEMA_VERSION: i32 = 6;

pub(crate) const EFFECT_SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS runtime_effect_replay (
    scope_id             TEXT NOT NULL,
    replay_key           TEXT NOT NULL,
    envelope_hash        TEXT NOT NULL,
    status               TEXT NOT NULL,
    outcome_json         TEXT,
    error_json           TEXT,
    lease_owner_id       TEXT,
    lease_token          TEXT,
    lease_expires_at_ms  INTEGER NOT NULL DEFAULT 0,
    due_at_ms            INTEGER,
    created_at_ms        INTEGER NOT NULL,
    updated_at_ms        INTEGER NOT NULL,
    PRIMARY KEY (scope_id, replay_key)
);

CREATE INDEX IF NOT EXISTS idx_runtime_effect_replay_lease
    ON runtime_effect_replay(status, lease_expires_at_ms);
";

pub(crate) const EFFECT_SCHEMA_VERSION: i32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum StoreBacking {
    File,
    Memory,
}

pub(crate) async fn apply_pragmas(conn: &Connection, backing: StoreBacking) -> turso::Result<()> {
    conn.busy_timeout(TURSO_BUSY_TIMEOUT)?;
    conn.execute_batch(
        "PRAGMA synchronous = NORMAL;
         PRAGMA foreign_keys = ON;
         PRAGMA cache_size = -2000;",
    )
    .await?;
    if matches!(backing, StoreBacking::File) {
        drain_pragma_rows(conn, "PRAGMA journal_mode = 'mvcc'").await?;
    }
    Ok(())
}

async fn drain_pragma_rows(conn: &Connection, sql: &str) -> turso::Result<()> {
    let mut rows = conn.query(sql, ()).await?;
    while rows.next().await?.is_some() {}
    Ok(())
}

pub(crate) async fn ensure_schema(conn: &Connection) -> turso::Result<()> {
    for attempt in 0..SCHEMA_CHANGED_RETRY_ATTEMPTS {
        match ensure_schema_once(conn).await {
            Err(err)
                if is_database_schema_changed(&err)
                    && attempt + 1 < SCHEMA_CHANGED_RETRY_ATTEMPTS =>
            {
                schema_changed_backoff(attempt);
            }
            result => return result,
        }
    }
    unreachable!("schema retry loop always returns")
}

async fn ensure_schema_once(conn: &Connection) -> turso::Result<()> {
    let user_version = pragma_user_version(conn).await?;
    if user_version == SCHEMA_VERSION {
        conn.execute_batch(SCHEMA).await?;
        return Ok(());
    }

    if user_version == 0 && !has_user_schema_objects(conn).await? {
        initialize_schema(conn, SCHEMA, SCHEMA_VERSION).await?;
        return Ok(());
    }

    Err(turso::Error::Misuse(unsupported_schema_message(
        "session",
        SCHEMA_VERSION,
        user_version,
    )))
}

fn is_database_schema_changed(err: &turso::Error) -> bool {
    err.to_string().contains("Database schema changed")
}

fn schema_changed_backoff(attempt: usize) {
    let multiplier = (attempt + 1) as u32;
    std::thread::sleep(SCHEMA_CHANGED_RETRY_BASE_DELAY * multiplier);
}

async fn initialize_schema(
    conn: &Connection,
    schema: &str,
    schema_version: i32,
) -> turso::Result<()> {
    conn.execute("BEGIN IMMEDIATE", ()).await?;
    let result = async {
        conn.execute_batch(schema).await?;
        conn.pragma_update("user_version", schema_version).await?;
        Ok(())
    }
    .await;
    match result {
        Ok(()) => conn.execute("COMMIT", ()).await.map(|_| ()),
        Err(err) => {
            let _ = conn.execute("ROLLBACK", ()).await;
            Err(err)
        }
    }
}

async fn pragma_user_version(conn: &Connection) -> turso::Result<i32> {
    let row = required_row(conn, "PRAGMA user_version", ()).await?;
    Ok(row_i64(&row, 0)? as i32)
}

pub(crate) async fn has_user_schema_objects(conn: &Connection) -> turso::Result<bool> {
    let row = required_row(
        conn,
        "SELECT COUNT(*) FROM sqlite_master
         WHERE name NOT LIKE 'sqlite_%'
           AND type IN ('table', 'index', 'trigger', 'view')",
        (),
    )
    .await?;
    let count = row_i64(&row, 0)?;
    Ok(count > 0)
}

/// Build the error message for an unsupported on-disk schema. The expected and
/// found `PRAGMA user_version` values are reported accurately (the message used
/// to hard-code "version 1 only" while [`SCHEMA_VERSION`] had moved to 2). There
/// is no migration chain — the database must be deleted before reopening.
pub(crate) fn unsupported_schema_message(
    database_kind: &str,
    expected_version: i32,
    found_version: i32,
) -> String {
    format!(
        "Unsupported lash {database_kind} schema: this binary supports schema version \
         {expected_version}, but the database reports version {found_version}. There is no \
         migration chain — delete the {database_kind} database and start fresh."
    )
}
