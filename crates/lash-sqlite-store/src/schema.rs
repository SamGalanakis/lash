//! Canonical SQLite schema + the open/ensure helpers built on
//! [`SqliteConnection`].
//!
//! The `SCHEMA` / `PROCESS_SCHEMA` / `EFFECT_SCHEMA` strings are plain SQLite
//! and are copied verbatim from the prior store. The only thing that changes in
//! the rusqlite port is the *open path*: the prior store's `Builder::new_local` +
//! `experimental_multiprocess_wal` + `PRAGMA journal_mode='mvcc'` is replaced by
//! [`SqliteConnection::open`], which applies real `journal_mode=WAL` and a
//! 15-second `busy_timeout` (see `conn.rs`).

use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum StoreBacking {
    File,
    Memory,
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
    cache_read_input_tokens  INTEGER NOT NULL,
    cache_write_input_tokens INTEGER NOT NULL,
    reasoning_output_tokens     INTEGER NOT NULL
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

CREATE TABLE IF NOT EXISTS session_execution_leases (
    session_id               TEXT PRIMARY KEY,
    lease_owner_id           TEXT,
    lease_owner_incarnation_id TEXT,
    lease_owner_liveness_json TEXT,
    lease_token              TEXT,
    lease_fencing_token      INTEGER NOT NULL DEFAULT 0,
    lease_claimed_at_ms      INTEGER NOT NULL DEFAULT 0,
    lease_expires_at_ms      INTEGER NOT NULL DEFAULT 0
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
    claim_owner_incarnation_id TEXT,
    claim_owner_liveness_json TEXT,
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

CREATE TABLE IF NOT EXISTS pending_turn_inputs (
    enqueue_seq       INTEGER PRIMARY KEY,
    input_id          TEXT NOT NULL UNIQUE,
    session_id        TEXT NOT NULL,
    source_key        TEXT,
    ingress_json      TEXT NOT NULL,
    state             TEXT NOT NULL,
    input_json        TEXT NOT NULL,
    enqueued_at_ms    INTEGER NOT NULL,
    claim_id          TEXT,
    claim_owner_id    TEXT,
    claim_owner_incarnation_id TEXT,
    claim_owner_liveness_json TEXT,
    claim_token       TEXT,
    claim_fencing_token INTEGER NOT NULL DEFAULT 0,
    claim_claimed_at_ms INTEGER NOT NULL DEFAULT 0,
    claim_expires_at_ms INTEGER NOT NULL DEFAULT 0,
    UNIQUE (session_id, source_key)
        ON CONFLICT IGNORE
);

CREATE INDEX IF NOT EXISTS idx_pending_turn_inputs_session
    ON pending_turn_inputs(session_id, state, enqueue_seq);

CREATE INDEX IF NOT EXISTS idx_pending_turn_inputs_claim
    ON pending_turn_inputs(session_id, claim_id, claim_token);

CREATE TABLE IF NOT EXISTS attachment_manifest (
    attachment_id    TEXT PRIMARY KEY,
    session_id       TEXT NOT NULL,
    canonical_uri    TEXT NOT NULL,
    intent_at_ms     INTEGER NOT NULL,
    committed_at_ms  INTEGER
);

CREATE TABLE IF NOT EXISTS artifact_refs (
    namespace    TEXT NOT NULL,
    artifact_ref TEXT NOT NULL,
    blob_ref     TEXT NOT NULL,
    PRIMARY KEY (namespace, artifact_ref)
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
pub(crate) const SCHEMA_VERSION: i32 = 8;

pub(crate) const PROCESS_SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS processes (
    process_id            TEXT PRIMARY KEY,
    registration_hash     TEXT NOT NULL,
    owner_scope_id       TEXT NOT NULL,
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

pub(crate) const PROCESS_SCHEMA_VERSION: i32 = 7;

pub(crate) const TRIGGER_SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS trigger_subscription_seq (
    id INTEGER PRIMARY KEY AUTOINCREMENT
);

CREATE TABLE IF NOT EXISTS trigger_subscriptions (
    subscription_id      TEXT PRIMARY KEY,
    registrant_scope_id  TEXT NOT NULL,
    handle               TEXT NOT NULL,
    source_type          TEXT NOT NULL,
    source_key           TEXT NOT NULL,
    enabled              INTEGER NOT NULL,
    created_at_ms        INTEGER NOT NULL,
    updated_at_ms        INTEGER NOT NULL,
    record_json          TEXT NOT NULL,
    UNIQUE(registrant_scope_id, handle)
);

CREATE INDEX IF NOT EXISTS idx_trigger_subscriptions_registrant
    ON trigger_subscriptions(registrant_scope_id, handle);

CREATE INDEX IF NOT EXISTS idx_trigger_subscriptions_source
    ON trigger_subscriptions(source_type, source_key, enabled);

CREATE TABLE IF NOT EXISTS trigger_occurrences (
    occurrence_id    TEXT PRIMARY KEY,
    idempotency_key  TEXT NOT NULL UNIQUE,
    request_hash     TEXT NOT NULL,
    source_type      TEXT NOT NULL,
    source_key       TEXT NOT NULL,
    occurred_at_ms   INTEGER NOT NULL,
    record_json      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_trigger_occurrences_source
    ON trigger_occurrences(source_type, source_key, occurred_at_ms);

CREATE TABLE IF NOT EXISTS trigger_deliveries (
    occurrence_id    TEXT NOT NULL,
    subscription_id  TEXT NOT NULL,
    process_id       TEXT NOT NULL,
    created_at_ms    INTEGER NOT NULL,
    PRIMARY KEY (occurrence_id, subscription_id),
    FOREIGN KEY (occurrence_id) REFERENCES trigger_occurrences(occurrence_id) ON DELETE CASCADE,
    FOREIGN KEY (subscription_id) REFERENCES trigger_subscriptions(subscription_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_trigger_deliveries_process
    ON trigger_deliveries(process_id);
";

pub(crate) const TRIGGER_SCHEMA_VERSION: i32 = 1;

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

pub(crate) async fn apply_pragmas(
    conn: &SqliteConnection,
    backing: StoreBacking,
) -> rusqlite::Result<()> {
    // WAL + busy_timeout are already applied in `SqliteConnection::open` /
    // `open_in_memory`. The remaining tuning PRAGMAs match the prior store. The
    // `backing` argument is retained so the lifecycle call sites read the same
    // as the prior store port; WAL is only meaningful for file-backed databases.
    let _ = backing;
    conn.call(|c| {
        c.execute_batch(
            "PRAGMA synchronous = NORMAL;
             PRAGMA foreign_keys = ON;
             PRAGMA cache_size = -2000;",
        )?;
        Ok(())
    })
    .await
}

pub(crate) async fn ensure_schema(conn: &SqliteConnection) -> rusqlite::Result<()> {
    ensure_versioned_schema(conn, "session", SCHEMA, SCHEMA_VERSION).await
}

pub(crate) async fn ensure_process_schema(conn: &SqliteConnection) -> rusqlite::Result<()> {
    ensure_versioned_schema(
        conn,
        "process registry",
        PROCESS_SCHEMA,
        PROCESS_SCHEMA_VERSION,
    )
    .await
}

pub(crate) async fn ensure_trigger_schema(conn: &SqliteConnection) -> rusqlite::Result<()> {
    ensure_versioned_schema(
        conn,
        "trigger store",
        TRIGGER_SCHEMA,
        TRIGGER_SCHEMA_VERSION,
    )
    .await
}

pub(crate) async fn ensure_effect_schema(conn: &SqliteConnection) -> rusqlite::Result<()> {
    ensure_versioned_schema(conn, "effect replay", EFFECT_SCHEMA, EFFECT_SCHEMA_VERSION).await
}

/// Apply `schema` if the database is already at `schema_version`, initialise it
/// (under one transaction stamping `user_version`) if the database is empty, or
/// reject the open if the on-disk `user_version` is anything else. Runs entirely
/// on the connection thread so the version check and DDL share one connection.
async fn ensure_versioned_schema(
    conn: &SqliteConnection,
    database_kind: &'static str,
    schema: &'static str,
    schema_version: i32,
) -> rusqlite::Result<()> {
    conn.call(move |c| {
        // The whole check-then-initialise runs inside one `BEGIN IMMEDIATE`
        // transaction so the write lock is held across the `user_version` read.
        // Reading the version outside the transaction and only then upgrading to
        // a writer races concurrent first-openers into a lock-upgrade deadlock
        // (SQLite returns "database is locked" immediately, bypassing
        // `busy_timeout`). Holding the write lock from the first statement makes
        // every contender serialise on the busy handler instead.
        let tx = c.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let user_version: i32 = tx.query_row("PRAGMA user_version", [], |row| row.get(0))?;
        if user_version == schema_version {
            tx.execute_batch(schema)?;
            tx.commit()?;
            return Ok(());
        }
        if user_version == 0 && !has_user_schema_objects(&tx)? {
            tx.execute_batch(schema)?;
            tx.pragma_update(None, "user_version", schema_version)?;
            tx.commit()?;
            return Ok(());
        }
        Err(rusqlite::Error::SqliteFailure(
            rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_MISUSE),
            Some(unsupported_schema_message(
                database_kind,
                schema_version,
                user_version,
            )),
        ))
    })
    .await
}

pub(crate) fn has_user_schema_objects(conn: &Connection) -> rusqlite::Result<bool> {
    let count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM sqlite_master
         WHERE name NOT LIKE 'sqlite_%'
           AND type IN ('table', 'index', 'trigger', 'view')",
        [],
        |row| row.get(0),
    )?;
    Ok(count > 0)
}

/// Build the error message for an unsupported on-disk schema. The expected and
/// found `PRAGMA user_version` values are reported accurately. There is no
/// migration chain — the database must be deleted before reopening.
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
