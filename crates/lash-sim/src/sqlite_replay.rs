use std::fmt;
use std::path::{Path, PathBuf};

use rusqlite::{Connection, params};
use serde::{Deserialize, Serialize};

use crate::replay::{ReplayError, replay_trace};
use crate::trace::{
    AbstractWorldSummary, OracleVerdict, SimulationTrace, TraceIoError, read_trace,
};

pub const SQLITE_REPLAY_REPORT_SCHEMA: &str = "lash.sim.sqlite-replay-report.v1";

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SqliteReplayReport {
    pub schema: String,
    pub trace_path: PathBuf,
    pub database_path: PathBuf,
    pub terminal_verdict: OracleVerdict,
    pub delivered_event_count: usize,
    pub inserted_event_count: usize,
    pub final_summary: AbstractWorldSummary,
}

#[derive(Debug)]
pub enum SqliteReplayError {
    TraceIo(TraceIoError),
    Replay(ReplayError),
    Sqlite(rusqlite::Error),
    Json(serde_json::Error),
    Io(std::io::Error),
    Divergence(String),
}

impl fmt::Display for SqliteReplayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TraceIo(err) => write!(f, "{err}"),
            Self::Replay(err) => write!(f, "{err}"),
            Self::Sqlite(err) => write!(f, "SQLite replay failed: {err}"),
            Self::Json(err) => write!(f, "SQLite replay JSON failed: {err}"),
            Self::Io(err) => write!(f, "SQLite replay I/O failed: {err}"),
            Self::Divergence(message) => write!(f, "SQLite replay diverged: {message}"),
        }
    }
}

impl std::error::Error for SqliteReplayError {}

impl From<TraceIoError> for SqliteReplayError {
    fn from(value: TraceIoError) -> Self {
        Self::TraceIo(value)
    }
}

impl From<ReplayError> for SqliteReplayError {
    fn from(value: ReplayError) -> Self {
        Self::Replay(value)
    }
}

impl From<rusqlite::Error> for SqliteReplayError {
    fn from(value: rusqlite::Error) -> Self {
        Self::Sqlite(value)
    }
}

impl From<serde_json::Error> for SqliteReplayError {
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value)
    }
}

impl From<std::io::Error> for SqliteReplayError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

pub fn replay_trace_file_to_sqlite(
    trace_path: &Path,
    db_path: &Path,
    report_path: Option<&Path>,
) -> Result<SqliteReplayReport, SqliteReplayError> {
    let trace = read_trace(trace_path)?;
    replay_trace_to_sqlite(trace_path, &trace, db_path, report_path)
}

pub fn replay_trace_to_sqlite(
    trace_path: &Path,
    trace: &SimulationTrace,
    db_path: &Path,
    report_path: Option<&Path>,
) -> Result<SqliteReplayReport, SqliteReplayError> {
    let replay = replay_trace(trace_path, trace)?;
    if let Some(parent) = db_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    if db_path.exists() {
        std::fs::remove_file(db_path)?;
    }

    let mut conn = Connection::open(db_path)?;
    create_schema(&conn)?;
    let tx = conn.transaction()?;
    for (sequence, event) in trace.events.iter().enumerate() {
        tx.execute(
            "INSERT INTO sim_boundary_events (
                sequence, boundary_id, actor_alias, kind, at_tick, label, payload_json, observed_json
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                sequence as i64,
                event.boundary_id,
                event.actor_alias,
                format!("{:?}", event.kind),
                event.at as i64,
                event.label,
                serde_json::to_string(&event.payload)?,
                serde_json::to_string(&event.observed)?,
            ],
        )?;
    }
    tx.execute(
        "INSERT INTO sim_final_summary (id, digest, summary_json) VALUES (1, ?1, ?2)",
        params![
            replay.final_summary.digest,
            serde_json::to_string(&replay.final_summary)?,
        ],
    )?;
    tx.commit()?;
    drop(conn);

    let conn = Connection::open(db_path)?;
    let inserted_event_count: usize =
        conn.query_row("SELECT COUNT(*) FROM sim_boundary_events", [], |row| {
            row.get::<_, i64>(0)
        })? as usize;
    let stored_summary_json: String = conn.query_row(
        "SELECT summary_json FROM sim_final_summary WHERE id = 1",
        [],
        |row| row.get(0),
    )?;
    let stored_summary: AbstractWorldSummary = serde_json::from_str(&stored_summary_json)?;
    if inserted_event_count != trace.events.len() {
        return Err(SqliteReplayError::Divergence(format!(
            "inserted {inserted_event_count} events, expected {}",
            trace.events.len()
        )));
    }
    if stored_summary != replay.final_summary {
        return Err(SqliteReplayError::Divergence(
            "stored summary did not round-trip from SQLite".to_string(),
        ));
    }

    let report = SqliteReplayReport {
        schema: SQLITE_REPLAY_REPORT_SCHEMA.to_string(),
        trace_path: trace_path.to_path_buf(),
        database_path: db_path.to_path_buf(),
        terminal_verdict: replay.terminal_verdict,
        delivered_event_count: replay.delivered_event_count,
        inserted_event_count,
        final_summary: replay.final_summary,
    };
    if let Some(report_path) = report_path {
        if let Some(parent) = report_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(report_path, serde_json::to_vec_pretty(&report)?)?;
    }
    Ok(report)
}

fn create_schema(conn: &Connection) -> Result<(), rusqlite::Error> {
    conn.execute_batch(
        "
        CREATE TABLE sim_boundary_events (
            sequence INTEGER PRIMARY KEY,
            boundary_id TEXT NOT NULL,
            actor_alias TEXT NOT NULL,
            kind TEXT NOT NULL,
            at_tick INTEGER NOT NULL,
            label TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            observed_json TEXT NOT NULL
        );
        CREATE TABLE sim_final_summary (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            digest TEXT NOT NULL,
            summary_json TEXT NOT NULL
        );
        ",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generator::generate_workload;
    use crate::runner::run_generated_workload_for_test;

    #[tokio::test]
    async fn sqlite_replay_persists_trace_and_summary_round_trip() {
        let workload = generate_workload(7, "fast-random", 24);
        let trace = run_generated_workload_for_test(workload, "bundle")
            .await
            .expect("trace");
        let tmp = tempfile::tempdir().expect("tempdir");
        let db_path = tmp.path().join("replay.sqlite");
        let report_path = tmp.path().join("sqlite-replay.json");

        let report = replay_trace_to_sqlite(
            Path::new("trace.json"),
            &trace,
            &db_path,
            Some(&report_path),
        )
        .expect("sqlite replay");

        assert_eq!(report.schema, SQLITE_REPLAY_REPORT_SCHEMA);
        assert_eq!(report.inserted_event_count, trace.events.len());
        assert_eq!(report.final_summary, trace.final_summary);
        assert!(db_path.exists());
        assert!(report_path.exists());
    }
}
