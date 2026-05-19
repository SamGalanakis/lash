use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use rusqlite::{Connection, OptionalExtension, params};

use super::{
    Candidate, CandidateRecord, EvaluationRecord, ExampleRun, HarnessExample, HarnessOptError,
    HarnessOptStore, OptimizationConfig, OptimizationRun, ProposalRecord, Result, Split,
    StoreStats,
};

pub struct SqliteHarnessStore {
    conn: Arc<Mutex<Connection>>,
}

impl SqliteHarnessStore {
    pub async fn open(run_dir: impl AsRef<Path>) -> Result<Self> {
        tokio::fs::create_dir_all(run_dir.as_ref()).await?;
        let path = run_dir.as_ref().join("harness-opt.sqlite");
        let conn = Connection::open(path)?;
        let store = Self {
            conn: Arc::new(Mutex::new(conn)),
        };
        store.create_schema()?;
        Ok(store)
    }

    fn create_schema(&self) -> Result<()> {
        let conn = self.conn()?;
        conn.execute_batch(
            r#"
            PRAGMA journal_mode = WAL;
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                experiment_id TEXT NOT NULL,
                run_dir TEXT NOT NULL,
                config_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS candidates (
                id TEXT PRIMARY KEY,
                fingerprint TEXT NOT NULL,
                candidate_json TEXT NOT NULL,
                parent_ids_json TEXT NOT NULL,
                generation INTEGER NOT NULL,
                source_strategy TEXT NOT NULL,
                component_cursor INTEGER NOT NULL,
                discovery_budget INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_id TEXT NOT NULL,
                candidate_fingerprint TEXT NOT NULL,
                example_id TEXT NOT NULL,
                split TEXT NOT NULL,
                score REAL NOT NULL,
                metrics_json TEXT NOT NULL,
                feedback TEXT,
                diagnostics_json TEXT NOT NULL,
                artifacts_json TEXT NOT NULL,
                trace_json TEXT,
                metric_calls INTEGER NOT NULL,
                cache_hit INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS eval_cache (
                candidate_fingerprint TEXT NOT NULL,
                example_id TEXT NOT NULL,
                split TEXT NOT NULL,
                example_run_json TEXT NOT NULL,
                metric_calls INTEGER NOT NULL,
                PRIMARY KEY (candidate_fingerprint, example_id, split)
            );
            CREATE TABLE IF NOT EXISTS proposals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                generation INTEGER NOT NULL,
                parent_ids_json TEXT NOT NULL,
                selected_components_json TEXT NOT NULL,
                minibatch_ids_json TEXT NOT NULL,
                patches_json TEXT NOT NULL,
                rlm_prompt_ref TEXT,
                rlm_output_ref TEXT,
                before_score REAL NOT NULL,
                after_score REAL NOT NULL,
                accepted INTEGER NOT NULL,
                reason TEXT NOT NULL,
                candidate_id TEXT
            );
            "#,
        )?;
        Ok(())
    }

    fn conn(&self) -> Result<std::sync::MutexGuard<'_, Connection>> {
        self.conn
            .lock()
            .map_err(|error| HarnessOptError::Store(error.to_string()))
    }
}

#[async_trait]
impl HarnessOptStore for SqliteHarnessStore {
    async fn init_run(&self, run: &OptimizationRun) -> Result<()> {
        let existing = self.load_run().await?;
        if let Some(existing) = existing {
            if existing.experiment_id != run.experiment_id || existing.config != run.config {
                return Err(HarnessOptError::Store(
                    "existing harness-opt.sqlite has incompatible run config".to_string(),
                ));
            }
            return Ok(());
        }
        let conn = self.conn()?;
        conn.execute(
            "INSERT INTO runs (run_id, experiment_id, run_dir, config_json) VALUES (?1, ?2, ?3, ?4)",
            params![
                run.run_id,
                run.experiment_id,
                run.run_dir.to_string_lossy(),
                serde_json::to_string(&run.config)?,
            ],
        )?;
        Ok(())
    }

    async fn load_run(&self) -> Result<Option<OptimizationRun>> {
        let conn = self.conn()?;
        conn.query_row(
            "SELECT run_id, experiment_id, run_dir, config_json FROM runs LIMIT 1",
            [],
            |row| {
                let config_json: String = row.get(3)?;
                let config: OptimizationConfig = serde_json::from_str(&config_json)
                    .map_err(|error| rusqlite::Error::ToSqlConversionFailure(Box::new(error)))?;
                Ok(OptimizationRun {
                    run_id: row.get(0)?,
                    experiment_id: row.get(1)?,
                    run_dir: PathBuf::from(row.get::<_, String>(2)?),
                    config,
                })
            },
        )
        .optional()
        .map_err(Into::into)
    }

    async fn upsert_candidate(&self, record: &CandidateRecord) -> Result<()> {
        let conn = self.conn()?;
        conn.execute(
            r#"
            INSERT INTO candidates
                (id, fingerprint, candidate_json, parent_ids_json, generation, source_strategy, component_cursor, discovery_budget)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
            ON CONFLICT(id) DO UPDATE SET
                fingerprint=excluded.fingerprint,
                candidate_json=excluded.candidate_json,
                parent_ids_json=excluded.parent_ids_json,
                generation=excluded.generation,
                source_strategy=excluded.source_strategy,
                component_cursor=excluded.component_cursor,
                discovery_budget=excluded.discovery_budget
            "#,
            params![
                record.candidate.id,
                record.fingerprint,
                serde_json::to_string(&record.candidate)?,
                serde_json::to_string(&record.parent_ids)?,
                record.generation as i64,
                record.source_strategy,
                record.component_cursor as i64,
                record.discovery_budget as i64,
            ],
        )?;
        Ok(())
    }

    async fn candidates(&self) -> Result<Vec<CandidateRecord>> {
        let conn = self.conn()?;
        let mut stmt = conn.prepare(
            "SELECT candidate_json, fingerprint, parent_ids_json, generation, source_strategy, component_cursor, discovery_budget FROM candidates ORDER BY generation, id",
        )?;
        let rows = stmt.query_map([], |row| {
            let candidate: Candidate = serde_json::from_str(&row.get::<_, String>(0)?)
                .map_err(|error| rusqlite::Error::ToSqlConversionFailure(Box::new(error)))?;
            let parent_ids: Vec<String> = serde_json::from_str(&row.get::<_, String>(2)?)
                .map_err(|error| rusqlite::Error::ToSqlConversionFailure(Box::new(error)))?;
            Ok(CandidateRecord {
                candidate,
                fingerprint: row.get(1)?,
                parent_ids,
                generation: row.get::<_, i64>(3)? as usize,
                source_strategy: row.get(4)?,
                component_cursor: row.get::<_, i64>(5)? as usize,
                discovery_budget: row.get::<_, i64>(6)? as u64,
            })
        })?;
        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(Into::into)
    }

    async fn insert_evaluation(&self, record: &EvaluationRecord) -> Result<()> {
        let conn = self.conn()?;
        conn.execute(
            r#"
            INSERT INTO evaluations
                (candidate_id, candidate_fingerprint, example_id, split, score, metrics_json, feedback, diagnostics_json, artifacts_json, trace_json, metric_calls, cache_hit)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)
            "#,
            params![
                record.candidate_id,
                record.candidate_fingerprint,
                record.example_id,
                record.split.as_str(),
                record.score,
                serde_json::to_string(&record.metrics)?,
                record.feedback,
                serde_json::to_string(&record.diagnostics)?,
                serde_json::to_string(&record.artifacts)?,
                record.trace.as_ref().map(serde_json::to_string).transpose()?,
                record.metric_calls as i64,
                i64::from(record.cache_hit),
            ],
        )?;
        Ok(())
    }

    async fn evaluations(&self) -> Result<Vec<EvaluationRecord>> {
        let conn = self.conn()?;
        let mut stmt = conn.prepare(
            r#"
            SELECT candidate_id, candidate_fingerprint, example_id, split, score, metrics_json,
                   feedback, diagnostics_json, artifacts_json, trace_json, metric_calls, cache_hit
            FROM evaluations ORDER BY id
            "#,
        )?;
        let rows = stmt.query_map([], |row| {
            let split = match row.get::<_, String>(3)?.as_str() {
                "train" => Split::Train,
                "val" => Split::Val,
                "test" => Split::Test,
                other => {
                    return Err(rusqlite::Error::FromSqlConversionFailure(
                        3,
                        rusqlite::types::Type::Text,
                        format!("unknown split {other}").into(),
                    ));
                }
            };
            let trace_json: Option<String> = row.get(9)?;
            Ok(EvaluationRecord {
                candidate_id: row.get(0)?,
                candidate_fingerprint: row.get(1)?,
                example_id: row.get(2)?,
                split,
                score: row.get(4)?,
                metrics: serde_json::from_str(&row.get::<_, String>(5)?)
                    .map_err(|error| rusqlite::Error::ToSqlConversionFailure(Box::new(error)))?,
                feedback: row.get(6)?,
                diagnostics: serde_json::from_str(&row.get::<_, String>(7)?)
                    .map_err(|error| rusqlite::Error::ToSqlConversionFailure(Box::new(error)))?,
                artifacts: serde_json::from_str(&row.get::<_, String>(8)?)
                    .map_err(|error| rusqlite::Error::ToSqlConversionFailure(Box::new(error)))?,
                trace: trace_json
                    .map(|json| serde_json::from_str(&json))
                    .transpose()
                    .map_err(|error| rusqlite::Error::ToSqlConversionFailure(Box::new(error)))?,
                metric_calls: row.get::<_, i64>(10)? as u64,
                cache_hit: row.get::<_, i64>(11)? != 0,
            })
        })?;
        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(Into::into)
    }

    async fn cached_example(
        &self,
        candidate_fingerprint: &str,
        example: &HarnessExample,
    ) -> Result<Option<ExampleRun>> {
        let conn = self.conn()?;
        conn.query_row(
            "SELECT example_run_json FROM eval_cache WHERE candidate_fingerprint=?1 AND example_id=?2 AND split=?3",
            params![candidate_fingerprint, example.id, example.split.as_str()],
            |row| {
                serde_json::from_str(&row.get::<_, String>(0)?)
                    .map_err(|error| rusqlite::Error::ToSqlConversionFailure(Box::new(error)))
            },
        )
        .optional()
        .map_err(Into::into)
    }

    async fn put_cached_example(
        &self,
        candidate_fingerprint: &str,
        run: &ExampleRun,
    ) -> Result<()> {
        let conn = self.conn()?;
        conn.execute(
            r#"
            INSERT OR REPLACE INTO eval_cache
                (candidate_fingerprint, example_id, split, example_run_json, metric_calls)
            VALUES (?1, ?2, ?3, ?4, ?5)
            "#,
            params![
                candidate_fingerprint,
                run.example.id,
                run.example.split.as_str(),
                serde_json::to_string(run)?,
                run.metric_calls as i64,
            ],
        )?;
        Ok(())
    }

    async fn insert_proposal(&self, record: &ProposalRecord) -> Result<()> {
        let conn = self.conn()?;
        conn.execute(
            r#"
            INSERT INTO proposals
                (run_id, generation, parent_ids_json, selected_components_json, minibatch_ids_json, patches_json,
                 rlm_prompt_ref, rlm_output_ref, before_score, after_score, accepted, reason, candidate_id)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)
            "#,
            params![
                record.run_id,
                record.generation as i64,
                serde_json::to_string(&record.parent_ids)?,
                serde_json::to_string(&record.selected_components)?,
                serde_json::to_string(&record.minibatch_ids)?,
                serde_json::to_string(&record.patches)?,
                record.rlm_prompt_ref.as_ref().map(|path| path.to_string_lossy().to_string()),
                record.rlm_output_ref.as_ref().map(|path| path.to_string_lossy().to_string()),
                record.before_score,
                record.after_score,
                i64::from(record.accepted),
                record.reason,
                record.candidate_id,
            ],
        )?;
        Ok(())
    }

    async fn stats(&self) -> Result<StoreStats> {
        let conn = self.conn()?;
        let metric_calls_used: u64 = conn.query_row(
            "SELECT COALESCE(SUM(metric_calls), 0) FROM evaluations WHERE cache_hit = 0",
            [],
            |row| row.get::<_, i64>(0),
        )? as u64;
        let cache_hits: u64 = conn.query_row(
            "SELECT COUNT(*) FROM evaluations WHERE cache_hit != 0",
            [],
            |row| row.get::<_, i64>(0),
        )? as u64;
        let cache_misses: u64 = conn.query_row(
            "SELECT COUNT(*) FROM evaluations WHERE cache_hit = 0",
            [],
            |row| row.get::<_, i64>(0),
        )? as u64;
        let accepted_proposals: u64 = conn.query_row(
            "SELECT COUNT(*) FROM proposals WHERE accepted != 0",
            [],
            |row| row.get::<_, i64>(0),
        )? as u64;
        let rejected_proposals: u64 = conn.query_row(
            "SELECT COUNT(*) FROM proposals WHERE accepted = 0",
            [],
            |row| row.get::<_, i64>(0),
        )? as u64;
        let candidates: usize = conn.query_row("SELECT COUNT(*) FROM candidates", [], |row| {
            row.get::<_, i64>(0)
        })? as usize;
        let evaluations: usize = conn.query_row("SELECT COUNT(*) FROM evaluations", [], |row| {
            row.get::<_, i64>(0)
        })? as usize;
        Ok(StoreStats {
            metric_calls_used,
            cache_hits,
            cache_misses,
            accepted_proposals,
            rejected_proposals,
            candidates,
            evaluations,
        })
    }
}
