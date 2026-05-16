use std::collections::{BTreeMap, BTreeSet};
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use lash_sansio::{
    PromptBuiltin, PromptContribution, PromptSlot, PromptTemplate, PromptTemplateEntry,
    PromptTemplateSection,
};
use lash_trace::{TraceContext, TraceEvent, TraceRecord};
use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::sync::Semaphore;
use tokio_util::sync::CancellationToken;

#[derive(Debug, thiserror::Error)]
pub enum HarnessOptError {
    #[error("unknown mutable component `{0}`")]
    UnknownComponent(String),
    #[error("component `{component_id}` violates constraint: {reason}")]
    ConstraintViolation {
        component_id: String,
        reason: String,
    },
    #[error("invalid proposal: {0}")]
    InvalidProposal(String),
    #[error("harness error: {0}")]
    Harness(String),
    #[error("strategy error: {0}")]
    Strategy(String),
    #[error("optimizer cancelled")]
    Cancelled,
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("sqlite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("store error: {0}")]
    Store(String),
}

pub type Result<T> = std::result::Result<T, HarnessOptError>;

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Split {
    Train,
    Val,
    Test,
}

impl Split {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Train => "train",
            Self::Val => "val",
            Self::Test => "test",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ComponentValue {
    Text { text: String },
    Json { value: Value },
    PromptTemplate { template: PromptTemplate },
    PromptContribution { contribution: PromptContribution },
}

impl ComponentValue {
    fn text_for_constraints(&self) -> Option<&str> {
        match self {
            Self::Text { text } => Some(text),
            Self::PromptContribution { contribution } => Some(&contribution.content),
            Self::Json { .. } | Self::PromptTemplate { .. } => None,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComponentConstraints {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_chars: Option<usize>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub preserve_terms: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub forbidden_terms: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format_hint: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MutableComponent {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub value: ComponentValue,
    #[serde(default)]
    pub constraints: ComponentConstraints,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Candidate {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_id: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub mutable_components: BTreeMap<String, MutableComponent>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub immutable_context: BTreeMap<String, Value>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, Value>,
}

impl Candidate {
    pub fn with_component(mut self, component: MutableComponent) -> Self {
        self.mutable_components
            .insert(component.id.clone(), component);
        self
    }

    pub fn component(&self, id: &str) -> Result<&MutableComponent> {
        self.mutable_components
            .get(id)
            .ok_or_else(|| HarnessOptError::UnknownComponent(id.to_string()))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ComponentPatch {
    ReplaceValue {
        component_id: String,
        value: ComponentValue,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CandidateProposal {
    pub parent_candidate_id: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub patches: Vec<ComponentPatch>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rationale: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HarnessExample {
    pub id: String,
    pub split: Split,
    pub input: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected: Option<Value>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RunArtifacts {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_json: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub candidate_json: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_json: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_db: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub typed_trace_jsonl: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rendered_evidence_json: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score_report_json: Option<PathBuf>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EvaluationResult {
    pub example_id: String,
    pub split: Split,
    pub score: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub passed: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub feedback: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metrics: BTreeMap<String, f64>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub diagnostics: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct TraceBundle {
    pub example_id: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub records: Vec<TraceRecord>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExampleRun {
    pub example: HarnessExample,
    pub result: EvaluationResult,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trace: Option<TraceBundle>,
    #[serde(default)]
    pub artifacts: RunArtifacts,
    #[serde(default = "default_metric_calls")]
    pub metric_calls: u64,
}

fn default_metric_calls() -> u64 {
    1
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EvidenceBatch {
    pub parent: Candidate,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evaluated_examples: Vec<ExampleRun>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub frontier: Vec<CandidateEvaluation>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StrategyRequest {
    pub run_id: String,
    pub experiment_id: String,
    pub generation: usize,
    pub artifact_dir: PathBuf,
    pub evidence: EvidenceBatch,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CandidateEvaluation {
    pub candidate: Candidate,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub evaluations: BTreeMap<String, EvaluationResult>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub traces: BTreeMap<String, TraceBundle>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub artifacts: BTreeMap<String, RunArtifacts>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metric_calls: BTreeMap<String, u64>,
}

impl CandidateEvaluation {
    pub fn mean_score(&self) -> f64 {
        if self.evaluations.is_empty() {
            return 0.0;
        }
        self.evaluations
            .values()
            .map(|result| result.score)
            .sum::<f64>()
            / self.evaluations.len() as f64
    }

    fn example_runs_for(&self, examples: &[HarnessExample]) -> Vec<ExampleRun> {
        examples
            .iter()
            .filter_map(|example| {
                let result = self.evaluations.get(&example.id)?;
                Some(ExampleRun {
                    example: example.clone(),
                    result: result.clone(),
                    trace: self.traces.get(&example.id).cloned(),
                    artifacts: self.artifacts.get(&example.id).cloned().unwrap_or_default(),
                    metric_calls: self.metric_calls.get(&example.id).copied().unwrap_or(0),
                })
            })
            .collect()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OptimizationConfig {
    pub max_metric_calls: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_iterations: Option<usize>,
    pub minibatch_size: usize,
    pub max_concurrency: usize,
    pub perfect_score: f64,
    pub skip_perfect_score: bool,
    pub candidate_selection: CandidateSelection,
    pub component_selection: ComponentSelection,
    pub frontier: FrontierMode,
    pub use_merge: bool,
    pub max_merge_invocations: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub per_example_timeout_secs: Option<u64>,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            max_metric_calls: 64,
            max_iterations: None,
            minibatch_size: 8,
            max_concurrency: 4,
            perfect_score: 1.0,
            skip_perfect_score: true,
            candidate_selection: CandidateSelection::Pareto,
            component_selection: ComponentSelection::RoundRobin,
            frontier: FrontierMode::Instance,
            use_merge: false,
            max_merge_invocations: 0,
            per_example_timeout_secs: Some(300),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CandidateSelection {
    #[default]
    Pareto,
    CurrentBest,
    EpsilonGreedy,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComponentSelection {
    #[default]
    RoundRobin,
    All,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FrontierMode {
    #[default]
    Instance,
    Objective,
    Hybrid,
    Cartesian,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OptimizationRun {
    pub run_id: String,
    pub experiment_id: String,
    pub run_dir: PathBuf,
    pub config: OptimizationConfig,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GenerationReport {
    pub generation: usize,
    pub parent_candidate_id: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub proposed_candidate_ids: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OptimizationState {
    pub run: OptimizationRun,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evaluated_candidates: Vec<CandidateEvaluation>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub best_candidate_id: Option<String>,
    #[serde(default)]
    pub metric_calls_used: u64,
    #[serde(default)]
    pub cache_hits: u64,
    #[serde(default)]
    pub cache_misses: u64,
    #[serde(default)]
    pub accepted_proposals: u64,
    #[serde(default)]
    pub rejected_proposals: u64,
}

impl OptimizationState {
    pub fn best(&self) -> Option<&CandidateEvaluation> {
        best_state(&self.evaluated_candidates)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CandidateRecord {
    pub candidate: Candidate,
    pub fingerprint: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub parent_ids: Vec<String>,
    pub generation: usize,
    pub source_strategy: String,
    pub component_cursor: usize,
    pub discovery_budget: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EvaluationRecord {
    pub candidate_id: String,
    pub candidate_fingerprint: String,
    pub example_id: String,
    pub split: Split,
    pub score: f64,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metrics: BTreeMap<String, f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub feedback: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub diagnostics: BTreeMap<String, Value>,
    #[serde(default)]
    pub artifacts: RunArtifacts,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trace: Option<TraceBundle>,
    #[serde(default)]
    pub metric_calls: u64,
    #[serde(default)]
    pub cache_hit: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProposalRecord {
    pub run_id: String,
    pub generation: usize,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub parent_ids: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub selected_components: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub minibatch_ids: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub patches: Vec<ComponentPatch>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rlm_prompt_ref: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rlm_output_ref: Option<PathBuf>,
    pub before_score: f64,
    pub after_score: f64,
    pub accepted: bool,
    pub reason: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub candidate_id: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct StoreStats {
    pub metric_calls_used: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub accepted_proposals: u64,
    pub rejected_proposals: u64,
    pub candidates: usize,
    pub evaluations: usize,
}

#[async_trait]
pub trait HarnessOptStore: Send + Sync {
    async fn init_run(&self, run: &OptimizationRun) -> Result<()>;
    async fn load_run(&self) -> Result<Option<OptimizationRun>>;
    async fn upsert_candidate(&self, record: &CandidateRecord) -> Result<()>;
    async fn candidates(&self) -> Result<Vec<CandidateRecord>>;
    async fn insert_evaluation(&self, record: &EvaluationRecord) -> Result<()>;
    async fn evaluations(&self) -> Result<Vec<EvaluationRecord>>;
    async fn cached_example(
        &self,
        candidate_fingerprint: &str,
        example: &HarnessExample,
    ) -> Result<Option<ExampleRun>>;
    async fn put_cached_example(&self, candidate_fingerprint: &str, run: &ExampleRun)
    -> Result<()>;
    async fn insert_proposal(&self, record: &ProposalRecord) -> Result<()>;
    async fn stats(&self) -> Result<StoreStats>;
}

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

#[async_trait]
pub trait HarnessProject: Send + Sync {
    async fn seed_candidate(&self) -> Result<Candidate>;
    async fn trainset(&self) -> Result<Vec<HarnessExample>>;
    async fn valset(&self) -> Result<Vec<HarnessExample>>;
    async fn testset(&self) -> Result<Vec<HarnessExample>> {
        Ok(Vec::new())
    }
    async fn evaluate_example(
        &self,
        run: &OptimizationRun,
        candidate: &Candidate,
        example: &HarnessExample,
        context: TraceContext,
        cancellation: CancellationToken,
    ) -> Result<ExampleRun>;
}

#[async_trait]
pub trait HarnessRunner: Send + Sync {
    async fn evaluate_candidate(
        &self,
        run: &OptimizationRun,
        candidate: Candidate,
        examples: Vec<HarnessExample>,
        cancellation: CancellationToken,
    ) -> Result<CandidateEvaluation>;
}

pub struct ProjectHarnessRunner<P> {
    project: Arc<P>,
}

impl<P> ProjectHarnessRunner<P> {
    pub fn new(project: Arc<P>) -> Self {
        Self { project }
    }
}

#[async_trait]
impl<P> HarnessRunner for ProjectHarnessRunner<P>
where
    P: HarnessProject + 'static,
{
    async fn evaluate_candidate(
        &self,
        run: &OptimizationRun,
        candidate: Candidate,
        examples: Vec<HarnessExample>,
        cancellation: CancellationToken,
    ) -> Result<CandidateEvaluation> {
        let semaphore = Arc::new(Semaphore::new(run.config.max_concurrency.max(1)));
        let mut handles = Vec::new();
        for example in examples {
            let permit = semaphore
                .clone()
                .acquire_owned()
                .await
                .map_err(|_| HarnessOptError::Cancelled)?;
            let project = self.project.clone();
            let run = run.clone();
            let candidate = candidate.clone();
            let cancellation = cancellation.clone();
            handles.push(tokio::spawn(async move {
                let _permit = permit;
                if cancellation.is_cancelled() {
                    return Err(HarnessOptError::Cancelled);
                }
                let context = TraceContext {
                    run_id: Some(run.run_id.clone()),
                    experiment_id: Some(run.experiment_id.clone()),
                    candidate_id: Some(candidate.id.clone()),
                    candidate_parent_id: candidate.parent_id.clone(),
                    example_id: Some(example.id.clone()),
                    split: Some(example.split.as_str().to_string()),
                    ..TraceContext::default()
                };
                if let Some(timeout_secs) = run.config.per_example_timeout_secs {
                    tokio::time::timeout(
                        std::time::Duration::from_secs(timeout_secs),
                        project.evaluate_example(&run, &candidate, &example, context, cancellation),
                    )
                    .await
                    .map_err(|_| HarnessOptError::Harness("example timed out".to_string()))?
                } else {
                    project
                        .evaluate_example(&run, &candidate, &example, context, cancellation)
                        .await
                }
            }));
        }

        let mut evaluations = BTreeMap::new();
        let mut traces = BTreeMap::new();
        let mut artifacts = BTreeMap::new();
        let mut metric_calls = BTreeMap::new();
        for handle in handles {
            let run = handle
                .await
                .map_err(|error| HarnessOptError::Harness(error.to_string()))??;
            let example_id = run.result.example_id.clone();
            evaluations.insert(example_id.clone(), run.result);
            if let Some(trace) = run.trace {
                traces.insert(example_id.clone(), trace);
            }
            artifacts.insert(example_id.clone(), run.artifacts);
            metric_calls.insert(example_id, run.metric_calls);
        }

        Ok(CandidateEvaluation {
            candidate,
            evaluations,
            traces,
            artifacts,
            metric_calls,
        })
    }
}

#[async_trait]
pub trait OptimizerStrategy: Send + Sync {
    async fn propose(
        &self,
        request: StrategyRequest,
        cancellation: CancellationToken,
    ) -> Result<Vec<CandidateProposal>>;
}

pub struct HarnessOptimizer<R, S> {
    runner: R,
    strategy: S,
}

impl<R, S> HarnessOptimizer<R, S>
where
    R: HarnessRunner,
    S: OptimizerStrategy,
{
    pub fn new(runner: R, strategy: S) -> Self {
        Self { runner, strategy }
    }

    pub async fn run(
        &self,
        run: OptimizationRun,
        seed: Candidate,
        trainset: Vec<HarnessExample>,
        valset: Vec<HarnessExample>,
        cancellation: CancellationToken,
    ) -> Result<OptimizationState> {
        let store = SqliteHarnessStore::open(&run.run_dir).await?;
        self.run_with_store(run, seed, trainset, valset, &store, cancellation)
            .await
    }

    pub async fn run_with_store<T>(
        &self,
        run: OptimizationRun,
        seed: Candidate,
        trainset: Vec<HarnessExample>,
        valset: Vec<HarnessExample>,
        store: &T,
        cancellation: CancellationToken,
    ) -> Result<OptimizationState>
    where
        T: HarnessOptStore,
    {
        tokio::fs::create_dir_all(&run.run_dir).await?;
        store.init_run(&run).await?;
        let run = store.load_run().await?.unwrap_or(run);
        let valset = if valset.is_empty() {
            trainset.clone()
        } else {
            valset
        };
        if store.candidates().await?.is_empty() {
            let fingerprint = candidate_fingerprint(&seed)?;
            store
                .upsert_candidate(&CandidateRecord {
                    candidate: seed.clone(),
                    fingerprint,
                    parent_ids: Vec::new(),
                    generation: 0,
                    source_strategy: "seed".to_string(),
                    component_cursor: 0,
                    discovery_budget: run.config.max_metric_calls,
                })
                .await?;
            self.evaluate_cached(&run, &seed, &valset, store, cancellation.clone())
                .await?;
        }

        let mut merge_invocations = 0usize;
        let mut iteration = next_generation(store).await?;
        loop {
            if cancellation.is_cancelled() {
                return Err(HarnessOptError::Cancelled);
            }
            let stats = store.stats().await?;
            if stats.metric_calls_used >= run.config.max_metric_calls {
                break;
            }
            if run
                .config
                .max_iterations
                .is_some_and(|max| iteration >= max)
            {
                break;
            }

            let state = load_optimization_state(&run, store).await?;
            let parent_state = select_parent(&state.evaluated_candidates, &run.config, iteration)
                .ok_or_else(|| {
                HarnessOptError::Store("optimizer has no candidates".to_string())
            })?;
            let parent = parent_state.candidate.clone();
            let parent_record = store
                .candidates()
                .await?
                .into_iter()
                .find(|record| record.candidate.id == parent.id)
                .ok_or_else(|| {
                    HarnessOptError::Store("missing selected parent record".to_string())
                })?;
            let selected_components = select_components(&parent_record, &run.config);
            let batch = select_batch(&trainset, run.config.minibatch_size, iteration + 1);
            let parent_train_state = self
                .evaluate_cached(&run, &parent, &batch, store, cancellation.clone())
                .await?;
            if run.config.skip_perfect_score
                && parent_train_state
                    .evaluations
                    .values()
                    .all(|result| result.score >= run.config.perfect_score)
            {
                store
                    .insert_proposal(&ProposalRecord {
                        run_id: run.run_id.clone(),
                        generation: iteration,
                        parent_ids: vec![parent.id.clone()],
                        selected_components,
                        minibatch_ids: batch.iter().map(|example| example.id.clone()).collect(),
                        patches: Vec::new(),
                        rlm_prompt_ref: None,
                        rlm_output_ref: None,
                        before_score: parent_train_state.mean_score(),
                        after_score: parent_train_state.mean_score(),
                        accepted: false,
                        reason: "skipped_perfect_minibatch".to_string(),
                        candidate_id: None,
                    })
                    .await?;
                iteration += 1;
                continue;
            }

            let mut evidence_parent = parent.clone();
            evidence_parent
                .mutable_components
                .retain(|id, _| selected_components.contains(id));
            let request = StrategyRequest {
                run_id: run.run_id.clone(),
                experiment_id: run.experiment_id.clone(),
                generation: iteration,
                artifact_dir: run.run_dir.join("proposals").join(iteration.to_string()),
                evidence: EvidenceBatch {
                    parent: evidence_parent,
                    evaluated_examples: parent_train_state.example_runs_for(&batch),
                    frontier: frontier_by_mode(&state.evaluated_candidates, &run.config.frontier),
                },
            };
            let proposals = self.strategy.propose(request, cancellation.clone()).await?;
            for proposal in proposals.into_iter().take(1) {
                validate_patch_scope(&proposal, &selected_components)?;
                let candidate = apply_proposal(&parent, iteration, &proposal)?;
                let candidate_train_state = self
                    .evaluate_cached(&run, &candidate, &batch, store, cancellation.clone())
                    .await?;
                let before_score = parent_train_state.mean_score();
                let after_score = candidate_train_state.mean_score();
                let accepted = after_score > before_score;
                if accepted {
                    let fingerprint = candidate_fingerprint(&candidate)?;
                    store
                        .upsert_candidate(&CandidateRecord {
                            candidate: candidate.clone(),
                            fingerprint,
                            parent_ids: vec![parent.id.clone()],
                            generation: iteration + 1,
                            source_strategy: "reflection".to_string(),
                            component_cursor: next_component_cursor(
                                &parent_record,
                                &run.config,
                                selected_components.len(),
                            ),
                            discovery_budget: run
                                .config
                                .max_metric_calls
                                .saturating_sub(store.stats().await?.metric_calls_used),
                        })
                        .await?;
                    self.evaluate_cached(&run, &candidate, &valset, store, cancellation.clone())
                        .await?;
                }
                store
                    .insert_proposal(&ProposalRecord {
                        run_id: run.run_id.clone(),
                        generation: iteration,
                        parent_ids: vec![parent.id.clone()],
                        selected_components: selected_components.clone(),
                        minibatch_ids: batch.iter().map(|example| example.id.clone()).collect(),
                        patches: proposal.patches,
                        rlm_prompt_ref: proposal
                            .metadata
                            .get("rlm_prompt_ref")
                            .and_then(Value::as_str)
                            .map(PathBuf::from),
                        rlm_output_ref: proposal
                            .metadata
                            .get("rlm_output_ref")
                            .and_then(Value::as_str)
                            .map(PathBuf::from),
                        before_score,
                        after_score,
                        accepted,
                        reason: if accepted {
                            "strict_sum_improved".to_string()
                        } else {
                            "strict_sum_not_improved".to_string()
                        },
                        candidate_id: Some(candidate.id),
                    })
                    .await?;
            }

            if run.config.use_merge && merge_invocations < run.config.max_merge_invocations {
                merge_invocations += usize::from(
                    self.try_merge(&run, &valset, store, iteration, cancellation.clone())
                        .await?,
                );
            }
            iteration += 1;
        }

        load_optimization_state(&run, store).await
    }

    async fn evaluate_cached<T>(
        &self,
        run: &OptimizationRun,
        candidate: &Candidate,
        examples: &[HarnessExample],
        store: &T,
        cancellation: CancellationToken,
    ) -> Result<CandidateEvaluation>
    where
        T: HarnessOptStore,
    {
        let fingerprint = candidate_fingerprint(candidate)?;
        let mut evaluations = BTreeMap::new();
        let mut traces = BTreeMap::new();
        let mut artifacts = BTreeMap::new();
        let mut metric_calls = BTreeMap::new();
        let mut misses = Vec::new();

        for example in examples {
            if let Some(cached) = store.cached_example(&fingerprint, example).await? {
                let mut hit = cached;
                hit.metric_calls = 0;
                record_example_run(store, candidate, &fingerprint, &hit, true).await?;
                evaluations.insert(hit.result.example_id.clone(), hit.result.clone());
                if let Some(trace) = hit.trace.clone() {
                    traces.insert(hit.result.example_id.clone(), trace);
                }
                artifacts.insert(hit.result.example_id.clone(), hit.artifacts.clone());
                metric_calls.insert(hit.result.example_id.clone(), 0);
            } else {
                misses.push(example.clone());
            }
        }

        if !misses.is_empty() {
            let miss_state = self
                .runner
                .evaluate_candidate(run, candidate.clone(), misses, cancellation)
                .await?;
            for example_id in miss_state.evaluations.keys() {
                let example = examples
                    .iter()
                    .find(|example| &example.id == example_id)
                    .ok_or_else(|| {
                        HarnessOptError::Store("evaluated unknown example".to_string())
                    })?;
                let example_run = ExampleRun {
                    example: example.clone(),
                    result: miss_state.evaluations[example_id].clone(),
                    trace: miss_state.traces.get(example_id).cloned(),
                    artifacts: miss_state
                        .artifacts
                        .get(example_id)
                        .cloned()
                        .unwrap_or_default(),
                    metric_calls: miss_state
                        .metric_calls
                        .get(example_id)
                        .copied()
                        .unwrap_or(1),
                };
                store.put_cached_example(&fingerprint, &example_run).await?;
                record_example_run(store, candidate, &fingerprint, &example_run, false).await?;
            }
            evaluations.extend(miss_state.evaluations);
            traces.extend(miss_state.traces);
            artifacts.extend(miss_state.artifacts);
            metric_calls.extend(miss_state.metric_calls);
        }

        Ok(CandidateEvaluation {
            candidate: candidate.clone(),
            evaluations,
            traces,
            artifacts,
            metric_calls,
        })
    }

    async fn try_merge<T>(
        &self,
        run: &OptimizationRun,
        valset: &[HarnessExample],
        store: &T,
        generation: usize,
        cancellation: CancellationToken,
    ) -> Result<bool>
    where
        T: HarnessOptStore,
    {
        let state = load_optimization_state(run, store).await?;
        let frontier = frontier_by_mode(&state.evaluated_candidates, &run.config.frontier);
        let Some((left, right)) = complementary_pair(&frontier) else {
            return Ok(false);
        };
        let merged = merge_candidates(&left.candidate, &right.candidate, generation)?;
        let sample = select_batch(valset, run.config.minibatch_size, generation + 17);
        let left_score = self
            .evaluate_cached(run, &left.candidate, &sample, store, cancellation.clone())
            .await?
            .mean_score();
        let right_score = self
            .evaluate_cached(run, &right.candidate, &sample, store, cancellation.clone())
            .await?
            .mean_score();
        let merged_state = self
            .evaluate_cached(run, &merged, &sample, store, cancellation.clone())
            .await?;
        let accepted = merged_state.mean_score() >= left_score.max(right_score);
        if accepted {
            store
                .upsert_candidate(&CandidateRecord {
                    fingerprint: candidate_fingerprint(&merged)?,
                    candidate: merged.clone(),
                    parent_ids: vec![left.candidate.id.clone(), right.candidate.id.clone()],
                    generation: generation + 1,
                    source_strategy: "merge".to_string(),
                    component_cursor: 0,
                    discovery_budget: run
                        .config
                        .max_metric_calls
                        .saturating_sub(store.stats().await?.metric_calls_used),
                })
                .await?;
            self.evaluate_cached(run, &merged, valset, store, cancellation)
                .await?;
        }
        store
            .insert_proposal(&ProposalRecord {
                run_id: run.run_id.clone(),
                generation,
                parent_ids: vec![left.candidate.id.clone(), right.candidate.id.clone()],
                selected_components: merged.mutable_components.keys().cloned().collect(),
                minibatch_ids: sample.iter().map(|example| example.id.clone()).collect(),
                patches: Vec::new(),
                rlm_prompt_ref: None,
                rlm_output_ref: None,
                before_score: left_score.max(right_score),
                after_score: merged_state.mean_score(),
                accepted,
                reason: if accepted {
                    "merge_subsample_tied_or_improved".to_string()
                } else {
                    "merge_subsample_worse".to_string()
                },
                candidate_id: Some(merged.id),
            })
            .await?;
        Ok(true)
    }
}

pub fn evaluation_cache_key(candidate: &Candidate, example: &HarnessExample) -> String {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    candidate_fingerprint(candidate)
        .unwrap_or_else(|_| candidate.id.clone())
        .hash(&mut hasher);
    example.id.hash(&mut hasher);
    example.split.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

pub fn candidate_fingerprint(candidate: &Candidate) -> Result<String> {
    #[derive(Serialize)]
    struct Fingerprint<'a> {
        mutable_components: &'a BTreeMap<String, MutableComponent>,
        immutable_context: &'a BTreeMap<String, Value>,
    }
    let bytes = serde_json::to_vec(&Fingerprint {
        mutable_components: &candidate.mutable_components,
        immutable_context: &candidate.immutable_context,
    })?;
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    bytes.hash(&mut hasher);
    Ok(format!("{:016x}", hasher.finish()))
}

async fn record_example_run<T>(
    store: &T,
    candidate: &Candidate,
    candidate_fingerprint: &str,
    run: &ExampleRun,
    cache_hit: bool,
) -> Result<()>
where
    T: HarnessOptStore,
{
    store
        .insert_evaluation(&EvaluationRecord {
            candidate_id: candidate.id.clone(),
            candidate_fingerprint: candidate_fingerprint.to_string(),
            example_id: run.result.example_id.clone(),
            split: run.result.split.clone(),
            score: run.result.score,
            metrics: run.result.metrics.clone(),
            feedback: run.result.feedback.clone(),
            diagnostics: run.result.diagnostics.clone(),
            artifacts: run.artifacts.clone(),
            trace: run.trace.clone(),
            metric_calls: if cache_hit {
                0
            } else {
                run.metric_calls.max(1)
            },
            cache_hit,
        })
        .await
}

async fn load_optimization_state<T>(run: &OptimizationRun, store: &T) -> Result<OptimizationState>
where
    T: HarnessOptStore,
{
    let candidates = store.candidates().await?;
    let evaluations = store.evaluations().await?;
    let mut by_candidate: BTreeMap<String, CandidateEvaluation> = candidates
        .into_iter()
        .map(|record| {
            (
                record.candidate.id.clone(),
                CandidateEvaluation {
                    candidate: record.candidate,
                    evaluations: BTreeMap::new(),
                    traces: BTreeMap::new(),
                    artifacts: BTreeMap::new(),
                    metric_calls: BTreeMap::new(),
                },
            )
        })
        .collect();
    for evaluation in evaluations {
        let Some(state) = by_candidate.get_mut(&evaluation.candidate_id) else {
            continue;
        };
        let result = EvaluationResult {
            example_id: evaluation.example_id.clone(),
            split: evaluation.split,
            score: evaluation.score,
            passed: Some(evaluation.score >= run.config.perfect_score),
            feedback: evaluation.feedback,
            metrics: evaluation.metrics,
            diagnostics: evaluation.diagnostics,
        };
        state
            .evaluations
            .insert(evaluation.example_id.clone(), result);
        if let Some(trace) = evaluation.trace {
            state.traces.insert(evaluation.example_id.clone(), trace);
        }
        state
            .artifacts
            .insert(evaluation.example_id.clone(), evaluation.artifacts);
        state
            .metric_calls
            .insert(evaluation.example_id, evaluation.metric_calls);
    }
    let evaluated_candidates = by_candidate.into_values().collect::<Vec<_>>();
    let stats = store.stats().await?;
    let best_candidate_id = best_state(&evaluated_candidates).map(|best| best.candidate.id.clone());
    Ok(OptimizationState {
        run: run.clone(),
        evaluated_candidates,
        best_candidate_id,
        metric_calls_used: stats.metric_calls_used,
        cache_hits: stats.cache_hits,
        cache_misses: stats.cache_misses,
        accepted_proposals: stats.accepted_proposals,
        rejected_proposals: stats.rejected_proposals,
    })
}

pub async fn load_state<T>(run: &OptimizationRun, store: &T) -> Result<OptimizationState>
where
    T: HarnessOptStore,
{
    load_optimization_state(run, store).await
}

async fn next_generation<T>(store: &T) -> Result<usize>
where
    T: HarnessOptStore,
{
    Ok(store
        .candidates()
        .await?
        .into_iter()
        .map(|record| record.generation)
        .max()
        .unwrap_or(0))
}

fn select_parent<'a>(
    states: &'a [CandidateEvaluation],
    config: &OptimizationConfig,
    iteration: usize,
) -> Option<&'a CandidateEvaluation> {
    match config.candidate_selection {
        CandidateSelection::CurrentBest => best_state(states),
        CandidateSelection::Pareto => select_pareto_parent(states, iteration),
        CandidateSelection::EpsilonGreedy => {
            if iteration.is_multiple_of(10) {
                states.get(iteration % states.len().max(1))
            } else {
                select_pareto_parent(states, iteration)
            }
        }
    }
}

fn select_components(record: &CandidateRecord, config: &OptimizationConfig) -> Vec<String> {
    let mut ids = record
        .candidate
        .mutable_components
        .keys()
        .cloned()
        .collect::<Vec<_>>();
    if matches!(config.component_selection, ComponentSelection::All) || ids.is_empty() {
        return ids;
    }
    ids.sort();
    vec![ids[record.component_cursor % ids.len()].clone()]
}

fn next_component_cursor(
    record: &CandidateRecord,
    config: &OptimizationConfig,
    selected_count: usize,
) -> usize {
    if matches!(config.component_selection, ComponentSelection::All) {
        record.component_cursor
    } else {
        record.component_cursor + selected_count.max(1)
    }
}

fn validate_patch_scope(
    proposal: &CandidateProposal,
    selected_components: &[String],
) -> Result<()> {
    let selected = selected_components.iter().collect::<BTreeSet<_>>();
    for patch in &proposal.patches {
        let component_id = match patch {
            ComponentPatch::ReplaceValue { component_id, .. } => component_id,
        };
        if !selected.contains(component_id) {
            return Err(HarnessOptError::InvalidProposal(format!(
                "proposal patched unselected component `{component_id}`"
            )));
        }
    }
    Ok(())
}

pub fn apply_proposal(
    parent: &Candidate,
    generation: usize,
    proposal: &CandidateProposal,
) -> Result<Candidate> {
    if proposal.parent_candidate_id != parent.id {
        return Err(HarnessOptError::InvalidProposal(format!(
            "proposal parent `{}` does not match selected parent `{}`",
            proposal.parent_candidate_id, parent.id
        )));
    }
    let mut candidate = parent.clone();
    candidate.id = format!("{}-g{}-{}", parent.id, generation, uuid::Uuid::new_v4());
    candidate.parent_id = Some(parent.id.clone());
    candidate.metadata.extend(proposal.metadata.clone());
    for patch in &proposal.patches {
        apply_patch(&mut candidate, patch)?;
    }
    Ok(candidate)
}

pub fn apply_patch(candidate: &mut Candidate, patch: &ComponentPatch) -> Result<()> {
    match patch {
        ComponentPatch::ReplaceValue {
            component_id,
            value,
        } => {
            let component = candidate
                .mutable_components
                .get_mut(component_id)
                .ok_or_else(|| HarnessOptError::UnknownComponent(component_id.clone()))?;
            validate_component_value(component_id, &component.constraints, value)?;
            component.value = value.clone();
        }
    }
    Ok(())
}

pub fn validate_component_value(
    component_id: &str,
    constraints: &ComponentConstraints,
    value: &ComponentValue,
) -> Result<()> {
    let Some(text) = value.text_for_constraints() else {
        return Ok(());
    };
    if let Some(max_chars) = constraints.max_chars
        && text.chars().count() > max_chars
    {
        return Err(HarnessOptError::ConstraintViolation {
            component_id: component_id.to_string(),
            reason: format!("text exceeds {max_chars} characters"),
        });
    }
    for term in &constraints.preserve_terms {
        if !text.contains(term) {
            return Err(HarnessOptError::ConstraintViolation {
                component_id: component_id.to_string(),
                reason: format!("missing preserved term `{term}`"),
            });
        }
    }
    for term in &constraints.forbidden_terms {
        if text.contains(term) {
            return Err(HarnessOptError::ConstraintViolation {
                component_id: component_id.to_string(),
                reason: format!("contains forbidden term `{term}`"),
            });
        }
    }
    Ok(())
}

pub fn select_batch(
    examples: &[HarnessExample],
    minibatch_size: usize,
    generation: usize,
) -> Vec<HarnessExample> {
    if examples.is_empty() {
        return Vec::new();
    }
    let limit = minibatch_size.max(1).min(examples.len());
    (0..limit)
        .map(|offset| examples[(generation + offset) % examples.len()].clone())
        .collect()
}

pub fn frontier(states: &[CandidateEvaluation]) -> Vec<CandidateEvaluation> {
    frontier_by_mode(states, &FrontierMode::Instance)
}

pub fn frontier_by_mode(
    states: &[CandidateEvaluation],
    mode: &FrontierMode,
) -> Vec<CandidateEvaluation> {
    states
        .iter()
        .filter(|candidate| {
            !states.iter().any(|other| {
                other.candidate.id != candidate.candidate.id
                    && dominates_by_mode(other, candidate, mode)
            })
        })
        .cloned()
        .collect()
}

fn best_state(states: &[CandidateEvaluation]) -> Option<&CandidateEvaluation> {
    states.iter().max_by(|left, right| {
        left.mean_score()
            .partial_cmp(&right.mean_score())
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| right.candidate.id.cmp(&left.candidate.id))
    })
}

fn select_pareto_parent(
    states: &[CandidateEvaluation],
    generation: usize,
) -> Option<&CandidateEvaluation> {
    let mut best_by_example: BTreeMap<String, f64> = BTreeMap::new();
    for state in states {
        for (example_id, result) in &state.evaluations {
            best_by_example
                .entry(example_id.clone())
                .and_modify(|score| {
                    if result.score > *score {
                        *score = result.score;
                    }
                })
                .or_insert(result.score);
        }
    }

    let mut covered_candidate_ids = Vec::new();
    for state in states {
        for (example_id, result) in &state.evaluations {
            if best_by_example
                .get(example_id)
                .is_some_and(|score| result.score == *score)
            {
                covered_candidate_ids.push(state.candidate.id.clone());
            }
        }
    }

    if covered_candidate_ids.is_empty() {
        return best_state(states);
    }
    covered_candidate_ids.sort();
    let selected_id = &covered_candidate_ids[generation % covered_candidate_ids.len()];
    states
        .iter()
        .find(|state| &state.candidate.id == selected_id)
        .or_else(|| best_state(states))
}

fn dominates(left: &CandidateEvaluation, right: &CandidateEvaluation) -> bool {
    dominates_by_mode(left, right, &FrontierMode::Instance)
}

fn dominates_by_mode(
    left: &CandidateEvaluation,
    right: &CandidateEvaluation,
    mode: &FrontierMode,
) -> bool {
    match mode {
        FrontierMode::Instance => dominates_instances(left, right),
        FrontierMode::Objective => dominates_objectives(left, right),
        FrontierMode::Hybrid | FrontierMode::Cartesian => {
            dominates_instances(left, right) || dominates_objectives(left, right)
        }
    }
}

fn dominates_instances(left: &CandidateEvaluation, right: &CandidateEvaluation) -> bool {
    let shared = left
        .evaluations
        .keys()
        .filter(|example_id| right.evaluations.contains_key(*example_id))
        .cloned()
        .collect::<BTreeSet<_>>();
    if shared.is_empty() {
        return false;
    }
    let mut strictly_better = false;
    for example_id in shared {
        let left_score = left.evaluations[&example_id].score;
        let right_score = right.evaluations[&example_id].score;
        if left_score < right_score {
            return false;
        }
        strictly_better |= left_score > right_score;
    }
    strictly_better
}

fn dominates_objectives(left: &CandidateEvaluation, right: &CandidateEvaluation) -> bool {
    let mut shared = BTreeSet::new();
    for result in left.evaluations.values() {
        shared.extend(result.metrics.keys().cloned());
    }
    shared.retain(|key| {
        right
            .evaluations
            .values()
            .any(|result| result.metrics.contains_key(key))
    });
    if shared.is_empty() {
        return false;
    }
    let mut strictly_better = false;
    for key in shared {
        let left_score = mean_metric(left, &key);
        let right_score = mean_metric(right, &key);
        if left_score < right_score {
            return false;
        }
        strictly_better |= left_score > right_score;
    }
    strictly_better
}

fn mean_metric(state: &CandidateEvaluation, key: &str) -> f64 {
    let values = state
        .evaluations
        .values()
        .filter_map(|result| result.metrics.get(key).copied())
        .collect::<Vec<_>>();
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn complementary_pair(
    states: &[CandidateEvaluation],
) -> Option<(&CandidateEvaluation, &CandidateEvaluation)> {
    for left in states {
        for right in states {
            if left.candidate.id != right.candidate.id
                && !dominates(left, right)
                && !dominates(right, left)
            {
                return Some((left, right));
            }
        }
    }
    None
}

fn merge_candidates(left: &Candidate, right: &Candidate, generation: usize) -> Result<Candidate> {
    let mut candidate = left.clone();
    candidate.id = format!("merge-g{}-{}", generation, uuid::Uuid::new_v4());
    candidate.parent_id = Some(left.id.clone());
    for (index, (id, component)) in right.mutable_components.iter().enumerate() {
        if index % 2 == 1 {
            candidate
                .mutable_components
                .insert(id.clone(), component.clone());
        }
    }
    candidate.metadata.insert(
        "merge_parent_ids".to_string(),
        json!([left.id.clone(), right.id.clone()]),
    );
    for component in candidate.mutable_components.values() {
        validate_component_value(&component.id, &component.constraints, &component.value)?;
    }
    Ok(candidate)
}

pub mod clbench {
    use super::*;

    pub const MEMORY_GUIDANCE_COMPONENT: &str = "clbench.memory_guidance";
    pub const PROMPT_TEMPLATE_COMPONENT: &str = "clbench.prompt_template";
    pub const USER_DIRECTIVE_COMPONENT: &str = "clbench.user_directive";

    pub const CLBENCH_MEMORY_GUIDANCE: &str = r#"Use your persistent RLM REPL as memory. The following globals are bound for this turn:

- `iteration: int` — current benchmark iteration
- `current_query: str` — the task this turn must answer
- `current_feedback: str | null` — score/feedback from the previous action; `null` on the first iteration
- `diary: list` — persistent across iterations; each entry is `{ history_index: int, summary: str, learnings: str }`
- `history` — auto-bound read-only projection of past turn entries; index into it via `entry.history_index`

The current query and previous feedback are visible in the user turn and are also bound as `current_query` and `current_feedback` for exact access from lashlang. The shape your `submit` value must take is shown in the **Required output** block at the end of the user turn — consult it before building the action.

At the start of each turn:

- Use the visible current query and feedback first. Inspect `diary` only when you need details not already visible.
- Pull the matching `history[entry.history_index]` when an entry's `summary` is too compressed to act on.
- Apply prior learnings before choosing the benchmark action.

Before every `submit`, append exactly one diary record and submit a value matching the **Required output** shape:

```lashlang
diary = push(diary, {
  history_index: len(history) - 1,
  summary: "brief task/action summary",
  learnings: "reusable lesson from this interaction"
})
submit answer
```

`answer` must match the **Required output** block exactly — the shape varies per task and per step, so build the value to fit the announced contract rather than assuming a fixed wrapper. Keep diary entries short, factual, and reusable; do not duplicate old lessons; incorporate feedback and revise strategy in later entries."#;

    pub const DEFAULT_USER_DIRECTIVE: &str = "Choose the next benchmark action.";

    #[derive(Clone, Debug, Serialize, Deserialize)]
    pub struct ClbenchConfig {
        #[serde(default = "default_experiment_id")]
        pub experiment_id: String,
        #[serde(default)]
        pub train: Vec<HarnessExample>,
        #[serde(default)]
        pub val: Vec<HarnessExample>,
        #[serde(default)]
        pub test: Vec<HarnessExample>,
    }

    fn default_experiment_id() -> String {
        "clbench".to_string()
    }

    #[derive(Clone, Debug)]
    pub struct ClbenchProject {
        config: ClbenchConfig,
    }

    impl ClbenchProject {
        pub fn new(config: ClbenchConfig) -> Self {
            Self { config }
        }

        pub fn seed_candidate_static() -> Candidate {
            Candidate {
                id: "seed".to_string(),
                parent_id: None,
                mutable_components: BTreeMap::new(),
                immutable_context: BTreeMap::from([(
                    "tool_surface".to_string(),
                    json!([
                        "llm_query",
                        "spawn_agent",
                        "continue_as",
                        "list_async_handles"
                    ]),
                )]),
                metadata: BTreeMap::from([("project".to_string(), json!("clbench"))]),
            }
            .with_component(MutableComponent {
                id: MEMORY_GUIDANCE_COMPONENT.to_string(),
                description: Some("CLBench-specific persistent memory guidance".to_string()),
                value: ComponentValue::Text {
                    text: CLBENCH_MEMORY_GUIDANCE.to_string(),
                },
                constraints: ComponentConstraints {
                    max_chars: Some(8_000),
                    preserve_terms: vec![
                        "diary".to_string(),
                        "submit".to_string(),
                        "llm_query".to_string(),
                        "spawn_agent".to_string(),
                        "continue_as".to_string(),
                        "list_async_handles".to_string(),
                    ],
                    forbidden_terms: vec![
                        "exec_command".to_string(),
                        "read_file".to_string(),
                        "apply_patch".to_string(),
                    ],
                    format_hint: Some(
                        "Markdown guidance with a fenced lashlang diary example".to_string(),
                    ),
                },
            })
            .with_component(MutableComponent {
                id: PROMPT_TEMPLATE_COMPONENT.to_string(),
                description: Some(
                    "CLBench-specific prompt-template section layout and static text".to_string(),
                ),
                value: ComponentValue::PromptTemplate {
                    template: clbench_prompt_template(CLBENCH_MEMORY_GUIDANCE),
                },
                constraints: ComponentConstraints {
                    preserve_terms: vec!["Continual Memory".to_string(), "Execution".to_string()],
                    ..Default::default()
                },
            })
            .with_component(MutableComponent {
                id: USER_DIRECTIVE_COMPONENT.to_string(),
                description: Some("Per-turn CLBench next-action directive".to_string()),
                value: ComponentValue::Text {
                    text: DEFAULT_USER_DIRECTIVE.to_string(),
                },
                constraints: ComponentConstraints {
                    max_chars: Some(1_000),
                    preserve_terms: vec!["submit".to_string()],
                    ..Default::default()
                },
            })
        }

        pub fn prompt_template(candidate: &Candidate) -> Result<PromptTemplate> {
            match &candidate.component(PROMPT_TEMPLATE_COMPONENT)?.value {
                ComponentValue::PromptTemplate { template } => Ok(template.clone()),
                _ => Err(HarnessOptError::ConstraintViolation {
                    component_id: PROMPT_TEMPLATE_COMPONENT.to_string(),
                    reason: "expected prompt_template component".to_string(),
                }),
            }
        }

        pub fn user_directive(candidate: &Candidate) -> Result<String> {
            match &candidate.component(USER_DIRECTIVE_COMPONENT)?.value {
                ComponentValue::Text { text } => Ok(text.clone()),
                _ => Err(HarnessOptError::ConstraintViolation {
                    component_id: USER_DIRECTIVE_COMPONENT.to_string(),
                    reason: "expected text component".to_string(),
                }),
            }
        }

        pub async fn write_seed_candidate(path: &Path) -> Result<()> {
            let candidate = Self::seed_candidate_static();
            tokio::fs::write(path, serde_json::to_vec_pretty(&candidate)?).await?;
            Ok(())
        }
    }

    #[async_trait]
    impl HarnessProject for ClbenchProject {
        async fn seed_candidate(&self) -> Result<Candidate> {
            Ok(Self::seed_candidate_static())
        }

        async fn trainset(&self) -> Result<Vec<HarnessExample>> {
            Ok(self.config.train.clone())
        }

        async fn valset(&self) -> Result<Vec<HarnessExample>> {
            Ok(self.config.val.clone())
        }

        async fn testset(&self) -> Result<Vec<HarnessExample>> {
            Ok(self.config.test.clone())
        }

        async fn evaluate_example(
            &self,
            run: &OptimizationRun,
            candidate: &Candidate,
            example: &HarnessExample,
            context: TraceContext,
            _cancellation: CancellationToken,
        ) -> Result<ExampleRun> {
            let example_dir = run
                .run_dir
                .join("examples")
                .join(&candidate.id)
                .join(&example.id);
            tokio::fs::create_dir_all(&example_dir).await?;
            let request_path = example_dir.join("request.json");
            let candidate_path = example_dir.join("candidate.json");
            let response_path = example_dir.join("response.json");
            let trace_path = example_dir.join("typed_trace.jsonl");
            let evidence_path = example_dir.join("rendered_evidence.json");
            let score_path = example_dir.join("score_report.json");
            let session_db = example_dir.join("session.db");

            let directive = Self::user_directive(candidate)?;
            let request = json!({
                "input": example.input,
                "expected": example.expected,
                "directive": directive,
                "trace_context": context,
            });
            tokio::fs::write(&request_path, serde_json::to_vec_pretty(&request)?).await?;
            tokio::fs::write(&candidate_path, serde_json::to_vec_pretty(candidate)?).await?;

            let expected = example.expected.as_ref();
            let score = expected
                .and_then(|expected| expected.get("score"))
                .and_then(Value::as_f64)
                .unwrap_or(0.0);
            let feedback = expected
                .and_then(|expected| expected.get("feedback"))
                .and_then(Value::as_str)
                .map(ToOwned::to_owned);
            let result = EvaluationResult {
                example_id: example.id.clone(),
                split: example.split.clone(),
                score,
                passed: Some(score >= 1.0),
                feedback,
                metrics: BTreeMap::new(),
                diagnostics: BTreeMap::from([
                    ("tool_call_count".to_string(), json!(0)),
                    ("error_count".to_string(), json!(0)),
                    ("turn_outcome".to_string(), json!("synthetic")),
                ]),
            };
            let trace = TraceBundle {
                example_id: example.id.clone(),
                records: vec![TraceRecord::new(
                    context,
                    TraceEvent::TurnStarted {
                        metadata: BTreeMap::from([("project".to_string(), json!("clbench"))]),
                    },
                )],
            };
            let evidence = strategies::gepa::render_reflective_evidence(&[ExampleRun {
                example: example.clone(),
                result: result.clone(),
                trace: Some(trace.clone()),
                artifacts: RunArtifacts::default(),
                metric_calls: 1,
            }]);
            tokio::fs::write(
                &response_path,
                serde_json::to_vec_pretty(&json!({ "action": null }))?,
            )
            .await?;
            tokio::fs::write(
                &trace_path,
                trace
                    .records
                    .iter()
                    .map(serde_json::to_string)
                    .collect::<std::result::Result<Vec<_>, _>>()?
                    .join("\n"),
            )
            .await?;
            tokio::fs::write(&evidence_path, serde_json::to_vec_pretty(&evidence)?).await?;
            tokio::fs::write(&score_path, serde_json::to_vec_pretty(&result)?).await?;

            Ok(ExampleRun {
                example: example.clone(),
                result,
                trace: Some(trace),
                artifacts: RunArtifacts {
                    request_json: Some(request_path),
                    candidate_json: Some(candidate_path),
                    response_json: Some(response_path),
                    session_db: Some(session_db),
                    typed_trace_jsonl: Some(trace_path),
                    rendered_evidence_json: Some(evidence_path),
                    score_report_json: Some(score_path),
                },
                metric_calls: 1,
            })
        }
    }

    pub fn clbench_prompt_template(memory_guidance: &str) -> PromptTemplate {
        PromptTemplate::new(vec![
            PromptTemplateSection::untitled(vec![PromptTemplateEntry::text(
                "You are being evaluated by Continual Learning Bench, which tests whether an agent improves from feedback across sequential task instances.",
            )]),
            PromptTemplateSection::titled(
                "Execution",
                vec![
                    PromptTemplateEntry::builtin(PromptBuiltin::ExecutionInstructions),
                    PromptTemplateEntry::slot(PromptSlot::Execution),
                ],
            ),
            PromptTemplateSection::titled(
                "Continual Memory",
                vec![PromptTemplateEntry::text(memory_guidance)],
            ),
            PromptTemplateSection::titled(
                "Guidance",
                vec![
                    PromptTemplateEntry::slot(PromptSlot::Guidance),
                    PromptTemplateEntry::slot(PromptSlot::ProjectInstructions),
                ],
            ),
        ])
    }
}

pub mod strategies {
    pub mod gepa {
        use super::super::*;

        #[derive(Clone, Debug, Default)]
        pub struct ReflectiveGepaStrategy<P> {
            proposer: P,
        }

        impl<P> ReflectiveGepaStrategy<P> {
            pub fn new(proposer: P) -> Self {
                Self { proposer }
            }
        }

        #[async_trait]
        pub trait ReflectiveProposer: Send + Sync {
            async fn propose_json(
                &self,
                request: ReflectiveProposalRequest,
                cancellation: CancellationToken,
            ) -> Result<Value>;
        }

        #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
        pub struct ReflectiveProposalRequest {
            pub run_id: String,
            pub experiment_id: String,
            pub generation: usize,
            pub artifact_dir: PathBuf,
            pub parent_candidate_id: String,
            pub mutable_components: BTreeMap<String, MutableComponent>,
            pub evidence: Value,
            pub output_schema: Value,
        }

        #[async_trait]
        impl<P> OptimizerStrategy for ReflectiveGepaStrategy<P>
        where
            P: ReflectiveProposer,
        {
            async fn propose(
                &self,
                request: StrategyRequest,
                cancellation: CancellationToken,
            ) -> Result<Vec<CandidateProposal>> {
                let reflective_request = ReflectiveProposalRequest {
                    run_id: request.run_id,
                    experiment_id: request.experiment_id,
                    generation: request.generation,
                    artifact_dir: request.artifact_dir,
                    parent_candidate_id: request.evidence.parent.id.clone(),
                    mutable_components: request.evidence.parent.mutable_components.clone(),
                    evidence: render_reflective_evidence(&request.evidence.evaluated_examples),
                    output_schema: proposal_output_schema(),
                };
                let value = self
                    .proposer
                    .propose_json(reflective_request, cancellation)
                    .await?;
                parse_candidate_proposals(value)
            }
        }

        pub fn proposal_output_schema() -> Value {
            json!({
                "type": "object",
                "additionalProperties": false,
                "required": ["proposals"],
                "properties": {
                    "proposals": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "additionalProperties": false,
                            "required": ["parent_candidate_id", "patches"],
                            "properties": {
                                "parent_candidate_id": { "type": "string" },
                                "rationale": { "type": "string" },
                                "patches": {
                                    "type": "array",
                                    "minItems": 1,
                                    "items": {
                                        "type": "object",
                                        "additionalProperties": false,
                                        "required": ["kind", "component_id", "value"],
                                        "properties": {
                                            "kind": { "const": "replace_value" },
                                            "component_id": { "type": "string" },
                                            "value": {
                                                "oneOf": [
                                                    {
                                                        "type": "object",
                                                        "additionalProperties": false,
                                                        "required": ["kind", "text"],
                                                        "properties": {
                                                            "kind": { "const": "text" },
                                                            "text": { "type": "string" }
                                                        }
                                                    },
                                                    {
                                                        "type": "object",
                                                        "additionalProperties": false,
                                                        "required": ["kind", "value"],
                                                        "properties": {
                                                            "kind": { "const": "json" },
                                                            "value": {}
                                                        }
                                                    },
                                                    {
                                                        "type": "object",
                                                        "additionalProperties": false,
                                                        "required": ["kind", "template"],
                                                        "properties": {
                                                            "kind": { "const": "prompt_template" },
                                                            "template": { "type": "object" }
                                                        }
                                                    },
                                                    {
                                                        "type": "object",
                                                        "additionalProperties": false,
                                                        "required": ["kind", "contribution"],
                                                        "properties": {
                                                            "kind": { "const": "prompt_contribution" },
                                                            "contribution": { "type": "object" }
                                                        }
                                                    }
                                                ]
                                            }
                                        }
                                    }
                                },
                                "metadata": { "type": "object" }
                            }
                        }
                    }
                }
            })
        }

        pub fn parse_candidate_proposals(value: Value) -> Result<Vec<CandidateProposal>> {
            let proposals = value
                .get("proposals")
                .cloned()
                .ok_or_else(|| HarnessOptError::InvalidProposal("missing proposals".to_string()))?;
            let proposals: Vec<CandidateProposal> = serde_json::from_value(proposals)
                .map_err(|error| HarnessOptError::InvalidProposal(error.to_string()))?;
            if proposals.is_empty() {
                return Err(HarnessOptError::InvalidProposal(
                    "proposal list is empty".to_string(),
                ));
            }
            Ok(proposals)
        }

        pub fn render_reflective_evidence(runs: &[ExampleRun]) -> Value {
            let examples = runs
                .iter()
                .map(|run| {
                    let records = run
                        .trace
                        .as_ref()
                        .map(|trace| {
                            trace
                                .records
                                .iter()
                                .map(render_trace_record)
                                .collect::<Vec<_>>()
                        })
                        .unwrap_or_default();
                    json!({
                        "example_id": run.example.id,
                        "split": run.example.split,
                        "score": run.result.score,
                        "passed": run.result.passed,
                        "feedback": run.result.feedback,
                        "turn_outcome": run.result.diagnostics.get("turn_outcome"),
                        "tool_call_count": run.result.diagnostics.get("tool_call_count"),
                        "error_count": run.result.diagnostics.get("error_count"),
                        "diary_behavior": run.result.diagnostics.get("diary_behavior"),
                        "schema_failure": run.result.diagnostics.get("schema_failure"),
                        "trace": records,
                    })
                })
                .collect::<Vec<_>>();
            json!({ "examples": examples })
        }

        fn render_trace_record(record: &TraceRecord) -> Value {
            let event = match &record.event {
                TraceEvent::TurnStarted { .. } => "turn_started",
                TraceEvent::TurnCompleted { .. } => "turn_completed",
                TraceEvent::ToolCallStarted { .. } => "tool_call_started",
                TraceEvent::ToolCallCompleted { .. } => "tool_call_completed",
                TraceEvent::LlmCallFailed { .. } => "llm_call_failed",
                TraceEvent::LlmCallStarted { .. } => "llm_call_started",
                TraceEvent::LlmCallCompleted { .. } => "llm_call_completed",
                TraceEvent::PromptBuilt { .. } => "prompt_built",
                _ => "other",
            };
            json!({
                "event": event,
                "timestamp": record.timestamp,
                "context": {
                    "run_id": record.context.run_id,
                    "experiment_id": record.context.experiment_id,
                    "candidate_id": record.context.candidate_id,
                    "candidate_parent_id": record.context.candidate_parent_id,
                    "example_id": record.context.example_id,
                    "split": record.context.split,
                }
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::strategies::gepa::{
        ReflectiveGepaStrategy, ReflectiveProposalRequest, ReflectiveProposer,
        parse_candidate_proposals, render_reflective_evidence,
    };
    use super::*;

    fn example(id: &str, split: Split) -> HarnessExample {
        HarnessExample {
            id: id.to_string(),
            split,
            input: json!({"question": id}),
            expected: None,
            metadata: BTreeMap::new(),
        }
    }

    fn seed_candidate() -> Candidate {
        Candidate {
            id: "seed".to_string(),
            parent_id: None,
            mutable_components: BTreeMap::new(),
            immutable_context: BTreeMap::new(),
            metadata: BTreeMap::new(),
        }
        .with_component(MutableComponent {
            id: "instruction".to_string(),
            description: Some("instruction".to_string()),
            value: ComponentValue::Text {
                text: "do ok".to_string(),
            },
            constraints: ComponentConstraints {
                max_chars: Some(32),
                ..Default::default()
            },
        })
    }

    #[test]
    fn component_patch_validation_rejects_invalid_mutation() {
        let mut candidate = seed_candidate();
        let err = apply_patch(
            &mut candidate,
            &ComponentPatch::ReplaceValue {
                component_id: "instruction".to_string(),
                value: ComponentValue::Text {
                    text: "this text is far too long for the configured limit".to_string(),
                },
            },
        )
        .unwrap_err();

        assert!(matches!(err, HarnessOptError::ConstraintViolation { .. }));
    }

    #[test]
    fn candidate_lineage_sets_parent_id() {
        let proposal = CandidateProposal {
            parent_candidate_id: "seed".to_string(),
            patches: vec![ComponentPatch::ReplaceValue {
                component_id: "instruction".to_string(),
                value: ComponentValue::Text {
                    text: "do better".to_string(),
                },
            }],
            rationale: None,
            metadata: BTreeMap::new(),
        };

        let candidate = apply_proposal(&seed_candidate(), 2, &proposal).unwrap();

        assert_eq!(candidate.parent_id.as_deref(), Some("seed"));
        assert!(candidate.id.starts_with("seed-g2-"));
    }

    #[test]
    fn train_val_split_batches_only_given_examples() {
        let train = vec![example("a", Split::Train), example("b", Split::Train)];
        let val = [example("c", Split::Val)];

        let batch = select_batch(&train, 8, 0);

        assert_eq!(batch.len(), 2);
        assert!(batch.iter().all(|example| example.split == Split::Train));
        assert_eq!(val[0].split, Split::Val);
    }

    #[test]
    fn evaluation_cache_key_includes_split() {
        let candidate = seed_candidate();
        let train = example("x", Split::Train);
        let val = example("x", Split::Val);

        assert_ne!(
            evaluation_cache_key(&candidate, &train),
            evaluation_cache_key(&candidate, &val)
        );
    }

    #[test]
    fn candidate_fingerprint_excludes_lineage_and_metadata() {
        let mut left = seed_candidate();
        let mut right = left.clone();
        right.id = "other".to_string();
        right.parent_id = Some("parent".to_string());
        right.metadata.insert("note".to_string(), json!("ignored"));

        assert_eq!(
            candidate_fingerprint(&left).unwrap(),
            candidate_fingerprint(&right).unwrap()
        );

        apply_patch(
            &mut left,
            &ComponentPatch::ReplaceValue {
                component_id: "instruction".to_string(),
                value: ComponentValue::Text {
                    text: "do better".to_string(),
                },
            },
        )
        .unwrap();

        assert_ne!(
            candidate_fingerprint(&left).unwrap(),
            candidate_fingerprint(&right).unwrap()
        );
    }

    #[tokio::test]
    async fn sqlite_store_roundtrips_records_and_cache_stats() {
        let temp =
            std::env::temp_dir().join(format!("lash-harness-opt-store-{}", uuid::Uuid::new_v4()));
        let store = SqliteHarnessStore::open(&temp).await.unwrap();
        let run = OptimizationRun {
            run_id: "run".to_string(),
            experiment_id: "mock".to_string(),
            run_dir: temp,
            config: OptimizationConfig::default(),
        };
        store.init_run(&run).await.unwrap();
        let candidate = seed_candidate();
        let fingerprint = candidate_fingerprint(&candidate).unwrap();
        store
            .upsert_candidate(&CandidateRecord {
                candidate: candidate.clone(),
                fingerprint: fingerprint.clone(),
                parent_ids: Vec::new(),
                generation: 0,
                source_strategy: "seed".to_string(),
                component_cursor: 0,
                discovery_budget: 10,
            })
            .await
            .unwrap();
        let example_run = ExampleRun {
            example: example("ex", Split::Train),
            result: EvaluationResult {
                example_id: "ex".to_string(),
                split: Split::Train,
                score: 0.7,
                passed: Some(false),
                feedback: Some("try again".to_string()),
                metrics: BTreeMap::from([("reward".to_string(), 0.7)]),
                diagnostics: BTreeMap::new(),
            },
            trace: None,
            artifacts: RunArtifacts::default(),
            metric_calls: 2,
        };
        store
            .put_cached_example(&fingerprint, &example_run)
            .await
            .unwrap();
        record_example_run(&store, &candidate, &fingerprint, &example_run, false)
            .await
            .unwrap();
        let cached = store
            .cached_example(&fingerprint, &example_run.example)
            .await
            .unwrap()
            .unwrap();

        assert_eq!(cached.result.score, 0.7);
        assert_eq!(store.candidates().await.unwrap().len(), 1);
        assert_eq!(store.evaluations().await.unwrap().len(), 1);
        assert_eq!(store.stats().await.unwrap().metric_calls_used, 2);
    }

    #[test]
    fn gepa_proposal_schema_validation_rejects_missing_proposals() {
        let err = parse_candidate_proposals(json!({ "patches": [] })).unwrap_err();

        assert!(matches!(err, HarnessOptError::InvalidProposal(_)));
    }

    #[test]
    fn reflective_evidence_renders_trace_feedback() {
        let context = TraceContext {
            run_id: Some("run".to_string()),
            experiment_id: Some("exp".to_string()),
            candidate_id: Some("cand".to_string()),
            candidate_parent_id: Some("parent".to_string()),
            example_id: Some("ex".to_string()),
            split: Some("train".to_string()),
            ..TraceContext::default()
        };
        let run = ExampleRun {
            example: example("ex", Split::Train),
            result: EvaluationResult {
                example_id: "ex".to_string(),
                split: Split::Train,
                score: 0.5,
                passed: Some(false),
                feedback: Some("needs better memory use".to_string()),
                metrics: BTreeMap::new(),
                diagnostics: BTreeMap::from([
                    ("tool_call_count".to_string(), json!(2)),
                    ("error_count".to_string(), json!(1)),
                ]),
            },
            trace: Some(TraceBundle {
                example_id: "ex".to_string(),
                records: vec![TraceRecord::new(
                    context,
                    TraceEvent::TurnStarted {
                        metadata: BTreeMap::new(),
                    },
                )],
            }),
            artifacts: RunArtifacts::default(),
            metric_calls: 1,
        };

        let evidence = render_reflective_evidence(&[run]);

        assert_eq!(
            evidence["examples"][0]["feedback"],
            "needs better memory use"
        );
        assert_eq!(
            evidence["examples"][0]["trace"][0]["context"]["run_id"],
            "run"
        );
    }

    struct ScoreProject;

    #[async_trait]
    impl HarnessProject for ScoreProject {
        async fn seed_candidate(&self) -> Result<Candidate> {
            Ok(seed_candidate())
        }

        async fn trainset(&self) -> Result<Vec<HarnessExample>> {
            Ok(vec![example("ex", Split::Train)])
        }

        async fn valset(&self) -> Result<Vec<HarnessExample>> {
            Ok(Vec::new())
        }

        async fn evaluate_example(
            &self,
            _run: &OptimizationRun,
            candidate: &Candidate,
            example: &HarnessExample,
            _context: TraceContext,
            _cancellation: CancellationToken,
        ) -> Result<ExampleRun> {
            let score = match &candidate.component("instruction")?.value {
                ComponentValue::Text { text } if text.contains("better") => 1.0,
                _ => 0.1,
            };
            Ok(ExampleRun {
                example: example.clone(),
                result: EvaluationResult {
                    example_id: example.id.clone(),
                    split: example.split.clone(),
                    score,
                    passed: Some(score > 0.5),
                    feedback: None,
                    metrics: BTreeMap::new(),
                    diagnostics: BTreeMap::new(),
                },
                trace: None,
                artifacts: RunArtifacts::default(),
                metric_calls: 1,
            })
        }
    }

    struct BetterProposer;

    #[async_trait]
    impl ReflectiveProposer for BetterProposer {
        async fn propose_json(
            &self,
            request: ReflectiveProposalRequest,
            _cancellation: CancellationToken,
        ) -> Result<Value> {
            Ok(json!({
                "proposals": [{
                    "parent_candidate_id": request.parent_candidate_id,
                    "patches": [{
                        "kind": "replace_value",
                        "component_id": "instruction",
                        "value": { "kind": "text", "text": "do better" }
                    }]
                }]
            }))
        }
    }

    struct WorseProposer;

    #[async_trait]
    impl ReflectiveProposer for WorseProposer {
        async fn propose_json(
            &self,
            request: ReflectiveProposalRequest,
            _cancellation: CancellationToken,
        ) -> Result<Value> {
            Ok(json!({
                "proposals": [{
                    "parent_candidate_id": request.parent_candidate_id,
                    "patches": [{
                        "kind": "replace_value",
                        "component_id": "instruction",
                        "value": { "kind": "text", "text": "do ok" }
                    }]
                }]
            }))
        }
    }

    struct CountingProposer(Arc<std::sync::atomic::AtomicUsize>);

    #[async_trait]
    impl ReflectiveProposer for CountingProposer {
        async fn propose_json(
            &self,
            request: ReflectiveProposalRequest,
            _cancellation: CancellationToken,
        ) -> Result<Value> {
            self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(json!({
                "proposals": [{
                    "parent_candidate_id": request.parent_candidate_id,
                    "patches": [{
                        "kind": "replace_value",
                        "component_id": "instruction",
                        "value": { "kind": "text", "text": "do better" }
                    }]
                }]
            }))
        }
    }

    #[tokio::test]
    async fn mocked_harness_improves_from_reflective_gepa_proposal() {
        let temp =
            std::env::temp_dir().join(format!("lash-harness-opt-test-{}", uuid::Uuid::new_v4()));
        let project = Arc::new(ScoreProject);
        let runner = ProjectHarnessRunner::new(project.clone());
        let strategy = ReflectiveGepaStrategy::new(BetterProposer);
        let optimizer = HarnessOptimizer::new(runner, strategy);
        let run = OptimizationRun {
            run_id: "run".to_string(),
            experiment_id: "mock".to_string(),
            run_dir: temp,
            config: OptimizationConfig {
                max_metric_calls: 8,
                max_iterations: Some(1),
                minibatch_size: 1,
                max_concurrency: 1,
                per_example_timeout_secs: None,
                ..OptimizationConfig::default()
            },
        };

        let state = optimizer
            .run(
                run,
                project.seed_candidate().await.unwrap(),
                project.trainset().await.unwrap(),
                project.trainset().await.unwrap(),
                CancellationToken::new(),
            )
            .await
            .unwrap();

        assert!(state.best().unwrap().mean_score() > 0.5);
        assert_eq!(state.metric_calls_used, 2);
        assert!(state.cache_hits >= 1);
    }

    #[tokio::test]
    async fn reflective_gepa_rejects_non_improving_minibatch_proposal() {
        let temp =
            std::env::temp_dir().join(format!("lash-harness-opt-test-{}", uuid::Uuid::new_v4()));
        let project = Arc::new(ScoreProject);
        let optimizer = HarnessOptimizer::new(
            ProjectHarnessRunner::new(project.clone()),
            ReflectiveGepaStrategy::new(WorseProposer),
        );
        let run = OptimizationRun {
            run_id: "run".to_string(),
            experiment_id: "mock".to_string(),
            run_dir: temp,
            config: OptimizationConfig {
                max_metric_calls: 8,
                max_iterations: Some(1),
                minibatch_size: 1,
                max_concurrency: 1,
                per_example_timeout_secs: None,
                ..OptimizationConfig::default()
            },
        };

        let state = optimizer
            .run(
                run,
                project.seed_candidate().await.unwrap(),
                project.trainset().await.unwrap(),
                project.trainset().await.unwrap(),
                CancellationToken::new(),
            )
            .await
            .unwrap();

        assert_eq!(state.evaluated_candidates.len(), 1);
        assert_eq!(state.best().unwrap().candidate.id, "seed");
    }

    #[tokio::test]
    async fn skip_perfect_score_avoids_proposer_call() {
        let temp =
            std::env::temp_dir().join(format!("lash-harness-opt-test-{}", uuid::Uuid::new_v4()));
        let project = Arc::new(ScoreProject);
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let optimizer = HarnessOptimizer::new(
            ProjectHarnessRunner::new(project.clone()),
            ReflectiveGepaStrategy::new(CountingProposer(calls.clone())),
        );
        let run = OptimizationRun {
            run_id: "run".to_string(),
            experiment_id: "mock".to_string(),
            run_dir: temp,
            config: OptimizationConfig {
                max_metric_calls: 8,
                max_iterations: Some(1),
                minibatch_size: 1,
                max_concurrency: 1,
                perfect_score: 0.1,
                skip_perfect_score: true,
                per_example_timeout_secs: None,
                ..OptimizationConfig::default()
            },
        };

        let state = optimizer
            .run(
                run,
                project.seed_candidate().await.unwrap(),
                project.trainset().await.unwrap(),
                project.trainset().await.unwrap(),
                CancellationToken::new(),
            )
            .await
            .unwrap();

        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 0);
        assert_eq!(state.rejected_proposals, 1);
    }

    #[test]
    fn proposal_patch_scope_rejects_unselected_component() {
        let proposal = CandidateProposal {
            parent_candidate_id: "seed".to_string(),
            patches: vec![ComponentPatch::ReplaceValue {
                component_id: "instruction".to_string(),
                value: ComponentValue::Text {
                    text: "do better".to_string(),
                },
            }],
            rationale: None,
            metadata: BTreeMap::new(),
        };

        assert!(matches!(
            validate_patch_scope(&proposal, &["other".to_string()]),
            Err(HarnessOptError::InvalidProposal(_))
        ));
    }

    #[test]
    fn pareto_parent_selection_cycles_through_covered_examples() {
        let mut left = CandidateEvaluation {
            candidate: seed_candidate(),
            evaluations: BTreeMap::from([
                (
                    "ex1".to_string(),
                    EvaluationResult {
                        example_id: "ex1".to_string(),
                        split: Split::Val,
                        score: 1.0,
                        passed: None,
                        feedback: None,
                        metrics: BTreeMap::new(),
                        diagnostics: BTreeMap::new(),
                    },
                ),
                (
                    "ex2".to_string(),
                    EvaluationResult {
                        example_id: "ex2".to_string(),
                        split: Split::Val,
                        score: 0.0,
                        passed: None,
                        feedback: None,
                        metrics: BTreeMap::new(),
                        diagnostics: BTreeMap::new(),
                    },
                ),
            ]),
            traces: BTreeMap::new(),
            artifacts: BTreeMap::new(),
            metric_calls: BTreeMap::new(),
        };
        let mut right = left.clone();
        left.candidate.id = "left".to_string();
        right.candidate.id = "right".to_string();
        right.evaluations.get_mut("ex1").unwrap().score = 0.0;
        right.evaluations.get_mut("ex2").unwrap().score = 1.0;

        assert_eq!(
            select_pareto_parent(&[left.clone(), right.clone()], 0)
                .unwrap()
                .candidate
                .id,
            "left"
        );
        assert_eq!(
            select_pareto_parent(&[left, right], 1)
                .unwrap()
                .candidate
                .id,
            "right"
        );
    }

    #[test]
    fn clbench_seed_candidate_exposes_only_mutable_clbench_components() {
        let candidate = clbench::ClbenchProject::seed_candidate_static();
        let keys = candidate
            .mutable_components
            .keys()
            .cloned()
            .collect::<Vec<_>>();

        assert_eq!(
            keys,
            vec![
                clbench::MEMORY_GUIDANCE_COMPONENT,
                clbench::PROMPT_TEMPLATE_COMPONENT,
                clbench::USER_DIRECTIVE_COMPONENT
            ]
        );
        assert!(
            !candidate
                .mutable_components
                .contains_key("generic_rlm_execution_protocol")
        );
        assert!(
            !candidate
                .mutable_components
                .contains_key("lashlang_reference")
        );
    }
}
