//! Core data model: candidates, evaluations, config, records, and store/runner traits.

use crate::*;

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
    pub(crate) fn text_for_constraints(&self) -> Option<&str> {
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

    pub(crate) fn example_runs_for(&self, examples: &[HarnessExample]) -> Vec<ExampleRun> {
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
