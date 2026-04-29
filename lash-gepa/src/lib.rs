use std::collections::{BTreeMap, BTreeSet};

use lash_sansio::{PromptContribution, PromptTemplate};
use lash_trace::TraceRecord;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, thiserror::Error)]
pub enum GepaError {
    #[error("unknown mutable component `{0}`")]
    UnknownComponent(String),
    #[error("component `{component_id}` violates constraint: {reason}")]
    ConstraintViolation {
        component_id: String,
        reason: String,
    },
    #[error("runner error: {0}")]
    Runner(String),
    #[error("proposer error: {0}")]
    Proposer(String),
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
pub struct OptimizableProgram {
    pub id: String,
    #[serde(default)]
    pub immutable_context: Value,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub mutable_components: BTreeMap<String, MutableComponent>,
}

impl OptimizableProgram {
    pub fn with_component(mut self, component: MutableComponent) -> Self {
        self.mutable_components
            .insert(component.id.clone(), component);
        self
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Candidate {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_id: Option<String>,
    pub program: OptimizableProgram,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Example {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub split: Option<String>,
    pub input: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected: Option<Value>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EvaluationResult {
    pub example_id: String,
    pub score: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub passed: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub feedback: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metrics: BTreeMap<String, f64>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct TraceBundle {
    pub example_id: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub records: Vec<TraceRecord>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExampleRun {
    pub example: Example,
    pub result: EvaluationResult,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trace: Option<TraceBundle>,
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
pub struct Mutation {
    pub patch: ComponentPatch,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rationale: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CandidateProposal {
    pub parent_candidate_id: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub mutations: Vec<Mutation>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProposalRequest {
    pub generation: usize,
    pub parent: Candidate,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evaluated_examples: Vec<ExampleRun>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub frontier: Vec<Candidate>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, Value>,
}

pub trait CandidateRunner {
    fn evaluate(
        &mut self,
        candidate: &Candidate,
        examples: &[Example],
    ) -> Result<Vec<ExampleRun>, GepaError>;
}

pub trait Proposer {
    fn propose(&mut self, request: ProposalRequest) -> Result<Vec<CandidateProposal>, GepaError>;
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GepaConfig {
    pub max_generations: usize,
    pub minibatch_size: usize,
    pub proposals_per_generation: usize,
}

impl Default for GepaConfig {
    fn default() -> Self {
        Self {
            max_generations: 8,
            minibatch_size: 8,
            proposals_per_generation: 4,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CandidateEvaluation {
    pub candidate: Candidate,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub evaluations: BTreeMap<String, EvaluationResult>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub traces: BTreeMap<String, TraceBundle>,
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
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GenerationReport {
    pub generation: usize,
    pub parent_candidate_id: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub proposed_candidate_ids: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OptimizationReport {
    pub best: CandidateEvaluation,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub frontier: Vec<CandidateEvaluation>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub generations: Vec<GenerationReport>,
}

pub struct GepaOptimizer<R, P> {
    config: GepaConfig,
    runner: R,
    proposer: P,
}

impl<R, P> GepaOptimizer<R, P>
where
    R: CandidateRunner,
    P: Proposer,
{
    pub fn new(config: GepaConfig, runner: R, proposer: P) -> Self {
        Self {
            config,
            runner,
            proposer,
        }
    }

    pub fn run(
        &mut self,
        seed: Candidate,
        examples: Vec<Example>,
    ) -> Result<OptimizationReport, GepaError> {
        let batch = self.select_batch(&examples, 0);
        let seed_state = self.evaluate_candidate(seed, &batch)?;
        let mut states = vec![seed_state];
        let mut generations = Vec::new();

        for generation in 0..self.config.max_generations {
            let parent = best_state(&states)
                .expect("optimizer keeps at least one evaluated candidate")
                .candidate
                .clone();
            let parent_state = states
                .iter()
                .find(|state| state.candidate.id == parent.id)
                .expect("parent came from state set");
            let batch = self.select_batch(&examples, generation + 1);
            let request = ProposalRequest {
                generation,
                parent: parent.clone(),
                evaluated_examples: parent_state.example_runs_for(&batch),
                frontier: frontier(&states)
                    .into_iter()
                    .map(|state| state.candidate.clone())
                    .collect(),
                metadata: BTreeMap::new(),
            };
            let proposals = self.proposer.propose(request)?;
            let mut proposed_candidate_ids = Vec::new();

            for proposal in proposals
                .into_iter()
                .take(self.config.proposals_per_generation)
            {
                let candidate = apply_proposal(&parent, generation, &proposal)?;
                proposed_candidate_ids.push(candidate.id.clone());
                states.push(self.evaluate_candidate(candidate, &batch)?);
            }

            generations.push(GenerationReport {
                generation,
                parent_candidate_id: parent.id,
                proposed_candidate_ids,
            });
        }

        let mut final_frontier = frontier(&states);
        final_frontier.sort_by(|left, right| {
            right
                .mean_score()
                .partial_cmp(&left.mean_score())
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.candidate.id.cmp(&right.candidate.id))
        });
        let best = final_frontier
            .first()
            .cloned()
            .or_else(|| best_state(&states).cloned())
            .expect("optimizer keeps at least one evaluated candidate");

        Ok(OptimizationReport {
            best,
            frontier: final_frontier,
            generations,
        })
    }

    fn select_batch(&self, examples: &[Example], generation: usize) -> Vec<Example> {
        let limit = self.config.minibatch_size.max(1).min(examples.len());
        if examples.is_empty() {
            return Vec::new();
        }
        (0..limit)
            .map(|offset| examples[(generation + offset) % examples.len()].clone())
            .collect()
    }

    fn evaluate_candidate(
        &mut self,
        candidate: Candidate,
        examples: &[Example],
    ) -> Result<CandidateEvaluation, GepaError> {
        let runs = self.runner.evaluate(&candidate, examples)?;
        let mut evaluations = BTreeMap::new();
        let mut traces = BTreeMap::new();
        for run in runs {
            let example_id = run.result.example_id.clone();
            evaluations.insert(example_id.clone(), run.result);
            if let Some(trace) = run.trace {
                traces.insert(example_id, trace);
            }
        }
        Ok(CandidateEvaluation {
            candidate,
            evaluations,
            traces,
        })
    }
}

impl CandidateEvaluation {
    fn example_runs_for(&self, examples: &[Example]) -> Vec<ExampleRun> {
        examples
            .iter()
            .filter_map(|example| {
                let result = self.evaluations.get(&example.id)?;
                Some(ExampleRun {
                    example: example.clone(),
                    result: result.clone(),
                    trace: self.traces.get(&example.id).cloned(),
                })
            })
            .collect()
    }
}

pub fn apply_proposal(
    parent: &Candidate,
    generation: usize,
    proposal: &CandidateProposal,
) -> Result<Candidate, GepaError> {
    let mut candidate = parent.clone();
    candidate.id = format!("{}-g{}-{}", parent.id, generation, uuid::Uuid::new_v4());
    candidate.parent_id = Some(parent.id.clone());
    candidate.metadata.extend(proposal.metadata.clone());

    for mutation in &proposal.mutations {
        apply_patch(&mut candidate.program, &mutation.patch)?;
    }

    Ok(candidate)
}

pub fn apply_patch(
    program: &mut OptimizableProgram,
    patch: &ComponentPatch,
) -> Result<(), GepaError> {
    match patch {
        ComponentPatch::ReplaceValue {
            component_id,
            value,
        } => {
            let component = program
                .mutable_components
                .get_mut(component_id)
                .ok_or_else(|| GepaError::UnknownComponent(component_id.clone()))?;
            validate_component_value(component_id, &component.constraints, value)?;
            component.value = value.clone();
        }
    }
    Ok(())
}

fn validate_component_value(
    component_id: &str,
    constraints: &ComponentConstraints,
    value: &ComponentValue,
) -> Result<(), GepaError> {
    let Some(text) = value.text_for_constraints() else {
        return Ok(());
    };
    if let Some(max_chars) = constraints.max_chars
        && text.chars().count() > max_chars
    {
        return Err(GepaError::ConstraintViolation {
            component_id: component_id.to_string(),
            reason: format!("text exceeds {max_chars} characters"),
        });
    }
    for term in &constraints.preserve_terms {
        if !text.contains(term) {
            return Err(GepaError::ConstraintViolation {
                component_id: component_id.to_string(),
                reason: format!("missing preserved term `{term}`"),
            });
        }
    }
    for term in &constraints.forbidden_terms {
        if text.contains(term) {
            return Err(GepaError::ConstraintViolation {
                component_id: component_id.to_string(),
                reason: format!("contains forbidden term `{term}`"),
            });
        }
    }
    Ok(())
}

fn best_state(states: &[CandidateEvaluation]) -> Option<&CandidateEvaluation> {
    states.iter().max_by(|left, right| {
        left.mean_score()
            .partial_cmp(&right.mean_score())
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| right.candidate.id.cmp(&left.candidate.id))
    })
}

pub fn frontier(states: &[CandidateEvaluation]) -> Vec<CandidateEvaluation> {
    states
        .iter()
        .filter(|candidate| {
            !states.iter().any(|other| {
                other.candidate.id != candidate.candidate.id && dominates(other, candidate)
            })
        })
        .cloned()
        .collect()
}

fn dominates(left: &CandidateEvaluation, right: &CandidateEvaluation) -> bool {
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

#[cfg(test)]
mod tests {
    use super::*;

    struct ScoreRunner;

    impl CandidateRunner for ScoreRunner {
        fn evaluate(
            &mut self,
            candidate: &Candidate,
            examples: &[Example],
        ) -> Result<Vec<ExampleRun>, GepaError> {
            let score = match candidate
                .program
                .mutable_components
                .get("instruction")
                .map(|component| &component.value)
            {
                Some(ComponentValue::Text { text }) if text.contains("better") => 1.0,
                _ => 0.2,
            };
            Ok(examples
                .iter()
                .map(|example| ExampleRun {
                    example: example.clone(),
                    result: EvaluationResult {
                        example_id: example.id.clone(),
                        score,
                        passed: Some(score > 0.5),
                        feedback: None,
                        metrics: BTreeMap::new(),
                    },
                    trace: None,
                })
                .collect())
        }
    }

    struct BetterProposer;

    impl Proposer for BetterProposer {
        fn propose(
            &mut self,
            request: ProposalRequest,
        ) -> Result<Vec<CandidateProposal>, GepaError> {
            Ok(vec![CandidateProposal {
                parent_candidate_id: request.parent.id,
                mutations: vec![Mutation {
                    patch: ComponentPatch::ReplaceValue {
                        component_id: "instruction".to_string(),
                        value: ComponentValue::Text {
                            text: "do better".to_string(),
                        },
                    },
                    rationale: Some("feedback requested better behavior".to_string()),
                }],
                metadata: BTreeMap::new(),
            }])
        }
    }

    fn seed_candidate() -> Candidate {
        Candidate {
            id: "seed".to_string(),
            parent_id: None,
            program: OptimizableProgram {
                id: "program".to_string(),
                immutable_context: Value::Null,
                mutable_components: BTreeMap::from([(
                    "instruction".to_string(),
                    MutableComponent {
                        id: "instruction".to_string(),
                        description: Some("system instruction".to_string()),
                        value: ComponentValue::Text {
                            text: "do ok".to_string(),
                        },
                        constraints: ComponentConstraints {
                            max_chars: Some(32),
                            ..Default::default()
                        },
                    },
                )]),
            },
            metadata: BTreeMap::new(),
        }
    }

    #[test]
    fn optimizer_uses_proposer_to_improve_candidate() {
        let examples = vec![Example {
            id: "ex1".to_string(),
            split: Some("train".to_string()),
            input: serde_json::json!({"question": "q"}),
            expected: None,
            metadata: BTreeMap::new(),
        }];
        let mut optimizer = GepaOptimizer::new(
            GepaConfig {
                max_generations: 1,
                minibatch_size: 1,
                proposals_per_generation: 1,
            },
            ScoreRunner,
            BetterProposer,
        );

        let report = optimizer.run(seed_candidate(), examples).unwrap();

        assert!(report.best.mean_score() > 0.5);
        assert_eq!(report.frontier.len(), 1);
    }

    #[test]
    fn patch_validation_rejects_invalid_mutation() {
        let mut candidate = seed_candidate();
        let err = apply_patch(
            &mut candidate.program,
            &ComponentPatch::ReplaceValue {
                component_id: "instruction".to_string(),
                value: ComponentValue::Text {
                    text: "this text is far too long for the configured limit".to_string(),
                },
            },
        )
        .unwrap_err();

        assert!(matches!(err, GepaError::ConstraintViolation { .. }));
    }

    #[test]
    fn frontier_keeps_non_dominated_candidates() {
        let mut weak = CandidateEvaluation {
            candidate: seed_candidate(),
            evaluations: BTreeMap::from([(
                "ex1".to_string(),
                EvaluationResult {
                    example_id: "ex1".to_string(),
                    score: 0.1,
                    passed: None,
                    feedback: None,
                    metrics: BTreeMap::new(),
                },
            )]),
            traces: BTreeMap::new(),
        };
        let mut strong = weak.clone();
        strong.candidate.id = "strong".to_string();
        strong.evaluations.get_mut("ex1").unwrap().score = 0.9;
        weak.candidate.id = "weak".to_string();

        let front = frontier(&[weak, strong]);

        assert_eq!(front.len(), 1);
        assert_eq!(front[0].candidate.id, "strong");
    }
}
