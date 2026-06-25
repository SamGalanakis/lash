use std::collections::{BTreeMap, BTreeSet};
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use lash_sansio::{
    PromptBuiltin, PromptContribution, PromptSlot, PromptTemplate, PromptTemplateEntry,
    PromptTemplateSection,
};
use lash_trace::{TraceContext, TraceEvent, TraceRecord};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::sync::Semaphore;
use tokio_util::sync::CancellationToken;

mod engine;
mod model;
mod pareto;

pub mod strategies;
pub mod toybench;

mod sqlite_store;
pub use sqlite_store::SqliteHarnessStore;

pub use engine::*;
pub use model::*;
pub use pareto::*;

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
    fn toybench_seed_candidate_exposes_only_mutable_toybench_components() {
        let candidate = toybench::ToybenchProject::seed_candidate_static();
        let keys = candidate
            .mutable_components
            .keys()
            .cloned()
            .collect::<Vec<_>>();

        assert_eq!(
            keys,
            vec![
                toybench::MEMORY_GUIDANCE_COMPONENT,
                toybench::PROMPT_TEMPLATE_COMPONENT,
                toybench::USER_DIRECTIVE_COMPONENT
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
