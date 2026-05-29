//! Optimization engine: harness runner, optimizer loop, candidate proposal/patch application.

use crate::*;

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

pub(crate) async fn record_example_run<T>(
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

pub(crate) fn validate_patch_scope(
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
