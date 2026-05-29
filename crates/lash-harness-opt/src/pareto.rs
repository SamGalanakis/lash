//! Pure pareto-frontier math: dominance, frontier selection, candidate merging.

use crate::*;

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

pub(crate) fn best_state(states: &[CandidateEvaluation]) -> Option<&CandidateEvaluation> {
    states.iter().max_by(|left, right| {
        left.mean_score()
            .partial_cmp(&right.mean_score())
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| right.candidate.id.cmp(&left.candidate.id))
    })
}

pub(crate) fn select_pareto_parent(
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

pub(crate) fn complementary_pair(
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

pub(crate) fn merge_candidates(
    left: &Candidate,
    right: &Candidate,
    generation: usize,
) -> Result<Candidate> {
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
