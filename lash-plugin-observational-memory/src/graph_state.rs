use std::collections::HashMap;

use lash_core::{MessageRole, SessionGraph};

use crate::constants::{
    ACTIVE_STATE_PLUGIN_TYPE, BUFFERED_OBSERVATION_PLUGIN_TYPE, BUFFERED_REFLECTION_PLUGIN_TYPE,
};
use crate::model::{
    ActiveMemoryNode, ActiveMemoryState, BufferedObservationNode, BufferedObservationState,
    BufferedReflectionNode, BufferedReflectionState, MessageNode, ObservedMessageNode,
    OmGraphState,
};
use crate::prompts::format_message_for_observer;

pub(crate) fn build_graph_state(graph: &SessionGraph) -> OmGraphState {
    let mut state = OmGraphState::default();
    for node in graph.active_path_nodes() {
        let Some((kind, _)) = node.plugin() else {
            continue;
        };
        match kind {
            ACTIVE_STATE_PLUGIN_TYPE => {
                let Some(active) = node.plugin_body::<ActiveMemoryNode>() else {
                    continue;
                };
                state.active = Some(ActiveMemoryState {
                    state_node_id: node.node_id.clone(),
                    observed_through_message_id: Some(active.observed_through_message_id),
                    observations: active.observations,
                    current_task: active.current_task,
                    suggested_response: active.suggested_response,
                });
                state.buffered_observations.clear();
                state.buffered_reflection = None;
            }
            BUFFERED_OBSERVATION_PLUGIN_TYPE => {
                let Some(buffered) = node.plugin_body::<BufferedObservationNode>() else {
                    continue;
                };
                if state.buffered_observations.iter().any(|chunk| {
                    chunk.observed_through_message_id == buffered.observed_through_message_id
                }) {
                    continue;
                }
                state.buffered_observations.push(BufferedObservationState {
                    observed_through_message_id: buffered.observed_through_message_id,
                    observations: buffered.observations,
                    current_task: buffered.current_task,
                    suggested_response: buffered.suggested_response,
                });
            }
            BUFFERED_REFLECTION_PLUGIN_TYPE => {
                let Some(buffered) = node.plugin_body::<BufferedReflectionNode>() else {
                    continue;
                };
                let Some(active) = state.active.as_ref() else {
                    continue;
                };
                if buffered.source_state_node_id != active.state_node_id {
                    continue;
                }
                state.buffered_reflection = Some(BufferedReflectionState {
                    source_state_node_id: buffered.source_state_node_id,
                    observed_through_message_id: buffered.observed_through_message_id,
                    observations: buffered.observations,
                    current_task: buffered.current_task,
                    suggested_response: buffered.suggested_response,
                });
            }
            _ => {}
        }
    }
    state
}

pub(crate) fn active_unobserved_message_nodes(
    graph: &SessionGraph,
    observed_through_message_id: Option<&str>,
) -> Vec<MessageNode> {
    let mut seen_observed = observed_through_message_id.is_none();
    graph
        .active_path_nodes()
        .into_iter()
        .filter_map(|node| {
            let message = node.message()?;
            if matches!(message.role, MessageRole::System) {
                return None;
            }
            if !seen_observed {
                if observed_through_message_id == Some(message.id.as_str()) {
                    seen_observed = true;
                }
                return None;
            }
            Some(MessageNode {
                timestamp: node.timestamp.clone(),
                message,
            })
        })
        .collect()
}

pub(crate) fn retained_message_tokens_by_message_id<N: ObservedMessageNode>(
    messages: &[N],
) -> HashMap<&str, usize> {
    let mut retained = HashMap::new();
    let mut suffix_tokens = 0usize;
    for message in messages.iter().rev() {
        retained.insert(message.message().id.as_str(), suffix_tokens);
        suffix_tokens = suffix_tokens.saturating_add(approx_message_tokens(message));
    }
    retained
}

pub(crate) fn prefix_len_leaving_tail_budget<N: ObservedMessageNode>(
    messages: &[N],
    tail_budget_tokens: usize,
) -> usize {
    if messages.is_empty() {
        return 0;
    }
    if tail_budget_tokens == 0 {
        return messages.len();
    }
    let mut suffix_tokens = 0usize;
    for (idx, message) in messages.iter().enumerate().rev() {
        suffix_tokens = suffix_tokens.saturating_add(approx_message_tokens(message));
        if suffix_tokens > tail_budget_tokens {
            return idx + 1;
        }
    }
    0
}

pub(crate) fn prefix_len_covering_tokens<N: ObservedMessageNode>(
    messages: &[N],
    target_tokens: usize,
) -> Option<usize> {
    if target_tokens == 0 {
        return Some(0);
    }
    let mut total = 0usize;
    for (idx, message) in messages.iter().enumerate() {
        total = total.saturating_add(approx_message_tokens(message));
        if total >= target_tokens {
            return Some(idx + 1);
        }
    }
    None
}

pub(crate) fn split_message_batches<N: ObservedMessageNode + Clone>(
    messages: &[N],
    max_tokens_per_batch: usize,
) -> Vec<Vec<N>> {
    let mut batches = Vec::new();
    let mut current = Vec::new();
    let mut current_tokens = 0usize;

    for message in messages {
        let tokens = approx_message_tokens(message).max(1);
        if !current.is_empty() && current_tokens + tokens > max_tokens_per_batch {
            batches.push(current);
            current = Vec::new();
            current_tokens = 0;
        }
        current.push(message.clone());
        current_tokens += tokens;
    }

    if !current.is_empty() {
        batches.push(current);
    }
    batches
}

pub(crate) fn approx_message_nodes_tokens<N: ObservedMessageNode>(messages: &[N]) -> usize {
    messages.iter().map(approx_message_tokens).sum()
}

pub(crate) fn approx_message_tokens<N: ObservedMessageNode>(message: &N) -> usize {
    approx_token_count(&format_message_for_observer(message))
}

pub(crate) fn approx_token_count(text: &str) -> usize {
    text.chars().count().div_ceil(4)
}
