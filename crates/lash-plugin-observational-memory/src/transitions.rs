use lash_core::SessionGraph;
use lash_core::plugin::PluginError;

use crate::ObservationalMemoryConfig;
use crate::constants::{
    ACTIVE_STATE_PLUGIN_TYPE, BUFFERED_OBSERVATION_PLUGIN_TYPE, BUFFERED_REFLECTION_PLUGIN_TYPE,
};
use crate::graph_state::{
    active_unobserved_message_nodes, approx_message_nodes_tokens, approx_token_count,
    build_graph_state, prefix_len_covering_tokens, prefix_len_leaving_tail_budget,
    retained_message_tokens_by_message_id, split_message_batches,
};
use crate::host::OmRuntimeHost;
use crate::model::{
    ActiveMemoryNode, ActiveMemoryState, BufferedObservationNode, BufferedReflectionNode,
    ParsedMemoryOutput,
};
use crate::worker::{run_observer_batch, run_reflector};

pub(crate) fn should_run_async_maintenance(
    config: &ObservationalMemoryConfig,
    graph: &SessionGraph,
) -> bool {
    let om_state = build_graph_state(graph);
    let start_after = om_state
        .buffered_observations
        .last()
        .map(|chunk| chunk.observed_through_message_id.as_str())
        .or_else(|| {
            om_state
                .active
                .as_ref()
                .and_then(|state| state.observed_through_message_id.as_deref())
        });
    let observation_interval = config.observation_buffer_interval_tokens();
    if observation_interval > 0
        && approx_message_nodes_tokens(&active_unobserved_message_nodes(graph, start_after))
            >= observation_interval
    {
        return true;
    }

    om_state.buffered_reflection.is_none()
        && om_state
            .active
            .as_ref()
            .map(|active| approx_token_count(&active.observations))
            .unwrap_or(0)
            >= config.reflection_buffer_activation_tokens()
}

pub(crate) async fn maybe_advance_memory_state(
    config: &ObservationalMemoryConfig,
    om_host: &OmRuntimeHost<'_>,
    policy: &lash_core::SessionPolicy,
    mut graph: SessionGraph,
) -> Result<SessionGraph, PluginError> {
    for _ in 0..6 {
        let om_state = build_graph_state(&graph);
        let pending_messages = active_unobserved_message_nodes(
            &graph,
            om_state
                .active
                .as_ref()
                .and_then(|state| state.observed_through_message_id.as_deref()),
        );
        let pending_tokens = approx_message_nodes_tokens(&pending_messages);
        if pending_tokens >= config.observation_message_tokens {
            if let Some(next) = activate_buffered_observations(config, om_host, &graph).await? {
                graph = next;
                continue;
            }
            if pending_tokens >= config.observation_block_after_tokens
                && let Some(next) =
                    sync_observe_pending_messages(config, om_host, policy, &graph).await?
            {
                graph = next;
                continue;
            }
        }

        let om_state = build_graph_state(&graph);
        let active_tokens = om_state
            .active
            .as_ref()
            .map(|state| approx_token_count(&state.observations))
            .unwrap_or(0);
        if active_tokens >= config.reflection_observation_tokens {
            if let Some(next) = activate_buffered_reflection(om_host, &graph).await? {
                graph = next;
                continue;
            }
            if active_tokens >= config.reflection_block_after_tokens
                && let Some(next) = sync_reflect_active_memory(om_host, policy, &graph).await?
            {
                graph = next;
                continue;
            }
        }

        break;
    }

    Ok(graph)
}

pub(crate) async fn maybe_buffer_observations(
    config: &ObservationalMemoryConfig,
    om_host: &OmRuntimeHost<'_>,
    policy: &lash_core::SessionPolicy,
    graph: &SessionGraph,
) -> Result<(), PluginError> {
    let om_state = build_graph_state(graph);
    let start_after = om_state
        .buffered_observations
        .last()
        .map(|chunk| chunk.observed_through_message_id.as_str())
        .or_else(|| {
            om_state
                .active
                .as_ref()
                .and_then(|state| state.observed_through_message_id.as_deref())
        });
    let unbuffered = active_unobserved_message_nodes(graph, start_after);
    let interval_tokens = config.observation_buffer_interval_tokens();
    if interval_tokens == 0 {
        return Ok(());
    }
    let total_tokens = approx_message_nodes_tokens(&unbuffered);
    if total_tokens < interval_tokens {
        return Ok(());
    }

    let target_tokens = total_tokens - (total_tokens % interval_tokens);
    let Some(prefix_len) = prefix_len_covering_tokens(&unbuffered, target_tokens) else {
        return Ok(());
    };
    if prefix_len == 0 {
        return Ok(());
    }
    let batch_target = config
        .observation_max_tokens_per_batch
        .min(interval_tokens)
        .max(1);
    let batches = split_message_batches(&unbuffered[..prefix_len], batch_target);
    if batches.is_empty() {
        return Ok(());
    }

    let mut preview = om_state.active.clone();
    let mut nodes = Vec::new();
    for batch in batches {
        let output =
            run_observer_batch(config, om_host, policy.clone(), preview.as_ref(), &batch).await?;
        if output.observations.trim().is_empty()
            && output.current_task.is_none()
            && output.suggested_response.is_none()
        {
            continue;
        }
        let Some(last_message) = batch.last() else {
            continue;
        };
        let node = BufferedObservationNode {
            observed_through_message_id: last_message.message.id.clone(),
            observations: output.observations.trim().to_string(),
            current_task: output.current_task.clone(),
            suggested_response: output.suggested_response.clone(),
            observation_tokens: approx_token_count(output.observations.trim()),
        };
        preview = Some(merge_active_state_with_observation(preview.as_ref(), &node));
        nodes.push((
            BUFFERED_OBSERVATION_PLUGIN_TYPE.to_string(),
            serde_json::to_value(node).map_err(|err| {
                PluginError::Snapshot(format!("failed to encode OM buffered observation: {err}"))
            })?,
        ));
    }

    if nodes.is_empty() {
        return Ok(());
    }

    let _ = om_host.append_plugin_nodes(graph, nodes).await?;
    Ok(())
}

pub(crate) async fn maybe_buffer_reflection(
    config: &ObservationalMemoryConfig,
    om_host: &OmRuntimeHost<'_>,
    policy: &lash_core::SessionPolicy,
    graph: &SessionGraph,
) -> Result<(), PluginError> {
    let om_state = build_graph_state(graph);
    let Some(active) = om_state.active else {
        return Ok(());
    };
    if om_state.buffered_reflection.is_some() {
        return Ok(());
    }

    let observation_tokens = approx_token_count(&active.observations);
    if observation_tokens < config.reflection_buffer_activation_tokens() {
        return Ok(());
    }

    let output = run_reflector(om_host, policy.clone(), &active.observations).await?;
    if output.observations.trim().is_empty() {
        return Ok(());
    }
    if output.observations.trim() == active.observations.trim()
        && output.current_task == active.current_task
        && output.suggested_response == active.suggested_response
    {
        return Ok(());
    }

    let node = BufferedReflectionNode {
        source_state_node_id: active.state_node_id,
        observed_through_message_id: active.observed_through_message_id.unwrap_or_default(),
        observations: output.observations.trim().to_string(),
        current_task: output.current_task,
        suggested_response: output.suggested_response,
        observation_tokens,
    };
    let _ = om_host
        .append_plugin_nodes(
            graph,
            vec![(
                BUFFERED_REFLECTION_PLUGIN_TYPE.to_string(),
                serde_json::to_value(node).map_err(|err| {
                    PluginError::Snapshot(format!("failed to encode OM buffered reflection: {err}"))
                })?,
            )],
        )
        .await?;
    Ok(())
}

async fn activate_buffered_observations(
    config: &ObservationalMemoryConfig,
    om_host: &OmRuntimeHost<'_>,
    graph: &SessionGraph,
) -> Result<Option<SessionGraph>, PluginError> {
    let om_state = build_graph_state(graph);
    if om_state.buffered_observations.is_empty() {
        return Ok(None);
    }
    let pending_messages = active_unobserved_message_nodes(
        graph,
        om_state
            .active
            .as_ref()
            .and_then(|state| state.observed_through_message_id.as_deref()),
    );
    if pending_messages.is_empty() {
        return Ok(None);
    }

    let retained_after = retained_message_tokens_by_message_id(&pending_messages);
    let mut activated = Vec::new();
    let mut merged = om_state.active.clone();
    for chunk in &om_state.buffered_observations {
        activated.push(chunk.clone());
        merged = Some(merge_active_state_with_observation(
            merged.as_ref(),
            &BufferedObservationNode {
                observed_through_message_id: chunk.observed_through_message_id.clone(),
                observations: chunk.observations.clone(),
                current_task: chunk.current_task.clone(),
                suggested_response: chunk.suggested_response.clone(),
                observation_tokens: approx_token_count(&chunk.observations),
            },
        ));
        let remaining = retained_after
            .get(chunk.observed_through_message_id.as_str())
            .copied()
            .unwrap_or(0);
        if remaining <= config.observation_retention_tokens() {
            break;
        }
    }

    if activated.is_empty() {
        return Ok(None);
    }

    let Some(next_active) = merged else {
        return Ok(None);
    };
    let node = ActiveMemoryNode {
        observed_through_message_id: next_active.observed_through_message_id.unwrap_or_default(),
        observations: next_active.observations,
        current_task: next_active.current_task,
        suggested_response: next_active.suggested_response,
    };
    match om_host
        .append_plugin_nodes(
            graph,
            vec![(
                ACTIVE_STATE_PLUGIN_TYPE.to_string(),
                serde_json::to_value(node).map_err(|err| {
                    PluginError::Snapshot(format!("failed to encode OM state activation: {err}"))
                })?,
            )],
        )
        .await?
    {
        Some(next) => Ok(Some(next)),
        None => Ok(None),
    }
}

async fn activate_buffered_reflection(
    om_host: &OmRuntimeHost<'_>,
    graph: &SessionGraph,
) -> Result<Option<SessionGraph>, PluginError> {
    let om_state = build_graph_state(graph);
    let Some(active) = om_state.active else {
        return Ok(None);
    };
    let Some(buffered) = om_state.buffered_reflection else {
        return Ok(None);
    };
    if buffered.source_state_node_id != active.state_node_id {
        return Ok(None);
    }
    let node = ActiveMemoryNode {
        observed_through_message_id: buffered.observed_through_message_id,
        observations: buffered.observations,
        current_task: buffered.current_task,
        suggested_response: buffered.suggested_response,
    };
    om_host
        .append_plugin_nodes(
            graph,
            vec![(
                ACTIVE_STATE_PLUGIN_TYPE.to_string(),
                serde_json::to_value(node).map_err(|err| {
                    PluginError::Snapshot(format!(
                        "failed to encode OM reflection activation: {err}"
                    ))
                })?,
            )],
        )
        .await
}

async fn sync_observe_pending_messages(
    config: &ObservationalMemoryConfig,
    om_host: &OmRuntimeHost<'_>,
    policy: &lash_core::SessionPolicy,
    graph: &SessionGraph,
) -> Result<Option<SessionGraph>, PluginError> {
    let om_state = build_graph_state(graph);
    let unobserved = active_unobserved_message_nodes(
        graph,
        om_state
            .active
            .as_ref()
            .and_then(|state| state.observed_through_message_id.as_deref()),
    );
    if unobserved.is_empty() {
        return Ok(None);
    }
    let observe_until =
        prefix_len_leaving_tail_budget(&unobserved, config.observation_retention_tokens());
    if observe_until == 0 {
        return Ok(None);
    }
    let batches = split_message_batches(
        &unobserved[..observe_until],
        config.observation_max_tokens_per_batch.max(1),
    );
    if batches.is_empty() {
        return Ok(None);
    }

    let mut merged = om_state.active.clone();
    for batch in batches {
        let output =
            run_observer_batch(config, om_host, policy.clone(), merged.as_ref(), &batch).await?;
        if output.observations.trim().is_empty()
            && output.current_task.is_none()
            && output.suggested_response.is_none()
        {
            continue;
        }
        let Some(last_message) = batch.last() else {
            continue;
        };
        merged = Some(merge_active_state_with_observer_output(
            merged.as_ref(),
            last_message.message.id.clone(),
            output,
        ));
    }

    let Some(active) = merged else {
        return Ok(None);
    };
    let node = ActiveMemoryNode {
        observed_through_message_id: active.observed_through_message_id.unwrap_or_default(),
        observations: active.observations,
        current_task: active.current_task,
        suggested_response: active.suggested_response,
    };
    om_host
        .append_plugin_nodes(
            graph,
            vec![(
                ACTIVE_STATE_PLUGIN_TYPE.to_string(),
                serde_json::to_value(node).map_err(|err| {
                    PluginError::Snapshot(format!("failed to encode OM sync observation: {err}"))
                })?,
            )],
        )
        .await
}

async fn sync_reflect_active_memory(
    om_host: &OmRuntimeHost<'_>,
    policy: &lash_core::SessionPolicy,
    graph: &SessionGraph,
) -> Result<Option<SessionGraph>, PluginError> {
    let om_state = build_graph_state(graph);
    let Some(active) = om_state.active else {
        return Ok(None);
    };
    let output = run_reflector(om_host, policy.clone(), &active.observations).await?;
    if output.observations.trim().is_empty() {
        return Ok(None);
    }
    if output.observations.trim() == active.observations.trim()
        && output.current_task == active.current_task
        && output.suggested_response == active.suggested_response
    {
        return Ok(None);
    }
    let node = ActiveMemoryNode {
        observed_through_message_id: active.observed_through_message_id.unwrap_or_default(),
        observations: output.observations.trim().to_string(),
        current_task: output.current_task,
        suggested_response: output.suggested_response,
    };
    om_host
        .append_plugin_nodes(
            graph,
            vec![(
                ACTIVE_STATE_PLUGIN_TYPE.to_string(),
                serde_json::to_value(node).map_err(|err| {
                    PluginError::Snapshot(format!("failed to encode OM sync reflection: {err}"))
                })?,
            )],
        )
        .await
}

fn merge_active_state_with_observation(
    previous: Option<&ActiveMemoryState>,
    batch: &BufferedObservationNode,
) -> ActiveMemoryState {
    let mut observations = previous
        .map(|state| state.observations.clone())
        .unwrap_or_default();
    if !observations.trim().is_empty() && !batch.observations.trim().is_empty() {
        observations.push_str("\n\n");
    }
    observations.push_str(batch.observations.trim());
    ActiveMemoryState {
        state_node_id: previous
            .map(|state| state.state_node_id.clone())
            .unwrap_or_default(),
        observations,
        current_task: batch
            .current_task
            .clone()
            .or_else(|| previous.and_then(|state| state.current_task.clone())),
        suggested_response: batch
            .suggested_response
            .clone()
            .or_else(|| previous.and_then(|state| state.suggested_response.clone())),
        observed_through_message_id: Some(batch.observed_through_message_id.clone()),
    }
}

fn merge_active_state_with_observer_output(
    previous: Option<&ActiveMemoryState>,
    observed_through_message_id: String,
    output: ParsedMemoryOutput,
) -> ActiveMemoryState {
    let mut observations = previous
        .map(|state| state.observations.clone())
        .unwrap_or_default();
    if !observations.trim().is_empty() && !output.observations.trim().is_empty() {
        observations.push_str("\n\n");
    }
    observations.push_str(output.observations.trim());
    ActiveMemoryState {
        state_node_id: previous
            .map(|state| state.state_node_id.clone())
            .unwrap_or_default(),
        observations,
        current_task: output
            .current_task
            .or_else(|| previous.and_then(|state| state.current_task.clone())),
        suggested_response: output
            .suggested_response
            .or_else(|| previous.and_then(|state| state.suggested_response.clone())),
        observed_through_message_id: Some(observed_through_message_id),
    }
}
