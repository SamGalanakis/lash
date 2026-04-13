use lash::Store;
#[cfg(test)]
use lash::{DynamicStateSnapshot, SessionStateEnvelope};

use crate::app::UiResumeState;
use crate::ui_resume;

#[derive(Clone, Debug)]
pub(crate) struct LoadedLiveResumeSnapshot {
    pub(crate) snapshot: lash::LiveResumeSnapshot,
    pub(crate) graph: lash::SessionGraph,
    pub(crate) delta: Option<lash::LiveResumeDelta>,
    pub(crate) ui_state: UiResumeState,
}

#[cfg(test)]
pub(crate) fn save_live_resume_snapshot(
    store: &Store,
    state: &SessionStateEnvelope,
    ui_state: &UiResumeState,
    dynamic_state: &DynamicStateSnapshot,
) -> Result<(), String> {
    let projected_messages = state.project_messages();
    if !lash::messages_are_live_resume_safe(&projected_messages) {
        return Ok(());
    }
    ui_resume::save_ui_resume_state(store, ui_state);
    let base_graph = store
        .load_live_resume()
        .map(|snapshot| snapshot.graph)
        .or_else(|| store.load_session_head().map(|head| head.graph))
        .unwrap_or_else(|| state.session_graph.clone());
    let projected_tool_calls = state.project_tool_calls();
    let mut graph = base_graph.clone();
    graph.merge_active_projection(&projected_messages, &projected_tool_calls);
    let checkpoint = lash::HydratedSessionCheckpoint {
        turn_state: lash::PersistedTurnState {
            iteration: state.iteration,
            token_usage: state.token_usage.clone(),
            last_prompt_usage: state.last_prompt_usage.clone(),
        },
        dynamic_state_ref: None,
        dynamic_state: Some(dynamic_state.clone()),
        plugin_snapshot_ref: None,
        plugin_snapshot_revision: None,
        plugin_snapshot: state.plugin_snapshot.clone(),
    };
    let checkpoint_ref = store.put_checkpoint(&checkpoint).checkpoint_ref;
    let delta_ref = store.put_typed_blob(&lash::LiveResumeDelta {
        appended_graph_nodes: graph.nodes[base_graph.nodes.len()..].to_vec(),
        leaf_node_id: graph.leaf_node_id.clone(),
        turn_state: checkpoint.turn_state.clone(),
        dynamic_state: checkpoint.dynamic_state.clone(),
        plugin_snapshot: checkpoint.plugin_snapshot.clone(),
        execution_state_snapshot: state.execution_state_snapshot.clone(),
        token_ledger: state.token_ledger.clone(),
    });
    store.save_live_resume(lash::LiveResumeSnapshot {
        graph: base_graph,
        config: lash::PersistedSessionConfig {
            provider_id: state.policy.provider.id().to_string(),
            configured_model: state.policy.model.clone(),
            context_window: state.policy.max_context_tokens.unwrap_or_default() as u64,
            execution_mode: state.policy.execution_mode,
            context_approach: state.policy.context_approach.clone(),
            model_variant: state.policy.model_variant.clone(),
        },
        checkpoint_ref: Some(checkpoint_ref),
        delta_ref: Some(delta_ref),
    });
    Ok(())
}

pub(crate) fn load_live_resume_snapshot(store: &Store) -> Option<LoadedLiveResumeSnapshot> {
    let snapshot = store.load_live_resume()?;
    let delta = snapshot
        .delta_ref
        .as_ref()
        .and_then(|blob_ref| store.get_typed_blob::<lash::LiveResumeDelta>(blob_ref));
    let graph = lash::materialize_live_resume_graph(&snapshot, delta.as_ref());
    let safe_messages = lash::messages_are_live_resume_safe(&graph.project_messages());
    if !safe_messages {
        store.clear_live_resume();
        return None;
    }
    Some(LoadedLiveResumeSnapshot {
        snapshot,
        graph,
        delta,
        ui_state: ui_resume::load_ui_resume_state(store),
    })
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use super::*;
    use lash::session_model::{Message, MessageRole, Part, PartKind, PruneState};

    fn text_message(id: &str, content: &str) -> Message {
        Message {
            id: id.to_string(),
            role: MessageRole::User,
            parts: vec![Part {
                id: format!("{id}.p0"),
                kind: PartKind::Text,
                content: content.to_string(),
                attachment: None,
                tool_call_id: None,
                tool_name: None,
                prune_state: PruneState::Intact,
            }],
            user_input: None,
            origin: None,
        }
    }

    fn unsafe_tool_call_message(id: &str, call_id: &str) -> Message {
        Message {
            id: id.to_string(),
            role: MessageRole::Assistant,
            parts: vec![Part {
                id: format!("{id}.p0"),
                kind: PartKind::ToolCall,
                content: "{}".to_string(),
                attachment: None,
                tool_call_id: Some(call_id.to_string()),
                tool_name: Some("read_file".to_string()),
                prune_state: PruneState::Intact,
            }],
            user_input: None,
            origin: None,
        }
    }

    fn snapshot_state(messages: Vec<Message>) -> SessionStateEnvelope {
        let mut state = SessionStateEnvelope {
            session_id: "root".to_string(),
            policy: lash::SessionPolicy::default(),
            session_graph: lash::SessionGraph::from_projection(&messages, &[]),
            ..SessionStateEnvelope::default()
        };
        state.replace_projection(&messages, &[]);
        state
    }

    fn empty_dynamic_state() -> DynamicStateSnapshot {
        DynamicStateSnapshot {
            base_generation: 0,
            tools: BTreeMap::new(),
            enabled_tools: BTreeSet::new(),
        }
    }

    #[test]
    fn unsafe_live_snapshot_does_not_replace_last_safe_snapshot() {
        let store = Store::memory().expect("store");
        let dynamic_state = empty_dynamic_state();
        let ui_state = UiResumeState::default();
        let safe_state = snapshot_state(vec![text_message("m0", "hello")]);
        save_live_resume_snapshot(&store, &safe_state, &ui_state, &dynamic_state)
            .expect("save safe snapshot");

        let unsafe_state = snapshot_state(vec![unsafe_tool_call_message("m1", "call_1")]);
        save_live_resume_snapshot(&store, &unsafe_state, &ui_state, &dynamic_state)
            .expect("skip unsafe snapshot");

        let loaded = load_live_resume_snapshot(&store).expect("safe snapshot still present");
        let messages = loaded.graph.project_messages();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].parts[0].content, "hello");
        assert!(loaded.delta.is_some());
    }
}
