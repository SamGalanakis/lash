#[cfg(test)]
use lash::{DynamicStateSnapshot, SessionStateEnvelope};
use lash::{SessionGraph, Store};

use crate::app::UiResumeState;
use crate::ui_resume;

#[derive(Clone, Debug)]
pub(crate) struct LoadedLiveResumeSnapshot {
    pub(crate) graph: SessionGraph,
    pub(crate) ui_state: UiResumeState,
}

#[cfg(test)]
fn stamp_live_graph(
    mut graph: SessionGraph,
    state: &SessionStateEnvelope,
    dynamic_state: &DynamicStateSnapshot,
) -> SessionGraph {
    graph.merge_active_projection(&state.messages, &state.tool_calls);
    let plugin_snapshot = graph.latest_plugin_snapshot();
    graph.record_runtime_state(
        &lash::PersistedSessionConfig {
            provider_id: state.policy.provider.id().to_string(),
            configured_model: state.policy.model.clone(),
            context_window: state.policy.max_context_tokens.unwrap_or_default() as u64,
            execution_mode: state.policy.execution_mode,
            context_approach: state.policy.context_approach.clone(),
            model_variant: state.policy.model_variant.clone(),
        },
        &lash::PersistedTurnState {
            iteration: state.iteration,
            token_usage: state.token_usage.clone(),
            last_prompt_usage: state.last_prompt_usage.clone(),
        },
        Some(dynamic_state),
        plugin_snapshot.as_ref(),
        state.execution_state_snapshot.as_deref(),
        &state.token_ledger,
    );
    graph
}

#[cfg(test)]
pub(crate) fn save_live_resume_snapshot(
    store: &Store,
    state: &SessionStateEnvelope,
    ui_state: &UiResumeState,
    dynamic_state: &DynamicStateSnapshot,
) -> Result<(), String> {
    if !lash::messages_are_live_resume_safe(&state.messages) {
        return Ok(());
    }
    ui_resume::save_ui_resume_state(store, ui_state);
    let graph = stamp_live_graph(state.session_graph.clone(), state, dynamic_state);
    store.save_live_session_graph(graph);
    Ok(())
}

pub(crate) fn load_live_resume_snapshot(store: &Store) -> Option<LoadedLiveResumeSnapshot> {
    let graph = store.load_live_session_graph()?;
    if !lash::messages_are_live_resume_safe(&graph.project_messages()) {
        store.clear_live_session_graph();
        return None;
    }
    Some(LoadedLiveResumeSnapshot {
        graph,
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
        SessionStateEnvelope {
            session_id: "root".to_string(),
            policy: lash::SessionPolicy::default(),
            session_graph: lash::SessionGraph::from_projection(&messages, &[]),
            messages,
            ..SessionStateEnvelope::default()
        }
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
    }
}
