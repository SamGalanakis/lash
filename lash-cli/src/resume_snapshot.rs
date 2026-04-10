use lash::store::LiveSessionSnapshot;
use lash::{DynamicStateSnapshot, SessionStateEnvelope, Store};

use crate::app::UiResumeState;

const LIVE_RESUME_SNAPSHOT_VERSION: u8 = 2;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct LiveResumeSnapshotPayload {
    version: u8,
    state: SessionStateEnvelope,
    dynamic_state: DynamicStateSnapshot,
    ui_state: UiResumeState,
}

#[derive(Clone, Debug)]
pub(crate) struct LoadedLiveResumeSnapshot {
    pub(crate) state: SessionStateEnvelope,
    pub(crate) dynamic_state: DynamicStateSnapshot,
    pub(crate) ui_state: UiResumeState,
}

pub(crate) fn save_live_resume_snapshot(
    store: &Store,
    state: &SessionStateEnvelope,
    ui_state: &UiResumeState,
    dynamic_state: &DynamicStateSnapshot,
) -> Result<(), String> {
    if !lash::messages_are_live_resume_safe(&state.messages) {
        return Ok(());
    }
    let execution_state_snapshot = state.execution_state_snapshot.clone();
    let mut stored_state = state.clone();
    stored_state.execution_state_snapshot = None;
    let payload = LiveResumeSnapshotPayload {
        version: LIVE_RESUME_SNAPSHOT_VERSION,
        state: stored_state,
        dynamic_state: dynamic_state.clone(),
        ui_state: ui_state.clone(),
    };
    let snapshot_json = serde_json::to_string(&payload).map_err(|err| err.to_string())?;
    store.save_live_session_snapshot(LiveSessionSnapshot {
        snapshot_json,
        execution_state_snapshot,
    });
    Ok(())
}

pub(crate) fn load_live_resume_snapshot(store: &Store) -> Option<LoadedLiveResumeSnapshot> {
    let stored = store.load_live_session_snapshot()?;
    let payload: LiveResumeSnapshotPayload = serde_json::from_str(&stored.snapshot_json).ok()?;
    if payload.version != LIVE_RESUME_SNAPSHOT_VERSION {
        store.clear_live_session_snapshot();
        return None;
    }
    if !lash::messages_are_live_resume_safe(&payload.state.messages) {
        store.clear_live_session_snapshot();
        return None;
    }
    let mut state = payload.state;
    state.execution_state_snapshot = stored.execution_state_snapshot;
    Some(LoadedLiveResumeSnapshot {
        state,
        dynamic_state: payload.dynamic_state,
        ui_state: payload.ui_state,
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
        assert_eq!(loaded.state.messages.len(), 1);
        assert_eq!(loaded.state.messages[0].parts[0].content, "hello");
    }

    #[test]
    fn old_live_snapshot_versions_are_ignored() {
        let store = Store::memory().expect("store");
        store.save_live_session_snapshot(LiveSessionSnapshot {
            snapshot_json: serde_json::json!({
                "version": 1,
                "state": snapshot_state(vec![text_message("m0", "legacy")]),
                "dynamic_state": empty_dynamic_state(),
                "ui_state": UiResumeState::default(),
            })
            .to_string(),
            execution_state_snapshot: None,
        });

        assert!(load_live_resume_snapshot(&store).is_none());
        assert!(store.load_live_session_snapshot().is_none());
    }
}
