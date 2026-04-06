use lash::store::LiveSessionSnapshot;
use lash::{DynamicStateSnapshot, SessionStateEnvelope, Store};

use crate::app::UiResumeState;

const LIVE_RESUME_SNAPSHOT_VERSION: u8 = 1;

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
    let repl_snapshot = state.repl_snapshot.clone();
    let mut stored_state = state.clone();
    stored_state.repl_snapshot = None;
    let payload = LiveResumeSnapshotPayload {
        version: LIVE_RESUME_SNAPSHOT_VERSION,
        state: stored_state,
        dynamic_state: dynamic_state.clone(),
        ui_state: ui_state.clone(),
    };
    let snapshot_json = serde_json::to_string(&payload).map_err(|err| err.to_string())?;
    store.save_live_session_snapshot(LiveSessionSnapshot {
        snapshot_json,
        repl_snapshot,
    });
    Ok(())
}

pub(crate) fn load_live_resume_snapshot(store: &Store) -> Option<LoadedLiveResumeSnapshot> {
    let stored = store.load_live_session_snapshot()?;
    let payload: LiveResumeSnapshotPayload = serde_json::from_str(&stored.snapshot_json).ok()?;
    if payload.version != LIVE_RESUME_SNAPSHOT_VERSION {
        return None;
    }
    let mut state = payload.state;
    state.repl_snapshot = stored.repl_snapshot;
    Some(LoadedLiveResumeSnapshot {
        state,
        dynamic_state: payload.dynamic_state,
        ui_state: payload.ui_state,
    })
}
