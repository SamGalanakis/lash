use anyhow::Context;
use lash::{DynamicStateSnapshot, LashRuntime, RuntimePersistenceState, SessionGraph, Store};

use crate::app::UiResumeState;
use crate::ui_resume;

pub(crate) fn persist_committed_runtime_state(
    store: &Store,
    state: &mut RuntimePersistenceState,
    ui_state: &UiResumeState,
    dynamic_state: &DynamicStateSnapshot,
) {
    ui_resume::save_ui_resume_state(store, ui_state);
    store.persist_committed_runtime_state(state, dynamic_state);
}

pub(crate) fn persist_live_runtime_state(
    store: &Store,
    seed_graph: Option<SessionGraph>,
    live_graph: SessionGraph,
    ui_state: &UiResumeState,
    state: &RuntimePersistenceState,
    dynamic_state: &DynamicStateSnapshot,
) {
    ui_resume::save_ui_resume_state(store, ui_state);
    store.persist_live_runtime_state(seed_graph, live_graph, state, dynamic_state);
}

pub(crate) async fn snapshot_execution_state(
    runtime: &mut LashRuntime,
    state: &mut RuntimePersistenceState,
) -> anyhow::Result<()> {
    let snapshot = runtime
        .snapshot_execution_state()
        .await
        .context("failed to snapshot execution state")?;
    state.set_execution_state_snapshot(snapshot);
    Ok(())
}
