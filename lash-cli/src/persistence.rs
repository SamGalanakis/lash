use anyhow::Context;
use lash::{LashRuntime, PersistedSessionState, SessionGraph, Store};

use crate::app::UiResumeState;
use crate::ui_resume;

pub(crate) async fn persist_committed_runtime_state(
    store: &Store,
    state: &mut PersistedSessionState,
    ui_state: &UiResumeState,
) {
    ui_resume::save_ui_resume_state(store, ui_state);
    match lash::RuntimeStore::apply_runtime_commit(
        store,
        lash::RuntimeCommit::persisted_state(state, &[]),
    )
    .await
    {
        Ok(lash::RuntimeCommitResult::PersistedState(result)) => {
            state.apply_persisted_commit_result(result);
        }
        Ok(_) => unreachable!("persisted state commit should return persisted result"),
        Err(err) => tracing::warn!("failed to persist committed runtime state: {err}"),
    }
}

pub(crate) async fn persist_live_runtime_state(
    store: &Store,
    seed_graph: Option<SessionGraph>,
    live_graph: SessionGraph,
    ui_state: &UiResumeState,
    state: &PersistedSessionState,
) {
    ui_resume::save_ui_resume_state(store, ui_state);
    if let Some(commit) = lash::RuntimeCommit::live_resume(seed_graph, live_graph, state)
        && let Err(err) = lash::RuntimeStore::apply_runtime_commit(store, commit).await
    {
        tracing::warn!("failed to persist live runtime state: {err}");
    }
}

pub(crate) async fn snapshot_execution_state(
    runtime: &mut LashRuntime,
    state: &mut PersistedSessionState,
) -> anyhow::Result<()> {
    let snapshot = runtime
        .snapshot_execution_state()
        .await
        .context("failed to snapshot execution state")?;
    state.set_execution_state_snapshot(snapshot);
    Ok(())
}
