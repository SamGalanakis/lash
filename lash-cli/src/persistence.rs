use anyhow::Context;
use lash::{LashRuntime, PersistedSessionState};
use lash_sqlite_store::Store;

pub(crate) async fn persist_committed_runtime_state(
    store: &Store,
    state: &mut PersistedSessionState,
) {
    match lash::RuntimePersistence::commit_runtime_state(
        store,
        lash::RuntimeCommit::persisted_state(state, &[]),
    )
    .await
    {
        Ok(result) => {
            state.apply_persisted_commit_result(result);
        }
        Err(err) => tracing::warn!("failed to persist committed runtime state: {err}"),
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
