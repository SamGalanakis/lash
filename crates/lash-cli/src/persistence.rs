use lash_core::runtime::RuntimeSessionState;
use lash_core::store::RuntimeCommit;
use lash_sqlite_store::Store;

pub(crate) async fn persist_committed_runtime_state(
    store: &Store,
    state: &mut RuntimeSessionState,
) {
    match lash_core::SessionCommitStore::commit_runtime_state(
        store,
        RuntimeCommit::persisted_state(state, &[]),
    )
    .await
    {
        Ok(result) => {
            state.apply_persisted_commit_result(result);
        }
        Err(err) => tracing::warn!("failed to persist committed runtime state: {err}"),
    }
}
