use lash_core::PersistedSessionState;
use lash_sqlite_store::Store;

pub(crate) async fn persist_committed_runtime_state(
    store: &Store,
    state: &mut PersistedSessionState,
) {
    match lash_core::RuntimePersistence::commit_runtime_state(
        store,
        lash_core::RuntimeCommit::persisted_state(state, &[]),
    )
    .await
    {
        Ok(result) => {
            state.apply_persisted_commit_result(result);
        }
        Err(err) => tracing::warn!("failed to persist committed runtime state: {err}"),
    }
}
