#![cfg(feature = "sqlite-store")]

use lash::{
    BlobArtifactDescriptor, ContextApproach, DynamicStateSnapshot, ExecutionMode,
    HydratedSessionCheckpoint, PersistedSessionConfig, PersistedSessionState, PersistedStateCommit,
    PersistedTurnState, PluginSessionSnapshot, SessionGraph, SessionHead, Store, TokenUsage,
};

#[test]
fn gc_unreachable_keeps_rooted_checkpoint_blobs() {
    let store = Store::memory().expect("store");
    let stored = store.put_checkpoint(&HydratedSessionCheckpoint {
        turn_state: PersistedTurnState {
            iteration: 1,
            token_usage: TokenUsage::default(),
            last_prompt_usage: None,
        },
        dynamic_state_ref: None,
        dynamic_state: Some(DynamicStateSnapshot {
            tools: Default::default(),
            base_generation: 7,
        }),
        plugin_snapshot_ref: None,
        plugin_snapshot_revision: Some(11),
        plugin_snapshot: Some(PluginSessionSnapshot {
            plugins: Default::default(),
        }),
        execution_state_ref: None,
        execution_state: None,
    });
    store.save_session_head(SessionHead {
        session_id: "root".to_string(),
        graph: SessionGraph::default(),
        config: PersistedSessionConfig {
            provider_id: "openai-compatible".into(),
            configured_model: "gpt-5.4-mini".into(),
            context_window: 200_000,
            execution_mode: ExecutionMode::Standard,
            context_approach: ContextApproach::default(),
            model_variant: None,
        },
        checkpoint_ref: Some(stored.checkpoint_ref.clone()),
        token_ledger: Vec::new(),
    });
    let orphan =
        store.put_artifact_blob(BlobArtifactDescriptor::plugin_session_snapshot(), b"orphan");

    let report = store.gc_unreachable();

    assert_eq!(report.deleted_blob_count, 1);
    let checkpoint = store
        .get_checkpoint(&stored.checkpoint_ref)
        .expect("checkpoint manifest");
    let dynamic_ref = checkpoint.dynamic_state_ref.expect("dynamic state ref");
    let plugin_ref = checkpoint.plugin_snapshot_ref.expect("plugin snapshot ref");
    assert!(store.get_blob(&stored.checkpoint_ref).is_some());
    assert!(store.get_blob(&dynamic_ref).is_some());
    assert!(store.get_blob(&plugin_ref).is_some());
    assert!(store.get_blob(&orphan).is_none());
}

#[test]
fn runtime_commit_rejects_different_session_id_on_single_session_store() {
    let store = Store::memory().expect("store");
    let alpha = PersistedSessionState {
        session_id: "alpha".to_string(),
        ..PersistedSessionState::default()
    };
    let first = store.apply_runtime_commit(PersistedStateCommit::persisted_state(&alpha, &[]));
    assert!(first.is_ok());

    let beta = PersistedSessionState {
        session_id: "beta".to_string(),
        ..PersistedSessionState::default()
    };
    let err = store
        .apply_runtime_commit(PersistedStateCommit::persisted_state(&beta, &[]))
        .expect_err("mismatched session commit should fail");
    assert!(err.to_string().contains("bound to session `alpha`"));
}
