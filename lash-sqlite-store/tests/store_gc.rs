use lash::{
    DynamicStateSnapshot, ExecutionMode, HydratedSessionCheckpoint, PersistedSessionConfig,
    PersistedSessionState, PersistedTurnState, PluginSessionSnapshot, RuntimeCommit,
    RuntimePersistence, SessionGraph, SessionHead, SessionReadScope, StandardContextApproach,
    TokenLedgerEntry, TokenUsage,
};
use lash_sqlite_store::{
    BlobArtifactDescriptor, BuiltinBlobProfile, Store, StoreGcPolicy, StoreOptions,
};

#[test]
fn gc_unreachable_keeps_rooted_checkpoint_blobs() {
    let store = Store::memory().expect("store");
    let stored = store.put_checkpoint(&HydratedSessionCheckpoint {
        turn_state: PersistedTurnState {
            iteration: 1,
            token_usage: TokenUsage::default(),
            last_prompt_usage: None,
            mode_turn_options: Default::default(),
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
        head_revision: 0,
        graph: SessionGraph::default(),
        config: PersistedSessionConfig {
            provider_id: "openai-compatible".into(),
            configured_model: "gpt-5.4-mini".into(),
            context_window: 200_000,
            execution_mode: ExecutionMode::standard(),
            standard_context_approach: Some(StandardContextApproach::default()),
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

#[tokio::test]
async fn runtime_commit_rejects_different_session_id_on_single_session_store() {
    let store = Store::memory().expect("store");
    let alpha = PersistedSessionState {
        session_id: "alpha".to_string(),
        ..PersistedSessionState::default()
    };
    let first = store
        .commit_runtime_state(RuntimeCommit::persisted_state(&alpha, &[]))
        .await;
    assert!(first.is_ok());

    let beta = PersistedSessionState {
        session_id: "beta".to_string(),
        ..PersistedSessionState::default()
    };
    let err = store
        .commit_runtime_state(RuntimeCommit::persisted_state(&beta, &[]))
        .await
        .expect_err("mismatched session commit should fail");
    assert!(err.to_string().contains("bound to session `alpha`"));
}

#[tokio::test]
async fn load_session_hydrates_checkpoint_and_usage_without_reentrant_locking() {
    let store = Store::memory().expect("store");
    let state = PersistedSessionState {
        session_id: "hydrated".to_string(),
        dynamic_state_snapshot: Some(DynamicStateSnapshot {
            tools: Default::default(),
            base_generation: 9,
        }),
        plugin_snapshot_revision: Some(12),
        plugin_snapshot: Some(PluginSessionSnapshot {
            plugins: Default::default(),
        }),
        ..PersistedSessionState::default()
    };
    let usage = TokenLedgerEntry {
        source: "turn".to_string(),
        model: "mock-model".to_string(),
        usage: TokenUsage {
            input_tokens: 11,
            output_tokens: 7,
            cached_input_tokens: 3,
            reasoning_tokens: 5,
        },
    };

    store
        .commit_runtime_state(RuntimeCommit::persisted_state(&state, &[usage]))
        .await
        .expect("commit");

    let read = store
        .load_session(SessionReadScope::FullGraph)
        .await
        .expect("load")
        .expect("session");
    let checkpoint = read.checkpoint.expect("checkpoint");
    assert_eq!(read.session_id, "hydrated");
    assert_eq!(
        checkpoint
            .dynamic_state
            .expect("dynamic snapshot")
            .base_generation,
        9
    );
    assert_eq!(checkpoint.plugin_snapshot_revision, Some(12));
    assert_eq!(read.token_ledger.len(), 1);
    assert_eq!(read.token_ledger[0].usage.input_tokens, 11);
}

#[tokio::test]
async fn auto_gc_runs_after_commit_without_reentrant_locking() {
    let store = Store::memory_with_options(StoreOptions {
        blob_profile: BuiltinBlobProfile::LowLatency,
        gc_policy: StoreGcPolicy {
            auto_run_every_commits: Some(1),
        },
    })
    .expect("store");
    let orphan =
        store.put_artifact_blob(BlobArtifactDescriptor::plugin_session_snapshot(), b"orphan");
    let state = PersistedSessionState {
        session_id: "auto-gc".to_string(),
        ..PersistedSessionState::default()
    };

    store
        .commit_runtime_state(RuntimeCommit::persisted_state(&state, &[]))
        .await
        .expect("commit");

    assert!(store.get_blob(&orphan).is_none());
}
