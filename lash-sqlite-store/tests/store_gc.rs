use lash::{
    ExecutionMode, HydratedSessionCheckpoint, PersistedSessionConfig, PersistedSessionState,
    PersistedTurnState, PluginSessionSnapshot, RuntimeCommit, RuntimePersistence, SessionGraph,
    SessionHead, SessionPolicy, SessionReadScope, SessionStoreCreateRequest, SessionStoreFactory,
    StandardContextApproach, TokenLedgerEntry, TokenUsage, ToolState,
};
use lash_sqlite_store::{
    BlobArtifactDescriptor, BuiltinBlobProfile, SqliteSessionStoreFactory, Store, StoreGcPolicy,
    StoreOptions,
};

#[test]
fn gc_unreachable_keeps_rooted_checkpoint_blobs() {
    let store = Store::memory().expect("store");
    let stored = store.put_checkpoint(&HydratedSessionCheckpoint {
        turn_state: PersistedTurnState {
            turn_index: 1,
            token_usage: TokenUsage::default(),
            last_prompt_usage: None,
            mode_turn_options: Default::default(),
        },
        tool_state_ref: None,
        tool_state: Some(ToolState::default().with_generation(7)),
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
    let dynamic_ref = checkpoint.tool_state_ref.expect("dynamic state ref");
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
        tool_state_snapshot: Some(ToolState::default().with_generation(9)),
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
            .tool_state
            .expect("dynamic snapshot")
            .generation(),
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

#[test]
fn sqlite_factory_uses_deterministic_safe_session_paths() {
    let root = unique_temp_dir("paths");
    let factory = SqliteSessionStoreFactory::new(&root);

    let first = factory.path_for_session("../weird/session");
    let second = factory.path_for_session("../weird/session");

    assert_eq!(first, second);
    assert_eq!(first.parent(), Some(root.as_path()));
    assert!(
        first
            .file_name()
            .unwrap()
            .to_string_lossy()
            .ends_with(".db")
    );
    assert!(!first.file_name().unwrap().to_string_lossy().contains('/'));
}

#[tokio::test]
async fn sqlite_factory_creates_metadata_once_and_preserves_on_reopen() {
    let root = unique_temp_dir("metadata");
    let factory = SqliteSessionStoreFactory::new(&root);
    let request = SessionStoreCreateRequest {
        session_id: "chat/alpha".to_string(),
        parent_session_id: Some("parent".to_string()),
        policy: SessionPolicy {
            model: "first-model".to_string(),
            ..SessionPolicy::default()
        },
    };

    let store = factory.create_store(&request).expect("create store");
    let meta = store
        .load_session_meta()
        .await
        .expect("load meta")
        .expect("meta");
    assert_eq!(meta.session_id, "chat/alpha");
    assert_eq!(meta.model, "first-model");

    store
        .save_session_meta(lash::SessionMeta {
            session_id: "chat/alpha".to_string(),
            session_name: "Renamed".to_string(),
            created_at: "original".to_string(),
            model: "preserved-model".to_string(),
            cwd: Some("/tmp/original".to_string()),
            parent_session_id: Some("parent".to_string()),
        })
        .await
        .expect("save meta");

    let reopened = factory
        .create_store(&SessionStoreCreateRequest {
            policy: SessionPolicy {
                model: "second-model".to_string(),
                ..SessionPolicy::default()
            },
            ..request
        })
        .expect("reopen store");
    let reopened_meta = reopened
        .load_session_meta()
        .await
        .expect("load reopened meta")
        .expect("reopened meta");
    assert_eq!(reopened_meta.session_name, "Renamed");
    assert_eq!(reopened_meta.model, "preserved-model");
    assert_eq!(reopened_meta.created_at, "original");
}

#[tokio::test]
async fn sqlite_factory_is_explicitly_usable_as_session_store_factory() {
    let root = unique_temp_dir("explicit");
    let factory: std::sync::Arc<dyn SessionStoreFactory> =
        std::sync::Arc::new(SqliteSessionStoreFactory::new(&root));
    let request = SessionStoreCreateRequest {
        session_id: "explicit".to_string(),
        parent_session_id: None,
        policy: SessionPolicy {
            model: "model".to_string(),
            ..SessionPolicy::default()
        },
    };

    let store = factory.create_store(&request).expect("create store");

    assert!(
        store
            .load_session_meta()
            .await
            .expect("load meta")
            .is_some()
    );
}

fn unique_temp_dir(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "lash-sqlite-store-{name}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).expect("temp dir");
    dir
}
