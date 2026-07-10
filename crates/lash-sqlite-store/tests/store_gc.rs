use lash_core::{
    HydratedSessionCheckpoint, LeaseOwnerIdentity, ModelSpec, PersistedSessionConfig,
    PersistedTurnState, PluginSessionSnapshot, RuntimeCommit, RuntimeSessionState,
    SessionCommitStore, SessionExecutionLeaseStore, SessionGraph, SessionHead, SessionPolicy,
    SessionStoreCreateRequest, SessionStoreFactory, TokenUsage, ToolState,
};
use lash_sqlite_store::{
    BlobArtifactDescriptor, BuiltinBlobProfile, SqliteSessionStoreFactory, Store, StoreGcPolicy,
    StoreOptions,
};

fn model_spec(id: &str) -> ModelSpec {
    ModelSpec::from_token_limits(id, Default::default(), 200_000, None)
        .expect("valid test model spec")
}

fn test_model_spec() -> ModelSpec {
    model_spec("gpt-5.4-mini")
}

fn lease_owner(owner_id: &str) -> LeaseOwnerIdentity {
    LeaseOwnerIdentity::opaque(owner_id, format!("{owner_id}:incarnation"))
}

#[tokio::test]
async fn gc_unreachable_keeps_rooted_checkpoint_blobs() {
    let store = Store::memory().await.expect("store");
    let stored = store
        .put_checkpoint(&HydratedSessionCheckpoint {
            turn_state: PersistedTurnState {
                turn_index: 1,
                token_usage: TokenUsage::default(),
                last_prompt_usage: None,
                protocol_turn_options: Default::default(),
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
        })
        .await;
    store
        .save_session_head(SessionHead {
            session_id: "root".to_string(),
            head_revision: 0,
            agent_frames: Vec::new(),
            current_agent_frame_id: String::new(),
            graph: SessionGraph::default(),
            config: PersistedSessionConfig {
                provider_id: "openai-compatible".into(),
                model: test_model_spec(),
            },
            checkpoint_ref: Some(stored.checkpoint_ref.clone()),
            token_ledger: Vec::new(),
        })
        .await;
    let orphan = store
        .put_artifact_blob(BlobArtifactDescriptor::plugin_session_snapshot(), b"orphan")
        .await;

    let report = store.gc_unreachable().await;

    assert_eq!(report.deleted_blob_count, 1);
    let checkpoint = store
        .get_checkpoint(&stored.checkpoint_ref)
        .await
        .expect("checkpoint manifest");
    let dynamic_ref = checkpoint.tool_state_ref.expect("dynamic state ref");
    let plugin_ref = checkpoint.plugin_snapshot_ref.expect("plugin snapshot ref");
    assert!(store.get_blob(&stored.checkpoint_ref).await.is_some());
    assert!(store.get_blob(&dynamic_ref).await.is_some());
    assert!(store.get_blob(&plugin_ref).await.is_some());
    assert!(store.get_blob(&orphan).await.is_none());
}

#[tokio::test]
async fn auto_gc_runs_after_commit_without_reentrant_locking() {
    let store = Store::memory_with_options(StoreOptions {
        blob_profile: BuiltinBlobProfile::LowLatency,
        gc_policy: StoreGcPolicy {
            auto_run_every_commits: Some(1),
        },
    })
    .await
    .expect("store");
    let orphan = store
        .put_artifact_blob(BlobArtifactDescriptor::plugin_session_snapshot(), b"orphan")
        .await;
    let state = RuntimeSessionState {
        session_id: "auto-gc".to_string(),
        ..RuntimeSessionState::default()
    };
    let owner = lease_owner("auto-gc-test");
    let session_lease = store
        .try_claim_session_execution_lease("auto-gc", &owner, 60_000)
        .await
        .expect("claim session execution lease")
        .acquired()
        .expect("session execution lease");

    store
        .commit_runtime_state(
            RuntimeCommit::persisted_state(&state, &[])
                .with_session_execution_lease(session_lease.fence())
                .releasing_session_execution_lease(session_lease.completion()),
        )
        .await
        .expect("commit");

    assert!(store.get_blob(&orphan).await.is_none());
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
        relation: lash_core::SessionRelation::Child {
            parent_session_id: "parent".to_string(),
            caused_by: None,
        },
        policy: SessionPolicy {
            model: model_spec("first-model"),
            ..SessionPolicy::default()
        },
    };

    let store = factory.create_store(&request).await.expect("create store");
    let meta = store
        .load_session_meta()
        .await
        .expect("load meta")
        .expect("meta");
    assert_eq!(meta.session_id, "chat/alpha");
    assert_eq!(meta.model, "first-model");

    store
        .save_session_meta(lash_core::SessionMeta {
            session_id: "chat/alpha".to_string(),
            session_name: "Renamed".to_string(),
            created_at: "original".to_string(),
            model: "preserved-model".to_string(),
            cwd: Some("/tmp/original".to_string()),
            relation: lash_core::SessionRelation::Child {
                parent_session_id: "parent".to_string(),
                caused_by: None,
            },
        })
        .await
        .expect("save meta");

    let reopened = factory
        .create_store(&SessionStoreCreateRequest {
            policy: SessionPolicy {
                model: model_spec("second-model"),
                ..SessionPolicy::default()
            },
            ..request
        })
        .await
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
        relation: lash_core::SessionRelation::Root,
        policy: SessionPolicy {
            model: model_spec("model"),
            ..SessionPolicy::default()
        },
    };

    let store = factory.create_store(&request).await.expect("create store");

    assert!(
        store
            .load_session_meta()
            .await
            .expect("load meta")
            .is_some()
    );
}

#[tokio::test]
async fn sqlite_factory_delete_session_removes_database_and_sidecars_idempotently() {
    let root = unique_temp_dir("delete-session");
    let factory = SqliteSessionStoreFactory::new(&root);
    let db_path = factory.path_for_session("delete/me");
    let wal_path = sidecar_path(&db_path, "-wal");
    let shm_path = sidecar_path(&db_path, "-shm");
    std::fs::create_dir_all(&root).expect("create session root");
    std::fs::write(&db_path, b"db").expect("write db file");
    std::fs::write(&wal_path, b"wal").expect("write wal sidecar");
    std::fs::write(&shm_path, b"shm").expect("write shm sidecar");

    factory
        .delete_session("delete/me")
        .await
        .expect("delete session");
    factory
        .delete_session("delete/me")
        .await
        .expect("delete session again");

    assert!(!db_path.exists());
    assert!(!wal_path.exists());
    assert!(!shm_path.exists());
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

fn sidecar_path(path: &std::path::Path, suffix: &str) -> std::path::PathBuf {
    let mut sidecar = path.as_os_str().to_os_string();
    sidecar.push(suffix);
    std::path::PathBuf::from(sidecar)
}
