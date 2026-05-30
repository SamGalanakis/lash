use lash_core::{
    HostTurnProtocol, HydratedSessionCheckpoint, ModelSpec, PersistedSessionConfig,
    PersistedTurnState, PluginSessionSnapshot, PreparedPrompt, PromptContext, ProtocolTurnOptions,
    RUNTIME_EFFECT_JOURNAL_SCHEMA_VERSION, RuntimeCommit, RuntimeEffectJournalRecord,
    RuntimeEffectKind, RuntimeEffectOutcome, RuntimePersistence, RuntimeSessionState,
    RuntimeTurnCheckpoint, RuntimeTurnMachineConfigSnapshot, SessionGraph, SessionHead,
    SessionPolicy, SessionStoreCreateRequest, SessionStoreFactory, TokenUsage, ToolState,
    TurnDriverConfig, TurnDriverPreamble,
};
use lash_sqlite_store::{
    BlobArtifactDescriptor, BuiltinBlobProfile, SqliteSessionStoreFactory, Store, StoreGcPolicy,
    StoreOptions,
};
use std::sync::Arc;

fn model_spec(id: &str) -> ModelSpec {
    ModelSpec::from_token_limits(id, None, 200_000, None, None).expect("valid test model spec")
}

fn test_model_spec() -> ModelSpec {
    model_spec("gpt-5.4-mini")
}

#[test]
fn gc_unreachable_keeps_rooted_checkpoint_blobs() {
    let store = Store::memory().expect("store");
    let stored = store.put_checkpoint(&HydratedSessionCheckpoint {
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
    });
    store.save_session_head(SessionHead {
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
    let state = RuntimeSessionState {
        session_id: "auto-gc".to_string(),
        ..RuntimeSessionState::default()
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
        relation: lash_core::SessionRelation::Child {
            parent_session_id: "parent".to_string(),
            caused_by: None,
        },
        policy: SessionPolicy {
            model: model_spec("first-model"),
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

    let store = factory.create_store(&request).expect("create store");

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
    let request = SessionStoreCreateRequest {
        session_id: "delete/me".to_string(),
        relation: lash_core::SessionRelation::Root,
        policy: SessionPolicy {
            model: model_spec("model"),
            ..SessionPolicy::default()
        },
    };
    let store = factory.create_store(&request).expect("create store");
    drop(store);
    let db_path = factory.path_for_session("delete/me");
    let wal_path = sidecar_path(&db_path, "-wal");
    let shm_path = sidecar_path(&db_path, "-shm");
    std::fs::write(&wal_path, b"wal").expect("write wal sidecar");
    std::fs::write(&shm_path, b"shm").expect("write shm sidecar");

    factory.delete_session("delete/me").expect("delete session");
    factory
        .delete_session("delete/me")
        .expect("delete session again");

    assert!(!db_path.exists());
    assert!(!wal_path.exists());
    assert!(!shm_path.exists());
}

#[tokio::test]
async fn abandon_runtime_turn_lease_releases_owner_and_preserves_resume_data() {
    let store = Store::memory().expect("store");
    let lease = store
        .claim_runtime_turn_lease("root", "turn-abandon", "owner-a", 60_000)
        .await
        .expect("lease");
    let checkpoint = runtime_turn_checkpoint("root", "turn-abandon");
    store
        .save_runtime_turn_checkpoint(&lease, checkpoint)
        .await
        .expect("save checkpoint");
    let record = runtime_effect_record("root", "turn-abandon", "effect-a");
    store
        .save_runtime_effect_outcome(&lease, record.clone())
        .await
        .expect("save journal");

    store
        .abandon_runtime_turn_lease(&lease)
        .await
        .expect("abandon lease");

    assert!(
        store
            .load_runtime_turn_checkpoint("root", "turn-abandon")
            .await
            .expect("load checkpoint")
            .is_some()
    );
    assert!(
        store
            .load_runtime_effect_outcome("root", "turn-abandon", &record.replay_key)
            .await
            .expect("load journal")
            .is_some()
    );
    store
        .claim_runtime_turn_lease("root", "turn-abandon", "owner-b", 60_000)
        .await
        .expect("new owner can claim abandoned turn");
}

struct NoopDriver;

impl lash_sansio::ProtocolDriverHandle<HostTurnProtocol> for NoopDriver {
    fn prepare_protocol_iteration(
        &self,
        _ctx: lash_core::DriverContextView<'_>,
    ) -> Vec<lash_core::DriverAction> {
        Vec::new()
    }

    fn handle_llm_success(
        &self,
        _ctx: lash_core::DriverContextView<'_>,
        _waiting: lash_sansio::WaitingLlmState<HostTurnProtocol>,
        _llm_response: lash_core::LlmResponse,
        _text_streamed: bool,
    ) -> Vec<lash_core::DriverAction> {
        Vec::new()
    }

    fn handle_tool_results(
        &self,
        _ctx: lash_core::DriverContextView<'_>,
        _completed: Vec<lash_sansio::CompletedToolCall>,
    ) -> Vec<lash_core::DriverAction> {
        Vec::new()
    }

    fn handle_exec_result(
        &self,
        _ctx: lash_core::DriverContextView<'_>,
        _waiting: lash_sansio::WaitingExecState<HostTurnProtocol>,
        _result: Result<lash_core::ExecResponse, String>,
    ) -> Vec<lash_core::DriverAction> {
        Vec::new()
    }
}

fn runtime_turn_checkpoint(session_id: &str, turn_id: &str) -> RuntimeTurnCheckpoint {
    let termination = ProtocolTurnOptions::default();
    let tool_names = Arc::new(Vec::new());
    let turn_driver_preamble = Arc::new(TurnDriverPreamble {
        config: TurnDriverConfig::chat(
            Arc::new(NoopDriver),
            false,
            Arc::new(|message_id, _max_turns| lash_core::Message {
                id: message_id.clone(),
                role: lash_core::MessageRole::System,
                parts: lash_core::shared_parts(Vec::new()),
                origin: None,
            }),
        ),
        tool_specs: Arc::new(Vec::new()),
        tool_names: Arc::clone(&tool_names),
        tool_names_fingerprint: lash_core::prompt_tool_names_fingerprint(&tool_names),
        omitted_tool_count: 0,
        execution_prompt: Arc::from(""),
        prompt_contributions: Vec::new(),
    });
    let prepared = lash_core::build_turn(lash_core::SansIoTurnInput {
        session_id: session_id.to_string(),
        run_session_id: None,
        autonomous: false,
        model: "mock-model".to_string(),
        messages: lash_core::MessageSequence::default(),
        events: Arc::new(Vec::new()),
        turn_causes: Vec::new(),
        protocol_run_offset: 0,
        turn_driver_preamble: Arc::clone(&turn_driver_preamble),
        prepared_prompt: PreparedPrompt {
            context: PromptContext::default(),
            system_prompt: Arc::from(""),
        },
        max_turns: None,
        model_variant: None,
        generation: lash_core::GenerationOptions::default(),
        emit_llm_trace: false,
        termination: termination.clone(),
    });
    let checkpoint = prepared.machine.checkpoint();
    let checkpoint_hash =
        lash_core::runtime_turn_checkpoint_hash(&checkpoint).expect("checkpoint hash");
    RuntimeTurnCheckpoint {
        schema_version: lash_core::RUNTIME_TURN_CHECKPOINT_SCHEMA_VERSION,
        session_id: session_id.to_string(),
        turn_id: turn_id.to_string(),
        turn_index: 1,
        protocol_iteration: 0,
        checkpoint_hash,
        machine_config: RuntimeTurnMachineConfigSnapshot {
            session_id: session_id.to_string(),
            run_session_id: None,
            autonomous: false,
            model: model_spec("mock-model"),
            generation: lash_core::GenerationOptions::default(),
            max_turns: None,
            sync_execution_surface: false,
            tool_specs: Arc::new(Vec::new()),
            system_prompt: String::new(),
            termination,
        },
        checkpoint,
        protocol_turn_options: ProtocolTurnOptions::default(),
        turn_prompt_layer: lash_core::PromptLayer::new(),
        provider_id: "mock-provider".to_string(),
        model: model_spec("mock-model"),
        updated_at_epoch_ms: 1,
    }
}

fn runtime_effect_record(
    session_id: &str,
    turn_id: &str,
    effect: &str,
) -> RuntimeEffectJournalRecord {
    RuntimeEffectJournalRecord {
        schema_version: RUNTIME_EFFECT_JOURNAL_SCHEMA_VERSION,
        session_id: session_id.to_string(),
        turn_id: turn_id.to_string(),
        replay_key: format!("{session_id}:{turn_id}:{effect}"),
        envelope_hash: format!("hash-{effect}"),
        effect_kind: RuntimeEffectKind::Sleep,
        outcome: RuntimeEffectOutcome::Sleep,
        created_at_epoch_ms: 1,
    }
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
