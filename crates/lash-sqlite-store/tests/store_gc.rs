use lash_core::{
    AttachmentId, AttachmentIntent, AttachmentManifest, ExecutionMode, HostModeProtocol,
    HydratedSessionCheckpoint, ModeConfig, ModePreamble, ModeTurnOptions, PersistedSessionConfig,
    RuntimeSessionState, PersistedTurnState, PluginSessionSnapshot, PreparedPrompt,
    PromptContext, RUNTIME_EFFECT_JOURNAL_SCHEMA_VERSION, RuntimeCommit, RuntimeEffectJournalRecord,
    RuntimeEffectKind, RuntimeEffectOutcome, RuntimePersistence, RuntimeTurnCheckpoint,
    RuntimeTurnCompletion, RuntimeTurnMachineConfigSnapshot, SessionGraph, SessionHead,
    SessionPolicy, SessionReadScope, SessionStoreCreateRequest, SessionStoreFactory,
    StandardContextApproach, StoreError, TokenLedgerEntry, TokenUsage, ToolState,
};
use lash_sqlite_store::{
    BlobArtifactDescriptor, BuiltinBlobProfile, SqliteSessionStoreFactory, Store, StoreGcPolicy,
    StoreOptions,
};
use std::sync::Arc;

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
    let alpha = RuntimeSessionState {
        session_id: "alpha".to_string(),
        ..RuntimeSessionState::default()
    };
    let first = store
        .commit_runtime_state(RuntimeCommit::persisted_state(&alpha, &[]))
        .await;
    assert!(first.is_ok());

    let beta = RuntimeSessionState {
        session_id: "beta".to_string(),
        ..RuntimeSessionState::default()
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
    let state = RuntimeSessionState {
        session_id: "hydrated".to_string(),
        tool_state_snapshot: Some(ToolState::default().with_generation(9)),
        plugin_snapshot_revision: Some(12),
        plugin_snapshot: Some(PluginSessionSnapshot {
            plugins: Default::default(),
        }),
        ..RuntimeSessionState::default()
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
            originating_tool_call_id: None,
        },
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
        .save_session_meta(lash_core::SessionMeta {
            session_id: "chat/alpha".to_string(),
            session_name: "Renamed".to_string(),
            created_at: "original".to_string(),
            model: "preserved-model".to_string(),
            cwd: Some("/tmp/original".to_string()),
            relation: lash_core::SessionRelation::Child {
                parent_session_id: "parent".to_string(),
                originating_tool_call_id: None,
            },
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
        relation: lash_core::SessionRelation::Root,
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

#[tokio::test]
async fn runtime_effect_journal_replays_by_idempotency_key_and_clears_on_final_commit() {
    let store = Store::memory().expect("store");
    let lease = store
        .claim_runtime_turn_lease("root", "turn-1", "test-owner", 60_000)
        .await
        .expect("lease");
    let record = RuntimeEffectJournalRecord {
        schema_version: RUNTIME_EFFECT_JOURNAL_SCHEMA_VERSION,
        session_id: "root".to_string(),
        turn_id: "turn-1".to_string(),
        idempotency_key: "root:turn-1:1:0:sleep:1".to_string(),
        envelope_hash: "hash-a".to_string(),
        effect_kind: RuntimeEffectKind::Sleep,
        outcome: RuntimeEffectOutcome::Sleep,
        created_at_epoch_ms: 1,
    };
    store
        .save_runtime_effect_outcome(&lease, record.clone())
        .await
        .expect("save journal");

    let loaded = store
        .load_runtime_effect_outcome("root", "turn-1", &record.idempotency_key)
        .await
        .expect("load journal")
        .expect("journal record");
    assert_eq!(loaded.envelope_hash, "hash-a");

    let state = RuntimeSessionState {
        session_id: "root".to_string(),
        ..RuntimeSessionState::default()
    };
    let commit = RuntimeCommit::persisted_state(&state, &[])
        .clearing_completed_turn(RuntimeTurnCompletion::from_lease(&lease));
    store
        .commit_runtime_state(commit)
        .await
        .expect("final commit clears turn");

    let cleared = store
        .load_runtime_effect_outcome("root", "turn-1", &record.idempotency_key)
        .await
        .expect("load after clear");
    assert!(cleared.is_none());
    assert!(
        store
            .load_runtime_turn_checkpoint("root", "turn-1")
            .await
            .expect("load checkpoint")
            .is_none()
    );
}

#[tokio::test]
async fn stale_completed_turn_commit_rejects_and_preserves_resume_state() {
    let store = Store::memory().expect("store");
    let old = store
        .claim_runtime_turn_lease("root", "turn-stale-final", "owner-a", 20)
        .await
        .expect("old lease");
    store
        .save_runtime_turn_checkpoint(&old, runtime_turn_checkpoint("root", "turn-stale-final"))
        .await
        .expect("save checkpoint");
    let record = runtime_effect_record("root", "turn-stale-final", "current");
    store
        .save_runtime_effect_outcome(&old, record.clone())
        .await
        .expect("save journal");
    tokio::time::sleep(std::time::Duration::from_millis(30)).await;
    let current = store
        .claim_runtime_turn_lease("root", "turn-stale-final", "owner-b", 60_000)
        .await
        .expect("new lease");

    let state = RuntimeSessionState {
        session_id: "root".to_string(),
        ..RuntimeSessionState::default()
    };
    let err = store
        .commit_runtime_state(
            RuntimeCommit::persisted_state(
                &state,
                &[TokenLedgerEntry {
                    source: "turn".to_string(),
                    model: "mock".to_string(),
                    usage: TokenUsage {
                        input_tokens: 1,
                        output_tokens: 0,
                        cached_input_tokens: 0,
                        reasoning_tokens: 0,
                    },
                }],
            )
            .clearing_completed_turn(RuntimeTurnCompletion::from_lease(&old)),
        )
        .await
        .expect_err("stale final commit must fail");

    assert!(matches!(err, StoreError::RuntimeTurnLeaseExpired { .. }));
    assert!(
        store
            .load_runtime_turn_checkpoint("root", "turn-stale-final")
            .await
            .expect("load checkpoint")
            .is_some()
    );
    assert!(
        store
            .load_runtime_effect_outcome("root", "turn-stale-final", &record.idempotency_key)
            .await
            .expect("load journal")
            .is_some()
    );
    assert!(
        store
            .load_session(SessionReadScope::FullGraph)
            .await
            .expect("load session")
            .is_none()
    );
    store
        .save_runtime_effect_outcome(
            &current,
            runtime_effect_record("root", "turn-stale-final", "after-stale-final"),
        )
        .await
        .expect("current owner can still write");
}

#[tokio::test]
async fn expired_completed_turn_commit_rejects_and_preserves_resume_state() {
    let store = Store::memory().expect("store");
    let lease = store
        .claim_runtime_turn_lease("root", "turn-expired-final", "owner-a", 20)
        .await
        .expect("lease");
    store
        .save_runtime_turn_checkpoint(
            &lease,
            runtime_turn_checkpoint("root", "turn-expired-final"),
        )
        .await
        .expect("save checkpoint");
    let record = runtime_effect_record("root", "turn-expired-final", "effect");
    store
        .save_runtime_effect_outcome(&lease, record.clone())
        .await
        .expect("save journal");
    tokio::time::sleep(std::time::Duration::from_millis(30)).await;

    let state = RuntimeSessionState {
        session_id: "root".to_string(),
        ..RuntimeSessionState::default()
    };
    let err = store
        .commit_runtime_state(
            RuntimeCommit::persisted_state(&state, &[])
                .clearing_completed_turn(RuntimeTurnCompletion::from_lease(&lease)),
        )
        .await
        .expect_err("expired final commit must fail");

    assert!(matches!(err, StoreError::RuntimeTurnLeaseExpired { .. }));
    assert!(
        store
            .load_runtime_turn_checkpoint("root", "turn-expired-final")
            .await
            .expect("load checkpoint")
            .is_some()
    );
    assert!(
        store
            .load_runtime_effect_outcome("root", "turn-expired-final", &record.idempotency_key)
            .await
            .expect("load journal")
            .is_some()
    );
    assert!(
        store
            .load_session(SessionReadScope::FullGraph)
            .await
            .expect("load session")
            .is_none()
    );
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
            .load_runtime_effect_outcome("root", "turn-abandon", &record.idempotency_key)
            .await
            .expect("load journal")
            .is_some()
    );
    store
        .claim_runtime_turn_lease("root", "turn-abandon", "owner-b", 60_000)
        .await
        .expect("new owner can claim abandoned turn");
}

#[tokio::test]
async fn superseded_runtime_turn_lease_cannot_write_or_clear_newer_owner() {
    let store = Store::memory().expect("store");
    let old = store
        .claim_runtime_turn_lease("root", "turn-superseded", "owner-a", 0)
        .await
        .expect("old lease");
    let current = store
        .claim_runtime_turn_lease("root", "turn-superseded", "owner-b", 60_000)
        .await
        .expect("new lease");

    let stale_save = store
        .save_runtime_effect_outcome(
            &old,
            runtime_effect_record("root", "turn-superseded", "stale"),
        )
        .await;
    assert!(matches!(
        stale_save,
        Err(StoreError::RuntimeTurnLeaseExpired { .. })
    ));

    store
        .abandon_runtime_turn_lease(&old)
        .await
        .expect("stale abandon is ignored");

    let conflict = store
        .claim_runtime_turn_lease("root", "turn-superseded", "owner-c", 60_000)
        .await;
    assert!(matches!(
        conflict,
        Err(StoreError::RuntimeTurnLeaseConflict { .. })
    ));
    store
        .save_runtime_effect_outcome(
            &current,
            runtime_effect_record("root", "turn-superseded", "current"),
        )
        .await
        .expect("current owner can still write");
}

#[tokio::test]
async fn renewed_runtime_turn_lease_survives_original_expiry() {
    let store = Store::memory().expect("store");
    let lease = store
        .claim_runtime_turn_lease("root", "turn-renew", "owner-a", 20)
        .await
        .expect("lease");
    let renewed = store
        .renew_runtime_turn_lease(&lease, 60_000)
        .await
        .expect("renew lease");
    tokio::time::sleep(std::time::Duration::from_millis(5)).await;

    store
        .save_runtime_effect_outcome(
            &renewed,
            runtime_effect_record("root", "turn-renew", "renewed"),
        )
        .await
        .expect("renewed lease can write after original expiry");
}

#[tokio::test]
async fn active_runtime_turn_lease_fences_competing_claims() {
    let store = Store::memory().expect("store");
    store
        .claim_runtime_turn_lease("root", "turn-active", "owner-a", 60_000)
        .await
        .expect("lease");

    let conflict = store
        .claim_runtime_turn_lease("root", "turn-active", "owner-b", 60_000)
        .await;

    assert!(matches!(
        conflict,
        Err(StoreError::RuntimeTurnLeaseConflict { .. })
    ));
}

struct NoopDriver;

impl lash_sansio::ProtocolDriverHandle<HostModeProtocol> for NoopDriver {
    fn prepare_mode_iteration(
        &self,
        _ctx: lash_core::DriverContextView<'_>,
    ) -> Vec<lash_core::DriverAction> {
        Vec::new()
    }

    fn handle_llm_success(
        &self,
        _ctx: lash_core::DriverContextView<'_>,
        _waiting: lash_sansio::WaitingLlmState<HostModeProtocol>,
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
        _waiting: lash_sansio::WaitingExecState<HostModeProtocol>,
        _result: Result<lash_core::ExecResponse, String>,
    ) -> Vec<lash_core::DriverAction> {
        Vec::new()
    }
}

fn runtime_turn_checkpoint(session_id: &str, turn_id: &str) -> RuntimeTurnCheckpoint {
    let termination = ModeTurnOptions::default();
    let tool_names = Arc::new(Vec::new());
    let mode_preamble = Arc::new(ModePreamble {
        config: ModeConfig::chat(
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
        mode: ExecutionMode::standard(),
        messages: lash_core::MessageSequence::default(),
        events: Arc::new(Vec::new()),
        mode_run_offset: 0,
        mode_preamble: Arc::clone(&mode_preamble),
        prepared_prompt: PreparedPrompt {
            context: PromptContext::default(),
            system_prompt: Arc::from(""),
        },
        max_turns: None,
        model_variant: None,
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
        mode_iteration: 0,
        checkpoint_hash,
        machine_config: RuntimeTurnMachineConfigSnapshot {
            execution_mode: ExecutionMode::standard(),
            session_id: session_id.to_string(),
            run_session_id: None,
            autonomous: false,
            model: "mock-model".to_string(),
            model_variant: None,
            max_turns: None,
            sync_execution_surface: false,
            tool_specs: Vec::new(),
            system_prompt: String::new(),
            termination,
        },
        checkpoint,
        mode_turn_options: ModeTurnOptions::default(),
        turn_prompt_layer: lash_core::PromptLayer::new(),
        provider_id: "mock-provider".to_string(),
        model: "mock-model".to_string(),
        model_variant: None,
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
        idempotency_key: format!("{session_id}:{turn_id}:{effect}"),
        envelope_hash: format!("hash-{effect}"),
        effect_kind: RuntimeEffectKind::Sleep,
        outcome: RuntimeEffectOutcome::Sleep,
        created_at_epoch_ms: 1,
    }
}

#[tokio::test]
async fn attachment_manifest_records_intent_and_commit_stamps_atomically() {
    let store = Store::memory().expect("store");
    let attachment_a = AttachmentId::new("aaaa".to_string());
    let attachment_b = AttachmentId::new("bbbb".to_string());

    AttachmentManifest::record_intent(
        &store,
        AttachmentIntent {
            attachment_id: attachment_a.clone(),
            session_id: "root".to_string(),
            canonical_uri: "sha256:aaaa".to_string(),
            intent_at_epoch_ms: 100,
        },
    )
    .expect("intent a");
    AttachmentManifest::record_intent(
        &store,
        AttachmentIntent {
            attachment_id: attachment_b.clone(),
            session_id: "root".to_string(),
            canonical_uri: "sha256:bbbb".to_string(),
            intent_at_epoch_ms: 100,
        },
    )
    .expect("intent b");

    // Both are uncommitted; both surface when sweeping for intents
    // older than 200.
    let uncommitted = AttachmentManifest::list_uncommitted(&store, 200).expect("list");
    assert_eq!(uncommitted.len(), 2);

    // Commit one of them via a RuntimeCommit that names it. The other
    // remains uncommitted so a later GC sweep can reconcile it.
    let commit = RuntimeCommit {
        session_id: "root".to_string(),
        expected_head_revision: Some(0),
        config: PersistedSessionConfig::default(),
        graph: lash_core::GraphCommitDelta::ReplaceFull(SessionGraph::default()),
        checkpoint: HydratedSessionCheckpoint {
            turn_state: PersistedTurnState::default(),
            tool_state_ref: None,
            tool_state: None,
            plugin_snapshot_ref: None,
            plugin_snapshot_revision: None,
            plugin_snapshot: None,
            execution_state_ref: None,
            execution_state: None,
        },
        usage_deltas: Vec::new(),
        completed_turn: None,
        committed_attachment_ids: vec![attachment_a.clone()],
    };
    store
        .commit_runtime_state(commit)
        .await
        .expect("commit succeeds");

    let still_uncommitted = AttachmentManifest::list_uncommitted(&store, 200).expect("list");
    assert_eq!(still_uncommitted.len(), 1);
    assert_eq!(still_uncommitted[0].attachment_id, attachment_b);
    assert!(still_uncommitted[0].committed_at_epoch_ms.is_none());

    // Forget the orphan after the GC removes its bytes.
    AttachmentManifest::forget(&store, &attachment_b).expect("forget");
    let after_forget = AttachmentManifest::list_uncommitted(&store, 200).expect("list");
    assert!(after_forget.is_empty());
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
