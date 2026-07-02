//! Runs the backend-agnostic `ProcessRegistry` conformance suite against the
//! Sqlite implementation. The same suite runs against the in-memory registry
//! in lash-core, so both backends are held to one contract.

use std::collections::BTreeMap;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use lash_core::runtime::RuntimeScope;
use lash_core::testing::conformance::{
    ReopenableProcessRegistry, ReopenableRuntimePersistence, ReopenableTriggerStore,
};
use lash_core::{
    DurabilityTier, EffectHost, ExecutionScope, ProcessExecutionEnvRef, ProcessExecutionEnvStore,
    ProcessOriginator, ProcessRegistry, RuntimeEffectCommand, RuntimeEffectController,
    RuntimeEffectControllerError, RuntimeEffectEnvelope, RuntimeEffectKind,
    RuntimeEffectLocalExecutor, RuntimeEffectOutcome, RuntimeInvocation, RuntimePersistence,
    SessionScope, SessionStoreFactory, TriggerOccurrenceRequest, TriggerStore,
    TriggerSubscriptionDraft, TriggerSubscriptionFilter,
};
use lash_sqlite_store::{
    SqliteEffectHost, SqliteEffectReplayOptions, SqliteProcessRegistry,
    SqliteRuntimeEffectController, SqliteSessionStoreFactory, SqliteTriggerStore, Store,
};
use tempfile::TempDir;

fn fresh_db_path(dirs: &Arc<Mutex<Vec<TempDir>>>, file_name: &str) -> PathBuf {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join(file_name);
    dirs.lock().expect("dirs lock").push(dir);
    path
}

fn sync_await<T, F>(future: F) -> T
where
    T: Send + 'static,
    F: Future<Output = T> + Send + 'static,
{
    std::thread::spawn(move || {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("runtime")
            .block_on(future)
    })
    .join()
    .expect("runtime thread")
}

fn open_registry(path: &Path) -> Arc<dyn ProcessRegistry> {
    let path = path.to_path_buf();
    Arc::new(sync_await(async move {
        SqliteProcessRegistry::open(&path)
            .await
            .expect("file registry")
    })) as Arc<dyn ProcessRegistry>
}

fn open_store(path: &Path) -> Arc<dyn RuntimePersistence> {
    let path = path.to_path_buf();
    Arc::new(sync_await(async move {
        Store::open(&path).await.expect("file store")
    })) as Arc<dyn RuntimePersistence>
}

fn artifact_store_handles(
    path: &Path,
) -> lash_lashlang_runtime::testing::conformance::ArtifactStoreHandles {
    let path = path.to_path_buf();
    let store = Arc::new(sync_await(async move {
        Store::open(&path).await.expect("file artifact store")
    }));
    lash_lashlang_runtime::testing::conformance::ArtifactStoreHandles {
        artifacts: Arc::clone(&store) as Arc<dyn lashlang::LashlangArtifactStore>,
        process_env: store as Arc<dyn ProcessExecutionEnvStore>,
    }
}

fn open_trigger_store(path: &Path) -> Arc<dyn TriggerStore> {
    let path = path.to_path_buf();
    Arc::new(sync_await(async move {
        SqliteTriggerStore::open(&path)
            .await
            .expect("file trigger store")
    })) as Arc<dyn TriggerStore>
}

#[tokio::test]
async fn sqlite_artifact_store_satisfies_conformance() {
    let dirs = Arc::new(Mutex::new(Vec::new()));
    lash_lashlang_runtime::testing::conformance::artifact_store_reopenable(|| {
        let path = fresh_db_path(&dirs, "artifacts.db");
        lash_lashlang_runtime::testing::conformance::ReopenableArtifactStore {
            open: artifact_store_handles(&path),
            reopen: artifact_store_handles(&path),
        }
    })
    .await;
}

fn trigger_subscription_draft(
    session_id: &str,
    source_key: &str,
    process_name: &str,
) -> TriggerSubscriptionDraft {
    let mut inputs = BTreeMap::new();
    inputs.insert("event".to_string(), lash_core::TriggerInputBinding::Event);
    let registrant_scope = SessionScope::new(session_id);
    TriggerSubscriptionDraft {
        registrant: ProcessOriginator::session(registrant_scope.clone()),
        env_ref: ProcessExecutionEnvRef::new(format!("process-env:{session_id}")),
        wake_target: Some(registrant_scope),
        name: Some(process_name.to_string()),
        source_type: "ui.button.pressed".to_string(),
        source_key: source_key.to_string(),
        source: serde_json::json!({}),
        payload_schema: lash_core::LashSchema::new(serde_json::json!({
            "type": "object",
            "required": ["button"],
            "properties": {
                "button": { "type": "string" }
            }
        })),
        target: lash_core::ProcessInput::Engine {
            kind: "test-trigger".to_string(),
            payload: serde_json::json!({}),
        },
        target_identity: lash_core::ProcessIdentity::new("test-trigger")
            .with_label(Some(process_name.to_string()))
            .with_definition(Some(serde_json::json!({ "process_name": process_name }))),
        event_types: Vec::new(),
        input_template: inputs,
        target_label: Some(process_name.to_string()),
    }
}

fn exec_envelope(replay_key: &str, code: &str) -> RuntimeEffectEnvelope {
    RuntimeEffectEnvelope::new(
        RuntimeInvocation::effect(
            RuntimeScope::for_turn("effect-session", "effect-turn", 1, 0),
            replay_key,
            RuntimeEffectKind::ExecCode,
            replay_key,
        ),
        RuntimeEffectCommand::ExecCode {
            language: "code".to_string(),
            code: code.to_string(),
        },
    )
}

fn exec_outcome(marker: &str) -> RuntimeEffectOutcome {
    RuntimeEffectOutcome::ExecCode {
        result: Ok(lash_core::ExecResponse {
            observations: Vec::new(),
            observation_truncation: Vec::new(),
            tool_calls: Vec::new(),
            images: Vec::new(),
            printed_images: Vec::new(),
            error: None,
            duration_ms: 0,
            terminal_finish: Some(serde_json::json!(marker)),
        }),
    }
}

fn assert_exec_marker(outcome: RuntimeEffectOutcome, expected: &str) {
    let RuntimeEffectOutcome::ExecCode { result } = outcome else {
        panic!("expected exec-code outcome");
    };
    let response = result.expect("exec-code response");
    assert_eq!(response.terminal_finish, Some(serde_json::json!(expected)));
}

fn returning_executor(marker: &'static str) -> RuntimeEffectLocalExecutor<'static> {
    RuntimeEffectLocalExecutor::testing(move |_| async move { Ok(exec_outcome(marker)) })
}

fn failing_executor() -> RuntimeEffectLocalExecutor<'static> {
    RuntimeEffectLocalExecutor::testing(|_| async move {
        Err(RuntimeEffectControllerError::new(
            "test_local_executor_called",
            "replay must not invoke the local executor",
        ))
    })
}

fn current_epoch_ms_for_test() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock before unix epoch")
        .as_millis()
        .min(u128::from(u64::MAX)) as u64
}

#[tokio::test]
async fn sqlite_process_registry_satisfies_conformance() {
    let dirs = Arc::new(Mutex::new(Vec::new()));
    lash_core::testing::conformance::process_registry_reopenable(|| {
        let path = fresh_db_path(&dirs, "processes.db");
        ReopenableProcessRegistry {
            open: open_registry(&path),
            reopen: open_registry(&path),
        }
    })
    .await;
}

#[tokio::test]
async fn sqlite_session_store_factory_satisfies_conformance() {
    let dirs = Arc::new(Mutex::new(Vec::new()));
    lash_core::testing::conformance::session_store_factory(
        || {
            let dir = tempfile::tempdir().expect("tempdir");
            let factory = Arc::new(SqliteSessionStoreFactory::new(dir.path()))
                as Arc<dyn SessionStoreFactory>;
            dirs.lock().expect("dirs lock").push(dir);
            factory
        },
        DurabilityTier::Durable,
    )
    .await;
}

#[tokio::test]
async fn sqlite_trigger_store_satisfies_conformance() {
    let dirs = Arc::new(Mutex::new(Vec::new()));
    lash_core::testing::conformance::trigger_store_reopenable(
        || {
            let path = fresh_db_path(&dirs, "triggers.db");
            ReopenableTriggerStore {
                open: open_trigger_store(&path),
                reopen: open_trigger_store(&path),
            }
        },
        DurabilityTier::Durable,
    )
    .await;
}

#[tokio::test]
async fn sqlite_trigger_store_persists_subscriptions_and_reserves_idempotently_after_reopen() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("triggers.db");
    let source_key = lash_core::empty_trigger_source_key("ui.button.pressed").expect("source key");

    let open = SqliteTriggerStore::open(&path)
        .await
        .expect("open trigger store");
    assert_eq!(open.durability_tier(), DurabilityTier::Durable);
    let registration = open
        .register_subscription(trigger_subscription_draft(
            "session-a",
            &source_key,
            "on_button",
        ))
        .await
        .expect("register subscription");
    assert_eq!(registration.handle, "trigger:1");

    let reopened = SqliteTriggerStore::open(&path)
        .await
        .expect("reopen trigger store");
    let mut source_filter = TriggerSubscriptionFilter::for_source_type("ui.button.pressed");
    source_filter.enabled = Some(true);
    let subscriptions = reopened
        .list_subscriptions(source_filter)
        .await
        .expect("list subscriptions");
    assert_eq!(subscriptions.len(), 1);
    assert_eq!(subscriptions[0].source_key, source_key);

    let occurrence = reopened
        .record_occurrence(TriggerOccurrenceRequest::new(
            "ui.button.pressed",
            subscriptions[0].source_key.clone(),
            serde_json::json!({ "button": "Blue" }),
            "button-blue-1",
        ))
        .await
        .expect("record occurrence");
    let first = reopened
        .reserve_matching_deliveries(&occurrence.occurrence_id)
        .await
        .expect("reserve first delivery");
    assert_eq!(first.len(), 1);
    assert_eq!(first[0].subscription.handle, registration.handle);

    let replayed = reopened
        .record_occurrence(TriggerOccurrenceRequest::new(
            "ui.button.pressed",
            subscriptions[0].source_key.clone(),
            serde_json::json!({ "button": "Blue" }),
            "button-blue-1",
        ))
        .await
        .expect("replay occurrence");
    assert_eq!(replayed.occurrence_id, occurrence.occurrence_id);
    let duplicate = reopened
        .reserve_matching_deliveries(&replayed.occurrence_id)
        .await
        .expect("reserve duplicate delivery");
    assert!(duplicate.is_empty());
}

#[tokio::test]
async fn sqlite_trigger_store_skips_malformed_matching_subscription_during_reservation() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("triggers.db");
    let source_key = lash_core::empty_trigger_source_key("ui.button.pressed").expect("source key");

    let (malformed, current) = {
        let store = SqliteTriggerStore::open(&path)
            .await
            .expect("open trigger store");
        let malformed = store
            .register_subscription(trigger_subscription_draft(
                "malformed-session",
                &source_key,
                "malformed_button",
            ))
            .await
            .expect("register malformed");
        let current = store
            .register_subscription(trigger_subscription_draft(
                "current-session",
                &source_key,
                "current_button",
            ))
            .await
            .expect("register current");
        (malformed, current)
    };

    let conn = rusqlite::Connection::open(&path).expect("open raw trigger db");
    conn.execute(
        "UPDATE trigger_subscriptions SET record_json = ?2 WHERE subscription_id = ?1",
        rusqlite::params![malformed.subscription_id.as_str(), "{not valid json"],
    )
    .expect("poison trigger row");
    drop(conn);

    let reopened = SqliteTriggerStore::open(&path)
        .await
        .expect("reopen trigger store");
    let occurrence = reopened
        .record_occurrence(TriggerOccurrenceRequest::new(
            "ui.button.pressed",
            source_key,
            serde_json::json!({ "button": "Blue" }),
            "button-blue-malformed-row",
        ))
        .await
        .expect("record occurrence");
    let deliveries = reopened
        .reserve_matching_deliveries(&occurrence.occurrence_id)
        .await
        .expect("reserve deliveries");

    assert_eq!(deliveries.len(), 1);
    assert_eq!(deliveries[0].subscription.handle, current.handle);
}

#[tokio::test]
async fn sqlite_trigger_store_cancels_by_session_and_handle() {
    let store = SqliteTriggerStore::memory()
        .await
        .expect("memory trigger store");
    let source_key = lash_core::empty_trigger_source_key("ui.button.pressed").expect("source key");
    let first = store
        .register_subscription(trigger_subscription_draft(
            "session-a",
            &source_key,
            "first",
        ))
        .await
        .expect("register first");
    let second = store
        .register_subscription(trigger_subscription_draft(
            "session-b",
            &source_key,
            "second",
        ))
        .await
        .expect("register second");

    assert!(
        !store
            .cancel_subscription("session-b", &first.handle)
            .await
            .expect("wrong session cancel")
    );
    assert!(
        store
            .cancel_subscription("session-b", &second.handle)
            .await
            .expect("cancel second")
    );

    let occurrence = store
        .record_occurrence(TriggerOccurrenceRequest::new(
            "ui.button.pressed",
            source_key,
            serde_json::json!({ "button": "Red" }),
            "button-red-1",
        ))
        .await
        .expect("record occurrence");
    let deliveries = store
        .reserve_matching_deliveries(&occurrence.occurrence_id)
        .await
        .expect("reserve deliveries");
    assert_eq!(deliveries.len(), 1);
    assert_eq!(deliveries[0].subscription.handle, first.handle);
}

#[tokio::test]
async fn sqlite_store_satisfies_runtime_persistence_conformance() {
    let dirs = Arc::new(Mutex::new(Vec::new()));
    lash_core::testing::conformance::runtime_persistence_reopenable(|| {
        let path = fresh_db_path(&dirs, "session.db");
        ReopenableRuntimePersistence {
            open: open_store(&path),
            reopen: open_store(&path),
        }
    })
    .await;
}

#[tokio::test]
async fn sqlite_store_schema_excludes_embedded_turn_replay_tables() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("schema.db");
    drop(Store::open(&path).await.expect("open store"));
    let conn = rusqlite::Connection::open(&path).expect("open raw sqlite");
    for removed in [
        concat!("runtime_", "turn_", "checkpoints"),
        concat!("runtime_", "effect_", "journal"),
    ] {
        let count = raw_count(
            &conn,
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = ?1",
            removed,
        );
        assert_eq!(count, 0, "{removed} table must not exist");
    }
    let turn_commits = raw_count(
        &conn,
        "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = ?1",
        "runtime_turn_commits",
    );
    assert_eq!(turn_commits, 1);
}

fn raw_count(conn: &rusqlite::Connection, sql: &str, name: &str) -> i64 {
    conn.query_row(sql, rusqlite::params![name], |row| row.get::<_, i64>(0))
        .expect("query sqlite_master")
}

#[tokio::test]
async fn sqlite_effect_host_satisfies_scope_conformance() {
    lash_core::testing::conformance::effect_host(|| {
        Arc::new(sync_await(async {
            SqliteEffectHost::memory().await.expect("effect host")
        })) as Arc<dyn EffectHost>
    })
    .await;
}

#[tokio::test]
async fn sqlite_effect_controller_satisfies_replay_conformance() {
    let controller = SqliteRuntimeEffectController::memory(ExecutionScope::turn(
        "effect-conformance-session",
        "effect-conformance-turn",
    ))
    .await
    .expect("controller");

    lash_core::testing::conformance::effect_controller_concurrent_replay_deterministic(
        &controller,
        || controller.start_replay(),
    )
    .await;

    let tool_controller = SqliteRuntimeEffectController::memory(ExecutionScope::turn(
        "tool-attempt-conformance-session",
        "tool-attempt-conformance-turn",
    ))
    .await
    .expect("tool attempt controller");
    lash_core::testing::conformance::effect_controller_tool_attempt_fanout_replay_deterministic(
        &tool_controller,
        || tool_controller.start_replay(),
    )
    .await;

    let durable_controller = SqliteRuntimeEffectController::memory(ExecutionScope::turn(
        "durable-step-session",
        "durable-step-turn",
    ))
    .await
    .expect("durable step controller");
    lash_core::testing::conformance::effect_controller_durable_steps_replay(
        &durable_controller,
        || durable_controller.start_replay(),
    )
    .await;
}

#[tokio::test]
async fn sqlite_effect_controller_replays_without_local_executor() {
    let controller = SqliteRuntimeEffectController::memory(ExecutionScope::turn("session", "turn"))
        .await
        .expect("controller");
    let envelope = exec_envelope("exec-replay", "first");
    let first = controller
        .execute_effect(envelope.clone(), returning_executor("recorded"))
        .await
        .expect("first effect");
    assert_exec_marker(first, "recorded");

    controller.start_replay();
    let replayed = controller
        .execute_effect(envelope, failing_executor())
        .await
        .expect("replayed effect");
    assert_exec_marker(replayed, "recorded");
}

#[tokio::test]
async fn sqlite_effect_controller_rejects_envelope_hash_conflict() {
    let controller = SqliteRuntimeEffectController::memory(ExecutionScope::turn("session", "turn"))
        .await
        .expect("controller");
    controller
        .execute_effect(
            exec_envelope("same-key", "first"),
            returning_executor("first"),
        )
        .await
        .expect("first effect");

    let err = controller
        .execute_effect(
            exec_envelope("same-key", "changed"),
            returning_executor("changed"),
        )
        .await
        .expect_err("same replay key with changed envelope must fail");
    assert_eq!(err.code, "sqlite_effect_replay_hash_conflict");
}

#[tokio::test]
async fn sqlite_effect_controller_satisfies_lease_fencing_conformance() {
    let dirs = Arc::new(Mutex::new(Vec::new()));
    let path = fresh_db_path(&dirs, "effect-lease-fencing.db");
    let make_path = path.clone();
    let steal_path = path.clone();
    let expire_path = path.clone();
    lash_core::testing::conformance::effect_controller_lease_fencing(
        lash_core::testing::conformance::EffectLeaseFencingBackend {
            make_controller: Box::new(move |ttl| {
                let path = make_path.clone();
                Box::pin(async move {
                    let controller = SqliteRuntimeEffectController::open_with_options(
                        &path,
                        ExecutionScope::turn("session", "turn"),
                        SqliteEffectReplayOptions { lease_ttl: ttl },
                    )
                    .await
                    .expect("controller");
                    let for_replay = controller.clone();
                    lash_core::testing::conformance::LeaseFencingController {
                        controller: Arc::new(controller),
                        start_replay: Box::new(move || for_replay.start_replay()),
                    }
                })
            }),
            steal_lease: Box::new(move |replay_key| {
                let path = steal_path.clone();
                Box::pin(async move {
                    let stolen_until = current_epoch_ms_for_test().saturating_add(10_000);
                    let conn = rusqlite::Connection::open(&path).expect("open sqlite");
                    let changed = conn
                        .execute(
                            "UPDATE runtime_effect_replay
                             SET lease_owner_id = 'stolen-owner',
                                 lease_token = 'stolen-token',
                                 lease_expires_at_ms = ?1
                             WHERE replay_key = ?2",
                            rusqlite::params![stolen_until as i64, replay_key],
                        )
                        .expect("steal lease row");
                    assert_eq!(changed, 1);
                })
            }),
            expire_lease: Box::new(move |replay_key| {
                let path = expire_path.clone();
                Box::pin(async move {
                    let conn = rusqlite::Connection::open(&path).expect("open sqlite");
                    let changed = conn
                        .execute(
                            "UPDATE runtime_effect_replay
                             SET lease_expires_at_ms = 0
                             WHERE replay_key = ?1",
                            rusqlite::params![replay_key],
                        )
                        .expect("expire lease row");
                    assert_eq!(changed, 1);
                })
            }),
        },
    )
    .await;
}

#[tokio::test]
async fn sqlite_sleep_replay_returns_after_recorded_due_time() {
    let controller = SqliteRuntimeEffectController::memory(ExecutionScope::turn("session", "turn"))
        .await
        .expect("controller");
    let envelope = RuntimeEffectEnvelope::new(
        RuntimeInvocation::effect(
            RuntimeScope::for_turn("session", "turn", 1, 0),
            "sleep",
            RuntimeEffectKind::Sleep,
            "sleep-key",
        ),
        RuntimeEffectCommand::Sleep { duration_ms: 120 },
    );

    let started = std::time::Instant::now();
    let first = controller
        .execute_effect(envelope.clone(), RuntimeEffectLocalExecutor::unavailable())
        .await
        .expect("first sleep");
    assert!(matches!(first, RuntimeEffectOutcome::Sleep));
    assert!(
        started.elapsed() >= std::time::Duration::from_millis(100),
        "first sleep must wait until the recorded due_at"
    );

    controller.start_replay();
    let replayed = tokio::time::timeout(
        std::time::Duration::from_millis(50),
        controller.execute_effect(envelope, failing_executor()),
    )
    .await
    .expect("replay must not sleep the full original duration")
    .expect("sleep replay");
    assert!(matches!(replayed, RuntimeEffectOutcome::Sleep));
}
