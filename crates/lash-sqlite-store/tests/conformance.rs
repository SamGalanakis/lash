//! Runs the backend-agnostic `ProcessRegistry` conformance suite against the
//! Sqlite implementation. The same suite runs against the in-memory registry
//! in lash-core, so both backends are held to one contract.

use std::future::Future;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use lash_core::runtime::RuntimeScope;
use lash_core::testing::conformance::{
    ReopenableLashlangArtifactStore, ReopenableProcessRegistry, ReopenableRuntimePersistence,
};
use lash_core::{
    DurabilityTier, EffectHost, EffectScope, LashlangArtifactStore, ProcessRegistry,
    RuntimeEffectCommand, RuntimeEffectController, RuntimeEffectControllerError,
    RuntimeEffectEnvelope, RuntimeEffectKind, RuntimeEffectLocalExecutor, RuntimeEffectOutcome,
    RuntimeInvocation, RuntimePersistence,
};
use lash_sqlite_store::{
    Store, SqliteEffectHost, SqliteEffectReplayOptions, SqliteProcessRegistry,
    SqliteRuntimeEffectController,
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

fn open_artifact_store(path: &Path) -> Arc<dyn LashlangArtifactStore> {
    let path = path.to_path_buf();
    Arc::new(sync_await(async move {
        Store::open(&path).await.expect("file store")
    })) as Arc<dyn LashlangArtifactStore>
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
async fn sqlite_store_satisfies_lashlang_artifact_store_conformance() {
    let dirs = Arc::new(Mutex::new(Vec::new()));
    lash_core::testing::conformance::lashlang_artifact_store_reopenable(
        || {
            let path = fresh_db_path(&dirs, "artifacts.db");
            ReopenableLashlangArtifactStore {
                open: open_artifact_store(&path),
                reopen: open_artifact_store(&path),
            }
        },
        DurabilityTier::Durable,
    )
    .await;
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
    let controller = SqliteRuntimeEffectController::memory(EffectScope::turn(
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
}

#[tokio::test]
async fn sqlite_effect_controller_replays_without_local_executor() {
    let controller = SqliteRuntimeEffectController::memory(EffectScope::turn("session", "turn"))
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
    let controller = SqliteRuntimeEffectController::memory(EffectScope::turn("session", "turn"))
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
async fn sqlite_effect_controller_reclaims_stale_in_progress_lease() {
    let controller = SqliteRuntimeEffectController::memory_with_options(
        EffectScope::turn("session", "turn"),
        SqliteEffectReplayOptions {
            lease_ttl: std::time::Duration::from_millis(20),
        },
    )
    .await
    .expect("controller");
    let envelope = exec_envelope("stale-lease", "work");
    let (entered_tx, entered_rx) = tokio::sync::oneshot::channel();
    let release = Arc::new(tokio::sync::Notify::new());
    let first_controller = controller.clone();
    let first_envelope = envelope.clone();
    let first_release = Arc::clone(&release);
    let first = tokio::spawn(async move {
        first_controller
            .execute_effect(
                first_envelope,
                RuntimeEffectLocalExecutor::testing(move |_| async move {
                    let _ = entered_tx.send(());
                    first_release.notified().await;
                    Ok(exec_outcome("stale-owner"))
                }),
            )
            .await
    });
    entered_rx.await.expect("first executor entered");
    tokio::time::sleep(std::time::Duration::from_millis(40)).await;

    let second = controller
        .execute_effect(envelope.clone(), returning_executor("reclaimed-owner"))
        .await
        .expect("stale lease reclaimed");
    assert_exec_marker(second, "reclaimed-owner");

    release.notify_waiters();
    let first_err = first
        .await
        .expect("first task joins")
        .expect_err("stale owner must not finalize after lease loss");
    assert_eq!(first_err.code, "sqlite_effect_replay_lease_lost");

    controller.start_replay();
    let replayed = controller
        .execute_effect(envelope, failing_executor())
        .await
        .expect("replayed reclaimed outcome");
    assert_exec_marker(replayed, "reclaimed-owner");
}

#[tokio::test]
async fn sqlite_sleep_replay_returns_after_recorded_due_time() {
    let controller = SqliteRuntimeEffectController::memory(EffectScope::turn("session", "turn"))
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
