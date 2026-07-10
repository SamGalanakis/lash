use std::sync::{Arc, LazyLock};

use lash_core::testing::conformance::{
    ReopenableProcessRegistry, ReopenableRuntimePersistence, ReopenableTriggerStore,
};
use lash_core::{
    DurabilityTier, ExecutionScope, ProcessExecutionEnvStore, ProcessRegistry, RuntimePersistence,
    SessionStoreFactory, TriggerStore,
};
use lash_postgres_store::{
    PostgresEffectReplayOptions, PostgresRuntimeEffectController, PostgresStorage,
};

/// All backend suites share one database and `reset()` truncates every `lash_*`
/// table between cases, so they must not touch it concurrently. This guard
/// serializes them intrinsically — correctness no longer depends on the test
/// harness being invoked with `--test-threads=1`.
static DB_GUARD: LazyLock<tokio::sync::Mutex<()>> = LazyLock::new(|| tokio::sync::Mutex::new(()));

fn database_url() -> Option<String> {
    std::env::var("LASH_POSTGRES_DATABASE_URL").ok()
}

fn sync_await<T: Send + 'static>(
    future: impl std::future::Future<Output = T> + Send + 'static,
) -> T {
    // Drive the future on the CURRENT (multi-thread) test runtime rather than a
    // throwaway one. The sqlx pool's connections are bound to this runtime's
    // reactor; polling them from a different runtime wedges the connection (it
    // never returns to the pool), which starves the pool and surfaces as
    // PoolTimedOut. `block_in_place` lets this worker block while tokio spins up a
    // replacement, so the conformance harness keeps making progress.
    tokio::task::block_in_place(|| tokio::runtime::Handle::current().block_on(future))
}

async fn storage() -> Option<PostgresStorage> {
    let url = database_url()?;
    Some(
        PostgresStorage::connect(&url)
            .await
            .expect("connect postgres"),
    )
}

async fn reset(storage: &PostgresStorage) {
    let pool = storage.pool();
    // Derive the truncate set from the live catalog rather than hand-maintaining
    // a table list: a new `lash_*` table can no longer silently bleed state
    // between conformance cases. `lash_schema_versions` is excluded — it holds
    // the component schema version gate, not per-case fixture rows.
    let tables: Vec<String> = sqlx::query_scalar(
        "SELECT tablename FROM pg_tables
         WHERE schemaname = 'public'
           AND tablename LIKE 'lash\\_%'
           AND tablename <> 'lash_schema_versions'
         ORDER BY tablename",
    )
    .fetch_all(pool)
    .await
    .expect("list lash_* conformance tables");
    assert!(
        !tables.is_empty(),
        "expected the lash_* schema tables to exist before reset"
    );
    let truncate = format!("TRUNCATE {} RESTART IDENTITY CASCADE", tables.join(", "));
    sqlx::query(&truncate)
        .execute(pool)
        .await
        .expect("reset postgres conformance tables");
    sqlx::query(
        "INSERT INTO lash_process_change_clock (singleton, current_seq)
         VALUES (TRUE, 0)
         ON CONFLICT (singleton) DO UPDATE SET current_seq = EXCLUDED.current_seq",
    )
    .execute(pool)
    .await
    .expect("reset postgres process change clock");
    // `lash_trigger_subscription_seq` is a standalone sequence (not owned by a
    // truncated table), so RESTART IDENTITY does not reset it. Reset it in a
    // separate statement — sqlx's prepared protocol rejects multiple commands in
    // one query.
    sqlx::query("ALTER SEQUENCE lash_trigger_subscription_seq RESTART WITH 1")
        .execute(pool)
        .await
        .expect("reset postgres trigger subscription sequence");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn postgres_runtime_persistence_satisfies_conformance_when_configured() {
    let _db_guard = DB_GUARD.lock().await;
    let Some(storage) = storage().await else {
        eprintln!("skipping Postgres conformance: LASH_POSTGRES_DATABASE_URL is not set");
        return;
    };
    let storage = Arc::new(storage);
    lash_core::testing::conformance::runtime_persistence_reopenable(|| {
        let storage = Arc::clone(&storage);
        sync_await(async move {
            reset(&storage).await;
            let open = Arc::new(storage.unbound_session_store()) as Arc<dyn RuntimePersistence>;
            let reopen = Arc::new(storage.unbound_session_store()) as Arc<dyn RuntimePersistence>;
            ReopenableRuntimePersistence { open, reopen }
        })
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn postgres_artifact_store_satisfies_conformance_when_configured() {
    let _db_guard = DB_GUARD.lock().await;
    let Some(storage) = storage().await else {
        eprintln!(
            "skipping Postgres artifact-store conformance: LASH_POSTGRES_DATABASE_URL is not set"
        );
        return;
    };
    let storage = Arc::new(storage);
    lash_lashlang_runtime::testing::conformance::artifact_store_reopenable(|| {
        let storage = Arc::clone(&storage);
        sync_await(async move {
            reset(&storage).await;
            let handles = || lash_lashlang_runtime::testing::conformance::ArtifactStoreHandles {
                artifacts: Arc::new(storage.lashlang_artifact_store())
                    as Arc<dyn lashlang::LashlangArtifactStore>,
                process_env: Arc::new(storage.process_env_store())
                    as Arc<dyn ProcessExecutionEnvStore>,
            };
            lash_lashlang_runtime::testing::conformance::ReopenableArtifactStore {
                open: handles(),
                reopen: handles(),
            }
        })
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn postgres_session_store_factory_satisfies_conformance_when_configured() {
    let _db_guard = DB_GUARD.lock().await;
    let Some(storage) = storage().await else {
        eprintln!(
            "skipping Postgres session-store-factory conformance: LASH_POSTGRES_DATABASE_URL is not set"
        );
        return;
    };
    let storage = Arc::new(storage);
    lash_core::testing::conformance::session_store_factory(
        || {
            let storage = Arc::clone(&storage);
            sync_await(async move {
                reset(&storage).await;
                Arc::new(storage.session_store_factory()) as Arc<dyn SessionStoreFactory>
            })
        },
        DurabilityTier::Durable,
    )
    .await;
}

// Blocker 1: `from_pool` must enforce the same component schema-version gate as
// `connect`/`connect_with`. Writing a stale version (10) into
// `lash_schema_versions` and then constructing over the pool must fail loudly with
// the mismatch error, so a pre-cutover database can never be adopted post-bump.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn postgres_from_pool_enforces_schema_version_gate_when_configured() {
    let _db_guard = DB_GUARD.lock().await;
    let Some(storage) = storage().await else {
        eprintln!("skipping Postgres from_pool gate test: LASH_POSTGRES_DATABASE_URL is not set");
        return;
    };
    let pool = storage.pool().clone();
    // Force the recorded component version to a stale value.
    sqlx::query(
        "INSERT INTO lash_schema_versions (component, version) VALUES ('lash-postgres-store', 10)
         ON CONFLICT (component) DO UPDATE SET version = EXCLUDED.version",
    )
    .execute(&pool)
    .await
    .expect("write stale schema version");

    let result = PostgresStorage::from_pool(pool.clone()).await;

    // Restore the correct version BEFORE asserting so a failed assert never leaves
    // the shared database wedged for other cases.
    sqlx::query(
        "UPDATE lash_schema_versions SET version = 11 WHERE component = 'lash-postgres-store'",
    )
    .execute(&pool)
    .await
    .expect("restore schema version");

    let message = match result {
        Ok(_) => panic!("from_pool must reject a version-10 database"),
        Err(err) => err.to_string(),
    };
    assert!(
        message.contains("version 10") && message.contains("expected 11"),
        "expected a schema-version mismatch error, got: {message}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn postgres_runtime_effect_controller_satisfies_conformance_when_configured() {
    let _db_guard = DB_GUARD.lock().await;
    let Some(storage) = storage().await else {
        eprintln!(
            "skipping Postgres runtime-effect conformance: LASH_POSTGRES_DATABASE_URL is not set"
        );
        return;
    };
    reset(&storage).await;

    let host = Arc::new(storage.effect_host()) as Arc<dyn lash_core::EffectHost>;
    lash_core::testing::conformance::effect_host_durable_steps(|| Arc::clone(&host)).await;

    let controller = storage.runtime_effect_controller(ExecutionScope::runtime_operation(
        "postgres-effect-controller-conformance",
    ));
    lash_core::testing::conformance::effect_controller_durable_steps_replay(&controller, || {
        controller.start_replay()
    })
    .await;

    let controller = storage.runtime_effect_controller(ExecutionScope::runtime_operation(
        "postgres-effect-controller-concurrent-conformance",
    ));
    lash_core::testing::conformance::effect_controller_concurrent_replay_deterministic(
        &controller,
        || controller.start_replay(),
    )
    .await;

    let controller = storage.runtime_effect_controller(ExecutionScope::runtime_operation(
        "postgres-effect-controller-tool-conformance",
    ));
    lash_core::testing::conformance::effect_controller_tool_attempt_fanout_replay_deterministic(
        &controller,
        || controller.start_replay(),
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn postgres_effect_controller_satisfies_lease_fencing_conformance_when_configured() {
    let _db_guard = DB_GUARD.lock().await;
    let Some(storage) = storage().await else {
        eprintln!(
            "skipping Postgres effect lease-fencing conformance: LASH_POSTGRES_DATABASE_URL is not set"
        );
        return;
    };
    reset(&storage).await;

    let make_storage = storage.clone();
    let steal_pool = storage.pool().clone();
    let expire_pool = storage.pool().clone();
    lash_core::testing::conformance::effect_controller_lease_fencing(
        lash_core::testing::conformance::EffectLeaseFencingBackend {
            make_controller: Box::new(move |ttl| {
                let storage = make_storage.clone();
                Box::pin(async move {
                    let controller = PostgresRuntimeEffectController::with_options(
                        &storage,
                        ExecutionScope::turn("session", "turn"),
                        PostgresEffectReplayOptions {
                            lease_timings: lash_core::LeaseTimings::from_ttl(ttl)
                                .expect("conformance lease timings"),
                        },
                    );
                    let for_replay = controller.clone();
                    lash_core::testing::conformance::LeaseFencingController {
                        controller: Arc::new(controller),
                        start_replay: Box::new(move || for_replay.start_replay()),
                    }
                })
            }),
            steal_lease: Box::new(move |replay_key| {
                let pool = steal_pool.clone();
                Box::pin(async move {
                    let stolen_until = epoch_ms_for_test().saturating_add(10_000) as i64;
                    let changed = sqlx::query(
                        "UPDATE lash_runtime_effect_replay
                         SET lease_owner_id = 'stolen-owner',
                             lease_token = 'stolen-token',
                             lease_expires_at_ms = $1
                         WHERE replay_key = $2",
                    )
                    .bind(stolen_until)
                    .bind(&replay_key)
                    .execute(&pool)
                    .await
                    .expect("steal lease row")
                    .rows_affected();
                    assert_eq!(changed, 1);
                })
            }),
            expire_lease: Box::new(move |replay_key| {
                let pool = expire_pool.clone();
                Box::pin(async move {
                    let changed = sqlx::query(
                        "UPDATE lash_runtime_effect_replay
                         SET lease_expires_at_ms = 0
                         WHERE replay_key = $1",
                    )
                    .bind(&replay_key)
                    .execute(&pool)
                    .await
                    .expect("expire lease row")
                    .rows_affected();
                    assert_eq!(changed, 1);
                })
            }),
        },
    )
    .await;
}

fn epoch_ms_for_test() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock after epoch")
        .as_millis() as u64
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn postgres_process_registry_satisfies_conformance_when_configured() {
    let _db_guard = DB_GUARD.lock().await;
    let Some(storage) = storage().await else {
        eprintln!("skipping Postgres process conformance: LASH_POSTGRES_DATABASE_URL is not set");
        return;
    };
    let storage = Arc::new(storage);
    lash_core::testing::conformance::process_registry_reopenable(|| {
        let storage = Arc::clone(&storage);
        sync_await(async move {
            reset(&storage).await;
            let open = Arc::new(storage.process_registry()) as Arc<dyn ProcessRegistry>;
            let reopen = Arc::new(storage.process_registry()) as Arc<dyn ProcessRegistry>;
            ReopenableProcessRegistry { open, reopen }
        })
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn postgres_trigger_store_satisfies_conformance_when_configured() {
    let _db_guard = DB_GUARD.lock().await;
    let Some(storage) = storage().await else {
        eprintln!("skipping Postgres trigger conformance: LASH_POSTGRES_DATABASE_URL is not set");
        return;
    };
    let storage = Arc::new(storage);
    lash_core::testing::conformance::trigger_store_reopenable(
        || {
            let storage = Arc::clone(&storage);
            sync_await(async move {
                reset(&storage).await;
                let open = Arc::new(storage.trigger_store()) as Arc<dyn TriggerStore>;
                let reopen = Arc::new(storage.trigger_store()) as Arc<dyn TriggerStore>;
                ReopenableTriggerStore { open, reopen }
            })
        },
        DurabilityTier::Durable,
    )
    .await;
}
