use std::sync::{Arc, LazyLock};

use lash_core::testing::conformance::{
    ReopenableProcessRegistry, ReopenableRuntimePersistence, ReopenableTriggerStore,
};
use lash_core::{
    DurabilityTier, PluginOptions, ProcessExecutionEnvSpec, ProcessExecutionEnvStore,
    ProcessRegistry, RuntimePersistence, SessionPolicy, SessionStoreFactory, TriggerStore,
};
use lash_postgres_store::PostgresStorage;

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
    sqlx::query(
        r#"
        TRUNCATE
            lash_trigger_deliveries,
            lash_trigger_occurrences,
            lash_trigger_subscriptions,
            lash_process_wake_acks,
            lash_process_handle_grants,
            lash_process_leases,
            lash_process_events,
            lash_processes,
            lash_queued_work_items,
            lash_queued_work_batches,
            lash_runtime_turn_commits,
            lash_session_execution_leases,
            lash_session_meta,
            lash_usage_deltas,
            lash_graph_nodes,
            lash_sessions,
            lash_attachment_manifest,
            lash_lashlang_artifacts,
            lash_blobs
        RESTART IDENTITY CASCADE
        "#,
    )
    .execute(pool)
    .await
    .expect("reset postgres conformance tables");
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
async fn postgres_process_execution_env_store_round_trips_when_configured() {
    let _db_guard = DB_GUARD.lock().await;
    let Some(storage) = storage().await else {
        eprintln!(
            "skipping Postgres process-env conformance: LASH_POSTGRES_DATABASE_URL is not set"
        );
        return;
    };
    reset(&storage).await;

    let spec = ProcessExecutionEnvSpec::new(PluginOptions::default(), SessionPolicy::default());
    let env_ref = spec.stable_ref().expect("stable env ref");
    let bytes = spec.to_store_bytes().expect("encode env spec");
    let store = Arc::new(storage.process_env_store()) as Arc<dyn ProcessExecutionEnvStore>;
    assert_eq!(store.durability_tier(), DurabilityTier::Durable);
    store
        .put_process_execution_env(&env_ref, &bytes)
        .await
        .expect("put env");
    assert_eq!(
        store
            .get_process_execution_env(&env_ref)
            .await
            .expect("get env"),
        Some(bytes.clone())
    );

    let reopened = Arc::new(storage.process_env_store()) as Arc<dyn ProcessExecutionEnvStore>;
    assert_eq!(
        reopened
            .get_process_execution_env(&env_ref)
            .await
            .expect("get reopened env"),
        Some(bytes)
    );
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
