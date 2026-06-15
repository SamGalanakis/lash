use std::collections::BTreeMap;
use std::sync::{Arc, LazyLock};

use lash_core::testing::conformance::{
    ReopenableLashlangArtifactStore, ReopenableProcessRegistry, ReopenableRuntimePersistence,
    ReopenableTriggerStore,
};
use lash_core::{
    DurabilityTier, ProcessExecutionEnvRef, ProcessOriginator, ProcessRegistry, RuntimePersistence,
    SessionScope, SessionStoreFactory, TriggerOccurrenceRequest, TriggerStore,
    TriggerSubscriptionDraft, TriggerSubscriptionFilter,
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

fn trigger_subscription_draft(
    session_id: &str,
    source_key: &str,
    process_name: &str,
) -> TriggerSubscriptionDraft {
    let mut inputs = BTreeMap::new();
    inputs.insert("event".to_string(), lashlang::TriggerInputBinding::Event);
    let registrant_scope = SessionScope::new(session_id);
    TriggerSubscriptionDraft {
        registrant: ProcessOriginator::session(registrant_scope.clone()),
        env_ref: ProcessExecutionEnvRef::new(format!("process-env:{session_id}")),
        wake_target: Some(registrant_scope),
        name: Some(process_name.to_string()),
        source_type: "ui.button.pressed".to_string(),
        source_key: source_key.to_string(),
        source: serde_json::json!({}),
        event_ty: lashlang::TypeExpr::Object(vec![lashlang::TypeField {
            name: "button".into(),
            ty: lashlang::TypeExpr::Str,
            optional: false,
        }]),
        module_ref: lashlang::ModuleRef::new(&lashlang::ContentHash::new("module")),
        host_requirements_ref: lashlang::HostRequirementsRef::new(&lashlang::ContentHash::new(
            "surface",
        )),
        process_ref: lashlang::ProcessRef::new(lashlang::ContentHash::new("process"), 1),
        process_name: process_name.to_string(),
        input_template: lashlang::TriggerInputTemplate::new(inputs),
    }
}

async fn rewrite_postgres_subscription_to_required_surface_ref(
    pool: &sqlx::PgPool,
    subscription_id: &str,
) {
    let record_json: String = sqlx::query_scalar(
        "SELECT record_json FROM lash_trigger_subscriptions WHERE subscription_id = $1",
    )
    .bind(subscription_id)
    .fetch_one(pool)
    .await
    .expect("subscription record json");
    let mut legacy_value: serde_json::Value =
        serde_json::from_str(&record_json).expect("subscription json value");
    let legacy_object = legacy_value
        .as_object_mut()
        .expect("subscription json object");
    let host_requirements_ref = legacy_object
        .remove("host_requirements_ref")
        .expect("host requirements ref");
    legacy_object.insert("required_surface_ref".to_string(), host_requirements_ref);
    sqlx::query(
        "UPDATE lash_trigger_subscriptions SET record_json = $2 WHERE subscription_id = $1",
    )
    .bind(subscription_id)
    .bind(serde_json::to_string(&legacy_value).expect("legacy json text"))
    .execute(pool)
    .await
    .expect("rewrite legacy trigger row");
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

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn postgres_trigger_store_skips_legacy_required_surface_ref_when_configured() {
    let _db_guard = DB_GUARD.lock().await;
    let Some(storage) = storage().await else {
        eprintln!(
            "skipping Postgres trigger malformed-row test: LASH_POSTGRES_DATABASE_URL is not set"
        );
        return;
    };
    reset(&storage).await;
    let store = storage.trigger_store();
    let source_key = lash_core::empty_trigger_source_key("ui.button.pressed").expect("source key");
    let legacy = store
        .register_subscription(trigger_subscription_draft(
            "legacy-session",
            &source_key,
            "legacy_button",
        ))
        .await
        .expect("register legacy");
    let current = store
        .register_subscription(trigger_subscription_draft(
            "current-session",
            &source_key,
            "current_button",
        ))
        .await
        .expect("register current");
    rewrite_postgres_subscription_to_required_surface_ref(storage.pool(), &legacy.subscription_id)
        .await;

    let mut source_filter = TriggerSubscriptionFilter::for_source_type("ui.button.pressed");
    source_filter.source_key = Some(source_key.clone());
    source_filter.enabled = Some(true);
    let listed = store
        .list_subscriptions(source_filter)
        .await
        .expect("list subscriptions");
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].handle, current.handle);

    let occurrence = store
        .record_occurrence(TriggerOccurrenceRequest::new(
            "ui.button.pressed",
            source_key.clone(),
            serde_json::json!({ "button": "Blue" }),
            "button-blue-legacy-postgres-row",
        ))
        .await
        .expect("record occurrence");
    let deliveries = store
        .reserve_matching_deliveries(&occurrence.occurrence_id)
        .await
        .expect("reserve deliveries");
    assert_eq!(deliveries.len(), 1);
    assert_eq!(deliveries[0].subscription.handle, current.handle);

    assert!(
        store
            .cancel_subscription("legacy-session", &legacy.handle)
            .await
            .expect("cancel legacy row")
    );
    let (enabled, record_json): (bool, String) = sqlx::query_as(
        "SELECT enabled, record_json FROM lash_trigger_subscriptions WHERE subscription_id = $1",
    )
    .bind(&legacy.subscription_id)
    .fetch_one(storage.pool())
    .await
    .expect("legacy row after cancel");
    assert!(!enabled);
    assert!(record_json.contains("required_surface_ref"));
    assert!(!record_json.contains("host_requirements_ref"));

    assert_eq!(
        store
            .delete_session_subscriptions("legacy-session")
            .await
            .expect("delete legacy session rows"),
        1
    );
    let legacy_rows: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM lash_trigger_subscriptions WHERE subscription_id = $1",
    )
    .bind(&legacy.subscription_id)
    .fetch_one(storage.pool())
    .await
    .expect("legacy row count");
    assert_eq!(legacy_rows, 0);

    let canonical_rows: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM lash_trigger_subscriptions WHERE subscription_id = $1",
    )
    .bind(&current.subscription_id)
    .fetch_one(storage.pool())
    .await
    .expect("canonical row count");
    assert_eq!(canonical_rows, 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn postgres_lashlang_artifact_store_satisfies_conformance_when_configured() {
    let _db_guard = DB_GUARD.lock().await;
    let Some(storage) = storage().await else {
        eprintln!("skipping Postgres artifact conformance: LASH_POSTGRES_DATABASE_URL is not set");
        return;
    };
    let storage = Arc::new(storage);
    lash_core::testing::conformance::lashlang_artifact_store_reopenable(
        || {
            let storage = Arc::clone(&storage);
            sync_await(async move {
                reset(&storage).await;
                let open = Arc::new(storage.lashlang_artifact_store())
                    as Arc<dyn lashlang::LashlangArtifactStore>;
                let reopen = Arc::new(storage.lashlang_artifact_store())
                    as Arc<dyn lashlang::LashlangArtifactStore>;
                ReopenableLashlangArtifactStore { open, reopen }
            })
        },
        DurabilityTier::Durable,
    )
    .await;
}
