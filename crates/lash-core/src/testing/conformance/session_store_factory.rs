//! [`SessionStoreFactory`](crate::SessionStoreFactory) conformance: create,
//! reopen, delete, session metadata, and declared durability.

use super::*;

/// Run the [`SessionStoreFactory`](crate::SessionStoreFactory) conformance
/// suite against the backend produced by `make`. `make` must return a fresh,
/// empty factory on each call.
pub async fn session_store_factory<F>(make: F, expected_tier: DurabilityTier)
where
    F: Fn() -> Arc<dyn crate::SessionStoreFactory>,
{
    session_store_factory_reports_declared_tier(make(), expected_tier);
    session_store_factory_open_missing_returns_none(make()).await;
    session_store_factory_create_seeds_and_reopens_meta(make(), expected_tier).await;
    session_store_factory_create_is_idempotent(make()).await;
    session_store_factory_delete_removes_store_and_is_idempotent(make()).await;
}

fn session_store_request(
    session_id: &str,
    model_id: &str,
    relation: crate::SessionRelation,
) -> crate::SessionStoreCreateRequest {
    crate::SessionStoreCreateRequest {
        session_id: session_id.to_string(),
        relation,
        policy: crate::SessionPolicy {
            model: crate::ModelSpec::from_token_limits(model_id, None, 200_000, None)
                .expect("valid conformance model"),
            provider_id: "conformance-provider".to_string(),
            session_id: Some(session_id.to_string()),
            autonomous: false,
            max_turns: None,
            prompt: crate::PromptLayer::new(),
        },
    }
}

fn assert_meta_matches_request(
    meta: &SessionMeta,
    request: &crate::SessionStoreCreateRequest,
    expected_model: &str,
) {
    assert_eq!(meta.session_id, request.session_id);
    assert_eq!(meta.session_name, request.session_id);
    assert_eq!(meta.model, expected_model);
    assert_eq!(meta.relation, request.relation);
    assert!(
        !meta.created_at.is_empty(),
        "created session metadata must carry a timestamp"
    );
}

fn session_store_factory_reports_declared_tier(
    factory: Arc<dyn crate::SessionStoreFactory>,
    expected: DurabilityTier,
) {
    assert_eq!(
        factory.durability_tier(),
        expected,
        "factory durability tier must match the backend"
    );
}

async fn session_store_factory_open_missing_returns_none(
    factory: Arc<dyn crate::SessionStoreFactory>,
) {
    let request = session_store_request(
        "missing-session",
        "missing-model",
        crate::SessionRelation::Root,
    );
    let opened = factory
        .open_existing_store(&request)
        .await
        .expect("open missing session");
    assert!(
        opened.is_none(),
        "open_existing_store must return None for unknown sessions"
    );
}

async fn session_store_factory_create_seeds_and_reopens_meta(
    factory: Arc<dyn crate::SessionStoreFactory>,
    expected_tier: DurabilityTier,
) {
    let relation = crate::SessionRelation::Child {
        parent_session_id: "parent-session".to_string(),
        caused_by: None,
    };
    let request = session_store_request("session-a", "model-a", relation);

    let created = factory
        .create_store(&request)
        .await
        .expect("create session store");
    assert_eq!(
        created.durability_tier(),
        expected_tier,
        "created store durability tier must match the factory"
    );
    let created_meta = created
        .load_session_meta()
        .await
        .expect("load created session meta")
        .expect("created session meta");
    assert_meta_matches_request(&created_meta, &request, "model-a");

    let reopened = factory
        .open_existing_store(&request)
        .await
        .expect("open existing session store")
        .expect("existing session store");
    let reopened_meta = reopened
        .load_session_meta()
        .await
        .expect("load reopened session meta")
        .expect("reopened session meta");
    assert_meta_matches_request(&reopened_meta, &request, "model-a");
}

async fn session_store_factory_create_is_idempotent(factory: Arc<dyn crate::SessionStoreFactory>) {
    let initial = session_store_request(
        "stable-session",
        "initial-model",
        crate::SessionRelation::Root,
    );
    let created = factory
        .create_store(&initial)
        .await
        .expect("create stable session");
    created
        .save_session_meta(SessionMeta {
            session_id: "stable-session".to_string(),
            session_name: "custom-name".to_string(),
            created_at: "custom-created-at".to_string(),
            model: "custom-model".to_string(),
            cwd: Some("/tmp/conformance".to_string()),
            relation: crate::SessionRelation::Child {
                parent_session_id: "custom-parent".to_string(),
                caused_by: None,
            },
        })
        .await
        .expect("write custom meta");

    let changed = session_store_request(
        "stable-session",
        "changed-model",
        crate::SessionRelation::Root,
    );
    let recreated = factory
        .create_store(&changed)
        .await
        .expect("recreate stable session");
    let meta = recreated
        .load_session_meta()
        .await
        .expect("load recreated meta")
        .expect("recreated meta");
    assert_eq!(
        meta.session_name, "custom-name",
        "create_store must not overwrite existing session metadata"
    );
    assert_eq!(meta.model, "custom-model");
    assert_eq!(
        meta.parent_session_id(),
        Some("custom-parent"),
        "create_store must preserve the original relation"
    );
}

async fn session_store_factory_delete_removes_store_and_is_idempotent(
    factory: Arc<dyn crate::SessionStoreFactory>,
) {
    let request = session_store_request(
        "delete-session",
        "delete-model",
        crate::SessionRelation::Root,
    );
    factory
        .create_store(&request)
        .await
        .expect("create deleted session");
    assert!(
        factory
            .open_existing_store(&request)
            .await
            .expect("open before delete")
            .is_some(),
        "session must exist before delete"
    );

    factory
        .delete_session(&request.session_id)
        .await
        .expect("delete session");
    assert!(
        factory
            .open_existing_store(&request)
            .await
            .expect("open after delete")
            .is_none(),
        "delete_session must remove the session store"
    );
    factory
        .delete_session(&request.session_id)
        .await
        .expect("second delete must be idempotent");

    let recreated_request = session_store_request(
        "delete-session",
        "recreated-model",
        crate::SessionRelation::Root,
    );
    let recreated = factory
        .create_store(&recreated_request)
        .await
        .expect("recreate deleted session");
    let meta = recreated
        .load_session_meta()
        .await
        .expect("load recreated session meta")
        .expect("recreated session meta");
    assert_meta_matches_request(&meta, &recreated_request, "recreated-model");
}
