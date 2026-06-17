//! [`TriggerStore`](crate::TriggerStore) conformance: trigger
//! subscriptions and idempotent occurrence reservations.

use super::*;

// ---------------------------------------------------------------------------
// TriggerStore conformance
// ---------------------------------------------------------------------------

/// Run the full [`TriggerStore`](crate::TriggerStore) conformance suite
/// against the backend produced by `make`. `make` must return a fresh, empty
/// store on each call.
pub async fn trigger_store<F>(make: F, expected_tier: DurabilityTier)
where
    F: Fn() -> Arc<dyn crate::TriggerStore>,
{
    trigger_store_reports_declared_tier(make(), expected_tier);
    trigger_source_key_is_stable(make()).await;
    trigger_store_registers_lists_and_cancels(make()).await;
    trigger_store_records_and_reserves_idempotently(make()).await;
}

/// Run the full [`TriggerStore`](crate::TriggerStore) suite plus durable
/// reopen checks.
pub async fn trigger_store_reopenable<F>(make: F, expected_tier: DurabilityTier)
where
    F: Fn() -> ReopenableTriggerStore,
{
    trigger_store(|| make().open, expected_tier).await;
    trigger_store_survives_reopen(make()).await;
}

fn sample_trigger_subscription_draft(
    session_id: &str,
    source_key: &str,
    process_name: &str,
) -> crate::TriggerSubscriptionDraft {
    let mut inputs = BTreeMap::new();
    inputs.insert("event".to_string(), crate::TriggerInputBinding::Event);
    let registrant_scope = crate::SessionScope::new(session_id);
    crate::TriggerSubscriptionDraft {
        registrant: crate::ProcessOriginator::session(registrant_scope.clone()),
        env_ref: crate::ProcessExecutionEnvRef::new(format!("process-env:{session_id}")),
        wake_target: Some(registrant_scope),
        name: Some(process_name.to_string()),
        source_type: "ui.button.pressed".to_string(),
        source_key: source_key.to_string(),
        source: serde_json::json!({}),
        payload_schema: crate::LashSchema::new(serde_json::json!({
            "type": "object",
            "properties": {
                "button": { "type": "string" }
            },
            "required": ["button"],
            "additionalProperties": false
        })),
        target: crate::ProcessInput::Engine {
            kind: "test".to_string(),
            payload: serde_json::json!({ "process": process_name }),
        },
        event_types: Vec::new(),
        input_template: inputs,
        target_label: Some(process_name.to_string()),
    }
}

fn button_occurrence_request(
    source_key: impl Into<String>,
    idempotency_key: impl Into<String>,
) -> crate::TriggerOccurrenceRequest {
    crate::TriggerOccurrenceRequest::new(
        "ui.button.pressed",
        source_key,
        serde_json::json!({ "button": "Blue" }),
        idempotency_key,
    )
    .with_source(serde_json::json!({}))
}

fn trigger_store_reports_declared_tier(
    store: Arc<dyn crate::TriggerStore>,
    expected: DurabilityTier,
) {
    assert_eq!(
        store.durability_tier(),
        expected,
        "durability tier must match the backend"
    );
}

async fn trigger_source_key_is_stable(store: Arc<dyn crate::TriggerStore>) {
    let source = serde_json::json!({ "button": "Blue" });
    let first = store
        .source_key_for_subscription("ui.button.pressed", &source)
        .await
        .expect("first source key");
    let second = store
        .source_key_for_subscription("ui.button.pressed", &source)
        .await
        .expect("second source key");
    assert_eq!(first, second, "source keys must be stable");
    assert!(!first.is_empty(), "source keys must be non-empty");
}

async fn trigger_store_registers_lists_and_cancels(store: Arc<dyn crate::TriggerStore>) {
    let source_key = store
        .source_key_for_subscription("ui.button.pressed", &serde_json::json!({}))
        .await
        .expect("source key");
    let first = store
        .register_subscription(sample_trigger_subscription_draft(
            "session-a",
            &source_key,
            "first",
        ))
        .await
        .expect("register first subscription");
    let second = store
        .register_subscription(sample_trigger_subscription_draft(
            "session-b",
            &source_key,
            "second",
        ))
        .await
        .expect("register second subscription");

    assert!(!first.subscription_id.is_empty());
    assert!(!first.handle.is_empty());
    assert_ne!(first.handle, second.handle);

    let by_session = store
        .list_subscriptions(crate::TriggerSubscriptionFilter::for_session("session-a"))
        .await
        .expect("list by session");
    assert_eq!(by_session.len(), 1);
    assert_eq!(by_session[0].handle, first.handle);

    let mut by_source = crate::TriggerSubscriptionFilter::for_source_type("ui.button.pressed");
    by_source.source_key = Some(source_key.clone());
    by_source.enabled = Some(true);
    let source_matches = store
        .list_subscriptions(by_source)
        .await
        .expect("list by source");
    assert_eq!(source_matches.len(), 2);

    assert!(
        !store
            .cancel_subscription("session-b", &first.handle)
            .await
            .expect("wrong-session cancel"),
        "cancel must be scoped by session"
    );
    assert!(
        store
            .cancel_subscription("session-a", &first.handle)
            .await
            .expect("cancel first")
    );

    let mut disabled_filter = crate::TriggerSubscriptionFilter::for_session("session-a");
    disabled_filter.handle = Some(first.handle.clone());
    let disabled = store
        .list_subscriptions(disabled_filter)
        .await
        .expect("list disabled");
    assert_eq!(disabled.len(), 1);
    assert!(!disabled[0].enabled);
}

async fn trigger_store_records_and_reserves_idempotently(store: Arc<dyn crate::TriggerStore>) {
    let source_key = store
        .source_key_for_subscription("ui.button.pressed", &serde_json::json!({}))
        .await
        .expect("source key");
    let subscription = store
        .register_subscription(sample_trigger_subscription_draft(
            "session-a",
            &source_key,
            "on_button",
        ))
        .await
        .expect("register subscription");

    let occurrence = store
        .record_occurrence(button_occurrence_request(
            source_key.clone(),
            "button-blue-1",
        ))
        .await
        .expect("record occurrence");
    assert!(!occurrence.occurrence_id.is_empty());
    assert_eq!(occurrence.source_type, "ui.button.pressed");
    assert_eq!(occurrence.source_key, source_key);

    let first = store
        .reserve_matching_deliveries(&occurrence.occurrence_id)
        .await
        .expect("reserve first delivery");
    assert_eq!(first.len(), 1);
    assert_eq!(first[0].subscription.handle, subscription.handle);
    assert_eq!(first[0].occurrence.occurrence_id, occurrence.occurrence_id);
    assert_eq!(
        first[0].process_id,
        crate::deterministic_delivery_process_id(
            &occurrence.occurrence_id,
            &subscription.subscription_id
        )
        .expect("deterministic delivery process id")
    );

    let duplicate = store
        .reserve_matching_deliveries(&occurrence.occurrence_id)
        .await
        .expect("reserve duplicate delivery");
    assert!(duplicate.is_empty());

    let replayed = store
        .record_occurrence(button_occurrence_request(
            source_key.clone(),
            "button-blue-1",
        ))
        .await
        .expect("replay occurrence");
    assert_eq!(replayed.occurrence_id, occurrence.occurrence_id);
    let replayed_delivery = store
        .reserve_matching_deliveries(&replayed.occurrence_id)
        .await
        .expect("reserve replayed delivery");
    assert!(replayed_delivery.is_empty());

    assert!(
        store
            .cancel_subscription("session-a", &subscription.handle)
            .await
            .expect("cancel subscription")
    );
    let disabled = store
        .record_occurrence(button_occurrence_request(source_key, "button-blue-2"))
        .await
        .expect("record disabled occurrence");
    let disabled_deliveries = store
        .reserve_matching_deliveries(&disabled.occurrence_id)
        .await
        .expect("reserve disabled occurrence");
    assert!(disabled_deliveries.is_empty());
}

async fn trigger_store_survives_reopen(factory: ReopenableTriggerStore) {
    let source_key = factory
        .open
        .source_key_for_subscription("ui.button.pressed", &serde_json::json!({}))
        .await
        .expect("source key");
    let subscription = factory
        .open
        .register_subscription(sample_trigger_subscription_draft(
            "session-a",
            &source_key,
            "on_button",
        ))
        .await
        .expect("register subscription before reopen");
    let occurrence = factory
        .open
        .record_occurrence(button_occurrence_request(
            source_key.clone(),
            "button-blue-1",
        ))
        .await
        .expect("record occurrence before reopen");
    let first_delivery = factory
        .open
        .reserve_matching_deliveries(&occurrence.occurrence_id)
        .await
        .expect("reserve before reopen");
    assert_eq!(first_delivery.len(), 1);

    let reopened_subscriptions = factory
        .reopen
        .list_subscriptions(crate::TriggerSubscriptionFilter::for_session("session-a"))
        .await
        .expect("list subscriptions after reopen");
    assert_eq!(reopened_subscriptions.len(), 1);
    assert_eq!(reopened_subscriptions[0].handle, subscription.handle);

    let replayed = factory
        .reopen
        .record_occurrence(button_occurrence_request(
            source_key.clone(),
            "button-blue-1",
        ))
        .await
        .expect("replay after reopen");
    assert_eq!(replayed.occurrence_id, occurrence.occurrence_id);
    let replayed_delivery = factory
        .reopen
        .reserve_matching_deliveries(&replayed.occurrence_id)
        .await
        .expect("reserve replay after reopen");
    assert!(replayed_delivery.is_empty());

    let next = factory
        .reopen
        .record_occurrence(button_occurrence_request(source_key, "button-blue-2"))
        .await
        .expect("record new occurrence after reopen");
    let next_delivery = factory
        .reopen
        .reserve_matching_deliveries(&next.occurrence_id)
        .await
        .expect("reserve new occurrence after reopen");
    assert_eq!(next_delivery.len(), 1);
    assert_eq!(next_delivery[0].subscription.handle, subscription.handle);
}
