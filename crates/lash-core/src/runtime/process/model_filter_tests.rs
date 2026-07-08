use serde_json::json;

use super::model::{
    ProcessExecutionEnvRef, ProcessIdentity, ProcessInput, ProcessListFilter, ProcessProvenance,
    ProcessRecord, ProcessRegistration, RecoveryDisposition, SessionScope,
};

fn record(process_id: &str, label: &str, created_at_ms: u64) -> ProcessRecord {
    let mut record = ProcessRecord::from_registration(
        ProcessRegistration::new(
            process_id,
            ProcessInput::Engine {
                kind: "test-engine".to_string(),
                payload: json!({}),
            },
            RecoveryDisposition::Rerunnable,
            ProcessProvenance::host(),
        )
        .with_identity(ProcessIdentity::new("test-engine").with_label(Some(label)))
        .with_execution_env_ref(Some(ProcessExecutionEnvRef::new(format!(
            "process-env:test:{process_id}"
        )))),
    );
    record.created_at_ms = created_at_ms;
    record
}

#[test]
fn process_list_filter_matches_enriched_facets() {
    let mut matching = record("matching", "target", 100);
    matching.provenance = ProcessProvenance::session(SessionScope::new("origin-session"))
        .with_caused_by(Some(crate::CausalRef::TriggerOccurrence {
            occurrence_id: "occurrence-target".to_string(),
        }));
    let wrong = record("wrong", "other", 200);

    let filter = ProcessListFilter::decode(&json!({
        "originator_scope_id": "session:origin-session",
        "identity_kind": "test-engine",
        "identity_label": "target",
        "caused_by_occurrence_id": "occurrence-target",
        "created_at_start_ms": 100,
        "created_at_end_ms": 101
    }))
    .expect("decode enriched filter");
    assert!(filter.matches_record(&matching));
    assert!(!filter.matches_record(&wrong));

    let subscription_filter = ProcessListFilter::decode(&json!({
        "caused_by_subscription_id": "subscription-target"
    }))
    .expect("decode subscription filter");
    assert!(!subscription_filter.matches_record(&matching));
    assert!(
        ProcessListFilter::decode(&json!({ "identity_kind": true }))
            .expect_err("invalid identity kind")
            .contains("must be a string")
    );
    assert!(
        ProcessListFilter::decode(&json!({ "created_at_start_ms": "old" }))
            .expect_err("invalid created-at start")
            .contains("must be an integer")
    );
}
