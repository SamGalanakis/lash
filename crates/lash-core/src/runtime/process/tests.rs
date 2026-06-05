use std::collections::BTreeMap;
use std::sync::Arc;

use super::materialization::select_value;
use super::*;

fn registration(id: &str) -> ProcessRegistration {
    ProcessRegistration::new(
        id,
        ProcessInput::External {
            metadata: serde_json::Value::Null,
        },
    )
}

#[test]
fn process_wake_input_from_event_payload_prefers_text_field() {
    let payload = serde_json::json!({
        "text": "ready",
        "value": "ignored"
    });

    assert_eq!(process_wake_input_from_event_payload(&payload), "ready");
}

#[test]
fn process_wake_input_from_event_payload_falls_back_to_value_field() {
    let payload = serde_json::json!({
        "value": { "status": "ready" }
    });

    assert_eq!(
        process_wake_input_from_event_payload(&payload),
        r#"{"status":"ready"}"#
    );
}

#[test]
fn process_wake_input_from_event_payload_renders_malformed_payload_as_json() {
    let payload = serde_json::json!({
        "unexpected": true
    });

    assert_eq!(
        process_wake_input_from_event_payload(&payload),
        r#"{"unexpected":true}"#
    );
}

#[test]
fn process_wake_input_from_event_payload_renders_plain_scalar_payload_as_json() {
    let payload = serde_json::json!(42);

    assert_eq!(process_wake_input_from_event_payload(&payload), "42");
}

#[test]
fn process_wake_turn_text_frames_process_id_sequence_and_input() {
    let wake = wake_delivery("process.ready", None);

    assert_eq!(
        process_wake_turn_text(&wake),
        "Background process wake\nProcess: process-1\nEvent: process.ready #7\nWake input:\nline one\nline two"
    );
}

#[test]
fn process_wake_turn_cause_preserves_process_origin() {
    let process_caused_by = crate::CausalRef::SessionNode {
        session_id: "target".to_string(),
        node_id: "host-event:button".to_string(),
    };
    let wake = wake_delivery("process.ready", Some(process_caused_by.clone()));

    let cause = process_wake_turn_cause(&wake);

    assert_eq!(cause.id, "wake:abc");
    assert_eq!(cause.event_type, "process.ready");
    assert_eq!(
        cause.text,
        "Background process wake\nProcess: process-1\nEvent: process.ready #7\nWake input:\nline one\nline two"
    );
    assert!(matches!(
        cause.origin,
        crate::MessageOrigin::Process {
            process_id,
            event_type,
            sequence,
            wake_id,
            caused_by,
        } if process_id == "process-1"
            && event_type == "process.ready"
            && sequence == 7
            && wake_id.as_deref() == Some("wake:abc")
            && caused_by == Some(process_caused_by)
    ));
}

#[test]
fn process_wake_delivery_carries_event_invocation_and_process_cause() {
    let process_caused_by = crate::CausalRef::SessionNode {
        session_id: "target".to_string(),
        node_id: "host-event:button".to_string(),
    };
    let wake = wake_delivery("process.ready", Some(process_caused_by.clone()));

    assert_eq!(wake.event_type, "process.ready");
    assert_eq!(wake.process_caused_by, Some(process_caused_by));
    assert!(matches!(
        wake.event_invocation.subject,
        crate::RuntimeSubject::ProcessEvent {
            process_id,
            sequence: 7,
            event_type,
        } if process_id == "process-1" && event_type == "process.ready"
    ));
}

fn wake_delivery(
    event_type: impl Into<String>,
    process_caused_by: Option<crate::CausalRef>,
) -> ProcessWakeDelivery {
    let event_type = event_type.into();
    ProcessWakeDelivery {
        wake_id: "wake:abc".to_string(),
        target_session_id: "target".to_string(),
        target_scope_id: ProcessScope::new("target").id(),
        process_id: "process-1".to_string(),
        sequence: 7,
        event_type: event_type.clone(),
        event_invocation: crate::RuntimeInvocation {
            scope: crate::RuntimeScope::new("target"),
            subject: crate::RuntimeSubject::ProcessEvent {
                process_id: "process-1".to_string(),
                sequence: 7,
                event_type,
            },
            caused_by: Some(crate::CausalRef::Process {
                process_id: "process-1".to_string(),
            }),
            replay: None,
        },
        process_caused_by,
        dedupe_key: "process-1:7".to_string(),
        input: "line one\nline two".to_string(),
        created_at_ms: 123,
    }
}

#[test]
fn selector_extracts_payload_pointer_const_template_and_present() {
    let payload = serde_json::json!({
        "line": "done",
        "wake_input": "wake me"
    });

    assert_eq!(
        select_value(&payload, &ProcessValueSelector::Payload).unwrap(),
        payload
    );
    assert_eq!(
        select_value(
            &payload,
            &ProcessValueSelector::Pointer("/line".to_string())
        )
        .unwrap(),
        serde_json::json!("done")
    );
    assert_eq!(
        select_value(
            &payload,
            &ProcessValueSelector::Const(serde_json::json!({"ok": true}))
        )
        .unwrap(),
        serde_json::json!({"ok": true})
    );
    assert_eq!(
        select_value(
            &payload,
            &ProcessValueSelector::Template {
                template: "event: {line}".to_string(),
                fields: BTreeMap::from([(
                    "line".to_string(),
                    ProcessValueSelector::Pointer("/line".to_string())
                )]),
            },
        )
        .unwrap(),
        serde_json::json!("event: done")
    );
    assert_eq!(
        select_value(
            &payload,
            &ProcessValueSelector::Present("/wake_input".to_string())
        )
        .unwrap(),
        serde_json::json!(true)
    );
}

// Contract invariants (registration idempotency, event/wake materialization,
// ack suppression, terminal/await, handle grants, session deletion) live in the
// backend-agnostic conformance suite so the in-memory and Turso registries are
// held to one spec. See `crate::testing::conformance`.
#[tokio::test]
async fn test_local_process_registry_satisfies_conformance() {
    crate::testing::conformance::process_registry(|| {
        Arc::new(TestLocalProcessRegistry::default()) as Arc<dyn ProcessRegistry>
    })
    .await;
}

#[tokio::test]
async fn delete_session_process_command_requests_cancel_only_for_unshared_active_processes() {
    let registry = Arc::new(TestLocalProcessRegistry::default());
    let registry_dyn = Arc::clone(&registry) as Arc<dyn ProcessRegistry>;
    let deleted_scope = ProcessScope::new("deleted");
    let remaining_scope = ProcessScope::new("remaining");
    for process_id in ["sole", "shared"] {
        registry
            .register_process(registration(process_id))
            .await
            .expect("register");
        registry
            .grant_handle(
                &deleted_scope,
                process_id,
                ProcessHandleDescriptor::new(Some("test"), Some(process_id)),
            )
            .await
            .expect("grant deleted");
    }
    registry
        .grant_handle(
            &remaining_scope,
            "shared",
            ProcessHandleDescriptor::new(Some("test"), Some("shared")),
        )
        .await
        .expect("grant remaining");
    let controller = crate::InlineRuntimeEffectController;
    let invocation = crate::RuntimeInvocation::effect(
        crate::RuntimeScope::new("deleted"),
        "process:delete-session:deleted",
        crate::RuntimeEffectKind::Process,
        "deleted:delete-session",
    );

    let outcome = crate::RuntimeEffectController::execute_effect(
        &controller,
        crate::RuntimeEffectEnvelope::new(
            invocation,
            crate::RuntimeEffectCommand::Process {
                command: crate::ProcessCommand::DeleteSession {
                    session_id: "deleted".to_string(),
                },
            },
        ),
        crate::RuntimeEffectLocalExecutor::process_control(registry_dyn),
    )
    .await
    .expect("delete session process command");

    assert!(matches!(
        outcome,
        crate::RuntimeEffectOutcome::Process {
            result: crate::ProcessEffectOutcome::DeleteSession { .. }
        }
    ));
    let sole_events = registry.events_after("sole", 0).await.expect("sole events");
    assert!(
        sole_events
            .iter()
            .any(|event| event.event_type == "process.cancel_requested")
    );
    assert!(
        registry
            .events_after("shared", 0)
            .await
            .expect("shared events")
            .is_empty()
    );
}
