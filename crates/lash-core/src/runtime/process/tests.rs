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

fn wake_event_type() -> ProcessEventType {
    ProcessEventType {
        name: "producer.wake".to_string(),
        payload_schema: crate::LashSchema::any(),
        semantics: ProcessEventSemanticsSpec {
            wake: Some(ProcessWakeSpec {
                when: Some(ProcessValueSelector::Present("/wake_input".to_string())),
                input: ProcessValueSelector::Pointer("/wake_input".to_string()),
                dedupe_key: ProcessWakeDedupeKey::EventIdentity,
            }),
            ..ProcessEventSemanticsSpec::default()
        },
    }
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
    let wake = ProcessWakeDelivery {
        wake_id: "wake:abc".to_string(),
        target_session_id: "target".to_string(),
        target_scope_id: ProcessScope::new("target").id(),
        process_id: "process-1".to_string(),
        sequence: 7,
        dedupe_key: "process-1:7".to_string(),
        input: "line one\nline two".to_string(),
        created_at_ms: 123,
    };

    assert_eq!(
        process_wake_turn_text(&wake),
        "Background process wake\nProcess: process-1\nEvent: process.wake #7\nWake input:\nline one\nline two"
    );
}

#[test]
fn process_wake_turn_cause_preserves_process_origin() {
    let wake = ProcessWakeDelivery {
        wake_id: "wake:abc".to_string(),
        target_session_id: "target".to_string(),
        target_scope_id: ProcessScope::new("target").id(),
        process_id: "process-1".to_string(),
        sequence: 7,
        dedupe_key: "process-1:7".to_string(),
        input: "line one\nline two".to_string(),
        created_at_ms: 123,
    };

    let cause = process_wake_turn_cause(&wake);

    assert_eq!(cause.id, "wake:abc");
    assert_eq!(cause.event_type, "process.wake");
    assert_eq!(
        cause.text,
        "Background process wake\nProcess: process-1\nEvent: process.wake #7\nWake input:\nline one\nline two"
    );
    assert!(matches!(
        cause.origin,
        crate::MessageOrigin::Process {
            process_id,
            event_type,
            sequence,
            wake_id,
        } if process_id == "process-1"
            && event_type == "process.wake"
            && sequence == 7
            && wake_id.as_deref() == Some("wake:abc")
    ));
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

#[tokio::test]
async fn process_registry_validates_custom_events_and_materializes_wakes() {
    let registry = TestLocalProcessRegistry::default();
    let target_scope = ProcessScope::new("s1");
    let mut properties = serde_json::Map::new();
    properties.insert("line".to_string(), serde_json::json!({ "type": "string" }));
    properties.insert(
        "wake_input".to_string(),
        serde_json::json!({ "type": "string" }),
    );
    let event_type = ProcessEventType {
        name: "producer.line".to_string(),
        payload_schema: crate::LashSchema::object(properties, vec!["line".to_string()]),
        semantics: ProcessEventSemanticsSpec {
            wake: Some(ProcessWakeSpec {
                when: Some(ProcessValueSelector::Present("/wake_input".to_string())),
                input: ProcessValueSelector::Pointer("/wake_input".to_string()),
                dedupe_key: ProcessWakeDedupeKey::EventIdentity,
            }),
            ..ProcessEventSemanticsSpec::default()
        },
    };
    registry
        .register_process(registration("proc-1").with_extra_event_types([event_type]))
        .await
        .expect("register");

    let event = registry
        .append_event(
            "proc-1",
            ProcessEventAppendRequest::new(
                "producer.line",
                serde_json::json!({
                    "line": "deploy failed",
                    "wake_input": "Process event: deploy failed"
                }),
            )
            .with_wake_target_scope(target_scope),
        )
        .await
        .expect("append");

    assert_eq!(event.event.sequence, 1);
    assert_eq!(
        event
            .event
            .semantics
            .wake
            .as_ref()
            .map(|wake| wake.input.as_str()),
        Some("Process event: deploy failed")
    );
    assert_eq!(
        registry
            .wake_events_after("proc-1", 0)
            .await
            .expect("wake events")
            .len(),
        1
    );
    registry
        .ack_wake("proc-1", event.event.sequence)
        .await
        .expect("ack wake");
    assert!(
        registry
            .wake_events_after("proc-1", 0)
            .await
            .expect("wake events")
            .is_empty()
    );
    assert!(
        registry
            .append_event(
                "proc-1",
                ProcessEventAppendRequest::new(
                    "producer.line",
                    serde_json::json!({ "wake_input": "missing required line" }),
                ),
            )
            .await
            .is_err()
    );
}

#[tokio::test]
async fn await_process_reads_terminal_event_materialized_output() {
    let registry = TestLocalProcessRegistry::default();
    registry
        .register_process(registration("proc-2"))
        .await
        .expect("register");
    registry
        .complete_process(
            "proc-2",
            ProcessAwaitOutput::Success {
                value: serde_json::json!({ "ok": true }),
                control: None,
            },
        )
        .await
        .expect("complete");

    assert_eq!(
        registry.await_process("proc-2").await.expect("await"),
        ProcessAwaitOutput::Success {
            value: serde_json::json!({ "ok": true }),
            control: None,
        }
    );
    assert!(
        registry
            .get_process("proc-2")
            .await
            .expect("record")
            .is_terminal()
    );
}

#[tokio::test]
async fn transfer_handle_grants_moves_addressability_without_process_events() {
    let registry = TestLocalProcessRegistry::default();
    let s1 = ProcessScope::new("s1");
    let s2 = ProcessScope::new("s2");
    registry
        .register_process(registration("proc-3"))
        .await
        .expect("register");
    registry
        .grant_handle(
            &s1,
            "proc-3",
            ProcessHandleDescriptor::new(Some("tool"), Some("demo")),
        )
        .await
        .expect("grant");
    registry
        .transfer_handle_grants(&s1, &s2, &["proc-3".to_string()])
        .await
        .expect("transfer");

    assert_eq!(
        registry
            .list_handle_grants(&s1)
            .await
            .expect("grants")
            .len(),
        0
    );
    assert_eq!(
        registry
            .list_handle_grants(&s2)
            .await
            .expect("grants")
            .len(),
        1
    );
    assert!(
        registry
            .events_after("proc-3", 0)
            .await
            .expect("events")
            .is_empty()
    );
}

#[tokio::test]
async fn multiple_sessions_can_hold_grants_to_one_process() {
    let registry = TestLocalProcessRegistry::default();
    let s1 = ProcessScope::new("s1");
    let s2 = ProcessScope::new("s2");
    let s3 = ProcessScope::new("s3");
    registry
        .register_process(registration("proc-5"))
        .await
        .expect("register");
    registry
        .grant_handle(
            &s1,
            "proc-5",
            ProcessHandleDescriptor::new(Some("tool"), Some("demo")),
        )
        .await
        .expect("grant s1");
    registry
        .grant_handle(
            &s2,
            "proc-5",
            ProcessHandleDescriptor::new(Some("worker"), Some("demo")),
        )
        .await
        .expect("grant s2");

    let grant_sessions = registry
        .handle_grants_for_process("proc-5")
        .await
        .expect("process grants")
        .into_iter()
        .map(|grant| grant.session_id)
        .collect::<Vec<_>>();
    assert_eq!(grant_sessions, vec!["s1".to_string(), "s2".to_string()]);

    registry
        .transfer_handle_grants(&s1, &s3, &["proc-5".to_string()])
        .await
        .expect("transfer s1");
    let grant_sessions = registry
        .handle_grants_for_process("proc-5")
        .await
        .expect("process grants")
        .into_iter()
        .map(|grant| grant.session_id)
        .collect::<Vec<_>>();
    assert_eq!(grant_sessions, vec!["s2".to_string(), "s3".to_string()]);
    assert!(
        registry
            .events_after("proc-5", 0)
            .await
            .expect("events")
            .is_empty()
    );
}

#[tokio::test]
async fn processes_can_exist_with_zero_grants() {
    let registry = TestLocalProcessRegistry::default();
    let s1 = ProcessScope::new("s1");
    registry
        .register_process(registration("proc-4"))
        .await
        .expect("register");
    assert!(
        registry
            .list_handle_grants(&s1)
            .await
            .expect("grants")
            .is_empty()
    );
}

#[tokio::test]
async fn delete_session_process_state_revokes_handles_by_session_id() {
    let registry = TestLocalProcessRegistry::default();
    let deleted_scope = ProcessScope::new("deleted");
    let remaining_scope = ProcessScope::new("remaining");
    for process_id in ["sole", "shared", "terminal"] {
        registry
            .register_process(registration(process_id).with_extra_event_types([wake_event_type()]))
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
    registry
        .complete_process(
            "terminal",
            ProcessAwaitOutput::Success {
                value: serde_json::Value::Null,
                control: None,
            },
        )
        .await
        .expect("complete terminal");
    registry
        .append_event(
            "sole",
            ProcessEventAppendRequest::new(
                "producer.wake",
                serde_json::json!({ "wake_input": "wake deleted" }),
            )
            .with_wake_target_scope(deleted_scope.clone()),
        )
        .await
        .expect("append wake");

    let report = registry
        .delete_session_process_state("deleted")
        .await
        .expect("delete session process state");

    assert_eq!(report.revoked_handle_count, 3);
    assert_eq!(report.deleted_wake_count, 0);
    assert_eq!(report.cancel_process_ids, vec!["sole".to_string()]);
    assert_eq!(report.preserved_process_ids, vec!["shared".to_string()]);
    assert!(
        registry
            .list_handle_grants(&deleted_scope)
            .await
            .expect("deleted grants")
            .is_empty()
    );
    assert_eq!(
        registry
            .list_handle_grants(&remaining_scope)
            .await
            .expect("remaining grants")
            .len(),
        1
    );
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
    let controller = crate::InlineRuntimeEffectController::default();
    let invocation = crate::RuntimeInvocation::effect(
        crate::RuntimeScope::new("deleted"),
        "process:delete-session:deleted",
        crate::RuntimeEffectKind::Process,
        "deleted:delete-session",
        None,
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
