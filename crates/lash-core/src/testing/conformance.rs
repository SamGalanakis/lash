//! Backend-agnostic conformance suites for durable-backend traits.
//!
//! Each suite is parameterized over a factory that produces a *fresh* backend
//! instance and asserts the trait's contract invariants. Run the same suite
//! against every implementation (the production backend and any in-memory test
//! double) so the contract has one executable source of truth and the doubles
//! can't drift from production behavior.
//!
//! Suites panic on the first violated invariant — call them from a
//! `#[tokio::test]`. Embedders with custom backends can run them via
//! `lash::testing::conformance`.

use std::sync::Arc;

use crate::{
    AgentFrameReason, AgentFrameRecord, AttachmentId, AttachmentIntent, CausalRef, DeliveryPolicy,
    MergeKey, ModelSpec, PluginSessionSnapshot, ProtocolEvent, ProtocolTurnOptions,
    QueuedWorkBatch, QueuedWorkBatchDraft, QueuedWorkClaimBoundary, QueuedWorkPayload,
    RUNTIME_EFFECT_JOURNAL_SCHEMA_VERSION, RuntimeCommit, RuntimeEffectJournalRecord,
    RuntimeEffectKind, RuntimeEffectOutcome, RuntimePersistence, RuntimeScope, RuntimeSessionState,
    RuntimeSubject, RuntimeTurnCompletion, SessionMeta, SessionNodePayload, SessionNodeRecord,
    SessionPolicy, SessionReadScope, SessionRelation, SlotPolicy, StoreError, TokenLedgerEntry,
    TokenUsage, ToolState, TurnInput,
};
use crate::{
    LashSchema, ProcessAwaitOutput, ProcessEventAppendRequest, ProcessEventSemanticsSpec,
    ProcessEventType, ProcessHandleDescriptor, ProcessInput, ProcessLeaseCompletion,
    ProcessProvenance, ProcessRegistration, ProcessRegistry, ProcessScope, ProcessTerminalState,
    ProcessValueSelector, ProcessWakeDedupeKey, ProcessWakeSpec,
};

/// A pair of [`ProcessRegistry`] handles opened against the same durable
/// backing store.
pub struct ReopenableProcessRegistry {
    pub open: Arc<dyn ProcessRegistry>,
    pub reopen: Arc<dyn ProcessRegistry>,
}

/// A pair of [`RuntimePersistence`] handles opened against the same durable
/// backing store.
pub struct ReopenableRuntimePersistence {
    pub open: Arc<dyn RuntimePersistence>,
    pub reopen: Arc<dyn RuntimePersistence>,
}

/// A pair of [`AttachmentStore`](crate::AttachmentStore) handles opened against
/// the same durable backing store.
pub struct ReopenableAttachmentStore {
    pub open: Arc<dyn crate::AttachmentStore>,
    pub reopen: Arc<dyn crate::AttachmentStore>,
}

/// A pair of [`LashlangArtifactStore`] handles opened against the same durable
/// backing store.
pub struct ReopenableLashlangArtifactStore {
    pub open: Arc<dyn crate::LashlangArtifactStore>,
    pub reopen: Arc<dyn crate::LashlangArtifactStore>,
}

/// Run the full [`ProcessRegistry`] conformance suite against the backend
/// produced by `make`. `make` must return a fresh, empty registry on each call.
pub async fn process_registry<F>(make: F)
where
    F: Fn() -> Arc<dyn ProcessRegistry>,
{
    registration_is_idempotent_and_hash_conflicts_fail(make()).await;
    validates_custom_events_and_materializes_wakes(make()).await;
    custom_wake_events_preserve_typed_provenance_and_replay(make()).await;
    event_streams_filter_order_and_wait_without_leaking_old_events(make()).await;
    wake_semantics_matrix_materializes_declared_wakes(make()).await;
    keyed_events_materialize_idempotent_wakes(make()).await;
    terminal_and_cancel_events_require_keys(make()).await;
    await_reads_terminal_materialized_output(make()).await;
    transfer_handle_grants_moves_addressability(make()).await;
    multiple_sessions_can_hold_grants(make()).await;
    processes_can_exist_with_zero_grants(make()).await;
    delete_session_revokes_handles_by_session(make()).await;
    list_non_terminal_excludes_terminal_processes(make()).await;
    list_live_handle_grants_excludes_terminal_history(make()).await;
    active_process_lease_fences_competing_owner(make()).await;
    superseded_process_lease_cannot_renew(make()).await;
    renewed_process_lease_survives_original_expiry(make()).await;
    completed_lease_releases_and_reclaim_bumps_fencing(make()).await;
    stale_lease_completion_cannot_release_live_lease(make()).await;
}

/// Run the full [`ProcessRegistry`] suite plus durable reopen checks.
pub async fn process_registry_reopenable<F>(make: F)
where
    F: Fn() -> ReopenableProcessRegistry,
{
    process_registry(|| make().open).await;
    process_registry_survives_reopen(make()).await;
}

fn registration(id: &str) -> ProcessRegistration {
    ProcessRegistration::new(
        id,
        ProcessInput::External {
            metadata: serde_json::Value::Null,
        },
    )
}

fn wake_event_type(name: &str) -> ProcessEventType {
    ProcessEventType {
        name: name.to_string(),
        payload_schema: LashSchema::any(),
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

fn wake_event_type_with(name: &str, wake: ProcessWakeSpec) -> ProcessEventType {
    ProcessEventType {
        name: name.to_string(),
        payload_schema: LashSchema::any(),
        semantics: ProcessEventSemanticsSpec {
            wake: Some(wake),
            ..ProcessEventSemanticsSpec::default()
        },
    }
}

fn plain_event_type(name: &str) -> ProcessEventType {
    ProcessEventType {
        name: name.to_string(),
        payload_schema: LashSchema::any(),
        semantics: ProcessEventSemanticsSpec::default(),
    }
}

async fn registration_is_idempotent_and_hash_conflicts_fail(registry: Arc<dyn ProcessRegistry>) {
    let first = registry
        .register_process(registration("proc-idempotent"))
        .await
        .expect("first register");
    let second = registry
        .register_process(registration("proc-idempotent"))
        .await
        .expect("replay register");
    assert_eq!(
        first.registration_hash, second.registration_hash,
        "identical registration must be idempotent"
    );
    assert!(
        registry
            .register_process(
                registration("proc-idempotent")
                    .with_extra_event_types([wake_event_type("producer.wake")]),
            )
            .await
            .is_err(),
        "a different registration under the same id must fail with a hash conflict"
    );
}

async fn validates_custom_events_and_materializes_wakes(registry: Arc<dyn ProcessRegistry>) {
    let target_scope = ProcessScope::new("s1");
    let mut properties = serde_json::Map::new();
    properties.insert("line".to_string(), serde_json::json!({ "type": "string" }));
    properties.insert(
        "wake_input".to_string(),
        serde_json::json!({ "type": "string" }),
    );
    let event_type = ProcessEventType {
        name: "producer.line".to_string(),
        payload_schema: LashSchema::object(properties, vec!["line".to_string()]),
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

    assert_eq!(event.event.sequence, 1, "first event is sequence 1");
    assert_eq!(
        event
            .event
            .semantics
            .wake
            .as_ref()
            .map(|wake| wake.input.as_str()),
        Some("Process event: deploy failed"),
        "wake input materialized from the declared selector"
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
            .is_empty(),
        "ack_wake must suppress the acked wake from wake_events_after"
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
            .is_err(),
        "payload missing a required field must be rejected"
    );
}

async fn custom_wake_events_preserve_typed_provenance_and_replay(
    registry: Arc<dyn ProcessRegistry>,
) {
    let target_scope = ProcessScope::for_agent_frame("target-session", "target-frame");
    let target_scope_id = target_scope.id();
    let process_caused_by = CausalRef::SessionNode {
        session_id: "target-session".to_string(),
        node_id: "host-event:button".to_string(),
    };
    let event_type = wake_event_type_with(
        "producer.custom_wake",
        ProcessWakeSpec {
            when: Some(ProcessValueSelector::Present("/wake_input".to_string())),
            input: ProcessValueSelector::Pointer("/wake_input".to_string()),
            dedupe_key: ProcessWakeDedupeKey::EventIdentity,
        },
    );
    registry
        .register_process(
            registration("proc-provenance")
                .with_extra_event_types([event_type])
                .with_process_provenance(
                    ProcessProvenance::new(ProcessScope::new("owner-session"), "host-profile")
                        .with_caused_by(Some(process_caused_by.clone())),
                ),
        )
        .await
        .expect("register");

    let request = ProcessEventAppendRequest::new(
        "producer.custom_wake",
        serde_json::json!({
            "line": "build failed",
            "wake_input": "custom wake: build failed",
        }),
    )
    .with_replay_key("custom-wake:build-failed")
    .with_wake_target_scope(target_scope);
    let first = registry
        .append_event("proc-provenance", request.clone())
        .await
        .expect("append");
    let replay = registry
        .append_event("proc-provenance", request)
        .await
        .expect("replay append");

    assert_eq!(first.event.sequence, 1);
    assert_eq!(replay.event.sequence, first.event.sequence);
    assert_eq!(
        registry
            .events_after("proc-provenance", 0)
            .await
            .expect("events")
            .len(),
        1,
        "a replayed custom wake event must not append a second event row"
    );
    assert_eq!(
        first.event.invocation.scope,
        RuntimeScope::new("owner-session")
    );
    assert!(matches!(
        &first.event.invocation.subject,
        RuntimeSubject::ProcessEvent {
            process_id,
            sequence: 1,
            event_type,
        } if process_id == "proc-provenance" && event_type == "producer.custom_wake"
    ));
    assert_eq!(
        first.event.invocation.caused_by,
        Some(CausalRef::Process {
            process_id: "proc-provenance".to_string()
        })
    );
    assert_eq!(
        first
            .event
            .invocation
            .replay
            .as_ref()
            .map(|replay| replay.key.as_str()),
        Some("custom-wake:build-failed")
    );

    let wake = first.wake_delivery.expect("wake delivery");
    assert_eq!(wake.event_type, "producer.custom_wake");
    assert_eq!(wake.event_invocation, first.event.invocation);
    assert_eq!(wake.process_caused_by, Some(process_caused_by));
    assert_eq!(wake.target_session_id, "target-session");
    assert_eq!(wake.target_scope_id, target_scope_id);
    assert_eq!(wake.process_id, "proc-provenance");
    assert_eq!(wake.sequence, first.event.sequence);
    assert_eq!(wake.dedupe_key, "proc-provenance:1");
    assert_eq!(wake.input, "custom wake: build failed");
    assert_eq!(
        replay
            .wake_delivery
            .expect("replayed wake delivery")
            .wake_id,
        wake.wake_id,
        "replaying a wake event must re-materialize the same wake identity"
    );
}

async fn event_streams_filter_order_and_wait_without_leaking_old_events(
    registry: Arc<dyn ProcessRegistry>,
) {
    registry
        .register_process(registration("proc-stream").with_extra_event_types([
            plain_event_type("producer.line"),
            wake_event_type("producer.wake"),
            plain_event_type("producer.future"),
        ]))
        .await
        .expect("register");
    registry
        .append_event(
            "proc-stream",
            ProcessEventAppendRequest::new("producer.line", serde_json::json!({"line": "one"})),
        )
        .await
        .expect("append line one");
    registry
        .append_event(
            "proc-stream",
            ProcessEventAppendRequest::new(
                "producer.wake",
                serde_json::json!({"wake_input": "wake two"}),
            )
            .with_wake_target_scope(ProcessScope::new("root")),
        )
        .await
        .expect("append wake");
    registry
        .append_event(
            "proc-stream",
            ProcessEventAppendRequest::new("producer.line", serde_json::json!({"line": "three"})),
        )
        .await
        .expect("append line three");

    let after_one = registry
        .events_after("proc-stream", 1)
        .await
        .expect("events after one");
    assert_eq!(
        after_one
            .iter()
            .map(|event| (event.sequence, event.event_type.as_str()))
            .collect::<Vec<_>>(),
        vec![(2, "producer.wake"), (3, "producer.line")],
        "events_after must preserve sequence order and exclude older events"
    );
    assert!(
        registry
            .events_after("proc-stream", 3)
            .await
            .expect("events after three")
            .is_empty(),
        "events_after must not leak events at or before the cursor"
    );
    let wake_after_one = registry
        .wake_events_after("proc-stream", 1)
        .await
        .expect("wake events after one");
    assert_eq!(
        wake_after_one
            .iter()
            .map(|event| (event.sequence, event.event_type.as_str()))
            .collect::<Vec<_>>(),
        vec![(2, "producer.wake")],
        "wake_events_after must filter to unacked wake events after the cursor"
    );
    assert!(
        registry
            .wake_events_after("proc-stream", 2)
            .await
            .expect("wake events after wake")
            .is_empty(),
        "wake_events_after must not return the cursor event itself"
    );
    let immediate = registry
        .wait_event_after("proc-stream", "producer.line", 1)
        .await
        .expect("immediate wait");
    assert_eq!(
        immediate.sequence, 3,
        "wait_event_after must return an existing matching event immediately"
    );

    let waiter_registry = Arc::clone(&registry);
    let waiter = tokio::spawn(async move {
        waiter_registry
            .wait_event_after("proc-stream", "producer.future", 3)
            .await
            .expect("future wait")
    });
    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    registry
        .append_event(
            "proc-stream",
            ProcessEventAppendRequest::new("producer.future", serde_json::json!({"line": "four"})),
        )
        .await
        .expect("append future event");
    let future = tokio::time::timeout(std::time::Duration::from_secs(1), waiter)
        .await
        .expect("future wait timeout")
        .expect("future waiter task");
    assert_eq!(future.sequence, 4);
}

async fn wake_semantics_matrix_materializes_declared_wakes(registry: Arc<dyn ProcessRegistry>) {
    registry
        .register_process(
            registration("proc-wake-matrix").with_extra_event_types([
                wake_event_type_with(
                    "matrix.when_false",
                    ProcessWakeSpec {
                        when: Some(ProcessValueSelector::Const(serde_json::json!(false))),
                        input: ProcessValueSelector::Const(serde_json::json!("must not wake")),
                        dedupe_key: ProcessWakeDedupeKey::EventIdentity,
                    },
                ),
                wake_event_type_with(
                    "matrix.payload",
                    ProcessWakeSpec {
                        when: None,
                        input: ProcessValueSelector::Payload,
                        dedupe_key: ProcessWakeDedupeKey::EventIdentity,
                    },
                ),
                wake_event_type_with(
                    "matrix.const_input",
                    ProcessWakeSpec {
                        when: None,
                        input: ProcessValueSelector::Const(serde_json::json!(
                            "constant wake input"
                        )),
                        dedupe_key: ProcessWakeDedupeKey::EventIdentity,
                    },
                ),
                wake_event_type_with(
                    "matrix.template",
                    ProcessWakeSpec {
                        when: None,
                        input: ProcessValueSelector::Template {
                            template: "line {line} #{n}".to_string(),
                            fields: [
                                (
                                    "line".to_string(),
                                    ProcessValueSelector::Pointer("/line".to_string()),
                                ),
                                (
                                    "n".to_string(),
                                    ProcessValueSelector::Pointer("/n".to_string()),
                                ),
                            ]
                            .into_iter()
                            .collect(),
                        },
                        dedupe_key: ProcessWakeDedupeKey::EventIdentity,
                    },
                ),
                wake_event_type_with(
                    "matrix.selector_dedupe",
                    ProcessWakeSpec {
                        when: None,
                        input: ProcessValueSelector::Pointer("/wake_input".to_string()),
                        dedupe_key: ProcessWakeDedupeKey::Selector(ProcessValueSelector::Pointer(
                            "/dedupe".to_string(),
                        )),
                    },
                ),
                wake_event_type_with(
                    "matrix.const_dedupe",
                    ProcessWakeSpec {
                        when: None,
                        input: ProcessValueSelector::Pointer("/wake_input".to_string()),
                        dedupe_key: ProcessWakeDedupeKey::Const("constant-dedupe".to_string()),
                    },
                ),
            ]),
        )
        .await
        .expect("register");
    let target = ProcessScope::new("root");

    let no_wake = registry
        .append_event(
            "proc-wake-matrix",
            ProcessEventAppendRequest::new("matrix.when_false", serde_json::json!({}))
                .with_wake_target_scope(target.clone()),
        )
        .await
        .expect("append when false");
    assert!(
        no_wake.wake_delivery.is_none(),
        "a false wake.when selector must suppress wake materialization"
    );
    let payload = registry
        .append_event(
            "proc-wake-matrix",
            ProcessEventAppendRequest::new("matrix.payload", serde_json::json!("payload wake"))
                .with_wake_target_scope(target.clone()),
        )
        .await
        .expect("append payload wake")
        .wake_delivery
        .expect("payload wake");
    assert_eq!(payload.input, "payload wake");
    assert_eq!(payload.dedupe_key, "proc-wake-matrix:2");
    let const_input = registry
        .append_event(
            "proc-wake-matrix",
            ProcessEventAppendRequest::new("matrix.const_input", serde_json::json!({}))
                .with_wake_target_scope(target.clone()),
        )
        .await
        .expect("append const wake")
        .wake_delivery
        .expect("const wake");
    assert_eq!(const_input.input, "constant wake input");
    let template = registry
        .append_event(
            "proc-wake-matrix",
            ProcessEventAppendRequest::new(
                "matrix.template",
                serde_json::json!({"line": "done", "n": 7}),
            )
            .with_wake_target_scope(target.clone()),
        )
        .await
        .expect("append template wake")
        .wake_delivery
        .expect("template wake");
    assert_eq!(template.input, "line done #7");
    let selector_first = registry
        .append_event(
            "proc-wake-matrix",
            ProcessEventAppendRequest::new(
                "matrix.selector_dedupe",
                serde_json::json!({"wake_input": "selector one", "dedupe": "group-a"}),
            )
            .with_wake_target_scope(target.clone()),
        )
        .await
        .expect("append selector wake one")
        .wake_delivery
        .expect("selector wake one");
    let selector_second = registry
        .append_event(
            "proc-wake-matrix",
            ProcessEventAppendRequest::new(
                "matrix.selector_dedupe",
                serde_json::json!({"wake_input": "selector two", "dedupe": "group-a"}),
            )
            .with_wake_target_scope(target.clone()),
        )
        .await
        .expect("append selector wake two")
        .wake_delivery
        .expect("selector wake two");
    assert_eq!(selector_first.dedupe_key, "group-a");
    assert_eq!(
        selector_first.wake_id, selector_second.wake_id,
        "selector dedupe must produce a stable wake id for the same target and selector value"
    );
    let const_dedupe_first = registry
        .append_event(
            "proc-wake-matrix",
            ProcessEventAppendRequest::new(
                "matrix.const_dedupe",
                serde_json::json!({"wake_input": "const one"}),
            )
            .with_wake_target_scope(target.clone()),
        )
        .await
        .expect("append const dedupe one")
        .wake_delivery
        .expect("const dedupe one");
    let const_dedupe_second = registry
        .append_event(
            "proc-wake-matrix",
            ProcessEventAppendRequest::new(
                "matrix.const_dedupe",
                serde_json::json!({"wake_input": "const two"}),
            )
            .with_wake_target_scope(target),
        )
        .await
        .expect("append const dedupe two")
        .wake_delivery
        .expect("const dedupe two");
    assert_eq!(const_dedupe_first.dedupe_key, "constant-dedupe");
    assert_eq!(
        const_dedupe_first.wake_id, const_dedupe_second.wake_id,
        "const dedupe must produce a stable wake id for the same target"
    );
    let wake_sequences = registry
        .wake_events_after("proc-wake-matrix", 0)
        .await
        .expect("wake events")
        .into_iter()
        .map(|event| event.sequence)
        .collect::<Vec<_>>();
    assert_eq!(
        wake_sequences,
        vec![2, 3, 4, 5, 6, 7, 8],
        "wake_events_after must include only events whose wake semantics materialized"
    );
}

async fn process_registry_survives_reopen(factory: ReopenableProcessRegistry) {
    let scope = ProcessScope::new("reopen-session");
    factory
        .open
        .register_process(
            registration("proc-reopen")
                .with_extra_event_types([wake_event_type("producer.reopen_wake")]),
        )
        .await
        .expect("register");
    factory
        .open
        .grant_handle(
            &scope,
            "proc-reopen",
            ProcessHandleDescriptor::new(Some("test"), Some("reopen")),
        )
        .await
        .expect("grant");
    let appended = factory
        .open
        .append_event(
            "proc-reopen",
            ProcessEventAppendRequest::new(
                "producer.reopen_wake",
                serde_json::json!({"wake_input": "survived reopen"}),
            )
            .with_replay_key("producer:reopen")
            .with_wake_target_scope(scope.clone()),
        )
        .await
        .expect("append");

    let reopened_record = factory
        .reopen
        .get_process("proc-reopen")
        .await
        .expect("process exists after reopen");
    assert_eq!(reopened_record.id, "proc-reopen");
    let reopened_events = factory
        .reopen
        .events_after("proc-reopen", 0)
        .await
        .expect("events after reopen");
    assert_eq!(reopened_events.len(), 1);
    assert_eq!(reopened_events[0].sequence, appended.event.sequence);
    assert_eq!(
        factory
            .reopen
            .list_handle_grants(&scope)
            .await
            .expect("grants after reopen")
            .len(),
        1
    );
    let replayed = factory
        .reopen
        .append_event(
            "proc-reopen",
            ProcessEventAppendRequest::new(
                "producer.reopen_wake",
                serde_json::json!({"wake_input": "survived reopen"}),
            )
            .with_replay_key("producer:reopen")
            .with_wake_target_scope(scope),
        )
        .await
        .expect("replay after reopen");
    assert_eq!(replayed.event.sequence, appended.event.sequence);
}

async fn keyed_events_materialize_idempotent_wakes(registry: Arc<dyn ProcessRegistry>) {
    let target_scope = ProcessScope::new("session");
    let target_scope_id = target_scope.id();
    registry
        .register_process(
            registration("proc-wake").with_extra_event_types([wake_event_type("process.wake")]),
        )
        .await
        .expect("register");
    let request = ProcessEventAppendRequest::new(
        "process.wake",
        serde_json::json!({
            "message": "deploy failed",
            "wake_input": "Process wake: deploy failed",
        }),
    )
    .with_replay_key("wake:deploy failed")
    .with_wake_target_scope(target_scope);

    let first = registry
        .append_event("proc-wake", request.clone())
        .await
        .expect("append");
    let second = registry
        .append_event("proc-wake", request)
        .await
        .expect("replay append");

    assert_eq!(
        first.event.sequence, second.event.sequence,
        "replaying the same key must return the same sequence, not a new event"
    );
    assert_eq!(first.wake_delivery, second.wake_delivery);
    let wake = first.wake_delivery.expect("wake delivery");
    assert_eq!(wake.input, "Process wake: deploy failed");
    assert_eq!(wake.target_scope_id, target_scope_id);
    assert_eq!(wake.process_id, "proc-wake");
    assert_eq!(wake.sequence, first.event.sequence);
    assert!(
        registry
            .append_event(
                "proc-wake",
                ProcessEventAppendRequest::new(
                    "process.wake",
                    serde_json::json!({
                        "message": "other",
                        "wake_input": "Process wake: other",
                    }),
                )
                .with_replay_key("wake:deploy failed"),
            )
            .await
            .is_err(),
        "a different payload under an existing replay key must be rejected"
    );
}

async fn terminal_and_cancel_events_require_keys(registry: Arc<dyn ProcessRegistry>) {
    registry
        .register_process(registration("proc-terminal"))
        .await
        .expect("register");

    assert!(
        registry
            .append_event(
                "proc-terminal",
                ProcessEventAppendRequest::new(
                    "process.cancel_requested",
                    serde_json::json!({"reason": "stop"}),
                ),
            )
            .await
            .is_err(),
        "cancel_requested without a replay key must be rejected"
    );
    registry
        .append_event(
            "proc-terminal",
            ProcessEventAppendRequest::cancel_requested("proc-terminal", Some("stop".to_string())),
        )
        .await
        .expect("cancel intent");
    registry
        .complete_process(
            "proc-terminal",
            ProcessAwaitOutput::Cancelled {
                message: "stopped".to_string(),
                raw: None,
                control: None,
            },
        )
        .await
        .expect("complete cancelled");
    assert_eq!(
        registry
            .get_process("proc-terminal")
            .await
            .and_then(|record| record.status.terminal_state()),
        Some(ProcessTerminalState::Cancelled)
    );
}

async fn await_reads_terminal_materialized_output(registry: Arc<dyn ProcessRegistry>) {
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

async fn transfer_handle_grants_moves_addressability(registry: Arc<dyn ProcessRegistry>) {
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
            .is_empty(),
        "addressability transfer must not append process events"
    );
}

async fn multiple_sessions_can_hold_grants(registry: Arc<dyn ProcessRegistry>) {
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

async fn processes_can_exist_with_zero_grants(registry: Arc<dyn ProcessRegistry>) {
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

async fn delete_session_revokes_handles_by_session(registry: Arc<dyn ProcessRegistry>) {
    let deleted_scope = ProcessScope::new("deleted");
    let remaining_scope = ProcessScope::new("remaining");
    for process_id in ["sole", "shared", "terminal"] {
        registry
            .register_process(
                registration(process_id).with_extra_event_types([wake_event_type("producer.wake")]),
            )
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

async fn list_non_terminal_excludes_terminal_processes(registry: Arc<dyn ProcessRegistry>) {
    registry
        .register_process(registration("proc-live"))
        .await
        .expect("register live");
    registry
        .register_process(registration("proc-done"))
        .await
        .expect("register done");
    registry
        .complete_process(
            "proc-done",
            ProcessAwaitOutput::Success {
                value: serde_json::Value::Null,
                control: None,
            },
        )
        .await
        .expect("complete done");

    let ids = registry
        .list_non_terminal()
        .await
        .expect("list non-terminal")
        .into_iter()
        .map(|record| record.id)
        .collect::<Vec<_>>();
    assert_eq!(
        ids,
        vec!["proc-live".to_string()],
        "list_non_terminal must exclude terminal processes and be process_id ordered"
    );
}

async fn list_live_handle_grants_excludes_terminal_history(registry: Arc<dyn ProcessRegistry>) {
    let scope = ProcessScope::new("history-owner");
    for process_id in ["proc-live-grant", "proc-done-grant"] {
        registry
            .register_process(registration(process_id))
            .await
            .expect("register");
        registry
            .grant_handle(
                &scope,
                process_id,
                ProcessHandleDescriptor::new(Some("test"), Some(process_id)),
            )
            .await
            .expect("grant");
    }
    registry
        .complete_process(
            "proc-done-grant",
            ProcessAwaitOutput::Success {
                value: serde_json::Value::Null,
                control: None,
            },
        )
        .await
        .expect("complete done");

    let live_ids = registry
        .list_live_handle_grants(&scope)
        .await
        .expect("list live grants")
        .into_iter()
        .map(|(grant, _)| grant.process_id)
        .collect::<Vec<_>>();
    assert_eq!(
        live_ids,
        vec!["proc-live-grant".to_string()],
        "list_live_handle_grants must exclude completed historical handles"
    );

    let all_ids = registry
        .list_handle_grants(&scope)
        .await
        .expect("list all grants")
        .into_iter()
        .map(|(grant, _)| grant.process_id)
        .collect::<Vec<_>>();
    assert_eq!(
        all_ids,
        vec!["proc-done-grant".to_string(), "proc-live-grant".to_string()],
        "list_handle_grants remains the explicit all-history path"
    );
}

async fn active_process_lease_fences_competing_owner(registry: Arc<dyn ProcessRegistry>) {
    registry
        .register_process(registration("proc-lease-active"))
        .await
        .expect("register");
    registry
        .claim_process_lease("proc-lease-active", "owner-a", 60_000)
        .await
        .expect("first claim");
    let conflict = registry
        .claim_process_lease("proc-lease-active", "owner-b", 60_000)
        .await;
    assert!(
        conflict
            .as_ref()
            .is_err_and(|err| err.to_string().contains("already leased")),
        "an active lease must fence a competing owner, got {conflict:?}"
    );
    // The original owner may re-claim its own live lease (idempotent ownership).
    registry
        .claim_process_lease("proc-lease-active", "owner-a", 60_000)
        .await
        .expect("owner re-claims its own live lease");
}

async fn superseded_process_lease_cannot_renew(registry: Arc<dyn ProcessRegistry>) {
    registry
        .register_process(registration("proc-lease-superseded"))
        .await
        .expect("register");
    let old = registry
        .claim_process_lease("proc-lease-superseded", "owner-a", 0)
        .await
        .expect("old lease");
    registry
        .claim_process_lease("proc-lease-superseded", "owner-b", 60_000)
        .await
        .expect("new owner claims the expired lease");
    let stale = registry.renew_process_lease(&old, 60_000).await;
    assert!(
        stale
            .as_ref()
            .is_err_and(|err| err.to_string().contains("missing or expired")),
        "a superseded lease must not renew, got {stale:?}"
    );
}

async fn renewed_process_lease_survives_original_expiry(registry: Arc<dyn ProcessRegistry>) {
    registry
        .register_process(registration("proc-lease-renew"))
        .await
        .expect("register");
    let lease = registry
        .claim_process_lease("proc-lease-renew", "owner-a", 20)
        .await
        .expect("lease");
    let renewed = registry
        .renew_process_lease(&lease, 60_000)
        .await
        .expect("renew");
    tokio::time::sleep(std::time::Duration::from_millis(30)).await;
    registry
        .renew_process_lease(&renewed, 60_000)
        .await
        .expect("a renewed lease survives the original TTL");
}

async fn completed_lease_releases_and_reclaim_bumps_fencing(registry: Arc<dyn ProcessRegistry>) {
    registry
        .register_process(registration("proc-lease-complete"))
        .await
        .expect("register");
    let first = registry
        .claim_process_lease("proc-lease-complete", "owner-a", 60_000)
        .await
        .expect("first claim");
    registry
        .complete_process_lease(&ProcessLeaseCompletion::from_lease(&first))
        .await
        .expect("complete lease");
    let second = registry
        .claim_process_lease("proc-lease-complete", "owner-b", 60_000)
        .await
        .expect("a new owner can claim a released lease");
    assert!(
        second.fencing_token > first.fencing_token,
        "a re-claim must bump the fencing token (was {}, now {})",
        first.fencing_token,
        second.fencing_token
    );
}

async fn stale_lease_completion_cannot_release_live_lease(registry: Arc<dyn ProcessRegistry>) {
    registry
        .register_process(registration("proc-lease-stale-complete"))
        .await
        .expect("register");
    let old = registry
        .claim_process_lease("proc-lease-stale-complete", "owner-a", 0)
        .await
        .expect("old lease");
    let current = registry
        .claim_process_lease("proc-lease-stale-complete", "owner-b", 60_000)
        .await
        .expect("new live lease");
    // A stale completion (old token) must not release the live lease.
    registry
        .complete_process_lease(&ProcessLeaseCompletion::from_lease(&old))
        .await
        .expect("stale completion is ignored");
    let conflict = registry
        .claim_process_lease("proc-lease-stale-complete", "owner-c", 60_000)
        .await;
    assert!(
        conflict
            .as_ref()
            .is_err_and(|err| err.to_string().contains("already leased")),
        "a stale completion must not release the live lease, got {conflict:?}"
    );
    // The live owner can still renew.
    registry
        .renew_process_lease(&current, 60_000)
        .await
        .expect("the live owner can still renew");
}

/// Attachment-manifest behavior expected from a [`RuntimePersistence`] backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttachmentManifestConformance {
    /// The backend stores and reconciles attachment intent rows.
    Persistent,
    /// The backend explicitly has no attachment-write story and uses the no-op
    /// manifest contract.
    Noop,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RuntimePersistenceConformance {
    pub attachment_manifest: AttachmentManifestConformance,
}

impl RuntimePersistenceConformance {
    pub const fn noop_attachment_manifest() -> Self {
        Self {
            attachment_manifest: AttachmentManifestConformance::Noop,
        }
    }
}

impl Default for RuntimePersistenceConformance {
    fn default() -> Self {
        Self {
            attachment_manifest: AttachmentManifestConformance::Persistent,
        }
    }
}

/// Run the [`RuntimePersistence`] durability conformance suite against the
/// backend produced by `make`. `make` must return a fresh, empty,
/// single-session store on each call.
///
/// Covers the durability crown jewels: optimistic head CAS, session binding,
/// checkpoint/usage hydration, queued work claim fencing, attachment manifest
/// intent/commit/GC reconciliation, lease fencing (claim/renew/abandon/
/// supersede/expire), lease-guarded journal writes, replay-key journal
/// idempotency, and atomic final commit that clears the journal only under a
/// live lease (else preserves resume state). In-flight
/// `RuntimeTurnCheckpoint` round-tripping — whose hash validation is
/// backend-specific — is exercised per backend.
pub async fn runtime_persistence<F>(make: F)
where
    F: Fn() -> Arc<dyn RuntimePersistence>,
{
    runtime_persistence_with_options(make, RuntimePersistenceConformance::default()).await;
}

/// Run the full [`RuntimePersistence`] suite plus durable reopen checks.
pub async fn runtime_persistence_reopenable<F>(make: F)
where
    F: Fn() -> ReopenableRuntimePersistence,
{
    runtime_persistence(|| make().open).await;
    runtime_persistence_survives_reopen(make()).await;
}

pub async fn runtime_persistence_with_options<F>(make: F, options: RuntimePersistenceConformance)
where
    F: Fn() -> Arc<dyn RuntimePersistence>,
{
    commit_increments_head_and_round_trips_agent_frames(make()).await;
    commit_rejects_a_different_session_id(make()).await;
    load_hydrates_checkpoint_and_usage(make()).await;
    match options.attachment_manifest {
        AttachmentManifestConformance::Persistent => {
            attachment_manifest_records_intent_and_commit_stamps(make()).await;
        }
        AttachmentManifestConformance::Noop => {
            noop_attachment_manifest_is_explicit_and_empty(make()).await;
        }
    }
    queued_work_source_keys_are_idempotent_and_list_ordered(make()).await;
    queued_work_claims_respect_boundaries_renewal_and_abandon(make()).await;
    queued_work_respects_availability_limits_exclusivity_reclaim_and_sessions(make()).await;
    queued_work_join_groups_by_delivery_policy_and_merge_key(make()).await;
    queued_work_completion_is_lease_guarded(make()).await;
    queue_completion_state_commit_and_journal_clear_are_atomic(make()).await;
    session_metadata_round_trips(make()).await;
    tombstone_vacuum_and_gc_are_minimally_consistent(make()).await;
    journal_is_idempotent_and_cleared_on_final_commit(make()).await;
    substrate_native_final_commit_is_idempotent_and_conflicts_on_changed_hash(make()).await;
    active_lease_fences_competing_claims(make()).await;
    superseded_lease_cannot_write_or_clear(make()).await;
    renewed_lease_survives_original_expiry(make()).await;
    abandon_releases_owner_and_preserves_journal(make()).await;
    stale_final_commit_rejects_and_preserves_resume(make()).await;
    expired_final_commit_rejects_and_preserves_resume(make()).await;
}

fn effect_record(session_id: &str, turn_id: &str, effect: &str) -> RuntimeEffectJournalRecord {
    RuntimeEffectJournalRecord {
        schema_version: RUNTIME_EFFECT_JOURNAL_SCHEMA_VERSION,
        session_id: session_id.to_string(),
        turn_id: turn_id.to_string(),
        replay_key: format!("{session_id}:{turn_id}:{effect}"),
        envelope_hash: format!("hash-{effect}"),
        effect_kind: RuntimeEffectKind::Sleep,
        outcome: RuntimeEffectOutcome::Sleep,
        created_at_epoch_ms: 1,
    }
}

fn queued_draft(
    session_id: &str,
    text: &str,
    delivery_policy: DeliveryPolicy,
    slot_policy: SlotPolicy,
) -> QueuedWorkBatchDraft {
    QueuedWorkBatchDraft::new(
        session_id,
        delivery_policy,
        slot_policy,
        vec![QueuedWorkPayload::turn_input(TurnInput::text(text))],
    )
}

fn queued_batch_text(batch: &QueuedWorkBatch) -> Option<&str> {
    let payload = batch.items.first().map(|item| &item.payload)?;
    match payload {
        QueuedWorkPayload::TurnInput { input } => input.items.first().and_then(|item| match item {
            crate::InputItem::Text { text } => Some(text.as_str()),
            crate::InputItem::ImageRef { .. } => None,
        }),
        QueuedWorkPayload::ProcessWake { .. }
        | QueuedWorkPayload::HostEvent { .. }
        | QueuedWorkPayload::Timer { .. }
        | QueuedWorkPayload::Resume { .. } => None,
    }
}

fn sample_session_node(id: &str, parent: Option<&str>) -> SessionNodeRecord {
    SessionNodeRecord {
        node_id: id.to_string(),
        parent_node_id: parent.map(ToOwned::to_owned),
        caused_by: None,
        agent_frame_id: None,
        timestamp: "1970-01-01T00:00:00Z".to_string(),
        payload: SessionNodePayload::Event {
            event: crate::SessionEventRecord::Protocol(
                ProtocolEvent::typed("conformance", serde_json::json!({ "node": id }))
                    .expect("protocol event"),
            ),
        },
    }
}

fn attachment_intent(id: &str) -> AttachmentIntent {
    AttachmentIntent {
        attachment_id: AttachmentId::new(id.to_string()),
        session_id: "root".to_string(),
        canonical_uri: format!("sha256:{id}"),
        intent_at_epoch_ms: 100,
    }
}

async fn commit_increments_head_and_round_trips_agent_frames(store: Arc<dyn RuntimePersistence>) {
    let mut state = RuntimeSessionState {
        session_id: "root".to_string(),
        policy: SessionPolicy {
            model: ModelSpec::from_token_limits("gpt-5.4-mini", None, 200_000, None, None)
                .expect("valid model spec"),
            ..SessionPolicy::default()
        },
        ..RuntimeSessionState::default()
    };
    state.ensure_agent_frame_initialized();
    let previous_frame_id = state.current_agent_frame_id.clone();
    let assignment = state
        .current_agent_frame()
        .expect("initial frame")
        .assignment
        .clone();
    state.append_agent_frame(AgentFrameRecord::new(
        "frame-2".to_string(),
        "root".to_string(),
        Some(previous_frame_id),
        AgentFrameReason::ContinueAs,
        None,
        assignment,
        ProtocolTurnOptions::default(),
    ));
    state.set_execution_state_snapshot(Some(b"frame-vm".to_vec()));

    store
        .commit_runtime_state(RuntimeCommit::persisted_state(&state, &[]))
        .await
        .expect("commit runtime state");
    let read = store
        .load_session(SessionReadScope::FullGraph)
        .await
        .expect("load session")
        .expect("session read");

    assert_eq!(read.current_agent_frame_id, "frame-2");
    assert_eq!(read.agent_frames.len(), 2);
    let current = read
        .agent_frames
        .iter()
        .find(|frame| frame.frame_id == "frame-2")
        .expect("current frame");
    assert_eq!(
        current.execution_state_snapshot.as_deref(),
        Some(&b"frame-vm"[..])
    );
    assert_eq!(
        read.checkpoint
            .as_ref()
            .and_then(|checkpoint| checkpoint.execution_state.as_deref()),
        Some(&b"frame-vm"[..])
    );
}

async fn commit_rejects_a_different_session_id(store: Arc<dyn RuntimePersistence>) {
    let alpha = RuntimeSessionState {
        session_id: "alpha".to_string(),
        ..RuntimeSessionState::default()
    };
    store
        .commit_runtime_state(RuntimeCommit::persisted_state(&alpha, &[]))
        .await
        .expect("first commit binds the session");
    let beta = RuntimeSessionState {
        session_id: "beta".to_string(),
        ..RuntimeSessionState::default()
    };
    let result = store
        .commit_runtime_state(RuntimeCommit::persisted_state(&beta, &[]))
        .await;
    assert!(
        result.is_err(),
        "a single-session store must reject a commit for a different session id"
    );
}

async fn load_hydrates_checkpoint_and_usage(store: Arc<dyn RuntimePersistence>) {
    let state = RuntimeSessionState {
        session_id: "hydrated".to_string(),
        tool_state_snapshot: Some(ToolState::default().with_generation(9)),
        plugin_snapshot_revision: Some(12),
        plugin_snapshot: Some(PluginSessionSnapshot {
            plugins: Default::default(),
        }),
        ..RuntimeSessionState::default()
    };
    let usage = TokenLedgerEntry {
        source: "turn".to_string(),
        model: "mock-model".to_string(),
        usage: TokenUsage {
            input_tokens: 11,
            output_tokens: 7,
            cached_input_tokens: 3,
            reasoning_tokens: 5,
        },
    };

    store
        .commit_runtime_state(RuntimeCommit::persisted_state(&state, &[usage]))
        .await
        .expect("commit");

    let read = store
        .load_session(SessionReadScope::FullGraph)
        .await
        .expect("load")
        .expect("session");
    let checkpoint = read.checkpoint.expect("checkpoint");
    assert_eq!(read.session_id, "hydrated");
    assert_eq!(
        checkpoint
            .tool_state
            .expect("dynamic snapshot")
            .generation(),
        9
    );
    assert_eq!(checkpoint.plugin_snapshot_revision, Some(12));
    assert_eq!(read.token_ledger.len(), 1);
    assert_eq!(read.token_ledger[0].usage.input_tokens, 11);
}

async fn attachment_manifest_records_intent_and_commit_stamps(store: Arc<dyn RuntimePersistence>) {
    let committed_by_runtime = AttachmentId::new("runtime-commit".to_string());
    let committed_out_of_band = AttachmentId::new("manual-commit".to_string());
    let orphan = AttachmentId::new("orphan".to_string());
    for id in [&committed_by_runtime, &committed_out_of_band, &orphan] {
        store
            .record_intent(attachment_intent(id.as_str()))
            .expect("record attachment intent");
    }

    let mut uncommitted = store
        .list_uncommitted(200)
        .expect("list uncommitted attachment intents");
    uncommitted.sort_by(|left, right| left.attachment_id.cmp(&right.attachment_id));
    assert_eq!(uncommitted.len(), 3);

    store
        .commit_refs("root", std::slice::from_ref(&committed_out_of_band))
        .expect("commit attachment ref out of band");
    let state = RuntimeSessionState {
        session_id: "root".to_string(),
        ..RuntimeSessionState::default()
    };
    store
        .commit_runtime_state(
            RuntimeCommit::persisted_state(&state, &[])
                .with_committed_attachments([committed_by_runtime.clone()]),
        )
        .await
        .expect("runtime commit stamps attachment manifest");

    let still_uncommitted = store
        .list_uncommitted(200)
        .expect("list remaining uncommitted attachments");
    assert_eq!(still_uncommitted.len(), 1);
    assert_eq!(still_uncommitted[0].attachment_id, orphan);
    assert!(still_uncommitted[0].committed_at_epoch_ms.is_none());

    store.forget(&orphan).expect("forget orphan attachment");
    assert!(
        store
            .list_uncommitted(200)
            .expect("list after forget")
            .is_empty()
    );
}

async fn noop_attachment_manifest_is_explicit_and_empty(store: Arc<dyn RuntimePersistence>) {
    let attachment = AttachmentId::new("noop".to_string());
    store
        .record_intent(attachment_intent(attachment.as_str()))
        .expect("noop record intent succeeds");
    store
        .commit_refs("root", std::slice::from_ref(&attachment))
        .expect("noop commit refs succeeds");
    assert!(
        store
            .list_uncommitted(200)
            .expect("noop list uncommitted")
            .is_empty(),
        "declared no-op attachment manifests must not retain intent rows"
    );
    store.forget(&attachment).expect("noop forget succeeds");
}

async fn queued_work_source_keys_are_idempotent_and_list_ordered(
    store: Arc<dyn RuntimePersistence>,
) {
    let first = store
        .enqueue_queued_work(
            queued_draft(
                "root",
                "first",
                DeliveryPolicy::EarliestSafeBoundary,
                SlotPolicy::Join,
            )
            .with_source_key("source:first"),
        )
        .await
        .expect("enqueue first batch");
    let replay = store
        .enqueue_queued_work(
            queued_draft(
                "root",
                "different replay payload",
                DeliveryPolicy::EarliestSafeBoundary,
                SlotPolicy::Join,
            )
            .with_source_key("source:first"),
        )
        .await
        .expect("replay first batch");
    let second = store
        .enqueue_queued_work(queued_draft(
            "root",
            "second",
            DeliveryPolicy::EarliestSafeBoundary,
            SlotPolicy::Exclusive,
        ))
        .await
        .expect("enqueue second batch");
    store
        .enqueue_queued_work(queued_draft(
            "other",
            "other session",
            DeliveryPolicy::EarliestSafeBoundary,
            SlotPolicy::Exclusive,
        ))
        .await
        .expect("enqueue other session");

    assert_eq!(
        first.batch_id, replay.batch_id,
        "replaying a source key must return the original batch"
    );
    assert_eq!(first.items[0].item_id, replay.items[0].item_id);
    assert_eq!(
        queued_batch_text(&replay),
        Some("first"),
        "source-key replay must return the original stored payload, not the replay attempt"
    );
    let listed = store
        .list_queued_work("root")
        .await
        .expect("list queued work");
    assert_eq!(
        listed
            .iter()
            .map(|batch| batch.batch_id.as_str())
            .collect::<Vec<_>>(),
        vec![first.batch_id.as_str(), second.batch_id.as_str()]
    );
    assert!(listed[0].enqueue_seq < listed[1].enqueue_seq);
}

async fn queued_work_claims_respect_boundaries_renewal_and_abandon(
    store: Arc<dyn RuntimePersistence>,
) {
    let after_commit = store
        .enqueue_queued_work(queued_draft(
            "root",
            "after current commit",
            DeliveryPolicy::AfterCurrentTurnCommit,
            SlotPolicy::Exclusive,
        ))
        .await
        .expect("enqueue after-commit work");
    let earliest = store
        .enqueue_queued_work(queued_draft(
            "root",
            "earliest",
            DeliveryPolicy::EarliestSafeBoundary,
            SlotPolicy::Exclusive,
        ))
        .await
        .expect("enqueue earliest work");

    assert!(
        store
            .claim_ready_queued_work(
                "root",
                "owner-a",
                QueuedWorkClaimBoundary::ActiveTurnCheckpoint,
                60_000,
                10,
            )
            .await
            .expect("checkpoint claim")
            .is_none(),
        "after-current-commit work at the queue head must wait for the idle boundary"
    );

    let idle_claim = store
        .claim_ready_queued_work("root", "owner-a", QueuedWorkClaimBoundary::Idle, 60_000, 10)
        .await
        .expect("idle claim")
        .expect("idle claim exists");
    assert_eq!(idle_claim.batches.len(), 1);
    assert_eq!(idle_claim.batches[0].batch_id, after_commit.batch_id);

    let checkpoint_claim = store
        .claim_ready_queued_work(
            "root",
            "owner-b",
            QueuedWorkClaimBoundary::ActiveTurnCheckpoint,
            60_000,
            10,
        )
        .await
        .expect("checkpoint claim after head is leased")
        .expect("checkpoint claim exists");
    assert_eq!(checkpoint_claim.batches[0].batch_id, earliest.batch_id);

    store
        .abandon_queued_work_claim(&idle_claim)
        .await
        .expect("abandon idle claim");
    let reclaimed = store
        .claim_ready_queued_work("root", "owner-c", QueuedWorkClaimBoundary::Idle, 60_000, 10)
        .await
        .expect("reclaim abandoned work")
        .expect("reclaimed work exists");
    assert_eq!(reclaimed.batches[0].batch_id, after_commit.batch_id);
    assert!(
        reclaimed.fencing_token > idle_claim.fencing_token,
        "reclaiming abandoned work must advance the fencing token"
    );

    let renewed = store
        .renew_queued_work_claim(&reclaimed, 60_000)
        .await
        .expect("renew queued work claim");
    assert_eq!(renewed.claim_id, reclaimed.claim_id);
    assert_eq!(renewed.lease_token, reclaimed.lease_token);
    assert_eq!(renewed.batches[0].batch_id, reclaimed.batches[0].batch_id);
    assert!(renewed.expires_at_epoch_ms >= reclaimed.expires_at_epoch_ms);
}

async fn queued_work_respects_availability_limits_exclusivity_reclaim_and_sessions(
    store: Arc<dyn RuntimePersistence>,
) {
    store
        .enqueue_queued_work(
            queued_draft(
                "root",
                "not ready",
                DeliveryPolicy::EarliestSafeBoundary,
                SlotPolicy::Exclusive,
            )
            .with_available_at_ms(4_102_444_800_000),
        )
        .await
        .expect("enqueue unavailable work");
    let exclusive = store
        .enqueue_queued_work(queued_draft(
            "root",
            "exclusive",
            DeliveryPolicy::EarliestSafeBoundary,
            SlotPolicy::Exclusive,
        ))
        .await
        .expect("enqueue exclusive work");
    let joined = store
        .enqueue_queued_work(
            queued_draft(
                "root",
                "joined",
                DeliveryPolicy::EarliestSafeBoundary,
                SlotPolicy::Join,
            )
            .with_merge_key(MergeKey::Group("root".to_string())),
        )
        .await
        .expect("enqueue joined work");
    let other = store
        .enqueue_queued_work(queued_draft(
            "other",
            "other session",
            DeliveryPolicy::EarliestSafeBoundary,
            SlotPolicy::Exclusive,
        ))
        .await
        .expect("enqueue other session work");

    let claim = store
        .claim_ready_queued_work("root", "owner-a", QueuedWorkClaimBoundary::Idle, 60_000, 10)
        .await
        .expect("claim root")
        .expect("root claim");
    assert_eq!(
        claim
            .batches
            .iter()
            .map(|batch| batch.batch_id.as_str())
            .collect::<Vec<_>>(),
        vec![exclusive.batch_id.as_str()],
        "an exclusive batch must claim alone and unavailable earlier work must be skipped"
    );
    let next_root = store
        .claim_ready_queued_work("root", "owner-b", QueuedWorkClaimBoundary::Idle, 60_000, 10)
        .await
        .expect("claim joined")
        .expect("joined claim");
    assert_eq!(next_root.batches[0].batch_id, joined.batch_id);
    let other_claim = store
        .claim_ready_queued_work(
            "other",
            "owner-c",
            QueuedWorkClaimBoundary::Idle,
            60_000,
            10,
        )
        .await
        .expect("claim other")
        .expect("other claim");
    assert_eq!(
        other_claim.batches[0].batch_id, other.batch_id,
        "claiming one session must not consume queued work from another session"
    );

    let reclaimed_source = store
        .enqueue_queued_work(queued_draft(
            "reclaim",
            "expired claim",
            DeliveryPolicy::EarliestSafeBoundary,
            SlotPolicy::Exclusive,
        ))
        .await
        .expect("enqueue reclaim work");
    let expired = store
        .claim_ready_queued_work("reclaim", "owner-a", QueuedWorkClaimBoundary::Idle, 0, 1)
        .await
        .expect("claim with zero ttl")
        .expect("expired claim");
    let reclaimed = store
        .claim_ready_queued_work(
            "reclaim",
            "owner-b",
            QueuedWorkClaimBoundary::Idle,
            60_000,
            1,
        )
        .await
        .expect("reclaim expired")
        .expect("reclaimed expired claim");
    assert_eq!(reclaimed.batches[0].batch_id, reclaimed_source.batch_id);
    assert!(
        reclaimed.fencing_token > expired.fencing_token,
        "reclaiming an expired queued-work claim must bump the fencing token"
    );

    let limited_first = store
        .enqueue_queued_work(
            queued_draft(
                "limited",
                "one",
                DeliveryPolicy::EarliestSafeBoundary,
                SlotPolicy::Join,
            )
            .with_merge_key(MergeKey::Group("limited".to_string())),
        )
        .await
        .expect("enqueue limited one");
    let limited_second = store
        .enqueue_queued_work(
            queued_draft(
                "limited",
                "two",
                DeliveryPolicy::EarliestSafeBoundary,
                SlotPolicy::Join,
            )
            .with_merge_key(MergeKey::Group("limited".to_string())),
        )
        .await
        .expect("enqueue limited two");
    let limited_third = store
        .enqueue_queued_work(
            queued_draft(
                "limited",
                "three",
                DeliveryPolicy::EarliestSafeBoundary,
                SlotPolicy::Join,
            )
            .with_merge_key(MergeKey::Group("limited".to_string())),
        )
        .await
        .expect("enqueue limited three");
    let limited = store
        .claim_ready_queued_work("limited", "owner", QueuedWorkClaimBoundary::Idle, 60_000, 2)
        .await
        .expect("limited claim")
        .expect("limited claim exists");
    assert_eq!(
        limited
            .batches
            .iter()
            .map(|batch| batch.batch_id.as_str())
            .collect::<Vec<_>>(),
        vec![
            limited_first.batch_id.as_str(),
            limited_second.batch_id.as_str()
        ],
        "max_batches must cap a join claim"
    );
    let remaining = store
        .claim_ready_queued_work(
            "limited",
            "owner-next",
            QueuedWorkClaimBoundary::Idle,
            60_000,
            10,
        )
        .await
        .expect("remaining claim")
        .expect("remaining claim exists");
    assert_eq!(remaining.batches[0].batch_id, limited_third.batch_id);
}

async fn queued_work_join_groups_by_delivery_policy_and_merge_key(
    store: Arc<dyn RuntimePersistence>,
) {
    let first = store
        .enqueue_queued_work(
            queued_draft(
                "root",
                "group a one",
                DeliveryPolicy::EarliestSafeBoundary,
                SlotPolicy::Join,
            )
            .with_merge_key(MergeKey::Group("a".to_string())),
        )
        .await
        .expect("enqueue group a one");
    let second = store
        .enqueue_queued_work(
            queued_draft(
                "root",
                "group a two",
                DeliveryPolicy::EarliestSafeBoundary,
                SlotPolicy::Join,
            )
            .with_merge_key(MergeKey::Group("a".to_string())),
        )
        .await
        .expect("enqueue group a two");
    let different_merge = store
        .enqueue_queued_work(
            queued_draft(
                "root",
                "group b",
                DeliveryPolicy::EarliestSafeBoundary,
                SlotPolicy::Join,
            )
            .with_merge_key(MergeKey::Group("b".to_string())),
        )
        .await
        .expect("enqueue group b");
    let different_delivery = store
        .enqueue_queued_work(
            queued_draft(
                "root",
                "after commit",
                DeliveryPolicy::AfterCurrentTurnCommit,
                SlotPolicy::Join,
            )
            .with_merge_key(MergeKey::Group("a".to_string())),
        )
        .await
        .expect("enqueue after-commit");

    let first_claim = store
        .claim_ready_queued_work("root", "owner-a", QueuedWorkClaimBoundary::Idle, 60_000, 10)
        .await
        .expect("claim first group")
        .expect("first group claim");
    assert_eq!(
        first_claim
            .batches
            .iter()
            .map(|batch| batch.batch_id.as_str())
            .collect::<Vec<_>>(),
        vec![first.batch_id.as_str(), second.batch_id.as_str()],
        "join claims must group only adjacent batches with the same delivery policy and merge key"
    );
    let second_claim = store
        .claim_ready_queued_work("root", "owner-b", QueuedWorkClaimBoundary::Idle, 60_000, 10)
        .await
        .expect("claim second group")
        .expect("second group claim");
    assert_eq!(second_claim.batches[0].batch_id, different_merge.batch_id);
    let third_claim = store
        .claim_ready_queued_work("root", "owner-c", QueuedWorkClaimBoundary::Idle, 60_000, 10)
        .await
        .expect("claim third group")
        .expect("third group claim");
    assert_eq!(third_claim.batches[0].batch_id, different_delivery.batch_id);
}

async fn queued_work_completion_is_lease_guarded(store: Arc<dyn RuntimePersistence>) {
    let first = store
        .enqueue_queued_work(
            queued_draft(
                "root",
                "join one",
                DeliveryPolicy::EarliestSafeBoundary,
                SlotPolicy::Join,
            )
            .with_merge_key(MergeKey::Group("joined".to_string())),
        )
        .await
        .expect("enqueue first joined batch");
    let second = store
        .enqueue_queued_work(
            queued_draft(
                "root",
                "join two",
                DeliveryPolicy::EarliestSafeBoundary,
                SlotPolicy::Join,
            )
            .with_merge_key(MergeKey::Group("joined".to_string())),
        )
        .await
        .expect("enqueue second joined batch");
    let claim = store
        .claim_ready_queued_work("root", "owner-a", QueuedWorkClaimBoundary::Idle, 60_000, 10)
        .await
        .expect("claim joined batches")
        .expect("joined claim exists");
    assert_eq!(
        claim
            .batches
            .iter()
            .map(|batch| batch.batch_id.as_str())
            .collect::<Vec<_>>(),
        vec![first.batch_id.as_str(), second.batch_id.as_str()]
    );

    let mut stale_completion = claim.completion();
    stale_completion.lease_token.push_str(":stale");
    let state = RuntimeSessionState {
        session_id: "root".to_string(),
        ..RuntimeSessionState::default()
    };
    let err = store
        .commit_runtime_state(
            RuntimeCommit::persisted_state(&state, &[]).completing_queue_claim(stale_completion),
        )
        .await
        .expect_err("stale queued-work completion must fail");
    assert!(matches!(err, StoreError::QueuedWorkClaimExpired { .. }));
    assert_eq!(
        store
            .list_queued_work("root")
            .await
            .expect("stale completion preserves queued work")
            .len(),
        2
    );

    store
        .commit_runtime_state(
            RuntimeCommit::persisted_state(&state, &[]).completing_queue_claim(claim.completion()),
        )
        .await
        .expect("valid queued-work completion commits");
    assert!(
        store
            .list_queued_work("root")
            .await
            .expect("valid completion clears queued work")
            .is_empty()
    );
}

async fn queue_completion_state_commit_and_journal_clear_are_atomic(
    store: Arc<dyn RuntimePersistence>,
) {
    let batch = store
        .enqueue_queued_work(queued_draft(
            "root",
            "atomic queue",
            DeliveryPolicy::EarliestSafeBoundary,
            SlotPolicy::Exclusive,
        ))
        .await
        .expect("enqueue queue batch");
    let claim = store
        .claim_ready_queued_work(
            "root",
            "queue-owner",
            QueuedWorkClaimBoundary::Idle,
            60_000,
            1,
        )
        .await
        .expect("claim queue")
        .expect("queue claim");
    assert_eq!(claim.batches[0].batch_id, batch.batch_id);
    let turns = store
        .embedded_durable_turn_store()
        .expect("embedded durable turn");
    let lease = turns
        .claim_runtime_turn_lease("root", "turn-atomic", "turn-owner", 60_000)
        .await
        .expect("turn lease");
    let record = effect_record("root", "turn-atomic", "atomic-effect");
    turns
        .save_runtime_effect_outcome(&lease, record.clone())
        .await
        .expect("save journal");
    let state = RuntimeSessionState {
        session_id: "root".to_string(),
        turn_index: 41,
        ..RuntimeSessionState::default()
    };
    let base_commit = RuntimeCommit::persisted_state(&state, &[]);
    let commit_hash = base_commit.turn_commit_hash().expect("turn commit hash");
    let mut stale_queue_completion = claim.completion();
    stale_queue_completion.lease_token.push_str(":stale");
    let err = store
        .commit_runtime_state(
            base_commit
                .clone()
                .clearing_completed_turn(RuntimeTurnCompletion::from_lease(
                    &lease,
                    commit_hash.clone(),
                ))
                .completing_queue_claim(stale_queue_completion),
        )
        .await
        .expect_err("stale queue completion must reject the whole final commit");
    assert!(matches!(err, StoreError::QueuedWorkClaimExpired { .. }));
    assert!(
        store
            .load_session(SessionReadScope::FullGraph)
            .await
            .expect("load after rejected atomic commit")
            .is_none(),
        "rejected queue completion must not persist session state"
    );
    assert_eq!(
        store
            .list_queued_work("root")
            .await
            .expect("list after rejected atomic commit")
            .len(),
        1,
        "rejected queue completion must preserve queued work"
    );
    assert!(
        store
            .embedded_durable_turn_store()
            .expect("embedded durable turn")
            .load_runtime_effect_outcome("root", "turn-atomic", &record.replay_key)
            .await
            .expect("load journal after rejected atomic commit")
            .is_some(),
        "rejected queue completion must preserve turn journal rows"
    );

    store
        .commit_runtime_state(
            base_commit
                .clearing_completed_turn(RuntimeTurnCompletion::from_lease(&lease, commit_hash))
                .completing_queue_claim(claim.completion()),
        )
        .await
        .expect("valid final commit clears queue and journal atomically");
    assert!(
        store
            .load_session(SessionReadScope::FullGraph)
            .await
            .expect("load after accepted atomic commit")
            .is_some()
    );
    assert!(
        store
            .list_queued_work("root")
            .await
            .expect("list after accepted atomic commit")
            .is_empty()
    );
    assert!(
        store
            .embedded_durable_turn_store()
            .expect("embedded durable turn")
            .load_runtime_effect_outcome("root", "turn-atomic", &record.replay_key)
            .await
            .expect("load journal after accepted atomic commit")
            .is_none()
    );
}

async fn session_metadata_round_trips(store: Arc<dyn RuntimePersistence>) {
    let meta = SessionMeta {
        session_id: "root".to_string(),
        session_name: "Conformance Root".to_string(),
        created_at: "2026-06-02T00:00:00Z".to_string(),
        model: "gpt-5.4-mini".to_string(),
        cwd: Some("/tmp/lash-conformance".to_string()),
        relation: SessionRelation::Root,
    };
    store
        .save_session_meta(meta.clone())
        .await
        .expect("save session meta");
    let loaded = store
        .load_session_meta()
        .await
        .expect("load session meta")
        .expect("session meta present");
    assert_eq!(loaded.session_id, meta.session_id);
    assert_eq!(loaded.session_name, meta.session_name);
    assert_eq!(loaded.created_at, meta.created_at);
    assert_eq!(loaded.model, meta.model);
    assert_eq!(loaded.cwd, meta.cwd);
    assert_eq!(loaded.relation, meta.relation);
}

async fn tombstone_vacuum_and_gc_are_minimally_consistent(store: Arc<dyn RuntimePersistence>) {
    let mut state = RuntimeSessionState {
        session_id: "root".to_string(),
        session_graph: crate::SessionGraph::from_nodes(
            vec![
                sample_session_node("node-live", None),
                sample_session_node("node-delete", Some("node-live")),
            ],
            Some("node-delete".to_string()),
        ),
        graph_replace_required: true,
        ..RuntimeSessionState::default()
    };
    state.head_revision = None;
    store
        .commit_runtime_state(RuntimeCommit::persisted_state(&state, &[]))
        .await
        .expect("commit graph");
    assert!(
        store
            .load_node("node-delete")
            .await
            .expect("load node before tombstone")
            .is_some()
    );
    store
        .tombstone_nodes(&["node-delete".to_string()])
        .await
        .expect("tombstone node");
    assert!(
        store
            .load_node("node-delete")
            .await
            .expect("load node after tombstone")
            .is_none(),
        "tombstoned nodes must be hidden from direct loads"
    );
    let read = store
        .load_session(SessionReadScope::FullGraph)
        .await
        .expect("load graph after tombstone")
        .expect("session after tombstone");
    assert!(
        !read
            .graph
            .nodes
            .iter()
            .any(|node| node.node_id == "node-delete"),
        "tombstoned nodes must be hidden from session graph loads"
    );
    let vacuum = store.vacuum().await.expect("vacuum");
    assert!(
        vacuum.removed_node_count <= 1,
        "vacuum must report only rows removed by this call, got {vacuum:?}"
    );
    store
        .gc_unreachable()
        .await
        .expect("gc_unreachable should be safe to call");
}

async fn runtime_persistence_survives_reopen(factory: ReopenableRuntimePersistence) {
    let meta = SessionMeta {
        session_id: "root".to_string(),
        session_name: "Durable Root".to_string(),
        created_at: "2026-06-02T00:00:00Z".to_string(),
        model: "gpt-5.4-mini".to_string(),
        cwd: Some("/tmp/lash-reopen".to_string()),
        relation: SessionRelation::Root,
    };
    factory
        .open
        .save_session_meta(meta.clone())
        .await
        .expect("save meta");
    let state = RuntimeSessionState {
        session_id: "root".to_string(),
        tool_state_snapshot: Some(ToolState::default().with_generation(77)),
        ..RuntimeSessionState::default()
    };
    factory
        .open
        .commit_runtime_state(RuntimeCommit::persisted_state(&state, &[]))
        .await
        .expect("commit state");
    let queued = factory
        .open
        .enqueue_queued_work(
            queued_draft(
                "root",
                "survives reopen",
                DeliveryPolicy::EarliestSafeBoundary,
                SlotPolicy::Exclusive,
            )
            .with_source_key("reopen:queued"),
        )
        .await
        .expect("enqueue queued work");
    let attachment = AttachmentId::new("reopen-attachment".to_string());
    factory
        .open
        .record_intent(AttachmentIntent {
            attachment_id: attachment.clone(),
            session_id: "root".to_string(),
            canonical_uri: "sha256:reopen-attachment".to_string(),
            intent_at_epoch_ms: 100,
        })
        .expect("record attachment intent");

    let reopened_meta = factory
        .reopen
        .load_session_meta()
        .await
        .expect("load reopened meta")
        .expect("reopened meta");
    assert_eq!(reopened_meta.session_name, meta.session_name);
    let reopened = factory
        .reopen
        .load_session(SessionReadScope::FullGraph)
        .await
        .expect("load reopened state")
        .expect("reopened state");
    assert_eq!(reopened.session_id, "root");
    assert_eq!(
        reopened
            .checkpoint
            .as_ref()
            .and_then(|checkpoint| checkpoint.tool_state.as_ref())
            .map(|tool_state| tool_state.generation()),
        Some(77)
    );
    let reopened_queue = factory
        .reopen
        .list_queued_work("root")
        .await
        .expect("list reopened queue");
    assert_eq!(reopened_queue.len(), 1);
    assert_eq!(reopened_queue[0].batch_id, queued.batch_id);
    assert_eq!(
        queued_batch_text(&reopened_queue[0]),
        Some("survives reopen")
    );
    let reopened_intents = factory
        .reopen
        .list_uncommitted(200)
        .expect("list reopened attachment intents");
    assert!(
        reopened_intents
            .iter()
            .any(|intent| intent.attachment_id == attachment),
        "attachment intent rows must survive reopening a durable store"
    );
}

async fn journal_is_idempotent_and_cleared_on_final_commit(store: Arc<dyn RuntimePersistence>) {
    let turns = store
        .embedded_durable_turn_store()
        .expect("embedded durable turn");
    let lease = turns
        .claim_runtime_turn_lease("root", "turn-1", "test-owner", 60_000)
        .await
        .expect("lease");
    let record = effect_record("root", "turn-1", "sleep");
    turns
        .save_runtime_effect_outcome(&lease, record.clone())
        .await
        .expect("save journal");
    let loaded = turns
        .load_runtime_effect_outcome("root", "turn-1", &record.replay_key)
        .await
        .expect("load journal")
        .expect("journal record");
    assert_eq!(loaded.envelope_hash, record.envelope_hash);
    // Replaying the same key is idempotent (overwrites, no duplicate row).
    turns
        .save_runtime_effect_outcome(&lease, record.clone())
        .await
        .expect("replay save is idempotent");

    let state = RuntimeSessionState {
        session_id: "root".to_string(),
        ..RuntimeSessionState::default()
    };
    let commit = RuntimeCommit::persisted_state(&state, &[]);
    let commit_hash = commit.turn_commit_hash().expect("turn commit hash");
    let commit =
        commit.clearing_completed_turn(RuntimeTurnCompletion::from_lease(&lease, commit_hash));
    store
        .commit_runtime_state(commit)
        .await
        .expect("final commit clears turn");

    assert!(
        store
            .embedded_durable_turn_store()
            .expect("embedded durable turn")
            .load_runtime_effect_outcome("root", "turn-1", &record.replay_key)
            .await
            .expect("load after clear")
            .is_none(),
        "a final commit under a live lease must clear the journal"
    );
    assert!(
        store
            .embedded_durable_turn_store()
            .expect("embedded durable turn")
            .load_runtime_turn_checkpoint("root", "turn-1")
            .await
            .expect("load checkpoint")
            .is_none()
    );
}

async fn substrate_native_final_commit_is_idempotent_and_conflicts_on_changed_hash(
    store: Arc<dyn RuntimePersistence>,
) {
    let state = RuntimeSessionState {
        session_id: "root".to_string(),
        ..RuntimeSessionState::default()
    };
    let commit = RuntimeCommit::persisted_state(&state, &[]);
    let turn_commit_hash = commit.turn_commit_hash().expect("turn commit hash");
    let commit = commit.clearing_completed_turn(RuntimeTurnCompletion::substrate_native(
        "root",
        "provider-turn",
        turn_commit_hash.clone(),
    ));

    let first = store
        .commit_runtime_state(commit.clone())
        .await
        .expect("substrate-native final commit does not require a Lash lease");
    let retry = store
        .commit_runtime_state(commit)
        .await
        .expect("same substrate-native final commit retries idempotently");
    assert_eq!(retry.head_revision, first.head_revision);
    assert_eq!(retry.checkpoint_ref, first.checkpoint_ref);

    let mut retry_from_new_head = RuntimeCommit::persisted_state(&state, &[]);
    retry_from_new_head.expected_head_revision = Some(first.head_revision);
    let retry_hash = retry_from_new_head
        .turn_commit_hash()
        .expect("retry commit hash");
    assert_eq!(
        retry_hash, turn_commit_hash,
        "turn commit identity must not depend on the optimistic CAS revision"
    );

    let changed_state = RuntimeSessionState {
        session_id: "root".to_string(),
        turn_index: 1,
        ..RuntimeSessionState::default()
    };
    let changed = RuntimeCommit::persisted_state(&changed_state, &[]);
    let changed_hash = changed.turn_commit_hash().expect("changed commit hash");
    let err = store
        .commit_runtime_state(changed.clearing_completed_turn(
            RuntimeTurnCompletion::substrate_native("root", "provider-turn", changed_hash),
        ))
        .await
        .expect_err("same provider turn id with a different commit hash must conflict");
    assert!(matches!(err, StoreError::RuntimeTurnCommitConflict { .. }));
}

async fn active_lease_fences_competing_claims(store: Arc<dyn RuntimePersistence>) {
    let turns = store
        .embedded_durable_turn_store()
        .expect("embedded durable turn");
    turns
        .claim_runtime_turn_lease("root", "turn-active", "owner-a", 60_000)
        .await
        .expect("lease");
    let conflict = turns
        .claim_runtime_turn_lease("root", "turn-active", "owner-b", 60_000)
        .await;
    assert!(
        matches!(conflict, Err(StoreError::RuntimeTurnLeaseConflict { .. })),
        "an active lease must fence a competing owner"
    );
}

async fn superseded_lease_cannot_write_or_clear(store: Arc<dyn RuntimePersistence>) {
    let turns = store
        .embedded_durable_turn_store()
        .expect("embedded durable turn");
    let old = turns
        .claim_runtime_turn_lease("root", "turn-superseded", "owner-a", 0)
        .await
        .expect("old lease");
    let current = turns
        .claim_runtime_turn_lease("root", "turn-superseded", "owner-b", 60_000)
        .await
        .expect("new lease");

    let stale_save = turns
        .save_runtime_effect_outcome(&old, effect_record("root", "turn-superseded", "stale"))
        .await;
    assert!(
        matches!(stale_save, Err(StoreError::RuntimeTurnLeaseExpired { .. })),
        "a superseded lease must not write"
    );

    turns
        .abandon_runtime_turn_lease(&old)
        .await
        .expect("a stale abandon is ignored");
    let conflict = turns
        .claim_runtime_turn_lease("root", "turn-superseded", "owner-c", 60_000)
        .await;
    assert!(
        matches!(conflict, Err(StoreError::RuntimeTurnLeaseConflict { .. })),
        "a stale abandon must not release the live lease"
    );
    turns
        .save_runtime_effect_outcome(
            &current,
            effect_record("root", "turn-superseded", "current"),
        )
        .await
        .expect("the current owner can still write");
}

async fn renewed_lease_survives_original_expiry(store: Arc<dyn RuntimePersistence>) {
    let turns = store
        .embedded_durable_turn_store()
        .expect("embedded durable turn");
    let lease = turns
        .claim_runtime_turn_lease("root", "turn-renew", "owner-a", 20)
        .await
        .expect("lease");
    let renewed = turns
        .renew_runtime_turn_lease(&lease, 60_000)
        .await
        .expect("renew lease");
    tokio::time::sleep(std::time::Duration::from_millis(30)).await;

    turns
        .save_runtime_effect_outcome(&renewed, effect_record("root", "turn-renew", "renewed"))
        .await
        .expect("a renewed lease can write after the original TTL would have expired");
}

async fn abandon_releases_owner_and_preserves_journal(store: Arc<dyn RuntimePersistence>) {
    let turns = store
        .embedded_durable_turn_store()
        .expect("embedded durable turn");
    let lease = turns
        .claim_runtime_turn_lease("root", "turn-abandon", "owner-a", 60_000)
        .await
        .expect("lease");
    let record = effect_record("root", "turn-abandon", "effect-a");
    turns
        .save_runtime_effect_outcome(&lease, record.clone())
        .await
        .expect("save journal");

    turns
        .abandon_runtime_turn_lease(&lease)
        .await
        .expect("abandon lease");

    assert!(
        store
            .embedded_durable_turn_store()
            .expect("embedded durable turn")
            .load_runtime_effect_outcome("root", "turn-abandon", &record.replay_key)
            .await
            .expect("load journal")
            .is_some(),
        "abandon must preserve the journal for a resuming owner"
    );
    turns
        .claim_runtime_turn_lease("root", "turn-abandon", "owner-b", 60_000)
        .await
        .expect("a new owner can claim an abandoned turn");
}

async fn stale_final_commit_rejects_and_preserves_resume(store: Arc<dyn RuntimePersistence>) {
    let turns = store
        .embedded_durable_turn_store()
        .expect("embedded durable turn");
    let old = turns
        .claim_runtime_turn_lease("root", "turn-stale", "owner-a", 20)
        .await
        .expect("old lease");
    let record = effect_record("root", "turn-stale", "current");
    turns
        .save_runtime_effect_outcome(&old, record.clone())
        .await
        .expect("save journal");
    tokio::time::sleep(std::time::Duration::from_millis(30)).await;
    let current = turns
        .claim_runtime_turn_lease("root", "turn-stale", "owner-b", 60_000)
        .await
        .expect("new lease");

    let state = RuntimeSessionState {
        session_id: "root".to_string(),
        ..RuntimeSessionState::default()
    };
    let commit = RuntimeCommit::persisted_state(&state, &[]);
    let commit_hash = commit.turn_commit_hash().expect("turn commit hash");
    let err = store
        .commit_runtime_state(
            commit.clearing_completed_turn(RuntimeTurnCompletion::from_lease(&old, commit_hash)),
        )
        .await
        .expect_err("a stale final commit must fail");
    assert!(matches!(err, StoreError::RuntimeTurnLeaseExpired { .. }));
    assert!(
        store
            .embedded_durable_turn_store()
            .expect("embedded durable turn")
            .load_runtime_effect_outcome("root", "turn-stale", &record.replay_key)
            .await
            .expect("load journal")
            .is_some(),
        "a rejected final commit must preserve the journal"
    );
    assert!(
        store
            .load_session(SessionReadScope::FullGraph)
            .await
            .expect("load session")
            .is_none(),
        "a rejected commit must not persist session state"
    );
    turns
        .save_runtime_effect_outcome(&current, effect_record("root", "turn-stale", "after"))
        .await
        .expect("the current owner can still write");
}

async fn expired_final_commit_rejects_and_preserves_resume(store: Arc<dyn RuntimePersistence>) {
    let turns = store
        .embedded_durable_turn_store()
        .expect("embedded durable turn");
    let lease = turns
        .claim_runtime_turn_lease("root", "turn-expired", "owner-a", 20)
        .await
        .expect("lease");
    let record = effect_record("root", "turn-expired", "effect");
    turns
        .save_runtime_effect_outcome(&lease, record.clone())
        .await
        .expect("save journal");
    tokio::time::sleep(std::time::Duration::from_millis(30)).await;

    let state = RuntimeSessionState {
        session_id: "root".to_string(),
        ..RuntimeSessionState::default()
    };
    let commit = RuntimeCommit::persisted_state(&state, &[]);
    let commit_hash = commit.turn_commit_hash().expect("turn commit hash");
    let err = store
        .commit_runtime_state(
            commit.clearing_completed_turn(RuntimeTurnCompletion::from_lease(&lease, commit_hash)),
        )
        .await
        .expect_err("an expired final commit must fail");
    assert!(matches!(err, StoreError::RuntimeTurnLeaseExpired { .. }));
    assert!(
        store
            .embedded_durable_turn_store()
            .expect("embedded durable turn")
            .load_runtime_effect_outcome("root", "turn-expired", &record.replay_key)
            .await
            .expect("load journal")
            .is_some()
    );
    assert!(
        store
            .load_session(SessionReadScope::FullGraph)
            .await
            .expect("load session")
            .is_none()
    );
}

// ---------------------------------------------------------------------------
// AttachmentStore conformance
// ---------------------------------------------------------------------------

use crate::{
    AttachmentStore, AttachmentStoreError, AttachmentStorePersistence, DurabilityTier,
    LashlangArtifactStore,
};
use lash_sansio::{AttachmentCreateMeta, ImageMediaType, MediaType};

/// Run the full [`AttachmentStore`] conformance suite against the backend
/// produced by `make`. `make` must return a fresh, empty store on each call.
/// `expected_persistence` is the tier this backend declares (`Ephemeral` for
/// in-memory, `Durable` for file/SQLite-backed).
pub fn attachment_store<F>(make: F, expected_persistence: AttachmentStorePersistence)
where
    F: Fn() -> Arc<dyn AttachmentStore>,
{
    attachment_put_get_round_trips_bytes_and_meta(make());
    attachment_is_content_addressed(make());
    attachment_get_unknown_is_not_found(make());
    attachment_reports_declared_persistence(make(), expected_persistence);
}

/// Run the full [`AttachmentStore`] suite plus durable reopen checks.
pub fn attachment_store_reopenable<F>(make: F, expected_persistence: AttachmentStorePersistence)
where
    F: Fn() -> ReopenableAttachmentStore,
{
    attachment_store(|| make().open, expected_persistence);
    attachment_store_survives_reopen(make());
}

fn attachment_meta() -> AttachmentCreateMeta {
    AttachmentCreateMeta::new(
        MediaType::Image(ImageMediaType::Png),
        Some(7),
        Some(11),
        Some("pixel".to_string()),
    )
}

fn attachment_put_get_round_trips_bytes_and_meta(store: Arc<dyn AttachmentStore>) {
    let bytes = vec![1u8, 2, 3, 4, 5];
    let reference = store
        .put(bytes.clone(), attachment_meta())
        .expect("put attachment");
    let stored = store.get(&reference.id).expect("get attachment");

    assert_eq!(stored.bytes, bytes, "bytes must round-trip unchanged");
    assert_eq!(stored.meta.id, reference.id);
    assert_eq!(stored.meta.byte_len, bytes.len() as u64);
    assert_eq!(
        stored.meta.media_type,
        MediaType::Image(ImageMediaType::Png)
    );
    assert_eq!(stored.meta.width, Some(7));
    assert_eq!(stored.meta.height, Some(11));
    assert_eq!(stored.meta.label.as_deref(), Some("pixel"));
}

fn attachment_is_content_addressed(store: Arc<dyn AttachmentStore>) {
    let first = store
        .put(vec![9u8, 9, 9], attachment_meta())
        .expect("put first");
    let same = store
        .put(vec![9u8, 9, 9], attachment_meta())
        .expect("put identical bytes");
    let different = store
        .put(vec![9u8, 9, 8], attachment_meta())
        .expect("put different bytes");

    assert_eq!(
        first.id, same.id,
        "identical bytes must map to the same content-addressed id"
    );
    assert_ne!(
        first.id, different.id,
        "different bytes must map to different ids"
    );
}

fn attachment_get_unknown_is_not_found(store: Arc<dyn AttachmentStore>) {
    let err = store
        .get(&AttachmentId::new("sha256:does-not-exist"))
        .expect_err("get of an unknown id must fail");
    assert!(
        matches!(err, AttachmentStoreError::NotFound(_)),
        "unknown id must map to NotFound, got {err:?}"
    );
}

fn attachment_reports_declared_persistence(
    store: Arc<dyn AttachmentStore>,
    expected: AttachmentStorePersistence,
) {
    assert_eq!(
        store.persistence(),
        expected,
        "persistence tier must match the backend's declared durability"
    );
}

fn attachment_store_survives_reopen(factory: ReopenableAttachmentStore) {
    let reference = factory
        .open
        .put(vec![4u8, 3, 2, 1], attachment_meta())
        .expect("put attachment before reopen");
    let reopened = factory
        .reopen
        .get(&reference.id)
        .expect("get attachment after reopen");
    assert_eq!(reopened.bytes, vec![4u8, 3, 2, 1]);
    assert_eq!(reopened.meta.id, reference.id);
    assert_eq!(reopened.meta.byte_len, 4);
}

// ---------------------------------------------------------------------------
// LashlangArtifactStore conformance
// ---------------------------------------------------------------------------

/// Run the full [`LashlangArtifactStore`] conformance suite against the backend
/// produced by `make`. `make` must return a fresh, empty store on each call.
/// `expected_tier` is the tier this backend declares (`Inline` for in-memory,
/// `Durable` for SQLite-backed).
pub fn lashlang_artifact_store<F>(make: F, expected_tier: DurabilityTier)
where
    F: Fn() -> Arc<dyn LashlangArtifactStore>,
{
    artifact_put_get_round_trips(make());
    artifact_get_unknown_is_none(make());
    artifact_reports_declared_tier(make(), expected_tier);
}

/// Run the full [`LashlangArtifactStore`] suite plus durable reopen checks.
pub fn lashlang_artifact_store_reopenable<F>(make: F, expected_tier: DurabilityTier)
where
    F: Fn() -> ReopenableLashlangArtifactStore,
{
    lashlang_artifact_store(|| make().open, expected_tier);
    lashlang_artifact_store_survives_reopen(make());
}

fn sample_artifact() -> lashlang::ModuleArtifact {
    let program = lashlang::parse("process echo(value: str) { finish value }")
        .expect("sample lashlang module parses");
    lashlang::ModuleArtifact::from_program(program).expect("module artifact builds")
}

fn artifact_put_get_round_trips(store: Arc<dyn LashlangArtifactStore>) {
    let artifact = sample_artifact();
    store
        .put_module_artifact(&artifact)
        .expect("put module artifact");
    let loaded = store
        .get_module_artifact(&artifact.module_ref)
        .expect("get module artifact")
        .expect("artifact present after put");

    assert_eq!(loaded.module_ref, artifact.module_ref);
    assert_eq!(loaded.required_surface_ref, artifact.required_surface_ref);
    assert_eq!(loaded.exports, artifact.exports);
    assert_eq!(
        loaded.to_store_bytes().expect("re-encode loaded artifact"),
        artifact
            .to_store_bytes()
            .expect("re-encode source artifact"),
        "stored artifact must round-trip byte-identically"
    );
}

fn artifact_get_unknown_is_none(store: Arc<dyn LashlangArtifactStore>) {
    let unknown = sample_artifact().module_ref;
    let result = store
        .get_module_artifact(&unknown)
        .expect("get of an unknown ref must not error");
    assert!(
        result.is_none(),
        "an unknown module ref must return Ok(None), not a backend error"
    );
}

fn artifact_reports_declared_tier(store: Arc<dyn LashlangArtifactStore>, expected: DurabilityTier) {
    assert_eq!(
        store.durability_tier(),
        expected,
        "durability tier must match the backend"
    );
}

fn lashlang_artifact_store_survives_reopen(factory: ReopenableLashlangArtifactStore) {
    let artifact = sample_artifact();
    factory
        .open
        .put_module_artifact(&artifact)
        .expect("put module artifact before reopen");
    let loaded = factory
        .reopen
        .get_module_artifact(&artifact.module_ref)
        .expect("get module artifact after reopen")
        .expect("artifact present after reopen");
    assert_eq!(loaded.module_ref, artifact.module_ref);
    assert_eq!(loaded.required_surface_ref, artifact.required_surface_ref);
    assert_eq!(loaded.exports, artifact.exports);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn in_memory_attachment_store_satisfies_conformance() {
        attachment_store(
            || Arc::new(crate::InMemoryAttachmentStore::new()) as Arc<dyn AttachmentStore>,
            AttachmentStorePersistence::Ephemeral,
        );
    }

    #[test]
    fn in_memory_lashlang_artifact_store_satisfies_conformance() {
        lashlang_artifact_store(
            || {
                Arc::new(crate::InMemoryLashlangArtifactStore::new())
                    as Arc<dyn LashlangArtifactStore>
            },
            DurabilityTier::Inline,
        );
    }

    #[test]
    fn module_artifact_rejects_corrupted_store_bytes() {
        let err = lashlang::ModuleArtifact::from_store_bytes(b"not an artifact")
            .expect_err("corrupted artifact bytes must be rejected");
        assert!(matches!(err, lashlang::ModuleArtifactError::Codec(_)));
    }
}
