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

use std::sync::{Arc, Mutex};

use crate::{
    AgentFrameReason, AgentFrameRecord, AttachmentId, AttachmentIntent, CausalRef, DeliveryPolicy,
    EffectHost, EffectScope, MergeKey, ModelSpec, PluginSessionSnapshot, ProtocolEvent,
    ProtocolTurnOptions, QueuedWorkBatch, QueuedWorkBatchDraft, QueuedWorkClaimBoundary,
    QueuedWorkPayload, RuntimeCommit, RuntimeEffectCommand, RuntimeEffectController,
    RuntimeEffectControllerError, RuntimeEffectEnvelope, RuntimeEffectKind,
    RuntimeEffectLocalExecutor, RuntimeEffectOutcome, RuntimeInvocation, RuntimePersistence,
    RuntimeScope, RuntimeSessionState, RuntimeSubject, RuntimeTurnCommitStamp,
    ScopedEffectController, SessionMeta, SessionNodePayload, SessionNodeRecord, SessionPolicy,
    SessionReadScope, SessionRelation, SlotPolicy, StoreError, TokenLedgerEntry, TokenUsage,
    ToolState, TurnInput,
};
use crate::{
    LashSchema, ProcessAwaitOutput, ProcessEventAppendRequest, ProcessEventSemanticsSpec,
    ProcessEventType, ProcessHandleDescriptor, ProcessInput, ProcessLeaseCompletion,
    ProcessProvenance, ProcessRegistration, ProcessRegistry, ProcessScope, ProcessScopeId,
    ProcessTerminalState, ProcessValueSelector, ProcessWakeDedupeKey, ProcessWakeDelivery,
    ProcessWakeSpec,
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

/// One scope selected by an [`EffectHost`] and one effect envelope executed
/// through the scoped controller.
#[derive(Clone, Debug)]
pub struct RecordingEffectHostRecord {
    pub runtime_scope: RuntimeScope,
    pub effect_scope: EffectScope,
    pub effect_id: String,
    pub effect_kind: RuntimeEffectKind,
    pub replay_key: Option<String>,
    pub envelope_hash: String,
}

#[derive(Clone)]
struct RecordingEffectHostController {
    effect_scope: EffectScope,
    records: Arc<Mutex<Vec<RecordingEffectHostRecord>>>,
}

#[async_trait::async_trait]
impl RuntimeEffectController for RecordingEffectHostController {
    async fn execute_effect(
        &self,
        envelope: RuntimeEffectEnvelope,
        _local_executor: RuntimeEffectLocalExecutor<'_>,
    ) -> Result<RuntimeEffectOutcome, RuntimeEffectControllerError> {
        let envelope_hash = envelope.stable_hash()?;
        self.records
            .lock()
            .expect("effect host records")
            .push(RecordingEffectHostRecord {
                runtime_scope: envelope.invocation.scope.clone(),
                effect_scope: self.effect_scope.clone(),
                effect_id: envelope
                    .invocation
                    .effect_id()
                    .expect("effect invocation")
                    .to_string(),
                effect_kind: envelope.command.kind(),
                replay_key: envelope.invocation.replay_key().map(ToOwned::to_owned),
                envelope_hash,
            });
        match envelope.command {
            RuntimeEffectCommand::Sleep { .. } => Ok(RuntimeEffectOutcome::Sleep),
            command => Err(RuntimeEffectControllerError::new(
                "recording_effect_host_unsupported_command",
                format!(
                    "recording effect host cannot synthesize {} outcomes",
                    command.kind().as_str()
                ),
            )),
        }
    }
}

/// Test fixture that records every selected [`EffectScope`] and every effect
/// envelope executed through the returned scoped controller.
#[derive(Clone, Default)]
pub struct RecordingEffectHost {
    selected_scopes: Arc<Mutex<Vec<EffectScope>>>,
    records: Arc<Mutex<Vec<RecordingEffectHostRecord>>>,
}

impl RecordingEffectHost {
    pub fn selected_scopes(&self) -> Vec<EffectScope> {
        self.selected_scopes
            .lock()
            .expect("selected effect scopes")
            .clone()
    }

    pub fn records(&self) -> Vec<RecordingEffectHostRecord> {
        self.records.lock().expect("effect host records").clone()
    }

    fn scoped_for<'run>(
        &self,
        scope: EffectScope,
    ) -> Result<ScopedEffectController<'run>, crate::RuntimeError> {
        self.selected_scopes
            .lock()
            .expect("selected effect scopes")
            .push(scope.clone());
        ScopedEffectController::shared(
            Arc::new(RecordingEffectHostController {
                effect_scope: scope.clone(),
                records: Arc::clone(&self.records),
            }),
            scope,
        )
    }
}

impl EffectHost for RecordingEffectHost {
    fn scoped<'run>(
        &'run self,
        scope: EffectScope,
    ) -> Result<ScopedEffectController<'run>, crate::RuntimeError> {
        self.scoped_for(scope)
    }

    fn scoped_static(
        &self,
        scope: EffectScope,
    ) -> Result<Option<ScopedEffectController<'static>>, crate::RuntimeError> {
        Ok(Some(self.scoped_for(scope)?))
    }
}

/// Run the generic [`EffectHost`] scope-factory conformance suite.
///
/// This suite checks the deployment-level contract: effect scopes must carry
/// stable semantic identity, empty ids must fail loudly, and hosts that expose
/// a static scoped controller must preserve the same scope metadata. It does
/// not assert durability; that remains a property of each implementation.
/// Substrate-native hosts such as Restate complete the in-flight contract at
/// this effect-host/controller boundary; [`RuntimePersistence`] remains the
/// committed-state store contract, not a workflow-history contract.
pub async fn effect_host<F>(make: F)
where
    F: Fn() -> Arc<dyn EffectHost>,
{
    effect_host_preserves_scope_metadata(make()).await;
    effect_host_rejects_missing_scope_ids(make()).await;
    effect_host_static_scope_preserves_metadata_when_available(make()).await;
}

async fn effect_host_preserves_scope_metadata(host: Arc<dyn EffectHost>) {
    let scope = EffectScope::queue_drain("session-1", "drain-1");
    let scoped = host.scoped(scope.clone()).expect("queue drain scope");
    assert_eq!(
        scoped.effect_scope(),
        &scope,
        "scoped controller must retain the selected semantic scope"
    );
    assert_eq!(scoped.scope_id(), "drain-1");
    assert_eq!(scoped.turn_id(), None);

    let turn_scope = EffectScope::turn("session-1", "turn-1");
    let scoped_turn = host.scoped(turn_scope.clone()).expect("turn scope");
    assert_eq!(scoped_turn.effect_scope(), &turn_scope);
    assert_eq!(scoped_turn.scope_id(), "turn-1");
    assert_eq!(scoped_turn.turn_id(), Some("turn-1"));
}

async fn effect_host_rejects_missing_scope_ids(host: Arc<dyn EffectHost>) {
    let invalid_scopes = [
        EffectScope::turn("", "turn"),
        EffectScope::turn("session", ""),
        EffectScope::process(""),
        EffectScope::host_event("session", ""),
        EffectScope::queue_drain("session", ""),
        EffectScope::cron("job", ""),
        EffectScope::session_delete(""),
        EffectScope::runtime_operation(""),
    ];

    for scope in invalid_scopes {
        let err = match host.scoped(scope) {
            Ok(_) => panic!("invalid effect scope must be rejected"),
            Err(err) => err,
        };
        assert_eq!(
            err.code,
            crate::RuntimeErrorCode::MissingEffectScopeId,
            "invalid scope ids must fail with the stable missing-scope code"
        );
    }
}

async fn effect_host_static_scope_preserves_metadata_when_available(host: Arc<dyn EffectHost>) {
    let scope = EffectScope::runtime_operation("static-runtime-op");
    let Some(scoped) = host
        .scoped_static(scope.clone())
        .expect("static scope factory")
    else {
        return;
    };
    assert_eq!(scoped.effect_scope(), &scope);
    assert_eq!(scoped.scope_id(), "static-runtime-op");
}

/// Run the concurrent recorded-effect replay conformance case for a
/// handler-scoped durable controller.
///
/// The first pass starts two recorded effects concurrently and intentionally
/// lets the second finish before the first. After `start_replay`, the same
/// effects are requested in the opposite order with local executors that fail
/// if called. A compliant controller returns the recorded outcomes by
/// `replay.key`, independent of local completion/request ordering.
#[cfg(any(test, feature = "testing"))]
pub async fn effect_controller_concurrent_replay_deterministic(
    controller: &dyn RuntimeEffectController,
    start_replay: impl FnOnce(),
) {
    let slow = replay_conformance_exec_envelope("effect-slow");
    let fast = replay_conformance_exec_envelope("effect-fast");
    let completion_order = Arc::new(Mutex::new(Vec::new()));
    let barrier = Arc::new(tokio::sync::Barrier::new(2));
    let release_slow = Arc::new(tokio::sync::Notify::new());

    let first_pass = tokio::time::timeout(std::time::Duration::from_secs(2), async {
        tokio::join!(
            controller.execute_effect(
                slow.clone(),
                replay_conformance_recording_executor(
                    "effect-slow",
                    Arc::clone(&barrier),
                    Arc::clone(&release_slow),
                    Arc::clone(&completion_order),
                ),
            ),
            controller.execute_effect(
                fast.clone(),
                replay_conformance_recording_executor(
                    "effect-fast",
                    Arc::clone(&barrier),
                    Arc::clone(&release_slow),
                    Arc::clone(&completion_order),
                ),
            ),
        )
    })
    .await
    .expect("concurrent first-pass effects must both enter their local executors");
    let slow_first = first_pass.0.expect("slow first pass");
    let fast_first = first_pass.1.expect("fast first pass");
    assert_replay_conformance_exec_marker(slow_first, "effect-slow");
    assert_replay_conformance_exec_marker(fast_first, "effect-fast");
    assert_eq!(
        completion_order
            .lock()
            .expect("completion order")
            .as_slice(),
        &["effect-fast".to_string(), "effect-slow".to_string()],
        "first pass must prove local completion order can differ from effect request order"
    );

    start_replay();
    let replay_local_calls = Arc::new(Mutex::new(Vec::new()));
    let replay_pass = tokio::time::timeout(std::time::Duration::from_secs(2), async {
        tokio::join!(
            controller.execute_effect(
                fast,
                replay_conformance_failing_executor(Arc::clone(&replay_local_calls)),
            ),
            controller.execute_effect(
                slow,
                replay_conformance_failing_executor(Arc::clone(&replay_local_calls)),
            ),
        )
    })
    .await
    .expect("concurrent replay effects must resolve from host history");
    let fast_replay = replay_pass.0.expect("fast replay");
    let slow_replay = replay_pass.1.expect("slow replay");
    assert_replay_conformance_exec_marker(fast_replay, "effect-fast");
    assert_replay_conformance_exec_marker(slow_replay, "effect-slow");
    assert!(
        replay_local_calls
            .lock()
            .expect("replay local calls")
            .is_empty(),
        "replay must return recorded outcomes without invoking local executors"
    );
}

#[cfg(any(test, feature = "testing"))]
fn replay_conformance_exec_envelope(effect_id: &'static str) -> RuntimeEffectEnvelope {
    RuntimeEffectEnvelope::new(
        RuntimeInvocation::effect(
            RuntimeScope::for_turn(
                "effect-conformance-session",
                "effect-conformance-turn",
                7,
                0,
            ),
            effect_id,
            RuntimeEffectKind::ExecCode,
            format!("effect-conformance:effect-conformance-turn:{effect_id}"),
        ),
        RuntimeEffectCommand::ExecCode {
            code: format!("emit {effect_id}"),
        },
    )
}

#[cfg(any(test, feature = "testing"))]
fn replay_conformance_recording_executor(
    effect_id: &'static str,
    barrier: Arc<tokio::sync::Barrier>,
    release_slow: Arc<tokio::sync::Notify>,
    completion_order: Arc<Mutex<Vec<String>>>,
) -> RuntimeEffectLocalExecutor<'static> {
    RuntimeEffectLocalExecutor::testing(move |envelope| async move {
        assert_eq!(envelope.invocation.effect_id(), Some(effect_id));
        barrier.wait().await;
        if effect_id == "effect-slow" {
            release_slow.notified().await;
        } else {
            completion_order
                .lock()
                .expect("completion order")
                .push(effect_id.to_string());
            release_slow.notify_one();
        }
        if effect_id == "effect-slow" {
            completion_order
                .lock()
                .expect("completion order")
                .push(effect_id.to_string());
        }
        Ok(replay_conformance_exec_outcome(effect_id))
    })
}

#[cfg(any(test, feature = "testing"))]
fn replay_conformance_failing_executor(
    replay_local_calls: Arc<Mutex<Vec<String>>>,
) -> RuntimeEffectLocalExecutor<'static> {
    RuntimeEffectLocalExecutor::testing(move |envelope| async move {
        replay_local_calls
            .lock()
            .expect("replay local calls")
            .push(envelope.invocation.effect_id().unwrap_or("").to_string());
        Err(RuntimeEffectControllerError::new(
            "conformance_replay_local_executor_called",
            "recorded replay must not invoke local effect execution",
        ))
    })
}

#[cfg(any(test, feature = "testing"))]
fn replay_conformance_exec_outcome(effect_id: &str) -> RuntimeEffectOutcome {
    RuntimeEffectOutcome::ExecCode {
        result: Ok(crate::ExecResponse {
            observations: Vec::new(),
            observation_truncation: Vec::new(),
            tool_calls: Vec::new(),
            images: Vec::new(),
            printed_images: Vec::new(),
            error: None,
            duration_ms: 0,
            terminal_finish: Some(serde_json::json!(effect_id)),
        }),
    }
}

#[cfg(any(test, feature = "testing"))]
fn assert_replay_conformance_exec_marker(outcome: RuntimeEffectOutcome, expected: &str) {
    let RuntimeEffectOutcome::ExecCode { result } = outcome else {
        panic!("expected exec-code effect outcome");
    };
    let response = result.expect("exec-code response");
    assert_eq!(
        response.terminal_finish,
        Some(serde_json::json!(expected)),
        "replayed outcome must come from the matching replay key"
    );
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
    wake_semantic_events_without_target_fail_without_persisting(make()).await;
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

async fn wake_semantic_events_without_target_fail_without_persisting(
    registry: Arc<dyn ProcessRegistry>,
) {
    registry
        .register_process(
            registration("proc-missing-wake-target")
                .with_extra_event_types([wake_event_type("process.wake")]),
        )
        .await
        .expect("register");

    let err = registry
        .append_event(
            "proc-missing-wake-target",
            ProcessEventAppendRequest::new(
                "process.wake",
                serde_json::json!({
                    "message": "target missing",
                    "wake_input": "Process wake: target missing",
                }),
            )
            .with_replay_key("wake:missing-target"),
        )
        .await
        .expect_err("wake-semantic event without target scope must fail");
    assert!(
        err.to_string().contains("without a wake target scope"),
        "unexpected missing-target error: {err}"
    );
    assert!(
        registry
            .events_after("proc-missing-wake-target", 0)
            .await
            .expect("events after failed append")
            .is_empty(),
        "failed wake append must not persist a partial process event"
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
/// Covers the durability crown jewels owned by the store: optimistic head CAS,
/// session binding, checkpoint/usage hydration, queued work claim fencing,
/// attachment manifest intent/commit/GC reconciliation, session metadata,
/// tombstone/GC behavior, and idempotent final turn commit stamps.
/// Effect-host workflow history is deliberately outside this suite.
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
    queued_work_cancel_removes_only_unclaimed_batches(make()).await;
    queued_work_claims_respect_boundaries_renewal_and_abandon(make()).await;
    queued_work_respects_availability_limits_exclusivity_reclaim_and_sessions(make()).await;
    queued_work_join_groups_by_delivery_policy_and_merge_key(make()).await;
    queued_work_completion_is_lease_guarded(make()).await;
    queued_wake_delivery_is_source_key_idempotent_and_claimed_once(make()).await;
    queue_completion_and_turn_commit_stamp_are_atomic(make()).await;
    session_metadata_round_trips(make()).await;
    tombstone_vacuum_and_gc_are_minimally_consistent(make()).await;
    final_commit_stamp_is_idempotent_and_conflicts_on_changed_hash(make()).await;
}

/// Build a queued turn-input draft for backend conformance tests.
pub fn queued_turn_input_draft(
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

fn queued_draft(
    session_id: &str,
    text: &str,
    delivery_policy: DeliveryPolicy,
    slot_policy: SlotPolicy,
) -> QueuedWorkBatchDraft {
    queued_turn_input_draft(session_id, text, delivery_policy, slot_policy)
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
        | QueuedWorkPayload::SessionCommand { .. } => None,
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

async fn queued_work_cancel_removes_only_unclaimed_batches(store: Arc<dyn RuntimePersistence>) {
    let cancellable = store
        .enqueue_queued_work(queued_draft(
            "root",
            "cancel me",
            DeliveryPolicy::AfterCurrentTurnCommit,
            SlotPolicy::Exclusive,
        ))
        .await
        .expect("enqueue cancellable batch");
    let cancelled = store
        .cancel_queued_work_batch("root", &cancellable.batch_id)
        .await
        .expect("cancel unclaimed batch")
        .expect("unclaimed batch is returned");
    assert_eq!(cancelled.batch_id, cancellable.batch_id);
    assert_eq!(queued_batch_text(&cancelled), Some("cancel me"));
    assert!(
        store
            .list_queued_work("root")
            .await
            .expect("list after cancellation")
            .is_empty(),
        "cancelled batches must be removed from the durable queue"
    );

    let claimed = store
        .enqueue_queued_work(queued_draft(
            "root",
            "claimed",
            DeliveryPolicy::AfterCurrentTurnCommit,
            SlotPolicy::Exclusive,
        ))
        .await
        .expect("enqueue claimed batch");
    let claim = store
        .claim_ready_queued_work("root", "owner", QueuedWorkClaimBoundary::Idle, 60_000, 1)
        .await
        .expect("claim batch")
        .expect("claim exists");
    assert_eq!(claim.batches[0].batch_id, claimed.batch_id);
    assert!(
        store
            .list_pending_queued_work("root")
            .await
            .expect("list pending during active claim")
            .is_empty(),
        "active claims must disappear from user-editable queue snapshots"
    );
    assert_eq!(
        store
            .list_queued_work("root")
            .await
            .expect("raw durable list during active claim")
            .len(),
        1,
        "claimed batches remain durable until their claim is completed"
    );
    assert!(
        store
            .cancel_queued_work_batch("root", &claimed.batch_id)
            .await
            .expect("cancel active claim")
            .is_none(),
        "actively claimed batches must not be cancelled"
    );
    store
        .abandon_queued_work_claim(&claim)
        .await
        .expect("abandon claim");
    assert_eq!(
        store
            .list_pending_queued_work("root")
            .await
            .expect("list pending after abandoned claim")
            .len(),
        1,
        "abandoned claims become user-editable queue work again"
    );
    assert!(
        store
            .cancel_queued_work_batch("root", &claimed.batch_id)
            .await
            .expect("cancel abandoned claim")
            .is_some(),
        "abandoned batches become cancellable again"
    );
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

async fn queue_completion_and_turn_commit_stamp_are_atomic(store: Arc<dyn RuntimePersistence>) {
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
    let state = RuntimeSessionState {
        session_id: "root".to_string(),
        turn_index: 41,
        ..RuntimeSessionState::default()
    };
    let base_commit = RuntimeCommit::persisted_state(&state, &[]);
    let commit_hash = base_commit.turn_commit_hash().expect("turn commit hash");
    let turn_commit = RuntimeTurnCommitStamp::new("root", "turn-atomic", commit_hash.clone());
    let mut stale_queue_completion = claim.completion();
    stale_queue_completion.lease_token.push_str(":stale");
    let err = store
        .commit_runtime_state(
            base_commit
                .clone()
                .with_turn_commit(turn_commit.clone())
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

    let first = store
        .commit_runtime_state(
            base_commit
                .clone()
                .with_turn_commit(turn_commit.clone())
                .completing_queue_claim(claim.completion()),
        )
        .await
        .expect("valid final commit clears queue and records the turn stamp atomically");
    let retry = store
        .commit_runtime_state(
            base_commit
                .with_turn_commit(RuntimeTurnCommitStamp::new(
                    "root",
                    "turn-atomic",
                    commit_hash,
                ))
                .completing_queue_claim(claim.completion()),
        )
        .await
        .expect("same final turn commit stamp retries idempotently");
    assert_eq!(retry.head_revision, first.head_revision);
    assert_eq!(retry.checkpoint_ref, first.checkpoint_ref);
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

async fn queued_wake_delivery_is_source_key_idempotent_and_claimed_once(
    store: Arc<dyn RuntimePersistence>,
) {
    let wake = ProcessWakeDelivery {
        wake_id: "wake-1".to_string(),
        target_session_id: "root".to_string(),
        target_scope_id: ProcessScopeId::new("session:root"),
        process_id: "process-1".to_string(),
        sequence: 7,
        event_type: "process.wake".to_string(),
        event_invocation: RuntimeInvocation {
            scope: RuntimeScope::new("root"),
            subject: RuntimeSubject::ProcessEvent {
                process_id: "process-1".to_string(),
                sequence: 7,
                event_type: "process.wake".to_string(),
            },
            caused_by: None,
            replay: None,
        },
        process_caused_by: None,
        dedupe_key: "wake-dedupe-1".to_string(),
        input: "wake payload".to_string(),
        created_at_ms: 1,
    };
    let first = store
        .enqueue_queued_work(crate::process_wake_batch_draft(wake.clone()))
        .await
        .expect("enqueue wake");
    let replay = store
        .enqueue_queued_work(crate::process_wake_batch_draft(wake))
        .await
        .expect("replay wake enqueue");
    assert_eq!(
        first.batch_id, replay.batch_id,
        "wake source-key replay must return the original queued batch"
    );
    assert_eq!(
        store
            .list_queued_work("root")
            .await
            .expect("list queued wakes")
            .len(),
        1,
        "replayed wake must not create a second queued delivery"
    );

    let claim = store
        .claim_ready_queued_work(
            "root",
            "wake-owner",
            QueuedWorkClaimBoundary::Idle,
            60_000,
            10,
        )
        .await
        .expect("claim wake")
        .expect("wake claim");
    assert_eq!(claim.batches.len(), 1);
    assert_eq!(claim.batches[0].items.len(), 1);
    assert!(matches!(
        claim.batches[0].items[0].payload,
        QueuedWorkPayload::ProcessWake { .. }
    ));
    let state = RuntimeSessionState {
        session_id: "root".to_string(),
        ..RuntimeSessionState::default()
    };
    store
        .commit_runtime_state(
            RuntimeCommit::persisted_state(&state, &[]).completing_queue_claim(claim.completion()),
        )
        .await
        .expect("wake delivery completion commits");
    assert!(
        store
            .list_queued_work("root")
            .await
            .expect("list after wake completion")
            .is_empty(),
        "completed wake delivery must be removed exactly once"
    );
}

async fn final_commit_stamp_is_idempotent_and_conflicts_on_changed_hash(
    store: Arc<dyn RuntimePersistence>,
) {
    let state = RuntimeSessionState {
        session_id: "root".to_string(),
        ..RuntimeSessionState::default()
    };
    let commit = RuntimeCommit::persisted_state(&state, &[]);
    let turn_commit_hash = commit.turn_commit_hash().expect("turn commit hash");
    let commit = commit.with_turn_commit(RuntimeTurnCommitStamp::new(
        "root",
        "provider-turn",
        turn_commit_hash.clone(),
    ));

    let first = store
        .commit_runtime_state(commit.clone())
        .await
        .expect("host-replayed final commit does not require a Lash lease");
    let retry = store
        .commit_runtime_state(commit)
        .await
        .expect("same host-replayed final commit retries idempotently");
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
        .commit_runtime_state(changed.with_turn_commit(RuntimeTurnCommitStamp::new(
            "root",
            "provider-turn",
            changed_hash,
        )))
        .await
        .expect_err("same provider turn id with a different commit hash must conflict");
    assert!(matches!(err, StoreError::RuntimeTurnCommitConflict { .. }));
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
/// in-memory, `Durable` for file/Turso-backed).
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
/// `Durable` for Turso-backed).
pub async fn lashlang_artifact_store<F>(make: F, expected_tier: DurabilityTier)
where
    F: Fn() -> Arc<dyn LashlangArtifactStore>,
{
    artifact_put_get_round_trips(make()).await;
    artifact_get_unknown_is_none(make()).await;
    artifact_reports_declared_tier(make(), expected_tier);
}

/// Run the full [`LashlangArtifactStore`] suite plus durable reopen checks.
pub async fn lashlang_artifact_store_reopenable<F>(make: F, expected_tier: DurabilityTier)
where
    F: Fn() -> ReopenableLashlangArtifactStore,
{
    lashlang_artifact_store(|| make().open, expected_tier).await;
    lashlang_artifact_store_survives_reopen(make()).await;
}

fn sample_artifact() -> lashlang::ModuleArtifact {
    let program = lashlang::parse("process echo(value: str) { finish value }")
        .expect("sample lashlang module parses");
    lashlang::ModuleArtifact::from_program(program).expect("module artifact builds")
}

async fn artifact_put_get_round_trips(store: Arc<dyn LashlangArtifactStore>) {
    let artifact = sample_artifact();
    store
        .put_module_artifact(&artifact)
        .await
        .expect("put module artifact");
    let loaded = store
        .get_module_artifact(&artifact.module_ref)
        .await
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

async fn artifact_get_unknown_is_none(store: Arc<dyn LashlangArtifactStore>) {
    let unknown = sample_artifact().module_ref;
    let result = store
        .get_module_artifact(&unknown)
        .await
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

async fn lashlang_artifact_store_survives_reopen(factory: ReopenableLashlangArtifactStore) {
    let artifact = sample_artifact();
    factory
        .open
        .put_module_artifact(&artifact)
        .await
        .expect("put module artifact before reopen");
    let loaded = factory
        .reopen
        .get_module_artifact(&artifact.module_ref)
        .await
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

    #[tokio::test]
    async fn in_memory_lashlang_artifact_store_satisfies_conformance() {
        lashlang_artifact_store(
            || {
                Arc::new(crate::InMemoryLashlangArtifactStore::new())
                    as Arc<dyn LashlangArtifactStore>
            },
            DurabilityTier::Inline,
        )
        .await;
    }

    #[tokio::test]
    async fn inline_effect_host_satisfies_conformance() {
        effect_host(|| Arc::new(crate::InlineEffectHost::default())).await;
    }

    #[tokio::test]
    async fn recording_effect_host_records_selected_scope_and_envelope() {
        let host = RecordingEffectHost::default();
        let scope = EffectScope::host_event("session-1", "button-1");
        let scoped = host.scoped(scope.clone()).expect("scoped controller");
        let envelope = RuntimeEffectEnvelope::new(
            crate::RuntimeInvocation::effect(
                RuntimeScope::new("session-1"),
                "sleep-effect",
                RuntimeEffectKind::Sleep,
                "button-1:sleep-effect",
            ),
            RuntimeEffectCommand::Sleep { duration_ms: 0 },
        );

        let outcome = scoped
            .controller()
            .execute_effect(envelope, RuntimeEffectLocalExecutor::unavailable())
            .await
            .expect("execute sleep");

        assert!(matches!(outcome, RuntimeEffectOutcome::Sleep));
        assert_eq!(host.selected_scopes(), vec![scope.clone()]);
        let records = host.records();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].effect_scope, scope);
        assert_eq!(records[0].runtime_scope, RuntimeScope::new("session-1"));
        assert_eq!(records[0].effect_id, "sleep-effect");
        assert_eq!(records[0].effect_kind, RuntimeEffectKind::Sleep);
        assert_eq!(
            records[0].replay_key.as_deref(),
            Some("button-1:sleep-effect")
        );
    }

    #[test]
    fn module_artifact_rejects_corrupted_store_bytes() {
        let err = lashlang::ModuleArtifact::from_store_bytes(b"not an artifact")
            .expect_err("corrupted artifact bytes must be rejected");
        assert!(matches!(err, lashlang::ModuleArtifactError::Codec(_)));
    }
}
