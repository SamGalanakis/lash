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
    AgentFrameReason, AgentFrameRecord, ModelSpec, PluginSessionSnapshot, ProtocolTurnOptions,
    RUNTIME_EFFECT_JOURNAL_SCHEMA_VERSION, RuntimeCommit, RuntimeEffectJournalRecord,
    RuntimeEffectKind, RuntimeEffectOutcome, RuntimePersistence, RuntimeSessionState,
    RuntimeTurnCompletion, SessionPolicy, SessionReadScope, StoreError, TokenLedgerEntry,
    TokenUsage, ToolState,
};
use crate::{
    LashSchema, ProcessAwaitOutput, ProcessEventAppendRequest, ProcessEventSemanticsSpec,
    ProcessEventType, ProcessHandleDescriptor, ProcessInput, ProcessRegistration, ProcessRegistry,
    ProcessScope, ProcessTerminalState, ProcessValueSelector, ProcessWakeDedupeKey,
    ProcessWakeSpec,
};

/// Run the full [`ProcessRegistry`] conformance suite against the backend
/// produced by `make`. `make` must return a fresh, empty registry on each call.
pub async fn process_registry<F>(make: F)
where
    F: Fn() -> Arc<dyn ProcessRegistry>,
{
    registration_is_idempotent_and_hash_conflicts_fail(make()).await;
    validates_custom_events_and_materializes_wakes(make()).await;
    keyed_events_materialize_idempotent_wakes(make()).await;
    terminal_and_cancel_events_require_keys(make()).await;
    await_reads_terminal_materialized_output(make()).await;
    transfer_handle_grants_moves_addressability(make()).await;
    multiple_sessions_can_hold_grants(make()).await;
    processes_can_exist_with_zero_grants(make()).await;
    delete_session_revokes_handles_by_session(make()).await;
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
            .and_then(|record| record.terminal.map(|terminal| terminal.state)),
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

/// Run the [`RuntimePersistence`] durability conformance suite against the
/// backend produced by `make`. `make` must return a fresh, empty,
/// single-session store on each call.
///
/// Covers the durability crown jewels: optimistic head CAS, session binding,
/// checkpoint/usage hydration, lease fencing (claim/renew/abandon/supersede/
/// expire), lease-guarded journal writes, replay-key journal idempotency, and
/// atomic final commit that clears the journal only under a live lease (else
/// preserves resume state). In-flight `RuntimeTurnCheckpoint` round-tripping —
/// whose hash validation is backend-specific — is exercised per backend.
pub async fn runtime_persistence<F>(make: F)
where
    F: Fn() -> Arc<dyn RuntimePersistence>,
{
    commit_increments_head_and_round_trips_agent_frames(make()).await;
    commit_rejects_a_different_session_id(make()).await;
    load_hydrates_checkpoint_and_usage(make()).await;
    journal_is_idempotent_and_cleared_on_final_commit(make()).await;
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

async fn journal_is_idempotent_and_cleared_on_final_commit(store: Arc<dyn RuntimePersistence>) {
    let lease = store
        .claim_runtime_turn_lease("root", "turn-1", "test-owner", 60_000)
        .await
        .expect("lease");
    let record = effect_record("root", "turn-1", "sleep");
    store
        .save_runtime_effect_outcome(&lease, record.clone())
        .await
        .expect("save journal");
    let loaded = store
        .load_runtime_effect_outcome("root", "turn-1", &record.replay_key)
        .await
        .expect("load journal")
        .expect("journal record");
    assert_eq!(loaded.envelope_hash, record.envelope_hash);
    // Replaying the same key is idempotent (overwrites, no duplicate row).
    store
        .save_runtime_effect_outcome(&lease, record.clone())
        .await
        .expect("replay save is idempotent");

    let state = RuntimeSessionState {
        session_id: "root".to_string(),
        ..RuntimeSessionState::default()
    };
    let commit = RuntimeCommit::persisted_state(&state, &[])
        .clearing_completed_turn(RuntimeTurnCompletion::from_lease(&lease));
    store
        .commit_runtime_state(commit)
        .await
        .expect("final commit clears turn");

    assert!(
        store
            .load_runtime_effect_outcome("root", "turn-1", &record.replay_key)
            .await
            .expect("load after clear")
            .is_none(),
        "a final commit under a live lease must clear the journal"
    );
    assert!(
        store
            .load_runtime_turn_checkpoint("root", "turn-1")
            .await
            .expect("load checkpoint")
            .is_none()
    );
}

async fn active_lease_fences_competing_claims(store: Arc<dyn RuntimePersistence>) {
    store
        .claim_runtime_turn_lease("root", "turn-active", "owner-a", 60_000)
        .await
        .expect("lease");
    let conflict = store
        .claim_runtime_turn_lease("root", "turn-active", "owner-b", 60_000)
        .await;
    assert!(
        matches!(conflict, Err(StoreError::RuntimeTurnLeaseConflict { .. })),
        "an active lease must fence a competing owner"
    );
}

async fn superseded_lease_cannot_write_or_clear(store: Arc<dyn RuntimePersistence>) {
    let old = store
        .claim_runtime_turn_lease("root", "turn-superseded", "owner-a", 0)
        .await
        .expect("old lease");
    let current = store
        .claim_runtime_turn_lease("root", "turn-superseded", "owner-b", 60_000)
        .await
        .expect("new lease");

    let stale_save = store
        .save_runtime_effect_outcome(&old, effect_record("root", "turn-superseded", "stale"))
        .await;
    assert!(
        matches!(stale_save, Err(StoreError::RuntimeTurnLeaseExpired { .. })),
        "a superseded lease must not write"
    );

    store
        .abandon_runtime_turn_lease(&old)
        .await
        .expect("a stale abandon is ignored");
    let conflict = store
        .claim_runtime_turn_lease("root", "turn-superseded", "owner-c", 60_000)
        .await;
    assert!(
        matches!(conflict, Err(StoreError::RuntimeTurnLeaseConflict { .. })),
        "a stale abandon must not release the live lease"
    );
    store
        .save_runtime_effect_outcome(
            &current,
            effect_record("root", "turn-superseded", "current"),
        )
        .await
        .expect("the current owner can still write");
}

async fn renewed_lease_survives_original_expiry(store: Arc<dyn RuntimePersistence>) {
    let lease = store
        .claim_runtime_turn_lease("root", "turn-renew", "owner-a", 20)
        .await
        .expect("lease");
    let renewed = store
        .renew_runtime_turn_lease(&lease, 60_000)
        .await
        .expect("renew lease");
    tokio::time::sleep(std::time::Duration::from_millis(30)).await;

    store
        .save_runtime_effect_outcome(&renewed, effect_record("root", "turn-renew", "renewed"))
        .await
        .expect("a renewed lease can write after the original TTL would have expired");
}

async fn abandon_releases_owner_and_preserves_journal(store: Arc<dyn RuntimePersistence>) {
    let lease = store
        .claim_runtime_turn_lease("root", "turn-abandon", "owner-a", 60_000)
        .await
        .expect("lease");
    let record = effect_record("root", "turn-abandon", "effect-a");
    store
        .save_runtime_effect_outcome(&lease, record.clone())
        .await
        .expect("save journal");

    store
        .abandon_runtime_turn_lease(&lease)
        .await
        .expect("abandon lease");

    assert!(
        store
            .load_runtime_effect_outcome("root", "turn-abandon", &record.replay_key)
            .await
            .expect("load journal")
            .is_some(),
        "abandon must preserve the journal for a resuming owner"
    );
    store
        .claim_runtime_turn_lease("root", "turn-abandon", "owner-b", 60_000)
        .await
        .expect("a new owner can claim an abandoned turn");
}

async fn stale_final_commit_rejects_and_preserves_resume(store: Arc<dyn RuntimePersistence>) {
    let old = store
        .claim_runtime_turn_lease("root", "turn-stale", "owner-a", 20)
        .await
        .expect("old lease");
    let record = effect_record("root", "turn-stale", "current");
    store
        .save_runtime_effect_outcome(&old, record.clone())
        .await
        .expect("save journal");
    tokio::time::sleep(std::time::Duration::from_millis(30)).await;
    let current = store
        .claim_runtime_turn_lease("root", "turn-stale", "owner-b", 60_000)
        .await
        .expect("new lease");

    let state = RuntimeSessionState {
        session_id: "root".to_string(),
        ..RuntimeSessionState::default()
    };
    let err = store
        .commit_runtime_state(
            RuntimeCommit::persisted_state(&state, &[])
                .clearing_completed_turn(RuntimeTurnCompletion::from_lease(&old)),
        )
        .await
        .expect_err("a stale final commit must fail");
    assert!(matches!(err, StoreError::RuntimeTurnLeaseExpired { .. }));
    assert!(
        store
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
    store
        .save_runtime_effect_outcome(&current, effect_record("root", "turn-stale", "after"))
        .await
        .expect("the current owner can still write");
}

async fn expired_final_commit_rejects_and_preserves_resume(store: Arc<dyn RuntimePersistence>) {
    let lease = store
        .claim_runtime_turn_lease("root", "turn-expired", "owner-a", 20)
        .await
        .expect("lease");
    let record = effect_record("root", "turn-expired", "effect");
    store
        .save_runtime_effect_outcome(&lease, record.clone())
        .await
        .expect("save journal");
    tokio::time::sleep(std::time::Duration::from_millis(30)).await;

    let state = RuntimeSessionState {
        session_id: "root".to_string(),
        ..RuntimeSessionState::default()
    };
    let err = store
        .commit_runtime_state(
            RuntimeCommit::persisted_state(&state, &[])
                .clearing_completed_turn(RuntimeTurnCompletion::from_lease(&lease)),
        )
        .await
        .expect_err("an expired final commit must fail");
    assert!(matches!(err, StoreError::RuntimeTurnLeaseExpired { .. }));
    assert!(
        store
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
