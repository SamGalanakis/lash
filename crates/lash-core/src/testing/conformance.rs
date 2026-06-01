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
    AgentFrameReason, AgentFrameRecord, AttachmentId, AttachmentIntent, DeliveryPolicy, MergeKey,
    ModelSpec, PluginSessionSnapshot, ProtocolTurnOptions, QueuedWorkBatchDraft,
    QueuedWorkClaimBoundary, QueuedWorkPayload, RUNTIME_EFFECT_JOURNAL_SCHEMA_VERSION,
    RuntimeCommit, RuntimeEffectJournalRecord, RuntimeEffectKind, RuntimeEffectOutcome,
    RuntimePersistence, RuntimeSessionState, RuntimeTurnCompletion, SessionPolicy,
    SessionReadScope, SlotPolicy, StoreError, TokenLedgerEntry, TokenUsage, ToolState, TurnInput,
};
use crate::{
    LashSchema, ProcessAwaitOutput, ProcessEventAppendRequest, ProcessEventSemanticsSpec,
    ProcessEventType, ProcessHandleDescriptor, ProcessInput, ProcessLeaseCompletion,
    ProcessRegistration, ProcessRegistry, ProcessScope, ProcessTerminalState, ProcessValueSelector,
    ProcessWakeDedupeKey, ProcessWakeSpec,
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
    list_non_terminal_excludes_terminal_processes(make()).await;
    list_live_handle_grants_excludes_terminal_history(make()).await;
    active_process_lease_fences_competing_owner(make()).await;
    superseded_process_lease_cannot_renew(make()).await;
    renewed_process_lease_survives_original_expiry(make()).await;
    completed_lease_releases_and_reclaim_bumps_fencing(make()).await;
    stale_lease_completion_cannot_release_live_lease(make()).await;
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
    queued_work_completion_is_lease_guarded(make()).await;
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
    // A successful `get` already re-ran `verify()` internally; round-tripping
    // the bytes again must reproduce the identical store encoding.
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

    // The corrupt-bytes rejection path is exercised here rather than in the
    // shared suite: only the in-memory store exposes a raw-bytes injection
    // seam (`put_raw_module_artifact_bytes`); durable backends re-verify on
    // read through the same `ModuleArtifact::from_store_bytes` path.
    #[test]
    fn in_memory_artifact_store_rejects_corrupted_bytes_on_read() {
        let store = crate::InMemoryLashlangArtifactStore::new();
        let artifact = sample_artifact();
        store.put_raw_module_artifact_bytes(
            artifact.module_ref.clone(),
            b"not an artifact".to_vec(),
        );
        let err = store
            .get_module_artifact(&artifact.module_ref)
            .expect_err("corrupted stored bytes must be rejected on read");
        assert!(
            matches!(err, lashlang::ArtifactStoreError::Decode(_)),
            "tampered bytes must surface a decode error, got {err:?}"
        );
    }
}
