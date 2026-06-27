//! [`RuntimePersistence`] conformance: head CAS, checkpoint hydration,
//! queued-work claim fencing, attachment manifest, and turn-commit stamps.

use super::*;

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
    pub durability_tier: crate::DurabilityTier,
}

impl RuntimePersistenceConformance {
    pub const fn persistent_attachment_manifest(durability_tier: crate::DurabilityTier) -> Self {
        Self {
            attachment_manifest: AttachmentManifestConformance::Persistent,
            durability_tier,
        }
    }

    pub const fn noop_attachment_manifest(durability_tier: crate::DurabilityTier) -> Self {
        Self {
            attachment_manifest: AttachmentManifestConformance::Noop,
            durability_tier,
        }
    }
}

impl Default for RuntimePersistenceConformance {
    fn default() -> Self {
        Self {
            attachment_manifest: AttachmentManifestConformance::Persistent,
            durability_tier: crate::DurabilityTier::Durable,
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
    runtime_persistence_with_options(
        make,
        RuntimePersistenceConformance::persistent_attachment_manifest(
            crate::DurabilityTier::Inline,
        ),
    )
    .await;
}

/// Run the full [`RuntimePersistence`] suite plus durable reopen checks.
pub async fn runtime_persistence_reopenable<F>(make: F)
where
    F: Fn() -> ReopenableRuntimePersistence,
{
    runtime_persistence_with_options(
        || make().open,
        RuntimePersistenceConformance::persistent_attachment_manifest(
            crate::DurabilityTier::Durable,
        ),
    )
    .await;
    runtime_persistence_survives_reopen(make()).await;
}

pub async fn runtime_persistence_with_options<F>(make: F, options: RuntimePersistenceConformance)
where
    F: Fn() -> Arc<dyn RuntimePersistence>,
{
    runtime_persistence_reports_declared_durability(make(), options.durability_tier).await;
    commit_increments_head_and_round_trips_agent_frames(make()).await;
    commit_rejects_a_different_session_id(make()).await;
    load_hydrates_checkpoint_and_usage(make()).await;
    active_path_read_scope_selects_only_requested_ancestry(make()).await;
    session_execution_lease_contract(make()).await;
    session_execution_lease_reclaim_contract(make()).await;
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
    queued_work_exact_claim_uses_selected_batch_ids(make()).await;
    queued_work_classes_gate_command_and_turn_claims(make()).await;
    queued_work_claims_respect_boundaries_renewal_and_abandon(make()).await;
    queued_work_respects_membership_limits_exclusivity_reclaim_and_sessions(make()).await;
    queued_work_join_groups_by_delivery_policy_and_merge_key(make()).await;
    queued_work_completion_is_lease_guarded(make()).await;
    queued_wake_delivery_is_source_key_idempotent_and_claimed_once(make()).await;
    queue_completion_and_turn_commit_stamp_are_atomic(make()).await;
    pending_turn_inputs_source_keys_order_cancel_and_cross_session(make()).await;
    pending_turn_input_claims_reclaim_complete_and_fence(make()).await;
    pending_active_turn_inputs_defer_unaccepted_once_on_interrupt(make()).await;
    session_metadata_round_trips(make()).await;
    tombstone_vacuum_and_gc_are_minimally_consistent(make()).await;
    final_commit_stamp_is_idempotent_and_conflicts_on_changed_hash(make()).await;
}

/// Build a queued process-wake draft for backend conformance tests.
pub fn queued_process_wake_draft(
    session_id: &str,
    text: &str,
    delivery_policy: DeliveryPolicy,
    slot_policy: SlotPolicy,
) -> QueuedWorkBatchDraft {
    let wake = ProcessWakeDelivery {
        wake_id: format!("wake:{session_id}:{text}"),
        target_session_id: session_id.to_string(),
        target_scope_id: SessionScopeId::new(format!("session:{session_id}")),
        process_id: format!("process:{text}"),
        sequence: 1,
        event_type: "process.wake".to_string(),
        event_invocation: RuntimeInvocation {
            scope: RuntimeScope::new(session_id),
            subject: RuntimeSubject::ProcessEvent {
                process_id: format!("process:{text}"),
                sequence: 1,
                event_type: "process.wake".to_string(),
            },
            caused_by: None,
            replay: None,
        },
        process_caused_by: None,
        dedupe_key: format!("wake:{session_id}:{text}:1"),
        input: text.to_string(),
        created_at_ms: 1,
    };
    QueuedWorkBatchDraft::new(
        session_id,
        delivery_policy,
        slot_policy,
        vec![QueuedWorkPayload::process_wake(wake)],
    )
}

fn queued_draft(
    session_id: &str,
    text: &str,
    delivery_policy: DeliveryPolicy,
    slot_policy: SlotPolicy,
) -> QueuedWorkBatchDraft {
    queued_process_wake_draft(session_id, text, delivery_policy, slot_policy)
}

fn queued_session_command_draft(session_id: &str, reason: &str) -> QueuedWorkBatchDraft {
    QueuedWorkBatchDraft::new(
        session_id,
        DeliveryPolicy::EarliestSafeBoundary,
        SlotPolicy::Exclusive,
        vec![QueuedWorkPayload::session_command(
            crate::SessionCommand::RefreshToolCatalog {
                reason: reason.to_string(),
            },
        )],
    )
}

fn queued_batch_text(batch: &QueuedWorkBatch) -> Option<&str> {
    let payload = batch.items.first().map(|item| &item.payload)?;
    match payload {
        QueuedWorkPayload::ProcessWake { wake } => Some(wake.input.as_str()),
        QueuedWorkPayload::SessionCommand { .. } => None,
    }
}

fn pending_next_turn_input_draft(session_id: &str, text: &str) -> crate::PendingTurnInputDraft {
    crate::PendingTurnInputDraft::new(
        session_id,
        crate::TurnInputIngress::NextTurn,
        crate::TurnInput::text(text),
    )
}

fn pending_active_turn_input_draft(
    session_id: &str,
    turn_id: &str,
    min_boundary: crate::TurnInputCheckpointBoundary,
    text: &str,
) -> crate::PendingTurnInputDraft {
    crate::PendingTurnInputDraft::new(
        session_id,
        crate::TurnInputIngress::active_turn(turn_id, min_boundary),
        crate::TurnInput::text(text),
    )
}

fn pending_input_text(input: &crate::PendingTurnInput) -> Option<&str> {
    match input.input.items.first()? {
        crate::InputItem::Text { text } => Some(text.as_str()),
        crate::InputItem::ImageRef { .. } => None,
    }
}

fn lease_owner(owner_id: &str) -> crate::LeaseOwnerIdentity {
    crate::LeaseOwnerIdentity::opaque(owner_id, format!("{owner_id}:incarnation"))
}

fn local_lease_owner(
    owner_id: &str,
    incarnation_id: &str,
    host_id: &str,
    boot_id: &str,
    pid: u32,
    process_start: &str,
) -> crate::LeaseOwnerIdentity {
    crate::LeaseOwnerIdentity {
        owner_id: owner_id.to_string(),
        incarnation_id: incarnation_id.to_string(),
        liveness: crate::LeaseOwnerLiveness::local_process_for_test(
            host_id,
            boot_id,
            pid,
            process_start,
        ),
    }
}

async fn claim_session_execution_lease_for_test(
    store: &Arc<dyn RuntimePersistence>,
    session_id: &str,
    owner_id: &str,
) -> crate::SessionExecutionLease {
    let owner = lease_owner(owner_id);
    store
        .try_claim_session_execution_lease(session_id, &owner, 60_000)
        .await
        .expect("claim session execution lease")
        .acquired()
        .expect("session execution lease is free")
}

async fn release_session_execution_lease_for_test(
    store: &Arc<dyn RuntimePersistence>,
    lease: &crate::SessionExecutionLease,
) {
    store
        .release_session_execution_lease(&lease.completion())
        .await
        .expect("release session execution lease");
}

async fn commit_runtime_state_for_test(
    store: &Arc<dyn RuntimePersistence>,
    commit: RuntimeCommit,
    owner_id: &str,
) -> Result<crate::store::RuntimeCommitResult, StoreError> {
    let session_id = commit.session_id.clone();
    let lease = claim_session_execution_lease_for_test(store, &session_id, owner_id).await;
    store
        .commit_runtime_state(
            commit
                .with_session_execution_lease(lease.fence())
                .releasing_session_execution_lease(lease.completion()),
        )
        .await
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
            model: ModelSpec::from_token_limits("gpt-5.4-mini", None, 200_000, None)
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
    let custom_reason = AgentFrameReason::new("plan_mode");
    state.append_agent_frame(AgentFrameRecord::new(
        "frame-2".to_string(),
        "root".to_string(),
        Some(previous_frame_id),
        custom_reason.clone(),
        None,
        assignment,
        ProtocolTurnOptions::default(),
    ));
    state.set_execution_state_snapshot(Some(b"frame-vm".to_vec()));

    commit_runtime_state_for_test(
        &store,
        RuntimeCommit::persisted_state(&state, &[]),
        "commit-round-trip",
    )
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
    assert_eq!(current.reason, custom_reason);
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
    commit_runtime_state_for_test(
        &store,
        RuntimeCommit::persisted_state(&alpha, &[]),
        "bind-alpha",
    )
    .await
    .expect("first commit binds the session");
    let beta = RuntimeSessionState {
        session_id: "beta".to_string(),
        ..RuntimeSessionState::default()
    };
    let result = commit_runtime_state_for_test(
        &store,
        RuntimeCommit::persisted_state(&beta, &[]),
        "bind-beta",
    )
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

    commit_runtime_state_for_test(
        &store,
        RuntimeCommit::persisted_state(&state, &[usage]),
        "hydrate",
    )
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

async fn runtime_persistence_reports_declared_durability(
    store: Arc<dyn RuntimePersistence>,
    expected_tier: crate::DurabilityTier,
) {
    assert_eq!(
        store.durability_tier(),
        expected_tier,
        "runtime persistence conformance must pin the backend's declared durability tier"
    );
}

async fn session_execution_lease_contract(store: Arc<dyn RuntimePersistence>) {
    let first = claim_session_execution_lease_for_test(&store, "root", "owner-a").await;
    let owner_a = lease_owner("owner-a");
    let owner_a_next = crate::LeaseOwnerIdentity::opaque("owner-a", "owner-a:next-incarnation");
    let owner_b = lease_owner("owner-b");
    let owner_c = lease_owner("owner-c");
    let owner_expired = lease_owner("owner-expired");
    let reentered = store
        .try_claim_session_execution_lease("root", &owner_a, 120_000)
        .await
        .expect("same incarnation may re-enter live session lease")
        .acquired()
        .expect("same incarnation receives existing session lease");
    assert_eq!(reentered.lease_token, first.lease_token);
    assert_eq!(reentered.fencing_token, first.fencing_token);
    assert!(reentered.expires_at_epoch_ms >= first.expires_at_epoch_ms);
    assert!(
        matches!(
            store
                .try_claim_session_execution_lease("root", &owner_a_next, 60_000)
                .await
                .expect("try same owner next incarnation"),
            crate::SessionExecutionLeaseClaimOutcome::Busy { .. }
        ),
        "a live session execution lease must exclude the same owner in a different incarnation"
    );
    assert!(
        matches!(
            store
                .try_claim_session_execution_lease("root", &owner_b, 60_000)
                .await
                .expect("try concurrent session lease"),
            crate::SessionExecutionLeaseClaimOutcome::Busy { .. }
        ),
        "a live session execution lease must exclude concurrent owners"
    );
    let renewed = store
        .renew_session_execution_lease(&reentered.fence(), 120_000)
        .await
        .expect("renew live session lease");
    assert_eq!(renewed.lease_token, first.lease_token);
    assert!(renewed.expires_at_epoch_ms >= reentered.expires_at_epoch_ms);

    let mut stale_fence = reentered.fence();
    stale_fence.lease_token.push_str(":stale");
    let err = store
        .renew_session_execution_lease(&stale_fence, 60_000)
        .await
        .expect_err("stale session lease renew must fail");
    assert!(matches!(
        err,
        StoreError::SessionExecutionLeaseExpired { .. }
    ));
    store
        .release_session_execution_lease(&crate::SessionExecutionLeaseCompletion {
            session_id: first.session_id.clone(),
            owner: first.owner.clone(),
            lease_token: format!("{}:stale", first.lease_token),
            fencing_token: first.fencing_token,
        })
        .await
        .expect("stale release is fenced and idempotent");
    assert!(
        matches!(
            store
                .try_claim_session_execution_lease("root", &owner_b, 60_000)
                .await
                .expect("try after stale release"),
            crate::SessionExecutionLeaseClaimOutcome::Busy { .. }
        ),
        "stale release must not clear the live lease"
    );
    release_session_execution_lease_for_test(&store, &renewed).await;
    let second = claim_session_execution_lease_for_test(&store, "root", "owner-b").await;
    assert!(
        second.fencing_token > first.fencing_token,
        "reclaimed session leases must advance the fencing token"
    );
    store
        .release_session_execution_lease(&first.completion())
        .await
        .expect("old release is idempotent");
    assert!(
        matches!(
            store
                .try_claim_session_execution_lease("root", &owner_c, 60_000)
                .await
                .expect("try after old release"),
            crate::SessionExecutionLeaseClaimOutcome::Busy { .. }
        ),
        "old release must not clear a newer lease"
    );
    release_session_execution_lease_for_test(&store, &second).await;

    let expired = store
        .try_claim_session_execution_lease("root", &owner_expired, 0)
        .await
        .expect("claim expiring lease")
        .acquired()
        .expect("expiring lease");
    let reclaimed = claim_session_execution_lease_for_test(&store, "root", "owner-reclaim").await;
    assert!(reclaimed.fencing_token > expired.fencing_token);
    release_session_execution_lease_for_test(&store, &reclaimed).await;

    let state = RuntimeSessionState {
        session_id: "root".to_string(),
        ..RuntimeSessionState::default()
    };
    let err = store
        .commit_runtime_state(RuntimeCommit::persisted_state(&state, &[]))
        .await
        .expect_err("session-head commits require a live session lease");
    assert!(matches!(
        err,
        StoreError::SessionExecutionLeaseExpired { .. }
    ));

    let commit_lease = claim_session_execution_lease_for_test(&store, "root", "commit-owner").await;
    store
        .commit_runtime_state(
            RuntimeCommit::persisted_state(&state, &[])
                .with_session_execution_lease(commit_lease.fence())
                .releasing_session_execution_lease(commit_lease.completion()),
        )
        .await
        .expect("commit under live session lease");
    let after_commit = claim_session_execution_lease_for_test(&store, "root", "after-commit").await;
    release_session_execution_lease_for_test(&store, &after_commit).await;

    let turn_state = RuntimeSessionState {
        session_id: "root".to_string(),
        turn_index: 1,
        ..RuntimeSessionState::default()
    };
    let turn_commit = RuntimeCommit::persisted_state(&turn_state, &[]);
    let turn_hash = turn_commit.turn_commit_hash().expect("turn hash");
    let turn_commit = turn_commit.with_turn_commit(RuntimeTurnCommitStamp::new(
        "root",
        "lease-replay-turn",
        turn_hash,
    ));
    let turn_lease = claim_session_execution_lease_for_test(&store, "root", "turn-owner").await;
    let first_result = store
        .commit_runtime_state(
            turn_commit
                .clone()
                .with_session_execution_lease(turn_lease.fence())
                .releasing_session_execution_lease(turn_lease.completion()),
        )
        .await
        .expect("first final commit under session lease");
    let replay = store
        .commit_runtime_state(turn_commit)
        .await
        .expect("idempotent replay returns without live session lease");
    assert_eq!(replay.head_revision, first_result.head_revision);

    let batch = store
        .enqueue_queued_work(queued_draft(
            "root",
            "fenced queue",
            DeliveryPolicy::EarliestSafeBoundary,
            SlotPolicy::Exclusive,
        ))
        .await
        .expect("enqueue fenced queue work");
    let err = store
        .claim_ready_queued_work(
            "root",
            &commit_lease.fence(),
            &lease_owner("queue-owner"),
            QueuedWorkClaimBoundary::Idle,
            60_000,
            1,
        )
        .await
        .expect_err("queued-work claims require a live session lease");
    assert!(matches!(
        err,
        StoreError::SessionExecutionLeaseExpired { .. }
    ));
    let queue_lease = claim_session_execution_lease_for_test(&store, "root", "queue-owner").await;
    let claim = store
        .claim_ready_queued_work(
            "root",
            &queue_lease.fence(),
            &lease_owner("queue-owner"),
            QueuedWorkClaimBoundary::Idle,
            60_000,
            1,
        )
        .await
        .expect("claim fenced queue work")
        .expect("queue work claim");
    assert_eq!(claim.batches[0].batch_id, batch.batch_id);
    release_session_execution_lease_for_test(&store, &queue_lease).await;
}

async fn session_execution_lease_reclaim_contract(store: Arc<dyn RuntimePersistence>) {
    let pid = std::process::id();
    let dead_holder = local_lease_owner(
        "dead-holder",
        "dead-holder:incarnation",
        "host-a",
        "boot-a",
        pid,
        "not-the-current-process-start",
    );
    let claimant = local_lease_owner(
        "claimant",
        "claimant:incarnation",
        "host-a",
        "boot-a",
        pid,
        "claimant-start",
    );
    let holder = store
        .try_claim_session_execution_lease("reclaim-dead", &dead_holder, 60_000)
        .await
        .expect("claim dead-holder lease")
        .acquired()
        .expect("dead-holder lease acquired");
    assert!(
        matches!(
            store
                .try_claim_session_execution_lease("reclaim-dead", &claimant, 60_000)
                .await
                .expect("try claimant against dead holder"),
            crate::SessionExecutionLeaseClaimOutcome::Busy { .. }
        ),
        "plain claim must report busy before the caller performs fenced reclaim"
    );
    let reclaimed = store
        .reclaim_session_execution_lease("reclaim-dead", &claimant, &holder.fence(), 60_000)
        .await
        .expect("reclaim dead holder")
        .acquired()
        .expect("dead holder is reclaimable before ttl");
    assert!(
        reclaimed.fencing_token > holder.fencing_token,
        "fenced reclaim must advance the fencing token"
    );
    let stale_reclaim = store
        .reclaim_session_execution_lease(
            "reclaim-dead",
            &local_lease_owner(
                "late-claimant",
                "late-claimant:incarnation",
                "host-a",
                "boot-a",
                pid,
                "late-claimant-start",
            ),
            &holder.fence(),
            60_000,
        )
        .await
        .expect("stale observed-holder reclaim");
    assert!(
        matches!(
            stale_reclaim,
            crate::SessionExecutionLeaseClaimOutcome::Busy { .. }
        ),
        "a stale observed holder must not clear the newer lease"
    );
    release_session_execution_lease_for_test(&store, &reclaimed).await;

    let race_holder = store
        .try_claim_session_execution_lease("reclaim-race", &dead_holder, 60_000)
        .await
        .expect("claim race holder")
        .acquired()
        .expect("race holder acquired");
    let race_fence = race_holder.fence();
    let barrier = Arc::new(tokio::sync::Barrier::new(3));
    let left_store = Arc::clone(&store);
    let right_store = Arc::clone(&store);
    let left_barrier = Arc::clone(&barrier);
    let right_barrier = Arc::clone(&barrier);
    let left_fence = race_fence.clone();
    let right_fence = race_fence.clone();
    let left_claimant = local_lease_owner(
        "race-left",
        "race-left:incarnation",
        "host-a",
        "boot-a",
        pid,
        "race-left-start",
    );
    let right_claimant = local_lease_owner(
        "race-right",
        "race-right:incarnation",
        "host-a",
        "boot-a",
        pid,
        "race-right-start",
    );
    let left = tokio::spawn(async move {
        left_barrier.wait().await;
        left_store
            .reclaim_session_execution_lease("reclaim-race", &left_claimant, &left_fence, 60_000)
            .await
    });
    let right = tokio::spawn(async move {
        right_barrier.wait().await;
        right_store
            .reclaim_session_execution_lease("reclaim-race", &right_claimant, &right_fence, 60_000)
            .await
    });
    barrier.wait().await;
    let left = left
        .await
        .expect("join left reclaim race")
        .expect("left reclaim race");
    let right = right
        .await
        .expect("join right reclaim race")
        .expect("right reclaim race");
    let mut race_winners = [left, right]
        .into_iter()
        .filter_map(crate::SessionExecutionLeaseClaimOutcome::acquired)
        .collect::<Vec<_>>();
    assert_eq!(
        race_winners.len(),
        1,
        "exactly one claimant may win a fenced reclaim race"
    );
    let race_winner = race_winners.pop().expect("race winner");
    assert!(race_winner.fencing_token > race_holder.fencing_token);
    release_session_execution_lease_for_test(&store, &race_winner).await;

    let cross_host_holder = store
        .try_claim_session_execution_lease("reclaim-cross-host", &dead_holder, 60_000)
        .await
        .expect("claim cross-host holder")
        .acquired()
        .expect("cross-host holder acquired");
    let cross_host_result = store
        .reclaim_session_execution_lease(
            "reclaim-cross-host",
            &local_lease_owner(
                "cross-host-claimant",
                "cross-host-claimant:incarnation",
                "host-b",
                "boot-a",
                pid,
                "claimant-start",
            ),
            &cross_host_holder.fence(),
            60_000,
        )
        .await
        .expect("cross-host reclaim");
    assert!(
        matches!(
            cross_host_result,
            crate::SessionExecutionLeaseClaimOutcome::Busy { .. }
        ),
        "dead-looking local-process holders are reclaimable only on the same host"
    );
    release_session_execution_lease_for_test(&store, &cross_host_holder).await;

    let different_boot_holder = store
        .try_claim_session_execution_lease("reclaim-different-boot", &dead_holder, 60_000)
        .await
        .expect("claim different-boot holder")
        .acquired()
        .expect("different-boot holder acquired");
    let different_boot_result = store
        .reclaim_session_execution_lease(
            "reclaim-different-boot",
            &local_lease_owner(
                "different-boot-claimant",
                "different-boot-claimant:incarnation",
                "host-a",
                "boot-b",
                pid,
                "claimant-start",
            ),
            &different_boot_holder.fence(),
            60_000,
        )
        .await
        .expect("different-boot reclaim");
    assert!(
        matches!(
            different_boot_result,
            crate::SessionExecutionLeaseClaimOutcome::Busy { .. }
        ),
        "same-host process metadata from a different boot is not proof of death"
    );
    release_session_execution_lease_for_test(&store, &different_boot_holder).await;

    let opaque_holder = store
        .try_claim_session_execution_lease("reclaim-opaque", &lease_owner("opaque-holder"), 60_000)
        .await
        .expect("claim opaque holder")
        .acquired()
        .expect("opaque holder acquired");
    let opaque_result = store
        .reclaim_session_execution_lease(
            "reclaim-opaque",
            &claimant,
            &opaque_holder.fence(),
            60_000,
        )
        .await
        .expect("opaque reclaim");
    assert!(
        matches!(
            opaque_result,
            crate::SessionExecutionLeaseClaimOutcome::Busy { .. }
        ),
        "opaque holders fall back to ttl rather than fast reclaim"
    );
    release_session_execution_lease_for_test(&store, &opaque_holder).await;

    if let Some(liveness) = crate::LeaseOwnerLiveness::current_local_process("live-host") {
        let crate::LeaseOwnerLiveness::LocalProcess {
            host_id, boot_id, ..
        } = &liveness
        else {
            unreachable!("current local-process liveness is either local or absent");
        };
        let host_id = host_id.clone();
        let boot_id = boot_id.clone();
        let live_holder_owner = crate::LeaseOwnerIdentity {
            owner_id: "live-holder".to_string(),
            incarnation_id: "live-holder:incarnation".to_string(),
            liveness,
        };
        let live_claimant = local_lease_owner(
            "live-claimant",
            "live-claimant:incarnation",
            &host_id,
            &boot_id,
            pid,
            "live-claimant-start",
        );
        let live_holder = store
            .try_claim_session_execution_lease("reclaim-live", &live_holder_owner, 60_000)
            .await
            .expect("claim live holder")
            .acquired()
            .expect("live holder acquired");
        let live_result = store
            .reclaim_session_execution_lease(
                "reclaim-live",
                &live_claimant,
                &live_holder.fence(),
                60_000,
            )
            .await
            .expect("live reclaim");
        assert!(
            matches!(
                live_result,
                crate::SessionExecutionLeaseClaimOutcome::Busy { .. }
            ),
            "a live local process holder remains busy before ttl"
        );
        release_session_execution_lease_for_test(&store, &live_holder).await;
    }
}

async fn active_path_read_scope_selects_only_requested_ancestry(
    store: Arc<dyn RuntimePersistence>,
) {
    let graph = crate::SessionGraph::from_nodes(
        vec![
            sample_session_node("root-node", None),
            sample_session_node("left-node", Some("root-node")),
            sample_session_node("left-leaf", Some("left-node")),
            sample_session_node("right-leaf", Some("root-node")),
        ],
        Some("left-leaf".to_string()),
    );
    let state = RuntimeSessionState {
        session_id: "branchy".to_string(),
        session_graph: graph,
        graph_replace_required: true,
        ..RuntimeSessionState::default()
    };
    commit_runtime_state_for_test(
        &store,
        RuntimeCommit::persisted_state(&state, &[]),
        "active-path",
    )
    .await
    .expect("commit branchy graph");

    let full = store
        .load_session(SessionReadScope::FullGraph)
        .await
        .expect("load full graph")
        .expect("full graph exists");
    assert_eq!(
        full.graph
            .nodes
            .iter()
            .map(|node| node.node_id.as_str())
            .collect::<Vec<_>>(),
        vec!["root-node", "left-node", "left-leaf", "right-leaf"],
        "FullGraph must retain every non-tombstoned branch"
    );

    let persisted_leaf_path = store
        .load_session(SessionReadScope::ActivePath { leaf_node_id: None })
        .await
        .expect("load persisted active path")
        .expect("active path exists");
    assert_eq!(
        persisted_leaf_path
            .graph
            .nodes
            .iter()
            .map(|node| node.node_id.as_str())
            .collect::<Vec<_>>(),
        vec!["root-node", "left-node", "left-leaf"],
        "ActivePath with no explicit leaf must use the persisted leaf and hide sibling branches"
    );
    assert_eq!(
        persisted_leaf_path.graph.leaf_node_id.as_deref(),
        Some("left-leaf")
    );

    let explicit_right_path = store
        .load_session(SessionReadScope::ActivePath {
            leaf_node_id: Some("right-leaf".to_string()),
        })
        .await
        .expect("load explicit active path")
        .expect("explicit active path exists");
    assert_eq!(
        explicit_right_path
            .graph
            .nodes
            .iter()
            .map(|node| node.node_id.as_str())
            .collect::<Vec<_>>(),
        vec!["root-node", "right-leaf"],
        "ActivePath with an explicit leaf must select that ancestry, not the persisted leaf"
    );
    assert_eq!(
        explicit_right_path.graph.leaf_node_id.as_deref(),
        Some("right-leaf")
    );
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
    commit_runtime_state_for_test(
        &store,
        RuntimeCommit::persisted_state(&state, &[])
            .with_committed_attachments([committed_by_runtime.clone()]),
        "attachment-manifest",
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
    let session_lease = claim_session_execution_lease_for_test(&store, "root", "owner").await;
    let claim = store
        .claim_ready_queued_work(
            "root",
            &session_lease.fence(),
            &lease_owner("owner"),
            QueuedWorkClaimBoundary::Idle,
            60_000,
            1,
        )
        .await
        .expect("claim batch")
        .expect("claim exists");
    release_session_execution_lease_for_test(&store, &session_lease).await;
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

async fn queued_work_exact_claim_uses_selected_batch_ids(store: Arc<dyn RuntimePersistence>) {
    let first = store
        .enqueue_queued_work(queued_draft(
            "root",
            "first",
            DeliveryPolicy::AfterCurrentTurnCommit,
            SlotPolicy::Exclusive,
        ))
        .await
        .expect("enqueue first batch");
    let second = store
        .enqueue_queued_work(queued_draft(
            "root",
            "second",
            DeliveryPolicy::AfterCurrentTurnCommit,
            SlotPolicy::Exclusive,
        ))
        .await
        .expect("enqueue second batch");

    let rejected_session_lease =
        claim_session_execution_lease_for_test(&store, "root", "owner").await;
    assert!(
        store
            .claim_ready_queued_work_by_batch_ids(
                "root",
                &rejected_session_lease.fence(),
                &lease_owner("owner"),
                QueuedWorkClaimBoundary::Idle,
                60_000,
                std::slice::from_ref(&second.batch_id),
            )
            .await
            .expect("claim out-of-order exact batch")
            .is_none(),
        "exact claims must not skip earlier durable queue work"
    );
    release_session_execution_lease_for_test(&store, &rejected_session_lease).await;
    assert_eq!(
        store
            .list_pending_queued_work("root")
            .await
            .expect("list after rejected exact claim")
            .iter()
            .map(|batch| batch.batch_id.as_str())
            .collect::<Vec<_>>(),
        vec![first.batch_id.as_str(), second.batch_id.as_str()]
    );

    let accepted_session_lease =
        claim_session_execution_lease_for_test(&store, "root", "owner").await;
    let claim = store
        .claim_ready_queued_work_by_batch_ids(
            "root",
            &accepted_session_lease.fence(),
            &lease_owner("owner"),
            QueuedWorkClaimBoundary::Idle,
            60_000,
            std::slice::from_ref(&first.batch_id),
        )
        .await
        .expect("claim first exact batch")
        .expect("first exact claim exists");
    release_session_execution_lease_for_test(&store, &accepted_session_lease).await;
    assert_eq!(
        claim
            .batches
            .iter()
            .map(|batch| batch.batch_id.as_str())
            .collect::<Vec<_>>(),
        vec![first.batch_id.as_str()]
    );
    assert_eq!(
        store
            .list_pending_queued_work("root")
            .await
            .expect("list pending after exact claim")
            .iter()
            .map(|batch| batch.batch_id.as_str())
            .collect::<Vec<_>>(),
        vec![second.batch_id.as_str()]
    );
}

async fn queued_work_classes_gate_command_and_turn_claims(store: Arc<dyn RuntimePersistence>) {
    let command = store
        .enqueue_queued_work(queued_session_command_draft("root", "refresh before turn"))
        .await
        .expect("enqueue command");
    let turn = store
        .enqueue_queued_work(queued_draft(
            "root",
            "user turn",
            DeliveryPolicy::AfterCurrentTurnCommit,
            SlotPolicy::Exclusive,
        ))
        .await
        .expect("enqueue turn");

    let rejected_turn_lease =
        claim_session_execution_lease_for_test(&store, "root", "turn-owner").await;
    assert!(
        store
            .claim_ready_queued_work(
                "root",
                &rejected_turn_lease.fence(),
                &lease_owner("turn-owner"),
                QueuedWorkClaimBoundary::Idle,
                60_000,
                10,
            )
            .await
            .expect("turn claim with leading command")
            .is_none(),
        "turn claims must not skip a leading session command"
    );
    release_session_execution_lease_for_test(&store, &rejected_turn_lease).await;

    let command_lease =
        claim_session_execution_lease_for_test(&store, "root", "command-owner").await;
    let command_claim = store
        .claim_leading_ready_session_command(
            "root",
            &command_lease.fence(),
            &lease_owner("command-owner"),
            60_000,
        )
        .await
        .expect("claim leading command")
        .expect("leading command claim exists");
    assert_eq!(
        command_claim
            .batches
            .iter()
            .map(|batch| batch.batch_id.as_str())
            .collect::<Vec<_>>(),
        vec![command.batch_id.as_str()]
    );
    let state = RuntimeSessionState {
        session_id: "root".to_string(),
        ..RuntimeSessionState::default()
    };
    store
        .commit_runtime_state(
            RuntimeCommit::persisted_state(&state, &[])
                .with_session_execution_lease(command_lease.fence())
                .releasing_session_execution_lease(command_lease.completion())
                .completing_queue_claim(command_claim.completion()),
        )
        .await
        .expect("complete command claim");

    let selected_turn_lease =
        claim_session_execution_lease_for_test(&store, "root", "turn-owner").await;
    let selected_turn = store
        .claim_ready_queued_work_by_batch_ids(
            "root",
            &selected_turn_lease.fence(),
            &lease_owner("turn-owner"),
            QueuedWorkClaimBoundary::Idle,
            60_000,
            std::slice::from_ref(&turn.batch_id),
        )
        .await
        .expect("claim selected turn after command")
        .expect("selected turn claim exists");
    release_session_execution_lease_for_test(&store, &selected_turn_lease).await;
    assert_eq!(selected_turn.batches[0].batch_id, turn.batch_id);

    let first_turn = store
        .enqueue_queued_work(queued_draft(
            "turn-first",
            "first turn",
            DeliveryPolicy::AfterCurrentTurnCommit,
            SlotPolicy::Exclusive,
        ))
        .await
        .expect("enqueue first turn");
    let second_command = store
        .enqueue_queued_work(queued_session_command_draft("turn-first", "later refresh"))
        .await
        .expect("enqueue later command");
    let rejected_command_lease =
        claim_session_execution_lease_for_test(&store, "turn-first", "command-owner").await;
    assert!(
        store
            .claim_leading_ready_session_command(
                "turn-first",
                &rejected_command_lease.fence(),
                &lease_owner("command-owner"),
                60_000,
            )
            .await
            .expect("claim command behind turn")
            .is_none(),
        "session commands must not jump ahead of earlier turn work"
    );
    let turn_claim = store
        .claim_ready_queued_work(
            "turn-first",
            &rejected_command_lease.fence(),
            &lease_owner("command-owner"),
            QueuedWorkClaimBoundary::Idle,
            60_000,
            10,
        )
        .await
        .expect("claim turn before later command")
        .expect("turn claim exists");
    assert_eq!(turn_claim.batches[0].batch_id, first_turn.batch_id);
    store
        .abandon_queued_work_claim(&turn_claim)
        .await
        .expect("abandon turn claim");
    release_session_execution_lease_for_test(&store, &rejected_command_lease).await;
    assert_eq!(
        store
            .list_queued_work("turn-first")
            .await
            .expect("list turn-first queue")
            .iter()
            .map(|batch| batch.batch_id.as_str())
            .collect::<Vec<_>>(),
        vec![
            first_turn.batch_id.as_str(),
            second_command.batch_id.as_str()
        ]
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

    let checkpoint_empty_lease =
        claim_session_execution_lease_for_test(&store, "root", "owner-a").await;
    assert!(
        store
            .claim_ready_queued_work(
                "root",
                &checkpoint_empty_lease.fence(),
                &lease_owner("owner-a"),
                QueuedWorkClaimBoundary::ActiveTurnCheckpoint,
                60_000,
                10,
            )
            .await
            .expect("checkpoint claim")
            .is_none(),
        "after-current-commit work at the queue head must wait for the idle boundary"
    );
    release_session_execution_lease_for_test(&store, &checkpoint_empty_lease).await;

    let idle_session_lease =
        claim_session_execution_lease_for_test(&store, "root", "owner-a").await;
    let idle_claim = store
        .claim_ready_queued_work(
            "root",
            &idle_session_lease.fence(),
            &lease_owner("owner-a"),
            QueuedWorkClaimBoundary::Idle,
            60_000,
            10,
        )
        .await
        .expect("idle claim")
        .expect("idle claim exists");
    release_session_execution_lease_for_test(&store, &idle_session_lease).await;
    assert_eq!(idle_claim.batches.len(), 1);
    assert_eq!(idle_claim.batches[0].batch_id, after_commit.batch_id);

    let checkpoint_session_lease =
        claim_session_execution_lease_for_test(&store, "root", "owner-b").await;
    let checkpoint_claim = store
        .claim_ready_queued_work(
            "root",
            &checkpoint_session_lease.fence(),
            &lease_owner("owner-b"),
            QueuedWorkClaimBoundary::ActiveTurnCheckpoint,
            60_000,
            10,
        )
        .await
        .expect("checkpoint claim after head is leased")
        .expect("checkpoint claim exists");
    release_session_execution_lease_for_test(&store, &checkpoint_session_lease).await;
    assert_eq!(checkpoint_claim.batches[0].batch_id, earliest.batch_id);

    store
        .abandon_queued_work_claim(&idle_claim)
        .await
        .expect("abandon idle claim");
    let reclaim_session_lease =
        claim_session_execution_lease_for_test(&store, "root", "owner-c").await;
    let reclaimed = store
        .claim_ready_queued_work(
            "root",
            &reclaim_session_lease.fence(),
            &lease_owner("owner-c"),
            QueuedWorkClaimBoundary::Idle,
            60_000,
            10,
        )
        .await
        .expect("reclaim abandoned work")
        .expect("reclaimed work exists");
    release_session_execution_lease_for_test(&store, &reclaim_session_lease).await;
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

async fn queued_work_respects_membership_limits_exclusivity_reclaim_and_sessions(
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

    let root_session_lease =
        claim_session_execution_lease_for_test(&store, "root", "owner-a").await;
    let claim = store
        .claim_ready_queued_work(
            "root",
            &root_session_lease.fence(),
            &lease_owner("owner-a"),
            QueuedWorkClaimBoundary::Idle,
            60_000,
            10,
        )
        .await
        .expect("claim root")
        .expect("root claim");
    release_session_execution_lease_for_test(&store, &root_session_lease).await;
    assert_eq!(
        claim
            .batches
            .iter()
            .map(|batch| batch.batch_id.as_str())
            .collect::<Vec<_>>(),
        vec![exclusive.batch_id.as_str()],
        "an exclusive batch must claim alone and unavailable earlier work must be skipped"
    );
    let next_root_session_lease =
        claim_session_execution_lease_for_test(&store, "root", "owner-b").await;
    let next_root = store
        .claim_ready_queued_work(
            "root",
            &next_root_session_lease.fence(),
            &lease_owner("owner-b"),
            QueuedWorkClaimBoundary::Idle,
            60_000,
            10,
        )
        .await
        .expect("claim joined")
        .expect("joined claim");
    release_session_execution_lease_for_test(&store, &next_root_session_lease).await;
    assert_eq!(next_root.batches[0].batch_id, joined.batch_id);
    let other_session_lease =
        claim_session_execution_lease_for_test(&store, "other", "owner-c").await;
    let other_claim = store
        .claim_ready_queued_work(
            "other",
            &other_session_lease.fence(),
            &lease_owner("owner-c"),
            QueuedWorkClaimBoundary::Idle,
            60_000,
            10,
        )
        .await
        .expect("claim other")
        .expect("other claim");
    release_session_execution_lease_for_test(&store, &other_session_lease).await;
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
    let expired_session_lease =
        claim_session_execution_lease_for_test(&store, "reclaim", "owner-a").await;
    let expired = store
        .claim_ready_queued_work(
            "reclaim",
            &expired_session_lease.fence(),
            &lease_owner("owner-a"),
            QueuedWorkClaimBoundary::Idle,
            0,
            1,
        )
        .await
        .expect("claim with zero ttl")
        .expect("expired claim");
    release_session_execution_lease_for_test(&store, &expired_session_lease).await;
    let reclaim_session_lease =
        claim_session_execution_lease_for_test(&store, "reclaim", "owner-b").await;
    let reclaimed = store
        .claim_ready_queued_work(
            "reclaim",
            &reclaim_session_lease.fence(),
            &lease_owner("owner-b"),
            QueuedWorkClaimBoundary::Idle,
            60_000,
            1,
        )
        .await
        .expect("reclaim expired")
        .expect("reclaimed expired claim");
    release_session_execution_lease_for_test(&store, &reclaim_session_lease).await;
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
    let limited_session_lease =
        claim_session_execution_lease_for_test(&store, "limited", "owner").await;
    let limited = store
        .claim_ready_queued_work(
            "limited",
            &limited_session_lease.fence(),
            &lease_owner("owner"),
            QueuedWorkClaimBoundary::Idle,
            60_000,
            2,
        )
        .await
        .expect("limited claim")
        .expect("limited claim exists");
    release_session_execution_lease_for_test(&store, &limited_session_lease).await;
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
    let remaining_session_lease =
        claim_session_execution_lease_for_test(&store, "limited", "owner-next").await;
    let remaining = store
        .claim_ready_queued_work(
            "limited",
            &remaining_session_lease.fence(),
            &lease_owner("owner-next"),
            QueuedWorkClaimBoundary::Idle,
            60_000,
            10,
        )
        .await
        .expect("remaining claim")
        .expect("remaining claim exists");
    release_session_execution_lease_for_test(&store, &remaining_session_lease).await;
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

    let first_session_lease =
        claim_session_execution_lease_for_test(&store, "root", "owner-a").await;
    let first_claim = store
        .claim_ready_queued_work(
            "root",
            &first_session_lease.fence(),
            &lease_owner("owner-a"),
            QueuedWorkClaimBoundary::Idle,
            60_000,
            10,
        )
        .await
        .expect("claim first group")
        .expect("first group claim");
    release_session_execution_lease_for_test(&store, &first_session_lease).await;
    assert_eq!(
        first_claim
            .batches
            .iter()
            .map(|batch| batch.batch_id.as_str())
            .collect::<Vec<_>>(),
        vec![first.batch_id.as_str(), second.batch_id.as_str()],
        "join claims must group only adjacent batches with the same delivery policy and merge key"
    );
    let second_session_lease =
        claim_session_execution_lease_for_test(&store, "root", "owner-b").await;
    let second_claim = store
        .claim_ready_queued_work(
            "root",
            &second_session_lease.fence(),
            &lease_owner("owner-b"),
            QueuedWorkClaimBoundary::Idle,
            60_000,
            10,
        )
        .await
        .expect("claim second group")
        .expect("second group claim");
    release_session_execution_lease_for_test(&store, &second_session_lease).await;
    assert_eq!(second_claim.batches[0].batch_id, different_merge.batch_id);
    let third_session_lease =
        claim_session_execution_lease_for_test(&store, "root", "owner-c").await;
    let third_claim = store
        .claim_ready_queued_work(
            "root",
            &third_session_lease.fence(),
            &lease_owner("owner-c"),
            QueuedWorkClaimBoundary::Idle,
            60_000,
            10,
        )
        .await
        .expect("claim third group")
        .expect("third group claim");
    release_session_execution_lease_for_test(&store, &third_session_lease).await;
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
    let claim_session_lease =
        claim_session_execution_lease_for_test(&store, "root", "owner-a").await;
    let claim = store
        .claim_ready_queued_work(
            "root",
            &claim_session_lease.fence(),
            &lease_owner("owner-a"),
            QueuedWorkClaimBoundary::Idle,
            60_000,
            10,
        )
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
            RuntimeCommit::persisted_state(&state, &[])
                .with_session_execution_lease(claim_session_lease.fence())
                .completing_queue_claim(stale_completion),
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
            RuntimeCommit::persisted_state(&state, &[])
                .with_session_execution_lease(claim_session_lease.fence())
                .releasing_session_execution_lease(claim_session_lease.completion())
                .completing_queue_claim(claim.completion()),
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
    let session_lease = claim_session_execution_lease_for_test(&store, "root", "queue-owner").await;
    let claim = store
        .claim_ready_queued_work(
            "root",
            &session_lease.fence(),
            &lease_owner("queue-owner"),
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
                .with_session_execution_lease(session_lease.fence())
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
                .with_session_execution_lease(session_lease.fence())
                .releasing_session_execution_lease(session_lease.completion())
                .with_turn_commit(turn_commit.clone())
                .completing_queue_claim(claim.completion()),
        )
        .await
        .expect("valid final commit clears queue and records the turn stamp atomically");
    let retry = store
        .commit_runtime_state(
            base_commit
                .with_session_execution_lease(session_lease.fence())
                .releasing_session_execution_lease(session_lease.completion())
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

async fn pending_turn_inputs_source_keys_order_cancel_and_cross_session(
    store: Arc<dyn RuntimePersistence>,
) {
    let first = store
        .enqueue_pending_turn_input(
            pending_next_turn_input_draft("root", "first").with_source_key("source:first"),
        )
        .await
        .expect("enqueue first pending input");
    let replay = store
        .enqueue_pending_turn_input(
            pending_next_turn_input_draft("root", "different replay payload")
                .with_source_key("source:first"),
        )
        .await
        .expect("replay first pending input");
    let second = store
        .enqueue_pending_turn_input(pending_next_turn_input_draft("root", "second"))
        .await
        .expect("enqueue second pending input");
    store
        .enqueue_pending_turn_input(pending_next_turn_input_draft("other", "other session"))
        .await
        .expect("enqueue other session pending input");

    assert_eq!(
        first.input_id, replay.input_id,
        "replaying a source key must return the original pending input"
    );
    assert_eq!(
        pending_input_text(&replay),
        Some("first"),
        "source-key replay must return the original stored payload, not the replay attempt"
    );
    let listed = store
        .list_pending_turn_inputs("root")
        .await
        .expect("list pending turn inputs");
    assert_eq!(
        listed
            .iter()
            .map(|input| input.input_id.as_str())
            .collect::<Vec<_>>(),
        vec![first.input_id.as_str(), second.input_id.as_str()]
    );
    assert!(listed[0].enqueue_seq < listed[1].enqueue_seq);
    assert!(listed.iter().all(|input| input.session_id == "root"));

    let cancelled = store
        .cancel_pending_turn_input("root", &second.input_id)
        .await
        .expect("cancel pending turn input")
        .expect("cancelled input");
    assert_eq!(cancelled.input_id, second.input_id);
    assert_eq!(
        store
            .list_pending_turn_inputs("root")
            .await
            .expect("list after cancel")
            .iter()
            .map(|input| input.input_id.as_str())
            .collect::<Vec<_>>(),
        vec![first.input_id.as_str()]
    );
}

async fn pending_turn_input_claims_reclaim_complete_and_fence(store: Arc<dyn RuntimePersistence>) {
    let first = store
        .enqueue_pending_turn_input(
            crate::PendingTurnInputDraft::new(
                "root",
                crate::TurnInputIngress::NextTurn,
                crate::TurnInput::text("first next").with_image_ref("next-image", vec![1, 2, 3]),
            )
            .with_source_key("next:first"),
        )
        .await
        .expect("enqueue first next input");
    let second = store
        .enqueue_pending_turn_input(pending_next_turn_input_draft("root", "second next"))
        .await
        .expect("enqueue second next input");
    let lease = claim_session_execution_lease_for_test(&store, "root", "turn-input-owner").await;
    let claim = store
        .claim_next_turn_inputs(
            "root",
            &lease.fence(),
            &lease_owner("turn-input-owner"),
            60_000,
            10,
        )
        .await
        .expect("claim next inputs")
        .expect("next input claim");
    assert_eq!(
        claim
            .inputs
            .iter()
            .map(|input| input.input_id.as_str())
            .collect::<Vec<_>>(),
        vec![first.input_id.as_str(), second.input_id.as_str()]
    );
    assert!(
        claim
            .materialize_for_turn()
            .image_blobs
            .contains_key("next-image"),
        "claimed next-turn input must preserve image blobs"
    );
    assert!(
        store
            .cancel_pending_turn_input("root", &first.input_id)
            .await
            .expect("cancel claimed input")
            .is_none(),
        "live claimed pending input must not be cancellable"
    );
    assert!(
        store
            .list_pending_turn_inputs("root")
            .await
            .expect("list claimed inputs")
            .is_empty(),
        "live claimed pending inputs must be hidden from queue previews"
    );

    store
        .abandon_turn_input_claim(&claim)
        .await
        .expect("abandon pending input claim");
    assert_eq!(
        store
            .list_pending_turn_inputs("root")
            .await
            .expect("list after abandon")
            .len(),
        2
    );
    let reclaimed = store
        .claim_next_turn_inputs(
            "root",
            &lease.fence(),
            &lease_owner("turn-input-owner"),
            60_000,
            10,
        )
        .await
        .expect("reclaim next inputs")
        .expect("reclaimed next claim");
    assert!(
        reclaimed.fencing_token > claim.fencing_token,
        "reclaiming abandoned pending inputs must advance the fencing token"
    );

    let state = RuntimeSessionState {
        session_id: "root".to_string(),
        ..RuntimeSessionState::default()
    };
    let err = store
        .commit_runtime_state(
            RuntimeCommit::persisted_state(&state, &[])
                .with_session_execution_lease(lease.fence())
                .completing_turn_input_claim(claim.completion()),
        )
        .await
        .expect_err("stale turn-input completion must fail");
    assert!(matches!(err, StoreError::TurnInputClaimExpired { .. }));
    assert!(
        store
            .list_pending_turn_inputs("root")
            .await
            .expect("list reclaimed live inputs")
            .is_empty(),
        "stale completion must not abandon the live reclaimed claim"
    );

    store
        .commit_runtime_state(
            RuntimeCommit::persisted_state(&state, &[])
                .with_session_execution_lease(lease.fence())
                .releasing_session_execution_lease(lease.completion())
                .completing_turn_input_claim(reclaimed.completion()),
        )
        .await
        .expect("valid pending input completion commits");
    assert!(
        store
            .list_pending_turn_inputs("root")
            .await
            .expect("list after valid completion")
            .is_empty()
    );
}

async fn pending_active_turn_inputs_defer_unaccepted_once_on_interrupt(
    store: Arc<dyn RuntimePersistence>,
) {
    let turn_id = "active-turn-1";
    let accepted = store
        .enqueue_pending_turn_input(
            crate::PendingTurnInputDraft::new(
                "root",
                crate::TurnInputIngress::active_turn(
                    turn_id,
                    crate::TurnInputCheckpointBoundary::AfterWork,
                ),
                crate::TurnInput::text("accepted active")
                    .with_image_ref("accepted-active-image", vec![9, 8, 7]),
            )
            .with_source_key("active:accepted"),
        )
        .await
        .expect("enqueue accepted active input");
    let unaccepted = store
        .enqueue_pending_turn_input(pending_active_turn_input_draft(
            "root",
            turn_id,
            crate::TurnInputCheckpointBoundary::AfterWork,
            "unaccepted active",
        ))
        .await
        .expect("enqueue unaccepted active input");
    let before_completion = store
        .enqueue_pending_turn_input(pending_active_turn_input_draft(
            "root",
            turn_id,
            crate::TurnInputCheckpointBoundary::BeforeCompletion,
            "before-completion active",
        ))
        .await
        .expect("enqueue before-completion active input");
    let other_active = store
        .enqueue_pending_turn_input(pending_active_turn_input_draft(
            "root",
            "other-turn",
            crate::TurnInputCheckpointBoundary::AfterWork,
            "other active",
        ))
        .await
        .expect("enqueue other active input");

    let lease = claim_session_execution_lease_for_test(&store, "root", "active-input-owner").await;
    let claim = store
        .claim_active_turn_inputs(
            "root",
            &lease.fence(),
            &lease_owner("active-input-owner"),
            turn_id,
            crate::CheckpointKind::AfterWork,
            60_000,
            1,
        )
        .await
        .expect("claim active inputs")
        .expect("active input claim");
    assert_eq!(
        claim
            .inputs
            .iter()
            .map(|input| input.input_id.as_str())
            .collect::<Vec<_>>(),
        vec![accepted.input_id.as_str()],
        "AfterWork claims must include matching active inputs admitted at that boundary in order"
    );
    assert!(
        claim
            .materialize_for_turn()
            .image_blobs
            .contains_key("accepted-active-image"),
        "accepted active claim must preserve image blobs"
    );

    let state = RuntimeSessionState {
        session_id: "root".to_string(),
        ..RuntimeSessionState::default()
    };
    store
        .commit_runtime_state(
            RuntimeCommit::persisted_state(&state, &[])
                .with_session_execution_lease(lease.fence())
                .completing_turn_input_claim(claim.completion())
                .deferring_interrupted_turn_inputs(turn_id),
        )
        .await
        .expect("interrupt commit completes accepted inputs and defers unaccepted inputs");
    let pending_after_interrupt = store
        .list_pending_turn_inputs("root")
        .await
        .expect("list after interrupt deferral");
    assert_eq!(
        pending_after_interrupt
            .iter()
            .map(|input| input.input_id.as_str())
            .collect::<Vec<_>>(),
        vec![
            unaccepted.input_id.as_str(),
            before_completion.input_id.as_str(),
            other_active.input_id.as_str(),
        ],
        "interrupt must complete accepted input, defer matching unaccepted inputs, and retain other-turn active input"
    );
    let deferred_after_interrupt = pending_after_interrupt
        .iter()
        .filter(|input| input.ingress.active_turn_id().is_none())
        .collect::<Vec<_>>();
    assert_eq!(
        deferred_after_interrupt
            .iter()
            .map(|input| input.input_id.as_str())
            .collect::<Vec<_>>(),
        vec![
            unaccepted.input_id.as_str(),
            before_completion.input_id.as_str()
        ],
        "accepted active inputs must be completed and only unaccepted matching active inputs become next-turn work"
    );
    assert!(deferred_after_interrupt.iter().all(|input| {
        matches!(input.ingress, crate::TurnInputIngress::NextTurn)
            && input.state == crate::TurnInputState::DeferredNextTurn
    }));
    assert!(
        pending_after_interrupt
            .iter()
            .any(|input| input.ingress.active_turn_id() == Some("other-turn")),
        "inputs for other active turns must not be deferred by this interrupt"
    );

    let next_claim = store
        .claim_next_turn_inputs(
            "root",
            &lease.fence(),
            &lease_owner("active-input-owner"),
            60_000,
            10,
        )
        .await
        .expect("claim deferred next inputs")
        .expect("deferred next input claim");
    assert_eq!(
        next_claim
            .inputs
            .iter()
            .map(|input| input.input_id.as_str())
            .collect::<Vec<_>>(),
        vec![
            unaccepted.input_id.as_str(),
            before_completion.input_id.as_str()
        ]
    );
    store
        .commit_runtime_state(
            RuntimeCommit::persisted_state(&state, &[])
                .with_session_execution_lease(lease.fence())
                .releasing_session_execution_lease(lease.completion())
                .completing_turn_input_claim(next_claim.completion()),
        )
        .await
        .expect("complete deferred next input");
    assert!(
        store
            .list_pending_turn_inputs("root")
            .await
            .expect("list after completing deferred input")
            .iter()
            .all(|input| input.ingress.active_turn_id() == Some("other-turn")),
        "inputs for other active turns must not be deferred by this interrupt"
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
    commit_runtime_state_for_test(
        &store,
        RuntimeCommit::persisted_state(&state, &[]),
        "tombstone",
    )
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
    session_execution_lease_first_claim_excludes_concurrent_reopen_handles(&factory).await;

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
    commit_runtime_state_for_test(
        &factory.open,
        RuntimeCommit::persisted_state(&state, &[]),
        "reopen",
    )
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

async fn session_execution_lease_first_claim_excludes_concurrent_reopen_handles(
    factory: &ReopenableRuntimePersistence,
) {
    let barrier = Arc::new(tokio::sync::Barrier::new(3));
    let open = Arc::clone(&factory.open);
    let reopen = Arc::clone(&factory.reopen);
    let open_barrier = Arc::clone(&barrier);
    let reopen_barrier = Arc::clone(&barrier);
    let open_owner = lease_owner("owner-a");
    let reopen_owner = lease_owner("owner-b");

    let open_claim = tokio::spawn(async move {
        open_barrier.wait().await;
        open.try_claim_session_execution_lease("first-claim-race", &open_owner, 60_000)
            .await
    });
    let reopen_claim = tokio::spawn(async move {
        reopen_barrier.wait().await;
        reopen
            .try_claim_session_execution_lease("first-claim-race", &reopen_owner, 60_000)
            .await
    });

    barrier.wait().await;
    let open_claim = open_claim
        .await
        .expect("join open first-claim race")
        .expect("open first-claim race");
    let reopen_claim = reopen_claim
        .await
        .expect("join reopen first-claim race")
        .expect("reopen first-claim race");
    let open_lease = open_claim.acquired();
    let reopen_lease = reopen_claim.acquired();
    let claim_count = usize::from(open_lease.is_some()) + usize::from(reopen_lease.is_some());
    assert_eq!(
        claim_count, 1,
        "exactly one concurrent first claim may acquire a session execution lease"
    );
    if let Some(lease) = open_lease.as_ref().or(reopen_lease.as_ref()) {
        factory
            .open
            .release_session_execution_lease(&lease.completion())
            .await
            .expect("release first-claim race winner");
    }
}

async fn queued_wake_delivery_is_source_key_idempotent_and_claimed_once(
    store: Arc<dyn RuntimePersistence>,
) {
    let wake = ProcessWakeDelivery {
        wake_id: "wake-1".to_string(),
        target_session_id: "root".to_string(),
        target_scope_id: SessionScopeId::new("session:root"),
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

    let session_lease = claim_session_execution_lease_for_test(&store, "root", "wake-owner").await;
    let claim = store
        .claim_ready_queued_work(
            "root",
            &session_lease.fence(),
            &lease_owner("wake-owner"),
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
            RuntimeCommit::persisted_state(&state, &[])
                .with_session_execution_lease(session_lease.fence())
                .releasing_session_execution_lease(session_lease.completion())
                .completing_queue_claim(claim.completion()),
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
    let stamped_commit = commit.with_turn_commit(RuntimeTurnCommitStamp::new(
        "root",
        "provider-turn",
        turn_commit_hash.clone(),
    ));

    let session_lease =
        claim_session_execution_lease_for_test(&store, "root", "provider-turn").await;
    let first = store
        .commit_runtime_state(
            stamped_commit
                .clone()
                .with_session_execution_lease(session_lease.fence())
                .releasing_session_execution_lease(session_lease.completion()),
        )
        .await
        .expect("first final commit requires a live session execution lease");
    let retry = store
        .commit_runtime_state(stamped_commit)
        .await
        .expect("same final commit retries idempotently without a live lease");
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
