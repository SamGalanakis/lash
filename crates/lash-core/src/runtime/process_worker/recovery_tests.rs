use std::time::Duration;

use super::*;
use crate::{
    AbandonRequest, DurabilityTier, LeaseOwnerIdentity, LeaseOwnerLiveness, ProcessInput,
    ProcessListFilter, ProcessRegistration, ProcessStarted, ProcessStatus,
    TestLocalProcessRegistry, TriggerStore,
};

fn inline_worker(
    registry: Arc<dyn ProcessRegistry>,
    lease_owner: LeaseOwnerIdentity,
) -> DurableProcessWorker {
    inline_worker_with_trigger_store(
        registry,
        lease_owner,
        Arc::new(crate::InMemoryTriggerStore::default()),
    )
}

fn inline_worker_with_trigger_store(
    registry: Arc<dyn ProcessRegistry>,
    lease_owner: LeaseOwnerIdentity,
    trigger_store: Arc<dyn TriggerStore>,
) -> DurableProcessWorker {
    struct InlineSessionStoreFactory;

    #[async_trait::async_trait]
    impl SessionStoreFactory for InlineSessionStoreFactory {
        fn durability_tier(&self) -> DurabilityTier {
            DurabilityTier::Inline
        }

        async fn create_store(
            &self,
            _request: &crate::SessionStoreCreateRequest,
        ) -> Result<Arc<dyn crate::RuntimePersistence>, String> {
            Ok(Arc::new(InMemorySessionStore::default()))
        }

        async fn delete_session(&self, _session_id: &str) -> Result<(), String> {
            Ok(())
        }
    }

    DurableProcessWorker::new(
        DurableProcessWorkerConfig::new(
            Arc::new(PluginHost::new(Vec::new())),
            RuntimeHostConfig::in_memory(),
            Arc::new(InlineSessionStoreFactory),
            registry,
        )
        .with_trigger_store(trigger_store)
        .with_lease_owner(lease_owner),
    )
}

/// A registration with an explicit disposition. Uses an External input as a
/// convenient no-env placeholder; the disposition-driven sweep keys off the
/// declared disposition, not the input kind, so these unit tests exercise the
/// verdict without standing up execution infrastructure.
fn registration_with_disposition(
    id: &str,
    disposition: crate::RecoveryDisposition,
) -> ProcessRegistration {
    ProcessRegistration::new(
        id,
        ProcessInput::External {
            metadata: serde_json::json!({}),
        },
        disposition,
        crate::ProcessProvenance::host(),
    )
}

async fn abandoned_evidence(
    registry: &Arc<dyn ProcessRegistry>,
    process_id: &str,
) -> crate::AbandonEvidence {
    let record = registry
        .get_process(process_id)
        .await
        .expect("process exists");
    match record.status {
        ProcessStatus::Abandoned {
            await_output: ProcessAwaitOutput::Abandoned { evidence, .. },
        } => *evidence,
        other => panic!("expected an Abandoned terminal, got {other:?}"),
    }
}

fn local_owner(owner_id: &str, host_id: &str, process_start: &str) -> LeaseOwnerIdentity {
    LeaseOwnerIdentity {
        owner_id: owner_id.to_string(),
        incarnation_id: format!("{owner_id}:incarnation"),
        liveness: LeaseOwnerLiveness::local_process_for_test(
            host_id,
            "boot-a",
            std::process::id(),
            process_start,
        ),
    }
}

async fn seed_reserved_trigger_delivery(
    trigger_store: &Arc<dyn TriggerStore>,
) -> crate::TriggerDeliveryReservation {
    let source_type = "ui.button.pressed";
    let source_key =
        crate::empty_trigger_source_key(source_type).expect("empty trigger source key");
    let subscription = trigger_store
        .register_subscription(
            crate::TriggerSubscriptionDraft::for_process(
                crate::ProcessOriginator::host(),
                crate::ProcessExecutionEnvRef::new("process-env:test"),
                source_type,
                source_key.clone(),
                ProcessInput::Engine {
                    kind: "test-engine".to_string(),
                    payload: serde_json::json!({ "target": "reconcile" }),
                },
                crate::ProcessIdentity::new("test-engine"),
            )
            .with_payload_schema(crate::LashSchema::any()),
        )
        .await
        .expect("register trigger subscription");
    let occurrence = trigger_store
        .record_occurrence(crate::TriggerOccurrenceRequest::new(
            source_type,
            source_key,
            serde_json::json!({ "button": "Blue" }),
            "button-blue-reconcile",
        ))
        .await
        .expect("record trigger occurrence");
    let deliveries = trigger_store
        .reserve_matching_deliveries(&occurrence.occurrence_id)
        .await
        .expect("reserve trigger delivery");
    assert_eq!(deliveries.len(), 1);
    assert_eq!(
        deliveries[0].subscription.subscription_id,
        subscription.subscription_id
    );
    deliveries[0].clone()
}

async fn process_count(registry: &Arc<dyn ProcessRegistry>, process_id: &str) -> usize {
    registry
        .list_processes(&ProcessListFilter {
            status: crate::ProcessStatusFilter::Any,
            ..ProcessListFilter::default()
        })
        .await
        .expect("list processes")
        .into_iter()
        .filter(|record| record.id == process_id)
        .count()
}

async fn await_terminal(registry: &Arc<dyn ProcessRegistry>, process_id: &str) {
    let awaiter = crate::ProcessAwaiter::polling(Arc::clone(registry));
    tokio::time::timeout(
        std::time::Duration::from_secs(5),
        awaiter.await_terminal(process_id),
    )
    .await
    .expect("recovered process reaches terminal within the sweep")
    .expect("recovered process terminal output");
}

#[tokio::test]
async fn sweep_reconciles_reserved_trigger_delivery_without_process() {
    let registry: Arc<dyn ProcessRegistry> = Arc::new(TestLocalProcessRegistry::default());
    let trigger_store: Arc<dyn TriggerStore> = Arc::new(crate::InMemoryTriggerStore::default());
    let delivery = seed_reserved_trigger_delivery(&trigger_store).await;
    assert!(
        registry.get_process(&delivery.process_id).await.is_none(),
        "test starts in the reserve/start crash window"
    );

    let worker = inline_worker_with_trigger_store(
        Arc::clone(&registry),
        local_owner("trigger-worker", "host-a", "claimant-start"),
        Arc::clone(&trigger_store),
    );
    worker
        .drive_pending_processes()
        .await
        .expect("sweep dispatches");

    let record = registry
        .get_process(&delivery.process_id)
        .await
        .expect("sweep registers missing trigger delivery process");
    assert_eq!(record.id, delivery.process_id);
    assert_eq!(process_count(&registry, &delivery.process_id).await, 1);
    assert!(matches!(
        record.provenance.caused_by,
        Some(crate::CausalRef::TriggerOccurrence {
            occurrence_id,
            subscription_id: Some(subscription_id),
        }) if occurrence_id == delivery.occurrence.occurrence_id
            && subscription_id == delivery.subscription.subscription_id
    ));

    worker
        .drive_pending_processes()
        .await
        .expect("second sweep dispatches");
    assert_eq!(
        process_count(&registry, &delivery.process_id).await,
        1,
        "re-running the sweep must not create a duplicate process row"
    );
}

#[tokio::test]
async fn sweep_does_not_reconcile_trigger_delivery_pruned_with_terminal_process() {
    let trigger_store = Arc::new(crate::InMemoryTriggerStore::default());
    let registry: Arc<dyn ProcessRegistry> = Arc::new(
        TestLocalProcessRegistry::default().with_trigger_store(Arc::clone(&trigger_store)),
    );
    let trigger_store_dyn: Arc<dyn TriggerStore> = trigger_store.clone();
    let delivery = seed_reserved_trigger_delivery(&trigger_store_dyn).await;
    assert!(
        registry.get_process(&delivery.process_id).await.is_none(),
        "test starts in the reserve/start crash window"
    );

    let worker = inline_worker_with_trigger_store(
        Arc::clone(&registry),
        local_owner("trigger-worker", "host-a", "claimant-start"),
        Arc::clone(&trigger_store_dyn),
    );
    worker
        .drive_pending_processes()
        .await
        .expect("sweep dispatches");
    registry
        .get_process(&delivery.process_id)
        .await
        .expect("sweep registers missing trigger delivery process");

    let terminal = registry
        .complete_process(
            &delivery.process_id,
            ProcessAwaitOutput::Success {
                value: serde_json::json!({ "done": true }),
                control: None,
            },
            crate::ProcessCompletionAuthority::workflow_key(&delivery.process_id),
        )
        .await
        .expect("complete trigger delivery process");
    let report = registry
        .prune_terminal_processes(terminal.updated_at_ms.saturating_add(1), None, None)
        .await
        .expect("prune completed trigger delivery process");
    assert_eq!(report.pruned_processes, 1);
    assert!(
        registry.get_process(&delivery.process_id).await.is_none(),
        "terminal trigger delivery process is pruned"
    );
    assert!(
        trigger_store
            .list_deliveries_by_process_id(&delivery.process_id)
            .await
            .expect("list trigger deliveries after prune")
            .is_empty(),
        "prune removes the delivery row together with the process"
    );

    worker
        .drive_pending_processes()
        .await
        .expect("post-prune sweep dispatches");
    assert_eq!(
        process_count(&registry, &delivery.process_id).await,
        0,
        "recovery sweep must not resurrect a delivery whose terminal process was pruned"
    );
}

#[tokio::test]
async fn sweep_does_not_reconcile_trigger_delivery_when_process_exists() {
    let registry: Arc<dyn ProcessRegistry> = Arc::new(TestLocalProcessRegistry::default());
    let trigger_store: Arc<dyn TriggerStore> = Arc::new(crate::InMemoryTriggerStore::default());
    let delivery = seed_reserved_trigger_delivery(&trigger_store).await;
    registry
        .register_process(ProcessRegistration::new(
            delivery.process_id.clone(),
            ProcessInput::External {
                metadata: serde_json::json!({ "already": "registered" }),
            },
            RecoveryDisposition::Rerunnable,
            crate::ProcessProvenance::host(),
        ))
        .await
        .expect("pre-register delivery process");

    let worker = inline_worker_with_trigger_store(
        Arc::clone(&registry),
        local_owner("trigger-worker", "host-a", "claimant-start"),
        trigger_store,
    );
    worker
        .drive_pending_processes()
        .await
        .expect("sweep dispatches");

    let record = registry
        .get_process(&delivery.process_id)
        .await
        .expect("existing process remains");
    assert_eq!(record.provenance.caused_by, None);
    assert_eq!(
        process_count(&registry, &delivery.process_id).await,
        1,
        "existing process row must be treated as already started"
    );
}

/// ExternallyOwned rows are never claimed and never run: lash does not own
/// their execution (ADR 0019).
#[tokio::test]
async fn sweep_never_claims_externally_owned_rows() {
    let registry: Arc<dyn ProcessRegistry> = Arc::new(TestLocalProcessRegistry::default());
    registry
        .register_process(registration_with_disposition(
            "proc-ext",
            RecoveryDisposition::ExternallyOwned,
        ))
        .await
        .expect("register");

    let worker = inline_worker(
        Arc::clone(&registry),
        local_owner("live-worker", "host-a", "claimant-start"),
    );
    worker
        .drive_pending_processes()
        .await
        .expect("sweep dispatches");
    tokio::time::sleep(Duration::from_millis(200)).await;

    let record = registry.get_process("proc-ext").await.expect("process");
    assert!(
        !record.is_terminal(),
        "an externally-owned row must never be claimed or run by the sweep"
    );
    assert!(
        registry
            .get_process_lease("proc-ext")
            .await
            .expect("lease read")
            .is_none(),
        "the sweep must not claim a lease on an externally-owned row"
    );
}

/// A pending Abandon Request on an externally-owned row is reconciled into
/// `Abandoned{reconciled_request}` — there is no owner lease to wait out.
#[tokio::test]
async fn sweep_reconciles_externally_owned_abandon_request() {
    let registry: Arc<dyn ProcessRegistry> = Arc::new(TestLocalProcessRegistry::default());
    registry
        .register_process(registration_with_disposition(
            "proc-ext-abandon",
            RecoveryDisposition::ExternallyOwned,
        ))
        .await
        .expect("register");
    registry
        .request_process_abandon(
            "proc-ext-abandon",
            AbandonRequest {
                requested_by: "operator".to_string(),
                requested_at_ms: 1,
                reason: Some("host retired".to_string()),
            },
        )
        .await
        .expect("request abandon");

    let worker = inline_worker(
        Arc::clone(&registry),
        local_owner("live-worker", "host-a", "claimant-start"),
    );
    worker
        .drive_pending_processes()
        .await
        .expect("sweep dispatches");
    await_terminal(&registry, "proc-ext-abandon").await;

    let evidence = abandoned_evidence(&registry, "proc-ext-abandon").await;
    assert_eq!(evidence.writer, AbandonWriter::ReconciledRequest);
    assert!(
        evidence.owner.is_none(),
        "externally-owned work names no lash execution owner"
    );
}

/// A started OwnerBound row whose holder is provably dead is terminalized as
/// `Abandoned{sweep}`, never re-run — replacing a re-execution would repeat
/// non-idempotent side effects.
#[tokio::test]
async fn sweep_abandons_started_owner_bound_with_provably_dead_holder() {
    let registry: Arc<dyn ProcessRegistry> = Arc::new(TestLocalProcessRegistry::default());
    registry
        .register_process(registration_with_disposition(
            "proc-ob-dead",
            RecoveryDisposition::OwnerBound,
        ))
        .await
        .expect("register");
    let dead_holder = local_owner("dead-worker", "host-a", "not-the-current-process-start");
    registry
        .record_first_started(
            "proc-ob-dead",
            ProcessStarted {
                owner: dead_holder.clone(),
                started_at_ms: 1,
            },
        )
        .await
        .expect("record started");
    registry
        .claim_process_lease("proc-ob-dead", &dead_holder, 60_000)
        .await
        .expect("dead holder claims")
        .acquired()
        .expect("dead holder lease acquired");

    let worker = inline_worker(
        Arc::clone(&registry),
        local_owner("live-worker", "host-a", "claimant-start"),
    );
    worker
        .drive_pending_processes()
        .await
        .expect("sweep dispatches");
    await_terminal(&registry, "proc-ob-dead").await;

    let evidence = abandoned_evidence(&registry, "proc-ob-dead").await;
    assert_eq!(evidence.writer, AbandonWriter::Sweep);
    assert_eq!(
        evidence.owner.as_ref().map(|owner| owner.owner_id.as_str()),
        Some("dead-worker"),
        "the sweep names the provably-dead holder as the abandoned owner"
    );

    // Revenant: the dead owner reappears and tries to complete the row. The
    // row is already terminal, so the write is rejected — the sweep stayed the
    // single writer.
    assert!(
        registry
            .complete_process(
                "proc-ob-dead",
                ProcessAwaitOutput::Success {
                    value: serde_json::json!("revenant"),
                    control: None,
                },
                crate::ProcessCompletionAuthority::workflow_key("proc-ob-dead"),
            )
            .await
            .is_err(),
        "a revenant cannot overwrite an Abandoned terminal"
    );
}

/// A started OwnerBound row whose holder is merely silent (no death evidence)
/// and carries no Abandon Request is left non-terminal — elapsed time alone
/// never terminalizes.
#[tokio::test]
async fn sweep_skips_started_owner_bound_with_silent_holder() {
    let registry: Arc<dyn ProcessRegistry> = Arc::new(TestLocalProcessRegistry::default());
    registry
        .register_process(registration_with_disposition(
            "proc-ob-silent",
            RecoveryDisposition::OwnerBound,
        ))
        .await
        .expect("register");
    registry
        .record_first_started(
            "proc-ob-silent",
            ProcessStarted {
                owner: LeaseOwnerIdentity::opaque("started-worker", "started-incarnation"),
                started_at_ms: 1,
            },
        )
        .await
        .expect("record started");
    // Opaque live holder: no liveness proof, so it is never provably dead.
    registry
        .claim_process_lease(
            "proc-ob-silent",
            &LeaseOwnerIdentity::opaque("other-worker", "other-incarnation"),
            60_000,
        )
        .await
        .expect("live holder claims")
        .acquired()
        .expect("live holder lease acquired");

    let worker = inline_worker(
        Arc::clone(&registry),
        local_owner("live-worker", "host-a", "claimant-start"),
    );
    worker
        .drive_pending_processes()
        .await
        .expect("sweep dispatches");
    tokio::time::sleep(Duration::from_millis(200)).await;

    let record = registry
        .get_process("proc-ob-silent")
        .await
        .expect("process");
    assert!(
        !record.is_terminal(),
        "a silent, not-provably-dead holder with no abandon request stays non-terminal"
    );
}

/// A started OwnerBound row with a lapsed lease and a pending Abandon Request
/// is reconciled into `Abandoned{reconciled_request}`, naming the started
/// owner as the lapsed owner.
#[tokio::test]
async fn sweep_reconciles_started_owner_bound_after_lease_lapse() {
    let registry: Arc<dyn ProcessRegistry> = Arc::new(TestLocalProcessRegistry::default());
    registry
        .register_process(registration_with_disposition(
            "proc-ob-lapse",
            RecoveryDisposition::OwnerBound,
        ))
        .await
        .expect("register");
    registry
        .record_first_started(
            "proc-ob-lapse",
            ProcessStarted {
                owner: LeaseOwnerIdentity::opaque("lapsed-owner", "lapsed-incarnation"),
                started_at_ms: 1,
            },
        )
        .await
        .expect("record started");
    registry
        .request_process_abandon(
            "proc-ob-lapse",
            AbandonRequest {
                requested_by: "operator".to_string(),
                requested_at_ms: 2,
                reason: None,
            },
        )
        .await
        .expect("request abandon");
    // No live lease held: the row's owner lease has lapsed.

    let worker = inline_worker(
        Arc::clone(&registry),
        local_owner("live-worker", "host-a", "claimant-start"),
    );
    worker
        .drive_pending_processes()
        .await
        .expect("sweep dispatches");
    await_terminal(&registry, "proc-ob-lapse").await;

    let evidence = abandoned_evidence(&registry, "proc-ob-lapse").await;
    assert_eq!(evidence.writer, AbandonWriter::ReconciledRequest);
    assert_eq!(
        evidence.owner.as_ref().map(|owner| owner.owner_id.as_str()),
        Some("lapsed-owner"),
        "the reconciled abandonment names the started owner as the lapsed owner"
    );
}

/// An OwnerBound row that has never started is claimable and runnable by any
/// worker (first execution is not re-execution): the runner records
/// `first_started` and drives it to a run terminal, not an Abandoned one.
#[tokio::test]
async fn owner_bound_unstarted_runs_once() {
    let registry: Arc<dyn ProcessRegistry> = Arc::new(TestLocalProcessRegistry::default());
    registry
        .register_process(registration_with_disposition(
            "proc-ob-unstarted",
            RecoveryDisposition::OwnerBound,
        ))
        .await
        .expect("register");

    let worker = inline_worker(
        Arc::clone(&registry),
        local_owner("live-worker", "host-a", "claimant-start"),
    );
    worker
        .drive_pending_processes()
        .await
        .expect("sweep dispatches");
    await_terminal(&registry, "proc-ob-unstarted").await;

    let record = registry
        .get_process("proc-ob-unstarted")
        .await
        .expect("process");
    assert!(
        record.first_started.is_some(),
        "the runner must record first_started before executing an unstarted OwnerBound row"
    );
    // A run terminal (Failed here, because the External placeholder input has
    // no execution runtime) — crucially NOT Abandoned. First execution ran.
    assert!(
        matches!(record.status, ProcessStatus::Failed { .. }),
        "an unstarted OwnerBound row reaches a run terminal, not an abandoned one, got {:?}",
        record.status
    );
}

/// Owner drain (ADR 0019): a host closing gracefully terminalizes its own
/// started OwnerBound work inline as `Abandoned{OwnerDrain}` under a live lease,
/// while leaving rerunnable, not-yet-started, and other-owner rows untouched.
#[tokio::test]
async fn drain_terminalizes_this_hosts_started_owner_bound_work() {
    let registry: Arc<dyn ProcessRegistry> = Arc::new(TestLocalProcessRegistry::default());
    let owner = local_owner("drain-host", "host-a", "start-a");
    let worker = inline_worker(Arc::clone(&registry), owner.clone());

    // (a) OwnerBound row this worker started -> drained.
    registry
        .register_process(registration_with_disposition(
            "mine-started",
            RecoveryDisposition::OwnerBound,
        ))
        .await
        .expect("register mine-started");
    registry
        .record_first_started(
            "mine-started",
            ProcessStarted {
                owner: owner.clone(),
                started_at_ms: 1,
            },
        )
        .await
        .expect("record first_started for mine-started");

    // (b) OwnerBound row a DIFFERENT owner started -> not ours to drain.
    registry
        .register_process(registration_with_disposition(
            "theirs-started",
            RecoveryDisposition::OwnerBound,
        ))
        .await
        .expect("register theirs-started");
    registry
        .record_first_started(
            "theirs-started",
            ProcessStarted {
                owner: local_owner("other-host", "host-b", "start-b"),
                started_at_ms: 1,
            },
        )
        .await
        .expect("record first_started for theirs-started");

    // (c) OwnerBound row never started -> still claimable by anyone.
    registry
        .register_process(registration_with_disposition(
            "mine-unstarted",
            RecoveryDisposition::OwnerBound,
        ))
        .await
        .expect("register mine-unstarted");

    // (d) Rerunnable in-flight row this worker started -> left non-terminal for
    // the next worker (its contract; drain never terminalizes rerunnable work).
    registry
        .register_process(registration_with_disposition(
            "rerunnable",
            RecoveryDisposition::Rerunnable,
        ))
        .await
        .expect("register rerunnable");
    registry
        .record_first_started(
            "rerunnable",
            ProcessStarted {
                owner: owner.clone(),
                started_at_ms: 1,
            },
        )
        .await
        .expect("record first_started for rerunnable");

    let report = worker.drain_owner_bound_work().await.expect("drain");
    assert_eq!(report.abandoned, vec!["mine-started".to_string()]);

    let evidence = abandoned_evidence(&registry, "mine-started").await;
    assert_eq!(evidence.writer, AbandonWriter::OwnerDrain);
    assert_eq!(evidence.owner.as_ref(), Some(&owner));

    for untouched in ["theirs-started", "mine-unstarted", "rerunnable"] {
        assert!(
            !registry
                .get_process(untouched)
                .await
                .expect("row exists")
                .is_terminal(),
            "{untouched} must be left non-terminal by owner drain",
        );
    }
}
