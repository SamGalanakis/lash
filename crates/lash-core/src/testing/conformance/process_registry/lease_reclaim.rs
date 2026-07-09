use super::*;

/// Fenced reclaim of a dead holder's process lease, mirroring the session
/// execution lane's `session_execution_lease_reclaim_contract`:
///
/// - a plain claim against a live-but-dead holder reports busy; the fenced
///   reclaim acquires and advances the fencing token;
/// - a stale observed holder must not clear the newer lease;
/// - a fenced reclaim race has exactly one winner;
/// - a holder on another host (or with opaque liveness) is never provably
///   dead and stays busy.
pub(super) async fn process_lease_reclaim_contract(registry: Arc<dyn ProcessRegistry>) {
    let pid = std::process::id();
    let dead_holder = local_process_lease_owner(
        "dead-holder",
        "host-a",
        "boot-a",
        pid,
        "not-the-current-process-start",
    );
    let claimant = local_process_lease_owner("claimant", "host-a", "boot-a", pid, "claimant-start");

    registry
        .register_process(registration("proc-lease-reclaim-dead"))
        .await
        .expect("register reclaim-dead");
    let holder = registry
        .claim_process_lease("proc-lease-reclaim-dead", &dead_holder, 60_000)
        .await
        .expect("claim dead-holder lease")
        .acquired()
        .expect("dead-holder lease acquired");
    assert!(
        matches!(
            registry
                .claim_process_lease("proc-lease-reclaim-dead", &claimant, 60_000)
                .await
                .expect("try claimant against dead holder"),
            crate::ProcessLeaseClaimOutcome::Busy { .. }
        ),
        "plain claim must report busy before the caller performs fenced reclaim"
    );
    let reclaimed = registry
        .reclaim_process_lease("proc-lease-reclaim-dead", &claimant, &holder, 60_000)
        .await
        .expect("reclaim dead holder")
        .acquired()
        .expect("dead holder is reclaimable before ttl");
    assert!(
        reclaimed.fencing_token > holder.fencing_token,
        "fenced reclaim must advance the fencing token"
    );
    let stale_reclaim = registry
        .reclaim_process_lease(
            "proc-lease-reclaim-dead",
            &local_process_lease_owner(
                "late-claimant",
                "host-a",
                "boot-a",
                pid,
                "late-claimant-start",
            ),
            &holder,
            60_000,
        )
        .await
        .expect("stale observed-holder reclaim");
    assert!(
        matches!(stale_reclaim, crate::ProcessLeaseClaimOutcome::Busy { .. }),
        "a stale observed holder must not clear the newer lease"
    );
    registry
        .complete_process_lease(&ProcessLeaseCompletion::from_lease(&reclaimed))
        .await
        .expect("release reclaimed lease");

    registry
        .register_process(registration("proc-lease-reclaim-race"))
        .await
        .expect("register reclaim-race");
    let race_holder = registry
        .claim_process_lease("proc-lease-reclaim-race", &dead_holder, 60_000)
        .await
        .expect("claim race holder")
        .acquired()
        .expect("race holder acquired");
    let barrier = Arc::new(tokio::sync::Barrier::new(3));
    let left_registry = Arc::clone(&registry);
    let right_registry = Arc::clone(&registry);
    let left_barrier = Arc::clone(&barrier);
    let right_barrier = Arc::clone(&barrier);
    let left_holder = race_holder.clone();
    let right_holder = race_holder.clone();
    let left_claimant =
        local_process_lease_owner("race-left", "host-a", "boot-a", pid, "race-left-start");
    let right_claimant =
        local_process_lease_owner("race-right", "host-a", "boot-a", pid, "race-right-start");
    let left = tokio::spawn(async move {
        left_barrier.wait().await;
        left_registry
            .reclaim_process_lease(
                "proc-lease-reclaim-race",
                &left_claimant,
                &left_holder,
                60_000,
            )
            .await
    });
    let right = tokio::spawn(async move {
        right_barrier.wait().await;
        right_registry
            .reclaim_process_lease(
                "proc-lease-reclaim-race",
                &right_claimant,
                &right_holder,
                60_000,
            )
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
        .filter_map(crate::ProcessLeaseClaimOutcome::acquired)
        .collect::<Vec<_>>();
    assert_eq!(
        race_winners.len(),
        1,
        "exactly one claimant may win a fenced reclaim race"
    );
    let race_winner = race_winners.pop().expect("race winner");
    assert!(race_winner.fencing_token > race_holder.fencing_token);
    registry
        .complete_process_lease(&ProcessLeaseCompletion::from_lease(&race_winner))
        .await
        .expect("release race winner");

    registry
        .register_process(registration("proc-lease-reclaim-cross-host"))
        .await
        .expect("register reclaim-cross-host");
    let cross_host_holder = registry
        .claim_process_lease("proc-lease-reclaim-cross-host", &dead_holder, 60_000)
        .await
        .expect("claim cross-host holder")
        .acquired()
        .expect("cross-host holder acquired");
    let cross_host_result = registry
        .reclaim_process_lease(
            "proc-lease-reclaim-cross-host",
            &local_process_lease_owner(
                "cross-host-claimant",
                "host-b",
                "boot-a",
                pid,
                "claimant-start",
            ),
            &cross_host_holder,
            60_000,
        )
        .await
        .expect("cross-host reclaim resolves");
    assert!(
        matches!(
            cross_host_result,
            crate::ProcessLeaseClaimOutcome::Busy { .. }
        ),
        "a holder on another host is never provably dead and must stay busy"
    );

    registry
        .register_process(registration("proc-lease-reclaim-opaque"))
        .await
        .expect("register reclaim-opaque");
    let opaque_holder = registry
        .claim_process_lease(
            "proc-lease-reclaim-opaque",
            &process_lease_owner("opaque-holder"),
            60_000,
        )
        .await
        .expect("claim opaque holder")
        .acquired()
        .expect("opaque holder acquired");
    let opaque_result = registry
        .reclaim_process_lease(
            "proc-lease-reclaim-opaque",
            &claimant,
            &opaque_holder,
            60_000,
        )
        .await
        .expect("opaque reclaim resolves");
    assert!(
        matches!(opaque_result, crate::ProcessLeaseClaimOutcome::Busy { .. }),
        "an opaque holder carries no liveness proof and must stay busy"
    );
}
