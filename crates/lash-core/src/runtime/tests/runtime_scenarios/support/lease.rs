use super::*;

impl RuntimeScenarioContext {
    pub(super) async fn lease_phase(&mut self, phase: RuntimeLeasePhase) {
        match phase {
            RuntimeLeasePhase::ReclaimDeadHolder {
                assert_stale_observed_holder_busy,
            } => {
                self.reclaim_dead_holder(assert_stale_observed_holder_busy)
                    .await
            }
        }
    }

    async fn reclaim_dead_holder(&mut self, assert_stale_observed_holder_busy: bool) {
        if self.lease.is_some() {
            panic!(
                "{} dead-holder reclaim must run before any other session lease claim",
                self.name
            );
        }
        let dead_owner = dead_local_lease_owner("runtime-scenario-dead-holder");
        let holder = self
            .store()
            .try_claim_session_execution_lease(self.session_id, &dead_owner, 60_000)
            .await
            .expect("claim dead-holder session execution lease")
            .acquired()
            .expect("dead-holder session execution lease");
        let claimant = local_lease_owner(self.host_behavior.lease_owner_id, "claimant-start");
        let busy = self
            .store()
            .try_claim_session_execution_lease(self.session_id, &claimant, 60_000)
            .await
            .expect("claimant observes busy dead-holder lease");
        assert!(
            matches!(busy, SessionExecutionLeaseClaimOutcome::Busy { .. }),
            "{} expected the dead-holder lease to be observed as busy before reclaim",
            self.name
        );
        let reclaimed = self
            .store()
            .reclaim_session_execution_lease(self.session_id, &claimant, &holder.fence(), 60_000)
            .await
            .expect("reclaim dead-holder session execution lease")
            .acquired()
            .expect("dead-holder session execution lease should be reclaimable");
        assert!(
            reclaimed.fencing_token > holder.fencing_token,
            "{} reclaimed session lease should advance the fencing token",
            self.name
        );
        if assert_stale_observed_holder_busy {
            let stale = self
                .store()
                .reclaim_session_execution_lease(
                    self.session_id,
                    &local_lease_owner("runtime-scenario-late-claimant", "late-claimant-start"),
                    &holder.fence(),
                    60_000,
                )
                .await
                .expect("stale observed-holder reclaim");
            assert!(
                matches!(stale, SessionExecutionLeaseClaimOutcome::Busy { .. }),
                "{} stale observed-holder reclaim should not clear the newer lease",
                self.name
            );
        }
        self.owner = Some(claimant);
        self.lease = Some(reclaimed);
    }
}
