use super::*;

impl RuntimeScenarioContext {
    pub(super) async fn fault(&mut self, phase: RuntimeFaultPhase) {
        match phase {
            RuntimeFaultPhase::StaleQueueCompletion => self.stale_queue_completion_fault().await,
            RuntimeFaultPhase::CommitAfterSessionLeaseRelease => {
                self.commit_after_session_lease_release_fault().await
            }
        }
    }

    async fn stale_queue_completion_fault(&mut self) {
        self.ensure_lease().await;
        let claim = self
            .turn_claim
            .as_ref()
            .expect("stale queue-completion fault requires a prior TurnWorkClaim phase");
        let (_, lease) = self.owner_and_lease();
        let mut stale_completion = claim.completion();
        stale_completion.lease_token.push_str(":stale");
        let err = self
            .store()
            .commit_runtime_state(
                RuntimeCommit::persisted_state(&self.state, &[])
                    .with_session_execution_lease(lease.fence())
                    .completing_queue_claim(stale_completion),
            )
            .await
            .expect_err("stale queue-completion fault should reject the commit");
        assert!(
            matches!(err, StoreError::QueuedWorkClaimSuperseded { .. }),
            "{} stale queue-completion fault produced the wrong error: {err:?}",
            self.name
        );
    }

    async fn commit_after_session_lease_release_fault(&mut self) {
        if !self.lease_released {
            self.commit(RuntimeCommitPhase::new()).await;
        }
        let lease = self
            .lease
            .as_ref()
            .expect("released-lease fault requires a previous session lease");
        self.state.turn_index = self.state.turn_index.saturating_add(1);
        let err = self
            .store()
            .commit_runtime_state(
                RuntimeCommit::persisted_state(&self.state, &[])
                    .with_session_execution_lease(lease.fence()),
            )
            .await
            .expect_err("released session execution lease should reject follow-up commit");
        assert!(
            matches!(
                err,
                StoreError::SessionExecutionLeaseExpired { ref session_id }
                    if session_id == self.session_id
            ),
            "{} released session execution lease produced the wrong error: {err:?}",
            self.name
        );
    }
}
