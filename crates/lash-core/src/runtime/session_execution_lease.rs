use std::sync::Arc;
use std::sync::Mutex as StdMutex;
use std::sync::atomic::{AtomicBool, Ordering};

use tokio_util::sync::CancellationToken;

use super::Clock;
use crate::LeaseTimings;
use crate::store::{
    RuntimeCommit, RuntimeCommitResult, RuntimePersistence, SessionExecutionLease,
    SessionExecutionLeaseClaimOutcome, SessionExecutionLeaseCompletion, SessionExecutionLeaseFence,
    StoreError,
};

pub(super) struct SessionExecutionLeaseGuard {
    store: Arc<dyn RuntimePersistence>,
    lease: Arc<StdMutex<SessionExecutionLease>>,
    released: Arc<AtomicBool>,
    lost: Arc<AtomicBool>,
    timings: LeaseTimings,
    renew_task: tokio::task::JoinHandle<()>,
}

impl SessionExecutionLeaseGuard {
    pub(super) async fn try_acquire(
        store: Arc<dyn RuntimePersistence>,
        session_id: &str,
        owner: &crate::LeaseOwnerIdentity,
        timings: LeaseTimings,
        clock: Arc<dyn Clock>,
        cancel: CancellationToken,
    ) -> Result<Option<Self>, StoreError> {
        let lease = match store
            .try_claim_session_execution_lease(session_id, owner, timings.ttl_ms())
            .await?
        {
            SessionExecutionLeaseClaimOutcome::Acquired(lease) => lease,
            SessionExecutionLeaseClaimOutcome::Busy { holder }
                if holder.owner.is_definitely_dead_for_claimant(owner) =>
            {
                match store
                    .reclaim_session_execution_lease(
                        session_id,
                        owner,
                        &holder.fence(),
                        timings.ttl_ms(),
                    )
                    .await?
                {
                    SessionExecutionLeaseClaimOutcome::Acquired(lease) => lease,
                    SessionExecutionLeaseClaimOutcome::Busy { holder } => {
                        trace_busy(session_id, owner, &holder);
                        return Ok(None);
                    }
                }
            }
            SessionExecutionLeaseClaimOutcome::Busy { holder } => {
                trace_busy(session_id, owner, &holder);
                return Ok(None);
            }
        };
        tracing::debug!(
            session_id = %lease.session_id,
            owner_id = %lease.owner.owner_id,
            incarnation_id = %lease.owner.incarnation_id,
            fencing_token = lease.fencing_token,
            event = "session_execution_lease.acquired",
            "acquired session execution lease"
        );
        let lease = Arc::new(StdMutex::new(lease));
        let released = Arc::new(AtomicBool::new(false));
        let lost = Arc::new(AtomicBool::new(false));
        let renew_task = spawn_renewal_task(
            Arc::clone(&store),
            Arc::clone(&lease),
            Arc::clone(&released),
            Arc::clone(&lost),
            timings,
            clock,
            cancel,
        );
        Ok(Some(Self {
            store,
            lease,
            released,
            lost,
            timings,
            renew_task,
        }))
    }

    pub(super) fn fence(&self) -> SessionExecutionLeaseFence {
        self.lease.lock().expect("session lease lock").fence()
    }

    pub(super) fn completion(&self) -> SessionExecutionLeaseCompletion {
        self.lease.lock().expect("session lease lock").completion()
    }

    pub(super) fn mark_released(&self) {
        if self.released.swap(true, Ordering::AcqRel) {
            return;
        }
        self.renew_task.abort();
        let completion = self.completion();
        tracing::debug!(
            session_id = %completion.session_id,
            owner_id = %completion.owner.owner_id,
            incarnation_id = %completion.owner.incarnation_id,
            fencing_token = completion.fencing_token,
            event = "session_execution_lease.released",
            "released session execution lease"
        );
    }

    pub(super) fn is_lost(&self) -> bool {
        self.lost.load(Ordering::Acquire)
    }

    pub(super) async fn refresh_or_mark_lost(&self) -> Result<(), StoreError> {
        if self.is_lost() {
            let fence = self.fence();
            return Err(StoreError::SessionExecutionLeaseExpired {
                session_id: fence.session_id,
            });
        }
        let fence = self.fence();
        match self
            .store
            .renew_session_execution_lease(&fence, self.timings.ttl_ms())
            .await
        {
            Ok(renewed) => {
                tracing::debug!(
                    session_id = %renewed.session_id,
                    owner_id = %renewed.owner.owner_id,
                    incarnation_id = %renewed.owner.incarnation_id,
                    fencing_token = renewed.fencing_token,
                    event = "session_execution_lease.renewed",
                    "renewed session execution lease"
                );
                *self.lease.lock().expect("session lease lock") = renewed;
                Ok(())
            }
            Err(err) => {
                self.lost.store(true, Ordering::Release);
                self.renew_task.abort();
                tracing::warn!(
                    error = %err,
                    session_id = %fence.session_id,
                    owner_id = %fence.owner.owner_id,
                    incarnation_id = %fence.owner.incarnation_id,
                    fencing_token = fence.fencing_token,
                    event = "session_execution_lease.lost",
                    "lost session execution lease"
                );
                Err(err)
            }
        }
    }

    pub(super) async fn release_if_live(&self) -> Result<(), StoreError> {
        if self.released.swap(true, Ordering::AcqRel) {
            return Ok(());
        }
        self.renew_task.abort();
        if self.is_lost() {
            return Ok(());
        }
        let completion = self.completion();
        self.store
            .release_session_execution_lease(&completion)
            .await?;
        tracing::debug!(
            session_id = %completion.session_id,
            owner_id = %completion.owner.owner_id,
            incarnation_id = %completion.owner.incarnation_id,
            fencing_token = completion.fencing_token,
            event = "session_execution_lease.released",
            "released session execution lease"
        );
        Ok(())
    }
}

pub(super) async fn commit_runtime_state_with_fresh_session_execution_lease(
    store: Arc<dyn RuntimePersistence>,
    commit: RuntimeCommit,
    owner: &crate::LeaseOwnerIdentity,
    timings: LeaseTimings,
    clock: Arc<dyn Clock>,
) -> Result<RuntimeCommitResult, StoreError> {
    let session_id = commit.session_id.clone();
    let Some(lease) = SessionExecutionLeaseGuard::try_acquire(
        Arc::clone(&store),
        &session_id,
        owner,
        timings,
        clock,
        CancellationToken::new(),
    )
    .await?
    else {
        return Err(StoreError::Backend(format!(
            "session execution lease for session `{session_id}` is busy"
        )));
    };
    let commit = commit
        .with_session_execution_lease(lease.fence())
        .releasing_session_execution_lease(lease.completion());
    let result = store.commit_runtime_state(commit).await?;
    lease.mark_released();
    Ok(result)
}

impl Drop for SessionExecutionLeaseGuard {
    fn drop(&mut self) {
        self.renew_task.abort();
    }
}

fn spawn_renewal_task(
    store: Arc<dyn RuntimePersistence>,
    lease: Arc<StdMutex<SessionExecutionLease>>,
    released: Arc<AtomicBool>,
    lost: Arc<AtomicBool>,
    timings: LeaseTimings,
    clock: Arc<dyn Clock>,
    cancel: CancellationToken,
) -> tokio::task::JoinHandle<()> {
    let renew_every = timings.renew_interval();
    tokio::spawn(async move {
        loop {
            clock.sleep(renew_every).await;
            if released.load(Ordering::Acquire) {
                break;
            }
            let fence = lease.lock().expect("session lease lock").fence();
            match store
                .renew_session_execution_lease(&fence, timings.ttl_ms())
                .await
            {
                Ok(renewed) => {
                    tracing::debug!(
                        session_id = %renewed.session_id,
                        owner_id = %renewed.owner.owner_id,
                        incarnation_id = %renewed.owner.incarnation_id,
                        fencing_token = renewed.fencing_token,
                        event = "session_execution_lease.renewed",
                        "renewed session execution lease"
                    );
                    *lease.lock().expect("session lease lock") = renewed;
                }
                Err(err) => {
                    lost.store(true, Ordering::Release);
                    tracing::warn!(
                        error = %err,
                        session_id = %fence.session_id,
                        owner_id = %fence.owner.owner_id,
                        incarnation_id = %fence.owner.incarnation_id,
                        fencing_token = fence.fencing_token,
                        event = "session_execution_lease.lost",
                        "lost session execution lease"
                    );
                    cancel.cancel();
                    break;
                }
            }
        }
    })
}

fn trace_busy(
    session_id: &str,
    claimant: &crate::LeaseOwnerIdentity,
    holder: &SessionExecutionLease,
) {
    tracing::debug!(
        session_id,
        claimant_owner_id = %claimant.owner_id,
        claimant_incarnation_id = %claimant.incarnation_id,
        holder_owner_id = %holder.owner.owner_id,
        holder_incarnation_id = %holder.owner.incarnation_id,
        holder_fencing_token = holder.fencing_token,
        holder_expires_at_epoch_ms = holder.expires_at_epoch_ms,
        event = "session_execution_lease.busy",
        "session execution lease is busy"
    );
}
