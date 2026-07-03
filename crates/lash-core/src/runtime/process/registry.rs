use crate::plugin::PluginError;

use super::events::{
    ProcessAwaitOutput, ProcessEvent, ProcessEventAppendRequest, ProcessEventAppendResult,
};
use super::model::{
    ProcessExternalRef, ProcessHandleDescriptor, ProcessHandleGrant, ProcessHandleGrantEntry,
    ProcessLease, ProcessLeaseClaimOutcome, ProcessLeaseCompletion, ProcessListFilter,
    ProcessRecord, ProcessRegistration, ProcessSessionDeleteReport, SessionScope, WaitState,
};

/// Outcome of [`ProcessRegistry::prune_terminal_processes`]: how many terminal
/// process rows and event rows were physically deleted.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ProcessPruneReport {
    /// Terminal process rows deleted.
    pub pruned_processes: usize,
    /// Event rows deleted across those processes.
    pub pruned_events: usize,
}

/// Durability-neutral process registry.
///
/// Process waits are coordination behavior and live on
/// [`ProcessWorkDriver`](crate::ProcessWorkDriver) /
/// [`ProcessAwaiter`](crate::ProcessAwaiter), not on persistence
/// implementations. Registry methods are point reads and writes only. See
/// `docs/adr/0016-process-waits-live-on-the-work-driver-seam.md`.
#[async_trait::async_trait]
pub trait ProcessRegistry: Send + Sync {
    /// Durability tier this process registry provides; defaults to
    /// [`DurabilityTier`](crate::DurabilityTier)`::Inline`.
    fn durability_tier(&self) -> crate::DurabilityTier {
        crate::DurabilityTier::Inline
    }

    async fn register_process(
        &self,
        registration: ProcessRegistration,
    ) -> Result<ProcessRecord, PluginError>;

    /// Attach a durable backend reference to a registered process.
    ///
    /// Implementations must reject unknown process ids. The first assignment
    /// stores the reference. Repeating the exact same assignment is an
    /// idempotent no-op that returns the existing record unchanged. Assigning a
    /// different reference after one has been stored is a registry model error.
    async fn set_external_ref(
        &self,
        process_id: &str,
        external_ref: ProcessExternalRef,
    ) -> Result<ProcessRecord, PluginError>;

    async fn grant_handle(
        &self,
        session_scope: &SessionScope,
        process_id: &str,
        descriptor: ProcessHandleDescriptor,
    ) -> Result<ProcessHandleGrant, PluginError>;

    async fn revoke_handle(
        &self,
        session_scope: &SessionScope,
        process_id: &str,
    ) -> Result<(), PluginError>;

    async fn transfer_handle_grants(
        &self,
        from_scope: &SessionScope,
        to_scope: &SessionScope,
        process_ids: &[String],
    ) -> Result<(), PluginError>;

    async fn list_handle_grants(
        &self,
        session_scope: &SessionScope,
    ) -> Result<Vec<ProcessHandleGrantEntry>, PluginError>;

    async fn list_live_handle_grants(
        &self,
        session_scope: &SessionScope,
    ) -> Result<Vec<ProcessHandleGrantEntry>, PluginError> {
        Ok(self
            .list_handle_grants(session_scope)
            .await?
            .into_iter()
            .filter(|(_, record)| !record.is_terminal())
            .collect())
    }

    async fn has_handle_grant(
        &self,
        session_scope: &SessionScope,
        process_id: &str,
    ) -> Result<bool, PluginError> {
        Ok(self
            .list_handle_grants(session_scope)
            .await?
            .into_iter()
            .any(|(grant, _)| grant.process_id == process_id))
    }

    async fn handle_grants_for_process(
        &self,
        process_id: &str,
    ) -> Result<Vec<ProcessHandleGrant>, PluginError>;

    async fn delete_session_process_state(
        &self,
        session_id: &str,
    ) -> Result<ProcessSessionDeleteReport, PluginError>;

    async fn append_event(
        &self,
        process_id: &str,
        request: ProcessEventAppendRequest,
    ) -> Result<ProcessEventAppendResult, PluginError>;

    async fn events_after(
        &self,
        process_id: &str,
        after_sequence: u64,
    ) -> Result<Vec<ProcessEvent>, PluginError>;

    /// Count events of `event_type` with `sequence <= up_to_sequence`.
    ///
    /// This is the signal-ordinal query: the Nth occurrence of a signal event
    /// resolves the Nth durable wait key. The default scans the event log;
    /// store backends override it with a COUNT so per-signal cost stays flat
    /// instead of growing with a long-lived process's history.
    async fn count_events_through(
        &self,
        process_id: &str,
        event_type: &str,
        up_to_sequence: u64,
    ) -> Result<u64, PluginError> {
        Ok(self
            .events_after(process_id, 0)
            .await?
            .into_iter()
            .filter(|event| event.sequence <= up_to_sequence && event.event_type == event_type)
            .count() as u64)
    }

    /// The most recent `limit` events, in ascending sequence order.
    ///
    /// Observation snapshots use this to show a bounded activity tail without
    /// fetching a process's entire history on every poll. The default scans
    /// the event log; store backends override it with ORDER BY ... LIMIT.
    async fn recent_events(
        &self,
        process_id: &str,
        limit: usize,
    ) -> Result<Vec<ProcessEvent>, PluginError> {
        let mut events = self.events_after(process_id, 0).await?;
        if events.len() > limit {
            events.drain(..events.len() - limit);
        }
        Ok(events)
    }

    async fn wake_events_after(
        &self,
        process_id: &str,
        after_sequence: u64,
    ) -> Result<Vec<ProcessEvent>, PluginError>;

    async fn complete_process(
        &self,
        process_id: &str,
        await_output: ProcessAwaitOutput,
    ) -> Result<ProcessRecord, PluginError>;

    async fn set_process_wait(
        &self,
        process_id: &str,
        wait: WaitState,
    ) -> Result<ProcessRecord, PluginError>;

    async fn clear_process_wait(&self, process_id: &str) -> Result<ProcessRecord, PluginError>;

    async fn get_process(&self, process_id: &str) -> Option<ProcessRecord>;

    async fn list_processes(
        &self,
        filter: &ProcessListFilter,
    ) -> Result<Vec<ProcessRecord>, PluginError>;

    async fn ack_wake(&self, process_id: &str, sequence: u64) -> Result<(), PluginError>;

    /// All non-terminal process records, in stable `process_id` order.
    ///
    /// This is the recovery sweep's worklist: every process that was started
    /// but has not reached a terminal event is a candidate for re-execution by
    /// a [`DurableProcessWorker`](crate::DurableProcessWorker) after a crash.
    /// Terminal processes are excluded — they are already done and idempotent by
    /// `process_id`, so re-running them would be wasted work.
    async fn list_non_terminal(&self) -> Result<Vec<ProcessRecord>, PluginError>;

    /// Claim the durable single-owner lease over a non-terminal process.
    ///
    /// An unexpired lease held by a *different* owner returns
    /// [`ProcessLeaseClaimOutcome::Busy`] carrying the observed holder;
    /// claiming a free or expired lease succeeds and bumps the
    /// `fencing_token`, and the same incarnation re-entering its own live
    /// lease extends it without changing token or fence. The returned
    /// [`ProcessLease`]'s `(owner, lease_token)` plus `fencing_token` are the
    /// contract a worker presents on every subsequent renew/complete — a stale
    /// writer is rejected.
    async fn claim_process_lease(
        &self,
        process_id: &str,
        owner: &crate::LeaseOwnerIdentity,
        lease_ttl_ms: u64,
    ) -> Result<ProcessLeaseClaimOutcome, PluginError>;

    /// Reclaim an unexpired process lease whose observed holder is definitely
    /// dead according to persisted local-process liveness metadata.
    ///
    /// Mirrors
    /// [`RuntimePersistence::reclaim_session_execution_lease`](crate::RuntimePersistence::reclaim_session_execution_lease):
    /// backends must CAS on `observed_holder` (owner identity, lease token,
    /// and fencing token) so a stale claimant cannot clear a newer live lease
    /// that won the race after the busy observation, and a successful reclaim
    /// must advance the fencing token monotonically.
    async fn reclaim_process_lease(
        &self,
        process_id: &str,
        owner: &crate::LeaseOwnerIdentity,
        observed_holder: &ProcessLease,
        lease_ttl_ms: u64,
    ) -> Result<ProcessLeaseClaimOutcome, PluginError>;

    /// Extend the expiry of a live lease the caller still owns.
    ///
    /// The lease must match the persisted `(owner, lease_token, fencing_token)`
    /// and be unexpired, else the renewal is rejected (the lease was superseded
    /// or expired). Workers renew across long-running effects so a healthy
    /// process is not swept out from under its live owner.
    async fn renew_process_lease(
        &self,
        lease: &ProcessLease,
        lease_ttl_ms: u64,
    ) -> Result<ProcessLease, PluginError>;

    /// Release a lease the caller owns, fenced by the completion's
    /// `(process_id, lease_token)`.
    ///
    /// Mirrors clearing a runtime turn lease: a stale completion (whose token no
    /// longer matches the live lease) is a no-op so it cannot release a lease a
    /// newer owner now holds. Idempotent — completing an already-released lease
    /// succeeds.
    async fn complete_process_lease(
        &self,
        completion: &ProcessLeaseCompletion,
    ) -> Result<(), PluginError>;

    /// Physically delete terminal process rows whose `updated_at_ms` is older
    /// than `cutoff_epoch_ms`, together with their events, wake acks, handle
    /// grants, and lease rows. Host-scheduled retention: hosts that project
    /// results/events into their own store call this to keep the registry
    /// bounded. Non-terminal rows are never touched. Callers must choose a
    /// retention window comfortably longer than any waiter lifetime — a
    /// pruned process id becomes "unknown process" to late awaits.
    async fn prune_terminal_processes(
        &self,
        cutoff_epoch_ms: u64,
    ) -> Result<ProcessPruneReport, PluginError>;
}
