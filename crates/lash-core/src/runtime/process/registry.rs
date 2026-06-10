use crate::plugin::PluginError;

use super::events::{
    ProcessAwaitOutput, ProcessEvent, ProcessEventAppendRequest, ProcessEventAppendResult,
};
use super::model::{
    ProcessExternalRef, ProcessHandleDescriptor, ProcessHandleGrant, ProcessHandleGrantEntry,
    ProcessLease, ProcessLeaseCompletion, ProcessListFilter, ProcessRecord, ProcessRegistration,
    ProcessSessionDeleteReport, SessionScope, WaitState,
};

/// Durability-neutral process registry.
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

    async fn wake_events_after(
        &self,
        process_id: &str,
        after_sequence: u64,
    ) -> Result<Vec<ProcessEvent>, PluginError>;

    async fn wait_event_after(
        &self,
        process_id: &str,
        event_type: &str,
        after_sequence: u64,
    ) -> Result<ProcessEvent, PluginError>;

    async fn await_process(&self, process_id: &str) -> Result<ProcessAwaitOutput, PluginError>;

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
    /// An unexpired lease held by a *different* owner fences the claim (returns
    /// an error); claiming a free, expired, or own lease succeeds and bumps the
    /// `fencing_token`. The returned [`ProcessLease`]'s
    /// `(owner_id, lease_token)` plus `fencing_token` are the contract a worker
    /// presents on every subsequent renew/complete — a stale writer is rejected.
    async fn claim_process_lease(
        &self,
        process_id: &str,
        owner_id: &str,
        lease_ttl_ms: u64,
    ) -> Result<ProcessLease, PluginError>;

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
}
