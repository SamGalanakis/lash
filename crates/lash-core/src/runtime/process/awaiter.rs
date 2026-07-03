use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::sync::watch;

use super::events::{
    ProcessAwaitOutput, ProcessEvent, ProcessEventAppendRequest, ProcessEventAppendResult,
};
use super::model::{
    ProcessExternalRef, ProcessHandleDescriptor, ProcessHandleGrant, ProcessHandleGrantEntry,
    ProcessLease, ProcessLeaseClaimOutcome, ProcessLeaseCompletion, ProcessListFilter,
    ProcessRecord, ProcessRegistration, ProcessSessionDeleteReport, SessionScope, WaitState,
};
use super::registry::ProcessRegistry;
use crate::PluginError;

const AWAIT_BACKOFF_MIN: Duration = Duration::from_millis(25);
const AWAIT_BACKOFF_MAX: Duration = Duration::from_secs(1);

#[derive(Clone, Default)]
pub struct ProcessChangeHub {
    inner: Arc<Mutex<HashMap<String, watch::Sender<u64>>>>,
}

impl ProcessChangeHub {
    pub fn new() -> Self {
        Self::default()
    }

    /// Subscribe before reading a process row. The receiver carries only a
    /// version counter; waiters always re-read the registry after a bump.
    pub fn subscribe(&self, process_id: &str) -> watch::Receiver<u64> {
        let mut guard = self.inner.lock().expect("process change hub lock poisoned");
        guard
            .entry(process_id.to_string())
            .or_insert_with(|| {
                let (tx, _rx) = watch::channel(0);
                tx
            })
            .subscribe()
    }

    pub fn notify(&self, process_id: &str) {
        let mut guard = self.inner.lock().expect("process change hub lock poisoned");
        let mut remove = false;
        if let Some(tx) = guard.get(process_id) {
            if tx.receiver_count() == 0 {
                remove = true;
            } else {
                let next = (*tx.borrow()).wrapping_add(1);
                if tx.send(next).is_err() {
                    remove = true;
                }
            }
        }
        if remove {
            guard.remove(process_id);
        }
    }

    #[cfg(test)]
    fn tracked_processes(&self) -> usize {
        self.inner
            .lock()
            .expect("process change hub lock poisoned")
            .len()
    }
}

struct WatchedProcessRegistry {
    inner: Arc<dyn ProcessRegistry>,
    hub: ProcessChangeHub,
}

pub fn watch_process_registry(
    inner: Arc<dyn ProcessRegistry>,
) -> (Arc<dyn ProcessRegistry>, ProcessChangeHub) {
    let hub = ProcessChangeHub::new();
    (
        Arc::new(WatchedProcessRegistry {
            inner,
            hub: hub.clone(),
        }),
        hub,
    )
}

#[derive(Clone)]
pub struct ProcessAwaiter {
    registry: Arc<dyn ProcessRegistry>,
    hub: Option<ProcessChangeHub>,
}

impl ProcessAwaiter {
    pub fn new(registry: Arc<dyn ProcessRegistry>, hub: ProcessChangeHub) -> Self {
        Self {
            registry,
            hub: Some(hub),
        }
    }

    pub fn polling(registry: Arc<dyn ProcessRegistry>) -> Self {
        Self {
            registry,
            hub: None,
        }
    }

    pub async fn await_terminal(
        &self,
        process_id: &str,
    ) -> Result<ProcessAwaitOutput, PluginError> {
        let mut backoff = AWAIT_BACKOFF_MIN;
        if let Some(hub) = self.hub.as_ref() {
            let mut rx = hub.subscribe(process_id);
            loop {
                if let Some(output) = self.read_terminal(process_id).await? {
                    return Ok(output);
                }
                tokio::select! {
                    changed = rx.changed() => {
                        match changed {
                            Ok(()) => backoff = AWAIT_BACKOFF_MIN,
                            // Sender dropped (unreachable today given the hub
                            // GC invariant, but latent): a dead receiver would
                            // otherwise fire immediately on every loop turn.
                            // Stop selecting on it and degrade to the
                            // sleep-only backoff loop below.
                            Err(_) => break,
                        }
                    }
                    _ = tokio::time::sleep(backoff) => {
                        backoff = next_backoff(backoff);
                    }
                }
            }
        }
        loop {
            if let Some(output) = self.read_terminal(process_id).await? {
                return Ok(output);
            }
            tokio::time::sleep(backoff).await;
            backoff = next_backoff(backoff);
        }
    }

    pub async fn await_event(
        &self,
        process_id: &str,
        event_type: &str,
        after_sequence: u64,
    ) -> Result<ProcessEvent, PluginError> {
        let mut backoff = AWAIT_BACKOFF_MIN;
        if let Some(hub) = self.hub.as_ref() {
            let mut rx = hub.subscribe(process_id);
            loop {
                if let Some(event) = self
                    .read_event(process_id, event_type, after_sequence)
                    .await?
                {
                    return Ok(event);
                }
                tokio::select! {
                    changed = rx.changed() => {
                        match changed {
                            Ok(()) => backoff = AWAIT_BACKOFF_MIN,
                            // Sender dropped (unreachable today given the hub
                            // GC invariant, but latent): a dead receiver would
                            // otherwise fire immediately on every loop turn.
                            // Stop selecting on it and degrade to the
                            // sleep-only backoff loop below.
                            Err(_) => break,
                        }
                    }
                    _ = tokio::time::sleep(backoff) => {
                        backoff = next_backoff(backoff);
                    }
                }
            }
        }
        loop {
            if let Some(event) = self
                .read_event(process_id, event_type, after_sequence)
                .await?
            {
                return Ok(event);
            }
            tokio::time::sleep(backoff).await;
            backoff = next_backoff(backoff);
        }
    }

    async fn read_terminal(
        &self,
        process_id: &str,
    ) -> Result<Option<ProcessAwaitOutput>, PluginError> {
        let record = self
            .registry
            .get_process(process_id)
            .await
            .ok_or_else(|| PluginError::Session(format!("unknown process `{process_id}`")))?;
        Ok(record.status.await_output().cloned())
    }

    async fn read_event(
        &self,
        process_id: &str,
        event_type: &str,
        after_sequence: u64,
    ) -> Result<Option<ProcessEvent>, PluginError> {
        Ok(self
            .registry
            .events_after(process_id, after_sequence)
            .await?
            .into_iter()
            .find(|event| event.event_type == event_type))
    }
}

fn next_backoff(current: Duration) -> Duration {
    current.saturating_mul(2).min(AWAIT_BACKOFF_MAX)
}

#[async_trait::async_trait]
pub trait ProcessAttach: Send + Sync {
    async fn await_terminal(&self, process_id: &str) -> Result<ProcessAwaitOutput, PluginError>;
}

#[async_trait::async_trait]
impl ProcessRegistry for WatchedProcessRegistry {
    fn durability_tier(&self) -> crate::DurabilityTier {
        self.inner.durability_tier()
    }

    async fn register_process(
        &self,
        registration: ProcessRegistration,
    ) -> Result<ProcessRecord, PluginError> {
        let process_id = registration.id.clone();
        let record = self.inner.register_process(registration).await?;
        self.hub.notify(&process_id);
        Ok(record)
    }

    async fn set_external_ref(
        &self,
        process_id: &str,
        external_ref: ProcessExternalRef,
    ) -> Result<ProcessRecord, PluginError> {
        let record = self
            .inner
            .set_external_ref(process_id, external_ref)
            .await?;
        self.hub.notify(process_id);
        Ok(record)
    }

    async fn grant_handle(
        &self,
        session_scope: &SessionScope,
        process_id: &str,
        descriptor: ProcessHandleDescriptor,
    ) -> Result<ProcessHandleGrant, PluginError> {
        self.inner
            .grant_handle(session_scope, process_id, descriptor)
            .await
    }

    async fn revoke_handle(
        &self,
        session_scope: &SessionScope,
        process_id: &str,
    ) -> Result<(), PluginError> {
        self.inner.revoke_handle(session_scope, process_id).await
    }

    async fn transfer_handle_grants(
        &self,
        from_scope: &SessionScope,
        to_scope: &SessionScope,
        process_ids: &[String],
    ) -> Result<(), PluginError> {
        self.inner
            .transfer_handle_grants(from_scope, to_scope, process_ids)
            .await
    }

    async fn list_handle_grants(
        &self,
        session_scope: &SessionScope,
    ) -> Result<Vec<ProcessHandleGrantEntry>, PluginError> {
        self.inner.list_handle_grants(session_scope).await
    }

    async fn list_live_handle_grants(
        &self,
        session_scope: &SessionScope,
    ) -> Result<Vec<ProcessHandleGrantEntry>, PluginError> {
        self.inner.list_live_handle_grants(session_scope).await
    }

    async fn has_handle_grant(
        &self,
        session_scope: &SessionScope,
        process_id: &str,
    ) -> Result<bool, PluginError> {
        self.inner.has_handle_grant(session_scope, process_id).await
    }

    async fn handle_grants_for_process(
        &self,
        process_id: &str,
    ) -> Result<Vec<ProcessHandleGrant>, PluginError> {
        self.inner.handle_grants_for_process(process_id).await
    }

    async fn delete_session_process_state(
        &self,
        session_id: &str,
    ) -> Result<ProcessSessionDeleteReport, PluginError> {
        self.inner.delete_session_process_state(session_id).await
    }

    async fn append_event(
        &self,
        process_id: &str,
        request: ProcessEventAppendRequest,
    ) -> Result<ProcessEventAppendResult, PluginError> {
        let result = self.inner.append_event(process_id, request).await?;
        self.hub.notify(process_id);
        Ok(result)
    }

    async fn events_after(
        &self,
        process_id: &str,
        after_sequence: u64,
    ) -> Result<Vec<ProcessEvent>, PluginError> {
        self.inner.events_after(process_id, after_sequence).await
    }

    async fn count_events_through(
        &self,
        process_id: &str,
        event_type: &str,
        up_to_sequence: u64,
    ) -> Result<u64, PluginError> {
        self.inner
            .count_events_through(process_id, event_type, up_to_sequence)
            .await
    }

    async fn recent_events(
        &self,
        process_id: &str,
        limit: usize,
    ) -> Result<Vec<ProcessEvent>, PluginError> {
        self.inner.recent_events(process_id, limit).await
    }

    async fn wake_events_after(
        &self,
        process_id: &str,
        after_sequence: u64,
    ) -> Result<Vec<ProcessEvent>, PluginError> {
        self.inner
            .wake_events_after(process_id, after_sequence)
            .await
    }

    async fn complete_process(
        &self,
        process_id: &str,
        await_output: ProcessAwaitOutput,
    ) -> Result<ProcessRecord, PluginError> {
        let record = self
            .inner
            .complete_process(process_id, await_output)
            .await?;
        self.hub.notify(process_id);
        Ok(record)
    }

    async fn set_process_wait(
        &self,
        process_id: &str,
        wait: WaitState,
    ) -> Result<ProcessRecord, PluginError> {
        let record = self.inner.set_process_wait(process_id, wait).await?;
        self.hub.notify(process_id);
        Ok(record)
    }

    async fn clear_process_wait(&self, process_id: &str) -> Result<ProcessRecord, PluginError> {
        let record = self.inner.clear_process_wait(process_id).await?;
        self.hub.notify(process_id);
        Ok(record)
    }

    async fn get_process(&self, process_id: &str) -> Option<ProcessRecord> {
        self.inner.get_process(process_id).await
    }

    async fn list_processes(
        &self,
        filter: &ProcessListFilter,
    ) -> Result<Vec<ProcessRecord>, PluginError> {
        self.inner.list_processes(filter).await
    }

    async fn ack_wake(&self, process_id: &str, sequence: u64) -> Result<(), PluginError> {
        self.inner.ack_wake(process_id, sequence).await?;
        self.hub.notify(process_id);
        Ok(())
    }

    async fn list_non_terminal(&self) -> Result<Vec<ProcessRecord>, PluginError> {
        self.inner.list_non_terminal().await
    }

    async fn claim_process_lease(
        &self,
        process_id: &str,
        owner: &crate::LeaseOwnerIdentity,
        lease_ttl_ms: u64,
    ) -> Result<ProcessLeaseClaimOutcome, PluginError> {
        self.inner
            .claim_process_lease(process_id, owner, lease_ttl_ms)
            .await
    }

    async fn reclaim_process_lease(
        &self,
        process_id: &str,
        owner: &crate::LeaseOwnerIdentity,
        observed_holder: &ProcessLease,
        lease_ttl_ms: u64,
    ) -> Result<ProcessLeaseClaimOutcome, PluginError> {
        self.inner
            .reclaim_process_lease(process_id, owner, observed_holder, lease_ttl_ms)
            .await
    }

    async fn renew_process_lease(
        &self,
        lease: &ProcessLease,
        lease_ttl_ms: u64,
    ) -> Result<ProcessLease, PluginError> {
        self.inner.renew_process_lease(lease, lease_ttl_ms).await
    }

    async fn complete_process_lease(
        &self,
        completion: &ProcessLeaseCompletion,
    ) -> Result<(), PluginError> {
        self.inner.complete_process_lease(completion).await
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::{
        ProcessInput, ProcessProvenance, ProcessRegistration, TestLocalProcessRegistry, ToolControl,
    };

    fn registration(process_id: &str) -> ProcessRegistration {
        ProcessRegistration::new(
            process_id,
            ProcessInput::External {
                metadata: serde_json::json!({}),
            },
            ProcessProvenance::host(),
        )
    }

    fn success(value: serde_json::Value) -> ProcessAwaitOutput {
        ProcessAwaitOutput::Success {
            value,
            control: None::<ToolControl>,
        }
    }

    #[tokio::test]
    async fn hub_subscribe_then_notify_wakes_and_gc_drops_empty_entry() {
        let hub = ProcessChangeHub::new();
        let mut rx = hub.subscribe("proc");
        hub.notify("proc");
        tokio::time::timeout(Duration::from_millis(100), rx.changed())
            .await
            .expect("notify should wake")
            .expect("sender remains open");

        drop(rx);
        hub.notify("proc");
        assert_eq!(hub.tracked_processes(), 0);
    }

    #[tokio::test]
    async fn await_event_returns_historical_event_immediately() {
        let raw = Arc::new(TestLocalProcessRegistry::default()) as Arc<dyn ProcessRegistry>;
        let (registry, hub) = watch_process_registry(raw);
        registry
            .register_process(registration("proc"))
            .await
            .expect("register");
        let appended = registry
            .append_event(
                "proc",
                ProcessEventAppendRequest::cancel_requested("proc", Some("stop".to_string())),
            )
            .await
            .expect("append");

        let event = ProcessAwaiter::new(Arc::clone(&registry), hub)
            .await_event("proc", "process.cancel_requested", 0)
            .await
            .expect("await event");
        assert_eq!(event.sequence, appended.event.sequence);
    }

    #[tokio::test]
    async fn await_terminal_unknown_process_errors() {
        let registry = Arc::new(TestLocalProcessRegistry::default()) as Arc<dyn ProcessRegistry>;
        let err = ProcessAwaiter::polling(registry)
            .await_terminal("missing")
            .await
            .expect_err("unknown process should error");
        assert!(err.to_string().contains("unknown process `missing`"));
    }

    #[tokio::test]
    async fn polling_awaiter_resolves_via_backoff() {
        let registry = Arc::new(TestLocalProcessRegistry::default()) as Arc<dyn ProcessRegistry>;
        registry
            .register_process(registration("proc"))
            .await
            .expect("register");
        let writer = Arc::clone(&registry);
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(10)).await;
            writer
                .complete_process("proc", success(serde_json::json!({ "ok": true })))
                .await
                .expect("complete");
        });

        let output = tokio::time::timeout(
            Duration::from_secs(1),
            ProcessAwaiter::polling(registry).await_terminal("proc"),
        )
        .await
        .expect("polling await timeout")
        .expect("await terminal");
        assert_eq!(output, success(serde_json::json!({ "ok": true })));
    }

    #[tokio::test]
    async fn watched_awaiter_observes_terminal_without_lost_wakeup() {
        let raw = Arc::new(TestLocalProcessRegistry::default()) as Arc<dyn ProcessRegistry>;
        let (registry, hub) = watch_process_registry(raw);
        registry
            .register_process(registration("proc"))
            .await
            .expect("register");
        let awaiter = ProcessAwaiter::new(Arc::clone(&registry), hub);
        let waiter = tokio::spawn(async move { awaiter.await_terminal("proc").await });
        registry
            .complete_process("proc", success(serde_json::json!("done")))
            .await
            .expect("complete");

        let output = tokio::time::timeout(Duration::from_millis(200), waiter)
            .await
            .expect("watched await timeout")
            .expect("join")
            .expect("await terminal");
        assert_eq!(output, success(serde_json::json!("done")));
    }

    #[tokio::test]
    async fn watched_registry_bumps_on_mutations() {
        let raw = Arc::new(TestLocalProcessRegistry::default()) as Arc<dyn ProcessRegistry>;
        let (registry, hub) = watch_process_registry(raw);
        let mut rx = hub.subscribe("proc");
        registry
            .register_process(registration("proc"))
            .await
            .expect("register");
        tokio::time::timeout(Duration::from_millis(100), rx.changed())
            .await
            .expect("register bump")
            .expect("sender remains open");

        registry
            .append_event(
                "proc",
                ProcessEventAppendRequest::cancel_requested("proc", None),
            )
            .await
            .expect("append");
        tokio::time::timeout(Duration::from_millis(100), rx.changed())
            .await
            .expect("append bump")
            .expect("sender remains open");
    }

    struct NoopRunHandle;

    #[async_trait::async_trait]
    impl crate::ProcessRunHandle for NoopRunHandle {
        async fn claim_and_run_pending(&self) -> Result<(), PluginError> {
            Ok(())
        }
    }

    struct PanicAttach;

    #[async_trait::async_trait]
    impl ProcessAttach for PanicAttach {
        async fn await_terminal(
            &self,
            _process_id: &str,
        ) -> Result<ProcessAwaitOutput, PluginError> {
            panic!("attach should not be called for already-terminal process")
        }
    }

    struct ErrorAttach;

    #[async_trait::async_trait]
    impl ProcessAttach for ErrorAttach {
        async fn await_terminal(
            &self,
            _process_id: &str,
        ) -> Result<ProcessAwaitOutput, PluginError> {
            Err(PluginError::Session("attach failed".to_string()))
        }
    }

    #[tokio::test]
    async fn driver_short_circuits_terminal_before_attach() {
        let raw = Arc::new(TestLocalProcessRegistry::default()) as Arc<dyn ProcessRegistry>;
        let driver = crate::ProcessWorkDriver::new(raw, Arc::new(NoopRunHandle))
            .with_attach(Arc::new(PanicAttach));
        let registry = driver.process_registry();
        registry
            .register_process(registration("proc"))
            .await
            .expect("register");
        registry
            .complete_process("proc", success(serde_json::json!("ready")))
            .await
            .expect("complete");

        let output = driver.await_terminal("proc").await.expect("await terminal");
        assert_eq!(output, success(serde_json::json!("ready")));
    }

    #[tokio::test]
    async fn driver_attach_errors_propagate_without_poll_fallback() {
        let raw = Arc::new(TestLocalProcessRegistry::default()) as Arc<dyn ProcessRegistry>;
        let driver = crate::ProcessWorkDriver::new(raw, Arc::new(NoopRunHandle))
            .with_attach(Arc::new(ErrorAttach));
        driver
            .process_registry()
            .register_process(registration("proc"))
            .await
            .expect("register");

        let err = driver
            .await_terminal("proc")
            .await
            .expect_err("attach error should propagate");
        assert!(err.to_string().contains("attach failed"));
    }
}
