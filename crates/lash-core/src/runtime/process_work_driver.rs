use std::sync::Arc;

use super::DurableProcessWorker;
use super::process::{
    ProcessAttach, ProcessAwaiter, ProcessChangeHub, ProcessEvent, ProcessEventSink,
    ProcessRegistry, watch_process_registry_with_sink,
};
use crate::{PluginError, ProcessAwaitOutput};

/// Registry and run handle for process work owned outside
/// [`LashCore`](https://docs.rs/lash/latest/lash/struct.LashCore.html).
///
/// The registry non-terminal rows are the durable work queue. Hosts drive that
/// queue explicitly by calling [`claim_and_run_pending`](Self::claim_and_run_pending)
/// on each relevant event. Cross-process idempotency belongs to the registry
/// claim; there is no core-owned polling loop.
#[derive(Clone)]
pub struct ProcessWorkDriver {
    registry: Arc<dyn ProcessRegistry>,
    run_handle: Arc<dyn ProcessRunHandle>,
    awaiter: ProcessAwaiter,
    attach: Option<Arc<dyn ProcessAttach>>,
    hub: ProcessChangeHub,
}

impl ProcessWorkDriver {
    pub fn new(registry: Arc<dyn ProcessRegistry>, run_handle: Arc<dyn ProcessRunHandle>) -> Self {
        Self::new_with_sink(registry, run_handle, None)
    }

    /// Like [`new`](Self::new), but installs a host-facing
    /// [`ProcessEventSink`] on the registry decorator this driver wraps.
    ///
    /// The sink receives every appended event, best-effort, after its durable
    /// write — see [`ProcessEventSink`] for the freshness-not-truth contract.
    pub fn new_with_sink(
        registry: Arc<dyn ProcessRegistry>,
        run_handle: Arc<dyn ProcessRunHandle>,
        sink: Option<Arc<dyn ProcessEventSink>>,
    ) -> Self {
        let (registry, hub) = watch_process_registry_with_sink(registry, sink);
        Self::from_watched(registry, hub, run_handle)
    }

    pub fn from_watched(
        registry: Arc<dyn ProcessRegistry>,
        hub: ProcessChangeHub,
        run_handle: Arc<dyn ProcessRunHandle>,
    ) -> Self {
        let awaiter = ProcessAwaiter::new(Arc::clone(&registry), hub.clone());
        Self {
            registry,
            run_handle,
            awaiter,
            attach: None,
            hub,
        }
    }

    pub fn with_attach(mut self, attach: Arc<dyn ProcessAttach>) -> Self {
        self.attach = Some(attach);
        self
    }

    pub fn inline(registry: Arc<dyn ProcessRegistry>, worker: DurableProcessWorker) -> Self {
        Self::new(registry, Arc::new(InlineProcessRunHandle::new(worker)))
    }

    pub fn process_registry(&self) -> Arc<dyn ProcessRegistry> {
        Arc::clone(&self.registry)
    }

    pub fn change_hub(&self) -> ProcessChangeHub {
        self.hub.clone()
    }

    pub fn awaiter(&self) -> ProcessAwaiter {
        self.awaiter.clone()
    }

    pub async fn await_terminal(
        &self,
        process_id: &str,
    ) -> Result<ProcessAwaitOutput, PluginError> {
        let record = self
            .registry
            .get_process(process_id)
            .await
            .ok_or_else(|| PluginError::Session(format!("unknown process `{process_id}`")))?;
        if let Some(output) = record.status.await_output() {
            return Ok(output.clone());
        }
        if let Some(attach) = self.attach.as_ref() {
            return attach.await_terminal(process_id).await;
        }
        self.awaiter.await_terminal(process_id).await
    }

    pub async fn await_event(
        &self,
        process_id: &str,
        event_type: &str,
        after_sequence: u64,
    ) -> Result<ProcessEvent, PluginError> {
        self.awaiter
            .await_event(process_id, event_type, after_sequence)
            .await
    }

    pub async fn claim_and_run_pending(&self, reason: &str) -> Result<(), PluginError> {
        if let Err(err) = self.run_handle.claim_and_run_pending().await {
            tracing::warn!("process work drive ({reason}) failed: {err}");
            return Err(err);
        }
        Ok(())
    }
}

/// One lease-protected drive of the registry's pending (non-terminal) processes.
///
/// Implementations claim the single-owner [`ProcessLease`](crate::ProcessLease)
/// per non-terminal row to fence execution, so a concurrent drive on another
/// owner skips an already-leased process and a process runs exactly once.
#[async_trait::async_trait]
pub trait ProcessRunHandle: Send + Sync {
    /// Claim and run every pending process this owner can claim, driving each to
    /// a terminal state. Idempotent: leased and terminal rows are skipped.
    async fn claim_and_run_pending(&self) -> Result<(), PluginError>;
}

/// Inline run handle: drives the worker's own lease-protected sweep in-process.
///
/// Delegates to [`DurableProcessWorker::drive_pending_processes`], the existing
/// `list_non_terminal -> claim lease -> run -> complete -> release` loop, so the
/// inline tier reuses the same coordination point as the durable tier.
pub struct InlineProcessRunHandle {
    worker: DurableProcessWorker,
}

impl InlineProcessRunHandle {
    pub fn new(worker: DurableProcessWorker) -> Self {
        Self { worker }
    }
}

#[async_trait::async_trait]
impl ProcessRunHandle for InlineProcessRunHandle {
    async fn claim_and_run_pending(&self) -> Result<(), PluginError> {
        self.worker.drive_pending_processes().await
    }
}
