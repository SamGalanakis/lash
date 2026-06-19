use std::sync::Arc;

use super::DurableProcessWorker;
use super::process::ProcessRegistry;
use crate::PluginError;

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
}

impl ProcessWorkDriver {
    pub fn new(registry: Arc<dyn ProcessRegistry>, run_handle: Arc<dyn ProcessRunHandle>) -> Self {
        Self {
            registry,
            run_handle,
        }
    }

    pub fn inline(registry: Arc<dyn ProcessRegistry>, worker: DurableProcessWorker) -> Self {
        Self::new(registry, Arc::new(InlineProcessRunHandle::new(worker)))
    }

    pub fn process_registry(&self) -> Arc<dyn ProcessRegistry> {
        Arc::clone(&self.registry)
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
