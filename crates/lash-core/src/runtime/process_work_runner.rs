use std::sync::Arc;
use std::time::Duration;

use tokio::sync::Notify;

use super::DurableProcessWorker;
use super::process::ProcessRegistry;
use crate::PluginError;

/// How often the runner re-drives pending processes absent a poke.
///
/// Pokes make consumption prompt; the poll is a safety net that picks up work
/// no poke reached (a crash-orphaned non-terminal row, or a poke dropped while
/// the runner was mid-drive).
const PROCESS_WORK_POLL_INTERVAL: Duration = Duration::from_millis(400);

/// Drives the registry's non-terminal rows (the durable work queue) to terminal
/// on poke, on a poll tick, and once at startup.
///
/// The registry non-terminal rows *are* the queue; a poke makes consumption
/// prompt. The single coordination point is the `ProcessLease` claimed inside
/// the [`ProcessRunHandle`], so a poke is idempotent (a leased or terminal row
/// is skipped) and the same control seam can poke after any process start.
///
/// The loop is a [`tokio::select`] over a [`Notify`] (poke) and an interval
/// (poll), plus one startup drive that folds in the former startup-only
/// recovery sweep.
pub struct ProcessWorkRunner {
    run_handle: Arc<dyn ProcessRunHandle>,
    notify: Arc<Notify>,
}

impl ProcessWorkRunner {
    /// Build a runner over the given [`ProcessRunHandle`].
    pub fn new(run_handle: Arc<dyn ProcessRunHandle>) -> Self {
        Self {
            run_handle,
            notify: Arc::new(Notify::new()),
        }
    }

    /// Build a runner that drives an inline [`DurableProcessWorker`] directly.
    pub fn inline(worker: DurableProcessWorker) -> Self {
        Self::new(Arc::new(InlineProcessRunHandle::new(worker)))
    }

    /// A cloneable poke handle that wakes the loop. Hand a clone to the control
    /// seam so a successful process start can make consumption prompt.
    pub fn poke_handle(&self) -> ProcessWorkPoke {
        ProcessWorkPoke {
            notify: Arc::clone(&self.notify),
        }
    }

    /// Spawn the loop on the current tokio runtime, returning the poke handle.
    ///
    /// The loop drives once at startup (folding in the former startup-only
    /// recovery sweep), then on every poke and every poll tick until the
    /// process exits. Each drive is idempotent, so a poke racing a poll never
    /// double-runs a process.
    pub fn spawn(self) -> ProcessWorkPoke {
        let poke = self.poke_handle();
        tokio::spawn(async move {
            self.run().await;
        });
        poke
    }

    async fn run(self) {
        // Startup drive: the runner first tick replaces the startup-only
        // recovery sweep, so crash-orphaned non-terminal rows are picked up
        // without a separate boot-time sweep.
        self.drive("startup").await;
        let mut poll = tokio::time::interval(PROCESS_WORK_POLL_INTERVAL);
        poll.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            tokio::select! {
                _ = self.notify.notified() => {
                    self.drive("poke").await;
                }
                _ = poll.tick() => {
                    self.drive("poll").await;
                }
            }
        }
    }

    async fn drive(&self, reason: &str) {
        if let Err(err) = self.run_handle.claim_and_run_pending().await {
            tracing::warn!("process work runner drive ({reason}) failed: {err}");
        }
    }
}

/// Registry and wake handle for a process work runner owned outside
/// [`LashCore`](https://docs.rs/lash/latest/lash/struct.LashCore.html).
///
/// Durable deployments use this to bind one registry to the external runner
/// that consumes that registry's non-terminal process rows. The facade can then
/// configure process lifecycle support from the driver without accepting a
/// second, potentially divergent registry argument.
#[derive(Clone)]
pub struct ProcessWorkDriver {
    registry: Arc<dyn ProcessRegistry>,
    poke: ProcessWorkPoke,
}

impl ProcessWorkDriver {
    pub fn new(registry: Arc<dyn ProcessRegistry>, poke: ProcessWorkPoke) -> Self {
        Self { registry, poke }
    }

    pub fn process_registry(&self) -> Arc<dyn ProcessRegistry> {
        Arc::clone(&self.registry)
    }

    pub fn poke_handle(&self) -> ProcessWorkPoke {
        self.poke.clone()
    }
}

/// Cloneable handle that wakes a [`ProcessWorkRunner`] loop.
///
/// Poking is idempotent — the runner skips leased and terminal rows — so the
/// control seam can poke after any successful process start (in-turn-inline,
/// trigger or trigger occurrence) without coordinating with the runner.
#[derive(Clone)]
pub struct ProcessWorkPoke {
    notify: Arc<Notify>,
}

impl ProcessWorkPoke {
    /// Wake the runner to drive pending processes promptly.
    pub fn poke(&self) {
        self.notify.notify_one();
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
