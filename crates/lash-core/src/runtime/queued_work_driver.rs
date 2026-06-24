use std::sync::Arc;

use crate::PluginError;

#[derive(Clone, Debug)]
pub struct QueuedWorkRunRequest {
    pub session_id: Option<String>,
    pub reason: String,
    pub trace_idle: bool,
}

impl QueuedWorkRunRequest {
    fn new(session_id: Option<String>, reason: impl Into<String>, trace_idle: bool) -> Self {
        Self {
            session_id,
            reason: reason.into(),
            trace_idle,
        }
    }
}

#[async_trait::async_trait]
pub trait QueuedWorkRunHandle: Send + Sync {
    async fn run_queued_work(&self, request: QueuedWorkRunRequest) -> Result<(), PluginError>;

    /// Host-driven single pass: claim and submit ready queued work, optionally
    /// narrowed to one session. The symmetric counterpart to
    /// [`ProcessRunHandle::claim_and_run_pending`](super::ProcessRunHandle::claim_and_run_pending).
    ///
    /// Idempotency is the store scheduler's job, not a same-process memory
    /// guard. Hosts call this on an event (enqueue, process wake, turn
    /// completion) instead of polling.
    async fn claim_and_run_pending(
        &self,
        session_id: Option<&str>,
        reason: &str,
    ) -> Result<(), PluginError> {
        let request =
            QueuedWorkRunRequest::new(session_id.map(str::to_string), reason.to_string(), false);
        self.run_queued_work(request).await
    }
}

#[derive(Clone)]
pub struct QueuedWorkDriver {
    run_handle: Arc<dyn QueuedWorkRunHandle>,
}

impl QueuedWorkDriver {
    pub fn new(run_handle: Arc<dyn QueuedWorkRunHandle>) -> Self {
        Self { run_handle }
    }

    pub async fn claim_and_run_pending(
        &self,
        session_id: Option<&str>,
        reason: &str,
    ) -> Result<(), PluginError> {
        if let Err(err) = self
            .run_handle
            .claim_and_run_pending(session_id, reason)
            .await
        {
            tracing::warn!("queued work drive ({reason}) failed: {err}");
            return Err(err);
        }
        Ok(())
    }
}
