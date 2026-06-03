use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::mpsc;

use crate::PluginError;

/// How often the queued-work runner checks for ready work absent a poke.
///
/// Pokes are the normal prompt path. Polling is the recovery path for queued
/// work that already existed at startup or whose wake notification was dropped.
const QUEUED_WORK_POLL_INTERVAL: Duration = Duration::from_millis(400);

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

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum QueuedWorkRunOutcome {
    Submitted { session_id: String },
    Idle,
}

#[async_trait::async_trait]
pub trait QueuedWorkRunHandle: Send + Sync {
    async fn run_queued_work(
        &self,
        request: QueuedWorkRunRequest,
    ) -> Result<QueuedWorkRunOutcome, PluginError>;
}

enum QueuedWorkRunnerCommand {
    Poke {
        session_id: Option<String>,
        reason: String,
    },
    Complete {
        session_id: String,
        reason: String,
    },
}

pub struct QueuedWorkRunner {
    run_handle: Arc<dyn QueuedWorkRunHandle>,
    tx: mpsc::UnboundedSender<QueuedWorkRunnerCommand>,
    rx: mpsc::UnboundedReceiver<QueuedWorkRunnerCommand>,
}

impl QueuedWorkRunner {
    pub fn new(run_handle: Arc<dyn QueuedWorkRunHandle>) -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        Self { run_handle, tx, rx }
    }

    pub fn poke_handle(&self) -> QueuedWorkPoke {
        QueuedWorkPoke {
            tx: self.tx.clone(),
        }
    }

    pub fn spawn(self) -> QueuedWorkPoke {
        let poke = self.poke_handle();
        tokio::spawn(async move {
            self.run().await;
        });
        poke
    }

    async fn run(mut self) {
        let mut inflight = HashSet::new();
        self.drive(
            QueuedWorkRunRequest::new(None, "startup", false),
            &mut inflight,
        )
        .await;
        let mut poll = tokio::time::interval(QUEUED_WORK_POLL_INTERVAL);
        poll.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            tokio::select! {
                command = self.rx.recv() => {
                    let Some(command) = command else {
                        break;
                    };
                    match command {
                        QueuedWorkRunnerCommand::Poke { session_id, reason } => {
                            self.drive(
                                QueuedWorkRunRequest::new(session_id, reason, true),
                                &mut inflight,
                            )
                            .await;
                        }
                        QueuedWorkRunnerCommand::Complete { session_id, reason } => {
                            inflight.remove(&session_id);
                            self.drive(
                                QueuedWorkRunRequest::new(Some(session_id), reason, false),
                                &mut inflight,
                            )
                            .await;
                        }
                    }
                }
                _ = poll.tick() => {
                    self.drive(
                        QueuedWorkRunRequest::new(None, "poll", false),
                        &mut inflight,
                    )
                    .await;
                }
            }
        }
    }

    async fn drive(&self, request: QueuedWorkRunRequest, inflight: &mut HashSet<String>) {
        if let Some(session_id) = request.session_id.as_deref()
            && inflight.contains(session_id)
        {
            return;
        }
        if request.session_id.is_none() && !inflight.is_empty() {
            return;
        }
        match self.run_handle.run_queued_work(request).await {
            Ok(QueuedWorkRunOutcome::Submitted { session_id }) => {
                inflight.insert(session_id);
            }
            Ok(QueuedWorkRunOutcome::Idle) => {}
            Err(err) => tracing::warn!("queued work runner drive failed: {err}"),
        }
    }
}

#[derive(Clone)]
pub struct QueuedWorkPoke {
    tx: mpsc::UnboundedSender<QueuedWorkRunnerCommand>,
}

impl QueuedWorkPoke {
    pub fn poke(&self, reason: impl Into<String>) {
        let _ = self.tx.send(QueuedWorkRunnerCommand::Poke {
            session_id: None,
            reason: reason.into(),
        });
    }

    pub fn poke_session(&self, session_id: impl Into<String>, reason: impl Into<String>) {
        let _ = self.tx.send(QueuedWorkRunnerCommand::Poke {
            session_id: Some(session_id.into()),
            reason: reason.into(),
        });
    }

    pub fn complete_session(&self, session_id: impl Into<String>, reason: impl Into<String>) {
        let _ = self.tx.send(QueuedWorkRunnerCommand::Complete {
            session_id: session_id.into(),
            reason: reason.into(),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{HashSet, VecDeque};
    use std::sync::Mutex;

    struct RecordingQueuedWorkRunHandle {
        requests: mpsc::UnboundedSender<QueuedWorkRunRequest>,
        responses: Mutex<VecDeque<QueuedWorkRunOutcome>>,
    }

    #[async_trait::async_trait]
    impl QueuedWorkRunHandle for RecordingQueuedWorkRunHandle {
        async fn run_queued_work(
            &self,
            request: QueuedWorkRunRequest,
        ) -> Result<QueuedWorkRunOutcome, PluginError> {
            self.requests
                .send(request)
                .expect("record queued work request");
            Ok(self
                .responses
                .lock()
                .expect("lock responses")
                .pop_front()
                .unwrap_or(QueuedWorkRunOutcome::Idle))
        }
    }

    fn recording_handle(
        responses: impl IntoIterator<Item = QueuedWorkRunOutcome>,
    ) -> (
        Arc<RecordingQueuedWorkRunHandle>,
        mpsc::UnboundedReceiver<QueuedWorkRunRequest>,
    ) {
        let (requests, request_rx) = mpsc::unbounded_channel();
        (
            Arc::new(RecordingQueuedWorkRunHandle {
                requests,
                responses: Mutex::new(responses.into_iter().collect()),
            }),
            request_rx,
        )
    }

    #[tokio::test]
    async fn drive_holds_submitted_session_inflight_until_completion() {
        let (handle, mut requests) = recording_handle([
            QueuedWorkRunOutcome::Submitted {
                session_id: "root".to_string(),
            },
            QueuedWorkRunOutcome::Idle,
        ]);
        let runner = QueuedWorkRunner::new(handle);
        let mut inflight = HashSet::new();

        runner
            .drive(
                QueuedWorkRunRequest::new(Some("root".to_string()), "process_wake", true),
                &mut inflight,
            )
            .await;

        let first = requests.try_recv().expect("first queued work request");
        assert_eq!(first.session_id.as_deref(), Some("root"));
        assert_eq!(first.reason, "process_wake");
        assert!(first.trace_idle);
        assert!(inflight.contains("root"));

        runner
            .drive(
                QueuedWorkRunRequest::new(Some("root".to_string()), "duplicate", true),
                &mut inflight,
            )
            .await;
        assert!(
            requests.try_recv().is_err(),
            "inflight session should suppress duplicate submission"
        );

        runner
            .drive(
                QueuedWorkRunRequest::new(None, "poll", false),
                &mut inflight,
            )
            .await;
        assert!(
            requests.try_recv().is_err(),
            "global poll should not submit while work is inflight"
        );

        inflight.remove("root");
        runner
            .drive(
                QueuedWorkRunRequest::new(Some("root".to_string()), "queued_turn_completed", false),
                &mut inflight,
            )
            .await;

        let resumed = requests
            .try_recv()
            .expect("completion should re-drive the session");
        assert_eq!(resumed.session_id.as_deref(), Some("root"));
        assert_eq!(resumed.reason, "queued_turn_completed");
        assert!(!resumed.trace_idle);
    }
}
