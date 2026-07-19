use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

use crate::{ErrorEnvelope, TurnOutcome};

use super::{
    AwaitEventKey, AwaitEventResolver, AwaitEventWaitIdentity, EffectHost, ExecutionScope,
    Resolution, ResolveOutcome, RuntimeError,
};

/// Stable routing identity for one foreground turn.
///
/// These identifiers select work; they are not authorization credentials.
/// Hosts exposing turn control to untrusted callers must authenticate and
/// authorize the request before calling Lash.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TurnAddress {
    pub session_id: String,
    pub turn_id: String,
}

impl TurnAddress {
    pub fn new(session_id: impl Into<String>, turn_id: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
            turn_id: turn_id.into(),
        }
    }

    fn scope(&self) -> ExecutionScope {
        ExecutionScope::turn(&self.session_id, &self.turn_id)
    }

    fn validate(&self) -> Result<(), RuntimeError> {
        self.scope().validate()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TurnCancelSource {
    UserInterrupt,
    Host,
    Shutdown,
    Superseded,
}

/// Shared source hint for a process-local cancellation token.
///
/// The hint is set by the local entry point that fires the token. It is not a
/// durable cancellation request and must not be used as authorization.
#[doc(hidden)]
#[derive(Clone, Default)]
pub struct TurnCancelSourceHint {
    source: Arc<Mutex<Option<TurnCancelSource>>>,
}

impl TurnCancelSourceHint {
    pub fn set(&self, source: TurnCancelSource) {
        let mut hint = self.source.lock().expect("turn cancel source hint lock");
        if hint.is_none() {
            *hint = Some(source);
        }
    }

    pub(crate) fn get(&self) -> Option<TurnCancelSource> {
        *self.source.lock().expect("turn cancel source hint lock")
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TurnCancellationEvidence {
    pub request_id: String,
    pub source: TurnCancelSource,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TurnCancelRequest {
    pub address: TurnAddress,
    pub request_id: String,
    pub source: TurnCancelSource,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

impl TurnCancelRequest {
    pub fn new(
        address: TurnAddress,
        request_id: impl Into<String>,
        source: TurnCancelSource,
    ) -> Self {
        Self {
            address,
            request_id: request_id.into(),
            source,
            reason: None,
        }
    }

    pub fn with_reason(mut self, reason: impl Into<String>) -> Self {
        self.reason = Some(reason.into());
        self
    }

    fn validate(&self) -> Result<(), RuntimeError> {
        self.address.validate()?;
        if self.request_id.trim().is_empty() {
            return Err(RuntimeError::new(
                "invalid_turn_cancel_request",
                "turn cancellation requires a non-empty request id",
            ));
        }
        Ok(())
    }

    fn evidence(&self) -> TurnCancellationEvidence {
        TurnCancellationEvidence {
            request_id: self.request_id.clone(),
            source: self.source,
            reason: self.reason.clone(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "outcome", content = "cancellation", rename_all = "snake_case")]
pub enum TurnCancelOutcome {
    Requested(TurnCancellationEvidence),
    AlreadyRequested(TurnCancellationEvidence),
    CompletionWonRace,
    UnknownOrRevoked,
}

/// Result of addressing one turn-cancellation gate.
///
/// [`durability_tier`](Self::durability_tier) describes the keyed-promise
/// deployment that accepted the request. [`crate::DurabilityTier::Inline`]
/// receipts are process-local: they do not prove that an owner in another OS
/// process observed the request. Durable cross-process cancellation requires
/// a [`crate::DurabilityTier::Durable`] effect-host deployment.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TurnCancelReceipt {
    pub durability_tier: crate::DurabilityTier,
    pub outcome: TurnCancelOutcome,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum TurnTerminal {
    Committed {
        outcome: TurnOutcome,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cancellation: Option<TurnCancellationEvidence>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        session_revision: Option<u64>,
    },
    Failed {
        error: ErrorEnvelope,
    },
}

/// Backend-specific terminal attachment for a foreground turn.
#[async_trait::async_trait]
pub trait TurnAttach: Send + Sync {
    async fn await_terminal(&self, address: &TurnAddress) -> Result<TurnTerminal, RuntimeError>;
}

/// Cooperative, exact-turn control compiled onto Lash's keyed-promise seam.
///
/// `Requested` means the cancellation request won this driver's keyed-promise
/// gate. On an inline effect host that promise is process-local, so the driver
/// only reaches owners in the same OS process. A durable effect-host deployment
/// is required for another process or replayed owner to observe the request.
/// The returned [`TurnCancelReceipt`] exposes that tier so hosts can gate their
/// UX rather than treating an inline receipt as cross-process proof.
///
/// Lash asks the running or replayed owner to unwind and commit a cancelled
/// result; it cannot guarantee that detached tasks, subprocesses, or
/// non-cooperative providers have stopped. Engine invocation cancellation
/// remains a host-owned break-glass action and is never proof of a Lash
/// `Cancelled` result.
///
/// Session and turn ids are routing identity, not authorization. Hosts must
/// enforce authorization before exposing this driver across a trust boundary.
#[derive(Clone)]
pub struct TurnWorkDriver {
    effect_host: Arc<dyn EffectHost>,
    attach: Option<Arc<dyn TurnAttach>>,
}

impl TurnWorkDriver {
    pub fn new(effect_host: Arc<dyn EffectHost>) -> Self {
        Self {
            effect_host,
            attach: None,
        }
    }

    pub fn with_attach(mut self, attach: Arc<dyn TurnAttach>) -> Self {
        self.attach = Some(attach);
        self
    }

    pub fn effect_host(&self) -> Arc<dyn EffectHost> {
        Arc::clone(&self.effect_host)
    }

    pub async fn request_cancel(
        &self,
        request: TurnCancelRequest,
    ) -> Result<TurnCancelReceipt, RuntimeError> {
        request.validate()?;
        let durability_tier = self.effect_host.durability_tier();
        let key = cancel_gate_key(self.effect_host.as_ref(), &request.address).await?;
        let evidence = request.evidence();
        let resolution = gate_resolution(TurnGateTerminal::CancelRequested(evidence.clone()))?;
        let outcome = match self
            .effect_host
            .resolve_await_event(&key, resolution)
            .await?
        {
            ResolveOutcome::Accepted => Ok(TurnCancelOutcome::Requested(evidence)),
            ResolveOutcome::AlreadyResolved { terminal } => match decode_gate(terminal)? {
                TurnGateTerminal::CancelRequested(existing) => {
                    Ok(TurnCancelOutcome::AlreadyRequested(existing))
                }
                TurnGateTerminal::CompletionSealed => Ok(TurnCancelOutcome::CompletionWonRace),
            },
            ResolveOutcome::UnknownOrRevoked => Ok(TurnCancelOutcome::UnknownOrRevoked),
        }?;
        Ok(TurnCancelReceipt {
            durability_tier,
            outcome,
        })
    }

    pub async fn await_terminal(
        &self,
        address: &TurnAddress,
    ) -> Result<TurnTerminal, RuntimeError> {
        address.validate()?;
        if let Some(attach) = self.attach.as_ref() {
            return attach.await_terminal(address).await;
        }
        let key = terminal_key(self.effect_host.as_ref(), address).await?;
        let resolution = self
            .effect_host
            .await_await_event(&key, CancellationToken::new(), None)
            .await?;
        decode_terminal(address, resolution)
    }

    /// Await a terminal publication for at most `timeout`.
    ///
    /// Timing out only stops this caller's attachment. It never resolves or
    /// poisons the turn's first-writer-wins keyed promises.
    pub async fn await_terminal_with_timeout(
        &self,
        address: &TurnAddress,
        timeout: Duration,
    ) -> Result<TurnTerminal, RuntimeError> {
        tokio::time::timeout(timeout, self.await_terminal(address))
            .await
            .map_err(|_| {
                RuntimeError::new(
                    "turn_terminal_await_timeout",
                    format!(
                        "timed out awaiting terminal for turn `{}` in session `{}` after {} ms",
                        address.turn_id,
                        address.session_id,
                        timeout.as_millis()
                    ),
                )
            })?
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "state", content = "cancellation", rename_all = "snake_case")]
enum TurnGateTerminal {
    CancelRequested(TurnCancellationEvidence),
    CompletionSealed,
}

fn gate_resolution(value: TurnGateTerminal) -> Result<Resolution, RuntimeError> {
    serde_json::to_value(value)
        .map(Resolution::Ok)
        .map_err(|err| RuntimeError::new("turn_cancel_gate_encode", err.to_string()))
}

fn decode_gate(resolution: Resolution) -> Result<TurnGateTerminal, RuntimeError> {
    match resolution {
        Resolution::Ok(value) => serde_json::from_value(value)
            .map_err(|err| RuntimeError::new("turn_cancel_gate_decode", err.to_string())),
        other => Err(RuntimeError::new(
            "turn_cancel_gate_invalid_terminal",
            format!("turn cancellation gate resolved with {other:?}"),
        )),
    }
}

fn terminal_resolution(value: &TurnTerminal) -> Result<Resolution, RuntimeError> {
    serde_json::to_value(value)
        .map(Resolution::Ok)
        .map_err(|err| RuntimeError::new("turn_terminal_encode", err.to_string()))
}

fn decode_terminal(
    address: &TurnAddress,
    resolution: Resolution,
) -> Result<TurnTerminal, RuntimeError> {
    match resolution {
        Resolution::Ok(value) => serde_json::from_value(value).map_err(|err| {
            RuntimeError::new(
                "turn_terminal_decode",
                format!(
                    "invalid terminal result for turn `{}` in session `{}`: {err}",
                    address.turn_id, address.session_id
                ),
            )
        }),
        other => Err(RuntimeError::new(
            "turn_terminal_invalid_resolution",
            format!(
                "terminal result for turn `{}` in session `{}` resolved with {other:?}",
                address.turn_id, address.session_id
            ),
        )),
    }
}

async fn cancel_gate_key(
    resolver: &dyn AwaitEventResolver,
    address: &TurnAddress,
) -> Result<AwaitEventKey, RuntimeError> {
    resolver
        .await_event_key(&address.scope(), AwaitEventWaitIdentity::TurnCancelGate)
        .await
}

async fn terminal_key(
    resolver: &dyn AwaitEventResolver,
    address: &TurnAddress,
) -> Result<AwaitEventKey, RuntimeError> {
    resolver
        .await_event_key(&address.scope(), AwaitEventWaitIdentity::TurnTerminal)
        .await
}

/// Per-execution bridge between the durable gate and the turn's internal
/// cancellation token.
pub(crate) struct ActiveTurnControl {
    address: TurnAddress,
    cancel_key: AwaitEventKey,
    terminal_key: AwaitEventKey,
    evidence: Mutex<Option<TurnCancellationEvidence>>,
    local_cancel_source: TurnCancelSourceHint,
}

impl ActiveTurnControl {
    pub(crate) async fn new(
        resolver: &dyn AwaitEventResolver,
        address: TurnAddress,
    ) -> Result<Self, RuntimeError> {
        address.validate()?;
        Ok(Self {
            cancel_key: cancel_gate_key(resolver, &address).await?,
            terminal_key: terminal_key(resolver, &address).await?,
            address,
            evidence: Mutex::new(None),
            local_cancel_source: TurnCancelSourceHint::default(),
        })
    }

    pub(crate) fn with_local_cancel_source(mut self, source: TurnCancelSourceHint) -> Self {
        self.local_cancel_source = source;
        self
    }

    pub(crate) async fn await_cancel(
        &self,
        resolver: &dyn AwaitEventResolver,
        stop_wait: CancellationToken,
    ) -> Result<Option<TurnCancellationEvidence>, RuntimeError> {
        let resolution = resolver
            .await_await_event(&self.cancel_key, stop_wait, None)
            .await?;
        match decode_gate(resolution)? {
            TurnGateTerminal::CancelRequested(evidence) => {
                self.remember(evidence.clone());
                Ok(Some(evidence))
            }
            TurnGateTerminal::CompletionSealed => Ok(None),
        }
    }

    pub(crate) async fn settle_before_commit(
        &self,
        resolver: &dyn AwaitEventResolver,
        locally_cancelled: bool,
    ) -> Result<Option<TurnCancellationEvidence>, RuntimeError> {
        if let Some(evidence) = self.evidence() {
            return Ok(Some(evidence));
        }
        let proposed = if locally_cancelled {
            TurnGateTerminal::CancelRequested(self.internal_evidence())
        } else {
            TurnGateTerminal::CompletionSealed
        };
        let outcome = resolver
            .resolve_await_event(&self.cancel_key, gate_resolution(proposed.clone())?)
            .await?;
        let terminal = match outcome {
            ResolveOutcome::Accepted => proposed,
            ResolveOutcome::AlreadyResolved { terminal } => decode_gate(terminal)?,
            ResolveOutcome::UnknownOrRevoked => {
                return Err(RuntimeError::new(
                    "turn_control_unknown_or_revoked",
                    format!(
                        "turn `{}` in session `{}` was revoked before final commit",
                        self.address.turn_id, self.address.session_id
                    ),
                ));
            }
        };
        match terminal {
            TurnGateTerminal::CancelRequested(evidence) => {
                self.remember(evidence.clone());
                Ok(Some(evidence))
            }
            TurnGateTerminal::CompletionSealed => Ok(None),
        }
    }

    pub(crate) async fn publish_terminal(
        &self,
        resolver: &dyn AwaitEventResolver,
        terminal: &TurnTerminal,
    ) -> Result<(), RuntimeError> {
        match resolver
            .resolve_await_event(&self.terminal_key, terminal_resolution(terminal)?)
            .await?
        {
            ResolveOutcome::Accepted | ResolveOutcome::AlreadyResolved { .. } => Ok(()),
            ResolveOutcome::UnknownOrRevoked => Err(RuntimeError::new(
                "turn_terminal_unknown_or_revoked",
                format!(
                    "terminal promise for turn `{}` in session `{}` was revoked",
                    self.address.turn_id, self.address.session_id
                ),
            )),
        }
    }

    pub(crate) fn evidence(&self) -> Option<TurnCancellationEvidence> {
        self.evidence
            .lock()
            .expect("turn cancellation evidence lock")
            .clone()
    }

    fn remember(&self, evidence: TurnCancellationEvidence) {
        *self
            .evidence
            .lock()
            .expect("turn cancellation evidence lock") = Some(evidence);
    }

    fn internal_evidence(&self) -> TurnCancellationEvidence {
        let source = self.local_cancel_source.get();
        TurnCancellationEvidence {
            request_id: format!("internal:{}", self.address.turn_id),
            source: source.unwrap_or(TurnCancelSource::Host),
            reason: Some(match source {
                Some(TurnCancelSource::UserInterrupt) => {
                    "process-local user interrupt token fired".to_string()
                }
                Some(source) => format!("process-local {source:?} cancellation token fired"),
                None => "process-local cancellation token fired; origin is unknown".to_string(),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{InlineEffectHost, TurnFinish, TurnStop};

    fn address(label: &str) -> TurnAddress {
        TurnAddress::new(
            format!("turn-control-{label}-{}", uuid::Uuid::new_v4()),
            "turn-a",
        )
    }

    fn request(address: TurnAddress, request_id: &str) -> TurnCancelRequest {
        TurnCancelRequest::new(address, request_id, TurnCancelSource::UserInterrupt)
            .with_reason("stop button")
    }

    #[tokio::test]
    async fn cancel_before_start_duplicate_and_terminal_attach() {
        let host = Arc::new(InlineEffectHost::default());
        let driver = TurnWorkDriver::new(host.clone());
        let address = address("before-start");

        let first = driver
            .request_cancel(request(address.clone(), "request-1"))
            .await
            .expect("request cancellation");
        assert_eq!(first.durability_tier, crate::DurabilityTier::Inline);
        let evidence = match first.outcome {
            TurnCancelOutcome::Requested(evidence) => evidence,
            other => panic!("expected requested, got {other:?}"),
        };
        assert_eq!(evidence.request_id, "request-1");

        let duplicate = driver
            .request_cancel(request(address.clone(), "request-2"))
            .await
            .expect("duplicate cancellation");
        assert!(matches!(
            duplicate.outcome,
            TurnCancelOutcome::AlreadyRequested(TurnCancellationEvidence { ref request_id, .. })
                if request_id == "request-1"
        ));

        let active = ActiveTurnControl::new(host.as_ref(), address.clone())
            .await
            .expect("active control");
        let observed = active
            .settle_before_commit(host.as_ref(), false)
            .await
            .expect("settle")
            .expect("cancellation won");
        assert_eq!(observed, evidence);
        let terminal = TurnTerminal::Committed {
            outcome: TurnOutcome::Stopped(TurnStop::Cancelled),
            cancellation: Some(observed),
            session_revision: Some(7),
        };
        active
            .publish_terminal(host.as_ref(), &terminal)
            .await
            .expect("publish terminal");
        let attached = driver
            .await_terminal(&address)
            .await
            .expect("attach terminal");
        assert!(matches!(
            attached,
            TurnTerminal::Committed {
                outcome: TurnOutcome::Stopped(TurnStop::Cancelled),
                cancellation: Some(_),
                session_revision: Some(7),
            }
        ));
    }

    #[tokio::test]
    async fn concurrent_completion_seal_vs_cancel_is_first_writer_wins() {
        let host = Arc::new(InlineEffectHost::default());
        let driver = TurnWorkDriver::new(host.clone());
        let address = address("race");
        let active = ActiveTurnControl::new(host.as_ref(), address.clone())
            .await
            .expect("active control");

        let (seal, cancel) = tokio::join!(
            active.settle_before_commit(host.as_ref(), false),
            driver.request_cancel(request(address, "race-request")),
        );
        match (seal.expect("seal"), cancel.expect("cancel").outcome) {
            (None, TurnCancelOutcome::CompletionWonRace) => {}
            (Some(evidence), TurnCancelOutcome::Requested(requested)) => {
                assert_eq!(evidence, requested);
            }
            other => panic!("inconsistent gate race result: {other:?}"),
        }
    }

    #[tokio::test]
    async fn recovered_owner_observes_pending_cancel_after_control_recreation() {
        let host = Arc::new(InlineEffectHost::default());
        let driver = TurnWorkDriver::new(host.clone());
        let address = address("replay");
        let requested = driver
            .request_cancel(request(address.clone(), "request-before-replay"))
            .await
            .expect("request cancellation");
        let expected = match requested.outcome {
            TurnCancelOutcome::Requested(evidence) => evidence,
            other => panic!("expected requested, got {other:?}"),
        };

        let recovered = ActiveTurnControl::new(host.as_ref(), address)
            .await
            .expect("recreate active control under the recovered owner");
        let observed = recovered
            .settle_before_commit(host.as_ref(), false)
            .await
            .expect("settle recovered turn")
            .expect("pending cancellation survives owner loss");
        assert_eq!(observed, expected);
    }

    #[tokio::test]
    async fn turn_control_is_exact_scope_and_excluded_from_wait_cancel_sweep() {
        let host = Arc::new(InlineEffectHost::default());
        let driver = TurnWorkDriver::new(host.clone());
        let address_a = address("scope");
        let address_b = TurnAddress::new(&address_a.session_id, "turn-b");
        let address_future = TurnAddress::new(&address_a.session_id, "turn-future");

        driver
            .request_cancel(request(address_a.clone(), "request-a"))
            .await
            .expect("cancel a");

        let tool_key = host
            .await_event_key(
                &ExecutionScope::turn(&address_a.session_id, "tool-turn"),
                AwaitEventWaitIdentity::tool_completion("tool-call"),
            )
            .await
            .expect("tool key");
        let tool_host = host.clone();
        let tool_wait = tokio::spawn(async move {
            tool_host
                .await_await_event(&tool_key, CancellationToken::new(), None)
                .await
        });
        tokio::task::yield_now().await;
        host.cancel_await_events_for_session(&address_a.session_id)
            .await
            .expect("cancel durable waits");
        assert!(matches!(
            tool_wait
                .await
                .expect("tool wait task")
                .expect("tool resolution"),
            Resolution::Cancelled
        ));

        assert!(matches!(
            driver
                .request_cancel(request(address_a.clone(), "request-a-duplicate"))
                .await
                .expect("duplicate a")
                .outcome,
            TurnCancelOutcome::AlreadyRequested(_)
        ));
        assert!(matches!(
            driver
                .request_cancel(request(address_b, "request-b"))
                .await
                .expect("cancel b")
                .outcome,
            TurnCancelOutcome::Requested(_)
        ));
        assert!(matches!(
            driver
                .request_cancel(request(address_future, "request-future"))
                .await
                .expect("cancel future")
                .outcome,
            TurnCancelOutcome::Requested(_)
        ));
    }

    #[tokio::test]
    async fn session_deletion_revokes_control_promises() {
        let host = Arc::new(InlineEffectHost::default());
        let driver = TurnWorkDriver::new(host.clone());
        let address = address("revoke");
        host.revoke_await_events_for_session(&address.session_id)
            .await
            .expect("revoke session");
        assert!(matches!(
            driver
                .request_cancel(request(address, "request-after-delete"))
                .await
                .expect("revoked outcome")
                .outcome,
            TurnCancelOutcome::UnknownOrRevoked
        ));
    }

    #[tokio::test]
    async fn terminal_attachment_timeout_does_not_poison_later_publication() {
        let host = Arc::new(InlineEffectHost::default());
        let driver = TurnWorkDriver::new(host.clone());
        let address = address("terminal-timeout");
        let error = driver
            .await_terminal_with_timeout(&address, Duration::from_millis(1))
            .await
            .expect_err("unpublished terminal must time out");
        assert_eq!(error.code.as_str(), "turn_terminal_await_timeout");

        let active = ActiveTurnControl::new(host.as_ref(), address.clone())
            .await
            .expect("active control after timed-out attach");
        active
            .settle_before_commit(host.as_ref(), false)
            .await
            .expect("seal after timed-out attach");
        active
            .publish_terminal(
                host.as_ref(),
                &TurnTerminal::Committed {
                    outcome: TurnOutcome::Finished(TurnFinish::AssistantMessage {
                        text: "done".to_string(),
                    }),
                    cancellation: None,
                    session_revision: None,
                },
            )
            .await
            .expect("publish after timed-out attach");
        assert!(matches!(
            driver.await_terminal(&address).await.expect("late attach"),
            TurnTerminal::Committed {
                outcome: TurnOutcome::Finished(_),
                cancellation: None,
                ..
            }
        ));
    }

    #[test]
    fn local_cancel_source_hint_preserves_first_origin() {
        let hint = TurnCancelSourceHint::default();
        hint.set(TurnCancelSource::Shutdown);
        hint.set(TurnCancelSource::UserInterrupt);

        assert_eq!(hint.get(), Some(TurnCancelSource::Shutdown));
    }

    #[test]
    fn terminal_success_has_no_cancellation_evidence() {
        let terminal = TurnTerminal::Committed {
            outcome: TurnOutcome::Finished(TurnFinish::AssistantMessage {
                text: "done".to_string(),
            }),
            cancellation: None,
            session_revision: None,
        };
        let encoded = terminal_resolution(&terminal).expect("encode terminal");
        assert!(matches!(encoded, Resolution::Ok(_)));
    }
}
