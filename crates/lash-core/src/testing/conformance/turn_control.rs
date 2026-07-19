//! Shared foreground-turn control conformance.

use std::sync::Arc;

use tokio_util::sync::CancellationToken;

use crate::runtime::turn_control::ActiveTurnControl;
use crate::{
    AwaitEventWaitIdentity, EffectHost, ExecutionScope, Resolution, TurnAddress, TurnCancelOutcome,
    TurnCancelRequest, TurnCancelSource, TurnCancellationEvidence, TurnStop, TurnTerminal,
    TurnWorkDriver,
};

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

/// Run the exact-address, replay, terminal, sweep, and revocation contract for
/// a keyed-promise adapter.
pub async fn turn_work_driver(host: Arc<dyn EffectHost>) {
    cancel_before_start_duplicate_replay_and_terminal_attach(Arc::clone(&host)).await;
    completion_seal_vs_cancel_is_first_writer_wins(Arc::clone(&host)).await;
    exact_scope_and_session_sweep_isolation(Arc::clone(&host)).await;
    session_deletion_revokes_control_promises(host).await;
}

async fn cancel_before_start_duplicate_replay_and_terminal_attach(host: Arc<dyn EffectHost>) {
    let driver = TurnWorkDriver::new(Arc::clone(&host));
    let address = address("before-start");
    let first = driver
        .request_cancel(request(address.clone(), "request-1"))
        .await
        .expect("request cancellation");
    let evidence = match first {
        TurnCancelOutcome::Requested(evidence) => evidence,
        other => panic!("expected requested, got {other:?}"),
    };
    let duplicate = driver
        .request_cancel(request(address.clone(), "request-2"))
        .await
        .expect("duplicate cancellation");
    assert!(matches!(
        duplicate,
        TurnCancelOutcome::AlreadyRequested(TurnCancellationEvidence { ref request_id, .. })
            if request_id == "request-1"
    ));

    // Recreating this bridge models a new lease generation after owner loss;
    // store-side generation fencing remains responsible for rejecting a stale
    // owner's final commit.
    let recovered = ActiveTurnControl::new(host.as_ref(), address.clone())
        .await
        .expect("recreate active control");
    let observed = recovered
        .settle_before_commit(host.as_ref(), false)
        .await
        .expect("settle recovered turn")
        .expect("pending cancellation survives replay");
    assert_eq!(observed, evidence);

    let terminal = TurnTerminal::Committed {
        outcome: crate::TurnOutcome::Stopped(TurnStop::Cancelled),
        cancellation: Some(observed),
        session_revision: Some(7),
    };
    recovered
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
            outcome: crate::TurnOutcome::Stopped(TurnStop::Cancelled),
            cancellation: Some(_),
            session_revision: Some(7),
        }
    ));
}

async fn completion_seal_vs_cancel_is_first_writer_wins(host: Arc<dyn EffectHost>) {
    let driver = TurnWorkDriver::new(Arc::clone(&host));
    let address = address("race");
    let active = ActiveTurnControl::new(host.as_ref(), address.clone())
        .await
        .expect("active control");
    let (seal, cancel) = tokio::join!(
        active.settle_before_commit(host.as_ref(), false),
        driver.request_cancel(request(address, "race-request")),
    );
    match (seal.expect("seal"), cancel.expect("cancel")) {
        (None, TurnCancelOutcome::CompletionWonRace) => {}
        (Some(evidence), TurnCancelOutcome::Requested(requested)) => {
            assert_eq!(evidence, requested);
        }
        other => panic!("inconsistent gate race result: {other:?}"),
    }
}

async fn exact_scope_and_session_sweep_isolation(host: Arc<dyn EffectHost>) {
    let driver = TurnWorkDriver::new(Arc::clone(&host));
    let address_a = address("scope");
    let address_b = TurnAddress::new(&address_a.session_id, "turn-b");
    let address_future = TurnAddress::new(&address_a.session_id, "turn-future");

    let active = Arc::new(
        ActiveTurnControl::new(host.as_ref(), address_a.clone())
            .await
            .expect("active control"),
    );
    let waiter_host = Arc::clone(&host);
    let waiter_active = Arc::clone(&active);
    let cancel_wait = tokio::spawn(async move {
        waiter_active
            .await_cancel(waiter_host.as_ref(), CancellationToken::new())
            .await
    });
    tokio::task::yield_now().await;

    let tool_key = host
        .await_event_key(
            &ExecutionScope::turn(&address_a.session_id, "tool-turn"),
            AwaitEventWaitIdentity::tool_completion("tool-call"),
        )
        .await
        .expect("tool key");
    let tool_host = Arc::clone(&host);
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
    assert!(
        !cancel_wait.is_finished(),
        "wait sweep cancelled the turn gate"
    );

    assert!(matches!(
        driver
            .request_cancel(request(address_a.clone(), "request-a"))
            .await
            .expect("cancel a"),
        TurnCancelOutcome::Requested(_)
    ));
    assert!(
        cancel_wait
            .await
            .expect("turn cancellation waiter")
            .expect("turn cancellation observation")
            .is_some()
    );
    assert!(matches!(
        driver
            .request_cancel(request(address_b, "request-b"))
            .await
            .expect("cancel b"),
        TurnCancelOutcome::Requested(_)
    ));
    assert!(matches!(
        driver
            .request_cancel(request(address_future, "request-future"))
            .await
            .expect("cancel future"),
        TurnCancelOutcome::Requested(_)
    ));
}

async fn session_deletion_revokes_control_promises(host: Arc<dyn EffectHost>) {
    let driver = TurnWorkDriver::new(Arc::clone(&host));
    let address = address("revoke");
    host.revoke_await_events_for_session(&address.session_id)
        .await
        .expect("revoke session");
    assert!(matches!(
        driver
            .request_cancel(request(address, "request-after-delete"))
            .await
            .expect("revoked outcome"),
        TurnCancelOutcome::UnknownOrRevoked
    ));
}
