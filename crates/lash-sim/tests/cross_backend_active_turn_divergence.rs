//! DISCRIMINATOR (scratch) for the cross-backend SQLite active-turn divergence.
//!
//! Builds TWO real `lash::StandardCore`s identical except the session store
//! factory (in-memory vs lash-sqlite-store) and drives the SAME operation
//! sequence on both over the production transport seam with an UN-GATED
//! `ScriptedLlmHttpTransport` (no `ScriptedTransportSchedule` / no per-event
//! Notify gating — that gating is the harness artifact under test). Compares
//! committed assistant message + cumulative provider exchange count per turn.

use std::path::Path;
use std::sync::Arc;

use lash::persistence::{
    AttachmentStore, FileAttachmentStore, InMemoryAttachmentStore,
    InMemoryProcessExecutionEnvStore, InMemorySessionStoreFactory, PendingTurnInputCancelOutcome,
    ProcessExecutionEnvStore,
};
use lash::{StandardCore, TurnInput};
use lash_core::{SessionStoreFactory, TurnInputCheckpointBoundary, TurnInputIngress};
use lash_sim::ProviderWireScript;
use lash_sim::ScriptedLlmHttpTransport;
use lash_sim::runtime_providers::{
    OPENAI_COMPATIBLE, runtime_provider_components, runtime_scripts_for_texts,
};

const PROVIDER_KIND: &str = OPENAI_COMPATIBLE;

fn scripts(n: usize) -> Vec<ProviderWireScript> {
    let texts: Vec<String> = (1..=n).map(|i| format!("answer {i}")).collect();
    runtime_scripts_for_texts(PROVIDER_KIND, &texts).expect("scripts")
}

async fn build_core(
    store_factory: Arc<dyn SessionStoreFactory>,
    process_env_store: Arc<dyn ProcessExecutionEnvStore>,
    attachment_store: Arc<dyn AttachmentStore>,
    scripts: Vec<ProviderWireScript>,
) -> (StandardCore, Arc<ScriptedLlmHttpTransport>) {
    // UN-GATED transport: deliver each scripted response complete/synchronously.
    let transport = Arc::new(ScriptedLlmHttpTransport::from_scripts(scripts));
    let (provider_handle, model, _kind) =
        runtime_provider_components(PROVIDER_KIND, &transport).expect("provider components");
    let core = StandardCore::builder()
        .effect_host(Arc::new(lash::durability::InlineEffectHost::default()))
        .attachment_store(attachment_store)
        .process_env_store(process_env_store)
        .store_factory(store_factory)
        .provider(provider_handle)
        .model(model)
        .build()
        .expect("core");
    (core, transport)
}

async fn build_in_memory(n_scripts: usize) -> (StandardCore, Arc<ScriptedLlmHttpTransport>) {
    build_core(
        Arc::new(InMemorySessionStoreFactory::new()),
        Arc::new(InMemoryProcessExecutionEnvStore::new()),
        Arc::new(InMemoryAttachmentStore::new()),
        scripts(n_scripts),
    )
    .await
}

async fn build_sqlite(
    dir: &Path,
    n_scripts: usize,
) -> (StandardCore, Arc<ScriptedLlmHttpTransport>) {
    std::fs::create_dir_all(dir).expect("create sqlite dir");
    let process_env_store: Arc<dyn ProcessExecutionEnvStore> = Arc::new(
        lash_sqlite_store::Store::open(&dir.join("process-env.sqlite"))
            .await
            .expect("process env store"),
    );
    build_core(
        Arc::new(lash_sqlite_store::SqliteSessionStoreFactory::new(
            dir.to_path_buf(),
        )),
        process_env_store,
        Arc::new(FileAttachmentStore::new(dir.join("attachments"))),
        scripts(n_scripts),
    )
    .await
}

#[derive(Debug, PartialEq, Eq)]
struct TurnObs {
    label: String,
    assistant_message: Option<String>,
    is_success: bool,
    cumulative_exchanges: usize,
}

/// Drive the trace-faithful sequence: active-turn input targeting the
/// currently-/about-to-run turn 1 is enqueued, CANCELLED, then turn 1 runs
/// (its AfterWork checkpoint sees the already-cancelled row), then subsequent
/// turns. `t1` is both the active_turn_id and turn 1's turn_id (as in the
/// trace, where active_turn_id == the running turn's boundary id).
async fn drive_cancel_before_turn(
    core: &StandardCore,
    transport: &Arc<ScriptedLlmHttpTransport>,
    session_id: &str,
) -> (Vec<TurnObs>, String) {
    let session = core
        .session(session_id.to_string())
        .open_fresh()
        .await
        .expect("open fresh");
    let t1 = format!("{session_id}:provider:001");
    let t2 = format!("{session_id}:provider:002");
    let t3 = format!("{session_id}:provider:003");

    let pending = session
        .enqueue(TurnInput::text("queued follow-up"))
        .id("follow-up-001")
        .ingress(TurnInputIngress::active_turn(
            t1.clone(),
            TurnInputCheckpointBoundary::AfterWork,
        ))
        .send()
        .await
        .expect("enqueue active-turn input");
    let cancel = session
        .cancel_pending_turn_input(&pending.input_id)
        .await
        .expect("cancel");
    let cancel_outcome = cancel_label(&cancel);

    let mut obs = Vec::new();
    for (label, turn_id) in [("turn-1", &t1), ("turn-2", &t2), ("turn-3", &t3)] {
        obs.push(run_turn(&session, transport, label, turn_id).await);
    }
    (obs, cancel_outcome)
}

/// Task-literal ordering: run turn 1, THEN enqueue + cancel an active-turn
/// input targeting turn 1 (already finished), then subsequent turns.
async fn drive_cancel_after_turn(
    core: &StandardCore,
    transport: &Arc<ScriptedLlmHttpTransport>,
    session_id: &str,
) -> (Vec<TurnObs>, String) {
    let session = core
        .session(session_id.to_string())
        .open_fresh()
        .await
        .expect("open fresh");
    let t1 = format!("{session_id}:provider:001");
    let t2 = format!("{session_id}:provider:002");
    let t3 = format!("{session_id}:provider:003");

    let mut obs = Vec::new();
    obs.push(run_turn(&session, transport, "turn-1", &t1).await);

    let pending = session
        .enqueue(TurnInput::text("queued follow-up"))
        .id("follow-up-001")
        .ingress(TurnInputIngress::active_turn(
            t1.clone(),
            TurnInputCheckpointBoundary::AfterWork,
        ))
        .send()
        .await
        .expect("enqueue active-turn input");
    let cancel = session
        .cancel_pending_turn_input(&pending.input_id)
        .await
        .expect("cancel");
    let cancel_outcome = cancel_label(&cancel);

    obs.push(run_turn(&session, transport, "turn-2", &t2).await);
    obs.push(run_turn(&session, transport, "turn-3", &t3).await);
    (obs, cancel_outcome)
}

/// Control: enqueue an active-turn input targeting turn 1 and DO NOT cancel,
/// then run turn 1 (whose AfterWork checkpoint should claim and inject it,
/// driving an extra exchange), then turn 2.
async fn drive_no_cancel_control(
    core: &StandardCore,
    transport: &Arc<ScriptedLlmHttpTransport>,
    session_id: &str,
) -> Vec<TurnObs> {
    let session = core
        .session(session_id.to_string())
        .open_fresh()
        .await
        .expect("open fresh");
    let t1 = format!("{session_id}:provider:001");
    let t2 = format!("{session_id}:provider:002");

    session
        .enqueue(TurnInput::text("queued follow-up"))
        .id("follow-up-001")
        .ingress(TurnInputIngress::active_turn(
            t1.clone(),
            TurnInputCheckpointBoundary::AfterWork,
        ))
        .send()
        .await
        .expect("enqueue active-turn input");

    let mut obs = Vec::new();
    obs.push(run_turn(&session, transport, "turn-1", &t1).await);
    obs.push(run_turn(&session, transport, "turn-2", &t2).await);
    obs
}

/// Claim-then-cancel: enqueue active-turn input targeting turn 1, run turn 1
/// (its AfterWork checkpoint CLAIMS and completes the input -> extra exchange),
/// THEN cancel (must observe the post-claim terminal state), then run turn 2.
async fn drive_claim_then_cancel(
    core: &StandardCore,
    transport: &Arc<ScriptedLlmHttpTransport>,
    session_id: &str,
) -> (Vec<TurnObs>, String) {
    let session = core
        .session(session_id.to_string())
        .open_fresh()
        .await
        .expect("open fresh");
    let t1 = format!("{session_id}:provider:001");
    let t2 = format!("{session_id}:provider:002");

    let pending = session
        .enqueue(TurnInput::text("queued follow-up"))
        .id("follow-up-001")
        .ingress(TurnInputIngress::active_turn(
            t1.clone(),
            TurnInputCheckpointBoundary::AfterWork,
        ))
        .send()
        .await
        .expect("enqueue active-turn input");

    let mut obs = Vec::new();
    obs.push(run_turn(&session, transport, "turn-1", &t1).await);
    let cancel = session
        .cancel_pending_turn_input(&pending.input_id)
        .await
        .expect("cancel");
    let cancel_outcome = cancel_label(&cancel);
    obs.push(run_turn(&session, transport, "turn-2", &t2).await);
    (obs, cancel_outcome)
}

fn cancel_label(outcome: &PendingTurnInputCancelOutcome) -> String {
    match outcome {
        PendingTurnInputCancelOutcome::Cancelled(_) => "cancelled",
        PendingTurnInputCancelOutcome::AlreadyClaimed { .. } => "already_claimed",
        PendingTurnInputCancelOutcome::AlreadyCompleted(_) => "already_completed",
        PendingTurnInputCancelOutcome::AlreadyCancelled(_) => "already_cancelled",
        PendingTurnInputCancelOutcome::NotFound => "not_found",
    }
    .to_string()
}

async fn run_turn(
    session: &lash::LashSession,
    transport: &Arc<ScriptedLlmHttpTransport>,
    label: &str,
    turn_id: &str,
) -> TurnObs {
    let result = session
        .turn(TurnInput::text(format!("user prompt for {label}")))
        .turn_id(turn_id.to_string())
        .run()
        .await;
    let cumulative_exchanges = transport.exchanges().map(|e| e.len()).unwrap_or(usize::MAX);
    match result {
        Ok(output) => TurnObs {
            label: label.to_string(),
            assistant_message: output.assistant_message().map(str::to_string),
            is_success: output.is_success(),
            cumulative_exchanges,
        },
        Err(err) => TurnObs {
            label: format!("{label} (ERR: {err})"),
            assistant_message: None,
            is_success: false,
            cumulative_exchanges,
        },
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cross_backend_active_turn_cancel_then_turn_agrees() {
    let tmp = tempfile::tempdir().expect("tempdir");

    // Variant B: cancel BEFORE turn 1 runs (closest to the recorded trace
    // timing: cancel precedes turn 1's AfterWork checkpoint).
    let (mem_core, mem_tx) = build_in_memory(8).await;
    let (b_mem, b_mem_cancel) =
        drive_cancel_before_turn(&mem_core, &mem_tx, "mem-B").await;
    let (sq_core, sq_tx) = build_sqlite(&tmp.path().join("sqlite-B"), 8).await;
    let (b_sq, b_sq_cancel) =
        drive_cancel_before_turn(&sq_core, &sq_tx, "sql-B").await;

    // Variant A: cancel AFTER turn 1 (task-literal ordering).
    let (mem_core_a, mem_tx_a) = build_in_memory(8).await;
    let (a_mem, a_mem_cancel) =
        drive_cancel_after_turn(&mem_core_a, &mem_tx_a, "mem-A").await;
    let (sq_core_a, sq_tx_a) = build_sqlite(&tmp.path().join("sqlite-A"), 8).await;
    let (a_sq, a_sq_cancel) =
        drive_cancel_after_turn(&sq_core_a, &sq_tx_a, "sql-A").await;

    // Variant C: no cancel control (active-turn input should be claimed).
    let (mem_core_c, mem_tx_c) = build_in_memory(8).await;
    let c_mem = drive_no_cancel_control(&mem_core_c, &mem_tx_c, "mem-C").await;
    let (sq_core_c, sq_tx_c) = build_sqlite(&tmp.path().join("sqlite-C"), 8).await;
    let c_sq = drive_no_cancel_control(&sq_core_c, &sq_tx_c, "sql-C").await;

    // Normalize session-id-derived labels so the per-backend Vecs are
    // comparable (labels are "turn-1/2/3" already, independent of session id).
    println!("=== Variant B (cancel BEFORE turn 1) ===");
    println!("  in-memory cancel: {b_mem_cancel}");
    println!("  sqlite    cancel: {b_sq_cancel}");
    println!("  in-memory turns:  {b_mem:#?}");
    println!("  sqlite    turns:  {b_sq:#?}");

    println!("=== Variant A (cancel AFTER turn 1) ===");
    println!("  in-memory cancel: {a_mem_cancel}");
    println!("  sqlite    cancel: {a_sq_cancel}");
    println!("  in-memory turns:  {a_mem:#?}");
    println!("  sqlite    turns:  {a_sq:#?}");

    println!("=== Variant C (no cancel control) ===");
    println!("  in-memory turns:  {c_mem:#?}");
    println!("  sqlite    turns:  {c_sq:#?}");

    // Variant D: claim-then-cancel (cancel observes post-claim terminal state).
    let (mem_core_d, mem_tx_d) = build_in_memory(8).await;
    let (d_mem, d_mem_cancel) = drive_claim_then_cancel(&mem_core_d, &mem_tx_d, "mem-D").await;
    let (sq_core_d, sq_tx_d) = build_sqlite(&tmp.path().join("sqlite-D"), 8).await;
    let (d_sq, d_sq_cancel) = drive_claim_then_cancel(&sq_core_d, &sq_tx_d, "sql-D").await;
    println!("=== Variant D (claim then cancel) ===");
    println!("  in-memory cancel: {d_mem_cancel}");
    println!("  sqlite    cancel: {d_sq_cancel}");
    println!("  in-memory turns:  {d_mem:#?}");
    println!("  sqlite    turns:  {d_sq:#?}");

    assert_eq!(b_mem_cancel, b_sq_cancel, "variant B cancel outcome diverged");
    assert_eq!(b_mem, b_sq, "variant B per-turn output diverged");
    assert_eq!(a_mem_cancel, a_sq_cancel, "variant A cancel outcome diverged");
    assert_eq!(a_mem, a_sq, "variant A per-turn output diverged");
    assert_eq!(c_mem, c_sq, "variant C (control) per-turn output diverged");
    assert_eq!(d_mem_cancel, d_sq_cancel, "variant D cancel outcome diverged");
    assert_eq!(d_mem, d_sq, "variant D per-turn output diverged");
}
