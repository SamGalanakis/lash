#[test]
#[ignore = "requires a running Restate server; use `just agent-workbench-restate-e2e`"]
fn live_restate_ingress_owner_restart_resumes_and_remains_cancellable() {
    if std::env::var_os("AGENT_WORKBENCH_RECOVERY_E2E_CHILD").is_some() {
        run_async_test_on_stack_budget_multi_thread("workbench-recovery-child", 4, || {
            live_restate_recovery_child()
        });
        return;
    }
    run_async_test_on_stack_budget_multi_thread("workbench-recovery-parent", 4, || {
        live_restate_ingress_owner_restart_resumes_and_remains_cancellable_inner()
    });
}

#[test]
#[ignore = "requires a running Restate server; use `just agent-workbench-restate-e2e`"]
fn live_restate_turn_input_ingress_delivers_once_and_queues_after_settle() {
    run_async_test_on_stack_budget_multi_thread("workbench-turn-ingress-e2e", 4, || {
        live_restate_turn_input_ingress_delivers_once_and_queues_after_settle_inner()
    });
}

#[test]
#[ignore = "requires a running Restate server; use `just agent-workbench-restate-e2e`"]
fn live_restate_processes_outlive_session_delete_and_cancel_globally() {
    run_async_test_on_stack_budget_multi_thread("workbench-process-lifecycle-e2e", 4, || {
        live_restate_processes_outlive_session_delete_and_cancel_globally_inner()
    });
}

async fn live_restate_processes_outlive_session_delete_and_cancel_globally_inner() {
    let ingress_url = std::env::var("RESTATE_INGRESS_URL")
        .expect("RESTATE_INGRESS_URL must be set by the workbench Restate E2E recipe");
    let admin_url =
        std::env::var("RESTATE_ADMIN_URL").unwrap_or_else(|_| "http://127.0.0.1:19071".to_string());
    let endpoint_bind: SocketAddr = std::env::var("AGENT_WORKBENCH_E2E_ENDPOINT_BIND")
        .unwrap_or_else(|_| "127.0.0.1:19081".to_string())
        .parse()
        .expect("valid workbench E2E endpoint bind");
    let endpoint_url = std::env::var("AGENT_WORKBENCH_E2E_ENDPOINT_URL")
        .unwrap_or_else(|_| format!("http://{endpoint_bind}"));
    let data_dir = std::env::temp_dir().join(format!(
        "agent-workbench-process-lifecycle-e2e-{}",
        uuid::Uuid::new_v4()
    ));
    std::fs::create_dir_all(&data_dir).expect("create process lifecycle E2E data dir");

    let provider = lash::testing::TestProvider::builder()
        .kind("workbench-process-lifecycle-e2e")
        .complete(|_| async {
            Ok(text_response(
                r#"<lashlang>
process survivor() {
  sleep for "8s"
  finish "survived session deletion"
}
process cancellable() {
  sleep for "60s"
  finish "cancellation failed"
}
survivor_handle = start survivor()
cancellable_handle = start cancellable()
finish "started lifecycle gates"
</lashlang>"#,
            ))
        })
        .build()
        .into_handle();
    let harness = live_workbench_restate_state_with_provider(
        &data_dir,
        ingress_url,
        provider,
        WorkbenchSessionIds::fresh(),
        ActiveTurns::default(),
    )
    .await;
    restate::spawn_restate_endpoint(
        endpoint_bind,
        harness.state.clone(),
        harness.process_deployment,
        harness.process_worker,
    );
    wait_for_endpoint_socket(endpoint_bind).await;
    register_restate_deployment(&admin_url, &endpoint_url).await;

    let deleted_session_id = harness.state.current_session_id();
    let turn_invocation_id =
        run_workbench_turn_via_restate(&harness.state, "start process lifecycle gates").await;
    wait_for_restate_invocation_success(
        &harness.state,
        &turn_invocation_id,
        Duration::from_secs(30),
    )
    .await;
    let (survivor_id, cancellable_id) = wait_for_named_running_processes(
        &harness.state,
        &["survivor", "cancellable"],
        Duration::from_secs(20),
    )
    .await;

    let delete_invocation_id = restate::submit_session_delete(
        &harness.state,
        restate::WorkbenchSessionDeleteWorkflowRequest {
            operation_id: format!("workbench-delete-{}", uuid::Uuid::new_v4()),
            session_id: deleted_session_id.clone(),
        },
    )
    .await
    .expect("submit session deletion while processes run");
    wait_for_restate_invocation_success(
        &harness.state,
        &delete_invocation_id,
        Duration::from_secs(20),
    )
    .await;

    let Json(work_after_delete) = list_work(State(harness.state.clone()))
        .await
        .expect("list runtime work after session deletion");
    for process_id in [&survivor_id, &cancellable_id] {
        assert!(
            work_after_delete
                .iter()
                .any(|item| item.process.process_id == *process_id && !item.process.terminal),
            "work rail lost live process {process_id} after deleting {deleted_session_id}: {work_after_delete:#?}"
        );
    }

    let Json(cancel_receipt) = cancel_work(
        AxumPath(cancellable_id.clone()),
        State(harness.state.clone()),
    )
    .await
    .expect("cancel orphaned process through work API");
    assert!(cancel_receipt.accepted);
    wait_for_process_event(
        &harness.state,
        &cancellable_id,
        "process.cancel_requested",
        Duration::from_secs(20),
    )
    .await;
    let cancelled = tokio::time::timeout(
        Duration::from_secs(20),
        harness
            .state
            .process_work_driver
            .await_terminal(&cancellable_id),
    )
    .await
    .expect("cancelled process terminal timeout")
    .expect("await cancelled process");
    assert!(
        matches!(
            cancelled,
            lash::process::ProcessAwaitOutput::Cancelled { .. }
        ),
        "process cancellation settled with the wrong outcome: {cancelled:#?}"
    );

    let survived = tokio::time::timeout(
        Duration::from_secs(20),
        harness
            .state
            .process_work_driver
            .await_terminal(&survivor_id),
    )
    .await
    .expect("surviving process terminal timeout")
    .expect("await surviving process");
    assert!(
        matches!(
            &survived,
            lash::process::ProcessAwaitOutput::Success { value, .. }
                if value == &json!("survived session deletion")
        ),
        "session-independent process did not complete successfully: {survived:#?}"
    );
    let Json(terminal_work) = list_work(State(harness.state.clone()))
        .await
        .expect("list terminal runtime work");
    assert!(terminal_work.iter().any(|item| {
        item.process.process_id == survivor_id
            && item.process.terminal
            && item.process.lifecycle == lash::process::ProcessLifecycleStatus::Completed
    }));
    assert!(terminal_work.iter().any(|item| {
        item.process.process_id == cancellable_id
            && item.process.terminal
            && item.process.lifecycle == lash::process::ProcessLifecycleStatus::Cancelled
            && item
                .events
                .iter()
                .any(|event| event.event_type == "process.cancel_requested")
    }));
    let _ = std::fs::remove_dir_all(data_dir);
}

async fn wait_for_named_running_processes(
    state: &AppState,
    labels: &[&str],
    timeout: Duration,
) -> (String, String) {
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        let processes = state
            .process_observer
            .list(&lash::process::ProcessListFilter {
                status: lash::process::ProcessStatusFilter::Running,
                ..lash::process::ProcessListFilter::default()
            })
            .await
            .expect("list running processes");
        let named = labels
            .iter()
            .filter_map(|label| {
                processes
                    .iter()
                    .find(|process| process.identity.label.as_deref() == Some(*label))
                    .map(|process| process.process_id.clone())
            })
            .collect::<Vec<_>>();
        if named.len() == labels.len() {
            return (named[0].clone(), named[1].clone());
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "timed out waiting for named running processes {labels:?}; observed={processes:#?}"
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

async fn wait_for_process_event(
    state: &AppState,
    process_id: &str,
    event_type: &str,
    timeout: Duration,
) {
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        let events = state
            .process_observer
            .events_after(process_id, 0)
            .await
            .expect("read process events");
        if events.iter().any(|event| event.event_type == event_type) {
            return;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "timed out waiting for {event_type} on {process_id}; events={events:#?}"
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

async fn live_restate_turn_input_ingress_delivers_once_and_queues_after_settle_inner() {
    let ingress_url = std::env::var("RESTATE_INGRESS_URL")
        .expect("RESTATE_INGRESS_URL must be set by the workbench Restate E2E recipe");
    let admin_url =
        std::env::var("RESTATE_ADMIN_URL").unwrap_or_else(|_| "http://127.0.0.1:19071".to_string());
    let endpoint_bind: SocketAddr = std::env::var("AGENT_WORKBENCH_E2E_ENDPOINT_BIND")
        .unwrap_or_else(|_| "127.0.0.1:19081".to_string())
        .parse()
        .expect("valid workbench E2E endpoint bind");
    let endpoint_url = std::env::var("AGENT_WORKBENCH_E2E_ENDPOINT_URL")
        .unwrap_or_else(|_| format!("http://{endpoint_bind}"));
    let data_dir = std::env::temp_dir().join(format!(
        "agent-workbench-turn-ingress-e2e-{}",
        uuid::Uuid::new_v4()
    ));
    std::fs::create_dir_all(&data_dir).expect("create turn ingress E2E data dir");

    let requests = Arc::new(Mutex::new(Vec::<String>::new()));
    let requests_for_provider = Arc::clone(&requests);
    let (provider_call_tx, mut provider_call_rx) = mpsc::unbounded_channel::<usize>();
    let release_first_provider_call = Arc::new(tokio::sync::Notify::new());
    let release_first_provider_call_for_provider = Arc::clone(&release_first_provider_call);
    let response_index = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let response_index_for_provider = Arc::clone(&response_index);
    let provider = lash::testing::TestProvider::builder()
        .kind("workbench-turn-ingress-e2e")
        .complete(move |request| {
            let requests = Arc::clone(&requests_for_provider);
            let provider_call_tx = provider_call_tx.clone();
            let release_first_provider_call = Arc::clone(&release_first_provider_call_for_provider);
            let response_index = Arc::clone(&response_index_for_provider);
            async move {
                let serialized =
                    serde_json::to_string(&request).expect("serialize provider request");
                requests
                    .lock()
                    .expect("provider request lock")
                    .push(serialized);
                let call_index = response_index.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let _ = provider_call_tx.send(call_index);
                if call_index == 0 {
                    release_first_provider_call.notified().await;
                }
                Ok(match call_index {
                    0 => text_response("<lashlang>\nsleep for \"2s\"\n</lashlang>"),
                    1 => text_response("<lashlang>\nfinish \"current turn settled\"\n</lashlang>"),
                    2 => text_response("<lashlang>\nfinish \"queued turn settled\"\n</lashlang>"),
                    other => panic!("unexpected provider call {other}"),
                })
            }
        })
        .build()
        .into_handle();
    let harness = live_workbench_restate_state_with_provider(
        &data_dir,
        ingress_url,
        provider,
        WorkbenchSessionIds::fresh(),
        ActiveTurns::default(),
    )
    .await;
    restate::spawn_restate_endpoint(
        endpoint_bind,
        harness.state.clone(),
        harness.process_deployment,
        harness.process_worker,
    );
    wait_for_endpoint_socket(endpoint_bind).await;
    register_restate_deployment(&admin_url, &endpoint_url).await;

    let turn_invocation_id =
        run_workbench_turn_via_restate(&harness.state, "initial turn input_ingress_gate=true")
            .await;
    assert_eq!(
        tokio::time::timeout(Duration::from_secs(20), provider_call_rx.recv())
            .await
            .expect("first provider call timeout"),
        Some(0)
    );

    let Json(injected) = enqueue_turn_input(
        State(harness.state.clone()),
        Json(TurnInputRequest {
            text: "active injection marker".to_string(),
            ingress: TurnInputIngressRequest::ActiveTurn,
        }),
    )
    .await
    .expect("enqueue active-turn input through workbench API");
    let Json(queued) = enqueue_turn_input(
        State(harness.state.clone()),
        Json(TurnInputRequest {
            text: "queued next marker".to_string(),
            ingress: TurnInputIngressRequest::NextTurn,
        }),
    )
    .await
    .expect("enqueue next-turn input through workbench API");
    assert!(matches!(
        injected.ingress,
        lash::persistence::TurnInputIngress::ActiveTurn { .. }
    ));
    assert!(matches!(
        queued.ingress,
        lash::persistence::TurnInputIngress::NextTurn
    ));
    release_first_provider_call.notify_one();

    for expected in [1, 2] {
        assert_eq!(
            tokio::time::timeout(Duration::from_secs(30), provider_call_rx.recv())
                .await
                .unwrap_or_else(|_| panic!("provider call {expected} timeout")),
            Some(expected)
        );
    }
    wait_for_restate_invocation_success(
        &harness.state,
        &turn_invocation_id,
        Duration::from_secs(30),
    )
    .await;
    wait_for_workbench_message(
        &harness.state,
        "queued turn settled",
        Duration::from_secs(30),
    )
    .await;

    let captured = requests.lock().expect("provider request lock").clone();
    assert_eq!(captured.len(), 3, "unexpected provider request sequence");
    assert!(!captured[0].contains("active injection marker"));
    assert_eq!(
        captured[1].matches("active injection marker").count(),
        1,
        "active-turn input must reach the next provider iteration exactly once"
    );
    let completed_in_running_turn = std::fs::read_to_string(&harness.trace_path)
        .expect("read turn ingress trace")
        .lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .filter(|record| {
            record.get("name").and_then(Value::as_str) == Some("turn_input.completed")
                && record.pointer("/context/turn_id").and_then(Value::as_str)
                    == injected.ingress.active_turn_id()
                && record
                    .pointer("/payload/claims")
                    .and_then(Value::as_array)
                    .is_some_and(|claims| {
                        claims.iter().any(|claim| {
                            claim
                                .get("input_ids")
                                .and_then(Value::as_array)
                                .is_some_and(|ids| {
                                    ids.iter().any(|id| id.as_str() == Some(&injected.input_id))
                                })
                        })
                    })
        })
        .count();
    assert_eq!(
        completed_in_running_turn,
        1,
        "active-turn input must complete exactly once under the in-flight turn id; trace={} ",
        trace_tail(&harness.trace_path)
    );
    assert!(
        !captured[2].contains("active injection marker"),
        "transient active-turn input leaked into the queued full turn"
    );
    assert!(!captured[0].contains("queued next marker"));
    assert!(!captured[1].contains("queued next marker"));
    assert_eq!(captured[2].matches("queued next marker").count(), 1);

    let session = harness
        .state
        .core
        .session(harness.state.current_session_id())
        .open()
        .await
        .expect("open settled ingress session");
    let read_view = session.read_view();
    assert_eq!(
        read_view.turn_index(),
        2,
        "queued input must commit its own turn"
    );
    let committed = read_view
        .messages()
        .iter()
        .map(lash::message_text)
        .collect::<Vec<_>>();
    assert!(
        committed
            .iter()
            .any(|text| text.contains("queued next marker")),
        "queued input missing from committed transcript: {committed:#?}"
    );
    assert!(
        committed
            .iter()
            .all(|text| !text.contains("active injection marker")),
        "active injection must remain transient: {committed:#?}"
    );
    assert!(
        session
            .pending_turn_inputs()
            .await
            .expect("pending inputs after settle")
            .is_empty(),
        "both ingress claims must settle"
    );
    session.close().await.expect("close ingress session");
    let _ = std::fs::remove_dir_all(data_dir);
}

async fn live_restate_ingress_owner_restart_resumes_and_remains_cancellable_inner() {
    for backend in ["sqlite", "postgres"] {
        live_restate_ingress_owner_restart_for_store(backend).await;
    }
}

async fn live_restate_ingress_owner_restart_for_store(backend: &'static str) {
    let ingress_url = std::env::var("RESTATE_INGRESS_URL")
        .expect("RESTATE_INGRESS_URL must be set by the workbench Restate E2E recipe");
    let admin_url =
        std::env::var("RESTATE_ADMIN_URL").unwrap_or_else(|_| "http://127.0.0.1:19071".to_string());
    let endpoint_bind: SocketAddr = std::env::var("AGENT_WORKBENCH_E2E_ENDPOINT_BIND")
        .unwrap_or_else(|_| "127.0.0.1:19081".to_string())
        .parse()
        .expect("valid workbench E2E endpoint bind");
    let endpoint_url = std::env::var("AGENT_WORKBENCH_E2E_ENDPOINT_URL")
        .unwrap_or_else(|_| format!("http://{endpoint_bind}"));
    let data_dir = std::env::temp_dir().join(format!(
        "agent-workbench-recovery-{backend}-e2e-{}",
        uuid::Uuid::new_v4()
    ));
    std::fs::create_dir_all(&data_dir).expect("create recovery E2E data dir");
    let session_id = format!("workbench-recovery-{backend}-e2e");
    let turn_id = format!("workbench-turn-recovery-{backend}-e2e");
    std::fs::write(data_dir.join("session-id"), &session_id)
        .expect("write recovery E2E session id");
    let active_turns = ActiveTurns::persistent(data_dir.join("active-turns.json"))
        .expect("open recovery E2E active-turn routing");
    active_turns.insert(&session_id, &turn_id);

    let mut first = spawn_recovery_e2e_child(&data_dir, endpoint_bind, &ingress_url, backend);
    let first_pid = first.id().expect("first recovery child pid");
    wait_for_endpoint_socket(endpoint_bind).await;
    register_restate_deployment(&admin_url, &endpoint_url).await;
    let request = restate::WorkbenchTurnWorkflowRequest {
        turn_id: turn_id.clone(),
        session_id: session_id.clone(),
        text: "hold until durable cancellation".to_string(),
        model: ModelSelection {
            model: "mock-model".to_string(),
            model_variant: Some("high".to_string()),
        },
    };
    lash_restate::RestateIngressClient::new(ingress_url.clone())
        .send_workflow_json("WorkbenchTurnWorkflow", &turn_id, "run", &request)
        .await
        .expect("submit recovery E2E turn");
    wait_for_provider_owner(&data_dir, first_pid, Duration::from_secs(20)).await;
    wait_for_trace_event_count(
        &data_dir.join("trace.jsonl"),
        "llm_call_completed",
        1,
        Duration::from_secs(20),
    )
    .await;
    tokio::time::sleep(Duration::from_millis(500)).await;
    let first_generation = session_lease_generation(&data_dir, backend, &session_id).await;

    first.kill().await.expect("kill first ingress owner");
    first.wait().await.expect("reap first ingress owner");

    let restart_started = tokio::time::Instant::now();
    let mut replacement = spawn_recovery_e2e_child(&data_dir, endpoint_bind, &ingress_url, backend);
    let _replacement_pid = replacement.id().expect("replacement recovery child pid");
    wait_for_endpoint_socket(endpoint_bind).await;
    register_restate_deployment(&admin_url, &endpoint_url).await;
    wait_for_session_lease_generation(
        &data_dir,
        backend,
        &session_id,
        first_generation + 1,
        Duration::from_secs(10),
    )
    .await;
    assert!(
        restart_started.elapsed() < Duration::from_secs(10),
        "replacement waited for the session lease TTL instead of fencing the dead owner"
    );
    assert_eq!(
        session_lease_generation(&data_dir, backend, &session_id).await,
        first_generation + 1,
        "replacement must resume under a superseding session-lease generation"
    );

    let recovered_active_turns = ActiveTurns::persistent(data_dir.join("active-turns.json"))
        .expect("reopen recovered active-turn routing");
    assert_eq!(
        recovered_active_turns.for_session(&session_id),
        vec![lash::TurnAddress::new(&session_id, &turn_id)],
        "retryable resume failures must not permanently clear the durable turn address"
    );

    let address = lash::TurnAddress::new(&session_id, &turn_id);
    let driver = lash_restate::RestateTurnDeployment::new(ingress_url).turn_work_driver();
    let receipt = driver
        .request_cancel(
            lash::TurnCancelRequest::new(
                address.clone(),
                format!("workbench-recovery-{backend}-e2e-cancel"),
                Some("user".to_string()),
            )
            .with_reason("deterministic ingress-owner restart gate"),
        )
        .await
        .expect("request cancellation after ingress-owner restart");
    assert!(
        matches!(
            receipt.outcome,
            lash::TurnCancelOutcome::Requested(_) | lash::TurnCancelOutcome::AlreadyRequested(_)
        ),
        "recovered turn cancellation did not reach the durable gate: {receipt:#?}"
    );
    let terminal = driver
        .await_terminal_with_timeout(&address, Duration::from_secs(20))
        .await
        .expect("recovered turn must commit a cancellation terminal");
    let lash::TurnTerminal::Committed {
        outcome,
        cancellation,
        ..
    } = terminal
    else {
        panic!("recovered turn returned non-committed terminal: {terminal:#?}");
    };
    assert!(
        matches!(
            outcome,
            lash::TurnOutcome::Stopped(lash::TurnStop::Cancelled)
        ),
        "recovered turn did not commit Cancelled: {outcome:#?}"
    );
    let evidence = cancellation.expect("Cancelled terminal must carry evidence");
    assert_eq!(
        evidence.request_id,
        format!("workbench-recovery-{backend}-e2e-cancel")
    );
    assert_eq!(evidence.origin.as_deref(), Some("user"));
    assert_eq!(
        evidence.reason.as_deref(),
        Some("deterministic ingress-owner restart gate")
    );
    replacement.kill().await.expect("stop replacement child");
    replacement.wait().await.expect("reap replacement child");
    println!("workbench ingress-owner restart gate passed: backend={backend}");
    let _ = std::fs::remove_dir_all(data_dir);
}

fn spawn_recovery_e2e_child(
    data_dir: &std::path::Path,
    endpoint_bind: SocketAddr,
    ingress_url: &str,
    backend: &str,
) -> tokio::process::Child {
    let mut command = tokio::process::Command::new(
        std::env::current_exe().expect("resolve workbench test executable"),
    );
    command
        .arg("live_restate_ingress_owner_restart_resumes_and_remains_cancellable")
        .arg("--ignored")
        .arg("--nocapture")
        .env("AGENT_WORKBENCH_RECOVERY_E2E_CHILD", "1")
        .env("AGENT_WORKBENCH_RECOVERY_E2E_DATA_DIR", data_dir)
        .env("AGENT_WORKBENCH_RECOVERY_E2E_BACKEND", backend)
        .env(
            "AGENT_WORKBENCH_RECOVERY_E2E_ENDPOINT_BIND",
            endpoint_bind.to_string(),
        )
        .env("RESTATE_INGRESS_URL", ingress_url)
        .kill_on_drop(true);
    command.spawn().expect("spawn workbench recovery child")
}

async fn live_restate_recovery_child() {
    let data_dir = PathBuf::from(
        std::env::var("AGENT_WORKBENCH_RECOVERY_E2E_DATA_DIR").expect("recovery child data dir"),
    );
    let endpoint_bind: SocketAddr = std::env::var("AGENT_WORKBENCH_RECOVERY_E2E_ENDPOINT_BIND")
        .expect("recovery child endpoint bind")
        .parse()
        .expect("valid recovery child endpoint bind");
    let ingress_url =
        std::env::var("RESTATE_INGRESS_URL").expect("recovery child Restate ingress URL");
    let backend = std::env::var("AGENT_WORKBENCH_RECOVERY_E2E_BACKEND")
        .expect("recovery child store backend");
    let database_url = match backend.as_str() {
        "sqlite" => None,
        "postgres" => Some(
            std::env::var("AGENT_WORKBENCH_E2E_DATABASE_URL")
                .expect("Postgres recovery E2E database URL"),
        ),
        other => panic!("unsupported recovery E2E backend `{other}`"),
    };
    let provider_owner_path = data_dir.join("provider-owner");
    let provider = lash::testing::TestProvider::builder()
        .kind("workbench-recovery-e2e")
        .complete(move |_| {
            let provider_owner_path = provider_owner_path.clone();
            async move {
                std::fs::write(provider_owner_path, std::process::id().to_string())
                    .expect("record provider owner pid");
                Ok(text_response(
                    "<lashlang>\nsleep for \"60s\"\nfinish \"unreachable\"\n</lashlang>",
                ))
            }
        })
        .build()
        .into_handle();
    let active_turns = ActiveTurns::persistent(data_dir.join("active-turns.json"))
        .expect("open child active-turn routing");
    let session_ids = WorkbenchSessionIds::persistent(data_dir.join("session-id"))
        .expect("open child session id");
    let harness = live_workbench_restate_state_with_provider_and_database(
        &data_dir,
        ingress_url,
        provider,
        session_ids,
        active_turns,
        database_url.as_deref(),
    )
    .await;
    restate::spawn_restate_endpoint(
        endpoint_bind,
        harness.state,
        harness.process_deployment,
        harness.process_worker,
    );
    std::future::pending::<()>().await;
}

async fn wait_for_provider_owner(data_dir: &std::path::Path, expected_pid: u32, timeout: Duration) {
    let path = data_dir.join("provider-owner");
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        if std::fs::read_to_string(&path)
            .ok()
            .and_then(|value| value.trim().parse::<u32>().ok())
            == Some(expected_pid)
        {
            return;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "replacement ingress owner {expected_pid} did not resume the durable turn within {timeout:?}; trace tail={}",
            trace_tail(&data_dir.join("trace.jsonl")),
        );
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
}

async fn session_lease_generation(
    data_dir: &std::path::Path,
    backend: &str,
    session_id: &str,
) -> i64 {
    match backend {
        "sqlite" => {
            let sessions_dir = data_dir.join("lash-sessions");
            let database_path = std::fs::read_dir(&sessions_dir)
                .expect("read recovery E2E session store directory")
                .filter_map(Result::ok)
                .map(|entry| entry.path())
                .find(|path| path.extension().is_some_and(|extension| extension == "db"))
                .expect("locate recovery E2E SQLite session store");
            rusqlite::Connection::open(database_path)
                .expect("open recovery E2E SQLite session store")
                .query_row(
                    "SELECT lease_fencing_token FROM session_execution_leases",
                    [],
                    |row| row.get(0),
                )
                .expect("read recovery E2E SQLite session lease generation")
        }
        "postgres" => {
            let database_url = std::env::var("AGENT_WORKBENCH_E2E_DATABASE_URL")
                .expect("Postgres recovery E2E database URL");
            let pool = sqlx::PgPool::connect(&database_url)
                .await
                .expect("connect to recovery E2E Postgres");
            sqlx::query_scalar(
                "SELECT lease_fencing_token FROM lash_session_execution_leases
                 WHERE session_id = $1",
            )
            .bind(session_id)
            .fetch_one(&pool)
            .await
            .expect("read recovery E2E Postgres session lease generation")
        }
        other => panic!("unsupported recovery E2E backend `{other}`"),
    }
}

async fn wait_for_session_lease_generation(
    data_dir: &std::path::Path,
    backend: &str,
    session_id: &str,
    expected: i64,
    timeout: Duration,
) {
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        if session_lease_generation(data_dir, backend, session_id).await == expected {
            return;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "replacement did not supersede the dead session-lease generation within {timeout:?}"
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

async fn live_workbench_restate_state(
    data_dir: &std::path::Path,
    restate_ingress_url: String,
) -> LiveWorkbenchRestateHarness {
    let response_index = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let response_index_for_provider = Arc::clone(&response_index);
    let provider = lash::testing::TestProvider::builder()
        .kind("workbench-restate-e2e")
        .complete(move |_| {
            let response_index = Arc::clone(&response_index_for_provider);
            async move {
                if response_index.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 0 {
                    Ok(text_response(&format!(
                        "<lashlang>\n{}\n</lashlang>",
                        test_cron_trigger_source().trim()
                    )))
                } else {
                    Ok(text_response(
                        "<lashlang>\nfinish \"cron tick observed\"\n</lashlang>",
                    ))
                }
            }
        })
        .build()
        .into_handle();
    live_workbench_restate_state_with_provider(
        data_dir,
        restate_ingress_url,
        provider,
        WorkbenchSessionIds::fresh(),
        ActiveTurns::default(),
    )
    .await
}
