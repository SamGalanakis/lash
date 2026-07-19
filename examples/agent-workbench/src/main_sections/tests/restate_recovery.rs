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

async fn live_restate_ingress_owner_restart_resumes_and_remains_cancellable_inner() {
    let ingress_url = std::env::var("RESTATE_INGRESS_URL")
        .expect("RESTATE_INGRESS_URL must be set by the workbench Restate E2E recipe");
    let admin_url = std::env::var("RESTATE_ADMIN_URL")
        .unwrap_or_else(|_| "http://127.0.0.1:19071".to_string());
    let endpoint_bind: SocketAddr = std::env::var("AGENT_WORKBENCH_E2E_ENDPOINT_BIND")
        .unwrap_or_else(|_| "127.0.0.1:19081".to_string())
        .parse()
        .expect("valid workbench E2E endpoint bind");
    let endpoint_url = std::env::var("AGENT_WORKBENCH_E2E_ENDPOINT_URL")
        .unwrap_or_else(|_| format!("http://{endpoint_bind}"));
    let data_dir = std::env::temp_dir().join(format!(
        "agent-workbench-recovery-e2e-{}",
        uuid::Uuid::new_v4()
    ));
    std::fs::create_dir_all(&data_dir).expect("create recovery E2E data dir");
    let session_id = "workbench-recovery-e2e".to_string();
    let turn_id = "workbench-turn-recovery-e2e".to_string();
    std::fs::write(data_dir.join("session-id"), &session_id)
        .expect("write recovery E2E session id");
    let active_turns = ActiveTurns::persistent(data_dir.join("active-turns.json"))
        .expect("open recovery E2E active-turn routing");
    active_turns.insert(&session_id, &turn_id);

    let mut first = spawn_recovery_e2e_child(&data_dir, endpoint_bind, &ingress_url);
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

    first.kill().await.expect("kill first ingress owner");
    first.wait().await.expect("reap first ingress owner");

    let mut replacement = spawn_recovery_e2e_child(&data_dir, endpoint_bind, &ingress_url);
    let replacement_pid = replacement.id().expect("replacement recovery child pid");
    wait_for_endpoint_socket(endpoint_bind).await;
    register_restate_deployment(&admin_url, &endpoint_url).await;
    wait_for_provider_owner(&data_dir, replacement_pid, Duration::from_secs(45)).await;

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
                "workbench-recovery-e2e-cancel",
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
        matches!(outcome, lash::TurnOutcome::Stopped(lash::TurnStop::Cancelled)),
        "recovered turn did not commit Cancelled: {outcome:#?}"
    );
    let evidence = cancellation.expect("Cancelled terminal must carry evidence");
    assert_eq!(evidence.request_id, "workbench-recovery-e2e-cancel");
    assert_eq!(evidence.origin.as_deref(), Some("user"));
    assert_eq!(
        evidence.reason.as_deref(),
        Some("deterministic ingress-owner restart gate")
    );
    replacement.kill().await.expect("stop replacement child");
    replacement.wait().await.expect("reap replacement child");
    let _ = std::fs::remove_dir_all(data_dir);
}

fn spawn_recovery_e2e_child(
    data_dir: &std::path::Path,
    endpoint_bind: SocketAddr,
    ingress_url: &str,
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
        std::env::var("AGENT_WORKBENCH_RECOVERY_E2E_DATA_DIR")
            .expect("recovery child data dir"),
    );
    let endpoint_bind: SocketAddr = std::env::var("AGENT_WORKBENCH_RECOVERY_E2E_ENDPOINT_BIND")
        .expect("recovery child endpoint bind")
        .parse()
        .expect("valid recovery child endpoint bind");
    let ingress_url =
        std::env::var("RESTATE_INGRESS_URL").expect("recovery child Restate ingress URL");
    let provider_owner_path = data_dir.join("provider-owner");
    let provider = lash::testing::TestProvider::builder()
        .kind("workbench-recovery-e2e")
        .complete(move |_| {
            let provider_owner_path = provider_owner_path.clone();
            async move {
                std::fs::write(provider_owner_path, std::process::id().to_string())
                    .expect("record provider owner pid");
                std::future::pending::<()>().await;
                Ok(text_response("<lashlang>\nfinish \"unreachable\"\n</lashlang>"))
            }
        })
        .build()
        .into_handle();
    let active_turns = ActiveTurns::persistent(data_dir.join("active-turns.json"))
        .expect("open child active-turn routing");
    let session_ids = WorkbenchSessionIds::persistent(data_dir.join("session-id"))
        .expect("open child session id");
    let harness = live_workbench_restate_state_with_provider(
        &data_dir,
        ingress_url,
        provider,
        session_ids,
        active_turns,
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

async fn wait_for_provider_owner(
    data_dir: &std::path::Path,
    expected_pid: u32,
    timeout: Duration,
) {
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
