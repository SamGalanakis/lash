use super::*;

#[test]
fn session_usage_report_aggregates_sources_and_models() {
    let entries = vec![
        TokenLedgerEntry {
            source: "turn".to_string(),
            model: "gpt-5.4-mini".to_string(),
            usage: TokenUsage {
                input_tokens: 10,
                output_tokens: 2,
                cached_input_tokens: 3,
                reasoning_tokens: 1,
            },
        },
        TokenLedgerEntry {
            source: "observer".to_string(),
            model: "gpt-5.4-mini".to_string(),
            usage: TokenUsage {
                input_tokens: 7,
                output_tokens: 1,
                cached_input_tokens: 0,
                reasoning_tokens: 0,
            },
        },
        TokenLedgerEntry {
            source: "turn".to_string(),
            model: "gpt-5.4".to_string(),
            usage: TokenUsage {
                input_tokens: 20,
                output_tokens: 4,
                cached_input_tokens: 5,
                reasoning_tokens: 2,
            },
        },
    ];

    let report = SessionUsageReport::from_entries(&entries);

    assert_eq!(report.entry_count, 3);
    assert_eq!(report.usage.input_tokens, 37);
    assert_eq!(report.usage.output_tokens, 7);
    assert_eq!(report.usage.cached_input_tokens, 8);
    assert_eq!(report.usage.reasoning_tokens, 3);
    assert_eq!(report.usage.total_tokens, 47);
    assert_eq!(report.usage.context_total_tokens, 55);
    assert_eq!(report.by_source["turn"].input_tokens, 30);
    assert_eq!(report.by_source["observer"].output_tokens, 1);
    assert_eq!(report.by_model["gpt-5.4-mini"].input_tokens, 17);
    assert_eq!(report.by_model["gpt-5.4"].reasoning_tokens, 2);

    let delta = diff_token_ledger(
        &[TokenLedgerEntry {
            source: "turn".to_string(),
            model: "gpt-5.4-mini".to_string(),
            usage: TokenUsage {
                input_tokens: 10,
                output_tokens: 2,
                cached_input_tokens: 3,
                reasoning_tokens: 1,
            },
        }],
        &entries,
    )
    .expect("delta");
    assert_eq!(delta.len(), 2);
    assert_eq!(delta[0].source, "observer");
    assert_eq!(delta[1].model, "gpt-5.4");
}

#[tokio::test]
async fn await_background_work_waits_for_registered_jobs() {
    let runtime = standard_runtime_with_transport_and_background(mock_provider(Vec::new())).await;
    let manager = runtime.session_manager().expect("session manager");
    let observed = Arc::new(AtomicBool::new(false));
    let observed_task = Arc::clone(&observed);

    manager
        .spawn_hidden_task(
            "root",
            "test",
            Box::pin(async move {
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                observed_task.store(true, Ordering::SeqCst);
                Ok(())
            }),
        )
        .await
        .expect("spawn background job");

    let mut runtime = runtime;
    runtime
        .await_background_work()
        .await
        .expect("await background work");
    assert!(observed.load(Ordering::SeqCst));
}

#[tokio::test]
async fn await_background_work_does_not_cross_runtime_sessions_with_same_logical_id() {
    let executor: Arc<dyn SessionTaskExecutor> = Arc::new(TokioSessionTaskExecutor::default());
    let runtime_one = standard_runtime_with_shared_background_executor(
        mock_provider(Vec::new()),
        Arc::clone(&executor),
    )
    .await;
    let runtime_two = standard_runtime_with_shared_background_executor(
        mock_provider(Vec::new()),
        Arc::clone(&executor),
    )
    .await;
    let manager_one = runtime_one.session_manager().expect("session manager");
    let observed = Arc::new(AtomicBool::new(false));
    let observed_task = Arc::clone(&observed);
    manager_one
        .spawn_hidden_task(
            "root",
            "test",
            Box::pin(async move {
                tokio::time::sleep(std::time::Duration::from_millis(40)).await;
                observed_task.store(true, Ordering::SeqCst);
                Ok(())
            }),
        )
        .await
        .expect("spawn background job");

    let mut runtime_two = runtime_two;
    tokio::time::timeout(
        std::time::Duration::from_millis(10),
        runtime_two.await_background_work(),
    )
    .await
    .expect("second runtime should not block on first runtime jobs")
    .expect("await background work");
    assert!(!observed.load(Ordering::SeqCst));

    let mut runtime_one = runtime_one;
    runtime_one
        .await_background_work()
        .await
        .expect("first runtime await background work");
    assert!(observed.load(Ordering::SeqCst));
}
