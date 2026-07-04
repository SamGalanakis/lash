#[cfg(test)]
mod process_work_tests {
    use super::tests::{explicit_durable_test_facets, run_async_test_on_stack_budget};
    use super::*;

    #[test]
    fn await_work_route_returns_terminal_outcome_and_reconciled_events() {
        run_async_test_on_stack_budget("workbench-await-work-test", || {
            await_work_route_returns_terminal_outcome_and_reconciled_events_inner()
        });
    }

    async fn await_work_route_returns_terminal_outcome_and_reconciled_events_inner() {
        let data_dir = std::env::temp_dir().join(format!(
            "agent-workbench-await-work-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&data_dir).expect("create temp workbench dir");
        let process_registry = Arc::new(
            lash_sqlite_store::SqliteProcessRegistry::open(&data_dir.join("processes.db"))
                .await
                .expect("open registry"),
        ) as Arc<dyn lash::process::ProcessRegistry>;
        let session_store_factory = Arc::new(lash_sqlite_store::SqliteSessionStoreFactory::new(
            data_dir.join("lash-sessions"),
        ));
        let core_store_factory: Arc<dyn lash::persistence::SessionStoreFactory> =
            session_store_factory;
        let provider = lash::testing::TestProvider::builder()
            .kind("workbench-test")
            .complete_error("await-work route test should not call the provider")
            .build()
            .into_handle();
        let model =
            lash::ModelSpec::from_token_limits("test-model", None, 4096, None).expect("model spec");
        let (event_tx, _) = broadcast::channel(16);
        // The app sink, wired exactly as bootstrap wires it — through the
        // driver's watched decorator, feeding an mpsc channel.
        let (sink_tx, mut sink_rx) = mpsc::channel::<lash::process::ProcessEvent>(16);
        let driver = lash::process::ProcessWorkDriver::new_with_sink(
            Arc::clone(&process_registry),
            Arc::new(NoopProcessRunHandle),
            Some(Arc::new(ChannelProcessEventSink::new(sink_tx))),
        );
        let core = explicit_durable_test_facets(&data_dir)
            .provider(provider)
            .model(model)
            .store_factory(Arc::clone(&core_store_factory))
            .process_registry(Arc::clone(&process_registry))
            .build()
            .expect("build core");
        let process_observer = core
            .processes()
            .observer()
            .expect("process observer configured");
        let state = AppState {
            core,
            process_observer,
            process_work_driver: driver.clone(),
            session_ids: WorkbenchSessionIds::fresh(),
            messages: Arc::new(Mutex::new(Vec::new())),
            selected_model: Arc::new(Mutex::new(ModelSelection {
                model: "test-model".to_string(),
                model_variant: None,
            })),
            web_configured: false,
            trace_sink: None,
            lashlang_execution: Arc::new(TraceLashlangGraphStore::default()),
            event_tx,
            queued_work_driver: inert_queued_work_driver(),
            restate_ingress_url: "http://127.0.0.1:8080".to_string(),
            restate_admin_url: "http://127.0.0.1:9070".to_string(),
            restate_http: reqwest::Client::new(),
            restate_cron_job_keys: Arc::new(Mutex::new(BTreeSet::new())),
            mail_world: mail::MailWorld::new(),
            active_restate_invocations: ActiveRestateInvocations::default(),
        };

        // Register, append one non-terminal event, and complete — all through
        // the driver's decorated registry handle, the same one the sink watches.
        let watched = driver.process_registry();
        watched
            .register_process(
                lash::process::ProcessRegistration::new(
                    "await-route-proc",
                    lash::process::ProcessInput::External {
                        metadata: Value::Null,
                    },
                    lash::process::ProcessProvenance::host(),
                )
                .with_extra_event_types([lash::process::ProcessEventType {
                    name: "progress".to_string(),
                    payload_schema: lash::triggers::LashSchema::any(),
                    semantics: Default::default(),
                }]),
            )
            .await
            .expect("register process");
        watched
            .append_event(
                "await-route-proc",
                lash::process::ProcessEventAppendRequest::new("progress", json!({ "step": 1 })),
            )
            .await
            .expect("append progress event");
        watched
            .complete_process(
                "await-route-proc",
                lash::process::ProcessAwaitOutput::Success {
                    value: json!("done"),
                    control: None,
                },
            )
            .await
            .expect("complete process");

        let Json(result) = await_work(
            AxumPath("await-route-proc".to_string()),
            State(state.clone()),
        )
        .await
        .expect("await work route");

        // Terminal outcome rides the await seam (ADR 0016)...
        assert!(matches!(
            &result.outcome,
            lash::process::ProcessAwaitOutput::Success { value, .. } if value == &json!("done")
        ));
        // ...and the event log reconciled from the durable store is complete.
        assert!(
            result
                .events
                .iter()
                .any(|event| event.event_type == "progress"),
            "reconciled events missing the appended progress event: {:?}",
            result.events
        );

        // The sink saw the non-terminal append (best-effort freshness)...
        let mut sunk = Vec::new();
        while let Ok(event) = sink_rx.try_recv() {
            sunk.push(event);
        }
        assert!(
            sunk.iter().any(|event| event.event_type == "progress"),
            "sink missed the non-terminal append: {sunk:?}"
        );
        // ...but never the terminal event — completion rides `await_terminal`,
        // not the sink (ADR 0017).
        assert!(
            sunk.iter().all(|event| event.semantics.terminal.is_none()),
            "terminal event must not ride the sink: {sunk:?}"
        );

        // An unknown process id errors instead of hanging.
        let missing = await_work(AxumPath("no-such-process".to_string()), State(state)).await;
        assert!(missing.is_err(), "unknown process id must error");
        let _ = std::fs::remove_dir_all(data_dir);
    }
}
