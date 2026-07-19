#[cfg(test)]
mod turn_control_timeout_tests {
    use super::tests::{explicit_durable_test_facets, run_async_test_on_stack_budget};
    use super::*;

    #[test]
    fn dangling_routed_turn_does_not_hang_stop_and_is_pruned() {
        run_async_test_on_stack_budget("workbench-dangling-turn-cancel-test", || {
            dangling_routed_turn_does_not_hang_stop_and_is_pruned_inner()
        });
    }

    async fn dangling_routed_turn_does_not_hang_stop_and_is_pruned_inner() {
        let data_dir = std::env::temp_dir().join(format!(
            "agent-workbench-dangling-turn-cancel-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&data_dir).expect("create temp workbench dir");
        let process_registry = Arc::new(
            lash_sqlite_store::SqliteProcessRegistry::open(&data_dir.join("processes.db"))
                .await
                .expect("open registry"),
        ) as Arc<dyn lash::process::ProcessRegistry>;
        let store_factory: Arc<dyn lash::persistence::SessionStoreFactory> = Arc::new(
            lash_sqlite_store::SqliteSessionStoreFactory::new(data_dir.join("lash-sessions")),
        );
        let provider = lash::testing::TestProvider::builder()
            .kind("workbench-test")
            .complete_error("dangling turn test should not call the provider")
            .build()
            .into_handle();
        let model =
            lash::ModelSpec::from_token_limits("test-model", Default::default(), 4096, None)
                .expect("model spec");
        let (event_tx, _) = broadcast::channel(16);
        let core = explicit_durable_test_facets(&data_dir)
            .provider(provider)
            .model(model)
            .store_factory(Arc::clone(&store_factory))
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
            process_work_driver: inert_process_work_driver(Arc::clone(&process_registry)),
            session_ids: WorkbenchSessionIds::fresh(),
            messages: Arc::new(Mutex::new(Vec::new())),
            selected_model: Arc::new(Mutex::new(ModelSelection {
                model: "test-model".to_string(),
                model_variant: Default::default(),
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
            active_turns: ActiveTurns::persistent(data_dir.join("active-turns.json"))
                .expect("open active turns"),
        };
        let session_id = state.current_session_id();
        state.track_turn(&session_id, "dangling-turn");

        let receipts = tokio::time::timeout(
            Duration::from_secs(1),
            state.cancel_turns_for_session(&session_id),
        )
        .await
        .expect("Stop must not hang on a dangling routed turn")
        .expect("cancel dangling turn");

        assert!(matches!(
            receipts.as_slice(),
            [TurnCancelReceipt {
                outcome: lash::TurnCancelOutcome::Requested(_),
                terminal: None,
                terminal_error: Some(error),
                ..
            }] if error.code.as_str() == "turn_terminal_await_timeout"
        ));
        assert!(state.active_turns.for_session(&session_id).is_empty());
        let recovered = ActiveTurns::persistent(data_dir.join("active-turns.json"))
            .expect("reopen active turns");
        assert!(recovered.for_session(&session_id).is_empty());
        let _ = std::fs::remove_dir_all(data_dir);
    }
}
