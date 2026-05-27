use super::*;

#[tokio::test]
async fn plugin_surface_streams_as_semantic_turn_event() -> Result<()> {
    let core = LashCore::standard()
        .provider(mock_provider())
        .model(mock_model_spec())
        .plugin(Arc::new(SurfacePluginFactory))
        .build()?;
    let session = core.session("plugin-surface").open().await?;
    let events = RecordingEvents::default();

    session
        .turn(TurnInput::text("hello"))
        .stream(&events)
        .await?;

    let surface = events
        .snapshot()
        .await
        .into_iter()
        .find(|event| matches!(&event.event, TurnEvent::PluginRuntime { .. }))
        .expect("plugin surface event");
    let TurnEvent::PluginRuntime { plugin_id, event } = surface.event else {
        unreachable!();
    };
    assert_eq!(plugin_id, "surface_test");
    assert!(matches!(
        event,
        lash_core::PluginRuntimeEvent::Status { key, label, .. }
        if key == "surface" && label == "working"
    ));
    Ok(())
}

#[tokio::test]
async fn embedded_sessions_always_expose_tool_state() -> Result<()> {
    let core = standard_core();
    let session = core.session("dynamic-default").open().await?;

    let state = session.control().tools().state().await?;

    assert!(state.generation() > 0);
    Ok(())
}

#[tokio::test]
async fn registered_static_tools_appear_in_tool_state() -> Result<()> {
    let core = LashCore::standard()
        .provider(mock_provider())
        .model(mock_model_spec())
        .tools(Arc::new(AppTools))
        .build()?;
    let session = core.session("static-tools").open().await?;

    let state = session.control().tools().state().await?;

    assert!(state.contains("app_lookup"));
    Ok(())
}

#[tokio::test]
async fn apply_tool_state_and_availability_update_live_catalog() -> Result<()> {
    let core = LashCore::standard()
        .provider(mock_provider())
        .model(mock_model_spec())
        .tools(Arc::new(AppTools))
        .build()?;
    let session = core.session("tool-state").open().await?;

    let generation = session
        .control()
        .tools()
        .set_availability_many(&[("app_lookup", ToolAvailability::Showcased)])
        .await?;
    let showcased = session.control().tools().state().await?;

    assert_eq!(showcased.generation(), generation);
    assert_eq!(
        showcased
            .get("app_lookup")
            .and_then(|spec| spec.manifest().availability_override),
        Some(ToolAvailability::Showcased)
    );

    let generation = session
        .control()
        .tools()
        .clear_availability_override("app_lookup")
        .await?;
    let cleared = session.control().tools().state().await?;

    assert_eq!(cleared.generation(), generation);
    assert_eq!(
        cleared
            .get("app_lookup")
            .and_then(|spec| spec.manifest().availability_override),
        None
    );

    let generation = session
        .control()
        .tools()
        .set_availability("app_lookup", ToolAvailability::Off)
        .await?;
    let off = session.control().tools().state().await?;

    assert_eq!(off.generation(), generation);
    assert_eq!(
        off.get("app_lookup")
            .and_then(|spec| spec.manifest().availability_override),
        Some(ToolAvailability::Off)
    );

    let mut callable = off;
    callable
        .set_availability("app_lookup", Some(ToolAvailability::Callable))
        .expect("app tool");
    let generation = session
        .control()
        .tools()
        .advanced()
        .apply_state(callable)
        .await?;
    let callable = session.control().tools().state().await?;

    assert_eq!(callable.generation(), generation);
    assert_eq!(
        callable
            .get("app_lookup")
            .and_then(|spec| spec.manifest().availability_override),
        Some(ToolAvailability::Callable)
    );
    Ok(())
}

#[tokio::test]
async fn persisted_session_restores_tool_state() -> Result<()> {
    let core = LashCore::standard()
        .provider(mock_provider())
        .model(mock_model_spec())
        .tools(Arc::new(AppTools))
        .build()?;
    let session = core.session("persisted-tools").open().await?;
    session
        .control()
        .tools()
        .set_availability("app_lookup", ToolAvailability::Off)
        .await?;
    let persisted_tool_state = session.control().tools().state().await?;
    let state = RuntimeSessionState {
        session_id: "persisted-tools".to_string(),
        policy: lash_core::SessionPolicy {
            provider: mock_provider(),
            model: mock_model_spec(),
            ..Default::default()
        },
        tool_state_snapshot: Some(persisted_tool_state),
        ..Default::default()
    };
    let store: Arc<dyn lash_core::RuntimePersistence> = Arc::new(SnapshotStore::with_state(state));
    let reopened_core = LashCore::standard()
        .provider(mock_provider())
        .model(mock_model_spec())
        .tools(Arc::new(AppTools))
        .store_factory(Arc::new(ReusableStoreFactory { store }))
        .build()?;

    let reopened = reopened_core.session("persisted-tools").open().await?;
    let state = reopened.control().tools().state().await?;

    assert_eq!(
        state
            .get("app_lookup")
            .and_then(|spec| spec.manifest().availability_override),
        Some(ToolAvailability::Off)
    );
    Ok(())
}

#[test]
fn tool_completed_activity_is_canonical_while_model_observation_is_projected() -> Result<()> {
    std::thread::Builder::new()
        .name("tool-projection-stack-test".to_string())
        .stack_size(4 * 1024 * 1024)
        .spawn(|| {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("test runtime")
                .block_on(async {
                    let projection = Arc::new(crate::plugins::ToolOutputBudgetPluginFactory::new(
                        crate::plugins::ToolOutputBudgetConfig {
                            mode: crate::plugins::ToolOutputBudgetMode::Bytes,
                            limit: 12,
                            max_lines: 4,
                        },
                    ));
                    let observed_tool_results = Arc::new(TokioMutex::new(Vec::<String>::new()));
                    let observed_tool_results_provider = Arc::clone(&observed_tool_results);
                    let responses = Arc::new(TokioMutex::new(VecDeque::from([
                        LlmResponse {
                            parts: vec![LlmOutputPart::ToolCall {
                                call_id: "call-1".to_string(),
                                tool_name: "app_lookup".to_string(),
                                input_json: "{}".to_string(),
                                replay: None,
                            }],
                            ..LlmResponse::default()
                        },
                        LlmResponse {
                            full_text: "done".to_string(),
                            parts: vec![LlmOutputPart::Text {
                                text: "done".to_string(),
                                response_meta: None,
                            }],
                            ..LlmResponse::default()
                        },
                    ])));
                    let standard_provider = lash_core::testing::TestProvider::builder()
                        .kind("embed-test")
                        .complete(move |request| {
                            let observed_tool_results = Arc::clone(&observed_tool_results_provider);
                            let responses = Arc::clone(&responses);
                            async move {
                                for message in &request.messages {
                                    for block in message.blocks.iter() {
                                        if let LlmContentBlock::ToolResult { content, .. } = block {
                                            observed_tool_results
                                                .lock()
                                                .await
                                                .push(content.clone());
                                        }
                                    }
                                }
                                Ok(responses.lock().await.pop_front().expect("queued response"))
                            }
                        })
                        .build()
                        .into_handle();
                    let standard_core = LashCore::standard()
                        .provider(standard_provider)
                        .model(mock_model_spec())
                        .tools(Arc::new(LongTextTools))
                        .process_registry(Arc::new(TestLocalProcessRegistry::default()))
                        .configure_plugins(|plugins| {
                            plugins.replace(projection.clone());
                        })
                        .build()?;
                    let standard_session =
                        standard_core.session("standard-projection").open().await?;
                    let standard_events = RecordingEvents::default();
                    let _ = standard_session
                        .turn(TurnInput::text("use tool"))
                        .stream(&standard_events)
                        .await?;
                    let standard_view = standard_events
                        .snapshot()
                        .await
                        .into_iter()
                        .find_map(|event| match event.event {
                            TurnEvent::ToolCallCompleted { output, .. } => {
                                Some(output.value_for_projection())
                            }
                            _ => None,
                        })
                        .expect("standard tool completion");
                    assert_eq!(
                        standard_view,
                        serde_json::json!("abcdefghijklmnopqrstuvwxyz0123456789")
                    );
                    let observed = observed_tool_results.lock().await;
                    let model_observation = observed
                        .iter()
                        .find(|content| content.contains("bytes truncated"))
                        .expect("projected model observation");
                    assert!(model_observation.contains("Full output saved to:"));

                    let rlm_core = LashCore::rlm()
                        .provider(queued_text_provider(vec![
                            "```lashlang\nvalue = await TOOL.default.app_lookup({})?\nsubmit \"done\"\n```",
                        ]))
                        .model(mock_model_spec())
                        .tools(Arc::new(LongTextTools))
                        .process_registry(Arc::new(TestLocalProcessRegistry::default()))
                        .configure_plugins(|plugins| {
                            plugins.replace(projection);
                        })
                        .build()?;
                    let rlm_session = rlm_core.session("rlm-projection").open().await?;
                    let rlm_events = RecordingEvents::default();
                    let _ = rlm_session
                        .turn(TurnInput::text("use tool"))
                        .stream(&rlm_events)
                        .await?;
                    let rlm_view = rlm_events
                        .snapshot()
                        .await
                        .into_iter()
                        .find_map(|event| match event.event {
                            TurnEvent::ToolCallCompleted { output, .. } => {
                                Some(output.value_for_projection())
                            }
                            _ => None,
                        })
                        .expect("rlm tool completion");

                    assert_eq!(rlm_view, standard_view);
                    Ok(())
                })
        })
        .expect("spawn stack-sized test thread")
        .join()
        .expect("stack-sized test thread panicked")
}
