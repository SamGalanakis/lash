use super::*;

#[tokio::test]
async fn tool_result_projector_only_changes_model_observation() {
    let committed_results = Arc::new(tokio::sync::Mutex::new(Vec::<(
        serde_json::Value,
        serde_json::Value,
    )>::new()));
    let committed_results_hook = Arc::clone(&committed_results);
    let plugin = Arc::new(RuntimeTestPluginFactory {
        build: Arc::new(move |_| {
            let committed_results = Arc::clone(&committed_results_hook);
            Ok(Arc::new(RuntimeTestPlugin {
                before_turn: None,
                checkpoint: None,
                tool_result_projector: Some(Arc::new(|ctx| {
                    Box::pin(async move {
                        Ok(crate::ModelToolReturn::text(
                            ctx.call_id,
                            ctx.tool_name,
                            "model projection",
                        ))
                    })
                })),
                runtime_event: Some(Arc::new(move |event| {
                    let committed_results = Arc::clone(&committed_results);
                    Box::pin(async move {
                        if let crate::plugin::PluginLifecycleEvent::TurnFinalized(turn) = event {
                            committed_results.lock().await.push((
                                turn.tool_calls
                                    .first()
                                    .map(|call| call.output.value_for_projection().clone())
                                    .unwrap_or(serde_json::Value::Null),
                                turn.state
                                    .read_model()
                                    .tool_calls
                                    .first()
                                    .map(|call| call.output.value_for_projection().clone())
                                    .unwrap_or(serde_json::Value::Null),
                            ));
                        }
                        Ok(())
                    })
                })),
                external_registrar: None,
            }))
        }),
    });
    let transport = mock_provider(vec![
        MockCall {
            stream_events: Vec::new(),
            response: Ok(LlmResponse {
                parts: vec![
                    LlmOutputPart::Text {
                        text: "checking tool".to_string(),
                        response_meta: None,
                    },
                    LlmOutputPart::ToolCall {
                        call_id: "tool-1".to_string(),
                        tool_name: "echo_tool".to_string(),
                        input_json: r#"{"value":"sample"}"#.to_string(),
                        replay: None,
                    },
                ],
                ..LlmResponse::default()
            }),
        },
        MockCall {
            stream_events: Vec::new(),
            response: Ok(LlmResponse {
                full_text: "done".to_string(),
                parts: vec![LlmOutputPart::Text {
                    text: "done".to_string(),
                    response_meta: None,
                }],
                ..LlmResponse::default()
            }),
        },
    ]);
    let tools: Arc<dyn crate::ToolProvider> = Arc::new(EchoTool);
    let mut runtime = runtime_with_plugins_and_tools(vec![plugin], tools, transport).await;

    let turn = runtime
        .run_turn_assembled(
            TurnInput {
                items: vec![InputItem::Text {
                    text: "run the tool".to_string(),
                }],
                image_blobs: HashMap::new(),
                mode_turn_options: None,
                trace_turn_id: None,
                mode_extension: None,
                turn_context: crate::TurnContext::default(),
            },
            CancellationToken::new(),
        )
        .await
        .expect("turn");

    assert!(
        active_conversation_messages(&turn.state)
            .iter()
            .any(|message| {
                message.parts.iter().any(|part| {
                    part.content.contains("model projection")
                        && matches!(part.kind, PartKind::ToolResult)
                })
            })
    );
    let committed = committed_results.lock().await;
    assert_eq!(
        committed.as_slice(),
        &[(
            serde_json::json!({ "payload": "raw:sample" }),
            serde_json::json!({ "payload": "raw:sample" }),
        )]
    );
    assert_eq!(active_tool_calls(&turn.state).len(), 1);
    assert_eq!(
        active_tool_calls(&turn.state)[0].call_id.as_deref(),
        Some("tool-1")
    );
    assert_eq!(turn.tool_calls.len(), 1);
    assert_eq!(turn.tool_calls[0].call_id.as_deref(), Some("tool-1"));
    assert_eq!(
        active_tool_calls(&turn.state)[0]
            .output
            .value_for_projection(),
        serde_json::json!({ "payload": "raw:sample" })
    );
    assert_eq!(
        turn.tool_calls[0].output.value_for_projection(),
        serde_json::json!({ "payload": "raw:sample" })
    );
}

#[tokio::test]
async fn completed_turns_are_persisted_for_custom_runtime_store() {
    let transport = mock_provider(vec![MockCall {
        stream_events: vec![LlmStreamEvent::Delta("Stored answer".to_string())],
        response: Ok(LlmResponse {
            full_text: "Stored answer".to_string(),
            parts: vec![LlmOutputPart::Text {
                text: "Stored answer".to_string(),
                response_meta: None,
            }],
            usage: LlmUsage {
                input_tokens: 12,
                output_tokens: 4,
                cached_input_tokens: 1,
                reasoning_tokens: 2,
            },
            ..LlmResponse::default()
        }),
    }]);

    let store = Arc::new(RecordingStore::default());
    let plugins =
        plugin_session_with_tools("root", ExecutionMode::standard(), Arc::new(EmptyTools));
    let mut runtime = LashRuntime::from_persistent_embedded_state(
        standard_test_policy(),
        test_host_config(),
        crate::PersistentRuntimeServices::new(
            Arc::clone(&plugins),
            store.clone() as Arc<dyn crate::store::RuntimePersistence>,
        ),
        RuntimeSessionState::default(),
    )
    .await
    .expect("runtime");
    runtime.policy.provider = transport.clone().into_handle();

    let _turn = runtime
        .run_turn_assembled(
            TurnInput {
                items: vec![InputItem::Text {
                    text: "where did this go?".to_string(),
                }],
                image_blobs: HashMap::new(),
                mode_turn_options: None,
                trace_turn_id: None,
                mode_extension: None,
                turn_context: crate::TurnContext::default(),
            },
            CancellationToken::new(),
        )
        .await
        .expect("turn");

    let read_model = crate::store::RuntimePersistence::load_session(
        store.as_ref(),
        crate::store::SessionReadScope::FullGraph,
    )
    .await
    .expect("load session")
    .expect("session head")
    .graph
    .read_model();
    let messages = read_model.messages.as_slice();
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0].role, MessageRole::User);
    assert_eq!(messages[0].parts[0].content, "where did this go?");
    assert_eq!(messages[1].role, MessageRole::Assistant);
    assert_eq!(messages[1].parts[0].content, "Stored answer");
}

#[tokio::test]
async fn park_returns_error_when_final_commit_fails() {
    let store = Arc::new(RecordingStore::default());
    store
        .save_session_head_meta(crate::SessionHeadMeta {
            session_id: "other-session".to_string(),
            ..crate::SessionHeadMeta::default()
        })
        .await;
    let plugins = plugin_session_with_tools(
        "park-session",
        ExecutionMode::standard(),
        Arc::new(EmptyTools),
    );
    let runtime = LashRuntime::from_persistent_embedded_state(
        standard_test_policy(),
        test_host_config(),
        crate::PersistentRuntimeServices::new(
            plugins,
            store as Arc<dyn crate::store::RuntimePersistence>,
        ),
        RuntimeSessionState {
            session_id: "park-session".to_string(),
            policy: standard_test_policy(),
            ..RuntimeSessionState::default()
        },
    )
    .await
    .expect("runtime");

    let err = match runtime.park().await {
        Ok(_) => panic!("park should fail when final persistence fails"),
        Err(err) => err,
    };

    let message = err.to_string();
    assert!(message.contains("failed to persist runtime state"));
    assert!(message.contains("other-session"));
    assert!(message.contains("park-session"));
}

#[tokio::test]
async fn completed_turns_are_persisted_in_session_graph() {
    let transport = mock_provider(vec![MockCall {
        stream_events: vec![
            LlmStreamEvent::Delta("Stored answer".to_string()),
            LlmStreamEvent::Usage(LlmUsage {
                input_tokens: 12,
                output_tokens: 4,
                cached_input_tokens: 1,
                reasoning_tokens: 2,
            }),
        ],
        response: Ok(LlmResponse {
            full_text: "Stored answer".to_string(),
            parts: vec![LlmOutputPart::Text {
                text: "Stored answer".to_string(),
                response_meta: None,
            }],
            usage: LlmUsage {
                input_tokens: 12,
                output_tokens: 4,
                cached_input_tokens: 1,
                reasoning_tokens: 2,
            },
            ..LlmResponse::default()
        }),
    }]);

    let store = Arc::new(RecordingStore::default());
    let base_provider: Arc<dyn crate::ToolProvider> = Arc::new(EmptyTools);
    let base_provider_factory = Arc::clone(&base_provider);
    let plugin_host = crate::PluginHost::new(vec![Arc::new(StaticPluginFactory::new(
        "base_tools",
        crate::PluginSpec::new().with_tool_provider(Arc::clone(&base_provider_factory)),
    ))]);
    let plugins = plugin_host
        .build_standard_session("root", None)
        .expect("plugins");
    let mut runtime = LashRuntime::from_persistent_embedded_state(
        standard_test_policy(),
        test_host_config(),
        crate::PersistentRuntimeServices::new(
            Arc::clone(&plugins),
            store.clone() as Arc<dyn crate::store::RuntimePersistence>,
        ),
        RuntimeSessionState::default(),
    )
    .await
    .expect("runtime");
    runtime.policy.provider = transport.clone().into_handle();

    let _turn = runtime
        .run_turn_assembled(
            TurnInput {
                items: vec![InputItem::Text {
                    text: "where did this go?".to_string(),
                }],
                image_blobs: HashMap::new(),
                mode_turn_options: None,
                trace_turn_id: None,
                mode_extension: None,
                turn_context: crate::TurnContext::default(),
            },
            CancellationToken::new(),
        )
        .await
        .expect("turn");

    let read = crate::store::RuntimePersistence::load_session(
        store.as_ref(),
        crate::store::SessionReadScope::FullGraph,
    )
    .await
    .expect("load session")
    .expect("session read");
    let graph = read.graph;
    let read_model = graph.read_model();
    let messages = read_model.messages.as_slice();
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0].parts[0].content, "where did this go?");
    assert_eq!(messages[1].parts[0].content, "Stored answer");
    let _checkpoint = read.checkpoint.expect("checkpoint");
    let ledger = read.token_ledger;
    assert_eq!(ledger.len(), 1);
    assert_eq!(ledger[0].source, "turn");
    assert_eq!(ledger[0].model, standard_test_policy().model.id);
    assert_eq!(ledger[0].usage.input_tokens, 12);
    assert_eq!(ledger[0].usage.output_tokens, 4);
    assert_eq!(ledger[0].usage.cached_input_tokens, 1);
    assert_eq!(ledger[0].usage.reasoning_tokens, 2);
}
