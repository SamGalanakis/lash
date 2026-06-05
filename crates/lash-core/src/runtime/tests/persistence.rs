use super::*;

// The in-memory `RecordingStore` stands in for the real store across these
// runtime tests; the conformance suite holds it to the same durability
// contract as the durable backend so it can't silently drift.
#[tokio::test]
async fn recording_store_satisfies_runtime_persistence_conformance() {
    crate::testing::conformance::runtime_persistence_with_options(
        || {
            std::sync::Arc::new(RecordingStore::default())
                as std::sync::Arc<dyn crate::RuntimePersistence>
        },
        crate::testing::conformance::RuntimePersistenceConformance::noop_attachment_manifest(),
    )
    .await;
}

#[tokio::test]
async fn standard_runtime_assembles_stream_only_text_response() {
    let transport = mock_provider(vec![MockCall {
        stream_events: vec![
            LlmStreamEvent::Delta("What time ".to_string()),
            LlmStreamEvent::Part(LlmOutputPart::Text {
                text: "is it?".to_string(),
                response_meta: None,
            }),
            LlmStreamEvent::Usage(LlmUsage {
                input_tokens: 11,
                output_tokens: 4,
                cached_input_tokens: 0,
                reasoning_tokens: 0,
            }),
        ],
        response: Ok(LlmResponse {
            full_text: "What time is it?".to_string(),
            parts: vec![LlmOutputPart::Text {
                text: "What time is it?".to_string(),
                response_meta: None,
            }],
            ..LlmResponse::default()
        }),
    }]);
    let mut runtime = standard_runtime_with_transport(transport).await;
    let sink = RecordingSink::default();

    let turn = runtime
        .stream_turn(
            TurnInput {
                items: vec![InputItem::Text {
                    text: "hi".to_string(),
                }],
                image_blobs: HashMap::new(),
                protocol_turn_options: None,
                trace_turn_id: None,
                protocol_extension: None,
                turn_context: crate::TurnContext::default(),
            },
            TurnOptions::new(CancellationToken::new()).with_events(&sink),
        )
        .await
        .expect("turn");

    assert!(matches!(
        &turn.outcome,
        TurnOutcome::Finished(_) | TurnOutcome::AgentFrameSwitch { .. }
    ));
    assert!(matches!(
        &turn.outcome,
        TurnOutcome::Finished(TurnFinish::AssistantMessage { .. })
    ));
    assert_eq!(turn.assistant_output.safe_text, "What time is it?");
    let assistant_messages = active_conversation_messages(&turn.state)
        .into_iter()
        .filter(|message| message.role == MessageRole::Assistant)
        .collect::<Vec<_>>();
    assert_eq!(assistant_messages.len(), 1);
    assert_eq!(assistant_messages[0].parts[0].content, "What time is it?");

    let streamed_text: String = sink
        .snapshot()
        .into_iter()
        .filter_map(|event| match event {
            SessionEvent::TextDelta { content } => Some(content),
            _ => None,
        })
        .collect();
    assert_eq!(streamed_text, "What time is it?");
}

#[tokio::test]
async fn standard_runtime_recovers_streamed_text_when_final_response_is_empty() {
    let expected =
        "I’m continuing with a type-safety cleanup now: replace the remaining raw JSON paths.";
    let transport = mock_provider(vec![MockCall {
        stream_events: vec![
            LlmStreamEvent::Delta("I’m continuing with a type-safety cleanup now: ".to_string()),
            LlmStreamEvent::Part(LlmOutputPart::Text {
                text: "replace the remaining raw JSON paths.".to_string(),
                response_meta: None,
            }),
        ],
        response: Ok(LlmResponse::default()),
    }]);
    let mut runtime = standard_runtime_with_transport(transport).await;
    let sink = RecordingSink::default();

    let turn = runtime
        .stream_turn(
            TurnInput {
                items: vec![InputItem::Text {
                    text: "continue".to_string(),
                }],
                image_blobs: HashMap::new(),
                protocol_turn_options: None,
                trace_turn_id: None,
                protocol_extension: None,
                turn_context: crate::TurnContext::default(),
            },
            TurnOptions::new(CancellationToken::new()).with_events(&sink),
        )
        .await
        .expect("turn");

    assert!(matches!(
        &turn.outcome,
        TurnOutcome::Finished(_) | TurnOutcome::AgentFrameSwitch { .. }
    ));
    assert!(matches!(
        &turn.outcome,
        TurnOutcome::Finished(TurnFinish::AssistantMessage { .. })
    ));
    assert_eq!(turn.assistant_output.safe_text, expected);
    assert!(turn.errors.is_empty());
    let assistant_messages = active_conversation_messages(&turn.state)
        .into_iter()
        .filter(|message| message.role == MessageRole::Assistant)
        .collect::<Vec<_>>();
    assert_eq!(assistant_messages.len(), 1);
    assert_eq!(assistant_messages[0].parts[0].content, expected);

    let streamed_text: String = sink
        .snapshot()
        .into_iter()
        .filter_map(|event| match event {
            SessionEvent::TextDelta { content } => Some(content),
            _ => None,
        })
        .collect();
    assert_eq!(streamed_text, expected);
}

#[tokio::test]
async fn standard_runtime_cancels_in_flight_tool_calls_when_token_fires() {
    // Model emits one tool call that would sleep for 10s; we cancel the turn
    // and expect run_tool_calls to tear down promptly (< 2s), either via
    // JoinSet::abort_all or via the tool observing the cancellation token.
    let transport = mock_provider(vec![
        MockCall {
            stream_events: vec![
                LlmStreamEvent::Part(LlmOutputPart::ToolCall {
                    call_id: "slow-1".to_string(),
                    tool_name: "slow_tool".to_string(),
                    input_json: "{}".to_string(),
                    replay: None,
                }),
                LlmStreamEvent::Usage(LlmUsage {
                    input_tokens: 10,
                    output_tokens: 1,
                    cached_input_tokens: 0,
                    reasoning_tokens: 0,
                }),
            ],
            response: Ok(LlmResponse::default()),
        },
        // Extra call not expected to happen — provided as a safety net in case
        // the turn machine makes a second LLM call before noticing cancel.
        MockCall {
            stream_events: Vec::new(),
            response: Ok(LlmResponse {
                full_text: "stopped".to_string(),
                parts: vec![LlmOutputPart::Text {
                    text: "stopped".to_string(),
                    response_meta: None,
                }],
                ..LlmResponse::default()
            }),
        },
    ]);
    let observed_cancel = Arc::new(AtomicBool::new(false));
    let tools: Arc<dyn crate::ToolProvider> = Arc::new(SlowTool {
        observed_cancel: Arc::clone(&observed_cancel),
    });
    let mut runtime = runtime_with_plugins_and_tools(Vec::new(), tools, transport).await;
    let cancel = CancellationToken::new();
    let cancel_trigger = cancel.clone();
    tokio::spawn(async move {
        // Give the turn time to spawn the slow tool before we cancel.
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        cancel_trigger.cancel();
    });

    let start = std::time::Instant::now();
    let _ = runtime
        .run_turn_assembled(
            TurnInput {
                items: vec![InputItem::Text {
                    text: "trigger slow tool".to_string(),
                }],
                image_blobs: HashMap::new(),
                protocol_turn_options: None,
                trace_turn_id: None,
                protocol_extension: None,
                turn_context: crate::TurnContext::default(),
            },
            cancel,
        )
        .await;
    let elapsed = start.elapsed();

    assert!(
        elapsed < std::time::Duration::from_secs(2),
        "turn cancellation did not tear down in-flight tool call quickly: elapsed={elapsed:?}"
    );
    // The tool either saw the cancellation token and returned, or its future
    // was aborted by the JoinSet. Either outcome is acceptable — what matters
    // is the prompt return above. We still assert cooperative observation as a
    // stronger signal that the token is now plumbed through to tool context.
    assert!(
        observed_cancel.load(Ordering::SeqCst),
        "slow tool did not observe cancellation token through ToolContext"
    );
}

#[tokio::test]
async fn standard_runtime_tool_control_finish_emits_terminal_output() {
    let transport = mock_provider(vec![
        MockCall {
            stream_events: Vec::new(),
            response: Ok(LlmResponse {
                parts: vec![
                    LlmOutputPart::ToolCall {
                        call_id: "tool-1".to_string(),
                        tool_name: "terminal_tool_0".to_string(),
                        input_json: "{}".to_string(),
                        replay: None,
                    },
                    LlmOutputPart::ToolCall {
                        call_id: "tool-2".to_string(),
                        tool_name: "terminal_tool_1".to_string(),
                        input_json: "{}".to_string(),
                        replay: None,
                    },
                ],
                ..LlmResponse::default()
            }),
        },
        MockCall {
            stream_events: Vec::new(),
            response: Ok(LlmResponse {
                full_text: "unexpected follow-up".to_string(),
                parts: vec![LlmOutputPart::Text {
                    text: "unexpected follow-up".to_string(),
                    response_meta: None,
                }],
                ..LlmResponse::default()
            }),
        },
    ]);
    let tools: Arc<dyn crate::ToolProvider> = Arc::new(TerminalControlTool {
        controls: vec![
            crate::ToolControl::Finish {
                value: json!("first").into(),
            },
            crate::ToolControl::Finish {
                value: json!("second").into(),
            },
        ],
    });
    let mut runtime = runtime_with_plugins_and_tools(Vec::new(), tools, transport).await;
    let turn_events = RecordingTurnEvents::default();

    let turn = runtime
        .stream_turn(
            TurnInput {
                items: vec![InputItem::Text {
                    text: "run terminal tools".to_string(),
                }],
                image_blobs: HashMap::new(),
                protocol_turn_options: None,
                trace_turn_id: None,
                protocol_extension: None,
                turn_context: crate::TurnContext::default(),
            },
            TurnOptions::new(CancellationToken::new()).with_turn_events(&turn_events),
        )
        .await
        .expect("turn");

    assert!(
        matches!(
        turn.outcome,
        TurnOutcome::Finished(TurnFinish::ToolValue {
            ref tool_name,
            ref value,
        }) if tool_name == "terminal_tool_0" && *value == json!("first")
        ),
        "outcome={:?} calls={:?}",
        turn.outcome,
        turn.tool_calls
    );
    assert_eq!(turn.tool_calls.len(), 2);
    let events = turn_events.snapshot();
    let first_completed = events
        .iter()
        .position(|event| matches!(&event.event, TurnEvent::ToolCallCompleted { name, .. } if name == "terminal_tool_0"))
        .expect("first completed");
    let second_completed = events
        .iter()
        .position(|event| matches!(&event.event, TurnEvent::ToolCallCompleted { name, .. } if name == "terminal_tool_1"))
        .expect("second completed");
    let terminal = events
        .iter()
        .position(|event| matches!(&event.event, TurnEvent::ToolValue { .. }))
        .expect("terminal output");
    assert!(first_completed < terminal);
    assert!(second_completed < terminal);
    assert!(matches!(
        &events[terminal].event,
        TurnEvent::ToolValue {
            tool_name: name,
            value,
        } if name == "terminal_tool_0" && *value == json!("first")
    ));
}

#[tokio::test]
async fn standard_runtime_tool_control_fail_stops_without_terminal_output_event() {
    let transport = mock_provider(vec![
        MockCall {
            stream_events: Vec::new(),
            response: Ok(LlmResponse {
                parts: vec![LlmOutputPart::ToolCall {
                    call_id: "tool-1".to_string(),
                    tool_name: "terminal_tool_0".to_string(),
                    input_json: "{}".to_string(),
                    replay: None,
                }],
                ..LlmResponse::default()
            }),
        },
        MockCall {
            stream_events: Vec::new(),
            response: Ok(LlmResponse {
                full_text: "unexpected follow-up".to_string(),
                parts: vec![LlmOutputPart::Text {
                    text: "unexpected follow-up".to_string(),
                    response_meta: None,
                }],
                ..LlmResponse::default()
            }),
        },
    ]);
    let tools: Arc<dyn crate::ToolProvider> = Arc::new(TerminalControlTool {
        controls: vec![crate::ToolControl::Fail {
            failure: crate::ToolFailure::tool(
                crate::ToolFailureClass::Execution,
                "terminal_control_failed",
                "failed",
            ),
        }],
    });
    let mut runtime = runtime_with_plugins_and_tools(Vec::new(), tools, transport).await;
    let turn_events = RecordingTurnEvents::default();

    let turn = runtime
        .stream_turn(
            TurnInput {
                items: vec![InputItem::Text {
                    text: "run failing terminal tool".to_string(),
                }],
                image_blobs: HashMap::new(),
                protocol_turn_options: None,
                trace_turn_id: None,
                protocol_extension: None,
                turn_context: crate::TurnContext::default(),
            },
            TurnOptions::new(CancellationToken::new()).with_turn_events(&turn_events),
        )
        .await
        .expect("turn");

    assert!(
        matches!(
        turn.outcome,
        TurnOutcome::Stopped(TurnStop::ToolError {
            ref tool_name,
            ref value,
        }) if tool_name == "terminal_tool_0"
            && value["code"] == "terminal_control_failed"
            && value["message"] == "failed"
        ),
        "outcome={:?} calls={:?}",
        turn.outcome,
        turn.tool_calls
    );
    assert!(!turn_events.snapshot().iter().any(|event| matches!(
        &event.event,
        TurnEvent::SubmittedValue { .. } | TurnEvent::ToolValue { .. }
    )));
}

#[tokio::test]
async fn standard_runtime_executes_streamed_tool_call_when_final_response_is_empty() {
    let transport = mock_provider(vec![
        MockCall {
            stream_events: vec![
                LlmStreamEvent::Part(LlmOutputPart::ToolCall {
                    call_id: "tool-1".to_string(),
                    tool_name: "echo_tool".to_string(),
                    input_json: r#"{"value":"sample"}"#.to_string(),
                    replay: None,
                }),
                LlmStreamEvent::Usage(LlmUsage {
                    input_tokens: 12,
                    output_tokens: 3,
                    cached_input_tokens: 0,
                    reasoning_tokens: 0,
                }),
            ],
            response: Ok(LlmResponse::default()),
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
    let mut runtime = runtime_with_plugins_and_tools(Vec::new(), tools, transport).await;

    let turn = runtime
        .run_turn_assembled(
            TurnInput {
                items: vec![InputItem::Text {
                    text: "run the tool".to_string(),
                }],
                image_blobs: HashMap::new(),
                protocol_turn_options: None,
                trace_turn_id: None,
                protocol_extension: None,
                turn_context: crate::TurnContext::default(),
            },
            CancellationToken::new(),
        )
        .await
        .expect("turn");

    assert_eq!(turn.assistant_output.safe_text, "done");
    assert_eq!(active_tool_calls(&turn.state).len(), 1);
    assert_eq!(
        active_tool_calls(&turn.state)[0].call_id.as_deref(),
        Some("tool-1")
    );
    assert_eq!(
        active_tool_calls(&turn.state)[0]
            .output
            .value_for_projection(),
        serde_json::json!({
            "payload": "raw:sample"
        })
    );
}

#[tokio::test]
async fn standard_runtime_preserves_part_boundaries_when_response_is_not_streamed() {
    let transport = mock_provider(vec![MockCall {
        stream_events: vec![],
        response: Ok(LlmResponse {
            full_text: "Intro paragraph.\n\n## Heading".to_string(),
            parts: vec![
                LlmOutputPart::Text {
                    text: "Intro paragraph.".to_string(),
                    response_meta: None,
                },
                LlmOutputPart::Text {
                    text: "## Heading".to_string(),
                    response_meta: None,
                },
            ],
            ..LlmResponse::default()
        }),
    }]);
    let mut runtime = standard_runtime_with_transport(transport).await;
    let sink = RecordingSink::default();

    let turn = runtime
        .stream_turn(
            TurnInput {
                items: vec![InputItem::Text {
                    text: "hi".to_string(),
                }],
                image_blobs: HashMap::new(),
                protocol_turn_options: None,
                trace_turn_id: None,
                protocol_extension: None,
                turn_context: crate::TurnContext::default(),
            },
            TurnOptions::new(CancellationToken::new()).with_events(&sink),
        )
        .await
        .expect("turn");

    assert_eq!(
        turn.assistant_output.safe_text,
        "Intro paragraph.\n\n## Heading"
    );

    let streamed_text: String = sink
        .snapshot()
        .into_iter()
        .filter_map(|event| match event {
            SessionEvent::TextDelta { content } => Some(content),
            _ => None,
        })
        .collect();
    assert_eq!(streamed_text, "Intro paragraph.\n\n## Heading");
}

#[tokio::test]
async fn standard_runtime_uses_streamed_usage_when_final_usage_missing() {
    let transport = mock_provider(vec![MockCall {
        stream_events: vec![
            LlmStreamEvent::Delta("Hi".to_string()),
            LlmStreamEvent::Usage(LlmUsage {
                input_tokens: 9,
                output_tokens: 3,
                cached_input_tokens: 2,
                reasoning_tokens: 0,
            }),
        ],
        response: Ok(LlmResponse {
            full_text: "Hi".to_string(),
            parts: vec![LlmOutputPart::Text {
                text: "Hi".to_string(),
                response_meta: None,
            }],
            usage: LlmUsage::default(),
            ..LlmResponse::default()
        }),
    }]);
    let mut runtime = standard_runtime_with_transport(transport).await;

    let turn = runtime
        .run_turn_assembled(
            TurnInput {
                items: vec![InputItem::Text {
                    text: "hello".to_string(),
                }],
                image_blobs: HashMap::new(),
                protocol_turn_options: None,
                trace_turn_id: None,
                protocol_extension: None,
                turn_context: crate::TurnContext::default(),
            },
            CancellationToken::new(),
        )
        .await
        .expect("turn");

    assert_eq!(turn.token_usage.input_tokens, 9);
    assert_eq!(turn.token_usage.output_tokens, 3);
    assert_eq!(turn.token_usage.cached_input_tokens, 2);
}

#[tokio::test]
async fn standard_runtime_prefers_final_usage_over_streamed_usage() {
    let transport = mock_provider(vec![MockCall {
        stream_events: vec![
            LlmStreamEvent::Delta("Hi".to_string()),
            LlmStreamEvent::Usage(LlmUsage {
                input_tokens: 9,
                output_tokens: 3,
                cached_input_tokens: 2,
                reasoning_tokens: 0,
            }),
        ],
        response: Ok(LlmResponse {
            full_text: "Hi".to_string(),
            parts: vec![LlmOutputPart::Text {
                text: "Hi".to_string(),
                response_meta: None,
            }],
            usage: LlmUsage {
                input_tokens: 12,
                output_tokens: 4,
                cached_input_tokens: 1,
                reasoning_tokens: 0,
            },
            ..LlmResponse::default()
        }),
    }]);
    let mut runtime = standard_runtime_with_transport(transport).await;

    let turn = runtime
        .run_turn_assembled(
            TurnInput {
                items: vec![InputItem::Text {
                    text: "hello".to_string(),
                }],
                image_blobs: HashMap::new(),
                protocol_turn_options: None,
                trace_turn_id: None,
                protocol_extension: None,
                turn_context: crate::TurnContext::default(),
            },
            CancellationToken::new(),
        )
        .await
        .expect("turn");

    assert_eq!(turn.token_usage.input_tokens, 12);
    assert_eq!(turn.token_usage.output_tokens, 4);
    assert_eq!(turn.token_usage.cached_input_tokens, 1);
}
