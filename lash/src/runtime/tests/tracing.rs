use super::*;

#[tokio::test]
async fn standard_runtime_trace_records_stream_event_entries() {
    let transport = mock_provider(vec![MockCall {
        stream_events: vec![
            LlmStreamEvent::Delta("Hello ".to_string()),
            LlmStreamEvent::Part(LlmOutputPart::Text {
                text: "world".to_string(),
                response_meta: None,
            }),
            LlmStreamEvent::Usage(LlmUsage {
                input_tokens: 10,
                output_tokens: 2,
                cached_input_tokens: 0,
                reasoning_tokens: 0,
            }),
        ],
        response: Ok(LlmResponse {
            full_text: "Hello world".to_string(),
            parts: vec![LlmOutputPart::Text {
                text: "Hello world".to_string(),
                response_meta: None,
            }],
            ..LlmResponse::default()
        }),
    }]);
    let trace_path = std::env::temp_dir().join(format!(
        "lash-standard-trace-{}-{}.jsonl",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos()
    ));
    let mut runtime = standard_runtime_with_transport_and_host(
        transport,
        test_host_config_with_trace_path_and_stream_events(trace_path.clone()),
    )
    .await;

    let turn = runtime
        .run_turn_assembled(
            TurnInput {
                items: vec![InputItem::Text {
                    text: "hello".to_string(),
                }],
                image_blobs: HashMap::new(),
                mode: None,
                mode_turn_options: None,
                trace_turn_id: None,
                mode_extension: None,
                turn_context: crate::TurnContext::default(),
            },
            CancellationToken::new(),
        )
        .await
        .expect("turn");

    assert!(matches!(
        &turn.outcome,
        TurnOutcome::Finished(_) | TurnOutcome::Handoff { .. }
    ));

    let logged = std::fs::read_to_string(&trace_path).expect("read trace");
    let entries = logged
        .lines()
        .map(|line| serde_json::from_str::<serde_json::Value>(line).expect("json log entry"))
        .collect::<Vec<_>>();

    assert!(
        entries
            .iter()
            .any(|entry| entry.get("type").and_then(|v| v.as_str())
                == Some("runtime_stream_event")
                && entry
                    .get("event")
                    .and_then(|payload| payload.get("event_name"))
                    .and_then(|v| v.as_str())
                    == Some("delta")
                && entry
                    .get("event")
                    .and_then(|payload| payload.get("raw_text"))
                    .and_then(|v| v.as_str())
                    == Some("Hello ")),
        "expected delta stream event in trace: {entries:?}"
    );
    assert!(
        entries
            .iter()
            .any(|entry| entry.get("type").and_then(|v| v.as_str())
                == Some("runtime_stream_event")
                && entry
                    .get("event")
                    .and_then(|payload| payload.get("event_name"))
                    .and_then(|v| v.as_str())
                    == Some("text_part")
                && entry
                    .get("event")
                    .and_then(|payload| payload.get("raw_text"))
                    .and_then(|v| v.as_str())
                    == Some("world")),
        "expected text_part stream event in trace: {entries:?}"
    );
    assert!(
        entries
            .iter()
            .any(|entry| entry.get("type").and_then(|v| v.as_str()) == Some("llm_call_completed")),
        "expected final llm trace entry in trace: {entries:?}"
    );
    let response_entry = entries
        .iter()
        .find(|entry| entry.get("type").and_then(|v| v.as_str()) == Some("llm_call_completed"))
        .expect("completed llm call entry");
    let stream_summary = response_entry
        .get("stream_summary")
        .and_then(|value| value.as_object())
        .expect("stream summary");
    assert_eq!(
        stream_summary
            .get("text_delta_count")
            .and_then(|value| value.as_u64()),
        Some(2)
    );
    assert_eq!(
        stream_summary
            .get("visible_chunk_count")
            .and_then(|value| value.as_u64()),
        Some(2)
    );
    assert_eq!(
        stream_summary
            .get("max_visible_chunk_chars")
            .and_then(|value| value.as_u64()),
        Some(6)
    );
    let avg_chunk_chars = stream_summary
        .get("avg_visible_chunk_chars")
        .and_then(|value| value.as_f64())
        .expect("avg visible chunk chars");
    assert!((avg_chunk_chars - 5.5).abs() < f64::EPSILON);
    assert!(
        stream_summary
            .get("first_visible_token_latency_ms")
            .is_some_and(|value| !value.is_null())
    );
    assert!(
        stream_summary
            .get("stream_duration_ms")
            .is_some_and(|value| !value.is_null())
    );

    let _ = std::fs::remove_file(&trace_path);
}

#[tokio::test]
async fn extended_runtime_trace_records_provider_stream_events() {
    let transport = TestProvider::builder()
        .kind("mock")
        .default_model("mock-model")
        .requires_streaming(true)
        .complete(|req| async move {
            if let Some(tx) = req.provider_trace.as_ref() {
                tx.send(LlmProviderTraceEvent {
                    provider: "codex",
                    event_name: "response.output_item.done".to_string(),
                    raw: serde_json::json!({
                        "type": "response.output_item.done",
                        "output_index": 0,
                        "item": { "id": "msg_1" }
                    })
                    .to_string(),
                });
                tx.send(LlmProviderTraceEvent {
                    provider: "codex",
                    event_name: "response.output_item.done".to_string(),
                    raw: serde_json::json!({
                        "type": "response.output_item.done",
                        "output_index": 1,
                        "item": { "id": "msg_2" }
                    })
                    .to_string(),
                });
            }
            Ok(LlmResponse {
                full_text: "Hello".to_string(),
                parts: vec![LlmOutputPart::Text {
                    text: "Hello".to_string(),
                    response_meta: None,
                }],
                ..LlmResponse::default()
            })
        })
        .build();
    let trace_path = std::env::temp_dir().join(format!(
        "lash-provider-trace-{}-{}.jsonl",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos()
    ));
    let mut runtime = standard_runtime_with_transport_and_host(
        transport,
        test_host_config_with_trace_path_and_stream_events(trace_path.clone()),
    )
    .await;

    let turn = runtime
        .run_turn_assembled(
            TurnInput {
                items: vec![InputItem::Text {
                    text: "hello".to_string(),
                }],
                image_blobs: HashMap::new(),
                mode: None,
                mode_turn_options: None,
                trace_turn_id: None,
                mode_extension: None,
                turn_context: crate::TurnContext::default(),
            },
            CancellationToken::new(),
        )
        .await
        .expect("turn");

    assert!(matches!(
        &turn.outcome,
        TurnOutcome::Finished(_) | TurnOutcome::Handoff { .. }
    ));

    let logged = std::fs::read_to_string(&trace_path).expect("read trace");
    let entries = logged
        .lines()
        .map(|line| serde_json::from_str::<serde_json::Value>(line).expect("json log entry"))
        .collect::<Vec<_>>();
    let provider_events = entries
        .iter()
        .filter(|entry| entry.get("type").and_then(|v| v.as_str()) == Some("provider_stream_event"))
        .collect::<Vec<_>>();
    assert_eq!(
        provider_events.len(),
        2,
        "provider trace entries: {entries:?}"
    );
    assert_eq!(provider_events[0]["event"]["item_id"], "msg_1");
    assert_eq!(provider_events[0]["event"]["output_index"], 0);
    assert_eq!(provider_events[1]["event"]["item_id"], "msg_2");
    assert_eq!(
        provider_events[1]["event"]["raw_json"]["item"]["id"],
        "msg_2"
    );
    assert!(
        provider_events[1]["event"]["raw_sha256"]
            .as_str()
            .is_some_and(|hash| !hash.is_empty())
    );

    let _ = std::fs::remove_file(&trace_path);
}

#[tokio::test]
async fn standard_runtime_trace_omits_stream_event_entries_by_default() {
    let transport = mock_provider(vec![MockCall {
        stream_events: vec![
            LlmStreamEvent::Delta("Hello ".to_string()),
            LlmStreamEvent::Part(LlmOutputPart::Text {
                text: "world".to_string(),
                response_meta: None,
            }),
        ],
        response: Ok(LlmResponse {
            full_text: "Hello world".to_string(),
            parts: vec![LlmOutputPart::Text {
                text: "Hello world".to_string(),
                response_meta: None,
            }],
            ..LlmResponse::default()
        }),
    }]);
    let trace_path = std::env::temp_dir().join(format!(
        "lash-standard-trace-summary-{}-{}.jsonl",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos()
    ));
    let mut runtime = standard_runtime_with_transport_and_host(
        transport,
        test_host_config_with_trace_path(trace_path.clone()),
    )
    .await;

    let turn = runtime
        .run_turn_assembled(
            TurnInput {
                items: vec![InputItem::Text {
                    text: "hello".to_string(),
                }],
                image_blobs: HashMap::new(),
                mode: None,
                mode_turn_options: None,
                trace_turn_id: None,
                mode_extension: None,
                turn_context: crate::TurnContext::default(),
            },
            CancellationToken::new(),
        )
        .await
        .expect("turn");

    assert!(matches!(
        &turn.outcome,
        TurnOutcome::Finished(_) | TurnOutcome::Handoff { .. }
    ));

    let logged = std::fs::read_to_string(&trace_path).expect("read trace");
    let entries = logged
        .lines()
        .map(|line| serde_json::from_str::<serde_json::Value>(line).expect("json log entry"))
        .collect::<Vec<_>>();

    assert!(
        !entries.iter().any(
            |entry| entry.get("type").and_then(|v| v.as_str()) == Some("runtime_stream_event")
        ),
        "stream event entries should be opt-in: {entries:?}"
    );
    let response_entry = entries
        .iter()
        .find(|entry| entry.get("type").and_then(|v| v.as_str()) == Some("llm_call_completed"))
        .expect("completed llm call entry");
    assert!(
        response_entry
            .get("stream_summary")
            .is_some_and(|value| !value.is_null()),
        "stream summary should remain in completed LLM trace: {response_entry:?}"
    );

    let _ = std::fs::remove_file(&trace_path);
}

#[tokio::test]
async fn standard_runtime_trace_records_failed_llm_calls() {
    let transport = mock_provider(vec![MockCall {
        stream_events: Vec::new(),
        response: Err(crate::llm::transport::LlmTransportError::new(
            "HTTP request failed: builder error",
        )
        .with_code("builder")
        .with_raw("transport raw body")
        .with_request_body("{\"model\":\"mock-model\"}")),
    }]);
    let trace_path = std::env::temp_dir().join(format!(
        "lash-standard-trace-error-{}-{}.jsonl",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos()
    ));
    let mut runtime = standard_runtime_with_transport_and_host(
        transport,
        test_host_config_with_trace_path(trace_path.clone()),
    )
    .await;

    let turn = runtime
        .run_turn_assembled(
            TurnInput {
                items: vec![InputItem::Text {
                    text: "hello".to_string(),
                }],
                image_blobs: HashMap::new(),
                mode: None,
                mode_turn_options: None,
                trace_turn_id: None,
                mode_extension: None,
                turn_context: crate::TurnContext::default(),
            },
            CancellationToken::new(),
        )
        .await
        .expect("turn");

    assert!(matches!(&turn.outcome, TurnOutcome::Stopped(_)));
    assert_eq!(turn.errors.len(), 1);
    assert_eq!(turn.errors[0].raw.as_deref(), Some("transport raw body"));

    let logged = std::fs::read_to_string(&trace_path).expect("read trace");
    let entries = logged
        .lines()
        .map(|line| serde_json::from_str::<serde_json::Value>(line).expect("json log entry"))
        .collect::<Vec<_>>();
    let error_entry = entries
        .iter()
        .find(|entry| entry.get("type").and_then(|v| v.as_str()) == Some("llm_call_failed"))
        .expect("llm error entry");
    assert_eq!(
        error_entry["error"]["message"].as_str(),
        Some("HTTP request failed: builder error")
    );
    assert_eq!(error_entry["error"]["code"].as_str(), Some("builder"));
    assert_eq!(
        error_entry["error"]["raw"].as_str(),
        Some("transport raw body")
    );
    let request_entry = entries
        .iter()
        .find(|entry| entry.get("type").and_then(|v| v.as_str()) == Some("llm_call_started"))
        .expect("llm request entry");
    assert_eq!(
        request_entry["request"]["model"].as_str(),
        Some("mock-model")
    );
}

#[test]
fn normalize_prompt_usage_uses_input_tokens_for_openai_compatible() {
    let usage = TokenUsage {
        input_tokens: 80,
        output_tokens: 0,
        cached_input_tokens: 20,
        reasoning_tokens: 0,
    };
    let stub = mock_provider(Vec::new()).into_handle();
    let prompt_usage = normalize_prompt_usage(&stub, &usage).expect("prompt usage");
    assert_eq!(prompt_usage.prompt_context_tokens, 80);
    assert_eq!(prompt_usage.context_budget_tokens, 80);
}
