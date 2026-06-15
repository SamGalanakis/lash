use super::*;

#[derive(Clone)]
struct VecRegistry(Vec<RemoteToolGrant>);

impl RemoteToolRegistry for VecRegistry {
    fn grants(&self) -> Vec<RemoteToolGrant> {
        self.0.clone()
    }
}

#[test]
fn remote_llm_request_json_round_trips() {
    let request = RemoteLlmRequest {
        protocol_version: REMOTE_PROTOCOL_VERSION,
        request_id: "request-1".to_string(),
        model_intent: RemoteModelIntent::new("gpt-test"),
        messages: vec![RemoteLlmMessage {
            role: RemoteLlmRole::User,
            content: vec![RemoteLlmContentBlock::Text {
                text: "hello".to_string(),
                response_meta: None,
                cache_breakpoint: false,
            }],
        }],
        attachments: vec![RemoteLlmAttachment {
            id: Some("img".to_string()),
            mime: "image/png".to_string(),
            data_base64: Some("AQID".to_string()),
            reference: None,
            metadata: HashMap::new(),
        }],
        tools: Vec::new(),
        tool_choice: RemoteLlmToolChoice::Auto,
        output_spec: Some(RemoteLlmOutputSpec::JsonObject),
        generation: RemoteGenerationOptions {
            output_token_cap: Some(128),
            ..Default::default()
        },
        request_metadata: RemoteLlmRequestMetadata {
            session_id: Some("session".to_string()),
            idempotency_key: Some("idem".to_string()),
            trace_id: None,
        },
        metadata: HashMap::new(),
    };

    request.validate().expect("valid request");
    let value = serde_json::to_value(&request).expect("serialize");
    let decoded: RemoteLlmRequest = serde_json::from_value(value).expect("deserialize");
    assert_eq!(decoded.protocol_version, REMOTE_PROTOCOL_VERSION);
    assert_eq!(decoded.request_id, request.request_id);
    assert_eq!(decoded.messages, request.messages);
}

#[test]
fn remote_llm_response_json_round_trips() {
    let response = RemoteLlmResponse {
        protocol_version: REMOTE_PROTOCOL_VERSION,
        request_id: "request-1".to_string(),
        full_text: "done".to_string(),
        output_parts: vec![RemoteLlmOutputPart::Text {
            text: "done".to_string(),
            response_meta: None,
        }],
        usage: RemoteUsage {
            input_tokens: 1,
            output_tokens: 2,
            cached_input_tokens: 0,
            reasoning_tokens: 0,
        },
        terminal_reason: RemoteLlmTerminalReason::Stop,
        diagnostics: Vec::new(),
        provider_metadata: RemoteProviderMetadata::default(),
    };

    response.validate().expect("valid response");
    let value = serde_json::to_value(&response).expect("serialize");
    let decoded: RemoteLlmResponse = serde_json::from_value(value).expect("deserialize");
    assert_eq!(decoded.protocol_version, REMOTE_PROTOCOL_VERSION);
    assert_eq!(decoded.full_text, "done");
}

#[test]
fn remote_turn_request_json_round_trips() {
    let request = RemoteTurnRequest {
        protocol_version: REMOTE_PROTOCOL_VERSION,
        session_id: "session".to_string(),
        turn_id: "turn".to_string(),
        idempotency_key: Some("idem".to_string()),
        input: RemoteTurnInput {
            protocol_version: REMOTE_PROTOCOL_VERSION,
            items: vec![
                RemoteInputItem::Text {
                    text: "first".to_string(),
                },
                RemoteInputItem::ImageRef {
                    id: "img".to_string(),
                },
            ],
            image_blobs_base64: HashMap::from([("img".to_string(), "AQID".to_string())]),
            protocol_turn_options: Some(RemoteProtocolTurnOptions {
                payload: serde_json::json!({ "answer": "raw" }),
            }),
            trace_turn_id: Some("trace".to_string()),
            prompt_layer: Some(RemotePromptLayer::new()),
        },
        tool_grants: vec![demo_grant("demo", "tools", "search")],
        model_intent: Some(RemoteModelIntent::new("gpt-test")),
        metadata: HashMap::new(),
    };

    request.validate().expect("valid request");
    let value = serde_json::to_value(&request).expect("serialize");
    let decoded: RemoteTurnRequest = serde_json::from_value(value).expect("deserialize");

    assert_eq!(decoded.protocol_version, REMOTE_PROTOCOL_VERSION);
    assert_eq!(decoded.session_id, "session");
    assert_eq!(decoded.input.image_blobs_base64["img"], "AQID");
    assert_eq!(decoded.tool_grants.len(), 1);
}

#[test]
fn remote_turn_result_json_round_trips() {
    let result = RemoteTurnResult {
        protocol_version: REMOTE_PROTOCOL_VERSION,
        session_id: "session".to_string(),
        turn_id: "turn".to_string(),
        status: RemoteTurnStatus::Completed,
        outcome: RemoteTurnOutcome::Finished {
            finish: RemoteTurnFinish::AssistantMessage {
                text: "done".to_string(),
            },
        },
        assistant_output: RemoteAssistantOutput {
            safe_text: "done".to_string(),
            raw_text: "done".to_string(),
            state: RemoteAssistantOutputState::Usable,
        },
        usage: RemoteTurnUsageSummary::default(),
        execution: RemoteExecutionSummary::default(),
        tool_calls: vec![RemoteToolCallSummary {
            call_id: Some("call".to_string()),
            tool_name: "demo".to_string(),
            args: serde_json::json!({"x": 1}),
            outcome: RemoteToolCallOutcome::Success(serde_json::json!({"ok": true})),
            duration_ms: 5,
        }],
        issues: Vec::new(),
        activities: vec![RemoteTurnActivity {
            protocol_version: REMOTE_PROTOCOL_VERSION,
            sequence: 1,
            id: "event".to_string(),
            correlation_id: "corr".to_string(),
            event: RemoteTurnEvent::AssistantProseDelta {
                text: "done".to_string(),
            },
        }],
        metadata: HashMap::new(),
    };

    result.validate().expect("valid result");
    let value = serde_json::to_value(&result).expect("serialize");
    let decoded: RemoteTurnResult = serde_json::from_value(value).expect("deserialize");
    assert_eq!(decoded.protocol_version, REMOTE_PROTOCOL_VERSION);
    assert_eq!(decoded.session_id, "session");
    assert_eq!(decoded.tool_calls.len(), 1);
}

#[test]
fn remote_trigger_dtos_json_round_trip() {
    let request = RemoteTriggerOccurrenceRequest::new(
        "ui.button.pressed",
        "source-key",
        serde_json::json!({ "button": "Blue" }),
        "button-blue-1",
    )
    .with_source(serde_json::json!({ "id": "blue" }));
    request
        .validate()
        .expect("valid trigger occurrence request");
    let decoded: RemoteTriggerOccurrenceRequest =
        serde_json::from_value(serde_json::to_value(&request).expect("serialize request"))
            .expect("deserialize request");
    assert_eq!(decoded.protocol_version, REMOTE_PROTOCOL_VERSION);
    assert_eq!(decoded.source_type, "ui.button.pressed");
    assert_eq!(decoded.source.as_ref().unwrap()["id"], "blue");

    let report = RemoteTriggerEmitReport {
        protocol_version: REMOTE_PROTOCOL_VERSION,
        occurrence_id: "occurrence:1".to_string(),
        started_process_ids: vec!["process:1".to_string()],
    };
    report.validate().expect("valid report");
    let decoded: RemoteTriggerEmitReport =
        serde_json::from_value(serde_json::to_value(&report).expect("serialize report"))
            .expect("deserialize report");
    assert_eq!(decoded.started_process_ids, vec!["process:1".to_string()]);

    let mut filter = RemoteTriggerSubscriptionFilter::for_source_type("ui.button.pressed");
    filter.source_key = Some("source-key".to_string());
    filter.enabled = Some(true);
    filter.validate().expect("valid filter");
    let decoded: RemoteTriggerSubscriptionFilter =
        serde_json::from_value(serde_json::to_value(&filter).expect("serialize filter"))
            .expect("deserialize filter");
    assert_eq!(decoded.source_key.as_deref(), Some("source-key"));

    let registration = RemoteTriggerRegistration {
        handle: "trigger:1".to_string(),
        source_key: "source-key".to_string(),
        name: Some("button watcher".to_string()),
        source_type: "ui.button.pressed".to_string(),
        source: serde_json::json!({}),
        target: RemoteTriggerTargetSummary {
            process_name: "on_button".to_string(),
            inputs: serde_json::json!({ "event": "trigger.event" }),
        },
        enabled: true,
    };
    let decoded: RemoteTriggerRegistration = serde_json::from_value(
        serde_json::to_value(&registration).expect("serialize registration"),
    )
    .expect("deserialize registration");
    assert_eq!(decoded.target.process_name, "on_button");

    let cause = RemoteCausalRef::TriggerOccurrence {
        occurrence_id: "occurrence:1".to_string(),
    };
    let value = serde_json::to_value(&cause).expect("serialize cause");
    assert_eq!(value["type"], "trigger_occurrence");
    assert_eq!(value["occurrence_id"], "occurrence:1");
}

#[test]
fn remote_session_observation_dtos_json_round_trip_typed_kinds() {
    let event = RemoteSessionObservationEvent {
        protocol_version: REMOTE_PROTOCOL_VERSION,
        session_id: "session".to_string(),
        revision: 3,
        cursor: "lashsc1:3:7:session".to_string(),
        event: RemoteSessionObservationEventPayload::QueueChanged {
            kind: RemoteSessionQueueEventKind::Enqueued,
            batch_ids: vec!["batch-1".to_string()],
        },
    };
    event.validate().expect("valid queue event");
    let value = serde_json::to_value(&event).expect("serialize event");
    assert!(
        value.to_string().contains("\"kind\":\"enqueued\""),
        "queue kind should serialize as snake_case: {value}"
    );
    let decoded: RemoteSessionObservationEvent =
        serde_json::from_value(value).expect("deserialize event");
    assert_eq!(decoded, event);

    let process = RemoteSessionObservationEventPayload::ProcessChanged {
        kind: RemoteSessionProcessEventKind::Cancelled,
        process_ids: vec!["process-1".to_string()],
    };
    let value = serde_json::to_value(&process).expect("serialize process payload");
    assert!(
        value.to_string().contains("\"kind\":\"cancelled\""),
        "process kind should serialize as snake_case: {value}"
    );
    let decoded: RemoteSessionObservationEventPayload =
        serde_json::from_value(value).expect("deserialize process payload");
    assert_eq!(decoded, process);
}

#[test]
fn remote_session_observation_schema_includes_typed_kind_enums() {
    let schema = schemars::schema_for!(RemoteSessionObservationEvent);
    let schema_text = serde_json::to_value(&schema)
        .expect("schema json")
        .to_string();
    assert!(
        schema_text.contains("enqueued") && schema_text.contains("started"),
        "schema did not include typed observation kind enum values: {schema_text}"
    );
}

#[test]
fn wrong_protocol_versions_are_rejected() {
    let mut input = RemoteTurnInput::text("hello");
    input.protocol_version = REMOTE_PROTOCOL_VERSION + 1;
    assert!(matches!(
        input.validate(),
        Err(RemoteProtocolError::UnsupportedProtocolVersion { .. })
    ));

    let mut grant = demo_grant("one", "tools", "search");
    grant.protocol_version = REMOTE_PROTOCOL_VERSION + 1;
    assert!(matches!(
        grant.validate(),
        Err(RemoteProtocolError::UnsupportedProtocolVersion { .. })
    ));

    let request = RemoteToolCallRequest {
        protocol_version: REMOTE_PROTOCOL_VERSION + 1,
        tool_name: "demo".to_string(),
        call_path: "tools.demo".to_string(),
        args: serde_json::Value::Null,
        session_id: "session".to_string(),
        completion_key: serde_json::json!({"key": "test"}),
        tool_call_id: None,
        replay_key: None,
        attempt_number: 1,
        max_attempts: 1,
        headers: HashMap::new(),
    };
    assert!(matches!(
        request.validate(),
        Err(RemoteProtocolError::UnsupportedProtocolVersion { .. })
    ));

    let response = RemoteToolCallResponse::Success {
        protocol_version: REMOTE_PROTOCOL_VERSION + 1,
        value: serde_json::Value::Null,
    };
    assert!(matches!(
        response.validate(),
        Err(RemoteProtocolError::UnsupportedProtocolVersion { .. })
    ));

    let activity = RemoteTurnActivity {
        protocol_version: REMOTE_PROTOCOL_VERSION + 1,
        sequence: 1,
        id: "event".to_string(),
        correlation_id: "corr".to_string(),
        event: RemoteTurnEvent::AssistantProseDelta {
            text: "hi".to_string(),
        },
    };
    assert!(matches!(
        activity.validate(),
        Err(RemoteProtocolError::UnsupportedProtocolVersion { .. })
    ));

    let mut event = RemoteTriggerOccurrenceRequest::new(
        "ui.button.pressed",
        "source-key",
        serde_json::Value::Null,
        "idem",
    );
    event.protocol_version = REMOTE_PROTOCOL_VERSION + 1;
    assert!(matches!(
        event.validate(),
        Err(RemoteProtocolError::UnsupportedProtocolVersion { .. })
    ));

    let mut filter = RemoteTriggerSubscriptionFilter::for_session("session");
    filter.protocol_version = REMOTE_PROTOCOL_VERSION + 1;
    assert!(matches!(
        filter.validate(),
        Err(RemoteProtocolError::UnsupportedProtocolVersion { .. })
    ));

    let report = RemoteTriggerEmitReport {
        protocol_version: REMOTE_PROTOCOL_VERSION + 1,
        occurrence_id: "occurrence:1".to_string(),
        started_process_ids: Vec::new(),
    };
    assert!(matches!(
        report.validate(),
        Err(RemoteProtocolError::UnsupportedProtocolVersion { .. })
    ));
}

#[test]
fn remote_tool_call_request_requires_completion_key() {
    let request = RemoteToolCallRequest {
        protocol_version: REMOTE_PROTOCOL_VERSION,
        tool_name: "demo".to_string(),
        call_path: "tools.demo".to_string(),
        args: serde_json::Value::Null,
        session_id: "session".to_string(),
        completion_key: serde_json::Value::Null,
        tool_call_id: None,
        replay_key: None,
        attempt_number: 1,
        max_attempts: 1,
        headers: HashMap::new(),
    };

    let err = request
        .validate()
        .expect_err("remote tool requests require a completion key");
    assert!(matches!(
        err,
        RemoteProtocolError::RemoteToolTransport(message)
            if message.contains("completion_key")
    ));
}

#[cfg(feature = "core-conversions")]
#[test]
fn remote_pending_tool_response_maps_to_pending_tool_result() {
    let result = RemoteToolCallResponse::Pending {
        protocol_version: REMOTE_PROTOCOL_VERSION,
        deadline_ms: Some(250),
        on_timeout: RemoteTimeoutBehavior::FailTurn,
        on_cancel: RemoteCancelHint::Ignore,
    }
    .into_tool_result();

    let pending = result
        .into_done_output()
        .expect_err("pending remote response must not project as completed output");
    assert_eq!(
        pending.deadline,
        Some(std::time::Duration::from_millis(250))
    );
    assert_eq!(pending.on_timeout, lash_core::TimeoutBehavior::FailTurn);
    assert_eq!(pending.on_cancel, lash_core::CancelHint::Ignore);
}

#[test]
fn nested_protocol_versions_must_match_envelope() {
    let mut request = RemoteTurnRequest {
        protocol_version: REMOTE_PROTOCOL_VERSION,
        session_id: "session".to_string(),
        turn_id: "turn".to_string(),
        idempotency_key: None,
        input: RemoteTurnInput::text("hello"),
        tool_grants: Vec::new(),
        model_intent: None,
        metadata: HashMap::new(),
    };
    request.input.protocol_version = REMOTE_PROTOCOL_VERSION + 1;
    assert!(matches!(
        request.validate(),
        Err(RemoteProtocolError::MismatchedNestedProtocolVersion { .. })
    ));
}

#[test]
fn top_level_protocol_schema_exports_include_versions() {
    assert_schema_has_protocol_version::<RemoteLlmRequest>();
    assert_schema_has_protocol_version::<RemoteLlmResponse>();
    assert_schema_has_protocol_version::<RemoteTurnInput>();
    assert_schema_has_protocol_version::<RemoteTurnRequest>();
    assert_schema_has_protocol_version::<RemoteTurnResult>();
    assert_schema_has_protocol_version::<RemoteSessionCursor>();
    assert_schema_has_protocol_version::<RemoteSessionObservationEvent>();
    assert_schema_has_protocol_version::<RemoteLiveReplayGap>();
    assert_schema_has_protocol_version::<RemoteToolGrant>();
    assert_schema_has_protocol_version::<RemoteToolCallRequest>();
    assert_schema_has_protocol_version::<RemoteToolCallResponse>();
    assert_schema_has_protocol_version::<RemoteTurnActivity>();
    assert_schema_has_protocol_version::<RemoteTriggerOccurrenceRequest>();
    assert_schema_has_protocol_version::<RemoteTriggerEmitReport>();
    assert_schema_has_protocol_version::<RemoteTriggerSubscriptionFilter>();
}

#[test]
fn remote_tool_registry_reopen_conformance_compares_call_paths() {
    let before = VecRegistry(vec![demo_grant("one", "tools", "search")]);
    let reopened = VecRegistry(vec![demo_grant("one", "tools", "search")]);
    assert_remote_tool_registry_reopenable(&before, &reopened).expect("same registry");

    let changed = VecRegistry(vec![demo_grant("one", "tools", "read")]);
    assert!(matches!(
        assert_remote_tool_registry_reopenable(&before, &changed),
        Err(RemoteProtocolError::RemoteToolRegistryReopenMismatch { .. })
    ));
}

fn demo_grant(name: &str, module: &str, operation: &str) -> RemoteToolGrant {
    RemoteToolGrant {
        protocol_version: REMOTE_PROTOCOL_VERSION,
        id: None,
        name: name.to_string(),
        description: "demo".to_string(),
        input_schema: default_input_schema(),
        output_schema: serde_json::Value::Null,
        input_schema_projections: Vec::new(),
        output_schema_projections: Vec::new(),
        output_contract: RemoteToolOutputContract::Static,
        examples: Vec::new(),
        availability: None,
        activation: None,
        argument_projection: None,
        scheduling: None,
        retry_policy: None,
        lashlang_binding: Some(RemoteLashlangToolBinding::new([module], operation)),
    }
}

fn assert_schema_has_protocol_version<T: JsonSchema>() {
    let schema = schemars::schema_for!(T);
    let schema_json = serde_json::to_value(&schema).expect("schema json");
    let schema_text = schema_json.to_string();
    assert!(
        schema_text.contains("protocol_version"),
        "schema did not include protocol_version: {schema_text}"
    );
}
