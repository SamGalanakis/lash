use std::sync::{Arc, Mutex};

use super::*;

struct RecordingTransport {
    requests: Mutex<Vec<RemoteToolCallRequest>>,
    response: RemoteToolCallResponse,
}

impl RemoteToolTransport for RecordingTransport {
    fn send<'a>(
        &'a self,
        request: RemoteToolCallRequest,
    ) -> Pin<
        Box<dyn Future<Output = Result<RemoteToolCallResponse, RemoteProtocolError>> + Send + 'a>,
    > {
        Box::pin(async move {
            self.requests.lock().expect("requests lock").push(request);
            Ok(self.response.clone())
        })
    }
}

#[test]
fn turn_input_round_trips_remote_safe_fields() {
    let mut prompt = lash_core::PromptLayer::new();
    prompt.add_contribution(lash_core::PromptContribution::guidance("Guide", "remote"));
    let mut input = lash_core::TurnInput::items([
        lash_core::InputItem::text("a"),
        lash_core::InputItem::text("b"),
        lash_core::InputItem::image_ref("img"),
    ])
    .with_image_blob("img", vec![1, 2, 3])
    .with_protocol_turn_options(lash_core::ProtocolTurnOptions {
        payload: serde_json::json!({ "mode": "remote" }),
    })
    .with_trace_turn_id("trace-1");
    input.turn_context.set_prompt_layer(prompt.clone());

    let remote = RemoteTurnInput::try_from(input).expect("remote conversion");
    assert_eq!(remote.items.len(), 3);
    assert_eq!(remote.image_blobs_base64["img"], "AQID");
    assert_eq!(remote.trace_turn_id.as_deref(), Some("trace-1"));
    assert_eq!(
        remote.protocol_turn_options.as_ref().unwrap().payload,
        serde_json::json!({ "mode": "remote" })
    );
    assert_eq!(remote.prompt_layer, Some(prompt.clone().into()));

    let core = lash_core::TurnInput::try_from(remote).expect("core conversion");
    assert_eq!(core.image_blobs["img"], vec![1, 2, 3]);
    assert_eq!(core.trace_turn_id.as_deref(), Some("trace-1"));
    assert_eq!(
        core.protocol_turn_options.unwrap().payload,
        serde_json::json!({ "mode": "remote" })
    );
    assert_eq!(core.turn_context.prompt_layer(), &prompt);
}

#[test]
fn turn_input_rejects_non_remote_safe_fields() {
    struct DummyTurnExtension;

    impl lash_core::ProtocolTurnExtension for DummyTurnExtension {
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    let mut input = lash_core::TurnInput::text("extension");
    input.protocol_extension = Some(lash_core::ProtocolTurnExtensionHandle::new(
        DummyTurnExtension,
    ));
    assert!(matches!(
        RemoteTurnInput::try_from(input),
        Err(RemoteProtocolError::NonRemoteSafeTurnInput(message))
            if message.contains("protocol turn")
    ));

    let mut input = lash_core::TurnInput::text("live");
    input.turn_context.insert_plugin_input("demo", 1_u32);
    assert!(matches!(
        RemoteTurnInput::try_from(input),
        Err(RemoteProtocolError::NonRemoteSafeTurnInput(message))
            if message.contains("live plugin")
    ));

    let mut input = lash_core::TurnInput::text("provider");
    input.turn_context.set_provider(
        lash_core::testing::TestProvider::builder()
            .build()
            .into_handle(),
    );
    assert!(matches!(
        RemoteTurnInput::try_from(input),
        Err(RemoteProtocolError::NonRemoteSafeTurnInput(message))
            if message.contains("provider")
    ));

    let mut input = lash_core::TurnInput::text("model");
    input
        .turn_context
        .set_model(lash_core::ModelSpec::from_token_limits("m", None, 100, None).expect("model"));
    assert!(matches!(
        RemoteTurnInput::try_from(input),
        Err(RemoteProtocolError::NonRemoteSafeTurnInput(message))
            if message.contains("model")
    ));
}

#[test]
fn llm_request_and_response_round_trip_owned_dtos() {
    let request = core_llm::LlmRequest {
        model: "gpt-test".to_string(),
        messages: vec![core_llm::LlmMessage::text(core_llm::LlmRole::User, "hello")],
        attachments: vec![core_llm::LlmAttachment::bytes("image/png", vec![1, 2, 3])],
        tools: Arc::new(vec![core_llm::LlmToolSpec {
            name: "search".to_string(),
            description: "Search".to_string(),
            input_schema: serde_json::json!({"type": "object"}),
            output_schema: serde_json::Value::Null,
            input_schema_projections: Vec::new(),
            output_schema_projections: Vec::new(),
        }]),
        tool_choice: core_llm::LlmToolChoice::Auto,
        model_variant: Some("fast".to_string()),
        generation: core_llm::GenerationOptions {
            output_token_cap: NonZeroUsize::new(42),
        },
        session_id: Some("session-1".to_string()),
        output_spec: Some(core_llm::LlmOutputSpec::JsonObject),
        stream_events: None,
        provider_trace: None,
    };

    let remote = RemoteLlmRequest::from_core("request-1", request);
    remote.validate().expect("valid remote request");
    assert_eq!(remote.protocol_version, REMOTE_PROTOCOL_VERSION);
    assert_eq!(remote.request_id, "request-1");
    let core = core_llm::LlmRequest::try_from(remote).expect("core request");
    assert_eq!(core.model, "gpt-test");
    assert_eq!(core.model_variant.as_deref(), Some("fast"));
    assert_eq!(core.attachments[0].data, vec![1, 2, 3]);

    let response = core_llm::LlmResponse {
        full_text: "done".to_string(),
        parts: vec![core_llm::LlmOutputPart::Text {
            text: "done".to_string(),
            response_meta: None,
        }],
        usage: core_llm::LlmUsage {
            input_tokens: 1,
            output_tokens: 2,
            cached_input_tokens: 0,
            reasoning_tokens: 0,
        },
        terminal_reason: core_llm::LlmTerminalReason::Stop,
        terminal_diagnostic: Some("ok".to_string()),
        provider_usage: Some(serde_json::json!({"provider": "usage"})),
        request_body: Some("{}".to_string()),
        http_summary: Some("200".to_string()),
    };
    let remote = RemoteLlmResponse::from_core("request-1", response);
    remote.validate().expect("valid remote response");
    let core = core_llm::LlmResponse::from(remote);
    assert_eq!(core.full_text, "done");
    assert_eq!(core.terminal_reason, core_llm::LlmTerminalReason::Stop);
    assert_eq!(
        core.provider_usage,
        Some(serde_json::json!({"provider": "usage"}))
    );
}

#[test]
fn prompt_layer_round_trips_without_protocol_crate_depending_on_core_by_default() {
    let template = lash_core::PromptTemplate::new(vec![lash_core::PromptTemplateSection::titled(
        "Custom",
        vec![lash_core::PromptTemplateEntry::slot(
            lash_core::PromptSlot::Guidance,
        )],
    )]);
    let prompt = lash_core::PromptLayer::with_template(template)
        .with_contribution(lash_core::PromptContribution::guidance("Guide", "remote"));

    let remote = RemotePromptLayer::from(prompt.clone());
    let core = lash_core::PromptLayer::from(remote);
    assert_eq!(core, prompt);
}

#[test]
fn remote_turn_result_maps_core_semantics() {
    let turn = lash_core::AssembledTurn {
        state: Default::default(),
        outcome: lash_core::TurnOutcome::Finished(lash_core::TurnFinish::AssistantMessage {
            text: "done".to_string(),
        }),
        assistant_output: lash_core::AssistantOutput {
            safe_text: "done".to_string(),
            raw_text: "done".to_string(),
            state: lash_core::OutputState::Usable,
        },
        execution: lash_core::ExecutionSummary {
            had_tool_calls: true,
            had_code_execution: false,
        },
        token_usage: lash_core::TokenUsage {
            input_tokens: 1,
            output_tokens: 2,
            cached_input_tokens: 0,
            reasoning_tokens: 0,
        },
        children_usage: vec![lash_core::TokenLedgerEntry {
            source: "subagent".to_string(),
            model: "m".to_string(),
            usage: lash_core::TokenUsage {
                input_tokens: 3,
                output_tokens: 4,
                cached_input_tokens: 0,
                reasoning_tokens: 0,
            },
        }],
        tool_calls: Vec::new(),
        errors: Vec::new(),
    };

    let remote = RemoteTurnResult::from_core("session", "turn", turn, []);
    remote.validate().expect("valid turn result");
    assert_eq!(remote.status, RemoteTurnStatus::Completed);
    assert_eq!(remote.usage.total.input_tokens, 4);
    assert_eq!(remote.usage.total.output_tokens, 6);
}

#[test]
fn remote_tool_grants_validate_explicit_surfaces_and_duplicates() {
    let grant = demo_grant("one", "tools", "search");
    grant.validate().expect("valid grant");
    assert_eq!(grant.call_path().unwrap(), "tools.search");

    let mut missing_surface = grant.clone();
    missing_surface.agent_surface = None;
    assert!(matches!(
        missing_surface.validate(),
        Err(RemoteProtocolError::MissingToolSurface { .. })
    ));

    let duplicate = demo_grant("two", "tools", "search");
    assert!(matches!(
        RemoteToolGrant::validate_all(&[grant, duplicate]),
        Err(RemoteProtocolError::DuplicateRemoteCallPath { .. })
    ));
}

#[tokio::test]
async fn remote_tool_provider_forwards_idempotency_headers_and_failures() {
    let transport = RecordingTransport {
        requests: Mutex::new(Vec::new()),
        response: RemoteToolCallResponse::Failure {
            protocol_version: REMOTE_PROTOCOL_VERSION,
            code: "failed".to_string(),
            message: "nope".to_string(),
            raw: Some(serde_json::json!({ "detail": true })),
            retry_after_ms: Some(5),
        },
    };
    let provider = RemoteToolProvider::new(vec![demo_grant("demo", "tools", "run")], transport)
        .expect("provider");
    let host = Arc::new(lash_core::testing::MockSessionManager::default());
    let sessions: Arc<dyn lash_core::plugin::SessionStateService> = host.clone();
    let session_lifecycle: Arc<dyn lash_core::plugin::SessionLifecycleService> = host.clone();
    let session_graph: Arc<dyn lash_core::plugin::SessionGraphService> = host;
    let context = lash_core::ToolContext::__for_testing(
        "session-1".to_string(),
        sessions,
        session_lifecycle,
        session_graph,
        Arc::new(lash_core::UnavailableProcessService),
        Arc::new(lash_core::InMemoryAttachmentStore::new()),
        lash_core::DirectCompletionClient::from_fn(|_, _| {
            Err(lash_core::PluginError::Session("unavailable".to_string()))
        }),
        Some("call-1".to_string()),
    );
    let result = provider
        .execute(lash_core::ToolCall {
            name: "demo",
            args: &serde_json::json!({ "x": 1 }),
            context: &context,
            progress: None,
        })
        .await;
    assert!(!result.is_success());
    let request = provider
        .transport
        .requests
        .lock()
        .expect("requests lock")
        .pop()
        .expect("request");
    assert_eq!(request.headers["x-lash-tool-call-id"], "call-1");
    assert_eq!(
        request.headers["x-lash-replay-key"],
        "lash-tool:session-1:call-1:demo"
    );
    assert_eq!(request.call_path, "tools.run");
}

#[test]
fn remote_activity_preserves_semantic_fields_and_collapses_runtime_diagnostics() {
    let output = lash_core::ToolCallOutput::success(serde_json::json!({ "ok": true }));
    let activity = lash_core::TurnActivity::new(
        lash_core::TurnActivityId::new("corr"),
        lash_core::TurnEvent::ToolCallCompleted {
            call_id: Some("call".to_string()),
            name: "demo".to_string(),
            args: serde_json::json!({ "a": 1 }),
            output,
            duration_ms: 42,
        },
    );
    let remote = RemoteTurnActivity::from_core(9, activity);
    assert_eq!(remote.sequence, 9);
    match remote.event {
        RemoteTurnEvent::ToolCallCompleted {
            call_id,
            args,
            duration_ms,
            ..
        } => {
            assert_eq!(call_id.as_deref(), Some("call"));
            assert_eq!(args, serde_json::json!({ "a": 1 }));
            assert_eq!(duration_ms, 42);
        }
        other => panic!("unexpected event: {other:?}"),
    }
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
        agent_surface: Some(RemoteToolAgentSurface::new([module], operation)),
    }
}
