use crate::support::*;
use lash::llm::types::{LlmJsonSchema, LlmMessage, LlmToolSpec};
use std::sync::Arc;

fn request(messages: Vec<LlmMessage>) -> LlmRequest {
    LlmRequest {
        model: "openai/gpt-5.4".to_string(),
        messages,
        attachments: Vec::new(),
        tools: Arc::new(Vec::<LlmToolSpec>::new()),
        tool_choice: LlmToolChoice::Auto,
        model_variant: None,
        session_id: Some("session-1".to_string()),
        output_spec: None,
        stream_events: None,
        provider_trace: None,
    }
}

#[test]
fn builds_responses_body_with_instructions_and_input() {
    let provider = OpenAiGenericProvider::new("key", OPENROUTER_BASE_URL);
    let req = request(vec![
        LlmMessage::text(LlmRole::System, "system prompt"),
        LlmMessage::text(LlmRole::User, "hello"),
    ]);
    let body = provider.build_responses_request_body(&req, true).unwrap();
    assert_eq!(body["instructions"], "system prompt");
    assert_eq!(body["stream"], true);
    assert!(body.get("messages").is_none());
    assert!(body.get("cache_control").is_none());
    assert_eq!(body["prompt_cache_key"], "session-1");
    assert_eq!(body["include"], json!(["reasoning.encrypted_content"]));
    assert_eq!(body["input"][0]["role"], "user");
    assert_eq!(body["input"][0]["content"][0]["type"], "input_text");
}

#[test]
fn responses_body_does_not_request_reasoning_summaries_by_default() {
    let provider = OpenAiGenericProvider::new("key", OPENROUTER_BASE_URL);
    let mut req = request(vec![LlmMessage::text(LlmRole::User, "hello")]);
    req.model = "openai/gpt-5.5".to_string();
    req.model_variant = Some("medium".to_string());

    let body = provider.build_responses_request_body(&req, true).unwrap();

    assert_eq!(body["reasoning"], json!({ "effort": "medium" }));
}

#[test]
fn responses_body_requests_reasoning_summaries_when_provider_exposes_thinking() {
    let provider =
        OpenAiGenericProvider::new("key", OPENROUTER_BASE_URL).with_options(ProviderOptions {
            thinking: lash::provider::ProviderThinkingPolicy { expose: true },
            ..ProviderOptions::default()
        });
    let mut req = request(vec![LlmMessage::text(LlmRole::User, "hello")]);
    req.model = "openai/gpt-5.5".to_string();
    req.model_variant = Some("medium".to_string());

    let body = provider.build_responses_request_body(&req, true).unwrap();

    assert_eq!(body["reasoning"]["summary"], "auto");
}

#[test]
fn wire_api_auto_routes_only_openrouter_claude_to_chat_completions() {
    let provider = OpenAiGenericProvider::new("key", OPENROUTER_BASE_URL);
    let mut claude = request(vec![LlmMessage::text(LlmRole::User, "hello")]);
    claude.model = "anthropic/claude-sonnet-4.6".to_string();
    assert_eq!(
        provider.resolved_wire_api(&claude),
        OpenAiWireApi::ChatCompletions
    );

    let mut gpt = request(vec![LlmMessage::text(LlmRole::User, "hello")]);
    gpt.model = "openai/gpt-5.4".to_string();
    assert_eq!(provider.resolved_wire_api(&gpt), OpenAiWireApi::Responses);

    let local = OpenAiGenericProvider::new("key", "http://localhost:11434/v1");
    assert_eq!(local.resolved_wire_api(&claude), OpenAiWireApi::Responses);

    let forced = local.with_wire_api(OpenAiWireApi::ChatCompletions);
    assert_eq!(
        forced.resolved_wire_api(&gpt),
        OpenAiWireApi::ChatCompletions
    );
}

#[test]
fn chat_body_uses_messages_and_not_responses_input() {
    let mut req = request(vec![
        LlmMessage::text(LlmRole::System, "system prompt"),
        LlmMessage::text(LlmRole::User, "hello"),
    ]);
    req.model = "anthropic/claude-sonnet-4.6".to_string();
    req.model_variant = Some("high".to_string());
    req.output_spec = Some(LlmOutputSpec::JsonObject);

    let body = OpenAiGenericProvider::new("key", OPENROUTER_BASE_URL)
        .build_chat_request_body(&req, true)
        .unwrap();

    assert!(body.get("input").is_none());
    assert!(body.get("instructions").is_none());
    assert_eq!(body["messages"][0]["role"], "system");
    assert_eq!(body["messages"][1]["role"], "user");
    assert_eq!(body["stream_options"], json!({ "include_usage": true }));
    assert_eq!(body["reasoning"], json!({ "effort": "high" }));
    assert_eq!(body["response_format"], json!({ "type": "json_object" }));
}

#[test]
fn openrouter_claude_chat_body_marks_anthropic_cache_breakpoints() {
    let mut req = request(vec![
        LlmMessage::text(LlmRole::System, "stable system prompt"),
        LlmMessage::text(LlmRole::User, "dynamic tail"),
    ]);
    req.model = "anthropic/claude-sonnet-4.6".to_string();
    req.tools = Arc::new(vec![LlmToolSpec {
        name: "search".to_string(),
        description: "Search".to_string(),
        input_schema: json!({"type": "object"}),
        output_schema: json!({}),
        input_schema_projections: Vec::new(),
        output_schema_projections: Vec::new(),
    }]);

    let body = OpenAiGenericProvider::new("key", OPENROUTER_BASE_URL)
        .build_chat_request_body(&req, true)
        .unwrap();

    assert_eq!(
        body["messages"][0]["content"][0]["cache_control"],
        json!({ "type": "ephemeral" })
    );
    assert_eq!(
        body["tools"][0]["cache_control"],
        json!({ "type": "ephemeral" })
    );
    assert_eq!(
        body["messages"][1]["content"][0]["cache_control"],
        json!({ "type": "ephemeral" })
    );
}

#[test]
fn chat_tools_use_projected_openai_schema_and_preserve_override() {
    let mut req = request(vec![LlmMessage::text(LlmRole::User, "hello")]);
    req.model = "anthropic/claude-sonnet-4.6".to_string();
    req.tools = Arc::new(vec![
        LlmToolSpec {
            name: "empty".to_string(),
            description: "Empty".to_string(),
            input_schema: json!({"type": "object"}),
            output_schema: json!({}),
            input_schema_projections: Vec::new(),
            output_schema_projections: Vec::new(),
        },
        LlmToolSpec {
            name: "override".to_string(),
            description: "Override".to_string(),
            input_schema: json!({"type": "object", "properties": {"raw": {"const": "x"}}}),
            output_schema: json!({}),
            input_schema_projections: vec![lash::SchemaProjectionOverride {
                profile: OpenAiSchemaProfile::ToolParameters
                    .projection_id()
                    .to_string(),
                schema: json!({
                    "type": "object",
                    "properties": { "raw": { "type": "string", "enum": ["x"] } }
                }),
            }],
            output_schema_projections: Vec::new(),
        },
    ]);

    let body = OpenAiGenericProvider::new("key", OPENROUTER_BASE_URL)
        .build_chat_request_body(&req, true)
        .unwrap();

    assert_eq!(
        body["tools"][0]["function"]["parameters"]["properties"],
        json!({})
    );
    assert_eq!(
        body["tools"][1]["function"]["parameters"]["properties"]["raw"],
        json!({ "type": "string", "enum": ["x"] })
    );
}

#[test]
fn structured_output_schema_is_projected_or_rejected_locally() {
    let mut req = request(vec![LlmMessage::text(LlmRole::User, "hello")]);
    req.output_spec = Some(LlmOutputSpec::JsonSchema(LlmJsonSchema {
        name: "result".to_string(),
        schema: json!({
            "type": "object",
            "properties": { "summary": { "type": "string" } }
        }),
        strict: true,
    }));

    let body = OpenAiGenericProvider::new("key", OPENROUTER_BASE_URL)
        .build_responses_request_body(&req, false)
        .unwrap();
    assert_eq!(
        body["text"]["format"]["schema"]["required"],
        json!(["summary"])
    );
    assert_eq!(
        body["text"]["format"]["schema"]["additionalProperties"],
        false
    );

    req.output_spec = Some(LlmOutputSpec::JsonSchema(LlmJsonSchema {
        name: "bad".to_string(),
        schema: json!({"type": "object", "allOf": []}),
        strict: true,
    }));
    let err = OpenAiGenericProvider::new("key", OPENROUTER_BASE_URL)
        .build_responses_request_body(&req, false)
        .unwrap_err();
    assert_eq!(err.kind, ProviderFailureKind::Validation);
    assert!(err.message.contains("allOf"));
}

#[test]
fn openrouter_claude_chat_body_prefers_explicit_text_cache_breakpoint() {
    let mut req = request(vec![
        LlmMessage::text(LlmRole::System, "stable system prompt"),
        LlmMessage::new(
            LlmRole::User,
            vec![
                LlmContentBlock::Text {
                    text: "stable history".into(),
                    response_meta: None,
                    cache_breakpoint: true,
                },
                LlmContentBlock::Text {
                    text: "dynamic current iteration".into(),
                    response_meta: None,
                    cache_breakpoint: false,
                },
            ],
        ),
    ]);
    req.model = "anthropic/claude-sonnet-4.6".to_string();

    let body = OpenAiGenericProvider::new("key", OPENROUTER_BASE_URL)
        .build_chat_request_body(&req, true)
        .unwrap();

    assert_eq!(
        body["messages"][1]["content"][0]["cache_control"],
        json!({ "type": "ephemeral" })
    );
    assert!(
        body["messages"][1]["content"][1]
            .get("cache_control")
            .is_none()
    );
    assert!(
        body["messages"][1]["content"][0]
            .get("__lash_cache_breakpoint")
            .is_none()
    );
}

#[test]
fn cache_retention_none_removes_chat_cache_markers() {
    let mut req = request(vec![
        LlmMessage::text(LlmRole::System, "stable system prompt"),
        LlmMessage::text(LlmRole::User, "dynamic tail"),
    ]);
    req.model = "anthropic/claude-sonnet-4.6".to_string();

    let body = OpenAiGenericProvider::new("key", OPENROUTER_BASE_URL)
        .with_cache_retention(OpenAiCacheRetention::None)
        .build_chat_request_body(&req, true)
        .unwrap();

    assert!(body["messages"][0]["content"].is_string());
    assert!(body["messages"][1]["content"].is_string());
    assert!(body.get("tools").is_none());
}

#[test]
fn cache_retention_long_uses_anthropic_ttl_on_chat_cache_markers() {
    let mut req = request(vec![
        LlmMessage::text(LlmRole::System, "stable system prompt"),
        LlmMessage::text(LlmRole::User, "dynamic tail"),
    ]);
    req.model = "anthropic/claude-sonnet-4.6".to_string();

    let body = OpenAiGenericProvider::new("key", OPENROUTER_BASE_URL)
        .with_cache_retention(OpenAiCacheRetention::Long)
        .build_chat_request_body(&req, true)
        .unwrap();

    assert_eq!(
        body["messages"][0]["content"][0]["cache_control"],
        json!({ "type": "ephemeral", "ttl": "1h" })
    );
}

#[test]
fn responses_long_cache_retention_emits_openai_retention() {
    let provider = OpenAiGenericProvider::new("key", OPENROUTER_BASE_URL)
        .with_cache_retention(OpenAiCacheRetention::Long);
    let req = request(vec![LlmMessage::text(LlmRole::User, "hello")]);

    let body = provider.build_responses_request_body(&req, true).unwrap();

    assert_eq!(body["prompt_cache_key"], "session-1");
    assert_eq!(body["prompt_cache_retention"], "24h");
    assert!(body.get("cache_control").is_none());
}

#[test]
fn assistant_text_preserves_response_meta() {
    let provider = OpenAiGenericProvider::new("key", OPENROUTER_BASE_URL);
    let req = request(vec![LlmMessage::new(
        LlmRole::Assistant,
        vec![LlmContentBlock::Text {
            text: "done".into(),
            response_meta: Some(ResponseTextMeta {
                id: Some("msg_1".to_string()),
                status: Some("completed".to_string()),
                phase: Some(ResponseTextPhase::FinalAnswer),
            }),
            cache_breakpoint: false,
        }],
    )]);
    let body = provider.build_responses_request_body(&req, false).unwrap();
    assert_eq!(body["input"][0]["type"], "message");
    assert_eq!(body["input"][0]["id"], "msg_1");
    assert_eq!(body["input"][0]["phase"], "final_answer");
}

#[test]
fn legacy_assistant_text_gets_deterministic_id() {
    let provider = OpenAiGenericProvider::new("key", OPENROUTER_BASE_URL);
    let req = request(vec![LlmMessage::text(LlmRole::Assistant, "legacy")]);
    let body = provider.build_responses_request_body(&req, false).unwrap();
    assert_eq!(body["input"][0]["id"], "msg_lash_0_0");
    assert_eq!(body["input"][0]["status"], "completed");
    assert!(body["input"][0].get("phase").is_none());
}

#[test]
fn local_base_url_omits_openai_only_fields() {
    let provider = OpenAiGenericProvider::new("key", "http://localhost:11434/v1");
    let req = request(vec![LlmMessage::text(LlmRole::User, "hello")]);
    let body = provider.build_responses_request_body(&req, true).unwrap();
    assert!(body.get("include").is_none());
    assert!(body.get("store").is_none());
    assert!(body.get("parallel_tool_calls").is_none());
    assert!(body.get("text").is_none());
}

#[test]
fn usage_parser_accepts_responses_and_chat_completion_shapes() {
    let responses_usage = OpenAiGenericProvider::usage_from_response_value(&serde_json::json!({
        "usage": {
            "input_tokens": 11,
            "output_tokens": 7,
            "input_tokens_details": { "cached_tokens": 3 },
            "output_tokens_details": { "reasoning_tokens": 5 }
        }
    }));
    assert_eq!(responses_usage.input_tokens, 11);
    assert_eq!(responses_usage.output_tokens, 7);
    assert_eq!(responses_usage.cached_input_tokens, 3);
    assert_eq!(responses_usage.reasoning_tokens, 5);

    let chat_usage = OpenAiGenericProvider::usage_from_response_value(&serde_json::json!({
        "usage": {
            "prompt_tokens": 13,
            "completion_tokens": 17,
            "prompt_tokens_details": { "cached_tokens": 7, "cache_write_tokens": 5 },
            "completion_tokens_details": { "reasoning_tokens": 4 }
        }
    }));
    assert_eq!(chat_usage.input_tokens, 13);
    assert_eq!(chat_usage.output_tokens, 17);
    assert_eq!(chat_usage.cached_input_tokens, 2);
    assert_eq!(chat_usage.reasoning_tokens, 4);

    let write_only_usage = OpenAiGenericProvider::usage_from_response_value(&serde_json::json!({
        "usage": {
            "prompt_tokens": 5353,
            "completion_tokens": 433,
            "prompt_tokens_details": { "cached_tokens": 3, "cache_write_tokens": 5353 }
        }
    }));
    assert_eq!(write_only_usage.cached_input_tokens, 0);
}

#[test]
fn chat_body_replays_openrouter_reasoning_details_on_tool_calls() {
    let req = request(vec![LlmMessage::new(
        LlmRole::Assistant,
        vec![LlmContentBlock::ToolCall {
            call_id: "call_1".to_string(),
            tool_name: "lookup".to_string(),
            input_json: "{\"q\":\"x\"}".to_string(),
            item_id: None,
            signature: Some(
                json!({
                    "type": "reasoning.encrypted",
                    "id": "call_1",
                    "data": "encrypted"
                })
                .to_string(),
            ),
        }],
    )]);

    let body = OpenAiGenericProvider::new("key", OPENROUTER_BASE_URL)
        .build_chat_request_body(&req, false)
        .unwrap();

    assert_eq!(
        body["messages"][0]["reasoning_details"][0],
        json!({
            "type": "reasoning.encrypted",
            "id": "call_1",
            "data": "encrypted"
        })
    );
}

#[test]
fn non_streaming_chat_parser_captures_text_tool_and_usage() {
    let value = json!({
        "choices": [{
            "message": {
                "role": "assistant",
                "reasoning_content": "think",
                "content": "hello",
                "reasoning_details": [{
                    "type": "reasoning.encrypted",
                    "id": "call_1",
                    "data": "encrypted"
                }],
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": "{\"q\":\"x\"}"
                    }
                }]
            }
        }],
        "usage": {
            "prompt_tokens": 13,
            "completion_tokens": 17,
            "prompt_tokens_details": { "cached_tokens": 2 }
        }
    });

    let parts = OpenAiGenericProvider::chat_response_parts_from_value(&value);
    let usage = OpenAiGenericProvider::usage_from_response_value(&value);

    assert!(matches!(&parts[0], LlmOutputPart::Reasoning { text, .. } if text == "think"));
    assert!(matches!(&parts[1], LlmOutputPart::Text { text, .. } if text == "hello"));
    assert!(matches!(
        &parts[2],
        LlmOutputPart::ToolCall {
            call_id,
            tool_name,
            input_json,
            signature,
            ..
        } if call_id == "call_1"
            && tool_name == "lookup"
            && input_json == "{\"q\":\"x\"}"
            && signature.as_deref().is_some_and(|value| value.contains("encrypted"))
    ));
    assert_eq!(usage.input_tokens, 13);
    assert_eq!(usage.output_tokens, 17);
    assert_eq!(usage.cached_input_tokens, 2);
}

#[test]
fn chat_stream_parser_captures_text_tool_done_and_usage() {
    let mut state = ChatStreamState::default();
    OpenAiGenericProvider::process_chat_sse_event(
        r#"{"choices":[{"delta":{"content":"Hi"}}]}"#,
        &mut state,
    )
    .unwrap();
    OpenAiGenericProvider::process_chat_sse_event(
        r#"{"choices":[{"delta":{"reasoning_content":"Think"}}]}"#,
        &mut state,
    )
    .unwrap();
    OpenAiGenericProvider::process_chat_sse_event(
        r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{\"q\":"}}]}}]}"#,
        &mut state,
    )
    .unwrap();
    OpenAiGenericProvider::process_chat_sse_event(
        r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"x\"}"}}]}}]}"#,
        &mut state,
    )
    .unwrap();
    OpenAiGenericProvider::process_chat_sse_event(
        r#"{"choices":[{"delta":{"reasoning_details":[{"type":"reasoning.encrypted","id":"call_1","data":"encrypted"}]}}],"usage":{"prompt_tokens":9,"completion_tokens":4,"prompt_tokens_details":{"cached_tokens":3,"cache_write_tokens":2}}}"#,
        &mut state,
    )
    .unwrap();
    OpenAiGenericProvider::process_chat_sse_event(
        r#"{"choices":[{"usage":{"prompt_tokens":11,"completion_tokens":5,"prompt_tokens_details":{"cached_tokens":4,"cache_write_tokens":2}}}]}"#,
        &mut state,
    )
    .unwrap();
    OpenAiGenericProvider::process_chat_sse_event("[DONE]", &mut state).unwrap();

    assert_eq!(state.full_text, "Hi");
    assert_eq!(state.reasoning_text, "Think");
    let parts = state.parts();
    assert!(matches!(&parts[0], LlmOutputPart::Reasoning { text, .. } if text == "Think"));
    assert!(matches!(&parts[1], LlmOutputPart::Text { text, .. } if text == "Hi"));
    assert!(matches!(
        &parts[2],
        LlmOutputPart::ToolCall {
            call_id,
            tool_name,
            input_json,
            signature,
            ..
        } if call_id == "call_1"
            && tool_name == "lookup"
            && input_json == "{\"q\":\"x\"}"
            && signature.as_deref().is_some_and(|value| value.contains("encrypted"))
    ));
    assert_eq!(state.usage.input_tokens, 11);
    assert_eq!(state.usage.output_tokens, 5);
    assert_eq!(state.usage.cached_input_tokens, 2);
}

#[test]
fn stream_parser_captures_text_reasoning_tool_and_phase() {
    let mut state = ResponsesStreamState::default();
    OpenAiGenericProvider::process_sse_event(
        r#"{"type":"response.output_item.added","item":{"type":"reasoning","id":"rs_1"}}"#,
        &mut state,
        None,
    )
    .unwrap();
    OpenAiGenericProvider::process_sse_event(
        r#"{"type":"response.reasoning_summary_text.delta","delta":"Think"}"#,
        &mut state,
        None,
    )
    .unwrap();
    OpenAiGenericProvider::process_sse_event(
        r#"{"type":"response.output_item.done","item":{"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text","text":"Think"}],"encrypted_content":"enc"}}"#,
        &mut state,
        None,
    )
    .unwrap();
    OpenAiGenericProvider::process_sse_event(
        r#"{"type":"response.output_item.added","item":{"type":"message","id":"msg_1"}}"#,
        &mut state,
        None,
    )
    .unwrap();
    OpenAiGenericProvider::process_sse_event(
        r#"{"type":"response.output_text.delta","delta":"Hi"}"#,
        &mut state,
        None,
    )
    .unwrap();
    OpenAiGenericProvider::process_sse_event(
        r#"{"type":"response.output_item.done","item":{"type":"message","id":"msg_1","status":"completed","phase":"commentary","content":[{"type":"output_text","text":"Hi"}]}}"#,
        &mut state,
        None,
    )
    .unwrap();
    OpenAiGenericProvider::process_sse_event(
        r#"{"type":"response.output_item.added","item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"tool","arguments":""}}"#,
        &mut state,
        None,
    )
    .unwrap();
    OpenAiGenericProvider::process_sse_event(
        r#"{"type":"response.function_call_arguments.delta","item_id":"fc_1","delta":"{\"x\":"}"#,
        &mut state,
        None,
    )
    .unwrap();
    OpenAiGenericProvider::process_sse_event(
        r#"{"type":"response.function_call_arguments.done","item_id":"fc_1","arguments":"{\"x\":1}" }"#,
        &mut state,
        None,
    )
    .unwrap();
    OpenAiGenericProvider::process_sse_event(
        r#"{"type":"response.output_item.done","item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"tool","arguments":"{\"x\":1}","status":"completed"}}"#,
        &mut state,
        None,
    )
    .unwrap();

    assert_eq!(state.full_text, "Hi");
    let parts = state.response_parts();
    assert!(matches!(
        &parts[0],
        LlmOutputPart::Reasoning {
            item_id: Some(id),
            encrypted_content: Some(blob),
            ..
        } if id == "rs_1" && blob == "enc"
    ));
    assert!(matches!(
        &parts[1],
        LlmOutputPart::Text {
            response_meta: Some(ResponseTextMeta {
                phase: Some(ResponseTextPhase::Commentary),
                ..
            }),
            ..
        }
    ));
    assert!(matches!(
        &parts[2],
        LlmOutputPart::ToolCall {
            item_id: Some(id),
            input_json,
            ..
        } if id == "fc_1" && input_json == "{\"x\":1}"
    ));
}

#[test]
fn response_failed_server_error_is_retryable() {
    let mut state = ResponsesStreamState::default();
    let err = OpenAiGenericProvider::process_sse_event(
        r#"{"type":"response.failed","response":{"status":"failed","error":{"code":"server_error","message":"internal stream ended unexpectedly"}}}"#,
        &mut state,
        None,
    )
    .unwrap_err();

    assert!(err.retryable);
    assert_eq!(err.message, "internal stream ended unexpectedly");
}
