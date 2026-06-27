use crate::support::*;
use crate::{OpenAiCompatibleProviderFactory, OpenAiProviderFactory};
use lash_core::llm::transport::ProviderFailureKind;
use lash_core::llm::types::{LlmJsonSchema, LlmMessage, LlmToolChoice, LlmToolSpec};
use lash_core::provider::CacheRetention;
use std::num::NonZeroUsize;
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
        generation: lash_core::GenerationOptions::default(),
        provider_trace: None,
    }
}

#[test]
fn chat_image_attachment_serializes_as_data_url() {
    let provider = OpenAiCompatibleProvider::new("key", OPENROUTER_BASE_URL);
    let png_bytes = vec![0x89, 0x50, 0x4E, 0x47];
    let mut req = request(vec![LlmMessage::new(
        LlmRole::User,
        vec![
            LlmContentBlock::Text {
                text: "look".into(),
                response_meta: None,
                cache_breakpoint: false,
            },
            LlmContentBlock::Image { attachment_idx: 0 },
        ],
    )]);
    req.attachments = vec![LlmAttachment::bytes("image/png", png_bytes.clone())];

    let body = provider.build_chat_request_body(&req, false).unwrap();

    let messages = body["messages"].as_array().expect("messages");
    let user_msg = messages.last().expect("user message");
    let content = user_msg["content"].as_array().expect("content array");
    let image_part = content
        .iter()
        .find(|part| part["type"] == "image_url")
        .expect("image_url part");
    let expected_b64 = base64::engine::general_purpose::STANDARD.encode(&png_bytes);
    assert_eq!(
        image_part["image_url"]["url"],
        format!("data:image/png;base64,{expected_b64}")
    );
}

#[test]
fn chat_unsupported_image_mime_is_rejected_at_request_boundary() {
    let provider = OpenAiCompatibleProvider::new("key", OPENROUTER_BASE_URL);
    let mut req = request(vec![LlmMessage::new(
        LlmRole::User,
        vec![LlmContentBlock::Image { attachment_idx: 0 }],
    )]);
    req.attachments = vec![LlmAttachment::bytes("image/bmp", vec![0x42, 0x4D])];

    let err = provider
        .build_chat_request_body(&req, false)
        .expect_err("bmp should be rejected before wire");

    assert_eq!(err.code.as_deref(), Some("unsupported_image_format"));
    assert!(err.message.contains("OpenAI"));
    assert!(err.message.contains("image/bmp"));
}

#[test]
fn responses_image_attachment_serializes_as_input_image_data_url() {
    let provider = OpenAiProvider::new("key");
    let png_bytes = vec![0x89, 0x50, 0x4E, 0x47];
    let mut req = request(vec![LlmMessage::new(
        LlmRole::User,
        vec![
            LlmContentBlock::Text {
                text: "look".into(),
                response_meta: None,
                cache_breakpoint: false,
            },
            LlmContentBlock::Image { attachment_idx: 0 },
        ],
    )]);
    req.attachments = vec![LlmAttachment::bytes("image/png", png_bytes.clone())];

    let body = provider.build_responses_request_body(&req, false).unwrap();

    let input = body["input"].as_array().expect("input array");
    let user_msg = input.last().expect("user message");
    let content = user_msg["content"].as_array().expect("content array");
    let image_part = content
        .iter()
        .find(|part| part["type"] == "input_image")
        .expect("input_image part");
    let expected_b64 = base64::engine::general_purpose::STANDARD.encode(&png_bytes);
    assert_eq!(
        image_part["image_url"],
        format!("data:image/png;base64,{expected_b64}")
    );
}

#[test]
fn responses_unsupported_image_mime_is_rejected_at_request_boundary() {
    let provider = OpenAiProvider::new("key");
    let mut req = request(vec![LlmMessage::new(
        LlmRole::User,
        vec![LlmContentBlock::Image { attachment_idx: 0 }],
    )]);
    req.attachments = vec![LlmAttachment::bytes("image/bmp", vec![0x42, 0x4D])];

    let err = provider
        .build_responses_request_body(&req, false)
        .expect_err("bmp should be rejected before wire");

    assert_eq!(err.code.as_deref(), Some("unsupported_image_format"));
    assert!(err.message.contains("OpenAI"));
}

#[test]
fn builds_responses_body_with_instructions_and_input() {
    let provider = OpenAiProvider::new("key");
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
    let provider = OpenAiProvider::new("key");
    let mut req = request(vec![LlmMessage::text(LlmRole::User, "hello")]);
    req.model = "openai/gpt-5.5".to_string();
    req.model_variant = Some("medium".to_string());

    let body = provider.build_responses_request_body(&req, true).unwrap();

    assert_eq!(body["reasoning"], json!({ "effort": "medium" }));
}

#[test]
fn responses_body_requests_reasoning_summaries_when_provider_exposes_thinking() {
    let provider = OpenAiProvider::new("key").with_options(ProviderOptions {
        expose_thinking: true,
        ..ProviderOptions::default()
    });
    let mut req = request(vec![LlmMessage::text(LlmRole::User, "hello")]);
    req.model = "openai/gpt-5.5".to_string();
    req.model_variant = Some("medium".to_string());

    let body = provider.build_responses_request_body(&req, true).unwrap();

    assert_eq!(body["reasoning"]["summary"], "auto");
}

#[test]
fn providers_serialize_distinct_config_shapes() {
    let direct = OpenAiProvider::new("key");
    let direct_config = direct.serialize_config();
    assert_eq!(direct.kind(), "openai");
    assert_eq!(direct_config["api_key"], "key");
    assert!(direct_config.get("base_url").is_none());
    assert!(direct_config.get("wire_api").is_none());

    let compatible = OpenAiCompatibleProvider::new("key", OPENROUTER_BASE_URL);
    let compatible_config = compatible.serialize_config();
    assert_eq!(compatible.kind(), "openai-compatible");
    assert_eq!(compatible_config["api_key"], "key");
    assert_eq!(compatible_config["base_url"], OPENROUTER_BASE_URL);
    assert!(compatible_config.get("wire_api").is_none());
}

#[test]
fn provider_factories_reject_stale_wire_api_configs() {
    let direct_err = OpenAiProviderFactory
        .deserialize(json!({
            "api_key": "key",
            "wire_api": "responses"
        }))
        .expect_err("direct OpenAI rejects wire_api");
    assert!(direct_err.contains("wire_api"));

    let compatible_err = OpenAiCompatibleProviderFactory
        .deserialize(json!({
            "api_key": "key",
            "base_url": OPENROUTER_BASE_URL,
            "wire_api": "chat-completions"
        }))
        .expect_err("compatible OpenAI rejects wire_api");
    assert!(compatible_err.contains("wire_api"));
}

#[test]
fn provider_factories_materialize_distinct_kinds() {
    let direct = OpenAiProviderFactory
        .deserialize(json!({ "api_key": "key" }))
        .expect("direct config");
    assert_eq!(direct.provider.kind(), "openai");

    let compatible = OpenAiCompatibleProviderFactory
        .deserialize(json!({
            "api_key": "key",
            "base_url": OPENROUTER_BASE_URL
        }))
        .expect("compatible config");
    assert_eq!(compatible.provider.kind(), "openai-compatible");
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

    let body = OpenAiCompatibleProvider::new("key", OPENROUTER_BASE_URL)
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

    let body = OpenAiCompatibleProvider::new("key", OPENROUTER_BASE_URL)
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
            input_schema_projections: vec![lash_core::SchemaProjectionOverride {
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
        LlmToolSpec {
            name: "schemars".to_string(),
            description: "Schemars".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "limit": {
                        "description": "Maximum number of results.",
                        "allOf": [
                            {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 100
                            }
                        ]
                    }
                }
            }),
            output_schema: json!({}),
            input_schema_projections: Vec::new(),
            output_schema_projections: Vec::new(),
        },
    ]);

    let body = OpenAiCompatibleProvider::new("key", OPENROUTER_BASE_URL)
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
    assert_eq!(
        body["tools"][2]["function"]["parameters"]["properties"]["limit"],
        json!({
            "description": "Maximum number of results.",
            "type": "integer",
            "minimum": 1,
            "maximum": 100
        })
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

    let body = OpenAiProvider::new("key")
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
    let err = OpenAiProvider::new("key")
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

    let body = OpenAiCompatibleProvider::new("key", OPENROUTER_BASE_URL)
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

    let body = OpenAiCompatibleProvider::new("key", OPENROUTER_BASE_URL)
        .with_options(ProviderOptions {
            cache_retention: CacheRetention::None,
            ..ProviderOptions::default()
        })
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

    let body = OpenAiCompatibleProvider::new("key", OPENROUTER_BASE_URL)
        .with_options(ProviderOptions {
            cache_retention: CacheRetention::Long,
            ..ProviderOptions::default()
        })
        .build_chat_request_body(&req, true)
        .unwrap();

    assert_eq!(
        body["messages"][0]["content"][0]["cache_control"],
        json!({ "type": "ephemeral", "ttl": "1h" })
    );
}

#[test]
fn responses_long_cache_retention_emits_openai_retention() {
    let provider = OpenAiProvider::new("key").with_options(ProviderOptions {
        cache_retention: CacheRetention::Long,
        ..ProviderOptions::default()
    });
    let req = request(vec![LlmMessage::text(LlmRole::User, "hello")]);

    let body = provider.build_responses_request_body(&req, true).unwrap();

    assert_eq!(body["prompt_cache_key"], "session-1");
    assert_eq!(body["prompt_cache_retention"], "24h");
    assert!(body.get("cache_control").is_none());
}

#[test]
fn responses_none_cache_retention_omits_prompt_cache_fields() {
    let provider = OpenAiProvider::new("key").with_options(ProviderOptions {
        cache_retention: CacheRetention::None,
        ..ProviderOptions::default()
    });
    let req = request(vec![LlmMessage::text(LlmRole::User, "hello")]);

    let body = provider.build_responses_request_body(&req, true).unwrap();

    assert!(body.get("prompt_cache_key").is_none());
    assert!(body.get("prompt_cache_retention").is_none());
}

#[test]
fn openai_compat_config_serializes_when_non_default() {
    let provider =
        OpenAiCompatibleProvider::new("key", OPENROUTER_BASE_URL).with_compat(OpenAiCompat {
            max_tokens_field: Some(OpenAiCompatMaxTokensField::MaxCompletionTokens),
            streaming_usage: Some(false),
            ..OpenAiCompat::default()
        });

    let config = provider.serialize_config();

    assert_eq!(
        config["compat"]["max_tokens_field"],
        json!("max_completion_tokens")
    );
    assert_eq!(config["compat"]["streaming_usage"], false);
}

#[test]
fn openai_compat_resolver_covers_openrouter_local_and_session_affinity() {
    let openrouter = OpenAiCompatibleProvider::new("key", OPENROUTER_BASE_URL);
    let openrouter_caps = openrouter.resolved_compat(CompletionEndpoint::ChatCompletions);
    assert_eq!(
        openrouter_caps.reasoning_format,
        OpenAiCompatReasoningFormat::OpenRouter
    );
    assert!(openrouter_caps.streaming_usage);
    assert!(openrouter_caps.cache_session_affinity);

    let local = OpenAiCompatibleProvider::new("key", "http://localhost:11434/v1");
    let local_caps = local.resolved_compat(CompletionEndpoint::ChatCompletions);
    assert_eq!(
        local_caps.reasoning_format,
        OpenAiCompatReasoningFormat::None
    );
    assert!(!local_caps.request_fields);
    assert!(!local_caps.streaming_usage);
    assert!(!local_caps.cache_session_affinity);
}

#[test]
fn chat_body_honors_compat_max_token_field_streaming_usage_and_strict_tools() {
    let mut req = request(vec![LlmMessage::text(LlmRole::User, "hello")]);
    req.tools = Arc::new(vec![LlmToolSpec {
        name: "lookup".to_string(),
        description: "Lookup".to_string(),
        input_schema: json!({
            "type": "object",
            "properties": { "q": { "type": "string" } }
        }),
        output_schema: json!({}),
        input_schema_projections: Vec::new(),
        output_schema_projections: Vec::new(),
    }]);

    let body = OpenAiCompatibleProvider::new("key", OPENROUTER_BASE_URL)
        .with_compat(OpenAiCompat {
            max_tokens_field: Some(OpenAiCompatMaxTokensField::MaxCompletionTokens),
            streaming_usage: Some(false),
            strict_tools: Some(true),
            ..OpenAiCompat::default()
        })
        .build_chat_request_body(&req, true)
        .unwrap();

    assert!(body.get("max_tokens").is_none());
    assert_eq!(body["max_completion_tokens"], DEFAULT_MAX_OUTPUT_TOKENS);
    assert!(body.get("stream_options").is_none());
    assert_eq!(body["tools"][0]["function"]["strict"], true);
    assert_eq!(
        body["tools"][0]["function"]["parameters"]["additionalProperties"],
        false
    );
}

#[test]
fn local_openai_compatible_suppresses_optional_openai_fields() {
    let mut req = request(vec![LlmMessage::text(LlmRole::User, "hello")]);
    req.model_variant = Some("medium".to_string());
    req.tools = Arc::new(vec![LlmToolSpec {
        name: "lookup".to_string(),
        description: "Lookup".to_string(),
        input_schema: json!({"type": "object"}),
        output_schema: json!({}),
        input_schema_projections: Vec::new(),
        output_schema_projections: Vec::new(),
    }]);

    let local = OpenAiCompatibleProvider::new("key", "http://localhost:11434/v1");
    let chat = local.build_chat_request_body(&req, true).unwrap();
    assert!(chat.get("stream_options").is_none());
    assert!(chat.get("parallel_tool_calls").is_none());
    assert!(chat.get("reasoning").is_none());

    let responses = local.build_responses_request_body(&req, true).unwrap();
    assert!(responses.get("include").is_none());
    assert!(responses.get("store").is_none());
    assert!(responses.get("text").is_none());
    assert!(responses.get("prompt_cache_key").is_none());
}

#[test]
fn output_token_cap_maps_to_wire_fields() {
    let options = ProviderOptions {
        max_output_tokens: Some(9999),
        ..ProviderOptions::default()
    };
    let mut req = request(vec![LlmMessage::text(LlmRole::User, "hello")]);
    req.generation.output_token_cap = NonZeroUsize::new(2048);

    let responses_body = OpenAiProvider::new("key")
        .with_options(options.clone())
        .build_responses_request_body(&req, true)
        .unwrap();
    assert_eq!(responses_body["max_output_tokens"], 2048);
    let provider_limited_responses_body = OpenAiProvider::new("key")
        .with_options(options.clone())
        .build_responses_request_body(
            &request(vec![LlmMessage::text(LlmRole::User, "hello")]),
            true,
        )
        .unwrap();
    assert_eq!(provider_limited_responses_body["max_output_tokens"], 9999);

    let mut chat_req = req;
    chat_req.model = "anthropic/claude-sonnet-4.6".to_string();
    let chat_body = OpenAiCompatibleProvider::new("key", OPENROUTER_BASE_URL)
        .with_options(options.clone())
        .build_chat_request_body(&chat_req, true)
        .unwrap();
    assert_eq!(chat_body["max_tokens"], 2048);
    let mut provider_limited_chat_req = request(vec![LlmMessage::text(LlmRole::User, "hello")]);
    provider_limited_chat_req.model = "anthropic/claude-sonnet-4.6".to_string();
    let provider_limited_chat_body = OpenAiCompatibleProvider::new("key", OPENROUTER_BASE_URL)
        .with_options(options)
        .build_chat_request_body(&provider_limited_chat_req, true)
        .unwrap();
    assert_eq!(provider_limited_chat_body["max_tokens"], 9999);
}

#[test]
fn assistant_text_preserves_response_meta() {
    let provider = OpenAiProvider::new("key");
    let req = request(vec![LlmMessage::new(
        LlmRole::Assistant,
        vec![LlmContentBlock::Text {
            text: "done".into(),
            response_meta: Some(ResponseTextMeta {
                id: Some("msg_1".to_string()),
                status: Some("completed".to_string()),
                phase: Some("final_answer".to_string()),
                ..ResponseTextMeta::default()
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
    let provider = OpenAiProvider::new("key");
    let req = request(vec![LlmMessage::text(LlmRole::Assistant, "legacy")]);
    let body = provider.build_responses_request_body(&req, false).unwrap();
    assert_eq!(body["input"][0]["id"], "msg_lash_0_0");
    assert_eq!(body["input"][0]["status"], "completed");
    assert!(body["input"][0].get("phase").is_none());
}

#[test]
fn chat_body_replays_openrouter_reasoning_details_on_tool_calls() {
    let req = request(vec![LlmMessage::new(
        LlmRole::Assistant,
        vec![LlmContentBlock::ToolCall {
            call_id: "call_1".to_string(),
            tool_name: "lookup".to_string(),
            input_json: "{\"q\":\"x\"}".to_string(),
            replay: Some(ProviderReplayMeta {
                item_id: None,
                opaque: Some(
                    json!({
                        "type": "reasoning.encrypted",
                        "id": "call_1",
                        "data": "encrypted"
                    })
                    .to_string(),
                ),
            }),
        }],
    )]);

    let body = OpenAiCompatibleProvider::new("key", OPENROUTER_BASE_URL)
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

    let parts = OpenAiCompatibleProvider::chat_response_parts_from_value(&value);
    let usage = lash_llm_transport::openai_usage_from_response_value(&value);

    assert!(matches!(&parts[0], LlmOutputPart::Reasoning { text, .. } if text == "think"));
    assert!(matches!(&parts[1], LlmOutputPart::Text { text, .. } if text == "hello"));
    assert!(matches!(
        &parts[2],
        LlmOutputPart::ToolCall {
            call_id,
            tool_name,
            input_json,
            replay,
            ..
        } if call_id == "call_1"
            && tool_name == "lookup"
            && input_json == "{\"q\":\"x\"}"
            && replay
                .as_ref()
                .and_then(|meta| meta.opaque.as_deref())
                .is_some_and(|value| value.contains("encrypted"))
    ));
    assert_eq!(usage.input_tokens, 13);
    assert_eq!(usage.output_tokens, 17);
    assert_eq!(usage.cached_input_tokens, 2);
}

#[test]
fn chat_stream_parser_captures_text_tool_done_and_usage() {
    let mut state = ChatStreamState::default();
    OpenAiCompatibleProvider::process_chat_sse_event(
        r#"{"choices":[{"delta":{"content":"Hi"}}]}"#,
        &mut state,
    )
    .unwrap();
    OpenAiCompatibleProvider::process_chat_sse_event(
        r#"{"choices":[{"delta":{"reasoning_content":"Think"}}]}"#,
        &mut state,
    )
    .unwrap();
    OpenAiCompatibleProvider::process_chat_sse_event(
        r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{\"q\":"}}]}}]}"#,
        &mut state,
    )
    .unwrap();
    OpenAiCompatibleProvider::process_chat_sse_event(
        r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"x\"}"}}]}}]}"#,
        &mut state,
    )
    .unwrap();
    OpenAiCompatibleProvider::process_chat_sse_event(
        r#"{"choices":[{"delta":{"reasoning_details":[{"type":"reasoning.encrypted","id":"call_1","data":"encrypted"}]}}],"usage":{"prompt_tokens":9,"completion_tokens":4,"prompt_tokens_details":{"cached_tokens":3,"cache_write_tokens":2}}}"#,
        &mut state,
    )
    .unwrap();
    OpenAiCompatibleProvider::process_chat_sse_event(
        r#"{"choices":[{"usage":{"prompt_tokens":11,"completion_tokens":5,"prompt_tokens_details":{"cached_tokens":4,"cache_write_tokens":2}}}]}"#,
        &mut state,
    )
    .unwrap();
    OpenAiCompatibleProvider::process_chat_sse_event("[DONE]", &mut state).unwrap();

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
            replay,
            ..
        } if call_id == "call_1"
            && tool_name == "lookup"
            && input_json == "{\"q\":\"x\"}"
            && replay
                .as_ref()
                .and_then(|meta| meta.opaque.as_deref())
                .is_some_and(|value| value.contains("encrypted"))
    ));
    assert_eq!(state.usage.input_tokens, 11);
    assert_eq!(state.usage.output_tokens, 5);
    assert_eq!(state.usage.cached_input_tokens, 2);
}

#[test]
fn stream_parser_captures_text_reasoning_tool_and_phase() {
    let mut state = ResponsesStreamState::default();
    OpenAiCompatibleProvider::process_sse_event(
        r#"{"type":"response.output_item.added","item":{"type":"reasoning","id":"rs_1"}}"#,
        &mut state,
        None,
    )
    .unwrap();
    OpenAiCompatibleProvider::process_sse_event(
        r#"{"type":"response.reasoning_summary_text.delta","delta":"Think"}"#,
        &mut state,
        None,
    )
    .unwrap();
    OpenAiCompatibleProvider::process_sse_event(
        r#"{"type":"response.output_item.done","item":{"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text","text":"Think"}],"encrypted_content":"enc"}}"#,
        &mut state,
        None,
    )
    .unwrap();
    OpenAiCompatibleProvider::process_sse_event(
        r#"{"type":"response.output_item.added","item":{"type":"message","id":"msg_1"}}"#,
        &mut state,
        None,
    )
    .unwrap();
    OpenAiCompatibleProvider::process_sse_event(
        r#"{"type":"response.output_text.delta","delta":"Hi"}"#,
        &mut state,
        None,
    )
    .unwrap();
    OpenAiCompatibleProvider::process_sse_event(
        r#"{"type":"response.output_item.done","item":{"type":"message","id":"msg_1","status":"completed","phase":"commentary","content":[{"type":"output_text","text":"Hi"}]}}"#,
        &mut state,
        None,
    )
    .unwrap();
    OpenAiCompatibleProvider::process_sse_event(
        r#"{"type":"response.output_item.added","item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"tool","arguments":""}}"#,
        &mut state,
        None,
    )
    .unwrap();
    OpenAiCompatibleProvider::process_sse_event(
        r#"{"type":"response.function_call_arguments.delta","item_id":"fc_1","delta":"{\"x\":"}"#,
        &mut state,
        None,
    )
    .unwrap();
    OpenAiCompatibleProvider::process_sse_event(
        r#"{"type":"response.function_call_arguments.done","item_id":"fc_1","arguments":"{\"x\":1}" }"#,
        &mut state,
        None,
    )
    .unwrap();
    OpenAiCompatibleProvider::process_sse_event(
        r#"{"type":"response.output_item.done","item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"tool","arguments":"{\"x\":1}","status":"completed"}}"#,
        &mut state,
        None,
    )
    .unwrap();

    assert_eq!(state.full_text, "Hi");
    let parts = state.response_parts();
    assert!(matches!(
        &parts[0],
        LlmOutputPart::Reasoning { replay: Some(replay), .. }
            if replay.item_id.as_deref() == Some("rs_1")
                && replay.encrypted_content.as_deref() == Some("enc")
    ));
    assert!(matches!(
        &parts[1],
        LlmOutputPart::Text {
            response_meta: Some(ResponseTextMeta {
                phase: Some(phase),
                ..
            }),
            ..
        } if phase == "commentary"
    ));
    assert!(matches!(
        &parts[2],
        LlmOutputPart::ToolCall {
            replay: Some(replay),
            input_json,
            ..
        } if replay.item_id.as_deref() == Some("fc_1") && input_json == "{\"x\":1}"
    ));
}

#[test]
fn responses_final_answer_phase_hides_commentary_from_visible_text() {
    let mut state = ResponsesStreamState::default();
    for event in [
        r#"{"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"msg_commentary","phase":"commentary"}}"#,
        r#"{"type":"response.output_text.delta","output_index":0,"item_id":"msg_commentary","delta":"Working notes."}"#,
        r#"{"type":"response.output_item.done","output_index":0,"item":{"type":"message","id":"msg_commentary","status":"completed","phase":"commentary","content":[{"type":"output_text","text":"Working notes."}]}}"#,
        r#"{"type":"response.output_item.added","output_index":1,"item":{"type":"message","id":"msg_final","phase":"final_answer"}}"#,
        r#"{"type":"response.output_text.delta","output_index":1,"item_id":"msg_final","delta":"Final answer."}"#,
        r#"{"type":"response.output_item.done","output_index":1,"item":{"type":"message","id":"msg_final","status":"completed","phase":"final_answer","content":[{"type":"output_text","text":"Final answer."}]}}"#,
        r#"{"type":"response.completed","response":{"id":"resp_1","status":"completed","output":[{"type":"message","id":"msg_commentary","status":"completed","phase":"commentary","content":[{"type":"output_text","text":"Working notes."}]},{"type":"message","id":"msg_final","status":"completed","phase":"final_answer","content":[{"type":"output_text","text":"Final answer."}]}]}}"#,
    ] {
        OpenAiCompatibleProvider::process_sse_event(event, &mut state, None).unwrap();
    }

    let parts = state.response_parts();
    assert_eq!(state.full_text, "Final answer.");
    assert_eq!(
        parts
            .iter()
            .filter(|part| matches!(part, LlmOutputPart::Text { .. }))
            .count(),
        2
    );
    let response = LlmResponse {
        full_text: state.full_text.clone(),
        parts,
        ..LlmResponse::default()
    };
    let visible = lash_core::normalized_response_parts(&response);
    assert_eq!(
        visible
            .iter()
            .filter_map(|part| match part {
                LlmOutputPart::Text { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect::<String>(),
        "Final answer."
    );
}

#[test]
fn response_failed_server_error_is_retryable() {
    let mut state = ResponsesStreamState::default();
    let err = OpenAiCompatibleProvider::process_sse_event(
        r#"{"type":"response.failed","response":{"status":"failed","error":{"code":"server_error","message":"internal stream ended unexpectedly"}}}"#,
        &mut state,
        None,
    )
    .unwrap_err();

    assert!(err.retryable);
    assert_eq!(err.message, "internal stream ended unexpectedly");
}

/// Cross-provider response-normalization conformance. The shared suite lives in
/// `lash_llm_transport::conformance`; here we wrap this crate's (private) chat
/// parsers in a `ProviderNormalizer` and supply OpenAI chat-API wire fixtures
/// for each canonical scenario.
#[cfg(feature = "testing")]
mod conformance {
    use crate::OpenAiCompatibleProvider;
    use crate::chat::ChatStreamState;
    use lash_core::llm::types::{LlmOutputPart, LlmTerminalReason, LlmUsage};
    use lash_llm_transport::conformance::{
        CanonicalUsage as U, ProviderNormalizer, ProviderWire, Scenario, StreamAssembly,
        provider_conformance,
    };
    use serde_json::{Value, json};

    struct OpenAiNormalizer;

    impl ProviderNormalizer for OpenAiNormalizer {
        fn name(&self) -> &str {
            "openai-chat"
        }

        fn wire_for(&self, scenario: Scenario) -> Option<ProviderWire> {
            let wire = match scenario {
                Scenario::PlainTextStop => ProviderWire::body(json!({
                    "choices": [{
                        "message": { "role": "assistant", "content": "hello" },
                        "finish_reason": "stop"
                    }],
                    "usage": { "prompt_tokens": U::BASE_INPUT, "completion_tokens": U::BASE_OUTPUT }
                })),
                Scenario::OutputCapped => ProviderWire::body(json!({
                    "choices": [{
                        "message": { "role": "assistant", "content": "trunc" },
                        "finish_reason": "length"
                    }]
                })),
                Scenario::ContentFilter => ProviderWire::body(json!({
                    "choices": [{
                        "message": { "role": "assistant", "content": "" },
                        "finish_reason": "content_filter"
                    }]
                })),
                Scenario::NonStreamingToolUse => ProviderWire::body(json!({
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "tool_calls": [{
                                "id": "call_1",
                                "type": "function",
                                "function": { "name": "lookup", "arguments": "{\"q\":\"x\"}" }
                            }]
                        },
                        "finish_reason": "tool_calls"
                    }]
                })),
                Scenario::StreamingTextAssembly => ProviderWire::body(json!({})).with_text_stream(
                    vec![
                        r#"{"choices":[{"delta":{"content":"hello "}}]}"#.to_string(),
                        r#"{"choices":[{"delta":{"content":"world"}}]}"#.to_string(),
                        r#"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#.to_string(),
                        "[DONE]".to_string(),
                    ],
                    "hello world",
                ),
                Scenario::StreamingToolArgumentMerge => {
                    ProviderWire::body(json!({})).with_tool_call_stream(
                        vec![
                            // arguments deliberately split across two SSE events
                            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{\"q\":"}}]}}]}"#.to_string(),
                            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"x\"}"}}]}}]}"#.to_string(),
                            r#"{"choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#.to_string(),
                            "[DONE]".to_string(),
                        ],
                        "lookup",
                        json!({ "q": "x" }),
                    )
                }
                Scenario::UsageCacheHit => ProviderWire::body(json!({
                    "choices": [{
                        "message": { "role": "assistant", "content": "ok" },
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": U::BASE_INPUT,
                        "completion_tokens": U::BASE_OUTPUT,
                        "prompt_tokens_details": { "cached_tokens": U::CACHED_INPUT }
                    }
                })),
                Scenario::UsageReasoning => ProviderWire::body(json!({
                    "choices": [{
                        "message": { "role": "assistant", "content": "ok" },
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": U::BASE_INPUT,
                        "completion_tokens": U::BASE_OUTPUT,
                        "completion_tokens_details": { "reasoning_tokens": U::REASONING }
                    }
                })),
                Scenario::ReasoningExtraction => ProviderWire::body(json!({
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "reasoning_content": "thinking about it",
                            "content": "answer"
                        },
                        "finish_reason": "stop"
                    }]
                }))
                .with_reasoning_text("thinking about it"),
                Scenario::StreamingUsageMerge => {
                    ProviderWire::body(json!({})).with_usage_merge_stream(vec![
                        // input arrives first, with no output yet
                        format!(
                            r#"{{"choices":[{{"delta":{{"content":"hi"}}}}],"usage":{{"prompt_tokens":{}}}}}"#,
                            U::BASE_INPUT
                        ),
                        // output arrives in a later event; merge must keep input
                        format!(
                            r#"{{"choices":[{{"delta":{{}}}}],"usage":{{"completion_tokens":{}}}}}"#,
                            U::BASE_OUTPUT
                        ),
                        "[DONE]".to_string(),
                    ])
                }
            };
            Some(wire)
        }

        fn parts_from_wire(&self, body: &Value) -> Vec<LlmOutputPart> {
            OpenAiCompatibleProvider::chat_response_parts_from_value(body)
        }

        fn usage_from_wire(&self, body: &Value) -> LlmUsage {
            lash_llm_transport::openai_usage_from_response_value(body)
        }

        fn terminal_from_wire(&self, body: &Value, parts: &[LlmOutputPart]) -> LlmTerminalReason {
            lash_llm_transport::openai_terminal_reason_from_chat_value(body, parts)
        }

        fn assemble_stream(&self, sse_events: &[String]) -> StreamAssembly {
            let mut state = ChatStreamState::default();
            for raw in sse_events {
                OpenAiCompatibleProvider::process_chat_sse_event(raw, &mut state)
                    .expect("chat sse event parses");
            }
            StreamAssembly {
                parts: state.parts(),
                usage: state.usage.clone(),
            }
        }
    }

    #[test]
    fn openai_satisfies_provider_conformance() {
        provider_conformance(&OpenAiNormalizer);
    }
}
