#![allow(clippy::result_large_err)]

mod config;
mod policy;
mod provider;
mod request;
mod stream;
mod support;

pub use config::{AnthropicProvider, AnthropicProviderFactory, DEFAULT_BASE_URL};

#[cfg(test)]
mod tests {
    use crate::stream::StreamState;
    use crate::{AnthropicProvider, DEFAULT_BASE_URL};
    use lash_core::llm::types::{
        LlmAttachment, LlmContentBlock, LlmJsonSchema, LlmMessage, LlmOutputPart, LlmOutputSpec,
        LlmRequest, LlmRole, LlmTerminalReason, LlmToolChoice, LlmToolSpec, LlmUsage,
    };
    use lash_core::provider::{CacheRetention, ProviderOptions};
    use serde_json::json;
    use std::num::NonZeroUsize;
    use std::sync::Arc;

    // `DEFAULT_BASE_URL` is part of the crate's public surface and exercised by
    // downstream hosts; reference it here so the re-export stays covered.
    const _: &str = DEFAULT_BASE_URL;

    fn request(messages: Vec<LlmMessage>) -> LlmRequest {
        LlmRequest {
            model: "claude-sonnet-4-6".to_string(),
            messages,
            attachments: Vec::new(),
            tools: Arc::new(Vec::<LlmToolSpec>::new()),
            tool_choice: LlmToolChoice::Auto,
            model_variant: None,
            scope: lash_core::LlmRequestScope::new(
                "session-1",
                "session-1:frame:test",
                "session-1:request:test",
            ),
            output_spec: None,
            stream_events: None,
            generation: lash_core::GenerationOptions::default(),
            provider_trace: None,
        }
    }

    #[test]
    fn image_attachment_serializes_as_base64_image_block() {
        use base64::Engine;
        let provider = AnthropicProvider::new("key");
        let png_bytes = vec![0x89, 0x50, 0x4E, 0x47];
        let mut req = request(vec![LlmMessage::new(
            LlmRole::User,
            vec![
                LlmContentBlock::Text {
                    text: "look at this".into(),
                    response_meta: None,
                    cache_breakpoint: false,
                },
                LlmContentBlock::Image { attachment_idx: 0 },
            ],
        )]);
        req.attachments = vec![LlmAttachment::bytes("image/png", png_bytes.clone())];

        let body = provider.build_request_body(&req).expect("body");

        let messages = body["messages"].as_array().expect("messages array");
        let user_msg = messages.last().expect("user message");
        let content = user_msg["content"].as_array().expect("content array");
        let image_block = content
            .iter()
            .find(|b| b["type"] == "image")
            .expect("image block");
        assert_eq!(image_block["source"]["type"], "base64");
        assert_eq!(image_block["source"]["media_type"], "image/png");
        let expected_b64 = base64::engine::general_purpose::STANDARD.encode(&png_bytes);
        assert_eq!(image_block["source"]["data"], expected_b64);
    }

    #[test]
    fn unsupported_image_mime_is_rejected_at_request_boundary() {
        let provider = AnthropicProvider::new("key");
        let mut req = request(vec![LlmMessage::new(
            LlmRole::User,
            vec![LlmContentBlock::Image { attachment_idx: 0 }],
        )]);
        req.attachments = vec![LlmAttachment::bytes("image/bmp", vec![0x42, 0x4D])];

        let err = provider
            .build_request_body(&req)
            .expect_err("bmp should be rejected before wire");

        assert_eq!(err.code.as_deref(), Some("unsupported_image_format"));
        assert!(err.message.contains("Anthropic"));
        assert!(err.message.contains("image/bmp"));
    }

    #[test]
    fn structured_output_uses_native_output_config_format() {
        let provider = AnthropicProvider::new("key");
        let mut req = request(vec![
            LlmMessage::text(LlmRole::System, "system prompt"),
            LlmMessage::text(LlmRole::User, "extract"),
        ]);
        req.output_spec = Some(LlmOutputSpec::JsonSchema(LlmJsonSchema {
            name: "extract_result".to_string(),
            strict: true,
            schema: json!({
                "type": "object",
                "additionalProperties": false,
                "required": ["answer"],
                "properties": {
                    "answer": { "type": "string" }
                }
            })
            .into(),
        }));

        let body = provider.build_request_body(&req).expect("body");

        assert_eq!(
            body["output_config"]["format"],
            json!({
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["answer"],
                    "properties": {
                        "answer": { "type": "string" }
                    }
                }
            })
        );
        let system_text = body["system"][0]["text"].as_str().unwrap_or_default();
        assert_eq!(system_text, "system prompt");
        assert!(!system_text.contains("Respond with a single JSON object"));
    }

    #[test]
    fn structured_output_preserves_adaptive_effort_config() {
        let provider = AnthropicProvider::new("key");
        let mut req = request(vec![LlmMessage::text(LlmRole::User, "extract")]);
        req.model_variant = Some("medium".to_string());
        req.output_spec = Some(LlmOutputSpec::JsonObject);

        let body = provider.build_request_body(&req).expect("body");

        assert_eq!(body["output_config"]["effort"], json!("medium"));
        assert_eq!(
            body["output_config"]["format"],
            json!({
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "additionalProperties": true,
                }
            })
        );
    }

    #[test]
    fn structured_output_strips_bedrock_unsafe_array_constraints() {
        let provider = AnthropicProvider::new("key");
        let mut req = request(vec![LlmMessage::text(LlmRole::User, "rank")]);
        req.output_spec = Some(LlmOutputSpec::JsonSchema(LlmJsonSchema {
            name: "rank_result".to_string(),
            strict: true,
            schema: json!({
                "type": "object",
                "required": ["ranked"],
                "properties": {
                    "ranked": {
                        "type": "array",
                        "minItems": 2,
                        "maxItems": 2,
                        "items": { "type": "string" }
                    }
                }
            })
            .into(),
        }));

        let body = provider.build_request_body(&req).expect("body");
        let ranked = &body["output_config"]["format"]["schema"]["properties"]["ranked"];

        assert!(ranked.get("minItems").is_none());
        assert!(ranked.get("maxItems").is_none());
        assert!(
            ranked["description"]
                .as_str()
                .is_some_and(|description| description.contains("maxItems=2"))
        );
    }

    #[test]
    fn tool_input_schema_uses_anthropic_projection() {
        let provider = AnthropicProvider::new("key");
        let mut req = request(vec![LlmMessage::text(LlmRole::User, "rank")]);
        req.tools = Arc::new(vec![LlmToolSpec {
            name: "rank".to_string(),
            description: "Rank".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "ids": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 3,
                        "items": { "type": "string" }
                    }
                }
            })
            .into(),
            output_schema: json!({}).into(),
        }]);

        let body = provider.build_request_body(&req).expect("body");
        let ids = &body["tools"][0]["input_schema"]["properties"]["ids"];

        assert!(ids.get("minItems").is_none());
        assert!(ids.get("maxItems").is_none());
    }

    #[test]
    fn pause_turn_maps_to_stop() {
        let state = StreamState {
            stop_reason: Some("pause_turn".to_string()),
            ..StreamState::default()
        };

        let (_, _, _, terminal_reason) = AnthropicProvider::finalize(state);

        assert_eq!(terminal_reason, LlmTerminalReason::Stop);
    }

    #[test]
    fn unknown_stop_reason_maps_to_provider_error() {
        let state = StreamState {
            stop_reason: Some("new_provider_reason".to_string()),
            ..StreamState::default()
        };

        let (_, _, _, terminal_reason) = AnthropicProvider::finalize(state);

        assert_eq!(terminal_reason, LlmTerminalReason::ProviderError);
    }

    #[test]
    fn thinking_display_is_omitted_unless_provider_exposes_thinking() {
        let mut req = request(vec![LlmMessage::text(LlmRole::User, "extract")]);
        req.model_variant = Some("medium".to_string());

        let hidden = AnthropicProvider::new("key")
            .build_request_body(&req)
            .expect("body");
        assert_eq!(hidden["thinking"]["display"], "omitted");

        let exposed = AnthropicProvider::new("key")
            .with_options(ProviderOptions {
                expose_thinking: true,
                ..ProviderOptions::default()
            })
            .build_request_body(&req)
            .expect("body");
        assert_eq!(exposed["thinking"]["display"], "summarized");
    }

    #[test]
    fn request_body_omits_temperature_without_explicit_temperature_option() {
        let provider = AnthropicProvider::new("key");
        let plain = provider
            .build_request_body(&request(vec![LlmMessage::text(LlmRole::User, "hello")]))
            .expect("plain body");
        assert!(plain.get("temperature").is_none());

        let mut thinking_req = request(vec![LlmMessage::text(LlmRole::User, "think")]);
        thinking_req.model_variant = Some("medium".to_string());
        let thinking = provider
            .build_request_body(&thinking_req)
            .expect("thinking body");
        assert!(thinking.get("temperature").is_none());
    }

    #[test]
    fn explicit_text_cache_breakpoint_beats_last_user_block() {
        let provider = AnthropicProvider::new("key");
        let req = request(vec![LlmMessage::new(
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
        )]);

        let body = provider.build_request_body(&req).expect("body");

        assert_eq!(
            body["messages"][0]["content"][0]["cache_control"],
            json!({ "type": "ephemeral" })
        );
        assert!(
            body["messages"][0]["content"][1]
                .get("cache_control")
                .is_none()
        );
        assert!(
            body["messages"][0]["content"][0]
                .get("__lash_cache_breakpoint")
                .is_none()
        );
    }

    #[test]
    fn cache_retention_none_removes_cache_control() {
        let provider = AnthropicProvider::new("key").with_options(ProviderOptions {
            cache_retention: CacheRetention::None,
            ..ProviderOptions::default()
        });
        let req = request(vec![
            LlmMessage::text(LlmRole::System, "stable system prompt"),
            LlmMessage::text(LlmRole::User, "dynamic tail"),
        ]);

        let body = provider.build_request_body(&req).expect("body");

        assert!(body["system"][0].get("cache_control").is_none());
        assert!(
            body["messages"][0]["content"][0]
                .get("cache_control")
                .is_none()
        );
    }

    #[test]
    fn cache_retention_long_emits_ttl() {
        let provider = AnthropicProvider::new("key").with_options(ProviderOptions {
            cache_retention: CacheRetention::Long,
            ..ProviderOptions::default()
        });
        let req = request(vec![
            LlmMessage::text(LlmRole::System, "stable system prompt"),
            LlmMessage::text(LlmRole::User, "dynamic tail"),
        ]);

        let body = provider.build_request_body(&req).expect("body");

        assert_eq!(
            body["system"][0]["cache_control"],
            json!({ "type": "ephemeral", "ttl": "1h" })
        );
        assert_eq!(
            body["messages"][0]["content"][0]["cache_control"],
            json!({ "type": "ephemeral", "ttl": "1h" })
        );
    }

    #[test]
    fn output_token_cap_maps_to_max_tokens() {
        let provider = AnthropicProvider::new("key").with_options(ProviderOptions {
            max_output_tokens: Some(9999),
            ..ProviderOptions::default()
        });
        let mut req = request(vec![LlmMessage::text(LlmRole::User, "hello")]);
        req.generation.output_token_cap = NonZeroUsize::new(2048);

        let body = provider.build_request_body(&req).expect("body");

        assert_eq!(body["max_tokens"], 2048);
        let provider_limited_body = provider
            .build_request_body(&request(vec![LlmMessage::text(LlmRole::User, "hello")]))
            .expect("body");
        assert_eq!(provider_limited_body["max_tokens"], 9999);
    }

    /// Cross-provider response-normalization conformance. Anthropic is
    /// streaming-first (no non-streaming `parts_from_value`), so each scenario's
    /// `body` carries the SSE event sequence as a JSON array of strings, and all
    /// three accessors replay it through `process_sse_event` + `finalize`.
    /// Anthropic reports no separate reasoning-token count (`parse_usage`
    /// hardcodes `reasoning_tokens: 0`), so it opts out of `UsageReasoning`.
    #[cfg(feature = "testing")]
    mod conformance {
        use super::*;
        use lash_llm_transport::conformance::{
            CanonicalUsage as U, ProviderConformanceSpec, ProviderNormalizer, ProviderWire,
            Scenario, StreamAssembly, provider_conformance,
        };
        use serde_json::Value;

        struct AnthropicNormalizer;

        // Replay a `body` that encodes a JSON array of SSE event strings through
        // the streaming parser and finalize into normalized outputs.
        fn replay(body: &Value) -> (Vec<LlmOutputPart>, LlmUsage, LlmTerminalReason) {
            let mut state = StreamState::default();
            if let Some(events) = body.as_array() {
                for event in events {
                    let raw = event.as_str().expect("sse event is a string");
                    AnthropicProvider::process_sse_event(raw, &mut state, None, true)
                        .expect("anthropic sse event parses");
                }
            }
            let (parts, _text, usage, terminal) = AnthropicProvider::finalize(state);
            (parts, usage, terminal)
        }

        // Build the SSE event array for a plain single-text-block message with a
        // given stop_reason and message_start usage block.
        fn text_message(stop_reason: &str, text: &str, usage: Value) -> Value {
            json!([
                json!({ "type": "message_start", "message": { "usage": usage } }).to_string(),
                json!({ "type": "content_block_start", "index": 0, "content_block": { "type": "text" } }).to_string(),
                json!({ "type": "content_block_delta", "index": 0, "delta": { "type": "text_delta", "text": text } }).to_string(),
                json!({ "type": "content_block_stop", "index": 0 }).to_string(),
                json!({ "type": "message_delta", "delta": { "stop_reason": stop_reason } }).to_string(),
            ])
        }

        fn tool_use_message() -> Value {
            json!([
                json!({ "type": "message_start", "message": { "usage": { "input_tokens": U::BASE_INPUT, "output_tokens": U::BASE_OUTPUT } } }).to_string(),
                json!({ "type": "content_block_start", "index": 0, "content_block": { "type": "tool_use", "id": "call_1", "name": "lookup", "input": {} } }).to_string(),
                // arguments deliberately split across two delta events
                json!({ "type": "content_block_delta", "index": 0, "delta": { "type": "input_json_delta", "partial_json": "{\"q\":" } }).to_string(),
                json!({ "type": "content_block_delta", "index": 0, "delta": { "type": "input_json_delta", "partial_json": "\"x\"}" } }).to_string(),
                json!({ "type": "content_block_stop", "index": 0 }).to_string(),
                json!({ "type": "message_delta", "delta": { "stop_reason": "tool_use" } }).to_string(),
            ])
        }

        impl ProviderNormalizer for AnthropicNormalizer {
            fn name(&self) -> &str {
                "anthropic"
            }

            fn conformance_spec(&self) -> ProviderConformanceSpec {
                ProviderConformanceSpec::with_unsupported(&[(
                    Scenario::UsageReasoning,
                    "Anthropic usage does not expose separate reasoning-token counts",
                )])
            }

            fn wire_for(&self, scenario: Scenario) -> Option<ProviderWire> {
                let wire = match scenario {
                    Scenario::PlainTextStop => ProviderWire::body(text_message(
                        "end_turn",
                        "hello",
                        json!({ "input_tokens": U::BASE_INPUT, "output_tokens": U::BASE_OUTPUT }),
                    )),
                    Scenario::OutputCapped => ProviderWire::body(text_message(
                        "max_tokens",
                        "trunc",
                        json!({ "input_tokens": U::BASE_INPUT, "output_tokens": U::BASE_OUTPUT }),
                    )),
                    Scenario::ContentFilter => ProviderWire::body(text_message(
                        "refusal",
                        "",
                        json!({ "input_tokens": U::BASE_INPUT, "output_tokens": U::BASE_OUTPUT }),
                    )),
                    Scenario::NonStreamingToolUse => ProviderWire::body(tool_use_message()),
                    Scenario::StreamingTextAssembly => {
                        ProviderWire::body(Value::Null).with_text_stream(
                            vec![
                                json!({ "type": "message_start", "message": { "usage": { "input_tokens": U::BASE_INPUT, "output_tokens": U::BASE_OUTPUT } } }).to_string(),
                                json!({ "type": "content_block_start", "index": 0, "content_block": { "type": "text" } }).to_string(),
                                json!({ "type": "content_block_delta", "index": 0, "delta": { "type": "text_delta", "text": "hello " } }).to_string(),
                                json!({ "type": "content_block_delta", "index": 0, "delta": { "type": "text_delta", "text": "world" } }).to_string(),
                                json!({ "type": "content_block_stop", "index": 0 }).to_string(),
                                json!({ "type": "message_delta", "delta": { "stop_reason": "end_turn" } }).to_string(),
                            ],
                            "hello world",
                        )
                    }
                    Scenario::StreamingToolArgumentMerge => {
                        let events = tool_use_message();
                        ProviderWire::body(events.clone()).with_tool_call_stream(
                            events
                                .as_array()
                                .unwrap()
                                .iter()
                                .map(|v| v.as_str().unwrap().to_string())
                                .collect(),
                            "lookup",
                            json!({ "q": "x" }),
                        )
                    }
                    Scenario::UsageCacheHit => ProviderWire::body(text_message(
                        "end_turn",
                        "ok",
                        // Anthropic's input_tokens is net of cache; the suite's
                        // canonical input is the gross total, so split it.
                        json!({
                            "input_tokens": U::BASE_INPUT - U::CACHED_INPUT,
                            "output_tokens": U::BASE_OUTPUT,
                            "cache_read_input_tokens": U::CACHED_INPUT
                        }),
                    )),
                    // Anthropic does not report a separate reasoning-token count.
                    Scenario::UsageReasoning => return None,
                    Scenario::ReasoningExtraction => ProviderWire::body(json!([
                        json!({ "type": "message_start", "message": { "usage": { "input_tokens": U::BASE_INPUT, "output_tokens": U::BASE_OUTPUT } } }).to_string(),
                        json!({ "type": "content_block_start", "index": 0, "content_block": { "type": "thinking" } }).to_string(),
                        json!({ "type": "content_block_delta", "index": 0, "delta": { "type": "thinking_delta", "thinking": "thinking about it" } }).to_string(),
                        json!({ "type": "content_block_stop", "index": 0 }).to_string(),
                        json!({ "type": "content_block_start", "index": 1, "content_block": { "type": "text" } }).to_string(),
                        json!({ "type": "content_block_delta", "index": 1, "delta": { "type": "text_delta", "text": "answer" } }).to_string(),
                        json!({ "type": "content_block_stop", "index": 1 }).to_string(),
                        json!({ "type": "message_delta", "delta": { "stop_reason": "end_turn" } }).to_string(),
                    ]))
                    .with_reasoning_text("thinking about it"),
                    Scenario::StreamingUsageMerge => ProviderWire::body(Value::Null)
                        .with_usage_merge_stream(vec![
                            // input arrives in message_start
                            json!({ "type": "message_start", "message": { "usage": { "input_tokens": U::BASE_INPUT } } }).to_string(),
                            // output arrives later in message_delta; merge must keep input
                            json!({ "type": "message_delta", "delta": { "stop_reason": "end_turn" }, "usage": { "output_tokens": U::BASE_OUTPUT } }).to_string(),
                        ]),
                };
                Some(wire)
            }

            fn parts_from_wire(&self, body: &Value) -> Vec<LlmOutputPart> {
                replay(body).0
            }

            fn usage_from_wire(&self, body: &Value) -> LlmUsage {
                replay(body).1
            }

            fn terminal_from_wire(
                &self,
                body: &Value,
                _parts: &[LlmOutputPart],
            ) -> LlmTerminalReason {
                replay(body).2
            }

            fn assemble_stream(&self, sse_events: &[String]) -> StreamAssembly {
                let mut state = StreamState::default();
                for raw in sse_events {
                    AnthropicProvider::process_sse_event(raw, &mut state, None, true)
                        .expect("anthropic sse event parses");
                }
                let (parts, _text, usage, _terminal) = AnthropicProvider::finalize(state);
                StreamAssembly { parts, usage }
            }
        }

        #[test]
        fn anthropic_satisfies_provider_conformance() {
            provider_conformance(&AnthropicNormalizer);
        }
    }
}
