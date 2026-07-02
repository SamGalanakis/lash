#![allow(clippy::result_large_err)]

mod config;
pub mod oauth;
mod policy;
mod provider;
mod request;
mod stream;
mod support;
mod upload;

pub use config::{GoogleOAuthProvider, GoogleOAuthProviderFactory};

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;
    use std::sync::Arc;

    use super::GoogleOAuthProvider;
    use base64::Engine;
    use lash_core::llm::transport::validate_image_attachments;
    use lash_core::llm::types::{
        LlmContentBlock, LlmEventSender, LlmMessage, LlmOutputPart, LlmRequest, LlmRole,
        LlmTerminalReason, LlmToolChoice, LlmToolSpec, LlmUsage, ResponseTextMeta,
    };
    use lash_core::provider::ProviderOptions;
    use serde_json::{Value, json};

    fn request(model_variant: Option<&str>) -> LlmRequest {
        LlmRequest {
            model: "gemini-3.1-pro-preview".to_string(),
            messages: vec![LlmMessage::text(LlmRole::User, "hello")],
            attachments: Vec::new(),
            tools: Arc::new(Vec::<LlmToolSpec>::new()),
            tool_choice: LlmToolChoice::Auto,
            model_variant: model_variant.map(str::to_string),
            scope: lash_core::LlmRequestScope::new(
                "session-1",
                "session-1:frame:test",
                "session-1:request:test",
            ),
            output_spec: None,
            stream_events: None::<LlmEventSender>,
            generation: lash_core::GenerationOptions::default(),
            provider_trace: None,
        }
    }

    #[test]
    fn usage_payload_maps_canonical_token_buckets() {
        let usage = GoogleOAuthProvider::usage_from_event(&json!({
            "response": {
                "usageMetadata": {
                    "promptTokenCount": 21,
                    "cachedContentTokenCount": 5,
                    "candidatesTokenCount": 10,
                    "thoughtsTokenCount": 3
                }
            }
        }));

        assert_eq!(
            usage,
            LlmUsage {
                input_tokens: 16,
                output_tokens: 13,
                cache_read_input_tokens: 5,
                cache_write_input_tokens: 0,
                reasoning_output_tokens: 3,
            }
        );
    }

    #[test]
    fn google_image_attachment_serializes_as_inline_data_part() {
        let png_bytes = vec![0x89, 0x50, 0x4E, 0x47];
        let attachment =
            lash_core::llm::types::LlmAttachment::bytes("image/png", png_bytes.clone());

        let part = GoogleOAuthProvider::inline_attachment_part(&attachment);

        let expected_b64 = base64::engine::general_purpose::STANDARD.encode(&png_bytes);
        assert_eq!(part["inlineData"]["mimeType"], "image/png");
        assert_eq!(part["inlineData"]["data"], expected_b64);
    }

    #[test]
    fn google_rejects_gif_attachment_at_request_boundary() {
        let mut req = request(None);
        req.attachments = vec![lash_core::llm::types::LlmAttachment::bytes(
            "image/gif",
            vec![0x47, 0x49, 0x46],
        )];

        let err = validate_image_attachments(
            &req,
            &[
                "image/jpeg",
                "image/png",
                "image/webp",
                "image/heic",
                "image/heif",
            ],
            "Google Gemini",
        )
        .expect_err("gif should be rejected for Gemini");

        assert_eq!(err.code.as_deref(), Some("unsupported_image_format"));
        assert!(err.message.contains("Google Gemini"));
        assert!(err.message.contains("image/gif"));
    }

    #[test]
    fn google_accepts_webp_attachment_through_validation() {
        let mut req = request(None);
        req.attachments = vec![lash_core::llm::types::LlmAttachment::bytes(
            "image/webp",
            vec![0],
        )];

        validate_image_attachments(
            &req,
            &[
                "image/jpeg",
                "image/png",
                "image/webp",
                "image/heic",
                "image/heif",
            ],
            "Google Gemini",
        )
        .expect("webp is supported");
    }

    #[test]
    fn google_unknown_finish_reason_maps_to_provider_error() {
        let terminal_reason = GoogleOAuthProvider::terminal_reason_from_value(
            &json!({"candidates":[{"finishReason":"NEW_REASON"}]}),
            &[],
        );

        assert_eq!(terminal_reason, LlmTerminalReason::ProviderError);
    }

    #[test]
    fn google_image_safety_finish_reason_maps_to_content_filter() {
        let terminal_reason = GoogleOAuthProvider::terminal_reason_from_value(
            &json!({"candidates":[{"finishReason":"IMAGE_SAFETY"}]}),
            &[],
        );

        assert_eq!(terminal_reason, LlmTerminalReason::ContentFilter);
    }

    #[test]
    fn streaming_captures_finish_reason_instead_of_hardcoding_stop() {
        // Regression: the streaming finalizer used to hardcode terminal_reason
        // = Stop, mislabeling MAX_TOKENS / tool-call / safety turns. Drive the
        // SSE events through process_sse_event and confirm the captured
        // finishReason maps through terminal_reason_from_value (here MAX_TOKENS
        // -> OutputLimit), exactly like the non-streaming path.
        let mut full = String::new();
        let mut usage = LlmUsage::default();
        let mut tool_calls: Vec<LlmOutputPart> = Vec::new();
        let mut finish_event: Option<serde_json::Value> = None;
        for raw in [
            r#"{"candidates":[{"content":{"parts":[{"text":"hi"}]}}]}"#,
            r#"{"candidates":[{"finishReason":"MAX_TOKENS"}]}"#,
        ] {
            GoogleOAuthProvider::process_sse_event(
                raw,
                &mut full,
                &mut Vec::new(),
                &mut usage,
                Some(&mut tool_calls),
                &mut finish_event,
            )
            .expect("sse event");
        }
        assert!(
            finish_event.is_some(),
            "finishReason event must be captured"
        );
        let terminal_reason = GoogleOAuthProvider::terminal_reason_from_value(
            finish_event.as_ref().unwrap_or(&serde_json::Value::Null),
            &[],
        );
        assert_eq!(terminal_reason, LlmTerminalReason::OutputLimit);
    }

    #[test]
    fn streaming_captures_raw_usage_metadata_sidecar() {
        let mut full = String::new();
        let mut text_deltas = Vec::new();
        let mut usage = LlmUsage::default();
        let mut provider_usage: Option<Value> = None;
        let mut finish_event: Option<Value> = None;
        let meta = json!({"promptTokenCount": 6, "candidatesTokenCount": 4});
        for raw in [
            json!({"response":{"candidates":[{"content":{"parts":[{"text":"hi"}]}}]}}).to_string(),
            json!({"response":{"usageMetadata": meta}}).to_string(),
            // A trailing empty usage block must not clobber the captured raw
            // sidecar, mirroring the normalized-usage non-zero guard.
            json!({"response":{"usageMetadata": {}}}).to_string(),
        ] {
            GoogleOAuthProvider::process_sse_event_with_text_parts(
                &raw,
                crate::support::SseTextPartSink {
                    full: &mut full,
                    text_deltas: &mut text_deltas,
                    usage: &mut usage,
                    provider_usage: &mut provider_usage,
                    tool_call_parts: None,
                    text_parts: None,
                    finish_event: &mut finish_event,
                },
                None,
            )
            .expect("sse event");
        }
        assert_eq!(provider_usage, Some(meta));
        assert_eq!(usage.input_tokens, 6);
        assert_eq!(usage.output_tokens, 4);
    }

    #[test]
    fn thinking_config_omits_thoughts_unless_provider_exposes_thinking() {
        let hidden_provider = GoogleOAuthProvider::new("access", "refresh", 0);
        let hidden = GoogleOAuthProvider::build_request(
            &hidden_provider,
            &request(Some("medium")),
            Vec::new(),
            None,
        );
        assert_eq!(
            hidden["request"]["generationConfig"]["thinkingConfig"]["thinkingLevel"],
            "medium"
        );
        assert!(
            hidden["request"]["generationConfig"]["thinkingConfig"]
                .get("includeThoughts")
                .is_none()
        );

        let exposed_provider =
            GoogleOAuthProvider::new("access", "refresh", 0).with_options(ProviderOptions {
                expose_thinking: true,
                ..ProviderOptions::default()
            });
        let exposed = GoogleOAuthProvider::build_request(
            &exposed_provider,
            &request(Some("medium")),
            Vec::new(),
            None,
        );
        assert_eq!(
            exposed["request"]["generationConfig"]["thinkingConfig"]["includeThoughts"],
            true
        );
    }

    #[test]
    fn output_token_cap_maps_to_max_output_tokens() {
        let provider =
            GoogleOAuthProvider::new("access", "refresh", 0).with_options(ProviderOptions {
                max_output_tokens: Some(9999),
                ..ProviderOptions::default()
            });

        let mut req = request(None);
        req.generation.output_token_cap = NonZeroUsize::new(4096);
        let body = GoogleOAuthProvider::build_request(&provider, &req, Vec::new(), None);

        assert_eq!(body["request"]["generationConfig"]["maxOutputTokens"], 4096);
        let provider_limited =
            GoogleOAuthProvider::build_request(&provider, &request(None), Vec::new(), None);
        assert_eq!(
            provider_limited["request"]["generationConfig"]["maxOutputTokens"],
            9999
        );
    }

    #[test]
    fn google_text_thought_signature_is_stored_and_replayed_for_same_origin() {
        let signature = base64::engine::general_purpose::STANDARD.encode("sig");
        let value = json!({
            "candidates": [{
                "content": {
                    "parts": [{
                        "text": "hello",
                        "thoughtSignature": signature
                    }]
                }
            }]
        });
        let parts =
            GoogleOAuthProvider::response_parts_from_value(&value, Some("gemini-3.1-pro-preview"));
        let meta = match &parts[0] {
            LlmOutputPart::Text {
                response_meta: Some(meta),
                ..
            } => meta,
            other => panic!("expected text metadata, got {other:?}"),
        };
        assert_eq!(meta.provider_payload.as_deref(), Some(signature.as_str()));
        assert_eq!(
            meta.origin_provider.as_deref(),
            Some(GoogleOAuthProvider::PROVIDER_KIND)
        );
        assert_eq!(meta.origin_model.as_deref(), Some("gemini-3.1-pro-preview"));

        let mut req = request(None);
        req.messages = vec![LlmMessage::new(
            LlmRole::Assistant,
            vec![LlmContentBlock::Text {
                text: "hello".into(),
                response_meta: Some(meta.clone()),
                cache_breakpoint: false,
            }],
        )];
        let contents = GoogleOAuthProvider::build_contents_with_attachment_parts(&req, &[]);
        assert_eq!(contents[0]["parts"][0]["thoughtSignature"], signature);
    }

    #[test]
    fn google_text_thought_signature_replay_rejects_invalid_or_cross_origin_metadata() {
        let valid = base64::engine::general_purpose::STANDARD.encode("sig");
        for meta in [
            ResponseTextMeta {
                provider_payload: Some("not base64!".to_string()),
                origin_provider: Some(GoogleOAuthProvider::PROVIDER_KIND.to_string()),
                origin_model: Some("gemini-3.1-pro-preview".to_string()),
                ..ResponseTextMeta::default()
            },
            ResponseTextMeta {
                provider_payload: Some(valid.clone()),
                origin_provider: Some("other_provider".to_string()),
                origin_model: Some("gemini-3.1-pro-preview".to_string()),
                ..ResponseTextMeta::default()
            },
            ResponseTextMeta {
                provider_payload: Some(valid.clone()),
                origin_provider: Some(GoogleOAuthProvider::PROVIDER_KIND.to_string()),
                origin_model: Some("gemini-2.5-pro".to_string()),
                ..ResponseTextMeta::default()
            },
        ] {
            let mut req = request(None);
            req.messages = vec![LlmMessage::new(
                LlmRole::Assistant,
                vec![LlmContentBlock::Text {
                    text: "hello".into(),
                    response_meta: Some(meta),
                    cache_breakpoint: false,
                }],
            )];
            let contents = GoogleOAuthProvider::build_contents_with_attachment_parts(&req, &[]);
            assert!(contents[0]["parts"][0].get("thoughtSignature").is_none());
        }
    }

    #[test]
    fn google_claude_on_vertex_tool_parameters_strip_json_schema_meta_declarations() {
        let provider = GoogleOAuthProvider::new("access", "refresh", 0);
        let mut claude_on_vertex = request(None);
        claude_on_vertex.model = "claude-sonnet-4-6".to_string();
        claude_on_vertex.tools = Arc::new(vec![LlmToolSpec {
            name: "lookup".to_string(),
            description: "Lookup".to_string(),
            input_schema: json!({
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "$id": "tool.schema.json",
                "$defs": { "unused": { "type": "string" } },
                "definitions": { "old": { "type": "string" } },
                "type": "object",
                "properties": {
                    "nested": {
                        "$id": "nested",
                        "$defs": { "x": { "type": "string" } },
                        "type": "object"
                    }
                }
            })
            .into(),
            output_schema: json!({}).into(),
        }]);
        let claude_on_vertex_body =
            GoogleOAuthProvider::build_request(&provider, &claude_on_vertex, Vec::new(), None);
        let parameters =
            &claude_on_vertex_body["request"]["tools"][0]["functionDeclarations"][0]["parameters"];
        assert!(parameters.get("$schema").is_none());
        assert!(parameters.get("$id").is_none());
        assert!(parameters.get("$defs").is_none());
        assert!(parameters.get("definitions").is_none());
        assert!(parameters["properties"]["nested"].get("$id").is_none());
        assert!(
            claude_on_vertex_body["request"]["tools"][0]["functionDeclarations"][0]
                .get("parametersJsonSchema")
                .is_none()
        );

        let mut gemini = claude_on_vertex;
        gemini.model = "gemini-3.1-pro-preview".to_string();
        let gemini_body = GoogleOAuthProvider::build_request(&provider, &gemini, Vec::new(), None);
        assert!(
            gemini_body["request"]["tools"][0]["functionDeclarations"][0]["parametersJsonSchema"]
                .get("$schema")
                .is_some()
        );
    }

    /// Cross-provider response-normalization conformance. Wraps this crate's
    /// (private) Gemini parsers in a `ProviderNormalizer`. Gemini materializes
    /// non-streaming function calls, but it does not expose the streaming
    /// chunk-merge scenarios in the same shape as SSE-first providers.
    #[cfg(feature = "testing")]
    mod conformance {
        use super::*;
        use lash_llm_transport::conformance::{
            CanonicalUsage as U, ProviderConformanceSpec, ProviderNormalizer, ProviderWire,
            Scenario, StreamAssembly, provider_conformance,
        };

        struct GoogleNormalizer;

        impl ProviderNormalizer for GoogleNormalizer {
            fn name(&self) -> &str {
                "google-gemini"
            }

            fn conformance_spec(&self) -> ProviderConformanceSpec {
                ProviderConformanceSpec::with_unsupported(&[
                    (
                        Scenario::StreamingToolArgumentMerge,
                        "Gemini streams complete functionCall objects, not argument deltas",
                    ),
                    (
                        Scenario::StreamingUsageMerge,
                        "Gemini usage events replace aggregate usage instead of incremental SSE deltas",
                    ),
                ])
            }

            fn wire_for(&self, scenario: Scenario) -> Option<ProviderWire> {
                let wire = match scenario {
                    Scenario::PlainTextStop => ProviderWire::body(json!({
                        "candidates": [{
                            "content": { "parts": [{ "text": "hello" }] },
                            "finishReason": "STOP"
                        }],
                        "usageMetadata": {
                            "promptTokenCount": U::BASE_INPUT,
                            "candidatesTokenCount": U::BASE_OUTPUT
                        }
                    })),
                    Scenario::OutputCapped => ProviderWire::body(json!({
                        "candidates": [{
                            "content": { "parts": [{ "text": "trunc" }] },
                            "finishReason": "MAX_TOKENS"
                        }]
                    })),
                    Scenario::ContentFilter => ProviderWire::body(json!({
                        "candidates": [{ "content": { "parts": [] }, "finishReason": "SAFETY" }]
                    })),
                    Scenario::NonStreamingToolUse => ProviderWire::body(json!({
                        "candidates": [{
                            "content": { "parts": [{
                                "functionCall": {
                                    "id": "call_1",
                                    "name": "lookup",
                                    "args": { "q": "x" }
                                }
                            }] }
                        }]
                    })),
                    Scenario::StreamingTextAssembly => {
                        ProviderWire::body(json!({})).with_text_stream(
                            vec![
                                r#"{"response":{"candidates":[{"content":{"parts":[{"text":"hello "}]}}]}}"#.to_string(),
                                r#"{"response":{"candidates":[{"content":{"parts":[{"text":"world"}]},"finishReason":"STOP"}]}}"#.to_string(),
                            ],
                            "hello world",
                        )
                    }
                    Scenario::StreamingToolArgumentMerge => return None,
                    Scenario::UsageCacheHit => ProviderWire::body(json!({
                        "candidates": [{
                            "content": { "parts": [{ "text": "ok" }] },
                            "finishReason": "STOP"
                        }],
                        "usageMetadata": {
                            "promptTokenCount": U::BASE_INPUT,
                            "candidatesTokenCount": U::BASE_OUTPUT,
                            "cachedContentTokenCount": U::CACHED_INPUT
                        }
                    })),
                    Scenario::UsageReasoning => ProviderWire::body(json!({
                        "candidates": [{
                            "content": { "parts": [{ "text": "ok" }] },
                            "finishReason": "STOP"
                        }],
                        "usageMetadata": {
                            "promptTokenCount": U::BASE_INPUT,
                            "candidatesTokenCount": U::BASE_OUTPUT,
                            "thoughtsTokenCount": U::REASONING
                        }
                    })),
                    Scenario::ReasoningExtraction => ProviderWire::body(json!({
                        "candidates": [{
                            "content": { "parts": [
                                { "text": "thinking about it", "thought": true },
                                { "text": "answer" }
                            ] },
                            "finishReason": "STOP"
                        }]
                    }))
                    .with_reasoning_text("thinking about it"),
                    Scenario::StreamingUsageMerge => return None,
                };
                Some(wire)
            }

            fn parts_from_wire(&self, body: &Value) -> Vec<LlmOutputPart> {
                GoogleOAuthProvider::response_parts_from_value(body, None)
            }

            fn usage_from_wire(&self, body: &Value) -> LlmUsage {
                // `usage_from_event` reads `event.response.usageMetadata`; the
                // unwrapped body carries `usageMetadata` at the top level, so
                // re-wrap it exactly as the non-streaming path does.
                let meta = body.get("usageMetadata").cloned().unwrap_or(Value::Null);
                GoogleOAuthProvider::usage_from_event(&json!({
                    "response": { "usageMetadata": meta }
                }))
            }

            fn terminal_from_wire(
                &self,
                body: &Value,
                parts: &[LlmOutputPart],
            ) -> LlmTerminalReason {
                GoogleOAuthProvider::terminal_reason_from_value(body, parts)
            }

            fn assemble_stream(&self, sse_events: &[String]) -> StreamAssembly {
                let mut full = String::new();
                let mut text_deltas = Vec::new();
                let mut usage = LlmUsage::default();
                let mut tool_calls = Vec::new();
                let mut finish_event = None;
                for raw in sse_events {
                    GoogleOAuthProvider::process_sse_event(
                        raw,
                        &mut full,
                        &mut text_deltas,
                        &mut usage,
                        Some(&mut tool_calls),
                        &mut finish_event,
                    )
                    .expect("google sse event parses");
                }
                let mut parts = text_deltas
                    .into_iter()
                    .map(|text| LlmOutputPart::Text {
                        text,
                        response_meta: None,
                    })
                    .collect::<Vec<_>>();
                parts.extend(tool_calls);
                StreamAssembly { parts, usage }
            }
        }

        #[test]
        fn google_satisfies_provider_conformance() {
            provider_conformance(&GoogleNormalizer);
        }
    }
}
