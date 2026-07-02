//! Streaming and response parsing: extract usage, text, and tool-call parts
//! from Cloud Code (Gemini) events, accumulate streamed text, and map the
//! `finishReason` to a normalized terminal reason.

use crate::support::*;

impl GoogleOAuthProvider {
    pub(crate) fn usage_from_event(event: &Value) -> LlmUsage {
        let meta = event
            .get("response")
            .and_then(|r| r.get("usageMetadata"))
            .unwrap_or(&Value::Null);
        let prompt_tokens = parse_i64(
            meta.get("promptTokenCount")
                .or_else(|| meta.get("inputTokenCount"))
                .or_else(|| meta.get("inputTokens")),
        );
        let cache_read = parse_i64(
            meta.get("cachedContentTokenCount")
                .or_else(|| meta.get("cachedPromptTokenCount"))
                .or_else(|| meta.get("cachedInputTokenCount")),
        );
        let candidate_tokens = parse_i64(
            meta.get("candidatesTokenCount")
                .or_else(|| meta.get("outputTokenCount"))
                .or_else(|| meta.get("outputTokens")),
        );
        let reasoning = parse_i64(
            meta.get("thoughtsTokenCount")
                .or_else(|| meta.get("reasoningTokenCount"))
                .or_else(|| meta.get("reasoningTokens")),
        );
        LlmUsage {
            input_tokens: prompt_tokens.saturating_sub(cache_read).max(0),
            output_tokens: candidate_tokens.saturating_add(reasoning),
            cache_read_input_tokens: cache_read,
            cache_write_input_tokens: 0,
            reasoning_output_tokens: reasoning,
        }
    }

    fn text_parts_from_event(event: &Value) -> Vec<(String, Option<String>)> {
        let mut out = Vec::new();
        let Some(candidates) = event
            .get("response")
            .and_then(|r| r.get("candidates"))
            .and_then(|c| c.as_array())
        else {
            return out;
        };

        for candidate in candidates {
            let Some(parts) = candidate
                .get("content")
                .and_then(|c| c.get("parts"))
                .and_then(|p| p.as_array())
            else {
                continue;
            };
            for part in parts {
                // Skip reasoning text (Gemini marks it with `thought:true`);
                // it isn't assistant-visible output and shouldn't accumulate
                // into `full`. The signature ride-along is captured when
                // we finalize the response.
                if part.get("thought").and_then(|v| v.as_bool()) == Some(true) {
                    continue;
                }
                if let Some(text) = part.get("text").and_then(|t| t.as_str())
                    && !text.is_empty()
                {
                    let signature = part
                        .get("thoughtSignature")
                        .and_then(|v| v.as_str())
                        .filter(|s| !s.is_empty())
                        .map(str::to_string);
                    out.push((text.to_string(), signature));
                }
            }
        }

        out
    }

    fn tool_call_parts_from_event(event: &Value) -> Vec<LlmOutputPart> {
        let mut out = Vec::new();
        let Some(candidates) = event
            .get("response")
            .and_then(|r| r.get("candidates"))
            .and_then(|c| c.as_array())
        else {
            return out;
        };
        for candidate in candidates {
            let Some(parts) = candidate
                .get("content")
                .and_then(|c| c.get("parts"))
                .and_then(|p| p.as_array())
            else {
                continue;
            };
            for part in parts {
                if let Some(function_call) = part.get("functionCall") {
                    let Some(name) = function_call.get("name").and_then(|v| v.as_str()) else {
                        continue;
                    };
                    let input_json = function_call
                        .get("args")
                        .map(|v| v.to_string())
                        .unwrap_or_else(|| "{}".to_string());
                    // Capture `thoughtSignature` (if present) alongside
                    // the functionCall. Gemini 3 will reject the next
                    // turn if we don't echo it back.
                    let signature = part
                        .get("thoughtSignature")
                        .and_then(|v| v.as_str())
                        .filter(|s| !s.is_empty())
                        .map(str::to_string);
                    out.push(LlmOutputPart::ToolCall {
                        call_id: function_call
                            .get("id")
                            .and_then(|v| v.as_str())
                            .map(str::to_string)
                            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
                        tool_name: name.to_string(),
                        input_json,
                        replay: signature.map(|opaque| ProviderReplayMeta {
                            item_id: None,
                            opaque: Some(opaque),
                        }),
                    });
                }
            }
        }
        out
    }

    fn apply_stream_piece(
        full: &mut String,
        text_deltas: &mut Vec<String>,
        piece: &str,
    ) -> Option<String> {
        if piece.is_empty() {
            return None;
        }
        if piece.starts_with(full.as_str()) {
            let delta = &piece[full.len()..];
            if !delta.is_empty() {
                full.push_str(delta);
                text_deltas.push(delta.to_string());
                return Some(delta.to_string());
            }
            return None;
        }
        full.push_str(piece);
        text_deltas.push(piece.to_string());
        Some(piece.to_string())
    }

    #[cfg(test)]
    pub(crate) fn process_sse_event(
        raw: &str,
        full: &mut String,
        text_deltas: &mut Vec<String>,
        usage: &mut LlmUsage,
        tool_call_parts: Option<&mut Vec<LlmOutputPart>>,
        finish_event: &mut Option<Value>,
    ) -> Result<(), LlmTransportError> {
        let mut provider_usage = None;
        Self::process_sse_event_with_text_parts(
            raw,
            SseTextPartSink {
                full,
                text_deltas,
                usage,
                provider_usage: &mut provider_usage,
                tool_call_parts,
                text_parts: None,
                finish_event,
            },
            None,
        )
    }

    pub(crate) fn process_sse_event_with_text_parts(
        raw: &str,
        sink: SseTextPartSink<'_>,
        origin_model: Option<&str>,
    ) -> Result<(), LlmTransportError> {
        let SseTextPartSink {
            full,
            text_deltas,
            usage,
            provider_usage,
            tool_call_parts,
            text_parts,
            finish_event,
        } = sink;
        if raw.trim().is_empty() || raw.trim() == "[DONE]" {
            return Ok(());
        }
        let event: Value = serde_json::from_str(raw)
            .map_err(|e| LlmTransportError::new(format!("Invalid Cloud Code SSE payload: {e}")))?;
        let new_usage = Self::usage_from_event(&event);
        if new_usage.input_tokens > 0
            || new_usage.output_tokens > 0
            || new_usage.cache_read_input_tokens > 0
            || new_usage.cache_write_input_tokens > 0
            || new_usage.reasoning_output_tokens > 0
        {
            *usage = new_usage;
            // Keep the raw `usageMetadata` block alongside the normalized
            // counters, under the same non-zero guard so a trailing empty
            // block cannot clobber the captured sidecar.
            *provider_usage = event
                .get("response")
                .and_then(|response| response.get("usageMetadata"))
                .cloned();
        }
        let mut text_parts = text_parts;
        for (piece, signature) in Self::text_parts_from_event(&event) {
            let Some(delta) = Self::apply_stream_piece(full, text_deltas, &piece) else {
                continue;
            };
            if let Some(parts) = text_parts.as_deref_mut() {
                parts.push(LlmOutputPart::Text {
                    text: delta,
                    response_meta: signature.map(|signature| ResponseTextMeta {
                        provider_payload: Some(signature),
                        origin_provider: Some(Self::PROVIDER_KIND.to_string()),
                        origin_model: origin_model.map(str::to_string),
                        ..ResponseTextMeta::default()
                    }),
                });
            }
        }
        if let Some(parts) = tool_call_parts {
            parts.extend(Self::tool_call_parts_from_event(&event));
        }
        // Capture the last event carrying a non-empty `finishReason` so the
        // streaming finalizer can derive the terminal reason exactly like the
        // non-streaming path instead of hardcoding Stop.
        if Self::finish_reason_str(&event).is_some() {
            *finish_event = Some(event);
        }
        Ok(())
    }

    /// The non-empty `finishReason` carried by the first candidate of an event,
    /// honouring the streaming `response.candidates` wrapper as well as the
    /// unwrapped top-level shape.
    fn finish_reason_str(value: &Value) -> Option<&str> {
        value
            .get("candidates")
            .and_then(Value::as_array)
            .and_then(|candidates| candidates.first())
            .and_then(|candidate| candidate.get("finishReason"))
            .or_else(|| {
                value
                    .get("response")
                    .and_then(|response| response.get("candidates"))
                    .and_then(Value::as_array)
                    .and_then(|candidates| candidates.first())
                    .and_then(|candidate| candidate.get("finishReason"))
            })
            .and_then(Value::as_str)
            .filter(|reason| !reason.is_empty())
    }

    pub(crate) fn response_parts_from_value(
        value: &Value,
        origin_model: Option<&str>,
    ) -> Vec<LlmOutputPart> {
        let mut parts = Vec::new();
        let Some(candidates) = value.get("candidates").and_then(|c| c.as_array()) else {
            return parts;
        };
        for candidate in candidates {
            let Some(items) = candidate
                .get("content")
                .and_then(|c| c.get("parts"))
                .and_then(|p| p.as_array())
            else {
                continue;
            };
            for item in items {
                let signature = item
                    .get("thoughtSignature")
                    .and_then(|v| v.as_str())
                    .filter(|s| !s.is_empty())
                    .map(str::to_string);
                let is_thought = item
                    .get("thought")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);

                if let Some(text) = item.get("text").and_then(|t| t.as_str())
                    && !text.is_empty()
                {
                    if is_thought {
                        // Gemini flags reasoning text with `thought: true`.
                        // Route those into Reasoning so downstream code
                        // doesn't show them as assistant prose. Signature
                        // lives on the same part.
                        parts.push(LlmOutputPart::Reasoning {
                            text: text.to_string(),
                            replay: signature.clone().map(|signature| ProviderReasoningReplay {
                                item_id: None,
                                encrypted_content: None,
                                signature: Some(signature),
                                redacted: false,
                                summary: Vec::new(),
                            }),
                        });
                    } else {
                        parts.push(LlmOutputPart::Text {
                            text: text.to_string(),
                            response_meta: signature.clone().map(|signature| ResponseTextMeta {
                                provider_payload: Some(signature),
                                origin_provider: Some(Self::PROVIDER_KIND.to_string()),
                                origin_model: origin_model.map(str::to_string),
                                ..ResponseTextMeta::default()
                            }),
                        });
                    }
                }
                if let Some(function_call) = item.get("functionCall") {
                    let Some(name) = function_call.get("name").and_then(|v| v.as_str()) else {
                        continue;
                    };
                    let input_json = function_call
                        .get("args")
                        .map(|v| v.to_string())
                        .unwrap_or_else(|| "{}".to_string());
                    parts.push(LlmOutputPart::ToolCall {
                        call_id: function_call
                            .get("id")
                            .and_then(|v| v.as_str())
                            .map(str::to_string)
                            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
                        tool_name: name.to_string(),
                        input_json,
                        replay: signature.clone().map(|opaque| ProviderReplayMeta {
                            item_id: None,
                            opaque: Some(opaque),
                        }),
                    });
                }
            }
        }
        parts
    }

    pub(crate) fn terminal_reason_from_value(
        value: &Value,
        parts: &[LlmOutputPart],
    ) -> LlmTerminalReason {
        let finish = Self::finish_reason_str(value).unwrap_or("");
        match finish {
            "STOP" => LlmTerminalReason::Stop,
            "MAX_TOKENS" => LlmTerminalReason::OutputLimit,
            "SAFETY"
            | "RECITATION"
            | "BLOCKLIST"
            | "PROHIBITED_CONTENT"
            | "SPII"
            | "IMAGE_SAFETY"
            | "IMAGE_PROHIBITED_CONTENT"
            | "IMAGE_RECITATION"
            | "IMAGE_OTHER"
            | "LANGUAGE" => LlmTerminalReason::ContentFilter,
            "MALFORMED_FUNCTION_CALL"
            | "UNEXPECTED_TOOL_CALL"
            | "FINISH_REASON_UNSPECIFIED"
            | "OTHER"
            | "NO_IMAGE" => LlmTerminalReason::ProviderError,
            "" => terminal_reason_from_parts(parts),
            _ => LlmTerminalReason::ProviderError,
        }
    }
}
