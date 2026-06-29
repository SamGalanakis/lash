use std::fmt;
use std::sync::Arc;

use lash_core::provider::ProviderHandle;
use lash_llm_transport::LlmHttpTransport;
use lash_provider_anthropic::AnthropicProvider;
use lash_provider_google::GoogleOAuthProvider;
use lash_provider_openai::{OpenAiCompatibleProvider, OpenAiProvider};
use serde_json::{Value, json};

use crate::provider::{ProviderWireScript, ScriptedLlmHttpTransport};

pub const OPENAI_COMPATIBLE: &str = "openai-compatible";
pub const OPENAI: &str = "openai";
pub const ANTHROPIC: &str = "anthropic";
pub const GOOGLE_OAUTH: &str = "google_oauth";

pub const MIGRATED_RUNTIME_PROVIDER_KINDS: &[&str] =
    &[OPENAI_COMPATIBLE, OPENAI, ANTHROPIC, GOOGLE_OAUTH];

const OPENAI_COMPAT_RUNTIME_TEXT: &str =
    include_str!("../provider-scripts/runtime/openai-compatible.chat-runtime-text-stream.json");
const OPENAI_RESPONSES_TEXT: &str =
    include_str!("../provider-scripts/canonical/openai.responses-text-stream.json");
const ANTHROPIC_MESSAGES_TEXT: &str =
    include_str!("../provider-scripts/canonical/anthropic.messages-text-stream.json");
const GOOGLE_STREAM_GENERATE_TEXT: &str =
    include_str!("../provider-scripts/canonical/google.stream-generate-content-text-stream.json");

#[derive(Debug)]
pub struct RuntimeProviderError {
    message: String,
}

impl RuntimeProviderError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for RuntimeProviderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for RuntimeProviderError {}

impl From<serde_json::Error> for RuntimeProviderError {
    fn from(value: serde_json::Error) -> Self {
        Self::new(value.to_string())
    }
}

impl From<lash_core::LlmTransportError> for RuntimeProviderError {
    fn from(value: lash_core::LlmTransportError) -> Self {
        Self::new(value.to_string())
    }
}

pub fn runtime_provider_kind_for_session(session_index: usize) -> &'static str {
    MIGRATED_RUNTIME_PROVIDER_KINDS[session_index % MIGRATED_RUNTIME_PROVIDER_KINDS.len()]
}

pub fn runtime_script_name_for_kind(
    provider_kind: &str,
) -> Result<&'static str, RuntimeProviderError> {
    Ok(match provider_kind {
        OPENAI_COMPATIBLE => "openai-compatible.chat-runtime-text-stream",
        OPENAI => "openai.responses-text-stream",
        ANTHROPIC => "anthropic.messages-text-stream",
        GOOGLE_OAUTH => "google.stream-generate-content-text-stream",
        other => {
            return Err(RuntimeProviderError::new(format!(
                "unsupported generated runtime provider `{other}`"
            )));
        }
    })
}

pub fn runtime_scripts_for_texts(
    provider_kind: &str,
    texts: &[String],
) -> Result<Vec<ProviderWireScript>, RuntimeProviderError> {
    texts
        .iter()
        .map(|text| runtime_script_for_text(provider_kind, text))
        .collect()
}

pub fn runtime_script_for_text(
    provider_kind: &str,
    text: &str,
) -> Result<ProviderWireScript, RuntimeProviderError> {
    let mut script: Value = match provider_kind {
        OPENAI_COMPATIBLE => serde_json::from_str(OPENAI_COMPAT_RUNTIME_TEXT)?,
        OPENAI => serde_json::from_str(OPENAI_RESPONSES_TEXT)?,
        ANTHROPIC => serde_json::from_str(ANTHROPIC_MESSAGES_TEXT)?,
        GOOGLE_OAUTH => serde_json::from_str(GOOGLE_STREAM_GENERATE_TEXT)?,
        other => {
            return Err(RuntimeProviderError::new(format!(
                "unsupported generated runtime provider `{other}`"
            )));
        }
    };

    match provider_kind {
        OPENAI_COMPATIBLE => {
            set_sse_data(
                &mut script,
                1,
                json!({
                    "choices": [
                        {
                            "delta": {
                                "content": text,
                            }
                        }
                    ]
                }),
            )?;
        }
        OPENAI => {
            set_sse_data(
                &mut script,
                2,
                json!({
                    "type": "response.output_text.delta",
                    "output_index": 0,
                    "item_id": "msg_1",
                    "delta": text,
                }),
            )?;
            let done_item = json!({
                "type": "message",
                "id": "msg_1",
                "status": "completed",
                "phase": "final_answer",
                "content": [{"type": "output_text", "text": text}],
            });
            set_sse_data(
                &mut script,
                3,
                json!({
                    "type": "response.output_item.done",
                    "output_index": 0,
                    "item": done_item,
                }),
            )?;
            set_sse_data(
                &mut script,
                4,
                json!({
                    "type": "response.completed",
                    "response": {
                        "id": "resp_1",
                        "status": "completed",
                        "output": [done_item],
                        "usage": {"input_tokens": 5, "output_tokens": 2},
                    }
                }),
            )?;
        }
        ANTHROPIC => {
            set_sse_data(
                &mut script,
                3,
                json!({
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": text},
                }),
            )?;
        }
        GOOGLE_OAUTH => {
            if let Some(body) = script
                .pointer_mut("/request_match/body")
                .and_then(Value::as_object_mut)
            {
                body.remove("request.contents[0].parts[0].text");
                body.remove("request.sessionId");
            }
            set_sse_data(
                &mut script,
                1,
                json!({
                    "response": {
                        "candidates": [
                            {
                                "content": {
                                    "parts": [
                                        {
                                            "text": text,
                                        }
                                    ]
                                }
                            }
                        ],
                        "usageMetadata": {
                            "promptTokenCount": 6,
                            "candidatesTokenCount": 3,
                            "thoughtsTokenCount": 1,
                        }
                    }
                }),
            )?;
        }
        _ => unreachable!("provider kind was validated above"),
    }

    script["expected_provider"]["text"] = Value::String(text.to_string());
    let encoded = serde_json::to_string(&script)?;
    Ok(ProviderWireScript::from_json_str(&encoded)?)
}

/// A runtime provider wire script whose mid-stream content chunk is a
/// non-retryable MALFORMED SSE payload, so a live runtime turn that consumes it
/// terminalizes with a terminal provider parser failure and commits no output.
/// The request match is left intact (identical to the happy-path runtime script),
/// so a real `session.turn().run()` matches it and the failure is delivered
/// through the real provider wire parser — not an isolated `provider.complete()`.
pub fn runtime_malformed_sse_script(
    provider_kind: &str,
) -> Result<ProviderWireScript, RuntimeProviderError> {
    // The content SSE chunk index differs per provider; only openai-compatible is
    // needed for the live-failure proof and its content delta is timeline[1].
    let (raw, content_index) = match provider_kind {
        OPENAI_COMPATIBLE => (OPENAI_COMPAT_RUNTIME_TEXT, 1usize),
        other => {
            return Err(RuntimeProviderError::new(format!(
                "malformed runtime script not supported for provider `{other}`"
            )));
        }
    };
    let mut script: Value = serde_json::from_str(raw)?;
    let slot = script
        .get_mut("timeline")
        .and_then(Value::as_array_mut)
        .and_then(|timeline| timeline.get_mut(content_index))
        .and_then(|event| event.get_mut("data"))
        .ok_or_else(|| {
            RuntimeProviderError::new(format!(
                "runtime script missing timeline[{content_index}].data to malform"
            ))
        })?;
    // Invalid JSON SSE chunk: the production provider parser rejects this as a
    // terminal (non-retryable) parser error mid-stream.
    *slot = Value::String("{ malformed provider event".to_string());
    script["expected_provider"] = json!({
        "mutation": "malformed_sse_chunk",
        "expected": "provider parser error",
    });
    let encoded = serde_json::to_string(&script)?;
    Ok(ProviderWireScript::from_json_str(&encoded)?)
}

/// Two real openai-compatible provider wire scripts for a suspend-roundtrip
/// session: the first streams a native tool call for `tool_name` (which parks the
/// live turn on the await key), the second streams the final `resumed` answer
/// after the scheduler resolves the await. Routing these through the real
/// `ScriptedLlmHttpTransport` (rather than a `TestProvider`) means the parked turn
/// also exercises real provider wire parsing on both exchanges.
pub fn suspend_roundtrip_scripts(
    tool_name: &str,
) -> Result<Vec<ProviderWireScript>, RuntimeProviderError> {
    let tool_call_delta = json!({
        "choices": [{
            "delta": {
                "tool_calls": [{
                    "index": 0,
                    "id": "suspend-call-1",
                    "type": "function",
                    "function": { "name": tool_name, "arguments": "{}" }
                }]
            }
        }]
    })
    .to_string();
    let finish_delta = json!({
        "choices": [{ "finish_reason": "tool_calls", "delta": {} }]
    })
    .to_string();
    let tool_call_script = json!({
        "schema": "lash.provider-wire-script.v1",
        "name": format!("openai-compatible.chat-suspend-{tool_name}"),
        "provider_kind": "openai-compatible",
        "endpoint": { "method": "POST", "path": "/chat/completions" },
        "request_match": {
            "body": {
                "model": { "equals": "openai/gpt-5.4" },
                "stream": { "equals": true },
                "messages": { "contains_role": "user" }
            },
            "headers": {
                "authorization": { "present": true },
                "content-type": { "contains": "application/json" }
            }
        },
        "timeline": [
            {
                "at": 10,
                "event": "response_start",
                "status": 200,
                "headers": [
                    { "name": "content-type", "value": "text/event-stream; charset=utf-8" },
                    { "name": "x-request-id", "value": "req-suspend-tool" }
                ]
            },
            { "at": 20, "event": "sse", "data": tool_call_delta },
            { "at": 21, "event": "sse", "data": finish_delta },
            { "at": 22, "event": "sse", "data": "[DONE]" },
            { "at": 23, "event": "end" }
        ],
        "expected_provider": { "terminal_reason": "tool_calls" }
    });
    let tool_call_script = ProviderWireScript::from_json_str(&tool_call_script.to_string())?;
    let resumed_script = runtime_script_for_text(OPENAI_COMPATIBLE, "resumed")?;
    Ok(vec![tool_call_script, resumed_script])
}

pub fn runtime_provider_components(
    provider_kind: &str,
    transport: &Arc<ScriptedLlmHttpTransport>,
) -> Result<(ProviderHandle, lash::ModelSpec, String), RuntimeProviderError> {
    let transport = provider_transport(transport);
    let (provider, model_name): (ProviderHandle, &str) = match provider_kind {
        OPENAI_COMPATIBLE => {
            let provider = OpenAiCompatibleProvider::new("test-key", "https://provider.test")
                .with_transport(transport);
            (
                ProviderHandle::new(provider.into_components()),
                "openai/gpt-5.4",
            )
        }
        OPENAI => {
            let provider = OpenAiProvider::new("test-key").with_transport(transport);
            (ProviderHandle::new(provider.into_components()), "gpt-5.4")
        }
        ANTHROPIC => {
            let provider = AnthropicProvider::new("test-key")
                .with_base_url(Some("https://anthropic.test".to_string()))
                .with_transport(transport);
            (
                ProviderHandle::new(provider.into_components()),
                "claude-sonnet-4-20250514",
            )
        }
        GOOGLE_OAUTH => {
            let provider = GoogleOAuthProvider::new("access-token", "refresh-token", 0)
                .with_project_id(Some("project-1".to_string()))
                .with_transport(transport);
            (
                ProviderHandle::new(provider.into_components()),
                "gemini-3.1-pro-preview",
            )
        }
        other => {
            return Err(RuntimeProviderError::new(format!(
                "unsupported generated runtime provider `{other}`"
            )));
        }
    };
    let model = lash::ModelSpec::from_token_limits(model_name, None, 200_000, None)
        .map_err(|err| RuntimeProviderError::new(err.to_string()))?;
    Ok((provider, model, provider_kind.to_string()))
}

fn set_sse_data(
    script: &mut Value,
    timeline_index: usize,
    data: Value,
) -> Result<(), RuntimeProviderError> {
    let slot = script
        .get_mut("timeline")
        .and_then(Value::as_array_mut)
        .and_then(|timeline| timeline.get_mut(timeline_index))
        .and_then(|event| event.get_mut("data"))
        .ok_or_else(|| {
            RuntimeProviderError::new(format!(
                "provider runtime script missing timeline[{timeline_index}].data"
            ))
        })?;
    *slot = Value::String(data.to_string());
    Ok(())
}

fn provider_transport(transport: &Arc<ScriptedLlmHttpTransport>) -> Arc<dyn LlmHttpTransport> {
    transport.clone()
}
