#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LlmToolSpec {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
    pub output_schema: serde_json::Value,
}

#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub enum LlmToolChoice {
    #[default]
    Auto,
    None,
    Required,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LlmOutputPart {
    Text {
        text: String,
    },
    /// Human-readable summary of the model's reasoning ("thinking"), as
    /// returned by providers that support reasoning summaries (e.g. Codex's
    /// `response.reasoning_summary_text.*` events). Display-only — the
    /// session replay layer must NOT resend this content to the model as
    /// assistant input; encrypted reasoning re-feeding is a separate path.
    Reasoning {
        text: String,
    },
    ToolCall {
        call_id: String,
        tool_name: String,
        input_json: String,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LlmRole {
    User,
    Assistant,
    System,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LlmMessage {
    pub role: LlmRole,
    pub content: String,
    pub kind: String,
    pub image_idx: i64,
    pub tool_call_id: Option<String>,
    pub tool_name: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LlmToolCall {
    pub call_id: String,
    pub tool_name: String,
    pub input_json: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LlmReplayChunk {
    Message(LlmMessage),
    AssistantToolCalls {
        text: Option<String>,
        tool_calls: Vec<LlmToolCall>,
    },
    ToolResults {
        results: Vec<LlmMessage>,
    },
}

pub fn coalesce_replay_messages(messages: &[LlmMessage]) -> Vec<LlmReplayChunk> {
    let mut out = Vec::new();
    let mut idx = 0;
    while idx < messages.len() {
        let msg = &messages[idx];

        if matches!(msg.role, LlmRole::Assistant) && msg.kind == "text" {
            let mut end = idx + 1;
            let mut tool_calls = Vec::new();
            while end < messages.len() {
                let next = &messages[end];
                if matches!(next.role, LlmRole::Assistant) && next.kind == "tool_call" {
                    tool_calls.push(LlmToolCall {
                        call_id: next.tool_call_id.clone().unwrap_or_default(),
                        tool_name: next.tool_name.clone().unwrap_or_default(),
                        input_json: next.content.clone(),
                    });
                    end += 1;
                } else {
                    break;
                }
            }
            if !tool_calls.is_empty() {
                out.push(LlmReplayChunk::AssistantToolCalls {
                    text: Some(msg.content.clone()),
                    tool_calls,
                });
                idx = end;
                continue;
            }
        }

        if matches!(msg.role, LlmRole::Assistant) && msg.kind == "tool_call" {
            let mut end = idx;
            let mut tool_calls = Vec::new();
            while end < messages.len() {
                let next = &messages[end];
                if matches!(next.role, LlmRole::Assistant) && next.kind == "tool_call" {
                    tool_calls.push(LlmToolCall {
                        call_id: next.tool_call_id.clone().unwrap_or_default(),
                        tool_name: next.tool_name.clone().unwrap_or_default(),
                        input_json: next.content.clone(),
                    });
                    end += 1;
                } else {
                    break;
                }
            }
            out.push(LlmReplayChunk::AssistantToolCalls {
                text: None,
                tool_calls,
            });
            idx = end;
            continue;
        }

        if msg.kind == "tool_result" {
            let mut end = idx;
            let mut results = Vec::new();
            while end < messages.len() {
                let next = &messages[end];
                if next.kind == "tool_result" {
                    results.push(next.clone());
                    end += 1;
                } else {
                    break;
                }
            }
            out.push(LlmReplayChunk::ToolResults { results });
            idx = end;
            continue;
        }

        out.push(LlmReplayChunk::Message(msg.clone()));
        idx += 1;
    }
    out
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LlmAttachment {
    pub mime: String,
    pub data: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LlmJsonSchema {
    pub name: String,
    pub schema: serde_json::Value,
    pub strict: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LlmOutputSpec {
    JsonObject,
    JsonSchema(LlmJsonSchema),
}

#[derive(Clone, Debug)]
pub struct LlmRequest {
    pub model: String,
    pub messages: Vec<LlmMessage>,
    pub attachments: Vec<LlmAttachment>,
    pub tools: Arc<Vec<LlmToolSpec>>,
    pub tool_choice: LlmToolChoice,
    pub model_variant: Option<String>,
    pub session_id: Option<String>,
    pub output_spec: Option<LlmOutputSpec>,
    pub stream_events: Option<LlmEventSender>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LlmUsage {
    pub input_tokens: i64,
    pub output_tokens: i64,
    pub cached_input_tokens: i64,
    pub reasoning_tokens: i64,
}

#[derive(Clone, Debug)]
pub enum LlmStreamEvent {
    Delta(String),
    /// Incremental reasoning-summary text. Kept separate from `Delta` so
    /// the UI can render it in a distinct muted/italic style rather than
    /// mixing it into the assistant's final text.
    ReasoningDelta(String),
    Part(LlmOutputPart),
    Usage(LlmUsage),
}

#[derive(Clone)]
pub struct LlmEventSender(Arc<dyn Fn(LlmStreamEvent) + Send + Sync>);

impl LlmEventSender {
    pub fn new<F>(send: F) -> Self
    where
        F: Fn(LlmStreamEvent) + Send + Sync + 'static,
    {
        Self(Arc::new(send))
    }

    pub fn send(&self, event: LlmStreamEvent) {
        (self.0)(event);
    }
}

impl std::fmt::Debug for LlmEventSender {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlmEventSender").finish_non_exhaustive()
    }
}

#[derive(Clone, Debug, Default)]
pub struct LlmResponse {
    pub full_text: String,
    pub deltas: Vec<String>,
    pub parts: Vec<LlmOutputPart>,
    pub usage: LlmUsage,
    pub provider_usage: Option<serde_json::Value>,
    pub request_body: Option<String>,
    pub http_summary: Option<String>,
}

#[derive(Clone, Debug)]
pub struct ModelSelection {
    pub model: &'static str,
    pub variant: Option<&'static str>,
}
use std::sync::Arc;
