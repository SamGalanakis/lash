use std::sync::Arc;

use crate::{AttachmentRef, SchemaProjectionOverride};

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseTextPhase {
    Commentary,
    FinalAnswer,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ResponseTextMeta {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phase: Option<ResponseTextPhase>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LlmToolSpec {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
    pub output_schema: serde_json::Value,
    pub input_schema_projections: Vec<SchemaProjectionOverride>,
    pub output_schema_projections: Vec<SchemaProjectionOverride>,
}

#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub enum LlmToolChoice {
    #[default]
    Auto,
    None,
    Required,
}

#[derive(Clone, Debug, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub struct ProviderReplayMeta {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub item_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub opaque: Option<String>,
}

impl ProviderReplayMeta {
    pub fn is_empty(&self) -> bool {
        self.item_id.is_none() && self.opaque.is_none()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub struct ProviderReasoningReplay {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub item_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub encrypted_content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub redacted: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub summary: Vec<String>,
}

impl ProviderReasoningReplay {
    pub fn is_empty(&self) -> bool {
        self.item_id.is_none()
            && self.encrypted_content.is_none()
            && self.signature.is_none()
            && !self.redacted
            && self.summary.is_empty()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LlmOutputPart {
    Text {
        text: String,
        response_meta: Option<ResponseTextMeta>,
    },
    /// Model "thinking" / reasoning output from providers that expose a
    /// chain-of-thought channel.
    ///
    /// * `text` — human-readable summary for display.
    /// * `replay` — opaque provider replay state. Provider crates decide
    ///   how to map it back to their wire format on the next turn.
    Reasoning {
        text: String,
        replay: Option<ProviderReasoningReplay>,
    },
    ToolCall {
        call_id: String,
        tool_name: String,
        input_json: String,
        /// Opaque provider replay state. Core may use `item_id` for stable
        /// correlation, but provider crates own the wire semantics.
        replay: Option<ProviderReplayMeta>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LlmRole {
    User,
    Assistant,
    System,
}

/// A structured content block inside an `LlmMessage`. Mirrors pi-mono's
/// per-provider block types and maps cleanly onto each wire format so the
/// adapters can emit the right shape without re-coalescing flat messages.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LlmContentBlock {
    Text {
        text: Arc<str>,
        response_meta: Option<ResponseTextMeta>,
        cache_breakpoint: bool,
    },
    /// Index into the enclosing `LlmRequest.attachments` vector. User-role
    /// messages may embed images; adapters drop them for providers that
    /// don't accept vision input.
    Image { attachment_idx: usize },
    /// Assistant tool call with optional opaque provider replay state.
    ToolCall {
        call_id: String,
        tool_name: String,
        input_json: String,
        replay: Option<ProviderReplayMeta>,
    },
    /// User tool-result block. Some providers allow multiple per user turn;
    /// adapters that want one-per-message split as needed.
    ToolResult {
        call_id: String,
        content: String,
        /// Name of the tool that produced this result. Some provider replay
        /// formats require this; others ignore it.
        tool_name: Option<String>,
    },
    /// Chain-of-thought / reasoning block. See [`LlmOutputPart::Reasoning`]
    /// for field semantics. Adapters that don't support reasoning replay
    /// drop these blocks silently.
    Reasoning {
        text: String,
        replay: Option<ProviderReasoningReplay>,
    },
}

/// A single role turn in the LLM conversation. `blocks` holds structured
/// content that maps 1:1 onto provider wire types. The old flat
/// `content: String` + `kind` discriminator has been retired in favor of
/// this block model.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LlmMessage {
    pub role: LlmRole,
    pub blocks: Arc<Vec<LlmContentBlock>>,
}

impl LlmMessage {
    pub fn new(role: LlmRole, blocks: Vec<LlmContentBlock>) -> Self {
        Self {
            role,
            blocks: Arc::new(blocks),
        }
    }

    /// Convenience constructor for a single-text-block message.
    pub fn text(role: LlmRole, text: impl Into<Arc<str>>) -> Self {
        Self {
            role,
            blocks: Arc::new(vec![LlmContentBlock::Text {
                text: text.into(),
                response_meta: None,
                cache_breakpoint: false,
            }]),
        }
    }

    /// True if every block is a `Text` whose content is whitespace-only.
    pub fn is_blank(&self) -> bool {
        self.blocks.iter().all(|b| match b {
            LlmContentBlock::Text { text, .. } => text.trim().is_empty(),
            _ => false,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LlmAttachment {
    pub mime: String,
    pub data: Vec<u8>,
    pub reference: Option<AttachmentRef>,
}

impl LlmAttachment {
    pub fn bytes(mime: impl Into<String>, data: Vec<u8>) -> Self {
        Self {
            mime: mime.into(),
            data,
            reference: None,
        }
    }

    pub fn reference(reference: AttachmentRef) -> Self {
        Self {
            mime: reference.canonical_mime().to_string(),
            data: Vec::new(),
            reference: Some(reference),
        }
    }

    pub fn is_resolved(&self) -> bool {
        !self.data.is_empty() || self.reference.is_none()
    }
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
    pub provider_trace: Option<LlmProviderTraceSender>,
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
    RetryStatus {
        wait_seconds: u64,
        attempt: usize,
        max_attempts: usize,
        reason: String,
    },
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

#[derive(Clone, Debug)]
pub struct LlmProviderTraceEvent {
    pub provider: &'static str,
    pub event_name: String,
    pub raw: String,
}

#[derive(Clone)]
pub struct LlmProviderTraceSender(Arc<dyn Fn(LlmProviderTraceEvent) + Send + Sync>);

impl LlmProviderTraceSender {
    pub fn new<F>(send: F) -> Self
    where
        F: Fn(LlmProviderTraceEvent) + Send + Sync + 'static,
    {
        Self(Arc::new(send))
    }

    pub fn send(&self, event: LlmProviderTraceEvent) {
        (self.0)(event);
    }
}

impl std::fmt::Debug for LlmProviderTraceSender {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlmProviderTraceSender")
            .finish_non_exhaustive()
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
