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
        response_meta: Option<ResponseTextMeta>,
    },
    /// Model "thinking" / reasoning output from providers that expose a
    /// chain-of-thought channel (Anthropic `thinking`, Codex Responses API
    /// `reasoning`).
    ///
    /// * `text` — human-readable summary for display.
    /// * `signature` — Anthropic thinking signature (base64). Required for
    ///   replaying a `thinking` block so Anthropic accepts the next turn.
    /// * `redacted` — Anthropic `redacted_thinking` marker; when set,
    ///   `signature` carries the opaque `data` payload.
    /// * `item_id` — Codex reasoning item id (`rs_...`). Pairs with the
    ///   sibling `ToolCall.item_id` so re-emission links them.
    /// * `encrypted_content` — Codex opaque chain-of-thought blob. Required
    ///   for re-feed (display-only reasoning has `None` and must be
    ///   dropped by adapters rather than re-sent).
    /// * `summary` — Codex `summary[*].text` list, preserved verbatim.
    Reasoning {
        text: String,
        signature: Option<String>,
        redacted: bool,
        item_id: Option<String>,
        encrypted_content: Option<String>,
        summary: Vec<String>,
    },
    ToolCall {
        call_id: String,
        tool_name: String,
        input_json: String,
        /// Provider-specific item identifier (Codex Responses API `fc_...`
        /// id, Anthropic `toolu_...`). Used to pair a tool call with its
        /// sibling reasoning item across turns.
        item_id: Option<String>,
        /// Opaque provider-side signature to round-trip alongside this
        /// tool call. Gemini stores `thoughtSignature` here (required by
        /// Gemini 3 when thinking mode is active). OpenRouter stores a
        /// serialized `reasoning_details` entry so the next turn can be
        /// re-posted with the encrypted CoT intact.
        signature: Option<String>,
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
    },
    /// Index into the enclosing `LlmRequest.attachments` vector. User-role
    /// messages may embed images; adapters drop them for providers that
    /// don't accept vision input.
    Image { attachment_idx: usize },
    /// Assistant tool call. `item_id` carries provider-specific pairing ids
    /// (Codex `fc_...`, Anthropic `toolu_...`).
    ToolCall {
        call_id: String,
        tool_name: String,
        input_json: String,
        item_id: Option<String>,
        /// Matches [`LlmContentBlock::ToolCall::signature`]. See field
        /// docs there for the per-provider meaning.
        signature: Option<String>,
    },
    /// User tool-result block. Anthropic allows multiple per user turn;
    /// adapters that want one-per-message split as needed.
    ToolResult {
        call_id: String,
        content: String,
        /// Name of the tool that produced this result. Gemini's
        /// `functionResponse` requires this on replay; other providers
        /// ignore it.
        tool_name: Option<String>,
    },
    /// Chain-of-thought / reasoning block. See [`LlmOutputPart::Reasoning`]
    /// for field semantics. Adapters that don't support reasoning replay
    /// drop these blocks silently.
    Reasoning {
        text: String,
        signature: Option<String>,
        redacted: bool,
        item_id: Option<String>,
        encrypted_content: Option<String>,
        summary: Vec<String>,
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
