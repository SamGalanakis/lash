use crate::{AttachmentRef, ToolCallRecord};

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ExecImage {
    pub mime: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference: Option<AttachmentRef>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub data: Vec<u8>,
    pub label: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub width: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub height: Option<u32>,
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct TextProjectionMetadata {
    pub truncated: bool,
    pub original_chars: usize,
    pub projected_chars: usize,
    pub original_lines: usize,
    pub projected_lines: usize,
    pub limit: usize,
    pub limit_mode: String,
    pub max_lines: usize,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ExecResponse {
    pub observations: Vec<String>,
    pub observation_truncation: Vec<TextProjectionMetadata>,
    pub tool_calls: Vec<ToolCallRecord>,
    pub images: Vec<ExecImage>,
    pub printed_images: Vec<AttachmentRef>,
    pub error: Option<String>,
    pub duration_ms: u64,
    /// When the surrounding session uses protocol-specific finish behavior,
    /// this carries the protocol's terminal value. The dispatch loop uses it
    /// as the terminal result of the session. `None` for chat-style sessions
    /// and for typed sessions whose step continued without finishing.
    pub terminal_finish: Option<serde_json::Value>,
}

/// Exact prompt-usage snapshot from the most recent completed LLM call.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct PromptUsage {
    pub prompt_context_tokens: usize,
    pub input_tokens: usize,
    pub cache_read_input_tokens: usize,
    pub cache_write_input_tokens: usize,
    pub context_budget_tokens: usize,
}
