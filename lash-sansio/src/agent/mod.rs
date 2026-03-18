pub mod exec;
pub mod message;
pub mod prompt;

pub use message::{
    Message, MessageRole, Part, PartKind, PruneState, render_prompt, render_transcript_prompt,
};
pub use prompt::{
    DefaultPromptRenderer, PromptOverrideMode, PromptRenderer, PromptSectionName,
    PromptSectionOverride, default_prompt_renderer,
};

use crate::ToolDefinition;
use crate::llm::types::LlmToolSpec;
use crate::plugin::{CheckpointKind, PluginMessage, PluginSurfaceEvent};

/// Token usage statistics from an LLM call.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct TokenUsage {
    pub input_tokens: i64,
    pub output_tokens: i64,
    pub cached_input_tokens: i64,
    #[serde(default)]
    pub reasoning_tokens: i64,
}

impl TokenUsage {
    pub fn total(&self) -> i64 {
        self.input_tokens + self.output_tokens + self.reasoning_tokens
    }

    pub fn context_total(&self) -> i64 {
        self.total() + self.cached_input_tokens
    }

    pub fn add(&mut self, other: &TokenUsage) {
        self.input_tokens += other.input_tokens;
        self.output_tokens += other.output_tokens;
        self.cached_input_tokens += other.cached_input_tokens;
        self.reasoning_tokens += other.reasoning_tokens;
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ErrorEnvelope {
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
    pub user_message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize)]
#[serde(tag = "type")]
pub enum AgentEvent {
    #[serde(rename = "text_delta")]
    TextDelta { content: String },
    #[serde(rename = "tool_call")]
    ToolCall {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        call_id: Option<String>,
        name: String,
        args: serde_json::Value,
        result: serde_json::Value,
        success: bool,
        duration_ms: u64,
    },
    #[serde(rename = "code_block")]
    CodeBlock { code: String },
    #[serde(rename = "code_output")]
    CodeOutput {
        output: String,
        error: Option<String>,
    },
    #[serde(rename = "message")]
    Message { text: String, kind: String },
    #[serde(rename = "llm_request")]
    LlmRequest {
        iteration: usize,
        message_count: usize,
        tool_list: String,
    },
    #[serde(rename = "llm_response")]
    LlmResponse {
        iteration: usize,
        content: String,
        duration_ms: u64,
    },
    #[serde(rename = "token_usage")]
    TokenUsage {
        iteration: usize,
        usage: TokenUsage,
        cumulative: TokenUsage,
    },
    #[serde(rename = "retry_status")]
    RetryStatus {
        wait_seconds: u64,
        attempt: usize,
        max_attempts: usize,
        reason: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        envelope: Option<ErrorEnvelope>,
    },
    #[serde(rename = "injected_messages_committed")]
    InjectedMessagesCommitted {
        messages: Vec<PluginMessage>,
        checkpoint: CheckpointKind,
    },
    #[serde(rename = "plugin_event")]
    PluginEvent {
        plugin_id: String,
        event: PluginSurfaceEvent,
    },
    #[serde(rename = "done")]
    Done,
    #[serde(rename = "error")]
    Error {
        message: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        envelope: Option<ErrorEnvelope>,
    },
    #[serde(rename = "prompt")]
    Prompt {
        question: String,
        options: Vec<String>,
        #[serde(skip)]
        response_tx: std::sync::mpsc::Sender<String>,
    },
}

pub const LLM_MAX_RETRIES: usize = 3;
pub const LLM_RETRY_DELAYS: [std::time::Duration; 3] = [
    std::time::Duration::from_secs(2),
    std::time::Duration::from_secs(5),
    std::time::Duration::from_secs(10),
];

pub struct TurnTerminationPolicyState {
    max_steps_final: bool,
}

impl Default for TurnTerminationPolicyState {
    fn default() -> Self {
        Self::new()
    }
}

impl TurnTerminationPolicyState {
    pub fn new() -> Self {
        Self {
            max_steps_final: false,
        }
    }

    pub fn should_force_exit_after_grace_turn(&self) -> bool {
        self.max_steps_final
    }

    pub fn maybe_schedule_turn_limit_final(
        &mut self,
        iteration: usize,
        run_offset: usize,
        max_turns: Option<usize>,
        msgs: &mut Vec<Message>,
    ) {
        let Some(max) = max_turns else { return };
        if iteration < run_offset + max {
            return;
        }
        let sys_id = format!("m{}", msgs.len());
        msgs.push(Message {
            id: sys_id.clone(),
            role: MessageRole::System,
            parts: vec![Part {
                id: format!("{}.p0", sys_id),
                kind: PartKind::Text,
                content: format!(
                    "Turn limit reached ({max}). You MUST reply in plain prose now containing:\n\
                        1. Summary of what you accomplished\n\
                        2. List of remaining tasks not yet completed\n\
                        3. Recommended next steps\n\
                        Do NOT make any more tool calls and do NOT emit `<repl>`."
                ),
                tool_call_id: None,
                tool_name: None,
                prune_state: PruneState::Intact,
            }],
            origin: None,
        });
        self.max_steps_final = true;
    }
}

pub fn make_error_envelope(
    kind: &str,
    code: Option<&str>,
    user_message: impl Into<String>,
    raw: Option<String>,
) -> ErrorEnvelope {
    let user_message = user_message.into();
    ErrorEnvelope {
        kind: kind.to_string(),
        code: code.map(str::to_string),
        user_message,
        raw: raw.map(|s| truncate_raw_error(s.trim())),
    }
}

pub fn make_error_event(
    kind: &str,
    code: Option<&str>,
    user_message: impl Into<String>,
    raw: Option<String>,
) -> AgentEvent {
    let user_message = user_message.into();
    AgentEvent::Error {
        message: user_message.clone(),
        envelope: Some(make_error_envelope(kind, code, user_message, raw)),
    }
}

pub fn truncate_raw_error(s: &str) -> String {
    const MAX_RAW: usize = 4000;
    if s.len() <= MAX_RAW {
        return s.to_string();
    }
    let keep = MAX_RAW / 2;
    let omitted = s.len() - MAX_RAW;
    format!(
        "{}\n\n... ({omitted} chars omitted) ...\n\n{}",
        &s[..keep],
        &s[s.len() - keep..]
    )
}

pub fn is_malformed_assistant_output(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return false;
    }

    if matches!(
        trimmed,
        "<" | "</" | "<repl" | "</repl" | "<repl>" | "</repl>"
    ) {
        return true;
    }

    if (trimmed.starts_with("<repl") && !trimmed.contains("</repl>"))
        || trimmed.starts_with("</repl")
        || trimmed.ends_with("<repl")
        || trimmed.ends_with("</repl")
    {
        return true;
    }

    if trimmed.starts_with('<') && !trimmed.contains('>') && trimmed.len() <= 16 {
        let looks_like_fragment = trimmed
            .chars()
            .skip(1)
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '/' || ch == '_' || ch == '-');
        if looks_like_fragment {
            return true;
        }
    }

    false
}

pub fn format_tool_result_content(success: bool, result: &serde_json::Value) -> String {
    if success {
        match result {
            serde_json::Value::String(text) => text.clone(),
            other => serde_json::to_string(other).unwrap_or_else(|_| "null".to_string()),
        }
    } else {
        match result {
            serde_json::Value::String(text) => {
                if text.is_empty() {
                    "[Tool execution failed]".to_string()
                } else {
                    format!("[Tool execution failed]\n{text}")
                }
            }
            other => serde_json::to_string(&serde_json::json!({ "error": other }))
                .unwrap_or_else(|_| "{\"error\":\"tool execution failed\"}".to_string()),
        }
    }
}

pub fn build_assistant_parts(
    msg_id: &str,
    prose_parts: &[String],
    code_parts: &[String],
) -> Vec<Part> {
    let mut parts = Vec::new();
    let mut idx = 0;
    let mut prose_iter = prose_parts.iter();
    let mut code_iter = code_parts.iter();

    loop {
        let prose = prose_iter.next();
        let code = code_iter.next();
        if prose.is_none() && code.is_none() {
            break;
        }
        if let Some(p) = prose
            && !p.is_empty()
        {
            parts.push(Part {
                id: format!("{}.p{}", msg_id, idx),
                kind: PartKind::Prose,
                content: p.clone(),
                tool_call_id: None,
                tool_name: None,
                prune_state: PruneState::Intact,
            });
            idx += 1;
        }
        if let Some(c) = code
            && !c.is_empty()
        {
            parts.push(Part {
                id: format!("{}.p{}", msg_id, idx),
                kind: PartKind::Code,
                content: c.clone(),
                tool_call_id: None,
                tool_name: None,
                prune_state: PruneState::Intact,
            });
            idx += 1;
        }
    }

    if parts.is_empty() {
        parts.push(Part {
            id: format!("{}.p0", msg_id),
            kind: PartKind::Prose,
            content: String::new(),
            tool_call_id: None,
            tool_name: None,
            prune_state: PruneState::Intact,
        });
    }

    parts
}

pub struct FenceLineParse {
    pub prose_delta: String,
    pub codes_to_execute: Vec<String>,
}

pub fn append_line_segment(target: &mut String, segment: &str, line_started: &mut bool) {
    if segment.is_empty() {
        return;
    }
    if !*line_started {
        if !target.is_empty() {
            target.push('\n');
        }
        *line_started = true;
    }
    target.push_str(segment);
}

pub fn parse_fence_line(
    line: &str,
    in_code_fence: &mut bool,
    current_prose: &mut String,
    current_code: &mut String,
    prose_parts: &mut Vec<String>,
) -> FenceLineParse {
    const OPEN_TAG: &str = "<repl>";
    const CLOSE_TAG: &str = "</repl>";

    let mut out = FenceLineParse {
        prose_delta: String::new(),
        codes_to_execute: Vec::new(),
    };

    let mut remaining = line;
    let mut prose_started_this_line = false;
    let mut code_started_this_line = false;

    if *in_code_fence && line.is_empty() && !current_code.is_empty() {
        current_code.push('\n');
        return out;
    }

    loop {
        if !*in_code_fence {
            if let Some(idx) = remaining.find(OPEN_TAG) {
                let before = &remaining[..idx];
                append_line_segment(current_prose, before, &mut prose_started_this_line);
                out.prose_delta.push_str(before);

                let prose = current_prose.trim().to_string();
                if !prose.is_empty() {
                    prose_parts.push(prose);
                }
                current_prose.clear();
                *in_code_fence = true;
                current_code.clear();
                code_started_this_line = false;
                remaining = &remaining[idx + OPEN_TAG.len()..];
                continue;
            }

            append_line_segment(current_prose, remaining, &mut prose_started_this_line);
            out.prose_delta.push_str(remaining);
            break;
        }

        if let Some(idx) = remaining.find(CLOSE_TAG) {
            let before = &remaining[..idx];
            append_line_segment(current_code, before, &mut code_started_this_line);
            *in_code_fence = false;
            let code = std::mem::take(current_code);
            if !code.trim().is_empty() {
                out.codes_to_execute.push(code);
            }
            break;
        }

        append_line_segment(current_code, remaining, &mut code_started_this_line);
        break;
    }

    out
}

pub fn model_tool_specs(tools: &[ToolDefinition]) -> Vec<LlmToolSpec> {
    tools
        .iter()
        .map(|tool| {
            let model_tool = tool.model_tool();
            LlmToolSpec {
                name: model_tool.name,
                description: model_tool.description,
                input_schema: model_tool.input_schema,
                output_schema: model_tool.output_schema,
            }
        })
        .collect()
}
