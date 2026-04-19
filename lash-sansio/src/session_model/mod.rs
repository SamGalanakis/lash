pub mod message;
pub mod prompt;

pub use message::{
    Message, MessageRole, MessageSequence, Part, PartKind, PruneState, RenderedPrompt,
    append_rendered_prompt, messages_are_live_resume_safe, render_prompt, render_transcript_prompt,
};
pub use prompt::{
    CORE_GUIDANCE_SECTION, MAIN_AGENT_INTRO, PromptBuiltin, PromptSlot, PromptTemplate,
    PromptTemplateEntry, PromptTemplateSection, default_prompt_template,
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
pub enum SessionEvent {
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
    #[serde(rename = "child_token_usage")]
    ChildTokenUsage {
        session_id: String,
        source: String,
        model: String,
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
    #[serde(rename = "injected_turn_input_accepted")]
    InjectedTurnInputAccepted {
        messages: Vec<PluginMessage>,
        checkpoint: CheckpointKind,
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
    /// Emitted when a typed RLM session terminates via `finish <expr>`.
    /// The `value` is the captured (and schema-validated) result.
    /// Hosts that want the typed shape back (e.g. the parent of a
    /// typed subagent workflow) listen for this event on the child's stream;
    /// it is also stamped onto `AssembledTurn::typed_finish` for
    /// non-streaming consumers.
    #[serde(rename = "typed_finish")]
    TypedFinish { value: serde_json::Value },
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
        request: PromptRequest,
        #[serde(skip)]
        response_tx: std::sync::mpsc::Sender<PromptResponse>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PromptRequest {
    pub question: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub panel: Option<PromptPanel>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub options: Vec<String>,
    #[serde(default)]
    pub selection_mode: PromptSelectionMode,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub allow_note: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PromptPanel {
    pub title: String,
    pub markdown: String,
}

impl PromptRequest {
    pub fn freeform(question: impl Into<String>) -> Self {
        Self {
            question: question.into(),
            panel: None,
            options: Vec::new(),
            selection_mode: PromptSelectionMode::Single,
            allow_note: false,
        }
    }

    pub fn single(question: impl Into<String>, options: Vec<String>) -> Self {
        Self {
            question: question.into(),
            panel: None,
            options,
            selection_mode: PromptSelectionMode::Single,
            allow_note: false,
        }
    }

    pub fn multi(question: impl Into<String>, options: Vec<String>) -> Self {
        Self {
            question: question.into(),
            panel: None,
            options,
            selection_mode: PromptSelectionMode::Multi,
            allow_note: false,
        }
    }

    pub fn with_optional_note(mut self) -> Self {
        self.allow_note = !self.is_freeform();
        self
    }

    pub fn with_markdown_panel(
        mut self,
        title: impl Into<String>,
        markdown: impl Into<String>,
    ) -> Self {
        self.panel = Some(PromptPanel {
            title: title.into(),
            markdown: markdown.into(),
        });
        self
    }

    pub fn is_freeform(&self) -> bool {
        self.options.is_empty()
    }

    pub fn allows_note(&self) -> bool {
        self.allow_note && !self.is_freeform()
    }

    pub fn empty_response(&self) -> PromptResponse {
        if self.is_freeform() {
            PromptResponse::Text {
                text: String::new(),
            }
        } else {
            match self.selection_mode {
                PromptSelectionMode::Single => PromptResponse::Single {
                    selection: String::new(),
                    note: None,
                },
                PromptSelectionMode::Multi => PromptResponse::Multi {
                    selections: Vec::new(),
                    note: None,
                },
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PromptSelectionMode {
    #[default]
    Single,
    Multi,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PromptResponse {
    Text {
        text: String,
    },
    Single {
        selection: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        note: Option<String>,
    },
    Multi {
        selections: Vec<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        note: Option<String>,
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
        let sys_id = fresh_message_id();
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
                        Do NOT make any more tool calls and do NOT emit `<rlm>`."
                ),
                attachment: None,
                tool_call_id: None,
                tool_name: None,
                tool_item_id: None,
                prune_state: PruneState::Intact,
            }],
            user_input: None,
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
) -> SessionEvent {
    let user_message = user_message.into();
    SessionEvent::Error {
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

pub fn fresh_message_id() -> String {
    format!("m{}", uuid::Uuid::new_v4().simple())
}

pub fn reassign_part_ids(message_id: &str, parts: &mut [Part]) {
    for (idx, part) in parts.iter_mut().enumerate() {
        part.id = format!("{message_id}.p{idx}");
    }
}

pub fn model_tool_specs_iter<'a>(
    tools: impl IntoIterator<Item = &'a ToolDefinition>,
) -> Vec<LlmToolSpec> {
    tools
        .into_iter()
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

pub fn model_tool_specs(tools: &[ToolDefinition]) -> Vec<LlmToolSpec> {
    model_tool_specs_iter(tools.iter())
}
