pub mod message;
pub mod prompt;

pub use message::{
    BaseRenderCache, Message, MessageRole, MessageSequence, Part, PartAttachment, PartKind,
    PruneState, RenderedPrompt, append_rendered_prompt, messages_are_prompt_resume_safe,
    render_prompt, render_transcript_prompt, shared_parts,
};
pub use prompt::{
    MAIN_AGENT_INTRO, PromptBuiltin, PromptLayer, PromptSlot, PromptSlotLayer, PromptTemplate,
    PromptTemplateEntry, PromptTemplateSection, ResolvedPromptLayer, default_prompt_template,
    resolve_prompt_layers,
};

use std::sync::Arc;

use crate::ToolDefinition;
use crate::llm::types::LlmToolSpec;
use crate::plugin::{CheckpointKind, PluginMessage, PluginRuntimeEvent};
use crate::{MessageOrigin, ToolCallRecord};

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[allow(clippy::large_enum_variant)]
pub enum SessionEventRecord<PE = ()> {
    Conversation(ConversationRecord),
    Tool(ToolEvent),
    Protocol(PE),
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ConversationRecord {
    pub id: String,
    pub role: MessageRole,
    pub parts: Arc<Vec<Part>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub origin: Option<MessageOrigin>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct AcceptedInjectedTurnInput {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub message: PluginMessage,
}

impl ConversationRecord {
    pub fn from_message(message: Message) -> Self {
        Self {
            id: message.id,
            role: message.role,
            parts: message.parts,
            origin: message.origin,
        }
    }

    pub fn to_message(&self) -> Message {
        Message {
            id: self.id.clone(),
            role: self.role,
            parts: Arc::clone(&self.parts),
            origin: self.origin.clone(),
        }
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum ToolEvent {
    Invocation {
        stable_key: String,
        record: ToolCallRecord,
    },
}

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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub terminal_reason: Option<crate::llm::types::LlmTerminalReason>,
    pub user_message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type")]
#[allow(clippy::large_enum_variant)]
pub enum SessionEvent {
    #[serde(rename = "text_delta")]
    TextDelta { content: String },
    /// Streaming update for the model's reasoning summary ("thinking").
    /// The UI renders these incrementally in a muted/italic style;
    /// reasoning content is never fed back to the model on subsequent
    /// turns.
    #[serde(rename = "reasoning_delta")]
    ReasoningDelta { content: String },
    #[serde(rename = "tool_call")]
    ToolCall {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        call_id: Option<String>,
        name: String,
        args: serde_json::Value,
        output: crate::ToolCallOutput,
        duration_ms: u64,
    },
    #[serde(rename = "tool_call_start")]
    ToolCallStart {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        call_id: Option<String>,
        name: String,
        args: serde_json::Value,
    },
    #[serde(rename = "message")]
    Message { text: String, kind: String },
    #[serde(rename = "llm_request")]
    LlmRequest {
        protocol_iteration: usize,
        message_count: usize,
        tool_list: String,
    },
    #[serde(rename = "llm_response")]
    LlmResponse {
        protocol_iteration: usize,
        content: String,
        duration_ms: u64,
    },
    #[serde(rename = "token_usage")]
    TokenUsage {
        protocol_iteration: usize,
        usage: TokenUsage,
        cumulative: TokenUsage,
    },
    #[serde(rename = "child_token_usage")]
    ChildTokenUsage {
        session_id: String,
        source: String,
        model: String,
        protocol_iteration: usize,
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
        inputs: Vec<AcceptedInjectedTurnInput>,
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
        event: PluginRuntimeEvent,
    },
    /// Semantic result for a completed turn. `Done` remains the machine
    /// lifecycle marker emitted after this event.
    #[serde(rename = "turn_outcome")]
    TurnOutcome { outcome: TurnOutcome },
    #[serde(rename = "done")]
    Done,
    #[serde(rename = "error")]
    Error {
        message: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        envelope: Option<ErrorEnvelope>,
    },
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TurnOutcome {
    Finished(TurnFinish),
    AgentFrameSwitch { frame_id: String },
    Stopped(TurnStop),
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TurnFinish {
    AssistantMessage {
        text: String,
    },
    SubmittedValue {
        value: serde_json::Value,
    },
    ToolValue {
        tool_name: String,
        value: serde_json::Value,
    },
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TurnStop {
    Cancelled,
    Incomplete,
    InvalidInput,
    MaxTurns,
    ToolFailure,
    ProviderError,
    PluginAbort,
    RuntimeError,
    SubmittedError {
        value: serde_json::Value,
    },
    ToolError {
        tool_name: String,
        value: serde_json::Value,
    },
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TurnTerminationPolicyState {
    turn_limit_final_scheduled: bool,
}

impl Default for TurnTerminationPolicyState {
    fn default() -> Self {
        Self::new()
    }
}

impl TurnTerminationPolicyState {
    pub fn new() -> Self {
        Self {
            turn_limit_final_scheduled: false,
        }
    }

    pub fn should_force_exit_after_grace_turn(&self) -> bool {
        self.turn_limit_final_scheduled
    }

    pub fn turn_limit_final_to_schedule(
        &self,
        protocol_iteration: usize,
        protocol_run_offset: usize,
        max_turns: Option<usize>,
    ) -> Option<usize> {
        if self.turn_limit_final_scheduled {
            return None;
        }
        let max = max_turns?;
        if protocol_iteration < protocol_run_offset + max {
            return None;
        }
        Some(max)
    }

    pub fn mark_turn_limit_final_scheduled(&mut self) {
        self.turn_limit_final_scheduled = true;
    }
}

pub fn make_error_envelope(
    kind: &str,
    code: Option<&str>,
    terminal_reason: Option<crate::llm::types::LlmTerminalReason>,
    user_message: impl Into<String>,
    raw: Option<String>,
) -> ErrorEnvelope {
    let user_message = user_message.into();
    ErrorEnvelope {
        kind: kind.to_string(),
        code: code.map(str::to_string),
        terminal_reason,
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
        envelope: Some(make_error_envelope(kind, code, None, user_message, raw)),
    }
}

pub fn truncate_raw_error(s: &str) -> String {
    const MAX_RAW: usize = 4000;
    let raw_len = s.chars().count();
    if raw_len <= MAX_RAW {
        return s.to_string();
    }
    let keep = MAX_RAW / 2;
    let head = s.chars().take(keep).collect::<String>();
    let tail = s
        .chars()
        .rev()
        .take(keep)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<String>();
    let omitted = raw_len.saturating_sub(keep * 2);
    format!("{head}\n\n... ({omitted} chars omitted) ...\n\n{tail}")
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
                input_schema_projections: model_tool.input_schema_projections,
                output_schema_projections: model_tool.output_schema_projections,
            }
        })
        .collect()
}

pub fn model_tool_specs(tools: &[ToolDefinition]) -> Vec<LlmToolSpec> {
    model_tool_specs_iter(tools.iter())
}
