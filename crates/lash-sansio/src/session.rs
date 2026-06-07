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
    /// this carries the value the lashlang program ended with via
    /// `submit <expr>`. The dispatch loop uses it as the terminal
    /// result of the session. `None` for chat-style sessions and for
    /// typed sessions whose step continued without finishing.
    pub terminal_finish: Option<serde_json::Value>,
}

/// Exact prompt-usage snapshot from the most recent completed LLM call.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct PromptUsage {
    pub prompt_context_tokens: usize,
    pub input_tokens: usize,
    pub cached_input_tokens: usize,
    #[serde(default)]
    pub context_budget_tokens: usize,
}

/// Pure multi-turn session state for hosts that want lash behavior without the runtime.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct SansIoSessionState {
    pub session_id: String,
    #[serde(default)]
    pub messages: Vec<crate::Message>,
    #[serde(default)]
    pub protocol_iteration: usize,
    #[serde(default)]
    pub token_usage: crate::TokenUsage,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_prompt_usage: Option<PromptUsage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub protocol_state: Option<serde_json::Value>,
}

#[derive(Clone, Debug, Default)]
pub struct CompletedTurn {
    pub messages: Vec<crate::Message>,
    pub tool_calls: Vec<ToolCallRecord>,
    pub protocol_iteration: usize,
    pub token_usage: crate::TokenUsage,
    pub last_prompt_usage: Option<PromptUsage>,
    pub protocol_state: Option<serde_json::Value>,
}

pub fn apply_completed_turn(
    mut state: SansIoSessionState,
    turn: CompletedTurn,
) -> SansIoSessionState {
    state.messages = turn.messages;
    state.protocol_iteration = turn.protocol_iteration;
    state.token_usage = turn.token_usage;
    state.last_prompt_usage = turn.last_prompt_usage;
    state.protocol_state = turn.protocol_state;
    state
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completed_turn_replaces_projected_session_state() {
        let state = SansIoSessionState {
            session_id: "session".to_string(),
            protocol_iteration: 1,
            ..SansIoSessionState::default()
        };
        let reduced = apply_completed_turn(
            state,
            CompletedTurn {
                protocol_iteration: 4,
                token_usage: crate::TokenUsage {
                    input_tokens: 10,
                    output_tokens: 3,
                    cached_input_tokens: 1,
                    reasoning_tokens: 2,
                },
                last_prompt_usage: Some(PromptUsage {
                    prompt_context_tokens: 7,
                    input_tokens: 6,
                    cached_input_tokens: 1,
                    context_budget_tokens: 100,
                }),
                ..CompletedTurn::default()
            },
        );

        assert_eq!(reduced.protocol_iteration, 4);
        assert_eq!(reduced.token_usage.input_tokens, 10);
        assert_eq!(
            reduced
                .last_prompt_usage
                .expect("prompt usage present")
                .prompt_context_tokens,
            7
        );
    }
}
