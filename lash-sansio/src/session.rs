use crate::{ToolCallRecord, ToolImage};

#[derive(Clone, Debug)]
pub struct ExecResponse {
    pub output: String,
    pub observations: Vec<String>,
    pub tool_calls: Vec<ToolCallRecord>,
    pub images: Vec<ToolImage>,
    pub error: Option<String>,
    pub duration_ms: u64,
    /// When the surrounding session uses `mode-specific finish`,
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
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCallRecord>,
    #[serde(default)]
    pub iteration: usize,
    #[serde(default)]
    pub token_usage: crate::TokenUsage,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_prompt_usage: Option<PromptUsage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mode_state: Option<serde_json::Value>,
}

#[derive(Clone, Debug, Default)]
pub struct CompletedTurn {
    pub messages: Vec<crate::Message>,
    pub tool_calls: Vec<ToolCallRecord>,
    pub iteration: usize,
    pub token_usage: crate::TokenUsage,
    pub last_prompt_usage: Option<PromptUsage>,
    pub mode_state: Option<serde_json::Value>,
}

pub fn apply_completed_turn(
    mut state: SansIoSessionState,
    turn: CompletedTurn,
) -> SansIoSessionState {
    state.messages = turn.messages;
    state.tool_calls = turn.tool_calls;
    state.iteration = turn.iteration;
    state.token_usage = turn.token_usage;
    state.last_prompt_usage = turn.last_prompt_usage;
    state.mode_state = turn.mode_state;
    state
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completed_turn_replaces_projected_session_state() {
        let state = SansIoSessionState {
            session_id: "session".to_string(),
            iteration: 1,
            ..SansIoSessionState::default()
        };
        let reduced = apply_completed_turn(
            state,
            CompletedTurn {
                iteration: 4,
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

        assert_eq!(reduced.iteration, 4);
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
