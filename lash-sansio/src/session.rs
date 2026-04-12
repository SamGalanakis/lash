use crate::{ToolCallRecord, ToolImage};

#[derive(Clone, Debug)]
pub struct ExecResponse {
    pub output: String,
    pub observations: Vec<String>,
    pub tool_calls: Vec<ToolCallRecord>,
    pub images: Vec<ToolImage>,
    pub error: Option<String>,
    pub duration_ms: u64,
    /// When the surrounding session uses `ReplTermination::Finish`,
    /// this carries the value the lashlang program ended with via
    /// `finish <expr>`. The dispatch loop uses it as the terminal
    /// result of the session. `None` for chat-style sessions and for
    /// typed sessions whose step continued without finishing.
    pub terminal_finish: Option<serde_json::Value>,
}
