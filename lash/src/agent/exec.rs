/// Accumulated state from executing code blocks during a single LLM turn.
pub(crate) struct ExecAccumulator {
    pub tool_calls: Vec<crate::ToolCallRecord>,
    pub images: Vec<crate::ToolImage>,
    pub combined_output: String,
    pub final_response: String,
    pub exec_error: Option<String>,
    pub had_failure: bool,
}

impl ExecAccumulator {
    pub fn new() -> Self {
        Self {
            tool_calls: Vec::new(),
            images: Vec::new(),
            combined_output: String::new(),
            final_response: String::new(),
            exec_error: None,
            had_failure: false,
        }
    }
}
