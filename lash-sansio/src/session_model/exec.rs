/// Accumulated state from executing code blocks during a single LLM turn.
pub(crate) struct ExecAccumulator {
    pub tool_calls: Vec<crate::ToolCallRecord>,
    pub images: Vec<crate::ToolImage>,
    pub combined_output: String,
    pub exec_error: Option<String>,
    /// The lashlang source that was executed. Carried through so the
    /// synthetic `SessionEvent::ToolCall` telemetry event can include
    /// the code in its `args` for the activity projector to render.
    pub executed_code: Option<String>,
    /// Set when the lashlang program ended with `finish <expr>` AND
    /// the surrounding session was started in a typed termination
    /// mode that accepts that as the terminal result. The dispatch
    /// loop reads this in `process_repl_result` to terminate the turn
    /// cleanly with the captured value.
    pub terminal_finish: Option<serde_json::Value>,
}

impl ExecAccumulator {
    pub fn new() -> Self {
        Self {
            tool_calls: Vec::new(),
            images: Vec::new(),
            combined_output: String::new(),
            exec_error: None,
            executed_code: None,
            terminal_finish: None,
        }
    }
}
