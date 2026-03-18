use crate::{ToolCallRecord, ToolImage};

#[derive(Clone, Debug)]
pub struct ExecResponse {
    pub output: String,
    pub observations: Vec<String>,
    pub tool_calls: Vec<ToolCallRecord>,
    pub images: Vec<ToolImage>,
    pub error: Option<String>,
    pub duration_ms: u64,
}
