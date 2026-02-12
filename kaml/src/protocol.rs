use serde::{Deserialize, Serialize};

// --- Host → Python (stdin) ---

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum HostMessage {
    #[serde(rename = "init")]
    Init { tools: String },
    #[serde(rename = "exec")]
    Exec { id: String, code: String },
    #[serde(rename = "tool_result")]
    ToolResult {
        id: String,
        success: bool,
        result: String,
    },
    #[serde(rename = "snapshot")]
    Snapshot { id: String },
    #[serde(rename = "restore")]
    Restore { id: String, data: String },
    #[serde(rename = "shutdown")]
    Shutdown,
}

// --- Python → Host (stdout) ---

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub enum PythonMessage {
    #[serde(rename = "ready")]
    Ready,
    #[serde(rename = "tool_call")]
    ToolCall {
        id: String,
        name: String,
        args: String,
    },
    #[serde(rename = "message")]
    Message { text: String, kind: String },
    #[serde(rename = "exec_result")]
    ExecResult {
        id: String,
        output: String,
        response: String,
        error: Option<String>,
    },
    #[serde(rename = "snapshot_result")]
    SnapshotResult { id: String, data: String },
}
