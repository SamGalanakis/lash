use std::sync::mpsc as std_mpsc;

#[derive(Debug)]
pub enum PythonRequest {
    Init {
        tools_json: String,
        agent_id: String,
        headless: bool,
        capabilities_json: String,
    },
    Exec {
        id: String,
        code: String,
    },
    Snapshot {
        id: String,
    },
    Restore {
        id: String,
        data: String,
    },
    Reset {
        id: String,
    },
    Reconfigure {
        tools_json: String,
        capabilities_json: String,
        generation: u64,
    },
    CheckComplete {
        code: String,
    },
    Shutdown,
}

#[derive(Debug)]
pub enum PythonResponse {
    Ready,
    ToolCall {
        id: String,
        name: String,
        args: String,
        result_tx: std_mpsc::Sender<String>,
    },
    Message {
        text: String,
        kind: String,
    },
    ExecResult {
        id: String,
        output: String,
        error: Option<String>,
    },
    SnapshotResult {
        id: String,
        data: String,
    },
    ResetResult {
        id: String,
    },
    ReconfigureResult {
        generation: u64,
        error: Option<String>,
    },
    CheckCompleteResult {
        is_complete: bool,
    },
    AskUser {
        question: String,
        options: Vec<String>,
        result_tx: std_mpsc::Sender<String>,
    },
}

pub struct PythonRuntime;

impl PythonRuntime {
    pub fn start() -> Result<Self, std::io::Error> {
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "embedded Python runtime is not available in this build",
        ))
    }

    pub fn send(&self, _request: PythonRequest) -> Result<(), std::io::Error> {
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "embedded Python runtime is not available in this build",
        ))
    }

    pub fn recv(&self) -> Result<PythonResponse, std::io::Error> {
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "embedded Python runtime is not available in this build",
        ))
    }

    pub fn try_recv(&self) -> Result<Option<PythonResponse>, std::io::Error> {
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "embedded Python runtime is not available in this build",
        ))
    }
}
