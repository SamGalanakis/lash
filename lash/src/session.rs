use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde_json::json;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout};
use tokio::sync::mpsc::UnboundedSender;

use crate::protocol::{HostMessage, PythonMessage};
use crate::{SandboxMessage, ToolCallRecord, ToolProvider};

const REPL_PY: &str = include_str!("../python/repl.py");

#[derive(Debug, thiserror::Error)]
pub enum SessionError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("child process exited unexpectedly")]
    ChildExited,
    #[error("protocol error: {0}")]
    Protocol(String),
    #[error("spawn error: {0}")]
    Spawn(String),
}

/// Configuration for a Python REPL session.
pub struct SessionConfig {
    /// Optional syd config file path for syscall sandboxing.
    pub syd_config: Option<PathBuf>,
    /// Override the Python command (default: auto-detect uv/python3).
    pub python: Option<String>,
    /// Working directory for the subprocess.
    pub working_dir: Option<PathBuf>,
    /// Extra environment variables.
    pub env: HashMap<String, String>,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            syd_config: None,
            python: None,
            working_dir: None,
            env: HashMap::new(),
        }
    }
}

pub struct Session {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    tools: Arc<dyn ToolProvider>,
    tool_calls: Vec<ToolCallRecord>,
    message_tx: Option<UnboundedSender<SandboxMessage>>,
    final_response: Option<String>,
    scratch_dir: tempfile::TempDir,
}

impl Session {
    pub async fn new(
        tools: Arc<dyn ToolProvider>,
        config: SessionConfig,
    ) -> Result<Self, SessionError> {
        let scratch_dir = tempfile::TempDir::new()?;

        // Write embedded repl.py to a temp file
        let repl_file = tempfile::NamedTempFile::new()?;
        std::fs::write(repl_file.path(), REPL_PY)?;

        // Find Python command
        let (program, mut base_args) = find_python(&config, repl_file.path())?;

        // If syd_config is set, wrap with syd
        let (final_program, final_args) = if let Some(syd_cfg) = &config.syd_config {
            let syd_path = std::env::var("SYD_PATH").unwrap_or_else(|_| "syd".to_string());
            let mut args = vec![
                "-c".to_string(),
                syd_cfg.to_string_lossy().to_string(),
                "--".to_string(),
                program,
            ];
            args.append(&mut base_args);
            (syd_path, args)
        } else {
            (program, base_args)
        };

        let mut cmd = tokio::process::Command::new(&final_program);
        cmd.args(&final_args)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::inherit())
            .env("SCRATCH_DIR", scratch_dir.path());

        if let Some(cwd) = &config.working_dir {
            cmd.current_dir(cwd);
        }
        for (k, v) in &config.env {
            cmd.env(k, v);
        }

        let mut child = cmd.spawn().map_err(|e| SessionError::Spawn(e.to_string()))?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| SessionError::Spawn("missing stdin".to_string()))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| SessionError::Spawn("missing stdout".to_string()))?;

        // Keep the temp file alive by leaking the NamedTempFile handle.
        // It will be cleaned up when the process exits.
        let _persisted = repl_file.into_temp_path();

        let mut session = Self {
            child,
            stdin,
            stdout: BufReader::new(stdout),
            tools,
            tool_calls: Vec::new(),
            message_tx: None,
            final_response: None,
            scratch_dir,
        };

        // Send init with tool definitions
        let defs = session.tools.definitions();
        let tools_json = serde_json::to_string(&defs).unwrap_or_else(|_| "[]".to_string());
        session
            .send(HostMessage::Init { tools: tools_json })
            .await?;

        // Wait for ready
        match session.recv().await? {
            PythonMessage::Ready => {}
            other => {
                return Err(SessionError::Protocol(format!(
                    "expected ready, got: {:?}",
                    other
                )));
            }
        }

        Ok(session)
    }

    /// Set the message sender for streaming messages during execution.
    pub fn set_message_sender(&mut self, tx: UnboundedSender<SandboxMessage>) {
        self.message_tx = Some(tx);
    }

    /// Clear the message sender (drops the sender, causing receivers to terminate).
    pub fn clear_message_sender(&mut self) {
        self.message_tx = None;
    }

    /// Execute code in the persistent Python REPL.
    pub async fn run_code(&mut self, code: &str) -> Result<ExecResponse, SessionError> {
        self.tool_calls.clear();
        self.final_response = None;
        let start = std::time::Instant::now();
        let id = uuid::Uuid::new_v4().to_string();

        self.send(HostMessage::Exec {
            id: id.clone(),
            code: code.to_string(),
        })
        .await?;

        // Read messages until we get exec_result (60s timeout per message)
        let exec_timeout = std::time::Duration::from_secs(60);
        loop {
            let msg = self.recv_timeout(exec_timeout).await?;
            match msg {
                PythonMessage::ToolCall {
                    id: call_id,
                    name,
                    args,
                } => {
                    let tools = Arc::clone(&self.tools);
                    let parsed_args: serde_json::Value =
                        serde_json::from_str(&args).unwrap_or(json!({}));
                    let tool_start = std::time::Instant::now();
                    let result = tools.execute(&name, &parsed_args).await;

                    self.tool_calls.push(ToolCallRecord {
                        tool: name,
                        args: parsed_args,
                        result: result.result.clone(),
                        success: result.success,
                        duration_ms: tool_start.elapsed().as_millis() as u64,
                    });

                    let result_str = serde_json::to_string(&result.result)
                        .unwrap_or_else(|_| "null".to_string());

                    self.send(HostMessage::ToolResult {
                        id: call_id,
                        success: result.success,
                        result: result_str,
                    })
                    .await?;
                }
                PythonMessage::Message { text, kind } => {
                    if kind == "final" {
                        self.final_response = Some(text.clone());
                    }
                    if let Some(tx) = &self.message_tx {
                        let _ = tx.send(SandboxMessage { text, kind });
                    }
                }
                PythonMessage::ExecResult {
                    id: _,
                    output,
                    response: _,
                    error,
                } => {
                    let response = self.final_response.clone().unwrap_or_default();
                    return Ok(ExecResponse {
                        output,
                        response,
                        tool_calls: self.tool_calls.clone(),
                        error,
                        duration_ms: start.elapsed().as_millis() as u64,
                    });
                }
                PythonMessage::Ready => {
                    // Unexpected but harmless
                }
                PythonMessage::SnapshotResult { .. } | PythonMessage::ResetResult { .. } => {
                    return Err(SessionError::Protocol(
                        "unexpected snapshot/reset result during exec".to_string(),
                    ));
                }
            }
        }
    }

    /// Reset the Python REPL namespace and re-register tools.
    pub async fn reset(&mut self) -> Result<(), SessionError> {
        let id = uuid::Uuid::new_v4().to_string();
        self.send(HostMessage::Reset { id: id.clone() }).await?;

        loop {
            match self.recv().await? {
                PythonMessage::ResetResult { .. } => break,
                _ => continue,
            }
        }

        Ok(())
    }

    /// Snapshot the session: Python namespace (via dill) + scratch filesystem.
    pub async fn snapshot(&mut self) -> Result<Vec<u8>, SessionError> {
        let id = uuid::Uuid::new_v4().to_string();
        self.send(HostMessage::Snapshot { id: id.clone() })
            .await?;

        let data = loop {
            match self.recv().await? {
                PythonMessage::SnapshotResult { id: _, data } => break data,
                _ => continue,
            }
        };

        // Collect scratch files
        let files = collect_files(self.scratch_dir.path()).unwrap_or_default();

        // `data` is opaque (hex-encoded dill bytes or JSON string) — store as-is
        let combined = json!({ "vars": data, "files": files });
        Ok(serde_json::to_vec(&combined).unwrap())
    }

    /// Restore a session from a snapshot.
    pub async fn restore(&mut self, data: &[u8]) -> Result<(), SessionError> {
        let parsed: serde_json::Value = serde_json::from_slice(data).unwrap_or(json!({}));

        // `vars` is a hex-encoded dill blob
        let vars_str = parsed
            .get("vars")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let id = uuid::Uuid::new_v4().to_string();
        self.send(HostMessage::Restore {
            id,
            data: vars_str,
        })
        .await?;

        // Wait for acknowledgment (exec_result with empty response)
        loop {
            match self.recv().await? {
                PythonMessage::ExecResult { .. } => break,
                _ => continue,
            }
        }

        // Restore scratch files
        if let Some(files_val) = parsed.get("files")
            && let Ok(files) =
                serde_json::from_value::<HashMap<String, String>>(files_val.clone())
        {
            // Clear existing scratch contents
            if let Ok(entries) = std::fs::read_dir(self.scratch_dir.path()) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_dir() {
                        let _ = std::fs::remove_dir_all(&path);
                    } else {
                        let _ = std::fs::remove_file(&path);
                    }
                }
            }
            let _ = restore_files(self.scratch_dir.path(), &files);
        }

        Ok(())
    }

    /// Path to the scratch directory on the host filesystem.
    pub fn scratch_path(&self) -> &Path {
        self.scratch_dir.path()
    }

    /// Access the tool provider.
    pub fn tools(&self) -> &Arc<dyn ToolProvider> {
        &self.tools
    }

    /// Get tool call records from the last run_code execution.
    pub fn last_tool_calls(&self) -> &[ToolCallRecord] {
        &self.tool_calls
    }

    /// Check if the Python child process is still alive.
    fn check_alive(&mut self) -> Result<(), SessionError> {
        match self.child.try_wait() {
            Ok(Some(_status)) => Err(SessionError::ChildExited),
            Ok(None) => Ok(()), // still running
            Err(e) => Err(SessionError::Io(e)),
        }
    }

    async fn send(&mut self, msg: HostMessage) -> Result<(), SessionError> {
        self.check_alive()?;
        let mut line = serde_json::to_string(&msg)?;
        line.push('\n');
        self.stdin.write_all(line.as_bytes()).await?;
        self.stdin.flush().await?;
        Ok(())
    }

    async fn recv(&mut self) -> Result<PythonMessage, SessionError> {
        let mut line = String::new();
        let n = self.stdout.read_line(&mut line).await?;
        if n == 0 {
            return Err(SessionError::ChildExited);
        }
        let msg: PythonMessage = serde_json::from_str(line.trim())?;
        Ok(msg)
    }

    /// Receive with a timeout. Returns SessionError on timeout or if the child is dead.
    async fn recv_timeout(&mut self, timeout: std::time::Duration) -> Result<PythonMessage, SessionError> {
        match tokio::time::timeout(timeout, self.recv()).await {
            Ok(result) => result,
            Err(_) => {
                // Check if the child died while we were waiting
                self.check_alive()?;
                Err(SessionError::Protocol("recv timed out".to_string()))
            }
        }
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        // Best-effort: send shutdown and kill if needed
        let _ = self.child.start_kill();
    }
}

#[derive(Clone, Debug)]
pub struct ExecResponse {
    /// Captured stdout — model's own context (print, auto-printed expressions).
    pub output: String,
    /// User-facing final response from message(kind="final").
    pub response: String,
    pub tool_calls: Vec<ToolCallRecord>,
    pub error: Option<String>,
    pub duration_ms: u64,
}

/// Build the Python command. Requires uv.
fn find_python(config: &SessionConfig, repl_path: &Path) -> Result<(String, Vec<String>), SessionError> {
    let repl = repl_path.to_string_lossy().to_string();

    // Explicit override
    if let Some(python) = &config.python {
        return Ok((python.clone(), vec![repl]));
    }

    Ok((
        "uv".to_string(),
        vec![
            "run".to_string(),
            "--python".to_string(),
            "3.13".to_string(),
            "--with".to_string(),
            "dill".to_string(),
            "python3".to_string(),
            repl,
        ],
    ))
}

/// Walk a directory recursively and collect all files as relative_path -> contents.
fn collect_files(root: &Path) -> std::io::Result<HashMap<String, String>> {
    let mut files = HashMap::new();
    walk_dir(root, root, &mut files)?;
    Ok(files)
}

fn walk_dir(root: &Path, dir: &Path, files: &mut HashMap<String, String>) -> std::io::Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            walk_dir(root, &path, files)?;
        } else {
            let rel = path
                .strip_prefix(root)
                .unwrap_or(&path)
                .to_string_lossy()
                .to_string();
            // Best-effort: skip binary files that aren't valid UTF-8
            if let Ok(content) = std::fs::read_to_string(&path) {
                files.insert(rel, content);
            }
        }
    }
    Ok(())
}

/// Recreate files in a directory from a path -> content map.
fn restore_files(root: &Path, files: &HashMap<String, String>) -> std::io::Result<()> {
    for (rel_path, content) in files {
        let full = root.join(rel_path);
        if let Some(parent) = full.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&full, content)?;
    }
    Ok(())
}
