use std::collections::{HashMap, VecDeque};
use std::path::Path;
use std::sync::Arc;

use serde_json::json;
use tokio::sync::mpsc::UnboundedSender;

use crate::embedded::{LashlangRequest, LashlangResponse, LashlangRuntime};
use crate::tool_dispatch::{ToolDispatchContext, dispatch_tool_call};
use crate::{
    AgentCapabilities, AgentEvent, PluginMessage, RuntimeServices, SandboxMessage, SessionManager,
    ToolCallRecord, ToolImage, ToolProvider, resolve_tool_capability_payload,
};

const REPL_SNAPSHOT_VERSION: u32 = 3;

fn repl_tools_json(tools: &Arc<dyn ToolProvider>) -> String {
    let defs: Vec<_> = tools
        .definitions()
        .into_iter()
        .filter(|def| !def.description_for(crate::ExecutionMode::Repl).is_empty())
        .map(|def| def.project(crate::ExecutionMode::Repl))
        .collect();
    serde_json::to_string(&defs).unwrap_or_else(|_| "[]".to_string())
}

fn execution_surface_json(
    tools: &Arc<dyn ToolProvider>,
    capabilities: &AgentCapabilities,
) -> (String, String) {
    (
        repl_tools_json(tools),
        resolve_tool_capability_payload(tools.as_ref(), capabilities).to_json(),
    )
}

/// A prompt from the agent asking the user a question.
/// The `response_tx` travels all the way to the TUI, which sends the answer
/// directly back to the REPL bridge thread.
pub struct UserPrompt {
    pub question: String,
    pub options: Vec<String>,
    pub response_tx: std::sync::mpsc::Sender<String>,
}

#[derive(Clone, Default)]
pub struct PromptBridge {
    sender: std::sync::Arc<std::sync::Mutex<Option<UnboundedSender<UserPrompt>>>>,
}

impl PromptBridge {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_sender(&self, tx: UnboundedSender<UserPrompt>) {
        *self.sender.lock().expect("prompt bridge poisoned") = Some(tx);
    }

    pub fn clear_sender(&self) {
        *self.sender.lock().expect("prompt bridge poisoned") = None;
    }

    pub async fn prompt(&self, question: String, options: Vec<String>) -> Result<String, String> {
        let sender = self
            .sender
            .lock()
            .map_err(|_| "prompt bridge poisoned".to_string())?
            .clone()
            .ok_or_else(|| "ask is unavailable in this session".to_string())?;
        let (response_tx, response_rx) = std::sync::mpsc::channel::<String>();
        sender
            .send(UserPrompt {
                question,
                options,
                response_tx,
            })
            .map_err(|_| "prompt channel closed".to_string())?;
        tokio::task::spawn_blocking(move || response_rx.recv())
            .await
            .map_err(|err| format!("prompt wait task failed: {err}"))?
            .map_err(|_| "prompt response channel closed".to_string())
    }
}

#[derive(Clone, Default)]
pub struct TurnInjectionBridge {
    queue: std::sync::Arc<std::sync::Mutex<VecDeque<PluginMessage>>>,
}

impl TurnInjectionBridge {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn enqueue(&self, messages: Vec<PluginMessage>) -> Result<(), String> {
        let mut queue = self
            .queue
            .lock()
            .map_err(|_| "turn injection bridge poisoned".to_string())?;
        queue.extend(messages);
        Ok(())
    }

    pub fn drain(&self) -> Result<Vec<PluginMessage>, String> {
        let mut queue = self
            .queue
            .lock()
            .map_err(|_| "turn injection bridge poisoned".to_string())?;
        Ok(queue.drain(..).collect())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SessionError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("repl execution mode is not available in this build or session")]
    ReplUnavailable,
    #[error("repl runtime exited unexpectedly")]
    RuntimeExited,
    #[error("protocol error: {0}")]
    Protocol(String),
}

enum SessionBackend {
    Standard,
    Lashlang(LashlangRuntime),
}

pub struct Session {
    backend: SessionBackend,
    services: RuntimeServices,
    tool_calls: Vec<ToolCallRecord>,
    tool_images: Vec<ToolImage>,
    message_tx: Option<UnboundedSender<SandboxMessage>>,
    prompt_tx: Option<UnboundedSender<UserPrompt>>,
    scratch_dir: tempfile::TempDir,
}

impl Session {
    pub async fn new(
        services: RuntimeServices,
        agent_id: &str,
        capabilities: AgentCapabilities,
        execution_mode: crate::ExecutionMode,
    ) -> Result<Self, SessionError> {
        let scratch_dir = tempfile::TempDir::new()?;

        let mut session = Self {
            backend: SessionBackend::Standard,
            services,
            tool_calls: Vec::new(),
            tool_images: Vec::new(),
            message_tx: None,
            prompt_tx: None,
            scratch_dir,
        };

        if matches!(execution_mode, crate::ExecutionMode::Repl) {
            let runtime = LashlangRuntime::start()?;
            session.backend = SessionBackend::Lashlang(runtime);
            session
                .initialize_execution_surface(agent_id, &capabilities)
                .await?;
        }

        Ok(session)
    }

    fn runtime(&self) -> Result<&LashlangRuntime, SessionError> {
        match &self.backend {
            SessionBackend::Standard => Err(SessionError::ReplUnavailable),
            SessionBackend::Lashlang(runtime) => Ok(runtime),
        }
    }

    pub fn supports_repl(&self) -> bool {
        matches!(self.backend, SessionBackend::Lashlang(_))
    }

    pub fn tools(&self) -> &Arc<dyn ToolProvider> {
        &self.services.tools
    }

    pub fn plugins(&self) -> &Arc<crate::PluginSession> {
        &self.services.plugins
    }

    pub fn turn_injection_bridge(&self) -> &TurnInjectionBridge {
        &self.services.turn_injection_bridge
    }

    /// Set the message sender for streaming messages during execution.
    pub fn set_message_sender(&mut self, tx: UnboundedSender<SandboxMessage>) {
        self.message_tx = Some(tx);
    }

    /// Clear the message sender (drops the sender, causing receivers to terminate).
    pub fn clear_message_sender(&mut self) {
        self.message_tx = None;
    }

    /// Set the prompt sender for forwarding user prompts during execution.
    pub fn set_prompt_sender(&mut self, tx: UnboundedSender<UserPrompt>) {
        self.services.prompt_bridge.set_sender(tx.clone());
        self.prompt_tx = Some(tx);
    }

    /// Clear the prompt sender.
    pub fn clear_prompt_sender(&mut self) {
        self.services.prompt_bridge.clear_sender();
        self.prompt_tx = None;
    }

    /// Execute code in the persistent lashlang REPL.
    pub async fn run_code(
        &mut self,
        session_id: &str,
        host: Arc<dyn SessionManager>,
        event_tx: &tokio::sync::mpsc::Sender<AgentEvent>,
        code: &str,
    ) -> Result<ExecResponse, SessionError> {
        self.tool_calls.clear();
        self.tool_images.clear();
        let start = std::time::Instant::now();
        let id = uuid::Uuid::new_v4().to_string();

        // Strip markdown separator lines (all dashes + whitespace) the LLM
        // sometimes emits between code blocks.
        let clean_code: String = code
            .lines()
            .filter(|line| {
                let trimmed = line.trim();
                trimmed.is_empty()
                    || trimmed
                        .trim_matches('-')
                        .chars()
                        .any(|c| !c.is_whitespace())
            })
            .collect::<Vec<_>>()
            .join("\n");

        self.runtime()?.send(LashlangRequest::Exec {
            id: id.clone(),
            code: clean_code,
        })?;

        // Read messages until we get exec_result.
        // Tool calls are spawned as concurrent tokio tasks so REPL parallel branches
        // can dispatch multiple tools at the same time.
        let dispatch = Arc::new(ToolDispatchContext {
            tool_provider: Arc::clone(self.tools()),
            plugins: Arc::clone(self.plugins()),
            host,
            session_id: session_id.to_string(),
            execution_mode: crate::ExecutionMode::Repl,
            event_tx: event_tx.clone(),
        });
        let mut tool_handles: Vec<tokio::task::JoinHandle<(ToolCallRecord, Vec<ToolImage>)>> =
            Vec::new();

        // Use block_in_place so tokio knows this thread is blocked and can
        // schedule drain tasks (prompt forwarding, message forwarding) on other threads.
        loop {
            let runtime = self.runtime()?;
            let response = tokio::task::block_in_place(|| runtime.recv())
                .map_err(|_| SessionError::RuntimeExited)?;
            match response {
                LashlangResponse::ToolCall {
                    id: _call_id,
                    name,
                    args,
                    result_tx,
                } => {
                    let tc_num = tool_handles.len();
                    tracing::info!(
                        "PARALLEL: ToolCall #{tc_num} '{name}' received at t+{:.3}s",
                        start.elapsed().as_secs_f64()
                    );
                    let parsed_args: serde_json::Value =
                        serde_json::from_str(&args).unwrap_or(json!({}));
                    let msg_tx = self.message_tx.clone();
                    let run_start = start;
                    let dispatch = Arc::clone(&dispatch);

                    let handle = tokio::spawn(async move {
                        let tool_name_for_log = name.clone();
                        tracing::info!(
                            "PARALLEL: task #{tc_num} '{name}' executing at t+{:.3}s",
                            run_start.elapsed().as_secs_f64()
                        );
                        let outcome =
                            dispatch_tool_call(&dispatch, name, parsed_args, msg_tx.as_ref()).await;

                        // Send the tool result back to the embedded lashlang runtime.
                        let result_json = json!({
                            "success": outcome.record.success,
                            "result": serde_json::to_string(&outcome.record.result)
                                .unwrap_or_else(|_| "null".to_string()),
                        });
                        let _ = result_tx.send(result_json.to_string());
                        tracing::info!(
                            "PARALLEL: task #{tc_num} '{tool_name_for_log}' done at t+{:.3}s",
                            run_start.elapsed().as_secs_f64()
                        );

                        (outcome.record, outcome.images)
                    });
                    tool_handles.push(handle);
                }
                LashlangResponse::ExecResult {
                    id: _,
                    output,
                    response,
                    finished,
                    observations,
                    error,
                } => {
                    tracing::info!(
                        "PARALLEL: ExecResult received at t+{:.3}s ({} handles)",
                        start.elapsed().as_secs_f64(),
                        tool_handles.len()
                    );
                    // Collect results from all concurrent tool calls.
                    // By the time the runtime sends ExecResult, all tool futures have
                    // resolved, so these awaits are instant.
                    for handle in tool_handles {
                        match handle.await {
                            Ok((record, images)) => {
                                self.tool_calls.push(record);
                                self.tool_images.extend(images);
                            }
                            Err(e) => {
                                self.tool_calls.push(ToolCallRecord {
                                    tool: "unknown".into(),
                                    args: json!({}),
                                    result: json!({"error": format!("task panicked: {e}")}),
                                    success: false,
                                    duration_ms: 0,
                                });
                            }
                        }
                    }
                    let error = error.filter(|value| !value.trim().is_empty());
                    return Ok(ExecResponse {
                        output,
                        observations,
                        response,
                        finished,
                        tool_calls: self.tool_calls.clone(),
                        images: std::mem::take(&mut self.tool_images),
                        error,
                        duration_ms: start.elapsed().as_millis() as u64,
                    });
                }
                LashlangResponse::Ready => {
                    // Unexpected but harmless
                }
                LashlangResponse::AskUser {
                    question,
                    options,
                    result_tx,
                } => {
                    if let Some(tx) = &self.prompt_tx {
                        let _ = tx.send(UserPrompt {
                            question,
                            options,
                            response_tx: result_tx,
                        });
                    } else {
                        // No prompt handler — unblock the embedded lashlang runtime with an empty string.
                        let _ = result_tx.send(String::new());
                    }
                }
                LashlangResponse::SnapshotResult { .. }
                | LashlangResponse::ResetResult { .. }
                | LashlangResponse::ReconfigureResult { .. }
                | LashlangResponse::CheckCompleteResult { .. } => {
                    return Err(SessionError::Protocol(
                        "unexpected response during exec".to_string(),
                    ));
                }
            }
        }
    }

    /// Check if a code string is syntactically complete for the lashlang REPL.
    pub fn check_complete(&self, code: &str) -> Result<bool, SessionError> {
        self.runtime()?.send(LashlangRequest::CheckComplete {
            code: code.to_string(),
        })?;
        let runtime = self.runtime()?;
        let response = tokio::task::block_in_place(|| runtime.recv())
            .map_err(|_| SessionError::RuntimeExited)?;
        match response {
            LashlangResponse::CheckCompleteResult { is_complete } => Ok(is_complete),
            _ => Ok(false),
        }
    }

    /// Reset the lashlang REPL state and re-register tools.
    pub async fn reset(&mut self) -> Result<(), SessionError> {
        let id = uuid::Uuid::new_v4().to_string();
        self.runtime()?
            .send(LashlangRequest::Reset { id: id.clone() })?;

        loop {
            match self.runtime()?.recv()? {
                LashlangResponse::ResetResult { .. } => break,
                _ => continue,
            }
        }

        Ok(())
    }

    /// Re-register the current tool definitions and capability payload in the live REPL.
    /// This is intended for turn-boundary runtime reconfiguration.
    pub async fn refresh_execution_surface(
        &mut self,
        capabilities: &AgentCapabilities,
    ) -> Result<(), SessionError> {
        if !self.supports_repl() {
            return Ok(());
        }

        let (tools_json, capabilities_json) = execution_surface_json(self.tools(), capabilities);
        let generation = self.tools().dynamic_generation().unwrap_or(0);
        self.runtime()?.send(LashlangRequest::Reconfigure {
            tools_json,
            capabilities_json,
            generation,
        })?;

        loop {
            match self.runtime()?.recv()? {
                LashlangResponse::ReconfigureResult {
                    generation: got_generation,
                    error,
                } => {
                    if got_generation != generation {
                        return Err(SessionError::Protocol(format!(
                            "reconfigure generation mismatch: expected {generation}, got {got_generation}"
                        )));
                    }
                    if let Some(err) = error {
                        return Err(SessionError::Protocol(format!("reconfigure failed: {err}")));
                    }
                    break;
                }
                _ => continue,
            }
        }

        Ok(())
    }

    async fn initialize_execution_surface(
        &mut self,
        agent_id: &str,
        capabilities: &AgentCapabilities,
    ) -> Result<(), SessionError> {
        let (tools_json, capabilities_json) = execution_surface_json(self.tools(), capabilities);
        self.runtime()?.send(LashlangRequest::Init {
            tools_json,
            agent_id: agent_id.to_string(),
            capabilities_json,
        })?;

        match self.runtime()?.recv()? {
            LashlangResponse::Ready => Ok(()),
            other => Err(SessionError::Protocol(format!(
                "expected ready, got: {:?}",
                std::mem::discriminant(&other)
            ))),
        }
    }

    /// Snapshot the session: REPL state + scratch filesystem.
    pub async fn snapshot(&mut self) -> Result<Vec<u8>, SessionError> {
        let id = uuid::Uuid::new_v4().to_string();
        self.runtime()?
            .send(LashlangRequest::Snapshot { id: id.clone() })?;

        let data = loop {
            match self.runtime()?.recv()? {
                LashlangResponse::SnapshotResult { id: _, data } => break data,
                _ => continue,
            }
        };

        // Collect scratch files
        let files = collect_files(self.scratch_dir.path()).unwrap_or_default();

        let combined = json!({
            "version": REPL_SNAPSHOT_VERSION,
            "engine": "lashlang",
            "vars": data,
            "files": files,
        });
        Ok(serde_json::to_vec(&combined).unwrap())
    }

    /// Restore a session from a snapshot.
    pub async fn restore(&mut self, data: &[u8]) -> Result<(), SessionError> {
        let parsed: serde_json::Value = serde_json::from_slice(data).unwrap_or(json!({}));

        if parsed.get("version").is_none() || parsed.get("engine").is_none() {
            return Err(SessionError::Protocol(
                "unsupported REPL snapshot format".to_string(),
            ));
        }
        if parsed.get("version").and_then(|v| v.as_u64()) != Some(REPL_SNAPSHOT_VERSION as u64) {
            return Err(SessionError::Protocol(
                "unsupported REPL snapshot version".to_string(),
            ));
        }
        if parsed.get("engine").and_then(|v| v.as_str()) != Some("lashlang") {
            return Err(SessionError::Protocol(
                "unsupported REPL snapshot engine".to_string(),
            ));
        }

        // `vars` is an opaque lashlang snapshot string from the executor thread.
        let vars_str = parsed
            .get("vars")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let id = uuid::Uuid::new_v4().to_string();
        self.runtime()?
            .send(LashlangRequest::Restore { id, data: vars_str })?;

        // Wait for acknowledgment (exec_result with optional restore error)
        loop {
            match self.runtime()?.recv()? {
                LashlangResponse::ExecResult { error, .. } => {
                    if let Some(err) = error {
                        return Err(SessionError::Protocol(format!(
                            "executor restore failed: {err}"
                        )));
                    }
                    break;
                }
                _ => continue,
            }
        }

        // Restore scratch files
        if let Some(files_val) = parsed.get("files")
            && let Ok(files) = serde_json::from_value::<HashMap<String, String>>(files_val.clone())
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
}

#[derive(Clone, Debug)]
pub struct ExecResponse {
    /// Captured stdout from the REPL runtime.
    pub output: String,
    /// Hidden intermediate observations surfaced back to the model on the next REPL step.
    pub observations: Vec<String>,
    /// User-facing final response from `finish`.
    pub response: String,
    /// True when execution ended the turn via `finish`, even if `response` is empty.
    pub finished: bool,
    pub tool_calls: Vec<ToolCallRecord>,
    /// Images returned by tools during this execution (e.g. read_file on a PNG).
    pub images: Vec<ToolImage>,
    pub error: Option<String>,
    pub duration_ms: u64,
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
