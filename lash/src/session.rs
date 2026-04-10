use std::collections::{HashMap, VecDeque};
use std::path::Path;
use std::sync::Arc;

use serde_json::json;
use tokio::sync::mpsc::UnboundedSender;

use crate::embedded::{LashlangRequest, LashlangResponse, LashlangRuntime};
use crate::tool_dispatch::{ToolDispatchContext, dispatch_tool_call};
use crate::{
    ExecResponse, PluginMessage, PromptContribution, RuntimeServices, SandboxMessage, SessionEvent,
    SessionManager, ToolCallRecord, ToolImage, ToolProvider,
};

const REPL_SNAPSHOT_VERSION: u32 = 3;

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

pub struct Session {
    session_id: String,
    repl_runtime: Option<LashlangRuntime>,
    last_repl_tools_json: Option<String>,
    services: RuntimeServices,
    include_base_tools: bool,
    context_tools: Vec<Arc<dyn ToolProvider>>,
    context_prompt_contributions: Vec<PromptContribution>,
    tool_calls: Vec<ToolCallRecord>,
    tool_images: Vec<ToolImage>,
    message_tx: Option<UnboundedSender<SandboxMessage>>,
    scratch_dir: tempfile::TempDir,
}

impl Session {
    pub async fn new(
        services: RuntimeServices,
        session_id: &str,
        execution_mode: crate::ExecutionMode,
    ) -> Result<Self, SessionError> {
        let scratch_dir = tempfile::TempDir::new()?;

        let mut session = Self {
            session_id: session_id.to_string(),
            repl_runtime: None,
            last_repl_tools_json: None,
            services,
            include_base_tools: true,
            context_tools: Vec::new(),
            context_prompt_contributions: Vec::new(),
            tool_calls: Vec::new(),
            tool_images: Vec::new(),
            message_tx: None,
            scratch_dir,
        };

        if matches!(execution_mode, crate::ExecutionMode::Repl) {
            let runtime = LashlangRuntime::start()?;
            session.repl_runtime = Some(runtime);
            session.initialize_execution_surface(session_id).await?;
        }

        Ok(session)
    }

    fn runtime(&self) -> Result<&LashlangRuntime, SessionError> {
        self.repl_runtime
            .as_ref()
            .ok_or(SessionError::ReplUnavailable)
    }

    pub fn supports_repl(&self) -> bool {
        self.repl_runtime.is_some()
    }

    pub fn tools(&self) -> Arc<dyn ToolProvider> {
        if self.include_base_tools && self.context_tools.is_empty() {
            return self.services.plugins.tools();
        }

        let mut providers = Vec::new();
        if self.include_base_tools {
            providers.push(self.services.plugins.tools());
        }
        providers.extend(self.context_tools.iter().cloned());
        Arc::new(crate::tools::CompositeToolProvider::from_providers(
            providers,
        ))
    }

    pub fn plugins(&self) -> &Arc<crate::PluginSession> {
        &self.services.plugins
    }

    pub fn set_context_surface(
        &mut self,
        tool_providers: Vec<Arc<dyn ToolProvider>>,
        prompt_contributions: Vec<PromptContribution>,
        include_base_tools: bool,
    ) {
        self.include_base_tools = include_base_tools;
        self.context_tools = tool_providers;
        self.context_prompt_contributions = prompt_contributions;
    }

    pub fn context_prompt_contributions(&self) -> &[PromptContribution] {
        &self.context_prompt_contributions
    }

    pub fn history_store(&self) -> Option<Arc<dyn crate::store::RuntimeStore>> {
        self.services.store.clone()
    }

    pub fn execution_surface(
        &self,
        session_id: &str,
        mode: crate::ExecutionMode,
    ) -> crate::plugin::ExecutionSurface {
        let mut tools = self.tools().definitions();
        if self.include_base_tools {
            tools.extend(
                crate::tools::native_tools(mode)
                    .iter()
                    .copied()
                    .map(crate::tools::NativeTool::definition),
            );
        }
        self.plugins()
            .resolve_tool_surface(crate::plugin::ToolSurfaceContext {
                session_id: session_id.to_string(),
                mode,
                tools: tools.clone(),
            })
            .unwrap_or_else(|err| {
                tracing::warn!("failed to resolve tool surface: {err}");
                crate::plugin::ExecutionSurface::from_tools(tools)
            })
    }

    pub fn tool_catalog(
        &self,
        session_id: &str,
        mode: crate::ExecutionMode,
    ) -> Vec<serde_json::Value> {
        crate::tools::project_tool_catalog(self.execution_surface(session_id, mode).enabled_tools())
    }

    fn repl_tools_json(&self, session_id: &str) -> String {
        let catalog = self.tool_catalog(session_id, crate::ExecutionMode::Repl);
        tracing::debug!(
            session_id,
            tool_count = catalog.len(),
            tool_names = ?catalog
                .iter()
                .filter_map(|tool| tool.get("name").and_then(|value| value.as_str()))
                .collect::<Vec<_>>(),
            "serializing REPL tool catalog"
        );
        serde_json::to_string(&catalog).unwrap_or_else(|_| "[]".to_string())
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

    /// Execute code in the persistent lashlang REPL.
    pub async fn run_code(
        &mut self,
        session_id: &str,
        host: Arc<dyn SessionManager>,
        event_tx: &tokio::sync::mpsc::Sender<SessionEvent>,
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
            plugins: Arc::clone(self.plugins()),
            tools: self.tools(),
            surface: self.execution_surface(session_id, crate::ExecutionMode::Repl),
            host,
            session_id: session_id.to_string(),
            execution_mode: crate::ExecutionMode::Repl,
            event_tx: event_tx.clone(),
            turn_injection_bridge: self.turn_injection_bridge().clone(),
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
                                    call_id: None,
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
                        tool_calls: self.tool_calls.clone(),
                        images: std::mem::take(&mut self.tool_images),
                        error,
                        duration_ms: start.elapsed().as_millis() as u64,
                    });
                }
                LashlangResponse::Ready => {
                    // Unexpected but harmless
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
        tracing::debug!(
            code_preview = %code.chars().take(300).collect::<String>(),
            "checking REPL completeness"
        );
        self.runtime()?.send(LashlangRequest::CheckComplete {
            code: code.to_string(),
        })?;
        let runtime = self.runtime()?;
        let response = tokio::task::block_in_place(|| runtime.recv())
            .map_err(|_| SessionError::RuntimeExited)?;
        match response {
            LashlangResponse::CheckCompleteResult { is_complete } => {
                tracing::debug!(is_complete, "received REPL completeness result");
                Ok(is_complete)
            }
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

        self.last_repl_tools_json = None;
        Ok(())
    }

    /// Re-register the current tool definitions in the live REPL.
    /// This is intended for turn-boundary runtime reconfiguration.
    pub async fn refresh_execution_surface(&mut self) -> Result<(), SessionError> {
        if !self.supports_repl() {
            return Ok(());
        }

        let tools_json = self.repl_tools_json(&self.session_id);
        tracing::debug!(
            session_id = self.session_id,
            generation = self.tools().dynamic_generation().unwrap_or(0),
            tools_json_preview = %tools_json.chars().take(400).collect::<String>(),
            "refreshing REPL execution surface"
        );
        if self.last_repl_tools_json.as_deref() == Some(tools_json.as_str()) {
            return Ok(());
        }
        let generation = self.tools().dynamic_generation().unwrap_or(0);
        self.runtime()?.send(LashlangRequest::Reconfigure {
            tools_json: tools_json.clone(),
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
                    self.last_repl_tools_json = Some(tools_json.clone());
                    break;
                }
                _ => continue,
            }
        }

        Ok(())
    }

    async fn initialize_execution_surface(&mut self, session_id: &str) -> Result<(), SessionError> {
        let tools_json = self.repl_tools_json(session_id);
        tracing::debug!(
            session_id,
            tools_json_preview = %tools_json.chars().take(400).collect::<String>(),
            "initializing REPL execution surface"
        );
        self.runtime()?.send(LashlangRequest::Init {
            tools_json: tools_json.clone(),
            session_id: session_id.to_string(),
        })?;

        match self.runtime()?.recv()? {
            LashlangResponse::Ready => {
                self.last_repl_tools_json = Some(tools_json);
                Ok(())
            }
            other => Err(SessionError::Protocol(format!(
                "expected ready, got: {:?}",
                std::mem::discriminant(&other)
            ))),
        }
    }

    /// Snapshot execution-mode-local state, if any.
    pub async fn snapshot_execution_state(&mut self) -> Result<Option<Vec<u8>>, SessionError> {
        if !self.supports_repl() {
            return Ok(None);
        }
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
        Ok(Some(serde_json::to_vec(&combined).unwrap()))
    }

    /// Restore execution-mode-local state from a snapshot blob.
    pub async fn restore_execution_state(&mut self, data: &[u8]) -> Result<(), SessionError> {
        if !self.supports_repl() {
            return Ok(());
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plugin::StaticPluginFactory;
    use crate::tools::UpdatePlanTool;
    use crate::{PluginError, PluginHost, PluginSpec, SessionHandle, SessionSnapshot, TurnInput};

    struct NoopManager;

    #[async_trait::async_trait]
    impl SessionManager for NoopManager {
        async fn snapshot_current(&self) -> Result<SessionSnapshot, PluginError> {
            Err(PluginError::Session("snapshot unavailable".to_string()))
        }

        async fn snapshot_session(
            &self,
            _session_id: &str,
        ) -> Result<SessionSnapshot, PluginError> {
            Err(PluginError::Session("snapshot unavailable".to_string()))
        }

        async fn tool_catalog(
            &self,
            _session_id: &str,
        ) -> Result<Vec<serde_json::Value>, PluginError> {
            Err(PluginError::Session("tool catalog unavailable".to_string()))
        }

        async fn create_session(
            &self,
            _request: crate::SessionCreateRequest,
        ) -> Result<SessionHandle, PluginError> {
            Err(PluginError::Session(
                "session creation unavailable".to_string(),
            ))
        }

        async fn close_session(&self, _session_id: &str) -> Result<(), PluginError> {
            Err(PluginError::Session(
                "session close unavailable".to_string(),
            ))
        }

        async fn start_turn_stream(
            &self,
            _session_id: &str,
            _input: TurnInput,
        ) -> Result<crate::plugin::SessionTurnHandle, PluginError> {
            Err(PluginError::Session(
                "turn streaming unavailable".to_string(),
            ))
        }

        async fn await_turn(&self, _turn_id: &str) -> Result<crate::AssembledTurn, PluginError> {
            Err(PluginError::Session("await turn unavailable".to_string()))
        }

        async fn cancel_turn(&self, _turn_id: &str) -> Result<(), PluginError> {
            Err(PluginError::Session("cancel turn unavailable".to_string()))
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn run_code_reports_finish_as_an_error_after_tool_call() {
        let tools: Arc<dyn crate::ToolProvider> = Arc::new(UpdatePlanTool::new());
        let plugin_host = PluginHost::new(vec![Arc::new(StaticPluginFactory::new(
            "test_tools",
            PluginSpec::new().with_tool_provider(Arc::clone(&tools)),
        ))]);
        let plugin_session = plugin_host
            .build_session("root", crate::ExecutionMode::Repl, None)
            .expect("plugin session");
        let mut session = Session::new(
            crate::RuntimeServices::new(plugin_session),
            "root",
            crate::ExecutionMode::Repl,
        )
        .await
        .expect("session");

        let (event_tx, _event_rx) = tokio::sync::mpsc::channel(16);
        let manager: Arc<dyn SessionManager> = Arc::new(NoopManager);

        let response = session
            .run_code(
                "root",
                manager,
                &event_tx,
                r#"
call update_plan {
  explanation: "done",
  plan: [{ step: "ship it", status: "completed" }]
}
finish "ok"
"#,
            )
            .await
            .expect("exec response");

        assert_eq!(
            response.error,
            Some(
                "This lashlang step tried to terminate the task directly. End the task by replying in plain prose instead of calling `execute_lashlang` again.".to_string()
            )
        );
        assert_eq!(response.tool_calls.len(), 1);
        assert_eq!(response.tool_calls[0].tool, "update_plan");
        assert!(response.tool_calls[0].success);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn repl_run_code_can_call_common_repo_tools_without_model() {
        let temp = tempfile::tempdir().expect("tempdir");
        std::fs::write(temp.path().join("alpha.txt"), "hello\n").expect("write alpha");
        std::fs::write(temp.path().join("beta.rs"), "fn main() {}\n").expect("write beta");

        let deps = crate::DefaultToolPluginDeps {
            enable_user_prompts: true,
            ..Default::default()
        };
        let plugin_host = PluginHost::new(crate::default_tool_plugin_factories(
            crate::ExecutionMode::Repl,
            deps,
        ));
        let plugin_session = plugin_host
            .build_session("root", crate::ExecutionMode::Repl, None)
            .expect("plugin session");
        let mut session = Session::new(
            crate::RuntimeServices::new(plugin_session),
            "root",
            crate::ExecutionMode::Repl,
        )
        .await
        .expect("session");

        let (event_tx, _event_rx) = tokio::sync::mpsc::channel(16);
        let manager: Arc<dyn SessionManager> = Arc::new(NoopManager);

        let response = session
            .run_code(
                "root",
                manager,
                &event_tx,
                &format!(
                    r#"
cwd = {:?}
files = call ls {{ path: cwd }}
match = call grep {{ path: {:?}, pattern: "fn main" }}
observe files
observe match
"#,
                    temp.path(),
                    temp.path()
                ),
            )
            .await
            .expect("exec response");

        assert!(
            response.error.is_none(),
            "unexpected repl error: {:?}",
            response.error
        );
        assert_eq!(response.tool_calls.len(), 2);
        assert_eq!(response.tool_calls[0].tool, "ls");
        assert!(response.tool_calls[0].success);
        assert_eq!(response.tool_calls[1].tool, "grep");
        assert!(response.tool_calls[1].success);
    }
}
