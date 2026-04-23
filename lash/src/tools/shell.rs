use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{
    Arc, Mutex as StdMutex,
    atomic::{AtomicBool, AtomicI32, Ordering},
};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use portable_pty::{ChildKiller, CommandBuilder, MasterPty, PtySize, native_pty_system};
use serde_json::json;
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;

use crate::{
    ProgressSender, PromptContribution, SandboxMessage, ToolDefinition, ToolExecutionContext,
    ToolExecutionMode, ToolParam, ToolProvider, ToolResult,
};

use super::require_str;

struct ShellProcess {
    _master: Box<dyn MasterPty + Send>,
    writer: Arc<StdMutex<Option<Box<dyn Write + Send>>>>,
    buffer: Arc<StdMutex<Vec<u8>>>,
    buffer_start: Arc<StdMutex<usize>>,
    truncated: Arc<AtomicBool>,
    read_cursor: Arc<StdMutex<usize>>,
    exit_code: Arc<StdMutex<Option<i32>>>,
    exit_notify: Arc<Notify>,
    output_notify: Arc<Notify>,
    spill: Arc<StdMutex<Option<ShellOutputSpill>>>,
    killer: Arc<StdMutex<Option<Box<dyn ChildKiller + Send + Sync>>>>,
}

pub fn shell_prompt_contributions() -> Vec<PromptContribution> {
    vec![
        PromptContribution::guidance(
            "Command Execution",
            "Use `exec_command` for one-shot commands and to start long-lived processes. Use `write_stdin` only when the prior `exec_command` is still alive (it returned a `session_id`) — e.g. feeding a REPL or responding to a prompt; otherwise start a fresh `exec_command`. For services or background daemons, prefer startup patterns that survive after the tool call returns, then verify readiness from a fresh command before concluding.",
        ),
        PromptContribution::guidance(
            "Git Safety",
            "Avoid destructive git commands unless explicitly requested.",
        ),
    ]
}

#[derive(Clone)]
struct ProcessState {
    buffer: Arc<StdMutex<Vec<u8>>>,
    buffer_start: Arc<StdMutex<usize>>,
    exit_code: Arc<StdMutex<Option<i32>>>,
    exit_notify: Arc<Notify>,
    output_notify: Arc<Notify>,
    killer: Arc<StdMutex<Option<Box<dyn ChildKiller + Send + Sync>>>>,
}

struct ShellOutputSpill {
    path: PathBuf,
    file: File,
}

const MAX_OUTPUT: usize = 512_000;
const SPILL_OUTPUT_THRESHOLD: usize = 50 * 1024;
const DEFAULT_EXEC_YIELD_MS: u64 = 10_000;
const DEFAULT_WRITE_STDIN_YIELD_MS: u64 = 250;
const OUTPUT_QUIET_PERIOD_MS: u64 = 75;
const DEFAULT_PTY_SIZE: PtySize = PtySize {
    rows: 24,
    cols: 80,
    pixel_width: 0,
    pixel_height: 0,
};

#[derive(Clone, Debug)]
struct ExecCommandParams {
    cmd: String,
    workdir: PathBuf,
    shell_path: String,
    login: bool,
    yield_time_ms: u64,
    max_output_tokens: Option<usize>,
}

#[derive(Clone)]
struct ShellRuntime {
    shell_path: String,
    cwd: PathBuf,
    processes: Arc<StdMutex<HashMap<String, ShellProcess>>>,
    next_session_id: Arc<AtomicI32>,
}

#[derive(Clone, Copy)]
struct WaitBehavior {
    baseline_len: usize,
}

impl ShellRuntime {
    fn new() -> Self {
        let shell_path = std::env::var("SHELL").unwrap_or_else(|_| "bash".into());
        let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        Self {
            shell_path,
            cwd,
            processes: Arc::new(StdMutex::new(HashMap::new())),
            next_session_id: Arc::new(AtomicI32::new(1)),
        }
    }

    fn with_cwd(mut self, cwd: impl Into<PathBuf>) -> Self {
        self.cwd = cwd.into();
        self
    }

    fn shell_name(shell_path: &str) -> &str {
        shell_path.rsplit('/').next().unwrap_or(shell_path)
    }

    fn resolve_workdir(&self, workdir: Option<&str>) -> PathBuf {
        match workdir {
            None => self.cwd.clone(),
            Some(path) => {
                let path = PathBuf::from(path);
                if path.is_absolute() {
                    path
                } else {
                    self.cwd.join(path)
                }
            }
        }
    }

    fn command_for_spawn(&self, command: &str, shell_path: &str) -> String {
        let shell_name = Self::shell_name(shell_path);
        // Disable terminal echo so bytes delivered via `write_stdin`
        // don't appear in the captured output stream. The PTY allocates
        // with `ECHO` on by default (matching interactive terminals),
        // but agents drive these sessions programmatically and the echo
        // is pure noise. `stty -echo || true` keeps the prefix
        // harmless on environments where `stty` isn't available.
        let echo_off = "stty -echo 2>/dev/null || true\n";
        let pipefail_prefix = if command.contains('|') && shell_supports_pipefail(shell_name) {
            "set -o pipefail\n"
        } else {
            ""
        };
        format!("{echo_off}{pipefail_prefix}{command}")
    }

    fn shell_args(
        &self,
        command: &str,
        login: bool,
        shell_path: &str,
    ) -> Result<Vec<String>, String> {
        let command = self.command_for_spawn(command, shell_path);
        if login {
            if !shell_supports_login(Self::shell_name(shell_path)) {
                return Err(format!(
                    "Login shell mode is not supported for {}",
                    Self::shell_name(shell_path)
                ));
            }
            Ok(vec!["-l".to_string(), "-c".to_string(), command])
        } else {
            Ok(vec!["-c".to_string(), command])
        }
    }

    fn spawn_process(
        &self,
        id: String,
        command: &str,
        workdir: &Path,
        login: bool,
        shell_path: &str,
    ) -> Result<(), String> {
        let pty_system = native_pty_system();
        let pair = pty_system
            .openpty(DEFAULT_PTY_SIZE)
            .map_err(|err| format!("Failed to open PTY: {err}"))?;

        let mut cmd = CommandBuilder::new(shell_path);
        for arg in self.shell_args(command, login, shell_path)? {
            cmd.arg(arg);
        }
        cmd.cwd(workdir.as_os_str());

        let child = pair
            .slave
            .spawn_command(cmd)
            .map_err(|err| format!("Failed to spawn PTY command: {err}"))?;
        let killer = child.clone_killer();
        let reader = pair
            .master
            .try_clone_reader()
            .map_err(|err| format!("Failed to clone PTY reader: {err}"))?;
        let writer = pair
            .master
            .take_writer()
            .map_err(|err| format!("Failed to take PTY writer: {err}"))?;
        drop(pair.slave);

        let buffer = Arc::new(StdMutex::new(Vec::new()));
        let buffer_start = Arc::new(StdMutex::new(0usize));
        let truncated = Arc::new(AtomicBool::new(false));
        let read_cursor = Arc::new(StdMutex::new(0usize));
        let exit_code = Arc::new(StdMutex::new(None));
        let exit_notify = Arc::new(Notify::new());
        let output_notify = Arc::new(Notify::new());
        let spill = Arc::new(StdMutex::new(None));
        let killer = Arc::new(StdMutex::new(Some(killer)));

        spawn_reader_thread(
            id.clone(),
            reader,
            Arc::clone(&buffer),
            Arc::clone(&buffer_start),
            Arc::clone(&truncated),
            Arc::clone(&spill),
            Arc::clone(&output_notify),
        );
        spawn_wait_thread(
            child,
            Arc::clone(&exit_code),
            Arc::clone(&exit_notify),
            Arc::clone(&output_notify),
        );

        let process = ShellProcess {
            _master: pair.master,
            writer: Arc::new(StdMutex::new(Some(writer))),
            buffer,
            buffer_start,
            truncated,
            read_cursor,
            exit_code,
            exit_notify,
            output_notify,
            spill,
            killer,
        };
        self.processes.lock().unwrap().insert(id, process);
        Ok(())
    }

    fn allocate_handle_id(&self) -> String {
        self.next_session_id
            .fetch_add(1, Ordering::SeqCst)
            .to_string()
    }

    fn process_state(&self, id: &str) -> Result<ProcessState, String> {
        let procs = self.processes.lock().unwrap();
        let proc = procs
            .get(id)
            .ok_or_else(|| format!("No process with id: {id}"))?;
        Ok(ProcessState {
            buffer: Arc::clone(&proc.buffer),
            buffer_start: Arc::clone(&proc.buffer_start),
            exit_code: Arc::clone(&proc.exit_code),
            exit_notify: Arc::clone(&proc.exit_notify),
            output_notify: Arc::clone(&proc.output_notify),
            killer: Arc::clone(&proc.killer),
        })
    }

    fn output_state(&self, id: &str) -> Result<(usize, usize), String> {
        let procs = self.processes.lock().unwrap();
        let proc = procs
            .get(id)
            .ok_or_else(|| format!("No process with id: {id}"))?;
        let buffer_len = proc.buffer.lock().unwrap().len();
        let buffer_start = *proc.buffer_start.lock().unwrap();
        let read_cursor = *proc.read_cursor.lock().unwrap();
        Ok((buffer_start + buffer_len, read_cursor))
    }

    fn take_incremental_output(
        &self,
        id: &str,
        max_output_tokens: Option<usize>,
    ) -> Result<(String, Option<usize>, Option<PathBuf>), String> {
        let (buffer, buffer_start, truncated, read_cursor, spill) = {
            let procs = self.processes.lock().unwrap();
            let proc = procs
                .get(id)
                .ok_or_else(|| format!("Unknown session id {id}"))?;
            (
                Arc::clone(&proc.buffer),
                Arc::clone(&proc.buffer_start),
                Arc::clone(&proc.truncated),
                Arc::clone(&proc.read_cursor),
                Arc::clone(&proc.spill),
            )
        };

        let buf = buffer.lock().unwrap();
        let start_offset = *buffer_start.lock().unwrap();
        let end_offset = start_offset + buf.len();
        let mut cursor = read_cursor.lock().unwrap();
        let had_gap = *cursor < start_offset;
        let start = (*cursor).max(start_offset);
        let mut rendered =
            String::from_utf8_lossy(&buf[start.saturating_sub(start_offset)..]).to_string();
        *cursor = end_offset;
        if !rendered.is_empty()
            && (had_gap || truncated.load(Ordering::SeqCst) && *cursor == end_offset)
        {
            if !rendered.ends_with('\n') {
                rendered.push('\n');
            }
            rendered.push_str("[truncated]");
        }
        let rendered = clean_terminal_output(&rendered);
        let (rendered, original_token_count, token_truncated) =
            truncate_exec_output(rendered, max_output_tokens);
        let mut spill_guard = spill.lock().unwrap();
        let mut full_output_path = spill_guard.as_ref().map(|spill| spill.path.clone());
        if token_truncated && full_output_path.is_none() {
            full_output_path = activate_spill(id, &buf, &mut spill_guard);
        }
        Ok((rendered, original_token_count, full_output_path))
    }

    async fn wait_until_exit_or_timeout(
        &self,
        id: &str,
        timeout: Option<Duration>,
        progress: Option<&ProgressSender>,
        max_output_tokens: Option<usize>,
        behavior: WaitBehavior,
        cancel: Option<CancellationToken>,
    ) -> Result<PollOutcome, String> {
        let state = self.process_state(id)?;
        let deadline = timeout.map(|value| tokio::time::Instant::now() + value);
        let mut sent_len = behavior.baseline_len;

        loop {
            if let Some(token) = cancel.as_ref()
                && token.is_cancelled()
            {
                kill_child(&state);
                wait_for_child_exit(&state, Duration::from_millis(500)).await;
                return Ok(PollOutcome::Cancelled);
            }

            if let Some(tx) = progress {
                let new_chunk = {
                    let buf = state.buffer.lock().unwrap();
                    let buffer_start = *state.buffer_start.lock().unwrap();
                    let buffer_end = buffer_start + buf.len();
                    if buffer_end > sent_len {
                        let start = sent_len.max(buffer_start);
                        let mut chunk =
                            String::from_utf8_lossy(&buf[start.saturating_sub(buffer_start)..])
                                .to_string();
                        if sent_len < buffer_start && !chunk.is_empty() {
                            if !chunk.ends_with('\n') {
                                chunk.push('\n');
                            }
                            chunk.push_str("[truncated]");
                        }
                        sent_len = buffer_end;
                        Some(clean_terminal_output(&chunk))
                    } else {
                        None
                    }
                };
                if let Some(chunk) = new_chunk
                    && !chunk.is_empty()
                {
                    let _ = tx.send(SandboxMessage {
                        text: chunk,
                        kind: "tool_output".into(),
                    });
                }
            }

            let exited = state.exit_code.lock().unwrap().is_some();
            if exited {
                wait_for_buffer_settle(&state, Duration::from_millis(OUTPUT_QUIET_PERIOD_MS)).await;
                let (output, original_token_count, full_output_path) =
                    self.take_incremental_output(id, max_output_tokens)?;
                let exit_code = state.exit_code.lock().unwrap().unwrap_or(-1);
                return Ok(PollOutcome::Exited {
                    output,
                    original_token_count,
                    exit_code,
                    full_output_path,
                });
            }

            if let Some(dl) = deadline
                && tokio::time::Instant::now() >= dl
            {
                let exit_code = *state.exit_code.lock().unwrap();
                if let Some(exit_code) = exit_code {
                    wait_for_buffer_settle(&state, Duration::from_millis(OUTPUT_QUIET_PERIOD_MS))
                        .await;
                    let (output, original_token_count, full_output_path) =
                        self.take_incremental_output(id, max_output_tokens)?;
                    return Ok(PollOutcome::Exited {
                        output,
                        original_token_count,
                        exit_code,
                        full_output_path,
                    });
                }
                let (output, original_token_count, full_output_path) =
                    self.take_incremental_output(id, max_output_tokens)?;
                return Ok(PollOutcome::Running {
                    output,
                    original_token_count,
                    full_output_path,
                });
            }

            let cancel_future = async {
                match cancel.as_ref() {
                    Some(token) => token.cancelled().await,
                    None => std::future::pending::<()>().await,
                }
            };

            if let Some(wake_at) = deadline {
                tokio::select! {
                    _ = state.exit_notify.notified() => {}
                    _ = state.output_notify.notified() => {}
                    _ = tokio::time::sleep_until(wake_at) => {}
                    _ = cancel_future => {}
                }
            } else {
                tokio::select! {
                    _ = state.exit_notify.notified() => {}
                    _ = state.output_notify.notified() => {}
                    _ = cancel_future => {}
                }
            }
        }
    }

    fn remove_process(&self, id: &str) {
        if let Some(proc) = self.processes.lock().unwrap().remove(id)
            && let Some(mut spill) = proc.spill.lock().unwrap().take()
        {
            let _ = spill.file.flush();
        }
    }

    async fn write_stdin(&self, id: &str, input: &str) -> Result<(), String> {
        let writer = {
            let procs = self.processes.lock().unwrap();
            let proc = procs
                .get(id)
                .ok_or_else(|| format!("Unknown session id {id}"))?;
            Arc::clone(&proc.writer)
        };
        let input = input.to_string();
        tokio::task::spawn_blocking(move || {
            let mut writer = writer.lock().unwrap();
            let writer = writer
                .as_mut()
                .ok_or_else(|| "Process stdin not available".to_string())?;
            writer
                .write_all(input.as_bytes())
                .map_err(|err| format!("Write failed: {err}"))?;
            writer.flush().map_err(|err| format!("Flush failed: {err}"))
        })
        .await
        .map_err(|err| format!("Write task failed: {err}"))?
    }

    async fn close_stdin(&self, id: &str) -> Result<(), String> {
        let writer = {
            let procs = self.processes.lock().unwrap();
            let proc = procs
                .get(id)
                .ok_or_else(|| format!("Unknown session id {id}"))?;
            Arc::clone(&proc.writer)
        };
        tokio::task::spawn_blocking(move || {
            let mut writer = writer.lock().unwrap();
            writer.take();
            Ok(())
        })
        .await
        .map_err(|err| format!("Close stdin task failed: {err}"))?
    }
}

fn kill_child(state: &ProcessState) {
    if let Some(mut killer) = state.killer.lock().unwrap().take() {
        let _ = killer.kill();
    }
}

async fn wait_for_child_exit(state: &ProcessState, timeout: Duration) {
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        if state.exit_code.lock().unwrap().is_some() {
            return;
        }
        if tokio::time::Instant::now() >= deadline {
            return;
        }
        tokio::select! {
            _ = state.exit_notify.notified() => {
                if state.exit_code.lock().unwrap().is_some() {
                    return;
                }
            }
            _ = tokio::time::sleep_until(deadline) => return,
        }
    }
}

async fn wait_for_buffer_settle(state: &ProcessState, quiet_period: Duration) {
    let mut last_len = state.buffer.lock().unwrap().len();
    let mut quiet_until = tokio::time::Instant::now() + quiet_period;

    loop {
        tokio::select! {
            _ = state.output_notify.notified() => {
                let buffer_len = state.buffer.lock().unwrap().len();
                if buffer_len != last_len {
                    last_len = buffer_len;
                    quiet_until = tokio::time::Instant::now() + quiet_period;
                }
            }
            _ = tokio::time::sleep_until(quiet_until) => break,
        }
    }
}

enum PollOutcome {
    Running {
        output: String,
        original_token_count: Option<usize>,
        full_output_path: Option<PathBuf>,
    },
    Exited {
        output: String,
        original_token_count: Option<usize>,
        exit_code: i32,
        full_output_path: Option<PathBuf>,
    },
    Cancelled,
}

pub struct StandardShell {
    runtime: ShellRuntime,
}

impl StandardShell {
    pub fn new() -> Self {
        Self {
            runtime: ShellRuntime::new(),
        }
    }

    pub fn with_cwd(mut self, cwd: impl Into<PathBuf>) -> Self {
        self.runtime = self.runtime.with_cwd(cwd);
        self
    }

    fn parse_exec_command_params(
        &self,
        args: &serde_json::Value,
    ) -> Result<ExecCommandParams, ToolResult> {
        let cmd = require_str(args, "cmd")?.to_string();
        let workdir = self.runtime.resolve_workdir(
            args.get("workdir")
                .and_then(|value| value.as_str())
                .filter(|value| !value.is_empty()),
        );
        let shell_path = args
            .get("shell")
            .and_then(|value| value.as_str())
            .filter(|value| !value.is_empty())
            .unwrap_or(&self.runtime.shell_path)
            .to_string();
        let login = args
            .get("login")
            .and_then(|value| value.as_bool())
            .unwrap_or(false);
        let yield_time_ms = args
            .get("yield_time_ms")
            .and_then(|value| value.as_u64())
            .unwrap_or(DEFAULT_EXEC_YIELD_MS);
        let max_output_tokens = args
            .get("max_output_tokens")
            .and_then(|value| value.as_u64())
            .map(|value| value as usize);

        Ok(ExecCommandParams {
            cmd,
            workdir,
            shell_path,
            login,
            yield_time_ms,
            max_output_tokens,
        })
    }

    async fn exec_command(
        &self,
        params: &ExecCommandParams,
        progress: Option<&ProgressSender>,
        cancel: Option<CancellationToken>,
    ) -> ToolResult {
        let started = Instant::now();
        let handle_id = self.runtime.allocate_handle_id();

        if let Err(err) = self.runtime.spawn_process(
            handle_id.clone(),
            &params.cmd,
            &params.workdir,
            params.login,
            &params.shell_path,
        ) {
            return ToolResult::err(json!(err));
        }

        match self
            .runtime
            .wait_until_exit_or_timeout(
                &handle_id,
                Some(Duration::from_millis(params.yield_time_ms)),
                progress,
                params.max_output_tokens,
                WaitBehavior { baseline_len: 0 },
                cancel,
            )
            .await
        {
            Ok(PollOutcome::Running {
                output,
                original_token_count,
                full_output_path,
                ..
            }) => ToolResult::ok(standard_shell_io_record(
                &handle_id,
                output,
                None,
                original_token_count,
                full_output_path.as_deref(),
                started.elapsed().as_secs_f64(),
            )),
            Ok(PollOutcome::Exited {
                output,
                original_token_count,
                exit_code,
                full_output_path,
            }) => {
                self.runtime.remove_process(&handle_id);
                ToolResult::ok(standard_shell_io_record(
                    &handle_id,
                    output,
                    Some(exit_code),
                    original_token_count,
                    full_output_path.as_deref(),
                    started.elapsed().as_secs_f64(),
                ))
            }
            Ok(PollOutcome::Cancelled) => {
                self.runtime.remove_process(&handle_id);
                ToolResult::err_fmt("tool call cancelled")
            }
            Err(err) => {
                self.runtime.remove_process(&handle_id);
                ToolResult::err(json!(err))
            }
        }
    }

    async fn write_stdin_call(
        &self,
        args: &serde_json::Value,
        progress: Option<&ProgressSender>,
        cancel: Option<CancellationToken>,
    ) -> ToolResult {
        let id = match parse_standard_session_id(args) {
            Ok(value) => value,
            Err(err) => return err,
        };
        let chars = args
            .get("chars")
            .and_then(|value| value.as_str())
            .unwrap_or("");
        let yield_time_ms = args
            .get("yield_time_ms")
            .and_then(|value| value.as_u64())
            .unwrap_or(DEFAULT_WRITE_STDIN_YIELD_MS);
        let close_stdin = args
            .get("close_stdin")
            .and_then(|value| value.as_bool())
            .unwrap_or(false);
        let max_output_tokens = args
            .get("max_output_tokens")
            .and_then(|value| value.as_u64())
            .map(|value| value as usize);
        let started = Instant::now();
        let (baseline_len, _) = match self.runtime.output_state(&id) {
            Ok(state) => state,
            Err(err) => return ToolResult::err(json!(err)),
        };

        if let Err(err) = self.runtime.write_stdin(&id, chars).await {
            return ToolResult::err(json!(err));
        }
        if close_stdin && let Err(err) = self.runtime.close_stdin(&id).await {
            return ToolResult::err(json!(err));
        }

        match self
            .runtime
            .wait_until_exit_or_timeout(
                &id,
                Some(Duration::from_millis(yield_time_ms)),
                progress,
                max_output_tokens,
                WaitBehavior { baseline_len },
                cancel,
            )
            .await
        {
            Ok(PollOutcome::Running {
                output,
                original_token_count,
                full_output_path,
                ..
            }) => ToolResult::ok(standard_shell_io_record(
                &id,
                output,
                None,
                original_token_count,
                full_output_path.as_deref(),
                started.elapsed().as_secs_f64(),
            )),
            Ok(PollOutcome::Exited {
                output,
                original_token_count,
                exit_code,
                full_output_path,
            }) => {
                self.runtime.remove_process(&id);
                ToolResult::ok(standard_shell_io_record(
                    &id,
                    output,
                    Some(exit_code),
                    original_token_count,
                    full_output_path.as_deref(),
                    started.elapsed().as_secs_f64(),
                ))
            }
            Ok(PollOutcome::Cancelled) => {
                self.runtime.remove_process(&id);
                ToolResult::err_fmt("tool call cancelled")
            }
            Err(err) => ToolResult::err(json!(err)),
        }
    }
}

impl Default for StandardShell {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl ToolProvider for StandardShell {
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![
            ToolDefinition {
                name: "exec_command".into(),
                description: "Run a command in a PTY. Completed commands return cleaned `output` and `exit_code`; longer-running commands return `session_id` so you can continue the same process with `write_stdin`. ANSI/control noise is stripped from returned output. Large or truncated output may also include `full_output_path` pointing at the saved raw stream.".into(),
                params: vec![
                    ToolParam {
                        name: "cmd".into(),
                        r#type: "str".into(),
                        description: "Shell command to execute.".into(),
                        default_value: None,
                        required: true,
                    },
                    ToolParam {
                        name: "workdir".into(),
                        r#type: "str".into(),
                        description:
                            "Optional working directory to run the command in; defaults to the turn cwd.".into(),
                        default_value: None,
                        required: false,
                    },
                    ToolParam {
                        name: "shell".into(),
                        r#type: "str".into(),
                        description: "Shell binary to launch. Defaults to the user's default shell.".into(),
                        default_value: None,
                        required: false,
                    },
                    ToolParam {
                        name: "login".into(),
                        r#type: "bool".into(),
                        description: "Whether to run the shell with -l semantics. Defaults to false to avoid startup prompts and shell init noise.".into(),
                        default_value: Some(serde_json::json!(false)),
                        required: false,
                    },
                    ToolParam {
                        name: "yield_time_ms".into(),
                        r#type: "int".into(),
                        description: "How long to wait (in milliseconds) for output before yielding.".into(),
                        default_value: None,
                        required: false,
                    },
                    ToolParam {
                        name: "max_output_tokens".into(),
                        r#type: "int".into(),
                        description: "Maximum number of tokens to return. Excess output will be truncated.".into(),
                        default_value: None,
                        required: false,
                    },
                ],
                returns: "dict".into(),
                examples: vec![],
                availability: crate::ToolAvailabilityConfig::documented(),
                activation: crate::ToolActivation::Always,
                availability_override: None,
                input_schema_override: None,
                output_schema_override: None,
                // exec_command can fork/move/delete files and mutate shell
                // state (cwd, env, background processes); serialize it so
                // concurrent commands don't race.
                execution_mode: ToolExecutionMode::Serial,
            },
            ToolDefinition {
                name: "write_stdin".into(),
                description: "Write bytes to a running command handle and wait briefly for the next settled cleaned output chunk. Use `close_stdin: true` to send EOF. ANSI/control noise is stripped from returned output. Large or truncated output may also include `full_output_path` pointing at the saved raw stream.".into(),
                params: vec![
                    ToolParam {
                        name: "session_id".into(),
                        r#type: "int".into(),
                        description: "Identifier of the running command handle.".into(),
                        default_value: None,
                        required: true,
                    },
                    ToolParam {
                        name: "chars".into(),
                        r#type: "str".into(),
                        description: "Bytes to write to stdin (may be empty to poll).".into(),
                        default_value: Some(serde_json::json!("")),
                        required: false,
                    },
                    ToolParam {
                        name: "yield_time_ms".into(),
                        r#type: "int".into(),
                        description: "How long to wait (in milliseconds) for output before yielding.".into(),
                        default_value: None,
                        required: false,
                    },
                    ToolParam {
                        name: "close_stdin".into(),
                        r#type: "bool".into(),
                        description: "Close stdin after writing to send EOF to the process.".into(),
                        default_value: Some(serde_json::json!(false)),
                        required: false,
                    },
                    ToolParam {
                        name: "max_output_tokens".into(),
                        r#type: "int".into(),
                        description: "Maximum number of tokens to return. Excess output will be truncated.".into(),
                        default_value: None,
                        required: false,
                    },
                ],
                returns: "dict".into(),
                examples: vec![],
                availability: crate::ToolAvailabilityConfig::documented(),
                activation: crate::ToolActivation::Always,
                availability_override: None,
                input_schema_override: None,
                output_schema_override: None,
                // write_stdin targets a specific running command session and
                // mutates its state; serialize it alongside exec_command.
                execution_mode: ToolExecutionMode::Serial,
            },
        ]
    }

    async fn execute(&self, name: &str, args: &serde_json::Value) -> ToolResult {
        self.execute_streaming(name, args, None).await
    }

    async fn execute_streaming(
        &self,
        name: &str,
        args: &serde_json::Value,
        progress: Option<&ProgressSender>,
    ) -> ToolResult {
        self.dispatch(name, args, progress, None).await
    }

    async fn execute_streaming_with_context(
        &self,
        name: &str,
        args: &serde_json::Value,
        context: &ToolExecutionContext,
        progress: Option<&ProgressSender>,
    ) -> ToolResult {
        self.dispatch(name, args, progress, context.cancellation_token.clone())
            .await
    }
}

impl StandardShell {
    async fn dispatch(
        &self,
        name: &str,
        args: &serde_json::Value,
        progress: Option<&ProgressSender>,
        cancel: Option<CancellationToken>,
    ) -> ToolResult {
        match name {
            "exec_command" => {
                let params = match self.parse_exec_command_params(args) {
                    Ok(params) => params,
                    Err(err) => return err,
                };
                self.exec_command(&params, progress, cancel).await
            }
            "write_stdin" => self.write_stdin_call(args, progress, cancel).await,
            _ => ToolResult::err_fmt(format_args!("Unknown tool: {name}")),
        }
    }
}

fn spawn_reader_thread(
    id: String,
    mut reader: Box<dyn Read + Send>,
    buffer: Arc<StdMutex<Vec<u8>>>,
    buffer_start: Arc<StdMutex<usize>>,
    truncated: Arc<AtomicBool>,
    spill: Arc<StdMutex<Option<ShellOutputSpill>>>,
    output_notify: Arc<Notify>,
) {
    thread::spawn(move || {
        let mut chunk = [0u8; 4096];
        loop {
            match reader.read(&mut chunk) {
                Ok(0) => break,
                Ok(n) => {
                    {
                        let mut buf = buffer.lock().unwrap();
                        let mut spill = spill.lock().unwrap();
                        if buf.len() + n > SPILL_OUTPUT_THRESHOLD {
                            let _ = activate_spill(&id, &buf, &mut spill);
                        }
                        let mut clear_spill = false;
                        if let Some(spill_file) = spill.as_mut()
                            && spill_file.file.write_all(&chunk[..n]).is_err()
                        {
                            clear_spill = true;
                        }
                        if clear_spill {
                            *spill = None;
                        }

                        buf.extend_from_slice(&chunk[..n]);
                        if buf.len() > MAX_OUTPUT {
                            let to_drop = buf.len() - MAX_OUTPUT;
                            buf.drain(..to_drop);
                            *buffer_start.lock().unwrap() += to_drop;
                            truncated.store(true, Ordering::SeqCst);
                        }
                    }
                    output_notify.notify_waiters();
                }
                Err(err) if err.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(_) => break,
            }
        }
        output_notify.notify_waiters();
    });
}

fn spawn_wait_thread(
    mut child: Box<dyn portable_pty::Child + Send + Sync>,
    exit_code: Arc<StdMutex<Option<i32>>>,
    exit_notify: Arc<Notify>,
    output_notify: Arc<Notify>,
) {
    thread::spawn(move || {
        let code = child
            .wait()
            .map(|status| i32::try_from(status.exit_code()).unwrap_or(i32::MAX))
            .unwrap_or(-1);
        *exit_code.lock().unwrap() = Some(code);
        exit_notify.notify_waiters();
        output_notify.notify_waiters();
    });
}

fn shell_supports_pipefail(shell_name: &str) -> bool {
    matches!(shell_name, "bash" | "zsh" | "ksh" | "mksh")
}

fn shell_supports_login(shell_name: &str) -> bool {
    matches!(shell_name, "bash" | "zsh" | "ksh" | "mksh" | "fish")
}

fn shell_output_dir() -> std::io::Result<PathBuf> {
    let dir = std::env::temp_dir().join("lash-tool-output");
    fs::create_dir_all(&dir)?;
    Ok(dir)
}

fn shell_output_path(id: &str) -> std::io::Result<PathBuf> {
    let dir = shell_output_dir()?;
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    Ok(dir.join(format!("exec_command-{id}-{nonce}.log")))
}

fn activate_spill(
    id: &str,
    existing_output: &[u8],
    spill: &mut Option<ShellOutputSpill>,
) -> Option<PathBuf> {
    if let Some(spill) = spill.as_ref() {
        return Some(spill.path.clone());
    }

    let path = shell_output_path(id).ok()?;
    let mut file = File::create(&path).ok()?;
    if file.write_all(existing_output).is_err() {
        let _ = fs::remove_file(&path);
        return None;
    }
    *spill = Some(ShellOutputSpill {
        path: path.clone(),
        file,
    });
    Some(path)
}

fn clean_terminal_output(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut chars = input.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\x1b' {
            match chars.peek().copied() {
                Some('[') => {
                    chars.next();
                    for next in chars.by_ref() {
                        if ('@'..='~').contains(&next) {
                            break;
                        }
                    }
                }
                Some(']') => {
                    chars.next();
                    let mut previous_was_escape = false;
                    for next in chars.by_ref() {
                        if next == '\x07' || (previous_was_escape && next == '\\') {
                            break;
                        }
                        previous_was_escape = next == '\x1b';
                    }
                }
                Some(_) => {
                    chars.next();
                }
                None => {}
            }
            continue;
        }
        match ch {
            '\r' => {
                if !matches!(chars.peek(), Some('\n')) {
                    out.push('\n');
                }
            }
            '\x08' => {
                out.pop();
            }
            ch if ch.is_control() && ch != '\n' && ch != '\t' => {}
            ch => out.push(ch),
        }
    }
    out
}

fn truncate_exec_output(
    output: String,
    max_output_tokens: Option<usize>,
) -> (String, Option<usize>, bool) {
    let original_token_count = max_output_tokens.map(|_| estimate_token_count(&output));
    let Some(limit) = max_output_tokens else {
        return (output, original_token_count, false);
    };
    let max_chars = limit.saturating_mul(4);
    let char_count = output.chars().count();
    if char_count <= max_chars {
        return (output, original_token_count, false);
    }
    let truncated = output.chars().take(max_chars).collect::<String>() + "\n[truncated]";
    (truncated, original_token_count, true)
}

fn estimate_token_count(text: &str) -> usize {
    text.chars().count().div_ceil(4)
}

fn standard_shell_io_record(
    id: &str,
    output: String,
    exit_code: Option<i32>,
    original_token_count: Option<usize>,
    full_output_path: Option<&Path>,
    wall_time_seconds: f64,
) -> serde_json::Value {
    let session_id = exit_code
        .is_none()
        .then(|| id.parse::<i64>().ok())
        .flatten();
    let mut record = serde_json::Map::new();
    record.insert("output".into(), json!(output));
    record.insert("wall_time_seconds".into(), json!(wall_time_seconds));
    if let Some(exit_code) = exit_code {
        record.insert("exit_code".into(), json!(exit_code));
    }
    if let Some(session_id) = session_id {
        record.insert("session_id".into(), json!(session_id));
    }
    if let Some(original_token_count) = original_token_count {
        record.insert("original_token_count".into(), json!(original_token_count));
    }
    if let Some(path) = full_output_path {
        record.insert(
            "full_output_path".into(),
            json!(path.to_string_lossy().to_string()),
        );
    }
    serde_json::Value::Object(record)
}

fn parse_standard_session_id(args: &serde_json::Value) -> Result<String, ToolResult> {
    if let Some(value) = args.get("session_id") {
        if let Some(id) = value.as_i64() {
            return Ok(id.to_string());
        }
        if let Some(id) = value.as_u64() {
            return Ok(id.to_string());
        }
        return Err(ToolResult::err_fmt(format_args!(
            "Invalid session_id: expected int"
        )));
    }

    require_str(args, "id").map(str::to_string)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::fs;

    #[tokio::test]
    async fn exec_command_returns_exit_code_when_command_finishes() {
        let shell = StandardShell::default();
        let result = shell
            .execute("exec_command", &json!({"cmd": "echo hello"}))
            .await;
        assert!(result.success);
        assert!(result.result.get("session_id").is_none());
        assert_eq!(result.result["exit_code"], 0);
        assert!(result.result["wall_time_seconds"].as_f64().is_some());
        assert!(result.result["output"].as_str().unwrap().contains("hello"));
    }

    #[tokio::test]
    async fn exec_command_returns_handle_id_for_running_process() {
        let shell = StandardShell::new().with_cwd("/");
        let result = shell
            .execute(
                "exec_command",
                &json!({"cmd": "sleep 1; echo done", "yield_time_ms": 10}),
            )
            .await;
        assert!(result.success);
        assert!(result.result["session_id"].as_i64().is_some());
        assert!(result.result["exit_code"].is_null());
    }

    #[tokio::test]
    async fn write_stdin_reuses_running_exec_handle() {
        let shell = StandardShell::default();
        let cmd = "python3 -u -c 'import sys; line = sys.stdin.readline(); print(\"got:\" + line.strip())'";
        let open = shell
            .execute(
                "exec_command",
                &json!({"cmd": cmd, "yield_time_ms": 10, "login": false}),
            )
            .await;
        assert!(open.success, "{}", open.result);
        let session_id = open.result["session_id"].as_i64().unwrap();

        let result = shell
            .execute(
                "write_stdin",
                &json!({"session_id": session_id, "chars": "hello\n", "yield_time_ms": 1000}),
            )
            .await;
        assert!(result.success);
        assert!(result.result.get("session_id").is_none());
        assert_eq!(result.result["exit_code"], 0);
        assert!(
            result.result["output"]
                .as_str()
                .unwrap()
                .contains("got:hello")
        );
    }

    #[tokio::test]
    async fn write_stdin_prefers_completed_state_for_short_lived_commands() {
        let shell = StandardShell::default();
        let cmd = "python3 -u -c 'import sys; line = sys.stdin.readline(); print(\"got:\" + line.strip())'";
        for _ in 0..16 {
            let open = shell
                .execute(
                    "exec_command",
                    &json!({"cmd": cmd, "yield_time_ms": 10, "login": false}),
                )
                .await;
            assert!(open.success);
            let session_id = open.result["session_id"].as_i64().unwrap();

            let result = shell
                .execute(
                    "write_stdin",
                    &json!({"session_id": session_id, "chars": "hello\n", "yield_time_ms": 1000}),
                )
                .await;
            assert!(result.success);
            assert!(
                result.result.get("session_id").is_none(),
                "expected completed handle, got: {}",
                result.result
            );
            assert_eq!(result.result["exit_code"], 0);
            assert!(
                result.result["output"]
                    .as_str()
                    .unwrap()
                    .contains("got:hello")
            );
        }
    }

    #[tokio::test]
    async fn write_stdin_can_close_stdin_to_send_eof() {
        let shell = StandardShell::default();
        let open = shell
            .execute(
                "exec_command",
                &json!({"cmd": "cat", "yield_time_ms": 10, "login": false}),
            )
            .await;
        assert!(open.success);
        let session_id = open.result["session_id"].as_i64().unwrap();

        let result = shell
            .execute(
                "write_stdin",
                &json!({"session_id": session_id, "chars": "hello", "close_stdin": true, "yield_time_ms": 1000}),
            )
            .await;
        assert!(result.success, "{}", result.result);
        assert!(result.result.get("session_id").is_none());
        assert_eq!(result.result["exit_code"], 0);
        let output = result.result["output"].as_str().unwrap();
        assert!(
            output.contains("hello"),
            "expected cat to echo input, got: {output}"
        );
    }

    #[tokio::test]
    async fn exec_command_honors_workdir() {
        let shell = StandardShell::new().with_cwd("/");
        let result = shell
            .execute("exec_command", &json!({"cmd": "pwd", "workdir": "tmp"}))
            .await;
        assert!(result.success);
        assert_eq!(result.result["output"].as_str().unwrap().trim_end(), "/tmp");
    }

    #[tokio::test]
    async fn exec_command_pipeline_failure_uses_pipefail() {
        let shell = StandardShell::default();
        let result = shell
            .execute("exec_command", &json!({"cmd": "false | cat"}))
            .await;
        assert!(result.success);
        assert_ne!(result.result["exit_code"], 0);
    }

    #[tokio::test]
    async fn exec_command_reports_full_output_path_when_token_truncated() {
        let shell = StandardShell::default();
        let result = shell
            .execute(
                "exec_command",
                &json!({"cmd": "python3 -c 'print(\"hello \" * 4000)'", "max_output_tokens": 16, "login": false}),
            )
            .await;
        assert!(result.success, "{}", result.result);
        let output = result.result["output"].as_str().unwrap();
        let full_output_path = result.result["full_output_path"].as_str().unwrap();
        let full_output = fs::read_to_string(full_output_path).expect("full output file");
        assert!(output.contains("[truncated]"));
        assert!(full_output.contains("hello hello"));
    }

    #[tokio::test]
    async fn exec_command_spills_full_output_when_buffer_overflows() {
        let shell = StandardShell::default();
        let result = shell
            .execute(
                "exec_command",
                &json!({"cmd": format!("python3 -c 'import sys; sys.stdout.write(\"x\" * {})'", MAX_OUTPUT + 8192), "login": false}),
            )
            .await;
        assert!(result.success, "{}", result.result);
        let output = result.result["output"].as_str().unwrap();
        let full_output_path = result.result["full_output_path"].as_str().unwrap();
        let full_output = fs::read_to_string(full_output_path).expect("full output file");
        assert!(output.contains("[truncated]"));
        assert!(full_output.len() >= MAX_OUTPUT + 8192);
    }

    #[tokio::test]
    async fn exec_command_reports_full_output_path_for_large_output() {
        let shell = StandardShell::default();
        let result = shell
            .execute(
                "exec_command",
                &json!({"cmd": format!("python3 -c 'import sys; sys.stdout.write(\"x\" * {})'", SPILL_OUTPUT_THRESHOLD + 4096), "login": false}),
            )
            .await;
        assert!(result.success, "{}", result.result);
        assert!(result.result["output"].as_str().is_some());
        let full_output_path = result.result["full_output_path"].as_str().unwrap();
        let full_output = fs::read_to_string(full_output_path).expect("full output file");
        assert!(full_output.len() >= SPILL_OUTPUT_THRESHOLD + 4096);
    }

    #[tokio::test]
    async fn write_stdin_reports_full_output_path_when_token_truncated() {
        let shell = StandardShell::default();
        let cmd = "python3 -u -c 'import sys; data = sys.stdin.read(); sys.stdout.write(data)'";
        let open = shell
            .execute(
                "exec_command",
                &json!({"cmd": cmd, "yield_time_ms": 10, "login": false}),
            )
            .await;
        assert!(open.success, "{}", open.result);
        let session_id = open.result["session_id"].as_i64().unwrap();
        let payload = "segment ".repeat(5000);

        let result = shell
            .execute(
                "write_stdin",
                &json!({"session_id": session_id, "chars": payload, "close_stdin": true, "yield_time_ms": 1000, "max_output_tokens": 24}),
            )
            .await;
        assert!(result.success, "{}", result.result);
        let output = result.result["output"].as_str().unwrap();
        let full_output_path = result.result["full_output_path"].as_str().unwrap();
        let full_output = fs::read_to_string(full_output_path).expect("full output file");
        assert!(output.contains("[truncated]"));
        assert!(full_output.contains("segment segment"));
    }

    #[test]
    fn shell_definitions_are_compact_and_non_empty() {
        let shell = StandardShell::default();
        let defs = shell.definitions();
        assert_eq!(defs.len(), 2);
        assert!(defs.iter().all(|def| !def.description.is_empty()));
    }

    #[test]
    fn exec_command_defaults_to_non_login_shell() {
        let shell = StandardShell::default();
        let params = shell
            .parse_exec_command_params(&json!({"cmd": "echo hello"}))
            .expect("params");

        assert!(!params.login);
    }

    #[test]
    fn clean_terminal_output_strips_ansi_and_controls() {
        let raw = "\x1b[?2004h\x1b[31mred\x1b[0m\r\nab\x08c\x1b]0;title\x07\x00";

        assert_eq!(clean_terminal_output(raw), "red\nac");
    }

    #[tokio::test]
    async fn exec_command_cancel_token_kills_running_child() {
        use crate::testing::MockSessionManager;
        use std::sync::Arc;
        use std::time::Instant;

        let shell = StandardShell::default();
        let token = CancellationToken::new();
        let ctx = ToolExecutionContext {
            session_id: "test".to_string(),
            host: Arc::new(MockSessionManager::default()),
            cancellation_token: Some(token.clone()),
            async_task_id: None,
        };

        // A long-running sleep that would otherwise hold the tool call for
        // 5s. The dispatcher must return promptly once the token fires, and
        // the PTY child must be killed rather than left to run.
        let args = json!({
            "cmd": "sleep 5",
            "yield_time_ms": 30_000,
            "login": false,
        });

        let cancel_handle = {
            let token = token.clone();
            tokio::spawn(async move {
                tokio::time::sleep(Duration::from_millis(100)).await;
                token.cancel();
            })
        };

        let started = Instant::now();
        let result = shell
            .execute_streaming_with_context("exec_command", &args, &ctx, None)
            .await;
        let elapsed = started.elapsed();
        let _ = cancel_handle.await;

        assert!(
            elapsed < Duration::from_secs(1),
            "cancelled dispatch should return in under 1s (took {elapsed:?})"
        );
        assert!(!result.success, "cancelled result should be an error");
        assert_eq!(result.result.as_str(), Some("tool call cancelled"));
    }

    #[tokio::test]
    async fn cancel_during_write_stdin_wait_kills_child_by_pid() {
        use crate::testing::MockSessionManager;
        use std::sync::Arc;
        use std::time::Instant;

        fn pid_alive(pid: i32) -> bool {
            // On Linux, /proc/<pid> disappears once the kernel reaps the
            // task. Use that as a portable stand-in for kill(pid, 0) without
            // pulling in a new dep.
            std::path::Path::new(&format!("/proc/{pid}")).exists()
        }

        let shell = StandardShell::default();
        let token = CancellationToken::new();
        let ctx = ToolExecutionContext {
            session_id: "test".to_string(),
            host: Arc::new(MockSessionManager::default()),
            cancellation_token: Some(token.clone()),
            async_task_id: None,
        };

        // Open a long-lived child. `echo $$` reports the shell's pid, then
        // `exec sleep 5` replaces the shell with sleep so the printed pid is
        // exactly the process the ChildKiller targets.
        let args = json!({
            "cmd": "echo $$; exec sleep 5",
            "yield_time_ms": 500,
            "login": false,
        });
        let open = shell.execute("exec_command", &args).await;
        assert!(open.success, "{}", open.result);
        let session_id = open.result["session_id"]
            .as_i64()
            .expect("expected a running session_id");
        let captured = open.result["output"].as_str().unwrap_or("");
        let pid: Option<i32> = captured
            .lines()
            .find_map(|line| line.trim().parse::<i32>().ok());

        let cancel_handle = {
            let token = token.clone();
            tokio::spawn(async move {
                tokio::time::sleep(Duration::from_millis(100)).await;
                token.cancel();
            })
        };

        let started = Instant::now();
        let result = shell
            .execute_streaming_with_context(
                "write_stdin",
                &json!({"session_id": session_id, "chars": "", "yield_time_ms": 30_000}),
                &ctx,
                None,
            )
            .await;
        let elapsed = started.elapsed();
        let _ = cancel_handle.await;

        assert!(
            elapsed < Duration::from_secs(1),
            "cancelled dispatch should return in under 1s (took {elapsed:?})"
        );
        assert!(!result.success, "cancelled result should be an error");
        assert_eq!(result.result.as_str(), Some("tool call cancelled"));

        if let Some(pid) = pid
            && cfg!(target_os = "linux")
        {
            // Give the kernel a moment to reap.
            let mut gone = false;
            for _ in 0..50 {
                if !pid_alive(pid) {
                    gone = true;
                    break;
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
            assert!(gone, "child pid {pid} was still alive after cancel");
        }
    }
}
