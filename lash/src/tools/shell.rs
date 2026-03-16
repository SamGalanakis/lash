use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{
    Arc, Mutex as StdMutex,
    atomic::{AtomicBool, AtomicI32, Ordering},
};
use std::thread;
use std::time::{Duration, Instant};

use portable_pty::{CommandBuilder, MasterPty, PtySize, native_pty_system};
use serde_json::json;
use tokio::sync::Notify;

use crate::{
    ProgressSender, PromptContribution, SandboxMessage, ToolDefinition, ToolParam, ToolProvider,
    ToolResult,
};

use super::require_str;

struct ShellProcess {
    _master: Box<dyn MasterPty + Send>,
    writer: Arc<StdMutex<Option<Box<dyn Write + Send>>>>,
    buffer: Arc<StdMutex<Vec<u8>>>,
    truncated: Arc<AtomicBool>,
    read_cursor: Arc<StdMutex<usize>>,
    exit_code: Arc<StdMutex<Option<i32>>>,
    exit_notify: Arc<Notify>,
    output_notify: Arc<Notify>,
}

pub(crate) fn shell_prompt_contributions() -> Vec<PromptContribution> {
    vec![PromptContribution::guidance(
        "### Command Execution\nUse `exec_command` for one-shot commands and for starting long-lived processes. If it returns `session_id`, continue that same process with `write_stdin`; otherwise the command already exited. For services or background daemons, prefer startup patterns that survive after the tool call returns, then verify readiness from a fresh command before concluding.\n\n### Git Safety\nDo not revert user changes you did not make. Avoid destructive git commands unless explicitly requested.",
    )]
}

#[derive(Clone)]
struct ProcessState {
    buffer: Arc<StdMutex<Vec<u8>>>,
    exit_code: Arc<StdMutex<Option<i32>>>,
    exit_notify: Arc<Notify>,
    output_notify: Arc<Notify>,
}

const MAX_OUTPUT: usize = 512_000;
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
        if !command.contains('|') || !shell_supports_pipefail(shell_name) {
            return command.to_string();
        }
        format!("set -o pipefail\n{command}")
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
        let truncated = Arc::new(AtomicBool::new(false));
        let read_cursor = Arc::new(StdMutex::new(0usize));
        let exit_code = Arc::new(StdMutex::new(None));
        let exit_notify = Arc::new(Notify::new());
        let output_notify = Arc::new(Notify::new());

        spawn_reader_thread(
            reader,
            Arc::clone(&buffer),
            Arc::clone(&truncated),
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
            truncated,
            read_cursor,
            exit_code,
            exit_notify,
            output_notify,
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
            exit_code: Arc::clone(&proc.exit_code),
            exit_notify: Arc::clone(&proc.exit_notify),
            output_notify: Arc::clone(&proc.output_notify),
        })
    }

    fn output_state(&self, id: &str) -> Result<(usize, usize), String> {
        let procs = self.processes.lock().unwrap();
        let proc = procs
            .get(id)
            .ok_or_else(|| format!("No process with id: {id}"))?;
        let buffer_len = proc.buffer.lock().unwrap().len();
        let read_cursor = *proc.read_cursor.lock().unwrap();
        Ok((buffer_len, read_cursor))
    }

    fn take_incremental_output(
        &self,
        id: &str,
        max_output_tokens: Option<usize>,
    ) -> Result<(String, Option<usize>), String> {
        let (buffer, truncated, read_cursor) = {
            let procs = self.processes.lock().unwrap();
            let proc = procs
                .get(id)
                .ok_or_else(|| format!("Unknown session id {id}"))?;
            (
                Arc::clone(&proc.buffer),
                Arc::clone(&proc.truncated),
                Arc::clone(&proc.read_cursor),
            )
        };

        let buf = buffer.lock().unwrap();
        let mut cursor = read_cursor.lock().unwrap();
        let start = (*cursor).min(buf.len());
        let mut rendered = String::from_utf8_lossy(&buf[start..]).to_string();
        *cursor = buf.len();
        if !rendered.is_empty() && truncated.load(Ordering::SeqCst) && *cursor == buf.len() {
            if !rendered.ends_with('\n') {
                rendered.push('\n');
            }
            rendered.push_str("[truncated]");
        }
        Ok(truncate_exec_output(rendered, max_output_tokens))
    }

    async fn wait_until_exit_or_timeout(
        &self,
        id: &str,
        timeout: Option<Duration>,
        progress: Option<&ProgressSender>,
        max_output_tokens: Option<usize>,
        behavior: WaitBehavior,
    ) -> Result<PollOutcome, String> {
        let state = self.process_state(id)?;
        let deadline = timeout.map(|value| tokio::time::Instant::now() + value);
        let mut sent_len = behavior.baseline_len;

        loop {
            if let Some(tx) = progress {
                let new_chunk = {
                    let buf = state.buffer.lock().unwrap();
                    if buf.len() > sent_len {
                        let chunk = String::from_utf8_lossy(&buf[sent_len..]).to_string();
                        sent_len = buf.len();
                        Some(chunk)
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
                let (output, original_token_count) =
                    self.take_incremental_output(id, max_output_tokens)?;
                let exit_code = state.exit_code.lock().unwrap().unwrap_or(-1);
                return Ok(PollOutcome::Exited {
                    output,
                    original_token_count,
                    exit_code,
                });
            }

            if let Some(dl) = deadline
                && tokio::time::Instant::now() >= dl
            {
                let exit_code = *state.exit_code.lock().unwrap();
                if let Some(exit_code) = exit_code {
                    wait_for_buffer_settle(&state, Duration::from_millis(OUTPUT_QUIET_PERIOD_MS))
                        .await;
                    let (output, original_token_count) =
                        self.take_incremental_output(id, max_output_tokens)?;
                    return Ok(PollOutcome::Exited {
                        output,
                        original_token_count,
                        exit_code,
                    });
                }
                let (output, original_token_count) =
                    self.take_incremental_output(id, max_output_tokens)?;
                return Ok(PollOutcome::Running {
                    output,
                    original_token_count,
                });
            }

            if let Some(wake_at) = deadline {
                tokio::select! {
                    _ = state.exit_notify.notified() => {}
                    _ = state.output_notify.notified() => {}
                    _ = tokio::time::sleep_until(wake_at) => {}
                }
            } else {
                tokio::select! {
                    _ = state.exit_notify.notified() => {}
                    _ = state.output_notify.notified() => {}
                }
            }
        }
    }

    fn remove_process(&self, id: &str) {
        self.processes.lock().unwrap().remove(id);
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
    },
    Exited {
        output: String,
        original_token_count: Option<usize>,
        exit_code: i32,
    },
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
            .unwrap_or(true);
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
            )
            .await
        {
            Ok(PollOutcome::Running {
                output,
                original_token_count,
                ..
            }) => ToolResult::ok(standard_shell_io_record(
                &handle_id,
                output,
                None,
                original_token_count,
                started.elapsed().as_secs_f64(),
            )),
            Ok(PollOutcome::Exited {
                output,
                original_token_count,
                exit_code,
            }) => {
                self.runtime.remove_process(&handle_id);
                ToolResult::ok(standard_shell_io_record(
                    &handle_id,
                    output,
                    Some(exit_code),
                    original_token_count,
                    started.elapsed().as_secs_f64(),
                ))
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
            )
            .await
        {
            Ok(PollOutcome::Running {
                output,
                original_token_count,
                ..
            }) => ToolResult::ok(standard_shell_io_record(
                &id,
                output,
                None,
                original_token_count,
                started.elapsed().as_secs_f64(),
            )),
            Ok(PollOutcome::Exited {
                output,
                original_token_count,
                exit_code,
            }) => {
                self.runtime.remove_process(&id);
                ToolResult::ok(standard_shell_io_record(
                    &id,
                    output,
                    Some(exit_code),
                    original_token_count,
                    started.elapsed().as_secs_f64(),
                ))
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
                description: "Run a command in a PTY. Completed commands return `output` and `exit_code`; longer-running commands return `session_id` so you can continue the same process with `write_stdin`.".into(),
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
                        description: "Whether to run the shell with -l/-i semantics. Defaults to true.".into(),
                        default_value: Some(serde_json::json!(true)),
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
                enabled: true,
                injected: true,
            },
            ToolDefinition {
                name: "write_stdin".into(),
                description: "Write bytes to a running command handle and wait briefly for the next settled output chunk. Use `close_stdin: true` to send EOF.".into(),
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
                enabled: true,
                injected: true,
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
        match name {
            "exec_command" => {
                let params = match self.parse_exec_command_params(args) {
                    Ok(params) => params,
                    Err(err) => return err,
                };
                self.exec_command(&params, progress).await
            }
            "write_stdin" => self.write_stdin_call(args, progress).await,
            _ => ToolResult::err_fmt(format_args!("Unknown tool: {name}")),
        }
    }
}

fn spawn_reader_thread(
    mut reader: Box<dyn Read + Send>,
    buffer: Arc<StdMutex<Vec<u8>>>,
    truncated: Arc<AtomicBool>,
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
                        if buf.len() < MAX_OUTPUT {
                            let remaining = MAX_OUTPUT - buf.len();
                            let to_copy = remaining.min(n);
                            buf.extend_from_slice(&chunk[..to_copy]);
                            if to_copy < n {
                                truncated.store(true, Ordering::SeqCst);
                            }
                        } else {
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

fn truncate_exec_output(
    output: String,
    max_output_tokens: Option<usize>,
) -> (String, Option<usize>) {
    let original_token_count = max_output_tokens.map(|_| estimate_token_count(&output));
    let Some(limit) = max_output_tokens else {
        return (output, original_token_count);
    };
    let max_chars = limit.saturating_mul(4);
    let char_count = output.chars().count();
    if char_count <= max_chars {
        return (output, original_token_count);
    }
    let truncated = output.chars().take(max_chars).collect::<String>() + "\n[truncated]";
    (truncated, original_token_count)
}

fn estimate_token_count(text: &str) -> usize {
    text.chars().count().div_ceil(4)
}

fn standard_shell_io_record(
    id: &str,
    output: String,
    exit_code: Option<i32>,
    original_token_count: Option<usize>,
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
        let shell = StandardShell::default();
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
        let open = shell
            .execute(
                "exec_command",
                &json!({"cmd": "read line; echo got:$line", "yield_time_ms": 10}),
            )
            .await;
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
        for _ in 0..16 {
            let open = shell
                .execute(
                    "exec_command",
                    &json!({"cmd": "read line; echo got:$line", "yield_time_ms": 10}),
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
            .execute("exec_command", &json!({"cmd": "cat", "yield_time_ms": 10}))
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

    #[test]
    fn shell_definitions_are_compact_and_non_empty() {
        let shell = StandardShell::default();
        let defs = shell.definitions();
        assert_eq!(defs.len(), 2);
        assert!(defs.iter().all(|def| !def.description.is_empty()));
    }
}
