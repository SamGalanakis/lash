use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{
    Arc, Mutex as StdMutex,
    atomic::{AtomicBool, AtomicI32, Ordering},
};
use std::thread;
use std::time::{Duration, Instant};

use portable_pty::{ChildKiller, CommandBuilder, MasterPty, PtySize, native_pty_system};
use serde_json::json;
use tokio::sync::Notify;

use crate::{
    ProgressSender, SandboxMessage, ToolDefinition, ToolParam, ToolPromptContext, ToolProvider,
    ToolResult,
};

use super::require_str;

struct ShellProcess {
    pid: u32,
    _master: Box<dyn MasterPty + Send>,
    writer: Arc<StdMutex<Option<Box<dyn Write + Send>>>>,
    killer: Arc<StdMutex<Box<dyn ChildKiller + Send + Sync>>>,
    buffer: Arc<StdMutex<Vec<u8>>>,
    truncated: Arc<AtomicBool>,
    read_cursor: Arc<StdMutex<usize>>,
    exit_code: Arc<StdMutex<Option<i32>>>,
    exit_notify: Arc<Notify>,
    output_notify: Arc<Notify>,
    default_timeout_ms: Option<u64>,
}

#[derive(Clone)]
struct ProcessState {
    buffer: Arc<StdMutex<Vec<u8>>>,
    exit_code: Arc<StdMutex<Option<i32>>>,
    exit_notify: Arc<Notify>,
    output_notify: Arc<Notify>,
    default_timeout_ms: Option<u64>,
}

const MAX_OUTPUT: usize = 512_000;
const DEFAULT_EXEC_YIELD_MS: u64 = 10_000;
const DEFAULT_WRITE_STDIN_YIELD_MS: u64 = 250;
const DEFAULT_SHELL_IO_TIMEOUT_MS: u64 = 250;
const OUTPUT_QUIET_PERIOD_MS: u64 = 75;
const DEFAULT_PTY_SIZE: PtySize = PtySize {
    rows: 24,
    cols: 80,
    pixel_width: 0,
    pixel_height: 0,
};

#[derive(Clone, Debug)]
struct ReplShellParams {
    command: String,
    workdir: PathBuf,
    timeout_ms: Option<u64>,
    login: bool,
}

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
    return_on_output: bool,
    baseline_len: usize,
    quiet_period: Option<Duration>,
    exit_grace_period: Option<Duration>,
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

    fn default_shell_name(&self) -> &str {
        Self::shell_name(&self.shell_path)
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
        timeout_ms: Option<u64>,
    ) -> Result<u32, String> {
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
        let pid = child.process_id().unwrap_or(0);
        let reader = pair
            .master
            .try_clone_reader()
            .map_err(|err| format!("Failed to clone PTY reader: {err}"))?;
        let writer = pair
            .master
            .take_writer()
            .map_err(|err| format!("Failed to take PTY writer: {err}"))?;
        let killer = child.clone_killer();
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
            pid,
            _master: pair.master,
            writer: Arc::new(StdMutex::new(Some(writer))),
            killer: Arc::new(StdMutex::new(killer)),
            buffer,
            truncated,
            read_cursor,
            exit_code,
            exit_notify,
            output_notify,
            default_timeout_ms: timeout_ms,
        };
        self.processes.lock().unwrap().insert(id, process);
        Ok(pid)
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
            default_timeout_ms: proc.default_timeout_ms,
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

    fn exit_code(&self, id: &str) -> Result<Option<i32>, String> {
        let state = self.process_state(id)?;
        Ok(*state.exit_code.lock().unwrap())
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
        let mut observed_len = behavior.baseline_len;
        let mut quiet_deadline = None;

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
            let buffer_len = state.buffer.lock().unwrap().len();
            if buffer_len > observed_len {
                observed_len = buffer_len;
                if behavior.return_on_output {
                    quiet_deadline = Some(
                        tokio::time::Instant::now() + behavior.quiet_period.unwrap_or_default(),
                    );
                }
            }

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

            if behavior.return_on_output
                && observed_len > behavior.baseline_len
                && quiet_deadline
                    .is_none_or(|quiet_until| tokio::time::Instant::now() >= quiet_until)
            {
                if let Some(grace_period) = behavior.exit_grace_period {
                    let _ = tokio::time::timeout(grace_period, state.exit_notify.notified()).await;
                }
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
                    deadline_reached: false,
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
                    deadline_reached: true,
                });
            }

            let wake_at = match (deadline, quiet_deadline) {
                (Some(deadline), Some(quiet_until)) => Some(deadline.min(quiet_until)),
                (Some(deadline), None) => Some(deadline),
                (None, Some(quiet_until)) => Some(quiet_until),
                (None, None) => None,
            };

            if let Some(wake_at) = wake_at {
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

    async fn kill_process(&self, id: &str) -> Result<(), String> {
        let (pid, killer, exit_notify, state) = {
            let procs = self.processes.lock().unwrap();
            let proc = procs
                .get(id)
                .ok_or_else(|| format!("No process with id: {id}"))?;
            (
                proc.pid,
                Arc::clone(&proc.killer),
                Arc::clone(&proc.exit_notify),
                ProcessState {
                    buffer: Arc::clone(&proc.buffer),
                    exit_code: Arc::clone(&proc.exit_code),
                    exit_notify: Arc::clone(&proc.exit_notify),
                    output_notify: Arc::clone(&proc.output_notify),
                    default_timeout_ms: proc.default_timeout_ms,
                },
            )
        };

        tokio::task::spawn_blocking({
            let killer = Arc::clone(&killer);
            move || {
                let mut killer = killer.lock().unwrap();
                request_termination(pid, killer.as_mut())
            }
        })
        .await
        .map_err(|err| format!("Kill task failed: {err}"))??;

        let exited = tokio::time::timeout(Duration::from_secs(2), exit_notify.notified())
            .await
            .is_ok();
        if !exited {
            tokio::task::spawn_blocking(move || {
                let mut killer = killer.lock().unwrap();
                force_termination(pid, killer.as_mut())
            })
            .await
            .map_err(|err| format!("Kill task failed: {err}"))??;
            let _ = tokio::time::timeout(Duration::from_secs(1), exit_notify.notified()).await;
        }

        wait_for_buffer_settle(&state, Duration::from_millis(OUTPUT_QUIET_PERIOD_MS)).await;
        self.processes.lock().unwrap().remove(id);
        Ok(())
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
        deadline_reached: bool,
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
            None,
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
                WaitBehavior {
                    return_on_output: false,
                    baseline_len: 0,
                    quiet_period: None,
                    exit_grace_period: None,
                },
            )
            .await
        {
            Ok(PollOutcome::Running {
                output,
                original_token_count,
                ..
            }) => ToolResult::ok(shell_io_record(
                &handle_id,
                output,
                true,
                None,
                original_token_count,
                Some(started.elapsed().as_secs_f64()),
                false,
            )),
            Ok(PollOutcome::Exited {
                output,
                original_token_count,
                exit_code,
            }) => {
                self.runtime.remove_process(&handle_id);
                ToolResult::ok(shell_io_record(
                    &handle_id,
                    output,
                    false,
                    Some(exit_code),
                    original_token_count,
                    Some(started.elapsed().as_secs_f64()),
                    false,
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
        let id = match require_str(args, "id") {
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
        let max_output_tokens = args
            .get("max_output_tokens")
            .and_then(|value| value.as_u64())
            .map(|value| value as usize);
        let started = Instant::now();
        let (baseline_len, _) = match self.runtime.output_state(id) {
            Ok(state) => state,
            Err(err) => return ToolResult::err(json!(err)),
        };

        if let Err(err) = self.runtime.write_stdin(id, chars).await {
            return ToolResult::err(json!(err));
        }

        match self
            .runtime
            .wait_until_exit_or_timeout(
                id,
                Some(Duration::from_millis(yield_time_ms)),
                progress,
                max_output_tokens,
                WaitBehavior {
                    return_on_output: true,
                    baseline_len,
                    quiet_period: Some(Duration::from_millis(OUTPUT_QUIET_PERIOD_MS)),
                    exit_grace_period: Some(Duration::from_millis(OUTPUT_QUIET_PERIOD_MS)),
                },
            )
            .await
        {
            Ok(PollOutcome::Running {
                output,
                original_token_count,
                ..
            }) => ToolResult::ok(shell_io_record(
                id,
                output,
                true,
                None,
                original_token_count,
                Some(started.elapsed().as_secs_f64()),
                false,
            )),
            Ok(PollOutcome::Exited {
                output,
                original_token_count,
                exit_code,
            }) => {
                self.runtime.remove_process(id);
                ToolResult::ok(shell_io_record(
                    id,
                    output,
                    false,
                    Some(exit_code),
                    original_token_count,
                    Some(started.elapsed().as_secs_f64()),
                    false,
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

pub struct ReplShell {
    runtime: ShellRuntime,
}

impl ReplShell {
    pub fn new() -> Self {
        Self {
            runtime: ShellRuntime::new(),
        }
    }

    pub fn with_cwd(mut self, cwd: impl Into<PathBuf>) -> Self {
        self.runtime = self.runtime.with_cwd(cwd);
        self
    }

    pub fn shell_name(&self) -> &str {
        self.runtime.default_shell_name()
    }

    fn parse_repl_params(&self, args: &serde_json::Value) -> Result<ReplShellParams, ToolResult> {
        let command = require_str(args, "command")?.to_string();
        let workdir = self.runtime.resolve_workdir(
            args.get("workdir")
                .and_then(|value| value.as_str())
                .filter(|value| !value.is_empty()),
        );
        let timeout_ms = parse_timeout_ms(args)?;
        let login = args
            .get("login")
            .and_then(|value| value.as_bool())
            .unwrap_or(false);
        Ok(ReplShellParams {
            command,
            workdir,
            timeout_ms,
            login,
        })
    }

    async fn spawn_handle(&self, params: &ReplShellParams) -> ToolResult {
        let id = uuid::Uuid::new_v4().to_string();
        match self.runtime.spawn_process(
            id.clone(),
            &params.command,
            &params.workdir,
            params.login,
            &self.runtime.shell_path,
            params.timeout_ms,
        ) {
            Ok(pid) => ToolResult::ok(shell_handle_record(
                &id,
                pid,
                &params.command,
                &params.workdir,
                params.timeout_ms,
                params.login,
            )),
            Err(err) => ToolResult::err(json!(err)),
        }
    }

    async fn shell_wait(
        &self,
        id: &str,
        timeout_ms: Option<u64>,
        max_output_tokens: Option<usize>,
        progress: Option<&ProgressSender>,
    ) -> ToolResult {
        let started = Instant::now();
        let state = match self.runtime.process_state(id) {
            Ok(state) => state,
            Err(err) => return ToolResult::err(json!(err)),
        };
        let timeout_ms = timeout_ms.or(state.default_timeout_ms);

        match self
            .runtime
            .wait_until_exit_or_timeout(
                id,
                timeout_ms.map(Duration::from_millis),
                progress,
                max_output_tokens,
                WaitBehavior {
                    return_on_output: false,
                    baseline_len: 0,
                    quiet_period: None,
                    exit_grace_period: None,
                },
            )
            .await
        {
            Ok(PollOutcome::Exited {
                output,
                original_token_count,
                exit_code,
            }) => {
                self.runtime.remove_process(id);
                ToolResult::ok(shell_io_record(
                    id,
                    output,
                    false,
                    Some(exit_code),
                    original_token_count,
                    Some(started.elapsed().as_secs_f64()),
                    false,
                ))
            }
            Ok(PollOutcome::Running {
                output,
                original_token_count,
                deadline_reached,
            }) => ToolResult::ok(shell_io_record(
                id,
                output,
                true,
                None,
                original_token_count,
                Some(started.elapsed().as_secs_f64()),
                deadline_reached && timeout_ms.is_some(),
            )),
            Err(err) => ToolResult::err(json!(err)),
        }
    }

    async fn shell_read(
        &self,
        id: &str,
        timeout_ms: Option<u64>,
        max_output_tokens: Option<usize>,
    ) -> ToolResult {
        let exit_code = match self.runtime.exit_code(id) {
            Ok(exit_code) => exit_code,
            Err(err) => return ToolResult::err_fmt(err),
        };
        if exit_code.is_some() {
            let output = match self.runtime.take_incremental_output(id, max_output_tokens) {
                Ok(output) => output,
                Err(err) => return ToolResult::err_fmt(err),
            };
            return ToolResult::ok(shell_io_record(
                id, output.0, false, exit_code, output.1, None, false,
            ));
        }

        let (buffer_len, read_cursor) = match self.runtime.output_state(id) {
            Ok(state) => state,
            Err(err) => return ToolResult::err_fmt(err),
        };
        if buffer_len > read_cursor {
            let output = match self.runtime.take_incremental_output(id, max_output_tokens) {
                Ok(output) => output,
                Err(err) => return ToolResult::err_fmt(err),
            };
            return ToolResult::ok(shell_io_record(
                id, output.0, true, None, output.1, None, false,
            ));
        }

        let timeout = timeout_ms.unwrap_or(0);
        match self
            .runtime
            .wait_until_exit_or_timeout(
                id,
                Some(Duration::from_millis(timeout)),
                None,
                max_output_tokens,
                WaitBehavior {
                    return_on_output: true,
                    baseline_len: read_cursor,
                    quiet_period: Some(Duration::from_millis(OUTPUT_QUIET_PERIOD_MS)),
                    exit_grace_period: None,
                },
            )
            .await
        {
            Ok(PollOutcome::Running {
                output,
                original_token_count,
                deadline_reached,
            }) => ToolResult::ok(shell_io_record(
                id,
                output,
                true,
                None,
                original_token_count,
                None,
                deadline_reached && timeout_ms.is_some(),
            )),
            Ok(PollOutcome::Exited {
                output,
                original_token_count,
                exit_code,
            }) => {
                self.runtime.remove_process(id);
                ToolResult::ok(shell_io_record(
                    id,
                    output,
                    false,
                    Some(exit_code),
                    original_token_count,
                    None,
                    false,
                ))
            }
            Err(err) => ToolResult::err_fmt(err),
        }
    }

    async fn shell_write(
        &self,
        id: &str,
        input: &str,
        timeout_ms: Option<u64>,
        max_output_tokens: Option<usize>,
    ) -> ToolResult {
        let started = Instant::now();
        let (baseline_len, _) = match self.runtime.output_state(id) {
            Ok(state) => state,
            Err(err) => return ToolResult::err(json!(err)),
        };
        if let Err(err) = self.runtime.write_stdin(id, input).await {
            return ToolResult::err(json!(err));
        }

        let timeout_ms = timeout_ms.unwrap_or(DEFAULT_SHELL_IO_TIMEOUT_MS);
        match self
            .runtime
            .wait_until_exit_or_timeout(
                id,
                Some(Duration::from_millis(timeout_ms)),
                None,
                max_output_tokens,
                WaitBehavior {
                    return_on_output: true,
                    baseline_len,
                    quiet_period: Some(Duration::from_millis(OUTPUT_QUIET_PERIOD_MS)),
                    exit_grace_period: None,
                },
            )
            .await
        {
            Ok(PollOutcome::Running {
                output,
                original_token_count,
                deadline_reached,
            }) => ToolResult::ok(shell_io_record(
                id,
                output,
                true,
                None,
                original_token_count,
                Some(started.elapsed().as_secs_f64()),
                deadline_reached,
            )),
            Ok(PollOutcome::Exited {
                output,
                original_token_count,
                exit_code,
            }) => {
                self.runtime.remove_process(id);
                ToolResult::ok(shell_io_record(
                    id,
                    output,
                    false,
                    Some(exit_code),
                    original_token_count,
                    Some(started.elapsed().as_secs_f64()),
                    false,
                ))
            }
            Err(err) => ToolResult::err(json!(err)),
        }
    }

    async fn shell_kill(&self, id: &str) -> ToolResult {
        match self.runtime.kill_process(id).await {
            Ok(()) => ToolResult::ok(json!({
                "id": id,
                "running": false,
                "killed": true,
            })),
            Err(err) => ToolResult::err(json!(err)),
        }
    }
}

impl Default for ReplShell {
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
                description: vec![crate::ToolText::new(
                    "Runs a command in a PTY and returns a shell result record. PTY output is terminal-style and stdout/stderr are merged. Long-running commands return `running: true` with a reusable handle `id`; completed commands return `running: false` with `exit_code` and incremental `output`.",
                    [crate::ExecutionMode::Standard],
                )],
                params: vec![
                    ToolParam::typed("cmd", "str"),
                    ToolParam::optional("workdir", "str"),
                    ToolParam::optional("shell", "str"),
                    ToolParam::optional("login", "bool"),
                    ToolParam::optional("yield_time_ms", "int"),
                    ToolParam::optional("max_output_tokens", "int"),
                ],
                returns: "dict".into(),
                examples: vec![],
                hidden: false,
                inject_into_prompt: true,
            },
            ToolDefinition {
                name: "write_stdin".into(),
                description: vec![crate::ToolText::new(
                    "Writes characters to an existing PTY shell handle and returns the next settled chunk of terminal output. Pass empty `chars` to poll for more output without sending input.",
                    [crate::ExecutionMode::Standard],
                )],
                params: vec![
                    ToolParam::typed("id", "str"),
                    ToolParam::optional("chars", "str"),
                    ToolParam::optional("yield_time_ms", "int"),
                    ToolParam::optional("max_output_tokens", "int"),
                ],
                returns: "dict".into(),
                examples: vec![],
                hidden: false,
                inject_into_prompt: true,
            },
        ]
    }

    fn prompt_guides(&self, _context: &ToolPromptContext) -> Vec<String> {
        Vec::new()
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

#[async_trait::async_trait]
impl ToolProvider for ReplShell {
    fn definitions(&self) -> Vec<ToolDefinition> {
        let shell_name = self.shell_name();
        vec![
            ToolDefinition {
                name: "shell".into(),
                description: vec![crate::ToolText::new(
                    format!(
                        "Start a PTY-backed command via {shell_name} and return a handle for interactive or long-running work. Companion tools are `shell_wait`, `shell_read`, `shell_write`, and `shell_kill`. Use this for terminal-sensitive programs like REPLs, prompts, and tools that change behavior when attached to a terminal."
                    ),
                    [crate::ExecutionMode::Repl],
                )],
                params: vec![
                    ToolParam::typed("command", "str"),
                    ToolParam::optional("workdir", "str"),
                    ToolParam::optional("timeout_ms", "int"),
                    ToolParam::optional("login", "bool"),
                ],
                returns: "dict".into(),
                examples: vec![],
                hidden: false,
                inject_into_prompt: true,
            },
            ToolDefinition {
                name: "shell_wait".into(),
                description: vec![crate::ToolText::new(
                    "Wait for a PTY shell handle to exit or until `timeout_ms` elapses. Returns the next drained output chunk; if the timeout elapses first, returns `running: true` with `timed_out: true` without killing the process.",
                    [crate::ExecutionMode::Repl],
                )],
                params: vec![
                    ToolParam::typed("id", "str"),
                    ToolParam::optional("timeout_ms", "int"),
                    ToolParam::optional("max_output_tokens", "int"),
                ],
                returns: "dict".into(),
                examples: vec![],
                hidden: false,
                inject_into_prompt: true,
            },
            ToolDefinition {
                name: "shell_read".into(),
                description: vec![crate::ToolText::new(
                    "Read and drain PTY output from a shell handle. Unread output returns immediately; otherwise, if `timeout_ms` is provided, wait up to that long for new output before returning.",
                    [crate::ExecutionMode::Repl],
                )],
                params: vec![
                    ToolParam::typed("id", "str"),
                    ToolParam::optional("timeout_ms", "int"),
                    ToolParam::optional("max_output_tokens", "int"),
                ],
                returns: "dict".into(),
                examples: vec![],
                hidden: false,
                inject_into_prompt: true,
            },
            ToolDefinition {
                name: "shell_write".into(),
                description: vec![crate::ToolText::new(
                    "Write input to a PTY shell handle's stdin and wait briefly for the next settled output chunk. Terminal echo and prompts may appear in the returned `output`.",
                    [crate::ExecutionMode::Repl],
                )],
                params: vec![
                    ToolParam::typed("id", "str"),
                    ToolParam::typed("input", "str"),
                    ToolParam::optional("timeout_ms", "int"),
                    ToolParam::optional("max_output_tokens", "int"),
                ],
                returns: "dict".into(),
                examples: vec![],
                hidden: false,
                inject_into_prompt: true,
            },
            ToolDefinition {
                name: "shell_kill".into(),
                description: vec![crate::ToolText::new(
                    "Terminate a running PTY shell handle.",
                    [crate::ExecutionMode::Repl],
                )],
                params: vec![ToolParam::typed("id", "str")],
                returns: "dict".into(),
                examples: vec![],
                hidden: false,
                inject_into_prompt: true,
            },
        ]
    }

    fn prompt_guides(&self, _context: &ToolPromptContext) -> Vec<String> {
        vec!["### REPL Shell Shapes\n`call shell { ... }` starts a PTY-backed terminal session and returns a handle record in `result.value` with an `id` field. `shell_wait`, `shell_read`, and `shell_write` all return a dict with `id`, `running`, `output`, and `exit_code`. Output is incremental and draining, `shell_read` returns unread output immediately, and `shell_write` waits briefly for output to settle before returning. Stdout/stderr are merged by terminal semantics. Example:\n`proc = call shell { command: \"python3 -q\" }`\n`prompt = call shell_read { id: proc.value.id, timeout_ms: 1000 }`\n`reply = call shell_write { id: proc.value.id, input: \"print(2 + 2)\\n\", timeout_ms: 1000 }`\n`finish reply.value.output`".to_string()]
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
            "shell" => {
                let params = match self.parse_repl_params(args) {
                    Ok(params) => params,
                    Err(err) => return err,
                };
                self.spawn_handle(&params).await
            }
            "shell_wait" => {
                let id = match require_str(args, "id") {
                    Ok(value) => value,
                    Err(err) => return err,
                };
                let timeout_ms = match parse_timeout_ms(args) {
                    Ok(timeout) => timeout,
                    Err(err) => return err,
                };
                let max_output_tokens = args
                    .get("max_output_tokens")
                    .and_then(|value| value.as_u64())
                    .map(|value| value as usize);
                self.shell_wait(id, timeout_ms, max_output_tokens, progress)
                    .await
            }
            "shell_read" => {
                let id = match require_str(args, "id") {
                    Ok(value) => value,
                    Err(err) => return err,
                };
                let timeout_ms = match parse_timeout_ms(args) {
                    Ok(timeout) => timeout,
                    Err(err) => return err,
                };
                let max_output_tokens = args
                    .get("max_output_tokens")
                    .and_then(|value| value.as_u64())
                    .map(|value| value as usize);
                self.shell_read(id, timeout_ms, max_output_tokens).await
            }
            "shell_write" => {
                let id = match require_str(args, "id") {
                    Ok(value) => value,
                    Err(err) => return err,
                };
                let input = match require_str(args, "input") {
                    Ok(value) => value,
                    Err(err) => return err,
                };
                let timeout_ms = match parse_timeout_ms(args) {
                    Ok(timeout) => timeout,
                    Err(err) => return err,
                };
                let max_output_tokens = args
                    .get("max_output_tokens")
                    .and_then(|value| value.as_u64())
                    .map(|value| value as usize);
                self.shell_write(id, input, timeout_ms, max_output_tokens)
                    .await
            }
            "shell_kill" => {
                let id = match require_str(args, "id") {
                    Ok(value) => value,
                    Err(err) => return err,
                };
                self.shell_kill(id).await
            }
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

#[cfg(unix)]
fn send_signal_to_pid(pid: u32, signal: i32) -> std::io::Result<()> {
    if pid == 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "missing process id",
        ));
    }

    if unsafe { libc::kill(pid as i32, signal) } == 0 {
        return Ok(());
    }

    Err(std::io::Error::last_os_error())
}

#[cfg(unix)]
fn send_signal_to_process_group(pid: u32, signal: i32) -> std::io::Result<()> {
    if pid == 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "missing process id",
        ));
    }

    if unsafe { libc::kill(-(pid as i32), signal) } == 0 {
        return Ok(());
    }

    Err(std::io::Error::last_os_error())
}

fn request_termination(pid: u32, killer: &mut dyn ChildKiller) -> Result<(), String> {
    #[cfg(unix)]
    if send_signal_to_process_group(pid, libc::SIGHUP).is_ok()
        || send_signal_to_pid(pid, libc::SIGHUP).is_ok()
    {
        return Ok(());
    }

    killer.kill().map_err(|err| format!("Kill failed: {err}"))
}

fn force_termination(pid: u32, killer: &mut dyn ChildKiller) -> Result<(), String> {
    #[cfg(unix)]
    if send_signal_to_process_group(pid, libc::SIGKILL).is_ok()
        || send_signal_to_pid(pid, libc::SIGKILL).is_ok()
    {
        return Ok(());
    }

    killer.kill().map_err(|err| format!("Kill failed: {err}"))
}

fn shell_handle_record(
    id: &str,
    pid: u32,
    command: &str,
    workdir: &Path,
    timeout_ms: Option<u64>,
    login: bool,
) -> serde_json::Value {
    json!({
        "__handle__": "shell",
        "id": id,
        "pid": pid,
        "command": command,
        "workdir": workdir.display().to_string(),
        "timeout_ms": timeout_ms,
        "login": login,
        "running": true,
    })
}

fn shell_io_record(
    id: &str,
    output: String,
    running: bool,
    exit_code: Option<i32>,
    original_token_count: Option<usize>,
    wall_time_seconds: Option<f64>,
    timed_out: bool,
) -> serde_json::Value {
    json!({
        "id": id,
        "running": running,
        "exit_code": exit_code,
        "output": output,
        "original_token_count": original_token_count,
        "wall_time_seconds": wall_time_seconds,
        "timed_out": timed_out,
    })
}

fn parse_timeout_ms(args: &serde_json::Value) -> Result<Option<u64>, ToolResult> {
    if let Some(value) = args.get("timeout_ms") {
        if let Some(timeout) = value.as_u64() {
            return Ok(Some(timeout));
        }
        if let Some(timeout) = value.as_f64()
            && timeout >= 0.0
        {
            return Ok(Some(timeout.round() as u64));
        }
        return Err(ToolResult::err_fmt(format_args!(
            "Invalid timeout_ms: expected non-negative number"
        )));
    }

    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn buffered_output(state: &ProcessState) -> String {
        let buffer = state.buffer.lock().unwrap();
        String::from_utf8_lossy(&buffer).into_owned()
    }

    #[tokio::test]
    async fn repl_shell_spawn_and_wait() {
        let shell = ReplShell::default();
        let result = shell
            .execute("shell", &json!({"command": "echo hello"}))
            .await;
        assert!(result.success);
        let handle = result.result;
        let id = handle["id"].as_str().unwrap();

        let result = shell
            .execute("shell_wait", &json!({"id": id, "timeout_ms": 5000}))
            .await;
        assert!(result.success);
        assert_eq!(result.result["running"], false);
        assert_eq!(result.result["exit_code"], 0);
        assert!(result.result["output"].as_str().unwrap().contains("hello"));
    }

    #[tokio::test]
    async fn repl_shell_reads_python_prompt_from_pty() {
        let shell = ReplShell::default();
        let open = shell
            .execute("shell", &json!({"command": "python3 -q"}))
            .await;
        assert!(open.success);
        let id = open.result["id"].as_str().unwrap();

        let prompt = shell
            .execute("shell_read", &json!({"id": id, "timeout_ms": 1000}))
            .await;
        assert!(prompt.success);
        assert!(prompt.result["running"].as_bool().unwrap());
        assert!(prompt.result["output"].as_str().unwrap().contains(">>>"));

        let reply = shell
            .execute(
                "shell_write",
                &json!({"id": id, "input": "print(2 + 2)\n", "timeout_ms": 1000}),
            )
            .await;
        assert!(reply.success);
        assert!(reply.result["output"].as_str().unwrap().contains("4"));

        let killed = shell.execute("shell_kill", &json!({"id": id})).await;
        assert!(killed.success);
    }

    #[tokio::test]
    async fn exec_command_returns_exit_code_when_command_finishes() {
        let shell = StandardShell::default();
        let result = shell
            .execute("exec_command", &json!({"cmd": "echo hello"}))
            .await;
        assert!(result.success);
        assert!(result.result["id"].as_str().is_some());
        assert_eq!(result.result["running"], false);
        assert_eq!(result.result["exit_code"], 0);
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
        assert!(result.result["id"].as_str().is_some());
        assert_eq!(result.result["running"], true);
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
        let id = open.result["id"].as_str().unwrap();

        let result = shell
            .execute(
                "write_stdin",
                &json!({"id": id, "chars": "hello\n", "yield_time_ms": 1000}),
            )
            .await;
        assert!(result.success);
        assert_eq!(result.result["running"], false);
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
            let id = open.result["id"].as_str().unwrap();

            let result = shell
                .execute(
                    "write_stdin",
                    &json!({"id": id, "chars": "hello\n", "yield_time_ms": 1000}),
                )
                .await;
            assert!(result.success);
            assert_eq!(
                result.result["running"], false,
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
    async fn repl_shell_wait_timeout_keeps_process_running() {
        let shell = ReplShell::default();
        let open = shell
            .execute("shell", &json!({"command": "sleep 0.2; echo done"}))
            .await;
        let id = open.result["id"].as_str().unwrap();

        let waiting = shell
            .execute("shell_wait", &json!({"id": id, "timeout_ms": 10}))
            .await;
        assert!(waiting.success);
        assert_eq!(waiting.result["running"], true);
        assert_eq!(waiting.result["timed_out"], true);

        let final_result = shell
            .execute("shell_wait", &json!({"id": id, "timeout_ms": 1000}))
            .await;
        assert!(final_result.success);
        assert_eq!(final_result.result["running"], false);
        assert_eq!(final_result.result["exit_code"], 0);
        assert!(
            final_result.result["output"]
                .as_str()
                .unwrap()
                .contains("done")
        );
    }

    #[tokio::test]
    async fn repl_shell_kill_preserves_trapped_hup_output() {
        let shell = ReplShell::default();
        let open = shell
            .execute(
                "shell",
                &json!({
                    "command": "trap 'echo caught_hup; exit 0' HUP; echo ready; sleep 30"
                }),
            )
            .await;
        assert!(open.success);
        let id = open.result["id"].as_str().unwrap().to_string();

        let prompt = shell
            .execute("shell_read", &json!({"id": id, "timeout_ms": 1000}))
            .await;
        assert!(prompt.success);
        assert!(prompt.result["running"].as_bool().unwrap());
        assert!(prompt.result["output"].as_str().unwrap().contains("ready"));

        let state = shell.runtime.process_state(&id).unwrap();

        let killed = shell.execute("shell_kill", &json!({"id": id})).await;
        assert!(killed.success);
        assert_eq!(killed.result["killed"], true);

        let exit_code = *state.exit_code.lock().unwrap();
        assert_eq!(exit_code, Some(0));

        let output = buffered_output(&state);
        assert!(
            output.contains("caught_hup"),
            "expected shell_kill to preserve trapped HUP output, got {output:?}"
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
    async fn repl_shell_rejects_missing_command() {
        let shell = ReplShell::default();
        let result = shell.execute("shell", &json!({})).await;
        assert!(!result.success);
    }

    #[test]
    fn standard_shell_definitions_match_standard_mode_only() {
        let shell = StandardShell::default();
        let defs = shell.definitions();
        assert_eq!(defs.len(), 2);
        assert!(defs.iter().all(|def| {
            !def.description_for(crate::ExecutionMode::Standard)
                .is_empty()
                && def.description_for(crate::ExecutionMode::Repl).is_empty()
        }));
    }
}
