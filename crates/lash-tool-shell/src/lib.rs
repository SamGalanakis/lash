use std::collections::{BTreeMap, HashMap};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::{
    Arc, Mutex as StdMutex,
    atomic::{AtomicBool, AtomicI32, Ordering},
};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use portable_pty::{ChildKiller, CommandBuilder, MasterPty, PtySize, native_pty_system};
use serde_json::json;
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncReadExt};
use tokio::process::Command as TokioCommand;
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;

use lash_core::plugin::{
    PluginError, PluginFactory, PluginSessionContext, PluginSpec, PluginSpecFactory, SessionPlugin,
};
use lash_core::{
    ProgressSender, PromptContribution, SandboxMessage, SessionToolAccess, ToolCall, ToolContract,
    ToolDefinition, ToolExecutionMode, ToolFailure, ToolFailureClass, ToolManifest,
    ToolProcessStartMode, ToolProvider, ToolResult,
};
use lash_plugin_monitor::{
    MAX_MONITOR_TIMEOUT_MS, MONITOR_LINE_EVENT, MonitorSpec, MonitorWakePolicy,
    monitor_process_descriptor, monitor_process_registration,
};

use lash_tool_support::{object_schema, require_str};

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
    shell_prompt_contributions_for_access(&SessionToolAccess::default())
}

/// Returns the shell prompt contributions, gating the `write_stdin`
/// reference on whether that tool is actually callable in the current
/// session.
pub fn shell_prompt_contributions_for_access(
    access: &SessionToolAccess,
) -> Vec<PromptContribution> {
    let mut command_execution = String::from(
        "Use `exec_command` for one-shot commands; it returns only after the process exits and successful results include `status: \"completed\"`, `done: true`, and `exit_code`. Use `start_command` only for interactive or intentionally long-lived processes; it may return `status: \"running\"`, `done: false`, and `session_id`, which means the output is partial and must not be treated as completion.",
    );
    if tool_callable_from_authority(access, "write_stdin") {
        command_execution.push_str(" Continue running sessions with `write_stdin`.");
    }
    command_execution.push_str(
        " For builds, installs, tests, migrations, service setup, and verification commands, use `exec_command` and wait for completion before concluding.",
    );
    vec![
        PromptContribution::guidance("Command Execution", command_execution),
        PromptContribution::guidance(
            "Git Safety",
            "Avoid destructive git commands unless explicitly requested.",
        ),
    ]
}

fn tool_callable_from_authority(access: &SessionToolAccess, name: &str) -> bool {
    if access.hides(name) {
        return false;
    }
    access.tools.is_empty() || access.tools.iter().any(|tool| tool.name == name)
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

struct PipeExecProcessRequest<'a> {
    id: &'a str,
    command: &'a str,
    workdir: &'a Path,
    login: bool,
    shell_path: &'a str,
    timeout: Option<Duration>,
    progress: Option<&'a ProgressSender>,
    max_output_tokens: Option<usize>,
    cancel: Option<CancellationToken>,
}

const MAX_OUTPUT: usize = 512_000;
const SPILL_OUTPUT_THRESHOLD: usize = 50 * 1024;
const DEFAULT_EXEC_COMMAND_TIMEOUT_MS: u64 = 10 * 60 * 1000;
const DEFAULT_START_COMMAND_POLL_MS: u64 = 250;
const DEFAULT_WRITE_STDIN_POLL_MS: u64 = 250;
const OUTPUT_QUIET_PERIOD_MS: u64 = 75;
const DEFAULT_PTY_SIZE: PtySize = PtySize {
    rows: 24,
    cols: 80,
    pixel_width: 0,
    pixel_height: 0,
};

#[derive(Clone, Debug)]
struct CommonCommandParams {
    cmd: String,
    workdir: PathBuf,
    shell_path: String,
    login: bool,
    allow_nonzero_exit: bool,
    max_output_tokens: Option<usize>,
}

#[derive(Clone, Debug)]
struct ExecCommandParams {
    cmd: String,
    workdir: PathBuf,
    shell_path: String,
    login: bool,
    allow_nonzero_exit: bool,
    timeout_ms: u64,
    max_output_tokens: Option<usize>,
}

#[derive(Clone, Debug)]
struct StartCommandParams {
    cmd: String,
    workdir: PathBuf,
    shell_path: String,
    login: bool,
    allow_nonzero_exit: bool,
    poll_ms: u64,
    max_output_tokens: Option<usize>,
}

#[derive(Clone, Debug)]
struct MonitorCommandParams {
    id: String,
    command: String,
    description: String,
    workdir: PathBuf,
    cwd: Option<String>,
    env: BTreeMap<String, String>,
    persistent: bool,
    timeout_ms: u64,
    wake_policy: MonitorWakePolicy,
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

    fn command_for_spawn(&self, command: &str, shell_path: &str, pty: bool) -> String {
        let shell_name = Self::shell_name(shell_path);
        let echo_off = if pty {
            // Disable terminal echo so bytes delivered via `write_stdin`
            // don't appear in the captured output stream. The PTY allocates
            // with `ECHO` on by default (matching interactive terminals),
            // but agents drive these sessions programmatically and the echo
            // is pure noise. `stty -echo || true` keeps the prefix
            // harmless on environments where `stty` isn't available.
            "stty -echo 2>/dev/null || true\n"
        } else {
            ""
        };
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
        pty: bool,
    ) -> Result<Vec<String>, String> {
        let command = self.command_for_spawn(command, shell_path, pty);
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
        for arg in self.shell_args(command, login, shell_path, true)? {
            cmd.arg(arg);
        }
        cmd.cwd(workdir.as_os_str());

        let child = pair.slave.spawn_command(cmd).map_err(|err| {
            format!(
                "Failed to spawn PTY command with shell `{}` in `{}`: {err}",
                shell_path,
                workdir.display()
            )
        })?;
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

    async fn exec_pipe_process(
        &self,
        request: PipeExecProcessRequest<'_>,
    ) -> Result<PollOutcome, String> {
        let PipeExecProcessRequest {
            id,
            command,
            workdir,
            login,
            shell_path,
            timeout,
            progress,
            max_output_tokens,
            cancel,
        } = request;
        let mut cmd = TokioCommand::new(shell_path);
        for arg in self.shell_args(command, login, shell_path, false)? {
            cmd.arg(arg);
        }
        cmd.current_dir(workdir)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        #[cfg(unix)]
        unsafe {
            cmd.pre_exec(|| {
                if libc::setsid() == -1 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }

        let mut child = cmd.spawn().map_err(|err| {
            format!(
                "Failed to spawn command with shell `{}` in `{}`: {err}",
                shell_path,
                workdir.display()
            )
        })?;
        let child_pid = child.id();
        let stdout = child.stdout.take();
        let stderr = child.stderr.take();

        let buffer = Arc::new(StdMutex::new(Vec::new()));
        let buffer_start = Arc::new(StdMutex::new(0usize));
        let truncated = Arc::new(AtomicBool::new(false));
        let spill = Arc::new(StdMutex::new(None));
        let output_notify = Arc::new(Notify::new());

        if let Some(stdout) = stdout {
            spawn_async_reader(
                id.to_string(),
                stdout,
                Arc::clone(&buffer),
                Arc::clone(&buffer_start),
                Arc::clone(&truncated),
                Arc::clone(&spill),
                Arc::clone(&output_notify),
            );
        }
        if let Some(stderr) = stderr {
            spawn_async_reader(
                id.to_string(),
                stderr,
                Arc::clone(&buffer),
                Arc::clone(&buffer_start),
                Arc::clone(&truncated),
                Arc::clone(&spill),
                Arc::clone(&output_notify),
            );
        }

        let deadline = timeout.map(|value| tokio::time::Instant::now() + value);
        let wait_handle = tokio::spawn(async move { child.wait().await });
        tokio::pin!(wait_handle);
        let mut sent_len = 0usize;

        loop {
            if let Some(token) = cancel.as_ref()
                && token.is_cancelled()
            {
                terminate_pipe_process(child_pid);
                let _ = tokio::time::timeout(Duration::from_millis(500), &mut wait_handle).await;
                return Ok(PollOutcome::Cancelled);
            }

            if let Some(tx) = progress
                && let Some(chunk) = progress_chunk(&buffer, &buffer_start, &mut sent_len)
                && !chunk.is_empty()
            {
                let _ = tx.send(SandboxMessage {
                    text: chunk,
                    kind: "tool_output".into(),
                });
            }

            if let Some(dl) = deadline
                && tokio::time::Instant::now() >= dl
            {
                terminate_pipe_process(child_pid);
                let _ = tokio::time::timeout(Duration::from_millis(500), &mut wait_handle).await;
                wait_for_pipe_buffer_settle(
                    &buffer,
                    &output_notify,
                    Duration::from_millis(OUTPUT_QUIET_PERIOD_MS),
                )
                .await;
                let (output, original_token_count, full_output_path) = render_buffer_output(
                    id,
                    &buffer,
                    &buffer_start,
                    Arc::as_ref(&truncated),
                    &spill,
                    max_output_tokens,
                );
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
                    status = &mut wait_handle => {
                        let exit_code = status
                            .map_err(|err| format!("Wait task failed: {err}"))?
                            .map(exit_status_code)
                            .unwrap_or(-1);
                        wait_for_pipe_buffer_settle(&buffer, &output_notify, Duration::from_millis(OUTPUT_QUIET_PERIOD_MS)).await;
                        let (output, original_token_count, full_output_path) = render_buffer_output(
                            id,
                            &buffer,
                            &buffer_start,
                            Arc::as_ref(&truncated),
                            &spill,
                            max_output_tokens,
                        );
                        return Ok(PollOutcome::Exited {
                            output,
                            original_token_count,
                            exit_code,
                            full_output_path,
                        });
                    }
                    _ = output_notify.notified() => {}
                    _ = tokio::time::sleep_until(wake_at) => {}
                    _ = cancel_future => {}
                }
            } else {
                tokio::select! {
                    status = &mut wait_handle => {
                        let exit_code = status
                            .map_err(|err| format!("Wait task failed: {err}"))?
                            .map(exit_status_code)
                            .unwrap_or(-1);
                        wait_for_pipe_buffer_settle(&buffer, &output_notify, Duration::from_millis(OUTPUT_QUIET_PERIOD_MS)).await;
                        let (output, original_token_count, full_output_path) = render_buffer_output(
                            id,
                            &buffer,
                            &buffer_start,
                            Arc::as_ref(&truncated),
                            &spill,
                            max_output_tokens,
                        );
                        return Ok(PollOutcome::Exited {
                            output,
                            original_token_count,
                            exit_code,
                            full_output_path,
                        });
                    }
                    _ = output_notify.notified() => {}
                    _ = cancel_future => {}
                }
            }
        }
    }
}

fn kill_child(state: &ProcessState) {
    if let Some(mut killer) = state.killer.lock().unwrap().take() {
        let _ = killer.kill();
    }
}

#[cfg(unix)]
fn terminate_pipe_process(pid: Option<u32>) {
    let Some(pid) = pid else {
        return;
    };
    let pgid = -(pid as i32);
    unsafe {
        if libc::kill(pgid, libc::SIGKILL) == -1 {
            let _ = libc::kill(pid as i32, libc::SIGKILL);
        }
    }
}

#[cfg(not(unix))]
fn terminate_pipe_process(_pid: Option<u32>) {}

fn exit_status_code(status: std::process::ExitStatus) -> i32 {
    status.code().unwrap_or(-1)
}

fn progress_chunk(
    buffer: &Arc<StdMutex<Vec<u8>>>,
    buffer_start: &Arc<StdMutex<usize>>,
    sent_len: &mut usize,
) -> Option<String> {
    let buf = buffer.lock().unwrap();
    let start_offset = *buffer_start.lock().unwrap();
    let buffer_end = start_offset + buf.len();
    if buffer_end <= *sent_len {
        return None;
    }
    let start = (*sent_len).max(start_offset);
    let mut chunk = String::from_utf8_lossy(&buf[start.saturating_sub(start_offset)..]).to_string();
    if *sent_len < start_offset && !chunk.is_empty() {
        if !chunk.ends_with('\n') {
            chunk.push('\n');
        }
        chunk.push_str("[truncated]");
    }
    *sent_len = buffer_end;
    Some(clean_terminal_output(&chunk))
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

async fn wait_for_pipe_buffer_settle(
    buffer: &Arc<StdMutex<Vec<u8>>>,
    output_notify: &Arc<Notify>,
    quiet_period: Duration,
) {
    let mut last_len = buffer.lock().unwrap().len();
    let mut quiet_until = tokio::time::Instant::now() + quiet_period;

    loop {
        tokio::select! {
            _ = output_notify.notified() => {
                let buffer_len = buffer.lock().unwrap().len();
                if buffer_len != last_len {
                    last_len = buffer_len;
                    quiet_until = tokio::time::Instant::now() + quiet_period;
                }
            }
            _ = tokio::time::sleep_until(quiet_until) => break,
        }
    }
}

fn render_buffer_output(
    id: &str,
    buffer: &Arc<StdMutex<Vec<u8>>>,
    buffer_start: &Arc<StdMutex<usize>>,
    truncated: &AtomicBool,
    spill: &Arc<StdMutex<Option<ShellOutputSpill>>>,
    max_output_tokens: Option<usize>,
) -> (String, Option<usize>, Option<PathBuf>) {
    let buf = buffer.lock().unwrap();
    let start_offset = *buffer_start.lock().unwrap();
    let mut rendered = String::from_utf8_lossy(&buf).to_string();
    if truncated.load(Ordering::SeqCst) || start_offset > 0 {
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
    if let Some(spill) = spill_guard.as_mut() {
        let _ = spill.file.flush();
    }
    (rendered, original_token_count, full_output_path)
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

    fn parse_common_command_params(
        &self,
        args: &serde_json::Value,
    ) -> Result<CommonCommandParams, ToolResult> {
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
        let allow_nonzero_exit = args
            .get("allow_nonzero_exit")
            .and_then(|value| value.as_bool())
            .unwrap_or(false);
        let max_output_tokens = args
            .get("max_output_tokens")
            .and_then(|value| value.as_u64())
            .map(|value| value as usize);

        Ok(CommonCommandParams {
            cmd,
            workdir,
            shell_path,
            login,
            allow_nonzero_exit,
            max_output_tokens,
        })
    }

    fn parse_exec_command_params(
        &self,
        args: &serde_json::Value,
    ) -> Result<ExecCommandParams, ToolResult> {
        let common = self.parse_common_command_params(args)?;
        let timeout_ms = args
            .get("timeout_ms")
            .and_then(|value| value.as_u64())
            .filter(|value| *value > 0)
            .unwrap_or(DEFAULT_EXEC_COMMAND_TIMEOUT_MS);

        Ok(ExecCommandParams {
            cmd: common.cmd,
            workdir: common.workdir,
            shell_path: common.shell_path,
            login: common.login,
            allow_nonzero_exit: common.allow_nonzero_exit,
            timeout_ms,
            max_output_tokens: common.max_output_tokens,
        })
    }

    fn parse_start_command_params(
        &self,
        args: &serde_json::Value,
    ) -> Result<StartCommandParams, ToolResult> {
        let common = self.parse_common_command_params(args)?;
        let poll_ms = args
            .get("poll_ms")
            .and_then(|value| value.as_u64())
            .unwrap_or(DEFAULT_START_COMMAND_POLL_MS);

        Ok(StartCommandParams {
            cmd: common.cmd,
            workdir: common.workdir,
            shell_path: common.shell_path,
            login: common.login,
            allow_nonzero_exit: common.allow_nonzero_exit,
            poll_ms,
            max_output_tokens: common.max_output_tokens,
        })
    }

    fn parse_monitor_command_params(
        &self,
        args: &serde_json::Value,
    ) -> Result<MonitorCommandParams, ToolResult> {
        let id = args
            .get("id")
            .and_then(|value| value.as_str())
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| self.runtime.allocate_handle_id());
        let command = args
            .get("command")
            .and_then(|value| value.as_str())
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| ToolResult::err_fmt("monitor requires `command`"))?
            .to_string();
        let description = args
            .get("description")
            .and_then(|value| value.as_str())
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .unwrap_or(&command)
            .to_string();
        let cwd = args
            .get("cwd")
            .and_then(|value| value.as_str())
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned);
        let workdir = self.runtime.resolve_workdir(cwd.as_deref());
        let env = match args.get("env") {
            Some(value) => serde_json::from_value::<BTreeMap<String, String>>(value.clone())
                .map_err(|err| ToolResult::err_fmt(format_args!("invalid monitor env: {err}")))?,
            None => BTreeMap::new(),
        };
        let persistent = args
            .get("persistent")
            .and_then(|value| value.as_bool())
            .unwrap_or(false);
        let timeout_ms = args
            .get("timeout_ms")
            .and_then(|value| value.as_u64())
            .unwrap_or(300_000);
        if timeout_ms > MAX_MONITOR_TIMEOUT_MS {
            return Err(ToolResult::err_fmt(format_args!(
                "monitor timeout_ms must be <= {MAX_MONITOR_TIMEOUT_MS}"
            )));
        }
        let wake_policy = match args.get("wake_policy").and_then(|value| value.as_str()) {
            Some("notify") => MonitorWakePolicy::Notify,
            Some("queue_turn") | None => MonitorWakePolicy::QueueTurn,
            Some(other) => {
                return Err(ToolResult::err_fmt(format_args!(
                    "invalid monitor wake_policy `{other}`"
                )));
            }
        };

        Ok(MonitorCommandParams {
            id,
            command,
            description,
            workdir,
            cwd,
            env,
            persistent,
            timeout_ms,
            wake_policy,
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

        match self
            .runtime
            .exec_pipe_process(PipeExecProcessRequest {
                id: &handle_id,
                command: &params.cmd,
                workdir: &params.workdir,
                login: params.login,
                shell_path: &params.shell_path,
                timeout: Some(Duration::from_millis(params.timeout_ms)),
                progress,
                max_output_tokens: params.max_output_tokens,
                cancel,
            })
            .await
        {
            Ok(PollOutcome::Running {
                output,
                original_token_count,
                full_output_path,
                ..
            }) => timed_out_shell_io_result(
                &handle_id,
                output,
                original_token_count,
                full_output_path.as_deref(),
                started.elapsed().as_secs_f64(),
                params.timeout_ms,
                params.allow_nonzero_exit,
            ),
            Ok(PollOutcome::Exited {
                output,
                original_token_count,
                exit_code,
                full_output_path,
            }) => shell_io_result(
                &handle_id,
                output,
                Some(exit_code),
                original_token_count,
                full_output_path.as_deref(),
                started.elapsed().as_secs_f64(),
                params.allow_nonzero_exit,
            ),
            Ok(PollOutcome::Cancelled) => ToolResult::cancelled("tool call cancelled"),
            Err(err) => ToolResult::err(json!(err)),
        }
    }

    async fn start_command(
        &self,
        params: &StartCommandParams,
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
                Some(Duration::from_millis(params.poll_ms)),
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
                shell_io_result(
                    &handle_id,
                    output,
                    Some(exit_code),
                    original_token_count,
                    full_output_path.as_deref(),
                    started.elapsed().as_secs_f64(),
                    params.allow_nonzero_exit,
                )
            }
            Ok(PollOutcome::Cancelled) => {
                self.runtime.remove_process(&handle_id);
                ToolResult::cancelled("tool call cancelled")
            }
            Err(err) => {
                self.runtime.remove_process(&handle_id);
                ToolResult::err(json!(err))
            }
        }
    }

    async fn start_monitor_process(
        &self,
        params: &MonitorCommandParams,
        context: &lash_core::ToolContext<'_>,
    ) -> ToolResult {
        let spec = MonitorSpec {
            id: params.id.clone(),
            command: params.command.clone(),
            cwd: params.cwd.clone(),
            env: params.env.clone(),
            persistent: params.persistent,
            timeout_ms: params.timeout_ms,
            wake_policy: params.wake_policy,
            ..MonitorSpec::default()
        };
        let registration = monitor_process_registration(&spec, Some(&params.description));
        let descriptor = monitor_process_descriptor(&spec);
        match context
            .processes()
            .start_process(registration, Some(descriptor))
            .await
        {
            Ok(record) => ToolResult::ok(json!({
                "__handle__": "process",
                "id": record.id,
            })),
            Err(err) => ToolResult::err_fmt(err.to_string()),
        }
    }

    async fn run_monitor_process(
        &self,
        params: &MonitorCommandParams,
        context: &lash_core::ToolContext<'_>,
        cancel: Option<CancellationToken>,
    ) -> ToolResult {
        let mut cmd = TokioCommand::new(&self.runtime.shell_path);
        let args =
            match self
                .runtime
                .shell_args(&params.command, false, &self.runtime.shell_path, false)
            {
                Ok(args) => args,
                Err(err) => return ToolResult::err_fmt(err),
            };
        for arg in args {
            cmd.arg(arg);
        }
        cmd.current_dir(&params.workdir)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        if !params.env.is_empty() {
            cmd.envs(params.env.iter());
        }

        #[cfg(unix)]
        unsafe {
            cmd.pre_exec(|| {
                if libc::setsid() == -1 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }

        let mut child = match cmd.spawn() {
            Ok(child) => child,
            Err(err) => {
                return monitor_failure(
                    "monitor_start_failed",
                    format!(
                        "failed to spawn monitor with shell `{}` in `{}`: {err}",
                        self.runtime.shell_path,
                        params.workdir.display()
                    ),
                );
            }
        };
        let child_pid = child.id();
        let stdout = match child.stdout.take() {
            Some(stdout) => stdout,
            None => return monitor_failure("monitor_stdout_unavailable", "stdout unavailable"),
        };
        let stderr = match child.stderr.take() {
            Some(stderr) => stderr,
            None => return monitor_failure("monitor_stderr_unavailable", "stderr unavailable"),
        };
        let mut stdout_lines = tokio::io::BufReader::new(stdout).lines();
        let mut stderr_lines = tokio::io::BufReader::new(stderr).lines();
        let mut stdout_done = false;
        let mut stderr_done = false;
        let deadline = (!params.persistent)
            .then(|| tokio::time::Instant::now() + Duration::from_millis(params.timeout_ms));
        let mut timeout = deadline.map(|deadline| Box::pin(tokio::time::sleep_until(deadline)));
        let mut timed_out = false;
        let mut cancelled = false;
        let mut line_sequence = 0u64;

        while !stdout_done || !stderr_done {
            let cancel_future = async {
                match cancel.as_ref() {
                    Some(token) => token.cancelled().await,
                    None => std::future::pending::<()>().await,
                }
            };
            tokio::select! {
                _ = timeout.as_mut().unwrap(), if timeout.is_some() => {
                    timed_out = true;
                    break;
                }
                _ = cancel_future => {
                    cancelled = true;
                    break;
                }
                line = stdout_lines.next_line(), if !stdout_done => {
                    match line {
                        Ok(Some(line)) => {
                            line_sequence = line_sequence.saturating_add(1);
                            if let Err(err) = emit_monitor_line_event(context, params, line, true, line_sequence).await {
                                return monitor_failure("monitor_event_failed", err.to_string());
                            }
                        }
                        Ok(None) => stdout_done = true,
                        Err(err) => return monitor_failure("monitor_stdout_read_failed", err.to_string()),
                    }
                }
                line = stderr_lines.next_line(), if !stderr_done => {
                    match line {
                        Ok(Some(line)) => {
                            line_sequence = line_sequence.saturating_add(1);
                            if let Err(err) = emit_monitor_line_event(context, params, line, false, line_sequence).await {
                                return monitor_failure("monitor_event_failed", err.to_string());
                            }
                        }
                        Ok(None) => stderr_done = true,
                        Err(err) => return monitor_failure("monitor_stderr_read_failed", err.to_string()),
                    }
                }
            }
        }

        if timed_out || cancelled {
            terminate_pipe_process(child_pid);
        }

        let exit = if let Some(deadline) = deadline.filter(|_| !timed_out && !cancelled) {
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            match tokio::time::timeout(remaining, child.wait()).await {
                Ok(result) => result,
                Err(_) => {
                    timed_out = true;
                    terminate_pipe_process(child_pid);
                    child.wait().await
                }
            }
        } else {
            child.wait().await
        };

        if cancelled {
            return ToolResult::cancelled("monitor command was cancelled");
        }
        if timed_out {
            return monitor_failure(
                "monitor_timeout",
                format!("monitor timed out after {}ms", params.timeout_ms),
            );
        }
        match exit {
            Ok(status) if status.success() => ToolResult::ok(json!({
                "exit_status": status.code(),
            })),
            Ok(status) => monitor_failure(
                "monitor_command_failed",
                format!(
                    "monitor command exited with status {}",
                    status.code().unwrap_or_default()
                ),
            ),
            Err(err) => monitor_failure("monitor_wait_failed", err.to_string()),
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
        let poll_ms = args
            .get("poll_ms")
            .and_then(|value| value.as_u64())
            .unwrap_or(DEFAULT_WRITE_STDIN_POLL_MS);
        let close_stdin = args
            .get("close_stdin")
            .and_then(|value| value.as_bool())
            .unwrap_or(false);
        let allow_nonzero_exit = args
            .get("allow_nonzero_exit")
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
                Some(Duration::from_millis(poll_ms)),
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
                shell_io_result(
                    &id,
                    output,
                    Some(exit_code),
                    original_token_count,
                    full_output_path.as_deref(),
                    started.elapsed().as_secs_f64(),
                    allow_nonzero_exit,
                )
            }
            Ok(PollOutcome::Cancelled) => {
                self.runtime.remove_process(&id);
                ToolResult::cancelled("tool call cancelled")
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
    fn tool_manifests(&self) -> Vec<ToolManifest> {
        self.tool_definitions()
            .into_iter()
            .map(|tool| tool.manifest())
            .collect()
    }

    fn resolve_contract(&self, name: &str) -> Option<Arc<ToolContract>> {
        self.tool_definitions()
            .into_iter()
            .find(|tool| tool.name == name)
            .map(|tool| Arc::new(tool.contract()))
    }

    async fn execute(&self, call: ToolCall<'_>) -> ToolResult {
        let cancellation_token = call.context.cancellation_token().cloned();
        self.dispatch(
            call.name,
            call.args,
            call.context,
            call.progress,
            cancellation_token,
        )
        .await
    }
}

impl StandardShell {
    fn tool_definitions(&self) -> Vec<ToolDefinition> {
        let exec_command_description = "Run a noninteractive one-shot command with stdin closed and stdout/stderr captured, then wait for it to finish. Successful results always include `status: \"completed\"`, `done: true`, `running: false`, cleaned `output`, and `exit_code`. Commands time out after 600000 ms by default; set `timeout_ms` to override the hard timeout. Timed-out commands are killed and the result has `status: \"timed_out\"`, `timed_out: true`, and no `exit_code`; by default this fails the tool. Use `start_command` instead for interactive, TTY-dependent, or intentionally long-lived processes. Nonzero exit codes (including SIGPIPE 141 from `cmd | head`-style pipelines) fail the tool by default. Pass `allow_nonzero_exit: true` to receive the result without failure on either nonzero exit or timeout, then inspect `exit_code` and `timed_out`. ANSI/control noise is stripped from returned output. Large or truncated output may also include `full_output_path` pointing at the saved raw stream.";
        let start_command_description = "Start an interactive or intentionally long-lived command in a PTY. If the process is still alive after the initial poll window, the result includes `status: \"running\"`, `done: false`, `running: true`, and `session_id`; that output is partial and is not proof of completion. If the process exits during the poll window, the result is a normal completed command result. Nonzero exit codes fail the tool by default; pass `allow_nonzero_exit: true` only when nonzero is expected data, then inspect `exit_code`. Use `poll_ms` only to choose the initial observation window; use `exec_command.timeout_ms` for bounded one-shot commands. Use `exec_command` for builds, installs, tests, service setup, verification, and other commands that must complete before the next step.";
        let command_common = |command_description: &str| {
            json!({
                "cmd": {
                    "type": "string",
                    "description": command_description
                },
                "workdir": {
                    "type": "string",
                    "description": "Optional working directory to run the command in; defaults to the turn cwd."
                },
                "shell": {
                    "type": "string",
                    "description": "Shell binary to launch. Defaults to the user's default shell."
                },
                "login": {
                    "type": "boolean",
                    "default": false,
                    "description": "Whether to run the shell with -l semantics. Defaults to false to avoid startup prompts and shell init noise."
                },
                "allow_nonzero_exit": {
                    "type": "boolean",
                    "default": false,
                    "description": "Shell-only flag. When true, nonzero exit codes are returned as successful tool results instead of failed tool calls; inspect `exit_code` yourself. Defaults to false."
                },
                "max_output_tokens": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Maximum number of tokens to return. Excess output will be truncated."
                }
            })
        };
        let output_schema = json!({ "type": "object", "additionalProperties": true });
        vec![
            monitor_tool_definition(),
            ToolDefinition::raw(
                "tool:exec_command",
                "exec_command",
                exec_command_description,
                {
                    let mut properties = command_common("Shell command to execute.");
                    properties["timeout_ms"] = json!({
                        "type": "integer",
                        "minimum": 1,
                        "default": DEFAULT_EXEC_COMMAND_TIMEOUT_MS,
                        "description": "Hard timeout in milliseconds. If reached before the command exits, the process is killed and the result has `status: \"timed_out\"` and `timed_out: true`. By default this fails the tool; pass `allow_nonzero_exit: true` to receive the timed-out result without failure. Defaults to 600000 ms."
                    });
                    object_schema(properties, &["cmd"])
                },
                output_schema.clone(),
            )
            .with_examples(vec![
                r#"exec_command(cmd="cargo test -p lash-mode-rlm", timeout_ms=600000)"#.into(),
                r#"exec_command(cmd="test -f Cargo.lock", allow_nonzero_exit=true)"#.into(),
            ])
            .with_discovery(lash_tool_support::discovery_metadata("shell", &["shell", "bash"]))
            .with_execution_mode(ToolExecutionMode::Serial),
            ToolDefinition::raw(
                "tool:start_command",
                "start_command",
                start_command_description,
                {
                    let mut properties = command_common("Shell command to start.");
                    properties["poll_ms"] = json!({
                        "type": "integer",
                        "minimum": 1,
                        "default": DEFAULT_START_COMMAND_POLL_MS,
                        "description": "Initial observation window in milliseconds before returning a running `session_id` if the process has not exited. Defaults to 250. This is not a hard timeout."
                    });
                    object_schema(properties, &["cmd"])
                },
                output_schema.clone(),
            )
            .with_examples(vec![
                r#"start_command(cmd="python -m http.server 8000", poll_ms=1000)"#.into(),
            ])
            .with_discovery(lash_tool_support::discovery_metadata(
                "shell",
                &["long_running_command", "pty"],
            ))
            .with_execution_mode(ToolExecutionMode::Serial),
            ToolDefinition::raw(
                "tool:write_stdin",
                "write_stdin",
                "Write bytes to a running command handle from `start_command` and poll for the next settled cleaned output chunk. Use `close_stdin: true` to send EOF. Results with `status: \"running\"`, `done: false`, and `session_id` are partial; continue polling or writing until a completed result with `exit_code` if command completion matters. If the process exits, nonzero exit codes fail the tool by default; pass `allow_nonzero_exit: true` only when nonzero is expected data, then inspect `exit_code`. ANSI/control noise is stripped from returned output. Large or truncated output may also include `full_output_path` pointing at the saved raw stream.",
                object_schema(
                    json!({
                        "session_id": {
                            "type": "integer",
                            "description": "Identifier of the running command handle."
                        },
                        "chars": {
                            "type": "string",
                            "default": "",
                            "description": "Bytes to write to stdin (may be empty to poll)."
                        },
                        "poll_ms": {
                            "type": "integer",
                            "minimum": 1,
                            "default": DEFAULT_WRITE_STDIN_POLL_MS,
                            "description": "Poll window in milliseconds before returning another running result if the process has not exited. Defaults to 250."
                        },
                        "close_stdin": {
                            "type": "boolean",
                            "default": false,
                            "description": "Close stdin after writing to send EOF to the process."
                        },
                        "allow_nonzero_exit": {
                            "type": "boolean",
                            "default": false,
                            "description": "Shell-only flag. When true, nonzero process exit codes are returned as successful tool results instead of failed tool calls; inspect `exit_code` yourself. Defaults to false."
                        },
                        "max_output_tokens": {
                            "type": "integer",
                            "minimum": 1,
                            "description": "Maximum number of tokens to return. Excess output will be truncated."
                        }
                    }),
                    &["session_id"],
                ),
                output_schema,
            )
            .with_examples(vec![
                r#"write_stdin(session_id=1, chars="status\n", poll_ms=1000)"#.into(),
                r#"write_stdin(session_id=1, chars="", close_stdin=true)"#.into(),
            ])
            .with_discovery(lash_tool_support::discovery_metadata(
                "shell",
                &["send_stdin", "poll_command"],
            ))
            .with_execution_mode(ToolExecutionMode::Serial),
        ]
    }

    async fn dispatch(
        &self,
        name: &str,
        args: &serde_json::Value,
        context: &lash_core::ToolContext<'_>,
        progress: Option<&ProgressSender>,
        cancel: Option<CancellationToken>,
    ) -> ToolResult {
        match name {
            "monitor" => {
                let params = match self.parse_monitor_command_params(args) {
                    Ok(params) => params,
                    Err(err) => return err,
                };
                if context.async_process_id().is_some() {
                    self.run_monitor_process(&params, context, cancel).await
                } else {
                    self.start_monitor_process(&params, context).await
                }
            }
            "exec_command" => {
                let params = match self.parse_exec_command_params(args) {
                    Ok(params) => params,
                    Err(err) => return err,
                };
                self.exec_command(&params, progress, cancel).await
            }
            "start_command" => {
                let params = match self.parse_start_command_params(args) {
                    Ok(params) => params,
                    Err(err) => return err,
                };
                self.start_command(&params, progress, cancel).await
            }
            "write_stdin" => self.write_stdin_call(args, progress, cancel).await,
            _ => ToolResult::err_fmt(format_args!("Unknown tool: {name}")),
        }
    }
}

/// Build the shell-backed `monitor` tool definition.
pub fn monitor_tool_definition() -> ToolDefinition {
    ToolDefinition::raw(
        "tool:monitor",
        "monitor",
        "Run a background script that turns each stdout line into a process event and optional turn wake-up. Use for streaming watches (`tell me every time X happens`); for one-shot `wait until X`, run the command synchronously instead. This returns a process handle; use `list_process_handles` to rediscover live monitors and `cancel handle` to stop one.\n\nEvents arrive automatically as user-like input - do not call another tool to collect them. Return your turn after starting the monitor; the runtime wakes a new turn on the first matching line.\n\n**Pipe guards**\n- Always use `grep --line-buffered` in pipes (otherwise pipe buffering delays events by minutes).\n- Merge stderr into stdout (`cmd 2>&1 | grep ...`) - stderr alone is not observed.\n- In poll loops wrap transient failures (`curl ... || true`) and pick intervals >=30s for remote APIs, 0.5-1s for local checks.\n\n**Coverage - silence is not success.** Your filter must match every terminal state, not just the happy path. A monitor that greps only for the success marker stays silent through a crashloop, a hang, or an unexpected exit - and silence looks identical to `still running`. If you can't enumerate the failure signatures, broaden the alternation rather than narrow it.\n\nSet `persistent: true` for session-length watches. Timeout -> killed; exit ends the watch (exit code is reported).",
        json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command or script. Each stdout line is an event; exit ends the watch. Filter with `grep --line-buffered` (or equivalent) so only the lines you'd act on become events - including failure signatures, not just success."
                },
                "description": {
                    "type": "string",
                    "description": "Short human-readable description of what you are monitoring (shown in every notification). Be specific - \"errors in deploy.log\" beats \"watching logs\"."
                },
                "id": {
                    "type": "string",
                    "description": "Optional stable monitor id. Defaults to an opaque id."
                },
                "cwd": {
                    "type": "string",
                    "description": "Optional working directory to run the monitor command in; defaults to the turn cwd."
                },
                "env": {
                    "type": "object",
                    "additionalProperties": { "type": "string" },
                    "description": "Optional environment variables for the monitor command."
                },
                "persistent": {
                    "type": "boolean",
                    "default": false,
                    "description": "Run for the lifetime of the session (no timeout). Use for session-length watches like PR monitoring or log tails."
                },
                "timeout_ms": {
                    "type": "number",
                    "minimum": 1,
                    "maximum": MAX_MONITOR_TIMEOUT_MS,
                    "default": 300000,
                    "description": "Kill the monitor after this deadline. Default 300000ms, max 3600000ms. Ignored when persistent is true."
                }
            },
            "required": ["command", "description"],
            "additionalProperties": false
        }),
        json!({ "type": "object", "additionalProperties": true }),
    )
    .with_examples(vec![
        r#"monitor(command="tail -f /var/log/app.log | grep -E --line-buffered 'ERROR|Traceback|FAILED'", description="errors in app.log")"#.into(),
        r#"monitor(command="while true; do curl -sf http://localhost:3000/health && echo ready && break; sleep 2; done", description="local server ready", timeout_ms=300000)"#.into(),
    ])
    .with_execution_mode(ToolExecutionMode::Parallel)
    .with_process_start_mode(ToolProcessStartMode::ToolManaged)
}

async fn emit_monitor_line_event(
    context: &lash_core::ToolContext<'_>,
    params: &MonitorCommandParams,
    line: String,
    from_stdout: bool,
    line_sequence: u64,
) -> Result<(), PluginError> {
    let message = line.trim().to_string();
    if message.is_empty() {
        return Ok(());
    }
    let mut payload = json!({
        "line": message,
        "stream": if from_stdout { "stdout" } else { "stderr" },
    });
    if from_stdout && params.wake_policy != MonitorWakePolicy::Notify {
        payload["wake_input"] = json!(format!(
            "Monitor event \"{}\": {}",
            params.id,
            payload["line"].as_str().unwrap_or_default()
        ));
    }
    let idempotency_key = context
        .async_process_id()
        .map(|process_id| format!("{MONITOR_LINE_EVENT}:{process_id}:{line_sequence}"));
    context
        .emit_process_event_request(
            lash_core::ProcessEventAppendRequest::new(MONITOR_LINE_EVENT, payload)
                .with_optional_idempotency_key(idempotency_key),
        )
        .await
        .map(|_| ())
}

fn monitor_failure(code: impl Into<String>, message: impl Into<String>) -> ToolResult {
    ToolResult::failure(ToolFailure::tool(
        ToolFailureClass::Execution,
        code,
        message,
    ))
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

fn spawn_async_reader<R>(
    id: String,
    mut reader: R,
    buffer: Arc<StdMutex<Vec<u8>>>,
    buffer_start: Arc<StdMutex<usize>>,
    truncated: Arc<AtomicBool>,
    spill: Arc<StdMutex<Option<ShellOutputSpill>>>,
    output_notify: Arc<Notify>,
) where
    R: AsyncRead + Unpin + Send + 'static,
{
    tokio::spawn(async move {
        let mut chunk = [0u8; 4096];
        loop {
            match reader.read(&mut chunk).await {
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
    let running = exit_code.is_none();
    let status = if running { "running" } else { "completed" };
    let session_id = exit_code
        .is_none()
        .then(|| id.parse::<i64>().ok())
        .flatten();
    let mut record = serde_json::Map::new();
    record.insert("output".into(), json!(output));
    record.insert("status".into(), json!(status));
    record.insert("done".into(), json!(!running));
    record.insert("running".into(), json!(running));
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

fn shell_io_result(
    id: &str,
    output: String,
    exit_code: Option<i32>,
    original_token_count: Option<usize>,
    full_output_path: Option<&Path>,
    wall_time_seconds: f64,
    allow_nonzero_exit: bool,
) -> ToolResult {
    let mut record = standard_shell_io_record(
        id,
        output,
        exit_code,
        original_token_count,
        full_output_path,
        wall_time_seconds,
    );
    if let Some(code) = exit_code
        && code != 0
        && !allow_nonzero_exit
    {
        if let Some(object) = record.as_object_mut() {
            object.insert(
                "error".into(),
                json!(format!("Command exited with code {code}")),
            );
        }
        return ToolResult::err(record);
    }
    ToolResult::ok(record)
}

fn timed_out_shell_io_result(
    id: &str,
    output: String,
    original_token_count: Option<usize>,
    full_output_path: Option<&Path>,
    wall_time_seconds: f64,
    timeout_ms: u64,
    allow_nonzero_exit: bool,
) -> ToolResult {
    let mut record = standard_shell_io_record(
        id,
        output,
        None,
        original_token_count,
        full_output_path,
        wall_time_seconds,
    );
    if let Some(object) = record.as_object_mut() {
        object.insert("status".into(), json!("timed_out"));
        object.insert("done".into(), json!(true));
        object.insert("running".into(), json!(false));
        object.remove("session_id");
        object.insert("timed_out".into(), json!(true));
        object.insert(
            "error".into(),
            json!(format!("Command timed out after {timeout_ms} ms")),
        );
    }
    if allow_nonzero_exit {
        ToolResult::ok(record)
    } else {
        ToolResult::err(record)
    }
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

/// PluginFactory for the built-in shell tool surface.
///
/// Wires `StandardShell` into the active session with the access-gated
/// `write_stdin` mention in the prompt contribution so the model only
/// sees that bullet when the tool is actually callable.
#[derive(Default)]
pub struct StandardShellPluginFactory;

impl StandardShellPluginFactory {
    pub fn new() -> Self {
        Self
    }
}

impl PluginFactory for StandardShellPluginFactory {
    fn id(&self) -> &'static str {
        "shell"
    }

    fn build(&self, ctx: &PluginSessionContext) -> Result<Arc<dyn SessionPlugin>, PluginError> {
        let tool_access = ctx.tool_access.clone();
        let provider = Arc::new(StandardShell::new()) as Arc<dyn ToolProvider>;
        PluginSpecFactory::new(
            "shell",
            Arc::new(move |_ctx| {
                let provider = Arc::clone(&provider);
                let tool_access = tool_access.clone();
                Ok(PluginSpec::new()
                    .with_tool_provider(provider)
                    .with_prompt_contributor(Arc::new(move |_ctx| {
                        let tool_access = tool_access.clone();
                        Box::pin(
                            async move { Ok(shell_prompt_contributions_for_access(&tool_access)) },
                        )
                    })))
            }),
        )
        .build(ctx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::fs;

    fn test_shell() -> StandardShell {
        StandardShell::new().with_cwd("/")
    }

    async fn run(shell: &StandardShell, name: &str, args: &serde_json::Value) -> ToolResult {
        lash_core::testing::run_tool(shell, name, args).await
    }

    #[tokio::test]
    async fn monitor_starts_as_durable_tool_call_process() {
        use lash_core::ProcessRegistry;

        let shell = test_shell();
        let host = Arc::new(lash_core::testing::MockSessionManager::default());
        let context = lash_core::testing::mock_tool_context_with_host_and_processes(
            host.clone(),
            host.clone() as Arc<dyn lash_core::ProcessService>,
        );
        let args = json!({
            "command": "printf 'ready\\n'",
            "description": "readiness probe",
            "persistent": false,
            "timeout_ms": 1000,
        });

        let result = shell
            .execute(ToolCall {
                name: "monitor",
                args: &args,
                context: &context,
                progress: None,
            })
            .await;

        assert!(result.is_success(), "{}", result.value_for_projection());
        assert_eq!(result.value_for_projection()["__handle__"], "process");
        let process_id = result.value_for_projection()["id"]
            .as_str()
            .expect("process id")
            .to_string();
        let record = host
            .process_registry
            .get_process(&process_id)
            .await
            .expect("process record");
        let lash_core::ProcessInput::ToolCall { call } = record.input.as_ref() else {
            panic!("monitor should start as a tool-call process");
        };
        assert_eq!(call.tool_name, "monitor");
        assert_eq!(call.args["command"], args["command"]);
        assert!(
            record
                .event_types
                .iter()
                .any(|event_type| event_type.name == MONITOR_LINE_EVENT)
        );
    }

    #[test]
    fn monitor_tool_declares_tool_managed_process_start() {
        assert_eq!(
            monitor_tool_definition().process_start_mode,
            ToolProcessStartMode::ToolManaged
        );
    }

    #[tokio::test]
    async fn exec_command_returns_exit_code_when_command_finishes() {
        let shell = test_shell();
        let result = run(&shell, "exec_command", &json!({"cmd": "echo hello"})).await;
        assert!(result.is_success());
        assert!(result.value_for_projection().get("session_id").is_none());
        assert_eq!(result.value_for_projection()["status"], "completed");
        assert_eq!(result.value_for_projection()["done"], true);
        assert_eq!(result.value_for_projection()["running"], false);
        assert_eq!(result.value_for_projection()["exit_code"], 0);
        assert!(
            result.value_for_projection()["wall_time_seconds"]
                .as_f64()
                .is_some()
        );
        assert!(
            result.value_for_projection()["output"]
                .as_str()
                .unwrap()
                .contains("hello")
        );
    }

    #[tokio::test]
    async fn exec_command_waits_for_process_exit() {
        let shell = StandardShell::new().with_cwd("/");
        let result = run(
            &shell,
            "exec_command",
            &json!({"cmd": "sleep 0.05; echo done"}),
        )
        .await;
        assert!(result.is_success(), "{}", result.value_for_projection());
        assert!(result.value_for_projection().get("session_id").is_none());
        assert_eq!(result.value_for_projection()["status"], "completed");
        assert_eq!(result.value_for_projection()["done"], true);
        assert_eq!(result.value_for_projection()["exit_code"], 0);
        assert!(
            result.value_for_projection()["output"]
                .as_str()
                .unwrap()
                .contains("done")
        );
    }

    #[tokio::test]
    async fn exec_command_runs_without_a_tty() {
        let shell = test_shell();
        let result = run(
            &shell,
            "exec_command",
            &json!({"cmd": "if [ -t 0 ] || [ -t 1 ] || [ -t 2 ]; then echo tty; exit 1; else echo no-tty; fi"}),
        )
        .await;

        assert!(result.is_success(), "{}", result.value_for_projection());
        assert_eq!(result.value_for_projection()["exit_code"], 0);
        assert_eq!(
            result.value_for_projection()["output"]
                .as_str()
                .unwrap()
                .trim(),
            "no-tty"
        );
    }

    #[tokio::test]
    async fn exec_command_closes_stdin() {
        let shell = test_shell();
        let result = run(
            &shell,
            "exec_command",
            &json!({"cmd": "python3 -c 'import sys; print(sys.stdin.read() == \"\")'"}),
        )
        .await;

        assert!(result.is_success(), "{}", result.value_for_projection());
        assert_eq!(
            result.value_for_projection()["output"]
                .as_str()
                .unwrap()
                .trim(),
            "True"
        );
    }

    #[tokio::test]
    async fn exec_command_captures_stdout_and_stderr() {
        let shell = test_shell();
        let result = run(
            &shell,
            "exec_command",
            &json!({"cmd": "echo stdout-line; echo stderr-line >&2"}),
        )
        .await;

        assert!(result.is_success(), "{}", result.value_for_projection());
        let result_value = result.value_for_projection();
        let output = result_value["output"].as_str().unwrap();
        assert!(output.contains("stdout-line"), "{output}");
        assert!(output.contains("stderr-line"), "{output}");
    }

    #[tokio::test]
    async fn start_command_runs_in_a_pty() {
        let shell = test_shell();
        let result = run(
            &shell,
            "start_command",
            &json!({"cmd": "if [ -t 0 ] && [ -t 1 ]; then echo tty; else echo no-tty; exit 1; fi", "poll_ms": 1000}),
        )
        .await;

        assert!(result.is_success(), "{}", result.value_for_projection());
        assert_eq!(result.value_for_projection()["exit_code"], 0);
        assert_eq!(
            result.value_for_projection()["output"]
                .as_str()
                .unwrap()
                .trim(),
            "tty"
        );
    }

    #[tokio::test]
    async fn exec_command_timeout_kills_and_fails_running_process() {
        let shell = StandardShell::new().with_cwd("/");
        let result = run(
            &shell,
            "exec_command",
            &json!({"cmd": "printf started; sleep 5", "timeout_ms": 50}),
        )
        .await;
        assert!(!result.is_success(), "{}", result.value_for_projection());
        assert_eq!(result.value_for_projection()["status"], "timed_out");
        assert_eq!(result.value_for_projection()["done"], true);
        assert_eq!(result.value_for_projection()["running"], false);
        assert!(result.value_for_projection().get("session_id").is_none());
        assert!(
            result.value_for_projection()["output"]
                .as_str()
                .unwrap_or("")
                .contains("started")
        );
    }

    #[tokio::test]
    async fn exec_command_timeout_kills_process_group_children() {
        let shell = test_shell();
        let marker = std::env::temp_dir().join(format!(
            "lash-exec-timeout-child-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let cmd = format!(
            "sh -c 'sleep 0.4; echo leaked > {}' & wait",
            marker.display()
        );

        let result = run(
            &shell,
            "exec_command",
            &json!({"cmd": cmd, "timeout_ms": 50, "allow_nonzero_exit": true}),
        )
        .await;

        assert!(result.is_success(), "{}", result.value_for_projection());
        assert_eq!(result.value_for_projection()["status"], "timed_out");
        tokio::time::sleep(Duration::from_millis(600)).await;
        assert!(!marker.exists(), "timed-out child process wrote marker");
        let _ = fs::remove_file(marker);
    }

    #[tokio::test]
    async fn start_command_returns_handle_id_for_running_process() {
        let shell = StandardShell::new().with_cwd("/");
        let result = run(
            &shell,
            "start_command",
            &json!({"cmd": "sleep 1; echo done", "poll_ms": 10}),
        )
        .await;
        assert!(result.is_success());
        assert!(
            result.value_for_projection()["session_id"]
                .as_i64()
                .is_some()
        );
        assert_eq!(result.value_for_projection()["status"], "running");
        assert_eq!(result.value_for_projection()["done"], false);
        assert_eq!(result.value_for_projection()["running"], true);
        assert!(result.value_for_projection()["exit_code"].is_null());
    }

    #[tokio::test]
    async fn write_stdin_reuses_running_exec_handle() {
        let shell = test_shell();
        let cmd = "python3 -u -c 'import sys; line = sys.stdin.readline(); print(\"got:\" + line.strip())'";
        let open = run(
            &shell,
            "start_command",
            &json!({"cmd": cmd, "poll_ms": 10, "login": false}),
        )
        .await;
        assert!(open.is_success(), "{}", open.value_for_projection());
        let session_id = open.value_for_projection()["session_id"].as_i64().unwrap();

        let result = run(
            &shell,
            "write_stdin",
            &json!({"session_id": session_id, "chars": "hello\n", "poll_ms": 1000}),
        )
        .await;
        assert!(result.is_success());
        assert!(result.value_for_projection().get("session_id").is_none());
        assert_eq!(result.value_for_projection()["status"], "completed");
        assert_eq!(result.value_for_projection()["exit_code"], 0);
        assert!(
            result.value_for_projection()["output"]
                .as_str()
                .unwrap()
                .contains("got:hello")
        );
    }

    #[tokio::test]
    async fn write_stdin_prefers_completed_state_for_short_lived_commands() {
        let shell = test_shell();
        let cmd = "python3 -u -c 'import sys; line = sys.stdin.readline(); print(\"got:\" + line.strip())'";
        for _ in 0..16 {
            let open = run(
                &shell,
                "start_command",
                &json!({"cmd": cmd, "poll_ms": 10, "login": false}),
            )
            .await;
            assert!(open.is_success());
            let session_id = open.value_for_projection()["session_id"].as_i64().unwrap();

            let result = run(
                &shell,
                "write_stdin",
                &json!({"session_id": session_id, "chars": "hello\n", "poll_ms": 1000}),
            )
            .await;
            assert!(result.is_success());
            assert!(
                result.value_for_projection().get("session_id").is_none(),
                "expected completed handle, got: {}",
                result.value_for_projection()
            );
            assert_eq!(result.value_for_projection()["exit_code"], 0);
            assert!(
                result.value_for_projection()["output"]
                    .as_str()
                    .unwrap()
                    .contains("got:hello")
            );
        }
    }

    #[tokio::test]
    async fn write_stdin_can_close_stdin_to_send_eof() {
        let shell = test_shell();
        let open = run(
            &shell,
            "start_command",
            &json!({"cmd": "cat", "poll_ms": 10, "login": false}),
        )
        .await;
        assert!(open.is_success());
        let session_id = open.value_for_projection()["session_id"].as_i64().unwrap();

        let result = run(
            &shell,
            "write_stdin",
            &json!({"session_id": session_id, "chars": "hello", "close_stdin": true, "poll_ms": 1000}),
        )
        .await;
        assert!(result.is_success(), "{}", result.value_for_projection());
        let result_value = result.value_for_projection();
        assert!(result_value.get("session_id").is_none());
        assert_eq!(result_value["exit_code"], 0);
        let output = result_value["output"].as_str().unwrap();
        assert!(
            output.contains("hello"),
            "expected cat to echo input, got: {output}"
        );
    }

    #[tokio::test]
    async fn exec_command_honors_workdir() {
        let shell = StandardShell::new().with_cwd("/");
        let result = run(
            &shell,
            "exec_command",
            &json!({"cmd": "pwd", "workdir": "tmp"}),
        )
        .await;
        assert!(result.is_success());
        assert_eq!(
            result.value_for_projection()["output"]
                .as_str()
                .unwrap()
                .trim_end(),
            "/tmp"
        );
    }

    #[tokio::test]
    async fn exec_command_pipeline_failure_uses_pipefail() {
        let shell = test_shell();
        let result = run(&shell, "exec_command", &json!({"cmd": "false | cat"})).await;
        assert!(!result.is_success());
        assert_ne!(result.value_for_projection()["exit_code"], 0);
        assert_eq!(
            result.value_for_projection()["error"].as_str(),
            Some("Command exited with code 1")
        );
    }

    #[tokio::test]
    async fn exec_command_allow_nonzero_exit_returns_nonzero_as_success() {
        let shell = test_shell();
        let result = run(
            &shell,
            "exec_command",
            &json!({"cmd": "echo expected failure; exit 7", "allow_nonzero_exit": true}),
        )
        .await;
        assert!(result.is_success(), "{}", result.value_for_projection());
        assert_eq!(result.value_for_projection()["exit_code"], 7);
        assert!(result.value_for_projection()["error"].is_null());
        assert!(
            result.value_for_projection()["output"]
                .as_str()
                .unwrap()
                .contains("expected failure")
        );
    }

    #[tokio::test]
    async fn write_stdin_nonzero_exit_fails_by_default() {
        let shell = test_shell();
        let cmd = "python3 -u -c 'import sys; sys.stdin.readline(); sys.exit(7)'";
        let open = run(
            &shell,
            "start_command",
            &json!({"cmd": cmd, "poll_ms": 10, "login": false}),
        )
        .await;
        assert!(open.is_success(), "{}", open.value_for_projection());
        let session_id = open.value_for_projection()["session_id"].as_i64().unwrap();

        let result = run(
            &shell,
            "write_stdin",
            &json!({"session_id": session_id, "chars": "go\n", "poll_ms": 1000}),
        )
        .await;
        assert!(!result.is_success(), "{}", result.value_for_projection());
        assert_eq!(result.value_for_projection()["exit_code"], 7);
        assert_eq!(
            result.value_for_projection()["error"].as_str(),
            Some("Command exited with code 7")
        );
    }

    #[tokio::test]
    async fn exec_command_reports_full_output_path_when_token_truncated() {
        let shell = test_shell();
        let result = run(
            &shell,
            "exec_command",
            &json!({"cmd": "python3 -c 'print(\"hello \" * 4000)'", "max_output_tokens": 16, "login": false}),
        )
        .await;
        assert!(result.is_success(), "{}", result.value_for_projection());
        let result_value = result.value_for_projection();
        let output = result_value["output"].as_str().unwrap();
        let full_output_path = result_value["full_output_path"].as_str().unwrap();
        let full_output = fs::read_to_string(full_output_path).expect("full output file");
        assert!(output.contains("[truncated]"));
        assert!(full_output.contains("hello hello"));
    }

    #[tokio::test]
    async fn exec_command_spills_full_output_when_buffer_overflows() {
        let shell = test_shell();
        let result = run(
            &shell,
            "exec_command",
            &json!({"cmd": format!("python3 -c 'import sys; sys.stdout.write(\"x\" * {})'", MAX_OUTPUT + 8192), "login": false}),
        )
        .await;
        assert!(result.is_success(), "{}", result.value_for_projection());
        let result_value = result.value_for_projection();
        let output = result_value["output"].as_str().unwrap();
        let full_output_path = result_value["full_output_path"].as_str().unwrap();
        let full_output = fs::read_to_string(full_output_path).expect("full output file");
        assert!(output.contains("[truncated]"));
        assert!(full_output.len() >= MAX_OUTPUT + 8192);
    }

    #[tokio::test]
    async fn exec_command_reports_full_output_path_for_large_output() {
        let shell = test_shell();
        let result = run(
            &shell,
            "exec_command",
            &json!({"cmd": format!("python3 -c 'import sys; sys.stdout.write(\"x\" * {})'", SPILL_OUTPUT_THRESHOLD + 4096), "login": false}),
        )
        .await;
        assert!(result.is_success(), "{}", result.value_for_projection());
        let result_value = result.value_for_projection();
        assert!(result_value["output"].as_str().is_some());
        let full_output_path = result_value["full_output_path"].as_str().unwrap();
        let full_output = fs::read_to_string(full_output_path).expect("full output file");
        assert!(full_output.len() >= SPILL_OUTPUT_THRESHOLD + 4096);
    }

    #[tokio::test]
    async fn write_stdin_reports_full_output_path_when_token_truncated() {
        let shell = test_shell();
        let cmd = "python3 -u -c 'import sys; data = sys.stdin.read(); sys.stdout.write(data)'";
        let open = run(
            &shell,
            "start_command",
            &json!({"cmd": cmd, "poll_ms": 10, "login": false}),
        )
        .await;
        assert!(open.is_success(), "{}", open.value_for_projection());
        let session_id = open.value_for_projection()["session_id"].as_i64().unwrap();
        let payload = "segment ".repeat(5000);

        let result = run(
            &shell,
            "write_stdin",
            &json!({"session_id": session_id, "chars": payload, "close_stdin": true, "poll_ms": 1000, "max_output_tokens": 24}),
        )
        .await;
        assert!(result.is_success(), "{}", result.value_for_projection());
        let result_value = result.value_for_projection();
        let output = result_value["output"].as_str().unwrap();
        let full_output_path = result_value["full_output_path"].as_str().unwrap();
        let full_output = fs::read_to_string(full_output_path).expect("full output file");
        assert!(output.contains("[truncated]"));
        assert!(full_output.contains("segment segment"));
    }

    #[test]
    fn shell_definitions_are_compact_and_non_empty() {
        let shell = StandardShell::default();
        let defs = shell.tool_definitions();
        assert_eq!(defs.len(), 4);
        assert!(defs.iter().any(|def| def.name == "monitor"));
        assert!(defs.iter().all(|def| !def.description.is_empty()));
    }

    #[test]
    fn start_command_contract_distinguishes_poll_from_timeout() {
        let shell = StandardShell::default();
        let definition = shell
            .tool_definitions()
            .into_iter()
            .find(|definition| definition.name == "start_command")
            .expect("start_command definition");
        let properties = definition
            .input_schema
            .get("properties")
            .and_then(serde_json::Value::as_object)
            .expect("properties");

        assert!(properties.contains_key("poll_ms"));
        assert!(!properties.contains_key("timeout_ms"));
        assert!(definition.description.contains("exec_command.timeout_ms"));
        assert!(
            properties["poll_ms"]["description"]
                .as_str()
                .unwrap()
                .contains("not a hard timeout")
        );
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
    fn exec_command_defaults_to_generous_timeout() {
        let shell = StandardShell::default();
        let params = shell
            .parse_exec_command_params(&json!({"cmd": "echo hello"}))
            .expect("params");

        assert_eq!(params.timeout_ms, DEFAULT_EXEC_COMMAND_TIMEOUT_MS);
    }

    #[test]
    fn exec_command_timeout_schema_documents_default() {
        let shell = StandardShell::default();
        let definition = shell
            .tool_definitions()
            .into_iter()
            .find(|definition| definition.name == "exec_command")
            .expect("exec_command definition");
        let properties = definition
            .input_schema
            .get("properties")
            .and_then(serde_json::Value::as_object)
            .expect("properties");

        assert_eq!(
            properties["timeout_ms"]["default"],
            DEFAULT_EXEC_COMMAND_TIMEOUT_MS
        );
        assert!(
            definition
                .description
                .contains("Commands time out after 600000 ms by default")
        );
    }

    #[test]
    fn clean_terminal_output_strips_ansi_and_controls() {
        let raw = "\x1b[?2004h\x1b[31mred\x1b[0m\r\nab\x08c\x1b]0;title\x07\x00";

        assert_eq!(clean_terminal_output(raw), "red\nac");
    }

    #[tokio::test]
    async fn exec_command_cancel_token_kills_running_child() {
        use std::time::Instant;

        let shell = test_shell();
        let token = CancellationToken::new();
        let ctx = lash_core::testing::mock_tool_context().with_async_process("test", token.clone());

        // A long-running sleep that would otherwise hold the tool call for
        // 5s. The dispatcher must return promptly once the token fires, and
        // the pipe-backed process group must be killed rather than left to run.
        let args = json!({
            "cmd": "sleep 5",
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
            .execute(ToolCall {
                name: "exec_command",
                args: &args,
                context: &ctx,
                progress: None,
            })
            .await;
        let elapsed = started.elapsed();
        let _ = cancel_handle.await;

        assert!(
            elapsed < Duration::from_secs(1),
            "cancelled dispatch should return in under 1s (took {elapsed:?})"
        );
        assert!(!result.is_success(), "cancelled result should be an error");
        assert!(
            result
                .value_for_projection()
                .to_string()
                .contains("tool call cancelled")
        );
    }

    #[tokio::test]
    async fn cancel_during_write_stdin_wait_kills_child_by_pid() {
        use std::time::Instant;

        fn pid_alive(pid: i32) -> bool {
            // On Linux, /proc/<pid> disappears once the kernel reaps the
            // task. Use that as a portable stand-in for kill(pid, 0) without
            // pulling in a new dep.
            std::path::Path::new(&format!("/proc/{pid}")).exists()
        }

        let shell = test_shell();
        let token = CancellationToken::new();
        let ctx = lash_core::testing::mock_tool_context().with_async_process("test", token.clone());

        // Open a long-lived child. `echo $$` reports the shell's pid, then
        // `exec sleep 5` replaces the shell with sleep so the printed pid is
        // exactly the process the ChildKiller targets.
        let args = json!({
            "cmd": "echo $$; exec sleep 5",
            "poll_ms": 500,
            "login": false,
        });
        let open = run(&shell, "start_command", &args).await;
        assert!(open.is_success(), "{}", open.value_for_projection());
        let open_value = open.value_for_projection();
        let session_id = open_value["session_id"]
            .as_i64()
            .expect("expected a running session_id");
        let captured = open_value["output"].as_str().unwrap_or("");
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

        let stdin_args = json!({"session_id": session_id, "chars": "", "poll_ms": 30_000});
        let started = Instant::now();
        let result = shell
            .execute(ToolCall {
                name: "write_stdin",
                args: &stdin_args,
                context: &ctx,
                progress: None,
            })
            .await;
        let elapsed = started.elapsed();
        let _ = cancel_handle.await;

        assert!(
            elapsed < Duration::from_secs(1),
            "cancelled dispatch should return in under 1s (took {elapsed:?})"
        );
        assert!(!result.is_success(), "cancelled result should be an error");
        assert!(
            result
                .value_for_projection()
                .to_string()
                .contains("tool call cancelled")
        );

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
