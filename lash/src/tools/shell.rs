use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{
    Arc, Mutex as StdMutex,
    atomic::{AtomicI32, Ordering},
};
use std::time::{Duration, Instant};

use serde_json::json;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::Child;
use tokio::sync::Notify;

use crate::{ProgressSender, SandboxMessage, ToolDefinition, ToolParam, ToolProvider, ToolResult};

use super::require_str;

struct ShellProcess {
    pid: u32,
    stdin: Option<tokio::process::ChildStdin>,
    buffer: Arc<StdMutex<String>>,
    read_cursor: Arc<StdMutex<usize>>,
    exit_code: Arc<StdMutex<Option<i32>>>,
    exit_notify: Arc<Notify>,
    default_timeout_ms: Option<u64>,
}

type ProcessState = (u32, Arc<StdMutex<Option<i32>>>, Arc<Notify>, Option<u64>);

const MAX_OUTPUT: usize = 512_000;
const DEFAULT_EXEC_YIELD_MS: u64 = 10_000;
const DEFAULT_WRITE_STDIN_YIELD_MS: u64 = 250;

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
    tty: bool,
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

    fn spawn_child(
        &self,
        command: &str,
        workdir: &PathBuf,
        login: bool,
        shell_path: &str,
    ) -> Result<tokio::process::Child, String> {
        let mut cmd = tokio::process::Command::new(shell_path);
        for arg in self.shell_args(command, login, shell_path)? {
            cmd.arg(arg);
        }
        cmd.current_dir(workdir)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());
        #[cfg(unix)]
        cmd.process_group(0);
        cmd.spawn().map_err(|e| format!("Failed to spawn: {e}"))
    }

    fn spawn_process(
        &self,
        id: String,
        command: &str,
        workdir: &PathBuf,
        login: bool,
        shell_path: &str,
        timeout_ms: Option<u64>,
    ) -> Result<u32, String> {
        let mut child = self.spawn_child(command, workdir, login, shell_path)?;
        let pid = child.id().unwrap_or(0);
        let stdin = child.stdin.take();
        let buffer = Arc::new(StdMutex::new(String::new()));
        let read_cursor = Arc::new(StdMutex::new(0usize));
        let exit_code = Arc::new(StdMutex::new(None));
        let exit_notify = Arc::new(Notify::new());

        spawn_reader(
            child,
            buffer.clone(),
            exit_code.clone(),
            exit_notify.clone(),
        );

        let process = ShellProcess {
            pid,
            stdin,
            buffer,
            read_cursor,
            exit_code,
            exit_notify,
            default_timeout_ms: timeout_ms,
        };
        self.processes.lock().unwrap().insert(id, process);
        Ok(pid)
    }

    fn allocate_session_id(&self) -> i32 {
        self.next_session_id.fetch_add(1, Ordering::SeqCst)
    }

    fn take_incremental_output(
        &self,
        id: &str,
        max_output_tokens: Option<usize>,
    ) -> Result<(String, Option<usize>), String> {
        let (buffer, read_cursor) = {
            let procs = self.processes.lock().unwrap();
            let proc = procs
                .get(id)
                .ok_or_else(|| format!("Unknown session id {id}"))?;
            (proc.buffer.clone(), proc.read_cursor.clone())
        };

        let buf = buffer.lock().unwrap();
        let mut cursor = read_cursor.lock().unwrap();
        let new_output = if *cursor <= buf.len() {
            buf[*cursor..].to_string()
        } else {
            buf.clone()
        };
        *cursor = buf.len();
        Ok(truncate_exec_output(new_output, max_output_tokens))
    }

    fn full_output(&self, id: &str) -> Result<String, String> {
        let procs = self.processes.lock().unwrap();
        let proc = procs
            .get(id)
            .ok_or_else(|| format!("No process with id: {id}"))?;
        let mut output = proc.buffer.lock().unwrap().clone();
        if output.len() > MAX_OUTPUT {
            output.truncate(MAX_OUTPUT);
            output.push_str("\n[truncated]");
        }
        Ok(output)
    }

    fn process_state(&self, id: &str) -> Result<ProcessState, String> {
        let procs = self.processes.lock().unwrap();
        let proc = procs
            .get(id)
            .ok_or_else(|| format!("No process with id: {id}"))?;
        Ok((
            proc.pid,
            proc.exit_code.clone(),
            proc.exit_notify.clone(),
            proc.default_timeout_ms,
        ))
    }

    async fn wait_until_exit_or_timeout(
        &self,
        id: &str,
        timeout: Option<Duration>,
        progress: Option<&ProgressSender>,
        incremental: bool,
        max_output_tokens: Option<usize>,
    ) -> Result<PollOutcome, String> {
        let (_pid, exit_code, exit_notify, _) = self.process_state(id)?;
        let deadline = timeout.map(|value| tokio::time::Instant::now() + value);
        let mut sent_len = 0usize;

        loop {
            let exited = exit_code.lock().unwrap().is_some();

            if let Some(tx) = progress {
                let new_chunk = {
                    let procs = self.processes.lock().unwrap();
                    let proc = procs
                        .get(id)
                        .ok_or_else(|| format!("No process with id: {id}"))?;
                    let buf = proc.buffer.lock().unwrap();
                    if buf.len() > sent_len {
                        let chunk = buf[sent_len..].to_string();
                        sent_len = buf.len();
                        Some(chunk)
                    } else {
                        None
                    }
                };
                if let Some(chunk) = new_chunk {
                    let _ = tx.send(SandboxMessage {
                        text: chunk,
                        kind: "tool_output".into(),
                    });
                }
            }

            if exited {
                let output = if incremental {
                    self.take_incremental_output(id, max_output_tokens)?
                } else {
                    (self.full_output(id)?, None)
                };
                let code = exit_code.lock().unwrap().unwrap_or(-1);
                return Ok(PollOutcome::Exited {
                    output: output.0,
                    original_token_count: output.1,
                    exit_code: code,
                });
            }

            if let Some(dl) = deadline
                && tokio::time::Instant::now() >= dl
            {
                let output = if incremental {
                    self.take_incremental_output(id, max_output_tokens)?
                } else {
                    (self.full_output(id)?, None)
                };
                return Ok(PollOutcome::Running {
                    output: output.0,
                    original_token_count: output.1,
                });
            }

            tokio::select! {
                _ = exit_notify.notified() => {}
                _ = tokio::time::sleep(Duration::from_millis(25)) => {}
            }
        }
    }

    async fn kill_process(&self, id: &str) -> Result<(), String> {
        let (pid, exit_notify, _) = {
            let procs = self.processes.lock().unwrap();
            let proc = procs
                .get(id)
                .ok_or_else(|| format!("No process with id: {id}"))?;
            (proc.pid, proc.exit_notify.clone(), proc.default_timeout_ms)
        };

        let _ = tokio::process::Command::new("kill")
            .arg("--")
            .arg(format!("-{pid}"))
            .stderr(std::process::Stdio::null())
            .status()
            .await;

        let exited = tokio::time::timeout(Duration::from_secs(2), exit_notify.notified()).await;
        if exited.is_err() {
            let _ = tokio::process::Command::new("kill")
                .args(["-9", "--", &format!("-{pid}")])
                .stderr(std::process::Stdio::null())
                .status()
                .await;
            let _ = tokio::process::Command::new("kill")
                .args(["-9", &pid.to_string()])
                .stderr(std::process::Stdio::null())
                .status()
                .await;
        }

        self.processes.lock().unwrap().remove(id);
        Ok(())
    }

    async fn kill_for_timeout(&self, id: &str) -> Result<(), String> {
        self.kill_process(id).await
    }

    fn remove_process(&self, id: &str) {
        self.processes.lock().unwrap().remove(id);
    }

    async fn write_stdin(&self, id: &str, input: &str) -> Result<(), String> {
        let stdin = {
            let mut procs = self.processes.lock().unwrap();
            let proc = procs
                .get_mut(id)
                .ok_or_else(|| format!("Unknown session id {id}"))?;
            proc.stdin.take()
        };

        if let Some(mut stdin) = stdin {
            let result = stdin.write_all(input.as_bytes()).await;
            let mut procs = self.processes.lock().unwrap();
            if let Some(proc) = procs.get_mut(id) {
                proc.stdin = Some(stdin);
            }
            result.map_err(|e| format!("Write failed: {e}"))
        } else {
            Err("Process stdin not available".to_string())
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
        let tty = args
            .get("tty")
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
            tty,
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
        let session_id = self.runtime.allocate_session_id();
        let session_key = session_id.to_string();
        let _ = params.tty;

        if let Err(err) = self.runtime.spawn_process(
            session_key.clone(),
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
                &session_key,
                Some(Duration::from_millis(params.yield_time_ms)),
                progress,
                true,
                params.max_output_tokens,
            )
            .await
        {
            Ok(PollOutcome::Running {
                output,
                original_token_count,
            }) => ToolResult::ok(json!({
                "wall_time_seconds": started.elapsed().as_secs_f64(),
                "session_id": session_id,
                "original_token_count": original_token_count,
                "output": output,
            })),
            Ok(PollOutcome::Exited {
                output,
                original_token_count,
                exit_code,
            }) => {
                self.runtime.remove_process(&session_key);
                ToolResult::ok(json!({
                    "wall_time_seconds": started.elapsed().as_secs_f64(),
                    "exit_code": exit_code,
                    "original_token_count": original_token_count,
                    "output": output,
                }))
            }
            Err(err) => {
                self.runtime.remove_process(&session_key);
                ToolResult::err(json!(err))
            }
        }
    }

    async fn write_stdin_call(
        &self,
        args: &serde_json::Value,
        progress: Option<&ProgressSender>,
    ) -> ToolResult {
        let session_id = args
            .get("session_id")
            .and_then(|value| value.as_i64())
            .ok_or_else(|| ToolResult::err_fmt("Missing required parameter: session_id"));
        let session_id = match session_id {
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
        let session_key = session_id.to_string();
        let started = Instant::now();

        if let Err(err) = self.runtime.write_stdin(&session_key, chars).await {
            return ToolResult::err(json!(err));
        }

        match self
            .runtime
            .wait_until_exit_or_timeout(
                &session_key,
                Some(Duration::from_millis(yield_time_ms)),
                progress,
                true,
                max_output_tokens,
            )
            .await
        {
            Ok(PollOutcome::Running {
                output,
                original_token_count,
            }) => ToolResult::ok(json!({
                "wall_time_seconds": started.elapsed().as_secs_f64(),
                "session_id": session_id,
                "original_token_count": original_token_count,
                "output": output,
            })),
            Ok(PollOutcome::Exited {
                output,
                original_token_count,
                exit_code,
            }) => {
                self.runtime.remove_process(&session_key);
                ToolResult::ok(json!({
                    "wall_time_seconds": started.elapsed().as_secs_f64(),
                    "exit_code": exit_code,
                    "original_token_count": original_token_count,
                    "output": output,
                }))
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
            Ok(pid) => ToolResult::ok(json!({
                "__handle__": "shell",
                "id": id,
                "pid": pid,
                "command": params.command,
                "workdir": params.workdir.display().to_string(),
                "timeout_ms": params.timeout_ms,
                "login": params.login,
            })),
            Err(err) => ToolResult::err(json!(err)),
        }
    }

    async fn shell_wait(
        &self,
        id: &str,
        timeout: Option<f64>,
        progress: Option<&ProgressSender>,
    ) -> ToolResult {
        let (_, _, _, default_timeout_ms) = match self.runtime.process_state(id) {
            Ok(state) => state,
            Err(err) => return ToolResult::err(json!(err)),
        };
        let timeout = timeout.or_else(|| default_timeout_ms.map(|ms| ms as f64 / 1000.0));

        match self
            .runtime
            .wait_until_exit_or_timeout(
                id,
                timeout.map(Duration::from_secs_f64),
                progress,
                false,
                None,
            )
            .await
        {
            Ok(PollOutcome::Exited {
                mut output,
                exit_code,
                ..
            }) => {
                if exit_code != 0 {
                    output.push_str(&format!("\n[exit code: {exit_code}]"));
                }
                self.runtime.remove_process(id);
                ToolResult {
                    success: exit_code == 0,
                    result: json!(output),
                    images: vec![],
                }
            }
            Ok(PollOutcome::Running { mut output, .. }) => {
                if let Err(err) = self.runtime.kill_for_timeout(id).await {
                    return ToolResult::err(json!(err));
                }
                output.push_str(
                    "\n[timed out: process exceeded timeout; retry with a larger timeout or split into smaller commands]",
                );
                ToolResult::err(json!(output))
            }
            Err(err) => ToolResult::err(json!(err)),
        }
    }

    fn shell_read(&self, id: &str) -> ToolResult {
        let output = match self.runtime.take_incremental_output(id, None) {
            Ok((output, _)) => output,
            Err(err) => return ToolResult::err_fmt(err),
        };
        ToolResult::ok(json!(output))
    }

    async fn shell_write(&self, id: &str, input: &str) -> ToolResult {
        match self.runtime.write_stdin(id, input).await {
            Ok(()) => ToolResult::ok(json!(null)),
            Err(err) => ToolResult::err(json!(err)),
        }
    }

    async fn shell_kill(&self, id: &str) -> ToolResult {
        match self.runtime.kill_process(id).await {
            Ok(()) => ToolResult::ok(json!(null)),
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
                    "Runs a command in a PTY, returning output or a session ID for ongoing interaction.",
                    [crate::ExecutionMode::Standard],
                )],
                params: vec![
                    ToolParam::typed("cmd", "str"),
                    ToolParam::optional("workdir", "str"),
                    ToolParam::optional("shell", "str"),
                    ToolParam::optional("login", "bool"),
                    ToolParam::optional("tty", "bool"),
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
                    "Writes characters to an existing unified exec session and returns recent output.",
                    [crate::ExecutionMode::Standard],
                )],
                params: vec![
                    ToolParam::typed("session_id", "int"),
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
                        "Start a command via {shell_name} and return a handle for interactive or long-running work. Use this when you need to wait, poll output, send stdin, or kill the process later. Set `workdir` instead of relying on `cd` when possible."
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
                    "Wait for a shell handle to exit and return its full output. Use this for commands started with `shell(...)` that should eventually terminate.",
                    [crate::ExecutionMode::Repl],
                )],
                params: vec![
                    ToolParam::typed("id", "str"),
                    ToolParam::optional("timeout", "float"),
                ],
                returns: "str".into(),
                examples: vec![],
                hidden: false,
                inject_into_prompt: false,
            },
            ToolDefinition {
                name: "shell_read".into(),
                description: vec![crate::ToolText::new(
                    "Read and drain buffered output from a running shell handle. Non-blocking; returns an empty string if nothing new is available.",
                    [crate::ExecutionMode::Repl],
                )],
                params: vec![ToolParam::typed("id", "str")],
                returns: "str".into(),
                examples: vec![],
                hidden: false,
                inject_into_prompt: false,
            },
            ToolDefinition {
                name: "shell_write".into(),
                description: vec![crate::ToolText::new(
                    "Write input to a shell handle's stdin.",
                    [crate::ExecutionMode::Repl],
                )],
                params: vec![
                    ToolParam::typed("id", "str"),
                    ToolParam::typed("input", "str"),
                ],
                returns: "None".into(),
                examples: vec![],
                hidden: false,
                inject_into_prompt: false,
            },
            ToolDefinition {
                name: "shell_kill".into(),
                description: vec![crate::ToolText::new(
                    "Terminate a running shell handle.",
                    [crate::ExecutionMode::Repl],
                )],
                params: vec![ToolParam::typed("id", "str")],
                returns: "None".into(),
                examples: vec![],
                hidden: false,
                inject_into_prompt: false,
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
                let timeout = args.get("timeout").and_then(|value| value.as_f64());
                self.shell_wait(id, timeout, progress).await
            }
            "shell_read" => {
                let id = match require_str(args, "id") {
                    Ok(value) => value,
                    Err(err) => return err,
                };
                self.shell_read(id)
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
                self.shell_write(id, input).await
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

fn spawn_reader(
    mut child: Child,
    buffer: Arc<StdMutex<String>>,
    exit_code: Arc<StdMutex<Option<i32>>>,
    exit_notify: Arc<Notify>,
) {
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();

    tokio::spawn(async move {
        let buf_out = buffer.clone();
        let buf_err = buffer.clone();

        let out_task = tokio::spawn(async move {
            if let Some(stdout) = stdout {
                let mut reader = BufReader::new(stdout);
                let mut line = String::new();
                while let Ok(n) = reader.read_line(&mut line).await {
                    if n == 0 {
                        break;
                    }
                    let mut buf = buf_out.lock().unwrap();
                    if buf.len() < MAX_OUTPUT {
                        buf.push_str(&line);
                    }
                    line.clear();
                }
            }
        });

        let err_task = tokio::spawn(async move {
            if let Some(stderr) = stderr {
                let mut reader = BufReader::new(stderr);
                let mut line = String::new();
                while let Ok(n) = reader.read_line(&mut line).await {
                    if n == 0 {
                        break;
                    }
                    let mut buf = buf_err.lock().unwrap();
                    if buf.len() < MAX_OUTPUT {
                        buf.push_str(&line);
                    }
                    line.clear();
                }
            }
        });

        let _ = out_task.await;
        let _ = err_task.await;

        let status = child.wait().await;
        let code = status.map(|s| s.code().unwrap_or(-1)).unwrap_or(-1);
        *exit_code.lock().unwrap() = Some(code);
        exit_notify.notify_waiters();
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

    if let Some(timeout) = args.get("timeout").and_then(|value| value.as_f64()) {
        return Ok(Some((timeout * 1000.0).round() as u64));
    }

    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

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
            .execute("shell_wait", &json!({"id": id, "timeout": 5.0}))
            .await;
        assert!(result.success);
        assert!(result.result.as_str().unwrap().contains("hello"));
    }

    #[tokio::test]
    async fn exec_command_returns_exit_code_when_command_finishes() {
        let shell = StandardShell::default();
        let result = shell
            .execute("exec_command", &json!({"cmd": "echo hello"}))
            .await;
        assert!(result.success);
        assert_eq!(result.result["exit_code"], 0);
        assert!(result.result["output"].as_str().unwrap().contains("hello"));
        assert!(result.result.get("session_id").is_none());
    }

    #[tokio::test]
    async fn exec_command_returns_session_id_for_running_process() {
        let shell = StandardShell::default();
        let result = shell
            .execute(
                "exec_command",
                &json!({"cmd": "sleep 1; echo done", "yield_time_ms": 10}),
            )
            .await;
        assert!(result.success);
        assert!(result.result["session_id"].as_i64().is_some());
        assert!(result.result.get("exit_code").is_none());
    }

    #[tokio::test]
    async fn write_stdin_reuses_running_exec_session() {
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
        assert_eq!(result.result["exit_code"], 0);
        assert!(
            result.result["output"]
                .as_str()
                .unwrap()
                .contains("got:hello")
        );
    }

    #[tokio::test]
    async fn exec_command_honors_workdir() {
        let shell = StandardShell::new().with_cwd("/");
        let result = shell
            .execute("exec_command", &json!({"cmd": "pwd", "workdir": "tmp"}))
            .await;
        assert!(result.success);
        assert!(result.result["output"].as_str().unwrap().trim_end() == "/tmp");
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
