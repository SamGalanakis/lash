//! Built-in shell tool surface (`shell.exec` / `shell.start` /
//! `shell.write`).
//!
//! This module is the *surface* layer: tool definitions, argument parsing,
//! the [`StandardShell`] executor, prompt contributions, and the plugin
//! factory. The process-lifecycle machinery lives in [`runtime`] and the
//! output-buffer plumbing in [`output`].

mod output;
mod runtime;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::json;
use tokio_util::sync::CancellationToken;

use lash_core::plugin::{
    PluginError, PluginFactory, PluginSessionContext, PluginSpec, PluginSpecFactory, SessionPlugin,
};
use lash_core::{
    ProgressSender, PromptContribution, SessionToolAccess, ToolCall, ToolDefinition, ToolProvider,
    ToolResult, ToolScheduling,
};

use lash_tool_support::{
    StaticToolExecute, StaticToolProvider, object_schema, parse_optional_bool,
    parse_optional_usize_arg, require_str,
};

use crate::output::{
    PollOutcome, shell_io_result, standard_shell_io_record, timed_out_shell_io_result,
};
use crate::runtime::{
    CommonCommandParams, DEFAULT_EXEC_COMMAND_TIMEOUT_MS, DEFAULT_START_COMMAND_POLL_MS,
    DEFAULT_WRITE_STDIN_POLL_MS, ExecCommandParams, PipeExecProcessRequest, ShellRuntime,
    StartCommandParams, WaitBehavior,
};

pub fn shell_prompt_contributions() -> Vec<PromptContribution> {
    shell_prompt_contributions_for_access(&SessionToolAccess::default())
}

/// Returns the shell prompt contributions, gating the `shell.write`
/// reference on whether that tool is actually callable in the current
/// session.
pub fn shell_prompt_contributions_for_access(
    access: &SessionToolAccess,
) -> Vec<PromptContribution> {
    let mut command_execution = String::from(
        "Use `shell.exec` for one-shot commands; it returns only after the process exits and successful results include `status: \"completed\"`, `done: true`, and `exit_code`. Use `shell.start` only for interactive or intentionally long-lived processes; it may return `status: \"running\"`, `done: false`, and `session_id`, which means the output is partial and must not be treated as completion.",
    );
    if tool_callable_from_authority(access, "write_stdin") {
        command_execution.push_str(" Continue running sessions with `shell.write`.");
    }
    command_execution.push_str(
        " For builds, installs, tests, migrations, service setup, and verification commands, use `shell.exec` and wait for completion before concluding.",
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
    access.tools.is_empty() || access.tools.iter().any(|tool| tool.name() == name)
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
        let login = parse_optional_bool(args, "login", false)?;
        let allow_nonzero_exit = parse_optional_bool(args, "allow_nonzero_exit", false)?;
        let max_output_tokens = parse_optional_usize_arg(args, "max_output_tokens", None, true, 1)?;

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
        let timeout_ms = parse_optional_usize_arg(args, "timeout_ms", None, false, 1)?
            .map(|value| value as u64)
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
        let poll_ms = parse_optional_usize_arg(args, "poll_ms", None, false, 1)?
            .map(|value| value as u64)
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
        let poll_ms = match parse_optional_usize_arg(args, "poll_ms", None, false, 1) {
            Ok(value) => value
                .map(|value| value as u64)
                .unwrap_or(DEFAULT_WRITE_STDIN_POLL_MS),
            Err(err) => return err,
        };
        let close_stdin = match parse_optional_bool(args, "close_stdin", false) {
            Ok(value) => value,
            Err(err) => return err,
        };
        let allow_nonzero_exit = match parse_optional_bool(args, "allow_nonzero_exit", false) {
            Ok(value) => value,
            Err(err) => return err,
        };
        let max_output_tokens =
            match parse_optional_usize_arg(args, "max_output_tokens", None, true, 1) {
                Ok(value) => value,
                Err(err) => return err,
            };
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

/// Build the cached shell tool provider (`shell.exec` / `shell.start`).
pub fn shell_provider(shell: StandardShell) -> StaticToolProvider<StandardShell> {
    let definitions = shell.tool_definitions();
    StaticToolProvider::new(definitions, shell)
}

#[async_trait::async_trait]
impl StaticToolExecute for StandardShell {
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
        let exec_command_description = "Run a noninteractive one-shot command with stdin closed and stdout/stderr captured, then wait for it to finish. Successful results always include `status: \"completed\"`, `done: true`, `running: false`, cleaned `output`, and `exit_code`. Commands time out after 600000 ms by default; set `timeout_ms` to override the hard timeout. Timed-out commands are killed and the result has `status: \"timed_out\"`, `timed_out: true`, and no `exit_code`; by default this fails the tool. Use `shell.start` instead for interactive, TTY-dependent, or intentionally long-lived processes. Nonzero exit codes (including SIGPIPE 141 from `cmd | head`-style pipelines) fail the tool by default. Pass `allow_nonzero_exit: true` to receive the result without failure on either nonzero exit or timeout, then inspect `exit_code` and `timed_out`. ANSI/control noise is stripped from returned output. Large or truncated output may also include `full_output_path` pointing at the saved raw stream.";
        let start_command_description = "Start an interactive or intentionally long-lived command in a PTY. If the process is still alive after the initial poll window, the result includes `status: \"running\"`, `done: false`, `running: true`, and `session_id`; that output is partial and is not proof of completion. If the process exits during the poll window, the result is a normal completed command result. Nonzero exit codes fail the tool by default; pass `allow_nonzero_exit: true` only when nonzero is expected data, then inspect `exit_code`. Use `poll_ms` only to choose the initial observation window; use `shell.exec.timeout_ms` for bounded one-shot commands. Use `shell.exec` for builds, installs, tests, service setup, verification, and other commands that must complete before the next step.";
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
                r#"await shell.exec({ cmd: "cargo test -p lash-protocol-rlm", timeout_ms: 600000 })?"#.into(),
                r#"await shell.exec({ cmd: "test -f Cargo.lock", allow_nonzero_exit: true })?"#.into(),
            ])
            .with_agent_surface(lash_tool_support::agent_surface(
                ["shell"],
                "exec",
                &["shell", "bash"],
            ))
            .with_scheduling(ToolScheduling::Serial),
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
                r#"await shell.start({ cmd: "python -m http.server 8000", poll_ms: 1000 })?"#.into(),
            ])
            .with_agent_surface(lash_tool_support::agent_surface(
                ["shell"],
                "start",
                &["long_running_command", "pty"],
            ))
            .with_scheduling(ToolScheduling::Serial),
            ToolDefinition::raw(
                "tool:write_stdin",
                "write_stdin",
                "Write bytes to a running command handle from `shell.start` and poll for the next settled cleaned output chunk. Use `close_stdin: true` to send EOF. Results with `status: \"running\"`, `done: false`, and `session_id` are partial; continue polling or writing until a completed result with `exit_code` if command completion matters. If the process exits, nonzero exit codes fail the tool by default; pass `allow_nonzero_exit: true` only when nonzero is expected data, then inspect `exit_code`. ANSI/control noise is stripped from returned output. Large or truncated output may also include `full_output_path` pointing at the saved raw stream.",
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
                r#"await shell.write({ session_id: 1, chars: "status\n", poll_ms: 1000 })?"#.into(),
                r#"await shell.write({ session_id: 1, chars: "", close_stdin: true })?"#.into(),
            ])
            .with_agent_surface(lash_tool_support::agent_surface(
                ["shell"],
                "write",
                &["send_stdin", "poll_command"],
            ))
            .with_scheduling(ToolScheduling::Serial),
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
        let _ = context;
        match name {
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
/// `shell.write` mention in the prompt contribution so the model only
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
        let provider = Arc::new(shell_provider(StandardShell::new())) as Arc<dyn ToolProvider>;
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
    use crate::output::{MAX_OUTPUT, SPILL_OUTPUT_THRESHOLD, clean_terminal_output};
    use serde_json::json;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn test_shell() -> StaticToolProvider<StandardShell> {
        shell_provider(StandardShell::new().with_cwd("/"))
    }

    async fn run(
        shell: &StaticToolProvider<StandardShell>,
        name: &str,
        args: &serde_json::Value,
    ) -> ToolResult {
        lash_core::testing::run_tool(shell, name, args).await
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
        let shell = shell_provider(StandardShell::new().with_cwd("/"));
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
        let shell = shell_provider(StandardShell::new().with_cwd("/"));
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
        let shell = shell_provider(StandardShell::new().with_cwd("/"));
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
        let shell = shell_provider(StandardShell::new().with_cwd("/"));
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
        assert_eq!(defs.len(), 3);
        assert!(defs.iter().all(|def| !def.description().is_empty()));
    }

    #[test]
    fn start_command_contract_distinguishes_poll_from_timeout() {
        let shell = StandardShell::default();
        let definition = shell
            .tool_definitions()
            .into_iter()
            .find(|definition| definition.name() == "start_command")
            .expect("start_command definition");
        let properties = definition
            .contract
            .input_schema
            .get("properties")
            .and_then(serde_json::Value::as_object)
            .expect("properties");

        assert!(properties.contains_key("poll_ms"));
        assert!(!properties.contains_key("timeout_ms"));
        assert!(definition.description().contains("shell.exec.timeout_ms"));
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
            .find(|definition| definition.name() == "exec_command")
            .expect("exec_command definition");
        let properties = definition
            .contract
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
                .description()
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
