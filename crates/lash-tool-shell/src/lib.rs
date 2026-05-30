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
    PreparedToolCall, ProcessEventSemanticsSpec, ProcessEventType, ProcessHandleDescriptor,
    ProcessInput, ProcessRegistration, ProgressSender, PromptContribution, SessionToolAccess,
    ToolCall, ToolDefinition, ToolProvider, ToolResult, ToolScheduling,
};

use lash_tool_support::{
    StaticToolExecute, StaticToolProvider, object_schema, parse_optional_bool,
    parse_optional_usize_arg, require_str,
};

use crate::output::{PollOutcome, shell_io_result, timed_out_shell_io_result};
use crate::runtime::{
    CommonCommandParams, DEFAULT_EXEC_COMMAND_TIMEOUT_MS, ExecCommandParams,
    PipeExecProcessRequest, ShellRuntime, StartCommandParams, WaitBehavior,
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
        "Use `shell.exec` for one-shot commands; it returns only after the process exits and successful results include `status: \"completed\"`, `done: true`, and `exit_code`. Use `shell.start` only for interactive or intentionally long-lived processes; it returns a process handle that is visible to `processes.list` and cancellable with `processes.cancel`.",
    );
    if tool_callable_from_authority(access, "write_stdin") {
        command_execution.push_str(" Send stdin to running shell processes with `shell.write`.");
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

        Ok(StartCommandParams {
            cmd: common.cmd,
            workdir: common.workdir,
            shell_path: common.shell_path,
            login: common.login,
            allow_nonzero_exit: common.allow_nonzero_exit,
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
        context: &lash_core::ToolContext<'_>,
        progress: Option<&ProgressSender>,
        cancel: Option<CancellationToken>,
    ) -> ToolResult {
        if let Some(process_id) = context.async_process_id() {
            return self
                .run_start_command_process(process_id, params, context, progress, cancel)
                .await;
        }
        self.register_start_command_process(params, context).await
    }

    async fn register_start_command_process(
        &self,
        params: &StartCommandParams,
        context: &lash_core::ToolContext<'_>,
    ) -> ToolResult {
        let process_id = context
            .tool_call_id()
            .filter(|id| !id.is_empty())
            .map(str::to_string)
            .unwrap_or_else(|| format!("shell:{}", self.runtime.allocate_handle_id()));
        let args = start_command_process_args(params);
        let call = PreparedToolCall::from_parts(
            process_id.clone(),
            "start_command",
            args,
            None,
            serde_json::Value::Null,
        );
        let registration =
            ProcessRegistration::new(process_id.clone(), ProcessInput::ToolCall { call })
                .with_extra_event_types([shell_signal_event_type()]);
        let descriptor = ProcessHandleDescriptor::new(Some("shell"), Some(params.cmd.clone()));
        match context.processes().start(registration, descriptor).await {
            Ok(_) => {
                let mut handle = lash_core::lashlang_bridge::process_handle_json(&process_id);
                if let Some(object) = handle.as_object_mut() {
                    object.insert("process_id".to_string(), json!(process_id));
                    object.insert("status".to_string(), json!("running"));
                    object.insert("done".to_string(), json!(false));
                    object.insert("running".to_string(), json!(true));
                }
                ToolResult::ok(handle)
            }
            Err(err) => ToolResult::err_fmt(err.to_string()),
        }
    }

    async fn run_start_command_process(
        &self,
        process_id: &str,
        params: &StartCommandParams,
        context: &lash_core::ToolContext<'_>,
        progress: Option<&ProgressSender>,
        cancel: Option<CancellationToken>,
    ) -> ToolResult {
        let started = Instant::now();
        let handle_id = process_id.to_string();

        if let Err(err) = self.runtime.spawn_process(
            handle_id.clone(),
            &params.cmd,
            &params.workdir,
            params.login,
            &params.shell_path,
        ) {
            return ToolResult::err(json!(err));
        }

        let signal_done = CancellationToken::new();
        let signal_forwarder =
            self.spawn_stdin_signal_forwarder(handle_id.clone(), context, signal_done.clone());
        match self
            .runtime
            .wait_until_exit_or_timeout(
                &handle_id,
                None,
                progress,
                params.max_output_tokens,
                WaitBehavior { baseline_len: 0 },
                cancel,
            )
            .await
        {
            Ok(PollOutcome::Running { .. }) => {
                signal_done.cancel();
                let _ = signal_forwarder.await;
                self.runtime.remove_process(&handle_id);
                ToolResult::err_fmt("background shell process returned running without a timeout")
            }
            Ok(PollOutcome::Exited {
                output,
                original_token_count,
                exit_code,
                full_output_path,
            }) => {
                signal_done.cancel();
                let _ = signal_forwarder.await;
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
                signal_done.cancel();
                let _ = signal_forwarder.await;
                self.runtime.remove_process(&handle_id);
                ToolResult::cancelled("tool call cancelled")
            }
            Err(err) => {
                signal_done.cancel();
                let _ = signal_forwarder.await;
                self.runtime.remove_process(&handle_id);
                ToolResult::err(json!(err))
            }
        }
    }

    fn spawn_stdin_signal_forwarder(
        &self,
        process_id: String,
        context: &lash_core::ToolContext<'_>,
        done: CancellationToken,
    ) -> tokio::task::JoinHandle<()> {
        let runtime = self.runtime.clone();
        let events = context.process_events();
        tokio::spawn(async move {
            let mut after_sequence = 0;
            loop {
                let event = tokio::select! {
                    _ = done.cancelled() => break,
                    event = events.wait_event_after("process.signal", after_sequence) => event,
                };
                let Ok(event) = event else {
                    break;
                };
                after_sequence = event.sequence;
                if let Some(chars) = event.payload.get("chars").and_then(|value| value.as_str()) {
                    let _ = runtime.write_stdin(&process_id, chars).await;
                }
                if event
                    .payload
                    .get("close_stdin")
                    .and_then(|value| value.as_bool())
                    .unwrap_or(false)
                {
                    let _ = runtime.close_stdin(&process_id).await;
                }
            }
        })
    }

    async fn write_stdin_call(
        &self,
        args: &serde_json::Value,
        context: &lash_core::ToolContext<'_>,
    ) -> ToolResult {
        let process_id = match parse_process_id(args) {
            Ok(value) => value,
            Err(err) => return err,
        };
        let chars = args
            .get("chars")
            .and_then(|value| value.as_str())
            .unwrap_or("");
        let close_stdin = match parse_optional_bool(args, "close_stdin", false) {
            Ok(value) => value,
            Err(err) => return err,
        };
        match context
            .processes()
            .signal(
                &process_id,
                json!({
                    "chars": chars,
                    "close_stdin": close_stdin,
                }),
            )
            .await
        {
            Ok(event) => ToolResult::ok(json!({
                "process_id": process_id,
                "status": "signalled",
                "sequence": event.sequence,
            })),
            Err(err) => ToolResult::err_fmt(err.to_string()),
        }
    }
}

fn start_command_process_args(params: &StartCommandParams) -> serde_json::Value {
    let mut args = serde_json::Map::new();
    args.insert("cmd".to_string(), json!(params.cmd.clone()));
    args.insert(
        "workdir".to_string(),
        json!(params.workdir.to_string_lossy().to_string()),
    );
    args.insert("shell".to_string(), json!(params.shell_path.clone()));
    args.insert("login".to_string(), json!(params.login));
    args.insert(
        "allow_nonzero_exit".to_string(),
        json!(params.allow_nonzero_exit),
    );
    if let Some(max_output_tokens) = params.max_output_tokens {
        args.insert("max_output_tokens".to_string(), json!(max_output_tokens));
    }
    serde_json::Value::Object(args)
}

fn shell_signal_event_type() -> ProcessEventType {
    ProcessEventType {
        name: "process.signal".to_string(),
        payload_schema: lash_core::LashSchema::any(),
        semantics: ProcessEventSemanticsSpec::default(),
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
        let start_command_description = "Start an interactive or intentionally long-lived command in a PTY as a durable background process. The result is a process handle with `__handle__: \"process\"`, `id`, `process_id`, `status: \"running\"`, `done: false`, and `running: true`; use `processes.list` to see it and `processes.cancel` to stop it. Nonzero exit codes fail the eventual process output by default; pass `allow_nonzero_exit: true` only when nonzero is expected data. Use `shell.exec` for builds, installs, tests, service setup, verification, and other commands that must complete before the next step.";
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
                object_schema(command_common("Shell command to start."), &["cmd"]),
                output_schema.clone(),
            )
            .with_examples(vec![
                r#"await shell.start({ cmd: "python -m http.server 8000" })?"#.into(),
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
                "Send bytes to stdin for a running shell process started by `shell.start`. Use `close_stdin: true` to send EOF. This only acknowledges delivery of the signal; use process lifecycle tools to inspect or cancel the background process.",
                object_schema(
                    json!({
                        "process_id": {
                            "type": "string",
                            "description": "Process id returned by `shell.start`."
                        },
                        "chars": {
                            "type": "string",
                            "default": "",
                            "description": "Bytes to write to stdin; may be empty when only closing stdin."
                        },
                        "close_stdin": {
                            "type": "boolean",
                            "default": false,
                            "description": "Close stdin after writing to send EOF to the process."
                        }
                    }),
                    &["process_id"],
                ),
                output_schema,
            )
            .with_examples(vec![
                r#"await shell.write({ process_id: "call-shell-1", chars: "status\n" })?"#.into(),
                r#"await shell.write({ process_id: "call-shell-1", chars: "", close_stdin: true })?"#.into(),
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
                self.start_command(&params, context, progress, cancel).await
            }
            "write_stdin" => self.write_stdin_call(args, context).await,
            _ => ToolResult::err_fmt(format_args!("Unknown tool: {name}")),
        }
    }
}

fn parse_process_id(args: &serde_json::Value) -> Result<String, ToolResult> {
    require_str(args, "process_id").map(str::to_string)
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
    use lash_core::ProcessRegistry as _;
    use serde_json::json;
    use std::fs;
    use std::sync::Arc;
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

    async fn run_with_context(
        shell: &StaticToolProvider<StandardShell>,
        name: &str,
        args: &serde_json::Value,
        context: &lash_core::ToolContext<'_>,
    ) -> ToolResult {
        shell
            .execute(ToolCall {
                name,
                args,
                context,
                progress: None,
            })
            .await
    }

    fn async_process_context(
        process_id: &str,
        cancel: CancellationToken,
    ) -> lash_core::ToolContext<'static> {
        lash_core::testing::mock_tool_context().with_async_process(process_id, cancel)
    }

    fn async_process_context_with_events(
        process_id: &str,
        registry: Arc<dyn lash_core::ProcessRegistry>,
        cancel: CancellationToken,
    ) -> lash_core::ToolContext<'static> {
        lash_core::testing::mock_tool_context()
            .with_async_process(process_id, cancel)
            .with_process_events_for_testing(process_id, registry)
    }

    #[derive(Clone, Default)]
    struct TestProcessService {
        registry: Arc<lash_core::TestLocalProcessRegistry>,
    }

    impl TestProcessService {
        fn registry(&self) -> Arc<lash_core::TestLocalProcessRegistry> {
            Arc::clone(&self.registry)
        }

        fn owner_scope(
            session_id: &str,
            scope: &lash_core::ProcessOpScope<'_>,
        ) -> lash_core::ProcessScope {
            scope
                .agent_frame_id
                .as_deref()
                .filter(|frame_id| !frame_id.is_empty())
                .map(|frame_id| lash_core::ProcessScope::for_agent_frame(session_id, frame_id))
                .unwrap_or_else(|| lash_core::ProcessScope::new(session_id))
        }
    }

    #[async_trait::async_trait]
    impl lash_core::ProcessService for TestProcessService {
        async fn start(
            &self,
            session_id: &str,
            registration: lash_core::ProcessRegistration,
            options: lash_core::ProcessStartOptions,
            scope: lash_core::ProcessOpScope<'_>,
        ) -> Result<lash_core::ProcessRecord, PluginError> {
            let process_id = registration.id.clone();
            let record = self.registry.register_process(registration).await?;
            if let Some(descriptor) = options.descriptor {
                self.registry
                    .grant_handle(
                        &Self::owner_scope(session_id, &scope),
                        &process_id,
                        descriptor,
                    )
                    .await?;
            }
            Ok(record)
        }

        async fn await_process(
            &self,
            process_id: &str,
            _scope: lash_core::ProcessOpScope<'_>,
        ) -> Result<lash_core::ProcessAwaitOutput, PluginError> {
            self.registry.await_process(process_id).await
        }

        async fn list_visible(
            &self,
            session_id: &str,
            mode: lash_core::ProcessListMode,
            scope: lash_core::ProcessOpScope<'_>,
        ) -> Result<Vec<lash_core::ProcessHandleGrantEntry>, PluginError> {
            let owner_scope = Self::owner_scope(session_id, &scope);
            match mode {
                lash_core::ProcessListMode::Live => {
                    self.registry.list_live_handle_grants(&owner_scope).await
                }
                lash_core::ProcessListMode::All => {
                    self.registry.list_handle_grants(&owner_scope).await
                }
            }
        }

        async fn validate_visible(
            &self,
            session_id: &str,
            process_ids: &[String],
            scope: lash_core::ProcessOpScope<'_>,
        ) -> Result<(), PluginError> {
            let owner_scope = Self::owner_scope(session_id, &scope);
            for process_id in process_ids {
                if !self
                    .registry
                    .has_handle_grant(&owner_scope, process_id)
                    .await?
                {
                    return Err(PluginError::Session(format!(
                        "process handle `{process_id}` is not live or visible in this session"
                    )));
                }
            }
            Ok(())
        }

        async fn cancel(
            &self,
            _session_id: &str,
            process_id: &str,
            _scope: lash_core::ProcessOpScope<'_>,
        ) -> Result<lash_core::ProcessRecord, PluginError> {
            self.registry
                .append_event(
                    process_id,
                    lash_core::ProcessEventAppendRequest::cancel_requested(
                        process_id,
                        Some("requested by test".to_string()),
                    ),
                )
                .await?;
            self.registry
                .get_process(process_id)
                .await
                .ok_or_else(|| PluginError::Session(format!("unknown process `{process_id}`")))
        }

        async fn signal(
            &self,
            _session_id: &str,
            process_id: &str,
            signal_id: String,
            payload: serde_json::Value,
            _scope: lash_core::ProcessOpScope<'_>,
        ) -> Result<lash_core::ProcessEvent, PluginError> {
            self.registry
                .append_event(
                    process_id,
                    lash_core::ProcessEventAppendRequest::new("process.signal", payload)
                        .with_replay_key(format!("process:{process_id}:signal:{signal_id}")),
                )
                .await
                .map(|result| result.event)
        }

        async fn cancel_all(
            &self,
            session_id: &str,
            scope: lash_core::ProcessOpScope<'_>,
        ) -> Result<Vec<lash_core::ProcessRecord>, PluginError> {
            let entries = self
                .list_visible(session_id, lash_core::ProcessListMode::Live, scope.clone())
                .await?;
            let mut cancelled = Vec::new();
            for (grant, _record) in entries {
                cancelled.push(
                    self.cancel(session_id, &grant.process_id, scope.clone())
                        .await?,
                );
            }
            Ok(cancelled)
        }

        async fn transfer(
            &self,
            _from_session_id: &str,
            _to_session_id: &str,
            _process_ids: Vec<String>,
            _scope: lash_core::ProcessOpScope<'_>,
        ) -> Result<(), PluginError> {
            Ok(())
        }

        async fn cancel_unreferenced(
            &self,
            _session_id: &str,
            _keep_process_ids: Vec<String>,
            _scope: lash_core::ProcessOpScope<'_>,
        ) -> Result<Vec<lash_core::ProcessRecord>, PluginError> {
            Ok(Vec::new())
        }
    }

    fn context_with_processes(
        service: Arc<TestProcessService>,
        tool_call_id: &str,
    ) -> lash_core::ToolContext<'static> {
        let host = Arc::new(lash_core::testing::MockSessionManager::default());
        let processes: Arc<dyn lash_core::ProcessService> = service;
        lash_core::ToolContext::__for_testing(
            "test-session".to_string(),
            host,
            processes,
            Arc::new(lash_core::InMemoryAttachmentStore::new()),
            lash_core::DirectCompletionClient::from_fn(|_, _| {
                Err(lash_core::PluginError::Session(
                    "direct completions are unavailable in shell tests".to_string(),
                ))
            }),
            Some(tool_call_id.to_string()),
        )
    }

    async fn register_signal_target(
        registry: &lash_core::TestLocalProcessRegistry,
        process_id: &str,
    ) {
        registry
            .register_process(
                lash_core::ProcessRegistration::new(
                    process_id,
                    lash_core::ProcessInput::External {
                        metadata: serde_json::json!({}),
                    },
                )
                .with_extra_event_types([shell_signal_event_type()]),
            )
            .await
            .expect("register process");
        registry
            .grant_handle(
                &lash_core::ProcessScope::new("test-session"),
                process_id,
                lash_core::ProcessHandleDescriptor::new(Some("shell"), Some("test")),
            )
            .await
            .expect("grant handle");
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
        let ctx = async_process_context("shell-pty", CancellationToken::new());
        let result = run_with_context(
            &shell,
            "start_command",
            &json!({"cmd": "if [ -t 0 ] && [ -t 1 ]; then echo tty; else echo no-tty; exit 1; fi"}),
            &ctx,
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
    async fn start_command_registers_process_handle() {
        let shell = shell_provider(StandardShell::new().with_cwd("/"));
        let service = Arc::new(TestProcessService::default());
        let ctx = context_with_processes(Arc::clone(&service), "shell-call-1");
        let result = run_with_context(
            &shell,
            "start_command",
            &json!({"cmd": "sleep 1; echo done"}),
            &ctx,
        )
        .await;
        assert!(result.is_success(), "{}", result.value_for_projection());
        assert_eq!(result.value_for_projection()["status"], "running");
        assert_eq!(result.value_for_projection()["done"], false);
        assert_eq!(result.value_for_projection()["running"], true);
        assert_eq!(result.value_for_projection()["__handle__"], "process");
        assert_eq!(result.value_for_projection()["id"], "shell-call-1");
        assert_eq!(result.value_for_projection()["process_id"], "shell-call-1");

        let entries = service
            .registry()
            .list_live_handle_grants(&lash_core::ProcessScope::new("test-session"))
            .await
            .expect("list live handles");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].0.process_id, "shell-call-1");
        assert_eq!(entries[0].0.descriptor.kind.as_deref(), Some("shell"));
    }

    #[tokio::test]
    async fn write_stdin_emits_process_signal() {
        let shell = test_shell();
        let service = Arc::new(TestProcessService::default());
        let registry = service.registry();
        register_signal_target(registry.as_ref(), "shell-call-1").await;
        let ctx = context_with_processes(Arc::clone(&service), "write-call-1");

        let result = run_with_context(
            &shell,
            "write_stdin",
            &json!({"process_id": "shell-call-1", "chars": "hello\n", "close_stdin": true}),
            &ctx,
        )
        .await;
        assert!(result.is_success(), "{}", result.value_for_projection());
        assert_eq!(result.value_for_projection()["status"], "signalled");
        assert_eq!(result.value_for_projection()["process_id"], "shell-call-1");

        let events = service
            .registry()
            .events_after("shell-call-1", 0)
            .await
            .expect("events");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, "process.signal");
        assert_eq!(events[0].payload["chars"], "hello\n");
        assert_eq!(events[0].payload["close_stdin"], true);
    }

    #[tokio::test]
    async fn start_command_process_consumes_stdin_signals() {
        let shell = test_shell();
        let registry = Arc::new(lash_core::TestLocalProcessRegistry::default());
        register_signal_target(registry.as_ref(), "shell-worker").await;
        let registry_dyn: Arc<dyn lash_core::ProcessRegistry> = registry.clone();
        let ctx = Arc::new(async_process_context_with_events(
            "shell-worker",
            registry_dyn,
            CancellationToken::new(),
        ));
        let args = Arc::new(json!({
            "cmd": "python3 -u -c 'import sys; line = sys.stdin.readline(); print(\"got:\" + line.strip())'",
            "login": false,
        }));
        let shell = Arc::new(shell);
        let worker = {
            let shell = Arc::clone(&shell);
            let ctx = Arc::clone(&ctx);
            let args = Arc::clone(&args);
            tokio::spawn(async move {
                shell
                    .execute(ToolCall {
                        name: "start_command",
                        args: &args,
                        context: &ctx,
                        progress: None,
                    })
                    .await
            })
        };

        tokio::time::sleep(Duration::from_millis(100)).await;
        registry
            .append_event(
                "shell-worker",
                lash_core::ProcessEventAppendRequest::new(
                    "process.signal",
                    json!({"chars": "hello\n", "close_stdin": false}),
                ),
            )
            .await
            .expect("signal");

        let result = worker.await.expect("worker task");
        assert!(result.is_success(), "{}", result.value_for_projection());
        assert_eq!(result.value_for_projection()["exit_code"], 0);
        assert!(
            result.value_for_projection()["output"]
                .as_str()
                .unwrap()
                .contains("got:hello")
        );
    }

    #[tokio::test]
    async fn start_command_process_can_close_stdin_from_signal() {
        let shell = test_shell();
        let registry = Arc::new(lash_core::TestLocalProcessRegistry::default());
        register_signal_target(registry.as_ref(), "shell-close-stdin").await;
        let registry_dyn: Arc<dyn lash_core::ProcessRegistry> = registry.clone();
        let ctx = Arc::new(async_process_context_with_events(
            "shell-close-stdin",
            registry_dyn,
            CancellationToken::new(),
        ));
        let args = Arc::new(json!({"cmd": "cat", "login": false}));
        let shell = Arc::new(shell);
        let worker = {
            let shell = Arc::clone(&shell);
            let ctx = Arc::clone(&ctx);
            let args = Arc::clone(&args);
            tokio::spawn(async move {
                shell
                    .execute(ToolCall {
                        name: "start_command",
                        args: &args,
                        context: &ctx,
                        progress: None,
                    })
                    .await
            })
        };

        tokio::time::sleep(Duration::from_millis(100)).await;
        registry
            .append_event(
                "shell-close-stdin",
                lash_core::ProcessEventAppendRequest::new(
                    "process.signal",
                    json!({"chars": "hello", "close_stdin": true}),
                ),
            )
            .await
            .expect("signal");

        let result = worker.await.expect("worker task");
        assert!(result.is_success(), "{}", result.value_for_projection());
        assert_eq!(result.value_for_projection()["exit_code"], 0);
        assert!(
            result.value_for_projection()["output"]
                .as_str()
                .unwrap()
                .contains("hello")
        );
    }

    #[tokio::test]
    async fn start_command_process_nonzero_exit_fails_by_default() {
        let shell = test_shell();
        let ctx = async_process_context("shell-exit-7", CancellationToken::new());
        let result = run_with_context(
            &shell,
            "start_command",
            &json!({"cmd": "exit 7", "login": false}),
            &ctx,
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
    async fn start_command_process_reports_full_output_path_when_token_truncated() {
        let shell = test_shell();
        let ctx = async_process_context("shell-token-truncated", CancellationToken::new());
        let result = run_with_context(
            &shell,
            "start_command",
            &json!({"cmd": "python3 -c 'print(\"segment \" * 5000)'", "login": false, "max_output_tokens": 24}),
            &ctx,
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

    #[tokio::test]
    async fn start_command_process_completes_short_lived_commands() {
        let shell = test_shell();
        let cmd = "python3 -u -c 'import sys; line = sys.stdin.readline(); print(\"got:\" + line.strip())'";
        let registry = Arc::new(lash_core::TestLocalProcessRegistry::default());
        register_signal_target(registry.as_ref(), "shell-short").await;
        let registry_dyn: Arc<dyn lash_core::ProcessRegistry> = registry.clone();
        let ctx = Arc::new(async_process_context_with_events(
            "shell-short",
            registry_dyn,
            CancellationToken::new(),
        ));
        let args = Arc::new(json!({"cmd": cmd, "login": false}));
        let shell = Arc::new(shell);
        let worker = {
            let shell = Arc::clone(&shell);
            let ctx = Arc::clone(&ctx);
            let args = Arc::clone(&args);
            tokio::spawn(async move {
                shell
                    .execute(ToolCall {
                        name: "start_command",
                        args: &args,
                        context: &ctx,
                        progress: None,
                    })
                    .await
            })
        };

        tokio::time::sleep(Duration::from_millis(100)).await;
        registry
            .append_event(
                "shell-short",
                lash_core::ProcessEventAppendRequest::new(
                    "process.signal",
                    json!({"chars": "hello\n", "close_stdin": false}),
                ),
            )
            .await
            .expect("signal");

        let result = worker.await.expect("worker task");
        assert!(result.is_success());
        assert!(result.value_for_projection().get("session_id").is_none());
        assert_eq!(result.value_for_projection()["exit_code"], 0);
        assert!(
            result.value_for_projection()["output"]
                .as_str()
                .unwrap()
                .contains("got:hello")
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

    #[test]
    fn shell_definitions_are_compact_and_non_empty() {
        let shell = StandardShell::default();
        let defs = shell.tool_definitions();
        assert_eq!(defs.len(), 3);
        assert!(defs.iter().all(|def| !def.description().is_empty()));
    }

    #[test]
    fn start_command_contract_uses_process_handles() {
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

        assert!(!properties.contains_key("poll_ms"));
        assert!(!properties.contains_key("timeout_ms"));
        assert!(definition.description().contains("processes.list"));
        assert!(definition.description().contains("processes.cancel"));
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
    async fn start_command_cancel_token_kills_running_child() {
        use std::time::Instant;

        let shell = test_shell();
        let token = CancellationToken::new();
        let ctx = async_process_context("shell-cancel", token.clone());
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
        let result = run_with_context(&shell, "start_command", &args, &ctx).await;
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
}
