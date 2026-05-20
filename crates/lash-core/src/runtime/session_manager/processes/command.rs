use super::*;
use std::sync::Arc;

impl RuntimeSessionManager {
    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime::session_manager::processes) async fn run_command_process(
        &self,
        registration: crate::ProcessRegistration,
        registry: Arc<dyn crate::ProcessRegistry>,
        command_text: String,
        cwd: Option<String>,
        env: std::collections::BTreeMap<String, String>,
        timeout_ms: u64,
        persistent: bool,
        line_event: Option<crate::ProcessCommandLineEventSpec>,
        wake_session_id: Option<String>,
        cancellation: tokio_util::sync::CancellationToken,
    ) -> crate::ProcessAwaitOutput {
        let mut command = tokio::process::Command::new("bash");
        command.arg("-lc").arg(&command_text);
        if let Some(cwd) = cwd.as_ref() {
            command.current_dir(cwd);
        }
        if !env.is_empty() {
            command.envs(env.iter());
        }
        command.kill_on_drop(true);
        command.stdout(std::process::Stdio::piped());
        command.stderr(std::process::Stdio::piped());
        configure_command_process(&mut command);
        let mut child = match command.spawn() {
            Ok(child) => child,
            Err(err) => {
                return process_command_failure("process_command_start_failed", err.to_string());
            }
        };
        let runtime_pid = child.id();
        let stdout = match child.stdout.take() {
            Some(stdout) => stdout,
            None => return process_command_failure("process_command_stdout", "stdout unavailable"),
        };
        let stderr = match child.stderr.take() {
            Some(stderr) => stderr,
            None => return process_command_failure("process_command_stderr", "stderr unavailable"),
        };
        let mut stdout_lines = tokio::io::AsyncBufReadExt::lines(tokio::io::BufReader::new(stdout));
        let mut stderr_lines = tokio::io::AsyncBufReadExt::lines(tokio::io::BufReader::new(stderr));
        let mut stdout_done = false;
        let mut stderr_done = false;
        let deadline = (!persistent)
            .then(|| tokio::time::Instant::now() + std::time::Duration::from_millis(timeout_ms));
        let mut timeout = deadline.map(|deadline| Box::pin(tokio::time::sleep_until(deadline)));
        let mut timed_out = false;
        let mut cancelled = false;

        while !stdout_done || !stderr_done {
            tokio::select! {
                _ = timeout.as_mut().unwrap(), if timeout.is_some() => {
                    timed_out = true;
                    break;
                }
                _ = cancellation.cancelled() => {
                    cancelled = true;
                    break;
                }
                line = stdout_lines.next_line(), if !stdout_done => {
                    match line {
                        Ok(Some(line)) => {
                            if let Err(err) = self.append_command_line_event(
                                &registration,
                                Arc::clone(&registry),
                                line_event.as_ref(),
                                wake_session_id.as_deref(),
                                line,
                                true,
                            ).await {
                                return process_command_failure("process_command_event_failed", err.to_string());
                            }
                        }
                        Ok(None) => stdout_done = true,
                        Err(err) => return process_command_failure("process_command_stdout_read_failed", err.to_string()),
                    }
                }
                line = stderr_lines.next_line(), if !stderr_done => {
                    match line {
                        Ok(Some(line)) => {
                            if let Err(err) = self.append_command_line_event(
                                &registration,
                                Arc::clone(&registry),
                                line_event.as_ref(),
                                wake_session_id.as_deref(),
                                line,
                                false,
                            ).await {
                                return process_command_failure("process_command_event_failed", err.to_string());
                            }
                        }
                        Ok(None) => stderr_done = true,
                        Err(err) => return process_command_failure("process_command_stderr_read_failed", err.to_string()),
                    }
                }
            }
        }

        if timed_out || cancelled {
            let _ = terminate_command_process_tree(runtime_pid).await;
        }

        let exit = if let Some(deadline) = deadline.filter(|_| !timed_out && !cancelled) {
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            match tokio::time::timeout(remaining, child.wait()).await {
                Ok(result) => result,
                Err(_) => {
                    timed_out = true;
                    let _ = terminate_command_process_tree(runtime_pid).await;
                    child.wait().await
                }
            }
        } else {
            child.wait().await
        };

        if cancelled {
            return crate::ProcessAwaitOutput::from_tool_output(crate::ToolCallOutput::cancelled(
                crate::ToolCancellation::runtime("process command was cancelled"),
            ));
        }
        if timed_out {
            return process_command_failure(
                "process_command_timeout",
                format!("process command timed out after {timeout_ms}ms"),
            );
        }
        crate::ProcessAwaitOutput::from_tool_output(match exit {
            Ok(status) if status.success() => crate::ToolCallOutput::success(serde_json::json!({
                "exit_status": status.code(),
            })),
            Ok(status) => crate::ToolCallOutput::failure(crate::ToolFailure::tool(
                crate::ToolFailureClass::Execution,
                "process_command_failed",
                format!(
                    "process command exited with status {}",
                    status.code().unwrap_or_default()
                ),
            )),
            Err(err) => crate::ToolCallOutput::failure(crate::ToolFailure::tool(
                crate::ToolFailureClass::Execution,
                "process_command_wait_failed",
                err.to_string(),
            )),
        })
    }

    async fn append_command_line_event(
        &self,
        registration: &crate::ProcessRegistration,
        registry: Arc<dyn crate::ProcessRegistry>,
        line_event: Option<&crate::ProcessCommandLineEventSpec>,
        wake_session_id: Option<&str>,
        line: String,
        from_stdout: bool,
    ) -> Result<(), crate::PluginError> {
        let Some(line_event) = line_event else {
            return Ok(());
        };
        let message = line.trim().to_string();
        if message.is_empty() {
            return Ok(());
        }
        let mut payload = serde_json::json!({
            "line": message,
            "stream": if from_stdout { "stdout" } else { "stderr" },
            "timestamp": chrono::Utc::now().to_rfc3339(),
        });
        if from_stdout && let Some(template) = line_event.wake_input_template.as_ref() {
            payload["wake_input"] = serde_json::json!(
                template
                    .replace("{process_id}", &registration.id)
                    .replace("{line}", payload["line"].as_str().unwrap_or_default())
            );
        }
        let event = registry
            .append_event(&registration.id, line_event.event_type.clone(), payload)
            .await?;
        if let Some(wake) = event.semantics.wake.as_ref() {
            if self
                .managed
                .inject_turn_input(
                    wake_session_id.unwrap_or(&self.current.session_id),
                    crate::InjectedTurnInput {
                        id: Some(format!(
                            "process:{}:wake:{}",
                            registration.id, event.sequence
                        )),
                        message: crate::PluginMessage::text(
                            crate::MessageRole::System,
                            wake.input.clone(),
                        ),
                    },
                )
                .await
                .is_ok()
            {
                registry.ack_wake(&registration.id, event.sequence).await?;
            }
        }
        Ok(())
    }
}

fn process_command_failure(code: &str, message: impl Into<String>) -> crate::ProcessAwaitOutput {
    crate::ProcessAwaitOutput::from_tool_output(crate::ToolCallOutput::failure(
        crate::ToolFailure::tool(
            crate::ToolFailureClass::Execution,
            code.to_string(),
            message.into(),
        ),
    ))
}

#[cfg(unix)]
fn configure_command_process(command: &mut tokio::process::Command) {
    unsafe {
        command.pre_exec(|| {
            if libc::setsid() == -1 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
}

#[cfg(not(unix))]
fn configure_command_process(_command: &mut tokio::process::Command) {}

#[cfg(unix)]
async fn terminate_command_process_tree(
    runtime_pid: Option<u32>,
) -> Result<(), crate::PluginError> {
    let Some(pid) = runtime_pid else {
        return Ok(());
    };
    let pgid = -(pid as i32);
    send_process_group_signal(pgid, libc::SIGTERM)?;
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    if process_group_exists(pgid) {
        send_process_group_signal(pgid, libc::SIGKILL)?;
    }
    Ok(())
}

#[cfg(not(unix))]
async fn terminate_command_process_tree(
    _runtime_pid: Option<u32>,
) -> Result<(), crate::PluginError> {
    Ok(())
}

#[cfg(unix)]
fn process_group_exists(pgid: i32) -> bool {
    let rc = unsafe { libc::kill(pgid, 0) };
    if rc == 0 {
        return true;
    }
    let err = std::io::Error::last_os_error();
    !matches!(err.raw_os_error(), Some(libc::ESRCH))
}

#[cfg(unix)]
fn send_process_group_signal(pgid: i32, signal: libc::c_int) -> Result<(), crate::PluginError> {
    let rc = unsafe { libc::kill(pgid, signal) };
    if rc == 0 {
        return Ok(());
    }
    let err = std::io::Error::last_os_error();
    if matches!(err.raw_os_error(), Some(libc::ESRCH)) {
        return Ok(());
    }
    Err(crate::PluginError::Session(format!(
        "failed to signal process group {pgid}: {err}"
    )))
}
