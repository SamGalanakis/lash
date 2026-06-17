#[cfg(test)]
mod tests {
    use super::*;
    use crate::shell::output::{MAX_OUTPUT, SPILL_OUTPUT_THRESHOLD, clean_terminal_output};
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

        fn session_scope(
            session_id: &str,
            scope: &lash_core::ProcessOpScope<'_>,
        ) -> lash_core::SessionScope {
            scope
                .agent_frame_id()
                .filter(|frame_id| !frame_id.is_empty())
                .map(|frame_id| lash_core::SessionScope::for_agent_frame(session_id, frame_id))
                .unwrap_or_else(|| lash_core::SessionScope::new(session_id))
        }
    }

    #[async_trait::async_trait]
    impl lash_core::ProcessService for TestProcessService {
        async fn start_from_request(
            &self,
            session_id: &str,
            request: lash_core::ProcessStartRequest,
            scope: lash_core::ProcessOpScope<'_>,
        ) -> Result<lash_core::ProcessHandleSummary, PluginError> {
            let env_ref = request
                .env_spec
                .as_ref()
                .map(lash_core::ProcessExecutionEnvSpec::stable_ref)
                .transpose()
                .map_err(|err| {
                    PluginError::Session(format!("failed to hash test process env: {err}"))
                })?;
            let descriptor = request
                .grant
                .as_ref()
                .map(|grant| grant.descriptor.clone())
                .unwrap_or_default();
            let registration = request.into_registration(env_ref);
            let record = self
                .start(
                    session_id,
                    registration,
                    lash_core::ProcessStartOptions::new().with_descriptor(descriptor.clone()),
                    scope,
                )
                .await?;
            let definition = record.input.as_ref().definition();
            Ok(lash_core::ProcessHandleSummary::new(
                record.id,
                descriptor,
                lash_core::ProcessLifecycleStatus::from(record.status),
            )
            .with_definition(definition))
        }

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
                        &Self::session_scope(session_id, &scope),
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
        ) -> Result<Vec<lash_core::runtime::ProcessHandleGrantEntry>, PluginError> {
            let session_scope = Self::session_scope(session_id, &scope);
            match mode {
                lash_core::ProcessListMode::Live => {
                    self.registry.list_live_handle_grants(&session_scope).await
                }
                lash_core::ProcessListMode::All => {
                    self.registry.list_handle_grants(&session_scope).await
                }
            }
        }

        async fn validate_visible(
            &self,
            session_id: &str,
            process_ids: &[String],
            scope: lash_core::ProcessOpScope<'_>,
        ) -> Result<(), PluginError> {
            let session_scope = Self::session_scope(session_id, &scope);
            for process_id in process_ids {
                if !self
                    .registry
                    .has_handle_grant(&session_scope, process_id)
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
            signal_name: String,
            signal_id: String,
            payload: serde_json::Value,
            _scope: lash_core::ProcessOpScope<'_>,
        ) -> Result<lash_core::ProcessEvent, PluginError> {
            let event_type = lash_core::process_signal_event_type(&signal_name)?;
            self.registry
                .append_event(
                    process_id,
                    lash_core::ProcessEventAppendRequest::new(event_type, payload).with_replay_key(
                        format!("process:{process_id}:signal.{signal_name}:{signal_id}"),
                    ),
                )
                .await
                .map(|result| result.event)
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
            host.clone(),
            host.clone(),
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
                    lash_core::ProcessProvenance::host(),
                )
                .with_extra_event_types([shell_signal_event_type()]),
            )
            .await
            .expect("register process");
        registry
            .grant_handle(
                &lash_core::SessionScope::new("test-session"),
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
            .list_live_handle_grants(&lash_core::SessionScope::new("test-session"))
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
        assert_eq!(events[0].event_type, SHELL_STDIN_SIGNAL_EVENT);
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
                    SHELL_STDIN_SIGNAL_EVENT,
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
                    SHELL_STDIN_SIGNAL_EVENT,
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
                    SHELL_STDIN_SIGNAL_EVENT,
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
    fn shell_definitions_document_distinct_result_shapes() {
        let shell = StandardShell::default();
        let defs = shell.tool_definitions();
        let exec = defs
            .iter()
            .find(|definition| definition.name() == "exec_command")
            .expect("exec_command definition");
        let start = defs
            .iter()
            .find(|definition| definition.name() == "start_command")
            .expect("start_command definition");
        let write = defs
            .iter()
            .find(|definition| definition.name() == "write_stdin")
            .expect("write_stdin definition");

        assert!(
            exec.compact_contract()
                .render_signature()
                .contains("exit_code")
        );
        assert!(
            start
                .compact_contract()
                .render_signature()
                .contains("__handle__")
        );
        assert!(
            write
                .compact_contract()
                .render_signature()
                .contains("sequence")
        );
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
