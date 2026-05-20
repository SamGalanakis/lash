use super::*;
use std::collections::HashSet;
use std::sync::atomic::Ordering;

impl ProcessCapability {
    pub(in crate::runtime::session_manager) fn process_scope_key(
        &self,
        session_id: &str,
    ) -> String {
        format!("{}:{session_id}", self.runtime_scope_id)
    }

    pub(in crate::runtime::session_manager) async fn start_process(
        &self,
        current: &CurrentSessionCapability,
        managed: &ManagedSessionCapability,
        session_id: &str,
        registration: crate::ProcessRegistration,
        runner: Arc<dyn crate::runtime::effect::ProcessRunner>,
    ) -> Result<crate::ProcessRecord, crate::PluginError> {
        self.start_process_scoped(
            current,
            managed,
            session_id,
            registration,
            runner,
            None,
            None,
        )
        .await
    }

    pub(in crate::runtime::session_manager) async fn start_process_scoped(
        &self,
        current: &CurrentSessionCapability,
        managed: &ManagedSessionCapability,
        session_id: &str,
        mut registration: crate::ProcessRegistration,
        runner: Arc<dyn crate::runtime::effect::ProcessRunner>,
        effect_metadata: Option<crate::EffectInvocationMetadata>,
        effect_controller: Option<&dyn crate::RuntimeEffectController>,
    ) -> Result<crate::ProcessRecord, crate::PluginError> {
        self.ensure_known_process_session(current, managed, session_id)
            .await?;
        let Some(registry) = &current.host.process_registry else {
            return Err(crate::PluginError::Session(
                "processes are unavailable in this runtime".to_string(),
            ));
        };
        registration.scope = crate::ProcessScope {
            session_id: self.process_scope_key(session_id),
        };
        self.mark_current_process_sync_needed(current, session_id);
        let outcome = self
            .execute_process_effect(
                current,
                Arc::clone(registry),
                crate::ProcessCommand::Start { registration },
                Some(runner),
                effect_metadata,
                effect_controller,
            )
            .await?;
        match outcome {
            crate::ProcessEffectOutcome::Start { record } => Ok(record),
            _ => Err(crate::PluginError::Session(
                "process start returned the wrong outcome".to_string(),
            )),
        }
    }

    pub(in crate::runtime::session_manager) async fn await_process(
        &self,
        current: &CurrentSessionCapability,
        process_id: &str,
    ) -> Result<crate::ProcessAwaitOutput, crate::PluginError> {
        self.await_process_scoped(current, process_id, None, None)
            .await
    }

    pub(in crate::runtime::session_manager) async fn await_process_scoped(
        &self,
        current: &CurrentSessionCapability,
        process_id: &str,
        effect_metadata: Option<crate::EffectInvocationMetadata>,
        effect_controller: Option<&dyn crate::RuntimeEffectController>,
    ) -> Result<crate::ProcessAwaitOutput, crate::PluginError> {
        let Some(registry) = &current.host.process_registry else {
            return Err(crate::PluginError::Session(
                "process registry is unavailable in this runtime".to_string(),
            ));
        };
        let outcome = self
            .execute_process_effect(
                current,
                Arc::clone(registry),
                crate::ProcessCommand::Await {
                    process_id: process_id.to_string(),
                },
                None,
                effect_metadata,
                effect_controller,
            )
            .await?;
        match outcome {
            crate::ProcessEffectOutcome::Await { output } => Ok(output),
            _ => Err(crate::PluginError::Session(
                "process await returned the wrong outcome".to_string(),
            )),
        }
    }

    pub(in crate::runtime::session_manager) async fn list_processes(
        &self,
        current: &CurrentSessionCapability,
        session_id: &str,
    ) -> Result<Vec<crate::ProcessRecord>, crate::PluginError> {
        self.list_processes_scoped(current, session_id, None, None)
            .await
    }

    pub(in crate::runtime::session_manager) async fn list_processes_scoped(
        &self,
        current: &CurrentSessionCapability,
        session_id: &str,
        effect_metadata: Option<crate::EffectInvocationMetadata>,
        effect_controller: Option<&dyn crate::RuntimeEffectController>,
    ) -> Result<Vec<crate::ProcessRecord>, crate::PluginError> {
        let Some(registry) = &current.host.process_registry else {
            return Err(crate::PluginError::Session(
                "process registry is unavailable in this runtime".to_string(),
            ));
        };
        let outcome = self
            .execute_process_effect(
                current,
                Arc::clone(registry),
                crate::ProcessCommand::List {
                    filter: crate::ProcessFilter {
                        session_id: Some(self.process_scope_key(session_id)),
                        producer: None,
                        tags: Vec::new(),
                        handle_visible: None,
                        include_terminal: true,
                    },
                },
                None,
                effect_metadata,
                effect_controller,
            )
            .await?;
        match outcome {
            crate::ProcessEffectOutcome::List { records } => Ok(records),
            _ => Err(crate::PluginError::Session(
                "process list returned the wrong outcome".to_string(),
            )),
        }
    }

    pub(in crate::runtime::session_manager) async fn cancel_process(
        &self,
        current: &CurrentSessionCapability,
        managed: &ManagedSessionCapability,
        host: Arc<dyn crate::plugin::RuntimeSessionHost>,
        session_id: &str,
        process_id: &str,
    ) -> Result<crate::ProcessRecord, crate::PluginError> {
        self.cancel_process_scoped(current, managed, host, session_id, process_id, None, None)
            .await
    }

    pub(in crate::runtime::session_manager) async fn cancel_process_scoped(
        &self,
        current: &CurrentSessionCapability,
        managed: &ManagedSessionCapability,
        host: Arc<dyn crate::plugin::RuntimeSessionHost>,
        session_id: &str,
        process_id: &str,
        effect_metadata: Option<crate::EffectInvocationMetadata>,
        effect_controller: Option<&dyn crate::RuntimeEffectController>,
    ) -> Result<crate::ProcessRecord, crate::PluginError> {
        let Some(registry) = &current.host.process_registry else {
            return Err(crate::PluginError::Session(
                "process registry is unavailable in this runtime".to_string(),
            ));
        };
        let Some(_status) = registry.get(process_id).await else {
            return Err(crate::PluginError::Session(format!(
                "unknown process `{process_id}`"
            )));
        };
        let _ = (managed, host, session_id);
        let outcome = self
            .execute_process_effect(
                current,
                Arc::clone(registry),
                crate::ProcessCommand::Cancel {
                    process_id: process_id.to_string(),
                    reason: Some("requested by host".to_string()),
                },
                None,
                effect_metadata,
                effect_controller,
            )
            .await?;
        match outcome {
            crate::ProcessEffectOutcome::Cancel { record } => Ok(record),
            _ => Err(crate::PluginError::Session(
                "process cancel returned the wrong outcome".to_string(),
            )),
        }
    }

    pub(in crate::runtime::session_manager) async fn cancel_all_processes(
        &self,
        current: &CurrentSessionCapability,
        managed: &ManagedSessionCapability,
        host: Arc<dyn crate::plugin::RuntimeSessionHost>,
        session_id: &str,
    ) -> Result<Vec<crate::ProcessRecord>, crate::PluginError> {
        let tasks = self.list_processes(current, session_id).await?;
        let mut cancelled = Vec::new();
        for task in tasks {
            if task.state.is_terminal() {
                continue;
            }
            cancelled.push(
                self.cancel_process(current, managed, Arc::clone(&host), session_id, &task.id)
                    .await?,
            );
        }
        Ok(cancelled)
    }

    pub(in crate::runtime::session_manager) async fn validate_process_handles_visible(
        &self,
        current: &CurrentSessionCapability,
        _managed: &ManagedSessionCapability,
        session_id: &str,
        handle_ids: &[String],
    ) -> Result<(), crate::PluginError> {
        if handle_ids.is_empty() {
            return Ok(());
        }
        let requested = handle_ids.iter().cloned().collect::<HashSet<_>>();
        let mut visible = HashSet::new();
        if let Some(registry) = &current.host.process_registry {
            let scope_key = self.process_scope_key(session_id);
            for task in registry
                .list(crate::ProcessFilter {
                    session_id: Some(scope_key),
                    producer: None,
                    tags: Vec::new(),
                    handle_visible: Some(true),
                    include_terminal: false,
                })
                .await
            {
                if !task.state.is_terminal() {
                    visible.insert(task.id);
                }
            }
        }
        if let Some(missing) = requested.iter().find(|id| !visible.contains(*id)) {
            return Err(crate::PluginError::Session(format!(
                "process handle `{missing}` is not live or visible in this session"
            )));
        }
        Ok(())
    }

    pub(in crate::runtime::session_manager) async fn transfer_process_handles(
        &self,
        current: &CurrentSessionCapability,
        _managed: &ManagedSessionCapability,
        from_session_id: &str,
        to_session_id: &str,
        handle_ids: &[String],
    ) -> Result<(), crate::PluginError> {
        if handle_ids.is_empty() {
            return Ok(());
        }
        if let Some(registry) = &current.host.process_registry {
            let to_scope = crate::ProcessScope {
                session_id: self.process_scope_key(to_session_id),
            };
            for handle_id in handle_ids {
                let task = registry.get(handle_id).await.ok_or_else(|| {
                    crate::PluginError::Session(format!(
                        "process handle `{handle_id}` is not live or visible in this session"
                    ))
                })?;
                if task.scope.session_id != self.process_scope_key(from_session_id) {
                    return Err(crate::PluginError::Session(format!(
                        "process handle `{handle_id}` is not owned by session `{from_session_id}`"
                    )));
                }
                registry.transfer(handle_id, to_scope.clone()).await?;
            }
        }
        Ok(())
    }

    pub(in crate::runtime::session_manager) async fn cancel_unreferenced_process_handles(
        &self,
        current: &CurrentSessionCapability,
        _managed: &ManagedSessionCapability,
        host: Arc<dyn crate::plugin::RuntimeSessionHost>,
        session_id: &str,
        keep_handle_ids: &[String],
    ) -> Result<Vec<crate::ProcessRecord>, crate::PluginError> {
        let keep = keep_handle_ids.iter().cloned().collect::<HashSet<_>>();
        let tasks = self.list_processes(current, session_id).await?;
        let mut cancelled = Vec::new();
        for task in tasks {
            if task.state.is_terminal() || keep.contains(&task.id) {
                continue;
            }
            cancelled.push(
                self.cancel_process(current, _managed, Arc::clone(&host), session_id, &task.id)
                    .await?,
            );
        }
        Ok(cancelled)
    }

    async fn ensure_known_process_session(
        &self,
        current: &CurrentSessionCapability,
        managed: &ManagedSessionCapability,
        session_id: &str,
    ) -> Result<(), crate::PluginError> {
        if session_id == current.session_id
            || managed.registry.lock().await.contains_key(session_id)
        {
            return Ok(());
        }
        Err(crate::PluginError::Session(format!(
            "unknown session `{session_id}`"
        )))
    }

    fn mark_current_process_sync_needed(
        &self,
        current: &CurrentSessionCapability,
        session_id: &str,
    ) {
        if session_id == current.session_id {
            self.sync_needed.store(true, Ordering::Release);
        }
    }

    async fn execute_process_effect(
        &self,
        current: &CurrentSessionCapability,
        registry: Arc<dyn crate::ProcessRegistry>,
        command: crate::ProcessCommand,
        runner: Option<Arc<dyn crate::runtime::effect::ProcessRunner>>,
        parent_metadata: Option<crate::EffectInvocationMetadata>,
        effect_controller: Option<&dyn crate::RuntimeEffectController>,
    ) -> Result<crate::ProcessEffectOutcome, crate::PluginError> {
        let effect_id = command.effect_id();
        let metadata = self.process_effect_metadata(current, &effect_id, parent_metadata)?;
        let envelope = crate::RuntimeEffectEnvelope::new(
            metadata,
            crate::RuntimeEffectCommand::Process { command },
        );
        let controller = effect_controller.unwrap_or(current.host.core.effect_controller.as_ref());
        let outcome = crate::runtime::effect::execute_effect_with_journal(
            current.store.as_ref().map(|store| store.as_ref()),
            current.turn_lease.as_ref(),
            controller,
            envelope,
            crate::RuntimeEffectLocalExecutor::process(registry, runner),
        )
        .await?;
        outcome.into_process().map_err(crate::PluginError::from)
    }

    fn process_effect_metadata(
        &self,
        current: &CurrentSessionCapability,
        effect_id: &str,
        parent_metadata: Option<crate::EffectInvocationMetadata>,
    ) -> Result<crate::EffectInvocationMetadata, crate::PluginError> {
        if let Some(parent) = parent_metadata {
            let Some(turn_id) = parent.turn_id.clone() else {
                return Err(crate::PluginError::Session(format!(
                    "process effect `{effect_id}` requires active turn metadata"
                )));
            };
            return Ok(crate::EffectInvocationMetadata {
                session_id: current.session_id.clone(),
                origin: parent.origin,
                turn_id: Some(turn_id),
                turn_index: parent.turn_index,
                mode_iteration: parent.mode_iteration,
                effect_id: effect_id.to_string(),
                effect_kind: crate::RuntimeEffectKind::Process,
                idempotency_key: format!("{}:{effect_id}", parent.idempotency_key),
                turn_checkpoint_hash: parent.turn_checkpoint_hash,
            });
        }
        Ok(crate::EffectInvocationMetadata {
            session_id: current.session_id.clone(),
            origin: crate::EffectOrigin::Turn,
            turn_id: None,
            turn_index: None,
            mode_iteration: None,
            effect_id: effect_id.to_string(),
            effect_kind: crate::RuntimeEffectKind::Process,
            idempotency_key: format!("{}:{effect_id}", current.session_id),
            turn_checkpoint_hash: None,
        })
    }
}

#[async_trait::async_trait]
impl crate::runtime::effect::ProcessRunner for RuntimeSessionManager {
    async fn run_process(
        &self,
        registration: crate::ProcessRegistration,
        registry: Arc<dyn crate::ProcessRegistry>,
        cancellation: tokio_util::sync::CancellationToken,
    ) -> crate::ProcessAwaitOutput {
        match registration.input.clone() {
            crate::ProcessInput::ToolCall { call } => {
                self.run_process_tool_call(registration, call, cancellation)
                    .await
            }
            crate::ProcessInput::SessionTurn {
                create_request,
                turn_input,
                ..
            } => {
                self.run_process_session_turn(
                    registration,
                    *create_request,
                    *turn_input,
                    cancellation,
                )
                .await
            }
            crate::ProcessInput::Command {
                command,
                cwd,
                env,
                timeout_ms,
                persistent,
            } => {
                self.run_command_process(
                    registration,
                    registry,
                    command,
                    cwd,
                    env,
                    timeout_ms,
                    persistent,
                    cancellation,
                )
                .await
            }
            crate::ProcessInput::External { metadata } => crate::ProcessAwaitOutput::Success {
                value: serde_json::json!({ "metadata": metadata }),
                control: None,
            },
        }
    }
}

impl RuntimeSessionManager {
    async fn run_process_tool_call(
        &self,
        registration: crate::ProcessRegistration,
        call: crate::PreparedToolCall,
        cancellation: tokio_util::sync::CancellationToken,
    ) -> crate::ProcessAwaitOutput {
        let result = self
            .execute_process_tool_call(registration, call, cancellation)
            .await;
        match result {
            Ok(output) => crate::ProcessAwaitOutput::from_tool_output(output),
            Err(err) => crate::ProcessAwaitOutput::from_tool_output(
                crate::ToolCallOutput::failure(crate::ToolFailure::runtime(
                    crate::ToolFailureClass::Internal,
                    "process_tool_failed",
                    err.to_string(),
                )),
            ),
        }
    }

    async fn execute_process_tool_call(
        &self,
        registration: crate::ProcessRegistration,
        call: crate::PreparedToolCall,
        cancellation: tokio_util::sync::CancellationToken,
    ) -> Result<crate::ToolCallOutput, crate::PluginError> {
        let (event_tx, mut event_rx) = tokio::sync::mpsc::channel::<crate::SessionEvent>(64);
        let event_drain = tokio::spawn(async move { while event_rx.recv().await.is_some() {} });
        let host = Arc::new(self.clone()) as Arc<dyn crate::plugin::RuntimeSessionHost>;
        let tool_effect_metadata = registration
            .metadata
            .get("tool_effect_metadata")
            .cloned()
            .and_then(|value| serde_json::from_value(value).ok());
        let direct_completions = crate::DirectCompletionClient::runtime(
            Arc::new(self.clone()),
            crate::runtime::RuntimeEffectControllerHandle::shared(Arc::clone(
                &self.current.host.core.effect_controller,
            )),
            tool_effect_metadata
                .as_ref()
                .and_then(|metadata: &crate::EffectInvocationMetadata| metadata.turn_id.clone()),
            self.current.turn_lease.clone(),
        );
        let dispatch = crate::tool_dispatch::ToolDispatchContext {
            plugins: Arc::clone(&self.current.plugins),
            tools: self.current.plugins.tools(),
            surface: self.current.plugins.tool_surface(
                &self.current.session_id,
                self.current.policy.execution_mode.clone(),
            )?,
            host: Arc::clone(&host),
            effect_controller: crate::runtime::RuntimeEffectControllerHandle::shared(Arc::clone(
                &self.current.host.core.effect_controller,
            )),
            direct_completions: direct_completions.clone(),
            tool_effect_metadata: None,
            session_id: self.current.session_id.clone(),
            event_tx,
            turn_injection_bridge: crate::TurnInjectionBridge::new(),
            attachment_store: Arc::clone(&self.current.host.core.attachment_store),
            turn_context: crate::TurnContext::default(),
        };
        let tool_context = crate::ToolContext::new(
            self.current.session_id.clone(),
            host,
            crate::TurnContext::default(),
            Arc::clone(&self.current.host.core.attachment_store),
            direct_completions,
            dispatch.effect_controller.clone_scoped(),
            Some(call.call_id.clone()),
        )
        .with_async_process(registration.id.clone(), cancellation)
        .with_tool_effect_metadata(tool_effect_metadata);
        let outcome = crate::tool_dispatch::dispatch_prepared_tool_call_with_execution_context(
            &dispatch,
            call,
            None,
            tool_context,
        )
        .await;
        drop(dispatch);
        let _ = event_drain.await;
        Ok(outcome.record.output)
    }

    async fn run_process_session_turn(
        &self,
        registration: crate::ProcessRegistration,
        create_request: crate::SessionCreateRequest,
        turn_input: crate::TurnInput,
        cancellation: tokio_util::sync::CancellationToken,
    ) -> crate::ProcessAwaitOutput {
        let child = match self
            .managed
            .create_session(&self.current, &self.usage, create_request)
            .await
        {
            Ok(child) => child,
            Err(err) => {
                return crate::ProcessAwaitOutput::from_tool_output(
                    crate::ToolCallOutput::failure(crate::ToolFailure::tool(
                        crate::ToolFailureClass::Execution,
                        "process_session_create_failed",
                        err.to_string(),
                    )),
                );
            }
        };
        let child_session_id = child.session_id.clone();
        let turn =
            self.managed
                .start_turn(&self.current, &self.usage, &child_session_id, turn_input);
        tokio::pin!(turn);
        let outcome = tokio::select! {
            _ = cancellation.cancelled() => {
                let _ = self
                    .managed
                    .close_session(&self.current, &self.usage, &child_session_id)
                    .await;
                return crate::ProcessAwaitOutput::from_tool_output(
                    crate::ToolCallOutput::cancelled(
                        crate::ToolCancellation::runtime("background session turn was cancelled"),
                    ),
                );
            }
            outcome = &mut turn => outcome,
        };
        let _ = self
            .managed
            .close_session(&self.current, &self.usage, &child_session_id)
            .await;
        match outcome {
            Ok(turn) => {
                let state = process_state_for_turn(&turn);
                crate::ProcessAwaitOutput::from_tool_output(output_from_process_turn(
                    &registration,
                    &child_session_id,
                    turn,
                    state,
                ))
            }
            Err(err) => crate::ProcessAwaitOutput::from_tool_output(
                crate::ToolCallOutput::failure(crate::ToolFailure::tool(
                    crate::ToolFailureClass::Execution,
                    "process_session_turn_failed",
                    err.to_string(),
                )),
            ),
        }
    }
}

fn process_state_for_turn(turn: &crate::AssembledTurn) -> crate::ProcessState {
    match &turn.outcome {
        crate::TurnOutcome::Finished(_) | crate::TurnOutcome::Handoff { .. } => {
            crate::ProcessState::Completed
        }
        crate::TurnOutcome::Stopped(crate::TurnStop::Cancelled) => crate::ProcessState::Cancelled,
        crate::TurnOutcome::Stopped(_) => crate::ProcessState::Failed,
    }
}

fn process_turn_summary(turn: &crate::AssembledTurn, state: crate::ProcessState) -> Option<String> {
    if state != crate::ProcessState::Failed {
        return None;
    }
    match &turn.outcome {
        crate::TurnOutcome::Stopped(
            crate::TurnStop::SubmittedError { value } | crate::TurnStop::ToolError { value, .. },
        ) => value
            .get("reason")
            .and_then(serde_json::Value::as_str)
            .map(ToOwned::to_owned),
        _ => Some("background session turn failed".to_string()),
    }
}

fn output_from_process_turn(
    registration: &crate::ProcessRegistration,
    child_session_id: &str,
    turn: crate::AssembledTurn,
    state: crate::ProcessState,
) -> crate::ToolCallOutput {
    if state == crate::ProcessState::Cancelled {
        return crate::ToolCallOutput::cancelled(crate::ToolCancellation::runtime(
            "background session turn was cancelled",
        ));
    }
    if state == crate::ProcessState::Failed {
        return crate::ToolCallOutput::failure(crate::ToolFailure::tool(
            crate::ToolFailureClass::Execution,
            "process_session_turn_failed",
            process_turn_summary(&turn, state)
                .unwrap_or_else(|| "background session turn failed".to_string()),
        ));
    }
    crate::ToolCallOutput::success(serde_json::json!({
        "process_id": registration.id,
        "child_session_id": child_session_id,
        "turn": turn,
    }))
}

impl RuntimeSessionManager {
    #[allow(clippy::too_many_arguments)]
    async fn run_command_process(
        &self,
        registration: crate::ProcessRegistration,
        registry: Arc<dyn crate::ProcessRegistry>,
        command_text: String,
        cwd: Option<String>,
        env: std::collections::BTreeMap<String, String>,
        timeout_ms: u64,
        persistent: bool,
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
        line: String,
        from_stdout: bool,
    ) -> Result<(), crate::PluginError> {
        if registration.producer != "monitor"
            && !registration.tags.iter().any(|tag| tag == "monitor")
        {
            return Ok(());
        }
        let message = line.trim().to_string();
        if message.is_empty() {
            return Ok(());
        }
        let monitor_id = registration
            .metadata
            .get("monitor_id")
            .and_then(serde_json::Value::as_str)
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| {
                registration
                    .id
                    .strip_prefix("monitor:")
                    .unwrap_or(registration.id.as_str())
                    .to_string()
            });
        let wake_policy = registration
            .metadata
            .get("wake_policy")
            .and_then(serde_json::Value::as_str);
        let mut payload = serde_json::json!({
            "line": message,
            "stream": if from_stdout { "stdout" } else { "stderr" },
            "timestamp": chrono::Utc::now().to_rfc3339(),
        });
        if from_stdout && wake_policy != Some("notify") {
            payload["wake_input"] = serde_json::json!(format!(
                "Monitor event \"{monitor_id}\": {}",
                payload["line"].as_str().unwrap_or_default()
            ));
        }
        let event = registry
            .append_event(&registration.id, "monitor.line".to_string(), payload)
            .await?;
        if let Some(wake) = event.semantics.wake.as_ref() {
            if self
                .managed
                .inject_turn_input(
                    &registration.scope.session_id,
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
