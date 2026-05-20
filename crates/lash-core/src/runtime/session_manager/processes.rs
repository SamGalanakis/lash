use super::*;
use std::collections::{HashMap, HashSet};
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
        descriptor: Option<crate::ProcessHandleDescriptor>,
        runner: Arc<dyn crate::runtime::effect::ProcessRunner>,
        execution_context: crate::ProcessExecutionContext,
    ) -> Result<crate::ProcessRecord, crate::PluginError> {
        self.start_process_scoped(
            current,
            managed,
            session_id,
            registration,
            descriptor,
            runner,
            execution_context,
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
        registration: crate::ProcessRegistration,
        descriptor: Option<crate::ProcessHandleDescriptor>,
        runner: Arc<dyn crate::runtime::effect::ProcessRunner>,
        execution_context: crate::ProcessExecutionContext,
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
        self.mark_current_process_sync_needed(current, session_id);
        let outcome = self
            .execute_process_effect(
                current,
                Arc::clone(registry),
                crate::ProcessCommand::Start {
                    registration,
                    grant: descriptor.map(|descriptor| crate::ProcessStartGrant {
                        session_id: self.process_scope_key(session_id),
                        descriptor,
                    }),
                    execution_context,
                },
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

    pub(in crate::runtime::session_manager) async fn list_process_handles(
        &self,
        current: &CurrentSessionCapability,
        session_id: &str,
    ) -> Result<Vec<crate::ProcessHandleGrantEntry>, crate::PluginError> {
        self.list_process_handles_scoped(current, session_id, None, None)
            .await
    }

    pub(in crate::runtime::session_manager) async fn list_process_handles_scoped(
        &self,
        current: &CurrentSessionCapability,
        session_id: &str,
        effect_metadata: Option<crate::EffectInvocationMetadata>,
        effect_controller: Option<&dyn crate::RuntimeEffectController>,
    ) -> Result<Vec<crate::ProcessHandleGrantEntry>, crate::PluginError> {
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
                    session_id: self.process_scope_key(session_id),
                },
                None,
                effect_metadata,
                effect_controller,
            )
            .await?;
        match outcome {
            crate::ProcessEffectOutcome::List { entries } => Ok(entries),
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
        if registry.get_process(process_id).await.is_none() {
            return Err(crate::PluginError::Session(format!(
                "unknown process `{process_id}`"
            )));
        }
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
        let tasks = self.list_process_handles(current, session_id).await?;
        let mut cancelled = Vec::new();
        for (grant, record) in tasks {
            if record.is_terminal() {
                continue;
            }
            cancelled.push(
                self.cancel_process(
                    current,
                    managed,
                    Arc::clone(&host),
                    session_id,
                    &grant.process_id,
                )
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
            for (grant, _record) in registry.list_handle_grants(&scope_key).await? {
                visible.insert(grant.process_id);
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
        let Some(registry) = &current.host.process_registry else {
            return Ok(());
        };
        let outcome = self
            .execute_process_effect(
                current,
                Arc::clone(registry),
                crate::ProcessCommand::Transfer {
                    from_session_id: self.process_scope_key(from_session_id),
                    to_session_id: self.process_scope_key(to_session_id),
                    process_ids: handle_ids.to_vec(),
                },
                None,
                None,
                None,
            )
            .await?;
        match outcome {
            crate::ProcessEffectOutcome::Transfer => Ok(()),
            _ => Err(crate::PluginError::Session(
                "process transfer returned the wrong outcome".to_string(),
            )),
        }
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
        let Some(registry) = &current.host.process_registry else {
            return Ok(Vec::new());
        };
        let scope_key = self.process_scope_key(session_id);
        let tasks = registry.list_handle_grants(&scope_key).await?;
        let mut cancelled = Vec::new();
        for (grant, record) in tasks {
            if keep.contains(&grant.process_id) {
                continue;
            }
            registry
                .revoke_handle(&scope_key, &grant.process_id)
                .await?;
            if record.is_terminal() {
                continue;
            }
            if !registry
                .handle_grants_for_process(&grant.process_id)
                .await?
                .is_empty()
            {
                continue;
            }
            cancelled.push(
                self.cancel_process(
                    current,
                    _managed,
                    Arc::clone(&host),
                    session_id,
                    &grant.process_id,
                )
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
        execution_context: crate::ProcessExecutionContext,
        registry: Arc<dyn crate::ProcessRegistry>,
        cancellation: tokio_util::sync::CancellationToken,
    ) -> crate::ProcessAwaitOutput {
        match registration.input.clone() {
            crate::ProcessInput::ToolCall { call } => {
                self.run_process_tool_call(
                    registration,
                    call,
                    execution_context.tool_effect_metadata,
                    cancellation,
                )
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
                line_event,
            } => {
                self.run_command_process(
                    registration,
                    registry,
                    command,
                    cwd,
                    env,
                    timeout_ms,
                    persistent,
                    line_event,
                    execution_context.wake_session_id,
                    cancellation,
                )
                .await
            }
            crate::ProcessInput::LashlangBlock {
                program,
                input,
                tool_bindings,
                timeout_ms,
                display_name: _,
            } => {
                self.run_lashlang_process(
                    registration,
                    registry,
                    program,
                    input,
                    tool_bindings,
                    timeout_ms,
                    execution_context,
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
        tool_effect_metadata: Option<crate::EffectInvocationMetadata>,
        cancellation: tokio_util::sync::CancellationToken,
    ) -> crate::ProcessAwaitOutput {
        let result = self
            .execute_process_tool_call(registration, call, tool_effect_metadata, cancellation)
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
        tool_effect_metadata: Option<crate::EffectInvocationMetadata>,
        cancellation: tokio_util::sync::CancellationToken,
    ) -> Result<crate::ToolCallOutput, crate::PluginError> {
        let (event_tx, mut event_rx) = tokio::sync::mpsc::channel::<crate::SessionEvent>(64);
        let event_drain = tokio::spawn(async move { while event_rx.recv().await.is_some() {} });
        let host = Arc::new(self.clone()) as Arc<dyn crate::plugin::RuntimeSessionHost>;
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

    #[allow(clippy::too_many_arguments)]
    async fn run_lashlang_process(
        &self,
        registration: crate::ProcessRegistration,
        registry: Arc<dyn crate::ProcessRegistry>,
        program: serde_json::Value,
        input: serde_json::Map<String, serde_json::Value>,
        tool_bindings: Vec<crate::LashlangProcessToolBinding>,
        timeout_ms: Option<u64>,
        execution_context: crate::ProcessExecutionContext,
        cancellation: tokio_util::sync::CancellationToken,
    ) -> crate::ProcessAwaitOutput {
        let program = match serde_json::from_value::<lashlang::Program>(program) {
            Ok(program) => program,
            Err(err) => {
                return process_lashlang_failure(
                    "process_block_decode_failed",
                    format!("failed to decode lashlang process block: {err}"),
                    None,
                );
            }
        };
        let mut globals = lashlang::Record::with_capacity(input.len());
        for (name, value) in input {
            globals.insert(name, lashlang::from_json(value));
        }
        let mut state = lashlang::State::from_snapshot(lashlang::Snapshot { globals });

        let (event_tx, mut event_rx) = tokio::sync::mpsc::channel::<crate::SessionEvent>(64);
        let event_drain = tokio::spawn(async move { while event_rx.recv().await.is_some() {} });
        let runtime_host = Arc::new(self.clone()) as Arc<dyn crate::plugin::RuntimeSessionHost>;
        let direct_completions = crate::DirectCompletionClient::runtime(
            Arc::new(self.clone()),
            crate::runtime::RuntimeEffectControllerHandle::shared(Arc::clone(
                &self.current.host.core.effect_controller,
            )),
            execution_context
                .tool_effect_metadata
                .as_ref()
                .and_then(|metadata| metadata.turn_id.clone()),
            self.current.turn_lease.clone(),
        );
        let dispatch = match self.current.plugins.tool_surface(
            &self.current.session_id,
            self.current.policy.execution_mode.clone(),
        ) {
            Ok(surface) => crate::tool_dispatch::ToolDispatchContext {
                plugins: Arc::clone(&self.current.plugins),
                tools: self.current.plugins.tools(),
                surface,
                host: Arc::clone(&runtime_host),
                effect_controller: crate::runtime::RuntimeEffectControllerHandle::shared(
                    Arc::clone(&self.current.host.core.effect_controller),
                ),
                direct_completions: direct_completions.clone(),
                tool_effect_metadata: None,
                session_id: self.current.session_id.clone(),
                event_tx,
                turn_injection_bridge: crate::TurnInjectionBridge::new(),
                attachment_store: Arc::clone(&self.current.host.core.attachment_store),
                turn_context: crate::TurnContext::default(),
            },
            Err(err) => {
                drop(runtime_host);
                let _ = event_drain.await;
                return process_lashlang_failure(
                    "process_block_tool_surface_failed",
                    err.to_string(),
                    None,
                );
            }
        };
        let mut ctx = crate::ModeExecutionContext::new(
            self.current.session_id.clone(),
            self.current.policy.execution_mode.clone(),
            Arc::new(dispatch),
            Arc::clone(&self.current.host.core.attachment_store),
            Arc::new(crate::ChronologicalProjection::default()),
            None,
            crate::TurnContext::default(),
        )
        .with_cancellation_token(cancellation.clone());
        if let Some(metadata) = execution_context.tool_effect_metadata.clone() {
            ctx = ctx.with_effect_metadata(metadata);
        }

        let host = LashlangBlockProcessHost {
            manager: self.clone(),
            ctx,
            registry: Arc::clone(&registry),
            process_id: registration.id.clone(),
            tool_bindings: tool_bindings
                .into_iter()
                .map(|binding| (binding.name, binding.tool_id))
                .collect(),
            wake_session_id: execution_context.wake_session_id,
        };
        let env = lashlang::ExecutionEnvironment::new(&host).process();

        let output = if let Some(timeout_ms) = timeout_ms {
            let timeout = tokio::time::Duration::from_millis(timeout_ms);
            tokio::select! {
                _ = cancellation.cancelled() => process_lashlang_cancelled("lashlang process block was cancelled"),
                result = tokio::time::timeout(timeout, lashlang::execute(&program, &mut state, &env)) => {
                    match result {
                        Ok(result) => process_lashlang_execution_result(result),
                        Err(_) => process_lashlang_failure(
                            "process_block_timeout",
                            format!("lashlang process block timed out after {timeout_ms}ms"),
                            None,
                        ),
                    }
                }
            }
        } else {
            tokio::select! {
                _ = cancellation.cancelled() => process_lashlang_cancelled("lashlang process block was cancelled"),
                result = lashlang::execute(&program, &mut state, &env) => {
                    process_lashlang_execution_result(result)
                }
            }
        };
        drop(env);
        drop(host);
        let _ = event_drain.await;
        output
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
                let state = process_terminal_state_for_turn(&turn);
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

struct LashlangBlockProcessHost<'run> {
    manager: RuntimeSessionManager,
    ctx: crate::ModeExecutionContext<'run>,
    registry: Arc<dyn crate::ProcessRegistry>,
    process_id: String,
    tool_bindings: HashMap<String, crate::ToolId>,
    wake_session_id: Option<String>,
}

impl LashlangBlockProcessHost<'_> {
    fn captured_manifest(
        &self,
        name: &str,
    ) -> Result<crate::ToolManifest, lashlang::ExecutionHostError> {
        let tool_id = self.tool_bindings.get(name).ok_or_else(|| {
            lashlang::ExecutionHostError::new(format!(
                "tool `{name}` was not captured for this lashlang process"
            ))
        })?;
        self.ctx
            .callable_tool_manifest_by_id(tool_id)
            .ok_or_else(|| {
                lashlang::ExecutionHostError::new(format!(
                    "captured tool `{name}` with id `{}` is unavailable in this session",
                    tool_id.as_str()
                ))
            })
    }

    fn tool_payload(
        &self,
        args: &lashlang::Record,
    ) -> Result<serde_json::Value, lashlang::ExecutionHostError> {
        let mut payload = crate::lashlang_bridge::lashlang_value_to_json(
            &lashlang::Value::Record(std::sync::Arc::new(args.clone())),
        )?;
        if let Some(obj) = payload.as_object_mut() {
            obj.entry("__session_id__".to_string())
                .or_insert_with(|| serde_json::Value::String(self.ctx.session_id().to_string()));
        }
        Ok(payload)
    }
}

impl LashlangBlockProcessHost<'_> {
    async fn call(
        &self,
        name: String,
        args: lashlang::Record,
    ) -> Result<lashlang::Value, lashlang::ExecutionHostError> {
        let manifest = self.captured_manifest(&name)?;
        let reply = self
            .ctx
            .call_tool(
                uuid::Uuid::new_v4().to_string(),
                manifest.name.clone(),
                self.tool_payload(&args)?,
                0,
            )
            .await;
        mode_reply_to_lashlang_value(reply)
    }

    async fn start_call(
        &self,
        name: String,
        args: lashlang::Record,
    ) -> Result<lashlang::Value, lashlang::ExecutionHostError> {
        let manifest = self.captured_manifest(&name)?;
        let reply = self
            .ctx
            .start_tool_call(
                uuid::Uuid::new_v4().to_string(),
                manifest.name.clone(),
                self.tool_payload(&args)?,
            )
            .await;
        mode_reply_to_lashlang_value(reply)
    }

    async fn await_handle(
        &self,
        handle: lashlang::Value,
    ) -> Result<lashlang::Value, lashlang::ExecutionHostError> {
        let reply = self
            .ctx
            .await_tool_handle(
                uuid::Uuid::new_v4().to_string(),
                crate::lashlang_bridge::lashlang_value_to_json(&handle)?,
            )
            .await;
        mode_reply_to_lashlang_value(reply)
    }

    async fn cancel_handle(
        &self,
        handle: lashlang::Value,
    ) -> Result<lashlang::Value, lashlang::ExecutionHostError> {
        let reply = self
            .ctx
            .cancel_tool_handle(
                uuid::Uuid::new_v4().to_string(),
                crate::lashlang_bridge::lashlang_value_to_json(&handle)?,
            )
            .await;
        mode_reply_to_lashlang_value(reply)
    }

    async fn start_process(
        &self,
        start: lashlang::ProcessBlockStart,
    ) -> Result<lashlang::Value, lashlang::ExecutionHostError> {
        let (registration, label) = self
            .ctx
            .prepare_lashlang_process_start(start)
            .map_err(lashlang::ExecutionHostError::new)?;
        let reply = self.ctx.start_lashlang_process(registration, label).await;
        mode_reply_to_lashlang_value(reply)
    }

    async fn process_event(
        &self,
        event: lashlang::ProcessBlockEvent,
    ) -> Result<(), lashlang::ExecutionHostError> {
        let event_type = match event.kind {
            lashlang::ProcessBlockEventKind::Yield => "process.yield",
            lashlang::ProcessBlockEventKind::Wake => "process.wake",
        };
        let event = self
            .registry
            .append_event(
                &self.process_id,
                event_type.to_string(),
                crate::lashlang_bridge::process_event_payload(&event.value)?,
            )
            .await
            .map_err(|err| lashlang::ExecutionHostError::new(err.to_string()))?;
        if let Some(wake) = event.semantics.wake.as_ref()
            && self
                .manager
                .managed
                .inject_turn_input(
                    self.wake_session_id
                        .as_deref()
                        .unwrap_or_else(|| self.ctx.session_id()),
                    crate::InjectedTurnInput {
                        id: Some(format!(
                            "process:{}:wake:{}",
                            self.process_id, event.sequence
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
            self.registry
                .ack_wake(&self.process_id, event.sequence)
                .await
                .map_err(|err| lashlang::ExecutionHostError::new(err.to_string()))?;
        }
        Ok(())
    }
}

impl lashlang::ExecutionHost for LashlangBlockProcessHost<'_> {
    async fn perform(
        &self,
        op: lashlang::AbilityOp,
    ) -> Result<lashlang::AbilityResult, lashlang::ExecutionHostError> {
        match op {
            lashlang::AbilityOp::CallTool { name, args } => self
                .call(name, args)
                .await
                .map(lashlang::AbilityResult::Value),
            lashlang::AbilityOp::StartToolCall { name, args } => self
                .start_call(name, args)
                .await
                .map(lashlang::AbilityResult::Value),
            lashlang::AbilityOp::Await(handle) => self
                .await_handle(handle)
                .await
                .map(lashlang::AbilityResult::Value),
            lashlang::AbilityOp::Cancel(handle) => self
                .cancel_handle(handle)
                .await
                .map(lashlang::AbilityResult::Value),
            lashlang::AbilityOp::StartProcess(start) => self
                .start_process(start)
                .await
                .map(lashlang::AbilityResult::Value),
            lashlang::AbilityOp::ProcessEvent(event) => {
                self.process_event(event).await?;
                Ok(lashlang::AbilityResult::Unit)
            }
            lashlang::AbilityOp::Print(_) => Err(lashlang::ExecutionHostError::new(
                "`print` is not available inside lashlang process blocks",
            )),
        }
    }

    async fn yield_now(&self) {
        tokio::task::yield_now().await;
    }
}

fn mode_reply_to_lashlang_value(
    reply: crate::ModeToolReply,
) -> Result<lashlang::Value, lashlang::ExecutionHostError> {
    crate::lashlang_bridge::mode_tool_reply_to_lashlang_value(reply)
}

fn process_lashlang_execution_result(
    result: Result<lashlang::ExecutionOutcome, lashlang::RuntimeError>,
) -> crate::ProcessAwaitOutput {
    match result {
        Ok(lashlang::ExecutionOutcome::Finished(value)) => crate::ProcessAwaitOutput::Success {
            value: crate::lashlang_bridge::lashlang_value_to_json(&value)
                .unwrap_or(serde_json::Value::Null),
            control: None,
        },
        Ok(lashlang::ExecutionOutcome::Failed(value)) => process_lashlang_failure(
            "process_block_failed",
            value.to_string(),
            Some(
                crate::lashlang_bridge::lashlang_value_to_json(&value)
                    .unwrap_or(serde_json::Value::Null),
            ),
        ),
        Ok(lashlang::ExecutionOutcome::Continued) => crate::ProcessAwaitOutput::Success {
            value: serde_json::Value::Null,
            control: None,
        },
        Err(err) => process_lashlang_failure("process_block_runtime_error", err.to_string(), None),
    }
}

fn process_lashlang_failure(
    code: &str,
    message: String,
    raw: Option<serde_json::Value>,
) -> crate::ProcessAwaitOutput {
    crate::ProcessAwaitOutput::Failure {
        class: crate::ToolFailureClass::Execution,
        code: code.to_string(),
        message,
        raw,
        control: None,
    }
}

fn process_lashlang_cancelled(message: impl Into<String>) -> crate::ProcessAwaitOutput {
    crate::ProcessAwaitOutput::Cancelled {
        message: message.into(),
        raw: None,
        control: None,
    }
}

fn process_terminal_state_for_turn(turn: &crate::AssembledTurn) -> crate::ProcessTerminalState {
    match &turn.outcome {
        crate::TurnOutcome::Finished(_) | crate::TurnOutcome::Handoff { .. } => {
            crate::ProcessTerminalState::Completed
        }
        crate::TurnOutcome::Stopped(crate::TurnStop::Cancelled) => {
            crate::ProcessTerminalState::Cancelled
        }
        crate::TurnOutcome::Stopped(_) => crate::ProcessTerminalState::Failed,
    }
}

fn process_turn_summary(
    turn: &crate::AssembledTurn,
    state: crate::ProcessTerminalState,
) -> Option<String> {
    if state != crate::ProcessTerminalState::Failed {
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
    state: crate::ProcessTerminalState,
) -> crate::ToolCallOutput {
    if state == crate::ProcessTerminalState::Cancelled {
        return crate::ToolCallOutput::cancelled(crate::ToolCancellation::runtime(
            "background session turn was cancelled",
        ));
    }
    if state == crate::ProcessTerminalState::Failed {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::tests::helpers::{
        mock_provider, runtime_with_plugins, runtime_with_plugins_and_tools,
    };

    fn probe_event_type() -> crate::ProcessEventType {
        crate::ProcessEventType {
            name: "probe.event".to_string(),
            payload_schema: crate::LashSchema::any(),
            semantics: crate::ProcessEventSemanticsSpec::default(),
        }
    }

    fn external_registration(process_id: &str) -> crate::ProcessRegistration {
        crate::ProcessRegistration::new(
            process_id,
            crate::ProcessInput::External {
                metadata: serde_json::json!({ "process_id": process_id }),
            },
        )
        .with_extra_event_types([probe_event_type()])
    }

    fn lashlang_block_registration(
        process_id: &str,
        program: lashlang::Program,
        input: serde_json::Map<String, serde_json::Value>,
    ) -> crate::ProcessRegistration {
        crate::ProcessRegistration::new(
            process_id,
            crate::ProcessInput::LashlangBlock {
                program: serde_json::to_value(program).expect("serialize lashlang program"),
                input,
                tool_bindings: vec![crate::LashlangProcessToolBinding {
                    name: "alias".to_string(),
                    tool_id: crate::ToolId::new("tool:process_echo"),
                }],
                timeout_ms: None,
                display_name: Some("block".to_string()),
            },
        )
        .with_extra_event_types(crate::lashlang_process_event_types())
    }

    struct ProcessEchoTool;

    fn process_echo_tool_definition() -> crate::ToolDefinition {
        crate::ToolDefinition::raw(
            "tool:process_echo",
            "process_echo",
            "Echo process input.",
            serde_json::json!({ "type": "object", "additionalProperties": true }),
            serde_json::json!({ "type": "object", "additionalProperties": true }),
        )
    }

    #[async_trait::async_trait]
    impl crate::ToolProvider for ProcessEchoTool {
        fn tool_manifests(&self) -> Vec<crate::ToolManifest> {
            vec![process_echo_tool_definition().manifest()]
        }

        fn resolve_contract(&self, name: &str) -> Option<Arc<crate::ToolContract>> {
            (name == "process_echo").then(|| Arc::new(process_echo_tool_definition().contract()))
        }

        async fn execute(&self, call: crate::ToolCall<'_>) -> crate::ToolResult {
            let value = call
                .args
                .get("value")
                .and_then(|value| value.as_str())
                .unwrap_or_default();
            crate::ToolResult::ok(serde_json::json!({ "payload": format!("raw:{value}") }))
        }
    }

    async fn register_open_process(registry: &Arc<dyn crate::ProcessRegistry>, process_id: &str) {
        registry
            .register_process(external_registration(process_id))
            .await
            .expect("register process");
        registry
            .append_event(
                process_id,
                "probe.event".to_string(),
                serde_json::json!({ "marker": process_id }),
            )
            .await
            .expect("append probe event");
    }

    async fn grant_handle(
        registry: &Arc<dyn crate::ProcessRegistry>,
        scope_key: &str,
        process_id: &str,
    ) {
        registry
            .grant_handle(
                scope_key,
                process_id,
                crate::ProcessHandleDescriptor::new(Some("test"), Some(process_id)),
            )
            .await
            .expect("grant handle");
    }

    #[tokio::test]
    async fn lashlang_block_process_runs_with_input_events_wake_and_captured_tool_id() {
        let runtime = runtime_with_plugins_and_tools(
            Vec::new(),
            Arc::new(ProcessEchoTool),
            mock_provider(Vec::new()),
        )
        .await;
        let manager = RuntimeSessionManager::new(&runtime, true, None, None)
            .expect("runtime session manager");
        let wake_target = manager
            .managed
            .create_session(
                &manager.current,
                &manager.usage,
                crate::SessionCreateRequest {
                    session_id: Some("wake-target".to_string()),
                    relation: crate::SessionRelation::Root,
                    start: crate::SessionStartPoint::Empty,
                    policy: None,
                    plugin_mode: crate::SessionPluginMode::InheritCurrent,
                    initial_nodes: Vec::new(),
                    first_turn_input: None,
                    tool_access: crate::SessionToolAccess::default(),
                    subagent: None,
                    context_surface: crate::SessionContextSurface::default(),
                    mode_extras: crate::ModeExtras::default(),
                    usage_source: None,
                },
            )
            .await
            .expect("wake target session");
        let mut input = serde_json::Map::new();
        input.insert("root".to_string(), serde_json::json!("seed"));
        let program = lashlang::Program::block(vec![
            lashlang::Expr::Yield(Box::new(lashlang::Expr::Variable("root".into()))),
            lashlang::Expr::Assign {
                target: lashlang::AssignTarget::variable("called".into()),
                expr: Box::new(lashlang::Expr::ResultUnwrap(Box::new(
                    lashlang::Expr::ToolCall {
                        mode: lashlang::ToolCallMode::Call,
                        call: lashlang::CallExpr {
                            name: "alias".into(),
                            args: vec![("value".into(), lashlang::Expr::Variable("root".into()))],
                        },
                    },
                ))),
            },
            lashlang::Expr::Wake(Box::new(lashlang::Expr::Field {
                target: Box::new(lashlang::Expr::Variable("called".into())),
                field: "payload".into(),
            })),
            lashlang::Expr::Assign {
                target: lashlang::AssignTarget::variable("handle".into()),
                expr: Box::new(lashlang::Expr::ToolCall {
                    mode: lashlang::ToolCallMode::Start,
                    call: lashlang::CallExpr {
                        name: "alias".into(),
                        args: vec![("value".into(), lashlang::Expr::String("nested".into()))],
                    },
                }),
            },
            lashlang::Expr::Assign {
                target: lashlang::AssignTarget::variable("nested".into()),
                expr: Box::new(lashlang::Expr::ResultUnwrap(Box::new(
                    lashlang::Expr::Await(Box::new(lashlang::Expr::Variable("handle".into()))),
                ))),
            },
            lashlang::Expr::Finish(Some(Box::new(lashlang::Expr::Record(vec![
                (
                    "first".into(),
                    lashlang::Expr::Field {
                        target: Box::new(lashlang::Expr::Variable("called".into())),
                        field: "payload".into(),
                    },
                ),
                (
                    "nested".into(),
                    lashlang::Expr::Field {
                        target: Box::new(lashlang::Expr::Variable("nested".into())),
                        field: "payload".into(),
                    },
                ),
            ])))),
        ]);
        let registration = lashlang_block_registration("block-1", program, input);

        manager
            .processes
            .start_process(
                &manager.current,
                &manager.managed,
                "root",
                registration,
                Some(crate::ProcessHandleDescriptor::new(
                    Some("lashlang"),
                    Some("block"),
                )),
                Arc::new(manager.clone()),
                crate::ProcessExecutionContext::default()
                    .with_wake_session_id(wake_target.session_id.clone()),
            )
            .await
            .expect("start process");
        let output = manager
            .processes
            .await_process(&manager.current, "block-1")
            .await
            .expect("await process");

        let crate::ProcessAwaitOutput::Success { value, .. } = output else {
            panic!("process should succeed");
        };
        assert_eq!(
            value,
            serde_json::json!({ "first": "raw:seed", "nested": "raw:nested" })
        );
        let registry = manager
            .current
            .host
            .process_registry
            .as_ref()
            .expect("process registry");
        let events = registry.events_after("block-1", 0).await.expect("events");
        assert!(
            events
                .iter()
                .any(|event| event.event_type == "process.yield"
                    && event.payload["value"] == serde_json::json!("seed"))
        );
        assert!(events.iter().any(|event| event.event_type == "process.wake"
            && event.payload["text"] == serde_json::json!("raw:seed")));
        assert!(
            registry
                .wake_events_after("block-1", 0)
                .await
                .expect("wake events")
                .is_empty()
        );
        let child_runtime = manager
            .managed
            .registry
            .lock()
            .await
            .get(&wake_target.session_id)
            .cloned()
            .expect("wake target runtime");
        let injected = child_runtime
            .runtime
            .lock()
            .await
            .session
            .as_ref()
            .expect("wake target session")
            .turn_input_injection_bridge()
            .drain()
            .expect("injected input");
        assert_eq!(injected.len(), 1);
        assert_eq!(injected[0].message.content, "raw:seed");
    }

    #[tokio::test]
    async fn lashlang_block_process_failure_retains_raw_value() {
        let runtime = runtime_with_plugins_and_tools(
            Vec::new(),
            Arc::new(ProcessEchoTool),
            mock_provider(Vec::new()),
        )
        .await;
        let manager = RuntimeSessionManager::new(&runtime, true, None, None)
            .expect("runtime session manager");
        let program = lashlang::Program::block(vec![lashlang::Expr::Fail(Box::new(
            lashlang::Expr::Record(vec![(
                "reason".into(),
                lashlang::Expr::String("bad".into()),
            )]),
        ))]);
        let registration =
            lashlang_block_registration("block-fail", program, serde_json::Map::new());

        manager
            .processes
            .start_process(
                &manager.current,
                &manager.managed,
                "root",
                registration,
                Some(crate::ProcessHandleDescriptor::new(
                    Some("lashlang"),
                    Some("fail"),
                )),
                Arc::new(manager.clone()),
                crate::ProcessExecutionContext::default(),
            )
            .await
            .expect("start process");
        let output = manager
            .processes
            .await_process(&manager.current, "block-fail")
            .await
            .expect("await process");

        let crate::ProcessAwaitOutput::Failure {
            message, raw, code, ..
        } = output
        else {
            panic!("process should fail");
        };
        assert_eq!(code, "process_block_failed");
        assert_eq!(raw, Some(serde_json::json!({ "reason": "bad" })));
        assert!(message.contains("reason"));
        assert!(message.contains("bad"));
    }

    #[tokio::test]
    async fn lashlang_block_process_has_no_parent_capture_or_tool_name_fallback() {
        let runtime = runtime_with_plugins_and_tools(
            Vec::new(),
            Arc::new(ProcessEchoTool),
            mock_provider(Vec::new()),
        )
        .await;
        let manager = RuntimeSessionManager::new(&runtime, true, None, None)
            .expect("runtime session manager");
        let no_parent_program = lashlang::Program::block(vec![lashlang::Expr::Finish(Some(
            Box::new(lashlang::Expr::Variable("parent".into())),
        ))]);
        let no_parent = lashlang_block_registration(
            "block-no-parent",
            no_parent_program,
            serde_json::Map::new(),
        );
        manager
            .processes
            .start_process(
                &manager.current,
                &manager.managed,
                "root",
                no_parent,
                Some(crate::ProcessHandleDescriptor::new(
                    Some("lashlang"),
                    Some("no-parent"),
                )),
                Arc::new(manager.clone()),
                crate::ProcessExecutionContext::default(),
            )
            .await
            .expect("start no-parent process");
        let output = manager
            .processes
            .await_process(&manager.current, "block-no-parent")
            .await
            .expect("await no-parent process");
        let crate::ProcessAwaitOutput::Failure { message, .. } = output else {
            panic!("process should fail");
        };
        assert!(message.contains("unknown name `parent`"), "{message}");

        let fallback_program =
            lashlang::Program::block(vec![lashlang::Expr::Finish(Some(Box::new(
                lashlang::Expr::ResultUnwrap(Box::new(lashlang::Expr::ToolCall {
                    mode: lashlang::ToolCallMode::Call,
                    call: lashlang::CallExpr {
                        name: "process_echo".into(),
                        args: vec![("value".into(), lashlang::Expr::String("x".into()))],
                    },
                })),
            )))]);
        let fallback_registration = crate::ProcessRegistration::new(
            "block-no-fallback",
            crate::ProcessInput::LashlangBlock {
                program: serde_json::to_value(fallback_program).expect("serialize program"),
                input: serde_json::Map::new(),
                tool_bindings: Vec::new(),
                timeout_ms: None,
                display_name: None,
            },
        )
        .with_extra_event_types(crate::lashlang_process_event_types());
        manager
            .processes
            .start_process(
                &manager.current,
                &manager.managed,
                "root",
                fallback_registration,
                Some(crate::ProcessHandleDescriptor::new(
                    Some("lashlang"),
                    Some("no-fallback"),
                )),
                Arc::new(manager.clone()),
                crate::ProcessExecutionContext::default(),
            )
            .await
            .expect("start no-fallback process");
        let output = manager
            .processes
            .await_process(&manager.current, "block-no-fallback")
            .await
            .expect("await no-fallback process");
        let crate::ProcessAwaitOutput::Failure { message, .. } = output else {
            panic!("process should fail");
        };
        assert!(
            message.contains("tool `process_echo` was not captured"),
            "{message}"
        );
    }

    async fn process_event_projection(
        registry: &Arc<dyn crate::ProcessRegistry>,
        process_ids: &[&str],
    ) -> Vec<(String, Vec<(u64, String, serde_json::Value)>)> {
        let mut projection = Vec::new();
        for process_id in process_ids {
            let events = registry
                .events_after(process_id, 0)
                .await
                .expect("process events")
                .into_iter()
                .map(|event| (event.sequence, event.event_type, event.payload))
                .collect();
            projection.push(((*process_id).to_string(), events));
        }
        projection
    }

    #[tokio::test]
    async fn cancel_unreferenced_process_handles_revokes_current_grants_and_cancels_only_unowned() {
        let runtime = runtime_with_plugins(Vec::new(), mock_provider(Vec::new())).await;
        let manager = RuntimeSessionManager::new(&runtime, true, None, None)
            .expect("runtime session manager");
        let registry = runtime
            .host
            .process_registry
            .as_ref()
            .expect("process registry")
            .clone();
        let current_scope = manager.processes.process_scope_key("root");
        let other_scope = manager.processes.process_scope_key("other");
        let process_ids = ["keep", "sole", "shared"];

        for process_id in process_ids {
            register_open_process(&registry, process_id).await;
            grant_handle(&registry, &current_scope, process_id).await;
        }
        grant_handle(&registry, &other_scope, "shared").await;

        let events_before = process_event_projection(&registry, &process_ids).await;
        let cancelled = manager
            .processes
            .cancel_unreferenced_process_handles(
                &manager.current,
                &manager.managed,
                Arc::new(manager.clone()),
                "root",
                &["keep".to_string()],
            )
            .await
            .expect("cancel unreferenced handles");

        assert_eq!(
            cancelled
                .iter()
                .map(|record| record.id.as_str())
                .collect::<Vec<_>>(),
            vec!["sole"]
        );
        assert_eq!(
            registry
                .list_handle_grants(&current_scope)
                .await
                .expect("current grants")
                .into_iter()
                .map(|(grant, _)| grant.process_id)
                .collect::<Vec<_>>(),
            vec!["keep".to_string()]
        );
        assert!(
            registry
                .handle_grants_for_process("sole")
                .await
                .expect("sole grants")
                .is_empty()
        );
        assert_eq!(
            registry
                .handle_grants_for_process("shared")
                .await
                .expect("shared grants")
                .into_iter()
                .map(|grant| grant.session_id)
                .collect::<Vec<_>>(),
            vec![other_scope]
        );
        assert_eq!(
            process_event_projection(&registry, &process_ids).await,
            events_before
        );
    }
}
