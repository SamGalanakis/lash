use super::*;
use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::Ordering;

struct ProcessCommandRunner<'scope> {
    current: &'scope CurrentSessionCapability,
    registry: Arc<dyn crate::ProcessRegistry>,
    parent_invocation: Option<crate::RuntimeInvocation>,
    effect_controller: &'scope dyn crate::RuntimeEffectController,
}

impl<'scope> ProcessCommandRunner<'scope> {
    fn new(
        current: &'scope CurrentSessionCapability,
        scope: &'scope crate::ProcessOpScope<'scope>,
        unavailable_message: &'static str,
    ) -> Result<Self, crate::PluginError> {
        let Some(registry) = current.host.process_registry.as_ref() else {
            return Err(crate::PluginError::Session(unavailable_message.to_string()));
        };
        let effect_controller = scope.controller();
        Ok(Self {
            current,
            registry: Arc::clone(registry),
            parent_invocation: scope.parent_invocation.clone(),
            effect_controller,
        })
    }

    fn registry(&self) -> &Arc<dyn crate::ProcessRegistry> {
        &self.registry
    }

    async fn start(
        &self,
        registration: crate::ProcessRegistration,
        grant: Option<crate::ProcessStartGrant>,
        execution_context: crate::ProcessExecutionContext,
    ) -> Result<crate::ProcessRecord, crate::PluginError> {
        match self
            .run(crate::ProcessCommand::Start {
                registration,
                grant,
                execution_context: Box::new(execution_context),
            })
            .await?
        {
            crate::ProcessEffectOutcome::Start { record } => Ok(record),
            _ => Err(wrong_process_outcome("start")),
        }
    }

    async fn await_process(
        &self,
        process_id: &str,
    ) -> Result<crate::ProcessAwaitOutput, crate::PluginError> {
        match self
            .run(crate::ProcessCommand::Await {
                process_id: process_id.to_string(),
            })
            .await?
        {
            crate::ProcessEffectOutcome::Await { output } => Ok(output),
            _ => Err(wrong_process_outcome("await")),
        }
    }

    async fn list(
        &self,
        session_scope: crate::SessionScope,
        mode: crate::ProcessListMode,
    ) -> Result<Vec<crate::runtime::ProcessHandleGrantEntry>, crate::PluginError> {
        match self
            .run(crate::ProcessCommand::List {
                session_scope,
                mode,
            })
            .await?
        {
            crate::ProcessEffectOutcome::List { entries } => Ok(entries),
            _ => Err(wrong_process_outcome("list")),
        }
    }

    async fn cancel(&self, process_id: &str) -> Result<crate::ProcessRecord, crate::PluginError> {
        match self
            .run(crate::ProcessCommand::Cancel {
                process_id: process_id.to_string(),
                reason: Some("requested by host".to_string()),
            })
            .await?
        {
            crate::ProcessEffectOutcome::Cancel { record } => Ok(record),
            _ => Err(wrong_process_outcome("cancel")),
        }
    }

    async fn signal(
        &self,
        process_id: &str,
        signal_name: String,
        signal_id: String,
        request: crate::ProcessEventAppendRequest,
    ) -> Result<crate::ProcessEvent, crate::PluginError> {
        match self
            .run(crate::ProcessCommand::Signal {
                process_id: process_id.to_string(),
                signal_name,
                signal_id,
                request,
            })
            .await?
        {
            crate::ProcessEffectOutcome::Signal { event } => Ok(event),
            _ => Err(wrong_process_outcome("signal")),
        }
    }

    async fn transfer(
        &self,
        from_scope: crate::SessionScope,
        to_scope: crate::SessionScope,
        process_ids: Vec<String>,
    ) -> Result<(), crate::PluginError> {
        match self
            .run(crate::ProcessCommand::Transfer {
                from_scope,
                to_scope,
                process_ids,
            })
            .await?
        {
            crate::ProcessEffectOutcome::Transfer => Ok(()),
            _ => Err(wrong_process_outcome("transfer")),
        }
    }

    async fn run(
        &self,
        command: crate::ProcessCommand,
    ) -> Result<crate::ProcessEffectOutcome, crate::PluginError> {
        let effect_id = command.effect_id();
        let is_start = matches!(command, crate::ProcessCommand::Start { .. });
        let invocation = crate::runtime::causal::process_effect_invocation(
            &self.current.session_id,
            self.parent_invocation.clone(),
            &effect_id,
        );
        let envelope = crate::RuntimeEffectEnvelope::new(
            invocation,
            crate::RuntimeEffectCommand::process(command),
        );
        // Route through the controller explicitly selected by the process
        // operation scope: host-configured for host/API paths, scoped for
        // in-turn paths.
        let outcome = self
            .effect_controller
            .execute_effect(
                envelope,
                crate::RuntimeEffectLocalExecutor::process_control(Arc::clone(&self.registry)),
            )
            .await?;
        if is_start && let Some(poke) = self.current.host.process_work_poke.as_ref() {
            poke.poke();
        }
        outcome.into_process().map_err(crate::PluginError::from)
    }
}

fn wrong_process_outcome(op: &str) -> crate::PluginError {
    crate::PluginError::Session(format!("process {op} returned the wrong outcome"))
}

impl ProcessCapability {
    fn command_runner<'scope>(
        &self,
        current: &'scope CurrentSessionCapability,
        scope: &'scope crate::ProcessOpScope<'scope>,
    ) -> Result<ProcessCommandRunner<'scope>, crate::PluginError> {
        ProcessCommandRunner::new(
            current,
            scope,
            "process registry is unavailable in this runtime",
        )
    }

    fn process_scope_for_op(
        &self,
        session_id: &str,
        agent_frame_id: Option<&str>,
    ) -> crate::SessionScope {
        agent_frame_id
            .filter(|frame_id| !frame_id.is_empty())
            .map(|frame_id| crate::SessionScope::for_agent_frame(session_id, frame_id))
            .unwrap_or_else(|| crate::SessionScope::new(session_id))
    }

    fn current_execution_env_spec(
        &self,
        current: &CurrentSessionCapability,
    ) -> crate::ProcessExecutionEnvSpec {
        let state = current.snapshot.to_runtime_state();
        state.process_execution_env_spec(&current.policy)
    }

    async fn capture_execution_env_ref(
        &self,
        current: &CurrentSessionCapability,
        registration: &crate::ProcessRegistration,
    ) -> Result<Option<crate::ProcessExecutionEnvRef>, crate::PluginError> {
        if let Some(env_ref) = registration.env_ref.clone() {
            return Ok(Some(env_ref));
        }
        match registration.input.as_ref() {
            crate::ProcessInput::ToolCall { .. } | crate::ProcessInput::LashlangProcess { .. } => {
                let spec = self.current_execution_env_spec(current);
                crate::persist_process_execution_env(
                    current
                        .host
                        .core
                        .durability
                        .lashlang_artifact_store
                        .as_ref(),
                    &spec,
                )
                .await
                .map(Some)
            }
            crate::ProcessInput::External { .. } | crate::ProcessInput::SessionTurn { .. } => {
                Ok(None)
            }
        }
    }

    pub(in crate::runtime::session_manager) async fn start_process(
        &self,
        current: &CurrentSessionCapability,
        managed: &ManagedSessionCapability,
        session_id: &str,
        registration: crate::ProcessRegistration,
        options: crate::ProcessStartOptions,
        scope: crate::ProcessOpScope<'_>,
    ) -> Result<crate::ProcessRecord, crate::PluginError> {
        self.ensure_known_process_session(current, managed, session_id)
            .await?;
        self.mark_current_process_sync_needed(current, session_id);
        let creator_scope = self.process_scope_for_op(session_id, scope.agent_frame_id());
        let caused_by = scope
            .parent_invocation
            .as_ref()
            .and_then(crate::RuntimeInvocation::causal_ref);
        let env_ref = self
            .capture_execution_env_ref(current, &registration)
            .await?;
        // Children started *by a process* inherit the chain's provenance (the
        // run context provides it); in-session starts stamp the creating
        // session. The grant always follows the wake target, mirroring the
        // trigger fire path — the ephemeral execution scope must never appear
        // on a record.
        let (originator, wake_target) = match options.spawn_provenance.clone() {
            Some(spawn) => (spawn.originator, spawn.wake_target),
            None => (
                crate::ProcessOriginator::session(creator_scope.clone()),
                Some(creator_scope.clone()),
            ),
        };
        let grant_scope = wake_target.clone();
        let registration = registration
            .with_process_provenance(
                crate::ProcessProvenance::new(
                    originator,
                    current.host.core.profile.host_profile_id.clone(),
                )
                .with_caused_by(caused_by),
            )
            .with_execution_env_ref(env_ref)
            .with_wake_target(wake_target);
        let execution_context = options.execution_context(&scope);
        let runner = ProcessCommandRunner::new(
            current,
            &scope,
            "processes are unavailable in this runtime",
        )?;
        runner
            .start(
                registration,
                options.descriptor.and_then(|descriptor| {
                    grant_scope.map(|session_scope| crate::ProcessStartGrant {
                        session_scope,
                        descriptor,
                    })
                }),
                execution_context,
            )
            .await
    }

    pub(in crate::runtime::session_manager) async fn await_process(
        &self,
        current: &CurrentSessionCapability,
        process_id: &str,
        scope: crate::ProcessOpScope<'_>,
    ) -> Result<crate::ProcessAwaitOutput, crate::PluginError> {
        self.command_runner(current, &scope)?
            .await_process(process_id)
            .await
    }

    pub(in crate::runtime::session_manager) async fn list_process_handles(
        &self,
        current: &CurrentSessionCapability,
        session_id: &str,
        mode: crate::ProcessListMode,
        scope: crate::ProcessOpScope<'_>,
    ) -> Result<Vec<crate::runtime::ProcessHandleGrantEntry>, crate::PluginError> {
        self.command_runner(current, &scope)?
            .list(
                self.process_scope_for_op(session_id, scope.agent_frame_id()),
                mode,
            )
            .await
    }

    pub(in crate::runtime::session_manager) async fn cancel_process(
        &self,
        current: &CurrentSessionCapability,
        managed: &ManagedSessionCapability,
        session_id: &str,
        process_id: &str,
        scope: crate::ProcessOpScope<'_>,
    ) -> Result<crate::ProcessRecord, crate::PluginError> {
        let runner = self.command_runner(current, &scope)?;
        if runner.registry().get_process(process_id).await.is_none() {
            return Err(crate::PluginError::Session(format!(
                "unknown process `{process_id}`"
            )));
        }
        let _ = (managed, session_id);
        runner.cancel(process_id).await
    }

    pub(in crate::runtime::session_manager) async fn signal_process(
        &self,
        current: &CurrentSessionCapability,
        session_id: &str,
        process_id: &str,
        signal_name: String,
        signal_id: String,
        payload: serde_json::Value,
        scope: crate::ProcessOpScope<'_>,
    ) -> Result<crate::ProcessEvent, crate::PluginError> {
        let runner = self.command_runner(current, &scope)?;
        let session_scope = self.process_scope_for_op(session_id, scope.agent_frame_id());
        let visible = runner
            .registry()
            .list_live_handle_grants(&session_scope)
            .await?
            .into_iter()
            .any(|(grant, _record)| grant.process_id == process_id);
        if !visible {
            return Err(crate::PluginError::Session(format!(
                "process handle `{process_id}` is not live or visible in this session"
            )));
        }
        let event_type = crate::process_signal_event_type(&signal_name)?;
        let request = crate::ProcessEventAppendRequest::new(event_type, payload).with_replay_key(
            format!("process:{process_id}:signal.{signal_name}:{signal_id}"),
        );
        runner
            .signal(process_id, signal_name, signal_id, request)
            .await
    }

    pub(in crate::runtime::session_manager) async fn validate_process_handles_visible(
        &self,
        current: &CurrentSessionCapability,
        _managed: &ManagedSessionCapability,
        session_id: &str,
        handle_ids: &[String],
        scope: crate::ProcessOpScope<'_>,
    ) -> Result<(), crate::PluginError> {
        if handle_ids.is_empty() {
            return Ok(());
        }
        let runner = self.command_runner(current, &scope)?;
        let session_scope = self.process_scope_for_op(session_id, scope.agent_frame_id());
        for process_id in handle_ids {
            if !runner
                .registry()
                .has_handle_grant(&session_scope, process_id)
                .await?
            {
                return Err(crate::PluginError::Session(format!(
                    "process handle `{process_id}` is not live or visible in this session"
                )));
            }
        }
        Ok(())
    }

    pub(in crate::runtime::session_manager) async fn transfer_process_handles(
        &self,
        current: &CurrentSessionCapability,
        _managed: &ManagedSessionCapability,
        from_session_id: &str,
        to_session_id: &str,
        process_ids: Vec<String>,
        scope: crate::ProcessOpScope<'_>,
    ) -> Result<(), crate::PluginError> {
        if process_ids.is_empty() {
            return Ok(());
        }
        self.command_runner(current, &scope)?
            .transfer(
                self.process_scope_for_op(from_session_id, scope.agent_frame_id()),
                self.process_scope_for_op(to_session_id, scope.target_agent_frame_id()),
                process_ids,
            )
            .await
    }

    pub(in crate::runtime::session_manager) async fn cancel_unreferenced_process_handles(
        &self,
        current: &CurrentSessionCapability,
        _managed: &ManagedSessionCapability,
        session_id: &str,
        keep_process_ids: Vec<String>,
        scope: crate::ProcessOpScope<'_>,
    ) -> Result<Vec<crate::ProcessRecord>, crate::PluginError> {
        let keep = keep_process_ids.iter().cloned().collect::<HashSet<_>>();
        let runner = self.command_runner(current, &scope)?;
        let session_scope = self.process_scope_for_op(session_id, scope.agent_frame_id());
        let tasks = runner.registry().list_handle_grants(&session_scope).await?;
        let mut cancelled = Vec::new();
        for (grant, record) in tasks {
            if keep.contains(&grant.process_id) {
                continue;
            }
            runner
                .registry()
                .revoke_handle(&session_scope, &grant.process_id)
                .await?;
            if record.is_terminal() {
                continue;
            }
            if !runner
                .registry()
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
                    session_id,
                    &grant.process_id,
                    scope.clone(),
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
}
