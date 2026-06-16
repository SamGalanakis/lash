pub use crate::session::SessionConfigPatch;
use crate::support::*;
pub use lash_core::{AcceptedInjectedTurnInput, PluginAction};

#[derive(Clone)]
pub struct Completions {
    pub(crate) core: LashCore,
}

impl Completions {
    pub async fn resolve(
        &self,
        key: lash_core::AwaitEventKey,
        resolution: lash_core::Resolution,
    ) -> Result<lash_core::ResolveOutcome> {
        self.core
            .env
            .core
            .control
            .effect_host
            .resolve_await_event(&key, resolution)
            .await
            .map_err(|err| EmbedError::Plugin(lash_core::PluginError::Session(err.to_string())))
    }
}

#[derive(Clone)]
pub struct CoreTriggerAdmin {
    pub(crate) core: LashCore,
}

impl CoreTriggerAdmin {
    pub async fn emit(
        &self,
        request: lash_core::TriggerOccurrenceRequest,
        scoped_effect_controller: ScopedEffectController<'_>,
    ) -> Result<lash_core::TriggerEmitReport> {
        let store = self.core.env.trigger_store.as_ref().ok_or_else(|| {
            EmbedError::Plugin(lash_core::PluginError::Session(
                "trigger store is unavailable in this runtime".to_string(),
            ))
        })?;
        let process_work_poke = self.core.process_work_runner.poke().await;
        let router = lash_core::TriggerRouter::new(
            Arc::clone(store),
            Arc::clone(&self.core.env.core.durability.lashlang_artifact_store),
            self.core.env.process_registry.clone(),
            process_work_poke,
        );
        router
            .emit(request, scoped_effect_controller.controller())
            .await
            .map_err(Into::into)
    }

    pub async fn subscriptions(
        &self,
        filter: lash_core::TriggerSubscriptionFilter,
    ) -> Result<Vec<lash_core::TriggerRegistration>> {
        let store = self.core.env.trigger_store.as_ref().ok_or_else(|| {
            EmbedError::Plugin(lash_core::PluginError::Session(
                "trigger store is unavailable in this runtime".to_string(),
            ))
        })?;
        let records = store.list_subscriptions(filter).await?;
        Ok(records
            .iter()
            .map(lash_core::TriggerRegistration::from)
            .collect())
    }
}

#[derive(Clone)]
pub struct Processes {
    pub(crate) core: LashCore,
}

impl Processes {
    fn registry(&self) -> Result<Arc<dyn lash_core::ProcessRegistry>> {
        self.core
            .env
            .process_registry
            .as_ref()
            .cloned()
            .ok_or_else(|| {
                EmbedError::Plugin(lash_core::PluginError::Session(
                    "process registry is unavailable in this runtime".to_string(),
                ))
            })
    }

    fn make_observer(&self) -> Result<lash_core::ProcessWorkObserver> {
        Ok(lash_core::ProcessWorkObserver::new(self.registry()?))
    }

    fn process_invocation(command: &lash_core::ProcessCommand) -> lash_core::RuntimeInvocation {
        let effect_id = command.effect_id();
        lash_core::RuntimeInvocation::effect(
            lash_core::runtime::RuntimeScope::new("runtime"),
            effect_id.clone(),
            lash_core::RuntimeEffectKind::Process,
            effect_id,
        )
    }

    async fn run_command(
        &self,
        command: lash_core::ProcessCommand,
        scoped_effect_controller: ScopedEffectController<'_>,
    ) -> Result<lash_core::ProcessEffectOutcome> {
        let registry = self.registry()?;
        let invocation = Self::process_invocation(&command);
        let outcome = scoped_effect_controller
            .controller()
            .execute_effect(
                lash_core::RuntimeEffectEnvelope::new(
                    invocation,
                    lash_core::RuntimeEffectCommand::process(command),
                ),
                lash_core::RuntimeEffectLocalExecutor::processes(registry),
            )
            .await
            .map_err(|err| EmbedError::Plugin(lash_core::PluginError::Session(err.to_string())))?;
        match outcome {
            lash_core::RuntimeEffectOutcome::Process { result } => Ok(result),
            _ => Err(EmbedError::Plugin(lash_core::PluginError::Session(
                "process effect returned non-process outcome".to_string(),
            ))),
        }
    }

    pub async fn start(
        &self,
        request: lash_core::ProcessStartRequest,
        scoped_effect_controller: ScopedEffectController<'_>,
    ) -> Result<lash_core::ProcessRecord> {
        let env_ref = match request.env_spec.as_ref() {
            Some(env_spec) => Some(
                lash_core::runtime::persist_process_execution_env(
                    self.core
                        .env
                        .core
                        .durability
                        .lashlang_artifact_store
                        .as_ref(),
                    env_spec,
                )
                .await?,
            ),
            None => None,
        };
        let grant = request.grant.clone();
        let registration = request.into_registration(env_ref);
        let command = lash_core::ProcessCommand::Start {
            registration,
            grant,
            execution_context: Box::new(lash_core::ProcessExecutionContext::default()),
        };
        let outcome = self
            .run_command(command, scoped_effect_controller.clone())
            .await?;
        let lash_core::ProcessEffectOutcome::Start { record } = outcome else {
            return Err(EmbedError::Plugin(lash_core::PluginError::Session(
                "process start returned the wrong outcome".to_string(),
            )));
        };
        if let Some(poke) = self.core.process_work_runner.poke().await {
            poke.poke();
        }
        Ok(record)
    }

    pub async fn list(
        &self,
        filter: &lash_core::ProcessListFilter,
    ) -> Result<Vec<lash_core::ObservedProcess>> {
        self.make_observer()?.list(filter).await.map_err(Into::into)
    }

    pub async fn get(&self, process_id: &str) -> Result<Option<lash_core::ObservedProcess>> {
        Ok(self.make_observer()?.process(process_id).await)
    }

    pub async fn events(
        &self,
        process_id: &str,
        after_sequence: u64,
    ) -> Result<Vec<lash_core::ObservedProcessEvent>> {
        self.make_observer()?
            .events_after(process_id, after_sequence)
            .await
            .map_err(Into::into)
    }

    pub async fn await_output(&self, process_id: &str) -> Result<lash_core::ProcessAwaitOutput> {
        self.registry()?
            .await_process(process_id)
            .await
            .map_err(Into::into)
    }

    pub async fn cancel(
        &self,
        process_id: &str,
        scoped_effect_controller: ScopedEffectController<'_>,
    ) -> Result<lash_core::ProcessCancelSummary> {
        let command = lash_core::ProcessCommand::Cancel {
            process_id: process_id.to_string(),
            reason: Some("requested by host".to_string()),
        };
        let outcome = self
            .run_command(command, scoped_effect_controller.clone())
            .await?;
        let lash_core::ProcessEffectOutcome::Cancel { record } = outcome else {
            return Err(EmbedError::Plugin(lash_core::PluginError::Session(
                "process cancel returned the wrong outcome".to_string(),
            )));
        };
        Ok(lash_core::ProcessCancelSummary::from_record(record))
    }

    pub async fn signal(
        &self,
        process_id: &str,
        signal_name: impl Into<String>,
        signal_id: impl Into<String>,
        request: lash_core::ProcessEventAppendRequest,
        scoped_effect_controller: ScopedEffectController<'_>,
    ) -> Result<lash_core::ProcessEvent> {
        let signal_name = signal_name.into();
        let event_type = request.event_type.clone();
        let payload = request.payload.clone();
        let command = lash_core::ProcessCommand::Signal {
            process_id: process_id.to_string(),
            signal_name: signal_name.clone(),
            signal_id: signal_id.into(),
            request,
        };
        let outcome = self
            .run_command(command, scoped_effect_controller.clone())
            .await?;
        let lash_core::ProcessEffectOutcome::Signal { event } = outcome else {
            return Err(EmbedError::Plugin(lash_core::PluginError::Session(
                "process signal returned the wrong outcome".to_string(),
            )));
        };
        let registry = self.registry()?;
        let waiting_ordinal =
            registry
                .get_process(process_id)
                .await
                .and_then(|record| match record.wait {
                    Some(lash_core::WaitState {
                        kind:
                            lash_core::WaitKind::Signal {
                                name,
                                event_type: wait_event_type,
                                ordinal,
                                ..
                            },
                        ..
                    }) if name == signal_name && wait_event_type == event_type => Some(ordinal),
                    _ => None,
                });
        let ordinal = match waiting_ordinal {
            Some(ordinal) => ordinal,
            None => {
                registry
                    .count_events_through(process_id, &event_type, event.sequence)
                    .await?
            }
        };
        if ordinal > 0 {
            let key = scoped_effect_controller
                .controller()
                .await_event_key(
                    &lash_core::ExecutionScope::process(process_id),
                    lash_core::AwaitEventWaitIdentity::process_signal(
                        process_id,
                        &signal_name,
                        ordinal,
                    ),
                )
                .await
                .map_err(|err| {
                    EmbedError::Plugin(lash_core::PluginError::Session(err.to_string()))
                })?;
            let _ = scoped_effect_controller
                .controller()
                .resolve_await_event(&key, lash_core::Resolution::Ok(payload))
                .await
                .map_err(|err| {
                    EmbedError::Plugin(lash_core::PluginError::Session(err.to_string()))
                })?;
        }
        Ok(event)
    }

    pub async fn session_snapshot(
        &self,
        session_id: impl Into<String>,
    ) -> Result<lash_core::ProcessWorkSnapshot> {
        self.make_observer()?
            .snapshot_for_session(session_id)
            .await
            .map_err(Into::into)
    }

    pub fn observer(&self) -> Result<lash_core::ProcessWorkObserver> {
        self.make_observer()
    }
}

#[derive(Clone)]
pub struct SessionAdmin {
    pub(crate) runtime: RuntimeHandle,
}

impl SessionAdmin {
    pub fn config(&self) -> SessionConfigAdmin {
        SessionConfigAdmin {
            control: self.clone(),
        }
    }

    pub fn tools(&self) -> ToolAdmin {
        ToolAdmin {
            control: self.clone(),
        }
    }

    pub fn commands(&self) -> SessionCommandAdmin {
        SessionCommandAdmin {
            control: self.clone(),
        }
    }

    pub fn triggers(&self) -> SessionTriggerAdmin {
        SessionTriggerAdmin {
            control: self.clone(),
        }
    }

    pub fn state(&self) -> SessionStateAdmin {
        SessionStateAdmin {
            control: self.clone(),
        }
    }

    pub fn children(&self) -> ChildSessionAdmin {
        ChildSessionAdmin {
            control: self.clone(),
        }
    }

    pub fn injection(&self) -> InjectionAdmin {
        InjectionAdmin {
            control: self.clone(),
        }
    }

    pub fn mode(&self) -> ModeAdmin {
        ModeAdmin {
            control: self.clone(),
        }
    }

    pub fn processes(&self) -> SessionProcessAdmin {
        SessionProcessAdmin {
            control: self.clone(),
        }
    }

    /// Run `f` against the locked runtime writer, then publish the resulting
    /// observation. The body is the canonical `lock → call → publish_from`
    /// stamp shared by nearly every mutating control method; publish happens
    /// unconditionally once the closure returns.
    async fn with_writer<F, T>(&self, f: F) -> T
    where
        F: AsyncFnOnce(&mut LashRuntime) -> T,
    {
        let writer = self.runtime.writer();
        let mut runtime = writer.lock().await;
        let value = f(&mut runtime).await;
        self.runtime.publish_from(&runtime);
        value
    }

    async fn update_config(&self, patch: SessionConfigPatch) -> Result<()> {
        self.update_session_config(patch.provider, patch.model, patch.prompt)
            .await?;
        Ok(())
    }

    async fn update_session_config(
        &self,
        provider: Option<ProviderHandle>,
        model: Option<lash_core::ModelSpec>,
        prompt: Option<PromptLayer>,
    ) -> Result<()> {
        self.with_writer(async |runtime: &mut LashRuntime| {
            runtime.update_session_config(provider, model, prompt).await;
        })
        .await;
        Ok(())
    }

    async fn export_state(&self) -> lash_core::SessionSnapshot {
        self.runtime.observe().read_view.to_snapshot()
    }

    async fn append_messages(&self, messages: Vec<PluginMessage>) -> Result<()> {
        self.with_writer(async |runtime: &mut LashRuntime| {
            runtime
                .append_session_nodes(lash_core::AppendSessionNodesRequest {
                    nodes: messages
                        .into_iter()
                        .map(lash_core::SessionAppendNode::message)
                        .collect(),
                    requires_ancestor_node_id: None,
                })
                .await
                .map(|_| ())
                .map_err(Into::into)
        })
        .await
    }

    async fn append_plugin_body(
        &self,
        plugin_type: impl Into<String>,
        body: serde_json::Value,
    ) -> Result<()> {
        self.with_writer(async |runtime: &mut LashRuntime| {
            runtime
                .append_session_nodes(lash_core::AppendSessionNodesRequest {
                    nodes: vec![lash_core::SessionAppendNode::plugin(plugin_type, body)],
                    requires_ancestor_node_id: None,
                })
                .await
                .map(|_| ())
                .map_err(Into::into)
        })
        .await
    }

    async fn set_persisted_state(&self, state: RuntimeSessionState) -> Result<()> {
        self.with_writer(async |runtime: &mut LashRuntime| {
            runtime.set_persisted_state(state).map_err(Into::into)
        })
        .await
    }

    async fn set_prompt_template(&self, template: PromptTemplate) -> Result<()> {
        self.with_writer(async |runtime: &mut LashRuntime| {
            runtime.set_prompt_template(template).await;
        })
        .await;
        Ok(())
    }

    async fn clear_prompt_template(&self) -> Result<()> {
        self.with_writer(async |runtime: &mut LashRuntime| {
            runtime.clear_prompt_template().await;
        })
        .await;
        Ok(())
    }

    async fn add_prompt_contribution(&self, contribution: PromptContribution) -> Result<()> {
        self.with_writer(async |runtime: &mut LashRuntime| {
            runtime.add_prompt_contribution(contribution).await;
        })
        .await;
        Ok(())
    }

    async fn replace_prompt_slot(
        &self,
        slot: PromptSlot,
        contributions: impl IntoIterator<Item = PromptContribution>,
    ) -> Result<()> {
        self.with_writer(async |runtime: &mut LashRuntime| {
            runtime.replace_prompt_slot(slot, contributions).await;
        })
        .await;
        Ok(())
    }

    async fn clear_prompt_slot(&self, slot: PromptSlot) -> Result<()> {
        self.with_writer(async |runtime: &mut LashRuntime| {
            runtime.clear_prompt_slot(slot).await;
        })
        .await;
        Ok(())
    }

    async fn apply_protocol_session_extension(
        &self,
        extension: lash_core::ProtocolSessionExtensionHandle,
    ) -> Result<()> {
        self.with_writer(async |runtime: &mut LashRuntime| {
            runtime
                .apply_protocol_session_extension(extension)
                .await
                .map_err(Into::into)
        })
        .await
    }

    async fn branch_to_node(
        &self,
        target_leaf: Option<String>,
    ) -> Result<lash_core::SessionSnapshot> {
        self.with_writer(async |runtime: &mut LashRuntime| {
            runtime
                .branch_to_node(target_leaf)
                .await
                .map_err(Into::into)
        })
        .await
    }

    async fn await_background_work(&self) -> Result<()> {
        self.with_writer(async |runtime: &mut LashRuntime| {
            runtime.await_background_work().await.map_err(Into::into)
        })
        .await
    }

    async fn refresh_tool_catalog(&self) -> Result<()> {
        self.with_writer(async |runtime: &mut LashRuntime| {
            runtime
                .refresh_session_tool_catalog()
                .await
                .map_err(Into::into)
        })
        .await
    }

    async fn submit_session_command(
        &self,
        command: lash_core::SessionCommand,
        idempotency_key: impl Into<String>,
    ) -> Result<lash_core::SessionCommandReceipt> {
        let idempotency_key = idempotency_key.into();
        self.with_writer(async |runtime: &mut LashRuntime| {
            runtime
                .submit_session_command(command, idempotency_key)
                .await
                .map_err(Into::into)
        })
        .await
    }

    async fn list_lashlang_trigger_registrations(
        &self,
    ) -> Result<Vec<lash_core::TriggerRegistration>> {
        self.with_writer(async |runtime: &mut LashRuntime| {
            runtime
                .list_lashlang_trigger_registrations()
                .await
                .map_err(Into::into)
        })
        .await
    }

    async fn lashlang_trigger_registrations_by_source_type(
        &self,
        source_type: impl Into<lash_core::TriggerEventType>,
    ) -> Result<Vec<lash_core::TriggerRegistration>> {
        self.with_writer(async |runtime: &mut LashRuntime| {
            runtime
                .lashlang_trigger_registrations_by_source_type(source_type)
                .await
                .map_err(Into::into)
        })
        .await
    }

    async fn invoke_plugin_action(
        &self,
        name: &str,
        args: serde_json::Value,
    ) -> Result<ToolResult> {
        let session_id = self.runtime.observe().session_id().to_string();
        let writer = self.runtime.writer();
        writer
            .lock()
            .await
            .invoke_plugin_action(name, args, Some(session_id))
            .await
            .map_err(Into::into)
    }

    async fn call_plugin_action<Op: lash_core::PluginAction>(
        &self,
        args: Op::Args,
    ) -> Result<Op::Output> {
        let result = self
            .invoke_plugin_action(
                Op::NAME,
                serde_json::to_value(args).map_err(|err| {
                    EmbedError::Plugin(lash_core::PluginError::Invoke(format!(
                        "invalid {} args: {err}",
                        Op::NAME
                    )))
                })?,
            )
            .await?;
        let Some(output) = result.as_done_output() else {
            return Err(EmbedError::Plugin(lash_core::PluginError::Invoke(format!(
                "{} returned a pending result where completed output is required",
                Op::NAME
            ))));
        };
        if !output.is_success() {
            return Err(EmbedError::Plugin(lash_core::PluginError::Invoke(format!(
                "{} failed: {}",
                Op::NAME,
                output.value_for_projection()
            ))));
        }
        serde_json::from_value(output.value_for_projection()).map_err(|err| {
            EmbedError::Plugin(lash_core::PluginError::Invoke(format!(
                "invalid {} output: {err}",
                Op::NAME
            )))
        })
    }

    async fn compact_context(
        &self,
        instructions: Option<String>,
        scoped_effect_controller: ScopedEffectController<'_>,
    ) -> Result<bool> {
        self.with_writer(async |runtime: &mut LashRuntime| {
            runtime
                .compact_context(instructions, scoped_effect_controller)
                .await
                .map_err(Into::into)
        })
        .await
    }

    async fn persist_current_state(&self) -> Result<RuntimeSessionState> {
        self.with_writer(async |runtime: &mut LashRuntime| {
            runtime.await_background_work().await?;
            Ok(runtime.export_persisted_state())
        })
        .await
    }

    async fn list_process_handles(&self) -> Result<Vec<lash_core::ProcessHandleSummary>> {
        Ok(self.runtime.observe().list_process_handles().await)
    }

    async fn list_all_process_handles(&self) -> Result<Vec<lash_core::ProcessHandleSummary>> {
        Ok(self.runtime.observe().list_all_process_handles().await)
    }

    async fn start_process(
        &self,
        request: lash_core::ProcessStartRequest,
        scoped_effect_controller: ScopedEffectController<'_>,
    ) -> Result<lash_core::ProcessHandleSummary> {
        let writer = self.runtime.writer();
        let runtime = writer.lock().await;
        let session_id = runtime.session_id().to_string();
        let processes = runtime.process_service()?;
        let scope = lash_core::ProcessOpScope::new(scoped_effect_controller);
        let summary = processes
            .start_from_request(&session_id, request, scope)
            .await
            .map_err(EmbedError::Plugin)?;
        self.runtime.record_process_changed(
            SessionProcessEventKind::Started,
            vec![summary.process_id.clone()],
        );
        Ok(summary)
    }

    async fn session_state_service(&self) -> Result<Arc<dyn SessionStateService>> {
        self.runtime
            .writer()
            .lock()
            .await
            .session_state_service()
            .map_err(Into::into)
    }

    async fn cancel_process(
        &self,
        process_id: &str,
        scoped_effect_controller: ScopedEffectController<'_>,
    ) -> Result<lash_core::ProcessCancelSummary> {
        let writer = self.runtime.writer();
        let runtime = writer.lock().await;
        let session_id = runtime.session_id().to_string();
        let processes = runtime.process_service()?;
        let cancel_ability = runtime.process_cancel_ability();
        let scope = lash_core::ProcessOpScope::new(scoped_effect_controller);
        let summary = cancel_ability
            .cancel_summary(
                processes.as_ref(),
                lash_core::ProcessCancelRequest::new(
                    &session_id,
                    process_id,
                    scope,
                    lash_core::ProcessCancelSource::HostApi,
                )
                .with_reason("requested by host API"),
            )
            .await
            .map_err(EmbedError::Plugin)?;
        self.runtime.record_process_changed(
            SessionProcessEventKind::Cancelled,
            vec![summary.process_id.clone()],
        );
        Ok(summary)
    }

    async fn cancel_visible_processes(
        &self,
        scoped_effect_controller: ScopedEffectController<'_>,
    ) -> Result<Vec<lash_core::ProcessCancelSummary>> {
        let writer = self.runtime.writer();
        let runtime = writer.lock().await;
        let session_id = runtime.session_id().to_string();
        let processes = runtime.process_service()?;
        let cancel_ability = runtime.process_cancel_ability();
        let scope = lash_core::ProcessOpScope::new(scoped_effect_controller);
        let summaries = cancel_ability
            .cancel_all_visible(
                processes.as_ref(),
                lash_core::ProcessCancelAllRequest::new(
                    &session_id,
                    scope,
                    lash_core::ProcessCancelSource::HostApi,
                )
                .with_reason("requested by host API"),
            )
            .await
            .map_err(EmbedError::Plugin)?;
        self.runtime.record_process_changed(
            SessionProcessEventKind::Cancelled,
            summaries
                .iter()
                .map(|summary| summary.process_id.clone())
                .collect(),
        );
        Ok(summaries)
    }

    async fn snapshot_execution_state(&self) -> Result<Option<Vec<u8>>> {
        self.with_writer(async |runtime: &mut LashRuntime| {
            runtime.snapshot_execution_state().await.map_err(Into::into)
        })
        .await
    }

    async fn restore_execution_state(&self, bytes: &[u8]) -> Result<()> {
        self.with_writer(async |runtime: &mut LashRuntime| {
            runtime
                .restore_execution_state(bytes)
                .await
                .map_err(Into::into)
        })
        .await
    }

    async fn tool_state(&self) -> Result<ToolState> {
        self.runtime.observe().tool_state.clone().ok_or_else(|| {
            EmbedError::Session(SessionError::Protocol(
                "runtime session not available".to_string(),
            ))
        })
    }

    async fn apply_tool_state(&self, state: ToolState) -> Result<u64> {
        self.with_writer(async |runtime: &mut LashRuntime| {
            runtime
                .apply_tool_state(state)
                .await
                .map_err(EmbedError::from)
        })
        .await
    }

    async fn restore_tool_state(&self, state: ToolState) -> Result<ToolRestoreReport> {
        self.with_writer(async |runtime: &mut LashRuntime| {
            runtime
                .restore_tool_state(state)
                .await
                .map_err(EmbedError::from)
        })
        .await
    }

    async fn set_tool_availability(
        &self,
        name: &str,
        availability: ToolAvailability,
    ) -> Result<u64> {
        self.set_tool_availability_many(&[(name, availability)])
            .await
    }

    async fn set_tool_availability_many<N: AsRef<str>>(
        &self,
        updates: &[(N, ToolAvailability)],
    ) -> Result<u64> {
        let mut state = self.tool_state().await?;
        for (name, availability) in updates {
            state
                .set_availability(name.as_ref(), Some(*availability))
                .map_err(|err| EmbedError::Session(SessionError::Protocol(err.to_string())))?;
        }
        self.apply_tool_state(state).await
    }

    async fn clear_tool_availability_override(&self, name: &str) -> Result<u64> {
        let mut state = self.tool_state().await?;
        state
            .set_availability(name, None)
            .map_err(|err| EmbedError::Session(SessionError::Protocol(err.to_string())))?;
        self.apply_tool_state(state).await
    }

    async fn active_tool_manifests(&self) -> Result<Vec<ToolManifest>> {
        Ok(self.tool_state().await?.tool_manifests())
    }

    async fn add_tool_provider(&self, provider: Arc<dyn ToolProvider>) -> Result<ToolSourceHandle> {
        let tool_registry = self.tool_registry().await?;
        let handle = tool_registry
            .add_tool_provider(provider)
            .map_err(|err| EmbedError::Session(SessionError::Protocol(err.to_string())))?;
        self.refresh_tool_catalog().await?;
        Ok(handle)
    }

    async fn remove_tool_source(&self, handle: &ToolSourceHandle) -> Result<u64> {
        let tool_registry = self.tool_registry().await?;
        let generation = tool_registry
            .remove_source(handle)
            .map_err(|err| EmbedError::Session(SessionError::Protocol(err.to_string())))?;
        self.refresh_tool_catalog().await?;
        Ok(generation)
    }

    async fn create_child_session(&self, request: SessionCreateRequest) -> Result<SessionHandle> {
        let writer = self.runtime.writer();
        let runtime = writer.lock().await;
        let lifecycle = runtime.session_lifecycle_service()?;
        lifecycle.create_session(request).await.map_err(Into::into)
    }

    async fn close_child_session(&self, session_id: &str) -> Result<()> {
        let writer = self.runtime.writer();
        let runtime = writer.lock().await;
        let lifecycle = runtime.session_lifecycle_service()?;
        lifecycle
            .close_session(session_id)
            .await
            .map_err(Into::into)
    }

    async fn activate_managed_session(&self, session_id: &str) -> Result<()> {
        self.with_writer(async |runtime: &mut LashRuntime| {
            runtime
                .activate_managed_session(session_id)
                .await
                .map_err(Into::into)
        })
        .await
    }

    async fn inject_turn_input(&self, id: Option<String>, message: PluginMessage) -> Result<()> {
        self.inject_turn_inputs(vec![lash_core::InjectedTurnInput { id, message }])
            .await
    }

    async fn inject_turn_inputs(&self, messages: Vec<lash_core::InjectedTurnInput>) -> Result<()> {
        for input in messages {
            let source_key = input.id.map(|id| format!("injection:{id}"));
            let turn_input = turn_input_from_plugin_message(input.message);
            self.runtime
                .enqueue_turn_input(
                    turn_input,
                    lash_core::DeliveryPolicy::EarliestSafeBoundary,
                    lash_core::SlotPolicy::Join,
                    source_key,
                )
                .await
                .map(|_| ())
                .map_err(EmbedError::Runtime)?;
        }
        Ok(())
    }

    async fn tool_registry(&self) -> Result<Arc<lash_core::ToolRegistry>> {
        self.runtime
            .writer()
            .lock()
            .await
            .plugin_session()
            .map(|session| session.tool_registry())
            .ok_or_else(|| {
                EmbedError::Session(SessionError::Protocol(
                    "tool registry is unavailable in this runtime session".to_string(),
                ))
            })
    }
}

fn turn_input_from_plugin_message(message: PluginMessage) -> TurnInput {
    let mut input = TurnInput::empty();
    if !message.content.is_empty() {
        input.items.push(InputItem::Text {
            text: message.content,
        });
    }
    for (index, bytes) in message.images.into_iter().enumerate() {
        let id = format!("injected-image-{index}");
        input.items.push(InputItem::ImageRef { id: id.clone() });
        input.image_blobs.insert(id, bytes);
    }
    input
}

#[derive(Clone)]
pub struct SessionConfigAdmin {
    control: SessionAdmin,
}

impl SessionConfigAdmin {
    pub async fn update(&self, patch: SessionConfigPatch) -> Result<()> {
        self.control.update_config(patch).await
    }

    pub async fn update_session_config(
        &self,
        provider: Option<ProviderHandle>,
        model: Option<lash_core::ModelSpec>,
        prompt: Option<PromptLayer>,
    ) -> Result<()> {
        self.control
            .update_session_config(provider, model, prompt)
            .await
    }

    pub async fn set_prompt_template(&self, template: PromptTemplate) -> Result<()> {
        self.control.set_prompt_template(template).await
    }

    pub async fn clear_prompt_template(&self) -> Result<()> {
        self.control.clear_prompt_template().await
    }

    pub async fn add_prompt_contribution(&self, contribution: PromptContribution) -> Result<()> {
        self.control.add_prompt_contribution(contribution).await
    }

    pub async fn replace_prompt_slot(
        &self,
        slot: PromptSlot,
        contributions: impl IntoIterator<Item = PromptContribution>,
    ) -> Result<()> {
        self.control.replace_prompt_slot(slot, contributions).await
    }

    pub async fn clear_prompt_slot(&self, slot: PromptSlot) -> Result<()> {
        self.control.clear_prompt_slot(slot).await
    }
}

#[derive(Clone)]
pub struct ToolAdmin {
    control: SessionAdmin,
}

impl ToolAdmin {
    pub(crate) fn new(control: SessionAdmin) -> Self {
        Self { control }
    }
}

impl ToolAdmin {
    pub async fn state(&self) -> Result<ToolState> {
        self.control.tool_state().await
    }

    pub fn advanced(&self) -> AdvancedToolAdmin {
        AdvancedToolAdmin {
            control: self.control.clone(),
        }
    }

    pub async fn set_availability(
        &self,
        name: impl AsRef<str>,
        availability: ToolAvailability,
    ) -> Result<u64> {
        self.control
            .set_tool_availability(name.as_ref(), availability)
            .await
    }

    pub async fn set_availability_many<N: AsRef<str>>(
        &self,
        updates: &[(N, ToolAvailability)],
    ) -> Result<u64> {
        self.control.set_tool_availability_many(updates).await
    }

    pub async fn clear_availability_override(&self, name: impl AsRef<str>) -> Result<u64> {
        self.control
            .clear_tool_availability_override(name.as_ref())
            .await
    }

    pub async fn active_manifests(&self) -> Result<Vec<ToolManifest>> {
        self.control.active_tool_manifests().await
    }

    pub async fn add_provider(&self, provider: Arc<dyn ToolProvider>) -> Result<ToolSourceHandle> {
        self.control.add_tool_provider(provider).await
    }

    pub async fn remove_source(&self, handle: &ToolSourceHandle) -> Result<u64> {
        self.control.remove_tool_source(handle).await
    }
}

#[derive(Clone)]
pub struct AdvancedToolAdmin {
    control: SessionAdmin,
}

impl AdvancedToolAdmin {
    /// Replace the entire tool-state snapshot.
    ///
    /// This is a generation-checked escape hatch for hosts that intentionally
    /// edit the full snapshot. Prefer `ToolAdmin` availability methods for
    /// ordinary tool policy changes.
    pub async fn apply_state(&self, state: ToolState) -> Result<u64> {
        self.control.apply_tool_state(state).await
    }

    /// Restore a persisted tool-state snapshot, adopting its generation.
    ///
    /// Use this when re-applying a snapshot read from durable storage (session
    /// resume), not an edited delta: it reconstructs the exact persisted surface
    /// idempotently rather than requiring the snapshot to match the current
    /// generation. A cold resume of a session whose surface reached generation
    /// ≥ 2 needs this — [`apply_state`](Self::apply_state) would reject it.
    ///
    /// Persisted tools whose source is not currently registered (e.g. a
    /// detached MCP server) do not fail the restore: they are kept as orphans,
    /// forced `Off`, listed in the returned [`ToolRestoreReport`], and rebind
    /// automatically when a source re-advertises the same tool.
    pub async fn restore_state(&self, state: ToolState) -> Result<ToolRestoreReport> {
        self.control.restore_tool_state(state).await
    }
}

#[derive(Clone)]
pub struct SessionCommandAdmin {
    control: SessionAdmin,
}

impl SessionCommandAdmin {
    /// Enqueue an unconditional tool-catalog refresh. The command drains
    /// asynchronously and recomputes the surface from live sources, so it
    /// takes no generation guard — any generation observed at enqueue time
    /// could legitimately have advanced by drain time.
    pub async fn refresh_tool_catalog(
        &self,
        reason: impl Into<String>,
        idempotency_key: impl Into<String>,
    ) -> Result<lash_core::SessionCommandReceipt> {
        self.control
            .submit_session_command(
                lash_core::SessionCommand::RefreshToolCatalog {
                    reason: reason.into(),
                },
                idempotency_key,
            )
            .await
    }

    pub async fn reset(
        &self,
        reason: impl Into<String>,
        idempotency_key: impl Into<String>,
    ) -> Result<lash_core::SessionCommandReceipt> {
        self.control
            .submit_session_command(
                lash_core::SessionCommand::ResetSession {
                    reason: reason.into(),
                },
                idempotency_key,
            )
            .await
    }
}

/// Session-scoped read controls for Lashlang trigger registrations.
#[derive(Clone)]
pub struct SessionTriggerAdmin {
    control: SessionAdmin,
}

impl SessionTriggerAdmin {
    /// Return every trigger registration in the session.
    ///
    /// This is an admin/introspection view. Source owners should prefer
    /// [`Self::by_source_type`] so they only inspect registrations for the
    /// concrete source type they own.
    pub async fn list_all(&self) -> Result<Vec<lash_core::TriggerRegistration>> {
        self.control.list_lashlang_trigger_registrations().await
    }

    /// Return registrations whose source value has the given host descriptor type.
    ///
    /// This is the source-owner API: a timer, UI, webhook, or other host-owned
    /// source uses it to inspect registrations for keys it may schedule and emit.
    pub async fn by_source_type(
        &self,
        source_type: impl Into<lash_core::TriggerEventType>,
    ) -> Result<Vec<lash_core::TriggerRegistration>> {
        self.control
            .lashlang_trigger_registrations_by_source_type(source_type)
            .await
    }
}

#[derive(Clone)]
pub struct SessionProcessAdmin {
    control: SessionAdmin,
}

impl SessionProcessAdmin {
    pub(crate) fn new(control: SessionAdmin) -> Self {
        Self { control }
    }

    pub async fn start(
        &self,
        request: lash_core::ProcessStartRequest,
        scoped_effect_controller: ScopedEffectController<'_>,
    ) -> Result<lash_core::ProcessHandleSummary> {
        self.control
            .start_process(request, scoped_effect_controller)
            .await
    }

    pub async fn list(&self) -> Result<Vec<lash_core::ProcessHandleSummary>> {
        self.control.list_process_handles().await
    }

    pub async fn list_all(&self) -> Result<Vec<lash_core::ProcessHandleSummary>> {
        self.control.list_all_process_handles().await
    }

    pub async fn await_all(&self) -> Result<()> {
        self.control.await_background_work().await
    }

    pub async fn cancel(
        &self,
        process_id: &str,
        scoped_effect_controller: ScopedEffectController<'_>,
    ) -> Result<lash_core::ProcessCancelSummary> {
        self.control
            .cancel_process(process_id, scoped_effect_controller)
            .await
    }

    pub async fn cancel_all(
        &self,
        scoped_effect_controller: ScopedEffectController<'_>,
    ) -> Result<Vec<lash_core::ProcessCancelSummary>> {
        self.control
            .cancel_visible_processes(scoped_effect_controller)
            .await
    }
}

#[derive(Clone)]
pub struct SessionStateAdmin {
    control: SessionAdmin,
}

impl SessionStateAdmin {
    pub async fn export(&self) -> lash_core::SessionSnapshot {
        self.control.export_state().await
    }

    pub async fn append_messages(&self, messages: Vec<PluginMessage>) -> Result<()> {
        self.control.append_messages(messages).await
    }

    pub async fn append_plugin_body(
        &self,
        plugin_type: impl Into<String>,
        body: serde_json::Value,
    ) -> Result<()> {
        self.control.append_plugin_body(plugin_type, body).await
    }

    pub async fn set_persisted(&self, state: RuntimeSessionState) -> Result<()> {
        self.control.set_persisted_state(state).await
    }

    pub async fn branch_to_node(
        &self,
        target_leaf: Option<String>,
    ) -> Result<lash_core::SessionSnapshot> {
        self.control.branch_to_node(target_leaf).await
    }

    pub async fn persist_current(&self) -> Result<RuntimeSessionState> {
        self.control.persist_current_state().await
    }

    pub async fn session_state_service(&self) -> Result<Arc<dyn SessionStateService>> {
        self.control.session_state_service().await
    }

    pub async fn snapshot_execution(&self) -> Result<Option<Vec<u8>>> {
        self.control.snapshot_execution_state().await
    }

    pub async fn restore_execution(&self, bytes: &[u8]) -> Result<()> {
        self.control.restore_execution_state(bytes).await
    }

    pub async fn compact_context(
        &self,
        instructions: Option<String>,
        scoped_effect_controller: ScopedEffectController<'_>,
    ) -> Result<bool> {
        self.control
            .compact_context(instructions, scoped_effect_controller)
            .await
    }
}

#[derive(Clone)]
pub struct PluginActions {
    pub(crate) control: SessionAdmin,
}

impl PluginActions {
    pub async fn call<Op: lash_core::PluginAction>(&self, args: Op::Args) -> Result<Op::Output> {
        self.control.call_plugin_action::<Op>(args).await
    }
}

#[derive(Clone)]
pub struct ChildSessionAdmin {
    control: SessionAdmin,
}

impl ChildSessionAdmin {
    pub async fn create_session(&self, request: SessionCreateRequest) -> Result<SessionHandle> {
        self.control.create_child_session(request).await
    }

    pub async fn close_session(&self, session_id: &str) -> Result<()> {
        self.control.close_child_session(session_id).await
    }

    pub async fn activate_managed_session(&self, session_id: &str) -> Result<()> {
        self.control.activate_managed_session(session_id).await
    }
}

#[derive(Clone)]
pub struct InjectionAdmin {
    control: SessionAdmin,
}

impl InjectionAdmin {
    pub async fn inject_turn_input(
        &self,
        id: Option<String>,
        message: PluginMessage,
    ) -> Result<()> {
        self.control.inject_turn_input(id, message).await
    }

    pub async fn inject_turn_inputs(
        &self,
        messages: Vec<lash_core::InjectedTurnInput>,
    ) -> Result<()> {
        self.control.inject_turn_inputs(messages).await
    }
}

#[derive(Clone)]
pub struct ModeAdmin {
    control: SessionAdmin,
}

impl ModeAdmin {
    pub async fn apply_session_extension(
        &self,
        extension: lash_core::ProtocolSessionExtensionHandle,
    ) -> Result<()> {
        self.control
            .apply_protocol_session_extension(extension)
            .await
    }
}
