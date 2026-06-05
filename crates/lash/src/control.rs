pub use crate::session::SessionConfigPatch;
use crate::support::*;
pub use lash_core::{AcceptedInjectedTurnInput, PluginAction};

#[derive(Clone)]
pub struct SessionControl {
    pub(crate) runtime: RuntimeHandle,
}

impl SessionControl {
    pub fn config(&self) -> ConfigControl {
        ConfigControl {
            control: self.clone(),
        }
    }

    pub fn tools(&self) -> ToolsControl {
        ToolsControl {
            control: self.clone(),
        }
    }

    pub fn commands(&self) -> SessionCommandsControl {
        SessionCommandsControl {
            control: self.clone(),
        }
    }

    pub fn triggers(&self) -> TriggersControl {
        TriggersControl {
            control: self.clone(),
        }
    }

    pub fn state(&self) -> StateControl {
        StateControl {
            control: self.clone(),
        }
    }

    pub fn children(&self) -> ChildrenControl {
        ChildrenControl {
            control: self.clone(),
        }
    }

    pub fn injection(&self) -> InjectionControl {
        InjectionControl {
            control: self.clone(),
        }
    }

    pub fn mode(&self) -> ModeControl {
        ModeControl {
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

    async fn refresh_tool_surface(&self) -> Result<()> {
        self.with_writer(async |runtime: &mut LashRuntime| {
            runtime
                .refresh_session_tool_surface()
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

    async fn activate_lashlang_trigger(
        &self,
        handle: &str,
        payload: serde_json::Value,
    ) -> Result<lash_core::HostEventEmitReport> {
        self.with_writer(async |runtime: &mut LashRuntime| {
            runtime
                .activate_lashlang_trigger(handle, payload)
                .await
                .map_err(Into::into)
        })
        .await
    }

    async fn activate_lashlang_trigger_with_effect_scope(
        &self,
        handle: &str,
        payload: serde_json::Value,
        scoped_effect_controller: ScopedEffectController<'_>,
    ) -> Result<lash_core::HostEventEmitReport> {
        let writer = self.runtime.writer();
        let mut runtime = writer.lock().await;
        let value = runtime
            .activate_lashlang_trigger_with_effect_scope(handle, payload, scoped_effect_controller)
            .await
            .map_err(Into::into);
        self.runtime.publish_from(&runtime);
        value
    }

    async fn activate_lashlang_trigger_source_type(
        &self,
        source_type: impl AsRef<str>,
        payload: serde_json::Value,
    ) -> Result<lash_core::HostEventEmitReport> {
        let source_type = source_type.as_ref().to_string();
        self.with_writer(async |runtime: &mut LashRuntime| {
            runtime
                .activate_lashlang_trigger_source_type(&source_type, payload)
                .await
                .map_err(Into::into)
        })
        .await
    }

    async fn activate_lashlang_trigger_source_type_with_effect_scope(
        &self,
        source_type: impl AsRef<str>,
        payload: serde_json::Value,
        scoped_effect_controller: ScopedEffectController<'_>,
    ) -> Result<lash_core::HostEventEmitReport> {
        let source_type = source_type.as_ref().to_string();
        let writer = self.runtime.writer();
        let mut runtime = writer.lock().await;
        let value = runtime
            .activate_lashlang_trigger_source_type_with_effect_scope(
                &source_type,
                payload,
                scoped_effect_controller,
            )
            .await
            .map_err(Into::into);
        self.runtime.publish_from(&runtime);
        value
    }

    async fn list_lashlang_trigger_registrations(
        &self,
    ) -> Result<Vec<lash_core::TriggerRegistration>> {
        self.with_writer(async |runtime: &mut LashRuntime| {
            runtime
                .list_lashlang_trigger_registrations()
                .map_err(Into::into)
        })
        .await
    }

    async fn lashlang_trigger_registrations_by_source_type(
        &self,
        source_type: impl Into<lash_core::TriggerSourceType>,
    ) -> Result<Vec<lash_core::TriggerRegistration>> {
        self.with_writer(async |runtime: &mut LashRuntime| {
            runtime
                .lashlang_trigger_registrations_by_source_type(source_type)
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
        if !result.is_success() {
            return Err(EmbedError::Plugin(lash_core::PluginError::Invoke(format!(
                "{} failed: {}",
                Op::NAME,
                result.value_for_projection()
            ))));
        }
        serde_json::from_value(result.into_output().value_for_projection()).map_err(|err| {
            EmbedError::Plugin(lash_core::PluginError::Invoke(format!(
                "invalid {} output: {err}",
                Op::NAME
            )))
        })
    }

    async fn rewrite_history(&self, trigger: RewriteTrigger) -> Result<bool> {
        self.with_writer(async |runtime: &mut LashRuntime| {
            runtime.rewrite_history(trigger).await.map_err(Into::into)
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
    ) -> Result<lash_core::ProcessHandleSummary> {
        let writer = self.runtime.writer();
        let runtime = writer.lock().await;
        let session_id = runtime.session_id().to_string();
        let processes = runtime.process_service()?;
        let effect_host = runtime.effect_host();
        let scope = lash_core::ProcessOpScope::new(
            effect_host
                .scoped(lash_core::EffectScope::process(request.id.clone()))
                .map_err(EmbedError::from)?,
        );
        processes
            .start_from_request(&session_id, request, scope)
            .await
            .map_err(Into::into)
    }

    async fn session_state_service(&self) -> Result<Arc<dyn SessionStateService>> {
        self.runtime
            .writer()
            .lock()
            .await
            .session_state_service()
            .map_err(Into::into)
    }

    async fn cancel_process(&self, process_id: &str) -> Result<lash_core::ProcessCancelSummary> {
        let writer = self.runtime.writer();
        let runtime = writer.lock().await;
        let session_id = runtime.session_id().to_string();
        let processes = runtime.process_service()?;
        let cancel_ability = runtime.process_cancel_ability();
        let effect_host = runtime.effect_host();
        let scope = lash_core::ProcessOpScope::new(
            effect_host
                .scoped(lash_core::EffectScope::process(process_id.to_string()))
                .map_err(EmbedError::from)?,
        );
        cancel_ability
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
            .map_err(Into::into)
    }

    async fn cancel_visible_processes(&self) -> Result<Vec<lash_core::ProcessCancelSummary>> {
        let writer = self.runtime.writer();
        let runtime = writer.lock().await;
        let session_id = runtime.session_id().to_string();
        let processes = runtime.process_service()?;
        let cancel_ability = runtime.process_cancel_ability();
        let effect_host = runtime.effect_host();
        let scope = lash_core::ProcessOpScope::new(
            effect_host
                .scoped(lash_core::EffectScope::runtime_operation(format!(
                    "process:cancel-visible:{session_id}"
                )))
                .map_err(EmbedError::from)?,
        );
        cancel_ability
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
            .map_err(Into::into)
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

    async fn restore_tool_state(&self, state: ToolState) -> Result<u64> {
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

    async fn active_tool_definitions(&self) -> Result<Vec<ToolManifest>> {
        Ok(self.tool_state().await?.tool_manifests())
    }

    async fn add_tool_provider(&self, provider: Arc<dyn ToolProvider>) -> Result<ToolSourceHandle> {
        let tool_registry = self.tool_registry().await?;
        let handle = tool_registry
            .add_tool_provider(provider)
            .map_err(|err| EmbedError::Session(SessionError::Protocol(err.to_string())))?;
        self.refresh_tool_surface().await?;
        Ok(handle)
    }

    async fn remove_tool_source(&self, handle: &ToolSourceHandle) -> Result<u64> {
        let tool_registry = self.tool_registry().await?;
        let generation = tool_registry
            .remove_source(handle)
            .map_err(|err| EmbedError::Session(SessionError::Protocol(err.to_string())))?;
        self.refresh_tool_surface().await?;
        Ok(generation)
    }

    async fn create_child_session(&self, request: SessionCreateRequest) -> Result<SessionHandle> {
        let writer = self.runtime.writer();
        let runtime = writer.lock().await;
        let lifecycle = runtime.session_lifecycle_service()?;
        lifecycle.create_session(request).await.map_err(Into::into)
    }

    async fn start_child_turn(
        &self,
        session_id: &str,
        turn_id: &str,
        input: TurnInput,
    ) -> Result<AssembledTurn> {
        let (lifecycle, scoped_effect_controller) = {
            let writer = self.runtime.writer();
            let runtime = writer.lock().await;
            let lifecycle = runtime.session_lifecycle_service()?;
            let scoped_effect_controller = runtime
                .effect_host()
                .scoped_static(lash_core::EffectScope::turn(session_id, turn_id))
                .map_err(EmbedError::from)?
                .ok_or_else(|| {
                    EmbedError::Session(lash_core::SessionError::Protocol(
                        "child turn execution requires an effect host with static scoped controllers"
                            .to_string(),
                    ))
                })?;
            (lifecycle, scoped_effect_controller)
        };
        let request = lash_core::SessionTurnRequest::new(
            session_id,
            turn_id,
            input,
            scoped_effect_controller,
        )
        .map_err(EmbedError::from)?;
        lifecycle.start_turn(request).await.map_err(Into::into)
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
pub struct ConfigControl {
    control: SessionControl,
}

impl ConfigControl {
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
pub struct ToolsControl {
    control: SessionControl,
}

impl ToolsControl {
    pub(crate) fn new(control: SessionControl) -> Self {
        Self { control }
    }
}

impl ToolsControl {
    pub async fn state(&self) -> Result<ToolState> {
        self.control.tool_state().await
    }

    pub fn advanced(&self) -> AdvancedToolsControl {
        AdvancedToolsControl {
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

    pub async fn active_definitions(&self) -> Result<Vec<ToolManifest>> {
        self.control.active_tool_definitions().await
    }

    pub async fn add_provider(&self, provider: Arc<dyn ToolProvider>) -> Result<ToolSourceHandle> {
        self.control.add_tool_provider(provider).await
    }

    pub async fn remove_source(&self, handle: &ToolSourceHandle) -> Result<u64> {
        self.control.remove_tool_source(handle).await
    }
}

#[derive(Clone)]
pub struct AdvancedToolsControl {
    control: SessionControl,
}

impl AdvancedToolsControl {
    /// Replace the entire tool-state snapshot.
    ///
    /// This is a generation-checked escape hatch for hosts that intentionally
    /// edit the full snapshot. Prefer `ToolsControl` availability methods for
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
    pub async fn restore_state(&self, state: ToolState) -> Result<u64> {
        self.control.restore_tool_state(state).await
    }
}

#[derive(Clone)]
pub struct SessionCommandsControl {
    control: SessionControl,
}

impl SessionCommandsControl {
    pub async fn refresh_tool_surface(
        &self,
        reason: impl Into<String>,
        expected_generation: Option<u64>,
        idempotency_key: impl Into<String>,
    ) -> Result<lash_core::SessionCommandReceipt> {
        self.control
            .submit_session_command(
                lash_core::SessionCommand::RefreshToolSurface {
                    reason: reason.into(),
                    expected_generation,
                },
                idempotency_key,
            )
            .await
    }

    pub async fn emit_host_event(
        &self,
        resource_type: impl AsRef<str>,
        alias: impl AsRef<str>,
        event: impl AsRef<str>,
        payload: serde_json::Value,
        idempotency_key: impl Into<String>,
    ) -> Result<lash_core::SessionCommandReceipt> {
        self.control
            .submit_session_command(
                lash_core::SessionCommand::EmitHostEvent {
                    resource_type: resource_type.as_ref().to_string(),
                    alias: alias.as_ref().to_string(),
                    event: event.as_ref().to_string(),
                    payload,
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

/// Host controls for Lashlang trigger registrations.
///
/// Lash does not own trigger source schedulers. Hosts or plugins that own a
/// concrete source type can activate every enabled route for that source type,
/// while scheduler-like sources may still activate an exact selected handle.
#[derive(Clone)]
pub struct TriggersControl {
    control: SessionControl,
}

impl TriggersControl {
    /// Return every trigger registration in the session.
    ///
    /// This is an admin/introspection view. Source owners should prefer
    /// [`Self::by_source_type`] so they only inspect registrations for the
    /// concrete source type they own.
    pub async fn list_all(&self) -> Result<Vec<lash_core::TriggerRegistration>> {
        self.control.list_lashlang_trigger_registrations().await
    }

    /// Return registrations whose source value has the given host value type.
    ///
    /// This is the source-owner API: a timer, UI, webhook, or other host-owned
    /// source uses it to find the exact handles it may activate.
    pub async fn by_source_type(
        &self,
        source_type: impl Into<lash_core::TriggerSourceType>,
    ) -> Result<Vec<lash_core::TriggerRegistration>> {
        self.control
            .lashlang_trigger_registrations_by_source_type(source_type)
            .await
    }

    /// Activate one exact trigger handle with an event payload.
    ///
    /// The payload must match the event type declared for the registered source.
    /// Disabled or unknown handles do not start a process.
    pub async fn activate(
        &self,
        handle: impl AsRef<str>,
        payload: serde_json::Value,
    ) -> Result<lash_core::HostEventEmitReport> {
        self.control
            .activate_lashlang_trigger(handle.as_ref(), payload)
            .await
    }

    pub async fn activate_with_effect_scope(
        &self,
        handle: impl AsRef<str>,
        payload: serde_json::Value,
        scoped_effect_controller: ScopedEffectController<'_>,
    ) -> Result<lash_core::HostEventEmitReport> {
        self.control
            .activate_lashlang_trigger_with_effect_scope(
                handle.as_ref(),
                payload,
                scoped_effect_controller,
            )
            .await
    }

    /// Activate every enabled trigger route whose source value has this host
    /// value type.
    pub async fn activate_source_type(
        &self,
        source_type: impl AsRef<str>,
        payload: serde_json::Value,
    ) -> Result<lash_core::HostEventEmitReport> {
        self.control
            .activate_lashlang_trigger_source_type(source_type.as_ref(), payload)
            .await
    }

    pub async fn activate_source_type_with_effect_scope(
        &self,
        source_type: impl AsRef<str>,
        payload: serde_json::Value,
        scoped_effect_controller: ScopedEffectController<'_>,
    ) -> Result<lash_core::HostEventEmitReport> {
        self.control
            .activate_lashlang_trigger_source_type_with_effect_scope(
                source_type.as_ref(),
                payload,
                scoped_effect_controller,
            )
            .await
    }
}

#[derive(Clone)]
pub struct ProcessControl {
    control: SessionControl,
}

impl ProcessControl {
    pub(crate) fn new(control: SessionControl) -> Self {
        Self { control }
    }

    pub async fn start(
        &self,
        request: lash_core::ProcessStartRequest,
    ) -> Result<lash_core::ProcessHandleSummary> {
        self.control.start_process(request).await
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

    pub async fn cancel(&self, process_id: &str) -> Result<lash_core::ProcessCancelSummary> {
        self.control.cancel_process(process_id).await
    }

    pub async fn cancel_all(&self) -> Result<Vec<lash_core::ProcessCancelSummary>> {
        self.control.cancel_visible_processes().await
    }
}

#[derive(Clone)]
pub struct StateControl {
    control: SessionControl,
}

impl StateControl {
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

    pub async fn rewrite_history(&self, trigger: RewriteTrigger) -> Result<bool> {
        self.control.rewrite_history(trigger).await
    }
}

#[derive(Clone)]
pub struct PluginActions {
    pub(crate) control: SessionControl,
}

impl PluginActions {
    pub async fn call<Op: lash_core::PluginAction>(&self, args: Op::Args) -> Result<Op::Output> {
        self.control.call_plugin_action::<Op>(args).await
    }
}

#[derive(Clone)]
pub struct ChildrenControl {
    control: SessionControl,
}

impl ChildrenControl {
    pub async fn create_session(&self, request: SessionCreateRequest) -> Result<SessionHandle> {
        self.control.create_child_session(request).await
    }

    pub async fn start_turn(
        &self,
        session_id: &str,
        turn_id: &str,
        input: TurnInput,
    ) -> Result<AssembledTurn> {
        self.control
            .start_child_turn(session_id, turn_id, input)
            .await
    }

    pub async fn close_session(&self, session_id: &str) -> Result<()> {
        self.control.close_child_session(session_id).await
    }

    pub async fn activate_managed_session(&self, session_id: &str) -> Result<()> {
        self.control.activate_managed_session(session_id).await
    }
}

#[derive(Clone)]
pub struct InjectionControl {
    control: SessionControl,
}

impl InjectionControl {
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
pub struct ModeControl {
    control: SessionControl,
}

impl ModeControl {
    pub async fn apply_session_extension(
        &self,
        extension: lash_core::ProtocolSessionExtensionHandle,
    ) -> Result<()> {
        self.control
            .apply_protocol_session_extension(extension)
            .await
    }
}
