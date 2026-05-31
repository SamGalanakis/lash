pub use crate::session::SessionConfigPatch;
use crate::support::*;
use lash_core::runtime::{QueuedWorkBatchDraft, QueuedWorkPayload};
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

    pub fn triggers(&self) -> TriggersControl {
        TriggersControl {
            control: self.clone(),
        }
    }

    pub fn host_events(&self) -> HostEventsControl {
        HostEventsControl {
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
            .await;
        Ok(())
    }

    async fn update_session_config(
        &self,
        provider: Option<ProviderHandle>,
        model: Option<lash_core::ModelSpec>,
        prompt: Option<PromptLayer>,
    ) {
        self.with_writer(async |runtime: &mut LashRuntime| {
            runtime.update_session_config(provider, model, prompt).await;
        })
        .await
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

    async fn install_lashlang_trigger_source(
        &self,
        source: &str,
    ) -> Result<lash_core::SessionTriggerInstallReport> {
        self.with_writer(async |runtime: &mut LashRuntime| {
            runtime
                .install_lashlang_trigger_source(source)
                .await
                .map_err(Into::into)
        })
        .await
    }

    async fn emit_host_event(
        &self,
        resource_type: &str,
        alias: &str,
        event: &str,
        payload: serde_json::Value,
    ) -> Result<lash_core::HostEventEmitReport> {
        self.with_writer(async |runtime: &mut LashRuntime| {
            runtime
                .emit_host_event(resource_type, alias, event, payload)
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

    async fn list_process_handles(&self) -> Result<Vec<ProcessHandleGrantEntry>> {
        Ok(self.runtime.observe().list_process_handles().await)
    }

    async fn list_all_process_handles(&self) -> Result<Vec<ProcessHandleGrantEntry>> {
        Ok(self.runtime.observe().list_all_process_handles().await)
    }

    async fn start_process(
        &self,
        registration: lash_core::ProcessRegistration,
        options: lash_core::ProcessStartOptions,
    ) -> Result<ProcessRecord> {
        let writer = self.runtime.writer();
        let runtime = writer.lock().await;
        let session_id = runtime.session_id().to_string();
        let processes = runtime.process_service()?;
        processes
            .start(
                &session_id,
                registration,
                options,
                lash_core::ProcessOpScope::new(),
            )
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

    async fn cancel_process(&self, process_id: &str) -> Result<ProcessRecord> {
        let writer = self.runtime.writer();
        let runtime = writer.lock().await;
        let session_id = runtime.session_id().to_string();
        let processes = runtime.process_service()?;
        processes
            .cancel(&session_id, process_id, lash_core::ProcessOpScope::new())
            .await
            .map_err(Into::into)
    }

    async fn cancel_all_processes(&self) -> Result<Vec<ProcessRecord>> {
        let writer = self.runtime.writer();
        let runtime = writer.lock().await;
        let session_id = runtime.session_id().to_string();
        let processes = runtime.process_service()?;
        processes
            .cancel_all(&session_id, lash_core::ProcessOpScope::new())
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

    async fn start_child_turn(&self, session_id: &str, input: TurnInput) -> Result<AssembledTurn> {
        let lifecycle = {
            let writer = self.runtime.writer();
            let runtime = writer.lock().await;
            runtime.session_lifecycle_service()?
        };
        lifecycle
            .start_turn(session_id, input)
            .await
            .map_err(Into::into)
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
        let observation = self.runtime.observe();
        let store = observation.queue_store.as_ref().ok_or_else(|| {
            EmbedError::Runtime(lash_core::RuntimeError::new(
                lash_core::RuntimeErrorCode::StoreCommitFailed,
                "queued turn input requires a persistent runtime store",
            ))
        })?;
        for input in messages {
            let source_key = input.id.map(|id| format!("injection:{id}"));
            let turn_input = turn_input_from_plugin_message(input.message);
            lash_core::ensure_durable_turn_input(&turn_input).map_err(EmbedError::Runtime)?;
            let mut draft = QueuedWorkBatchDraft::new(
                observation.session_id().to_string(),
                lash_core::DeliveryPolicy::EarliestSafeBoundary,
                lash_core::SlotPolicy::Join,
                vec![QueuedWorkPayload::turn_input(turn_input)],
            );
            draft.source_key = source_key;
            store.enqueue_queued_work(draft).await.map_err(|err| {
                EmbedError::Runtime(lash_core::RuntimeError::new(
                    lash_core::RuntimeErrorCode::StoreCommitFailed,
                    err.to_string(),
                ))
            })?;
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
    ) {
        self.control
            .update_session_config(provider, model, prompt)
            .await;
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
    /// Re-register the current tool registry with the live runtime session.
    ///
    /// Use this after lower-level code mutates tool providers outside this
    /// control surface. `add_provider` and `remove_source` refresh
    /// automatically.
    pub async fn refresh_surface(&self) -> Result<()> {
        self.control.refresh_tool_surface().await
    }

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
pub struct TriggersControl {
    control: SessionControl,
}

impl TriggersControl {
    /// Register trigger declarations from a Lashlang module.
    ///
    /// Foreground expressions in the module are ignored by this installer; they
    /// are only executed when the same Lashlang source is run through a turn.
    pub async fn install_lashlang_source(
        &self,
        source: impl AsRef<str>,
    ) -> Result<lash_core::SessionTriggerInstallReport> {
        self.control
            .install_lashlang_trigger_source(source.as_ref())
            .await
    }
}

#[derive(Clone)]
pub struct HostEventsControl {
    control: SessionControl,
}

impl HostEventsControl {
    pub async fn emit(
        &self,
        resource_type: impl AsRef<str>,
        alias: impl AsRef<str>,
        event: impl AsRef<str>,
        payload: serde_json::Value,
    ) -> Result<lash_core::HostEventEmitReport> {
        self.control
            .emit_host_event(
                resource_type.as_ref(),
                alias.as_ref(),
                event.as_ref(),
                payload,
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
        registration: lash_core::ProcessRegistration,
        options: lash_core::ProcessStartOptions,
    ) -> Result<ProcessRecord> {
        self.control.start_process(registration, options).await
    }

    pub async fn list(&self) -> Result<Vec<ProcessHandleGrantEntry>> {
        self.control.list_process_handles().await
    }

    pub async fn list_all(&self) -> Result<Vec<ProcessHandleGrantEntry>> {
        self.control.list_all_process_handles().await
    }

    pub async fn await_all(&self) -> Result<()> {
        self.control.await_background_work().await
    }

    pub async fn cancel(&self, process_id: &str) -> Result<ProcessRecord> {
        self.control.cancel_process(process_id).await
    }

    pub async fn cancel_all(&self) -> Result<Vec<ProcessRecord>> {
        self.control.cancel_all_processes().await
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

    pub async fn start_turn(&self, session_id: &str, input: TurnInput) -> Result<AssembledTurn> {
        self.control.start_child_turn(session_id, input).await
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
