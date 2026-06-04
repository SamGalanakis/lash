use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use futures_util::stream::{FuturesUnordered, StreamExt};
use sha2::{Digest, Sha256};

use super::*;

mod directives;
mod tools;

async fn collect_owned_async<C, O, H, F>(
    hooks: &[RegisteredHook<H>],
    ctx: C,
    hook_kind: &'static str,
    phase_probe: Option<&Arc<dyn crate::runtime::RuntimeTurnPhaseProbe>>,
    invoke: F,
) -> Result<Vec<PluginOwned<O>>, PluginError>
where
    C: Clone,
    F: Fn(&H, C) -> PluginFuture<Vec<O>>,
{
    let mut out = Vec::new();
    for registered in hooks {
        let phase_name = plugin_hook_phase_name(hook_kind, &registered.plugin_id);
        if let Some(probe) = phase_probe {
            probe.begin_named(&phase_name);
        }
        let result = invoke(&registered.hook, ctx.clone()).await;
        if let Some(probe) = phase_probe {
            probe.end_named(&phase_name);
        }
        for value in result? {
            out.push(PluginOwned {
                plugin_id: registered.plugin_id.clone(),
                value,
            });
        }
    }
    Ok(out)
}

fn plugin_hook_phase_name(hook_kind: &str, plugin_id: &str) -> String {
    format!("plugin_hook.{hook_kind}.{plugin_id}")
}

fn lifecycle_event_hook_kind(event: &PluginLifecycleEvent<'_>) -> &'static str {
    match event {
        PluginLifecycleEvent::TurnFinalized(_) => "turn_finalized",
        PluginLifecycleEvent::TurnPersisted(_) => "turn_persisted",
        PluginLifecycleEvent::SessionRestored(_) => "session_restored",
        PluginLifecycleEvent::SessionConfigChanged(_) => "session_config_changed",
    }
}

fn collect_owned_sync<C, O, H, F>(
    hooks: &[RegisteredHook<H>],
    ctx: C,
    invoke: F,
) -> Result<Vec<PluginOwned<O>>, PluginError>
where
    C: Clone,
    F: Fn(&H, C) -> Result<O, PluginError>,
{
    let mut out = Vec::new();
    for registered in hooks {
        out.push(PluginOwned {
            plugin_id: registered.plugin_id.clone(),
            value: invoke(&registered.hook, ctx.clone())?,
        });
    }
    Ok(out)
}

struct EmptySnapshotReader;

impl SnapshotReader for EmptySnapshotReader {
    fn read_blob(&self, _name: &str) -> Option<&[u8]> {
        None
    }
}

pub struct PluginSession {
    pub(super) host: PluginHost,
    pub(super) session_id: String,
    pub(super) plugins: Vec<Arc<dyn SessionPlugin>>,
    pub(super) tools: Arc<dyn ToolProvider>,
    pub(super) tool_registry: Arc<crate::ToolRegistry>,
    pub(super) tool_surface_overlay: ToolSurfaceContribution,
    pub(super) tool_access: SessionToolAccess,
    pub(super) subagent: Option<SubagentSessionContext>,
    pub(super) lashlang_abilities: lashlang::LashlangAbilities,
    pub(super) lashlang_language_features: lashlang::LashlangLanguageFeatures,
    pub(super) lashlang_resources: lashlang::ResourceCatalog,
    pub(super) host_events: crate::HostEventCatalog,
    pub(super) trigger_registry: Arc<SessionTriggerRegistry>,
    pub(super) contributions: PluginContributions,
}
impl PluginSession {
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    pub fn tool_access(&self) -> &SessionToolAccess {
        &self.tool_access
    }

    pub fn subagent_context(&self) -> Option<&SubagentSessionContext> {
        self.subagent.as_ref()
    }

    pub fn lashlang_abilities(&self) -> lashlang::LashlangAbilities {
        self.lashlang_abilities
    }

    pub fn lashlang_language_features(&self) -> lashlang::LashlangLanguageFeatures {
        self.lashlang_language_features
    }

    pub fn lashlang_resources(&self) -> lashlang::ResourceCatalog {
        self.lashlang_resources.clone()
    }

    pub fn host_events(&self) -> &crate::HostEventCatalog {
        &self.host_events
    }

    pub fn register_lashlang_trigger(
        &self,
        request: serde_json::Value,
        artifact_store: Arc<dyn lashlang::LashlangArtifactStore>,
    ) -> Result<serde_json::Value, PluginError> {
        let route = self.trigger_registry.register_route(
            request,
            &self.lashlang_resources,
            artifact_store.as_ref(),
        )?;
        Ok(super::trigger_registry::trigger_handle_json(&route.handle))
    }

    pub fn list_lashlang_triggers(
        &self,
        request: serde_json::Value,
    ) -> Result<serde_json::Value, PluginError> {
        serde_json::to_value(self.trigger_registry.list(request)?).map_err(|err| {
            PluginError::Session(format!("failed to encode trigger registrations: {err}"))
        })
    }

    pub fn list_all_lashlang_triggers(&self) -> Result<Vec<TriggerRegistration>, PluginError> {
        self.trigger_registry.list_all()
    }

    pub fn lashlang_trigger_registrations_by_source_type(
        &self,
        source_type: TriggerSourceType,
    ) -> Result<Vec<TriggerRegistration>, PluginError> {
        self.trigger_registry.routes_by_source_type(&source_type)
    }

    pub fn cancel_lashlang_trigger(
        &self,
        request: serde_json::Value,
    ) -> Result<serde_json::Value, PluginError> {
        let changed = self.trigger_registry.cancel(request)?;
        Ok(serde_json::json!(changed))
    }

    pub(crate) fn trigger_activation_service<'a>(
        &'a self,
        processes: Arc<dyn crate::ProcessService>,
        scoped_effect_controller: crate::ScopedEffectController<'a>,
    ) -> crate::TriggerActivationService<'a> {
        crate::TriggerActivationService::new(
            self.session_id.clone(),
            Arc::clone(&self.trigger_registry),
            processes,
            scoped_effect_controller,
        )
    }

    pub fn host(&self) -> &PluginHost {
        &self.host
    }

    pub fn tools(&self) -> Arc<dyn ToolProvider> {
        Arc::clone(&self.tools)
    }

    pub fn tool_registry(&self) -> Arc<crate::ToolRegistry> {
        Arc::clone(&self.tool_registry)
    }

    pub(crate) fn protocol_session(&self) -> &Arc<dyn ProtocolSessionPlugin> {
        &self
            .contributions
            .protocol_session
            .as_ref()
            .expect("plugin session must have a protocol session")
            .hook
    }

    pub(crate) fn code_executor(&self) -> Option<Arc<dyn CodeExecutorPlugin>> {
        self.contributions
            .code_executor
            .as_ref()
            .map(|entry| Arc::clone(&entry.hook))
    }

    pub(crate) fn assistant_prose_projector(
        &self,
    ) -> Option<Arc<dyn AssistantProseProjectorPlugin>> {
        self.contributions
            .assistant_prose_projector
            .as_ref()
            .map(|entry| Arc::clone(&entry.hook))
    }

    pub fn protocol_driver(&self) -> Arc<dyn ProtocolDriverPlugin> {
        self.contributions
            .protocol_driver
            .as_ref()
            .map(|entry| Arc::clone(&entry.hook))
            .expect("plugin session must have a protocol driver")
    }

    pub fn plugin_actions(&self) -> Vec<PluginActionDef> {
        self.contributions
            .plugin_actions
            .values()
            .map(|op| op.def.clone())
            .collect()
    }

    pub fn has_assistant_stream_hooks(&self) -> bool {
        !self.contributions.assistant_stream_hooks.is_empty()
    }

    /// Chain registered turn-context transforms, piping each one's output
    /// into the next in priority order.
    pub async fn prepare_turn_context(
        &self,
        ctx: &TurnTransformContext<'_>,
        input: crate::session_model::context::PreparedContext,
        phase_probe: Option<Arc<dyn crate::runtime::RuntimeTurnPhaseProbe>>,
    ) -> Result<crate::session_model::context::PreparedContext, HistoryError> {
        let mut current = input;
        for (_, registered) in &self.contributions.turn_context_transforms {
            let phase_name =
                plugin_hook_phase_name("context_transform", registered.plugin_id.as_str());
            if let Some(probe) = phase_probe.as_ref() {
                probe.begin_named(&phase_name);
            }
            let result = registered.hook.transform(ctx, current).await;
            if let Some(probe) = phase_probe.as_ref() {
                probe.end_named(&phase_name);
            }
            current = result?;
        }
        Ok(current)
    }

    /// Chain registered history rewriters, skipping any that opt out of
    /// the current trigger via `accepts()`.
    pub async fn rewrite_history(
        &self,
        ctx: &RewriteContext<'_>,
        input: HistoryState,
    ) -> Result<HistoryState, HistoryError> {
        let mut current = input;
        for (_, registered) in &self.contributions.history_rewriters {
            if !registered.hook.accepts(&ctx.trigger) {
                continue;
            }
            current = registered.hook.rewrite(ctx, current).await?;
        }
        Ok(current)
    }

    pub async fn collect_prompt_contributions(
        &self,
        ctx: PromptHookContext,
    ) -> Result<Vec<PromptContribution>, PluginError> {
        let mut out = collect_owned_async(
            &self.contributions.prompt_contributors,
            ctx,
            "prompt_contributor",
            None,
            |hook, ctx| hook(ctx),
        )
        .await?
        .into_iter()
        .map(|owned| owned.value)
        .collect::<Vec<_>>();
        let mut seen = BTreeSet::new();
        out.retain(|contribution| {
            seen.insert((
                format!("{:?}", contribution.slot),
                contribution.priority,
                contribution.content.trim().to_string(),
            ))
        });
        out.sort_by(|a, b| {
            format!("{:?}", a.slot)
                .cmp(&format!("{:?}", b.slot))
                .then(a.priority.cmp(&b.priority))
        });
        Ok(out)
    }

    pub async fn before_turn(
        &self,
        ctx: TurnHookContext,
    ) -> Result<Vec<PluginOwned<PluginDirective>>, PluginError> {
        self.before_turn_with_phase_probe(ctx, None).await
    }

    async fn before_turn_with_phase_probe(
        &self,
        ctx: TurnHookContext,
        phase_probe: Option<&Arc<dyn crate::runtime::RuntimeTurnPhaseProbe>>,
    ) -> Result<Vec<PluginOwned<PluginDirective>>, PluginError> {
        collect_owned_async(
            &self.contributions.before_turn_hooks,
            ctx,
            "before_turn",
            phase_probe,
            |hook, ctx| hook(ctx),
        )
        .await
    }

    pub async fn before_tool_call(
        &self,
        ctx: ToolCallHookContext,
    ) -> Result<Vec<PluginOwned<PluginDirective>>, PluginError> {
        collect_owned_async(
            &self.contributions.before_tool_call_hooks,
            ctx,
            "before_tool_call",
            None,
            |hook, ctx| hook(ctx),
        )
        .await
    }

    pub async fn after_tool_call(
        &self,
        ctx: ToolResultHookContext,
    ) -> Result<Vec<PluginOwned<PluginDirective>>, PluginError> {
        collect_owned_async(
            &self.contributions.after_tool_call_hooks,
            ctx,
            "after_tool_call",
            None,
            |hook, ctx| hook(ctx),
        )
        .await
    }

    pub async fn after_turn(
        &self,
        ctx: TurnResultHookContext,
    ) -> Result<Vec<PluginOwned<PluginDirective>>, PluginError> {
        self.after_turn_with_phase_probe(ctx, None).await
    }

    async fn after_turn_with_phase_probe(
        &self,
        ctx: TurnResultHookContext,
        phase_probe: Option<&Arc<dyn crate::runtime::RuntimeTurnPhaseProbe>>,
    ) -> Result<Vec<PluginOwned<PluginDirective>>, PluginError> {
        collect_owned_async(
            &self.contributions.after_turn_hooks,
            ctx,
            "after_turn",
            phase_probe,
            |hook, ctx| hook(ctx),
        )
        .await
    }

    pub async fn at_checkpoint(
        &self,
        ctx: CheckpointHookContext,
    ) -> Result<Vec<PluginOwned<PluginDirective>>, PluginError> {
        collect_owned_async(
            &self.contributions.checkpoint_hooks,
            ctx,
            "checkpoint",
            None,
            |hook, ctx| hook(ctx),
        )
        .await
    }

    pub async fn transform_assistant_stream(
        &self,
        session_id: &str,
        chunk: String,
    ) -> Result<Vec<PluginOwned<AssistantStreamTransform>>, PluginError> {
        let mut current = chunk;
        let mut transforms = Vec::new();
        for registered in &self.contributions.assistant_stream_hooks {
            let transform = (registered.hook)(AssistantStreamHookContext {
                session_id: session_id.to_string(),
                chunk: current.clone(),
            })
            .await?;
            current = transform.chunk.clone();
            transforms.push(PluginOwned {
                plugin_id: registered.plugin_id.clone(),
                value: transform,
            });
        }
        Ok(transforms)
    }

    pub async fn transform_assistant_response(
        &self,
        session_id: &str,
        response: crate::llm::types::LlmResponse,
    ) -> Result<Vec<PluginOwned<AssistantResponseTransform>>, PluginError> {
        let mut current = response;
        let mut transforms = Vec::new();
        for registered in &self.contributions.assistant_response_hooks {
            let transform = (registered.hook)(AssistantResponseHookContext {
                session_id: session_id.to_string(),
                response: current.clone(),
            })
            .await?;
            current = transform.response.clone();
            transforms.push(PluginOwned {
                plugin_id: registered.plugin_id.clone(),
                value: transform,
            });
        }
        Ok(transforms)
    }

    pub async fn project_tool_result(
        &self,
        ctx: ToolResultProjectionContext,
    ) -> Result<crate::ModelToolReturn, PluginError> {
        let Some(projector) = &self.contributions.tool_result_projector else {
            return Ok(crate::ModelToolReturn::from_output(
                ctx.call_id.clone(),
                ctx.tool_name.clone(),
                &ctx.output,
            ));
        };
        (projector.hook)(ctx).await
    }

    pub async fn emit_runtime_event(&self, event: PluginLifecycleEvent<'_>) {
        self.emit_runtime_event_with_phase_probe(event, None).await;
    }

    pub async fn emit_runtime_event_with_phase_probe(
        &self,
        event: PluginLifecycleEvent<'_>,
        phase_probe: Option<Arc<dyn crate::runtime::RuntimeTurnPhaseProbe>>,
    ) {
        let hook_kind = lifecycle_event_hook_kind(&event);
        let mut pending = FuturesUnordered::new();
        for registered in &self.contributions.runtime_event_hooks {
            let hook = Arc::clone(&registered.hook);
            let plugin_id = registered.plugin_id.clone();
            let phase_name = plugin_hook_phase_name(hook_kind, registered.plugin_id.as_str());
            let event = event.clone();
            let phase_probe = phase_probe.clone();
            pending.push(async move {
                if let Some(probe) = phase_probe.as_ref() {
                    probe.begin_named(&phase_name);
                }
                let result = hook(event).await;
                if let Some(probe) = phase_probe.as_ref() {
                    probe.end_named(&phase_name);
                }
                (plugin_id, result)
            });
        }
        while let Some((plugin_id, result)) = pending.next().await {
            if let Err(err) = result {
                tracing::warn!(plugin_id, "plugin runtime event hook failed: {err}");
            }
        }
    }

    pub fn has_runtime_event_hooks(&self) -> bool {
        !self.contributions.runtime_event_hooks.is_empty()
    }

    pub async fn mutate_session_config(
        &self,
        ctx: SessionConfigChangedContext,
        mut policy: SessionPolicy,
    ) -> SessionPolicy {
        for hook in &self.contributions.session_config_mutators {
            match hook(ctx.clone(), policy.clone()).await {
                Ok(next_policy) => policy = next_policy,
                Err(err) => tracing::warn!("plugin config mutator failed: {err}"),
            }
        }
        policy
    }

    pub fn snapshot(&self) -> Result<PluginSessionSnapshot, PluginError> {
        let mut plugins = BTreeMap::new();
        for plugin in &self.plugins {
            let mut writer = InMemorySnapshotWriter::default();
            let meta = plugin.snapshot(&mut writer)?;
            plugins.insert(
                plugin.id().to_string(),
                PluginSnapshotEntry {
                    meta,
                    artifacts: writer.finish(),
                },
            );
        }
        Ok(PluginSessionSnapshot { plugins })
    }

    pub fn snapshot_is_current(&self, previous: Option<&PluginSessionSnapshot>) -> bool {
        let Some(previous) = previous else {
            return false;
        };
        if previous.plugins.len() != self.plugins.len() {
            return false;
        }
        for plugin in &self.plugins {
            let Some(entry) = previous.plugins.get(plugin.id()) else {
                return false;
            };
            if entry.meta.plugin_version != plugin.version()
                || entry.meta.revision != plugin.snapshot_revision()
            {
                return false;
            }
        }
        true
    }

    pub fn snapshot_revision_fingerprint(&self) -> u64 {
        let mut hasher = Sha256::new();
        for plugin in &self.plugins {
            hasher.update(plugin.id().as_bytes());
            hasher.update([0]);
            hasher.update(plugin.version().as_bytes());
            hasher.update([0]);
            hasher.update(plugin.snapshot_revision().to_le_bytes());
            hasher.update([0xff]);
        }
        let digest = hasher.finalize();
        u64::from_le_bytes(digest[..8].try_into().expect("digest prefix"))
    }

    pub fn restore(&self, snapshot: &PluginSessionSnapshot) -> Result<(), PluginError> {
        for plugin in &self.plugins {
            if let Some(entry) = snapshot.plugins.get(plugin.id()) {
                let reader = InMemorySnapshotReader { entry };
                plugin.restore(&entry.meta, &reader)?;
            } else {
                plugin.restore(
                    &PluginSnapshotMeta {
                        plugin_id: plugin.id().to_string(),
                        plugin_version: plugin.version().to_string(),
                        revision: plugin.snapshot_revision(),
                        state: None,
                    },
                    &EmptySnapshotReader,
                )?;
            }
        }
        Ok(())
    }

    pub fn fork_for_session(
        &self,
        session_id: impl Into<String>,
    ) -> Result<Arc<PluginSession>, PluginError> {
        let snapshot = self.snapshot()?;
        self.host.build_session_with_surface(
            session_id,
            Some(&snapshot),
            self.tool_surface_overlay.clone(),
            Some(self.tool_registry.export_state()),
        )
    }

    pub fn fork_for_child_session(
        &self,
        session_id: impl Into<String>,
        parent_session_id: Option<String>,
        authority: super::SessionAuthorityContext,
    ) -> Result<Arc<PluginSession>, PluginError> {
        let snapshot = self.snapshot()?;
        self.host.build_session_with_parent_and_surface(
            session_id,
            parent_session_id,
            Some(&snapshot),
            self.tool_surface_overlay.clone(),
            Some(self.tool_registry.export_state()),
            authority,
        )
    }

    pub fn fork_for_session_with_tool_surface(
        &self,
        session_id: impl Into<String>,
        tool_surface_overlay: ToolSurfaceContribution,
    ) -> Result<Arc<PluginSession>, PluginError> {
        let snapshot = self.snapshot()?;
        self.host.build_session_with_surface(
            session_id,
            Some(&snapshot),
            tool_surface_overlay,
            Some(self.tool_registry.export_state()),
        )
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "plugin action invocation carries the explicit host services exposed to actions"
    )]
    pub async fn invoke_plugin_action(
        &self,
        name: &str,
        args: serde_json::Value,
        session_id: Option<String>,
        default_to_current_session: bool,
        sessions: Arc<dyn SessionStateService>,
        session_lifecycle: Arc<dyn SessionLifecycleService>,
        session_graph: Arc<dyn SessionGraphService>,
        processes: Arc<dyn crate::ProcessService>,
    ) -> Result<ToolResult, PluginActionInvokeError> {
        let Some(op) = self.contributions.plugin_actions.get(name).cloned() else {
            return Err(PluginActionInvokeError::Unknown(name.to_string()));
        };

        let effective_session = session_id.or_else(|| {
            if default_to_current_session && !self.session_id.is_empty() {
                Some(self.session_id.clone())
            } else {
                None
            }
        });

        match (op.def.session_param, effective_session.as_ref()) {
            (SessionParam::Required, None) => {
                return Err(PluginActionInvokeError::MissingSession(name.to_string()));
            }
            (SessionParam::Forbidden, Some(_)) => {
                return Err(PluginActionInvokeError::UnexpectedSession(name.to_string()));
            }
            _ => {}
        }

        Ok((op.handler)(
            PluginActionContext {
                session_id: effective_session,
                sessions,
                session_lifecycle,
                session_graph,
                processes,
            },
            args,
        )
        .await)
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "typed action invocation mirrors the raw plugin action host service boundary"
    )]
    pub async fn call_plugin_action<Op: PluginAction>(
        &self,
        args: Op::Args,
        session_id: Option<String>,
        default_to_current_session: bool,
        sessions: Arc<dyn SessionStateService>,
        session_lifecycle: Arc<dyn SessionLifecycleService>,
        session_graph: Arc<dyn SessionGraphService>,
        processes: Arc<dyn crate::ProcessService>,
    ) -> Result<Op::Output, PluginError> {
        let args = serde_json::to_value(args)
            .map_err(|err| PluginError::Invoke(format!("invalid {} args: {err}", Op::NAME)))?;
        let result = self
            .invoke_plugin_action(
                Op::NAME,
                args,
                session_id,
                default_to_current_session,
                sessions,
                session_lifecycle,
                session_graph,
                processes,
            )
            .await
            .map_err(|err| PluginError::Invoke(err.to_string()))?;
        if !result.is_success() {
            return Err(PluginError::Invoke(format!(
                "{} failed: {}",
                Op::NAME,
                result.value_for_projection()
            )));
        }
        serde_json::from_value(result.into_output().value_for_projection())
            .map_err(|err| PluginError::Invoke(format!("invalid {} output: {err}", Op::NAME)))
    }
}
