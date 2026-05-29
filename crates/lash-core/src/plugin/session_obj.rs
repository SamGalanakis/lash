use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use sha2::{Digest, Sha256};
use tokio::task::JoinSet;

use super::*;

mod directives;
mod tools;

async fn collect_owned_async<C, O, H, F>(
    hooks: &[RegisteredHook<H>],
    ctx: C,
    invoke: F,
) -> Result<Vec<PluginOwned<O>>, PluginError>
where
    C: Clone,
    F: Fn(&H, C) -> PluginFuture<Vec<O>>,
{
    let mut out = Vec::new();
    for registered in hooks {
        for value in invoke(&registered.hook, ctx.clone()).await? {
            out.push(PluginOwned {
                plugin_id: registered.plugin_id.clone(),
                value,
            });
        }
    }
    Ok(out)
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

    pub fn lashlang_resources(&self) -> lashlang::ResourceCatalog {
        self.lashlang_resources.clone()
    }

    pub fn host_events(&self) -> &crate::HostEventCatalog {
        &self.host_events
    }

    pub fn install_lashlang_trigger_source(
        &self,
        source: &str,
        surface: lashlang::LashlangSurface,
        artifact_store: &dyn lashlang::LashlangArtifactStore,
    ) -> Result<crate::SessionTriggerInstallReport, PluginError> {
        self.trigger_registry
            .install_lashlang_source(source, surface, artifact_store)
    }

    pub fn install_linked_lashlang_trigger_source(
        &self,
        source: &str,
        linked: &lashlang::LinkedModule,
        artifact_store: &dyn lashlang::LashlangArtifactStore,
    ) -> Result<crate::SessionTriggerInstallReport, PluginError> {
        self.trigger_registry
            .install_linked_lashlang_source(source, linked, artifact_store)
    }

    pub(crate) fn installed_lashlang_trigger_routes(
        &self,
    ) -> Result<Vec<SessionTriggerRoute>, PluginError> {
        self.trigger_registry.installed_routes()
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
        ctx: &TurnTransformContext,
        input: crate::session_model::context::PreparedContext,
    ) -> Result<crate::session_model::context::PreparedContext, HistoryError> {
        let mut current = input;
        for (_, transform) in &self.contributions.turn_context_transforms {
            current = transform.transform(ctx, current).await?;
        }
        Ok(current)
    }

    /// Chain registered history rewriters, skipping any that opt out of
    /// the current trigger via `accepts()`.
    pub async fn rewrite_history(
        &self,
        ctx: &RewriteContext,
        input: HistoryState,
    ) -> Result<HistoryState, HistoryError> {
        let mut current = input;
        for (_, rewriter) in &self.contributions.history_rewriters {
            if !rewriter.accepts(&ctx.trigger) {
                continue;
            }
            current = rewriter.rewrite(ctx, current).await?;
        }
        Ok(current)
    }

    pub async fn collect_prompt_contributions(
        &self,
        ctx: PromptHookContext,
    ) -> Result<Vec<PromptContribution>, PluginError> {
        let mut out =
            collect_owned_async(&self.contributions.prompt_contributors, ctx, |hook, ctx| {
                hook(ctx)
            })
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
        collect_owned_async(&self.contributions.before_turn_hooks, ctx, |hook, ctx| {
            hook(ctx)
        })
        .await
    }

    pub async fn before_tool_call(
        &self,
        ctx: ToolCallHookContext,
    ) -> Result<Vec<PluginOwned<PluginDirective>>, PluginError> {
        collect_owned_async(
            &self.contributions.before_tool_call_hooks,
            ctx,
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
            |hook, ctx| hook(ctx),
        )
        .await
    }

    pub async fn after_turn(
        &self,
        ctx: TurnResultHookContext,
    ) -> Result<Vec<PluginOwned<PluginDirective>>, PluginError> {
        collect_owned_async(&self.contributions.after_turn_hooks, ctx, |hook, ctx| {
            hook(ctx)
        })
        .await
    }

    pub async fn at_checkpoint(
        &self,
        ctx: CheckpointHookContext,
    ) -> Result<Vec<PluginOwned<PluginDirective>>, PluginError> {
        collect_owned_async(&self.contributions.checkpoint_hooks, ctx, |hook, ctx| {
            hook(ctx)
        })
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

    pub async fn emit_runtime_event(&self, event: PluginLifecycleEvent) {
        let mut tasks = JoinSet::new();
        for hook in &self.contributions.runtime_event_hooks {
            let hook = Arc::clone(hook);
            let event = event.clone();
            tasks.spawn(async move { hook(event).await });
        }

        while let Some(result) = tasks.join_next().await {
            match result {
                Ok(Ok(())) => {}
                Ok(Err(err)) => tracing::warn!("plugin runtime event hook failed: {err}"),
                Err(err) => tracing::warn!("plugin runtime event hook task failed: {err}"),
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

    pub async fn invoke_plugin_action(
        &self,
        name: &str,
        args: serde_json::Value,
        session_id: Option<String>,
        default_to_current_session: bool,
        host: Arc<dyn RuntimeSessionHost>,
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
                host,
                processes,
            },
            args,
        )
        .await)
    }

    pub async fn call_plugin_action<Op: PluginAction>(
        &self,
        args: Op::Args,
        session_id: Option<String>,
        default_to_current_session: bool,
        host: Arc<dyn RuntimeSessionHost>,
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
                host,
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
