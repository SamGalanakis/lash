use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use sha2::{Digest, Sha256};
use tokio::task::JoinSet;

use super::*;
use crate::session_model::plugin_message_to_message;

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

fn merge_string_array(
    obj: &mut serde_json::Map<String, serde_json::Value>,
    key: &str,
    values: Vec<String>,
) {
    let mut existing = obj
        .remove(key)
        .and_then(|value| value.as_array().cloned())
        .unwrap_or_default()
        .into_iter()
        .filter_map(|value| value.as_str().map(str::to_string))
        .collect::<BTreeSet<_>>();
    existing.extend(
        values
            .into_iter()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty()),
    );
    if !existing.is_empty() {
        obj.insert(key.to_string(), serde_json::json!(existing));
    }
}

fn apply_tool_discovery_contributions(
    catalog: &mut [serde_json::Value],
    contributions: impl IntoIterator<Item = ToolDiscoveryContribution>,
) {
    let mut by_name = BTreeMap::new();
    for (idx, tool) in catalog.iter().enumerate() {
        if let Some(name) = tool.get("name").and_then(serde_json::Value::as_str) {
            by_name.insert(name.to_string(), idx);
        }
    }

    for contribution in contributions {
        for patch in contribution.tools {
            let Some(idx) = by_name.get(&patch.tool_name).copied() else {
                continue;
            };
            let Some(obj) = catalog[idx].as_object_mut() else {
                continue;
            };
            if let Some(namespace) = patch
                .namespace
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty())
            {
                obj.insert("namespace".to_string(), serde_json::json!(namespace));
            }
            merge_string_array(obj, "aliases", patch.aliases);
        }
    }
}

fn append_plugin_messages(
    messages: &mut crate::MessageSequence,
    plugin_messages: &[PluginMessage],
) {
    let new_messages = plugin_messages
        .iter()
        .filter(|message| matches!(message.role, MessageRole::User | MessageRole::System))
        .map(plugin_message_to_message)
        .collect::<Vec<_>>();
    if !new_messages.is_empty() {
        messages.extend(new_messages);
    }
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
    pub(super) execution_mode: ExecutionMode,
    pub(super) plugins: Vec<Arc<dyn SessionPlugin>>,
    pub(super) tools: Arc<dyn ToolProvider>,
    pub(super) tool_registry: Arc<crate::ToolRegistry>,
    pub(super) tool_surface_overlay: ToolSurfaceContribution,
    pub(super) tool_access: SessionToolAccess,
    pub(super) subagent: Option<SubagentSessionAuthority>,
    pub(super) prompt_contributors: Vec<RegisteredHook<PromptContributor>>,
    pub(super) tool_surface_contributors: Vec<RegisteredHook<ToolSurfaceContributor>>,
    pub(super) tool_discovery_contributors: Vec<RegisteredHook<ToolDiscoveryContributor>>,
    pub(super) before_turn_hooks: Vec<RegisteredHook<BeforeTurnHook>>,
    pub(super) before_tool_call_hooks: Vec<RegisteredHook<BeforeToolCallHook>>,
    pub(super) after_tool_call_hooks: Vec<RegisteredHook<AfterToolCallHook>>,
    pub(super) after_turn_hooks: Vec<RegisteredHook<AfterTurnHook>>,
    pub(super) checkpoint_hooks: Vec<RegisteredHook<CheckpointHook>>,
    pub(super) assistant_stream_hooks: Vec<RegisteredHook<AssistantStreamHook>>,
    pub(super) assistant_response_hooks: Vec<RegisteredHook<AssistantResponseHook>>,
    pub(super) tool_result_projector: Option<RegisteredExclusiveHook<ToolResultProjector>>,
    pub(super) runtime_event_hooks: Vec<PluginRuntimeEventHook>,
    pub(super) session_config_mutators: Vec<SessionConfigMutator>,
    pub(super) plugin_actions: BTreeMap<String, RegisteredPluginAction>,
    pub(super) monitor_specs: Vec<PluginOwned<crate::MonitorSpec>>,
    pub(super) turn_context_transforms: Vec<Arc<dyn TurnContextTransform>>,
    pub(super) history_rewriters: Vec<Arc<dyn HistoryRewriter>>,
    pub(super) mode_session: Arc<dyn ModeSessionPlugin>,
    pub(super) mode_native_tools: Vec<Arc<dyn ModeNativeToolsPlugin>>,
    pub(super) mode_protocol_driver: Option<Arc<dyn ModeProtocolDriverPlugin>>,
}
impl PluginSession {
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    pub fn execution_mode(&self) -> ExecutionMode {
        self.execution_mode.clone()
    }

    pub fn tool_access(&self) -> &SessionToolAccess {
        &self.tool_access
    }

    pub fn subagent_authority(&self) -> Option<&SubagentSessionAuthority> {
        self.subagent.as_ref()
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

    pub(crate) fn mode_session(&self) -> &Arc<dyn ModeSessionPlugin> {
        &self.mode_session
    }

    pub(crate) fn mode_native_tools(&self) -> &[Arc<dyn ModeNativeToolsPlugin>] {
        &self.mode_native_tools
    }

    pub(crate) fn mode_native_tool_manifests(&self) -> Vec<ToolManifest> {
        self.mode_native_tools
            .iter()
            .flat_map(|provider| provider.tool_manifests())
            .collect()
    }

    /// Plugin-registered protocol driver for this session, if any plugin
    /// claimed the singleton slot. When `None`, callers fall back to
    /// `lash_sansio::build_mode_preamble` (hardcoded Standard/RLM).
    pub fn mode_protocol_driver(&self) -> Option<Arc<dyn ModeProtocolDriverPlugin>> {
        self.mode_protocol_driver.clone()
    }

    pub fn tool_surface(&self, session_id: &str, mode: ExecutionMode) -> Arc<crate::ToolSurface> {
        let mut tools = self.tools.tool_manifests();
        let contract_provider = Arc::clone(&self.tools);
        let native_contract_providers = self.mode_native_tools.to_vec();
        let resolve_contract: lash_sansio::ToolContractResolver = Arc::new(move |name: &str| {
            contract_provider.resolve_contract(name).or_else(|| {
                native_contract_providers
                    .iter()
                    .find_map(|provider| provider.resolve_contract(name))
            })
        });
        if mode == self.execution_mode {
            let native_tools = self.mode_native_tool_manifests();
            tools.extend(native_tools);
        }
        match self.resolve_tool_surface(ToolSurfaceContext {
            session_id: session_id.to_string(),
            mode: mode.clone(),
            tools,
            resolve_contract: Some(Arc::clone(&resolve_contract)),
            tool_access: self.tool_access.clone(),
            subagent: self.subagent.clone(),
        }) {
            Ok(surface) => Arc::new(surface),
            Err(err) => {
                tracing::warn!("failed to resolve tool surface: {err}");
                let mut fallback_tools = self.tools.tool_manifests();
                if mode == self.execution_mode {
                    let native_tools = self.mode_native_tool_manifests();
                    fallback_tools.extend(native_tools);
                }
                Arc::new(crate::build_tool_surface(crate::ToolSurfaceBuildInput {
                    tools: fallback_tools,
                    mode,
                    resolve_contract: Some(resolve_contract),
                    contributions: Vec::new(),
                }))
            }
        }
    }

    pub fn tool_catalog(&self, session_id: &str, mode: ExecutionMode) -> Vec<serde_json::Value> {
        let surface = self.tool_surface(session_id, mode.clone());
        let mut catalog =
            crate::tool_registry::project_tool_catalog(surface.searchable_tools_iter().cloned());
        let contributions = collect_owned_sync(
            &self.tool_discovery_contributors,
            ToolDiscoveryContext {
                session_id: session_id.to_string(),
                mode,
                catalog: catalog.clone(),
            },
            |hook, ctx| hook(ctx),
        )
        .unwrap_or_else(|err| {
            tracing::warn!("failed to resolve tool discovery metadata: {err}");
            Vec::new()
        });
        apply_tool_discovery_contributions(
            &mut catalog,
            contributions.into_iter().map(|owned| owned.value),
        );
        catalog
    }

    pub fn resolve_tool_surface(
        &self,
        ctx: ToolSurfaceContext,
    ) -> Result<crate::ToolSurface, PluginError> {
        let mut contributions = collect_owned_sync(
            &self.tool_surface_contributors,
            ToolSurfaceContext {
                session_id: ctx.session_id.clone(),
                mode: ctx.mode.clone(),
                tools: ctx.tools.clone(),
                resolve_contract: ctx.resolve_contract.clone(),
                tool_access: ctx.tool_access.clone(),
                subagent: ctx.subagent.clone(),
            },
            |hook, ctx| hook(ctx),
        )?
        .into_iter()
        .map(|owned| owned.value)
        .collect::<Vec<_>>();
        contributions.push(self.tool_surface_overlay.clone());
        let (tools, resolve_contract) = if ctx.tool_access.tools.is_empty() {
            (ctx.tools, ctx.resolve_contract)
        } else {
            let contracts = ctx
                .tool_access
                .tools
                .iter()
                .map(|tool| (tool.name.clone(), Arc::new(tool.contract())))
                .collect::<BTreeMap<_, _>>();
            (
                ctx.tool_access
                    .tools
                    .iter()
                    .map(|tool| tool.manifest())
                    .collect(),
                Some(Arc::new(move |name: &str| contracts.get(name).cloned())
                    as lash_sansio::ToolContractResolver),
            )
        };
        let authority_hidden_tools = tools
            .iter()
            .filter(|tool| ctx.tool_access.hides(&tool.name))
            .map(|tool| tool.name.clone())
            .collect::<BTreeSet<_>>();
        if !authority_hidden_tools.is_empty() {
            contributions.push(ToolSurfaceContribution {
                overrides: authority_hidden_tools
                    .into_iter()
                    .map(|tool_name| ToolSurfaceOverride {
                        tool_name,
                        availability: Some(crate::ToolAvailability::Off),
                    })
                    .collect(),
                ..Default::default()
            });
        }
        Ok(crate::build_tool_surface(crate::ToolSurfaceBuildInput {
            tools,
            mode: ctx.mode,
            resolve_contract,
            contributions,
        }))
    }

    pub fn plugin_actions(&self) -> Vec<PluginActionDef> {
        self.plugin_actions
            .values()
            .map(|op| op.def.clone())
            .collect()
    }

    pub fn monitor_specs(&self) -> &[PluginOwned<crate::MonitorSpec>] {
        self.monitor_specs.as_slice()
    }

    pub fn has_assistant_stream_hooks(&self) -> bool {
        !self.assistant_stream_hooks.is_empty()
    }

    /// Chain registered turn-context transforms, piping each one's output
    /// into the next in priority order.
    pub async fn prepare_turn_context(
        &self,
        ctx: &TurnTransformContext,
        input: crate::session_model::context::PreparedContext,
    ) -> Result<crate::session_model::context::PreparedContext, HistoryError> {
        let mut current = input;
        for transform in &self.turn_context_transforms {
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
        for rewriter in &self.history_rewriters {
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
        let mut out = collect_owned_async(&self.prompt_contributors, ctx, |hook, ctx| hook(ctx))
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

    async fn apply_turn_directives(
        &self,
        directives: Vec<PluginOwned<PluginDirective>>,
        mut messages: crate::MessageSequence,
        host: Arc<dyn TurnHookHost>,
        allow_abort: bool,
        invalid_context: &'static str,
    ) -> Result<TurnPreparation, PluginError> {
        let mut events = Vec::new();
        let mut abort = None;

        for emitted in directives {
            match emitted.value {
                PluginDirective::AbortTurn { code, message } => {
                    if !allow_abort {
                        return Err(PluginError::Session(invalid_context.to_string()));
                    }
                    abort = Some(PluginAbort { code, message });
                }
                PluginDirective::EnqueueMessages {
                    messages: plugin_messages,
                } => append_plugin_messages(&mut messages, &plugin_messages),
                PluginDirective::CreateSession { request } => {
                    host.create_session(*request)
                        .await
                        .map_err(|err| PluginError::Session(err.to_string()))?;
                }
                PluginDirective::HandoffSession { .. } => {
                    return Err(PluginError::Session(invalid_context.to_string()));
                }
                PluginDirective::EmitEvents { events: surface } => {
                    events.extend(crate::plugin::plugin_surface_session_events(
                        &emitted.plugin_id,
                        surface,
                    ));
                }
                PluginDirective::EmitTrace {
                    name,
                    payload,
                    context,
                } => {
                    host.emit_trace_event(
                        *context,
                        lash_trace::TraceEvent::Custom {
                            name: format!("plugin.{}.{}", emitted.plugin_id, name),
                            payload,
                        },
                    )
                    .await?;
                }
                PluginDirective::ReplaceToolArgs { .. }
                | PluginDirective::ShortCircuitTool { .. } => {
                    return Err(PluginError::Session(invalid_context.to_string()));
                }
            }
        }

        Ok(TurnPreparation {
            messages,
            events,
            abort,
        })
    }

    pub async fn prepare_turn(
        &self,
        request: PrepareTurnRequest,
    ) -> Result<TurnPreparation, PluginError> {
        let PrepareTurnRequest {
            session_id,
            state,
            messages,
            host,
            turn_context,
        } = request;
        let directives = self
            .before_turn(TurnHookContext {
                session_id,
                state,
                host: host.clone(),
                turn_context,
            })
            .await?;
        self.apply_turn_directives(
            directives,
            messages,
            host,
            true,
            "tool directives are not valid in before_turn",
        )
        .await
    }

    pub async fn apply_checkpoint(
        &self,
        ctx: CheckpointHookContext,
    ) -> Result<CheckpointApplication, PluginError> {
        let directives = self.at_checkpoint(ctx.clone()).await?;
        let mut messages = Vec::new();
        let mut events = Vec::new();
        let mut abort = None;

        for emitted in directives {
            match emitted.value {
                PluginDirective::EnqueueMessages { messages: queued } => messages.extend(queued),
                PluginDirective::CreateSession { request } => {
                    ctx.host
                        .create_session(*request)
                        .await
                        .map_err(|err| PluginError::Session(err.to_string()))?;
                }
                PluginDirective::HandoffSession { .. } => {
                    return Err(PluginError::Session(
                        "checkpoint hooks do not support session handoff".to_string(),
                    ));
                }
                PluginDirective::AbortTurn { code, message } => {
                    abort = Some(PluginAbort { code, message });
                }
                PluginDirective::EmitEvents { events: surface } => {
                    events.extend(crate::plugin::plugin_surface_session_events(
                        &emitted.plugin_id,
                        surface,
                    ));
                }
                PluginDirective::EmitTrace {
                    name,
                    payload,
                    context,
                } => {
                    ctx.host
                        .emit_trace_event(
                            *context,
                            lash_trace::TraceEvent::Custom {
                                name: format!("plugin.{}.{}", emitted.plugin_id, name),
                                payload,
                            },
                        )
                        .await?;
                }
                PluginDirective::ReplaceToolArgs { .. }
                | PluginDirective::ShortCircuitTool { .. } => {
                    return Err(PluginError::Session(
                        "checkpoint hooks only support abort, message enqueue, session creation, events, and trace events"
                            .to_string(),
                    ));
                }
            }
        }

        Ok(CheckpointApplication {
            messages,
            events,
            abort,
        })
    }

    pub async fn before_turn(
        &self,
        ctx: TurnHookContext,
    ) -> Result<Vec<PluginOwned<PluginDirective>>, PluginError> {
        collect_owned_async(&self.before_turn_hooks, ctx, |hook, ctx| hook(ctx)).await
    }

    pub async fn before_tool_call(
        &self,
        ctx: ToolCallHookContext,
    ) -> Result<Vec<PluginOwned<PluginDirective>>, PluginError> {
        collect_owned_async(&self.before_tool_call_hooks, ctx, |hook, ctx| hook(ctx)).await
    }

    pub async fn after_tool_call(
        &self,
        ctx: ToolResultHookContext,
    ) -> Result<Vec<PluginOwned<PluginDirective>>, PluginError> {
        collect_owned_async(&self.after_tool_call_hooks, ctx, |hook, ctx| hook(ctx)).await
    }

    pub async fn after_turn(
        &self,
        ctx: TurnResultHookContext,
    ) -> Result<Vec<PluginOwned<PluginDirective>>, PluginError> {
        collect_owned_async(&self.after_turn_hooks, ctx, |hook, ctx| hook(ctx)).await
    }

    pub async fn at_checkpoint(
        &self,
        ctx: CheckpointHookContext,
    ) -> Result<Vec<PluginOwned<PluginDirective>>, PluginError> {
        collect_owned_async(&self.checkpoint_hooks, ctx, |hook, ctx| hook(ctx)).await
    }

    pub async fn transform_assistant_stream(
        &self,
        session_id: &str,
        chunk: String,
    ) -> Result<Vec<PluginOwned<AssistantStreamTransform>>, PluginError> {
        let mut current = chunk;
        let mut transforms = Vec::new();
        for registered in &self.assistant_stream_hooks {
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
        for registered in &self.assistant_response_hooks {
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
        let Some(projector) = &self.tool_result_projector else {
            return Ok(crate::ModelToolReturn::from_output(
                ctx.call_id.clone(),
                ctx.tool_name.clone(),
                &ctx.output,
            ));
        };
        (projector.hook)(ctx).await
    }

    pub async fn emit_runtime_event(&self, event: PluginRuntimeEvent) {
        let mut tasks = JoinSet::new();
        for hook in &self.runtime_event_hooks {
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
        !self.runtime_event_hooks.is_empty()
    }

    pub async fn mutate_session_config(
        &self,
        ctx: SessionConfigChangedContext,
        mut policy: SessionPolicy,
    ) -> SessionPolicy {
        for hook in &self.session_config_mutators {
            match hook(ctx.clone(), policy.clone()).await {
                Ok(next_policy) => policy = next_policy,
                Err(err) => tracing::warn!("plugin config mutator failed: {err}"),
            }
        }
        policy
    }

    pub async fn finalize_turn(
        &self,
        mut turn: AssembledTurn,
        host: Arc<dyn ToolHookHost>,
    ) -> Result<TurnFinalization, PluginError> {
        let session_id = turn.state.session_id.clone();
        let directives = if self.after_turn_hooks.is_empty() {
            Vec::new()
        } else {
            self.after_turn(TurnResultHookContext {
                session_id: session_id.clone(),
                turn: Arc::new(crate::plugin::TurnResultSummary::from_assembled(&turn)),
                host: host.clone(),
            })
            .await?
        };
        let mut events = Vec::new();
        let mut updated_messages: Option<crate::MessageSequence> = None;
        for emitted in directives {
            match emitted.value {
                PluginDirective::AbortTurn { .. } => {
                    return Err(PluginError::Session(
                        "only message enqueue and session creation are valid in after_turn"
                            .to_string(),
                    ));
                }
                PluginDirective::EnqueueMessages {
                    messages: plugin_messages,
                } => {
                    let messages = updated_messages.get_or_insert_with(|| {
                        crate::MessageSequence::from_base(
                            turn.state.read_view().messages().to_vec().into(),
                        )
                    });
                    append_plugin_messages(messages, &plugin_messages);
                }
                PluginDirective::CreateSession { request } => {
                    host.create_session(*request)
                        .await
                        .map_err(|err| PluginError::Session(err.to_string()))?;
                }
                PluginDirective::HandoffSession { .. } => {
                    return Err(PluginError::Session(
                        "after_turn hooks do not support session handoff".to_string(),
                    ));
                }
                PluginDirective::EmitEvents { events: surface } => {
                    events.extend(crate::plugin::plugin_surface_session_events(
                        &emitted.plugin_id,
                        surface,
                    ));
                }
                PluginDirective::EmitTrace {
                    name,
                    payload,
                    context,
                } => {
                    host.emit_trace_event(
                        *context,
                        lash_trace::TraceEvent::Custom {
                            name: format!("plugin.{}.{}", emitted.plugin_id, name),
                            payload,
                        },
                    )
                    .await?;
                }
                PluginDirective::ReplaceToolArgs { .. }
                | PluginDirective::ShortCircuitTool { .. } => {
                    return Err(PluginError::Session(
                        "only message enqueue, session creation, events, and trace events are valid in after_turn"
                            .to_string(),
                    ));
                }
            }
        }
        if let Some(messages) = updated_messages.as_ref() {
            let tool_calls = turn.state.read_view().tool_calls().to_vec();
            turn.state
                .replace_active_read_state(messages.as_slice(), &tool_calls);
        }

        if self.has_runtime_event_hooks() {
            self.emit_runtime_event(PluginRuntimeEvent::TurnCommitted(Arc::new(turn.clone())))
                .await;
        }

        Ok(TurnFinalization { turn, events })
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
        execution_mode: ExecutionMode,
        standard_context_approach: Option<crate::StandardContextApproach>,
    ) -> Result<Arc<PluginSession>, PluginError> {
        let snapshot = self.snapshot()?;
        self.host.build_session_with_surface(
            session_id,
            execution_mode,
            standard_context_approach,
            Some(&snapshot),
            self.tool_surface_overlay.clone(),
            Some(self.tool_registry.export_state()),
        )
    }

    pub fn fork_for_child_session(
        &self,
        session_id: impl Into<String>,
        parent_session_id: Option<String>,
        execution_mode: ExecutionMode,
        standard_context_approach: Option<crate::StandardContextApproach>,
        authority: super::SessionAuthorityContext,
    ) -> Result<Arc<PluginSession>, PluginError> {
        let snapshot = self.snapshot()?;
        self.host.build_session_with_parent_and_surface(
            session_id,
            parent_session_id,
            execution_mode,
            standard_context_approach,
            Some(&snapshot),
            self.tool_surface_overlay.clone(),
            Some(self.tool_registry.export_state()),
            authority,
        )
    }

    pub fn fork_for_session_with_tool_surface(
        &self,
        session_id: impl Into<String>,
        execution_mode: ExecutionMode,
        standard_context_approach: Option<crate::StandardContextApproach>,
        tool_surface_overlay: ToolSurfaceContribution,
    ) -> Result<Arc<PluginSession>, PluginError> {
        let snapshot = self.snapshot()?;
        self.host.build_session_with_surface(
            session_id,
            execution_mode,
            standard_context_approach,
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
        host: Arc<dyn PluginActionHost>,
    ) -> Result<ToolResult, PluginActionInvokeError> {
        let Some(op) = self.plugin_actions.get(name).cloned() else {
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
        host: Arc<dyn PluginActionHost>,
    ) -> Result<Op::Output, PluginError> {
        let args = serde_json::to_value(args)
            .map_err(|err| PluginError::Invoke(format!("invalid {} args: {err}", Op::NAME)))?;
        let result = self
            .invoke_plugin_action(Op::NAME, args, session_id, default_to_current_session, host)
            .await
            .map_err(|err| PluginError::Invoke(err.to_string()))?;
        if !result.is_success() {
            return Err(PluginError::Invoke(format!(
                "{} failed: {}",
                Op::NAME,
                result.value_for_projection()
            )));
        }
        serde_json::from_value(result.into_value_for_projection())
            .map_err(|err| PluginError::Invoke(format!("invalid {} output: {err}", Op::NAME)))
    }
}
