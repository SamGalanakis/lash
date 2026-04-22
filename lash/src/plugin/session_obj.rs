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

fn append_plugin_messages(
    messages: &mut crate::MessageSequence,
    plugin_messages: &[PluginMessage],
) {
    let new_messages = plugin_messages
        .iter()
        .filter(|message| matches!(message.role, MessageRole::User | MessageRole::System))
        .map(|message| plugin_message_to_message(message, message.user_input.clone()))
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
    pub(super) dynamic_tools: Option<Arc<crate::DynamicToolProvider>>,
    pub(super) tool_surface_overlay: ToolSurfaceContribution,
    pub(super) prompt_contributors: Vec<RegisteredHook<PromptContributor>>,
    pub(super) prompt_request_hooks: Vec<RegisteredHook<PromptRequestHook>>,
    pub(super) tool_surface_contributors: Vec<RegisteredHook<ToolSurfaceContributor>>,
    pub(super) before_turn_hooks: Vec<RegisteredHook<BeforeTurnHook>>,
    pub(super) before_tool_call_hooks: Vec<RegisteredHook<BeforeToolCallHook>>,
    pub(super) after_tool_call_hooks: Vec<RegisteredHook<AfterToolCallHook>>,
    pub(super) after_turn_hooks: Vec<RegisteredHook<AfterTurnHook>>,
    pub(super) checkpoint_hooks: Vec<RegisteredHook<CheckpointHook>>,
    pub(super) assistant_stream_hooks: Vec<RegisteredHook<AssistantStreamHook>>,
    pub(super) assistant_response_hooks: Vec<RegisteredHook<AssistantResponseHook>>,
    pub(super) tool_result_projectors:
        BTreeMap<ToolResultProjectionHook, RegisteredExclusiveHook<ToolResultProjector>>,
    pub(super) runtime_event_hooks: Vec<PluginRuntimeEventHook>,
    pub(super) session_config_mutators: Vec<SessionConfigMutator>,
    pub(super) external_ops: BTreeMap<String, RegisteredExternalOp>,
    pub(super) commands: BTreeMap<String, RegisteredCommand>,
    pub(super) monitor_specs: Vec<PluginOwned<crate::MonitorSpec>>,
    pub(super) turn_context_transforms: Vec<Arc<dyn TurnContextTransform>>,
    pub(super) history_rewriters: Vec<Arc<dyn HistoryRewriter>>,
    pub(super) mode_session: Arc<dyn ModeSessionPlugin>,
    pub(super) mode_native_tools: Arc<dyn ModeNativeToolsPlugin>,
    pub(super) mode_protocol_driver: Option<Arc<dyn ModeProtocolDriverPlugin>>,
}
impl PluginSession {
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    pub fn execution_mode(&self) -> ExecutionMode {
        self.execution_mode
    }

    pub fn host(&self) -> &PluginHost {
        &self.host
    }

    pub fn tools(&self) -> Arc<dyn ToolProvider> {
        Arc::clone(&self.tools)
    }

    pub fn dynamic_tools(&self) -> Option<Arc<crate::DynamicToolProvider>> {
        self.dynamic_tools.clone()
    }

    pub(crate) fn mode_session(&self) -> &Arc<dyn ModeSessionPlugin> {
        &self.mode_session
    }

    pub(crate) fn mode_native_tools(&self) -> &Arc<dyn ModeNativeToolsPlugin> {
        &self.mode_native_tools
    }

    /// Plugin-registered protocol driver for this session, if any plugin
    /// claimed the singleton slot. When `None`, callers fall back to
    /// `lash_sansio::build_mode_preamble` (hardcoded Standard/RLM).
    pub fn mode_protocol_driver(&self) -> Option<Arc<dyn ModeProtocolDriverPlugin>> {
        self.mode_protocol_driver.clone()
    }

    pub fn tool_surface(&self, session_id: &str, mode: ExecutionMode) -> crate::ToolSurface {
        let mut tools = self.tools.definitions();
        if mode == self.execution_mode {
            tools.extend(self.mode_native_tools.definitions());
        }
        self.resolve_tool_surface(ToolSurfaceContext {
            session_id: session_id.to_string(),
            mode,
            tools: tools.clone(),
        })
        .unwrap_or_else(|err| {
            tracing::warn!("failed to resolve tool surface: {err}");
            crate::ToolSurface::from_tools(tools)
        })
    }

    pub fn tool_catalog(&self, session_id: &str, mode: ExecutionMode) -> Vec<serde_json::Value> {
        let surface = self.tool_surface(session_id, mode);
        crate::tools::project_tool_catalog(surface.enabled_tools_iter())
    }

    pub fn resolve_tool_surface(
        &self,
        ctx: ToolSurfaceContext,
    ) -> Result<crate::ToolSurface, PluginError> {
        let mut contributions = collect_owned_sync(
            &self.tool_surface_contributors,
            ToolSurfaceContext {
                session_id: ctx.session_id.clone(),
                mode: ctx.mode,
                tools: ctx.tools.clone(),
            },
            |hook, ctx| hook(ctx),
        )?
        .into_iter()
        .map(|owned| owned.value)
        .collect::<Vec<_>>();
        contributions.push(self.tool_surface_overlay.clone());
        Ok(crate::build_tool_surface(crate::ToolSurfaceBuildInput {
            tools: ctx.tools,
            mode: ctx.mode,
            contributions,
        }))
    }

    pub fn external_ops(&self) -> Vec<ExternalOpDef> {
        self.external_ops
            .values()
            .map(|op| op.def.clone())
            .collect()
    }

    pub fn monitor_specs(&self) -> &[PluginOwned<crate::MonitorSpec>] {
        self.monitor_specs.as_slice()
    }

    /// Catalog of slash commands contributed by plugins registered on this session.
    pub fn command_catalog(&self) -> Vec<CommandDef> {
        self.commands.values().map(|cmd| cmd.def.clone()).collect()
    }

    pub fn has_assistant_stream_hooks(&self) -> bool {
        !self.assistant_stream_hooks.is_empty()
    }

    /// Invoke a plugin-registered slash command.
    pub async fn invoke_command(
        &self,
        name: &str,
        argument: Option<String>,
        host: Arc<dyn SessionManager>,
    ) -> Result<CommandOutcome, PluginError> {
        let Some(cmd) = self.commands.get(name).cloned() else {
            return Err(PluginError::Session(format!(
                "unknown plugin command `{name}`"
            )));
        };
        (cmd.handler)(CommandInvocation {
            name: name.to_string(),
            argument,
            session_id: self.session_id.clone(),
            host,
        })
        .await
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

    pub async fn on_prompt_request(
        &self,
        ctx: PromptRequestHookContext,
    ) -> Result<Vec<PluginOwned<PluginSurfaceEvent>>, PluginError> {
        collect_owned_async(&self.prompt_request_hooks, ctx, |hook, ctx| hook(ctx)).await
    }

    async fn apply_turn_directives(
        &self,
        directives: Vec<PluginOwned<PluginDirective>>,
        mut messages: crate::MessageSequence,
        host: Arc<dyn SessionManager>,
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
                PluginDirective::EmitEvents { events: surface } => {
                    events.extend(crate::plugin::plugin_surface_session_events(
                        &emitted.plugin_id,
                        surface,
                    ));
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
        } = request;
        let directives = self
            .before_turn(TurnHookContext {
                session_id,
                state,
                host: Arc::clone(&host),
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
                PluginDirective::AbortTurn { code, message } => {
                    abort = Some(PluginAbort { code, message });
                }
                PluginDirective::EmitEvents { events: surface } => {
                    events.extend(crate::plugin::plugin_surface_session_events(
                        &emitted.plugin_id,
                        surface,
                    ));
                }
                PluginDirective::ReplaceToolArgs { .. }
                | PluginDirective::ShortCircuitTool { .. } => {
                    return Err(PluginError::Session(
                        "checkpoint hooks only support abort, message enqueue, and session creation"
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
        host: Arc<dyn SessionManager>,
    ) -> Result<Vec<PluginOwned<AssistantStreamTransform>>, PluginError> {
        let mut current = chunk;
        let mut transforms = Vec::new();
        for registered in &self.assistant_stream_hooks {
            let transform = (registered.hook)(AssistantStreamHookContext {
                session_id: session_id.to_string(),
                chunk: current.clone(),
                host: Arc::clone(&host),
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
        host: Arc<dyn SessionManager>,
    ) -> Result<Vec<PluginOwned<AssistantResponseTransform>>, PluginError> {
        let mut current = response;
        let mut transforms = Vec::new();
        for registered in &self.assistant_response_hooks {
            let transform = (registered.hook)(AssistantResponseHookContext {
                session_id: session_id.to_string(),
                response: current.clone(),
                host: Arc::clone(&host),
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
    ) -> Result<ToolResult, PluginError> {
        let Some(projector) = self.tool_result_projectors.get(&ctx.hook) else {
            return Ok(ctx.result);
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
        host: Arc<dyn SessionManager>,
    ) -> Result<TurnFinalization, PluginError> {
        let session_id = turn.state.session_id.clone();
        let directives = if self.after_turn_hooks.is_empty() {
            Vec::new()
        } else {
            self.after_turn(TurnResultHookContext {
                session_id: session_id.clone(),
                turn: Arc::new(crate::plugin::TurnResultSummary::from_assembled(&turn)),
                host: Arc::clone(&host),
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
                            turn.state.session_graph.shared_projected_messages(),
                        )
                    });
                    append_plugin_messages(messages, &plugin_messages);
                }
                PluginDirective::CreateSession { request } => {
                    host.create_session(*request)
                        .await
                        .map_err(|err| PluginError::Session(err.to_string()))?;
                }
                PluginDirective::EmitEvents { events: surface } => {
                    events.extend(crate::plugin::plugin_surface_session_events(
                        &emitted.plugin_id,
                        surface,
                    ));
                }
                PluginDirective::ReplaceToolArgs { .. }
                | PluginDirective::ShortCircuitTool { .. } => {
                    return Err(PluginError::Session(
                        "only message enqueue and session creation are valid in after_turn"
                            .to_string(),
                    ));
                }
            }
        }
        if let Some(messages) = updated_messages.as_ref() {
            let tool_calls = turn.state.projected_tool_calls().to_vec();
            turn.state
                .replace_projection(messages.as_slice(), &tool_calls);
        }

        if self.has_runtime_event_hooks() {
            let mut history_tool_calls = turn.state.project_tool_calls();
            let mut history_changed = false;
            for tool_call in &mut history_tool_calls {
                let projected = self
                    .project_tool_result(ToolResultProjectionContext {
                        hook: ToolResultProjectionHook::BeforeHistory,
                        session_id: session_id.clone(),
                        tool_name: tool_call.tool.clone(),
                        args: tool_call.args.clone(),
                        result: ToolResult {
                            success: tool_call.success,
                            result: tool_call.result.clone(),
                            images: Vec::new(),
                        },
                        duration_ms: tool_call.duration_ms,
                        host: Arc::clone(&host),
                    })
                    .await?;
                history_changed |=
                    projected.success != tool_call.success || projected.result != tool_call.result;
                tool_call.result = projected.result;
                tool_call.success = projected.success;
            }
            let committed_turn = if history_changed {
                let mut committed_turn = turn.clone();
                committed_turn
                    .state
                    .replace_tool_call_projection(&history_tool_calls);
                committed_turn.tool_calls = history_tool_calls;
                Arc::new(committed_turn)
            } else {
                Arc::new(turn.clone())
            };
            self.emit_runtime_event(PluginRuntimeEvent::TurnCommitted(committed_turn))
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
        context_approach: crate::ContextApproach,
    ) -> Result<Arc<PluginSession>, PluginError> {
        let snapshot = self.snapshot()?;
        self.host.build_session_with_surface(
            session_id,
            execution_mode,
            context_approach,
            Some(&snapshot),
            self.tool_surface_overlay.clone(),
            self.tools.dynamic_snapshot(),
        )
    }

    pub fn fork_for_session_with_tool_surface(
        &self,
        session_id: impl Into<String>,
        execution_mode: ExecutionMode,
        context_approach: crate::ContextApproach,
        tool_surface_overlay: ToolSurfaceContribution,
    ) -> Result<Arc<PluginSession>, PluginError> {
        let snapshot = self.snapshot()?;
        self.host.build_session_with_surface(
            session_id,
            execution_mode,
            context_approach,
            Some(&snapshot),
            tool_surface_overlay,
            self.tools.dynamic_snapshot(),
        )
    }

    pub async fn invoke_external(
        &self,
        name: &str,
        args: serde_json::Value,
        session_id: Option<String>,
        default_to_current_session: bool,
        host: Arc<dyn SessionManager>,
    ) -> Result<ToolResult, ExternalInvokeError> {
        let Some(op) = self.external_ops.get(name).cloned() else {
            return Err(ExternalInvokeError::Unknown(name.to_string()));
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
                return Err(ExternalInvokeError::MissingSession(name.to_string()));
            }
            (SessionParam::Forbidden, Some(_)) => {
                return Err(ExternalInvokeError::UnexpectedSession(name.to_string()));
            }
            _ => {}
        }

        Ok((op.handler)(
            ExternalInvokeContext {
                session_id: effective_session,
                host,
            },
            args,
        )
        .await)
    }
}
