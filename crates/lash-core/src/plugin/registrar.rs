use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use super::*;

#[derive(Clone)]
pub(crate) struct RegisteredHook<T> {
    pub(crate) plugin_id: String,
    pub(crate) hook: T,
}

#[derive(Clone)]
pub(crate) struct RegisteredExclusiveHook<T> {
    pub(crate) plugin_id: String,
    pub(crate) hook: T,
}

pub(crate) fn current_plugin_id(registering_plugin_id: &Option<String>) -> String {
    registering_plugin_id
        .clone()
        .unwrap_or_else(|| "__unknown__".to_string())
}

fn push_registered_hook<T>(
    hooks: &mut Vec<RegisteredHook<T>>,
    registering_plugin_id: &Option<String>,
    hook: T,
) {
    hooks.push(RegisteredHook {
        plugin_id: current_plugin_id(registering_plugin_id),
        hook,
    });
}

fn exclusive_hook_owner(
    existing_owner: Option<&str>,
    registering_plugin_id: &Option<String>,
    hook_kind: &str,
    hook_name: &str,
) -> Result<String, PluginError> {
    let plugin_id = registering_plugin_id
        .clone()
        .ok_or_else(|| PluginError::Registration("missing registering plugin id".to_string()))?;
    if let Some(existing) = existing_owner {
        return Err(PluginError::Registration(format!(
            "duplicate {hook_kind} for `{hook_name}`: `{plugin_id}` conflicts with `{existing}`"
        )));
    }
    Ok(plugin_id)
}

fn register_singleton_hook<H>(
    slot: &mut Option<RegisteredExclusiveHook<H>>,
    registering_plugin_id: &Option<String>,
    hook_kind: &str,
    hook_name: &str,
    hook: H,
) -> Result<(), PluginError> {
    let plugin_id = exclusive_hook_owner(
        slot.as_ref()
            .map(|registered| registered.plugin_id.as_str()),
        registering_plugin_id,
        hook_kind,
        hook_name,
    )?;
    *slot = Some(RegisteredExclusiveHook { plugin_id, hook });
    Ok(())
}

pub struct PluginRegistrar {
    pub(crate) tool_names: BTreeSet<String>,
    pub(crate) tool_providers: Vec<Arc<dyn ToolProvider>>,
    pub(crate) prompt_contributors: Vec<RegisteredHook<PromptContributor>>,
    pub(crate) tool_surface_contributors: Vec<RegisteredHook<ToolSurfaceContributor>>,
    pub(crate) tool_discovery_contributors: Vec<RegisteredHook<ToolDiscoveryContributor>>,
    pub(crate) before_turn_hooks: Vec<RegisteredHook<BeforeTurnHook>>,
    pub(crate) before_tool_call_hooks: Vec<RegisteredHook<BeforeToolCallHook>>,
    pub(crate) after_tool_call_hooks: Vec<RegisteredHook<AfterToolCallHook>>,
    pub(crate) after_turn_hooks: Vec<RegisteredHook<AfterTurnHook>>,
    pub(crate) checkpoint_hooks: Vec<RegisteredHook<CheckpointHook>>,
    pub(crate) assistant_stream_hooks: Vec<RegisteredHook<AssistantStreamHook>>,
    pub(crate) assistant_response_hooks: Vec<RegisteredHook<AssistantResponseHook>>,
    pub(crate) tool_result_projector: Option<RegisteredExclusiveHook<ToolResultProjector>>,
    pub(crate) runtime_event_hooks: Vec<PluginLifecycleEventHook>,
    pub(crate) session_config_mutators: Vec<SessionConfigMutator>,
    pub(crate) plugin_actions: BTreeMap<String, RegisteredPluginAction>,
    pub(crate) monitor_specs: Vec<PluginOwned<crate::MonitorSpec>>,
    pub(crate) turn_context_transforms: Vec<(i32, Arc<dyn TurnContextTransform>)>,
    pub(crate) history_rewriters: Vec<(i32, Arc<dyn HistoryRewriter>)>,
    pub(crate) mode_session: Option<RegisteredExclusiveHook<Arc<dyn ModeSessionPlugin>>>,
    pub(crate) mode_native_tools: Vec<RegisteredHook<Arc<dyn ModeNativeToolsPlugin>>>,
    pub(crate) mode_protocol_driver:
        Option<RegisteredExclusiveHook<Arc<dyn ModeProtocolDriverPlugin>>>,
    pub(crate) registering_plugin_id: Option<String>,
}

pub struct ToolRegistrations<'a> {
    reg: &'a mut PluginRegistrar,
}

impl ToolRegistrations<'_> {
    pub fn provider(self, provider: Arc<dyn ToolProvider>) -> Result<(), PluginError> {
        self.reg.add_tool_provider(provider)
    }
}

pub struct PromptRegistrations<'a> {
    reg: &'a mut PluginRegistrar,
}

impl PromptRegistrations<'_> {
    pub fn contribute(self, contributor: PromptContributor) {
        self.reg.add_prompt_contributor(contributor);
    }
}

pub struct SurfaceRegistrations<'a> {
    reg: &'a mut PluginRegistrar,
}

impl SurfaceRegistrations<'_> {
    pub fn contribute(self, contributor: ToolSurfaceContributor) {
        self.reg.add_tool_surface_contributor(contributor);
    }
}

pub struct DiscoveryRegistrations<'a> {
    reg: &'a mut PluginRegistrar,
}

impl DiscoveryRegistrations<'_> {
    pub fn contribute(self, contributor: ToolDiscoveryContributor) {
        self.reg.add_tool_discovery_contributor(contributor);
    }
}

pub struct TurnRegistrations<'a> {
    reg: &'a mut PluginRegistrar,
}

impl TurnRegistrations<'_> {
    pub fn before(self, hook: BeforeTurnHook) {
        self.reg.add_before_turn_hook(hook);
    }

    pub fn after(self, hook: AfterTurnHook) {
        self.reg.add_after_turn_hook(hook);
    }

    pub fn checkpoint(self, hook: CheckpointHook) {
        self.reg.add_checkpoint_hook(hook);
    }
}

pub struct ToolCallRegistrations<'a> {
    reg: &'a mut PluginRegistrar,
}

impl ToolCallRegistrations<'_> {
    pub fn before(self, hook: BeforeToolCallHook) {
        self.reg.add_before_tool_call_hook(hook);
    }

    pub fn after(self, hook: AfterToolCallHook) {
        self.reg.add_after_tool_call_hook(hook);
    }
}

pub struct OutputRegistrations<'a> {
    reg: &'a mut PluginRegistrar,
}

impl OutputRegistrations<'_> {
    pub fn stream(self, hook: AssistantStreamHook) {
        self.reg.add_assistant_stream_hook(hook);
    }

    pub fn response(self, hook: AssistantResponseHook) {
        self.reg.add_assistant_response_hook(hook);
    }
}

pub struct ToolResultRegistrations<'a> {
    reg: &'a mut PluginRegistrar,
}

impl ToolResultRegistrations<'_> {
    pub fn projector(self, hook: ToolResultProjector) -> Result<(), PluginError> {
        self.reg.add_tool_result_projector(hook)
    }
}

pub struct SessionRegistrations<'a> {
    reg: &'a mut PluginRegistrar,
}

impl SessionRegistrations<'_> {
    pub fn on_event(self, hook: PluginLifecycleEventHook) {
        self.reg.runtime_event_hooks.push(hook);
    }

    pub fn config_mutator(self, hook: SessionConfigMutator) {
        self.reg.session_config_mutators.push(hook);
    }
}

pub struct MonitorRegistrations<'a> {
    reg: &'a mut PluginRegistrar,
}

impl MonitorRegistrations<'_> {
    pub fn register(self, spec: crate::MonitorSpec) {
        self.reg.add_monitor_spec(spec);
    }
}

pub struct PluginActionRegistrations<'a> {
    reg: &'a mut PluginRegistrar,
}

impl PluginActionRegistrations<'_> {
    pub fn op(self, def: PluginActionDef, handler: PluginActionHandler) -> Result<(), PluginError> {
        self.reg.add_plugin_action(def, handler)
    }

    pub fn typed<Op, F, Fut>(self, handler: F) -> Result<(), PluginError>
    where
        Op: PluginAction,
        F: Fn(PluginActionContext, Op::Args) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Op::Output, PluginActionFailure>> + Send + 'static,
    {
        self.op(
            plugin_action_def::<Op>(),
            Arc::new(move |ctx, args| {
                let parsed = serde_json::from_value::<Op::Args>(args);
                match parsed {
                    Ok(args) => {
                        let fut = handler(ctx, args);
                        Box::pin(async move {
                            match fut.await {
                                Ok(output) => match serde_json::to_value(output) {
                                    Ok(value) => ToolResult::ok(value),
                                    Err(err) => ToolResult::err(serde_json::json!(format!(
                                        "failed to serialize {} output: {err}",
                                        Op::NAME
                                    ))),
                                },
                                Err(err) => ToolResult::err(serde_json::json!(err.to_string())),
                            }
                        })
                    }
                    Err(err) => Box::pin(async move {
                        ToolResult::err(serde_json::json!(format!(
                            "invalid {} args: {err}",
                            Op::NAME
                        )))
                    }),
                }
            }),
        )
    }
}

pub struct HistoryRegistrations<'a> {
    reg: &'a mut PluginRegistrar,
}

impl HistoryRegistrations<'_> {
    /// Register a per-turn context transform. Higher priority runs first.
    pub fn prepare_turn(self, priority: i32, transform: Arc<dyn TurnContextTransform>) {
        self.reg.turn_context_transforms.push((priority, transform));
    }

    /// Register a permanent history rewriter. Higher priority runs first.
    pub fn rewrite(self, priority: i32, rewriter: Arc<dyn HistoryRewriter>) {
        self.reg.history_rewriters.push((priority, rewriter));
    }
}

pub struct ModeRegistrations<'a> {
    reg: &'a mut PluginRegistrar,
}

impl ModeRegistrations<'_> {
    pub fn session(self, provider: Arc<dyn ModeSessionPlugin>) -> Result<(), PluginError> {
        self.reg.add_mode_session(provider)
    }

    pub fn native_tools(self, provider: Arc<dyn ModeNativeToolsPlugin>) -> Result<(), PluginError> {
        self.reg.add_mode_native_tools(provider)
    }

    /// Claim the session-wide singleton protocol-driver slot. The
    /// plugin provides a `ProtocolDriverHandle` via `build_preamble`
    /// and identifies itself with a `mode_id` that the session's
    /// `ExecutionMode` must match for the driver to be selected.
    pub fn protocol_driver(
        self,
        provider: Arc<dyn ModeProtocolDriverPlugin>,
    ) -> Result<(), PluginError> {
        self.reg.add_mode_protocol_driver(provider)
    }
}

impl PluginRegistrar {
    pub(crate) fn new() -> Self {
        Self {
            tool_names: BTreeSet::new(),
            tool_providers: Vec::new(),
            prompt_contributors: Vec::new(),
            tool_surface_contributors: Vec::new(),
            tool_discovery_contributors: Vec::new(),
            before_turn_hooks: Vec::new(),
            before_tool_call_hooks: Vec::new(),
            after_tool_call_hooks: Vec::new(),
            after_turn_hooks: Vec::new(),
            checkpoint_hooks: Vec::new(),
            assistant_stream_hooks: Vec::new(),
            assistant_response_hooks: Vec::new(),
            tool_result_projector: None,
            runtime_event_hooks: Vec::new(),
            session_config_mutators: Vec::new(),
            plugin_actions: BTreeMap::new(),
            monitor_specs: Vec::new(),
            turn_context_transforms: Vec::new(),
            history_rewriters: Vec::new(),
            mode_session: None,
            mode_native_tools: Vec::new(),
            mode_protocol_driver: None,
            registering_plugin_id: None,
        }
    }

    pub fn tools(&mut self) -> ToolRegistrations<'_> {
        ToolRegistrations { reg: self }
    }

    pub fn prompt(&mut self) -> PromptRegistrations<'_> {
        PromptRegistrations { reg: self }
    }

    pub fn surface(&mut self) -> SurfaceRegistrations<'_> {
        SurfaceRegistrations { reg: self }
    }

    pub fn discovery(&mut self) -> DiscoveryRegistrations<'_> {
        DiscoveryRegistrations { reg: self }
    }

    pub fn turn(&mut self) -> TurnRegistrations<'_> {
        TurnRegistrations { reg: self }
    }

    pub fn tool_calls(&mut self) -> ToolCallRegistrations<'_> {
        ToolCallRegistrations { reg: self }
    }

    pub fn output(&mut self) -> OutputRegistrations<'_> {
        OutputRegistrations { reg: self }
    }

    pub fn tool_results(&mut self) -> ToolResultRegistrations<'_> {
        ToolResultRegistrations { reg: self }
    }

    pub fn session(&mut self) -> SessionRegistrations<'_> {
        SessionRegistrations { reg: self }
    }

    pub fn actions(&mut self) -> PluginActionRegistrations<'_> {
        PluginActionRegistrations { reg: self }
    }

    pub fn monitors(&mut self) -> MonitorRegistrations<'_> {
        MonitorRegistrations { reg: self }
    }

    pub fn history(&mut self) -> HistoryRegistrations<'_> {
        HistoryRegistrations { reg: self }
    }

    pub fn mode(&mut self) -> ModeRegistrations<'_> {
        ModeRegistrations { reg: self }
    }

    fn add_tool_provider(&mut self, provider: Arc<dyn ToolProvider>) -> Result<(), PluginError> {
        for manifest in provider.tool_manifests() {
            if !self.tool_names.insert(manifest.name.clone()) {
                return Err(PluginError::Registration(format!(
                    "duplicate plugin tool name `{}`",
                    manifest.name
                )));
            }
        }
        self.tool_providers.push(provider);
        Ok(())
    }

    fn add_prompt_contributor(&mut self, contributor: PromptContributor) {
        push_registered_hook(
            &mut self.prompt_contributors,
            &self.registering_plugin_id,
            contributor,
        );
    }

    fn add_tool_surface_contributor(&mut self, contributor: ToolSurfaceContributor) {
        push_registered_hook(
            &mut self.tool_surface_contributors,
            &self.registering_plugin_id,
            contributor,
        );
    }

    fn add_tool_discovery_contributor(&mut self, contributor: ToolDiscoveryContributor) {
        push_registered_hook(
            &mut self.tool_discovery_contributors,
            &self.registering_plugin_id,
            contributor,
        );
    }

    fn add_before_turn_hook(&mut self, hook: BeforeTurnHook) {
        push_registered_hook(
            &mut self.before_turn_hooks,
            &self.registering_plugin_id,
            hook,
        );
    }

    fn add_before_tool_call_hook(&mut self, hook: BeforeToolCallHook) {
        push_registered_hook(
            &mut self.before_tool_call_hooks,
            &self.registering_plugin_id,
            hook,
        );
    }

    fn add_after_tool_call_hook(&mut self, hook: AfterToolCallHook) {
        push_registered_hook(
            &mut self.after_tool_call_hooks,
            &self.registering_plugin_id,
            hook,
        );
    }

    fn add_after_turn_hook(&mut self, hook: AfterTurnHook) {
        push_registered_hook(
            &mut self.after_turn_hooks,
            &self.registering_plugin_id,
            hook,
        );
    }

    fn add_checkpoint_hook(&mut self, hook: CheckpointHook) {
        push_registered_hook(
            &mut self.checkpoint_hooks,
            &self.registering_plugin_id,
            hook,
        );
    }

    fn add_assistant_stream_hook(&mut self, hook: AssistantStreamHook) {
        push_registered_hook(
            &mut self.assistant_stream_hooks,
            &self.registering_plugin_id,
            hook,
        );
    }

    fn add_assistant_response_hook(&mut self, hook: AssistantResponseHook) {
        push_registered_hook(
            &mut self.assistant_response_hooks,
            &self.registering_plugin_id,
            hook,
        );
    }

    fn add_tool_result_projector(&mut self, hook: ToolResultProjector) -> Result<(), PluginError> {
        register_singleton_hook(
            &mut self.tool_result_projector,
            &self.registering_plugin_id,
            "tool result projector",
            "model_observation",
            hook,
        )
    }

    fn add_plugin_action(
        &mut self,
        def: PluginActionDef,
        handler: PluginActionHandler,
    ) -> Result<(), PluginError> {
        if self.plugin_actions.contains_key(&def.name) {
            return Err(PluginError::Registration(format!(
                "duplicate plugin action name `{}`",
                def.name
            )));
        }
        self.plugin_actions
            .insert(def.name.clone(), RegisteredPluginAction { def, handler });
        Ok(())
    }

    fn add_monitor_spec(&mut self, spec: crate::MonitorSpec) {
        self.monitor_specs.push(PluginOwned {
            plugin_id: current_plugin_id(&self.registering_plugin_id),
            value: spec,
        });
    }

    fn add_mode_session(
        &mut self,
        provider: Arc<dyn ModeSessionPlugin>,
    ) -> Result<(), PluginError> {
        register_singleton_hook(
            &mut self.mode_session,
            &self.registering_plugin_id,
            "mode session capability",
            "mode_session",
            provider,
        )
    }

    fn add_mode_native_tools(
        &mut self,
        provider: Arc<dyn ModeNativeToolsPlugin>,
    ) -> Result<(), PluginError> {
        push_registered_hook(
            &mut self.mode_native_tools,
            &self.registering_plugin_id,
            provider,
        );
        Ok(())
    }

    fn add_mode_protocol_driver(
        &mut self,
        provider: Arc<dyn ModeProtocolDriverPlugin>,
    ) -> Result<(), PluginError> {
        register_singleton_hook(
            &mut self.mode_protocol_driver,
            &self.registering_plugin_id,
            "mode protocol driver capability",
            "mode_protocol_driver",
            provider,
        )
    }
}
