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

#[derive(Clone)]
pub(crate) struct RegisteredCommand {
    pub plugin_id: String,
    pub def: CommandDef,
    pub handler: CommandHandler,
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

fn register_exclusive_hook<K, H>(
    hooks: &mut BTreeMap<K, RegisteredExclusiveHook<H>>,
    registering_plugin_id: &Option<String>,
    hook_kind: &str,
    hook_name: &str,
    key: K,
    hook: H,
) -> Result<(), PluginError>
where
    K: Ord,
{
    let plugin_id = exclusive_hook_owner(
        hooks
            .get(&key)
            .map(|registered| registered.plugin_id.as_str()),
        registering_plugin_id,
        hook_kind,
        hook_name,
    )?;
    hooks.insert(key, RegisteredExclusiveHook { plugin_id, hook });
    Ok(())
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
    pub(crate) prompt_request_hooks: Vec<RegisteredHook<PromptRequestHook>>,
    pub(crate) tool_surface_contributors: Vec<RegisteredHook<ToolSurfaceContributor>>,
    pub(crate) tool_discovery_contributors: Vec<RegisteredHook<ToolDiscoveryContributor>>,
    pub(crate) before_turn_hooks: Vec<RegisteredHook<BeforeTurnHook>>,
    pub(crate) before_tool_call_hooks: Vec<RegisteredHook<BeforeToolCallHook>>,
    pub(crate) after_tool_call_hooks: Vec<RegisteredHook<AfterToolCallHook>>,
    pub(crate) after_turn_hooks: Vec<RegisteredHook<AfterTurnHook>>,
    pub(crate) checkpoint_hooks: Vec<RegisteredHook<CheckpointHook>>,
    pub(crate) assistant_stream_hooks: Vec<RegisteredHook<AssistantStreamHook>>,
    pub(crate) assistant_response_hooks: Vec<RegisteredHook<AssistantResponseHook>>,
    pub(crate) tool_result_projectors:
        BTreeMap<ToolResultProjectionHook, RegisteredExclusiveHook<ToolResultProjector>>,
    pub(crate) runtime_event_hooks: Vec<PluginRuntimeEventHook>,
    pub(crate) session_config_mutators: Vec<SessionConfigMutator>,
    pub(crate) external_ops: BTreeMap<String, RegisteredExternalOp>,
    pub(crate) commands: BTreeMap<String, RegisteredCommand>,
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

    pub fn on_request(self, hook: PromptRequestHook) {
        self.reg.add_prompt_request_hook(hook);
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
    pub fn projector(
        self,
        hook_name: ToolResultProjectionHook,
        hook: ToolResultProjector,
    ) -> Result<(), PluginError> {
        self.reg.add_tool_result_projector(hook_name, hook)
    }
}

pub struct SessionRegistrations<'a> {
    reg: &'a mut PluginRegistrar,
}

impl SessionRegistrations<'_> {
    pub fn on_event(self, hook: PluginRuntimeEventHook) {
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

pub struct ExternalRegistrations<'a> {
    reg: &'a mut PluginRegistrar,
}

impl ExternalRegistrations<'_> {
    pub fn op(self, def: ExternalOpDef, handler: ExternalInvokeHandler) -> Result<(), PluginError> {
        self.reg.add_external_op(def, handler)
    }
}

pub struct CommandRegistrations<'a> {
    reg: &'a mut PluginRegistrar,
}

impl CommandRegistrations<'_> {
    pub fn register(self, def: CommandDef, handler: CommandHandler) -> Result<(), PluginError> {
        self.reg.add_command(def, handler)
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
            prompt_request_hooks: Vec::new(),
            tool_surface_contributors: Vec::new(),
            tool_discovery_contributors: Vec::new(),
            before_turn_hooks: Vec::new(),
            before_tool_call_hooks: Vec::new(),
            after_tool_call_hooks: Vec::new(),
            after_turn_hooks: Vec::new(),
            checkpoint_hooks: Vec::new(),
            assistant_stream_hooks: Vec::new(),
            assistant_response_hooks: Vec::new(),
            tool_result_projectors: BTreeMap::new(),
            runtime_event_hooks: Vec::new(),
            session_config_mutators: Vec::new(),
            external_ops: BTreeMap::new(),
            commands: BTreeMap::new(),
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

    pub fn external(&mut self) -> ExternalRegistrations<'_> {
        ExternalRegistrations { reg: self }
    }

    pub fn commands(&mut self) -> CommandRegistrations<'_> {
        CommandRegistrations { reg: self }
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
        for def in provider.definitions() {
            if !self.tool_names.insert(def.name.clone()) {
                return Err(PluginError::Registration(format!(
                    "duplicate plugin tool name `{}`",
                    def.name
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

    fn add_prompt_request_hook(&mut self, hook: PromptRequestHook) {
        push_registered_hook(
            &mut self.prompt_request_hooks,
            &self.registering_plugin_id,
            hook,
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

    fn add_tool_result_projector(
        &mut self,
        hook_name: ToolResultProjectionHook,
        hook: ToolResultProjector,
    ) -> Result<(), PluginError> {
        register_exclusive_hook(
            &mut self.tool_result_projectors,
            &self.registering_plugin_id,
            "tool result projector",
            hook_name.as_str(),
            hook_name,
            hook,
        )
    }

    fn add_external_op(
        &mut self,
        def: ExternalOpDef,
        handler: ExternalInvokeHandler,
    ) -> Result<(), PluginError> {
        if self.external_ops.contains_key(&def.name) {
            return Err(PluginError::Registration(format!(
                "duplicate external invoke name `{}`",
                def.name
            )));
        }
        self.external_ops
            .insert(def.name.clone(), RegisteredExternalOp { def, handler });
        Ok(())
    }

    fn add_command(&mut self, def: CommandDef, handler: CommandHandler) -> Result<(), PluginError> {
        let key = def.name.to_string();
        if let Some(existing) = self.commands.get(&key) {
            return Err(PluginError::Registration(format!(
                "duplicate slash command `{}`: `{}` conflicts with `{}`",
                def.name,
                current_plugin_id(&self.registering_plugin_id),
                existing.plugin_id,
            )));
        }
        let plugin_id = current_plugin_id(&self.registering_plugin_id);
        self.commands.insert(
            key,
            RegisteredCommand {
                plugin_id,
                def,
                handler,
            },
        );
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
