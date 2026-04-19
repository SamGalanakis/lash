use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::sync::{Mutex as StdMutex, Weak};

use sha2::{Digest, Sha256};
use tokio::task::JoinSet;

use super::*;
use crate::session_model::{fresh_message_id, plugin_message_to_message, reassign_part_ids};
use crate::{Message, Part, PartKind, PruneState};

#[derive(Clone)]
struct RegisteredHook<T> {
    plugin_id: String,
    hook: T,
}

#[derive(Clone)]
struct RegisteredExclusiveHook<T> {
    plugin_id: String,
    hook: T,
}

#[derive(Clone)]
pub(crate) struct RegisteredCommand {
    pub plugin_id: String,
    pub def: CommandDef,
    pub handler: CommandHandler,
}

fn current_plugin_id(registering_plugin_id: &Option<String>) -> String {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plugin_message_to_message_preserves_image_parts() {
        let plugin_message = PluginMessage {
            role: MessageRole::User,
            content: "before [Image #1] after".to_string(),
            parts: vec![
                Part {
                    id: String::new(),
                    kind: PartKind::Text,
                    content: "before ".to_string(),
                    attachment: None,
                    tool_call_id: None,
                    tool_name: None,
                    tool_item_id: None,
                    prune_state: PruneState::Intact,
                },
                Part {
                    id: String::new(),
                    kind: PartKind::Image,
                    content: String::new(),
                    attachment: Some(lash_sansio::session_model::message::PartAttachment {
                        mime: "image/png".to_string(),
                        url: lash_sansio::session_model::message::data_url_for_bytes(
                            "image/png",
                            &[1, 2, 3, 4],
                        ),
                        filename: None,
                    }),
                    tool_call_id: None,
                    tool_name: None,
                    tool_item_id: None,
                    prune_state: PruneState::Intact,
                },
                Part {
                    id: String::new(),
                    kind: PartKind::Text,
                    content: " after".to_string(),
                    attachment: None,
                    tool_call_id: None,
                    tool_name: None,
                    tool_item_id: None,
                    prune_state: PruneState::Intact,
                },
            ],
            images: Vec::new(),
            user_input: None,
        };

        let message = plugin_message_to_message(&plugin_message, plugin_message.user_input.clone());

        assert_eq!(message.parts.len(), 3);
        assert!(matches!(message.parts[1].kind, PartKind::Image));
        assert!(message.parts[1].attachment.is_some());
        assert_eq!(message.parts[0].content, "before ");
        assert_eq!(message.parts[2].content, " after");
    }
}

fn normalize_message_ids(messages: &mut [Message]) {
    for message in messages.iter_mut() {
        if message.id.is_empty() {
            message.id = fresh_message_id();
        }
        if message.parts.is_empty() {
            message.parts.push(Part {
                id: String::new(),
                kind: PartKind::Text,
                content: String::new(),
                attachment: None,
                tool_call_id: None,
                tool_name: None,
                tool_item_id: None,
                prune_state: PruneState::Intact,
            });
        }
        if !matches!(message.role, MessageRole::User) {
            message.user_input = None;
        }
        let message_id = message.id.clone();
        reassign_part_ids(&message_id, &mut message.parts);
    }
}

pub struct PluginRegistrar {
    tool_names: BTreeSet<String>,
    tool_providers: Vec<Arc<dyn ToolProvider>>,
    prompt_contributors: Vec<RegisteredHook<PromptContributor>>,
    prompt_request_hooks: Vec<RegisteredHook<PromptRequestHook>>,
    tool_surface_contributors: Vec<RegisteredHook<ToolSurfaceContributor>>,
    before_turn_hooks: Vec<RegisteredHook<BeforeTurnHook>>,
    before_tool_call_hooks: Vec<RegisteredHook<BeforeToolCallHook>>,
    after_tool_call_hooks: Vec<RegisteredHook<AfterToolCallHook>>,
    after_turn_hooks: Vec<RegisteredHook<AfterTurnHook>>,
    checkpoint_hooks: Vec<RegisteredHook<CheckpointHook>>,
    assistant_stream_hooks: Vec<RegisteredHook<AssistantStreamHook>>,
    assistant_response_hooks: Vec<RegisteredHook<AssistantResponseHook>>,
    tool_result_projectors:
        BTreeMap<ToolResultProjectionHook, RegisteredExclusiveHook<ToolResultProjector>>,
    runtime_event_hooks: Vec<PluginRuntimeEventHook>,
    session_config_mutators: Vec<SessionConfigMutator>,
    external_ops: BTreeMap<String, RegisteredExternalOp>,
    commands: BTreeMap<String, RegisteredCommand>,
    monitor_specs: Vec<PluginOwned<crate::MonitorSpec>>,
    turn_context_transforms: Vec<(i32, Arc<dyn TurnContextTransform>)>,
    history_rewriters: Vec<(i32, Arc<dyn HistoryRewriter>)>,
    mode_session: Option<RegisteredExclusiveHook<Arc<dyn ModeSessionPlugin>>>,
    mode_native_tools: Option<RegisteredExclusiveHook<Arc<dyn ModeNativeToolsPlugin>>>,
    registering_plugin_id: Option<String>,
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

pub(crate) struct ModeRegistrations<'a> {
    reg: &'a mut PluginRegistrar,
}

impl ModeRegistrations<'_> {
    pub(crate) fn session(self, provider: Arc<dyn ModeSessionPlugin>) -> Result<(), PluginError> {
        self.reg.add_mode_session(provider)
    }

    pub(crate) fn native_tools(
        self,
        provider: Arc<dyn ModeNativeToolsPlugin>,
    ) -> Result<(), PluginError> {
        self.reg.add_mode_native_tools(provider)
    }
}

impl PluginRegistrar {
    fn new() -> Self {
        Self {
            tool_names: BTreeSet::new(),
            tool_providers: Vec::new(),
            prompt_contributors: Vec::new(),
            prompt_request_hooks: Vec::new(),
            tool_surface_contributors: Vec::new(),
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
            mode_native_tools: None,
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

    pub(crate) fn mode(&mut self) -> ModeRegistrations<'_> {
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
        register_singleton_hook(
            &mut self.mode_native_tools,
            &self.registering_plugin_id,
            "mode native tool capability",
            "mode_native_tools",
            provider,
        )
    }
}

#[derive(Clone)]
pub struct PluginHost {
    factories: Arc<Vec<Arc<dyn PluginFactory>>>,
    dynamic_tools_enabled: bool,
    sessions: Arc<StdMutex<BTreeMap<String, Weak<PluginSession>>>>,
}

impl PluginHost {
    pub fn empty() -> Self {
        Self::new(Vec::new())
    }

    pub fn new(factories: Vec<Arc<dyn PluginFactory>>) -> Self {
        let override_ids: BTreeSet<&'static str> =
            factories.iter().map(|factory| factory.id()).collect();
        let mut all_factories = super::builtin_plugin_factories();
        if !override_ids.is_empty() {
            all_factories.retain(|factory| !override_ids.contains(factory.id()));
        }
        all_factories.extend(factories);
        Self {
            factories: Arc::new(all_factories),
            dynamic_tools_enabled: false,
            sessions: Arc::new(StdMutex::new(BTreeMap::new())),
        }
    }

    pub fn with_dynamic_tools(mut self) -> Self {
        self.dynamic_tools_enabled = true;
        self
    }

    pub fn isolated_registry(&self) -> Self {
        Self {
            factories: Arc::clone(&self.factories),
            dynamic_tools_enabled: self.dynamic_tools_enabled,
            sessions: Arc::new(StdMutex::new(BTreeMap::new())),
        }
    }

    pub fn factories(&self) -> &[Arc<dyn PluginFactory>] {
        self.factories.as_ref().as_slice()
    }

    pub fn supports_context_approach(&self, context_approach: &crate::ContextApproach) -> bool {
        let required = context_approach.kind();
        self.factories()
            .iter()
            .any(|factory| factory.supported_context_approaches().contains(&required))
    }

    pub fn build_standard_session(
        &self,
        session_id: impl Into<String>,
        snapshot: Option<&PluginSessionSnapshot>,
    ) -> Result<Arc<PluginSession>, PluginError> {
        self.build_session(
            session_id,
            ExecutionMode::Standard,
            crate::ContextApproach::default(),
            snapshot,
        )
    }

    pub fn build_session(
        &self,
        session_id: impl Into<String>,
        execution_mode: ExecutionMode,
        context_approach: crate::ContextApproach,
        snapshot: Option<&PluginSessionSnapshot>,
    ) -> Result<Arc<PluginSession>, PluginError> {
        self.build_session_with_surface(
            session_id,
            execution_mode,
            context_approach,
            snapshot,
            ToolSurfaceContribution::default(),
            None,
        )
    }

    pub fn build_session_with_surface(
        &self,
        session_id: impl Into<String>,
        execution_mode: ExecutionMode,
        context_approach: crate::ContextApproach,
        snapshot: Option<&PluginSessionSnapshot>,
        tool_surface_overlay: ToolSurfaceContribution,
        tool_snapshot: Option<crate::DynamicStateSnapshot>,
    ) -> Result<Arc<PluginSession>, PluginError> {
        if matches!(
            context_approach,
            crate::ContextApproach::ObservationalMemory(_)
        ) && !self.supports_context_approach(&context_approach)
        {
            return Err(PluginError::Registration(format!(
                "context approach `{}` requires a supporting plugin factory on this plugin host",
                context_approach.label()
            )));
        }
        let ctx = PluginSessionContext {
            session_id: session_id.into(),
            execution_mode,
            context_approach: context_approach.clone(),
        };
        let session_id = ctx.session_id.clone();
        let mut plugins = Vec::new();
        let mut reg = PluginRegistrar::new();
        for factory in self.factories() {
            let plugin = factory.build(&ctx)?;
            reg.registering_plugin_id = Some(plugin.id().to_string());
            plugin.register(&mut reg)?;
            reg.registering_plugin_id = None;
            plugins.push(plugin);
        }
        let mode_session = reg
            .mode_session
            .take()
            .ok_or_else(|| {
                PluginError::Registration(format!(
                    "missing mode session capability for {:?}",
                    execution_mode
                ))
            })?
            .hook;
        let mode_native_tools = reg
            .mode_native_tools
            .take()
            .ok_or_else(|| {
                PluginError::Registration(format!(
                    "missing mode native tool capability for {:?}",
                    execution_mode
                ))
            })?
            .hook;
        for def in mode_native_tools.definitions() {
            if !reg.tool_names.insert(def.name.clone()) {
                return Err(PluginError::Registration(format!(
                    "duplicate mode native tool name `{}`",
                    def.name
                )));
            }
        }
        let base_tools: Arc<dyn ToolProvider> = Arc::new(
            crate::tools::CompositeToolProvider::from_providers(reg.tool_providers.clone()),
        );
        let (tools, dynamic_tools) = if self.dynamic_tools_enabled {
            let dynamic = match tool_snapshot {
                Some(snapshot) => Arc::new(
                    crate::DynamicToolProvider::from_tool_provider(base_tools)
                        .map_err(|err| {
                            PluginError::Registration(format!(
                                "failed to build dynamic tool provider: {err}"
                            ))
                        })?
                        .fork_with_snapshot(snapshot)
                        .map_err(|err| {
                            PluginError::Session(format!(
                                "tool state cannot be applied to this plugin host session: {err}"
                            ))
                        })?,
                ),
                None => Arc::new(
                    crate::DynamicToolProvider::from_tool_provider(base_tools).map_err(|err| {
                        PluginError::Registration(format!(
                            "failed to build dynamic tool provider: {err}"
                        ))
                    })?,
                ),
            };
            (Arc::clone(&dynamic) as Arc<dyn ToolProvider>, Some(dynamic))
        } else if tool_snapshot.is_some() {
            return Err(PluginError::Session(
                "tool state requires dynamic tools on this plugin host".to_string(),
            ));
        } else {
            (base_tools, None)
        };

        let session = Arc::new(PluginSession {
            host: self.clone(),
            session_id: ctx.session_id,
            execution_mode,
            plugins,
            tools,
            dynamic_tools,
            tool_surface_overlay,
            prompt_contributors: reg.prompt_contributors,
            prompt_request_hooks: reg.prompt_request_hooks,
            tool_surface_contributors: reg.tool_surface_contributors,
            before_turn_hooks: reg.before_turn_hooks,
            before_tool_call_hooks: reg.before_tool_call_hooks,
            after_tool_call_hooks: reg.after_tool_call_hooks,
            after_turn_hooks: reg.after_turn_hooks,
            checkpoint_hooks: reg.checkpoint_hooks,
            assistant_stream_hooks: reg.assistant_stream_hooks,
            assistant_response_hooks: reg.assistant_response_hooks,
            tool_result_projectors: reg.tool_result_projectors,
            runtime_event_hooks: reg.runtime_event_hooks,
            session_config_mutators: reg.session_config_mutators,
            external_ops: reg.external_ops,
            commands: reg.commands,
            monitor_specs: reg.monitor_specs,
            turn_context_transforms: {
                let mut list = reg.turn_context_transforms;
                list.sort_by(|a, b| b.0.cmp(&a.0));
                list.into_iter().map(|(_, t)| t).collect()
            },
            history_rewriters: {
                let mut list = reg.history_rewriters;
                list.sort_by(|a, b| b.0.cmp(&a.0));
                list.into_iter().map(|(_, r)| r).collect()
            },
            mode_session,
            mode_native_tools,
        });
        self.register_session(&session_id, &session)?;
        let ready = SessionReadyContext {
            session_id: session.session_id.clone(),
            execution_mode,
            context_approach,
            host: self.clone(),
        };
        for plugin in &session.plugins {
            plugin.session_ready(ready.clone())?;
        }
        if let Some(snapshot) = snapshot {
            session.restore(snapshot)?;
        }
        Ok(session)
    }

    pub async fn invoke_external_sessionless(
        &self,
        name: &str,
        args: serde_json::Value,
    ) -> Result<ToolResult, PluginError> {
        let session = self.build_standard_session(
            format!("__external__-{}", uuid::Uuid::new_v4().simple()),
            None,
        )?;
        session
            .invoke_external(name, args, None, false, Arc::new(NoopSessionManager))
            .await
            .map_err(|err| PluginError::Invoke(err.to_string()))
    }

    fn register_session(
        &self,
        session_id: &str,
        session: &Arc<PluginSession>,
    ) -> Result<(), PluginError> {
        let mut sessions = self.sessions.lock().map_err(|_| {
            PluginError::Session("plugin host session registry poisoned".to_string())
        })?;
        if let Some(existing) = sessions.get(session_id).and_then(Weak::upgrade) {
            if !Arc::ptr_eq(&existing, session) {
                return Err(PluginError::Session(format!(
                    "session `{session_id}` is already registered on this plugin host"
                )));
            }
            return Ok(());
        }
        sessions.insert(session_id.to_string(), Arc::downgrade(session));
        Ok(())
    }

    pub fn unregister_session(&self, session_id: &str) -> Result<(), PluginError> {
        let mut sessions = self.sessions.lock().map_err(|_| {
            PluginError::Session("plugin host session registry poisoned".to_string())
        })?;
        sessions.remove(session_id);
        Ok(())
    }

    pub fn session(&self, session_id: &str) -> Result<Arc<PluginSession>, ExternalInvokeError> {
        let mut sessions = self
            .sessions
            .lock()
            .map_err(|_| ExternalInvokeError::SessionRegistryPoisoned)?;
        let Some(weak) = sessions.get(session_id).cloned() else {
            return Err(ExternalInvokeError::UnknownSession(session_id.to_string()));
        };
        match weak.upgrade() {
            Some(session) => Ok(session),
            None => {
                sessions.remove(session_id);
                Err(ExternalInvokeError::UnknownSession(session_id.to_string()))
            }
        }
    }

    pub async fn invoke_external_for_session(
        &self,
        session_id: &str,
        name: &str,
        args: serde_json::Value,
        host: Arc<dyn SessionManager>,
    ) -> Result<ToolResult, ExternalInvokeError> {
        let session = self.session(session_id)?;
        session
            .invoke_external(name, args, Some(session_id.to_string()), false, host)
            .await
    }

    pub fn monitor_specs_for_session(
        &self,
        session_id: &str,
    ) -> Result<Vec<crate::PluginOwned<crate::MonitorSpec>>, ExternalInvokeError> {
        Ok(self.session(session_id)?.monitor_specs().to_vec())
    }
}

pub struct PluginSession {
    host: PluginHost,
    session_id: String,
    execution_mode: ExecutionMode,
    plugins: Vec<Arc<dyn SessionPlugin>>,
    tools: Arc<dyn ToolProvider>,
    dynamic_tools: Option<Arc<crate::DynamicToolProvider>>,
    tool_surface_overlay: ToolSurfaceContribution,
    prompt_contributors: Vec<RegisteredHook<PromptContributor>>,
    prompt_request_hooks: Vec<RegisteredHook<PromptRequestHook>>,
    tool_surface_contributors: Vec<RegisteredHook<ToolSurfaceContributor>>,
    before_turn_hooks: Vec<RegisteredHook<BeforeTurnHook>>,
    before_tool_call_hooks: Vec<RegisteredHook<BeforeToolCallHook>>,
    after_tool_call_hooks: Vec<RegisteredHook<AfterToolCallHook>>,
    after_turn_hooks: Vec<RegisteredHook<AfterTurnHook>>,
    checkpoint_hooks: Vec<RegisteredHook<CheckpointHook>>,
    assistant_stream_hooks: Vec<RegisteredHook<AssistantStreamHook>>,
    assistant_response_hooks: Vec<RegisteredHook<AssistantResponseHook>>,
    tool_result_projectors:
        BTreeMap<ToolResultProjectionHook, RegisteredExclusiveHook<ToolResultProjector>>,
    runtime_event_hooks: Vec<PluginRuntimeEventHook>,
    session_config_mutators: Vec<SessionConfigMutator>,
    external_ops: BTreeMap<String, RegisteredExternalOp>,
    commands: BTreeMap<String, RegisteredCommand>,
    monitor_specs: Vec<PluginOwned<crate::MonitorSpec>>,
    turn_context_transforms: Vec<Arc<dyn TurnContextTransform>>,
    history_rewriters: Vec<Arc<dyn HistoryRewriter>>,
    mode_session: Arc<dyn ModeSessionPlugin>,
    mode_native_tools: Arc<dyn ModeNativeToolsPlugin>,
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

        normalize_message_ids(messages.make_mut());
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
        let directives = self
            .after_turn(TurnResultHookContext {
                session_id: session_id.clone(),
                turn: Arc::new(crate::plugin::TurnResultSummary::from_assembled(&turn)),
                host: Arc::clone(&host),
            })
            .await?;
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
        if let Some(messages) = updated_messages.as_mut() {
            normalize_message_ids(messages.make_mut());
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

#[derive(Debug, thiserror::Error)]
pub enum ExternalInvokeError {
    #[error("unknown external invoke `{0}`")]
    Unknown(String),
    #[error("unknown plugin session `{0}`")]
    UnknownSession(String),
    #[error("external invoke `{0}` requires a session")]
    MissingSession(String),
    #[error("external invoke `{0}` does not accept a session")]
    UnexpectedSession(String),
    #[error("plugin session registry is unavailable")]
    SessionRegistryPoisoned,
}

#[derive(Clone)]
pub struct RuntimeServices {
    pub plugins: Arc<PluginSession>,
    pub turn_injection_bridge: crate::session::TurnInjectionBridge,
    pub turn_input_injection_bridge: crate::session::TurnInputInjectionBridge,
    pub(crate) store: Option<Arc<dyn crate::store::RuntimeStore>>,
}

#[derive(Clone)]
pub struct PersistentRuntimeServices(RuntimeServices);

impl std::ops::Deref for PersistentRuntimeServices {
    type Target = RuntimeServices;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

struct EmptySnapshotReader;

impl SnapshotReader for EmptySnapshotReader {
    fn read_blob(&self, _name: &str) -> Option<&[u8]> {
        None
    }
}

pub(crate) struct NoopSessionManager;

#[async_trait::async_trait]
impl SessionManager for NoopSessionManager {
    async fn snapshot_current(&self) -> Result<SessionSnapshot, PluginError> {
        Err(PluginError::Session(
            "session snapshots are unavailable in this runtime".to_string(),
        ))
    }

    async fn snapshot_session(&self, _session_id: &str) -> Result<SessionSnapshot, PluginError> {
        Err(PluginError::Session(
            "session lookup is unavailable in this runtime".to_string(),
        ))
    }

    async fn tool_catalog(&self, _session_id: &str) -> Result<Vec<serde_json::Value>, PluginError> {
        Err(PluginError::Session(
            "tool catalogs are unavailable in this runtime".to_string(),
        ))
    }

    async fn create_session(
        &self,
        _request: SessionCreateRequest,
    ) -> Result<SessionHandle, PluginError> {
        Err(PluginError::Session(
            "session creation is unavailable in this runtime".to_string(),
        ))
    }

    async fn close_session(&self, _session_id: &str) -> Result<(), PluginError> {
        Err(PluginError::Session(
            "session closing is unavailable in this runtime".to_string(),
        ))
    }

    async fn start_turn_stream(
        &self,
        _session_id: &str,
        _input: TurnInput,
    ) -> Result<crate::plugin::SessionTurnHandle, PluginError> {
        Err(PluginError::Session(
            "session execution is unavailable in this runtime".to_string(),
        ))
    }

    async fn await_turn(&self, _turn_id: &str) -> Result<AssembledTurn, PluginError> {
        Err(PluginError::Session(
            "session execution is unavailable in this runtime".to_string(),
        ))
    }

    async fn cancel_turn(&self, _turn_id: &str) -> Result<(), PluginError> {
        Err(PluginError::Session(
            "session execution is unavailable in this runtime".to_string(),
        ))
    }
}

impl RuntimeServices {
    pub fn new(plugins: Arc<PluginSession>) -> Self {
        Self::new_with_bridges(
            plugins,
            crate::session::TurnInjectionBridge::new(),
            crate::session::TurnInputInjectionBridge::new(),
        )
    }

    pub fn new_with_bridges(
        plugins: Arc<PluginSession>,
        turn_injection_bridge: crate::session::TurnInjectionBridge,
        turn_input_injection_bridge: crate::session::TurnInputInjectionBridge,
    ) -> Self {
        Self {
            plugins,
            turn_injection_bridge,
            turn_input_injection_bridge,
            store: None,
        }
    }
}

impl PersistentRuntimeServices {
    pub fn new(plugins: Arc<PluginSession>, store: Arc<dyn crate::store::RuntimeStore>) -> Self {
        Self::new_with_bridges(
            plugins,
            crate::session::TurnInjectionBridge::new(),
            crate::session::TurnInputInjectionBridge::new(),
            store,
        )
    }

    pub fn new_with_bridges(
        plugins: Arc<PluginSession>,
        turn_injection_bridge: crate::session::TurnInjectionBridge,
        turn_input_injection_bridge: crate::session::TurnInputInjectionBridge,
        store: Arc<dyn crate::store::RuntimeStore>,
    ) -> Self {
        Self(RuntimeServices {
            plugins,
            turn_injection_bridge,
            turn_input_injection_bridge,
            store: Some(store),
        })
    }

    pub(crate) fn into_runtime_services(self) -> RuntimeServices {
        self.0
    }

    pub fn store(&self) -> Arc<dyn crate::store::RuntimeStore> {
        self.0
            .store
            .as_ref()
            .expect("persistent runtime services must carry a store")
            .clone()
    }
}
