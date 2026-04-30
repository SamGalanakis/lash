//! Plugin registration: `PluginSpec` (the declarative bundle of all a
//! plugin's hooks), the `PluginFactory` / `SessionPlugin` traits
//! plugin crates implement, and the two convenience factories
//! (`StaticPluginFactory`, `PluginSpecFactory`) + the `SpecPlugin`
//! glue that walks a spec and wires each field into the registrar.
//!
//! Split out of `plugin/mod.rs` for file size; outer path preserved by
//! `pub use` in `plugin/mod.rs`.

use std::collections::BTreeMap;
use std::sync::Arc;

use super::{
    AfterToolCallHook, AfterTurnHook, AssistantResponseHook, AssistantStreamHook,
    BeforeToolCallHook, BeforeTurnHook, CheckpointHook, CommandDef, CommandHandler,
    ExternalInvokeHandler, ExternalOpDef, HistoryRewriter, PluginError, PluginHost,
    PluginRegistrar, PluginRuntimeEventHook, PluginSnapshotMeta, PromptContributor,
    PromptRequestHook, SessionConfigMutator, SnapshotReader, SnapshotWriter,
    ToolDiscoveryContributor, ToolResultProjectionHook, ToolResultProjector,
    ToolSurfaceContributor, TurnContextTransform,
};
use crate::{ContextApproachKind, ExecutionMode, ToolProvider};

#[derive(Clone, Default)]
pub struct PluginSpec {
    pub tool_providers: Vec<Arc<dyn ToolProvider>>,
    pub prompt_contributors: Vec<PromptContributor>,
    pub prompt_request_hooks: Vec<PromptRequestHook>,
    pub tool_surface_contributors: Vec<ToolSurfaceContributor>,
    pub tool_discovery_contributors: Vec<ToolDiscoveryContributor>,
    pub before_turn_hooks: Vec<BeforeTurnHook>,
    pub before_tool_call_hooks: Vec<BeforeToolCallHook>,
    pub after_tool_call_hooks: Vec<AfterToolCallHook>,
    pub after_turn_hooks: Vec<AfterTurnHook>,
    pub checkpoint_hooks: Vec<CheckpointHook>,
    pub assistant_stream_hooks: Vec<AssistantStreamHook>,
    pub assistant_response_hooks: Vec<AssistantResponseHook>,
    pub tool_result_projectors: BTreeMap<ToolResultProjectionHook, ToolResultProjector>,
    pub runtime_event_hooks: Vec<PluginRuntimeEventHook>,
    pub session_config_mutators: Vec<SessionConfigMutator>,
    pub external_ops: Vec<(ExternalOpDef, ExternalInvokeHandler)>,
    pub commands: Vec<(CommandDef, CommandHandler)>,
    pub turn_context_transforms: Vec<(i32, Arc<dyn TurnContextTransform>)>,
    pub history_rewriters: Vec<(i32, Arc<dyn HistoryRewriter>)>,
}

impl PluginSpec {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_tool_provider(mut self, provider: Arc<dyn ToolProvider>) -> Self {
        self.tool_providers.push(provider);
        self
    }

    pub fn with_prompt_contributor(mut self, contributor: PromptContributor) -> Self {
        self.prompt_contributors.push(contributor);
        self
    }

    pub fn with_prompt_request(mut self, hook: PromptRequestHook) -> Self {
        self.prompt_request_hooks.push(hook);
        self
    }

    pub fn with_tool_surface_contributor(mut self, contributor: ToolSurfaceContributor) -> Self {
        self.tool_surface_contributors.push(contributor);
        self
    }

    pub fn with_tool_discovery_contributor(
        mut self,
        contributor: ToolDiscoveryContributor,
    ) -> Self {
        self.tool_discovery_contributors.push(contributor);
        self
    }

    pub fn with_before_turn(mut self, hook: BeforeTurnHook) -> Self {
        self.before_turn_hooks.push(hook);
        self
    }

    pub fn with_before_tool_call(mut self, hook: BeforeToolCallHook) -> Self {
        self.before_tool_call_hooks.push(hook);
        self
    }

    pub fn with_after_tool_call(mut self, hook: AfterToolCallHook) -> Self {
        self.after_tool_call_hooks.push(hook);
        self
    }

    pub fn with_after_turn(mut self, hook: AfterTurnHook) -> Self {
        self.after_turn_hooks.push(hook);
        self
    }

    pub fn with_checkpoint(mut self, hook: CheckpointHook) -> Self {
        self.checkpoint_hooks.push(hook);
        self
    }

    pub fn with_assistant_stream(mut self, hook: AssistantStreamHook) -> Self {
        self.assistant_stream_hooks.push(hook);
        self
    }

    pub fn with_assistant_response(mut self, hook: AssistantResponseHook) -> Self {
        self.assistant_response_hooks.push(hook);
        self
    }

    pub fn with_tool_result_projector(
        mut self,
        hook: ToolResultProjectionHook,
        projector: ToolResultProjector,
    ) -> Self {
        self.tool_result_projectors.insert(hook, projector);
        self
    }

    pub fn with_runtime_event(mut self, hook: PluginRuntimeEventHook) -> Self {
        self.runtime_event_hooks.push(hook);
        self
    }

    pub fn with_session_config_mutator(mut self, hook: SessionConfigMutator) -> Self {
        self.session_config_mutators.push(hook);
        self
    }

    pub fn with_external_op(mut self, def: ExternalOpDef, handler: ExternalInvokeHandler) -> Self {
        self.external_ops.push((def, handler));
        self
    }

    pub fn with_command(mut self, def: CommandDef, handler: CommandHandler) -> Self {
        self.commands.push((def, handler));
        self
    }

    pub fn with_turn_context_transform(
        mut self,
        priority: i32,
        transform: Arc<dyn TurnContextTransform>,
    ) -> Self {
        self.turn_context_transforms.push((priority, transform));
        self
    }

    pub fn with_history_rewriter(
        mut self,
        priority: i32,
        rewriter: Arc<dyn HistoryRewriter>,
    ) -> Self {
        self.history_rewriters.push((priority, rewriter));
        self
    }
}

#[derive(Clone, Debug)]
pub struct PluginSessionContext {
    pub session_id: String,
    pub execution_mode: ExecutionMode,
    pub context_approach: crate::ContextApproach,
    /// Session id of the caller that created this one. `None` identifies
    /// a root session; any subagent / compaction / forked-child session
    /// carries the parent here so plugin factories can gate themselves
    /// on root-only behavior (e.g. `update_plan`'s sticky plan dock).
    pub parent_session_id: Option<String>,
}

impl PluginSessionContext {
    /// Returns `true` when this context represents a root session, not a
    /// subagent or internal child. Plugins that should only surface in
    /// user-facing top-level turns check this in their `build`.
    pub fn is_root_session(&self) -> bool {
        self.parent_session_id.is_none()
    }
}

#[derive(Clone)]
pub struct SessionReadyContext {
    pub session_id: String,
    pub execution_mode: ExecutionMode,
    pub context_approach: crate::ContextApproach,
    pub host: PluginHost,
}

pub trait SessionPlugin: Send + Sync {
    fn id(&self) -> &'static str;

    fn version(&self) -> &'static str {
        "1"
    }

    fn register(&self, reg: &mut PluginRegistrar) -> Result<(), PluginError>;

    fn snapshot(
        &self,
        _writer: &mut dyn SnapshotWriter,
    ) -> Result<PluginSnapshotMeta, PluginError> {
        Ok(PluginSnapshotMeta {
            plugin_id: self.id().to_string(),
            plugin_version: self.version().to_string(),
            revision: self.snapshot_revision(),
            state: None,
        })
    }

    fn snapshot_revision(&self) -> u64 {
        0
    }

    fn restore(
        &self,
        _meta: &PluginSnapshotMeta,
        _reader: &dyn SnapshotReader,
    ) -> Result<(), PluginError> {
        Ok(())
    }

    fn session_ready(&self, _ctx: SessionReadyContext) -> Result<(), PluginError> {
        Ok(())
    }
}

/// Registers a plugin with the runtime and produces a per-session
/// `SessionPlugin` instance for each new session.
///
/// # Cheap-build / stateful-factory contract
///
/// `build(ctx)` **must be cheap**. It runs on the hot path every time
/// a new session is created (including subagents, forked children,
/// and compaction children) and any latency here is paid per session.
///
/// Specifically, `build` must **not**:
/// - perform any I/O (disk reads, HTTP calls, DB queries),
/// - compile regexes, templates, or schemas,
/// - open network connections or initialize connection pools,
/// - load models, parse large config files, or allocate large buffers,
/// - block the current thread for non-trivial work.
///
/// Expensive state belongs on the `PluginFactory` struct itself,
/// wrapped in `Arc` so it can be cheaply cloned into per-session
/// closures. The `PluginFactory` is constructed once by the embedder
/// and held in the `RuntimeEnvironment`; its fields outlive every
/// session. Hooks captured into a `PluginSpec` are closures that
/// clone the `Arc`s off `self` and reference the shared state
/// directly, so every session sees the same pool / cache / compiled
/// artifact without rebuilding it.
///
/// The typical shape is:
/// ```ignore
/// pub struct MyFactory {
///     pool: Arc<ConnectionPool>,          // expensive, built once
///     compiled: Arc<Regex>,               // expensive, built once
/// }
///
/// impl PluginFactory for MyFactory {
///     fn id(&self) -> &'static str { "my_plugin" }
///
///     fn build(&self, _ctx: &PluginSessionContext)
///         -> Result<Arc<dyn SessionPlugin>, PluginError>
///     {
///         // Cheap: clone Arcs, assemble spec, wrap in SpecPlugin.
///         let pool = Arc::clone(&self.pool);
///         let spec = PluginSpec::new().with_before_turn(Arc::new(move |_ctx| {
///             let pool = Arc::clone(&pool);
///             Box::pin(async move { /* use pool */ Ok(vec![]) })
///         }));
///         Ok(Arc::new(SpecPluginFromSpec::new("my_plugin", spec)))
///     }
/// }
/// ```
pub trait PluginFactory: Send + Sync {
    fn id(&self) -> &'static str;
    fn supported_context_approaches(&self) -> &'static [ContextApproachKind] {
        &[]
    }
    /// Produce a session-scoped plugin. **Must be cheap** — see the
    /// trait-level docs for the full contract.
    fn build(&self, ctx: &PluginSessionContext) -> Result<Arc<dyn SessionPlugin>, PluginError>;
}

pub type PluginSpecBuilder =
    Arc<dyn Fn(&PluginSessionContext) -> Result<PluginSpec, PluginError> + Send + Sync>;

pub struct PluginSpecFactory {
    id: &'static str,
    builder: PluginSpecBuilder,
}

impl PluginSpecFactory {
    pub fn new(id: &'static str, builder: PluginSpecBuilder) -> Self {
        Self { id, builder }
    }
}

pub struct StaticPluginFactory {
    id: &'static str,
    spec: PluginSpec,
}

impl StaticPluginFactory {
    pub fn new(id: &'static str, spec: PluginSpec) -> Self {
        Self { id, spec }
    }
}

struct SpecPlugin {
    id: &'static str,
    spec: PluginSpec,
}

impl PluginFactory for PluginSpecFactory {
    fn id(&self) -> &'static str {
        self.id
    }

    fn build(&self, ctx: &PluginSessionContext) -> Result<Arc<dyn SessionPlugin>, PluginError> {
        Ok(Arc::new(SpecPlugin {
            id: self.id,
            spec: (self.builder)(ctx)?,
        }))
    }
}

impl PluginFactory for StaticPluginFactory {
    fn id(&self) -> &'static str {
        self.id
    }

    fn build(&self, _ctx: &PluginSessionContext) -> Result<Arc<dyn SessionPlugin>, PluginError> {
        Ok(Arc::new(SpecPlugin {
            id: self.id,
            spec: self.spec.clone(),
        }))
    }
}

impl SessionPlugin for SpecPlugin {
    fn id(&self) -> &'static str {
        self.id
    }

    fn register(&self, reg: &mut PluginRegistrar) -> Result<(), PluginError> {
        for provider in &self.spec.tool_providers {
            reg.tools().provider(Arc::clone(provider))?;
        }
        for contributor in &self.spec.prompt_contributors {
            reg.prompt().contribute(Arc::clone(contributor));
        }
        for hook in &self.spec.prompt_request_hooks {
            reg.prompt().on_request(Arc::clone(hook));
        }
        for contributor in &self.spec.tool_surface_contributors {
            reg.surface().contribute(Arc::clone(contributor));
        }
        for contributor in &self.spec.tool_discovery_contributors {
            reg.discovery().contribute(Arc::clone(contributor));
        }
        for hook in &self.spec.before_turn_hooks {
            reg.turn().before(Arc::clone(hook));
        }
        for hook in &self.spec.before_tool_call_hooks {
            reg.tool_calls().before(Arc::clone(hook));
        }
        for hook in &self.spec.after_tool_call_hooks {
            reg.tool_calls().after(Arc::clone(hook));
        }
        for hook in &self.spec.after_turn_hooks {
            reg.turn().after(Arc::clone(hook));
        }
        for hook in &self.spec.checkpoint_hooks {
            reg.turn().checkpoint(Arc::clone(hook));
        }
        for hook in &self.spec.assistant_stream_hooks {
            reg.output().stream(Arc::clone(hook));
        }
        for hook in &self.spec.assistant_response_hooks {
            reg.output().response(Arc::clone(hook));
        }
        for (hook, projector) in &self.spec.tool_result_projectors {
            reg.tool_results().projector(*hook, Arc::clone(projector))?;
        }
        for hook in &self.spec.runtime_event_hooks {
            reg.session().on_event(Arc::clone(hook));
        }
        for hook in &self.spec.session_config_mutators {
            reg.session().config_mutator(Arc::clone(hook));
        }
        for (def, handler) in &self.spec.external_ops {
            reg.external().op(def.clone(), Arc::clone(handler))?;
        }
        for (def, handler) in &self.spec.commands {
            reg.commands().register(def.clone(), Arc::clone(handler))?;
        }
        for (priority, transform) in &self.spec.turn_context_transforms {
            reg.history().prepare_turn(*priority, Arc::clone(transform));
        }
        for (priority, rewriter) in &self.spec.history_rewriters {
            reg.history().rewrite(*priority, Arc::clone(rewriter));
        }
        Ok(())
    }
}
