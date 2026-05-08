//! App-facing embedding facade for Lash.
//!
//! `lash-embed` is intentionally a small layer above the lower-level
//! `lash` runtime crates. Host applications own providers, persistence,
//! app state, HTTP protocols, auth, and frontend streaming; this crate
//! owns only the ergonomic core/session/turn API.

use std::any::Any;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt;
use std::path::PathBuf;
use std::sync::{Arc, Mutex as StdMutex};

use async_trait::async_trait;
use lash::plugin::StaticPluginFactory;
use lash::{
    ExecutionMode, LashRuntime, MessageRole, PersistedSessionState, PluginHost, PluginSpec,
    RuntimeEnvironment, SessionPolicy, TurnContext, TurnInput,
};
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

pub use lash::{
    AttachmentStore, InputItem, Message, ModeTurnOptions, PluginFactory, ProviderHandle, Residency,
    RuntimeError, RuntimePersistence, SanitizerPolicy, SessionError, SessionReadView,
    SessionStoreCreateRequest, SessionStoreFactory, SessionTaskExecutor, StandardContextApproach,
    TerminalOutputSource, TerminationPolicy, TokenUsage, TokioSessionTaskExecutor,
    ToolAvailability, ToolProvider, ToolResultView, TurnEvent, TurnEventSink, TurnIssue,
    TurnOutcome,
};
pub use lash_mode_rlm::RlmProjectedBindings;
pub use lash_trace::{TraceContext, TraceLevel, TraceSink};

pub type ToolState = lash::DynamicStateSnapshot;
pub type ToolSpec = lash::DynamicToolSpec;

pub trait EmbedPlugin: Send + Sync + 'static {
    const ID: &'static str;
    type SessionConfig: Clone + Send + Sync + 'static;
    type TurnInput: Clone + Send + Sync + 'static;

    fn factory(config: &Self::SessionConfig) -> Arc<dyn PluginFactory>;

    fn requires_turn_input(_config: &Self::SessionConfig) -> bool {
        false
    }
}

type PluginConfigMap<P> = Arc<StdMutex<HashMap<String, <P as EmbedPlugin>::SessionConfig>>>;

struct EmbedPluginFactory<P: EmbedPlugin> {
    configs: PluginConfigMap<P>,
}

impl<P: EmbedPlugin> EmbedPluginFactory<P> {
    fn new(configs: PluginConfigMap<P>) -> Self {
        Self { configs }
    }
}

impl<P: EmbedPlugin> lash::PluginFactory for EmbedPluginFactory<P> {
    fn id(&self) -> &'static str {
        P::ID
    }

    fn build(
        &self,
        ctx: &lash::PluginSessionContext,
    ) -> std::result::Result<Arc<dyn lash::SessionPlugin>, lash::PluginError> {
        let config = self
            .configs
            .lock()
            .map_err(|_| {
                lash::PluginError::Session(format!("plugin `{}` config lock poisoned", P::ID))
            })?
            .get(&ctx.session_id)
            .cloned();
        let Some(config) = config else {
            return Ok(Arc::new(InactiveEmbedPlugin { id: P::ID }));
        };
        P::factory(&config).build(ctx)
    }
}

struct InactiveEmbedPlugin {
    id: &'static str,
}

impl lash::SessionPlugin for InactiveEmbedPlugin {
    fn id(&self) -> &'static str {
        self.id
    }

    fn register(
        &self,
        _reg: &mut lash::PluginRegistrar,
    ) -> std::result::Result<(), lash::PluginError> {
        Ok(())
    }
}

/// Stable mode identifier exposed by the embedding facade.
#[derive(
    Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
pub struct ModeId(String);

impl ModeId {
    pub fn new(mode: impl Into<String>) -> Self {
        Self(mode.into())
    }

    pub fn standard() -> Self {
        Self("standard".to_string())
    }

    pub fn rlm() -> Self {
        Self("rlm".to_string())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    fn execution_mode(&self) -> ExecutionMode {
        ExecutionMode::new(self.0.clone())
    }
}

impl fmt::Display for ModeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Semantic mode preset installed on a [`LashCore`].
#[derive(Clone)]
pub struct ModePreset {
    mode_id: ModeId,
    factory: Arc<dyn PluginFactory>,
    standard_context_approach: Option<StandardContextApproach>,
}

impl ModePreset {
    pub fn standard() -> Self {
        Self {
            mode_id: ModeId::standard(),
            factory: Arc::new(lash_mode_standard::BuiltinStandardModePluginFactory::new()),
            standard_context_approach: Some(StandardContextApproach::default()),
        }
    }

    pub fn rlm() -> Self {
        Self {
            mode_id: ModeId::rlm(),
            factory: Arc::new(lash_mode_rlm::BuiltinRlmModePluginFactory::default()),
            standard_context_approach: None,
        }
    }

    pub fn rlm_with_config(config: lash_mode_rlm::RlmModePluginConfig) -> Self {
        Self {
            mode_id: ModeId::rlm(),
            factory: Arc::new(lash_mode_rlm::BuiltinRlmModePluginFactory::new(config)),
            standard_context_approach: None,
        }
    }

    pub fn mode_id(&self) -> &ModeId {
        &self.mode_id
    }
}

#[derive(Debug, thiserror::Error)]
pub enum EmbedError {
    #[error("no mode presets installed; call install_mode(ModePreset::...) first")]
    NoModesInstalled,
    #[error("default mode `{mode}` is not installed on this LashCore")]
    DefaultModeNotInstalled { mode: ModeId },
    #[error("mode `{mode}` is not installed on this LashCore")]
    ModeNotInstalled { mode: ModeId },
    #[error("max_context_tokens is required; hosts must supply explicit model metadata")]
    MissingMaxContextTokens,
    #[error("failed to create store for session `{session_id}`: {message}")]
    StoreFactory { session_id: String, message: String },
    #[error("RLM projected binding error: {0}")]
    RlmProjectedBinding(String),
    #[error("embed plugin `{plugin_id}` is not registered on this LashCore")]
    PluginNotRegistered { plugin_id: &'static str },
    #[error("missing required turn input for plugin `{plugin_id}`")]
    MissingPluginTurnInput { plugin_id: &'static str },
    #[error("embed plugin `{plugin_id}` config error: {message}")]
    PluginConfig {
        plugin_id: &'static str,
        message: String,
    },
    #[error("runtime session error: {0}")]
    Session(#[from] SessionError),
    #[error("runtime turn error: {0}")]
    Runtime(#[from] lash::RuntimeError),
}

pub type Result<T> = std::result::Result<T, EmbedError>;

/// Shared app-owned Lash core. Clone it freely; expensive infrastructure
/// lives behind `Arc`s inside the runtime environment.
#[derive(Clone)]
pub struct LashCore {
    env: RuntimeEnvironment,
    policy: SessionPolicy,
    modes: Arc<BTreeMap<ModeId, ModePreset>>,
    default_mode: ModeId,
    store_factory: Option<Arc<dyn SessionStoreFactory>>,
    embed_plugin_ids: Arc<BTreeSet<&'static str>>,
    embed_plugin_configs: Arc<HashMap<&'static str, Arc<dyn Any + Send + Sync>>>,
}

impl LashCore {
    pub fn builder() -> LashCoreBuilder {
        LashCoreBuilder::default()
    }

    pub fn standard() -> LashCoreBuilder {
        Self::builder()
            .install_mode(ModePreset::standard())
            .default_mode(ModeId::standard())
    }

    pub fn rlm() -> LashCoreBuilder {
        Self::builder()
            .install_mode(ModePreset::rlm())
            .default_mode(ModeId::rlm())
    }

    pub fn session(&self, session_id: impl Into<String>) -> SessionBuilder {
        SessionBuilder {
            core: self.clone(),
            session_id: session_id.into(),
            mode: None,
            parent_session_id: None,
            store: None,
            active_plugins: Vec::new(),
        }
    }

    pub fn installed_modes(&self) -> impl Iterator<Item = &ModeId> {
        self.modes.keys()
    }
}

#[derive(Default)]
pub struct LashCoreBuilder {
    modes: BTreeMap<ModeId, ModePreset>,
    default_mode: Option<ModeId>,
    provider: Option<ProviderHandle>,
    model: Option<String>,
    model_variant: Option<String>,
    max_context_tokens: Option<usize>,
    store_factory: Option<Arc<dyn SessionStoreFactory>>,
    attachment_store: Option<Arc<dyn AttachmentStore>>,
    tool_providers: Vec<Arc<dyn ToolProvider>>,
    plugin_factories: Vec<Arc<dyn PluginFactory>>,
    embed_plugin_ids: BTreeSet<&'static str>,
    embed_plugin_configs: HashMap<&'static str, Arc<dyn Any + Send + Sync>>,
    trace_sink: Option<Arc<dyn lash_trace::TraceSink>>,
    trace_level: Option<lash_trace::TraceLevel>,
    trace_context: Option<lash_trace::TraceContext>,
    residency: Option<Residency>,
    session_task_executor: Option<Arc<dyn SessionTaskExecutor>>,
    base_dir: Option<PathBuf>,
    sanitizer: Option<SanitizerPolicy>,
    termination: Option<TerminationPolicy>,
}

impl LashCoreBuilder {
    pub fn install_mode(mut self, preset: ModePreset) -> Self {
        let mode_id = preset.mode_id.clone();
        if self.default_mode.is_none() {
            self.default_mode = Some(mode_id.clone());
        }
        self.modes.insert(mode_id, preset);
        self
    }

    pub fn default_mode(mut self, mode: ModeId) -> Self {
        self.default_mode = Some(mode);
        self
    }

    pub fn provider(mut self, provider: ProviderHandle) -> Self {
        self.provider = Some(provider);
        self
    }

    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    pub fn model_variant(mut self, model_variant: impl Into<String>) -> Self {
        self.model_variant = Some(model_variant.into());
        self
    }

    pub fn max_context_tokens(mut self, max_context_tokens: usize) -> Self {
        self.max_context_tokens = Some(max_context_tokens);
        self
    }

    pub fn store_factory(mut self, store_factory: Arc<dyn SessionStoreFactory>) -> Self {
        self.store_factory = Some(store_factory);
        self
    }

    pub fn attachment_store(mut self, attachment_store: Arc<dyn AttachmentStore>) -> Self {
        self.attachment_store = Some(attachment_store);
        self
    }

    pub fn tools(mut self, tools: Arc<dyn ToolProvider>) -> Self {
        self.tool_providers.push(tools);
        self
    }

    pub fn plugin(mut self, plugin: Arc<dyn PluginFactory>) -> Self {
        self.plugin_factories.push(plugin);
        self
    }

    pub fn register_plugin<P: EmbedPlugin>(mut self) -> Self {
        let configs: PluginConfigMap<P> = Arc::new(StdMutex::new(HashMap::new()));
        self.embed_plugin_ids.insert(P::ID);
        self.embed_plugin_configs
            .insert(P::ID, configs.clone() as Arc<dyn Any + Send + Sync>);
        self.plugin_factories
            .push(Arc::new(EmbedPluginFactory::<P>::new(configs)) as Arc<dyn PluginFactory>);
        self
    }

    pub fn trace_sink(mut self, trace_sink: Option<Arc<dyn lash_trace::TraceSink>>) -> Self {
        self.trace_sink = trace_sink;
        self
    }

    pub fn trace_level(mut self, trace_level: lash_trace::TraceLevel) -> Self {
        self.trace_level = Some(trace_level);
        self
    }

    pub fn trace_context(mut self, trace_context: lash_trace::TraceContext) -> Self {
        self.trace_context = Some(trace_context);
        self
    }

    pub fn residency(mut self, residency: Residency) -> Self {
        self.residency = Some(residency);
        self
    }

    pub fn session_task_executor(mut self, executor: Arc<dyn SessionTaskExecutor>) -> Self {
        self.session_task_executor = Some(executor);
        self
    }

    pub fn base_dir(mut self, base_dir: impl Into<PathBuf>) -> Self {
        self.base_dir = Some(base_dir.into());
        self
    }

    pub fn sanitizer(mut self, sanitizer: SanitizerPolicy) -> Self {
        self.sanitizer = Some(sanitizer);
        self
    }

    pub fn termination(mut self, termination: TerminationPolicy) -> Self {
        self.termination = Some(termination);
        self
    }

    pub fn build(self) -> Result<LashCore> {
        if self.modes.is_empty() {
            return Err(EmbedError::NoModesInstalled);
        }
        let default_mode = self
            .default_mode
            .clone()
            .ok_or(EmbedError::NoModesInstalled)?;
        if !self.modes.contains_key(&default_mode) {
            return Err(EmbedError::DefaultModeNotInstalled { mode: default_mode });
        }
        let max_context_tokens = self
            .max_context_tokens
            .ok_or(EmbedError::MissingMaxContextTokens)?;
        let provider = self.provider.unwrap_or_default();
        let model = self
            .model
            .unwrap_or_else(|| provider.default_model().to_string());

        let default_preset = self
            .modes
            .get(&default_mode)
            .expect("default mode was validated");
        let policy = SessionPolicy {
            provider,
            model,
            model_variant: self.model_variant,
            max_context_tokens: Some(max_context_tokens),
            execution_mode: default_mode.execution_mode(),
            standard_context_approach: default_preset.standard_context_approach.clone(),
            ..SessionPolicy::default()
        };

        let mut factories = Vec::new();
        factories.extend(
            self.modes
                .values()
                .map(|preset| Arc::clone(&preset.factory)),
        );
        if !self.tool_providers.is_empty() {
            let spec = self
                .tool_providers
                .into_iter()
                .fold(PluginSpec::new(), PluginSpec::with_tool_provider);
            factories
                .push(Arc::new(StaticPluginFactory::new("embed_tools", spec))
                    as Arc<dyn PluginFactory>);
        }
        factories.extend(self.plugin_factories);

        let executor = self
            .session_task_executor
            .unwrap_or_else(|| Arc::new(TokioSessionTaskExecutor::default()));
        let mut env_builder = RuntimeEnvironment::builder()
            .with_plugin_host(Arc::new(
                PluginHost::new(factories)
                    .with_background_tasks()
                    .with_dynamic_tools(),
            ))
            .with_session_task_executor(executor);
        if let Some(attachment_store) = self.attachment_store {
            env_builder = env_builder.with_attachment_store(attachment_store);
        }
        if let Some(trace_sink) = self.trace_sink {
            env_builder = env_builder.with_trace_sink(Some(trace_sink));
        }
        if let Some(trace_level) = self.trace_level {
            env_builder = env_builder.with_trace_level(trace_level);
        }
        if let Some(trace_context) = self.trace_context {
            env_builder = env_builder.with_trace_context(trace_context);
        }
        if let Some(residency) = self.residency {
            env_builder = env_builder.with_residency(residency);
        }
        if let Some(base_dir) = self.base_dir {
            env_builder = env_builder.with_base_dir(base_dir);
        }
        if let Some(sanitizer) = self.sanitizer {
            env_builder = env_builder.with_sanitizer(sanitizer);
        }
        if let Some(termination) = self.termination {
            env_builder = env_builder.with_termination(termination);
        }

        Ok(LashCore {
            env: env_builder.build(),
            policy,
            modes: Arc::new(self.modes),
            default_mode,
            store_factory: self.store_factory,
            embed_plugin_ids: Arc::new(self.embed_plugin_ids),
            embed_plugin_configs: Arc::new(self.embed_plugin_configs),
        })
    }
}

#[derive(Clone, Debug)]
struct ActiveEmbedPlugin {
    id: &'static str,
    requires_turn_input: bool,
}

pub struct SessionBuilder {
    core: LashCore,
    session_id: String,
    mode: Option<ModeId>,
    parent_session_id: Option<String>,
    store: Option<Arc<dyn RuntimePersistence>>,
    active_plugins: Vec<ActiveEmbedPlugin>,
}

impl SessionBuilder {
    pub fn standard(mut self) -> Self {
        self.mode = Some(ModeId::standard());
        self
    }

    pub fn rlm(mut self) -> Self {
        self.mode = Some(ModeId::rlm());
        self
    }

    pub fn mode(mut self, mode: ModeId) -> Self {
        self.mode = Some(mode);
        self
    }

    pub fn parent(mut self, parent_session_id: impl Into<String>) -> Self {
        self.parent_session_id = Some(parent_session_id.into());
        self
    }

    pub fn store(mut self, store: Arc<dyn RuntimePersistence>) -> Self {
        self.store = Some(store);
        self
    }

    pub fn use_plugin<P: EmbedPlugin>(mut self, config: P::SessionConfig) -> Result<Self> {
        if !self.core.embed_plugin_ids.contains(P::ID) {
            return Err(EmbedError::PluginNotRegistered { plugin_id: P::ID });
        }
        let configs = self
            .core
            .embed_plugin_configs
            .get(P::ID)
            .cloned()
            .and_then(|configs| {
                Arc::downcast::<StdMutex<HashMap<String, P::SessionConfig>>>(configs).ok()
            })
            .ok_or_else(|| EmbedError::PluginConfig {
                plugin_id: P::ID,
                message: "registered config store has an unexpected type".to_string(),
            })?;
        configs
            .lock()
            .map_err(|_| EmbedError::PluginConfig {
                plugin_id: P::ID,
                message: "config lock poisoned".to_string(),
            })?
            .insert(self.session_id.clone(), config.clone());
        self.active_plugins.push(ActiveEmbedPlugin {
            id: P::ID,
            requires_turn_input: P::requires_turn_input(&config),
        });
        Ok(self)
    }

    pub async fn open(self) -> Result<LashSession> {
        let mode = self
            .mode
            .clone()
            .unwrap_or_else(|| self.core.default_mode.clone());
        let preset = self
            .core
            .modes
            .get(&mode)
            .ok_or_else(|| EmbedError::ModeNotInstalled { mode: mode.clone() })?;
        let mut policy = self.core.policy.clone();
        policy.session_id = Some(self.session_id.clone());
        policy.execution_mode = mode.execution_mode();
        policy.standard_context_approach = preset.standard_context_approach.clone();
        let store = self.create_store(&policy)?;
        let state = match store.as_ref() {
            Some(store) => {
                let loaded = lash::load_persisted_session_state(store.as_ref())
                    .await
                    .map_err(|err| {
                        SessionError::Protocol(format!("failed to load store: {err}"))
                    })?;
                let mut state = loaded.unwrap_or_else(|| PersistedSessionState {
                    session_id: self.session_id.clone(),
                    policy: policy.clone(),
                    ..PersistedSessionState::default()
                });
                state.session_id = self.session_id.clone();
                state.policy = policy.clone();
                if let Some(snapshot) = state.dynamic_state_snapshot.as_mut() {
                    snapshot.base_generation = 1;
                    state.dynamic_state_generation = Some(snapshot.base_generation);
                }
                state
            }
            None => PersistedSessionState {
                session_id: self.session_id.clone(),
                policy: policy.clone(),
                ..PersistedSessionState::default()
            },
        };
        let runtime = LashRuntime::from_environment(&self.core.env, policy, state, store).await?;
        Ok(LashSession {
            runtime: Arc::new(Mutex::new(runtime)),
            mode,
            parent_session_id: self.parent_session_id,
            active_plugins: self.active_plugins,
        })
    }

    fn create_store(&self, policy: &SessionPolicy) -> Result<Option<Arc<dyn RuntimePersistence>>> {
        if let Some(store) = self.store.as_ref() {
            return Ok(Some(Arc::clone(store)));
        }
        let Some(factory) = self.core.store_factory.as_ref() else {
            return Ok(None);
        };
        let request = SessionStoreCreateRequest {
            session_id: self.session_id.clone(),
            parent_session_id: self.parent_session_id.clone(),
            policy: policy.clone(),
        };
        factory
            .create_store(&request)
            .map(Some)
            .map_err(|message| EmbedError::StoreFactory {
                session_id: self.session_id.clone(),
                message,
            })
    }
}

#[derive(Clone)]
pub struct LashSession {
    runtime: Arc<Mutex<LashRuntime>>,
    mode: ModeId,
    parent_session_id: Option<String>,
    active_plugins: Vec<ActiveEmbedPlugin>,
}

impl LashSession {
    pub fn mode(&self) -> &ModeId {
        &self.mode
    }

    pub fn parent_session_id(&self) -> Option<&str> {
        self.parent_session_id.as_deref()
    }

    pub async fn run(&self, input: Input) -> Result<CollectedTurnResult> {
        self.turn(input).run().await
    }

    pub async fn rlm_project(&self, bindings: RlmProjectedBindings) -> Result<()> {
        self.runtime
            .lock()
            .await
            .apply_mode_session_extension(lash_mode_rlm::rlm_session_projection_extension(bindings))
            .await
            .map_err(|err| EmbedError::RlmProjectedBinding(err.to_string()))
    }

    pub fn turn(&self, input: Input) -> TurnBuilder<'_> {
        TurnBuilder {
            session: self,
            input,
            cancel: CancellationToken::new(),
            mode_turn_options: None,
            rlm_projected_bindings: None,
            turn_context: TurnContext::default(),
        }
    }

    pub async fn read_view(&self) -> SessionReadView {
        self.runtime.lock().await.read_view()
    }

    pub async fn tool_state(&self) -> Result<ToolState> {
        self.runtime
            .lock()
            .await
            .dynamic_tool_state()
            .map_err(Into::into)
    }

    pub async fn apply_tool_state(&self, state: ToolState) -> Result<u64> {
        self.runtime
            .lock()
            .await
            .apply_dynamic_tool_state(state)
            .await
            .map_err(Into::into)
    }

    pub async fn set_tool_availability(
        &self,
        names: &[String],
        availability: Option<ToolAvailability>,
    ) -> Result<u64> {
        let mut state = self.tool_state().await?;
        for name in names {
            let Some(spec) = state.tools.get_mut(name) else {
                return Err(EmbedError::Session(SessionError::Protocol(format!(
                    "unknown dynamic tool `{name}`"
                ))));
            };
            spec.definition.availability_override = availability.clone();
        }
        self.apply_tool_state(state).await
    }
}

pub struct TurnBuilder<'a> {
    session: &'a LashSession,
    input: Input,
    cancel: CancellationToken,
    mode_turn_options: Option<ModeTurnOptions>,
    rlm_projected_bindings: Option<RlmProjectedBindings>,
    turn_context: TurnContext,
}

impl<'a> TurnBuilder<'a> {
    pub fn cancel(mut self, cancel: CancellationToken) -> Self {
        self.cancel = cancel;
        self
    }

    pub fn mode_turn_options(mut self, options: ModeTurnOptions) -> Self {
        self.mode_turn_options = Some(options);
        self
    }

    pub fn rlm_project(mut self, bindings: RlmProjectedBindings) -> Result<Self> {
        self.rlm_projected_bindings = Some(match self.rlm_projected_bindings.take() {
            Some(existing) => existing
                .merge(bindings)
                .map_err(|err| EmbedError::RlmProjectedBinding(err.to_string()))?,
            None => bindings,
        });
        Ok(self)
    }

    pub fn with_plugin_input<P: EmbedPlugin>(mut self, input: P::TurnInput) -> Self {
        self.turn_context.insert_plugin_input(P::ID, input);
        self
    }

    pub async fn run(self) -> Result<CollectedTurnResult> {
        let collector = TurnCollector::default();
        let result = self.stream(&collector).await?;
        Ok(CollectedTurnResult {
            transcript: collector.snapshot(),
            outcome: result.outcome,
            usage: result.usage,
            errors: result.errors,
        })
    }

    pub async fn stream(self, events: &dyn TurnEventSink) -> Result<TurnResult> {
        let mut input = self.input.into_turn_input();
        input.mode_turn_options = self.mode_turn_options;
        input.turn_context = self.turn_context;
        if let Some(turn_bindings) = self.rlm_projected_bindings {
            input = lash_mode_rlm::RlmTurnInputExt::rlm_project(input, turn_bindings)
                .map_err(|err| EmbedError::RlmProjectedBinding(err.to_string()))?;
        }
        for plugin in &self.session.active_plugins {
            if plugin.requires_turn_input && !input.turn_context.has_plugin_input(plugin.id) {
                return Err(EmbedError::MissingPluginTurnInput {
                    plugin_id: plugin.id,
                });
            }
        }
        let mut runtime = self.session.runtime.lock().await;
        if let Some(extension) = input.mode_extension.as_ref() {
            runtime
                .validate_mode_turn_extension(extension)
                .await
                .map_err(|err| EmbedError::RlmProjectedBinding(err.to_string()))?;
        }
        let turn = runtime
            .stream_turn_events_following_handoffs(input, events, self.cancel)
            .await?;
        let final_turn = turn.into_final_turn().ok_or_else(|| {
            EmbedError::Runtime(lash::RuntimeError {
                code: "empty_followed_turn".to_string(),
                message: "runtime completed without an assembled turn".to_string(),
            })
        })?;
        Ok(TurnResult::from_assembled(final_turn))
    }
}

#[derive(Clone, Debug, Default)]
pub struct Input {
    items: Vec<InputItem>,
    image_blobs: HashMap<String, Vec<u8>>,
}

impl Input {
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            items: vec![InputItem::Text { text: text.into() }],
            image_blobs: HashMap::new(),
        }
    }

    pub fn items(items: Vec<InputItem>) -> Self {
        Self {
            items,
            image_blobs: HashMap::new(),
        }
    }

    pub fn with_image_blob(mut self, id: impl Into<String>, bytes: Vec<u8>) -> Self {
        self.image_blobs.insert(id.into(), bytes);
        self
    }

    fn into_turn_input(self) -> TurnInput {
        TurnInput {
            items: self.items,
            image_blobs: self.image_blobs,
            user_input: None,
            mode: None,
            mode_turn_options: None,
            trace_turn_id: None,
            mode_extension: None,
            turn_context: lash::TurnContext::default(),
        }
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TurnResult {
    pub outcome: TurnOutcome,
    pub usage: TokenUsage,
    pub errors: Vec<TurnIssue>,
}

impl TurnResult {
    fn from_assembled(turn: lash::AssembledTurn) -> Self {
        Self {
            outcome: turn.outcome.clone(),
            usage: turn.token_usage.clone(),
            errors: turn.errors.clone(),
        }
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CollectedTurnResult {
    pub transcript: TurnTranscript,
    pub outcome: TurnOutcome,
    pub usage: TokenUsage,
    pub errors: Vec<TurnIssue>,
}

#[derive(Default)]
struct TurnCollectorState {
    transcript: TurnTranscript,
    events: Vec<TurnEvent>,
    active_code: Vec<(String, String)>,
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct TurnTranscript {
    pub assistant_prose: String,
    pub reasoning: String,
    pub terminal_outputs: Vec<CollectedTerminalOutput>,
    pub visible_outputs: Vec<CollectedVisibleOutput>,
    pub tool_calls: Vec<CollectedToolCall>,
    pub code_blocks: Vec<CollectedCodeBlock>,
    pub errors: Vec<String>,
}

impl TurnTranscript {
    pub fn rendered_output(&self) -> String {
        self.visible_outputs
            .iter()
            .map(|output| output.text.as_str())
            .collect::<String>()
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CollectedToolCall {
    pub call_id: Option<String>,
    pub name: String,
    pub args: serde_json::Value,
    pub result: ToolResultView,
    pub success: bool,
    pub duration_ms: u64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CollectedTerminalOutput {
    pub source: TerminalOutputSource,
    pub value: serde_json::Value,
    pub rendered: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CollectedVisibleOutput {
    pub kind: CollectedVisibleOutputKind,
    pub text: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CollectedVisibleOutputKind {
    AssistantProse,
    TerminalOutput,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CollectedCodeBlock {
    pub language: String,
    pub code: String,
    pub output: String,
    pub error: Option<String>,
    pub success: bool,
    pub duration_ms: u64,
}

#[derive(Clone, Default)]
pub struct TurnCollector {
    state: Arc<StdMutex<TurnCollectorState>>,
}

impl TurnCollector {
    pub fn snapshot(&self) -> TurnTranscript {
        self.state
            .lock()
            .expect("turn collector lock")
            .transcript
            .clone()
    }

    pub fn events(&self) -> Vec<TurnEvent> {
        self.state
            .lock()
            .expect("turn collector lock")
            .events
            .clone()
    }

    pub fn rendered_output(&self) -> String {
        self.state
            .lock()
            .expect("turn collector lock")
            .transcript
            .rendered_output()
    }

    pub fn sink(&self) -> Arc<dyn TurnEventSink> {
        Arc::new(self.clone())
    }
}

pub struct TurnEventFanout {
    sinks: Vec<Arc<dyn TurnEventSink>>,
}

impl TurnEventFanout {
    pub fn new(sinks: impl IntoIterator<Item = Arc<dyn TurnEventSink>>) -> Self {
        Self {
            sinks: sinks.into_iter().collect(),
        }
    }
}

#[async_trait]
impl TurnEventSink for TurnCollector {
    async fn emit(&self, event: TurnEvent) {
        let mut state = self.state.lock().expect("turn collector lock");
        match &event {
            TurnEvent::AssistantProseDelta { text } => {
                state.transcript.assistant_prose.push_str(text);
                state
                    .transcript
                    .visible_outputs
                    .push(CollectedVisibleOutput {
                        kind: CollectedVisibleOutputKind::AssistantProse,
                        text: text.clone(),
                    });
            }
            TurnEvent::ReasoningDelta { text } => {
                state.transcript.reasoning.push_str(text);
            }
            TurnEvent::ToolCallCompleted {
                call_id,
                name,
                args,
                result,
                success,
                duration_ms,
            } => {
                state.transcript.tool_calls.push(CollectedToolCall {
                    call_id: call_id.clone(),
                    name: name.clone(),
                    args: args.clone(),
                    result: result.clone(),
                    success: *success,
                    duration_ms: *duration_ms,
                });
            }
            TurnEvent::CodeBlockStarted { language, code } => {
                state.active_code.push((language.clone(), code.clone()));
            }
            TurnEvent::CodeBlockCompleted {
                language,
                output,
                error,
                success,
                duration_ms,
            } => {
                let code = state
                    .active_code
                    .iter()
                    .rposition(|(active_language, _)| active_language == language)
                    .map(|index| state.active_code.remove(index).1)
                    .unwrap_or_default();
                state.transcript.code_blocks.push(CollectedCodeBlock {
                    language: language.clone(),
                    code,
                    output: output.clone(),
                    error: error.clone(),
                    success: *success,
                    duration_ms: *duration_ms,
                });
            }
            TurnEvent::Error { message } => {
                state.transcript.errors.push(message.clone());
            }
            TurnEvent::TerminalOutput { source, value } => {
                let rendered = lash::render_terminal_output_value(value);
                state
                    .transcript
                    .terminal_outputs
                    .push(CollectedTerminalOutput {
                        source: source.clone(),
                        value: value.clone(),
                        rendered: rendered.clone(),
                    });
                state
                    .transcript
                    .visible_outputs
                    .push(CollectedVisibleOutput {
                        kind: CollectedVisibleOutputKind::TerminalOutput,
                        text: rendered,
                    });
            }
            TurnEvent::ToolCallStarted { .. } | TurnEvent::Usage { .. } => {}
        }
        state.events.push(event);
    }
}

#[async_trait]
impl TurnEventSink for TurnEventFanout {
    async fn emit(&self, event: TurnEvent) {
        for sink in &self.sinks {
            sink.emit(event.clone()).await;
        }
    }
}

pub fn message_text(message: &Message) -> String {
    message
        .parts
        .iter()
        .map(|part| part.content.as_str())
        .collect::<Vec<_>>()
        .join("\n")
}

pub fn message_role(message: &Message) -> &'static str {
    match message.role {
        MessageRole::User => "user",
        MessageRole::Assistant => "assistant",
        MessageRole::System => "system",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;

    use lash::LlmOutputPart;
    use lash::llm::types::{LlmContentBlock, LlmRequest, LlmResponse, LlmRole, LlmStreamEvent};
    use tokio::sync::Mutex as TokioMutex;

    #[derive(Default)]
    struct SnapshotStore {
        read: std::sync::Mutex<Option<lash::PersistedSessionRead>>,
    }

    impl SnapshotStore {
        fn with_state(state: PersistedSessionState) -> Self {
            let turn_state = state.turn_state();
            let config = lash::PersistedSessionConfig {
                provider_id: state.policy.provider.kind().to_string(),
                configured_model: state.policy.model.clone(),
                context_window: state.policy.max_context_tokens.unwrap_or_default() as u64,
                execution_mode: state.policy.execution_mode.clone(),
                standard_context_approach: state.policy.standard_context_approach.clone(),
                model_variant: state.policy.model_variant.clone(),
            };
            Self {
                read: std::sync::Mutex::new(Some(lash::PersistedSessionRead {
                    session_id: state.session_id,
                    head_revision: 7,
                    config,
                    graph: state.session_graph,
                    checkpoint_ref: None,
                    checkpoint: Some(lash::HydratedSessionCheckpoint {
                        turn_state,
                        dynamic_state: state.dynamic_state_snapshot,
                        ..Default::default()
                    }),
                    token_ledger: Vec::new(),
                })),
            }
        }
    }

    #[async_trait]
    impl lash::RuntimePersistence for SnapshotStore {
        async fn load_session(
            &self,
            _scope: lash::SessionReadScope,
        ) -> std::result::Result<Option<lash::PersistedSessionRead>, lash::store::StoreError>
        {
            Ok(self.read.lock().expect("snapshot store lock").clone())
        }

        async fn load_node(
            &self,
            _node_id: &str,
        ) -> std::result::Result<Option<lash::SessionNodeRecord>, lash::store::StoreError> {
            Ok(None)
        }

        async fn commit_runtime_state(
            &self,
            _commit: lash::RuntimeCommit,
        ) -> std::result::Result<lash::RuntimeCommitResult, lash::store::StoreError> {
            Ok(lash::RuntimeCommitResult {
                head_revision: 8,
                checkpoint_ref: lash::BlobRef("checkpoint".to_string()),
                manifest: lash::SessionCheckpoint::default(),
            })
        }

        async fn save_session_meta(
            &self,
            _meta: lash::SessionMeta,
        ) -> std::result::Result<(), lash::store::StoreError> {
            Ok(())
        }

        async fn load_session_meta(
            &self,
        ) -> std::result::Result<Option<lash::SessionMeta>, lash::store::StoreError> {
            Ok(None)
        }

        async fn tombstone_nodes(
            &self,
            _ids: &[String],
        ) -> std::result::Result<(), lash::store::StoreError> {
            Ok(())
        }

        async fn vacuum(&self) -> std::result::Result<lash::VacuumReport, lash::store::StoreError> {
            Ok(lash::VacuumReport::default())
        }

        async fn gc_unreachable(
            &self,
        ) -> std::result::Result<lash::GcReport, lash::store::StoreError> {
            Ok(lash::GcReport::default())
        }
    }

    #[derive(Clone)]
    struct ReusableStoreFactory {
        store: Arc<dyn lash::RuntimePersistence>,
    }

    impl lash::SessionStoreFactory for ReusableStoreFactory {
        fn create_store(
            &self,
            _request: &lash::SessionStoreCreateRequest,
        ) -> std::result::Result<Arc<dyn lash::RuntimePersistence>, String> {
            Ok(Arc::clone(&self.store))
        }
    }

    #[derive(Default)]
    struct RecordingEvents {
        events: TokioMutex<Vec<TurnEvent>>,
    }

    impl RecordingEvents {
        async fn snapshot(&self) -> Vec<TurnEvent> {
            self.events.lock().await.clone()
        }
    }

    #[async_trait]
    impl TurnEventSink for RecordingEvents {
        async fn emit(&self, event: TurnEvent) {
            self.events.lock().await.push(event);
        }
    }

    struct AppTools;

    #[async_trait]
    impl ToolProvider for AppTools {
        fn definitions(&self) -> Vec<lash::ToolDefinition> {
            vec![lash::ToolDefinition::new(
                "app_lookup",
                "Look up app state.",
                serde_json::json!({
                    "type": "object",
                    "properties": {},
                    "additionalProperties": false
                }),
                serde_json::json!({ "type": "object" }),
            )]
        }

        async fn execute(&self, _name: &str, _args: &serde_json::Value) -> lash::ToolResult {
            lash::ToolResult::ok(serde_json::json!({ "ok": true }))
        }
    }

    struct LongTextTools;

    #[async_trait]
    impl ToolProvider for LongTextTools {
        fn definitions(&self) -> Vec<lash::ToolDefinition> {
            vec![lash::ToolDefinition::new(
                "app_lookup",
                "Look up verbose app state.",
                serde_json::json!({
                    "type": "object",
                    "properties": {},
                    "additionalProperties": false
                }),
                serde_json::json!({ "type": "string" }),
            )]
        }

        async fn execute(&self, _name: &str, _args: &serde_json::Value) -> lash::ToolResult {
            lash::ToolResult::ok(serde_json::json!("abcdefghijklmnopqrstuvwxyz0123456789"))
        }
    }

    fn mock_provider() -> ProviderHandle {
        lash::testing::TestProvider::builder()
            .kind("embed-test")
            .default_model("mock-model")
            .requires_streaming(true)
            .complete(|request| async move {
                let user_text = last_user_text(&request);
                let reply = format!("echo: {user_text}");
                if let Some(events) = request.stream_events.as_ref() {
                    events.send(LlmStreamEvent::Delta(reply.clone()));
                }
                Ok(LlmResponse {
                    full_text: reply.clone(),
                    parts: vec![LlmOutputPart::Text {
                        text: reply,
                        response_meta: None,
                    }],
                    usage: lash::llm::types::LlmUsage {
                        input_tokens: user_text.split_whitespace().count() as i64,
                        output_tokens: 2,
                        cached_input_tokens: 0,
                        reasoning_tokens: 0,
                    },
                    ..LlmResponse::default()
                })
            })
            .build()
            .into_handle()
    }

    fn tool_roundtrip_provider() -> ProviderHandle {
        let responses = Arc::new(TokioMutex::new(VecDeque::from([
            LlmResponse {
                parts: vec![LlmOutputPart::ToolCall {
                    call_id: "call-1".to_string(),
                    tool_name: "app_lookup".to_string(),
                    input_json: "{}".to_string(),
                    item_id: None,
                    signature: None,
                }],
                ..LlmResponse::default()
            },
            LlmResponse {
                full_text: "done".to_string(),
                parts: vec![LlmOutputPart::Text {
                    text: "done".to_string(),
                    response_meta: None,
                }],
                ..LlmResponse::default()
            },
        ])));
        lash::testing::TestProvider::builder()
            .kind("embed-test")
            .default_model("mock-model")
            .complete(move |_request| {
                let responses = Arc::clone(&responses);
                async move { Ok(responses.lock().await.pop_front().expect("queued response")) }
            })
            .build()
            .into_handle()
    }

    fn queued_text_provider(texts: Vec<&'static str>) -> ProviderHandle {
        let responses = Arc::new(TokioMutex::new(VecDeque::from(
            texts
                .into_iter()
                .map(|text| LlmResponse {
                    full_text: text.to_string(),
                    parts: vec![LlmOutputPart::Text {
                        text: text.to_string(),
                        response_meta: None,
                    }],
                    ..LlmResponse::default()
                })
                .collect::<Vec<_>>(),
        )));
        lash::testing::TestProvider::builder()
            .kind("embed-test")
            .default_model("mock-model")
            .complete(move |_request| {
                let responses = Arc::clone(&responses);
                async move { Ok(responses.lock().await.pop_front().expect("queued response")) }
            })
            .build()
            .into_handle()
    }

    fn last_user_text(request: &LlmRequest) -> String {
        request
            .messages
            .iter()
            .rev()
            .find(|message| message.role == LlmRole::User)
            .map(|message| {
                message
                    .blocks
                    .iter()
                    .filter_map(|block| match block {
                        LlmContentBlock::Text { text, .. } => Some(text.as_ref()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
            })
            .unwrap_or_default()
    }

    fn standard_core() -> LashCore {
        LashCore::standard()
            .provider(mock_provider())
            .model("mock-model")
            .max_context_tokens(200_000)
            .build()
            .expect("standard core")
    }

    #[tokio::test]
    async fn standard_core_runs_mock_turn() -> Result<()> {
        let core = standard_core();
        let session = core.session("main").open().await?;
        let events = RecordingEvents::default();

        let result = session.turn(Input::text("hello")).stream(&events).await?;

        assert!(matches!(
            result.outcome,
            TurnOutcome::Finished(lash::TurnFinish::AssistantMessage { .. })
        ));
        let events = events.snapshot().await;
        assert!(
            events
                .iter()
                .any(|event| matches!(event, TurnEvent::AssistantProseDelta { .. }))
        );
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, TurnEvent::ToolCallCompleted { .. }))
        );
        Ok(())
    }

    #[tokio::test]
    async fn embedded_sessions_always_expose_dynamic_tool_state() -> Result<()> {
        let core = standard_core();
        let session = core.session("dynamic-default").open().await?;

        let state = session.tool_state().await?;

        assert!(state.base_generation > 0);
        Ok(())
    }

    #[tokio::test]
    async fn registered_static_tools_appear_in_tool_state() -> Result<()> {
        let core = LashCore::standard()
            .provider(mock_provider())
            .model("mock-model")
            .max_context_tokens(200_000)
            .tools(Arc::new(AppTools))
            .build()?;
        let session = core.session("static-tools").open().await?;

        let state = session.tool_state().await?;

        assert!(state.tools.contains_key("app_lookup"));
        Ok(())
    }

    #[tokio::test]
    async fn apply_tool_state_and_availability_update_live_catalog() -> Result<()> {
        let core = LashCore::standard()
            .provider(mock_provider())
            .model("mock-model")
            .max_context_tokens(200_000)
            .tools(Arc::new(AppTools))
            .build()?;
        let session = core.session("tool-state").open().await?;

        let generation = session
            .set_tool_availability(&["app_lookup".to_string()], Some(ToolAvailability::Hidden))
            .await?;
        let hidden = session.tool_state().await?;

        assert_eq!(hidden.base_generation, generation);
        assert_eq!(
            hidden
                .tools
                .get("app_lookup")
                .and_then(|spec| spec.definition.availability_override.clone()),
            Some(ToolAvailability::Hidden)
        );

        let mut callable = hidden;
        callable
            .tools
            .get_mut("app_lookup")
            .expect("app tool")
            .definition
            .availability_override = Some(ToolAvailability::Callable);
        let generation = session.apply_tool_state(callable).await?;
        let callable = session.tool_state().await?;

        assert_eq!(callable.base_generation, generation);
        assert_eq!(
            callable
                .tools
                .get("app_lookup")
                .and_then(|spec| spec.definition.availability_override.clone()),
            Some(ToolAvailability::Callable)
        );
        Ok(())
    }

    #[tokio::test]
    async fn persisted_session_restores_tool_state() -> Result<()> {
        let core = LashCore::standard()
            .provider(mock_provider())
            .model("mock-model")
            .max_context_tokens(200_000)
            .tools(Arc::new(AppTools))
            .build()?;
        let session = core.session("persisted-tools").open().await?;
        session
            .set_tool_availability(&["app_lookup".to_string()], Some(ToolAvailability::Hidden))
            .await?;
        let persisted_tool_state = session.tool_state().await?;
        let state = PersistedSessionState {
            session_id: "persisted-tools".to_string(),
            policy: lash::SessionPolicy {
                provider: mock_provider(),
                model: "mock-model".to_string(),
                max_context_tokens: Some(200_000),
                execution_mode: lash::ExecutionMode::standard(),
                ..Default::default()
            },
            dynamic_state_snapshot: Some(persisted_tool_state),
            ..Default::default()
        };
        let store: Arc<dyn lash::RuntimePersistence> = Arc::new(SnapshotStore::with_state(state));
        let reopened_core = LashCore::standard()
            .provider(mock_provider())
            .model("mock-model")
            .max_context_tokens(200_000)
            .tools(Arc::new(AppTools))
            .store_factory(Arc::new(ReusableStoreFactory { store }))
            .build()?;

        let reopened = reopened_core.session("persisted-tools").open().await?;
        let state = reopened.tool_state().await?;

        assert_eq!(
            state
                .tools
                .get("app_lookup")
                .and_then(|spec| spec.definition.availability_override.clone()),
            Some(ToolAvailability::Hidden)
        );
        Ok(())
    }

    #[tokio::test]
    async fn rlm_core_opens_rlm_session_and_rejects_standard_session() -> Result<()> {
        let core = LashCore::rlm()
            .provider(mock_provider())
            .model("mock-model")
            .max_context_tokens(200_000)
            .build()?;

        let rlm = core.session("rlm").open().await?;
        assert_eq!(rlm.mode(), &ModeId::rlm());

        let err = match core.session("standard").standard().open().await {
            Ok(_) => panic!("standard mode should not be installed"),
            Err(err) => err,
        };
        assert!(matches!(err, EmbedError::ModeNotInstalled { mode } if mode == ModeId::standard()));
        Ok(())
    }

    #[tokio::test]
    async fn rlm_session_and_turn_projection_duplicates_fail_before_run() -> Result<()> {
        let core = LashCore::rlm()
            .provider(mock_provider())
            .model("mock-model")
            .max_context_tokens(200_000)
            .build()?;
        let session = core.session("rlm").open().await?;
        session
            .rlm_project(
                RlmProjectedBindings::new()
                    .bind_json("current_query", serde_json::json!("session"))
                    .expect("session bind"),
            )
            .await?;

        let err = match session
            .turn(Input::text("hello"))
            .rlm_project(
                RlmProjectedBindings::new()
                    .bind_json("current_query", serde_json::json!("turn"))
                    .expect("turn bind"),
            )?
            .run()
            .await
        {
            Ok(_) => panic!("duplicate session and turn projection should fail"),
            Err(err) => err,
        };
        assert!(
            matches!(err, EmbedError::RlmProjectedBinding(message) if message.contains("current_query"))
        );
        Ok(())
    }

    #[tokio::test]
    async fn explicit_dual_mode_install_allows_standard_parent_and_rlm_child() -> Result<()> {
        let core = LashCore::builder()
            .install_mode(ModePreset::standard())
            .install_mode(ModePreset::rlm())
            .default_mode(ModeId::standard())
            .provider(mock_provider())
            .model("mock-model")
            .max_context_tokens(200_000)
            .build()?;

        let parent = core.session("main").standard().open().await?;
        let child = core.session("child").rlm().parent("main").open().await?;

        assert_eq!(parent.mode(), &ModeId::standard());
        assert_eq!(child.mode(), &ModeId::rlm());
        assert_eq!(child.parent_session_id(), Some("main"));
        Ok(())
    }

    #[tokio::test]
    async fn uninstalled_mode_fails_at_open_time() -> Result<()> {
        let core = standard_core();
        let err = match core.session("rlm").rlm().open().await {
            Ok(_) => panic!("rlm mode should not be installed"),
            Err(err) => err,
        };
        assert!(matches!(err, EmbedError::ModeNotInstalled { mode } if mode == ModeId::rlm()));
        Ok(())
    }

    #[tokio::test]
    async fn run_collects_assistant_prose_without_result_duplication() -> Result<()> {
        let core = standard_core();
        let session = core.session("main").open().await?;

        let result = session.run(Input::text("visible")).await?;

        assert_eq!(result.transcript.assistant_prose, "echo: visible");
        assert!(result.transcript.tool_calls.is_empty());
        assert!(result.transcript.code_blocks.is_empty());
        assert!(matches!(
            result.outcome,
            TurnOutcome::Finished(lash::TurnFinish::AssistantMessage { .. })
        ));
        assert_eq!(result.usage.output_tokens, 2);
        Ok(())
    }

    #[tokio::test]
    async fn turn_collector_records_streamed_tool_and_code_events() -> Result<()> {
        let collector = TurnCollector::default();

        collector
            .emit(TurnEvent::CodeBlockStarted {
                language: "lashlang".to_string(),
                code: "x = (call app_lookup {})?".to_string(),
            })
            .await;
        collector
            .emit(TurnEvent::ToolCallCompleted {
                call_id: Some("call-1".to_string()),
                name: "app_lookup".to_string(),
                args: serde_json::json!({}),
                result: ToolResultView {
                    raw: serde_json::json!({ "ok": true }),
                    for_model: serde_json::json!({ "ok": true }),
                    for_state: serde_json::json!({ "ok": true }),
                },
                success: true,
                duration_ms: 3,
            })
            .await;
        collector
            .emit(TurnEvent::CodeBlockCompleted {
                language: "lashlang".to_string(),
                output: String::new(),
                error: None,
                success: true,
                duration_ms: 4,
            })
            .await;

        let transcript = collector.snapshot();
        assert_eq!(collector.events().len(), 3);
        assert_eq!(transcript.tool_calls.len(), 1);
        assert_eq!(transcript.tool_calls[0].name, "app_lookup");
        assert_eq!(
            transcript.tool_calls[0].result.raw,
            serde_json::json!({ "ok": true })
        );
        assert_eq!(transcript.code_blocks.len(), 1);
        assert_eq!(transcript.code_blocks[0].language, "lashlang");
        assert_eq!(transcript.code_blocks[0].code, "x = (call app_lookup {})?");
        assert!(transcript.code_blocks[0].success);
        Ok(())
    }

    #[tokio::test]
    async fn turn_event_fanout_streams_to_collector_and_live_sink() -> Result<()> {
        let collector = TurnCollector::default();
        let live = Arc::new(RecordingEvents::default()) as Arc<dyn TurnEventSink>;
        let fanout = TurnEventFanout::new(vec![collector.sink(), live]);
        let core = LashCore::standard()
            .provider(tool_roundtrip_provider())
            .model("mock-model")
            .tools(Arc::new(AppTools))
            .max_context_tokens(200_000)
            .build()?;
        let session = core.session("fanout-tool-events").open().await?;

        let result = session
            .turn(Input::text("use tool"))
            .stream(&fanout)
            .await?;

        assert!(matches!(
            result.outcome,
            TurnOutcome::Finished(lash::TurnFinish::AssistantMessage { .. })
        ));
        let transcript = collector.snapshot();
        assert_eq!(transcript.assistant_prose, "done");
        assert_eq!(transcript.tool_calls.len(), 1);
        assert_eq!(transcript.tool_calls[0].name, "app_lookup");
        assert_eq!(
            transcript.tool_calls[0].result.raw,
            serde_json::json!({ "ok": true })
        );
        Ok(())
    }

    #[tokio::test]
    async fn stream_returns_terminal_metadata_without_prose() -> Result<()> {
        let core = standard_core();
        let session = core.session("semantic-events").open().await?;
        let events = RecordingEvents::default();

        let result = session.turn(Input::text("stream")).stream(&events).await?;

        assert!(matches!(
            result.outcome,
            TurnOutcome::Finished(lash::TurnFinish::AssistantMessage { .. })
        ));
        let prose = events
            .snapshot()
            .await
            .into_iter()
            .filter_map(|event| match event {
                TurnEvent::AssistantProseDelta { text } => Some(text),
                _ => None,
            })
            .collect::<String>();
        assert_eq!(prose, "echo: stream");
        assert!(
            !events
                .snapshot()
                .await
                .iter()
                .any(|event| matches!(event, TurnEvent::TerminalOutput { .. }))
        );
        Ok(())
    }

    #[tokio::test]
    async fn stream_emits_chronological_tool_events_without_prose_pollution() -> Result<()> {
        let core = LashCore::standard()
            .provider(tool_roundtrip_provider())
            .model("mock-model")
            .tools(Arc::new(AppTools))
            .max_context_tokens(200_000)
            .build()?;
        let session = core.session("tool-events").open().await?;
        let events = RecordingEvents::default();

        let collected = session
            .turn(Input::text("use tool"))
            .stream(&events)
            .await?;

        assert!(matches!(
            collected.outcome,
            TurnOutcome::Finished(lash::TurnFinish::AssistantMessage { .. })
        ));
        let events = events.snapshot().await;
        let started = events
            .iter()
            .position(|event| matches!(event, TurnEvent::ToolCallStarted { .. }))
            .expect("tool start event");
        let completed = events
            .iter()
            .position(|event| matches!(event, TurnEvent::ToolCallCompleted { .. }))
            .expect("tool completed event");
        assert!(started < completed);
        let TurnEvent::ToolCallCompleted { result, .. } = &events[completed] else {
            unreachable!();
        };
        assert_eq!(result.raw, serde_json::json!({ "ok": true }));
        assert_eq!(result.for_model, serde_json::json!({ "ok": true }));
        assert_eq!(result.for_state, serde_json::json!({ "ok": true }));
        let prose = events
            .into_iter()
            .filter_map(|event| match event {
                TurnEvent::AssistantProseDelta { text } => Some(text),
                _ => None,
            })
            .collect::<String>();
        assert_eq!(prose, "done");
        assert!(!prose.contains("ok"));
        Ok(())
    }

    #[tokio::test]
    async fn rlm_tool_calls_stream_from_live_exec_boundary() -> Result<()> {
        let core = LashCore::rlm()
            .provider(queued_text_provider(vec![
                "```lashlang\nvalue = (call app_lookup {})?\nsubmit \"done\"\n```",
            ]))
            .model("mock-model")
            .tools(Arc::new(AppTools))
            .max_context_tokens(200_000)
            .build()?;
        let session = core.session("rlm-live-tool-events").open().await?;
        let events = Arc::new(RecordingEvents::default());
        let collector = TurnCollector::default();
        let fanout = TurnEventFanout::new(vec![
            collector.sink(),
            events.clone() as Arc<dyn TurnEventSink>,
        ]);

        let result = session
            .turn(Input::text("use tool"))
            .stream(&fanout)
            .await?;

        assert!(matches!(
            result.outcome,
            TurnOutcome::Finished(lash::TurnFinish::Value {
                source: lash::TerminalOutputSource::RlmSubmit,
                ..
            })
        ));
        let events = events.snapshot().await;
        let code_started = events
            .iter()
            .position(|event| matches!(event, TurnEvent::CodeBlockStarted { .. }))
            .expect("code started");
        let tool_started = events
            .iter()
            .position(|event| matches!(event, TurnEvent::ToolCallStarted { .. }))
            .expect("tool started");
        let tool_completed = events
            .iter()
            .position(|event| matches!(event, TurnEvent::ToolCallCompleted { .. }))
            .expect("tool completed");
        let code_completed = events
            .iter()
            .position(|event| matches!(event, TurnEvent::CodeBlockCompleted { .. }))
            .expect("code completed");
        let terminal_output = events
            .iter()
            .position(|event| matches!(event, TurnEvent::TerminalOutput { .. }))
            .expect("terminal output");
        assert!(code_started < tool_started);
        assert!(tool_started < tool_completed);
        assert!(tool_completed < code_completed);
        assert!(code_completed < terminal_output);
        assert!(!events[code_completed + 1..].iter().any(|event| matches!(
            event,
            TurnEvent::ToolCallStarted { .. } | TurnEvent::ToolCallCompleted { .. }
        )));

        let TurnEvent::ToolCallCompleted { result, .. } = &events[tool_completed] else {
            unreachable!();
        };
        assert_eq!(result.raw, serde_json::json!({ "ok": true }));
        assert_eq!(result.for_model, serde_json::json!({ "ok": true }));
        assert_eq!(result.for_state, serde_json::json!({ "ok": true }));
        let TurnEvent::CodeBlockCompleted {
            language,
            success,
            error,
            ..
        } = &events[code_completed]
        else {
            unreachable!();
        };
        assert_eq!(language, "lashlang");
        assert!(*success);
        assert!(error.is_none());
        let TurnEvent::TerminalOutput { source, value } = &events[terminal_output] else {
            unreachable!();
        };
        assert!(matches!(source, lash::TerminalOutputSource::RlmSubmit));
        assert_eq!(value, &serde_json::json!("done"));
        assert_eq!(collector.snapshot().terminal_outputs.len(), 1);
        assert_eq!(collector.rendered_output(), "done");
        Ok(())
    }

    #[tokio::test]
    async fn prose_or_submit_rlm_completion_emits_no_terminal_output() -> Result<()> {
        let core = LashCore::rlm()
            .provider(queued_text_provider(vec!["done in prose"]))
            .model("mock-model")
            .max_context_tokens(200_000)
            .build()?;
        let session = core.session("rlm-prose-completion").open().await?;
        let events = Arc::new(RecordingEvents::default());
        let collector = TurnCollector::default();
        let fanout = TurnEventFanout::new(vec![
            collector.sink(),
            events.clone() as Arc<dyn TurnEventSink>,
        ]);

        let result = session
            .turn(Input::text("answer directly"))
            .mode_turn_options(
                ModeTurnOptions::typed(
                    lash::ExecutionMode::new("rlm"),
                    lash_rlm_types::RlmTermination::ProseOrSubmit,
                )
                .expect("rlm termination options serialize"),
            )
            .stream(&fanout)
            .await?;

        assert!(matches!(
            result.outcome,
            TurnOutcome::Finished(lash::TurnFinish::AssistantMessage { .. })
        ));
        let events = events.snapshot().await;
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, TurnEvent::TerminalOutput { .. }))
        );
        assert_eq!(collector.rendered_output(), "done in prose");
        Ok(())
    }

    #[tokio::test]
    async fn submit_required_rlm_completion_emits_terminal_output() -> Result<()> {
        let core = LashCore::rlm()
            .provider(queued_text_provider(vec![
                "```lashlang\nsubmit \"done via submit\"\n```",
            ]))
            .model("mock-model")
            .max_context_tokens(200_000)
            .build()?;
        let session = core
            .session("rlm-submit-required-completion")
            .open()
            .await?;
        let events = Arc::new(RecordingEvents::default());
        let collector = TurnCollector::default();
        let fanout = TurnEventFanout::new(vec![
            collector.sink(),
            events.clone() as Arc<dyn TurnEventSink>,
        ]);

        let result = session
            .turn(Input::text("submit"))
            .mode_turn_options(
                ModeTurnOptions::typed(
                    lash::ExecutionMode::new("rlm"),
                    lash_rlm_types::RlmTermination::SubmitRequired { schema: None },
                )
                .expect("rlm termination options serialize"),
            )
            .stream(&fanout)
            .await?;

        assert!(matches!(
            result.outcome,
            TurnOutcome::Finished(lash::TurnFinish::Value {
                source: lash::TerminalOutputSource::RlmSubmit,
                ..
            })
        ));
        let events = events.snapshot().await;
        let terminal_output = events
            .iter()
            .find(|event| matches!(event, TurnEvent::TerminalOutput { .. }))
            .expect("terminal output");
        let TurnEvent::TerminalOutput { source, value } = terminal_output else {
            unreachable!();
        };
        assert!(matches!(source, lash::TerminalOutputSource::RlmSubmit));
        assert_eq!(value, &serde_json::json!("done via submit"));
        assert_eq!(collector.rendered_output(), "done via submit");
        Ok(())
    }

    #[tokio::test]
    async fn rlm_tool_completed_uses_standard_projection_view() -> Result<()> {
        let projection = Arc::new(lash::BuiltinToolResultProjectionPluginFactory::new(
            lash::ToolResultProjectionPluginConfig {
                mode: lash::ToolResultProjectionMode::Bytes,
                limit: 12,
                max_lines: 4,
            },
        ));
        let standard_core = LashCore::standard()
            .provider(tool_roundtrip_provider())
            .model("mock-model")
            .tools(Arc::new(LongTextTools))
            .plugin(projection.clone())
            .max_context_tokens(200_000)
            .build()?;
        let standard_session = standard_core.session("standard-projection").open().await?;
        let standard_events = RecordingEvents::default();
        let _ = standard_session
            .turn(Input::text("use tool"))
            .stream(&standard_events)
            .await?;
        let standard_view = standard_events
            .snapshot()
            .await
            .into_iter()
            .find_map(|event| match event {
                TurnEvent::ToolCallCompleted { result, .. } => Some(result),
                _ => None,
            })
            .expect("standard tool completion");

        let rlm_core = LashCore::rlm()
            .provider(queued_text_provider(vec![
                "```lashlang\nvalue = (call app_lookup {})?\nsubmit \"done\"\n```",
            ]))
            .model("mock-model")
            .tools(Arc::new(LongTextTools))
            .plugin(projection)
            .max_context_tokens(200_000)
            .build()?;
        let rlm_session = rlm_core.session("rlm-projection").open().await?;
        let rlm_events = RecordingEvents::default();
        let _ = rlm_session
            .turn(Input::text("use tool"))
            .stream(&rlm_events)
            .await?;
        let rlm_view = rlm_events
            .snapshot()
            .await
            .into_iter()
            .find_map(|event| match event {
                TurnEvent::ToolCallCompleted { result, .. } => Some(result),
                _ => None,
            })
            .expect("rlm tool completion");

        assert_eq!(rlm_view.raw, standard_view.raw);
        assert_ne!(rlm_view.raw, rlm_view.for_model);
        assert_ne!(rlm_view.raw, rlm_view.for_state);
        let standard_model = standard_view.for_model.as_str().expect("model projection");
        let rlm_model = rlm_view.for_model.as_str().expect("rlm model projection");
        assert!(standard_model.contains("bytes truncated"));
        assert!(rlm_model.contains("bytes truncated"));
        assert!(standard_model.contains("Full output saved to:"));
        assert!(rlm_model.contains("Full output saved to:"));
        Ok(())
    }

    #[tokio::test]
    async fn rlm_failed_code_emits_failed_code_completion_without_fake_tools() -> Result<()> {
        let core = LashCore::rlm()
            .provider(queued_text_provider(vec![
                "```lashlang\nthis is not valid lashlang\n```",
                "```lashlang\nsubmit \"recovered\"\n```",
            ]))
            .model("mock-model")
            .tools(Arc::new(AppTools))
            .max_context_tokens(200_000)
            .build()?;
        let session = core.session("rlm-failed-code-event").open().await?;
        let events = RecordingEvents::default();

        let _result = session
            .turn(Input::text("bad code"))
            .stream(&events)
            .await?;

        let events = events.snapshot().await;
        let failed = events
            .iter()
            .position(|event| {
                matches!(
                    event,
                    TurnEvent::CodeBlockCompleted {
                        success: false,
                        error: Some(_),
                        ..
                    }
                )
            })
            .expect("failed code completion");
        let next_code = events[failed + 1..]
            .iter()
            .position(|event| matches!(event, TurnEvent::CodeBlockStarted { .. }))
            .map(|offset| failed + 1 + offset)
            .unwrap_or(events.len());
        assert!(
            !events[failed + 1..next_code]
                .iter()
                .any(|event| matches!(event, TurnEvent::ToolCallCompleted { .. }))
        );
        Ok(())
    }

    #[tokio::test]
    async fn store_factory_reopens_persisted_session_state() -> Result<()> {
        let mut state = PersistedSessionState {
            session_id: "persisted".to_string(),
            policy: lash::SessionPolicy {
                provider: mock_provider(),
                model: "mock-model".to_string(),
                max_context_tokens: Some(200_000),
                execution_mode: lash::ExecutionMode::standard(),
                ..Default::default()
            },
            ..Default::default()
        };
        state.append_active_conversation_messages(&[text_message(
            lash::MessageRole::User,
            "already stored",
        )]);
        let store: Arc<dyn lash::RuntimePersistence> = Arc::new(SnapshotStore::with_state(state));
        let core = LashCore::standard()
            .provider(mock_provider())
            .model("mock-model")
            .max_context_tokens(200_000)
            .store_factory(Arc::new(ReusableStoreFactory { store }))
            .build()?;

        let reopened = core.session("persisted").open().await?;
        let messages = reopened.read_view().await.messages().to_vec();
        assert_eq!(messages.len(), 1);
        assert_eq!(message_text(&messages[0]), "already stored");
        Ok(())
    }

    #[tokio::test]
    async fn explicit_session_store_takes_precedence_over_core_store_factory() -> Result<()> {
        let mut explicit_state = PersistedSessionState {
            session_id: "store-precedence".to_string(),
            policy: lash::SessionPolicy {
                provider: mock_provider(),
                model: "mock-model".to_string(),
                max_context_tokens: Some(200_000),
                execution_mode: lash::ExecutionMode::standard(),
                ..Default::default()
            },
            ..Default::default()
        };
        explicit_state.append_active_conversation_messages(&[text_message(
            lash::MessageRole::User,
            "explicit store",
        )]);
        let mut factory_state = explicit_state.clone();
        factory_state.append_active_conversation_messages(&[text_message(
            lash::MessageRole::Assistant,
            "factory store",
        )]);
        let explicit_store: Arc<dyn lash::RuntimePersistence> =
            Arc::new(SnapshotStore::with_state(explicit_state));
        let factory_store: Arc<dyn lash::RuntimePersistence> =
            Arc::new(SnapshotStore::with_state(factory_state));
        let core = LashCore::standard()
            .provider(mock_provider())
            .model("mock-model")
            .max_context_tokens(200_000)
            .store_factory(Arc::new(ReusableStoreFactory {
                store: factory_store,
            }))
            .build()?;

        let reopened = core
            .session("store-precedence")
            .store(explicit_store)
            .open()
            .await?;
        let messages = reopened.read_view().await.messages().to_vec();

        assert_eq!(messages.len(), 1);
        assert_eq!(message_text(&messages[0]), "explicit store");
        Ok(())
    }

    fn text_message(role: lash::MessageRole, text: &str) -> lash::Message {
        let id = "stored-message".to_string();
        lash::Message {
            id: id.clone(),
            role,
            parts: lash::shared_parts(vec![lash::Part {
                id: format!("{id}.p0"),
                kind: lash::PartKind::Text,
                content: text.to_string(),
                attachment: None,
                tool_call_id: None,
                tool_name: None,
                tool_item_id: None,
                tool_signature: None,
                prune_state: lash::PruneState::Intact,
                reasoning_meta: None,
                response_meta: None,
            }]),
            user_input: None,
            origin: None,
        }
    }
}
