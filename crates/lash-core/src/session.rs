use std::collections::VecDeque;
use std::sync::{Arc, OnceLock};

use tokio::sync::mpsc::UnboundedSender;

use crate::PluginMessage;
use crate::tool_dispatch::ToolDispatchContext;
use crate::{PromptContribution, RuntimeServices, SandboxMessage, SessionEvent, ToolProvider};

pub(crate) mod async_handles;
mod execution_context;
mod monitor_handles;
mod tool_execution;

pub use execution_context::ModeExecutionContext;
pub use tool_execution::{ModeToolBatchItem, ModeToolReply};

#[derive(Clone, Debug, PartialEq, Eq)]
struct ToolSurfaceCacheKey {
    mode: crate::ExecutionMode,
    include_base_tools: bool,
    context_surface_revision: u64,
    tool_generation: u64,
    plugin_revision: u64,
}

#[derive(Debug, Default)]
struct ToolSurfaceDerived {
    catalog: OnceLock<Arc<Vec<serde_json::Value>>>,
}

struct ToolSurfaceArtifact {
    surface: Arc<crate::ToolSurface>,
    preamble: Arc<crate::ModePreamble>,
    derived: ToolSurfaceDerived,
}

#[derive(Clone)]
pub(crate) struct ToolSurfaceHandle(Arc<ToolSurfaceArtifact>);

impl ToolSurfaceHandle {
    fn surface(&self) -> Arc<crate::ToolSurface> {
        Arc::clone(&self.0.surface)
    }

    fn preamble(&self) -> Arc<crate::ModePreamble> {
        Arc::clone(&self.0.preamble)
    }

    fn catalog(&self) -> Arc<Vec<serde_json::Value>> {
        Arc::clone(self.0.derived.catalog.get_or_init(|| {
            Arc::new(crate::tool_registry::project_tool_catalog(
                self.0.surface.searchable_tools_iter().cloned(),
            ))
        }))
    }
}

#[derive(Clone, Default)]
pub struct TurnInjectionBridge {
    queue: std::sync::Arc<std::sync::Mutex<VecDeque<PluginMessage>>>,
}

#[derive(Clone, Debug)]
pub struct InjectedTurnInput {
    pub id: Option<String>,
    pub message: PluginMessage,
}

#[derive(Clone, Default)]
pub struct TurnInputInjectionBridge {
    queue: std::sync::Arc<std::sync::Mutex<VecDeque<InjectedTurnInput>>>,
}

impl TurnInjectionBridge {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn enqueue(&self, messages: Vec<PluginMessage>) -> Result<(), String> {
        let mut queue = self
            .queue
            .lock()
            .map_err(|_| "turn injection bridge poisoned".to_string())?;
        queue.extend(messages);
        Ok(())
    }

    pub fn drain(&self) -> Result<Vec<PluginMessage>, String> {
        let mut queue = self
            .queue
            .lock()
            .map_err(|_| "turn injection bridge poisoned".to_string())?;
        Ok(queue.drain(..).collect())
    }
}

impl TurnInputInjectionBridge {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn enqueue(&self, messages: Vec<InjectedTurnInput>) -> Result<(), String> {
        let mut queue = self
            .queue
            .lock()
            .map_err(|_| "turn injection bridge poisoned".to_string())?;
        queue.extend(messages);
        Ok(())
    }

    pub fn drain(&self) -> Result<Vec<InjectedTurnInput>, String> {
        let mut queue = self
            .queue
            .lock()
            .map_err(|_| "turn input injection bridge poisoned".to_string())?;
        Ok(queue.drain(..).collect())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SessionError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("rlm execution mode is not available in this build or session")]
    RlmUnavailable,
    #[error("rlm runtime exited unexpectedly")]
    RuntimeExited,
    #[error("protocol error: {0}")]
    Protocol(String),
}

#[derive(Clone, Debug)]
pub struct ExecRequest {
    pub code: String,
    pub accept_finish: bool,
}

pub struct Session {
    session_id: String,
    execution_mode: crate::ExecutionMode,
    services: RuntimeServices,
    include_base_tools: bool,
    context_surface_revision: u64,
    context_tools: Vec<Arc<dyn ToolProvider>>,
    context_prompt_contributions: Vec<PromptContribution>,
    message_tx: Option<UnboundedSender<SandboxMessage>>,
    tool_surface_cache: std::sync::Mutex<Vec<(ToolSurfaceCacheKey, ToolSurfaceHandle)>>,
    /// Memoizes the rendered system prompt across turns. Most consecutive
    /// turns reuse the same template + context surface, so the cache hits
    /// and we skip the section/Vec-join work in
    /// `lash_sansio::PromptTemplate::render`.
    prompt_cache: Arc<lash_sansio::PromptCache>,
}

impl Session {
    pub async fn new(
        services: RuntimeServices,
        session_id: &str,
        execution_mode: crate::ExecutionMode,
    ) -> Result<Self, SessionError> {
        let mut session = Self {
            session_id: session_id.to_string(),
            execution_mode,
            services,
            include_base_tools: true,
            context_surface_revision: 0,
            context_tools: Vec::new(),
            context_prompt_contributions: Vec::new(),
            message_tx: None,
            tool_surface_cache: std::sync::Mutex::new(Vec::new()),
            prompt_cache: Arc::new(lash_sansio::PromptCache::new()),
        };

        let mode_session = Arc::clone(session.plugins().mode_session());
        mode_session
            .initialize_session(crate::plugin::ModeSessionContext::new(
                &mut session,
                session_id,
            ))
            .await?;

        Ok(session)
    }

    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    pub(crate) fn mode_extra_prompt_contributions(
        &self,
        _mode: &crate::ExecutionMode,
    ) -> Vec<PromptContribution> {
        // Mode-specific prompt contributions are owned by the mode
        // plugins (`lash-mode-standard`, `lash-mode-rlm`) via their
        // `reg.prompt().contribute(...)` hooks. Nothing to add here.
        Vec::new()
    }

    pub fn tools(&self) -> Arc<dyn ToolProvider> {
        if self.include_base_tools && self.context_tools.is_empty() {
            return self.services.plugins.tools();
        }

        let mut providers = Vec::new();
        if self.include_base_tools {
            providers.push(self.services.plugins.tools());
        }
        providers.extend(self.context_tools.iter().cloned());
        Arc::new(crate::tool_provider::CompositeToolProvider::from_providers(
            providers,
        ))
    }

    pub fn plugins(&self) -> &Arc<crate::PluginSession> {
        &self.services.plugins
    }

    pub fn set_context_surface(
        &mut self,
        tool_providers: Vec<Arc<dyn ToolProvider>>,
        prompt_contributions: Vec<PromptContribution>,
        include_base_tools: bool,
    ) {
        let tool_providers_unchanged = self.context_tools.len() == tool_providers.len()
            && self
                .context_tools
                .iter()
                .zip(&tool_providers)
                .all(|(current, next)| Arc::ptr_eq(current, next));
        if self.include_base_tools == include_base_tools
            && self.context_prompt_contributions == prompt_contributions
            && tool_providers_unchanged
        {
            return;
        }
        self.include_base_tools = include_base_tools;
        self.context_surface_revision = self.context_surface_revision.wrapping_add(1);
        self.context_tools = tool_providers;
        self.context_prompt_contributions = prompt_contributions;
        self.tool_surface_cache
            .lock()
            .expect("tool surface cache lock")
            .clear();
    }

    pub fn prompt_cache(&self) -> Arc<lash_sansio::PromptCache> {
        Arc::clone(&self.prompt_cache)
    }

    pub fn context_prompt_contributions(&self) -> &[PromptContribution] {
        &self.context_prompt_contributions
    }

    pub fn history_store(&self) -> Option<Arc<dyn crate::store::RuntimePersistence>> {
        self.services.store.clone()
    }

    fn tool_surface_cache_key(&self, mode: &crate::ExecutionMode) -> ToolSurfaceCacheKey {
        ToolSurfaceCacheKey {
            mode: mode.clone(),
            include_base_tools: self.include_base_tools,
            context_surface_revision: self.context_surface_revision,
            tool_generation: self.plugins().tool_registry().generation(),
            plugin_revision: self.plugins().snapshot_revision_fingerprint(),
        }
    }

    fn build_tool_surface_entry(
        &self,
        session_id: &str,
        mode: crate::ExecutionMode,
    ) -> Result<ToolSurfaceHandle, crate::PluginError> {
        let provider = self.tools();
        let mut tools = provider.tool_manifests();
        let contract_provider = Arc::clone(&provider);
        let plugins = self.plugins();
        let native_contract_providers = plugins.mode_native_tools().to_vec();
        let resolve_contract: lash_sansio::ToolContractResolver = Arc::new(move |name: &str| {
            contract_provider.resolve_contract(name).or_else(|| {
                native_contract_providers
                    .iter()
                    .find_map(|provider| provider.resolve_contract(name))
            })
        });
        if self.include_base_tools && mode == self.plugins().execution_mode() {
            let native_tools = self.plugins().mode_native_tool_manifests();
            tools.extend(native_tools);
        }
        let surface = Arc::new(self.plugins().resolve_tool_surface(
            crate::plugin::ToolSurfaceContext {
                session_id: session_id.to_string(),
                mode: mode.clone(),
                tools,
                resolve_contract: Some(Arc::clone(&resolve_contract)),
                tool_access: self.plugins().tool_access().clone(),
                subagent: self.plugins().subagent_context().cloned(),
            },
        )?);
        let input = crate::ModeBuildInput {
            mode: mode.clone(),
            tool_surface: Arc::clone(&surface),
            extra_prompt_contributions: self.mode_extra_prompt_contributions(&mode),
        };
        let driver = self.plugins().mode_protocol_driver().unwrap_or_else(|| {
            panic!(
                "no protocol driver registered for execution mode `{}` — \
                 did you forget to register the mode plugin (e.g. \
                 `lash_mode_standard::BuiltinStandardModePluginFactory` or \
                 `lash_mode_rlm::BuiltinRlmModePluginFactory`)?",
                mode.plugin_id()
            )
        });
        assert_eq!(
            driver.mode_id(),
            mode.plugin_id(),
            "protocol driver `{}` does not match session mode `{}`",
            driver.mode_id(),
            mode.plugin_id(),
        );
        let preamble = driver.build_preamble(input);
        Ok(ToolSurfaceHandle(Arc::new(ToolSurfaceArtifact {
            surface,
            preamble: Arc::new(preamble),
            derived: ToolSurfaceDerived::default(),
        })))
    }

    fn tool_surface_cache_entry(
        &self,
        session_id: &str,
        mode: crate::ExecutionMode,
    ) -> Result<ToolSurfaceHandle, crate::PluginError> {
        let key = self.tool_surface_cache_key(&mode);
        let mut cache = self
            .tool_surface_cache
            .lock()
            .expect("tool surface cache lock");
        if let Some((_, entry)) = cache.iter().find(|(entry_key, _)| *entry_key == key) {
            return Ok(entry.clone());
        }
        let entry = self.build_tool_surface_entry(session_id, mode)?;
        cache.push((key, entry.clone()));
        Ok(entry)
    }

    pub fn tool_surface(
        &self,
        session_id: &str,
        mode: crate::ExecutionMode,
    ) -> Result<Arc<crate::ToolSurface>, crate::PluginError> {
        Ok(self.tool_surface_cache_entry(session_id, mode)?.surface())
    }

    pub(crate) fn mode_preamble(
        &self,
        session_id: &str,
        mode: crate::ExecutionMode,
    ) -> Result<Arc<crate::ModePreamble>, crate::PluginError> {
        Ok(self.tool_surface_cache_entry(session_id, mode)?.preamble())
    }

    pub(crate) fn shared_tool_catalog(
        &self,
        session_id: &str,
        mode: crate::ExecutionMode,
    ) -> Result<Arc<Vec<serde_json::Value>>, crate::PluginError> {
        Ok(self.tool_surface_cache_entry(session_id, mode)?.catalog())
    }

    pub fn tool_catalog(
        &self,
        session_id: &str,
        mode: crate::ExecutionMode,
    ) -> Result<Vec<serde_json::Value>, crate::PluginError> {
        Ok(self.shared_tool_catalog(session_id, mode)?.as_ref().clone())
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "mode execution bridge carries explicit per-turn runtime dependencies"
    )]
    pub(crate) fn mode_execution_context<'run>(
        &self,
        session_id: &str,
        host: Arc<dyn crate::plugin::RuntimeSessionHost>,
        effect_controller: crate::runtime::RuntimeEffectControllerHandle<'run>,
        detached_effect_controller: Arc<dyn crate::RuntimeEffectController>,
        direct_completions: crate::DirectCompletionClient<'run>,
        event_tx: tokio::sync::mpsc::Sender<SessionEvent>,
        chronological_projection: Arc<crate::ChronologicalProjection>,
        mode_extension: Option<crate::ModeTurnExtensionHandle>,
        turn_context: crate::TurnContext,
    ) -> Result<ModeExecutionContext<'run>, crate::PluginError> {
        let dispatch = Arc::new(ToolDispatchContext {
            plugins: Arc::clone(self.plugins()),
            tools: self.tools(),
            surface: self.tool_surface(session_id, self.execution_mode.clone())?,
            host,
            effect_controller,
            direct_completions: direct_completions.clone(),
            session_id: session_id.to_string(),
            event_tx,
            turn_injection_bridge: self.turn_injection_bridge().clone(),
            attachment_store: Arc::clone(&self.services.attachment_store),
            turn_context: turn_context.clone(),
        });
        Ok(ModeExecutionContext::new(
            session_id.to_string(),
            self.execution_mode.clone(),
            dispatch,
            detached_effect_controller,
            Arc::clone(&self.services.attachment_store),
            chronological_projection,
            mode_extension,
            turn_context,
        ))
    }

    pub fn turn_injection_bridge(&self) -> &TurnInjectionBridge {
        &self.services.turn_injection_bridge
    }

    pub fn turn_input_injection_bridge(&self) -> &TurnInputInjectionBridge {
        &self.services.turn_input_injection_bridge
    }

    /// Set the message sender for streaming messages during execution.
    pub fn set_message_sender(&mut self, tx: UnboundedSender<SandboxMessage>) {
        self.message_tx = Some(tx);
    }

    /// Clear the message sender (drops the sender, causing receivers to terminate).
    pub fn clear_message_sender(&mut self) {
        self.message_tx = None;
    }

    pub async fn reset(&mut self) -> Result<(), SessionError> {
        self.tool_surface_cache
            .lock()
            .expect("tool surface cache lock")
            .clear();
        Ok(())
    }

    pub async fn refresh_tool_surface(&mut self) -> Result<(), SessionError> {
        self.tool_surface_cache
            .lock()
            .expect("tool surface cache lock")
            .clear();
        Ok(())
    }
}
