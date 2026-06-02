use std::sync::{Arc, OnceLock};

use tokio::sync::mpsc::UnboundedSender;

use crate::PluginMessage;
use crate::tool_dispatch::ToolDispatchContext;
use crate::{PromptContribution, RuntimeServices, SandboxMessage, SessionEvent, ToolProvider};

mod execution_context;
pub(crate) mod process_handles;
mod tool_execution;
pub(crate) mod triggers;

pub use execution_context::RuntimeExecutionContext;
pub(crate) use execution_context::lashlang_surface_from_tool_surface;
pub use tool_execution::{ToolInvocation, ToolInvocationReply};

#[derive(Clone, Debug, PartialEq, Eq)]
struct ToolSurfaceCacheKey {
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
    preamble: Arc<crate::TurnDriverPreamble>,
    derived: ToolSurfaceDerived,
}

#[derive(Clone)]
pub(crate) struct ToolSurfaceHandle(Arc<ToolSurfaceArtifact>);

impl ToolSurfaceHandle {
    fn surface(&self) -> Arc<crate::ToolSurface> {
        Arc::clone(&self.0.surface)
    }

    fn preamble(&self) -> Arc<crate::TurnDriverPreamble> {
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

#[derive(Clone, Debug)]
pub struct InjectedTurnInput {
    pub id: Option<String>,
    pub message: PluginMessage,
}

#[derive(Debug, thiserror::Error)]
pub enum SessionError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("code execution is not available in this session")]
    CodeExecutionUnavailable,
    #[error("code execution runtime exited unexpectedly")]
    CodeExecutionRuntimeStopped,
    #[error(
        "provider mismatch for session `{session_id}`: persisted provider `{expected}` does not match live provider `{actual}`"
    )]
    ProviderMismatch {
        expected: String,
        actual: String,
        session_id: String,
    },
    #[error("provider is not configured for session `{session_id}`")]
    ProviderUnconfigured { session_id: String },
    #[error("provider `{provider_id}` is not registered for session `{session_id}`")]
    ProviderUnavailable {
        provider_id: String,
        session_id: String,
    },
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
    services: RuntimeServices,
    include_base_tools: bool,
    context_surface_revision: u64,
    context_tools: Vec<Arc<dyn ToolProvider>>,
    tool_registry: Arc<crate::ToolRegistry>,
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
    pub async fn new(services: RuntimeServices, session_id: &str) -> Result<Self, SessionError> {
        let tool_registry = services.plugins.tool_registry();
        let mut session = Self {
            session_id: session_id.to_string(),
            services,
            include_base_tools: true,
            context_surface_revision: 0,
            context_tools: Vec::new(),
            tool_registry,
            context_prompt_contributions: Vec::new(),
            message_tx: None,
            tool_surface_cache: std::sync::Mutex::new(Vec::new()),
            prompt_cache: Arc::new(lash_sansio::PromptCache::new()),
        };

        let protocol_session = Arc::clone(session.plugins().protocol_session());
        protocol_session
            .initialize_session(crate::plugin::ProtocolSessionContext::new(
                &mut session,
                session_id,
            ))
            .await?;

        Ok(session)
    }

    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    pub(crate) fn protocol_extra_prompt_contributions(&self) -> Vec<PromptContribution> {
        // Protocol-specific prompt contributions are owned by the protocol
        // plugins via their
        // `reg.prompt().contribute(...)` hooks. Nothing to add here.
        Vec::new()
    }

    pub fn tools(&self) -> Arc<dyn ToolProvider> {
        Arc::clone(&self.tool_registry) as Arc<dyn ToolProvider>
    }

    pub(crate) fn tool_registry(&self) -> Arc<crate::ToolRegistry> {
        Arc::clone(&self.tool_registry)
    }

    pub fn plugins(&self) -> &Arc<crate::PluginSession> {
        &self.services.plugins
    }

    pub fn set_context_surface(
        &mut self,
        tool_providers: Vec<Arc<dyn ToolProvider>>,
        prompt_contributions: Vec<PromptContribution>,
        include_base_tools: bool,
    ) -> Result<(), crate::PluginError> {
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
            return Ok(());
        }
        let registry = self
            .services
            .plugins
            .tool_registry()
            .compose_session_surface(include_base_tools, tool_providers.clone())
            .map(Arc::new)
            .map_err(|err| {
                crate::PluginError::Session(format!("failed to build session tool registry: {err}"))
            })?;
        self.include_base_tools = include_base_tools;
        self.context_surface_revision = self.context_surface_revision.wrapping_add(1);
        self.context_tools = tool_providers;
        self.tool_registry = registry;
        self.context_prompt_contributions = prompt_contributions;
        self.tool_surface_cache
            .lock()
            .expect("tool surface cache lock")
            .clear();
        Ok(())
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

    fn tool_surface_cache_key(&self) -> ToolSurfaceCacheKey {
        ToolSurfaceCacheKey {
            include_base_tools: self.include_base_tools,
            context_surface_revision: self.context_surface_revision,
            tool_generation: self.tool_registry.generation(),
            plugin_revision: self.plugins().snapshot_revision_fingerprint(),
        }
    }

    fn build_tool_surface_entry(
        &self,
        session_id: &str,
    ) -> Result<ToolSurfaceHandle, crate::PluginError> {
        let provider = self.tools();
        let tools = provider.tool_manifests();
        let contract_provider = Arc::clone(&provider);
        let resolve_contract: lash_sansio::ToolContractResolver =
            Arc::new(move |name: &str| contract_provider.resolve_contract(name));
        let surface = Arc::new(self.plugins().resolve_tool_surface(
            crate::plugin::ToolSurfaceContext {
                session_id: session_id.to_string(),
                tools,
                resolve_contract: Some(Arc::clone(&resolve_contract)),
                tool_access: self.plugins().tool_access().clone(),
                subagent: self.plugins().subagent_context().cloned(),
                lashlang_abilities: self.plugins().lashlang_abilities(),
            },
        )?);
        let input = crate::ProtocolBuildInput {
            tool_surface: Arc::clone(&surface),
            lashlang_surface: execution_context::lashlang_surface_from_tool_surface(
                &surface,
                self.plugins().lashlang_abilities(),
                self.plugins().lashlang_resources(),
            ),
            extra_prompt_contributions: self.protocol_extra_prompt_contributions(),
        };
        let driver = self.plugins().protocol_driver();
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
    ) -> Result<ToolSurfaceHandle, crate::PluginError> {
        let key = self.tool_surface_cache_key();
        let mut cache = self
            .tool_surface_cache
            .lock()
            .expect("tool surface cache lock");
        if let Some((_, entry)) = cache.iter().find(|(entry_key, _)| *entry_key == key) {
            return Ok(entry.clone());
        }
        let entry = self.build_tool_surface_entry(session_id)?;
        cache.push((key, entry.clone()));
        Ok(entry)
    }

    pub fn tool_surface(
        &self,
        session_id: &str,
    ) -> Result<Arc<crate::ToolSurface>, crate::PluginError> {
        Ok(self.tool_surface_cache_entry(session_id)?.surface())
    }

    pub(crate) fn turn_driver_preamble(
        &self,
        session_id: &str,
    ) -> Result<Arc<crate::TurnDriverPreamble>, crate::PluginError> {
        Ok(self.tool_surface_cache_entry(session_id)?.preamble())
    }

    pub(crate) fn shared_tool_catalog(
        &self,
        session_id: &str,
    ) -> Result<Arc<Vec<serde_json::Value>>, crate::PluginError> {
        Ok(self.tool_surface_cache_entry(session_id)?.catalog())
    }

    pub fn tool_catalog(
        &self,
        session_id: &str,
    ) -> Result<Vec<serde_json::Value>, crate::PluginError> {
        Ok(self.shared_tool_catalog(session_id)?.as_ref().clone())
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "code execution bridge carries explicit per-turn runtime dependencies"
    )]
    pub(crate) fn code_execution_context<'run>(
        &self,
        session_id: &str,
        agent_frame_id: &str,
        sessions: Arc<dyn crate::plugin::SessionStateService>,
        session_lifecycle: Arc<dyn crate::plugin::SessionLifecycleService>,
        session_graph: Arc<dyn crate::plugin::SessionGraphService>,
        processes: Arc<dyn crate::ProcessService>,
        process_cancel_ability: Arc<dyn crate::ProcessCancelAbility>,
        effect_controller: crate::runtime::RuntimeEffectControllerHandle<'run>,
        direct_completions: crate::DirectCompletionClient<'run>,
        event_tx: tokio::sync::mpsc::Sender<SessionEvent>,
        chronological_projection: Arc<crate::ChronologicalProjection>,
        protocol_extension: Option<crate::ProtocolTurnExtensionHandle>,
        turn_context: crate::TurnContext,
        checkpoint_messages: crate::tool_dispatch::CheckpointMessageBuffer,
    ) -> Result<RuntimeExecutionContext<'run>, crate::PluginError> {
        let dispatch = Arc::new(ToolDispatchContext {
            plugins: Arc::clone(self.plugins()),
            tools: self.tools(),
            surface: self.tool_surface(session_id)?,
            sessions,
            session_lifecycle,
            session_graph,
            processes,
            process_cancel_ability,
            effect_controller,
            direct_completions: direct_completions.clone(),
            parent_invocation: None,
            session_id: session_id.to_string(),
            agent_frame_id: agent_frame_id.to_string(),
            event_tx,
            checkpoint_messages,
            attachment_store: Arc::clone(&self.services.attachment_store),
            turn_context: turn_context.clone(),
        });
        Ok(RuntimeExecutionContext::new(
            session_id.to_string(),
            dispatch,
            self.plugins().lashlang_abilities(),
            Arc::clone(&self.services.lashlang_artifact_store),
            Arc::clone(&self.services.attachment_store),
            chronological_projection,
            protocol_extension,
            turn_context,
        ))
    }

    /// Set the message sender for streaming messages during execution.
    pub fn set_message_sender(&mut self, tx: UnboundedSender<SandboxMessage>) {
        self.message_tx = Some(tx);
    }

    /// Clear the message sender (drops the sender, causing receivers to terminate).
    pub fn clear_message_sender(&mut self) {
        self.message_tx = None;
    }

    pub fn invalidate_runtime_caches(&self) {
        self.tool_surface_cache
            .lock()
            .expect("tool surface cache lock")
            .clear();
        self.prompt_cache.clear();
    }

    pub async fn refresh_tool_surface(&mut self) -> Result<(), SessionError> {
        self.tool_registry = self
            .services
            .plugins
            .tool_registry()
            .compose_session_surface(self.include_base_tools, self.context_tools.clone())
            .map(Arc::new)
            .map_err(|err| SessionError::Protocol(format!("tool reconfigure failed: {err}")))?;
        self.tool_surface_cache
            .lock()
            .expect("tool surface cache lock")
            .clear();
        Ok(())
    }
}
