use std::sync::{Arc, OnceLock, RwLock};

use lash_core::plugin::{
    PluginError, PluginFactory, PluginRegistrar, PluginSessionContext,
    ProcessEngineContributionContext, SessionAuthorityContext, SessionPlugin,
};
use lash_core::{PluginHost, ProcessEngine, TraceContext, TraceSink};
use lash_lashlang_runtime::{
    LashlangArtifactStore, LashlangHostEnvironment, LashlangProcessEngine, LashlangSurface,
    SharedDeferredToolResolver,
};

use super::registration::register_rlm_protocol_plugin;
use super::{RLM_PROTOCOL_PLUGIN_ID, RlmProtocolPluginConfig};
use crate::driver::SharedPromptUsage;
use crate::executor::RlmLashlangExecutionTraceConfig;
use crate::projection::{ProjectionRegistry, ProjectionResolver};

/// Apply the RLM protocol config transformation: enable label annotations, and
/// (when process lifecycle is available) the process/sleep/signal abilities.
///
/// This is protocol logic; it lives here rather than in the facade because both
/// the plugin surface and the contributed Lashlang process engine derive from
/// it.
pub fn rlm_protocol_config(
    config: RlmProtocolPluginConfig,
    process_lifecycle: bool,
) -> RlmProtocolPluginConfig {
    let language_features = config.lashlang_language_features.with_label_annotations();
    let mut config = config.with_lashlang_language_features(language_features);
    if process_lifecycle {
        config.lashlang_abilities = config
            .lashlang_abilities
            .with_sleep()
            .with_processes()
            .with_process_signals();
    }
    config
}

/// Build the Lashlang surface for the contributed process engine from an
/// (already [`rlm_protocol_config`]-transformed) config.
pub fn rlm_lashlang_surface(
    config: &RlmProtocolPluginConfig,
    process_lifecycle: bool,
) -> LashlangSurface {
    let surface = LashlangSurface::new(
        config.lashlang_abilities,
        config.lashlang_language_features,
        lashlang::LashlangHostCatalog::new(),
    );
    if process_lifecycle {
        surface.for_process_registry(true)
    } else {
        surface
    }
}

pub struct RlmProtocolPluginFactory {
    config: RlmProtocolPluginConfig,
    projection_resolver: Arc<dyn ProjectionResolver>,
    deferred_tool_resolver: Option<SharedDeferredToolResolver>,
    artifact_store: Arc<dyn LashlangArtifactStore>,
    lashlang_execution_trace_config: RlmLashlangExecutionTraceConfig,
    /// Whether this deployment has process lifecycle available, learned when
    /// core installs process-engine contributions (before any session is built)
    /// and read back when building the per-session plugin surface so the prompt
    /// advertises the same abilities the engine offers.
    process_lifecycle: OnceLock<bool>,
}

impl RlmProtocolPluginFactory {
    /// Construct the factory. The Lashlang artifact store is a required argument:
    /// there is no valid RLM deployment without one, so the previously build-time
    /// "missing artifact store" error is now unrepresentable.
    pub fn new(
        config: RlmProtocolPluginConfig,
        artifact_store: Arc<dyn LashlangArtifactStore>,
    ) -> Self {
        Self {
            config,
            projection_resolver: Arc::new(ProjectionRegistry::default()),
            deferred_tool_resolver: None,
            artifact_store,
            lashlang_execution_trace_config: RlmLashlangExecutionTraceConfig::default(),
            process_lifecycle: OnceLock::new(),
        }
    }

    pub fn with_projection_resolver(
        mut self,
        projection_resolver: Arc<dyn ProjectionResolver>,
    ) -> Self {
        self.projection_resolver = projection_resolver;
        self
    }

    /// Wire a host-provided [`DeferredToolResolver`](lash_lashlang_runtime::DeferredToolResolver)
    /// that resolves Lashlang call-paths absent from the link-time host
    /// environment into Tool Grants. Most hosts ship none.
    pub fn with_deferred_tool_resolver(mut self, resolver: SharedDeferredToolResolver) -> Self {
        self.deferred_tool_resolver = Some(resolver);
        self
    }

    pub fn with_lashlang_execution_trace(
        mut self,
        sink: Option<Arc<dyn TraceSink>>,
        trace_context: TraceContext,
    ) -> Self {
        self.lashlang_execution_trace_config = RlmLashlangExecutionTraceConfig {
            sink,
            trace_context,
        };
        self
    }

    pub fn with_lashlang_execution_sink(mut self, sink: Arc<dyn TraceSink>) -> Self {
        self.lashlang_execution_trace_config.sink = Some(sink);
        self
    }

    pub fn with_lashlang_execution_jsonl_path(
        mut self,
        path: impl Into<std::path::PathBuf>,
    ) -> Self {
        self.lashlang_execution_trace_config.sink =
            Some(Arc::new(lash_core::JsonlTraceSink::new(path.into())));
        self
    }

    pub fn artifact_store(&self) -> Arc<dyn LashlangArtifactStore> {
        Arc::clone(&self.artifact_store)
    }

    fn process_lifecycle(&self) -> bool {
        self.process_lifecycle.get().copied().unwrap_or(false)
    }

    /// Build the compile-time Lashlang surface over a host-built plugin host.
    ///
    /// Operation over the factory and a plugin host: the caller supplies a plugin
    /// host containing this protocol factory plus any tool plugins to resolve,
    /// and whether process lifecycle is available.
    pub fn lashlang_compile_surface(
        &self,
        plugin_host: &PluginHost,
        process_lifecycle_available: bool,
        request: LashlangCompileSurfaceRequest,
    ) -> Result<LashlangCompileSurface, PluginError> {
        let plugins = plugin_host.build_session_with_parent(
            &request.session_id,
            None,
            None,
            SessionAuthorityContext {
                plugin_options: request.execution_env_spec.plugin_options,
                ..Default::default()
            },
        )?;
        let tool_catalog = plugins.resolved_tool_catalog(&request.session_id)?;
        let config = rlm_protocol_config(self.config.clone(), process_lifecycle_available);
        let surface = rlm_lashlang_surface(&config, process_lifecycle_available)
            .with_plugin_extensions(plugin_host.extensions())
            .map_err(PluginError::Registration)?;
        let host_environment = surface
            .host_environment(&tool_catalog)
            .map_err(PluginError::Registration)?;
        Ok(LashlangCompileSurface {
            host_environment,
            tool_catalog,
            surface,
        })
    }

    /// Compile a Lashlang module against the compile-time surface, persisting the
    /// artifact through this factory's artifact store.
    pub async fn compile_lashlang_module(
        &self,
        plugin_host: &PluginHost,
        process_lifecycle_available: bool,
        request: LashlangModuleCompileRequest,
    ) -> Result<ModuleCompileOutput, LashlangModuleCompileError> {
        let surface = self
            .lashlang_compile_surface(
                plugin_host,
                process_lifecycle_available,
                LashlangCompileSurfaceRequest {
                    session_id: request.session_id,
                    execution_env_spec: request.execution_env_spec,
                },
            )
            .map_err(|err| {
                lashlang::ModuleCompileError::Link(lashlang::ModuleCompileDiagnostic {
                    stage: lashlang::ModuleCompileStage::Link,
                    message: err.to_string(),
                    offset: None,
                    span: None,
                    line: None,
                    column: None,
                    diagnostic: Some(err.to_string()),
                })
            })?;
        lashlang::compile_module(lashlang::ModuleCompileRequest {
            source: &request.source,
            environment: &surface.host_environment,
            artifact_store: Some(self.artifact_store.as_ref()),
        })
        .await
    }
}

impl Default for RlmProtocolPluginFactory {
    fn default() -> Self {
        Self::new(
            RlmProtocolPluginConfig::default(),
            lashlang::global_in_memory_lashlang_artifact_store(),
        )
    }
}

impl PluginFactory for RlmProtocolPluginFactory {
    fn id(&self) -> &'static str {
        RLM_PROTOCOL_PLUGIN_ID
    }

    fn process_engine_contributions(
        &self,
        ctx: &ProcessEngineContributionContext<'_>,
    ) -> Result<Vec<Arc<dyn ProcessEngine>>, PluginError> {
        let process_lifecycle = ctx.process_lifecycle_available();
        // Record for the per-session plugin surface; install runs before any
        // session is built on this (shared) factory.
        let _ = self.process_lifecycle.set(process_lifecycle);
        let config = rlm_protocol_config(self.config.clone(), process_lifecycle);
        let surface = rlm_lashlang_surface(&config, process_lifecycle)
            .with_plugin_extensions(ctx.extensions())
            .map_err(PluginError::Registration)?;
        let engine = LashlangProcessEngine::new(Arc::clone(&self.artifact_store), surface)
            .with_execution_trace(
                self.lashlang_execution_trace_config.sink.clone(),
                ctx.trace_context().clone(),
            );
        Ok(vec![Arc::new(engine)])
    }

    fn build(&self, ctx: &PluginSessionContext) -> Result<Arc<dyn SessionPlugin>, PluginError> {
        let config = rlm_protocol_config(self.config.clone(), self.process_lifecycle());
        let lashlang_surface = LashlangSurface::new(
            config.lashlang_abilities,
            config.lashlang_language_features,
            lashlang::LashlangHostCatalog::new(),
        )
        .with_plugin_extensions(&ctx.extensions)
        .map_err(PluginError::Registration)?;
        Ok(Arc::new(RlmProtocolPlugin {
            config,
            projection_resolver: Arc::clone(&self.projection_resolver),
            deferred_tool_resolver: self.deferred_tool_resolver.clone(),
            artifact_store: Arc::clone(&self.artifact_store),
            lashlang_execution_trace_config: self.lashlang_execution_trace_config.clone(),
            lashlang_surface,
            last_prompt_usage: Arc::new(RwLock::new(None)),
        }))
    }
}

/// Request for [`RlmProtocolPluginFactory::lashlang_compile_surface`].
pub struct LashlangCompileSurfaceRequest {
    pub session_id: String,
    pub execution_env_spec: lash_core::ProcessExecutionEnvSpec,
}

impl LashlangCompileSurfaceRequest {
    pub fn new(
        session_id: impl Into<String>,
        execution_env_spec: lash_core::ProcessExecutionEnvSpec,
    ) -> Self {
        Self {
            session_id: session_id.into(),
            execution_env_spec,
        }
    }
}

/// Request for [`RlmProtocolPluginFactory::compile_lashlang_module`].
pub struct LashlangModuleCompileRequest {
    pub session_id: String,
    pub source: String,
    pub execution_env_spec: lash_core::ProcessExecutionEnvSpec,
}

impl LashlangModuleCompileRequest {
    pub fn new(
        session_id: impl Into<String>,
        source: impl Into<String>,
        execution_env_spec: lash_core::ProcessExecutionEnvSpec,
    ) -> Self {
        Self {
            session_id: session_id.into(),
            source: source.into(),
            execution_env_spec,
        }
    }
}

/// Output of [`RlmProtocolPluginFactory::lashlang_compile_surface`].
pub struct LashlangCompileSurface {
    pub host_environment: LashlangHostEnvironment,
    pub tool_catalog: Arc<lash_core::ToolCatalog>,
    pub surface: LashlangSurface,
}

pub type LashlangModuleCompileError = lashlang::ModuleCompileError;
pub type ModuleCompileOutput = lashlang::ModuleCompileOutput;

struct RlmProtocolPlugin {
    config: RlmProtocolPluginConfig,
    projection_resolver: Arc<dyn ProjectionResolver>,
    deferred_tool_resolver: Option<SharedDeferredToolResolver>,
    artifact_store: Arc<dyn LashlangArtifactStore>,
    lashlang_execution_trace_config: RlmLashlangExecutionTraceConfig,
    lashlang_surface: LashlangSurface,
    last_prompt_usage: SharedPromptUsage,
}

impl SessionPlugin for RlmProtocolPlugin {
    fn id(&self) -> &'static str {
        RLM_PROTOCOL_PLUGIN_ID
    }

    fn register(&self, reg: &mut PluginRegistrar) -> Result<(), PluginError> {
        register_rlm_protocol_plugin(
            reg,
            self.config.clone(),
            Arc::clone(&self.projection_resolver),
            self.deferred_tool_resolver.clone(),
            Arc::clone(&self.artifact_store),
            self.lashlang_execution_trace_config.clone(),
            self.lashlang_surface.clone(),
            Arc::clone(&self.last_prompt_usage),
        )
    }
}
