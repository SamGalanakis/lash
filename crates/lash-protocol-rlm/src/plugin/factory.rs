use std::sync::{Arc, RwLock};

use lash_core::plugin::{
    PluginError, PluginFactory, PluginRegistrar, PluginSessionContext, SessionPlugin,
};
use lash_core::{TraceContext, TraceSink};
use lash_lashlang_runtime::{LashlangArtifactStore, LashlangSurface};

use super::registration::register_rlm_protocol_plugin;
use super::{RLM_PROTOCOL_PLUGIN_ID, RlmProtocolPluginConfig};
use crate::driver::SharedPromptUsage;
use crate::executor::RlmLashlangExecutionTraceConfig;
use crate::projection::{ProjectionRegistry, ProjectionResolver};

pub struct RlmProtocolPluginFactory {
    config: RlmProtocolPluginConfig,
    projection_resolver: Arc<dyn ProjectionResolver>,
    artifact_store: Arc<dyn LashlangArtifactStore>,
    lashlang_execution_trace_config: RlmLashlangExecutionTraceConfig,
}

impl RlmProtocolPluginFactory {
    pub fn new(config: RlmProtocolPluginConfig) -> Self {
        Self {
            config,
            projection_resolver: Arc::new(ProjectionRegistry::default()),
            artifact_store: lashlang::global_in_memory_lashlang_artifact_store(),
            lashlang_execution_trace_config: RlmLashlangExecutionTraceConfig::default(),
        }
    }

    pub fn with_projection_resolver(
        mut self,
        projection_resolver: Arc<dyn ProjectionResolver>,
    ) -> Self {
        self.projection_resolver = projection_resolver;
        self
    }

    pub fn with_lashlang_artifact_store(
        mut self,
        artifact_store: Arc<dyn LashlangArtifactStore>,
    ) -> Self {
        self.artifact_store = artifact_store;
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
}

impl Default for RlmProtocolPluginFactory {
    fn default() -> Self {
        Self::new(RlmProtocolPluginConfig::default())
    }
}

impl PluginFactory for RlmProtocolPluginFactory {
    fn id(&self) -> &'static str {
        RLM_PROTOCOL_PLUGIN_ID
    }

    fn build(&self, ctx: &PluginSessionContext) -> Result<Arc<dyn SessionPlugin>, PluginError> {
        let lashlang_surface = LashlangSurface::new(
            self.config.lashlang_abilities,
            self.config.lashlang_language_features,
            lashlang::LashlangHostCatalog::new(),
        )
        .with_plugin_extensions(&ctx.extensions)
        .map_err(PluginError::Registration)?;
        Ok(Arc::new(RlmProtocolPlugin {
            config: self.config.clone(),
            projection_resolver: Arc::clone(&self.projection_resolver),
            artifact_store: Arc::clone(&self.artifact_store),
            lashlang_execution_trace_config: self.lashlang_execution_trace_config.clone(),
            lashlang_surface,
            last_prompt_usage: Arc::new(RwLock::new(None)),
        }))
    }
}

struct RlmProtocolPlugin {
    config: RlmProtocolPluginConfig,
    projection_resolver: Arc<dyn ProjectionResolver>,
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
            Arc::clone(&self.artifact_store),
            self.lashlang_execution_trace_config.clone(),
            self.lashlang_surface.clone(),
            Arc::clone(&self.last_prompt_usage),
        )
    }
}
