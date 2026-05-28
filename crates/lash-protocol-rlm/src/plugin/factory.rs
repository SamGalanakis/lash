use std::sync::{Arc, RwLock};

use lash_core::plugin::{
    PluginError, PluginFactory, PluginRegistrar, PluginSessionContext, SessionPlugin,
};

use super::registration::register_rlm_protocol_plugin;
use super::{RLM_PROTOCOL_PLUGIN_ID, RlmProtocolPluginConfig};
use crate::driver::SharedPromptUsage;
use crate::projection::{ProjectionRegistry, ProjectionResolver};

pub struct RlmProtocolPluginFactory {
    config: RlmProtocolPluginConfig,
    projection_resolver: Arc<dyn ProjectionResolver>,
}

impl RlmProtocolPluginFactory {
    pub fn new(config: RlmProtocolPluginConfig) -> Self {
        Self {
            config,
            projection_resolver: Arc::new(ProjectionRegistry::default()),
        }
    }

    pub fn with_projection_resolver(
        mut self,
        projection_resolver: Arc<dyn ProjectionResolver>,
    ) -> Self {
        self.projection_resolver = projection_resolver;
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

    fn lashlang_abilities(&self) -> lashlang::LashlangAbilities {
        self.config.lashlang_abilities
    }

    fn build(&self, _ctx: &PluginSessionContext) -> Result<Arc<dyn SessionPlugin>, PluginError> {
        Ok(Arc::new(RlmProtocolPlugin {
            config: self.config.clone(),
            projection_resolver: Arc::clone(&self.projection_resolver),
            last_prompt_usage: Arc::new(RwLock::new(None)),
        }))
    }
}

struct RlmProtocolPlugin {
    config: RlmProtocolPluginConfig,
    projection_resolver: Arc<dyn ProjectionResolver>,
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
            Arc::clone(&self.last_prompt_usage),
        )
    }
}
