use std::sync::Arc;

use lash_core::ToolProvider;
use lash_core::plugin::{
    PluginError, PluginFactory, PluginRegistrar, PluginSessionContext, SessionPlugin,
};

use crate::service::ToolDiscoveryToolsProvider;
use crate::surface::rlm_tool_surface;

#[derive(Default)]
pub struct ToolDiscoveryPluginFactory;

impl ToolDiscoveryPluginFactory {
    pub fn new() -> Self {
        Self
    }
}

impl PluginFactory for ToolDiscoveryPluginFactory {
    fn id(&self) -> &'static str {
        "tool_discovery"
    }

    fn build(&self, _ctx: &PluginSessionContext) -> Result<Arc<dyn SessionPlugin>, PluginError> {
        Ok(Arc::new(ToolDiscoveryPlugin {
            provider: Arc::new(ToolDiscoveryToolsProvider::new()),
        }))
    }
}

struct ToolDiscoveryPlugin {
    provider: Arc<ToolDiscoveryToolsProvider>,
}

impl SessionPlugin for ToolDiscoveryPlugin {
    fn id(&self) -> &'static str {
        "tool_discovery"
    }

    fn register(&self, reg: &mut PluginRegistrar) -> Result<(), PluginError> {
        reg.tools()
            .provider(Arc::clone(&self.provider) as Arc<dyn ToolProvider>)?;
        reg.surface().contribute(Arc::new(rlm_tool_surface));
        Ok(())
    }
}
