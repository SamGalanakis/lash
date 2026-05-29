use std::sync::Arc;

use lash_core::ToolProvider;
use lash_core::plugin::{
    PluginError, PluginFactory, PluginSessionContext, PluginSpec, SessionPlugin,
    StaticPluginFactory,
};

use crate::service::tool_discovery_provider;
use crate::surface::rlm_tool_surface;

/// Plugin factory for the `search_tools` discovery surface.
///
/// Declares its provider and tool-surface contribution through a
/// [`PluginSpec`] driven by [`StaticPluginFactory`], so it does not
/// hand-roll the `SessionPlugin` + `register` ceremony.
pub struct ToolDiscoveryPluginFactory {
    inner: StaticPluginFactory,
}

impl ToolDiscoveryPluginFactory {
    pub fn new() -> Self {
        let spec = PluginSpec::new()
            .with_tool_provider(Arc::new(tool_discovery_provider()) as Arc<dyn ToolProvider>)
            .with_tool_surface_contributor(Arc::new(rlm_tool_surface));
        Self {
            inner: StaticPluginFactory::new("tool_discovery", spec),
        }
    }
}

impl Default for ToolDiscoveryPluginFactory {
    fn default() -> Self {
        Self::new()
    }
}

impl PluginFactory for ToolDiscoveryPluginFactory {
    fn id(&self) -> &'static str {
        self.inner.id()
    }

    fn build(&self, ctx: &PluginSessionContext) -> Result<Arc<dyn SessionPlugin>, PluginError> {
        self.inner.build(ctx)
    }
}
