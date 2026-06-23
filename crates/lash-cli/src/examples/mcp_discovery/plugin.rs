use std::sync::Arc;

use lash_core::ToolProvider;
use lash_core::plugin::{
    PluginError, PluginFactory, PluginSessionContext, PluginSpec, SessionPlugin,
    StaticPluginFactory,
};

use super::catalogue_preview::catalogue_preview_contribution;
use super::service::tool_discovery_provider_with_catalog;

/// Plugin factory for the `search_tools` discovery catalog.
///
/// Declares its provider through a [`PluginSpec`] driven by
/// [`StaticPluginFactory`], so it does not hand-roll the `SessionPlugin` +
/// `register` ceremony. Alongside the `search_tools` tool, it advertises the
/// deferred (non-resident) MCP tools to the model through a catalogue-preview
/// prompt contribution — the recommended way to make a large MCP tool set
/// discoverable under the flat catalog.
pub struct ToolDiscoveryPluginFactory {
    inner: StaticPluginFactory,
}

impl ToolDiscoveryPluginFactory {
    pub fn new() -> Self {
        Self::with_catalog(Vec::new())
    }

    pub fn with_catalog(extra_catalog: Vec<serde_json::Value>) -> Self {
        let extra_catalog = Arc::new(extra_catalog);
        let discovery_catalog = Arc::clone(&extra_catalog);
        let spec = PluginSpec::new()
            .with_tool_provider(Arc::new(tool_discovery_provider_with_catalog(
                extra_catalog.as_ref().clone(),
            )) as Arc<dyn ToolProvider>)
            .with_prompt_contributor(Arc::new(
                move |ctx: lash_core::plugin::PromptHookContext| {
                    let discovery_catalog = Arc::clone(&discovery_catalog);
                    Box::pin(async move {
                        // The projected resident catalog ranges over members. When
                        // a non-resident discovery catalog is supplied, preview that
                        // indexed tail; otherwise preserve the default resident
                        // discovery behavior.
                        let resident_catalog = ctx
                            .sessions
                            .shared_tool_catalog(&ctx.session_id)
                            .await
                            .unwrap_or_default();
                        let preview_catalog = if discovery_catalog.is_empty() {
                            resident_catalog
                        } else {
                            discovery_catalog
                        };
                        Ok(catalogue_preview_contribution(preview_catalog.as_ref())
                            .into_iter()
                            .collect())
                    })
                },
            ));
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
