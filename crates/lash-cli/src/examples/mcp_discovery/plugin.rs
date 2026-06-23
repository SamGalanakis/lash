use std::sync::Arc;

use lash_core::ToolProvider;
use lash_core::plugin::{
    PluginError, PluginFactory, PluginSessionContext, PluginSpec, SessionPlugin,
    StaticPluginFactory,
};

use super::catalogue_preview::catalogue_preview_contribution;
use super::service::tool_discovery_provider;

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
        let spec = PluginSpec::new()
            .with_tool_provider(Arc::new(tool_discovery_provider()) as Arc<dyn ToolProvider>)
            .with_prompt_contributor(Arc::new(|ctx: lash_core::plugin::PromptHookContext| {
                Box::pin(async move {
                    // The projected catalog ranges over every member; the
                    // resident core is already documented as full prompt docs,
                    // so the preview advertises the searchable tail.
                    let catalog = ctx
                        .sessions
                        .shared_tool_catalog(&ctx.session_id)
                        .await
                        .unwrap_or_default();
                    Ok(catalogue_preview_contribution(catalog.as_ref())
                        .into_iter()
                        .collect())
                })
            }));
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
