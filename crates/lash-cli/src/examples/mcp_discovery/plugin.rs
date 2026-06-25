use std::sync::Arc;

use lash_core::ToolProvider;
use lash_core::plugin::{
    PluginError, PluginFactory, PluginSessionContext, PluginSpec, SessionPlugin,
    StaticPluginFactory,
};
use lash_lashlang_runtime::catalogue_preview_contribution;

use super::service::tool_discovery_provider_with_catalog;

/// Plugin factory for the `search_tools` deferred-discovery catalog.
///
/// Declares its provider through a [`PluginSpec`] driven by
/// [`StaticPluginFactory`], so it does not hand-roll the `SessionPlugin` +
/// `register` ceremony. Alongside the `search_tools` tool, it advertises the
/// deferred (non-resident) MCP tools to the model through a catalogue-preview
/// prompt contribution. `with_catalog(...)` is the CLI/RLM path for a large MCP
/// tail: `search_tools` indexes resident tools plus that explicit tail, while
/// the prompt preview lists only the deferred catalog so resident members keep
/// their full RLM docs.
pub struct ToolDiscoveryPluginFactory {
    inner: StaticPluginFactory,
}

impl ToolDiscoveryPluginFactory {
    pub fn new() -> Self {
        Self::with_catalog(Vec::new())
    }

    pub fn with_catalog(extra_catalog: Vec<serde_json::Value>) -> Self {
        if extra_catalog.is_empty() {
            return Self {
                inner: StaticPluginFactory::new("tool_discovery", PluginSpec::new()),
            };
        }
        let extra_catalog = Arc::new(extra_catalog);
        let discovery_catalog = Arc::clone(&extra_catalog);
        let spec = PluginSpec::new()
            .with_tool_provider(Arc::new(tool_discovery_provider_with_catalog(
                extra_catalog.as_ref().clone(),
            )) as Arc<dyn ToolProvider>)
            .with_prompt_contributor(Arc::new(
                move |_ctx: lash_core::plugin::PromptHookContext| {
                    let discovery_catalog = Arc::clone(&discovery_catalog);
                    Box::pin(async move {
                        Ok(catalogue_preview_contribution(discovery_catalog.as_ref())
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use lash_core::testing::MockSessionManager;
    use lash_core::{
        PluginHost, PromptHookContext, ProtocolTurnOptions, SessionReadView, SessionSnapshot,
        ToolDefinition, TurnContext,
    };
    use lash_tool_support::{LashlangToolBinding, ToolDefinitionLashlangExt};
    use serde_json::json;

    use super::*;

    fn prompt_context() -> PromptHookContext {
        let snapshot = SessionSnapshot::default();
        PromptHookContext {
            session_id: "root".to_string(),
            sessions: Arc::new(MockSessionManager::default()),
            state: SessionReadView::from_snapshot(&snapshot),
            protocol_turn_options: ProtocolTurnOptions::default(),
            turn_context: TurnContext::default(),
        }
    }

    fn catalog_record() -> serde_json::Value {
        let definition = ToolDefinition::raw(
            "tool:mcp/venmo_send",
            "venmo_send",
            "Send a Venmo payment.",
            json!({ "type": "object" }),
            json!({ "type": "object" }),
        )
        .with_lashlang_binding(LashlangToolBinding::new(["appworld"], "venmo_send"));
        let manifest = definition.manifest();
        json!({
            "id": manifest.id,
            "name": manifest.name,
            "description": manifest.description,
            "bindings": manifest.bindings,
            "activation": manifest.activation,
            "contract": manifest.compact_contract,
        })
    }

    fn plugin_host(factory: ToolDiscoveryPluginFactory) -> PluginHost {
        let mut factories = lash_core::testing::test_standard_protocol_factories();
        factories.push(Arc::new(factory));
        PluginHost::new(factories)
    }

    #[tokio::test]
    async fn empty_deferred_catalog_does_not_install_search_overlay() {
        let host = plugin_host(ToolDiscoveryPluginFactory::new());
        let session = host.build_session("root", None).expect("session");

        assert!(
            session
                .tools()
                .tool_manifests()
                .iter()
                .all(|manifest| manifest.name != "search_tools")
        );
        assert!(
            session
                .collect_prompt_contributions(prompt_context())
                .await
                .expect("prompt contributions")
                .is_empty()
        );
    }

    #[tokio::test]
    async fn non_empty_deferred_catalog_installs_search_overlay_and_preview() {
        let host = plugin_host(ToolDiscoveryPluginFactory::with_catalog(vec![
            catalog_record(),
        ]));
        let session = host.build_session("root", None).expect("session");

        assert!(
            session
                .tools()
                .tool_manifests()
                .iter()
                .any(|manifest| manifest.name == "search_tools")
        );
        let contributions = session
            .collect_prompt_contributions(prompt_context())
            .await
            .expect("prompt contributions");
        assert_eq!(contributions.len(), 1);
        assert_eq!(
            contributions[0].title.as_deref(),
            Some("Catalogued Capabilities")
        );
        assert!(contributions[0].content.contains("appworld.venmo_send"));
    }
}
