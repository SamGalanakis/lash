mod capability;
mod rlm;
mod rlm_support;

use std::sync::Arc;

pub use capability::{
    Capability, CapabilityRegistry, StaticCapability, SubagentSpawnContext, TierCapability,
    TierPluginSource, default_explore_plugin_source, default_registry,
};

use lash_core::plugin::{PluginError, PluginFactory, PluginSessionContext};
use lash_core::{PluginSpec, PluginSpecFactory, SessionSpec, SessionToolAccess, ToolProvider};

pub use rlm::spawn_agent_tool_definition;

pub struct SubagentsPluginFactory {
    session_spec: SessionSpec,
    tool_access: SessionToolAccess,
    registry: Arc<CapabilityRegistry>,
}

impl SubagentsPluginFactory {
    pub fn new(registry: Arc<CapabilityRegistry>) -> Self {
        Self {
            session_spec: SessionSpec::inherit(),
            tool_access: SessionToolAccess::default(),
            registry,
        }
    }

    pub fn with_session_spec(mut self, spec: SessionSpec) -> Self {
        self.session_spec = spec;
        self
    }

    pub fn with_tool_access(mut self, access: SessionToolAccess) -> Self {
        self.tool_access = access;
        self
    }

    pub fn with_hidden_tools<I, S>(mut self, tools: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.tool_access
            .hidden_tools
            .extend(tools.into_iter().map(Into::into));
        self
    }
}

impl PluginFactory for SubagentsPluginFactory {
    fn id(&self) -> &'static str {
        "subagents"
    }

    fn build(
        &self,
        ctx: &PluginSessionContext,
    ) -> Result<Arc<dyn lash_core::SessionPlugin>, PluginError> {
        let registry = Arc::clone(&self.registry);
        let session_spec = self.session_spec.clone();
        let tool_access = self.tool_access.clone();
        let parent_subagent = ctx.subagent.clone();

        let provider: Arc<dyn ToolProvider> = Arc::new(
            rlm::RlmSubagentToolsProvider {
                registry: Arc::clone(&registry),
                session_spec: session_spec.clone(),
                tool_access,
                parent_subagent,
                include_submit_error: ctx.subagent.is_some(),
            }
            .into_provider(),
        );

        PluginSpecFactory::new(
            "subagents",
            Arc::new(move |_ctx| {
                Ok(PluginSpec::new()
                    .with_tool_surface_contributor(Arc::new(move |ctx| {
                        rlm_support::subagent_surface_contribution(ctx)
                    }))
                    .with_tool_provider(Arc::clone(&provider)))
            }),
        )
        .build(ctx)
    }
}

#[cfg(test)]
mod tests;
