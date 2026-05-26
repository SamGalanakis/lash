mod capability;
mod rlm;
mod rlm_support;

use std::sync::Arc;

pub use capability::{
    Capability, CapabilityContext, CapabilityRegistry, DEFAULT_EXPLORE_EXECUTION_MODE,
    StaticCapability, TierCapability, TierExecutionMode, default_explore_execution_mode,
    default_registry,
};

use lash_core::plugin::{PluginError, PluginFactory, PluginSessionContext};
use lash_core::{PluginSpec, PluginSpecFactory, SessionPolicy, SessionSpec, ToolProvider};

pub use rlm::spawn_agent_tool_definition;

pub struct SubagentSpawnContext<'a> {
    pub parent_session_id: &'a str,
    pub capability: &'a str,
    pub parent_policy: &'a SessionPolicy,
    pub child_policy: &'a SessionPolicy,
}

pub trait SubagentSessionConfigurator: Send + Sync {
    fn configure(
        &self,
        ctx: &SubagentSpawnContext<'_>,
        request: &mut lash_core::SessionCreateRequest,
    ) -> Result<(), String>;
}

#[derive(Default)]
pub struct NoopSubagentSessionConfigurator;

impl SubagentSessionConfigurator for NoopSubagentSessionConfigurator {
    fn configure(
        &self,
        _ctx: &SubagentSpawnContext<'_>,
        _request: &mut lash_core::SessionCreateRequest,
    ) -> Result<(), String> {
        Ok(())
    }
}

pub struct SubagentsPluginFactory {
    session_spec: SessionSpec,
    registry: Arc<CapabilityRegistry>,
    configurator: Arc<dyn SubagentSessionConfigurator>,
}

impl SubagentsPluginFactory {
    pub fn new(registry: Arc<CapabilityRegistry>) -> Self {
        Self {
            session_spec: SessionSpec::inherit(),
            registry,
            configurator: Arc::new(NoopSubagentSessionConfigurator),
        }
    }

    pub fn with_session_spec(mut self, spec: SessionSpec) -> Self {
        self.session_spec = spec;
        self
    }

    pub fn with_session_configurator(
        mut self,
        configurator: Arc<dyn SubagentSessionConfigurator>,
    ) -> Self {
        self.configurator = configurator;
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
        let configurator = Arc::clone(&self.configurator);
        let execution_mode = ctx.execution_mode.clone();
        let parent_subagent = ctx.subagent.clone();

        let is_rlm = execution_mode == lash_core::ExecutionMode::new("rlm");
        let provider: Option<Arc<dyn ToolProvider>> = if is_rlm {
            Some(Arc::new(rlm::RlmSubagentToolsProvider {
                registry: Arc::clone(&registry),
                session_spec: session_spec.clone(),
                configurator: Arc::clone(&configurator),
                parent_subagent,
                include_submit_error: ctx.subagent.is_some(),
            }))
        } else {
            None
        };

        PluginSpecFactory::new(
            "subagents",
            Arc::new(move |_ctx| {
                let mut spec =
                    PluginSpec::new().with_tool_surface_contributor(Arc::new(move |ctx| {
                        rlm_support::subagent_surface_contribution(ctx)
                    }));
                if let Some(provider) = provider.as_ref() {
                    spec = spec.with_tool_provider(Arc::clone(provider));
                }
                Ok(spec)
            }),
        )
        .build(ctx)
    }
}

#[cfg(test)]
mod tests;
