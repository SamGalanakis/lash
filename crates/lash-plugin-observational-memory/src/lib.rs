use std::sync::Arc;

use lash_core::plugin::{
    PluginError, PluginFactory, PluginLifecycleEvent, PluginRegistrar, PluginSessionContext,
    SessionPlugin,
};
use lash_core::{
    ObservationalMemoryConfig, SessionAppendNode, SessionStateChangedContext,
    StandardContextApproach,
};

mod constants;
mod context_transform;
mod graph_state;
mod host;
mod model;
mod prompts;
mod transitions;
mod worker;

pub use constants::{
    ACTIVE_STATE_PLUGIN_TYPE, BUFFERED_OBSERVATION_PLUGIN_TYPE, BUFFERED_REFLECTION_PLUGIN_TYPE,
    OBSERVATIONAL_MEMORY_PLUGIN_ID,
};

use context_transform::ObservationalMemoryTransform;
use host::OmRuntimeHost;
use transitions::{
    maybe_buffer_observations, maybe_buffer_reflection, should_run_async_maintenance,
};

pub fn active_memory_state_node(
    body: impl serde::Serialize,
) -> Result<SessionAppendNode, serde_json::Error> {
    Ok(SessionAppendNode::plugin(
        ACTIVE_STATE_PLUGIN_TYPE,
        serde_json::to_value(body)?,
    ))
}

#[derive(Default)]
pub struct ObservationalMemoryPluginFactory;

impl PluginFactory for ObservationalMemoryPluginFactory {
    fn id(&self) -> &'static str {
        OBSERVATIONAL_MEMORY_PLUGIN_ID
    }

    fn supported_standard_context_approaches(
        &self,
    ) -> &'static [lash_core::StandardContextApproachKind] {
        &[lash_core::StandardContextApproachKind::ObservationalMemory]
    }

    fn build(&self, ctx: &PluginSessionContext) -> Result<Arc<dyn SessionPlugin>, PluginError> {
        if ctx.execution_mode != lash_core::ExecutionMode::standard() {
            return Ok(Arc::new(DisabledObservationalMemoryPlugin));
        }
        let Some(StandardContextApproach::ObservationalMemory(config)) =
            &ctx.standard_context_approach
        else {
            return Ok(Arc::new(DisabledObservationalMemoryPlugin));
        };
        Ok(Arc::new(ObservationalMemoryPlugin {
            config: config.clone(),
        }))
    }
}

struct DisabledObservationalMemoryPlugin;

impl SessionPlugin for DisabledObservationalMemoryPlugin {
    fn id(&self) -> &'static str {
        OBSERVATIONAL_MEMORY_PLUGIN_ID
    }

    fn register(&self, _reg: &mut PluginRegistrar) -> Result<(), PluginError> {
        Ok(())
    }
}

struct ObservationalMemoryPlugin {
    config: ObservationalMemoryConfig,
}

impl SessionPlugin for ObservationalMemoryPlugin {
    fn id(&self) -> &'static str {
        OBSERVATIONAL_MEMORY_PLUGIN_ID
    }

    fn register(&self, reg: &mut PluginRegistrar) -> Result<(), PluginError> {
        reg.history().prepare_turn(
            100,
            Arc::new(ObservationalMemoryTransform::new(self.config.clone())),
        );

        let config = self.config.clone();
        reg.session()
            .on_event(observational_memory_event_hook(config));

        Ok(())
    }
}

fn observational_memory_event_hook(
    config: ObservationalMemoryConfig,
) -> lash_core::plugin::PluginLifecycleEventHook {
    Arc::new(move |event| {
        let config = config.clone();
        Box::pin(async move {
            if let PluginLifecycleEvent::TurnPersisted(ctx) = event {
                maybe_spawn_post_persist_memory_maintenance(config, ctx).await?;
            }
            Ok(())
        })
    })
}

async fn maybe_spawn_post_persist_memory_maintenance(
    config: ObservationalMemoryConfig,
    ctx: SessionStateChangedContext,
) -> Result<(), PluginError> {
    let graph = ctx.state.to_owned_state().session_graph;
    if !should_run_async_maintenance(&config, &graph) {
        return Ok(());
    }
    run_async_maintenance(config, graph, ctx).await
}

async fn run_async_maintenance(
    config: ObservationalMemoryConfig,
    graph: lash_core::SessionGraph,
    ctx: SessionStateChangedContext,
) -> Result<(), PluginError> {
    let om_host = OmRuntimeHost::new(&ctx.session_id, &ctx.host, ctx.direct_completions.clone());
    maybe_buffer_observations(&config, &om_host, ctx.state.policy(), &graph).await?;
    maybe_buffer_reflection(&config, &om_host, ctx.state.policy(), &graph).await?;
    Ok(())
}

#[cfg(test)]
mod tests;
