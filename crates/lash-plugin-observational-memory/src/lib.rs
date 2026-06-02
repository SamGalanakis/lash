use std::sync::Arc;

use lash_core::plugin::{
    PluginError, PluginFactory, PluginLifecycleEvent, PluginRegistrar, PluginSessionContext,
    SessionPlugin,
};
use lash_core::{SessionAppendNode, SessionStateChangedContext};

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

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ObservationalMemoryConfig {
    pub observation_message_tokens: usize,
    pub observation_buffer_tokens: usize,
    pub observation_block_after_tokens: usize,
    pub observation_max_tokens_per_batch: usize,
    pub previous_observer_tokens: usize,
    pub reflection_observation_tokens: usize,
    #[serde(default = "default_reflection_buffer_activation_bps")]
    pub reflection_buffer_activation_bps: u16,
    pub reflection_block_after_tokens: usize,
}

impl Default for ObservationalMemoryConfig {
    fn default() -> Self {
        Self {
            observation_message_tokens: 30_000,
            observation_buffer_tokens: 6_000,
            observation_block_after_tokens: 36_000,
            observation_max_tokens_per_batch: 10_000,
            previous_observer_tokens: 2_000,
            reflection_observation_tokens: 40_000,
            reflection_buffer_activation_bps: default_reflection_buffer_activation_bps(),
            reflection_block_after_tokens: 48_000,
        }
    }
}

impl ObservationalMemoryConfig {
    pub fn observation_buffer_interval_tokens(&self) -> usize {
        self.observation_buffer_tokens
    }

    pub fn observation_retention_tokens(&self) -> usize {
        self.observation_buffer_tokens
    }

    pub fn reflection_buffer_activation_tokens(&self) -> usize {
        self.reflection_observation_tokens
            .saturating_mul(self.reflection_buffer_activation_bps as usize)
            / 10_000
    }
}

const fn default_reflection_buffer_activation_bps() -> u16 {
    5_000
}

pub fn active_memory_state_node(
    body: impl serde::Serialize,
) -> Result<SessionAppendNode, serde_json::Error> {
    Ok(SessionAppendNode::plugin(
        ACTIVE_STATE_PLUGIN_TYPE,
        serde_json::to_value(body)?,
    ))
}

#[derive(Clone, Debug)]
pub struct ObservationalMemoryPluginFactory {
    config: ObservationalMemoryConfig,
}

impl ObservationalMemoryPluginFactory {
    pub fn new(config: ObservationalMemoryConfig) -> Self {
        Self { config }
    }
}

impl Default for ObservationalMemoryPluginFactory {
    fn default() -> Self {
        Self::new(ObservationalMemoryConfig::default())
    }
}

impl PluginFactory for ObservationalMemoryPluginFactory {
    fn id(&self) -> &'static str {
        OBSERVATIONAL_MEMORY_PLUGIN_ID
    }

    fn build(&self, _ctx: &PluginSessionContext) -> Result<Arc<dyn SessionPlugin>, PluginError> {
        Ok(Arc::new(ObservationalMemoryPlugin {
            config: self.config.clone(),
        }))
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
                maybe_spawn_post_persist_memory_maintenance(config, *ctx).await?;
            }
            Ok(())
        })
    })
}

async fn maybe_spawn_post_persist_memory_maintenance(
    config: ObservationalMemoryConfig,
    ctx: SessionStateChangedContext,
) -> Result<(), PluginError> {
    let graph = ctx.state.session_graph();
    if !should_run_async_maintenance(&config, graph) {
        return Ok(());
    }
    run_async_maintenance(config, graph, &ctx).await
}

async fn run_async_maintenance(
    config: ObservationalMemoryConfig,
    graph: &lash_core::SessionGraph,
    ctx: &SessionStateChangedContext,
) -> Result<(), PluginError> {
    let om_host = OmRuntimeHost::new(
        &ctx.session_id,
        &ctx.session_graph,
        ctx.direct_completions.clone(),
    );
    maybe_buffer_observations(&config, &om_host, ctx.state.policy(), graph).await?;
    maybe_buffer_reflection(&config, &om_host, ctx.state.policy(), graph).await?;
    Ok(())
}

#[cfg(test)]
mod tests;
