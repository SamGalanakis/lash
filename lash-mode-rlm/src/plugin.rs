use std::sync::Arc;

use lash::plugin::{
    ModeProtocolDriverPlugin, ModeRuntimeContext, ModeSessionContext, ModeSessionPlugin,
    PluginError, PluginFactory, PluginRegistrar, PluginSessionContext, SessionPlugin,
};
use lash::{
    ExecutionMode, ModeBuildInput, ModePreamble, PromptContribution, SessionError,
    ToolResultProjectionMode, ToolResultProjectionPluginConfig,
};

use crate::driver::build_rlm_preamble;
use crate::rlm_support::{SearchToolsProvider, bound_variables_prompt_contributions};
use crate::stream_mask;

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize, Default)]
#[serde(default)]
pub struct RlmModePluginConfig {
    pub observe_projection: ToolResultProjectionPluginConfig,
}

pub struct BuiltinRlmModePluginFactory {
    config: RlmModePluginConfig,
}

impl BuiltinRlmModePluginFactory {
    pub fn new(config: RlmModePluginConfig) -> Self {
        Self { config }
    }
}

impl Default for BuiltinRlmModePluginFactory {
    fn default() -> Self {
        Self::new(RlmModePluginConfig::default())
    }
}

impl PluginFactory for BuiltinRlmModePluginFactory {
    fn id(&self) -> &'static str {
        "mode_rlm"
    }

    fn build(&self, ctx: &PluginSessionContext) -> Result<Arc<dyn SessionPlugin>, PluginError> {
        Ok(Arc::new(RlmModePlugin {
            active: matches!(ctx.execution_mode, ExecutionMode::Rlm),
            provider: Arc::new(SearchToolsProvider::new()),
            config: self.config.clone(),
        }))
    }
}

struct RlmModePlugin {
    active: bool,
    provider: Arc<SearchToolsProvider>,
    config: RlmModePluginConfig,
}

impl SessionPlugin for RlmModePlugin {
    fn id(&self) -> &'static str {
        "mode_rlm"
    }

    fn register(&self, reg: &mut PluginRegistrar) -> Result<(), PluginError> {
        if !self.active {
            return Ok(());
        }
        reg.mode().session(Arc::new(RlmModeSession {
            config: self.config.clone(),
        }))?;
        reg.mode().protocol_driver(Arc::new(RlmProtocolDriver))?;
        reg.tools()
            .provider(Arc::clone(&self.provider) as Arc<dyn lash::ToolProvider>)?;
        let bound_vars_hook: lash::plugin::PromptContributor = Arc::new(move |ctx| {
            Box::pin(async move { Ok(bound_variables_prompt_contributions(&ctx)) })
        });
        reg.prompt().contribute(bound_vars_hook);
        let projection_config = self.config.observe_projection.clone();
        let print_output_hook: lash::plugin::PromptContributor = Arc::new(move |_ctx| {
            let projection_config = projection_config.clone();
            Box::pin(async move { Ok(vec![print_output_prompt_contribution(&projection_config)]) })
        });
        reg.prompt().contribute(print_output_hook);
        stream_mask::register_stream_mask(reg)?;
        Ok(())
    }
}

struct RlmProtocolDriver;

impl ModeProtocolDriverPlugin for RlmProtocolDriver {
    fn mode_id(&self) -> &'static str {
        ExecutionMode::Rlm.plugin_id()
    }

    fn build_preamble(&self, input: ModeBuildInput) -> ModePreamble {
        build_rlm_preamble(input)
    }
}

fn print_output_prompt_contribution(
    config: &ToolResultProjectionPluginConfig,
) -> PromptContribution {
    let mode_label = match config.mode {
        ToolResultProjectionMode::Bytes => "bytes",
        ToolResultProjectionMode::Tokens => "tokens",
    };
    PromptContribution::execution(
        "Print Output",
        format!(
            "`print` output is capped before reinjection using the current RLM print limit (mode: `{}`, limit: {}, max_lines: {}). If you see a cap/truncation note, narrow the expression and inspect specific fields or slices instead of dumping the whole value.",
            mode_label, config.limit, config.max_lines,
        ),
    )
}

struct RlmModeSession {
    config: RlmModePluginConfig,
}

#[async_trait::async_trait]
impl ModeSessionPlugin for RlmModeSession {
    async fn initialize_session(
        &self,
        mut ctx: ModeSessionContext<'_>,
    ) -> Result<(), SessionError> {
        ctx.set_execution_output_projection(self.config.observe_projection.clone());
        ctx.start_lashlang_runtime().await
    }

    async fn restore_session(
        &self,
        mut ctx: ModeSessionContext<'_>,
        state: &lash::runtime::PersistedSessionState,
    ) -> Result<(), SessionError> {
        if let Some(snapshot) = state.execution_state_snapshot().map(|bytes| bytes.to_vec()) {
            ctx.restore_execution_state(&snapshot).await?;
        }
        for body in state
            .session_graph
            .active_path_plugins(lash::INTERNAL_RLM_GLOBALS_PATCH_PLUGIN_TYPE)
        {
            let patch = serde_json::from_value::<lash::RlmGlobalsPatchPluginBody>(body.clone())
                .map_err(|err| {
                    SessionError::Protocol(format!("invalid RLM globals patch node: {err}"))
                })?;
            ctx.apply_rlm_globals_patch(&patch).await?;
        }
        Ok(())
    }

    async fn append_session_nodes(
        &self,
        mut ctx: ModeSessionContext<'_>,
        nodes: &[lash::SessionAppendNode],
    ) -> Result<(), SessionError> {
        for node in nodes {
            let lash::SessionAppendNode::Plugin { plugin_type, body } = node else {
                continue;
            };
            if plugin_type != lash::INTERNAL_RLM_GLOBALS_PATCH_PLUGIN_TYPE {
                continue;
            }
            let patch = serde_json::from_value::<lash::RlmGlobalsPatchPluginBody>(body.clone())
                .map_err(|err| {
                    SessionError::Protocol(format!("invalid RLM globals patch node body: {err}"))
                })?;
            ctx.apply_rlm_globals_patch(&patch).await?;
        }
        Ok(())
    }

    fn configure_runtime_from_request(
        &self,
        mut ctx: ModeRuntimeContext<'_>,
        request: &lash::SessionCreateRequest,
    ) {
        if let lash::ModeExtras::Rlm(extras) = &request.mode_extras {
            ctx.set_termination_mode(extras.termination.clone());
        }
    }
}
