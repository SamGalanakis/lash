use std::sync::Arc;

use lash::plugin::{
    ModeProtocolDriverPlugin, ModeRuntimeContext, ModeSessionContext, ModeSessionPlugin,
    PluginError, PluginFactory, PluginRegistrar, PluginSessionContext, SessionPlugin,
};
use lash::tools::DiscoveryToolsProvider;
use lash::{
    ExecutionMode, ModeBuildInput, ModePreamble, PromptContribution, SessionError,
    ToolResultProjectionPluginConfig,
};

use crate::driver::build_rlm_preamble;
use crate::rlm_support::bound_variables_prompt_contributions;
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
            active: ctx.execution_mode == ExecutionMode::new("rlm"),
            provider: Arc::new(DiscoveryToolsProvider::new()),
            config: self.config.clone(),
        }))
    }
}

struct RlmModePlugin {
    active: bool,
    provider: Arc<DiscoveryToolsProvider>,
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
        let root_final_response_hook: lash::plugin::PromptContributor = Arc::new(move |ctx| {
            Box::pin(async move {
                Ok(root_final_response_prompt_contribution(
                    &ctx.state,
                    &ctx.mode_turn_options.rlm_termination(),
                ))
            })
        });
        reg.prompt().contribute(root_final_response_hook);
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
    fn mode_id(&self) -> &str {
        "rlm"
    }

    fn build_preamble(&self, input: ModeBuildInput) -> ModePreamble {
        build_rlm_preamble(input)
    }
}

fn print_output_prompt_contribution(
    _config: &ToolResultProjectionPluginConfig,
) -> PromptContribution {
    // The concrete cap numbers (bytes/tokens/max_lines) change per
    // session and the model has no way to act on them. Keep the prompt
    // focused on the recovery rule.
    PromptContribution::execution(
        "Print Output",
        "`print` output is capped before reinjection. If you see a cap/truncation note, narrow the expression and inspect specific fields or slices instead of dumping the whole value.",
    )
}

fn root_final_response_prompt_contribution(
    state: &lash::SessionReadView,
    termination: &lash_rlm_types::RlmTermination,
) -> Vec<PromptContribution> {
    let is_root_chat_session = state.policy().execution_mode == ExecutionMode::new("rlm")
        && state.session_id() == "root"
        && !state.policy().autonomous
        && matches!(
            termination,
            lash_rlm_types::RlmTermination::ProseWithoutFence
        );
    if !is_root_chat_session {
        return Vec::new();
    }

    vec![
        PromptContribution::guidance(
            "Final Response Formatting",
            "This is the root session and the final reply is rendered directly to the user. When there is no Required output schema, finish with prose or `submit` a polished Markdown string. Do not `submit` records, lists, raw tool results, or JSON-shaped diagnostics as the final root answer; use `print` for inspection and summarize the result instead. Structured records are appropriate for subagents and typed output schemas, not for untyped root replies.",
        )
        .with_priority(100),
    ]
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
        for patch in state
            .session_graph
            .active_events()
            .into_iter()
            .filter_map(|event| match event {
                lash::session_model::SessionEventRecord::Mode(event) => match event.rlm_event() {
                    Some(lash_rlm_types::RlmModeEvent::RlmGlobalsPatch(patch)) => Some(patch),
                    _ => None,
                },
                _ => None,
            })
        {
            ctx.apply_mode_globals_patch(&patch).await?;
        }
        Ok(())
    }

    async fn append_session_nodes(
        &self,
        mut ctx: ModeSessionContext<'_>,
        nodes: &[lash::SessionAppendNode],
    ) -> Result<(), SessionError> {
        for node in nodes {
            match node {
                lash::SessionAppendNode::Event {
                    event: lash::SessionEventRecord::Mode(event),
                } => {
                    if let Some(lash_rlm_types::RlmModeEvent::RlmGlobalsPatch(patch)) =
                        event.rlm_event()
                    {
                        ctx.apply_mode_globals_patch(&patch).await?;
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn configure_runtime_from_request(
        &self,
        mut ctx: ModeRuntimeContext<'_>,
        request: &lash::SessionCreateRequest,
    ) {
        if let Ok(Some(extras)) = request
            .mode_extras
            .decode::<lash_rlm_types::RlmCreateExtras>(&ExecutionMode::new("rlm"))
        {
            ctx.set_rlm_termination_mode(extras.termination);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state(
        execution_mode: ExecutionMode,
        session_id: &str,
        autonomous: bool,
    ) -> lash::SessionReadView {
        lash::SessionReadView::new(lash::SessionStateEnvelope {
            session_id: session_id.to_string(),
            policy: lash::SessionPolicy {
                execution_mode,
                autonomous,
                ..lash::SessionPolicy::default()
            },
            ..lash::SessionStateEnvelope::default()
        })
    }

    #[test]
    fn root_final_response_guidance_is_root_untyped_only() {
        let contribution = root_final_response_prompt_contribution(
            &state(ExecutionMode::new("rlm"), "root", false),
            &lash_rlm_types::RlmTermination::ProseWithoutFence,
        );
        assert_eq!(contribution.len(), 1);
        assert_eq!(
            contribution[0].title.as_deref(),
            Some("Final Response Formatting")
        );
        assert!(contribution[0].content.contains("polished Markdown string"));

        assert!(
            root_final_response_prompt_contribution(
                &state(ExecutionMode::new("rlm"), "child-session", false),
                &lash_rlm_types::RlmTermination::ProseWithoutFence,
            )
            .is_empty()
        );
        assert!(
            root_final_response_prompt_contribution(
                &state(ExecutionMode::new("rlm"), "root", true),
                &lash_rlm_types::RlmTermination::ProseWithoutFence,
            )
            .is_empty()
        );
        assert!(
            root_final_response_prompt_contribution(
                &state(ExecutionMode::new("rlm"), "root", false),
                &lash_rlm_types::RlmTermination::Finish { schema: None },
            )
            .is_empty()
        );
        assert!(
            root_final_response_prompt_contribution(
                &state(ExecutionMode::standard(), "root", false),
                &lash_rlm_types::RlmTermination::ProseWithoutFence,
            )
            .is_empty()
        );
    }
}
