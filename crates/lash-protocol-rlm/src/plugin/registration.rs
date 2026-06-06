use std::sync::Arc;

use lash_core::plugin::{PluginError, PluginRegistrar};

use super::RlmProtocolPluginConfig;
use super::budget_warning::BudgetUsageObserver;
use super::prose_projector::RlmAssistantProseProjector;
use super::protocol_driver::RlmProtocolDriver;
use super::protocol_session::RlmProtocolSession;
use super::runtime_state::{RlmCodeExecutor, RlmRuntimeState};
use super::tool_args::normalize_projected_tool_args;
use crate::driver::SharedPromptUsage;
use crate::projection::{
    ProjectionResolver, RLM_TURN_INPUT_PLUGIN_ID, RlmProjectionExtension, rlm_history_projection,
};
use crate::rlm_support::render_bound_variables;
use crate::stream_mask;

pub(super) fn register_rlm_protocol_plugin(
    reg: &mut PluginRegistrar,
    config: RlmProtocolPluginConfig,
    projection_resolver: Arc<dyn ProjectionResolver>,
    last_prompt_usage: SharedPromptUsage,
) -> Result<(), PluginError> {
    let bound_variables_inline_char_limit = config.bound_variables_inline_char_limit;
    let runtime_state = Arc::new(
        RlmRuntimeState::new(config.clone(), projection_resolver)
            .map_err(|err| PluginError::Session(err.to_string()))?,
    );
    let code_executor = Arc::new(RlmCodeExecutor::new(Arc::clone(&runtime_state)));
    let protocol_session = Arc::new(RlmProtocolSession::new(
        config.clone(),
        Arc::clone(&runtime_state),
    ));
    reg.protocol().session(protocol_session.clone())?;
    reg.execution().code_executor(code_executor)?;
    reg.output()
        .assistant_prose_projector(Arc::new(RlmAssistantProseProjector))?;
    reg.protocol().protocol_driver(Arc::new(RlmProtocolDriver {
        config,
        last_prompt_usage: Arc::clone(&last_prompt_usage),
    }))?;
    reg.tools()
        .provider(Arc::new(crate::control_tools::RlmControlToolsProvider))?;
    reg.tool_calls().before(Arc::new(|ctx| {
        Box::pin(async move { normalize_projected_tool_args(ctx) })
    }));

    register_bound_variables_prompt_contributor(
        reg,
        Arc::clone(&runtime_state),
        bound_variables_inline_char_limit,
    );
    register_projected_bindings_prompt_contributor(reg, Arc::clone(&protocol_session));

    // Per-turn `prompt_usage` is captured here and passed to the projector via a
    // shared cell so the budget line can ride in the volatile turn-tail message
    // instead of poisoning the cached system prefix.
    reg.history().prepare_turn(
        10,
        Arc::new(BudgetUsageObserver {
            cell: last_prompt_usage,
        }),
    );

    let warn_session = protocol_session.clone();
    reg.turn().checkpoint(Arc::new(move |ctx| {
        let session = warn_session.clone();
        Box::pin(async move { session.soft_warn_directives(ctx) })
    }));

    stream_mask::register_stream_mask(reg)?;
    Ok(())
}

fn register_bound_variables_prompt_contributor(
    reg: &mut PluginRegistrar,
    runtime_state: Arc<RlmRuntimeState>,
    inline_char_limit: usize,
) {
    let bound_vars_hook: lash_core::plugin::PromptContributor = Arc::new(move |ctx| {
        let runtime_state = Arc::clone(&runtime_state);
        Box::pin(async move {
            let globals = runtime_state.bound_variable_values().await;
            let history_len = rlm_history_projection(&ctx.state.chronological_projection()).len();
            Ok(vec![render_bound_variables(
                &globals,
                history_len,
                inline_char_limit,
            )])
        })
    });
    reg.prompt().contribute(bound_vars_hook);
}

fn register_projected_bindings_prompt_contributor(
    reg: &mut PluginRegistrar,
    protocol_session: Arc<RlmProtocolSession>,
) {
    reg.prompt().contribute(Arc::new(move |ctx| {
        let session = protocol_session.clone();
        Box::pin(async move {
            let mut contributions = session.projected_binding_prompt_contributions().await;
            if let Some(extension) = ctx
                .turn_context
                .plugin_input::<RlmProjectionExtension>(RLM_TURN_INPUT_PLUGIN_ID)
            {
                contributions.extend(RlmProjectionExtension::prompt_contributions_for(
                    &extension.bindings,
                ));
            }
            Ok(contributions)
        })
    }));
}
