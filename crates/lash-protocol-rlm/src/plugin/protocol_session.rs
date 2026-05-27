use std::sync::{Arc, Mutex};

use lash_core::llm::types::LlmRequest;
use lash_core::plugin::{
    CheckpointHookContext, PluginDirective, PluginError, ProtocolBeforeLlmCallContext,
    ProtocolLlmCallAction, ProtocolRuntimeContext, ProtocolSessionContext, ProtocolSessionPlugin,
};
use lash_core::{CheckpointKind, SessionError};
use lash_rlm_types::RlmCreateExtras;

use super::budget_warning::BUDGET_WARNING_STATUS;
use super::forced_continuation::{forced_continue_as_args, process_support_unavailable};
use super::runtime_state::RlmRuntimeState;
use super::{RLM_PROTOCOL_PLUGIN_ID, RlmProtocolPluginConfig};
use crate::projection::RlmSeed;

pub(super) struct RlmProtocolSession {
    config: RlmProtocolPluginConfig,
    runtime_state: Arc<RlmRuntimeState>,
    warned_at_threshold: Mutex<bool>,
}

impl RlmProtocolSession {
    pub(super) fn new(
        config: RlmProtocolPluginConfig,
        runtime_state: Arc<RlmRuntimeState>,
    ) -> Result<Self, SessionError> {
        config
            .validate()
            .map_err(|err| SessionError::Protocol(err.to_string()))?;
        Ok(Self {
            runtime_state,
            config,
            warned_at_threshold: Mutex::new(false),
        })
    }

    pub(super) async fn projected_binding_prompt_contributions(
        &self,
    ) -> Vec<lash_core::PromptContribution> {
        self.runtime_state
            .projected_binding_prompt_contributions()
            .await
    }

    pub(super) fn soft_warn_directives(
        &self,
        ctx: CheckpointHookContext,
    ) -> Result<Vec<PluginDirective>, PluginError> {
        if ctx.checkpoint != CheckpointKind::AfterWork {
            return Ok(Vec::new());
        }
        let Some(threshold) = self.config.continue_as_soft_warn_tokens else {
            return Ok(Vec::new());
        };
        let used = ctx.state.token_usage().total().max(0) as usize;
        if used == 0 || used < threshold {
            return Ok(Vec::new());
        }
        let mut warned = self
            .warned_at_threshold
            .lock()
            .map_err(|_| PluginError::Session("rlm soft-warning state poisoned".to_string()))?;
        if *warned {
            return Ok(Vec::new());
        }
        *warned = true;
        Ok(vec![PluginDirective::emit_runtime_events(vec![
            lash_core::PluginRuntimeEvent::Status {
                key: BUDGET_WARNING_STATUS.to_string(),
                label: "context budget".to_string(),
                detail: Some(format!(
                    "{used} tokens used; warn at {threshold}; choose handoff path"
                )),
            },
        ])])
    }
}

#[async_trait::async_trait]
impl ProtocolSessionPlugin for RlmProtocolSession {
    async fn initialize_session(
        &self,
        _ctx: ProtocolSessionContext<'_>,
    ) -> Result<(), SessionError> {
        Ok(())
    }

    async fn restore_session(
        &self,
        _ctx: ProtocolSessionContext<'_>,
        state: &lash_core::runtime::RuntimeSessionState,
    ) -> Result<(), SessionError> {
        self.runtime_state
            .restore_runtime_session_state(state)
            .await
    }

    async fn append_session_nodes(
        &self,
        _ctx: ProtocolSessionContext<'_>,
        nodes: &[lash_core::SessionAppendNode],
    ) -> Result<(), SessionError> {
        self.runtime_state.append_session_nodes(nodes).await
    }

    async fn apply_session_extension(
        &self,
        extension: lash_core::ProtocolSessionExtensionHandle,
    ) -> Result<(), SessionError> {
        self.runtime_state.apply_session_extension(extension).await
    }

    async fn validate_turn_extension(
        &self,
        extension: &lash_core::ProtocolTurnExtensionHandle,
    ) -> Result<(), SessionError> {
        self.runtime_state.validate_turn_extension(extension).await
    }

    async fn before_llm_call(
        &self,
        ctx: ProtocolBeforeLlmCallContext<'_>,
        request: &LlmRequest,
    ) -> Result<Option<ProtocolLlmCallAction>, PluginError> {
        let Some(threshold) = self.config.continue_as_forced_fallback_tokens else {
            return Ok(None);
        };
        let Some(usage) = ctx.latest_prompt_usage.as_ref() else {
            return Ok(None);
        };
        if usage.context_budget_tokens < threshold {
            return Ok(None);
        }

        let observed_tokens = usage.context_budget_tokens;
        let args = forced_continue_as_args(&ctx, request, threshold, observed_tokens).await?;
        let seed = RlmSeed::from_tool_args(&args)
            .map_err(|err| PluginError::Session(format!("forced continue_as {err}")))?;
        let referenced_handles =
            crate::control_tools::collect_seed_process_handle_ids(args.get("seed"));
        let referenced_handles_vec = referenced_handles.into_iter().collect::<Vec<_>>();
        ctx.processes
            .validate_visible(&ctx.session_id, &referenced_handles_vec)
            .await
            .map_err(|err| {
                PluginError::Session(format!(
                    "forced continue_as process handle validation failed: {err}"
                ))
            })?;
        let task = args
            .get("task")
            .and_then(serde_json::Value::as_str)
            .filter(|task| !task.trim().is_empty())
            .ok_or_else(|| {
                PluginError::Session(
                    "forced continue_as fallback returned missing or empty `task`".to_string(),
                )
            })?
            .to_string();

        let metadata = serde_json::Map::from_iter([
            (
                "trigger".to_string(),
                serde_json::Value::String("context_budget_tokens".to_string()),
            ),
            ("threshold_tokens".to_string(), serde_json::json!(threshold)),
            (
                "observed_tokens".to_string(),
                serde_json::json!(observed_tokens),
            ),
            (
                "source".to_string(),
                serde_json::Value::String("rlm_forced_fallback".to_string()),
            ),
        ]);
        let current_snapshot = ctx
            .host
            .snapshot_session(&ctx.session_id)
            .await
            .map_err(|err| {
                PluginError::Session(format!("failed to snapshot current session: {err}"))
            })?;
        let session_id = crate::control_tools::create_continue_as_successor(
            ctx.host.as_ref(),
            &ctx.session_id,
            current_snapshot,
            task,
            seed,
            crate::control_tools::ContinueAsHandoff {
                relation_reason: "continue_as_forced_context_fallback".to_string(),
                usage_source: "continue_as_forced_context_fallback".to_string(),
                metadata,
            },
        )
        .await
        .map_err(PluginError::Session)?;
        if let Err(err) = ctx
            .processes
            .transfer(
                &ctx.session_id,
                &session_id,
                referenced_handles_vec.clone(),
                ctx.process_scope(),
            )
            .await
        {
            let _ = ctx.host.close_session(&session_id).await;
            return Err(PluginError::Session(format!(
                "forced continue_as process handle transfer failed: {err}"
            )));
        }
        if let Err(err) = ctx
            .processes
            .cancel_unreferenced(
                &ctx.session_id,
                referenced_handles_vec.clone(),
                ctx.process_scope(),
            )
            .await
        {
            if referenced_handles_vec.is_empty() && process_support_unavailable(&err) {
                return Ok(Some(ProtocolLlmCallAction::Handoff { session_id }));
            }
            let _ = ctx.host.close_session(&session_id).await;
            return Err(PluginError::Session(format!(
                "forced continue_as process handle cleanup failed after successor creation: {err}"
            )));
        }
        Ok(Some(ProtocolLlmCallAction::Handoff { session_id }))
    }

    fn configure_runtime_from_request(
        &self,
        mut ctx: ProtocolRuntimeContext<'_>,
        request: &lash_core::SessionCreateRequest,
    ) -> Result<(), SessionError> {
        if let Some(extras) = request
            .plugin_options
            .decode::<RlmCreateExtras>(RLM_PROTOCOL_PLUGIN_ID)
            .map_err(|err| SessionError::Protocol(format!("invalid RLM create options: {err}")))?
        {
            let options = lash_core::ProtocolTurnOptions::typed(extras.termination)?;
            ctx.set_protocol_turn_options(options);
        }
        Ok(())
    }
}
