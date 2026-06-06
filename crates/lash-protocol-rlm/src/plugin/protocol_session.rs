use std::sync::{Arc, Mutex};

use lash_core::plugin::{
    CheckpointHookContext, PluginDirective, PluginError, ProtocolRuntimeContext,
    ProtocolSessionContext, ProtocolSessionPlugin,
};
use lash_core::{CheckpointKind, SessionError};
use lash_rlm_types::RlmCreateExtras;

use super::budget_warning::BUDGET_WARNING_STATUS;
use super::runtime_state::RlmRuntimeState;
use super::{RLM_PROTOCOL_PLUGIN_ID, RlmProtocolPluginConfig};

pub(super) struct RlmProtocolSession {
    config: RlmProtocolPluginConfig,
    runtime_state: Arc<RlmRuntimeState>,
    warned_at_threshold: Mutex<bool>,
}

impl RlmProtocolSession {
    pub(super) fn new(
        config: RlmProtocolPluginConfig,
        runtime_state: Arc<RlmRuntimeState>,
    ) -> Self {
        Self {
            runtime_state,
            config,
            warned_at_threshold: Mutex::new(false),
        }
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
                    "{used} tokens used; warn at {threshold}; choose frame switch path"
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
            let options = lash_core::ProtocolTurnOptions::typed(extras)?;
            ctx.set_protocol_turn_options(options);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::plugin::budget_warning::BUDGET_WARNING_STATUS;
    use crate::projection::{ProjectionRegistry, RlmProjectedBindings};

    struct NoopPromptManager;

    #[async_trait::async_trait]
    impl lash_core::plugin::runtime_host::SessionStateService for NoopPromptManager {
        async fn snapshot_current(
            &self,
        ) -> Result<lash_core::SessionSnapshot, lash_core::plugin::PluginError> {
            Err(lash_core::plugin::PluginError::Session(
                "not used".to_string(),
            ))
        }

        async fn snapshot_session(
            &self,
            _session_id: &str,
        ) -> Result<lash_core::SessionSnapshot, lash_core::plugin::PluginError> {
            Err(lash_core::plugin::PluginError::Session(
                "not used".to_string(),
            ))
        }

        async fn tool_catalog(
            &self,
            _session_id: &str,
        ) -> Result<Vec<serde_json::Value>, lash_core::plugin::PluginError> {
            Ok(Vec::new())
        }
    }

    #[async_trait::async_trait]
    impl lash_core::plugin::runtime_host::SessionLifecycleService for NoopPromptManager {
        async fn create_session(
            &self,
            _request: lash_core::SessionCreateRequest,
        ) -> Result<lash_core::SessionHandle, lash_core::plugin::PluginError> {
            Err(lash_core::plugin::PluginError::Session(
                "not used".to_string(),
            ))
        }

        async fn close_session(
            &self,
            _session_id: &str,
        ) -> Result<(), lash_core::plugin::PluginError> {
            Ok(())
        }
    }

    #[async_trait::async_trait]
    impl lash_core::plugin::runtime_host::SessionGraphService for NoopPromptManager {}

    fn test_session(config: RlmProtocolPluginConfig) -> RlmProtocolSession {
        let runtime_state = Arc::new(
            RlmRuntimeState::new(config.clone(), Arc::new(ProjectionRegistry::default()))
                .expect("runtime state"),
        );
        RlmProtocolSession::new(config, runtime_state)
    }

    #[tokio::test]
    async fn session_projection_extension_rejects_duplicate_names() {
        let session = test_session(RlmProtocolPluginConfig::default());
        session
            .apply_session_extension(crate::rlm_session_projection_extension(
                RlmProjectedBindings::new()
                    .bind_json("current_query", serde_json::json!("first"))
                    .expect("first bind"),
            ))
            .await
            .expect("first projection");

        let duplicate = session
            .apply_session_extension(crate::rlm_session_projection_extension(
                RlmProjectedBindings::new()
                    .bind_json("current_query", serde_json::json!("second"))
                    .expect("second bind"),
            ))
            .await;
        let Err(err) = duplicate else {
            panic!("duplicate session projection should fail");
        };
        assert!(err.to_string().contains("current_query"));
    }

    #[tokio::test]
    async fn session_projection_prompt_contribution_lists_names() {
        let session = test_session(RlmProtocolPluginConfig::default());
        session
            .apply_session_extension(crate::rlm_session_projection_extension(
                RlmProjectedBindings::new()
                    .bind_json("current_query", serde_json::json!("first"))
                    .expect("bind"),
            ))
            .await
            .expect("projection");

        let contributions = session.projected_binding_prompt_contributions().await;
        assert_eq!(contributions.len(), 1);
        assert!(contributions[0].content.contains("`current_query`"));
        assert!(contributions[0].content.contains("read-only host value"));
    }

    #[test]
    fn soft_budget_warning_emits_surface_event_not_user_message() {
        let session = test_session(RlmProtocolPluginConfig {
            continue_as_soft_warn_tokens: Some(100_000),
            ..Default::default()
        });
        let state = lash_core::SessionSnapshot {
            token_usage: lash_core::TokenUsage {
                input_tokens: 120_292,
                ..Default::default()
            },
            ..Default::default()
        };
        let directives = session
            .soft_warn_directives(lash_core::plugin::CheckpointHookContext {
                session_id: "root".to_string(),
                checkpoint: lash_core::CheckpointKind::AfterWork,
                state: lash_core::SessionReadView::from_snapshot(&state),
                sessions: Arc::new(NoopPromptManager),
                session_lifecycle: Arc::new(NoopPromptManager),
                session_graph: Arc::new(NoopPromptManager),
            })
            .expect("warning directives");

        assert_eq!(directives.len(), 1);
        let lash_core::plugin::PluginDirective::EmitRuntimeEvents { events } = &directives[0]
        else {
            panic!("budget warning must be a runtime event, not an injected message");
        };
        assert_eq!(events.len(), 1);
        let lash_core::PluginRuntimeEvent::Status { key, label, detail } = &events[0] else {
            panic!("budget warning should use a typed status runtime event");
        };
        assert_eq!(key, BUDGET_WARNING_STATUS);
        assert_eq!(label, "context budget");
        assert!(detail.as_deref().is_some_and(|text| {
            text.contains("120292 tokens used") && text.contains("choose frame switch path")
        }));
    }
}
