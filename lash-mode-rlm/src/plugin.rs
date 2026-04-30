use std::sync::{Arc, Mutex};

use lash::plugin::{
    CheckpointHookContext, ModeProtocolDriverPlugin, ModeRuntimeContext, ModeSessionContext,
    ModeSessionPlugin, PluginDirective, PluginError, PluginFactory, PluginRegistrar,
    PluginSessionContext, SessionPlugin,
};
use lash::{
    CheckpointKind, ExecutionMode, ModeBuildInput, ModePreamble, PromptContribution, SessionError,
    ToolResultProjectionPluginConfig,
};
use lash_rlm_types::RlmCreateExtras;

use crate::driver::{RlmProjectorConfig, build_rlm_preamble};
use crate::rlm_support::{
    BoundVariablesCache, apply_globals_patch_nodes, budget_prompt_contributions,
    restore_execution_state_and_globals, user_input_patch_from_events, user_input_patch_from_nodes,
};
use crate::stream_mask;

const BUDGET_WARNING_STATUS: &str = "rlm_context_budget_warning";

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct RlmModePluginConfig {
    pub observe_projection: ToolResultProjectionPluginConfig,
    #[serde(default = "default_max_output_chars")]
    pub max_output_chars: usize,
    #[serde(default = "default_baton_soft_warn_tokens")]
    pub baton_soft_warn_tokens: Option<usize>,
}

fn default_max_output_chars() -> usize {
    10_000
}

fn default_baton_soft_warn_tokens() -> Option<usize> {
    Some(100_000)
}

impl Default for RlmModePluginConfig {
    fn default() -> Self {
        Self {
            observe_projection: ToolResultProjectionPluginConfig::default(),
            max_output_chars: default_max_output_chars(),
            baton_soft_warn_tokens: default_baton_soft_warn_tokens(),
        }
    }
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
            config: self.config.clone(),
        }))
    }
}

struct RlmModePlugin {
    active: bool,
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
        let mode_session = Arc::new(RlmModeSession::new(self.config.clone()));
        reg.mode().session(mode_session.clone())?;
        reg.mode().protocol_driver(Arc::new(RlmProtocolDriver {
            config: self.config.clone(),
        }))?;

        let bound_vars_cache = Arc::new(BoundVariablesCache::new());
        let bound_vars_hook: lash::plugin::PromptContributor = Arc::new(move |ctx| {
            let cache = Arc::clone(&bound_vars_cache);
            Box::pin(async move { Ok(cache.contributions(&ctx)) })
        });
        reg.prompt().contribute(bound_vars_hook);

        let max_budget_tokens = self.config.baton_soft_warn_tokens;
        let budget_hook: lash::plugin::PromptContributor = Arc::new(move |ctx| {
            Box::pin(async move { Ok(budget_prompt_contributions(&ctx, max_budget_tokens)) })
        });
        reg.prompt().contribute(budget_hook);

        let print_output_hook: lash::plugin::PromptContributor = Arc::new(move |_ctx| {
            Box::pin(async move { Ok(vec![print_output_prompt_contribution()]) })
        });
        reg.prompt().contribute(print_output_hook);

        let warn_session = mode_session.clone();
        reg.turn().checkpoint(Arc::new(move |ctx| {
            let session = warn_session.clone();
            Box::pin(async move { session.soft_warn_directives(ctx) })
        }));

        stream_mask::register_stream_mask(reg)?;
        Ok(())
    }
}

struct RlmProtocolDriver {
    config: RlmModePluginConfig,
}

impl ModeProtocolDriverPlugin for RlmProtocolDriver {
    fn mode_id(&self) -> &str {
        "rlm"
    }

    fn build_preamble(&self, input: ModeBuildInput) -> ModePreamble {
        build_rlm_preamble(
            input,
            RlmProjectorConfig {
                max_output_chars: self.config.max_output_chars,
            },
        )
    }
}

fn print_output_prompt_contribution() -> PromptContribution {
    PromptContribution::execution(
        "Print Output",
        "`print` output is capped. Keep full tool results in variables; print only lengths, selected fields, samples, or slices. Do not print large objects just to hand-copy IDs back into code.",
    )
}

struct RlmModeSession {
    config: RlmModePluginConfig,
    user_input_count: Mutex<usize>,
    warned_at_threshold: Mutex<bool>,
}

impl RlmModeSession {
    fn new(config: RlmModePluginConfig) -> Self {
        Self {
            config,
            user_input_count: Mutex::new(0),
            warned_at_threshold: Mutex::new(false),
        }
    }

    fn soft_warn_directives(
        &self,
        ctx: CheckpointHookContext,
    ) -> Result<Vec<PluginDirective>, PluginError> {
        if ctx.checkpoint != CheckpointKind::AfterWork {
            return Ok(Vec::new());
        }
        let Some(threshold) = self.config.baton_soft_warn_tokens else {
            return Ok(Vec::new());
        };
        let used = ctx.state.token_usage().total().max(0) as usize;
        if used == 0 {
            return Ok(Vec::new());
        }
        if used < threshold {
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
        Ok(vec![PluginDirective::emit_events(vec![
            lash::PluginSurfaceEvent::Status {
                key: BUDGET_WARNING_STATUS.to_string(),
                label: "context budget".to_string(),
                detail: Some(format!(
                    "{used} tokens used; warn at {threshold}; choose handoff path"
                )),
                transient_ms: Some(8_000),
            },
        ])])
    }
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
        restore_execution_state_and_globals(&mut ctx, state).await?;
        let projection = state.shared_projection();
        let (patch, count) = user_input_patch_from_events(projection.active_events.iter(), 0);
        *self
            .user_input_count
            .lock()
            .expect("rlm user input count lock") = count;
        ctx.apply_mode_globals_patch(&patch).await?;
        Ok(())
    }

    async fn append_session_nodes(
        &self,
        mut ctx: ModeSessionContext<'_>,
        nodes: &[lash::SessionAppendNode],
    ) -> Result<(), SessionError> {
        let start_index = *self
            .user_input_count
            .lock()
            .expect("rlm user input count lock");
        let (user_patch, next_count) = user_input_patch_from_nodes(nodes, start_index);
        ctx.apply_mode_globals_patch(&user_patch).await?;
        *self
            .user_input_count
            .lock()
            .expect("rlm user input count lock") = next_count;

        apply_globals_patch_nodes(&mut ctx, nodes).await
    }

    fn configure_runtime_from_request(
        &self,
        mut ctx: ModeRuntimeContext<'_>,
        request: &lash::SessionCreateRequest,
    ) {
        if let Ok(Some(RlmCreateExtras { termination })) = request
            .mode_extras
            .decode::<RlmCreateExtras>(&ExecutionMode::new("rlm"))
        {
            ctx.set_rlm_termination_mode(termination);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct NoopPromptManager;

    #[async_trait::async_trait]
    impl lash::plugin::SessionManager for NoopPromptManager {
        async fn snapshot_current(
            &self,
        ) -> Result<lash::PersistedSessionState, lash::plugin::PluginError> {
            Err(lash::plugin::PluginError::Session("not used".to_string()))
        }

        async fn snapshot_session(
            &self,
            _session_id: &str,
        ) -> Result<lash::PersistedSessionState, lash::plugin::PluginError> {
            Err(lash::plugin::PluginError::Session("not used".to_string()))
        }

        async fn tool_catalog(
            &self,
            _session_id: &str,
        ) -> Result<Vec<serde_json::Value>, lash::plugin::PluginError> {
            Ok(Vec::new())
        }

        async fn create_session(
            &self,
            _request: lash::SessionCreateRequest,
        ) -> Result<lash::SessionHandle, lash::plugin::PluginError> {
            Err(lash::plugin::PluginError::Session("not used".to_string()))
        }

        async fn close_session(&self, _session_id: &str) -> Result<(), lash::plugin::PluginError> {
            Ok(())
        }

        async fn start_turn_stream(
            &self,
            _session_id: &str,
            _input: lash::TurnInput,
        ) -> Result<lash::SessionTurnHandle, lash::plugin::PluginError> {
            Err(lash::plugin::PluginError::Session("not used".to_string()))
        }

        async fn await_turn(
            &self,
            _turn_id: &str,
        ) -> Result<lash::AssembledTurn, lash::plugin::PluginError> {
            Err(lash::plugin::PluginError::Session("not used".to_string()))
        }

        async fn cancel_turn(&self, _turn_id: &str) -> Result<(), lash::plugin::PluginError> {
            Ok(())
        }
    }

    fn user_event(id: &str, text: &str) -> lash::SessionEventRecord {
        lash::SessionEventRecord::Conversation(lash::ConversationRecord {
            id: id.to_string(),
            role: lash::MessageRole::User,
            parts: vec![lash::Part {
                id: format!("{id}.p0"),
                kind: lash::PartKind::Text,
                content: text.to_string(),
                attachment: None,
                tool_call_id: None,
                tool_name: None,
                tool_item_id: None,
                tool_signature: None,
                prune_state: lash::PruneState::Intact,
                reasoning_meta: None,
                response_meta: None,
            }]
            .into(),
            user_input: None,
            origin: None,
        })
    }

    #[test]
    fn user_input_patch_from_events_binds_messages_in_order() {
        let events = [
            user_event("u1", "first"),
            lash::SessionEventRecord::Conversation(lash::ConversationRecord {
                id: "a1".to_string(),
                role: lash::MessageRole::Assistant,
                parts: Arc::new(Vec::new()),
                user_input: None,
                origin: None,
            }),
            user_event("u2", "second"),
        ];
        let (patch, count) = user_input_patch_from_events(events.iter(), 0);

        assert_eq!(count, 2);
        assert_eq!(patch.set["user_input_1"], serde_json::json!("first"));
        assert_eq!(patch.set["user_input_2"], serde_json::json!("second"));
        assert!(patch.unset.is_empty());
    }

    #[test]
    fn user_input_patch_from_nodes_continues_existing_count() {
        let nodes = vec![lash::SessionAppendNode::Event {
            event: user_event("u3", "third"),
        }];
        let (patch, count) = user_input_patch_from_nodes(&nodes, 2);

        assert_eq!(count, 3);
        assert_eq!(patch.set["user_input_3"], serde_json::json!("third"));
    }

    #[test]
    fn plugin_origin_user_message_does_not_pollute_user_input_count() {
        let mut event = user_event("plugin", "soft warning");
        let lash::SessionEventRecord::Conversation(record) = &mut event else {
            unreachable!("user_event returns conversation")
        };
        record.origin = Some(lash::MessageOrigin::Plugin {
            plugin_id: "mode_rlm".to_string(),
            transient: false,
        });

        let (patch, count) = user_input_patch_from_events([event].iter(), 0);

        assert_eq!(count, 0);
        assert!(patch.set.is_empty());
    }

    #[test]
    fn budget_prompt_contribution_renders_used_over_configured_budget() {
        let state = lash::SessionStateEnvelope {
            policy: lash::SessionPolicy {
                max_context_tokens: Some(1_050_000),
                ..Default::default()
            },
            last_prompt_usage: Some(lash::PromptUsage {
                prompt_context_tokens: 40_000,
                input_tokens: 40_000,
                cached_input_tokens: 0,
                context_budget_tokens: 47_213,
            }),
            ..Default::default()
        };
        let ctx = lash::plugin::PromptHookContext {
            session_id: "root".to_string(),
            host: std::sync::Arc::new(NoopPromptManager),
            state: lash::SessionReadView::new(state),
            mode_turn_options: lash::ModeTurnOptions::default(),
        };

        let contributions = budget_prompt_contributions(&ctx, Some(200_000));

        assert_eq!(contributions.len(), 1);
        assert!(contributions[0].content.contains("47213 / 200000"));
        assert!(contributions[0].content.contains("23%"));
        assert!(contributions[0].content.contains("Prepare to hand off"));
    }

    #[test]
    fn budget_prompt_contribution_over_threshold_forces_handoff_choice() {
        let state = lash::SessionStateEnvelope {
            last_prompt_usage: Some(lash::PromptUsage {
                prompt_context_tokens: 120_292,
                input_tokens: 120_292,
                cached_input_tokens: 0,
                context_budget_tokens: 120_292,
            }),
            ..Default::default()
        };
        let ctx = lash::plugin::PromptHookContext {
            session_id: "root".to_string(),
            host: std::sync::Arc::new(NoopPromptManager),
            state: lash::SessionReadView::new(state),
            mode_turn_options: lash::ModeTurnOptions::default(),
        };

        let contributions = budget_prompt_contributions(&ctx, Some(100_000));

        assert_eq!(contributions.len(), 1);
        let content = &contributions[0].content;
        assert!(content.contains("120292 / 100000"));
        assert!(content.contains("Do not continue ordinary work"));
        assert!(content.contains("Call `pass_baton(task=..., seed={...})`"));
        assert!(content.contains("directly relevant context"));
        assert!(content.contains("The next agent keeps the same tool access"));
        assert!(content.contains("do not carry irrelevant history"));
        assert!(!content.contains("Otherwise"));
        assert!(!content.contains("otherwise"));
    }

    #[test]
    fn soft_budget_warning_emits_surface_event_not_user_message() {
        let session = RlmModeSession::new(RlmModePluginConfig {
            baton_soft_warn_tokens: Some(100_000),
            ..Default::default()
        });
        let state = lash::SessionStateEnvelope {
            token_usage: lash::TokenUsage {
                input_tokens: 120_292,
                ..Default::default()
            },
            ..Default::default()
        };
        let directives = session
            .soft_warn_directives(lash::plugin::CheckpointHookContext {
                session_id: "root".to_string(),
                checkpoint: lash::CheckpointKind::AfterWork,
                state: lash::SessionReadView::new(state),
                host: std::sync::Arc::new(NoopPromptManager),
            })
            .expect("warning directives");

        assert_eq!(directives.len(), 1);
        let lash::plugin::PluginDirective::EmitEvents { events } = &directives[0] else {
            panic!("budget warning must be a surface event, not an injected message");
        };
        assert_eq!(events.len(), 1);
        let lash::PluginSurfaceEvent::Status {
            key,
            label,
            detail,
            transient_ms,
        } = &events[0]
        else {
            panic!("budget warning should use a typed status surface event");
        };
        assert_eq!(key, BUDGET_WARNING_STATUS);
        assert_eq!(label, "context budget");
        assert!(detail.as_deref().is_some_and(|text| {
            text.contains("120292 tokens used") && text.contains("choose handoff path")
        }));
        assert_eq!(*transient_ms, Some(8_000));
    }

    #[test]
    fn budget_prompt_contribution_omits_without_configured_budget() {
        let state = lash::SessionStateEnvelope {
            policy: lash::SessionPolicy {
                max_context_tokens: Some(1_050_000),
                ..Default::default()
            },
            token_usage: lash::TokenUsage {
                input_tokens: 47_213,
                ..Default::default()
            },
            ..Default::default()
        };
        let ctx = lash::plugin::PromptHookContext {
            session_id: "root".to_string(),
            host: std::sync::Arc::new(NoopPromptManager),
            state: lash::SessionReadView::new(state),
            mode_turn_options: lash::ModeTurnOptions::default(),
        };

        let contributions = budget_prompt_contributions(&ctx, None);

        assert!(contributions.is_empty());
    }

    #[test]
    fn budget_prompt_contribution_omits_without_used_tokens() {
        let state = lash::SessionStateEnvelope {
            policy: lash::SessionPolicy {
                max_context_tokens: Some(1_050_000),
                ..Default::default()
            },
            ..Default::default()
        };
        let ctx = lash::plugin::PromptHookContext {
            session_id: "root".to_string(),
            host: std::sync::Arc::new(NoopPromptManager),
            state: lash::SessionReadView::new(state),
            mode_turn_options: lash::ModeTurnOptions::default(),
        };

        let contributions = budget_prompt_contributions(&ctx, Some(200_000));

        assert!(contributions.is_empty());
    }
}
