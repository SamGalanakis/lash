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
use crate::executor::{RlmExecutionState, execute_code};
use crate::rlm_support::{BoundVariablesCache, budget_prompt_contributions};
use crate::stream_mask;

const BUDGET_WARNING_STATUS: &str = "rlm_context_budget_warning";

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct RlmModePluginConfig {
    pub observe_projection: ToolResultProjectionPluginConfig,
    #[serde(default = "default_max_output_chars")]
    pub max_output_chars: usize,
    #[serde(default = "default_continue_as_soft_warn_tokens")]
    pub continue_as_soft_warn_tokens: Option<usize>,
}

fn default_max_output_chars() -> usize {
    10_000
}

fn default_continue_as_soft_warn_tokens() -> Option<usize> {
    Some(100_000)
}

impl Default for RlmModePluginConfig {
    fn default() -> Self {
        Self {
            observe_projection: ToolResultProjectionPluginConfig::default(),
            max_output_chars: default_max_output_chars(),
            continue_as_soft_warn_tokens: default_continue_as_soft_warn_tokens(),
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
        let mode_session = Arc::new(
            RlmModeSession::new(self.config.clone())
                .map_err(|err| PluginError::Session(err.to_string()))?,
        );
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

        let max_budget_tokens = self.config.continue_as_soft_warn_tokens;
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
    warned_at_threshold: Mutex<bool>,
    execution: tokio::sync::Mutex<Option<RlmExecutionState>>,
}

impl RlmModeSession {
    fn new(config: RlmModePluginConfig) -> Result<Self, SessionError> {
        Ok(Self {
            execution: tokio::sync::Mutex::new(Some(RlmExecutionState::new(
                config.observe_projection.clone(),
            )?)),
            config,
            warned_at_threshold: Mutex::new(false),
        })
    }

    fn soft_warn_directives(
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
    async fn initialize_session(&self, _ctx: ModeSessionContext<'_>) -> Result<(), SessionError> {
        Ok(())
    }

    async fn restore_session(
        &self,
        _ctx: ModeSessionContext<'_>,
        state: &lash::runtime::PersistedSessionState,
    ) -> Result<(), SessionError> {
        let mut execution = self.execution.lock().await;
        let execution = execution
            .as_mut()
            .ok_or_else(|| SessionError::Protocol("RLM execution state is busy".to_string()))?;
        if let Some(snapshot) = state.execution_state_snapshot().map(|bytes| bytes.to_vec()) {
            execution.restore_execution_state(&snapshot)?;
        }
        for event in state.read_view().active_events() {
            if let lash::SessionEventRecord::Mode(event) = event
                && let Some(lash_rlm_types::RlmModeEvent::RlmGlobalsPatch(patch)) =
                    event.rlm_event()
            {
                execution.patch_globals(&patch)?;
            }
        }
        Ok(())
    }

    async fn append_session_nodes(
        &self,
        _ctx: ModeSessionContext<'_>,
        nodes: &[lash::SessionAppendNode],
    ) -> Result<(), SessionError> {
        let mut execution = self.execution.lock().await;
        let execution = execution
            .as_mut()
            .ok_or_else(|| SessionError::Protocol("RLM execution state is busy".to_string()))?;
        for node in nodes {
            if let lash::SessionAppendNode::Event {
                event: lash::SessionEventRecord::Mode(event),
            } = node
                && let Some(lash_rlm_types::RlmModeEvent::RlmGlobalsPatch(patch)) =
                    event.rlm_event()
            {
                execution.patch_globals(&patch)?;
            }
        }
        Ok(())
    }

    async fn execute_code(
        &self,
        ctx: lash::ModeExecutionContext,
        request: lash::ExecRequest,
    ) -> Result<lash::ExecResponse, SessionError> {
        let mut guard = self.execution.lock().await;
        let state = guard
            .take()
            .ok_or_else(|| SessionError::Protocol("RLM execution state is busy".to_string()))?;

        let result = execute_code(state, ctx, request).await;
        match result {
            Ok((state, response)) => {
                *guard = Some(state);
                Ok(response)
            }
            Err(err) => {
                *guard = Some(RlmExecutionState::new(
                    self.config.observe_projection.clone(),
                )?);
                Err(err)
            }
        }
    }

    fn execution_state_dirty(&self) -> bool {
        self.execution
            .try_lock()
            .map(|execution| {
                execution
                    .as_ref()
                    .map(|execution| execution.execution_state_dirty())
                    .unwrap_or(true)
            })
            .unwrap_or(true)
    }

    async fn snapshot_execution_state(
        &self,
        _ctx: ModeSessionContext<'_>,
    ) -> Result<Option<Vec<u8>>, SessionError> {
        self.execution
            .lock()
            .await
            .as_mut()
            .ok_or_else(|| SessionError::Protocol("RLM execution state is busy".to_string()))?
            .snapshot_execution_state()
    }

    async fn restore_execution_state(
        &self,
        _ctx: ModeSessionContext<'_>,
        data: &[u8],
    ) -> Result<(), SessionError> {
        self.execution
            .lock()
            .await
            .as_mut()
            .ok_or_else(|| SessionError::Protocol("RLM execution state is busy".to_string()))?
            .restore_execution_state(data)
    }

    fn configure_runtime_from_request(
        &self,
        mut ctx: ModeRuntimeContext<'_>,
        request: &lash::SessionCreateRequest,
    ) {
        if let Ok(Some(RlmCreateExtras { termination })) = request
            .mode_extras
            .decode::<RlmCreateExtras>(&ExecutionMode::new("rlm"))
            && let Ok(options) =
                lash::ModeTurnOptions::typed(ExecutionMode::new("rlm"), termination)
        {
            ctx.set_mode_turn_options(options);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct NoopPromptManager;

    #[async_trait::async_trait]
    impl lash::SessionSnapshotHost for NoopPromptManager {
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
    }

    #[async_trait::async_trait]
    impl lash::ToolCatalogHost for NoopPromptManager {
        async fn tool_catalog(
            &self,
            _session_id: &str,
        ) -> Result<Vec<serde_json::Value>, lash::plugin::PluginError> {
            Ok(Vec::new())
        }
    }

    impl lash::TaskHost for NoopPromptManager {}
    impl lash::DirectCompletionHost for NoopPromptManager {}
    impl lash::TraceHost for NoopPromptManager {}

    #[async_trait::async_trait]
    impl lash::SessionLifecycleHost for NoopPromptManager {
        async fn create_session(
            &self,
            _request: lash::SessionCreateRequest,
        ) -> Result<lash::SessionHandle, lash::plugin::PluginError> {
            Err(lash::plugin::PluginError::Session("not used".to_string()))
        }

        async fn close_session(&self, _session_id: &str) -> Result<(), lash::plugin::PluginError> {
            Ok(())
        }
    }

    #[test]
    fn budget_prompt_contribution_below_advisory_floor_emits_status_only() {
        // 23% of the configured handoff threshold — emit the status line so
        // the model has continuous context-size awareness, but no
        // `continue_as` nag.
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
            state: lash::SessionReadView::from_exported_state(&state),
            mode_turn_options: lash::ModeTurnOptions::default(),
        };

        let contributions = budget_prompt_contributions(&ctx, Some(200_000));

        assert_eq!(contributions.len(), 1);
        let content = &contributions[0].content;
        assert!(content.contains("Tokens: 47213 · handoff threshold: 200000 (23%)"));
        assert!(content.contains("Iteration:"));
        assert!(!content.contains("Look for a clean handoff point"));
        assert!(!content.contains("Budget tight"));
        assert!(!content.contains("Past the handoff threshold"));
    }

    #[test]
    fn budget_prompt_contribution_advisory_tier_60_to_89_pct() {
        // 75% of threshold — advisory: scout for a clean handoff point.
        let state = lash::SessionStateEnvelope {
            last_prompt_usage: Some(lash::PromptUsage {
                prompt_context_tokens: 75_000,
                input_tokens: 75_000,
                cached_input_tokens: 0,
                context_budget_tokens: 75_000,
            }),
            ..Default::default()
        };
        let ctx = lash::plugin::PromptHookContext {
            session_id: "root".to_string(),
            host: std::sync::Arc::new(NoopPromptManager),
            state: lash::SessionReadView::from_exported_state(&state),
            mode_turn_options: lash::ModeTurnOptions::default(),
        };

        let contributions = budget_prompt_contributions(&ctx, Some(100_000));

        assert_eq!(contributions.len(), 1);
        let content = &contributions[0].content;
        assert!(content.contains("Tokens: 75000 · handoff threshold: 100000 (75%)"));
        assert!(content.contains("Look for a clean handoff point"));
        assert!(!content.contains("Budget tight"));
        assert!(!content.contains("Past the handoff threshold"));
    }

    #[test]
    fn budget_prompt_contribution_tight_tier_90_to_99_pct() {
        // 95% of threshold — tight: wrap the current step, then continue_as.
        let state = lash::SessionStateEnvelope {
            last_prompt_usage: Some(lash::PromptUsage {
                prompt_context_tokens: 95_000,
                input_tokens: 95_000,
                cached_input_tokens: 0,
                context_budget_tokens: 95_000,
            }),
            ..Default::default()
        };
        let ctx = lash::plugin::PromptHookContext {
            session_id: "root".to_string(),
            host: std::sync::Arc::new(NoopPromptManager),
            state: lash::SessionReadView::from_exported_state(&state),
            mode_turn_options: lash::ModeTurnOptions::default(),
        };

        let contributions = budget_prompt_contributions(&ctx, Some(100_000));

        assert_eq!(contributions.len(), 1);
        let content = &contributions[0].content;
        assert!(content.contains("Tokens: 95000 · handoff threshold: 100000 (95%)"));
        assert!(content.contains("Budget tight"));
        assert!(content.contains("`continue_as`"));
        assert!(!content.contains("Past the handoff threshold"));
        assert!(!content.contains("Look for a clean handoff point"));
    }

    #[test]
    fn budget_prompt_contribution_over_threshold_forces_handoff() {
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
            state: lash::SessionReadView::from_exported_state(&state),
            mode_turn_options: lash::ModeTurnOptions::default(),
        };

        let contributions = budget_prompt_contributions(&ctx, Some(100_000));

        assert_eq!(contributions.len(), 1);
        let content = &contributions[0].content;
        assert!(content.contains("Tokens: 120292 · handoff threshold: 100000 (120%)"));
        assert!(content.contains("Past the handoff threshold"));
        assert!(content.contains("Do not continue ordinary work"));
        assert!(content.contains("`continue_as` now"));
        assert!(content.contains("`task` + `seed`"));
    }

    #[test]
    fn soft_budget_warning_emits_surface_event_not_user_message() {
        let session = RlmModeSession::new(RlmModePluginConfig {
            continue_as_soft_warn_tokens: Some(100_000),
            ..Default::default()
        })
        .expect("rlm mode session");
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
                state: lash::SessionReadView::from_exported_state(&state),
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
            state: lash::SessionReadView::from_exported_state(&state),
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
            state: lash::SessionReadView::from_exported_state(&state),
            mode_turn_options: lash::ModeTurnOptions::default(),
        };

        let contributions = budget_prompt_contributions(&ctx, Some(200_000));

        assert!(contributions.is_empty());
    }
}
