use std::sync::{Arc, Mutex};

use lash::plugin::{
    CheckpointHookContext, ModeProtocolDriverPlugin, ModeRuntimeContext, ModeSessionContext,
    ModeSessionPlugin, PluginDirective, PluginError, PluginFactory, PluginRegistrar,
    PluginSessionContext, SessionPlugin,
};
use lash::tools::DiscoveryToolsProvider;
use lash::{
    CheckpointKind, ExecutionMode, ModeBuildInput, ModePreamble, PromptContribution, SessionError,
    ToolResultProjectionPluginConfig,
};
use lash_rlm_types::{RlmGlobalsPatchPluginBody, RlmModeEvent, RlmpureCreateExtras};

use lash_mode_rlm::{BoundVariablesCache, budget_prompt_contributions};

use crate::driver::{RlmpureProjectorConfig, build_rlmpure_preamble};
use crate::stream_mask;

const BUDGET_WARNING_EVENT: &str = "rlmpure_context_budget_warning";

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct RlmpureModePluginConfig {
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

impl Default for RlmpureModePluginConfig {
    fn default() -> Self {
        Self {
            observe_projection: ToolResultProjectionPluginConfig::default(),
            max_output_chars: default_max_output_chars(),
            baton_soft_warn_tokens: default_baton_soft_warn_tokens(),
        }
    }
}

pub struct BuiltinRlmpureModePluginFactory {
    config: RlmpureModePluginConfig,
}

impl BuiltinRlmpureModePluginFactory {
    pub fn new(config: RlmpureModePluginConfig) -> Self {
        Self { config }
    }
}

impl Default for BuiltinRlmpureModePluginFactory {
    fn default() -> Self {
        Self::new(RlmpureModePluginConfig::default())
    }
}

impl PluginFactory for BuiltinRlmpureModePluginFactory {
    fn id(&self) -> &'static str {
        "mode_rlmpure"
    }

    fn build(&self, ctx: &PluginSessionContext) -> Result<Arc<dyn SessionPlugin>, PluginError> {
        Ok(Arc::new(RlmpureModePlugin {
            active: ctx.execution_mode == ExecutionMode::new("rlmpure"),
            provider: Arc::new(DiscoveryToolsProvider::new()),
            config: self.config.clone(),
        }))
    }
}

struct RlmpureModePlugin {
    active: bool,
    provider: Arc<DiscoveryToolsProvider>,
    config: RlmpureModePluginConfig,
}

impl SessionPlugin for RlmpureModePlugin {
    fn id(&self) -> &'static str {
        "mode_rlmpure"
    }

    fn register(&self, reg: &mut PluginRegistrar) -> Result<(), PluginError> {
        if !self.active {
            return Ok(());
        }
        let mode_session = Arc::new(RlmpureModeSession::new(self.config.clone()));
        reg.mode().session(mode_session.clone())?;
        reg.mode().protocol_driver(Arc::new(RlmpureProtocolDriver {
            config: self.config.clone(),
        }))?;
        reg.tools()
            .provider(Arc::clone(&self.provider) as Arc<dyn lash::ToolProvider>)?;

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

struct RlmpureProtocolDriver {
    config: RlmpureModePluginConfig,
}

impl ModeProtocolDriverPlugin for RlmpureProtocolDriver {
    fn mode_id(&self) -> &str {
        "rlmpure"
    }

    fn build_preamble(&self, input: ModeBuildInput) -> ModePreamble {
        build_rlmpure_preamble(
            input,
            RlmpureProjectorConfig {
                max_output_chars: self.config.max_output_chars,
            },
        )
    }
}

fn print_output_prompt_contribution() -> PromptContribution {
    PromptContribution::execution(
        "Print Output",
        "`print` output is capped. If you see truncation, print narrower slices or specific fields instead of dumping the whole value.",
    )
}

struct RlmpureModeSession {
    config: RlmpureModePluginConfig,
    user_input_count: Mutex<usize>,
    warned_at_threshold: Mutex<bool>,
}

impl RlmpureModeSession {
    fn new(config: RlmpureModePluginConfig) -> Self {
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
            .map_err(|_| PluginError::Session("rlmpure soft-warning state poisoned".to_string()))?;
        if *warned {
            return Ok(Vec::new());
        }
        *warned = true;
        Ok(vec![PluginDirective::emit_events(vec![
            lash::PluginSurfaceEvent::Custom {
                name: BUDGET_WARNING_EVENT.to_string(),
                payload: serde_json::json!({
                    "used": used,
                    "threshold": threshold,
                }),
            },
        ])])
    }
}

#[async_trait::async_trait]
impl ModeSessionPlugin for RlmpureModeSession {
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
                    Some(RlmModeEvent::RlmGlobalsPatch(patch)) => Some(patch),
                    _ => None,
                },
                _ => None,
            })
        {
            ctx.apply_mode_globals_patch(&patch).await?;
        }
        let active_events = state.session_graph.active_events();
        let (patch, count) = user_input_patch_from_events(active_events.iter(), 0);
        *self
            .user_input_count
            .lock()
            .expect("rlmpure user input count lock") = count;
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
            .expect("rlmpure user input count lock");
        let (user_patch, next_count) = user_input_patch_from_nodes(nodes, start_index);
        ctx.apply_mode_globals_patch(&user_patch).await?;
        *self
            .user_input_count
            .lock()
            .expect("rlmpure user input count lock") = next_count;

        for node in nodes {
            if let lash::SessionAppendNode::Event {
                event: lash::SessionEventRecord::Mode(event),
            } = node
                && let Some(RlmModeEvent::RlmGlobalsPatch(patch)) = event.rlm_event()
            {
                ctx.apply_mode_globals_patch(&patch).await?;
            }
        }
        Ok(())
    }

    fn configure_runtime_from_request(
        &self,
        mut ctx: ModeRuntimeContext<'_>,
        request: &lash::SessionCreateRequest,
    ) {
        if let Ok(Some(RlmpureCreateExtras { termination })) =
            request
                .mode_extras
                .decode::<RlmpureCreateExtras>(&ExecutionMode::new("rlmpure"))
        {
            ctx.set_rlm_termination_mode(termination);
        }
    }
}

fn user_input_patch_from_nodes(
    nodes: &[lash::SessionAppendNode],
    start_index: usize,
) -> (RlmGlobalsPatchPluginBody, usize) {
    let mut next_index = start_index;
    let mut patch = RlmGlobalsPatchPluginBody::default();
    for node in nodes {
        let text = match node {
            lash::SessionAppendNode::Event {
                event: lash::SessionEventRecord::Conversation(record),
            } if should_bind_conversation_user_input(record) => conversation_text(record),
            lash::SessionAppendNode::Message { message }
                if should_bind_plugin_message_user_input(message) =>
            {
                plugin_message_text(message)
            }
            _ => None,
        };
        let Some(text) = text else {
            continue;
        };
        next_index += 1;
        patch.set.insert(
            format!("user_input_{next_index}"),
            serde_json::Value::String(text),
        );
    }
    (patch, next_index)
}

fn user_input_patch_from_events<'a>(
    events: impl IntoIterator<Item = &'a lash::SessionEventRecord>,
    start_index: usize,
) -> (RlmGlobalsPatchPluginBody, usize) {
    let mut next_index = start_index;
    let mut patch = RlmGlobalsPatchPluginBody::default();
    for event in events {
        let lash::SessionEventRecord::Conversation(record) = event else {
            continue;
        };
        if !should_bind_conversation_user_input(record) {
            continue;
        }
        let Some(text) = conversation_text(record) else {
            continue;
        };
        next_index += 1;
        patch.set.insert(
            format!("user_input_{next_index}"),
            serde_json::Value::String(text),
        );
    }
    (patch, next_index)
}

fn should_bind_conversation_user_input(record: &lash::ConversationRecord) -> bool {
    record.role == lash::MessageRole::User
        && !matches!(
            record.origin.as_ref(),
            Some(lash::MessageOrigin::Plugin { .. })
        )
}

fn should_bind_plugin_message_user_input(_message: &lash::PluginMessage) -> bool {
    false
}

fn conversation_text(record: &lash::ConversationRecord) -> Option<String> {
    if let Some(user_input) = &record.user_input
        && !user_input.effective_text.trim().is_empty()
    {
        return Some(user_input.effective_text.clone());
    }
    let chunks = record
        .parts
        .iter()
        .filter(|part| matches!(part.kind, lash::PartKind::Text | lash::PartKind::Prose))
        .map(|part| part.content.trim())
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>();
    (!chunks.is_empty()).then(|| chunks.join("\n\n"))
}

fn plugin_message_text(message: &lash::PluginMessage) -> Option<String> {
    if let Some(user_input) = &message.user_input
        && !user_input.effective_text.trim().is_empty()
    {
        return Some(user_input.effective_text.clone());
    }
    if !message.content.trim().is_empty() {
        return Some(message.content.clone());
    }
    let chunks = message
        .parts
        .iter()
        .filter(|part| matches!(part.kind, lash::PartKind::Text | lash::PartKind::Prose))
        .map(|part| part.content.trim())
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>();
    (!chunks.is_empty()).then(|| chunks.join("\n\n"))
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
            plugin_id: "mode_rlmpure".to_string(),
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
        assert!(content.contains("Choose exactly one:"));
        assert!(content.contains("1. Hand off now"));
        assert!(content.contains("2. If completion is one small bounded step away"));
        assert!(content.contains("3. If required state is not captured yet"));
        assert!(!content.contains("Otherwise"));
        assert!(!content.contains("otherwise"));
    }

    #[test]
    fn soft_budget_warning_emits_surface_event_not_user_message() {
        let session = RlmpureModeSession::new(RlmpureModePluginConfig {
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
        let lash::PluginSurfaceEvent::Custom { name, payload } = &events[0] else {
            panic!("budget warning should use a custom surface event");
        };
        assert_eq!(name, BUDGET_WARNING_EVENT);
        assert_eq!(payload["used"], serde_json::json!(120_292));
        assert_eq!(payload["threshold"], serde_json::json!(100_000));
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
