use std::sync::{Arc, Mutex};

use lash::plugin::{
    ModeProtocolDriverPlugin, ModeRuntimeContext, ModeSessionContext, ModeSessionPlugin,
    PluginError, PluginFactory, PluginRegistrar, PluginSessionContext, SessionPlugin,
};
use lash::tools::DiscoveryToolsProvider;
use lash::{
    ExecutionMode, ModeBuildInput, ModePreamble, PromptContribution, SessionError,
    ToolResultProjectionPluginConfig,
};
use lash_rlm_types::{RlmGlobalsPatchPluginBody, RlmModeEvent, RlmpureCreateExtras};

use crate::driver::{RlmpureProjectorConfig, build_rlmpure_preamble};
use crate::rlm_support::bound_variables_prompt_contributions;
use crate::stream_mask;

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct RlmpureModePluginConfig {
    pub observe_projection: ToolResultProjectionPluginConfig,
    #[serde(default = "default_max_output_chars")]
    pub max_output_chars: usize,
}

fn default_max_output_chars() -> usize {
    10_000
}

impl Default for RlmpureModePluginConfig {
    fn default() -> Self {
        Self {
            observe_projection: ToolResultProjectionPluginConfig::default(),
            max_output_chars: default_max_output_chars(),
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
        reg.mode()
            .session(Arc::new(RlmpureModeSession::new(self.config.clone())))?;
        reg.mode().protocol_driver(Arc::new(RlmpureProtocolDriver {
            config: self.config.clone(),
        }))?;
        reg.tools()
            .provider(Arc::clone(&self.provider) as Arc<dyn lash::ToolProvider>)?;

        let bound_vars_hook: lash::plugin::PromptContributor = Arc::new(move |ctx| {
            Box::pin(async move { Ok(bound_variables_prompt_contributions(&ctx)) })
        });
        reg.prompt().contribute(bound_vars_hook);

        let print_output_hook: lash::plugin::PromptContributor = Arc::new(move |_ctx| {
            Box::pin(async move { Ok(vec![print_output_prompt_contribution()]) })
        });
        reg.prompt().contribute(print_output_hook);

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
        "`print` output is capped before reinjection into the REPL trajectory. If you see truncation, print narrower slices or specific fields instead of dumping the whole value.",
    )
}

struct RlmpureModeSession {
    config: RlmpureModePluginConfig,
    user_input_count: Mutex<usize>,
}

impl RlmpureModeSession {
    fn new(config: RlmpureModePluginConfig) -> Self {
        Self {
            config,
            user_input_count: Mutex::new(0),
        }
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
            match node {
                lash::SessionAppendNode::Event {
                    event: lash::SessionEventRecord::Mode(event),
                } => {
                    if let Some(RlmModeEvent::RlmGlobalsPatch(patch)) = event.rlm_event() {
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
            } if record.role == lash::MessageRole::User => conversation_text(record),
            lash::SessionAppendNode::Message { message }
                if message.role == lash::MessageRole::User =>
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
        if record.role != lash::MessageRole::User {
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

fn conversation_text(record: &lash::ConversationRecord) -> Option<String> {
    if let Some(user_input) = &record.user_input {
        if !user_input.effective_text.trim().is_empty() {
            return Some(user_input.effective_text.clone());
        }
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
    if let Some(user_input) = &message.user_input {
        if !user_input.effective_text.trim().is_empty() {
            return Some(user_input.effective_text.clone());
        }
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
            }],
            user_input: None,
            origin: None,
        })
    }

    #[test]
    fn user_input_patch_from_events_binds_messages_in_order() {
        let events = vec![
            user_event("u1", "first"),
            lash::SessionEventRecord::Conversation(lash::ConversationRecord {
                id: "a1".to_string(),
                role: lash::MessageRole::Assistant,
                parts: Vec::new(),
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
        let nodes = vec![lash::SessionAppendNode::Message {
            message: lash::PluginMessage::text(lash::MessageRole::User, "third"),
        }];
        let (patch, count) = user_input_patch_from_nodes(&nodes, 2);

        assert_eq!(count, 3);
        assert_eq!(patch.set["user_input_3"], serde_json::json!("third"));
    }
}
