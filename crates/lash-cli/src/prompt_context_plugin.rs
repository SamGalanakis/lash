use std::sync::Arc;

use chrono::Utc;

use lash_core::PreparedContext;
use lash_core::PromptContribution;
use lash_core::plugin::{
    HistoryError, PluginDirective, PluginError, PluginFactory, PluginRegistrar,
    PluginSessionContext, TurnContextTransform, TurnTransformContext,
};
use lash_core::{
    ExecutionMode, Message, MessageRole, Part, PartKind, PluginMessage, PruneState, shared_parts,
};

/// Host-provided source for project instructions.
///
/// This is plugin-owned prompt policy, not a core runtime concept. The
/// `lash-cli` host wires an [`InstructionSource`] implementation when
/// installing the prompt-context plugin.
pub trait InstructionSource: Send + Sync {
    fn system_instructions(&self) -> String;
    fn context_instructions_for_reads(&self, read_paths: &[String]) -> String;
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PromptContextPluginConfig {
    pub include_environment: bool,
    pub include_project_instructions: bool,
}

impl Default for PromptContextPluginConfig {
    fn default() -> Self {
        Self {
            include_environment: true,
            include_project_instructions: true,
        }
    }
}

pub struct PromptContextPluginFactory {
    instruction_source: Arc<dyn InstructionSource>,
    config: PromptContextPluginConfig,
}

impl PromptContextPluginFactory {
    pub fn new(
        instruction_source: Arc<dyn InstructionSource>,
        config: PromptContextPluginConfig,
    ) -> Self {
        Self {
            instruction_source,
            config,
        }
    }
}

impl PluginFactory for PromptContextPluginFactory {
    fn id(&self) -> &'static str {
        "prompt_context"
    }

    fn build(
        &self,
        ctx: &PluginSessionContext,
    ) -> Result<Arc<dyn lash_core::SessionPlugin>, PluginError> {
        Ok(Arc::new(PromptContextPlugin {
            instruction_source: Arc::clone(&self.instruction_source),
            config: self.config.clone(),
            execution_mode: ctx.execution_mode.clone(),
        }))
    }
}

struct PromptContextPlugin {
    instruction_source: Arc<dyn InstructionSource>,
    config: PromptContextPluginConfig,
    execution_mode: ExecutionMode,
}

impl lash_core::SessionPlugin for PromptContextPlugin {
    fn id(&self) -> &'static str {
        "prompt_context"
    }

    fn register(&self, reg: &mut PluginRegistrar) -> Result<(), PluginError> {
        let instruction_source = Arc::clone(&self.instruction_source);
        let include_project_instructions = self.config.include_project_instructions;
        reg.prompt().contribute(Arc::new(move |_ctx| {
            let instruction_source = Arc::clone(&instruction_source);
            Box::pin(async move {
                let mut contributions = Vec::new();
                if include_project_instructions {
                    let project_instructions = instruction_source.system_instructions();
                    if !project_instructions.trim().is_empty() {
                        contributions.push(PromptContribution::project_instructions(
                            project_instructions,
                        ));
                    }
                }
                Ok(contributions)
            })
        }));
        let instruction_source = Arc::clone(&self.instruction_source);
        reg.tool_calls().after(Arc::new(move |ctx| {
            let instruction_source = Arc::clone(&instruction_source);
            Box::pin(async move {
                if !ctx.result.is_success() || ctx.tool_name != "read_file" {
                    return Ok(Vec::new());
                }
                let Some(path) = ctx.args.get("path").and_then(|value| value.as_str()) else {
                    return Ok(Vec::new());
                };
                if path.is_empty() {
                    return Ok(Vec::new());
                }
                let instructions =
                    instruction_source.context_instructions_for_reads(&[path.to_string()]);
                if instructions.trim().is_empty() {
                    return Ok(Vec::new());
                }
                Ok(vec![PluginDirective::EnqueueMessages {
                    messages: vec![PluginMessage::text(MessageRole::System, instructions)],
                }])
            })
        }));

        if self.config.include_environment && self.execution_mode == ExecutionMode::standard() {
            reg.history()
                .prepare_turn(50, Arc::new(EnvironmentTailTransform));
        }
        Ok(())
    }
}

struct EnvironmentTailTransform;

#[async_trait::async_trait]
impl TurnContextTransform for EnvironmentTailTransform {
    fn id(&self) -> &'static str {
        "prompt_context.environment_tail"
    }

    async fn transform(
        &self,
        _ctx: &TurnTransformContext,
        input: PreparedContext,
    ) -> Result<PreparedContext, HistoryError> {
        let context = build_prompt_environment_context();
        if context.trim().is_empty() {
            return Ok(input);
        }

        let mut messages: Vec<Message> = input.messages.as_slice().to_vec();
        messages.push(environment_tail_message(context));

        let base = Arc::new(messages);
        let cache = Arc::new(lash_core::BaseRenderCache::new());
        Ok(PreparedContext {
            messages: lash_core::MessageSequence::from_base(base).with_base_render_cache(cache),
            ..input
        })
    }
}

fn environment_tail_message(content: String) -> Message {
    let id = "prompt-context-env";
    Message {
        id: id.to_string(),
        role: MessageRole::User,
        parts: shared_parts(vec![Part {
            id: format!("{id}.p0"),
            kind: PartKind::Prose,
            content: format!("<system-reminder>\n{content}\n</system-reminder>"),
            attachment: None,
            tool_call_id: None,
            tool_name: None,
            tool_replay: None,
            prune_state: PruneState::Intact,
            reasoning_meta: None,
            response_meta: None,
        }]),
        origin: Some(lash_core::MessageOrigin::Plugin {
            plugin_id: "prompt_context".to_string(),
            transient: true,
        }),
    }
}

fn build_prompt_environment_context() -> String {
    let mut parts = Vec::new();
    let now = Utc::now();
    parts.push(format!("Current date (UTC): {}", now.format("%Y-%m-%d")));

    if let Ok(cwd) = std::env::current_dir() {
        parts.push(format!("Working directory: {}", cwd.display()));

        if cwd.join(".git").exists() {
            parts.push("Git repository: yes".to_string());
        }
    }

    parts.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use lash_core::plugin::PromptHookContext;
    use lash_core::testing::{MockSessionManager, mock_assembled_turn};
    use lash_core::{
        RuntimeSessionState, PluginHost, PromptSlot, SessionPolicy, SessionReadView,
        SessionSnapshot, SessionStateEnvelope,
    };

    struct StaticInstructionSource {
        text: String,
        read_text: String,
    }

    impl InstructionSource for StaticInstructionSource {
        fn system_instructions(&self) -> String {
            self.text.clone()
        }

        fn context_instructions_for_reads(&self, _read_paths: &[String]) -> String {
            self.read_text.clone()
        }
    }

    fn mock_snapshot(run_session_id: &str) -> SessionSnapshot {
        RuntimeSessionState::from_state(SessionStateEnvelope {
            session_id: "root".to_string(),
            policy: SessionPolicy {
                execution_mode: ExecutionMode::standard(),
                session_id: Some(run_session_id.to_string()),
                ..Default::default()
            },
            ..Default::default()
        })
    }

    fn mock_session_manager(run_session_id: &str) -> MockSessionManager {
        MockSessionManager::default()
            .with_snapshot(mock_snapshot(run_session_id))
            .with_turn(mock_assembled_turn(run_session_id, ""))
    }

    #[tokio::test]
    async fn prompt_context_plugin_contributes_environment_and_project_instruction_sections() {
        let mut factories = lash_core::testing::test_mode_factories();
        factories.push(Arc::new(PromptContextPluginFactory::new(
            Arc::new(StaticInstructionSource {
                text: "Repo rules".to_string(),
                read_text: String::new(),
            }),
            PromptContextPluginConfig::default(),
        )));
        let host = PluginHost::new(factories);
        let session = host.build_standard_session("root", None).expect("session");
        let contributions = session
            .collect_prompt_contributions(PromptHookContext {
                session_id: "root".to_string(),
                host: Arc::new(mock_session_manager("run-session")),
                state: SessionReadView::from_exported_state(&SessionStateEnvelope::default()),
                mode_turn_options: lash_core::ModeTurnOptions::default(),
                turn_context: lash_core::TurnContext::default(),
            })
            .await
            .expect("prompt contributions");

        assert!(
            !contributions
                .iter()
                .any(|contribution| contribution.slot == PromptSlot::RuntimeContext),
            "environment context should not appear as a runtime-context contribution",
        );
        assert!(contributions.iter().any(|contribution| {
            contribution.slot == PromptSlot::ProjectInstructions
                && contribution.title.as_deref() == Some("Project Instructions")
                && contribution.content.as_ref() == "Repo rules"
        }));
    }
}
