use std::sync::Arc;

use chrono::Utc;

use lash::PromptContribution;
use lash::instructions::InstructionSource;
use lash::plugin::{
    HistoryError, PluginError, PluginFactory, PluginRegistrar, PluginSessionContext,
    TurnContextTransform, TurnTransformContext,
};
use lash::session_model::context::PreparedContext;
use lash::{ExecutionMode, Message, MessageRole, Part, PartKind, PruneState, shared_parts};

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
    ) -> Result<Arc<dyn lash::SessionPlugin>, PluginError> {
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

impl lash::SessionPlugin for PromptContextPlugin {
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
        let cache = Arc::new(lash::BaseRenderCache::new());
        Ok(PreparedContext {
            messages: lash::MessageSequence::from_base(base).with_base_render_cache(cache),
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
            tool_item_id: None,
            tool_signature: None,
            prune_state: PruneState::Intact,
            reasoning_meta: None,
            response_meta: None,
        }]),
        user_input: None,
        origin: Some(lash::MessageOrigin::Plugin {
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
