use std::sync::Arc;

use crate::PromptContribution;
use crate::agent::PromptSectionName;
use crate::instructions::InstructionSource;
use crate::plugin::{PluginError, PluginFactory, PluginRegistrar, PluginSessionContext};

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
        _ctx: &PluginSessionContext,
    ) -> Result<Arc<dyn crate::SessionPlugin>, PluginError> {
        Ok(Arc::new(PromptContextPlugin {
            instruction_source: Arc::clone(&self.instruction_source),
            config: self.config.clone(),
        }))
    }
}

struct PromptContextPlugin {
    instruction_source: Arc<dyn InstructionSource>,
    config: PromptContextPluginConfig,
}

impl crate::SessionPlugin for PromptContextPlugin {
    fn id(&self) -> &'static str {
        "prompt_context"
    }

    fn register(&self, reg: &mut PluginRegistrar) -> Result<(), PluginError> {
        let instruction_source = Arc::clone(&self.instruction_source);
        let config = self.config.clone();
        reg.prompt().contribute(Arc::new(move |_ctx| {
            let instruction_source = Arc::clone(&instruction_source);
            let config = config.clone();
            Box::pin(async move {
                let mut contributions = Vec::new();
                let base_context = build_prompt_environment_context();
                if config.include_environment && !base_context.trim().is_empty() {
                    contributions.push(PromptContribution {
                        section: PromptSectionName::Environment,
                        priority: 0,
                        content: base_context,
                    });
                }
                let project_instructions = instruction_source.system_instructions();
                if config.include_project_instructions && !project_instructions.trim().is_empty() {
                    contributions.push(PromptContribution {
                        section: PromptSectionName::Guidance,
                        priority: 0,
                        content: format!("### Project Instructions\n{}", project_instructions),
                    });
                }
                Ok(contributions)
            })
        }));
        Ok(())
    }
}

fn build_prompt_environment_context() -> String {
    let mut parts = Vec::new();

    if let Ok(cwd) = std::env::current_dir() {
        parts.push(format!("Working directory: {}", cwd.display()));

        if cwd.join(".git").exists() {
            parts.push("Git repository: yes".to_string());
        }

        if let Ok(entries) = std::fs::read_dir(&cwd) {
            let mut names: Vec<String> = entries
                .filter_map(|entry| entry.ok())
                .map(|entry| {
                    let name = entry.file_name().to_string_lossy().to_string();
                    if entry.file_type().map(|ty| ty.is_dir()).unwrap_or(false) {
                        format!("{name}/")
                    } else {
                        name
                    }
                })
                .filter(|name| !name.starts_with('.'))
                .collect();
            names.sort();
            if !names.is_empty() {
                parts.push(format!("Top-level entries: {}", names.join(", ")));
            }
        }
    }

    parts.push("REPL third-party packages: none".to_string());
    parts.join("\n")
}
