use std::collections::{BTreeSet, HashSet};

use crate::plugin::{ToolSurfaceContribution, ToolSurfaceOverride};
use crate::provider::AgentModels;
use crate::{
    AgentConfig, PromptContribution, SessionConfigOverrides, SessionCreateRequest,
    SessionStartPoint, ToolDefinition, ToolParam,
};

#[cfg(test)]
use crate::{PluginSession, plugin::PluginError};

use super::AgentCall;

/// Intelligence tier determines model choice, tool access, and turn limits.
pub(super) enum Tier {
    Low,
    Medium,
    High,
}

impl Tier {
    pub(super) fn from_str(s: &str) -> Option<Self> {
        match s {
            "low" => Some(Tier::Low),
            "medium" => Some(Tier::Medium),
            "high" => Some(Tier::High),
            _ => None,
        }
    }

    fn as_str(&self) -> &'static str {
        match self {
            Tier::Low => "low",
            Tier::Medium => "medium",
            Tier::High => "high",
        }
    }
}

pub(super) fn pick_model_and_variant(
    config: &AgentConfig,
    models: &Option<AgentModels>,
    tier: &Tier,
) -> (String, Option<String>) {
    if let Some(m) = models {
        match tier {
            Tier::Low => {
                if let Some(ref q) = m.low {
                    let variant = preferred_override_variant(&config.provider, q, tier);
                    return (q.clone(), variant);
                }
            }
            Tier::Medium => {
                if let Some(ref b) = m.medium {
                    let variant = preferred_override_variant(&config.provider, b, tier);
                    return (b.clone(), variant);
                }
            }
            Tier::High => {
                if let Some(ref t) = m.high {
                    let variant = preferred_override_variant(&config.provider, t, tier);
                    return (t.clone(), variant);
                }
            }
        }
    }

    if let Some((model, variant)) = config.provider.default_agent_model(tier.as_str()) {
        return (model.to_string(), variant.map(str::to_string));
    }

    let model = config.model.clone();
    let variant = config
        .provider
        .default_model_variant(&model)
        .map(str::to_string)
        .or_else(|| config.model_variant.clone());
    (model, variant)
}

fn preferred_override_variant(
    provider: &crate::Provider,
    model: &str,
    tier: &Tier,
) -> Option<String> {
    let tier_variant = tier.as_str();
    if provider.supported_variants(model).contains(&tier_variant) {
        return Some(tier_variant.to_string());
    }
    provider.default_model_variant(model).map(str::to_string)
}

fn low_tier_denied_tools() -> HashSet<&'static str> {
    [
        "apply_patch",
        "agent_call",
        "agent_result",
        "agent_kill",
        "ask",
    ]
    .into_iter()
    .collect()
}

impl AgentCall {
    pub(super) fn build_agent_config(&self, tier: &Tier) -> AgentConfig {
        let (model, model_variant) = pick_model_and_variant(&self.config, &self.agent_models, tier);
        AgentConfig {
            model,
            model_variant,
            session_id: self.config.session_id.clone(),
            provider: self.config.provider.clone(),
            sub_agent: true,
            include_soul: matches!(tier, Tier::High),
            max_context_tokens: self.config.max_context_tokens,
            max_turns: None,
            llm_log_path: self.config.llm_log_path.clone(),
            prompt_overrides: self.config.prompt_overrides.clone(),
            prompt_renderer: std::sync::Arc::clone(&self.config.prompt_renderer),
            instruction_source: self.config.instruction_source.clone(),
            execution_mode: match tier {
                Tier::Low => self.tool_config.low_tier_execution_mode,
                Tier::Medium | Tier::High => self.config.execution_mode,
            },
            ..Default::default()
        }
    }

    pub(super) fn tool_surface_for_tier(&self, tier: &Tier) -> ToolSurfaceContribution {
        let denied = match tier {
            Tier::Low => low_tier_denied_tools()
                .into_iter()
                .map(str::to_string)
                .collect::<BTreeSet<_>>(),
            Tier::Medium | Tier::High => BTreeSet::from(["ask".to_string()]),
        };
        ToolSurfaceContribution {
            overrides: denied
                .into_iter()
                .map(|tool_name| ToolSurfaceOverride {
                    tool_name,
                    enabled: Some(false),
                    injected: Some(false),
                })
                .collect(),
            tool_list_notes: Vec::new(),
        }
    }

    #[cfg(test)]
    pub(super) fn session_plugins_for_tier(
        &self,
        agent_id: &str,
        tier: &Tier,
    ) -> Result<std::sync::Arc<PluginSession>, PluginError> {
        let execution_mode = self.build_agent_config(tier).execution_mode;
        self.plugins.fork_for_agent_with_tool_surface(
            agent_id,
            execution_mode,
            self.tool_surface_for_tier(tier),
        )
    }

    #[cfg(test)]
    pub(super) fn visible_tool_names_for_tier(
        &self,
        tier: &Tier,
    ) -> Result<Vec<String>, PluginError> {
        let session = self.session_plugins_for_tier("__tier_probe__", tier)?;
        let surface = session.execution_surface(
            session.agent_id(),
            self.build_agent_config(tier).execution_mode,
        );
        Ok(surface
            .enabled_tools()
            .into_iter()
            .map(|tool| tool.name)
            .collect())
    }

    pub(super) fn build_create_request(
        &self,
        agent_id: String,
        tier: &Tier,
    ) -> SessionCreateRequest {
        let agent_config = self.build_agent_config(tier);
        SessionCreateRequest {
            agent_id: Some(agent_id),
            start: SessionStartPoint::Empty,
            config_overrides: SessionConfigOverrides {
                model: Some(agent_config.model),
                model_variant: agent_config.model_variant,
                max_context_tokens: agent_config.max_context_tokens,
                execution_mode: Some(agent_config.execution_mode),
                context_folding: Some(agent_config.context_folding),
                session_id: agent_config.session_id,
                max_turns: agent_config.max_turns,
                include_soul: Some(agent_config.include_soul),
                sub_agent: Some(true),
            },
            tool_surface: self.tool_surface_for_tier(tier),
            initial_messages: Vec::new(),
        }
    }
}

pub(super) fn agent_call_prompt_contributions() -> Vec<PromptContribution> {
    vec![
        PromptContribution::guidance(
            "### Delegation\nUse `agent_call` for scoped sub-tasks. Each delegate runs in its own session. Prefer low-intelligence delegates for read-only lookup or summarization work, and avoid overlapping file edits across concurrent delegates.",
        ),
        PromptContribution::guidance(
            "### Agent Lifecycle\n`agent_result(id)` blocks until the child session finishes and returns an object in `result.value` with fields like `result`, `context`, and `_sub_agent` (including session/config metadata). The agent ID remains valid afterwards, including after `agent_kill(id)`, so you can query the terminal result again even after the child session has been stopped.",
        ),
    ]
}

pub(super) fn agent_call_definitions(
    execution_mode: crate::ExecutionMode,
    low_tier_execution_mode: crate::ExecutionMode,
) -> Vec<ToolDefinition> {
    let low_tier_summary = match low_tier_execution_mode {
        crate::ExecutionMode::Standard => "by default, low runs in standard mode",
        crate::ExecutionMode::Repl => "in this session, low runs in repl mode",
    };
    let (agent_call_description, agent_call_examples) = match execution_mode {
        crate::ExecutionMode::Repl => (
            format!(
                "Spawn a child session for scoped work and return a handle. In REPL mode, use `call agent_result {{ id: handle.value.id }}` or `call agent_kill {{ id: handle.value.id }}` with the returned id. Use `intelligence=\"low\"` for fast read-only work; {}. Medium/high inherit the parent execution mode.",
                low_tier_summary
            ),
            vec![
                r#"handle = call agent_call { prompt: "Summarize the auth flow", intelligence: "low" }"#.into(),
                r#"result = call agent_result { id: handle.value.id }"#.into(),
            ],
        ),
        crate::ExecutionMode::Standard => (
            format!(
                "Spawn a child session for scoped work and return a handle. Use `agent_result(id)` or `agent_kill(id)` with the returned id. Use `intelligence=\"low\"` for fast read-only work; {}. Medium/high inherit the parent execution mode.",
                low_tier_summary
            ),
            vec![
                "handle = agent_call(prompt=\"Summarize the auth flow\", intelligence=\"low\")"
                    .into(),
            ],
        ),
    };
    vec![
        ToolDefinition {
            name: "agent_call".into(),
            description: agent_call_description,
            params: vec![
                ToolParam::typed("prompt", "str"),
                ToolParam::typed("intelligence", "str"),
                ToolParam {
                    name: "schema".into(),
                    r#type: "str".into(),
                    description: "JSON schema to include in the agent's prompt as output guidance (not enforced at runtime)".into(),
                    default_value: None,
                    required: false,
                },
            ],
            returns: "dict".into(),
            examples: agent_call_examples,
            enabled: true,
            injected: true,
        },
        ToolDefinition {
            name: "agent_result".into(),
            description: "Wait for a child session to finish and return its final result.".into(),
            params: vec![
                ToolParam::typed("id", "str"),
                ToolParam::optional("timeout", "float"),
            ],
            returns: "dict".into(),
            examples: vec![],
            enabled: true,
            injected: true,
        },
        ToolDefinition {
            name: "agent_kill".into(),
            description:
                "Cancel a running child session while leaving its terminal result queryable via `agent_result`.".into(),
            params: vec![ToolParam::typed("id", "str")],
            returns: "None".into(),
            examples: vec![],
            enabled: true,
            injected: true,
        },
    ]
}
