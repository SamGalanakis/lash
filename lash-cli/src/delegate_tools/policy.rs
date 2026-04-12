use std::collections::HashSet;
use std::sync::Arc;

use lash::provider::AgentModels;
use lash::{
    PromptContribution, SessionContextSurface, SessionCreateRequest, SessionPluginMode,
    SessionPolicy, SessionStartPoint, ToolDefinition, ToolParam, ToolProvider,
};

use super::{DelegateTools, FilteredToolProvider};

pub(super) enum Tier {
    Low,
    Medium,
    High,
}

impl Tier {
    pub(super) fn from_str(value: &str) -> Option<Self> {
        match value {
            "low" => Some(Self::Low),
            "medium" => Some(Self::Medium),
            "high" => Some(Self::High),
            _ => None,
        }
    }

    fn as_str(&self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
        }
    }
}

pub(super) fn pick_model_and_variant(
    config: &SessionPolicy,
    models: &Option<AgentModels>,
    tier: &Tier,
) -> (String, Option<String>) {
    if let Some(models) = models {
        match tier {
            Tier::Low => {
                if let Some(model) = &models.low {
                    let variant = preferred_override_variant(&config.provider, model, tier);
                    return (model.clone(), variant);
                }
            }
            Tier::Medium => {
                if let Some(model) = &models.medium {
                    let variant = preferred_override_variant(&config.provider, model, tier);
                    return (model.clone(), variant);
                }
            }
            Tier::High => {
                if let Some(model) = &models.high {
                    let variant = preferred_override_variant(&config.provider, model, tier);
                    return (model.clone(), variant);
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
    provider: &lash::Provider,
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

fn medium_high_denied_tools() -> HashSet<&'static str> {
    ["ask"].into_iter().collect()
}

impl DelegateTools {
    pub(super) fn build_session_policy(&self, tier: &Tier) -> SessionPolicy {
        let (model, model_variant) = pick_model_and_variant(&self.policy, &self.agent_models, tier);
        SessionPolicy {
            model,
            model_variant,
            provider: self.policy.provider.clone(),
            max_context_tokens: self.policy.max_context_tokens,
            max_turns: None,
            execution_mode: match tier {
                Tier::Low => self.tool_config.low_tier_execution_mode,
                Tier::Medium | Tier::High => self.policy.execution_mode,
            },
            ..Default::default()
        }
    }

    fn tier_allowed_tools(&self, tier: &Tier) -> Vec<String> {
        let denied = match tier {
            Tier::Low => low_tier_denied_tools(),
            Tier::Medium | Tier::High => medium_high_denied_tools(),
        };
        self.base_tools
            .definitions()
            .into_iter()
            .map(|definition| definition.name)
            .filter(|name| !denied.contains(name.as_str()))
            .collect()
    }

    pub(super) fn tier_context_surface(&self, tier: &Tier) -> SessionContextSurface {
        SessionContextSurface {
            include_base_tools: false,
            tool_providers: vec![Arc::new(FilteredToolProvider::new(
                Arc::clone(&self.base_tools),
                self.tier_allowed_tools(tier),
            )) as Arc<dyn ToolProvider>],
            prompt_contributions: Vec::new(),
        }
    }

    #[cfg(test)]
    pub(super) fn visible_tool_names_for_tier(
        &self,
        tier: &Tier,
    ) -> Result<Vec<String>, lash::plugin::PluginError> {
        Ok(self
            .tier_context_surface(tier)
            .tool_providers
            .iter()
            .flat_map(|provider| provider.definitions())
            .map(|tool| tool.name)
            .collect())
    }

    pub(super) fn build_create_request(
        &self,
        session_id: String,
        parent_session_id: String,
        tier: &Tier,
    ) -> SessionCreateRequest {
        SessionCreateRequest {
            session_id: Some(session_id),
            parent_session_id: Some(parent_session_id),
            start: SessionStartPoint::Empty,
            policy: Some(self.build_session_policy(tier)),
            plugin_mode: SessionPluginMode::InheritCurrent,
            initial_messages: Vec::new(),
            context_surface: self.tier_context_surface(tier),
            mode_extras: lash::ModeExtras::default(),
        }
    }
}

pub(super) fn delegate_prompt_contributions() -> Vec<PromptContribution> {
    vec![
        PromptContribution::guidance(
            "delegation",
            "Delegation",
            "Use `agent_call` proactively for scoped, self-contained sub-tasks when it will make concrete progress without blocking your next local step. Treat delegation as sidecar work, not a handoff of the immediate critical path. Before delegating, identify what you can do locally right now and what can run in parallel.\n\nDelegation rules:\n- Do not duplicate work between the main agent and child sessions. Once a child owns a trace, question, or validation pass, trust it and use your local effort on non-overlapping work until the result is needed.\n- Do not delegate the next blocking step in a single-threaded workflow. If your very next step depends on the result and you have no meaningful parallel work, do it yourself instead.\n- Keep delegated asks concrete, well-bounded, and self-contained.\n- Avoid overlapping file edits across concurrent child sessions.\n- Call `agent_result` sparingly; wait only when the child result is needed to continue.\n\nChoose intelligence by task shape:\n\n- `low`: fast, read-only exploration and synthesis. Use for codebase discovery, tracing behavior, finding examples, summarizing logs or failures, scanning docs, searching history, comparing implementations, or other informational sidecar work.\n  Examples:\n  - \"Find where auth tokens are refreshed\"\n  - \"Summarize the config loader\"\n  - \"Scan the repo for queue-related paths\"\n  - \"Check the docs for the current API shape\"\n\n- `medium`: bounded implementation or analysis with a contained scope. Use for small features, targeted bug fixes, focused tests, contained refactors, single-module edits, or validating one concrete hypothesis.\n  Examples:\n  - \"Add tests for the retry helper\"\n  - \"Refactor this parser module without changing behavior\"\n  - \"Fix the null handling bug in this endpoint\"\n  - \"Implement this small CLI flag\"\n\n- `high`: peer-level independent work with a clearly separate line of ownership. Use it for substantial parallel tasks, larger isolated implementations, strong validation passes, or serious design investigation when the write scope or responsibility boundary is distinct.\n  Examples:\n  - \"Implement the backend half while I handle the UI\"\n  - \"Own the persistence changes while I update the command flow\"\n  - \"Review this design for race conditions and regressions\"\n  - \"Validate whether this architectural direction is sound\"",
        ),
        PromptContribution::guidance(
            "agent_lifecycle",
            "Agent Lifecycle",
            "`agent_result(id)` blocks until the child session finishes and returns an object in `result.value` with the child result and terminal status. The agent ID remains valid afterwards, including after `agent_kill(id)`, so you can query the terminal result again even after the child session has been stopped.",
        ),
    ]
}

pub(super) fn delegate_tool_definitions(
    execution_mode: lash::ExecutionMode,
    low_tier_execution_mode: lash::ExecutionMode,
) -> Vec<ToolDefinition> {
    let low_tier_summary = match low_tier_execution_mode {
        lash::ExecutionMode::Standard => "by default, low runs in standard mode",
        lash::ExecutionMode::Repl => "in this session, low runs in repl mode",
    };
    let (agent_call_description, agent_call_examples) = match execution_mode {
        lash::ExecutionMode::Repl => (
            format!(
                "Spawn a child session for a scoped sub-task and return a handle. Choose `intelligence` based on the delegation guidance above. In REPL mode, use `call agent_result {{ id: handle.value.id }}` or `call agent_kill {{ id: handle.value.id }}` with the returned id; {}. Medium/high inherit the parent execution mode.",
                low_tier_summary
            ),
            vec![
                r#"handle = call agent_call { prompt: "Summarize the auth flow", intelligence: "low" }"#.into(),
                r#"result = call agent_result { id: handle.value.id }"#.into(),
            ],
        ),
        lash::ExecutionMode::Standard => (
            format!(
                "Spawn a child session for a scoped sub-task and return a handle. Choose `intelligence` based on the delegation guidance above. Use `agent_result(id)` or `agent_kill(id)` with the returned id; {}. Medium/high inherit the parent execution mode.",
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
                    description: "JSON schema to include in the child session prompt as output guidance (not enforced at runtime)".into(),
                    default_value: None,
                    required: false,
                },
            ],
            returns: "dict".into(),
            examples: agent_call_examples,
            enabled: true,
            injected: true,
            input_schema_override: None,
            output_schema_override: None,
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
            input_schema_override: None,
            output_schema_override: None,
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
            input_schema_override: None,
            output_schema_override: None,
        },
        ToolDefinition {
            name: "predict".into(),
            description: predict_tool_description(execution_mode),
            params: vec![
                ToolParam::typed("task", "str"),
                ToolParam {
                    name: "vars".into(),
                    r#type: "dict".into(),
                    description: "Named inputs forwarded to the child session. Each key/value becomes a seeded variable in the child's lashlang state and is described in the child's initial prompt.".into(),
                    default_value: None,
                    required: false,
                },
                ToolParam {
                    name: "output".into(),
                    r#type: "dict".into(),
                    description: "Declarative output schema. A record whose keys are field names and values are type descriptors: \"str\", \"int\", \"float\", \"bool\", \"list[<scalar>]\", or \"record\". The child must terminate via `finish { ... }` with a value matching this shape.".into(),
                    default_value: None,
                    required: true,
                },
            ],
            returns: "dict".into(),
            examples: predict_tool_examples(execution_mode),
            enabled: true,
            injected: true,
            input_schema_override: None,
            output_schema_override: None,
        },
    ]
}

fn predict_tool_description(execution_mode: lash::ExecutionMode) -> String {
    let prefix = "Spawn a typed sub-session that runs the given task with seeded inputs and returns a record matching the declared `output` schema. The child runs in REPL mode and must terminate by calling `finish { ... }` with a value matching the schema; mismatches loop with a validation error so the child can retry.";
    match execution_mode {
        lash::ExecutionMode::Repl => format!(
            "{prefix}\n\nUse this for scoped extraction and perception sub-tasks where you want a known-shape result back instead of freeform prose.",
        ),
        lash::ExecutionMode::Standard => format!(
            "{prefix}\n\nUse this for scoped extraction and perception sub-tasks where you want a known-shape result back instead of freeform prose. The child session runs in REPL mode regardless of the parent's execution mode.",
        ),
    }
}

fn predict_tool_examples(execution_mode: lash::ExecutionMode) -> Vec<String> {
    match execution_mode {
        lash::ExecutionMode::Repl => vec![
            r#"r = call predict { task: "Extract the longest line and its length", vars: { path: "src/main.rs" }, output: { line: "str", length: "int" } }"#.into(),
            r#"observe r.line"#.into(),
        ],
        lash::ExecutionMode::Standard => vec![
            r#"r = predict(task="Extract the longest line and its length", vars={"path": "src/main.rs"}, output={"line": "str", "length": "int"})"#.into(),
        ],
    }
}
