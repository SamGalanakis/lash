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
    ["apply_patch", "delegate", "ask"].into_iter().collect()
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
            prompt_overrides: Vec::new(),
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
            initial_nodes: Vec::new(),
            context_surface: self.tier_context_surface(tier),
            mode_extras: lash::ModeExtras::default(),
            usage_source: Some("delegate".to_string()),
        }
    }
}

pub(super) fn delegate_prompt_contributions() -> Vec<PromptContribution> {
    vec![
        PromptContribution::guidance(
            "delegation",
            "Delegation",
            "Use `delegate` proactively for scoped, self-contained sub-tasks when it will make concrete progress without blocking your next local step. Treat delegation as sidecar work, not a handoff of the immediate critical path. Before delegating, identify what you can do locally right now and what can run in parallel.\n\nDelegation rules:\n- Do not duplicate work between the main agent and child sessions. Once a child owns a trace, question, or validation pass, trust it and use your local effort on non-overlapping work until the result is needed.\n- Do not delegate the next blocking step in a single-threaded workflow. If your very next step depends on the result and you have no meaningful parallel work, do it yourself instead.\n- Keep delegated asks concrete, well-bounded, and self-contained.\n- Avoid overlapping file edits across concurrent child sessions.\n- In RLM mode, use `start call delegate { ... }` when the child can run in the background, then `await handle` only when the result is needed.\n\nChoose intelligence by task shape:\n\n- `low`: fast, read-only exploration and synthesis. Use for codebase discovery, tracing behavior, finding examples, summarizing logs or failures, scanning docs, searching history, comparing implementations, or other informational sidecar work.\n- `medium`: bounded implementation or analysis with a contained scope. Use for small features, targeted bug fixes, focused tests, contained refactors, single-module edits, or validating one concrete hypothesis.\n- `high`: peer-level independent work with a clearly separate line of ownership. Use it for substantial parallel tasks, larger isolated implementations, strong validation passes, or serious design investigation when the write scope or responsibility boundary is distinct.",
        ),
        PromptContribution::guidance(
            "delegate_lifecycle",
            "Delegate Lifecycle",
            "Plain `delegate(...)` waits for the child and returns its terminal result. In RLM mode you can make delegation asynchronous with lashlang itself: `handle = start call delegate { ... }`, then later `result = await handle` or `cancel handle`. Cancellation is best-effort and asks the child turn to stop.",
        ),
    ]
}

pub(super) fn delegate_tool_definitions(
    execution_mode: lash::ExecutionMode,
    low_tier_execution_mode: lash::ExecutionMode,
) -> Vec<ToolDefinition> {
    let low_tier_summary = match low_tier_execution_mode {
        lash::ExecutionMode::Standard => "by default, low runs in standard mode",
        lash::ExecutionMode::Rlm => "in this session, low runs in rlm mode",
    };
    let (delegate_description, delegate_examples) = match execution_mode {
        lash::ExecutionMode::Rlm => (
            format!(
                "Run a scoped child session and return its terminal result. Choose `intelligence` based on the delegation guidance above. In lashlang, use `start call delegate {{ ... }}` when the child should run in the background, then `await handle` when the result is needed; {}. If `output` is provided, the child runs in RLM mode and must terminate with `finish {{ ... }}` matching the declared shape.",
                low_tier_summary
            ),
            vec![
                r#"handle = start call delegate { task: "Summarize the auth flow", intelligence: "low" }"#.into(),
                r#"result = await handle"#.into(),
                r#"typed = call delegate { task: "Extract the longest line", intelligence: "low", vars: { path: "src/main.rs" }, output: { line: "str", length: "int" } }"#.into(),
            ],
        ),
        lash::ExecutionMode::Standard => (
            format!(
                "Run a scoped child session and wait for its terminal result. Choose `intelligence` based on the delegation guidance above; {}. If `output` is provided, the child runs in RLM mode and must terminate with `finish {{ ... }}` matching the declared shape.",
                low_tier_summary
            ),
            vec![
                "result = delegate(task=\"Summarize the auth flow\", intelligence=\"low\")".into(),
                "typed = delegate(task=\"Extract the longest line\", intelligence=\"low\", vars={\"path\": \"src/main.rs\"}, output={\"line\": \"str\", \"length\": \"int\"})".into(),
            ],
        ),
    };

    vec![
        ToolDefinition {
            name: "delegate".into(),
            description: delegate_description,
            params: vec![
                ToolParam::typed("task", "str"),
                ToolParam::typed("intelligence", "str"),
                ToolParam {
                    name: "vars".into(),
                    r#type: "dict".into(),
                    description: "Optional named inputs forwarded to the child session.".into(),
                    default_value: None,
                    required: false,
                },
                ToolParam {
                    name: "output".into(),
                    r#type: "dict".into(),
                    description: "Optional declarative output schema. A record whose keys are field names and values are type descriptors: \"str\", \"int\", \"float\", \"bool\", \"list[<scalar>]\", or \"record\". When present, the child runs in RLM mode and must end with `finish { ... }` matching this shape.".into(),
                    default_value: None,
                    required: false,
                },
            ],
            returns: "dict".into(),
            examples: delegate_examples,
            enabled: true,
            injected: true,
            input_schema_override: None,
            output_schema_override: None,
        },
    ]
}
