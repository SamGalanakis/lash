use crate::support::*;

#[derive(Clone, Debug)]
pub(crate) struct OpenAiModelPolicy {
    pub(crate) base_url: String,
}

impl OpenAiModelPolicy {
    pub(crate) fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct OpenAiDirectModelPolicy;

pub(crate) fn clamp_reasoning_effort(model: &str, effort: &str) -> String {
    let id = model_id(model).to_ascii_lowercase();
    if (id.starts_with("gpt-5.2") || id.starts_with("gpt-5.3") || id.starts_with("gpt-5.4"))
        && effort == "minimal"
    {
        return "low".to_string();
    }
    effort.to_string()
}

impl ProviderModelPolicy for OpenAiModelPolicy {
    fn default_model(&self) -> &str {
        "anthropic/claude-sonnet-4.6"
    }

    fn supported_variants(&self, model: &str) -> &'static [&'static str] {
        if !base_url_is_openrouter(&self.base_url) {
            return &[];
        }
        let lower = model.to_ascii_lowercase();
        if lower.contains("gpt") || lower.contains("claude") || lower.contains("gemini-3") {
            OPENROUTER_REASONING_VARIANTS
        } else {
            &[]
        }
    }

    fn default_model_variant(&self, model: &str) -> Option<&'static str> {
        let variants = self.supported_variants(model);
        if variants.is_empty() {
            return None;
        }
        if model.to_ascii_lowercase().contains("gpt") {
            Some("medium")
        } else {
            Some("high")
        }
    }

    fn request_variant_config(&self, model: &str, variant: &str) -> Option<VariantRequestConfig> {
        if !self.supported_variants(model).contains(&variant) {
            return None;
        }
        Some(VariantRequestConfig::ReasoningEffort(variant.to_string()))
    }

    fn default_agent_model(&self, tier: &str) -> Option<AgentModelSelection> {
        match tier {
            "low" => Some(AgentModelSelection {
                model: "minimax/minimax-m2.5".to_string(),
                variant: None,
            }),
            "medium" => Some(AgentModelSelection {
                model: "z-ai/glm-5".to_string(),
                variant: None,
            }),
            "high" => Some(AgentModelSelection {
                model: "anthropic/claude-sonnet-4.6".to_string(),
                variant: Some("high".to_string()),
            }),
            _ => None,
        }
    }

    fn context_lookup_model(&self, model: &str) -> String {
        if model.starts_with("openrouter/") {
            model.to_string()
        } else {
            format!("openrouter/{model}")
        }
    }
}

impl ProviderModelPolicy for OpenAiDirectModelPolicy {
    fn default_model(&self) -> &str {
        "gpt-5.4"
    }

    fn supported_variants(&self, model: &str) -> &'static [&'static str] {
        let id = model_id(model).to_ascii_lowercase();
        if id.starts_with("gpt-5") || id.starts_with("o") {
            OPENROUTER_REASONING_VARIANTS
        } else {
            &[]
        }
    }

    fn default_model_variant(&self, model: &str) -> Option<&'static str> {
        self.supported_variants(model)
            .contains(&"medium")
            .then_some("medium")
    }

    fn request_variant_config(&self, model: &str, variant: &str) -> Option<VariantRequestConfig> {
        if !self.supported_variants(model).contains(&variant) {
            return None;
        }
        Some(VariantRequestConfig::ReasoningEffort(
            clamp_reasoning_effort(model, variant),
        ))
    }

    fn default_agent_model(&self, tier: &str) -> Option<AgentModelSelection> {
        match tier {
            "low" => Some(AgentModelSelection {
                model: "gpt-5.4-mini".to_string(),
                variant: Some("low".to_string()),
            }),
            "medium" => Some(AgentModelSelection {
                model: "gpt-5.4".to_string(),
                variant: Some("medium".to_string()),
            }),
            "high" => Some(AgentModelSelection {
                model: "gpt-5.4".to_string(),
                variant: Some("high".to_string()),
            }),
            _ => None,
        }
    }
}
