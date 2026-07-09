use crate::support::*;
use lash_core::provider::{ModelCapability, ModelReasoningCapability};

fn reasoning_capability_for_variants(variants: &[&str]) -> ModelCapability {
    if variants.is_empty() {
        return ModelCapability::default();
    }
    ModelCapability {
        reasoning: Some(ModelReasoningCapability {
            supported_efforts: variants.iter().map(|variant| (*variant).to_string()).collect(),
            ..ModelReasoningCapability::default()
        }),
    }
}

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

    pub(crate) fn reasoning_effort(&self, model: &str, variant: &str) -> Option<String> {
        let capability = self.model_capability(model);
        capability
            .reasoning
            .as_ref()
            .is_some_and(|reasoning| {
                reasoning
                    .supported_efforts
                    .iter()
                    .any(|effort| effort == variant)
            })
            .then(|| variant.to_string())
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

impl OpenAiDirectModelPolicy {
    pub(crate) fn reasoning_effort(&self, model: &str, variant: &str) -> Option<String> {
        self.model_capability(model)
            .reasoning
            .as_ref()
            .is_some_and(|reasoning| {
                reasoning
                    .supported_efforts
                    .iter()
                    .any(|effort| effort == variant)
            })
            .then(|| clamp_reasoning_effort(model, variant))
    }
}

impl ProviderModelPolicy for OpenAiModelPolicy {
    fn model_capability(&self, model: &str) -> ModelCapability {
        if !base_url_is_openrouter(&self.base_url) {
            return ModelCapability::default();
        }
        let lower = model.to_ascii_lowercase();
        if lower.contains("gpt") || lower.contains("claude") || lower.contains("gemini-3") {
            return reasoning_capability_for_variants(OPENROUTER_REASONING_VARIANTS);
        }
        ModelCapability::default()
    }
}

impl ProviderModelPolicy for OpenAiDirectModelPolicy {
    fn model_capability(&self, model: &str) -> ModelCapability {
        let id = model_id(model).to_ascii_lowercase();
        if id.starts_with("gpt-5") || id.starts_with("o") {
            return reasoning_capability_for_variants(OPENROUTER_REASONING_VARIANTS);
        }
        ModelCapability::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openrouter_capability_is_model_specific() {
        let policy = OpenAiModelPolicy::new(OPENROUTER_BASE_URL);
        let gpt_capability = policy.model_capability("openai/gpt-5.4");
        let llama_capability = policy.model_capability("meta-llama/llama-3.3-70b");

        assert_eq!(
            gpt_capability
                .reasoning
                .as_ref()
                .map(|reasoning| reasoning.supported_efforts.clone()),
            Some(
                OPENROUTER_REASONING_VARIANTS
                    .iter()
                    .map(|effort| (*effort).to_string())
                    .collect()
            )
        );
        assert_eq!(llama_capability, ModelCapability::default());
    }

    #[test]
    fn direct_openai_capability_is_absent_for_non_reasoning_models() {
        let policy = OpenAiDirectModelPolicy;

        assert!(policy.model_capability("gpt-5.4").reasoning.is_some());
        assert_eq!(
            policy.model_capability("gpt-4.1"),
            ModelCapability::default()
        );
    }

    #[test]
    fn direct_openai_reasoning_effort_preserves_gpt_53_minimal_clamp() {
        let policy = OpenAiDirectModelPolicy;

        assert_eq!(
            policy.reasoning_effort("gpt-5.3", "minimal").as_deref(),
            Some("low")
        );
    }
}
