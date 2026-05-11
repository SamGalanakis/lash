use super::support::*;

#[derive(Clone, Debug)]
pub struct StaticModelPolicy {
    default_model: &'static str,
    supported_variants: &'static [&'static str],
    default_variant: Option<&'static str>,
}

impl StaticModelPolicy {
    pub fn new(default_model: &'static str) -> Self {
        Self {
            default_model,
            supported_variants: &[],
            default_variant: None,
        }
    }

    pub fn with_variants(
        default_model: &'static str,
        supported_variants: &'static [&'static str],
        default_variant: Option<&'static str>,
    ) -> Self {
        Self {
            default_model,
            supported_variants,
            default_variant,
        }
    }
}

impl ProviderModelPolicy for StaticModelPolicy {
    fn default_model(&self) -> &str {
        self.default_model
    }

    fn supported_variants(&self, _model: &str) -> &'static [&'static str] {
        self.supported_variants
    }

    fn default_model_variant(&self, _model: &str) -> Option<&'static str> {
        self.default_variant
    }

    fn request_variant_config(&self, _model: &str, variant: &str) -> Option<VariantRequestConfig> {
        self.supported_variants
            .contains(&variant)
            .then(|| VariantRequestConfig::ReasoningEffort(variant.to_string()))
    }

    fn default_agent_model(&self, _tier: &str) -> Option<AgentModelSelection> {
        None
    }
}
