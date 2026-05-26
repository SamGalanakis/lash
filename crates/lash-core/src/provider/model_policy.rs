use super::support::*;

#[derive(Clone, Debug)]
pub struct StaticModelPolicy {
    supported_variants: &'static [&'static str],
}

impl StaticModelPolicy {
    pub fn new() -> Self {
        Self {
            supported_variants: &[],
        }
    }

    pub fn with_variants(supported_variants: &'static [&'static str]) -> Self {
        Self { supported_variants }
    }
}

impl Default for StaticModelPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl ProviderModelPolicy for StaticModelPolicy {
    fn supported_variants(&self, _model: &str) -> &'static [&'static str] {
        self.supported_variants
    }

    fn request_variant_config(&self, _model: &str, variant: &str) -> Option<VariantRequestConfig> {
        self.supported_variants
            .contains(&variant)
            .then(|| VariantRequestConfig::ReasoningEffort(variant.to_string()))
    }
}
