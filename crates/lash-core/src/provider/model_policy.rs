use super::support::*;
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct StaticModelPolicy {
    supported_variants: &'static [&'static str],
    model_capabilities: HashMap<String, ModelCapability>,
}

impl StaticModelPolicy {
    pub fn new() -> Self {
        Self {
            supported_variants: &[],
            model_capabilities: HashMap::new(),
        }
    }

    pub fn with_variants(supported_variants: &'static [&'static str]) -> Self {
        Self {
            supported_variants,
            model_capabilities: HashMap::new(),
        }
    }

    pub fn with_model_capability(mut self, model: impl Into<String>, capability: ModelCapability) -> Self {
        self.model_capabilities.insert(model.into(), capability);
        self
    }

    fn capability_from_variants(&self) -> ModelCapability {
        if self.supported_variants.is_empty() {
            return ModelCapability::default();
        }
        ModelCapability {
            reasoning: Some(ModelReasoningCapability {
                supported_efforts: self
                    .supported_variants
                    .iter()
                    .map(|effort| (*effort).to_string())
                    .collect(),
                ..ModelReasoningCapability::default()
            }),
        }
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

    fn model_capability(&self, model: &str) -> ModelCapability {
        self.model_capabilities
            .get(model)
            .cloned()
            .unwrap_or_else(|| self.capability_from_variants())
    }
}
