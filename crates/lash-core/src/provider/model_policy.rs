use super::support::*;
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct StaticModelPolicy {
    model_capabilities: HashMap<String, ModelCapability>,
}

impl StaticModelPolicy {
    pub fn new() -> Self {
        Self {
            model_capabilities: HashMap::new(),
        }
    }

    pub fn with_model_capability(mut self, model: impl Into<String>, capability: ModelCapability) -> Self {
        self.model_capabilities.insert(model.into(), capability);
        self
    }
}

impl Default for StaticModelPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl ProviderModelPolicy for StaticModelPolicy {
    fn model_capability(&self, model: &str) -> ModelCapability {
        self.model_capabilities
            .get(model)
            .cloned()
            .unwrap_or_default()
    }
}
