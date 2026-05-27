use lash_plugin_tool_output_budget::ToolOutputBudgetConfig;

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RlmProtocolPluginConfig {
    pub observe_projection: ToolOutputBudgetConfig,
    #[serde(default)]
    pub prompt_features: crate::protocol::RlmPromptFeatures,
    #[serde(default = "default_max_output_chars")]
    pub max_output_chars: usize,
    #[serde(default = "default_continue_as_soft_warn_tokens")]
    pub continue_as_soft_warn_tokens: Option<usize>,
}

fn default_max_output_chars() -> usize {
    10_000
}

fn default_continue_as_soft_warn_tokens() -> Option<usize> {
    Some(100_000)
}

impl Default for RlmProtocolPluginConfig {
    fn default() -> Self {
        Self {
            observe_projection: ToolOutputBudgetConfig::default(),
            prompt_features: crate::protocol::RlmPromptFeatures::default(),
            max_output_chars: default_max_output_chars(),
            continue_as_soft_warn_tokens: default_continue_as_soft_warn_tokens(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rlm_config_defaults_soft_budget_threshold() {
        let config = RlmProtocolPluginConfig::default();

        assert_eq!(config.continue_as_soft_warn_tokens, Some(100_000));
    }
}
