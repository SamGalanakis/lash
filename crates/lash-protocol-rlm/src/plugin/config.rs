use lash_plugin_tool_output_budget::ToolOutputBudgetConfig;

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RlmProtocolPluginConfig {
    pub observe_projection: ToolOutputBudgetConfig,
    #[serde(default)]
    pub prompt_features: crate::protocol::RlmPromptFeatures,
    #[serde(default)]
    pub lashlang_abilities: lashlang::LashlangAbilities,
    #[serde(default)]
    pub lashlang_language_features: lashlang::LashlangLanguageFeatures,
    #[serde(default = "default_max_output_chars")]
    pub max_output_chars: usize,
    #[serde(default = "default_continue_as_soft_warn_tokens")]
    pub continue_as_soft_warn_tokens: Option<usize>,
    /// In the "Bound Variables" prompt section, render a variable's value
    /// inline (so the model can read it directly) when its compact JSON form
    /// fits within this many characters; larger values fall back to a
    /// type + size hint. Set to 0 to always use hints.
    #[serde(default = "default_bound_variables_inline_char_limit")]
    pub bound_variables_inline_char_limit: usize,
}

fn default_max_output_chars() -> usize {
    10_000
}

fn default_continue_as_soft_warn_tokens() -> Option<usize> {
    Some(100_000)
}

fn default_bound_variables_inline_char_limit() -> usize {
    1_024
}

impl Default for RlmProtocolPluginConfig {
    fn default() -> Self {
        Self {
            observe_projection: ToolOutputBudgetConfig::default(),
            prompt_features: crate::protocol::RlmPromptFeatures::default(),
            lashlang_abilities: lashlang::LashlangAbilities::default(),
            lashlang_language_features: lashlang::LashlangLanguageFeatures::default(),
            max_output_chars: default_max_output_chars(),
            continue_as_soft_warn_tokens: default_continue_as_soft_warn_tokens(),
            bound_variables_inline_char_limit: default_bound_variables_inline_char_limit(),
        }
    }
}

impl RlmProtocolPluginConfig {
    pub fn with_lashlang_abilities(mut self, abilities: lashlang::LashlangAbilities) -> Self {
        self.lashlang_abilities = abilities;
        self
    }

    pub fn with_bound_variables_inline_char_limit(mut self, limit: usize) -> Self {
        self.bound_variables_inline_char_limit = limit;
        self
    }

    pub fn with_lashlang_language_features(
        mut self,
        language_features: lashlang::LashlangLanguageFeatures,
    ) -> Self {
        self.lashlang_language_features = language_features;
        self
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
