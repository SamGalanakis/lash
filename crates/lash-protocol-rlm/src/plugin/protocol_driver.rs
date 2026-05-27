use std::sync::Arc;

use lash_core::plugin::ProtocolDriverPlugin;
use lash_core::{ProtocolBuildInput, TurnDriverPreamble};

use super::RlmProtocolPluginConfig;
use crate::driver::{RlmProjectorConfig, SharedPromptUsage, build_rlm_preamble};

pub(super) struct RlmProtocolDriver {
    pub(super) config: RlmProtocolPluginConfig,
    pub(super) last_prompt_usage: SharedPromptUsage,
}

impl ProtocolDriverPlugin for RlmProtocolDriver {
    fn build_preamble(&self, input: ProtocolBuildInput) -> TurnDriverPreamble {
        build_rlm_preamble(
            input,
            RlmProjectorConfig {
                max_output_chars: self.config.max_output_chars,
                max_budget_tokens: self.config.continue_as_soft_warn_tokens,
                last_prompt_usage: Arc::clone(&self.last_prompt_usage),
                prompt_features: self.config.prompt_features,
            },
        )
    }
}
