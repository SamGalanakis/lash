//! Model-capability policy: which thinking variants each Gemini model supports
//! and how a requested variant maps onto the wire `thinkingConfig`.

use crate::support::*;

const GEMINI_31_VARIANTS: &[&str] = &["low", "medium", "high"];
const GEMINI_3_VARIANTS: &[&str] = &["low", "high"];
const GEMINI_25_VARIANTS: &[&str] = &["high", "max"];

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum GoogleThinkingConfig {
    Level { level: String },
    Budget { budget_tokens: i32 },
}

#[derive(Clone, Debug)]
pub(crate) struct GoogleModelPolicy;

impl ProviderModelPolicy for GoogleModelPolicy {
    fn supported_variants(&self, model: &str) -> &'static [&'static str] {
        let lower = model.to_ascii_lowercase();
        if lower.contains("gemini-2.5") {
            GEMINI_25_VARIANTS
        } else if lower.contains("gemini-3.1") {
            GEMINI_31_VARIANTS
        } else if lower.contains("gemini-3") {
            GEMINI_3_VARIANTS
        } else {
            &[]
        }
    }
}

impl GoogleModelPolicy {
    pub(crate) fn thinking_config(
        &self,
        model: &str,
        variant: &str,
    ) -> Option<GoogleThinkingConfig> {
        if !self.supported_variants(model).contains(&variant) {
            return None;
        }
        let lower = model.to_ascii_lowercase();
        if lower.contains("gemini-2.5") {
            let budget_tokens = match variant {
                "high" => 16_000,
                "max" => 24_576,
                _ => return None,
            };
            Some(GoogleThinkingConfig::Budget { budget_tokens })
        } else {
            Some(GoogleThinkingConfig::Level {
                level: variant.to_string(),
            })
        }
    }
}
