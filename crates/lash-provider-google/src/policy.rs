//! Model-capability policy: which thinking variants each Gemini model supports
//! and how a requested variant maps onto the wire `thinkingConfig`.

use crate::support::*;
use lash_core::provider::{ModelCapability, ModelReasoningCapability};

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

fn reasoning_capability_for_variants(
    variants: &[&str],
    supports_max_tokens: bool,
) -> ModelCapability {
    if variants.is_empty() {
        return ModelCapability::default();
    }
    ModelCapability {
        reasoning: Some(ModelReasoningCapability {
            supported_efforts: variants.iter().map(|variant| (*variant).to_string()).collect(),
            supports_max_tokens,
            ..ModelReasoningCapability::default()
        }),
    }
}

impl ProviderModelPolicy for GoogleModelPolicy {
    fn model_capability(&self, model: &str) -> ModelCapability {
        let lower = model.to_ascii_lowercase();
        if lower.contains("gemini-2.5") {
            return reasoning_capability_for_variants(GEMINI_25_VARIANTS, true);
        }
        if lower.contains("gemini-3.1") {
            return reasoning_capability_for_variants(GEMINI_31_VARIANTS, false);
        }
        if lower.contains("gemini-3") {
            return reasoning_capability_for_variants(GEMINI_3_VARIANTS, false);
        }
        ModelCapability::default()
    }
}

impl GoogleModelPolicy {
    pub(crate) fn thinking_config(
        &self,
        model: &str,
        variant: &str,
    ) -> Option<GoogleThinkingConfig> {
        let capability = self.model_capability(model);
        if !capability.reasoning.as_ref().is_some_and(|reasoning| {
            reasoning
                .supported_efforts
                .iter()
                .any(|effort| effort == variant)
        }) {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn google_capability_is_model_specific_between_family_generations() {
        let policy = GoogleModelPolicy;
        let gemini_31 = policy.model_capability("gemini-3.1-pro-preview");
        let gemini_3 = policy.model_capability("gemini-3-pro");
        let gemini_25 = policy.model_capability("gemini-2.5-pro");
        let gemini_2 = policy.model_capability("gemini-2.0-flash");

        assert_eq!(
            gemini_31
                .reasoning
                .as_ref()
                .map(|reasoning| reasoning.supported_efforts.clone()),
            Some(vec![
                "low".to_string(),
                "medium".to_string(),
                "high".to_string()
            ])
        );
        assert_eq!(
            gemini_3
                .reasoning
                .as_ref()
                .map(|reasoning| reasoning.supported_efforts.clone()),
            Some(vec!["low".to_string(), "high".to_string()])
        );
        assert_eq!(
            gemini_25
                .reasoning
                .as_ref()
                .map(|reasoning| reasoning.supports_max_tokens),
            Some(true)
        );
        assert_eq!(gemini_2, ModelCapability::default());
    }

    #[test]
    fn google_thinking_config_respects_capability_effort_membership() {
        let policy = GoogleModelPolicy;

        assert!(matches!(
            policy.thinking_config("gemini-3.1-pro-preview", "medium"),
            Some(GoogleThinkingConfig::Level { .. })
        ));
        assert_eq!(
            policy.thinking_config("gemini-3.1-pro-preview", "max"),
            None
        );
    }
}
