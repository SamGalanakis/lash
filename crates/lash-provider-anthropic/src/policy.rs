//! Model-capability policy: which thinking variants each Claude model supports,
//! how a requested effort maps onto the wire `thinking`/`output_config`, and
//! effort clamping.

use crate::support::*;
use lash_core::provider::{ModelCapability, ModelReasoningCapability};

pub(crate) const ANTHROPIC_VERSION: &str = "2023-06-01";
pub(crate) const FINE_GRAINED_BETA: &str = "fine-grained-tool-streaming-2025-05-14";
pub(crate) const INTERLEAVED_THINKING_BETA: &str = "interleaved-thinking-2025-05-14";
pub(crate) const DEFAULT_MAX_OUTPUT_TOKENS: u64 = 32_768;

const CLAUDE_ADAPTIVE_XHIGH_VARIANTS: &[&str] = &["low", "medium", "high", "xhigh"];
const CLAUDE_ADAPTIVE_MAX_VARIANTS: &[&str] = &["low", "medium", "high", "max"];
const CLAUDE_ADAPTIVE_VARIANTS: &[&str] = &["low", "medium", "high"];
const CLAUDE_BUDGET_VARIANTS: &[&str] = &["none", "low", "medium", "high"];

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum AnthropicThinkingConfig {
    Adaptive { effort: String },
    Budget { budget_tokens: i32 },
}

#[derive(Clone, Debug)]
pub(crate) struct AnthropicModelPolicy;

pub(crate) fn anthropic_supports_adaptive_thinking(model: &str) -> bool {
    let lower = model.to_ascii_lowercase();
    lower.contains("opus-4-6")
        || lower.contains("opus-4.6")
        || lower.contains("opus-4-7")
        || lower.contains("opus-4.7")
        || lower.contains("sonnet-4-6")
        || lower.contains("sonnet-4.6")
}

pub(crate) fn anthropic_supports_xhigh(model: &str) -> bool {
    let lower = model.to_ascii_lowercase();
    lower.contains("opus-4-7") || lower.contains("opus-4.7")
}

pub(crate) fn anthropic_supports_max(model: &str) -> bool {
    let lower = model.to_ascii_lowercase();
    lower.contains("opus-4-6") || lower.contains("opus-4.6")
}

/// Clamp a requested effort level to what the target Anthropic model actually
/// supports. `xhigh` is Opus-4.7-only; on Opus 4.6 it maps to `max`; every
/// other adaptive model collapses unknown effort down to `high`.
pub(crate) fn clamp_effort(model: &str, effort: &str) -> String {
    let normalized = effort.trim().to_ascii_lowercase();
    match normalized.as_str() {
        "xhigh" => {
            if anthropic_supports_xhigh(model) {
                "xhigh".to_string()
            } else if anthropic_supports_max(model) {
                "max".to_string()
            } else {
                "high".to_string()
            }
        }
        "max" => {
            if anthropic_supports_max(model) {
                "max".to_string()
            } else if anthropic_supports_xhigh(model) {
                "xhigh".to_string()
            } else {
                "high".to_string()
            }
        }
        other => other.to_string(),
    }
}

fn reasoning_capability_for_variants(
    variants: &[&str],
    supports_max_tokens: bool,
) -> ModelCapability {
    if variants.is_empty() {
        return ModelCapability::default();
    }
    ModelCapability {
        reasoning: Some(ModelReasoningCapability {
            supported_efforts: variants.iter().map(|effort| (*effort).to_string()).collect(),
            supports_max_tokens,
            ..ModelReasoningCapability::default()
        }),
    }
}

impl ProviderModelPolicy for AnthropicModelPolicy {
    fn model_capability(&self, model: &str) -> ModelCapability {
        let lower = model.to_ascii_lowercase();
        if anthropic_supports_adaptive_thinking(model) {
            if anthropic_supports_xhigh(model) {
                return reasoning_capability_for_variants(CLAUDE_ADAPTIVE_XHIGH_VARIANTS, false);
            }
            if anthropic_supports_max(model) {
                return reasoning_capability_for_variants(CLAUDE_ADAPTIVE_MAX_VARIANTS, false);
            }
            return reasoning_capability_for_variants(CLAUDE_ADAPTIVE_VARIANTS, false);
        }
        if lower.contains("haiku-4")
            || lower.contains("claude-opus-4")
            || lower.contains("claude-sonnet-4")
        {
            return reasoning_capability_for_variants(CLAUDE_BUDGET_VARIANTS, true);
        }
        ModelCapability::default()
    }
}

impl AnthropicModelPolicy {
    pub(crate) fn thinking_config(
        &self,
        model: &str,
        variant: &str,
    ) -> Option<AnthropicThinkingConfig> {
        let capability = self.model_capability(model);
        if !capability.reasoning.as_ref().is_some_and(|reasoning| {
            reasoning
                .supported_efforts
                .iter()
                .any(|effort| effort == variant)
        }) {
            return None;
        }
        if anthropic_supports_adaptive_thinking(model) {
            if variant == "none" {
                return None;
            }
            Some(AnthropicThinkingConfig::Adaptive {
                effort: variant.to_string(),
            })
        } else {
            let budget_tokens = match variant {
                "none" => return None,
                "low" => 1_024,
                "medium" => 4_096,
                "high" => 12_288,
                _ => return None,
            };
            Some(AnthropicThinkingConfig::Budget { budget_tokens })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anthropic_capability_is_model_specific_between_adaptive_and_budget() {
        let policy = AnthropicModelPolicy;
        let adaptive = policy.model_capability("claude-sonnet-4.6");
        let budget = policy.model_capability("claude-haiku-4-5");

        assert_eq!(
            adaptive
                .reasoning
                .as_ref()
                .map(|reasoning| reasoning.supports_max_tokens),
            Some(false)
        );
        assert_eq!(
            adaptive
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
            budget
                .reasoning
                .as_ref()
                .map(|reasoning| reasoning.supports_max_tokens),
            Some(true)
        );
        assert_eq!(
            budget
                .reasoning
                .as_ref()
                .map(|reasoning| reasoning.supported_efforts.clone()),
            Some(vec![
                "none".to_string(),
                "low".to_string(),
                "medium".to_string(),
                "high".to_string()
            ])
        );
    }

    #[test]
    fn anthropic_capability_tracks_opus_xhigh_and_max_split() {
        let policy = AnthropicModelPolicy;

        assert_eq!(
            policy
                .model_capability("claude-opus-4.7")
                .reasoning
                .as_ref()
                .map(|reasoning| reasoning.supported_efforts.clone()),
            Some(vec![
                "low".to_string(),
                "medium".to_string(),
                "high".to_string(),
                "xhigh".to_string()
            ])
        );
        assert_eq!(
            policy
                .model_capability("claude-opus-4.6")
                .reasoning
                .as_ref()
                .map(|reasoning| reasoning.supported_efforts.clone()),
            Some(vec![
                "low".to_string(),
                "medium".to_string(),
                "high".to_string(),
                "max".to_string()
            ])
        );
    }

    #[test]
    fn anthropic_unknown_models_expose_no_reasoning_capability() {
        let policy = AnthropicModelPolicy;
        assert_eq!(
            policy.model_capability("claude-3.5-sonnet"),
            ModelCapability::default()
        );
    }
}
