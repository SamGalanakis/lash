use crate::provider::{OPENROUTER_BASE_URL, Provider};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VariantRequestConfig {
    ReasoningEffort(String),
    GoogleThinkingLevel {
        level: String,
    },
    GoogleThinkingBudget {
        budget_tokens: i32,
    },
    /// Anthropic adaptive thinking (Opus 4.6+ and Sonnet 4.6): the model
    /// decides when and how much to think. `effort` controls the ceiling
    /// ("low"/"medium"/"high"/"xhigh").
    AnthropicAdaptiveThinking {
        effort: String,
    },
    /// Anthropic budget-based thinking for models without adaptive support
    /// (Haiku, older Claude 4). `budget_tokens` gates how much reasoning
    /// the model is allowed to emit per turn.
    AnthropicThinkingBudget {
        budget_tokens: i32,
    },
}

const OPENAI_GPT5_VARIANTS: &[&str] = &["minimal", "low", "medium", "high"];
const OPENAI_GPT5_XHIGH_VARIANTS: &[&str] = &["minimal", "low", "medium", "high", "xhigh"];
const CODEX_VARIANTS: &[&str] = &["low", "medium", "high"];
const CODEX_XHIGH_VARIANTS: &[&str] = &["low", "medium", "high", "xhigh"];
const GEMINI_31_VARIANTS: &[&str] = &["low", "medium", "high"];
const GEMINI_3_VARIANTS: &[&str] = &["low", "high"];
const GEMINI_25_VARIANTS: &[&str] = &["high", "max"];
const OPENROUTER_REASONING_VARIANTS: &[&str] =
    &["none", "minimal", "low", "medium", "high", "xhigh"];
const CLAUDE_ADAPTIVE_XHIGH_VARIANTS: &[&str] = &["low", "medium", "high", "xhigh"];
const CLAUDE_ADAPTIVE_MAX_VARIANTS: &[&str] = &["low", "medium", "high", "max"];
const CLAUDE_ADAPTIVE_VARIANTS: &[&str] = &["low", "medium", "high"];
const CLAUDE_BUDGET_VARIANTS: &[&str] = &["none", "low", "medium", "high"];

fn has_xhigh_suffix(model: &str) -> bool {
    let lower = model.to_ascii_lowercase();
    lower.contains("5.2") || lower.contains("5.3") || lower.contains("5.4")
}

fn is_openrouter(provider: &Provider) -> bool {
    matches!(
        provider,
        Provider::OpenAiGeneric { base_url, .. }
            if base_url.trim_end_matches('/') == OPENROUTER_BASE_URL
    )
}

/// True for Anthropic models that use the adaptive-thinking API
/// (Opus 4.6+ and Sonnet 4.6). These also accept per-request `output_config.effort`.
pub(crate) fn anthropic_supports_adaptive_thinking(model: &str) -> bool {
    let lower = model.to_ascii_lowercase();
    lower.contains("opus-4-6")
        || lower.contains("opus-4.6")
        || lower.contains("opus-4-7")
        || lower.contains("opus-4.7")
        || lower.contains("sonnet-4-6")
        || lower.contains("sonnet-4.6")
}

/// True for Anthropic models on which `xhigh` is a distinct effort level
/// (Opus 4.7). Other adaptive models clamp `xhigh` back to `high` (or
/// `max` on Opus 4.6 — see [`anthropic_supports_max`]).
pub(crate) fn anthropic_supports_xhigh(model: &str) -> bool {
    let lower = model.to_ascii_lowercase();
    lower.contains("opus-4-7") || lower.contains("opus-4.7")
}

/// True for Anthropic models that accept `"max"` as a distinct effort
/// level (Opus 4.6 only, per pi-mono `mapThinkingLevelToEffort`).
pub(crate) fn anthropic_supports_max(model: &str) -> bool {
    let lower = model.to_ascii_lowercase();
    lower.contains("opus-4-6") || lower.contains("opus-4.6")
}

pub fn supported_variants(provider: &Provider, model: &str) -> &'static [&'static str] {
    let lower = model.to_ascii_lowercase();
    match provider {
        Provider::Anthropic { .. } => {
            if anthropic_supports_adaptive_thinking(model) {
                if anthropic_supports_xhigh(model) {
                    CLAUDE_ADAPTIVE_XHIGH_VARIANTS
                } else if anthropic_supports_max(model) {
                    CLAUDE_ADAPTIVE_MAX_VARIANTS
                } else {
                    CLAUDE_ADAPTIVE_VARIANTS
                }
            } else if lower.contains("haiku-4")
                || lower.contains("claude-opus-4")
                || lower.contains("claude-sonnet-4")
            {
                CLAUDE_BUDGET_VARIANTS
            } else {
                &[]
            }
        }
        Provider::Codex { .. } => {
            if lower.contains("gpt-5") {
                if lower.contains("codex") {
                    if has_xhigh_suffix(&lower) {
                        CODEX_XHIGH_VARIANTS
                    } else {
                        CODEX_VARIANTS
                    }
                } else if has_xhigh_suffix(&lower) {
                    OPENAI_GPT5_XHIGH_VARIANTS
                } else {
                    OPENAI_GPT5_VARIANTS
                }
            } else {
                &[]
            }
        }
        Provider::GoogleOAuth { .. } => {
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
        Provider::OpenAiGeneric { .. } => {
            if !is_openrouter(provider) {
                return &[];
            }
            if lower.contains("gpt") || lower.contains("claude") || lower.contains("gemini-3") {
                OPENROUTER_REASONING_VARIANTS
            } else {
                &[]
            }
        }
    }
}

pub fn default_variant(provider: &Provider, model: &str) -> Option<&'static str> {
    let variants = supported_variants(provider, model);
    if variants.is_empty() {
        return None;
    }
    let lower = model.to_ascii_lowercase();
    match provider {
        Provider::Anthropic { .. } => {
            if variants.contains(&"xhigh") {
                Some("xhigh")
            } else if variants.contains(&"max") {
                Some("max")
            } else {
                Some("high")
            }
        }
        Provider::Codex { .. } => {
            if variants.contains(&"xhigh") {
                Some("xhigh")
            } else {
                Some("high")
            }
        }
        Provider::GoogleOAuth { .. } => Some("high"),
        Provider::OpenAiGeneric { .. } => {
            if lower.contains("gpt") {
                Some("medium")
            } else {
                Some("high")
            }
        }
    }
}

pub fn validate(provider: &Provider, model: &str, variant: &str) -> Result<(), String> {
    let variants = supported_variants(provider, model);
    if variants.is_empty() {
        return Err(format!(
            "Model `{}` on {} does not expose configurable variants.",
            model,
            provider.label()
        ));
    }
    if variants.contains(&variant) {
        return Ok(());
    }
    Err(format!(
        "Unsupported variant `{}` for `{}` on {}. Available: {}",
        variant,
        model,
        provider.label(),
        variants.join(", ")
    ))
}

pub fn request_config(
    provider: &Provider,
    model: &str,
    variant: &str,
) -> Option<VariantRequestConfig> {
    if validate(provider, model, variant).is_err() {
        return None;
    }

    match provider {
        Provider::Anthropic { .. } => {
            if anthropic_supports_adaptive_thinking(model) {
                if variant == "none" {
                    return None;
                }
                Some(VariantRequestConfig::AnthropicAdaptiveThinking {
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
                Some(VariantRequestConfig::AnthropicThinkingBudget { budget_tokens })
            }
        }
        Provider::Codex { .. } => Some(VariantRequestConfig::ReasoningEffort(variant.to_string())),
        Provider::GoogleOAuth { .. } => {
            let lower = model.to_ascii_lowercase();
            if lower.contains("gemini-2.5") {
                let budget_tokens = match variant {
                    "high" => 16_000,
                    "max" => 24_576,
                    _ => return None,
                };
                Some(VariantRequestConfig::GoogleThinkingBudget { budget_tokens })
            } else {
                Some(VariantRequestConfig::GoogleThinkingLevel {
                    level: variant.to_string(),
                })
            }
        }
        Provider::OpenAiGeneric { .. } => {
            if is_openrouter(provider) {
                Some(VariantRequestConfig::ReasoningEffort(variant.to_string()))
            } else {
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn codex() -> Provider {
        Provider::Codex {
            access_token: "tok".into(),
            refresh_token: "ref".into(),
            expires_at: u64::MAX,
            account_id: None,
            options: crate::provider::ProviderOptions::default(),
        }
    }

    fn google() -> Provider {
        Provider::GoogleOAuth {
            access_token: "tok".into(),
            refresh_token: "ref".into(),
            expires_at: u64::MAX,
            project_id: None,
            options: crate::provider::ProviderOptions::default(),
        }
    }

    fn openrouter() -> Provider {
        Provider::OpenAiGeneric {
            api_key: "key".into(),
            base_url: OPENROUTER_BASE_URL.into(),
            options: crate::provider::ProviderOptions::default(),
        }
    }

    fn anthropic() -> Provider {
        Provider::Anthropic {
            api_key: "key".into(),
            base_url: None,
            options: crate::provider::ProviderOptions::default(),
        }
    }

    #[test]
    fn codex_variants_follow_model_family() {
        assert_eq!(
            supported_variants(&codex(), "gpt-5.4"),
            OPENAI_GPT5_XHIGH_VARIANTS
        );
        assert_eq!(
            supported_variants(&codex(), "gpt-5.3-codex"),
            CODEX_XHIGH_VARIANTS
        );
        assert_eq!(default_variant(&codex(), "gpt-5.4"), Some("xhigh"));
    }

    #[test]
    fn google_variants_switch_between_level_and_budget_modes() {
        assert_eq!(
            supported_variants(&google(), "gemini-3.1-pro-preview"),
            GEMINI_31_VARIANTS
        );
        assert_eq!(
            request_config(&google(), "gemini-3.1-pro-preview", "medium"),
            Some(VariantRequestConfig::GoogleThinkingLevel {
                level: "medium".into()
            })
        );
        assert_eq!(
            request_config(&google(), "gemini-2.5-pro", "max"),
            Some(VariantRequestConfig::GoogleThinkingBudget {
                budget_tokens: 24_576
            })
        );
    }

    #[test]
    fn openrouter_variants_are_limited_to_reasoning_models() {
        assert_eq!(
            supported_variants(&openrouter(), "anthropic/claude-sonnet-4.6"),
            OPENROUTER_REASONING_VARIANTS
        );
        assert!(supported_variants(&openrouter(), "minimax/minimax-m2.5").is_empty());
    }

    #[test]
    fn anthropic_opus_47_exposes_xhigh_variant() {
        let p = anthropic();
        assert_eq!(
            supported_variants(&p, "claude-opus-4-7"),
            CLAUDE_ADAPTIVE_XHIGH_VARIANTS
        );
        assert_eq!(default_variant(&p, "claude-opus-4-7"), Some("xhigh"));
        assert_eq!(
            request_config(&p, "claude-opus-4-7", "xhigh"),
            Some(VariantRequestConfig::AnthropicAdaptiveThinking {
                effort: "xhigh".into()
            })
        );
    }

    #[test]
    fn anthropic_sonnet_46_has_adaptive_without_xhigh() {
        let p = anthropic();
        assert_eq!(
            supported_variants(&p, "claude-sonnet-4-6"),
            CLAUDE_ADAPTIVE_VARIANTS
        );
        assert_eq!(default_variant(&p, "claude-sonnet-4-6"), Some("high"));
        assert!(!supported_variants(&p, "claude-sonnet-4-6").contains(&"xhigh"));
    }

    #[test]
    fn anthropic_haiku_46_uses_budget_variants() {
        let p = anthropic();
        assert_eq!(
            supported_variants(&p, "claude-haiku-4-6"),
            CLAUDE_BUDGET_VARIANTS
        );
        assert_eq!(
            request_config(&p, "claude-haiku-4-6", "medium"),
            Some(VariantRequestConfig::AnthropicThinkingBudget {
                budget_tokens: 4_096
            })
        );
        assert_eq!(request_config(&p, "claude-haiku-4-6", "none"), None);
    }
}
