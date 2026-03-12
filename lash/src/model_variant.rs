use crate::provider::{OPENAI_GENERIC_DEFAULT_BASE_URL, Provider};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VariantRequestConfig {
    ReasoningEffort(String),
    AnthropicThinkingBudget { budget_tokens: u32 },
    GoogleThinkingLevel { level: String },
    GoogleThinkingBudget { budget_tokens: i32 },
}

const OPENAI_GPT5_VARIANTS: &[&str] = &["minimal", "low", "medium", "high"];
const OPENAI_GPT5_XHIGH_VARIANTS: &[&str] = &["minimal", "low", "medium", "high", "xhigh"];
const CODEX_VARIANTS: &[&str] = &["low", "medium", "high"];
const CODEX_XHIGH_VARIANTS: &[&str] = &["low", "medium", "high", "xhigh"];
const CLAUDE_VARIANTS: &[&str] = &["low", "medium", "high", "max"];
const GEMINI_31_VARIANTS: &[&str] = &["low", "medium", "high"];
const GEMINI_3_VARIANTS: &[&str] = &["low", "high"];
const GEMINI_25_VARIANTS: &[&str] = &["high", "max"];
const OPENROUTER_REASONING_VARIANTS: &[&str] =
    &["none", "minimal", "low", "medium", "high", "xhigh"];

fn has_xhigh_suffix(model: &str) -> bool {
    let lower = model.to_ascii_lowercase();
    lower.contains("5.2") || lower.contains("5.3")
}

fn is_openrouter(provider: &Provider) -> bool {
    matches!(
        provider,
        Provider::OpenAiGeneric { base_url, .. }
            if base_url.trim_end_matches('/') == OPENAI_GENERIC_DEFAULT_BASE_URL
    )
}

pub fn supported_variants(provider: &Provider, model: &str) -> &'static [&'static str] {
    let lower = model.to_ascii_lowercase();
    match provider {
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
        Provider::Claude { .. } => {
            if lower.contains("claude") {
                CLAUDE_VARIANTS
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
        Provider::Codex { .. } => Some("medium"),
        Provider::Claude { .. } => Some("high"),
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
        Provider::Codex { .. } => Some(VariantRequestConfig::ReasoningEffort(variant.to_string())),
        Provider::Claude { .. } => {
            let budget_tokens = match variant {
                "low" => 4_000,
                "medium" => 8_000,
                "high" => 16_000,
                "max" => 31_999,
                _ => return None,
            };
            Some(VariantRequestConfig::AnthropicThinkingBudget { budget_tokens })
        }
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
        }
    }

    fn claude() -> Provider {
        Provider::Claude {
            access_token: "tok".into(),
            refresh_token: "ref".into(),
            expires_at: u64::MAX,
        }
    }

    fn google() -> Provider {
        Provider::GoogleOAuth {
            access_token: "tok".into(),
            refresh_token: "ref".into(),
            expires_at: u64::MAX,
            project_id: None,
        }
    }

    fn openrouter() -> Provider {
        Provider::OpenAiGeneric {
            api_key: "key".into(),
            base_url: OPENAI_GENERIC_DEFAULT_BASE_URL.into(),
        }
    }

    #[test]
    fn codex_variants_follow_model_family() {
        assert_eq!(
            supported_variants(&codex(), "gpt-5.4"),
            OPENAI_GPT5_VARIANTS
        );
        assert_eq!(
            supported_variants(&codex(), "gpt-5.3-codex"),
            CODEX_XHIGH_VARIANTS
        );
        assert_eq!(default_variant(&codex(), "gpt-5.4"), Some("medium"));
    }

    #[test]
    fn claude_uses_named_budget_presets() {
        assert_eq!(
            supported_variants(&claude(), "claude-sonnet-4-6"),
            CLAUDE_VARIANTS
        );
        assert_eq!(
            request_config(&claude(), "claude-sonnet-4-6", "high"),
            Some(VariantRequestConfig::AnthropicThinkingBudget {
                budget_tokens: 16_000
            })
        );
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
}
