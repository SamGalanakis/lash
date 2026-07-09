pub(crate) struct ProviderCliMetadata {
    pub kind: &'static str,
    pub cli_label: &'static str,
    pub setup_name: &'static str,
    pub setup_description: &'static str,
}

pub(crate) const PROVIDER_SETUP_ORDER: &[&str] = &[
    "anthropic",
    "openai",
    "openai-compatible",
    "codex",
    "google_oauth",
];

const PROVIDERS: &[ProviderCliMetadata] = &[
    ProviderCliMetadata {
        kind: "anthropic",
        cli_label: "Anthropic API (Claude)",
        setup_name: "Anthropic API",
        setup_description: "Claude via Anthropic API key",
    },
    ProviderCliMetadata {
        kind: "openai",
        cli_label: "OpenAI API (API key)",
        setup_name: "OpenAI API",
        setup_description: "OpenAI Responses API key",
    },
    ProviderCliMetadata {
        kind: "openai-compatible",
        cli_label: "OpenAI-compatible (API key)",
        setup_name: "OpenAI-compatible",
        setup_description: "Any OpenAI-compatible API endpoint",
    },
    ProviderCliMetadata {
        kind: "codex",
        cli_label: "OpenAI Codex OAuth",
        setup_name: "Codex",
        setup_description: "ChatGPT Plus/Pro/Team",
    },
    ProviderCliMetadata {
        kind: "google_oauth",
        cli_label: "Google OAuth (Gemini)",
        setup_name: "Google OAuth",
        setup_description: "Gemini via Google account",
    },
];

pub(crate) fn provider_metadata(kind: &str) -> Option<&'static ProviderCliMetadata> {
    PROVIDERS.iter().find(|metadata| metadata.kind == kind)
}

pub(crate) fn provider_cli_label(kind: &str) -> &'static str {
    provider_metadata(kind)
        .map(|metadata| metadata.cli_label)
        .unwrap_or("Provider")
}

pub(crate) fn provider_setup_name(kind: &str) -> &'static str {
    provider_metadata(kind)
        .map(|metadata| metadata.setup_name)
        .unwrap_or("Unknown")
}

pub(crate) fn provider_setup_description(kind: &str) -> &'static str {
    provider_metadata(kind)
        .map(|metadata| metadata.setup_description)
        .unwrap_or("")
}

pub(crate) fn default_model_for_provider(kind: &str) -> Result<&'static str, String> {
    match kind {
        "anthropic" => Ok("claude-opus-4-7"),
        "openai" => Ok("gpt-5.4"),
        "openai-compatible" => Ok("anthropic/claude-sonnet-4.6"),
        "codex" => Ok("gpt-5.5"),
        "google_oauth" => Ok("gemini-3.1-pro-preview"),
        other => Err(format!(
            "No CLI default model is defined for provider `{other}`. Pass `--model` explicitly."
        )),
    }
}

pub(crate) fn provider_catalog_model_id(kind: &str, model: &str) -> String {
    match kind {
        "anthropic" => prefixed_unless_prefix("anthropic", model),
        "openai" | "codex" => prefixed_model_id("openai", model),
        "openai-compatible" => prefixed_unless_prefix("openrouter", model),
        "google_oauth" => prefixed_model_id("google", model),
        _ => model.to_string(),
    }
}

pub(crate) fn provider_wire_model_id(kind: &str, model: &str) -> String {
    match kind {
        "google_oauth" => model.strip_prefix("google/").unwrap_or(model).to_string(),
        _ => model.to_string(),
    }
}

pub(crate) fn default_model_variant_for_provider(
    kind: &str,
    model: &str,
    supported_efforts: &[String],
    capability_default_effort: Option<&str>,
) -> Option<String> {
    if supported_efforts.is_empty() {
        return None;
    }
    if let Some(default_effort) = capability_default_effort
        && supported_efforts
            .iter()
            .any(|supported| supported == default_effort)
    {
        return Some(default_effort.to_string());
    }
    heuristic_default_model_variant_for_provider(kind, model, supported_efforts).map(str::to_string)
}

fn heuristic_default_model_variant_for_provider(
    kind: &str,
    model: &str,
    supported_efforts: &[String],
) -> Option<&'static str> {
    let supports = |effort: &str| supported_efforts.iter().any(|supported| supported == effort);
    match kind {
        "anthropic" => {
            if supports("xhigh") {
                Some("xhigh")
            } else if supports("max") {
                Some("max")
            } else {
                Some("high")
            }
        }
        "openai" => supports("medium").then_some("medium"),
        "openai-compatible" => {
            if model.to_ascii_lowercase().contains("gpt") {
                Some("medium")
            } else {
                Some("high")
            }
        }
        "codex" => {
            if model.eq_ignore_ascii_case("gpt-5.5") || supports("xhigh") {
                Some("xhigh")
            } else {
                Some("high")
            }
        }
        "google_oauth" => Some("high"),
        _ => None,
    }
}

fn prefixed_model_id(provider: &str, model: &str) -> String {
    if model.contains('/') {
        model.to_string()
    } else {
        format!("{provider}/{model}")
    }
}

fn prefixed_unless_prefix(provider: &str, model: &str) -> String {
    let prefix = format!("{provider}/");
    if model.starts_with(&prefix) {
        model.to_string()
    } else {
        format!("{provider}/{model}")
    }
}

#[cfg(test)]
mod tests {
    use super::default_model_variant_for_provider;

    #[test]
    fn capability_default_effort_is_preferred_when_supported() {
        let supported = vec!["low".to_string(), "medium".to_string(), "high".to_string()];

        assert_eq!(
            default_model_variant_for_provider(
                "openai",
                "gpt-5.4",
                &supported,
                Some("low"),
            ),
            Some("low".to_string())
        );
    }

    #[test]
    fn missing_capability_default_effort_falls_back_to_heuristics() {
        let supported = vec!["low".to_string(), "medium".to_string(), "high".to_string()];

        assert_eq!(
            default_model_variant_for_provider("openai", "gpt-5.4", &supported, None),
            Some("medium".to_string())
        );
    }

    #[test]
    fn unsupported_capability_default_effort_falls_back_to_heuristics() {
        let supported = vec!["low".to_string(), "medium".to_string(), "high".to_string()];

        assert_eq!(
            default_model_variant_for_provider(
                "openai",
                "gpt-5.4",
                &supported,
                Some("xhigh"),
            ),
            Some("medium".to_string())
        );
    }
}
