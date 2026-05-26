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
    supported_variants: &[&str],
) -> Option<&'static str> {
    if supported_variants.is_empty() {
        return None;
    }
    match kind {
        "anthropic" => {
            if supported_variants.contains(&"xhigh") {
                Some("xhigh")
            } else if supported_variants.contains(&"max") {
                Some("max")
            } else {
                Some("high")
            }
        }
        "openai" => supported_variants.contains(&"medium").then_some("medium"),
        "openai-compatible" => {
            if model.to_ascii_lowercase().contains("gpt") {
                Some("medium")
            } else {
                Some("high")
            }
        }
        "codex" => {
            if model.eq_ignore_ascii_case("gpt-5.5") {
                Some("medium")
            } else if supported_variants.contains(&"xhigh") {
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
