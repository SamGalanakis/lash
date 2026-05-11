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
