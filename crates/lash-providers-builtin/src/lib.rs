//! Metadata for the first-party provider crates.

/// Provider kinds compiled into the standard CLI host.
pub fn provider_kinds() -> &'static [&'static str] {
    &[
        "anthropic",
        "openai",
        "openai-compatible",
        "codex",
        "google_oauth",
    ]
}
