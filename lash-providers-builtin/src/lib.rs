//! Registration helper for all first-party provider crates.

use std::sync::Once;

/// Register every built-in provider factory with lash's global provider registry.
pub fn register_all() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        lash_provider_anthropic::AnthropicProviderFactory::register();
        lash_provider_openai::OpenAiProviderFactory::register();
        lash_provider_openai::OpenAiCompatibleProviderFactory::register();
        lash_provider_codex::CodexProviderFactory::register();
        lash_provider_google::GoogleOAuthProviderFactory::register();
    });
}
