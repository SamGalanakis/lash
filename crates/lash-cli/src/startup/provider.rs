//! Startup stage: resolve the on-disk config and the active provider's
//! credentials. Interactive onboarding runs only when there is no usable
//! stored/shortcut credential (or `--provider` forces it).

use crate::Args;
use crate::config::LashConfig;
use lash::provider::ProviderHandle;
use lash_provider_openai::{OpenAiCompatibleProvider, OpenAiProvider};

use super::onboarding;

pub(super) fn openai_shortcut_provider(api_key: String, base_url: &str) -> ProviderHandle {
    if base_url.trim().is_empty() {
        ProviderHandle::new(OpenAiProvider::new(api_key).into_components())
    } else {
        ProviderHandle::new(OpenAiCompatibleProvider::new(api_key, base_url).into_components())
    }
}

/// The `--api-key` / `OPENAI_API_KEY` / `OPENAI_COMPATIBLE_API_KEY` shortcut
/// that bypasses interactive onboarding for OpenAI-compatible endpoints.
pub(super) fn shortcut_api_key(args: &Args) -> Option<String> {
    args.api_key.clone().or_else(|| {
        if args.base_url.trim().is_empty() {
            std::env::var("OPENAI_API_KEY").ok()
        } else {
            std::env::var("OPENAI_COMPATIBLE_API_KEY")
                .ok()
                .or_else(|| std::env::var("OPENAI_API_KEY").ok())
        }
    })
}

/// Resolve the config and active provider. Runs the interactive setup wizard
/// only when setup is forced (`--provider`) or no config exists, and even
/// then prefers the non-interactive API-key shortcut when one is supplied.
pub(super) async fn resolve_config_and_provider(
    args: &Args,
    existing_config: Option<LashConfig>,
    shortcut_api_key: Option<&str>,
) -> anyhow::Result<(LashConfig, ProviderHandle)> {
    if args.provider || existing_config.is_none() {
        if let Some(key) = shortcut_api_key {
            let provider = openai_shortcut_provider(key.to_string(), &args.base_url);
            let mut cfg = existing_config.unwrap_or_else(|| LashConfig::new(&provider));
            cfg.upsert_provider(&provider);
            let _ = cfg.set_active_provider_kind(provider.kind());
            cfg.set_tavily_api_key(args.tavily_api_key.clone());
            return Ok((cfg, provider));
        }
        let cfg = onboarding::run_setup_with_existing(existing_config.as_ref()).await?;
        let provider = cfg
            .build_active_provider()
            .map_err(|err| anyhow::anyhow!("build active provider: {err}"))?;
        return Ok((cfg, provider));
    }
    let cfg = existing_config.expect("checked above: existing config present");
    let provider = cfg
        .build_active_provider()
        .map_err(|err| anyhow::anyhow!("build active provider: {err}"))?;
    Ok((cfg, provider))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn api_key_shortcut_selects_direct_openai_without_base_url() {
        let provider = openai_shortcut_provider("key".to_string(), "");
        assert_eq!(provider.kind(), "openai");
        assert_eq!(provider.to_spec().kind, "openai");
        assert!(provider.to_spec().config.get("base_url").is_none());
    }

    #[test]
    fn api_key_shortcut_selects_compatible_provider_with_base_url() {
        let provider = openai_shortcut_provider("key".to_string(), "https://example.invalid/v1");
        let spec = provider.to_spec();
        assert_eq!(provider.kind(), "openai-compatible");
        assert_eq!(spec.kind, "openai-compatible");
        assert_eq!(spec.config["base_url"], "https://example.invalid/v1");
    }
}
