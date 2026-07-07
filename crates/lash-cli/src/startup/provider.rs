//! Startup stage: resolve the on-disk config and the active provider's
//! credentials. Interactive onboarding runs only when there is no usable
//! stored/shortcut credential (or `--provider` forces it).

use crate::Args;
use crate::config::{ConfigLoadOutcome, LashConfig};
use lash::provider::ProviderHandle;
use lash_provider_openai::{OPENROUTER_BASE_URL, OpenAiCompatibleProvider, OpenAiProvider};

use super::onboarding;

pub(super) fn openai_shortcut_provider(api_key: String, base_url: &str) -> ProviderHandle {
    if base_url.trim().is_empty() {
        ProviderHandle::new(OpenAiProvider::new(api_key).into_components())
    } else {
        ProviderHandle::new(OpenAiCompatibleProvider::new(api_key, base_url).into_components())
    }
}

fn base_url_is_openrouter(base_url: &str) -> bool {
    base_url.trim_end_matches('/') == OPENROUTER_BASE_URL
}

fn compatible_env_api_key(base_url: &str) -> Option<String> {
    std::env::var("OPENAI_COMPATIBLE_API_KEY")
        .ok()
        .or_else(|| std::env::var("OPENAI_API_KEY").ok())
        .or_else(|| {
            if base_url_is_openrouter(base_url) {
                std::env::var("OPENROUTER_API_KEY").ok()
            } else {
                None
            }
        })
}

/// The `--api-key` / `OPENAI_API_KEY` / `OPENAI_COMPATIBLE_API_KEY` shortcut
/// that bypasses interactive onboarding for OpenAI-compatible endpoints.
/// When `--base-url` points at OpenRouter, `OPENROUTER_API_KEY` is also accepted.
pub(super) fn shortcut_api_key(args: &Args) -> Option<String> {
    args.api_key.clone().or_else(|| {
        if args.base_url.trim().is_empty() {
            std::env::var("OPENAI_API_KEY").ok()
        } else {
            compatible_env_api_key(&args.base_url)
        }
    })
}

fn no_usable_config_message(path: &std::path::Path, load_outcome: &ConfigLoadOutcome) -> String {
    let cause = match load_outcome {
        ConfigLoadOutcome::Missing => "config file not found".to_string(),
        ConfigLoadOutcome::Invalid { reason } => format!("config present but invalid: {reason}"),
        ConfigLoadOutcome::Loaded(_) => unreachable!("loaded config is usable"),
    };
    format!(
        "no usable lash config at {}\n\
         caused by: {cause}\n\
         run `lash --provider` in an interactive terminal, pass `--api-key`, \
         or set `OPENAI_COMPATIBLE_API_KEY` (or `OPENROUTER_API_KEY` with `--base-url` pointing at OpenRouter)",
        path.display()
    )
}

/// Resolve the config and active provider. Runs the interactive setup wizard
/// only when setup is forced (`--provider`) or no config exists, and even
/// then prefers the non-interactive API-key shortcut when one is supplied.
pub(super) async fn resolve_config_and_provider(
    args: &Args,
    config_path: &std::path::Path,
    load_outcome: ConfigLoadOutcome,
    shortcut_api_key: Option<&str>,
    allow_onboarding: bool,
) -> anyhow::Result<(LashConfig, ProviderHandle)> {
    let existing_config = load_outcome.loaded().cloned();
    if args.provider || existing_config.is_none() {
        if let Some(key) = shortcut_api_key {
            let provider = openai_shortcut_provider(key.to_string(), &args.base_url);
            let mut cfg = existing_config.unwrap_or_else(|| LashConfig::new(&provider));
            cfg.upsert_provider(&provider);
            let _ = cfg.set_active_provider_kind(provider.kind());
            cfg.set_tavily_api_key(args.tavily_api_key.clone());
            return Ok((cfg, provider));
        }
        if args.provider && !allow_onboarding {
            return Err(anyhow::anyhow!(
                "`--provider` cannot be used with `--print`, `--mode json`, or `--mode rpc`; \
                 run provider setup from an interactive terminal instead"
            ));
        }
        if !allow_onboarding {
            return Err(anyhow::anyhow!(no_usable_config_message(
                config_path,
                &load_outcome,
            )));
        }
        crate::util::require_interactive_terminal("provider setup")?;
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
    use crate::test_support::env_lock;
    use clap::Parser as _;

    fn with_env_vars(vars: &[(&str, Option<&str>)], f: impl FnOnce()) {
        let _env_guard = env_lock().blocking_lock();
        let previous: Vec<_> = vars
            .iter()
            .map(|(key, _)| (*key, std::env::var_os(key)))
            .collect();
        for (key, value) in vars {
            match value {
                Some(value) => unsafe { std::env::set_var(key, value) },
                None => unsafe { std::env::remove_var(key) },
            }
        }
        f();
        for (key, value) in previous {
            match value {
                Some(value) => unsafe { std::env::set_var(key, value) },
                None => unsafe { std::env::remove_var(key) },
            }
        }
    }

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

    #[test]
    fn openrouter_env_key_is_used_for_openrouter_base_url() {
        with_env_vars(
            &[
                ("OPENAI_API_KEY", None),
                ("OPENAI_COMPATIBLE_API_KEY", None),
                ("OPENROUTER_API_KEY", Some("or-key")),
            ],
            || {
                let args = crate::Args::try_parse_from([
                    "lash",
                    "--base-url",
                    "https://openrouter.ai/api/v1",
                ])
                .expect("parse args");
                assert_eq!(shortcut_api_key(&args).as_deref(), Some("or-key"));
            },
        );
    }

    #[test]
    fn openrouter_env_key_is_ignored_without_openrouter_base_url() {
        with_env_vars(
            &[
                ("OPENAI_API_KEY", None),
                ("OPENAI_COMPATIBLE_API_KEY", None),
                ("OPENROUTER_API_KEY", Some("or-key")),
            ],
            || {
                let args = crate::Args::try_parse_from([
                    "lash",
                    "--base-url",
                    "https://example.invalid/v1",
                ])
                .expect("parse args");
                assert!(shortcut_api_key(&args).is_none());
            },
        );
    }

    #[test]
    fn no_usable_config_message_describes_invalid_config() {
        let path = std::path::Path::new("/tmp/lash/config.json");
        let message = no_usable_config_message(
            path,
            &ConfigLoadOutcome::Invalid {
                reason: "invalid config JSON: unknown field `theme`".to_string(),
            },
        );
        assert!(message.contains("/tmp/lash/config.json"));
        assert!(message.contains("config present but invalid"));
        assert!(message.contains("unknown field `theme`"));
    }

    #[tokio::test]
    async fn autonomous_startup_rejects_missing_config_without_onboarding() {
        let args = crate::Args::try_parse_from(["lash", "--print", "hello"]).expect("parse args");
        let err = resolve_config_and_provider(
            &args,
            std::path::Path::new("/tmp/lash/config.json"),
            ConfigLoadOutcome::Missing,
            None,
            false,
        )
        .await
        .expect_err("missing config should fail fast");
        assert!(err.to_string().contains("no usable lash config"));
        assert!(err.to_string().contains("config file not found"));
    }

    #[tokio::test]
    async fn autonomous_startup_rejects_provider_flag_without_onboarding() {
        let args =
            crate::Args::try_parse_from(["lash", "--provider", "--print", "hello"]).expect("parse");
        let err = resolve_config_and_provider(
            &args,
            std::path::Path::new("/tmp/lash/config.json"),
            ConfigLoadOutcome::Loaded(LashConfig::new(&openai_shortcut_provider(
                "key".into(),
                "",
            ))),
            None,
            false,
        )
        .await
        .expect_err("--provider should fail fast headlessly");
        assert!(err.to_string().contains("`--provider` cannot be used"));
    }
}
