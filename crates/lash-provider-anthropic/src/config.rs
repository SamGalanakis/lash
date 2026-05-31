//! Provider construction: the [`AnthropicProvider`] struct, its builders, and
//! the [`AnthropicProviderFactory`] that materializes one from a stored config.

use crate::policy::AnthropicModelPolicy;
use crate::support::*;

pub const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";

/// Anthropic API (Claude) provider state and transport.
#[derive(Clone, Debug)]
pub struct AnthropicProvider {
    pub api_key: String,
    pub base_url: Option<String>,
    pub options: ProviderOptions,
    pub(crate) client: reqwest::Client,
}

impl AnthropicProvider {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: None,
            options: ProviderOptions::default(),
            client: build_http_client(),
        }
    }

    pub fn with_base_url(mut self, base_url: Option<String>) -> Self {
        self.base_url = base_url;
        self
    }

    pub fn with_options(mut self, options: ProviderOptions) -> Self {
        self.options = options;
        self
    }

    /// Share an embedder-provided `reqwest::Client` instead of building
    /// a fresh one. Saves ~42 MB of TLS state per provider when the
    /// host pools connections across sessions.
    pub fn with_client(mut self, client: std::sync::Arc<reqwest::Client>) -> Self {
        self.client = (*client).clone();
        self
    }

    pub fn into_components(self) -> ProviderComponents {
        ProviderComponents::new(Box::new(self), std::sync::Arc::new(AnthropicModelPolicy))
    }
}

/// Deserialize payload for `ProviderSpec::config` when building an
/// `AnthropicProvider` from a stored [`lash_core::LashConfig`].
#[derive(Deserialize)]
struct AnthropicProviderConfig {
    api_key: String,
    #[serde(default)]
    base_url: Option<String>,
    #[serde(default)]
    options: ProviderOptions,
}

/// Factory that materializes [`AnthropicProvider`] from a host-owned
/// [`ProviderSpec`](lash_core::ProviderSpec).
pub struct AnthropicProviderFactory;

impl ProviderFactory for AnthropicProviderFactory {
    fn kind(&self) -> &'static str {
        "anthropic"
    }
    fn deserialize(&self, config: serde_json::Value) -> Result<ProviderComponents, String> {
        let cfg: AnthropicProviderConfig =
            serde_json::from_value(config).map_err(|err| err.to_string())?;
        Ok(AnthropicProvider {
            api_key: cfg.api_key,
            base_url: cfg.base_url,
            options: cfg.options,
            client: build_http_client(),
        }
        .into_components())
    }
}
