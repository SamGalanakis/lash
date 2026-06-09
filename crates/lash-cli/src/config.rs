//! User config-file schema (`~/.lash/config.json`) and helpers.
//!
//! `LashConfig` owns the data shape that the CLI persists to disk. The lash
//! runtime itself takes already-built primitives (`ProviderHandle`, plugin
//! factories) — it does not load this file.

use std::collections::BTreeMap;

use lash_core::{ProviderFactory, ProviderHandle, ProviderSpec};
use lash_plugin_mcp::McpServerConfig;
use lash_provider_anthropic::AnthropicProviderFactory;
use lash_provider_google::GoogleOAuthProviderFactory;
use lash_provider_openai::{
    CodexProviderFactory, OpenAiCompatibleProviderFactory, OpenAiProviderFactory,
};
use serde::{Deserialize, Serialize};

/// Auxiliary service secrets that are independent of LLM provider auth.
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct AuxiliarySecrets {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tavily_api_key: Option<String>,
}

impl AuxiliarySecrets {
    fn is_empty(&self) -> bool {
        self.tavily_api_key.is_none()
    }
}

/// User-selected default model for fresh sessions, scoped to a provider kind.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelDefault {
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub variant: Option<String>,
}

/// Stored configuration: provider credentials + service API keys + MCP
/// servers + per-session defaults.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LashConfig {
    pub active_provider: String,
    pub providers: BTreeMap<String, ProviderSpec>,
    #[serde(default, skip_serializing_if = "AuxiliarySecrets::is_empty")]
    pub auxiliary_secrets: AuxiliarySecrets,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub mcp_servers: BTreeMap<String, McpServerConfig>,
    /// User-overridable model names per subagent capability. Generic
    /// name → model map; the meaning of each name is owned by whatever
    /// builds the subagent capability registry.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub agent_models: BTreeMap<String, String>,
    /// Fresh-session model defaults keyed by provider kind. Session
    /// resumes still use the session head's persisted model instead.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub model_defaults: BTreeMap<String, ModelDefault>,
}

impl LashConfig {
    /// Construct a config from an already-built provider handle. The
    /// provider's current spec becomes the active entry.
    pub fn new(provider: &ProviderHandle) -> Self {
        let spec = provider.to_spec();
        let kind = spec.kind.clone();
        let mut providers = BTreeMap::new();
        providers.insert(kind.clone(), spec);
        Self {
            active_provider: kind,
            providers,
            auxiliary_secrets: AuxiliarySecrets::default(),
            mcp_servers: BTreeMap::new(),
            agent_models: BTreeMap::new(),
            model_defaults: BTreeMap::new(),
        }
    }

    pub fn active_provider_spec(&self) -> &ProviderSpec {
        self.providers
            .get(&self.active_provider)
            .expect("active provider missing from config")
    }

    pub fn active_provider_kind(&self) -> &str {
        &self.active_provider
    }

    pub fn set_active_provider_kind(&mut self, kind: &str) -> Result<(), String> {
        if !self.providers.contains_key(kind) {
            return Err(format!("provider `{}` is not configured", kind));
        }
        self.active_provider = kind.to_string();
        Ok(())
    }

    pub fn provider_spec(&self, kind: &str) -> Option<&ProviderSpec> {
        self.providers.get(kind)
    }

    pub fn provider_kinds(&self) -> Vec<String> {
        self.providers.keys().cloned().collect()
    }

    pub fn has_provider(&self, kind: &str) -> bool {
        self.providers.contains_key(kind)
    }

    pub fn upsert_provider_spec(&mut self, spec: ProviderSpec) {
        self.providers.insert(spec.kind.clone(), spec);
    }

    pub fn upsert_provider(&mut self, provider: &ProviderHandle) {
        self.upsert_provider_spec(provider.to_spec());
    }

    pub fn remove_provider(&mut self, kind: &str) -> Option<ProviderSpec> {
        let removed = self.providers.remove(kind)?;
        if self.providers.is_empty() {
            return Some(removed);
        }
        if self.active_provider == kind {
            self.active_provider = self
                .providers
                .keys()
                .next()
                .cloned()
                .expect("providers should be non-empty after removal");
        }
        Some(removed)
    }

    pub fn provider_count(&self) -> usize {
        self.providers.len()
    }

    pub fn model_default(&self, provider_kind: &str) -> Option<&ModelDefault> {
        self.model_defaults.get(provider_kind)
    }

    pub fn set_model_default(
        &mut self,
        provider_kind: impl Into<String>,
        model: impl Into<String>,
        variant: Option<String>,
    ) {
        self.model_defaults.insert(
            provider_kind.into(),
            ModelDefault {
                model: model.into(),
                variant,
            },
        );
    }

    /// Materialize the active provider with the providers compiled into the CLI.
    pub fn build_active_provider(&self) -> Result<ProviderHandle, String> {
        materialize_provider_spec(self.active_provider_spec())
    }

    /// Load from the given config path. Returns `None` if missing or
    /// malformed.
    pub fn load(path: &std::path::Path) -> Option<Self> {
        if let Ok(data) = std::fs::read_to_string(path)
            && let Ok(config) = serde_json::from_str::<Self>(&data)
            && config.providers.contains_key(&config.active_provider)
        {
            return Some(config);
        }

        None
    }

    /// Save to the given config path (mode 0o600 on Unix).
    pub fn save(&self, path: &std::path::Path) -> Result<(), std::io::Error> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let data = serde_json::to_string_pretty(self).map_err(std::io::Error::other)?;
        std::fs::write(path, &data)?;

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))?;
        }

        Ok(())
    }

    pub fn tavily_api_key(&self) -> Option<&str> {
        self.auxiliary_secrets.tavily_api_key.as_deref()
    }

    pub fn set_tavily_api_key(&mut self, key: Option<String>) {
        self.auxiliary_secrets.tavily_api_key = key;
    }

    pub fn mcp_servers(&self) -> &BTreeMap<String, McpServerConfig> {
        &self.mcp_servers
    }

    /// Delete the config file at `path`.
    pub fn clear(path: &std::path::Path) -> Result<(), std::io::Error> {
        if path.exists() {
            std::fs::remove_file(path)?;
        }
        Ok(())
    }
}

pub(crate) fn materialize_provider_spec(spec: &ProviderSpec) -> Result<ProviderHandle, String> {
    let components = match spec.kind.as_str() {
        "anthropic" => AnthropicProviderFactory.deserialize(spec.config.clone()),
        "openai" => OpenAiProviderFactory.deserialize(spec.config.clone()),
        "openai-compatible" => OpenAiCompatibleProviderFactory.deserialize(spec.config.clone()),
        "codex" => CodexProviderFactory.deserialize(spec.config.clone()),
        "google_oauth" => GoogleOAuthProviderFactory.deserialize(spec.config.clone()),
        #[cfg(feature = "test-provider")]
        "test" => return materialize_test_provider_spec(spec),
        other => Err(format!(
            "provider `{}` is not supported by this CLI build",
            other
        )),
    }?;
    Ok(ProviderHandle::new(components))
}

#[cfg(feature = "test-provider")]
fn materialize_test_provider_spec(spec: &ProviderSpec) -> Result<ProviderHandle, String> {
    let scenario = spec
        .config
        .get("scenario")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("rlm-subagent-smoke");
    match scenario {
        "standard-echo" => Ok(standard_echo_provider().into_handle()),
        "rlm-subagent-smoke" => Ok(rlm_subagent_smoke_provider().into_handle()),
        other => Err(format!("unknown CLI test provider scenario `{other}`")),
    }
}

#[cfg(feature = "test-provider")]
fn standard_echo_provider() -> lash_core::testing::TestProvider {
    lash_core::testing::TestProvider::builder()
        .kind("test")
        .serialize_config(|| {
            serde_json::json!({
                "scenario": "standard-echo",
            })
        })
        .complete(|request| async move {
            let prompt = if request_contains_text(&request, "hello from pty") {
                "hello from pty"
            } else {
                "interactive prompt"
            };
            let response = format!("test-provider echo: {prompt}");
            Ok(lash_core::LlmResponse {
                full_text: response.clone(),
                parts: vec![lash_core::llm::types::LlmOutputPart::Text {
                    text: response,
                    response_meta: None,
                }],
                ..Default::default()
            })
        })
        .build()
}

#[cfg(feature = "test-provider")]
fn rlm_subagent_smoke_provider() -> lash_core::testing::TestProvider {
    lash_core::testing::TestProvider::builder()
        .kind("test")
        .serialize_config(|| {
            serde_json::json!({
                "scenario": "rlm-subagent-smoke",
            })
        })
        .complete(|request| async move {
            let response = if request_contains_subagent_prompt(&request) {
                r#"```lashlang
submit { value: "subagent-ok" }
```"#
            } else {
                r#"```lashlang
result = await agents.spawn({
  capability: "explore",
  task: "Submit `{ value: \"subagent-ok\" }` exactly.",
  output: Type { value: str }
})?
submit result.value
```"#
            };
            Ok(lash_core::LlmResponse {
                full_text: response.to_string(),
                parts: vec![lash_core::llm::types::LlmOutputPart::Text {
                    text: response.to_string(),
                    response_meta: None,
                }],
                ..Default::default()
            })
        })
        .build()
}

#[cfg(feature = "test-provider")]
fn request_contains_text(request: &lash_core::LlmRequest, needle: &str) -> bool {
    request.messages.iter().any(|message| {
        message.blocks.iter().any(|part| match part {
            lash_core::llm::types::LlmContentBlock::Text { text, .. } => text.contains(needle),
            _ => false,
        })
    })
}

#[cfg(feature = "test-provider")]
fn request_contains_subagent_prompt(request: &lash_core::LlmRequest) -> bool {
    request_contains_text(request, "Subagent capability: explore. Depth: 1/5.")
        || request_contains_text(request, "Submit `{ value: \\\"subagent-ok\\\" }` exactly.")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lash_config_roundtrips_existing_shape() {
        let raw = serde_json::json!({
            "active_provider": "openai",
            "providers": {
                "openai": {
                    "type": "openai",
                    "api_key": "direct-k"
                },
                "openai-compatible": {
                    "type": "openai-compatible",
                    "api_key": "k",
                    "base_url": "https://example.com/v1"
                }
            }
        });
        let cfg: LashConfig = serde_json::from_value(raw).expect("valid config");
        assert_eq!(cfg.active_provider, "openai");
        let spec = cfg.active_provider_spec();
        assert_eq!(spec.kind, "openai");
        assert_eq!(spec.config["api_key"], serde_json::json!("direct-k"));
        let compatible = cfg.provider_spec("openai-compatible").expect("compatible");
        assert_eq!(compatible.config["base_url"], "https://example.com/v1");
    }

    #[test]
    fn rejects_unknown_top_level_config_fields() {
        let raw = serde_json::json!({
            "active_provider": "openai-compatible",
            "providers": {
                "openai-compatible": {
                    "type": "openai-compatible",
                    "api_key": "k",
                    "base_url": "https://example.com/v1"
                }
            },
            "tavily_api_key": "legacy-key"
        });
        let err = serde_json::from_value::<LashConfig>(raw).expect_err("unknown field rejected");
        assert!(err.to_string().contains("unknown field `tavily_api_key`"));
    }

    #[test]
    fn auxiliary_secrets_preserved() {
        let raw = serde_json::json!({
            "active_provider": "openai-compatible",
            "providers": {
                "openai-compatible": {
                    "type": "openai-compatible",
                    "api_key": "k",
                    "base_url": "https://example.com/v1"
                }
            },
            "auxiliary_secrets": {
                "tavily_api_key": "new-key"
            }
        });
        let cfg: LashConfig = serde_json::from_value(raw).expect("valid config json");
        assert_eq!(cfg.tavily_api_key(), Some("new-key"));
    }

    #[test]
    fn model_defaults_are_provider_scoped() {
        let raw = serde_json::json!({
            "active_provider": "openai-compatible",
            "providers": {
                "openai-compatible": {
                    "type": "openai-compatible",
                    "api_key": "k",
                    "base_url": "https://example.com/v1"
                }
            },
            "model_defaults": {
                "openai-compatible": {
                    "model": "gpt-5.4",
                    "variant": "high"
                }
            }
        });
        let mut cfg: LashConfig = serde_json::from_value(raw).expect("valid config json");
        assert_eq!(
            cfg.model_default("openai-compatible"),
            Some(&ModelDefault {
                model: "gpt-5.4".to_string(),
                variant: Some("high".to_string()),
            })
        );

        cfg.set_model_default("anthropic", "claude-sonnet-4.6", None);
        assert_eq!(
            cfg.model_default("anthropic"),
            Some(&ModelDefault {
                model: "claude-sonnet-4.6".to_string(),
                variant: None,
            })
        );
    }
}
