use std::collections::HashMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::oauth::{self, OAuthError};

fn default_base_url() -> String {
    "https://openrouter.ai/api/v1".to_string()
}

/// User-overridable model names for delegate_task tiers.
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct DelegateModels {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quick: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub balanced: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thorough: Option<String>,
}

/// Stored configuration: provider credentials + service API keys.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LashConfig {
    pub provider: Provider,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tavily_api_key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub delegate_models: Option<DelegateModels>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Provider {
    OpenRouter {
        api_key: String,
        #[serde(default = "default_base_url")]
        base_url: String,
    },
    Claude {
        access_token: String,
        refresh_token: String,
        expires_at: u64,
    },
    Codex {
        access_token: String,
        refresh_token: String,
        expires_at: u64,
        account_id: Option<String>,
    },
}

impl Provider {
    /// Default model for this provider.
    pub fn default_model(&self) -> &str {
        match self {
            Provider::OpenRouter { .. } => "anthropic/claude-opus-4.6",
            Provider::Claude { .. } => "claude-opus-4-6",
            Provider::Codex { .. } => "gpt-5.1-codex",
        }
    }

    /// BAML provider string for ClientRegistry.
    pub fn baml_provider(&self) -> &str {
        match self {
            Provider::OpenRouter { .. } => "openai-generic",
            Provider::Claude { .. } => "anthropic",
            Provider::Codex { .. } => "openai-responses",
        }
    }

    /// Built-in model for a delegate tier. Returns (model_name, optional_reasoning_effort).
    /// For "thorough" on OpenRouter, returns None (caller should inherit parent model).
    pub fn default_delegate_model(&self, tier: &str) -> Option<(&str, Option<&str>)> {
        match (self, tier) {
            (Provider::Claude { .. }, "quick") => Some(("claude-haiku-4-5", None)),
            (Provider::Claude { .. }, "balanced") => Some(("claude-sonnet-4-5", None)),
            (Provider::Claude { .. }, "thorough") => Some(("claude-opus-4-6", None)),

            (Provider::OpenRouter { .. }, "quick") => Some(("minimax/minimax-m2.5", None)),
            (Provider::OpenRouter { .. }, "balanced") => Some(("z-ai/glm-5", None)),
            (Provider::OpenRouter { .. }, "thorough") => None, // inherit parent

            (Provider::Codex { .. }, "quick") => Some(("gpt-5.1-codex-mini", None)),
            (Provider::Codex { .. }, "balanced") => Some(("gpt-5.3-codex", Some("medium"))),
            (Provider::Codex { .. }, "thorough") => Some(("gpt-5.3-codex", Some("high"))),

            _ => None,
        }
    }

    /// Build BAML ClientRegistry options for this provider.
    pub fn baml_options(&self, model: &str, reasoning_effort: Option<&str>) -> HashMap<String, serde_json::Value> {
        match self {
            Provider::OpenRouter { api_key, base_url } => HashMap::from([
                ("base_url".into(), serde_json::json!(base_url)),
                ("api_key".into(), serde_json::json!(api_key)),
                ("model".into(), serde_json::json!(model)),
                ("temperature".into(), serde_json::json!(0)),
                ("max_tokens".into(), serde_json::json!(16384)),
            ]),
            Provider::Claude { access_token, .. } => HashMap::from([
                ("api_key".into(), serde_json::json!("noop")),
                ("model".into(), serde_json::json!(model)),
                ("temperature".into(), serde_json::json!(0)),
                ("max_tokens".into(), serde_json::json!(16384)),
                (
                    "headers".into(),
                    serde_json::json!({
                        "authorization": format!("Bearer {}", access_token),
                        "anthropic-beta": "oauth-2025-04-20,interleaved-thinking-2025-05-14",
                        "x-api-key": "",
                    }),
                ),
            ]),
            Provider::Codex {
                access_token,
                account_id,
                ..
            } => {
                let mut headers = serde_json::json!({
                    "authorization": format!("Bearer {}", access_token),
                    "originator": "lash",
                });
                if let Some(id) = account_id {
                    headers["chatgpt-account-id"] = serde_json::json!(id);
                }
                let mut opts = HashMap::from([
                    ("base_url".into(), serde_json::json!("https://chatgpt.com/backend-api/codex")),
                    ("api_key".into(), serde_json::json!("noop")),
                    ("model".into(), serde_json::json!(model)),
                    ("temperature".into(), serde_json::json!(0)),
                    ("max_tokens".into(), serde_json::json!(16384)),
                    ("headers".into(), headers),
                ]);
                if let Some(effort) = reasoning_effort {
                    opts.insert("reasoning".into(), serde_json::json!({"effort": effort}));
                }
                opts
            }
        }
    }

    /// Resolve model name: strip "anthropic/" prefix for direct Claude API.
    pub fn resolve_model(&self, model: &str) -> String {
        match self {
            Provider::Claude { .. } => model
                .strip_prefix("anthropic/")
                .unwrap_or(model)
                .to_string(),
            Provider::OpenRouter { .. } | Provider::Codex { .. } => model.to_string(),
        }
    }

    /// Refresh OAuth tokens if needed. No-op for OpenRouter.
    /// Returns `true` if tokens were updated (caller should persist).
    pub async fn ensure_fresh(&mut self) -> Result<bool, OAuthError> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        match self {
            Provider::Claude {
                access_token,
                refresh_token,
                expires_at,
            } => {
                if now + 300 >= *expires_at {
                    let tokens = oauth::refresh_tokens(refresh_token).await?;
                    *access_token = tokens.access_token;
                    *refresh_token = tokens.refresh_token;
                    *expires_at = tokens.expires_at;
                    return Ok(true);
                }
            }
            Provider::Codex {
                access_token,
                refresh_token,
                expires_at,
                ..
            } => {
                if now + 300 >= *expires_at {
                    let tokens = oauth::codex_refresh_tokens(refresh_token).await?;
                    *access_token = tokens.access_token;
                    *refresh_token = tokens.refresh_token;
                    *expires_at = tokens.expires_at;
                    return Ok(true);
                }
            }
            Provider::OpenRouter { .. } => {}
        }
        Ok(false)
    }
}

impl LashConfig {
    fn config_path() -> PathBuf {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
        PathBuf::from(home).join(".lash").join("config.json")
    }

    /// Load from ~/.lash/config.json (falls back to legacy provider.json)
    pub fn load() -> Option<Self> {
        let path = Self::config_path();
        if let Ok(data) = std::fs::read_to_string(&path) {
            if let Ok(config) = serde_json::from_str(&data) {
                return Some(config);
            }
        }

        // Migrate legacy provider.json
        let legacy = path.with_file_name("provider.json");
        if let Ok(data) = std::fs::read_to_string(&legacy) {
            if let Ok(provider) = serde_json::from_str::<Provider>(&data) {
                let config = LashConfig {
                    provider,
                    tavily_api_key: None,
                    delegate_models: None,
                };
                let _ = config.save();
                let _ = std::fs::remove_file(&legacy);
                return Some(config);
            }
        }

        None
    }

    /// Save to ~/.lash/config.json (mode 0o600)
    pub fn save(&self) -> Result<(), std::io::Error> {
        let path = Self::config_path();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let data = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(&path, &data)?;

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))?;
        }

        Ok(())
    }

    /// Delete ~/.lash/config.json
    pub fn clear() -> Result<(), std::io::Error> {
        let path = Self::config_path();
        if path.exists() {
            std::fs::remove_file(&path)?;
        }
        // Also clean up legacy file
        let legacy = path.with_file_name("provider.json");
        if legacy.exists() {
            std::fs::remove_file(&legacy)?;
        }
        Ok(())
    }
}

/// Save just the provider portion (preserves other config fields like API keys).
/// Used by the agent loop after token refresh.
pub fn save_provider(provider: &Provider) -> Result<(), std::io::Error> {
    let mut config = LashConfig::load().unwrap_or_else(|| LashConfig {
        provider: provider.clone(),
        tavily_api_key: None,
        delegate_models: None,
    });
    config.provider = provider.clone();
    config.save()
}

