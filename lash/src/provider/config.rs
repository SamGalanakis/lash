use super::support::*;

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

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct RuntimeSettings {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub explore_tier_subagent_execution_mode: Option<crate::ExecutionMode>,
}

impl RuntimeSettings {
    fn is_default(&self) -> bool {
        self.explore_tier_subagent_execution_mode.is_none()
    }
}

/// User-selected default model for fresh sessions, scoped to a provider kind.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelDefault {
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub variant: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProviderSpec {
    pub kind: String,
    pub config: serde_json::Value,
}

impl Serialize for ProviderSpec {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut value = match &self.config {
            serde_json::Value::Object(map) => serde_json::Value::Object(map.clone()),
            serde_json::Value::Null => serde_json::Value::Object(serde_json::Map::new()),
            other => {
                return Err(serde::ser::Error::custom(format!(
                    "ProviderSpec.config must serialize to a JSON object, got {}",
                    other
                )));
            }
        };
        if let serde_json::Value::Object(ref mut map) = value {
            map.insert(
                "type".to_string(),
                serde_json::Value::String(self.kind.clone()),
            );
        }
        value.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ProviderSpec {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let mut value = serde_json::Value::deserialize(deserializer)?;
        let kind = if let serde_json::Value::Object(ref mut map) = value {
            let raw = map
                .remove("type")
                .ok_or_else(|| serde::de::Error::missing_field("type"))?;
            raw.as_str()
                .ok_or_else(|| serde::de::Error::custom("provider `type` must be a string"))?
                .to_string()
        } else {
            return Err(serde::de::Error::custom(
                "provider spec must be a JSON object",
            ));
        };
        Ok(Self {
            kind,
            config: value,
        })
    }
}

/// Stored configuration: provider credentials + service API keys.
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
    #[serde(default, skip_serializing_if = "RuntimeSettings::is_default")]
    pub runtime: RuntimeSettings,
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
            runtime: RuntimeSettings::default(),
        }
    }

    pub fn from_spec(spec: ProviderSpec) -> Self {
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
            runtime: RuntimeSettings::default(),
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

    /// Materialize the active provider via the global registry.
    pub fn build_active_provider(&self) -> Result<ProviderHandle, String> {
        build_provider(self.active_provider_spec()).map(ProviderHandle::new)
    }

    /// Load from the given config path. Returns `None` if missing or
    /// malformed. Host decides where the file lives (e.g. lash-cli uses
    /// `~/.lash/config.json`).
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

    pub fn set_mcp_servers(&mut self, servers: BTreeMap<String, McpServerConfig>) {
        self.mcp_servers = servers;
    }

    /// Delete the config file at `path`.
    pub fn clear(path: &std::path::Path) -> Result<(), std::io::Error> {
        if path.exists() {
            std::fs::remove_file(path)?;
        }
        Ok(())
    }
}
