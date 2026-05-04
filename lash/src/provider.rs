//! Provider components and registry for pluggable LLM backends.
//!
//! A provider is split into narrow capabilities: persisted state,
//! authentication refresh, readiness/warmup, request transport, and
//! model policy. [`ProviderHandle`] owns those components and is the
//! persisted handle stored by session policy and config.
//!
//! Serialization: [`ProviderHandle`] is the owning handle that
//! [`SessionPolicy`] stores. It round-trips through [`ProviderSpec`] —
//! a `{ "type": kind, …config }` JSON object whose shape matches the
//! legacy `#[serde(tag = "type")]` enum exactly, so existing
//! `~/.lash/config.json` files load without migration.

use std::collections::BTreeMap;
use std::sync::{Arc, LazyLock, Mutex, RwLock};
use std::time::Duration;

use async_trait::async_trait;
use serde::de::{self, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::llm::timeouts::{DEFAULT_CHUNK_TIMEOUT_MS, DEFAULT_REQUEST_TIMEOUT_MS, LlmTimeouts};

use crate::llm::transport::LlmTransportError;
use crate::llm::types::{LlmRequest, LlmResponse};
use crate::mcp::McpServerConfig;
use crate::model_info::{ModelCatalog, ResolvedModelSpec};
use crate::oauth::OAuthError;

/// Per-request tuning a provider produces for a model + variant. Each
/// concrete provider crate interprets its own variant strings and emits
/// the request-shaping parameters its wire protocol needs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VariantRequestConfig {
    ReasoningEffort(String),
    GoogleThinkingLevel { level: String },
    GoogleThinkingBudget { budget_tokens: i32 },
    AnthropicAdaptiveThinking { effort: String },
    AnthropicThinkingBudget { budget_tokens: i32 },
}

/// Model + optional variant returned by provider model policies for agent tiers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AgentModelSelection {
    pub model: String,
    pub variant: Option<String>,
}

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RequestTimeout {
    Disabled,
    Millis(u64),
}

impl Serialize for RequestTimeout {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Disabled => serializer.serialize_bool(false),
            Self::Millis(value) => serializer.serialize_u64(*value),
        }
    }
}

impl<'de> Deserialize<'de> for RequestTimeout {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct RequestTimeoutVisitor;

        impl Visitor<'_> for RequestTimeoutVisitor {
            type Value = RequestTimeout;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("a positive timeout in milliseconds or false")
            }

            fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                if value {
                    return Err(E::custom("timeout must be a positive integer or false"));
                }
                Ok(RequestTimeout::Disabled)
            }

            fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                if value == 0 {
                    return Err(E::custom("timeout must be greater than 0"));
                }
                Ok(RequestTimeout::Millis(value))
            }

            fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                if value <= 0 {
                    return Err(E::custom("timeout must be greater than 0"));
                }
                Ok(RequestTimeout::Millis(value as u64))
            }
        }

        deserializer.deserialize_any(RequestTimeoutVisitor)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct ProviderOptions {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout: Option<RequestTimeout>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chunk_timeout: Option<u64>,
}

impl ProviderOptions {
    pub fn is_default(&self) -> bool {
        self.timeout.is_none() && self.chunk_timeout.is_none()
    }

    pub fn llm_timeouts(&self) -> LlmTimeouts {
        let request_timeout = match self.timeout {
            Some(RequestTimeout::Disabled) => None,
            Some(RequestTimeout::Millis(ms)) => Some(Duration::from_millis(ms)),
            None => Some(Duration::from_millis(DEFAULT_REQUEST_TIMEOUT_MS)),
        };
        let chunk_timeout_ms = self
            .chunk_timeout
            .filter(|value| *value > 0)
            .unwrap_or(DEFAULT_CHUNK_TIMEOUT_MS);
        LlmTimeouts {
            request_timeout,
            chunk_timeout: Duration::from_millis(chunk_timeout_ms),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
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

/// Persisted provider state: stable kind, serializable config, and
/// request timeout options.
pub trait ProviderState: Send + Sync + std::fmt::Debug {
    fn kind(&self) -> &'static str;

    fn options(&self) -> ProviderOptions;
    fn set_options(&mut self, options: ProviderOptions);

    /// Emit the provider-specific JSON body used by [`ProviderSpec`]. The
    /// object must NOT contain a `type` field — [`ProviderSpec::Serialize`]
    /// layers that on top.
    fn serialize_config(&self) -> serde_json::Value;

    fn clone_boxed(&self) -> Box<dyn ProviderState>;
}

#[async_trait]
pub trait ProviderAuth: Send + Sync + std::fmt::Debug {
    /// Refresh OAuth tokens if needed. Returns `true` if persisted
    /// credentials changed.
    async fn ensure_fresh(&mut self) -> Result<bool, OAuthError>;

    fn clone_boxed(&self) -> Box<dyn ProviderAuth>;
}

#[async_trait]
pub trait ProviderReadiness: Send + Sync + std::fmt::Debug {
    /// Adapter-level warmup (project-id discovery, handshake, etc.).
    /// Returns `true` if provider state changed and should be persisted.
    async fn ensure_ready(&mut self) -> Result<bool, LlmTransportError>;

    fn requires_streaming(&self) -> bool;

    fn clone_boxed(&self) -> Box<dyn ProviderReadiness>;
}

#[async_trait]
pub trait ProviderTransport: Send + Sync + std::fmt::Debug {
    async fn complete(&mut self, request: LlmRequest) -> Result<LlmResponse, LlmTransportError>;

    fn clone_boxed(&self) -> Box<dyn ProviderTransport>;
}

pub trait ProviderModelPolicy: Send + Sync + std::fmt::Debug {
    fn default_model(&self) -> &str;
    fn supported_variants(&self, model: &str) -> &'static [&'static str];
    fn default_model_variant(&self, model: &str) -> Option<&'static str>;
    fn request_variant_config(&self, model: &str, variant: &str) -> Option<VariantRequestConfig>;
    fn default_agent_model(&self, tier: &str) -> Option<AgentModelSelection>;

    fn resolve_model(&self, model: &str) -> String {
        model.to_string()
    }

    fn context_lookup_model(&self, model: &str) -> String {
        model.to_string()
    }

    fn input_usage_excludes_cached_tokens(&self) -> bool {
        false
    }
}

/// Provider auth component for API-key providers.
#[derive(Clone, Debug, Default)]
pub struct NoopProviderAuth;

#[async_trait]
impl ProviderAuth for NoopProviderAuth {
    async fn ensure_fresh(&mut self) -> Result<bool, OAuthError> {
        Ok(false)
    }

    fn clone_boxed(&self) -> Box<dyn ProviderAuth> {
        Box::new(self.clone())
    }
}

/// Readiness component for providers with no warmup requirement.
#[derive(Clone, Debug, Default)]
pub struct NoopProviderReadiness {
    requires_streaming: bool,
}

impl NoopProviderReadiness {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn requiring_streaming() -> Self {
        Self {
            requires_streaming: true,
        }
    }
}

#[async_trait]
impl ProviderReadiness for NoopProviderReadiness {
    async fn ensure_ready(&mut self) -> Result<bool, LlmTransportError> {
        Ok(false)
    }

    fn requires_streaming(&self) -> bool {
        self.requires_streaming
    }

    fn clone_boxed(&self) -> Box<dyn ProviderReadiness> {
        Box::new(self.clone())
    }
}

/// Static model policy for simple providers and tests.
#[derive(Clone, Debug)]
pub struct StaticModelPolicy {
    default_model: &'static str,
    supported_variants: &'static [&'static str],
    default_variant: Option<&'static str>,
}

impl StaticModelPolicy {
    pub fn new(default_model: &'static str) -> Self {
        Self {
            default_model,
            supported_variants: &[],
            default_variant: None,
        }
    }

    pub fn with_variants(
        default_model: &'static str,
        supported_variants: &'static [&'static str],
        default_variant: Option<&'static str>,
    ) -> Self {
        Self {
            default_model,
            supported_variants,
            default_variant,
        }
    }
}

impl ProviderModelPolicy for StaticModelPolicy {
    fn default_model(&self) -> &str {
        self.default_model
    }

    fn supported_variants(&self, _model: &str) -> &'static [&'static str] {
        self.supported_variants
    }

    fn default_model_variant(&self, _model: &str) -> Option<&'static str> {
        self.default_variant
    }

    fn request_variant_config(&self, _model: &str, variant: &str) -> Option<VariantRequestConfig> {
        self.supported_variants
            .contains(&variant)
            .then(|| VariantRequestConfig::ReasoningEffort(variant.to_string()))
    }

    fn default_agent_model(&self, _tier: &str) -> Option<AgentModelSelection> {
        None
    }
}

#[derive(Debug)]
struct SharedProviderComponent<T> {
    inner: Arc<Mutex<T>>,
}

impl<T> Clone for SharedProviderComponent<T> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<T> SharedProviderComponent<T> {
    fn new(inner: Arc<Mutex<T>>) -> Self {
        Self { inner }
    }
}

impl<T> ProviderState for SharedProviderComponent<T>
where
    T: ProviderState + Clone + Send + Sync + std::fmt::Debug + 'static,
{
    fn kind(&self) -> &'static str {
        self.inner.lock().expect("provider state lock").kind()
    }

    fn options(&self) -> ProviderOptions {
        self.inner.lock().expect("provider state lock").options()
    }

    fn set_options(&mut self, options: ProviderOptions) {
        self.inner
            .lock()
            .expect("provider state lock")
            .set_options(options);
    }

    fn serialize_config(&self) -> serde_json::Value {
        self.inner
            .lock()
            .expect("provider state lock")
            .serialize_config()
    }

    fn clone_boxed(&self) -> Box<dyn ProviderState> {
        Box::new(self.clone())
    }
}

#[async_trait]
impl<T> ProviderAuth for SharedProviderComponent<T>
where
    T: ProviderAuth + Clone + Send + Sync + std::fmt::Debug + 'static,
{
    async fn ensure_fresh(&mut self) -> Result<bool, OAuthError> {
        let mut provider = self.inner.lock().expect("provider auth lock").clone();
        let result = provider.ensure_fresh().await;
        *self.inner.lock().expect("provider auth lock") = provider;
        result
    }

    fn clone_boxed(&self) -> Box<dyn ProviderAuth> {
        Box::new(self.clone())
    }
}

#[async_trait]
impl<T> ProviderReadiness for SharedProviderComponent<T>
where
    T: ProviderReadiness + Clone + Send + Sync + std::fmt::Debug + 'static,
{
    async fn ensure_ready(&mut self) -> Result<bool, LlmTransportError> {
        let mut provider = self.inner.lock().expect("provider readiness lock").clone();
        let result = provider.ensure_ready().await;
        *self.inner.lock().expect("provider readiness lock") = provider;
        result
    }

    fn requires_streaming(&self) -> bool {
        self.inner
            .lock()
            .expect("provider readiness lock")
            .requires_streaming()
    }

    fn clone_boxed(&self) -> Box<dyn ProviderReadiness> {
        Box::new(self.clone())
    }
}

#[async_trait]
impl<T> ProviderTransport for SharedProviderComponent<T>
where
    T: ProviderTransport + Clone + Send + Sync + std::fmt::Debug + 'static,
{
    async fn complete(&mut self, request: LlmRequest) -> Result<LlmResponse, LlmTransportError> {
        let mut provider = self.inner.lock().expect("provider transport lock").clone();
        let result = provider.complete(request).await;
        *self.inner.lock().expect("provider transport lock") = provider;
        result
    }

    fn clone_boxed(&self) -> Box<dyn ProviderTransport> {
        Box::new(self.clone())
    }
}

/// Component bundle returned by provider factories.
#[derive(Debug)]
pub struct ProviderComponents {
    pub state: Box<dyn ProviderState>,
    pub auth: Box<dyn ProviderAuth>,
    pub readiness: Box<dyn ProviderReadiness>,
    pub transport: Box<dyn ProviderTransport>,
    pub model_policy: Arc<dyn ProviderModelPolicy>,
}

impl ProviderComponents {
    pub fn new(
        state: Box<dyn ProviderState>,
        auth: Box<dyn ProviderAuth>,
        readiness: Box<dyn ProviderReadiness>,
        transport: Box<dyn ProviderTransport>,
        model_policy: Arc<dyn ProviderModelPolicy>,
    ) -> Self {
        Self {
            state,
            auth,
            readiness,
            transport,
            model_policy,
        }
    }

    pub fn shared<T>(provider: T, model_policy: Arc<dyn ProviderModelPolicy>) -> Self
    where
        T: ProviderState
            + ProviderAuth
            + ProviderReadiness
            + ProviderTransport
            + Clone
            + Send
            + Sync
            + std::fmt::Debug
            + 'static,
    {
        let inner = Arc::new(Mutex::new(provider));
        Self {
            state: Box::new(SharedProviderComponent::new(Arc::clone(&inner))),
            auth: Box::new(SharedProviderComponent::new(Arc::clone(&inner))),
            readiness: Box::new(SharedProviderComponent::new(Arc::clone(&inner))),
            transport: Box::new(SharedProviderComponent::new(inner)),
            model_policy,
        }
    }

    pub fn map_transport(
        mut self,
        map: impl FnOnce(Box<dyn ProviderTransport>) -> Box<dyn ProviderTransport>,
    ) -> Self {
        self.transport = map(self.transport);
        self
    }
}

impl Clone for ProviderComponents {
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone_boxed(),
            auth: self.auth.clone_boxed(),
            readiness: self.readiness.clone_boxed(),
            transport: self.transport.clone_boxed(),
            model_policy: Arc::clone(&self.model_policy),
        }
    }
}

/// Owning handle to provider components. Session state + config store this
/// so we can add Clone / Serialize / Deserialize impls without running
/// into orphan-rule conflicts.
pub struct ProviderHandle {
    components: ProviderComponents,
}

impl ProviderHandle {
    pub fn new(components: ProviderComponents) -> Self {
        Self { components }
    }

    pub fn components(&self) -> &ProviderComponents {
        &self.components
    }

    pub fn components_mut(&mut self) -> &mut ProviderComponents {
        &mut self.components
    }

    pub fn kind(&self) -> &'static str {
        self.components.state.kind()
    }

    pub fn default_model(&self) -> &str {
        self.components.model_policy.default_model()
    }

    pub fn supported_variants(&self, model: &str) -> &'static [&'static str] {
        self.components.model_policy.supported_variants(model)
    }

    pub fn default_model_variant(&self, model: &str) -> Option<&'static str> {
        self.components.model_policy.default_model_variant(model)
    }

    pub fn validate_variant(&self, model: &str, variant: &str) -> Result<(), String> {
        let variants = self.supported_variants(model);
        if variants.is_empty() {
            return Err(format!(
                "Model `{}` on {} does not expose configurable variants.",
                model,
                provider_cli_label(self.kind())
            ));
        }
        if variants.contains(&variant) {
            return Ok(());
        }
        Err(format!(
            "Unsupported variant `{}` for `{}` on {}. Available: {}",
            variant,
            model,
            provider_cli_label(self.kind()),
            variants.join(", ")
        ))
    }

    pub fn request_variant_config(
        &self,
        model: &str,
        variant: &str,
    ) -> Option<VariantRequestConfig> {
        self.components
            .model_policy
            .request_variant_config(model, variant)
    }

    pub fn default_agent_model(&self, tier: &str) -> Option<AgentModelSelection> {
        self.components.model_policy.default_agent_model(tier)
    }

    pub fn resolve_model(&self, model: &str) -> String {
        self.components.model_policy.resolve_model(model)
    }

    pub fn context_lookup_model(&self, model: &str) -> String {
        self.components.model_policy.context_lookup_model(model)
    }

    pub fn input_usage_excludes_cached_tokens(&self) -> bool {
        self.components
            .model_policy
            .input_usage_excludes_cached_tokens()
    }

    pub fn options(&self) -> ProviderOptions {
        self.components.state.options()
    }

    pub fn set_options(&mut self, options: ProviderOptions) {
        self.components.state.set_options(options)
    }

    pub fn requires_streaming(&self) -> bool {
        self.components.readiness.requires_streaming()
    }

    pub async fn ensure_fresh(&mut self) -> Result<bool, OAuthError> {
        self.components.auth.ensure_fresh().await
    }

    pub async fn ensure_ready(&mut self) -> Result<bool, LlmTransportError> {
        self.components.readiness.ensure_ready().await
    }

    pub async fn complete(
        &mut self,
        request: LlmRequest,
    ) -> Result<LlmResponse, LlmTransportError> {
        self.components.transport.complete(request).await
    }

    pub fn to_spec(&self) -> ProviderSpec {
        ProviderSpec {
            kind: self.kind().to_string(),
            config: self.components.state.serialize_config(),
        }
    }

    /// Validate model syntax only.
    pub fn validate_model_name(&self, model: &str) -> Result<(), String> {
        let m = model.trim();
        if m.is_empty() {
            return Err("model cannot be empty".to_string());
        }
        if m.contains(char::is_whitespace) {
            return Err("model cannot contain whitespace".to_string());
        }
        Ok(())
    }

    /// Resolve a model against an explicit catalog supplied by the host.
    pub fn resolve_model_spec(
        &self,
        model: &str,
        catalog: &ModelCatalog,
    ) -> Result<ResolvedModelSpec, String> {
        self.validate_model_name(model)?;
        let configured_model = model.trim();
        let catalog_model_id = self.context_lookup_model(configured_model);
        let Some(info) = catalog.get(&catalog_model_id).cloned() else {
            return Err(format!(
                "model `{}` has no context-window entry in the supplied model catalog for {}. Provide an explicit model spec or choose a cataloged model.",
                configured_model,
                provider_cli_label(self.kind()),
            ));
        };
        Ok(ResolvedModelSpec {
            configured_model: configured_model.to_string(),
            resolved_model: self.resolve_model(configured_model),
            catalog_model_id,
            info,
        })
    }
}

impl std::fmt::Debug for ProviderHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.components.fmt(f)
    }
}

impl Clone for ProviderHandle {
    fn clone(&self) -> Self {
        Self {
            components: self.components.clone(),
        }
    }
}

impl PartialEq for ProviderHandle {
    fn eq(&self, other: &Self) -> bool {
        self.kind() == other.kind() && self.to_spec().config == other.to_spec().config
    }
}

impl Eq for ProviderHandle {}

impl Serialize for ProviderHandle {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.to_spec().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ProviderHandle {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let spec = ProviderSpec::deserialize(deserializer)?;
        build_provider(&spec)
            .map(ProviderHandle::new)
            .map_err(serde::de::Error::custom)
    }
}

impl Default for ProviderHandle {
    fn default() -> Self {
        Self::new(UnconfiguredProvider::default().into_components())
    }
}

/// Placeholder provider used when `SessionPolicy::default()` is
/// constructed without an explicit provider. Every transport-level
/// method errors; calling code MUST replace this before executing a
/// turn. It exists solely so `..Default::default()` shorthand keeps
/// working in host code that always overrides the provider field.
#[derive(Clone, Debug, Default)]
pub struct UnconfiguredProvider {
    options: ProviderOptions,
}

impl UnconfiguredProvider {
    fn into_components(self) -> ProviderComponents {
        ProviderComponents::shared(self, Arc::new(StaticModelPolicy::new("")))
    }
}

impl ProviderState for UnconfiguredProvider {
    fn kind(&self) -> &'static str {
        "unconfigured"
    }

    fn options(&self) -> ProviderOptions {
        self.options.clone()
    }

    fn set_options(&mut self, options: ProviderOptions) {
        self.options = options;
    }

    fn serialize_config(&self) -> serde_json::Value {
        serde_json::Value::Object(Default::default())
    }

    fn clone_boxed(&self) -> Box<dyn ProviderState> {
        Box::new(self.clone())
    }
}

#[async_trait]
impl ProviderAuth for UnconfiguredProvider {
    async fn ensure_fresh(&mut self) -> Result<bool, OAuthError> {
        Ok(false)
    }

    fn clone_boxed(&self) -> Box<dyn ProviderAuth> {
        Box::new(self.clone())
    }
}

#[async_trait]
impl ProviderReadiness for UnconfiguredProvider {
    async fn ensure_ready(&mut self) -> Result<bool, LlmTransportError> {
        Ok(false)
    }

    fn requires_streaming(&self) -> bool {
        false
    }

    fn clone_boxed(&self) -> Box<dyn ProviderReadiness> {
        Box::new(self.clone())
    }
}

#[async_trait]
impl ProviderTransport for UnconfiguredProvider {
    async fn complete(&mut self, _request: LlmRequest) -> Result<LlmResponse, LlmTransportError> {
        Err(LlmTransportError::new(
            "no provider configured: host must install a provider factory and set SessionPolicy.provider before running a turn",
        ))
    }

    fn clone_boxed(&self) -> Box<dyn ProviderTransport> {
        Box::new(self.clone())
    }
}

/// Serializable representation of a provider. JSON shape is the flat
/// form `{"type": kind, …config}` so `~/.lash/config.json` stays
/// backward-compatible with the old `#[serde(tag = "type")]` enum.
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
            let kind = raw
                .as_str()
                .ok_or_else(|| serde::de::Error::custom("provider `type` must be a string"))?
                .to_string();
            // Legacy alias normalization.
            match kind.as_str() {
                "openai-generic" => "openai-compatible".to_string(),
                _ => kind,
            }
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

/// Registers a concrete provider type with lash's global registry. Each
/// provider crate ships one factory; hosts (`lash-cli`, bench runners,
/// custom embedders) call [`register_provider_factory`] for every
/// backend they want to offer.
pub trait ProviderFactory: Send + Sync {
    fn kind(&self) -> &'static str;

    /// Human-readable label shown in `/provider` and setup UI.
    fn cli_label(&self) -> &'static str;

    /// Short name used in the setup menu header.
    fn setup_name(&self) -> &'static str;

    /// One-line description shown next to the setup menu option.
    fn setup_description(&self) -> &'static str;

    /// Suggested default base URL (shown as placeholder in setup).
    fn default_base_url(&self) -> Option<&'static str> {
        None
    }

    /// Instantiate a provider from its [`ProviderSpec::config`] blob.
    fn deserialize(&self, config: serde_json::Value) -> Result<ProviderComponents, String>;
}

#[derive(Clone, Default)]
pub struct ProviderRegistry {
    factories: BTreeMap<&'static str, Arc<dyn ProviderFactory>>,
}

impl ProviderRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, factory: Arc<dyn ProviderFactory>) {
        self.factories.insert(factory.kind(), factory);
    }

    pub fn build_from_spec(&self, spec: &ProviderSpec) -> Result<ProviderComponents, String> {
        let factory = self.factories.get(spec.kind.as_str()).ok_or_else(|| {
            format!(
                "provider `{}` is not registered. Call `lash::register_provider_factory` at startup.",
                spec.kind
            )
        })?;
        factory.deserialize(spec.config.clone())
    }

    pub fn factory(&self, kind: &str) -> Option<&Arc<dyn ProviderFactory>> {
        self.factories.get(kind)
    }
}

static PROVIDER_REGISTRY: LazyLock<RwLock<ProviderRegistry>> =
    LazyLock::new(|| RwLock::new(ProviderRegistry::new()));

/// Register a provider factory in the global registry. Hosts call this
/// once per backend at process startup, before constructing any
/// `LashConfig` or session from disk.
pub fn register_provider_factory(factory: Arc<dyn ProviderFactory>) {
    PROVIDER_REGISTRY.write().unwrap().register(factory);
}

/// Materialize a provider from its serialized form using the global
/// registry. Returns `Err` if no factory is registered for `spec.kind`.
pub fn build_provider(spec: &ProviderSpec) -> Result<ProviderComponents, String> {
    PROVIDER_REGISTRY.read().unwrap().build_from_spec(spec)
}

/// Look up a registered provider factory by kind. Returns `None` if no
/// factory with that kind is registered. Hosts use this to render UI
/// labels / descriptions (`cli_label`, `setup_name`, …) without
/// hard-coding per-kind strings.
pub fn provider_factory(kind: &str) -> Option<Arc<dyn ProviderFactory>> {
    PROVIDER_REGISTRY.read().unwrap().factory(kind).cloned()
}

/// Human-readable label for a provider kind when its factory is registered.
/// Falls back to the stable kind string for unregistered test/internal providers.
pub fn provider_cli_label(kind: &'static str) -> &'static str {
    provider_factory(kind)
        .map(|factory| factory.cli_label())
        .unwrap_or(kind)
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

/// Write back the active provider's refreshed credentials, preserving
/// other fields in the stored config. Called by the runtime after each
/// successful OAuth refresh.
pub fn save_provider(
    path: &std::path::Path,
    provider: &ProviderHandle,
) -> Result<(), std::io::Error> {
    let spec = provider.to_spec();
    let mut config = LashConfig::load(path).unwrap_or_else(|| LashConfig::from_spec(spec.clone()));
    config.upsert_provider_spec(spec.clone());
    config.active_provider = spec.kind;
    config.save(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use crate::llm::types::{LlmToolChoice, LlmUsage};

    #[derive(Clone, Debug, Default)]
    struct MutatingProvider {
        options: ProviderOptions,
        marker: String,
    }

    impl MutatingProvider {
        fn into_components(self, policy: Arc<dyn ProviderModelPolicy>) -> ProviderComponents {
            ProviderComponents::shared(self, policy)
        }
    }

    impl ProviderState for MutatingProvider {
        fn kind(&self) -> &'static str {
            "mutating"
        }

        fn options(&self) -> ProviderOptions {
            self.options.clone()
        }

        fn set_options(&mut self, options: ProviderOptions) {
            self.options = options;
        }

        fn serialize_config(&self) -> serde_json::Value {
            serde_json::json!({ "marker": self.marker })
        }

        fn clone_boxed(&self) -> Box<dyn ProviderState> {
            Box::new(self.clone())
        }
    }

    #[async_trait::async_trait]
    impl ProviderAuth for MutatingProvider {
        async fn ensure_fresh(&mut self) -> Result<bool, OAuthError> {
            self.marker = "fresh".to_string();
            Ok(true)
        }

        fn clone_boxed(&self) -> Box<dyn ProviderAuth> {
            Box::new(self.clone())
        }
    }

    #[async_trait::async_trait]
    impl ProviderReadiness for MutatingProvider {
        async fn ensure_ready(&mut self) -> Result<bool, LlmTransportError> {
            Ok(false)
        }

        fn requires_streaming(&self) -> bool {
            false
        }

        fn clone_boxed(&self) -> Box<dyn ProviderReadiness> {
            Box::new(self.clone())
        }
    }

    #[async_trait::async_trait]
    impl ProviderTransport for MutatingProvider {
        async fn complete(
            &mut self,
            _request: LlmRequest,
        ) -> Result<LlmResponse, LlmTransportError> {
            self.marker = "complete".to_string();
            Ok(LlmResponse {
                deltas: vec!["ok".to_string()],
                full_text: "ok".to_string(),
                parts: Vec::new(),
                usage: LlmUsage::default(),
                provider_usage: None,
                request_body: None,
                http_summary: None,
            })
        }

        fn clone_boxed(&self) -> Box<dyn ProviderTransport> {
            Box::new(self.clone())
        }
    }

    #[derive(Debug)]
    struct MetricsTransport {
        inner: Box<dyn ProviderTransport>,
        hits: Arc<AtomicUsize>,
    }

    impl Clone for MetricsTransport {
        fn clone(&self) -> Self {
            Self {
                inner: self.inner.clone_boxed(),
                hits: Arc::clone(&self.hits),
            }
        }
    }

    #[async_trait::async_trait]
    impl ProviderTransport for MetricsTransport {
        async fn complete(
            &mut self,
            request: LlmRequest,
        ) -> Result<LlmResponse, LlmTransportError> {
            self.hits.fetch_add(1, Ordering::SeqCst);
            self.inner.complete(request).await
        }

        fn clone_boxed(&self) -> Box<dyn ProviderTransport> {
            Box::new(self.clone())
        }
    }

    fn empty_request() -> LlmRequest {
        LlmRequest {
            model: "model".to_string(),
            messages: Vec::new(),
            attachments: Vec::new(),
            tools: Arc::new(Vec::new()),
            tool_choice: LlmToolChoice::None,
            model_variant: None,
            session_id: None,
            output_spec: None,
            stream_events: None,
            provider_trace: None,
        }
    }

    #[test]
    fn provider_spec_roundtrips_as_flat_object() {
        let spec = ProviderSpec {
            kind: "anthropic".to_string(),
            config: serde_json::json!({
                "api_key": "sk-ant-test",
                "base_url": null
            }),
        };
        let serialized = serde_json::to_value(&spec).expect("serialize");
        assert_eq!(serialized["type"], serde_json::json!("anthropic"));
        assert_eq!(serialized["api_key"], serde_json::json!("sk-ant-test"));
        let roundtripped: ProviderSpec = serde_json::from_value(serialized).expect("deserialize");
        assert_eq!(roundtripped.kind, spec.kind);
        assert_eq!(roundtripped.config["api_key"], spec.config["api_key"]);
    }

    #[test]
    fn provider_spec_accepts_legacy_openai_generic_alias() {
        let raw = serde_json::json!({
            "type": "openai-generic",
            "api_key": "k"
        });
        let spec: ProviderSpec = serde_json::from_value(raw).expect("legacy alias");
        assert_eq!(spec.kind, "openai-compatible");
    }

    #[test]
    fn lash_config_roundtrips_existing_shape() {
        let raw = serde_json::json!({
            "active_provider": "openai-compatible",
            "providers": {
                "openai-compatible": {
                    "type": "openai-compatible",
                    "api_key": "k",
                    "base_url": "https://example.com/v1"
                }
            }
        });
        let cfg: LashConfig = serde_json::from_value(raw).expect("valid config");
        assert_eq!(cfg.active_provider, "openai-compatible");
        let spec = cfg.active_provider_spec();
        assert_eq!(spec.kind, "openai-compatible");
        assert_eq!(spec.config["api_key"], serde_json::json!("k"));
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

    #[test]
    fn provider_handle_delegates_model_policy_resolution() {
        static VARIANTS: &[&str] = &["low", "high"];
        let handle = ProviderHandle::new(MutatingProvider::default().into_components(Arc::new(
            StaticModelPolicy::with_variants("model-a", VARIANTS, Some("high")),
        )));

        assert_eq!(handle.default_model(), "model-a");
        assert_eq!(handle.supported_variants("model-a"), VARIANTS);
        assert_eq!(handle.default_model_variant("model-a"), Some("high"));
        assert!(handle.validate_variant("model-a", "low").is_ok());
        assert!(handle.validate_variant("model-a", "medium").is_err());
    }

    #[tokio::test]
    async fn refreshed_provider_state_is_visible_to_serialization() {
        let mut handle = ProviderHandle::new(
            MutatingProvider::default()
                .into_components(Arc::new(StaticModelPolicy::new("model-a"))),
        );

        assert!(handle.ensure_fresh().await.expect("refresh succeeds"));

        assert_eq!(
            handle.to_spec().config["marker"],
            serde_json::json!("fresh")
        );
    }

    #[tokio::test]
    async fn transport_mutations_are_visible_after_completion_returns() {
        let mut handle = ProviderHandle::new(
            MutatingProvider::default()
                .into_components(Arc::new(StaticModelPolicy::new("model-a"))),
        );

        handle.complete(empty_request()).await.expect("complete");

        assert_eq!(
            handle.to_spec().config["marker"],
            serde_json::json!("complete")
        );
    }

    #[tokio::test]
    async fn map_transport_installs_transport_only_decorator() {
        let hits = Arc::new(AtomicUsize::new(0));
        let components = MutatingProvider::default()
            .into_components(Arc::new(StaticModelPolicy::new("model-a")))
            .map_transport({
                let hits = Arc::clone(&hits);
                move |inner| Box::new(MetricsTransport { inner, hits })
            });
        let mut handle = ProviderHandle::new(components);

        handle.complete(empty_request()).await.expect("complete");

        assert_eq!(hits.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn legacy_runtime_context_strategy_is_ignored() {
        let raw = serde_json::json!({
            "active_provider": "openai-compatible",
            "providers": {
                "openai-compatible": {
                    "type": "openai-compatible",
                    "api_key": "k",
                    "base_url": "https://example.com/v1"
                }
            },
            "runtime": {
                "context_strategy": {
                    "type": "rolling_context"
                }
            }
        });
        let _cfg: LashConfig =
            serde_json::from_value(raw).expect("legacy config json still deserializes");
    }
}
