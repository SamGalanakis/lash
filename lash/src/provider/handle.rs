use super::support::*;

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

    fn requires_streaming(&self) -> bool {
        self.inner
            .lock()
            .expect("provider transport lock")
            .requires_streaming()
    }

    fn clone_boxed(&self) -> Box<dyn ProviderTransport> {
        Box::new(self.clone())
    }
}

/// Component bundle returned by provider factories.
#[derive(Debug)]
pub struct ProviderComponents {
    pub state: Box<dyn ProviderState>,
    pub transport: Box<dyn ProviderTransport>,
    pub model_policy: Arc<dyn ProviderModelPolicy>,
    pub failure_classifier: Arc<dyn ProviderFailureClassifier>,
    pub rate_limiter: Arc<ProviderRateLimiter>,
}

impl ProviderComponents {
    pub fn new(
        state: Box<dyn ProviderState>,
        transport: Box<dyn ProviderTransport>,
        model_policy: Arc<dyn ProviderModelPolicy>,
    ) -> Self {
        let options = state.options();
        Self {
            state,
            transport,
            model_policy,
            failure_classifier: Arc::new(DefaultProviderFailureClassifier),
            rate_limiter: Arc::new(ProviderRateLimiter::new(options.reliability.rate_limits)),
        }
    }

    pub fn shared<T>(provider: T, model_policy: Arc<dyn ProviderModelPolicy>) -> Self
    where
        T: ProviderState + ProviderTransport + Clone + Send + Sync + std::fmt::Debug + 'static,
    {
        let inner = Arc::new(Mutex::new(provider));
        let options = inner.lock().expect("provider state lock").options();
        Self {
            state: Box::new(SharedProviderComponent::new(Arc::clone(&inner))),
            transport: Box::new(SharedProviderComponent::new(inner)),
            model_policy,
            failure_classifier: Arc::new(DefaultProviderFailureClassifier),
            rate_limiter: Arc::new(ProviderRateLimiter::new(options.reliability.rate_limits)),
        }
    }

    pub fn map_transport(
        mut self,
        map: impl FnOnce(Box<dyn ProviderTransport>) -> Box<dyn ProviderTransport>,
    ) -> Self {
        self.transport = map(self.transport);
        self
    }

    pub fn with_failure_classifier(
        mut self,
        classifier: Arc<dyn ProviderFailureClassifier>,
    ) -> Self {
        self.failure_classifier = classifier;
        self
    }
}

impl Clone for ProviderComponents {
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone_boxed(),
            transport: self.transport.clone_boxed(),
            model_policy: Arc::clone(&self.model_policy),
            failure_classifier: Arc::clone(&self.failure_classifier),
            rate_limiter: Arc::clone(&self.rate_limiter),
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
                self.kind()
            ));
        }
        if variants.contains(&variant) {
            return Ok(());
        }
        Err(format!(
            "Unsupported variant `{}` for `{}` on {}. Available: {}",
            variant,
            model,
            self.kind(),
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
        self.components
            .rate_limiter
            .configure(options.reliability.rate_limits.clone());
        self.components.state.set_options(options)
    }

    pub fn requires_streaming(&self) -> bool {
        self.components.transport.requires_streaming()
    }

    pub async fn complete(
        &mut self,
        request: LlmRequest,
    ) -> Result<LlmResponse, LlmTransportError> {
        let reliability = self.options().reliability;
        let attempts = reliability.retry.attempts();
        let mut attempt = 0;
        loop {
            let _permit = self.components.rate_limiter.admit(&request).await;
            let result = self.components.transport.complete(request.clone()).await;
            match result {
                Ok(response) => return Ok(response),
                Err(failure) => {
                    let failure = self.components.failure_classifier.classify(failure);
                    if attempt + 1 >= attempts || !failure.retryable {
                        return Err(failure);
                    }
                    let delay = reliability
                        .retry
                        .delay_for_attempt(attempt, failure.retry_after);
                    tracing::debug!(
                        target: "lash::provider::reliability",
                        provider = self.kind(),
                        attempt = attempt + 1,
                        max_attempts = attempts,
                        delay_ms = delay.as_millis() as u64,
                        err = %failure.message,
                        "provider call failed with retryable failure; sleeping before retry"
                    );
                    if let Some(events) = request.stream_events.as_ref() {
                        events.send(crate::llm::types::LlmStreamEvent::RetryStatus {
                            wait_seconds: delay.as_secs(),
                            attempt: (attempt + 1) as usize,
                            max_attempts: attempts as usize,
                            reason: failure.message.clone(),
                        });
                    }
                    tokio::time::sleep(delay).await;
                    attempt += 1;
                }
            }
        }
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
                self.kind(),
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
