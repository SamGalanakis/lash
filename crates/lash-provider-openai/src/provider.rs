use crate::support::*;

impl OpenAiCompatibleProvider {
    pub fn new(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: base_url.into(),
            options: ProviderOptions::default(),
            compat: OpenAiCompat::default(),
            transport: DEFAULT_HTTP_TRANSPORT.clone(),
        }
    }

    pub fn with_options(mut self, options: ProviderOptions) -> Self {
        self.options = options;
        self
    }

    pub fn with_compat(mut self, compat: OpenAiCompat) -> Self {
        self.compat = compat;
        self
    }

    pub fn with_schema_capabilities(mut self, capabilities: ProviderSchemaCapabilities) -> Self {
        self.compat.schema_capabilities = Some(capabilities);
        self
    }

    pub fn with_transport(mut self, transport: std::sync::Arc<dyn LlmHttpTransport>) -> Self {
        self.transport = transport;
        self
    }

    pub fn into_components(self) -> ProviderComponents {
        ProviderComponents::new(Box::new(self))
    }
}

impl OpenAiProvider {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            inner: OpenAiCompatibleProvider::new(api_key, OPENAI_BASE_URL),
        }
    }

    pub fn with_options(mut self, options: ProviderOptions) -> Self {
        self.inner.options = options;
        self
    }

    pub fn with_transport(mut self, transport: std::sync::Arc<dyn LlmHttpTransport>) -> Self {
        self.inner.transport = transport;
        self
    }

    pub fn into_components(self) -> ProviderComponents {
        ProviderComponents::new(Box::new(self))
    }

    #[cfg(test)]
    pub(crate) fn build_responses_request_body(
        &self,
        req: &LlmRequest,
        stream: bool,
    ) -> Result<Value, LlmTransportError> {
        self.inner.build_responses_request_body(req, stream)
    }
}

#[async_trait]
impl Provider for OpenAiCompatibleProvider {
    fn kind(&self) -> &'static str {
        "openai-compatible"
    }

    fn options(&self) -> ProviderOptions {
        self.options.clone()
    }

    fn set_options(&mut self, options: ProviderOptions) {
        self.options = options;
    }

    fn serialize_config(&self) -> serde_json::Value {
        let mut map = serde_json::Map::new();
        map.insert(
            "api_key".to_string(),
            serde_json::Value::String(self.api_key.clone()),
        );
        map.insert(
            "base_url".to_string(),
            serde_json::Value::String(self.base_url.clone()),
        );
        if !self.options.is_default() {
            map.insert(
                "options".to_string(),
                serde_json::to_value(&self.options).unwrap_or(serde_json::Value::Null),
            );
        }
        if self.compat != OpenAiCompat::default() {
            map.insert(
                "compat".to_string(),
                serde_json::to_value(&self.compat).unwrap_or(serde_json::Value::Null),
            );
        }
        serde_json::Value::Object(map)
    }

    async fn complete(&mut self, req: LlmRequest) -> Result<LlmResponse, LlmTransportError> {
        complete(self, req, CompletionEndpoint::ChatCompletions).await
    }

    fn clone_boxed(&self) -> Box<dyn Provider> {
        Box::new(self.clone())
    }
}

#[async_trait]
impl Provider for OpenAiProvider {
    fn kind(&self) -> &'static str {
        "openai"
    }

    fn options(&self) -> ProviderOptions {
        self.inner.options.clone()
    }

    fn set_options(&mut self, options: ProviderOptions) {
        self.inner.options = options;
    }

    fn serialize_config(&self) -> serde_json::Value {
        let mut map = serde_json::Map::new();
        map.insert(
            "api_key".to_string(),
            serde_json::Value::String(self.inner.api_key.clone()),
        );
        if !self.inner.options.is_default() {
            map.insert(
                "options".to_string(),
                serde_json::to_value(&self.inner.options).unwrap_or(serde_json::Value::Null),
            );
        }
        serde_json::Value::Object(map)
    }

    async fn complete(&mut self, req: LlmRequest) -> Result<LlmResponse, LlmTransportError> {
        complete(&mut self.inner, req, CompletionEndpoint::Responses).await
    }

    fn clone_boxed(&self) -> Box<dyn Provider> {
        Box::new(self.clone())
    }
}
