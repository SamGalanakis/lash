use super::support::*;

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
pub trait ProviderTransport: Send + Sync + std::fmt::Debug {
    async fn complete(&mut self, request: LlmRequest) -> Result<LlmResponse, LlmTransportError>;

    fn requires_streaming(&self) -> bool {
        false
    }

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

pub trait ProviderFailureClassifier: Send + Sync + std::fmt::Debug {
    fn classify(&self, failure: ProviderFailure) -> ProviderFailure;
}

#[derive(Clone, Debug, Default)]
pub struct DefaultProviderFailureClassifier;

impl ProviderFailureClassifier for DefaultProviderFailureClassifier {
    fn classify(&self, mut failure: ProviderFailure) -> ProviderFailure {
        if let Some(status) = failure.status.or_else(|| {
            failure
                .code
                .as_deref()
                .and_then(|code| code.parse::<u16>().ok())
        }) {
            failure.status = Some(status);
            if failure.kind == ProviderFailureKind::Unknown {
                failure.kind = ProviderFailureKind::Http;
            }
            failure.retryable = matches!(status, 408 | 409 | 425 | 429 | 500 | 502 | 503 | 504);
            if matches!(status, 401 | 403) {
                failure.kind = ProviderFailureKind::Auth;
            } else if matches!(status, 400 | 413 | 422) {
                failure.kind = ProviderFailureKind::Validation;
            }
        } else if matches!(
            failure.kind,
            ProviderFailureKind::Transport | ProviderFailureKind::Timeout
        ) {
            failure.retryable = true;
        }

        let haystack = format!(
            "{}\n{}\n{}",
            failure.code.as_deref().unwrap_or_default(),
            failure.message,
            failure.raw.as_deref().unwrap_or_default()
        )
        .to_ascii_lowercase();
        if haystack.contains("context_length")
            || haystack.contains("context length")
            || haystack.contains("maximum context")
            || haystack.contains("invalid_request_error")
        {
            failure.kind = ProviderFailureKind::Validation;
            failure.retryable = false;
        }
        if haystack.contains("insufficient_quota")
            || haystack.contains("usage_limit_reached")
            || haystack.contains("usage_not_included")
            || haystack.contains("quota")
        {
            failure.kind = ProviderFailureKind::Quota;
            failure.retryable = false;
        }
        if haystack.contains("model_not_found")
            || haystack.contains("unsupported model")
            || haystack.contains("does not exist")
        {
            failure.kind = ProviderFailureKind::Unsupported;
            failure.retryable = false;
        }
        failure
    }
}
