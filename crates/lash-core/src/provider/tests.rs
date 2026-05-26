use super::support::*;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::GenerationOptions;
use crate::llm::types::{LlmToolChoice, LlmUsage};

#[derive(Clone, Debug, Default)]
struct MutatingProvider {
    options: ProviderOptions,
    marker: String,
}

#[derive(Clone, Debug)]
struct FailingProvider {
    options: ProviderOptions,
    attempts: Arc<AtomicUsize>,
    fail_until: usize,
    retryable: bool,
}

impl FailingProvider {
    fn into_components(self) -> ProviderComponents {
        ProviderComponents::shared(self, Arc::new(StaticModelPolicy::new()))
    }
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

impl ProviderState for FailingProvider {
    fn kind(&self) -> &'static str {
        "failing"
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

#[async_trait::async_trait]
impl ProviderTransport for MutatingProvider {
    async fn complete(&mut self, _request: LlmRequest) -> Result<LlmResponse, LlmTransportError> {
        self.marker = "complete".to_string();
        Ok(LlmResponse {
            full_text: "ok".to_string(),
            parts: Vec::new(),
            usage: LlmUsage::default(),
            terminal_reason: crate::LlmTerminalReason::Stop,
            terminal_diagnostic: None,
            provider_usage: None,
            request_body: None,
            http_summary: None,
        })
    }

    fn clone_boxed(&self) -> Box<dyn ProviderTransport> {
        Box::new(self.clone())
    }
}

#[async_trait::async_trait]
impl ProviderTransport for FailingProvider {
    async fn complete(&mut self, _request: LlmRequest) -> Result<LlmResponse, LlmTransportError> {
        let attempt = self.attempts.fetch_add(1, Ordering::SeqCst) + 1;
        if attempt <= self.fail_until {
            let kind = if self.retryable {
                ProviderFailureKind::Transport
            } else {
                ProviderFailureKind::Validation
            };
            return Err(LlmTransportError::new("temporary failure")
                .with_kind(kind)
                .retryable(self.retryable));
        }
        Ok(LlmResponse {
            full_text: "ok".to_string(),
            parts: Vec::new(),
            usage: LlmUsage::default(),
            terminal_reason: crate::LlmTerminalReason::Stop,
            terminal_diagnostic: None,
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
    async fn complete(&mut self, request: LlmRequest) -> Result<LlmResponse, LlmTransportError> {
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
        generation: GenerationOptions::default(),
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
fn provider_options_serialize_only_reliability_shape() {
    let options = ProviderOptions {
        reliability: ProviderReliability::builder()
            .request_timeout(Some(RequestTimeout::Millis(1_234)))
            .stream_chunk_timeout_ms(Some(567))
            .max_attempts(2)
            .build(),
        ..ProviderOptions::default()
    };

    let value = serde_json::to_value(options).expect("serialize");
    assert!(value.get("timeout").is_none());
    assert!(value.get("chunk_timeout").is_none());
    assert_eq!(
        value["reliability"]["timeouts"]["request_timeout"],
        serde_json::json!(1234)
    );
    assert_eq!(
        value["reliability"]["timeouts"]["chunk_timeout"],
        serde_json::json!(567)
    );
}

#[test]
fn provider_options_roundtrip_output_limit_and_cache_retention() {
    let options = ProviderOptions {
        max_output_tokens: Some(16_384),
        cache_retention: CacheRetention::Long,
        ..ProviderOptions::default()
    };

    let value = serde_json::to_value(&options).expect("serialize");
    assert_eq!(value["max_output_tokens"], serde_json::json!(16_384));
    assert_eq!(value["cache_retention"], serde_json::json!("long"));

    let roundtripped: ProviderOptions = serde_json::from_value(value).expect("deserialize");
    assert_eq!(roundtripped, options);
}

#[test]
fn provider_options_default_omits_and_restores_shared_output_fields() {
    let value = serde_json::to_value(ProviderOptions::default()).expect("serialize");
    assert!(value.get("max_output_tokens").is_none());
    assert!(value.get("cache_retention").is_none());

    let restored: ProviderOptions = serde_json::from_value(serde_json::json!({})).expect("default");
    assert_eq!(restored.max_output_tokens, None);
    assert_eq!(restored.cache_retention, CacheRetention::Short);
    assert!(restored.is_default());
}

#[test]
fn generation_policy_prefers_request_then_provider_then_default() {
    let provider_options = ProviderOptions {
        max_output_tokens: Some(8_192),
        cache_retention: CacheRetention::Long,
        thinking: ProviderThinkingPolicy { expose: true },
        ..ProviderOptions::default()
    };
    let defaulted = resolve_generation_policy(
        &GenerationOptions::default(),
        &ProviderOptions::default(),
        32_768,
        "thinking",
    );
    assert_eq!(defaulted.max_output_tokens, 32_768);
    assert_eq!(defaulted.cache_retention, CacheRetention::Short);
    assert!(!defaulted.expose_thinking);
    assert_eq!(defaulted.thinking, "thinking");

    let provider_limited =
        resolve_generation_policy(&GenerationOptions::default(), &provider_options, 32_768, ());
    assert_eq!(provider_limited.max_output_tokens, 8_192);
    assert_eq!(provider_limited.cache_retention, CacheRetention::Long);
    assert!(provider_limited.expose_thinking);

    let request_generation = GenerationOptions {
        output_token_cap: NonZeroUsize::new(2_048),
    };
    let request_limited =
        resolve_generation_policy(&request_generation, &provider_options, 32_768, ());
    assert_eq!(request_limited.max_output_tokens, 2_048);
}

#[test]
fn provider_handle_delegates_variant_policy() {
    static VARIANTS: &[&str] = &["low", "high"];
    let handle = ProviderHandle::new(
        MutatingProvider::default()
            .into_components(Arc::new(StaticModelPolicy::with_variants(VARIANTS))),
    );

    assert_eq!(handle.supported_variants("model-a"), VARIANTS);
    assert!(handle.validate_variant("model-a", "low").is_ok());
    assert!(handle.validate_variant("model-a", "medium").is_err());
}

#[tokio::test]
async fn transport_mutations_are_visible_after_completion_returns() {
    let mut handle = ProviderHandle::new(
        MutatingProvider::default().into_components(Arc::new(StaticModelPolicy::new())),
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
        .into_components(Arc::new(StaticModelPolicy::new()))
        .map_transport({
            let hits = Arc::clone(&hits);
            move |inner| Box::new(MetricsTransport { inner, hits })
        });
    let mut handle = ProviderHandle::new(components);

    handle.complete(empty_request()).await.expect("complete");

    assert_eq!(hits.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn provider_handle_retries_retryable_failures_in_shared_executor() {
    let attempts = Arc::new(AtomicUsize::new(0));
    let provider = FailingProvider {
        options: ProviderOptions {
            reliability: ProviderReliability::builder()
                .max_attempts(3)
                .base_delay_ms(0)
                .max_delay_ms(0)
                .build(),
            ..ProviderOptions::default()
        },
        attempts: Arc::clone(&attempts),
        fail_until: 2,
        retryable: true,
    };
    let mut handle = ProviderHandle::new(provider.into_components());

    handle
        .complete(empty_request())
        .await
        .expect("eventual success");

    assert_eq!(attempts.load(Ordering::SeqCst), 3);
}

#[tokio::test]
async fn provider_handle_stops_on_non_retryable_failure() {
    let attempts = Arc::new(AtomicUsize::new(0));
    let provider = FailingProvider {
        options: ProviderOptions {
            reliability: ProviderReliability::builder()
                .max_attempts(3)
                .base_delay_ms(0)
                .max_delay_ms(0)
                .build(),
            ..ProviderOptions::default()
        },
        attempts: Arc::clone(&attempts),
        fail_until: 3,
        retryable: false,
    };
    let mut handle = ProviderHandle::new(provider.into_components());

    let err = handle
        .complete(empty_request())
        .await
        .expect_err("non retryable");

    assert!(!err.retryable);
    assert_eq!(attempts.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn provider_handle_set_options_affects_retry_behavior() {
    let attempts = Arc::new(AtomicUsize::new(0));
    let provider = FailingProvider {
        options: ProviderOptions {
            reliability: ProviderReliability::disabled(),
            ..ProviderOptions::default()
        },
        attempts: Arc::clone(&attempts),
        fail_until: 1,
        retryable: true,
    };
    let mut handle = ProviderHandle::new(provider.into_components());
    handle.set_options(ProviderOptions {
        reliability: ProviderReliability::builder()
            .max_attempts(2)
            .base_delay_ms(0)
            .max_delay_ms(0)
            .build(),
        ..ProviderOptions::default()
    });

    handle
        .complete(empty_request())
        .await
        .expect("retry after set_options");

    assert_eq!(attempts.load(Ordering::SeqCst), 2);
}

#[test]
fn default_failure_classifier_marks_context_overflow() {
    let classifier = DefaultProviderFailureClassifier;
    for message in [
        "Anthropic request failed: prompt is too long",
        "context_length_exceeded",
        "This model's maximum context length is 128000 tokens",
        "Google says input is too long",
        "OpenRouter error: too many tokens",
        "Together: request too large",
        "Copilot: exceeds the maximum number of tokens",
        "local model: context window exceeded",
        "Anthropic: request_too_large",
        "OpenAI: Your input exceeds the context window of this model",
        "Google: The input token count (1196265) exceeds the maximum number of tokens allowed",
        "xAI: This model's maximum prompt length is 131072 but the request contains more",
        "Groq: Please reduce the length of the messages",
        "OpenRouter: This endpoint's maximum context length is 128000 tokens",
        "Together: The input (150000 tokens) is longer than the model's context length (128000 tokens).",
        "llama.cpp: the request exceeds the available context size",
        "LM Studio: tokens to keep from the initial prompt is greater than the context length",
        "MiniMax: invalid params, context window exceeds limit",
        "Kimi: Your request exceeded model token limit: 131072 (requested: 150000)",
        "Mistral: Prompt contains too many tokens; too large for model with 128000 maximum context length",
        "z.ai: model_context_window_exceeded",
    ] {
        let failure = classifier.classify(ProviderFailure::new(message).with_status(400));
        assert_eq!(
            failure.terminal_reason,
            crate::LlmTerminalReason::ContextOverflow
        );
        assert!(!failure.retryable);
    }
}

#[test]
fn default_failure_classifier_marks_content_filter() {
    let classifier = DefaultProviderFailureClassifier;
    for message in [
        "Provider finish_reason: content_filter",
        "blocked by prohibited_content",
        "safety filter tripped",
        "sensitive content refused",
    ] {
        let failure = classifier.classify(ProviderFailure::new(message).with_status(400));
        assert_eq!(
            failure.terminal_reason,
            crate::LlmTerminalReason::ContentFilter
        );
    }
}

#[test]
fn default_failure_classifier_does_not_treat_rate_limits_as_context_overflow() {
    let classifier = DefaultProviderFailureClassifier;
    for message in [
        "rate limit: too many tokens per minute",
        "Too many requests",
        "throttling because token rate exceeded",
        "insufficient_quota",
    ] {
        let failure = classifier.classify(ProviderFailure::new(message).with_status(429));
        assert_ne!(
            failure.terminal_reason,
            crate::LlmTerminalReason::ContextOverflow
        );
    }
}
