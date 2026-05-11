use super::support::*;
use std::sync::atomic::{AtomicUsize, Ordering};

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
        ProviderComponents::shared(self, Arc::new(StaticModelPolicy::new("model-a")))
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
async fn transport_mutations_are_visible_after_completion_returns() {
    let mut handle = ProviderHandle::new(
        MutatingProvider::default().into_components(Arc::new(StaticModelPolicy::new("model-a"))),
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
