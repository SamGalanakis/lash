use std::sync::{Arc, Mutex};

use crate::OpenAiCompatibleProvider;
use async_trait::async_trait;
use lash_core::LlmTransportError;
use lash_core::llm::types::{
    LlmMessage, LlmProviderTraceEvent, LlmProviderTraceSender, LlmRequest, LlmRequestScope,
    LlmRole, LlmToolChoice,
};
use lash_core::provider::{ModelCapability, Provider};
use lash_llm_transport::{LlmHttpBody, LlmHttpRequest, LlmHttpResponse, LlmHttpTransport};

#[derive(Debug, Default)]
struct RecordingTransport {
    requests: Mutex<Vec<LlmHttpRequest>>,
}

#[async_trait]
impl LlmHttpTransport for RecordingTransport {
    async fn send(
        &self,
        request: LlmHttpRequest,
        _timeout: Option<std::time::Duration>,
    ) -> Result<LlmHttpResponse, LlmTransportError> {
        self.requests.lock().expect("request lock").push(request);
        Ok(LlmHttpResponse {
            status: 200,
            headers: Vec::new(),
            body: LlmHttpBody::buffered(
                r#"{"choices":[{"message":{"role":"assistant","content":"done"},"finish_reason":"stop"}]}"#,
            ),
        })
    }
}

fn request() -> LlmRequest {
    LlmRequest {
        model: "test-model".to_string(),
        messages: vec![LlmMessage::text(
            LlmRole::User,
            format!("large prompt: {}", "x".repeat(3_000)),
        )],
        attachments: Vec::new(),
        tools: Arc::new(Vec::new()),
        tool_choice: LlmToolChoice::Auto,
        model_variant: Default::default(),
        model_capability: ModelCapability::default(),
        generation: Default::default(),
        scope: LlmRequestScope::new("session", "frame", "request"),
        output_spec: None,
        stream_events: None,
        provider_trace: None,
    }
}

#[tokio::test]
async fn extended_provider_trace_captures_exact_serialized_chat_body() {
    let transport = Arc::new(RecordingTransport::default());
    let mut provider = OpenAiCompatibleProvider::new("key", "https://example.test/v1")
        .with_transport(transport.clone());
    let mut req = request();
    let expected_value = provider
        .build_chat_request_body(&req, false)
        .expect("build request body");
    let expected_body = serde_json::to_string(&expected_value).expect("serialize request body");
    assert!(expected_body.len() > 2_048);

    let events = Arc::new(Mutex::new(Vec::<LlmProviderTraceEvent>::new()));
    let event_sink = Arc::clone(&events);
    req.provider_trace = Some(LlmProviderTraceSender::new(move |event| {
        event_sink.lock().expect("event lock").push(event);
    }));

    let response = provider.complete(req).await.expect("completion succeeds");

    let request_event = events
        .lock()
        .expect("event lock")
        .iter()
        .find(|event| event.request_endpoint().is_some())
        .cloned()
        .expect("provider request trace");
    assert_eq!(request_event.provider, "openai_compatible");
    assert_eq!(request_event.request_endpoint(), Some("chat/completions"));
    assert_eq!(request_event.raw, expected_body);

    let traced_body = {
        let requests = transport.requests.lock().expect("request lock");
        assert_eq!(requests.len(), 1);
        requests[0].body.clone()
    };
    assert_eq!(traced_body.as_ref(), request_event.raw.as_bytes());
    assert_eq!(
        response.request_body.as_deref(),
        Some(request_event.raw.as_str())
    );

    let untraced_transport = Arc::new(RecordingTransport::default());
    let mut untraced_provider = OpenAiCompatibleProvider::new("key", "https://example.test/v1")
        .with_transport(untraced_transport.clone());
    untraced_provider
        .complete(request())
        .await
        .expect("untraced completion succeeds");
    let untraced_requests = untraced_transport.requests.lock().expect("request lock");
    assert_eq!(untraced_requests[0].body, traced_body);
}
