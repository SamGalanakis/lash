//! The [`Provider`] trait implementation: config serialization plus the
//! `complete` request/stream driver.

use lash_llm_transport::util::extract_error_detail;

use crate::config::DEFAULT_BASE_URL;
use crate::policy::{
    ANTHROPIC_VERSION, FINE_GRAINED_BETA, INTERLEAVED_THINKING_BETA,
    anthropic_supports_adaptive_thinking,
};
use crate::stream::StreamState;
use crate::support::*;

#[async_trait]
impl Provider for AnthropicProvider {
    fn kind(&self) -> &'static str {
        "anthropic"
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
        if let Some(base_url) = &self.base_url {
            map.insert(
                "base_url".to_string(),
                serde_json::Value::String(base_url.clone()),
            );
        }
        serialize_options_tail(&mut map, &self.options);
        serde_json::Value::Object(map)
    }

    async fn complete(&mut self, req: LlmRequest) -> Result<LlmResponse, LlmTransportError> {
        let stream_events = req.stream_events.clone();
        let provider_trace = req.provider_trace.clone();
        let timeouts = self.options.llm_timeouts();
        let base_url = self
            .base_url
            .clone()
            .unwrap_or_else(|| DEFAULT_BASE_URL.to_string());

        let body = self.build_request_body(&req)?;
        let request_body = serde_json::to_string(&body).ok();

        // `fine-grained-tool-streaming-2025-05-14` streams partial JSON so we
        // can surface tool arguments incrementally. Interleaved thinking is
        // built-in on adaptive models; for older models we opt into the beta.
        let mut betas = vec![FINE_GRAINED_BETA.to_string()];
        if !anthropic_supports_adaptive_thinking(&req.model) {
            betas.push(INTERLEAVED_THINKING_BETA.to_string());
        }

        let url = format!("{}/v1/messages", base_url.trim_end_matches('/'));
        let request = self
            .client
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("anthropic-beta", betas.join(","))
            .header("Content-Type", "application/json")
            .header("Accept", "text/event-stream")
            .json(&body);

        let resp = send_request(
            request,
            request_body.clone().map(request_body_snapshot),
            response_start_timeout(timeouts.request_timeout, timeouts.chunk_timeout, true),
            "Anthropic response start timed out",
        )
        .await?;

        let status = resp.status();
        if !status.is_success() {
            let headers = resp.headers().clone();
            let text = read_response_text(
                resp,
                timeouts.request_timeout,
                "Anthropic response body timed out",
            )
            .await
            .unwrap_or_default();
            let detail = extract_error_detail(&text);
            let message = if let Some(detail) = detail {
                format!(
                    "Anthropic request failed with {}: {}",
                    status.as_u16(),
                    detail,
                )
            } else {
                format!("Anthropic request failed with {}", status.as_u16())
            };
            return Err(http_error_envelope(
                message,
                status.as_u16(),
                &headers,
                text,
                request_body,
            ));
        }

        let mut state = StreamState::default();
        let expose_thinking = self.options.thinking.expose;
        drive_sse_response(
            resp,
            timeouts.chunk_timeout,
            "Anthropic stream chunk timed out",
            |raw| {
                emit_provider_trace(provider_trace.as_ref(), "anthropic", raw);
                Self::process_sse_event(raw, &mut state, stream_events.as_ref(), expose_thinking)
            },
        )
        .await?;

        let (parts, full_text, usage, terminal_reason) = Self::finalize(state);
        Ok(LlmResponse {
            full_text,
            parts,
            usage,
            terminal_reason,
            terminal_diagnostic: None,
            provider_usage: None,
            request_body,
            http_summary: Some(format!("HTTP POST {} (stream)", url)),
        })
    }

    fn clone_boxed(&self) -> Box<dyn Provider> {
        Box::new(self.clone())
    }
}
