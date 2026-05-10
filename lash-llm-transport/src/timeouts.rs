use std::future::Future;
use std::time::Duration;

use lash::{LlmTransportError, ProviderFailureKind};

pub const DEFAULT_REQUEST_TIMEOUT_MS: u64 = 300_000;
pub const DEFAULT_CHUNK_TIMEOUT_MS: u64 = 120_000;

pub type RequestBodySnapshot = bytes::Bytes;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LlmTimeouts {
    pub request_timeout: Option<Duration>,
    pub chunk_timeout: Duration,
}

impl Default for LlmTimeouts {
    fn default() -> Self {
        Self {
            request_timeout: Some(Duration::from_millis(DEFAULT_REQUEST_TIMEOUT_MS)),
            chunk_timeout: Duration::from_millis(DEFAULT_CHUNK_TIMEOUT_MS),
        }
    }
}

pub fn build_http_client() -> reqwest::Client {
    reqwest::Client::builder()
        .build()
        .expect("failed to build reqwest client for llm transport")
}

pub fn request_body_snapshot(body: String) -> RequestBodySnapshot {
    bytes::Bytes::from(body)
}

pub fn request_body_snapshot_bytes(body: Vec<u8>) -> RequestBodySnapshot {
    bytes::Bytes::from(body)
}

fn is_retryable_http_error(error: &reqwest::Error) -> bool {
    error.is_timeout() || error.is_connect() || error.is_body() || error.is_decode()
}

pub fn response_start_timeout(
    request_timeout: Option<Duration>,
    chunk_timeout: Duration,
    streaming: bool,
) -> Option<Duration> {
    if !streaming {
        return request_timeout;
    }
    Some(match request_timeout {
        Some(timeout) => timeout.min(chunk_timeout),
        None => chunk_timeout,
    })
}

pub async fn run_with_timeout<T, F>(
    future: F,
    timeout: Option<Duration>,
    timeout_message: &str,
) -> Result<T, LlmTransportError>
where
    F: Future<Output = Result<T, LlmTransportError>>,
{
    match timeout {
        Some(duration) => tokio::time::timeout(duration, future).await.map_err(|_| {
            LlmTransportError::new(timeout_message)
                .with_kind(ProviderFailureKind::Timeout)
                .retryable(true)
                .with_code("timeout")
        })?,
        None => future.await,
    }
}

pub async fn send_request(
    request: reqwest::RequestBuilder,
    request_body: Option<RequestBodySnapshot>,
    timeout: Option<Duration>,
    timeout_message: &str,
) -> Result<reqwest::Response, LlmTransportError> {
    run_with_timeout(
        async move {
            request.send().await.map_err(|e| {
                let error = LlmTransportError::new(format!("HTTP request failed: {e}"))
                    .with_kind(ProviderFailureKind::Transport)
                    .retryable(is_retryable_http_error(&e));
                if let Some(request_body) = request_body {
                    error.with_request_body(String::from_utf8_lossy(&request_body).into_owned())
                } else {
                    error
                }
            })
        },
        timeout,
        timeout_message,
    )
    .await
}

pub async fn read_response_text(
    response: reqwest::Response,
    timeout: Option<Duration>,
    timeout_message: &str,
) -> Result<String, LlmTransportError> {
    run_with_timeout(
        async move {
            response.text().await.map_err(|e| {
                LlmTransportError::new(format!("HTTP response read failed: {e}"))
                    .with_kind(ProviderFailureKind::Transport)
                    .retryable(is_retryable_http_error(&e))
            })
        },
        timeout,
        timeout_message,
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streaming_response_start_timeout_prefers_chunk_deadline() {
        let timeout = response_start_timeout(
            Some(Duration::from_secs(300)),
            Duration::from_secs(120),
            true,
        );
        assert_eq!(timeout, Some(Duration::from_secs(120)));
    }

    #[test]
    fn non_stream_response_start_timeout_uses_request_deadline() {
        let timeout = response_start_timeout(
            Some(Duration::from_secs(300)),
            Duration::from_secs(120),
            false,
        );
        assert_eq!(timeout, Some(Duration::from_secs(300)));
    }

    #[tokio::test]
    async fn run_with_timeout_returns_timeout_error() {
        let result = run_with_timeout(
            async {
                tokio::time::sleep(Duration::from_millis(25)).await;
                Ok::<_, LlmTransportError>(())
            },
            Some(Duration::from_millis(5)),
            "request timed out",
        )
        .await;

        let err = result.expect_err("expected timeout");
        assert_eq!(err.message, "request timed out");
        assert_eq!(err.code.as_deref(), Some("timeout"));
        assert!(err.retryable);
    }

    #[tokio::test]
    async fn run_with_timeout_allows_successful_completion() {
        let result = run_with_timeout(
            async { Ok::<_, LlmTransportError>(42) },
            Some(Duration::from_secs(1)),
            "request timed out",
        )
        .await;

        assert_eq!(result.expect("success"), 42);
    }
}
