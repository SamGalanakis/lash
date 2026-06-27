#![allow(clippy::result_large_err)]

use std::fmt;
use std::time::Duration;

use async_trait::async_trait;
use bytes::{Bytes, BytesMut};
use lash_core::{LlmTransportError, ProviderFailureKind};

use crate::timeouts::{header_pairs, reqwest_error_is_retryable, run_with_timeout};

#[async_trait]
pub trait LlmHttpTransport: Send + Sync + fmt::Debug {
    async fn send(
        &self,
        request: LlmHttpRequest,
        timeout: Option<Duration>,
    ) -> Result<LlmHttpResponse, LlmTransportError>;
}

#[async_trait]
pub trait LlmByteStream: Send + fmt::Debug {
    async fn next_chunk(&mut self) -> Result<Option<Bytes>, LlmTransportError>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LlmHttpMethod {
    Get,
    Post,
    Put,
    Patch,
    Delete,
}

impl LlmHttpMethod {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Get => "GET",
            Self::Post => "POST",
            Self::Put => "PUT",
            Self::Patch => "PATCH",
            Self::Delete => "DELETE",
        }
    }

    fn as_reqwest(self) -> reqwest::Method {
        match self {
            Self::Get => reqwest::Method::GET,
            Self::Post => reqwest::Method::POST,
            Self::Put => reqwest::Method::PUT,
            Self::Patch => reqwest::Method::PATCH,
            Self::Delete => reqwest::Method::DELETE,
        }
    }
}

impl fmt::Display for LlmHttpMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LlmHttpRequest {
    pub method: LlmHttpMethod,
    pub url: String,
    pub headers: Vec<(String, String)>,
    pub body: Bytes,
    pub body_for_error: Option<String>,
    pub response_start_timeout_message: Option<String>,
}

impl LlmHttpRequest {
    pub fn new(method: LlmHttpMethod, url: impl Into<String>, body: impl Into<Bytes>) -> Self {
        Self {
            method,
            url: url.into(),
            headers: Vec::new(),
            body: body.into(),
            body_for_error: None,
            response_start_timeout_message: None,
        }
    }

    pub fn post(url: impl Into<String>, body: impl Into<Bytes>) -> Self {
        Self::new(LlmHttpMethod::Post, url, body)
    }

    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    pub fn with_headers<I, K, V>(mut self, headers: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<String>,
    {
        self.headers.extend(
            headers
                .into_iter()
                .map(|(name, value)| (name.into(), value.into())),
        );
        self
    }

    pub fn with_body_for_error(mut self, body: impl Into<String>) -> Self {
        self.body_for_error = Some(body.into());
        self
    }

    pub fn with_response_start_timeout_message(mut self, message: impl Into<String>) -> Self {
        self.response_start_timeout_message = Some(message.into());
        self
    }
}

pub struct LlmHttpResponse {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: LlmHttpBody,
}

impl LlmHttpResponse {
    pub fn is_success(&self) -> bool {
        (200..300).contains(&self.status)
    }
}

impl fmt::Debug for LlmHttpResponse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LlmHttpResponse")
            .field("status", &self.status)
            .field("headers", &self.headers)
            .field("body", &self.body)
            .finish()
    }
}

pub enum LlmHttpBody {
    Buffered(Bytes),
    Streamed(Box<dyn LlmByteStream>),
}

impl LlmHttpBody {
    pub fn buffered(body: impl Into<Bytes>) -> Self {
        Self::Buffered(body.into())
    }

    pub fn streamed(stream: impl LlmByteStream + 'static) -> Self {
        Self::Streamed(Box::new(stream))
    }

    pub fn from_reqwest_response(response: reqwest::Response) -> Self {
        Self::streamed(ReqwestByteStream::new(response))
    }
}

impl fmt::Debug for LlmHttpBody {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Buffered(bytes) => f
                .debug_tuple("Buffered")
                .field(&format_args!("{} bytes", bytes.len()))
                .finish(),
            Self::Streamed(_) => f.write_str("Streamed(<byte stream>)"),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ReqwestLlmHttpTransport {
    client: reqwest::Client,
}

impl Default for ReqwestLlmHttpTransport {
    fn default() -> Self {
        Self::new()
    }
}

impl ReqwestLlmHttpTransport {
    pub fn new() -> Self {
        Self {
            client: crate::timeouts::build_http_client(),
        }
    }

    pub fn from_client(client: reqwest::Client) -> Self {
        Self { client }
    }

    pub fn client(&self) -> &reqwest::Client {
        &self.client
    }
}

#[async_trait]
impl LlmHttpTransport for ReqwestLlmHttpTransport {
    async fn send(
        &self,
        request: LlmHttpRequest,
        timeout: Option<Duration>,
    ) -> Result<LlmHttpResponse, LlmTransportError> {
        let mut http = self
            .client
            .request(request.method.as_reqwest(), &request.url);
        for (name, value) in request.headers {
            http = http.header(name.as_str(), value.as_str());
        }
        http = http.body(request.body);

        let body_for_error = request.body_for_error;
        let timeout_message = request
            .response_start_timeout_message
            .unwrap_or_else(|| "LLM HTTP response start timed out".to_string());

        run_with_timeout(
            async move {
                http.send()
                    .await
                    .map(|response| LlmHttpResponse {
                        status: response.status().as_u16(),
                        headers: header_pairs(response.headers()),
                        body: LlmHttpBody::from_reqwest_response(response),
                    })
                    .map_err(|err| {
                        let error = LlmTransportError::new(format!("HTTP request failed: {err}"))
                            .with_kind(ProviderFailureKind::Transport)
                            .retryable(reqwest_error_is_retryable(&err));
                        if let Some(body) = body_for_error {
                            error.with_request_body(body)
                        } else {
                            error
                        }
                    })
            },
            timeout,
            &timeout_message,
        )
        .await
    }
}

#[derive(Debug)]
pub struct ReqwestByteStream {
    response: reqwest::Response,
}

impl ReqwestByteStream {
    pub fn new(response: reqwest::Response) -> Self {
        Self { response }
    }
}

#[async_trait]
impl LlmByteStream for ReqwestByteStream {
    async fn next_chunk(&mut self) -> Result<Option<Bytes>, LlmTransportError> {
        self.response.chunk().await.map_err(|err| {
            LlmTransportError::new(format!("HTTP response read failed: {err}"))
                .with_kind(ProviderFailureKind::Transport)
                .retryable(reqwest_error_is_retryable(&err))
        })
    }
}

pub async fn read_http_body_bytes(
    body: LlmHttpBody,
    timeout: Option<Duration>,
    timeout_message: &str,
) -> Result<Bytes, LlmTransportError> {
    match body {
        LlmHttpBody::Buffered(bytes) => Ok(bytes),
        LlmHttpBody::Streamed(mut stream) => {
            run_with_timeout(
                async move {
                    let mut body = BytesMut::new();
                    while let Some(chunk) = stream.next_chunk().await? {
                        body.extend_from_slice(&chunk);
                    }
                    Ok(body.freeze())
                },
                timeout,
                timeout_message,
            )
            .await
        }
    }
}

pub async fn read_http_body_text(
    body: LlmHttpBody,
    timeout: Option<Duration>,
    timeout_message: &str,
) -> Result<String, LlmTransportError> {
    let body = read_http_body_bytes(body, timeout, timeout_message).await?;
    Ok(String::from_utf8_lossy(&body).into_owned())
}

pub fn header_contains(headers: &[(String, String)], name: &str, needle: &str) -> bool {
    let needle = needle.to_ascii_lowercase();
    headers.iter().any(|(header_name, value)| {
        header_name.eq_ignore_ascii_case(name) && value.to_ascii_lowercase().contains(&needle)
    })
}

pub fn first_header_value<'a>(headers: &'a [(String, String)], name: &str) -> Option<&'a str> {
    headers
        .iter()
        .find(|(header_name, _)| header_name.eq_ignore_ascii_case(name))
        .map(|(_, value)| value.as_str())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn buffered_response_preserves_status_headers_and_body() {
        let response = LlmHttpResponse {
            status: 429,
            headers: vec![
                ("retry-after".to_string(), "3".to_string()),
                ("set-cookie".to_string(), "a=1".to_string()),
                ("set-cookie".to_string(), "b=2".to_string()),
                ("content-type".to_string(), "application/json".to_string()),
            ],
            body: LlmHttpBody::buffered(r#"{"error":"rate limit"}"#),
        };

        assert_eq!(response.status, 429);
        assert_eq!(
            response
                .headers
                .iter()
                .filter(|(name, _)| name.eq_ignore_ascii_case("set-cookie"))
                .count(),
            2
        );
        assert!(header_contains(
            &response.headers,
            "content-type",
            "application/json"
        ));

        let text = read_http_body_text(response.body, Some(Duration::from_secs(1)), "timed out")
            .await
            .expect("buffered body");
        assert_eq!(text, r#"{"error":"rate limit"}"#);
    }
}
