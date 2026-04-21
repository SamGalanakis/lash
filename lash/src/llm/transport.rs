//! Transport-level error type shared by all provider impls. The
//! `LlmTransport` trait that used to live here has merged into
//! [`crate::provider::Provider`] — each concrete provider now owns its
//! own wire code and exposes a single `complete` method.

#[derive(Debug, thiserror::Error, Clone)]
#[error("{message}")]
pub struct LlmTransportError {
    pub message: String,
    pub retryable: bool,
    pub raw: Option<String>,
    pub code: Option<String>,
    pub request_body: Option<String>,
}

impl LlmTransportError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            retryable: false,
            raw: None,
            code: None,
            request_body: None,
        }
    }

    pub fn retryable(mut self, retryable: bool) -> Self {
        self.retryable = retryable;
        self
    }

    pub fn with_raw(mut self, raw: impl Into<String>) -> Self {
        self.raw = Some(raw.into());
        self
    }

    pub fn with_code(mut self, code: impl Into<String>) -> Self {
        self.code = Some(code.into());
        self
    }

    pub fn with_request_body(mut self, request_body: impl Into<String>) -> Self {
        self.request_body = Some(request_body.into());
        self
    }
}
