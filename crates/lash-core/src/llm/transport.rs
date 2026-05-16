//! Transport-level failure type shared by provider transport components.

#[derive(Debug, thiserror::Error, Clone)]
#[error("{message}")]
pub struct ProviderFailure {
    pub kind: ProviderFailureKind,
    pub message: String,
    pub retryable: bool,
    pub status: Option<u16>,
    pub raw: Option<String>,
    pub code: Option<String>,
    pub terminal_reason: lash_sansio::llm::types::LlmTerminalReason,
    pub headers: Vec<(String, String)>,
    pub retry_after: Option<std::time::Duration>,
    pub request_body: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderFailureKind {
    Transport,
    Timeout,
    Http,
    Stream,
    Auth,
    Validation,
    Quota,
    Unsupported,
    Unknown,
}

impl ProviderFailure {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            kind: ProviderFailureKind::Unknown,
            message: message.into(),
            retryable: false,
            status: None,
            raw: None,
            code: None,
            terminal_reason: lash_sansio::llm::types::LlmTerminalReason::ProviderError,
            headers: Vec::new(),
            retry_after: None,
            request_body: None,
        }
    }

    pub fn with_kind(mut self, kind: ProviderFailureKind) -> Self {
        self.kind = kind;
        self
    }

    pub fn retryable(mut self, retryable: bool) -> Self {
        self.retryable = retryable;
        self
    }

    pub fn with_status(mut self, status: u16) -> Self {
        self.status = Some(status);
        if self.code.is_none() {
            self.code = Some(status.to_string());
        }
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

    pub fn with_terminal_reason(
        mut self,
        reason: lash_sansio::llm::types::LlmTerminalReason,
    ) -> Self {
        self.terminal_reason = reason;
        self
    }

    pub fn with_headers(mut self, headers: &reqwest::header::HeaderMap) -> Self {
        self.headers = headers
            .iter()
            .filter_map(|(name, value)| {
                value
                    .to_str()
                    .ok()
                    .map(|value| (name.as_str().to_string(), value.to_string()))
            })
            .collect();
        self.retry_after = retry_after_from_headers(&self.headers);
        self
    }

    pub fn with_retry_after(mut self, retry_after: std::time::Duration) -> Self {
        self.retry_after = Some(retry_after);
        self
    }

    pub fn with_request_body(mut self, request_body: impl Into<String>) -> Self {
        self.request_body = Some(request_body.into());
        self
    }
}

pub fn retry_after_from_headers(headers: &[(String, String)]) -> Option<std::time::Duration> {
    let value = headers
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case("retry-after"))?
        .1
        .trim();
    if let Ok(seconds) = value.parse::<u64>() {
        return Some(std::time::Duration::from_secs(seconds));
    }
    None
}

pub type LlmTransportError = ProviderFailure;
