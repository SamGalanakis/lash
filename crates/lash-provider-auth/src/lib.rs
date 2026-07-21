//! Shared OAuth primitives used by provider crates that implement
//! OAuth-based auth (Codex, Google). API-key backends bypass this module.
//!
//! Provider-specific endpoints, device-code flows, PKCE helpers, and
//! refresh logic live in each provider crate under `oauth.rs`.

use base64::Engine;
use sha2::{Digest, Sha256};

mod credential;

pub use credential::{
    Credential, CredentialCallError, CredentialError, CredentialErrorKind, CredentialExecuteError,
    CredentialManager, CredentialPolicy, CredentialRefresher, Lease, RefreshCause,
};

#[derive(Debug)]
pub struct OAuthTokens {
    pub access_token: String,
    pub refresh_token: String,
    pub expires_at: u64,
}

#[derive(Debug, thiserror::Error)]
pub enum OAuthError {
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("Token exchange failed: {0}")]
    TokenExchange(String),
    #[error("Token endpoint returned HTTP {status}: {message}")]
    TokenEndpoint {
        status: u16,
        message: String,
        error_code: Option<OAuthTokenErrorCode>,
    },
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OAuthTokenErrorCode {
    InvalidGrant,
    InvalidClient,
    InvalidRequest,
    UnauthorizedClient,
    UnsupportedGrantType,
    InvalidScope,
}

impl OAuthTokenErrorCode {
    fn parse(value: &str) -> Option<Self> {
        match value {
            "invalid_grant" => Some(Self::InvalidGrant),
            "invalid_client" => Some(Self::InvalidClient),
            "invalid_request" => Some(Self::InvalidRequest),
            "unauthorized_client" => Some(Self::UnauthorizedClient),
            "unsupported_grant_type" => Some(Self::UnsupportedGrantType),
            "invalid_scope" => Some(Self::InvalidScope),
            _ => None,
        }
    }
}

impl OAuthError {
    pub fn token_endpoint(status: u16, response_body: &str, fallback_message: &str) -> Self {
        let body = serde_json::from_str::<serde_json::Value>(response_body).ok();
        let error_code = body
            .as_ref()
            .and_then(|body| body["error"].as_str())
            .and_then(OAuthTokenErrorCode::parse);
        let message = body
            .as_ref()
            .and_then(|body| {
                body["error_description"]
                    .as_str()
                    .or(body["error"].as_str())
            })
            .map(str::to_owned)
            .unwrap_or_else(|| {
                let response_body = response_body.trim();
                if response_body.is_empty() {
                    fallback_message.to_owned()
                } else {
                    response_body.to_owned()
                }
            });
        Self::TokenEndpoint {
            status,
            message,
            error_code,
        }
    }
}

/// Convert a provider OAuth refresh failure into the credential error
/// categories used by credential refresh and retry policy.
pub fn classify_oauth_refresh_error(error: OAuthError) -> CredentialError {
    if matches!(
        &error,
        OAuthError::TokenEndpoint {
            error_code: Some(OAuthTokenErrorCode::InvalidGrant),
            ..
        }
    ) {
        CredentialError::invalid_grant()
    } else if matches!(
        error,
        OAuthError::Http(_)
            | OAuthError::TokenEndpoint {
                status: 408 | 429 | 500..=599,
                ..
            }
    ) {
        CredentialError::transient()
    } else {
        CredentialError::new(CredentialErrorKind::Other, false)
    }
}

/// Generate a PKCE code verifier and challenge pair. PKCE verifier is
/// 32 bytes of OS entropy (via two UUID v4s) base64url-encoded; the
/// challenge is its SHA-256 base64url-encoded.
pub fn generate_pkce() -> (String, String) {
    let mut verifier_bytes = Vec::with_capacity(32);
    verifier_bytes.extend_from_slice(uuid::Uuid::new_v4().as_bytes());
    verifier_bytes.extend_from_slice(uuid::Uuid::new_v4().as_bytes());
    let verifier = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(&verifier_bytes);

    let mut hasher = Sha256::new();
    hasher.update(verifier.as_bytes());
    let digest = hasher.finalize();
    let challenge = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(digest);

    (verifier, challenge)
}

pub fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

/// Form-urlencoded body encoder for OAuth token endpoints.
pub fn url_form_encode(pairs: &[(&str, &str)]) -> String {
    pairs
        .iter()
        .map(|(k, v)| format!("{}={}", form_escape(k), form_escape(v)))
        .collect::<Vec<_>>()
        .join("&")
}

/// Percent-encode a value for `application/x-www-form-urlencoded`.
fn form_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char);
            }
            _ => {
                out.push_str(&format!("%{:02X}", b));
            }
        }
    }
    out
}

/// Minimal percent-encoding for URL query parameters.
pub fn urlencoded(s: &str) -> String {
    s.replace('%', "%25")
        .replace(' ', "%20")
        .replace(':', "%3A")
        .replace('/', "%2F")
}

/// Percent-decode a query-string value (handles `+` → space).
pub fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0usize;
    while i < bytes.len() {
        match bytes[i] {
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            b'%' if i + 2 < bytes.len() => {
                if let (Some(hi), Some(lo)) = (hex_nibble(bytes[i + 1]), hex_nibble(bytes[i + 2])) {
                    out.push((hi << 4) | lo);
                    i += 3;
                } else {
                    out.push(bytes[i]);
                    i += 1;
                }
            }
            b => {
                out.push(b);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).to_string()
}

fn hex_nibble(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(10 + (b - b'a')),
        b'A'..=b'F' => Some(10 + (b - b'A')),
        _ => None,
    }
}

/// Extract the value of a given query parameter from a URL or raw
/// query-string. Returns `None` if the key is absent.
pub fn extract_query_param(url_or_query: &str, key: &str) -> Option<String> {
    let query = if let Some(idx) = url_or_query.find('?') {
        &url_or_query[idx + 1..]
    } else {
        url_or_query
    };
    for pair in query.split('&') {
        if pair.is_empty() {
            continue;
        }
        let mut parts = pair.splitn(2, '=');
        let k = parts.next().unwrap_or("");
        let v = parts.next().unwrap_or("");
        if k == key {
            return Some(percent_decode(v));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn token_endpoint_parses_all_rfc_6749_error_codes() {
        let cases = [
            ("invalid_grant", OAuthTokenErrorCode::InvalidGrant),
            ("invalid_client", OAuthTokenErrorCode::InvalidClient),
            ("invalid_request", OAuthTokenErrorCode::InvalidRequest),
            (
                "unauthorized_client",
                OAuthTokenErrorCode::UnauthorizedClient,
            ),
            (
                "unsupported_grant_type",
                OAuthTokenErrorCode::UnsupportedGrantType,
            ),
            ("invalid_scope", OAuthTokenErrorCode::InvalidScope),
        ];

        for (code, expected) in cases {
            let error = OAuthError::token_endpoint(
                400,
                &format!(r#"{{"error":"{code}"}}"#),
                "token refresh failed",
            );

            assert!(matches!(
                error,
                OAuthError::TokenEndpoint {
                    status: 400,
                    error_code: Some(actual),
                    ..
                } if actual == expected
            ));
        }
    }

    #[test]
    fn invalid_grant_maps_to_visible_non_retryable_credential_error() {
        let error =
            OAuthError::token_endpoint(400, r#"{"error":"invalid_grant"}"#, "token refresh failed");

        let error = classify_oauth_refresh_error(error);

        assert_eq!(error.kind, CredentialErrorKind::InvalidGrant);
        assert!(!error.retryable);
        assert!(error.to_string().contains("sign in again"));
    }

    #[test]
    fn unparseable_body_mentioning_invalid_grant_is_not_invalid_grant() {
        let error = OAuthError::token_endpoint(
            400,
            "<html>proxy could not determine whether this was an invalid grant</html>",
            "token refresh failed",
        );

        assert!(matches!(
            &error,
            OAuthError::TokenEndpoint { status: 400, .. }
        ));
        let error = classify_oauth_refresh_error(error);
        assert_eq!(error.kind, CredentialErrorKind::Other);
        assert!(!error.retryable);
    }

    #[test]
    fn ambiguous_400_error_codes_are_not_invalid_grant() {
        let bodies = [
            r#"{"error":"invalid_client"}"#,
            r#"{"error":"unauthorized_client"}"#,
            r#"{"error":"provider_extension","error_description":"invalid_grant"}"#,
        ];

        for body in bodies {
            let error = classify_oauth_refresh_error(OAuthError::token_endpoint(
                400,
                body,
                "token refresh failed",
            ));

            assert_eq!(error.kind, CredentialErrorKind::Other);
            assert!(!error.retryable);
        }
    }

    #[test]
    fn rate_limit_and_server_errors_are_retryable() {
        for status in [429, 500, 503, 599] {
            let error = classify_oauth_refresh_error(OAuthError::token_endpoint(
                status,
                r#"{"error":"provider_failure"}"#,
                "token refresh failed",
            ));

            assert_eq!(error.kind, CredentialErrorKind::Transient);
            assert!(error.retryable);
        }
    }

    #[tokio::test]
    async fn network_failure_is_retryable() {
        let listener = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let address = listener.local_addr().unwrap();
        drop(listener);
        let request_error = reqwest::get(format!("http://{address}")).await.unwrap_err();
        assert!(request_error.is_connect());

        let error = classify_oauth_refresh_error(OAuthError::Http(request_error));

        assert_eq!(error.kind, CredentialErrorKind::Transient);
        assert!(error.retryable);
    }

    #[tokio::test]
    async fn timeout_failure_is_retryable() {
        let listener = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let address = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let (_stream, _) = listener.accept().unwrap();
            std::thread::sleep(Duration::from_millis(250));
        });
        let client = reqwest::Client::builder()
            .timeout(Duration::from_millis(25))
            .build()
            .unwrap();
        let request_error = client
            .get(format!("http://{address}"))
            .send()
            .await
            .unwrap_err();
        assert!(request_error.is_timeout());
        server.join().unwrap();

        let error = classify_oauth_refresh_error(OAuthError::Http(request_error));

        assert_eq!(error.kind, CredentialErrorKind::Transient);
        assert!(error.retryable);
    }
}
