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
    TokenEndpoint { status: u16, message: String },
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
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
