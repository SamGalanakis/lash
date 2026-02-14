use base64::Engine;
use sha2::{Digest, Sha256};

const CLIENT_ID: &str = "9d1c250a-e61b-44d9-88ed-5944d1962f5e";
const TOKEN_URL: &str = "https://console.anthropic.com/v1/oauth/token";
const AUTH_URL: &str = "https://claude.ai/oauth/authorize";
const REDIRECT_URI: &str = "https://console.anthropic.com/oauth/code/callback";
const SCOPES: &str = "org:create_api_key user:profile user:inference";

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
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

/// Generate a PKCE code verifier and challenge pair.
pub fn generate_pkce() -> (String, String) {
    let verifier_bytes: Vec<u8> = (0..32).map(|_| rand_byte()).collect();
    let verifier = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(&verifier_bytes);

    let mut hasher = Sha256::new();
    hasher.update(verifier.as_bytes());
    let digest = hasher.finalize();
    let challenge = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(digest);

    (verifier, challenge)
}

/// Build the authorization URL for the browser.
/// The `verifier` is embedded in the `state` param (matches opencode's convention).
pub fn authorize_url(challenge: &str, verifier: &str) -> String {
    format!(
        "{}?code=true&client_id={}&response_type=code&redirect_uri={}&scope={}&code_challenge={}&code_challenge_method=S256&state={}",
        AUTH_URL,
        CLIENT_ID,
        urlencoded(REDIRECT_URI),
        urlencoded(SCOPES),
        challenge,
        verifier,
    )
}

/// Exchange an authorization code for tokens.
/// The pasted code may contain `code#state` — we split on `#`.
pub async fn exchange_code(code: &str, verifier: &str) -> Result<OAuthTokens, OAuthError> {
    let parts: Vec<&str> = code.splitn(2, '#').collect();
    let auth_code = parts[0];
    let state = parts.get(1).copied().unwrap_or("");

    let client = reqwest::Client::new();
    let resp = client
        .post(TOKEN_URL)
        .json(&serde_json::json!({
            "grant_type": "authorization_code",
            "client_id": CLIENT_ID,
            "code": auth_code,
            "state": state,
            "redirect_uri": REDIRECT_URI,
            "code_verifier": verifier,
        }))
        .send()
        .await?;

    let status = resp.status();
    let body: serde_json::Value = resp.json().await?;

    if !status.is_success() {
        let err = body["error_description"]
            .as_str()
            .or(body["error"].as_str())
            .unwrap_or("unknown error");
        return Err(OAuthError::TokenExchange(err.to_string()));
    }

    parse_token_response(&body)
}

/// Refresh an expired access token.
pub async fn refresh_tokens(refresh: &str) -> Result<OAuthTokens, OAuthError> {
    let client = reqwest::Client::new();
    let resp = client
        .post(TOKEN_URL)
        .json(&serde_json::json!({
            "grant_type": "refresh_token",
            "client_id": CLIENT_ID,
            "refresh_token": refresh,
        }))
        .send()
        .await?;

    let status = resp.status();
    let body: serde_json::Value = resp.json().await?;

    if !status.is_success() {
        let err = body["error_description"]
            .as_str()
            .or(body["error"].as_str())
            .unwrap_or("unknown error");
        return Err(OAuthError::TokenExchange(err.to_string()));
    }

    // Refresh response may not include a new refresh_token; keep the old one.
    let now = now_secs();
    let expires_in = body["expires_in"].as_u64().unwrap_or(3600);

    Ok(OAuthTokens {
        access_token: body["access_token"]
            .as_str()
            .ok_or_else(|| OAuthError::TokenExchange("missing access_token".into()))?
            .to_string(),
        refresh_token: body["refresh_token"]
            .as_str()
            .unwrap_or(refresh)
            .to_string(),
        expires_at: now + expires_in,
    })
}

fn parse_token_response(body: &serde_json::Value) -> Result<OAuthTokens, OAuthError> {
    let now = now_secs();
    let expires_in = body["expires_in"].as_u64().unwrap_or(3600);

    Ok(OAuthTokens {
        access_token: body["access_token"]
            .as_str()
            .ok_or_else(|| OAuthError::TokenExchange("missing access_token".into()))?
            .to_string(),
        refresh_token: body["refresh_token"]
            .as_str()
            .ok_or_else(|| OAuthError::TokenExchange("missing refresh_token".into()))?
            .to_string(),
        expires_at: now + expires_in,
    })
}

// ── Codex (OpenAI) device-code OAuth ─────────────────────────────

const CODEX_CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
const CODEX_TOKEN_URL: &str = "https://auth.openai.com/oauth/token";
const CODEX_DEVICE_CODE_URL: &str = "https://auth.openai.com/api/accounts/deviceauth/usercode";
const CODEX_DEVICE_POLL_URL: &str = "https://auth.openai.com/api/accounts/deviceauth/token";
const CODEX_DEVICE_CALLBACK: &str = "https://auth.openai.com/deviceauth/callback";
pub const CODEX_DEVICE_VERIFY_URL: &str = "https://auth.openai.com/codex/device";

#[derive(Debug)]
pub struct CodexDeviceCode {
    pub device_auth_id: String,
    pub user_code: String,
    pub interval: u64,
}

#[derive(Debug)]
pub struct CodexOAuthTokens {
    pub access_token: String,
    pub refresh_token: String,
    pub expires_at: u64,
    pub account_id: Option<String>,
}

/// Request a device code from OpenAI for the Codex auth flow.
pub async fn codex_request_device_code() -> Result<CodexDeviceCode, OAuthError> {
    let client = reqwest::Client::new();
    let resp = client
        .post(CODEX_DEVICE_CODE_URL)
        .json(&serde_json::json!({ "client_id": CODEX_CLIENT_ID }))
        .send()
        .await?;

    let status = resp.status();
    let body: serde_json::Value = resp.json().await?;

    if !status.is_success() {
        let err = body["error"]
            .as_str()
            .unwrap_or("failed to initiate device authorization");
        return Err(OAuthError::TokenExchange(err.to_string()));
    }

    Ok(CodexDeviceCode {
        device_auth_id: body["device_auth_id"]
            .as_str()
            .ok_or_else(|| OAuthError::TokenExchange("missing device_auth_id".into()))?
            .to_string(),
        user_code: body["user_code"]
            .as_str()
            .ok_or_else(|| OAuthError::TokenExchange("missing user_code".into()))?
            .to_string(),
        interval: body["interval"]
            .as_str()
            .and_then(|s| s.parse().ok())
            .or(body["interval"].as_u64())
            .map(|v| v.max(1))
            .unwrap_or(5),
    })
}

/// Poll the device auth endpoint. Returns `Ok(Some((auth_code, code_verifier)))` when approved,
/// `Ok(None)` when still pending, `Err` on failure.
pub async fn codex_poll_device_auth(
    device_auth_id: &str,
    user_code: &str,
) -> Result<Option<(String, String)>, OAuthError> {
    let client = reqwest::Client::new();
    let resp = client
        .post(CODEX_DEVICE_POLL_URL)
        .json(&serde_json::json!({
            "device_auth_id": device_auth_id,
            "user_code": user_code,
        }))
        .send()
        .await?;

    if resp.status().is_success() {
        let body: serde_json::Value = resp.json().await?;
        let auth_code = body["authorization_code"]
            .as_str()
            .ok_or_else(|| OAuthError::TokenExchange("missing authorization_code".into()))?
            .to_string();
        let code_verifier = body["code_verifier"]
            .as_str()
            .ok_or_else(|| OAuthError::TokenExchange("missing code_verifier".into()))?
            .to_string();
        Ok(Some((auth_code, code_verifier)))
    } else if resp.status().as_u16() == 403 || resp.status().as_u16() == 404 {
        // Still pending
        Ok(None)
    } else {
        let body: serde_json::Value = resp.json().await.unwrap_or_default();
        let err = body["error"]
            .as_str()
            .unwrap_or("device auth polling failed");
        Err(OAuthError::TokenExchange(err.to_string()))
    }
}

/// Exchange the device authorization code for tokens.
/// Uses form-urlencoded (not JSON) as required by OpenAI's token endpoint.
pub async fn codex_exchange_code(
    code: &str,
    code_verifier: &str,
) -> Result<CodexOAuthTokens, OAuthError> {
    let client = reqwest::Client::new();
    let resp = client
        .post(CODEX_TOKEN_URL)
        .header("Content-Type", "application/x-www-form-urlencoded")
        .body(
            url_form_encode(&[
                ("grant_type", "authorization_code"),
                ("code", code),
                ("redirect_uri", CODEX_DEVICE_CALLBACK),
                ("client_id", CODEX_CLIENT_ID),
                ("code_verifier", code_verifier),
            ]),
        )
        .send()
        .await?;

    let status = resp.status();
    let body: serde_json::Value = resp.json().await?;

    if !status.is_success() {
        let err = body["error_description"]
            .as_str()
            .or(body["error"].as_str())
            .unwrap_or("token exchange failed");
        return Err(OAuthError::TokenExchange(err.to_string()));
    }

    let now = now_secs();
    let expires_in = body["expires_in"].as_u64().unwrap_or(3600);

    let access_token = body["access_token"]
        .as_str()
        .ok_or_else(|| OAuthError::TokenExchange("missing access_token".into()))?
        .to_string();
    let refresh_token = body["refresh_token"]
        .as_str()
        .ok_or_else(|| OAuthError::TokenExchange("missing refresh_token".into()))?
        .to_string();

    // Extract account ID from id_token or access_token JWT
    let account_id = body["id_token"]
        .as_str()
        .and_then(extract_codex_account_id)
        .or_else(|| extract_codex_account_id(&access_token));

    Ok(CodexOAuthTokens {
        access_token,
        refresh_token,
        expires_at: now + expires_in,
        account_id,
    })
}

/// Refresh Codex OAuth tokens. Uses form-urlencoded.
pub async fn codex_refresh_tokens(refresh: &str) -> Result<OAuthTokens, OAuthError> {
    let client = reqwest::Client::new();
    let resp = client
        .post(CODEX_TOKEN_URL)
        .header("Content-Type", "application/x-www-form-urlencoded")
        .body(
            url_form_encode(&[
                ("grant_type", "refresh_token"),
                ("refresh_token", refresh),
                ("client_id", CODEX_CLIENT_ID),
            ]),
        )
        .send()
        .await?;

    let status = resp.status();
    let body: serde_json::Value = resp.json().await?;

    if !status.is_success() {
        let err = body["error_description"]
            .as_str()
            .or(body["error"].as_str())
            .unwrap_or("token refresh failed");
        return Err(OAuthError::TokenExchange(err.to_string()));
    }

    let now = now_secs();
    let expires_in = body["expires_in"].as_u64().unwrap_or(3600);

    Ok(OAuthTokens {
        access_token: body["access_token"]
            .as_str()
            .ok_or_else(|| OAuthError::TokenExchange("missing access_token".into()))?
            .to_string(),
        refresh_token: body["refresh_token"]
            .as_str()
            .unwrap_or(refresh)
            .to_string(),
        expires_at: now + expires_in,
    })
}

/// Extract the ChatGPT account ID from a JWT token (no crypto verification needed).
fn extract_codex_account_id(jwt: &str) -> Option<String> {
    let parts: Vec<&str> = jwt.split('.').collect();
    if parts.len() != 3 {
        return None;
    }
    // JWT payload is base64url-encoded; try with and without padding
    let payload = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(parts[1])
        .or_else(|_| base64::engine::general_purpose::URL_SAFE.decode(parts[1]))
        .ok()?;
    let claims: serde_json::Value = serde_json::from_slice(&payload).ok()?;

    // Try direct field first
    if let Some(id) = claims["chatgpt_account_id"].as_str() {
        if !id.is_empty() {
            return Some(id.to_string());
        }
    }
    // Try nested auth claim
    if let Some(id) = claims["https://api.openai.com/auth"]["chatgpt_account_id"].as_str() {
        if !id.is_empty() {
            return Some(id.to_string());
        }
    }
    // Fall back to first organization ID
    if let Some(orgs) = claims["organizations"].as_array() {
        if let Some(org) = orgs.first() {
            if let Some(id) = org["id"].as_str() {
                if !id.is_empty() {
                    return Some(id.to_string());
                }
            }
        }
    }
    None
}

/// Simple form-urlencoded encoder.
fn url_form_encode(pairs: &[(&str, &str)]) -> String {
    pairs
        .iter()
        .map(|(k, v)| format!("{}={}", form_escape(k), form_escape(v)))
        .collect::<Vec<_>>()
        .join("&")
}

/// Percent-encode a value for application/x-www-form-urlencoded.
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

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

/// Minimal percent-encoding for URL query parameters.
fn urlencoded(s: &str) -> String {
    s.replace('%', "%25")
        .replace(' ', "%20")
        .replace(':', "%3A")
        .replace('/', "%2F")
}

/// Simple pseudo-random byte using thread-based entropy.
fn rand_byte() -> u8 {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};
    let s = RandomState::new();
    let mut h = s.build_hasher();
    h.write_u64(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64,
    );
    h.finish() as u8
}
