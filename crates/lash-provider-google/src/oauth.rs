//! Google OAuth authorize-URL / code-exchange / refresh-token flow.
//! Public so Host Applications such as `lash-cli` can drive interactive login.

use lash_provider_auth::{OAuthError, OAuthTokens, now_secs, url_form_encode, urlencoded};

const GOOGLE_CLIENT_ID_ENV: &str = "LASH_GOOGLE_CLIENT_ID";
const GOOGLE_CLIENT_SECRET_ENV: &str = "LASH_GOOGLE_CLIENT_SECRET";
const GOOGLE_AUTH_URL: &str = "https://accounts.google.com/o/oauth2/v2/auth";
const GOOGLE_TOKEN_URL: &str = "https://oauth2.googleapis.com/token";
const GOOGLE_REDIRECT_URI: &str = "https://codeassist.google.com/authcode";
const GOOGLE_SCOPES: &str = "https://www.googleapis.com/auth/cloud-platform https://www.googleapis.com/auth/userinfo.email https://www.googleapis.com/auth/userinfo.profile";
const GOOGLE_PROMPT: &str = "consent select_account";

fn google_client_id() -> Option<String> {
    std::env::var(GOOGLE_CLIENT_ID_ENV)
        .ok()
        .filter(|v| !v.is_empty())
}

fn google_client_secret() -> Option<String> {
    std::env::var(GOOGLE_CLIENT_SECRET_ENV)
        .ok()
        .filter(|v| !v.is_empty())
}

fn google_client_credentials() -> Result<(String, String), OAuthError> {
    match (google_client_id(), google_client_secret()) {
        (Some(client_id), Some(client_secret)) => Ok((client_id, client_secret)),
        (None, None) => Err(OAuthError::TokenExchange(format!(
            "Missing Google OAuth env config: set both {} and {}.",
            GOOGLE_CLIENT_ID_ENV, GOOGLE_CLIENT_SECRET_ENV
        ))),
        _ => Err(OAuthError::TokenExchange(format!(
            "Invalid Google OAuth env config: set both {} and {}, or set neither.",
            GOOGLE_CLIENT_ID_ENV, GOOGLE_CLIENT_SECRET_ENV
        ))),
    }
}

/// Build the Google OAuth authorization URL for manual code entry.
pub fn authorize_url(challenge: &str) -> Result<String, OAuthError> {
    let state = uuid::Uuid::new_v4().to_string();
    let (client_id, _) = google_client_credentials()?;
    Ok(format!(
        "{}?client_id={}&response_type=code&redirect_uri={}&scope={}&access_type=offline&prompt={}&code_challenge={}&code_challenge_method=S256&state={}",
        GOOGLE_AUTH_URL,
        client_id,
        urlencoded(GOOGLE_REDIRECT_URI),
        urlencoded(GOOGLE_SCOPES),
        urlencoded(GOOGLE_PROMPT),
        challenge,
        state,
    ))
}

/// Exchange an authorization code (or a redirect URL containing
/// `code=...`) for tokens.
pub async fn exchange_code(code: &str, verifier: &str) -> Result<OAuthTokens, OAuthError> {
    let (client_id, client_secret) = google_client_credentials()?;
    let auth_code = extract_auth_code(code);
    let client = reqwest::Client::new();
    let resp = client
        .post(GOOGLE_TOKEN_URL)
        .header("Content-Type", "application/x-www-form-urlencoded")
        .body(url_form_encode(&[
            ("grant_type", "authorization_code"),
            ("code", auth_code.as_str()),
            ("redirect_uri", GOOGLE_REDIRECT_URI),
            ("client_id", client_id.as_str()),
            ("client_secret", client_secret.as_str()),
            ("code_verifier", verifier),
        ]))
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
    parse_token_response(&body)
}

/// Refresh Google OAuth tokens.
pub async fn refresh_tokens(refresh: &str) -> Result<OAuthTokens, OAuthError> {
    let (client_id, client_secret) = google_client_credentials()?;
    let client = reqwest::Client::new();
    let resp = client
        .post(GOOGLE_TOKEN_URL)
        .header("Content-Type", "application/x-www-form-urlencoded")
        .body(url_form_encode(&[
            ("grant_type", "refresh_token"),
            ("refresh_token", refresh),
            ("client_id", client_id.as_str()),
            ("client_secret", client_secret.as_str()),
        ]))
        .send()
        .await?;

    let status = resp.status();
    let body: serde_json::Value = resp.json().await?;
    if !status.is_success() {
        let err = body["error_description"]
            .as_str()
            .or(body["error"].as_str())
            .unwrap_or("token refresh failed");
        return Err(OAuthError::TokenEndpoint {
            status: status.as_u16(),
            message: err.to_string(),
        });
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

fn extract_auth_code(input: &str) -> String {
    let trimmed = input.trim();
    if let Some(code) = lash_provider_auth::extract_query_param(trimmed, "code") {
        return code;
    }
    if let Some(pos) = trimmed.find("code=") {
        let rest = &trimmed[pos + 5..];
        let end = rest
            .char_indices()
            .find_map(|(i, ch)| (ch == '&' || ch == '#').then_some(i))
            .unwrap_or(rest.len());
        return lash_provider_auth::extract_query_param(&format!("?code={}", &rest[..end]), "code")
            .unwrap_or_else(|| rest[..end].to_string());
    }
    trimmed.to_string()
}
