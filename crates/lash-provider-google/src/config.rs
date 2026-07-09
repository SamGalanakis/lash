//! Provider construction: the [`GoogleOAuthProvider`] struct, its builders,
//! endpoint-URL helpers, the uploaded-attachment cache types, and the
//! [`GoogleOAuthProviderFactory`].

use std::collections::HashMap;
use std::sync::{Arc, LazyLock, OnceLock};

use crate::support::*;

pub(crate) const CODE_ASSIST_ENDPOINT: &str = "https://cloudcode-pa.googleapis.com";
pub(crate) const CODE_ASSIST_API_VERSION: &str = "v1internal";

pub(crate) static DEFAULT_HTTP_TRANSPORT: LazyLock<Arc<dyn LlmHttpTransport>> =
    LazyLock::new(|| Arc::new(ReqwestLlmHttpTransport::new()));

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct UploadedAttachmentCacheKey {
    pub(crate) project_id: String,
    pub(crate) mime: String,
    pub(crate) hash: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct UploadedAttachmentRef {
    pub(crate) uri: String,
}

/// Google OAuth (Gemini via Code Assist) provider.
#[derive(Clone, Debug)]
pub struct GoogleOAuthProvider {
    pub access_token: String,
    pub refresh_token: String,
    pub expires_at: u64,
    pub project_id: Option<String>,
    pub options: ProviderOptions,
    pub(crate) transport: Arc<dyn LlmHttpTransport>,
}

impl GoogleOAuthProvider {
    pub(crate) fn uploaded_attachment_cache()
    -> &'static tokio::sync::Mutex<HashMap<UploadedAttachmentCacheKey, UploadedAttachmentRef>> {
        static CACHE: OnceLock<
            tokio::sync::Mutex<HashMap<UploadedAttachmentCacheKey, UploadedAttachmentRef>>,
        > = OnceLock::new();
        CACHE.get_or_init(|| tokio::sync::Mutex::new(HashMap::new()))
    }

    pub fn new(
        access_token: impl Into<String>,
        refresh_token: impl Into<String>,
        expires_at: u64,
    ) -> Self {
        Self {
            access_token: access_token.into(),
            refresh_token: refresh_token.into(),
            expires_at,
            project_id: None,
            options: ProviderOptions::default(),
            transport: Arc::clone(&DEFAULT_HTTP_TRANSPORT),
        }
    }

    pub fn with_project_id(mut self, project_id: Option<String>) -> Self {
        self.project_id = project_id;
        self
    }

    pub fn with_options(mut self, options: ProviderOptions) -> Self {
        self.options = options;
        self
    }

    pub fn with_transport(mut self, transport: Arc<dyn LlmHttpTransport>) -> Self {
        self.transport = transport;
        self
    }

    pub fn with_client(mut self, client: std::sync::Arc<reqwest::Client>) -> Self {
        self.transport = Arc::new(ReqwestLlmHttpTransport::from_client((*client).clone()));
        self
    }

    pub(crate) fn endpoint_base_url() -> String {
        let endpoint = std::env::var("CODE_ASSIST_ENDPOINT")
            .unwrap_or_else(|_| CODE_ASSIST_ENDPOINT.to_string());
        let version = std::env::var("CODE_ASSIST_API_VERSION")
            .unwrap_or_else(|_| CODE_ASSIST_API_VERSION.to_string());
        format!("{endpoint}/{version}")
    }

    pub(crate) fn method_url(method: &str) -> String {
        format!("{}:{}", Self::endpoint_base_url(), method)
    }

    pub fn into_components(self) -> ProviderComponents {
        ProviderComponents::new(Box::new(self))
    }
}

#[derive(Deserialize)]
struct GoogleProviderConfig {
    access_token: String,
    refresh_token: String,
    expires_at: u64,
    #[serde(default)]
    project_id: Option<String>,
    #[serde(default)]
    options: ProviderOptions,
}

pub struct GoogleOAuthProviderFactory;

impl ProviderFactory for GoogleOAuthProviderFactory {
    fn kind(&self) -> &'static str {
        "google_oauth"
    }
    fn deserialize(&self, config: serde_json::Value) -> Result<ProviderComponents, String> {
        let cfg: GoogleProviderConfig =
            serde_json::from_value(config).map_err(|err| err.to_string())?;
        Ok(GoogleOAuthProvider {
            access_token: cfg.access_token,
            refresh_token: cfg.refresh_token,
            expires_at: cfg.expires_at,
            project_id: cfg.project_id,
            options: cfg.options,
            transport: Arc::clone(&DEFAULT_HTTP_TRANSPORT),
        }
        .into_components())
    }
}
