use crate::support::*;

#[derive(Clone, Debug)]
pub struct OpenAiCompatibleProvider {
    pub api_key: String,
    pub base_url: String,
    pub options: ProviderOptions,
    pub(crate) client: reqwest::Client,
}

#[derive(Clone, Debug)]
pub struct OpenAiProvider {
    pub(crate) inner: OpenAiCompatibleProvider,
}

#[derive(Clone, Debug)]
pub(crate) struct OpenAiModelPolicy {
    pub(crate) base_url: String,
}

impl OpenAiModelPolicy {
    pub(crate) fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
        }
    }
}
