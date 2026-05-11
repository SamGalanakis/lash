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
