use crate::support::*;

#[derive(Clone, Debug)]
pub struct OpenAiCompatibleProvider {
    pub api_key: String,
    pub base_url: String,
    pub options: ProviderOptions,
    pub cache_retention: OpenAiCacheRetention,
    pub(crate) client: reqwest::Client,
}

#[derive(Clone, Debug)]
pub struct OpenAiProvider {
    pub(crate) inner: OpenAiCompatibleProvider,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, serde::Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum OpenAiCacheRetention {
    None,
    #[default]
    Short,
    Long,
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
