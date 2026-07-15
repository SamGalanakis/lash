use crate::support::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpenAiCompatMaxTokensField {
    MaxTokens,
    MaxCompletionTokens,
    MaxOutputTokens,
    Omit,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpenAiCompatReasoningFormat {
    None,
    OpenAi,
    OpenRouter,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OpenAiCompat {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_fields: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens_field: Option<OpenAiCompatMaxTokensField>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_format: Option<OpenAiCompatReasoningFormat>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_session_affinity: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_cache_key: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_cache_retention: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strict_tools: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub streaming_usage: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub developer_role: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub schema_capabilities: Option<ProviderSchemaCapabilities>,
}

impl OpenAiCompat {
    /// Explicit endpoint capabilities for OpenRouter and compatible proxies.
    pub fn openrouter() -> Self {
        Self {
            reasoning_format: Some(OpenAiCompatReasoningFormat::OpenRouter),
            cache_session_affinity: Some(true),
            ..Self::default()
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct OpenAiResolvedCompat {
    pub(crate) request_fields: bool,
    pub(crate) max_tokens_field: OpenAiCompatMaxTokensField,
    pub(crate) reasoning_format: OpenAiCompatReasoningFormat,
    pub(crate) cache_session_affinity: bool,
    pub(crate) prompt_cache_key: bool,
    pub(crate) prompt_cache_retention: bool,
    pub(crate) strict_tools: bool,
    pub(crate) store: bool,
    pub(crate) streaming_usage: bool,
    pub(crate) developer_role: bool,
    pub(crate) schema_capabilities: ProviderSchemaCapabilities,
}

#[derive(Clone, Debug)]
pub struct OpenAiCompatibleProvider {
    pub api_key: String,
    pub base_url: String,
    pub options: ProviderOptions,
    pub compat: OpenAiCompat,
    pub(crate) transport: std::sync::Arc<dyn LlmHttpTransport>,
}

#[derive(Clone, Debug)]
pub struct OpenAiProvider {
    pub(crate) inner: OpenAiCompatibleProvider,
}

impl OpenAiCompatibleProvider {
    pub(crate) fn local_style_base_url(base_url: &str) -> bool {
        let normalized = base_url.trim().to_ascii_lowercase();
        normalized.contains("localhost")
            || normalized.contains("127.0.0.1")
            || normalized.contains("0.0.0.0")
            || normalized.contains("ollama")
    }

    pub(crate) fn resolved_compat(&self, endpoint: CompletionEndpoint) -> OpenAiResolvedCompat {
        let local = Self::local_style_base_url(&self.base_url);
        let direct_openai = self.base_url.trim_end_matches('/') == OPENAI_BASE_URL;

        let request_fields = !local;
        let max_tokens_field = match endpoint {
            CompletionEndpoint::Responses => OpenAiCompatMaxTokensField::MaxOutputTokens,
            CompletionEndpoint::ChatCompletions => OpenAiCompatMaxTokensField::MaxTokens,
        };
        let reasoning_format = match endpoint {
            CompletionEndpoint::Responses if direct_openai => OpenAiCompatReasoningFormat::OpenAi,
            _ => OpenAiCompatReasoningFormat::None,
        };
        let defaults = OpenAiResolvedCompat {
            request_fields,
            max_tokens_field,
            reasoning_format,
            cache_session_affinity: false,
            prompt_cache_key: false,
            prompt_cache_retention: false,
            strict_tools: false,
            store: !local,
            streaming_usage: !local,
            developer_role: direct_openai,
            schema_capabilities: ProviderSchemaCapabilities::openai(false),
        };
        let strict_tools = self.compat.strict_tools.unwrap_or(defaults.strict_tools);
        OpenAiResolvedCompat {
            request_fields: self
                .compat
                .request_fields
                .unwrap_or(defaults.request_fields),
            max_tokens_field: self
                .compat
                .max_tokens_field
                .unwrap_or(defaults.max_tokens_field),
            reasoning_format: self
                .compat
                .reasoning_format
                .unwrap_or(defaults.reasoning_format),
            cache_session_affinity: self
                .compat
                .cache_session_affinity
                .unwrap_or(defaults.cache_session_affinity),
            prompt_cache_key: self
                .compat
                .prompt_cache_key
                .unwrap_or(defaults.prompt_cache_key),
            prompt_cache_retention: self
                .compat
                .prompt_cache_retention
                .unwrap_or(defaults.prompt_cache_retention),
            strict_tools,
            store: self.compat.store.unwrap_or(defaults.store),
            streaming_usage: self
                .compat
                .streaming_usage
                .unwrap_or(defaults.streaming_usage),
            developer_role: self
                .compat
                .developer_role
                .unwrap_or(defaults.developer_role),
            schema_capabilities: self
                .compat
                .schema_capabilities
                .clone()
                .unwrap_or_else(|| ProviderSchemaCapabilities::openai(strict_tools)),
        }
    }
}
