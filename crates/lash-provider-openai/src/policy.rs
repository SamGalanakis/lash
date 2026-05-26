use crate::support::*;

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

    pub(crate) fn reasoning_effort(&self, model: &str, variant: &str) -> Option<String> {
        self.supported_variants(model)
            .contains(&variant)
            .then(|| variant.to_string())
    }
}

#[derive(Clone, Debug)]
pub(crate) struct OpenAiDirectModelPolicy;

pub(crate) fn clamp_reasoning_effort(model: &str, effort: &str) -> String {
    let id = model_id(model).to_ascii_lowercase();
    if (id.starts_with("gpt-5.2") || id.starts_with("gpt-5.3") || id.starts_with("gpt-5.4"))
        && effort == "minimal"
    {
        return "low".to_string();
    }
    effort.to_string()
}

impl OpenAiDirectModelPolicy {
    pub(crate) fn reasoning_effort(&self, model: &str, variant: &str) -> Option<String> {
        self.supported_variants(model)
            .contains(&variant)
            .then(|| clamp_reasoning_effort(model, variant))
    }
}

impl ProviderModelPolicy for OpenAiModelPolicy {
    fn supported_variants(&self, model: &str) -> &'static [&'static str] {
        if !base_url_is_openrouter(&self.base_url) {
            return &[];
        }
        let lower = model.to_ascii_lowercase();
        if lower.contains("gpt") || lower.contains("claude") || lower.contains("gemini-3") {
            OPENROUTER_REASONING_VARIANTS
        } else {
            &[]
        }
    }
}

impl ProviderModelPolicy for OpenAiDirectModelPolicy {
    fn supported_variants(&self, model: &str) -> &'static [&'static str] {
        let id = model_id(model).to_ascii_lowercase();
        if id.starts_with("gpt-5") || id.starts_with("o") {
            OPENROUTER_REASONING_VARIANTS
        } else {
            &[]
        }
    }
}
