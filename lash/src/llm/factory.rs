use crate::llm::adapters::claude_oauth::ClaudeOAuthAdapter;
use crate::llm::adapters::codex_oauth::CodexOAuthAdapter;
use crate::llm::adapters::google_cloudcode::GoogleCloudCodeAdapter;
use crate::llm::adapters::openrouter::OpenAiGenericAdapter;
use crate::llm::transport::LlmTransport;
use crate::provider::Provider;

pub fn adapter_for(provider: &Provider) -> Box<dyn LlmTransport> {
    let timeouts = provider.llm_timeouts();
    match provider {
        Provider::OpenAiGeneric { .. } => Box::new(OpenAiGenericAdapter::new(timeouts)),
        Provider::Claude { .. } => Box::new(ClaudeOAuthAdapter::new(timeouts)),
        Provider::Codex { .. } => Box::new(CodexOAuthAdapter::new(timeouts)),
        Provider::GoogleOAuth { .. } => Box::new(GoogleCloudCodeAdapter::new(timeouts)),
    }
}
