use crate::llm::factory::adapter_for;
use crate::llm::transport::LlmTransportError;
use crate::llm::types::{
    LlmAttachment, LlmEventSender, LlmJsonSchema, LlmMessage, LlmOutputSpec, LlmRequest,
    LlmResponse, LlmRole, LlmStreamEvent, LlmToolChoice,
};
use crate::provider::{Provider, save_provider};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DirectRole {
    System,
    User,
    Assistant,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DirectPart {
    Text(String),
    Image(usize),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DirectMessage {
    pub role: DirectRole,
    pub parts: Vec<DirectPart>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DirectJsonSchema {
    pub name: String,
    pub schema: serde_json::Value,
    pub strict: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub enum DirectOutputSpec {
    #[default]
    Text,
    JsonObject,
    JsonSchema(DirectJsonSchema),
}

#[derive(Clone, Debug)]
pub struct DirectRequest {
    pub model: String,
    pub model_variant: Option<String>,
    pub messages: Vec<DirectMessage>,
    pub attachments: Vec<LlmAttachment>,
    pub output: DirectOutputSpec,
    pub stream_events: Option<LlmEventSender>,
    pub session_id: Option<String>,
}

impl DirectRequest {
    pub fn text(model: impl Into<String>, prompt: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            model_variant: None,
            messages: vec![DirectMessage {
                role: DirectRole::User,
                parts: vec![DirectPart::Text(prompt.into())],
            }],
            attachments: Vec::new(),
            output: DirectOutputSpec::Text,
            stream_events: None,
            session_id: None,
        }
    }

    pub fn json(model: impl Into<String>, prompt: impl Into<String>) -> Self {
        Self {
            output: DirectOutputSpec::JsonObject,
            ..Self::text(model, prompt)
        }
    }

    pub fn json_schema(
        model: impl Into<String>,
        prompt: impl Into<String>,
        schema: DirectJsonSchema,
    ) -> Self {
        Self {
            output: DirectOutputSpec::JsonSchema(schema),
            ..Self::text(model, prompt)
        }
    }
}

#[derive(Debug, thiserror::Error, Clone)]
pub enum DirectLlmError {
    #[error("invalid request: {0}")]
    InvalidRequest(String),
    #[error("provider auth failed: {0}")]
    OAuth(String),
    #[error("provider initialization failed: {0}")]
    ProviderInit(String),
    #[error("transport error: {0}")]
    Transport(String),
}

impl From<LlmTransportError> for DirectLlmError {
    fn from(value: LlmTransportError) -> Self {
        Self::Transport(value.message)
    }
}

pub struct DirectLlmClient {
    provider: Provider,
    persist_provider_updates: bool,
}

impl DirectLlmClient {
    pub fn new(provider: Provider) -> Self {
        Self {
            provider,
            persist_provider_updates: false,
        }
    }

    pub fn with_persisted_provider(mut self, persist: bool) -> Self {
        self.persist_provider_updates = persist;
        self
    }

    pub fn provider(&self) -> &Provider {
        &self.provider
    }

    pub fn provider_mut(&mut self) -> &mut Provider {
        &mut self.provider
    }

    pub async fn complete(
        &mut self,
        request: DirectRequest,
    ) -> Result<LlmResponse, DirectLlmError> {
        let refreshed = self
            .provider
            .ensure_fresh()
            .await
            .map_err(|err| DirectLlmError::OAuth(err.to_string()))?;
        if refreshed && self.persist_provider_updates {
            let _ = save_provider(&self.provider);
        }

        let llm = adapter_for(&self.provider);
        let normalized_model = llm.normalize_model(&request.model);
        if let Some(variant) = request.model_variant.as_deref() {
            self.provider
                .validate_variant(&normalized_model, variant)
                .map_err(DirectLlmError::InvalidRequest)?;
        }

        let changed = llm
            .ensure_ready(&mut self.provider)
            .await
            .map_err(|err| DirectLlmError::ProviderInit(err.message))?;
        if changed && self.persist_provider_updates {
            let _ = save_provider(&self.provider);
        }

        let llm_request = build_llm_request(&self.provider, request, normalized_model);
        llm.complete(&mut self.provider, llm_request)
            .await
            .map_err(DirectLlmError::from)
    }
}

pub(crate) fn build_llm_request(
    provider: &Provider,
    request: DirectRequest,
    model: String,
) -> LlmRequest {
    let stream_events = transport_stream_events_for_direct(provider, request.stream_events);
    let DirectRequest {
        model: _,
        model_variant,
        messages,
        attachments,
        output,
        stream_events: _,
        session_id,
    } = request;

    let output_spec = match output {
        DirectOutputSpec::Text => None,
        DirectOutputSpec::JsonObject => Some(LlmOutputSpec::JsonObject),
        DirectOutputSpec::JsonSchema(schema) => Some(LlmOutputSpec::JsonSchema(LlmJsonSchema {
            name: schema.name,
            schema: schema.schema,
            strict: schema.strict,
        })),
    };

    let mut llm_messages = Vec::new();
    for message in messages {
        let role = match message.role {
            DirectRole::System => LlmRole::System,
            DirectRole::User => LlmRole::User,
            DirectRole::Assistant => LlmRole::Assistant,
        };
        for part in message.parts {
            match part {
                DirectPart::Text(text) => {
                    if !text.is_empty() {
                        llm_messages.push(LlmMessage {
                            role: role.clone(),
                            content: text,
                            kind: "text".to_string(),
                            image_idx: -1,
                            tool_call_id: None,
                            tool_name: None,
                        });
                    }
                }
                DirectPart::Image(idx) => {
                    llm_messages.push(LlmMessage {
                        role: role.clone(),
                        content: String::new(),
                        kind: "image".to_string(),
                        image_idx: idx as i64,
                        tool_call_id: None,
                        tool_name: None,
                    });
                }
            }
        }
    }

    LlmRequest {
        model,
        messages: llm_messages,
        attachments,
        tools: Vec::new().into(),
        tool_choice: LlmToolChoice::None,
        model_variant,
        session_id,
        output_spec,
        stream_events,
    }
}

fn transport_stream_events_for_direct(
    provider: &Provider,
    requested: Option<LlmEventSender>,
) -> Option<LlmEventSender> {
    if requested.is_some() {
        return requested;
    }
    let llm = adapter_for(provider);
    if llm.requires_streaming() {
        Some(LlmEventSender::new(|_event: LlmStreamEvent| {}))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_llm_request_preserves_single_user_message_as_message() {
        let provider = Provider::OpenAiGeneric {
            api_key: "tok".into(),
            base_url: "https://openrouter.ai/api/v1".into(),
            options: crate::provider::ProviderOptions::default(),
        };
        let request = DirectRequest::json("gpt-5", "hello");

        let llm_request = build_llm_request(&provider, request, "gpt-5".into());

        assert_eq!(llm_request.messages.len(), 1);
        assert_eq!(llm_request.messages[0].role, LlmRole::User);
        assert_eq!(llm_request.messages[0].content, "hello");
        assert!(matches!(
            llm_request.output_spec,
            Some(LlmOutputSpec::JsonObject)
        ));
    }

    #[test]
    fn build_llm_request_preserves_multi_message_history() {
        let provider = Provider::OpenAiGeneric {
            api_key: "tok".into(),
            base_url: "https://openrouter.ai/api/v1".into(),
            options: crate::provider::ProviderOptions::default(),
        };
        let request = DirectRequest {
            model: "gpt-5".into(),
            model_variant: None,
            messages: vec![
                DirectMessage {
                    role: DirectRole::System,
                    parts: vec![DirectPart::Text("extra".into())],
                },
                DirectMessage {
                    role: DirectRole::User,
                    parts: vec![DirectPart::Text("q".into())],
                },
                DirectMessage {
                    role: DirectRole::Assistant,
                    parts: vec![DirectPart::Text("a".into())],
                },
            ],
            attachments: Vec::new(),
            output: DirectOutputSpec::Text,
            stream_events: None,
            session_id: None,
        };

        let llm_request = build_llm_request(&provider, request, "gpt-5".into());

        assert_eq!(llm_request.messages.len(), 3);
        assert_eq!(llm_request.messages[0].role, LlmRole::System);
        assert_eq!(llm_request.messages[0].content, "extra");
        assert_eq!(llm_request.messages[1].content, "q");
        assert_eq!(llm_request.messages[2].content, "a");
    }
}
