use crate::llm::transport::LlmTransportError;
use crate::llm::types::{
    LlmAttachment, LlmContentBlock, LlmEventSender, LlmJsonSchema, LlmMessage, LlmOutputSpec,
    LlmRequest, LlmResponse, LlmRole, LlmStreamEvent, LlmToolChoice,
};
use crate::provider::ProviderHandle;
use lash_trace::{TraceContext, TraceError, TraceEvent, TraceSink};
use std::sync::Arc;

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DirectRole {
    System,
    User,
    Assistant,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum DirectPart {
    Text(String),
    Image(usize),
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct DirectMessage {
    pub role: DirectRole,
    pub parts: Vec<DirectPart>,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct DirectJsonSchema {
    pub name: String,
    pub schema: serde_json::Value,
    pub strict: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum DirectOutputSpec {
    #[default]
    Text,
    JsonObject,
    JsonSchema(DirectJsonSchema),
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DirectRequest {
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_variant: Option<String>,
    #[serde(default)]
    pub messages: Vec<DirectMessage>,
    #[serde(default)]
    pub attachments: Vec<LlmAttachment>,
    #[serde(default)]
    pub output: DirectOutputSpec,
    #[serde(default)]
    pub generation: crate::GenerationOptions,
    #[serde(default, skip)]
    pub stream_events: Option<LlmEventSender>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    /// Set this when issuing a direct completion from inside a tool's
    /// `execute`. Carries the calling tool's call id all the way to the
    /// trace event so the renderer can group fan-outs (e.g.
    /// tournament_rerank's batch reranks) under their parent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub originating_tool_call_id: Option<String>,
    /// Stable caller-provided idempotency discriminator for direct effects.
    ///
    /// Calls from a parent tool automatically include that tool call id in the
    /// derived key. Set this when the same tool or plugin hook may issue more
    /// than one otherwise-identical direct request during a turn.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub idempotency_key: Option<String>,
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
            generation: crate::GenerationOptions::default(),
            stream_events: None,
            session_id: None,
            originating_tool_call_id: None,
            idempotency_key: None,
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

    pub fn with_idempotency_key(mut self, key: impl Into<String>) -> Self {
        self.idempotency_key = Some(key.into());
        self
    }
}

#[derive(Debug, thiserror::Error, Clone)]
pub enum DirectLlmError {
    #[error("invalid request: {0}")]
    InvalidRequest(String),
    #[error("transport error: {0}")]
    Transport(#[from] LlmTransportError),
}

pub struct DirectLlmClient {
    provider: ProviderHandle,
    trace_sink: Option<Arc<dyn TraceSink>>,
    trace_context: TraceContext,
}

impl DirectLlmClient {
    pub fn new(provider: ProviderHandle) -> Self {
        Self {
            provider,
            trace_sink: None,
            trace_context: TraceContext::default(),
        }
    }

    pub fn with_trace_sink(mut self, sink: Option<Arc<dyn TraceSink>>) -> Self {
        self.trace_sink = sink;
        self
    }

    pub fn with_trace_context(mut self, context: TraceContext) -> Self {
        self.trace_context = context;
        self
    }

    pub fn provider(&self) -> &ProviderHandle {
        &self.provider
    }

    pub fn provider_mut(&mut self) -> &mut ProviderHandle {
        &mut self.provider
    }

    pub async fn complete(
        &mut self,
        request: DirectRequest,
    ) -> Result<LlmResponse, DirectLlmError> {
        if let Some(variant) = request.model_variant.as_deref() {
            self.provider
                .validate_variant(&request.model, variant)
                .map_err(DirectLlmError::InvalidRequest)?;
        }

        let model = request.model.clone();
        let llm_request = build_llm_request(&self.provider, request, model);
        let llm_call_id = if self.trace_sink.is_some() {
            let id = uuid::Uuid::new_v4().to_string();
            crate::trace::emit_trace(
                &self.trace_sink,
                &self.trace_context,
                TraceContext::default().for_llm_call(id.clone()),
                TraceEvent::LlmCallStarted {
                    request: crate::trace::trace_llm_request(&llm_request),
                },
            );
            Some(id)
        } else {
            None
        };
        match self.provider.complete(llm_request).await {
            Ok(response) => {
                if let Some(llm_call_id) = llm_call_id {
                    crate::trace::emit_trace(
                        &self.trace_sink,
                        &self.trace_context,
                        TraceContext::default().for_llm_call(llm_call_id),
                        TraceEvent::LlmCallCompleted {
                            response: crate::trace::trace_llm_response(
                                response.full_text.clone(),
                                0,
                                Some(response.terminal_reason),
                                crate::trace::trace_output_parts(&response.parts),
                            ),
                            usage: Some(crate::trace::trace_usage_from_llm(&response.usage)),
                            provider_usage: response.provider_usage.clone(),
                            stream_summary: None,
                        },
                    );
                }
                Ok(response)
            }
            Err(error) => {
                if let Some(llm_call_id) = llm_call_id {
                    crate::trace::emit_trace(
                        &self.trace_sink,
                        &self.trace_context,
                        TraceContext::default().for_llm_call(llm_call_id),
                        TraceEvent::LlmCallFailed {
                            error: TraceError {
                                message: error.message.clone(),
                                retryable: error.retryable,
                                terminal_reason: Some(error.terminal_reason.code().to_string()),
                                code: error.code.clone(),
                                raw: error.raw.clone(),
                            },
                            stream_summary: None,
                        },
                    );
                }
                Err(DirectLlmError::from(error))
            }
        }
    }
}

pub(crate) fn build_llm_request(
    provider: &ProviderHandle,
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
        generation,
        stream_events: _,
        session_id,
        originating_tool_call_id: _,
        idempotency_key: _,
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
        let mut blocks: Vec<LlmContentBlock> = Vec::new();
        for part in message.parts {
            match part {
                DirectPart::Text(text) => {
                    if !text.is_empty() {
                        blocks.push(LlmContentBlock::Text {
                            text: text.into(),
                            response_meta: None,
                            cache_breakpoint: false,
                        });
                    }
                }
                DirectPart::Image(idx) => {
                    blocks.push(LlmContentBlock::Image {
                        attachment_idx: idx,
                    });
                }
            }
        }
        if !blocks.is_empty() {
            llm_messages.push(LlmMessage::new(role, blocks));
        }
    }

    LlmRequest {
        model,
        messages: llm_messages,
        attachments,
        tools: Vec::new().into(),
        tool_choice: LlmToolChoice::None,
        model_variant,
        generation,
        session_id,
        output_spec,
        stream_events,
        provider_trace: None,
    }
}

fn transport_stream_events_for_direct(
    provider: &ProviderHandle,
    requested: Option<LlmEventSender>,
) -> Option<LlmEventSender> {
    if requested.is_some() {
        return requested;
    }
    if provider.requires_streaming() {
        Some(LlmEventSender::new(|_event: LlmStreamEvent| {}))
    } else {
        None
    }
}
