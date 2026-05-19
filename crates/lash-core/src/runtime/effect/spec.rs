use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::llm::types::{
    LlmAttachment, LlmEventSender, LlmMessage, LlmOutputSpec, LlmProviderTraceSender,
    LlmToolChoice, LlmToolSpec,
};
use crate::{
    AttachmentCreateMeta, AttachmentRef, AttachmentStore, DirectMessage, DirectOutputSpec,
    DirectRequest, LlmRequest as CoreLlmRequest, MediaType,
};

use super::controller::RuntimeEffectControllerError;

/// Serializable attachment data for runtime effect envelopes.
///
/// Effect envelopes carry attachment references only. Local executors resolve
/// bytes from the configured attachment store when a provider request is
/// actually executed.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LlmAttachmentSpec {
    pub reference: AttachmentRef,
}

impl LlmAttachmentSpec {
    fn into_attachment(self) -> LlmAttachment {
        LlmAttachment::reference(self.reference)
    }
}

/// Serializable LLM request data. Live stream and provider-trace callbacks are
/// attached by the local executor, and attachment bytes are resolved locally
/// from refs rather than persisted in the effect envelope.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LlmRequestSpec {
    pub model: String,
    pub messages: Vec<LlmMessage>,
    pub attachments: Vec<LlmAttachmentSpec>,
    pub tools: Vec<LlmToolSpec>,
    pub tool_choice: LlmToolChoice,
    pub model_variant: Option<String>,
    pub session_id: Option<String>,
    pub output_spec: Option<LlmOutputSpec>,
}

impl LlmRequestSpec {
    pub fn from_request(
        request: &CoreLlmRequest,
        attachment_store: &dyn AttachmentStore,
    ) -> Result<Self, RuntimeEffectControllerError> {
        Ok(Self {
            model: request.model.clone(),
            messages: request.messages.clone(),
            attachments: attachment_specs_from_attachments(&request.attachments, attachment_store)?,
            tools: request.tools.iter().cloned().collect(),
            tool_choice: request.tool_choice.clone(),
            model_variant: request.model_variant.clone(),
            session_id: request.session_id.clone(),
            output_spec: request.output_spec.clone(),
        })
    }

    pub fn into_request(
        self,
        stream_events: Option<LlmEventSender>,
        provider_trace: Option<LlmProviderTraceSender>,
    ) -> CoreLlmRequest {
        CoreLlmRequest {
            model: self.model,
            messages: self.messages,
            attachments: self
                .attachments
                .into_iter()
                .map(LlmAttachmentSpec::into_attachment)
                .collect(),
            tools: Arc::new(self.tools),
            tool_choice: self.tool_choice,
            model_variant: self.model_variant,
            session_id: self.session_id,
            output_spec: self.output_spec,
            stream_events,
            provider_trace,
        }
    }
}

/// Serializable direct request data. Caller-provided stream callbacks remain
/// local process state and are reattached by local direct executors.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DirectRequestSpec {
    pub model: String,
    pub model_variant: Option<String>,
    pub messages: Vec<DirectMessage>,
    pub attachments: Vec<LlmAttachmentSpec>,
    pub output: DirectOutputSpec,
    pub session_id: Option<String>,
    pub originating_tool_call_id: Option<String>,
    pub idempotency_key: Option<String>,
}

impl DirectRequestSpec {
    pub fn from_request(
        request: &DirectRequest,
        attachment_store: &dyn AttachmentStore,
    ) -> Result<Self, RuntimeEffectControllerError> {
        Ok(Self {
            model: request.model.clone(),
            model_variant: request.model_variant.clone(),
            messages: request.messages.clone(),
            attachments: attachment_specs_from_attachments(&request.attachments, attachment_store)?,
            output: request.output.clone(),
            session_id: request.session_id.clone(),
            originating_tool_call_id: request.originating_tool_call_id.clone(),
            idempotency_key: request.idempotency_key.clone(),
        })
    }

    pub fn into_request(self, stream_events: Option<LlmEventSender>) -> DirectRequest {
        DirectRequest {
            model: self.model,
            model_variant: self.model_variant,
            messages: self.messages,
            attachments: self
                .attachments
                .into_iter()
                .map(LlmAttachmentSpec::into_attachment)
                .collect(),
            output: self.output,
            stream_events,
            session_id: self.session_id,
            originating_tool_call_id: self.originating_tool_call_id,
            idempotency_key: self.idempotency_key,
        }
    }
}

fn attachment_specs_from_attachments(
    attachments: &[LlmAttachment],
    attachment_store: &dyn AttachmentStore,
) -> Result<Vec<LlmAttachmentSpec>, RuntimeEffectControllerError> {
    attachments
        .iter()
        .map(|attachment| attachment_spec_from_attachment(attachment, attachment_store))
        .collect()
}

fn attachment_spec_from_attachment(
    attachment: &LlmAttachment,
    attachment_store: &dyn AttachmentStore,
) -> Result<LlmAttachmentSpec, RuntimeEffectControllerError> {
    if let Some(reference) = attachment.reference.as_ref() {
        return Ok(LlmAttachmentSpec {
            reference: reference.clone(),
        });
    }
    if attachment.data.is_empty() {
        return Err(RuntimeEffectControllerError::new(
            "runtime_effect_attachment_missing_reference",
            "runtime effect attachment has neither a durable reference nor inline bytes",
        ));
    }
    let media_type = MediaType::from_mime(&attachment.mime).ok_or_else(|| {
        RuntimeEffectControllerError::new(
            "runtime_effect_attachment_media_type",
            format!(
                "attachment media type `{}` cannot be represented durably",
                attachment.mime
            ),
        )
    })?;
    let reference = attachment_store
        .put(
            attachment.data.clone(),
            AttachmentCreateMeta::new(media_type, None, None, None),
        )
        .map_err(|err| {
            RuntimeEffectControllerError::new(
                "runtime_effect_attachment_store",
                format!("failed to store attachment before runtime effect invocation: {err}"),
            )
        })?;
    Ok(LlmAttachmentSpec { reference })
}
