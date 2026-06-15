//! Drift guard convention: conversions exhaustively destructure their source
//! (struct patterns without `..`, enum matches without catch-all `_` arms) so
//! a new field on either side fails compilation here instead of silently
//! dropping off the wire.

use std::collections::HashMap;
use std::future::Future;
use std::io::Write;
use std::num::NonZeroUsize;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use base64::Engine as _;
use lash_core::llm::types as core_llm;
use lash_core::{ToolCall, ToolContract, ToolDefinition, ToolManifest, ToolProvider, ToolResult};

use super::*;

impl From<RemoteProtocolTurnOptions> for lash_core::ProtocolTurnOptions {
    fn from(value: RemoteProtocolTurnOptions) -> Self {
        let RemoteProtocolTurnOptions { payload } = value;
        Self { payload }
    }
}

impl From<lash_core::ProtocolTurnOptions> for RemoteProtocolTurnOptions {
    fn from(value: lash_core::ProtocolTurnOptions) -> Self {
        let lash_core::ProtocolTurnOptions { payload } = value;
        Self { payload }
    }
}

impl TryFrom<RemoteTriggerOccurrenceRequest> for lash_core::TriggerOccurrenceRequest {
    type Error = RemoteProtocolError;

    fn try_from(value: RemoteTriggerOccurrenceRequest) -> Result<Self, Self::Error> {
        value.validate()?;
        let RemoteTriggerOccurrenceRequest {
            protocol_version: _,
            source_type,
            source_key,
            payload,
            idempotency_key,
            source,
        } = value;
        let mut request = lash_core::TriggerOccurrenceRequest::new(
            source_type,
            source_key,
            payload,
            idempotency_key,
        );
        request.source = source;
        Ok(request)
    }
}

impl From<lash_core::TriggerOccurrenceRequest> for RemoteTriggerOccurrenceRequest {
    fn from(value: lash_core::TriggerOccurrenceRequest) -> Self {
        let lash_core::TriggerOccurrenceRequest {
            source_type,
            source_key,
            payload,
            idempotency_key,
            source,
        } = value;
        Self {
            protocol_version: REMOTE_PROTOCOL_VERSION,
            source_type,
            source_key,
            payload,
            idempotency_key,
            source,
        }
    }
}

impl From<lash_core::TriggerOccurrenceRecord> for RemoteTriggerOccurrenceRecord {
    fn from(value: lash_core::TriggerOccurrenceRecord) -> Self {
        let lash_core::TriggerOccurrenceRecord {
            occurrence_id,
            source_type,
            source_key,
            payload,
            idempotency_key,
            source,
            occurred_at_ms,
        } = value;
        Self {
            occurrence_id,
            source_type,
            source_key,
            payload,
            idempotency_key,
            source,
            occurred_at_ms,
        }
    }
}

impl From<RemoteTriggerOccurrenceRecord> for lash_core::TriggerOccurrenceRecord {
    fn from(value: RemoteTriggerOccurrenceRecord) -> Self {
        let RemoteTriggerOccurrenceRecord {
            occurrence_id,
            source_type,
            source_key,
            payload,
            idempotency_key,
            source,
            occurred_at_ms,
        } = value;
        Self {
            occurrence_id,
            source_type,
            source_key,
            payload,
            idempotency_key,
            source,
            occurred_at_ms,
        }
    }
}

impl From<lash_core::TriggerEmitReport> for RemoteTriggerEmitReport {
    fn from(value: lash_core::TriggerEmitReport) -> Self {
        let lash_core::TriggerEmitReport {
            occurrence_id,
            started_process_ids,
        } = value;
        Self {
            protocol_version: REMOTE_PROTOCOL_VERSION,
            occurrence_id,
            started_process_ids,
        }
    }
}

impl TryFrom<RemoteTriggerEmitReport> for lash_core::TriggerEmitReport {
    type Error = RemoteProtocolError;

    fn try_from(value: RemoteTriggerEmitReport) -> Result<Self, Self::Error> {
        value.validate()?;
        let RemoteTriggerEmitReport {
            protocol_version: _,
            occurrence_id,
            started_process_ids,
        } = value;
        Ok(Self {
            occurrence_id,
            started_process_ids,
        })
    }
}

impl TryFrom<RemoteTriggerSubscriptionFilter> for lash_core::TriggerSubscriptionFilter {
    type Error = RemoteProtocolError;

    fn try_from(value: RemoteTriggerSubscriptionFilter) -> Result<Self, Self::Error> {
        value.validate()?;
        let RemoteTriggerSubscriptionFilter {
            protocol_version: _,
            session_id,
            handle,
            name,
            source_type,
            source_key,
            target,
            enabled,
        } = value;
        let target = target
            .map(serde_json::from_value)
            .transpose()
            .map_err(|err| RemoteProtocolError::InvalidEnvelope {
                type_name: "RemoteTriggerSubscriptionFilter",
                message: format!("invalid target identity: {err}"),
            })?;
        Ok(Self {
            session_id,
            handle,
            name,
            source_type,
            source_key,
            target,
            enabled,
        })
    }
}

impl From<lash_core::TriggerSubscriptionFilter> for RemoteTriggerSubscriptionFilter {
    fn from(value: lash_core::TriggerSubscriptionFilter) -> Self {
        let lash_core::TriggerSubscriptionFilter {
            session_id,
            handle,
            name,
            source_type,
            source_key,
            target,
            enabled,
        } = value;
        Self {
            protocol_version: REMOTE_PROTOCOL_VERSION,
            session_id,
            handle,
            name,
            source_type,
            source_key,
            target: target
                .map(|target| serde_json::to_value(target).expect("target identity serializes")),
            enabled,
        }
    }
}

impl From<lash_core::TriggerRegistration> for RemoteTriggerRegistration {
    fn from(value: lash_core::TriggerRegistration) -> Self {
        let lash_core::TriggerRegistration {
            handle,
            source_key,
            name,
            source_type,
            source,
            target,
            enabled,
        } = value;
        let lash_core::TriggerTargetSummary {
            process_name,
            inputs,
        } = target;
        Self {
            handle,
            source_key,
            name,
            source_type: source_type.to_string(),
            source,
            target: RemoteTriggerTargetSummary {
                process_name,
                inputs: serde_json::to_value(inputs).expect("trigger input template serializes"),
            },
            enabled,
        }
    }
}

impl TryFrom<RemoteTriggerRegistration> for lash_core::TriggerRegistration {
    type Error = RemoteProtocolError;

    fn try_from(value: RemoteTriggerRegistration) -> Result<Self, Self::Error> {
        let RemoteTriggerRegistration {
            handle,
            source_key,
            name,
            source_type,
            source,
            target,
            enabled,
        } = value;
        let RemoteTriggerTargetSummary {
            process_name,
            inputs,
        } = target;
        let inputs =
            serde_json::from_value(inputs).map_err(|err| RemoteProtocolError::InvalidEnvelope {
                type_name: "RemoteTriggerTargetSummary",
                message: format!("invalid input template: {err}"),
            })?;
        Ok(Self {
            handle,
            source_key,
            name,
            source_type: lash_core::TriggerEventType::new(source_type),
            source,
            target: lash_core::TriggerTargetSummary {
                process_name,
                inputs,
            },
            enabled,
        })
    }
}

impl From<lash_core::CausalRef> for RemoteCausalRef {
    fn from(value: lash_core::CausalRef) -> Self {
        match value {
            lash_core::CausalRef::Turn {
                session_id,
                turn_id,
            } => Self::Turn {
                session_id,
                turn_id,
            },
            lash_core::CausalRef::Effect {
                session_id,
                turn_id,
                effect_id,
            } => Self::Effect {
                session_id,
                turn_id,
                effect_id,
            },
            lash_core::CausalRef::ToolCall {
                session_id,
                call_id,
            } => Self::ToolCall {
                session_id,
                call_id,
            },
            lash_core::CausalRef::Process { process_id } => Self::Process { process_id },
            lash_core::CausalRef::ProcessEvent {
                process_id,
                sequence,
            } => Self::ProcessEvent {
                process_id,
                sequence,
            },
            lash_core::CausalRef::TriggerOccurrence { occurrence_id } => {
                Self::TriggerOccurrence { occurrence_id }
            }
            lash_core::CausalRef::SessionNode {
                session_id,
                node_id,
            } => Self::SessionNode {
                session_id,
                node_id,
            },
        }
    }
}

impl From<RemoteCausalRef> for lash_core::CausalRef {
    fn from(value: RemoteCausalRef) -> Self {
        match value {
            RemoteCausalRef::Turn {
                session_id,
                turn_id,
            } => Self::Turn {
                session_id,
                turn_id,
            },
            RemoteCausalRef::Effect {
                session_id,
                turn_id,
                effect_id,
            } => Self::Effect {
                session_id,
                turn_id,
                effect_id,
            },
            RemoteCausalRef::ToolCall {
                session_id,
                call_id,
            } => Self::ToolCall {
                session_id,
                call_id,
            },
            RemoteCausalRef::Process { process_id } => Self::Process { process_id },
            RemoteCausalRef::ProcessEvent {
                process_id,
                sequence,
            } => Self::ProcessEvent {
                process_id,
                sequence,
            },
            RemoteCausalRef::TriggerOccurrence { occurrence_id } => {
                Self::TriggerOccurrence { occurrence_id }
            }
            RemoteCausalRef::SessionNode {
                session_id,
                node_id,
            } => Self::SessionNode {
                session_id,
                node_id,
            },
        }
    }
}

impl TryFrom<RemoteTurnInput> for lash_core::TurnInput {
    type Error = RemoteProtocolError;

    fn try_from(value: RemoteTurnInput) -> Result<Self, Self::Error> {
        value.validate()?;
        let RemoteTurnInput {
            protocol_version: _,
            items,
            image_blobs_base64,
            protocol_turn_options,
            trace_turn_id,
            prompt_layer,
        } = value;
        let mut image_blobs = HashMap::new();
        for (id, encoded) in image_blobs_base64 {
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(encoded.as_bytes())
                .map_err(|err| RemoteProtocolError::InvalidImageBlob {
                    id: id.clone(),
                    message: err.to_string(),
                })?;
            image_blobs.insert(id, bytes);
        }
        let mut input = lash_core::TurnInput::items(items.into_iter().map(Into::into));
        input.image_blobs = image_blobs;
        input.protocol_turn_options = protocol_turn_options.map(Into::into);
        input.trace_turn_id = trace_turn_id;
        if let Some(prompt_layer) = prompt_layer {
            input.turn_context.set_prompt_layer(prompt_layer.into());
        }
        Ok(input)
    }
}

impl TryFrom<RemoteTurnRequest> for lash_core::TurnInput {
    type Error = RemoteProtocolError;

    fn try_from(value: RemoteTurnRequest) -> Result<Self, Self::Error> {
        value.validate()?;
        // Identity/routing fields are consumed by the transport layer, not the
        // core turn input; tool grants and model intent are applied separately.
        let RemoteTurnRequest {
            protocol_version: _,
            session_id: _,
            turn_id: _,
            idempotency_key: _,
            input,
            tool_grants: _,
            model_intent: _,
            metadata: _,
        } = value;
        input.try_into()
    }
}

impl TryFrom<lash_core::TurnInput> for RemoteTurnInput {
    type Error = RemoteProtocolError;

    fn try_from(value: lash_core::TurnInput) -> Result<Self, Self::Error> {
        // `turn_context` has private internals and is inspected through
        // accessors below; new TurnContext fields are not guarded here.
        let lash_core::TurnInput {
            items,
            image_blobs,
            protocol_turn_options,
            trace_turn_id,
            protocol_extension,
            turn_context,
        } = value;
        if protocol_extension.is_some() {
            return Err(RemoteProtocolError::NonRemoteSafeTurnInput(
                "live protocol turn extensions cannot cross a remote boundary".to_string(),
            ));
        }
        if turn_context.has_live_plugin_inputs() {
            return Err(RemoteProtocolError::NonRemoteSafeTurnInput(format!(
                "live plugin turn inputs cannot cross a remote boundary: {:?}",
                turn_context.live_plugin_input_ids()
            )));
        }
        if turn_context.provider().is_some() {
            return Err(RemoteProtocolError::NonRemoteSafeTurnInput(
                "per-turn provider overrides cannot cross a remote boundary".to_string(),
            ));
        }
        if turn_context.model_spec().is_some() {
            return Err(RemoteProtocolError::NonRemoteSafeTurnInput(
                "per-turn model overrides cannot cross a remote boundary".to_string(),
            ));
        }
        let prompt_layer = (!turn_context.prompt_layer().is_empty())
            .then(|| RemotePromptLayer::from(turn_context.prompt_layer().clone()));
        Ok(Self {
            protocol_version: REMOTE_PROTOCOL_VERSION,
            items: items.into_iter().map(Into::into).collect(),
            image_blobs_base64: image_blobs
                .into_iter()
                .map(|(id, bytes)| (id, base64::engine::general_purpose::STANDARD.encode(bytes)))
                .collect(),
            protocol_turn_options: protocol_turn_options.map(Into::into),
            trace_turn_id,
            prompt_layer,
        })
    }
}

impl From<RemoteInputItem> for lash_core::InputItem {
    fn from(value: RemoteInputItem) -> Self {
        match value {
            RemoteInputItem::Text { text } => Self::Text { text },
            RemoteInputItem::ImageRef { id } => Self::ImageRef { id },
        }
    }
}

impl From<lash_core::InputItem> for RemoteInputItem {
    fn from(value: lash_core::InputItem) -> Self {
        match value {
            lash_core::InputItem::Text { text } => Self::Text { text },
            lash_core::InputItem::ImageRef { id } => Self::ImageRef { id },
        }
    }
}

impl RemoteLlmRequest {
    pub fn from_core(request_id: impl Into<String>, value: core_llm::LlmRequest) -> Self {
        let core_llm::LlmRequest {
            model,
            messages,
            attachments,
            tools,
            tool_choice,
            model_variant,
            generation,
            session_id,
            output_spec,
            stream_events: _,
            provider_trace: _,
        } = value;
        Self {
            protocol_version: REMOTE_PROTOCOL_VERSION,
            request_id: request_id.into(),
            model_intent: RemoteModelIntent {
                model,
                variant: model_variant,
                provider: None,
                metadata: HashMap::new(),
            },
            messages: messages.into_iter().map(Into::into).collect(),
            attachments: attachments.into_iter().map(Into::into).collect(),
            tools: tools.iter().cloned().map(Into::into).collect(),
            tool_choice: tool_choice.into(),
            output_spec: output_spec.map(Into::into),
            generation: generation.into(),
            request_metadata: RemoteLlmRequestMetadata {
                session_id,
                idempotency_key: None,
                trace_id: None,
            },
            metadata: HashMap::new(),
        }
    }
}

impl TryFrom<RemoteLlmRequest> for core_llm::LlmRequest {
    type Error = RemoteProtocolError;

    fn try_from(value: RemoteLlmRequest) -> Result<Self, Self::Error> {
        value.validate()?;
        let RemoteLlmRequest {
            protocol_version: _,
            request_id: _,
            model_intent,
            messages,
            attachments,
            tools,
            tool_choice,
            output_spec,
            generation,
            request_metadata,
            metadata: _,
        } = value;
        let RemoteModelIntent {
            model,
            variant,
            provider: _,
            metadata: _,
        } = model_intent;
        let RemoteLlmRequestMetadata {
            session_id,
            idempotency_key: _,
            trace_id: _,
        } = request_metadata;
        Ok(Self {
            model,
            messages: messages.into_iter().map(Into::into).collect(),
            attachments: attachments
                .into_iter()
                .map(TryInto::try_into)
                .collect::<Result<Vec<_>, _>>()?,
            tools: Arc::new(tools.into_iter().map(Into::into).collect()),
            tool_choice: tool_choice.into(),
            model_variant: variant,
            generation: generation.try_into()?,
            session_id,
            output_spec: output_spec.map(Into::into),
            stream_events: None,
            provider_trace: None,
        })
    }
}

impl RemoteLlmResponse {
    pub fn from_core(request_id: impl Into<String>, value: core_llm::LlmResponse) -> Self {
        let core_llm::LlmResponse {
            full_text,
            parts,
            usage,
            terminal_reason,
            terminal_diagnostic,
            provider_usage,
            request_body,
            http_summary,
        } = value;
        let mut diagnostics = Vec::new();
        if let Some(message) = terminal_diagnostic {
            diagnostics.push(RemoteDiagnostic {
                kind: "terminal".to_string(),
                code: None,
                message,
                data: None,
            });
        }
        Self {
            protocol_version: REMOTE_PROTOCOL_VERSION,
            request_id: request_id.into(),
            full_text,
            output_parts: parts.into_iter().map(Into::into).collect(),
            usage: usage.into(),
            terminal_reason: terminal_reason.into(),
            diagnostics,
            provider_metadata: RemoteProviderMetadata {
                usage: provider_usage,
                request_body,
                http_summary,
                data: HashMap::new(),
            },
        }
    }
}

impl From<RemoteLlmResponse> for core_llm::LlmResponse {
    fn from(value: RemoteLlmResponse) -> Self {
        let RemoteLlmResponse {
            protocol_version: _,
            request_id: _,
            full_text,
            output_parts,
            usage,
            terminal_reason,
            diagnostics,
            provider_metadata,
        } = value;
        let RemoteProviderMetadata {
            usage: provider_usage,
            request_body,
            http_summary,
            data: _,
        } = provider_metadata;
        Self {
            full_text,
            parts: output_parts.into_iter().map(Into::into).collect(),
            usage: usage.into(),
            terminal_reason: terminal_reason.into(),
            terminal_diagnostic: diagnostics.first().map(|diag| diag.message.clone()),
            provider_usage,
            request_body,
            http_summary,
        }
    }
}

impl From<core_llm::GenerationOptions> for RemoteGenerationOptions {
    fn from(value: core_llm::GenerationOptions) -> Self {
        let core_llm::GenerationOptions { output_token_cap } = value;
        Self {
            output_token_cap: output_token_cap.map(|cap| cap.get() as u64),
            temperature: None,
            top_p: None,
            stop: Vec::new(),
            provider_options: HashMap::new(),
        }
    }
}

impl TryFrom<RemoteGenerationOptions> for core_llm::GenerationOptions {
    type Error = RemoteProtocolError;

    fn try_from(value: RemoteGenerationOptions) -> Result<Self, Self::Error> {
        value.validate("RemoteGenerationOptions")?;
        let RemoteGenerationOptions {
            output_token_cap,
            temperature: _,
            top_p: _,
            stop: _,
            provider_options: _,
        } = value;
        Ok(Self {
            output_token_cap: output_token_cap
                .and_then(|cap| usize::try_from(cap).ok())
                .and_then(NonZeroUsize::new),
        })
    }
}

impl From<core_llm::LlmMessage> for RemoteLlmMessage {
    fn from(value: core_llm::LlmMessage) -> Self {
        let core_llm::LlmMessage { role, blocks } = value;
        Self {
            role: role.into(),
            content: blocks.iter().cloned().map(Into::into).collect(),
        }
    }
}

impl From<RemoteLlmMessage> for core_llm::LlmMessage {
    fn from(value: RemoteLlmMessage) -> Self {
        let RemoteLlmMessage { role, content } = value;
        Self::new(role.into(), content.into_iter().map(Into::into).collect())
    }
}

impl From<core_llm::LlmRole> for RemoteLlmRole {
    fn from(value: core_llm::LlmRole) -> Self {
        match value {
            core_llm::LlmRole::User => Self::User,
            core_llm::LlmRole::Assistant => Self::Assistant,
            core_llm::LlmRole::System => Self::System,
        }
    }
}

impl From<RemoteLlmRole> for core_llm::LlmRole {
    fn from(value: RemoteLlmRole) -> Self {
        match value {
            RemoteLlmRole::User => Self::User,
            RemoteLlmRole::Assistant => Self::Assistant,
            RemoteLlmRole::System => Self::System,
        }
    }
}

impl From<core_llm::LlmContentBlock> for RemoteLlmContentBlock {
    fn from(value: core_llm::LlmContentBlock) -> Self {
        match value {
            core_llm::LlmContentBlock::Text {
                text,
                response_meta,
                cache_breakpoint,
            } => Self::Text {
                text: text.to_string(),
                response_meta: response_meta.map(Into::into),
                cache_breakpoint,
            },
            core_llm::LlmContentBlock::Image { attachment_idx } => Self::ImageAttachment {
                attachment_index: attachment_idx,
            },
            core_llm::LlmContentBlock::ToolCall {
                call_id,
                tool_name,
                input_json,
                replay,
            } => Self::ToolCall {
                call_id,
                tool_name,
                input_json,
                replay: replay.map(Into::into),
            },
            core_llm::LlmContentBlock::ToolResult {
                call_id,
                content,
                tool_name,
            } => Self::ToolResult {
                call_id,
                content,
                tool_name,
            },
            core_llm::LlmContentBlock::Reasoning { text, replay } => Self::Reasoning {
                text,
                replay: replay.map(Into::into),
            },
        }
    }
}

impl From<RemoteLlmContentBlock> for core_llm::LlmContentBlock {
    fn from(value: RemoteLlmContentBlock) -> Self {
        match value {
            RemoteLlmContentBlock::Text {
                text,
                response_meta,
                cache_breakpoint,
            } => Self::Text {
                text: Arc::<str>::from(text),
                response_meta: response_meta.map(Into::into),
                cache_breakpoint,
            },
            RemoteLlmContentBlock::ImageAttachment { attachment_index } => Self::Image {
                attachment_idx: attachment_index,
            },
            RemoteLlmContentBlock::ToolCall {
                call_id,
                tool_name,
                input_json,
                replay,
            } => Self::ToolCall {
                call_id,
                tool_name,
                input_json,
                replay: replay.map(Into::into),
            },
            RemoteLlmContentBlock::ToolResult {
                call_id,
                content,
                tool_name,
            } => Self::ToolResult {
                call_id,
                content,
                tool_name,
            },
            RemoteLlmContentBlock::Reasoning { text, replay } => Self::Reasoning {
                text,
                replay: replay.map(Into::into),
            },
        }
    }
}

impl From<core_llm::ResponseTextMeta> for RemoteResponseTextMeta {
    fn from(value: core_llm::ResponseTextMeta) -> Self {
        let core_llm::ResponseTextMeta { id, status, phase } = value;
        Self { id, status, phase }
    }
}

impl From<RemoteResponseTextMeta> for core_llm::ResponseTextMeta {
    fn from(value: RemoteResponseTextMeta) -> Self {
        let RemoteResponseTextMeta { id, status, phase } = value;
        Self { id, status, phase }
    }
}

impl From<core_llm::ProviderReplayMeta> for RemoteProviderReplayMeta {
    fn from(value: core_llm::ProviderReplayMeta) -> Self {
        let core_llm::ProviderReplayMeta { item_id, opaque } = value;
        Self { item_id, opaque }
    }
}

impl From<RemoteProviderReplayMeta> for core_llm::ProviderReplayMeta {
    fn from(value: RemoteProviderReplayMeta) -> Self {
        let RemoteProviderReplayMeta { item_id, opaque } = value;
        Self { item_id, opaque }
    }
}

impl From<core_llm::ProviderReasoningReplay> for RemoteProviderReasoningReplay {
    fn from(value: core_llm::ProviderReasoningReplay) -> Self {
        let core_llm::ProviderReasoningReplay {
            item_id,
            encrypted_content,
            signature,
            redacted,
            summary,
        } = value;
        Self {
            item_id,
            encrypted_content,
            signature,
            redacted,
            summary,
        }
    }
}

impl From<RemoteProviderReasoningReplay> for core_llm::ProviderReasoningReplay {
    fn from(value: RemoteProviderReasoningReplay) -> Self {
        let RemoteProviderReasoningReplay {
            item_id,
            encrypted_content,
            signature,
            redacted,
            summary,
        } = value;
        Self {
            item_id,
            encrypted_content,
            signature,
            redacted,
            summary,
        }
    }
}

impl From<core_llm::LlmAttachment> for RemoteLlmAttachment {
    fn from(value: core_llm::LlmAttachment) -> Self {
        let core_llm::LlmAttachment {
            mime,
            data,
            reference,
        } = value;
        Self {
            id: reference.as_ref().map(|reference| reference.id.to_string()),
            mime,
            data_base64: (!data.is_empty())
                .then(|| base64::engine::general_purpose::STANDARD.encode(data)),
            reference: reference.map(Into::into),
            metadata: HashMap::new(),
        }
    }
}

impl TryFrom<RemoteLlmAttachment> for core_llm::LlmAttachment {
    type Error = RemoteProtocolError;

    fn try_from(value: RemoteLlmAttachment) -> Result<Self, Self::Error> {
        let RemoteLlmAttachment {
            id,
            mime,
            data_base64,
            reference,
            metadata: _,
        } = value;
        let data = match data_base64 {
            Some(encoded) => base64::engine::general_purpose::STANDARD
                .decode(encoded.as_bytes())
                .map_err(|err| RemoteProtocolError::InvalidImageBlob {
                    id: id.unwrap_or_else(|| "<inline>".to_string()),
                    message: err.to_string(),
                })?,
            None => Vec::new(),
        };
        Ok(Self {
            mime,
            data,
            reference: reference.map(TryInto::try_into).transpose()?,
        })
    }
}

impl From<lash_core::AttachmentRef> for RemoteAttachmentRef {
    fn from(value: lash_core::AttachmentRef) -> Self {
        let mime = value.canonical_mime().to_string();
        let lash_core::AttachmentRef {
            id,
            media_type: _,
            byte_len,
            width,
            height,
            label,
        } = value;
        Self {
            id: id.to_string(),
            mime,
            byte_len,
            width,
            height,
            label,
            metadata: HashMap::new(),
        }
    }
}

impl TryFrom<RemoteAttachmentRef> for lash_core::AttachmentRef {
    type Error = RemoteProtocolError;

    fn try_from(value: RemoteAttachmentRef) -> Result<Self, Self::Error> {
        value.validate()?;
        let RemoteAttachmentRef {
            id,
            mime,
            byte_len,
            width,
            height,
            label,
            metadata: _,
        } = value;
        let media_type = lash_core::MediaType::from_mime(&mime).ok_or_else(|| {
            RemoteProtocolError::InvalidAttachmentRef {
                id: id.clone(),
                message: format!("unsupported attachment mime `{mime}`"),
            }
        })?;
        Ok(Self {
            id: lash_core::AttachmentId::new(id),
            media_type,
            byte_len,
            width,
            height,
            label,
        })
    }
}

impl From<core_llm::LlmToolSpec> for RemoteLlmToolSpec {
    fn from(value: core_llm::LlmToolSpec) -> Self {
        let core_llm::LlmToolSpec {
            name,
            description,
            input_schema,
            output_schema,
            input_schema_projections,
            output_schema_projections,
        } = value;
        Self {
            name,
            description,
            input_schema,
            output_schema,
            input_schema_projections: input_schema_projections
                .into_iter()
                .map(Into::into)
                .collect(),
            output_schema_projections: output_schema_projections
                .into_iter()
                .map(Into::into)
                .collect(),
        }
    }
}

impl From<RemoteLlmToolSpec> for core_llm::LlmToolSpec {
    fn from(value: RemoteLlmToolSpec) -> Self {
        let RemoteLlmToolSpec {
            name,
            description,
            input_schema,
            output_schema,
            input_schema_projections,
            output_schema_projections,
        } = value;
        Self {
            name,
            description,
            input_schema,
            output_schema,
            input_schema_projections: input_schema_projections
                .into_iter()
                .map(Into::into)
                .collect(),
            output_schema_projections: output_schema_projections
                .into_iter()
                .map(Into::into)
                .collect(),
        }
    }
}

impl From<core_llm::LlmToolChoice> for RemoteLlmToolChoice {
    fn from(value: core_llm::LlmToolChoice) -> Self {
        match value {
            core_llm::LlmToolChoice::Auto => Self::Auto,
            core_llm::LlmToolChoice::None => Self::None,
            core_llm::LlmToolChoice::Required => Self::Required,
        }
    }
}

impl From<RemoteLlmToolChoice> for core_llm::LlmToolChoice {
    fn from(value: RemoteLlmToolChoice) -> Self {
        match value {
            RemoteLlmToolChoice::Auto => Self::Auto,
            RemoteLlmToolChoice::None => Self::None,
            RemoteLlmToolChoice::Required => Self::Required,
        }
    }
}

impl From<core_llm::LlmOutputSpec> for RemoteLlmOutputSpec {
    fn from(value: core_llm::LlmOutputSpec) -> Self {
        match value {
            core_llm::LlmOutputSpec::JsonObject => Self::JsonObject,
            core_llm::LlmOutputSpec::JsonSchema(core_llm::LlmJsonSchema {
                name,
                schema,
                strict,
            }) => Self::JsonSchema {
                name,
                schema,
                strict,
            },
        }
    }
}

impl From<RemoteLlmOutputSpec> for core_llm::LlmOutputSpec {
    fn from(value: RemoteLlmOutputSpec) -> Self {
        match value {
            RemoteLlmOutputSpec::JsonObject => Self::JsonObject,
            RemoteLlmOutputSpec::JsonSchema {
                name,
                schema,
                strict,
            } => Self::JsonSchema(core_llm::LlmJsonSchema {
                name,
                schema,
                strict,
            }),
        }
    }
}

impl From<core_llm::LlmOutputPart> for RemoteLlmOutputPart {
    fn from(value: core_llm::LlmOutputPart) -> Self {
        match value {
            core_llm::LlmOutputPart::Text {
                text,
                response_meta,
            } => Self::Text {
                text,
                response_meta: response_meta.map(Into::into),
            },
            core_llm::LlmOutputPart::Reasoning { text, replay } => Self::Reasoning {
                text,
                replay: replay.map(Into::into),
            },
            core_llm::LlmOutputPart::ToolCall {
                call_id,
                tool_name,
                input_json,
                replay,
            } => Self::ToolCall {
                call_id,
                tool_name,
                input_json,
                replay: replay.map(Into::into),
            },
        }
    }
}

impl From<RemoteLlmOutputPart> for core_llm::LlmOutputPart {
    fn from(value: RemoteLlmOutputPart) -> Self {
        match value {
            RemoteLlmOutputPart::Text {
                text,
                response_meta,
            } => Self::Text {
                text,
                response_meta: response_meta.map(Into::into),
            },
            RemoteLlmOutputPart::Reasoning { text, replay } => Self::Reasoning {
                text,
                replay: replay.map(Into::into),
            },
            RemoteLlmOutputPart::ToolCall {
                call_id,
                tool_name,
                input_json,
                replay,
            } => Self::ToolCall {
                call_id,
                tool_name,
                input_json,
                replay: replay.map(Into::into),
            },
        }
    }
}

impl From<core_llm::LlmTerminalReason> for RemoteLlmTerminalReason {
    fn from(value: core_llm::LlmTerminalReason) -> Self {
        match value {
            core_llm::LlmTerminalReason::Stop => Self::Stop,
            core_llm::LlmTerminalReason::ToolUse => Self::ToolUse,
            core_llm::LlmTerminalReason::OutputLimit => Self::OutputLimit,
            core_llm::LlmTerminalReason::ContextOverflow => Self::ContextOverflow,
            core_llm::LlmTerminalReason::ContentFilter => Self::ContentFilter,
            core_llm::LlmTerminalReason::ProviderError => Self::ProviderError,
            core_llm::LlmTerminalReason::Cancelled => Self::Cancelled,
            core_llm::LlmTerminalReason::Unknown => Self::Unknown,
        }
    }
}

impl From<RemoteLlmTerminalReason> for core_llm::LlmTerminalReason {
    fn from(value: RemoteLlmTerminalReason) -> Self {
        match value {
            RemoteLlmTerminalReason::Stop => Self::Stop,
            RemoteLlmTerminalReason::ToolUse => Self::ToolUse,
            RemoteLlmTerminalReason::OutputLimit => Self::OutputLimit,
            RemoteLlmTerminalReason::ContextOverflow => Self::ContextOverflow,
            RemoteLlmTerminalReason::ContentFilter => Self::ContentFilter,
            RemoteLlmTerminalReason::ProviderError => Self::ProviderError,
            RemoteLlmTerminalReason::Cancelled => Self::Cancelled,
            RemoteLlmTerminalReason::Unknown => Self::Unknown,
        }
    }
}

impl From<core_llm::LlmUsage> for RemoteUsage {
    fn from(value: core_llm::LlmUsage) -> Self {
        let core_llm::LlmUsage {
            input_tokens,
            output_tokens,
            cached_input_tokens,
            reasoning_tokens,
        } = value;
        Self {
            input_tokens,
            output_tokens,
            cached_input_tokens,
            reasoning_tokens,
        }
    }
}

impl From<RemoteUsage> for core_llm::LlmUsage {
    fn from(value: RemoteUsage) -> Self {
        let RemoteUsage {
            input_tokens,
            output_tokens,
            cached_input_tokens,
            reasoning_tokens,
        } = value;
        Self {
            input_tokens,
            output_tokens,
            cached_input_tokens,
            reasoning_tokens,
        }
    }
}

impl From<lash_core::TokenUsage> for RemoteUsage {
    fn from(value: lash_core::TokenUsage) -> Self {
        let lash_core::TokenUsage {
            input_tokens,
            output_tokens,
            cached_input_tokens,
            reasoning_tokens,
        } = value;
        Self {
            input_tokens,
            output_tokens,
            cached_input_tokens,
            reasoning_tokens,
        }
    }
}

impl From<RemoteUsage> for lash_core::TokenUsage {
    fn from(value: RemoteUsage) -> Self {
        let RemoteUsage {
            input_tokens,
            output_tokens,
            cached_input_tokens,
            reasoning_tokens,
        } = value;
        Self {
            input_tokens,
            output_tokens,
            cached_input_tokens,
            reasoning_tokens,
        }
    }
}

impl From<lash_core::TokenLedgerEntry> for RemoteTokenLedgerEntry {
    fn from(value: lash_core::TokenLedgerEntry) -> Self {
        let lash_core::TokenLedgerEntry {
            source,
            model,
            usage,
        } = value;
        Self {
            source,
            model,
            usage: usage.into(),
        }
    }
}

impl From<RemoteTokenLedgerEntry> for lash_core::TokenLedgerEntry {
    fn from(value: RemoteTokenLedgerEntry) -> Self {
        let RemoteTokenLedgerEntry {
            source,
            model,
            usage,
        } = value;
        Self {
            source,
            model,
            usage: usage.into(),
        }
    }
}

impl RemoteTurnResult {
    pub fn from_core(
        session_id: impl Into<String>,
        turn_id: impl Into<String>,
        turn: lash_core::AssembledTurn,
        activities: impl IntoIterator<Item = RemoteTurnActivity>,
    ) -> Self {
        // `state` is the local session snapshot; it never crosses the wire.
        let lash_core::AssembledTurn {
            state: _,
            outcome,
            assistant_output,
            execution,
            token_usage,
            children_usage,
            tool_calls,
            errors,
        } = turn;
        let parent = RemoteUsage::from(token_usage);
        let children = children_usage
            .into_iter()
            .map(RemoteTokenLedgerEntry::from)
            .collect::<Vec<_>>();
        let mut total = parent.clone();
        for child in &children {
            total.add(&child.usage);
        }
        let outcome = RemoteTurnOutcome::from(outcome);
        let status = RemoteTurnStatus::from(&outcome);
        Self {
            protocol_version: REMOTE_PROTOCOL_VERSION,
            session_id: session_id.into(),
            turn_id: turn_id.into(),
            status,
            outcome,
            assistant_output: assistant_output.into(),
            usage: RemoteTurnUsageSummary {
                parent,
                children,
                total,
            },
            execution: execution.into(),
            tool_calls: tool_calls.into_iter().map(Into::into).collect(),
            issues: errors.into_iter().map(Into::into).collect(),
            activities: activities.into_iter().collect(),
            metadata: HashMap::new(),
        }
    }
}

impl From<&RemoteTurnOutcome> for RemoteTurnStatus {
    fn from(value: &RemoteTurnOutcome) -> Self {
        match value {
            RemoteTurnOutcome::Finished { .. } | RemoteTurnOutcome::AgentFrameSwitch { .. } => {
                Self::Completed
            }
            RemoteTurnOutcome::Stopped {
                stop: RemoteTurnStop::Cancelled,
            } => Self::Cancelled,
            RemoteTurnOutcome::Stopped { .. } => Self::Failed,
        }
    }
}

impl From<lash_core::TurnOutcome> for RemoteTurnOutcome {
    fn from(value: lash_core::TurnOutcome) -> Self {
        match value {
            lash_core::TurnOutcome::Finished(finish) => Self::Finished {
                finish: finish.into(),
            },
            lash_core::TurnOutcome::AgentFrameSwitch { frame_id, task } => {
                Self::AgentFrameSwitch { frame_id, task }
            }
            lash_core::TurnOutcome::Stopped(stop) => Self::Stopped { stop: stop.into() },
        }
    }
}

impl From<lash_core::TurnFinish> for RemoteTurnFinish {
    fn from(value: lash_core::TurnFinish) -> Self {
        match value {
            lash_core::TurnFinish::AssistantMessage { text } => Self::AssistantMessage { text },
            lash_core::TurnFinish::SubmittedValue { value } => Self::SubmittedValue { value },
            lash_core::TurnFinish::ToolValue { tool_name, value } => {
                Self::ToolValue { tool_name, value }
            }
        }
    }
}

impl From<lash_core::TurnStop> for RemoteTurnStop {
    fn from(value: lash_core::TurnStop) -> Self {
        match value {
            lash_core::TurnStop::Cancelled => Self::Cancelled,
            lash_core::TurnStop::Incomplete => Self::Incomplete,
            lash_core::TurnStop::InvalidInput => Self::InvalidInput,
            lash_core::TurnStop::MaxTurns => Self::MaxTurns,
            lash_core::TurnStop::ToolFailure => Self::ToolFailure,
            lash_core::TurnStop::ProviderError => Self::ProviderError,
            lash_core::TurnStop::PluginAbort => Self::PluginAbort,
            lash_core::TurnStop::RuntimeError => Self::RuntimeError,
            lash_core::TurnStop::SubmittedError { value } => Self::SubmittedError { value },
            lash_core::TurnStop::ToolError { tool_name, value } => {
                Self::ToolError { tool_name, value }
            }
        }
    }
}

impl From<lash_core::AssistantOutput> for RemoteAssistantOutput {
    fn from(value: lash_core::AssistantOutput) -> Self {
        let lash_core::AssistantOutput {
            safe_text,
            raw_text,
            state,
        } = value;
        Self {
            safe_text,
            raw_text,
            state: state.into(),
        }
    }
}

impl From<lash_core::OutputState> for RemoteAssistantOutputState {
    fn from(value: lash_core::OutputState) -> Self {
        match value {
            lash_core::OutputState::Usable => Self::Usable,
            lash_core::OutputState::EmptyOutput => Self::EmptyOutput,
            lash_core::OutputState::TracebackOnly => Self::TracebackOnly,
            lash_core::OutputState::RecoveredFromError => Self::RecoveredFromError,
        }
    }
}

impl From<lash_core::ExecutionSummary> for RemoteExecutionSummary {
    fn from(value: lash_core::ExecutionSummary) -> Self {
        let lash_core::ExecutionSummary {
            had_tool_calls,
            had_code_execution,
        } = value;
        Self {
            had_tool_calls,
            had_code_execution,
        }
    }
}

impl From<lash_core::ToolCallRecord> for RemoteToolCallSummary {
    fn from(value: lash_core::ToolCallRecord) -> Self {
        let lash_core::ToolCallRecord {
            call_id,
            tool,
            args,
            output,
            duration_ms,
        } = value;
        Self {
            call_id,
            tool_name: tool,
            args,
            outcome: output.into(),
            duration_ms,
        }
    }
}

impl From<lash_core::ToolCallOutput> for RemoteToolCallOutcome {
    fn from(value: lash_core::ToolCallOutput) -> Self {
        // `control` is a local turn-control signal and never crosses the wire.
        let lash_core::ToolCallOutput {
            outcome,
            control: _,
        } = value;
        match outcome {
            lash_core::ToolCallOutcome::Success(value) => Self::Success(value.to_json_value()),
            lash_core::ToolCallOutcome::Failure(value) => Self::Failure(value.to_json_value()),
            lash_core::ToolCallOutcome::Cancelled(value) => Self::Cancelled(value.to_json_value()),
        }
    }
}

impl From<lash_core::TurnIssue> for RemoteTurnIssue {
    fn from(value: lash_core::TurnIssue) -> Self {
        let lash_core::TurnIssue {
            kind,
            code,
            terminal_reason,
            message,
            raw,
        } = value;
        Self {
            kind,
            code,
            terminal_reason: terminal_reason.map(Into::into),
            message,
            raw,
        }
    }
}

impl RemoteTurnActivity {
    pub fn from_core(sequence: u64, activity: lash_core::TurnActivity) -> Self {
        let lash_core::TurnActivity {
            id: lash_core::TurnActivityId(id),
            correlation_id: lash_core::TurnActivityId(correlation_id),
            event,
        } = activity;
        Self {
            protocol_version: REMOTE_PROTOCOL_VERSION,
            sequence,
            id,
            correlation_id,
            event: RemoteTurnEvent::from(event),
        }
    }
}

impl From<lash_core::SessionQueueEventKind> for RemoteSessionQueueEventKind {
    fn from(value: lash_core::SessionQueueEventKind) -> Self {
        match value {
            lash_core::SessionQueueEventKind::Enqueued => Self::Enqueued,
            lash_core::SessionQueueEventKind::Cancelled => Self::Cancelled,
        }
    }
}

impl From<lash_core::SessionProcessEventKind> for RemoteSessionProcessEventKind {
    fn from(value: lash_core::SessionProcessEventKind) -> Self {
        match value {
            lash_core::SessionProcessEventKind::Started => Self::Started,
            lash_core::SessionProcessEventKind::Cancelled => Self::Cancelled,
        }
    }
}

impl RemoteSessionObservationEvent {
    pub fn from_core(sequence: u64, event: lash_core::SessionObservationEvent) -> Self {
        let lash_core::SessionObservationEvent {
            session_id,
            revision,
            cursor,
            payload,
        } = event;
        let payload = match payload {
            lash_core::SessionObservationEventPayload::TurnActivity(activity) => {
                RemoteSessionObservationEventPayload::TurnActivity {
                    activity: RemoteTurnActivity::from_core(sequence, activity),
                }
            }
            // The committed read view is a local handle; only the commit
            // signal itself crosses the wire.
            lash_core::SessionObservationEventPayload::Committed { read_view: _ } => {
                RemoteSessionObservationEventPayload::Committed
            }
            lash_core::SessionObservationEventPayload::AgentFrameSwitched { frame_id } => {
                RemoteSessionObservationEventPayload::AgentFrameSwitched { frame_id }
            }
            lash_core::SessionObservationEventPayload::QueueChanged { kind, batch_ids } => {
                RemoteSessionObservationEventPayload::QueueChanged {
                    kind: kind.into(),
                    batch_ids,
                }
            }
            lash_core::SessionObservationEventPayload::ProcessChanged { kind, process_ids } => {
                RemoteSessionObservationEventPayload::ProcessChanged {
                    kind: kind.into(),
                    process_ids,
                }
            }
        };
        Self {
            protocol_version: REMOTE_PROTOCOL_VERSION,
            session_id,
            revision: revision.as_u64(),
            cursor: cursor.to_string(),
            event: payload,
        }
    }
}

impl From<lash_core::LiveReplayGapReason> for RemoteLiveReplayGapReason {
    fn from(value: lash_core::LiveReplayGapReason) -> Self {
        match value {
            lash_core::LiveReplayGapReason::Trimmed => Self::Trimmed,
            lash_core::LiveReplayGapReason::Unavailable => Self::Unavailable,
        }
    }
}

impl From<lash_core::LiveReplayGap> for RemoteLiveReplayGap {
    fn from(value: lash_core::LiveReplayGap) -> Self {
        let lash_core::LiveReplayGap {
            session_id,
            requested_cursor,
            latest_cursor,
            latest_revision,
            reason,
        } = value;
        Self {
            protocol_version: REMOTE_PROTOCOL_VERSION,
            session_id,
            requested_cursor: requested_cursor.to_string(),
            latest_cursor: latest_cursor.to_string(),
            latest_revision: latest_revision.as_u64(),
            reason: reason.into(),
        }
    }
}

impl From<lash_core::TurnEvent> for RemoteTurnEvent {
    fn from(value: lash_core::TurnEvent) -> Self {
        match value {
            lash_core::TurnEvent::QueuedWorkStarted {
                boundary,
                batch_ids,
                causes,
            } => Self::RuntimeDiagnostic {
                kind: "queued_work_started".to_string(),
                data: serde_json::json!({
                    "boundary": boundary,
                    "batch_ids": batch_ids,
                    "causes": causes,
                }),
            },
            lash_core::TurnEvent::ModelRequestStarted { protocol_iteration } => {
                Self::ModelRequestStarted { protocol_iteration }
            }
            lash_core::TurnEvent::AssistantProseDelta { text } => {
                Self::AssistantProseDelta { text }
            }
            lash_core::TurnEvent::ReasoningDelta { text } => Self::ReasoningDelta { text },
            lash_core::TurnEvent::CodeBlockStarted {
                language,
                code,
                graph_key,
            } => Self::CodeBlockStarted {
                language,
                code,
                graph_key,
            },
            lash_core::TurnEvent::CodeBlockCompleted {
                language,
                output,
                error,
                success,
                duration_ms,
                tool_call_ids,
                graph_key,
            } => Self::CodeBlockCompleted {
                language,
                output,
                error,
                success,
                duration_ms,
                tool_call_ids,
                graph_key,
            },
            lash_core::TurnEvent::ToolCallStarted {
                call_id,
                name,
                args,
            } => Self::ToolCallStarted {
                call_id,
                name,
                args,
            },
            lash_core::TurnEvent::ToolCallCompleted {
                call_id,
                name,
                args,
                output,
                duration_ms,
            } => Self::ToolCallCompleted {
                call_id,
                name,
                args,
                output: serde_json::to_value(output).unwrap_or(serde_json::Value::Null),
                duration_ms,
            },
            lash_core::TurnEvent::SubmittedValue { value } => Self::SubmittedValue { value },
            lash_core::TurnEvent::ToolValue { tool_name, value } => {
                Self::ToolValue { tool_name, value }
            }
            lash_core::TurnEvent::Usage {
                protocol_iteration,
                usage,
                cumulative,
            } => Self::Usage {
                protocol_iteration,
                usage: usage.into(),
                cumulative: cumulative.into(),
            },
            lash_core::TurnEvent::ChildUsage {
                session_id,
                source,
                model,
                protocol_iteration,
                usage,
                cumulative,
            } => Self::ChildUsage {
                session_id,
                source,
                model,
                protocol_iteration,
                usage: usage.into(),
                cumulative: cumulative.into(),
            },
            lash_core::TurnEvent::RetryStatus {
                wait_seconds,
                attempt,
                max_attempts,
                reason,
            } => Self::RetryStatus {
                wait_seconds,
                attempt,
                max_attempts,
                reason,
            },
            lash_core::TurnEvent::PluginRuntime { plugin_id, event } => Self::RuntimeDiagnostic {
                kind: "plugin_runtime".to_string(),
                data: serde_json::json!({
                    "plugin_id": plugin_id,
                    "event": event,
                }),
            },
            lash_core::TurnEvent::QueuedInputAccepted { checkpoint, inputs } => {
                Self::RuntimeDiagnostic {
                    kind: "queued_input_accepted".to_string(),
                    data: serde_json::json!({
                        "checkpoint": checkpoint,
                        "inputs": inputs,
                    }),
                }
            }
            lash_core::TurnEvent::QueuedMessagesCommitted {
                messages,
                checkpoint,
            } => Self::RuntimeDiagnostic {
                kind: "queued_messages_committed".to_string(),
                data: serde_json::json!({
                    "messages": messages,
                    "checkpoint": checkpoint,
                }),
            },
            lash_core::TurnEvent::Error { message } => Self::Error { message },
        }
    }
}

pub fn replay_collected_activities(
    activities: impl IntoIterator<Item = lash_core::TurnActivity>,
    first_sequence: u64,
) -> Vec<RemoteTurnActivity> {
    activities
        .into_iter()
        .enumerate()
        .map(|(idx, activity)| {
            RemoteTurnActivity::from_core(first_sequence.saturating_add(idx as u64), activity)
        })
        .collect()
}

pub struct RemoteTurnActivitySink<W: Write + Send + 'static> {
    writer: Mutex<W>,
    next_sequence: AtomicU64,
    errors: Mutex<Vec<String>>,
}

impl<W: Write + Send + 'static> RemoteTurnActivitySink<W> {
    pub fn new(writer: W, first_sequence: u64) -> Self {
        Self {
            writer: Mutex::new(writer),
            next_sequence: AtomicU64::new(first_sequence),
            errors: Mutex::new(Vec::new()),
        }
    }

    pub fn take_errors(&self) -> Vec<String> {
        std::mem::take(&mut *self.errors.lock().expect("remote sink errors lock"))
    }

    pub fn into_inner(self) -> Result<W, W> {
        self.writer.into_inner().map_err(|err| err.into_inner())
    }
}

impl<W: Write + Send + 'static> lash_core::TurnActivitySink for RemoteTurnActivitySink<W> {
    fn emit<'life0, 'async_trait>(
        &'life0 self,
        activity: lash_core::TurnActivity,
    ) -> Pin<Box<dyn Future<Output = ()> + Send + 'async_trait>>
    where
        'life0: 'async_trait,
        Self: 'async_trait,
    {
        Box::pin(async move {
            let sequence = self.next_sequence.fetch_add(1, Ordering::SeqCst);
            let remote = RemoteTurnActivity::from_core(sequence, activity);
            let result = serde_json::to_writer(
                &mut *self.writer.lock().expect("remote sink writer lock"),
                &remote,
            )
            .and_then(|_| {
                self.writer
                    .lock()
                    .expect("remote sink writer lock")
                    .write_all(b"\n")
                    .map_err(serde_json::Error::io)
            });
            if let Err(err) = result {
                self.errors
                    .lock()
                    .expect("remote sink errors lock")
                    .push(err.to_string());
            }
        })
    }
}

impl From<lash_core::PromptLayer> for RemotePromptLayer {
    fn from(value: lash_core::PromptLayer) -> Self {
        let lash_core::PromptLayer { template, slots } = value;
        Self {
            template: template.map(Into::into),
            slots: slots
                .into_iter()
                .map(|(slot, layer)| (slot.into(), layer.into()))
                .collect(),
        }
    }
}

impl From<RemotePromptLayer> for lash_core::PromptLayer {
    fn from(value: RemotePromptLayer) -> Self {
        let RemotePromptLayer { template, slots } = value;
        Self {
            template: template.map(Into::into),
            slots: slots
                .into_iter()
                .map(|(slot, layer)| (slot.into(), layer.into()))
                .collect(),
        }
    }
}

impl From<lash_core::PromptTemplate> for RemotePromptTemplate {
    fn from(value: lash_core::PromptTemplate) -> Self {
        let lash_core::PromptTemplate { sections } = value;
        Self {
            sections: sections.into_iter().map(Into::into).collect(),
        }
    }
}

impl From<RemotePromptTemplate> for lash_core::PromptTemplate {
    fn from(value: RemotePromptTemplate) -> Self {
        let RemotePromptTemplate { sections } = value;
        Self {
            sections: sections.into_iter().map(Into::into).collect(),
        }
    }
}

impl From<lash_core::PromptTemplateSection> for RemotePromptTemplateSection {
    fn from(value: lash_core::PromptTemplateSection) -> Self {
        let lash_core::PromptTemplateSection { title, entries } = value;
        Self {
            title,
            entries: entries.into_iter().map(Into::into).collect(),
        }
    }
}

impl From<RemotePromptTemplateSection> for lash_core::PromptTemplateSection {
    fn from(value: RemotePromptTemplateSection) -> Self {
        let RemotePromptTemplateSection { title, entries } = value;
        Self {
            title,
            entries: entries.into_iter().map(Into::into).collect(),
        }
    }
}

impl From<lash_core::PromptTemplateEntry> for RemotePromptTemplateEntry {
    fn from(value: lash_core::PromptTemplateEntry) -> Self {
        match value {
            lash_core::PromptTemplateEntry::Text { content } => Self::Text { content },
            lash_core::PromptTemplateEntry::Builtin { builtin } => Self::Builtin {
                builtin: builtin.into(),
            },
            lash_core::PromptTemplateEntry::Slot { slot } => Self::Slot { slot: slot.into() },
        }
    }
}

impl From<RemotePromptTemplateEntry> for lash_core::PromptTemplateEntry {
    fn from(value: RemotePromptTemplateEntry) -> Self {
        match value {
            RemotePromptTemplateEntry::Text { content } => Self::Text { content },
            RemotePromptTemplateEntry::Builtin { builtin } => Self::Builtin {
                builtin: builtin.into(),
            },
            RemotePromptTemplateEntry::Slot { slot } => Self::Slot { slot: slot.into() },
        }
    }
}

impl From<lash_core::PromptBuiltin> for RemotePromptBuiltin {
    fn from(value: lash_core::PromptBuiltin) -> Self {
        match value {
            lash_core::PromptBuiltin::MainAgentIntro => Self::MainAgentIntro,
            lash_core::PromptBuiltin::ExecutionInstructions => Self::ExecutionInstructions,
            lash_core::PromptBuiltin::CoreGuidance => Self::CoreGuidance,
        }
    }
}

impl From<RemotePromptBuiltin> for lash_core::PromptBuiltin {
    fn from(value: RemotePromptBuiltin) -> Self {
        match value {
            RemotePromptBuiltin::MainAgentIntro => Self::MainAgentIntro,
            RemotePromptBuiltin::ExecutionInstructions => Self::ExecutionInstructions,
            RemotePromptBuiltin::CoreGuidance => Self::CoreGuidance,
        }
    }
}

impl From<lash_core::PromptSlot> for RemotePromptSlot {
    fn from(value: lash_core::PromptSlot) -> Self {
        match value {
            lash_core::PromptSlot::Intro => Self::Intro,
            lash_core::PromptSlot::Execution => Self::Execution,
            lash_core::PromptSlot::Guidance => Self::Guidance,
            lash_core::PromptSlot::ProjectInstructions => Self::ProjectInstructions,
            lash_core::PromptSlot::RuntimeContext => Self::RuntimeContext,
            lash_core::PromptSlot::Environment => Self::Environment,
        }
    }
}

impl From<RemotePromptSlot> for lash_core::PromptSlot {
    fn from(value: RemotePromptSlot) -> Self {
        match value {
            RemotePromptSlot::Intro => Self::Intro,
            RemotePromptSlot::Execution => Self::Execution,
            RemotePromptSlot::Guidance => Self::Guidance,
            RemotePromptSlot::ProjectInstructions => Self::ProjectInstructions,
            RemotePromptSlot::RuntimeContext => Self::RuntimeContext,
            RemotePromptSlot::Environment => Self::Environment,
        }
    }
}

impl From<lash_core::PromptSlotLayer> for RemotePromptSlotLayer {
    fn from(value: lash_core::PromptSlotLayer) -> Self {
        let lash_core::PromptSlotLayer {
            reset,
            contributions,
        } = value;
        Self {
            reset,
            contributions: contributions.into_iter().map(Into::into).collect(),
        }
    }
}

impl From<RemotePromptSlotLayer> for lash_core::PromptSlotLayer {
    fn from(value: RemotePromptSlotLayer) -> Self {
        let RemotePromptSlotLayer {
            reset,
            contributions,
        } = value;
        Self {
            reset,
            contributions: contributions.into_iter().map(Into::into).collect(),
        }
    }
}

impl From<lash_core::PromptContribution> for RemotePromptContribution {
    fn from(value: lash_core::PromptContribution) -> Self {
        let lash_core::PromptContribution {
            slot,
            title,
            priority,
            gate,
            content,
        } = value;
        Self {
            slot: slot.into(),
            title: title.map(|title| title.to_string()),
            priority,
            gate: gate.into(),
            content: content.to_string(),
        }
    }
}

impl From<RemotePromptContribution> for lash_core::PromptContribution {
    fn from(value: RemotePromptContribution) -> Self {
        let RemotePromptContribution {
            slot,
            title,
            priority,
            gate,
            content,
        } = value;
        Self {
            slot: slot.into(),
            title: title.map(Arc::from),
            priority,
            gate: gate.into(),
            content: Arc::from(content),
        }
    }
}

impl From<lash_core::PromptContributionGate> for RemotePromptContributionGate {
    fn from(value: lash_core::PromptContributionGate) -> Self {
        let lash_core::PromptContributionGate {
            tools,
            minimum_availability,
        } = value;
        Self {
            tools,
            minimum_availability: minimum_availability.into(),
        }
    }
}

impl From<RemotePromptContributionGate> for lash_core::PromptContributionGate {
    fn from(value: RemotePromptContributionGate) -> Self {
        let RemotePromptContributionGate {
            tools,
            minimum_availability,
        } = value;
        Self {
            tools,
            minimum_availability: minimum_availability.into(),
        }
    }
}

impl From<&RemoteLashlangToolBinding> for lash_core::LashlangToolBinding {
    fn from(value: &RemoteLashlangToolBinding) -> Self {
        let RemoteLashlangToolBinding {
            module_path,
            operation,
            authority_type,
            aliases,
        } = value;
        let mut binding =
            lash_core::LashlangToolBinding::new(module_path.clone(), operation.clone());
        if let Some(authority_type) = authority_type.as_ref() {
            binding = binding.with_authority_type(authority_type.clone());
        }
        binding.with_aliases(aliases.clone())
    }
}

impl From<lash_core::SchemaProjectionOverride> for RemoteSchemaProjectionOverride {
    fn from(value: lash_core::SchemaProjectionOverride) -> Self {
        let lash_core::SchemaProjectionOverride { profile, schema } = value;
        Self { profile, schema }
    }
}

impl From<RemoteSchemaProjectionOverride> for lash_core::SchemaProjectionOverride {
    fn from(value: RemoteSchemaProjectionOverride) -> Self {
        let RemoteSchemaProjectionOverride { profile, schema } = value;
        Self { profile, schema }
    }
}

impl From<RemoteToolAvailability> for lash_core::ToolAvailability {
    fn from(value: RemoteToolAvailability) -> Self {
        match value {
            RemoteToolAvailability::Off => Self::Off,
            RemoteToolAvailability::Searchable => Self::Searchable,
            RemoteToolAvailability::Callable => Self::Callable,
            RemoteToolAvailability::Showcased => Self::Showcased,
        }
    }
}

impl From<lash_core::ToolAvailability> for RemoteToolAvailability {
    fn from(value: lash_core::ToolAvailability) -> Self {
        match value {
            lash_core::ToolAvailability::Off => Self::Off,
            lash_core::ToolAvailability::Searchable => Self::Searchable,
            lash_core::ToolAvailability::Callable => Self::Callable,
            lash_core::ToolAvailability::Showcased => Self::Showcased,
        }
    }
}

impl From<RemoteToolActivation> for lash_core::ToolActivation {
    fn from(value: RemoteToolActivation) -> Self {
        match value {
            RemoteToolActivation::Always => Self::Always,
            RemoteToolActivation::Internal => Self::Internal,
        }
    }
}

impl From<RemoteToolScheduling> for lash_core::ToolScheduling {
    fn from(value: RemoteToolScheduling) -> Self {
        match value {
            RemoteToolScheduling::Parallel => Self::Parallel,
            RemoteToolScheduling::Serial => Self::Serial,
        }
    }
}

impl From<RemoteToolOutputContract> for lash_core::ToolOutputContract {
    fn from(value: RemoteToolOutputContract) -> Self {
        match value {
            RemoteToolOutputContract::Static => Self::Static,
            RemoteToolOutputContract::FromInputSchema {
                input_field,
                default_schema,
            } => Self::FromInputSchema {
                input_field,
                default_schema,
            },
        }
    }
}

impl From<RemoteToolArgumentProjectionPolicy> for lash_core::ToolArgumentProjectionPolicy {
    fn from(value: RemoteToolArgumentProjectionPolicy) -> Self {
        match value {
            RemoteToolArgumentProjectionPolicy::MaterializeProjectedValues => {
                Self::MaterializeProjectedValues
            }
            RemoteToolArgumentProjectionPolicy::PreserveProjectedRefsInField { field } => {
                Self::PreserveProjectedRefsInField { field }
            }
        }
    }
}

impl From<RemoteToolRetryPolicy> for lash_core::ToolRetryPolicy {
    fn from(value: RemoteToolRetryPolicy) -> Self {
        match value {
            RemoteToolRetryPolicy::Never => Self::Never,
            RemoteToolRetryPolicy::Safe {
                max_attempts,
                base_delay_ms,
                max_delay_ms,
            } => Self::Safe {
                max_attempts,
                base_delay_ms,
                max_delay_ms,
            },
            RemoteToolRetryPolicy::Idempotent {
                max_attempts,
                base_delay_ms,
                max_delay_ms,
            } => Self::Idempotent {
                max_attempts,
                base_delay_ms,
                max_delay_ms,
            },
        }
    }
}

impl TryFrom<&RemoteToolGrant> for ToolDefinition {
    type Error = RemoteProtocolError;

    fn try_from(value: &RemoteToolGrant) -> Result<Self, Self::Error> {
        value.validate()?;
        let RemoteToolGrant {
            protocol_version: _,
            id,
            name,
            description,
            input_schema,
            output_schema,
            input_schema_projections,
            output_schema_projections,
            output_contract,
            examples,
            availability,
            activation,
            argument_projection,
            scheduling,
            retry_policy,
            lashlang_binding,
        } = value;
        let mut definition = ToolDefinition::raw_with_id(
            id.clone()
                .unwrap_or_else(|| format!("remote-tool:{}", value.call_path().unwrap())),
            name.clone(),
            description.clone(),
            input_schema.clone(),
            output_schema.clone(),
        )
        .with_lashlang_binding(
            lashlang_binding
                .as_ref()
                .expect("validated lashlang binding")
                .into(),
        )
        .with_examples(examples.clone())
        .with_output_contract(output_contract.clone().into());
        if let Some(availability) = *availability {
            definition = definition
                .with_availability(lash_core::ToolAvailabilityConfig::same(availability.into()));
        }
        if let Some(activation) = *activation {
            definition = definition.with_activation(activation.into());
        }
        if let Some(argument_projection) = argument_projection.clone() {
            definition = definition.with_argument_projection(argument_projection.into());
        }
        if let Some(scheduling) = *scheduling {
            definition = definition.with_scheduling(scheduling.into());
        }
        if let Some(retry_policy) = *retry_policy {
            definition = definition.with_retry_policy(retry_policy.into());
        }
        for projection in input_schema_projections {
            definition = definition.with_input_schema_projection(
                projection.profile.clone(),
                projection.schema.clone(),
            );
        }
        for projection in output_schema_projections {
            definition = definition.with_output_schema_projection(
                projection.profile.clone(),
                projection.schema.clone(),
            );
        }
        Ok(definition)
    }
}

impl RemoteToolCallResponse {
    pub fn into_tool_result(self) -> ToolResult {
        match self {
            Self::Success {
                protocol_version: _,
                value,
            } => ToolResult::ok(value),
            Self::Failure {
                protocol_version: _,
                code,
                message,
                raw,
                retry_after_ms,
            } => {
                let mut failure = if let Some(after_ms) = retry_after_ms {
                    lash_core::ToolFailure::safe_retry(
                        lash_core::ToolFailureClass::Execution,
                        code,
                        message,
                        Some(after_ms),
                    )
                } else {
                    lash_core::ToolFailure::tool(
                        lash_core::ToolFailureClass::Execution,
                        code,
                        message,
                    )
                };
                failure.raw = raw.map(lash_core::ToolValue::from);
                ToolResult::failure(failure)
            }
            Self::Cancelled {
                protocol_version: _,
                message,
                raw,
            } => {
                if let Some(raw) = raw {
                    ToolResult::cancelled_with_raw(message, raw)
                } else {
                    ToolResult::cancelled(message)
                }
            }
            Self::Pending {
                protocol_version: _,
                deadline_ms,
                on_timeout,
                on_cancel,
            } => {
                let mut pending = lash_core::PendingCompletion::new();
                if let Some(deadline_ms) = deadline_ms {
                    pending.deadline = Some(std::time::Duration::from_millis(deadline_ms));
                }
                pending.on_timeout = match on_timeout {
                    RemoteTimeoutBehavior::ErrorAsResult => {
                        lash_core::TimeoutBehavior::ErrorAsResult
                    }
                    RemoteTimeoutBehavior::FailTurn => lash_core::TimeoutBehavior::FailTurn,
                };
                pending.on_cancel = match on_cancel {
                    RemoteCancelHint::Ignore => lash_core::CancelHint::Ignore,
                    RemoteCancelHint::CancelExternalWork => {
                        lash_core::CancelHint::CancelExternalWork
                    }
                };
                ToolResult::pending(pending)
            }
        }
    }
}

pub trait RemoteToolTransport: Send + Sync + 'static {
    fn send<'a>(
        &'a self,
        request: RemoteToolCallRequest,
    ) -> Pin<
        Box<dyn Future<Output = Result<RemoteToolCallResponse, RemoteProtocolError>> + Send + 'a>,
    >;
}

pub struct RemoteToolProvider<T: RemoteToolTransport> {
    manifests: Vec<ToolManifest>,
    contracts: HashMap<String, Arc<ToolContract>>,
    call_paths: HashMap<String, String>,
    transport: T,
}

impl<T: RemoteToolTransport> RemoteToolProvider<T> {
    pub fn new(grants: Vec<RemoteToolGrant>, transport: T) -> Result<Self, RemoteProtocolError> {
        RemoteToolGrant::validate_all(&grants)?;
        let mut manifests = Vec::with_capacity(grants.len());
        let mut contracts = HashMap::with_capacity(grants.len());
        let mut call_paths = HashMap::with_capacity(grants.len());
        for grant in grants {
            let definition = ToolDefinition::try_from(&grant)?;
            let manifest = definition.manifest();
            let executable = lash_core::LashlangToolBinding::required_for_remote(&manifest)
                .map_err(|message| RemoteProtocolError::InvalidToolGrant {
                    tool_name: manifest.name.clone(),
                    message,
                })?;
            contracts.insert(manifest.name.clone(), Arc::new(definition.contract()));
            call_paths.insert(manifest.name.clone(), executable.call_path());
            manifests.push(manifest);
        }
        Ok(Self {
            manifests,
            contracts,
            call_paths,
            transport,
        })
    }
}

impl<T: RemoteToolTransport> ToolProvider for RemoteToolProvider<T> {
    fn tool_manifests(&self) -> Vec<ToolManifest> {
        self.manifests.clone()
    }

    fn resolve_manifest(&self, name: &str) -> Option<ToolManifest> {
        self.manifests
            .iter()
            .find(|manifest| manifest.name == name)
            .cloned()
    }

    fn resolve_contract(&self, name: &str) -> Option<Arc<ToolContract>> {
        self.contracts.get(name).cloned()
    }

    fn execute<'life0, 'life1, 'async_trait>(
        &'life0 self,
        call: ToolCall<'life1>,
    ) -> Pin<Box<dyn Future<Output = ToolResult> + Send + 'async_trait>>
    where
        'life0: 'async_trait,
        'life1: 'async_trait,
        Self: 'async_trait,
    {
        Box::pin(async move {
            if call
                .context
                .cancellation_token()
                .is_some_and(|token| token.is_cancelled())
            {
                return ToolResult::cancelled("remote tool call cancelled before dispatch");
            }
            let Some(call_path) = self.call_paths.get(call.name) else {
                return ToolResult::err_fmt(format_args!("unknown remote tool `{}`", call.name));
            };
            let completion_key = match call.context.completion_key().await {
                Ok(key) => match serde_json::to_value(key) {
                    Ok(value) => value,
                    Err(err) => return ToolResult::err_fmt(err),
                },
                Err(err) => return ToolResult::err_fmt(err),
            };
            let mut headers = HashMap::new();
            if let Some(tool_call_id) = call.context.tool_call_id() {
                headers.insert("x-lash-tool-call-id".to_string(), tool_call_id.to_string());
            }
            let replay_key = call.context.replay_key().map(str::to_string).or_else(|| {
                call.context.tool_call_id().map(|call_id| {
                    format!(
                        "lash-tool:{}:{call_id}:{}",
                        call.context.session_id(),
                        call.name
                    )
                })
            });
            if let Some(replay_key) = replay_key.as_ref() {
                headers.insert("x-lash-replay-key".to_string(), replay_key.clone());
            }
            let request = RemoteToolCallRequest {
                protocol_version: REMOTE_PROTOCOL_VERSION,
                tool_name: call.name.to_string(),
                call_path: call_path.clone(),
                args: call.args.clone(),
                session_id: call.context.session_id().to_string(),
                completion_key,
                tool_call_id: call.context.tool_call_id().map(str::to_string),
                replay_key,
                attempt_number: call.context.attempt_number(),
                max_attempts: call.context.max_attempts(),
                headers,
            };
            match self.transport.send(request).await {
                Ok(response) => match response.validate() {
                    Ok(()) => response.into_tool_result(),
                    Err(err) => ToolResult::err_fmt(err),
                },
                Err(err) => ToolResult::err_fmt(err),
            }
        })
    }
}

#[cfg(test)]
#[path = "core_conversions_tests.rs"]
mod core_conversions_tests;
