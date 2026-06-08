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
        Self {
            payload: value.payload,
        }
    }
}

impl From<lash_core::ProtocolTurnOptions> for RemoteProtocolTurnOptions {
    fn from(value: lash_core::ProtocolTurnOptions) -> Self {
        Self {
            payload: value.payload,
        }
    }
}

impl TryFrom<RemoteHostEventOccurrenceRequest> for lash_core::HostEventOccurrenceRequest {
    type Error = RemoteProtocolError;

    fn try_from(value: RemoteHostEventOccurrenceRequest) -> Result<Self, Self::Error> {
        value.validate()?;
        let mut request = lash_core::HostEventOccurrenceRequest::new(
            value.source_type,
            value.source_key,
            value.payload,
            value.idempotency_key,
        );
        request.source = value.source;
        Ok(request)
    }
}

impl From<lash_core::HostEventOccurrenceRequest> for RemoteHostEventOccurrenceRequest {
    fn from(value: lash_core::HostEventOccurrenceRequest) -> Self {
        Self {
            protocol_version: REMOTE_PROTOCOL_VERSION,
            source_type: value.source_type,
            source_key: value.source_key,
            payload: value.payload,
            idempotency_key: value.idempotency_key,
            source: value.source,
        }
    }
}

impl From<lash_core::HostEventOccurrenceRecord> for RemoteHostEventOccurrenceRecord {
    fn from(value: lash_core::HostEventOccurrenceRecord) -> Self {
        Self {
            occurrence_id: value.occurrence_id,
            source_type: value.source_type,
            source_key: value.source_key,
            payload: value.payload,
            idempotency_key: value.idempotency_key,
            source: value.source,
            occurred_at_ms: value.occurred_at_ms,
        }
    }
}

impl From<RemoteHostEventOccurrenceRecord> for lash_core::HostEventOccurrenceRecord {
    fn from(value: RemoteHostEventOccurrenceRecord) -> Self {
        Self {
            occurrence_id: value.occurrence_id,
            source_type: value.source_type,
            source_key: value.source_key,
            payload: value.payload,
            idempotency_key: value.idempotency_key,
            source: value.source,
            occurred_at_ms: value.occurred_at_ms,
        }
    }
}

impl From<lash_core::HostEventEmitReport> for RemoteHostEventEmitReport {
    fn from(value: lash_core::HostEventEmitReport) -> Self {
        Self {
            protocol_version: REMOTE_PROTOCOL_VERSION,
            occurrence_id: value.occurrence_id,
            started_process_ids: value.started_process_ids,
        }
    }
}

impl TryFrom<RemoteHostEventEmitReport> for lash_core::HostEventEmitReport {
    type Error = RemoteProtocolError;

    fn try_from(value: RemoteHostEventEmitReport) -> Result<Self, Self::Error> {
        value.validate()?;
        Ok(Self {
            occurrence_id: value.occurrence_id,
            started_process_ids: value.started_process_ids,
        })
    }
}

impl TryFrom<RemoteTriggerSubscriptionFilter> for lash_core::TriggerSubscriptionFilter {
    type Error = RemoteProtocolError;

    fn try_from(value: RemoteTriggerSubscriptionFilter) -> Result<Self, Self::Error> {
        value.validate()?;
        let target = value
            .target
            .map(serde_json::from_value)
            .transpose()
            .map_err(|err| RemoteProtocolError::InvalidEnvelope {
                type_name: "RemoteTriggerSubscriptionFilter",
                message: format!("invalid target identity: {err}"),
            })?;
        Ok(Self {
            session_id: value.session_id,
            handle: value.handle,
            name: value.name,
            source_type: value.source_type,
            source_key: value.source_key,
            target,
            enabled: value.enabled,
        })
    }
}

impl From<lash_core::TriggerSubscriptionFilter> for RemoteTriggerSubscriptionFilter {
    fn from(value: lash_core::TriggerSubscriptionFilter) -> Self {
        Self {
            protocol_version: REMOTE_PROTOCOL_VERSION,
            session_id: value.session_id,
            handle: value.handle,
            name: value.name,
            source_type: value.source_type,
            source_key: value.source_key,
            target: value
                .target
                .map(|target| serde_json::to_value(target).expect("target identity serializes")),
            enabled: value.enabled,
        }
    }
}

impl From<lash_core::TriggerRegistration> for RemoteTriggerRegistration {
    fn from(value: lash_core::TriggerRegistration) -> Self {
        Self {
            handle: value.handle,
            source_key: value.source_key,
            name: value.name,
            source_type: value.source_type.to_string(),
            source: value.source,
            target: RemoteTriggerTargetSummary {
                process_name: value.target.process_name,
                inputs: serde_json::to_value(value.target.inputs)
                    .expect("trigger input template serializes"),
            },
            enabled: value.enabled,
        }
    }
}

impl TryFrom<RemoteTriggerRegistration> for lash_core::TriggerRegistration {
    type Error = RemoteProtocolError;

    fn try_from(value: RemoteTriggerRegistration) -> Result<Self, Self::Error> {
        let inputs = serde_json::from_value(value.target.inputs).map_err(|err| {
            RemoteProtocolError::InvalidEnvelope {
                type_name: "RemoteTriggerTargetSummary",
                message: format!("invalid input template: {err}"),
            }
        })?;
        Ok(Self {
            handle: value.handle,
            source_key: value.source_key,
            name: value.name,
            source_type: lash_core::TriggerSourceType::new(value.source_type),
            source: value.source,
            target: lash_core::TriggerTargetSummary {
                process_name: value.target.process_name,
                inputs,
            },
            enabled: value.enabled,
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
            lash_core::CausalRef::HostEvent { occurrence_id } => Self::HostEvent { occurrence_id },
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
            RemoteCausalRef::HostEvent { occurrence_id } => Self::HostEvent { occurrence_id },
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
        let mut image_blobs = HashMap::new();
        for (id, encoded) in value.image_blobs_base64 {
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(encoded.as_bytes())
                .map_err(|err| RemoteProtocolError::InvalidImageBlob {
                    id: id.clone(),
                    message: err.to_string(),
                })?;
            image_blobs.insert(id, bytes);
        }
        let mut input = lash_core::TurnInput::items(value.items.into_iter().map(Into::into));
        input.image_blobs = image_blobs;
        input.protocol_turn_options = value.protocol_turn_options.map(Into::into);
        input.trace_turn_id = value.trace_turn_id;
        if let Some(prompt_layer) = value.prompt_layer {
            input.turn_context.set_prompt_layer(prompt_layer.into());
        }
        Ok(input)
    }
}

impl TryFrom<RemoteTurnRequest> for lash_core::TurnInput {
    type Error = RemoteProtocolError;

    fn try_from(value: RemoteTurnRequest) -> Result<Self, Self::Error> {
        value.validate()?;
        value.input.try_into()
    }
}

impl TryFrom<lash_core::TurnInput> for RemoteTurnInput {
    type Error = RemoteProtocolError;

    fn try_from(value: lash_core::TurnInput) -> Result<Self, Self::Error> {
        if value.protocol_extension.is_some() {
            return Err(RemoteProtocolError::NonRemoteSafeTurnInput(
                "live protocol turn extensions cannot cross a remote boundary".to_string(),
            ));
        }
        if value.turn_context.has_live_plugin_inputs() {
            return Err(RemoteProtocolError::NonRemoteSafeTurnInput(format!(
                "live plugin turn inputs cannot cross a remote boundary: {:?}",
                value.turn_context.live_plugin_input_ids()
            )));
        }
        if value.turn_context.provider().is_some() {
            return Err(RemoteProtocolError::NonRemoteSafeTurnInput(
                "per-turn provider overrides cannot cross a remote boundary".to_string(),
            ));
        }
        if value.turn_context.model_spec().is_some() {
            return Err(RemoteProtocolError::NonRemoteSafeTurnInput(
                "per-turn model overrides cannot cross a remote boundary".to_string(),
            ));
        }
        let prompt_layer = (!value.turn_context.prompt_layer().is_empty())
            .then(|| RemotePromptLayer::from(value.turn_context.prompt_layer().clone()));
        Ok(Self {
            protocol_version: REMOTE_PROTOCOL_VERSION,
            items: value.items.into_iter().map(Into::into).collect(),
            image_blobs_base64: value
                .image_blobs
                .into_iter()
                .map(|(id, bytes)| (id, base64::engine::general_purpose::STANDARD.encode(bytes)))
                .collect(),
            protocol_turn_options: value.protocol_turn_options.map(Into::into),
            trace_turn_id: value.trace_turn_id,
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
        Self {
            protocol_version: REMOTE_PROTOCOL_VERSION,
            request_id: request_id.into(),
            model_intent: RemoteModelIntent {
                model: value.model,
                variant: value.model_variant,
                provider: None,
                metadata: HashMap::new(),
            },
            messages: value.messages.into_iter().map(Into::into).collect(),
            attachments: value.attachments.into_iter().map(Into::into).collect(),
            tools: value.tools.iter().cloned().map(Into::into).collect(),
            tool_choice: value.tool_choice.into(),
            output_spec: value.output_spec.map(Into::into),
            generation: value.generation.into(),
            request_metadata: RemoteLlmRequestMetadata {
                session_id: value.session_id,
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
        Ok(Self {
            model: value.model_intent.model,
            messages: value.messages.into_iter().map(Into::into).collect(),
            attachments: value
                .attachments
                .into_iter()
                .map(TryInto::try_into)
                .collect::<Result<Vec<_>, _>>()?,
            tools: Arc::new(value.tools.into_iter().map(Into::into).collect()),
            tool_choice: value.tool_choice.into(),
            model_variant: value.model_intent.variant,
            generation: value.generation.try_into()?,
            session_id: value.request_metadata.session_id,
            output_spec: value.output_spec.map(Into::into),
            stream_events: None,
            provider_trace: None,
        })
    }
}

impl RemoteLlmResponse {
    pub fn from_core(request_id: impl Into<String>, value: core_llm::LlmResponse) -> Self {
        let mut diagnostics = Vec::new();
        if let Some(message) = value.terminal_diagnostic {
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
            full_text: value.full_text,
            output_parts: value.parts.into_iter().map(Into::into).collect(),
            usage: value.usage.into(),
            terminal_reason: value.terminal_reason.into(),
            diagnostics,
            provider_metadata: RemoteProviderMetadata {
                usage: value.provider_usage,
                request_body: value.request_body,
                http_summary: value.http_summary,
                data: HashMap::new(),
            },
        }
    }
}

impl From<RemoteLlmResponse> for core_llm::LlmResponse {
    fn from(value: RemoteLlmResponse) -> Self {
        Self {
            full_text: value.full_text,
            parts: value.output_parts.into_iter().map(Into::into).collect(),
            usage: value.usage.into(),
            terminal_reason: value.terminal_reason.into(),
            terminal_diagnostic: value.diagnostics.first().map(|diag| diag.message.clone()),
            provider_usage: value.provider_metadata.usage,
            request_body: value.provider_metadata.request_body,
            http_summary: value.provider_metadata.http_summary,
        }
    }
}

impl From<core_llm::GenerationOptions> for RemoteGenerationOptions {
    fn from(value: core_llm::GenerationOptions) -> Self {
        Self {
            output_token_cap: value.output_token_cap_u64(),
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
        Ok(Self {
            output_token_cap: value
                .output_token_cap
                .and_then(|cap| usize::try_from(cap).ok())
                .and_then(NonZeroUsize::new),
        })
    }
}

impl From<core_llm::LlmMessage> for RemoteLlmMessage {
    fn from(value: core_llm::LlmMessage) -> Self {
        Self {
            role: value.role.into(),
            content: value.blocks.iter().cloned().map(Into::into).collect(),
        }
    }
}

impl From<RemoteLlmMessage> for core_llm::LlmMessage {
    fn from(value: RemoteLlmMessage) -> Self {
        Self::new(
            value.role.into(),
            value.content.into_iter().map(Into::into).collect(),
        )
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
        Self {
            id: value.id,
            status: value.status,
            phase: value.phase,
        }
    }
}

impl From<RemoteResponseTextMeta> for core_llm::ResponseTextMeta {
    fn from(value: RemoteResponseTextMeta) -> Self {
        Self {
            id: value.id,
            status: value.status,
            phase: value.phase,
        }
    }
}

impl From<core_llm::ProviderReplayMeta> for RemoteProviderReplayMeta {
    fn from(value: core_llm::ProviderReplayMeta) -> Self {
        Self {
            item_id: value.item_id,
            opaque: value.opaque,
        }
    }
}

impl From<RemoteProviderReplayMeta> for core_llm::ProviderReplayMeta {
    fn from(value: RemoteProviderReplayMeta) -> Self {
        Self {
            item_id: value.item_id,
            opaque: value.opaque,
        }
    }
}

impl From<core_llm::ProviderReasoningReplay> for RemoteProviderReasoningReplay {
    fn from(value: core_llm::ProviderReasoningReplay) -> Self {
        Self {
            item_id: value.item_id,
            encrypted_content: value.encrypted_content,
            signature: value.signature,
            redacted: value.redacted,
            summary: value.summary,
        }
    }
}

impl From<RemoteProviderReasoningReplay> for core_llm::ProviderReasoningReplay {
    fn from(value: RemoteProviderReasoningReplay) -> Self {
        Self {
            item_id: value.item_id,
            encrypted_content: value.encrypted_content,
            signature: value.signature,
            redacted: value.redacted,
            summary: value.summary,
        }
    }
}

impl From<core_llm::LlmAttachment> for RemoteLlmAttachment {
    fn from(value: core_llm::LlmAttachment) -> Self {
        Self {
            id: value
                .reference
                .as_ref()
                .map(|reference| reference.id.to_string()),
            mime: value.mime,
            data_base64: (!value.data.is_empty())
                .then(|| base64::engine::general_purpose::STANDARD.encode(value.data)),
            reference: value.reference.map(Into::into),
            metadata: HashMap::new(),
        }
    }
}

impl TryFrom<RemoteLlmAttachment> for core_llm::LlmAttachment {
    type Error = RemoteProtocolError;

    fn try_from(value: RemoteLlmAttachment) -> Result<Self, Self::Error> {
        let data = match value.data_base64 {
            Some(encoded) => base64::engine::general_purpose::STANDARD
                .decode(encoded.as_bytes())
                .map_err(|err| RemoteProtocolError::InvalidImageBlob {
                    id: value.id.unwrap_or_else(|| "<inline>".to_string()),
                    message: err.to_string(),
                })?,
            None => Vec::new(),
        };
        Ok(Self {
            mime: value.mime,
            data,
            reference: value.reference.map(TryInto::try_into).transpose()?,
        })
    }
}

impl From<lash_core::AttachmentRef> for RemoteAttachmentRef {
    fn from(value: lash_core::AttachmentRef) -> Self {
        Self {
            id: value.id.to_string(),
            mime: value.canonical_mime().to_string(),
            byte_len: value.byte_len,
            width: value.width,
            height: value.height,
            label: value.label,
            metadata: HashMap::new(),
        }
    }
}

impl TryFrom<RemoteAttachmentRef> for lash_core::AttachmentRef {
    type Error = RemoteProtocolError;

    fn try_from(value: RemoteAttachmentRef) -> Result<Self, Self::Error> {
        value.validate()?;
        let media_type = lash_core::MediaType::from_mime(&value.mime).ok_or_else(|| {
            RemoteProtocolError::InvalidAttachmentRef {
                id: value.id.clone(),
                message: format!("unsupported attachment mime `{}`", value.mime),
            }
        })?;
        Ok(Self {
            id: lash_core::AttachmentId::new(value.id),
            media_type,
            byte_len: value.byte_len,
            width: value.width,
            height: value.height,
            label: value.label,
        })
    }
}

impl From<core_llm::LlmToolSpec> for RemoteLlmToolSpec {
    fn from(value: core_llm::LlmToolSpec) -> Self {
        Self {
            name: value.name,
            description: value.description,
            input_schema: value.input_schema,
            output_schema: value.output_schema,
            input_schema_projections: value
                .input_schema_projections
                .into_iter()
                .map(Into::into)
                .collect(),
            output_schema_projections: value
                .output_schema_projections
                .into_iter()
                .map(Into::into)
                .collect(),
        }
    }
}

impl From<RemoteLlmToolSpec> for core_llm::LlmToolSpec {
    fn from(value: RemoteLlmToolSpec) -> Self {
        Self {
            name: value.name,
            description: value.description,
            input_schema: value.input_schema,
            output_schema: value.output_schema,
            input_schema_projections: value
                .input_schema_projections
                .into_iter()
                .map(Into::into)
                .collect(),
            output_schema_projections: value
                .output_schema_projections
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
            core_llm::LlmOutputSpec::JsonSchema(schema) => Self::JsonSchema {
                name: schema.name,
                schema: schema.schema,
                strict: schema.strict,
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
        Self {
            input_tokens: value.input_tokens,
            output_tokens: value.output_tokens,
            cached_input_tokens: value.cached_input_tokens,
            reasoning_tokens: value.reasoning_tokens,
        }
    }
}

impl From<RemoteUsage> for core_llm::LlmUsage {
    fn from(value: RemoteUsage) -> Self {
        Self {
            input_tokens: value.input_tokens,
            output_tokens: value.output_tokens,
            cached_input_tokens: value.cached_input_tokens,
            reasoning_tokens: value.reasoning_tokens,
        }
    }
}

impl From<lash_core::TokenUsage> for RemoteUsage {
    fn from(value: lash_core::TokenUsage) -> Self {
        Self {
            input_tokens: value.input_tokens,
            output_tokens: value.output_tokens,
            cached_input_tokens: value.cached_input_tokens,
            reasoning_tokens: value.reasoning_tokens,
        }
    }
}

impl From<RemoteUsage> for lash_core::TokenUsage {
    fn from(value: RemoteUsage) -> Self {
        Self {
            input_tokens: value.input_tokens,
            output_tokens: value.output_tokens,
            cached_input_tokens: value.cached_input_tokens,
            reasoning_tokens: value.reasoning_tokens,
        }
    }
}

impl From<lash_core::TokenLedgerEntry> for RemoteTokenLedgerEntry {
    fn from(value: lash_core::TokenLedgerEntry) -> Self {
        Self {
            source: value.source,
            model: value.model,
            usage: value.usage.into(),
        }
    }
}

impl From<RemoteTokenLedgerEntry> for lash_core::TokenLedgerEntry {
    fn from(value: RemoteTokenLedgerEntry) -> Self {
        Self {
            source: value.source,
            model: value.model,
            usage: value.usage.into(),
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
        let parent = RemoteUsage::from(turn.token_usage);
        let children = turn
            .children_usage
            .into_iter()
            .map(RemoteTokenLedgerEntry::from)
            .collect::<Vec<_>>();
        let mut total = parent.clone();
        for child in &children {
            total.add(&child.usage);
        }
        let outcome = RemoteTurnOutcome::from(turn.outcome);
        let status = RemoteTurnStatus::from(&outcome);
        Self {
            protocol_version: REMOTE_PROTOCOL_VERSION,
            session_id: session_id.into(),
            turn_id: turn_id.into(),
            status,
            outcome,
            assistant_output: turn.assistant_output.into(),
            usage: RemoteTurnUsageSummary {
                parent,
                children,
                total,
            },
            execution: turn.execution.into(),
            tool_calls: turn.tool_calls.into_iter().map(Into::into).collect(),
            issues: turn.errors.into_iter().map(Into::into).collect(),
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
        Self {
            safe_text: value.safe_text,
            raw_text: value.raw_text,
            state: value.state.into(),
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
        Self {
            had_tool_calls: value.had_tool_calls,
            had_code_execution: value.had_code_execution,
        }
    }
}

impl From<lash_core::ToolCallRecord> for RemoteToolCallSummary {
    fn from(value: lash_core::ToolCallRecord) -> Self {
        Self {
            call_id: value.call_id,
            tool_name: value.tool,
            args: value.args,
            outcome: value.output.into(),
            duration_ms: value.duration_ms,
        }
    }
}

impl From<lash_core::ToolCallOutput> for RemoteToolCallOutcome {
    fn from(value: lash_core::ToolCallOutput) -> Self {
        match value.outcome {
            lash_core::ToolCallOutcome::Success(value) => Self::Success(value.to_json_value()),
            lash_core::ToolCallOutcome::Failure(value) => Self::Failure(value.to_json_value()),
            lash_core::ToolCallOutcome::Cancelled(value) => Self::Cancelled(value.to_json_value()),
        }
    }
}

impl From<lash_core::TurnIssue> for RemoteTurnIssue {
    fn from(value: lash_core::TurnIssue) -> Self {
        Self {
            kind: value.kind,
            code: value.code,
            terminal_reason: value.terminal_reason.map(Into::into),
            message: value.message,
            raw: value.raw,
        }
    }
}

impl RemoteTurnActivity {
    pub fn from_core(sequence: u64, activity: lash_core::TurnActivity) -> Self {
        Self {
            protocol_version: REMOTE_PROTOCOL_VERSION,
            sequence,
            id: activity.id.0,
            correlation_id: activity.correlation_id.0,
            event: RemoteTurnEvent::from(activity.event),
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
        let payload = match event.payload {
            lash_core::SessionObservationEventPayload::TurnActivity(activity) => {
                RemoteSessionObservationEventPayload::TurnActivity {
                    activity: RemoteTurnActivity::from_core(sequence, activity),
                }
            }
            lash_core::SessionObservationEventPayload::Committed { .. } => {
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
            session_id: event.session_id,
            revision: event.revision.as_u64(),
            cursor: event.cursor.to_string(),
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
        Self {
            protocol_version: REMOTE_PROTOCOL_VERSION,
            session_id: value.session_id,
            requested_cursor: value.requested_cursor.to_string(),
            latest_cursor: value.latest_cursor.to_string(),
            latest_revision: value.latest_revision.as_u64(),
            reason: value.reason.into(),
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
        Self {
            template: value.template.map(Into::into),
            slots: value
                .slots
                .into_iter()
                .map(|(slot, layer)| (slot.into(), layer.into()))
                .collect(),
        }
    }
}

impl From<RemotePromptLayer> for lash_core::PromptLayer {
    fn from(value: RemotePromptLayer) -> Self {
        Self {
            template: value.template.map(Into::into),
            slots: value
                .slots
                .into_iter()
                .map(|(slot, layer)| (slot.into(), layer.into()))
                .collect(),
        }
    }
}

impl From<lash_core::PromptTemplate> for RemotePromptTemplate {
    fn from(value: lash_core::PromptTemplate) -> Self {
        Self {
            sections: value.sections.into_iter().map(Into::into).collect(),
        }
    }
}

impl From<RemotePromptTemplate> for lash_core::PromptTemplate {
    fn from(value: RemotePromptTemplate) -> Self {
        Self {
            sections: value.sections.into_iter().map(Into::into).collect(),
        }
    }
}

impl From<lash_core::PromptTemplateSection> for RemotePromptTemplateSection {
    fn from(value: lash_core::PromptTemplateSection) -> Self {
        Self {
            title: value.title,
            entries: value.entries.into_iter().map(Into::into).collect(),
        }
    }
}

impl From<RemotePromptTemplateSection> for lash_core::PromptTemplateSection {
    fn from(value: RemotePromptTemplateSection) -> Self {
        Self {
            title: value.title,
            entries: value.entries.into_iter().map(Into::into).collect(),
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
        Self {
            reset: value.reset,
            contributions: value.contributions.into_iter().map(Into::into).collect(),
        }
    }
}

impl From<RemotePromptSlotLayer> for lash_core::PromptSlotLayer {
    fn from(value: RemotePromptSlotLayer) -> Self {
        Self {
            reset: value.reset,
            contributions: value.contributions.into_iter().map(Into::into).collect(),
        }
    }
}

impl From<lash_core::PromptContribution> for RemotePromptContribution {
    fn from(value: lash_core::PromptContribution) -> Self {
        Self {
            slot: value.slot.into(),
            title: value.title.map(|title| title.to_string()),
            priority: value.priority,
            gate: value.gate.into(),
            content: value.content.to_string(),
        }
    }
}

impl From<RemotePromptContribution> for lash_core::PromptContribution {
    fn from(value: RemotePromptContribution) -> Self {
        Self {
            slot: value.slot.into(),
            title: value.title.map(Arc::from),
            priority: value.priority,
            gate: value.gate.into(),
            content: Arc::from(value.content),
        }
    }
}

impl From<lash_core::PromptContributionGate> for RemotePromptContributionGate {
    fn from(value: lash_core::PromptContributionGate) -> Self {
        Self {
            tools: value.tools,
            minimum_availability: value.minimum_availability.into(),
        }
    }
}

impl From<RemotePromptContributionGate> for lash_core::PromptContributionGate {
    fn from(value: RemotePromptContributionGate) -> Self {
        Self {
            tools: value.tools,
            minimum_availability: value.minimum_availability.into(),
        }
    }
}

impl From<&RemoteToolAgentSurface> for lash_core::ToolAgentSurface {
    fn from(value: &RemoteToolAgentSurface) -> Self {
        let mut surface =
            lash_core::ToolAgentSurface::new(value.module_path.clone(), value.operation.clone());
        if let Some(authority_type) = value.authority_type.as_ref() {
            surface = surface.with_authority_type(authority_type.clone());
        }
        surface.with_aliases(value.aliases.clone())
    }
}

impl From<lash_core::SchemaProjectionOverride> for RemoteSchemaProjectionOverride {
    fn from(value: lash_core::SchemaProjectionOverride) -> Self {
        Self {
            profile: value.profile,
            schema: value.schema,
        }
    }
}

impl From<RemoteSchemaProjectionOverride> for lash_core::SchemaProjectionOverride {
    fn from(value: RemoteSchemaProjectionOverride) -> Self {
        Self {
            profile: value.profile,
            schema: value.schema,
        }
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
        let mut definition = ToolDefinition::raw_with_id(
            value
                .id
                .clone()
                .unwrap_or_else(|| format!("remote-tool:{}", value.call_path().unwrap())),
            value.name.clone(),
            value.description.clone(),
            value.input_schema.clone(),
            value.output_schema.clone(),
        )
        .with_agent_surface(
            value
                .agent_surface
                .as_ref()
                .expect("validated agent surface")
                .into(),
        )
        .with_examples(value.examples.clone())
        .with_output_contract(value.output_contract.clone().into());
        if let Some(availability) = value.availability {
            definition = definition
                .with_availability(lash_core::ToolAvailabilityConfig::same(availability.into()));
        }
        if let Some(activation) = value.activation {
            definition = definition.with_activation(activation.into());
        }
        if let Some(argument_projection) = value.argument_projection.clone() {
            definition = definition.with_argument_projection(argument_projection.into());
        }
        if let Some(scheduling) = value.scheduling {
            definition = definition.with_scheduling(scheduling.into());
        }
        if let Some(retry_policy) = value.retry_policy {
            definition = definition.with_retry_policy(retry_policy.into());
        }
        for projection in &value.input_schema_projections {
            definition = definition.with_input_schema_projection(
                projection.profile.clone(),
                projection.schema.clone(),
            );
        }
        for projection in &value.output_schema_projections {
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
            Self::Success { value, .. } => ToolResult::ok(value),
            Self::Failure {
                code,
                message,
                raw,
                retry_after_ms,
                ..
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
            Self::Cancelled { message, raw, .. } => {
                if let Some(raw) = raw {
                    ToolResult::cancelled_with_raw(message, raw)
                } else {
                    ToolResult::cancelled(message)
                }
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
            let executable =
                lash_core::ToolAgentSurface::required_for_remote(&manifest).map_err(|message| {
                    RemoteProtocolError::InvalidToolGrant {
                        tool_name: manifest.name.clone(),
                        message,
                    }
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
