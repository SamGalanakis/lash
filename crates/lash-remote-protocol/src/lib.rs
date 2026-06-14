use std::collections::{HashMap, HashSet};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

pub const REMOTE_PROTOCOL_VERSION: u32 = 3;

pub fn ensure_protocol_version(actual: u32) -> Result<(), RemoteProtocolError> {
    if actual == REMOTE_PROTOCOL_VERSION {
        Ok(())
    } else {
        Err(RemoteProtocolError::UnsupportedProtocolVersion {
            actual,
            expected: REMOTE_PROTOCOL_VERSION,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RemoteLlmRequest {
    pub protocol_version: u32,
    pub request_id: String,
    pub model_intent: RemoteModelIntent,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub messages: Vec<RemoteLlmMessage>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub attachments: Vec<RemoteLlmAttachment>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<RemoteLlmToolSpec>,
    #[serde(default)]
    pub tool_choice: RemoteLlmToolChoice,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_spec: Option<RemoteLlmOutputSpec>,
    #[serde(default, skip_serializing_if = "RemoteGenerationOptions::is_empty")]
    pub generation: RemoteGenerationOptions,
    #[serde(default, skip_serializing_if = "RemoteLlmRequestMetadata::is_empty")]
    pub request_metadata: RemoteLlmRequestMetadata,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, serde_json::Value>,
}

impl RemoteLlmRequest {
    pub fn validate(&self) -> Result<(), RemoteProtocolError> {
        ensure_protocol_version(self.protocol_version)?;
        require_non_empty("RemoteLlmRequest", "request_id", &self.request_id)?;
        self.model_intent.validate()?;
        self.generation.validate("RemoteLlmRequest")?;
        for (index, message) in self.messages.iter().enumerate() {
            message.validate(index)?;
        }
        for (index, attachment) in self.attachments.iter().enumerate() {
            attachment.validate(index)?;
        }
        for tool in &self.tools {
            tool.validate()?;
        }
        if let Some(output_spec) = &self.output_spec {
            output_spec.validate()?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RemoteLlmResponse {
    pub protocol_version: u32,
    pub request_id: String,
    #[serde(default)]
    pub full_text: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub output_parts: Vec<RemoteLlmOutputPart>,
    #[serde(default)]
    pub usage: RemoteUsage,
    #[serde(default)]
    pub terminal_reason: RemoteLlmTerminalReason,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub diagnostics: Vec<RemoteDiagnostic>,
    #[serde(default, skip_serializing_if = "RemoteProviderMetadata::is_empty")]
    pub provider_metadata: RemoteProviderMetadata,
}

impl RemoteLlmResponse {
    pub fn validate(&self) -> Result<(), RemoteProtocolError> {
        ensure_protocol_version(self.protocol_version)?;
        require_non_empty("RemoteLlmResponse", "request_id", &self.request_id)?;
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct RemoteModelIntent {
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub variant: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, String>,
}

impl RemoteModelIntent {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            variant: None,
            provider: None,
            metadata: HashMap::new(),
        }
    }

    fn validate(&self) -> Result<(), RemoteProtocolError> {
        require_non_empty("RemoteModelIntent", "model", &self.model)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct RemoteGenerationOptions {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_token_cap: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stop: Vec<String>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub provider_options: HashMap<String, String>,
}

impl RemoteGenerationOptions {
    pub fn is_empty(&self) -> bool {
        self.output_token_cap.is_none()
            && self.temperature.is_none()
            && self.top_p.is_none()
            && self.stop.is_empty()
            && self.provider_options.is_empty()
    }

    fn validate(&self, type_name: &'static str) -> Result<(), RemoteProtocolError> {
        if self.output_token_cap == Some(0) {
            return Err(RemoteProtocolError::InvalidEnvelope {
                type_name,
                message: "generation.output_token_cap must be greater than zero".to_string(),
            });
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct RemoteLlmRequestMetadata {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub idempotency_key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trace_id: Option<String>,
}

impl RemoteLlmRequestMetadata {
    pub fn is_empty(&self) -> bool {
        self.session_id.is_none() && self.idempotency_key.is_none() && self.trace_id.is_none()
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RemoteLlmRole {
    #[default]
    User,
    Assistant,
    System,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct RemoteLlmMessage {
    pub role: RemoteLlmRole,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub content: Vec<RemoteLlmContentBlock>,
}

impl RemoteLlmMessage {
    fn validate(&self, index: usize) -> Result<(), RemoteProtocolError> {
        if self.content.is_empty() {
            return Err(RemoteProtocolError::InvalidEnvelope {
                type_name: "RemoteLlmMessage",
                message: format!("message at index {index} must contain at least one block"),
            });
        }
        for block in &self.content {
            block.validate()?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RemoteLlmContentBlock {
    Text {
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        response_meta: Option<RemoteResponseTextMeta>,
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        cache_breakpoint: bool,
    },
    ImageAttachment {
        attachment_index: usize,
    },
    ToolCall {
        call_id: String,
        tool_name: String,
        input_json: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        replay: Option<RemoteProviderReplayMeta>,
    },
    ToolResult {
        call_id: String,
        content: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        tool_name: Option<String>,
    },
    Reasoning {
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        replay: Option<RemoteProviderReasoningReplay>,
    },
}

impl RemoteLlmContentBlock {
    fn validate(&self) -> Result<(), RemoteProtocolError> {
        match self {
            Self::ToolCall {
                call_id, tool_name, ..
            } => {
                require_non_empty("RemoteLlmContentBlock::ToolCall", "call_id", call_id)?;
                require_non_empty("RemoteLlmContentBlock::ToolCall", "tool_name", tool_name)
            }
            Self::ToolResult { call_id, .. } => {
                require_non_empty("RemoteLlmContentBlock::ToolResult", "call_id", call_id)
            }
            Self::Text { .. } | Self::ImageAttachment { .. } | Self::Reasoning { .. } => Ok(()),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct RemoteResponseTextMeta {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phase: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct RemoteProviderReplayMeta {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub item_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub opaque: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct RemoteProviderReasoningReplay {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub item_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub encrypted_content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub redacted: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub summary: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct RemoteLlmAttachment {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub mime: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data_base64: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference: Option<RemoteAttachmentRef>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, String>,
}

impl RemoteLlmAttachment {
    fn validate(&self, index: usize) -> Result<(), RemoteProtocolError> {
        if self.mime.trim().is_empty() {
            return Err(RemoteProtocolError::InvalidEnvelope {
                type_name: "RemoteLlmAttachment",
                message: format!("attachment at index {index} requires a non-empty mime"),
            });
        }
        if let Some(reference) = &self.reference {
            reference.validate()?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct RemoteAttachmentRef {
    pub id: String,
    pub mime: String,
    pub byte_len: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub width: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub height: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, String>,
}

impl RemoteAttachmentRef {
    fn validate(&self) -> Result<(), RemoteProtocolError> {
        require_non_empty("RemoteAttachmentRef", "id", &self.id)?;
        require_non_empty("RemoteAttachmentRef", "mime", &self.mime)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct RemoteLlmToolSpec {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default = "default_input_schema")]
    pub input_schema: serde_json::Value,
    #[serde(default)]
    pub output_schema: serde_json::Value,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub input_schema_projections: Vec<RemoteSchemaProjectionOverride>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub output_schema_projections: Vec<RemoteSchemaProjectionOverride>,
}

impl RemoteLlmToolSpec {
    fn validate(&self) -> Result<(), RemoteProtocolError> {
        require_non_empty("RemoteLlmToolSpec", "name", &self.name)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RemoteLlmToolChoice {
    #[default]
    Auto,
    None,
    Required,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RemoteLlmOutputSpec {
    JsonObject,
    JsonSchema {
        name: String,
        schema: serde_json::Value,
        strict: bool,
    },
}

impl RemoteLlmOutputSpec {
    fn validate(&self) -> Result<(), RemoteProtocolError> {
        match self {
            Self::JsonObject => Ok(()),
            Self::JsonSchema { name, .. } => {
                require_non_empty("RemoteLlmOutputSpec::JsonSchema", "name", name)
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RemoteLlmOutputPart {
    Text {
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        response_meta: Option<RemoteResponseTextMeta>,
    },
    Reasoning {
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        replay: Option<RemoteProviderReasoningReplay>,
    },
    ToolCall {
        call_id: String,
        tool_name: String,
        input_json: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        replay: Option<RemoteProviderReplayMeta>,
    },
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RemoteLlmTerminalReason {
    Stop,
    ToolUse,
    OutputLimit,
    ContextOverflow,
    ContentFilter,
    ProviderError,
    Cancelled,
    #[default]
    Unknown,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct RemoteProviderMetadata {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_body: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub http_summary: Option<String>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub data: HashMap<String, serde_json::Value>,
}

impl RemoteProviderMetadata {
    pub fn is_empty(&self) -> bool {
        self.usage.is_none()
            && self.request_body.is_none()
            && self.http_summary.is_none()
            && self.data.is_empty()
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct RemoteDiagnostic {
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct RemoteProtocolTurnOptions {
    #[serde(default = "empty_protocol_turn_payload")]
    pub payload: serde_json::Value,
}

fn empty_protocol_turn_payload() -> serde_json::Value {
    serde_json::Value::Object(serde_json::Map::new())
}

impl Default for RemoteProtocolTurnOptions {
    fn default() -> Self {
        Self {
            payload: empty_protocol_turn_payload(),
        }
    }
}

impl RemoteProtocolTurnOptions {
    pub fn empty() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        match &self.payload {
            serde_json::Value::Object(map) => map.is_empty(),
            _ => false,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct RemoteTurnInput {
    pub protocol_version: u32,
    #[serde(default)]
    pub items: Vec<RemoteInputItem>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub image_blobs_base64: HashMap<String, String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub protocol_turn_options: Option<RemoteProtocolTurnOptions>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trace_turn_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_layer: Option<RemotePromptLayer>,
}

impl RemoteTurnInput {
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            protocol_version: REMOTE_PROTOCOL_VERSION,
            items: vec![RemoteInputItem::Text { text: text.into() }],
            image_blobs_base64: HashMap::new(),
            protocol_turn_options: None,
            trace_turn_id: None,
            prompt_layer: None,
        }
    }

    pub fn validate(&self) -> Result<(), RemoteProtocolError> {
        ensure_protocol_version(self.protocol_version)?;
        for item in &self.items {
            if let RemoteInputItem::ImageRef { id } = item {
                require_non_empty("RemoteInputItem::ImageRef", "id", id)?;
            }
        }
        for (id, blob) in &self.image_blobs_base64 {
            require_non_empty("RemoteTurnInput", "image_blobs_base64 key", id)?;
            if blob.trim().is_empty() {
                return Err(RemoteProtocolError::InvalidImageBlob {
                    id: id.clone(),
                    message: "base64 payload cannot be empty".to_string(),
                });
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RemoteInputItem {
    Text { text: String },
    ImageRef { id: String },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RemoteTurnRequest {
    pub protocol_version: u32,
    pub session_id: String,
    pub turn_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub idempotency_key: Option<String>,
    pub input: RemoteTurnInput,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_grants: Vec<RemoteToolGrant>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_intent: Option<RemoteModelIntent>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, serde_json::Value>,
}

impl RemoteTurnRequest {
    pub fn validate(&self) -> Result<(), RemoteProtocolError> {
        ensure_protocol_version(self.protocol_version)?;
        require_non_empty("RemoteTurnRequest", "session_id", &self.session_id)?;
        require_non_empty("RemoteTurnRequest", "turn_id", &self.turn_id)?;
        if self.input.protocol_version != self.protocol_version {
            return Err(RemoteProtocolError::MismatchedNestedProtocolVersion {
                parent: "RemoteTurnRequest",
                child: "input",
                parent_version: self.protocol_version,
                child_version: self.input.protocol_version,
            });
        }
        self.input.validate()?;
        RemoteToolGrant::validate_all(&self.tool_grants)?;
        if let Some(model_intent) = &self.model_intent {
            model_intent.validate()?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RemoteTurnResult {
    pub protocol_version: u32,
    pub session_id: String,
    pub turn_id: String,
    pub status: RemoteTurnStatus,
    pub outcome: RemoteTurnOutcome,
    pub assistant_output: RemoteAssistantOutput,
    #[serde(default)]
    pub usage: RemoteTurnUsageSummary,
    #[serde(default)]
    pub execution: RemoteExecutionSummary,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<RemoteToolCallSummary>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub issues: Vec<RemoteTurnIssue>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub activities: Vec<RemoteTurnActivity>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, serde_json::Value>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct RemoteSessionCursor {
    pub protocol_version: u32,
    pub cursor: String,
}

impl RemoteSessionCursor {
    pub fn new(cursor: impl Into<String>) -> Self {
        Self {
            protocol_version: REMOTE_PROTOCOL_VERSION,
            cursor: cursor.into(),
        }
    }

    pub fn validate(&self) -> Result<(), RemoteProtocolError> {
        ensure_protocol_version(self.protocol_version)?;
        require_non_empty("RemoteSessionCursor", "cursor", &self.cursor)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RemoteSessionObservationEvent {
    pub protocol_version: u32,
    pub session_id: String,
    pub revision: u64,
    pub cursor: String,
    #[serde(flatten)]
    pub event: RemoteSessionObservationEventPayload,
}

impl RemoteSessionObservationEvent {
    pub fn validate(&self) -> Result<(), RemoteProtocolError> {
        ensure_protocol_version(self.protocol_version)?;
        require_non_empty(
            "RemoteSessionObservationEvent",
            "session_id",
            &self.session_id,
        )?;
        require_non_empty("RemoteSessionObservationEvent", "cursor", &self.cursor)?;
        if let RemoteSessionObservationEventPayload::TurnActivity { activity } = &self.event {
            activity.validate()?;
            if activity.protocol_version != self.protocol_version {
                return Err(RemoteProtocolError::MismatchedNestedProtocolVersion {
                    parent: "RemoteSessionObservationEvent",
                    child: "activity",
                    parent_version: self.protocol_version,
                    child_version: activity.protocol_version,
                });
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RemoteSessionObservationEventPayload {
    TurnActivity {
        activity: RemoteTurnActivity,
    },
    Committed,
    AgentFrameSwitched {
        frame_id: String,
    },
    QueueChanged {
        kind: RemoteSessionQueueEventKind,
        batch_ids: Vec<String>,
    },
    ProcessChanged {
        kind: RemoteSessionProcessEventKind,
        process_ids: Vec<String>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RemoteSessionQueueEventKind {
    Enqueued,
    Cancelled,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RemoteSessionProcessEventKind {
    Started,
    Cancelled,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct RemoteLiveReplayGap {
    pub protocol_version: u32,
    pub session_id: String,
    pub requested_cursor: String,
    pub latest_cursor: String,
    pub latest_revision: u64,
    pub reason: RemoteLiveReplayGapReason,
}

impl RemoteLiveReplayGap {
    pub fn validate(&self) -> Result<(), RemoteProtocolError> {
        ensure_protocol_version(self.protocol_version)?;
        require_non_empty("RemoteLiveReplayGap", "session_id", &self.session_id)?;
        require_non_empty(
            "RemoteLiveReplayGap",
            "requested_cursor",
            &self.requested_cursor,
        )?;
        require_non_empty("RemoteLiveReplayGap", "latest_cursor", &self.latest_cursor)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RemoteLiveReplayGapReason {
    Trimmed,
    Unavailable,
}

impl RemoteTurnResult {
    pub fn validate(&self) -> Result<(), RemoteProtocolError> {
        ensure_protocol_version(self.protocol_version)?;
        require_non_empty("RemoteTurnResult", "session_id", &self.session_id)?;
        require_non_empty("RemoteTurnResult", "turn_id", &self.turn_id)?;
        for activity in &self.activities {
            if activity.protocol_version != self.protocol_version {
                return Err(RemoteProtocolError::MismatchedNestedProtocolVersion {
                    parent: "RemoteTurnResult",
                    child: "activities",
                    parent_version: self.protocol_version,
                    child_version: activity.protocol_version,
                });
            }
            activity.validate()?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RemoteTriggerOccurrenceRequest {
    pub protocol_version: u32,
    pub source_type: String,
    pub source_key: String,
    #[serde(default)]
    pub payload: serde_json::Value,
    pub idempotency_key: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<serde_json::Value>,
}

impl RemoteTriggerOccurrenceRequest {
    pub fn new(
        source_type: impl Into<String>,
        source_key: impl Into<String>,
        payload: serde_json::Value,
        idempotency_key: impl Into<String>,
    ) -> Self {
        Self {
            protocol_version: REMOTE_PROTOCOL_VERSION,
            source_type: source_type.into(),
            source_key: source_key.into(),
            payload,
            idempotency_key: idempotency_key.into(),
            source: None,
        }
    }

    pub fn with_source(mut self, source: serde_json::Value) -> Self {
        self.source = Some(source);
        self
    }

    pub fn validate(&self) -> Result<(), RemoteProtocolError> {
        ensure_protocol_version(self.protocol_version)?;
        require_non_empty(
            "RemoteTriggerOccurrenceRequest",
            "source_type",
            &self.source_type,
        )?;
        require_non_empty(
            "RemoteTriggerOccurrenceRequest",
            "source_key",
            &self.source_key,
        )?;
        require_non_empty(
            "RemoteTriggerOccurrenceRequest",
            "idempotency_key",
            &self.idempotency_key,
        )
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RemoteTriggerOccurrenceRecord {
    pub occurrence_id: String,
    pub source_type: String,
    pub source_key: String,
    #[serde(default)]
    pub payload: serde_json::Value,
    pub idempotency_key: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<serde_json::Value>,
    pub occurred_at_ms: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct RemoteTriggerEmitReport {
    pub protocol_version: u32,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub occurrence_id: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub started_process_ids: Vec<String>,
}

impl RemoteTriggerEmitReport {
    pub fn validate(&self) -> Result<(), RemoteProtocolError> {
        ensure_protocol_version(self.protocol_version)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RemoteTriggerSubscriptionFilter {
    pub protocol_version: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub handle: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
}

impl Default for RemoteTriggerSubscriptionFilter {
    fn default() -> Self {
        Self {
            protocol_version: REMOTE_PROTOCOL_VERSION,
            session_id: None,
            handle: None,
            name: None,
            source_type: None,
            source_key: None,
            target: None,
            enabled: None,
        }
    }
}

impl RemoteTriggerSubscriptionFilter {
    pub fn for_session(session_id: impl Into<String>) -> Self {
        Self {
            protocol_version: REMOTE_PROTOCOL_VERSION,
            session_id: Some(session_id.into()),
            ..Self::default()
        }
    }

    pub fn for_source_type(source_type: impl Into<String>) -> Self {
        Self {
            protocol_version: REMOTE_PROTOCOL_VERSION,
            source_type: Some(source_type.into()),
            ..Self::default()
        }
    }

    pub fn validate(&self) -> Result<(), RemoteProtocolError> {
        ensure_protocol_version(self.protocol_version)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RemoteTriggerRegistration {
    pub handle: String,
    pub source_key: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub source_type: String,
    #[serde(default)]
    pub source: serde_json::Value,
    pub target: RemoteTriggerTargetSummary,
    #[serde(default = "default_true")]
    pub enabled: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RemoteTriggerTargetSummary {
    pub process_name: String,
    #[serde(default)]
    pub inputs: serde_json::Value,
}

fn default_true() -> bool {
    true
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RemoteCausalRef {
    Turn {
        session_id: String,
        turn_id: String,
    },
    Effect {
        session_id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        turn_id: Option<String>,
        effect_id: String,
    },
    ToolCall {
        session_id: String,
        call_id: String,
    },
    Process {
        process_id: String,
    },
    ProcessEvent {
        process_id: String,
        sequence: u64,
    },
    TriggerOccurrence {
        occurrence_id: String,
    },
    SessionNode {
        session_id: String,
        node_id: String,
    },
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RemoteTurnStatus {
    #[default]
    Completed,
    Failed,
    Cancelled,
    InProgress,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RemoteTurnOutcome {
    Finished { finish: RemoteTurnFinish },
    AgentFrameSwitch { frame_id: String, task: String },
    Stopped { stop: RemoteTurnStop },
}

impl Default for RemoteTurnOutcome {
    fn default() -> Self {
        Self::Stopped {
            stop: RemoteTurnStop::Incomplete,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RemoteTurnFinish {
    AssistantMessage {
        text: String,
    },
    SubmittedValue {
        value: serde_json::Value,
    },
    ToolValue {
        tool_name: String,
        value: serde_json::Value,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RemoteTurnStop {
    Cancelled,
    Incomplete,
    InvalidInput,
    MaxTurns,
    ToolFailure,
    ProviderError,
    PluginAbort,
    RuntimeError,
    SubmittedError {
        value: serde_json::Value,
    },
    ToolError {
        tool_name: String,
        value: serde_json::Value,
    },
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct RemoteAssistantOutput {
    #[serde(default)]
    pub safe_text: String,
    #[serde(default)]
    pub raw_text: String,
    #[serde(default)]
    pub state: RemoteAssistantOutputState,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RemoteAssistantOutputState {
    #[default]
    Usable,
    EmptyOutput,
    TracebackOnly,
    RecoveredFromError,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct RemoteTurnUsageSummary {
    #[serde(default)]
    pub parent: RemoteUsage,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub children: Vec<RemoteTokenLedgerEntry>,
    #[serde(default)]
    pub total: RemoteUsage,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct RemoteExecutionSummary {
    #[serde(default)]
    pub had_tool_calls: bool,
    #[serde(default)]
    pub had_code_execution: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RemoteToolCallSummary {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub call_id: Option<String>,
    pub tool_name: String,
    #[serde(default)]
    pub args: serde_json::Value,
    pub outcome: RemoteToolCallOutcome,
    pub duration_ms: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "status", content = "payload", rename_all = "snake_case")]
pub enum RemoteToolCallOutcome {
    Success(serde_json::Value),
    Failure(serde_json::Value),
    Cancelled(serde_json::Value),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RemoteTurnIssue {
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub terminal_reason: Option<RemoteLlmTerminalReason>,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct RemotePromptLayer {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub template: Option<RemotePromptTemplate>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub slots: HashMap<RemotePromptSlot, RemotePromptSlotLayer>,
}

impl RemotePromptLayer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.template.is_none() && self.slots.is_empty()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RemotePromptBuiltin {
    MainAgentIntro,
    ExecutionInstructions,
    CoreGuidance,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RemotePromptSlot {
    Intro,
    Execution,
    Guidance,
    ProjectInstructions,
    RuntimeContext,
    Environment,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RemotePromptTemplateEntry {
    Text { content: String },
    Builtin { builtin: RemotePromptBuiltin },
    Slot { slot: RemotePromptSlot },
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
pub struct RemotePromptTemplateSection {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub entries: Vec<RemotePromptTemplateEntry>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
pub struct RemotePromptTemplate {
    pub sections: Vec<RemotePromptTemplateSection>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct RemotePromptSlotLayer {
    #[serde(default)]
    pub reset: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub contributions: Vec<RemotePromptContribution>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct RemotePromptContribution {
    pub slot: RemotePromptSlot,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(default)]
    pub priority: i32,
    #[serde(
        default,
        skip_serializing_if = "RemotePromptContributionGate::is_empty"
    )]
    pub gate: RemotePromptContributionGate,
    pub content: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct RemotePromptContributionGate {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<String>,
    #[serde(default)]
    pub minimum_availability: RemoteToolAvailability,
}

impl RemotePromptContributionGate {
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RemoteToolGrant {
    pub protocol_version: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub name: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub description: String,
    #[serde(default = "default_input_schema")]
    pub input_schema: serde_json::Value,
    #[serde(default)]
    pub output_schema: serde_json::Value,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub input_schema_projections: Vec<RemoteSchemaProjectionOverride>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub output_schema_projections: Vec<RemoteSchemaProjectionOverride>,
    #[serde(default, skip_serializing_if = "RemoteToolOutputContract::is_static")]
    pub output_contract: RemoteToolOutputContract,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub examples: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub availability: Option<RemoteToolAvailability>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub activation: Option<RemoteToolActivation>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub argument_projection: Option<RemoteToolArgumentProjectionPolicy>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scheduling: Option<RemoteToolScheduling>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retry_policy: Option<RemoteToolRetryPolicy>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lashlang_binding: Option<RemoteLashlangToolBinding>,
}

impl RemoteToolGrant {
    pub fn call_path(&self) -> Result<String, RemoteProtocolError> {
        let binding = self.required_lashlang_binding()?;
        Ok(format!(
            "{}.{}",
            binding.module_path.join("."),
            binding.operation
        ))
    }

    pub fn validate(&self) -> Result<(), RemoteProtocolError> {
        ensure_protocol_version(self.protocol_version)?;
        if self.name.trim().is_empty() {
            return Err(RemoteProtocolError::InvalidToolGrant {
                tool_name: self.name.clone(),
                message: "tool grant name cannot be empty".to_string(),
            });
        }
        self.required_lashlang_binding()?;
        Ok(())
    }

    pub fn validate_all(grants: &[Self]) -> Result<(), RemoteProtocolError> {
        let mut seen = HashSet::new();
        for grant in grants {
            grant.validate()?;
            let call_path = grant.call_path()?;
            if !seen.insert(call_path.clone()) {
                return Err(RemoteProtocolError::DuplicateRemoteCallPath { call_path });
            }
        }
        Ok(())
    }

    fn required_lashlang_binding(&self) -> Result<&RemoteLashlangToolBinding, RemoteProtocolError> {
        let Some(binding) = &self.lashlang_binding else {
            return Err(RemoteProtocolError::MissingLashlangToolBinding {
                tool_name: self.name.clone(),
            });
        };
        if binding.module_path.is_empty() {
            return Err(RemoteProtocolError::InvalidToolGrant {
                tool_name: self.name.clone(),
                message: "remote tool grant requires an explicit module path".to_string(),
            });
        }
        if binding
            .module_path
            .iter()
            .any(|part| part.trim().is_empty())
        {
            return Err(RemoteProtocolError::InvalidToolGrant {
                tool_name: self.name.clone(),
                message: "remote tool grant module path cannot contain empty segments".to_string(),
            });
        }
        if binding.operation.trim().is_empty() {
            return Err(RemoteProtocolError::InvalidToolGrant {
                tool_name: self.name.clone(),
                message: "remote tool grant requires an explicit operation".to_string(),
            });
        }
        Ok(binding)
    }
}

fn default_input_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {},
        "additionalProperties": true
    })
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct RemoteLashlangToolBinding {
    pub module_path: Vec<String>,
    pub operation: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub authority_type: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub aliases: Vec<String>,
}

impl RemoteLashlangToolBinding {
    pub fn new(
        module_path: impl IntoIterator<Item = impl Into<String>>,
        operation: impl Into<String>,
    ) -> Self {
        Self {
            module_path: module_path.into_iter().map(Into::into).collect(),
            operation: operation.into(),
            authority_type: None,
            aliases: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct RemoteSchemaProjectionOverride {
    pub profile: String,
    pub schema: serde_json::Value,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RemoteToolAvailability {
    Off,
    Searchable,
    Callable,
    #[default]
    Showcased,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RemoteToolActivation {
    #[default]
    Always,
    Internal,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RemoteToolScheduling {
    #[default]
    Parallel,
    Serial,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RemoteToolOutputContract {
    #[default]
    Static,
    FromInputSchema {
        input_field: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        default_schema: Option<serde_json::Value>,
    },
}

impl RemoteToolOutputContract {
    fn is_static(&self) -> bool {
        matches!(self, Self::Static)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RemoteToolArgumentProjectionPolicy {
    #[default]
    MaterializeProjectedValues,
    PreserveProjectedRefsInField {
        field: String,
    },
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RemoteToolRetryPolicy {
    #[default]
    Never,
    Safe {
        max_attempts: u32,
        base_delay_ms: u64,
        max_delay_ms: u64,
    },
    Idempotent {
        max_attempts: u32,
        base_delay_ms: u64,
        max_delay_ms: u64,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RemoteToolCallRequest {
    pub protocol_version: u32,
    pub tool_name: String,
    pub call_path: String,
    pub args: serde_json::Value,
    pub session_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub replay_key: Option<String>,
    pub attempt_number: u32,
    pub max_attempts: u32,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub headers: HashMap<String, String>,
}

impl RemoteToolCallRequest {
    pub fn validate(&self) -> Result<(), RemoteProtocolError> {
        ensure_protocol_version(self.protocol_version)?;
        if self.tool_name.trim().is_empty() {
            return Err(RemoteProtocolError::UnknownRemoteTool {
                tool_name: self.tool_name.clone(),
            });
        }
        if self.call_path.trim().is_empty() {
            return Err(RemoteProtocolError::RemoteToolTransport(
                "remote tool call request requires a non-empty call_path".to_string(),
            ));
        }
        if self.session_id.trim().is_empty() {
            return Err(RemoteProtocolError::RemoteToolTransport(
                "remote tool call request requires a non-empty session_id".to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum RemoteToolCallResponse {
    Success {
        protocol_version: u32,
        #[serde(default)]
        value: serde_json::Value,
    },
    Failure {
        protocol_version: u32,
        #[serde(default = "default_failure_code")]
        code: String,
        message: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        raw: Option<serde_json::Value>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        retry_after_ms: Option<u64>,
    },
    Cancelled {
        protocol_version: u32,
        message: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        raw: Option<serde_json::Value>,
    },
}

impl RemoteToolCallResponse {
    pub fn protocol_version(&self) -> u32 {
        match self {
            Self::Success {
                protocol_version, ..
            }
            | Self::Failure {
                protocol_version, ..
            }
            | Self::Cancelled {
                protocol_version, ..
            } => *protocol_version,
        }
    }

    pub fn validate(&self) -> Result<(), RemoteProtocolError> {
        ensure_protocol_version(self.protocol_version())
    }
}

fn default_failure_code() -> String {
    "remote_tool_error".to_string()
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct RemoteUsage {
    pub input_tokens: i64,
    pub output_tokens: i64,
    pub cached_input_tokens: i64,
    #[serde(default)]
    pub reasoning_tokens: i64,
}

impl RemoteUsage {
    pub fn add(&mut self, other: &Self) {
        self.input_tokens += other.input_tokens;
        self.output_tokens += other.output_tokens;
        self.cached_input_tokens += other.cached_input_tokens;
        self.reasoning_tokens += other.reasoning_tokens;
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct RemoteTokenLedgerEntry {
    pub source: String,
    pub model: String,
    pub usage: RemoteUsage,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RemoteTurnActivity {
    pub protocol_version: u32,
    pub sequence: u64,
    pub id: String,
    pub correlation_id: String,
    #[serde(flatten)]
    pub event: RemoteTurnEvent,
}

impl RemoteTurnActivity {
    pub fn validate(&self) -> Result<(), RemoteProtocolError> {
        ensure_protocol_version(self.protocol_version)?;
        require_non_empty("RemoteTurnActivity", "id", &self.id)?;
        require_non_empty("RemoteTurnActivity", "correlation_id", &self.correlation_id)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RemoteTurnEvent {
    ModelRequestStarted {
        protocol_iteration: usize,
    },
    AssistantProseDelta {
        text: String,
    },
    ReasoningDelta {
        text: String,
    },
    CodeBlockStarted {
        language: String,
        code: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        graph_key: Option<String>,
    },
    CodeBlockCompleted {
        language: String,
        output: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
        success: bool,
        duration_ms: u64,
        tool_call_ids: Vec<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        graph_key: Option<String>,
    },
    ToolCallStarted {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        call_id: Option<String>,
        name: String,
        args: serde_json::Value,
    },
    ToolCallCompleted {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        call_id: Option<String>,
        name: String,
        args: serde_json::Value,
        output: serde_json::Value,
        duration_ms: u64,
    },
    SubmittedValue {
        value: serde_json::Value,
    },
    ToolValue {
        tool_name: String,
        value: serde_json::Value,
    },
    Usage {
        protocol_iteration: usize,
        usage: RemoteUsage,
        cumulative: RemoteUsage,
    },
    ChildUsage {
        session_id: String,
        source: String,
        model: String,
        protocol_iteration: usize,
        usage: RemoteUsage,
        cumulative: RemoteUsage,
    },
    RetryStatus {
        wait_seconds: u64,
        attempt: usize,
        max_attempts: usize,
        reason: String,
    },
    RuntimeDiagnostic {
        kind: String,
        data: serde_json::Value,
    },
    Error {
        message: String,
    },
}

pub trait RemoteToolRegistry {
    fn grants(&self) -> Vec<RemoteToolGrant>;

    fn validate_registry(&self) -> Result<(), RemoteProtocolError> {
        RemoteToolGrant::validate_all(&self.grants())
    }
}

pub fn assert_remote_tool_registry_reopenable(
    before: &dyn RemoteToolRegistry,
    after_reopen: &dyn RemoteToolRegistry,
) -> Result<(), RemoteProtocolError> {
    let before_grants = before.grants();
    let after_grants = after_reopen.grants();
    RemoteToolGrant::validate_all(&before_grants)?;
    RemoteToolGrant::validate_all(&after_grants)?;
    let before_paths = remote_registry_call_paths(&before_grants)?;
    let after_paths = remote_registry_call_paths(&after_grants)?;
    if before_paths != after_paths {
        return Err(RemoteProtocolError::RemoteToolRegistryReopenMismatch {
            before_call_paths: before_paths,
            after_call_paths: after_paths,
        });
    }
    Ok(())
}

fn remote_registry_call_paths(
    grants: &[RemoteToolGrant],
) -> Result<Vec<String>, RemoteProtocolError> {
    let mut call_paths = grants
        .iter()
        .map(RemoteToolGrant::call_path)
        .collect::<Result<Vec<_>, _>>()?;
    call_paths.sort();
    Ok(call_paths)
}

fn require_non_empty(
    type_name: &'static str,
    field: &'static str,
    value: &str,
) -> Result<(), RemoteProtocolError> {
    if value.trim().is_empty() {
        Err(RemoteProtocolError::MissingRequiredField { type_name, field })
    } else {
        Ok(())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum RemoteProtocolError {
    #[error("unsupported remote protocol version {actual}; expected {expected}")]
    UnsupportedProtocolVersion { actual: u32, expected: u32 },
    #[error(
        "mismatched protocol version in {parent}.{child}: got {child_version}, expected {parent_version}"
    )]
    MismatchedNestedProtocolVersion {
        parent: &'static str,
        child: &'static str,
        parent_version: u32,
        child_version: u32,
    },
    #[error("{type_name}.{field} is required")]
    MissingRequiredField {
        type_name: &'static str,
        field: &'static str,
    },
    #[error("invalid {type_name}: {message}")]
    InvalidEnvelope {
        type_name: &'static str,
        message: String,
    },
    #[error("invalid image blob `{id}`: {message}")]
    InvalidImageBlob { id: String, message: String },
    #[error("invalid attachment reference `{id}`: {message}")]
    InvalidAttachmentRef { id: String, message: String },
    #[error("turn input is not remote-safe: {0}")]
    NonRemoteSafeTurnInput(String),
    #[error("remote tool grant `{tool_name}` is missing an explicit lashlang binding")]
    MissingLashlangToolBinding { tool_name: String },
    #[error("invalid remote tool grant `{tool_name}`: {message}")]
    InvalidToolGrant { tool_name: String, message: String },
    #[error("duplicate remote tool call path `{call_path}`")]
    DuplicateRemoteCallPath { call_path: String },
    #[error(
        "remote tool registry changed across reopen: before={before_call_paths:?}, after={after_call_paths:?}"
    )]
    RemoteToolRegistryReopenMismatch {
        before_call_paths: Vec<String>,
        after_call_paths: Vec<String>,
    },
    #[error("unknown remote tool `{tool_name}`")]
    UnknownRemoteTool { tool_name: String },
    #[error("remote tool transport failed: {0}")]
    RemoteToolTransport(String),
    #[error("failed to serialize remote activity: {0}")]
    ActivitySerialization(#[from] serde_json::Error),
    #[error("failed to write remote activity: {0}")]
    ActivityWrite(String),
}

#[cfg(feature = "core-conversions")]
mod core_conversions;

#[cfg(feature = "core-conversions")]
pub use core_conversions::{
    RemoteToolProvider, RemoteToolTransport, RemoteTurnActivitySink, replay_collected_activities,
};

#[cfg(test)]
mod tests;
