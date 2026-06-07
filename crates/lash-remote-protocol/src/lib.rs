use std::collections::{HashMap, HashSet};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

pub const REMOTE_PROTOCOL_VERSION: u32 = 2;

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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub activity_cursor: Option<String>,
}

impl RemoteLlmRequestMetadata {
    pub fn is_empty(&self) -> bool {
        self.session_id.is_none()
            && self.idempotency_key.is_none()
            && self.trace_id.is_none()
            && self.activity_cursor.is_none()
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub activity_cursor: Option<String>,
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
    AgentFrameSwitch { frame_id: String },
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

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
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

impl Default for RemotePromptLayer {
    fn default() -> Self {
        Self {
            template: None,
            slots: HashMap::new(),
        }
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
    pub agent_surface: Option<RemoteToolAgentSurface>,
    pub executor: RemoteToolExecutor,
}

impl RemoteToolGrant {
    pub fn call_path(&self) -> Result<String, RemoteProtocolError> {
        let surface = self.required_surface()?;
        Ok(format!(
            "{}.{}",
            surface.module_path.join("."),
            surface.operation
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
        self.required_surface()?;
        self.executor.validate(&self.name)?;
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

    fn required_surface(&self) -> Result<&RemoteToolAgentSurface, RemoteProtocolError> {
        let Some(surface) = &self.agent_surface else {
            return Err(RemoteProtocolError::MissingToolSurface {
                tool_name: self.name.clone(),
            });
        };
        if surface.module_path.is_empty() {
            return Err(RemoteProtocolError::InvalidToolGrant {
                tool_name: self.name.clone(),
                message: "remote tool grant requires an explicit module path".to_string(),
            });
        }
        if surface
            .module_path
            .iter()
            .any(|part| part.trim().is_empty())
        {
            return Err(RemoteProtocolError::InvalidToolGrant {
                tool_name: self.name.clone(),
                message: "remote tool grant module path cannot contain empty segments".to_string(),
            });
        }
        if surface.operation.trim().is_empty() {
            return Err(RemoteProtocolError::InvalidToolGrant {
                tool_name: self.name.clone(),
                message: "remote tool grant requires an explicit operation".to_string(),
            });
        }
        Ok(surface)
    }
}

fn default_input_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {},
        "additionalProperties": true
    })
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RemoteToolExecutor {
    Http {
        endpoint: String,
        #[serde(default = "default_http_method")]
        method: String,
        #[serde(default, skip_serializing_if = "HashMap::is_empty")]
        headers: HashMap<String, String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        timeout_ms: Option<u64>,
    },
    CallbackRef {
        ref_id: String,
        #[serde(default, skip_serializing_if = "HashMap::is_empty")]
        metadata: HashMap<String, serde_json::Value>,
    },
    QueueRef {
        queue: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        routing_key: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        timeout_ms: Option<u64>,
    },
}

fn default_http_method() -> String {
    "POST".to_string()
}

impl RemoteToolExecutor {
    pub fn post(endpoint: impl Into<String>) -> Self {
        Self::Http {
            endpoint: endpoint.into(),
            method: default_http_method(),
            headers: HashMap::new(),
            timeout_ms: None,
        }
    }

    pub fn validate(&self, tool_name: &str) -> Result<(), RemoteProtocolError> {
        match self {
            Self::Http {
                endpoint,
                method,
                timeout_ms,
                ..
            } => {
                let endpoint = endpoint.trim();
                if endpoint.is_empty() {
                    return Err(RemoteProtocolError::InvalidToolExecutor {
                        tool_name: tool_name.to_string(),
                        message: "HTTP executor endpoint cannot be empty".to_string(),
                    });
                }
                if !(endpoint.starts_with("http://") || endpoint.starts_with("https://")) {
                    return Err(RemoteProtocolError::InvalidToolExecutor {
                        tool_name: tool_name.to_string(),
                        message: "HTTP executor endpoint must start with http:// or https://"
                            .to_string(),
                    });
                }
                if method.trim().is_empty() {
                    return Err(RemoteProtocolError::InvalidToolExecutor {
                        tool_name: tool_name.to_string(),
                        message: "HTTP executor method cannot be empty".to_string(),
                    });
                }
                validate_timeout(*timeout_ms, tool_name)
            }
            Self::CallbackRef { ref_id, .. } => {
                require_non_empty("RemoteToolExecutor::CallbackRef", "ref_id", ref_id).map_err(
                    |err| RemoteProtocolError::InvalidToolExecutor {
                        tool_name: tool_name.to_string(),
                        message: err.to_string(),
                    },
                )
            }
            Self::QueueRef {
                queue,
                routing_key,
                timeout_ms,
            } => {
                require_non_empty("RemoteToolExecutor::QueueRef", "queue", queue).map_err(
                    |err| RemoteProtocolError::InvalidToolExecutor {
                        tool_name: tool_name.to_string(),
                        message: err.to_string(),
                    },
                )?;
                if routing_key
                    .as_ref()
                    .is_some_and(|key| key.trim().is_empty())
                {
                    return Err(RemoteProtocolError::InvalidToolExecutor {
                        tool_name: tool_name.to_string(),
                        message: "queue executor routing_key cannot be empty when provided"
                            .to_string(),
                    });
                }
                validate_timeout(*timeout_ms, tool_name)
            }
        }
    }
}

fn validate_timeout(timeout_ms: Option<u64>, tool_name: &str) -> Result<(), RemoteProtocolError> {
    if timeout_ms == Some(0) {
        return Err(RemoteProtocolError::InvalidToolExecutor {
            tool_name: tool_name.to_string(),
            message: "executor timeout_ms must be greater than zero".to_string(),
        });
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct RemoteToolAgentSurface {
    pub module_path: Vec<String>,
    pub operation: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub authority_type: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub aliases: Vec<String>,
}

impl RemoteToolAgentSurface {
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
    #[error("remote tool grant `{tool_name}` is missing an explicit agent surface")]
    MissingToolSurface { tool_name: String },
    #[error("invalid remote tool grant `{tool_name}`: {message}")]
    InvalidToolGrant { tool_name: String, message: String },
    #[error("duplicate remote tool call path `{call_path}`")]
    DuplicateRemoteCallPath { call_path: String },
    #[error("invalid executor for remote tool `{tool_name}`: {message}")]
    InvalidToolExecutor { tool_name: String, message: String },
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
mod core_conversions {
    use std::collections::HashMap;
    use std::future::Future;
    use std::io::Write;
    use std::num::NonZeroUsize;
    use std::pin::Pin;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Arc, Mutex};

    use base64::Engine as _;
    use lash_core::llm::types as core_llm;
    use lash_core::{
        ToolCall, ToolContract, ToolDefinition, ToolManifest, ToolProvider, ToolResult,
    };

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
                    .map(|(id, bytes)| {
                        (id, base64::engine::general_purpose::STANDARD.encode(bytes))
                    })
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
                    activity_cursor: None,
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
                lash_core::TurnOutcome::AgentFrameSwitch { frame_id } => {
                    Self::AgentFrameSwitch { frame_id }
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
                lash_core::ToolCallOutcome::Cancelled(value) => {
                    Self::Cancelled(value.to_json_value())
                }
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
                lash_core::TurnEvent::PluginRuntime { plugin_id, event } => {
                    Self::RuntimeDiagnostic {
                        kind: "plugin_runtime".to_string(),
                        data: serde_json::json!({
                            "plugin_id": plugin_id,
                            "event": event,
                        }),
                    }
                }
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
            let mut surface = lash_core::ToolAgentSurface::new(
                value.module_path.clone(),
                value.operation.clone(),
            );
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
                definition = definition.with_availability(lash_core::ToolAvailabilityConfig::same(
                    availability.into(),
                ));
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
            executor: &'a RemoteToolExecutor,
            request: RemoteToolCallRequest,
        ) -> Pin<
            Box<
                dyn Future<Output = Result<RemoteToolCallResponse, RemoteProtocolError>>
                    + Send
                    + 'a,
            >,
        >;
    }

    pub struct RemoteToolProvider<T: RemoteToolTransport> {
        manifests: Vec<ToolManifest>,
        contracts: HashMap<String, Arc<ToolContract>>,
        executors: HashMap<String, (String, RemoteToolExecutor)>,
        transport: T,
    }

    impl<T: RemoteToolTransport> RemoteToolProvider<T> {
        pub fn new(
            grants: Vec<RemoteToolGrant>,
            transport: T,
        ) -> Result<Self, RemoteProtocolError> {
            RemoteToolGrant::validate_all(&grants)?;
            let mut manifests = Vec::with_capacity(grants.len());
            let mut contracts = HashMap::with_capacity(grants.len());
            let mut executors = HashMap::with_capacity(grants.len());
            for grant in grants {
                let definition = ToolDefinition::try_from(&grant)?;
                let manifest = definition.manifest();
                let executable = lash_core::ToolAgentSurface::required_for_remote(&manifest)
                    .map_err(|message| RemoteProtocolError::InvalidToolGrant {
                        tool_name: manifest.name.clone(),
                        message,
                    })?;
                contracts.insert(manifest.name.clone(), Arc::new(definition.contract()));
                executors.insert(
                    manifest.name.clone(),
                    (executable.call_path(), grant.executor.clone()),
                );
                manifests.push(manifest);
            }
            Ok(Self {
                manifests,
                contracts,
                executors,
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
                let Some((call_path, executor)) = self.executors.get(call.name) else {
                    return ToolResult::err_fmt(format_args!(
                        "unknown remote tool `{}`",
                        call.name
                    ));
                };
                let mut headers = match executor {
                    RemoteToolExecutor::Http { headers, .. } => headers.clone(),
                    RemoteToolExecutor::CallbackRef { .. }
                    | RemoteToolExecutor::QueueRef { .. } => HashMap::new(),
                };
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
                match self.transport.send(executor, request).await {
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
    mod tests {
        use std::sync::{Arc, Mutex};

        use super::*;

        struct RecordingTransport {
            requests: Mutex<Vec<RemoteToolCallRequest>>,
            response: RemoteToolCallResponse,
        }

        impl RemoteToolTransport for RecordingTransport {
            fn send<'a>(
                &'a self,
                _executor: &'a RemoteToolExecutor,
                request: RemoteToolCallRequest,
            ) -> Pin<
                Box<
                    dyn Future<Output = Result<RemoteToolCallResponse, RemoteProtocolError>>
                        + Send
                        + 'a,
                >,
            > {
                Box::pin(async move {
                    self.requests.lock().expect("requests lock").push(request);
                    Ok(self.response.clone())
                })
            }
        }

        #[test]
        fn turn_input_round_trips_remote_safe_fields() {
            let mut prompt = lash_core::PromptLayer::new();
            prompt.add_contribution(lash_core::PromptContribution::guidance("Guide", "remote"));
            let mut input = lash_core::TurnInput::items([
                lash_core::InputItem::text("a"),
                lash_core::InputItem::text("b"),
                lash_core::InputItem::image_ref("img"),
            ])
            .with_image_blob("img", vec![1, 2, 3])
            .with_protocol_turn_options(lash_core::ProtocolTurnOptions {
                payload: serde_json::json!({ "mode": "remote" }),
            })
            .with_trace_turn_id("trace-1");
            input.turn_context.set_prompt_layer(prompt.clone());

            let remote = RemoteTurnInput::try_from(input).expect("remote conversion");
            assert_eq!(remote.items.len(), 3);
            assert_eq!(remote.image_blobs_base64["img"], "AQID");
            assert_eq!(remote.trace_turn_id.as_deref(), Some("trace-1"));
            assert_eq!(
                remote.protocol_turn_options.as_ref().unwrap().payload,
                serde_json::json!({ "mode": "remote" })
            );
            assert_eq!(remote.prompt_layer, Some(prompt.clone().into()));

            let core = lash_core::TurnInput::try_from(remote).expect("core conversion");
            assert_eq!(core.image_blobs["img"], vec![1, 2, 3]);
            assert_eq!(core.trace_turn_id.as_deref(), Some("trace-1"));
            assert_eq!(
                core.protocol_turn_options.unwrap().payload,
                serde_json::json!({ "mode": "remote" })
            );
            assert_eq!(core.turn_context.prompt_layer(), &prompt);
        }

        #[test]
        fn turn_input_rejects_non_remote_safe_fields() {
            struct DummyTurnExtension;

            impl lash_core::ProtocolTurnExtension for DummyTurnExtension {
                fn as_any(&self) -> &dyn std::any::Any {
                    self
                }
            }

            let mut input = lash_core::TurnInput::text("extension");
            input.protocol_extension = Some(lash_core::ProtocolTurnExtensionHandle::new(
                DummyTurnExtension,
            ));
            assert!(matches!(
                RemoteTurnInput::try_from(input),
                Err(RemoteProtocolError::NonRemoteSafeTurnInput(message))
                    if message.contains("protocol turn")
            ));

            let mut input = lash_core::TurnInput::text("live");
            input.turn_context.insert_plugin_input("demo", 1_u32);
            assert!(matches!(
                RemoteTurnInput::try_from(input),
                Err(RemoteProtocolError::NonRemoteSafeTurnInput(message))
                    if message.contains("live plugin")
            ));

            let mut input = lash_core::TurnInput::text("provider");
            input.turn_context.set_provider(
                lash_core::testing::TestProvider::builder()
                    .build()
                    .into_handle(),
            );
            assert!(matches!(
                RemoteTurnInput::try_from(input),
                Err(RemoteProtocolError::NonRemoteSafeTurnInput(message))
                    if message.contains("provider")
            ));

            let mut input = lash_core::TurnInput::text("model");
            input.turn_context.set_model(
                lash_core::ModelSpec::from_token_limits("m", None, 100, None, None).expect("model"),
            );
            assert!(matches!(
                RemoteTurnInput::try_from(input),
                Err(RemoteProtocolError::NonRemoteSafeTurnInput(message))
                    if message.contains("model")
            ));
        }

        #[test]
        fn llm_request_and_response_round_trip_owned_dtos() {
            let request = core_llm::LlmRequest {
                model: "gpt-test".to_string(),
                messages: vec![core_llm::LlmMessage::text(core_llm::LlmRole::User, "hello")],
                attachments: vec![core_llm::LlmAttachment::bytes("image/png", vec![1, 2, 3])],
                tools: Arc::new(vec![core_llm::LlmToolSpec {
                    name: "search".to_string(),
                    description: "Search".to_string(),
                    input_schema: serde_json::json!({"type": "object"}),
                    output_schema: serde_json::Value::Null,
                    input_schema_projections: Vec::new(),
                    output_schema_projections: Vec::new(),
                }]),
                tool_choice: core_llm::LlmToolChoice::Auto,
                model_variant: Some("fast".to_string()),
                generation: core_llm::GenerationOptions {
                    output_token_cap: NonZeroUsize::new(42),
                },
                session_id: Some("session-1".to_string()),
                output_spec: Some(core_llm::LlmOutputSpec::JsonObject),
                stream_events: None,
                provider_trace: None,
            };

            let remote = RemoteLlmRequest::from_core("request-1", request);
            remote.validate().expect("valid remote request");
            assert_eq!(remote.protocol_version, REMOTE_PROTOCOL_VERSION);
            assert_eq!(remote.request_id, "request-1");
            let core = core_llm::LlmRequest::try_from(remote).expect("core request");
            assert_eq!(core.model, "gpt-test");
            assert_eq!(core.model_variant.as_deref(), Some("fast"));
            assert_eq!(core.attachments[0].data, vec![1, 2, 3]);

            let response = core_llm::LlmResponse {
                full_text: "done".to_string(),
                parts: vec![core_llm::LlmOutputPart::Text {
                    text: "done".to_string(),
                    response_meta: None,
                }],
                usage: core_llm::LlmUsage {
                    input_tokens: 1,
                    output_tokens: 2,
                    cached_input_tokens: 0,
                    reasoning_tokens: 0,
                },
                terminal_reason: core_llm::LlmTerminalReason::Stop,
                terminal_diagnostic: Some("ok".to_string()),
                provider_usage: Some(serde_json::json!({"provider": "usage"})),
                request_body: Some("{}".to_string()),
                http_summary: Some("200".to_string()),
            };
            let remote = RemoteLlmResponse::from_core("request-1", response);
            remote.validate().expect("valid remote response");
            let core = core_llm::LlmResponse::from(remote);
            assert_eq!(core.full_text, "done");
            assert_eq!(core.terminal_reason, core_llm::LlmTerminalReason::Stop);
            assert_eq!(
                core.provider_usage,
                Some(serde_json::json!({"provider": "usage"}))
            );
        }

        #[test]
        fn prompt_layer_round_trips_without_protocol_crate_depending_on_core_by_default() {
            let template =
                lash_core::PromptTemplate::new(vec![lash_core::PromptTemplateSection::titled(
                    "Custom",
                    vec![lash_core::PromptTemplateEntry::slot(
                        lash_core::PromptSlot::Guidance,
                    )],
                )]);
            let prompt = lash_core::PromptLayer::with_template(template)
                .with_contribution(lash_core::PromptContribution::guidance("Guide", "remote"));

            let remote = RemotePromptLayer::from(prompt.clone());
            let core = lash_core::PromptLayer::from(remote);
            assert_eq!(core, prompt);
        }

        #[test]
        fn remote_turn_result_maps_core_semantics() {
            let turn = lash_core::AssembledTurn {
                state: Default::default(),
                outcome: lash_core::TurnOutcome::Finished(
                    lash_core::TurnFinish::AssistantMessage {
                        text: "done".to_string(),
                    },
                ),
                assistant_output: lash_core::AssistantOutput {
                    safe_text: "done".to_string(),
                    raw_text: "done".to_string(),
                    state: lash_core::OutputState::Usable,
                },
                execution: lash_core::ExecutionSummary {
                    had_tool_calls: true,
                    had_code_execution: false,
                },
                token_usage: lash_core::TokenUsage {
                    input_tokens: 1,
                    output_tokens: 2,
                    cached_input_tokens: 0,
                    reasoning_tokens: 0,
                },
                children_usage: vec![lash_core::TokenLedgerEntry {
                    source: "subagent".to_string(),
                    model: "m".to_string(),
                    usage: lash_core::TokenUsage {
                        input_tokens: 3,
                        output_tokens: 4,
                        cached_input_tokens: 0,
                        reasoning_tokens: 0,
                    },
                }],
                tool_calls: Vec::new(),
                errors: Vec::new(),
            };

            let remote = RemoteTurnResult::from_core("session", "turn", turn, []);
            remote.validate().expect("valid turn result");
            assert_eq!(remote.status, RemoteTurnStatus::Completed);
            assert_eq!(remote.usage.total.input_tokens, 4);
            assert_eq!(remote.usage.total.output_tokens, 6);
        }

        #[test]
        fn remote_tool_grants_validate_explicit_surfaces_and_duplicates() {
            let grant = demo_grant("one", "tools", "search");
            grant.validate().expect("valid grant");
            assert_eq!(grant.call_path().unwrap(), "tools.search");

            let mut missing_surface = grant.clone();
            missing_surface.agent_surface = None;
            assert!(matches!(
                missing_surface.validate(),
                Err(RemoteProtocolError::MissingToolSurface { .. })
            ));

            let duplicate = demo_grant("two", "tools", "search");
            assert!(matches!(
                RemoteToolGrant::validate_all(&[grant, duplicate]),
                Err(RemoteProtocolError::DuplicateRemoteCallPath { .. })
            ));
        }

        #[tokio::test]
        async fn remote_tool_provider_forwards_idempotency_headers_and_failures() {
            let transport = RecordingTransport {
                requests: Mutex::new(Vec::new()),
                response: RemoteToolCallResponse::Failure {
                    protocol_version: REMOTE_PROTOCOL_VERSION,
                    code: "failed".to_string(),
                    message: "nope".to_string(),
                    raw: Some(serde_json::json!({ "detail": true })),
                    retry_after_ms: Some(5),
                },
            };
            let provider =
                RemoteToolProvider::new(vec![demo_grant("demo", "tools", "run")], transport)
                    .expect("provider");
            let host = Arc::new(lash_core::testing::MockSessionManager::default());
            let sessions: Arc<dyn lash_core::plugin::SessionStateService> = host.clone();
            let session_lifecycle: Arc<dyn lash_core::plugin::SessionLifecycleService> =
                host.clone();
            let session_graph: Arc<dyn lash_core::plugin::SessionGraphService> = host;
            let context = lash_core::ToolContext::__for_testing(
                "session-1".to_string(),
                sessions,
                session_lifecycle,
                session_graph,
                Arc::new(lash_core::UnavailableProcessService),
                Arc::new(lash_core::InMemoryAttachmentStore::new()),
                lash_core::DirectCompletionClient::from_fn(|_, _| {
                    Err(lash_core::PluginError::Session("unavailable".to_string()))
                }),
                Some("call-1".to_string()),
            );
            let result = provider
                .execute(lash_core::ToolCall {
                    name: "demo",
                    args: &serde_json::json!({ "x": 1 }),
                    context: &context,
                    progress: None,
                })
                .await;
            assert!(!result.is_success());
            let request = provider
                .transport
                .requests
                .lock()
                .expect("requests lock")
                .pop()
                .expect("request");
            assert_eq!(request.headers["x-lash-tool-call-id"], "call-1");
            assert_eq!(
                request.headers["x-lash-replay-key"],
                "lash-tool:session-1:call-1:demo"
            );
            assert_eq!(request.call_path, "tools.run");
        }

        #[test]
        fn remote_activity_preserves_semantic_fields_and_collapses_runtime_diagnostics() {
            let output = lash_core::ToolCallOutput::success(serde_json::json!({ "ok": true }));
            let activity = lash_core::TurnActivity::new(
                lash_core::TurnActivityId::new("corr"),
                lash_core::TurnEvent::ToolCallCompleted {
                    call_id: Some("call".to_string()),
                    name: "demo".to_string(),
                    args: serde_json::json!({ "a": 1 }),
                    output,
                    duration_ms: 42,
                },
            );
            let remote = RemoteTurnActivity::from_core(9, activity);
            assert_eq!(remote.sequence, 9);
            match remote.event {
                RemoteTurnEvent::ToolCallCompleted {
                    call_id,
                    args,
                    duration_ms,
                    ..
                } => {
                    assert_eq!(call_id.as_deref(), Some("call"));
                    assert_eq!(args, serde_json::json!({ "a": 1 }));
                    assert_eq!(duration_ms, 42);
                }
                other => panic!("unexpected event: {other:?}"),
            }
        }

        fn demo_grant(name: &str, module: &str, operation: &str) -> RemoteToolGrant {
            RemoteToolGrant {
                protocol_version: REMOTE_PROTOCOL_VERSION,
                id: None,
                name: name.to_string(),
                description: "demo".to_string(),
                input_schema: default_input_schema(),
                output_schema: serde_json::Value::Null,
                input_schema_projections: Vec::new(),
                output_schema_projections: Vec::new(),
                output_contract: RemoteToolOutputContract::Static,
                examples: Vec::new(),
                availability: None,
                activation: None,
                argument_projection: None,
                scheduling: None,
                retry_policy: None,
                agent_surface: Some(RemoteToolAgentSurface::new([module], operation)),
                executor: RemoteToolExecutor::post("https://example.com/tool"),
            }
        }
    }
}

#[cfg(feature = "core-conversions")]
pub use core_conversions::{
    RemoteToolProvider, RemoteToolTransport, RemoteTurnActivitySink, replay_collected_activities,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone)]
    struct VecRegistry(Vec<RemoteToolGrant>);

    impl RemoteToolRegistry for VecRegistry {
        fn grants(&self) -> Vec<RemoteToolGrant> {
            self.0.clone()
        }
    }

    #[test]
    fn remote_llm_request_json_round_trips() {
        let request = RemoteLlmRequest {
            protocol_version: REMOTE_PROTOCOL_VERSION,
            request_id: "request-1".to_string(),
            model_intent: RemoteModelIntent::new("gpt-test"),
            messages: vec![RemoteLlmMessage {
                role: RemoteLlmRole::User,
                content: vec![RemoteLlmContentBlock::Text {
                    text: "hello".to_string(),
                    response_meta: None,
                    cache_breakpoint: false,
                }],
            }],
            attachments: vec![RemoteLlmAttachment {
                id: Some("img".to_string()),
                mime: "image/png".to_string(),
                data_base64: Some("AQID".to_string()),
                reference: None,
                metadata: HashMap::new(),
            }],
            tools: Vec::new(),
            tool_choice: RemoteLlmToolChoice::Auto,
            output_spec: Some(RemoteLlmOutputSpec::JsonObject),
            generation: RemoteGenerationOptions {
                output_token_cap: Some(128),
                ..Default::default()
            },
            request_metadata: RemoteLlmRequestMetadata {
                session_id: Some("session".to_string()),
                idempotency_key: Some("idem".to_string()),
                trace_id: None,
                activity_cursor: None,
            },
            metadata: HashMap::new(),
        };

        request.validate().expect("valid request");
        let value = serde_json::to_value(&request).expect("serialize");
        let decoded: RemoteLlmRequest = serde_json::from_value(value).expect("deserialize");
        assert_eq!(decoded.protocol_version, 2);
        assert_eq!(decoded.request_id, request.request_id);
        assert_eq!(decoded.messages, request.messages);
    }

    #[test]
    fn remote_llm_response_json_round_trips() {
        let response = RemoteLlmResponse {
            protocol_version: REMOTE_PROTOCOL_VERSION,
            request_id: "request-1".to_string(),
            full_text: "done".to_string(),
            output_parts: vec![RemoteLlmOutputPart::Text {
                text: "done".to_string(),
                response_meta: None,
            }],
            usage: RemoteUsage {
                input_tokens: 1,
                output_tokens: 2,
                cached_input_tokens: 0,
                reasoning_tokens: 0,
            },
            terminal_reason: RemoteLlmTerminalReason::Stop,
            diagnostics: Vec::new(),
            provider_metadata: RemoteProviderMetadata::default(),
        };

        response.validate().expect("valid response");
        let value = serde_json::to_value(&response).expect("serialize");
        let decoded: RemoteLlmResponse = serde_json::from_value(value).expect("deserialize");
        assert_eq!(decoded.protocol_version, 2);
        assert_eq!(decoded.full_text, "done");
    }

    #[test]
    fn remote_turn_request_json_round_trips() {
        let request = RemoteTurnRequest {
            protocol_version: REMOTE_PROTOCOL_VERSION,
            session_id: "session".to_string(),
            turn_id: "turn".to_string(),
            idempotency_key: Some("idem".to_string()),
            input: RemoteTurnInput {
                protocol_version: REMOTE_PROTOCOL_VERSION,
                items: vec![
                    RemoteInputItem::Text {
                        text: "first".to_string(),
                    },
                    RemoteInputItem::ImageRef {
                        id: "img".to_string(),
                    },
                ],
                image_blobs_base64: HashMap::from([("img".to_string(), "AQID".to_string())]),
                protocol_turn_options: Some(RemoteProtocolTurnOptions {
                    payload: serde_json::json!({ "answer": "raw" }),
                }),
                trace_turn_id: Some("trace".to_string()),
                prompt_layer: Some(RemotePromptLayer::new()),
            },
            tool_grants: vec![demo_grant("demo", "tools", "search")],
            model_intent: Some(RemoteModelIntent::new("gpt-test")),
            activity_cursor: Some("cursor".to_string()),
            metadata: HashMap::new(),
        };

        request.validate().expect("valid request");
        let value = serde_json::to_value(&request).expect("serialize");
        let decoded: RemoteTurnRequest = serde_json::from_value(value).expect("deserialize");

        assert_eq!(decoded.protocol_version, 2);
        assert_eq!(decoded.session_id, "session");
        assert_eq!(decoded.input.image_blobs_base64["img"], "AQID");
        assert_eq!(decoded.tool_grants.len(), 1);
    }

    #[test]
    fn remote_turn_result_json_round_trips() {
        let result = RemoteTurnResult {
            protocol_version: REMOTE_PROTOCOL_VERSION,
            session_id: "session".to_string(),
            turn_id: "turn".to_string(),
            status: RemoteTurnStatus::Completed,
            outcome: RemoteTurnOutcome::Finished {
                finish: RemoteTurnFinish::AssistantMessage {
                    text: "done".to_string(),
                },
            },
            assistant_output: RemoteAssistantOutput {
                safe_text: "done".to_string(),
                raw_text: "done".to_string(),
                state: RemoteAssistantOutputState::Usable,
            },
            usage: RemoteTurnUsageSummary::default(),
            execution: RemoteExecutionSummary::default(),
            tool_calls: vec![RemoteToolCallSummary {
                call_id: Some("call".to_string()),
                tool_name: "demo".to_string(),
                args: serde_json::json!({"x": 1}),
                outcome: RemoteToolCallOutcome::Success(serde_json::json!({"ok": true})),
                duration_ms: 5,
            }],
            issues: Vec::new(),
            activities: vec![RemoteTurnActivity {
                protocol_version: REMOTE_PROTOCOL_VERSION,
                sequence: 1,
                id: "event".to_string(),
                correlation_id: "corr".to_string(),
                event: RemoteTurnEvent::AssistantProseDelta {
                    text: "done".to_string(),
                },
            }],
            metadata: HashMap::new(),
        };

        result.validate().expect("valid result");
        let value = serde_json::to_value(&result).expect("serialize");
        let decoded: RemoteTurnResult = serde_json::from_value(value).expect("deserialize");
        assert_eq!(decoded.protocol_version, 2);
        assert_eq!(decoded.session_id, "session");
        assert_eq!(decoded.tool_calls.len(), 1);
    }

    #[test]
    fn invalid_executor_shapes_are_rejected() {
        let executor = RemoteToolExecutor::post("");
        let err = executor.validate("demo").expect_err("invalid endpoint");
        assert!(matches!(
            err,
            RemoteProtocolError::InvalidToolExecutor { .. }
        ));

        let executor = RemoteToolExecutor::CallbackRef {
            ref_id: String::new(),
            metadata: HashMap::new(),
        };
        assert!(matches!(
            executor.validate("demo"),
            Err(RemoteProtocolError::InvalidToolExecutor { .. })
        ));

        let executor = RemoteToolExecutor::QueueRef {
            queue: "jobs".to_string(),
            routing_key: Some(String::new()),
            timeout_ms: None,
        };
        assert!(matches!(
            executor.validate("demo"),
            Err(RemoteProtocolError::InvalidToolExecutor { .. })
        ));
    }

    #[test]
    fn all_executor_variants_validate_when_shaped() {
        RemoteToolExecutor::post("https://example.com/tool")
            .validate("demo")
            .expect("http");
        RemoteToolExecutor::CallbackRef {
            ref_id: "cb".to_string(),
            metadata: HashMap::new(),
        }
        .validate("demo")
        .expect("callback");
        RemoteToolExecutor::QueueRef {
            queue: "jobs".to_string(),
            routing_key: Some("tools".to_string()),
            timeout_ms: Some(1000),
        }
        .validate("demo")
        .expect("queue");
    }

    #[test]
    fn wrong_protocol_versions_are_rejected() {
        let mut input = RemoteTurnInput::text("hello");
        input.protocol_version = REMOTE_PROTOCOL_VERSION + 1;
        assert!(matches!(
            input.validate(),
            Err(RemoteProtocolError::UnsupportedProtocolVersion { .. })
        ));

        let mut grant = demo_grant("one", "tools", "search");
        grant.protocol_version = REMOTE_PROTOCOL_VERSION + 1;
        assert!(matches!(
            grant.validate(),
            Err(RemoteProtocolError::UnsupportedProtocolVersion { .. })
        ));

        let request = RemoteToolCallRequest {
            protocol_version: REMOTE_PROTOCOL_VERSION + 1,
            tool_name: "demo".to_string(),
            call_path: "tools.demo".to_string(),
            args: serde_json::Value::Null,
            session_id: "session".to_string(),
            tool_call_id: None,
            replay_key: None,
            attempt_number: 1,
            max_attempts: 1,
            headers: HashMap::new(),
        };
        assert!(matches!(
            request.validate(),
            Err(RemoteProtocolError::UnsupportedProtocolVersion { .. })
        ));

        let response = RemoteToolCallResponse::Success {
            protocol_version: REMOTE_PROTOCOL_VERSION + 1,
            value: serde_json::Value::Null,
        };
        assert!(matches!(
            response.validate(),
            Err(RemoteProtocolError::UnsupportedProtocolVersion { .. })
        ));

        let activity = RemoteTurnActivity {
            protocol_version: REMOTE_PROTOCOL_VERSION + 1,
            sequence: 1,
            id: "event".to_string(),
            correlation_id: "corr".to_string(),
            event: RemoteTurnEvent::AssistantProseDelta {
                text: "hi".to_string(),
            },
        };
        assert!(matches!(
            activity.validate(),
            Err(RemoteProtocolError::UnsupportedProtocolVersion { .. })
        ));
    }

    #[test]
    fn nested_protocol_versions_must_match_envelope() {
        let mut request = RemoteTurnRequest {
            protocol_version: REMOTE_PROTOCOL_VERSION,
            session_id: "session".to_string(),
            turn_id: "turn".to_string(),
            idempotency_key: None,
            input: RemoteTurnInput::text("hello"),
            tool_grants: Vec::new(),
            model_intent: None,
            activity_cursor: None,
            metadata: HashMap::new(),
        };
        request.input.protocol_version = REMOTE_PROTOCOL_VERSION + 1;
        assert!(matches!(
            request.validate(),
            Err(RemoteProtocolError::MismatchedNestedProtocolVersion { .. })
        ));
    }

    #[test]
    fn top_level_protocol_schema_exports_include_versions() {
        assert_schema_has_protocol_version::<RemoteLlmRequest>();
        assert_schema_has_protocol_version::<RemoteLlmResponse>();
        assert_schema_has_protocol_version::<RemoteTurnInput>();
        assert_schema_has_protocol_version::<RemoteTurnRequest>();
        assert_schema_has_protocol_version::<RemoteTurnResult>();
        assert_schema_has_protocol_version::<RemoteToolGrant>();
        assert_schema_has_protocol_version::<RemoteToolCallRequest>();
        assert_schema_has_protocol_version::<RemoteToolCallResponse>();
        assert_schema_has_protocol_version::<RemoteTurnActivity>();
    }

    #[test]
    fn remote_tool_registry_reopen_conformance_compares_call_paths() {
        let before = VecRegistry(vec![demo_grant("one", "tools", "search")]);
        let reopened = VecRegistry(vec![demo_grant("one", "tools", "search")]);
        assert_remote_tool_registry_reopenable(&before, &reopened).expect("same registry");

        let changed = VecRegistry(vec![demo_grant("one", "tools", "read")]);
        assert!(matches!(
            assert_remote_tool_registry_reopenable(&before, &changed),
            Err(RemoteProtocolError::RemoteToolRegistryReopenMismatch { .. })
        ));
    }

    fn demo_grant(name: &str, module: &str, operation: &str) -> RemoteToolGrant {
        RemoteToolGrant {
            protocol_version: REMOTE_PROTOCOL_VERSION,
            id: None,
            name: name.to_string(),
            description: "demo".to_string(),
            input_schema: default_input_schema(),
            output_schema: serde_json::Value::Null,
            input_schema_projections: Vec::new(),
            output_schema_projections: Vec::new(),
            output_contract: RemoteToolOutputContract::Static,
            examples: Vec::new(),
            availability: None,
            activation: None,
            argument_projection: None,
            scheduling: None,
            retry_policy: None,
            agent_surface: Some(RemoteToolAgentSurface::new([module], operation)),
            executor: RemoteToolExecutor::post("https://example.com/tool"),
        }
    }

    fn assert_schema_has_protocol_version<T: JsonSchema>() {
        let schema = schemars::schema_for!(T);
        let schema_json = serde_json::to_value(&schema).expect("schema json");
        let schema_text = schema_json.to_string();
        assert!(
            schema_text.contains("protocol_version"),
            "schema did not include protocol_version: {schema_text}"
        );
    }
}
