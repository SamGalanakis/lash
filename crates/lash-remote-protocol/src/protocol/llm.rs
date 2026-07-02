#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct RemoteSchemaContract {
    pub canonical: serde_json::Value,
    #[serde(
        default,
        skip_serializing_if = "RemoteSchemaProjectionPolicy::is_default"
    )]
    pub projection: RemoteSchemaProjectionPolicy,
}

impl RemoteSchemaContract {
    fn new(canonical: serde_json::Value) -> Self {
        Self {
            canonical,
            projection: RemoteSchemaProjectionPolicy::default(),
        }
    }
}

impl Default for RemoteSchemaContract {
    fn default() -> Self {
        Self::new(serde_json::Value::Null)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct RemoteSchemaProjectionPolicy {
    #[serde(default, skip_serializing_if = "RemoteProjectionMode::is_auto")]
    pub mode: RemoteProjectionMode,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub overrides: Vec<RemoteSchemaProjectionOverride>,
}

impl RemoteSchemaProjectionPolicy {
    fn is_default(&self) -> bool {
        self.mode == RemoteProjectionMode::Auto && self.overrides.is_empty()
    }
}

impl Default for RemoteSchemaProjectionPolicy {
    fn default() -> Self {
        Self {
            mode: RemoteProjectionMode::Auto,
            overrides: Vec::new(),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RemoteProjectionMode {
    #[default]
    Auto,
    ExplicitOnly,
    Exact,
}

impl RemoteProjectionMode {
    fn is_auto(&self) -> bool {
        *self == Self::Auto
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct RemoteSchemaProjectionOverride {
    pub dialect: String,
    pub schema: serde_json::Value,
}

fn default_remote_input_schema() -> RemoteSchemaContract {
    RemoteSchemaContract::new(serde_json::json!({
        "type": "object",
        "properties": {},
        "additionalProperties": true
    }))
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RemoteLlmRequest {
    pub protocol_version: u32,
    pub request_id: String,
    pub scope: RemoteLlmRequestScope,
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
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, serde_json::Value>,
}

impl RemoteLlmRequest {
    pub fn validate(&self) -> Result<(), RemoteProtocolError> {
        ensure_protocol_version(self.protocol_version)?;
        require_non_empty("RemoteLlmRequest", "request_id", &self.request_id)?;
        self.scope.validate()?;
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

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct RemoteLlmRequestScope {
    pub session_id: String,
    pub agent_frame_id: String,
    pub request_id: String,
}

impl RemoteLlmRequestScope {
    pub fn new(
        session_id: impl Into<String>,
        agent_frame_id: impl Into<String>,
        request_id: impl Into<String>,
    ) -> Self {
        Self {
            session_id: session_id.into(),
            agent_frame_id: agent_frame_id.into(),
            request_id: request_id.into(),
        }
    }

    fn validate(&self) -> Result<(), RemoteProtocolError> {
        require_non_empty("RemoteLlmRequestScope", "session_id", &self.session_id)?;
        require_non_empty(
            "RemoteLlmRequestScope",
            "agent_frame_id",
            &self.agent_frame_id,
        )?;
        require_non_empty("RemoteLlmRequestScope", "request_id", &self.request_id)?;
        Ok(())
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_payload: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub origin_provider: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub origin_model: Option<String>,
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
    #[serde(default = "default_remote_input_schema")]
    pub input_schema: RemoteSchemaContract,
    #[serde(default)]
    pub output_schema: RemoteSchemaContract,
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
        schema: RemoteSchemaContract,
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

/// Wire mirror of the core `ProviderFailureKind` classification.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RemoteProviderFailureKind {
    Transport,
    Timeout,
    Http,
    Stream,
    Auth,
    Validation,
    Quota,
    Unsupported,
    #[default]
    #[serde(other)]
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
