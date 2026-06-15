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
