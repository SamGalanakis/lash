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
