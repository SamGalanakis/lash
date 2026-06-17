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
    pub target: Option<RemoteProcessDefinitionIdentity>,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    pub input: RemoteProcessInput,
    #[serde(default)]
    pub inputs: RemoteTriggerInputTemplate,
}

fn default_true() -> bool {
    true
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(transparent)]
pub struct RemoteTriggerInputTemplate {
    pub entries: BTreeMap<String, RemoteTriggerInputBinding>,
}

impl RemoteTriggerInputTemplate {
    pub fn new(entries: BTreeMap<String, RemoteTriggerInputBinding>) -> Self {
        Self { entries }
    }

    pub fn validate(&self, type_name: &'static str) -> Result<(), RemoteProtocolError> {
        for name in self.entries.keys() {
            require_non_empty(type_name, "input_template key", name)?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RemoteTriggerInputBinding {
    Event,
    Fixed { value: serde_json::Value },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RemoteTriggerSubscriptionDraft {
    pub protocol_version: u32,
    pub registrant: RemoteProcessOriginator,
    pub env_ref: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wake_target: Option<RemoteSessionScope>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub source_type: String,
    pub source_key: String,
    #[serde(default)]
    pub source: serde_json::Value,
    #[serde(default)]
    pub payload_schema: serde_json::Value,
    pub target: RemoteProcessInput,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub event_types: Vec<RemoteProcessEventType>,
    #[serde(default)]
    pub input_template: RemoteTriggerInputTemplate,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_label: Option<String>,
}

impl RemoteTriggerSubscriptionDraft {
    pub fn validate(&self) -> Result<(), RemoteProtocolError> {
        ensure_protocol_version(self.protocol_version)?;
        self.registrant.validate("RemoteTriggerSubscriptionDraft")?;
        require_non_empty("RemoteTriggerSubscriptionDraft", "env_ref", &self.env_ref)?;
        if let Some(wake_target) = &self.wake_target {
            wake_target.validate("RemoteTriggerSubscriptionDraft")?;
        }
        require_non_empty(
            "RemoteTriggerSubscriptionDraft",
            "source_type",
            &self.source_type,
        )?;
        require_non_empty(
            "RemoteTriggerSubscriptionDraft",
            "source_key",
            &self.source_key,
        )?;
        self.target.validate("RemoteTriggerSubscriptionDraft")?;
        for event_type in &self.event_types {
            event_type.validate("RemoteTriggerSubscriptionDraft")?;
        }
        self.input_template
            .validate("RemoteTriggerSubscriptionDraft")
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RemoteTriggerSubscriptionRecord {
    pub subscription_id: String,
    pub registrant: RemoteProcessOriginator,
    pub env_ref: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wake_target: Option<RemoteSessionScope>,
    pub handle: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub source_type: String,
    pub source_key: String,
    #[serde(default)]
    pub source: serde_json::Value,
    #[serde(default)]
    pub payload_schema: serde_json::Value,
    pub target: RemoteProcessInput,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub event_types: Vec<RemoteProcessEventType>,
    #[serde(default)]
    pub input_template: RemoteTriggerInputTemplate,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_label: Option<String>,
    #[serde(default = "default_true")]
    pub enabled: bool,
    pub created_at_ms: u64,
    pub updated_at_ms: u64,
}

impl RemoteTriggerSubscriptionRecord {
    pub fn validate(&self, type_name: &'static str) -> Result<(), RemoteProtocolError> {
        require_non_empty(type_name, "subscription_id", &self.subscription_id)?;
        self.registrant.validate(type_name)?;
        require_non_empty(type_name, "env_ref", &self.env_ref)?;
        if let Some(wake_target) = &self.wake_target {
            wake_target.validate(type_name)?;
        }
        require_non_empty(type_name, "handle", &self.handle)?;
        require_non_empty(type_name, "source_type", &self.source_type)?;
        require_non_empty(type_name, "source_key", &self.source_key)?;
        self.target.validate(type_name)?;
        for event_type in &self.event_types {
            event_type.validate(type_name)?;
        }
        self.input_template.validate(type_name)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RemoteTriggerRegisterSubscriptionRequest {
    pub protocol_version: u32,
    pub draft: RemoteTriggerSubscriptionDraft,
}

impl RemoteTriggerRegisterSubscriptionRequest {
    pub fn validate(&self) -> Result<(), RemoteProtocolError> {
        ensure_protocol_version(self.protocol_version)?;
        if self.draft.protocol_version != self.protocol_version {
            return Err(RemoteProtocolError::MismatchedNestedProtocolVersion {
                parent: "RemoteTriggerRegisterSubscriptionRequest",
                child: "draft",
                parent_version: self.protocol_version,
                child_version: self.draft.protocol_version,
            });
        }
        self.draft.validate()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RemoteTriggerRegisterSubscriptionResult {
    pub protocol_version: u32,
    pub record: RemoteTriggerSubscriptionRecord,
}

impl RemoteTriggerRegisterSubscriptionResult {
    pub fn validate(&self) -> Result<(), RemoteProtocolError> {
        ensure_protocol_version(self.protocol_version)?;
        self.record
            .validate("RemoteTriggerRegisterSubscriptionResult")
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RemoteTriggerListSubscriptionsResponse {
    pub protocol_version: u32,
    #[serde(default)]
    pub subscriptions: Vec<RemoteTriggerSubscriptionRecord>,
}

impl RemoteTriggerListSubscriptionsResponse {
    pub fn validate(&self) -> Result<(), RemoteProtocolError> {
        ensure_protocol_version(self.protocol_version)?;
        for record in &self.subscriptions {
            record.validate("RemoteTriggerListSubscriptionsResponse")?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct RemoteTriggerCancelSubscriptionRequest {
    pub protocol_version: u32,
    pub session_id: String,
    pub handle: String,
}

impl RemoteTriggerCancelSubscriptionRequest {
    pub fn validate(&self) -> Result<(), RemoteProtocolError> {
        ensure_protocol_version(self.protocol_version)?;
        require_non_empty(
            "RemoteTriggerCancelSubscriptionRequest",
            "session_id",
            &self.session_id,
        )?;
        require_non_empty(
            "RemoteTriggerCancelSubscriptionRequest",
            "handle",
            &self.handle,
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct RemoteTriggerCancelSubscriptionResult {
    pub protocol_version: u32,
    pub session_id: String,
    pub handle: String,
    pub cancelled: bool,
}

impl RemoteTriggerCancelSubscriptionResult {
    pub fn validate(&self) -> Result<(), RemoteProtocolError> {
        ensure_protocol_version(self.protocol_version)?;
        require_non_empty(
            "RemoteTriggerCancelSubscriptionResult",
            "session_id",
            &self.session_id,
        )?;
        require_non_empty(
            "RemoteTriggerCancelSubscriptionResult",
            "handle",
            &self.handle,
        )
    }
}
