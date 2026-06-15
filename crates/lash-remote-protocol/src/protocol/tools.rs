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
    pub completion_key: serde_json::Value,
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
        if self.completion_key.is_null() {
            return Err(RemoteProtocolError::RemoteToolTransport(
                "remote tool call request requires completion_key".to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RemoteTimeoutBehavior {
    #[default]
    ErrorAsResult,
    FailTurn,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RemoteCancelHint {
    Ignore,
    #[default]
    CancelExternalWork,
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
    Pending {
        protocol_version: u32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        deadline_ms: Option<u64>,
        #[serde(default)]
        on_timeout: RemoteTimeoutBehavior,
        #[serde(default)]
        on_cancel: RemoteCancelHint,
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
            }
            | Self::Pending {
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
