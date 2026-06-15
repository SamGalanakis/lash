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
