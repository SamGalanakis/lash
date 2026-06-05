use std::collections::{HashMap, HashSet};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

pub const REMOTE_PROTOCOL_VERSION: u32 = 1;

pub type RemoteLlmRequest = lash_sansio::llm::types::LlmRequest;
pub type RemoteLlmResponse = lash_sansio::llm::types::LlmResponse;

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
    #[schemars(with = "Option<serde_json::Value>")]
    pub prompt_layer: Option<lash_sansio::PromptLayer>,
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
        ensure_protocol_version(self.protocol_version)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RemoteInputItem {
    Text { text: String },
    ImageRef { id: String },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
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
    pub http_executor: RemoteHttpToolExecutor,
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
        self.http_executor.validate(&self.name)?;
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

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct RemoteHttpToolExecutor {
    pub endpoint: String,
    #[serde(default = "default_http_method")]
    pub method: String,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub headers: HashMap<String, String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_ms: Option<u64>,
}

fn default_http_method() -> String {
    "POST".to_string()
}

impl RemoteHttpToolExecutor {
    pub fn post(endpoint: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
            method: default_http_method(),
            headers: HashMap::new(),
            timeout_ms: None,
        }
    }

    pub fn validate(&self, tool_name: &str) -> Result<(), RemoteProtocolError> {
        let endpoint = self.endpoint.trim();
        if endpoint.is_empty() {
            return Err(RemoteProtocolError::InvalidHttpExecutor {
                tool_name: tool_name.to_string(),
                message: "HTTP executor endpoint cannot be empty".to_string(),
            });
        }
        if !(endpoint.starts_with("http://") || endpoint.starts_with("https://")) {
            return Err(RemoteProtocolError::InvalidHttpExecutor {
                tool_name: tool_name.to_string(),
                message: "HTTP executor endpoint must start with http:// or https://".to_string(),
            });
        }
        if self.method.trim().is_empty() {
            return Err(RemoteProtocolError::InvalidHttpExecutor {
                tool_name: tool_name.to_string(),
                message: "HTTP executor method cannot be empty".to_string(),
            });
        }
        Ok(())
    }
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

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct RemoteUsage {
    pub input_tokens: i64,
    pub output_tokens: i64,
    pub cached_input_tokens: i64,
    #[serde(default)]
    pub reasoning_tokens: i64,
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
        ensure_protocol_version(self.protocol_version)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RemoteTurnEvent {
    QueuedWorkStarted {
        boundary: serde_json::Value,
        batch_ids: Vec<String>,
        causes: serde_json::Value,
    },
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
    PluginRuntime {
        plugin_id: String,
        event: serde_json::Value,
    },
    QueuedInputAccepted {
        checkpoint: serde_json::Value,
        inputs: serde_json::Value,
    },
    QueuedMessagesCommitted {
        messages: serde_json::Value,
        checkpoint: serde_json::Value,
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

#[derive(Debug, thiserror::Error)]
pub enum RemoteProtocolError {
    #[error("unsupported remote protocol version {actual}; expected {expected}")]
    UnsupportedProtocolVersion { actual: u32, expected: u32 },
    #[error("invalid image blob `{id}`: {message}")]
    InvalidImageBlob { id: String, message: String },
    #[error("turn input is not remote-safe: {0}")]
    NonRemoteSafeTurnInput(String),
    #[error("remote tool grant `{tool_name}` is missing an explicit agent surface")]
    MissingToolSurface { tool_name: String },
    #[error("invalid remote tool grant `{tool_name}`: {message}")]
    InvalidToolGrant { tool_name: String, message: String },
    #[error("duplicate remote tool call path `{call_path}`")]
    DuplicateRemoteCallPath { call_path: String },
    #[error("invalid HTTP executor for remote tool `{tool_name}`: {message}")]
    InvalidHttpExecutor { tool_name: String, message: String },
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
    use std::pin::Pin;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Arc, Mutex};

    use base64::Engine as _;
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
                input.turn_context.set_prompt_layer(prompt_layer);
            }
            Ok(input)
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
                .then(|| value.turn_context.prompt_layer().clone());
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
                } => Self::QueuedWorkStarted {
                    boundary: serde_json::to_value(boundary).unwrap_or(serde_json::Value::Null),
                    batch_ids,
                    causes: serde_json::to_value(causes).unwrap_or(serde_json::Value::Null),
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
                lash_core::TurnEvent::PluginRuntime { plugin_id, event } => Self::PluginRuntime {
                    plugin_id,
                    event: serde_json::to_value(event).unwrap_or(serde_json::Value::Null),
                },
                lash_core::TurnEvent::QueuedInputAccepted { checkpoint, inputs } => {
                    Self::QueuedInputAccepted {
                        checkpoint: serde_json::to_value(checkpoint)
                            .unwrap_or(serde_json::Value::Null),
                        inputs: serde_json::to_value(inputs).unwrap_or(serde_json::Value::Null),
                    }
                }
                lash_core::TurnEvent::QueuedMessagesCommitted {
                    messages,
                    checkpoint,
                } => Self::QueuedMessagesCommitted {
                    messages: serde_json::to_value(messages).unwrap_or(serde_json::Value::Null),
                    checkpoint: serde_json::to_value(checkpoint).unwrap_or(serde_json::Value::Null),
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

    pub trait RemoteHttpToolTransport: Send + Sync + 'static {
        fn send<'a>(
            &'a self,
            executor: &'a RemoteHttpToolExecutor,
            request: RemoteToolCallRequest,
        ) -> Pin<
            Box<
                dyn Future<Output = Result<RemoteToolCallResponse, RemoteProtocolError>>
                    + Send
                    + 'a,
            >,
        >;
    }

    pub struct RemoteHttpToolProvider<T: RemoteHttpToolTransport> {
        manifests: Vec<ToolManifest>,
        contracts: HashMap<String, Arc<ToolContract>>,
        executors: HashMap<String, (String, RemoteHttpToolExecutor)>,
        transport: T,
    }

    impl<T: RemoteHttpToolTransport> RemoteHttpToolProvider<T> {
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
                    (executable.call_path(), grant.http_executor.clone()),
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

    impl<T: RemoteHttpToolTransport> ToolProvider for RemoteHttpToolProvider<T> {
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
                let mut headers = executor.headers.clone();
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

        impl RemoteHttpToolTransport for RecordingTransport {
            fn send<'a>(
                &'a self,
                _executor: &'a RemoteHttpToolExecutor,
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
            assert_eq!(remote.prompt_layer, Some(prompt));

            let core = lash_core::TurnInput::try_from(remote).expect("core conversion");
            assert_eq!(core.image_blobs["img"], vec![1, 2, 3]);
            assert_eq!(core.trace_turn_id.as_deref(), Some("trace-1"));
            assert_eq!(
                core.protocol_turn_options.unwrap().payload,
                serde_json::json!({ "mode": "remote" })
            );
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
        async fn remote_http_provider_forwards_idempotency_headers_and_failures() {
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
                RemoteHttpToolProvider::new(vec![demo_grant("demo", "tools", "run")], transport)
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
        fn remote_activity_preserves_semantic_fields() {
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
                http_executor: RemoteHttpToolExecutor::post("https://example.com/tool"),
            }
        }
    }
}

#[cfg(feature = "core-conversions")]
pub use core_conversions::{
    RemoteHttpToolProvider, RemoteHttpToolTransport, RemoteTurnActivitySink,
    replay_collected_activities,
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
    fn remote_turn_input_json_round_trips() {
        let input = RemoteTurnInput {
            protocol_version: REMOTE_PROTOCOL_VERSION,
            items: vec![
                RemoteInputItem::Text {
                    text: "first".to_string(),
                },
                RemoteInputItem::Text {
                    text: "second".to_string(),
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
            prompt_layer: Some(lash_sansio::PromptLayer::new()),
        };

        let value = serde_json::to_value(&input).expect("serialize");
        let decoded: RemoteTurnInput = serde_json::from_value(value).expect("deserialize");

        assert_eq!(decoded.items, input.items);
        assert_eq!(decoded.image_blobs_base64["img"], "AQID");
        assert_eq!(decoded.trace_turn_id.as_deref(), Some("trace"));
        assert_eq!(
            decoded.protocol_turn_options.unwrap().payload,
            serde_json::json!({ "answer": "raw" })
        );
    }

    #[test]
    fn invalid_http_executor_is_rejected() {
        let executor = RemoteHttpToolExecutor::post("");
        let err = executor.validate("demo").expect_err("invalid endpoint");
        assert!(matches!(
            err,
            RemoteProtocolError::InvalidHttpExecutor { .. }
        ));
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
    fn top_level_protocol_schema_exports_include_versions() {
        assert_schema_has_protocol_version::<RemoteTurnInput>();
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
            http_executor: RemoteHttpToolExecutor::post("https://example.com/tool"),
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
