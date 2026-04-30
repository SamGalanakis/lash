pub mod llm;
pub mod mode;
pub mod plugin;
pub mod prompt;
pub mod sansio;
pub mod session;
pub mod session_model;
pub mod tool_surface;
pub mod turn;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

pub use mode::{
    ModeBuildInput, ModeConfig, ModePreamble, append_assistant_text_part,
    normalized_response_parts, reasoning_part, turn_limit_exhausted_message,
};
pub use plugin::{
    CheckpointKind, PluginMessage, PluginSurfaceEvent, PromptContribution, UserInputProvenance,
    UserInputTransform,
};
pub use prompt::{
    PreparedPrompt, PromptBuildInput, PromptCache, build_prompt, build_prompt_cached,
};
pub use sansio::{
    ChatContextProjector, CheckpointResumeAction, CompletedToolCall, ContextProjector,
    DriverAction, DriverContextView, Effect, EffectId, LlmCallError, ModeProtocol, PendingToolCall,
    ProjectorContext, ProtocolDriverHandle, Response, TurnMachine, TurnMachineConfig,
    UnitModeProtocol, WaitingExecState, WaitingLlmState, driver_state,
};
pub use session::{
    CompletedTurn, ExecResponse, PromptUsage, SansIoSessionState, apply_completed_turn,
};
pub use session_model::message::MessageOrigin;
pub use session_model::{
    BaseRenderCache, CORE_GUIDANCE_SECTION, ConversationRecord, ErrorEnvelope, MAIN_AGENT_INTRO,
    Message, MessageRole, MessageSequence, Part, PartKind, PromptBuiltin, PromptPanel,
    PromptRequest, PromptResponse, PromptSelectionMode, PromptSlot, PromptTemplate,
    PromptTemplateEntry, PromptTemplateSection, PruneState, RenderedPrompt, RetryPolicy,
    SessionEvent, SessionEventRecord, StateSnapshotEvent, TokenUsage, ToolEvent,
    default_prompt_template, messages_are_prompt_resume_safe, shared_parts,
};
pub use tool_surface::{
    ToolSurface, ToolSurfaceBuildInput, ToolSurfaceContribution, ToolSurfaceEntry,
    ToolSurfaceOverride, build_tool_surface,
};
pub use turn::{PreparedTurnMachine, SansIoTurnInput, build_turn};

/// Stable string id for the execution backend that owns a session turn.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ExecutionMode(std::sync::Arc<str>);

impl ExecutionMode {
    pub fn new(id: impl Into<std::sync::Arc<str>>) -> Self {
        Self(id.into())
    }

    pub fn standard() -> Self {
        Self::new("standard")
    }

    pub fn plugin_id(&self) -> &str {
        &self.0
    }
}

impl Default for ExecutionMode {
    fn default() -> Self {
        Self::standard()
    }
}

impl std::fmt::Display for ExecutionMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.plugin_id())
    }
}

impl serde::Serialize for ExecutionMode {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.plugin_id())
    }
}

impl<'de> serde::Deserialize<'de> for ExecutionMode {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let id = <String as serde::Deserialize>::deserialize(deserializer)?;
        Ok(Self::new(id))
    }
}

pub fn execution_mode_supported(_mode: &ExecutionMode) -> bool {
    true
}

pub fn default_execution_mode() -> ExecutionMode {
    ExecutionMode::default()
}

/// A typed parameter for a tool definition.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct ToolParam {
    pub name: String,
    /// Type hint: "str", "int", "float", "bool", "list", "dict", "any"
    #[serde(default = "ToolParam::default_type")]
    pub r#type: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub description: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_value: Option<serde_json::Value>,
    #[serde(default = "ToolParam::default_required")]
    pub required: bool,
}

impl ToolParam {
    fn default_type() -> String {
        "any".into()
    }

    fn default_required() -> bool {
        true
    }

    pub fn typed(name: &str, ty: &str) -> Self {
        Self {
            name: name.into(),
            r#type: ty.into(),
            description: String::new(),
            default_value: None,
            required: true,
        }
    }

    pub fn optional(name: &str, ty: &str) -> Self {
        Self {
            name: name.into(),
            r#type: ty.into(),
            description: String::new(),
            default_value: None,
            required: false,
        }
    }
}

/// How a tool's invocations should be scheduled relative to other tools in
/// the same batch of model-produced tool calls.
///
/// Tools that only *read* state (`read_file`, `grep`, `glob`, ...) can run
/// in parallel safely and should use the default [`ToolExecutionMode::Parallel`].
/// Tools that *mutate* shared state (`apply_patch`, `exec_command`,
/// `write_stdin`, `followup_task`) should declare
/// [`ToolExecutionMode::Serial`] so the dispatcher runs them one-at-a-time
/// and avoids interleaving with each other.
///
/// The name is intentionally distinct from the turn-level [`ExecutionMode`]
/// (which selects the execution driver) so the two concepts don't
/// collide in scope.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolExecutionMode {
    /// Safe to run concurrently with other parallel tools in the same batch.
    #[default]
    Parallel,
    /// Must run one-at-a-time relative to other serial tools in the batch.
    Serial,
}

fn default_tool_execution_mode() -> ToolExecutionMode {
    ToolExecutionMode::default()
}

fn is_default_tool_execution_mode(mode: &ToolExecutionMode) -> bool {
    *mode == ToolExecutionMode::default()
}

#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    serde::Serialize,
    serde::Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum ToolAvailability {
    #[default]
    Hidden,
    Discoverable,
    Callable,
    Documented,
}

impl ToolAvailability {
    pub fn is_discoverable(self) -> bool {
        self >= Self::Discoverable
    }

    pub fn is_callable(self) -> bool {
        self >= Self::Callable
    }

    pub fn is_documented(self) -> bool {
        self >= Self::Documented
    }
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ToolAvailabilityConfig {
    pub standard: ToolAvailability,
    #[serde(default, skip_serializing_if = "std::collections::HashMap::is_empty")]
    pub modes: std::collections::HashMap<String, ToolAvailability>,
}

impl ToolAvailabilityConfig {
    pub fn same(availability: ToolAvailability) -> Self {
        Self {
            standard: availability,
            modes: std::collections::HashMap::new(),
        }
    }

    pub fn documented() -> Self {
        Self::same(ToolAvailability::Documented)
    }

    pub fn callable() -> Self {
        Self::same(ToolAvailability::Callable)
    }

    pub fn hidden() -> Self {
        Self::same(ToolAvailability::Hidden)
    }

    pub fn for_mode(&self, mode: &ExecutionMode) -> ToolAvailability {
        self.modes
            .get(mode.plugin_id())
            .copied()
            .unwrap_or(self.standard)
    }
}

impl Default for ToolAvailabilityConfig {
    fn default() -> Self {
        Self::documented()
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolActivation {
    #[default]
    Always,
    Loadable,
    Internal,
}

fn is_default_tool_availability_config(config: &ToolAvailabilityConfig) -> bool {
    *config == ToolAvailabilityConfig::default()
}

fn is_default_tool_activation(activation: &ToolActivation) -> bool {
    *activation == ToolActivation::default()
}

#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ToolDiscoveryMetadata {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub namespace: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub aliases: Vec<String>,
}

impl ToolDiscoveryMetadata {
    pub fn is_empty(&self) -> bool {
        self.namespace.is_none() && self.aliases.is_empty()
    }
}

/// A tool definition exposed to the runtime.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub description: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub params: Vec<ToolParam>,
    #[serde(
        default = "ToolDefinition::default_returns",
        skip_serializing_if = "String::is_empty"
    )]
    pub returns: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub examples: Vec<String>,
    #[serde(default, skip_serializing_if = "is_default_tool_availability_config")]
    pub availability: ToolAvailabilityConfig,
    #[serde(default, skip_serializing_if = "is_default_tool_activation")]
    pub activation: ToolActivation,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub availability_override: Option<ToolAvailability>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_schema_override: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_schema_override: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "ToolDiscoveryMetadata::is_empty")]
    pub discovery: ToolDiscoveryMetadata,
    /// How this tool should be scheduled relative to peers when the model
    /// emits a batch of tool calls. Defaults to [`ToolExecutionMode::Parallel`].
    #[serde(
        default = "default_tool_execution_mode",
        skip_serializing_if = "is_default_tool_execution_mode"
    )]
    pub execution_mode: ToolExecutionMode,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelTool {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
    pub output_schema: serde_json::Value,
}

impl ToolDefinition {
    fn default_returns() -> String {
        "any".into()
    }

    pub fn signature(&self) -> String {
        let params: Vec<String> = self
            .params
            .iter()
            .map(|p| {
                let mut s = if p.default_value.is_none() && !p.required {
                    format!("{}?: {}", p.name, display_prompt_type(&p.r#type))
                } else {
                    format!("{}: {}", p.name, display_prompt_type(&p.r#type))
                };
                if let Some(default) = &p.default_value {
                    s.push_str(" = ");
                    s.push_str(&display_default_value(default));
                }
                s
            })
            .collect();
        let ret = if self.returns.is_empty() {
            "any".to_string()
        } else {
            display_prompt_type(&self.returns)
        };
        format!("{}({}) -> {}", self.name, params.join(", "), ret)
    }

    pub fn effective_availability(&self, mode: &ExecutionMode) -> ToolAvailability {
        self.availability_override
            .unwrap_or_else(|| self.availability.for_mode(mode))
    }

    pub fn model_tool(&self) -> ModelTool {
        ModelTool {
            name: self.name.clone(),
            description: self.description.clone(),
            input_schema: self.input_schema(),
            output_schema: self.output_schema(),
        }
    }

    pub fn format_tool_docs(tools: &[ToolDefinition]) -> String {
        Self::format_tool_docs_iter(tools.iter())
    }

    pub fn format_tool_docs_iter<'a>(
        tools: impl IntoIterator<Item = &'a ToolDefinition>,
    ) -> String {
        tools
            .into_iter()
            .map(|tool| {
                let mut sections = vec![format!("### `{}`", tool.signature())];
                if !tool.description.trim().is_empty() {
                    sections.push(tool.description.trim().to_string());
                }
                sections.join("\n")
            })
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    pub fn input_schema(&self) -> serde_json::Value {
        if let Some(schema) = &self.input_schema_override {
            return schema.clone();
        }

        let mut properties = serde_json::Map::new();
        let mut required = Vec::new();

        for param in &self.params {
            let schema = match param.r#type.as_str() {
                "str" | "string" => serde_json::json!({ "type": "string" }),
                "int" => serde_json::json!({ "type": "integer" }),
                "float" => serde_json::json!({ "type": "number" }),
                "bool" => serde_json::json!({ "type": "boolean" }),
                "list" => serde_json::json!({ "type": "array", "items": {} }),
                "dict" => serde_json::json!({ "type": "object", "additionalProperties": true }),
                _ => serde_json::json!({}),
            };
            let mut schema_obj = match schema {
                serde_json::Value::Object(obj) => obj,
                _ => serde_json::Map::new(),
            };
            if !param.description.is_empty() {
                schema_obj.insert(
                    "description".to_string(),
                    serde_json::Value::String(param.description.clone()),
                );
            }
            if let Some(default) = &param.default_value {
                schema_obj.insert("default".to_string(), default.clone());
            }
            properties.insert(param.name.clone(), serde_json::Value::Object(schema_obj));
            if param.required {
                required.push(param.name.clone());
            }
        }

        serde_json::json!({
            "type": "object",
            "properties": properties,
            "required": required,
        })
    }

    pub fn output_schema(&self) -> serde_json::Value {
        if let Some(schema) = &self.output_schema_override {
            return schema.clone();
        }

        let ty = self.returns.trim();
        match ty {
            "" | "any" => serde_json::json!({}),
            "str" | "string" => serde_json::json!({ "type": "string" }),
            "int" | "integer" => serde_json::json!({ "type": "integer" }),
            "float" | "number" => serde_json::json!({ "type": "number" }),
            "bool" | "boolean" => serde_json::json!({ "type": "boolean" }),
            "dict" | "json" => {
                serde_json::json!({ "type": "object", "additionalProperties": true })
            }
            "None" | "null" => serde_json::json!({ "type": "null" }),
            _ if ty.starts_with("list") => serde_json::json!({ "type": "array", "items": {} }),
            _ => serde_json::json!({}),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::ser::{Error as _, Serializer};

    #[test]
    fn tool_definition_uses_schema_overrides_for_model_tools() {
        let tool = ToolDefinition {
            name: "mcp__demo__search".to_string(),
            description: "Search demo server".to_string(),
            params: vec![ToolParam::typed("query", "str")],
            returns: "str".to_string(),
            examples: vec![],
            availability: ToolAvailabilityConfig::documented(),
            activation: ToolActivation::Always,
            availability_override: None,
            input_schema_override: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string" },
                    "limit": { "type": "integer" }
                },
                "required": ["query"],
                "additionalProperties": false
            })),
            output_schema_override: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "hits": { "type": "array", "items": { "type": "string" } }
                },
                "required": ["hits"],
                "additionalProperties": false
            })),
            discovery: Default::default(),
            execution_mode: ToolExecutionMode::Parallel,
        };

        let model_tool = tool.model_tool();
        assert_eq!(
            model_tool.input_schema["properties"]["limit"]["type"],
            serde_json::json!("integer")
        );
        assert_eq!(
            model_tool.output_schema["properties"]["hits"]["type"],
            serde_json::json!("array")
        );
    }

    #[test]
    fn tool_result_from_result_serializes_success_values() {
        let result: ToolResult = Result::<_, std::io::Error>::Ok(vec!["alpha", "beta"]).into();
        assert!(result.success);
        assert_eq!(result.result, serde_json::json!(["alpha", "beta"]));
        assert!(result.images.is_empty());
    }

    #[test]
    fn tool_result_from_result_formats_errors() {
        let result: ToolResult =
            Result::<serde_json::Value, _>::Err(std::io::Error::other("nope")).into();
        assert!(!result.success);
        assert_eq!(result.result, serde_json::json!("nope"));
    }

    #[test]
    fn tool_result_from_result_reports_serialize_failures() {
        struct BrokenValue;

        impl serde::Serialize for BrokenValue {
            fn serialize<S>(&self, _serializer: S) -> Result<S::Ok, S::Error>
            where
                S: Serializer,
            {
                Err(S::Error::custom("boom"))
            }
        }

        let result: ToolResult = Result::<BrokenValue, std::io::Error>::Ok(BrokenValue).into();
        assert!(!result.success);
        assert_eq!(
            result.result,
            serde_json::json!("Failed to serialize tool result: boom")
        );
    }

    #[test]
    fn tool_discovery_metadata_serde_defaults_are_empty() {
        let tool: ToolDefinition = serde_json::from_value(serde_json::json!({
            "name": "read_file",
            "description": "Read a file"
        }))
        .unwrap();
        assert!(tool.discovery.is_empty());
    }

    #[test]
    fn tool_discovery_metadata_does_not_render_prompt_docs() {
        let with_metadata = ToolDefinition {
            name: "read_file".to_string(),
            description: "Read a file".to_string(),
            params: Vec::new(),
            returns: "str".to_string(),
            examples: Vec::new(),
            availability: ToolAvailabilityConfig::documented(),
            activation: ToolActivation::Always,
            availability_override: None,
            input_schema_override: None,
            output_schema_override: None,
            discovery: ToolDiscoveryMetadata {
                namespace: Some("filesystem".to_string()),
                aliases: vec!["cat".to_string()],
            },
            execution_mode: ToolExecutionMode::Parallel,
        };
        let mut without_metadata = with_metadata.clone();
        without_metadata.discovery = Default::default();
        assert_eq!(
            ToolDefinition::format_tool_docs(&[with_metadata]),
            ToolDefinition::format_tool_docs(&[without_metadata])
        );
    }
}

fn display_prompt_type(ty: &str) -> String {
    let trimmed = ty.trim();
    if let Some(inner) = trimmed
        .strip_prefix("list[")
        .and_then(|rest| rest.strip_suffix(']'))
    {
        return format!("list[{}]", display_prompt_type(inner));
    }
    match trimmed {
        "string" => "str".to_string(),
        "integer" => "int".to_string(),
        "number" => "float".to_string(),
        "boolean" => "bool".to_string(),
        "dict" | "json" => "record".to_string(),
        "None" | "null" => "null".to_string(),
        other => other.to_string(),
    }
}

fn display_default_value(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Null => "null".to_string(),
        serde_json::Value::Bool(v) => v.to_string(),
        serde_json::Value::Number(v) => v.to_string(),
        serde_json::Value::String(v) => format!("{v:?}"),
        _ => serde_json::to_string(value).unwrap_or_else(|_| "null".to_string()),
    }
}

#[derive(Clone, Debug)]
pub struct ToolImage {
    pub mime: String,
    pub data: Vec<u8>,
    pub label: String,
}

#[derive(Clone, Debug)]
pub struct ToolResult {
    pub success: bool,
    pub result: serde_json::Value,
    pub images: Vec<ToolImage>,
}

impl ToolResult {
    pub fn ok(result: serde_json::Value) -> Self {
        Self {
            success: true,
            result,
            images: vec![],
        }
    }

    pub fn err(result: serde_json::Value) -> Self {
        Self {
            success: false,
            result,
            images: vec![],
        }
    }

    pub fn err_fmt(msg: impl std::fmt::Display) -> Self {
        Self::err(serde_json::json!(msg.to_string()))
    }

    pub fn with_images(success: bool, result: serde_json::Value, images: Vec<ToolImage>) -> Self {
        Self {
            success,
            result,
            images,
        }
    }
}

impl<T, E> From<Result<T, E>> for ToolResult
where
    T: serde::Serialize,
    E: std::fmt::Display,
{
    fn from(result: Result<T, E>) -> Self {
        match result {
            Ok(value) => match serde_json::to_value(value) {
                Ok(value) => Self::ok(value),
                Err(err) => Self::err_fmt(format_args!("Failed to serialize tool result: {err}")),
            },
            Err(err) => Self::err_fmt(err),
        }
    }
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ToolCallRecord {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub call_id: Option<String>,
    pub tool: String,
    pub args: serde_json::Value,
    pub result: serde_json::Value,
    pub success: bool,
    pub duration_ms: u64,
}

pub fn head_tail_truncate(value: &str, max_chars: usize) -> (String, usize) {
    let raw_len = value.chars().count();
    if max_chars == 0 || raw_len <= max_chars {
        return (value.to_string(), raw_len);
    }
    let head_len = max_chars / 2;
    let tail_len = max_chars.saturating_sub(head_len);
    let head = value.chars().take(head_len).collect::<String>();
    let tail = value
        .chars()
        .rev()
        .take(tail_len)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<String>();
    let omitted = raw_len.saturating_sub(head_len + tail_len);
    (
        format!("{head}\n\n... ({omitted} characters omitted) ...\n\n{tail}"),
        raw_len,
    )
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct PromptContext {
    pub mode: ExecutionMode,
    #[serde(default)]
    pub execution_prompt: String,
    pub tool_names: Vec<String>,
    pub omitted_tool_count: usize,
    pub contributions: Vec<PromptContribution>,
}

impl PromptContext {
    pub fn has_tool(&self, tool_name: &str) -> bool {
        self.tool_names.iter().any(|name| name == tool_name)
    }
}
