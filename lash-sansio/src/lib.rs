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
    CompletedTurn, ExecResponse, PromptUsage, SansIoSessionState, TextProjectionMetadata,
    apply_completed_turn,
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
    #[serde(default = "ToolDefinition::default_input_schema")]
    pub input_schema: serde_json::Value,
    #[serde(default)]
    pub output_schema: serde_json::Value,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub examples: Vec<String>,
    #[serde(default, skip_serializing_if = "is_default_tool_availability_config")]
    pub availability: ToolAvailabilityConfig,
    #[serde(default, skip_serializing_if = "is_default_tool_activation")]
    pub activation: ToolActivation,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub availability_override: Option<ToolAvailability>,
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

const COMPACT_TOOL_EXAMPLE_LIMIT: usize = 2;
const COMPACT_TOOL_EXAMPLE_CHAR_LIMIT: usize = 240;
const COMPACT_SCHEMA_FIELD_LIMIT: usize = 8;

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CompactToolContract {
    pub name: String,
    pub signature: String,
    pub returns: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub description: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub examples: Vec<String>,
}

impl CompactToolContract {
    pub fn render_markdown(&self) -> String {
        let mut sections = vec![format!("### `{}`", self.signature)];
        if !self.returns.trim().is_empty() {
            sections.push(format!("Returns: {}", self.returns.trim()));
        }
        if !self.description.trim().is_empty() {
            sections.push(self.description.trim().to_string());
        }
        if !self.examples.is_empty() {
            sections.push(format!("Examples: {}", self.examples.join("; ")));
        }
        sections.join("\n")
    }
}

impl ToolDefinition {
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        input_schema: serde_json::Value,
        output_schema: serde_json::Value,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            input_schema,
            output_schema,
            examples: Vec::new(),
            availability: ToolAvailabilityConfig::documented(),
            activation: ToolActivation::Always,
            availability_override: None,
            discovery: ToolDiscoveryMetadata::default(),
            execution_mode: ToolExecutionMode::Parallel,
        }
    }

    pub fn native<Args>(name: impl Into<String>, description: impl Into<String>) -> Self
    where
        Args: schemars::JsonSchema,
    {
        Self::new(
            name,
            description,
            schema_for::<Args>(),
            serde_json::json!({}),
        )
    }

    pub fn native_with_output<Args, Output>(
        name: impl Into<String>,
        description: impl Into<String>,
    ) -> Self
    where
        Args: schemars::JsonSchema,
        Output: schemars::JsonSchema,
    {
        Self::new(
            name,
            description,
            schema_for::<Args>(),
            schema_for::<Output>(),
        )
    }

    pub fn with_examples(mut self, examples: Vec<String>) -> Self {
        self.examples = examples;
        self
    }

    pub fn with_availability(mut self, availability: ToolAvailabilityConfig) -> Self {
        self.availability = availability;
        self
    }

    pub fn with_activation(mut self, activation: ToolActivation) -> Self {
        self.activation = activation;
        self
    }

    pub fn with_discovery(mut self, discovery: ToolDiscoveryMetadata) -> Self {
        self.discovery = discovery;
        self
    }

    pub fn with_execution_mode(mut self, execution_mode: ToolExecutionMode) -> Self {
        self.execution_mode = execution_mode;
        self
    }

    pub fn default_input_schema() -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {},
            "additionalProperties": true
        })
    }

    pub fn input_signature(&self) -> String {
        let params = schema_parameter_docs(&self.input_schema)
            .into_iter()
            .map(|p| p.signature_fragment())
            .collect::<Vec<_>>();
        format!("{}({})", self.name, params.join(", "))
    }

    pub fn output_summary(&self) -> String {
        compact_schema_label(&self.output_schema)
    }

    pub fn signature(&self) -> String {
        format!("{} -> {}", self.input_signature(), self.output_summary())
    }

    pub fn compact_contract(&self) -> CompactToolContract {
        self.compact_contract_with_example_limit(COMPACT_TOOL_EXAMPLE_LIMIT)
    }

    pub fn compact_contract_with_example_limit(&self, example_limit: usize) -> CompactToolContract {
        CompactToolContract {
            name: self.name.clone(),
            signature: self.input_signature(),
            returns: self.output_summary(),
            description: self.description.trim().to_string(),
            examples: compact_examples(&self.examples, example_limit),
        }
    }

    pub fn effective_availability(&self, mode: &ExecutionMode) -> ToolAvailability {
        self.availability_override
            .unwrap_or_else(|| self.availability.for_mode(mode))
    }

    pub fn model_tool(&self) -> ModelTool {
        ModelTool {
            name: self.name.clone(),
            description: self.description.clone(),
            input_schema: self.input_schema.clone(),
            output_schema: self.output_schema.clone(),
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
            .map(|tool| tool.compact_contract().render_markdown())
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    pub fn parameter_metadata(&self) -> Vec<serde_json::Value> {
        schema_parameter_docs(&self.input_schema)
            .into_iter()
            .map(|param| param.to_value())
            .collect()
    }
}

pub fn schema_for<T>() -> serde_json::Value
where
    T: schemars::JsonSchema,
{
    serde_json::to_value(schemars::schema_for!(T)).unwrap_or_else(|_| serde_json::json!({}))
}

#[derive(Clone, Debug, PartialEq)]
struct ParameterDoc {
    name: String,
    type_label: String,
    required: bool,
    nullable: bool,
    description: Option<String>,
    default_value: Option<serde_json::Value>,
    enum_values: Vec<serde_json::Value>,
    minimum: Option<serde_json::Value>,
    maximum: Option<serde_json::Value>,
    min_length: Option<u64>,
    max_length: Option<u64>,
    min_items: Option<u64>,
    max_items: Option<u64>,
    item_type: Option<String>,
}

impl ParameterDoc {
    fn signature_fragment(&self) -> String {
        let mut out = if self.required {
            format!("{}: {}", self.name, self.type_label)
        } else {
            format!("{}?: {}", self.name, self.type_label)
        };
        let constraints = self.constraint_fragments();
        if !constraints.is_empty() {
            out.push(' ');
            out.push_str(&constraints.join(" "));
        }
        if let Some(default) = &self.default_value {
            out.push_str(" = ");
            out.push_str(&display_default_value(default));
        }
        out
    }

    fn constraint_fragments(&self) -> Vec<String> {
        let mut out = Vec::new();
        if !self.enum_values.is_empty() {
            out.push(format!(
                "in {}",
                self.enum_values
                    .iter()
                    .map(display_default_value)
                    .collect::<Vec<_>>()
                    .join("|")
            ));
        }
        if let Some(minimum) = &self.minimum {
            out.push(format!(">= {}", display_default_value(minimum)));
        }
        if let Some(maximum) = &self.maximum {
            out.push(format!("<= {}", display_default_value(maximum)));
        }
        if let Some(min_length) = self.min_length {
            out.push(format!("min_len {min_length}"));
        }
        if let Some(max_length) = self.max_length {
            out.push(format!("max_len {max_length}"));
        }
        if let Some(min_items) = self.min_items {
            out.push(format!("min_items {min_items}"));
        }
        if let Some(max_items) = self.max_items {
            out.push(format!("max_items {max_items}"));
        }
        out
    }

    fn to_value(self) -> serde_json::Value {
        let mut out = serde_json::Map::new();
        out.insert("name".to_string(), serde_json::json!(self.name));
        out.insert("type".to_string(), serde_json::json!(self.type_label));
        out.insert("required".to_string(), serde_json::json!(self.required));
        if self.nullable {
            out.insert("nullable".to_string(), serde_json::json!(true));
        }
        if let Some(description) = self.description.filter(|value| !value.trim().is_empty()) {
            out.insert("description".to_string(), serde_json::json!(description));
        }
        if let Some(default_value) = self.default_value {
            out.insert("default".to_string(), default_value);
        }
        if !self.enum_values.is_empty() {
            out.insert("enum".to_string(), serde_json::json!(self.enum_values));
        }
        if let Some(value) = self.minimum {
            out.insert("minimum".to_string(), value);
        }
        if let Some(value) = self.maximum {
            out.insert("maximum".to_string(), value);
        }
        if let Some(value) = self.min_length {
            out.insert("min_length".to_string(), serde_json::json!(value));
        }
        if let Some(value) = self.max_length {
            out.insert("max_length".to_string(), serde_json::json!(value));
        }
        if let Some(value) = self.min_items {
            out.insert("min_items".to_string(), serde_json::json!(value));
        }
        if let Some(value) = self.max_items {
            out.insert("max_items".to_string(), serde_json::json!(value));
        }
        if let Some(value) = self.item_type {
            out.insert("items".to_string(), serde_json::json!(value));
        }
        serde_json::Value::Object(out)
    }
}

fn schema_parameter_docs(schema: &serde_json::Value) -> Vec<ParameterDoc> {
    let required = schema
        .get("required")
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(serde_json::Value::as_str)
        .collect::<std::collections::BTreeSet<_>>();
    let Some(properties) = schema
        .get("properties")
        .and_then(serde_json::Value::as_object)
    else {
        return Vec::new();
    };
    properties
        .iter()
        .map(|(name, schema)| parameter_doc(name, schema, required.contains(name.as_str())))
        .collect()
}

fn parameter_doc(name: &str, schema: &serde_json::Value, required: bool) -> ParameterDoc {
    let (type_label, nullable) = schema_type_label_and_nullability(schema);
    ParameterDoc {
        name: name.to_string(),
        type_label,
        required,
        nullable,
        description: schema
            .get("description")
            .and_then(serde_json::Value::as_str)
            .map(str::to_string),
        default_value: schema.get("default").cloned(),
        enum_values: schema
            .get("enum")
            .and_then(serde_json::Value::as_array)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .filter(|value| !value.is_null())
            .collect(),
        minimum: schema
            .get("minimum")
            .or_else(|| schema.get("exclusiveMinimum"))
            .cloned(),
        maximum: schema
            .get("maximum")
            .or_else(|| schema.get("exclusiveMaximum"))
            .cloned(),
        min_length: schema.get("minLength").and_then(serde_json::Value::as_u64),
        max_length: schema.get("maxLength").and_then(serde_json::Value::as_u64),
        min_items: schema.get("minItems").and_then(serde_json::Value::as_u64),
        max_items: schema.get("maxItems").and_then(serde_json::Value::as_u64),
        item_type: schema
            .get("items")
            .map(schema_type_label)
            .filter(|value| value != "any"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::ser::{Error as _, Serializer};

    #[test]
    fn tool_definition_uses_canonical_schemas_for_model_tools() {
        let tool = ToolDefinition::new(
            "mcp__demo__search",
            "Search demo server",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string" },
                    "limit": { "type": "integer" }
                },
                "required": ["query"],
                "additionalProperties": false
            }),
            serde_json::json!({
                "type": "object",
                "properties": {
                    "hits": { "type": "array", "items": { "type": "string" } }
                },
                "required": ["hits"],
                "additionalProperties": false
            }),
        );

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
    fn native_tool_definition_generates_schema_from_typed_args() {
        #[derive(schemars::JsonSchema)]
        #[allow(dead_code)]
        enum Mode {
            Fast,
            Slow,
        }

        #[derive(schemars::JsonSchema)]
        #[allow(dead_code)]
        struct Args {
            query: String,
            #[schemars(range(max = 20))]
            page_limit: u8,
            #[schemars(length(min = 1, max = 3))]
            tags: Vec<String>,
            mode: Option<Mode>,
        }

        let tool = ToolDefinition::native::<Args>("demo", "Demo");
        let metadata = tool.parameter_metadata();
        assert!(metadata.iter().any(|param| {
            param["name"] == "page_limit"
                && param["type"] == "int"
                && param["maximum"].as_f64() == Some(20.0)
        }));
        assert!(metadata.iter().any(|param| {
            param["name"] == "tags"
                && param["type"] == "list[str]"
                && param["min_items"] == 1
                && param["max_items"] == 3
        }));
        assert!(
            metadata
                .iter()
                .any(|param| { param["name"] == "mode" && param["nullable"] == true })
        );
    }

    #[test]
    fn compact_tool_contract_renders_prompt_and_search_shape_from_schemas() {
        let tool = ToolDefinition::new(
            "search_docs",
            "Search indexed docs",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string" },
                    "limit": { "type": "integer", "maximum": 10, "default": 5 }
                },
                "required": ["query"]
            }),
            serde_json::json!({
                "type": "object",
                "properties": {
                    "matches": {
                        "type": "array",
                        "items": { "type": "string" }
                    },
                    "next_page": { "type": ["string", "null"] }
                },
                "required": ["matches"]
            }),
        )
        .with_examples(vec![
            "search_docs(query=\"rust\")".to_string(),
            "search_docs(query=\"rust\", limit=3)".to_string(),
            "search_docs(query=\"ignored\")".to_string(),
        ]);

        let contract = tool.compact_contract();
        assert_eq!(
            contract.signature,
            "search_docs(limit?: int <= 10 = 5, query: str)"
        );
        assert_eq!(
            contract.returns,
            "record{matches: list[str], next_page?: str | null}"
        );
        assert_eq!(contract.examples.len(), 2);

        let docs = ToolDefinition::format_tool_docs(&[tool]);
        assert!(docs.contains("### `search_docs(limit?: int <= 10 = 5, query: str)`"));
        assert!(docs.contains("Returns: record{matches: list[str], next_page?: str | null}"));
        assert!(docs.contains(
            "Examples: search_docs(query=\"rust\"); search_docs(query=\"rust\", limit=3)"
        ));
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
        let mut with_metadata = ToolDefinition::new(
            "read_file",
            "Read a file",
            ToolDefinition::default_input_schema(),
            serde_json::json!({"type": "string"}),
        );
        with_metadata.discovery = ToolDiscoveryMetadata {
            namespace: Some("filesystem".to_string()),
            aliases: vec!["cat".to_string()],
        };
        let mut without_metadata = with_metadata.clone();
        without_metadata.discovery = Default::default();
        assert_eq!(
            ToolDefinition::format_tool_docs(&[with_metadata]),
            ToolDefinition::format_tool_docs(&[without_metadata])
        );
    }
}

fn schema_type_label(schema: &serde_json::Value) -> String {
    schema_type_label_and_nullability(schema).0
}

fn compact_schema_label(schema: &serde_json::Value) -> String {
    if let Some(any_of) = schema
        .get("anyOf")
        .or_else(|| schema.get("oneOf"))
        .and_then(serde_json::Value::as_array)
    {
        let labels = any_of
            .iter()
            .map(compact_schema_label)
            .collect::<std::collections::BTreeSet<_>>();
        let joined = labels.into_iter().collect::<Vec<_>>().join(" | ");
        return if joined.is_empty() {
            "any".to_string()
        } else {
            joined
        };
    }

    if let Some(types) = schema.get("type").and_then(serde_json::Value::as_array) {
        let labels = types
            .iter()
            .filter_map(serde_json::Value::as_str)
            .filter(|ty| *ty != "null")
            .map(|ty| compact_schema_label(&serde_json::json!({ "type": ty })))
            .collect::<std::collections::BTreeSet<_>>();
        let mut out = if labels.is_empty() {
            "any".to_string()
        } else {
            labels.into_iter().collect::<Vec<_>>().join(" | ")
        };
        if types.iter().any(|value| value.as_str() == Some("null")) {
            out.push_str(" | null");
        }
        return out;
    }

    match schema.get("type").and_then(serde_json::Value::as_str) {
        Some("array") => schema
            .get("items")
            .map(compact_schema_label)
            .filter(|value| !value.is_empty())
            .map(|item| format!("list[{item}]"))
            .unwrap_or_else(|| "list[any]".to_string()),
        Some("object") => compact_record_label(schema),
        _ => schema_type_label(schema),
    }
}

fn compact_record_label(schema: &serde_json::Value) -> String {
    let Some(properties) = schema
        .get("properties")
        .and_then(serde_json::Value::as_object)
    else {
        return "record".to_string();
    };
    if properties.is_empty() {
        return "record".to_string();
    }

    let required = schema
        .get("required")
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(serde_json::Value::as_str)
        .collect::<std::collections::BTreeSet<_>>();
    let mut fields = properties
        .iter()
        .take(COMPACT_SCHEMA_FIELD_LIMIT)
        .map(|(name, field_schema)| {
            let suffix = if required.contains(name.as_str()) {
                ""
            } else {
                "?"
            };
            format!("{name}{suffix}: {}", compact_schema_label(field_schema))
        })
        .collect::<Vec<_>>();
    if properties.len() > COMPACT_SCHEMA_FIELD_LIMIT {
        fields.push("...".to_string());
    }
    format!("record{{{}}}", fields.join(", "))
}

fn compact_examples(examples: &[String], limit: usize) -> Vec<String> {
    examples
        .iter()
        .map(|example| example.trim())
        .filter(|example| !example.is_empty())
        .take(limit)
        .map(|example| {
            if example.chars().count() <= COMPACT_TOOL_EXAMPLE_CHAR_LIMIT {
                return example.to_string();
            }
            let mut out = example
                .chars()
                .take(COMPACT_TOOL_EXAMPLE_CHAR_LIMIT.saturating_sub(3))
                .collect::<String>();
            out.push_str("...");
            out
        })
        .collect()
}

fn schema_type_label_and_nullability(schema: &serde_json::Value) -> (String, bool) {
    if let Some(values) = schema.get("enum").and_then(serde_json::Value::as_array) {
        let variants = values
            .iter()
            .filter(|value| !value.is_null())
            .map(display_default_value)
            .collect::<Vec<_>>();
        let nullable = values.iter().any(serde_json::Value::is_null);
        if !variants.is_empty() {
            let mut label = format!("enum[{}]", variants.join(", "));
            if nullable {
                label.push_str(" | null");
            }
            return (label, nullable);
        }
    }

    if let Some(types) = schema.get("type").and_then(serde_json::Value::as_array) {
        let nullable = types.iter().any(|value| value.as_str() == Some("null"));
        let non_null = types
            .iter()
            .filter_map(serde_json::Value::as_str)
            .filter(|ty| *ty != "null")
            .map(schema_type_name)
            .collect::<Vec<_>>();
        let mut label = if non_null.is_empty() {
            "any".to_string()
        } else {
            non_null.join(" | ")
        };
        if nullable {
            label.push_str(" | null");
        }
        return (label, nullable);
    }

    if let Some(any_of) = schema
        .get("anyOf")
        .or_else(|| schema.get("oneOf"))
        .and_then(serde_json::Value::as_array)
    {
        let mut nullable = false;
        let mut labels = Vec::new();
        for subschema in any_of {
            let (label, is_nullable) = schema_type_label_and_nullability(subschema);
            nullable |= is_nullable || label == "null";
            if label != "null" && !labels.iter().any(|existing| existing == &label) {
                labels.push(label);
            }
        }
        let mut label = if labels.is_empty() {
            "any".to_string()
        } else {
            labels.join(" | ")
        };
        if nullable {
            label.push_str(" | null");
        }
        return (label, nullable);
    }

    let nullable = schema.get("type").and_then(serde_json::Value::as_str) == Some("null");
    let label = match schema.get("type").and_then(serde_json::Value::as_str) {
        Some("array") => {
            let item = schema
                .get("items")
                .map(schema_type_label)
                .filter(|value| !value.is_empty())
                .unwrap_or_else(|| "any".to_string());
            format!("list[{item}]")
        }
        Some(ty) => schema_type_name(ty),
        None => "any".to_string(),
    };
    (label, nullable)
}

fn schema_type_name(ty: &str) -> String {
    match ty {
        "string" => "str".to_string(),
        "integer" => "int".to_string(),
        "number" => "float".to_string(),
        "boolean" => "bool".to_string(),
        "object" => "record".to_string(),
        "array" => "list".to_string(),
        "null" => "null".to_string(),
        _ => "any".to_string(),
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
