pub mod attachment;
pub mod llm;
pub mod mode;
pub mod plugin;
pub mod prompt;
pub mod sansio;
pub mod session;
pub mod session_model;
pub mod tool_surface;
pub mod turn;

use std::sync::Arc;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

pub use attachment::{AttachmentId, AttachmentMeta, AttachmentRef, ImageMediaType, MediaType};
pub use mode::{
    ModeBuildInput, ModeConfig, ModePreamble, append_assistant_text_part,
    normalized_response_parts, reasoning_part, turn_limit_exhausted_message,
};
pub use plugin::{
    CheckpointKind, PluginMessage, PluginSurfaceEvent, PromptContribution, PromptContributionGate,
};
pub use prompt::{
    PreparedPrompt, PromptBuildInput, PromptCache, PromptContributionSet, PromptFingerprint,
    build_prompt, build_prompt_cached, prompt_template_fingerprint, prompt_text_fingerprint,
    prompt_tool_names_fingerprint,
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
    AcceptedInjectedTurnInput, BaseRenderCache, ConversationRecord, ErrorEnvelope,
    MAIN_AGENT_INTRO, Message, MessageRole, MessageSequence, Part, PartAttachment, PartKind,
    PromptBuiltin, PromptLayer, PromptPanel, PromptRequest, PromptResponse, PromptSelectionMode,
    PromptSlot, PromptSlotLayer, PromptTemplate, PromptTemplateEntry, PromptTemplateSection,
    PruneState, RenderedPrompt, ResolvedPromptLayer, SessionEvent, SessionEventRecord,
    StateSnapshotEvent, TokenUsage, ToolEvent, TurnFinish, TurnOutcome, TurnStop,
    default_prompt_template, messages_are_prompt_resume_safe, resolve_prompt_layers, shared_parts,
};
pub use tool_surface::{
    ToolContractResolver, ToolSurface, ToolSurfaceBuildInput, ToolSurfaceContribution,
    ToolSurfaceEntry, ToolSurfaceOverride, build_tool_surface,
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
/// `write_stdin`) should declare
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
    /// Keep the tool out of the current surface entirely.
    ///
    /// The definition can remain in registry state so host or authority
    /// overrides survive refreshes, but the model cannot search, see, or call
    /// the tool.
    #[default]
    Off,
    /// Include the tool in the searchable catalog, but not in the model's
    /// callable tool list.
    Searchable,
    /// Include the tool in the model's callable tool list, without featuring
    /// it in prompt-side tool documentation.
    Callable,
    /// Include the tool in the callable list and feature it in prompt-side
    /// tool documentation.
    Showcased,
}

impl ToolAvailability {
    pub fn is_searchable(self) -> bool {
        self >= Self::Searchable
    }

    pub fn is_callable(self) -> bool {
        self >= Self::Callable
    }

    pub fn is_showcased(self) -> bool {
        self >= Self::Showcased
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

    pub fn showcased() -> Self {
        Self::same(ToolAvailability::Showcased)
    }

    pub fn callable() -> Self {
        Self::same(ToolAvailability::Callable)
    }

    pub fn off() -> Self {
        Self::same(ToolAvailability::Off)
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
        Self::showcased()
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolActivation {
    #[default]
    Always,
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

#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ToolOutputContract {
    #[default]
    Static,
    FromInputSchema {
        input_field: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        default_schema: Option<serde_json::Value>,
    },
}

impl ToolOutputContract {
    pub fn from_input_schema(
        input_field: impl Into<String>,
        default_schema: Option<serde_json::Value>,
    ) -> Self {
        Self::FromInputSchema {
            input_field: input_field.into(),
            default_schema,
        }
    }

    pub fn is_static(&self) -> bool {
        matches!(self, Self::Static)
    }

    fn return_type_label(&self, static_schema: &serde_json::Value) -> String {
        match self {
            Self::Static => compact_schema_label(static_schema),
            Self::FromInputSchema { .. } => "T".to_string(),
        }
    }

    fn type_parameter_suffix(&self) -> Option<String> {
        match self {
            Self::Static => None,
            Self::FromInputSchema { default_schema, .. } => {
                let default = default_schema
                    .as_ref()
                    .map(compact_schema_label)
                    .unwrap_or_else(|| "any".to_string());
                Some(format!("<T = {default}>"))
            }
        }
    }

    fn apply_type_witness_parameter(&self, params: &mut [ParameterDoc]) {
        let Self::FromInputSchema { input_field, .. } = self else {
            return;
        };
        if let Some(param) = params.iter_mut().find(|param| param.name == *input_field) {
            param.type_label = "TypeSpec<T>".to_string();
            param.nullable = false;
            param.default_value = None;
            param.enum_values.clear();
            param.minimum = None;
            param.maximum = None;
            param.min_length = None;
            param.max_length = None;
            param.min_items = None;
            param.max_items = None;
            param.item_type = None;
        }
    }

    fn return_fields(&self, static_schema: &serde_json::Value) -> Vec<serde_json::Value> {
        match self {
            Self::Static => return_field_metadata(static_schema),
            Self::FromInputSchema { .. } => Vec::new(),
        }
    }
}

/// Cheap tool metadata exposed to prompts, catalogs, UI, and availability checks.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ToolManifest {
    pub name: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub description: String,
    #[serde(default, skip_serializing_if = "is_default_tool_availability_config")]
    pub availability: ToolAvailabilityConfig,
    #[serde(default, skip_serializing_if = "is_default_tool_activation")]
    pub activation: ToolActivation,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub availability_override: Option<ToolAvailability>,
    #[serde(default, skip_serializing_if = "ToolDiscoveryMetadata::is_empty")]
    pub discovery: ToolDiscoveryMetadata,
    #[serde(
        default = "default_tool_execution_mode",
        skip_serializing_if = "is_default_tool_execution_mode"
    )]
    pub execution_mode: ToolExecutionMode,
}

impl ToolManifest {
    pub fn effective_availability(&self, mode: &ExecutionMode) -> ToolAvailability {
        self.availability_override
            .unwrap_or_else(|| self.availability.for_mode(mode))
    }
}

/// Heavy tool contract resolved only when a prompt or call needs schemas/docs.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ToolContract {
    #[serde(default = "ToolDefinition::default_input_schema")]
    pub input_schema: serde_json::Value,
    #[serde(default)]
    pub output_schema: serde_json::Value,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub input_schema_projections: Vec<SchemaProjectionOverride>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub output_schema_projections: Vec<SchemaProjectionOverride>,
    #[serde(default, skip_serializing_if = "ToolOutputContract::is_static")]
    pub output_contract: ToolOutputContract,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub examples: Vec<String>,
}

impl ToolContract {
    pub fn compact_contract(&self, manifest: &ToolManifest) -> CompactToolContract {
        self.compact_contract_with_example_limit(manifest, COMPACT_TOOL_EXAMPLE_LIMIT)
    }

    pub fn compact_contract_with_example_limit(
        &self,
        manifest: &ToolManifest,
        example_limit: usize,
    ) -> CompactToolContract {
        CompactToolContract {
            name: manifest.name.clone(),
            signature: self.input_signature(manifest),
            returns: self.output_summary(),
            parameters: self.parameter_metadata(),
            return_fields: self.output_contract.return_fields(&self.output_schema),
            description: manifest.description.trim().to_string(),
            examples: compact_examples(&self.examples, example_limit),
        }
    }

    pub fn input_signature(&self, manifest: &ToolManifest) -> String {
        let params = self
            .parameter_docs()
            .into_iter()
            .map(|p| p.signature_fragment())
            .collect::<Vec<_>>();
        format!(
            "{}{}({})",
            manifest.name,
            self.output_contract
                .type_parameter_suffix()
                .unwrap_or_default(),
            params.join(", ")
        )
    }

    pub fn output_summary(&self) -> String {
        self.output_contract.return_type_label(&self.output_schema)
    }

    pub fn parameter_metadata(&self) -> Vec<serde_json::Value> {
        self.parameter_docs()
            .into_iter()
            .map(|param| param.into_value())
            .collect()
    }

    pub fn model_tool(&self, manifest: &ToolManifest) -> ModelTool {
        ModelTool {
            name: manifest.name.clone(),
            description: manifest.description.clone(),
            input_schema: self.input_schema.clone(),
            output_schema: self.output_schema.clone(),
            input_schema_projections: self.input_schema_projections.clone(),
            output_schema_projections: self.output_schema_projections.clone(),
        }
    }

    fn parameter_docs(&self) -> Vec<ParameterDoc> {
        let mut params = schema_parameter_docs(&self.input_schema);
        self.output_contract
            .apply_type_witness_parameter(&mut params);
        params
    }
}

/// Static authoring helper for tools.
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
    pub input_schema_projections: Vec<SchemaProjectionOverride>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub output_schema_projections: Vec<SchemaProjectionOverride>,
    #[serde(default, skip_serializing_if = "ToolOutputContract::is_static")]
    pub output_contract: ToolOutputContract,
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
    pub input_schema_projections: Vec<SchemaProjectionOverride>,
    pub output_schema_projections: Vec<SchemaProjectionOverride>,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SchemaProjectionOverride {
    pub profile: String,
    pub schema: serde_json::Value,
}

const COMPACT_TOOL_EXAMPLE_LIMIT: usize = 2;
const COMPACT_TOOL_EXAMPLE_CHAR_LIMIT: usize = 240;

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CompactToolContract {
    pub name: String,
    pub signature: String,
    pub returns: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub parameters: Vec<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub return_fields: Vec<serde_json::Value>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub description: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub examples: Vec<String>,
}

impl CompactToolContract {
    pub fn render_signature_head(&self) -> String {
        format!("{} -> {}", self.signature.trim(), self.returns.trim())
    }

    pub fn render_signature(&self) -> String {
        let mut sections = vec![self.render_signature_head()];
        let parameter_lines = self
            .parameters
            .iter()
            .filter_map(compact_doc_line)
            .collect::<Vec<_>>();
        if !parameter_lines.is_empty() {
            sections.push(format!("Parameters:\n{}", parameter_lines.join("\n")));
        }
        let return_field_lines = self
            .return_fields
            .iter()
            .filter_map(compact_doc_line)
            .collect::<Vec<_>>();
        if !return_field_lines.is_empty() {
            sections.push(format!("Return fields:\n{}", return_field_lines.join("\n")));
        }
        sections.join("\n")
    }

    pub fn render_returns(&self) -> String {
        let mut sections = Vec::new();
        let return_field_lines = self
            .return_fields
            .iter()
            .filter_map(compact_doc_line)
            .collect::<Vec<_>>();
        if !return_field_lines.is_empty() {
            sections.push(format!("Return fields:\n{}", return_field_lines.join("\n")));
        }
        sections.join("\n")
    }

    pub fn render_markdown(&self) -> String {
        let mut sections = vec![format!("### {}", self.render_signature_head())];
        if !self.description.trim().is_empty() {
            sections.push(self.description.trim().to_string());
        }
        if !self.parameters.is_empty() {
            sections.push(format!(
                "Parameters:\n{}",
                self.parameters
                    .iter()
                    .filter_map(compact_doc_line)
                    .collect::<Vec<_>>()
                    .join("\n")
            ));
        }
        if !self.return_fields.is_empty() {
            sections.push(format!(
                "Return fields:\n{}",
                self.return_fields
                    .iter()
                    .filter_map(compact_doc_line)
                    .collect::<Vec<_>>()
                    .join("\n")
            ));
        }
        if !self.examples.is_empty() {
            sections.push(format!("Examples: {}", self.examples.join("; ")));
        }
        sections.join("\n")
    }
}

impl ToolDefinition {
    pub fn raw(
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
            input_schema_projections: Vec::new(),
            output_schema_projections: Vec::new(),
            output_contract: ToolOutputContract::Static,
            examples: Vec::new(),
            availability: ToolAvailabilityConfig::showcased(),
            activation: ToolActivation::Always,
            availability_override: None,
            discovery: ToolDiscoveryMetadata::default(),
            execution_mode: ToolExecutionMode::Parallel,
        }
    }

    pub fn typed<Args, Output>(name: impl Into<String>, description: impl Into<String>) -> Self
    where
        Args: schemars::JsonSchema,
        Output: schemars::JsonSchema,
    {
        Self::raw(
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

    pub fn with_output_contract(mut self, output_contract: ToolOutputContract) -> Self {
        self.output_contract = output_contract;
        self
    }

    pub fn with_input_schema_projection(
        mut self,
        profile: impl Into<String>,
        schema: serde_json::Value,
    ) -> Self {
        let profile = profile.into();
        self.input_schema_projections
            .retain(|projection| projection.profile != profile);
        self.input_schema_projections
            .push(SchemaProjectionOverride { profile, schema });
        self
    }

    pub fn with_output_schema_projection(
        mut self,
        profile: impl Into<String>,
        schema: serde_json::Value,
    ) -> Self {
        let profile = profile.into();
        self.output_schema_projections
            .retain(|projection| projection.profile != profile);
        self.output_schema_projections
            .push(SchemaProjectionOverride { profile, schema });
        self
    }

    pub fn with_output_from_input_schema(
        self,
        input_field: impl Into<String>,
        default_schema: Option<serde_json::Value>,
    ) -> Self {
        self.with_output_contract(ToolOutputContract::from_input_schema(
            input_field,
            default_schema,
        ))
    }

    pub fn default_input_schema() -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {},
            "additionalProperties": true
        })
    }

    pub fn input_signature(&self) -> String {
        let params = self
            .parameter_docs()
            .into_iter()
            .map(|p| p.signature_fragment())
            .collect::<Vec<_>>();
        format!(
            "{}{}({})",
            self.name,
            self.output_contract
                .type_parameter_suffix()
                .unwrap_or_default(),
            params.join(", ")
        )
    }

    pub fn output_summary(&self) -> String {
        self.contract().output_summary()
    }

    pub fn signature(&self) -> String {
        format!("{} -> {}", self.input_signature(), self.output_summary())
    }

    pub fn compact_contract(&self) -> CompactToolContract {
        self.compact_contract_with_example_limit(COMPACT_TOOL_EXAMPLE_LIMIT)
    }

    pub fn compact_contract_with_example_limit(&self, example_limit: usize) -> CompactToolContract {
        self.contract()
            .compact_contract_with_example_limit(&self.manifest(), example_limit)
    }

    pub fn effective_availability(&self, mode: &ExecutionMode) -> ToolAvailability {
        self.availability_override
            .unwrap_or_else(|| self.availability.for_mode(mode))
    }

    pub fn model_tool(&self) -> ModelTool {
        self.contract().model_tool(&self.manifest())
    }

    pub fn manifest(&self) -> ToolManifest {
        ToolManifest {
            name: self.name.clone(),
            description: self.description.clone(),
            availability: self.availability.clone(),
            activation: self.activation,
            availability_override: self.availability_override,
            discovery: self.discovery.clone(),
            execution_mode: self.execution_mode,
        }
    }

    pub fn contract(&self) -> ToolContract {
        ToolContract {
            input_schema: self.input_schema.clone(),
            output_schema: self.output_schema.clone(),
            input_schema_projections: self.input_schema_projections.clone(),
            output_schema_projections: self.output_schema_projections.clone(),
            output_contract: self.output_contract.clone(),
            examples: self.examples.clone(),
        }
    }

    pub fn from_manifest_and_contract(manifest: ToolManifest, contract: ToolContract) -> Self {
        Self {
            name: manifest.name,
            description: manifest.description,
            input_schema: contract.input_schema,
            output_schema: contract.output_schema,
            input_schema_projections: contract.input_schema_projections,
            output_schema_projections: contract.output_schema_projections,
            output_contract: contract.output_contract,
            examples: contract.examples,
            availability: manifest.availability,
            activation: manifest.activation,
            availability_override: manifest.availability_override,
            discovery: manifest.discovery,
            execution_mode: manifest.execution_mode,
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
        self.parameter_docs()
            .into_iter()
            .map(|param| param.into_value())
            .collect()
    }

    fn parameter_docs(&self) -> Vec<ParameterDoc> {
        let mut params = schema_parameter_docs(&self.input_schema);
        self.output_contract
            .apply_type_witness_parameter(&mut params);
        params
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
        if !self.enum_values.is_empty() && !self.type_label.starts_with("enum[") {
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

    fn into_value(self) -> serde_json::Value {
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
        out.insert(
            "signature".to_string(),
            serde_json::json!(parameter_signature_from_value(&out)),
        );
        serde_json::Value::Object(out)
    }
}

#[derive(Clone, Debug, PartialEq)]
struct FieldDoc {
    path: String,
    type_label: String,
    required: bool,
    nullable: bool,
    description: Option<String>,
    enum_values: Vec<serde_json::Value>,
    minimum: Option<serde_json::Value>,
    maximum: Option<serde_json::Value>,
    min_length: Option<u64>,
    max_length: Option<u64>,
    min_items: Option<u64>,
    max_items: Option<u64>,
    item_type: Option<String>,
}

impl FieldDoc {
    fn from_schema(path: String, schema: &serde_json::Value, required: bool) -> Self {
        let (type_label, nullable) = schema_type_label_and_nullability(schema);
        Self {
            path,
            type_label,
            required,
            nullable,
            description: schema
                .get("description")
                .and_then(serde_json::Value::as_str)
                .map(str::to_string),
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

    fn into_value(self) -> serde_json::Value {
        let mut out = serde_json::Map::new();
        out.insert("path".to_string(), serde_json::json!(self.path));
        out.insert("type".to_string(), serde_json::json!(self.type_label));
        out.insert("required".to_string(), serde_json::json!(self.required));
        if self.nullable {
            out.insert("nullable".to_string(), serde_json::json!(true));
        }
        if let Some(description) = self.description.filter(|value| !value.trim().is_empty()) {
            out.insert("description".to_string(), serde_json::json!(description));
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
        out.insert(
            "signature".to_string(),
            serde_json::json!(field_signature_from_value(&out)),
        );
        serde_json::Value::Object(out)
    }
}

fn schema_parameter_docs(schema: &serde_json::Value) -> Vec<ParameterDoc> {
    let required_order = schema
        .get("required")
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(serde_json::Value::as_str)
        .collect::<Vec<_>>();
    let required = required_order
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    let Some(properties) = schema
        .get("properties")
        .and_then(serde_json::Value::as_object)
    else {
        return Vec::new();
    };
    let mut params = properties
        .iter()
        .map(|(name, schema)| parameter_doc(name, schema, required.contains(name.as_str())))
        .collect::<Vec<_>>();
    params.sort_by(|left, right| {
        match (
            required_order
                .iter()
                .position(|name| *name == left.name.as_str()),
            required_order
                .iter()
                .position(|name| *name == right.name.as_str()),
        ) {
            (Some(left), Some(right)) => left.cmp(&right),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => left.name.cmp(&right.name),
        }
    });
    params
}

fn return_field_metadata(schema: &serde_json::Value) -> Vec<serde_json::Value> {
    let mut fields = Vec::new();
    collect_return_fields("", schema, true, &mut fields);
    merge_return_fields(fields)
        .into_iter()
        .map(FieldDoc::into_value)
        .collect()
}

fn collect_return_fields(
    path: &str,
    schema: &serde_json::Value,
    required: bool,
    fields: &mut Vec<FieldDoc>,
) {
    if let Some(any_of) = schema
        .get("anyOf")
        .or_else(|| schema.get("oneOf"))
        .and_then(serde_json::Value::as_array)
    {
        if should_emit_return_field(path, schema) {
            fields.push(FieldDoc::from_schema(path.to_string(), schema, required));
        }
        for subschema in any_of {
            collect_return_fields(path, subschema, required, fields);
        }
        return;
    }

    let schema_type = schema
        .get("type")
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
        .or_else(|| {
            schema
                .get("type")
                .and_then(serde_json::Value::as_array)
                .and_then(|types| {
                    let non_null = types
                        .iter()
                        .filter_map(serde_json::Value::as_str)
                        .filter(|ty| *ty != "null")
                        .collect::<Vec<_>>();
                    if non_null.len() == 1 {
                        Some(non_null[0].to_string())
                    } else {
                        None
                    }
                })
        });

    match schema_type.as_deref() {
        Some("object") => {
            if should_emit_return_field(path, schema) {
                fields.push(FieldDoc::from_schema(path.to_string(), schema, required));
            }
            let required_properties = schema
                .get("required")
                .and_then(serde_json::Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(serde_json::Value::as_str)
                .collect::<std::collections::BTreeSet<_>>();
            if let Some(properties) = schema
                .get("properties")
                .and_then(serde_json::Value::as_object)
            {
                for (name, property_schema) in properties {
                    collect_return_fields(
                        &join_compact_path(path, name),
                        property_schema,
                        required_properties.contains(name.as_str()),
                        fields,
                    );
                }
            }
        }
        Some("array") => {
            if should_emit_return_field(path, schema) {
                fields.push(FieldDoc::from_schema(path.to_string(), schema, required));
            }
            if let Some(items) = schema.get("items") {
                collect_return_fields(&format!("{path}[]"), items, true, fields);
            }
        }
        _ => {
            if !path.is_empty() {
                fields.push(FieldDoc::from_schema(path.to_string(), schema, required));
            }
        }
    }
}

fn should_emit_return_field(path: &str, schema: &serde_json::Value) -> bool {
    !path.is_empty()
        && (schema
            .get("description")
            .and_then(serde_json::Value::as_str)
            .is_some_and(|value| !value.trim().is_empty())
            || schema.get("enum").is_some()
            || schema.get("minimum").is_some()
            || schema.get("maximum").is_some()
            || schema.get("minLength").is_some()
            || schema.get("maxLength").is_some()
            || schema.get("minItems").is_some()
            || schema.get("maxItems").is_some())
}

fn join_compact_path(parent: &str, child: &str) -> String {
    if parent.is_empty() {
        child.to_string()
    } else {
        format!("{parent}.{child}")
    }
}

fn merge_return_fields(fields: Vec<FieldDoc>) -> Vec<FieldDoc> {
    let mut merged = Vec::<FieldDoc>::new();
    for field in fields {
        if let Some(existing) = merged
            .iter_mut()
            .find(|existing| existing.path == field.path)
        {
            existing.merge(field);
        } else {
            merged.push(field);
        }
    }
    merged
}

impl FieldDoc {
    fn merge(&mut self, other: FieldDoc) {
        self.type_label = merge_type_labels(&self.type_label, &other.type_label);
        self.required |= other.required;
        self.nullable |= other.nullable || type_label_is_nullable(&other.type_label);
        if self.nullable && !type_label_is_nullable(&self.type_label) {
            self.type_label = merge_type_labels(&self.type_label, "null");
        }
        if self
            .description
            .as_deref()
            .is_none_or(|value| value.trim().is_empty())
        {
            self.description = other.description;
        }
        for value in other.enum_values {
            if !self.enum_values.iter().any(|existing| existing == &value) {
                self.enum_values.push(value);
            }
        }
        if self.minimum.is_none() {
            self.minimum = other.minimum;
        }
        if self.maximum.is_none() {
            self.maximum = other.maximum;
        }
        if self.min_length.is_none() {
            self.min_length = other.min_length;
        }
        if self.max_length.is_none() {
            self.max_length = other.max_length;
        }
        if self.min_items.is_none() {
            self.min_items = other.min_items;
        }
        if self.max_items.is_none() {
            self.max_items = other.max_items;
        }
        if self.item_type.is_none() {
            self.item_type = other.item_type;
        }
    }
}

fn merge_type_labels(left: &str, right: &str) -> String {
    let mut labels = Vec::<String>::new();
    for label in left.split(" | ").chain(right.split(" | ")) {
        let label = label.trim();
        if label.is_empty() || label == "any" && (!left.is_empty() || !right.is_empty()) {
            continue;
        }
        if !labels.iter().any(|existing| existing == label) {
            labels.push(label.to_string());
        }
    }
    if labels.is_empty() {
        return "any".to_string();
    }
    labels.sort_by(|left, right| match (*left == "null", *right == "null") {
        (true, false) => std::cmp::Ordering::Greater,
        (false, true) => std::cmp::Ordering::Less,
        _ => std::cmp::Ordering::Equal,
    });
    labels.join(" | ")
}

fn type_label_is_nullable(label: &str) -> bool {
    label.split(" | ").any(|part| part.trim() == "null")
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

fn compact_doc_line(value: &serde_json::Value) -> Option<String> {
    let signature = value.get("signature")?.as_str()?.trim();
    if signature.is_empty() {
        return None;
    }
    let description = value
        .get("description")
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty());
    Some(match description {
        Some(description) => format!("- `{signature}` — {description}"),
        None => format!("- `{signature}`"),
    })
}

fn parameter_signature_from_value(map: &serde_json::Map<String, serde_json::Value>) -> String {
    let name = map
        .get("name")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default();
    doc_signature_from_value(name, map)
}

fn field_signature_from_value(map: &serde_json::Map<String, serde_json::Value>) -> String {
    let path = map
        .get("path")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default();
    doc_signature_from_value(path, map)
}

fn doc_signature_from_value(
    name: &str,
    map: &serde_json::Map<String, serde_json::Value>,
) -> String {
    let ty = map
        .get("type")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("any");
    let required = map
        .get("required")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let mut out = if required {
        format!("{name}: {ty}")
    } else {
        format!("{name}?: {ty}")
    };

    let mut constraints = Vec::new();
    if let Some(values) = map.get("enum").and_then(serde_json::Value::as_array)
        && !ty.starts_with("enum[")
    {
        constraints.push(format!(
            "in {}",
            values
                .iter()
                .map(display_default_value)
                .collect::<Vec<_>>()
                .join("|")
        ));
    }
    if let Some(value) = map.get("minimum") {
        constraints.push(format!(">= {}", display_default_value(value)));
    }
    if let Some(value) = map.get("maximum") {
        constraints.push(format!("<= {}", display_default_value(value)));
    }
    if let Some(value) = map.get("min_length").and_then(serde_json::Value::as_u64) {
        constraints.push(format!("min_len {value}"));
    }
    if let Some(value) = map.get("max_length").and_then(serde_json::Value::as_u64) {
        constraints.push(format!("max_len {value}"));
    }
    if let Some(value) = map.get("min_items").and_then(serde_json::Value::as_u64) {
        constraints.push(format!("min_items {value}"));
    }
    if let Some(value) = map.get("max_items").and_then(serde_json::Value::as_u64) {
        constraints.push(format!("max_items {value}"));
    }
    if !constraints.is_empty() {
        out.push(' ');
        out.push_str(&constraints.join(" "));
    }
    if let Some(default) = map.get("default") {
        out.push_str(" = ");
        out.push_str(&display_default_value(default));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::ser::{Error as _, Serializer};

    #[test]
    fn tool_definition_uses_canonical_model_schemas() {
        let tool = ToolDefinition::raw(
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
    fn model_tool_preserves_schema_projection_overrides() {
        let tool = ToolDefinition::raw(
            "demo",
            "Demo",
            serde_json::json!({
                "type": "object",
                "properties": { "raw": { "const": "x" } }
            }),
            serde_json::json!({ "type": "object" }),
        )
        .with_input_schema_projection(
            "provider.tool_parameters",
            serde_json::json!({
                "type": "object",
                "properties": { "raw": { "type": "string", "enum": ["x"] } }
            }),
        )
        .with_output_schema_projection(
            "provider.structured_output",
            serde_json::json!({
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": false
            }),
        );

        let model_tool = tool.model_tool();
        assert_eq!(model_tool.input_schema["properties"]["raw"]["const"], "x");
        assert_eq!(
            model_tool.input_schema_projections[0].schema["properties"]["raw"]["enum"],
            serde_json::json!(["x"])
        );
        assert_eq!(
            model_tool.output_schema_projections[0].profile,
            "provider.structured_output"
        );
    }

    #[test]
    fn typed_tool_definition_generates_input_and_output_schema() {
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

        #[derive(schemars::JsonSchema)]
        #[allow(dead_code)]
        struct Output {
            answer: String,
            #[schemars(range(min = 0))]
            confidence: f32,
        }

        let tool = ToolDefinition::typed::<Args, Output>("demo", "Demo");
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
        assert_eq!(tool.output_schema["properties"]["answer"]["type"], "string");
        assert_eq!(
            tool.output_schema["properties"]["confidence"]["minimum"].as_f64(),
            Some(0.0)
        );
    }

    #[test]
    fn raw_tool_definition_preserves_caller_provided_schemas() {
        let input_schema = serde_json::json!({
            "type": "object",
            "properties": {
                "query": { "type": "string", "minLength": 3 }
            },
            "required": ["query"],
            "x-custom": { "keep": true }
        });
        let output_schema = serde_json::json!({
            "type": "object",
            "properties": {
                "ok": { "type": "boolean" }
            },
            "required": ["ok"],
            "x-result": ["exact"]
        });

        let tool = ToolDefinition::raw(
            "raw_demo",
            "Raw demo",
            input_schema.clone(),
            output_schema.clone(),
        );

        assert_eq!(tool.input_schema, input_schema);
        assert_eq!(tool.output_schema, output_schema);
    }

    #[test]
    fn compact_tool_contract_renders_prompt_and_search_shape_from_schemas() {
        let tool = ToolDefinition::raw(
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
            "search_docs(query: str, limit?: int <= 10 = 5)"
        );
        assert_eq!(
            contract.returns,
            "record{matches: list[str], next_page?: str | null}"
        );
        assert_eq!(
            contract.parameters,
            vec![
                serde_json::json!({
                    "name": "query",
                    "type": "str",
                    "required": true,
                    "signature": "query: str"
                }),
                serde_json::json!({
                    "name": "limit",
                    "type": "int",
                    "required": false,
                    "default": 5,
                    "maximum": 10,
                    "signature": "limit?: int <= 10 = 5"
                }),
            ]
        );
        assert_eq!(contract.examples.len(), 2);

        let docs = ToolDefinition::format_tool_docs(&[tool]);
        assert!(docs.contains(
            "### search_docs(query: str, limit?: int <= 10 = 5) -> record{matches: list[str], next_page?: str | null}"
        ));
        assert!(!docs.contains("Returns:"));
        assert!(docs.contains("Parameters:\n- `query: str`\n- `limit?: int <= 10 = 5`"));
        assert!(docs.contains(
            "Examples: search_docs(query=\"rust\"); search_docs(query=\"rust\", limit=3)"
        ));
    }

    #[test]
    fn static_output_contract_keeps_existing_compact_docs_and_serde_shape() {
        let tool = ToolDefinition::raw(
            "read_text",
            "Read text",
            ToolDefinition::default_input_schema(),
            serde_json::json!({ "type": "string" }),
        );
        let explicit_static = tool
            .clone()
            .with_output_contract(ToolOutputContract::Static);

        assert_eq!(
            ToolDefinition::format_tool_docs(std::slice::from_ref(&tool)),
            ToolDefinition::format_tool_docs(&[explicit_static])
        );
        assert_eq!(tool.compact_contract().returns, "str");

        let serialized = serde_json::to_value(&tool).expect("serialize");
        assert!(serialized.get("output_contract").is_none());
        let deserialized: ToolDefinition = serde_json::from_value(serialized).expect("deserialize");
        assert!(deserialized.output_contract.is_static());
    }

    #[test]
    fn dynamic_output_contract_renders_schema_from_input_without_return_fields() {
        let tool = ToolDefinition::raw(
            "spawn_agent",
            "Run a subagent",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "output": { "type": "object", "additionalProperties": true }
                }
            }),
            serde_json::json!({ "type": "object", "additionalProperties": true }),
        )
        .with_output_from_input_schema("output", None);

        let contract = tool.compact_contract();
        assert_eq!(
            contract.signature,
            "spawn_agent<T = any>(output?: TypeSpec<T>)"
        );
        assert_eq!(contract.returns, "T");
        assert!(contract.return_fields.is_empty());
        assert_eq!(contract.render_returns(), "");
        assert_eq!(
            ToolDefinition::format_tool_docs(&[tool]),
            "### spawn_agent<T = any>(output?: TypeSpec<T>) -> T\nRun a subagent\nParameters:\n- `output?: TypeSpec<T>`"
        );
    }

    #[test]
    fn dynamic_output_contract_renders_default_schema() {
        let tool = ToolDefinition::raw(
            "llm_query",
            "Run a lightweight LLM query",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "task": { "type": "string" },
                    "output": { "type": "object", "additionalProperties": true }
                },
                "required": ["task"]
            }),
            serde_json::json!({ "type": "object", "additionalProperties": true }),
        )
        .with_output_from_input_schema("output", Some(serde_json::json!({ "type": "string" })));

        let contract = tool.compact_contract();
        assert_eq!(
            contract.signature,
            "llm_query<T = str>(task: str, output?: TypeSpec<T>)"
        );
        assert_eq!(contract.returns, "T");
        assert!(contract.return_fields.is_empty());
        assert_eq!(contract.render_returns(), "");
    }

    #[test]
    fn json_schema_loaded_contract_matches_hardcoded_renderer() {
        let tool: ToolDefinition = serde_json::from_value(serde_json::json!({
            "name": "mcp__appworld__spotify_search_songs",
            "description": "[MCP appworld] Search for songs with a query.",
            "examples": ["search songs by genre"],
            "input_schema": {
                "type": "object",
                "properties": {
                    "access_token": {
                        "type": "string",
                        "description": "Access token obtained from spotify app login."
                    },
                    "genre": {
                        "type": ["string", "null"],
                        "description": "Only include songs from this genre.",
                        "default": null
                    },
                    "page_limit": {
                        "type": "integer",
                        "description": "Maximum number of songs to return.",
                        "minimum": 1,
                        "maximum": 20,
                        "default": 5
                    },
                    "sort_by": {
                        "type": ["string", "null"],
                        "description": "Field to sort by. Prefix with '-' for descending order.",
                        "default": null
                    }
                },
                "required": ["access_token"],
                "additionalProperties": false
            },
            "output_schema": {
                "anyOf": [
                    {
                        "type": "object",
                        "properties": {
                            "response": {
                                "type": "array",
                                "description": "Matched songs.",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "album_id": {
                                            "type": ["integer", "null"],
                                            "description": "Album identifier when the song belongs to an album."
                                        },
                                        "album_title": { "type": ["string", "null"] },
                                        "artists": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "id": { "type": "integer" },
                                                    "name": { "type": "string" }
                                                },
                                                "required": ["id", "name"]
                                            }
                                        },
                                        "duration": { "type": "integer" },
                                        "genre": { "type": "string" },
                                        "like_count": { "type": "integer" },
                                        "play_count": {
                                            "type": "integer",
                                            "description": "Number of times the song was played.",
                                            "minimum": 0
                                        },
                                        "rating": { "type": "number" },
                                        "release_date": {
                                            "type": "string",
                                            "description": "Song release date in YYYY-MM-DD format."
                                        },
                                        "song_id": {
                                            "type": "integer",
                                            "description": "Stable song identifier."
                                        },
                                        "title": {
                                            "type": "string",
                                            "description": "Song title."
                                        }
                                    },
                                    "required": [
                                        "album_id",
                                        "album_title",
                                        "artists",
                                        "duration",
                                        "genre",
                                        "like_count",
                                        "play_count",
                                        "rating",
                                        "release_date",
                                        "song_id",
                                        "title"
                                    ]
                                }
                            }
                        },
                        "required": ["response"]
                    },
                    {
                        "type": "object",
                        "properties": {
                            "response": {
                                "type": "object",
                                "properties": {
                                    "message": {
                                        "type": "string",
                                        "description": "Failure or status message."
                                    }
                                },
                                "required": ["message"]
                            }
                        },
                        "required": ["response"]
                    }
                ]
            }
        }))
        .unwrap();

        let contract = tool.compact_contract();
        assert_eq!(
            serde_json::to_value(&contract).unwrap(),
            serde_json::json!({
                "name": "mcp__appworld__spotify_search_songs",
                "signature": "mcp__appworld__spotify_search_songs(access_token: str, genre?: str | null = null, page_limit?: int >= 1 <= 20 = 5, sort_by?: str | null = null)",
                "returns": "record{response: list[record{album_id: int | null, album_title: str | null, artists: list[record{id: int, name: str}], duration: int, genre: str, like_count: int, play_count: int, rating: float, release_date: str, song_id: int, title: str}]} | record{response: record{message: str}}",
                "parameters": [
                    {
                        "name": "access_token",
                        "type": "str",
                        "required": true,
                        "description": "Access token obtained from spotify app login.",
                        "signature": "access_token: str"
                    },
                    {
                        "name": "genre",
                        "type": "str | null",
                        "required": false,
                        "nullable": true,
                        "description": "Only include songs from this genre.",
                        "default": null,
                        "signature": "genre?: str | null = null"
                    },
                    {
                        "name": "page_limit",
                        "type": "int",
                        "required": false,
                        "description": "Maximum number of songs to return.",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 20,
                        "signature": "page_limit?: int >= 1 <= 20 = 5"
                    },
                    {
                        "name": "sort_by",
                        "type": "str | null",
                        "required": false,
                        "nullable": true,
                        "description": "Field to sort by. Prefix with '-' for descending order.",
                        "default": null,
                        "signature": "sort_by?: str | null = null"
                    }
                ],
                "return_fields": [
                    {
                        "path": "response",
                        "type": "list[record]",
                        "required": true,
                        "description": "Matched songs.",
                        "items": "record",
                        "signature": "response: list[record]"
                    },
                    {
                        "path": "response[].album_id",
                        "type": "int | null",
                        "required": true,
                        "nullable": true,
                        "description": "Album identifier when the song belongs to an album.",
                        "signature": "response[].album_id: int | null"
                    },
                    {
                        "path": "response[].album_title",
                        "type": "str | null",
                        "required": true,
                        "nullable": true,
                        "signature": "response[].album_title: str | null"
                    },
                    {
                        "path": "response[].artists[].id",
                        "type": "int",
                        "required": true,
                        "signature": "response[].artists[].id: int"
                    },
                    {
                        "path": "response[].artists[].name",
                        "type": "str",
                        "required": true,
                        "signature": "response[].artists[].name: str"
                    },
                    {
                        "path": "response[].duration",
                        "type": "int",
                        "required": true,
                        "signature": "response[].duration: int"
                    },
                    {
                        "path": "response[].genre",
                        "type": "str",
                        "required": true,
                        "signature": "response[].genre: str"
                    },
                    {
                        "path": "response[].like_count",
                        "type": "int",
                        "required": true,
                        "signature": "response[].like_count: int"
                    },
                    {
                        "path": "response[].play_count",
                        "type": "int",
                        "required": true,
                        "description": "Number of times the song was played.",
                        "minimum": 0,
                        "signature": "response[].play_count: int >= 0"
                    },
                    {
                        "path": "response[].rating",
                        "type": "float",
                        "required": true,
                        "signature": "response[].rating: float"
                    },
                    {
                        "path": "response[].release_date",
                        "type": "str",
                        "required": true,
                        "description": "Song release date in YYYY-MM-DD format.",
                        "signature": "response[].release_date: str"
                    },
                    {
                        "path": "response[].song_id",
                        "type": "int",
                        "required": true,
                        "description": "Stable song identifier.",
                        "signature": "response[].song_id: int"
                    },
                    {
                        "path": "response[].title",
                        "type": "str",
                        "required": true,
                        "description": "Song title.",
                        "signature": "response[].title: str"
                    },
                    {
                        "path": "response.message",
                        "type": "str",
                        "required": true,
                        "description": "Failure or status message.",
                        "signature": "response.message: str"
                    }
                ],
                "description": "[MCP appworld] Search for songs with a query.",
                "examples": ["search songs by genre"]
            })
        );

        assert_eq!(
            contract.render_markdown(),
            "### mcp__appworld__spotify_search_songs(access_token: str, genre?: str | null = null, page_limit?: int >= 1 <= 20 = 5, sort_by?: str | null = null) -> record{response: list[record{album_id: int | null, album_title: str | null, artists: list[record{id: int, name: str}], duration: int, genre: str, like_count: int, play_count: int, rating: float, release_date: str, song_id: int, title: str}]} | record{response: record{message: str}}\n[MCP appworld] Search for songs with a query.\nParameters:\n- `access_token: str` — Access token obtained from spotify app login.\n- `genre?: str | null = null` — Only include songs from this genre.\n- `page_limit?: int >= 1 <= 20 = 5` — Maximum number of songs to return.\n- `sort_by?: str | null = null` — Field to sort by. Prefix with '-' for descending order.\nReturn fields:\n- `response: list[record]` — Matched songs.\n- `response[].album_id: int | null` — Album identifier when the song belongs to an album.\n- `response[].album_title: str | null`\n- `response[].artists[].id: int`\n- `response[].artists[].name: str`\n- `response[].duration: int`\n- `response[].genre: str`\n- `response[].like_count: int`\n- `response[].play_count: int >= 0` — Number of times the song was played.\n- `response[].rating: float`\n- `response[].release_date: str` — Song release date in YYYY-MM-DD format.\n- `response[].song_id: int` — Stable song identifier.\n- `response[].title: str` — Song title.\n- `response.message: str` — Failure or status message.\nExamples: search songs by genre"
        );
        assert_eq!(
            contract.render_signature(),
            "mcp__appworld__spotify_search_songs(access_token: str, genre?: str | null = null, page_limit?: int >= 1 <= 20 = 5, sort_by?: str | null = null) -> record{response: list[record{album_id: int | null, album_title: str | null, artists: list[record{id: int, name: str}], duration: int, genre: str, like_count: int, play_count: int, rating: float, release_date: str, song_id: int, title: str}]} | record{response: record{message: str}}\nParameters:\n- `access_token: str` — Access token obtained from spotify app login.\n- `genre?: str | null = null` — Only include songs from this genre.\n- `page_limit?: int >= 1 <= 20 = 5` — Maximum number of songs to return.\n- `sort_by?: str | null = null` — Field to sort by. Prefix with '-' for descending order.\nReturn fields:\n- `response: list[record]` — Matched songs.\n- `response[].album_id: int | null` — Album identifier when the song belongs to an album.\n- `response[].album_title: str | null`\n- `response[].artists[].id: int`\n- `response[].artists[].name: str`\n- `response[].duration: int`\n- `response[].genre: str`\n- `response[].like_count: int`\n- `response[].play_count: int >= 0` — Number of times the song was played.\n- `response[].rating: float`\n- `response[].release_date: str` — Song release date in YYYY-MM-DD format.\n- `response[].song_id: int` — Stable song identifier.\n- `response[].title: str` — Song title.\n- `response.message: str` — Failure or status message."
        );
        assert_eq!(
            contract.render_returns(),
            "Return fields:\n- `response: list[record]` — Matched songs.\n- `response[].album_id: int | null` — Album identifier when the song belongs to an album.\n- `response[].album_title: str | null`\n- `response[].artists[].id: int`\n- `response[].artists[].name: str`\n- `response[].duration: int`\n- `response[].genre: str`\n- `response[].like_count: int`\n- `response[].play_count: int >= 0` — Number of times the song was played.\n- `response[].rating: float`\n- `response[].release_date: str` — Song release date in YYYY-MM-DD format.\n- `response[].song_id: int` — Stable song identifier.\n- `response[].title: str` — Song title.\n- `response.message: str` — Failure or status message."
        );
    }

    #[test]
    fn json_schema_loaded_contract_merges_nullable_anyof_return_fields() {
        let tool: ToolDefinition = serde_json::from_value(serde_json::json!({
            "name": "mcp__appworld__spotify_show_album_library",
            "description": "[MCP appworld] Search or show a list of albums in your album library.",
            "examples": ["show album library"],
            "input_schema": {
                "type": "object",
                "properties": {
                    "access_token": {
                        "type": "string",
                        "description": "Access token obtained from spotify app login."
                    },
                    "page_index": {
                        "type": "integer",
                        "description": "The index of the page to return.",
                        "minimum": 0,
                        "default": 0
                    },
                    "page_limit": {
                        "type": "integer",
                        "description": "The maximum number of results to return per page.",
                        "minimum": 1,
                        "maximum": 20,
                        "default": 5
                    }
                },
                "required": ["access_token"]
            },
            "output_schema": {
                "type": "object",
                "properties": {
                    "response": {
                        "anyOf": [
                            {
                                "type": "array",
                                "description": "Albums in the user's library.",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "added_at": {
                                            "description": "When the album was added to the library.",
                                            "anyOf": [
                                                { "type": "string" },
                                                { "type": "null" }
                                            ]
                                        },
                                        "album_id": { "type": "integer" },
                                        "genre": {
                                            "type": "string",
                                            "description": "Album genre.",
                                            "minLength": 1
                                        },
                                        "song_ids": {
                                            "type": "array",
                                            "items": { "type": "integer" }
                                        },
                                        "title": {
                                            "type": "string",
                                            "minLength": 1
                                        }
                                    },
                                    "required": ["added_at", "album_id", "genre", "song_ids", "title"]
                                }
                            },
                            {
                                "type": "object",
                                "properties": {
                                    "message": {
                                        "type": "string",
                                        "description": "Failure or status message."
                                    }
                                },
                                "required": ["message"]
                            }
                        ]
                    }
                },
                "required": ["response"]
            }
        }))
        .unwrap();

        let contract = tool.compact_contract();
        assert_eq!(
            serde_json::to_value(&contract).unwrap(),
            serde_json::json!({
                "name": "mcp__appworld__spotify_show_album_library",
                "signature": "mcp__appworld__spotify_show_album_library(access_token: str, page_index?: int >= 0 = 0, page_limit?: int >= 1 <= 20 = 5)",
                "returns": "record{response: list[record{added_at: null | str, album_id: int, genre: str, song_ids: list[int], title: str}] | record{message: str}}",
                "parameters": [
                    {
                        "name": "access_token",
                        "type": "str",
                        "required": true,
                        "description": "Access token obtained from spotify app login.",
                        "signature": "access_token: str"
                    },
                    {
                        "name": "page_index",
                        "type": "int",
                        "required": false,
                        "description": "The index of the page to return.",
                        "default": 0,
                        "minimum": 0,
                        "signature": "page_index?: int >= 0 = 0"
                    },
                    {
                        "name": "page_limit",
                        "type": "int",
                        "required": false,
                        "description": "The maximum number of results to return per page.",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 20,
                        "signature": "page_limit?: int >= 1 <= 20 = 5"
                    }
                ],
                "return_fields": [
                    {
                        "path": "response",
                        "type": "list[record]",
                        "required": true,
                        "description": "Albums in the user's library.",
                        "items": "record",
                        "signature": "response: list[record]"
                    },
                    {
                        "path": "response[].added_at",
                        "type": "str | null",
                        "required": true,
                        "nullable": true,
                        "description": "When the album was added to the library.",
                        "signature": "response[].added_at: str | null"
                    },
                    {
                        "path": "response[].album_id",
                        "type": "int",
                        "required": true,
                        "signature": "response[].album_id: int"
                    },
                    {
                        "path": "response[].genre",
                        "type": "str",
                        "required": true,
                        "description": "Album genre.",
                        "min_length": 1,
                        "signature": "response[].genre: str min_len 1"
                    },
                    {
                        "path": "response[].song_ids[]",
                        "type": "int",
                        "required": true,
                        "signature": "response[].song_ids[]: int"
                    },
                    {
                        "path": "response[].title",
                        "type": "str",
                        "required": true,
                        "min_length": 1,
                        "signature": "response[].title: str min_len 1"
                    },
                    {
                        "path": "response.message",
                        "type": "str",
                        "required": true,
                        "description": "Failure or status message.",
                        "signature": "response.message: str"
                    }
                ],
                "description": "[MCP appworld] Search or show a list of albums in your album library.",
                "examples": ["show album library"]
            })
        );
        assert_eq!(
            contract.render_markdown(),
            "### mcp__appworld__spotify_show_album_library(access_token: str, page_index?: int >= 0 = 0, page_limit?: int >= 1 <= 20 = 5) -> record{response: list[record{added_at: null | str, album_id: int, genre: str, song_ids: list[int], title: str}] | record{message: str}}\n[MCP appworld] Search or show a list of albums in your album library.\nParameters:\n- `access_token: str` — Access token obtained from spotify app login.\n- `page_index?: int >= 0 = 0` — The index of the page to return.\n- `page_limit?: int >= 1 <= 20 = 5` — The maximum number of results to return per page.\nReturn fields:\n- `response: list[record]` — Albums in the user's library.\n- `response[].added_at: str | null` — When the album was added to the library.\n- `response[].album_id: int`\n- `response[].genre: str min_len 1` — Album genre.\n- `response[].song_ids[]: int`\n- `response[].title: str min_len 1`\n- `response.message: str` — Failure or status message.\nExamples: show album library"
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
        let mut with_metadata = ToolDefinition::raw(
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
    let fields = properties
        .iter()
        .map(|(name, field_schema)| {
            let suffix = if required.contains(name.as_str()) {
                ""
            } else {
                "?"
            };
            format!("{name}{suffix}: {}", compact_schema_label(field_schema))
        })
        .collect::<Vec<_>>();
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

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ToolImage {
    pub mime: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference: Option<AttachmentRef>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub data: Vec<u8>,
    pub label: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub width: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub height: Option<u32>,
}

#[derive(Clone, Debug)]
pub struct ToolResult {
    pub success: bool,
    pub result: serde_json::Value,
    pub images: Vec<ToolImage>,
    pub control: Option<ToolControl>,
}

impl ToolResult {
    pub fn ok(result: serde_json::Value) -> Self {
        Self {
            success: true,
            result,
            images: vec![],
            control: None,
        }
    }

    pub fn err(result: serde_json::Value) -> Self {
        Self {
            success: false,
            result,
            images: vec![],
            control: None,
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
            control: None,
        }
    }

    pub fn with_control(mut self, control: ToolControl) -> Self {
        self.control = Some(control);
        self
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
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolControl {
    Handoff { session_id: String },
    Finish { value: serde_json::Value },
    Fail { value: serde_json::Value },
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub control: Option<ToolControl>,
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
    pub execution_prompt: Arc<str>,
    pub tool_names: Arc<Vec<String>>,
    pub omitted_tool_count: usize,
    pub contributions: Arc<Vec<PromptContribution>>,
}

impl PromptContext {
    pub fn has_tool(&self, tool_name: &str) -> bool {
        self.tool_names.iter().any(|name| name == tool_name)
    }
}
