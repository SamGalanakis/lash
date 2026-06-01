/// How a tool's invocations should be scheduled relative to other tools in
/// the same batch of model-produced tool calls.
///
/// Tools that only *read* state (`read_file`, `grep`, `glob`, ...) can run
/// in parallel safely and should use the default [`ToolScheduling::Parallel`].
/// Tools that *mutate* shared state (`apply_patch`, `exec_command`,
/// `write_stdin`) should declare
/// [`ToolScheduling::Serial`] so the dispatcher runs them one-at-a-time
/// and avoids interleaving with each other.
///
/// This controls scheduling within a batch of tool calls; protocol ownership
/// is selected by the host plugin stack.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolScheduling {
    /// Safe to run concurrently with other parallel tools in the same batch.
    #[default]
    Parallel,
    /// Must run one-at-a-time relative to other serial tools in the batch.
    Serial,
}

fn default_tool_scheduling() -> ToolScheduling {
    ToolScheduling::default()
}

fn is_default_tool_scheduling(mode: &ToolScheduling) -> bool {
    *mode == ToolScheduling::default()
}

/// Automatic retry policy for a tool's execution.
///
/// This is intentionally separate from [`ToolScheduling`]: scheduling
/// decides whether different tool calls may run together, while retry policy
/// decides whether one failed call may be attempted again inside its scheduled
/// slot.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolRetryPolicy {
    /// Never retry automatically. This is the default for every tool.
    #[default]
    Never,
    /// Retry only failures that explicitly report a safe retry disposition.
    Safe {
        max_attempts: u32,
        base_delay_ms: u64,
        max_delay_ms: u64,
    },
    /// Retry only failures that explicitly report a safe retry disposition,
    /// and only when the runtime can provide a stable replay key.
    Idempotent {
        max_attempts: u32,
        base_delay_ms: u64,
        max_delay_ms: u64,
    },
}

impl ToolRetryPolicy {
    pub fn safe(max_attempts: u32, base_delay_ms: u64, max_delay_ms: u64) -> Self {
        Self::Safe {
            max_attempts,
            base_delay_ms,
            max_delay_ms,
        }
    }

    pub fn idempotent(max_attempts: u32, base_delay_ms: u64, max_delay_ms: u64) -> Self {
        Self::Idempotent {
            max_attempts,
            base_delay_ms,
            max_delay_ms,
        }
    }

    pub fn max_attempts(self) -> u32 {
        match self {
            Self::Never => 1,
            Self::Safe { max_attempts, .. } | Self::Idempotent { max_attempts, .. } => {
                max_attempts.max(1)
            }
        }
    }

    pub fn delay_ms_for_retry(self, retry_index: u32, requested_after_ms: Option<u64>) -> u64 {
        let (base_delay_ms, max_delay_ms) = match self {
            Self::Never => return 0,
            Self::Safe {
                base_delay_ms,
                max_delay_ms,
                ..
            }
            | Self::Idempotent {
                base_delay_ms,
                max_delay_ms,
                ..
            } => (base_delay_ms, max_delay_ms),
        };
        let multiplier = 1_u64.checked_shl(retry_index).unwrap_or(u64::MAX);
        let backoff = base_delay_ms.saturating_mul(multiplier);
        let delay = requested_after_ms.unwrap_or(backoff);
        if max_delay_ms == 0 {
            delay
        } else {
            delay.min(max_delay_ms)
        }
    }

    pub fn requires_replay_key(self) -> bool {
        matches!(self, Self::Idempotent { .. })
    }
}

fn default_tool_retry_policy() -> ToolRetryPolicy {
    ToolRetryPolicy::default()
}

fn is_default_tool_retry_policy(policy: &ToolRetryPolicy) -> bool {
    *policy == ToolRetryPolicy::default()
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
    pub base: ToolAvailability,
}

impl ToolAvailabilityConfig {
    pub fn same(availability: ToolAvailability) -> Self {
        Self { base: availability }
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

    pub fn base(&self) -> ToolAvailability {
        self.base
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
pub struct ToolAgentSurface {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub module_path: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub operation: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub authority_type: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub aliases: Vec<String>,
}

impl ToolAgentSurface {
    pub fn new(
        module_path: impl IntoIterator<Item = impl Into<String>>,
        operation: impl Into<String>,
    ) -> Self {
        Self {
            module_path: module_path.into_iter().map(Into::into).collect(),
            operation: Some(operation.into()),
            authority_type: None,
            aliases: Vec::new(),
        }
    }

    pub fn with_authority_type(mut self, authority_type: impl Into<String>) -> Self {
        self.authority_type = Some(authority_type.into());
        self
    }

    pub fn with_aliases(mut self, aliases: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.aliases = aliases.into_iter().map(Into::into).collect();
        self
    }

    pub fn executable_for(&self, tool_name: &str) -> ToolAgentExecutableSurface {
        let module_path = if self.module_path.is_empty() {
            vec!["tools".to_string()]
        } else {
            self.module_path.clone()
        };
        let operation = self
            .operation
            .as_deref()
            .filter(|operation| !operation.trim().is_empty())
            .unwrap_or(tool_name)
            .to_string();
        let authority_type = self
            .authority_type
            .as_deref()
            .filter(|authority_type| !authority_type.trim().is_empty())
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| default_authority_type(&module_path));
        ToolAgentExecutableSurface {
            module_path,
            operation,
            authority_type,
            aliases: self.aliases.clone(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.module_path.is_empty()
            && self.operation.is_none()
            && self.authority_type.is_none()
            && self.aliases.is_empty()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolAgentExecutableSurface {
    pub module_path: Vec<String>,
    pub operation: String,
    pub authority_type: String,
    pub aliases: Vec<String>,
}

impl ToolAgentExecutableSurface {
    pub fn module_path_string(&self) -> String {
        self.module_path.join(".")
    }

    pub fn call_path(&self) -> String {
        format!("{}.{}", self.module_path_string(), self.operation)
    }
}

fn default_authority_type(module_path: &[String]) -> String {
    let base = module_path
        .first()
        .map(String::as_str)
        .unwrap_or("tools")
        .trim_matches('_');
    let mut out = String::new();
    for part in base.split('_').filter(|part| !part.is_empty()) {
        let mut chars = part.chars();
        if let Some(first) = chars.next() {
            out.extend(first.to_uppercase());
            out.extend(chars);
        }
    }
    if out.is_empty() {
        "Tools".to_string()
    } else {
        out
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

#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ToolArgumentProjectionPolicy {
    #[default]
    MaterializeProjectedValues,
    PreserveProjectedRefsInField {
        field: String,
    },
}

impl ToolArgumentProjectionPolicy {
    pub fn preserve_projected_refs_in_field(field: impl Into<String>) -> Self {
        Self::PreserveProjectedRefsInField {
            field: field.into(),
        }
    }

    pub fn is_materialize_projected_values(&self) -> bool {
        matches!(self, Self::MaterializeProjectedValues)
    }
}

fn is_default_tool_argument_projection_policy(policy: &ToolArgumentProjectionPolicy) -> bool {
    policy.is_materialize_projected_values()
}

#[derive(
    Clone,
    Debug,
    Default,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    serde::Serialize,
    serde::Deserialize,
)]
#[serde(transparent)]
pub struct ToolId(String);

impl ToolId {
    pub fn new(id: impl Into<String>) -> Self {
        let id = id.into();
        assert!(!id.trim().is_empty(), "tool id must not be empty");
        Self(id)
    }

    pub fn default_for_name(name: &str) -> Self {
        Self::new(format!("tool:{name}"))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<String> for ToolId {
    fn from(id: String) -> Self {
        Self::new(id)
    }
}

impl From<&str> for ToolId {
    fn from(id: &str) -> Self {
        Self::new(id)
    }
}

impl std::fmt::Display for ToolId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Tool metadata exposed to prompts, catalogs, UI, and availability checks.
/// The optional compact contract is the catalog-facing projection of the
/// resolved contract; full schemas stay in [`ToolContract`].
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct ToolManifest {
    pub id: ToolId,
    pub name: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub description: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compact_contract: Option<CompactToolContract>,
    #[serde(default, skip_serializing_if = "is_default_tool_availability_config")]
    pub availability: ToolAvailabilityConfig,
    #[serde(default, skip_serializing_if = "is_default_tool_activation")]
    pub activation: ToolActivation,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub availability_override: Option<ToolAvailability>,
    #[serde(default, skip_serializing_if = "ToolAgentSurface::is_empty")]
    pub agent_surface: ToolAgentSurface,
    #[serde(
        default,
        skip_serializing_if = "is_default_tool_argument_projection_policy"
    )]
    pub argument_projection: ToolArgumentProjectionPolicy,
    #[serde(
        default = "default_tool_scheduling",
        skip_serializing_if = "is_default_tool_scheduling"
    )]
    pub scheduling: ToolScheduling,
    #[serde(
        default = "default_tool_retry_policy",
        skip_serializing_if = "is_default_tool_retry_policy"
    )]
    pub retry_policy: ToolRetryPolicy,
}

impl ToolManifest {
    pub fn effective_availability(&self) -> ToolAvailability {
        self.availability_override
            .unwrap_or_else(|| self.availability.base())
    }
}

/// Heavy tool contract resolved only when a prompt or call needs schemas/docs.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ToolContract {
    #[serde(default = "ToolContract::default_input_schema")]
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

impl Default for ToolContract {
    fn default() -> Self {
        Self {
            input_schema: Self::default_input_schema(),
            output_schema: serde_json::Value::Null,
            input_schema_projections: Vec::new(),
            output_schema_projections: Vec::new(),
            output_contract: ToolOutputContract::Static,
            examples: Vec::new(),
        }
    }
}

impl ToolContract {
    pub fn default_input_schema() -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {},
            "additionalProperties": true
        })
    }

    pub fn compact_contract(&self, manifest: &ToolManifest) -> CompactToolContract {
        self.compact_contract_with_example_limit(manifest, COMPACT_TOOL_EXAMPLE_LIMIT)
    }

    pub fn compact_contract_with_example_limit(
        &self,
        manifest: &ToolManifest,
        example_limit: usize,
    ) -> CompactToolContract {
        let agent_surface = manifest.agent_surface.executable_for(&manifest.name);
        CompactToolContract {
            name: agent_surface.call_path(),
            signature: self.input_signature(manifest),
            returns: self.output_summary(),
            parameters: self.parameter_metadata(),
            return_fields: self.output_contract.return_fields(&self.output_schema),
            description: manifest.description.trim().to_string(),
            examples: compact_examples(&self.examples, example_limit),
        }
    }

    pub fn input_signature(&self, manifest: &ToolManifest) -> String {
        let agent_surface = manifest.agent_surface.executable_for(&manifest.name);
        let params = self
            .parameter_docs()
            .into_iter()
            .map(|p| p.signature_fragment())
            .collect::<Vec<_>>();
        let body = if params.is_empty() {
            "{}".to_string()
        } else {
            format!("{{ {} }}", params.join(", "))
        };
        format!(
            "await {}{}({})?",
            agent_surface.call_path(),
            self.output_contract
                .type_parameter_suffix()
                .unwrap_or_default(),
            body
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
///
/// Composes the runtime [`ToolManifest`] and [`ToolContract`] projections. Both
/// are `#[serde(flatten)]`ed so the serialized JSON shape stays flat (and wire/
/// persistence compatible); the two structs have disjoint field names.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ToolDefinition {
    #[serde(flatten)]
    pub manifest: ToolManifest,
    #[serde(flatten)]
    pub contract: ToolContract,
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
    pub fn raw_with_id(
        id: impl Into<ToolId>,
        name: impl Into<String>,
        description: impl Into<String>,
        input_schema: serde_json::Value,
        output_schema: serde_json::Value,
    ) -> Self {
        Self {
            manifest: ToolManifest {
                id: id.into(),
                name: name.into(),
                description: description.into(),
                compact_contract: None,
                ..ToolManifest::default()
            },
            contract: ToolContract {
                input_schema,
                output_schema,
                ..ToolContract::default()
            },
        }
    }

    pub fn raw_named(
        name: impl Into<String>,
        description: impl Into<String>,
        input_schema: serde_json::Value,
        output_schema: serde_json::Value,
    ) -> Self {
        let name = name.into();
        Self::raw_with_id(
            ToolId::default_for_name(&name),
            name,
            description,
            input_schema,
            output_schema,
        )
    }

    pub fn typed_with_id<Args, Output>(
        id: impl Into<ToolId>,
        name: impl Into<String>,
        description: impl Into<String>,
    ) -> Self
    where
        Args: schemars::JsonSchema,
        Output: schemars::JsonSchema,
    {
        Self::raw_with_id(
            id,
            name,
            description,
            schema_for::<Args>(),
            schema_for::<Output>(),
        )
    }

    pub fn typed<Args, Output>(name: impl Into<String>, description: impl Into<String>) -> Self
    where
        Args: schemars::JsonSchema,
        Output: schemars::JsonSchema,
    {
        let name = name.into();
        Self::typed_with_id::<Args, Output>(ToolId::default_for_name(&name), name, description)
    }

    pub fn raw(
        id: impl Into<ToolId>,
        name: impl Into<String>,
        description: impl Into<String>,
        input_schema: serde_json::Value,
        output_schema: serde_json::Value,
    ) -> Self {
        Self::raw_with_id(id, name, description, input_schema, output_schema)
    }

    pub fn with_examples(mut self, examples: Vec<String>) -> Self {
        self.contract.examples = examples;
        self
    }

    pub fn with_availability(mut self, availability: ToolAvailabilityConfig) -> Self {
        self.manifest.availability = availability;
        self
    }

    pub fn with_activation(mut self, activation: ToolActivation) -> Self {
        self.manifest.activation = activation;
        self
    }

    pub fn with_agent_surface(mut self, agent_surface: ToolAgentSurface) -> Self {
        self.manifest.agent_surface = agent_surface;
        self
    }

    pub fn with_argument_projection(
        mut self,
        argument_projection: ToolArgumentProjectionPolicy,
    ) -> Self {
        self.manifest.argument_projection = argument_projection;
        self
    }

    pub fn with_scheduling(mut self, scheduling: ToolScheduling) -> Self {
        self.manifest.scheduling = scheduling;
        self
    }

    pub fn with_retry_policy(mut self, retry_policy: ToolRetryPolicy) -> Self {
        self.manifest.retry_policy = retry_policy;
        self
    }

    pub fn with_output_contract(mut self, output_contract: ToolOutputContract) -> Self {
        self.contract.output_contract = output_contract;
        self
    }

    pub fn with_input_schema_projection(
        mut self,
        profile: impl Into<String>,
        schema: serde_json::Value,
    ) -> Self {
        let profile = profile.into();
        self.contract
            .input_schema_projections
            .retain(|projection| projection.profile != profile);
        self.contract
            .input_schema_projections
            .push(SchemaProjectionOverride { profile, schema });
        self
    }

    pub fn with_output_schema_projection(
        mut self,
        profile: impl Into<String>,
        schema: serde_json::Value,
    ) -> Self {
        let profile = profile.into();
        self.contract
            .output_schema_projections
            .retain(|projection| projection.profile != profile);
        self.contract
            .output_schema_projections
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
        ToolContract::default_input_schema()
    }

    /// Tool identity. Read very widely, so exposed as a thin accessor over the
    /// composed [`ToolManifest`].
    pub fn id(&self) -> &ToolId {
        &self.manifest.id
    }

    /// Tool name. Read very widely, so exposed as a thin accessor.
    pub fn name(&self) -> &str {
        &self.manifest.name
    }

    /// Tool description. Read very widely, so exposed as a thin accessor.
    pub fn description(&self) -> &str {
        &self.manifest.description
    }

    pub fn input_signature(&self) -> String {
        self.contract.input_signature(&self.manifest)
    }

    pub fn output_summary(&self) -> String {
        self.contract.output_summary()
    }

    pub fn signature(&self) -> String {
        format!("{} -> {}", self.input_signature(), self.output_summary())
    }

    pub fn compact_contract(&self) -> CompactToolContract {
        self.compact_contract_with_example_limit(COMPACT_TOOL_EXAMPLE_LIMIT)
    }

    pub fn compact_contract_with_example_limit(&self, example_limit: usize) -> CompactToolContract {
        self.contract
            .compact_contract_with_example_limit(&self.manifest, example_limit)
    }

    pub fn effective_availability(&self) -> ToolAvailability {
        self.manifest.effective_availability()
    }

    pub fn model_tool(&self) -> ModelTool {
        self.contract.model_tool(&self.manifest)
    }

    /// Project the manifest, computing the catalog-facing compact contract from
    /// the resolved [`ToolContract`].
    pub fn manifest(&self) -> ToolManifest {
        let mut manifest = self.manifest.clone();
        manifest.compact_contract = Some(self.contract.compact_contract(&manifest));
        manifest
    }

    pub fn contract(&self) -> ToolContract {
        self.contract.clone()
    }

    /// Recompose a definition from its [`ToolManifest`] and [`ToolContract`]
    /// projections — the inverse of [`ToolDefinition::manifest`]/[`ToolDefinition::contract`].
    pub fn from_parts(manifest: ToolManifest, contract: ToolContract) -> Self {
        Self { manifest, contract }
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
        let mut params = schema_parameter_docs(&self.contract.input_schema);
        self.contract
            .output_contract
            .apply_type_witness_parameter(&mut params);
        params
    }
}

mod schema_docs;
pub use schema_docs::schema_for;
use schema_docs::{
    ParameterDoc, compact_doc_line, compact_examples, compact_schema_label, return_field_metadata,
    schema_parameter_docs,
};

mod schema_validation;
pub use schema_validation::{LashSchema, validate_tool_input};

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tool_definition_uses_canonical_model_schemas() {
        let tool = ToolDefinition::raw_with_id(
            "tool:mcp__demo__search",
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
    fn tool_retry_policy_defaults_to_never_and_is_omitted_from_manifest_json() {
        let tool = ToolDefinition::raw_with_id(
            "tool:demo",
            "demo",
            "Demo",
            ToolDefinition::default_input_schema(),
            serde_json::json!({ "type": "string" }),
        );

        assert_eq!(tool.manifest.retry_policy, ToolRetryPolicy::Never);
        let manifest = tool.manifest();
        assert_eq!(manifest.retry_policy, ToolRetryPolicy::Never);
        let encoded = serde_json::to_value(&manifest).expect("manifest json");
        assert!(encoded.get("retry_policy").is_none());
    }

    #[test]
    fn tool_retry_policy_propagates_through_manifest_and_definition_roundtrip() {
        let tool = ToolDefinition::raw_with_id(
            "tool:demo",
            "demo",
            "Demo",
            ToolDefinition::default_input_schema(),
            serde_json::json!({ "type": "string" }),
        )
        .with_retry_policy(ToolRetryPolicy::safe(3, 10, 100));

        let manifest = tool.manifest();
        assert_eq!(
            manifest.retry_policy,
            ToolRetryPolicy::Safe {
                max_attempts: 3,
                base_delay_ms: 10,
                max_delay_ms: 100,
            }
        );

        let roundtrip = ToolDefinition::from_parts(manifest, tool.contract());
        assert_eq!(roundtrip.manifest.retry_policy, tool.manifest.retry_policy);
        let encoded = serde_json::to_value(roundtrip.manifest()).expect("manifest json");
        assert_eq!(encoded["retry_policy"]["type"], serde_json::json!("safe"));
    }

    #[test]
    fn tool_argument_projection_defaults_to_materialize_and_is_omitted_from_manifest_json() {
        let tool = ToolDefinition::raw_with_id(
            "tool:demo",
            "demo",
            "Demo",
            ToolDefinition::default_input_schema(),
            serde_json::json!({ "type": "string" }),
        );

        assert_eq!(
            tool.manifest.argument_projection,
            ToolArgumentProjectionPolicy::MaterializeProjectedValues
        );
        let manifest = tool.manifest();
        assert_eq!(
            manifest.argument_projection,
            ToolArgumentProjectionPolicy::MaterializeProjectedValues
        );
        let encoded = serde_json::to_value(&manifest).expect("manifest json");
        assert!(encoded.get("argument_projection").is_none());
    }

    #[test]
    fn tool_argument_projection_propagates_through_manifest_and_definition_roundtrip() {
        let tool = ToolDefinition::raw_with_id(
            "tool:demo",
            "demo",
            "Demo",
            ToolDefinition::default_input_schema(),
            serde_json::json!({ "type": "string" }),
        )
        .with_argument_projection(
            ToolArgumentProjectionPolicy::preserve_projected_refs_in_field("seed"),
        );

        let manifest = tool.manifest();
        assert_eq!(
            manifest.argument_projection,
            tool.manifest.argument_projection
        );

        let roundtrip = ToolDefinition::from_parts(manifest, tool.contract());
        assert_eq!(
            roundtrip.manifest.argument_projection,
            tool.manifest.argument_projection
        );
        let encoded = serde_json::to_value(roundtrip.manifest()).expect("manifest json");
        assert_eq!(
            encoded["argument_projection"],
            serde_json::json!({
                "kind": "preserve_projected_refs_in_field",
                "field": "seed"
            })
        );
    }

    #[test]
    fn model_tool_preserves_schema_projection_overrides() {
        let tool = ToolDefinition::raw_with_id(
            "tool:demo",
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
        assert_eq!(
            tool.contract.output_schema["properties"]["answer"]["type"],
            "string"
        );
        assert_eq!(
            tool.contract.output_schema["properties"]["confidence"]["minimum"].as_f64(),
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

        let tool = ToolDefinition::raw_with_id(
            "tool:raw_demo",
            "raw_demo",
            "Raw demo",
            input_schema.clone(),
            output_schema.clone(),
        );

        assert_eq!(tool.contract.input_schema, input_schema);
        assert_eq!(tool.contract.output_schema, output_schema);
    }

    #[test]
    fn compact_tool_contract_renders_prompt_and_search_shape_from_schemas() {
        let tool = ToolDefinition::raw_with_id(
            "tool:search_docs",
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
            "await tools.search_docs({ query: \"rust\" })?".to_string(),
            "await tools.search_docs({ query: \"rust\", limit: 3 })?".to_string(),
            "await tools.search_docs({ query: \"ignored\" })?".to_string(),
        ]);

        let contract = tool.compact_contract();
        assert_eq!(
            contract.signature,
            "await tools.search_docs({ query: str, limit?: int <= 10 = 5 })?"
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
            "### await tools.search_docs({ query: str, limit?: int <= 10 = 5 })? -> record{matches: list[str], next_page?: str | null}"
        ));
        assert!(!docs.contains("Returns:"));
        assert!(docs.contains("Parameters:\n- `query: str`\n- `limit?: int <= 10 = 5`"));
        assert!(docs.contains(
            "Examples: await tools.search_docs({ query: \"rust\" })?; await tools.search_docs({ query: \"rust\", limit: 3 })?"
        ));
    }

    #[test]
    fn compact_tool_contract_resolves_local_refs_in_string_or_list_parameters() {
        let tool = ToolDefinition::raw_with_id(
            "tool:search_tools",
            "search_tools",
            "Search tools",
            serde_json::json!({
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "$defs": {
                    "ModuleFilter": {
                        "anyOf": [
                            { "type": "string" },
                            {
                                "type": "array",
                                "items": { "type": "string" }
                            }
                        ]
                    }
                },
                "type": "object",
                "properties": {
                    "query": { "type": "string" },
                    "module": {
                        "anyOf": [
                            { "$ref": "#/$defs/ModuleFilter" },
                            { "type": "null" }
                        ]
                    }
                },
                "required": ["query"]
            }),
            serde_json::json!({
                "type": "array",
                "items": { "type": "object" }
            }),
        );

        let signature = tool.compact_contract().render_signature();

        assert!(
            signature.contains("module?: str | list[str] | null"),
            "{signature}"
        );
        assert!(!signature.contains("module?: any"), "{signature}");
    }

    #[test]
    fn static_output_contract_keeps_existing_compact_docs_and_serde_shape() {
        let tool = ToolDefinition::raw_with_id(
            "tool:read_text",
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
        assert!(deserialized.contract.output_contract.is_static());
    }

    #[test]
    fn dynamic_output_contract_renders_schema_from_input_without_return_fields() {
        let tool = ToolDefinition::raw_with_id(
            "tool:spawn_agent",
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
        .with_agent_surface(ToolAgentSurface::new(["agents"], "spawn"))
        .with_output_from_input_schema("output", None);

        let contract = tool.compact_contract();
        assert_eq!(
            contract.signature,
            "await agents.spawn<T = any>({ output?: TypeSpec<T> })?"
        );
        assert_eq!(contract.returns, "T");
        assert!(contract.return_fields.is_empty());
        assert_eq!(contract.render_returns(), "");
        assert_eq!(
            ToolDefinition::format_tool_docs(&[tool]),
            "### await agents.spawn<T = any>({ output?: TypeSpec<T> })? -> T\nRun a subagent\nParameters:\n- `output?: TypeSpec<T>`"
        );
    }

    #[test]
    fn dynamic_output_contract_renders_default_schema() {
        let tool = ToolDefinition::raw_with_id(
            "tool:llm_query",
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
        .with_agent_surface(ToolAgentSurface::new(["llm"], "query"))
        .with_output_from_input_schema("output", Some(serde_json::json!({ "type": "string" })));

        let contract = tool.compact_contract();
        assert_eq!(
            contract.signature,
            "await llm.query<T = str>({ task: str, output?: TypeSpec<T> })?"
        );
        assert_eq!(contract.returns, "T");
        assert!(contract.return_fields.is_empty());
        assert_eq!(contract.render_returns(), "");
    }

    #[test]
    fn json_schema_loaded_contract_matches_hardcoded_renderer() {
        let tool: ToolDefinition = serde_json::from_value(serde_json::json!({
            "id": "tool:mcp__appworld__spotify_search_songs",
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
                "name": "tools.mcp__appworld__spotify_search_songs",
                "signature": "await tools.mcp__appworld__spotify_search_songs({ access_token: str, genre?: str | null = null, page_limit?: int >= 1 <= 20 = 5, sort_by?: str | null = null })?",
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
            "### await tools.mcp__appworld__spotify_search_songs({ access_token: str, genre?: str | null = null, page_limit?: int >= 1 <= 20 = 5, sort_by?: str | null = null })? -> record{response: list[record{album_id: int | null, album_title: str | null, artists: list[record{id: int, name: str}], duration: int, genre: str, like_count: int, play_count: int, rating: float, release_date: str, song_id: int, title: str}]} | record{response: record{message: str}}\n[MCP appworld] Search for songs with a query.\nParameters:\n- `access_token: str` — Access token obtained from spotify app login.\n- `genre?: str | null = null` — Only include songs from this genre.\n- `page_limit?: int >= 1 <= 20 = 5` — Maximum number of songs to return.\n- `sort_by?: str | null = null` — Field to sort by. Prefix with '-' for descending order.\nReturn fields:\n- `response: list[record]` — Matched songs.\n- `response[].album_id: int | null` — Album identifier when the song belongs to an album.\n- `response[].album_title: str | null`\n- `response[].artists[].id: int`\n- `response[].artists[].name: str`\n- `response[].duration: int`\n- `response[].genre: str`\n- `response[].like_count: int`\n- `response[].play_count: int >= 0` — Number of times the song was played.\n- `response[].rating: float`\n- `response[].release_date: str` — Song release date in YYYY-MM-DD format.\n- `response[].song_id: int` — Stable song identifier.\n- `response[].title: str` — Song title.\n- `response.message: str` — Failure or status message.\nExamples: search songs by genre"
        );
        assert_eq!(
            contract.render_signature(),
            "await tools.mcp__appworld__spotify_search_songs({ access_token: str, genre?: str | null = null, page_limit?: int >= 1 <= 20 = 5, sort_by?: str | null = null })? -> record{response: list[record{album_id: int | null, album_title: str | null, artists: list[record{id: int, name: str}], duration: int, genre: str, like_count: int, play_count: int, rating: float, release_date: str, song_id: int, title: str}]} | record{response: record{message: str}}\nParameters:\n- `access_token: str` — Access token obtained from spotify app login.\n- `genre?: str | null = null` — Only include songs from this genre.\n- `page_limit?: int >= 1 <= 20 = 5` — Maximum number of songs to return.\n- `sort_by?: str | null = null` — Field to sort by. Prefix with '-' for descending order.\nReturn fields:\n- `response: list[record]` — Matched songs.\n- `response[].album_id: int | null` — Album identifier when the song belongs to an album.\n- `response[].album_title: str | null`\n- `response[].artists[].id: int`\n- `response[].artists[].name: str`\n- `response[].duration: int`\n- `response[].genre: str`\n- `response[].like_count: int`\n- `response[].play_count: int >= 0` — Number of times the song was played.\n- `response[].rating: float`\n- `response[].release_date: str` — Song release date in YYYY-MM-DD format.\n- `response[].song_id: int` — Stable song identifier.\n- `response[].title: str` — Song title.\n- `response.message: str` — Failure or status message."
        );
        assert_eq!(
            contract.render_returns(),
            "Return fields:\n- `response: list[record]` — Matched songs.\n- `response[].album_id: int | null` — Album identifier when the song belongs to an album.\n- `response[].album_title: str | null`\n- `response[].artists[].id: int`\n- `response[].artists[].name: str`\n- `response[].duration: int`\n- `response[].genre: str`\n- `response[].like_count: int`\n- `response[].play_count: int >= 0` — Number of times the song was played.\n- `response[].rating: float`\n- `response[].release_date: str` — Song release date in YYYY-MM-DD format.\n- `response[].song_id: int` — Stable song identifier.\n- `response[].title: str` — Song title.\n- `response.message: str` — Failure or status message."
        );
    }

    #[test]
    fn json_schema_loaded_contract_merges_nullable_anyof_return_fields() {
        let tool: ToolDefinition = serde_json::from_value(serde_json::json!({
            "id": "tool:mcp__appworld__spotify_show_album_library",
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
                "name": "tools.mcp__appworld__spotify_show_album_library",
                "signature": "await tools.mcp__appworld__spotify_show_album_library({ access_token: str, page_index?: int >= 0 = 0, page_limit?: int >= 1 <= 20 = 5 })?",
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
            "### await tools.mcp__appworld__spotify_show_album_library({ access_token: str, page_index?: int >= 0 = 0, page_limit?: int >= 1 <= 20 = 5 })? -> record{response: list[record{added_at: null | str, album_id: int, genre: str, song_ids: list[int], title: str}] | record{message: str}}\n[MCP appworld] Search or show a list of albums in your album library.\nParameters:\n- `access_token: str` — Access token obtained from spotify app login.\n- `page_index?: int >= 0 = 0` — The index of the page to return.\n- `page_limit?: int >= 1 <= 20 = 5` — The maximum number of results to return per page.\nReturn fields:\n- `response: list[record]` — Albums in the user's library.\n- `response[].added_at: str | null` — When the album was added to the library.\n- `response[].album_id: int`\n- `response[].genre: str min_len 1` — Album genre.\n- `response[].song_ids[]: int`\n- `response[].title: str min_len 1`\n- `response.message: str` — Failure or status message.\nExamples: show album library"
        );
    }

    #[test]
    fn tool_agent_surface_serde_defaults_are_empty() {
        let tool: ToolDefinition = serde_json::from_value(serde_json::json!({
            "id": "tool:read_file",
            "name": "read_file",
            "description": "Read a file"
        }))
        .unwrap();
        assert!(tool.manifest.agent_surface.is_empty());
    }

    #[test]
    fn tool_agent_surface_controls_prompt_call_form() {
        let mut with_metadata = ToolDefinition::raw_with_id(
            "tool:read_file",
            "read_file",
            "Read a file",
            ToolDefinition::default_input_schema(),
            serde_json::json!({"type": "string"}),
        );
        with_metadata.manifest.agent_surface =
            ToolAgentSurface::new(["fs"], "read").with_aliases(["cat"]);

        assert!(
            ToolDefinition::format_tool_docs(&[with_metadata])
                .contains("### await fs.read({})? -> str")
        );
    }
}
