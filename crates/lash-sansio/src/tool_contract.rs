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
pub struct LashlangToolBinding {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub module_path: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub operation: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub authority_type: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub aliases: Vec<String>,
}

impl LashlangToolBinding {
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

    pub fn executable_for(&self, tool_name: &str) -> ResolvedLashlangToolBinding {
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
        ResolvedLashlangToolBinding {
            module_path,
            operation,
            authority_type,
            aliases: self.aliases.clone(),
        }
    }

    /// Resolve a remote-callable surface without applying local prompt fallbacks.
    ///
    /// Remote hosts must provide an explicit module path and operation so
    /// serialized tool grants have one canonical call path. This deliberately
    /// rejects the prompt-only conveniences used by [`Self::executable_for`],
    /// where an empty module path falls back to `tools` and an empty operation
    /// falls back to the flat tool name.
    pub fn required_for_remote(
        manifest: &ToolManifest,
    ) -> Result<ResolvedLashlangToolBinding, String> {
        manifest
            .lashlang_binding
            .required_executable_for_remote(&manifest.name)
    }

    pub fn required_executable_for_remote(
        &self,
        tool_name: &str,
    ) -> Result<ResolvedLashlangToolBinding, String> {
        if self.module_path.is_empty() {
            return Err(format!(
                "tool `{tool_name}` is missing an explicit remote module path"
            ));
        }
        if let Some(empty) = self.module_path.iter().find(|part| part.trim().is_empty()) {
            return Err(format!(
                "tool `{tool_name}` has an empty remote module path segment `{empty}`"
            ));
        }
        let Some(operation) = self
            .operation
            .as_deref()
            .map(str::trim)
            .filter(|operation| !operation.is_empty())
        else {
            return Err(format!(
                "tool `{tool_name}` is missing an explicit remote operation"
            ));
        };
        let authority_type = self
            .authority_type
            .as_deref()
            .filter(|authority_type| !authority_type.trim().is_empty())
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| default_authority_type(&self.module_path));
        Ok(ResolvedLashlangToolBinding {
            module_path: self.module_path.clone(),
            operation: operation.to_string(),
            authority_type,
            aliases: self.aliases.clone(),
        })
    }

    pub fn is_empty(&self) -> bool {
        self.module_path.is_empty()
            && self.operation.is_none()
            && self.authority_type.is_none()
            && self.aliases.is_empty()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedLashlangToolBinding {
    pub module_path: Vec<String>,
    pub operation: String,
    pub authority_type: String,
    pub aliases: Vec<String>,
}

impl ResolvedLashlangToolBinding {
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
    #[serde(default, skip_serializing_if = "LashlangToolBinding::is_empty")]
    pub lashlang_binding: LashlangToolBinding,
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
        let lashlang_binding = manifest.lashlang_binding.executable_for(&manifest.name);
        CompactToolContract {
            name: lashlang_binding.call_path(),
            signature: self.input_signature(manifest),
            returns: self.output_summary(),
            parameters: self.parameter_metadata(),
            return_fields: self.output_contract.return_fields(&self.output_schema),
            description: manifest.description.trim().to_string(),
            examples: compact_examples(&self.examples, example_limit),
        }
    }

    pub fn input_signature(&self, manifest: &ToolManifest) -> String {
        let lashlang_binding = manifest.lashlang_binding.executable_for(&manifest.name);
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
            lashlang_binding.call_path(),
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

    pub fn with_lashlang_binding(mut self, lashlang_binding: LashlangToolBinding) -> Self {
        self.manifest.lashlang_binding = lashlang_binding;
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

include!("tool_contract/tests.rs");
