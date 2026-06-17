use std::sync::{Arc, Mutex};

pub use lash_trace::{
    TraceLashlangChildExecution, TraceLashlangEdgeSelection, TraceLashlangExecutionEvent,
    TraceLashlangExecutionIdentity, TraceLashlangGraph, TraceLashlangGraphChildLink,
    TraceLashlangGraphEdge, TraceLashlangGraphNode, TraceLashlangGraphStore, TraceLashlangMap,
    TraceLashlangMapEdge, TraceLashlangMapNode, TraceLashlangNodeStatus, TraceLashlangStatus,
};
pub use lashlang::{
    CompiledProcessCache, DurabilityTier as LashlangDurabilityTier, InMemoryLashlangArtifactStore,
    LASH_TYPE_KEY, LashlangAbilities, LashlangArtifactStore, LashlangHostCatalog,
    LashlangHostEnvironment, LashlangLanguageFeatures,
};

pub const LASHLANG_ENGINE_KIND: &str = "lashlang";
pub const LASHLANG_TOOL_BINDING_KEY: &str = "lashlang.tool";
pub const LASHLANG_SURFACE_EXTENSION_ID: &str = "lashlang.surface";

#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct LashlangSurfaceContribution {
    pub abilities: LashlangAbilities,
    pub language_features: LashlangLanguageFeatures,
    pub resources: LashlangHostCatalog,
}

impl LashlangSurfaceContribution {
    pub fn new(
        abilities: LashlangAbilities,
        language_features: LashlangLanguageFeatures,
        resources: LashlangHostCatalog,
    ) -> Self {
        Self {
            abilities,
            language_features,
            resources,
        }
    }

    pub fn from_surface(surface: LashlangSurface) -> Self {
        Self {
            abilities: surface.abilities,
            language_features: surface.language_features,
            resources: surface.resources,
        }
    }
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
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| tool_name.replace('_', "."));
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

    pub fn required_for_remote(
        manifest: &lash_core::ToolManifest,
    ) -> Result<ResolvedLashlangToolBinding, String> {
        tool_lashlang_binding(manifest).required_executable_for_remote(&manifest.name)
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
        let operation = self
            .operation
            .as_deref()
            .filter(|operation| !operation.trim().is_empty())
            .ok_or_else(|| {
                format!("tool `{tool_name}` is missing an explicit remote operation name")
            })?;
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
    module_path
        .last()
        .map(|segment| {
            let mut chars = segment.chars();
            match chars.next() {
                Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
                None => "Tool".to_string(),
            }
        })
        .unwrap_or_else(|| "Tool".to_string())
}

pub fn tool_lashlang_binding(manifest: &lash_core::ToolManifest) -> LashlangToolBinding {
    manifest
        .bindings
        .get(LASHLANG_TOOL_BINDING_KEY)
        .cloned()
        .and_then(|value| serde_json::from_value(value).ok())
        .unwrap_or_default()
}

pub trait ToolManifestLashlangExt {
    fn lashlang_binding(&self) -> Result<Option<LashlangToolBinding>, serde_json::Error>;
}

impl ToolManifestLashlangExt for lash_core::ToolManifest {
    fn lashlang_binding(&self) -> Result<Option<LashlangToolBinding>, serde_json::Error> {
        self.bindings
            .get(LASHLANG_TOOL_BINDING_KEY)
            .cloned()
            .map(serde_json::from_value)
            .transpose()
    }
}

pub trait ToolDefinitionLashlangExt {
    fn with_lashlang_binding(self, lashlang_binding: LashlangToolBinding) -> Self;
}

impl ToolDefinitionLashlangExt for lash_core::ToolDefinition {
    fn with_lashlang_binding(mut self, lashlang_binding: LashlangToolBinding) -> Self {
        let value = serde_json::to_value(lashlang_binding)
            .expect("lashlang tool binding must serialize to JSON");
        self.manifest
            .bindings
            .insert(LASHLANG_TOOL_BINDING_KEY.to_string(), value);
        self
    }
}

pub trait RemoteToolGrantLashlangExt {
    fn with_lashlang_binding(self, lashlang_binding: LashlangToolBinding) -> Self;
    fn lashlang_binding(&self) -> Result<Option<LashlangToolBinding>, serde_json::Error>;
}

impl RemoteToolGrantLashlangExt for lash_remote_protocol::RemoteToolGrant {
    fn with_lashlang_binding(mut self, lashlang_binding: LashlangToolBinding) -> Self {
        let value = serde_json::to_value(lashlang_binding)
            .expect("lashlang tool binding must serialize to JSON");
        self.bindings
            .insert(LASHLANG_TOOL_BINDING_KEY.to_string(), value);
        self
    }

    fn lashlang_binding(&self) -> Result<Option<LashlangToolBinding>, serde_json::Error> {
        self.bindings
            .get(LASHLANG_TOOL_BINDING_KEY)
            .cloned()
            .map(serde_json::from_value)
            .transpose()
    }
}

#[derive(Clone, Debug)]
pub struct LashlangSurface {
    pub abilities: LashlangAbilities,
    pub language_features: LashlangLanguageFeatures,
    pub resources: LashlangHostCatalog,
}

impl Default for LashlangSurface {
    fn default() -> Self {
        Self {
            abilities: LashlangAbilities::default().with_sleep(),
            language_features: LashlangLanguageFeatures::default(),
            resources: LashlangHostCatalog::new(),
        }
    }
}

impl LashlangSurface {
    pub fn new(
        abilities: LashlangAbilities,
        language_features: LashlangLanguageFeatures,
        resources: LashlangHostCatalog,
    ) -> Self {
        Self {
            abilities,
            language_features,
            resources,
        }
    }

    pub fn for_process_registry(mut self, process_registry_available: bool) -> Self {
        self.abilities = self.abilities.with_sleep();
        if process_registry_available {
            self.abilities = self.abilities.with_processes().with_process_signals();
        } else {
            self.abilities.processes = false;
            self.abilities.process_signals = false;
        }
        self
    }

    pub fn with_resources(mut self, resources: LashlangHostCatalog) -> Self {
        self.resources.extend(resources);
        self
    }

    pub fn with_plugin_extensions(
        mut self,
        extensions: &lash_core::PluginExtensions,
    ) -> Result<Self, String> {
        for payload in extensions.payloads(LASHLANG_SURFACE_EXTENSION_ID) {
            let contribution: LashlangSurfaceContribution = serde_json::from_value(payload.clone())
                .map_err(|err| {
                    format!("invalid `{LASHLANG_SURFACE_EXTENSION_ID}` extension payload: {err}")
                })?;
            self.abilities = self.abilities.union(contribution.abilities);
            self.language_features = self.language_features.union(contribution.language_features);
            self.resources.extend(contribution.resources);
        }
        Ok(self)
    }

    pub fn host_environment(&self, catalog: &lash_core::ToolCatalog) -> LashlangHostEnvironment {
        lashlang_host_environment_from_tool_catalog(
            catalog,
            self.abilities,
            self.language_features,
            self.resources.clone(),
        )
    }
}

pub fn lashlang_host_environment_from_tool_catalog(
    catalog: &lash_core::ToolCatalog,
    abilities: LashlangAbilities,
    language_features: LashlangLanguageFeatures,
    host_resources: LashlangHostCatalog,
) -> LashlangHostEnvironment {
    let mut resources = lashlang_resources_from_tool_catalog(catalog);
    resources.extend(host_resources);
    if abilities.triggers {
        lashlang::add_trigger_resource_operations(&mut resources);
    }
    LashlangHostEnvironment::new(resources, abilities).with_language_features(language_features)
}

pub fn lashlang_resources_from_tool_catalog(
    catalog: &lash_core::ToolCatalog,
) -> LashlangHostCatalog {
    let mut host_catalog = LashlangHostCatalog::new();
    for entry in catalog.tools.iter() {
        if entry.availability.is_callable() {
            let lashlang_binding =
                tool_lashlang_binding(&entry.manifest).executable_for(&entry.manifest.name);
            host_catalog.add_module_operation(
                lashlang_binding.module_path.iter().map(String::as_str),
                lashlang_binding.authority_type.clone(),
                lashlang_binding.operation.clone(),
                entry.manifest.name.clone(),
                lashlang::TypeExpr::Any,
                lashlang::TypeExpr::Any,
            );
        }
    }
    host_catalog
}

pub fn lashlang_host_environment_satisfies_requirements(
    required: &lashlang::HostRequirements,
    current: &LashlangHostEnvironment,
) -> Result<(), String> {
    let abilities = required.abilities;
    let current_abilities = current.abilities;
    if abilities.processes && !current_abilities.processes {
        return Err("processes are not available".to_string());
    }
    if abilities.sleep && !current_abilities.sleep {
        return Err("sleep is not available".to_string());
    }
    if abilities.process_signals && !current_abilities.process_signals {
        return Err("process signals are not available".to_string());
    }
    if abilities.triggers && !current_abilities.triggers {
        return Err("triggers are not available".to_string());
    }
    if required.language_features.label_annotations && !current.language_features.label_annotations
    {
        return Err("label annotations are not available".to_string());
    }

    for (_, module) in required.resources.module_instances() {
        let current_module = current
            .resources
            .resolve_module_path(&module.path)
            .ok_or_else(|| format!("module `{}` is not available", module.alias))?;
        if current_module.resource_type != module.resource_type {
            return Err(format!(
                "module `{}` has type `{}`, expected `{}`",
                module.alias, current_module.resource_type, module.resource_type
            ));
        }
        for (operation, required_binding) in &module.operations {
            match current.resources.resolve_module_operation(
                &module.resource_type,
                &module.alias,
                operation,
            ) {
                Some(current_binding) if current_binding == required_binding => {}
                Some(current_binding) => {
                    return Err(format!(
                        "module `{}` operation `{operation}` resolves to `{}`, expected `{}`",
                        module.alias,
                        current_binding.host_operation,
                        required_binding.host_operation
                    ));
                }
                None => {
                    return Err(format!(
                        "module `{}` does not expose operation `{operation}`",
                        module.alias
                    ));
                }
            }
        }
    }

    for (resource_type, required_type) in required.resources.resource_types() {
        if !current.resources.has_resource_type(resource_type) {
            return Err(format!("resource type `{resource_type}` is not available"));
        }
        for (operation, required_binding) in &required_type.operations {
            let current_binding = current
                .resources
                .resolve_operation(resource_type, operation)
                .ok_or_else(|| {
                    format!(
                        "resource type `{resource_type}` does not expose operation `{operation}`"
                    )
                })?;
            if current_binding.input_ty != required_binding.input_ty {
                return Err(format!(
                    "resource type `{resource_type}` operation `{operation}` has incompatible input type"
                ));
            }
            if current_binding.output_ty != required_binding.output_ty {
                return Err(format!(
                    "resource type `{resource_type}` operation `{operation}` has incompatible output type"
                ));
            }
        }
    }
    for (name, required_data_type) in required.resources.named_data_types() {
        let current_data_type = current
            .resources
            .resolve_named_data_type(name)
            .ok_or_else(|| format!("host data type `{name}` is not available"))?;
        if current_data_type != required_data_type {
            return Err(format!(
                "host data type `{name}` has incompatible structure"
            ));
        }
    }
    for (path, required_binding) in required.resources.value_constructors() {
        let current_binding = current
            .resources
            .resolve_value_constructor(&path.split('.').collect::<Vec<_>>())
            .ok_or_else(|| format!("value constructor `{path}` is not available"))?;
        if current_binding.input_ty != required_binding.input_ty {
            return Err(format!(
                "value constructor `{path}` has incompatible input type"
            ));
        }
        if current_binding.output_ty != required_binding.output_ty {
            return Err(format!(
                "value constructor `{path}` has incompatible output type"
            ));
        }
    }
    for (source_ty, required_binding) in required.resources.trigger_sources() {
        let current_binding = current
            .resources
            .resolve_trigger_source(source_ty)
            .ok_or_else(|| format!("trigger source type `{source_ty}` is not available"))?;
        if current_binding != required_binding {
            return Err(format!(
                "trigger source type `{source_ty}` has incompatible event type"
            ));
        }
    }

    Ok(())
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct LashlangProcessInput {
    pub module_ref: lashlang::ModuleRef,
    pub process_ref: lashlang::ProcessRef,
    pub host_requirements_ref: lashlang::HostRequirementsRef,
    pub process_name: String,
    #[serde(default)]
    pub args: serde_json::Map<String, serde_json::Value>,
}

impl LashlangProcessInput {
    pub fn process_identity(&self) -> lash_core::ProcessIdentity {
        lashlang_process_identity(self)
    }

    pub fn remote_identity(&self) -> lash_remote_protocol::RemoteProcessIdentity {
        lash_remote_protocol::RemoteProcessIdentity {
            kind: LASHLANG_ENGINE_KIND.to_string(),
            label: Some(self.process_name.clone()),
            definition: Some(lash_remote_protocol::RemoteProcessDefinitionIdentity {
                value: self.definition(),
            }),
        }
    }

    pub fn to_process_input(&self) -> Result<lash_core::ProcessInput, serde_json::Error> {
        Ok(lash_core::ProcessInput::Engine {
            kind: LASHLANG_ENGINE_KIND.to_string(),
            payload: serde_json::to_value(self)?,
        })
    }

    pub fn into_process_input(self) -> Result<lash_core::ProcessInput, serde_json::Error> {
        self.to_process_input()
    }

    pub fn remote_trigger_subscription_draft(
        &self,
        registrant: lash_remote_protocol::RemoteProcessOriginator,
        env_ref: lash_remote_protocol::RemoteProcessExecutionEnvRef,
        source_type: impl Into<String>,
        source_key: impl Into<String>,
    ) -> Result<lash_remote_protocol::RemoteTriggerSubscriptionDraft, serde_json::Error> {
        Ok(
            lash_remote_protocol::RemoteTriggerSubscriptionDraft::for_process(
                registrant,
                env_ref,
                source_type,
                source_key,
                self.clone().try_into()?,
                self.remote_identity(),
            ),
        )
    }

    pub fn from_payload(payload: serde_json::Value) -> Result<Self, serde_json::Error> {
        serde_json::from_value(payload)
    }

    pub fn definition(&self) -> serde_json::Value {
        serde_json::json!({
            "module_ref": self.module_ref,
            "process_ref": self.process_ref,
            "host_requirements_ref": self.host_requirements_ref,
            "process_name": self.process_name,
        })
    }
}

impl TryFrom<LashlangProcessInput> for lash_remote_protocol::RemoteProcessInput {
    type Error = serde_json::Error;

    fn try_from(value: LashlangProcessInput) -> Result<Self, Self::Error> {
        Ok(Self::Engine {
            kind: LASHLANG_ENGINE_KIND.to_string(),
            payload: serde_json::to_value(value)?,
        })
    }
}

#[derive(Clone, Debug)]
pub struct PreparedLashlangProcessStart {
    pub registration: lash_core::ProcessRegistration,
    pub label: Option<String>,
}

pub async fn prepare_lashlang_process_start(
    artifact_store: Arc<dyn LashlangArtifactStore>,
    start: lashlang::ProcessStart,
) -> Result<PreparedLashlangProcessStart, String> {
    let display_name = Some(start.process_name.clone());
    let artifact = artifact_store
        .get_module_artifact(&start.module_ref)
        .await
        .map_err(|err| format!("failed to load lashlang module artifact: {err}"))?
        .ok_or_else(|| {
            format!(
                "missing lashlang module artifact `{}` for process `{}`",
                start.module_ref, start.process_name
            )
        })?;
    if artifact.host_requirements_ref != start.host_requirements_ref {
        return Err(format!(
            "lashlang module artifact `{}` host requirements mismatch: process requested {}, artifact has {}",
            start.module_ref, start.host_requirements_ref, artifact.host_requirements_ref
        ));
    }
    if artifact.process_ref(&start.process_name) != Some(&start.process_ref) {
        return Err(format!(
            "lashlang module artifact `{}` does not export process `{}` as requested ref {:?}",
            start.module_ref, start.process_name, start.process_ref
        ));
    }
    let args = match serde_json::to_value(lashlang::Value::Record(Arc::new(start.args)))
        .map_err(|err| format!("failed to serialize process args: {err}"))?
    {
        serde_json::Value::Object(map) => map,
        _ => return Err("process args must serialize as a record".to_string()),
    };
    let signal_event_types = artifact
        .canonical_ir
        .process(&start.process_name)
        .map(lashlang_process_signal_event_types)
        .unwrap_or_default();
    let process_input = LashlangProcessInput {
        module_ref: start.module_ref,
        process_ref: start.process_ref,
        host_requirements_ref: start.host_requirements_ref,
        process_name: start.process_name,
        args,
    };
    let identity = lashlang_process_identity(&process_input);
    let process_input = process_input
        .into_process_input()
        .map_err(|err| format!("failed to encode process input: {err}"))?;
    let process_id = format!("process:{}", uuid::Uuid::new_v4());
    let registration = lash_core::ProcessRegistration::new(
        process_id,
        process_input,
        lash_core::ProcessProvenance::host(),
    )
    .with_identity(identity)
    .with_extra_event_types(
        lashlang_process_event_types()
            .into_iter()
            .chain(signal_event_types),
    );
    Ok(PreparedLashlangProcessStart {
        registration,
        label: display_name,
    })
}

pub fn resolve_lashlang_module_operation(
    host_environment: &lashlang::LashlangHostEnvironment,
    receiver: &lashlang::ResourceHandle,
    operation: &str,
) -> Result<String, lashlang::ExecutionHostError> {
    host_environment
        .resources
        .resolve_module_operation(&receiver.resource_type, &receiver.alias, operation)
        .map(|binding| binding.host_operation.clone())
        .ok_or_else(|| {
            lashlang::ExecutionHostError::new(format!(
                "module `{}` of type `{}` does not expose operation `{operation}`",
                receiver.alias, receiver.resource_type
            ))
        })
}

fn lashlang_process_identity(input: &LashlangProcessInput) -> lash_core::ProcessIdentity {
    lash_core::ProcessIdentity::new(LASHLANG_ENGINE_KIND)
        .with_label(Some(input.process_name.clone()))
        .with_definition(Some(input.definition()))
}

#[derive(Clone)]
pub struct LashlangProcessEngine {
    artifact_store: Arc<dyn LashlangArtifactStore>,
    process_cache: Arc<Mutex<CompiledProcessCache>>,
    surface: LashlangSurface,
    execution_sink: Option<Arc<dyn lash_trace::TraceSink>>,
    trace_context: lash_trace::TraceContext,
}

impl LashlangProcessEngine {
    pub fn new(artifact_store: Arc<dyn LashlangArtifactStore>, surface: LashlangSurface) -> Self {
        Self {
            artifact_store,
            process_cache: Arc::new(Mutex::new(CompiledProcessCache::new())),
            surface,
            execution_sink: None,
            trace_context: lash_trace::TraceContext::default(),
        }
    }

    pub fn in_memory(surface: LashlangSurface) -> Self {
        Self::new(
            lashlang::global_in_memory_lashlang_artifact_store(),
            surface,
        )
    }

    pub fn with_execution_trace(
        mut self,
        sink: Option<Arc<dyn lash_trace::TraceSink>>,
        trace_context: lash_trace::TraceContext,
    ) -> Self {
        self.execution_sink = sink;
        self.trace_context = trace_context;
        self
    }

    pub fn artifact_store(&self) -> Arc<dyn LashlangArtifactStore> {
        Arc::clone(&self.artifact_store)
    }
}

#[async_trait::async_trait]
impl lash_core::ProcessEngine for LashlangProcessEngine {
    fn kind(&self) -> &'static str {
        LASHLANG_ENGINE_KIND
    }

    async fn validate_start(
        &self,
        context: lash_core::ProcessEngineValidationContext<'_>,
        payload: &serde_json::Value,
        _env_spec: Option<&lash_core::ProcessExecutionEnvSpec>,
    ) -> Result<(), lash_core::PluginError> {
        let input: LashlangProcessInput =
            serde_json::from_value(payload.clone()).map_err(|err| {
                lash_core::PluginError::Session(format!("invalid lashlang process payload: {err}"))
            })?;
        let artifact = self
            .artifact_store
            .get_module_artifact(&input.module_ref)
            .await
            .map_err(|err| lash_core::PluginError::Session(format!("load module artifact: {err}")))?
            .ok_or_else(|| {
                lash_core::PluginError::Session(format!(
                    "missing lashlang module artifact `{}`",
                    input.module_ref
                ))
            })?;
        if artifact.host_requirements_ref != input.host_requirements_ref {
            return Err(lash_core::PluginError::Session(format!(
                "lashlang process `{}` requested surface {}, artifact has {}",
                input.process_name, input.host_requirements_ref, artifact.host_requirements_ref
            )));
        }
        if artifact.process_ref(&input.process_name) != Some(&input.process_ref) {
            return Err(lash_core::PluginError::Session(format!(
                "lashlang module `{}` does not export process `{}` as requested ref {:?}",
                input.module_ref, input.process_name, input.process_ref
            )));
        }
        let surface = self
            .surface
            .clone()
            .for_process_registry(context.process_registry_available());
        let host_environment = surface.host_environment(context.tool_catalog());
        if let Err(err) = lashlang_host_environment_satisfies_requirements(
            &artifact.host_requirements,
            &host_environment,
        ) {
            return Err(lash_core::PluginError::Session(format!(
                "lashlang process `{}` is incompatible with this host surface: {err}",
                input.process_name
            )));
        }
        Ok(())
    }

    async fn run(
        &self,
        context: lash_core::ProcessEngineRunContext<'_>,
        payload: serde_json::Value,
    ) -> lash_core::ProcessAwaitOutput {
        process::run_lashlang_process(self.clone(), context, payload).await
    }

    fn identity(&self, payload: &serde_json::Value) -> lash_core::ProcessIdentity {
        match LashlangProcessInput::from_payload(payload.clone()) {
            Ok(input) => lashlang_process_identity(&input),
            Err(_) => lash_core::ProcessIdentity::new(LASHLANG_ENGINE_KIND),
        }
    }
}

mod bridge;
mod process;

pub use bridge::{
    lashlang_value_to_json, process_event_payload, protocol_tool_output_to_lashlang_value,
    protocol_tool_reply_to_lashlang_value, sleep_duration_ms,
};
pub use process::{
    lashlang_process_event_types, lashlang_process_signal_event_types, lashlang_type_expr_schema,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn process_input_serializes_as_generic_engine_payload() {
        let hash = lashlang::ContentHash::new("abc123");
        let input = LashlangProcessInput {
            module_ref: lashlang::ModuleRef::new(&hash),
            process_ref: lashlang::ProcessRef::new(hash.clone(), 7),
            host_requirements_ref: lashlang::HostRequirementsRef::new(&hash),
            process_name: "main".to_string(),
            args: serde_json::Map::from_iter([("prompt".to_string(), serde_json::json!("go"))]),
        };

        let process_input = input
            .clone()
            .into_process_input()
            .expect("lashlang process input serializes");

        let lash_core::ProcessInput::Engine { kind, payload } = process_input else {
            panic!("lashlang runtime must use the generic engine process input");
        };
        assert_eq!(kind, LASHLANG_ENGINE_KIND);
        assert_eq!(
            LashlangProcessInput::from_payload(payload)
                .expect("engine payload decodes")
                .process_name,
            input.process_name
        );
    }

    #[test]
    fn process_input_remote_helpers_use_generic_engine_and_identity() {
        let hash = lashlang::ContentHash::new("abc123");
        let input = LashlangProcessInput {
            module_ref: lashlang::ModuleRef::new(&hash),
            process_ref: lashlang::ProcessRef::new(hash.clone(), 7),
            host_requirements_ref: lashlang::HostRequirementsRef::new(&hash),
            process_name: "main".to_string(),
            args: serde_json::Map::from_iter([("prompt".to_string(), serde_json::json!("go"))]),
        };

        let remote_input: lash_remote_protocol::RemoteProcessInput = input
            .clone()
            .try_into()
            .expect("lashlang process input serializes remotely");
        let lash_remote_protocol::RemoteProcessInput::Engine { kind, payload } = remote_input
        else {
            panic!("lashlang runtime must use the generic remote engine process input");
        };
        assert_eq!(kind, LASHLANG_ENGINE_KIND);
        assert_eq!(
            LashlangProcessInput::from_payload(payload)
                .expect("remote payload decodes")
                .process_name,
            "main"
        );

        let identity = input.process_identity();
        assert_eq!(identity.kind, LASHLANG_ENGINE_KIND);
        assert_eq!(identity.label.as_deref(), Some("main"));
        assert_eq!(input.remote_identity().label.as_deref(), Some("main"));

        let draft = input
            .remote_trigger_subscription_draft(
                lash_remote_protocol::RemoteProcessOriginator::Host,
                "process-env:sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                    .parse()
                    .expect("canonical env ref"),
                "ui.button.pressed",
                "source-key",
            )
            .expect("remote trigger draft");
        draft.validate().expect("draft validates");
        assert_eq!(draft.target_label.as_deref(), Some("main"));
        assert_eq!(draft.target_identity.label.as_deref(), Some("main"));
    }

    #[test]
    fn tool_binding_defaults_remain_lashlang_local_policy() {
        let tool = lash_core::ToolDefinition::raw_named(
            "read_file",
            "read a file",
            lash_core::ToolDefinition::default_input_schema(),
            serde_json::Value::Null,
        );

        let binding = tool_lashlang_binding(&tool.manifest).executable_for(&tool.manifest.name);

        assert_eq!(binding.module_path, vec!["tools"]);
        assert_eq!(binding.operation, "read.file");
        assert_eq!(binding.authority_type, "Tools");
        assert_eq!(binding.call_path(), "tools.read.file");
    }

    #[test]
    fn explicit_tool_binding_attaches_lashlang_metadata() {
        let tool = lash_core::ToolDefinition::raw_named(
            "read_file",
            "read a file",
            lash_core::ToolDefinition::default_input_schema(),
            serde_json::Value::Null,
        )
        .with_lashlang_binding(
            LashlangToolBinding::new(["fs"], "read")
                .with_authority_type("Filesystem")
                .with_aliases(["cat"]),
        );

        let binding = tool_lashlang_binding(&tool.manifest).executable_for(&tool.manifest.name);

        assert_eq!(binding.module_path, vec!["fs"]);
        assert_eq!(binding.operation, "read");
        assert_eq!(binding.authority_type, "Filesystem");
        assert_eq!(binding.aliases, vec!["cat"]);
    }

    #[test]
    fn manifest_lashlang_binding_accessor_reports_absent_valid_and_malformed() {
        let mut manifest = lash_core::ToolDefinition::raw_named(
            "read_file",
            "read a file",
            lash_core::ToolDefinition::default_input_schema(),
            serde_json::Value::Null,
        )
        .manifest;
        assert_eq!(manifest.lashlang_binding().expect("absent binding"), None);

        manifest.bindings.insert(
            LASHLANG_TOOL_BINDING_KEY.to_string(),
            serde_json::json!({
                "module_path": ["fs"],
                "operation": "read"
            }),
        );
        let binding = manifest
            .lashlang_binding()
            .expect("valid binding")
            .expect("present binding");
        assert_eq!(binding.module_path, vec!["fs"]);
        assert_eq!(binding.operation.as_deref(), Some("read"));

        manifest.bindings.insert(
            LASHLANG_TOOL_BINDING_KEY.to_string(),
            serde_json::json!({ "module_path": "fs" }),
        );
        assert!(manifest.lashlang_binding().is_err());
    }

    #[test]
    fn remote_grant_lashlang_binding_accessor_reports_absent_valid_and_malformed() {
        let grant = remote_tool_grant("read_file");
        assert_eq!(grant.lashlang_binding().expect("absent binding"), None);

        let grant = grant.with_lashlang_binding(LashlangToolBinding::new(["fs"], "read"));
        let binding = grant
            .lashlang_binding()
            .expect("valid binding")
            .expect("present binding");
        assert_eq!(binding.module_path, vec!["fs"]);
        assert_eq!(binding.operation.as_deref(), Some("read"));

        let mut malformed = grant;
        malformed.bindings.insert(
            LASHLANG_TOOL_BINDING_KEY.to_string(),
            serde_json::json!({ "module_path": "fs" }),
        );
        assert!(malformed.lashlang_binding().is_err());
    }

    #[test]
    fn surface_merges_plugin_extensions() {
        let contribution = LashlangSurfaceContribution::new(
            LashlangAbilities::default().with_processes(),
            LashlangLanguageFeatures::default().with_label_annotations(),
            LashlangHostCatalog::tool_default(["lookup"]),
        );
        let extensions = lash_core::PluginExtensions::from_contributions([
            lash_core::PluginExtensionContribution::new(
                LASHLANG_SURFACE_EXTENSION_ID,
                contribution,
            )
            .expect("extension payload serializes"),
        ]);

        let surface = LashlangSurface::default()
            .with_plugin_extensions(&extensions)
            .expect("lashlang surface extension merges");
        let environment = surface.host_environment(&lash_core::ToolCatalog::default());

        assert!(environment.abilities.sleep);
        assert!(environment.abilities.processes);
        assert!(environment.language_features.label_annotations);
        assert!(
            environment
                .resources
                .resolve_module_operation("Tools", "tools", "lookup")
                .is_some()
        );
    }

    fn remote_tool_grant(name: &str) -> lash_remote_protocol::RemoteToolGrant {
        lash_remote_protocol::RemoteToolGrant {
            protocol_version: lash_remote_protocol::REMOTE_PROTOCOL_VERSION,
            id: None,
            name: name.to_string(),
            description: String::new(),
            input_schema: lash_core::ToolDefinition::default_input_schema(),
            output_schema: serde_json::Value::Null,
            input_schema_projections: Vec::new(),
            output_schema_projections: Vec::new(),
            output_contract: lash_remote_protocol::RemoteToolOutputContract::Static,
            examples: Vec::new(),
            availability: None,
            activation: None,
            argument_projection: None,
            scheduling: None,
            retry_policy: None,
            bindings: Default::default(),
        }
    }
}
