use std::borrow::Borrow;
use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::artifact::{
    ModuleArtifact, SurfaceRequirements, surface_requirements_for_program_with_catalog,
};
use crate::ast::{
    AssignPathStep, AstString, Declaration, Expr, ProcessDecl, ProcessParam, Program,
    ResourceRefExpr, TypeExpr, TypeField, format_type_expr,
};
use crate::lexer::Span;

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceCatalog {
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    module_instances: BTreeMap<String, ModuleInstanceCatalog>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    resource_types: BTreeMap<String, ResourceTypeCatalog>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    named_data_types: BTreeMap<String, NamedDataType>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    value_constructors: BTreeMap<String, ValueConstructorBinding>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    trigger_sources: BTreeMap<String, TriggerSourceBinding>,
}

impl ResourceCatalog {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn tool_default(operations: impl IntoIterator<Item = impl Into<String>>) -> Self {
        let mut catalog = Self::new();
        catalog.add_module_instance(["tools"], "Tools");
        for operation in operations {
            let operation = operation.into();
            catalog.add_operation(
                "Tools",
                operation.clone(),
                operation,
                TypeExpr::Any,
                TypeExpr::Any,
            );
        }
        catalog
    }

    pub fn add_module_instance(
        &mut self,
        module_path: impl IntoIterator<Item = impl Into<String>>,
        resource_type: impl Into<String>,
    ) {
        let path = module_path.into_iter().map(Into::into).collect::<Vec<_>>();
        assert!(!path.is_empty(), "module path must not be empty");
        let resource_type = resource_type.into();
        let key = module_path_key(&path);
        self.module_instances.insert(
            key.clone(),
            ModuleInstanceCatalog {
                path,
                resource_type: resource_type.clone(),
                alias: key,
            },
        );
        self.ensure_resource_type(resource_type);
    }

    pub fn ensure_resource_type(&mut self, resource_type: impl Into<String>) {
        self.resource_types.entry(resource_type.into()).or_default();
    }

    pub fn add_operation(
        &mut self,
        resource_type: impl Into<String>,
        operation: impl Into<String>,
        host_operation: impl Into<String>,
        input_ty: TypeExpr,
        output_ty: TypeExpr,
    ) {
        self.resource_types
            .entry(resource_type.into())
            .or_default()
            .operations
            .insert(
                operation.into(),
                ResourceOperationBinding {
                    host_operation: host_operation.into(),
                    input_ty,
                    output_ty,
                },
            );
    }

    pub fn add_value_constructor(
        &mut self,
        path: impl IntoIterator<Item = impl Into<String>>,
        input_ty: TypeExpr,
        output_ty: TypeExpr,
    ) {
        let path = path.into_iter().map(Into::into).collect::<Vec<_>>();
        assert!(!path.is_empty(), "constructor path must not be empty");
        let key = module_path_key(&path);
        self.value_constructors.insert(
            key.clone(),
            ValueConstructorBinding {
                path,
                type_name: format_type_expr(&output_ty),
                input_ty,
                output_ty,
            },
        );
    }

    pub fn add_named_data_type(
        &mut self,
        data_type: NamedDataType,
    ) -> Result<(), ResourceCatalogError> {
        self.merge_named_data_type(data_type)
    }

    pub fn add_trigger_source_constructor(
        &mut self,
        path: impl IntoIterator<Item = impl Into<String>>,
        input_ty: TypeExpr,
        event_ty: NamedDataType,
    ) -> Result<(), ResourceCatalogError> {
        let path = path.into_iter().map(Into::into).collect::<Vec<_>>();
        assert!(!path.is_empty(), "constructor path must not be empty");
        let source_type = module_path_key(&path);
        self.check_named_data_type(&event_ty)?;
        if let Some(existing) = self.trigger_sources.get(source_type.as_str())
            && existing.event_type() != &event_ty
        {
            return Err(ResourceCatalogError::ConflictingTriggerSource {
                source_type,
                existing: existing.event_type().name().to_string(),
                incoming: event_ty.name().to_string(),
            });
        }
        self.add_value_constructor(path, input_ty, TypeExpr::Ref(source_type.clone().into()));
        self.add_trigger_source_type(source_type, event_ty)?;
        Ok(())
    }

    pub(crate) fn add_trigger_source_type(
        &mut self,
        source_ty: impl Into<String>,
        event_ty: NamedDataType,
    ) -> Result<(), ResourceCatalogError> {
        self.merge_named_data_type(event_ty.clone())?;
        self.trigger_sources
            .insert(source_ty.into(), TriggerSourceBinding::new(event_ty));
        Ok(())
    }

    pub fn extend(&mut self, other: Self) {
        self.try_extend(other)
            .expect("conflicting named host data type definitions");
    }

    pub fn try_extend(&mut self, other: Self) -> Result<(), ResourceCatalogError> {
        for (resource_type, incoming) in other.resource_types {
            let entry = self.resource_types.entry(resource_type).or_default();
            entry.operations.extend(incoming.operations);
        }
        self.module_instances.extend(other.module_instances);
        for data_type in other.named_data_types.into_values() {
            self.merge_named_data_type(data_type)?;
        }
        self.value_constructors.extend(other.value_constructors);
        self.trigger_sources.extend(other.trigger_sources);
        Ok(())
    }

    pub fn union(mut self, other: Self) -> Self {
        self.extend(other);
        self
    }

    pub fn satisfies(&self, required: &Self) -> bool {
        for (path, required_module) in &required.module_instances {
            if self.module_instances.get(path) != Some(required_module) {
                return false;
            }
        }
        for (resource_type, required_catalog) in &required.resource_types {
            let Some(catalog) = self.resource_types.get(resource_type) else {
                return false;
            };
            for (operation, required_binding) in &required_catalog.operations {
                if catalog.operations.get(operation) != Some(required_binding) {
                    return false;
                }
            }
        }
        for (path, required_constructor) in &required.value_constructors {
            if self.value_constructors.get(path) != Some(required_constructor) {
                return false;
            }
        }
        for (name, required_data_type) in &required.named_data_types {
            if self.named_data_types.get(name) != Some(required_data_type) {
                return false;
            }
        }
        for (source_type, required_binding) in &required.trigger_sources {
            if self.trigger_sources.get(source_type) != Some(required_binding) {
                return false;
            }
        }
        true
    }

    pub fn has_resource_type(&self, resource_type: &str) -> bool {
        self.resource_types.contains_key(resource_type)
    }

    pub fn has_named_data_type(&self, name: &str) -> bool {
        self.named_data_types.contains_key(name)
    }

    pub fn is_known_opaque_value_type(&self, name: &str) -> bool {
        self.trigger_sources.contains_key(name)
            || self.value_constructors.values().any(|constructor| {
                matches!(&constructor.output_ty, TypeExpr::Ref(type_name) if type_name == name)
            })
    }

    pub fn module_instances(&self) -> impl Iterator<Item = (&str, &ModuleInstanceCatalog)> {
        self.module_instances
            .iter()
            .map(|(path, module)| (path.as_str(), module))
    }

    pub fn resource_types(&self) -> impl Iterator<Item = (&str, &ResourceTypeCatalog)> {
        self.resource_types
            .iter()
            .map(|(resource_type, catalog)| (resource_type.as_str(), catalog))
    }

    pub fn named_data_types(&self) -> impl Iterator<Item = (&str, &NamedDataType)> {
        self.named_data_types
            .iter()
            .map(|(name, data_type)| (name.as_str(), data_type))
    }

    pub fn value_constructors(&self) -> impl Iterator<Item = (&str, &ValueConstructorBinding)> {
        self.value_constructors
            .iter()
            .map(|(path, constructor)| (path.as_str(), constructor))
    }

    pub fn trigger_sources(&self) -> impl Iterator<Item = (&str, &TriggerSourceBinding)> {
        self.trigger_sources
            .iter()
            .map(|(source_type, binding)| (source_type.as_str(), binding))
    }

    pub fn resolve_named_data_type(&self, name: &str) -> Option<&NamedDataType> {
        self.named_data_types.get(name)
    }

    pub fn resolve_trigger_source(&self, source_ty: &str) -> Option<&TriggerSourceBinding> {
        self.trigger_sources.get(source_ty)
    }

    pub fn resolve_module_path(&self, path: &[impl AsRef<str>]) -> Option<ResourceRefExpr> {
        let key = module_path_key(path);
        let module = self.module_instances.get(&key)?;
        Some(ResourceRefExpr::resolved(
            module
                .path
                .iter()
                .map(|segment| segment.as_str().into())
                .collect(),
            module.resource_type.clone(),
            module.alias.clone(),
        ))
    }

    pub fn resolve_alias(&self, resource: &ResourceRefExpr) -> Option<&ResourceTypeCatalog> {
        if !resource.resource_type.is_empty() {
            return self.resource_types.get(resource.resource_type.as_str());
        }
        let resolved = self.resolve_module_path(&resource.path)?;
        self.resource_types.get(resolved.resource_type.as_str())
    }

    pub fn resolve_operation(
        &self,
        resource_type: &str,
        operation: &str,
    ) -> Option<&ResourceOperationBinding> {
        self.resource_types
            .get(resource_type)?
            .operations
            .get(operation)
    }

    pub fn resolve_operation_by_host(
        &self,
        resource_type: &str,
        host_operation: &str,
    ) -> Option<(&str, &ResourceOperationBinding)> {
        self.resource_types
            .get(resource_type)?
            .operations
            .iter()
            .find_map(|(operation, binding)| {
                (binding.host_operation == host_operation).then_some((operation.as_str(), binding))
            })
    }

    pub fn has_operations(&self) -> bool {
        self.resource_types
            .values()
            .any(|resource_type| !resource_type.operations.is_empty())
    }

    pub fn resolve_value_constructor(
        &self,
        path: &[impl AsRef<str>],
    ) -> Option<&ValueConstructorBinding> {
        self.value_constructors.get(&module_path_key(path))
    }

    pub fn trigger_source_event(&self, source_ty: &TypeExpr) -> Option<TypeExpr> {
        let TypeExpr::Ref(name) = source_ty else {
            return None;
        };
        self.trigger_sources
            .get(name.as_str())
            .map(|binding| binding.event_type().to_ref_ty())
    }

    pub fn operation_suggestions_for_host(&self, host_operation: &str) -> Vec<String> {
        let mut suggestions = Vec::new();
        for module in self.module_instances.values() {
            let Some(resource_type) = self.resource_types.get(&module.resource_type) else {
                continue;
            };
            for (operation, binding) in &resource_type.operations {
                if binding.host_operation == host_operation {
                    suggestions.push(format!("{}.{}", module.alias, operation));
                }
            }
        }
        suggestions.sort();
        suggestions.dedup();
        suggestions
    }

    pub fn operation_suggestions_for_prefix(
        &self,
        prefix: &[impl AsRef<str>],
        operation: &str,
    ) -> Vec<String> {
        let prefix = module_path_key(prefix);
        let mut suggestions = Vec::new();
        for module in self.module_instances.values() {
            if module.alias == prefix || !module.alias.starts_with(&format!("{prefix}.")) {
                continue;
            }
            if self
                .resolve_operation(&module.resource_type, operation)
                .is_some()
            {
                suggestions.push(format!("{}.{}", module.alias, operation));
            }
        }
        suggestions.sort();
        suggestions.dedup();
        suggestions
    }

    fn check_named_data_type(&self, data_type: &NamedDataType) -> Result<(), ResourceCatalogError> {
        if let Some(existing) = self.named_data_types.get(data_type.name())
            && existing != data_type
        {
            return Err(ResourceCatalogError::ConflictingNamedDataType {
                name: data_type.name().to_string(),
            });
        }
        Ok(())
    }

    fn merge_named_data_type(
        &mut self,
        data_type: NamedDataType,
    ) -> Result<(), ResourceCatalogError> {
        self.check_named_data_type(&data_type)?;
        self.named_data_types
            .entry(data_type.name().to_string())
            .or_insert(data_type);
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NamedDataType {
    name: String,
    ty: TypeExpr,
}

impl NamedDataType {
    pub fn new(name: impl Into<String>, ty: TypeExpr) -> Result<Self, NamedDataTypeError> {
        let name = name.into();
        if !is_qualified_type_name(&name) {
            return Err(NamedDataTypeError::InvalidName { name });
        }
        if !matches!(ty, TypeExpr::Object(_)) {
            return Err(NamedDataTypeError::ExpectedObject { name });
        }
        validate_named_data_shape(&ty)?;
        Ok(Self { name, ty })
    }

    pub fn object(
        name: impl Into<String>,
        fields: Vec<TypeField>,
    ) -> Result<Self, NamedDataTypeError> {
        Self::new(name, TypeExpr::Object(fields))
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn ty(&self) -> &TypeExpr {
        &self.ty
    }

    pub fn to_ref_ty(&self) -> TypeExpr {
        TypeExpr::Ref(self.name.clone().into())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum NamedDataTypeError {
    #[error("host data type name `{name}` must be qualified")]
    InvalidName { name: String },
    #[error("host data type `{name}` must be an object type")]
    ExpectedObject { name: String },
    #[error("host data type object has duplicate field `{field}`")]
    DuplicateField { field: String },
    #[error("host data type enum has duplicate value `{value}`")]
    DuplicateEnumValue { value: String },
    #[error("host data type shape cannot contain nested type ref `{name}`")]
    NestedRef { name: String },
    #[error("host data type shape cannot contain {ty}")]
    UnsupportedType { ty: &'static str },
}

#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum ResourceCatalogError {
    #[error("conflicting host data type definition `{name}`")]
    ConflictingNamedDataType { name: String },
    #[error(
        "trigger source `{source_type}` already emits `{existing}`, cannot change it to `{incoming}`"
    )]
    ConflictingTriggerSource {
        source_type: String,
        existing: String,
        incoming: String,
    },
}

fn is_qualified_type_name(name: &str) -> bool {
    let mut segments = name.split('.');
    let mut count = 0usize;
    for segment in segments.by_ref() {
        count += 1;
        let mut chars = segment.chars();
        let Some(first) = chars.next() else {
            return false;
        };
        if !(first.is_ascii_alphabetic() || first == '_') {
            return false;
        }
        if !chars.all(|ch| ch.is_ascii_alphanumeric() || ch == '_') {
            return false;
        }
    }
    count >= 2
}

fn validate_named_data_shape(ty: &TypeExpr) -> Result<(), NamedDataTypeError> {
    match ty {
        TypeExpr::Any
        | TypeExpr::Str
        | TypeExpr::Int
        | TypeExpr::Float
        | TypeExpr::Bool
        | TypeExpr::Dict
        | TypeExpr::Null => Ok(()),
        TypeExpr::Enum(values) => {
            let mut seen = BTreeSet::new();
            for value in values {
                if !seen.insert(value.to_string()) {
                    return Err(NamedDataTypeError::DuplicateEnumValue {
                        value: value.to_string(),
                    });
                }
            }
            Ok(())
        }
        TypeExpr::List(item) => validate_named_data_shape(item),
        TypeExpr::Object(fields) => {
            let mut seen = BTreeSet::new();
            for field in fields {
                if !seen.insert(field.name.to_string()) {
                    return Err(NamedDataTypeError::DuplicateField {
                        field: field.name.to_string(),
                    });
                }
                validate_named_data_shape(&field.ty)?;
            }
            Ok(())
        }
        TypeExpr::Union(items) => {
            for item in items {
                validate_named_data_shape(item)?;
            }
            Ok(())
        }
        TypeExpr::Ref(name) => Err(NamedDataTypeError::NestedRef {
            name: name.to_string(),
        }),
        TypeExpr::Process { .. } => Err(NamedDataTypeError::UnsupportedType { ty: "process" }),
        TypeExpr::TriggerHandle(_) => Err(NamedDataTypeError::UnsupportedType {
            ty: "trigger handle",
        }),
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceTypeCatalog {
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub operations: BTreeMap<String, ResourceOperationBinding>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModuleInstanceCatalog {
    pub path: Vec<String>,
    pub resource_type: String,
    pub alias: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceOperationBinding {
    pub host_operation: String,
    pub input_ty: TypeExpr,
    pub output_ty: TypeExpr,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValueConstructorBinding {
    pub path: Vec<String>,
    pub type_name: String,
    pub input_ty: TypeExpr,
    pub output_ty: TypeExpr,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TriggerSourceBinding {
    event_type: NamedDataType,
}

impl TriggerSourceBinding {
    fn new(event_type: NamedDataType) -> Self {
        Self { event_type }
    }

    pub fn event_type(&self) -> &NamedDataType {
        &self.event_type
    }

    pub fn event_ty(&self) -> &TypeExpr {
        self.event_type.ty()
    }

    pub fn event_type_name(&self) -> &str {
        self.event_type.name()
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LashlangSurface {
    #[serde(default)]
    pub resources: ResourceCatalog,
    #[serde(default)]
    pub abilities: LashlangAbilities,
    #[serde(default)]
    pub language_features: LashlangLanguageFeatures,
}

impl LashlangSurface {
    pub fn new(resources: ResourceCatalog, abilities: LashlangAbilities) -> Self {
        Self {
            resources,
            abilities,
            language_features: LashlangLanguageFeatures::default(),
        }
    }

    pub fn with_language_features(mut self, language_features: LashlangLanguageFeatures) -> Self {
        self.language_features = language_features;
        self
    }

    pub fn satisfies(&self, requirements: &SurfaceRequirements) -> bool {
        self.abilities.satisfies(requirements.abilities)
            && self
                .language_features
                .satisfies(requirements.language_features)
            && self.resources.satisfies(&requirements.resources)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct LashlangLanguageFeatures {
    pub label_annotations: bool,
}

impl LashlangLanguageFeatures {
    pub fn union(self, other: Self) -> Self {
        Self {
            label_annotations: self.label_annotations || other.label_annotations,
        }
    }

    pub fn satisfies(self, required: Self) -> bool {
        !required.label_annotations || self.label_annotations
    }

    pub fn with_label_annotations(mut self) -> Self {
        self.label_annotations = true;
        self
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct LashlangAbilities {
    pub processes: bool,
    pub sleep: bool,
    pub process_signals: bool,
    pub triggers: bool,
}

impl LashlangAbilities {
    pub fn union(self, other: Self) -> Self {
        Self {
            processes: self.processes || other.processes,
            sleep: self.sleep || other.sleep,
            process_signals: self.process_signals || other.process_signals,
            triggers: self.triggers || other.triggers,
        }
    }

    pub fn satisfies(self, required: Self) -> bool {
        (!required.processes || self.processes)
            && (!required.sleep || self.sleep)
            && (!required.process_signals || self.process_signals)
            && (!required.triggers || self.triggers)
    }

    pub fn with_processes(mut self) -> Self {
        self.processes = true;
        self
    }

    pub fn with_sleep(mut self) -> Self {
        self.sleep = true;
        self
    }

    pub fn with_process_signals(mut self) -> Self {
        self.process_signals = true;
        self
    }

    pub fn with_triggers(mut self) -> Self {
        self.triggers = true;
        self
    }

    pub fn all() -> Self {
        Self::default()
            .with_sleep()
            .with_processes()
            .with_process_signals()
            .with_triggers()
    }
}

fn module_path_key(path: &[impl AsRef<str>]) -> String {
    path.iter()
        .map(|segment| segment.as_ref())
        .collect::<Vec<_>>()
        .join(".")
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LinkedModule {
    pub module_ref: crate::ModuleRef,
    pub required_surface_ref: crate::RequiredSurfaceRef,
    pub artifact: ModuleArtifact,
    #[serde(skip)]
    linked_program: Option<Program>,
}

impl LinkedModule {
    pub fn link(
        program: Program,
        surface: impl Borrow<LashlangSurface>,
    ) -> Result<Self, LinkError> {
        let surface = surface.borrow();
        let mut linker = Linker::new(&program, surface);
        let program = linker.link_program()?;
        let requirements =
            surface_requirements_for_program_with_catalog(&program, &surface.resources);
        let artifact =
            ModuleArtifact::from_program_with_requirements(program.clone(), requirements).map_err(
                |err| LinkError::ModuleHash {
                    message: err.to_string(),
                },
            )?;
        Ok(Self {
            module_ref: artifact.module_ref.clone(),
            required_surface_ref: artifact.required_surface_ref.clone(),
            artifact,
            linked_program: Some(program),
        })
    }

    pub fn program(&self) -> &Program {
        self.linked_program
            .as_ref()
            .unwrap_or(&self.artifact.canonical_ir)
    }
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum LinkError {
    #[error("duplicate declaration `{name}`")]
    DuplicateDeclaration { name: String, span: Option<Span> },
    #[error("duplicate process parameter `{name}`")]
    DuplicateProcessParam { name: String, span: Option<Span> },
    #[error("unknown process `{name}`")]
    UnknownProcess { name: String, span: Option<Span> },
    #[error("process `{process}` is missing argument `{arg}`")]
    MissingProcessArgument {
        process: String,
        arg: String,
        span: Option<Span>,
    },
    #[error("process `{process}` does not accept argument `{arg}`")]
    UnexpectedProcessArgument {
        process: String,
        arg: String,
        span: Option<Span>,
    },
    #[error("duplicate process argument `{arg}`")]
    DuplicateProcessArgument { arg: String, span: Option<Span> },
    #[error("unknown name `{name}`")]
    UnknownName { name: String, span: Option<Span> },
    #[error("unknown builtin `{name}`")]
    UnknownBuiltin { name: String, span: Option<Span> },
    #[error("unknown module `{path}`")]
    UnknownResource { path: String, span: Option<Span> },
    #[error("unknown type `{name}`")]
    UnknownType { name: String, span: Option<Span> },
    #[error("constructor `{path}` expects {expected}, got {actual}")]
    IncompatibleConstructorInput {
        path: String,
        expected: String,
        actual: String,
        span: Option<Span>,
    },
    #[error("operation `{operation}` expects {expected}, got {actual}")]
    IncompatibleOperationInput {
        operation: String,
        expected: String,
        actual: String,
        span: Option<Span>,
    },
    #[error("process `{process}` return type is incompatible: expected {expected}, got {actual}")]
    IncompatibleProcessReturn {
        process: String,
        expected: String,
        actual: String,
        span: Option<Span>,
    },
    #[error("trigger registration requires {{ source, target, inputs, name? }}")]
    InvalidTriggerRegistration { span: Option<Span> },
    #[error("trigger registration `inputs` must be a literal record")]
    InvalidTriggerInputs { span: Option<Span> },
    #[error("trigger registration input `{input}` is duplicated")]
    DuplicateTriggerInput { input: String, span: Option<Span> },
    #[error("trigger target `{process}` input `{input}` is not mapped")]
    MissingTriggerInput {
        process: String,
        input: String,
        span: Option<Span>,
    },
    #[error("trigger target `{process}` has no input `{input}`")]
    UnknownTriggerInput {
        process: String,
        input: String,
        span: Option<Span>,
    },
    #[error("trigger registration `inputs` must map at least one param to `trigger.event`")]
    MissingTriggerEventInput { span: Option<Span> },
    #[error("`trigger.event` is only valid as a direct value inside `triggers.register` inputs")]
    TriggerEventOutsideInputs { span: Option<Span> },
    #[error(
        "`trigger.event` represents the whole event; projections such as `trigger.event.field` are not supported"
    )]
    TriggerEventProjection { span: Option<Span> },
    #[error("trigger listing requires {{ target }}")]
    InvalidTriggerList { span: Option<Span> },
    #[error("trigger cancellation requires {{ handle }}")]
    InvalidTriggerCancel { span: Option<Span> },
    #[error("trigger source type `{source_ty}` is not registered as a TriggerSource")]
    UnknownTriggerSourceType {
        source_ty: String,
        span: Option<Span>,
    },
    #[error("trigger target must be a process value, got {actual}")]
    InvalidTriggerTarget { actual: String, span: Option<Span> },
    #[error("trigger source emits {event}, but target input `{input_name}` expects {input}")]
    TriggerEventMismatch {
        event: String,
        input_name: String,
        input: String,
        span: Option<Span>,
    },
    #[error("receiver for operation `{operation}` is not a module authority")]
    UnresolvedReceiver {
        operation: String,
        span: Option<Span>,
    },
    #[error("resource type `{resource_type}` does not expose operation `{operation}`")]
    UnknownResourceOperation {
        resource_type: String,
        operation: String,
        span: Option<Span>,
    },
    #[error("module `{module_path}` does not expose operation `{operation}`; available identity-qualified paths: {}", suggestions.join(", "))]
    AmbiguousModuleOperation {
        module_path: String,
        operation: String,
        suggestions: Vec<String>,
        span: Option<Span>,
    },
    #[error("tools must be called through module paths, e.g. `{suggestion}`")]
    BareToolCall {
        name: String,
        suggestion: String,
        span: Option<Span>,
    },
    #[error(
        "process `{process}` argument `{arg}` has incompatible authority type: expected {expected}, got {actual}"
    )]
    IncompatibleProcessArgument {
        process: String,
        arg: String,
        expected: String,
        actual: String,
        span: Option<Span>,
    },
    #[error("lashlang feature `{feature}` is disabled by this host")]
    FeatureDisabled {
        feature: &'static str,
        span: Option<Span>,
    },
    #[error("`{keyword}` can only be used inside a process body")]
    ProcessLifecycleOutsideProcess {
        keyword: &'static str,
        span: Option<Span>,
    },
    #[error("cannot access `{access}` on opaque host value `{type_name}`")]
    OpaqueHostValueAccess {
        type_name: String,
        access: String,
        span: Option<Span>,
    },
    #[error("failed to hash linked module: {message}")]
    ModuleHash { message: String },
}

impl LinkError {
    pub fn span(&self) -> Option<Span> {
        match self {
            Self::DuplicateDeclaration { span, .. }
            | Self::DuplicateProcessParam { span, .. }
            | Self::UnknownProcess { span, .. }
            | Self::MissingProcessArgument { span, .. }
            | Self::UnexpectedProcessArgument { span, .. }
            | Self::DuplicateProcessArgument { span, .. }
            | Self::UnknownName { span, .. }
            | Self::UnknownBuiltin { span, .. }
            | Self::UnknownResource { span, .. }
            | Self::UnknownType { span, .. }
            | Self::IncompatibleConstructorInput { span, .. }
            | Self::IncompatibleOperationInput { span, .. }
            | Self::IncompatibleProcessReturn { span, .. }
            | Self::InvalidTriggerRegistration { span }
            | Self::InvalidTriggerInputs { span }
            | Self::DuplicateTriggerInput { span, .. }
            | Self::MissingTriggerInput { span, .. }
            | Self::UnknownTriggerInput { span, .. }
            | Self::MissingTriggerEventInput { span }
            | Self::TriggerEventOutsideInputs { span }
            | Self::TriggerEventProjection { span }
            | Self::InvalidTriggerList { span }
            | Self::InvalidTriggerCancel { span }
            | Self::UnknownTriggerSourceType { span, .. }
            | Self::InvalidTriggerTarget { span, .. }
            | Self::TriggerEventMismatch { span, .. }
            | Self::UnresolvedReceiver { span, .. }
            | Self::UnknownResourceOperation { span, .. }
            | Self::AmbiguousModuleOperation { span, .. }
            | Self::BareToolCall { span, .. }
            | Self::IncompatibleProcessArgument { span, .. }
            | Self::FeatureDisabled { span, .. }
            | Self::ProcessLifecycleOutsideProcess { span, .. }
            | Self::OpaqueHostValueAccess { span, .. } => *span,
            Self::ModuleHash { .. } => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Binding {
    Value(TypeExpr),
    Resource { resource_type: String },
}

struct Linker<'module> {
    program: &'module Program,
    surface: &'module LashlangSurface,
    process_names: BTreeSet<String>,
    process_types: BTreeMap<String, TypeExpr>,
    type_names: BTreeSet<String>,
    type_defs: BTreeMap<String, TypeExpr>,
}

impl<'module> Linker<'module> {
    fn new(program: &'module Program, surface: &'module LashlangSurface) -> Self {
        Self {
            program,
            surface,
            process_names: BTreeSet::new(),
            process_types: BTreeMap::new(),
            type_names: BTreeSet::new(),
            type_defs: BTreeMap::new(),
        }
    }

    fn link_program(&mut self) -> Result<Program, LinkError> {
        // Single walk: collect declaration metadata, then lower (and validate)
        // declarations in source order, then lower main. Declaration errors
        // therefore still surface before main errors, matching the prior
        // two-pass (validate-then-lower) ordering.
        self.collect_declarations()?;
        let declarations = self
            .program
            .declarations
            .iter()
            .enumerate()
            .map(|(index, declaration)| {
                let span = self.program.declaration_spans.get(index).copied();
                self.lower_declaration(declaration, span)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut scope = Scope::new(true, false, None);
        let main = self.lower_expr(&self.program.main, &mut scope)?.0;
        Ok(Program {
            declarations,
            main,
            declaration_spans: self.program.declaration_spans.clone(),
            expression_spans: self.program.expression_spans.clone(),
        })
    }

    fn collect_declarations(&mut self) -> Result<(), LinkError> {
        self.ensure_label_annotations_enabled_for_program()?;
        let mut names = BTreeSet::new();
        for (index, declaration) in self.program.declarations.iter().enumerate() {
            let span = self.program.declaration_spans.get(index).copied();
            let (namespace, name) = match declaration {
                Declaration::Type(decl) => {
                    let name = decl.name.as_str();
                    if !names.insert(("type", name.to_string())) {
                        return Err(LinkError::DuplicateDeclaration {
                            name: name.to_string(),
                            span,
                        });
                    }
                    self.type_names.insert(decl.name.to_string());
                    self.type_defs
                        .insert(decl.name.to_string(), decl.ty.clone());
                    continue;
                }
                Declaration::Process(decl) => {
                    self.ensure_feature(self.surface.abilities.processes, "processes", span)?;
                    ("process", decl.name.as_str())
                }
            };
            if !names.insert((namespace, name.to_string())) {
                return Err(LinkError::DuplicateDeclaration {
                    name: name.to_string(),
                    span,
                });
            }
            if let Declaration::Process(decl) = declaration {
                self.process_names.insert(decl.name.to_string());
            }
        }
        for declaration in &self.program.declarations {
            match declaration {
                Declaration::Type(type_decl) => self.validate_type_refs(&type_decl.ty, None)?,
                Declaration::Process(process) => {
                    for param in &process.params {
                        self.validate_type_refs(&param.ty, None)?;
                    }
                    if let Some(return_ty) = &process.return_ty {
                        self.validate_type_refs(return_ty, None)?;
                    }
                }
            }
        }
        for declaration in &self.program.declarations {
            if let Declaration::Process(process) = declaration {
                self.process_types.insert(
                    process.name.to_string(),
                    process_type_for_decl(process, TypeExpr::Any),
                );
            }
        }
        for (index, declaration) in self.program.declarations.iter().enumerate() {
            let Declaration::Process(process) = declaration else {
                continue;
            };
            let span = self.program.declaration_spans.get(index).copied();
            let output = self.infer_process_output(process, span)?;
            if let Some(expected) = &process.return_ty
                && !self.is_type_assignable(&output, expected)
            {
                return Err(LinkError::IncompatibleProcessReturn {
                    process: process.name.to_string(),
                    expected: format_type_expr(&self.resolve_type_aliases(expected)),
                    actual: format_type_expr(&self.resolve_type_aliases(&output)),
                    span,
                });
            }
            self.process_types.insert(
                process.name.to_string(),
                process_type_for_decl(process, output),
            );
        }
        Ok(())
    }

    fn ensure_label_annotations_enabled_for_program(&self) -> Result<(), LinkError> {
        if self.surface.language_features.label_annotations {
            return Ok(());
        }
        for (index, declaration) in self.program.declarations.iter().enumerate() {
            let span = self.program.declaration_spans.get(index).copied();
            if let Declaration::Process(process) = declaration {
                if process.label.is_some() || expr_has_label_annotation(&process.body) {
                    return Err(LinkError::FeatureDisabled {
                        feature: "label annotations",
                        span,
                    });
                }
            }
        }
        if expr_has_label_annotation(&self.program.main) {
            return Err(LinkError::FeatureDisabled {
                feature: "label annotations",
                span: self.program.expression_spans.first().copied(),
            });
        }
        Ok(())
    }

    fn binding_for_type(&self, ty: &TypeExpr) -> Binding {
        match self.resource_type_for_type(ty) {
            Some(resource_type) => Binding::Resource { resource_type },
            _ => Binding::Value(ty.clone()),
        }
    }

    fn resource_type_for_type(&self, ty: &TypeExpr) -> Option<String> {
        match self.resolve_type_aliases(ty) {
            TypeExpr::Ref(name) if self.surface.resources.has_resource_type(name.as_str()) => {
                Some(name.to_string())
            }
            _ => None,
        }
    }

    fn resolve_type_aliases(&self, ty: &TypeExpr) -> TypeExpr {
        self.resolve_type_aliases_inner(ty, &mut BTreeSet::new())
    }

    fn resolve_type_aliases_inner(&self, ty: &TypeExpr, seen: &mut BTreeSet<String>) -> TypeExpr {
        match ty {
            TypeExpr::Ref(name) => {
                if !seen.insert(name.to_string()) {
                    return ty.clone();
                }
                let resolved = if let Some(ty) = self.type_defs.get(name.as_str()) {
                    self.resolve_type_aliases_inner(ty, seen)
                } else if let Some(data_type) = self
                    .surface
                    .resources
                    .resolve_named_data_type(name.as_str())
                {
                    data_type.ty().clone()
                } else {
                    ty.clone()
                };
                seen.remove(name.as_str());
                resolved
            }
            TypeExpr::List(item) => {
                TypeExpr::List(Box::new(self.resolve_type_aliases_inner(item, seen)))
            }
            TypeExpr::Object(fields) => TypeExpr::Object(
                fields
                    .iter()
                    .map(|field| TypeField {
                        name: field.name.clone(),
                        ty: self.resolve_type_aliases_inner(&field.ty, seen),
                        optional: field.optional,
                    })
                    .collect(),
            ),
            TypeExpr::Union(items) => TypeExpr::Union(
                items
                    .iter()
                    .map(|item| self.resolve_type_aliases_inner(item, seen))
                    .collect(),
            ),
            TypeExpr::Process {
                input,
                output,
                input_count,
            } => TypeExpr::Process {
                input: Box::new(self.resolve_type_aliases_inner(input, seen)),
                output: Box::new(self.resolve_type_aliases_inner(output, seen)),
                input_count: *input_count,
            },
            TypeExpr::TriggerHandle(event) => {
                TypeExpr::TriggerHandle(Box::new(self.resolve_type_aliases_inner(event, seen)))
            }
            TypeExpr::Any
            | TypeExpr::Str
            | TypeExpr::Int
            | TypeExpr::Float
            | TypeExpr::Bool
            | TypeExpr::Dict
            | TypeExpr::Null
            | TypeExpr::Enum(_) => ty.clone(),
        }
    }

    fn is_type_assignable(&self, source: &TypeExpr, target: &TypeExpr) -> bool {
        let source = self.resolve_type_aliases(source);
        let target = self.resolve_type_aliases(target);
        crate::trigger::is_resolved_type_assignable(&source, &target)
    }

    fn validate_type_refs(&self, ty: &TypeExpr, span: Option<Span>) -> Result<(), LinkError> {
        match ty {
            TypeExpr::Ref(name) => {
                if self.type_defs.contains_key(name.as_str())
                    || self.surface.resources.has_resource_type(name.as_str())
                    || self.surface.resources.has_named_data_type(name.as_str())
                    || self
                        .surface
                        .resources
                        .is_known_opaque_value_type(name.as_str())
                {
                    Ok(())
                } else {
                    Err(LinkError::UnknownType {
                        name: name.to_string(),
                        span,
                    })
                }
            }
            TypeExpr::List(item) => self.validate_type_refs(item, span),
            TypeExpr::Object(fields) => {
                for field in fields {
                    self.validate_type_refs(&field.ty, span)?;
                }
                Ok(())
            }
            TypeExpr::Union(items) => {
                for item in items {
                    self.validate_type_refs(item, span)?;
                }
                Ok(())
            }
            TypeExpr::Process { input, output, .. } => {
                self.validate_type_refs(input, span)?;
                self.validate_type_refs(output, span)
            }
            TypeExpr::TriggerHandle(event) => self.validate_type_refs(event, span),
            TypeExpr::Any
            | TypeExpr::Str
            | TypeExpr::Int
            | TypeExpr::Float
            | TypeExpr::Bool
            | TypeExpr::Dict
            | TypeExpr::Null
            | TypeExpr::Enum(_) => Ok(()),
        }
    }

    fn field_type(
        &self,
        target: &TypeExpr,
        field: &str,
        span: Option<Span>,
    ) -> Result<TypeExpr, LinkError> {
        let target = self.resolve_type_aliases(target);
        field_type(&target, field, span, |name| {
            self.surface.resources.is_known_opaque_value_type(name)
        })
    }

    fn index_type(&self, target: &TypeExpr, span: Option<Span>) -> Result<TypeExpr, LinkError> {
        let target = self.resolve_type_aliases(target);
        index_type(&target, span, |name| {
            self.surface.resources.is_known_opaque_value_type(name)
        })
    }

    fn ensure_feature(
        &self,
        enabled: bool,
        feature: &'static str,
        span: Option<Span>,
    ) -> Result<(), LinkError> {
        if enabled {
            Ok(())
        } else {
            Err(LinkError::FeatureDisabled { feature, span })
        }
    }

    fn validate_resource_ref(
        &self,
        resource: &ResourceRefExpr,
        span: Option<Span>,
    ) -> Result<ResourceRefExpr, LinkError> {
        if !resource.resource_type.is_empty() {
            return self
                .surface
                .resources
                .resolve_alias(resource)
                .map(|_| resource.clone())
                .ok_or_else(|| LinkError::UnknownResource {
                    path: resource.path_string(),
                    span,
                });
        }
        self.surface
            .resources
            .resolve_module_path(&resource.path)
            .ok_or_else(|| LinkError::UnknownResource {
                path: resource.path_string(),
                span,
            })
    }

    fn lower_declaration(
        &self,
        declaration: &Declaration,
        span: Option<Span>,
    ) -> Result<Declaration, LinkError> {
        Ok(match declaration {
            Declaration::Type(type_decl) => Declaration::Type(type_decl.clone()),
            Declaration::Process(process) => {
                self.ensure_feature(self.surface.abilities.processes, "processes", span)?;
                if process.label.is_some() {
                    self.ensure_feature(
                        self.surface.language_features.label_annotations,
                        "label annotations",
                        span,
                    )?;
                }
                let mut scope = Scope::new(false, true, span);
                let mut seen = BTreeSet::new();
                for param in &process.params {
                    if !seen.insert(param.name.to_string()) {
                        return Err(LinkError::DuplicateProcessParam {
                            name: param.name.to_string(),
                            span,
                        });
                    }
                    scope.bind(param.name.as_str(), self.binding_for_type(&param.ty));
                }
                scope.bind("input", Binding::Value(process_input_type(process)));
                scope.bind("inputs", Binding::Value(process_input_record_type(process)));
                let body = self.lower_expr(&process.body, &mut scope)?.0;
                Declaration::Process(ProcessDecl {
                    name: process.name.clone(),
                    params: process.params.clone(),
                    return_ty: process.return_ty.clone(),
                    label: process.label.clone(),
                    body,
                })
            }
        })
    }

    fn lower_expr(
        &self,
        expr: &Expr,
        scope: &mut Scope,
    ) -> Result<(Expr, Option<Binding>), LinkError> {
        self.reject_trigger_event_special_form(expr, scope.span)?;
        if matches!(expr, Expr::Variable(_) | Expr::Field { .. })
            && let Some(resource) = self.resolve_module_expr(expr, scope)
        {
            return Ok((
                Expr::ResourceRef(resource.clone()),
                Some(Binding::Resource {
                    resource_type: resource.resource_type.to_string(),
                }),
            ));
        }
        Ok(match expr {
            Expr::Block(expressions) => {
                let mut lowered = Vec::with_capacity(expressions.len());
                let mut last = None;
                for expression in expressions {
                    let (expr, binding) = self.lower_expr(expression, scope)?;
                    lowered.push(expr);
                    last = binding;
                }
                (Expr::Block(lowered), last)
            }
            Expr::LabelAnnotated { label, expr } => {
                self.ensure_feature(
                    self.surface.language_features.label_annotations,
                    "label annotations",
                    scope.span,
                )?;
                let (expr, binding) = self.lower_expr(expr, scope)?;
                (
                    Expr::LabelAnnotated {
                        label: label.clone(),
                        expr: Box::new(expr),
                    },
                    binding,
                )
            }
            Expr::Variable(name) => {
                if let Some(binding) = scope.get(name) {
                    (Expr::Variable(name.clone()), Some(binding))
                } else if let Some(process_ty) = self.process_types.get(name.as_str()) {
                    (
                        Expr::ProcessRef {
                            process: name.clone(),
                        },
                        Some(Binding::Value(process_ty.clone())),
                    )
                } else if scope.allow_unknown_globals {
                    // Top-level unknown globals are permitted; they surface as
                    // runtime errors rather than link errors.
                    (
                        Expr::Variable(name.clone()),
                        Some(Binding::Value(TypeExpr::Any)),
                    )
                } else {
                    return Err(LinkError::UnknownName {
                        name: name.to_string(),
                        span: scope.span,
                    });
                }
            }
            Expr::Null
            | Expr::Bool(_)
            | Expr::Number(_)
            | Expr::String(_)
            | Expr::Break
            | Expr::Continue
            | Expr::TypeLiteral(_) => (expr.clone(), Some(Binding::Value(literal_type(expr)))),
            Expr::List(items) => {
                let mut lowered = Vec::with_capacity(items.len());
                let mut item_types = Vec::with_capacity(items.len());
                for item in items {
                    let (item, binding) = self.lower_expr(item, scope)?;
                    lowered.push(item);
                    item_types.push(binding_type(binding.as_ref()));
                }
                (
                    Expr::List(lowered),
                    Some(Binding::Value(TypeExpr::List(Box::new(union_type(
                        item_types,
                    ))))),
                )
            }
            Expr::Record(entries) => {
                let mut lowered = Vec::with_capacity(entries.len());
                let mut fields = Vec::with_capacity(entries.len());
                for (name, value) in entries {
                    let (value, binding) = self.lower_expr(value, scope)?;
                    fields.push(TypeField {
                        name: name.clone(),
                        ty: binding_type(binding.as_ref()),
                        optional: false,
                    });
                    lowered.push((name.clone(), value));
                }
                (
                    Expr::Record(lowered),
                    Some(Binding::Value(TypeExpr::Object(fields))),
                )
            }
            Expr::Assign { target, expr } => {
                for step in &target.steps {
                    if let AssignPathStep::Index(index) = step {
                        self.lower_expr(index, scope)?;
                    }
                }
                let (lowered, binding) = self.lower_expr(expr, scope)?;
                if target.steps.is_empty() {
                    scope.bind(
                        target.root.as_str(),
                        binding.clone().unwrap_or(any_binding()),
                    );
                } else if scope.get(&target.root).is_none() && !scope.allow_unknown_globals {
                    return Err(LinkError::UnknownName {
                        name: target.root.to_string(),
                        span: scope.span,
                    });
                }
                (
                    Expr::Assign {
                        target: target.clone(),
                        expr: Box::new(lowered),
                    },
                    binding,
                )
            }
            Expr::If {
                condition,
                then_block,
                else_block,
            } => {
                let condition = self.lower_expr(condition, scope)?.0;
                let mut then_scope = scope.clone();
                let (then_block, then_binding) = self.lower_expr(then_block, &mut then_scope)?;
                let mut else_scope = scope.clone();
                let (else_block, else_binding) = self.lower_expr(else_block, &mut else_scope)?;
                scope.merge_from(then_scope);
                scope.merge_from(else_scope);
                (
                    Expr::If {
                        condition: Box::new(condition),
                        then_block: Box::new(then_block),
                        else_block: Box::new(else_block),
                    },
                    Some(Binding::Value(union_type(vec![
                        binding_type(then_binding.as_ref()),
                        binding_type(else_binding.as_ref()),
                    ]))),
                )
            }
            Expr::For {
                binding,
                iterable,
                body,
            } => {
                let iterable = self.lower_expr(iterable, scope)?.0;
                let previous = scope.bind(binding.as_str(), Binding::Value(TypeExpr::Any));
                let body = self.lower_expr(body, scope)?.0;
                scope.restore(binding.as_str(), previous);
                (
                    Expr::For {
                        binding: binding.clone(),
                        iterable: Box::new(iterable),
                        body: Box::new(body),
                    },
                    Some(Binding::Value(TypeExpr::Null)),
                )
            }
            Expr::While { condition, body } => {
                let condition = self.lower_expr(condition, scope)?.0;
                let body = self.lower_expr(body, scope)?.0;
                (
                    Expr::While {
                        condition: Box::new(condition),
                        body: Box::new(body),
                    },
                    Some(Binding::Value(TypeExpr::Null)),
                )
            }
            Expr::StartProcess(start) => {
                self.ensure_feature(self.surface.abilities.processes, "processes", scope.span)?;
                let Some(process) = self.program.process(start.process.as_str()) else {
                    return Err(LinkError::UnknownProcess {
                        name: start.process.to_string(),
                        span: scope.span,
                    });
                };
                let mut seen = BTreeSet::new();
                let mut lowered_args = Vec::with_capacity(start.args.len());
                for (arg, value) in &start.args {
                    if !seen.insert(arg.to_string()) {
                        return Err(LinkError::DuplicateProcessArgument {
                            arg: arg.to_string(),
                            span: scope.span,
                        });
                    }
                    let Some(param) = process.params.iter().find(|param| param.name == *arg) else {
                        return Err(LinkError::UnexpectedProcessArgument {
                            process: process.name.to_string(),
                            arg: arg.to_string(),
                            span: scope.span,
                        });
                    };
                    let (lowered, binding) = self.lower_expr(value, scope)?;
                    self.validate_process_arg_binding(
                        process.name.as_str(),
                        arg.as_str(),
                        &param.ty,
                        binding.as_ref(),
                        scope.span,
                    )?;
                    lowered_args.push((arg.clone(), lowered));
                }
                for param in &process.params {
                    if !seen.contains(param.name.as_str()) {
                        return Err(LinkError::MissingProcessArgument {
                            process: process.name.to_string(),
                            arg: param.name.to_string(),
                            span: scope.span,
                        });
                    }
                }
                (
                    Expr::StartProcess(crate::ast::ProcessStartExpr {
                        process: start.process.clone(),
                        args: lowered_args,
                    }),
                    Some(Binding::Value(TypeExpr::Any)),
                )
            }
            Expr::ProcessRef { process } => {
                let Some(process_ty) = self.process_types.get(process.as_str()) else {
                    return Err(LinkError::UnknownProcess {
                        name: process.to_string(),
                        span: scope.span,
                    });
                };
                (
                    Expr::ProcessRef {
                        process: process.clone(),
                    },
                    Some(Binding::Value(process_ty.clone())),
                )
            }
            Expr::HostValueConstructor { type_name, input } => (
                Expr::HostValueConstructor {
                    type_name: type_name.clone(),
                    input: Box::new(self.lower_expr(input, scope)?.0),
                },
                Some(Binding::Value(TypeExpr::Ref(type_name.clone()))),
            ),
            Expr::ResourceRef(resource) => {
                let resource = self.validate_resource_ref(resource, scope.span)?;
                (
                    Expr::ResourceRef(resource.clone()),
                    Some(Binding::Resource {
                        resource_type: resource.resource_type.to_string(),
                    }),
                )
            }
            Expr::ReceiverCall {
                receiver,
                operation,
                args,
            } => {
                if let Some(mut path) = module_path_for_expr(receiver) {
                    path.push(operation.clone());
                    if let Some(constructor) =
                        self.surface.resources.resolve_value_constructor(&path)
                    {
                        if args.len() != 1 {
                            return Err(LinkError::IncompatibleConstructorInput {
                                path: module_path_key(&path),
                                expected: format_type_expr(&constructor.input_ty),
                                actual: format!("{} arguments", args.len()),
                                span: scope.span,
                            });
                        }
                        let (input, input_binding) = self.lower_expr(&args[0], scope)?;
                        let actual_ty = binding_type(input_binding.as_ref());
                        if !self.is_type_assignable(&actual_ty, &constructor.input_ty) {
                            return Err(LinkError::IncompatibleConstructorInput {
                                path: module_path_key(&path),
                                expected: format_type_expr(
                                    &self.resolve_type_aliases(&constructor.input_ty),
                                ),
                                actual: format_type_expr(&self.resolve_type_aliases(&actual_ty)),
                                span: scope.span,
                            });
                        }
                        return Ok((
                            Expr::HostValueConstructor {
                                type_name: constructor.type_name.clone().into(),
                                input: Box::new(input),
                            },
                            Some(Binding::Value(constructor.output_ty.clone())),
                        ));
                    }
                }
                let (lowered_receiver, resource_type) =
                    if let Some(resource) = self.resolve_module_expr(receiver, scope) {
                        (
                            Expr::ResourceRef(resource.clone()),
                            Some(resource.resource_type.to_string()),
                        )
                    } else {
                        let (lowered_receiver, binding) = self.lower_expr(receiver, scope)?;
                        let resource_type = match binding {
                            Some(Binding::Resource { resource_type }) => Some(resource_type),
                            _ => None,
                        };
                        (lowered_receiver, resource_type)
                    };
                let Some(resource_type) = resource_type else {
                    if let Some(path) = module_path_for_expr(receiver) {
                        let suggestions = self
                            .surface
                            .resources
                            .operation_suggestions_for_prefix(&path, operation.as_str());
                        if !suggestions.is_empty() {
                            return Err(LinkError::AmbiguousModuleOperation {
                                module_path: module_path_key(&path),
                                operation: operation.to_string(),
                                suggestions,
                                span: scope.span,
                            });
                        }
                    }
                    return Err(LinkError::UnresolvedReceiver {
                        operation: operation.to_string(),
                        span: scope.span,
                    });
                };
                let Some(operation_binding) = self
                    .surface
                    .resources
                    .resolve_operation(&resource_type, operation)
                    .cloned()
                else {
                    return Err(LinkError::UnknownResourceOperation {
                        resource_type,
                        operation: operation.to_string(),
                        span: scope.span,
                    });
                };
                let trigger_operation = crate::TriggerHostOperation::from_host_operation(
                    operation_binding.host_operation.as_str(),
                );
                if let Some(operation) = trigger_operation {
                    self.ensure_feature(self.surface.abilities.triggers, "triggers", scope.span)?;
                    let (lowered_args, output_ty) =
                        self.lower_trigger_operation_args(operation, args, scope)?;
                    return Ok((
                        Expr::ReceiverCall {
                            receiver: Box::new(lowered_receiver),
                            operation: operation_binding.host_operation.into(),
                            args: lowered_args,
                        },
                        Some(Binding::Value(output_ty)),
                    ));
                }
                let mut lowered_args = Vec::with_capacity(args.len());
                let mut arg_types = Vec::with_capacity(args.len());
                for arg in args {
                    let (arg, binding) = self.lower_expr(arg, scope)?;
                    lowered_args.push(arg);
                    arg_types.push(binding_type(binding.as_ref()));
                }
                let actual_input = call_input_type(arg_types);
                if !self.is_type_assignable(&actual_input, &operation_binding.input_ty) {
                    return Err(LinkError::IncompatibleOperationInput {
                        operation: operation_binding.host_operation.clone(),
                        expected: format_type_expr(
                            &self.resolve_type_aliases(&operation_binding.input_ty),
                        ),
                        actual: format_type_expr(&self.resolve_type_aliases(&actual_input)),
                        span: scope.span,
                    });
                }
                (
                    Expr::ReceiverCall {
                        receiver: Box::new(lowered_receiver),
                        operation: operation_binding.host_operation.into(),
                        args: lowered_args,
                    },
                    Some(Binding::Value(operation_binding.output_ty.clone())),
                )
            }
            Expr::Await(inner) => {
                let (inner, binding) = self.lower_expr(inner, scope)?;
                (Expr::Await(Box::new(inner)), binding)
            }
            Expr::SleepFor(inner) => {
                self.ensure_feature(self.surface.abilities.sleep, "sleep", scope.span)?;
                (
                    Expr::SleepFor(Box::new(self.lower_expr(inner, scope)?.0)),
                    Some(Binding::Value(TypeExpr::Null)),
                )
            }
            Expr::SleepUntil(inner) => {
                self.ensure_feature(self.surface.abilities.sleep, "sleep", scope.span)?;
                (
                    Expr::SleepUntil(Box::new(self.lower_expr(inner, scope)?.0)),
                    Some(Binding::Value(TypeExpr::Null)),
                )
            }
            Expr::WaitSignal => {
                self.ensure_feature(
                    self.surface.abilities.process_signals,
                    "process signals",
                    scope.span,
                )?;
                if !scope.process_body {
                    return Err(LinkError::ProcessLifecycleOutsideProcess {
                        keyword: "wait signal",
                        span: scope.span,
                    });
                }
                (Expr::WaitSignal, Some(Binding::Value(TypeExpr::Any)))
            }
            Expr::SignalRun { run, payload } => {
                self.ensure_feature(
                    self.surface.abilities.process_signals,
                    "process signals",
                    scope.span,
                )?;
                if !scope.process_body {
                    return Err(LinkError::ProcessLifecycleOutsideProcess {
                        keyword: "signal run",
                        span: scope.span,
                    });
                }
                (
                    Expr::SignalRun {
                        run: Box::new(self.lower_expr(run, scope)?.0),
                        payload: Box::new(self.lower_expr(payload, scope)?.0),
                    },
                    Some(Binding::Value(TypeExpr::Null)),
                )
            }
            Expr::ResultUnwrap(inner) => {
                let (inner, binding) = self.lower_expr(inner, scope)?;
                (Expr::ResultUnwrap(Box::new(inner)), binding)
            }
            Expr::Cancel(inner) => (
                Expr::Cancel(Box::new(self.lower_expr(inner, scope)?.0)),
                Some(Binding::Value(TypeExpr::Any)),
            ),
            Expr::Print(inner) => (
                Expr::Print(Box::new(self.lower_expr(inner, scope)?.0)),
                Some(Binding::Value(TypeExpr::Null)),
            ),
            Expr::Submit(inner) => (
                Expr::Submit(
                    inner
                        .as_deref()
                        .map(|inner| {
                            self.lower_expr(inner, scope)
                                .map(|(expr, _)| Box::new(expr))
                        })
                        .transpose()?,
                ),
                Some(Binding::Value(TypeExpr::Null)),
            ),
            Expr::Yield(inner) => (
                Expr::Yield(Box::new(self.lower_expr(inner, scope)?.0)),
                Some(Binding::Value(TypeExpr::Null)),
            ),
            Expr::Wake(inner) => (
                Expr::Wake(Box::new(self.lower_expr(inner, scope)?.0)),
                Some(Binding::Value(TypeExpr::Null)),
            ),
            Expr::Finish(inner) => {
                let mut finish_ty = TypeExpr::Null;
                let inner = inner
                    .as_deref()
                    .map(|inner| {
                        let (expr, binding) = self.lower_expr(inner, scope)?;
                        finish_ty = binding_type(binding.as_ref());
                        Ok(Box::new(expr))
                    })
                    .transpose()?;
                (Expr::Finish(inner), Some(Binding::Value(finish_ty)))
            }
            Expr::Fail(inner) => (
                Expr::Fail(Box::new(self.lower_expr(inner, scope)?.0)),
                Some(Binding::Value(TypeExpr::Null)),
            ),
            Expr::BuiltinCall { name, args } => {
                if !crate::builtins::is_builtin(name.as_str()) {
                    if let Some(suggestion) = self
                        .surface
                        .resources
                        .operation_suggestions_for_host(name.as_str())
                        .into_iter()
                        .next()
                    {
                        return Err(LinkError::BareToolCall {
                            name: name.to_string(),
                            suggestion,
                            span: scope.span,
                        });
                    }
                    return Err(LinkError::UnknownBuiltin {
                        name: name.to_string(),
                        span: scope.span,
                    });
                }
                (
                    Expr::BuiltinCall {
                        name: name.clone(),
                        args: args
                            .iter()
                            .map(|arg| self.lower_expr(arg, scope).map(|(expr, _)| expr))
                            .collect::<Result<Vec<_>, _>>()?,
                    },
                    Some(Binding::Value(builtin_return_type(name.as_str()))),
                )
            }
            Expr::Field { target, field } => {
                let (target, binding) = self.lower_expr(target, scope)?;
                let ty =
                    self.field_type(&binding_type(binding.as_ref()), field.as_str(), scope.span)?;
                (
                    Expr::Field {
                        target: Box::new(target),
                        field: field.clone(),
                    },
                    Some(Binding::Value(ty)),
                )
            }
            Expr::Index { target, index } => {
                let (target, target_binding) = self.lower_expr(target, scope)?;
                let index = self.lower_expr(index, scope)?.0;
                (
                    Expr::Index {
                        target: Box::new(target),
                        index: Box::new(index),
                    },
                    Some(Binding::Value(self.index_type(
                        &binding_type(target_binding.as_ref()),
                        scope.span,
                    )?)),
                )
            }
            Expr::Unary { op, expr } => (
                Expr::Unary {
                    op: *op,
                    expr: Box::new(self.lower_expr(expr, scope)?.0),
                },
                Some(Binding::Value(match op {
                    crate::ast::UnaryOp::Not => TypeExpr::Bool,
                    crate::ast::UnaryOp::Negate => TypeExpr::Float,
                })),
            ),
            Expr::Binary { left, op, right } => (
                Expr::Binary {
                    left: Box::new(self.lower_expr(left, scope)?.0),
                    op: *op,
                    right: Box::new(self.lower_expr(right, scope)?.0),
                },
                Some(Binding::Value(binary_return_type(*op))),
            ),
        })
    }

    fn resolve_module_expr(&self, expr: &Expr, scope: &Scope) -> Option<ResourceRefExpr> {
        let path = module_path_for_expr(expr)?;
        if path
            .first()
            .and_then(|root| scope.get_str(root.as_str()))
            .is_some()
        {
            return None;
        }
        self.surface.resources.resolve_module_path(&path)
    }

    fn reject_trigger_event_special_form(
        &self,
        expr: &Expr,
        span: Option<Span>,
    ) -> Result<(), LinkError> {
        if is_trigger_event_projection_expr(expr) {
            return Err(LinkError::TriggerEventProjection { span });
        }
        if is_trigger_event_expr(expr) {
            return Err(LinkError::TriggerEventOutsideInputs { span });
        }
        Ok(())
    }

    fn validate_process_arg_binding(
        &self,
        process: &str,
        arg: &str,
        expected_ty: &TypeExpr,
        actual: Option<&Binding>,
        span: Option<Span>,
    ) -> Result<(), LinkError> {
        let Some(expected_resource) = self.resource_type_for_type(expected_ty) else {
            return Ok(());
        };
        match actual {
            Some(Binding::Resource { resource_type }) if *resource_type == expected_resource => {
                Ok(())
            }
            Some(Binding::Resource { resource_type }) => {
                Err(LinkError::IncompatibleProcessArgument {
                    process: process.to_string(),
                    arg: arg.to_string(),
                    expected: expected_resource,
                    actual: resource_type.clone(),
                    span,
                })
            }
            _ => Err(LinkError::IncompatibleProcessArgument {
                process: process.to_string(),
                arg: arg.to_string(),
                expected: expected_resource,
                actual: "value".to_string(),
                span,
            }),
        }
    }

    fn validate_trigger_operation_args(
        &self,
        operation: crate::TriggerHostOperation,
        args: &[Expr],
        scope: &Scope,
    ) -> Result<TypeExpr, LinkError> {
        match operation {
            crate::TriggerHostOperation::Register => {
                let call = crate::register_call_args(args)
                    .map_err(|_| LinkError::InvalidTriggerRegistration { span: scope.span })?;
                let source_ty = self.infer_expr_type(call.source, &mut scope.clone())?;
                let event_ty = self
                    .surface
                    .resources
                    .trigger_source_event(&source_ty)
                    .ok_or_else(|| LinkError::UnknownTriggerSourceType {
                        source_ty: format_type_expr(&source_ty),
                        span: scope.span,
                    })?;
                let target_ty = self.infer_expr_type(call.target, &mut scope.clone())?;
                let params = self.trigger_target_params(call.target, &target_ty, scope.span)?;
                let mut validation_scope = scope.clone();
                self.lower_trigger_input_record(
                    trigger_target_process_label(call.target).as_str(),
                    &params,
                    &event_ty,
                    call.inputs,
                    &mut validation_scope,
                )?;
                Ok(TypeExpr::TriggerHandle(Box::new(event_ty)))
            }
            crate::TriggerHostOperation::List => {
                let call = crate::list_call_args(args)
                    .map_err(|_| LinkError::InvalidTriggerList { span: scope.span })?;
                for (name, expr) in call.entries {
                    match name.as_str() {
                        "target" => {
                            let target_ty = self.infer_expr_type(expr, &mut scope.clone())?;
                            if !matches!(target_ty, TypeExpr::Process { .. }) {
                                return Err(LinkError::InvalidTriggerTarget {
                                    actual: format_type_expr(&target_ty),
                                    span: scope.span,
                                });
                            }
                        }
                        "name" | "source_type" => {
                            let filter_ty = self.infer_expr_type(expr, &mut scope.clone())?;
                            if !self.is_type_assignable(&filter_ty, &TypeExpr::Str) {
                                return Err(LinkError::IncompatibleOperationInput {
                                    operation: operation.host_operation().to_string(),
                                    expected: format_type_expr(&TypeExpr::Str),
                                    actual: format_type_expr(&filter_ty),
                                    span: scope.span,
                                });
                            }
                        }
                        "enabled" => {
                            let filter_ty = self.infer_expr_type(expr, &mut scope.clone())?;
                            if !self.is_type_assignable(&filter_ty, &TypeExpr::Bool) {
                                return Err(LinkError::IncompatibleOperationInput {
                                    operation: operation.host_operation().to_string(),
                                    expected: format_type_expr(&TypeExpr::Bool),
                                    actual: format_type_expr(&filter_ty),
                                    span: scope.span,
                                });
                            }
                        }
                        _ => unreachable!("list_call_args rejects unknown trigger filters"),
                    }
                }
                Ok(operation.output_ty())
            }
            crate::TriggerHostOperation::Cancel => {
                crate::cancel_call_args(args)
                    .map_err(|_| LinkError::InvalidTriggerCancel { span: scope.span })?;
                Ok(operation.output_ty())
            }
        }
    }

    fn lower_trigger_operation_args(
        &self,
        operation: crate::TriggerHostOperation,
        args: &[Expr],
        scope: &mut Scope,
    ) -> Result<(Vec<Expr>, TypeExpr), LinkError> {
        match operation {
            crate::TriggerHostOperation::Register => {
                self.lower_trigger_registration_args(args, scope)
            }
            crate::TriggerHostOperation::List => {
                let call = crate::list_call_args(args)
                    .map_err(|_| LinkError::InvalidTriggerList { span: scope.span })?;
                let mut entries = Vec::with_capacity(call.entries.len());
                for (name, expr) in call.entries {
                    match name.as_str() {
                        "target" => {
                            let target_ty = self.infer_expr_type(expr, &mut scope.clone())?;
                            if !matches!(target_ty, TypeExpr::Process { .. }) {
                                return Err(LinkError::InvalidTriggerTarget {
                                    actual: format_type_expr(&target_ty),
                                    span: scope.span,
                                });
                            }
                        }
                        "name" | "source_type" => {
                            let filter_ty = self.infer_expr_type(expr, &mut scope.clone())?;
                            if !self.is_type_assignable(&filter_ty, &TypeExpr::Str) {
                                return Err(LinkError::IncompatibleOperationInput {
                                    operation: operation.host_operation().to_string(),
                                    expected: format_type_expr(&TypeExpr::Str),
                                    actual: format_type_expr(&filter_ty),
                                    span: scope.span,
                                });
                            }
                        }
                        "enabled" => {
                            let filter_ty = self.infer_expr_type(expr, &mut scope.clone())?;
                            if !self.is_type_assignable(&filter_ty, &TypeExpr::Bool) {
                                return Err(LinkError::IncompatibleOperationInput {
                                    operation: operation.host_operation().to_string(),
                                    expected: format_type_expr(&TypeExpr::Bool),
                                    actual: format_type_expr(&filter_ty),
                                    span: scope.span,
                                });
                            }
                        }
                        _ => unreachable!("list_call_args rejects unknown trigger filters"),
                    }
                    entries.push((name.clone(), self.lower_expr(expr, scope)?.0));
                }
                Ok((vec![Expr::Record(entries)], operation.output_ty()))
            }
            crate::TriggerHostOperation::Cancel => {
                let call = crate::cancel_call_args(args)
                    .map_err(|_| LinkError::InvalidTriggerCancel { span: scope.span })?;
                Ok((
                    vec![Expr::Record(vec![(
                        "handle".into(),
                        self.lower_expr(call.handle, scope)?.0,
                    )])],
                    operation.output_ty(),
                ))
            }
        }
    }

    fn lower_trigger_registration_args(
        &self,
        args: &[Expr],
        scope: &mut Scope,
    ) -> Result<(Vec<Expr>, TypeExpr), LinkError> {
        let call = crate::register_call_args(args)
            .map_err(|_| LinkError::InvalidTriggerRegistration { span: scope.span })?;
        let source_ty = self.infer_expr_type(call.source, &mut scope.clone())?;
        let event_ty = self
            .surface
            .resources
            .trigger_source_event(&source_ty)
            .ok_or_else(|| LinkError::UnknownTriggerSourceType {
                source_ty: format_type_expr(&source_ty),
                span: scope.span,
            })?;
        let target_ty = self.infer_expr_type(call.target, &mut scope.clone())?;
        let params = self.trigger_target_params(call.target, &target_ty, scope.span)?;
        let process = trigger_target_process_label(call.target);

        let source = self.lower_expr(call.source, scope)?.0;
        let target = self.lower_expr(call.target, scope)?.0;
        let inputs = self.lower_trigger_input_record(
            process.as_str(),
            &params,
            &event_ty,
            call.inputs,
            scope,
        )?;
        let mut entries = vec![
            ("source".into(), source),
            ("target".into(), target),
            ("inputs".into(), inputs),
        ];
        if let Some(name) = call.name {
            entries.push(("name".into(), self.lower_expr(name, scope)?.0));
        }
        Ok((
            vec![Expr::Record(entries)],
            TypeExpr::TriggerHandle(Box::new(event_ty)),
        ))
    }

    fn lower_trigger_input_record(
        &self,
        process: &str,
        params: &[ProcessParam],
        event_ty: &TypeExpr,
        inputs: &Expr,
        scope: &mut Scope,
    ) -> Result<Expr, LinkError> {
        let Expr::Record(entries) = inputs else {
            return Err(LinkError::InvalidTriggerInputs { span: scope.span });
        };
        let mut seen = BTreeSet::new();
        let mut saw_event = false;
        let mut lowered = Vec::with_capacity(entries.len());
        for (name, value) in entries {
            if !seen.insert(name.to_string()) {
                return Err(LinkError::DuplicateTriggerInput {
                    input: name.to_string(),
                    span: scope.span,
                });
            }
            let Some(param) = params.iter().find(|param| param.name == *name) else {
                return Err(LinkError::UnknownTriggerInput {
                    process: process.to_string(),
                    input: name.to_string(),
                    span: scope.span,
                });
            };
            if is_trigger_event_projection_expr(value) {
                return Err(LinkError::TriggerEventProjection { span: scope.span });
            }
            if is_trigger_event_expr(value) {
                saw_event = true;
                if !self.is_type_assignable(event_ty, &param.ty) {
                    return Err(LinkError::TriggerEventMismatch {
                        event: format_type_expr(&self.resolve_type_aliases(event_ty)),
                        input_name: name.to_string(),
                        input: format_type_expr(&self.resolve_type_aliases(&param.ty)),
                        span: scope.span,
                    });
                }
                lowered.push((name.clone(), crate::trigger_event_placeholder_expr()));
                continue;
            }
            let (lowered_value, binding) = self.lower_expr(value, scope)?;
            self.validate_process_arg_binding(
                process,
                name.as_str(),
                &param.ty,
                binding.as_ref(),
                scope.span,
            )?;
            lowered.push((name.clone(), lowered_value));
        }
        for param in params {
            if !seen.contains(param.name.as_str()) {
                return Err(LinkError::MissingTriggerInput {
                    process: process.to_string(),
                    input: param.name.to_string(),
                    span: scope.span,
                });
            }
        }
        if !saw_event {
            return Err(LinkError::MissingTriggerEventInput { span: scope.span });
        }
        Ok(Expr::Record(lowered))
    }

    fn trigger_target_params(
        &self,
        target: &Expr,
        target_ty: &TypeExpr,
        span: Option<Span>,
    ) -> Result<Vec<ProcessParam>, LinkError> {
        if let Some(process_name) = trigger_target_process_name(target)
            && let Some(process) = self.program.process(process_name.as_str())
        {
            return Ok(process.params.clone());
        }
        let TypeExpr::Process {
            input, input_count, ..
        } = target_ty
        else {
            return Err(LinkError::InvalidTriggerTarget {
                actual: format_type_expr(target_ty),
                span,
            });
        };
        match (input_count, input.as_ref()) {
            (0, _) => Ok(Vec::new()),
            (count, TypeExpr::Object(fields)) if *count > 1 => Ok(fields
                .iter()
                .map(|field| ProcessParam {
                    name: field.name.clone(),
                    ty: field.ty.clone(),
                })
                .collect()),
            _ => Err(LinkError::InvalidTriggerTarget {
                actual: format_type_expr(target_ty),
                span,
            }),
        }
    }

    fn infer_process_output(
        &self,
        process: &ProcessDecl,
        span: Option<Span>,
    ) -> Result<TypeExpr, LinkError> {
        let mut scope = Scope::new(false, true, span);
        for param in &process.params {
            scope.bind(param.name.as_str(), self.binding_for_type(&param.ty));
        }
        scope.bind("input", Binding::Value(process_input_type(process)));
        scope.bind("inputs", Binding::Value(process_input_record_type(process)));
        let completion = self.infer_completion(&process.body, &mut scope)?;
        let mut outputs = completion.finishes;
        if completion.can_fallthrough {
            outputs.push(TypeExpr::Null);
        }
        Ok(union_type(outputs))
    }

    fn infer_completion(&self, expr: &Expr, scope: &mut Scope) -> Result<Completion, LinkError> {
        match expr {
            Expr::LabelAnnotated { expr, .. } => self.infer_completion(expr, scope),
            Expr::Finish(Some(value)) => Ok(Completion {
                finishes: vec![self.infer_expr_type(value, scope)?],
                can_fallthrough: false,
            }),
            Expr::Finish(None) => Ok(Completion {
                finishes: vec![TypeExpr::Null],
                can_fallthrough: false,
            }),
            Expr::Fail(_) => Ok(Completion {
                finishes: Vec::new(),
                can_fallthrough: false,
            }),
            Expr::Block(expressions) => {
                let mut finishes = Vec::new();
                let mut can_fallthrough = true;
                for expression in expressions {
                    if !can_fallthrough {
                        break;
                    }
                    let completion = self.infer_completion(expression, scope)?;
                    finishes.extend(completion.finishes);
                    can_fallthrough = completion.can_fallthrough;
                }
                Ok(Completion {
                    finishes,
                    can_fallthrough,
                })
            }
            Expr::If {
                condition,
                then_block,
                else_block,
            } => {
                self.infer_expr_type(condition, scope)?;
                let mut then_scope = scope.clone();
                let then_completion = self.infer_completion(then_block, &mut then_scope)?;
                let mut else_scope = scope.clone();
                let else_completion = self.infer_completion(else_block, &mut else_scope)?;
                scope.merge_from(then_scope);
                scope.merge_from(else_scope);
                let mut finishes = then_completion.finishes;
                finishes.extend(else_completion.finishes);
                Ok(Completion {
                    finishes,
                    can_fallthrough: then_completion.can_fallthrough
                        || else_completion.can_fallthrough,
                })
            }
            Expr::For {
                binding,
                iterable,
                body,
            } => {
                self.infer_expr_type(iterable, scope)?;
                let previous = scope.bind(binding.as_str(), Binding::Value(TypeExpr::Any));
                let mut completion = self.infer_completion(body, scope)?;
                scope.restore(binding.as_str(), previous);
                completion.can_fallthrough = true;
                Ok(completion)
            }
            Expr::While { condition, body } => {
                self.infer_expr_type(condition, scope)?;
                let mut completion = self.infer_completion(body, scope)?;
                completion.can_fallthrough = true;
                Ok(completion)
            }
            Expr::Assign { target, expr } if target.steps.is_empty() => {
                let ty = self.infer_expr_type(expr, scope)?;
                scope.bind(target.root.as_str(), self.binding_for_type(&ty));
                Ok(Completion::fallthrough())
            }
            other => {
                self.infer_expr_type(other, scope)?;
                Ok(Completion::fallthrough())
            }
        }
    }

    fn infer_expr_type(&self, expr: &Expr, scope: &mut Scope) -> Result<TypeExpr, LinkError> {
        self.reject_trigger_event_special_form(expr, scope.span)?;
        if matches!(expr, Expr::Variable(_) | Expr::Field { .. })
            && let Some(resource) = self.resolve_module_expr(expr, scope)
        {
            return Ok(TypeExpr::Ref(resource.resource_type));
        }
        Ok(match expr {
            Expr::LabelAnnotated { expr, .. } => self.infer_expr_type(expr, scope)?,
            Expr::Block(expressions) => {
                let mut last = TypeExpr::Null;
                for expression in expressions {
                    last = self.infer_expr_type(expression, scope)?;
                }
                last
            }
            Expr::Null
            | Expr::Bool(_)
            | Expr::Number(_)
            | Expr::String(_)
            | Expr::Break
            | Expr::Continue
            | Expr::TypeLiteral(_) => literal_type(expr),
            Expr::Variable(name) => {
                if let Some(binding) = scope.get(name) {
                    binding_type(Some(&binding))
                } else if let Some(process_ty) = self.process_types.get(name.as_str()) {
                    process_ty.clone()
                } else if scope.allow_unknown_globals {
                    TypeExpr::Any
                } else {
                    return Err(LinkError::UnknownName {
                        name: name.to_string(),
                        span: scope.span,
                    });
                }
            }
            Expr::ProcessRef { process } => self
                .process_types
                .get(process.as_str())
                .cloned()
                .ok_or_else(|| LinkError::UnknownProcess {
                    name: process.to_string(),
                    span: scope.span,
                })?,
            Expr::HostValueConstructor { type_name, .. } => TypeExpr::Ref(type_name.clone()),
            Expr::List(items) => TypeExpr::List(Box::new(union_type(
                items
                    .iter()
                    .map(|item| self.infer_expr_type(item, scope))
                    .collect::<Result<Vec<_>, _>>()?,
            ))),
            Expr::Record(entries) => TypeExpr::Object(
                entries
                    .iter()
                    .map(|(name, value)| {
                        Ok(TypeField {
                            name: name.clone(),
                            ty: self.infer_expr_type(value, scope)?,
                            optional: false,
                        })
                    })
                    .collect::<Result<Vec<_>, LinkError>>()?,
            ),
            Expr::Assign { target, expr } => {
                let ty = self.infer_expr_type(expr, scope)?;
                if target.steps.is_empty() {
                    scope.bind(target.root.as_str(), self.binding_for_type(&ty));
                }
                ty
            }
            Expr::If {
                condition,
                then_block,
                else_block,
            } => {
                self.infer_expr_type(condition, scope)?;
                let mut then_scope = scope.clone();
                let then_ty = self.infer_expr_type(then_block, &mut then_scope)?;
                let mut else_scope = scope.clone();
                let else_ty = self.infer_expr_type(else_block, &mut else_scope)?;
                scope.merge_from(then_scope);
                scope.merge_from(else_scope);
                union_type(vec![then_ty, else_ty])
            }
            Expr::For {
                binding,
                iterable,
                body,
            } => {
                self.infer_expr_type(iterable, scope)?;
                let previous = scope.bind(binding.as_str(), Binding::Value(TypeExpr::Any));
                self.infer_expr_type(body, scope)?;
                scope.restore(binding.as_str(), previous);
                TypeExpr::Null
            }
            Expr::While { condition, body } => {
                self.infer_expr_type(condition, scope)?;
                self.infer_expr_type(body, scope)?;
                TypeExpr::Null
            }
            Expr::StartProcess(_) => TypeExpr::Any,
            Expr::ResourceRef(resource) => TypeExpr::Ref(resource.resource_type.clone()),
            Expr::ReceiverCall {
                receiver,
                operation,
                args,
            } => {
                if let Some(mut path) = module_path_for_expr(receiver) {
                    path.push(operation.clone());
                    if let Some(constructor) =
                        self.surface.resources.resolve_value_constructor(&path)
                    {
                        return Ok(constructor.output_ty.clone());
                    }
                }
                let resource_type =
                    if let Some(resource) = self.resolve_module_expr(receiver, scope) {
                        resource.resource_type.to_string()
                    } else {
                        let receiver_ty = self.infer_expr_type(receiver, scope)?;
                        self.resource_type_for_type(&receiver_ty).ok_or_else(|| {
                            LinkError::UnresolvedReceiver {
                                operation: operation.to_string(),
                                span: scope.span,
                            }
                        })?
                    };
                let binding = self
                    .surface
                    .resources
                    .resolve_operation(&resource_type, operation)
                    .ok_or_else(|| LinkError::UnknownResourceOperation {
                        resource_type,
                        operation: operation.to_string(),
                        span: scope.span,
                    })?;
                match crate::TriggerHostOperation::from_host_operation(
                    binding.host_operation.as_str(),
                ) {
                    Some(operation) => {
                        self.validate_trigger_operation_args(operation, args, scope)?
                    }
                    None => binding.output_ty.clone(),
                }
            }
            Expr::Await(inner) | Expr::ResultUnwrap(inner) => self.infer_expr_type(inner, scope)?,
            Expr::SleepFor(_) | Expr::SleepUntil(_) => TypeExpr::Null,
            Expr::WaitSignal => TypeExpr::Any,
            Expr::SignalRun { .. }
            | Expr::Cancel(_)
            | Expr::Print(_)
            | Expr::Submit(_)
            | Expr::Yield(_)
            | Expr::Wake(_)
            | Expr::Fail(_) => TypeExpr::Null,
            Expr::Finish(Some(inner)) => self.infer_expr_type(inner, scope)?,
            Expr::Finish(None) => TypeExpr::Null,
            Expr::BuiltinCall { name, .. } => builtin_return_type(name.as_str()),
            Expr::Field { target, field } => {
                self.field_type(&self.infer_expr_type(target, scope)?, field, scope.span)?
            }
            Expr::Index { target, .. } => {
                self.index_type(&self.infer_expr_type(target, scope)?, scope.span)?
            }
            Expr::Unary { op, .. } => match op {
                crate::ast::UnaryOp::Not => TypeExpr::Bool,
                crate::ast::UnaryOp::Negate => TypeExpr::Float,
            },
            Expr::Binary { op, .. } => binary_return_type(*op),
        })
    }
}

#[derive(Clone)]
struct Scope {
    bindings: BTreeMap<String, Binding>,
    allow_unknown_globals: bool,
    process_body: bool,
    span: Option<Span>,
}

impl Scope {
    fn new(allow_unknown_globals: bool, process_body: bool, span: Option<Span>) -> Self {
        Self {
            bindings: BTreeMap::new(),
            allow_unknown_globals,
            process_body,
            span,
        }
    }

    fn bind(&mut self, name: &str, binding: Binding) -> Option<Binding> {
        self.bindings.insert(name.to_string(), binding)
    }

    fn restore(&mut self, name: &str, previous: Option<Binding>) {
        match previous {
            Some(binding) => {
                self.bindings.insert(name.to_string(), binding);
            }
            None => {
                self.bindings.remove(name);
            }
        }
    }

    fn get(&self, name: &AstString) -> Option<Binding> {
        self.bindings.get(name.as_str()).cloned()
    }

    fn get_str(&self, name: &str) -> Option<Binding> {
        self.bindings.get(name).cloned()
    }

    fn merge_from(&mut self, other: Scope) {
        for (name, binding) in other.bindings {
            self.bindings.entry(name).or_insert(binding);
        }
    }
}

struct Completion {
    finishes: Vec<TypeExpr>,
    can_fallthrough: bool,
}

impl Completion {
    fn fallthrough() -> Self {
        Self {
            finishes: Vec::new(),
            can_fallthrough: true,
        }
    }
}

fn any_binding() -> Binding {
    Binding::Value(TypeExpr::Any)
}

fn binding_type(binding: Option<&Binding>) -> TypeExpr {
    match binding {
        Some(Binding::Value(ty)) => ty.clone(),
        Some(Binding::Resource { resource_type }) => TypeExpr::Ref(resource_type.as_str().into()),
        None => TypeExpr::Any,
    }
}

fn literal_type(expr: &Expr) -> TypeExpr {
    match expr {
        Expr::Null => TypeExpr::Null,
        Expr::Bool(_) => TypeExpr::Bool,
        Expr::Number(_) => TypeExpr::Float,
        Expr::String(_) => TypeExpr::Str,
        Expr::TypeLiteral(_) => TypeExpr::Any,
        Expr::Break | Expr::Continue => TypeExpr::Null,
        Expr::LabelAnnotated { expr, .. } => literal_type(expr),
        _ => TypeExpr::Any,
    }
}

fn union_type(items: Vec<TypeExpr>) -> TypeExpr {
    let mut flattened = Vec::new();
    for item in items {
        match item {
            TypeExpr::Union(items) => flattened.extend(items),
            other => flattened.push(other),
        }
    }
    let mut unique = Vec::new();
    for item in flattened {
        if !unique.contains(&item) {
            unique.push(item);
        }
    }
    match unique.as_slice() {
        [] => TypeExpr::Null,
        [one] => one.clone(),
        _ => TypeExpr::Union(unique),
    }
}

fn call_input_type(arg_types: Vec<TypeExpr>) -> TypeExpr {
    match arg_types.as_slice() {
        [] => TypeExpr::Null,
        [one] => one.clone(),
        _ => TypeExpr::List(Box::new(union_type(arg_types))),
    }
}

fn field_type(
    target: &TypeExpr,
    field: &str,
    span: Option<Span>,
    is_opaque: impl Fn(&str) -> bool + Copy,
) -> Result<TypeExpr, LinkError> {
    match target {
        TypeExpr::Any | TypeExpr::Dict => Ok(TypeExpr::Any),
        TypeExpr::Ref(name) if is_opaque(name.as_str()) => Err(LinkError::OpaqueHostValueAccess {
            type_name: name.to_string(),
            access: format!(".{field}"),
            span,
        }),
        TypeExpr::Ref(_) => Ok(TypeExpr::Any),
        TypeExpr::Object(fields) => Ok(fields
            .iter()
            .find(|candidate| candidate.name.as_str() == field)
            .map(|field| field.ty.clone())
            .unwrap_or(TypeExpr::Any)),
        TypeExpr::Union(items) => {
            let fields = items
                .iter()
                .map(|item| field_type(item, field, span, is_opaque))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(union_type(fields))
        }
        _ => Ok(TypeExpr::Any),
    }
}

fn index_type(
    target: &TypeExpr,
    span: Option<Span>,
    is_opaque: impl Fn(&str) -> bool + Copy,
) -> Result<TypeExpr, LinkError> {
    match target {
        TypeExpr::List(item) => Ok(*item.clone()),
        TypeExpr::Ref(name) if is_opaque(name.as_str()) => Err(LinkError::OpaqueHostValueAccess {
            type_name: name.to_string(),
            access: "[]".to_string(),
            span,
        }),
        TypeExpr::Ref(_) => Ok(TypeExpr::Any),
        TypeExpr::Union(items) => {
            let items = items
                .iter()
                .map(|item| index_type(item, span, is_opaque))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(union_type(items))
        }
        _ => Ok(TypeExpr::Any),
    }
}

fn builtin_return_type(name: &str) -> TypeExpr {
    match name {
        "len" | "find" | "to_int" | "ceil_div" | "floor_div" => TypeExpr::Int,
        "empty" | "contains" | "starts_with" | "ends_with" => TypeExpr::Bool,
        "to_float" => TypeExpr::Float,
        "to_string" | "trim" | "join" => TypeExpr::Str,
        "keys" | "values" | "split" | "grep_text" | "range" | "push" => {
            TypeExpr::List(Box::new(TypeExpr::Any))
        }
        "json_parse" | "validate" | "format" => TypeExpr::Any,
        _ => TypeExpr::Any,
    }
}

fn binary_return_type(op: crate::ast::BinaryOp) -> TypeExpr {
    match op {
        crate::ast::BinaryOp::Equal
        | crate::ast::BinaryOp::NotEqual
        | crate::ast::BinaryOp::Less
        | crate::ast::BinaryOp::LessEqual
        | crate::ast::BinaryOp::Greater
        | crate::ast::BinaryOp::GreaterEqual
        | crate::ast::BinaryOp::And
        | crate::ast::BinaryOp::Or => TypeExpr::Bool,
        crate::ast::BinaryOp::Add
        | crate::ast::BinaryOp::Subtract
        | crate::ast::BinaryOp::Multiply
        | crate::ast::BinaryOp::Divide
        | crate::ast::BinaryOp::Modulo => TypeExpr::Float,
    }
}

fn process_input_type(process: &ProcessDecl) -> TypeExpr {
    match process.params.as_slice() {
        [] => TypeExpr::Null,
        [param] => param.ty.clone(),
        _ => process_input_record_type(process),
    }
}

fn process_input_record_type(process: &ProcessDecl) -> TypeExpr {
    TypeExpr::Object(
        process
            .params
            .iter()
            .map(|param| TypeField {
                name: param.name.clone(),
                ty: param.ty.clone(),
                optional: false,
            })
            .collect(),
    )
}

fn process_type_for_decl(process: &ProcessDecl, output: TypeExpr) -> TypeExpr {
    TypeExpr::Process {
        input: Box::new(process_input_type(process)),
        output: Box::new(output),
        input_count: process.params.len(),
    }
}

fn module_path_for_expr(expr: &Expr) -> Option<Vec<AstString>> {
    match expr {
        Expr::LabelAnnotated { expr, .. } => module_path_for_expr(expr),
        Expr::Variable(name) => Some(vec![name.clone()]),
        Expr::Field { target, field } => {
            let mut path = module_path_for_expr(target)?;
            path.push(field.clone());
            Some(path)
        }
        Expr::ResourceRef(resource) => Some(resource.path.clone()),
        _ => None,
    }
}

fn is_trigger_event_expr(expr: &Expr) -> bool {
    matches!(
        module_path_for_expr(expr).as_deref(),
        Some([trigger, event]) if trigger.as_str() == "trigger" && event.as_str() == "event"
    )
}

fn is_trigger_event_projection_expr(expr: &Expr) -> bool {
    module_path_for_expr(expr).is_some_and(|path| {
        path.len() > 2 && path[0].as_str() == "trigger" && path[1].as_str() == "event"
    })
}

fn trigger_target_process_name(expr: &Expr) -> Option<String> {
    match expr {
        Expr::LabelAnnotated { expr, .. } => trigger_target_process_name(expr),
        Expr::Variable(name) | Expr::ProcessRef { process: name } => Some(name.to_string()),
        _ => None,
    }
}

fn trigger_target_process_label(expr: &Expr) -> String {
    trigger_target_process_name(expr).unwrap_or_else(|| "target".to_string())
}

fn expr_has_label_annotation(expr: &Expr) -> bool {
    match expr {
        Expr::LabelAnnotated { .. } => true,
        other => other.children().any(expr_has_label_annotation),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resources() -> ResourceCatalog {
        let mut catalog = ResourceCatalog::new();
        catalog.add_module_instance(["tools"], "Tools");
        catalog.add_operation(
            "Tools",
            "read_file",
            "read_file",
            TypeExpr::Object(vec![TypeField {
                name: "path".into(),
                ty: TypeExpr::Str,
                optional: false,
            }]),
            TypeExpr::Str,
        );
        catalog.add_operation("Tools", "echo", "echo", TypeExpr::Any, TypeExpr::Any);
        crate::add_trigger_resource_operations(&mut catalog);
        catalog
            .add_trigger_source_constructor(
                ["timer", "Schedule"],
                TypeExpr::Object(vec![
                    TypeField {
                        name: "expr".into(),
                        ty: TypeExpr::Str,
                        optional: false,
                    },
                    TypeField {
                        name: "tz".into(),
                        ty: TypeExpr::Str,
                        optional: true,
                    },
                ]),
                NamedDataType::object(
                    "timer.Tick",
                    vec![TypeField {
                        name: "fired_at".into(),
                        ty: TypeExpr::Str,
                        optional: false,
                    }],
                )
                .expect("valid timer tick type"),
            )
            .expect("valid timer trigger source");
        catalog
    }

    fn full_surface() -> LashlangSurface {
        LashlangSurface::new(resources(), LashlangAbilities::all())
    }

    fn full_label_surface() -> LashlangSurface {
        full_surface()
            .with_language_features(LashlangLanguageFeatures::default().with_label_annotations())
    }

    fn timer_tick_type_with_field(field: &'static str) -> NamedDataType {
        NamedDataType::object(
            "timer.Tick",
            vec![TypeField {
                name: field.into(),
                ty: TypeExpr::Str,
                optional: false,
            }],
        )
        .expect("valid timer tick type")
    }

    fn resources_with_timer_event(event_type: NamedDataType) -> ResourceCatalog {
        let mut catalog = ResourceCatalog::new();
        crate::add_trigger_resource_operations(&mut catalog);
        catalog
            .add_trigger_source_constructor(
                ["timer", "Schedule"],
                TypeExpr::Object(vec![TypeField {
                    name: "expr".into(),
                    ty: TypeExpr::Str,
                    optional: false,
                }]),
                event_type,
            )
            .expect("valid timer trigger source");
        catalog
    }

    #[test]
    fn named_host_data_type_validation_rejects_invalid_shapes() {
        let duplicate_field = NamedDataType::object(
            "timer.Tick",
            vec![
                TypeField {
                    name: "fired_at".into(),
                    ty: TypeExpr::Str,
                    optional: false,
                },
                TypeField {
                    name: "fired_at".into(),
                    ty: TypeExpr::Str,
                    optional: false,
                },
            ],
        )
        .expect_err("duplicate fields should be rejected");
        assert!(matches!(
            duplicate_field,
            NamedDataTypeError::DuplicateField { .. }
        ));

        let nested_ref = NamedDataType::object(
            "timer.Tick",
            vec![TypeField {
                name: "nested".into(),
                ty: TypeExpr::Ref("Other.Type".into()),
                optional: false,
            }],
        )
        .expect_err("nested refs should be rejected");
        assert!(matches!(nested_ref, NamedDataTypeError::NestedRef { .. }));

        let duplicate_enum = NamedDataType::object(
            "timer.Tick",
            vec![TypeField {
                name: "kind".into(),
                ty: TypeExpr::Enum(vec!["Red".into(), "Red".into()]),
                optional: false,
            }],
        )
        .expect_err("duplicate enum values should be rejected");
        assert!(matches!(
            duplicate_enum,
            NamedDataTypeError::DuplicateEnumValue { .. }
        ));

        let simple_name = NamedDataType::object("Tick", vec![])
            .expect_err("host data type names must be qualified");
        assert!(matches!(
            simple_name,
            NamedDataTypeError::InvalidName { .. }
        ));
    }

    #[test]
    fn resource_catalog_rejects_conflicting_named_host_data_type_definitions() {
        let mut catalog = ResourceCatalog::new();
        catalog
            .add_named_data_type(timer_tick_type_with_field("fired_at"))
            .expect("first definition");
        let err = catalog
            .add_named_data_type(timer_tick_type_with_field("delivered_at"))
            .expect_err("same host type name with different shape should be rejected");

        assert!(matches!(
            err,
            ResourceCatalogError::ConflictingNamedDataType { .. }
        ));
    }

    #[test]
    fn linked_module_accepts_named_processes_resource_params_and_activations() {
        let program = crate::parse(
            r#"
            type ChangeEvent = { path: str }
            process scan(tool: Tools, event: ChangeEvent) {
              text = await tool.read_file({ path: "changed.txt" })?
              finish text
            }
            process watcher(run: any) {
              sleep for "0ms"
              signal = wait signal
              signal run run with signal
              finish signal
            }
            process from_tick(tick: timer.Tick) {
              finish tick.fired_at
            }
            source = timer.Schedule({ expr: "0 8 * * *", tz: "UTC" })
            handle = await triggers.register({
              source: source,
              target: from_tick,
              inputs: { tick: trigger.event },
              name: "changed"
            })?
            submit handle
            "#,
        )
        .expect("parse module");

        let linked = LinkedModule::link(program, full_surface()).expect("link module");

        assert!(
            linked
                .module_ref
                .as_str()
                .starts_with("lashlang:v1:sha256:")
        );
    }

    #[test]
    fn linked_module_allows_trigger_registration_name_to_match_target_process() {
        let program = crate::parse(
            r#"
            process changed(tick: timer.Tick) {
              finish true
            }
            source = timer.Schedule({ expr: "0 8 * * *" })
            await triggers.register({
              source: source,
              target: changed,
              inputs: { tick: trigger.event },
              name: "changed"
            })?
            "#,
        )
        .expect("parse module");

        LinkedModule::link(program, full_surface())
            .expect("trigger registration names and process names occupy different namespaces");
    }

    #[test]
    fn linked_module_resolves_host_named_data_refs_for_fields_and_structural_assignability() {
        let direct_ref = crate::parse(
            r#"
            process from_tick(tick: timer.Tick) {
              finish tick.fired_at
            }
            submit true
            "#,
        )
        .expect("parse direct host data ref");
        LinkedModule::link(direct_ref, full_surface()).expect("host data ref fields should link");

        let structural_input = crate::parse(
            r#"
            process from_tick(tick: { fired_at: str }) {
              finish tick.fired_at
            }
            source = timer.Schedule({ expr: "0 8 * * *" })
            await triggers.register({
              source: source,
              target: from_tick,
              inputs: { tick: trigger.event }
            })?
            "#,
        )
        .expect("parse structural target input");
        LinkedModule::link(structural_input, full_surface())
            .expect("host data shape should be structurally assignable");
    }

    #[test]
    fn linked_module_rejects_unknown_host_data_refs_and_opaque_source_field_access() {
        let unknown = crate::parse(
            r#"
            process from_tick(tick: foo.Tick) {
              finish true
            }
            submit true
            "#,
        )
        .expect("parse unknown host type");
        assert!(matches!(
            LinkedModule::link(unknown, full_surface()),
            Err(LinkError::UnknownType { name, .. }) if name == "foo.Tick"
        ));

        let opaque = crate::parse(
            r#"
            source = timer.Schedule({ expr: "0 8 * * *" })
            submit source.expr
            "#,
        )
        .expect("parse opaque source access");
        assert!(matches!(
            LinkedModule::link(opaque, full_surface()),
            Err(LinkError::OpaqueHostValueAccess { type_name, .. }) if type_name == "timer.Schedule"
        ));
    }

    #[test]
    fn required_surface_ref_tracks_host_named_data_type_shape_changes() {
        let program = crate::parse(
            r#"
            process from_tick(tick: any) {
              finish true
            }
            source = timer.Schedule({ expr: "0 8 * * *" })
            await triggers.register({
              source: source,
              target: from_tick,
              inputs: { tick: trigger.event }
            })?
            "#,
        )
        .expect("parse trigger registration");
        let first = LinkedModule::link(
            program.clone(),
            LashlangSurface::new(
                resources_with_timer_event(timer_tick_type_with_field("fired_at")),
                LashlangAbilities::all(),
            ),
        )
        .expect("link first host event shape");
        let second = LinkedModule::link(
            program,
            LashlangSurface::new(
                resources_with_timer_event(timer_tick_type_with_field("delivered_at")),
                LashlangAbilities::all(),
            ),
        )
        .expect("link changed host event shape");

        assert_ne!(first.required_surface_ref, second.required_surface_ref);
    }

    #[test]
    fn linked_module_accepts_top_level_sleep() {
        let program = crate::parse("sleep for 1").expect("parse sleep");

        LinkedModule::link(program, full_surface()).expect("top-level sleep should link");
    }

    #[test]
    fn linked_module_rejects_process_lifecycle_outside_process_body() {
        let program = crate::parse("payload = wait signal").expect("parse wait signal");

        let err = LinkedModule::link(program, full_surface())
            .expect_err("top-level process lifecycle should be rejected");

        assert!(
            matches!(
                err,
                LinkError::ProcessLifecycleOutsideProcess {
                    keyword: "wait signal",
                    ..
                }
            ),
            "{err}"
        );
    }

    #[test]
    fn linked_module_rejects_bad_process_args_and_unresolved_operations() {
        let missing_arg = crate::parse(
            r#"
            process scan(tool: Tools, path: str) { finish path }
            start scan(tool: tools)
            "#,
        )
        .expect("parse missing arg");
        assert!(matches!(
            LinkedModule::link(missing_arg, full_surface()),
            Err(LinkError::MissingProcessArgument { arg, .. }) if arg == "path"
        ));

        let bad_operation = crate::parse(
            r#"
            process scan(tool: Tools) {
              finish await tool.missing({})?
            }
            "#,
        )
        .expect("parse bad operation");
        assert!(matches!(
            LinkedModule::link(bad_operation, full_surface()),
            Err(LinkError::UnknownResourceOperation { operation, .. }) if operation == "missing"
        ));
    }

    #[test]
    fn linked_module_rejects_disabled_abilities() {
        let process =
            crate::parse("process worker() { finish null }").expect("parse disabled process");
        assert!(matches!(
            LinkedModule::link(
                process,
                LashlangSurface::new(resources(), LashlangAbilities::default())
            ),
            Err(LinkError::FeatureDisabled {
                feature: "processes",
                ..
            })
        ));

        let start = crate::parse("start worker()").expect("parse disabled start");
        assert!(matches!(
            LinkedModule::link(
                start,
                LashlangSurface::new(resources(), LashlangAbilities::default())
            ),
            Err(LinkError::FeatureDisabled {
                feature: "processes",
                ..
            })
        ));

        let sleep = crate::parse("sleep for \"1s\"").expect("parse disabled sleep");
        assert!(matches!(
            LinkedModule::link(
                sleep,
                LashlangSurface::new(resources(), LashlangAbilities::default())
            ),
            Err(LinkError::FeatureDisabled {
                feature: "sleep",
                ..
            })
        ));

        let signal = crate::parse("process worker() { payload = wait signal }")
            .expect("parse disabled process signal");
        assert!(matches!(
            LinkedModule::link(
                signal,
                LashlangSurface::new(resources(), LashlangAbilities::default().with_processes())
            ),
            Err(LinkError::FeatureDisabled {
                feature: "process signals",
                ..
            })
        ));

        let trigger = crate::parse(
            r#"
            process worker(tick: timer.Tick) { finish true }
            source = timer.Schedule({ expr: "0 8 * * *" })
            await triggers.register({
              source: source,
              target: worker,
              inputs: { tick: trigger.event }
            })?
            "#,
        )
        .expect("parse disabled trigger");
        assert!(matches!(
            LinkedModule::link(
                trigger,
                LashlangSurface::new(resources(), LashlangAbilities::default().with_processes())
            ),
            Err(LinkError::FeatureDisabled {
                feature: "triggers",
                ..
            })
        ));
    }

    #[test]
    fn linked_module_validates_value_constructors_and_trigger_registry_ops() {
        let program = crate::parse(
            r#"
            process scan(tick: timer.Tick) -> bool {
              finish true
            }
            source = timer.Schedule({ expr: "0 8 * * *", tz: "UTC" })
            handle = await triggers.register({
              source: source,
              target: scan,
              inputs: { tick: trigger.event },
              name: "scan"
            })?
            registrations = await triggers.list({ target: scan })?
            cancelled = await triggers.cancel({ handle: handle })?
            submit { handle: handle, registrations: registrations, cancelled: cancelled }
            "#,
        )
        .expect("parse trigger registry program");
        assert!(LinkedModule::link(program, full_surface()).is_ok());
    }

    #[test]
    fn linked_module_accepts_explicit_trigger_input_mappings() {
        let repeated_event = crate::parse(
            r#"
            process scan(a: timer.Tick, b: { fired_at: str }) {
              finish { a: a.fired_at, b: b.fired_at }
            }
            source = timer.Schedule({ expr: "0 8 * * *" })
            await triggers.register({
              source: source,
              target: scan,
              inputs: { a: trigger.event, b: trigger.event }
            })?
            "#,
        )
        .expect("parse repeated event mapping");
        LinkedModule::link(repeated_event, full_surface())
            .expect("event payload should map to multiple assignable params");

        let fixed_authority = crate::parse(
            r#"
            process scan(tick: timer.Tick, tool: Tools) {
              text = await tool.read_file({ path: tick.fired_at })?
              finish text
            }
            source = timer.Schedule({ expr: "0 8 * * *" })
            await triggers.register({
              source: source,
              target: scan,
              inputs: { tick: trigger.event, tool: tools }
            })?
            "#,
        )
        .expect("parse fixed authority mapping");
        LinkedModule::link(fixed_authority, full_surface())
            .expect("fixed resource inputs should satisfy process authority params");
    }

    #[test]
    fn linked_module_captures_concrete_process_body_resources_statically() {
        let program = crate::parse(
            r#"
            process scan(tick: timer.Tick) {
              text = await tools.read_file({ path: tick.fired_at })?
              finish text
            }
            source = timer.Schedule({ expr: "0 8 * * *" })
            await triggers.register({
              source: source,
              target: scan,
              inputs: { tick: trigger.event }
            })?
            "#,
        )
        .expect("parse captured authority process");
        let linked = LinkedModule::link(program, full_surface())
            .expect("process body should capture concrete host resources");
        let process = linked
            .artifact
            .canonical_ir
            .process("scan")
            .expect("scan process");
        fn contains_resource_ref(expr: &Expr, path: &str) -> bool {
            matches!(expr, Expr::ResourceRef(resource) if resource.path_string() == path)
                || expr
                    .children()
                    .any(|child| contains_resource_ref(child, path))
        }
        assert!(
            contains_resource_ref(&process.body, "tools"),
            "linked process body should contain a persisted tools resource ref"
        );

        let shadowed = crate::parse(
            r#"
            tool = tools
            process scan(tick: timer.Tick) {
              text = await tool.read_file({ path: tick.fired_at })?
              finish text
            }
            source = timer.Schedule({ expr: "0 8 * * *" })
            await triggers.register({
              source: source,
              target: scan,
              inputs: { tick: trigger.event }
            })?
            "#,
        )
        .expect("parse foreground variable capture");
        assert!(matches!(
            LinkedModule::link(shadowed, full_surface()),
            Err(LinkError::UnknownName { name, .. }) if name == "tool"
        ));
    }

    #[test]
    fn linked_module_accepts_button_trigger_source_constructor() {
        let mut resources = resources();
        resources
            .add_trigger_source_constructor(
                ["ui", "button", "pressed"],
                TypeExpr::Object(vec![]),
                NamedDataType::object(
                    "ui.button.Pressed",
                    vec![
                        TypeField {
                            name: "button".into(),
                            ty: TypeExpr::Union(vec![
                                TypeExpr::Enum(vec!["Red".into()]),
                                TypeExpr::Enum(vec!["Blue".into()]),
                            ]),
                            optional: false,
                        },
                        TypeField {
                            name: "message".into(),
                            ty: TypeExpr::Str,
                            optional: false,
                        },
                        TypeField {
                            name: "pressed_at".into(),
                            ty: TypeExpr::Str,
                            optional: false,
                        },
                    ],
                )
                .expect("valid button event type"),
            )
            .expect("valid button trigger source");
        let program = crate::parse(
            r#"
            process on_button(event: ui.button.Pressed) {
              wake { kind: "button_pressed", button: event.button, message: event.message }
              finish true
            }

            handle = await triggers.register({
              source: ui.button.pressed({}),
              target: on_button,
              inputs: { event: trigger.event },
              name: "button watcher"
            })?
            submit handle
            "#,
        )
        .expect("parse button trigger source");

        LinkedModule::link(
            program,
            LashlangSurface::new(resources, LashlangAbilities::all()),
        )
        .expect("button trigger source should link");
    }

    #[test]
    fn linked_module_rejects_bad_trigger_registry_bindings() {
        let missing = crate::parse(
            r#"
            process scan(tick: timer.Tick) { finish true }
            source = timer.Schedule({ expr: "0 8 * * *" })
            await triggers.register({ target: scan })?
            "#,
        )
        .expect("parse missing source");
        assert!(matches!(
            LinkedModule::link(missing, full_surface()),
            Err(LinkError::InvalidTriggerRegistration { .. })
        ));

        let missing_inputs = crate::parse(
            r#"
            process scan(tick: timer.Tick) { finish true }
            source = timer.Schedule({ expr: "0 8 * * *" })
            await triggers.register({ source: source, target: scan })?
            "#,
        )
        .expect("parse missing inputs");
        assert!(matches!(
            LinkedModule::link(missing_inputs, full_surface()),
            Err(LinkError::InvalidTriggerRegistration { .. })
        ));

        let wrong_source = crate::parse(
            r#"
            process scan(tick: timer.Tick) { finish true }
            await triggers.register({
              source: { expr: "0 8 * * *" },
              target: scan,
              inputs: { tick: trigger.event }
            })?
            "#,
        )
        .expect("parse wrong source");
        assert!(matches!(
            LinkedModule::link(wrong_source, full_surface()),
            Err(LinkError::UnknownTriggerSourceType { .. })
        ));

        let payload_mismatch = crate::parse(
            r#"
            process scan(tick: str) { finish tick }
            source = timer.Schedule({ expr: "0 8 * * *" })
            await triggers.register({
              source: source,
              target: scan,
              inputs: { tick: trigger.event }
            })?
            "#,
        )
        .expect("parse payload mismatch");
        assert!(matches!(
            LinkedModule::link(payload_mismatch, full_surface()),
            Err(LinkError::TriggerEventMismatch { .. })
        ));

        let unknown_input = crate::parse(
            r#"
            process scan(tick: timer.Tick) { finish true }
            source = timer.Schedule({ expr: "0 8 * * *" })
            await triggers.register({
              source: source,
              target: scan,
              inputs: { tick: trigger.event, extra: "nope" }
            })?
            "#,
        )
        .expect("parse unknown input");
        assert!(matches!(
            LinkedModule::link(unknown_input, full_surface()),
            Err(LinkError::UnknownTriggerInput { input, .. }) if input == "extra"
        ));

        let duplicate_input = crate::parse(
            r#"
            process scan(tick: timer.Tick) { finish true }
            source = timer.Schedule({ expr: "0 8 * * *" })
            await triggers.register({
              source: source,
              target: scan,
              inputs: { tick: trigger.event, tick: trigger.event }
            })?
            "#,
        )
        .expect("parse duplicate input");
        assert!(matches!(
            LinkedModule::link(duplicate_input, full_surface()),
            Err(LinkError::DuplicateTriggerInput { input, .. }) if input == "tick"
        ));

        let no_event_input = crate::parse(
            r#"
            process scan(tick: timer.Tick, label: str) { finish label }
            source = timer.Schedule({ expr: "0 8 * * *" })
            await triggers.register({
              source: source,
              target: scan,
              inputs: { tick: { fired_at: "static" }, label: "static" }
            })?
            "#,
        )
        .expect("parse no event input");
        assert!(matches!(
            LinkedModule::link(no_event_input, full_surface()),
            Err(LinkError::MissingTriggerEventInput { .. })
        ));

        let event_projection = crate::parse(
            r#"
            process scan(fired_at: str) { finish fired_at }
            source = timer.Schedule({ expr: "0 8 * * *" })
            await triggers.register({
              source: source,
              target: scan,
              inputs: { fired_at: trigger.event.fired_at }
            })?
            "#,
        )
        .expect("parse event projection");
        assert!(matches!(
            LinkedModule::link(event_projection, full_surface()),
            Err(LinkError::TriggerEventProjection { .. })
        ));

        let event_outside_inputs = crate::parse(
            r#"
            process scan(tick: timer.Tick) { finish true }
            submit trigger.event
            "#,
        )
        .expect("parse event outside inputs");
        assert!(matches!(
            LinkedModule::link(event_outside_inputs, full_surface()),
            Err(LinkError::TriggerEventOutsideInputs { .. })
        ));

        let multi_input = crate::parse(
            r#"
            process scan(tick: timer.Tick, extra: str) { finish extra }
            source = timer.Schedule({ expr: "0 8 * * *" })
            await triggers.register({
              source: source,
              target: scan,
              inputs: { tick: trigger.event }
            })?
            "#,
        )
        .expect("parse multi-input target");
        assert!(matches!(
            LinkedModule::link(multi_input, full_surface()),
            Err(LinkError::MissingTriggerInput { input, .. }) if input == "extra"
        ));

        let target_is_not_process = crate::parse(
            r#"
            process scan(tick: timer.Tick) { finish true }
            source = timer.Schedule({ expr: "0 8 * * *" })
            await triggers.register({
              source: source,
              target: source,
              inputs: { tick: trigger.event }
            })?
            "#,
        )
        .expect("parse non-process target");
        assert!(matches!(
            LinkedModule::link(target_is_not_process, full_surface()),
            Err(LinkError::InvalidTriggerTarget { .. })
        ));

        let list_without_filters = crate::parse(
            r#"
            process scan(tick: timer.Tick) { finish true }
            await triggers.list({})?
            "#,
        )
        .expect("parse trigger list without filters");
        assert!(LinkedModule::link(list_without_filters, full_surface()).is_ok());

        let list_with_filters = crate::parse(
            r#"
            process scan(tick: timer.Tick) { finish true }
            await triggers.list({
              target: scan,
              name: "daily",
              source_type: "timer.Schedule",
              enabled: true
            })?
            "#,
        )
        .expect("parse trigger list filters");
        assert!(LinkedModule::link(list_with_filters, full_surface()).is_ok());

        let list_target_is_not_process = crate::parse(
            r#"
            process scan(tick: timer.Tick) { finish true }
            source = timer.Schedule({ expr: "0 8 * * *" })
            await triggers.list({ target: source })?
            "#,
        )
        .expect("parse trigger list non-process target");
        assert!(matches!(
            LinkedModule::link(list_target_is_not_process, full_surface()),
            Err(LinkError::InvalidTriggerTarget { .. })
                | Err(LinkError::IncompatibleOperationInput { .. })
        ));

        let constructor_mismatch = crate::parse(
            r#"
            source = timer.Schedule({ expr: 1 })
            submit source
            "#,
        )
        .expect("parse constructor mismatch");
        assert!(matches!(
            LinkedModule::link(constructor_mismatch, full_surface()),
            Err(LinkError::IncompatibleConstructorInput { .. })
        ));

        let operation_mismatch = crate::parse(
            r#"
            await tools.read_file({ path: 1 })?
            "#,
        )
        .expect("parse operation mismatch");
        assert!(matches!(
            LinkedModule::link(operation_mismatch, full_surface()),
            Err(LinkError::IncompatibleOperationInput { .. })
        ));
    }

    #[test]
    fn linked_module_infers_process_output_and_validates_return_annotations() {
        let inferred = crate::parse(
            r#"
            process done(tick: timer.Tick) -> bool {
              finish true
            }
            source = timer.Schedule({ expr: "0 8 * * *" })
            await triggers.register({
              source: source,
              target: done,
              inputs: { tick: trigger.event }
            })?
            "#,
        )
        .expect("parse inferred output");
        assert!(LinkedModule::link(inferred, full_surface()).is_ok());

        let union_mismatch = crate::parse(
            r#"
            process done(tick: timer.Tick) -> bool {
              if true {
                finish true
              }
              finish "done"
            }
            "#,
        )
        .expect("parse union mismatch");
        assert!(matches!(
            LinkedModule::link(union_mismatch, full_surface()),
            Err(LinkError::IncompatibleProcessReturn { .. })
        ));
    }

    #[test]
    fn linked_module_hash_ignores_unused_host_abilities() {
        let program = crate::parse("submit 1").expect("parse");
        let minimal = LinkedModule::link(
            program.clone(),
            LashlangSurface::new(resources(), LashlangAbilities::default()),
        )
        .expect("link minimal");
        let processes = LinkedModule::link(
            program,
            LashlangSurface::new(resources(), LashlangAbilities::default().with_processes()),
        )
        .expect("link process ability");

        assert_eq!(minimal.module_ref, processes.module_ref);
        assert_eq!(minimal.required_surface_ref, processes.required_surface_ref);
    }

    #[test]
    fn label_annotations_require_enabled_language_feature() {
        let program = crate::parse(
            r#"
            @label(title: "Scan files")
            process scan(tool: Tools) {
              @label(title: "Read file")
              text = await tool.read_file({ path: "." })?
              finish text
            }
            "#,
        )
        .expect("parse annotated process");

        let err = LinkedModule::link(program.clone(), full_surface())
            .expect_err("default surface should reject label annotations");
        assert!(matches!(
            err,
            LinkError::FeatureDisabled {
                feature: "label annotations",
                ..
            }
        ));

        let linked =
            LinkedModule::link(program, full_label_surface()).expect("enabled surface should link");
        assert!(
            linked
                .artifact
                .required_surface
                .language_features
                .label_annotations
        );
        let process = linked.program().process("scan").expect("linked process");
        assert_eq!(
            process.label.as_ref().map(|label| label.title.as_str()),
            Some("Scan files")
        );
    }

    #[test]
    fn label_annotation_text_inside_strings_does_not_require_feature() {
        let linked = LinkedModule::link(
            crate::parse(r####"submit r'''@label(title: "Plain text")'''"####)
                .expect("parse string"),
            full_surface(),
        )
        .expect("disabled label annotations should not reject string text");

        assert!(
            !linked
                .artifact
                .required_surface
                .language_features
                .label_annotations
        );
    }

    #[test]
    fn label_metadata_round_trips_and_changes_artifact_identity() {
        let first = LinkedModule::link(
            crate::parse(
                r#"
                @label(title: "Scan files")
                process scan(tool: Tools) {
                  @label(title: "Read file", description: "Load source text")
                  text = await tool.read_file({ path: "." })?
                  @label(title: "Finish")
                  finish text
                }
                "#,
            )
            .expect("parse first"),
            full_label_surface(),
        )
        .expect("link first");
        let changed = LinkedModule::link(
            crate::parse(
                r#"
                @label(title: "Scan files")
                process scan(tool: Tools) {
                  @label(title: "Read source", description: "Load source text")
                  text = await tool.read_file({ path: "." })?
                  @label(title: "Finish")
                  finish text
                }
                "#,
            )
            .expect("parse changed"),
            full_label_surface(),
        )
        .expect("link changed");

        let bytes = first
            .artifact
            .to_store_bytes()
            .expect("encode annotated artifact");
        let decoded = ModuleArtifact::from_store_bytes(&bytes).expect("decode annotated artifact");
        assert_eq!(decoded, first.artifact);
        assert_ne!(first.module_ref, changed.module_ref);
        assert_ne!(
            first.artifact.process_ref("scan"),
            changed.artifact.process_ref("scan")
        );
    }

    #[test]
    fn module_ref_ignores_spans_and_formatting() {
        let compact = LinkedModule::link(
            crate::parse("process scan(root: str) { finish root }").expect("parse compact"),
            full_surface(),
        )
        .expect("link compact");
        let formatted = LinkedModule::link(
            crate::parse(
                r#"
                process scan(root: str) {
                    finish root
                }
                "#,
            )
            .expect("parse formatted"),
            full_surface(),
        )
        .expect("link formatted");

        assert_eq!(compact.module_ref, formatted.module_ref);
    }

    #[test]
    fn process_ref_tracks_abi_and_body_but_not_local_binder_names() {
        let original = LinkedModule::link(
            crate::parse("process scan(root: str) { value = root\nfinish value }")
                .expect("parse original"),
            full_surface(),
        )
        .expect("link original");
        let renamed_local = LinkedModule::link(
            crate::parse("process scan(root: str) { renamed = root\nfinish renamed }")
                .expect("parse renamed local"),
            full_surface(),
        )
        .expect("link renamed local");
        let renamed_param = LinkedModule::link(
            crate::parse("process scan(path: str) { value = path\nfinish value }")
                .expect("parse renamed param"),
            full_surface(),
        )
        .expect("link renamed param");
        let changed_body = LinkedModule::link(
            crate::parse("process scan(root: str) { value = root\nfinish { value: value } }")
                .expect("parse changed body"),
            full_surface(),
        )
        .expect("link changed body");

        assert_eq!(
            original.artifact.process_ref("scan"),
            renamed_local.artifact.process_ref("scan")
        );
        assert_ne!(
            original.artifact.process_ref("scan"),
            renamed_param.artifact.process_ref("scan")
        );
        assert_ne!(
            original.artifact.process_ref("scan"),
            changed_body.artifact.process_ref("scan")
        );
    }

    #[test]
    fn required_surface_ref_tracks_resource_requirements_not_unrelated_tools() {
        let mut with_extra = resources();
        with_extra.add_operation(
            "Tools",
            "unrelated",
            "unrelated",
            TypeExpr::Any,
            TypeExpr::Any,
        );
        let program = crate::parse(
            "process scan(tool: Tools) { finish (await tool.read_file({ path: \".\" }))? }",
        )
        .expect("parse process");

        let base = LinkedModule::link(program.clone(), full_surface()).expect("link base");
        let extra = LinkedModule::link(
            program.clone(),
            LashlangSurface::new(with_extra, LashlangAbilities::all()),
        )
        .expect("link extra");
        let changed_requirement = LinkedModule::link(
            crate::parse(
                "process scan(tool: Tools) { finish (await tool.echo({ value: \".\" }))? }",
            )
            .expect("parse changed resource"),
            full_surface(),
        )
        .expect("link changed requirement");

        assert_eq!(base.module_ref, extra.module_ref);
        assert_eq!(base.required_surface_ref, extra.required_surface_ref);
        assert_ne!(
            base.required_surface_ref,
            changed_requirement.required_surface_ref
        );
    }

    // --- behaviour-pinning tests for the single linking walk -------------
    //
    // These lock in the error *set*, *ordering*, and *spans* the linker
    // produced when validation and lowering were two separate passes, so the
    // fold into one walk stays behaviour-preserving.

    #[test]
    fn declaration_errors_surface_before_main_errors() {
        // The process body references an unknown name AND the main block
        // references a different unknown name. The declaration error must win.
        let program = crate::parse(
            r#"
            process scan() { finish missing_in_body }
            submit missing_in_main
            "#,
        )
        .expect("parse");
        let err = LinkedModule::link(program, full_surface())
            .expect_err("both bodies reference unknowns");
        assert!(
            matches!(&err, LinkError::UnknownName { name, .. } if name == "missing_in_body"),
            "{err:?}"
        );
    }

    #[test]
    fn unknown_name_in_process_body_carries_declaration_span() {
        let program = crate::parse("process scan() { finish missing }").expect("parse");
        let err = LinkedModule::link(program, full_surface()).expect_err("unknown name");
        let LinkError::UnknownName { name, span } = &err else {
            panic!("expected UnknownName, got {err:?}");
        };
        assert_eq!(name, "missing");
        assert!(span.is_some(), "declaration-body error should carry a span");
    }

    #[test]
    fn linker_reproduces_full_error_set() {
        // One representative source per error variant that the expression walk
        // is responsible for raising.
        // Top-level scope allows unknown globals (they become runtime errors),
        // so unknown-name checks must be exercised inside a process body.
        type ErrorCase = (&'static str, fn(&LinkError) -> bool);
        let cases: &[ErrorCase] = &[
            (
                "process scan() { finish missing }",
                |err| matches!(err, LinkError::UnknownName { name, .. } if name == "missing"),
            ),
            (
                "process scan() { missing[0] = 1 }",
                |err| matches!(err, LinkError::UnknownName { name, .. } if name == "missing"),
            ),
            (
                "submit not_a_builtin(1)",
                |err| matches!(err, LinkError::UnknownBuiltin { name, .. } if name == "not_a_builtin"),
            ),
            (
                "x = 1\nsubmit x.read_file({})",
                |err| matches!(err, LinkError::UnresolvedReceiver { operation, .. } if operation == "read_file"),
            ),
            (
                "process scan() { finish 1 }\nstart scan(extra: 1)",
                |err| matches!(err, LinkError::UnexpectedProcessArgument { arg, .. } if arg == "extra"),
            ),
            (
                "process scan(needed: str) { finish needed }\nstart scan()",
                |err| matches!(err, LinkError::MissingProcessArgument { arg, .. } if arg == "needed"),
            ),
            (
                "start ghost()",
                |err| matches!(err, LinkError::UnknownProcess { name, .. } if name == "ghost"),
            ),
        ];

        for (source, predicate) in cases {
            let program =
                crate::parse(source).unwrap_or_else(|err| panic!("parse {source:?}: {err}"));
            let err = LinkedModule::link(program, full_surface())
                .err()
                .unwrap_or_else(|| panic!("{source:?} should fail to link"));
            assert!(predicate(&err), "unexpected error for {source:?}: {err:?}");
        }
    }

    #[test]
    fn unknown_resource_operation_still_rejected_after_receiver_resolves() {
        let program = crate::parse(
            r#"
            process scan(tool: Tools) { finish await tool.does_not_exist({})? }
            "#,
        )
        .expect("parse");
        let err = LinkedModule::link(program, full_surface()).expect_err("operation missing");
        assert!(
            matches!(&err, LinkError::UnknownResourceOperation { operation, .. } if operation == "does_not_exist"),
            "{err:?}"
        );
    }

    #[test]
    fn module_artifact_store_bytes_reject_corruption() {
        use crate::LashlangArtifactStore;

        let linked = LinkedModule::link(
            crate::parse("process scan() { finish 1 }").expect("parse module"),
            full_surface(),
        )
        .expect("link module");
        let store = crate::InMemoryLashlangArtifactStore::new();

        store
            .put_module_artifact(&linked.artifact)
            .expect("put artifact");
        assert_eq!(
            store
                .get_module_artifact(&linked.module_ref)
                .expect("get artifact")
                .expect("artifact exists")
                .module_ref,
            linked.module_ref
        );

        assert!(ModuleArtifact::from_store_bytes(b"not json").is_err());
    }
}
