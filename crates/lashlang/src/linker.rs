use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::artifact::{ModuleArtifact, surface_requirements_for_program_with_catalog};
use crate::ast::{
    AssignPathStep, AstString, Declaration, Expr, ProcessDecl, Program, ResourceRefExpr,
    ScheduleCadence, ScheduleDecl, TriggerArg, TriggerDecl, TriggerSource, TypeExpr, TypeField,
    format_type_expr,
};
use crate::lexer::Span;

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceCatalog {
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub module_instances: BTreeMap<String, ModuleInstanceCatalog>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub resource_types: BTreeMap<String, ResourceTypeCatalog>,
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
            catalog.add_operation("Tools", operation.clone(), operation);
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
    ) {
        self.resource_types
            .entry(resource_type.into())
            .or_default()
            .operations
            .insert(
                operation.into(),
                ResourceOperationBinding {
                    host_operation: host_operation.into(),
                },
            );
    }

    pub fn add_trigger_event(
        &mut self,
        resource_type: impl Into<String>,
        event: impl Into<String>,
        payload_ty: TypeExpr,
    ) {
        self.resource_types
            .entry(resource_type.into())
            .or_default()
            .trigger_events
            .insert(event.into(), TriggerEventBinding { payload_ty });
    }

    pub fn extend(&mut self, other: Self) {
        for (resource_type, incoming) in other.resource_types {
            let entry = self.resource_types.entry(resource_type).or_default();
            entry.operations.extend(incoming.operations);
            entry.trigger_events.extend(incoming.trigger_events);
        }
        self.module_instances.extend(other.module_instances);
    }

    pub fn union(mut self, other: Self) -> Self {
        self.extend(other);
        self
    }

    pub fn has_resource_type(&self, resource_type: &str) -> bool {
        self.resource_types.contains_key(resource_type)
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

    pub fn supports_trigger(&self, resource_type: &str, event: &str) -> bool {
        self.trigger_event(resource_type, event).is_some()
    }

    pub fn trigger_event(&self, resource_type: &str, event: &str) -> Option<&TriggerEventBinding> {
        self.resource_types
            .get(resource_type)?
            .trigger_events
            .get(event)
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
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceTypeCatalog {
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub operations: BTreeMap<String, ResourceOperationBinding>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub trigger_events: BTreeMap<String, TriggerEventBinding>,
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
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TriggerEventBinding {
    pub payload_ty: TypeExpr,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LashlangSurface {
    #[serde(default)]
    pub resources: ResourceCatalog,
    #[serde(default)]
    pub abilities: LashlangAbilities,
}

impl LashlangSurface {
    pub fn new(resources: ResourceCatalog, abilities: LashlangAbilities) -> Self {
        Self {
            resources,
            abilities,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct LashlangAbilities {
    pub processes: bool,
    pub sleep: bool,
    pub process_signals: bool,
    pub triggers: bool,
    pub schedules: LashlangScheduleAbilities,
}

impl LashlangAbilities {
    pub fn union(self, other: Self) -> Self {
        Self {
            processes: self.processes || other.processes,
            sleep: self.sleep || other.sleep,
            process_signals: self.process_signals || other.process_signals,
            triggers: self.triggers || other.triggers,
            schedules: LashlangScheduleAbilities {
                cron: self.schedules.cron || other.schedules.cron,
            },
        }
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

    pub fn with_cron_schedules(mut self) -> Self {
        self.schedules.cron = true;
        self
    }

    pub fn all() -> Self {
        Self::default()
            .with_sleep()
            .with_processes()
            .with_process_signals()
            .with_triggers()
            .with_cron_schedules()
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct LashlangScheduleAbilities {
    #[serde(default)]
    pub cron: bool,
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
    pub fn link(program: Program, surface: LashlangSurface) -> Result<Self, LinkError> {
        let mut linker = Linker::new(&program, &surface);
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
    #[error("resource type `{resource_type}` does not declare trigger event `{event}`")]
    UnknownTriggerEvent {
        resource_type: String,
        event: String,
        span: Option<Span>,
    },
    #[error("trigger `{trigger}` argument `{arg}` references unknown binding `{binding}`")]
    UnknownTriggerBinding {
        trigger: String,
        arg: String,
        binding: String,
        span: Option<Span>,
    },
    #[error(
        "trigger `{trigger}` binding for process `{process}` argument `{arg}` has incompatible type: expected {expected}, got {actual}"
    )]
    IncompatibleTriggerArgument {
        trigger: Box<str>,
        process: Box<str>,
        arg: Box<str>,
        expected: Box<str>,
        actual: Box<str>,
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
            | Self::UnknownTriggerEvent { span, .. }
            | Self::UnknownTriggerBinding { span, .. }
            | Self::IncompatibleTriggerArgument { span, .. }
            | Self::UnresolvedReceiver { span, .. }
            | Self::UnknownResourceOperation { span, .. }
            | Self::AmbiguousModuleOperation { span, .. }
            | Self::BareToolCall { span, .. }
            | Self::IncompatibleProcessArgument { span, .. }
            | Self::FeatureDisabled { span, .. }
            | Self::ProcessLifecycleOutsideProcess { span, .. } => *span,
            Self::ModuleHash { .. } => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Binding {
    Value,
    Resource { resource_type: String },
}

struct Linker<'module> {
    program: &'module Program,
    surface: &'module LashlangSurface,
    process_names: BTreeSet<String>,
    type_names: BTreeSet<String>,
    type_defs: BTreeMap<String, TypeExpr>,
}

impl<'module> Linker<'module> {
    fn new(program: &'module Program, surface: &'module LashlangSurface) -> Self {
        Self {
            program,
            surface,
            process_names: BTreeSet::new(),
            type_names: BTreeSet::new(),
            type_defs: BTreeMap::new(),
        }
    }

    fn link_program(&mut self) -> Result<Program, LinkError> {
        self.validate()?;
        let mut scope = Scope::new(true, false, None);
        let main = self.lower_expr(&self.program.main, &mut scope)?.0;
        let declarations = self
            .program
            .declarations
            .iter()
            .map(|declaration| self.lower_declaration(declaration))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Program {
            declarations,
            main,
            declaration_spans: self.program.declaration_spans.clone(),
            expression_spans: self.program.expression_spans.clone(),
        })
    }

    fn validate(&mut self) -> Result<(), LinkError> {
        self.collect_declarations()?;
        for (index, declaration) in self.program.declarations.iter().enumerate() {
            let span = self.program.declaration_spans.get(index).copied();
            match declaration {
                Declaration::Process(process) => self.validate_process(process, span)?,
                Declaration::Trigger(trigger) => {
                    self.ensure_feature(self.surface.abilities.triggers, "triggers", span)?;
                    self.ensure_feature(self.surface.abilities.processes, "processes", span)?;
                    let payload_ty = match &trigger.source {
                        TriggerSource::Binding { resource, event } => {
                            let resource = self.validate_resource_ref(resource, span)?;
                            self.trigger_event_payload(
                                resource.resource_type.as_str(),
                                event.as_str(),
                                span,
                            )?
                        }
                        TriggerSource::Each {
                            resource_type,
                            event,
                            ..
                        } => self.trigger_event_payload(
                            resource_type.as_str(),
                            event.as_str(),
                            span,
                        )?,
                    };
                    self.validate_trigger(trigger, &payload_ty, span)?;
                }
                Declaration::Schedule(schedule) => {
                    if matches!(schedule.cadence, ScheduleCadence::Cron { .. })
                        && !self.surface.abilities.schedules.cron
                    {
                        return Err(LinkError::FeatureDisabled {
                            feature: "cron schedules",
                            span,
                        });
                    }
                    let mut scope = Scope::new(false, false, span);
                    scope.bind(schedule.tick_binding.as_str(), Binding::Value);
                    match &schedule.cadence {
                        ScheduleCadence::Cron {
                            expression,
                            options,
                        } => {
                            self.validate_expr(expression, &mut scope)?;
                            for (_, option) in options {
                                self.validate_expr(option, &mut scope)?;
                            }
                        }
                    }
                    self.validate_expr(&schedule.body, &mut scope)?;
                }
                Declaration::Type(_) => {}
            }
        }

        let mut top_level = Scope::new(true, false, None);
        self.validate_expr(&self.program.main, &mut top_level)?;
        Ok(())
    }

    fn collect_declarations(&mut self) -> Result<(), LinkError> {
        let mut names = BTreeSet::new();
        for (index, declaration) in self.program.declarations.iter().enumerate() {
            let span = self.program.declaration_spans.get(index).copied();
            let name = match declaration {
                Declaration::Type(decl) => {
                    self.type_names.insert(decl.name.to_string());
                    self.type_defs
                        .insert(decl.name.to_string(), decl.ty.clone());
                    decl.name.as_str()
                }
                Declaration::Process(decl) => {
                    self.ensure_feature(self.surface.abilities.processes, "processes", span)?;
                    self.process_names.insert(decl.name.to_string());
                    decl.name.as_str()
                }
                Declaration::Trigger(decl) => {
                    self.ensure_feature(self.surface.abilities.triggers, "triggers", span)?;
                    self.ensure_feature(self.surface.abilities.processes, "processes", span)?;
                    decl.name.as_str()
                }
                Declaration::Schedule(decl) => decl.name.as_str(),
            };
            if !names.insert(name.to_string()) {
                return Err(LinkError::DuplicateDeclaration {
                    name: name.to_string(),
                    span,
                });
            }
        }
        Ok(())
    }

    fn validate_process(&self, process: &ProcessDecl, span: Option<Span>) -> Result<(), LinkError> {
        self.ensure_feature(self.surface.abilities.processes, "processes", span)?;
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
        scope.bind("input", Binding::Value);
        scope.bind("inputs", Binding::Value);
        self.validate_expr(&process.body, &mut scope)?;
        Ok(())
    }

    fn validate_trigger(
        &self,
        trigger: &TriggerDecl,
        payload_ty: &TypeExpr,
        span: Option<Span>,
    ) -> Result<(), LinkError> {
        let Some(process) = self.program.process(trigger.process_name.as_str()) else {
            return Err(LinkError::UnknownProcess {
                name: trigger.process_name.to_string(),
                span,
            });
        };

        let mut seen = BTreeSet::new();
        for (arg, binding) in &trigger.args {
            if !seen.insert(arg.to_string()) {
                return Err(LinkError::DuplicateProcessArgument {
                    arg: arg.to_string(),
                    span,
                });
            }
            let Some(param) = process.params.iter().find(|param| param.name == *arg) else {
                return Err(LinkError::UnexpectedProcessArgument {
                    process: process.name.to_string(),
                    arg: arg.to_string(),
                    span,
                });
            };
            self.validate_trigger_arg(trigger, process, param, binding, payload_ty, span)?;
        }
        for param in &process.params {
            if !seen.contains(param.name.as_str()) {
                return Err(LinkError::MissingProcessArgument {
                    process: process.name.to_string(),
                    arg: param.name.to_string(),
                    span,
                });
            }
        }
        Ok(())
    }

    fn validate_trigger_arg(
        &self,
        trigger: &TriggerDecl,
        process: &ProcessDecl,
        param: &crate::ast::ProcessParam,
        binding: &TriggerArg,
        payload_ty: &TypeExpr,
        span: Option<Span>,
    ) -> Result<(), LinkError> {
        match binding {
            TriggerArg::EventBinding(name) => {
                if name != trigger.event_binding {
                    return Err(LinkError::UnknownTriggerBinding {
                        trigger: trigger.name.to_string(),
                        arg: param.name.to_string(),
                        binding: name.to_string(),
                        span,
                    });
                }
                if self.is_type_assignable(payload_ty, &param.ty) {
                    return Ok(());
                }
                Err(self.incompatible_trigger_arg(
                    trigger,
                    process,
                    param.name.as_str(),
                    &param.ty,
                    payload_ty,
                    span,
                ))
            }
            TriggerArg::ResourceBinding(name) => {
                let TriggerSource::Each {
                    resource_type,
                    resource_binding,
                    ..
                } = &trigger.source
                else {
                    return Err(LinkError::UnknownTriggerBinding {
                        trigger: trigger.name.to_string(),
                        arg: param.name.to_string(),
                        binding: name.to_string(),
                        span,
                    });
                };
                if name != resource_binding {
                    return Err(LinkError::UnknownTriggerBinding {
                        trigger: trigger.name.to_string(),
                        arg: param.name.to_string(),
                        binding: name.to_string(),
                        span,
                    });
                }
                self.validate_resource_param(trigger, process, param, resource_type, span)
            }
            TriggerArg::ResourceRef(resource) => {
                let resource = self.validate_resource_ref(resource, span)?;
                self.validate_resource_param(
                    trigger,
                    process,
                    param,
                    resource.resource_type.as_str(),
                    span,
                )
            }
        }
    }

    fn validate_resource_param(
        &self,
        trigger: &TriggerDecl,
        process: &ProcessDecl,
        param: &crate::ast::ProcessParam,
        resource_type: &str,
        span: Option<Span>,
    ) -> Result<(), LinkError> {
        if self
            .resource_type_for_type(&param.ty)
            .is_some_and(|expected| expected == resource_type)
        {
            return Ok(());
        }
        Err(LinkError::IncompatibleTriggerArgument {
            trigger: trigger.name.to_string().into_boxed_str(),
            process: process.name.to_string().into_boxed_str(),
            arg: param.name.to_string().into_boxed_str(),
            expected: format_type_expr(&self.resolve_type_aliases(&param.ty)).into_boxed_str(),
            actual: resource_type.to_string().into_boxed_str(),
            span,
        })
    }

    fn incompatible_trigger_arg(
        &self,
        trigger: &TriggerDecl,
        process: &ProcessDecl,
        arg: &str,
        expected: &TypeExpr,
        actual: &TypeExpr,
        span: Option<Span>,
    ) -> LinkError {
        LinkError::IncompatibleTriggerArgument {
            trigger: trigger.name.to_string().into_boxed_str(),
            process: process.name.to_string().into_boxed_str(),
            arg: arg.to_string().into_boxed_str(),
            expected: format_type_expr(&self.resolve_type_aliases(expected)).into_boxed_str(),
            actual: format_type_expr(&self.resolve_type_aliases(actual)).into_boxed_str(),
            span,
        }
    }

    fn trigger_event_payload(
        &self,
        resource_type: &str,
        event: &str,
        span: Option<Span>,
    ) -> Result<TypeExpr, LinkError> {
        self.surface
            .resources
            .trigger_event(resource_type, event)
            .map(|binding| binding.payload_ty.clone())
            .ok_or_else(|| LinkError::UnknownTriggerEvent {
                resource_type: resource_type.to_string(),
                event: event.to_string(),
                span,
            })
    }

    fn binding_for_type(&self, ty: &TypeExpr) -> Binding {
        match self.resource_type_for_type(ty) {
            Some(resource_type) => Binding::Resource { resource_type },
            _ => Binding::Value,
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
                let resolved = self
                    .type_defs
                    .get(name.as_str())
                    .map(|ty| self.resolve_type_aliases_inner(ty, seen))
                    .unwrap_or_else(|| ty.clone());
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
        is_resolved_type_assignable(&source, &target)
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

    fn validate_expr(&self, expr: &Expr, scope: &mut Scope) -> Result<Option<Binding>, LinkError> {
        match expr {
            Expr::Block(expressions) => {
                let mut last = None;
                for expression in expressions {
                    last = self.validate_expr(expression, scope)?;
                }
                Ok(last)
            }
            Expr::Null
            | Expr::Bool(_)
            | Expr::Number(_)
            | Expr::String(_)
            | Expr::Break
            | Expr::Continue
            | Expr::TypeLiteral(_) => Ok(Some(Binding::Value)),
            Expr::Variable(name) => {
                if let Some(binding) = scope.get(name) {
                    return Ok(Some(binding));
                }
                if let Some(resource) = self.resolve_module_expr(expr, scope) {
                    return Ok(Some(Binding::Resource {
                        resource_type: resource.resource_type.to_string(),
                    }));
                }
                if scope.allow_unknown_globals {
                    return Ok(Some(Binding::Value));
                }
                Err(LinkError::UnknownName {
                    name: name.to_string(),
                    span: scope.span,
                })
            }
            Expr::List(items) => {
                for item in items {
                    self.validate_expr(item, scope)?;
                }
                Ok(Some(Binding::Value))
            }
            Expr::Record(entries) => {
                for (_, value) in entries {
                    self.validate_expr(value, scope)?;
                }
                Ok(Some(Binding::Value))
            }
            Expr::Assign { target, expr } => {
                for step in &target.steps {
                    if let AssignPathStep::Index(index) = step {
                        self.validate_expr(index, scope)?;
                    }
                }
                let binding = self.validate_expr(expr, scope)?.unwrap_or(Binding::Value);
                if target.steps.is_empty() {
                    scope.bind(target.root.as_str(), binding);
                } else if scope.get(&target.root).is_none() && !scope.allow_unknown_globals {
                    return Err(LinkError::UnknownName {
                        name: target.root.to_string(),
                        span: scope.span,
                    });
                }
                Ok(Some(Binding::Value))
            }
            Expr::If {
                condition,
                then_block,
                else_block,
            } => {
                self.validate_expr(condition, scope)?;
                let mut then_scope = scope.clone();
                self.validate_expr(then_block, &mut then_scope)?;
                let mut else_scope = scope.clone();
                self.validate_expr(else_block, &mut else_scope)?;
                scope.merge_from(then_scope);
                scope.merge_from(else_scope);
                Ok(Some(Binding::Value))
            }
            Expr::For {
                binding,
                iterable,
                body,
            } => {
                self.validate_expr(iterable, scope)?;
                let previous = scope.bind(binding.as_str(), Binding::Value);
                self.validate_expr(body, scope)?;
                scope.restore(binding.as_str(), previous);
                Ok(Some(Binding::Value))
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
                    let binding = self.validate_expr(value, scope)?;
                    self.validate_process_arg_binding(
                        process.name.as_str(),
                        arg.as_str(),
                        &param.ty,
                        binding.as_ref(),
                        scope.span,
                    )?;
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
                Ok(Some(Binding::Value))
            }
            Expr::SleepFor(expr) | Expr::SleepUntil(expr) => {
                self.ensure_feature(self.surface.abilities.sleep, "sleep", scope.span)?;
                self.validate_expr(expr, scope)?;
                Ok(Some(Binding::Value))
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
                Ok(Some(Binding::Value))
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
                self.validate_expr(run, scope)?;
                self.validate_expr(payload, scope)?;
                Ok(Some(Binding::Value))
            }
            Expr::ResourceRef(resource) => {
                let resource = self.validate_resource_ref(resource, scope.span)?;
                Ok(Some(Binding::Resource {
                    resource_type: resource.resource_type.to_string(),
                }))
            }
            Expr::ReceiverCall {
                receiver,
                operation,
                args,
            } => {
                let receiver_binding =
                    if let Some(resource) = self.resolve_module_expr(receiver, scope) {
                        Some(Binding::Resource {
                            resource_type: resource.resource_type.to_string(),
                        })
                    } else {
                        self.validate_expr(receiver, scope)?
                    };
                let Some(Binding::Resource { resource_type }) = receiver_binding else {
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
                if self
                    .surface
                    .resources
                    .resolve_operation(&resource_type, operation)
                    .is_none()
                {
                    return Err(LinkError::UnknownResourceOperation {
                        resource_type,
                        operation: operation.to_string(),
                        span: scope.span,
                    });
                }
                for arg in args {
                    self.validate_expr(arg, scope)?;
                }
                Ok(Some(Binding::Value))
            }
            Expr::Await(expr)
            | Expr::ResultUnwrap(expr)
            | Expr::Cancel(expr)
            | Expr::Print(expr)
            | Expr::Yield(expr)
            | Expr::Wake(expr)
            | Expr::Fail(expr)
            | Expr::Unary { expr, .. } => {
                self.validate_expr(expr, scope)?;
                Ok(Some(Binding::Value))
            }
            Expr::Submit(expr) | Expr::Finish(expr) => {
                if let Some(expr) = expr {
                    self.validate_expr(expr, scope)?;
                }
                Ok(Some(Binding::Value))
            }
            Expr::BuiltinCall { name, args } => {
                if !is_builtin(name.as_str()) {
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
                for arg in args {
                    self.validate_expr(arg, scope)?;
                }
                Ok(Some(Binding::Value))
            }
            Expr::Field { target, .. } => {
                if let Some(resource) = self.resolve_module_expr(expr, scope) {
                    return Ok(Some(Binding::Resource {
                        resource_type: resource.resource_type.to_string(),
                    }));
                }
                self.validate_expr(target, scope)?;
                Ok(Some(Binding::Value))
            }
            Expr::Index { target, index } => {
                self.validate_expr(target, scope)?;
                self.validate_expr(index, scope)?;
                Ok(Some(Binding::Value))
            }
            Expr::Binary { left, right, .. } => {
                self.validate_expr(left, scope)?;
                self.validate_expr(right, scope)?;
                Ok(Some(Binding::Value))
            }
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

    fn lower_declaration(&self, declaration: &Declaration) -> Result<Declaration, LinkError> {
        Ok(match declaration {
            Declaration::Type(type_decl) => Declaration::Type(type_decl.clone()),
            Declaration::Process(process) => {
                let mut scope = Scope::new(false, true, None);
                for param in &process.params {
                    scope.bind(param.name.as_str(), self.binding_for_type(&param.ty));
                }
                scope.bind("input", Binding::Value);
                scope.bind("inputs", Binding::Value);
                let body = self.lower_expr(&process.body, &mut scope)?.0;
                Declaration::Process(ProcessDecl {
                    name: process.name.clone(),
                    params: process.params.clone(),
                    return_ty: process.return_ty.clone(),
                    body,
                })
            }
            Declaration::Trigger(trigger) => {
                let source = match &trigger.source {
                    TriggerSource::Binding { resource, event } => TriggerSource::Binding {
                        resource: self.validate_resource_ref(resource, None)?,
                        event: event.clone(),
                    },
                    TriggerSource::Each {
                        resource_type,
                        event,
                        resource_binding,
                    } => TriggerSource::Each {
                        resource_type: resource_type.clone(),
                        event: event.clone(),
                        resource_binding: resource_binding.clone(),
                    },
                };
                let args = trigger
                    .args
                    .iter()
                    .map(|(name, arg)| {
                        let arg = match arg {
                            TriggerArg::ResourceRef(resource) => {
                                TriggerArg::ResourceRef(self.validate_resource_ref(resource, None)?)
                            }
                            TriggerArg::EventBinding(name) => {
                                TriggerArg::EventBinding(name.clone())
                            }
                            TriggerArg::ResourceBinding(name) => {
                                TriggerArg::ResourceBinding(name.clone())
                            }
                        };
                        Ok((name.clone(), arg))
                    })
                    .collect::<Result<Vec<_>, LinkError>>()?;
                Declaration::Trigger(TriggerDecl {
                    name: trigger.name.clone(),
                    source,
                    event_binding: trigger.event_binding.clone(),
                    process_name: trigger.process_name.clone(),
                    args,
                })
            }
            Declaration::Schedule(schedule) => {
                let mut scope = Scope::new(false, false, None);
                scope.bind(schedule.tick_binding.as_str(), Binding::Value);
                let cadence = match &schedule.cadence {
                    ScheduleCadence::Cron {
                        expression,
                        options,
                    } => ScheduleCadence::Cron {
                        expression: self.lower_expr(expression, &mut scope)?.0,
                        options: options
                            .iter()
                            .map(|(name, value)| {
                                Ok((name.clone(), self.lower_expr(value, &mut scope)?.0))
                            })
                            .collect::<Result<Vec<_>, LinkError>>()?,
                    },
                };
                let body = self.lower_expr(&schedule.body, &mut scope)?.0;
                Declaration::Schedule(ScheduleDecl {
                    name: schedule.name.clone(),
                    cadence,
                    tick_binding: schedule.tick_binding.clone(),
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
            Expr::Variable(name) => (Expr::Variable(name.clone()), scope.get(name)),
            Expr::Null
            | Expr::Bool(_)
            | Expr::Number(_)
            | Expr::String(_)
            | Expr::Break
            | Expr::Continue
            | Expr::TypeLiteral(_) => (expr.clone(), Some(Binding::Value)),
            Expr::List(items) => (
                Expr::List(
                    items
                        .iter()
                        .map(|item| self.lower_expr(item, scope).map(|(expr, _)| expr))
                        .collect::<Result<Vec<_>, _>>()?,
                ),
                Some(Binding::Value),
            ),
            Expr::Record(entries) => (
                Expr::Record(
                    entries
                        .iter()
                        .map(|(name, value)| Ok((name.clone(), self.lower_expr(value, scope)?.0)))
                        .collect::<Result<Vec<_>, LinkError>>()?,
                ),
                Some(Binding::Value),
            ),
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
                        binding.clone().unwrap_or(Binding::Value),
                    );
                }
                (
                    Expr::Assign {
                        target: target.clone(),
                        expr: Box::new(lowered),
                    },
                    Some(Binding::Value),
                )
            }
            Expr::If {
                condition,
                then_block,
                else_block,
            } => {
                let condition = self.lower_expr(condition, scope)?.0;
                let mut then_scope = scope.clone();
                let then_block = self.lower_expr(then_block, &mut then_scope)?.0;
                let mut else_scope = scope.clone();
                let else_block = self.lower_expr(else_block, &mut else_scope)?.0;
                scope.merge_from(then_scope);
                scope.merge_from(else_scope);
                (
                    Expr::If {
                        condition: Box::new(condition),
                        then_block: Box::new(then_block),
                        else_block: Box::new(else_block),
                    },
                    Some(Binding::Value),
                )
            }
            Expr::For {
                binding,
                iterable,
                body,
            } => {
                let iterable = self.lower_expr(iterable, scope)?.0;
                let previous = scope.bind(binding.as_str(), Binding::Value);
                let body = self.lower_expr(body, scope)?.0;
                scope.restore(binding.as_str(), previous);
                (
                    Expr::For {
                        binding: binding.clone(),
                        iterable: Box::new(iterable),
                        body: Box::new(body),
                    },
                    Some(Binding::Value),
                )
            }
            Expr::StartProcess(start) => (
                Expr::StartProcess(crate::ast::ProcessStartExpr {
                    process: start.process.clone(),
                    args: start
                        .args
                        .iter()
                        .map(|(name, value)| Ok((name.clone(), self.lower_expr(value, scope)?.0)))
                        .collect::<Result<Vec<_>, LinkError>>()?,
                }),
                Some(Binding::Value),
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
                let (receiver, resource_type) =
                    if let Some(resource) = self.resolve_module_expr(receiver, scope) {
                        (
                            Expr::ResourceRef(resource.clone()),
                            resource.resource_type.to_string(),
                        )
                    } else {
                        let (receiver, binding) = self.lower_expr(receiver, scope)?;
                        let resource_type = match binding {
                            Some(Binding::Resource { resource_type }) => resource_type,
                            _ => String::new(),
                        };
                        (receiver, resource_type)
                    };
                let host_operation = self
                    .surface
                    .resources
                    .resolve_operation(&resource_type, operation)
                    .map(|binding| binding.host_operation.clone())
                    .unwrap_or_else(|| operation.to_string());
                (
                    Expr::ReceiverCall {
                        receiver: Box::new(receiver),
                        operation: host_operation.into(),
                        args: args
                            .iter()
                            .map(|arg| self.lower_expr(arg, scope).map(|(expr, _)| expr))
                            .collect::<Result<Vec<_>, _>>()?,
                    },
                    Some(Binding::Value),
                )
            }
            Expr::Await(inner) => (
                Expr::Await(Box::new(self.lower_expr(inner, scope)?.0)),
                Some(Binding::Value),
            ),
            Expr::SleepFor(inner) => (
                Expr::SleepFor(Box::new(self.lower_expr(inner, scope)?.0)),
                Some(Binding::Value),
            ),
            Expr::SleepUntil(inner) => (
                Expr::SleepUntil(Box::new(self.lower_expr(inner, scope)?.0)),
                Some(Binding::Value),
            ),
            Expr::WaitSignal => (Expr::WaitSignal, Some(Binding::Value)),
            Expr::SignalRun { run, payload } => (
                Expr::SignalRun {
                    run: Box::new(self.lower_expr(run, scope)?.0),
                    payload: Box::new(self.lower_expr(payload, scope)?.0),
                },
                Some(Binding::Value),
            ),
            Expr::ResultUnwrap(inner) => (
                Expr::ResultUnwrap(Box::new(self.lower_expr(inner, scope)?.0)),
                Some(Binding::Value),
            ),
            Expr::Cancel(inner) => (
                Expr::Cancel(Box::new(self.lower_expr(inner, scope)?.0)),
                Some(Binding::Value),
            ),
            Expr::Print(inner) => (
                Expr::Print(Box::new(self.lower_expr(inner, scope)?.0)),
                Some(Binding::Value),
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
                Some(Binding::Value),
            ),
            Expr::Yield(inner) => (
                Expr::Yield(Box::new(self.lower_expr(inner, scope)?.0)),
                Some(Binding::Value),
            ),
            Expr::Wake(inner) => (
                Expr::Wake(Box::new(self.lower_expr(inner, scope)?.0)),
                Some(Binding::Value),
            ),
            Expr::Finish(inner) => (
                Expr::Finish(
                    inner
                        .as_deref()
                        .map(|inner| {
                            self.lower_expr(inner, scope)
                                .map(|(expr, _)| Box::new(expr))
                        })
                        .transpose()?,
                ),
                Some(Binding::Value),
            ),
            Expr::Fail(inner) => (
                Expr::Fail(Box::new(self.lower_expr(inner, scope)?.0)),
                Some(Binding::Value),
            ),
            Expr::BuiltinCall { name, args } => (
                Expr::BuiltinCall {
                    name: name.clone(),
                    args: args
                        .iter()
                        .map(|arg| self.lower_expr(arg, scope).map(|(expr, _)| expr))
                        .collect::<Result<Vec<_>, _>>()?,
                },
                Some(Binding::Value),
            ),
            Expr::Field { target, field } => (
                Expr::Field {
                    target: Box::new(self.lower_expr(target, scope)?.0),
                    field: field.clone(),
                },
                Some(Binding::Value),
            ),
            Expr::Index { target, index } => (
                Expr::Index {
                    target: Box::new(self.lower_expr(target, scope)?.0),
                    index: Box::new(self.lower_expr(index, scope)?.0),
                },
                Some(Binding::Value),
            ),
            Expr::Unary { op, expr } => (
                Expr::Unary {
                    op: *op,
                    expr: Box::new(self.lower_expr(expr, scope)?.0),
                },
                Some(Binding::Value),
            ),
            Expr::Binary { left, op, right } => (
                Expr::Binary {
                    left: Box::new(self.lower_expr(left, scope)?.0),
                    op: *op,
                    right: Box::new(self.lower_expr(right, scope)?.0),
                },
                Some(Binding::Value),
            ),
        })
    }

    fn resolve_module_expr(&self, expr: &Expr, scope: &Scope) -> Option<ResourceRefExpr> {
        if scope.process_body {
            return None;
        }
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

fn module_path_for_expr(expr: &Expr) -> Option<Vec<AstString>> {
    match expr {
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

fn is_builtin(name: &str) -> bool {
    matches!(
        name,
        "len"
            | "empty"
            | "keys"
            | "values"
            | "trim"
            | "to_string"
            | "to_int"
            | "to_float"
            | "json_parse"
            | "contains"
            | "grep_text"
            | "starts_with"
            | "ends_with"
            | "split"
            | "join"
            | "validate"
            | "ceil_div"
            | "floor_div"
            | "push"
            | "slice"
            | "find"
            | "format"
            | "range"
    )
}

fn is_resolved_type_assignable(source: &TypeExpr, target: &TypeExpr) -> bool {
    if matches!(target, TypeExpr::Any) {
        return true;
    }
    if source == target {
        return true;
    }

    match (source, target) {
        (TypeExpr::Any, _) => false,
        (TypeExpr::Union(sources), _) => sources
            .iter()
            .all(|source| is_resolved_type_assignable(source, target)),
        (_, TypeExpr::Union(targets)) => targets
            .iter()
            .any(|target| is_resolved_type_assignable(source, target)),
        (TypeExpr::Int, TypeExpr::Float) => true,
        (TypeExpr::Enum(_), TypeExpr::Str) => true,
        (TypeExpr::Enum(sources), TypeExpr::Enum(targets)) => {
            sources.iter().all(|source| targets.contains(source))
        }
        (TypeExpr::List(source), TypeExpr::List(target)) => {
            is_resolved_type_assignable(source, target)
        }
        (TypeExpr::Object(_), TypeExpr::Dict) => true,
        (TypeExpr::Object(source), TypeExpr::Object(target)) => {
            object_type_assignable(source, target)
        }
        (TypeExpr::Ref(source), TypeExpr::Ref(target)) => source == target,
        _ => false,
    }
}

fn object_type_assignable(source: &[TypeField], target: &[TypeField]) -> bool {
    target.iter().all(|target_field| {
        let Some(source_field) = source
            .iter()
            .find(|source_field| source_field.name == target_field.name)
        else {
            return target_field.optional;
        };
        if !target_field.optional && source_field.optional {
            return false;
        }
        is_resolved_type_assignable(&source_field.ty, &target_field.ty)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resources() -> ResourceCatalog {
        let mut catalog = ResourceCatalog::new();
        catalog.add_module_instance(["tools"], "Tools");
        catalog.add_operation("Tools", "read_file", "read_file");
        catalog.add_operation("Tools", "echo", "echo");
        catalog.add_trigger_event("Tools", "changed", change_event_type());
        catalog
    }

    fn change_event_type() -> TypeExpr {
        TypeExpr::Object(vec![TypeField {
            name: "path".into(),
            ty: TypeExpr::Str,
            optional: false,
        }])
    }

    fn full_surface() -> LashlangSurface {
        LashlangSurface::new(resources(), LashlangAbilities::all())
    }

    #[test]
    fn linked_module_accepts_named_processes_resource_params_and_activations() {
        let program = crate::parse(
            r#"
            type ChangeEvent = { path: str }
            process scan(tool: Tools, event: ChangeEvent) {
              text = await tool.read_file({ path: event.path })?
              finish text
            }
            process watcher(run: any) {
              sleep for "0ms"
              signal = wait signal
              signal run run with signal
              finish signal
            }
            trigger changed on tools.changed as event
              -> scan(tool: tools, event: event)
            schedule hourly every cron("0 * * * *") as tick {
              start scan(tool: tools, event: { path: tick.path })
            }
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
            process worker(event: any) { finish event }
            trigger changed on tools.changed as event
              -> worker(event: event)
            "#,
        )
        .expect("parse disabled trigger");
        assert!(matches!(
            LinkedModule::link(
                trigger,
                LashlangSurface::new(
                    resources(),
                    LashlangAbilities::default()
                        .with_processes()
                        .with_cron_schedules()
                )
            ),
            Err(LinkError::FeatureDisabled {
                feature: "triggers",
                ..
            })
        ));

        let trigger_without_process_ability = crate::parse(
            r#"
            trigger changed on tools.changed as event
              -> worker(event: event)
            "#,
        )
        .expect("parse trigger without process ability");
        assert!(matches!(
            LinkedModule::link(
                trigger_without_process_ability,
                LashlangSurface::new(resources(), LashlangAbilities::default().with_triggers())
            ),
            Err(LinkError::FeatureDisabled {
                feature: "processes",
                ..
            })
        ));

        let schedule = crate::parse(
            r#"
            schedule hourly every cron("0 * * * *") as tick {
              print tick
            }
            "#,
        )
        .expect("parse disabled schedule");
        assert!(
            LinkedModule::link(
                schedule,
                LashlangSurface::new(resources(), LashlangAbilities::all())
            )
            .is_ok()
        );
        assert!(matches!(
            LinkedModule::link(
                crate::parse(
                    r#"
                    schedule hourly every cron("0 * * * *") as tick {
                      print tick
                    }
                    "#,
                )
                .expect("parse disabled schedule again"),
                LashlangSurface::new(
                    resources(),
                    LashlangAbilities::default()
                        .with_processes()
                        .with_sleep()
                        .with_process_signals()
                )
            ),
            Err(LinkError::FeatureDisabled {
                feature: "cron schedules",
                ..
            })
        ));
    }

    #[test]
    fn linked_module_validates_declarative_trigger_bindings() {
        let fixed = crate::parse(
            r#"
            type ChangeEvent = { path: str }
            process scan(first: Tools, second: Tools, event: ChangeEvent) {
              finish event.path
            }
            trigger changed on tools.changed as event
              -> scan(first: tools, second: tools, event: event)
            "#,
        )
        .expect("parse fixed trigger");
        assert!(LinkedModule::link(fixed, full_surface()).is_ok());

        let each = crate::parse(
            r#"
            type ChangeEvent = { path: str }
            process scan(tool: Tools, event: ChangeEvent) {
              finish event.path
            }
            trigger changed on each Tools.changed as tool, event
              -> scan(tool: tool, event: event)
            "#,
        )
        .expect("parse each trigger");
        assert!(LinkedModule::link(each, full_surface()).is_ok());
    }

    #[test]
    fn linked_module_rejects_bad_declarative_trigger_bindings() {
        let missing = crate::parse(
            r#"
            process scan(tool: Tools, event: any) { finish event }
            trigger changed on tools.changed as event
              -> scan(tool: tools)
            "#,
        )
        .expect("parse missing trigger arg");
        assert!(matches!(
            LinkedModule::link(missing, full_surface()),
            Err(LinkError::MissingProcessArgument { arg, .. }) if arg == "event"
        ));

        let extra = crate::parse(
            r#"
            process scan(event: any) { finish event }
            trigger changed on tools.changed as event
              -> scan(event: event, tool: tools)
            "#,
        )
        .expect("parse extra trigger arg");
        assert!(matches!(
            LinkedModule::link(extra, full_surface()),
            Err(LinkError::UnexpectedProcessArgument { arg, .. }) if arg == "tool"
        ));

        let duplicate = crate::parse(
            r#"
            process scan(event: any) { finish event }
            trigger changed on tools.changed as event
              -> scan(event: event, event: event)
            "#,
        )
        .expect("parse duplicate trigger arg");
        assert!(matches!(
            LinkedModule::link(duplicate, full_surface()),
            Err(LinkError::DuplicateProcessArgument { arg, .. }) if arg == "event"
        ));

        let unknown_process = crate::parse(
            r#"
            trigger changed on tools.changed as event
              -> missing(event: event)
            "#,
        )
        .expect("parse unknown trigger target");
        assert!(matches!(
            LinkedModule::link(unknown_process, full_surface()),
            Err(LinkError::UnknownProcess { name, .. }) if name == "missing"
        ));

        let unknown_event = crate::parse(
            r#"
            process scan(event: any) { finish event }
            trigger changed on tools.missing as event
              -> scan(event: event)
            "#,
        )
        .expect("parse unknown event");
        assert!(matches!(
            LinkedModule::link(unknown_event, full_surface()),
            Err(LinkError::UnknownTriggerEvent { event, .. }) if event == "missing"
        ));

        let payload_mismatch = crate::parse(
            r#"
            type DifferentEvent = { id: str }
            process scan(event: DifferentEvent) { finish event.id }
            trigger changed on tools.changed as event
              -> scan(event: event)
            "#,
        )
        .expect("parse payload mismatch");
        assert!(matches!(
            LinkedModule::link(payload_mismatch, full_surface()),
            Err(LinkError::IncompatibleTriggerArgument { arg, .. }) if arg.as_ref() == "event"
        ));

        let resource_mismatch = crate::parse(
            r#"
            process scan(path: str) { finish path }
            trigger changed on tools.changed as event
              -> scan(path: tools)
            "#,
        )
        .expect("parse resource mismatch");
        assert!(matches!(
            LinkedModule::link(resource_mismatch, full_surface()),
            Err(LinkError::IncompatibleTriggerArgument { arg, .. }) if arg.as_ref() == "path"
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
        with_extra.add_operation("Tools", "unrelated", "unrelated");
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

    #[test]
    fn in_memory_artifact_store_rejects_corrupted_module_bytes() {
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

        store.put_raw_module_artifact_bytes(linked.module_ref.clone(), b"not json".to_vec());
        assert!(store.get_module_artifact(&linked.module_ref).is_err());
    }
}
