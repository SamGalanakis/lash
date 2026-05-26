use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::ast::{
    AssignPathStep, AstString, Declaration, Expr, ProcessDecl, Program, ResourceRefExpr,
    ScheduleCadence, TriggerSource, TypeExpr,
};
use crate::lexer::Span;

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceCatalog {
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub resource_types: BTreeMap<String, ResourceTypeCatalog>,
}

impl ResourceCatalog {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn tool_default(operations: impl IntoIterator<Item = impl Into<String>>) -> Self {
        let mut catalog = Self::new();
        catalog.add_alias("TOOL", "default");
        for operation in operations {
            let operation = operation.into();
            catalog.add_operation("TOOL", operation.clone(), operation);
        }
        catalog
    }

    pub fn add_alias(&mut self, resource_type: impl Into<String>, alias: impl Into<String>) {
        self.resource_types
            .entry(resource_type.into())
            .or_default()
            .aliases
            .insert(alias.into());
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
    ) {
        self.resource_types
            .entry(resource_type.into())
            .or_default()
            .trigger_events
            .insert(event.into());
    }

    pub fn has_resource_type(&self, resource_type: &str) -> bool {
        self.resource_types.contains_key(resource_type)
    }

    pub fn resolve_alias(&self, resource: &ResourceRefExpr) -> Option<&ResourceTypeCatalog> {
        let ty = self.resource_types.get(resource.resource_type.as_str())?;
        ty.aliases.contains(resource.alias.as_str()).then_some(ty)
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

    pub fn has_operations(&self) -> bool {
        self.resource_types
            .values()
            .any(|resource_type| !resource_type.operations.is_empty())
    }

    pub fn supports_trigger(&self, resource_type: &str, event: &str) -> bool {
        self.resource_types
            .get(resource_type)
            .is_some_and(|ty| ty.trigger_events.contains(event))
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceTypeCatalog {
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub aliases: BTreeSet<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub operations: BTreeMap<String, ResourceOperationBinding>,
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub trigger_events: BTreeSet<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceOperationBinding {
    pub host_operation: String,
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
    pub process_sleep: bool,
    pub process_signals: bool,
    pub triggers: bool,
    pub schedules: LashlangScheduleAbilities,
}

impl LashlangAbilities {
    pub fn with_processes(mut self) -> Self {
        self.processes = true;
        self
    }

    pub fn with_process_lifecycle(mut self) -> Self {
        self.process_sleep = true;
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
            .with_processes()
            .with_process_lifecycle()
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

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LinkedModule {
    pub module_version: String,
    pub program: Program,
    pub surface: LashlangSurface,
}

impl LinkedModule {
    pub fn link(program: Program, surface: LashlangSurface) -> Result<Self, LinkError> {
        let mut linker = Linker::new(&program, &surface);
        linker.validate()?;
        let module_version = module_version(&program, &surface)?;
        Ok(Self {
            module_version,
            program,
            surface,
        })
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
    #[error("unknown resource `{resource_type}.{alias}`")]
    UnknownResource {
        resource_type: String,
        alias: String,
        span: Option<Span>,
    },
    #[error("resource type `{resource_type}` does not declare trigger event `{event}`")]
    UnknownTriggerEvent {
        resource_type: String,
        event: String,
        span: Option<Span>,
    },
    #[error("receiver for operation `{operation}` is not a catalog resource handle")]
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
    #[error(
        "catalog resource `{resource_type}.{alias}` cannot be used inside a process body; pass it as a process parameter"
    )]
    CatalogRefInsideProcess {
        resource_type: String,
        alias: String,
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
            | Self::UnresolvedReceiver { span, .. }
            | Self::UnknownResourceOperation { span, .. }
            | Self::CatalogRefInsideProcess { span, .. }
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
}

impl<'module> Linker<'module> {
    fn new(program: &'module Program, surface: &'module LashlangSurface) -> Self {
        Self {
            program,
            surface,
            process_names: BTreeSet::new(),
            type_names: BTreeSet::new(),
        }
    }

    fn validate(&mut self) -> Result<(), LinkError> {
        self.collect_declarations()?;
        for (index, declaration) in self.program.declarations.iter().enumerate() {
            let span = self.program.declaration_spans.get(index).copied();
            match declaration {
                Declaration::Process(process) => self.validate_process(process, span)?,
                Declaration::Trigger(trigger) => {
                    self.ensure_feature(self.surface.abilities.triggers, "triggers", span)?;
                    let mut scope = Scope::new(false, true, false, span);
                    scope.bind(trigger.event_binding.as_str(), Binding::Value);
                    match &trigger.source {
                        TriggerSource::Binding { resource, event } => {
                            self.validate_resource_ref(resource, span)?;
                            if !self
                                .surface
                                .resources
                                .supports_trigger(resource.resource_type.as_str(), event)
                            {
                                return Err(LinkError::UnknownTriggerEvent {
                                    resource_type: resource.resource_type.to_string(),
                                    event: event.to_string(),
                                    span,
                                });
                            }
                        }
                        TriggerSource::Each {
                            resource_type,
                            event,
                            resource_binding,
                        } => {
                            if !self
                                .surface
                                .resources
                                .supports_trigger(resource_type, event)
                            {
                                return Err(LinkError::UnknownTriggerEvent {
                                    resource_type: resource_type.to_string(),
                                    event: event.to_string(),
                                    span,
                                });
                            }
                            scope.bind(
                                resource_binding.as_str(),
                                Binding::Resource {
                                    resource_type: resource_type.to_string(),
                                },
                            );
                        }
                    }
                    self.validate_expr(&trigger.body, &mut scope)?;
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
                    let mut scope = Scope::new(false, true, false, span);
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

        let mut top_level = Scope::new(true, true, false, None);
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
                    decl.name.as_str()
                }
                Declaration::Process(decl) => {
                    self.ensure_feature(self.surface.abilities.processes, "processes", span)?;
                    self.process_names.insert(decl.name.to_string());
                    decl.name.as_str()
                }
                Declaration::Trigger(decl) => {
                    self.ensure_feature(self.surface.abilities.triggers, "triggers", span)?;
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
        let mut scope = Scope::new(false, false, true, span);
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

    fn binding_for_type(&self, ty: &TypeExpr) -> Binding {
        match ty {
            TypeExpr::Ref(name) if self.surface.resources.has_resource_type(name) => {
                Binding::Resource {
                    resource_type: name.to_string(),
                }
            }
            _ => Binding::Value,
        }
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
                    if self
                        .surface
                        .resources
                        .resolve_operation("TOOL", start.process.as_str())
                        .is_some()
                    {
                        let mut seen = BTreeSet::new();
                        for (arg, value) in &start.args {
                            if !seen.insert(arg.to_string()) {
                                return Err(LinkError::DuplicateProcessArgument {
                                    arg: arg.to_string(),
                                    span: scope.span,
                                });
                            }
                            self.validate_expr(value, scope)?;
                        }
                        return Ok(Some(Binding::Value));
                    }
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
                    if !process.params.iter().any(|param| param.name == *arg) {
                        return Err(LinkError::UnexpectedProcessArgument {
                            process: process.name.to_string(),
                            arg: arg.to_string(),
                            span: scope.span,
                        });
                    }
                    self.validate_expr(value, scope)?;
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
                self.ensure_feature(
                    self.surface.abilities.process_sleep,
                    "process sleep",
                    scope.span,
                )?;
                if !scope.process_body {
                    return Err(LinkError::ProcessLifecycleOutsideProcess {
                        keyword: "sleep",
                        span: scope.span,
                    });
                }
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
                if !scope.allow_catalog_refs {
                    return Err(LinkError::CatalogRefInsideProcess {
                        resource_type: resource.resource_type.to_string(),
                        alias: resource.alias.to_string(),
                        span: scope.span,
                    });
                }
                self.validate_resource_ref(resource, scope.span)?;
                Ok(Some(Binding::Resource {
                    resource_type: resource.resource_type.to_string(),
                }))
            }
            Expr::ReceiverCall {
                receiver,
                operation,
                args,
            } => {
                let receiver = self.validate_expr(receiver, scope)?;
                let Some(Binding::Resource { resource_type }) = receiver else {
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
    ) -> Result<(), LinkError> {
        self.surface
            .resources
            .resolve_alias(resource)
            .map(|_| ())
            .ok_or_else(|| LinkError::UnknownResource {
                resource_type: resource.resource_type.to_string(),
                alias: resource.alias.to_string(),
                span,
            })
    }
}

#[derive(Clone)]
struct Scope {
    bindings: BTreeMap<String, Binding>,
    allow_unknown_globals: bool,
    allow_catalog_refs: bool,
    process_body: bool,
    span: Option<Span>,
}

impl Scope {
    fn new(
        allow_unknown_globals: bool,
        allow_catalog_refs: bool,
        process_body: bool,
        span: Option<Span>,
    ) -> Self {
        Self {
            bindings: BTreeMap::new(),
            allow_unknown_globals,
            allow_catalog_refs,
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

    fn merge_from(&mut self, other: Scope) {
        for (name, binding) in other.bindings {
            self.bindings.entry(name).or_insert(binding);
        }
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

fn module_version(program: &Program, surface: &LashlangSurface) -> Result<String, LinkError> {
    let payload = serde_json::json!({
        "program": program,
        "surface": surface,
    });
    let bytes = serde_json::to_vec(&payload).map_err(|err| LinkError::ModuleHash {
        message: err.to_string(),
    })?;
    let digest = Sha256::digest(bytes);
    Ok(format!("sha256:{}", hex_digest(&digest)))
}

fn hex_digest(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resources() -> ResourceCatalog {
        let mut catalog = ResourceCatalog::new();
        catalog.add_alias("TOOL", "default");
        catalog.add_operation("TOOL", "read_file", "read_file");
        catalog.add_operation("TOOL", "echo", "echo");
        catalog.add_trigger_event("TOOL", "changed");
        catalog
    }

    fn full_surface() -> LashlangSurface {
        LashlangSurface::new(resources(), LashlangAbilities::all())
    }

    #[test]
    fn linked_module_accepts_named_processes_resource_params_and_activations() {
        let program = crate::parse(
            r#"
            process scan(tool: TOOL, path: str) {
              text = await tool.read_file({ path: path })?
              finish text
            }
            process watcher(run: any) {
              sleep for "0ms"
              signal = wait signal
              signal run run with signal
              finish signal
            }
            trigger changed on TOOL.default.changed as event {
              start scan(tool: TOOL.default, path: event.path)
            }
            schedule hourly every cron("0 * * * *") as tick {
              start scan(tool: TOOL.default, path: tick.path)
            }
            "#,
        )
        .expect("parse module");

        let linked = LinkedModule::link(program, full_surface()).expect("link module");

        assert!(linked.module_version.starts_with("sha256:"));
    }

    #[test]
    fn linked_module_rejects_process_lifecycle_outside_process_body() {
        let program = crate::parse("sleep for 1").expect("parse sleep");

        let err =
            LinkedModule::link(program, full_surface()).expect_err("top-level sleep rejected");

        assert!(
            matches!(
                err,
                LinkError::ProcessLifecycleOutsideProcess {
                    keyword: "sleep",
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
            process scan(tool: TOOL, path: str) { finish path }
            start scan(tool: TOOL.default)
            "#,
        )
        .expect("parse missing arg");
        assert!(matches!(
            LinkedModule::link(missing_arg, full_surface()),
            Err(LinkError::MissingProcessArgument { arg, .. }) if arg == "path"
        ));

        let bad_operation = crate::parse(
            r#"
            process scan(tool: TOOL) {
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

        let sleep = crate::parse("process worker() { sleep for \"1s\" }")
            .expect("parse disabled process sleep");
        assert!(matches!(
            LinkedModule::link(
                sleep,
                LashlangSurface::new(resources(), LashlangAbilities::default().with_processes())
            ),
            Err(LinkError::FeatureDisabled {
                feature: "process sleep",
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
            trigger changed on TOOL.default.changed as event {
              print event
            }
            "#,
        )
        .expect("parse disabled trigger");
        assert!(matches!(
            LinkedModule::link(
                trigger,
                LashlangSurface::new(
                    resources(),
                    LashlangAbilities::default().with_cron_schedules()
                )
            ),
            Err(LinkError::FeatureDisabled {
                feature: "triggers",
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
        assert!(matches!(
            LinkedModule::link(
                schedule,
                LashlangSurface::new(resources(), LashlangAbilities::all())
            ),
            Ok(_)
        ));
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
                        .with_process_lifecycle()
                )
            ),
            Err(LinkError::FeatureDisabled {
                feature: "cron schedules",
                ..
            })
        ));
    }

    #[test]
    fn linked_module_hash_changes_when_abilities_change() {
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

        assert_ne!(minimal.module_version, processes.module_version);
    }
}
