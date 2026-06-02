use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::artifact::{ModuleArtifact, ModuleRef, ProcessRef, RequiredSurfaceRef};
use crate::ast::{AstString, Expr, TypeExpr, TypeField, format_type_expr};
use crate::linker::ResourceCatalog;
use crate::runtime::{
    LASH_HOST_VALUE_KEY, LASH_HOST_VALUE_TYPE_KEY, LASH_MODULE_REF_KEY, LASH_PROCESS_NAME_KEY,
    LASH_PROCESS_REF_KEY, LASH_PROCESS_VALUE_KEY, LASH_REQUIRED_SURFACE_REF_KEY,
};

const TRIGGERS_RESOURCE_TYPE: &str = "Triggers";
const TRIGGERS_ALIAS: &str = "triggers";
const TRIGGER_REGISTRATION_TYPE: &str = "TriggerRegistration";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TriggerHostOperation {
    Register,
    List,
    Cancel,
}

impl TriggerHostOperation {
    pub const fn host_operation(self) -> &'static str {
        match self {
            Self::Register => "triggers.register",
            Self::List => "triggers.list",
            Self::Cancel => "triggers.cancel",
        }
    }

    pub const fn receiver_method(self) -> &'static str {
        match self {
            Self::Register => "register",
            Self::List => "list",
            Self::Cancel => "cancel",
        }
    }

    pub fn from_host_operation(operation: &str) -> Option<Self> {
        [Self::Register, Self::List, Self::Cancel]
            .into_iter()
            .find(|candidate| candidate.host_operation() == operation)
    }

    pub fn input_ty(self) -> TypeExpr {
        match self {
            Self::Register => TypeExpr::Object(vec![
                required_field("source", TypeExpr::Any),
                required_field("target", TypeExpr::Any),
                optional_field("name", TypeExpr::Str),
            ]),
            Self::List => TypeExpr::Object(vec![required_field("target", TypeExpr::Any)]),
            Self::Cancel => TypeExpr::Object(vec![required_field("handle", TypeExpr::Any)]),
        }
    }

    pub fn output_ty(self) -> TypeExpr {
        match self {
            Self::Register => TypeExpr::TriggerHandle(Box::new(TypeExpr::Any)),
            Self::List => TypeExpr::List(Box::new(TypeExpr::Ref(TRIGGER_REGISTRATION_TYPE.into()))),
            Self::Cancel => TypeExpr::Bool,
        }
    }
}

pub fn add_trigger_resource_operations(catalog: &mut ResourceCatalog) {
    catalog.add_module_instance([TRIGGERS_ALIAS], TRIGGERS_RESOURCE_TYPE);
    for operation in [
        TriggerHostOperation::Register,
        TriggerHostOperation::List,
        TriggerHostOperation::Cancel,
    ] {
        catalog.add_operation(
            TRIGGERS_RESOURCE_TYPE,
            operation.receiver_method(),
            operation.host_operation(),
            operation.input_ty(),
            operation.output_ty(),
        );
    }
}

fn required_field(name: &'static str, ty: TypeExpr) -> TypeField {
    TypeField {
        name: name.into(),
        ty,
        optional: false,
    }
}

fn optional_field(name: &'static str, ty: TypeExpr) -> TypeField {
    TypeField {
        name: name.into(),
        ty,
        optional: true,
    }
}

pub struct TriggerRegistrationCall<'expr> {
    pub source: &'expr Expr,
    pub target: &'expr Expr,
    pub name: Option<&'expr Expr>,
}

pub struct TriggerListCall<'expr> {
    pub target: &'expr Expr,
}

pub struct TriggerCancelCall<'expr> {
    pub handle: &'expr Expr,
}

pub fn register_call_args(
    args: &[Expr],
) -> Result<TriggerRegistrationCall<'_>, TriggerCallShapeError> {
    let entries = record_entries(args).ok_or(TriggerCallShapeError::Registration)?;
    Ok(TriggerRegistrationCall {
        source: required_entry(entries, "source").ok_or(TriggerCallShapeError::Registration)?,
        target: required_entry(entries, "target").ok_or(TriggerCallShapeError::Registration)?,
        name: required_entry(entries, "name"),
    })
}

pub fn list_call_args(args: &[Expr]) -> Result<TriggerListCall<'_>, TriggerCallShapeError> {
    let entries = record_entries(args).ok_or(TriggerCallShapeError::List)?;
    Ok(TriggerListCall {
        target: required_entry(entries, "target").ok_or(TriggerCallShapeError::List)?,
    })
}

pub fn cancel_call_args(args: &[Expr]) -> Result<TriggerCancelCall<'_>, TriggerCallShapeError> {
    let entries = record_entries(args).ok_or(TriggerCallShapeError::Cancel)?;
    Ok(TriggerCancelCall {
        handle: required_entry(entries, "handle").ok_or(TriggerCallShapeError::Cancel)?,
    })
}

fn record_entries(args: &[Expr]) -> Option<&[(AstString, Expr)]> {
    let [Expr::Record(entries)] = args else {
        return None;
    };
    Some(entries)
}

fn required_entry<'expr>(entries: &'expr [(AstString, Expr)], name: &str) -> Option<&'expr Expr> {
    entries
        .iter()
        .find_map(|(entry_name, expr)| (entry_name.as_str() == name).then_some(expr))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TriggerCallShapeError {
    Registration,
    List,
    Cancel,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TriggerRegistrationRequest {
    pub source: TriggerSourceValue,
    pub target: TriggerTargetIdentity,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl TriggerRegistrationRequest {
    pub fn decode(request: &serde_json::Value) -> Result<Self, TriggerRequestDecodeError> {
        let operation = TriggerHostOperation::Register;
        Ok(Self {
            source: TriggerSourceValue::decode(required_json_field(request, "source", operation)?)?,
            target: TriggerTargetIdentity::decode(
                required_json_field(request, "target", operation)?,
                "trigger target",
            )?,
            name: request
                .get("name")
                .and_then(serde_json::Value::as_str)
                .map(ToOwned::to_owned),
        })
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TriggerListRequest {
    pub target: TriggerTargetIdentity,
}

impl TriggerListRequest {
    pub fn decode(request: &serde_json::Value) -> Result<Self, TriggerRequestDecodeError> {
        Ok(Self {
            target: TriggerTargetIdentity::decode(
                required_json_field(request, "target", TriggerHostOperation::List)?,
                "triggers.list target",
            )?,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TriggerCancelRequest {
    pub handle: String,
}

impl TriggerCancelRequest {
    pub fn decode(request: &serde_json::Value) -> Result<Self, TriggerRequestDecodeError> {
        let value = required_json_field(request, "handle", TriggerHostOperation::Cancel)?;
        let handle = value
            .as_str()
            .map(ToOwned::to_owned)
            .or_else(|| {
                value
                    .get("id")
                    .and_then(serde_json::Value::as_str)
                    .map(ToOwned::to_owned)
            })
            .ok_or_else(|| TriggerRequestDecodeError::InvalidField {
                operation: TriggerHostOperation::Cancel.host_operation(),
                field: "handle",
                message: "expected trigger handle string or object with `id`".to_string(),
            })?;
        Ok(Self { handle })
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TriggerSourceValue {
    pub source_type: String,
    pub value: serde_json::Value,
}

impl TriggerSourceValue {
    pub fn decode(source: &serde_json::Value) -> Result<Self, TriggerRequestDecodeError> {
        let source_type = source
            .get(LASH_HOST_VALUE_TYPE_KEY)
            .and_then(serde_json::Value::as_str)
            .map(ToOwned::to_owned)
            .ok_or(TriggerRequestDecodeError::InvalidSource)?;
        let value = source
            .get(LASH_HOST_VALUE_KEY)
            .cloned()
            .ok_or(TriggerRequestDecodeError::InvalidSource)?;
        Ok(Self { source_type, value })
    }

    pub fn to_host_value(&self) -> serde_json::Value {
        serde_json::json!({
            LASH_HOST_VALUE_TYPE_KEY: self.source_type,
            LASH_HOST_VALUE_KEY: self.value,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TriggerTargetIdentity {
    pub module_ref: ModuleRef,
    pub required_surface_ref: RequiredSurfaceRef,
    pub process_ref: ProcessRef,
    pub process_name: String,
}

impl TriggerTargetIdentity {
    pub fn decode(
        value: &serde_json::Value,
        label: &'static str,
    ) -> Result<Self, TriggerRequestDecodeError> {
        if value
            .get(LASH_PROCESS_VALUE_KEY)
            .and_then(serde_json::Value::as_bool)
            != Some(true)
        {
            return Err(TriggerRequestDecodeError::InvalidTarget {
                label,
                message: "must be a process value".to_string(),
            });
        }
        Ok(Self {
            module_ref: decode_json_field(value, LASH_MODULE_REF_KEY, label)?,
            required_surface_ref: decode_json_field(value, LASH_REQUIRED_SURFACE_REF_KEY, label)?,
            process_ref: decode_json_field(value, LASH_PROCESS_REF_KEY, label)?,
            process_name: value
                .get(LASH_PROCESS_NAME_KEY)
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| TriggerRequestDecodeError::InvalidTarget {
                    label,
                    message: format!("missing {LASH_PROCESS_NAME_KEY}"),
                })?
                .to_string(),
        })
    }

    pub fn matches(
        &self,
        module_ref: &ModuleRef,
        required_surface_ref: &RequiredSurfaceRef,
        process_ref: &ProcessRef,
        process_name: &str,
    ) -> bool {
        self.module_ref == *module_ref
            && self.required_surface_ref == *required_surface_ref
            && self.process_ref == *process_ref
            && self.process_name == process_name
    }
}

pub fn event_type_for_source(
    resources: &ResourceCatalog,
    source_type: &str,
) -> Result<TypeExpr, TriggerRequestDecodeError> {
    resources
        .trigger_sources
        .get(source_type)
        .map(|binding| binding.event_ty.clone())
        .ok_or_else(|| TriggerRequestDecodeError::UnknownSourceType {
            source_type: source_type.to_string(),
        })
}

#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum TriggerRequestDecodeError {
    #[error("{operation} requires `{field}`")]
    MissingField {
        operation: &'static str,
        field: &'static str,
    },
    #[error("{operation} field `{field}` is invalid: {message}")]
    InvalidField {
        operation: &'static str,
        field: &'static str,
        message: String,
    },
    #[error("trigger source must be a host value constructor result")]
    InvalidSource,
    #[error("{label} {message}")]
    InvalidTarget {
        label: &'static str,
        message: String,
    },
    #[error("host value `{source_type}` is not registered as a trigger source")]
    UnknownSourceType { source_type: String },
}

fn required_json_field<'json>(
    request: &'json serde_json::Value,
    field: &'static str,
    operation: TriggerHostOperation,
) -> Result<&'json serde_json::Value, TriggerRequestDecodeError> {
    request
        .get(field)
        .ok_or_else(|| TriggerRequestDecodeError::MissingField {
            operation: operation.host_operation(),
            field,
        })
}

fn decode_json_field<T: serde::de::DeserializeOwned>(
    value: &serde_json::Value,
    field: &'static str,
    label: &'static str,
) -> Result<T, TriggerRequestDecodeError> {
    serde_json::from_value(value.get(field).cloned().ok_or_else(|| {
        TriggerRequestDecodeError::InvalidTarget {
            label,
            message: format!("missing {field}"),
        }
    })?)
    .map_err(|err| TriggerRequestDecodeError::InvalidTarget {
        label,
        message: format!("invalid {field}: {err}"),
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TriggerTargetValidation {
    pub input_name: String,
    pub event_ty: TypeExpr,
    pub input_ty: TypeExpr,
}

pub fn validate_trigger_target(
    target: &TriggerTargetIdentity,
    event_ty: &TypeExpr,
    artifact: &ModuleArtifact,
) -> Result<TriggerTargetValidation, TriggerTargetValidationError> {
    if artifact.required_surface_ref != target.required_surface_ref {
        return Err(TriggerTargetValidationError::RequiredSurfaceMismatch {
            process_name: target.process_name.clone(),
            target_surface: target.required_surface_ref.to_string(),
            artifact_surface: artifact.required_surface_ref.to_string(),
        });
    }
    let Some(exported_process_name) = artifact.process_name_for_ref(&target.process_ref) else {
        return Err(TriggerTargetValidationError::ProcessRefMismatch {
            module_ref: target.module_ref.to_string(),
            process_name: target.process_name.clone(),
            process_ref: format!("{:?}", target.process_ref),
        });
    };
    if exported_process_name != target.process_name {
        return Err(TriggerTargetValidationError::ProcessRefMismatch {
            module_ref: target.module_ref.to_string(),
            process_name: target.process_name.clone(),
            process_ref: format!("{:?}", target.process_ref),
        });
    }
    let process = artifact
        .canonical_ir
        .process(exported_process_name)
        .ok_or_else(|| TriggerTargetValidationError::MissingProcess {
            module_ref: target.module_ref.to_string(),
            process_name: target.process_name.clone(),
        })?;
    let [param] = process.params.as_slice() else {
        return Err(TriggerTargetValidationError::InvalidTargetInputCount {
            process_name: target.process_name.clone(),
        });
    };
    let aliases = type_aliases(artifact);
    let event_ty = resolve_type_aliases(event_ty, &aliases);
    let input_ty = resolve_type_aliases(&param.ty, &aliases);
    if !is_resolved_type_assignable(&event_ty, &input_ty) {
        return Err(TriggerTargetValidationError::EventMismatch {
            event: format_type_expr(&event_ty),
            process_name: target.process_name.clone(),
            input: format_type_expr(&input_ty),
        });
    }
    Ok(TriggerTargetValidation {
        input_name: param.name.to_string(),
        event_ty,
        input_ty,
    })
}

#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum TriggerTargetValidationError {
    #[error(
        "trigger target `{process_name}` required surface mismatch: target has {target_surface}, artifact has {artifact_surface}"
    )]
    RequiredSurfaceMismatch {
        process_name: String,
        target_surface: String,
        artifact_surface: String,
    },
    #[error(
        "trigger target artifact `{module_ref}` does not export process `{process_name}` as requested ref {process_ref}"
    )]
    ProcessRefMismatch {
        module_ref: String,
        process_name: String,
        process_ref: String,
    },
    #[error("trigger target artifact `{module_ref}` is missing process `{process_name}`")]
    MissingProcess {
        module_ref: String,
        process_name: String,
    },
    #[error("trigger target `{process_name}` must have exactly one input")]
    InvalidTargetInputCount { process_name: String },
    #[error("trigger source emits {event}, but target `{process_name}` expects {input}")]
    EventMismatch {
        event: String,
        process_name: String,
        input: String,
    },
}

pub fn resolve_type_aliases(ty: &TypeExpr, aliases: &BTreeMap<String, TypeExpr>) -> TypeExpr {
    resolve_type_aliases_inner(ty, aliases, &mut BTreeSet::new())
}

fn resolve_type_aliases_inner(
    ty: &TypeExpr,
    aliases: &BTreeMap<String, TypeExpr>,
    seen: &mut BTreeSet<String>,
) -> TypeExpr {
    match ty {
        TypeExpr::Ref(name) if seen.insert(name.to_string()) => {
            let resolved = aliases
                .get(name.as_str())
                .map(|ty| resolve_type_aliases_inner(ty, aliases, seen))
                .unwrap_or_else(|| ty.clone());
            seen.remove(name.as_str());
            resolved
        }
        TypeExpr::List(item) => {
            TypeExpr::List(Box::new(resolve_type_aliases_inner(item, aliases, seen)))
        }
        TypeExpr::Object(fields) => TypeExpr::Object(
            fields
                .iter()
                .map(|field| TypeField {
                    name: field.name.clone(),
                    ty: resolve_type_aliases_inner(&field.ty, aliases, seen),
                    optional: field.optional,
                })
                .collect(),
        ),
        TypeExpr::Union(items) => TypeExpr::Union(
            items
                .iter()
                .map(|item| resolve_type_aliases_inner(item, aliases, seen))
                .collect(),
        ),
        TypeExpr::Process {
            input,
            output,
            input_count,
        } => TypeExpr::Process {
            input: Box::new(resolve_type_aliases_inner(input, aliases, seen)),
            output: Box::new(resolve_type_aliases_inner(output, aliases, seen)),
            input_count: *input_count,
        },
        TypeExpr::TriggerHandle(event) => {
            TypeExpr::TriggerHandle(Box::new(resolve_type_aliases_inner(event, aliases, seen)))
        }
        _ => ty.clone(),
    }
}

pub fn is_resolved_type_assignable(source: &TypeExpr, target: &TypeExpr) -> bool {
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
        (
            TypeExpr::Process {
                input: source_input,
                output: source_output,
                input_count: source_count,
            },
            TypeExpr::Process {
                input: target_input,
                output: target_output,
                input_count: target_count,
            },
        ) => {
            source_count == target_count
                && is_resolved_type_assignable(source_input, target_input)
                && is_resolved_type_assignable(source_output, target_output)
        }
        (TypeExpr::TriggerHandle(source), TypeExpr::TriggerHandle(target)) => {
            is_resolved_type_assignable(source, target)
        }
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

fn type_aliases(artifact: &ModuleArtifact) -> BTreeMap<String, TypeExpr> {
    artifact
        .canonical_ir
        .declarations
        .iter()
        .filter_map(|declaration| match declaration {
            crate::Declaration::Type(decl) => Some((decl.name.to_string(), decl.ty.clone())),
            _ => None,
        })
        .collect()
}
