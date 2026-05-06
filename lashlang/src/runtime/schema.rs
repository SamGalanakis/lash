use super::{
    RuntimeError, Value,
    record::{Symbol, intern_symbol},
    unwrap_type_value,
};
use smallvec::SmallVec;
use std::fmt::Write as _;
use std::sync::Arc;

#[derive(Clone)]
pub(super) struct CompiledSchema {
    kind: CompiledSchemaKind,
}

#[derive(Clone)]
enum CompiledSchemaKind {
    Any,
    Type(SchemaType),
    Enum(Box<[Value]>),
    List(Box<CompiledSchema>),
    Object {
        required: Box<[CompiledSchemaField]>,
        properties: Box<[CompiledSchemaField]>,
    },
    /// Union of alternative shapes (`str | null`, `int | str`, …).
    /// A value matches the union if it matches any of the variants.
    Union(Box<[CompiledSchema]>),
}

#[derive(Clone)]
struct CompiledSchemaField {
    symbol: Symbol,
    name: Arc<str>,
    schema: CompiledSchema,
}

#[derive(Clone, Copy)]
enum SchemaType {
    String,
    Number,
    Integer,
    Boolean,
    Array,
    Object,
    Null,
}

pub(super) fn execute_validate_builtin(
    value: Value,
    schema: &Value,
) -> Result<Value, RuntimeError> {
    let schema = unwrap_type_value(schema).ok_or_else(|| RuntimeError::TypeError {
        message: "`validate` requires a Type literal as the second argument".to_string(),
    })?;
    execute_validate_schema(value, schema)
}

fn execute_validate_schema(value: Value, schema: &Value) -> Result<Value, RuntimeError> {
    validate_value_against_schema(&value, schema).map_err(|message| RuntimeError::ValueError {
        message: format!("validation failed: {message}"),
    })?;
    Ok(value)
}

pub(super) fn execute_compiled_validate(
    value: Value,
    schema: &CompiledSchema,
) -> Result<Value, RuntimeError> {
    let mut path = SmallVec::<[PathSegment<'_>; 8]>::new();
    validate_compiled_schema_node(&value, schema, &mut path).map_err(|message| {
        RuntimeError::ValueError {
            message: format!("validation failed: {message}"),
        }
    })?;
    Ok(value)
}

pub(super) fn compile_schema_value(schema: &Value) -> CompiledSchema {
    let Some(schema_obj) = schema.as_record() else {
        return CompiledSchema {
            kind: CompiledSchemaKind::Any,
        };
    };

    if let Some(Value::List(variants)) = schema_obj.get("anyOf") {
        let compiled: Box<[CompiledSchema]> = variants
            .iter()
            .map(compile_schema_value)
            .collect::<Vec<_>>()
            .into_boxed_slice();
        return CompiledSchema {
            kind: CompiledSchemaKind::Union(compiled),
        };
    }

    if let Some(Value::List(allowed)) = schema_obj.get("enum") {
        return CompiledSchema {
            kind: CompiledSchemaKind::Enum(allowed.iter().cloned().collect::<Vec<_>>().into()),
        };
    }

    let schema_type = match schema_obj.get("type") {
        Some(Value::String(expected)) => SchemaType::from_schema_name(expected.as_str()),
        _ => None,
    };

    match schema_type {
        Some(SchemaType::Array) => {
            let item_schema =
                schema_obj
                    .get("items")
                    .map(compile_schema_value)
                    .unwrap_or(CompiledSchema {
                        kind: CompiledSchemaKind::Any,
                    });
            CompiledSchema {
                kind: CompiledSchemaKind::List(Box::new(item_schema)),
            }
        }
        Some(SchemaType::Object) => {
            let required = match schema_obj.get("required") {
                Some(Value::List(required)) => required
                    .iter()
                    .filter_map(|field| match field {
                        Value::String(name) => Some(CompiledSchemaField {
                            symbol: intern_symbol(name.as_str()),
                            name: Arc::<str>::from(name.as_str()),
                            schema: CompiledSchema {
                                kind: CompiledSchemaKind::Any,
                            },
                        }),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
                _ => Box::default(),
            };
            let properties = match schema_obj.get("properties") {
                Some(Value::Record(properties)) => properties
                    .entries
                    .iter()
                    .map(|entry| CompiledSchemaField {
                        symbol: entry.symbol,
                        name: entry.name.clone(),
                        schema: compile_schema_value(&entry.value),
                    })
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
                _ => Box::default(),
            };
            CompiledSchema {
                kind: CompiledSchemaKind::Object {
                    required,
                    properties,
                },
            }
        }
        Some(kind) => CompiledSchema {
            kind: CompiledSchemaKind::Type(kind),
        },
        None => CompiledSchema {
            kind: CompiledSchemaKind::Any,
        },
    }
}

fn validate_value_against_schema(value: &Value, schema: &Value) -> Result<(), String> {
    let mut path = "$".to_string();
    validate_schema_node(value, schema, &mut path)
}

#[derive(Clone, Copy)]
enum PathSegment<'a> {
    Field(&'a str),
    Index(usize),
}

impl SchemaType {
    fn from_schema_name(name: &str) -> Option<Self> {
        Some(match name {
            "string" => Self::String,
            "number" => Self::Number,
            "integer" => Self::Integer,
            "boolean" => Self::Boolean,
            "array" => Self::Array,
            "object" => Self::Object,
            "null" => Self::Null,
            _ => return None,
        })
    }

    fn schema_name(self) -> &'static str {
        match self {
            Self::String => "string",
            Self::Number => "number",
            Self::Integer => "integer",
            Self::Boolean => "boolean",
            Self::Array => "array",
            Self::Object => "object",
            Self::Null => "null",
        }
    }

    fn matches(self, value: &Value) -> bool {
        match self {
            Self::String => matches!(value, Value::String(_)),
            Self::Number => matches!(value, Value::Number(number) if number.is_finite()),
            Self::Integer => {
                matches!(value, Value::Number(number) if number.is_finite() && number.fract() == 0.0)
            }
            Self::Boolean => matches!(value, Value::Bool(_)),
            Self::Array => matches!(value, Value::List(_)),
            Self::Object => matches!(value, Value::Record(_)),
            Self::Null => matches!(value, Value::Null),
        }
    }
}

fn validate_compiled_schema_type(
    value: &Value,
    expected: SchemaType,
    path: &[PathSegment<'_>],
) -> Result<(), String> {
    if expected.matches(value) {
        return Ok(());
    }
    Err(format!(
        "{}: expected {}, got {}",
        format_schema_path(path),
        expected.schema_name(),
        schema_value_type_name(value)
    ))
}

fn validate_compiled_schema_node<'a>(
    value: &Value,
    schema: &'a CompiledSchema,
    path: &mut SmallVec<[PathSegment<'a>; 8]>,
) -> Result<(), String> {
    match &schema.kind {
        CompiledSchemaKind::Any => Ok(()),
        CompiledSchemaKind::Union(variants) => {
            // Union matches if any variant matches. We report the
            // first-variant error when nothing matches, which is
            // consistent with how a reader would debug "this didn't
            // fit any of the shapes I declared".
            for variant in variants.iter() {
                if validate_compiled_schema_node(value, variant, path).is_ok() {
                    return Ok(());
                }
            }
            Err(format!(
                "{}: expected one of [{}], got {}",
                format_schema_path(path),
                variants
                    .iter()
                    .map(describe_compiled_schema)
                    .collect::<Vec<_>>()
                    .join(", "),
                schema_value_type_name(value)
            ))
        }
        CompiledSchemaKind::Type(expected) => validate_compiled_schema_type(value, *expected, path),
        CompiledSchemaKind::Enum(allowed) => {
            validate_compiled_schema_type(value, SchemaType::String, path)?;
            if allowed.iter().any(|candidate| candidate == value) {
                return Ok(());
            }
            let allowed = allowed
                .iter()
                .map(Value::to_string)
                .collect::<Vec<_>>()
                .join(", ");
            Err(format!(
                "{}: expected one of [{allowed}], got {value}",
                format_schema_path(path)
            ))
        }
        CompiledSchemaKind::List(items_schema) => {
            validate_compiled_schema_type(value, SchemaType::Array, path)?;
            let Value::List(items) = value else {
                return Ok(());
            };
            for (index, item) in items.iter().enumerate() {
                path.push(PathSegment::Index(index));
                validate_compiled_schema_node(item, items_schema, path)?;
                path.pop();
            }
            Ok(())
        }
        CompiledSchemaKind::Object {
            required,
            properties,
        } => {
            validate_compiled_schema_type(value, SchemaType::Object, path)?;
            let Value::Record(record) = value else {
                return Ok(());
            };
            for field in required.iter() {
                if record.get_symbol(field.symbol).is_none() {
                    return Err(format!(
                        "{}: missing required field `{}`",
                        format_schema_path(path),
                        field.name
                    ));
                }
            }
            for field in properties.iter() {
                if let Some(field_value) = record.get_symbol(field.symbol) {
                    path.push(PathSegment::Field(field.name.as_ref()));
                    validate_compiled_schema_node(field_value, &field.schema, path)?;
                    path.pop();
                }
            }
            Ok(())
        }
    }
}

/// Short human-readable label for a compiled schema, used in union
/// error messages ("expected one of [string, null]").
fn describe_compiled_schema(schema: &CompiledSchema) -> String {
    match &schema.kind {
        CompiledSchemaKind::Any => "any".to_string(),
        CompiledSchemaKind::Type(kind) => kind.schema_name().to_string(),
        CompiledSchemaKind::Enum(values) => format!(
            "enum[{}]",
            values
                .iter()
                .map(Value::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        CompiledSchemaKind::List(_) => "array".to_string(),
        CompiledSchemaKind::Object { .. } => "object".to_string(),
        CompiledSchemaKind::Union(variants) => variants
            .iter()
            .map(describe_compiled_schema)
            .collect::<Vec<_>>()
            .join(" | "),
    }
}

fn format_schema_path(path: &[PathSegment<'_>]) -> String {
    let mut formatted = "$".to_string();
    for segment in path {
        match segment {
            PathSegment::Field(name) => {
                formatted.push('.');
                formatted.push_str(name);
            }
            PathSegment::Index(index) => {
                write!(formatted, "[{index}]").expect("string writes should not fail");
            }
        }
    }
    formatted
}

fn validate_schema_node(value: &Value, schema: &Value, path: &mut String) -> Result<(), String> {
    let Some(schema_obj) = schema.as_record() else {
        return Ok(());
    };

    if let Some(Value::List(variants)) = schema_obj.get("anyOf") {
        for variant in variants.iter() {
            let mut scratch = path.clone();
            if validate_schema_node(value, variant, &mut scratch).is_ok() {
                return Ok(());
            }
        }
        return Err(format!(
            "{path}: value does not match any variant of the union",
        ));
    }

    if let Some(Value::String(expected)) = schema_obj.get("type")
        && !matches_schema_type(value, expected)
    {
        return Err(format!(
            "{path}: expected {expected}, got {}",
            schema_value_type_name(value)
        ));
    }

    if let Some(Value::List(allowed)) = schema_obj.get("enum")
        && !allowed.iter().any(|candidate| candidate == value)
    {
        let allowed = allowed
            .iter()
            .map(Value::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        return Err(format!("{path}: expected one of [{allowed}], got {value}"));
    }

    if let Some(Value::Record(properties)) = schema_obj.get("properties")
        && matches!(value, Value::Record(_) | Value::Image(_))
    {
        if let Some(Value::List(required)) = schema_obj.get("required") {
            for field in required.iter().filter_map(|field| match field {
                Value::String(name) => Some(name.as_str()),
                _ => None,
            }) {
                if schema_record_field(value, field).is_none() {
                    return Err(format!("{path}: missing required field `{field}`"));
                }
            }
        }

        for entry in properties.entries.iter() {
            if let Some(field_value) = schema_record_field(value, entry.name.as_ref()) {
                let base_len = path.len();
                path.push('.');
                path.push_str(entry.name.as_ref());
                validate_schema_node(&field_value, &entry.value, path)?;
                path.truncate(base_len);
            }
        }
    }

    if let Some(items_schema) = schema_obj.get("items")
        && let Value::List(items) = value
    {
        for (index, item) in items.iter().enumerate() {
            let base_len = path.len();
            write!(path, "[{index}]").expect("string writes should not fail");
            validate_schema_node(item, items_schema, path)?;
            path.truncate(base_len);
        }
    }

    Ok(())
}

fn matches_schema_type(value: &Value, expected: &str) -> bool {
    match expected {
        "string" => matches!(value, Value::String(_)),
        "number" => matches!(value, Value::Number(number) if number.is_finite()),
        "integer" => {
            matches!(value, Value::Number(number) if number.is_finite() && number.fract() == 0.0)
        }
        "boolean" => matches!(value, Value::Bool(_)),
        "array" => matches!(value, Value::List(_) | Value::Projected(_)),
        "object" => matches!(value, Value::Record(_) | Value::Image(_)),
        "null" => matches!(value, Value::Null),
        _ => true,
    }
}

fn schema_value_type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Image(_) => "object",
        Value::List(_) => "array",
        Value::Record(_) => "object",
        Value::Projected(value) => match value.value_type_name() {
            "list" => "array",
            _ => "object",
        },
    }
}

fn schema_record_field(value: &Value, field: &str) -> Option<Value> {
    match value {
        Value::Record(record) => record.get(field).cloned(),
        Value::Image(image) => match field {
            "type" => Some(Value::String("image".into())),
            "id" => Some(Value::String(image.id.clone().into())),
            "label" => Some(Value::String(image.label.clone().into())),
            "size" => Some(Value::Number(image.size as f64)),
            "width" => Some(
                image
                    .width
                    .map(|width| Value::Number(width as f64))
                    .unwrap_or(Value::Null),
            ),
            "height" => Some(
                image
                    .height
                    .map(|height| Value::Number(height as f64))
                    .unwrap_or(Value::Null),
            ),
            _ => None,
        },
        _ => None,
    }
}
