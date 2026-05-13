//! Field- and index-access on `Value`s, plus the assignment-path machinery.
//! Every read of `record.field` / `list[index]` and every write of
//! `record.field = …` / `list[index] = …` flows through these helpers.
//!
//! The async `read_field` / `read_index` are the entry points the VM calls;
//! the `*_direct` and `*_blocking` variants are sync fast paths used during
//! const-folding and pure-expression evaluation. `assign_path` and
//! `assign_path_steps` walk a `CompiledAssignPath` to mutate nested
//! structures in place.

use std::sync::Arc;

use compact_str::ToCompactString;

use super::instruction::Name;
use super::*;

pub(crate) fn read_field_ref_direct(value: &Value, field: &Name) -> Result<Value, RuntimeError> {
    match value {
        Value::Record(record) => Ok(record
            .get_symbol(field.symbol)
            .cloned()
            .unwrap_or(Value::Null)),
        Value::Image(image) => read_image_field(image, field),
        Value::Null => Ok(Value::Null),
        _ => Err(RuntimeError::TypeError {
            message: format!(
                "can't read `.{}` from {}",
                field.text,
                value_type_name(value)
            ),
        }),
    }
}

pub(crate) fn unwrap_tool_result(value: Value) -> Result<Value, RuntimeError> {
    let Value::Record(record) = value else {
        return Err(RuntimeError::TypeError {
            message: format!(
                "`?` expected a tool result wrapper, got {}",
                value_type_name(&value)
            ),
        });
    };

    let result_names = result_wrapper_names();
    match record.get_symbol(result_names.ok.symbol) {
        Some(Value::Bool(true)) => record
            .get_symbol(result_names.value.symbol)
            .cloned()
            .ok_or_else(|| RuntimeError::TypeError {
                message: "`?` found a successful tool result wrapper missing `value`".to_string(),
            }),
        Some(Value::Bool(false)) => {
            let message = record
                .get_symbol(result_names.error.symbol)
                .map(Value::to_string)
                .unwrap_or_else(|| "unknown error".to_string());
            Err(RuntimeError::ValueError {
                message: format!("`?` unwrapped failed tool result: {message}"),
            })
        }
        _ => Err(RuntimeError::TypeError {
            message: "`?` expected a tool result wrapper with boolean `ok`".to_string(),
        }),
    }
}

pub(crate) fn is_async_handle_record(record: &Record) -> bool {
    record.get("__handle__").is_some() || record.get("handle").is_some()
}

pub(crate) async fn read_field(value: Value, field: &Name) -> Result<Value, RuntimeError> {
    match value {
        Value::Projected(value) => value.get_field(field).await,
        other => read_field_direct(other, field),
    }
}

pub(crate) fn read_field_direct(value: Value, field: &Name) -> Result<Value, RuntimeError> {
    match value {
        Value::Record(record) => Ok(record
            .get_symbol(field.symbol)
            .cloned()
            .unwrap_or(Value::Null)),
        Value::Image(image) => read_image_field(&image, field),
        Value::Null => Ok(Value::Null),
        _ => Err(RuntimeError::TypeError {
            message: format!(
                "can't read `.{}` from {}",
                field.text,
                value_type_name(&value)
            ),
        }),
    }
}

pub(crate) fn read_field_blocking(value: Value, field: &Name) -> Result<Value, RuntimeError> {
    futures_executor::block_on(read_field(value, field))
}

pub(crate) fn read_image_field(image: &ImageValue, field: &Name) -> Result<Value, RuntimeError> {
    match field.text.as_ref() {
        "id" => Ok(Value::String(image.id.clone().into())),
        "label" => Ok(Value::String(image.label.clone().into())),
        "size" => Ok(Value::Number(image.size as f64)),
        "width" => Ok(image
            .width
            .map(|width| Value::Number(width as f64))
            .unwrap_or(Value::Null)),
        "height" => Ok(image
            .height
            .map(|height| Value::Number(height as f64))
            .unwrap_or(Value::Null)),
        _ => Ok(Value::Null),
    }
}

pub(crate) async fn read_index(target: Value, index: Value) -> Result<Value, RuntimeError> {
    match target {
        Value::Projected(value) => value.get_index(&index).await,
        other => read_index_direct(other, index),
    }
}

pub(crate) fn read_index_direct(target: Value, index: Value) -> Result<Value, RuntimeError> {
    read_index_ref_direct(&target, &index)
}

pub(crate) fn read_index_ref_direct(target: &Value, index: &Value) -> Result<Value, RuntimeError> {
    match target {
        Value::List(values) => {
            let idx = resolve_index(index, values.len())?;
            Ok(idx
                .and_then(|idx| values.get(idx).cloned())
                .unwrap_or(Value::Null))
        }
        Value::String(value) => {
            let idx = resolve_index(index, value.chars().count())?;
            Ok(idx
                .and_then(|idx| value.chars().nth(idx))
                .map(|ch| Value::String(ch.to_compact_string()))
                .unwrap_or(Value::Null))
        }
        Value::Record(record) => Ok(record
            .get(coerce_string(index)?.as_ref())
            .cloned()
            .unwrap_or(Value::Null)),
        Value::Null => Ok(Value::Null),
        _ => Err(RuntimeError::TypeError {
            message: format!("can't index {}", value_type_name(target)),
        }),
    }
}

pub(crate) fn read_index_blocking(target: Value, index: Value) -> Result<Value, RuntimeError> {
    futures_executor::block_on(read_index(target, index))
}

pub(crate) fn assign_path(
    root: &mut Value,
    path: &CompiledAssignPath,
    indexes: &[Value],
    value: Value,
    names: &[Name],
) -> Result<(), RuntimeError> {
    let mut index_cursor = 0;
    assign_path_steps(root, &path.steps, indexes, &mut index_cursor, value, names)
}

pub(crate) fn assign_path_steps(
    target: &mut Value,
    steps: &[CompiledAssignPathStep],
    indexes: &[Value],
    index_cursor: &mut usize,
    value: Value,
    names: &[Name],
) -> Result<(), RuntimeError> {
    let Some((step, rest)) = steps.split_first() else {
        *target = value;
        return Ok(());
    };

    match *step {
        CompiledAssignPathStep::Field(field) if rest.is_empty() => {
            assign_record_field(target, &names[field], value)
        }
        CompiledAssignPathStep::Field(field) => {
            let child = descend_record_field(target, &names[field])?;
            assign_path_steps(child, rest, indexes, index_cursor, value, names)
        }
        CompiledAssignPathStep::Index if rest.is_empty() => {
            let index = next_assign_index(indexes, index_cursor)?;
            assign_index(target, index, value)
        }
        CompiledAssignPathStep::Index => {
            let index = next_assign_index(indexes, index_cursor)?;
            let child = descend_index(target, index)?;
            assign_path_steps(child, rest, indexes, index_cursor, value, names)
        }
    }
}

pub(crate) fn next_assign_index<'a>(
    indexes: &'a [Value],
    index_cursor: &mut usize,
) -> Result<&'a Value, RuntimeError> {
    let index = indexes
        .get(*index_cursor)
        .ok_or_else(|| RuntimeError::ValueError {
            message: "missing assignment index".to_string(),
        })?;
    *index_cursor += 1;
    Ok(index)
}

pub(crate) fn assign_record_field(
    target: &mut Value,
    field: &Name,
    value: Value,
) -> Result<(), RuntimeError> {
    match target {
        Value::Record(record) => {
            Arc::make_mut(record).insert_symbolized(field.symbol, field.text.clone(), value);
            Ok(())
        }
        Value::Image(_) => Err(RuntimeError::TypeError {
            message: "can't assign image fields; images are immutable".to_string(),
        }),
        _ => Err(RuntimeError::TypeError {
            message: format!(
                "can't assign `.{}` on {}",
                field.text,
                value_type_name(target)
            ),
        }),
    }
}

pub(crate) fn descend_record_field<'a>(
    target: &'a mut Value,
    field: &Name,
) -> Result<&'a mut Value, RuntimeError> {
    match target {
        Value::Record(record) => Arc::make_mut(record)
            .get_symbol_mut(field.symbol)
            .ok_or_else(|| RuntimeError::ValueError {
                message: format!("can't assign through missing field `.{}`", field.text),
            }),
        Value::Image(_) => Err(RuntimeError::TypeError {
            message: "can't assign through image fields; images are immutable".to_string(),
        }),
        _ => Err(RuntimeError::TypeError {
            message: format!(
                "can't assign through `.{}` on {}",
                field.text,
                value_type_name(target)
            ),
        }),
    }
}

pub(crate) fn assign_index(
    target: &mut Value,
    index: &Value,
    value: Value,
) -> Result<(), RuntimeError> {
    match target {
        Value::List(values) => {
            let idx = resolve_existing_list_assignment_index(index, values.len())?;
            Arc::make_mut(values)[idx] = value;
            Ok(())
        }
        Value::Record(record) => {
            let key = coerce_string(index)?;
            Arc::make_mut(record).insert_str(key.as_ref(), value);
            Ok(())
        }
        Value::Image(_) => Err(RuntimeError::TypeError {
            message: "can't assign image fields; images are immutable".to_string(),
        }),
        _ => Err(RuntimeError::TypeError {
            message: format!("can't assign index on {}", value_type_name(target)),
        }),
    }
}

pub(crate) fn add_assign_index_number(
    target: &mut Value,
    index: &Value,
    right: f64,
) -> Result<Value, RuntimeError> {
    match target {
        Value::List(values) => {
            let idx = resolve_existing_list_assignment_index(index, values.len())?;
            add_assign_value_number(&mut Arc::make_mut(values)[idx], right)
        }
        Value::Record(record) => {
            let key = coerce_string(index)?;
            let record = Arc::make_mut(record);
            if let Some(value) = record.get_mut(key.as_ref()) {
                add_assign_value_number(value, right)
            } else {
                let value = Value::Number(right);
                record.insert_str(key.as_ref(), value.clone());
                Ok(value)
            }
        }
        Value::Image(_) => Err(RuntimeError::TypeError {
            message: "can't assign image fields; images are immutable".to_string(),
        }),
        _ => Err(RuntimeError::TypeError {
            message: format!("can't assign index on {}", value_type_name(target)),
        }),
    }
}

pub(crate) fn add_assign_value_number(
    value: &mut Value,
    right: f64,
) -> Result<Value, RuntimeError> {
    match value {
        Value::Number(left) => {
            *left += right;
            Ok(Value::Number(*left))
        }
        left => {
            let value = add_values(left.clone(), Value::Number(right))?;
            *left = value.clone();
            Ok(value)
        }
    }
}

pub(crate) fn descend_index<'a>(
    target: &'a mut Value,
    index: &Value,
) -> Result<&'a mut Value, RuntimeError> {
    match target {
        Value::List(values) => {
            let idx = resolve_existing_list_assignment_index(index, values.len())?;
            Ok(&mut Arc::make_mut(values)[idx])
        }
        Value::Record(record) => {
            let key = coerce_string(index)?;
            let record = Arc::make_mut(record);
            if let Some(value) = record.get_mut(key.as_ref()) {
                Ok(value)
            } else {
                Err(RuntimeError::ValueError {
                    message: format!("can't assign through missing key `{}`", key.as_ref()),
                })
            }
        }
        Value::Image(_) => Err(RuntimeError::TypeError {
            message: "can't assign through image fields; images are immutable".to_string(),
        }),
        _ => Err(RuntimeError::TypeError {
            message: format!("can't assign through index on {}", value_type_name(target)),
        }),
    }
}

pub(crate) fn resolve_index(index: &Value, len: usize) -> Result<Option<usize>, RuntimeError> {
    let index = as_offset(index)?;
    let len = len as isize;
    let normalized = if index < 0 { len + index } else { index };
    if normalized < 0 || normalized >= len {
        return Ok(None);
    }
    Ok(Some(normalized as usize))
}

pub(crate) fn resolve_existing_list_assignment_index(
    index: &Value,
    len: usize,
) -> Result<usize, RuntimeError> {
    let Value::Number(index) = index else {
        return Err(RuntimeError::TypeError {
            message: "list assignment index must be an integer".to_string(),
        });
    };
    if !index.is_finite() || index.fract() != 0.0 {
        return Err(RuntimeError::TypeError {
            message: "list assignment index must be an integer".to_string(),
        });
    }
    let index = *index as isize;
    let len = len as isize;
    let normalized = if index < 0 { len + index } else { index };
    if normalized < 0 || normalized >= len {
        return Err(RuntimeError::ValueError {
            message: "list assignment index out of bounds".to_string(),
        });
    }
    Ok(normalized as usize)
}
