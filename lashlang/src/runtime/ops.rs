//! Binary operations, comparison, numeric coercion, range/iterator helpers,
//! `is_truthy` / `materialize_projected` / `value_type_name`, builtin
//! dispatch (`execute_builtin` and the per-builtin direct paths), plus
//! `eval_pure_expr` — the const-folder shared between `Compiler::compile_*`
//! and `Vm::run_pure_*`.

use std::borrow::Cow;
use std::sync::Arc;

use smallvec::SmallVec;

use crate::ast::{BinaryOp, UnaryOp};

use super::*;
use super::instruction::{Name, PureExpr};

pub(crate) fn eval_pure_expr(
    expr: &PureExpr,
    slots: &SlotState,
    names: &[Name],
    slot_names: &[Name],
) -> Result<Value, RuntimeError> {
    match expr {
        PureExpr::Const(value) => Ok(value.clone()),
        PureExpr::Slot(slot) => {
            slots
                .get(*slot)
                .cloned()
                .ok_or_else(|| RuntimeError::UndefinedVariable {
                    name: slot_names[*slot].text.to_string(),
                })
        }
        PureExpr::List(items) => Ok(Value::List(
            items
                .iter()
                .map(|item| eval_pure_expr(item, slots, names, slot_names))
                .collect::<Result<Vec<_>, _>>()?
                .into(),
        )),
        PureExpr::Record(entries) => {
            let mut record = record_with_capacity(entries.len());
            for (key, value) in entries.iter() {
                let name_entry = &names[*key];
                record.insert_symbolized(
                    name_entry.symbol,
                    name_entry.text.clone(),
                    eval_pure_expr(value, slots, names, slot_names)?,
                );
            }
            Ok(Value::Record(Arc::new(record)))
        }
        PureExpr::Builtin { builtin, args } => {
            let mut values = SmallVec::<[Value; 8]>::with_capacity(args.len());
            for arg in args.iter() {
                values.push(eval_pure_expr(arg, slots, names, slot_names)?);
            }
            execute_builtin_blocking(*builtin, names, &values)
        }
        PureExpr::Format { template, args } => {
            let mut values = SmallVec::<[Value; 8]>::with_capacity(args.len());
            for arg in args.iter() {
                values.push(eval_pure_expr(arg, slots, names, slot_names)?);
            }
            Ok(Value::String(
                execute_compiled_format_blocking(template, &values)?.into(),
            ))
        }
        PureExpr::ResultUnwrap(expr) => {
            let value = eval_pure_expr(expr, slots, names, slot_names)?;
            unwrap_tool_result(value)
        }
        PureExpr::Field { target, field } => {
            let value = eval_pure_expr(target, slots, names, slot_names)?;
            read_field_blocking(value, &names[*field])
        }
        PureExpr::Index { target, index } => {
            let target = eval_pure_expr(target, slots, names, slot_names)?;
            let index = eval_pure_expr(index, slots, names, slot_names)?;
            read_index_blocking(target, index)
        }
        PureExpr::Unary { op, expr } => {
            let value = eval_pure_expr(expr, slots, names, slot_names)?;
            match op {
                UnaryOp::Negate => Ok(Value::Number(-as_number(&value)?)),
                UnaryOp::Not => Ok(Value::Bool(!is_truthy(&value))),
            }
        }
        PureExpr::Conditional {
            condition,
            then_expr,
            else_expr,
        } => {
            if is_truthy(&eval_pure_expr(condition, slots, names, slot_names)?) {
                eval_pure_expr(then_expr, slots, names, slot_names)
            } else {
                eval_pure_expr(else_expr, slots, names, slot_names)
            }
        }
        PureExpr::Binary { left, op, right } => match op {
            BinaryOp::And => Ok(Value::Bool(
                is_truthy(&eval_pure_expr(left, slots, names, slot_names)?)
                    && is_truthy(&eval_pure_expr(right, slots, names, slot_names)?),
            )),
            BinaryOp::Or => Ok(Value::Bool(
                is_truthy(&eval_pure_expr(left, slots, names, slot_names)?)
                    || is_truthy(&eval_pure_expr(right, slots, names, slot_names)?),
            )),
            _ => {
                let left = eval_pure_expr(left, slots, names, slot_names)?;
                let right = eval_pure_expr(right, slots, names, slot_names)?;
                eval_binary_values(left, *op, right)
            }
        },
    }
}

pub(crate) fn expect_arg_count(name: &str, values: &[Value], expected: usize) -> Result<(), RuntimeError> {
    if values.len() == expected {
        Ok(())
    } else {
        Err(RuntimeError::TypeError {
            message: format!("`{name}` takes {expected} arg(s), got {}", values.len()),
        })
    }
}

pub(crate) async fn execute_builtin(
    builtin: Builtin,
    names: &[Name],
    values: &[Value],
) -> Result<Value, RuntimeError> {
    match builtin {
        Builtin::Len => {
            expect_arg_count("len", values, 1)?;
            execute_len_builtin(&values[0]).await
        }
        Builtin::Empty => {
            expect_arg_count("empty", values, 1)?;
            match &values[0] {
                Value::String(value) => Ok(Value::Bool(value.is_empty())),
                Value::List(values) => Ok(Value::Bool(values.is_empty())),
                Value::Record(record) => Ok(Value::Bool(record.is_empty())),
                Value::Projected(value) => Ok(Value::Bool(value.is_empty().await)),
                Value::Null => Ok(Value::Bool(true)),
                _ => Err(RuntimeError::TypeError {
                    message: "`empty` requires a string, list, record, or null".to_string(),
                }),
            }
        }
        Builtin::Keys => {
            expect_arg_count("keys", values, 1)?;
            match &values[0] {
                Value::Record(record) => Ok(Value::List(
                    record
                        .keys()
                        .map(|key| Value::String(key.to_string().into()))
                        .collect::<Vec<_>>()
                        .into(),
                )),
                Value::Projected(value) => Ok(Value::List(
                    value
                        .keys()
                        .await
                        .into_iter()
                        .map(|key| Value::String(key.into()))
                        .collect::<Vec<_>>()
                        .into(),
                )),
                Value::Null => Ok(Value::List(Vec::new().into())),
                _ => Err(RuntimeError::TypeError {
                    message: "`keys` requires a record or null".to_string(),
                }),
            }
        }
        Builtin::Values => {
            expect_arg_count("values", values, 1)?;
            match &values[0] {
                Value::Record(record) => Ok(Value::List(
                    record.values().cloned().collect::<Vec<_>>().into(),
                )),
                Value::Null => Ok(Value::List(Vec::new().into())),
                _ => Err(RuntimeError::TypeError {
                    message: "`values` requires a record or null".to_string(),
                }),
            }
        }
        Builtin::Contains => {
            expect_arg_count("contains", values, 2)?;
            execute_contains_builtin(&values[0], &values[1]).await
        }
        Builtin::StartsWith => {
            expect_arg_count("starts_with", values, 2)?;
            let value = coerce_string(&values[0])?;
            let prefix = coerce_string(&values[1])?;
            Ok(Value::Bool(value.starts_with(prefix.as_ref())))
        }
        Builtin::EndsWith => {
            expect_arg_count("ends_with", values, 2)?;
            let value = coerce_string(&values[0])?;
            let suffix = coerce_string(&values[1])?;
            Ok(Value::Bool(value.ends_with(suffix.as_ref())))
        }
        Builtin::Split => {
            expect_arg_count("split", values, 2)?;
            let value = coerce_string(&values[0])?;
            let needle = coerce_string(&values[1])?;
            Ok(Value::List(
                value
                    .split(needle.as_ref())
                    .map(|part| Value::String(part.to_string().into()))
                    .collect::<Vec<_>>()
                    .into(),
            ))
        }
        Builtin::Join => {
            expect_arg_count("join", values, 2)?;
            execute_join_builtin(&values[0], &values[1])
        }
        Builtin::Trim => {
            expect_arg_count("trim", values, 1)?;
            Ok(Value::String(
                coerce_string(&values[0])?.trim().to_string().into(),
            ))
        }
        Builtin::Slice => {
            expect_arg_count("slice", values, 3)?;
            let start = as_slice_bound(&values[1])?;
            let end = as_slice_bound(&values[2])?;
            match &values[0] {
                Value::String(value) => Ok(Value::String(slice_string(value, start, end).into())),
                Value::List(items) => {
                    let Some((start, end)) = clamp_slice_bounds(start, end, items.len()) else {
                        return Ok(Value::List(Vec::new().into()));
                    };
                    Ok(Value::List(items[start..end].to_vec().into()))
                }
                _ => Err(RuntimeError::TypeError {
                    message: "`slice` requires a string or list".to_string(),
                }),
            }
        }
        Builtin::ToString => {
            expect_arg_count("to_string", values, 1)?;
            let value = if value_contains_projected(&values[0]) {
                stringify_value_async(&values[0]).await?
            } else {
                stringify_value_direct(&values[0])?
            };
            Ok(Value::String(value.into()))
        }
        Builtin::ToInt => {
            expect_arg_count("to_int", values, 1)?;
            Ok(Value::Number(as_number(&values[0])?.trunc()))
        }
        Builtin::ToFloat => {
            expect_arg_count("to_float", values, 1)?;
            Ok(Value::Number(as_number(&values[0])?))
        }
        Builtin::JsonParse => {
            expect_arg_count("json_parse", values, 1)?;
            let parsed: serde_json::Value = serde_json::from_str(&coerce_string(&values[0])?)
                .map_err(|err| RuntimeError::ValueError {
                    message: format!("invalid json: {err}"),
                })?;
            Ok(from_json(parsed))
        }
        Builtin::Format => {
            if values.is_empty() {
                return Err(RuntimeError::TypeError {
                    message: "`format` requires at least a template string".to_string(),
                });
            }
            let template = match &values[0] {
                Value::String(value) => value.as_str(),
                other => {
                    return Err(RuntimeError::TypeError {
                        message: format!(
                            "`format` template must be a string, got {}",
                            value_type_name(other)
                        ),
                    });
                }
            };
            Ok(Value::String(
                apply_format_async(template, &values[1..]).await?.into(),
            ))
        }
        Builtin::Validate => {
            expect_arg_count("validate", values, 2)?;
            execute_validate_builtin(values[0].clone(), &values[1])
        }
        Builtin::Range => execute_range_builtin(values),
        Builtin::Push => {
            expect_arg_count("push", values, 2)?;
            execute_push_builtin(&values[0], values[1].clone())
        }
        Builtin::Unknown(index) => Err(RuntimeError::UnknownBuiltin {
            name: names[index].text.to_string(),
        }),
    }
}

pub(crate) fn execute_builtin_blocking(
    builtin: Builtin,
    names: &[Name],
    values: &[Value],
) -> Result<Value, RuntimeError> {
    futures_executor::block_on(execute_builtin(builtin, names, values))
}

pub(crate) async fn execute_len_builtin(value: &Value) -> Result<Value, RuntimeError> {
    if let Value::Projected(value) = value {
        return Ok(Value::Number(value.len().await as f64));
    }
    execute_len_direct(value)
}

pub(crate) fn execute_len_direct(value: &Value) -> Result<Value, RuntimeError> {
    value_len(value)
        .map(|len| Value::Number(len as f64))
        .ok_or_else(|| RuntimeError::TypeError {
            message: "`len` requires a string, list, record, or null; use `.size` for images"
                .to_string(),
        })
}

pub(crate) async fn execute_contains_builtin(haystack: &Value, needle: &Value) -> Result<Value, RuntimeError> {
    if !matches!(haystack, Value::Projected(_)) {
        return execute_contains_direct(haystack, needle).map(Value::Bool);
    }
    match (haystack, needle) {
        (Value::Projected(value), needle) => Ok(Value::Bool(value.contains(needle).await?)),
        (Value::Null, _) => Ok(Value::Bool(false)),
        _ => Err(RuntimeError::TypeError {
            message:
                "`contains` requires a string/string, list/value, record/key, or null/value pair"
                    .to_string(),
        }),
    }
}

pub(crate) fn execute_contains_direct(haystack: &Value, needle: &Value) -> Result<bool, RuntimeError> {
    match (haystack, needle) {
        (Value::String(haystack), needle) => Ok(haystack.contains(coerce_string(needle)?.as_ref())),
        (Value::List(items), needle) => Ok(items.contains(needle)),
        (Value::Record(record), needle) => {
            Ok(record.get(coerce_string(needle)?.as_ref()).is_some())
        }
        (Value::Null, _) => Ok(false),
        _ => Err(RuntimeError::TypeError {
            message:
                "`contains` requires a string/string, list/value, record/key, or null/value pair"
                    .to_string(),
        }),
    }
}

pub(crate) fn value_len(value: &Value) -> Option<usize> {
    match value {
        Value::String(value) => Some(value.chars().count()),
        Value::List(values) => Some(values.len()),
        Value::Record(record) => Some(record.len()),
        Value::Null => Some(0),
        _ => None,
    }
}

pub(crate) async fn iterable_values(value: Value) -> Result<Arc<[Value]>, RuntimeError> {
    match value {
        Value::List(values) => Ok(values),
        Value::Projected(value) => match value.materialize_async().await {
            Value::List(values) => Ok(values),
            _ => Err(RuntimeError::NonListIteration),
        },
        _ => Err(RuntimeError::NonListIteration),
    }
}

pub(crate) fn execute_join_builtin(items: &Value, sep: &Value) -> Result<Value, RuntimeError> {
    let Value::List(items) = items else {
        return Err(RuntimeError::TypeError {
            message: "`join` requires a list as the first argument".to_string(),
        });
    };
    let sep = coerce_string(sep)?;
    let mut joined = String::new();
    for (index, item) in items.iter().enumerate() {
        if index > 0 {
            joined.push_str(sep.as_ref());
        }
        joined.push_str(coerce_string(item)?.as_ref());
    }
    Ok(Value::String(joined.into()))
}

pub(crate) fn execute_range_builtin(values: &[Value]) -> Result<Value, RuntimeError> {
    let (start, end) = range_bounds(values)?;
    build_range(start, end)
}

pub(crate) fn range_bounds(values: &[Value]) -> Result<(i64, i64), RuntimeError> {
    let (start, end) = match values {
        [end] => (0, as_range_bound(end)?),
        [start, end] => (as_range_bound(start)?, as_range_bound(end)?),
        _ => {
            return Err(RuntimeError::TypeError {
                message: format!("`range` takes 1 or 2 arg(s), got {}", values.len()),
            });
        }
    };
    validate_range_len(start, end)?;
    Ok((start, end))
}

pub(crate) fn execute_push_builtin(list: &Value, item: Value) -> Result<Value, RuntimeError> {
    let Value::List(items) = list else {
        return Err(RuntimeError::TypeError {
            message: "`push` requires a list as the first argument".to_string(),
        });
    };
    let mut values = Vec::with_capacity(items.len() + 1);
    values.extend(items.iter().cloned());
    values.push(item);
    Ok(Value::List(values.into()))
}

pub(crate) fn as_range_bound(value: &Value) -> Result<i64, RuntimeError> {
    let Value::Number(number) = value else {
        return Err(RuntimeError::TypeError {
            message: format!(
                "`range` bounds must be finite integers, got {}",
                value_type_name(value)
            ),
        });
    };
    if !number.is_finite()
        || number.fract() != 0.0
        || *number < i64::MIN as f64
        || *number > i64::MAX as f64
    {
        return Err(RuntimeError::TypeError {
            message: "`range` bounds must be finite integers".to_string(),
        });
    }
    Ok(*number as i64)
}

pub(crate) fn build_range(start: i64, end: i64) -> Result<Value, RuntimeError> {
    if start >= end {
        return Ok(Value::List(Vec::new().into()));
    }
    Ok(Value::List(
        (start..end)
            .map(|value| Value::Number(value as f64))
            .collect::<Vec<_>>()
            .into(),
    ))
}

pub(crate) fn validate_range_len(start: i64, end: i64) -> Result<(), RuntimeError> {
    const MAX_RANGE_ITEMS: i64 = 1_000_000;
    if start >= end {
        return Ok(());
    }
    let len = end as i128 - start as i128;
    if len > MAX_RANGE_ITEMS as i128 {
        return Err(RuntimeError::ValueError {
            message: format!("`range` would create more than {MAX_RANGE_ITEMS} items"),
        });
    }
    Ok(())
}


pub(crate) fn eval_binary_values(left: Value, op: BinaryOp, right: Value) -> Result<Value, RuntimeError> {
    match op {
        BinaryOp::Add => add_values(left, right),
        BinaryOp::Subtract => numeric_binary_values(left, right, |a, b| a - b),
        BinaryOp::Multiply => numeric_binary_values(left, right, |a, b| a * b),
        BinaryOp::Divide => numeric_binary_values(left, right, |a, b| a / b),
        BinaryOp::Modulo => numeric_binary_values(left, right, |a, b| a % b),
        BinaryOp::Equal => Ok(Value::Bool(left == right)),
        BinaryOp::NotEqual => Ok(Value::Bool(left != right)),
        BinaryOp::Less => compare_ordered(left, right, |a, b| a < b, |a, b| a < b),
        BinaryOp::LessEqual => compare_ordered(left, right, |a, b| a <= b, |a, b| a <= b),
        BinaryOp::Greater => compare_ordered(left, right, |a, b| a > b, |a, b| a > b),
        BinaryOp::GreaterEqual => compare_ordered(left, right, |a, b| a >= b, |a, b| a >= b),
        BinaryOp::And | BinaryOp::Or => unreachable!("logical ops are compiled with jumps"),
    }
}

pub(crate) fn eval_number_binary_values(left: f64, op: BinaryOp, right: f64) -> Value {
    match op {
        BinaryOp::Add
        | BinaryOp::Subtract
        | BinaryOp::Multiply
        | BinaryOp::Divide
        | BinaryOp::Modulo => Value::Number(eval_number_numeric_binary_value(left, op, right)),
        BinaryOp::Equal => Value::Bool(left == right),
        BinaryOp::NotEqual => Value::Bool(left != right),
        BinaryOp::Less => Value::Bool(left < right),
        BinaryOp::LessEqual => Value::Bool(left <= right),
        BinaryOp::Greater => Value::Bool(left > right),
        BinaryOp::GreaterEqual => Value::Bool(left >= right),
        BinaryOp::And | BinaryOp::Or => unreachable!("logical ops are compiled with jumps"),
    }
}

pub(crate) fn eval_number_numeric_binary_value(left: f64, op: BinaryOp, right: f64) -> f64 {
    match op {
        BinaryOp::Add => left + right,
        BinaryOp::Subtract => left - right,
        BinaryOp::Multiply => left * right,
        BinaryOp::Divide => left / right,
        BinaryOp::Modulo => left % right,
        _ => unreachable!("non-numeric op in fused numeric branch"),
    }
}

pub(crate) fn eval_number_compare_values(left: f64, op: BinaryOp, right: f64) -> bool {
    match op {
        BinaryOp::Equal => left == right,
        BinaryOp::NotEqual => left != right,
        BinaryOp::Less => left < right,
        BinaryOp::LessEqual => left <= right,
        BinaryOp::Greater => left > right,
        BinaryOp::GreaterEqual => left >= right,
        _ => unreachable!("non-comparison op in fused slot branch"),
    }
}

pub(crate) fn is_comparison_binary_op(op: BinaryOp) -> bool {
    matches!(
        op,
        BinaryOp::Equal
            | BinaryOp::NotEqual
            | BinaryOp::Less
            | BinaryOp::LessEqual
            | BinaryOp::Greater
            | BinaryOp::GreaterEqual
    )
}

pub(crate) fn is_numeric_binary_op(op: BinaryOp) -> bool {
    matches!(
        op,
        BinaryOp::Add
            | BinaryOp::Subtract
            | BinaryOp::Multiply
            | BinaryOp::Divide
            | BinaryOp::Modulo
    )
}

pub(crate) fn eval_compare_values(left: Value, op: BinaryOp, right: Value) -> Result<bool, RuntimeError> {
    match (left, right) {
        (Value::Number(left), Value::Number(right)) => {
            Ok(eval_number_compare_values(left, op, right))
        }
        (left, right) => match op {
            BinaryOp::Equal => Ok(left == right),
            BinaryOp::NotEqual => Ok(left != right),
            BinaryOp::Less => {
                compare_ordered(left, right, |a, b| a < b, |a, b| a < b).map(expect_bool_value)
            }
            BinaryOp::LessEqual => {
                compare_ordered(left, right, |a, b| a <= b, |a, b| a <= b).map(expect_bool_value)
            }
            BinaryOp::Greater => {
                compare_ordered(left, right, |a, b| a > b, |a, b| a > b).map(expect_bool_value)
            }
            BinaryOp::GreaterEqual => {
                compare_ordered(left, right, |a, b| a >= b, |a, b| a >= b).map(expect_bool_value)
            }
            _ => unreachable!("non-comparison op in fused branch"),
        },
    }
}

pub(crate) async fn eval_compare_values_async(
    left: Value,
    op: BinaryOp,
    right: Value,
) -> Result<bool, RuntimeError> {
    let has_projected = matches!(left, Value::Projected(_)) || matches!(right, Value::Projected(_));
    if has_projected {
        eval_compare_values(
            materialize_projected_async(left).await,
            op,
            materialize_projected_async(right).await,
        )
    } else {
        eval_compare_values(left, op, right)
    }
}

pub(crate) fn expect_bool_value(value: Value) -> bool {
    match value {
        Value::Bool(value) => value,
        _ => unreachable!("comparison produced non-bool value"),
    }
}

pub(crate) async fn eval_binary_values_async(
    left: Value,
    op: BinaryOp,
    right: Value,
) -> Result<Value, RuntimeError> {
    eval_binary_values(
        materialize_projected_async(left).await,
        op,
        materialize_projected_async(right).await,
    )
}

pub(crate) async fn materialize_projected_async(value: Value) -> Value {
    match value {
        Value::Projected(projected) => projected.materialize_async().await,
        other => other,
    }
}

pub(crate) fn numeric_binary_values(
    left: Value,
    right: Value,
    op: impl FnOnce(f64, f64) -> f64,
) -> Result<Value, RuntimeError> {
    match (left, right) {
        (Value::Number(left), Value::Number(right)) => Ok(Value::Number(op(left, right))),
        (left, right) => Ok(Value::Number(op(as_number(&left)?, as_number(&right)?))),
    }
}

pub(crate) fn as_number(value: &Value) -> Result<f64, RuntimeError> {
    match value {
        Value::Number(value) => Ok(*value),
        Value::Bool(value) => Ok(if *value { 1.0 } else { 0.0 }),
        Value::Null => Ok(0.0),
        Value::String(value) => {
            let value = value.trim();
            if value.is_empty() {
                return Ok(0.0);
            }
            value.parse::<f64>().map_err(|_| RuntimeError::TypeError {
                message: "expected a number".to_string(),
            })
        }
        _ => Err(RuntimeError::TypeError {
            message: format!("expected a number, got {}", value_type_name(value)),
        }),
    }
}

pub(crate) fn coerce_string(value: &Value) -> Result<Cow<'_, str>, RuntimeError> {
    match value {
        Value::String(value) => Ok(Cow::Borrowed(value)),
        Value::Null => Ok(Cow::Borrowed("null")),
        Value::Bool(value) => Ok(Cow::Owned(value.to_string())),
        Value::Number(value) => Ok(Cow::Owned(value.to_string())),
        Value::Image(_) | Value::List(_) | Value::Record(_) | Value::Projected(_) => {
            Err(RuntimeError::TypeError {
                message: format!("expected text, got {}", value_type_name(value)),
            })
        }
    }
}

pub(crate) fn as_offset(value: &Value) -> Result<isize, RuntimeError> {
    let number = as_number(value)?;
    if !number.is_finite() || number.fract() != 0.0 {
        return Err(RuntimeError::TypeError {
            message: "index must be an integer".to_string(),
        });
    }
    Ok(number as isize)
}

pub(crate) fn as_slice_bound(value: &Value) -> Result<Option<isize>, RuntimeError> {
    match value {
        Value::Null => Ok(None),
        other => as_offset(other).map(Some),
    }
}

pub(crate) fn compare_numbers(
    left: Value,
    right: Value,
    cmp: impl FnOnce(f64, f64) -> bool,
) -> Result<Value, RuntimeError> {
    match (left, right) {
        (Value::Number(left), Value::Number(right)) => Ok(Value::Bool(cmp(left, right))),
        (left, right) => Ok(Value::Bool(cmp(as_number(&left)?, as_number(&right)?))),
    }
}

pub(crate) fn compare_ordered(
    left: Value,
    right: Value,
    number_cmp: impl FnOnce(f64, f64) -> bool,
    string_cmp: impl FnOnce(&str, &str) -> bool,
) -> Result<Value, RuntimeError> {
    match (&left, &right) {
        (Value::String(left), Value::String(right)) => Ok(Value::Bool(string_cmp(left, right))),
        _ => compare_numbers(left, right, number_cmp),
    }
}

pub(crate) fn add_values(left: Value, right: Value) -> Result<Value, RuntimeError> {
    match (left, right) {
        (Value::Number(a), Value::Number(b)) => Ok(Value::Number(a + b)),
        (Value::String(a), Value::String(b)) => Ok(Value::String(a + &b)),
        (Value::String(mut a), other) => {
            a.push_str(&stringify_value_blocking(&other)?);
            Ok(Value::String(a))
        }
        (other, Value::String(b)) => {
            let mut text = stringify_value_blocking(&other)?;
            text.push_str(&b);
            Ok(Value::String(text.into()))
        }
        (Value::List(a), Value::List(b)) => {
            let mut values = Vec::with_capacity(a.len() + b.len());
            values.extend(a.iter().cloned());
            values.extend(b.iter().cloned());
            Ok(Value::List(values.into()))
        }
        (left, right) => Ok(Value::Number(as_number(&left)? + as_number(&right)?)),
    }
}

pub(crate) fn is_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(value) => *value,
        Value::Number(value) => *value != 0.0 && !value.is_nan(),
        Value::String(value) => !value.is_empty(),
        Value::Image(_) | Value::List(_) | Value::Record(_) => true,
        Value::Projected(value) => !futures_executor::block_on(value.is_empty()),
    }
}

pub(crate) async fn is_truthy_async(value: &Value) -> bool {
    match value {
        Value::Projected(value) => !value.is_empty().await,
        other => is_truthy(other),
    }
}

pub(crate) fn success(value: Value) -> Value {
    let result_names = result_wrapper_names();
    let mut record = record_with_capacity(2);
    record.insert_symbolized(
        result_names.ok.symbol,
        result_names.ok.text.clone(),
        Value::Bool(true),
    );
    record.insert_symbolized(
        result_names.value.symbol,
        result_names.value.text.clone(),
        value,
    );
    Value::Record(Arc::new(record))
}

pub(crate) fn error_value(message: String) -> Value {
    let result_names = result_wrapper_names();
    let mut record = record_with_capacity(2);
    record.insert_symbolized(
        result_names.ok.symbol,
        result_names.ok.text.clone(),
        Value::Bool(false),
    );
    record.insert_symbolized(
        result_names.error.symbol,
        result_names.error.text.clone(),
        Value::String(message.into()),
    );
    Value::Record(Arc::new(record))
}

pub(crate) fn value_type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "bool",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Image(_) => "image",
        Value::List(_) => "list",
        Value::Record(_) => "record",
        Value::Projected(value) => value.value_type_name(),
    }
}

pub(crate) fn value_contains_projected(value: &Value) -> bool {
    match value {
        Value::Projected(_) => true,
        Value::List(values) => values.iter().any(value_contains_projected),
        Value::Record(record) => record.values().any(value_contains_projected),
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) | Value::Image(_) => {
            false
        }
    }
}

pub(crate) fn materialize_value(value: Value) -> Value {
    match value {
        Value::Projected(projected) => projected.materialize(),
        other => other,
    }
}
