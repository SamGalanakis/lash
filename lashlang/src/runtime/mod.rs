use crate::ast::{BinaryOp, UnaryOp};
use crate::lexer::Span;
use smallvec::SmallVec;
use std::borrow::Cow;
use std::fmt;
use std::sync::Arc;
use thiserror::Error;

mod cache;
mod compiler;
mod entry_points;
mod host;
mod instruction;
mod record;
mod schema;
mod state;
mod value;
mod vm;

pub use cache::{CompiledProgramCache, CompiledProgramCacheStats};
pub use entry_points::{
    compile_program, compile_source, execute_compiled, execute_compiled_traced,
    execute_compiled_traced_with_projected_bindings, execute_compiled_traced_with_scratch,
    execute_compiled_traced_with_scratch_and_projected_bindings,
    execute_compiled_with_projected_bindings, execute_compiled_with_scratch,
    execute_compiled_with_scratch_and_projected_bindings, execute_program, prewarm,
    profile_compiled, profile_compiled_with_projected_bindings, profile_compiled_with_scratch,
    profile_compiled_with_scratch_and_projected_bindings,
};
pub use host::{ToolHost, ToolHostCall, ToolHostError};
use compiler::Compiler;
use instruction::{
    Builtin, Chunk, CompiledAssignPath, CompiledAssignPathStep, CompiledFormatPart,
    CompiledFormatTemplate, Instruction, InstructionProfileTag, Name, NamedBranchChunk,
    NamedParallelCallBranch, ParallelCallBranch, ProfileAccumulator, PureExpr, merge_stats,
    result_wrapper_names, transient_name,
};
pub use record::Record;
use record::record_with_capacity;
use schema::execute_validate_builtin;
pub use state::{Snapshot, State};
use vm::{IterState, SlotState, Vm};
pub use value::{
    ImageValue, LASH_TYPE_KEY, ProjectedBindingError, ProjectedBindings, ProjectedFuture,
    ProjectedHostValue, ProjectedRead, ProjectedValue, Value,
};

#[derive(Debug, Error, PartialEq)]
pub enum RuntimeError {
    #[error("unknown name `{name}`")]
    UndefinedVariable { name: String },
    #[error("`for` expects a list")]
    NonListIteration,
    #[error("`finish` can't be used inside `parallel`")]
    FinishInsideParallel,
    #[error("`parallel` assigns `{name}` more than once")]
    ParallelConflict { name: String },
    #[error("unknown builtin `{name}`")]
    UnknownBuiltin { name: String },
    #[error("{message}")]
    TypeError { message: String },
    #[error("{message}")]
    ValueError { message: String },
}

#[derive(Debug, Error, PartialEq)]
#[error("{error}")]
pub struct RuntimeFailure {
    pub error: RuntimeError,
    pub span: Option<Span>,
}

#[derive(Default)]
pub struct ExecutionScratch {
    stack: Vec<Value>,
    iter_stack: Vec<IterState>,
    slot_values: Vec<Option<Value>>,
}

impl ExecutionScratch {
    pub fn new() -> Self {
        Self::default()
    }
}

pub(crate) const COOPERATIVE_YIELD_INSTRUCTION_BUDGET: usize = 1024;

#[derive(Clone)]
pub struct CompiledProgram {
    pub(crate) chunk: Chunk,
    pub(crate) compile_stats: CompileStats,
}

impl CompiledProgram {
    pub fn compile_stats(&self) -> &CompileStats {
        &self.compile_stats
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum ExecutionOutcome {
    Continued,
    Finished(Value),
}

#[derive(Clone, Debug, Default)]
pub struct ProfileReport {
    instruction_stats: Vec<ProfileStat>,
    builtin_stats: Vec<ProfileStat>,
    compile_stats: CompileStats,
}

impl ProfileReport {
    pub fn instruction_stats(&self) -> &[ProfileStat] {
        &self.instruction_stats
    }

    pub fn builtin_stats(&self) -> &[ProfileStat] {
        &self.builtin_stats
    }

    pub fn compile_stats(&self) -> &CompileStats {
        &self.compile_stats
    }

    pub fn merge(&mut self, other: &Self) {
        merge_stats(&mut self.instruction_stats, &other.instruction_stats);
        merge_stats(&mut self.builtin_stats, &other.builtin_stats);
        self.compile_stats.merge(&other.compile_stats);
    }
}

/// Compile-time statistics captured when a program is compiled. Independent
/// of run-time profiling — these counts reflect the shape of the compiled
/// program itself (how many Type literals it contains, how many got
/// const-folded, etc.). Runtime cost of `Type` evaluation appears in the
/// instruction profile under `build_type_ref` / `build_record` / etc.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CompileStats {
    pub type_literals_total: u64,
    pub type_literals_const_folded: u64,
    pub type_literals_dynamic: u64,
    pub type_ref_sites: u64,
}

impl CompileStats {
    pub fn merge(&mut self, other: &Self) {
        self.type_literals_total += other.type_literals_total;
        self.type_literals_const_folded += other.type_literals_const_folded;
        self.type_literals_dynamic += other.type_literals_dynamic;
        self.type_ref_sites += other.type_ref_sites;
    }
}

#[derive(Clone, Debug, Default)]
pub struct ProfileStat {
    pub name: &'static str,
    pub count: u64,
    pub total_ns: u128,
}

impl ProfileStat {
    pub fn avg_ns(&self) -> u128 {
        if self.count == 0 {
            0
        } else {
            self.total_ns / self.count as u128
        }
    }
}




/// Unwrap a `Value::Record` that carries the `$lash_type` marker back into the
/// inner JSON-Schema value. Returns `None` when the value is not a wrapped
/// Type literal.
pub fn unwrap_type_value(value: &Value) -> Option<&Value> {
    let record = value.as_record()?;
    if record.len() != 1 {
        return None;
    }
    record.get(LASH_TYPE_KEY)
}

fn eval_pure_expr(
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

fn expect_arg_count(name: &str, values: &[Value], expected: usize) -> Result<(), RuntimeError> {
    if values.len() == expected {
        Ok(())
    } else {
        Err(RuntimeError::TypeError {
            message: format!("`{name}` takes {expected} arg(s), got {}", values.len()),
        })
    }
}

async fn execute_builtin(
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

fn execute_builtin_blocking(
    builtin: Builtin,
    names: &[Name],
    values: &[Value],
) -> Result<Value, RuntimeError> {
    futures_executor::block_on(execute_builtin(builtin, names, values))
}

async fn execute_len_builtin(value: &Value) -> Result<Value, RuntimeError> {
    if let Value::Projected(value) = value {
        return Ok(Value::Number(value.len().await as f64));
    }
    execute_len_direct(value)
}

fn execute_len_direct(value: &Value) -> Result<Value, RuntimeError> {
    value_len(value)
        .map(|len| Value::Number(len as f64))
        .ok_or_else(|| RuntimeError::TypeError {
            message: "`len` requires a string, list, record, or null; use `.size` for images"
                .to_string(),
        })
}

async fn execute_contains_builtin(haystack: &Value, needle: &Value) -> Result<Value, RuntimeError> {
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

fn execute_contains_direct(haystack: &Value, needle: &Value) -> Result<bool, RuntimeError> {
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

fn value_len(value: &Value) -> Option<usize> {
    match value {
        Value::String(value) => Some(value.chars().count()),
        Value::List(values) => Some(values.len()),
        Value::Record(record) => Some(record.len()),
        Value::Null => Some(0),
        _ => None,
    }
}

async fn iterable_values(value: Value) -> Result<Arc<[Value]>, RuntimeError> {
    match value {
        Value::List(values) => Ok(values),
        Value::Projected(value) => match value.materialize_async().await {
            Value::List(values) => Ok(values),
            _ => Err(RuntimeError::NonListIteration),
        },
        _ => Err(RuntimeError::NonListIteration),
    }
}

fn execute_join_builtin(items: &Value, sep: &Value) -> Result<Value, RuntimeError> {
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

fn execute_range_builtin(values: &[Value]) -> Result<Value, RuntimeError> {
    let (start, end) = range_bounds(values)?;
    build_range(start, end)
}

fn range_bounds(values: &[Value]) -> Result<(i64, i64), RuntimeError> {
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

fn execute_push_builtin(list: &Value, item: Value) -> Result<Value, RuntimeError> {
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

fn as_range_bound(value: &Value) -> Result<i64, RuntimeError> {
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

fn build_range(start: i64, end: i64) -> Result<Value, RuntimeError> {
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

fn validate_range_len(start: i64, end: i64) -> Result<(), RuntimeError> {
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

fn read_field_ref_direct(value: &Value, field: &Name) -> Result<Value, RuntimeError> {
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

fn unwrap_tool_result(value: Value) -> Result<Value, RuntimeError> {
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

fn is_async_handle_record(record: &Record) -> bool {
    record.get("__handle__").is_some() || record.get("handle").is_some()
}

async fn read_field(value: Value, field: &Name) -> Result<Value, RuntimeError> {
    match value {
        Value::Projected(value) => value.get_field(field).await,
        other => read_field_direct(other, field),
    }
}

fn read_field_direct(value: Value, field: &Name) -> Result<Value, RuntimeError> {
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

fn read_field_blocking(value: Value, field: &Name) -> Result<Value, RuntimeError> {
    futures_executor::block_on(read_field(value, field))
}

fn read_image_field(image: &ImageValue, field: &Name) -> Result<Value, RuntimeError> {
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

async fn read_index(target: Value, index: Value) -> Result<Value, RuntimeError> {
    match target {
        Value::Projected(value) => value.get_index(&index).await,
        other => read_index_direct(other, index),
    }
}

fn read_index_direct(target: Value, index: Value) -> Result<Value, RuntimeError> {
    read_index_ref_direct(&target, &index)
}

fn read_index_ref_direct(target: &Value, index: &Value) -> Result<Value, RuntimeError> {
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
                .map(|ch| Value::String(ch.to_string().into()))
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

fn read_index_blocking(target: Value, index: Value) -> Result<Value, RuntimeError> {
    futures_executor::block_on(read_index(target, index))
}

fn assign_path(
    root: &mut Value,
    path: &CompiledAssignPath,
    indexes: &[Value],
    value: Value,
    names: &[Name],
) -> Result<(), RuntimeError> {
    let mut index_cursor = 0;
    assign_path_steps(root, &path.steps, indexes, &mut index_cursor, value, names)
}

fn assign_path_steps(
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

fn next_assign_index<'a>(
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

fn assign_record_field(target: &mut Value, field: &Name, value: Value) -> Result<(), RuntimeError> {
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

fn descend_record_field<'a>(
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

fn assign_index(target: &mut Value, index: &Value, value: Value) -> Result<(), RuntimeError> {
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

fn add_assign_index_number(
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

fn add_assign_value_number(value: &mut Value, right: f64) -> Result<Value, RuntimeError> {
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

fn descend_index<'a>(target: &'a mut Value, index: &Value) -> Result<&'a mut Value, RuntimeError> {
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

fn eval_binary_values(left: Value, op: BinaryOp, right: Value) -> Result<Value, RuntimeError> {
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

fn eval_number_binary_values(left: f64, op: BinaryOp, right: f64) -> Value {
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

fn eval_number_numeric_binary_value(left: f64, op: BinaryOp, right: f64) -> f64 {
    match op {
        BinaryOp::Add => left + right,
        BinaryOp::Subtract => left - right,
        BinaryOp::Multiply => left * right,
        BinaryOp::Divide => left / right,
        BinaryOp::Modulo => left % right,
        _ => unreachable!("non-numeric op in fused numeric branch"),
    }
}

fn eval_number_compare_values(left: f64, op: BinaryOp, right: f64) -> bool {
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

fn is_comparison_binary_op(op: BinaryOp) -> bool {
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

fn is_numeric_binary_op(op: BinaryOp) -> bool {
    matches!(
        op,
        BinaryOp::Add
            | BinaryOp::Subtract
            | BinaryOp::Multiply
            | BinaryOp::Divide
            | BinaryOp::Modulo
    )
}

fn eval_compare_values(left: Value, op: BinaryOp, right: Value) -> Result<bool, RuntimeError> {
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

async fn eval_compare_values_async(
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

fn expect_bool_value(value: Value) -> bool {
    match value {
        Value::Bool(value) => value,
        _ => unreachable!("comparison produced non-bool value"),
    }
}

async fn eval_binary_values_async(
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

async fn materialize_projected_async(value: Value) -> Value {
    match value {
        Value::Projected(projected) => projected.materialize_async().await,
        other => other,
    }
}

fn numeric_binary_values(
    left: Value,
    right: Value,
    op: impl FnOnce(f64, f64) -> f64,
) -> Result<Value, RuntimeError> {
    match (left, right) {
        (Value::Number(left), Value::Number(right)) => Ok(Value::Number(op(left, right))),
        (left, right) => Ok(Value::Number(op(as_number(&left)?, as_number(&right)?))),
    }
}

fn as_number(value: &Value) -> Result<f64, RuntimeError> {
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

fn coerce_string(value: &Value) -> Result<Cow<'_, str>, RuntimeError> {
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

fn as_offset(value: &Value) -> Result<isize, RuntimeError> {
    let number = as_number(value)?;
    if !number.is_finite() || number.fract() != 0.0 {
        return Err(RuntimeError::TypeError {
            message: "index must be an integer".to_string(),
        });
    }
    Ok(number as isize)
}

fn as_slice_bound(value: &Value) -> Result<Option<isize>, RuntimeError> {
    match value {
        Value::Null => Ok(None),
        other => as_offset(other).map(Some),
    }
}

fn compare_numbers(
    left: Value,
    right: Value,
    cmp: impl FnOnce(f64, f64) -> bool,
) -> Result<Value, RuntimeError> {
    match (left, right) {
        (Value::Number(left), Value::Number(right)) => Ok(Value::Bool(cmp(left, right))),
        (left, right) => Ok(Value::Bool(cmp(as_number(&left)?, as_number(&right)?))),
    }
}

fn compare_ordered(
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

fn add_values(left: Value, right: Value) -> Result<Value, RuntimeError> {
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

fn is_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(value) => *value,
        Value::Number(value) => *value != 0.0 && !value.is_nan(),
        Value::String(value) => !value.is_empty(),
        Value::Image(_) | Value::List(_) | Value::Record(_) => true,
        Value::Projected(value) => !futures_executor::block_on(value.is_empty()),
    }
}

async fn is_truthy_async(value: &Value) -> bool {
    match value {
        Value::Projected(value) => !value.is_empty().await,
        other => is_truthy(other),
    }
}

fn success(value: Value) -> Value {
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

fn error_value(message: String) -> Value {
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

async fn stringify_value_async(value: &Value) -> Result<String, RuntimeError> {
    let mut output = String::new();
    append_stringified_value_async(&mut output, value).await?;
    Ok(output)
}

fn stringify_value(value: &Value) -> Result<String, RuntimeError> {
    if value_contains_projected(value) {
        futures_executor::block_on(stringify_value_async(value))
    } else {
        stringify_value_direct(value)
    }
}

fn stringify_value_blocking(value: &Value) -> Result<String, RuntimeError> {
    stringify_value(value)
}

fn stringify_value_direct(value: &Value) -> Result<String, RuntimeError> {
    let mut output = String::new();
    append_stringified_value_direct(&mut output, value)?;
    Ok(output)
}

fn append_stringified_value_direct(output: &mut String, value: &Value) -> Result<(), RuntimeError> {
    match value {
        Value::String(value) => output.push_str(value),
        Value::Null => output.push_str("null"),
        Value::Bool(value) => output.push_str(if *value { "true" } else { "false" }),
        Value::Number(value) => {
            write_number(output, *value).expect("string writes should not fail")
        }
        Value::Projected(_) => unreachable!("projected values require async stringification"),
        Value::Image(_) | Value::List(_) | Value::Record(_) => output.push_str(
            &serde_json::to_string(&to_json_direct(value))
                .expect("value json serialization should succeed"),
        ),
    }
    Ok(())
}

fn append_stringified_value_async<'a>(
    output: &'a mut String,
    value: &'a Value,
) -> ProjectedFuture<'a, Result<(), RuntimeError>> {
    Box::pin(async move {
        match value {
            Value::String(value) => output.push_str(value),
            Value::Null => output.push_str("null"),
            Value::Bool(value) => output.push_str(if *value { "true" } else { "false" }),
            Value::Number(value) => {
                write_number(output, *value).expect("string writes should not fail")
            }
            Value::Projected(value) => output.push_str(&value.render().await),
            Value::Image(_) | Value::List(_) | Value::Record(_) => output.push_str(
                &serde_json::to_string(&to_json_async(value).await)
                    .expect("value json serialization should succeed"),
            ),
        }
        Ok(())
    })
}

#[cfg(test)]
fn append_stringified_value(output: &mut String, value: &Value) -> Result<(), RuntimeError> {
    futures_executor::block_on(append_stringified_value_async(output, value))
}

fn value_type_name(value: &Value) -> &'static str {
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

fn value_contains_projected(value: &Value) -> bool {
    match value {
        Value::Projected(_) => true,
        Value::List(values) => values.iter().any(value_contains_projected),
        Value::Record(record) => record.values().any(value_contains_projected),
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) | Value::Image(_) => {
            false
        }
    }
}

fn materialize_value(value: Value) -> Value {
    match value {
        Value::Projected(projected) => projected.materialize(),
        other => other,
    }
}

fn write_number(output: &mut impl fmt::Write, value: f64) -> fmt::Result {
    if value.is_finite() && value.fract() == 0.0 {
        let as_i64 = value as i64 as f64;
        if as_i64 == value {
            return write!(output, "{}", value as i64);
        }
        let as_u64 = value as u64 as f64;
        if as_u64 == value {
            return write!(output, "{}", value as u64);
        }
    }
    write!(output, "{value}")
}

fn resolve_index(index: &Value, len: usize) -> Result<Option<usize>, RuntimeError> {
    let index = as_offset(index)?;
    let len = len as isize;
    let normalized = if index < 0 { len + index } else { index };
    if normalized < 0 || normalized >= len {
        return Ok(None);
    }
    Ok(Some(normalized as usize))
}

fn resolve_existing_list_assignment_index(
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

async fn apply_format_async(template: &str, args: &[Value]) -> Result<String, RuntimeError> {
    let mut output = String::with_capacity(template.len());
    let bytes = template.as_bytes();
    let mut index = 0;
    let mut last_literal = 0;
    let mut uses_sequential = false;
    let mut uses_indexed = false;
    let mut next_sequential = 0usize;
    let mut used_indexed_bits = 0u64;
    let mut used_indexed_args: Option<Vec<bool>> = None;
    while index < bytes.len() {
        match bytes[index] {
            b'{' if bytes.get(index + 1) == Some(&b'{') => {
                output.push_str(&template[last_literal..index]);
                output.push('{');
                index += 2;
                last_literal = index;
                continue;
            }
            b'{' => {
                output.push_str(&template[last_literal..index]);
                let mut cursor = index + 1;
                while cursor < bytes.len() && bytes[cursor].is_ascii_digit() {
                    cursor += 1;
                }

                let (slot, slot_text) = if cursor >= bytes.len() {
                    return Err(RuntimeError::ValueError {
                        message: "unmatched `{` in format string".to_string(),
                    });
                } else if bytes[cursor] != b'}' {
                    return Err(RuntimeError::ValueError {
                        message: "invalid format placeholder".to_string(),
                    });
                } else if cursor == index + 1 {
                    if uses_indexed {
                        return Err(RuntimeError::ValueError {
                            message: "can't mix `{}` and indexed format placeholders".to_string(),
                        });
                    }
                    uses_sequential = true;
                    let slot = next_sequential;
                    next_sequential += 1;
                    (slot, None)
                } else {
                    if uses_sequential {
                        return Err(RuntimeError::ValueError {
                            message: "can't mix `{}` and indexed format placeholders".to_string(),
                        });
                    }
                    uses_indexed = true;
                    let digits = &template[index + 1..cursor];
                    let slot = digits
                        .parse::<usize>()
                        .map_err(|_| RuntimeError::ValueError {
                            message: format!("bad format slot `{digits}`"),
                        })?;
                    if slot < args.len() {
                        if args.len() <= u64::BITS as usize {
                            used_indexed_bits |= 1u64 << slot;
                        } else {
                            let used_args =
                                used_indexed_args.get_or_insert_with(|| vec![false; args.len()]);
                            used_args[slot] = true;
                        }
                    }
                    (slot, Some(digits))
                };

                let value = args.get(slot).ok_or_else(|| RuntimeError::ValueError {
                    message: match slot_text {
                        Some(slot_text) => format!("format slot `{slot_text}` is out of range"),
                        None => "format slot `{}` is out of range".to_string(),
                    },
                })?;
                append_stringified_value_async(&mut output, value).await?;
                index = cursor + 1;
                last_literal = index;
                continue;
            }
            b'}' if bytes.get(index + 1) == Some(&b'}') => {
                output.push_str(&template[last_literal..index]);
                output.push('}');
                index += 2;
                last_literal = index;
                continue;
            }
            b'}' => {
                return Err(RuntimeError::ValueError {
                    message: "unmatched `}` in format string".to_string(),
                });
            }
            _ => {}
        }
        index += 1;
    }
    output.push_str(&template[last_literal..]);
    let unused_index = if uses_sequential {
        (next_sequential < args.len()).then_some(next_sequential)
    } else if uses_indexed {
        if args.len() <= u64::BITS as usize {
            (0..args.len()).find(|slot| used_indexed_bits & (1u64 << slot) == 0)
        } else {
            used_indexed_args
                .as_ref()
                .and_then(|used_args| used_args.iter().position(|used| !*used))
        }
    } else {
        (!args.is_empty()).then_some(0)
    };
    if let Some(unused_index) = unused_index {
        return Err(RuntimeError::ValueError {
            message: format!("format argument `{unused_index}` is unused"),
        });
    }
    Ok(output)
}

#[cfg(test)]
fn apply_format(template: &str, args: &[Value]) -> Result<String, RuntimeError> {
    futures_executor::block_on(apply_format_async(template, args))
}

fn compile_format_template(template: &str, argc: usize) -> CompiledFormatTemplate {
    match parse_format_template(template, argc) {
        Ok(parts) => CompiledFormatTemplate {
            parts: parts.into_boxed_slice(),
            argc,
            min_capacity: template.len(),
            error: None,
        },
        Err(message) => CompiledFormatTemplate {
            parts: Box::new([]),
            argc,
            min_capacity: template.len(),
            error: Some(message),
        },
    }
}

fn parse_format_template(template: &str, argc: usize) -> Result<Vec<CompiledFormatPart>, String> {
    let mut parts = Vec::new();
    let bytes = template.as_bytes();
    let mut index = 0;
    let mut last_literal = 0;
    let mut uses_sequential = false;
    let mut uses_indexed = false;
    let mut next_sequential = 0usize;
    let mut used_indexed_bits = 0u64;
    let mut used_indexed_args: Option<Vec<bool>> = None;

    while index < bytes.len() {
        match bytes[index] {
            b'{' if bytes.get(index + 1) == Some(&b'{') => {
                push_format_literal(&mut parts, &template[last_literal..index]);
                push_format_literal(&mut parts, "{");
                index += 2;
                last_literal = index;
                continue;
            }
            b'{' => {
                push_format_literal(&mut parts, &template[last_literal..index]);
                let mut cursor = index + 1;
                while cursor < bytes.len() && bytes[cursor].is_ascii_digit() {
                    cursor += 1;
                }

                let (slot, slot_text) = if cursor >= bytes.len() {
                    return Err("unmatched `{` in format string".to_string());
                } else if bytes[cursor] != b'}' {
                    return Err("invalid format placeholder".to_string());
                } else if cursor == index + 1 {
                    if uses_indexed {
                        return Err("can't mix `{}` and indexed format placeholders".to_string());
                    }
                    uses_sequential = true;
                    let slot = next_sequential;
                    next_sequential += 1;
                    (slot, None)
                } else {
                    if uses_sequential {
                        return Err("can't mix `{}` and indexed format placeholders".to_string());
                    }
                    uses_indexed = true;
                    let digits = &template[index + 1..cursor];
                    let slot = digits
                        .parse::<usize>()
                        .map_err(|_| format!("bad format slot `{digits}`"))?;
                    if slot < argc {
                        if argc <= u64::BITS as usize {
                            used_indexed_bits |= 1u64 << slot;
                        } else {
                            let used_args =
                                used_indexed_args.get_or_insert_with(|| vec![false; argc]);
                            used_args[slot] = true;
                        }
                    }
                    (slot, Some(digits))
                };

                if slot >= argc {
                    return Err(match slot_text {
                        Some(slot_text) => format!("format slot `{slot_text}` is out of range"),
                        None => "format slot `{}` is out of range".to_string(),
                    });
                }
                parts.push(CompiledFormatPart::Arg(slot));
                index = cursor + 1;
                last_literal = index;
                continue;
            }
            b'}' if bytes.get(index + 1) == Some(&b'}') => {
                push_format_literal(&mut parts, &template[last_literal..index]);
                push_format_literal(&mut parts, "}");
                index += 2;
                last_literal = index;
                continue;
            }
            b'}' => {
                return Err("unmatched `}` in format string".to_string());
            }
            _ => {}
        }
        index += 1;
    }

    push_format_literal(&mut parts, &template[last_literal..]);
    let unused_index = if uses_sequential {
        (next_sequential < argc).then_some(next_sequential)
    } else if uses_indexed {
        if argc <= u64::BITS as usize {
            (0..argc).find(|slot| used_indexed_bits & (1u64 << slot) == 0)
        } else {
            used_indexed_args
                .as_ref()
                .and_then(|used_args| used_args.iter().position(|used| !*used))
        }
    } else {
        (argc > 0).then_some(0)
    };
    if let Some(unused_index) = unused_index {
        return Err(format!("format argument `{unused_index}` is unused"));
    }
    Ok(parts)
}

fn push_format_literal(parts: &mut Vec<CompiledFormatPart>, literal: &str) {
    if !literal.is_empty() {
        parts.push(CompiledFormatPart::Literal(Arc::<str>::from(literal)));
    }
}

async fn execute_compiled_format(
    template: &CompiledFormatTemplate,
    args: &[Value],
) -> Result<String, RuntimeError> {
    if let Some(message) = &template.error {
        return Err(RuntimeError::ValueError {
            message: message.clone(),
        });
    }

    let mut output = String::with_capacity(template.min_capacity);
    for part in template.parts.iter() {
        match part {
            CompiledFormatPart::Literal(literal) => output.push_str(literal),
            CompiledFormatPart::Arg(slot) => {
                let value = args.get(*slot).ok_or_else(|| RuntimeError::ValueError {
                    message: format!("format slot `{slot}` is out of range"),
                })?;
                append_stringified_value_async(&mut output, value).await?;
            }
        }
    }
    Ok(output)
}

fn execute_compiled_format_blocking(
    template: &CompiledFormatTemplate,
    args: &[Value],
) -> Result<String, RuntimeError> {
    if args
        .iter()
        .any(|value| matches!(value, Value::Projected(_)))
    {
        futures_executor::block_on(execute_compiled_format(template, args))
    } else {
        execute_compiled_format_direct(template, args)
    }
}

fn execute_compiled_format_direct(
    template: &CompiledFormatTemplate,
    args: &[Value],
) -> Result<String, RuntimeError> {
    if let Some(message) = &template.error {
        return Err(RuntimeError::ValueError {
            message: message.clone(),
        });
    }

    let mut output = String::with_capacity(template.min_capacity);
    for part in template.parts.iter() {
        match part {
            CompiledFormatPart::Literal(literal) => output.push_str(literal),
            CompiledFormatPart::Arg(slot) => {
                let value = args.get(*slot).ok_or_else(|| RuntimeError::ValueError {
                    message: format!("format slot `{slot}` is out of range"),
                })?;
                append_stringified_value_direct(&mut output, value)?;
            }
        }
    }
    Ok(output)
}

fn clamp_slice_bounds(
    start: Option<isize>,
    end: Option<isize>,
    len: usize,
) -> Option<(usize, usize)> {
    let len = len as isize;
    let start = normalize_slice_bound(start.unwrap_or(0), len);
    let end = normalize_slice_bound(end.unwrap_or(len), len);
    (start < end).then_some((start, end))
}

fn normalize_slice_bound(bound: isize, len: isize) -> usize {
    let normalized = if bound < 0 { len + bound } else { bound };
    normalized.clamp(0, len) as usize
}

fn slice_string(value: &str, start: Option<isize>, end: Option<isize>) -> String {
    let char_count = value.chars().count();
    let Some((start, end)) = clamp_slice_bounds(start, end, char_count) else {
        return String::new();
    };

    let mut indices = value
        .char_indices()
        .map(|(idx, _)| idx)
        .chain(std::iter::once(value.len()));
    let Some(byte_start) = indices.nth(start) else {
        return String::new();
    };
    let Some(byte_end) = indices.nth(end - start - 1) else {
        return value[byte_start..].to_string();
    };
    value[byte_start..byte_end].to_string()
}

fn json_number(value: f64) -> Option<serde_json::Number> {
    if !value.is_finite() {
        return None;
    }
    if value.is_finite() && value.fract() == 0.0 {
        let as_i64 = value as i64 as f64;
        if as_i64 == value {
            return Some(serde_json::Number::from(value as i64));
        }
        let as_u64 = value as u64 as f64;
        if as_u64 == value {
            return Some(serde_json::Number::from(value as u64));
        }
    }
    serde_json::Number::from_f64(value)
}

fn to_json_async<'a>(value: &'a Value) -> ProjectedFuture<'a, serde_json::Value> {
    Box::pin(async move {
        match value {
            Value::Null => serde_json::Value::Null,
            Value::Bool(value) => serde_json::Value::Bool(*value),
            Value::Number(value) => json_number(*value)
                .map(serde_json::Value::Number)
                .unwrap_or(serde_json::Value::Null),
            Value::String(value) => serde_json::Value::String(value.to_string()),
            Value::Image(image) => image_to_json(image),
            Value::List(values) => {
                let mut out = Vec::with_capacity(values.len());
                for value in values.iter() {
                    out.push(to_json_async(value).await);
                }
                serde_json::Value::Array(out)
            }
            Value::Record(record) => {
                let mut object = serde_json::Map::with_capacity(record.len());
                for (key, value) in record.iter() {
                    object.insert(key.to_string(), to_json_async(value).await);
                }
                serde_json::Value::Object(object)
            }
            Value::Projected(value) => to_json_async(&value.materialize_async().await).await,
        }
    })
}

fn to_json(value: &Value) -> serde_json::Value {
    if value_contains_projected(value) {
        futures_executor::block_on(to_json_async(value))
    } else {
        to_json_direct(value)
    }
}

fn to_json_blocking(value: &Value) -> serde_json::Value {
    to_json(value)
}

fn to_json_direct(value: &Value) -> serde_json::Value {
    match value {
        Value::Null => serde_json::Value::Null,
        Value::Bool(value) => serde_json::Value::Bool(*value),
        Value::Number(value) => json_number(*value)
            .map(serde_json::Value::Number)
            .unwrap_or(serde_json::Value::Null),
        Value::String(value) => serde_json::Value::String(value.to_string()),
        Value::Image(image) => image_to_json(image),
        Value::List(values) => {
            serde_json::Value::Array(values.iter().map(to_json_direct).collect())
        }
        Value::Record(record) => {
            let mut object = serde_json::Map::with_capacity(record.len());
            for (key, value) in record.iter() {
                object.insert(key.to_string(), to_json_direct(value));
            }
            serde_json::Value::Object(object)
        }
        Value::Projected(_) => unreachable!("projected values require async json conversion"),
    }
}

fn image_to_json(image: &ImageValue) -> serde_json::Value {
    let mut object = serde_json::Map::with_capacity(7);
    object.insert(
        "type".to_string(),
        serde_json::Value::String("image".to_string()),
    );
    object.insert(
        "id".to_string(),
        serde_json::Value::String(image.id.clone()),
    );
    object.insert(
        "label".to_string(),
        serde_json::Value::String(image.label.clone()),
    );
    object.insert(
        "size".to_string(),
        serde_json::Value::Number(serde_json::Number::from(image.size)),
    );
    object.insert(
        "width".to_string(),
        image
            .width
            .map(|width| serde_json::Value::Number(serde_json::Number::from(width)))
            .unwrap_or(serde_json::Value::Null),
    );
    object.insert(
        "height".to_string(),
        image
            .height
            .map(|height| serde_json::Value::Number(serde_json::Number::from(height)))
            .unwrap_or(serde_json::Value::Null),
    );
    serde_json::Value::Object(object)
}

pub fn from_json(value: serde_json::Value) -> Value {
    match value {
        serde_json::Value::Null => Value::Null,
        serde_json::Value::Bool(value) => Value::Bool(value),
        serde_json::Value::Number(value) => Value::Number(value.as_f64().unwrap_or_default()),
        serde_json::Value::String(value) => Value::String(value.into()),
        serde_json::Value::Array(values) => {
            Value::List(values.into_iter().map(from_json).collect::<Vec<_>>().into())
        }
        serde_json::Value::Object(map) => image_from_json_map(&map)
            .map(Value::Image)
            .unwrap_or_else(|| {
                Value::Record(Arc::new(
                    map.into_iter()
                        .map(|(key, value)| (key, from_json(value)))
                        .collect(),
                ))
            }),
    }
}

fn image_from_json_map(map: &serde_json::Map<String, serde_json::Value>) -> Option<ImageValue> {
    if map.get("type")?.as_str()? != "image" {
        return None;
    }
    Some(ImageValue {
        id: map.get("id")?.as_str()?.to_string(),
        label: map.get("label")?.as_str()?.to_string(),
        size: map.get("size")?.as_u64()?,
        width: optional_u32_field(map.get("width")?)?,
        height: optional_u32_field(map.get("height")?)?,
    })
}

fn optional_u32_field(value: &serde_json::Value) -> Option<Option<u32>> {
    match value {
        serde_json::Value::Null => Some(None),
        serde_json::Value::Number(number) => number
            .as_u64()
            .and_then(|value| u32::try_from(value).ok())
            .map(Some),
        _ => None,
    }
}

#[cfg(test)]
mod tests;
