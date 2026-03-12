use crate::ast::{BinaryOp, CallExpr, Expr, Program, Stmt, UnaryOp};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt;
use thiserror::Error;

pub type Record = HashMap<String, Value>;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Value {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    List(Vec<Value>),
    Record(Record),
}

impl Value {
    pub fn as_record(&self) -> Option<&Record> {
        match self {
            Self::Record(record) => Some(record),
            _ => None,
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Null => write!(f, "null"),
            Self::Bool(value) => write!(f, "{value}"),
            Self::Number(value) => write!(f, "{value}"),
            Self::String(value) => write!(f, "{value}"),
            Self::List(values) => {
                write!(f, "{}", serde_json::to_string(values).unwrap_or_default())
            }
            Self::Record(record) => {
                write!(f, "{}", serde_json::to_string(record).unwrap_or_default())
            }
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct State {
    globals: Record,
}

impl State {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn globals(&self) -> &Record {
        &self.globals
    }

    pub fn snapshot(&self) -> Snapshot {
        Snapshot {
            globals: self.globals.clone(),
        }
    }

    pub fn from_snapshot(snapshot: Snapshot) -> Self {
        Self {
            globals: snapshot.globals,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Snapshot {
    pub globals: Record,
}

pub trait ToolHost: Sync {
    fn call(&self, name: &str, args: &Record) -> Result<Value, ToolHostError>;
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[error("{message}")]
pub struct ToolHostError {
    message: String,
}

impl ToolHostError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

#[derive(Debug, Error, PartialEq)]
pub enum RuntimeError {
    #[error("program finished without `finish`")]
    MissingFinish,
    #[error("undefined variable `{name}`")]
    UndefinedVariable { name: String },
    #[error("condition must evaluate to a boolean")]
    NonBooleanCondition,
    #[error("for-loop iterable must be a list")]
    NonListIteration,
    #[error("`finish` is not allowed inside a parallel branch")]
    FinishInsideParallel,
    #[error("parallel branch conflict on variable `{name}`")]
    ParallelConflict { name: String },
    #[error("unknown builtin `{name}`")]
    UnknownBuiltin { name: String },
    #[error("{message}")]
    TypeError { message: String },
    #[error("{message}")]
    ValueError { message: String },
}

pub fn execute_program<H: ToolHost>(
    program: &Program,
    state: &mut State,
    host: &H,
) -> Result<Value, RuntimeError> {
    let mut executor = Executor::new(std::mem::take(&mut state.globals), host, false);
    let result = executor.execute_program(program);
    state.globals = executor.into_globals();
    result
}

struct Executor<'a, H> {
    scopes: ScopeStack,
    host: &'a H,
    in_parallel_branch: bool,
    assigned: HashSet<String>,
}

impl<'a, H: ToolHost> Executor<'a, H> {
    fn new(globals: Record, host: &'a H, in_parallel_branch: bool) -> Self {
        Self {
            scopes: ScopeStack::new(globals),
            host,
            in_parallel_branch,
            assigned: HashSet::new(),
        }
    }

    fn execute_program(&mut self, program: &Program) -> Result<Value, RuntimeError> {
        match self.exec_block(&program.statements)? {
            Flow::Continue => Err(RuntimeError::MissingFinish),
            Flow::Finished(value) => Ok(value),
        }
    }

    fn exec_block(&mut self, statements: &[Stmt]) -> Result<Flow, RuntimeError> {
        for statement in statements {
            match self.exec_stmt(statement)? {
                Flow::Continue => {}
                Flow::Finished(value) => return Ok(Flow::Finished(value)),
            }
        }
        Ok(Flow::Continue)
    }

    fn exec_stmt(&mut self, statement: &Stmt) -> Result<Flow, RuntimeError> {
        match statement {
            Stmt::Assign { name, expr } => {
                let value = self.eval_expr(expr)?;
                self.scopes.assign(name.clone(), value);
                self.assigned.insert(name.clone());
                Ok(Flow::Continue)
            }
            Stmt::Call(call) => {
                let _ = self.eval_tool_call(call)?;
                Ok(Flow::Continue)
            }
            Stmt::If {
                condition,
                then_block,
                else_block,
            } => {
                if self.eval_condition(condition)? {
                    self.exec_block(then_block)
                } else {
                    self.exec_block(else_block)
                }
            }
            Stmt::For {
                binding,
                iterable,
                body,
            } => {
                let values = match self.eval_expr(iterable)? {
                    Value::List(values) => values,
                    _ => return Err(RuntimeError::NonListIteration),
                };
                for item in values {
                    let restore = self.scopes.bind_temporary(binding, item);
                    match self.exec_block(body)? {
                        Flow::Continue => self.scopes.restore_temporary(binding, restore),
                        Flow::Finished(value) => {
                            self.scopes.restore_temporary(binding, restore);
                            return Ok(Flow::Finished(value));
                        }
                    }
                }
                Ok(Flow::Continue)
            }
            Stmt::Parallel { branches } => {
                self.exec_parallel(branches)?;
                Ok(Flow::Continue)
            }
            Stmt::Finish(expr) => {
                if self.in_parallel_branch {
                    return Err(RuntimeError::FinishInsideParallel);
                }
                Ok(Flow::Finished(self.eval_expr(expr)?))
            }
        }
    }

    fn exec_parallel(&mut self, branches: &[Stmt]) -> Result<(), RuntimeError> {
        let base_scopes = self.scopes.clone();
        let results = std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(branches.len());
            for branch in branches {
                let branch = branch.clone();
                let branch_scopes = base_scopes.clone();
                let host = self.host;
                handles.push(scope.spawn(move || {
                    let mut executor = Executor {
                        scopes: branch_scopes,
                        host,
                        in_parallel_branch: true,
                        assigned: HashSet::new(),
                    };
                    executor.exec_stmt(&branch)?;
                    Ok(executor.into_branch_result())
                }));
            }

            let mut joined = Vec::with_capacity(handles.len());
            for handle in handles {
                joined.push(handle.join().expect("parallel branch panicked")?);
            }
            Ok::<_, RuntimeError>(joined)
        })?;

        let mut merged = HashMap::new();
        for result in results {
            for (name, value) in result.values {
                if merged.insert(name.clone(), value).is_some() {
                    return Err(RuntimeError::ParallelConflict { name });
                }
            }
        }

        for (name, value) in merged {
            self.scopes.assign(name.clone(), value);
            self.assigned.insert(name);
        }
        Ok(())
    }

    fn eval_condition(&mut self, expr: &Expr) -> Result<bool, RuntimeError> {
        match self.eval_expr(expr)? {
            Value::Bool(value) => Ok(value),
            _ => Err(RuntimeError::NonBooleanCondition),
        }
    }

    fn eval_expr(&mut self, expr: &Expr) -> Result<Value, RuntimeError> {
        match expr {
            Expr::Null => Ok(Value::Null),
            Expr::Bool(value) => Ok(Value::Bool(*value)),
            Expr::Number(value) => Ok(Value::Number(*value)),
            Expr::String(value) => Ok(Value::String(value.clone())),
            Expr::Variable(name) => self
                .scopes
                .get(name)
                .cloned()
                .ok_or_else(|| RuntimeError::UndefinedVariable { name: name.clone() }),
            Expr::List(items) => {
                let mut values = Vec::with_capacity(items.len());
                for item in items {
                    values.push(self.eval_expr(item)?);
                }
                Ok(Value::List(values))
            }
            Expr::Record(entries) => {
                let mut record = Record::with_capacity(entries.len());
                for (key, value) in entries {
                    record.insert(key.clone(), self.eval_expr(value)?);
                }
                Ok(Value::Record(record))
            }
            Expr::ToolCall(call) => self.eval_tool_call(call),
            Expr::BuiltinCall { name, args } => self.eval_builtin(name, args),
            Expr::Field { target, field } => {
                let value = self.eval_expr(target)?;
                let Value::Record(record) = value else {
                    return Err(RuntimeError::TypeError {
                        message: "field access requires a record".to_string(),
                    });
                };
                record
                    .get(field)
                    .cloned()
                    .ok_or_else(|| RuntimeError::ValueError {
                        message: format!("record does not contain field `{field}`"),
                    })
            }
            Expr::Index { target, index } => {
                let target = self.eval_expr(target)?;
                let index = self.eval_expr(index)?;
                let idx = as_index(&index)?;
                match target {
                    Value::List(values) => {
                        values
                            .get(idx)
                            .cloned()
                            .ok_or_else(|| RuntimeError::ValueError {
                                message: format!("list index {idx} is out of bounds"),
                            })
                    }
                    Value::String(value) => value
                        .chars()
                        .nth(idx)
                        .map(|ch| Value::String(ch.to_string()))
                        .ok_or_else(|| RuntimeError::ValueError {
                            message: format!("string index {idx} is out of bounds"),
                        }),
                    _ => Err(RuntimeError::TypeError {
                        message: "index access requires a list or string".to_string(),
                    }),
                }
            }
            Expr::Unary { op, expr } => {
                let value = self.eval_expr(expr)?;
                match op {
                    UnaryOp::Negate => Ok(Value::Number(-as_number(&value)?)),
                    UnaryOp::Not => match value {
                        Value::Bool(value) => Ok(Value::Bool(!value)),
                        _ => Err(RuntimeError::TypeError {
                            message: "`not` requires a boolean".to_string(),
                        }),
                    },
                }
            }
            Expr::Binary { left, op, right } => self.eval_binary(left, *op, right),
        }
    }

    fn eval_binary(
        &mut self,
        left: &Expr,
        op: BinaryOp,
        right: &Expr,
    ) -> Result<Value, RuntimeError> {
        match op {
            BinaryOp::And => {
                let lhs = self.eval_condition(left)?;
                if !lhs {
                    return Ok(Value::Bool(false));
                }
                return Ok(Value::Bool(self.eval_condition(right)?));
            }
            BinaryOp::Or => {
                let lhs = self.eval_condition(left)?;
                if lhs {
                    return Ok(Value::Bool(true));
                }
                return Ok(Value::Bool(self.eval_condition(right)?));
            }
            BinaryOp::Add => {
                let lhs = self.eval_expr(left)?;
                let rhs = self.eval_expr(right)?;
                add_values(lhs, rhs)
            }
            BinaryOp::Subtract => {
                let lhs = self.eval_expr(left)?;
                let rhs = self.eval_expr(right)?;
                Ok(Value::Number(as_number(&lhs)? - as_number(&rhs)?))
            }
            BinaryOp::Multiply => {
                let lhs = self.eval_expr(left)?;
                let rhs = self.eval_expr(right)?;
                Ok(Value::Number(as_number(&lhs)? * as_number(&rhs)?))
            }
            BinaryOp::Divide => {
                let lhs = self.eval_expr(left)?;
                let rhs = self.eval_expr(right)?;
                Ok(Value::Number(as_number(&lhs)? / as_number(&rhs)?))
            }
            BinaryOp::Modulo => {
                let lhs = self.eval_expr(left)?;
                let rhs = self.eval_expr(right)?;
                Ok(Value::Number(as_number(&lhs)? % as_number(&rhs)?))
            }
            BinaryOp::Equal => {
                let lhs = self.eval_expr(left)?;
                let rhs = self.eval_expr(right)?;
                Ok(Value::Bool(lhs == rhs))
            }
            BinaryOp::NotEqual => {
                let lhs = self.eval_expr(left)?;
                let rhs = self.eval_expr(right)?;
                Ok(Value::Bool(lhs != rhs))
            }
            BinaryOp::Less => {
                let lhs = self.eval_expr(left)?;
                let rhs = self.eval_expr(right)?;
                compare_numbers(lhs, rhs, |a, b| a < b)
            }
            BinaryOp::LessEqual => {
                let lhs = self.eval_expr(left)?;
                let rhs = self.eval_expr(right)?;
                compare_numbers(lhs, rhs, |a, b| a <= b)
            }
            BinaryOp::Greater => {
                let lhs = self.eval_expr(left)?;
                let rhs = self.eval_expr(right)?;
                compare_numbers(lhs, rhs, |a, b| a > b)
            }
            BinaryOp::GreaterEqual => {
                let lhs = self.eval_expr(left)?;
                let rhs = self.eval_expr(right)?;
                compare_numbers(lhs, rhs, |a, b| a >= b)
            }
        }
    }

    fn eval_tool_call(&mut self, call: &CallExpr) -> Result<Value, RuntimeError> {
        let mut args = Record::with_capacity(call.args.len());
        for (name, expr) in &call.args {
            args.insert(name.clone(), self.eval_expr(expr)?);
        }
        let result = match self.host.call(&call.name, &args) {
            Ok(value) => success(value),
            Err(error) => error_value(error.to_string()),
        };
        Ok(result)
    }

    fn eval_builtin(&mut self, name: &str, args: &[Expr]) -> Result<Value, RuntimeError> {
        let mut values = Vec::with_capacity(args.len());
        for expr in args {
            values.push(self.eval_expr(expr)?);
        }

        match name {
            "len" => {
                expect_arg_count(name, &values, 1)?;
                match &values[0] {
                    Value::String(value) => Ok(Value::Number(value.chars().count() as f64)),
                    Value::List(values) => Ok(Value::Number(values.len() as f64)),
                    Value::Record(record) => Ok(Value::Number(record.len() as f64)),
                    _ => Err(RuntimeError::TypeError {
                        message: "`len` requires a string, list, or record".to_string(),
                    }),
                }
            }
            "empty" => {
                expect_arg_count(name, &values, 1)?;
                match &values[0] {
                    Value::String(value) => Ok(Value::Bool(value.is_empty())),
                    Value::List(values) => Ok(Value::Bool(values.is_empty())),
                    Value::Record(record) => Ok(Value::Bool(record.is_empty())),
                    Value::Null => Ok(Value::Bool(true)),
                    _ => Err(RuntimeError::TypeError {
                        message: "`empty` requires a string, list, record, or null".to_string(),
                    }),
                }
            }
            "keys" => {
                expect_arg_count(name, &values, 1)?;
                let Value::Record(record) = &values[0] else {
                    return Err(RuntimeError::TypeError {
                        message: "`keys` requires a record".to_string(),
                    });
                };
                Ok(Value::List(
                    record.keys().cloned().map(Value::String).collect(),
                ))
            }
            "values" => {
                expect_arg_count(name, &values, 1)?;
                let Value::Record(record) = &values[0] else {
                    return Err(RuntimeError::TypeError {
                        message: "`values` requires a record".to_string(),
                    });
                };
                Ok(Value::List(record.values().cloned().collect()))
            }
            "contains" => {
                expect_arg_count(name, &values, 2)?;
                match (&values[0], &values[1]) {
                    (Value::String(haystack), Value::String(needle)) => {
                        Ok(Value::Bool(haystack.contains(needle)))
                    }
                    (Value::List(items), needle) => Ok(Value::Bool(items.contains(needle))),
                    _ => Err(RuntimeError::TypeError {
                        message: "`contains` requires a string/string or list/value pair"
                            .to_string(),
                    }),
                }
            }
            "starts_with" => {
                expect_arg_count(name, &values, 2)?;
                Ok(Value::Bool(
                    as_string(&values[0])?.starts_with(as_string(&values[1])?),
                ))
            }
            "ends_with" => {
                expect_arg_count(name, &values, 2)?;
                Ok(Value::Bool(
                    as_string(&values[0])?.ends_with(as_string(&values[1])?),
                ))
            }
            "split" => {
                expect_arg_count(name, &values, 2)?;
                Ok(Value::List(
                    as_string(&values[0])?
                        .split(as_string(&values[1])?)
                        .map(|part| Value::String(part.to_string()))
                        .collect(),
                ))
            }
            "join" => {
                expect_arg_count(name, &values, 2)?;
                let Value::List(items) = &values[0] else {
                    return Err(RuntimeError::TypeError {
                        message: "`join` requires a list as the first argument".to_string(),
                    });
                };
                let sep = as_string(&values[1])?;
                let parts = items.iter().map(as_string).collect::<Result<Vec<_>, _>>()?;
                Ok(Value::String(parts.join(sep)))
            }
            "trim" => {
                expect_arg_count(name, &values, 1)?;
                Ok(Value::String(as_string(&values[0])?.trim().to_string()))
            }
            "slice" => {
                expect_arg_count(name, &values, 3)?;
                let start = as_index(&values[1])?;
                let end = as_index(&values[2])?;
                match &values[0] {
                    Value::String(value) => {
                        let chars: Vec<char> = value.chars().collect();
                        let start = start.min(chars.len());
                        let end = end.min(chars.len());
                        Ok(Value::String(chars[start..end].iter().collect()))
                    }
                    Value::List(items) => {
                        let start = start.min(items.len());
                        let end = end.min(items.len());
                        Ok(Value::List(items[start..end].to_vec()))
                    }
                    _ => Err(RuntimeError::TypeError {
                        message: "`slice` requires a string or list".to_string(),
                    }),
                }
            }
            "to_string" => {
                expect_arg_count(name, &values, 1)?;
                Ok(Value::String(stringify_value(&values[0])?))
            }
            "to_int" => {
                expect_arg_count(name, &values, 1)?;
                let value = match &values[0] {
                    Value::Number(value) => value.trunc(),
                    Value::String(value) => {
                        value.parse::<f64>().map_err(|_| RuntimeError::TypeError {
                            message: "`to_int` string argument must be numeric".to_string(),
                        })?
                    }
                    _ => {
                        return Err(RuntimeError::TypeError {
                            message: "`to_int` requires a number or numeric string".to_string(),
                        });
                    }
                };
                Ok(Value::Number(value.trunc()))
            }
            "to_float" => {
                expect_arg_count(name, &values, 1)?;
                let value = match &values[0] {
                    Value::Number(value) => *value,
                    Value::String(value) => {
                        value.parse::<f64>().map_err(|_| RuntimeError::TypeError {
                            message: "`to_float` string argument must be numeric".to_string(),
                        })?
                    }
                    _ => {
                        return Err(RuntimeError::TypeError {
                            message: "`to_float` requires a number or numeric string".to_string(),
                        });
                    }
                };
                Ok(Value::Number(value))
            }
            "json_parse" => {
                expect_arg_count(name, &values, 1)?;
                let parsed: serde_json::Value = serde_json::from_str(as_string(&values[0])?)
                    .map_err(|err| RuntimeError::ValueError {
                        message: format!("invalid json: {err}"),
                    })?;
                Ok(from_json(parsed))
            }
            "json_stringify" => {
                expect_arg_count(name, &values, 1)?;
                Ok(Value::String(
                    serde_json::to_string(&to_json(&values[0])).expect("json serialization failed"),
                ))
            }
            "format" => {
                if values.is_empty() {
                    return Err(RuntimeError::TypeError {
                        message: "`format` requires at least a template string".to_string(),
                    });
                }
                let template = as_string(&values[0])?;
                Ok(Value::String(apply_format(template, &values[1..])?))
            }
            _ => Err(RuntimeError::UnknownBuiltin {
                name: name.to_string(),
            }),
        }
    }

    fn into_branch_result(self) -> BranchResult {
        let mut values = Record::new();
        for name in self.assigned {
            if let Some(value) = self.scopes.get(&name).cloned() {
                values.insert(name, value);
            }
        }
        BranchResult { values }
    }

    fn into_globals(self) -> Record {
        self.scopes.globals
    }
}

#[derive(Clone)]
struct ScopeStack {
    globals: Record,
    locals: Vec<Record>,
}

impl ScopeStack {
    fn new(globals: Record) -> Self {
        Self {
            globals,
            locals: Vec::new(),
        }
    }

    fn get(&self, name: &str) -> Option<&Value> {
        for scope in self.locals.iter().rev() {
            if let Some(value) = scope.get(name) {
                return Some(value);
            }
        }
        self.globals.get(name)
    }

    fn assign(&mut self, name: String, value: Value) {
        for scope in self.locals.iter_mut().rev() {
            if let Some(slot) = scope.get_mut(&name) {
                *slot = value;
                return;
            }
        }
        if let Some(slot) = self.globals.get_mut(&name) {
            *slot = value;
            return;
        }
        if let Some(scope) = self.locals.last_mut() {
            scope.insert(name, value);
        } else {
            self.globals.insert(name, value);
        }
    }

    fn bind_temporary(&mut self, name: &str, value: Value) -> BindingRestore {
        let previous = self.get(name).cloned();
        self.assign(name.to_string(), value);
        BindingRestore { previous }
    }

    fn restore_temporary(&mut self, name: &str, restore: BindingRestore) {
        match restore.previous {
            Some(value) => self.assign(name.to_string(), value),
            None => {
                for scope in self.locals.iter_mut().rev() {
                    if scope.remove(name).is_some() {
                        return;
                    }
                }
                self.globals.remove(name);
            }
        }
    }
}

#[derive(Debug, PartialEq)]
enum Flow {
    Continue,
    Finished(Value),
}

struct BranchResult {
    values: Record,
}

struct BindingRestore {
    previous: Option<Value>,
}

fn expect_arg_count(name: &str, values: &[Value], expected: usize) -> Result<(), RuntimeError> {
    if values.len() == expected {
        Ok(())
    } else {
        Err(RuntimeError::TypeError {
            message: format!(
                "`{name}` expects {expected} arguments, got {}",
                values.len()
            ),
        })
    }
}

fn as_number(value: &Value) -> Result<f64, RuntimeError> {
    match value {
        Value::Number(value) => Ok(*value),
        _ => Err(RuntimeError::TypeError {
            message: "expected a number".to_string(),
        }),
    }
}

fn as_string(value: &Value) -> Result<&str, RuntimeError> {
    match value {
        Value::String(value) => Ok(value),
        _ => Err(RuntimeError::TypeError {
            message: "expected a string".to_string(),
        }),
    }
}

fn as_index(value: &Value) -> Result<usize, RuntimeError> {
    let number = as_number(value)?;
    if !number.is_finite() || number < 0.0 || number.fract() != 0.0 {
        return Err(RuntimeError::TypeError {
            message: "index must be a non-negative integer".to_string(),
        });
    }
    Ok(number as usize)
}

fn compare_numbers(
    left: Value,
    right: Value,
    cmp: impl FnOnce(f64, f64) -> bool,
) -> Result<Value, RuntimeError> {
    Ok(Value::Bool(cmp(as_number(&left)?, as_number(&right)?)))
}

fn add_values(left: Value, right: Value) -> Result<Value, RuntimeError> {
    match (left, right) {
        (Value::Number(a), Value::Number(b)) => Ok(Value::Number(a + b)),
        (Value::String(a), Value::String(b)) => Ok(Value::String(a + &b)),
        (Value::List(mut a), Value::List(b)) => {
            a.extend(b);
            Ok(Value::List(a))
        }
        _ => Err(RuntimeError::TypeError {
            message: "`+` supports number, string, or list pairs".to_string(),
        }),
    }
}

fn success(value: Value) -> Value {
    let mut record = Record::new();
    record.insert("ok".to_string(), Value::Bool(true));
    record.insert("value".to_string(), value);
    Value::Record(record)
}

fn error_value(message: String) -> Value {
    let mut record = Record::new();
    record.insert("ok".to_string(), Value::Bool(false));
    record.insert("error".to_string(), Value::String(message));
    Value::Record(record)
}

fn stringify_value(value: &Value) -> Result<String, RuntimeError> {
    match value {
        Value::String(value) => Ok(value.clone()),
        Value::Null => Ok("null".to_string()),
        Value::Bool(value) => Ok(value.to_string()),
        Value::Number(value) => Ok(value.to_string()),
        Value::List(_) | Value::Record(_) => Ok(
            serde_json::to_string(&to_json(value))
                .expect("value json serialization should succeed"),
        ),
    }
}

fn apply_format(template: &str, args: &[Value]) -> Result<String, RuntimeError> {
    let mut output = String::with_capacity(template.len());
    let chars: Vec<char> = template.chars().collect();
    let mut index = 0;
    while index < chars.len() {
        if chars[index] == '{' {
            let mut cursor = index + 1;
            let mut digits = String::new();
            while cursor < chars.len() && chars[cursor].is_ascii_digit() {
                digits.push(chars[cursor]);
                cursor += 1;
            }
            if cursor < chars.len() && chars[cursor] == '}' && !digits.is_empty() {
                let slot = digits
                    .parse::<usize>()
                    .map_err(|_| RuntimeError::ValueError {
                        message: format!("invalid format slot `{digits}`"),
                    })?;
                let value = args.get(slot).ok_or_else(|| RuntimeError::ValueError {
                    message: format!("format slot {{{slot}}} is out of range"),
                })?;
                output.push_str(&stringify_value(value)?);
                index = cursor + 1;
                continue;
            }
        }
        output.push(chars[index]);
        index += 1;
    }
    Ok(output)
}

fn to_json(value: &Value) -> serde_json::Value {
    match value {
        Value::Null => serde_json::Value::Null,
        Value::Bool(value) => serde_json::Value::Bool(*value),
        Value::Number(value) => serde_json::Number::from_f64(*value)
            .map(serde_json::Value::Number)
            .unwrap_or(serde_json::Value::Null),
        Value::String(value) => serde_json::Value::String(value.clone()),
        Value::List(values) => serde_json::Value::Array(values.iter().map(to_json).collect()),
        Value::Record(record) => serde_json::Value::Object(
            record
                .iter()
                .map(|(key, value)| (key.clone(), to_json(value)))
                .collect(),
        ),
    }
}

fn from_json(value: serde_json::Value) -> Value {
    match value {
        serde_json::Value::Null => Value::Null,
        serde_json::Value::Bool(value) => Value::Bool(value),
        serde_json::Value::Number(value) => Value::Number(value.as_f64().unwrap_or_default()),
        serde_json::Value::String(value) => Value::String(value),
        serde_json::Value::Array(values) => {
            Value::List(values.into_iter().map(from_json).collect())
        }
        serde_json::Value::Object(map) => Value::Record(
            map.into_iter()
                .map(|(key, value)| (key, from_json(value)))
                .collect(),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct Host;

    impl ToolHost for Host {
        fn call(&self, name: &str, args: &Record) -> Result<Value, ToolHostError> {
            match name {
                "echo" => Ok(args.get("value").cloned().unwrap_or(Value::Null)),
                "err" => Err(ToolHostError::new("boom")),
                _ => Err(ToolHostError::new(format!("unknown tool: {name}"))),
            }
        }
    }

    fn exec(source: &str) -> Result<Value, RuntimeError> {
        let program = crate::parse(source).expect("program should parse");
        let mut state = State::new();
        execute_program(&program, &mut state, &Host)
    }

    #[test]
    fn value_helpers_and_display_cover_all_variants() {
        let mut record = Record::new();
        record.insert("k".to_string(), Value::Number(1.0));

        assert_eq!(Value::Null.to_string(), "null");
        assert_eq!(Value::Bool(true).to_string(), "true");
        assert_eq!(Value::Number(1.5).to_string(), "1.5");
        assert_eq!(Value::String("x".to_string()).to_string(), "x");
        assert_eq!(Value::List(vec![Value::Bool(true)]).to_string(), "[true]");
        assert_eq!(
            Value::Record(record.clone()).as_record().unwrap()["k"],
            Value::Number(1.0)
        );
        assert!(Value::String("x".to_string()).as_record().is_none());
        assert!(Value::Record(record).to_string().contains("\"k\":1.0"));
    }

    #[test]
    fn missing_finish_and_undefined_variable_are_reported() {
        let err = exec("x = 1").expect_err("missing finish should fail");
        assert_eq!(err, RuntimeError::MissingFinish);

        let err = exec("finish x").expect_err("undefined variable should fail");
        assert_eq!(
            err,
            RuntimeError::UndefinedVariable {
                name: "x".to_string()
            }
        );
    }

    #[test]
    fn condition_and_iteration_errors_are_reported() {
        let err = exec("if 1 { finish 1 }").expect_err("non-bool condition should fail");
        assert_eq!(err, RuntimeError::NonBooleanCondition);

        let err = exec("for x in 1 { finish x }").expect_err("non-list iteration should fail");
        assert_eq!(err, RuntimeError::NonListIteration);
    }

    #[test]
    fn stmt_call_and_tool_results_cover_success_and_error() {
        exec("call echo { value: 1 } finish 1").expect("statement call should succeed");
        let missing = exec("bad = call missing {} finish bad").expect("missing tool should be wrapped");
        assert_eq!(
            missing.as_record().expect("result should be a record")["ok"],
            Value::Bool(false)
        );

        let value = exec("ok = call echo { value: 7 } bad = call err {} finish { ok: ok, bad: bad }")
            .expect("tool call program should succeed");
        let record = value.as_record().expect("expected record");
        assert_eq!(record["ok"].as_record().unwrap()["ok"], Value::Bool(true));
        assert_eq!(record["bad"].as_record().unwrap()["ok"], Value::Bool(false));
    }

    #[test]
    fn field_index_unary_and_boolean_paths_are_covered() {
        let value = exec(
            r#"
            rec = { nested: { name: "lash" } }
            xs = ["a", "b"]
            ok = false and missing
            alt = true or missing
            finish [rec.nested.name, xs[1], "abc"[2], -1, not false, ok, alt]
            "#,
        )
        .expect("program should succeed");

        assert_eq!(
            value,
            Value::List(vec![
                Value::String("lash".to_string()),
                Value::String("b".to_string()),
                Value::String("c".to_string()),
                Value::Number(-1.0),
                Value::Bool(true),
                Value::Bool(false),
                Value::Bool(true),
            ])
        );

        let value = exec("finish true and false").expect("and path should succeed");
        assert_eq!(value, Value::Bool(false));

        let value = exec("finish false or true").expect("or path should succeed");
        assert_eq!(value, Value::Bool(true));
    }

    #[test]
    fn field_index_and_type_errors_are_covered() {
        let err = exec("n = 1 finish n.name").expect_err("field access should fail");
        assert!(matches!(err, RuntimeError::TypeError { .. }));

        let err = exec("rec = {} finish rec.name").expect_err("missing field should fail");
        assert!(matches!(err, RuntimeError::ValueError { .. }));

        let err = exec("finish 1[0]").expect_err("bad index target should fail");
        assert!(matches!(err, RuntimeError::TypeError { .. }));

        let err = exec("finish [1][2]").expect_err("list oob should fail");
        assert!(matches!(err, RuntimeError::ValueError { .. }));

        let err = exec("finish \"a\"[2]").expect_err("string oob should fail");
        assert!(matches!(err, RuntimeError::ValueError { .. }));

        let err = exec("finish [1][1.5]").expect_err("fractional index should fail");
        assert!(matches!(err, RuntimeError::TypeError { .. }));

        let err = exec("finish [1][-1]").expect_err("negative index should fail");
        assert!(matches!(err, RuntimeError::TypeError { .. }));

        let err = exec("finish not 1").expect_err("not type error should fail");
        assert!(matches!(err, RuntimeError::TypeError { .. }));
    }

    #[test]
    fn arithmetic_and_compare_errors_are_covered() {
        assert_eq!(
            exec("finish 7 - 2").expect("subtract should succeed"),
            Value::Number(5.0)
        );
        assert_eq!(
            exec("finish 3 * 4").expect("multiply should succeed"),
            Value::Number(12.0)
        );
        assert_eq!(
            exec("finish 8 / 2").expect("divide should succeed"),
            Value::Number(4.0)
        );
        assert_eq!(
            exec("finish 8 % 3").expect("modulo should succeed"),
            Value::Number(2.0)
        );
        assert_eq!(
            exec("finish 1 != 2").expect("not equal should succeed"),
            Value::Bool(true)
        );
        assert_eq!(
            exec("finish 1 <= 2").expect("less-equal should succeed"),
            Value::Bool(true)
        );
        assert_eq!(
            exec("finish 2 > 1").expect("greater should succeed"),
            Value::Bool(true)
        );
        assert_eq!(
            exec("finish 2 >= 1").expect("greater-equal should succeed"),
            Value::Bool(true)
        );

        let value = exec("finish [1,2] + [3]").expect("list concat should succeed");
        assert_eq!(
            value,
            Value::List(vec![
                Value::Number(1.0),
                Value::Number(2.0),
                Value::Number(3.0)
            ])
        );

        let value = exec("finish \"a\" + \"b\"").expect("string add should succeed");
        assert_eq!(value, Value::String("ab".to_string()));

        let err = exec("finish 1 + true").expect_err("bad add should fail");
        assert!(matches!(err, RuntimeError::TypeError { .. }));

        let err = exec("finish 1 < true").expect_err("bad compare should fail");
        assert!(matches!(err, RuntimeError::TypeError { .. }));
    }

    #[test]
    fn builtin_success_matrix_is_covered() {
        let value = exec(
            r#"
            rec = { a: 1, b: 2 }
            finish {
              len_s: len("ab"),
              len_l: len([1,2,3]),
              len_r: len(rec),
              empty_n: empty(null),
              empty_s: empty(""),
              empty_l: empty([]),
              empty_r: empty({}),
              keys: keys(rec),
              values: values(rec),
              contains_s: contains("abc", "b"),
              contains_l: contains([1,2,3], 2),
              starts: starts_with("lash", "la"),
              ends: ends_with("lash", "sh"),
              split: split("a,b", ","),
              join: join(["a","b"], "-"),
              trim: trim(" hi "),
              slice_s: slice("abcd", 1, 3),
              slice_l: slice([1,2,3,4], 1, 3),
              to_s: to_string({ a: 1 }),
              to_i_n: to_int(3.9),
              to_i_s: to_int("4"),
              to_f_n: to_float(1),
              to_f_s: to_float("2.5"),
              json_s: json_stringify({ a: 1 }),
              fmt: format("x={0},y={1}", 1, true)
            }
            "#,
        )
        .expect("builtins should succeed");

        let record = value.as_record().expect("expected record");
        assert_eq!(record["len_s"], Value::Number(2.0));
        assert_eq!(record["contains_l"], Value::Bool(true));
        assert_eq!(record["trim"], Value::String("hi".to_string()));
        assert_eq!(record["slice_s"], Value::String("bc".to_string()));
        assert_eq!(record["to_i_n"], Value::Number(3.0));
        assert_eq!(record["to_f_s"], Value::Number(2.5));
        assert_eq!(record["fmt"], Value::String("x=1,y=true".to_string()));
    }

    #[test]
    fn builtin_error_matrix_is_covered() {
        let cases = [
            ("finish len(true)", "len"),
            ("finish empty(true)", "empty"),
            ("finish keys([])", "keys"),
            ("finish values([])", "values"),
            ("finish contains(1, 2)", "contains"),
            ("finish starts_with(1, \"a\")", "starts_with"),
            ("finish ends_with(1, \"a\")", "ends_with"),
            ("finish split(1, \",\")", "split"),
            ("finish join(1, \",\")", "join"),
            ("finish trim(1)", "trim"),
            ("finish slice(1, 0, 1)", "slice"),
            ("finish to_int(true)", "to_int"),
            ("finish to_int(\"x\")", "to_int"),
            ("finish to_float(true)", "to_float"),
            ("finish to_float(\"x\")", "to_float"),
            ("finish json_parse(\"{\")", "json_parse"),
            ("finish format()", "format"),
            ("finish format(1)", "format"),
            ("finish format(\"{1}\", \"x\")", "format"),
            ("finish no_such_builtin()", "no_such_builtin"),
        ];

        for (source, _) in cases {
            let err = exec(source).expect_err("builtin should fail");
            assert!(matches!(
                err,
                RuntimeError::TypeError { .. }
                    | RuntimeError::ValueError { .. }
                    | RuntimeError::UnknownBuiltin { .. }
            ));
        }

        let err = exec("finish len()").expect_err("arity error should fail");
        assert!(matches!(err, RuntimeError::TypeError { .. }));
    }

    #[test]
    fn scope_stack_local_paths_are_covered_directly() {
        let mut stack = ScopeStack {
            globals: Record::from([("global".to_string(), Value::Number(1.0))]),
            locals: vec![Record::from([("local".to_string(), Value::Number(2.0))])],
        };

        assert_eq!(stack.get("local"), Some(&Value::Number(2.0)));
        stack.assign("local".to_string(), Value::Number(3.0));
        assert_eq!(stack.get("local"), Some(&Value::Number(3.0)));

        let restore = stack.bind_temporary("temp", Value::Number(4.0));
        assert_eq!(stack.get("temp"), Some(&Value::Number(4.0)));
        stack.restore_temporary("temp", restore);
        assert_eq!(stack.get("temp"), None);

        let restore = stack.bind_temporary("global", Value::Number(9.0));
        stack.restore_temporary("global", restore);
        assert_eq!(stack.get("global"), Some(&Value::Number(1.0)));

        stack.locals.push(Record::from([("x".to_string(), Value::Number(1.0))]));
        stack.restore_temporary("x", BindingRestore { previous: None });
        assert_eq!(stack.get("x"), None);

        stack.locals.push(Record::from([("other".to_string(), Value::Number(5.0))]));
        stack.restore_temporary("missing", BindingRestore { previous: None });
        assert_eq!(stack.get("other"), Some(&Value::Number(5.0)));
        assert_eq!(stack.get("missing"), None);
    }

    #[test]
    fn helper_functions_are_covered_directly() {
        assert!(expect_arg_count("x", &[Value::Null], 1).is_ok());
        assert!(expect_arg_count("x", &[], 1).is_err());
        assert_eq!(as_number(&Value::Number(1.0)).expect("number"), 1.0);
        assert!(as_number(&Value::Bool(true)).is_err());
        assert_eq!(as_string(&Value::String("x".to_string())).expect("string"), "x");
        assert!(as_string(&Value::Bool(true)).is_err());
        assert_eq!(as_index(&Value::Number(2.0)).expect("index"), 2);

        assert_eq!(
            compare_numbers(Value::Number(1.0), Value::Number(2.0), |a, b| a < b)
                .expect("compare"),
            Value::Bool(true)
        );
        assert_eq!(
            add_values(Value::Number(1.0), Value::Number(2.0)).expect("add"),
            Value::Number(3.0)
        );
        assert_eq!(success(Value::Number(1.0)).as_record().unwrap()["ok"], Value::Bool(true));
        assert_eq!(
            error_value("x".to_string()).as_record().unwrap()["error"],
            Value::String("x".to_string())
        );
        assert_eq!(stringify_value(&Value::Null).expect("stringify"), "null");
        assert_eq!(apply_format("a{0}b", &[Value::Number(1.0)]).expect("format"), "a1b");
        assert_eq!(
            apply_format("{999999999999999999999999999999999999}", &[])
                .expect_err("overflow slot should fail"),
            RuntimeError::ValueError {
                message: "invalid format slot `999999999999999999999999999999999999`".to_string()
            }
        );
        assert_eq!(
            apply_format("{x}", &[]).expect("invalid brace pattern should pass through"),
            "{x}"
        );
    }

    #[test]
    fn json_helpers_cover_special_paths() {
        let json = to_json(&Value::Number(f64::NAN));
        assert_eq!(json, serde_json::Value::Null);
        assert_eq!(to_json(&Value::Null), serde_json::Value::Null);
        assert_eq!(to_json(&Value::Bool(true)), serde_json::Value::Bool(true));
        assert_eq!(
            to_json(&Value::String("x".to_string())),
            serde_json::Value::String("x".to_string())
        );
        assert_eq!(
            to_json(&Value::List(vec![Value::Number(1.0)])),
            serde_json::json!([1.0])
        );
        assert_eq!(
            to_json(&Value::Record(Record::from([(
                "a".to_string(),
                Value::Number(1.0)
            )]))),
            serde_json::json!({"a": 1.0})
        );

        let value = from_json(serde_json::json!({
            "a": [1, true, null, "x"]
        }));
        let record = value.as_record().expect("expected record");
        assert!(matches!(record["a"], Value::List(_)));
    }

    #[test]
    fn false_if_branch_and_finish_inside_loop_are_covered() {
        let value = exec(
            r#"
            if false {
              out = 1
            } else {
              out = 2
            }
            finish out
            "#,
        )
        .expect("else branch should succeed");
        assert_eq!(value, Value::Number(2.0));

        let value = exec(
            r#"
            for x in [1, 2] {
              finish x
            }
            finish 0
            "#,
        )
        .expect("finish inside loop should bubble out");
        assert_eq!(value, Value::Number(1.0));
    }
}
