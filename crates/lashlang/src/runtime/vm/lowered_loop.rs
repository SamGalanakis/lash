//! Optimizer-only loop IR and evaluator.
//!
//! These types are not semantic language structure. The compiler emits them
//! only after a loop is proven suitable for the lower-overhead evaluator, and
//! the VM falls back to normal bytecode whenever direct execution would need
//! projected or async host behavior.

use std::sync::Arc;
use std::time::Instant;

use crate::ast::{BinaryOp, UnaryOp};

use super::super::{
    CompiledFormatTemplate, ExecutionHost, IntrinsicOp, ListValue, RuntimeError, Value, as_number,
    assign_path, eval_binary_values, eval_binary_values_async, eval_number_binary_values,
    execute_compiled_format, execute_compiled_format_direct,
    execute_compiled_format_one_number_compact_direct, execute_integer_div_builtin,
    execute_intrinsic, execute_len_direct, execute_range_builtin, is_truthy, is_truthy_async,
    iterable_values, range_bounds, range_bounds_async, read_field, read_field_direct, read_index,
    read_index_direct, record_with_capacity,
};
use super::Vm;

#[derive(Clone)]
pub(crate) struct LoweredLoop {
    pub(crate) binding: usize,
    pub(crate) iterable: LoopIterable,
    pub(crate) body: Box<[LoopOp]>,
}

#[derive(Clone)]
pub(crate) enum LoopIterable {
    Range(Box<[LoopExpr]>),
    Values(Box<LoopExpr>),
    Keys(Box<LoopExpr>),
}

#[derive(Clone)]
pub(crate) enum LoopOp {
    Assign {
        slot: usize,
        expr: LoopExpr,
    },
    PathAssign {
        slot: usize,
        path: usize,
        indexes: Box<[LoopExpr]>,
        expr: LoopExpr,
    },
    AddAssign {
        slot: usize,
        expr: LoopExpr,
    },
    AddAssignNumber {
        slot: usize,
        right: f64,
    },
    AddAssignSlot {
        slot: usize,
        right: usize,
    },
    AddAssignIndexNumber {
        slot: usize,
        index: LoopExpr,
        right: f64,
    },
    AddAssignIndexSlotNumber {
        slot: usize,
        index: usize,
        right: f64,
    },
    AppendAssign {
        slot: usize,
        expr: LoopExpr,
    },
    Expr(LoopExpr),
    If {
        condition: LoopExpr,
        then_ops: Box<[LoopOp]>,
        else_ops: Box<[LoopOp]>,
    },
    Loop(Box<LoweredLoop>),
    Break,
    Continue,
}

#[derive(Clone)]
pub(crate) enum LoopExpr {
    Const(Value),
    Slot(usize),
    List(Box<[LoopExpr]>),
    Record(Box<[(usize, LoopExpr)]>),
    Intrinsic {
        op: IntrinsicOp,
        args: Box<[LoopExpr]>,
    },
    Format {
        template: CompiledFormatTemplate,
        args: Box<[LoopExpr]>,
    },
    Field {
        target: Box<LoopExpr>,
        field: usize,
    },
    Index {
        target: Box<LoopExpr>,
        index: Box<LoopExpr>,
    },
    Unary {
        op: UnaryOp,
        expr: Box<LoopExpr>,
    },
    Conditional {
        condition: Box<LoopExpr>,
        then_expr: Box<LoopExpr>,
        else_expr: Box<LoopExpr>,
    },
    Binary {
        left: Box<LoopExpr>,
        op: BinaryOp,
        right: Box<LoopExpr>,
    },
}

enum LoopFlow {
    Continue,
    BreakLoop,
    ContinueLoop,
}

impl<H: ExecutionHost> Vm<'_, H> {
    pub(super) async fn execute_lowered_loop(
        &mut self,
        lowered_loop: &LoweredLoop,
    ) -> Result<(), RuntimeError> {
        match &lowered_loop.iterable {
            LoopIterable::Range(args) => {
                let mut values = Vec::with_capacity(args.len());
                for arg in args.iter() {
                    values.push(self.eval_loop_expr(arg).await?);
                }
                let (start, end, step) = range_bounds_async(&values).await?;
                self.execute_lowered_range_loop(
                    lowered_loop.binding,
                    &lowered_loop.body,
                    start,
                    end,
                    step,
                )
                .await
            }
            LoopIterable::Values(expr) => {
                let value = self.eval_loop_expr(expr).await?;
                let values = iterable_values(value).await?;
                self.execute_lowered_value_loop(lowered_loop.binding, &lowered_loop.body, values)
                    .await
            }
            LoopIterable::Keys(expr) => {
                let value = self.eval_loop_expr(expr).await?;
                let value =
                    execute_intrinsic(IntrinsicOp::Keys, &self.chunk.names, &[value]).await?;
                let Value::List(values) = value else {
                    return Err(RuntimeError::NonListIteration);
                };
                self.execute_lowered_value_loop(lowered_loop.binding, &lowered_loop.body, values)
                    .await
            }
        }
    }

    pub(super) fn execute_lowered_loop_direct(
        &mut self,
        lowered_loop: &LoweredLoop,
    ) -> Result<bool, RuntimeError> {
        if !self.lowered_loop_can_eval_direct(lowered_loop) {
            return Ok(false);
        }
        match &lowered_loop.iterable {
            LoopIterable::Range(args) => {
                let mut values = Vec::with_capacity(args.len());
                for arg in args.iter() {
                    values.push(self.eval_loop_expr_direct(arg)?);
                }
                let (start, end, step) = range_bounds(&values)?;
                self.execute_lowered_range_loop_direct(
                    lowered_loop.binding,
                    &lowered_loop.body,
                    start,
                    end,
                    step,
                )?;
            }
            LoopIterable::Values(expr) => {
                let value = self.eval_loop_expr_direct(expr)?;
                let Value::List(values) = value else {
                    return Err(RuntimeError::NonListIteration);
                };
                self.execute_lowered_value_loop_direct(
                    lowered_loop.binding,
                    &lowered_loop.body,
                    values,
                )?;
            }
            LoopIterable::Keys(expr) => {
                let value = self.eval_loop_expr_direct(expr)?;
                let Some(Value::List(values)) =
                    execute_intrinsic_direct(IntrinsicOp::Keys, &[value])?
                else {
                    return Ok(false);
                };
                self.execute_lowered_value_loop_direct(
                    lowered_loop.binding,
                    &lowered_loop.body,
                    values,
                )?;
            }
        }
        Ok(true)
    }

    async fn execute_lowered_range_loop(
        &mut self,
        binding: usize,
        body: &[LoopOp],
        start: i64,
        end: i64,
        step: i64,
    ) -> Result<(), RuntimeError> {
        if !range_has_next(start, end, step) {
            return Ok(());
        }
        self.slots
            .ensure_assignable(binding, &self.chunk.slot_names)?;
        let restore = self.slots.capture_temporary(binding);
        let mut next = start;
        let mut result = Ok(());
        while range_has_next(next, end, step) {
            self.slots
                .assign_loop_binding(binding, Value::Number(next as f64));
            match Box::pin(self.execute_loop_ops(body)).await {
                Ok(LoopFlow::Continue) => {}
                Ok(LoopFlow::ContinueLoop) => {}
                Ok(LoopFlow::BreakLoop) => break,
                Err(error) => {
                    result = Err(error);
                    break;
                }
            }
            next = match next.checked_add(step) {
                Some(next) => next,
                None => {
                    result = Err(RuntimeError::ValueError {
                        message: "`range` overflowed".to_string(),
                    });
                    break;
                }
            };
        }
        self.slots.restore_temporary(binding, restore);
        result
    }

    fn execute_lowered_range_loop_direct(
        &mut self,
        binding: usize,
        body: &[LoopOp],
        start: i64,
        end: i64,
        step: i64,
    ) -> Result<(), RuntimeError> {
        if !range_has_next(start, end, step) {
            return Ok(());
        }
        self.slots
            .ensure_assignable(binding, &self.chunk.slot_names)?;
        let restore = self.slots.capture_temporary(binding);
        let mut next = start;
        let mut result = Ok(());
        while range_has_next(next, end, step) {
            self.slots
                .assign_loop_binding(binding, Value::Number(next as f64));
            match self.execute_loop_ops_direct(body) {
                Ok(LoopFlow::Continue) => {}
                Ok(LoopFlow::ContinueLoop) => {}
                Ok(LoopFlow::BreakLoop) => break,
                Err(error) => {
                    result = Err(error);
                    break;
                }
            }
            next = match next.checked_add(step) {
                Some(next) => next,
                None => {
                    result = Err(RuntimeError::ValueError {
                        message: "`range` overflowed".to_string(),
                    });
                    break;
                }
            };
        }
        self.slots.restore_temporary(binding, restore);
        result
    }

    async fn execute_lowered_value_loop(
        &mut self,
        binding: usize,
        body: &[LoopOp],
        values: ListValue,
    ) -> Result<(), RuntimeError> {
        if values.is_empty() {
            return Ok(());
        }
        self.slots
            .ensure_assignable(binding, &self.chunk.slot_names)?;
        let restore = self.slots.capture_temporary(binding);
        let mut result = Ok(());
        for value in values.iter().cloned() {
            self.slots.assign_loop_binding(binding, value);
            match Box::pin(self.execute_loop_ops(body)).await {
                Ok(LoopFlow::Continue) => {}
                Ok(LoopFlow::ContinueLoop) => continue,
                Ok(LoopFlow::BreakLoop) => break,
                Err(error) => {
                    result = Err(error);
                    break;
                }
            }
        }
        self.slots.restore_temporary(binding, restore);
        result
    }

    fn execute_lowered_value_loop_direct(
        &mut self,
        binding: usize,
        body: &[LoopOp],
        values: ListValue,
    ) -> Result<(), RuntimeError> {
        if values.is_empty() {
            return Ok(());
        }
        self.slots
            .ensure_assignable(binding, &self.chunk.slot_names)?;
        let restore = self.slots.capture_temporary(binding);
        let mut result = Ok(());
        for value in values.iter().cloned() {
            self.slots.assign_loop_binding(binding, value);
            match self.execute_loop_ops_direct(body) {
                Ok(LoopFlow::Continue) => {}
                Ok(LoopFlow::ContinueLoop) => continue,
                Ok(LoopFlow::BreakLoop) => break,
                Err(error) => {
                    result = Err(error);
                    break;
                }
            }
        }
        self.slots.restore_temporary(binding, restore);
        result
    }

    async fn execute_loop_ops(&mut self, ops: &[LoopOp]) -> Result<LoopFlow, RuntimeError> {
        for op in ops {
            match self.execute_loop_op(op).await? {
                LoopFlow::Continue => {}
                flow => return Ok(flow),
            }
        }
        Ok(LoopFlow::Continue)
    }

    fn execute_loop_ops_direct(&mut self, ops: &[LoopOp]) -> Result<LoopFlow, RuntimeError> {
        for op in ops {
            match self.execute_loop_op_direct(op)? {
                LoopFlow::Continue => {}
                flow => return Ok(flow),
            }
        }
        Ok(LoopFlow::Continue)
    }

    async fn execute_loop_op(&mut self, op: &LoopOp) -> Result<LoopFlow, RuntimeError> {
        match op {
            LoopOp::Assign { slot, expr } => {
                let value = self.eval_loop_expr(expr).await?;
                self.slots
                    .assign(*slot, value.clone(), &self.chunk.slot_names)?;
                self.record_assignment(*slot);
                self.last_value = Some(value);
            }
            LoopOp::PathAssign {
                slot,
                path,
                indexes,
                expr,
            } => {
                let value = self.eval_loop_expr(expr).await?;
                let last_value = value.clone();
                let mut index_values = Vec::with_capacity(indexes.len());
                for index in indexes.iter() {
                    index_values.push(self.eval_loop_expr(index).await?);
                }
                let root_name = &self.chunk.slot_names[*slot];
                self.slots
                    .ensure_assignable(*slot, &self.chunk.slot_names)?;
                let root =
                    self.slots
                        .get_mut(*slot)
                        .ok_or_else(|| RuntimeError::UndefinedVariable {
                            name: root_name.text.to_string(),
                        })?;
                assign_path(
                    root,
                    &self.chunk.assign_paths[*path],
                    &index_values,
                    value,
                    &self.chunk.names,
                )?;
                self.record_assignment(*slot);
                self.last_value = Some(last_value);
            }
            LoopOp::AddAssign { slot, expr } => {
                let right = self.eval_loop_expr(expr).await?;
                self.add_assign_value(*slot, right)?;
            }
            LoopOp::AddAssignNumber { slot, right } => {
                self.add_assign_value(*slot, Value::Number(*right))?;
            }
            LoopOp::AddAssignSlot { slot, right } => {
                let right = self.load_slot(*right)?.clone();
                self.add_assign_value(*slot, right)?;
            }
            LoopOp::AddAssignIndexNumber { slot, index, right } => {
                let index = self.eval_loop_expr(index).await?;
                self.add_assign_index_number(*slot, &index, *right)?;
            }
            LoopOp::AddAssignIndexSlotNumber { slot, index, right } => {
                let index = self.load_slot(*index)?.clone();
                self.add_assign_index_number(*slot, &index, *right)?;
            }
            LoopOp::AppendAssign { slot, expr } => {
                let item = self.eval_loop_expr(expr).await?;
                self.append_assign_value(*slot, item)?;
            }
            LoopOp::Expr(expr) => {
                self.last_value = Some(self.eval_loop_expr(expr).await?);
            }
            LoopOp::If {
                condition,
                then_ops,
                else_ops,
            } => {
                let condition = self.eval_loop_expr(condition).await?;
                let truthy = is_truthy_async(&condition).await;
                let ops = if truthy { then_ops } else { else_ops };
                match Box::pin(self.execute_loop_ops(ops)).await? {
                    LoopFlow::Continue => {}
                    flow => return Ok(flow),
                }
            }
            LoopOp::Loop(lowered_loop) => {
                Box::pin(self.execute_lowered_loop(lowered_loop)).await?;
            }
            LoopOp::Break => return Ok(LoopFlow::BreakLoop),
            LoopOp::Continue => return Ok(LoopFlow::ContinueLoop),
        }
        Ok(LoopFlow::Continue)
    }

    fn execute_loop_op_direct(&mut self, op: &LoopOp) -> Result<LoopFlow, RuntimeError> {
        match op {
            LoopOp::Assign { slot, expr } => {
                let value = self.eval_loop_expr_direct(expr)?;
                self.slots
                    .assign(*slot, value.clone(), &self.chunk.slot_names)?;
                self.record_assignment(*slot);
                self.last_value = Some(value);
            }
            LoopOp::PathAssign {
                slot,
                path,
                indexes,
                expr,
            } => {
                let value = self.eval_loop_expr_direct(expr)?;
                let last_value = value.clone();
                let mut index_values = Vec::with_capacity(indexes.len());
                for index in indexes.iter() {
                    index_values.push(self.eval_loop_expr_direct(index)?);
                }
                let root_name = &self.chunk.slot_names[*slot];
                self.slots
                    .ensure_assignable(*slot, &self.chunk.slot_names)?;
                let root =
                    self.slots
                        .get_mut(*slot)
                        .ok_or_else(|| RuntimeError::UndefinedVariable {
                            name: root_name.text.to_string(),
                        })?;
                assign_path(
                    root,
                    &self.chunk.assign_paths[*path],
                    &index_values,
                    value,
                    &self.chunk.names,
                )?;
                self.record_assignment(*slot);
                self.last_value = Some(last_value);
            }
            LoopOp::AddAssign { slot, expr } => {
                let right = self.eval_loop_expr_direct(expr)?;
                self.add_assign_value(*slot, right)?;
            }
            LoopOp::AddAssignNumber { slot, right } => {
                self.add_assign_value(*slot, Value::Number(*right))?;
            }
            LoopOp::AddAssignSlot { slot, right } => {
                let right = self.load_slot(*right)?.clone();
                self.add_assign_value(*slot, right)?;
            }
            LoopOp::AddAssignIndexNumber { slot, index, right } => {
                let index = self.eval_loop_expr_direct(index)?;
                self.add_assign_index_number(*slot, &index, *right)?;
            }
            LoopOp::AddAssignIndexSlotNumber { slot, index, right } => {
                let index = self.load_slot(*index)?.clone();
                self.add_assign_index_number(*slot, &index, *right)?;
            }
            LoopOp::AppendAssign { slot, expr } => {
                let item = self.eval_loop_expr_direct(expr)?;
                self.append_assign_value(*slot, item)?;
            }
            LoopOp::Expr(expr) => {
                self.last_value = Some(self.eval_loop_expr_direct(expr)?);
            }
            LoopOp::If {
                condition,
                then_ops,
                else_ops,
            } => {
                let condition = self.eval_loop_expr_direct(condition)?;
                let ops = if is_truthy(&condition) {
                    then_ops
                } else {
                    else_ops
                };
                match self.execute_loop_ops_direct(ops)? {
                    LoopFlow::Continue => {}
                    flow => return Ok(flow),
                }
            }
            LoopOp::Loop(lowered_loop) => {
                debug_assert!(self.lowered_loop_can_eval_direct(lowered_loop));
                self.execute_lowered_loop_direct(lowered_loop)?;
            }
            LoopOp::Break => return Ok(LoopFlow::BreakLoop),
            LoopOp::Continue => return Ok(LoopFlow::ContinueLoop),
        }
        Ok(LoopFlow::Continue)
    }

    async fn eval_loop_expr(&mut self, expr: &LoopExpr) -> Result<Value, RuntimeError> {
        if let Some(value) = self.eval_loop_expr_fast(expr)? {
            return Ok(value);
        }
        match expr {
            LoopExpr::Const(value) => Ok(value.clone()),
            LoopExpr::Slot(slot) => {
                self.slots
                    .get(*slot)
                    .cloned()
                    .ok_or_else(|| RuntimeError::UndefinedVariable {
                        name: self.chunk.slot_names[*slot].text.to_string(),
                    })
            }
            LoopExpr::List(items) => {
                let mut values = Vec::with_capacity(items.len());
                for item in items.iter() {
                    values.push(Box::pin(self.eval_loop_expr(item)).await?);
                }
                Ok(Value::List(values.into()))
            }
            LoopExpr::Record(entries) => {
                let mut record = record_with_capacity(entries.len());
                for (key, value) in entries.iter() {
                    let name = &self.chunk.names[*key];
                    let value = Box::pin(self.eval_loop_expr(value)).await?;
                    record.insert_symbolized(name.symbol, name.text.clone(), value);
                }
                Ok(Value::Record(Arc::new(record)))
            }
            LoopExpr::Intrinsic { op, args } => {
                let mut values = Vec::with_capacity(args.len());
                for arg in args.iter() {
                    values.push(Box::pin(self.eval_loop_expr(arg)).await?);
                }
                let start = self.profile.as_ref().map(|_| Instant::now());
                let value = execute_intrinsic(*op, &self.chunk.names, &values).await?;
                if let Some(start) = start {
                    self.record_builtin_profile(*op, start.elapsed().as_nanos());
                }
                Ok(value)
            }
            LoopExpr::Format { template, args } => {
                let mut values = Vec::with_capacity(args.len());
                for arg in args.iter() {
                    values.push(Box::pin(self.eval_loop_expr(arg)).await?);
                }
                let start = self.profile.as_ref().map(|_| Instant::now());
                let value = execute_compiled_format(template, &values).await?;
                if let Some(start) = start {
                    self.record_builtin_profile(
                        IntrinsicOp::Format(args.len()),
                        start.elapsed().as_nanos(),
                    );
                }
                Ok(Value::String(value.into()))
            }
            LoopExpr::Field { target, field } => {
                let target = Box::pin(self.eval_loop_expr(target)).await?;
                read_field(target, &self.chunk.names[*field]).await
            }
            LoopExpr::Index { target, index } => {
                let target = Box::pin(self.eval_loop_expr(target)).await?;
                let index = Box::pin(self.eval_loop_expr(index)).await?;
                read_index(target, index).await
            }
            LoopExpr::Unary { op, expr } => {
                let value = Box::pin(self.eval_loop_expr(expr)).await?;
                match op {
                    UnaryOp::Negate => {
                        let value = super::super::materialize_projected_async(value).await;
                        Ok(Value::Number(-as_number(&value)?))
                    }
                    UnaryOp::Not => Ok(Value::Bool(!is_truthy_async(&value).await)),
                }
            }
            LoopExpr::Conditional {
                condition,
                then_expr,
                else_expr,
            } => {
                let condition = Box::pin(self.eval_loop_expr(condition)).await?;
                if is_truthy_async(&condition).await {
                    Box::pin(self.eval_loop_expr(then_expr)).await
                } else {
                    Box::pin(self.eval_loop_expr(else_expr)).await
                }
            }
            LoopExpr::Binary { left, op, right } => match op {
                BinaryOp::And => {
                    let left = Box::pin(self.eval_loop_expr(left)).await?;
                    if !is_truthy_async(&left).await {
                        Ok(Value::Bool(false))
                    } else {
                        let right = Box::pin(self.eval_loop_expr(right)).await?;
                        Ok(Value::Bool(is_truthy_async(&right).await))
                    }
                }
                BinaryOp::Or => {
                    let left = Box::pin(self.eval_loop_expr(left)).await?;
                    if is_truthy_async(&left).await {
                        Ok(Value::Bool(true))
                    } else {
                        let right = Box::pin(self.eval_loop_expr(right)).await?;
                        Ok(Value::Bool(is_truthy_async(&right).await))
                    }
                }
                op => {
                    let left = Box::pin(self.eval_loop_expr(left)).await?;
                    let right = Box::pin(self.eval_loop_expr(right)).await?;
                    match (left, right) {
                        (Value::Number(left), Value::Number(right)) => {
                            Ok(eval_number_binary_values(left, *op, right))
                        }
                        (left, right) => eval_binary_values_async(left, *op, right).await,
                    }
                }
            },
        }
    }

    fn eval_loop_expr_direct(&mut self, expr: &LoopExpr) -> Result<Value, RuntimeError> {
        self.eval_loop_expr_fast(expr)?
            .ok_or_else(|| RuntimeError::ValueError {
                message: "lowered loop direct path reached an async expression".to_string(),
            })
    }

    fn eval_loop_expr_fast(&mut self, expr: &LoopExpr) -> Result<Option<Value>, RuntimeError> {
        match expr {
            LoopExpr::Const(value) => Ok(Some(value.clone())),
            LoopExpr::Slot(slot) => {
                let value =
                    self.slots
                        .get(*slot)
                        .ok_or_else(|| RuntimeError::UndefinedVariable {
                            name: self.chunk.slot_names[*slot].text.to_string(),
                        })?;
                if matches!(value, Value::Projected(_)) {
                    Ok(None)
                } else {
                    Ok(Some(value.clone()))
                }
            }
            LoopExpr::List(items) => {
                let Some(values) = self.eval_loop_exprs_fast(items)? else {
                    return Ok(None);
                };
                Ok(Some(Value::List(values.into())))
            }
            LoopExpr::Record(entries) => {
                let mut record = record_with_capacity(entries.len());
                for (key, value) in entries.iter() {
                    let Some(value) = self.eval_loop_expr_fast(value)? else {
                        return Ok(None);
                    };
                    let name = &self.chunk.names[*key];
                    record.insert_symbolized(name.symbol, name.text.clone(), value);
                }
                Ok(Some(Value::Record(Arc::new(record))))
            }
            LoopExpr::Intrinsic { op, args } => {
                let Some(values) = self.eval_loop_exprs_fast(args)? else {
                    return Ok(None);
                };
                let start = self.profile.as_ref().map(|_| Instant::now());
                let Some(value) = execute_intrinsic_direct(*op, &values)? else {
                    return Ok(None);
                };
                if let Some(start) = start {
                    self.record_builtin_profile(*op, start.elapsed().as_nanos());
                }
                Ok(Some(value))
            }
            LoopExpr::Format { template, args } => {
                if let [arg] = args.as_ref() {
                    let Some(value) = self.eval_loop_expr_fast(arg)? else {
                        return Ok(None);
                    };
                    let start = self.profile.as_ref().map(|_| Instant::now());
                    let value = match value {
                        Value::Number(value) => {
                            execute_compiled_format_one_number_compact_direct(template, value)?
                        }
                        value => {
                            execute_compiled_format_direct(template, std::slice::from_ref(&value))?
                                .into()
                        }
                    };
                    if let Some(start) = start {
                        self.record_builtin_profile(
                            IntrinsicOp::Format(args.len()),
                            start.elapsed().as_nanos(),
                        );
                    }
                    return Ok(Some(Value::String(value)));
                }
                let Some(values) = self.eval_loop_exprs_fast(args)? else {
                    return Ok(None);
                };
                let start = self.profile.as_ref().map(|_| Instant::now());
                let value = execute_compiled_format_direct(template, &values)?;
                if let Some(start) = start {
                    self.record_builtin_profile(
                        IntrinsicOp::Format(args.len()),
                        start.elapsed().as_nanos(),
                    );
                }
                Ok(Some(Value::String(value.into())))
            }
            LoopExpr::Field { target, field } => {
                let Some(target) = self.eval_loop_expr_fast(target)? else {
                    return Ok(None);
                };
                Ok(Some(read_field_direct(target, &self.chunk.names[*field])?))
            }
            LoopExpr::Index { target, index } => {
                let Some(target) = self.eval_loop_expr_fast(target)? else {
                    return Ok(None);
                };
                let Some(index) = self.eval_loop_expr_fast(index)? else {
                    return Ok(None);
                };
                Ok(Some(read_index_direct(target, index)?))
            }
            LoopExpr::Unary { op, expr } => {
                let Some(value) = self.eval_loop_expr_fast(expr)? else {
                    return Ok(None);
                };
                let value = match op {
                    UnaryOp::Negate => Value::Number(-as_number(&value)?),
                    UnaryOp::Not => Value::Bool(!is_truthy(&value)),
                };
                Ok(Some(value))
            }
            LoopExpr::Conditional {
                condition,
                then_expr,
                else_expr,
            } => {
                let Some(condition) = self.eval_loop_expr_fast(condition)? else {
                    return Ok(None);
                };
                if is_truthy(&condition) {
                    self.eval_loop_expr_fast(then_expr)
                } else {
                    self.eval_loop_expr_fast(else_expr)
                }
            }
            LoopExpr::Binary { left, op, right } => match op {
                BinaryOp::And => {
                    let Some(left) = self.eval_loop_expr_fast(left)? else {
                        return Ok(None);
                    };
                    if !is_truthy(&left) {
                        Ok(Some(Value::Bool(false)))
                    } else {
                        let Some(right) = self.eval_loop_expr_fast(right)? else {
                            return Ok(None);
                        };
                        Ok(Some(Value::Bool(is_truthy(&right))))
                    }
                }
                BinaryOp::Or => {
                    let Some(left) = self.eval_loop_expr_fast(left)? else {
                        return Ok(None);
                    };
                    if is_truthy(&left) {
                        Ok(Some(Value::Bool(true)))
                    } else {
                        let Some(right) = self.eval_loop_expr_fast(right)? else {
                            return Ok(None);
                        };
                        Ok(Some(Value::Bool(is_truthy(&right))))
                    }
                }
                op => {
                    let Some(left) = self.eval_loop_expr_fast(left)? else {
                        return Ok(None);
                    };
                    let Some(right) = self.eval_loop_expr_fast(right)? else {
                        return Ok(None);
                    };
                    let value = match (left, right) {
                        (Value::Number(left), Value::Number(right)) => {
                            eval_number_binary_values(left, *op, right)
                        }
                        (left, right) => eval_binary_values(left, *op, right)?,
                    };
                    Ok(Some(value))
                }
            },
        }
    }

    fn eval_loop_exprs_fast(
        &mut self,
        args: &[LoopExpr],
    ) -> Result<Option<Vec<Value>>, RuntimeError> {
        let mut values = Vec::with_capacity(args.len());
        for arg in args {
            let Some(value) = self.eval_loop_expr_fast(arg)? else {
                return Ok(None);
            };
            values.push(value);
        }
        Ok(Some(values))
    }

    fn lowered_loop_can_eval_direct(&self, lowered_loop: &LoweredLoop) -> bool {
        let iterable_direct = match &lowered_loop.iterable {
            LoopIterable::Range(args) => args.iter().all(|arg| self.loop_expr_can_eval_direct(arg)),
            LoopIterable::Values(expr) | LoopIterable::Keys(expr) => {
                self.loop_expr_can_eval_direct(expr)
            }
        };
        iterable_direct && self.loop_ops_can_eval_direct(&lowered_loop.body)
    }

    fn loop_ops_can_eval_direct(&self, ops: &[LoopOp]) -> bool {
        ops.iter().all(|op| self.loop_op_can_eval_direct(op))
    }

    fn loop_op_can_eval_direct(&self, op: &LoopOp) -> bool {
        match op {
            LoopOp::Assign { expr, .. }
            | LoopOp::AddAssign { expr, .. }
            | LoopOp::AppendAssign { expr, .. }
            | LoopOp::Expr(expr) => self.loop_expr_can_eval_direct(expr),
            LoopOp::PathAssign { indexes, expr, .. } => {
                self.loop_expr_can_eval_direct(expr)
                    && indexes
                        .iter()
                        .all(|index| self.loop_expr_can_eval_direct(index))
            }
            LoopOp::AddAssignNumber { .. } | LoopOp::AddAssignIndexSlotNumber { .. } => true,
            LoopOp::AddAssignSlot { right, .. } => self.slot_can_eval_direct(*right),
            LoopOp::AddAssignIndexNumber { index, .. } => self.loop_expr_can_eval_direct(index),
            LoopOp::If {
                condition,
                then_ops,
                else_ops,
            } => {
                self.loop_expr_can_eval_direct(condition)
                    && self.loop_ops_can_eval_direct(then_ops)
                    && self.loop_ops_can_eval_direct(else_ops)
            }
            LoopOp::Loop(lowered_loop) => self.lowered_loop_can_eval_direct(lowered_loop),
            LoopOp::Break | LoopOp::Continue => true,
        }
    }

    fn loop_expr_can_eval_direct(&self, expr: &LoopExpr) -> bool {
        match expr {
            LoopExpr::Const(_) => true,
            LoopExpr::Slot(slot) => self.slot_can_eval_direct(*slot),
            LoopExpr::List(items) => items
                .iter()
                .all(|item| self.loop_expr_can_eval_direct(item)),
            LoopExpr::Record(entries) => entries
                .iter()
                .all(|(_, value)| self.loop_expr_can_eval_direct(value)),
            LoopExpr::Intrinsic { op, args } => {
                intrinsic_can_eval_direct(*op)
                    && args.iter().all(|arg| self.loop_expr_can_eval_direct(arg))
            }
            LoopExpr::Format { args, .. } => {
                args.iter().all(|arg| self.loop_expr_can_eval_direct(arg))
            }
            LoopExpr::Field { target, .. } => self.loop_expr_can_eval_direct(target),
            LoopExpr::Index { target, index } => {
                self.loop_expr_can_eval_direct(target) && self.loop_expr_can_eval_direct(index)
            }
            LoopExpr::Unary { expr, .. } => self.loop_expr_can_eval_direct(expr),
            LoopExpr::Conditional {
                condition,
                then_expr,
                else_expr,
            } => {
                self.loop_expr_can_eval_direct(condition)
                    && self.loop_expr_can_eval_direct(then_expr)
                    && self.loop_expr_can_eval_direct(else_expr)
            }
            LoopExpr::Binary { left, right, .. } => {
                self.loop_expr_can_eval_direct(left) && self.loop_expr_can_eval_direct(right)
            }
        }
    }

    fn slot_can_eval_direct(&self, slot: usize) -> bool {
        !self.slots.projected.get(slot).copied().unwrap_or(false)
    }
}

pub(super) fn range_has_next(start: i64, end: i64, step: i64) -> bool {
    (step > 0 && start < end) || (step < 0 && start > end)
}

fn intrinsic_can_eval_direct(op: IntrinsicOp) -> bool {
    matches!(
        op,
        IntrinsicOp::Len
            | IntrinsicOp::Keys
            | IntrinsicOp::Values
            | IntrinsicOp::Range(_)
            | IntrinsicOp::CeilDiv
            | IntrinsicOp::FloorDiv
            | IntrinsicOp::Push
    )
}

fn execute_intrinsic_direct(
    op: IntrinsicOp,
    values: &[Value],
) -> Result<Option<Value>, RuntimeError> {
    match op {
        IntrinsicOp::Len if values.len() == 1 => execute_len_direct(&values[0]).map(Some),
        IntrinsicOp::Keys if values.len() == 1 => match &values[0] {
            Value::Record(record) => Ok(Some(Value::List(
                record
                    .keys()
                    .map(|key| Value::String(key.into()))
                    .collect::<Vec<_>>()
                    .into(),
            ))),
            Value::Null => Ok(Some(Value::List(Vec::new().into()))),
            Value::Projected(_) => Ok(None),
            _ => Err(RuntimeError::TypeError {
                message: "`keys` requires a record or null".to_string(),
            }),
        },
        IntrinsicOp::Values if values.len() == 1 => match &values[0] {
            Value::Record(record) => Ok(Some(Value::List(
                record.values().cloned().collect::<Vec<_>>().into(),
            ))),
            Value::Null => Ok(Some(Value::List(Vec::new().into()))),
            Value::Projected(_) => Ok(None),
            _ => Err(RuntimeError::TypeError {
                message: "`values` requires a record or null".to_string(),
            }),
        },
        IntrinsicOp::Range(_) => execute_range_builtin(values).map(Some),
        IntrinsicOp::CeilDiv => {
            execute_integer_div_builtin("ceil_div", values, f64::ceil).map(Some)
        }
        IntrinsicOp::FloorDiv => {
            execute_integer_div_builtin("floor_div", values, f64::floor).map(Some)
        }
        IntrinsicOp::Push if values.len() == 2 => match &values[0] {
            Value::List(items) => {
                let mut items = items.to_vec();
                if items.len() == items.capacity() {
                    items.reserve(1);
                }
                items.push(values[1].clone());
                Ok(Some(Value::List(items.into())))
            }
            Value::Projected(_) => Ok(None),
            _ => Err(RuntimeError::TypeError {
                message: "`push` requires a list as the first argument".to_string(),
            }),
        },
        _ => Ok(None),
    }
}
