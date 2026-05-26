//! Bytecode executor: walks a compiled `Chunk` of `Instruction`s,
//! materializes intermediate `Value`s, calls the host for tool dispatches,
//! and emits trace/profile data on the way through.
//!
//! The VM is the consumer side of the compiler/vm split: it never writes
//! `Instruction`s, only reads them. Cross-module helpers it relies on
//! (`read_field*`, `eval_binary_values`, `apply_format_async`,
//! `to_json_async`, …) currently live in `mod.rs` and will move to their
//! proper homes (`access.rs`, `ops.rs`, `format.rs`, `json.rs`) in
//! Stage 6.

use std::sync::Arc;
use std::time::Instant;

use rustc_hash::FxHashMap;

use crate::ast::UnaryOp;

mod control;
mod effects;
mod lowered_loop;

pub(crate) use lowered_loop::{LoopExpr, LoopIterable, LoopOp, LoweredLoop};

use control::{VmMode, VmStep};
use effects::VmEffect;
use lowered_loop::range_has_next;

use super::host::{ExecutionMode, ProcessEventKind, ProcessSleepKind};
use super::record::{Record, record_with_capacity};
use super::schema::{
    ValidationPlan, compile_schema_value, execute_validate_builtin, execute_validation_plan,
};
use super::value::ProjectedValue;
use super::{
    Chunk, ExecutionHost, ExecutionScratch, Instruction, InstructionProfileTag, IntrinsicOp,
    LASH_TYPE_KEY, ListValue, Name, ProfileAccumulator, ProfileReport, ProjectedBindings,
    RuntimeError, Value, add_assign_index_number, add_values, as_number, assign_path,
    eval_binary_values, eval_binary_values_async, eval_compare_values, eval_compare_values_async,
    eval_number_binary_values, eval_number_compare_values, eval_number_numeric_binary_value,
    execute_compiled_format, execute_compiled_format_direct,
    execute_compiled_format_one_number_compact_direct, execute_intrinsic,
    execute_push_builtin_async, is_truthy, is_truthy_async, iterable_values,
    materialize_projected_async, materialize_value, range_bounds, range_bounds_async,
    read_field_direct, read_field_ref_direct, read_index_direct, unwrap_tool_result,
    unwrap_type_value,
};

#[derive(Clone)]
pub(crate) struct SlotState {
    values: Vec<Option<Value>>,
    projected: Vec<bool>,
    extras: Record,
}

impl SlotState {
    pub(crate) fn from_globals(
        mut globals: Record,
        slot_names: &[Name],
        projected_bindings: &ProjectedBindings,
    ) -> Self {
        let mut values = Vec::with_capacity(slot_names.len());
        let mut projected = Vec::with_capacity(slot_names.len());
        for name in slot_names {
            if let Some(value) = projected_bindings.get_symbol(name.symbol) {
                globals.remove_symbol(name.symbol);
                values.push(Some(Value::Projected(value)));
                projected.push(true);
            } else {
                values.push(globals.remove_symbol(name.symbol));
                projected.push(false);
            }
        }
        Self {
            values,
            projected,
            extras: globals,
        }
    }

    pub(crate) fn from_globals_with_scratch(
        mut globals: Record,
        slot_names: &[Name],
        scratch: &mut ExecutionScratch,
        projected_bindings: &ProjectedBindings,
    ) -> Self {
        let mut values = std::mem::take(&mut scratch.slot_values);
        values.clear();
        if values.capacity() < slot_names.len() {
            values.reserve(slot_names.len() - values.capacity());
        }
        let mut projected = Vec::with_capacity(slot_names.len());
        for name in slot_names {
            if let Some(value) = projected_bindings.get_symbol(name.symbol) {
                globals.remove_symbol(name.symbol);
                values.push(Some(Value::Projected(value)));
                projected.push(true);
            } else {
                values.push(globals.remove_symbol(name.symbol));
                projected.push(false);
            }
        }
        Self {
            values,
            projected,
            extras: globals,
        }
    }

    pub(crate) fn get(&self, slot: usize) -> Option<&Value> {
        self.values.get(slot).and_then(Option::as_ref)
    }

    fn get_mut(&mut self, slot: usize) -> Option<&mut Value> {
        self.values.get_mut(slot).and_then(Option::as_mut)
    }

    fn assign(
        &mut self,
        slot: usize,
        value: Value,
        slot_names: &[Name],
    ) -> Result<(), RuntimeError> {
        self.ensure_assignable(slot, slot_names)?;
        self.values[slot] = Some(materialize_value(value));
        Ok(())
    }

    fn assign_loop_binding(&mut self, slot: usize, value: Value) {
        self.values[slot] = Some(materialize_value(value));
    }

    fn ensure_assignable(&self, slot: usize, slot_names: &[Name]) -> Result<(), RuntimeError> {
        if self.projected.get(slot).copied().unwrap_or(false) {
            return Err(RuntimeError::TypeError {
                message: format!(
                    "`{}` is a read-only projected binding",
                    slot_names[slot].text
                ),
            });
        }
        Ok(())
    }

    fn capture_temporary(&self, slot: usize) -> LoopRestore {
        LoopRestore {
            previous: self.values[slot].clone(),
        }
    }

    fn restore_temporary(&mut self, slot: usize, restore: LoopRestore) {
        self.values[slot] = restore.previous;
    }

    pub(crate) fn into_globals(self, slot_names: &[Name]) -> Record {
        let mut extras = self.extras;
        for ((name, value), projected) in slot_names.iter().zip(self.values).zip(self.projected) {
            if projected {
                extras.remove_symbol(name.symbol);
                continue;
            }
            match value {
                Some(value) => {
                    extras.insert_symbolized(
                        name.symbol,
                        name.text.clone(),
                        materialize_value(value),
                    );
                }
                None => {
                    extras.remove_symbol(name.symbol);
                }
            }
        }
        extras
    }

    fn recycle_into_globals(
        self,
        slot_names: &[Name],
        slot_values: &mut Vec<Option<Value>>,
    ) -> Record {
        let mut extras = self.extras;
        let mut values = self.values;
        for ((name, value), projected) in
            slot_names.iter().zip(values.iter_mut()).zip(self.projected)
        {
            if projected {
                extras.remove_symbol(name.symbol);
                continue;
            }
            match value.take() {
                Some(value) => {
                    extras.insert_symbolized(
                        name.symbol,
                        name.text.clone(),
                        materialize_value(value),
                    );
                }
                None => {
                    extras.remove_symbol(name.symbol);
                }
            }
        }
        values.clear();
        *slot_values = values;
        extras
    }
}

pub(crate) struct Vm<'a, H> {
    chunk: &'a Chunk,
    ip: usize,
    stack: Vec<Value>,
    last_value: Option<Value>,
    slots: SlotState,
    host: &'a H,
    mode: VmMode,
    iter_stack: Vec<IterState>,
    profile: Option<ProfileAccumulator>,
    validation_plans: FxHashMap<usize, (Arc<Record>, ValidationPlan)>,
}

enum AddAssignRight {
    Number(f64),
    Value(Value),
}

fn validation_plan_cache_entry(schema: &Value) -> Option<(usize, Arc<Record>)> {
    match schema {
        Value::Record(record) => Some((Arc::as_ptr(record) as usize, record.clone())),
        _ => None,
    }
}

impl<'a, H: ExecutionHost> Vm<'a, H> {
    pub(crate) fn new_with_mode(
        chunk: &'a Chunk,
        slots: SlotState,
        host: &'a H,
        mode: ExecutionMode,
    ) -> Self {
        Self {
            chunk,
            ip: 0,
            stack: Vec::new(),
            last_value: None,
            slots,
            host,
            mode: VmMode::from(mode),
            iter_stack: Vec::new(),
            profile: None,
            validation_plans: FxHashMap::default(),
        }
    }

    pub(crate) fn new_with_scratch_and_mode(
        chunk: &'a Chunk,
        slots: SlotState,
        host: &'a H,
        scratch: &mut ExecutionScratch,
        mode: ExecutionMode,
    ) -> Self {
        Self {
            chunk,
            ip: 0,
            stack: std::mem::take(&mut scratch.stack),
            last_value: None,
            slots,
            host,
            mode: VmMode::from(mode),
            iter_stack: std::mem::take(&mut scratch.iter_stack),
            profile: None,
            validation_plans: FxHashMap::default(),
        }
    }

    pub(crate) fn enable_profile(&mut self) {
        self.profile = Some(ProfileAccumulator::default());
    }

    fn execute_dynamic_validate(
        &mut self,
        value: Value,
        schema: &Value,
    ) -> Result<Value, RuntimeError> {
        let Some(schema) = unwrap_type_value(schema) else {
            return execute_validate_builtin(value, schema);
        };
        let Some((key, schema_record)) = validation_plan_cache_entry(schema) else {
            let plan = compile_schema_value(schema);
            return execute_validation_plan(value, &plan);
        };
        let plan = self
            .validation_plans
            .entry(key)
            .or_insert_with(|| (schema_record, compile_schema_value(schema)));
        execute_validation_plan(value, &plan.1)
    }

    #[inline(always)]
    fn step_instruction_fast(
        &mut self,
        instruction: Instruction,
    ) -> Result<Option<VmStep>, RuntimeError> {
        match instruction {
            Instruction::PushConst(index) => {
                self.stack.push(self.chunk.constants[index].clone());
            }
            Instruction::PushNull => self.stack.push(Value::Null),
            Instruction::PushBool(value) => self.stack.push(Value::Bool(value)),
            Instruction::PushNumber(value) => self.stack.push(Value::Number(value)),
            Instruction::LoadName(name) => {
                let value = self.load_slot(name)?.clone();
                self.stack.push(value);
            }
            Instruction::StoreName(name) => {
                let value = self.pop_stack()?;
                self.slots
                    .assign(name, value.clone(), &self.chunk.slot_names)?;
                self.record_assignment(name);
                self.last_value = Some(value);
            }
            Instruction::StoreConst { slot, constant } => {
                let value = self.chunk.constants[constant].clone();
                self.slots
                    .assign(slot, value.clone(), &self.chunk.slot_names)?;
                self.record_assignment(slot);
                self.last_value = Some(value);
            }
            Instruction::BuildList(len) => {
                let values = self.pop_n(len)?;
                self.stack.push(Value::List(values.into()));
            }
            Instruction::BuildRecord(keys) => {
                let record = self.drain_record_from_stack(keys)?;
                self.stack.push(Value::Record(Arc::new(record)));
            }
            Instruction::Field(field) => {
                if self
                    .stack
                    .last()
                    .is_some_and(|value| matches!(value, Value::Projected(_)))
                {
                    return Ok(None);
                }
                let target = self.pop_stack()?;
                let value = read_field_direct(target, &self.chunk.names[field])?;
                self.stack.push(value);
            }
            Instruction::Index => {
                if self.stack.len() < 2
                    || self.stack[self.stack.len() - 2..]
                        .iter()
                        .any(|value| matches!(value, Value::Projected(_)))
                {
                    return Ok(None);
                }
                let index = self.pop_stack()?;
                let target = self.pop_stack()?;
                self.stack.push(read_index_direct(target, index)?);
            }
            Instruction::ResultUnwrap => {
                let value = self.pop_stack()?;
                self.stack.push(unwrap_tool_result(value)?);
            }
            Instruction::Binary(op) => {
                if self.stack.len() < 2
                    || self.stack[self.stack.len() - 2..]
                        .iter()
                        .any(|value| matches!(value, Value::Projected(_)))
                {
                    return Ok(None);
                }
                let right = self.pop_stack()?;
                let left = self.pop_stack()?;
                let value = match (left, right) {
                    (Value::Number(left), Value::Number(right)) => {
                        eval_number_binary_values(left, op, right)
                    }
                    (left, right) => eval_binary_values(left, op, right)?,
                };
                self.stack.push(value);
            }
            Instruction::SlotNumberBinary { slot, op, right } => {
                let value = match self.load_slot(slot)? {
                    Value::Projected(_) => return Ok(None),
                    Value::Number(left) => {
                        Value::Number(eval_number_numeric_binary_value(*left, op, right))
                    }
                    left => eval_binary_values(left.clone(), op, Value::Number(right))?,
                };
                self.stack.push(value);
            }
            Instruction::SlotNumberCompare { slot, op, right } => {
                let value = match self.load_slot(slot)? {
                    Value::Projected(_) => return Ok(None),
                    Value::Number(left) => {
                        Value::Bool(eval_number_compare_values(*left, op, right))
                    }
                    left => {
                        Value::Bool(eval_compare_values(left.clone(), op, Value::Number(right))?)
                    }
                };
                self.stack.push(value);
            }
            Instruction::SlotNumberBinaryCompare {
                slot,
                binary_op,
                binary_right,
                compare_op,
                compare_right,
            } => {
                let truthy = match self.load_slot(slot)? {
                    Value::Projected(_) => return Ok(None),
                    Value::Number(left) => {
                        let value =
                            eval_number_numeric_binary_value(*left, binary_op, binary_right);
                        eval_number_compare_values(value, compare_op, compare_right)
                    }
                    left => {
                        let value = eval_binary_values(
                            left.clone(),
                            binary_op,
                            Value::Number(binary_right),
                        )?;
                        eval_compare_values(value, compare_op, Value::Number(compare_right))?
                    }
                };
                self.stack.push(Value::Bool(truthy));
            }
            Instruction::ToBool => {
                if self
                    .stack
                    .last()
                    .is_some_and(|value| matches!(value, Value::Projected(_)))
                {
                    return Ok(None);
                }
                let value = self.pop_stack()?;
                self.stack.push(Value::Bool(is_truthy(&value)));
            }
            Instruction::Jump(target) => self.ip = target,
            Instruction::JumpIfFalse(target) => {
                if self
                    .stack
                    .last()
                    .is_some_and(|value| matches!(value, Value::Projected(_)))
                {
                    return Ok(None);
                }
                let value = self.pop_stack()?;
                if !is_truthy(&value) {
                    self.ip = target;
                }
            }
            Instruction::JumpIfCompareFalse { op, target } => {
                if self.stack.len() < 2
                    || self.stack[self.stack.len() - 2..]
                        .iter()
                        .any(|value| matches!(value, Value::Projected(_)))
                {
                    return Ok(None);
                }
                let right = self.pop_stack()?;
                let left = self.pop_stack()?;
                if !eval_compare_values(left, op, right)? {
                    self.ip = target;
                }
            }
            Instruction::JumpIfSlotNumberCompareFalse {
                slot,
                op,
                right,
                target,
            } => {
                let truthy = match self.load_slot(slot)? {
                    Value::Projected(_) => return Ok(None),
                    Value::Number(left) => eval_number_compare_values(*left, op, right),
                    value => eval_compare_values(value.clone(), op, Value::Number(right))?,
                };
                if !truthy {
                    self.ip = target;
                }
            }
            Instruction::JumpIfSlotNumberBinaryCompareFalse {
                slot,
                binary_op,
                binary_right,
                compare_op,
                compare_right,
                target,
            } => {
                let truthy = match self.load_slot(slot)? {
                    Value::Projected(_) => return Ok(None),
                    Value::Number(left) => {
                        let value =
                            eval_number_numeric_binary_value(*left, binary_op, binary_right);
                        eval_number_compare_values(value, compare_op, compare_right)
                    }
                    value => {
                        let value = eval_binary_values(
                            value.clone(),
                            binary_op,
                            Value::Number(binary_right),
                        )?;
                        eval_compare_values(value, compare_op, Value::Number(compare_right))?
                    }
                };
                if !truthy {
                    self.ip = target;
                }
            }
            Instruction::JumpIfTrue(target) => {
                if self
                    .stack
                    .last()
                    .is_some_and(|value| matches!(value, Value::Projected(_)))
                {
                    return Ok(None);
                }
                let value = self.pop_stack()?;
                if is_truthy(&value) {
                    self.ip = target;
                }
            }
            Instruction::AddAssign(slot) => {
                let right = self.pop_stack()?;
                self.add_assign_value(slot, right)?;
            }
            Instruction::AddAssignNumber { slot, right } => {
                self.add_assign_value(slot, Value::Number(right))?;
            }
            Instruction::AddAssignSlot { slot, right } => {
                let right = self.load_slot(right)?.clone();
                self.add_assign_value(slot, right)?;
            }
            Instruction::AddAssignIndexNumber { slot, right } => {
                let index = self.pop_stack()?;
                self.add_assign_index_number(slot, &index, right)?;
            }
            Instruction::AddAssignIndexSlotNumber { slot, index, right } => {
                let index = self.load_slot(index)?.clone();
                self.add_assign_index_number(slot, &index, right)?;
            }
            Instruction::AppendAssign(slot) => {
                let item = self.pop_stack()?;
                let slot_name = &self.chunk.slot_names[slot];
                self.slots.ensure_assignable(slot, &self.chunk.slot_names)?;
                let current =
                    self.slots
                        .get_mut(slot)
                        .ok_or_else(|| RuntimeError::UndefinedVariable {
                            name: slot_name.text.to_string(),
                        })?;
                let value = if let Value::List(items) = current {
                    let values = items.make_mut();
                    if values.len() == values.capacity() {
                        values.reserve(1);
                    }
                    values.push(item);
                    Value::List(items.clone())
                } else {
                    add_values(current.clone(), Value::List(vec![item].into()))?
                };
                self.record_assignment(slot);
                self.last_value = Some(value);
            }
            Instruction::Submit => {
                if self.mode == VmMode::Process {
                    return Err(RuntimeError::ForegroundControlInsideProcess { keyword: "submit" });
                }
                return Ok(Some(VmStep::Effect(VmEffect::Submit)));
            }
            Instruction::ProcessSleepFor => {
                if self.mode != VmMode::Process {
                    return Err(RuntimeError::ProcessControlOutsideProcess { keyword: "sleep" });
                }
                return Ok(Some(VmStep::Effect(VmEffect::ProcessSleep(
                    ProcessSleepKind::For,
                ))));
            }
            Instruction::ProcessSleepUntil => {
                if self.mode != VmMode::Process {
                    return Err(RuntimeError::ProcessControlOutsideProcess { keyword: "sleep" });
                }
                return Ok(Some(VmStep::Effect(VmEffect::ProcessSleep(
                    ProcessSleepKind::Until,
                ))));
            }
            Instruction::ProcessWaitSignal => {
                if self.mode != VmMode::Process {
                    return Err(RuntimeError::ProcessControlOutsideProcess {
                        keyword: "wait signal",
                    });
                }
                return Ok(Some(VmStep::Effect(VmEffect::WaitSignal)));
            }
            Instruction::ProcessSignalRun => {
                if self.mode != VmMode::Process {
                    return Err(RuntimeError::ProcessControlOutsideProcess {
                        keyword: "signal run",
                    });
                }
                return Ok(Some(VmStep::Effect(VmEffect::SignalRun)));
            }
            Instruction::ProcessYield => {
                if self.mode != VmMode::Process {
                    return Err(RuntimeError::ProcessControlOutsideProcess { keyword: "yield" });
                }
                return Ok(Some(VmStep::Effect(VmEffect::ProcessEvent(
                    ProcessEventKind::Yield,
                ))));
            }
            Instruction::ProcessWake => {
                if self.mode != VmMode::Process {
                    return Err(RuntimeError::ProcessControlOutsideProcess { keyword: "wake" });
                }
                return Ok(Some(VmStep::Effect(VmEffect::ProcessEvent(
                    ProcessEventKind::Wake,
                ))));
            }
            Instruction::ProcessFinish => {
                if self.mode != VmMode::Process {
                    return Err(RuntimeError::ProcessControlOutsideProcess { keyword: "finish" });
                }
                return Ok(Some(VmStep::Effect(VmEffect::Finish)));
            }
            Instruction::ProcessFail => {
                if self.mode != VmMode::Process {
                    return Err(RuntimeError::ProcessControlOutsideProcess { keyword: "fail" });
                }
                return Ok(Some(VmStep::Effect(VmEffect::Fail)));
            }
            Instruction::Pop => {
                self.last_value = Some(self.pop_stack()?);
            }
            Instruction::BeginRangeIter { binding, argc } => {
                let start_index = self.stack_drain_start(argc)?;
                if self.stack[start_index..]
                    .iter()
                    .any(|value| matches!(value, Value::Projected(_)))
                {
                    return Ok(None);
                }
                let (start, end, step) = range_bounds(&self.stack[start_index..])?;
                self.stack.truncate(start_index);
                if range_has_next(start, end, step) {
                    self.slots
                        .ensure_assignable(binding, &self.chunk.slot_names)?;
                }
                self.iter_stack.push(IterState {
                    cursor: IterCursor::Range {
                        next: start,
                        end,
                        step,
                    },
                    binding,
                    restore: self.slots.capture_temporary(binding),
                });
            }
            Instruction::LoweredLoop(index) => {
                if !self.execute_lowered_loop_direct(&self.chunk.lowered_loops[index])? {
                    return Ok(None);
                }
            }
            Instruction::IterNext { jump_to } => {
                let Some(iter_state) = self.iter_stack.last_mut() else {
                    return Err(RuntimeError::ValueError {
                        message: "missing loop state".to_string(),
                    });
                };
                let Some(value) = iter_state.cursor.next_value() else {
                    self.ip = jump_to;
                    return Ok(Some(VmStep::Continue));
                };
                self.slots.assign_loop_binding(iter_state.binding, value);
            }
            Instruction::EndIter => {
                if let Some(iter_state) = self.iter_stack.pop() {
                    self.slots
                        .restore_temporary(iter_state.binding, iter_state.restore);
                }
            }
            _ => return Ok(None),
        }
        Ok(Some(VmStep::Continue))
    }

    #[inline(always)]
    fn add_assign_value(&mut self, slot: usize, right: Value) -> Result<(), RuntimeError> {
        let slot_name = &self.chunk.slot_names[slot];
        self.slots.ensure_assignable(slot, &self.chunk.slot_names)?;
        let value = {
            let left = self
                .slots
                .get_mut(slot)
                .ok_or_else(|| RuntimeError::UndefinedVariable {
                    name: slot_name.text.to_string(),
                })?;
            match (left, right) {
                (Value::Number(left), Value::Number(right)) => {
                    *left += right;
                    Value::Number(*left)
                }
                (left, right) => {
                    let value = add_values(left.clone(), right)?;
                    *left = value.clone();
                    value
                }
            }
        };
        self.record_assignment(slot);
        self.last_value = Some(value);
        Ok(())
    }

    #[inline(always)]
    fn add_assign_index_number(
        &mut self,
        slot: usize,
        index: &Value,
        right: f64,
    ) -> Result<(), RuntimeError> {
        let slot_name = &self.chunk.slot_names[slot];
        self.slots.ensure_assignable(slot, &self.chunk.slot_names)?;
        let root = self
            .slots
            .get_mut(slot)
            .ok_or_else(|| RuntimeError::UndefinedVariable {
                name: slot_name.text.to_string(),
            })?;
        let value = add_assign_index_number(root, index, right)?;
        self.record_assignment(slot);
        self.last_value = Some(value);
        Ok(())
    }

    #[inline(always)]
    fn append_assign_value(&mut self, slot: usize, item: Value) -> Result<(), RuntimeError> {
        let slot_name = &self.chunk.slot_names[slot];
        self.slots.ensure_assignable(slot, &self.chunk.slot_names)?;
        let current = self
            .slots
            .get_mut(slot)
            .ok_or_else(|| RuntimeError::UndefinedVariable {
                name: slot_name.text.to_string(),
            })?;
        let value = if let Value::List(items) = current {
            let values = items.make_mut();
            if values.len() == values.capacity() {
                values.reserve(1);
            }
            values.push(item);
            Value::List(items.clone())
        } else {
            let value = add_values(current.clone(), Value::List(vec![item].into()))?;
            *current = value.clone();
            value
        };
        self.record_assignment(slot);
        self.last_value = Some(value);
        Ok(())
    }

    #[inline(always)]
    async fn step_instruction(&mut self, instruction: Instruction) -> Result<VmStep, RuntimeError> {
        match instruction {
            Instruction::PushConst(index) => {
                self.stack.push(self.chunk.constants[index].clone());
            }
            Instruction::PushNull => {
                self.stack.push(Value::Null);
            }
            Instruction::PushBool(value) => {
                self.stack.push(Value::Bool(value));
            }
            Instruction::PushNumber(value) => {
                self.stack.push(Value::Number(value));
            }
            Instruction::LoadName(name) => {
                let value = self.load_slot(name)?.clone();
                self.stack.push(value);
            }
            Instruction::StoreName(name) => {
                let value = self.pop_stack()?;
                self.slots
                    .assign(name, value.clone(), &self.chunk.slot_names)?;
                self.record_assignment(name);
                self.last_value = Some(value);
            }
            Instruction::StoreConst { slot, constant } => {
                let value = self.chunk.constants[constant].clone();
                self.slots
                    .assign(slot, value.clone(), &self.chunk.slot_names)?;
                self.record_assignment(slot);
                self.last_value = Some(value);
            }
            Instruction::LoadField { slot, field } => {
                let value = self.load_slot(slot)?;
                let field = &self.chunk.names[field];
                let value = match value {
                    Value::Projected(projected) => {
                        let parent_name = projected.name().to_string();
                        let inner = projected.get_field(field).await?;
                        ProjectedValue::propagate_field(&parent_name, &field.text, inner)
                    }
                    value => read_field_ref_direct(value, field)?,
                };
                self.stack.push(value);
            }
            Instruction::LoadFieldUnwrap { slot, field } => {
                let value = self.load_slot(slot)?;
                let field = &self.chunk.names[field];
                let value = match value {
                    Value::Projected(projected) => {
                        let parent_name = projected.name().to_string();
                        let inner = projected.get_field(field).await?;
                        ProjectedValue::propagate_field(&parent_name, &field.text, inner)
                    }
                    value => read_field_ref_direct(value, field)?,
                };
                self.stack.push(unwrap_tool_result(value)?);
            }
            Instruction::BuildList(len) => {
                let values = self.pop_n(len)?;
                self.stack.push(Value::List(values.into()));
            }
            Instruction::BuildRecord(keys) => {
                let record = self.drain_record_from_stack(keys)?;
                self.stack.push(Value::Record(Arc::new(record)));
            }
            Instruction::Field(field) => {
                let target = self.pop_stack()?;
                let field = &self.chunk.names[field];
                let value = match target {
                    Value::Projected(projected) => {
                        let parent_name = projected.name().to_string();
                        let inner = projected.get_field(field).await?;
                        ProjectedValue::propagate_field(&parent_name, &field.text, inner)
                    }
                    target => read_field_direct(target, field)?,
                };
                self.stack.push(value);
            }
            Instruction::Index => {
                let index = self.pop_stack()?;
                let target = self.pop_stack()?;
                let value = match target {
                    Value::Projected(projected) => {
                        let parent_name = projected.name().to_string();
                        let inner = projected.get_index(&index).await?;
                        ProjectedValue::propagate_index(&parent_name, &index, inner)
                    }
                    target => read_index_direct(target, index)?,
                };
                self.stack.push(value);
            }
            Instruction::PathAssign { slot, path } => {
                let value = self.pop_stack()?;
                let last_value = value.clone();
                let path = &self.chunk.assign_paths[path];
                let index_start = self.stack_drain_start(path.dynamic_index_count)?;
                let indexes = &self.stack[index_start..];
                let root_name = &self.chunk.slot_names[slot];
                self.slots.ensure_assignable(slot, &self.chunk.slot_names)?;
                let root =
                    self.slots
                        .get_mut(slot)
                        .ok_or_else(|| RuntimeError::UndefinedVariable {
                            name: root_name.text.to_string(),
                        })?;
                assign_path(root, path, indexes, value, &self.chunk.names)?;
                self.stack.truncate(index_start);
                self.record_assignment(slot);
                self.last_value = Some(last_value);
            }
            Instruction::ResultUnwrap => {
                let value = self.pop_stack()?;
                self.stack.push(unwrap_tool_result(value)?);
            }
            Instruction::Unary(op) => {
                let value = self.pop_stack()?;
                let value = match op {
                    UnaryOp::Negate => {
                        let value = materialize_projected_async(value).await;
                        Value::Number(-as_number(&value)?)
                    }
                    UnaryOp::Not => Value::Bool(match &value {
                        Value::Projected(_) => !is_truthy_async(&value).await,
                        _ => !is_truthy(&value),
                    }),
                };
                self.stack.push(value);
            }
            Instruction::Binary(op) => {
                let right = self.pop_stack()?;
                let left = self.pop_stack()?;
                let value = match (left, right) {
                    (Value::Number(left), Value::Number(right)) => {
                        eval_number_binary_values(left, op, right)
                    }
                    (left, right) => {
                        let has_projected = matches!(left, Value::Projected(_))
                            || matches!(right, Value::Projected(_));
                        if has_projected {
                            eval_binary_values_async(left, op, right).await?
                        } else {
                            eval_binary_values(left, op, right)?
                        }
                    }
                };
                self.stack.push(value);
            }
            Instruction::SlotNumberBinary { slot, op, right } => {
                let value = match self.load_slot(slot)? {
                    Value::Number(left) => {
                        Value::Number(eval_number_numeric_binary_value(*left, op, right))
                    }
                    left => {
                        eval_binary_values_async(left.clone(), op, Value::Number(right)).await?
                    }
                };
                self.stack.push(value);
            }
            Instruction::SlotNumberCompare { slot, op, right } => {
                let value = match self.load_slot(slot)? {
                    Value::Number(left) => {
                        Value::Bool(eval_number_compare_values(*left, op, right))
                    }
                    left => Value::Bool(
                        eval_compare_values_async(left.clone(), op, Value::Number(right)).await?,
                    ),
                };
                self.stack.push(value);
            }
            Instruction::SlotNumberBinaryCompare {
                slot,
                binary_op,
                binary_right,
                compare_op,
                compare_right,
            } => {
                let value = match self.load_slot(slot)? {
                    Value::Number(left) => {
                        let value =
                            eval_number_numeric_binary_value(*left, binary_op, binary_right);
                        Value::Bool(eval_number_compare_values(value, compare_op, compare_right))
                    }
                    left => {
                        let value = eval_binary_values_async(
                            left.clone(),
                            binary_op,
                            Value::Number(binary_right),
                        )
                        .await?;
                        Value::Bool(
                            eval_compare_values_async(
                                value,
                                compare_op,
                                Value::Number(compare_right),
                            )
                            .await?,
                        )
                    }
                };
                self.stack.push(value);
            }
            Instruction::ToBool => {
                let value = self.pop_stack()?;
                let truthy = match &value {
                    Value::Projected(_) => is_truthy_async(&value).await,
                    _ => is_truthy(&value),
                };
                self.stack.push(Value::Bool(truthy));
            }
            Instruction::Jump(target) => self.ip = target,
            Instruction::JumpIfFalse(target) => {
                let value = self.pop_stack()?;
                let truthy = match &value {
                    Value::Projected(_) => is_truthy_async(&value).await,
                    _ => is_truthy(&value),
                };
                if !truthy {
                    self.ip = target;
                }
            }
            Instruction::JumpIfCompareFalse { op, target } => {
                let right = self.pop_stack()?;
                let left = self.pop_stack()?;
                if !eval_compare_values_async(left, op, right).await? {
                    self.ip = target;
                }
            }
            Instruction::JumpIfSlotNumberCompareFalse {
                slot,
                op,
                right,
                target,
            } => {
                let value = self.load_slot(slot)?;
                let truthy = match value {
                    Value::Number(left) => eval_number_compare_values(*left, op, right),
                    value => {
                        eval_compare_values_async(value.clone(), op, Value::Number(right)).await?
                    }
                };
                if !truthy {
                    self.ip = target;
                }
            }
            Instruction::JumpIfSlotNumberBinaryCompareFalse {
                slot,
                binary_op,
                binary_right,
                compare_op,
                compare_right,
                target,
            } => {
                let value = self.load_slot(slot)?;
                let truthy = match value {
                    Value::Number(left) => {
                        let value =
                            eval_number_numeric_binary_value(*left, binary_op, binary_right);
                        eval_number_compare_values(value, compare_op, compare_right)
                    }
                    value => {
                        let value = eval_binary_values_async(
                            value.clone(),
                            binary_op,
                            Value::Number(binary_right),
                        )
                        .await?;
                        eval_compare_values_async(value, compare_op, Value::Number(compare_right))
                            .await?
                    }
                };
                if !truthy {
                    self.ip = target;
                }
            }
            Instruction::JumpIfTrue(target) => {
                let value = self.pop_stack()?;
                let truthy = match &value {
                    Value::Projected(_) => is_truthy_async(&value).await,
                    _ => is_truthy(&value),
                };
                if truthy {
                    self.ip = target;
                }
            }
            Instruction::ResourceCall { operation, argc } => {
                return Ok(VmStep::Effect(VmEffect::ResourceCall { operation, argc }));
            }
            Instruction::ResourceCallUnwrap { operation, argc } => {
                return Ok(VmStep::Effect(VmEffect::ResourceCallUnwrap {
                    operation,
                    argc,
                }));
            }
            Instruction::StartProcess { process, keys } => {
                return Ok(VmStep::Effect(VmEffect::StartProcess { process, keys }));
            }
            Instruction::AwaitHandle => {
                return Ok(VmStep::Effect(VmEffect::AwaitHandle));
            }
            Instruction::ProcessSleepFor => {
                if self.mode != VmMode::Process {
                    return Err(RuntimeError::ProcessControlOutsideProcess { keyword: "sleep" });
                }
                return Ok(VmStep::Effect(VmEffect::ProcessSleep(
                    ProcessSleepKind::For,
                )));
            }
            Instruction::ProcessSleepUntil => {
                if self.mode != VmMode::Process {
                    return Err(RuntimeError::ProcessControlOutsideProcess { keyword: "sleep" });
                }
                return Ok(VmStep::Effect(VmEffect::ProcessSleep(
                    ProcessSleepKind::Until,
                )));
            }
            Instruction::ProcessWaitSignal => {
                if self.mode != VmMode::Process {
                    return Err(RuntimeError::ProcessControlOutsideProcess {
                        keyword: "wait signal",
                    });
                }
                return Ok(VmStep::Effect(VmEffect::WaitSignal));
            }
            Instruction::ProcessSignalRun => {
                if self.mode != VmMode::Process {
                    return Err(RuntimeError::ProcessControlOutsideProcess {
                        keyword: "signal run",
                    });
                }
                return Ok(VmStep::Effect(VmEffect::SignalRun));
            }
            Instruction::AwaitHandleUnwrap => {
                return Ok(VmStep::Effect(VmEffect::AwaitHandleUnwrap));
            }
            Instruction::CancelHandle => {
                return Ok(VmStep::Effect(VmEffect::CancelHandle));
            }
            Instruction::Intrinsic(op) => {
                self.execute_intrinsic_instruction(op).await?;
            }
            Instruction::AddAssign(slot) => {
                let right = self.pop_stack()?;
                let slot_name = &self.chunk.slot_names[slot];
                self.slots.ensure_assignable(slot, &self.chunk.slot_names)?;
                let value = {
                    let left = self.slots.get_mut(slot).ok_or_else(|| {
                        RuntimeError::UndefinedVariable {
                            name: slot_name.text.to_string(),
                        }
                    })?;
                    match (left, right) {
                        (Value::Number(left), Value::Number(right)) => {
                            *left += right;
                            Value::Number(*left)
                        }
                        (left, right) => {
                            let value = add_values(left.clone(), right)?;
                            *left = value.clone();
                            value
                        }
                    }
                };
                self.record_assignment(slot);
                self.last_value = Some(value);
            }
            Instruction::AddAssignNumber { slot, right } => {
                let slot_name = &self.chunk.slot_names[slot];
                self.slots.ensure_assignable(slot, &self.chunk.slot_names)?;
                let value = {
                    let left = self.slots.get_mut(slot).ok_or_else(|| {
                        RuntimeError::UndefinedVariable {
                            name: slot_name.text.to_string(),
                        }
                    })?;
                    match left {
                        Value::Number(left) => {
                            *left += right;
                            Value::Number(*left)
                        }
                        left => {
                            let value = add_values(left.clone(), Value::Number(right))?;
                            *left = value.clone();
                            value
                        }
                    }
                };
                self.record_assignment(slot);
                self.last_value = Some(value);
            }
            Instruction::AddAssignSlot { slot, right } => {
                let slot_name = &self.chunk.slot_names[slot];
                let right = match self.load_slot(right)? {
                    Value::Number(value) => AddAssignRight::Number(*value),
                    value => AddAssignRight::Value(value.clone()),
                };
                self.slots.ensure_assignable(slot, &self.chunk.slot_names)?;
                let value = {
                    let left = self.slots.get_mut(slot).ok_or_else(|| {
                        RuntimeError::UndefinedVariable {
                            name: slot_name.text.to_string(),
                        }
                    })?;
                    match (left, right) {
                        (Value::Number(left), AddAssignRight::Number(right)) => {
                            *left += right;
                            Value::Number(*left)
                        }
                        (left, AddAssignRight::Number(right)) => {
                            let value = add_values(left.clone(), Value::Number(right))?;
                            *left = value.clone();
                            value
                        }
                        (left, AddAssignRight::Value(right)) => {
                            let value = add_values(left.clone(), right)?;
                            *left = value.clone();
                            value
                        }
                    }
                };
                self.record_assignment(slot);
                self.last_value = Some(value);
            }
            Instruction::AddAssignIndexNumber { slot, right } => {
                let index = self.pop_stack()?;
                let slot_name = &self.chunk.slot_names[slot];
                self.slots.ensure_assignable(slot, &self.chunk.slot_names)?;
                let root =
                    self.slots
                        .get_mut(slot)
                        .ok_or_else(|| RuntimeError::UndefinedVariable {
                            name: slot_name.text.to_string(),
                        })?;
                let value = add_assign_index_number(root, &index, right)?;
                self.record_assignment(slot);
                self.last_value = Some(value);
            }
            Instruction::AddAssignIndexSlotNumber { slot, index, right } => {
                let index = self.load_slot(index)?.clone();
                let slot_name = &self.chunk.slot_names[slot];
                self.slots.ensure_assignable(slot, &self.chunk.slot_names)?;
                let root =
                    self.slots
                        .get_mut(slot)
                        .ok_or_else(|| RuntimeError::UndefinedVariable {
                            name: slot_name.text.to_string(),
                        })?;
                let value = add_assign_index_number(root, &index, right)?;
                self.record_assignment(slot);
                self.last_value = Some(value);
            }
            Instruction::AppendAssign(slot) => {
                let item = self.pop_stack()?;
                let slot_name = &self.chunk.slot_names[slot];
                self.slots.ensure_assignable(slot, &self.chunk.slot_names)?;
                let current =
                    self.slots
                        .get_mut(slot)
                        .ok_or_else(|| RuntimeError::UndefinedVariable {
                            name: slot_name.text.to_string(),
                        })?;
                if let Value::List(items) = current {
                    let values = items.make_mut();
                    if values.len() == values.capacity() {
                        values.reserve(1);
                    }
                    values.push(item);
                    let value = Value::List(items.clone());
                    self.record_assignment(slot);
                    self.last_value = Some(value);
                    return Ok(VmStep::Continue);
                }
                let current = current.clone();
                let value = add_values(current, Value::List(vec![item].into()))?;
                self.slots
                    .assign(slot, value.clone(), &self.chunk.slot_names)?;
                self.record_assignment(slot);
                self.last_value = Some(value);
            }
            Instruction::Print => {
                if self.mode == VmMode::Process {
                    return Err(RuntimeError::ForegroundControlInsideProcess { keyword: "print" });
                }
                return Ok(VmStep::Effect(VmEffect::Print));
            }
            Instruction::Submit => {
                if self.mode == VmMode::Process {
                    return Err(RuntimeError::ForegroundControlInsideProcess { keyword: "submit" });
                }
                return Ok(VmStep::Effect(VmEffect::Submit));
            }
            Instruction::ProcessYield => {
                if self.mode != VmMode::Process {
                    return Err(RuntimeError::ProcessControlOutsideProcess { keyword: "yield" });
                }
                return Ok(VmStep::Effect(VmEffect::ProcessEvent(
                    ProcessEventKind::Yield,
                )));
            }
            Instruction::ProcessWake => {
                if self.mode != VmMode::Process {
                    return Err(RuntimeError::ProcessControlOutsideProcess { keyword: "wake" });
                }
                return Ok(VmStep::Effect(VmEffect::ProcessEvent(
                    ProcessEventKind::Wake,
                )));
            }
            Instruction::ProcessFinish => {
                if self.mode != VmMode::Process {
                    return Err(RuntimeError::ProcessControlOutsideProcess { keyword: "finish" });
                }
                return Ok(VmStep::Effect(VmEffect::Finish));
            }
            Instruction::ProcessFail => {
                if self.mode != VmMode::Process {
                    return Err(RuntimeError::ProcessControlOutsideProcess { keyword: "fail" });
                }
                return Ok(VmStep::Effect(VmEffect::Fail));
            }
            Instruction::Pop => {
                self.last_value = Some(self.pop_stack()?);
            }
            Instruction::BeginIter(binding) => {
                let iterable = self.pop_stack()?;
                let values = iterable_values(iterable).await?;
                if !values.is_empty() {
                    self.slots
                        .ensure_assignable(binding, &self.chunk.slot_names)?;
                }
                self.iter_stack.push(IterState {
                    cursor: IterCursor::List { values, index: 0 },
                    binding,
                    restore: self.slots.capture_temporary(binding),
                });
            }
            Instruction::BeginRangeIter { binding, argc } => {
                let start_index = self.stack_drain_start(argc)?;
                let (start, end, step) = range_bounds_async(&self.stack[start_index..]).await?;
                self.stack.truncate(start_index);
                if range_has_next(start, end, step) {
                    self.slots
                        .ensure_assignable(binding, &self.chunk.slot_names)?;
                }
                self.iter_stack.push(IterState {
                    cursor: IterCursor::Range {
                        next: start,
                        end,
                        step,
                    },
                    binding,
                    restore: self.slots.capture_temporary(binding),
                });
            }
            Instruction::LoweredLoop(index) => {
                self.execute_lowered_loop(&self.chunk.lowered_loops[index])
                    .await?;
            }
            Instruction::IterNext { jump_to } => {
                let Some(iter_state) = self.iter_stack.last_mut() else {
                    return Err(RuntimeError::ValueError {
                        message: "missing loop state".to_string(),
                    });
                };
                let Some(value) = iter_state.cursor.next_value() else {
                    self.ip = jump_to;
                    return Ok(VmStep::Continue);
                };
                self.slots.assign_loop_binding(iter_state.binding, value);
            }
            Instruction::EndIter => {
                if let Some(iter_state) = self.iter_stack.pop() {
                    self.slots
                        .restore_temporary(iter_state.binding, iter_state.restore);
                }
            }
            Instruction::ResolveTypeRef(slot) => {
                let slot_name = &self.chunk.slot_names[slot];
                let value = self.slots.get(slot).cloned().ok_or_else(|| {
                    RuntimeError::UndefinedVariable {
                        name: slot_name.text.to_string(),
                    }
                })?;
                let schema =
                    unwrap_type_value(&value)
                        .cloned()
                        .ok_or_else(|| RuntimeError::TypeError {
                            message: format!(
                                "`{}` is not a Type value (missing `{LASH_TYPE_KEY}`)",
                                slot_name.text
                            ),
                        })?;
                self.stack.push(schema);
            }
            Instruction::WrapTypeLiteral => {
                let schema = self.pop_stack()?;
                let mut wrapper = record_with_capacity(1);
                wrapper.insert(LASH_TYPE_KEY.to_string(), schema);
                self.stack.push(Value::Record(Arc::new(wrapper)));
            }
        }
        Ok(VmStep::Continue)
    }

    async fn execute_intrinsic_instruction(&mut self, op: IntrinsicOp) -> Result<(), RuntimeError> {
        let start = self.profile.as_ref().map(|_| Instant::now());
        match op {
            IntrinsicOp::Validate => {
                let schema = self.pop_stack()?;
                let value = self.pop_stack()?;
                let schema = materialize_projected_async(schema).await;
                let value = self
                    .execute_dynamic_validate(materialize_projected_async(value).await, &schema)?;
                self.stack.push(value);
            }
            IntrinsicOp::ValidateCompiled(schema) => {
                let value = self.pop_stack()?;
                let value = execute_validation_plan(
                    materialize_projected_async(value).await,
                    &self.chunk.compiled_schemas[schema],
                )?;
                self.stack.push(value);
            }
            IntrinsicOp::PushAssign(slot) => {
                let item = materialize_projected_async(self.pop_stack()?).await;
                let slot_name = &self.chunk.slot_names[slot];
                self.slots.ensure_assignable(slot, &self.chunk.slot_names)?;
                let current =
                    self.slots
                        .get_mut(slot)
                        .ok_or_else(|| RuntimeError::UndefinedVariable {
                            name: slot_name.text.to_string(),
                        })?;
                let value = if let Value::List(items) = current {
                    let values = items.make_mut();
                    if values.len() == values.capacity() {
                        values.reserve(1);
                    }
                    values.push(item);
                    Value::List(items.clone())
                } else {
                    execute_push_builtin_async(current.clone(), item).await?
                };
                self.slots
                    .assign(slot, value.clone(), &self.chunk.slot_names)?;
                self.record_assignment(slot);
                self.last_value = Some(value);
            }
            IntrinsicOp::FormatCompiled(template) => {
                let template = &self.chunk.format_templates[template];
                let argc = template.argc;
                let values = self.stack_tail(argc)?;
                let value = if values
                    .iter()
                    .any(|value| matches!(value, Value::Projected(_)))
                {
                    execute_compiled_format(template, values).await?
                } else {
                    execute_compiled_format_direct(template, values)?
                };
                self.stack.truncate(self.stack.len() - argc);
                self.stack.push(Value::String(value.into()));
            }
            IntrinsicOp::FormatCompiledSlotNumber { template, slot } => {
                let template = &self.chunk.format_templates[template];
                let value = match self.load_slot(slot)? {
                    Value::Number(value) => Value::String(
                        execute_compiled_format_one_number_compact_direct(template, *value)?,
                    ),
                    value => {
                        let value = if matches!(value, Value::Projected(_)) {
                            execute_compiled_format(template, std::slice::from_ref(value)).await?
                        } else {
                            execute_compiled_format_direct(template, std::slice::from_ref(value))?
                        };
                        Value::String(value.into())
                    }
                };
                self.stack.push(value);
            }
            IntrinsicOp::FormatCompiledSlotNumberBinary {
                template,
                slot,
                op,
                right,
            } => {
                let template = &self.chunk.format_templates[template];
                let value = match self.load_slot(slot)? {
                    Value::Number(left) => {
                        Value::String(execute_compiled_format_one_number_compact_direct(
                            template,
                            eval_number_numeric_binary_value(*left, op, right),
                        )?)
                    }
                    left => {
                        let value =
                            eval_binary_values_async(left.clone(), op, Value::Number(right))
                                .await?;
                        let value = if matches!(value, Value::Projected(_)) {
                            execute_compiled_format(template, &[value]).await?
                        } else {
                            execute_compiled_format_direct(template, &[value])?
                        };
                        Value::String(value.into())
                    }
                };
                self.stack.push(value);
            }
            _ => {
                let argc = op.argc();
                let values = self.stack_tail(argc)?;
                let value = execute_intrinsic(op, &self.chunk.names, values).await?;
                self.stack.truncate(self.stack.len() - argc);
                self.stack.push(value);
            }
        }
        if let Some(start) = start {
            self.record_builtin_profile(op, start.elapsed().as_nanos());
        }
        Ok(())
    }

    fn pop_stack(&mut self) -> Result<Value, RuntimeError> {
        self.stack.pop().ok_or_else(|| RuntimeError::ValueError {
            message: "vm stack underflow".to_string(),
        })
    }

    fn load_slot(&self, slot: usize) -> Result<&Value, RuntimeError> {
        self.slots
            .get(slot)
            .ok_or_else(|| RuntimeError::UndefinedVariable {
                name: self.chunk.slot_names[slot].text.to_string(),
            })
    }

    fn drain_record_from_stack(&mut self, keys: usize) -> Result<Record, RuntimeError> {
        let key_indices = &self.chunk.key_lists[keys];
        let start = self.stack_drain_start(key_indices.len())?;
        let mut record = record_with_capacity(key_indices.len());
        for (key, value) in key_indices.iter().zip(self.stack.drain(start..)) {
            let name_entry = &self.chunk.names[*key];
            record.insert_symbolized(name_entry.symbol, name_entry.text.clone(), value);
        }
        Ok(record)
    }

    fn drain_receiver_call(&mut self, argc: usize) -> Result<(Value, Vec<Value>), RuntimeError> {
        let start = self.stack_drain_start(argc + 1)?;
        let mut values = self.stack.drain(start..).collect::<Vec<_>>();
        let receiver = values.remove(0);
        Ok((receiver, values))
    }

    fn pop_n(&mut self, len: usize) -> Result<Vec<Value>, RuntimeError> {
        if self.stack.len() < len {
            return Err(RuntimeError::ValueError {
                message: "vm stack underflow".to_string(),
            });
        }
        let start = self.stack.len() - len;
        Ok(self.stack.split_off(start))
    }

    fn stack_tail(&self, len: usize) -> Result<&[Value], RuntimeError> {
        if self.stack.len() < len {
            return Err(RuntimeError::ValueError {
                message: "vm stack underflow".to_string(),
            });
        }
        Ok(&self.stack[self.stack.len() - len..])
    }

    fn stack_drain_start(&self, len: usize) -> Result<usize, RuntimeError> {
        self.stack
            .len()
            .checked_sub(len)
            .ok_or_else(|| RuntimeError::ValueError {
                message: "vm stack underflow".to_string(),
            })
    }

    fn unwind_iterators(&mut self) {
        while let Some(iter_state) = self.iter_stack.pop() {
            self.slots
                .restore_temporary(iter_state.binding, iter_state.restore);
        }
    }

    fn record_assignment(&mut self, _slot: usize) {}

    pub(crate) fn into_globals(self) -> Record {
        self.slots.into_globals(&self.chunk.slot_names)
    }

    pub(crate) fn recycle_into_globals(mut self, scratch: &mut ExecutionScratch) -> Record {
        self.stack.clear();
        self.iter_stack.clear();
        scratch.stack = std::mem::take(&mut self.stack);
        scratch.iter_stack = std::mem::take(&mut self.iter_stack);
        self.slots
            .recycle_into_globals(&self.chunk.slot_names, &mut scratch.slot_values)
    }

    fn record_instruction_profile(&mut self, tag: InstructionProfileTag, elapsed_ns: u128) {
        let Some(profile) = &mut self.profile else {
            return;
        };
        let index = tag as usize;
        profile.instruction_counts[index] += 1;
        profile.instruction_times[index] += elapsed_ns;
    }

    fn record_builtin_profile(&mut self, builtin: IntrinsicOp, elapsed_ns: u128) {
        let Some(profile) = &mut self.profile else {
            return;
        };
        let index = builtin.profile_tag() as usize;
        profile.builtin_counts[index] += 1;
        profile.builtin_times[index] += elapsed_ns;
    }

    pub(crate) fn take_profile(&mut self) -> ProfileReport {
        let Some(profile) = self.profile.take() else {
            return ProfileReport::default();
        };
        profile.finish()
    }
}

pub(crate) struct IterState {
    cursor: IterCursor,
    binding: usize,
    restore: LoopRestore,
}

enum IterCursor {
    List { values: ListValue, index: usize },
    Range { next: i64, end: i64, step: i64 },
}

impl IterCursor {
    fn next_value(&mut self) -> Option<Value> {
        match self {
            Self::List { values, index } => {
                let value = values.get(*index)?.clone();
                *index += 1;
                Some(value)
            }
            Self::Range { next, end, step } => {
                if (*step > 0 && *next >= *end) || (*step < 0 && *next <= *end) {
                    return None;
                }
                let value = *next;
                *next = (*next).saturating_add(*step);
                Some(Value::Number(value as f64))
            }
        }
    }
}

struct LoopRestore {
    previous: Option<Value>,
}
