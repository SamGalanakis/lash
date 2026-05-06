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

use futures_util::{FutureExt as _, future::join_all};
use smallvec::SmallVec;

use crate::ast::UnaryOp;

use super::host::{ToolHostCall, ToolHostError};
use super::instruction::{NamedParallelCallBranch, ParallelCallBranch};
use super::record::{Record, record_with_capacity};
use super::schema::{execute_compiled_validate, execute_validate_builtin};
use super::value::ProjectedValue;
use super::{
    Builtin, COOPERATIVE_YIELD_INSTRUCTION_BUDGET, Chunk, ExecutionOutcome, ExecutionScratch,
    Instruction, InstructionProfileTag, LASH_TYPE_KEY, Name, ProfileAccumulator, ProfileReport,
    ProjectedBindings, RuntimeError, RuntimeFailure, ToolHost, Value, add_assign_index_number,
    add_values, as_number, assign_path, eval_binary_values, eval_binary_values_async,
    eval_compare_values_async, eval_number_binary_values, eval_number_compare_values,
    eval_number_numeric_binary_value, eval_pure_expr, error_value, execute_builtin,
    execute_compiled_format, execute_compiled_format_direct, execute_join_builtin,
    execute_len_builtin, execute_len_direct, execute_push_builtin, execute_range_builtin,
    is_async_handle_record, is_truthy, is_truthy_async, iterable_values, materialize_value,
    range_bounds, read_field_direct, read_field_ref_direct, read_index_direct, success,
    unwrap_tool_result, unwrap_type_value,
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
    in_parallel_branch: bool,
    assigned: Option<Vec<bool>>,
    iter_stack: Vec<IterState>,
    profile: Option<ProfileAccumulator>,
}

enum VmStep {
    Continue,
    Finish(Value),
    Effect(VmEffect),
}

#[derive(Clone, Copy)]
enum VmEffect {
    CallTool { name: usize, keys: usize },
    CallToolUnwrap { name: usize, keys: usize },
    StartCallTool { name: usize, keys: usize },
    AwaitHandle,
    AwaitHandleUnwrap,
    CancelHandle,
    Print,
    ParallelCalls(usize),
    ParallelCallsValue(usize),
    ParallelNamedCallsValue(usize),
    Parallel(usize),
    ParallelValue(usize),
    ParallelNamed(usize),
    ParallelNamedValue(usize),
}

struct VmTrap {
    error: RuntimeError,
    instruction_ip: usize,
}

impl<'a, H: ToolHost> Vm<'a, H> {
    pub(crate) fn new(
        chunk: &'a Chunk,
        slots: SlotState,
        host: &'a H,
        in_parallel_branch: bool,
    ) -> Self {
        Self {
            chunk,
            ip: 0,
            stack: Vec::new(),
            last_value: None,
            slots,
            host,
            in_parallel_branch,
            assigned: in_parallel_branch.then(|| vec![false; chunk.slot_names.len()]),
            iter_stack: Vec::new(),
            profile: None,
        }
    }

    pub(crate) fn new_with_scratch(
        chunk: &'a Chunk,
        slots: SlotState,
        host: &'a H,
        in_parallel_branch: bool,
        scratch: &mut ExecutionScratch,
    ) -> Self {
        Self {
            chunk,
            ip: 0,
            stack: std::mem::take(&mut scratch.stack),
            last_value: None,
            slots,
            host,
            in_parallel_branch,
            assigned: in_parallel_branch.then(|| vec![false; chunk.slot_names.len()]),
            iter_stack: std::mem::take(&mut scratch.iter_stack),
            profile: None,
        }
    }

    pub(crate) fn enable_profile(&mut self) {
        self.profile = Some(ProfileAccumulator::default());
    }

    pub(crate) async fn run(&mut self) -> Result<ExecutionOutcome, RuntimeError> {
        let result = self.run_loop().await.map_err(|trap| trap.error);
        self.unwind_iterators();
        result
    }

    pub(crate) async fn run_traced(&mut self) -> Result<ExecutionOutcome, RuntimeFailure> {
        let result = self.run_loop().await.map_err(|trap| RuntimeFailure {
            error: trap.error,
            span: self.chunk.spans.get(trap.instruction_ip).copied().flatten(),
        });
        self.unwind_iterators();
        result
    }

    async fn run_loop(&mut self) -> Result<ExecutionOutcome, VmTrap> {
        let mut budget = COOPERATIVE_YIELD_INSTRUCTION_BUDGET;
        while let Some(instruction) = self.chunk.code.get(self.ip).copied() {
            let instruction_ip = self.ip;
            self.ip += 1;
            let profile = self
                .profile
                .as_ref()
                .map(|_| (instruction.profile_tag(), Instant::now()));
            let result = match self.step_instruction(instruction).await {
                Ok(VmStep::Continue) => Ok(None),
                Ok(VmStep::Finish(value)) => Ok(Some(ExecutionOutcome::Finished(value))),
                Ok(VmStep::Effect(effect)) => self.resolve_effect(effect).await.map(|()| None),
                Err(error) => Err(error),
            };
            if let Some((tag, start)) = profile {
                self.record_instruction_profile(tag, start.elapsed().as_nanos());
            }
            match result {
                Ok(Some(outcome)) => return Ok(outcome),
                Ok(None) => {}
                Err(error) => {
                    return Err(VmTrap {
                        error,
                        instruction_ip,
                    });
                }
            }
            budget -= 1;
            if budget == 0 {
                self.host.yield_now().await;
                budget = COOPERATIVE_YIELD_INSTRUCTION_BUDGET;
            }
        }
        Ok(ExecutionOutcome::Continued)
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
                    UnaryOp::Negate => Value::Number(-as_number(&value)?),
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
            Instruction::CallTool { name, keys } => {
                return Ok(VmStep::Effect(VmEffect::CallTool { name, keys }));
            }
            Instruction::CallToolUnwrap { name, keys } => {
                return Ok(VmStep::Effect(VmEffect::CallToolUnwrap { name, keys }));
            }
            Instruction::StartCallTool { name, keys } => {
                return Ok(VmStep::Effect(VmEffect::StartCallTool { name, keys }));
            }
            Instruction::AwaitHandle => {
                return Ok(VmStep::Effect(VmEffect::AwaitHandle));
            }
            Instruction::AwaitHandleUnwrap => {
                return Ok(VmStep::Effect(VmEffect::AwaitHandleUnwrap));
            }
            Instruction::CancelHandle => {
                return Ok(VmStep::Effect(VmEffect::CancelHandle));
            }
            Instruction::CallBuiltin { builtin, argc } => {
                let values = self.stack_tail(argc)?;
                let start = self.profile.as_ref().map(|_| Instant::now());
                let value = execute_builtin(builtin, &self.chunk.names, values).await?;
                if let Some(start) = start {
                    self.record_builtin_profile(builtin, start.elapsed().as_nanos());
                }
                self.stack.truncate(self.stack.len() - argc);
                self.stack.push(value);
            }
            Instruction::Len => {
                let value = self.pop_stack()?;
                let start = self.profile.as_ref().map(|_| Instant::now());
                let value = match &value {
                    Value::Projected(_) => execute_len_builtin(&value).await?,
                    _ => execute_len_direct(&value)?,
                };
                if let Some(start) = start {
                    self.record_builtin_profile(Builtin::Len, start.elapsed().as_nanos());
                }
                self.stack.push(value);
            }
            Instruction::Join => {
                let sep = self.pop_stack()?;
                let items = self.pop_stack()?;
                let start = self.profile.as_ref().map(|_| Instant::now());
                let value = execute_join_builtin(&items, &sep)?;
                if let Some(start) = start {
                    self.record_builtin_profile(Builtin::Join, start.elapsed().as_nanos());
                }
                self.stack.push(value);
            }
            Instruction::Validate => {
                let schema = self.pop_stack()?;
                let value = self.pop_stack()?;
                let start = self.profile.as_ref().map(|_| Instant::now());
                let value = execute_validate_builtin(value, &schema)?;
                if let Some(start) = start {
                    self.record_builtin_profile(Builtin::Validate, start.elapsed().as_nanos());
                }
                self.stack.push(value);
            }
            Instruction::ValidateCompiled(schema) => {
                let value = self.pop_stack()?;
                let start = self.profile.as_ref().map(|_| Instant::now());
                let value = execute_compiled_validate(value, &self.chunk.compiled_schemas[schema])?;
                if let Some(start) = start {
                    self.record_builtin_profile(Builtin::Validate, start.elapsed().as_nanos());
                }
                self.stack.push(value);
            }
            Instruction::Push => {
                let item = self.pop_stack()?;
                let list = self.pop_stack()?;
                let start = self.profile.as_ref().map(|_| Instant::now());
                let value = execute_push_builtin(&list, item)?;
                if let Some(start) = start {
                    self.record_builtin_profile(Builtin::Push, start.elapsed().as_nanos());
                }
                self.stack.push(value);
            }
            Instruction::Range { argc } => {
                let start_index = self.stack_drain_start(argc)?;
                let start = self.profile.as_ref().map(|_| Instant::now());
                let value = execute_range_builtin(&self.stack[start_index..])?;
                if let Some(start) = start {
                    self.record_builtin_profile(Builtin::Range, start.elapsed().as_nanos());
                }
                self.stack.truncate(start_index);
                self.stack.push(value);
            }
            Instruction::FormatCompiled(template) => {
                let template = &self.chunk.format_templates[template];
                let values = self.stack_tail(template.argc)?;
                let start = self.profile.as_ref().map(|_| Instant::now());
                let value = if values
                    .iter()
                    .any(|value| matches!(value, Value::Projected(_)))
                {
                    execute_compiled_format(template, values).await?
                } else {
                    execute_compiled_format_direct(template, values)?
                };
                let value = Value::String(value.into());
                if let Some(start) = start {
                    self.record_builtin_profile(Builtin::Format, start.elapsed().as_nanos());
                }
                self.stack.truncate(self.stack.len() - template.argc);
                self.stack.push(value);
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
            Instruction::AppendAssign(slot) => {
                let item = self.pop_stack()?;
                let slot_name = &self.chunk.slot_names[slot];
                self.slots.ensure_assignable(slot, &self.chunk.slot_names)?;
                let current = self.slots.get(slot).cloned().ok_or_else(|| {
                    RuntimeError::UndefinedVariable {
                        name: slot_name.text.to_string(),
                    }
                })?;
                let value = match current {
                    Value::List(items) => {
                        let mut values = Vec::with_capacity(items.len() + 1);
                        values.extend(items.iter().cloned());
                        values.push(item);
                        Value::List(values.into())
                    }
                    other => add_values(other, Value::List(vec![item].into()))?,
                };
                self.slots
                    .assign(slot, value.clone(), &self.chunk.slot_names)?;
                self.record_assignment(slot);
                self.last_value = Some(value);
            }
            Instruction::Print => {
                return Ok(VmStep::Effect(VmEffect::Print));
            }
            Instruction::Submit => {
                if self.in_parallel_branch {
                    return Err(RuntimeError::FinishInsideParallel);
                }
                return Ok(VmStep::Finish(self.pop_stack()?));
            }
            Instruction::Pop => {
                self.last_value = Some(self.pop_stack()?);
            }
            Instruction::BeginIter(binding) => {
                let iterable = self.pop_stack()?;
                let values = iterable_values(iterable).await?;
                self.iter_stack.push(IterState {
                    cursor: IterCursor::List { values, index: 0 },
                    binding,
                    restore: self.slots.capture_temporary(binding),
                });
            }
            Instruction::BeginRangeIter { binding, argc } => {
                let start_index = self.stack_drain_start(argc)?;
                let (start, end) = range_bounds(&self.stack[start_index..])?;
                self.stack.truncate(start_index);
                self.iter_stack.push(IterState {
                    cursor: IterCursor::Range { next: start, end },
                    binding,
                    restore: self.slots.capture_temporary(binding),
                });
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
                self.slots
                    .assign(iter_state.binding, value, &self.chunk.slot_names)?;
            }
            Instruction::EndIter => {
                if let Some(iter_state) = self.iter_stack.pop() {
                    self.slots
                        .restore_temporary(iter_state.binding, iter_state.restore);
                }
            }
            Instruction::ParallelCalls(branches) => {
                return Ok(VmStep::Effect(VmEffect::ParallelCalls(branches)));
            }
            Instruction::ParallelCallsValue(branches) => {
                return Ok(VmStep::Effect(VmEffect::ParallelCallsValue(branches)));
            }
            Instruction::ParallelNamedCallsValue(branches) => {
                return Ok(VmStep::Effect(VmEffect::ParallelNamedCallsValue(branches)));
            }
            Instruction::PureParallelValue(branches) => {
                let value = self.exec_pure_parallel_value(branches)?;
                self.last_value = Some(value.clone());
                self.stack.push(value);
            }
            Instruction::PureParallelNamedValue(branches) => {
                let value = self.exec_pure_parallel_named_value(branches)?;
                self.last_value = Some(value.clone());
                self.stack.push(value);
            }
            Instruction::Parallel(branches) => {
                return Ok(VmStep::Effect(VmEffect::Parallel(branches)));
            }
            Instruction::ParallelValue(branches) => {
                return Ok(VmStep::Effect(VmEffect::ParallelValue(branches)));
            }
            Instruction::ParallelNamed(branches) => {
                return Ok(VmStep::Effect(VmEffect::ParallelNamed(branches)));
            }
            Instruction::ParallelNamedValue(branches) => {
                return Ok(VmStep::Effect(VmEffect::ParallelNamedValue(branches)));
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

    async fn resolve_effect(&mut self, effect: VmEffect) -> Result<(), RuntimeError> {
        match effect {
            VmEffect::CallTool { name, keys } => {
                let args = self.drain_record_from_stack(keys)?;
                let result = match self
                    .host
                    .call(self.chunk.names[name].text.to_string(), args)
                    .await
                {
                    Ok(value) => success(value),
                    Err(error) => error_value(error.to_string()),
                };
                self.stack.push(result);
            }
            VmEffect::CallToolUnwrap { name, keys } => {
                let args = self.drain_record_from_stack(keys)?;
                let value = self
                    .host
                    .call(self.chunk.names[name].text.to_string(), args)
                    .await
                    .map_err(|error| RuntimeError::ValueError {
                        message: format!("`?` unwrapped failed tool result: {error}"),
                    })?;
                self.stack.push(value);
            }
            VmEffect::StartCallTool { name, keys } => {
                let args = self.drain_record_from_stack(keys)?;
                let value = self
                    .host
                    .start_call(self.chunk.names[name].text.to_string(), args)
                    .await
                    .map_err(|err| RuntimeError::ValueError {
                        message: format!("async start failed: {err}"),
                    })?;
                self.stack.push(value);
            }
            VmEffect::AwaitHandle => {
                let handle = self.pop_stack()?;
                let result = self.await_value(handle).await;
                self.stack.push(result);
            }
            VmEffect::AwaitHandleUnwrap => {
                let handle = self.pop_stack()?;
                let result = self.await_value_unwrap(handle).await?;
                self.stack.push(result);
            }
            VmEffect::CancelHandle => {
                let handle = self.pop_stack()?;
                let value = self.host.cancel_handle(handle).await.map_err(|err| {
                    RuntimeError::ValueError {
                        message: format!("cancel failed: {err}"),
                    }
                })?;
                self.last_value = Some(value);
            }
            VmEffect::Print => {
                let value = self.pop_stack()?;
                let host_value = match &value {
                    Value::Projected(projected) => Value::String(projected.render().await.into()),
                    _ => value.clone(),
                };
                self.host
                    .print(host_value)
                    .await
                    .map_err(|err| RuntimeError::ValueError {
                        message: format!("print failed: {err}"),
                    })?;
                self.last_value = Some(value);
            }
            VmEffect::ParallelCalls(branches) => {
                self.exec_parallel_calls(branches).await?;
                self.last_value = Some(Value::Null);
            }
            VmEffect::ParallelCallsValue(branches) => {
                let value = self.exec_parallel_calls_value(branches).await?;
                self.last_value = Some(value.clone());
                self.stack.push(value);
            }
            VmEffect::ParallelNamedCallsValue(branches) => {
                let value = self.exec_parallel_named_calls_value(branches).await?;
                self.last_value = Some(value.clone());
                self.stack.push(value);
            }
            VmEffect::Parallel(branches) => {
                self.exec_parallel(branches).await?;
                self.last_value = Some(Value::Null);
            }
            VmEffect::ParallelValue(branches) => {
                let value = self.exec_parallel_value(branches).await?;
                self.last_value = Some(value.clone());
                self.stack.push(value);
            }
            VmEffect::ParallelNamed(branches) => {
                self.exec_parallel_named(branches).await?;
                self.last_value = Some(Value::Null);
            }
            VmEffect::ParallelNamedValue(branches) => {
                let value = self.exec_parallel_named_value(branches).await?;
                self.last_value = Some(value.clone());
                self.stack.push(value);
            }
        }
        Ok(())
    }

    async fn exec_parallel(&mut self, branches_index: usize) -> Result<(), RuntimeError> {
        let branches = &self.chunk.branch_sets[branches_index];
        if branches.is_empty() {
            return Ok(());
        }

        let base_slots = self.slots.clone();
        let results = join_all(
            branches
                .iter()
                .map(|branch| Self::run_branch(branch, base_slots.clone(), self.host, true)),
        )
        .await;

        let mut merged_names = vec![false; self.chunk.slot_names.len()];
        for result in results {
            self.merge_branch_result(result?, &mut merged_names)?;
        }
        Ok(())
    }

    async fn exec_parallel_value(&mut self, branches_index: usize) -> Result<Value, RuntimeError> {
        let branches = &self.chunk.branch_sets[branches_index];
        if branches.is_empty() {
            return Ok(Value::List(Vec::<Value>::new().into()));
        }

        let base_slots = self.slots.clone();
        let results = join_all(
            branches
                .iter()
                .map(|branch| Self::run_branch(branch, base_slots.clone(), self.host, true)),
        )
        .await;

        let mut outputs = Vec::with_capacity(results.len());
        let mut merged_names = vec![false; self.chunk.slot_names.len()];
        for result in results {
            let result = result?;
            outputs.push(result.output.clone());
            self.merge_branch_result(result, &mut merged_names)?;
        }
        Ok(Value::List(outputs.into()))
    }

    async fn exec_parallel_named(&mut self, branches_index: usize) -> Result<(), RuntimeError> {
        let branches = &self.chunk.named_branch_sets[branches_index];
        if branches.is_empty() {
            return Ok(());
        }

        let base_slots = self.slots.clone();
        let results =
            join_all(branches.iter().map(|branch| {
                Self::run_branch(&branch.chunk, base_slots.clone(), self.host, true)
            }))
            .await;

        let mut merged_names = vec![false; self.chunk.slot_names.len()];
        for result in results {
            self.merge_branch_result(result?, &mut merged_names)?;
        }
        Ok(())
    }

    async fn exec_parallel_named_value(
        &mut self,
        branches_index: usize,
    ) -> Result<Value, RuntimeError> {
        let branches = &self.chunk.named_branch_sets[branches_index];
        let base_slots = self.slots.clone();
        let results = join_all(branches.iter().map(|branch| async {
            let result = Self::run_branch(&branch.chunk, base_slots.clone(), self.host, true).await;
            (branch.name, result)
        }))
        .await;

        let mut record = record_with_capacity(results.len());
        let mut merged_names = vec![false; self.chunk.slot_names.len()];
        for (name, result) in results {
            let result = result?;
            let name_entry = &self.chunk.names[name];
            record.insert_symbolized(
                name_entry.symbol,
                name_entry.text.clone(),
                result.output.clone(),
            );
            self.merge_branch_result(result, &mut merged_names)?;
        }
        Ok(Value::Record(Arc::new(record)))
    }

    fn exec_pure_parallel_value(&self, branches_index: usize) -> Result<Value, RuntimeError> {
        let branches = &self.chunk.pure_parallel_sets[branches_index];
        Ok(Value::List(
            branches
                .iter()
                .map(|expr| {
                    eval_pure_expr(expr, &self.slots, &self.chunk.names, &self.chunk.slot_names)
                })
                .collect::<Result<Vec<_>, _>>()?
                .into(),
        ))
    }

    fn exec_pure_parallel_named_value(&self, branches_index: usize) -> Result<Value, RuntimeError> {
        let branches = &self.chunk.pure_named_parallel_sets[branches_index];
        let mut record = record_with_capacity(branches.len());
        for (name, expr) in branches.iter() {
            let name_entry = &self.chunk.names[*name];
            let value =
                eval_pure_expr(expr, &self.slots, &self.chunk.names, &self.chunk.slot_names)?;
            record.insert_symbolized(name_entry.symbol, name_entry.text.clone(), value);
        }
        Ok(Value::Record(Arc::new(record)))
    }

    async fn exec_parallel_named_calls_value(
        &self,
        branches_index: usize,
    ) -> Result<Value, RuntimeError> {
        let branches = &self.chunk.named_parallel_call_sets[branches_index];
        match branches.len() {
            0 => return Ok(Value::Record(Arc::new(Record::default()))),
            1 => {
                let call = self.prepare_named_parallel_call(&branches[0])?;
                let result = Self::run_prepared_named_call(self.chunk, call, self.host).await?;
                let mut record = record_with_capacity(1);
                self.insert_named_parallel_call_result(&mut record, result);
                return Ok(Value::Record(Arc::new(record)));
            }
            2 => {
                let left_call = self.prepare_named_parallel_call(&branches[0])?;
                let right_call = self.prepare_named_parallel_call(&branches[1])?;
                let calls = [left_call, right_call];
                let results =
                    Self::run_prepared_named_calls_batch_2(self.chunk, &calls, self.host).await?;
                let mut record = record_with_capacity(2);
                for result in results {
                    self.insert_named_parallel_call_result(&mut record, result);
                }
                return Ok(Value::Record(Arc::new(record)));
            }
            _ => {}
        }

        let mut calls = Vec::with_capacity(branches.len());
        for branch in branches {
            calls.push(self.prepare_named_parallel_call(branch)?);
        }

        let results = Self::run_prepared_named_calls_batch(self.chunk, &calls, self.host).await?;
        let mut record = record_with_capacity(results.len());
        for result in results {
            self.insert_named_parallel_call_result(&mut record, result);
        }
        Ok(Value::Record(Arc::new(record)))
    }

    async fn exec_parallel_calls(&mut self, branches_index: usize) -> Result<(), RuntimeError> {
        let branches = &self.chunk.parallel_call_sets[branches_index];
        match branches.len() {
            0 => Ok(()),
            1 => {
                let call = self.prepare_parallel_call(&branches[0])?;
                let result = Self::run_prepared_call(self.chunk, call, self.host).await?;
                self.slots
                    .assign(result.slot, result.output, &self.chunk.slot_names)?;
                self.record_assignment(result.slot);
                Ok(())
            }
            2 => {
                let left_call = self.prepare_parallel_call(&branches[0])?;
                let right_call = self.prepare_parallel_call(&branches[1])?;
                let calls = [left_call, right_call];
                if calls[0].slot == calls[1].slot {
                    return Err(RuntimeError::ParallelConflict {
                        name: self.chunk.slot_names[calls[0].slot].text.to_string(),
                    });
                }
                let results =
                    Self::run_prepared_calls_batch_2(self.chunk, &calls, self.host).await?;
                for result in results {
                    self.slots
                        .assign(result.slot, result.output, &self.chunk.slot_names)?;
                    self.record_assignment(result.slot);
                }
                Ok(())
            }
            _ => {
                let mut calls = Vec::with_capacity(branches.len());
                for branch in branches {
                    calls.push(self.prepare_parallel_call(branch)?);
                }
                self.ensure_distinct_parallel_call_slots(&calls)?;
                let results = Self::run_prepared_calls_batch(self.chunk, &calls, self.host).await?;
                for result in results {
                    self.slots
                        .assign(result.slot, result.output, &self.chunk.slot_names)?;
                    self.record_assignment(result.slot);
                }
                Ok(())
            }
        }
    }

    async fn exec_parallel_calls_value(
        &mut self,
        branches_index: usize,
    ) -> Result<Value, RuntimeError> {
        let branches = &self.chunk.parallel_call_sets[branches_index];
        match branches.len() {
            0 => Ok(Value::List(Vec::<Value>::new().into())),
            1 => {
                let call = self.prepare_parallel_call(&branches[0])?;
                let result = Self::run_prepared_call(self.chunk, call, self.host).await?;
                let output = result.output.clone();
                self.slots
                    .assign(result.slot, result.output, &self.chunk.slot_names)?;
                self.record_assignment(result.slot);
                Ok(Value::List(vec![output].into()))
            }
            2 => {
                let left_call = self.prepare_parallel_call(&branches[0])?;
                let right_call = self.prepare_parallel_call(&branches[1])?;
                let calls = [left_call, right_call];
                if calls[0].slot == calls[1].slot {
                    return Err(RuntimeError::ParallelConflict {
                        name: self.chunk.slot_names[calls[0].slot].text.to_string(),
                    });
                }
                let results =
                    Self::run_prepared_calls_batch_2(self.chunk, &calls, self.host).await?;
                let mut outputs = Vec::with_capacity(2);
                for result in results {
                    outputs.push(result.output.clone());
                    self.slots
                        .assign(result.slot, result.output, &self.chunk.slot_names)?;
                    self.record_assignment(result.slot);
                }
                Ok(Value::List(outputs.into()))
            }
            _ => {
                let mut calls = Vec::with_capacity(branches.len());
                for branch in branches {
                    calls.push(self.prepare_parallel_call(branch)?);
                }
                self.ensure_distinct_parallel_call_slots(&calls)?;
                let results = Self::run_prepared_calls_batch(self.chunk, &calls, self.host).await?;
                let mut outputs = Vec::with_capacity(results.len());
                for result in results {
                    outputs.push(result.output.clone());
                    self.slots
                        .assign(result.slot, result.output, &self.chunk.slot_names)?;
                    self.record_assignment(result.slot);
                }
                Ok(Value::List(outputs.into()))
            }
        }
    }

    fn prepare_parallel_call(
        &self,
        branch: &ParallelCallBranch,
    ) -> Result<PreparedParallelCall, RuntimeError> {
        let value = eval_pure_expr(
            &branch.args,
            &self.slots,
            &self.chunk.names,
            &self.chunk.slot_names,
        )?;
        let Value::Record(args) = value else {
            return Err(RuntimeError::TypeError {
                message: "parallel call args must compile to a record".to_string(),
            });
        };
        Ok(PreparedParallelCall {
            slot: branch.slot,
            name: branch.name,
            args: Arc::unwrap_or_clone(args),
        })
    }

    fn ensure_distinct_parallel_call_slots(
        &self,
        calls: &[PreparedParallelCall],
    ) -> Result<(), RuntimeError> {
        for (index, call) in calls.iter().enumerate() {
            if calls[..index]
                .iter()
                .any(|previous| previous.slot == call.slot)
            {
                return Err(RuntimeError::ParallelConflict {
                    name: self.chunk.slot_names[call.slot].text.to_string(),
                });
            }
        }
        Ok(())
    }

    fn prepare_named_parallel_call(
        &self,
        branch: &NamedParallelCallBranch,
    ) -> Result<PreparedNamedParallelCall, RuntimeError> {
        let value = eval_pure_expr(
            &branch.args,
            &self.slots,
            &self.chunk.names,
            &self.chunk.slot_names,
        )?;
        let Value::Record(args) = value else {
            return Err(RuntimeError::TypeError {
                message: "parallel call args must compile to a record".to_string(),
            });
        };
        Ok(PreparedNamedParallelCall {
            output_name: branch.output_name,
            name: branch.name,
            args: Arc::unwrap_or_clone(args),
        })
    }

    fn insert_named_parallel_call_result(
        &self,
        record: &mut Record,
        result: NamedParallelCallResult,
    ) {
        let name_entry = &self.chunk.names[result.output_name];
        record.insert_symbolized(name_entry.symbol, name_entry.text.clone(), result.output);
    }

    async fn run_host_batch<const N: usize>(
        host_calls: Vec<ToolHostCall>,
        host: &'a H,
    ) -> Result<HostBatchResults<N>, RuntimeError>
    where
        [HostBatchItemResult; N]: smallvec::Array<Item = HostBatchItemResult>,
    {
        let run = std::panic::AssertUnwindSafe(host.call_batch(host_calls))
            .catch_unwind()
            .await;
        match run {
            Ok(results) => Ok(results.into_iter().collect()),
            Err(_) => Err(RuntimeError::ValueError {
                message: "parallel branch panicked".to_string(),
            }),
        }
    }

    async fn run_prepared_calls_batch(
        chunk: &'a Chunk,
        calls: &[PreparedParallelCall],
        host: &'a H,
    ) -> Result<SmallVec<[ParallelCallResult; 4]>, RuntimeError> {
        let host_calls = calls
            .iter()
            .map(|call| ToolHostCall {
                name: chunk.names[call.name].text.to_string(),
                args: call.args.clone(),
            })
            .collect::<Vec<_>>();
        let results = Self::run_host_batch::<4>(host_calls, host).await?;
        Self::finish_prepared_calls_batch(calls, results)
    }

    async fn run_prepared_calls_batch_2(
        chunk: &'a Chunk,
        calls: &[PreparedParallelCall; 2],
        host: &'a H,
    ) -> Result<[ParallelCallResult; 2], RuntimeError> {
        let host_calls = vec![
            ToolHostCall {
                name: chunk.names[calls[0].name].text.to_string(),
                args: calls[0].args.clone(),
            },
            ToolHostCall {
                name: chunk.names[calls[1].name].text.to_string(),
                args: calls[1].args.clone(),
            },
        ];
        let results = Self::run_host_batch::<2>(host_calls, host).await?;
        Self::finish_prepared_calls_batch_2(calls, results)
    }

    fn finish_prepared_call_result(
        call: &PreparedParallelCall,
        result: Result<Value, ToolHostError>,
    ) -> ParallelCallResult {
        ParallelCallResult {
            slot: call.slot,
            output: match result {
                Ok(value) => success(value),
                Err(error) => error_value(error.to_string()),
            },
        }
    }

    fn finish_host_batch_pair(
        results: HostBatchResults<2>,
    ) -> Result<[HostBatchItemResult; 2], RuntimeError> {
        if results.len() != 2 {
            return Err(RuntimeError::ValueError {
                message: "parallel call batch returned the wrong number of results".to_string(),
            });
        }
        let mut results = results.into_iter();
        Ok([
            results.next().expect("length checked"),
            results.next().expect("length checked"),
        ])
    }

    fn finish_prepared_calls_batch_2(
        calls: &[PreparedParallelCall; 2],
        results: HostBatchResults<2>,
    ) -> Result<[ParallelCallResult; 2], RuntimeError> {
        let [left, right] = Self::finish_host_batch_pair(results)?;
        Ok([
            Self::finish_prepared_call_result(&calls[0], left),
            Self::finish_prepared_call_result(&calls[1], right),
        ])
    }

    fn finish_prepared_calls_batch(
        calls: &[PreparedParallelCall],
        results: HostBatchResults<4>,
    ) -> Result<SmallVec<[ParallelCallResult; 4]>, RuntimeError> {
        if results.len() != calls.len() {
            return Err(RuntimeError::ValueError {
                message: "parallel call batch returned the wrong number of results".to_string(),
            });
        }
        Ok(calls
            .iter()
            .zip(results)
            .map(|(call, result)| Self::finish_prepared_call_result(call, result))
            .collect())
    }

    async fn run_prepared_named_calls_batch(
        chunk: &'a Chunk,
        calls: &[PreparedNamedParallelCall],
        host: &'a H,
    ) -> Result<SmallVec<[NamedParallelCallResult; 4]>, RuntimeError> {
        let host_calls = calls
            .iter()
            .map(|call| ToolHostCall {
                name: chunk.names[call.name].text.to_string(),
                args: call.args.clone(),
            })
            .collect::<Vec<_>>();
        let results = Self::run_host_batch::<4>(host_calls, host).await?;
        Self::finish_prepared_named_calls_batch(calls, results)
    }

    async fn run_prepared_named_calls_batch_2(
        chunk: &'a Chunk,
        calls: &[PreparedNamedParallelCall; 2],
        host: &'a H,
    ) -> Result<[NamedParallelCallResult; 2], RuntimeError> {
        let host_calls = vec![
            ToolHostCall {
                name: chunk.names[calls[0].name].text.to_string(),
                args: calls[0].args.clone(),
            },
            ToolHostCall {
                name: chunk.names[calls[1].name].text.to_string(),
                args: calls[1].args.clone(),
            },
        ];
        let results = Self::run_host_batch::<2>(host_calls, host).await?;
        Self::finish_prepared_named_calls_batch_2(calls, results)
    }

    fn finish_prepared_named_call_result(
        call: &PreparedNamedParallelCall,
        result: Result<Value, ToolHostError>,
    ) -> NamedParallelCallResult {
        NamedParallelCallResult {
            output_name: call.output_name,
            output: match result {
                Ok(value) => success(value),
                Err(error) => error_value(error.to_string()),
            },
        }
    }

    fn finish_prepared_named_calls_batch_2(
        calls: &[PreparedNamedParallelCall; 2],
        results: HostBatchResults<2>,
    ) -> Result<[NamedParallelCallResult; 2], RuntimeError> {
        let [left, right] = Self::finish_host_batch_pair(results)?;
        Ok([
            Self::finish_prepared_named_call_result(&calls[0], left),
            Self::finish_prepared_named_call_result(&calls[1], right),
        ])
    }

    fn finish_prepared_named_calls_batch(
        calls: &[PreparedNamedParallelCall],
        results: HostBatchResults<4>,
    ) -> Result<SmallVec<[NamedParallelCallResult; 4]>, RuntimeError> {
        if results.len() != calls.len() {
            return Err(RuntimeError::ValueError {
                message: "parallel call batch returned the wrong number of results".to_string(),
            });
        }
        Ok(calls
            .iter()
            .zip(results)
            .map(|(call, result)| Self::finish_prepared_named_call_result(call, result))
            .collect())
    }

    async fn run_branch(
        chunk: &'a Chunk,
        slots: SlotState,
        host: &'a H,
        in_parallel_branch: bool,
    ) -> Result<BranchResult, RuntimeError> {
        let mut vm = Self::new(chunk, slots, host, in_parallel_branch);
        let run = std::panic::AssertUnwindSafe(vm.run()).catch_unwind().await;
        match run {
            Ok(Ok(ExecutionOutcome::Continued)) => Ok(vm.into_branch_result()),
            Ok(Ok(ExecutionOutcome::Finished(_))) => Err(RuntimeError::FinishInsideParallel),
            Ok(Err(error)) => Err(error),
            Err(_) => Err(RuntimeError::ValueError {
                message: "parallel branch panicked".to_string(),
            }),
        }
    }

    async fn run_prepared_call(
        chunk: &'a Chunk,
        call: PreparedParallelCall,
        host: &'a H,
    ) -> Result<ParallelCallResult, RuntimeError> {
        let slot = call.slot;
        let name = chunk.names[call.name].text.to_string();
        let run = std::panic::AssertUnwindSafe(host.call(name, call.args))
            .catch_unwind()
            .await;
        match run {
            Ok(result) => match result {
                Ok(value) => Ok(success(value)),
                Err(error) => Ok(error_value(error.to_string())),
            },
            Err(_) => Err(RuntimeError::ValueError {
                message: "parallel branch panicked".to_string(),
            }),
        }
        .map(|value| ParallelCallResult {
            slot,
            output: value,
        })
    }

    async fn run_prepared_named_call(
        chunk: &'a Chunk,
        call: PreparedNamedParallelCall,
        host: &'a H,
    ) -> Result<NamedParallelCallResult, RuntimeError> {
        let output_name = call.output_name;
        let name = chunk.names[call.name].text.to_string();
        let run = std::panic::AssertUnwindSafe(host.call(name, call.args))
            .catch_unwind()
            .await;
        match run {
            Ok(result) => match result {
                Ok(value) => Ok(success(value)),
                Err(error) => Ok(error_value(error.to_string())),
            },
            Err(_) => Err(RuntimeError::ValueError {
                message: "parallel branch panicked".to_string(),
            }),
        }
        .map(|output| NamedParallelCallResult {
            output_name,
            output,
        })
    }

    fn merge_branch_result(
        &mut self,
        result: BranchResult,
        merged_names: &mut [bool],
    ) -> Result<(), RuntimeError> {
        for (slot, value) in result.values {
            if std::mem::replace(&mut merged_names[slot], true) {
                return Err(RuntimeError::ParallelConflict {
                    name: self.chunk.slot_names[slot].text.to_string(),
                });
            }
            self.slots.assign(slot, value, &self.chunk.slot_names)?;
            self.record_assignment(slot);
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

    fn await_value(
        &self,
        handle: Value,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Value> + Send + '_>> {
        Box::pin(async move {
            match handle {
                Value::List(handles) => {
                    let mut values = Vec::with_capacity(handles.len());
                    for handle in handles.iter().cloned() {
                        values.push(self.await_value(handle).await);
                    }
                    Value::List(values.into())
                }
                Value::Record(handles) if is_async_handle_record(&handles) => {
                    match self.host.await_handle(Value::Record(handles)).await {
                        Ok(value) => success(value),
                        Err(error) => error_value(error.to_string()),
                    }
                }
                Value::Record(handles) => {
                    let mut record = record_with_capacity(handles.len());
                    for entry in handles.entries.iter() {
                        record.insert_symbolized(
                            entry.symbol,
                            entry.name.clone(),
                            self.await_value(entry.value.clone()).await,
                        );
                    }
                    Value::Record(Arc::new(record))
                }
                handle => match self.host.await_handle(handle).await {
                    Ok(value) => success(value),
                    Err(error) => error_value(error.to_string()),
                },
            }
        })
    }

    async fn await_value_unwrap(&self, handle: Value) -> Result<Value, RuntimeError> {
        match handle {
            Value::Record(handles) if is_async_handle_record(&handles) => self
                .host
                .await_handle(Value::Record(handles))
                .await
                .map_err(|error| RuntimeError::ValueError {
                    message: format!("`?` unwrapped failed tool result: {error}"),
                }),
            Value::List(_) | Value::Record(_) => unwrap_tool_result(self.await_value(handle).await),
            handle => {
                self.host
                    .await_handle(handle)
                    .await
                    .map_err(|error| RuntimeError::ValueError {
                        message: format!("`?` unwrapped failed tool result: {error}"),
                    })
            }
        }
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

    fn record_assignment(&mut self, slot: usize) {
        if let Some(assigned) = &mut self.assigned {
            assigned[slot] = true;
        }
    }

    fn into_branch_result(self) -> BranchResult {
        let assigned = self.assigned.unwrap_or_default();
        let mut values = Vec::with_capacity(assigned.iter().filter(|assigned| **assigned).count());
        for (slot, assigned) in assigned.into_iter().enumerate() {
            if assigned && let Some(value) = self.slots.get(slot).cloned() {
                values.push((slot, value));
            }
        }
        BranchResult {
            values,
            output: self.last_value.unwrap_or(Value::Null),
        }
    }

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

    fn record_builtin_profile(&mut self, builtin: Builtin, elapsed_ns: u128) {
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

struct BranchResult {
    values: Vec<(usize, Value)>,
    output: Value,
}

struct ParallelCallResult {
    slot: usize,
    output: Value,
}

struct NamedParallelCallResult {
    output_name: usize,
    output: Value,
}

struct PreparedParallelCall {
    slot: usize,
    name: usize,
    args: Record,
}

struct PreparedNamedParallelCall {
    output_name: usize,
    name: usize,
    args: Record,
}

type HostBatchItemResult = Result<Value, ToolHostError>;
type HostBatchResults<const N: usize> = SmallVec<[HostBatchItemResult; N]>;

pub(crate) struct IterState {
    cursor: IterCursor,
    binding: usize,
    restore: LoopRestore,
}

enum IterCursor {
    List { values: Arc<[Value]>, index: usize },
    Range { next: i64, end: i64 },
}

impl IterCursor {
    fn next_value(&mut self) -> Option<Value> {
        match self {
            Self::List { values, index } => {
                let value = values.get(*index)?.clone();
                *index += 1;
                Some(value)
            }
            Self::Range { next, end } => {
                if *next >= *end {
                    return None;
                }
                let value = *next;
                *next += 1;
                Some(Value::Number(value as f64))
            }
        }
    }
}

struct LoopRestore {
    previous: Option<Value>,
}
