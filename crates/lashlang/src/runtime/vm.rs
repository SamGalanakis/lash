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

use super::host::{
    AbilityOp, AbilityResult, ExecutionMode, ProcessBlockEvent, ProcessBlockEventKind,
    ProcessBlockStart,
};
use super::record::{Record, record_with_capacity};
use super::schema::{
    ValidationPlan, compile_schema_value, execute_validate_builtin, execute_validation_plan,
};
use super::value::ProjectedValue;
use super::{
    COOPERATIVE_YIELD_INSTRUCTION_BUDGET, Chunk, ExecutionHost, ExecutionOutcome, ExecutionScratch,
    Instruction, InstructionProfileTag, IntrinsicOp, LASH_TYPE_KEY, ListValue, LoopExpr,
    LoopIterable, LoopOp, LoweredLoop, Name, ProfileAccumulator, ProfileReport, ProjectedBindings,
    RuntimeError, RuntimeFailure, Value, add_assign_index_number, add_values, as_number,
    assign_path, error_value, eval_binary_values, eval_binary_values_async, eval_compare_values,
    eval_compare_values_async, eval_number_binary_values, eval_number_compare_values,
    eval_number_numeric_binary_value, execute_compiled_format, execute_compiled_format_direct,
    execute_compiled_format_one_number_compact_direct, execute_integer_div_builtin,
    execute_intrinsic, execute_len_direct, execute_push_builtin_async, execute_range_builtin,
    is_process_handle_record, is_truthy, is_truthy_async, iterable_values,
    materialize_projected_async, materialize_value, range_bounds, range_bounds_async, read_field,
    read_field_direct, read_field_ref_direct, read_index, read_index_direct, success,
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

enum VmStep {
    Continue,
    Finish(Value),
    FinishProcess(Value),
    FailProcess(Value),
    Effect(VmEffect),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VmMode {
    Foreground,
    Process,
}

impl From<ExecutionMode> for VmMode {
    fn from(mode: ExecutionMode) -> Self {
        match mode {
            ExecutionMode::Foreground => Self::Foreground,
            ExecutionMode::Process => Self::Process,
        }
    }
}

enum VmOutcome {
    Continued,
    Finished(Value),
    ProcessFinished(Value),
    ProcessFailed(Value),
}

#[derive(Clone, Copy)]
enum VmEffect {
    CallTool {
        name: usize,
        keys: usize,
    },
    CallToolUnwrap {
        name: usize,
        keys: usize,
    },
    StartCallTool {
        name: usize,
        keys: usize,
    },
    StartProcess {
        block: usize,
        has_name: bool,
        has_timeout_ms: bool,
        has_input: bool,
    },
    AwaitHandle,
    AwaitHandleUnwrap,
    CancelHandle,
    Print,
    ProcessEvent(ProcessBlockEventKind),
}

struct VmTrap {
    error: RuntimeError,
    instruction_ip: usize,
}

enum AddAssignRight {
    Number(f64),
    Value(Value),
}

enum LoopFlow {
    Continue,
    BreakLoop,
    ContinueLoop,
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

    pub(crate) async fn run(&mut self) -> Result<ExecutionOutcome, RuntimeError> {
        match self.run_raw().await? {
            VmOutcome::Continued => Ok(ExecutionOutcome::Continued),
            VmOutcome::Finished(value) => Ok(ExecutionOutcome::Finished(value)),
            VmOutcome::ProcessFinished(_) => {
                Err(RuntimeError::ProcessControlOutsideProcess { keyword: "finish" })
            }
            VmOutcome::ProcessFailed(_) => {
                Err(RuntimeError::ProcessControlOutsideProcess { keyword: "fail" })
            }
        }
    }

    pub(crate) async fn run_process(&mut self) -> Result<ExecutionOutcome, RuntimeError> {
        match self.run_raw().await? {
            VmOutcome::Continued => Ok(ExecutionOutcome::Finished(Value::Null)),
            VmOutcome::ProcessFinished(value) => Ok(ExecutionOutcome::Finished(value)),
            VmOutcome::ProcessFailed(value) => Ok(ExecutionOutcome::Failed(value)),
            VmOutcome::Finished(_) => {
                Err(RuntimeError::ForegroundControlInsideProcess { keyword: "submit" })
            }
        }
    }

    pub(crate) async fn run_for_mode(&mut self) -> Result<ExecutionOutcome, RuntimeError> {
        match self.mode {
            VmMode::Foreground => self.run().await,
            VmMode::Process => self.run_process().await,
        }
    }

    pub(crate) async fn run_traced(&mut self) -> Result<ExecutionOutcome, RuntimeFailure> {
        let result = self.run_raw_traced().await?;
        match result {
            VmOutcome::Continued => Ok(ExecutionOutcome::Continued),
            VmOutcome::Finished(value) => Ok(ExecutionOutcome::Finished(value)),
            VmOutcome::ProcessFinished(_) => Err(RuntimeFailure {
                error: RuntimeError::ProcessControlOutsideProcess { keyword: "finish" },
                span: None,
            }),
            VmOutcome::ProcessFailed(_) => Err(RuntimeFailure {
                error: RuntimeError::ProcessControlOutsideProcess { keyword: "fail" },
                span: None,
            }),
        }
    }

    pub(crate) async fn run_process_traced(&mut self) -> Result<ExecutionOutcome, RuntimeFailure> {
        let result = self.run_raw_traced().await?;
        match result {
            VmOutcome::Continued => Ok(ExecutionOutcome::Finished(Value::Null)),
            VmOutcome::ProcessFinished(value) => Ok(ExecutionOutcome::Finished(value)),
            VmOutcome::ProcessFailed(value) => Ok(ExecutionOutcome::Failed(value)),
            VmOutcome::Finished(_) => Err(RuntimeFailure {
                error: RuntimeError::ForegroundControlInsideProcess { keyword: "submit" },
                span: None,
            }),
        }
    }

    pub(crate) async fn run_traced_for_mode(&mut self) -> Result<ExecutionOutcome, RuntimeFailure> {
        match self.mode {
            VmMode::Foreground => self.run_traced().await,
            VmMode::Process => self.run_process_traced().await,
        }
    }

    async fn run_raw(&mut self) -> Result<VmOutcome, RuntimeError> {
        let result = self.run_loop().await.map_err(|trap| trap.error);
        self.unwind_iterators();
        result
    }

    async fn run_raw_traced(&mut self) -> Result<VmOutcome, RuntimeFailure> {
        let result = self.run_loop().await.map_err(|trap| RuntimeFailure {
            error: trap.error,
            span: self.chunk.spans.get(trap.instruction_ip).copied().flatten(),
        });
        self.unwind_iterators();
        result
    }

    async fn run_loop(&mut self) -> Result<VmOutcome, VmTrap> {
        let mut budget = COOPERATIVE_YIELD_INSTRUCTION_BUDGET;
        while let Some(instruction) = self.chunk.code.get(self.ip).copied() {
            let instruction_ip = self.ip;
            self.ip += 1;
            let profile = self
                .profile
                .as_ref()
                .map(|_| (instruction.profile_tag(), Instant::now()));
            let step = match self.step_instruction_fast(instruction) {
                Ok(Some(step)) => Ok(step),
                Ok(None) => self.step_instruction(instruction).await,
                Err(error) => Err(error),
            };
            let result = match step {
                Ok(VmStep::Continue) => Ok(None),
                Ok(VmStep::Finish(value)) => Ok(Some(VmOutcome::Finished(value))),
                Ok(VmStep::FinishProcess(value)) => Ok(Some(VmOutcome::ProcessFinished(value))),
                Ok(VmStep::FailProcess(value)) => Ok(Some(VmOutcome::ProcessFailed(value))),
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
        Ok(VmOutcome::Continued)
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
                return Ok(Some(VmStep::Finish(self.pop_stack()?)));
            }
            Instruction::ProcessYield => {
                if self.mode != VmMode::Process {
                    return Err(RuntimeError::ProcessControlOutsideProcess { keyword: "yield" });
                }
                return Ok(Some(VmStep::Effect(VmEffect::ProcessEvent(
                    ProcessBlockEventKind::Yield,
                ))));
            }
            Instruction::ProcessWake => {
                if self.mode != VmMode::Process {
                    return Err(RuntimeError::ProcessControlOutsideProcess { keyword: "wake" });
                }
                return Ok(Some(VmStep::Effect(VmEffect::ProcessEvent(
                    ProcessBlockEventKind::Wake,
                ))));
            }
            Instruction::ProcessFinish => {
                if self.mode != VmMode::Process {
                    return Err(RuntimeError::ProcessControlOutsideProcess { keyword: "finish" });
                }
                return Ok(Some(VmStep::FinishProcess(self.pop_stack()?)));
            }
            Instruction::ProcessFail => {
                if self.mode != VmMode::Process {
                    return Err(RuntimeError::ProcessControlOutsideProcess { keyword: "fail" });
                }
                return Ok(Some(VmStep::FailProcess(self.pop_stack()?)));
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
            Instruction::CallTool { name, keys } => {
                return Ok(VmStep::Effect(VmEffect::CallTool { name, keys }));
            }
            Instruction::CallToolUnwrap { name, keys } => {
                return Ok(VmStep::Effect(VmEffect::CallToolUnwrap { name, keys }));
            }
            Instruction::StartCallTool { name, keys } => {
                return Ok(VmStep::Effect(VmEffect::StartCallTool { name, keys }));
            }
            Instruction::StartProcess {
                block,
                has_name,
                has_timeout_ms,
                has_input,
            } => {
                return Ok(VmStep::Effect(VmEffect::StartProcess {
                    block,
                    has_name,
                    has_timeout_ms,
                    has_input,
                }));
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
                return Ok(VmStep::Finish(self.pop_stack()?));
            }
            Instruction::ProcessYield => {
                if self.mode != VmMode::Process {
                    return Err(RuntimeError::ProcessControlOutsideProcess { keyword: "yield" });
                }
                return Ok(VmStep::Effect(VmEffect::ProcessEvent(
                    ProcessBlockEventKind::Yield,
                )));
            }
            Instruction::ProcessWake => {
                if self.mode != VmMode::Process {
                    return Err(RuntimeError::ProcessControlOutsideProcess { keyword: "wake" });
                }
                return Ok(VmStep::Effect(VmEffect::ProcessEvent(
                    ProcessBlockEventKind::Wake,
                )));
            }
            Instruction::ProcessFinish => {
                if self.mode != VmMode::Process {
                    return Err(RuntimeError::ProcessControlOutsideProcess { keyword: "finish" });
                }
                return Ok(VmStep::FinishProcess(self.pop_stack()?));
            }
            Instruction::ProcessFail => {
                if self.mode != VmMode::Process {
                    return Err(RuntimeError::ProcessControlOutsideProcess { keyword: "fail" });
                }
                return Ok(VmStep::FailProcess(self.pop_stack()?));
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

    async fn execute_lowered_loop(
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

    fn execute_lowered_loop_direct(
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
                        let value = materialize_projected_async(value).await;
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
                crate::ast::BinaryOp::And => {
                    let left = Box::pin(self.eval_loop_expr(left)).await?;
                    if !is_truthy_async(&left).await {
                        Ok(Value::Bool(false))
                    } else {
                        let right = Box::pin(self.eval_loop_expr(right)).await?;
                        Ok(Value::Bool(is_truthy_async(&right).await))
                    }
                }
                crate::ast::BinaryOp::Or => {
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
                crate::ast::BinaryOp::And => {
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
                crate::ast::BinaryOp::Or => {
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

    async fn resolve_effect(&mut self, effect: VmEffect) -> Result<(), RuntimeError> {
        match effect {
            VmEffect::CallTool { name, keys } => {
                let args = self.drain_record_from_stack(keys)?;
                let result = match self
                    .host
                    .perform(AbilityOp::CallTool {
                        name: self.chunk.names[name].text.to_string(),
                        args,
                    })
                    .await
                {
                    Ok(AbilityResult::Value(value)) => success(value),
                    Ok(AbilityResult::Unit) => {
                        error_value("tool call returned no value".to_string())
                    }
                    Err(error) => error_value(error.to_string()),
                };
                self.stack.push(result);
            }
            VmEffect::CallToolUnwrap { name, keys } => {
                let args = self.drain_record_from_stack(keys)?;
                let value = self
                    .host
                    .perform(AbilityOp::CallTool {
                        name: self.chunk.names[name].text.to_string(),
                        args,
                    })
                    .await
                    .and_then(|result| result.into_value("tool call"))
                    .map_err(|error| RuntimeError::ValueError {
                        message: format!("`?` unwrapped failed tool result: {error}"),
                    })?;
                self.stack.push(value);
            }
            VmEffect::StartCallTool { name, keys } => {
                let args = self.drain_record_from_stack(keys)?;
                let value = self
                    .host
                    .perform(AbilityOp::StartToolCall {
                        name: self.chunk.names[name].text.to_string(),
                        args,
                    })
                    .await
                    .and_then(|result| result.into_value("async start"))
                    .map_err(|err| RuntimeError::ValueError {
                        message: format!("async start failed: {err}"),
                    })?;
                self.stack.push(value);
            }
            VmEffect::StartProcess {
                block,
                has_name,
                has_timeout_ms,
                has_input,
            } => {
                let input = has_input.then(|| self.pop_stack()).transpose()?;
                let timeout_ms = has_timeout_ms.then(|| self.pop_stack()).transpose()?;
                let name = has_name.then(|| self.pop_stack()).transpose()?;
                let block = &self.chunk.process_blocks[block];
                let value = self
                    .host
                    .perform(AbilityOp::StartProcess(ProcessBlockStart {
                        program: block.program.clone(),
                        tool_names: block
                            .tool_names
                            .iter()
                            .map(|name| name.text.to_string())
                            .collect(),
                        name,
                        timeout_ms,
                        input,
                    }))
                    .await
                    .and_then(|result| result.into_value("process start"))
                    .map_err(|err| RuntimeError::ValueError {
                        message: format!("process start failed: {err}"),
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
                let value = self
                    .host
                    .perform(AbilityOp::Cancel(handle))
                    .await
                    .and_then(|result| result.into_value("cancel"))
                    .map_err(|err| RuntimeError::ValueError {
                        message: format!("cancel failed: {err}"),
                    })?;
                self.last_value = Some(value.clone());
                self.stack.push(value);
            }
            VmEffect::ProcessEvent(kind) => {
                let value = self.pop_stack()?;
                self.host
                    .perform(AbilityOp::ProcessEvent(ProcessBlockEvent {
                        kind,
                        value: value.clone(),
                    }))
                    .await
                    .map_err(|err| RuntimeError::ValueError {
                        message: format!("process event failed: {err}"),
                    })?;
                self.last_value = Some(value.clone());
                self.stack.push(value);
            }
            VmEffect::Print => {
                let value = self.pop_stack()?;
                let host_value = match &value {
                    Value::Projected(projected) => Value::String(projected.render().await.into()),
                    _ => value.clone(),
                };
                self.host
                    .perform(AbilityOp::Print(host_value))
                    .await
                    .map_err(|err| RuntimeError::ValueError {
                        message: format!("print failed: {err}"),
                    })?;
                self.last_value = Some(Value::Null);
                self.stack.push(Value::Null);
            }
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
                Value::Record(handles) if is_process_handle_record(&handles) => {
                    match self
                        .host
                        .perform(AbilityOp::Await(Value::Record(handles)))
                        .await
                    {
                        Ok(AbilityResult::Value(value)) => success(value),
                        Ok(AbilityResult::Unit) => {
                            error_value("await returned no value".to_string())
                        }
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
                handle => match self.host.perform(AbilityOp::Await(handle)).await {
                    Ok(AbilityResult::Value(value)) => success(value),
                    Ok(AbilityResult::Unit) => error_value("await returned no value".to_string()),
                    Err(error) => error_value(error.to_string()),
                },
            }
        })
    }

    async fn await_value_unwrap(&self, handle: Value) -> Result<Value, RuntimeError> {
        match handle {
            Value::Record(handles) if is_process_handle_record(&handles) => self
                .host
                .perform(AbilityOp::Await(Value::Record(handles)))
                .await
                .and_then(|result| result.into_value("await"))
                .map_err(|error| RuntimeError::ValueError {
                    message: format!("`?` unwrapped failed tool result: {error}"),
                }),
            Value::List(_) | Value::Record(_) => unwrap_tool_result(self.await_value(handle).await),
            handle => self
                .host
                .perform(AbilityOp::Await(handle))
                .await
                .and_then(|result| result.into_value("await"))
                .map_err(|error| RuntimeError::ValueError {
                    message: format!("`?` unwrapped failed tool result: {error}"),
                }),
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

fn range_has_next(start: i64, end: i64, step: i64) -> bool {
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
