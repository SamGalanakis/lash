//! Bytecode compiler: lowers `crate::ast::Program` into a `Chunk` of
//! instructions plus the supporting compile-time tables (slot maps, format
//! templates, schema cache).
//!
//! All compile-time-only helpers live here too: `is_pure_expr` /
//! `contains_type_literal` (used to decide whether an expression can be
//! evaluated without entering the VM) and the `fold_type` /
//! `interned_scalar_schema` machinery (used to
//! convert `TypeExpr` AST nodes into JSON-Schema-shaped `Value` literals
//! at compile time).

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc, OnceLock};

use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::artifact::CompiledModuleContext;
use crate::ast::{
    AssignPathStep, AssignTarget, BinaryOp, Expr, ProcessStartExpr, Program, TypeExpr, UnaryOp,
};
use crate::lexer::Span;

use super::record::{Symbol, intern_symbol, lookup_symbol, record_with_capacity, symbol_name};
use super::schema::{ValidationPlan, compile_schema_value};
use super::{
    Chunk, CompileStats, CompiledAssignPath, CompiledAssignPathStep, CompiledFormatTemplate,
    Instruction, IntrinsicOp, LASH_TYPE_KEY, LoopExpr, LoopIterable, LoopOp, LoweredLoop, Name,
    Value, as_number, compile_format_template, eval_binary_values, execute_integer_div_builtin,
    execute_len_direct, execute_range_builtin, is_comparison_binary_op, is_numeric_binary_op,
    is_truthy, read_field_direct, read_index_direct, transient_name, unwrap_type_value,
};

pub(crate) struct Compiler {
    module_context: Option<CompiledModuleContext>,
    code: Vec<Instruction>,
    spans: Vec<Option<Span>>,
    constants: Vec<Value>,
    names: Vec<Name>,
    name_lookup: FxHashMap<Symbol, usize>,
    slots: Rc<RefCell<SlotTable>>,
    key_lists: Vec<Box<[usize]>>,
    format_templates: Vec<CompiledFormatTemplate>,
    compiled_schemas: Vec<ValidationPlan>,
    assign_paths: Vec<CompiledAssignPath>,
    lowered_loops: Vec<LoweredLoop>,
    compile_stats: Rc<RefCell<CompileStats>>,
    const_slots: Vec<Option<Value>>,
    loop_contexts: Vec<LoopContext>,
}

struct LoopContext {
    continue_target: usize,
    break_jumps: SmallVec<[usize; 4]>,
}

struct LoweredLoopOptimizer<'a, 'b> {
    compiler: &'a mut Compiler,
    binding_name: &'b str,
    assigned_slots: SmallVec<[usize; 8]>,
}

impl LoweredLoopOptimizer<'_, '_> {
    fn optimize_block(&mut self, block: &Expr) -> Option<Vec<LoopOp>> {
        match block {
            Expr::Block(expressions) => expressions
                .iter()
                .map(|expression| self.optimize_expr(expression))
                .collect(),
            expression => Some(vec![self.optimize_expr(expression)?]),
        }
    }

    fn optimize_expr(&mut self, expression: &Expr) -> Option<LoopOp> {
        match expression {
            Expr::Assign { target, expr } if target.root.as_str() == self.binding_name => None,
            Expr::Assign { target, expr } if target.is_simple() => {
                let slot = self.compiler.push_slot(&target.root);
                self.assigned_slots.push(slot);
                if let Expr::Binary {
                    left,
                    op: BinaryOp::Add,
                    right,
                } = expr.as_ref()
                    && matches!(left.as_ref(), Expr::Variable(var) if var.as_str() == target.root.as_str())
                {
                    if let Expr::List(items) = right.as_ref()
                        && items.len() == 1
                    {
                        return Some(LoopOp::AppendAssign {
                            slot,
                            expr: self.compiler.optimize_loop_expr(&items[0])?,
                        });
                    }
                    if let Some(Value::Number(right)) = self.compiler.fold_compile_time_expr(right)
                    {
                        return Some(LoopOp::AddAssignNumber { slot, right });
                    }
                    if let Expr::Variable(right_name) = right.as_ref() {
                        let right = self.compiler.push_slot(right_name);
                        return Some(LoopOp::AddAssignSlot { slot, right });
                    }
                    return Some(LoopOp::AddAssign {
                        slot,
                        expr: self.compiler.optimize_loop_expr(right)?,
                    });
                }
                if let Expr::BuiltinCall {
                    name: builtin_name,
                    args,
                } = expr.as_ref()
                    && builtin_name == "push"
                    && let [Expr::Variable(first_arg), item] = args.as_slice()
                    && first_arg.as_str() == target.root.as_str()
                {
                    return Some(LoopOp::AppendAssign {
                        slot,
                        expr: self.compiler.optimize_loop_expr(item)?,
                    });
                }
                Some(LoopOp::Assign {
                    slot,
                    expr: self.compiler.optimize_loop_expr(expr)?,
                })
            }
            Expr::Assign { target, expr } => {
                let slot = self.compiler.push_slot(&target.root);
                self.assigned_slots.push(slot);
                if let [AssignPathStep::Index(index)] = target.steps.as_slice()
                    && let Expr::Binary {
                        left,
                        op: BinaryOp::Add,
                        right,
                    } = expr.as_ref()
                    && let Expr::Index {
                        target: left_target,
                        index: left_index,
                    } = left.as_ref()
                    && matches!(left_target.as_ref(), Expr::Variable(name) if name.as_str() == target.root.as_str())
                    && left_index.as_ref() == index
                    && let Some(Value::Number(right)) = self.compiler.fold_compile_time_expr(right)
                {
                    if let Expr::Variable(index_name) = index {
                        let index = self.compiler.push_slot(index_name);
                        return Some(LoopOp::AddAssignIndexSlotNumber { slot, index, right });
                    }
                    return Some(LoopOp::AddAssignIndexNumber {
                        slot,
                        index: self.compiler.optimize_loop_expr(index)?,
                        right,
                    });
                }

                let indexes = target
                    .steps
                    .iter()
                    .filter_map(|step| match step {
                        AssignPathStep::Field(_) => None,
                        AssignPathStep::Index(index) => {
                            Some(self.compiler.optimize_loop_expr(index))
                        }
                    })
                    .collect::<Option<Vec<_>>>()?
                    .into_boxed_slice();
                let path = self.compiler.push_assign_path(&target.steps);
                Some(LoopOp::PathAssign {
                    slot,
                    path,
                    indexes,
                    expr: self.compiler.optimize_loop_expr(expr)?,
                })
            }
            Expr::If {
                condition,
                then_block,
                else_block,
            } => {
                let condition = self.compiler.optimize_loop_expr(condition)?;
                let then_ops = self.optimize_block(then_block)?.into_boxed_slice();
                let else_ops = self.optimize_block(else_block)?.into_boxed_slice();
                Some(LoopOp::If {
                    condition,
                    then_ops,
                    else_ops,
                })
            }
            Expr::For {
                binding,
                iterable,
                body,
            } => {
                let binding_slot = self.compiler.push_slot(binding);
                let iterable = self.compiler.optimize_loop_iterable(iterable)?;
                let mut nested = LoweredLoopOptimizer {
                    compiler: self.compiler,
                    binding_name: binding,
                    assigned_slots: SmallVec::new(),
                };
                let body = nested.optimize_block(body)?.into_boxed_slice();
                nested.compiler.set_const_slot(binding_slot, None);
                for slot in nested.assigned_slots {
                    nested.compiler.set_const_slot(slot, None);
                    self.assigned_slots.push(slot);
                }
                Some(LoopOp::Loop(Box::new(LoweredLoop {
                    binding: binding_slot,
                    iterable,
                    body,
                })))
            }
            Expr::Break => Some(LoopOp::Break),
            Expr::Continue => Some(LoopOp::Continue),
            Expr::Block(_)
            | Expr::ReceiverCall { .. }
            | Expr::Cancel(_)
            | Expr::Print(_)
            | Expr::Submit(_)
            | Expr::Yield(_)
            | Expr::Wake(_)
            | Expr::Finish(_)
            | Expr::Fail(_)
            | Expr::StartProcess(_)
            | Expr::ResourceRef(_)
            | Expr::SleepFor(_)
            | Expr::SleepUntil(_)
            | Expr::WaitSignal
            | Expr::SignalRun { .. }
            | Expr::Await(_) => None,
            expr => Some(LoopOp::Expr(self.compiler.optimize_loop_expr(expr)?)),
        }
    }
}

#[derive(Default)]
struct SlotTable {
    names: Vec<Name>,
    lookup: FxHashMap<Symbol, usize>,
}

impl Compiler {
    pub(crate) fn compile_program(program: &Program) -> (Chunk, CompileStats) {
        let stats = Rc::new(RefCell::new(CompileStats::default()));
        let mut compiler = Self::with_slots_and_stats(
            None,
            Rc::new(RefCell::new(SlotTable::default())),
            stats.clone(),
        );
        compiler.compile_program_block(program);
        let chunk = compiler.finish();
        let compile_stats = *stats.borrow();
        (chunk, compile_stats)
    }

    pub(crate) fn compile_linked_program(
        program: &Program,
        module_context: CompiledModuleContext,
    ) -> (Chunk, CompileStats) {
        let stats = Rc::new(RefCell::new(CompileStats::default()));
        let mut compiler = Self::with_slots_and_stats(
            Some(module_context),
            Rc::new(RefCell::new(SlotTable::default())),
            stats.clone(),
        );
        compiler.compile_program_block(program);
        let chunk = compiler.finish();
        let compile_stats = *stats.borrow();
        (chunk, compile_stats)
    }

    fn with_slots_and_stats(
        module_context: Option<CompiledModuleContext>,
        slots: Rc<RefCell<SlotTable>>,
        compile_stats: Rc<RefCell<CompileStats>>,
    ) -> Self {
        Self {
            module_context,
            code: Vec::new(),
            spans: Vec::new(),
            constants: Vec::new(),
            names: Vec::new(),
            name_lookup: FxHashMap::default(),
            slots,
            key_lists: Vec::new(),
            format_templates: Vec::new(),
            compiled_schemas: Vec::new(),
            assign_paths: Vec::new(),
            lowered_loops: Vec::new(),
            compile_stats,
            const_slots: Vec::new(),
            loop_contexts: Vec::new(),
        }
    }

    fn finish(self) -> Chunk {
        let slot_names = self.slots.borrow().names.clone();
        let mut spans = self.spans;
        spans.resize(self.code.len(), None);
        Chunk {
            module_context: self.module_context,
            code: self.code,
            spans,
            constants: self.constants,
            names: self.names,
            slot_names,
            key_lists: self.key_lists,
            format_templates: self.format_templates,
            compiled_schemas: self.compiled_schemas,
            assign_paths: self.assign_paths,
            lowered_loops: self.lowered_loops,
        }
    }

    fn push_const(&mut self, value: Value) -> usize {
        let index = self.constants.len();
        self.constants.push(value);
        index
    }

    fn emit_push_value(&mut self, value: Value) {
        match value {
            Value::Null => self.code.push(Instruction::PushNull),
            Value::Bool(value) => self.code.push(Instruction::PushBool(value)),
            Value::Number(value) => self.code.push(Instruction::PushNumber(value)),
            value => {
                let index = self.push_const(value);
                self.code.push(Instruction::PushConst(index));
            }
        }
    }

    fn push_name(&mut self, name: &str) -> usize {
        let symbol = intern_symbol(name);
        if let Some(index) = self.name_lookup.get(&symbol) {
            return *index;
        }

        let index = self.names.len();
        self.names.push(Name {
            symbol,
            text: symbol_name(symbol),
        });
        self.name_lookup.insert(symbol, index);
        index
    }

    fn push_slot(&mut self, name: &str) -> usize {
        let symbol = intern_symbol(name);
        let mut slots = self.slots.borrow_mut();
        if let Some(index) = slots.lookup.get(&symbol) {
            let index = *index;
            drop(slots);
            self.ensure_const_slot(index);
            return index;
        }
        let index = slots.names.len();
        slots.names.push(Name {
            symbol,
            text: symbol_name(symbol),
        });
        slots.lookup.insert(symbol, index);
        drop(slots);
        self.ensure_const_slot(index);
        index
    }

    fn push_key_list<'a>(&mut self, keys: impl Iterator<Item = &'a str>) -> usize {
        let index = self.key_lists.len();
        let keys = keys
            .map(|key| self.push_name(key))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        self.key_lists.push(keys);
        index
    }

    fn push_assign_path(&mut self, steps: &[AssignPathStep]) -> usize {
        let index = self.assign_paths.len();
        let mut dynamic_index_count = 0;
        let steps = steps
            .iter()
            .map(|step| match step {
                AssignPathStep::Field(field) => {
                    CompiledAssignPathStep::Field(self.push_name(field))
                }
                AssignPathStep::Index(_) => {
                    dynamic_index_count += 1;
                    CompiledAssignPathStep::Index
                }
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        self.assign_paths.push(CompiledAssignPath {
            steps,
            dynamic_index_count,
        });
        index
    }

    fn push_format_template(&mut self, template: &str, argc: usize) -> usize {
        let index = self.format_templates.len();
        self.format_templates
            .push(compile_format_template(template, argc));
        index
    }

    fn push_compiled_schema(&mut self, schema: &Value) -> usize {
        let index = self.compiled_schemas.len();
        self.compiled_schemas.push(compile_schema_value(schema));
        index
    }

    fn push_lowered_loop(&mut self, lowered_loop: LoweredLoop) -> usize {
        let index = self.lowered_loops.len();
        self.lowered_loops.push(lowered_loop);
        index
    }

    fn ensure_const_slot(&mut self, slot: usize) {
        if self.const_slots.len() <= slot {
            self.const_slots.resize(slot + 1, None);
        }
    }

    fn set_const_slot(&mut self, slot: usize, value: Option<Value>) {
        self.ensure_const_slot(slot);
        self.const_slots[slot] = value;
    }

    fn clear_const_slots(&mut self) {
        self.const_slots.fill(None);
    }

    fn const_for_slot(&self, slot: usize) -> Option<Value> {
        self.const_slots.get(slot).cloned().flatten()
    }

    fn const_for_name(&self, name: &str) -> Option<Value> {
        let symbol = lookup_symbol(name)?;
        let slots = self.slots.borrow();
        let slot = *slots.lookup.get(&symbol)?;
        drop(slots);
        self.const_for_slot(slot)
    }

    fn resolve_intrinsic(&mut self, name: &str, argc: usize) -> IntrinsicOp {
        let valid = match name {
            "len" | "empty" | "keys" | "values" | "trim" | "to_string" | "to_int" | "to_float"
            | "json_parse" => argc == 1,
            "contains" | "grep_text" | "starts_with" | "ends_with" | "split" | "join"
            | "validate" | "ceil_div" | "floor_div" | "push" => argc == 2,
            "slice" => argc == 3,
            "find" => argc == 2 || argc == 3,
            "format" => argc >= 1,
            "range" => (1..=3).contains(&argc),
            _ => true,
        };
        if !valid {
            return IntrinsicOp::InvalidArity {
                name: self.push_name(name),
                argc,
            };
        }
        match name {
            "len" => IntrinsicOp::Len,
            "empty" => IntrinsicOp::Empty,
            "keys" => IntrinsicOp::Keys,
            "values" => IntrinsicOp::Values,
            "contains" => IntrinsicOp::Contains,
            "find" => IntrinsicOp::Find(argc),
            "grep_text" => IntrinsicOp::GrepText,
            "starts_with" => IntrinsicOp::StartsWith,
            "ends_with" => IntrinsicOp::EndsWith,
            "split" => IntrinsicOp::Split,
            "join" => IntrinsicOp::Join,
            "trim" => IntrinsicOp::Trim,
            "slice" => IntrinsicOp::Slice,
            "to_string" => IntrinsicOp::ToString,
            "to_int" => IntrinsicOp::ToInt,
            "to_float" => IntrinsicOp::ToFloat,
            "json_parse" => IntrinsicOp::JsonParse,
            "format" => IntrinsicOp::Format(argc),
            "validate" => IntrinsicOp::Validate,
            "range" => IntrinsicOp::Range(argc),
            "ceil_div" => IntrinsicOp::CeilDiv,
            "floor_div" => IntrinsicOp::FloorDiv,
            "push" => IntrinsicOp::Push,
            _ => IntrinsicOp::Unknown {
                name: self.push_name(name),
                argc,
            },
        }
    }

    fn compile_program_block(&mut self, program: &Program) {
        match &program.main {
            Expr::Block(expressions) => {
                self.compile_block_value_with_spans(expressions, &program.expression_spans);
            }
            expression => self.compile_expr(expression),
        }
        if !is_terminal_expr(&program.main) {
            let pop = self.code.len();
            self.code.push(Instruction::Pop);
            if let Some(span) = program.expression_spans.last().copied() {
                self.mark_instruction_spans(pop, self.code.len(), span);
            }
        }
    }

    fn compile_block_value(&mut self, expressions: &[Expr]) {
        let Some((last, prefix)) = expressions.split_last() else {
            self.code.push(Instruction::PushNull);
            return;
        };
        for expression in prefix {
            self.compile_expr_discarding_value(expression);
        }
        self.compile_expr(last);
    }

    fn compile_block_value_with_spans(&mut self, expressions: &[Expr], spans: &[Span]) {
        let Some((last, prefix)) = expressions.split_last() else {
            self.code.push(Instruction::PushNull);
            return;
        };
        for (index, expression) in prefix.iter().enumerate() {
            let span = spans.get(index).copied();
            self.compile_expr_discarding_value_with_span(expression, span);
        }
        self.compile_expr_with_span(last, spans.get(expressions.len() - 1).copied());
    }

    fn compile_expr_with_span(&mut self, expression: &Expr, span: Option<Span>) {
        let start = self.code.len();
        self.compile_expr(expression);
        if let Some(span) = span {
            self.mark_instruction_spans(start, self.code.len(), span);
        }
    }

    fn compile_expr_discarding_value_with_span(&mut self, expression: &Expr, span: Option<Span>) {
        let start = self.code.len();
        self.compile_expr_discarding_value(expression);
        if let Some(span) = span {
            self.mark_instruction_spans(start, self.code.len(), span);
        }
    }

    fn compile_expr_discarding_value(&mut self, expression: &Expr) {
        match expression {
            Expr::Block(expressions) => {
                for expression in expressions {
                    self.compile_expr_discarding_value(expression);
                }
            }
            Expr::Assign { target, expr } => self.compile_assignment_expr(target, expr, false),
            Expr::For {
                binding,
                iterable,
                body,
            } => self.compile_for_expr(binding, iterable, body, false),
            Expr::Submit(_) | Expr::Finish(_) | Expr::Fail(_) | Expr::Break | Expr::Continue => {
                self.compile_expr(expression);
            }
            expression => {
                self.compile_expr(expression);
                self.code.push(Instruction::Pop);
            }
        }
    }

    fn mark_instruction_spans(&mut self, start: usize, end: usize, span: Span) {
        if self.spans.len() < end {
            self.spans.resize(end, None);
        }
        for instruction_span in &mut self.spans[start..end] {
            *instruction_span = Some(span);
        }
    }

    fn compile_block_discarding_values(&mut self, block: &Expr) {
        match block {
            Expr::Block(expressions) => {
                for expression in expressions {
                    self.compile_expr_discarding_value(expression);
                }
            }
            expression => {
                self.compile_expr_discarding_value(expression);
            }
        }
    }

    fn push_null_if(&mut self, leave_value: bool) {
        if leave_value {
            self.code.push(Instruction::PushNull);
        }
    }

    fn compile_assignment_expr(&mut self, target: &AssignTarget, expr: &Expr, leave_value: bool) {
        if target.is_simple() {
            let name = &target.root;
            let slot = self.push_slot(name);
            let has_type_literal = contains_type_literal(expr);
            let const_value = if let Expr::TypeLiteral(ty) = expr {
                fold_type(ty).map(wrap_type_schema_value)
            } else if has_type_literal {
                None
            } else {
                self.fold_compile_time_expr(expr)
            };
            if let Expr::Binary {
                left,
                op: BinaryOp::Add,
                right,
            } = expr
                && matches!(left.as_ref(), Expr::Variable(var) if var == name)
            {
                if let Expr::List(items) = right.as_ref()
                    && items.len() == 1
                {
                    self.compile_expr(&items[0]);
                    self.code.push(Instruction::AppendAssign(slot));
                    self.set_const_slot(slot, None);
                    self.push_null_if(leave_value);
                    return;
                }
                if let Some(Value::Number(right)) = self.fold_compile_time_expr(right) {
                    self.code.push(Instruction::AddAssignNumber { slot, right });
                    self.set_const_slot(slot, None);
                    self.push_null_if(leave_value);
                    return;
                }
                if let Expr::Variable(right_name) = right.as_ref() {
                    let right = self.push_slot(right_name);
                    self.code.push(Instruction::AddAssignSlot { slot, right });
                    self.set_const_slot(slot, None);
                    self.push_null_if(leave_value);
                    return;
                }
                self.compile_expr(right);
                self.code.push(Instruction::AddAssign(slot));
                self.set_const_slot(slot, None);
                self.push_null_if(leave_value);
                return;
            }
            if let Expr::BuiltinCall {
                name: builtin_name,
                args,
            } = expr
                && builtin_name == "push"
                && let [Expr::Variable(first_arg), item] = args.as_slice()
                && first_arg == name
            {
                self.compile_expr(item);
                self.code
                    .push(Instruction::Intrinsic(IntrinsicOp::PushAssign(slot)));
                self.set_const_slot(slot, None);
                self.push_null_if(leave_value);
                return;
            }
            if let Some(value) = const_value.clone()
                && !has_type_literal
            {
                let constant = self.push_const(value);
                self.code.push(Instruction::StoreConst { slot, constant });
                self.set_const_slot(slot, const_value);
                self.push_null_if(leave_value);
                return;
            }

            self.compile_expr(expr);
            self.code.push(Instruction::StoreName(slot));
            self.set_const_slot(slot, const_value);
            self.push_null_if(leave_value);
            return;
        }

        let slot = self.push_slot(&target.root);
        if let [AssignPathStep::Index(index)] = target.steps.as_slice()
            && is_pure_expr(index)
            && let Expr::Binary {
                left,
                op: BinaryOp::Add,
                right,
            } = expr
            && let Expr::Index {
                target: left_target,
                index: left_index,
            } = left.as_ref()
            && matches!(left_target.as_ref(), Expr::Variable(name) if name == target.root)
            && left_index.as_ref() == index
            && let Some(Value::Number(right)) = self.fold_compile_time_expr(right)
        {
            if let Expr::Variable(index_name) = index {
                let index = self.push_slot(index_name);
                self.code
                    .push(Instruction::AddAssignIndexSlotNumber { slot, index, right });
            } else {
                self.compile_expr(index);
                self.code
                    .push(Instruction::AddAssignIndexNumber { slot, right });
            }
            self.set_const_slot(slot, None);
            self.push_null_if(leave_value);
            return;
        }
        for step in &target.steps {
            if let AssignPathStep::Index(index) = step {
                self.compile_expr(index);
            }
        }
        self.compile_expr(expr);
        let path = self.push_assign_path(&target.steps);
        self.code.push(Instruction::PathAssign { slot, path });
        self.set_const_slot(slot, None);
        self.push_null_if(leave_value);
    }

    fn compile_for_expr(&mut self, binding: &str, iterable: &Expr, body: &Expr, leave_value: bool) {
        if let Some(loop_id) = self.compile_lowered_for(binding, iterable, body) {
            self.clear_const_slots();
            self.code.push(Instruction::LoweredLoop(loop_id));
            self.push_null_if(leave_value);
            return;
        }

        let binding = self.push_slot(binding);
        if let Expr::BuiltinCall { name, args } = iterable
            && name.as_str() == "range"
        {
            for arg in args {
                self.compile_expr(arg);
            }
            self.clear_const_slots();
            self.set_const_slot(binding, None);
            self.code.push(Instruction::BeginRangeIter {
                binding,
                argc: args.len(),
            });
            self.compile_for_loop_body(body);
            self.push_null_if(leave_value);
            return;
        }

        self.compile_expr(iterable);
        self.clear_const_slots();
        self.set_const_slot(binding, None);
        self.code.push(Instruction::BeginIter(binding));
        self.compile_for_loop_body(body);
        self.push_null_if(leave_value);
    }

    fn compile_for_loop_body(&mut self, body: &Expr) {
        let loop_start = self.code.len();
        let iter_next = self.code.len();
        self.code.push(Instruction::IterNext {
            jump_to: usize::MAX,
        });
        self.loop_contexts.push(LoopContext {
            continue_target: loop_start,
            break_jumps: SmallVec::new(),
        });
        self.compile_block_discarding_values(body);
        let loop_context = self
            .loop_contexts
            .pop()
            .expect("loop context should exist while compiling `for`");
        self.code.push(Instruction::Jump(loop_start));
        let loop_end = self.code.len();
        self.code.push(Instruction::EndIter);
        self.patch_jump(iter_next, loop_end);
        for break_jump in loop_context.break_jumps {
            self.patch_jump(break_jump, loop_end);
        }
        self.clear_const_slots();
    }

    fn compile_lowered_for(
        &mut self,
        binding_name: &str,
        iterable: &Expr,
        body: &Expr,
    ) -> Option<usize> {
        let binding = self.push_slot(binding_name);
        let iterable = self.optimize_loop_iterable(iterable)?;
        let mut lowerer = LoweredLoopOptimizer {
            compiler: self,
            binding_name,
            assigned_slots: SmallVec::new(),
        };
        let body = lowerer.optimize_block(body)?;
        lowerer.compiler.set_const_slot(binding, None);
        for slot in lowerer.assigned_slots {
            lowerer.compiler.set_const_slot(slot, None);
        }
        let lowered_loop = LoweredLoop {
            binding,
            iterable,
            body: body.into_boxed_slice(),
        };
        Some(lowerer.compiler.push_lowered_loop(lowered_loop))
    }

    fn optimize_loop_iterable(&mut self, iterable: &Expr) -> Option<LoopIterable> {
        match iterable {
            Expr::BuiltinCall { name, args }
                if name == "range" && (1..=3).contains(&args.len()) =>
            {
                Some(LoopIterable::Range(
                    args.iter()
                        .map(|arg| self.optimize_loop_expr(arg))
                        .collect::<Option<Vec<_>>>()?
                        .into_boxed_slice(),
                ))
            }
            Expr::BuiltinCall { name, args } if name == "keys" && args.len() == 1 => Some(
                LoopIterable::Keys(Box::new(self.optimize_loop_expr(&args[0])?)),
            ),
            _ => Some(LoopIterable::Values(Box::new(
                self.optimize_loop_expr(iterable)?,
            ))),
        }
    }

    fn optimize_loop_expr(&mut self, expr: &Expr) -> Option<LoopExpr> {
        match expr {
            Expr::Null => Some(LoopExpr::Const(Value::Null)),
            Expr::Bool(value) => Some(LoopExpr::Const(Value::Bool(*value))),
            Expr::Number(value) => Some(LoopExpr::Const(Value::Number(*value))),
            Expr::String(value) => Some(LoopExpr::Const(Value::String(value.clone()))),
            Expr::ResourceRef(resource) => Some(LoopExpr::Const(Value::Resource(
                super::ResourceHandle::new(
                    resource.resource_type.to_string(),
                    resource.alias.to_string(),
                ),
            ))),
            Expr::Variable(name) => Some(LoopExpr::Slot(self.push_slot(name))),
            Expr::List(items) => Some(LoopExpr::List(
                items
                    .iter()
                    .map(|item| self.optimize_loop_expr(item))
                    .collect::<Option<Vec<_>>>()?
                    .into_boxed_slice(),
            )),
            Expr::Record(entries) => Some(LoopExpr::Record(
                entries
                    .iter()
                    .map(|(key, value)| {
                        Some((self.push_name(key), self.optimize_loop_expr(value)?))
                    })
                    .collect::<Option<Vec<_>>>()?
                    .into_boxed_slice(),
            )),
            Expr::BuiltinCall { name, args } => {
                if name == "validate" {
                    return None;
                }
                if name == "format"
                    && let Some((Expr::String(template), value_args)) = args.split_first()
                {
                    return Some(LoopExpr::Format {
                        template: compile_format_template(template, value_args.len()),
                        args: value_args
                            .iter()
                            .map(|arg| self.optimize_loop_expr(arg))
                            .collect::<Option<Vec<_>>>()?
                            .into_boxed_slice(),
                    });
                }
                Some(LoopExpr::Intrinsic {
                    op: self.resolve_intrinsic(name, args.len()),
                    args: args
                        .iter()
                        .map(|arg| self.optimize_loop_expr(arg))
                        .collect::<Option<Vec<_>>>()?
                        .into_boxed_slice(),
                })
            }
            Expr::Field { target, field } => Some(LoopExpr::Field {
                target: Box::new(self.optimize_loop_expr(target)?),
                field: self.push_name(field),
            }),
            Expr::Index { target, index } => Some(LoopExpr::Index {
                target: Box::new(self.optimize_loop_expr(target)?),
                index: Box::new(self.optimize_loop_expr(index)?),
            }),
            Expr::Unary { op, expr } => Some(LoopExpr::Unary {
                op: *op,
                expr: Box::new(self.optimize_loop_expr(expr)?),
            }),
            Expr::If {
                condition,
                then_block,
                else_block,
            } => Some(LoopExpr::Conditional {
                condition: Box::new(self.optimize_loop_expr(condition)?),
                then_expr: Box::new(self.optimize_loop_expr(then_block)?),
                else_expr: Box::new(self.optimize_loop_expr(else_block)?),
            }),
            Expr::Binary { left, op, right } => Some(LoopExpr::Binary {
                left: Box::new(self.optimize_loop_expr(left)?),
                op: *op,
                right: Box::new(self.optimize_loop_expr(right)?),
            }),
            Expr::Block(_)
            | Expr::Assign { .. }
            | Expr::For { .. }
            | Expr::Break
            | Expr::Continue
            | Expr::ReceiverCall { .. }
            | Expr::StartProcess(_)
            | Expr::Await(_)
            | Expr::SleepFor(_)
            | Expr::SleepUntil(_)
            | Expr::WaitSignal
            | Expr::SignalRun { .. }
            | Expr::ResultUnwrap(_)
            | Expr::Cancel(_)
            | Expr::Print(_)
            | Expr::Submit(_)
            | Expr::Yield(_)
            | Expr::Wake(_)
            | Expr::Finish(_)
            | Expr::Fail(_)
            | Expr::TypeLiteral(_) => None,
        }
    }

    fn fold_compile_time_expr(&self, expr: &Expr) -> Option<Value> {
        match expr {
            Expr::Null => Some(Value::Null),
            Expr::Bool(value) => Some(Value::Bool(*value)),
            Expr::Number(value) => Some(Value::Number(*value)),
            Expr::String(value) => Some(Value::String(value.clone())),
            Expr::ResourceRef(resource) => Some(Value::Resource(super::ResourceHandle::new(
                resource.resource_type.to_string(),
                resource.alias.to_string(),
            ))),
            Expr::Variable(name) => self.const_for_name(name),
            Expr::List(items) => Some(Value::List(
                items
                    .iter()
                    .map(|item| self.fold_compile_time_expr(item))
                    .collect::<Option<Vec<_>>>()?
                    .into(),
            )),
            Expr::Record(entries) => {
                let mut record = record_with_capacity(entries.len());
                for (key, value) in entries {
                    record.insert(key.to_string(), self.fold_compile_time_expr(value)?);
                }
                Some(Value::Record(Arc::new(record)))
            }
            Expr::BuiltinCall { name, args } => {
                let values = args
                    .iter()
                    .map(|arg| self.fold_compile_time_expr(arg))
                    .collect::<Option<Vec<_>>>()?;
                let builtin = match name.as_str() {
                    "len" => IntrinsicOp::Len,
                    "empty" => IntrinsicOp::Empty,
                    "keys" => IntrinsicOp::Keys,
                    "values" => IntrinsicOp::Values,
                    "contains" => IntrinsicOp::Contains,
                    "find" => IntrinsicOp::Find(args.len()),
                    "grep_text" => IntrinsicOp::GrepText,
                    "starts_with" => IntrinsicOp::StartsWith,
                    "ends_with" => IntrinsicOp::EndsWith,
                    "split" => IntrinsicOp::Split,
                    "join" => IntrinsicOp::Join,
                    "trim" => IntrinsicOp::Trim,
                    "slice" => IntrinsicOp::Slice,
                    "to_string" => IntrinsicOp::ToString,
                    "to_int" => IntrinsicOp::ToInt,
                    "to_float" => IntrinsicOp::ToFloat,
                    "json_parse" => IntrinsicOp::JsonParse,
                    "format" => IntrinsicOp::Format(args.len()),
                    "validate" => IntrinsicOp::Validate,
                    "range" => IntrinsicOp::Range(args.len()),
                    "ceil_div" => IntrinsicOp::CeilDiv,
                    "floor_div" => IntrinsicOp::FloorDiv,
                    "push" => IntrinsicOp::Push,
                    _ => return None,
                };
                match builtin {
                    IntrinsicOp::Len => {
                        if values.len() == 1 {
                            execute_len_direct(&values[0]).ok()
                        } else {
                            None
                        }
                    }
                    IntrinsicOp::Range(_) => execute_range_builtin(&values).ok(),
                    IntrinsicOp::CeilDiv => {
                        execute_integer_div_builtin("ceil_div", &values, f64::ceil).ok()
                    }
                    IntrinsicOp::FloorDiv => {
                        execute_integer_div_builtin("floor_div", &values, f64::floor).ok()
                    }
                    IntrinsicOp::Push => {
                        if let [Value::List(items), item] = values.as_slice() {
                            let mut values = items.to_vec();
                            values.push(item.clone());
                            Some(Value::List(values.into()))
                        } else {
                            None
                        }
                    }
                    _ => None,
                }
            }
            Expr::Field { target, field } => {
                let target = self.fold_compile_time_expr(target)?;
                read_field_direct(target, &transient_name(field)).ok()
            }
            Expr::Index { target, index } => {
                let target = self.fold_compile_time_expr(target)?;
                let index = self.fold_compile_time_expr(index)?;
                read_index_direct(target, index).ok()
            }
            Expr::Unary { op, expr } => {
                let value = self.fold_compile_time_expr(expr)?;
                match op {
                    UnaryOp::Negate => Some(Value::Number(-as_number(&value).ok()?)),
                    UnaryOp::Not => Some(Value::Bool(!is_truthy(&value))),
                }
            }
            Expr::If {
                condition,
                then_block,
                else_block,
            } => {
                if is_truthy(&self.fold_compile_time_expr(condition)?) {
                    self.fold_compile_time_expr(then_block)
                } else {
                    self.fold_compile_time_expr(else_block)
                }
            }
            Expr::Binary { left, op, right } => match op {
                BinaryOp::And => {
                    let left = self.fold_compile_time_expr(left)?;
                    if !is_truthy(&left) {
                        Some(Value::Bool(false))
                    } else {
                        Some(Value::Bool(is_truthy(&self.fold_compile_time_expr(right)?)))
                    }
                }
                BinaryOp::Or => {
                    let left = self.fold_compile_time_expr(left)?;
                    if is_truthy(&left) {
                        Some(Value::Bool(true))
                    } else {
                        Some(Value::Bool(is_truthy(&self.fold_compile_time_expr(right)?)))
                    }
                }
                _ => {
                    let left = self.fold_compile_time_expr(left)?;
                    let right = self.fold_compile_time_expr(right)?;
                    eval_binary_values(left, *op, right).ok()
                }
            },
            Expr::TypeLiteral(ty) => {
                let schema = fold_type(ty)?;
                let mut wrapper = record_with_capacity(1);
                wrapper.insert(LASH_TYPE_KEY.to_string(), schema);
                Some(Value::Record(Arc::new(wrapper)))
            }
            Expr::Block(_)
            | Expr::Assign { .. }
            | Expr::For { .. }
            | Expr::Break
            | Expr::Continue
            | Expr::ReceiverCall { .. }
            | Expr::StartProcess(_)
            | Expr::Await(_)
            | Expr::SleepFor(_)
            | Expr::SleepUntil(_)
            | Expr::WaitSignal
            | Expr::SignalRun { .. }
            | Expr::ResultUnwrap(_)
            | Expr::Cancel(_)
            | Expr::Print(_)
            | Expr::Submit(_)
            | Expr::Yield(_)
            | Expr::Wake(_)
            | Expr::Finish(_)
            | Expr::Fail(_) => None,
        }
    }

    fn emit_builtin_call(&mut self, name: &str, args: &[Expr]) {
        if name == "format"
            && let Some((Expr::String(template), value_args)) = args.split_first()
        {
            if let [Expr::Variable(slot_name)] = value_args {
                let template = self.push_format_template(template, value_args.len());
                let slot = self.push_slot(slot_name);
                self.code.push(Instruction::Intrinsic(
                    IntrinsicOp::FormatCompiledSlotNumber { template, slot },
                ));
                return;
            }
            if let [Expr::Binary { left, op, right }] = value_args
                && is_numeric_binary_op(*op)
                && let (Expr::Variable(slot_name), Some(Value::Number(right))) =
                    (left.as_ref(), self.fold_compile_time_expr(right))
            {
                let template = self.push_format_template(template, value_args.len());
                let slot = self.push_slot(slot_name);
                self.code.push(Instruction::Intrinsic(
                    IntrinsicOp::FormatCompiledSlotNumberBinary {
                        template,
                        slot,
                        op: *op,
                        right,
                    },
                ));
                return;
            }
            for arg in value_args {
                self.compile_expr(arg);
            }
            let template = self.push_format_template(template, value_args.len());
            self.code
                .push(Instruction::Intrinsic(IntrinsicOp::FormatCompiled(
                    template,
                )));
            return;
        }

        match (name, args.len()) {
            ("len", 1) => {
                self.compile_expr(&args[0]);
                self.code.push(Instruction::Intrinsic(IntrinsicOp::Len));
            }
            ("join", 2) => {
                self.compile_expr(&args[0]);
                self.compile_expr(&args[1]);
                self.code.push(Instruction::Intrinsic(IntrinsicOp::Join));
            }
            ("validate", 2) => {
                if let Some(schema_wrapper) = self.fold_compile_time_expr(&args[1])
                    && let Some(schema) = unwrap_type_value(&schema_wrapper).cloned()
                {
                    self.compile_expr(&args[0]);
                    let schema = self.push_compiled_schema(&schema);
                    self.code
                        .push(Instruction::Intrinsic(IntrinsicOp::ValidateCompiled(
                            schema,
                        )));
                    return;
                }

                self.compile_expr(&args[0]);
                self.compile_expr(&args[1]);
                self.code
                    .push(Instruction::Intrinsic(IntrinsicOp::Validate));
            }
            ("push", 2) => {
                self.compile_expr(&args[0]);
                self.compile_expr(&args[1]);
                self.code.push(Instruction::Intrinsic(IntrinsicOp::Push));
            }
            ("range", 1..=3) => {
                for arg in args {
                    self.compile_expr(arg);
                }
                self.code
                    .push(Instruction::Intrinsic(IntrinsicOp::Range(args.len())));
            }
            _ => {
                for arg in args {
                    self.compile_expr(arg);
                }
                let builtin = self.resolve_intrinsic(name, args.len());
                self.code.push(Instruction::Intrinsic(builtin));
            }
        }
    }

    fn compile_expr(&mut self, expr: &Expr) {
        if !contains_type_literal(expr)
            && let Some(value) = self.fold_compile_time_expr(expr)
        {
            self.emit_push_value(value);
            return;
        }

        match expr {
            Expr::Block(expressions) => self.compile_block_value(expressions),
            Expr::Assign { target, expr } => self.compile_assignment_expr(target, expr, true),
            Expr::For {
                binding,
                iterable,
                body,
            } => self.compile_for_expr(binding, iterable, body, true),
            Expr::Break => {
                let jump = self.emit_jump();
                self.loop_contexts
                    .last_mut()
                    .expect("parser rejects `break` outside loops")
                    .break_jumps
                    .push(jump);
                self.clear_const_slots();
            }
            Expr::Continue => {
                let continue_target = self
                    .loop_contexts
                    .last()
                    .expect("parser rejects `continue` outside loops")
                    .continue_target;
                self.code.push(Instruction::Jump(continue_target));
                self.clear_const_slots();
            }
            Expr::Null => {
                self.code.push(Instruction::PushNull);
            }
            Expr::Bool(value) => {
                self.code.push(Instruction::PushBool(*value));
            }
            Expr::Number(value) => {
                self.code.push(Instruction::PushNumber(*value));
            }
            Expr::String(value) => {
                let value = self.push_const(Value::String(value.clone()));
                self.code.push(Instruction::PushConst(value));
            }
            Expr::Variable(name) => {
                let name = self.push_slot(name);
                if let Some(value) = self.const_for_slot(name) {
                    self.emit_push_value(value);
                } else {
                    self.code.push(Instruction::LoadName(name));
                }
            }
            Expr::List(items) => {
                for item in items {
                    self.compile_expr(item);
                }
                self.code.push(Instruction::BuildList(items.len()));
            }
            Expr::Record(entries) => {
                for (_, value) in entries {
                    self.compile_expr(value);
                }
                let keys = self.push_key_list(entries.iter().map(|(key, _)| key.as_str()));
                self.code.push(Instruction::BuildRecord(keys));
            }
            Expr::StartProcess(process) => self.compile_start_process_expr(process),
            Expr::ResourceRef(resource) => {
                self.emit_push_value(Value::Resource(super::ResourceHandle::new(
                    resource.resource_type.to_string(),
                    resource.alias.to_string(),
                )));
            }
            Expr::ReceiverCall {
                receiver,
                operation,
                args,
            } => self.compile_receiver_call_expr(receiver, operation, args, false),
            Expr::Await(handle) => match handle.as_ref() {
                Expr::ReceiverCall {
                    receiver,
                    operation,
                    args,
                } => self.compile_receiver_call_expr(receiver, operation, args, false),
                Expr::ResultUnwrap(inner) => {
                    if let Expr::ReceiverCall {
                        receiver,
                        operation,
                        args,
                    } = inner.as_ref()
                    {
                        self.compile_receiver_call_expr(receiver, operation, args, true);
                    } else {
                        self.compile_expr(inner);
                        self.code.push(Instruction::AwaitHandleUnwrap);
                    }
                }
                _ => {
                    self.compile_expr(handle);
                    self.code.push(Instruction::AwaitHandle);
                }
            },
            Expr::SleepFor(duration) => {
                self.compile_expr(duration);
                self.code.push(Instruction::ProcessSleepFor);
            }
            Expr::SleepUntil(deadline) => {
                self.compile_expr(deadline);
                self.code.push(Instruction::ProcessSleepUntil);
            }
            Expr::WaitSignal => {
                self.code.push(Instruction::ProcessWaitSignal);
            }
            Expr::SignalRun { run, payload } => {
                self.compile_expr(run);
                self.compile_expr(payload);
                self.code.push(Instruction::ProcessSignalRun);
            }
            Expr::ResultUnwrap(expr) => {
                if let Expr::ReceiverCall {
                    receiver,
                    operation,
                    args,
                } = expr.as_ref()
                {
                    self.compile_receiver_call_expr(receiver, operation, args, true);
                } else if let Expr::Await(handle) = expr.as_ref() {
                    self.compile_expr(handle);
                    self.code.push(Instruction::AwaitHandleUnwrap);
                } else if let Expr::Field { target, field } = expr.as_ref()
                    && let Expr::Variable(name) = target.as_ref()
                {
                    let slot = self.push_slot(name);
                    let field = self.push_name(field);
                    self.code.push(Instruction::LoadFieldUnwrap { slot, field });
                } else {
                    self.compile_expr(expr);
                    self.code.push(Instruction::ResultUnwrap);
                }
            }
            Expr::BuiltinCall { name, args } => {
                self.emit_builtin_call(name, args);
            }
            Expr::Field { target, field } => {
                if let Expr::Variable(name) = target.as_ref() {
                    let slot = self.push_slot(name);
                    let field = self.push_name(field);
                    self.code.push(Instruction::LoadField { slot, field });
                    return;
                }
                self.compile_expr(target);
                let field = self.push_name(field);
                self.code.push(Instruction::Field(field));
            }
            Expr::Index { target, index } => {
                self.compile_expr(target);
                self.compile_expr(index);
                self.code.push(Instruction::Index);
            }
            Expr::Unary { op, expr } => {
                self.compile_expr(expr);
                self.code.push(Instruction::Unary(*op));
            }
            Expr::If {
                condition,
                then_block,
                else_block,
            } => {
                let jump_to_else = self.compile_condition_jump_if_false(condition);
                let const_slots_before_branches = self.const_slots.clone();
                self.compile_expr(then_block);
                let jump_to_end = self.emit_jump();
                self.patch_jump(jump_to_else, self.code.len());
                self.const_slots = const_slots_before_branches;
                self.compile_expr(else_block);
                self.patch_jump(jump_to_end, self.code.len());
                self.clear_const_slots();
            }
            Expr::Cancel(handle) => {
                self.compile_expr(handle);
                self.code.push(Instruction::CancelHandle);
            }
            Expr::Print(expr) => {
                self.compile_expr(expr);
                self.code.push(Instruction::Print);
            }
            Expr::Submit(expr) => {
                if let Some(expr) = expr {
                    self.compile_expr(expr);
                } else {
                    self.compile_expr(&Expr::Null);
                }
                self.code.push(Instruction::Submit);
            }
            Expr::Yield(expr) => {
                self.compile_expr(expr);
                self.code.push(Instruction::ProcessYield);
            }
            Expr::Wake(expr) => {
                self.compile_expr(expr);
                self.code.push(Instruction::ProcessWake);
            }
            Expr::Finish(expr) => {
                if let Some(expr) = expr {
                    self.compile_expr(expr);
                } else {
                    self.compile_expr(&Expr::Null);
                }
                self.code.push(Instruction::ProcessFinish);
            }
            Expr::Fail(expr) => {
                self.compile_expr(expr);
                self.code.push(Instruction::ProcessFail);
            }
            Expr::TypeLiteral(ty) => self.compile_type_literal(ty),
            Expr::Binary { left, op, right } => match op {
                BinaryOp::And => {
                    self.compile_expr(left);
                    let jump_to_false = self.emit_jump_if_false();
                    self.compile_expr(right);
                    self.code.push(Instruction::ToBool);
                    let jump_to_end = self.emit_jump();
                    self.patch_jump(jump_to_false, self.code.len());
                    self.code.push(Instruction::PushBool(false));
                    self.patch_jump(jump_to_end, self.code.len());
                }
                BinaryOp::Or => {
                    self.compile_expr(left);
                    let jump_to_true = self.emit_jump_if_true();
                    self.compile_expr(right);
                    self.code.push(Instruction::ToBool);
                    let jump_to_end = self.emit_jump();
                    self.patch_jump(jump_to_true, self.code.len());
                    self.code.push(Instruction::PushBool(true));
                    self.patch_jump(jump_to_end, self.code.len());
                }
                _ => {
                    if is_comparison_binary_op(*op) {
                        if let (
                            Expr::Binary {
                                left: inner_left,
                                op: binary_op,
                                right: inner_right,
                            },
                            Some(Value::Number(compare_right)),
                        ) = (left.as_ref(), self.fold_compile_time_expr(right))
                            && is_numeric_binary_op(*binary_op)
                            && let (Expr::Variable(name), Some(Value::Number(binary_right))) = (
                                inner_left.as_ref(),
                                self.fold_compile_time_expr(inner_right),
                            )
                        {
                            let slot = self.push_slot(name);
                            self.code.push(Instruction::SlotNumberBinaryCompare {
                                slot,
                                binary_op: *binary_op,
                                binary_right,
                                compare_op: *op,
                                compare_right,
                            });
                            return;
                        }
                        if let (Expr::Variable(name), Some(Value::Number(right))) =
                            (left.as_ref(), self.fold_compile_time_expr(right))
                        {
                            let slot = self.push_slot(name);
                            self.code.push(Instruction::SlotNumberCompare {
                                slot,
                                op: *op,
                                right,
                            });
                            return;
                        }
                    }
                    if is_numeric_binary_op(*op)
                        && let (Expr::Variable(name), Some(Value::Number(right))) =
                            (left.as_ref(), self.fold_compile_time_expr(right))
                    {
                        let slot = self.push_slot(name);
                        self.code.push(Instruction::SlotNumberBinary {
                            slot,
                            op: *op,
                            right,
                        });
                        return;
                    }
                    self.compile_expr(left);
                    self.compile_expr(right);
                    self.code.push(Instruction::Binary(*op));
                }
            },
        }
    }

    fn compile_start_process_expr(&mut self, process: &ProcessStartExpr) {
        for (_, expr) in &process.args {
            self.compile_expr(expr);
        }
        let keys = self.push_key_list(process.args.iter().map(|(name, _)| name.as_str()));
        let process = self.push_name(&process.process);
        self.code.push(Instruction::StartProcess { process, keys });
    }

    fn compile_receiver_call_expr(
        &mut self,
        receiver: &Expr,
        operation: &str,
        args: &[Expr],
        unwrap: bool,
    ) {
        self.compile_expr(receiver);
        for arg in args {
            self.compile_expr(arg);
        }
        let operation = self.push_name(operation);
        if unwrap {
            self.code.push(Instruction::ResourceCallUnwrap {
                operation,
                argc: args.len(),
            });
        } else {
            self.code.push(Instruction::ResourceCall {
                operation,
                argc: args.len(),
            });
        }
    }

    fn emit_jump_if_false(&mut self) -> usize {
        let index = self.code.len();
        self.code.push(Instruction::JumpIfFalse(usize::MAX));
        index
    }

    fn compile_condition_jump_if_false(&mut self, condition: &Expr) -> usize {
        if !contains_type_literal(condition)
            && let Some(value) = self.fold_compile_time_expr(condition)
        {
            self.emit_push_value(value);
            return self.emit_jump_if_false();
        }

        if let Expr::Binary { left, op, right } = condition
            && is_comparison_binary_op(*op)
        {
            if let (
                Expr::Binary {
                    left: inner_left,
                    op: binary_op,
                    right: inner_right,
                },
                Some(Value::Number(compare_right)),
            ) = (left.as_ref(), self.fold_compile_time_expr(right))
                && is_numeric_binary_op(*binary_op)
                && let (Expr::Variable(name), Some(Value::Number(binary_right))) = (
                    inner_left.as_ref(),
                    self.fold_compile_time_expr(inner_right),
                )
            {
                let slot = self.push_slot(name);
                let index = self.code.len();
                self.code
                    .push(Instruction::JumpIfSlotNumberBinaryCompareFalse {
                        slot,
                        binary_op: *binary_op,
                        binary_right,
                        compare_op: *op,
                        compare_right,
                        target: usize::MAX,
                    });
                return index;
            }

            if let (Expr::Variable(name), Some(Value::Number(right))) =
                (left.as_ref(), self.fold_compile_time_expr(right))
            {
                let slot = self.push_slot(name);
                let index = self.code.len();
                self.code.push(Instruction::JumpIfSlotNumberCompareFalse {
                    slot,
                    op: *op,
                    right,
                    target: usize::MAX,
                });
                return index;
            }
            self.compile_expr(left);
            self.compile_expr(right);
            let index = self.code.len();
            self.code.push(Instruction::JumpIfCompareFalse {
                op: *op,
                target: usize::MAX,
            });
            return index;
        }

        self.compile_expr(condition);
        self.emit_jump_if_false()
    }

    fn emit_jump_if_true(&mut self) -> usize {
        let index = self.code.len();
        self.code.push(Instruction::JumpIfTrue(usize::MAX));
        index
    }

    fn emit_jump(&mut self) -> usize {
        let index = self.code.len();
        self.code.push(Instruction::Jump(usize::MAX));
        index
    }

    fn compile_type_literal(&mut self, ty: &TypeExpr) {
        self.compile_stats.borrow_mut().type_literals_total += 1;

        if let Some(schema) = fold_type(ty) {
            let idx = self.push_const(wrap_type_schema_value(schema));
            self.code.push(Instruction::PushConst(idx));
            self.compile_stats.borrow_mut().type_literals_const_folded += 1;
            return;
        }

        self.compile_type_expr(ty);
        self.code.push(Instruction::WrapTypeLiteral);
        self.compile_stats.borrow_mut().type_literals_dynamic += 1;
    }

    fn compile_type_expr(&mut self, ty: &TypeExpr) {
        if let Some(value) = fold_type(ty) {
            let idx = self.push_const(value);
            self.code.push(Instruction::PushConst(idx));
            return;
        }

        match ty {
            TypeExpr::Ref(name) => {
                let slot = self.push_slot(name);
                self.code.push(Instruction::ResolveTypeRef(slot));
                self.compile_stats.borrow_mut().type_ref_sites += 1;
            }
            TypeExpr::List(inner) => {
                let kind_idx = self.push_const(Value::String("array".into()));
                self.code.push(Instruction::PushConst(kind_idx));
                self.compile_type_expr(inner);
                let keys = self.push_key_list(["type", "items"].into_iter());
                self.code.push(Instruction::BuildRecord(keys));
            }
            TypeExpr::Object(fields) => {
                let kind_idx = self.push_const(Value::String("object".into()));
                self.code.push(Instruction::PushConst(kind_idx));

                for field in fields {
                    self.compile_type_expr(&field.ty);
                }
                let prop_keys = self.push_key_list(fields.iter().map(|f| f.name.as_str()));
                self.code.push(Instruction::BuildRecord(prop_keys));

                let required: Vec<&str> = fields
                    .iter()
                    .filter(|f| !f.optional)
                    .map(|f| f.name.as_str())
                    .collect();
                for name in &required {
                    let idx = self.push_const(Value::String((*name).into()));
                    self.code.push(Instruction::PushConst(idx));
                }
                self.code.push(Instruction::BuildList(required.len()));

                self.code.push(Instruction::PushBool(false));

                let obj_keys = self.push_key_list(
                    ["type", "properties", "required", "additionalProperties"].into_iter(),
                );
                self.code.push(Instruction::BuildRecord(obj_keys));
            }
            TypeExpr::Union(variants) => {
                // Union reaches this arm only when at least one variant
                // contains a `Ref` that couldn't const-fold. Compile
                // each variant and pack them into an `anyOf` list.
                for variant in variants {
                    self.compile_type_expr(variant);
                }
                self.code.push(Instruction::BuildList(variants.len()));
                let keys = self.push_key_list(["anyOf"].into_iter());
                self.code.push(Instruction::BuildRecord(keys));
            }
            TypeExpr::Any
            | TypeExpr::Str
            | TypeExpr::Int
            | TypeExpr::Float
            | TypeExpr::Bool
            | TypeExpr::Dict
            | TypeExpr::Null
            | TypeExpr::Enum(_) => {
                unreachable!("scalar/enum types must const-fold")
            }
        }
    }

    fn patch_jump(&mut self, index: usize, target: usize) {
        match &mut self.code[index] {
            Instruction::Jump(slot)
            | Instruction::JumpIfFalse(slot)
            | Instruction::JumpIfCompareFalse { target: slot, .. }
            | Instruction::JumpIfSlotNumberCompareFalse { target: slot, .. }
            | Instruction::JumpIfSlotNumberBinaryCompareFalse { target: slot, .. }
            | Instruction::JumpIfTrue(slot)
            | Instruction::IterNext { jump_to: slot } => *slot = target,
            _ => unreachable!("patched non-jump instruction"),
        }
    }
}

pub(crate) fn is_pure_expr(expr: &Expr) -> bool {
    match expr {
        Expr::Null
        | Expr::Bool(_)
        | Expr::Number(_)
        | Expr::String(_)
        | Expr::Variable(_)
        | Expr::ResourceRef(_) => true,
        Expr::List(items) => items.iter().all(is_pure_expr),
        Expr::Record(entries) => entries.iter().all(|(_, value)| is_pure_expr(value)),
        Expr::ResultUnwrap(expr) => is_pure_expr(expr),
        Expr::BuiltinCall { args, .. } => args.iter().all(is_pure_expr),
        Expr::Field { target, .. } => is_pure_expr(target),
        Expr::Index { target, index } => is_pure_expr(target) && is_pure_expr(index),
        Expr::Unary { expr, .. } => is_pure_expr(expr),
        Expr::If {
            condition,
            then_block,
            else_block,
        } => is_pure_expr(condition) && is_pure_expr(then_block) && is_pure_expr(else_block),
        Expr::Binary { left, right, .. } => is_pure_expr(left) && is_pure_expr(right),
        Expr::TypeLiteral(ty) => fold_type(ty).is_some(),
        Expr::Block(_)
        | Expr::Assign { .. }
        | Expr::For { .. }
        | Expr::Break
        | Expr::Continue
        | Expr::ReceiverCall { .. }
        | Expr::StartProcess(_)
        | Expr::Await(_)
        | Expr::SleepFor(_)
        | Expr::SleepUntil(_)
        | Expr::WaitSignal
        | Expr::SignalRun { .. }
        | Expr::Cancel(_)
        | Expr::Print(_)
        | Expr::Submit(_)
        | Expr::Yield(_)
        | Expr::Wake(_)
        | Expr::Finish(_)
        | Expr::Fail(_) => false,
    }
}

fn contains_type_literal(expr: &Expr) -> bool {
    match expr {
        Expr::TypeLiteral(_) => true,
        Expr::Block(expressions) => expressions.iter().any(contains_type_literal),
        Expr::Assign { target, expr } => {
            assign_target_contains_type_literal(target) || contains_type_literal(expr)
        }
        Expr::List(items) => items.iter().any(contains_type_literal),
        Expr::Record(entries) => entries
            .iter()
            .any(|(_, value)| contains_type_literal(value)),
        Expr::StartProcess(process) => process
            .args
            .iter()
            .any(|(_, value)| contains_type_literal(value)),
        Expr::ReceiverCall { receiver, args, .. } => {
            contains_type_literal(receiver) || args.iter().any(contains_type_literal)
        }
        Expr::Await(expr)
        | Expr::SleepFor(expr)
        | Expr::SleepUntil(expr)
        | Expr::ResultUnwrap(expr)
        | Expr::Unary { expr, .. }
        | Expr::Cancel(expr)
        | Expr::Print(expr)
        | Expr::Yield(expr)
        | Expr::Wake(expr)
        | Expr::Fail(expr) => contains_type_literal(expr),
        Expr::SignalRun { run, payload } => {
            contains_type_literal(run) || contains_type_literal(payload)
        }
        Expr::BuiltinCall { args, .. } => args.iter().any(contains_type_literal),
        Expr::Field { target, .. } => contains_type_literal(target),
        Expr::Index { target, index } => {
            contains_type_literal(target) || contains_type_literal(index)
        }
        Expr::If {
            condition,
            then_block,
            else_block,
        } => {
            contains_type_literal(condition)
                || contains_type_literal(then_block)
                || contains_type_literal(else_block)
        }
        Expr::Binary { left, right, .. } => {
            contains_type_literal(left) || contains_type_literal(right)
        }
        Expr::For { iterable, body, .. } => {
            contains_type_literal(iterable) || contains_type_literal(body)
        }
        Expr::Submit(expr) | Expr::Finish(expr) => {
            expr.as_deref().is_some_and(contains_type_literal)
        }
        Expr::Break | Expr::Continue | Expr::WaitSignal => false,
        Expr::Null
        | Expr::Bool(_)
        | Expr::Number(_)
        | Expr::String(_)
        | Expr::Variable(_)
        | Expr::ResourceRef(_) => false,
    }
}

fn assign_target_contains_type_literal(target: &AssignTarget) -> bool {
    target.steps.iter().any(|step| match step {
        AssignPathStep::Field(_) => false,
        AssignPathStep::Index(expr) => contains_type_literal(expr),
    })
}

/// Best-effort compile-time construction of a JSON-Schema Value for a
/// [`TypeExpr`]. Returns `None` when the expression contains a [`TypeExpr::Ref`]
/// (or a nested composite that contains one) — those must be resolved at
/// runtime via [`Instruction::ResolveTypeRef`].
fn fold_type(ty: &TypeExpr) -> Option<Value> {
    match ty {
        TypeExpr::Any => Some(interned_scalar_schema(ScalarSchemaKind::Any)),
        TypeExpr::Str => Some(interned_scalar_schema(ScalarSchemaKind::Str)),
        TypeExpr::Int => Some(interned_scalar_schema(ScalarSchemaKind::Int)),
        TypeExpr::Float => Some(interned_scalar_schema(ScalarSchemaKind::Float)),
        TypeExpr::Bool => Some(interned_scalar_schema(ScalarSchemaKind::Bool)),
        TypeExpr::Dict => Some(interned_scalar_schema(ScalarSchemaKind::Dict)),
        TypeExpr::Null => Some(interned_scalar_schema(ScalarSchemaKind::Null)),
        TypeExpr::Enum(values) => {
            let mut rec = record_with_capacity(2);
            rec.insert("type".into(), Value::String("string".into()));
            let items: Vec<Value> = values.iter().map(|v| Value::String(v.clone())).collect();
            rec.insert("enum".into(), Value::List(items.into()));
            Some(Value::Record(Arc::new(rec)))
        }
        TypeExpr::List(inner) => {
            let inner_value = fold_type(inner)?;
            let mut rec = record_with_capacity(2);
            rec.insert("type".into(), Value::String("array".into()));
            rec.insert("items".into(), inner_value);
            Some(Value::Record(Arc::new(rec)))
        }
        TypeExpr::Object(fields) => {
            let mut properties = record_with_capacity(fields.len());
            for field in fields {
                properties.insert(field.name.to_string(), fold_type(&field.ty)?);
            }
            let required: Vec<Value> = fields
                .iter()
                .filter(|f| !f.optional)
                .map(|f| Value::String(f.name.clone()))
                .collect();
            let mut rec = record_with_capacity(4);
            rec.insert("type".into(), Value::String("object".into()));
            rec.insert("properties".into(), Value::Record(Arc::new(properties)));
            rec.insert("required".into(), Value::List(required.into()));
            rec.insert("additionalProperties".into(), Value::Bool(false));
            Some(Value::Record(Arc::new(rec)))
        }
        TypeExpr::Union(variants) => {
            let folded: Option<Vec<Value>> = variants.iter().map(fold_type).collect();
            let folded = folded?;
            let mut rec = record_with_capacity(1);
            rec.insert("anyOf".into(), Value::List(folded.into()));
            Some(Value::Record(Arc::new(rec)))
        }
        TypeExpr::Ref(_) => None,
    }
}

fn wrap_type_schema_value(schema: Value) -> Value {
    let mut wrapper = record_with_capacity(1);
    wrapper.insert(LASH_TYPE_KEY.to_string(), schema);
    Value::Record(Arc::new(wrapper))
}

fn is_terminal_expr(expr: &Expr) -> bool {
    match expr {
        Expr::Submit(_) | Expr::Finish(_) | Expr::Fail(_) => true,
        Expr::Block(expressions) => expressions.last().is_some_and(is_terminal_expr),
        Expr::If {
            then_block,
            else_block,
            ..
        } => is_terminal_expr(then_block) && is_terminal_expr(else_block),
        _ => false,
    }
}

#[derive(Clone, Copy)]
enum ScalarSchemaKind {
    Any,
    Str,
    Int,
    Float,
    Bool,
    Dict,
    Null,
}

/// Returns an `Arc`-shared schema for a scalar. All sites referencing `str`
/// point at the same `Arc<Record>`, so emitting a Type literal with N string
/// fields allocates one record, not N.
fn interned_scalar_schema(kind: ScalarSchemaKind) -> Value {
    static CACHE: OnceLock<[Value; 7]> = OnceLock::new();
    let cache = CACHE.get_or_init(|| {
        let build = |ty: &str| {
            let mut rec = record_with_capacity(1);
            rec.insert("type".into(), Value::String(ty.into()));
            Value::Record(Arc::new(rec))
        };
        [
            Value::Record(Arc::new(record_with_capacity(0))), // Any == {}
            build("string"),
            build("integer"),
            build("number"),
            build("boolean"),
            build("object"),
            build("null"),
        ]
    });
    cache[kind as usize].clone()
}
