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
    AssignPathStep, AssignTarget, BinaryOp, Expr, LabelMetadata, ProcessStartExpr, Program,
    TypeExpr, UnaryOp,
};
use crate::lexer::Span;
use crate::tracking::{LashlangAstPath, LashlangExecutionContext, LashlangExecutionSite};

use super::record::{Symbol, intern_symbol, lookup_symbol, record_with_capacity, symbol_name};
use super::schema::{ValidationPlan, compile_schema_value};
use super::{
    Chunk, CompileStats, CompiledAssignPath, CompiledAssignPathStep, CompiledFormatTemplate,
    Instruction, IntrinsicOp, LASH_MODULE_REF_KEY, LASH_PROCESS_NAME_KEY, LASH_PROCESS_REF_KEY,
    LASH_PROCESS_VALUE_KEY, LASH_REQUIRED_SURFACE_REF_KEY, LASH_TYPE_KEY, Name, Value, as_number,
    compile_format_template, eval_binary_values, execute_integer_div_builtin, execute_len_direct,
    execute_range_builtin, is_comparison_binary_op, is_numeric_binary_op, is_truthy,
    read_field_direct, read_index_direct, transient_name, unwrap_type_value,
};

pub(crate) struct Compiler {
    module_context: Option<CompiledModuleContext>,
    lashlang_execution: Option<LashlangExecutionCompileContext>,
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
    compile_stats: Rc<RefCell<CompileStats>>,
    const_slots: Vec<Option<Value>>,
    loop_contexts: Vec<LoopContext>,
}

struct LashlangExecutionCompileContext {
    context: LashlangExecutionContext,
    paths: FxHashMap<usize, LashlangAstPath>,
    sites: Vec<Option<LashlangExecutionSite>>,
}

struct LoopContext {
    continue_target: usize,
    break_jumps: SmallVec<[usize; 4]>,
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
        lashlang_execution_context: LashlangExecutionContext,
    ) -> (Chunk, CompileStats) {
        let stats = Rc::new(RefCell::new(CompileStats::default()));
        let mut compiler = Self::with_slots_and_stats(
            Some(module_context),
            Rc::new(RefCell::new(SlotTable::default())),
            stats.clone(),
        );
        compiler.lashlang_execution = Some(LashlangExecutionCompileContext {
            context: lashlang_execution_context,
            paths: lashlang_execution_paths(program),
            sites: Vec::new(),
        });
        compiler.compile_program_block(program);
        let chunk = compiler.finish();
        let compile_stats = *stats.borrow();
        (chunk, compile_stats)
    }

    pub(crate) fn compile_linked_process_program(
        program: &Program,
        module_context: CompiledModuleContext,
        lashlang_execution_context: LashlangExecutionContext,
    ) -> (Chunk, CompileStats) {
        let stats = Rc::new(RefCell::new(CompileStats::default()));
        let mut compiler = Self::with_slots_and_stats(
            Some(module_context),
            Rc::new(RefCell::new(SlotTable::default())),
            stats.clone(),
        );
        compiler.lashlang_execution = Some(LashlangExecutionCompileContext {
            context: lashlang_execution_context,
            paths: lashlang_execution_paths(program),
            sites: Vec::new(),
        });
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
            lashlang_execution: None,
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
            compile_stats,
            const_slots: Vec::new(),
            loop_contexts: Vec::new(),
        }
    }

    fn finish(self) -> Chunk {
        let slot_names = self.slots.borrow().names.clone();
        let mut spans = self.spans;
        spans.resize(self.code.len(), None);
        let mut lashlang_execution_sites = self
            .lashlang_execution
            .map(|tracking| tracking.sites)
            .unwrap_or_default();
        lashlang_execution_sites.resize(self.code.len(), None);
        Chunk {
            module_context: self.module_context,
            code: self.code,
            spans,
            lashlang_execution_sites,
            constants: self.constants,
            names: self.names,
            slot_names,
            key_lists: self.key_lists,
            format_templates: self.format_templates,
            compiled_schemas: self.compiled_schemas,
            assign_paths: self.assign_paths,
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
        // Unknown names are not arity-checked here; they fall through to
        // `IntrinsicOp::Unknown`. Known builtins must satisfy their registered
        // arity or compile to `InvalidArity`.
        if let Some(builtin) = crate::builtins::lookup(name)
            && !builtin.arity.accepts(argc)
        {
            return IntrinsicOp::InvalidArity {
                name: self.push_name(name),
                argc,
            };
        }
        intrinsic_for_builtin(name, argc).unwrap_or_else(|| IntrinsicOp::Unknown {
            name: self.push_name(name),
            argc,
        })
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
            Expr::LabelAnnotated { label, expr } => {
                if !label_attaches_to_concrete_node(expr) {
                    self.emit_lashlang_execution_step(expression, label);
                }
                self.compile_expr_discarding_value(expr);
            }
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
            Expr::While { condition, body } => self.compile_while_expr(condition, body, false),
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

    fn mark_lashlang_execution_site(&mut self, instruction: usize, site: LashlangExecutionSite) {
        let Some(tracking) = self.lashlang_execution.as_mut() else {
            return;
        };
        if tracking.sites.len() <= instruction {
            tracking.sites.resize(instruction + 1, None);
        }
        tracking.sites[instruction] = Some(site);
    }

    fn lashlang_execution_site(
        &self,
        expression: &Expr,
        kind: &str,
        label: impl Into<String>,
    ) -> Option<LashlangExecutionSite> {
        let tracking = self.lashlang_execution.as_ref()?;
        let path = tracking.paths.get(&expr_key(expression))?;
        Some(tracking.context.builder().node_site(path, kind, label))
    }

    fn emit_lashlang_execution_step(&mut self, expression: &Expr, label: &LabelMetadata) {
        let instruction = self.code.len();
        self.code.push(Instruction::ObserveStep);
        if let Some(site) = self.lashlang_execution_site(expression, "step", label.title.as_str()) {
            self.mark_lashlang_execution_site(instruction, site);
        }
    }

    fn branch_execution_site(&self, expression: &Expr) -> Option<LashlangExecutionSite> {
        let tracking = self.lashlang_execution.as_ref()?;
        let path = tracking.paths.get(&expr_key(expression))?;
        Some(tracking.context.builder().branch_site(path))
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
                self.fold_type_expr(ty).map(wrap_type_schema_value)
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

    fn compile_while_expr(&mut self, condition: &Expr, body: &Expr, leave_value: bool) {
        self.clear_const_slots();
        let loop_start = self.code.len();
        let jump_to_end = self.compile_condition_jump_if_false(condition);
        self.clear_const_slots();
        self.loop_contexts.push(LoopContext {
            continue_target: loop_start,
            break_jumps: SmallVec::new(),
        });
        self.compile_block_discarding_values(body);
        let loop_context = self
            .loop_contexts
            .pop()
            .expect("loop context should exist while compiling `while`");
        self.code.push(Instruction::Jump(loop_start));
        let loop_end = self.code.len();
        self.patch_jump(jump_to_end, loop_end);
        for break_jump in loop_context.break_jumps {
            self.patch_jump(break_jump, loop_end);
        }
        self.clear_const_slots();
        self.push_null_if(leave_value);
    }

    fn fold_compile_time_expr(&self, expr: &Expr) -> Option<Value> {
        match expr {
            Expr::LabelAnnotated { expr, .. } => self.fold_compile_time_expr(expr),
            Expr::Null => Some(Value::Null),
            Expr::Bool(value) => Some(Value::Bool(*value)),
            Expr::Number(value) => Some(Value::Number(*value)),
            Expr::String(value) => Some(Value::String(value.clone())),
            Expr::ResourceRef(resource) => Some(Value::Resource(super::ResourceHandle::new(
                resource.resource_type.to_string(),
                resource.alias.to_string(),
            ))),
            Expr::ProcessRef { .. } | Expr::HostValueConstructor { .. } => None,
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
                let builtin = intrinsic_for_builtin(name.as_str(), args.len())?;
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
            Expr::TypeLiteral(ty) => self.fold_type_expr(ty).map(wrap_type_schema_value),
            Expr::Block(_)
            | Expr::Assign { .. }
            | Expr::For { .. }
            | Expr::While { .. }
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
            Expr::LabelAnnotated { label, expr } => {
                if !label_attaches_to_concrete_node(expr) {
                    self.emit_lashlang_execution_step(expr, label);
                }
                self.compile_expr(expr);
            }
            Expr::Block(expressions) => self.compile_block_value(expressions),
            Expr::Assign { target, expr } => self.compile_assignment_expr(target, expr, true),
            Expr::For {
                binding,
                iterable,
                body,
            } => self.compile_for_expr(binding, iterable, body, true),
            Expr::While { condition, body } => self.compile_while_expr(condition, body, true),
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
            Expr::StartProcess(process) => {
                let instruction = self.compile_start_process_expr(process);
                if let Some(site) = self.lashlang_execution_site(
                    expr,
                    "child_process",
                    format!("start {}", process.process),
                ) {
                    self.mark_lashlang_execution_site(instruction, site);
                }
            }
            Expr::ProcessRef { process } => self.compile_process_ref_expr(process),
            Expr::HostValueConstructor { type_name, input } => {
                self.compile_expr(input);
                let type_name = self.push_name(type_name);
                self.code.push(Instruction::WrapHostValue(type_name));
            }
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
            } => {
                let instruction = self.compile_receiver_call_expr(receiver, operation, args, false);
                if let Some(site) =
                    self.lashlang_execution_site(expr, "resource_operation", operation.as_str())
                {
                    self.mark_lashlang_execution_site(instruction, site);
                }
            }
            Expr::Await(handle) => match handle.as_ref() {
                Expr::ReceiverCall {
                    receiver,
                    operation,
                    args,
                } => {
                    let instruction =
                        self.compile_receiver_call_expr(receiver, operation, args, false);
                    if let Some(site) = self.lashlang_execution_site(
                        handle,
                        "resource_operation",
                        operation.as_str(),
                    ) {
                        self.mark_lashlang_execution_site(instruction, site);
                    }
                }
                Expr::ResultUnwrap(inner) => {
                    if let Expr::ReceiverCall {
                        receiver,
                        operation,
                        args,
                    } = inner.as_ref()
                    {
                        let instruction =
                            self.compile_receiver_call_expr(receiver, operation, args, true);
                        if let Some(site) = self.lashlang_execution_site(
                            inner,
                            "resource_operation",
                            operation.as_str(),
                        ) {
                            self.mark_lashlang_execution_site(instruction, site);
                        }
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
                let instruction = self.code.len();
                self.code.push(Instruction::SleepFor);
                if let Some(site) = self.lashlang_execution_site(expr, "sleep", "sleep for") {
                    self.mark_lashlang_execution_site(instruction, site);
                }
            }
            Expr::SleepUntil(deadline) => {
                self.compile_expr(deadline);
                let instruction = self.code.len();
                self.code.push(Instruction::SleepUntil);
                if let Some(site) = self.lashlang_execution_site(expr, "sleep", "sleep until") {
                    self.mark_lashlang_execution_site(instruction, site);
                }
            }
            Expr::WaitSignal => {
                let instruction = self.code.len();
                self.code.push(Instruction::ProcessWaitSignal);
                if let Some(site) = self.lashlang_execution_site(expr, "wait", "wait signal") {
                    self.mark_lashlang_execution_site(instruction, site);
                }
            }
            Expr::SignalRun { run, payload } => {
                self.compile_expr(run);
                self.compile_expr(payload);
                let instruction = self.code.len();
                self.code.push(Instruction::ProcessSignalRun);
                if let Some(site) = self.lashlang_execution_site(expr, "signal", "signal run") {
                    self.mark_lashlang_execution_site(instruction, site);
                }
            }
            Expr::ResultUnwrap(expr) => {
                if let Expr::ReceiverCall {
                    receiver,
                    operation,
                    args,
                } = expr.as_ref()
                {
                    let instruction =
                        self.compile_receiver_call_expr(receiver, operation, args, true);
                    if let Some(site) =
                        self.lashlang_execution_site(expr, "resource_operation", operation.as_str())
                    {
                        self.mark_lashlang_execution_site(instruction, site);
                    }
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
                if let Some(site) = self.branch_execution_site(expr) {
                    self.mark_lashlang_execution_site(jump_to_else, site);
                }
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
            Expr::Yield(value) => {
                self.compile_expr(value);
                let instruction = self.code.len();
                self.code.push(Instruction::ProcessYield);
                if let Some(site) = self.lashlang_execution_site(expr, "process_event", "yield") {
                    self.mark_lashlang_execution_site(instruction, site);
                }
            }
            Expr::Wake(value) => {
                self.compile_expr(value);
                let instruction = self.code.len();
                self.code.push(Instruction::ProcessWake);
                if let Some(site) = self.lashlang_execution_site(expr, "process_event", "wake") {
                    self.mark_lashlang_execution_site(instruction, site);
                }
            }
            Expr::Finish(value) => {
                if let Some(value) = value {
                    self.compile_expr(value);
                } else {
                    self.compile_expr(&Expr::Null);
                }
                let instruction = self.code.len();
                self.code.push(Instruction::ProcessFinish);
                if let Some(site) = self.lashlang_execution_site(expr, "terminal", "result") {
                    self.mark_lashlang_execution_site(instruction, site);
                }
            }
            Expr::Fail(value) => {
                self.compile_expr(value);
                let instruction = self.code.len();
                self.code.push(Instruction::ProcessFail);
                if let Some(site) = self.lashlang_execution_site(expr, "terminal", "failure") {
                    self.mark_lashlang_execution_site(instruction, site);
                }
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

    fn compile_start_process_expr(&mut self, process: &ProcessStartExpr) -> usize {
        for (_, expr) in &process.args {
            self.compile_expr(expr);
        }
        let keys = self.push_key_list(process.args.iter().map(|(name, _)| name.as_str()));
        let process = self.push_name(&process.process);
        let instruction = self.code.len();
        self.code.push(Instruction::StartProcess { process, keys });
        instruction
    }

    fn compile_process_ref_expr(&mut self, process: &str) {
        let Some(module_context) = self.module_context.as_ref() else {
            self.emit_push_value(Value::Null);
            return;
        };
        let Some(process_ref) = module_context.process_refs.get(process) else {
            self.emit_push_value(Value::Null);
            return;
        };
        let mut record = record_with_capacity(5);
        record.insert(LASH_PROCESS_VALUE_KEY.to_string(), Value::Bool(true));
        record.insert(
            LASH_PROCESS_NAME_KEY.to_string(),
            Value::String(process.into()),
        );
        record.insert(
            LASH_MODULE_REF_KEY.to_string(),
            Value::String(module_context.module_ref.to_string().into()),
        );
        let mut process_ref_record = record_with_capacity(2);
        process_ref_record.insert(
            "component".to_string(),
            Value::String(process_ref.component.to_string().into()),
        );
        process_ref_record.insert("pos".to_string(), Value::Number(process_ref.pos as f64));
        record.insert(
            LASH_PROCESS_REF_KEY.to_string(),
            Value::Record(Arc::new(process_ref_record)),
        );
        record.insert(
            LASH_REQUIRED_SURFACE_REF_KEY.to_string(),
            Value::String(module_context.required_surface_ref.to_string().into()),
        );
        self.emit_push_value(Value::Record(Arc::new(record)));
    }

    fn compile_receiver_call_expr(
        &mut self,
        receiver: &Expr,
        operation: &str,
        args: &[Expr],
        unwrap: bool,
    ) -> usize {
        self.compile_expr(receiver);
        for arg in args {
            self.compile_expr(arg);
        }
        let operation = self.push_name(operation);
        let instruction = self.code.len();
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
        instruction
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

        if let Some(schema) = self.fold_type_expr(ty) {
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
        if let Some(value) = self.fold_type_expr(ty) {
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
                let kind_idx = self.push_const(Value::String(schema_keys::ARRAY.into()));
                self.code.push(Instruction::PushConst(kind_idx));
                self.compile_type_expr(inner);
                let keys = self.push_key_list([schema_keys::TYPE, schema_keys::ITEMS].into_iter());
                self.code.push(Instruction::BuildRecord(keys));
            }
            TypeExpr::Object(fields) => {
                let kind_idx = self.push_const(Value::String(schema_keys::OBJECT.into()));
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
                    [
                        schema_keys::TYPE,
                        schema_keys::PROPERTIES,
                        schema_keys::REQUIRED,
                        schema_keys::ADDITIONAL_PROPERTIES,
                    ]
                    .into_iter(),
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
                let keys = self.push_key_list([schema_keys::ANY_OF].into_iter());
                self.code.push(Instruction::BuildRecord(keys));
            }
            TypeExpr::Process { .. } | TypeExpr::TriggerHandle(_) => {
                let idx = self.push_const(interned_scalar_schema(ScalarSchemaKind::Any));
                self.code.push(Instruction::PushConst(idx));
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

    fn fold_type_expr(&self, ty: &TypeExpr) -> Option<Value> {
        self.fold_type_expr_inner(ty, &mut SmallVec::new())
    }

    fn fold_type_expr_inner<'a>(
        &self,
        ty: &'a TypeExpr,
        resolving: &mut SmallVec<[&'a str; 4]>,
    ) -> Option<Value> {
        use schema_keys::*;
        match ty {
            TypeExpr::Ref(name) => {
                let name = name.as_str();
                if resolving.contains(&name) {
                    return None;
                }
                let wrapper = self.const_for_name(name)?;
                resolving.push(name);
                let schema = unwrap_type_value(&wrapper).cloned();
                resolving.pop();
                schema
            }
            TypeExpr::List(inner) => {
                let inner_value = self.fold_type_expr_inner(inner, resolving)?;
                let mut rec = record_with_capacity(2);
                rec.insert(TYPE.into(), Value::String(ARRAY.into()));
                rec.insert(ITEMS.into(), inner_value);
                Some(Value::Record(Arc::new(rec)))
            }
            TypeExpr::Object(fields) => {
                let mut properties = record_with_capacity(fields.len());
                for field in fields {
                    properties.insert(
                        field.name.to_string(),
                        self.fold_type_expr_inner(&field.ty, resolving)?,
                    );
                }
                let required: Vec<Value> = fields
                    .iter()
                    .filter(|f| !f.optional)
                    .map(|f| Value::String(f.name.clone()))
                    .collect();
                let mut rec = record_with_capacity(4);
                rec.insert(TYPE.into(), Value::String(OBJECT.into()));
                rec.insert(PROPERTIES.into(), Value::Record(Arc::new(properties)));
                rec.insert(REQUIRED.into(), Value::List(required.into()));
                rec.insert(ADDITIONAL_PROPERTIES.into(), Value::Bool(false));
                Some(Value::Record(Arc::new(rec)))
            }
            TypeExpr::Union(variants) => {
                let folded: Option<Vec<Value>> = variants
                    .iter()
                    .map(|variant| self.fold_type_expr_inner(variant, resolving))
                    .collect();
                let folded = folded?;
                let mut rec = record_with_capacity(1);
                rec.insert(ANY_OF.into(), Value::List(folded.into()));
                Some(Value::Record(Arc::new(rec)))
            }
            _ => fold_type(ty),
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

/// Maps a builtin name to the [`IntrinsicOp`] the VM dispatches on, threading
/// `argc` into the arity-carrying ops. Returns `None` for names that are not
/// builtins (the caller decides whether that is an `Unknown` op or a const-fold
/// miss). This is the single name -> op authority shared by `resolve_intrinsic`
/// and the const folder.
fn intrinsic_for_builtin(name: &str, argc: usize) -> Option<IntrinsicOp> {
    Some(match name {
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
        _ => return None,
    })
}

fn expr_key(expr: &Expr) -> usize {
    expr as *const Expr as usize
}

fn lashlang_execution_paths(program: &Program) -> FxHashMap<usize, LashlangAstPath> {
    let mut paths = FxHashMap::default();
    collect_lashlang_execution_paths(&program.main, LashlangAstPath::root(), &mut paths);
    paths
}

fn collect_lashlang_execution_paths(
    expr: &Expr,
    path: LashlangAstPath,
    paths: &mut FxHashMap<usize, LashlangAstPath>,
) {
    paths.insert(expr_key(expr), path.clone());
    if let Expr::LabelAnnotated { expr, .. } = expr {
        collect_lashlang_execution_paths(expr, path, paths);
        return;
    }
    for (index, child) in expr.children().enumerate() {
        collect_lashlang_execution_paths(child, path.child(index), paths);
    }
}

fn label_attaches_to_concrete_node(expr: &Expr) -> bool {
    match expr {
        Expr::LabelAnnotated { .. } => false,
        Expr::Assign { expr, .. } => label_attaches_to_assignment_value(expr),
        Expr::Await(expr) | Expr::ResultUnwrap(expr) => label_attaches_to_concrete_node(expr),
        Expr::ReceiverCall { .. }
        | Expr::StartProcess(_)
        | Expr::SleepFor(_)
        | Expr::SleepUntil(_)
        | Expr::WaitSignal
        | Expr::SignalRun { .. }
        | Expr::Submit(_)
        | Expr::Yield(_)
        | Expr::Wake(_)
        | Expr::Finish(_)
        | Expr::Fail(_)
        | Expr::If { .. } => true,
        Expr::Block(_)
        | Expr::Null
        | Expr::Bool(_)
        | Expr::Number(_)
        | Expr::String(_)
        | Expr::Variable(_)
        | Expr::List(_)
        | Expr::Record(_)
        | Expr::For { .. }
        | Expr::While { .. }
        | Expr::Break
        | Expr::Continue
        | Expr::ProcessRef { .. }
        | Expr::HostValueConstructor { .. }
        | Expr::ResourceRef(_)
        | Expr::Cancel(_)
        | Expr::Print(_)
        | Expr::BuiltinCall { .. }
        | Expr::Field { .. }
        | Expr::Index { .. }
        | Expr::Unary { .. }
        | Expr::Binary { .. }
        | Expr::TypeLiteral(_) => false,
    }
}

fn label_attaches_to_assignment_value(expr: &Expr) -> bool {
    match expr {
        Expr::Await(expr) | Expr::ResultUnwrap(expr) => label_attaches_to_assignment_value(expr),
        Expr::ReceiverCall { .. }
        | Expr::StartProcess(_)
        | Expr::SleepFor(_)
        | Expr::SleepUntil(_)
        | Expr::WaitSignal
        | Expr::SignalRun { .. }
        | Expr::Submit(_)
        | Expr::Yield(_)
        | Expr::Wake(_)
        | Expr::Finish(_)
        | Expr::Fail(_)
        | Expr::If { .. } => true,
        _ => false,
    }
}

pub(crate) fn is_pure_expr(expr: &Expr) -> bool {
    match expr {
        Expr::LabelAnnotated { expr, .. } => is_pure_expr(expr),
        Expr::Null
        | Expr::Bool(_)
        | Expr::Number(_)
        | Expr::String(_)
        | Expr::Variable(_)
        | Expr::ProcessRef { .. }
        | Expr::ResourceRef(_) => true,
        Expr::List(items) => items.iter().all(is_pure_expr),
        Expr::Record(entries) => entries.iter().all(|(_, value)| is_pure_expr(value)),
        Expr::ResultUnwrap(expr) => is_pure_expr(expr),
        Expr::HostValueConstructor { input, .. } => is_pure_expr(input),
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
        | Expr::While { .. }
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
    // `TypeLiteral` is the only node that introduces a type literal directly;
    // every other node contains one only via a child expression. `children()`
    // already yields an `Assign` target's dynamic index steps, so the generic
    // structural recursion covers the path-assignment case too.
    matches!(expr, Expr::TypeLiteral(_)) || expr.children().any(contains_type_literal)
}

/// The JSON-Schema vocabulary the language emits. Both the compile-time
/// builder ([`fold_type`]) and the runtime instruction builder
/// ([`Compiler::compile_type_expr`]) reference these names so the schema shape
/// is defined exactly once.
mod schema_keys {
    pub(super) const TYPE: &str = "type";
    pub(super) const ITEMS: &str = "items";
    pub(super) const PROPERTIES: &str = "properties";
    pub(super) const REQUIRED: &str = "required";
    pub(super) const ADDITIONAL_PROPERTIES: &str = "additionalProperties";
    pub(super) const ANY_OF: &str = "anyOf";
    pub(super) const ENUM: &str = "enum";

    pub(super) const ARRAY: &str = "array";
    pub(super) const OBJECT: &str = "object";
    pub(super) const STRING: &str = "string";
}

/// Best-effort compile-time construction of a JSON-Schema Value for a
/// [`TypeExpr`]. This is the single authority for the language's type -> schema
/// shape; the runtime instruction builder mirrors only the dynamic `Ref` paths
/// and shares the same key vocabulary ([`schema_keys`]).
///
/// Returns `None` when the expression contains a [`TypeExpr::Ref`] (or a nested
/// composite that contains one) — those must be resolved at runtime via
/// [`Instruction::ResolveTypeRef`].
fn fold_type(ty: &TypeExpr) -> Option<Value> {
    use schema_keys::*;
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
            rec.insert(TYPE.into(), Value::String(STRING.into()));
            let items: Vec<Value> = values.iter().map(|v| Value::String(v.clone())).collect();
            rec.insert(ENUM.into(), Value::List(items.into()));
            Some(Value::Record(Arc::new(rec)))
        }
        TypeExpr::List(inner) => {
            let inner_value = fold_type(inner)?;
            let mut rec = record_with_capacity(2);
            rec.insert(TYPE.into(), Value::String(ARRAY.into()));
            rec.insert(ITEMS.into(), inner_value);
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
            rec.insert(TYPE.into(), Value::String(OBJECT.into()));
            rec.insert(PROPERTIES.into(), Value::Record(Arc::new(properties)));
            rec.insert(REQUIRED.into(), Value::List(required.into()));
            rec.insert(ADDITIONAL_PROPERTIES.into(), Value::Bool(false));
            Some(Value::Record(Arc::new(rec)))
        }
        TypeExpr::Union(variants) => {
            let folded: Option<Vec<Value>> = variants.iter().map(fold_type).collect();
            let folded = folded?;
            let mut rec = record_with_capacity(1);
            rec.insert(ANY_OF.into(), Value::List(folded.into()));
            Some(Value::Record(Arc::new(rec)))
        }
        TypeExpr::Process { .. } | TypeExpr::TriggerHandle(_) => {
            Some(interned_scalar_schema(ScalarSchemaKind::Any))
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
        Expr::LabelAnnotated { expr, .. } => is_terminal_expr(expr),
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
            rec.insert(schema_keys::TYPE.into(), Value::String(ty.into()));
            Value::Record(Arc::new(rec))
        };
        [
            Value::Record(Arc::new(record_with_capacity(0))), // Any == {}
            build(schema_keys::STRING),
            build("integer"),
            build("number"),
            build("boolean"),
            build(schema_keys::OBJECT),
            build("null"),
        ]
    });
    cache[kind as usize].clone()
}
