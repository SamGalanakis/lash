//! Bytecode compiler: lowers `crate::ast::Program` into a `Chunk` of
//! instructions plus the supporting compile-time tables (slot maps,
//! parallel-call branches, format templates, schema cache).
//!
//! All compile-time-only helpers live here too: `is_pure_expr` /
//! `contains_type_literal` (used by the parallel-call optimization to
//! decide whether an expression can be evaluated without entering the VM)
//! and the `fold_type` / `interned_scalar_schema` machinery (used to
//! convert `TypeExpr` AST nodes into JSON-Schema-shaped `Value` literals
//! at compile time).

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc, OnceLock};

use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::ast::{
    AssignPathStep, AssignTarget, BinaryOp, CallExpr, Expr, ParallelBranches, Program, Stmt,
    TypeExpr, UnaryOp,
};
use crate::lexer::Span;

use super::record::{Symbol, intern_symbol, lookup_symbol, record_with_capacity, symbol_name};
use super::schema::{CompiledSchema, compile_schema_value};
use super::{
    Builtin, Chunk, CompileStats, CompiledAssignPath, CompiledAssignPathStep,
    CompiledFormatTemplate, Instruction, LASH_TYPE_KEY, Name, NamedBranchChunk,
    NamedParallelCallBranch, ParallelCallBranch, PureExpr, RuntimeError, Value, as_number,
    compile_format_template, eval_binary_values, execute_range_builtin, is_comparison_binary_op,
    is_numeric_binary_op, is_truthy, read_field_direct, read_index_direct, transient_name,
    unwrap_type_value,
};

pub(crate) struct Compiler {
    code: Vec<Instruction>,
    spans: Vec<Option<Span>>,
    constants: Vec<Value>,
    names: Vec<Name>,
    name_lookup: FxHashMap<Symbol, usize>,
    slots: Rc<RefCell<SlotTable>>,
    key_lists: Vec<Box<[usize]>>,
    format_templates: Vec<CompiledFormatTemplate>,
    compiled_schemas: Vec<CompiledSchema>,
    parallel_call_sets: Vec<Box<[ParallelCallBranch]>>,
    named_parallel_call_sets: Vec<Box<[NamedParallelCallBranch]>>,
    pure_parallel_sets: Vec<Box<[PureExpr]>>,
    pure_named_parallel_sets: Vec<Box<[(usize, PureExpr)]>>,
    branch_sets: Vec<Box<[Chunk]>>,
    named_branch_sets: Vec<Box<[NamedBranchChunk]>>,
    assign_paths: Vec<CompiledAssignPath>,
    compile_stats: Rc<RefCell<CompileStats>>,
    const_slots: Vec<Option<Value>>,
    loop_contexts: Vec<LoopContext>,
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
        let mut compiler =
            Self::with_slots_and_stats(Rc::new(RefCell::new(SlotTable::default())), stats.clone());
        compiler.compile_program_block(program);
        let chunk = compiler.finish();
        let compile_stats = *stats.borrow();
        (chunk, compile_stats)
    }

    fn with_slots_and_stats(
        slots: Rc<RefCell<SlotTable>>,
        compile_stats: Rc<RefCell<CompileStats>>,
    ) -> Self {
        Self {
            code: Vec::new(),
            spans: Vec::new(),
            constants: Vec::new(),
            names: Vec::new(),
            name_lookup: FxHashMap::default(),
            slots,
            key_lists: Vec::new(),
            format_templates: Vec::new(),
            compiled_schemas: Vec::new(),
            parallel_call_sets: Vec::new(),
            named_parallel_call_sets: Vec::new(),
            pure_parallel_sets: Vec::new(),
            pure_named_parallel_sets: Vec::new(),
            branch_sets: Vec::new(),
            named_branch_sets: Vec::new(),
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
        Chunk {
            code: self.code,
            spans,
            constants: self.constants,
            names: self.names,
            slot_names,
            key_lists: self.key_lists,
            format_templates: self.format_templates,
            compiled_schemas: self.compiled_schemas,
            parallel_call_sets: self.parallel_call_sets,
            named_parallel_call_sets: self.named_parallel_call_sets,
            pure_parallel_sets: self.pure_parallel_sets,
            pure_named_parallel_sets: self.pure_named_parallel_sets,
            branch_sets: self.branch_sets,
            named_branch_sets: self.named_branch_sets,
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

    fn push_branch_set(&mut self, branches: Vec<Chunk>) -> usize {
        let index = self.branch_sets.len();
        self.branch_sets.push(branches.into_boxed_slice());
        index
    }

    fn push_named_branch_set(&mut self, branches: Vec<NamedBranchChunk>) -> usize {
        let index = self.named_branch_sets.len();
        self.named_branch_sets.push(branches.into_boxed_slice());
        index
    }

    fn push_parallel_call_set(&mut self, branches: Vec<ParallelCallBranch>) -> usize {
        let index = self.parallel_call_sets.len();
        self.parallel_call_sets.push(branches.into_boxed_slice());
        index
    }

    fn push_named_parallel_call_set(&mut self, branches: Vec<NamedParallelCallBranch>) -> usize {
        let index = self.named_parallel_call_sets.len();
        self.named_parallel_call_sets
            .push(branches.into_boxed_slice());
        index
    }

    fn push_pure_parallel_set(&mut self, branches: Vec<PureExpr>) -> usize {
        let index = self.pure_parallel_sets.len();
        self.pure_parallel_sets.push(branches.into_boxed_slice());
        index
    }

    fn push_pure_named_parallel_set(&mut self, branches: Vec<(usize, PureExpr)>) -> usize {
        let index = self.pure_named_parallel_sets.len();
        self.pure_named_parallel_sets
            .push(branches.into_boxed_slice());
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

    fn resolve_builtin(&mut self, name: &str) -> Builtin {
        match name {
            "len" => Builtin::Len,
            "empty" => Builtin::Empty,
            "keys" => Builtin::Keys,
            "values" => Builtin::Values,
            "contains" => Builtin::Contains,
            "find" => Builtin::Find,
            "grep_text" => Builtin::GrepText,
            "starts_with" => Builtin::StartsWith,
            "ends_with" => Builtin::EndsWith,
            "split" => Builtin::Split,
            "join" => Builtin::Join,
            "trim" => Builtin::Trim,
            "slice" => Builtin::Slice,
            "to_string" => Builtin::ToString,
            "to_int" => Builtin::ToInt,
            "to_float" => Builtin::ToFloat,
            "json_parse" => Builtin::JsonParse,
            "format" => Builtin::Format,
            "validate" => Builtin::Validate,
            "range" => Builtin::Range,
            "push" => Builtin::Push,
            _ => Builtin::Unknown(self.push_name(name)),
        }
    }

    fn compile_program_block(&mut self, program: &Program) {
        for (index, statement) in program.statements.iter().enumerate() {
            let span = program.statement_spans.get(index).copied();
            self.compile_stmt_with_span(statement, span);
        }
    }

    fn compile_block(&mut self, statements: &[Stmt]) {
        for statement in statements {
            self.compile_stmt_with_span(statement, None);
        }
    }

    fn compile_stmt_with_span(&mut self, statement: &Stmt, span: Option<Span>) {
        let start = self.code.len();
        self.compile_stmt(statement);
        if self.spans.len() < self.code.len() {
            self.spans.resize(self.code.len(), None);
        }
        for entry in &mut self.spans[start..self.code.len()] {
            *entry = span;
        }
    }

    fn compile_stmt(&mut self, statement: &Stmt) {
        match statement {
            Stmt::Assign { target, expr } if target.is_simple() => {
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
                        return;
                    }
                    if let Some(Value::Number(right)) = self.fold_compile_time_expr(right) {
                        self.code.push(Instruction::AddAssignNumber { slot, right });
                        self.set_const_slot(slot, None);
                        return;
                    }
                    if let Expr::Variable(right_name) = right.as_ref() {
                        let right = self.push_slot(right_name);
                        self.code.push(Instruction::AddAssignSlot { slot, right });
                        self.set_const_slot(slot, None);
                        return;
                    }
                    self.compile_expr(right);
                    self.code.push(Instruction::AddAssign(slot));
                    self.set_const_slot(slot, None);
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
                    self.code.push(Instruction::PushAssign(slot));
                    self.set_const_slot(slot, None);
                    return;
                }
                if let Some(value) = const_value.clone()
                    && !has_type_literal
                {
                    let constant = self.push_const(value);
                    self.code.push(Instruction::StoreConst { slot, constant });
                    self.set_const_slot(slot, const_value);
                    return;
                }

                self.compile_expr(expr);
                self.code.push(Instruction::StoreName(slot));
                self.set_const_slot(slot, const_value);
            }
            Stmt::Assign { target, expr } => {
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
                    self.compile_expr(index);
                    self.code
                        .push(Instruction::AddAssignIndexNumber { slot, right });
                    self.set_const_slot(slot, None);
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
            }
            Stmt::Expr(expr) => {
                self.compile_expr(expr);
                self.code.push(Instruction::Pop);
            }
            Stmt::Call(call) => {
                self.compile_call_expr(call);
                self.code.push(Instruction::Pop);
            }
            Stmt::Cancel(handle) => {
                self.compile_expr(handle);
                self.code.push(Instruction::CancelHandle);
            }
            Stmt::Print(expr) => {
                self.compile_expr(expr);
                self.code.push(Instruction::Print);
            }
            Stmt::If {
                condition,
                then_block,
                else_block,
            } => {
                let jump_to_else = self.compile_condition_jump_if_false(condition);
                self.compile_block(then_block);
                if else_block.is_empty() {
                    self.patch_jump(jump_to_else, self.code.len());
                } else {
                    let jump_to_end = self.emit_jump();
                    self.patch_jump(jump_to_else, self.code.len());
                    self.compile_block(else_block);
                    self.patch_jump(jump_to_end, self.code.len());
                }
                self.clear_const_slots();
            }
            Stmt::For {
                binding,
                iterable,
                body,
            } => {
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
                    return;
                }

                self.compile_expr(iterable);
                self.clear_const_slots();
                self.set_const_slot(binding, None);
                self.code.push(Instruction::BeginIter(binding));
                self.compile_for_loop_body(body);
            }
            Stmt::Break => {
                let jump = self.emit_jump();
                self.loop_contexts
                    .last_mut()
                    .expect("parser rejects `break` outside loops")
                    .break_jumps
                    .push(jump);
                self.clear_const_slots();
            }
            Stmt::Continue => {
                let continue_target = self
                    .loop_contexts
                    .last()
                    .expect("parser rejects `continue` outside loops")
                    .continue_target;
                self.code.push(Instruction::Jump(continue_target));
                self.clear_const_slots();
            }
            Stmt::Parallel { branches } => {
                self.compile_parallel(branches, false);
                self.clear_const_slots();
            }
            Stmt::Submit(expr) => {
                if let Some(expr) = expr {
                    self.compile_expr(expr);
                } else {
                    self.compile_expr(&Expr::Null);
                }
                self.code.push(Instruction::Submit);
            }
        }
    }

    fn compile_for_loop_body(&mut self, body: &[Stmt]) {
        let loop_start = self.code.len();
        let iter_next = self.code.len();
        self.code.push(Instruction::IterNext {
            jump_to: usize::MAX,
        });
        self.loop_contexts.push(LoopContext {
            continue_target: loop_start,
            break_jumps: SmallVec::new(),
        });
        self.compile_block(body);
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

    fn compile_parallel_calls(&mut self, branches: &[Stmt]) -> Option<Vec<ParallelCallBranch>> {
        let mut compiled = Vec::with_capacity(branches.len());
        for branch in branches {
            let Stmt::Assign { target, expr } = branch else {
                return None;
            };
            if !target.is_simple() {
                return None;
            }
            let Expr::ToolCall(call) = expr else {
                return None;
            };
            if call.args.iter().any(|(_, expr)| !is_pure_expr(expr)) {
                return None;
            }

            let slot = self.push_slot(&target.root);
            let args = PureExpr::Record(
                call.args
                    .iter()
                    .map(|(key, expr)| Ok((self.push_name(key), self.compile_pure_expr(expr)?)))
                    .collect::<Result<Vec<_>, RuntimeError>>()
                    .ok()?
                    .into_boxed_slice(),
            );
            let name = self.push_name(&call.name);
            compiled.push(ParallelCallBranch { slot, name, args });
        }
        Some(compiled)
    }

    fn compile_pure_parallel_exprs(&mut self, branches: &[Stmt]) -> Option<Vec<PureExpr>> {
        let mut compiled = Vec::with_capacity(branches.len());
        for branch in branches {
            let Stmt::Expr(expr) = branch else {
                return None;
            };
            compiled.push(self.compile_pure_expr(expr).ok()?);
        }
        Some(compiled)
    }

    fn compile_named_parallel_calls(
        &mut self,
        branches: &[crate::ast::NamedParallelBranch],
    ) -> Option<Vec<NamedParallelCallBranch>> {
        let mut compiled = Vec::with_capacity(branches.len());
        for branch in branches {
            let call = match &branch.stmt {
                Stmt::Call(call) | Stmt::Expr(Expr::ToolCall(call)) => call,
                _ => return None,
            };
            if call.args.iter().any(|(_, expr)| !is_pure_expr(expr)) {
                return None;
            }
            let args = PureExpr::Record(
                call.args
                    .iter()
                    .map(|(key, expr)| Ok((self.push_name(key), self.compile_pure_expr(expr)?)))
                    .collect::<Result<Vec<_>, RuntimeError>>()
                    .ok()?
                    .into_boxed_slice(),
            );
            compiled.push(NamedParallelCallBranch {
                output_name: self.push_name(&branch.name),
                name: self.push_name(&call.name),
                args,
            });
        }
        Some(compiled)
    }

    fn compile_pure_named_parallel_exprs(
        &mut self,
        branches: &[crate::ast::NamedParallelBranch],
    ) -> Option<Vec<(usize, PureExpr)>> {
        let mut compiled = Vec::with_capacity(branches.len());
        for branch in branches {
            let Stmt::Expr(expr) = &branch.stmt else {
                return None;
            };
            let expr = self.compile_pure_expr(expr).ok()?;
            compiled.push((self.push_name(&branch.name), expr));
        }
        Some(compiled)
    }

    fn compile_pure_expr(&mut self, expr: &Expr) -> Result<PureExpr, RuntimeError> {
        match expr {
            Expr::Null => Ok(PureExpr::Const(Value::Null)),
            Expr::Bool(value) => Ok(PureExpr::Const(Value::Bool(*value))),
            Expr::Number(value) => Ok(PureExpr::Const(Value::Number(*value))),
            Expr::String(value) => Ok(PureExpr::Const(Value::String(value.clone()))),
            Expr::Variable(name) => Ok(PureExpr::Slot(self.push_slot(name))),
            Expr::List(items) => Ok(PureExpr::List(
                items
                    .iter()
                    .map(|item| self.compile_pure_expr(item))
                    .collect::<Result<Vec<_>, _>>()?
                    .into_boxed_slice(),
            )),
            Expr::Record(entries) => Ok(PureExpr::Record(
                entries
                    .iter()
                    .map(|(key, expr)| Ok((self.push_name(key), self.compile_pure_expr(expr)?)))
                    .collect::<Result<Vec<_>, RuntimeError>>()?
                    .into_boxed_slice(),
            )),
            Expr::ToolCall(_) => Err(RuntimeError::ValueError {
                message: "tool calls are not allowed in pure expressions".to_string(),
            }),
            Expr::StartToolCall(_) => Err(RuntimeError::ValueError {
                message: "async tool starts are not allowed in pure expressions".to_string(),
            }),
            Expr::Parallel { .. } => Err(RuntimeError::ValueError {
                message: "`parallel` is not allowed in pure expressions".to_string(),
            }),
            Expr::Await(_) => Err(RuntimeError::ValueError {
                message: "`await` is not allowed in pure expressions".to_string(),
            }),
            Expr::ResultUnwrap(expr) => Ok(PureExpr::ResultUnwrap(Box::new(
                self.compile_pure_expr(expr)?,
            ))),
            Expr::BuiltinCall { name, args } => {
                if name == "format"
                    && let Some((Expr::String(template), value_args)) = args.split_first()
                {
                    return Ok(PureExpr::Format {
                        template: compile_format_template(template, value_args.len()),
                        args: value_args
                            .iter()
                            .map(|arg| self.compile_pure_expr(arg))
                            .collect::<Result<Vec<_>, _>>()?
                            .into_boxed_slice(),
                    });
                }

                Ok(PureExpr::Builtin {
                    builtin: self.resolve_builtin(name),
                    args: args
                        .iter()
                        .map(|arg| self.compile_pure_expr(arg))
                        .collect::<Result<Vec<_>, _>>()?
                        .into_boxed_slice(),
                })
            }
            Expr::Field { target, field } => Ok(PureExpr::Field {
                target: Box::new(self.compile_pure_expr(target)?),
                field: self.push_name(field),
            }),
            Expr::Index { target, index } => Ok(PureExpr::Index {
                target: Box::new(self.compile_pure_expr(target)?),
                index: Box::new(self.compile_pure_expr(index)?),
            }),
            Expr::Unary { op, expr } => Ok(PureExpr::Unary {
                op: *op,
                expr: Box::new(self.compile_pure_expr(expr)?),
            }),
            Expr::Conditional {
                condition,
                then_expr,
                else_expr,
            } => Ok(PureExpr::Conditional {
                condition: Box::new(self.compile_pure_expr(condition)?),
                then_expr: Box::new(self.compile_pure_expr(then_expr)?),
                else_expr: Box::new(self.compile_pure_expr(else_expr)?),
            }),
            Expr::Binary { left, op, right } => Ok(PureExpr::Binary {
                left: Box::new(self.compile_pure_expr(left)?),
                op: *op,
                right: Box::new(self.compile_pure_expr(right)?),
            }),
            Expr::TypeLiteral(ty) => {
                let schema = fold_type(ty).ok_or_else(|| RuntimeError::ValueError {
                    message: "Type literals with `Ref` are not allowed in pure expressions"
                        .to_string(),
                })?;
                let mut wrapper = record_with_capacity(1);
                wrapper.insert(LASH_TYPE_KEY.to_string(), schema);
                Ok(PureExpr::Const(Value::Record(Arc::new(wrapper))))
            }
        }
    }

    fn fold_compile_time_expr(&self, expr: &Expr) -> Option<Value> {
        match expr {
            Expr::Null => Some(Value::Null),
            Expr::Bool(value) => Some(Value::Bool(*value)),
            Expr::Number(value) => Some(Value::Number(*value)),
            Expr::String(value) => Some(Value::String(value.clone())),
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
                    "len" => Builtin::Len,
                    "empty" => Builtin::Empty,
                    "keys" => Builtin::Keys,
                    "values" => Builtin::Values,
                    "contains" => Builtin::Contains,
                    "find" => Builtin::Find,
                    "grep_text" => Builtin::GrepText,
                    "starts_with" => Builtin::StartsWith,
                    "ends_with" => Builtin::EndsWith,
                    "split" => Builtin::Split,
                    "join" => Builtin::Join,
                    "trim" => Builtin::Trim,
                    "slice" => Builtin::Slice,
                    "to_string" => Builtin::ToString,
                    "to_int" => Builtin::ToInt,
                    "to_float" => Builtin::ToFloat,
                    "json_parse" => Builtin::JsonParse,
                    "format" => Builtin::Format,
                    "validate" => Builtin::Validate,
                    "range" => Builtin::Range,
                    "push" => Builtin::Push,
                    _ => return None,
                };
                match builtin {
                    Builtin::Range => execute_range_builtin(&values).ok(),
                    _ => {
                        let _ = values;
                        None
                    }
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
            Expr::Conditional {
                condition,
                then_expr,
                else_expr,
            } => {
                if is_truthy(&self.fold_compile_time_expr(condition)?) {
                    self.fold_compile_time_expr(then_expr)
                } else {
                    self.fold_compile_time_expr(else_expr)
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
            Expr::ToolCall(_)
            | Expr::StartToolCall(_)
            | Expr::Parallel { .. }
            | Expr::Await(_)
            | Expr::ResultUnwrap(_) => None,
        }
    }

    fn emit_builtin_call(&mut self, name: &str, args: &[Expr]) {
        if name == "format"
            && let Some((Expr::String(template), value_args)) = args.split_first()
        {
            for arg in value_args {
                self.compile_expr(arg);
            }
            let template = self.push_format_template(template, value_args.len());
            self.code.push(Instruction::FormatCompiled(template));
            return;
        }

        match (name, args.len()) {
            ("len", 1) => {
                self.compile_expr(&args[0]);
                self.code.push(Instruction::Len);
            }
            ("join", 2) => {
                self.compile_expr(&args[0]);
                self.compile_expr(&args[1]);
                self.code.push(Instruction::Join);
            }
            ("validate", 2) => {
                if let Some(schema_wrapper) = self.fold_compile_time_expr(&args[1])
                    && let Some(schema) = unwrap_type_value(&schema_wrapper).cloned()
                {
                    self.compile_expr(&args[0]);
                    let schema = self.push_compiled_schema(&schema);
                    self.code.push(Instruction::ValidateCompiled(schema));
                    return;
                }

                self.compile_expr(&args[0]);
                self.compile_expr(&args[1]);
                self.code.push(Instruction::Validate);
            }
            ("push", 2) => {
                self.compile_expr(&args[0]);
                self.compile_expr(&args[1]);
                self.code.push(Instruction::Push);
            }
            ("range", 1 | 2) => {
                for arg in args {
                    self.compile_expr(arg);
                }
                self.code.push(Instruction::Range { argc: args.len() });
            }
            _ => {
                for arg in args {
                    self.compile_expr(arg);
                }
                let builtin = self.resolve_builtin(name);
                self.code.push(Instruction::CallBuiltin {
                    builtin,
                    argc: args.len(),
                });
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
            Expr::ToolCall(call) => self.compile_call_expr(call),
            Expr::StartToolCall(call) => self.compile_start_call_expr(call),
            Expr::Parallel { branches } => self.compile_parallel(branches, true),
            Expr::Await(handle) => {
                self.compile_expr(handle);
                self.code.push(Instruction::AwaitHandle);
            }
            Expr::ResultUnwrap(expr) => {
                if let Expr::ToolCall(call) = expr.as_ref() {
                    self.compile_call_unwrap_expr(call);
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
            Expr::Conditional {
                condition,
                then_expr,
                else_expr,
            } => {
                let jump_to_else = self.compile_condition_jump_if_false(condition);
                self.compile_expr(then_expr);
                let jump_to_end = self.emit_jump();
                self.patch_jump(jump_to_else, self.code.len());
                self.compile_expr(else_expr);
                self.patch_jump(jump_to_end, self.code.len());
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
                    self.compile_expr(left);
                    self.compile_expr(right);
                    self.code.push(Instruction::Binary(*op));
                }
            },
        }
    }

    fn compile_call_expr(&mut self, call: &CallExpr) {
        for (_, expr) in &call.args {
            self.compile_expr(expr);
        }
        let keys = self.push_key_list(call.args.iter().map(|(name, _)| name.as_str()));
        let name = self.push_name(&call.name);
        self.code.push(Instruction::CallTool { name, keys });
    }

    fn compile_call_unwrap_expr(&mut self, call: &CallExpr) {
        for (_, expr) in &call.args {
            self.compile_expr(expr);
        }
        let keys = self.push_key_list(call.args.iter().map(|(name, _)| name.as_str()));
        let name = self.push_name(&call.name);
        self.code.push(Instruction::CallToolUnwrap { name, keys });
    }

    fn compile_start_call_expr(&mut self, call: &CallExpr) {
        for (_, expr) in &call.args {
            self.compile_expr(expr);
        }
        let keys = self.push_key_list(call.args.iter().map(|(name, _)| name.as_str()));
        let name = self.push_name(&call.name);
        self.code.push(Instruction::StartCallTool { name, keys });
    }

    fn compile_parallel(&mut self, branches: &ParallelBranches, want_value: bool) {
        match branches {
            ParallelBranches::Positional(branches) => {
                if let Some(branches) = self.compile_parallel_calls(branches) {
                    let branches = self.push_parallel_call_set(branches);
                    self.code.push(if want_value {
                        Instruction::ParallelCallsValue(branches)
                    } else {
                        Instruction::ParallelCalls(branches)
                    });
                    return;
                }

                if want_value && let Some(branches) = self.compile_pure_parallel_exprs(branches) {
                    let branches = self.push_pure_parallel_set(branches);
                    self.code.push(Instruction::PureParallelValue(branches));
                    return;
                }

                let branches = branches
                    .iter()
                    .map(|branch| {
                        let mut compiler = Self::with_slots_and_stats(
                            self.slots.clone(),
                            self.compile_stats.clone(),
                        );
                        compiler.compile_stmt(branch);
                        compiler.finish()
                    })
                    .collect::<Vec<_>>();
                let branches = self.push_branch_set(branches);
                self.code.push(if want_value {
                    Instruction::ParallelValue(branches)
                } else {
                    Instruction::Parallel(branches)
                });
            }
            ParallelBranches::Named(branches) => {
                if want_value && let Some(branches) = self.compile_named_parallel_calls(branches) {
                    let branches = self.push_named_parallel_call_set(branches);
                    self.code
                        .push(Instruction::ParallelNamedCallsValue(branches));
                    return;
                }

                if want_value
                    && let Some(branches) = self.compile_pure_named_parallel_exprs(branches)
                {
                    let branches = self.push_pure_named_parallel_set(branches);
                    self.code
                        .push(Instruction::PureParallelNamedValue(branches));
                    return;
                }

                let branches = branches
                    .iter()
                    .map(|branch| {
                        let mut compiler = Self::with_slots_and_stats(
                            self.slots.clone(),
                            self.compile_stats.clone(),
                        );
                        compiler.compile_stmt(&branch.stmt);
                        NamedBranchChunk {
                            name: self.push_name(&branch.name),
                            chunk: compiler.finish(),
                        }
                    })
                    .collect::<Vec<_>>();
                let branches = self.push_named_branch_set(branches);
                self.code.push(if want_value {
                    Instruction::ParallelNamedValue(branches)
                } else {
                    Instruction::ParallelNamed(branches)
                });
            }
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
        Expr::Null | Expr::Bool(_) | Expr::Number(_) | Expr::String(_) | Expr::Variable(_) => true,
        Expr::List(items) => items.iter().all(is_pure_expr),
        Expr::Record(entries) => entries.iter().all(|(_, value)| is_pure_expr(value)),
        Expr::ToolCall(_) => false,
        Expr::StartToolCall(_) => false,
        Expr::Parallel { .. } => false,
        Expr::Await(_) => false,
        Expr::ResultUnwrap(expr) => is_pure_expr(expr),
        Expr::BuiltinCall { args, .. } => args.iter().all(is_pure_expr),
        Expr::Field { target, .. } => is_pure_expr(target),
        Expr::Index { target, index } => is_pure_expr(target) && is_pure_expr(index),
        Expr::Unary { expr, .. } => is_pure_expr(expr),
        Expr::Conditional {
            condition,
            then_expr,
            else_expr,
        } => is_pure_expr(condition) && is_pure_expr(then_expr) && is_pure_expr(else_expr),
        Expr::Binary { left, right, .. } => is_pure_expr(left) && is_pure_expr(right),
        Expr::TypeLiteral(ty) => fold_type(ty).is_some(),
    }
}

fn contains_type_literal(expr: &Expr) -> bool {
    match expr {
        Expr::TypeLiteral(_) => true,
        Expr::List(items) => items.iter().any(contains_type_literal),
        Expr::Record(entries) => entries
            .iter()
            .any(|(_, value)| contains_type_literal(value)),
        Expr::ToolCall(call) | Expr::StartToolCall(call) => call
            .args
            .iter()
            .any(|(_, value)| contains_type_literal(value)),
        Expr::Parallel { branches } => match branches {
            ParallelBranches::Positional(statements) => {
                statements.iter().any(stmt_contains_type_literal)
            }
            ParallelBranches::Named(branches) => branches
                .iter()
                .any(|branch| stmt_contains_type_literal(&branch.stmt)),
        },
        Expr::Await(expr) | Expr::ResultUnwrap(expr) | Expr::Unary { expr, .. } => {
            contains_type_literal(expr)
        }
        Expr::BuiltinCall { args, .. } => args.iter().any(contains_type_literal),
        Expr::Field { target, .. } => contains_type_literal(target),
        Expr::Index { target, index } => {
            contains_type_literal(target) || contains_type_literal(index)
        }
        Expr::Conditional {
            condition,
            then_expr,
            else_expr,
        } => {
            contains_type_literal(condition)
                || contains_type_literal(then_expr)
                || contains_type_literal(else_expr)
        }
        Expr::Binary { left, right, .. } => {
            contains_type_literal(left) || contains_type_literal(right)
        }
        Expr::Null | Expr::Bool(_) | Expr::Number(_) | Expr::String(_) | Expr::Variable(_) => false,
    }
}

fn stmt_contains_type_literal(stmt: &Stmt) -> bool {
    match stmt {
        Stmt::Assign { target, expr } => {
            assign_target_contains_type_literal(target) || contains_type_literal(expr)
        }
        Stmt::Expr(expr) | Stmt::Cancel(expr) | Stmt::Print(expr) => contains_type_literal(expr),
        Stmt::Call(call) => call
            .args
            .iter()
            .any(|(_, expr)| contains_type_literal(expr)),
        Stmt::If {
            condition,
            then_block,
            else_block,
        } => {
            contains_type_literal(condition)
                || then_block.iter().any(stmt_contains_type_literal)
                || else_block.iter().any(stmt_contains_type_literal)
        }
        Stmt::For { iterable, body, .. } => {
            contains_type_literal(iterable) || body.iter().any(stmt_contains_type_literal)
        }
        Stmt::Break | Stmt::Continue => false,
        Stmt::Parallel { branches } => match branches {
            ParallelBranches::Positional(statements) => {
                statements.iter().any(stmt_contains_type_literal)
            }
            ParallelBranches::Named(branches) => branches
                .iter()
                .any(|branch| stmt_contains_type_literal(&branch.stmt)),
        },
        Stmt::Submit(expr) => expr.as_ref().is_some_and(contains_type_literal),
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
