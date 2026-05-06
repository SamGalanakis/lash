use crate::lexer::Span;
use thiserror::Error;

mod access;
mod cache;
mod compiler;
mod entry_points;
mod format;
mod host;
mod instruction;
mod json;
mod ops;
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
#[allow(unused_imports)]
pub(crate) use instruction::*;
#[allow(unused_imports)]
pub(crate) use compiler::*;
#[allow(unused_imports)]
pub(crate) use vm::*;
#[allow(unused_imports)]
pub(crate) use record::{Symbol, intern_symbol, lookup_symbol, record_with_capacity, symbol_name};
#[allow(unused_imports)]
pub(crate) use schema::{
    CompiledSchema, compile_schema_value, execute_compiled_validate, execute_validate_builtin,
};
pub use record::Record;
pub use json::from_json;
// Re-exports of helpers that live in the focused submodules but need to be
// reachable via `use super::*` from sibling submodules + via `super::name`
// from `vm.rs` / `compiler.rs`. These look "unused" from mod.rs's POV but
// are load-bearing for the rest of the runtime crate.
#[allow(unused_imports)]
pub(crate) use access::{
    add_assign_index_number, add_assign_value_number, assign_path, descend_index,
    descend_record_field, is_async_handle_record, read_field, read_field_direct,
    read_field_blocking, read_field_ref_direct, read_image_field, read_index, read_index_direct,
    read_index_ref_direct, read_index_blocking, resolve_existing_list_assignment_index,
    resolve_index, unwrap_tool_result,
};
#[allow(unused_imports)]
pub(crate) use format::{
    append_stringified_value_async, append_stringified_value_direct, apply_format_async,
    clamp_slice_bounds, compile_format_template, execute_compiled_format,
    execute_compiled_format_blocking, execute_compiled_format_direct, normalize_slice_bound,
    parse_format_template, push_format_literal, slice_string, stringify_value_async,
    stringify_value_blocking, stringify_value_direct, write_number,
};
#[cfg(test)]
#[allow(unused_imports)]
pub(crate) use format::{append_stringified_value, apply_format, stringify_value};
#[allow(unused_imports)]
pub(crate) use json::{
    image_from_json_map, image_to_json, json_number, optional_u32_field, to_json, to_json_async,
    to_json_blocking, to_json_direct,
};
#[allow(unused_imports)]
pub(crate) use ops::{
    add_values, as_number, as_offset, as_range_bound, as_slice_bound, build_range, coerce_string,
    compare_numbers, compare_ordered, error_value, eval_binary_values, eval_binary_values_async,
    eval_compare_values, eval_compare_values_async, eval_number_binary_values,
    eval_number_compare_values, eval_number_numeric_binary_value, eval_pure_expr,
    execute_builtin, execute_builtin_blocking, execute_contains_builtin, execute_contains_direct,
    execute_join_builtin, execute_len_builtin, execute_len_direct, execute_push_builtin,
    execute_range_builtin, expect_arg_count, expect_bool_value, is_comparison_binary_op,
    is_numeric_binary_op, is_truthy, is_truthy_async, iterable_values, materialize_projected_async,
    materialize_value, numeric_binary_values, range_bounds, success, validate_range_len,
    value_contains_projected, value_len, value_type_name,
};
pub use state::{Snapshot, State};
use vm::IterState;
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







#[cfg(test)]
mod tests;
