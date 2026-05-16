//! Public entry points for compiling and executing lashlang programs.
//!
//! These are thin wrappers that instantiate the `Compiler` and `Vm`, plumb
//! through optional `ExecutionScratch` reuse and projected bindings, and
//! emit either an `ExecutionOutcome` or a profile report. All real work
//! lives in `compiler.rs` and `vm.rs` (or, currently, in `mod.rs` until
//! those stages land).

use crate::ast::Program;

use super::record::intern_symbol;
use super::{
    CompiledProgram, Compiler, ExecutionOutcome, ExecutionScratch, LASH_TYPE_KEY, ProfileReport,
    ProjectedBindings, RuntimeError, RuntimeFailure, SlotState, State, ToolHost, Vm,
};

pub fn compile_source(source: &str) -> Result<CompiledProgram, crate::parser::ParseError> {
    crate::parse(source).map(|program| compile_program(&program))
}

pub fn prewarm() {
    for name in [
        "ok",
        "value",
        "error",
        "__handle__",
        "handle",
        LASH_TYPE_KEY,
        "type",
        "properties",
        "required",
        "items",
        "enum",
        "id",
        "label",
        "size",
        "width",
        "height",
    ] {
        intern_symbol(name);
    }
}

pub async fn execute_program<H: ToolHost>(
    program: &Program,
    state: &mut State,
    host: &H,
) -> Result<ExecutionOutcome, RuntimeError> {
    let compiled = compile_program(program);
    execute_compiled(&compiled, state, host).await
}

pub fn compile_program(program: &Program) -> CompiledProgram {
    let (chunk, compile_stats) = Compiler::compile_program(program);
    CompiledProgram {
        chunk,
        compile_stats,
    }
}

pub async fn execute_compiled<H: ToolHost>(
    program: &CompiledProgram,
    state: &mut State,
    host: &H,
) -> Result<ExecutionOutcome, RuntimeError> {
    execute_compiled_with_projected_bindings(program, state, host, &ProjectedBindings::default())
        .await
}

pub async fn execute_compiled_with_projected_bindings<H: ToolHost>(
    program: &CompiledProgram,
    state: &mut State,
    host: &H,
    projected: &ProjectedBindings,
) -> Result<ExecutionOutcome, RuntimeError> {
    let mut vm = Vm::new(
        &program.chunk,
        SlotState::from_globals(
            std::mem::take(&mut state.globals),
            &program.chunk.slot_names,
            projected,
        ),
        host,
        false,
    );
    let result = vm.run().await;
    state.globals = vm.into_globals();
    result
}

pub async fn execute_compiled_with_scratch<H: ToolHost>(
    program: &CompiledProgram,
    state: &mut State,
    host: &H,
    scratch: &mut ExecutionScratch,
) -> Result<ExecutionOutcome, RuntimeError> {
    execute_compiled_with_scratch_and_projected_bindings(
        program,
        state,
        host,
        scratch,
        &ProjectedBindings::default(),
    )
    .await
}

pub async fn execute_compiled_with_scratch_and_projected_bindings<H: ToolHost>(
    program: &CompiledProgram,
    state: &mut State,
    host: &H,
    scratch: &mut ExecutionScratch,
    projected: &ProjectedBindings,
) -> Result<ExecutionOutcome, RuntimeError> {
    let slots = SlotState::from_globals_with_scratch(
        std::mem::take(&mut state.globals),
        &program.chunk.slot_names,
        scratch,
        projected,
    );
    let mut vm = Vm::new_with_scratch(&program.chunk, slots, host, false, scratch);
    let result = vm.run().await;
    state.globals = vm.recycle_into_globals(scratch);
    result
}

pub async fn execute_compiled_traced<H: ToolHost>(
    program: &CompiledProgram,
    state: &mut State,
    host: &H,
) -> Result<ExecutionOutcome, RuntimeFailure> {
    execute_compiled_traced_with_projected_bindings(
        program,
        state,
        host,
        &ProjectedBindings::default(),
    )
    .await
}

pub async fn execute_compiled_traced_with_projected_bindings<H: ToolHost>(
    program: &CompiledProgram,
    state: &mut State,
    host: &H,
    projected: &ProjectedBindings,
) -> Result<ExecutionOutcome, RuntimeFailure> {
    let mut vm = Vm::new(
        &program.chunk,
        SlotState::from_globals(
            std::mem::take(&mut state.globals),
            &program.chunk.slot_names,
            projected,
        ),
        host,
        false,
    );
    let result = vm.run_traced().await;
    state.globals = vm.into_globals();
    result
}

pub async fn execute_compiled_traced_with_scratch<H: ToolHost>(
    program: &CompiledProgram,
    state: &mut State,
    host: &H,
    scratch: &mut ExecutionScratch,
) -> Result<ExecutionOutcome, RuntimeFailure> {
    execute_compiled_traced_with_scratch_and_projected_bindings(
        program,
        state,
        host,
        scratch,
        &ProjectedBindings::default(),
    )
    .await
}

pub async fn execute_compiled_traced_with_scratch_and_projected_bindings<H: ToolHost>(
    program: &CompiledProgram,
    state: &mut State,
    host: &H,
    scratch: &mut ExecutionScratch,
    projected: &ProjectedBindings,
) -> Result<ExecutionOutcome, RuntimeFailure> {
    let slots = SlotState::from_globals_with_scratch(
        std::mem::take(&mut state.globals),
        &program.chunk.slot_names,
        scratch,
        projected,
    );
    let mut vm = Vm::new_with_scratch(&program.chunk, slots, host, false, scratch);
    let result = vm.run_traced().await;
    state.globals = vm.recycle_into_globals(scratch);
    result
}

pub async fn profile_compiled<H: ToolHost>(
    program: &CompiledProgram,
    state: &mut State,
    host: &H,
) -> Result<(ExecutionOutcome, ProfileReport), RuntimeError> {
    profile_compiled_with_projected_bindings(program, state, host, &ProjectedBindings::default())
        .await
}

pub async fn profile_compiled_with_projected_bindings<H: ToolHost>(
    program: &CompiledProgram,
    state: &mut State,
    host: &H,
    projected: &ProjectedBindings,
) -> Result<(ExecutionOutcome, ProfileReport), RuntimeError> {
    let mut vm = Vm::new(
        &program.chunk,
        SlotState::from_globals(
            std::mem::take(&mut state.globals),
            &program.chunk.slot_names,
            projected,
        ),
        host,
        false,
    );
    vm.enable_profile();
    let result = vm.run().await;
    let mut profile = vm.take_profile();
    state.globals = vm.into_globals();
    profile.compile_stats = program.compile_stats;
    result.map(|outcome| (outcome, profile))
}

pub async fn profile_compiled_with_scratch<H: ToolHost>(
    program: &CompiledProgram,
    state: &mut State,
    host: &H,
    scratch: &mut ExecutionScratch,
) -> Result<(ExecutionOutcome, ProfileReport), RuntimeError> {
    profile_compiled_with_scratch_and_projected_bindings(
        program,
        state,
        host,
        scratch,
        &ProjectedBindings::default(),
    )
    .await
}

pub async fn profile_compiled_with_scratch_and_projected_bindings<H: ToolHost>(
    program: &CompiledProgram,
    state: &mut State,
    host: &H,
    scratch: &mut ExecutionScratch,
    projected: &ProjectedBindings,
) -> Result<(ExecutionOutcome, ProfileReport), RuntimeError> {
    let slots = SlotState::from_globals_with_scratch(
        std::mem::take(&mut state.globals),
        &program.chunk.slot_names,
        scratch,
        projected,
    );
    let mut vm = Vm::new_with_scratch(&program.chunk, slots, host, false, scratch);
    vm.enable_profile();
    let result = vm.run().await;
    let mut profile = vm.take_profile();
    state.globals = vm.recycle_into_globals(scratch);
    profile.compile_stats = program.compile_stats;
    result.map(|outcome| (outcome, profile))
}
