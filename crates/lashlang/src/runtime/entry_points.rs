//! Public compile/execute entry points for lashlang programs.

use crate::ast::Program;

use super::record::intern_symbol;
use super::{
    CompiledProgram, Compiler, ExecutionHost, ExecutionOutcome, ExecutionScratch, LASH_TYPE_KEY,
    ProjectedBindings, RuntimeError, SlotState, State, Vm,
};

pub enum ExecutableProgram<'program> {
    Program(&'program Program),
    Compiled(&'program CompiledProgram),
}

impl<'program> From<&'program Program> for ExecutableProgram<'program> {
    fn from(program: &'program Program) -> Self {
        Self::Program(program)
    }
}

impl<'program> From<&'program CompiledProgram> for ExecutableProgram<'program> {
    fn from(program: &'program CompiledProgram) -> Self {
        Self::Compiled(program)
    }
}

pub fn compile(source: &str) -> Result<CompiledProgram, crate::parser::ParseError> {
    crate::parse(source).map(|program| compile_program_internal(&program))
}

pub(crate) fn compile_program_internal(program: &Program) -> CompiledProgram {
    let (chunk, compile_stats) = Compiler::compile_program(program);
    CompiledProgram {
        chunk,
        compile_stats,
    }
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

pub async fn execute<'program, H: ExecutionHost>(
    program: impl Into<ExecutableProgram<'program>>,
    state: &mut State,
    host: &H,
) -> Result<ExecutionOutcome, RuntimeError> {
    match program.into() {
        ExecutableProgram::Program(program) => {
            let compiled = compile_program_internal(program);
            execute_compiled_internal(&compiled, state, host).await
        }
        ExecutableProgram::Compiled(compiled) => {
            execute_compiled_internal(compiled, state, host).await
        }
    }
}

pub(crate) async fn execute_compiled_internal<H: ExecutionHost>(
    program: &CompiledProgram,
    state: &mut State,
    host: &H,
) -> Result<ExecutionOutcome, RuntimeError> {
    let projected = host.projected_bindings();
    if let Some(mut scratch) = host.take_scratch() {
        let result =
            execute_with_optional_scratch(program, state, host, &projected, Some(&mut scratch))
                .await;
        host.store_scratch(scratch);
        result
    } else {
        execute_with_optional_scratch(program, state, host, &projected, None).await
    }
}

async fn execute_with_optional_scratch<H: ExecutionHost>(
    program: &CompiledProgram,
    state: &mut State,
    host: &H,
    projected: &ProjectedBindings,
    scratch: Option<&mut ExecutionScratch>,
) -> Result<ExecutionOutcome, RuntimeError> {
    if let Some(scratch) = scratch {
        let slots = SlotState::from_globals_with_scratch(
            std::mem::take(&mut state.globals),
            &program.chunk.slot_names,
            scratch,
            projected,
        );
        let mut vm = Vm::new_with_scratch_and_mode(
            &program.chunk,
            slots,
            host,
            scratch,
            host.execution_mode(),
        );
        let result = run_vm(program, host, &mut vm).await;
        state.globals = vm.recycle_into_globals(scratch);
        result
    } else {
        let slots = SlotState::from_globals(
            std::mem::take(&mut state.globals),
            &program.chunk.slot_names,
            projected,
        );
        let mut vm = Vm::new_with_mode(&program.chunk, slots, host, host.execution_mode());
        let result = run_vm(program, host, &mut vm).await;
        state.globals = vm.into_globals();
        result
    }
}

async fn run_vm<H: ExecutionHost>(
    program: &CompiledProgram,
    host: &H,
    vm: &mut Vm<'_, H>,
) -> Result<ExecutionOutcome, RuntimeError> {
    if host.profile_execution() {
        vm.enable_profile();
    }

    let result = if host.trace_runtime_errors() {
        vm.run_traced_for_mode().await.map_err(|failure| {
            let error = failure.error.clone();
            host.observe_runtime_failure(failure);
            error
        })
    } else {
        vm.run_for_mode().await
    };

    if host.profile_execution() {
        let mut profile = vm.take_profile();
        profile.compile_stats = program.compile_stats;
        host.observe_profile(profile);
    }

    result
}
