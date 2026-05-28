use super::*;
use crate::ast::{Expr, Program};
use std::fmt::Write as _;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};

#[derive(Default)]
struct Host;

impl ExecutionHost for Host {
    async fn perform(&self, op: AbilityOp) -> Result<AbilityResult, ExecutionHostError> {
        match op {
            AbilityOp::ResourceOperation(operation) => match operation.operation.as_str() {
                "echo" => {
                    let value = operation
                        .args
                        .first()
                        .and_then(Value::as_record)
                        .and_then(|record| record.get("value"))
                        .cloned()
                        .unwrap_or(Value::Null);
                    Ok(AbilityResult::Value(value))
                }
                "err" => Err(ExecutionHostError::new("boom")),
                other => Err(ExecutionHostError::new(format!(
                    "unknown module operation: {other}"
                ))),
            },
            AbilityOp::Await(handle) => match handle {
                Value::Record(_) => Ok(AbilityResult::Value(Value::Null)),
                _ => Err(ExecutionHostError::new("expected handle record")),
            },
            AbilityOp::Print(_) => Ok(AbilityResult::Unit),
            AbilityOp::Submit(value) | AbilityOp::Finish(value) | AbilityOp::Fail(value) => {
                Ok(AbilityResult::Value(value))
            }
            _ => Err(ExecutionHostError::new("unsupported host ability")),
        }
    }
}

#[derive(Default)]
struct RecordingProcessHost {
    starts: Mutex<Vec<ProcessStart>>,
    events: Mutex<Vec<ProcessEvent>>,
    sleeps: Mutex<Vec<ProcessSleep>>,
    signals: Mutex<Vec<ProcessSignal>>,
}

impl ExecutionHost for RecordingProcessHost {
    async fn perform(&self, op: AbilityOp) -> Result<AbilityResult, ExecutionHostError> {
        match op {
            AbilityOp::ResourceOperation(_) => Err(ExecutionHostError::new(
                "module operations are not supported by this host",
            )),
            AbilityOp::StartProcess(start) => {
                self.starts.lock().expect("starts lock").push(*start);
                let mut handle = Record::new();
                handle.insert("__handle__".to_string(), Value::String("process".into()));
                handle.insert("id".to_string(), Value::String("proc-1".into()));
                Ok(AbilityResult::Value(Value::Record(Arc::new(handle))))
            }
            AbilityOp::ProcessEvent(event) => {
                self.events.lock().expect("events lock").push(event);
                Ok(AbilityResult::Unit)
            }
            AbilityOp::ProcessSleep(sleep) => {
                self.sleeps.lock().expect("sleeps lock").push(sleep);
                Ok(AbilityResult::Value(Value::Null))
            }
            AbilityOp::WaitSignal => {
                Ok(AbilityResult::Value(Value::String("signal-payload".into())))
            }
            AbilityOp::SignalRun(signal) => {
                self.signals.lock().expect("signals lock").push(signal);
                Ok(AbilityResult::Value(Value::Null))
            }
            AbilityOp::Submit(value) | AbilityOp::Finish(value) | AbilityOp::Fail(value) => {
                Ok(AbilityResult::Value(value))
            }
            _ => Err(ExecutionHostError::new("unsupported host ability")),
        }
    }
}

async fn exec(source: &str) -> Result<Value, RuntimeError> {
    let program = crate::parse(source).expect("program should parse");
    let mut state = State::new();
    match execute_program(&program, &mut state, &Host).await? {
        ExecutionOutcome::Finished(value) => Ok(value),
        ExecutionOutcome::Continued => panic!("expected `submit` in test program"),
        ExecutionOutcome::Failed(value) => panic!("unexpected process failure: {value}"),
    }
}

async fn exec_outcome(source: &str) -> Result<ExecutionOutcome, RuntimeError> {
    let program = crate::parse(source).expect("program should parse");
    let mut state = State::new();
    execute_program(&program, &mut state, &Host).await
}

fn compile_source(source: &str) -> Result<CompiledProgram, crate::ParseError> {
    let program = crate::parse(source)?;
    if source.contains("tools.")
        && let Ok(linked) = crate::LinkedModule::link(program.clone(), runtime_test_surface())
    {
        Ok(crate::compile_linked(&linked))
    } else {
        Ok(compile_program(&program))
    }
}

fn compile_program(program: &Program) -> CompiledProgram {
    super::entry_points::compile_program_internal(program)
}

fn runtime_test_surface() -> crate::LashlangSurface {
    let mut resources = crate::ResourceCatalog::new();
    resources.add_module_instance(["tools"], "Tools");
    resources.add_operation("Tools", "echo", "echo");
    resources.add_operation("Tools", "err", "err");
    resources.add_operation("Tools", "missing", "missing");
    resources.add_operation("Tools", "spawn", "spawn");
    crate::LashlangSurface::new(resources, crate::LashlangAbilities::all())
}

async fn execute_program<H: ExecutionHost>(
    program: &Program,
    state: &mut State,
    host: &H,
) -> Result<ExecutionOutcome, RuntimeError> {
    if let Ok(linked) = crate::LinkedModule::link(program.clone(), runtime_test_surface()) {
        let compiled = crate::compile_linked(&linked);
        return super::execute(&compiled, state, host).await;
    }
    super::execute(program, state, host).await
}

async fn execute_compiled<H: ExecutionHost>(
    program: &CompiledProgram,
    state: &mut State,
    host: &H,
) -> Result<ExecutionOutcome, RuntimeError> {
    super::execute(program, state, host).await
}

async fn execute_compiled_with_projected_bindings<H: ExecutionHost>(
    program: &CompiledProgram,
    state: &mut State,
    host: &H,
    projected: &ProjectedBindings,
) -> Result<ExecutionOutcome, RuntimeError> {
    let env = ExecutionEnvironment::new(host).with_projected_bindings(projected.clone());
    super::execute(program, state, &env).await
}

async fn execute_compiled_with_scratch<H: ExecutionHost>(
    program: &CompiledProgram,
    state: &mut State,
    host: &H,
    scratch: &mut ExecutionScratch,
) -> Result<ExecutionOutcome, RuntimeError> {
    let env = ExecutionEnvironment::new(host).with_scratch(std::mem::take(scratch));
    let result = super::execute(program, state, &env).await;
    *scratch = env.take_recycled_scratch().unwrap_or_default();
    result
}

async fn execute_compiled_traced<H: ExecutionHost>(
    program: &CompiledProgram,
    state: &mut State,
    host: &H,
) -> Result<ExecutionOutcome, RuntimeFailure> {
    let env = ExecutionEnvironment::new(host).traced();
    match super::execute(program, state, &env).await {
        Ok(outcome) => Ok(outcome),
        Err(error) => Err(env
            .take_runtime_failure()
            .unwrap_or(RuntimeFailure { error, span: None })),
    }
}

async fn execute_compiled_traced_with_projected_bindings<H: ExecutionHost>(
    program: &CompiledProgram,
    state: &mut State,
    host: &H,
    projected: &ProjectedBindings,
) -> Result<ExecutionOutcome, RuntimeFailure> {
    let env = ExecutionEnvironment::new(host)
        .traced()
        .with_projected_bindings(projected.clone());
    match super::execute(program, state, &env).await {
        Ok(outcome) => Ok(outcome),
        Err(error) => Err(env
            .take_runtime_failure()
            .unwrap_or(RuntimeFailure { error, span: None })),
    }
}

async fn execute_compiled_process<H: ExecutionHost>(
    program: &CompiledProgram,
    state: &mut State,
    host: &H,
) -> Result<ExecutionOutcome, RuntimeError> {
    let env = ExecutionEnvironment::new(host).process();
    super::execute(program, state, &env).await
}

async fn profile_compiled<H: ExecutionHost>(
    program: &CompiledProgram,
    state: &mut State,
    host: &H,
) -> Result<(ExecutionOutcome, ProfileReport), RuntimeError> {
    let env = ExecutionEnvironment::new(host).profiled();
    let outcome = super::execute(program, state, &env).await?;
    let profile = env.take_profile().expect("profile should be recorded");
    Ok((outcome, profile))
}

const GOLDEN_CONTRACT_SOURCE: &str = r#"
source = join(history, ",")
beta_index = find(source, "beta")
matches = grep_text(source, "beta")
counts = {}
for token in split(source, ",") {
  counts[token] = counts[token] + 1
}
Payload = Type {
  beta_index: int | null,
  matches: list[dict],
  counts: dict
}
submit validate(
  { beta_index: beta_index, matches: matches, counts: counts },
  Payload
)
"#;

#[test]
fn golden_parser_ast_contract_covers_lashlang_surface() {
    let program = crate::parse(GOLDEN_CONTRACT_SOURCE).expect("program should parse");
    insta::assert_snapshot!(
        "lashlang_parser_ast_contract",
        serde_json::to_string_pretty(&program).expect("program should serialize")
    );
}

#[test]
fn golden_compiled_bytecode_contract_covers_lashlang_surface() {
    insta::assert_snapshot!(
        "lashlang_compiled_bytecode_contract",
        compiled_program_snapshot(GOLDEN_CONTRACT_SOURCE)
    );
}

#[tokio::test(flavor = "current_thread")]
async fn golden_runtime_diagnostic_contract_is_exact() {
    insta::assert_snapshot!(
        "lashlang_runtime_diagnostic_contract",
        runtime_diagnostic("x = 1\nsubmit len(true)").await
    );
}

#[tokio::test(flavor = "current_thread")]
async fn golden_lashlang_diagnostic_corpus_is_exact() {
    let mut projected = ProjectedBindings::new();
    projected.insert(
        "history",
        ProjectedValue::scalar(
            "history",
            Value::List(vec![Value::String("entry".into())].into()),
        ),
    );

    let mut cases = Vec::new();
    cases.push(diagnostic_case(
        "parse_unsupported_while",
        format_parse_diagnostic("x = 0\nwhile x < 3 {\n  x = x + 1\n}"),
    ));
    cases.push(diagnostic_case(
        "parse_inline_if",
        format_parse_diagnostic("submit if true { 1 }"),
    ));
    cases.push(diagnostic_case(
        "parse_inline_for",
        format_parse_diagnostic("submit [for x in [1] { x }]"),
    ));
    cases.push(diagnostic_case(
        "parse_loop_control_outside_loop",
        format_parse_diagnostic("break"),
    ));
    cases.push(diagnostic_case(
        "parse_duplicate_type_field",
        format_parse_diagnostic("Payload = Type { nested: { ok: bool, ok: str } }"),
    ));
    cases.push(diagnostic_case(
        "parse_removed_parallel_keyword",
        format_parse_diagnostic("parallel {\n  start echo(value: 1)\n}"),
    ));
    cases.push(diagnostic_case(
        "runtime_bad_wrapper_unwrap",
        runtime_diagnostic("submit ({ ok: false, error: \"boom\" })?").await,
    ));
    cases.push(diagnostic_case(
        "runtime_failed_resource_operation_unwrap",
        runtime_diagnostic("submit (await tools.err({})?)").await,
    ));
    cases.push(diagnostic_case(
        "runtime_invalid_await_handle",
        runtime_diagnostic("submit (await 1)?").await,
    ));
    cases.push(diagnostic_case(
        "runtime_read_only_projected_assignment",
        runtime_diagnostic_with_projected("history = []\nsubmit history", &projected).await,
    ));
    cases.push(diagnostic_case(
        "runtime_invalid_validate_type_argument",
        runtime_diagnostic("submit validate({ ok: true }, \"not a type\")").await,
    ));
    cases.push(diagnostic_case(
        "runtime_validation_failure",
        runtime_diagnostic("submit validate({ count: \"x\" }, Type { count: int })").await,
    ));
    cases.push(diagnostic_case(
        "runtime_unknown_name",
        runtime_diagnostic("submit missing").await,
    ));
    cases.push(diagnostic_case(
        "runtime_unknown_builtin",
        runtime_diagnostic("submit nope()").await,
    ));

    insta::assert_snapshot!("lashlang_diagnostic_corpus", cases.join("\n\n---\n\n"));
}

#[tokio::test(flavor = "current_thread")]
async fn golden_serialized_state_snapshot_contract_is_exact() {
    let program = crate::parse(
        r#"
counter = 7
Payload = Type { title: str, count: int }
submit Payload
"#,
    )
    .expect("program should parse");
    let mut state = State::new();
    let outcome = execute_program(&program, &mut state, &Host)
        .await
        .expect("program should execute");
    assert!(matches!(outcome, ExecutionOutcome::Finished(_)));
    state.globals.insert_str(
        "cover",
        Value::Image(ImageValue::new("img_1", "cover", 42, Some(640), Some(480))),
    );
    state.globals.insert_str(
        "projected",
        Value::Projected(ProjectedValue::custom(
            "matches[0].text",
            Arc::new(SnapshotGuardProjectedValue::default()),
        )),
    );

    insta::assert_snapshot!(
        "lashlang_serialized_state_snapshot_contract",
        serde_json::to_string_pretty(&state.snapshot()).expect("snapshot should serialize")
    );
}

fn format_parse_diagnostic(source: &str) -> String {
    let err = crate::parse(source).expect_err("parse should fail");
    crate::format_parse_diagnostic(source, &err)
}

async fn runtime_diagnostic(source: &str) -> String {
    runtime_diagnostic_with_projected(source, &ProjectedBindings::default()).await
}

async fn runtime_diagnostic_with_projected(source: &str, projected: &ProjectedBindings) -> String {
    let compiled = compile_source(source).expect("source should compile");
    let mut state = State::new();
    let failure =
        execute_compiled_traced_with_projected_bindings(&compiled, &mut state, &Host, projected)
            .await
            .expect_err("runtime should fail");
    crate::format_runtime_diagnostic(source, &failure.error, failure.span)
}

fn diagnostic_case(name: &str, diagnostic: String) -> String {
    format!("{name}\n{diagnostic}")
}

fn compiled_program_snapshot(source: &str) -> String {
    let compiled = compile_source(source).expect("source should compile");
    let chunk = &compiled.chunk;
    let mut out = String::new();

    let stats = compiled.compile_stats();
    writeln!(
        out,
        "compile_stats: total={} const_folded={} dynamic={} refs={}",
        stats.type_literals_total,
        stats.type_literals_const_folded,
        stats.type_literals_dynamic,
        stats.type_ref_sites
    )
    .unwrap();
    writeln!(
        out,
        "slots: [{}]",
        chunk
            .slot_names
            .iter()
            .map(|name| name.text.as_ref())
            .collect::<Vec<_>>()
            .join(", ")
    )
    .unwrap();
    writeln!(
        out,
        "names: [{}]",
        chunk
            .names
            .iter()
            .map(|name| name.text.as_ref())
            .collect::<Vec<_>>()
            .join(", ")
    )
    .unwrap();
    writeln!(out, "constants:").unwrap();
    for (index, value) in chunk.constants.iter().enumerate() {
        writeln!(out, "  c{index}: {}", compact_json(value)).unwrap();
    }
    writeln!(out, "code:").unwrap();
    for (index, instruction) in chunk.code.iter().copied().enumerate() {
        writeln!(
            out,
            "  {index:04}: {}",
            instruction_snapshot(chunk, instruction)
        )
        .unwrap();
    }

    out
}

fn instruction_snapshot(chunk: &Chunk, instruction: Instruction) -> String {
    match instruction {
        Instruction::PushConst(index) => {
            format!(
                "push_const c{index} {}",
                compact_json(&chunk.constants[index])
            )
        }
        Instruction::PushNull => "push_null".to_string(),
        Instruction::PushBool(value) => format!("push_bool {value}"),
        Instruction::PushNumber(value) => format!("push_number {value}"),
        Instruction::LoadName(slot) => format!("load_name {slot}:{}", slot_name(chunk, slot)),
        Instruction::StoreName(slot) => format!("store_name {slot}:{}", slot_name(chunk, slot)),
        Instruction::StoreConst { slot, constant } => format!(
            "store_const {slot}:{} c{constant} {}",
            slot_name(chunk, slot),
            compact_json(&chunk.constants[constant])
        ),
        Instruction::BuildList(count) => format!("build_list {count}"),
        Instruction::BuildRecord(keys) => format!("build_record {}", keys_snapshot(chunk, keys)),
        Instruction::LoadField { slot, field } => format!(
            "load_field {slot}:{} .{}",
            slot_name(chunk, slot),
            name(chunk, field)
        ),
        Instruction::LoadFieldUnwrap { slot, field } => format!(
            "load_field_unwrap {slot}:{} .{}",
            slot_name(chunk, slot),
            name(chunk, field)
        ),
        Instruction::Field(field) => format!("field .{}", name(chunk, field)),
        Instruction::Index => "index".to_string(),
        Instruction::PathAssign { slot, path } => format!(
            "path_assign {slot}:{} {}",
            slot_name(chunk, slot),
            assign_path_snapshot(chunk, path)
        ),
        Instruction::ResultUnwrap => "result_unwrap".to_string(),
        Instruction::Unary(op) => format!("unary {op:?}"),
        Instruction::Binary(op) => format!("binary {op:?}"),
        Instruction::SlotNumberBinary { slot, op, right } => format!(
            "slot_number_binary {slot}:{} {op:?} {right}",
            slot_name(chunk, slot)
        ),
        Instruction::SlotNumberCompare { slot, op, right } => format!(
            "slot_number_compare {slot}:{} {op:?} {right}",
            slot_name(chunk, slot)
        ),
        Instruction::SlotNumberBinaryCompare {
            slot,
            binary_op,
            binary_right,
            compare_op,
            compare_right,
        } => format!(
            "slot_number_binary_compare {slot}:{} {binary_op:?} {binary_right} {compare_op:?} {compare_right}",
            slot_name(chunk, slot)
        ),
        Instruction::ToBool => "to_bool".to_string(),
        Instruction::Jump(target) => format!("jump {target}"),
        Instruction::JumpIfFalse(target) => format!("jump_if_false {target}"),
        Instruction::JumpIfCompareFalse { op, target } => {
            format!("jump_if_compare_false {op:?} {target}")
        }
        Instruction::JumpIfSlotNumberCompareFalse {
            slot,
            op,
            right,
            target,
        } => format!(
            "jump_if_slot_number_compare_false {slot}:{} {op:?} {right} {target}",
            slot_name(chunk, slot)
        ),
        Instruction::JumpIfSlotNumberBinaryCompareFalse {
            slot,
            binary_op,
            binary_right,
            compare_op,
            compare_right,
            target,
        } => format!(
            "jump_if_slot_number_binary_compare_false {slot}:{} {binary_op:?} {binary_right} {compare_op:?} {compare_right} {target}",
            slot_name(chunk, slot)
        ),
        Instruction::JumpIfTrue(target) => format!("jump_if_true {target}"),
        Instruction::ResourceCall { operation, argc } => {
            format!("resource_call {} argc={argc}", name_text(chunk, operation))
        }
        Instruction::ResourceCallUnwrap { operation, argc } => {
            format!(
                "resource_call_unwrap {} argc={argc}",
                name_text(chunk, operation)
            )
        }
        Instruction::StartProcess { process, keys } => format!(
            "start_process {} {}",
            name_text(chunk, process),
            keys_snapshot(chunk, keys)
        ),
        Instruction::AwaitHandle => "await_handle".to_string(),
        Instruction::ProcessSleepFor => "process_sleep_for".to_string(),
        Instruction::ProcessSleepUntil => "process_sleep_until".to_string(),
        Instruction::ProcessWaitSignal => "process_wait_signal".to_string(),
        Instruction::ProcessSignalRun => "process_signal_run".to_string(),
        Instruction::AwaitHandleUnwrap => "await_handle_unwrap".to_string(),
        Instruction::CancelHandle => "cancel_handle".to_string(),
        Instruction::Intrinsic(op) => intrinsic_snapshot(chunk, op),
        Instruction::AddAssign(slot) => format!("add_assign {slot}:{}", slot_name(chunk, slot)),
        Instruction::AddAssignNumber { slot, right } => {
            format!(
                "add_assign_number {slot}:{} {right}",
                slot_name(chunk, slot)
            )
        }
        Instruction::AddAssignSlot { slot, right } => format!(
            "add_assign_slot {slot}:{} {right}:{}",
            slot_name(chunk, slot),
            slot_name(chunk, right)
        ),
        Instruction::AddAssignIndexNumber { slot, right } => format!(
            "add_assign_index_number {slot}:{} {right}",
            slot_name(chunk, slot)
        ),
        Instruction::AddAssignIndexSlotNumber { slot, index, right } => format!(
            "add_assign_index_slot_number {slot}:{} {index}:{} {right}",
            slot_name(chunk, slot),
            slot_name(chunk, index)
        ),
        Instruction::AppendAssign(slot) => {
            format!("append_assign {slot}:{}", slot_name(chunk, slot))
        }
        Instruction::Print => "print".to_string(),
        Instruction::Submit => "submit".to_string(),
        Instruction::ProcessYield => "process_yield".to_string(),
        Instruction::ProcessWake => "process_wake".to_string(),
        Instruction::ProcessFinish => "process_finish".to_string(),
        Instruction::ProcessFail => "process_fail".to_string(),
        Instruction::Pop => "pop".to_string(),
        Instruction::BeginIter(slot) => format!("begin_iter {slot}:{}", slot_name(chunk, slot)),
        Instruction::BeginRangeIter { binding, argc } => {
            format!(
                "begin_range_iter {binding}:{} argc={argc}",
                slot_name(chunk, binding)
            )
        }
        Instruction::LoweredLoop(index) => lowered_loop_snapshot(chunk, index),
        Instruction::IterNext { jump_to } => format!("iter_next {jump_to}"),
        Instruction::EndIter => "end_iter".to_string(),
        Instruction::ResolveTypeRef(slot) => {
            format!("resolve_type_ref {slot}:{}", slot_name(chunk, slot))
        }
        Instruction::WrapTypeLiteral => "wrap_type_literal".to_string(),
    }
}

fn compact_json(value: &Value) -> String {
    serde_json::to_string(value).expect("value should serialize")
}

fn lowered_loop_snapshot(chunk: &Chunk, index: usize) -> String {
    let lowered_loop = &chunk.lowered_loops[index];
    let iterable = match &lowered_loop.iterable {
        LoopIterable::Range(args) => format!("range argc={}", args.len()),
        LoopIterable::Values(_) => "values".to_string(),
        LoopIterable::Keys(_) => "keys".to_string(),
    };
    format!(
        "lowered_loop #{index} {}:{} {iterable} ops={}",
        lowered_loop.binding,
        slot_name(chunk, lowered_loop.binding),
        lowered_loop.body.len()
    )
}

fn slot_name(chunk: &Chunk, index: usize) -> &str {
    chunk.slot_names[index].text.as_ref()
}

fn name_text(chunk: &Chunk, index: usize) -> &str {
    chunk.names[index].text.as_ref()
}

fn name(chunk: &Chunk, index: usize) -> &str {
    name_text(chunk, index)
}

fn keys_snapshot(chunk: &Chunk, index: usize) -> String {
    let keys = &chunk.key_lists[index];
    let names = keys
        .iter()
        .map(|key| name_text(chunk, *key))
        .collect::<Vec<_>>();
    format!("[{}]", names.join(", "))
}

fn assign_path_snapshot(chunk: &Chunk, index: usize) -> String {
    let path = &chunk.assign_paths[index];
    let mut rendered = String::new();
    for step in path.steps.iter() {
        match step {
            CompiledAssignPathStep::Field(field) => {
                write!(rendered, ".{}", name_text(chunk, *field)).unwrap();
            }
            CompiledAssignPathStep::Index => rendered.push_str("[dynamic]"),
        }
    }
    rendered
}

fn format_template_snapshot(chunk: &Chunk, index: usize) -> String {
    let template = &chunk.format_templates[index];
    let parts = template
        .parts
        .iter()
        .map(|part| match part {
            CompiledFormatPart::Literal(value) => format!("{value:?}"),
            CompiledFormatPart::Arg(index) => format!("arg{index}"),
        })
        .collect::<Vec<_>>()
        .join(" + ");
    format!(
        "argc={} min={} parts={parts}",
        template.argc, template.min_capacity
    )
}

fn intrinsic_snapshot(chunk: &Chunk, op: IntrinsicOp) -> String {
    let argc = op.argc();
    match op {
        IntrinsicOp::Len => format!("intrinsic len argc={argc}"),
        IntrinsicOp::Empty => format!("intrinsic empty argc={argc}"),
        IntrinsicOp::Keys => format!("intrinsic keys argc={argc}"),
        IntrinsicOp::Values => format!("intrinsic values argc={argc}"),
        IntrinsicOp::Contains => format!("intrinsic contains argc={argc}"),
        IntrinsicOp::Find(_) => format!("intrinsic find argc={argc}"),
        IntrinsicOp::GrepText => format!("intrinsic grep_text argc={argc}"),
        IntrinsicOp::StartsWith => format!("intrinsic starts_with argc={argc}"),
        IntrinsicOp::EndsWith => format!("intrinsic ends_with argc={argc}"),
        IntrinsicOp::Split => format!("intrinsic split argc={argc}"),
        IntrinsicOp::Join => format!("intrinsic join argc={argc}"),
        IntrinsicOp::Trim => format!("intrinsic trim argc={argc}"),
        IntrinsicOp::Slice => format!("intrinsic slice argc={argc}"),
        IntrinsicOp::ToString => format!("intrinsic to_string argc={argc}"),
        IntrinsicOp::ToInt => format!("intrinsic to_int argc={argc}"),
        IntrinsicOp::ToFloat => format!("intrinsic to_float argc={argc}"),
        IntrinsicOp::JsonParse => format!("intrinsic json_parse argc={argc}"),
        IntrinsicOp::Format(_) => format!("intrinsic format argc={argc}"),
        IntrinsicOp::Validate => format!("intrinsic validate argc={argc}"),
        IntrinsicOp::Range(_) => format!("intrinsic range argc={argc}"),
        IntrinsicOp::CeilDiv => format!("intrinsic ceil_div argc={argc}"),
        IntrinsicOp::FloorDiv => format!("intrinsic floor_div argc={argc}"),
        IntrinsicOp::Push => format!("intrinsic push argc={argc}"),
        IntrinsicOp::InvalidArity { name, .. } => {
            format!(
                "intrinsic invalid_arity({}) argc={argc}",
                name_text(chunk, name)
            )
        }
        IntrinsicOp::Unknown { name, .. } => {
            format!("intrinsic unknown({}) argc={argc}", name_text(chunk, name))
        }
        IntrinsicOp::ValidateCompiled(schema) => {
            format!("intrinsic validate_compiled schema#{schema}")
        }
        IntrinsicOp::PushAssign(slot) => {
            format!("intrinsic push_assign {slot}:{}", slot_name(chunk, slot))
        }
        IntrinsicOp::FormatCompiled(template) => {
            format!(
                "intrinsic format_compiled {}",
                format_template_snapshot(chunk, template)
            )
        }
        IntrinsicOp::FormatCompiledSlotNumber { template, slot } => format!(
            "intrinsic format_compiled_slot_number {} {slot}:{}",
            format_template_snapshot(chunk, template),
            slot_name(chunk, slot)
        ),
        IntrinsicOp::FormatCompiledSlotNumberBinary {
            template,
            slot,
            op,
            right,
        } => format!(
            "intrinsic format_compiled_slot_number_binary {} {slot}:{} {op:?} {right}",
            format_template_snapshot(chunk, template),
            slot_name(chunk, slot)
        ),
    }
}

#[tokio::test(flavor = "current_thread")]
async fn value_helpers_and_display_cover_all_variants() {
    let mut record = Record::default();
    record.insert("k".to_string(), Value::Number(1.0));

    assert_eq!(Value::Null.to_string(), "null");
    assert_eq!(Value::Bool(true).to_string(), "true");
    assert_eq!(Value::Number(1.5).to_string(), "1.5");
    assert_eq!(Value::String("x".to_string().into()).to_string(), "x");
    assert_eq!(
        Value::List(vec![Value::Bool(true)].into()).to_string(),
        "[true]"
    );
    assert_eq!(
        Value::Record(record.clone().into()).as_record().unwrap()["k"],
        Value::Number(1.0)
    );
    assert!(Value::String("x".to_string().into()).as_record().is_none());
    assert!(Value::Record(record.into()).to_string().contains("\"k\":1"));
}

#[tokio::test(flavor = "current_thread")]
async fn compiler_folds_constant_list_and_record_literals() {
    let program = crate::parse(
        r#"
        items = [{ label: "a", weight: 1 }, { label: "b", weight: 2 }]
        submit items
        "#,
    )
    .expect("program should parse");
    let compiled = compile_program(&program);

    assert!(
        compiled
            .chunk
            .code
            .iter()
            .any(|instruction| matches!(instruction, Instruction::PushConst(_)))
    );
    assert!(
        !compiled
            .chunk
            .code
            .iter()
            .any(|instruction| matches!(instruction, Instruction::BuildRecord(_)))
    );

    let mut state = State::new();
    let outcome = execute_compiled(&compiled, &mut state, &Host)
        .await
        .expect("program should run");
    let ExecutionOutcome::Finished(Value::List(items)) = outcome else {
        panic!("expected folded list result");
    };
    assert_eq!(items.len(), 2);
}

#[tokio::test(flavor = "current_thread")]
async fn compiler_propagates_safe_straight_line_constants() {
    let program = crate::parse(
        r#"
        items = [1, 2, 3]
        indexes = range(0, len(items))
        extended = push(indexes, len(items))
        submit extended
        "#,
    )
    .expect("program should parse");
    let compiled = compile_program(&program);

    assert!(
        !compiled.chunk.code.iter().any(|instruction| matches!(
            instruction,
            Instruction::Intrinsic(IntrinsicOp::Len | IntrinsicOp::Range(_) | IntrinsicOp::Push)
        )),
        "straight-line constant builtins should fold out of the runtime instruction stream"
    );

    let mut state = State::new();
    let outcome = execute_compiled(&compiled, &mut state, &Host)
        .await
        .expect("program should run");
    assert_eq!(
        outcome,
        ExecutionOutcome::Finished(Value::List(
            vec![
                Value::Number(0.0),
                Value::Number(1.0),
                Value::Number(2.0),
                Value::Number(3.0),
            ]
            .into()
        ))
    );
}

#[tokio::test(flavor = "current_thread")]
async fn lowered_loops_cover_range_list_keys_nested_control_and_mutation() {
    let source = r#"
        counts = {}
        items = ["a", "b", "a"]
        seen = []
        total = 0
        for i in range(0, 5) {
          if i == 1 {
            continue
          }
          if i == 4 {
            break
          }
          total = total + i
        }
        for item in items {
          counts[item] = counts[item] + 1
          seen = push(seen, format("{}:{}", item, counts[item]))
        }
        pairs = []
        for key in keys(counts) {
          for n in range(0, counts[key]) {
            pairs = pairs + [format("{}{}", key, n)]
          }
        }
        submit { total: total, counts: counts, seen: seen, pairs: pairs }
    "#;
    let compiled = compile_source(source).expect("program should compile");
    assert!(
        compiled
            .chunk
            .code
            .iter()
            .filter(|instruction| matches!(instruction, Instruction::LoweredLoop(_)))
            .count()
            >= 3,
        "eligible loops should lower into loop-local instructions"
    );

    let mut state = State::new();
    let outcome = execute_compiled(&compiled, &mut state, &Host)
        .await
        .expect("program should run");
    let ExecutionOutcome::Finished(Value::Record(record)) = outcome else {
        panic!("expected record result");
    };
    assert_eq!(record["total"], Value::Number(5.0));
    let counts = record["counts"].as_record().expect("counts record");
    assert_eq!(counts["a"], Value::Number(2.0));
    assert_eq!(counts["b"], Value::Number(1.0));
    let Value::List(seen) = &record["seen"] else {
        panic!("seen should be a list");
    };
    assert_eq!(seen.len(), 3);
    let Value::List(pairs) = &record["pairs"] else {
        panic!("pairs should be a list");
    };
    assert_eq!(pairs.len(), 3);
}

#[test]
fn effectful_loop_bodies_stay_on_generic_iterator_bytecode() {
    let program = crate::parse(
        r#"
        items = [1, 2]
        for item in items {
          print item
        }
        submit null
        "#,
    )
    .expect("program should parse");
    let compiled = compile_program(&program);
    assert!(
        compiled
            .chunk
            .code
            .iter()
            .any(|instruction| matches!(instruction, Instruction::BeginIter(_))),
        "effectful loops should use the generic iterator fallback"
    );
    assert!(
        !compiled
            .chunk
            .code
            .iter()
            .any(|instruction| matches!(instruction, Instruction::LoweredLoop(_))),
        "effectful loops must not lower"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn constant_propagation_does_not_cross_control_flow_boundaries() {
    let value = exec(
        r#"
        x = 1
        if false {
          x = 2
        }
        y = x + 1
        submit y
        "#,
    )
    .await
    .expect("program should succeed");

    assert_eq!(value, Value::Number(2.0));
}

#[tokio::test(flavor = "current_thread")]
async fn reusable_execution_scratch_preserves_results_across_runs() {
    let program = crate::parse(
        r#"
        items = [1, 2, 3]
        total = 0
        for item in items {
          total = total + item
        }
        submit total
        "#,
    )
    .expect("program should parse");
    let compiled = compile_program(&program);
    let mut scratch = ExecutionScratch::new();

    for _ in 0..3 {
        let mut state = State::new();
        let outcome = execute_compiled_with_scratch(&compiled, &mut state, &Host, &mut scratch)
            .await
            .expect("program should run");
        assert_eq!(outcome, ExecutionOutcome::Finished(Value::Number(6.0)));
    }
}

#[tokio::test(flavor = "current_thread")]
async fn continuation_and_undefined_variable_are_reported() {
    let outcome = exec_outcome("x = 1")
        .await
        .expect("missing submit should continue");
    assert_eq!(outcome, ExecutionOutcome::Continued);

    let value = exec("submit").await.expect("bare submit should succeed");
    assert_eq!(value, Value::Null);

    let err = exec("submit x")
        .await
        .expect_err("undefined variable should fail");
    assert_eq!(
        err,
        RuntimeError::UndefinedVariable {
            name: "x".to_string()
        }
    );
}

#[tokio::test(flavor = "current_thread")]
async fn condition_and_iteration_errors_are_reported() {
    let value = exec("if 1 { submit 1 } else { submit 2 }")
        .await
        .expect("numeric truthiness should be accepted");
    assert_eq!(value, Value::Number(1.0));

    let value = exec("if \"\" { submit 1 } else { submit 2 }")
        .await
        .expect("empty string should be falsy");
    assert_eq!(value, Value::Number(2.0));

    let err = exec("for x in 1 { submit x }")
        .await
        .expect_err("non-list iteration should fail");
    assert_eq!(err, RuntimeError::NonListIteration);
}

#[tokio::test(flavor = "current_thread")]
async fn stmt_call_and_tool_results_cover_success_and_error() {
    exec("await tools.echo({ value: 1 }) submit 1")
        .await
        .expect("statement module operation should succeed");
    let missing = exec("bad = await tools.missing({}) submit bad")
        .await
        .expect("missing module operation should be wrapped");
    assert_eq!(
        missing.as_record().expect("result should be a record")["ok"],
        Value::Bool(false)
    );

    let value = exec(
        "ok = await tools.echo({ value: 7 }) bad = await tools.err({}) submit { ok: ok, bad: bad }",
    )
    .await
    .expect("module operation program should succeed");
    let record = value.as_record().expect("expected record");
    assert_eq!(record["ok"].as_record().unwrap()["ok"], Value::Bool(true));
    assert_eq!(record["bad"].as_record().unwrap()["ok"], Value::Bool(false));
}

#[tokio::test(flavor = "current_thread")]
async fn result_unwrap_extracts_success_and_preserves_manual_handling() {
    let value = exec("submit (await tools.echo({ value: 7 })?)")
        .await
        .expect("unwrap should succeed");
    assert_eq!(value, Value::Number(7.0));

    let value = exec(
        r#"
        result = await tools.err({})
        submit result.ok ? result.error : "unexpected"
        "#,
    )
    .await
    .expect("manual wrapper handling should still work");
    assert_eq!(value, Value::String("unexpected".into()));
}

#[tokio::test(flavor = "current_thread")]
async fn direct_module_operation_unwrap_skips_observable_wrapper() {
    let compiled =
        compile_source("submit (await tools.echo({ value: 7 })?)").expect("program should compile");
    assert!(
        compiled
            .chunk
            .code
            .iter()
            .any(|instruction| matches!(instruction, Instruction::ResourceCallUnwrap { .. }))
    );
    assert!(
        !compiled
            .chunk
            .code
            .iter()
            .any(|instruction| matches!(instruction, Instruction::ResultUnwrap))
    );

    let mut state = State::new();
    let outcome = execute_compiled(&compiled, &mut state, &Host)
        .await
        .expect("program should run");
    assert_eq!(outcome, ExecutionOutcome::Finished(Value::Number(7.0)));

    let err = exec("submit (await tools.err({})?)")
        .await
        .expect_err("failed unwrap should abort");
    assert_eq!(
        err,
        RuntimeError::ValueError {
            message: "`?` unwrapped failed module operation: boom".to_string(),
        }
    );
}

#[tokio::test(flavor = "current_thread")]
async fn result_unwrap_reports_failed_and_malformed_wrappers() {
    let err = exec("submit (await tools.err({})?)")
        .await
        .expect_err("failed module operation unwrap should abort");
    assert_eq!(
        err,
        RuntimeError::ValueError {
            message: "`?` unwrapped failed module operation: boom".to_string(),
        }
    );

    let err = exec("submit 1?")
        .await
        .expect_err("non-wrapper should fail");
    assert_eq!(
        err,
        RuntimeError::TypeError {
            message: "`?` expected a tool result wrapper, got number".to_string(),
        }
    );

    let err = exec("submit { ok: true }?")
        .await
        .expect_err("missing value should fail");
    assert_eq!(
        err,
        RuntimeError::TypeError {
            message: "`?` found a successful tool result wrapper missing `value`".to_string(),
        }
    );
}

#[tokio::test(flavor = "current_thread")]
async fn field_index_unary_and_boolean_paths_are_covered() {
    let value = exec(
        r#"
        rec = { nested: { name: "lash" } }
        xs = ["a", "b"]
        ok = false and missing
        alt = true or missing
        submit [rec.nested.name, xs[1], "abc"[2], -1, not false, !false, ok, alt]
        "#,
    )
    .await
    .expect("program should succeed");

    assert_eq!(
        value,
        Value::List(
            vec![
                Value::String("lash".to_string().into()),
                Value::String("b".to_string().into()),
                Value::String("c".to_string().into()),
                Value::Number(-1.0),
                Value::Bool(true),
                Value::Bool(true),
                Value::Bool(false),
                Value::Bool(true),
            ]
            .into()
        )
    );

    let value = exec("submit true and false")
        .await
        .expect("and path should succeed");
    assert_eq!(value, Value::Bool(false));

    let value = exec("submit false or true")
        .await
        .expect("or path should succeed");
    assert_eq!(value, Value::Bool(true));
}

#[tokio::test(flavor = "current_thread")]
async fn field_index_and_type_errors_are_covered() {
    let err = exec("n = 1 submit n.name")
        .await
        .expect_err("field access should fail");
    assert!(matches!(err, RuntimeError::TypeError { .. }));

    let value = exec("rec = {} submit rec.name")
        .await
        .expect("missing field should yield null");
    assert_eq!(value, Value::Null);

    let err = exec("submit 1[0]")
        .await
        .expect_err("bad index target should fail");
    assert!(matches!(err, RuntimeError::TypeError { .. }));

    let value = exec("submit [1][2]")
        .await
        .expect("list oob should yield null");
    assert_eq!(value, Value::Null);

    let value = exec("submit \"a\"[2]")
        .await
        .expect("string oob should yield null");
    assert_eq!(value, Value::Null);

    let err = exec("submit [1][1.5]")
        .await
        .expect_err("fractional index should fail");
    assert!(matches!(err, RuntimeError::TypeError { .. }));

    let value = exec("submit [1][-1]")
        .await
        .expect("negative index should resolve from the end");
    assert_eq!(value, Value::Number(1.0));

    let value = exec("submit not 1")
        .await
        .expect("not should use truthiness");
    assert_eq!(value, Value::Bool(false));

    let value = exec("submit not 0").await.expect("zero should be falsy");
    assert_eq!(value, Value::Bool(true));

    let value = exec("rec = { ok: false } submit len(rec.value.items)")
        .await
        .expect("null chain should work");
    assert_eq!(value, Value::Number(0.0));
}

#[tokio::test(flavor = "current_thread")]
async fn arithmetic_and_compare_errors_are_covered() {
    assert_eq!(
        exec("submit 7 - 2").await.expect("subtract should succeed"),
        Value::Number(5.0)
    );
    assert_eq!(
        exec("submit 3 * 4").await.expect("multiply should succeed"),
        Value::Number(12.0)
    );
    assert_eq!(
        exec("submit 8 / 2").await.expect("divide should succeed"),
        Value::Number(4.0)
    );
    assert_eq!(
        exec("submit 8 % 3").await.expect("modulo should succeed"),
        Value::Number(2.0)
    );
    assert_eq!(
        exec("submit 1 != 2")
            .await
            .expect("not equal should succeed"),
        Value::Bool(true)
    );
    assert_eq!(
        exec("submit 1 <= 2")
            .await
            .expect("less-equal should succeed"),
        Value::Bool(true)
    );
    assert_eq!(
        exec("submit 2 > 1").await.expect("greater should succeed"),
        Value::Bool(true)
    );
    assert_eq!(
        exec("submit 2 >= 1")
            .await
            .expect("greater-equal should succeed"),
        Value::Bool(true)
    );

    let value = exec("submit [1,2] + [3]")
        .await
        .expect("list concat should succeed");
    assert_eq!(
        value,
        Value::List(vec![Value::Number(1.0), Value::Number(2.0), Value::Number(3.0)].into())
    );

    let value = exec("submit \"a\" + \"b\"")
        .await
        .expect("string add should succeed");
    assert_eq!(value, Value::String("ab".to_string().into()));

    let value = exec("submit \"a\" + 1")
        .await
        .expect("string coercion should succeed");
    assert_eq!(value, Value::String("a1".to_string().into()));

    let value = exec("submit 1 + \"b\"")
        .await
        .expect("string coercion should succeed");
    assert_eq!(value, Value::String("1b".to_string().into()));

    let value = exec("submit 1 + true")
        .await
        .expect("bool should coerce for addition");
    assert_eq!(value, Value::Number(2.0));

    let value = exec("submit null + 2")
        .await
        .expect("null should coerce for addition");
    assert_eq!(value, Value::Number(2.0));

    let value = exec("submit \"2\" * 3")
        .await
        .expect("numeric strings should coerce");
    assert_eq!(value, Value::Number(6.0));

    let value = exec("submit \"2\" < 10")
        .await
        .expect("numeric strings should compare");
    assert_eq!(value, Value::Bool(true));

    let err = exec("submit {} + 1")
        .await
        .expect_err("records should still fail arithmetic");
    assert!(matches!(err, RuntimeError::TypeError { .. }));
}

#[tokio::test(flavor = "current_thread")]
async fn builtin_success_matrix_is_covered() {
    let value = exec(
        r#"
        rec = { a: 1, b: 2 }
        base = [1, 2]
        submit {
          len_s: len("ab"),
          len_l: len([1,2,3]),
          len_r: len(rec),
          len_n: len(null),
          empty_n: empty(null),
          empty_s: empty(""),
          empty_l: empty([]),
          empty_r: empty({}),
          keys_n: keys(null),
          values_n: values(null),
          keys: keys(rec),
          values: values(rec),
          contains_s: contains("abc", "b"),
          contains_num: contains("123", 2),
          contains_l: contains([1,2,3], 2),
          contains_r: contains({ foo: 1, bar: 2 }, "foo"),
          contains_n: contains(null, 2),
          find_hit: find("alpha beta", "beta"),
          find_from: find("banana", "na", 3),
          find_missing: find("alpha", "z"),
          starts: starts_with("lash", "la"),
          starts_num: starts_with(123, 12),
          ends: ends_with("lash", "sh"),
          split: split(101, 0),
          join: join(["a",2,true], "-"),
          trim: trim(101),
          slice_s: slice("abcd", 1, 3),
          slice_end_s: slice("abcd", 2, null),
          slice_back_s: slice("abcd", 3, 1),
          slice_from_start_s: slice("abcd", null, 2),
          slice_l: slice([1,2,3,4], 1, 3),
          slice_end_l: slice([1,2,3,4], 2, null),
          slice_back_l: slice([1,2,3,4], 3, 1),
          slice_from_start_l: slice([1,2,3,4], null, 2),
          to_s: to_string({ a: 1 }),
          to_i_n: to_int(3.9),
          to_i_s: to_int("4"),
          to_i_b: to_int(true),
          to_f_n: to_float(1),
          to_f_s: to_float("2.5"),
          to_f_nl: to_float(null),
          fmt: format("x={},y={}", 1, true),
          range_end: range(3),
          range_pair: range(-2, 2),
          range_step: range(0, 5, 2),
          range_step_down: range(5, 0, -2),
          range_empty: range(2, 2),
          ceil_div: ceil_div(10, 3),
          floor_div: floor_div(-10, 3),
          pushed: push(base, 3),
          base_after_push: base,
          valid: validate(
            {
              name: "pkg",
              version: "1.0.0",
              deps: [{ name: "dep", optional: true }],
              extra: "preserved"
            },
            Type {
              name: str,
              version: str,
              deps: list[Type { name: str, optional: bool? }]
            }
          )
        }
        "#,
    )
    .await
    .expect("builtins should succeed");

    let record = value.as_record().expect("expected record");
    assert_eq!(record["len_s"], Value::Number(2.0));
    assert_eq!(record["len_n"], Value::Number(0.0));
    assert_eq!(record["contains_num"], Value::Bool(true));
    assert_eq!(record["contains_l"], Value::Bool(true));
    assert_eq!(record["contains_r"], Value::Bool(true));
    assert_eq!(record["contains_n"], Value::Bool(false));
    assert_eq!(record["find_hit"], Value::Number(6.0));
    assert_eq!(record["find_from"], Value::Number(4.0));
    assert_eq!(record["find_missing"], Value::Null);
    assert_eq!(record["keys_n"], Value::List(Vec::new().into()));
    assert_eq!(record["values_n"], Value::List(Vec::new().into()));
    assert_eq!(record["starts_num"], Value::Bool(true));
    assert_eq!(
        record["split"],
        Value::List(
            vec![
                Value::String("1".to_string().into()),
                Value::String("1".to_string().into())
            ]
            .into()
        )
    );
    assert_eq!(record["join"], Value::String("a-2-true".to_string().into()));
    assert_eq!(record["trim"], Value::String("101".to_string().into()));
    assert_eq!(record["slice_s"], Value::String("bc".to_string().into()));
    assert_eq!(
        record["slice_end_s"],
        Value::String("cd".to_string().into())
    );
    assert_eq!(record["slice_back_s"], Value::String(String::new().into()));
    assert_eq!(
        record["slice_from_start_s"],
        Value::String("ab".to_string().into())
    );
    assert_eq!(
        record["slice_end_l"],
        Value::List(vec![Value::Number(3.0), Value::Number(4.0)].into())
    );
    assert_eq!(record["slice_back_l"], Value::List(Vec::new().into()));
    assert_eq!(
        record["slice_from_start_l"],
        Value::List(vec![Value::Number(1.0), Value::Number(2.0)].into())
    );
    assert_eq!(record["to_i_n"], Value::Number(3.0));
    assert_eq!(record["to_i_b"], Value::Number(1.0));
    assert_eq!(record["to_f_s"], Value::Number(2.5));
    assert_eq!(record["to_f_nl"], Value::Number(0.0));
    assert_eq!(
        record["fmt"],
        Value::String("x=1,y=true".to_string().into())
    );
    assert_eq!(
        record["range_end"],
        Value::List(vec![Value::Number(0.0), Value::Number(1.0), Value::Number(2.0)].into())
    );
    assert_eq!(
        record["range_pair"],
        Value::List(
            vec![
                Value::Number(-2.0),
                Value::Number(-1.0),
                Value::Number(0.0),
                Value::Number(1.0)
            ]
            .into()
        )
    );
    assert_eq!(record["range_empty"], Value::List(Vec::new().into()));
    assert_eq!(
        record["range_step"],
        Value::List(vec![Value::Number(0.0), Value::Number(2.0), Value::Number(4.0)].into())
    );
    assert_eq!(
        record["range_step_down"],
        Value::List(vec![Value::Number(5.0), Value::Number(3.0), Value::Number(1.0)].into())
    );
    assert_eq!(record["ceil_div"], Value::Number(4.0));
    assert_eq!(record["floor_div"], Value::Number(-4.0));
    assert_eq!(
        record["pushed"],
        Value::List(vec![Value::Number(1.0), Value::Number(2.0), Value::Number(3.0)].into())
    );
    assert_eq!(
        record["base_after_push"],
        Value::List(vec![Value::Number(1.0), Value::Number(2.0)].into())
    );
    let valid = record["valid"].as_record().expect("validated record");
    assert_eq!(valid["name"], Value::String("pkg".to_string().into()));
    assert_eq!(
        valid["extra"],
        Value::String("preserved".to_string().into())
    );
}

#[tokio::test(flavor = "current_thread")]
async fn grep_text_returns_documented_line_records() {
    let value = exec(
        r#"
        matches = grep_text("alpha\nbeta match\r\ngamma match\n", "match")
        submit matches
        "#,
    )
    .await
    .expect("grep_text should succeed");

    let Value::List(matches) = value else {
        panic!("expected list");
    };
    assert_eq!(matches.len(), 2);
    let first = matches[0].as_record().expect("first match record");
    assert_eq!(first["line"], Value::Number(2.0));
    assert_eq!(first["text"], Value::String("beta match".into()));
    assert_eq!(first["match"], Value::String("match".into()));
    assert_eq!(first["start"], Value::Number(5.0));
    assert_eq!(first["end"], Value::Number(10.0));
}

#[tokio::test(flavor = "current_thread")]
async fn find_uses_character_offsets() {
    let value = exec(
        r#"
        submit {
          first: find("éclair café", "café"),
          from_end: find("abc", "", 3),
          beyond_end: find("abc", "a", 4)
        }
        "#,
    )
    .await
    .expect("find should succeed");

    let record = value.as_record().expect("record");
    assert_eq!(record["first"], Value::Number(7.0));
    assert_eq!(record["from_end"], Value::Number(3.0));
    assert_eq!(record["beyond_end"], Value::Null);
}

#[tokio::test(flavor = "current_thread")]
async fn builtin_error_matrix_is_covered() {
    let cases = [
        ("submit len(true)", "len"),
        ("submit empty(true)", "empty"),
        ("submit keys([])", "keys"),
        ("submit values([])", "values"),
        ("submit contains(1, 2)", "contains"),
        ("submit find(\"a\")", "find"),
        ("submit find(\"a\", \"a\", -1)", "find"),
        ("submit grep_text({}, \"a\")", "grep_text"),
        ("submit grep_text(\"a\", \"\")", "grep_text"),
        ("submit starts_with({}, \"a\")", "starts_with"),
        ("submit ends_with({}, \"a\")", "ends_with"),
        ("submit split({}, \",\")", "split"),
        ("submit join(1, \",\")", "join"),
        ("submit trim({})", "trim"),
        ("submit slice(1, 0, 1)", "slice"),
        ("submit to_int({})", "to_int"),
        ("submit to_int(\"x\")", "to_int"),
        ("submit to_float({})", "to_float"),
        ("submit to_float(\"x\")", "to_float"),
        ("submit json_parse(\"{\")", "json_parse"),
        ("submit format()", "format"),
        ("submit format({})", "format"),
        ("submit format(\"{1}\", \"x\")", "format"),
        ("submit format(\"{}\", \"x\", \"y\")", "format"),
        ("submit format(\"{} {1}\", \"x\", \"y\")", "format"),
        ("submit format(\"{x}\")", "format"),
        ("submit format(\"{\")", "format"),
        ("submit format(\"}\")", "format"),
        (
            "submit validate({ name: \"pkg\" }, { type: \"object\" })",
            "validate",
        ),
        ("submit range()", "range"),
        ("submit range(1, 2, 3, 4)", "range"),
        ("submit range(\"3\")", "range"),
        ("submit range(1.5)", "range"),
        ("submit range(0, 5, 0)", "range"),
        ("submit range(0, 1000001)", "range"),
        ("submit range(1000001, 0, -1)", "range"),
        ("submit ceil_div()", "ceil_div"),
        ("submit ceil_div(1.5, 1)", "ceil_div"),
        ("submit ceil_div(1, 0)", "ceil_div"),
        ("submit floor_div(\"1\", 1)", "floor_div"),
        ("submit push(1, 2)", "push"),
        ("submit push([1])", "push"),
        ("submit no_such_builtin()", "no_such_builtin"),
    ];

    for (source, _) in cases {
        let err = exec(source).await.expect_err("builtin should fail");
        assert!(matches!(
            err,
            RuntimeError::TypeError { .. }
                | RuntimeError::ValueError { .. }
                | RuntimeError::UnknownBuiltin { .. }
        ));
    }

    let err = exec("submit len()")
        .await
        .expect_err("arity error should fail");
    assert!(matches!(err, RuntimeError::TypeError { .. }));
}

#[tokio::test(flavor = "current_thread")]
async fn validate_reports_precise_shape_errors() {
    let cases = [
        (
            "submit validate({ name: \"pkg\" }, Type { name: str, version: str })",
            "validation failed: $: missing required field `version`",
        ),
        (
            r#"submit validate({ packages: [{ name: "pkg", version: 1 }] }, Type { packages: list[Type { name: str, version: str }] })"#,
            "validation failed: $.packages[0].version: expected string, got number",
        ),
        (
            r#"submit validate({ status: "maybe" }, Type { status: enum["ok", "err"] })"#,
            "validation failed: $.status: expected one of [ok, err], got maybe",
        ),
        (
            r#"submit validate({ count: 1.5 }, Type { count: int })"#,
            "validation failed: $.count: expected integer, got number",
        ),
    ];

    for (source, expected) in cases {
        let err = exec(source)
            .await
            .expect_err("validate should reject bad value");
        assert_eq!(
            err,
            RuntimeError::ValueError {
                message: expected.to_string()
            }
        );
    }

    let err = exec("submit validate({ name: \"pkg\" }, { type: \"object\" })")
        .await
        .expect_err("raw schema records should be rejected");
    assert_eq!(
        err,
        RuntimeError::TypeError {
            message: "`validate` requires a Type literal as the second argument".to_string()
        }
    );
}

#[tokio::test(flavor = "current_thread")]
async fn validate_union_accepts_any_variant() {
    // `str | null` must accept both a string and a null.
    let out = exec(r#"submit validate({ email: "a@b" }, Type { email: str | null })"#)
        .await
        .expect("string-branch validate should succeed");
    assert_eq!(
        out,
        Value::Record(Arc::new({
            let mut rec = record_with_capacity(1);
            rec.insert("email".into(), Value::String("a@b".into()));
            rec
        }))
    );

    let out = exec(r#"submit validate({ email: null }, Type { email: str | null })"#)
        .await
        .expect("null-branch validate should succeed");
    let Value::Record(rec) = &out else {
        panic!("expected record");
    };
    assert!(matches!(rec.get("email"), Some(Value::Null)));
}

#[tokio::test(flavor = "current_thread")]
async fn validate_union_rejects_value_matching_no_variant() {
    let err = exec(r#"submit validate({ email: 42 }, Type { email: str | null })"#)
        .await
        .expect_err("number should not match str | null");
    let RuntimeError::ValueError { message } = err else {
        panic!("expected ValueError");
    };
    assert!(
        message.contains("$.email"),
        "error should point at the failing field: {message}",
    );
}

#[tokio::test(flavor = "current_thread")]
async fn validate_static_and_dynamic_type_paths_share_error_text() {
    let static_err = exec(r#"submit validate({ email: 42 }, Type { email: str | null })"#)
        .await
        .expect_err("static Type literal should reject number");
    let dynamic_err = exec(
        r#"
Schema = await tools.echo({ value: Type { email: str | null } })?
submit validate({ email: 42 }, Schema)
"#,
    )
    .await
    .expect_err("runtime Type value should reject number");

    assert_eq!(static_err, dynamic_err);
    assert_eq!(
        dynamic_err,
        RuntimeError::ValueError {
            message: "validation failed: $.email: expected one of [string, null], got number"
                .to_string()
        }
    );
}

#[tokio::test(flavor = "current_thread")]
async fn validate_object_type_accepts_image_descriptors() {
    let program = crate::parse(
        r#"
submit validate(img, Type {
  type: str,
  id: str,
  label: str,
  size: int,
  width: int | null,
  height: int | null
})
"#,
    )
    .expect("program should parse");
    let mut state = State::new();
    state.globals.insert_str(
        "img",
        Value::Image(ImageValue::new("img-1", "chart.png", 1234, Some(640), None)),
    );

    let outcome = execute_program(&program, &mut state, &Host)
        .await
        .expect("image descriptor validation should succeed");
    assert_eq!(
        outcome,
        ExecutionOutcome::Finished(Value::Image(ImageValue::new(
            "img-1",
            "chart.png",
            1234,
            Some(640),
            None
        )))
    );
}

#[tokio::test(flavor = "current_thread")]
async fn helper_functions_are_covered_directly() {
    assert!(expect_arg_count("x", &[Value::Null], 1).is_ok());
    assert!(expect_arg_count("x", &[], 1).is_err());
    assert_eq!(as_number(&Value::Number(1.0)).expect("number"), 1.0);
    assert_eq!(as_number(&Value::Bool(true)).expect("bool"), 1.0);
    assert_eq!(as_number(&Value::Null).expect("null"), 0.0);
    assert_eq!(
        as_number(&Value::String("2.5".to_string().into())).expect("numeric"),
        2.5
    );
    assert_eq!(
        coerce_string(&Value::String("x".to_string().into())).expect("string"),
        "x"
    );
    assert_eq!(coerce_string(&Value::Bool(true)).expect("bool"), "true");
    assert_eq!(as_offset(&Value::Number(-1.0)).expect("offset"), -1);
    assert_eq!(as_slice_bound(&Value::Null).expect("null bound"), None);
    assert_eq!(
        as_slice_bound(&Value::Number(2.0)).expect("numeric bound"),
        Some(2)
    );
    assert_eq!(
        as_slice_bound(&Value::Number(-2.0)).expect("negative numeric bound"),
        Some(-2)
    );
    assert_eq!(slice_string("héllo", Some(1), Some(4)), "éll");
    assert_eq!(slice_string("abcdef", Some(-2), None), "ef");
    assert_eq!(slice_string("abcdef", None, Some(-1)), "abcde");
    assert_eq!(slice_string("abcdef", Some(-5), Some(-2)), "bcd");
    assert_eq!(slice_string("abc", Some(1), None), "bc");
    assert_eq!(slice_string("abc", Some(3), Some(1)), "");
    assert_eq!(clamp_slice_bounds(Some(1), Some(3), 4), Some((1, 3)));
    assert_eq!(clamp_slice_bounds(Some(1), None, 4), Some((1, 4)));
    assert_eq!(clamp_slice_bounds(None, Some(2), 4), Some((0, 2)));
    assert_eq!(clamp_slice_bounds(Some(-2), None, 4), Some((2, 4)));
    assert_eq!(clamp_slice_bounds(None, Some(-1), 4), Some((0, 3)));
    assert_eq!(clamp_slice_bounds(Some(-10), Some(10), 4), Some((0, 4)));
    assert_eq!(clamp_slice_bounds(Some(3), Some(1), 4), None);
    assert_eq!(
        resolve_index(&Value::Number(-1.0), 3).expect("resolved"),
        Some(2)
    );
    assert_eq!(
        resolve_index(&Value::Number(-4.0), 3).expect("resolved"),
        None
    );

    assert_eq!(
        compare_numbers(Value::Number(1.0), Value::Number(2.0), |a, b| a < b).expect("compare"),
        Value::Bool(true)
    );
    assert_eq!(
        compare_ordered(
            Value::String("abc".to_string().into()),
            Value::String("def".to_string().into()),
            |a, b| a < b,
            |a, b| a < b,
        )
        .expect("string compare"),
        Value::Bool(true)
    );
    assert_eq!(
        add_values(Value::Number(1.0), Value::Number(2.0)).expect("add"),
        Value::Number(3.0)
    );
    assert_eq!(
        add_values(Value::String("a".to_string().into()), Value::Bool(true)).expect("concat"),
        Value::String("atrue".to_string().into())
    );
    assert_eq!(
        add_values(Value::Bool(true), Value::Number(2.0)).expect("numeric coercion"),
        Value::Number(3.0)
    );
    assert_eq!(
        success(Value::Number(1.0)).as_record().unwrap()["ok"],
        Value::Bool(true)
    );
    assert_eq!(
        error_value("x".to_string()).as_record().unwrap()["error"],
        Value::String("x".to_string().into())
    );
    assert_eq!(stringify_value(&Value::Null).expect("stringify"), "null");
    assert_eq!(
        stringify_value(&Value::Number(1.0)).expect("stringify"),
        "1"
    );
    assert_eq!(
        stringify_value(&Value::List(
            vec![Value::Number(1.0), Value::Number(2.0)].into()
        ))
        .expect("stringify"),
        "[1,2]"
    );
    let mut appended = String::from("prefix:");
    append_stringified_value(&mut appended, &Value::Bool(true)).expect("append stringify");
    assert_eq!(appended, "prefix:true");
    assert_eq!(
        apply_format("a{}b", &[Value::Number(1.0)]).expect("format"),
        "a1b"
    );
    assert_eq!(
        apply_format(
            "b={1} a={0}",
            &[Value::String("x".into()), Value::String("y".into())]
        )
        .expect("indexed format"),
        "b=y a=x"
    );
    assert_eq!(
        apply_format("{{{}}}", &[Value::Number(1.0)]).expect("escaped braces"),
        "{1}"
    );
    assert_eq!(
        apply_format("{999999999999999999999999999999999999}", &[])
            .expect_err("overflow slot should fail"),
        RuntimeError::ValueError {
            message: "bad format slot `999999999999999999999999999999999999`".to_string()
        }
    );
    assert_eq!(
        apply_format("{x}", &[]).expect_err("invalid placeholder should fail"),
        RuntimeError::ValueError {
            message: "invalid format placeholder".to_string()
        }
    );
    assert_eq!(
        apply_format(
            "{} {1}",
            &[Value::String("x".into()), Value::String("y".into())]
        )
        .expect_err("mixed placeholder styles should fail"),
        RuntimeError::ValueError {
            message: "can't mix `{}` and indexed format placeholders".to_string()
        }
    );
    assert_eq!(
        apply_format("{", &[]).expect_err("unmatched open brace should fail"),
        RuntimeError::ValueError {
            message: "unmatched `{` in format string".to_string()
        }
    );
    assert_eq!(
        apply_format("}", &[]).expect_err("unmatched close brace should fail"),
        RuntimeError::ValueError {
            message: "unmatched `}` in format string".to_string()
        }
    );
    assert_eq!(
        apply_format("plain", &[Value::Number(1.0)]).expect_err("unused arg should fail"),
        RuntimeError::ValueError {
            message: "format argument `0` is unused".to_string()
        }
    );
    assert_eq!(
        value_type_name(&Value::Record(Record::default().into())),
        "record"
    );
    assert_eq!(
        RuntimeError::UndefinedVariable {
            name: "x".to_string()
        }
        .to_string(),
        "unknown name `x`"
    );
    assert_eq!(
        RuntimeError::TypeError {
            message: "can't index record".to_string()
        }
        .to_string(),
        "can't index record"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn json_helpers_cover_special_paths() {
    let json = to_json(&Value::Number(f64::NAN));
    assert_eq!(json, serde_json::Value::Null);
    assert_eq!(to_json(&Value::Null), serde_json::Value::Null);
    assert_eq!(to_json(&Value::Bool(true)), serde_json::Value::Bool(true));
    assert_eq!(
        to_json(&Value::String("x".to_string().into())),
        serde_json::Value::String("x".to_string())
    );
    assert_eq!(
        to_json(&Value::List(vec![Value::Number(1.0)].into())),
        serde_json::json!([1])
    );
    assert_eq!(
        to_json(&Value::Record({
            let mut record = Record::default();
            record.insert("a".to_string(), Value::Number(1.0));
            record.into()
        })),
        serde_json::json!({"a": 1})
    );

    let value = from_json(serde_json::json!({
        "a": [1, true, null, "x"]
    }));
    let record = value.as_record().expect("expected record");
    assert!(matches!(record["a"], Value::List(_)));
}

fn test_image() -> Value {
    Value::Image(ImageValue::new(
        "img-1",
        "chart.png",
        1234,
        Some(640),
        Some(480),
    ))
}

async fn exec_with_global(name: &str, value: Value, source: &str) -> Result<Value, RuntimeError> {
    let program = crate::parse(source).expect("program should parse");
    let mut state = State::new();
    state.globals.insert(name.to_string(), value);
    match execute_program(&program, &mut state, &Host).await? {
        ExecutionOutcome::Finished(value) => Ok(value),
        ExecutionOutcome::Continued => panic!("expected `submit` in test program"),
        ExecutionOutcome::Failed(value) => panic!("unexpected process failure: {value}"),
    }
}

struct TestProjectedValue {
    values: Vec<Value>,
    get_count: AtomicUsize,
    materialize_count: AtomicUsize,
    render_count: AtomicUsize,
}

impl TestProjectedValue {
    fn new(values: Vec<Value>) -> Arc<Self> {
        Arc::new(Self {
            values,
            get_count: AtomicUsize::new(0),
            materialize_count: AtomicUsize::new(0),
            render_count: AtomicUsize::new(0),
        })
    }
}

#[derive(Default)]
struct SnapshotGuardProjectedValue {
    materialize_count: AtomicUsize,
    render_count: AtomicUsize,
}

struct SearchProjectedText {
    text: Arc<str>,
    slice_count: AtomicUsize,
    materialize_count: AtomicUsize,
    render_count: AtomicUsize,
    slices: Mutex<Vec<(Option<isize>, Option<isize>)>>,
}

impl SearchProjectedText {
    fn new(text: impl Into<Arc<str>>) -> Arc<Self> {
        Arc::new(Self {
            text: text.into(),
            slice_count: AtomicUsize::new(0),
            materialize_count: AtomicUsize::new(0),
            render_count: AtomicUsize::new(0),
            slices: Mutex::new(Vec::new()),
        })
    }

    fn slices(&self) -> Vec<(Option<isize>, Option<isize>)> {
        self.slices.lock().expect("slices lock").clone()
    }
}

impl ProjectedHostValue for SnapshotGuardProjectedValue {
    fn type_name(&self) -> &str {
        "string"
    }

    fn read_one(
        &self,
        request: ProjectedReadRequest,
    ) -> ProjectedFuture<'_, ProjectedReadResponse> {
        Box::pin(async move {
            match request {
                ProjectedReadRequest::Render => {
                    self.render_count.fetch_add(1, Ordering::SeqCst);
                    ProjectedReadResponse::Text("rendered full text".to_string())
                }
                ProjectedReadRequest::Materialize => {
                    self.materialize_count.fetch_add(1, Ordering::SeqCst);
                    ProjectedReadResponse::Value(Value::String("materialized full text".into()))
                }
                _ => ProjectedReadResponse::Missing,
            }
        })
    }
}

impl ProjectedHostValue for SearchProjectedText {
    fn type_name(&self) -> &str {
        "string"
    }

    fn read_one(
        &self,
        request: ProjectedReadRequest,
    ) -> ProjectedFuture<'_, ProjectedReadResponse> {
        Box::pin(async move {
            match request {
                ProjectedReadRequest::Len => ProjectedReadResponse::Len(self.text.chars().count()),
                ProjectedReadRequest::Slice { start, end } => {
                    self.slice_count.fetch_add(1, Ordering::SeqCst);
                    self.slices.lock().expect("slices lock").push((start, end));
                    ProjectedReadResponse::Value(Value::String(
                        slice_string(&self.text, start, end).into(),
                    ))
                }
                ProjectedReadRequest::Render => {
                    self.render_count.fetch_add(1, Ordering::SeqCst);
                    ProjectedReadResponse::Text(self.text.to_string())
                }
                ProjectedReadRequest::Materialize => {
                    self.materialize_count.fetch_add(1, Ordering::SeqCst);
                    ProjectedReadResponse::Value(Value::String(self.text.as_ref().into()))
                }
                _ => ProjectedReadResponse::Missing,
            }
        })
    }
}

impl ProjectedHostValue for TestProjectedValue {
    fn type_name(&self) -> &str {
        "list"
    }

    fn read_one(
        &self,
        request: ProjectedReadRequest,
    ) -> ProjectedFuture<'_, ProjectedReadResponse> {
        Box::pin(async move {
            let ProjectedReadRequest::Index(index) = request else {
                return match request {
                    ProjectedReadRequest::Len => ProjectedReadResponse::Len(self.values.len()),
                    ProjectedReadRequest::Render => {
                        self.render_count.fetch_add(1, Ordering::SeqCst);
                        ProjectedReadResponse::Text("<projected list>".to_string())
                    }
                    ProjectedReadRequest::Materialize => {
                        self.materialize_count.fetch_add(1, Ordering::SeqCst);
                        ProjectedReadResponse::Value(Value::List(self.values.clone().into()))
                    }
                    _ => ProjectedReadResponse::Missing,
                };
            };
            let Value::Number(index) = index else {
                return ProjectedReadResponse::Missing;
            };
            if !index.is_finite() || index.fract() != 0.0 {
                return ProjectedReadResponse::Missing;
            }
            let len = self.values.len() as isize;
            let index = index as isize;
            let index = if index < 0 { len + index } else { index };
            if index < 0 || index >= len {
                return ProjectedReadResponse::Missing;
            }
            self.get_count.fetch_add(1, Ordering::SeqCst);
            self.values
                .get(index as usize)
                .cloned()
                .map(ProjectedReadResponse::Value)
                .unwrap_or(ProjectedReadResponse::Missing)
        })
    }
}

fn projected_list_bindings(name: &str, list: Arc<TestProjectedValue>) -> ProjectedBindings {
    let mut projected = ProjectedBindings::new();
    projected.insert(name, ProjectedValue::custom(name.to_string(), list));
    projected
}

struct ProjectedFixture {
    value: Value,
    materialize_count: AtomicUsize,
}

impl ProjectedFixture {
    fn new(value: Value) -> Arc<Self> {
        Arc::new(Self {
            value,
            materialize_count: AtomicUsize::new(0),
        })
    }
}

fn projected_response_from_value(
    value: &Value,
    request: ProjectedReadRequest,
) -> ProjectedReadResponse {
    match request {
        ProjectedReadRequest::Len => value_len(value)
            .map(ProjectedReadResponse::Len)
            .unwrap_or(ProjectedReadResponse::Missing),
        ProjectedReadRequest::Empty => value_len(value)
            .map(|len| ProjectedReadResponse::Bool(len == 0))
            .unwrap_or(ProjectedReadResponse::Missing),
        ProjectedReadRequest::Truthy => ProjectedReadResponse::Bool(is_truthy(value)),
        ProjectedReadRequest::Field(field) => {
            let field = Name {
                symbol: intern_symbol(field.as_ref()),
                text: field,
            };
            read_field_ref_direct(value, &field)
                .map(ProjectedReadResponse::Value)
                .unwrap_or(ProjectedReadResponse::Missing)
        }
        ProjectedReadRequest::Index(index) => read_index_ref_direct(value, &index)
            .map(ProjectedReadResponse::Value)
            .unwrap_or(ProjectedReadResponse::Missing),
        ProjectedReadRequest::Contains(needle) => execute_contains_direct(value, &needle)
            .map(ProjectedReadResponse::Bool)
            .unwrap_or(ProjectedReadResponse::Missing),
        ProjectedReadRequest::Find { needle, start } => execute_find_direct(value, &needle, start)
            .map(ProjectedReadResponse::Value)
            .unwrap_or(ProjectedReadResponse::Missing),
        ProjectedReadRequest::GrepText(needle) => execute_grep_text_direct(value, &needle)
            .map(ProjectedReadResponse::Value)
            .unwrap_or(ProjectedReadResponse::Missing),
        ProjectedReadRequest::Keys => match value {
            Value::Record(record) => {
                ProjectedReadResponse::Keys(record.keys().map(ToString::to_string).collect())
            }
            _ => ProjectedReadResponse::Missing,
        },
        ProjectedReadRequest::Values => match value {
            Value::Record(record) => ProjectedReadResponse::Value(Value::List(
                record.values().cloned().collect::<Vec<_>>().into(),
            )),
            Value::Null => ProjectedReadResponse::Value(Value::List(Vec::new().into())),
            _ => ProjectedReadResponse::Missing,
        },
        ProjectedReadRequest::StartsWith(prefix) => {
            let Ok(value) = coerce_string(value) else {
                return ProjectedReadResponse::Missing;
            };
            let Ok(prefix) = coerce_string(&prefix) else {
                return ProjectedReadResponse::Missing;
            };
            ProjectedReadResponse::Bool(value.starts_with(prefix.as_ref()))
        }
        ProjectedReadRequest::EndsWith(suffix) => {
            let Ok(value) = coerce_string(value) else {
                return ProjectedReadResponse::Missing;
            };
            let Ok(suffix) = coerce_string(&suffix) else {
                return ProjectedReadResponse::Missing;
            };
            ProjectedReadResponse::Bool(value.ends_with(suffix.as_ref()))
        }
        ProjectedReadRequest::Split(needle) => {
            let Ok(value) = coerce_string(value) else {
                return ProjectedReadResponse::Missing;
            };
            let Ok(needle) = coerce_string(&needle) else {
                return ProjectedReadResponse::Missing;
            };
            ProjectedReadResponse::Value(Value::List(
                value
                    .split(needle.as_ref())
                    .map(|part| Value::String(part.into()))
                    .collect::<Vec<_>>()
                    .into(),
            ))
        }
        ProjectedReadRequest::Join(sep) => execute_join_builtin(value, &sep)
            .map(ProjectedReadResponse::Value)
            .unwrap_or(ProjectedReadResponse::Missing),
        ProjectedReadRequest::Trim => {
            let Ok(value) = coerce_string(value) else {
                return ProjectedReadResponse::Missing;
            };
            ProjectedReadResponse::Value(Value::String(value.trim().into()))
        }
        ProjectedReadRequest::Slice { start, end } => match value {
            Value::String(value) => {
                ProjectedReadResponse::Value(Value::String(slice_string(value, start, end).into()))
            }
            Value::List(items) => {
                let Some((start, end)) = clamp_slice_bounds(start, end, items.len()) else {
                    return ProjectedReadResponse::Value(Value::List(Vec::new().into()));
                };
                ProjectedReadResponse::Value(Value::List(items[start..end].to_vec().into()))
            }
            _ => ProjectedReadResponse::Missing,
        },
        ProjectedReadRequest::Push(item) => execute_push_builtin(value, item)
            .map(ProjectedReadResponse::Value)
            .unwrap_or(ProjectedReadResponse::Missing),
        ProjectedReadRequest::ToNumber => as_number(value)
            .map(Value::Number)
            .map(ProjectedReadResponse::Value)
            .unwrap_or(ProjectedReadResponse::Missing),
        ProjectedReadRequest::JsonParse => {
            let Ok(text) = coerce_string(value) else {
                return ProjectedReadResponse::Missing;
            };
            serde_json::from_str::<serde_json::Value>(&text)
                .map(from_json)
                .map(ProjectedReadResponse::Value)
                .unwrap_or(ProjectedReadResponse::Missing)
        }
        ProjectedReadRequest::SliceBound => as_slice_bound(value)
            .map(|bound| {
                ProjectedReadResponse::Value(match bound {
                    Some(value) => Value::Number(value as f64),
                    None => Value::Null,
                })
            })
            .unwrap_or(ProjectedReadResponse::Missing),
        ProjectedReadRequest::RangeBound => as_range_bound(value)
            .map(|value| ProjectedReadResponse::Value(Value::Number(value as f64)))
            .unwrap_or(ProjectedReadResponse::Missing),
        ProjectedReadRequest::Render => ProjectedReadResponse::Text(
            stringify_value(value).expect("projected fixture should stringify"),
        ),
        ProjectedReadRequest::Materialize => ProjectedReadResponse::Value(value.clone()),
    }
}

impl ProjectedHostValue for ProjectedFixture {
    fn type_name(&self) -> &str {
        value_type_name(&self.value)
    }

    fn read_one(
        &self,
        request: ProjectedReadRequest,
    ) -> ProjectedFuture<'_, ProjectedReadResponse> {
        Box::pin(async move {
            if matches!(request, ProjectedReadRequest::Materialize) {
                self.materialize_count.fetch_add(1, Ordering::SeqCst);
            }
            projected_response_from_value(&self.value, request)
        })
    }
}

fn projected_value_binding(name: &str, value: Value) -> ProjectedBindings {
    let mut projected = ProjectedBindings::new();
    projected.insert(name, ProjectedValue::scalar(name.to_string(), value));
    projected
}

fn projected_custom_binding(name: &str, value: Arc<dyn ProjectedHostValue>) -> ProjectedBindings {
    let mut projected = ProjectedBindings::new();
    projected.insert(name, ProjectedValue::custom(name.to_string(), value));
    projected
}

async fn exec_with_global_state(
    name: &str,
    value: Value,
    source: &str,
) -> Result<(Value, State), RuntimeError> {
    let program = crate::parse(source).expect("program should parse");
    let mut state = State::new();
    state.globals.insert(name.to_string(), value);
    let outcome = execute_compiled(&compile_program(&program), &mut state, &Host).await?;
    match outcome {
        ExecutionOutcome::Finished(value) => Ok((value, state)),
        ExecutionOutcome::Continued => panic!("expected `submit` in test program"),
        ExecutionOutcome::Failed(value) => panic!("unexpected process failure: {value}"),
    }
}

async fn assert_projected_parity(name: &str, value: Value, source: &str) {
    let (normal, _) = exec_with_global_state(name, value.clone(), source)
        .await
        .expect("normal global should run");
    let projected = projected_value_binding(name, value.clone());
    let (projected_scalar, _) = exec_with_projected(source, &projected)
        .await
        .expect("scalar projected binding should run");
    assert_eq!(
        to_json(&projected_scalar),
        to_json(&normal),
        "scalar projected binding diverged for `{source}`"
    );

    let custom_value = ProjectedFixture::new(value);
    let projected = projected_custom_binding(name, custom_value);
    let (projected_custom, _) = exec_with_projected(source, &projected)
        .await
        .expect("custom projected binding should run");
    assert_eq!(
        to_json(&projected_custom),
        to_json(&normal),
        "custom projected binding diverged for `{source}`"
    );
}

#[test]
fn projected_bindings_reject_duplicate_checked_insertions() {
    let mut projected = ProjectedBindings::new();
    projected
        .try_insert("history", ProjectedValue::scalar("history", Value::Null))
        .expect("first binding should succeed");
    let err = projected
        .try_insert("history", ProjectedValue::scalar("history", Value::Null))
        .expect_err("duplicate binding should fail");
    assert_eq!(err.name(), "history");
}

async fn exec_with_projected(
    source: &str,
    projected: &ProjectedBindings,
) -> Result<(Value, State), RuntimeError> {
    let program = crate::parse(source).expect("program should parse");
    let mut state = State::new();
    let outcome = execute_compiled_with_projected_bindings(
        &compile_program(&program),
        &mut state,
        &Host,
        projected,
    )
    .await?;
    match outcome {
        ExecutionOutcome::Finished(value) => Ok((value, state)),
        ExecutionOutcome::Continued => panic!("expected `submit` in test program"),
        ExecutionOutcome::Failed(value) => panic!("unexpected process failure: {value}"),
    }
}

#[tokio::test(flavor = "current_thread")]
async fn projected_list_len_and_index_are_lazy() {
    let list = TestProjectedValue::new(vec![Value::String("first".into()), Value::Number(2.0)]);
    let projected = projected_list_bindings("history", Arc::clone(&list));

    let (value, _) = exec_with_projected(
        "submit { n: len(history), first: history[0], missing: history[9] }",
        &projected,
    )
    .await
    .expect("projected read");

    let Value::Record(record) = value else {
        panic!("expected record");
    };
    assert_eq!(record["n"], Value::Number(2.0));
    assert_eq!(record["first"], Value::String("first".into()));
    assert_eq!(record["missing"], Value::Null);
    assert_eq!(list.get_count.load(Ordering::SeqCst), 1);
    assert_eq!(list.materialize_count.load(Ordering::SeqCst), 0);
}

#[tokio::test(flavor = "current_thread")]
async fn projected_bindings_are_read_only_and_not_snapshotted() {
    let list = TestProjectedValue::new(vec![Value::String("entry".into())]);
    let projected = projected_list_bindings("history", Arc::clone(&list));

    let err = exec_with_projected("history = []\nsubmit history", &projected)
        .await
        .expect_err("projected root assignment should fail");
    assert!(err.to_string().contains("read-only projected binding"));

    let (_, state) = exec_with_projected("alias = history\nsubmit alias[0]", &projected)
        .await
        .expect("alias should materialize");
    assert!(state.snapshot().globals.get("history").is_none());
    assert!(matches!(
        state.snapshot().globals.get("alias"),
        Some(Value::List(_))
    ));
}

#[tokio::test(flavor = "current_thread")]
async fn projected_children_can_be_lazy_inside_ordinary_records() {
    let body = TestProjectedValue::new(vec![Value::String("lazy markdown".into())]);
    let mut record = Record::default();
    record.insert("title".to_string(), Value::String("Rules".into()));
    record.insert(
        "body".to_string(),
        Value::Projected(ProjectedValue::custom("body", body.clone())),
    );
    let mut projected = ProjectedBindings::new();
    projected.insert(
        "rules",
        ProjectedValue::scalar("rules", Value::Record(Arc::new(record))),
    );

    let (value, _) = exec_with_projected(
        "submit { title: rules.title, first_body_item: rules.body[0] }",
        &projected,
    )
    .await
    .expect("projected child read");

    let Value::Record(record) = value else {
        panic!("expected record");
    };
    assert_eq!(record["title"], Value::String("Rules".into()));
    assert_eq!(
        record["first_body_item"],
        Value::String("lazy markdown".into())
    );
    assert_eq!(body.get_count.load(Ordering::SeqCst), 1);
    assert_eq!(body.materialize_count.load(Ordering::SeqCst), 0);
}

#[tokio::test(flavor = "current_thread")]
async fn print_projected_uses_render_and_submit_materializes() {
    let list = TestProjectedValue::new(vec![Value::String("entry".into())]);
    let projected = projected_list_bindings("history", Arc::clone(&list));

    let (value, _) = exec_with_projected("print history\nsubmit history", &projected)
        .await
        .expect("projected print and submit");
    let _ = to_json(&value);

    assert_eq!(list.render_count.load(Ordering::SeqCst), 1);
    assert_eq!(list.materialize_count.load(Ordering::SeqCst), 1);
}

#[test]
fn snapshot_serialization_marks_projected_values_without_materializing() {
    let projected = Arc::new(SnapshotGuardProjectedValue::default());
    let mut state = State::new();
    state.globals.insert(
        "match_text".to_string(),
        Value::Projected(ProjectedValue::custom("matches[0].text", projected.clone())),
    );

    let encoded = serde_json::to_vec(&state.snapshot()).expect("snapshot encode");
    let wire: serde_json::Value = serde_json::from_slice(&encoded).expect("snapshot json");

    assert_eq!(projected.render_count.load(Ordering::SeqCst), 0);
    assert_eq!(projected.materialize_count.load(Ordering::SeqCst), 0);
    assert_eq!(
        wire["globals"]["match_text"]["__lashlang_snapshot_projected__"],
        serde_json::Value::Bool(true)
    );
    assert_eq!(wire["globals"]["match_text"]["name"], "matches[0].text");
    assert_eq!(wire["globals"]["match_text"]["type_name"], "string");
    let encoded_text = String::from_utf8(encoded).expect("utf8 snapshot");
    assert!(!encoded_text.contains("rendered full text"));
    assert!(!encoded_text.contains("materialized full text"));
}

#[tokio::test(flavor = "current_thread")]
async fn snapshot_restore_projected_marker_becomes_unavailable_placeholder() {
    let snapshot: Snapshot = serde_json::from_value(serde_json::json!({
        "globals": {
            "match_text": {
                "__lashlang_snapshot_projected__": true,
                "name": "matches[0].text",
                "type_name": "string"
            }
        }
    }))
    .expect("snapshot decode");

    let Some(Value::Projected(projected)) = snapshot.globals.get("match_text") else {
        panic!("expected projected placeholder");
    };

    assert_eq!(projected.name(), "matches[0].text");
    assert_eq!(projected.value_type_name(), "string");
    let rendered = projected.render().await;
    assert!(rendered.contains("unavailable after snapshot restore"));
    assert!(rendered.contains("rerun the producing tool"));
    let materialized = projected.materialize_async().await;
    assert!(matches!(materialized, Value::String(_)));
    assert_ne!(materialized, Value::String("materialized full text".into()));
}

#[tokio::test(flavor = "current_thread")]
async fn flat_search_match_projected_text_separates_slice_snapshot_and_stringify_metrics() {
    let text = SearchProjectedText::new("0123456789abcdefghijklmnopqrstuvwxyz");
    let mut match_record = Record::default();
    match_record.insert("title".to_string(), Value::String("first".into()));
    match_record.insert(
        "text".to_string(),
        Value::Projected(ProjectedValue::custom(
            "search.matches[0].text",
            text.clone(),
        )),
    );
    let mut result_record = Record::default();
    result_record.insert(
        "matches".to_string(),
        Value::List(vec![Value::Record(Arc::new(match_record))].into()),
    );

    let (value, state) = exec_with_global_state(
        "r",
        Value::Record(Arc::new(result_record)),
        "m = r.matches[0]\nhead = slice(m.text, 10, 30)\nsubmit { title: m.title, head: head }",
    )
    .await
    .expect("projected search result should run");
    let record = value.as_record().expect("submitted record");
    assert_eq!(record["title"], Value::String("first".into()));
    assert_eq!(record["head"], Value::String("abcdefghijklmnopqrst".into()));
    assert_eq!(text.slice_count.load(Ordering::SeqCst), 1);
    assert_eq!(text.slices(), vec![(Some(10), Some(30))]);
    assert_eq!(text.render_count.load(Ordering::SeqCst), 0);
    assert_eq!(text.materialize_count.load(Ordering::SeqCst), 0);

    let snapshot = state.snapshot();
    let Some(Value::Record(stored_match)) = snapshot.globals.get("m") else {
        panic!("stored match should stay flat record");
    };
    assert!(matches!(
        stored_match.get("text"),
        Some(Value::Projected(_))
    ));
    let encoded = serde_json::to_string(&snapshot).expect("snapshot encode");
    assert!(encoded.contains("__lashlang_snapshot_projected__"));
    assert!(encoded.contains("search.matches[0].text"));
    assert_eq!(text.render_count.load(Ordering::SeqCst), 0);
    assert_eq!(text.materialize_count.load(Ordering::SeqCst), 0);

    let program = crate::parse("submit to_string(m.text)").expect("program should parse");
    let mut state = State::from_snapshot(snapshot);
    let outcome = execute_program(&program, &mut state, &Host)
        .await
        .expect("explicit stringify should run");
    let ExecutionOutcome::Finished(Value::String(full_text)) = outcome else {
        panic!("expected full text");
    };
    assert_eq!(full_text.as_str(), "0123456789abcdefghijklmnopqrstuvwxyz");
    assert_eq!(text.render_count.load(Ordering::SeqCst), 1);
    assert_eq!(text.materialize_count.load(Ordering::SeqCst), 0);
}

#[tokio::test(flavor = "current_thread")]
async fn projected_values_match_normal_values_for_language_operations() {
    assert_projected_parity(
        "input",
        from_json(serde_json::json!({
            "context": "  alpha,beta,gamma  ",
            "items": ["red", "green", "blue"],
            "record": { "a": 1, "b": 2 },
            "n": "42",
            "json": "{\"ok\":true}",
            "start": 1,
            "end": 4
        })),
        r#"
        out = {
          exact_smoke: slice(input.context, 2, 7),
          field: input.record.a,
          index: input.items[input.start],
          len_context: len(input.context),
          empty_items: empty(input.items),
          keys_record: keys(input.record),
          values_record: values(input.record),
          contains_text: contains(input.context, "beta"),
          contains_list: contains(input.items, "green"),
          contains_record: contains(input.record, "a"),
          find_text: find(input.context, "beta"),
          grep_text: grep_text(input.context, "beta"),
          starts: starts_with(trim(input.context), "alpha"),
          ends: ends_with(trim(input.context), "gamma"),
          split: split(trim(input.context), ","),
          joined: join(input.items, "|"),
          trimmed: trim(input.context),
          list_slice: slice(input.items, 0, 2),
          pushed: push(input.items, "yellow"),
          as_int: to_int(input.n),
          as_float: to_float(input.n),
          parsed: json_parse(input.json),
          plus: input.record.a + 1,
          neg: -input.record.a,
          cmp: input.record.a < input.record.b,
          truthy: input.record.a ? "yes" : "no",
          formatted: format("ctx={}", input.context),
          text: to_string(input.record)
        }
        submit out
        "#,
    )
    .await;
}

#[tokio::test(flavor = "current_thread")]
async fn projected_values_match_normal_values_for_ranges_validation_and_iteration() {
    assert_projected_parity(
        "input",
        from_json(serde_json::json!({
            "start": 2,
            "end": 5,
            "item": { "name": "pkg", "version": "1.0" }
        })),
        r#"
        total = 0
        for i in range(input.start, input.end) {
          total = total + i
        }
        submit {
          range_values: range(input.start, input.end),
          total: total,
          validated: validate(input.item, Type { name: str, version: str })
        }
        "#,
    )
    .await;
}

#[tokio::test(flavor = "current_thread")]
async fn projected_empty_rejects_scalar_like_normal_empty() {
    let normal = exec_with_global_state("n", Value::Number(1.0), "submit empty(n)")
        .await
        .expect_err("normal scalar empty should fail");
    let projected = projected_value_binding("n", Value::Number(1.0));
    let projected_err = exec_with_projected("submit empty(n)", &projected)
        .await
        .expect_err("projected scalar empty should fail");
    assert_eq!(projected_err, normal);
}

struct OverrideProjectedValue {
    value: Value,
    calls: std::sync::Mutex<Vec<&'static str>>,
}

impl OverrideProjectedValue {
    fn new(value: Value) -> Arc<Self> {
        Arc::new(Self {
            value,
            calls: std::sync::Mutex::new(Vec::new()),
        })
    }

    fn push_call(&self, name: &'static str) {
        self.calls.lock().expect("calls lock").push(name);
    }

    fn calls(&self) -> Vec<&'static str> {
        self.calls.lock().expect("calls lock").clone()
    }
}

impl ProjectedHostValue for OverrideProjectedValue {
    fn type_name(&self) -> &str {
        value_type_name(&self.value)
    }

    fn read_one(
        &self,
        request: ProjectedReadRequest,
    ) -> ProjectedFuture<'_, ProjectedReadResponse> {
        Box::pin(async move {
            match request {
                ProjectedReadRequest::Len => {
                    self.push_call("len");
                    value_len(&self.value)
                        .map(ProjectedReadResponse::Len)
                        .unwrap_or(ProjectedReadResponse::Missing)
                }
                ProjectedReadRequest::Empty => {
                    self.push_call("empty");
                    value_len(&self.value)
                        .map(|len| ProjectedReadResponse::Bool(len == 0))
                        .unwrap_or(ProjectedReadResponse::Missing)
                }
                ProjectedReadRequest::Truthy => {
                    self.push_call("truthy");
                    ProjectedReadResponse::Bool(is_truthy(&self.value))
                }
                ProjectedReadRequest::Index(index) => {
                    self.push_call("get_index");
                    read_index_ref_direct(&self.value, &index)
                        .map(ProjectedReadResponse::Value)
                        .unwrap_or(ProjectedReadResponse::Missing)
                }
                ProjectedReadRequest::Field(field) => {
                    self.push_call("get_field");
                    let field = Name {
                        symbol: intern_symbol(field.as_ref()),
                        text: field,
                    };
                    read_field_ref_direct(&self.value, &field)
                        .map(ProjectedReadResponse::Value)
                        .unwrap_or(ProjectedReadResponse::Missing)
                }
                ProjectedReadRequest::Contains(needle) => {
                    self.push_call("contains");
                    ProjectedReadResponse::Bool(
                        execute_contains_direct(&self.value, &needle).expect("contains override"),
                    )
                }
                ProjectedReadRequest::Find { needle, start } => {
                    self.push_call("find");
                    execute_find_direct(&self.value, &needle, start)
                        .map(ProjectedReadResponse::Value)
                        .unwrap_or(ProjectedReadResponse::Missing)
                }
                ProjectedReadRequest::GrepText(needle) => {
                    self.push_call("grep_text");
                    execute_grep_text_direct(&self.value, &needle)
                        .map(ProjectedReadResponse::Value)
                        .unwrap_or(ProjectedReadResponse::Missing)
                }
                ProjectedReadRequest::Keys => {
                    self.push_call("keys");
                    match &self.value {
                        Value::Record(record) => ProjectedReadResponse::Keys(
                            record.keys().map(ToString::to_string).collect(),
                        ),
                        _ => ProjectedReadResponse::Missing,
                    }
                }
                ProjectedReadRequest::Values => {
                    self.push_call("values");
                    match &self.value {
                        Value::Record(record) => ProjectedReadResponse::Value(Value::List(
                            record.values().cloned().collect::<Vec<_>>().into(),
                        )),
                        _ => ProjectedReadResponse::Missing,
                    }
                }
                ProjectedReadRequest::StartsWith(prefix) => {
                    self.push_call("starts_with");
                    let value = coerce_string(&self.value).expect("string receiver");
                    let prefix = coerce_string(&prefix).expect("string prefix");
                    ProjectedReadResponse::Bool(value.starts_with(prefix.as_ref()))
                }
                ProjectedReadRequest::EndsWith(suffix) => {
                    self.push_call("ends_with");
                    let value = coerce_string(&self.value).expect("string receiver");
                    let suffix = coerce_string(&suffix).expect("string suffix");
                    ProjectedReadResponse::Bool(value.ends_with(suffix.as_ref()))
                }
                ProjectedReadRequest::Split(needle) => {
                    self.push_call("split");
                    let value = coerce_string(&self.value).expect("string receiver");
                    let needle = coerce_string(&needle).expect("string needle");
                    ProjectedReadResponse::Value(Value::List(
                        value
                            .split(needle.as_ref())
                            .map(|part| Value::String(part.to_string().into()))
                            .collect::<Vec<_>>()
                            .into(),
                    ))
                }
                ProjectedReadRequest::Join(sep) => {
                    self.push_call("join");
                    execute_join_builtin(&self.value, &sep)
                        .map(ProjectedReadResponse::Value)
                        .unwrap_or(ProjectedReadResponse::Missing)
                }
                ProjectedReadRequest::Trim => {
                    self.push_call("trim");
                    let value = coerce_string(&self.value).expect("string receiver");
                    ProjectedReadResponse::Value(Value::String(value.trim().to_string().into()))
                }
                ProjectedReadRequest::Slice { start, end } => {
                    self.push_call("slice");
                    match &self.value {
                        Value::String(value) => ProjectedReadResponse::Value(Value::String(
                            slice_string(value, start, end).into(),
                        )),
                        Value::List(items) => {
                            let Some((start, end)) = clamp_slice_bounds(start, end, items.len())
                            else {
                                return ProjectedReadResponse::Value(Value::List(
                                    Vec::new().into(),
                                ));
                            };
                            ProjectedReadResponse::Value(Value::List(
                                items[start..end].to_vec().into(),
                            ))
                        }
                        _ => ProjectedReadResponse::Missing,
                    }
                }
                ProjectedReadRequest::Push(item) => {
                    self.push_call("push");
                    execute_push_builtin(&self.value, item)
                        .map(ProjectedReadResponse::Value)
                        .unwrap_or(ProjectedReadResponse::Missing)
                }
                ProjectedReadRequest::ToNumber => {
                    self.push_call("to_number");
                    as_number(&self.value)
                        .map(Value::Number)
                        .map(ProjectedReadResponse::Value)
                        .unwrap_or(ProjectedReadResponse::Missing)
                }
                ProjectedReadRequest::JsonParse => {
                    self.push_call("json_parse");
                    let value = coerce_string(&self.value).expect("json text");
                    serde_json::from_str::<serde_json::Value>(&value)
                        .map(from_json)
                        .map(ProjectedReadResponse::Value)
                        .unwrap_or(ProjectedReadResponse::Missing)
                }
                ProjectedReadRequest::SliceBound => {
                    self.push_call("slice_bound");
                    as_slice_bound(&self.value)
                        .map(|bound| {
                            ProjectedReadResponse::Value(match bound {
                                Some(value) => Value::Number(value as f64),
                                None => Value::Null,
                            })
                        })
                        .unwrap_or(ProjectedReadResponse::Missing)
                }
                ProjectedReadRequest::RangeBound => {
                    self.push_call("range_bound");
                    as_range_bound(&self.value)
                        .map(|value| ProjectedReadResponse::Value(Value::Number(value as f64)))
                        .unwrap_or(ProjectedReadResponse::Missing)
                }
                ProjectedReadRequest::Materialize => {
                    self.push_call("materialize");
                    ProjectedReadResponse::Value(self.value.clone())
                }
                ProjectedReadRequest::Render => ProjectedReadResponse::Text(
                    stringify_value(&self.value).expect("render projected override"),
                ),
            }
        })
    }
}

async fn assert_override_uses_hook(
    source: &str,
    name: &'static str,
    value: Value,
    expected_hook: &'static str,
) {
    let projected_value = OverrideProjectedValue::new(value);
    let mut projected = ProjectedBindings::new();
    projected.insert(
        name,
        ProjectedValue::custom(name, projected_value.clone() as Arc<dyn ProjectedHostValue>),
    );
    exec_with_projected(source, &projected)
        .await
        .expect("override projected operation should run");
    let calls = projected_value.calls();
    assert!(
        calls.contains(&expected_hook),
        "expected `{expected_hook}` override for `{source}`, got {calls:?}"
    );
    assert!(
        !calls.contains(&"materialize"),
        "`{source}` should use override hooks without materializing, got {calls:?}"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn projected_host_values_can_override_all_lazy_receiver_operations() {
    let record = from_json(serde_json::json!({ "a": 1, "b": 2 }));
    let list = from_json(serde_json::json!(["a", "b", "c"]));

    assert_override_uses_hook("submit p.a", "p", record.clone(), "get_field").await;
    assert_override_uses_hook("submit p[1]", "p", list.clone(), "get_index").await;
    assert_override_uses_hook("submit len(p)", "p", list.clone(), "len").await;
    assert_override_uses_hook("submit empty(p)", "p", list.clone(), "empty").await;
    assert_override_uses_hook("submit keys(p)", "p", record.clone(), "keys").await;
    assert_override_uses_hook("submit values(p)", "p", record.clone(), "values").await;
    assert_override_uses_hook(r#"submit contains(p, "b")"#, "p", list.clone(), "contains").await;
    assert_override_uses_hook(
        r#"submit find(p, "ph")"#,
        "p",
        Value::String("alpha".into()),
        "find",
    )
    .await;
    assert_override_uses_hook(
        r#"submit grep_text(p, "beta")"#,
        "p",
        Value::String("alpha\nbeta\n".into()),
        "grep_text",
    )
    .await;
    assert_override_uses_hook(
        r#"submit starts_with(p, "al")"#,
        "p",
        Value::String("alpha".into()),
        "starts_with",
    )
    .await;
    assert_override_uses_hook(
        r#"submit ends_with(p, "ha")"#,
        "p",
        Value::String("alpha".into()),
        "ends_with",
    )
    .await;
    assert_override_uses_hook(
        r#"submit split(p, ",")"#,
        "p",
        Value::String("a,b".into()),
        "split",
    )
    .await;
    assert_override_uses_hook(r#"submit join(p, "|")"#, "p", list.clone(), "join").await;
    assert_override_uses_hook(
        "submit trim(p)",
        "p",
        Value::String("  alpha  ".into()),
        "trim",
    )
    .await;
    assert_override_uses_hook(
        "submit slice(p, 1, 3)",
        "p",
        Value::String("alpha".into()),
        "slice",
    )
    .await;
    assert_override_uses_hook("submit push(p, \"d\")", "p", list, "push").await;
    assert_override_uses_hook(
        "submit to_int(p)",
        "p",
        Value::String("42".into()),
        "to_number",
    )
    .await;
    assert_override_uses_hook(
        "submit to_float(p)",
        "p",
        Value::String("42.5".into()),
        "to_number",
    )
    .await;
    assert_override_uses_hook(
        "submit json_parse(p)",
        "p",
        Value::String("{\"ok\":true}".into()),
        "json_parse",
    )
    .await;
    assert_override_uses_hook(
        "submit slice(\"abcdef\", p, null)",
        "p",
        Value::Number(2.0),
        "slice_bound",
    )
    .await;
    assert_override_uses_hook("submit range(p, 4)", "p", Value::Number(1.0), "range_bound").await;
    assert_override_uses_hook(
        "submit range(0, p, 2)",
        "p",
        Value::Number(4.0),
        "range_bound",
    )
    .await;
    assert_override_uses_hook("submit p ? 1 : 2", "p", Value::Number(1.0), "truthy").await;
}

#[tokio::test(flavor = "current_thread")]
async fn image_values_expose_read_only_metadata_fields() {
    let value = exec_with_global(
        "img",
        test_image(),
        "submit [img.id, img.label, img.size, img.width, img.height, img.missing]",
    )
    .await
    .expect("image fields should read");

    assert_eq!(
        value,
        Value::List(
            vec![
                Value::String("img-1".into()),
                Value::String("chart.png".into()),
                Value::Number(1234.0),
                Value::Number(640.0),
                Value::Number(480.0),
                Value::Null,
            ]
            .into()
        )
    );
}

#[tokio::test(flavor = "current_thread")]
async fn image_values_serialize_as_descriptors() {
    let image = test_image();
    assert_eq!(
        to_json(&image),
        serde_json::json!({
            "type": "image",
            "id": "img-1",
            "label": "chart.png",
            "size": 1234,
            "width": 640,
            "height": 480
        })
    );
    assert_eq!(
        stringify_value(&image).expect("stringify image"),
        r#"{"height":480,"id":"img-1","label":"chart.png","size":1234,"type":"image","width":640}"#
    );
    assert_eq!(
        exec_with_global("img", image.clone(), "submit img")
            .await
            .expect("submit image"),
        image
    );
}

#[tokio::test(flavor = "current_thread")]
async fn image_values_are_immutable_and_len_is_unsupported() {
    let err = exec_with_global("img", test_image(), "img.label = \"other\"\nsubmit img")
        .await
        .expect_err("image field assignment should fail");
    assert_eq!(
        err,
        RuntimeError::TypeError {
            message: "can't assign image fields; images are immutable".to_string()
        }
    );

    let err = exec_with_global("img", test_image(), "submit len(img)")
        .await
        .expect_err("len image should fail");
    assert_eq!(
        err,
        RuntimeError::TypeError {
            message: "`len` requires a string, list, record, or null; use `.size` for images"
                .to_string()
        }
    );
}

#[tokio::test(flavor = "current_thread")]
async fn false_if_branch_and_finish_inside_loop_are_covered() {
    let value = exec(
        r#"
        if false {
          out = 1
        } else {
          out = 2
        }
        submit out
        "#,
    )
    .await
    .expect("else branch should succeed");
    assert_eq!(value, Value::Number(2.0));

    let value = exec(
        r#"
        for x in [1, 2] {
          submit x
        }
        submit 0
        "#,
    )
    .await
    .expect("submit inside loop should bubble out");
    assert_eq!(value, Value::Number(1.0));
}

#[tokio::test(flavor = "current_thread")]
async fn await_record_process_starts_and_joins_handles() {
    struct BatchHost {
        calls: AtomicUsize,
        batches: AtomicUsize,
    }

    impl ExecutionHost for BatchHost {
        async fn perform(&self, op: AbilityOp) -> Result<AbilityResult, ExecutionHostError> {
            match op {
                AbilityOp::StartProcess(start) => {
                    self.calls.fetch_add(1, Ordering::Relaxed);
                    let mut handle = Record::new();
                    handle.insert("__handle__".to_string(), Value::String("process".into()));
                    handle.insert(
                        "process".to_string(),
                        Value::String(start.process_name.into()),
                    );
                    handle.insert(
                        "value".to_string(),
                        start.args.get("value").cloned().unwrap_or(Value::Null),
                    );
                    Ok(AbilityResult::Value(Value::Record(Arc::new(handle))))
                }
                AbilityOp::Await(handle) => {
                    let value = handle
                        .as_record()
                        .and_then(|record| record.get("value"))
                        .cloned()
                        .unwrap_or(Value::Null);
                    Ok(AbilityResult::Value(value))
                }
                AbilityOp::Submit(value) | AbilityOp::Finish(value) | AbilityOp::Fail(value) => {
                    Ok(AbilityResult::Value(value))
                }
                _ => Err(ExecutionHostError::new("unsupported host ability")),
            }
        }
    }

    let host = BatchHost {
        calls: AtomicUsize::new(0),
        batches: AtomicUsize::new(0),
    };
    let program = crate::parse(
        r#"
        process echo(value: str) { finish value }
        result = await {
          left: start echo(value: "a"),
          right: start echo(value: "b")
        }
        submit [result.left?, result.right?]
        "#,
    )
    .expect("program should parse");
    let mut state = State::new();
    let outcome = execute_program(&program, &mut state, &host)
        .await
        .expect("program should run");

    assert_eq!(
        outcome,
        ExecutionOutcome::Finished(Value::List(
            vec![Value::String("a".into()), Value::String("b".into())].into()
        ))
    );
    assert_eq!(host.calls.load(Ordering::Relaxed), 2);
    assert_eq!(host.batches.load(Ordering::Relaxed), 0);
}

#[tokio::test(flavor = "current_thread")]
async fn truthiness_covers_scalar_and_container_values() {
    assert!(!is_truthy(&Value::Null));
    assert!(!is_truthy(&Value::Bool(false)));
    assert!(!is_truthy(&Value::Number(0.0)));
    assert!(!is_truthy(&Value::String(String::new().into())));
    assert!(is_truthy(&Value::Bool(true)));
    assert!(is_truthy(&Value::Number(1.0)));
    assert!(is_truthy(&Value::List(Vec::new().into())));
    assert!(is_truthy(&Value::Record(Record::default().into())));
}

struct AsyncHost;

impl ExecutionHost for AsyncHost {
    async fn perform(&self, op: AbilityOp) -> Result<AbilityResult, ExecutionHostError> {
        match op {
            AbilityOp::ResourceOperation(operation) => {
                Host.perform(AbilityOp::ResourceOperation(operation)).await
            }
            AbilityOp::StartProcess(start) => {
                let mut record = Record::default();
                record.insert("__handle__".to_string(), Value::String("process".into()));
                record.insert(
                    "process".to_string(),
                    Value::String(start.process_name.into()),
                );
                record.insert(
                    "value".to_string(),
                    start.args.get("value").cloned().unwrap_or(Value::Null),
                );
                Ok(AbilityResult::Value(Value::Record(Arc::new(record))))
            }
            AbilityOp::Await(handle) => {
                let record = handle
                    .as_record()
                    .ok_or_else(|| ExecutionHostError::new("expected handle record"))?;
                Ok(AbilityResult::Value(
                    record.get("value").cloned().unwrap_or(Value::Null),
                ))
            }
            AbilityOp::Cancel(_) => Ok(AbilityResult::Value(Value::Null)),
            AbilityOp::Print(_) => Ok(AbilityResult::Unit),
            AbilityOp::Submit(value) | AbilityOp::Finish(value) | AbilityOp::Fail(value) => {
                Ok(AbilityResult::Value(value))
            }
            _ => Err(ExecutionHostError::new("unsupported host ability")),
        }
    }
}

#[test]
fn linked_trigger_declaration_records_process_binding() {
    let mut resources = crate::ResourceCatalog::new();
    resources.add_module_instance(["tools"], "Tools");
    resources.add_operation("Tools", "echo", "echo");
    resources.add_trigger_event(
        "Tools",
        "changed",
        crate::TypeExpr::Object(vec![crate::TypeField {
            name: "path".into(),
            ty: crate::TypeExpr::Str,
            optional: false,
        }]),
    );
    let surface = crate::LashlangSurface::new(resources, crate::LashlangAbilities::all());
    let program = crate::parse(
        r#"
        type Changed = { path: str }
        process scan(tool: Tools, event: Changed) {
          finish event.path
        }

        trigger changed on tools.changed as event
          -> scan(tool: tools, event: event)
        "#,
    )
    .expect("program should parse");
    let linked = crate::LinkedModule::link(program, surface).expect("program should link");
    let trigger = linked
        .artifact
        .canonical_ir
        .declarations
        .iter()
        .find_map(|declaration| match declaration {
            crate::Declaration::Trigger(trigger) => Some(trigger),
            _ => None,
        })
        .expect("trigger declaration");
    assert_eq!(trigger.process_name, "scan");
    assert!(matches!(
        trigger.args[0].1,
        crate::TriggerArg::ResourceRef(_)
    ));
    assert!(matches!(
        trigger.args[1].1,
        crate::TriggerArg::EventBinding(ref name) if name == "event"
    ));
}

#[tokio::test(flavor = "current_thread")]
async fn process_handles_can_be_started_awaited_and_cancelled() {
    let program = crate::parse(
        r#"
        process echo(value: str) { finish value }
        handle = start echo(value: "done")
        result = await handle
        cancel handle
        submit result
        "#,
    )
    .expect("program should parse");
    let mut state = State::new();
    let outcome = execute_program(&program, &mut state, &AsyncHost)
        .await
        .expect("program should run");
    let ExecutionOutcome::Finished(value) = outcome else {
        panic!("expected finish");
    };
    let record = value
        .as_record()
        .expect("await should return wrapped result");
    assert_eq!(record["ok"], Value::Bool(true));
    assert_eq!(record["value"], Value::String("done".into()));
}

#[tokio::test(flavor = "current_thread")]
async fn start_process_returns_raw_handle_and_passes_explicit_input() {
    let host = RecordingProcessHost::default();
    let program = crate::parse(
        r#"
        process scan(root: string) -> null {
          finish root
        }
        handle = start scan(root: ".")
        submit handle
        "#,
    )
    .expect("program should parse");
    let mut state = State::new();
    let outcome = execute_program(&program, &mut state, &host)
        .await
        .expect("program should run");
    let ExecutionOutcome::Finished(value) = outcome else {
        panic!("expected finish");
    };
    let handle = value.as_record().expect("start should return a handle");
    assert_eq!(handle["__handle__"], Value::String("process".into()));
    assert_eq!(handle["id"], Value::String("proc-1".into()));

    let starts = host.starts.lock().expect("starts lock");
    assert_eq!(starts.len(), 1);
    let start = &starts[0];
    assert_eq!(start.process_name, "scan");
    assert_eq!(start.args["root"], Value::String(".".into()));
    assert!(start.module_ref.as_str().starts_with("lashlang:v1:sha256:"));
}

#[tokio::test(flavor = "current_thread")]
async fn unlinked_compiled_program_rejects_process_starts() {
    let program = crate::parse(
        r#"
        process scan() { finish 1 }
        submit start scan()
        "#,
    )
    .expect("program should parse");
    let compiled = compile_program(&program);
    let mut state = State::new();

    let err = execute_compiled(&compiled, &mut state, &RecordingProcessHost::default())
        .await
        .expect_err("unlinked start should fail");

    assert!(err.to_string().contains("linked lashlang module artifact"));
}

#[test]
fn compiled_process_cache_reuses_process_ref_and_surface_ref() {
    let linked = crate::LinkedModule::link(
        crate::parse("process scan() { finish 1 }").expect("parse module"),
        runtime_test_surface(),
    )
    .expect("link module");
    let process_ref = linked
        .artifact
        .process_ref("scan")
        .expect("scan process ref")
        .clone();
    let mut cache = CompiledProcessCache::with_capacity(2);

    let first = cache
        .get_or_compile(&linked.artifact, &process_ref, &linked.required_surface_ref)
        .expect("compile first");
    let second = cache
        .get_or_compile(&linked.artifact, &process_ref, &linked.required_surface_ref)
        .expect("compile second");

    assert!(Arc::ptr_eq(&first, &second));
    assert_eq!(cache.stats().hits, 1);
    assert_eq!(cache.stats().misses, 1);
}

#[tokio::test(flavor = "current_thread")]
async fn receiver_module_operation_unwraps_result() {
    let value = exec(r#"submit (await tools.echo({ value: "ok" })?)"#)
        .await
        .expect("module operation should run");
    assert_eq!(value, Value::String("ok".into()));
}

#[tokio::test(flavor = "current_thread")]
async fn receiver_module_operation_errors_are_sanitized() {
    let err = exec(r#"submit (await tools.err({ value: "nope" })?)"#)
        .await
        .expect_err("module operation should fail");
    assert!(matches!(err, RuntimeError::ValueError { .. }));
    assert!(err.to_string().contains("module operation"));
}

#[tokio::test(flavor = "current_thread")]
async fn process_controls_emit_events_and_terminal_outcomes() {
    let host = RecordingProcessHost::default();
    let program = Program::block(vec![
        Expr::Yield(Box::new(Expr::String("checkpoint".into()))),
        Expr::Wake(Box::new(Expr::String("ready".into()))),
        Expr::Finish(Some(Box::new(Expr::String("done".into())))),
    ]);
    let mut state = State::new();
    let compiled = compile_program(&program);
    let outcome = execute_compiled_process(&compiled, &mut state, &host)
        .await
        .expect("process controls should run");
    assert_eq!(
        outcome,
        ExecutionOutcome::Finished(Value::String("done".into()))
    );
    let events = host.events.lock().expect("events lock");
    assert_eq!(events.len(), 2);
    assert_eq!(events[0].kind, ProcessEventKind::Yield);
    assert_eq!(events[0].value, Value::String("checkpoint".into()));
    assert_eq!(events[1].kind, ProcessEventKind::Wake);
    assert_eq!(events[1].value, Value::String("ready".into()));
}

#[tokio::test(flavor = "current_thread")]
async fn process_lifecycle_controls_sleep_wait_and_signal() {
    let host = RecordingProcessHost::default();
    let mut handle = Record::new();
    handle.insert("__handle__".to_string(), Value::String("process".into()));
    handle.insert("id".to_string(), Value::String("target".into()));
    let program = Program::block(vec![
        Expr::SleepFor(Box::new(Expr::Number(5.0))),
        Expr::Assign {
            target: crate::AssignTarget::variable("payload".into()),
            expr: Box::new(Expr::WaitSignal),
        },
        Expr::SignalRun {
            run: Box::new(Expr::Variable("run".into())),
            payload: Box::new(Expr::Variable("payload".into())),
        },
        Expr::Finish(Some(Box::new(Expr::Variable("payload".into())))),
    ]);
    let mut globals = Record::new();
    globals.insert("run".to_string(), Value::Record(Arc::new(handle)));
    let mut state = State::from_snapshot(Snapshot { globals });
    let compiled = compile_program(&program);

    let outcome = execute_compiled_process(&compiled, &mut state, &host)
        .await
        .expect("process lifecycle controls should run");

    assert_eq!(
        outcome,
        ExecutionOutcome::Finished(Value::String("signal-payload".into()))
    );
    let sleeps = host.sleeps.lock().expect("sleeps lock");
    assert_eq!(sleeps.len(), 1);
    assert_eq!(sleeps[0].kind, ProcessSleepKind::For);
    assert_eq!(sleeps[0].value, Value::Number(5.0));
    let signals = host.signals.lock().expect("signals lock");
    assert_eq!(signals.len(), 1);
    assert_eq!(signals[0].payload, Value::String("signal-payload".into()));
}

#[tokio::test(flavor = "current_thread")]
async fn process_fail_returns_terminal_failure_outcome() {
    let host = RecordingProcessHost::default();
    let program = Program::block(vec![Expr::Fail(Box::new(Expr::Record(vec![(
        "reason".into(),
        Expr::String("bad".into()),
    )])))]);
    let mut state = State::new();
    let compiled = compile_program(&program);
    let outcome = execute_compiled_process(&compiled, &mut state, &host)
        .await
        .expect("process fail should run");
    let ExecutionOutcome::Failed(value) = outcome else {
        panic!("expected process failure");
    };
    let failure = value
        .as_record()
        .expect("failure should preserve raw value");
    assert_eq!(failure["reason"], Value::String("bad".into()));
}

#[tokio::test(flavor = "current_thread")]
async fn process_mode_falling_off_end_finishes_null() {
    let host = RecordingProcessHost::default();
    let program = Program::block(vec![Expr::String("ignored".into())]);
    let compiled = compile_program(&program);
    let mut state = State::new();

    let outcome = execute_compiled_process(&compiled, &mut state, &host)
        .await
        .expect("process should run");

    assert_eq!(outcome, ExecutionOutcome::Finished(Value::Null));
}

#[tokio::test(flavor = "current_thread")]
async fn foreground_rejects_programmatic_process_controls() {
    for (keyword, stmt) in [
        ("yield", Expr::Yield(Box::new(Expr::String("event".into())))),
        ("wake", Expr::Wake(Box::new(Expr::String("event".into())))),
        ("sleep", Expr::SleepFor(Box::new(Expr::Number(1.0)))),
        ("wait signal", Expr::WaitSignal),
        (
            "signal run",
            Expr::SignalRun {
                run: Box::new(Expr::Null),
                payload: Box::new(Expr::Null),
            },
        ),
        (
            "finish",
            Expr::Finish(Some(Box::new(Expr::String("done".into())))),
        ),
        ("fail", Expr::Fail(Box::new(Expr::String("bad".into())))),
    ] {
        let program = Program::block(vec![stmt]);
        let mut state = State::new();
        let host = RecordingProcessHost::default();
        let err = execute_program(&program, &mut state, &host)
            .await
            .expect_err("foreground mode should reject process controls");
        assert_eq!(err, RuntimeError::ProcessControlOutsideProcess { keyword });
    }
}

#[tokio::test(flavor = "current_thread")]
async fn process_mode_rejects_programmatic_foreground_controls() {
    for (keyword, stmt) in [
        (
            "submit",
            Expr::Submit(Some(Box::new(Expr::String("done".into())))),
        ),
        ("print", Expr::Print(Box::new(Expr::String("debug".into())))),
    ] {
        let program = Program::block(vec![stmt]);
        let compiled = compile_program(&program);
        let mut state = State::new();
        let host = RecordingProcessHost::default();
        let err = execute_compiled_process(&compiled, &mut state, &host)
            .await
            .expect_err("process mode should reject foreground controls");
        assert_eq!(
            err,
            RuntimeError::ForegroundControlInsideProcess { keyword }
        );
    }
}

#[tokio::test(flavor = "current_thread")]
async fn sync_steps_resume_correctly_after_tool_effects() {
    let value = exec(
        r#"
        before = 20 + 2
        echoed = await tools.echo({ value: before })?
        after = echoed + 1
        submit [before, echoed, after]
        "#,
    )
    .await
    .expect("program should run");

    assert_eq!(
        value,
        Value::List(
            vec![
                Value::Number(22.0),
                Value::Number(22.0),
                Value::Number(23.0)
            ]
            .into()
        )
    );
}

#[tokio::test(flavor = "current_thread")]
async fn traced_started_tool_errors_keep_original_instruction_span() {
    let source = r#"
        before = 1
        value = await tools.err({})?
        submit value
        "#;
    let compiled = compile_source(source).expect("program should compile");
    let mut state = State::new();
    let failure = execute_compiled_traced(&compiled, &mut state, &Host)
        .await
        .expect_err("unwrapped module operation error should fail");
    let message = crate::format_runtime_diagnostic(source, &failure.error, failure.span);

    assert!(
        message.contains("`?` unwrapped failed module operation: boom"),
        "{message}"
    );
    assert!(message.contains("--> line 3, column 9"), "{message}");
    assert!(
        message.contains("value = await tools.err({})?"),
        "{message}"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn profiled_tool_effect_keeps_sync_instruction_counts() {
    let source = r#"
        before = 20 + 2
        echoed = await tools.echo({ value: before })?
        after = echoed + 1
        submit after
        "#;
    let compiled = compile_source(source).expect("program should compile");
    let mut state = State::new();
    let (_outcome, report) = profile_compiled(&compiled, &mut state, &Host)
        .await
        .expect("profile should succeed");
    let count = |name| {
        report
            .instruction_stats()
            .iter()
            .find(|stat| stat.name == name)
            .map_or(0, |stat| stat.count)
    };

    assert!(
        count("resource_call") > 0,
        "{:?}",
        report.instruction_stats()
    );
    assert!(count("binary") > 0, "{:?}", report.instruction_stats());
    assert!(count("load_name") > 0, "{:?}", report.instruction_stats());
    assert!(count("store_name") >= 3, "{:?}", report.instruction_stats());
}

#[tokio::test(flavor = "current_thread")]
async fn await_unknown_handle_surfaces_runtime_error() {
    let program = crate::parse(
        r#"
        result = await 1
        submit result
        "#,
    )
    .expect("program should parse");
    let mut state = State::new();
    let outcome = execute_program(&program, &mut state, &AsyncHost)
        .await
        .expect("program should run");
    let ExecutionOutcome::Finished(value) = outcome else {
        panic!("expected finish");
    };
    let record = value
        .as_record()
        .expect("await should return wrapped error");
    assert_eq!(record["ok"], Value::Bool(false));
    assert_eq!(
        record["error"],
        Value::String("expected handle record".into())
    );
}

#[tokio::test(flavor = "current_thread")]
async fn await_list_of_handles_returns_results_in_order() {
    let program = crate::parse(
        r#"
        process echo(value: str) { finish value }
        handles = [
          start echo(value: "first"),
          start echo(value: "second"),
          start echo(value: "third")
        ]
        results = await handles
        submit results
        "#,
    )
    .expect("program should parse");
    let mut state = State::new();
    let outcome = execute_program(&program, &mut state, &AsyncHost)
        .await
        .expect("program should run");
    let ExecutionOutcome::Finished(value) = outcome else {
        panic!("expected finish");
    };
    let Value::List(results) = value else {
        panic!("await list should return a list");
    };
    assert_eq!(results.len(), 3);
    for (result, expected) in results.iter().zip(["first", "second", "third"]) {
        let record = result
            .as_record()
            .expect("await should return wrapped result");
        assert_eq!(record["ok"], Value::Bool(true));
        assert_eq!(record["value"], Value::String(expected.into()));
    }
}

#[tokio::test(flavor = "current_thread")]
async fn await_list_preserves_per_item_errors() {
    let program = crate::parse(
        r#"
        process echo(value: str) { finish value }
        handles = [start echo(value: "done"), 1]
        results = await handles
        submit results
        "#,
    )
    .expect("program should parse");
    let mut state = State::new();
    let outcome = execute_program(&program, &mut state, &AsyncHost)
        .await
        .expect("program should run");
    let ExecutionOutcome::Finished(value) = outcome else {
        panic!("expected finish");
    };
    let Value::List(results) = value else {
        panic!("await list should return a list");
    };
    let ok = results[0]
        .as_record()
        .expect("first result should be wrapped");
    assert_eq!(ok["ok"], Value::Bool(true));
    assert_eq!(ok["value"], Value::String("done".into()));

    let err = results[1]
        .as_record()
        .expect("second result should be wrapped");
    assert_eq!(err["ok"], Value::Bool(false));
    assert_eq!(err["error"], Value::String("expected handle record".into()));
}

#[tokio::test(flavor = "current_thread")]
async fn await_record_of_handles_returns_record_of_wrappers() {
    let program = crate::parse(
        r#"
        process echo(value: str) { finish value }
        handles = {
          first: start echo(value: "one"),
          second: start echo(value: "two"),
        }
        results = await handles
        submit [results.first?, results.second?]
        "#,
    )
    .expect("program should parse");
    let mut state = State::new();
    let outcome = execute_program(&program, &mut state, &AsyncHost)
        .await
        .expect("program should run");
    let ExecutionOutcome::Finished(value) = outcome else {
        panic!("expected finish");
    };
    assert_eq!(
        value,
        Value::List(vec![Value::String("one".into()), Value::String("two".into())].into())
    );
}

#[tokio::test(flavor = "current_thread")]
async fn result_unwrap_extracts_awaited_handles_and_joined_results() {
    let program = crate::parse(
        r#"
        process echo(value: str) { finish value }
        handle = start echo(value: "done")
        result = (await handle)?
        submit result
        "#,
    )
    .expect("program should parse");
    let mut state = State::new();
    let outcome = execute_program(&program, &mut state, &AsyncHost)
        .await
        .expect("program should run");
    let ExecutionOutcome::Finished(value) = outcome else {
        panic!("expected finish");
    };
    assert_eq!(value, Value::String("done".into()));

    let program = crate::parse(
        r#"
        process echo(value: str) { finish value }
        results = await [
          start echo(value: "left"),
          start echo(value: "right")
        ]
        submit [(results[0])?, (results[1])?]
        "#,
    )
    .expect("program should parse");
    let mut state = State::new();
    let outcome = execute_program(&program, &mut state, &AsyncHost)
        .await
        .expect("program should run");
    let ExecutionOutcome::Finished(value) = outcome else {
        panic!("expected finish");
    };
    assert_eq!(
        value,
        Value::List(vec![Value::String("left".into()), Value::String("right".into()),].into())
    );
}

// ------------------------------------------------------------------
//  Type literals: syntactic signatures with enum, list, nested, ref,
//  optional fields. See the top-level README for the full grammar.
// ------------------------------------------------------------------

/// Extract the inner JSON Schema wrapped by a `$lash_type` value.
fn unwrap_schema(value: &Value) -> &Record {
    crate::runtime::unwrap_type_value(value)
        .and_then(Value::as_record)
        .expect("Type value must unwrap to a schema record")
}

#[tokio::test(flavor = "current_thread")]
async fn type_scalar_schemas_const_fold_to_json_schema() {
    for (src, expected) in [
        ("submit Type { v: str }", "string"),
        ("submit Type { v: int }", "integer"),
        ("submit Type { v: float }", "number"),
        ("submit Type { v: bool }", "boolean"),
        ("submit Type { v: dict }", "object"),
    ] {
        let value = exec(src).await.expect("should succeed");
        let schema = unwrap_schema(&value);
        assert_eq!(schema["type"], Value::String("object".into()));
        let props = schema["properties"]
            .as_record()
            .expect("properties must be record");
        let v = props["v"].as_record().expect("field schema");
        assert_eq!(v["type"], Value::String(expected.into()));
        assert_eq!(
            schema["additionalProperties"],
            Value::Bool(false),
            "additionalProperties must be false for {src}",
        );
    }
}

#[tokio::test(flavor = "current_thread")]
async fn type_any_is_empty_schema() {
    let value = exec("submit Type { v: any }")
        .await
        .expect("should succeed");
    let schema = unwrap_schema(&value);
    let props = schema["properties"].as_record().expect("properties");
    let v = props["v"].as_record().expect("field schema");
    assert!(v.is_empty(), "any must be an empty JSON Schema");
}

#[tokio::test(flavor = "current_thread")]
async fn type_enum_produces_string_with_enum_array() {
    let value = exec(r#"submit Type { status: enum["ok", "err", "pending"] }"#)
        .await
        .expect("should succeed");
    let schema = unwrap_schema(&value);
    let status = schema["properties"].as_record().unwrap()["status"]
        .as_record()
        .expect("enum field schema");
    assert_eq!(status["type"], Value::String("string".into()));
    let Value::List(values) = &status["enum"] else {
        panic!("enum must be a list");
    };
    let strings: Vec<_> = values.iter().collect();
    assert_eq!(strings.len(), 3);
    assert_eq!(strings[0], &Value::String("ok".into()));
    assert_eq!(strings[2], &Value::String("pending".into()));
}

#[tokio::test(flavor = "current_thread")]
async fn type_list_schema_wraps_inner_type_as_items() {
    let value = exec("submit Type { tags: list[str] }")
        .await
        .expect("should succeed");
    let schema = unwrap_schema(&value);
    let tags = schema["properties"].as_record().unwrap()["tags"]
        .as_record()
        .expect("list field schema");
    assert_eq!(tags["type"], Value::String("array".into()));
    let items = tags["items"].as_record().expect("items schema");
    assert_eq!(items["type"], Value::String("string".into()));
}

#[tokio::test(flavor = "current_thread")]
async fn type_list_of_enum_preserves_nested_shape() {
    let value = exec(r#"submit Type { labels: list[enum["a", "b"]] }"#)
        .await
        .expect("should succeed");
    let schema = unwrap_schema(&value);
    let labels = schema["properties"].as_record().unwrap()["labels"]
        .as_record()
        .expect("list schema");
    let items = labels["items"].as_record().expect("enum item schema");
    assert_eq!(items["type"], Value::String("string".into()));
    assert!(matches!(items["enum"], Value::List(_)));
}

#[tokio::test(flavor = "current_thread")]
async fn type_nested_object_is_full_subschema() {
    let value = exec(
        r#"
        submit Type {
          title: str,
          meta: Type {
            pages: int,
            published: int
          }
        }
        "#,
    )
    .await
    .expect("should succeed");
    let schema = unwrap_schema(&value);
    let meta = schema["properties"].as_record().unwrap()["meta"]
        .as_record()
        .expect("nested object schema");
    assert_eq!(meta["type"], Value::String("object".into()));
    let sub_props = meta["properties"].as_record().unwrap();
    assert_eq!(
        sub_props["pages"].as_record().unwrap()["type"],
        Value::String("integer".into())
    );
    let required = match &meta["required"] {
        Value::List(items) => items,
        _ => panic!("required must be list"),
    };
    assert_eq!(required.len(), 2);
}

#[tokio::test(flavor = "current_thread")]
async fn type_optional_field_drops_from_required() {
    let value = exec("submit Type { a: str, b: int? }")
        .await
        .expect("should succeed");
    let schema = unwrap_schema(&value);
    let required = match &schema["required"] {
        Value::List(items) => items,
        _ => panic!("required must be list"),
    };
    assert_eq!(required.len(), 1);
    assert_eq!(required[0], Value::String("a".into()));
    // Optional field still appears in properties (just not required).
    let props = schema["properties"].as_record().unwrap();
    assert!(props.get("b").is_some());
}

#[tokio::test(flavor = "current_thread")]
async fn type_ref_resolves_previously_defined_type() {
    let src = r#"
        Inner = Type { count: int }
        Outer = Type { name: str, nested: Inner }
        submit Outer
    "#;
    let value = exec(src).await.expect("should succeed");
    let schema = unwrap_schema(&value);
    let nested = schema["properties"].as_record().unwrap()["nested"]
        .as_record()
        .expect("nested resolved schema");
    assert_eq!(nested["type"], Value::String("object".into()));
    let nested_props = nested["properties"].as_record().unwrap();
    assert_eq!(
        nested_props["count"].as_record().unwrap()["type"],
        Value::String("integer".into())
    );
}

#[tokio::test(flavor = "current_thread")]
async fn type_ref_to_non_type_value_is_type_error() {
    let err = exec(
        r#"
        Inner = { count: 5 }
        Outer = Type { nested: Inner }
        submit Outer
        "#,
    )
    .await
    .expect_err("should fail: Inner is not a Type value");
    assert!(
        matches!(err, RuntimeError::TypeError { .. }),
        "expected TypeError, got {err:?}"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn type_ref_with_undefined_name_is_undefined_variable() {
    let err = exec("submit Type { nested: MissingType }")
        .await
        .expect_err("unknown ref should fail");
    assert_eq!(
        err,
        RuntimeError::UndefinedVariable {
            name: "MissingType".to_string()
        }
    );
}

#[tokio::test(flavor = "current_thread")]
async fn compile_stats_count_const_folded_and_dynamic_literals() {
    let src = r#"
        Inner = Type { n: int }
        A = Type { x: str }
        B = Type { nested: Inner }
        submit B
    "#;
    let compiled = compile_source(src).expect("should compile");
    let stats = compiled.compile_stats();
    assert_eq!(stats.type_literals_total, 3);
    assert_eq!(
        stats.type_literals_const_folded, 2,
        "Inner and A are constant"
    );
    assert_eq!(stats.type_literals_dynamic, 1, "B references Inner");
    assert_eq!(stats.type_ref_sites, 1);
}

#[tokio::test(flavor = "current_thread")]
async fn profile_report_surfaces_resolve_type_ref_counts() {
    let src = r#"
        Inner = Type { n: int }
        Outer = Type { nested: Inner }
        limit = await tools.echo({ value: 1 })?
        numbers = push(range(limit), limit)
        checked = validate({ nested: { n: numbers[0] } }, Outer)
        submit checked
    "#;
    let compiled = compile_source(src).expect("should compile");
    let mut state = State::new();
    let (_outcome, report) = profile_compiled(&compiled, &mut state, &Host)
        .await
        .expect("profile should succeed");
    let names: Vec<_> = report.instruction_stats().iter().map(|s| s.name).collect();
    assert!(
        names.contains(&"resolve_type_ref"),
        "profile should track resolve_type_ref: {names:?}"
    );
    assert!(
        names.contains(&"wrap_type_literal"),
        "profile should track wrap_type_literal: {names:?}"
    );
    let builtin_names: Vec<_> = report.builtin_stats().iter().map(|s| s.name).collect();
    assert!(
        builtin_names.contains(&"validate"),
        "profile should track validate: {builtin_names:?}"
    );
    assert!(
        builtin_names.contains(&"range"),
        "profile should track range: {builtin_names:?}"
    );
    assert!(
        builtin_names.contains(&"push"),
        "profile should track push: {builtin_names:?}"
    );
    assert_eq!(report.compile_stats().type_literals_total, 2);
}

#[tokio::test(flavor = "current_thread")]
async fn type_literal_inside_resource_operation_args_passes_through_as_record() {
    struct CaptureHost {
        captured: std::sync::Mutex<Option<Value>>,
    }
    impl ExecutionHost for CaptureHost {
        async fn perform(&self, op: AbilityOp) -> Result<AbilityResult, ExecutionHostError> {
            match op {
                AbilityOp::ResourceOperation(operation) => {
                    if operation.operation == "spawn" {
                        let schema = operation
                            .args
                            .first()
                            .and_then(Value::as_record)
                            .and_then(|record| record.get("output"))
                            .cloned()
                            .expect("output arg must be present");
                        *self.captured.lock().unwrap() = Some(schema);
                        return Ok(AbilityResult::Value(Value::Null));
                    }
                    Err(ExecutionHostError::new(format!(
                        "unknown: {}",
                        operation.operation
                    )))
                }
                AbilityOp::Submit(value) | AbilityOp::Finish(value) | AbilityOp::Fail(value) => {
                    Ok(AbilityResult::Value(value))
                }
                _ => Err(ExecutionHostError::new("unsupported host ability")),
            }
        }
    }
    let host = CaptureHost {
        captured: std::sync::Mutex::new(None),
    };
    let program = crate::parse(
        r#"
        Shape = Type { name: str, tags: list[str] }
        await tools.spawn({ output: Shape })
        submit null
        "#,
    )
    .expect("should parse");
    let mut state = State::new();
    execute_program(&program, &mut state, &host)
        .await
        .expect("should run");

    let captured = host.captured.lock().unwrap().clone().expect("captured");
    let inner = crate::runtime::unwrap_type_value(&captured).expect("has $lash_type");
    let schema = inner.as_record().expect("schema record");
    assert_eq!(schema["type"], Value::String("object".into()));
}

#[tokio::test(flavor = "current_thread")]
async fn duplicate_field_name_is_parse_error() {
    let err = crate::parse("x = Type { a: str, a: int }").expect_err("duplicate field");
    let message = format!("{err}");
    assert!(message.contains("duplicate field"), "{message}");
}

#[tokio::test(flavor = "current_thread")]
async fn empty_enum_is_parse_error() {
    let err = crate::parse("x = Type { status: enum[] }").expect_err("empty enum");
    let message = format!("{err}");
    assert!(message.contains("enum"), "{message}");
}

#[tokio::test(flavor = "current_thread")]
async fn unknown_type_constructor_becomes_ref_not_error_at_parse() {
    // Unknown identifiers in type position are treated as refs; runtime
    // resolution is what errors out.
    let program = crate::parse("submit Type { x: Unknown }").expect("should parse as ref");
    let Expr::Block(expressions) = program.main else {
        panic!("program should be a block");
    };
    assert!(matches!(expressions.last(), Some(Expr::Submit(_))));
}

#[tokio::test(flavor = "current_thread")]
async fn lash_type_wrapper_survives_round_trip_through_json() {
    let value = exec("submit Type { n: int }")
        .await
        .expect("should succeed");
    // to_json + from_json must preserve the Type-ness.
    let json = crate::runtime::to_json(&value);
    let recovered = crate::runtime::from_json(json);
    let schema = crate::runtime::unwrap_type_value(&recovered)
        .and_then(Value::as_record)
        .expect("round-trip must preserve wrapper");
    assert_eq!(schema["type"], Value::String("object".into()));
}

// ----------------------------------------------------------------------------
// Projection propagation: `Value::Projected` carries through path expressions
// (Field / Index) but is stripped by computation. This is the lashlang side
// of the unified `seed:` channel for spawn_agent / continue_as: the host wire
// format (`{"__projected__": <inner>}`) only needs a wrapper to survive the
// JSON boundary, but path-rooted entry-values must already be projected at
// runtime so they serialize that way.
// ----------------------------------------------------------------------------

fn projected_record_bindings(name: &str, record: serde_json::Value) -> ProjectedBindings {
    let mut projected = ProjectedBindings::new();
    projected.insert(
        name,
        ProjectedValue::scalar(name.to_string(), crate::runtime::from_json(record)),
    );
    projected
}

#[tokio::test(flavor = "current_thread")]
async fn field_access_on_projected_record_returns_projected() {
    let projected = projected_record_bindings(
        "input",
        serde_json::json!({ "prompt": "hello", "depth": 3 }),
    );
    let (value, _) = exec_with_projected("submit input.prompt", &projected)
        .await
        .expect("projected field read");
    assert!(
        matches!(value, Value::Projected(_)),
        "expected `input.prompt` to stay projected, got {value:?}"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn nested_field_access_keeps_projection() {
    let projected =
        projected_record_bindings("cfg", serde_json::json!({ "options": { "timeout": 30 } }));
    let (value, _) = exec_with_projected("submit cfg.options.timeout", &projected)
        .await
        .expect("nested projected field read");
    assert!(
        matches!(value, Value::Projected(_)),
        "expected nested field to stay projected, got {value:?}"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn index_on_projected_list_returns_projected() {
    let projected =
        projected_record_bindings("items", serde_json::json!(["alpha", "beta", "gamma"]));
    let (value, _) = exec_with_projected("submit items[1]", &projected)
        .await
        .expect("projected index read");
    assert!(
        matches!(value, Value::Projected(_)),
        "expected `items[1]` to stay projected, got {value:?}"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn computation_strips_projection() {
    let projected = projected_record_bindings("input", serde_json::json!({ "n": 7 }));
    let (value, _) = exec_with_projected("submit input.n + 1", &projected)
        .await
        .expect("computed value");
    assert!(
        !matches!(value, Value::Projected(_)),
        "computation should strip projection, got {value:?}"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn record_literal_preserves_per_entry_projection() {
    let projected = projected_record_bindings("input", serde_json::json!({ "prompt": "hello" }));
    let (value, _) = exec_with_projected(
        "g = 42\nsubmit { proj: input.prompt, glob: g, lit: 99 }",
        &projected,
    )
    .await
    .expect("record literal");
    let Value::Record(record) = value else {
        panic!("expected record");
    };
    assert!(
        matches!(record.get("proj"), Some(Value::Projected(_))),
        "expected `proj` entry to stay projected, got {:?}",
        record.get("proj")
    );
    assert!(
        !matches!(record.get("glob"), Some(Value::Projected(_))),
        "global `glob` should not be projected, got {:?}",
        record.get("glob")
    );
    assert!(
        !matches!(record.get("lit"), Some(Value::Projected(_))),
        "literal `lit` should not be projected, got {:?}",
        record.get("lit")
    );
}

// ---------------------------------------------------------------------------
// Terminator-op routing through the handler.
//
// `submit`, `finish`, `fail` go through `host.perform` as `AbilityOp::Submit`,
// `Finish`, `Fail`. Default behavior is identity pass-through (the host returns
// the value unchanged and the VM unwinds with that value). The handler may
// transform the value or refuse with an `Err`; it cannot prevent unwind.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
enum TerminatorMode {
    Identity,
    Transform,
    Err,
    Unit,
}

struct TerminatorHost {
    mode: TerminatorMode,
    observed: Mutex<Vec<AbilityOp>>,
}

impl TerminatorHost {
    fn new(mode: TerminatorMode) -> Self {
        Self {
            mode,
            observed: Mutex::new(Vec::new()),
        }
    }
}

impl ExecutionHost for TerminatorHost {
    async fn perform(&self, op: AbilityOp) -> Result<AbilityResult, ExecutionHostError> {
        match op {
            AbilityOp::Submit(value) | AbilityOp::Finish(value) | AbilityOp::Fail(value) => {
                let observed = match &value {
                    Value::Number(n) => AbilityOp::Submit(Value::Number(*n)),
                    other => AbilityOp::Submit(other.clone()),
                };
                self.observed.lock().expect("observed").push(observed);
                match self.mode {
                    TerminatorMode::Identity => Ok(AbilityResult::Value(value)),
                    TerminatorMode::Transform => match value {
                        Value::Number(n) => Ok(AbilityResult::Value(Value::Number(n + 100.0))),
                        other => Ok(AbilityResult::Value(other)),
                    },
                    TerminatorMode::Err => Err(ExecutionHostError::new("handler refused")),
                    TerminatorMode::Unit => Ok(AbilityResult::Unit),
                }
            }
            _ => Err(ExecutionHostError::new("unsupported host ability")),
        }
    }
}

async fn run_with_terminator_host(
    source: &str,
    mode: TerminatorMode,
) -> (Result<ExecutionOutcome, RuntimeError>, Vec<AbilityOp>) {
    let host = TerminatorHost::new(mode);
    let program = crate::parse(source).expect("program should parse");
    let mut state = State::new();
    let outcome = execute_program(&program, &mut state, &host).await;
    let observed = host.observed.lock().expect("observed").clone();
    (outcome, observed)
}

async fn run_process_with_terminator_host(
    program: Program,
    mode: TerminatorMode,
) -> (Result<ExecutionOutcome, RuntimeError>, Vec<AbilityOp>) {
    let host = TerminatorHost::new(mode);
    let compiled = compile_program(&program);
    let mut state = State::new();
    let outcome = execute_compiled_process(&compiled, &mut state, &host).await;
    let observed = host.observed.lock().expect("observed").clone();
    (outcome, observed)
}

#[tokio::test(flavor = "current_thread")]
async fn submit_routes_through_host() {
    let (outcome, observed) = run_with_terminator_host("submit 7", TerminatorMode::Identity).await;
    assert_eq!(
        outcome.expect("submit should succeed"),
        ExecutionOutcome::Finished(Value::Number(7.0))
    );
    assert_eq!(observed.len(), 1, "host should observe one terminator op");
    assert!(matches!(observed[0], AbilityOp::Submit(Value::Number(n)) if n == 7.0));
}

#[tokio::test(flavor = "current_thread")]
async fn host_transforms_submit_value() {
    let (outcome, _) = run_with_terminator_host("submit 7", TerminatorMode::Transform).await;
    assert_eq!(
        outcome.expect("submit should succeed"),
        ExecutionOutcome::Finished(Value::Number(107.0)),
        "handler should transform the submit value before the VM unwinds"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn host_error_during_submit_propagates_as_runtime_error() {
    let (outcome, _) = run_with_terminator_host("submit 7", TerminatorMode::Err).await;
    let err = outcome.expect_err("host error should surface");
    let message = err.to_string();
    assert!(message.contains("submit failed"), "{message}");
    assert!(message.contains("handler refused"), "{message}");
}

#[tokio::test(flavor = "current_thread")]
async fn host_returning_unit_for_submit_errors_cleanly() {
    let (outcome, _) = run_with_terminator_host("submit 7", TerminatorMode::Unit).await;
    let err = outcome.expect_err("unit result should error");
    let message = err.to_string();
    assert!(message.contains("submit failed"), "{message}");
    assert!(message.contains("returned no value"), "{message}");
}

#[tokio::test(flavor = "current_thread")]
async fn finish_routes_through_host_in_process_mode() {
    let program = Program::block(vec![Expr::Finish(Some(Box::new(Expr::Number(7.0))))]);
    let (outcome, observed) =
        run_process_with_terminator_host(program, TerminatorMode::Transform).await;
    assert_eq!(
        outcome.expect("finish should succeed"),
        ExecutionOutcome::Finished(Value::Number(107.0))
    );
    assert_eq!(observed.len(), 1);
}

#[tokio::test(flavor = "current_thread")]
async fn fail_routes_through_host_and_carries_failed_outcome() {
    let program = Program::block(vec![Expr::Fail(Box::new(Expr::String("boom".into())))]);
    let (outcome, observed) =
        run_process_with_terminator_host(program, TerminatorMode::Identity).await;
    assert_eq!(
        outcome.expect("fail should produce an outcome"),
        ExecutionOutcome::Failed(Value::String("boom".into()))
    );
    assert_eq!(observed.len(), 1);
}

#[tokio::test(flavor = "current_thread")]
async fn host_can_transform_fail_value_while_keeping_failure_path() {
    let program = Program::block(vec![Expr::Fail(Box::new(Expr::Number(7.0)))]);
    let (outcome, _) = run_process_with_terminator_host(program, TerminatorMode::Transform).await;
    assert_eq!(
        outcome.expect("fail should produce an outcome"),
        ExecutionOutcome::Failed(Value::Number(107.0)),
        "transformed value should still arrive on the failure path"
    );
}
