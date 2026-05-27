mod artifact;
mod ast;
mod graph;
mod lexer;
mod linker;
mod parser;
mod runtime;

pub use artifact::{
    ArtifactStoreError, ContentHash, InMemoryLashlangArtifactStore, LASHLANG_COMPILER_VERSION,
    LASHLANG_SEMANTIC_HASH_VERSION, LASHLANG_VM_ABI_VERSION, LashlangArtifactStore, ModuleArtifact,
    ModuleArtifactError, ModuleExports, ModuleRef, ProcessRef, RequiredSurfaceRef,
    SurfaceRequirements, canonical_program_ir, global_in_memory_lashlang_artifact_store,
    surface_requirements_for_program,
};
pub use ast::{
    AssignPathStep, AssignTarget, BinaryOp, Declaration, Expr, ProcessDecl, ProcessParam,
    ProcessStartExpr, Program, ResourceRefExpr, ScheduleCadence, ScheduleDecl, TriggerArg,
    TriggerDecl, TriggerSource, TypeDecl, TypeExpr, TypeField, UnaryOp, format_type_expr,
};
pub use graph::{
    ProcessMap, ProcessMapEdge, ProcessMapNode, ProcessMapOptions, linked_static_graph_json,
    map_process, static_graph_json,
};
pub use lexer::{LexError, Span, Token, TokenKind, lex};
pub use linker::{
    LashlangAbilities, LashlangScheduleAbilities, LashlangSurface, LinkError, LinkedModule,
    ResourceCatalog, ResourceOperationBinding, ResourceTypeCatalog, TriggerEventBinding,
};
pub use parser::{ParseError, parse};
pub use runtime::{
    AbilityOp, AbilityResult, CompileStats, CompiledProcessCache, CompiledProcessCacheKey,
    CompiledProgram, CompiledProgramCache, CompiledProgramCacheStats, ExecutableProgram,
    ExecutionEnvironment, ExecutionHost, ExecutionHostError, ExecutionMode, ExecutionOutcome,
    ExecutionScratch, ImageValue, LASH_TYPE_KEY, ListValue, ProcessEvent, ProcessEventKind,
    ProcessSignal, ProcessSleep, ProcessSleepKind, ProcessStart, ProfileReport, ProfileStat,
    ProjectedBindingError, ProjectedBindings, ProjectedFuture, ProjectedHostValue,
    ProjectedReadRequest, ProjectedReadResponse, ProjectedValue, Record, ResourceHandle,
    ResourceOperation, RuntimeError, RuntimeFailure, Snapshot, State, Value, compile,
    compile_linked, compile_linked_process, compile_module_artifact_process, compile_process,
    execute, from_json, prewarm, unwrap_type_value,
};

pub fn format_parse_diagnostic(source: &str, error: &ParseError) -> String {
    format_source_diagnostic(
        source,
        error.offset(),
        &error.to_string(),
        parse_hint(error),
    )
}

pub fn format_runtime_diagnostic(source: &str, error: &RuntimeError, span: Option<Span>) -> String {
    let Some(span) = span else {
        return format_message_with_hint(&error.to_string(), runtime_hint(error));
    };
    format_source_diagnostic(source, span.start, &error.to_string(), runtime_hint(error))
}

pub fn format_link_diagnostic(source: &str, error: &LinkError) -> String {
    match error.span() {
        Some(span) => format_source_diagnostic(source, span.start, &error.to_string(), None),
        None => error.to_string(),
    }
}

fn format_source_diagnostic(
    source: &str,
    offset: usize,
    message: &str,
    hint: Option<&'static str>,
) -> String {
    let (line, column, source_line) = line_column_snippet(source, offset);
    let caret_pad = " ".repeat(column.saturating_sub(1));
    let mut diagnostic =
        format!("{message}\n--> line {line}, column {column}\n{source_line}\n{caret_pad}^");
    if let Some(hint) = hint {
        diagnostic.push_str("\nhint: ");
        diagnostic.push_str(hint);
    }
    diagnostic
}

fn format_message_with_hint(message: &str, hint: Option<&'static str>) -> String {
    let mut diagnostic = message.to_string();
    if let Some(hint) = hint {
        diagnostic.push_str("\nhint: ");
        diagnostic.push_str(hint);
    }
    diagnostic
}

fn parse_hint(error: &ParseError) -> Option<&'static str> {
    match error {
        ParseError::Unexpected { found, .. } if found == "`if`" => {
            Some("use `cond ? yes : no` for inline conditionals")
        }
        ParseError::Unexpected { found, .. } if found == "`for`" => Some(
            "`for` is a statement. Put it on its own line, not inside an expression or record literal.",
        ),
        ParseError::UnsupportedLoop {
            keyword: "while", ..
        } => Some("use bounded `for` loops over ranges or lists"),
        ParseError::Expected { expected, .. } if expected.contains("type literals must start") => {
            Some("write nested object types as `Type { field: type }`")
        }
        ParseError::TriggerBodyNotAllowed { .. } => {
            Some("bind every target process parameter in `-> process_name(param: event)`")
        }
        _ => None,
    }
}

fn runtime_hint(error: &RuntimeError) -> Option<&'static str> {
    match error {
        RuntimeError::TypeError { message } | RuntimeError::ValueError { message } => {
            if message.starts_with("`?` unwrapped failed tool result:") {
                return Some(
                    "remove `?` and inspect `.ok` or `.error` when you need to handle failures",
                );
            }
            if message.contains("read-only projected binding") {
                return Some("copy the projected value into a new variable before changing it");
            }
            if message == "`validate` requires a Type literal as the second argument" {
                return Some("pass `Type { ... }` or a variable that holds a Type literal");
            }
            None
        }
        _ => None,
    }
}

fn line_column_snippet(source: &str, offset: usize) -> (usize, usize, String) {
    let offset = offset.min(source.len());
    let mut line = 1usize;
    let mut line_start = 0usize;
    for (idx, ch) in source.char_indices() {
        if idx >= offset {
            break;
        }
        if ch == '\n' {
            line += 1;
            line_start = idx + ch.len_utf8();
        }
    }
    let column = source[line_start..offset].chars().count() + 1;
    let line_end = source[offset..]
        .find('\n')
        .map(|rel| offset + rel)
        .unwrap_or(source.len());
    (
        line,
        column,
        source[line_start..line_end]
            .trim_end_matches('\r')
            .to_string(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Host;

    impl ExecutionHost for Host {
        async fn perform(&self, op: AbilityOp) -> Result<AbilityResult, ExecutionHostError> {
            match op {
                AbilityOp::ResourceOperation(operation) if operation.operation == "anything" => {
                    Ok(AbilityResult::Value(Value::Record(std::sync::Arc::new(
                        Record::from_iter([("ok".to_string(), Value::Bool(true))]),
                    ))))
                }
                AbilityOp::Submit(value) | AbilityOp::Finish(value) | AbilityOp::Fail(value) => {
                    Ok(AbilityResult::Value(value))
                }
                _ => Ok(AbilityResult::Value(Value::Null)),
            }
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn compile_reports_parse_errors() {
        let err = compile("if true").expect_err("parse should fail");
        assert!(matches!(err, ParseError::Expected { .. }));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn execute_reports_runtime_errors() {
        let compiled = compile("submit missing").expect("source should compile");
        let mut state = State::new();
        let err = execute(&compiled, &mut state, &Host)
            .await
            .expect_err("runtime should fail");
        assert!(matches!(err, RuntimeError::UndefinedVariable { .. }));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn traced_environment_records_source_location() {
        let source = "x = 1\nsubmit missing";
        let compiled = compile(source).expect("source should compile");
        let mut state = State::new();
        let env = ExecutionEnvironment::new(&Host).traced();
        execute(&compiled, &mut state, &env)
            .await
            .expect_err("runtime should fail");
        let failure = env
            .take_runtime_failure()
            .expect("traced host should receive runtime failure");
        let message = format_runtime_diagnostic(source, &failure.error, failure.span);
        assert!(message.contains("unknown name `missing`"), "{message}");
        assert!(message.contains("--> line 2, column 1"), "{message}");
        assert!(message.contains("submit missing"), "{message}");
        assert!(message.contains("^"), "{message}");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn compile_prewarm_and_environment_scratch_execution_work_together() {
        prewarm();
        let compiled = compile("submit 7").expect("source should compile");
        let mut state = State::new();
        let env = ExecutionEnvironment::new(&Host)
            .traced()
            .with_scratch(ExecutionScratch::new());
        let outcome = execute(&compiled, &mut state, &env)
            .await
            .expect("execution should succeed");
        assert_eq!(outcome, ExecutionOutcome::Finished(Value::Number(7.0)));
        assert!(env.take_recycled_scratch().is_some());
    }

    #[test]
    fn compiled_program_cache_reuses_source_and_tracks_lru_stats() {
        let mut cache = CompiledProgramCache::with_capacity(2);
        let first = cache.get_or_compile("submit 1").expect("compile first");
        let second = cache.get_or_compile("submit 1").expect("compile cache hit");
        let other = cache.get_or_compile("submit 2").expect("compile second");
        let third = cache.get_or_compile("submit 3").expect("compile third");

        assert!(std::sync::Arc::ptr_eq(&first, &second));
        assert!(!std::sync::Arc::ptr_eq(&first, &other));
        assert!(!std::sync::Arc::ptr_eq(&other, &third));

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 3);
        assert_eq!(stats.evictions, 1);
        assert_eq!(stats.entries, 2);
        assert_eq!(stats.capacity, 2);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn execute_with_diagnostics_covers_representative_runtime_failures() {
        let cases = [
            (
                "x = 1\nsubmit ({ ok: false, error: \"boom\" })?",
                "`?` unwrapped failed tool result: boom",
                "submit ({ ok: false, error: \"boom\" })?",
            ),
            (
                "x = 1\nsubmit len(true)",
                "`len` requires a string, list, record, or null",
                "submit len(true)",
            ),
            (
                "x = 1\nsubmit \"text\".field",
                "can't read `.field` from string",
                "submit \"text\".field",
            ),
            ("x = 1\nsubmit 7[0]", "can't index number", "submit 7[0]"),
        ];

        for (source, expected_error, expected_snippet) in cases {
            let compiled = compile(source).expect("source should compile");
            let mut state = State::new();
            let env = ExecutionEnvironment::new(&Host).traced();
            execute(&compiled, &mut state, &env)
                .await
                .expect_err("runtime should fail");
            let failure = env
                .take_runtime_failure()
                .expect("traced host should receive runtime failure");
            let message = format_runtime_diagnostic(source, &failure.error, failure.span);
            assert!(message.contains(expected_error), "{message}");
            assert!(message.contains("--> line 2, column 1"), "{message}");
            assert!(message.contains(expected_snippet), "{message}");
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn execute_success_path_uses_host() {
        let compiled = compile("v = await TOOL.default.anything({})? submit v")
            .expect("source should compile");
        let mut state = State::new();
        let outcome = execute(&compiled, &mut state, &Host)
            .await
            .expect("should succeed");
        let ExecutionOutcome::Finished(value) = outcome else {
            panic!("expected finish");
        };
        assert_eq!(
            value.as_record().expect("tool result should be record")["ok"],
            Value::Bool(true)
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn execute_allows_bare_finish() {
        let compiled = compile("submit").expect("source should compile");
        let mut state = State::new();
        let outcome = execute(&compiled, &mut state, &Host)
            .await
            .expect("should succeed");
        let ExecutionOutcome::Finished(value) = outcome else {
            panic!("expected finish");
        };
        assert_eq!(value, Value::Null);
    }
}
