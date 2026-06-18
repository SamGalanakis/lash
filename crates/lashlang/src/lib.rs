mod artifact;
mod ast;
mod builtins;
mod compile;
mod graph;
mod identity;
mod introspection;
mod lexer;
mod linker;
mod parser;
mod runtime;
mod source;
mod tracking;
mod trigger;

pub use artifact::{
    ArtifactStoreError, ContentHash, DurabilityTier, HostRequirements, HostRequirementsRef,
    InMemoryLashlangArtifactStore, LASHLANG_COMPILER_VERSION, LASHLANG_SEMANTIC_HASH_VERSION,
    LASHLANG_VM_ABI_VERSION, LashlangArtifactStore, ModuleArtifact, ModuleArtifactError,
    ModuleExports, ModuleRef, ProcessRef, canonical_program_ir,
    global_in_memory_lashlang_artifact_store, host_requirements_for_program,
};
pub use ast::{
    AssignPathStep, AssignTarget, BinaryOp, Declaration, Expr, ExprFolder, ExprVisitor,
    LabelMetadata, ProcessDecl, ProcessParam, ProcessStartExpr, Program, ResourceRefExpr, TypeDecl,
    TypeExpr, TypeField, UnaryOp, fold_expr_children, format_type_expr, walk_expr,
};
pub use compile::{
    ModuleCompileDiagnostic, ModuleCompileError, ModuleCompileOutput, ModuleCompileRequest,
    ModuleCompileStage, compile_module,
};
pub use graph::{
    LashlangMap, LashlangMapEdge, LashlangMapNode, LashlangMapOptions, map_lashlang_main,
    map_lashlang_process, static_graph_json,
};
pub use identity::{ProcessDefinitionIdentity, ProcessDefinitionIdentityError};
pub use introspection::{
    ModuleInstanceIntrospection, ModuleIntrospection, ModuleIntrospectionError,
    ModuleOperationIntrospection, NamedDataTypeIntrospection, ProcessInputIntrospection,
    ProcessIntrospection, ProcessSignalIntrospection, ResourceOperationIntrospection,
    ResourceTypeIntrospection, TriggerSourceIntrospection, TypeView, ValueConstructorIntrospection,
};
pub use lexer::{LexError, Span, Token, TokenKind, lex};
pub use linker::{
    LashlangAbilities, LashlangHostCatalog, LashlangHostCatalogError, LashlangHostEnvironment,
    LashlangLanguageFeatures, LinkError, LinkedModule, NamedDataType, NamedDataTypeError,
    ResourceOperationBinding, ResourceTypeCatalog, TriggerSourceBinding, ValueConstructorBinding,
};
pub use parser::{ParseError, parse};
pub use runtime::{
    AbilityOp, AbilityResult, CompileStats, CompiledLinkedProgram, CompiledProcessCache,
    CompiledProcessCacheKey, CompiledProgram, CompiledProgramCache, CompiledProgramCacheStats,
    ExecutableProgram, ExecutionEnvironment, ExecutionHost, ExecutionHostError, ExecutionMode,
    ExecutionOutcome, ExecutionScratch, ImageValue, LASH_HOST_DESCRIPTOR_TYPE_KEY,
    LASH_HOST_DESCRIPTOR_VALUE_KEY, LASH_HOST_REQUIREMENTS_REF_KEY, LASH_MODULE_REF_KEY,
    LASH_PROCESS_NAME_KEY, LASH_PROCESS_REF_KEY, LASH_PROCESS_VALUE_KEY, LASH_TYPE_KEY,
    LinkedProgramCache, LinkedProgramCacheError, ListValue, ProcessEvent, ProcessEventKind,
    ProcessSignal, ProcessStart, ProfileReport, ProfileStat, ProjectedBindingError,
    ProjectedBindings, ProjectedFuture, ProjectedHostDescriptor, ProjectedReadRequest,
    ProjectedReadResponse, ProjectedValue, Record, ResourceHandle, ResourceOperation,
    ResourceOperationBatch, ResourceOperationBatchResult, ResourceOperationResult, RuntimeError,
    RuntimeFailure, Sleep, SleepKind, Snapshot, State, Value, compile, compile_linked,
    compile_linked_process, compile_module_artifact_process, compile_process, execute, from_json,
    prewarm, unwrap_type_value,
};
pub use source::{
    CanonicalSourceError, canonical_process_source, canonical_process_source_with_requirements,
    canonical_program_source, canonical_program_source_with_requirements,
};
pub use tracking::{
    LashlangBranchSite, LashlangExecutionCallSite, LashlangExecutionChild,
    LashlangExecutionObservation, LashlangExecutionSite, ProcessBranchSelection, process_ref_key,
};
pub use trigger::{
    HostDescriptor, HostDescriptorError, LASH_TRIGGER_EVENT_KEY, TriggerCancelRequest,
    TriggerCompatibility, TriggerCompatibilityError, TriggerCompatibilityRequest,
    TriggerHostOperation, TriggerInputBinding, TriggerInputTemplate, TriggerListRequest,
    TriggerRegistrationRequest, add_trigger_resource_operations, cancel_call_args,
    check_trigger_compatibility, event_type_for_source, is_trigger_resource_type, list_call_args,
    register_call_args, trigger_event_placeholder_expr,
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
        ParseError::Expected { expected, .. } if expected.contains("type literals must start") => {
            Some("write nested object types as `Type { field: type }`")
        }
        ParseError::DeclarativeTriggerRemoved { .. } => Some(
            "construct a host-provided trigger source value and call the trigger registry register operation",
        ),
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
                AbilityOp::ResourceOperationBatch(batch) => Ok(
                    AbilityResult::ResourceOperationBatch(ResourceOperationBatchResult {
                        results: batch
                            .operations
                            .into_iter()
                            .map(|operation| {
                                if operation.operation == "anything" {
                                    ResourceOperationResult::Value(Value::Record(
                                        std::sync::Arc::new(Record::from_iter([(
                                            "ok".to_string(),
                                            Value::Bool(true),
                                        )])),
                                    ))
                                } else {
                                    ResourceOperationResult::Error(ExecutionHostError::new(
                                        "unsupported host ability",
                                    ))
                                }
                            })
                            .collect(),
                    }),
                ),
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
        let same_ast = cache
            .get_or_compile("submit 1\n")
            .expect("compile source-distinct program");
        let other = cache.get_or_compile("submit 2").expect("compile second");
        let third = cache.get_or_compile("submit 3").expect("compile third");

        assert!(std::sync::Arc::ptr_eq(&first, &second));
        assert!(!std::sync::Arc::ptr_eq(&first, &same_ast));
        assert!(!std::sync::Arc::ptr_eq(&first, &other));
        assert!(!std::sync::Arc::ptr_eq(&other, &third));

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 4);
        assert_eq!(stats.evictions, 2);
        assert_eq!(stats.entries, 2);
        assert_eq!(stats.capacity, 2);
    }

    #[test]
    fn linked_program_cache_reuses_source_when_host_environment_satisfies_requirements() {
        let source = r#"submit (await tools.read_file({ path: "." }))?"#;
        let base_environment = LashlangHostEnvironment::new(
            LashlangHostCatalog::tool_default(["read_file"]),
            LashlangAbilities::default(),
        );
        let extra_environment = LashlangHostEnvironment::new(
            LashlangHostCatalog::tool_default(["read_file", "unrelated"]),
            LashlangAbilities::default(),
        );
        let mut cache = LinkedProgramCache::with_capacity(2);

        let first = cache
            .get_or_compile(source, &base_environment)
            .expect("compile first linked program");
        let second = cache
            .get_or_compile(source, &base_environment)
            .expect("reuse same surface");
        let extra = cache
            .get_or_compile(source, &extra_environment)
            .expect("reuse when unrelated tools are added");

        assert!(std::sync::Arc::ptr_eq(&first, &second));
        assert!(std::sync::Arc::ptr_eq(&first, &extra));
        assert_eq!(
            first.linked_module().host_requirements_ref,
            extra.linked_module().host_requirements_ref
        );

        let stats = cache.stats();
        assert_eq!(stats.hits, 2);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.evictions, 0);
        assert_eq!(stats.entries, 1);
    }

    #[test]
    fn linked_program_cache_keeps_source_and_host_requirements_distinct() {
        let source = r#"submit (await tools.read_file({ path: "." }))?"#;
        let base_environment = LashlangHostEnvironment::new(
            LashlangHostCatalog::tool_default(["read_file"]),
            LashlangAbilities::default(),
        );
        let mut changed_resources = LashlangHostCatalog::new();
        changed_resources.add_module_operation(
            ["tools"],
            "Tools",
            "read_file",
            "read_file_v2",
            TypeExpr::Any,
            TypeExpr::Any,
        );
        let changed_environment =
            LashlangHostEnvironment::new(changed_resources, LashlangAbilities::default());
        let missing_environment = LashlangHostEnvironment::new(
            LashlangHostCatalog::tool_default(["echo"]),
            LashlangAbilities::default(),
        );
        let mut cache = LinkedProgramCache::with_capacity(4);

        let first = cache
            .get_or_compile(source, &base_environment)
            .expect("compile first linked program");
        let newline = cache
            .get_or_compile(&format!("{source}\n"), &base_environment)
            .expect("compile source-distinct linked program");
        let changed = cache
            .get_or_compile(source, &changed_environment)
            .expect("compile changed surface requirement");
        let missing = cache
            .get_or_compile(source, &missing_environment)
            .expect_err("missing resource operation should not reuse cached program");

        assert!(!std::sync::Arc::ptr_eq(&first, &newline));
        assert!(!std::sync::Arc::ptr_eq(&first, &changed));
        assert_ne!(
            first.linked_module().host_requirements_ref,
            changed.linked_module().host_requirements_ref
        );
        assert!(matches!(
            missing,
            LinkedProgramCacheError::Link(LinkError::UnknownResourceOperation {
                operation,
                ..
            }) if operation == "read_file"
        ));

        let stats = cache.stats();
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 4);
        assert_eq!(stats.evictions, 0);
        assert_eq!(stats.entries, 3);
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
        let linked = LinkedModule::link(
            parse("v = await tools.anything({})? submit v").expect("source should parse"),
            LashlangHostEnvironment::new(
                LashlangHostCatalog::tool_default(["anything"]),
                LashlangAbilities::default(),
            ),
        )
        .expect("source should link");
        let compiled = compile_linked(&linked);
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
