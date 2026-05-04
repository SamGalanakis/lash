mod ast;
mod lexer;
mod parser;
mod runtime;

pub use ast::{
    AssignPathStep, AssignTarget, BinaryOp, CallExpr, Expr, NamedParallelBranch, ParallelBranches,
    Program, Stmt, TypeExpr, TypeField, UnaryOp,
};
pub use lexer::{LexError, Span, Token, TokenKind, lex};
pub use parser::{ParseError, parse};
pub use runtime::{
    CompileStats, CompiledProgram, CompiledProgramCache, CompiledProgramCacheStats,
    ExecutionOutcome, ExecutionScratch, ImageValue, LASH_TYPE_KEY, ProfileReport, ProfileStat,
    Record, RuntimeError, RuntimeFailure, Snapshot, State, ToolHost, ToolHostCall, ToolHostError,
    Value, compile_program, compile_source, execute_compiled, execute_compiled_traced,
    execute_compiled_traced_with_scratch, execute_compiled_with_scratch, execute_program, prewarm,
    profile_compiled, profile_compiled_with_scratch, unwrap_type_value,
};

pub async fn execute<H: ToolHost>(
    source: &str,
    state: &mut State,
    host: &H,
) -> Result<ExecutionOutcome, ExecuteError> {
    let program = parse(source)?;
    execute_program(&program, state, host)
        .await
        .map_err(ExecuteError::Runtime)
}

pub async fn execute_with_diagnostics<H: ToolHost>(
    source: &str,
    state: &mut State,
    host: &H,
) -> Result<ExecutionOutcome, ExecuteError> {
    let compiled = compile_source(source)?;
    execute_compiled_traced(&compiled, state, host)
        .await
        .map_err(|failure| {
            ExecuteError::Runtime(RuntimeError::ValueError {
                message: format_runtime_diagnostic(source, &failure.error, failure.span),
            })
        })
}

pub fn format_runtime_diagnostic(source: &str, error: &RuntimeError, span: Option<Span>) -> String {
    let Some(span) = span else {
        return error.to_string();
    };
    let (line, column, snippet) = line_column_snippet(source, span.start);
    format!("{error}\n--> line {line}, column {column}\n{snippet}")
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
        source[line_start..line_end].trim_end().to_string(),
    )
}

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum ExecuteError {
    #[error(transparent)]
    Parse(#[from] ParseError),
    #[error(transparent)]
    Runtime(#[from] RuntimeError),
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Host;

    impl ToolHost for Host {
        async fn call(&self, _name: String, _args: Record) -> Result<Value, ToolHostError> {
            Ok(Value::Null)
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn execute_wraps_parse_errors() {
        let mut state = State::new();
        let err = execute("if true", &mut state, &Host)
            .await
            .expect_err("parse should fail");
        assert!(matches!(err, ExecuteError::Parse(_)));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn execute_wraps_runtime_errors() {
        let mut state = State::new();
        let err = execute("submit missing", &mut state, &Host)
            .await
            .expect_err("runtime should fail");
        assert!(matches!(err, ExecuteError::Runtime(_)));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn execute_with_diagnostics_includes_source_location() {
        let mut state = State::new();
        let err = execute_with_diagnostics("x = 1\nsubmit missing", &mut state, &Host)
            .await
            .expect_err("runtime should fail");
        let message = diagnostic_message(err);
        assert!(message.contains("unknown name `missing`"), "{message}");
        assert!(message.contains("--> line 2, column 1"), "{message}");
        assert!(message.contains("submit missing"), "{message}");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn compile_source_prewarm_and_traced_scratch_execution_work_together() {
        prewarm();
        let compiled = compile_source("submit 7").expect("source should compile");
        let mut state = State::new();
        let mut scratch = ExecutionScratch::new();
        let outcome =
            execute_compiled_traced_with_scratch(&compiled, &mut state, &Host, &mut scratch)
                .await
                .expect("execution should succeed");
        assert_eq!(outcome, ExecutionOutcome::Finished(Value::Number(7.0)));
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
            let mut state = State::new();
            let err = execute_with_diagnostics(source, &mut state, &Host)
                .await
                .expect_err("runtime should fail");
            let message = diagnostic_message(err);
            assert!(message.contains(expected_error), "{message}");
            assert!(message.contains("--> line 2, column 1"), "{message}");
            assert!(message.contains(expected_snippet), "{message}");
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn execute_success_path_uses_host() {
        let mut state = State::new();
        let outcome = execute("v = call anything {} submit v", &mut state, &Host)
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
        let mut state = State::new();
        let outcome = execute("submit", &mut state, &Host)
            .await
            .expect("should succeed");
        let ExecutionOutcome::Finished(value) = outcome else {
            panic!("expected finish");
        };
        assert_eq!(value, Value::Null);
    }

    fn diagnostic_message(err: ExecuteError) -> String {
        let ExecuteError::Runtime(RuntimeError::ValueError { message }) = err else {
            panic!("expected diagnostic runtime error");
        };
        message
    }
}
