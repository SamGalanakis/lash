mod ast;
mod lexer;
mod parser;
mod runtime;

pub use ast::{BinaryOp, CallExpr, Expr, Program, Stmt, UnaryOp};
pub use lexer::{LexError, Span, Token, TokenKind, lex};
pub use parser::{ParseError, parse};
pub use runtime::{
    CompiledProgram, ExecutionOutcome, ProfileReport, ProfileStat, Record, RuntimeError, Snapshot,
    State, ToolHost, ToolHostError, Value, compile_program, execute_compiled, execute_program,
    profile_compiled,
};

pub fn execute<H: ToolHost>(
    source: &str,
    state: &mut State,
    host: &H,
) -> Result<ExecutionOutcome, ExecuteError> {
    let program = parse(source)?;
    execute_program(&program, state, host).map_err(ExecuteError::Runtime)
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
        fn call(&self, _name: &str, _args: &Record) -> Result<Value, ToolHostError> {
            Ok(Value::Null)
        }
    }

    #[test]
    fn execute_wraps_parse_errors() {
        let mut state = State::new();
        let err = execute("if true", &mut state, &Host).expect_err("parse should fail");
        assert!(matches!(err, ExecuteError::Parse(_)));
    }

    #[test]
    fn execute_wraps_runtime_errors() {
        let mut state = State::new();
        let err = execute("finish missing", &mut state, &Host).expect_err("runtime should fail");
        assert!(matches!(err, ExecuteError::Runtime(_)));
    }

    #[test]
    fn execute_success_path_uses_host() {
        let mut state = State::new();
        let outcome =
            execute("v = call anything {} finish v", &mut state, &Host).expect("should succeed");
        let ExecutionOutcome::Finished(value) = outcome else {
            panic!("expected finish");
        };
        assert_eq!(
            value.as_record().expect("tool result should be record")["ok"],
            Value::Bool(true)
        );
    }
}
