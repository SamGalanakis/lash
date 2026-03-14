use crate::ast::{BinaryOp, CallExpr, Expr, Program, Stmt, UnaryOp};
use crate::lexer::{LexError, Span, Token, TokenKind, lex};
use thiserror::Error;

#[derive(Debug, Error, PartialEq)]
pub enum ParseError {
    #[error(transparent)]
    Lex(#[from] LexError),
    #[error("expected {expected}, found {found}")]
    Expected {
        expected: &'static str,
        found: String,
        span: Span,
    },
    #[error("unexpected {found}")]
    Unexpected { found: String, span: Span },
}

impl ParseError {
    pub fn offset(&self) -> usize {
        match self {
            Self::Lex(err) => err.offset(),
            Self::Expected { span, .. } | Self::Unexpected { span, .. } => span.start,
        }
    }
}

pub fn parse(source: &str) -> Result<Program, ParseError> {
    let tokens = lex(source)?;
    Parser { tokens, index: 0 }.parse_program()
}

struct Parser {
    tokens: Vec<Token>,
    index: usize,
}

impl Parser {
    fn parse_program(&mut self) -> Result<Program, ParseError> {
        let mut statements = Vec::new();
        while !self.at_eof() {
            statements.push(self.parse_stmt()?);
        }
        Ok(Program { statements })
    }

    fn parse_stmt(&mut self) -> Result<Stmt, ParseError> {
        match self.peek_kind() {
            TokenKind::If => self.parse_if(),
            TokenKind::For => self.parse_for(),
            TokenKind::Parallel => self.parse_parallel(),
            TokenKind::Finish => self.parse_finish(),
            TokenKind::Observe => self.parse_observe(),
            TokenKind::Call => Ok(Stmt::Call(self.parse_call_expr()?)),
            TokenKind::Ident(_) if self.peek_assign() => self.parse_assign(),
            _ => Ok(Stmt::Expr(self.parse_expr()?)),
        }
    }

    fn parse_assign(&mut self) -> Result<Stmt, ParseError> {
        let name = self.expect_ident()?;
        self.expect_exact(TokenKind::Equal, "`=`")?;
        let expr = self.parse_expr()?;
        Ok(Stmt::Assign { name, expr })
    }

    fn parse_if(&mut self) -> Result<Stmt, ParseError> {
        self.bump();
        let condition = self.parse_expr()?;
        let then_block = self.parse_block()?;
        let else_block = if matches!(self.peek_kind(), TokenKind::Else) {
            self.bump();
            if matches!(self.peek_kind(), TokenKind::If) {
                vec![self.parse_if()?]
            } else {
                self.parse_block()?
            }
        } else {
            Vec::new()
        };
        Ok(Stmt::If {
            condition,
            then_block,
            else_block,
        })
    }

    fn parse_for(&mut self) -> Result<Stmt, ParseError> {
        self.bump();
        let binding = self.expect_ident()?;
        self.expect_exact(TokenKind::In, "`in`")?;
        let iterable = self.parse_expr()?;
        let body = self.parse_block()?;
        Ok(Stmt::For {
            binding,
            iterable,
            body,
        })
    }

    fn parse_parallel(&mut self) -> Result<Stmt, ParseError> {
        self.bump();
        let branches = self.parse_parallel_branches()?;
        Ok(Stmt::Parallel { branches })
    }

    fn parse_parallel_branches(&mut self) -> Result<Vec<Stmt>, ParseError> {
        self.parse_block()
    }

    fn parse_finish(&mut self) -> Result<Stmt, ParseError> {
        self.bump();
        Ok(Stmt::Finish(self.parse_expr()?))
    }

    fn parse_observe(&mut self) -> Result<Stmt, ParseError> {
        self.bump();
        Ok(Stmt::Observe(self.parse_expr()?))
    }

    fn parse_block(&mut self) -> Result<Vec<Stmt>, ParseError> {
        self.expect_exact(TokenKind::LBrace, "`{`")?;
        let mut statements = Vec::new();
        while !matches!(self.peek_kind(), TokenKind::RBrace | TokenKind::Eof) {
            statements.push(self.parse_stmt()?);
        }
        self.expect_exact(TokenKind::RBrace, "`}`")?;
        Ok(statements)
    }

    fn parse_expr(&mut self) -> Result<Expr, ParseError> {
        self.parse_ternary()
    }

    fn parse_ternary(&mut self) -> Result<Expr, ParseError> {
        let condition = self.parse_or()?;
        if !matches!(self.peek_kind(), TokenKind::Question) {
            return Ok(condition);
        }
        self.bump();
        let then_expr = self.parse_expr()?;
        self.expect_exact(TokenKind::Colon, "`:`")?;
        let else_expr = self.parse_expr()?;
        Ok(Expr::Conditional {
            condition: Box::new(condition),
            then_expr: Box::new(then_expr),
            else_expr: Box::new(else_expr),
        })
    }

    fn parse_or(&mut self) -> Result<Expr, ParseError> {
        let mut expr = self.parse_and()?;
        while matches!(self.peek_kind(), TokenKind::Or | TokenKind::OrOr) {
            self.bump();
            let right = self.parse_and()?;
            expr = Expr::Binary {
                left: Box::new(expr),
                op: BinaryOp::Or,
                right: Box::new(right),
            };
        }
        Ok(expr)
    }

    fn parse_and(&mut self) -> Result<Expr, ParseError> {
        let mut expr = self.parse_compare()?;
        while matches!(self.peek_kind(), TokenKind::And | TokenKind::AndAnd) {
            self.bump();
            let right = self.parse_compare()?;
            expr = Expr::Binary {
                left: Box::new(expr),
                op: BinaryOp::And,
                right: Box::new(right),
            };
        }
        Ok(expr)
    }

    fn parse_compare(&mut self) -> Result<Expr, ParseError> {
        let mut expr = self.parse_add()?;
        loop {
            let op = match self.peek_kind() {
                TokenKind::DoubleEqual => BinaryOp::Equal,
                TokenKind::BangEqual => BinaryOp::NotEqual,
                TokenKind::Less => BinaryOp::Less,
                TokenKind::LessEqual => BinaryOp::LessEqual,
                TokenKind::Greater => BinaryOp::Greater,
                TokenKind::GreaterEqual => BinaryOp::GreaterEqual,
                _ => break,
            };
            self.bump();
            let right = self.parse_add()?;
            expr = Expr::Binary {
                left: Box::new(expr),
                op,
                right: Box::new(right),
            };
        }
        Ok(expr)
    }

    fn parse_add(&mut self) -> Result<Expr, ParseError> {
        let mut expr = self.parse_mul()?;
        loop {
            let op = match self.peek_kind() {
                TokenKind::Plus => BinaryOp::Add,
                TokenKind::Minus => BinaryOp::Subtract,
                _ => break,
            };
            self.bump();
            let right = self.parse_mul()?;
            expr = Expr::Binary {
                left: Box::new(expr),
                op,
                right: Box::new(right),
            };
        }
        Ok(expr)
    }

    fn parse_mul(&mut self) -> Result<Expr, ParseError> {
        let mut expr = self.parse_unary()?;
        loop {
            let op = match self.peek_kind() {
                TokenKind::Star => BinaryOp::Multiply,
                TokenKind::Slash => BinaryOp::Divide,
                TokenKind::Percent => BinaryOp::Modulo,
                _ => break,
            };
            self.bump();
            let right = self.parse_unary()?;
            expr = Expr::Binary {
                left: Box::new(expr),
                op,
                right: Box::new(right),
            };
        }
        Ok(expr)
    }

    fn parse_unary(&mut self) -> Result<Expr, ParseError> {
        match self.peek_kind() {
            TokenKind::Minus => {
                self.bump();
                Ok(Expr::Unary {
                    op: UnaryOp::Negate,
                    expr: Box::new(self.parse_unary()?),
                })
            }
            TokenKind::Not => {
                self.bump();
                Ok(Expr::Unary {
                    op: UnaryOp::Not,
                    expr: Box::new(self.parse_unary()?),
                })
            }
            TokenKind::Bang => {
                self.bump();
                Ok(Expr::Unary {
                    op: UnaryOp::Not,
                    expr: Box::new(self.parse_unary()?),
                })
            }
            _ => self.parse_postfix(),
        }
    }

    fn parse_postfix(&mut self) -> Result<Expr, ParseError> {
        let mut expr = self.parse_primary()?;
        loop {
            match self.peek_kind() {
                TokenKind::Dot => {
                    self.bump();
                    let field = self.expect_ident()?;
                    expr = Expr::Field {
                        target: Box::new(expr),
                        field,
                    };
                }
                TokenKind::LBracket => {
                    self.bump();
                    let index = self.parse_expr()?;
                    self.expect_exact(TokenKind::RBracket, "`]`")?;
                    expr = Expr::Index {
                        target: Box::new(expr),
                        index: Box::new(index),
                    };
                }
                _ => break,
            }
        }
        Ok(expr)
    }

    fn parse_primary(&mut self) -> Result<Expr, ParseError> {
        match self.peek_kind() {
            TokenKind::Null => {
                self.bump();
                Ok(Expr::Null)
            }
            TokenKind::True => {
                self.bump();
                Ok(Expr::Bool(true))
            }
            TokenKind::False => {
                self.bump();
                Ok(Expr::Bool(false))
            }
            TokenKind::Number(value) => {
                let value = *value;
                self.bump();
                Ok(Expr::Number(value))
            }
            TokenKind::String(value) => {
                let value = value.clone();
                self.bump();
                Ok(Expr::String(value))
            }
            TokenKind::Ident(name) => {
                let name = name.clone();
                self.bump();
                if matches!(self.peek_kind(), TokenKind::LParen) {
                    self.bump();
                    let mut args = Vec::new();
                    if !matches!(self.peek_kind(), TokenKind::RParen) {
                        loop {
                            args.push(self.parse_expr()?);
                            if matches!(self.peek_kind(), TokenKind::Comma) {
                                self.bump();
                                continue;
                            }
                            break;
                        }
                    }
                    self.expect_exact(TokenKind::RParen, "`)`")?;
                    Ok(Expr::BuiltinCall { name, args })
                } else {
                    Ok(Expr::Variable(name))
                }
            }
            TokenKind::LParen => {
                self.bump();
                let expr = self.parse_expr()?;
                self.expect_exact(TokenKind::RParen, "`)`")?;
                Ok(expr)
            }
            TokenKind::LBracket => self.parse_list(),
            TokenKind::LBrace => self.parse_record(),
            TokenKind::Call => Ok(Expr::ToolCall(self.parse_call_expr()?)),
            TokenKind::Parallel => {
                self.bump();
                let branches = self.parse_parallel_branches()?;
                Ok(Expr::Parallel { branches })
            }
            _ => Err(self.unexpected()),
        }
    }

    fn parse_list(&mut self) -> Result<Expr, ParseError> {
        self.expect_exact(TokenKind::LBracket, "`[`")?;
        let mut items = Vec::new();
        if !matches!(self.peek_kind(), TokenKind::RBracket) {
            loop {
                items.push(self.parse_expr()?);
                if matches!(self.peek_kind(), TokenKind::Comma) {
                    self.bump();
                    continue;
                }
                break;
            }
        }
        self.expect_exact(TokenKind::RBracket, "`]`")?;
        Ok(Expr::List(items))
    }

    fn parse_record(&mut self) -> Result<Expr, ParseError> {
        self.expect_exact(TokenKind::LBrace, "`{`")?;
        let entries = self.parse_record_entries()?;
        self.expect_exact(TokenKind::RBrace, "`}`")?;
        Ok(Expr::Record(entries))
    }

    fn parse_record_entries(&mut self) -> Result<Vec<(String, Expr)>, ParseError> {
        let mut entries = Vec::new();
        if !matches!(self.peek_kind(), TokenKind::RBrace) {
            loop {
                let key = self.expect_ident()?;
                self.expect_exact(TokenKind::Colon, "`:`")?;
                let value = self.parse_expr()?;
                entries.push((key, value));
                if matches!(self.peek_kind(), TokenKind::Comma) {
                    self.bump();
                    continue;
                }
                break;
            }
        }
        Ok(entries)
    }

    fn parse_call_expr(&mut self) -> Result<CallExpr, ParseError> {
        self.expect_exact(TokenKind::Call, "`call`")?;
        let name = self.expect_ident()?;
        self.expect_exact(TokenKind::LBrace, "`{`")?;
        let args = self.parse_record_entries()?;
        self.expect_exact(TokenKind::RBrace, "`}`")?;
        Ok(CallExpr { name, args })
    }

    fn expect_ident(&mut self) -> Result<String, ParseError> {
        let token = self.bump();
        match &token.kind {
            TokenKind::Ident(name) => Ok(name.clone()),
            other => Err(ParseError::Expected {
                expected: "identifier",
                found: render_kind(other),
                span: token.span,
            }),
        }
    }

    fn expect_exact(
        &mut self,
        expected_kind: TokenKind,
        expected: &'static str,
    ) -> Result<(), ParseError> {
        let token = self.bump();
        if std::mem::discriminant(&token.kind) == std::mem::discriminant(&expected_kind) {
            Ok(())
        } else {
            Err(ParseError::Expected {
                expected,
                found: render_kind(&token.kind),
                span: token.span,
            })
        }
    }

    fn unexpected(&mut self) -> ParseError {
        let token = self.peek();
        ParseError::Unexpected {
            found: render_kind(&token.kind),
            span: token.span,
        }
    }

    fn peek_assign(&self) -> bool {
        matches!(self.peek_kind(), TokenKind::Ident(_))
            && self
                .tokens
                .get(self.index + 1)
                .is_some_and(|token| matches!(token.kind, TokenKind::Equal))
    }

    fn at_eof(&self) -> bool {
        matches!(self.peek_kind(), TokenKind::Eof)
    }

    fn peek_kind(&self) -> &TokenKind {
        &self.tokens[self.index].kind
    }

    fn peek(&self) -> &Token {
        &self.tokens[self.index]
    }

    fn bump(&mut self) -> &Token {
        let token = &self.tokens[self.index];
        self.index += 1;
        token
    }

    #[cfg(test)]
    fn prev_span(&self) -> Span {
        self.tokens[self.index.saturating_sub(1)].span
    }
}

fn render_kind(kind: &TokenKind) -> String {
    match kind {
        TokenKind::Ident(name) => format!("identifier `{name}`"),
        TokenKind::String(value) => format!("string {:?}", value),
        TokenKind::Number(value) => format!("number {value}"),
        TokenKind::LBrace => "`{`".to_string(),
        TokenKind::RBrace => "`}`".to_string(),
        TokenKind::LParen => "`(`".to_string(),
        TokenKind::RParen => "`)`".to_string(),
        TokenKind::LBracket => "`[`".to_string(),
        TokenKind::RBracket => "`]`".to_string(),
        TokenKind::Comma => "`,`".to_string(),
        TokenKind::Colon => "`:`".to_string(),
        TokenKind::Question => "`?`".to_string(),
        TokenKind::Dot => "`.`".to_string(),
        TokenKind::Bang => "`!`".to_string(),
        TokenKind::Equal => "`=`".to_string(),
        TokenKind::DoubleEqual => "`==`".to_string(),
        TokenKind::BangEqual => "`!=`".to_string(),
        TokenKind::AndAnd => "`&&`".to_string(),
        TokenKind::OrOr => "`||`".to_string(),
        TokenKind::Less => "`<`".to_string(),
        TokenKind::LessEqual => "`<=`".to_string(),
        TokenKind::Greater => "`>`".to_string(),
        TokenKind::GreaterEqual => "`>=`".to_string(),
        TokenKind::Plus => "`+`".to_string(),
        TokenKind::Minus => "`-`".to_string(),
        TokenKind::Star => "`*`".to_string(),
        TokenKind::Slash => "`/`".to_string(),
        TokenKind::Percent => "`%`".to_string(),
        TokenKind::If => "`if`".to_string(),
        TokenKind::Else => "`else`".to_string(),
        TokenKind::For => "`for`".to_string(),
        TokenKind::In => "`in`".to_string(),
        TokenKind::Parallel => "`parallel`".to_string(),
        TokenKind::Finish => "`finish`".to_string(),
        TokenKind::Observe => "`observe`".to_string(),
        TokenKind::Call => "`call`".to_string(),
        TokenKind::And => "`and`".to_string(),
        TokenKind::Or => "`or`".to_string(),
        TokenKind::Not => "`not`".to_string(),
        TokenKind::True => "`true`".to_string(),
        TokenKind::False => "`false`".to_string(),
        TokenKind::Null => "`null`".to_string(),
        TokenKind::Eof => "end of input".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_full_language_shapes() {
        let program = parse(
            r#"
            call ping {}
            value = !false || -(5 - 2) < 0 && 8 / 2 >= 4 && 7 % 3 != 0 && 1 <= 2 && 3 > 2
            rec = { a: [1, 2], b: call tool { x: 1 } }
            if value {
              field = rec.b.ok
            } else {
              field = rec.a[0]
            }
            for item in rec.a {
              last = item
            }
            parallel {
              left = call alpha {}
              right = helper()
            }
            observe rec
            finish field
            "#,
        )
        .expect("program should parse");

        assert_eq!(program.statements.len(), 8);
    }

    #[test]
    fn parses_empty_records_lists_and_builtin_without_args() {
        let program = parse(
            r#"
            xs = []
            rec = {}
            out = now()
            finish out
            "#,
        )
        .expect("program should parse");

        assert_eq!(program.statements.len(), 4);
    }

    #[test]
    fn parses_ternary_expressions_with_low_precedence_and_right_association() {
        let program = parse(
            r#"
            value = false or true ? 1 : 2 ? 3 : 4
            finish value
            "#,
        )
        .expect("program should parse");

        let Stmt::Assign { expr, .. } = &program.statements[0] else {
            panic!("expected assignment");
        };
        let Expr::Conditional {
            condition,
            then_expr,
            else_expr,
        } = expr
        else {
            panic!("expected conditional expression");
        };
        assert!(matches!(
            condition.as_ref(),
            Expr::Binary {
                op: BinaryOp::Or,
                ..
            }
        ));
        assert!(matches!(then_expr.as_ref(), Expr::Number(1.0)));
        assert!(matches!(else_expr.as_ref(), Expr::Conditional { .. }));
    }

    #[test]
    fn parses_symbolic_boolean_operator_aliases() {
        let program = parse(
            r#"
            value = true && false || true
            finish value
            "#,
        )
        .expect("program should parse");

        assert_eq!(program.statements.len(), 2);
    }

    #[test]
    fn parses_expression_statements_and_parallel_expression_branches() {
        let program = parse(
            r#"
            "branch_a"
            results = parallel {
              "branch_b"
              40 + 2
              len([1,2,3])
            }
            finish results
            "#,
        )
        .expect("program should parse");

        assert_eq!(program.statements.len(), 3);
        assert!(matches!(program.statements[0], Stmt::Expr(Expr::String(_))));
    }

    #[test]
    fn parses_else_if_chains_without_extra_braces() {
        let program = parse(
            r#"
            if false {
              answer = 1
            } else if true {
              answer = 2
            } else {
              answer = 3
            }
            finish answer
            "#,
        )
        .expect("program should parse");

        let Stmt::If { else_block, .. } = &program.statements[0] else {
            panic!("expected if statement");
        };
        assert!(matches!(else_block.as_slice(), [Stmt::If { .. }]));
    }

    #[test]
    fn parse_errors_cover_expected_and_unexpected_paths() {
        let err = parse("{").expect_err("parse should fail");
        assert!(matches!(
            err,
            ParseError::Expected { .. } | ParseError::Unexpected { .. }
        ));

        let err = parse("x = ]").expect_err("parse should fail");
        assert!(matches!(err, ParseError::Unexpected { .. }));

        let err = parse("if true answer = 1").expect_err("parse should fail");
        assert!(matches!(
            err,
            ParseError::Expected {
                expected: "`{`",
                ..
            }
        ));

        let err = parse("for x 1 {}").expect_err("parse should fail");
        assert!(matches!(
            err,
            ParseError::Expected {
                expected: "`in`",
                ..
            }
        ));

        let err = parse("call {}").expect_err("parse should fail");
        assert!(matches!(
            err,
            ParseError::Expected {
                expected: "identifier",
                ..
            }
        ));

        let err = parse("x = [1").expect_err("parse should fail");
        assert!(matches!(
            err,
            ParseError::Expected {
                expected: "`]`",
                ..
            }
        ));

        let err = parse("x = {a 1}").expect_err("parse should fail");
        assert!(matches!(
            err,
            ParseError::Expected {
                expected: "`:`",
                ..
            }
        ));

        let err = parse("x = f(1").expect_err("parse should fail");
        assert!(matches!(
            err,
            ParseError::Expected {
                expected: "`)`",
                ..
            }
        ));

        let err = parse("finish true ? 1").expect_err("parse should fail");
        assert!(matches!(
            err,
            ParseError::Expected {
                expected: "`:`",
                ..
            }
        ));
    }

    #[test]
    fn peek_assign_and_prev_span_are_exercised() {
        let tokens = lex("x = 1").expect("lexing should succeed");
        let parser = Parser { tokens, index: 0 };
        assert!(parser.peek_assign());

        let tokens = lex("x").expect("lexing should succeed");
        let parser = Parser { tokens, index: 1 };
        assert_eq!(parser.prev_span(), Span { start: 0, end: 1 });
    }

    #[test]
    fn render_kind_covers_every_variant() {
        let samples = vec![
            TokenKind::Ident("x".to_string()),
            TokenKind::String("s".to_string()),
            TokenKind::Number(1.0),
            TokenKind::LBrace,
            TokenKind::RBrace,
            TokenKind::LParen,
            TokenKind::RParen,
            TokenKind::LBracket,
            TokenKind::RBracket,
            TokenKind::Comma,
            TokenKind::Colon,
            TokenKind::Question,
            TokenKind::Dot,
            TokenKind::Bang,
            TokenKind::Equal,
            TokenKind::DoubleEqual,
            TokenKind::BangEqual,
            TokenKind::AndAnd,
            TokenKind::OrOr,
            TokenKind::Less,
            TokenKind::LessEqual,
            TokenKind::Greater,
            TokenKind::GreaterEqual,
            TokenKind::Plus,
            TokenKind::Minus,
            TokenKind::Star,
            TokenKind::Slash,
            TokenKind::Percent,
            TokenKind::If,
            TokenKind::Else,
            TokenKind::For,
            TokenKind::In,
            TokenKind::Parallel,
            TokenKind::Finish,
            TokenKind::Observe,
            TokenKind::Call,
            TokenKind::And,
            TokenKind::Or,
            TokenKind::Not,
            TokenKind::True,
            TokenKind::False,
            TokenKind::Null,
            TokenKind::Eof,
        ];

        for sample in samples {
            assert!(!render_kind(&sample).is_empty());
        }
    }
}
