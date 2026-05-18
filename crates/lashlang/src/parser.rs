use crate::ast::{
    AssignPathStep, AssignTarget, AstString, BinaryOp, CallExpr, Expr, NamedParallelBranch,
    ParallelBranches, Program, Stmt, TypeExpr, TypeField, UnaryOp,
};
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
    #[error("`{keyword}` can only be used inside a `for` loop")]
    LoopControlOutsideLoop { keyword: &'static str, span: Span },
    #[error("unsupported `{keyword}` loop; use bounded `for` loops over ranges or lists")]
    UnsupportedLoop { keyword: &'static str, span: Span },
}

impl ParseError {
    pub fn offset(&self) -> usize {
        match self {
            Self::Lex(err) => err.offset(),
            Self::Expected { span, .. }
            | Self::Unexpected { span, .. }
            | Self::LoopControlOutsideLoop { span, .. }
            | Self::UnsupportedLoop { span, .. } => span.start,
        }
    }
}

pub fn parse(source: &str) -> Result<Program, ParseError> {
    let tokens = lex(source)?;
    Parser {
        tokens,
        index: 0,
        loop_depth: 0,
    }
    .parse_program()
}

struct Parser {
    tokens: Vec<Token>,
    index: usize,
    loop_depth: usize,
}

impl Parser {
    fn parse_program(&mut self) -> Result<Program, ParseError> {
        let statement_capacity = (self.tokens.len() / 20).max(1);
        let mut statements = Vec::with_capacity(statement_capacity);
        let mut statement_spans = Vec::with_capacity(statement_capacity);
        while !self.at_eof() {
            let start = self.peek().span.start;
            let stmt = self.parse_stmt()?;
            let end = self.tokens[self.index.saturating_sub(1)].span.end;
            statements.push(stmt);
            statement_spans.push(Span { start, end });
        }
        Ok(Program {
            statements,
            statement_spans,
        })
    }

    fn parse_stmt(&mut self) -> Result<Stmt, ParseError> {
        match self.peek_kind() {
            TokenKind::If => self.parse_if(),
            TokenKind::For => self.parse_for(),
            TokenKind::Parallel => self.parse_parallel(),
            TokenKind::Submit => self.parse_submit(),
            TokenKind::Cancel => self.parse_cancel(),
            TokenKind::Print => self.parse_print(),
            TokenKind::Call => Ok(Stmt::Call(self.parse_call_expr()?)),
            TokenKind::Ident(name) if name == "break" && !self.peek_assignment_target() => {
                self.parse_loop_control("break")
            }
            TokenKind::Ident(name) if name == "continue" && !self.peek_assignment_target() => {
                self.parse_loop_control("continue")
            }
            TokenKind::Ident(name) if name == "while" && !self.peek_assignment_target() => {
                Err(ParseError::UnsupportedLoop {
                    keyword: "while",
                    span: self.peek().span,
                })
            }
            TokenKind::Ident(_) if self.peek_assignment_target() => self.parse_assign(),
            _ => Ok(Stmt::Expr(self.parse_expr()?)),
        }
    }

    fn parse_assign(&mut self) -> Result<Stmt, ParseError> {
        let target = self.parse_assignment_target()?;
        self.expect_exact(TokenKind::Equal, "`=`")?;
        let expr = self.parse_expr()?;
        Ok(Stmt::Assign { target, expr })
    }

    fn parse_assignment_target(&mut self) -> Result<AssignTarget, ParseError> {
        let root = self.expect_ident()?;
        let mut steps = Vec::new();
        loop {
            match self.peek_kind() {
                TokenKind::Dot => {
                    self.bump();
                    steps.push(AssignPathStep::Field(self.expect_key_name()?));
                }
                TokenKind::LBracket => {
                    self.bump();
                    let index = self.parse_expr()?;
                    self.expect_exact(TokenKind::RBracket, "`]`")?;
                    steps.push(AssignPathStep::Index(index));
                }
                _ => break,
            }
        }
        Ok(AssignTarget { root, steps })
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
        self.loop_depth += 1;
        let body = self.parse_block()?;
        self.loop_depth -= 1;
        Ok(Stmt::For {
            binding,
            iterable,
            body,
        })
    }

    fn parse_loop_control(&mut self, keyword: &'static str) -> Result<Stmt, ParseError> {
        let span = self.bump().span;
        if self.loop_depth == 0 {
            return Err(ParseError::LoopControlOutsideLoop { keyword, span });
        }
        Ok(match keyword {
            "break" => Stmt::Break,
            "continue" => Stmt::Continue,
            _ => unreachable!("unknown loop control keyword"),
        })
    }

    fn parse_parallel(&mut self) -> Result<Stmt, ParseError> {
        self.bump();
        let branches = self.parse_parallel_branches()?;
        Ok(Stmt::Parallel { branches })
    }

    fn parse_parallel_branches(&mut self) -> Result<ParallelBranches, ParseError> {
        self.expect_exact(TokenKind::LBrace, "`{`")?;
        if matches!(self.peek_kind(), TokenKind::RBrace) {
            self.bump();
            return Ok(ParallelBranches::Positional(Vec::new()));
        }

        let outer_loop_depth = std::mem::take(&mut self.loop_depth);
        let named = self.peek_named_parallel_branch();
        if named {
            let mut branches = Vec::new();
            let mut seen = std::collections::HashSet::new();
            while !matches!(self.peek_kind(), TokenKind::RBrace | TokenKind::Eof) {
                let name_span = self.peek().span;
                let name = self.expect_key_name()?;
                if !seen.insert(name.clone()) {
                    return Err(ParseError::Expected {
                        expected: "unique branch name",
                        found: format!("duplicate branch `{name}`"),
                        span: name_span,
                    });
                }
                self.expect_exact(TokenKind::Colon, "`:`")?;
                let stmt = self.parse_stmt()?;
                branches.push(NamedParallelBranch { name, stmt });
                if matches!(self.peek_kind(), TokenKind::Comma) {
                    self.bump();
                    continue;
                }
            }
            self.expect_exact(TokenKind::RBrace, "`}`")?;
            self.loop_depth = outer_loop_depth;
            return Ok(ParallelBranches::Named(branches));
        }

        let mut statements = Vec::new();
        while !matches!(self.peek_kind(), TokenKind::RBrace | TokenKind::Eof) {
            statements.push(self.parse_stmt()?);
            // Accept commas as an optional branch separator so
            // `parallel { a; b }` and `parallel { a, b }` parse the
            // same — matches the leniency already in the named-branch
            // path above.
            if matches!(self.peek_kind(), TokenKind::Comma) {
                self.bump();
            }
        }
        self.expect_exact(TokenKind::RBrace, "`}`")?;
        self.loop_depth = outer_loop_depth;
        Ok(ParallelBranches::Positional(statements))
    }

    fn parse_submit(&mut self) -> Result<Stmt, ParseError> {
        self.bump();
        let expr = if matches!(self.peek_kind(), TokenKind::RBrace | TokenKind::Eof) {
            None
        } else {
            Some(self.parse_expr()?)
        };
        Ok(Stmt::Submit(expr))
    }

    fn parse_print(&mut self) -> Result<Stmt, ParseError> {
        self.bump();
        Ok(Stmt::Print(self.parse_expr()?))
    }

    fn parse_cancel(&mut self) -> Result<Stmt, ParseError> {
        self.bump();
        Ok(Stmt::Cancel(self.parse_expr()?))
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
            TokenKind::Await => {
                self.bump();
                Ok(Expr::Await(Box::new(self.parse_unary()?)))
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
                    let field = self.expect_key_name()?;
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
                TokenKind::Question if !self.question_starts_ternary() => {
                    self.bump();
                    expr = Expr::ResultUnwrap(Box::new(expr));
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
                if name == "start" && matches!(self.peek_kind(), TokenKind::Call) {
                    return Ok(Expr::StartToolCall(self.parse_call_expr()?));
                }
                if name == "Type" && matches!(self.peek_kind(), TokenKind::LBrace) {
                    let ty = self.parse_type_object()?;
                    return Ok(Expr::TypeLiteral(Box::new(ty)));
                }
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
        while !matches!(self.peek_kind(), TokenKind::RBracket) {
            items.push(self.parse_expr()?);
            if matches!(self.peek_kind(), TokenKind::Comma) {
                self.bump();
                continue;
            }
            break;
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

    fn parse_record_entries(&mut self) -> Result<Vec<(AstString, Expr)>, ParseError> {
        let mut entries = Vec::new();
        while !matches!(self.peek_kind(), TokenKind::RBrace) {
            let key = self.expect_key_name()?;
            self.expect_exact(TokenKind::Colon, "`:`")?;
            let value = self.parse_expr()?;
            entries.push((key, value));
            if matches!(self.peek_kind(), TokenKind::Comma) {
                self.bump();
                continue;
            }
            break;
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

    fn parse_type_object(&mut self) -> Result<TypeExpr, ParseError> {
        self.expect_exact(TokenKind::LBrace, "`{`")?;
        let mut fields = Vec::new();
        let mut seen = std::collections::HashSet::new();
        while !matches!(self.peek_kind(), TokenKind::RBrace) {
            let name_token_span = self.peek().span;
            let name = self.expect_key_name()?;
            if !seen.insert(name.clone()) {
                return Err(ParseError::Expected {
                    expected: "unique field name",
                    found: format!("duplicate field `{name}`"),
                    span: name_token_span,
                });
            }
            self.expect_exact(TokenKind::Colon, "`:`")?;
            let ty = self.parse_type_expr()?;
            let optional = if matches!(self.peek_kind(), TokenKind::Question) {
                self.bump();
                true
            } else {
                false
            };
            fields.push(TypeField { name, ty, optional });
            if matches!(self.peek_kind(), TokenKind::Comma) {
                self.bump();
                continue;
            }
            break;
        }
        self.expect_exact(TokenKind::RBrace, "`}`")?;
        Ok(TypeExpr::Object(fields))
    }

    fn parse_type_expr(&mut self) -> Result<TypeExpr, ParseError> {
        let first = self.parse_type_term()?;
        if !matches!(self.peek_kind(), TokenKind::Pipe) {
            return Ok(first);
        }
        // Union: `str | null`, `int | str | null`, etc. `|` has lower
        // precedence than any other type constructor — once we see it
        // at top level, keep parsing `| <term>` until the run ends.
        let mut variants = vec![first];
        while matches!(self.peek_kind(), TokenKind::Pipe) {
            self.bump();
            variants.push(self.parse_type_term()?);
        }
        Ok(TypeExpr::Union(variants))
    }

    fn parse_type_term(&mut self) -> Result<TypeExpr, ParseError> {
        let token = self.peek().clone();
        match token.kind {
            TokenKind::Null => {
                self.bump();
                Ok(TypeExpr::Null)
            }
            // A bare `{` in type position is the classic "forgot the
            // `Type` keyword" mistake (`foo: { ok: bool }` instead of
            // `foo: Type { ok: bool }`). Surface a targeted diagnostic
            // rather than the generic "expected type expression" shrug.
            TokenKind::LBrace => Err(ParseError::Expected {
                expected: "type expression (type literals must start with `Type`, e.g. `Type { ok: bool }`)",
                found: render_kind(&token.kind),
                span: token.span,
            }),
            TokenKind::Ident(name) => {
                self.bump();
                match name.as_str() {
                    "str" | "string" => Ok(TypeExpr::Str),
                    "int" | "integer" => Ok(TypeExpr::Int),
                    "float" | "number" => Ok(TypeExpr::Float),
                    "bool" | "boolean" => Ok(TypeExpr::Bool),
                    "dict" | "object" => Ok(TypeExpr::Dict),
                    "any" => Ok(TypeExpr::Any),
                    "enum" => {
                        self.expect_exact(TokenKind::LBracket, "`[`")?;
                        let mut values = Vec::new();
                        if !matches!(self.peek_kind(), TokenKind::RBracket) {
                            loop {
                                let value = self.expect_string_literal()?;
                                values.push(value);
                                if matches!(self.peek_kind(), TokenKind::Comma) {
                                    self.bump();
                                    continue;
                                }
                                break;
                            }
                        }
                        if values.is_empty() {
                            return Err(ParseError::Expected {
                                expected: "at least one enum string literal",
                                found: "empty enum".to_string(),
                                span: token.span,
                            });
                        }
                        self.expect_exact(TokenKind::RBracket, "`]`")?;
                        Ok(TypeExpr::Enum(values))
                    }
                    "list" => {
                        self.expect_exact(TokenKind::LBracket, "`[`")?;
                        let inner = self.parse_type_expr()?;
                        self.expect_exact(TokenKind::RBracket, "`]`")?;
                        Ok(TypeExpr::List(Box::new(inner)))
                    }
                    "Type" => self.parse_type_object(),
                    _ => Ok(TypeExpr::Ref(name)),
                }
            }
            _ => Err(ParseError::Expected {
                expected: "type expression",
                found: render_kind(&token.kind),
                span: token.span,
            }),
        }
    }

    fn expect_string_literal(&mut self) -> Result<AstString, ParseError> {
        let token = self.bump().clone();
        match token.kind {
            TokenKind::String(value) => Ok(value),
            other => Err(ParseError::Expected {
                expected: "string literal",
                found: render_kind(&other),
                span: token.span,
            }),
        }
    }

    fn expect_ident(&mut self) -> Result<AstString, ParseError> {
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

    fn expect_key_name(&mut self) -> Result<AstString, ParseError> {
        let token = self.bump();
        match &token.kind {
            TokenKind::Ident(name) | TokenKind::String(name) => Ok(name.clone()),
            other => keyword_key_name(other)
                .map(Into::into)
                .ok_or_else(|| ParseError::Expected {
                    expected: "identifier, string key, or keyword key",
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

    fn peek_assignment_target(&self) -> bool {
        if !matches!(self.peek_kind(), TokenKind::Ident(_)) {
            return false;
        }

        let mut index = self.index + 1;
        loop {
            match self.tokens.get(index).map(|token| &token.kind) {
                Some(TokenKind::Dot) => {
                    if !self
                        .tokens
                        .get(index + 1)
                        .is_some_and(|token| token_can_be_key(&token.kind))
                    {
                        return false;
                    }
                    index += 2;
                }
                Some(TokenKind::LBracket) => {
                    let Some(after_index) = self.skip_bracketed_index(index) else {
                        return false;
                    };
                    index = after_index;
                }
                Some(TokenKind::Equal) => return true,
                _ => return false,
            }
        }
    }

    fn skip_bracketed_index(&self, start: usize) -> Option<usize> {
        debug_assert!(matches!(
            self.tokens.get(start).map(|token| &token.kind),
            Some(TokenKind::LBracket)
        ));
        let mut parens = 0usize;
        let mut brackets = 1usize;
        let mut braces = 0usize;
        for (offset, token) in self.tokens.iter().enumerate().skip(start + 1) {
            match &token.kind {
                TokenKind::LParen => parens += 1,
                TokenKind::RParen => parens = parens.checked_sub(1)?,
                TokenKind::LBracket => brackets += 1,
                TokenKind::RBracket => {
                    brackets = brackets.checked_sub(1)?;
                    if brackets == 0 && parens == 0 && braces == 0 {
                        return Some(offset + 1);
                    }
                }
                TokenKind::LBrace => braces += 1,
                TokenKind::RBrace => braces = braces.checked_sub(1)?,
                TokenKind::Eof => return None,
                _ => {}
            }
        }
        None
    }

    fn peek_named_parallel_branch(&self) -> bool {
        token_can_be_key(self.peek_kind())
            && self
                .tokens
                .get(self.index + 1)
                .is_some_and(|token| matches!(token.kind, TokenKind::Colon))
    }

    fn question_starts_ternary(&self) -> bool {
        debug_assert!(matches!(self.peek_kind(), TokenKind::Question));
        let Some(next) = self.tokens.get(self.index + 1) else {
            return false;
        };
        if !token_can_start_expr(&next.kind) {
            return false;
        }

        let mut parens = 0usize;
        let mut brackets = 0usize;
        let mut braces = 0usize;
        for token in self.tokens.iter().skip(self.index + 1) {
            match &token.kind {
                TokenKind::Colon if parens == 0 && brackets == 0 && braces == 0 => return true,
                TokenKind::Equal if parens == 0 && brackets == 0 && braces == 0 => return false,
                TokenKind::Comma | TokenKind::RParen | TokenKind::RBracket | TokenKind::RBrace
                    if parens == 0 && brackets == 0 && braces == 0 =>
                {
                    return false;
                }
                TokenKind::Eof => return false,
                TokenKind::LParen => parens += 1,
                TokenKind::RParen => {
                    if parens == 0 {
                        return false;
                    }
                    parens -= 1;
                }
                TokenKind::LBracket => brackets += 1,
                TokenKind::RBracket => {
                    if brackets == 0 {
                        return false;
                    }
                    brackets -= 1;
                }
                TokenKind::LBrace => braces += 1,
                TokenKind::RBrace => {
                    if braces == 0 {
                        return false;
                    }
                    braces -= 1;
                }
                _ => {}
            }
        }
        false
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

fn token_can_be_key(kind: &TokenKind) -> bool {
    matches!(kind, TokenKind::Ident(_) | TokenKind::String(_)) || keyword_key_name(kind).is_some()
}

fn keyword_key_name(kind: &TokenKind) -> Option<&'static str> {
    Some(match kind {
        TokenKind::If => "if",
        TokenKind::Else => "else",
        TokenKind::For => "for",
        TokenKind::In => "in",
        TokenKind::Parallel => "parallel",
        TokenKind::Await => "await",
        TokenKind::Cancel => "cancel",
        TokenKind::Submit => "submit",
        TokenKind::Print => "print",
        TokenKind::Call => "call",
        TokenKind::And => "and",
        TokenKind::Or => "or",
        TokenKind::Not => "not",
        TokenKind::True => "true",
        TokenKind::False => "false",
        TokenKind::Null => "null",
        _ => return None,
    })
}

fn token_can_start_expr(kind: &TokenKind) -> bool {
    matches!(
        kind,
        TokenKind::Null
            | TokenKind::True
            | TokenKind::False
            | TokenKind::Number(_)
            | TokenKind::String(_)
            | TokenKind::Ident(_)
            | TokenKind::LParen
            | TokenKind::LBracket
            | TokenKind::LBrace
            | TokenKind::Call
            | TokenKind::Parallel
            | TokenKind::Await
            | TokenKind::Minus
            | TokenKind::Bang
            | TokenKind::Not
    )
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
        TokenKind::Pipe => "`|`".to_string(),
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
        TokenKind::Await => "`await`".to_string(),
        TokenKind::Cancel => "`cancel`".to_string(),
        TokenKind::Submit => "`submit`".to_string(),
        TokenKind::Print => "`print`".to_string(),
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
            print rec
            submit field
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
            submit out
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
            submit value
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
    fn parses_result_unwrap_without_breaking_ternary() {
        let program = parse(
            r#"
            a = r?
            b = (call echo { value: 1 })?
            c = items[0]?
            d = r? + 1
            e = cond ? yes : no
            f = r?[0]
            submit e
            "#,
        )
        .expect("program should parse");

        assert!(matches!(
            &program.statements[0],
            Stmt::Assign {
                expr: Expr::ResultUnwrap(_),
                ..
            }
        ));
        assert!(matches!(
            &program.statements[1],
            Stmt::Assign {
                expr: Expr::ResultUnwrap(inner),
                ..
            } if matches!(inner.as_ref(), Expr::ToolCall(_))
        ));
        assert!(matches!(
            &program.statements[2],
            Stmt::Assign {
                expr: Expr::ResultUnwrap(inner),
                ..
            } if matches!(inner.as_ref(), Expr::Index { .. })
        ));
        assert!(matches!(
            &program.statements[3],
            Stmt::Assign {
                expr: Expr::Binary { left, .. },
                ..
            } if matches!(left.as_ref(), Expr::ResultUnwrap(_))
        ));
        assert!(matches!(
            &program.statements[4],
            Stmt::Assign {
                expr: Expr::Conditional { .. },
                ..
            }
        ));
        assert!(matches!(
            &program.statements[5],
            Stmt::Assign {
                expr: Expr::Index { target, .. },
                ..
            } if matches!(target.as_ref(), Expr::ResultUnwrap(_))
        ));
    }

    #[test]
    fn parses_variable_rooted_assignment_targets() {
        let program = parse(
            r#"
            x = 1
            rec.a = 1
            rec[k] = 1
            items[0] = 1
            a[b].c[0] = 1
            submit x
            "#,
        )
        .expect("program should parse");

        assert!(matches!(
            &program.statements[0],
            Stmt::Assign { target, .. } if target.root == "x" && target.steps.is_empty()
        ));
        assert!(matches!(
            &program.statements[1],
            Stmt::Assign { target, .. }
                if target.root == "rec"
                    && matches!(target.steps.as_slice(), [AssignPathStep::Field(field)] if field == "a")
        ));
        assert!(matches!(
            &program.statements[2],
            Stmt::Assign { target, .. }
                if target.root == "rec"
                    && matches!(target.steps.as_slice(), [AssignPathStep::Index(Expr::Variable(key))] if key == "k")
        ));
        assert!(matches!(
            &program.statements[3],
            Stmt::Assign { target, .. }
                if target.root == "items"
                    && matches!(target.steps.as_slice(), [AssignPathStep::Index(Expr::Number(0.0))])
        ));
        assert!(matches!(
            &program.statements[4],
            Stmt::Assign { target, .. }
                if target.root == "a"
                    && matches!(
                        target.steps.as_slice(),
                        [
                            AssignPathStep::Index(Expr::Variable(b)),
                            AssignPathStep::Field(c),
                            AssignPathStep::Index(Expr::Number(0.0)),
                        ] if b == "b" && c == "c"
                    )
        ));
    }

    #[test]
    fn rejects_non_variable_rooted_assignment_targets() {
        for source in ["(x)[0] = 1", "{a: 1}.a = 2", "(call f {})[0] = 1"] {
            parse(source).expect_err("non-variable-rooted target should not parse as assignment");
        }
    }

    #[test]
    fn parses_symbolic_boolean_operator_aliases() {
        let program = parse(
            r#"
            value = true && false || true
            submit value
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
            submit results
            "#,
        )
        .expect("program should parse");

        assert_eq!(program.statements.len(), 3);
        assert!(matches!(program.statements[0], Stmt::Expr(Expr::String(_))));
    }

    #[test]
    fn positional_parallel_branches_accept_optional_commas() {
        let program = parse(
            r#"
            results = parallel {
              "branch_a",
              40 + 2,
              len([1,2,3]),
            }
            submit results
            "#,
        )
        .expect("program should parse");

        let Stmt::Assign {
            expr: Expr::Parallel { branches },
            ..
        } = &program.statements[0]
        else {
            panic!("expected parallel expression assignment");
        };
        let ParallelBranches::Positional(statements) = branches else {
            panic!("expected positional parallel branches");
        };
        assert_eq!(statements.len(), 3);
    }

    #[test]
    fn parses_named_parallel_branches() {
        let program = parse(
            r#"
            results = parallel {
              cargo: call exec_command { cmd: "cargo test" }
              fmt: "ok"
            }
            submit results
            "#,
        )
        .expect("program should parse");

        let Stmt::Assign {
            expr: Expr::Parallel { branches },
            ..
        } = &program.statements[0]
        else {
            panic!("expected parallel expression assignment");
        };
        let ParallelBranches::Named(branches) = branches else {
            panic!("expected named parallel branches");
        };
        assert_eq!(branches.len(), 2);
        assert_eq!(branches[0].name, "cargo");
        assert!(matches!(branches[0].stmt, Stmt::Call(_)));
        assert_eq!(branches[1].name, "fmt");
        assert!(matches!(branches[1].stmt, Stmt::Expr(Expr::String(_))));
    }

    #[test]
    fn named_parallel_branches_accept_optional_commas_and_keyword_names() {
        let program = parse(
            r#"
            results = parallel {
              parallel: "branch",
              "quoted": "ok",
            }
            submit results.parallel
            "#,
        )
        .expect("program should parse");

        let Stmt::Assign {
            expr: Expr::Parallel { branches },
            ..
        } = &program.statements[0]
        else {
            panic!("expected parallel expression assignment");
        };
        let ParallelBranches::Named(branches) = branches else {
            panic!("expected named parallel branches");
        };
        assert_eq!(branches.len(), 2);
        assert_eq!(branches[0].name, "parallel");
        assert_eq!(branches[1].name, "quoted");
    }

    #[test]
    fn named_parallel_rejects_duplicate_branch_names() {
        let err = parse(
            r#"
            results = parallel {
              same: 1
              same: 2
            }
            "#,
        )
        .expect_err("duplicate branch names should fail");
        assert!(matches!(
            err,
            ParseError::Expected {
                expected: "unique branch name",
                ..
            }
        ));
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
            submit answer
            "#,
        )
        .expect("program should parse");

        let Stmt::If { else_block, .. } = &program.statements[0] else {
            panic!("expected if statement");
        };
        assert!(matches!(else_block.as_slice(), [Stmt::If { .. }]));
    }

    #[test]
    fn parses_bare_finish_at_program_and_block_end() {
        let program = parse(
            r#"
            if true {
              submit
            }
            submit
            "#,
        )
        .expect("program should parse");

        assert!(matches!(
            program.statements.as_slice(),
            [
                Stmt::If { then_block, .. },
                Stmt::Submit(None)
            ] if matches!(then_block.as_slice(), [Stmt::Submit(None)])
        ));
    }

    #[test]
    fn parses_loop_control_inside_for() {
        let program = parse(
            r#"
            for item in [1, 2, 3] {
              if item == 1 {
                continue
              }
              break
            }
            "#,
        )
        .expect("program should parse");

        let [Stmt::For { body, .. }] = program.statements.as_slice() else {
            panic!("expected for statement");
        };
        assert!(matches!(
            body.as_slice(),
            [Stmt::If { then_block, .. }, Stmt::Break]
                if matches!(then_block.as_slice(), [Stmt::Continue])
        ));
    }

    #[test]
    fn rejects_loop_control_outside_for() {
        for (source, keyword) in [("break", "break"), ("continue", "continue")] {
            let err = parse(source).expect_err("loop control outside loop should fail");
            assert_eq!(
                err,
                ParseError::LoopControlOutsideLoop {
                    keyword,
                    span: Span {
                        start: 0,
                        end: keyword.len()
                    }
                }
            );
        }
    }

    #[test]
    fn loop_control_does_not_cross_parallel_branch_boundaries() {
        let err = parse(
            r#"
            for item in [1] {
              parallel {
                break
              }
            }
            "#,
        )
        .expect_err("parallel branch should not break outer loop");
        assert!(matches!(
            err,
            ParseError::LoopControlOutsideLoop {
                keyword: "break",
                ..
            }
        ));

        parse(
            r#"
            parallel {
              for item in [1] {
                continue
              }
            }
            "#,
        )
        .expect("loops inside parallel branches may use loop control");
    }

    #[test]
    fn loop_control_words_remain_contextual_identifiers() {
        let program = parse(
            r#"
            break = 1
            continue = break + 1
            submit break
            "#,
        )
        .expect("contextual identifiers should parse");

        assert!(matches!(
            program.statements.as_slice(),
            [
                Stmt::Assign { target: first, .. },
                Stmt::Assign { target: second, .. },
                Stmt::Submit(Some(Expr::Variable(submitted)))
            ] if first.root == "break" && second.root == "continue" && submitted == "break"
        ));
    }

    #[test]
    fn rejects_while_loop_at_while_keyword() {
        let source = r#"
            pool_i = 0
            while len(final_ids) < 100 && pool_i < len(candidate_pools) {
              for m in candidate_pools[pool_i].matches {
                print m
              }
            }
            "#;

        let err = parse(source).expect_err("while loops are not supported");
        let while_offset = source.find("while").expect("source should contain while");
        assert_eq!(
            err,
            ParseError::UnsupportedLoop {
                keyword: "while",
                span: Span {
                    start: while_offset,
                    end: while_offset + "while".len(),
                },
            }
        );
    }

    #[test]
    fn unsupported_loop_words_remain_contextual_identifiers_for_assignment() {
        let program = parse(
            r#"
            while = 1
            submit while
            "#,
        )
        .expect("contextual identifier should parse");

        assert!(matches!(
            program.statements.as_slice(),
            [
                Stmt::Assign { target, .. },
                Stmt::Submit(Some(Expr::Variable(submitted)))
            ] if target.root == "while" && submitted == "while"
        ));
    }

    #[test]
    fn parses_async_tool_syntax() {
        let program = parse(
            r#"
            handle = start call spawn_agent { task: "check", capability: "explore" }
            result = await handle
            cancel handle
            submit result
            "#,
        )
        .expect("program should parse");

        assert!(matches!(
            &program.statements[0],
            Stmt::Assign {
                expr: Expr::StartToolCall(_),
                ..
            }
        ));
        assert!(matches!(
            &program.statements[1],
            Stmt::Assign {
                expr: Expr::Await(_),
                ..
            }
        ));
        assert!(matches!(&program.statements[2], Stmt::Cancel(_)));
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

        let err = parse("submit true ? 1 :").expect_err("parse should fail");
        assert!(matches!(err, ParseError::Unexpected { .. }));
    }

    #[test]
    fn peek_assignment_target_and_prev_span_are_exercised() {
        let tokens = lex("x[0].field = 1").expect("lexing should succeed");
        let parser = Parser {
            tokens,
            index: 0,
            loop_depth: 0,
        };
        assert!(parser.peek_assignment_target());

        let tokens = lex("x").expect("lexing should succeed");
        let parser = Parser {
            tokens,
            index: 1,
            loop_depth: 0,
        };
        assert_eq!(parser.prev_span(), Span { start: 0, end: 1 });
    }

    #[test]
    fn parses_type_literal_with_all_kinds() {
        let program = parse(
            r#"
            Books = Type {
              title: str,
              count: int,
              rating: float,
              active: bool,
              meta: dict,
              extra: any,
              genre: enum["fiction", "non-fiction"],
              tags: list[str],
              chapters: list[Type { name: str, page: int }],
              nested: Type { pages: int },
              isbn: str?,
              reference: Books
            }
            submit Books
            "#,
        )
        .expect("program should parse");

        let Stmt::Assign { expr, .. } = &program.statements[0] else {
            panic!("expected assign");
        };
        let Expr::TypeLiteral(ty) = expr else {
            panic!("expected TypeLiteral");
        };
        let TypeExpr::Object(fields) = ty.as_ref() else {
            panic!("expected object");
        };
        assert_eq!(fields.len(), 12);
        assert!(matches!(fields[0].ty, TypeExpr::Str));
        assert!(matches!(fields[1].ty, TypeExpr::Int));
        assert!(matches!(fields[5].ty, TypeExpr::Any));
        assert!(matches!(fields[6].ty, TypeExpr::Enum(_)));
        assert!(matches!(fields[7].ty, TypeExpr::List(_)));
        assert!(matches!(fields[9].ty, TypeExpr::Object(_)));
        assert!(fields[10].optional);
        assert!(matches!(fields[11].ty, TypeExpr::Ref(ref name) if name == "Books"));
    }

    #[test]
    fn type_is_not_a_global_keyword_outside_literal_position() {
        // A bare `Type` identifier should still bind as a variable reference
        // when not followed by `{` — avoids breaking pre-existing programs
        // that may already use the name.
        let program = parse("Type = 1\nx = Type\nsubmit x").expect("should parse");
        assert_eq!(program.statements.len(), 3);
        let Stmt::Assign { expr, .. } = &program.statements[1] else {
            panic!("expected assign");
        };
        assert!(matches!(expr, Expr::Variable(name) if name == "Type"));
    }

    #[test]
    fn type_literal_rejects_empty_enum() {
        let err = parse("x = Type { v: enum[] }").expect_err("empty enum should fail");
        let message = format!("{err}");
        assert!(
            message.contains("empty enum") || message.contains("enum"),
            "error should mention enum: {message}"
        );
    }

    #[test]
    fn type_literal_rejects_duplicate_fields() {
        let err = parse("x = Type { a: str, a: int }").expect_err("duplicate field");
        let message = format!("{err}");
        assert!(message.contains("duplicate"), "error: {message}");
    }

    #[test]
    fn type_literal_rejects_non_string_enum_value() {
        let err = parse("x = Type { v: enum[1, 2] }").expect_err("numeric enum value");
        assert!(matches!(err, ParseError::Expected { .. }));
    }

    #[test]
    fn type_literal_parses_with_or_without_trailing_comma() {
        parse("x = Type { a: str, b: int }").expect("no trailing comma should parse");
        parse("x = Type { a: str, b: int, }").expect("trailing comma should parse");
    }

    #[test]
    fn type_literal_parses_nullable_field_as_union_with_null() {
        let program = parse(
            r#"
            User = Type { name: str, email: str | null }
            submit User
            "#,
        )
        .expect("program should parse");
        let Stmt::Assign { expr, .. } = &program.statements[0] else {
            panic!("expected assign");
        };
        let Expr::TypeLiteral(ty) = expr else {
            panic!("expected TypeLiteral");
        };
        let TypeExpr::Object(fields) = ty.as_ref() else {
            panic!("expected object");
        };
        assert_eq!(fields.len(), 2);
        let TypeExpr::Union(variants) = &fields[1].ty else {
            panic!("expected email to be a Union, got {:?}", fields[1].ty);
        };
        assert_eq!(variants.len(), 2);
        assert!(matches!(variants[0], TypeExpr::Str));
        assert!(matches!(variants[1], TypeExpr::Null));
    }

    #[test]
    fn type_literal_parses_three_way_union() {
        let program = parse("x = Type { v: str | int | null }").expect("should parse");
        let Stmt::Assign { expr, .. } = &program.statements[0] else {
            panic!("expected assign");
        };
        let Expr::TypeLiteral(ty) = expr else {
            panic!("expected TypeLiteral");
        };
        let TypeExpr::Object(fields) = ty.as_ref() else {
            panic!("expected object");
        };
        let TypeExpr::Union(variants) = &fields[0].ty else {
            panic!("expected union");
        };
        assert_eq!(variants.len(), 3);
    }

    #[test]
    fn type_literal_bare_brace_in_field_position_gives_targeted_diagnostic() {
        let err = parse("x = Type { nested: { ok: bool } }")
            .expect_err("bare `{` in type position should fail");
        let message = format!("{err}");
        assert!(
            message.contains("Type"),
            "diagnostic should mention the `Type` keyword: {message}",
        );
    }

    #[test]
    fn list_and_record_literals_accept_trailing_commas() {
        parse("x = [1, 2, 3,]\nsubmit x").expect("list trailing comma");
        parse("x = { a: 1, b: 2, }\nsubmit x").expect("record trailing comma");
        parse("x = { parallel: 1, \"with space\": 2 }\nsubmit x.parallel")
            .expect("record keyword and quoted keys");
        // Empty literals still work.
        parse("x = []\nsubmit x").expect("empty list");
        parse("x = {}\nsubmit x").expect("empty record");
    }

    #[test]
    fn render_kind_covers_every_variant() {
        let samples = vec![
            TokenKind::Ident("x".into()),
            TokenKind::String("s".into()),
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
            TokenKind::Await,
            TokenKind::Cancel,
            TokenKind::Submit,
            TokenKind::Print,
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
