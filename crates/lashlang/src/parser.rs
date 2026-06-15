use crate::ast::{
    AssignPathStep, AssignTarget, AstString, BinaryOp, Declaration, Expr, LabelMetadata,
    ProcessDecl, ProcessParam, ProcessSignalDecl, ProcessStartExpr, Program, TypeDecl, TypeExpr,
    TypeField, UnaryOp,
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
    #[error("`{keyword}` can only be used inside a loop")]
    LoopControlOutsideLoop { keyword: &'static str, span: Span },
    #[error("`{keyword}` can only be used inside a `process` body")]
    SessionProcessAdminOutsideBlock { keyword: &'static str, span: Span },
    #[error("`{keyword}` can't be used inside a `process` body")]
    ForegroundControlInsideProcess { keyword: &'static str, span: Span },
    #[error(
        "declarative trigger syntax has been removed; construct a source value and call the trigger registry register operation"
    )]
    DeclarativeTriggerRemoved { span: Span },
    #[error("invalid @label annotation: {message}")]
    InvalidLabelAnnotation { message: String, span: Span },
    #[error(
        "@label can annotate statements or process declarations, but not other declarations or another @label"
    )]
    InvalidLabelTarget { span: Span },
    #[error("expression nesting too deep (limit {limit}); flatten the program")]
    NestingTooDeep { limit: usize, span: Span },
}

impl ParseError {
    pub fn offset(&self) -> usize {
        match self {
            Self::Lex(err) => err.offset(),
            Self::Expected { span, .. }
            | Self::Unexpected { span, .. }
            | Self::LoopControlOutsideLoop { span, .. }
            | Self::SessionProcessAdminOutsideBlock { span, .. }
            | Self::ForegroundControlInsideProcess { span, .. }
            | Self::DeclarativeTriggerRemoved { span }
            | Self::InvalidLabelAnnotation { span, .. }
            | Self::InvalidLabelTarget { span }
            | Self::NestingTooDeep { span, .. } => span.start,
        }
    }
}

pub fn parse(source: &str) -> Result<Program, ParseError> {
    let tokens = lex(source)?;
    Parser {
        tokens,
        index: 0,
        loop_depth: 0,
        process_depth: 0,
        nesting_depth: 0,
    }
    .parse_program()
}

/// Maximum syntactic nesting depth (nested expressions *and* nested blocks).
/// Bounds recursive-descent stack growth so adversarial model-emitted source
/// (deeply nested brackets or `if`/`for` bodies) returns a `ParseError` instead
/// of overflowing the native stack and aborting the host.
///
/// The deepest chain is expression nesting: each level descends the full
/// precedence ladder (`parse_expr` -> ternary -> or -> and -> compare -> add ->
/// mul -> unary -> postfix -> primary -> grouping -> `parse_expr`), roughly a
/// dozen native frames carrying the large `Expr` enum. Empirically ~64 levels
/// parse comfortably on a 2 MiB thread stack while ~128 overflow it, so the
/// limit is kept well under that cliff. Block nesting (`parse_block` ->
/// `parse_statement_expr` -> `parse_if`/`parse_for`/`parse_while` -> `parse_block`) is a
/// shallower per-level chain and shares the same budget, so any mix of the two
/// stays bounded. Real generated programs nest only a handful deep, so this is
/// ample headroom; capping here also bounds every downstream AST walker
/// (validate, lower, compile, eval), since the tree can never be deeper than
/// the parser allowed.
const MAX_NESTING_DEPTH: usize = 64;

struct Parser {
    tokens: Vec<Token>,
    index: usize,
    loop_depth: usize,
    process_depth: usize,
    nesting_depth: usize,
}

impl Parser {
    fn parse_program(&mut self) -> Result<Program, ParseError> {
        let capacity = (self.tokens.len() / 20).max(1);
        let mut declarations = Vec::new();
        let mut declaration_spans = Vec::new();
        let mut expressions = Vec::with_capacity(capacity);
        let mut expression_spans = Vec::with_capacity(capacity);
        while !self.at_eof() {
            if matches!(self.peek_kind(), TokenKind::At) {
                let start = self.peek().span.start;
                let label = self.parse_label_annotation()?;
                if self.peek_contextual("process") && !self.peek_assignment_target() {
                    declarations.push(Declaration::Process(self.parse_process_decl(Some(label))?));
                    declaration_spans.push(self.span_from(start));
                    continue;
                }
                if self.peek_contextual("type")
                    || self.peek_contextual("trigger")
                    || matches!(self.peek_kind(), TokenKind::At)
                {
                    return Err(ParseError::InvalidLabelTarget {
                        span: self.peek().span,
                    });
                }
                let expr = self.parse_statement_expr()?;
                expressions.push(Expr::LabelAnnotated {
                    label,
                    expr: Box::new(expr),
                });
                let end = self
                    .tokens
                    .get(self.index.saturating_sub(1))
                    .map(|token| token.span.end)
                    .unwrap_or(start);
                expression_spans.push(Span { start, end });
                continue;
            }
            if self.peek_contextual("type") && !self.peek_assignment_target() {
                let start = self.peek().span.start;
                declarations.push(Declaration::Type(self.parse_type_decl()?));
                declaration_spans.push(self.span_from(start));
                continue;
            }
            if self.peek_contextual("process") && !self.peek_assignment_target() {
                let start = self.peek().span.start;
                declarations.push(Declaration::Process(self.parse_process_decl(None)?));
                declaration_spans.push(self.span_from(start));
                continue;
            }
            if self.peek_contextual("trigger") && !self.peek_assignment_target() {
                return Err(ParseError::DeclarativeTriggerRemoved {
                    span: self.peek().span,
                });
            }
            let start = self.peek().span.start;
            expressions.push(self.parse_statement_expr()?);
            let end = self
                .tokens
                .get(self.index.saturating_sub(1))
                .map(|token| token.span.end)
                .unwrap_or(start);
            expression_spans.push(Span { start, end });
        }
        Ok(Program::module_with_spans(
            declarations,
            declaration_spans,
            expressions,
            expression_spans,
        ))
    }

    fn span_from(&self, start: usize) -> Span {
        let end = self
            .tokens
            .get(self.index.saturating_sub(1))
            .map(|token| token.span.end)
            .unwrap_or(start);
        Span { start, end }
    }

    fn parse_type_decl(&mut self) -> Result<TypeDecl, ParseError> {
        self.expect_contextual("type")?;
        let name = self.expect_ident()?;
        self.expect_exact(TokenKind::Equal, "`=`")?;
        let ty = if matches!(self.peek_kind(), TokenKind::LBrace) {
            self.parse_type_object_body()?
        } else {
            self.parse_type_expr()?
        };
        Ok(TypeDecl { name, ty })
    }

    fn parse_process_decl(
        &mut self,
        label: Option<LabelMetadata>,
    ) -> Result<ProcessDecl, ParseError> {
        self.expect_contextual("process")?;
        let name = self.expect_ident()?;
        self.expect_exact(TokenKind::LParen, "`(`")?;
        let mut params = Vec::new();
        while !matches!(self.peek_kind(), TokenKind::RParen | TokenKind::Eof) {
            let param_name = self.expect_ident()?;
            self.expect_exact(TokenKind::Colon, "`:`")?;
            let ty = self.parse_type_expr()?;
            params.push(ProcessParam {
                name: param_name,
                ty,
            });
            if matches!(self.peek_kind(), TokenKind::Comma) {
                self.bump();
                continue;
            }
            break;
        }
        self.expect_exact(TokenKind::RParen, "`)`")?;
        let signals = if self.peek_contextual("signals") && !self.peek_assignment_target() {
            self.parse_process_signal_decls()?
        } else {
            Vec::new()
        };
        let return_ty = if matches!(self.peek_kind(), TokenKind::Minus)
            && self
                .tokens
                .get(self.index + 1)
                .is_some_and(|token| matches!(token.kind, TokenKind::Greater))
        {
            self.bump();
            self.bump();
            Some(self.parse_type_expr()?)
        } else {
            None
        };
        self.process_depth += 1;
        let body = self.parse_block()?;
        self.process_depth -= 1;
        Ok(ProcessDecl {
            name,
            params,
            signals,
            return_ty,
            label,
            body,
        })
    }

    fn parse_process_signal_decls(&mut self) -> Result<Vec<ProcessSignalDecl>, ParseError> {
        self.expect_contextual("signals")?;
        self.expect_exact(TokenKind::LBrace, "`{`")?;
        let mut signals = Vec::new();
        while !matches!(self.peek_kind(), TokenKind::RBrace | TokenKind::Eof) {
            let name = self.expect_ident()?;
            self.expect_exact(TokenKind::Colon, "`:`")?;
            let ty = self.parse_type_expr()?;
            signals.push(ProcessSignalDecl { name, ty });
            if matches!(self.peek_kind(), TokenKind::Comma) {
                self.bump();
                if matches!(self.peek_kind(), TokenKind::RBrace) {
                    break;
                }
                continue;
            }
            break;
        }
        self.expect_exact(TokenKind::RBrace, "`}`")?;
        Ok(signals)
    }

    fn parse_statement_expr(&mut self) -> Result<Expr, ParseError> {
        match self.peek_kind() {
            TokenKind::If => self.parse_if(),
            TokenKind::For => self.parse_for(),
            TokenKind::Submit => self.parse_submit(),
            TokenKind::Cancel => self.parse_cancel(),
            TokenKind::Print => self.parse_print(),
            TokenKind::Call => Err(ParseError::Unexpected {
                found: "`call`".to_string(),
                span: self.peek().span,
            }),
            TokenKind::Ident(name) if name == "let" && !self.peek_assignment_target() => {
                self.parse_let_assign()
            }
            TokenKind::Ident(name)
                if matches!(name.as_str(), "yield" | "wake" | "finish" | "fail")
                    && !self.peek_assignment_target() =>
            {
                self.parse_processes()
            }
            TokenKind::Ident(name) if name == "break" && !self.peek_assignment_target() => {
                self.parse_loop_control("break")
            }
            TokenKind::Ident(name) if name == "continue" && !self.peek_assignment_target() => {
                self.parse_loop_control("continue")
            }
            TokenKind::Ident(name) if name == "while" && !self.peek_assignment_target() => {
                self.parse_while()
            }
            TokenKind::Ident(_) if self.peek_assignment_target() => self.parse_assign(),
            _ => self.parse_expr(),
        }
    }

    fn parse_let_assign(&mut self) -> Result<Expr, ParseError> {
        self.bump();
        self.parse_assign()
    }

    fn parse_assign(&mut self) -> Result<Expr, ParseError> {
        let target = self.parse_assignment_target()?;
        self.expect_exact(TokenKind::Equal, "`=`")?;
        let expr = self.parse_expr()?;
        Ok(Expr::Assign {
            target,
            expr: Box::new(expr),
        })
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

    fn parse_if(&mut self) -> Result<Expr, ParseError> {
        self.bump();
        let condition = self.parse_expr()?;
        let then_block = self.parse_block()?;
        let else_block = if matches!(self.peek_kind(), TokenKind::Else) {
            self.bump();
            if matches!(self.peek_kind(), TokenKind::If) {
                self.parse_if()?
            } else {
                self.parse_block()?
            }
        } else {
            Expr::Block(Vec::new())
        };
        Ok(Expr::If {
            condition: Box::new(condition),
            then_block: Box::new(then_block),
            else_block: Box::new(else_block),
        })
    }

    fn parse_for(&mut self) -> Result<Expr, ParseError> {
        self.bump();
        let binding = self.expect_ident()?;
        self.expect_exact(TokenKind::In, "`in`")?;
        let iterable = self.parse_expr()?;
        self.loop_depth += 1;
        let body = self.parse_block()?;
        self.loop_depth -= 1;
        Ok(Expr::For {
            binding,
            iterable: Box::new(iterable),
            body: Box::new(body),
        })
    }

    fn parse_while(&mut self) -> Result<Expr, ParseError> {
        self.bump();
        let condition = self.parse_expr()?;
        self.loop_depth += 1;
        let body = self.parse_block()?;
        self.loop_depth -= 1;
        Ok(Expr::While {
            condition: Box::new(condition),
            body: Box::new(body),
        })
    }

    fn parse_loop_control(&mut self, keyword: &'static str) -> Result<Expr, ParseError> {
        let span = self.bump().span;
        if self.loop_depth == 0 {
            return Err(ParseError::LoopControlOutsideLoop { keyword, span });
        }
        Ok(match keyword {
            "break" => Expr::Break,
            "continue" => Expr::Continue,
            _ => unreachable!("unknown loop control keyword"),
        })
    }

    fn parse_submit(&mut self) -> Result<Expr, ParseError> {
        let span = self.bump().span;
        if self.process_depth > 0 {
            return Err(ParseError::ForegroundControlInsideProcess {
                keyword: "submit",
                span,
            });
        }
        let expr = if matches!(self.peek_kind(), TokenKind::RBrace | TokenKind::Eof) {
            None
        } else {
            Some(self.parse_expr()?)
        };
        Ok(Expr::Submit(expr.map(Box::new)))
    }

    fn parse_print(&mut self) -> Result<Expr, ParseError> {
        let span = self.bump().span;
        if self.process_depth > 0 {
            return Err(ParseError::ForegroundControlInsideProcess {
                keyword: "print",
                span,
            });
        }
        Ok(Expr::Print(Box::new(self.parse_expr()?)))
    }

    fn parse_processes(&mut self) -> Result<Expr, ParseError> {
        let token = self.bump().clone();
        let TokenKind::Ident(keyword) = token.kind else {
            unreachable!("process admins are contextual identifiers");
        };
        let keyword_static = match keyword.as_str() {
            "yield" => "yield",
            "wake" => "wake",
            "finish" => "finish",
            "fail" => "fail",
            _ => unreachable!("unknown process admin keyword"),
        };
        if self.process_depth == 0 {
            return Err(ParseError::SessionProcessAdminOutsideBlock {
                keyword: keyword_static,
                span: token.span,
            });
        }
        let stmt = match keyword_static {
            "yield" => Expr::Yield(Box::new(self.parse_expr()?)),
            "wake" => Expr::Wake(Box::new(self.parse_expr()?)),
            "finish" => {
                let expr = if matches!(self.peek_kind(), TokenKind::RBrace | TokenKind::Eof) {
                    None
                } else {
                    Some(self.parse_expr()?)
                };
                Expr::Finish(expr.map(Box::new))
            }
            "fail" => Expr::Fail(Box::new(self.parse_expr()?)),
            _ => unreachable!("unknown process admin keyword"),
        };
        Ok(stmt)
    }

    fn parse_cancel(&mut self) -> Result<Expr, ParseError> {
        self.bump();
        Ok(Expr::Cancel(Box::new(self.parse_expr()?)))
    }

    /// Account for entering one more level of syntactic nesting, rejecting
    /// input that would recurse deep enough to overflow the native stack.
    /// Pair every successful call with [`Parser::leave_nesting`].
    fn enter_nesting(&mut self) -> Result<(), ParseError> {
        if self.nesting_depth >= MAX_NESTING_DEPTH {
            return Err(ParseError::NestingTooDeep {
                limit: MAX_NESTING_DEPTH,
                span: self.peek().span,
            });
        }
        self.nesting_depth += 1;
        Ok(())
    }

    fn leave_nesting(&mut self) {
        self.nesting_depth -= 1;
    }

    fn parse_block(&mut self) -> Result<Expr, ParseError> {
        // Nested blocks (`if`/`for` bodies, bare braces) recurse through here
        // without passing through `parse_expr`, so they need their own guard.
        self.enter_nesting()?;
        let result = self.parse_block_inner();
        self.leave_nesting();
        result
    }

    fn parse_block_inner(&mut self) -> Result<Expr, ParseError> {
        self.expect_exact(TokenKind::LBrace, "`{`")?;
        let mut expressions = Vec::new();
        while !matches!(self.peek_kind(), TokenKind::RBrace | TokenKind::Eof) {
            if matches!(self.peek_kind(), TokenKind::At) {
                expressions.push(self.parse_annotated_statement()?);
            } else {
                expressions.push(self.parse_statement_expr()?);
            }
        }
        self.expect_exact(TokenKind::RBrace, "`}`")?;
        Ok(Expr::Block(expressions))
    }

    fn parse_annotated_statement(&mut self) -> Result<Expr, ParseError> {
        let label = self.parse_label_annotation()?;
        if matches!(self.peek_kind(), TokenKind::At)
            || self.peek_contextual("type")
            || self.peek_contextual("process")
            || self.peek_contextual("trigger")
        {
            return Err(ParseError::InvalidLabelTarget {
                span: self.peek().span,
            });
        }
        let expr = self.parse_statement_expr()?;
        Ok(Expr::LabelAnnotated {
            label,
            expr: Box::new(expr),
        })
    }

    fn parse_label_annotation(&mut self) -> Result<LabelMetadata, ParseError> {
        let span = self.peek().span;
        self.expect_exact(TokenKind::At, "`@`")?;
        self.expect_contextual("label")?;
        self.expect_exact(TokenKind::LParen, "`(`")?;

        let mut title = None;
        let mut description = None;
        while !matches!(self.peek_kind(), TokenKind::RParen | TokenKind::Eof) {
            let key_span = self.peek().span;
            let key = self.expect_ident()?;
            self.expect_exact(TokenKind::Colon, "`:`")?;
            let value = self.expect_string_literal()?;
            match key.as_str() {
                "title" => {
                    if title.replace(value).is_some() {
                        return Err(ParseError::InvalidLabelAnnotation {
                            message: "duplicate `title` field".to_string(),
                            span: key_span,
                        });
                    }
                }
                "description" => {
                    if description.replace(value).is_some() {
                        return Err(ParseError::InvalidLabelAnnotation {
                            message: "duplicate `description` field".to_string(),
                            span: key_span,
                        });
                    }
                }
                _ => {
                    return Err(ParseError::InvalidLabelAnnotation {
                        message: format!("unknown field `{key}`"),
                        span: key_span,
                    });
                }
            }
            if matches!(self.peek_kind(), TokenKind::Comma) {
                self.bump();
                if matches!(self.peek_kind(), TokenKind::RParen) {
                    break;
                }
                continue;
            }
            break;
        }
        self.expect_exact(TokenKind::RParen, "`)`")?;
        let Some(title) = title else {
            return Err(ParseError::InvalidLabelAnnotation {
                message: "`title` is required".to_string(),
                span,
            });
        };
        Ok(LabelMetadata { title, description })
    }

    fn parse_expr(&mut self) -> Result<Expr, ParseError> {
        // Every nested expression (parenthesised, list/record element, index,
        // ternary, ...) funnels through here, so one depth guard at this
        // chokepoint bounds total recursive-descent stack growth.
        self.enter_nesting()?;
        let result = self.parse_ternary();
        self.leave_nesting();
        result
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
        Ok(Expr::If {
            condition: Box::new(condition),
            then_block: Box::new(then_expr),
            else_block: Box::new(else_expr),
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
                    if matches!(self.peek_kind(), TokenKind::LParen) {
                        let args = self.parse_call_arguments()?;
                        expr = Expr::ReceiverCall {
                            receiver: Box::new(expr),
                            operation: field,
                            args,
                        };
                    } else {
                        expr = Expr::Field {
                            target: Box::new(expr),
                            field,
                        };
                    }
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
                if name == "parallel"
                    && self
                        .tokens
                        .get(self.index + 1)
                        .is_some_and(|token| matches!(token.kind, TokenKind::LBrace))
                {
                    return Err(ParseError::Unexpected {
                        found: "`parallel`".to_string(),
                        span: self.peek().span,
                    });
                }
                let name = name.clone();
                self.bump();
                if name == "sleep" {
                    return self.parse_sleep_expr();
                }
                if name == "start" && matches!(self.peek_kind(), TokenKind::Call) {
                    return Err(ParseError::Unexpected {
                        found: "`start call`".to_string(),
                        span: self.tokens[self.index.saturating_sub(1)].span,
                    });
                }
                if name == "start"
                    && (matches!(self.peek_kind(), TokenKind::Ident(_))
                        || matches!(self.peek_kind(), TokenKind::LBrace)
                        || self.paren_group_followed_by_lbrace())
                {
                    return self.parse_process_start_expr();
                }
                if name == "Type" && matches!(self.peek_kind(), TokenKind::LBrace) {
                    let ty = self.parse_type_object()?;
                    return Ok(Expr::TypeLiteral(Box::new(ty)));
                }
                if matches!(self.peek_kind(), TokenKind::LParen) {
                    let args = self.parse_call_arguments()?;
                    if name == "wait_signal" {
                        if args.len() != 1 {
                            return Err(ParseError::Expected {
                                expected: "one signal name argument",
                                found: format!("{} arguments", args.len()),
                                span: self.tokens[self.index.saturating_sub(1)].span,
                            });
                        }
                        return Ok(Expr::WaitSignal {
                            name: static_signal_name_arg(&args[0], "wait_signal")?,
                        });
                    }
                    if name == "signal_run" {
                        if args.len() != 3 {
                            return Err(ParseError::Expected {
                                expected: "run handle, signal name, and payload arguments",
                                found: format!("{} arguments", args.len()),
                                span: self.tokens[self.index.saturating_sub(1)].span,
                            });
                        }
                        return Ok(Expr::SignalRun {
                            run: Box::new(args[0].clone()),
                            name: static_signal_name_arg(&args[1], "signal_run")?,
                            payload: Box::new(args[2].clone()),
                        });
                    }
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
            TokenKind::Call => Err(ParseError::Unexpected {
                found: "`call`".to_string(),
                span: self.peek().span,
            }),
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

    fn parse_call_arguments(&mut self) -> Result<Vec<Expr>, ParseError> {
        self.expect_exact(TokenKind::LParen, "`(`")?;
        let mut args = Vec::new();
        if !matches!(self.peek_kind(), TokenKind::RParen) {
            loop {
                args.push(self.parse_expr()?);
                if matches!(self.peek_kind(), TokenKind::Comma) {
                    self.bump();
                    if matches!(self.peek_kind(), TokenKind::RParen) {
                        break;
                    }
                    continue;
                }
                break;
            }
        }
        self.expect_exact(TokenKind::RParen, "`)`")?;
        Ok(args)
    }

    fn parse_named_arguments(&mut self) -> Result<Vec<(AstString, Expr)>, ParseError> {
        let mut entries = Vec::new();
        while !matches!(self.peek_kind(), TokenKind::RParen | TokenKind::Eof) {
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

    fn parse_process_start_expr(&mut self) -> Result<Expr, ParseError> {
        if matches!(self.peek_kind(), TokenKind::LBrace) || self.paren_group_followed_by_lbrace() {
            return Err(ParseError::Unexpected {
                found: "inline `start` process body".to_string(),
                span: self.peek().span,
            });
        }
        let process = self.expect_ident()?;
        self.expect_exact(TokenKind::LParen, "`(`")?;
        let args = self.parse_named_arguments()?;
        self.expect_exact(TokenKind::RParen, "`)`")?;
        Ok(Expr::StartProcess(ProcessStartExpr { process, args }))
    }

    fn parse_sleep_expr(&mut self) -> Result<Expr, ParseError> {
        if matches!(self.peek_kind(), TokenKind::For) {
            self.bump();
            return Ok(Expr::SleepFor(Box::new(self.parse_expr()?)));
        }
        if self.peek_contextual("until") {
            self.bump();
            return Ok(Expr::SleepUntil(Box::new(self.parse_expr()?)));
        }
        Err(ParseError::Expected {
            expected: "`for` or `until`",
            found: render_kind(self.peek_kind()),
            span: self.peek().span,
        })
    }

    fn parse_type_object(&mut self) -> Result<TypeExpr, ParseError> {
        self.expect_exact(TokenKind::LBrace, "`{`")?;
        self.parse_type_object_body_after_lbrace()
    }

    fn parse_type_object_body(&mut self) -> Result<TypeExpr, ParseError> {
        self.expect_exact(TokenKind::LBrace, "`{`")?;
        self.parse_type_object_body_after_lbrace()
    }

    fn parse_type_object_body_after_lbrace(&mut self) -> Result<TypeExpr, ParseError> {
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
            TokenKind::String(value) => {
                self.bump();
                Ok(TypeExpr::Enum(vec![value]))
            }
            // A bare `{` in type position is the classic "forgot the
            // `Type` keyword" mistake (`foo: { ok: bool }` instead of
            // `foo: Type { ok: bool }`). Surface a targeted diagnostic
            // rather than the generic "expected type expression" shrug.
            TokenKind::LBrace => self.parse_type_object_body(),
            TokenKind::Ident(_name) => {
                let name = self.parse_type_name()?;
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
                    "Process" => {
                        self.expect_exact(TokenKind::Less, "`<`")?;
                        let input = self.parse_type_expr()?;
                        self.expect_exact(TokenKind::Comma, "`,`")?;
                        let output = self.parse_type_expr()?;
                        self.expect_exact(TokenKind::Greater, "`>`")?;
                        Ok(TypeExpr::Process {
                            input: Box::new(input),
                            output: Box::new(output),
                            input_count: 1,
                        })
                    }
                    "TriggerHandle" => {
                        self.expect_exact(TokenKind::Less, "`<`")?;
                        let event = self.parse_type_expr()?;
                        self.expect_exact(TokenKind::Greater, "`>`")?;
                        Ok(TypeExpr::TriggerHandle(Box::new(event)))
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

    fn parse_type_name(&mut self) -> Result<AstString, ParseError> {
        let mut path = vec![self.expect_ident()?];
        while matches!(self.peek_kind(), TokenKind::Dot) {
            self.bump();
            path.push(self.expect_ident()?);
        }
        Ok(path
            .iter()
            .map(AstString::as_str)
            .collect::<Vec<_>>()
            .join(".")
            .into())
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

    fn paren_group_followed_by_lbrace(&self) -> bool {
        if !matches!(self.peek_kind(), TokenKind::LParen) {
            return false;
        }
        let mut depth = 0usize;
        for (index, token) in self.tokens.iter().enumerate().skip(self.index) {
            match &token.kind {
                TokenKind::LParen => depth += 1,
                TokenKind::RParen => {
                    depth = depth.saturating_sub(1);
                    if depth == 0 {
                        return self
                            .tokens
                            .get(index + 1)
                            .is_some_and(|next| matches!(next.kind, TokenKind::LBrace));
                    }
                }
                TokenKind::Eof => return false,
                _ => {}
            }
        }
        false
    }

    fn peek_contextual(&self, keyword: &str) -> bool {
        matches!(self.peek_kind(), TokenKind::Ident(name) if name.as_str() == keyword)
    }

    fn expect_contextual(&mut self, keyword: &'static str) -> Result<(), ParseError> {
        let token = self.bump();
        match &token.kind {
            TokenKind::Ident(name) if name.as_str() == keyword => Ok(()),
            other => Err(ParseError::Expected {
                expected: keyword,
                found: render_kind(other),
                span: token.span,
            }),
        }
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
}

fn static_signal_name_arg(expr: &Expr, call: &'static str) -> Result<AstString, ParseError> {
    if let Expr::String(name) = expr {
        return Ok(name.clone());
    }
    Err(ParseError::Unexpected {
        found: format!("non-literal signal name in `{call}`"),
        span: Span { start: 0, end: 0 },
    })
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
        TokenKind::Await => "await",
        TokenKind::Cancel => "cancel",
        TokenKind::Submit => "submit",
        TokenKind::Print => "print",
        TokenKind::Call => "call",
        TokenKind::Ident(name) if matches!(name.as_str(), "yield" | "wake" | "finish" | "fail") => {
            return Some(match name.as_str() {
                "yield" => "yield",
                "wake" => "wake",
                "finish" => "finish",
                "fail" => "fail",
                _ => unreachable!(),
            });
        }
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
        TokenKind::At => "`@`".to_string(),
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

include!("parser/tests.rs");
