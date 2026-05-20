use compact_str::CompactString;
use serde::{Deserialize, Serialize};

use crate::lexer::Span;

pub type AstString = CompactString;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Program {
    pub expr: Expr,
    #[serde(default, skip)]
    pub expression_spans: Vec<Span>,
}

impl Program {
    pub fn block(expressions: Vec<Expr>) -> Self {
        Self {
            expr: Expr::Block(expressions),
            expression_spans: Vec::new(),
        }
    }

    pub(crate) fn block_with_spans(expressions: Vec<Expr>, expression_spans: Vec<Span>) -> Self {
        Self {
            expr: Expr::Block(expressions),
            expression_spans,
        }
    }
}

impl PartialEq for Program {
    fn eq(&self, other: &Self) -> bool {
        self.expr == other.expr
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AssignTarget {
    pub root: AstString,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub steps: Vec<AssignPathStep>,
}

impl AssignTarget {
    pub fn variable(root: AstString) -> Self {
        Self {
            root,
            steps: Vec::new(),
        }
    }

    pub fn is_simple(&self) -> bool {
        self.steps.is_empty()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum AssignPathStep {
    Field(AstString),
    Index(Expr),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Expr {
    Block(Vec<Expr>),
    Null,
    Bool(bool),
    Number(f64),
    String(AstString),
    Variable(AstString),
    List(Vec<Expr>),
    Record(Vec<(AstString, Expr)>),
    Assign {
        target: AssignTarget,
        expr: Box<Expr>,
    },
    If {
        condition: Box<Expr>,
        then_block: Box<Expr>,
        else_block: Box<Expr>,
    },
    For {
        binding: AstString,
        iterable: Box<Expr>,
        body: Box<Expr>,
    },
    Break,
    Continue,
    ToolCall {
        mode: ToolCallMode,
        call: CallExpr,
    },
    StartProcess(ProcessStartExpr),
    Await(Box<Expr>),
    ResultUnwrap(Box<Expr>),
    Cancel(Box<Expr>),
    Print(Box<Expr>),
    Submit(Option<Box<Expr>>),
    Yield(Box<Expr>),
    Wake(Box<Expr>),
    Finish(Option<Box<Expr>>),
    Fail(Box<Expr>),
    BuiltinCall {
        name: AstString,
        args: Vec<Expr>,
    },
    Field {
        target: Box<Expr>,
        field: AstString,
    },
    Index {
        target: Box<Expr>,
        index: Box<Expr>,
    },
    Unary {
        op: UnaryOp,
        expr: Box<Expr>,
    },
    Binary {
        left: Box<Expr>,
        op: BinaryOp,
        right: Box<Expr>,
    },
    TypeLiteral(Box<TypeExpr>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ToolCallMode {
    Call,
    Start,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum TypeExpr {
    Any,
    Str,
    Int,
    Float,
    Bool,
    Dict,
    /// The literal `null` type; usually only useful as part of a
    /// `Union` (e.g. `str | null` for a nullable field).
    Null,
    Enum(Vec<AstString>),
    List(Box<TypeExpr>),
    Object(Vec<TypeField>),
    Ref(AstString),
    /// Union of alternative type shapes, e.g. `str | int | null`.
    /// Always has two or more variants; single-variant parses collapse
    /// to the underlying `TypeExpr` in the parser.
    Union(Vec<TypeExpr>),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TypeField {
    pub name: AstString,
    pub ty: TypeExpr,
    pub optional: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CallExpr {
    pub name: AstString,
    pub args: Vec<(AstString, Expr)>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProcessStartExpr {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<Box<Expr>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_ms: Option<Box<Expr>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input: Option<Box<Expr>>,
    pub body: Box<Expr>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnaryOp {
    Negate,
    Not,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BinaryOp {
    Add,
    Subtract,
    Multiply,
    Divide,
    Modulo,
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    And,
    Or,
}
