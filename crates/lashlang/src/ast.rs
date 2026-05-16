use crate::lexer::Span;
use compact_str::CompactString;
use serde::{Deserialize, Serialize};

pub type AstString = CompactString;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Program {
    pub statements: Vec<Stmt>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub statement_spans: Vec<Span>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Stmt {
    Assign {
        target: AssignTarget,
        expr: Expr,
    },
    Expr(Expr),
    Call(CallExpr),
    Cancel(Expr),
    Print(Expr),
    If {
        condition: Expr,
        then_block: Vec<Stmt>,
        else_block: Vec<Stmt>,
    },
    For {
        binding: AstString,
        iterable: Expr,
        body: Vec<Stmt>,
    },
    Break,
    Continue,
    Parallel {
        branches: ParallelBranches,
    },
    Submit(Option<Expr>),
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
pub enum ParallelBranches {
    Positional(Vec<Stmt>),
    Named(Vec<NamedParallelBranch>),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NamedParallelBranch {
    pub name: AstString,
    pub stmt: Stmt,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Expr {
    Null,
    Bool(bool),
    Number(f64),
    String(AstString),
    Variable(AstString),
    List(Vec<Expr>),
    Record(Vec<(AstString, Expr)>),
    ToolCall(CallExpr),
    StartToolCall(CallExpr),
    Parallel {
        branches: ParallelBranches,
    },
    Await(Box<Expr>),
    ResultUnwrap(Box<Expr>),
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
    Conditional {
        condition: Box<Expr>,
        then_expr: Box<Expr>,
        else_expr: Box<Expr>,
    },
    Binary {
        left: Box<Expr>,
        op: BinaryOp,
        right: Box<Expr>,
    },
    TypeLiteral(Box<TypeExpr>),
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
