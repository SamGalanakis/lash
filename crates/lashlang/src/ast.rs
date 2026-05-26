use compact_str::CompactString;
use serde::{Deserialize, Serialize};

use crate::lexer::Span;

pub type AstString = CompactString;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Program {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub declarations: Vec<Declaration>,
    pub main: Expr,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub declaration_spans: Vec<Span>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub expression_spans: Vec<Span>,
}

impl Program {
    pub fn block(expressions: Vec<Expr>) -> Self {
        Self {
            declarations: Vec::new(),
            main: Expr::Block(expressions),
            declaration_spans: Vec::new(),
            expression_spans: Vec::new(),
        }
    }

    pub(crate) fn module_with_spans(
        declarations: Vec<Declaration>,
        declaration_spans: Vec<Span>,
        expressions: Vec<Expr>,
        expression_spans: Vec<Span>,
    ) -> Self {
        Self {
            declarations,
            main: Expr::Block(expressions),
            declaration_spans,
            expression_spans,
        }
    }

    pub fn process(&self, name: &str) -> Option<&ProcessDecl> {
        self.declarations
            .iter()
            .find_map(|declaration| match declaration {
                Declaration::Process(process) if process.name.as_str() == name => Some(process),
                _ => None,
            })
    }
}

impl PartialEq for Program {
    fn eq(&self, other: &Self) -> bool {
        self.declarations == other.declarations && self.main == other.main
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Declaration {
    Type(TypeDecl),
    Process(ProcessDecl),
    Trigger(TriggerDecl),
    Schedule(ScheduleDecl),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TypeDecl {
    pub name: AstString,
    pub ty: TypeExpr,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProcessDecl {
    pub name: AstString,
    pub params: Vec<ProcessParam>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub return_ty: Option<TypeExpr>,
    pub body: Expr,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProcessParam {
    pub name: AstString,
    pub ty: TypeExpr,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TriggerDecl {
    pub name: AstString,
    pub source: TriggerSource,
    pub event_binding: AstString,
    pub body: Expr,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum TriggerSource {
    Binding {
        resource: ResourceRefExpr,
        event: AstString,
    },
    Each {
        resource_type: AstString,
        event: AstString,
        resource_binding: AstString,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ScheduleDecl {
    pub name: AstString,
    pub cadence: ScheduleCadence,
    pub tick_binding: AstString,
    pub body: Expr,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ScheduleCadence {
    Cron {
        expression: Expr,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        options: Vec<(AstString, Expr)>,
    },
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
    StartProcess(ProcessStartExpr),
    ResourceRef(ResourceRefExpr),
    ReceiverCall {
        receiver: Box<Expr>,
        operation: AstString,
        args: Vec<Expr>,
    },
    Await(Box<Expr>),
    SleepFor(Box<Expr>),
    SleepUntil(Box<Expr>),
    WaitSignal,
    SignalRun {
        run: Box<Expr>,
        payload: Box<Expr>,
    },
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
pub struct ProcessStartExpr {
    pub process: AstString,
    pub args: Vec<(AstString, Expr)>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceRefExpr {
    pub resource_type: AstString,
    pub alias: AstString,
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
