use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Program {
    pub statements: Vec<Stmt>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Stmt {
    Assign {
        name: String,
        expr: Expr,
    },
    Expr(Expr),
    Call(CallExpr),
    Cancel(Expr),
    Observe(Expr),
    If {
        condition: Expr,
        then_block: Vec<Stmt>,
        else_block: Vec<Stmt>,
    },
    For {
        binding: String,
        iterable: Expr,
        body: Vec<Stmt>,
    },
    Parallel {
        branches: Vec<Stmt>,
    },
    Finish(Option<Expr>),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Expr {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Variable(String),
    List(Vec<Expr>),
    Record(Vec<(String, Expr)>),
    ToolCall(CallExpr),
    StartToolCall(CallExpr),
    Parallel {
        branches: Vec<Stmt>,
    },
    Await(Box<Expr>),
    BuiltinCall {
        name: String,
        args: Vec<Expr>,
    },
    Field {
        target: Box<Expr>,
        field: String,
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
    Enum(Vec<String>),
    List(Box<TypeExpr>),
    Object(Vec<TypeField>),
    Ref(String),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TypeField {
    pub name: String,
    pub ty: TypeExpr,
    pub optional: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CallExpr {
    pub name: String,
    pub args: Vec<(String, Expr)>,
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
