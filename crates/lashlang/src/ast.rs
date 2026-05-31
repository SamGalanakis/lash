use compact_str::CompactString;
use serde::{Deserialize, Serialize};
use std::fmt;

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
    pub process_name: AstString,
    pub args: Vec<(AstString, TriggerArg)>,
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
pub enum TriggerArg {
    EventBinding(AstString),
    ResourceBinding(AstString),
    ResourceRef(ResourceRefExpr),
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
    While {
        condition: Box<Expr>,
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

impl Expr {
    /// Yields every direct child expression of `self` in evaluation order.
    ///
    /// This is the single structural-traversal primitive: any pass that only
    /// needs to recurse into the sub-expressions of a node (without caring
    /// about the node's own kind) can fold over `children()` instead of
    /// re-spelling the full `match`. Leaf nodes (`Null`, `Bool`, `Number`,
    /// `String`, `Variable`, `Break`, `Continue`, `WaitSignal`,
    /// `ResourceRef`, `TypeLiteral`) yield nothing.
    ///
    /// `Assign` includes any dynamic index expressions in its `target` path
    /// (in path order) before the assigned value, matching the order in which
    /// the compiler and linker visit them.
    pub fn children(&self) -> ExprChildren<'_> {
        let mut buffer = SmallExprVec::new();
        match self {
            Expr::Null
            | Expr::Bool(_)
            | Expr::Number(_)
            | Expr::String(_)
            | Expr::Variable(_)
            | Expr::Break
            | Expr::Continue
            | Expr::WaitSignal
            | Expr::ResourceRef(_)
            | Expr::TypeLiteral(_) => {}
            Expr::Block(expressions) | Expr::List(expressions) => {
                buffer.extend(expressions.iter());
            }
            Expr::Record(entries) => buffer.extend(entries.iter().map(|(_, value)| value)),
            Expr::Assign { target, expr } => {
                for step in &target.steps {
                    if let AssignPathStep::Index(index) = step {
                        buffer.push(index);
                    }
                }
                buffer.push(expr);
            }
            Expr::If {
                condition,
                then_block,
                else_block,
            } => {
                buffer.push(condition);
                buffer.push(then_block);
                buffer.push(else_block);
            }
            Expr::For { iterable, body, .. } => {
                buffer.push(iterable);
                buffer.push(body);
            }
            Expr::While { condition, body } => {
                buffer.push(condition);
                buffer.push(body);
            }
            Expr::StartProcess(start) => buffer.extend(start.args.iter().map(|(_, value)| value)),
            Expr::ReceiverCall { receiver, args, .. } => {
                buffer.push(receiver);
                buffer.extend(args.iter());
            }
            Expr::SignalRun { run, payload } => {
                buffer.push(run);
                buffer.push(payload);
            }
            Expr::Await(expr)
            | Expr::SleepFor(expr)
            | Expr::SleepUntil(expr)
            | Expr::ResultUnwrap(expr)
            | Expr::Cancel(expr)
            | Expr::Print(expr)
            | Expr::Yield(expr)
            | Expr::Wake(expr)
            | Expr::Fail(expr)
            | Expr::Unary { expr, .. } => buffer.push(expr),
            Expr::Submit(expr) | Expr::Finish(expr) => {
                if let Some(expr) = expr {
                    buffer.push(expr);
                }
            }
            Expr::BuiltinCall { args, .. } => buffer.extend(args.iter()),
            Expr::Field { target, .. } => buffer.push(target),
            Expr::Index { target, index } => {
                buffer.push(target);
                buffer.push(index);
            }
            Expr::Binary { left, right, .. } => {
                buffer.push(left);
                buffer.push(right);
            }
        }
        ExprChildren {
            buffer,
            position: 0,
        }
    }
}

type SmallExprVec<'expr> = smallvec::SmallVec<[&'expr Expr; 3]>;

/// Iterator over the direct child expressions yielded by [`Expr::children`].
pub struct ExprChildren<'expr> {
    buffer: SmallExprVec<'expr>,
    position: usize,
}

impl<'expr> Iterator for ExprChildren<'expr> {
    type Item = &'expr Expr;

    fn next(&mut self) -> Option<Self::Item> {
        let item = self.buffer.get(self.position).copied();
        if item.is_some() {
            self.position += 1;
        }
        item
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.buffer.len() - self.position;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for ExprChildren<'_> {}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
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

pub fn format_type_expr(ty: &TypeExpr) -> String {
    match ty {
        TypeExpr::Any => "any".to_string(),
        TypeExpr::Str => "str".to_string(),
        TypeExpr::Int => "int".to_string(),
        TypeExpr::Float => "float".to_string(),
        TypeExpr::Bool => "bool".to_string(),
        TypeExpr::Dict => "dict".to_string(),
        TypeExpr::Null => "null".to_string(),
        TypeExpr::Enum(values) => format!(
            "enum[{}]",
            values
                .iter()
                .map(|value| format!("\"{value}\""))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        TypeExpr::List(item) => format!("list[{}]", format_type_expr(item)),
        TypeExpr::Object(fields) => {
            let fields = fields
                .iter()
                .map(|field| {
                    let optional = if field.optional { "?" } else { "" };
                    format!(
                        "{}: {}{}",
                        field.name,
                        format_type_expr(&field.ty),
                        optional
                    )
                })
                .collect::<Vec<_>>()
                .join(", ");
            format!("{{ {fields} }}")
        }
        TypeExpr::Ref(name) => name.to_string(),
        TypeExpr::Union(items) => items
            .iter()
            .map(format_type_expr)
            .collect::<Vec<_>>()
            .join(" | "),
    }
}

impl fmt::Display for TypeExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&format_type_expr(self))
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
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
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub path: Vec<AstString>,
    pub resource_type: AstString,
    pub alias: AstString,
}

impl ResourceRefExpr {
    pub fn unresolved(path: Vec<AstString>) -> Self {
        Self {
            path,
            resource_type: AstString::default(),
            alias: AstString::default(),
        }
    }

    pub fn resolved(
        path: Vec<AstString>,
        resource_type: impl Into<AstString>,
        alias: impl Into<AstString>,
    ) -> Self {
        Self {
            path,
            resource_type: resource_type.into(),
            alias: alias.into(),
        }
    }

    pub fn path_string(&self) -> String {
        if self.path.is_empty() {
            format!("{}.{}", self.resource_type, self.alias)
        } else {
            self.path
                .iter()
                .map(AstString::as_str)
                .collect::<Vec<_>>()
                .join(".")
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn type_expr_formatting_covers_nested_shapes() {
        let ty = TypeExpr::Object(vec![
            TypeField {
                name: "status".into(),
                ty: TypeExpr::Enum(vec!["ok".into(), "err".into()]),
                optional: false,
            },
            TypeField {
                name: "tags".into(),
                ty: TypeExpr::List(Box::new(TypeExpr::Str)),
                optional: true,
            },
            TypeField {
                name: "owner".into(),
                ty: TypeExpr::Ref("User".into()),
                optional: false,
            },
            TypeField {
                name: "value".into(),
                ty: TypeExpr::Union(vec![TypeExpr::Int, TypeExpr::Null]),
                optional: false,
            },
        ]);

        assert_eq!(
            format_type_expr(&ty),
            r#"{ status: enum["ok", "err"], tags: list[str]?, owner: User, value: int | null }"#
        );
        assert_eq!(ty.to_string(), format_type_expr(&ty));
    }

    fn var(name: &str) -> Expr {
        Expr::Variable(name.into())
    }

    fn child_vars(expr: &Expr) -> Vec<String> {
        expr.children()
            .map(|child| match child {
                Expr::Variable(name) => name.to_string(),
                other => format!("{other:?}"),
            })
            .collect()
    }

    #[test]
    fn children_yields_leaves_as_empty() {
        for leaf in [
            Expr::Null,
            Expr::Bool(true),
            Expr::Number(1.0),
            Expr::String("s".into()),
            var("x"),
            Expr::Break,
            Expr::Continue,
            Expr::WaitSignal,
            Expr::TypeLiteral(Box::new(TypeExpr::Str)),
        ] {
            let children: Vec<_> = leaf.children().collect();
            assert!(children.is_empty(), "{leaf:?} should have no children");
        }
    }

    #[test]
    fn children_yields_composite_subexpressions_in_order() {
        let block = Expr::Block(vec![var("a"), var("b"), var("c")]);
        assert_eq!(child_vars(&block), ["a", "b", "c"]);

        let record = Expr::Record(vec![("k1".into(), var("v1")), ("k2".into(), var("v2"))]);
        assert_eq!(child_vars(&record), ["v1", "v2"]);

        let if_expr = Expr::If {
            condition: Box::new(var("cond")),
            then_block: Box::new(var("then")),
            else_block: Box::new(var("else")),
        };
        assert_eq!(child_vars(&if_expr), ["cond", "then", "else"]);

        let while_expr = Expr::While {
            condition: Box::new(var("cond")),
            body: Box::new(var("body")),
        };
        assert_eq!(child_vars(&while_expr), ["cond", "body"]);

        let receiver = Expr::ReceiverCall {
            receiver: Box::new(var("recv")),
            operation: "op".into(),
            args: vec![var("arg0"), var("arg1")],
        };
        assert_eq!(child_vars(&receiver), ["recv", "arg0", "arg1"]);

        let binary = Expr::Binary {
            left: Box::new(var("left")),
            op: BinaryOp::Add,
            right: Box::new(var("right")),
        };
        assert_eq!(child_vars(&binary), ["left", "right"]);
    }

    #[test]
    fn children_yields_assign_index_steps_before_value() {
        let assign = Expr::Assign {
            target: AssignTarget {
                root: "root".into(),
                steps: vec![
                    AssignPathStep::Field("field".into()),
                    AssignPathStep::Index(var("idx")),
                ],
            },
            expr: Box::new(var("value")),
        };
        // Field steps contribute no child expressions; the dynamic index is
        // yielded before the assigned value.
        assert_eq!(child_vars(&assign), ["idx", "value"]);
    }

    #[test]
    fn children_handles_optional_finish_and_submit() {
        assert!(Expr::Submit(None).children().next().is_none());
        assert!(Expr::Finish(None).children().next().is_none());
        assert_eq!(
            child_vars(&Expr::Submit(Some(Box::new(var("done"))))),
            ["done"]
        );
    }

    #[test]
    fn children_size_hint_is_exact() {
        let block = Expr::Block(vec![var("a"), var("b"), var("c"), var("d")]);
        let iter = block.children();
        assert_eq!(iter.len(), 4);
        assert_eq!(iter.size_hint(), (4, Some(4)));
    }
}
