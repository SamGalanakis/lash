use serde_json::{Value, json};

use crate::LinkedModule;
use crate::ast::{Declaration, Expr, Program, ResourceRefExpr, ScheduleCadence, TriggerSource};
use crate::lexer::Span;

pub fn static_graph_json(program: &Program, module_version: impl Into<String>) -> Value {
    static_graph_for_program(program, module_version.into())
}

pub fn linked_static_graph_json(linked: &LinkedModule) -> Value {
    static_graph_for_program(&linked.program, linked.module_version.clone())
}

fn static_graph_for_program(program: &Program, module_version: String) -> Value {
    let mut nodes = Vec::new();
    let mut edges = Vec::new();

    for (index, declaration) in program.declarations.iter().enumerate() {
        let span = program.declaration_spans.get(index).copied();
        match declaration {
            Declaration::Process(process) => {
                let process_id = format!("process:{}", process.name);
                nodes.push(node(&process_id, "process", process.name.as_str(), span));
                collect_expr_graph(&process.body, &process_id, span, &mut nodes, &mut edges);
            }
            Declaration::Trigger(trigger) => {
                let trigger_id = format!("trigger:{}", trigger.name);
                nodes.push(node(&trigger_id, "trigger", trigger.name.as_str(), span));
                match &trigger.source {
                    TriggerSource::Binding { resource, event } => {
                        let resource_id = resource_node_id(resource);
                        nodes.push(node(
                            &resource_id,
                            "resource",
                            format!("{}.{}", resource.resource_type, resource.alias),
                            span,
                        ));
                        edges.push(edge(&resource_id, &trigger_id, event.as_str(), span));
                    }
                    TriggerSource::Each {
                        resource_type,
                        event,
                        ..
                    } => {
                        let resource_id = format!("resource_type:{resource_type}");
                        nodes.push(node(
                            &resource_id,
                            "resource_type",
                            resource_type.as_str(),
                            span,
                        ));
                        edges.push(edge(&resource_id, &trigger_id, event.as_str(), span));
                    }
                }
                collect_expr_graph(&trigger.body, &trigger_id, span, &mut nodes, &mut edges);
            }
            Declaration::Schedule(schedule) => {
                let schedule_id = format!("schedule:{}", schedule.name);
                nodes.push(node(&schedule_id, "schedule", schedule.name.as_str(), span));
                match &schedule.cadence {
                    ScheduleCadence::Cron { .. } => {
                        nodes.push(node(
                            format!("{schedule_id}:cron"),
                            "schedule_cadence",
                            "cron",
                            span,
                        ));
                        edges.push(edge(
                            format!("{schedule_id}:cron"),
                            &schedule_id,
                            "activates",
                            span,
                        ));
                    }
                }
                collect_expr_graph(&schedule.body, &schedule_id, span, &mut nodes, &mut edges);
            }
            Declaration::Type(type_decl) => {
                nodes.push(node(
                    format!("type:{}", type_decl.name),
                    "type",
                    type_decl.name.as_str(),
                    span,
                ));
            }
        }
    }

    let main_span = program
        .expression_spans
        .first()
        .copied()
        .or_else(|| program.declaration_spans.last().copied());
    collect_expr_graph(&program.main, "main", main_span, &mut nodes, &mut edges);

    json!({
        "module_version": module_version,
        "nodes": nodes,
        "edges": edges,
    })
}

fn collect_expr_graph(
    expr: &Expr,
    owner: &str,
    span: Option<Span>,
    nodes: &mut Vec<Value>,
    edges: &mut Vec<Value>,
) {
    match expr {
        Expr::Block(expressions) => {
            for expression in expressions {
                collect_expr_graph(expression, owner, span, nodes, edges);
            }
        }
        Expr::StartProcess(start) => {
            let target = format!("process:{}", start.process);
            edges.push(edge(owner, &target, "starts", span));
        }
        Expr::SleepFor(duration) => {
            let sleep_id = format!("{owner}:sleep:{}", nodes.len());
            nodes.push(node(&sleep_id, "sleep", "sleep for", span));
            edges.push(edge(owner, &sleep_id, "sleeps", span));
            collect_expr_graph(duration, &sleep_id, span, nodes, edges);
        }
        Expr::SleepUntil(deadline) => {
            let sleep_id = format!("{owner}:sleep:{}", nodes.len());
            nodes.push(node(&sleep_id, "sleep", "sleep until", span));
            edges.push(edge(owner, &sleep_id, "sleeps", span));
            collect_expr_graph(deadline, &sleep_id, span, nodes, edges);
        }
        Expr::WaitSignal => {
            let wait_id = format!("{owner}:wait:{}", nodes.len());
            nodes.push(node(&wait_id, "wait", "wait signal", span));
            edges.push(edge(owner, &wait_id, "waits", span));
        }
        Expr::SignalRun { run, payload } => {
            let signal_id = format!("{owner}:signal:{}", nodes.len());
            nodes.push(node(&signal_id, "signal", "signal run", span));
            edges.push(edge(owner, &signal_id, "signals", span));
            collect_expr_graph(run, &signal_id, span, nodes, edges);
            collect_expr_graph(payload, &signal_id, span, nodes, edges);
        }
        Expr::ReceiverCall {
            receiver,
            operation,
            args,
        } => {
            let op_id = format!("{owner}:op:{operation}:{}", nodes.len());
            nodes.push(node(&op_id, "resource_operation", operation.as_str(), span));
            edges.push(edge(owner, &op_id, "calls", span));
            collect_expr_graph(receiver, owner, span, nodes, edges);
            for arg in args {
                collect_expr_graph(arg, owner, span, nodes, edges);
            }
        }
        Expr::ResourceRef(resource) => {
            let resource_id = resource_node_id(resource);
            nodes.push(node(
                &resource_id,
                "resource",
                format!("{}.{}", resource.resource_type, resource.alias),
                span,
            ));
            edges.push(edge(owner, resource_id, "uses", span));
        }
        Expr::If {
            condition,
            then_block,
            else_block,
        } => {
            let branch_id = format!("{owner}:branch:{}", nodes.len());
            nodes.push(node(&branch_id, "branch", "if", span));
            edges.push(edge(owner, &branch_id, "branches", span));
            collect_expr_graph(condition, &branch_id, span, nodes, edges);
            collect_expr_graph(then_block, &branch_id, span, nodes, edges);
            collect_expr_graph(else_block, &branch_id, span, nodes, edges);
        }
        Expr::Await(expr)
        | Expr::ResultUnwrap(expr)
        | Expr::Cancel(expr)
        | Expr::Print(expr)
        | Expr::Yield(expr)
        | Expr::Wake(expr)
        | Expr::Fail(expr)
        | Expr::Unary { expr, .. } => collect_expr_graph(expr, owner, span, nodes, edges),
        Expr::Finish(expr) | Expr::Submit(expr) => {
            let terminal_id = format!("{owner}:terminal:{}", nodes.len());
            nodes.push(node(&terminal_id, "terminal", "result", span));
            edges.push(edge(owner, terminal_id, "terminates", span));
            if let Some(expr) = expr {
                collect_expr_graph(expr, owner, span, nodes, edges);
            }
        }
        Expr::Assign { expr, target } => {
            for step in &target.steps {
                if let crate::ast::AssignPathStep::Index(index) = step {
                    collect_expr_graph(index, owner, span, nodes, edges);
                }
            }
            collect_expr_graph(expr, owner, span, nodes, edges);
        }
        Expr::List(items) => {
            for item in items {
                collect_expr_graph(item, owner, span, nodes, edges);
            }
        }
        Expr::Record(entries) => {
            for (_, value) in entries {
                collect_expr_graph(value, owner, span, nodes, edges);
            }
        }
        Expr::BuiltinCall { args, .. } => {
            for arg in args {
                collect_expr_graph(arg, owner, span, nodes, edges);
            }
        }
        Expr::Field { target, .. } => collect_expr_graph(target, owner, span, nodes, edges),
        Expr::Index { target, index } => {
            collect_expr_graph(target, owner, span, nodes, edges);
            collect_expr_graph(index, owner, span, nodes, edges);
        }
        Expr::Binary { left, right, .. } => {
            collect_expr_graph(left, owner, span, nodes, edges);
            collect_expr_graph(right, owner, span, nodes, edges);
        }
        Expr::For { iterable, body, .. } => {
            collect_expr_graph(iterable, owner, span, nodes, edges);
            collect_expr_graph(body, owner, span, nodes, edges);
        }
        Expr::Null
        | Expr::Bool(_)
        | Expr::Number(_)
        | Expr::String(_)
        | Expr::Variable(_)
        | Expr::Break
        | Expr::Continue
        | Expr::TypeLiteral(_) => {}
    }
}

fn node(
    id: impl Into<String>,
    kind: &'static str,
    label: impl Into<String>,
    span: Option<Span>,
) -> Value {
    json!({
        "id": id.into(),
        "kind": kind,
        "label": label.into(),
        "span": span_value(span),
    })
}

fn edge(
    from: impl Into<String>,
    to: impl Into<String>,
    label: impl Into<String>,
    span: Option<Span>,
) -> Value {
    json!({
        "from": from.into(),
        "to": to.into(),
        "label": label.into(),
        "span": span_value(span),
    })
}

fn span_value(span: Option<Span>) -> Value {
    let span = span.unwrap_or(Span { start: 0, end: 1 });
    let end = if span.end > span.start {
        span.end
    } else {
        span.start + 1
    };
    json!({ "start": span.start, "end": end })
}

fn resource_node_id(resource: &ResourceRefExpr) -> String {
    format!("resource:{}.{}", resource.resource_type, resource.alias)
}
