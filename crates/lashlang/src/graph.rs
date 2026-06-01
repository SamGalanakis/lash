use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::ast::{
    Declaration, Expr, Program, ResourceRefExpr, ScheduleCadence, TriggerArg, TriggerSource,
};
use crate::lexer::Span;
use crate::{LinkedModule, ModuleArtifact, ModuleRef, ProcessRef};

pub fn static_graph_json(program: &Program, module_ref: impl Into<String>) -> Value {
    static_graph_for_program(program, module_ref.into(), &BTreeMap::new())
}

pub fn linked_static_graph_json(linked: &LinkedModule) -> Value {
    static_graph_for_program(
        linked.program(),
        linked.module_ref.to_string(),
        &linked.artifact.exports.processes,
    )
}

pub(crate) fn static_graph_json_for_module_ref(
    module_ref: ModuleRef,
    process_refs: &BTreeMap<String, ProcessRef>,
) -> Value {
    json!({
        "module_ref": module_ref,
        "processes": process_refs,
        "nodes": [],
        "edges": [],
    })
}

pub(crate) fn static_graph_json_without_ir(module_ref: impl Into<String>) -> Value {
    json!({
        "module_ref": module_ref.into(),
        "nodes": [],
        "edges": [],
    })
}

fn static_graph_for_program(
    program: &Program,
    module_ref: String,
    process_refs: &BTreeMap<String, ProcessRef>,
) -> Value {
    let mut nodes = Vec::new();
    let mut edges = Vec::new();

    for (index, declaration) in program.declarations.iter().enumerate() {
        let span = program.declaration_spans.get(index).copied();
        match declaration {
            Declaration::Process(process) => {
                let process_id = process_refs
                    .get(process.name.as_str())
                    .map(process_node_id)
                    .unwrap_or_else(|| format!("process:{}", process.name));
                nodes.push(node(&process_id, "process", process.name.as_str(), span));
                collect_expr_graph(
                    &process.body,
                    &process_id,
                    span,
                    process_refs,
                    &mut nodes,
                    &mut edges,
                );
            }
            Declaration::Trigger(trigger) => {
                let trigger_id = format!("trigger:{}", trigger.name);
                nodes.push(node(&trigger_id, "trigger", trigger.name.as_str(), span));
                match &trigger.source {
                    TriggerSource::Binding { resource, event } => {
                        let resource_id = resource_node_id(resource);
                        nodes.push(node(&resource_id, "resource", resource.path_string(), span));
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
                for (_, arg) in &trigger.args {
                    if let TriggerArg::ResourceRef(resource) = arg {
                        let resource_id = resource_node_id(resource);
                        nodes.push(node(&resource_id, "resource", resource.path_string(), span));
                        edges.push(edge(&trigger_id, resource_id, "binds", span));
                    }
                }
                let target = process_refs
                    .get(trigger.process_name.as_str())
                    .map(process_node_id)
                    .unwrap_or_else(|| format!("process:{}", trigger.process_name));
                edges.push(trigger_edge(
                    &trigger_id,
                    &target,
                    "starts",
                    span,
                    &trigger.args,
                ));
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
                collect_expr_graph(
                    &schedule.body,
                    &schedule_id,
                    span,
                    process_refs,
                    &mut nodes,
                    &mut edges,
                );
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
    collect_expr_graph(
        &program.main,
        "main",
        main_span,
        process_refs,
        &mut nodes,
        &mut edges,
    );

    json!({
        "module_ref": module_ref,
        "nodes": nodes,
        "edges": edges,
    })
}

fn collect_expr_graph(
    expr: &Expr,
    owner: &str,
    span: Option<Span>,
    process_refs: &BTreeMap<String, ProcessRef>,
    nodes: &mut Vec<Value>,
    edges: &mut Vec<Value>,
) {
    match expr {
        Expr::StartProcess(start) => {
            let target = process_refs
                .get(start.process.as_str())
                .map(process_node_id)
                .unwrap_or_else(|| format!("process:{}", start.process));
            edges.push(edge(owner, &target, "starts", span));
            for child in expr.children() {
                collect_expr_graph(child, owner, span, process_refs, nodes, edges);
            }
        }
        Expr::SleepFor(_) => {
            let sleep_id = format!("{owner}:sleep:{}", nodes.len());
            nodes.push(node(&sleep_id, "sleep", "sleep for", span));
            edges.push(edge(owner, &sleep_id, "sleeps", span));
            for child in expr.children() {
                collect_expr_graph(child, &sleep_id, span, process_refs, nodes, edges);
            }
        }
        Expr::SleepUntil(_) => {
            let sleep_id = format!("{owner}:sleep:{}", nodes.len());
            nodes.push(node(&sleep_id, "sleep", "sleep until", span));
            edges.push(edge(owner, &sleep_id, "sleeps", span));
            for child in expr.children() {
                collect_expr_graph(child, &sleep_id, span, process_refs, nodes, edges);
            }
        }
        Expr::WaitSignal => {
            let wait_id = format!("{owner}:wait:{}", nodes.len());
            nodes.push(node(&wait_id, "wait", "wait signal", span));
            edges.push(edge(owner, &wait_id, "waits", span));
        }
        Expr::SignalRun { .. } => {
            let signal_id = format!("{owner}:signal:{}", nodes.len());
            nodes.push(node(&signal_id, "signal", "signal run", span));
            edges.push(edge(owner, &signal_id, "signals", span));
            for child in expr.children() {
                collect_expr_graph(child, &signal_id, span, process_refs, nodes, edges);
            }
        }
        Expr::ReceiverCall { operation, .. } => {
            let op_id = format!("{owner}:op:{operation}:{}", nodes.len());
            nodes.push(node(&op_id, "resource_operation", operation.as_str(), span));
            edges.push(edge(owner, &op_id, "calls", span));
            for child in expr.children() {
                collect_expr_graph(child, owner, span, process_refs, nodes, edges);
            }
        }
        Expr::ResourceRef(resource) => {
            let resource_id = resource_node_id(resource);
            nodes.push(node(&resource_id, "resource", resource.path_string(), span));
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
            collect_expr_graph(condition, &branch_id, span, process_refs, nodes, edges);
            collect_expr_graph(then_block, &branch_id, span, process_refs, nodes, edges);
            collect_expr_graph(else_block, &branch_id, span, process_refs, nodes, edges);
        }
        Expr::Finish(expr) | Expr::Submit(expr) => {
            let terminal_id = format!("{owner}:terminal:{}", nodes.len());
            nodes.push(node(&terminal_id, "terminal", "result", span));
            edges.push(edge(owner, terminal_id, "terminates", span));
            if let Some(expr) = expr {
                collect_expr_graph(expr, owner, span, process_refs, nodes, edges);
            }
        }
        _ => {
            for child in expr.children() {
                collect_expr_graph(child, owner, span, process_refs, nodes, edges);
            }
        }
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

fn trigger_edge(
    from: impl Into<String>,
    to: impl Into<String>,
    label: impl Into<String>,
    span: Option<Span>,
    args: &[(crate::ast::AstString, TriggerArg)],
) -> Value {
    let mut value = edge(from, to, label, span);
    if let Some(object) = value.as_object_mut() {
        object.insert("bindings".to_string(), trigger_bindings_value(args));
    }
    value
}

fn trigger_bindings_value(args: &[(crate::ast::AstString, TriggerArg)]) -> Value {
    Value::Array(
        args.iter()
            .map(|(param, arg)| match arg {
                TriggerArg::EventBinding(name) => json!({
                    "param": param.as_str(),
                    "kind": "event",
                    "name": name.as_str(),
                }),
                TriggerArg::ResourceBinding(name) => json!({
                    "param": param.as_str(),
                    "kind": "resource_binding",
                    "name": name.as_str(),
                }),
                TriggerArg::ResourceRef(resource) => json!({
                    "param": param.as_str(),
                    "kind": "resource_ref",
                    "resource_type": resource.resource_type.as_str(),
                    "alias": resource.alias.as_str(),
                    "path": resource.path_string(),
                }),
            })
            .collect(),
    )
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
    format!("resource:{}", resource.path_string())
}

fn process_node_id(process_ref: &ProcessRef) -> String {
    format!("process:{}:{}", process_ref.component, process_ref.pos)
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProcessMapOptions {
    #[serde(default)]
    pub include_reachable_processes: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProcessMap {
    pub module_ref: ModuleRef,
    pub process_ref: ProcessRef,
    #[serde(default)]
    pub nodes: Vec<ProcessMapNode>,
    #[serde(default)]
    pub edges: Vec<ProcessMapEdge>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProcessMapNode {
    pub id: String,
    pub kind: String,
    pub label: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProcessMapEdge {
    pub from: String,
    pub to: String,
    pub label: String,
}

pub fn map_process(
    artifact: &ModuleArtifact,
    process_ref: &ProcessRef,
    options: ProcessMapOptions,
) -> Option<ProcessMap> {
    let process_name = artifact.process_name_for_ref(process_ref)?;
    let mut builder = ProcessMapBuilder {
        artifact,
        options,
        nodes: Vec::new(),
        edges: Vec::new(),
        visited_processes: BTreeSet::new(),
    };
    builder.visit_process(process_name);
    Some(ProcessMap {
        module_ref: artifact.module_ref.clone(),
        process_ref: process_ref.clone(),
        nodes: builder.nodes,
        edges: builder.edges,
    })
}

struct ProcessMapBuilder<'artifact> {
    artifact: &'artifact ModuleArtifact,
    options: ProcessMapOptions,
    nodes: Vec<ProcessMapNode>,
    edges: Vec<ProcessMapEdge>,
    visited_processes: BTreeSet<String>,
}

impl ProcessMapBuilder<'_> {
    fn visit_process(&mut self, process_name: &str) {
        if !self.visited_processes.insert(process_name.to_string()) {
            return;
        }
        let Some(process_ref) = self.artifact.process_ref(process_name) else {
            return;
        };
        let Some(process) = self.artifact.canonical_ir.process(process_name) else {
            return;
        };
        let process_id = process_node_id(process_ref);
        self.node(&process_id, "process", process_name);
        self.visit_expr(&process.body, &process_id);
    }

    fn visit_expr(&mut self, expr: &Expr, owner: &str) {
        match expr {
            Expr::Block(expressions) => {
                for expression in expressions {
                    self.visit_expr(expression, owner);
                }
            }
            Expr::StartProcess(start) => {
                let target = self
                    .artifact
                    .process_ref(start.process.as_str())
                    .map(process_node_id)
                    .unwrap_or_else(|| format!("process:{}", start.process));
                self.edge(owner, &target, "starts");
                if self.options.include_reachable_processes {
                    self.visit_process(start.process.as_str());
                }
                for child in expr.children() {
                    self.visit_expr(child, owner);
                }
            }
            Expr::ReceiverCall { operation, .. } => {
                let op_id = format!("{owner}:resource:{operation}:{}", self.nodes.len());
                self.node(&op_id, "resource_operation", operation.as_str());
                self.edge(owner, &op_id, "calls");
                for child in expr.children() {
                    self.visit_expr(child, &op_id);
                }
            }
            Expr::SleepFor(_) | Expr::SleepUntil(_) => {
                let sleep_id = format!("{owner}:sleep:{}", self.nodes.len());
                self.node(&sleep_id, "sleep", "sleep");
                self.edge(owner, &sleep_id, "sleeps");
                for child in expr.children() {
                    self.visit_expr(child, &sleep_id);
                }
            }
            Expr::WaitSignal => {
                let wait_id = format!("{owner}:wait:{}", self.nodes.len());
                self.node(&wait_id, "wait", "wait signal");
                self.edge(owner, &wait_id, "waits");
            }
            Expr::SignalRun { .. } => {
                let signal_id = format!("{owner}:signal:{}", self.nodes.len());
                self.node(&signal_id, "signal", "signal run");
                self.edge(owner, &signal_id, "signals");
                for child in expr.children() {
                    self.visit_expr(child, &signal_id);
                }
            }
            Expr::Finish(value) | Expr::Submit(value) => {
                let terminal_id = format!("{owner}:terminal:{}", self.nodes.len());
                self.node(&terminal_id, "terminal", "result");
                self.edge(owner, &terminal_id, "terminates");
                if let Some(value) = value {
                    self.visit_expr(value, &terminal_id);
                }
            }
            Expr::Fail(value) => {
                let terminal_id = format!("{owner}:terminal:{}", self.nodes.len());
                self.node(&terminal_id, "terminal", "failure");
                self.edge(owner, &terminal_id, "terminates");
                self.visit_expr(value, &terminal_id);
            }
            Expr::ResourceRef(resource) => {
                let resource_id = resource_node_id(resource);
                self.node(&resource_id, "resource", &resource.path_string());
                self.edge(owner, &resource_id, "uses");
            }
            Expr::If {
                condition,
                then_block,
                else_block,
            } => {
                let branch_id = format!("{owner}:branch:{}", self.nodes.len());
                self.node(&branch_id, "branch", "if");
                self.edge(owner, &branch_id, "branches");
                self.visit_expr(condition, &branch_id);
                self.visit_expr(then_block, &branch_id);
                self.visit_expr(else_block, &branch_id);
            }
            _ => {
                for child in expr.children() {
                    self.visit_expr(child, owner);
                }
            }
        }
    }

    fn node(&mut self, id: &str, kind: &str, label: &str) {
        if self.nodes.iter().any(|node| node.id == id) {
            return;
        }
        self.nodes.push(ProcessMapNode {
            id: id.to_string(),
            kind: kind.to_string(),
            label: label.to_string(),
        });
    }

    fn edge(&mut self, from: &str, to: &str, label: &str) {
        self.edges.push(ProcessMapEdge {
            from: from.to_string(),
            to: to.to_string(),
            label: label.to_string(),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn linked(source: &str) -> crate::LinkedModule {
        let mut resources = crate::ResourceCatalog::new();
        resources.add_module_instance(["tools"], "Tools");
        resources.add_operation("Tools", "read_file", "read_file");
        crate::LinkedModule::link(
            crate::parse(source).expect("parse module"),
            crate::LashlangSurface::new(resources, crate::LashlangAbilities::all()),
        )
        .expect("link module")
    }

    #[test]
    fn process_map_uses_stable_refs_and_handles_cycles() {
        let linked = linked(
            r#"
            process scan(tool: Tools) {
              start scan(tool: tool)
              text = await tool.read_file({ path: "." })?
              finish text
            }
            "#,
        );
        let process_ref = linked
            .artifact
            .process_ref("scan")
            .expect("scan process ref")
            .clone();

        let map = map_process(
            &linked.artifact,
            &process_ref,
            ProcessMapOptions {
                include_reachable_processes: true,
            },
        )
        .expect("map process");

        assert_eq!(map.module_ref, linked.module_ref);
        assert_eq!(map.process_ref, process_ref);
        assert!(map.nodes.iter().any(|node| node.kind == "process"));
        assert!(
            map.nodes
                .iter()
                .any(|node| node.kind == "resource_operation")
        );
        assert!(map.edges.iter().any(|edge| edge.label == "starts"));
    }
}
