use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::ast::{Declaration, Expr, Program, ResourceRefExpr};
use crate::lexer::Span;
use crate::tracking::{ProcessAstPath, ProcessBranchSelection, ProcessTrackingContext};
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
    pub id: String,
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
    builder.visit_process(process_name, ProcessAstPath::root());
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
    fn tracking_context(&self, process_name: &str) -> Option<ProcessTrackingContext> {
        let process_ref = self.artifact.process_ref(process_name)?.clone();
        Some(ProcessTrackingContext::new(
            self.artifact.module_ref.clone(),
            process_ref,
            process_name,
        ))
    }

    fn visit_process(&mut self, process_name: &str, path: ProcessAstPath) {
        if !self.visited_processes.insert(process_name.to_string()) {
            return;
        }
        let Some(context) = self.tracking_context(process_name) else {
            return;
        };
        let Some(process) = self.artifact.canonical_ir.process(process_name) else {
            return;
        };
        let site_builder = context.builder();
        let process_id = site_builder.process_node_id();
        self.node(&process_id, "process", process_name);
        self.visit_expr(&process.body, &context, &process_id, path);
    }

    fn visit_expr(
        &mut self,
        expr: &Expr,
        context: &ProcessTrackingContext,
        owner: &str,
        path: ProcessAstPath,
    ) {
        let site_builder = context.builder();
        match expr {
            Expr::Block(expressions) => {
                for (index, expression) in expressions.iter().enumerate() {
                    self.visit_expr(expression, context, owner, path.child(index));
                }
            }
            Expr::StartProcess(start) => {
                let site = site_builder.node_site(
                    &path,
                    "child_process",
                    format!("start {}", start.process),
                );
                self.node(&site.node_id, &site.node_kind, &site.label);
                self.edge_with_id(
                    site_builder.edge_id(&path, owner, &site.node_id, "starts"),
                    owner,
                    &site.node_id,
                    "starts",
                );
                let target = self
                    .tracking_context(start.process.as_str())
                    .map(|context| context.builder().process_node_id())
                    .unwrap_or_else(|| format!("process:{}", start.process));
                self.edge_with_id(
                    site_builder.edge_id(&path, &site.node_id, &target, "child"),
                    &site.node_id,
                    &target,
                    "child",
                );
                if self.options.include_reachable_processes {
                    self.visit_process(start.process.as_str(), ProcessAstPath::root());
                }
                for (index, child) in expr.children().enumerate() {
                    self.visit_expr(child, context, &site.node_id, path.child(index));
                }
            }
            Expr::ReceiverCall { operation, .. } => {
                let site = site_builder.node_site(&path, "resource_operation", operation.as_str());
                self.node(&site.node_id, &site.node_kind, &site.label);
                self.edge_with_id(
                    site_builder.edge_id(&path, owner, &site.node_id, "calls"),
                    owner,
                    &site.node_id,
                    "calls",
                );
                for (index, child) in expr.children().enumerate() {
                    self.visit_expr(child, context, &site.node_id, path.child(index));
                }
            }
            Expr::SleepFor(_) => {
                let site = site_builder.node_site(&path, "sleep", "sleep for");
                self.node(&site.node_id, &site.node_kind, &site.label);
                self.edge_with_id(
                    site_builder.edge_id(&path, owner, &site.node_id, "sleeps"),
                    owner,
                    &site.node_id,
                    "sleeps",
                );
                for (index, child) in expr.children().enumerate() {
                    self.visit_expr(child, context, &site.node_id, path.child(index));
                }
            }
            Expr::SleepUntil(_) => {
                let site = site_builder.node_site(&path, "sleep", "sleep until");
                self.node(&site.node_id, &site.node_kind, &site.label);
                self.edge_with_id(
                    site_builder.edge_id(&path, owner, &site.node_id, "sleeps"),
                    owner,
                    &site.node_id,
                    "sleeps",
                );
                for (index, child) in expr.children().enumerate() {
                    self.visit_expr(child, context, &site.node_id, path.child(index));
                }
            }
            Expr::WaitSignal => {
                let site = site_builder.node_site(&path, "wait", "wait signal");
                self.node(&site.node_id, &site.node_kind, &site.label);
                self.edge_with_id(
                    site_builder.edge_id(&path, owner, &site.node_id, "waits"),
                    owner,
                    &site.node_id,
                    "waits",
                );
            }
            Expr::SignalRun { .. } => {
                let site = site_builder.node_site(&path, "signal", "signal run");
                self.node(&site.node_id, &site.node_kind, &site.label);
                self.edge_with_id(
                    site_builder.edge_id(&path, owner, &site.node_id, "signals"),
                    owner,
                    &site.node_id,
                    "signals",
                );
                for (index, child) in expr.children().enumerate() {
                    self.visit_expr(child, context, &site.node_id, path.child(index));
                }
            }
            Expr::Finish(value) | Expr::Submit(value) => {
                let site = site_builder.node_site(&path, "terminal", "result");
                self.node(&site.node_id, &site.node_kind, &site.label);
                self.edge_with_id(
                    site_builder.edge_id(&path, owner, &site.node_id, "terminates"),
                    owner,
                    &site.node_id,
                    "terminates",
                );
                if let Some(value) = value {
                    self.visit_expr(value, context, &site.node_id, path.child(0));
                }
            }
            Expr::Fail(value) => {
                let site = site_builder.node_site(&path, "terminal", "failure");
                self.node(&site.node_id, &site.node_kind, &site.label);
                self.edge_with_id(
                    site_builder.edge_id(&path, owner, &site.node_id, "terminates"),
                    owner,
                    &site.node_id,
                    "terminates",
                );
                self.visit_expr(value, context, &site.node_id, path.child(0));
            }
            Expr::ResourceRef(resource) => {
                let resource_id = resource_node_id(resource);
                self.node(&resource_id, "resource", &resource.path_string());
                self.edge_with_id(
                    site_builder.edge_id(&path, owner, &resource_id, "uses"),
                    owner,
                    &resource_id,
                    "uses",
                );
            }
            Expr::If {
                condition,
                then_block,
                else_block,
            } => {
                let site = site_builder.branch_site(&path);
                self.node(&site.node_id, &site.node_kind, &site.label);
                self.edge_with_id(
                    site_builder.edge_id(&path, owner, &site.node_id, "branches"),
                    owner,
                    &site.node_id,
                    "branches",
                );
                self.visit_expr(condition, context, &site.node_id, path.child(0));
                let then_path = path.child(1);
                let else_path = path.child(2);
                let then_id =
                    site_builder.branch_arm_node_id(&then_path, ProcessBranchSelection::Then);
                let else_id =
                    site_builder.branch_arm_node_id(&else_path, ProcessBranchSelection::Else);
                self.node(&then_id, "branch_arm", "then");
                self.node(&else_id, "branch_arm", "else");
                if let Some(branch) = &site.branch {
                    self.edge_with_id(branch.then_edge_id.clone(), &site.node_id, &then_id, "then");
                    self.edge_with_id(branch.else_edge_id.clone(), &site.node_id, &else_id, "else");
                }
                self.visit_expr(then_block, context, &then_id, then_path);
                self.visit_expr(else_block, context, &else_id, else_path);
            }
            Expr::Yield(_) => {
                let site = site_builder.node_site(&path, "process_event", "yield");
                self.node(&site.node_id, &site.node_kind, &site.label);
                self.edge_with_id(
                    site_builder.edge_id(&path, owner, &site.node_id, "emits"),
                    owner,
                    &site.node_id,
                    "emits",
                );
                for (index, child) in expr.children().enumerate() {
                    self.visit_expr(child, context, &site.node_id, path.child(index));
                }
            }
            Expr::Wake(_) => {
                let site = site_builder.node_site(&path, "process_event", "wake");
                self.node(&site.node_id, &site.node_kind, &site.label);
                self.edge_with_id(
                    site_builder.edge_id(&path, owner, &site.node_id, "emits"),
                    owner,
                    &site.node_id,
                    "emits",
                );
                for (index, child) in expr.children().enumerate() {
                    self.visit_expr(child, context, &site.node_id, path.child(index));
                }
            }
            _ => {
                for (index, child) in expr.children().enumerate() {
                    self.visit_expr(child, context, owner, path.child(index));
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

    fn edge_with_id(&mut self, id: String, from: &str, to: &str, label: &str) {
        self.edges.push(ProcessMapEdge {
            id,
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
        resources.add_operation(
            "Tools",
            "read_file",
            "read_file",
            crate::TypeExpr::Any,
            crate::TypeExpr::Any,
        );
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
        assert!(map.edges.iter().all(|edge| !edge.id.is_empty()));

        let remapped = map_process(
            &linked.artifact,
            &process_ref,
            ProcessMapOptions {
                include_reachable_processes: true,
            },
        )
        .expect("remap process");
        assert_eq!(map, remapped);
        assert!(
            map.nodes
                .iter()
                .any(|node| node.id.starts_with("resource_operation:")),
            "resource operation node should use stable hashed identity: {map:?}"
        );
    }

    #[test]
    fn process_map_has_stable_branch_edges() {
        let linked_module = linked(
            r#"
            process choose(tool: Tools, flag: bool) {
              if flag {
                value = await tool.read_file({ path: "a" })?
                finish value
              } else {
                finish "none"
              }
            }
            "#,
        );
        let process_ref = linked_module
            .artifact
            .process_ref("choose")
            .expect("choose process ref")
            .clone();

        let map = map_process(
            &linked_module.artifact,
            &process_ref,
            ProcessMapOptions::default(),
        )
        .expect("map process");

        let branch_edges = map
            .edges
            .iter()
            .filter(|edge| matches!(edge.label.as_str(), "then" | "else"))
            .collect::<Vec<_>>();
        assert_eq!(branch_edges.len(), 2);
        assert!(branch_edges.iter().all(|edge| !edge.id.is_empty()));

        let reparsed = linked(
            r#"
            process choose(tool: Tools, flag: bool) {
              if flag {
                value = await tool.read_file({ path: "a" })?
                finish value
              } else {
                finish "none"
              }
            }
            "#,
        );
        let reparsed_ref = reparsed
            .artifact
            .process_ref("choose")
            .expect("choose process ref")
            .clone();
        let reparsed_map = map_process(
            &reparsed.artifact,
            &reparsed_ref,
            ProcessMapOptions::default(),
        )
        .expect("reparsed map");
        assert_eq!(map.nodes, reparsed_map.nodes);
        assert_eq!(map.edges, reparsed_map.edges);
    }
}
