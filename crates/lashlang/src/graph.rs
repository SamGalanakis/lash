use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::ast::{AssignPathStep, Declaration, Expr, LabelMetadata, Program, ResourceRefExpr};
use crate::lexer::Span;
use crate::tracking::{
    LashlangAstPath, LashlangExecutionContext, LashlangExecutionSiteBuilder, ProcessBranchSelection,
};
use crate::{ModuleArtifact, ModuleRef, ProcessRef};

pub fn static_graph_json(program: &Program, module_ref: impl Into<String>) -> Value {
    static_graph_for_program(program, module_ref.into())
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

fn static_graph_for_program(program: &Program, module_ref: String) -> Value {
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
        "module_ref": module_ref,
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
        Expr::StartProcess(start) => {
            let target = format!("process:{}", start.process);
            edges.push(edge(owner, &target, "starts", span));
            for child in expr.children() {
                collect_expr_graph(child, owner, span, nodes, edges);
            }
        }
        Expr::SleepFor(_) => {
            let sleep_id = format!("{owner}:sleep:{}", nodes.len());
            nodes.push(node(&sleep_id, "sleep", "sleep for", span));
            edges.push(edge(owner, &sleep_id, "sleeps", span));
            for child in expr.children() {
                collect_expr_graph(child, &sleep_id, span, nodes, edges);
            }
        }
        Expr::SleepUntil(_) => {
            let sleep_id = format!("{owner}:sleep:{}", nodes.len());
            nodes.push(node(&sleep_id, "sleep", "sleep until", span));
            edges.push(edge(owner, &sleep_id, "sleeps", span));
            for child in expr.children() {
                collect_expr_graph(child, &sleep_id, span, nodes, edges);
            }
        }
        Expr::WaitSignal { name } => {
            let wait_id = format!("{owner}:wait:{}", nodes.len());
            nodes.push(node(&wait_id, "wait", format!("wait_signal {name}"), span));
            edges.push(edge(owner, &wait_id, "waits", span));
        }
        Expr::SignalRun { .. } => {
            let signal_id = format!("{owner}:signal:{}", nodes.len());
            nodes.push(node(&signal_id, "signal", "signal_run", span));
            edges.push(edge(owner, &signal_id, "signals", span));
            for child in expr.children() {
                collect_expr_graph(child, &signal_id, span, nodes, edges);
            }
        }
        Expr::ReceiverCall { operation, .. } => {
            let op_id = format!("{owner}:op:{operation}:{}", nodes.len());
            nodes.push(node(&op_id, "resource_operation", operation.as_str(), span));
            edges.push(edge(owner, &op_id, "calls", span));
            for child in expr.children() {
                collect_expr_graph(child, owner, span, nodes, edges);
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
            collect_expr_graph(condition, &branch_id, span, nodes, edges);
            collect_expr_graph(then_block, &branch_id, span, nodes, edges);
            collect_expr_graph(else_block, &branch_id, span, nodes, edges);
        }
        Expr::Finish(expr) | Expr::Submit(expr) => {
            let terminal_id = format!("{owner}:terminal:{}", nodes.len());
            nodes.push(node(&terminal_id, "terminal", "result", span));
            edges.push(edge(owner, terminal_id, "terminates", span));
            if let Some(expr) = expr {
                collect_expr_graph(expr, owner, span, nodes, edges);
            }
        }
        _ => {
            for child in expr.children() {
                collect_expr_graph(child, owner, span, nodes, edges);
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

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LashlangMapOptions {
    #[serde(default)]
    pub include_reachable_processes: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LashlangMap {
    pub module_ref: ModuleRef,
    pub entry_kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entry_ref: Option<ProcessRef>,
    pub entry_name: String,
    #[serde(default)]
    pub nodes: Vec<LashlangMapNode>,
    #[serde(default)]
    pub edges: Vec<LashlangMapEdge>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LashlangMapNode {
    pub id: String,
    pub kind: String,
    pub label: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label_metadata: Option<LabelMetadata>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LashlangMapEdge {
    pub id: String,
    pub from: String,
    pub to: String,
    pub label: String,
}

pub fn map_lashlang_process(
    artifact: &ModuleArtifact,
    process_ref: &ProcessRef,
    options: LashlangMapOptions,
) -> Option<LashlangMap> {
    let process_name = artifact.process_name_for_ref(process_ref)?;
    let mut builder = LashlangMapBuilder {
        artifact,
        options,
        nodes: Vec::new(),
        edges: Vec::new(),
        visited_processes: BTreeSet::new(),
    };
    builder.visit_process(process_name, LashlangAstPath::root());
    Some(LashlangMap {
        module_ref: artifact.module_ref.clone(),
        entry_kind: "process".to_string(),
        entry_ref: Some(process_ref.clone()),
        entry_name: process_name.to_string(),
        nodes: builder.nodes,
        edges: builder.edges,
    })
}

pub fn map_lashlang_main(artifact: &ModuleArtifact, options: LashlangMapOptions) -> LashlangMap {
    let mut builder = LashlangMapBuilder {
        artifact,
        options,
        nodes: Vec::new(),
        edges: Vec::new(),
        visited_processes: BTreeSet::new(),
    };
    builder.visit_main();
    LashlangMap {
        module_ref: artifact.module_ref.clone(),
        entry_kind: "main".to_string(),
        entry_ref: None,
        entry_name: "main".to_string(),
        nodes: builder.nodes,
        edges: builder.edges,
    }
}

struct LashlangMapBuilder<'artifact> {
    artifact: &'artifact ModuleArtifact,
    options: LashlangMapOptions,
    nodes: Vec<LashlangMapNode>,
    edges: Vec<LashlangMapEdge>,
    visited_processes: BTreeSet<String>,
}

impl LashlangMapBuilder<'_> {
    fn tracking_context(&self, process_name: &str) -> Option<LashlangExecutionContext> {
        let process_ref = self.artifact.process_ref(process_name)?.clone();
        Some(LashlangExecutionContext::process(
            self.artifact.module_ref.clone(),
            process_ref,
            process_name,
        ))
    }

    fn visit_main(&mut self) {
        let context = LashlangExecutionContext::main(self.artifact.module_ref.clone());
        let site_builder = context.builder();
        let main_id = site_builder.main_node_id();
        self.node(&main_id, "main", "main", None);
        self.visit_expr(
            &self.artifact.canonical_ir.main,
            &context,
            std::slice::from_ref(&main_id),
            LashlangAstPath::root(),
        );
    }

    fn visit_process(&mut self, process_name: &str, path: LashlangAstPath) {
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
        self.node(&process_id, "process", process_name, process.label.clone());
        self.visit_expr(
            &process.body,
            &context,
            std::slice::from_ref(&process_id),
            path,
        );
    }

    fn visit_expr(
        &mut self,
        expr: &Expr,
        context: &LashlangExecutionContext,
        owners: &[String],
        path: LashlangAstPath,
    ) -> Vec<String> {
        self.visit_expr_with_label_metadata(expr, context, owners, path, None)
    }

    fn visit_expr_with_label_metadata(
        &mut self,
        expr: &Expr,
        context: &LashlangExecutionContext,
        owners: &[String],
        path: LashlangAstPath,
        label_metadata: Option<&LabelMetadata>,
    ) -> Vec<String> {
        let site_builder = context.builder();
        match expr {
            Expr::LabelAnnotated { label, expr } => {
                if label_attaches_to_concrete_node(expr) {
                    self.visit_expr_with_label_metadata(expr, context, owners, path, Some(label))
                } else {
                    let site = site_builder.node_site(&path, "step", label.title.as_str());
                    self.node(
                        &site.node_id,
                        &site.node_kind,
                        &site.label,
                        Some(label.clone()),
                    );
                    self.edges_from_owners(&site_builder, &path, owners, &site.node_id, "steps");
                    for (index, child) in expr.children().enumerate() {
                        self.visit_expr(
                            child,
                            context,
                            std::slice::from_ref(&site.node_id),
                            path.child(index),
                        );
                    }
                    vec![site.node_id]
                }
            }
            Expr::Block(expressions) => {
                let mut next_owners = owners.to_vec();
                for (index, expression) in expressions.iter().enumerate() {
                    next_owners =
                        self.visit_expr(expression, context, &next_owners, path.child(index));
                }
                next_owners
            }
            Expr::Assign { target, expr } if label_metadata.is_some() => {
                let value_index = target
                    .steps
                    .iter()
                    .filter(|step| matches!(step, AssignPathStep::Index(_)))
                    .count();
                self.visit_expr_with_label_metadata(
                    expr,
                    context,
                    owners,
                    path.child(value_index),
                    label_metadata,
                )
            }
            Expr::Await(expr) | Expr::ResultUnwrap(expr) if label_metadata.is_some() => self
                .visit_expr_with_label_metadata(
                    expr,
                    context,
                    owners,
                    path.child(0),
                    label_metadata,
                ),
            Expr::StartProcess(start) => {
                let site = site_builder.node_site(
                    &path,
                    "child_process",
                    format!("start {}", start.process),
                );
                self.node(
                    &site.node_id,
                    &site.node_kind,
                    &site.label,
                    label_metadata.cloned(),
                );
                self.edges_from_owners(&site_builder, &path, owners, &site.node_id, "starts");
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
                    self.visit_process(start.process.as_str(), LashlangAstPath::root());
                }
                for (index, child) in expr.children().enumerate() {
                    self.visit_expr(
                        child,
                        context,
                        std::slice::from_ref(&site.node_id),
                        path.child(index),
                    );
                }
                vec![site.node_id]
            }
            Expr::ReceiverCall { operation, .. } => {
                let site = site_builder.node_site(&path, "resource_operation", operation.as_str());
                self.node(
                    &site.node_id,
                    &site.node_kind,
                    &site.label,
                    label_metadata.cloned(),
                );
                self.edges_from_owners(&site_builder, &path, owners, &site.node_id, "calls");
                for (index, child) in expr.children().enumerate() {
                    self.visit_expr(
                        child,
                        context,
                        std::slice::from_ref(&site.node_id),
                        path.child(index),
                    );
                }
                vec![site.node_id]
            }
            Expr::SleepFor(_) => {
                let site = site_builder.node_site(&path, "sleep", "sleep for");
                self.node(
                    &site.node_id,
                    &site.node_kind,
                    &site.label,
                    label_metadata.cloned(),
                );
                self.edges_from_owners(&site_builder, &path, owners, &site.node_id, "sleeps");
                for (index, child) in expr.children().enumerate() {
                    self.visit_expr(
                        child,
                        context,
                        std::slice::from_ref(&site.node_id),
                        path.child(index),
                    );
                }
                vec![site.node_id]
            }
            Expr::SleepUntil(_) => {
                let site = site_builder.node_site(&path, "sleep", "sleep until");
                self.node(
                    &site.node_id,
                    &site.node_kind,
                    &site.label,
                    label_metadata.cloned(),
                );
                self.edges_from_owners(&site_builder, &path, owners, &site.node_id, "sleeps");
                for (index, child) in expr.children().enumerate() {
                    self.visit_expr(
                        child,
                        context,
                        std::slice::from_ref(&site.node_id),
                        path.child(index),
                    );
                }
                vec![site.node_id]
            }
            Expr::WaitSignal { name } => {
                let site = site_builder.node_site(&path, "wait", format!("wait_signal {name}"));
                self.node(
                    &site.node_id,
                    &site.node_kind,
                    &site.label,
                    label_metadata.cloned(),
                );
                self.edges_from_owners(&site_builder, &path, owners, &site.node_id, "waits");
                vec![site.node_id]
            }
            Expr::SignalRun { .. } => {
                let site = site_builder.node_site(&path, "signal", "signal_run");
                self.node(
                    &site.node_id,
                    &site.node_kind,
                    &site.label,
                    label_metadata.cloned(),
                );
                self.edges_from_owners(&site_builder, &path, owners, &site.node_id, "signals");
                for (index, child) in expr.children().enumerate() {
                    self.visit_expr(
                        child,
                        context,
                        std::slice::from_ref(&site.node_id),
                        path.child(index),
                    );
                }
                vec![site.node_id]
            }
            Expr::Finish(value) | Expr::Submit(value) => {
                let site = site_builder.node_site(&path, "terminal", "result");
                self.node(
                    &site.node_id,
                    &site.node_kind,
                    &site.label,
                    label_metadata.cloned(),
                );
                self.edges_from_owners(&site_builder, &path, owners, &site.node_id, "terminates");
                if let Some(value) = value {
                    self.visit_expr(
                        value,
                        context,
                        std::slice::from_ref(&site.node_id),
                        path.child(0),
                    );
                }
                vec![site.node_id]
            }
            Expr::Fail(value) => {
                let site = site_builder.node_site(&path, "terminal", "failure");
                self.node(
                    &site.node_id,
                    &site.node_kind,
                    &site.label,
                    label_metadata.cloned(),
                );
                self.edges_from_owners(&site_builder, &path, owners, &site.node_id, "terminates");
                self.visit_expr(
                    value,
                    context,
                    std::slice::from_ref(&site.node_id),
                    path.child(0),
                );
                vec![site.node_id]
            }
            Expr::ResourceRef(resource) => {
                let resource_id = resource_node_id(resource);
                self.node(&resource_id, "resource", &resource.path_string(), None);
                self.edges_from_owners(&site_builder, &path, owners, &resource_id, "uses");
                owners.to_vec()
            }
            Expr::If {
                condition,
                then_block,
                else_block,
            } => {
                let site = site_builder.branch_site(&path);
                self.node(
                    &site.node_id,
                    &site.node_kind,
                    &site.label,
                    label_metadata.cloned(),
                );
                self.edges_from_owners(&site_builder, &path, owners, &site.node_id, "branches");
                self.visit_expr(
                    condition,
                    context,
                    std::slice::from_ref(&site.node_id),
                    path.child(0),
                );
                let then_path = path.child(1);
                let else_path = path.child(2);
                let then_id =
                    site_builder.branch_arm_node_id(&then_path, ProcessBranchSelection::Then);
                let else_id =
                    site_builder.branch_arm_node_id(&else_path, ProcessBranchSelection::Else);
                self.node(&then_id, "branch_arm", "then", None);
                self.node(&else_id, "branch_arm", "else", None);
                if let Some(branch) = &site.branch {
                    self.edge_with_id(branch.then_edge_id.clone(), &site.node_id, &then_id, "then");
                    self.edge_with_id(branch.else_edge_id.clone(), &site.node_id, &else_id, "else");
                }
                let then_continuations = self.visit_expr(
                    then_block,
                    context,
                    std::slice::from_ref(&then_id),
                    then_path,
                );
                let else_continuations = self.visit_expr(
                    else_block,
                    context,
                    std::slice::from_ref(&else_id),
                    else_path,
                );
                let mut continuations = Vec::new();
                extend_unique_owners(&mut continuations, then_continuations);
                extend_unique_owners(&mut continuations, else_continuations);
                continuations
            }
            Expr::Yield(_) => {
                let site = site_builder.node_site(&path, "process_event", "yield");
                self.node(
                    &site.node_id,
                    &site.node_kind,
                    &site.label,
                    label_metadata.cloned(),
                );
                self.edges_from_owners(&site_builder, &path, owners, &site.node_id, "emits");
                for (index, child) in expr.children().enumerate() {
                    self.visit_expr(
                        child,
                        context,
                        std::slice::from_ref(&site.node_id),
                        path.child(index),
                    );
                }
                vec![site.node_id]
            }
            Expr::Wake(_) => {
                let site = site_builder.node_site(&path, "process_event", "wake");
                self.node(
                    &site.node_id,
                    &site.node_kind,
                    &site.label,
                    label_metadata.cloned(),
                );
                self.edges_from_owners(&site_builder, &path, owners, &site.node_id, "emits");
                for (index, child) in expr.children().enumerate() {
                    self.visit_expr(
                        child,
                        context,
                        std::slice::from_ref(&site.node_id),
                        path.child(index),
                    );
                }
                vec![site.node_id]
            }
            _ => {
                let mut next_owners = owners.to_vec();
                for (index, child) in expr.children().enumerate() {
                    next_owners = self.visit_expr(child, context, &next_owners, path.child(index));
                }
                next_owners
            }
        }
    }

    fn node(&mut self, id: &str, kind: &str, label: &str, label_metadata: Option<LabelMetadata>) {
        if let Some(node) = self.nodes.iter_mut().find(|node| node.id == id) {
            if node.label_metadata.is_none() && label_metadata.is_some() {
                node.label_metadata = label_metadata;
            }
            return;
        }
        self.nodes.push(LashlangMapNode {
            id: id.to_string(),
            kind: kind.to_string(),
            label: label.to_string(),
            label_metadata,
        });
    }

    fn edge_with_id(&mut self, id: String, from: &str, to: &str, label: &str) {
        if self.edges.iter().any(|edge| edge.id == id) {
            return;
        }
        self.edges.push(LashlangMapEdge {
            id,
            from: from.to_string(),
            to: to.to_string(),
            label: label.to_string(),
        });
    }

    fn edges_from_owners(
        &mut self,
        site_builder: &LashlangExecutionSiteBuilder<'_>,
        path: &LashlangAstPath,
        owners: &[String],
        to: &str,
        label: &str,
    ) {
        for owner in owners {
            self.edge_with_id(
                site_builder.edge_id(path, owner, to, label),
                owner,
                to,
                label,
            );
        }
    }
}

fn extend_unique_owners(target: &mut Vec<String>, owners: impl IntoIterator<Item = String>) {
    for owner in owners {
        if !target.contains(&owner) {
            target.push(owner);
        }
    }
}

fn label_attaches_to_concrete_node(expr: &Expr) -> bool {
    match expr {
        Expr::LabelAnnotated { .. } => false,
        Expr::Assign { expr, .. } => label_attaches_to_assignment_value(expr),
        Expr::Await(expr) | Expr::ResultUnwrap(expr) => label_attaches_to_concrete_node(expr),
        Expr::ReceiverCall { .. }
        | Expr::StartProcess(_)
        | Expr::SleepFor(_)
        | Expr::SleepUntil(_)
        | Expr::WaitSignal { .. }
        | Expr::SignalRun { .. }
        | Expr::Submit(_)
        | Expr::Yield(_)
        | Expr::Wake(_)
        | Expr::Finish(_)
        | Expr::Fail(_)
        | Expr::If { .. } => true,
        Expr::Block(_)
        | Expr::Null
        | Expr::Bool(_)
        | Expr::Number(_)
        | Expr::String(_)
        | Expr::Variable(_)
        | Expr::List(_)
        | Expr::Record(_)
        | Expr::For { .. }
        | Expr::While { .. }
        | Expr::Break
        | Expr::Continue
        | Expr::ProcessRef { .. }
        | Expr::HostDescriptorConstructor { .. }
        | Expr::ResourceRef(_)
        | Expr::Cancel(_)
        | Expr::Print(_)
        | Expr::BuiltinCall { .. }
        | Expr::Field { .. }
        | Expr::Index { .. }
        | Expr::Unary { .. }
        | Expr::Binary { .. }
        | Expr::TypeLiteral(_) => false,
    }
}

fn label_attaches_to_assignment_value(expr: &Expr) -> bool {
    match expr {
        Expr::Await(expr) | Expr::ResultUnwrap(expr) => label_attaches_to_assignment_value(expr),
        Expr::ReceiverCall { .. }
        | Expr::StartProcess(_)
        | Expr::SleepFor(_)
        | Expr::SleepUntil(_)
        | Expr::WaitSignal { .. }
        | Expr::SignalRun { .. }
        | Expr::Submit(_)
        | Expr::Yield(_)
        | Expr::Wake(_)
        | Expr::Finish(_)
        | Expr::Fail(_)
        | Expr::If { .. } => true,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn linked(source: &str) -> crate::LinkedModule {
        let mut resources = crate::LashlangHostCatalog::new();
        resources.add_module_operation(
            ["tools"],
            "Tools",
            "read_file",
            "read_file",
            crate::TypeExpr::Any,
            crate::TypeExpr::Any,
        );
        crate::LinkedModule::link(
            crate::parse(source).expect("parse module"),
            crate::LashlangHostEnvironment::new(resources, crate::LashlangAbilities::all())
                .with_language_features(
                    crate::LashlangLanguageFeatures::default().with_label_annotations(),
                ),
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

        let map = map_lashlang_process(
            &linked.artifact,
            &process_ref,
            LashlangMapOptions {
                include_reachable_processes: true,
            },
        )
        .expect("map process");

        assert_eq!(map.module_ref, linked.module_ref);
        assert_eq!(map.entry_ref.as_ref(), Some(&process_ref));
        assert!(map.nodes.iter().any(|node| node.kind == "process"));
        assert!(
            map.nodes
                .iter()
                .any(|node| node.kind == "resource_operation")
        );
        assert!(map.edges.iter().any(|edge| edge.label == "starts"));
        assert!(map.edges.iter().all(|edge| !edge.id.is_empty()));

        let remapped = map_lashlang_process(
            &linked.artifact,
            &process_ref,
            LashlangMapOptions {
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
    fn process_map_includes_label_metadata_on_visual_nodes() {
        let linked = linked(
            r#"
            @label(title: "Scan files", description: "Process node")
            process scan(tool: Tools, flag: bool) {
              @label(title: "Read file", description: "Operation node")
              text = await tool.read_file({ path: "." })?
              @label(title: "Choose path")
              if flag {
                @label(title: "Wake agent")
                wake text
              } else {
                @label(title: "Finish scan")
                finish text
              }
            }
            "#,
        );
        let process_ref = linked
            .artifact
            .process_ref("scan")
            .expect("scan process ref")
            .clone();

        let map = map_lashlang_process(
            &linked.artifact,
            &process_ref,
            LashlangMapOptions::default(),
        )
        .expect("map process");

        assert_label_metadata(&map, "process", "Scan files", Some("Process node"));
        assert_label_metadata(
            &map,
            "resource_operation",
            "Read file",
            Some("Operation node"),
        );
        assert_label_metadata(&map, "branch", "Choose path", None);
        assert_label_metadata(&map, "process_event", "Wake agent", None);
        assert_label_metadata(&map, "terminal", "Finish scan", None);
    }

    #[test]
    fn main_map_includes_labeled_pure_setup_step_nodes() {
        let linked = linked(
            r#"
            @label(title: "Prepare", description: "Pure setup")
            value = 1
            @label(title: "Return")
            submit value
            "#,
        );

        let map = map_lashlang_main(&linked.artifact, LashlangMapOptions::default());

        assert_eq!(map.entry_kind, "main");
        assert_eq!(map.entry_ref, None);
        assert_eq!(map.entry_name, "main");
        assert_label_metadata(&map, "step", "Prepare", Some("Pure setup"));
        assert_label_metadata(&map, "terminal", "Return", None);
        let main_id = node_id(&map, "main", "main");
        let step_id = node_id_with_label_metadata(&map, "step", "Prepare");
        let terminal_id = node_id_with_label_metadata(&map, "terminal", "Return");
        assert_edge(&map, &main_id, &step_id, "steps");
        assert_edge(&map, &step_id, &terminal_id, "terminates");
    }

    #[test]
    fn process_map_chains_sequential_visual_statements() {
        let linked = linked(
            r#"
            process on_button(event: any) {
              @label(title: "Button Pressed")
              wake event
              @label(title: "Finish")
              finish true
            }
            "#,
        );
        let process_ref = linked
            .artifact
            .process_ref("on_button")
            .expect("on_button process ref")
            .clone();

        let map = map_lashlang_process(
            &linked.artifact,
            &process_ref,
            LashlangMapOptions::default(),
        )
        .expect("map process");
        let process_id = node_id(&map, "process", "on_button");
        let wake_id = node_id_with_label_metadata(&map, "process_event", "Button Pressed");
        let terminal_id = node_id_with_label_metadata(&map, "terminal", "Finish");

        assert_edge(&map, &process_id, &wake_id, "emits");
        assert_edge(&map, &wake_id, &terminal_id, "terminates");
        assert!(
            !map.edges
                .iter()
                .any(|edge| edge.from == process_id && edge.to == terminal_id),
            "terminal should follow wake instead of branching from process: {map:?}"
        );
    }

    fn assert_label_metadata(
        map: &LashlangMap,
        kind: &str,
        title: &str,
        description: Option<&str>,
    ) {
        let node = map
            .nodes
            .iter()
            .find(|node| {
                node.kind == kind
                    && node
                        .label_metadata
                        .as_ref()
                        .is_some_and(|label| label.title.as_str() == title)
            })
            .unwrap_or_else(|| panic!("missing `{title}` {kind} node in {map:?}"));
        assert_eq!(
            node.label_metadata
                .as_ref()
                .and_then(|label| label.description.as_deref()),
            description
        );
    }

    fn node_id(map: &LashlangMap, kind: &str, label: &str) -> String {
        map.nodes
            .iter()
            .find(|node| node.kind == kind && node.label == label)
            .unwrap_or_else(|| panic!("missing `{label}` {kind} node in {map:?}"))
            .id
            .clone()
    }

    fn node_id_with_label_metadata(map: &LashlangMap, kind: &str, title: &str) -> String {
        map.nodes
            .iter()
            .find(|node| {
                node.kind == kind
                    && node
                        .label_metadata
                        .as_ref()
                        .is_some_and(|label| label.title.as_str() == title)
            })
            .unwrap_or_else(|| panic!("missing `{title}` {kind} node in {map:?}"))
            .id
            .clone()
    }

    fn assert_edge(map: &LashlangMap, from: &str, to: &str, label: &str) {
        assert!(
            map.edges
                .iter()
                .any(|edge| edge.from == from && edge.to == to && edge.label == label),
            "missing `{label}` edge {from} -> {to} in {map:?}"
        );
    }

    fn assert_no_edge(map: &LashlangMap, from: &str, to: &str, label: &str) {
        assert!(
            !map.edges
                .iter()
                .any(|edge| edge.from == from && edge.to == to && edge.label == label),
            "unexpected `{label}` edge {from} -> {to} in {map:?}"
        );
    }

    #[test]
    fn process_map_joins_value_conditional_continuations_from_branch_arms() {
        let linked = linked(
            r#"
            process choose(tool: Tools, flag: bool) {
              topic = flag
                ? "red"
                : "blue"

              @label(title: "Generate Queries")
              value = await tool.read_file({ path: topic })?
              finish value
            }
            "#,
        );
        let process_ref = linked
            .artifact
            .process_ref("choose")
            .expect("choose process ref")
            .clone();

        let map = map_lashlang_process(
            &linked.artifact,
            &process_ref,
            LashlangMapOptions::default(),
        )
        .expect("map process");
        let branch_id = node_id(&map, "branch", "if");
        let then_id = node_id(&map, "branch_arm", "then");
        let else_id = node_id(&map, "branch_arm", "else");
        let operation_id =
            node_id_with_label_metadata(&map, "resource_operation", "Generate Queries");

        assert_edge(&map, &then_id, &operation_id, "calls");
        assert_edge(&map, &else_id, &operation_id, "calls");
        assert_no_edge(&map, &branch_id, &operation_id, "calls");
    }

    #[test]
    fn process_map_joins_block_conditional_continuations_from_branch_bodies() {
        let linked = linked(
            r#"
            process choose(flag: bool) {
              if flag {
                @label(title: "Then Wake")
                wake { path: "then" }
              } else {
                @label(title: "Else Wake")
                wake { path: "else" }
              }

              @label(title: "Finish")
              finish true
            }
            "#,
        );
        let process_ref = linked
            .artifact
            .process_ref("choose")
            .expect("choose process ref")
            .clone();

        let map = map_lashlang_process(
            &linked.artifact,
            &process_ref,
            LashlangMapOptions::default(),
        )
        .expect("map process");
        let branch_id = node_id(&map, "branch", "if");
        let then_wake_id = node_id_with_label_metadata(&map, "process_event", "Then Wake");
        let else_wake_id = node_id_with_label_metadata(&map, "process_event", "Else Wake");
        let terminal_id = node_id_with_label_metadata(&map, "terminal", "Finish");

        assert_edge(&map, &then_wake_id, &terminal_id, "terminates");
        assert_edge(&map, &else_wake_id, &terminal_id, "terminates");
        assert_no_edge(&map, &branch_id, &terminal_id, "terminates");
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

        let map = map_lashlang_process(
            &linked_module.artifact,
            &process_ref,
            LashlangMapOptions::default(),
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
        let reparsed_map = map_lashlang_process(
            &reparsed.artifact,
            &reparsed_ref,
            LashlangMapOptions::default(),
        )
        .expect("reparsed map");
        assert_eq!(map.nodes, reparsed_map.nodes);
        assert_eq!(map.edges, reparsed_map.edges);
    }
}
