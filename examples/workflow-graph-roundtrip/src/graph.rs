use std::collections::{BTreeMap, BTreeSet};

use lashlang::{
    Expr, WorkflowContainer, WorkflowDeclaration, WorkflowEdge, WorkflowEdgeKind, WorkflowGraph,
    WorkflowListComprehensionClause, WorkflowNode, WorkflowNodeId, WorkflowNodeKind,
    WorkflowNodeNameSource, WorkflowSubgraph, canonical_expression_source, parse_expression,
};
use serde_json::json;

use crate::{
    ChildGroup, EdgeData, EditableComprehensionClause, EditableValue, FlowEdge, FlowNode,
    GraphRoots, NodeData, RenderErrorResponse, WorkflowDocument,
};

pub(crate) fn document_from_graph(
    version: u64,
    source: String,
    graph: WorkflowGraph,
) -> WorkflowDocument {
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    let mut roots = GraphRoots {
        main: node_ids(&graph.main),
        ..GraphRoots::default()
    };
    flatten_subgraph(&graph.main, "main", None, &mut nodes, &mut edges);
    for declaration in &graph.declarations {
        let WorkflowDeclaration::Process(process) = declaration else {
            continue;
        };
        let process_id = process.id.to_string();
        let scope = format!("process:{process_id}");
        let children = node_ids(&process.body);
        roots.processes.push(process_id.clone());
        nodes.push(FlowNode {
            id: process_id.clone(),
            node_type: "process".to_string(),
            parent_id: None,
            data: NodeData {
                kind: "process".to_string(),
                title: process.display_name.clone(),
                description: process.description.clone(),
                name_source: name_source(process.name_source),
                operation: None,
                effect: None,
                fields: BTreeMap::new(),
                binding: None,
                target: None,
                expression: None,
                condition: None,
                iterable: None,
                clauses: Vec::new(),
                source: None,
                children: vec![ChildGroup {
                    slot: "body".to_string(),
                    scope: scope.clone(),
                    node_ids: children,
                }],
            },
        });
        flatten_subgraph(
            &process.body,
            &scope,
            Some(process_id),
            &mut nodes,
            &mut edges,
        );
    }
    WorkflowDocument {
        schema_version: graph.schema_version,
        version,
        source,
        nodes,
        edges,
        roots,
    }
}

pub(crate) fn graph_from_document(
    document: WorkflowDocument,
    baseline: &WorkflowGraph,
) -> Result<WorkflowGraph, RenderErrorResponse> {
    let mut seen = BTreeSet::new();
    for node in &document.nodes {
        if !seen.insert(node.id.as_str()) {
            return Err(RenderErrorResponse::document(
                format!("duplicate flow node id `{}`", node.id),
                json!({ "nodeId": node.id }),
            ));
        }
    }
    let nodes = document
        .nodes
        .iter()
        .map(|node| (node.id.as_str(), node))
        .collect::<BTreeMap<_, _>>();
    let edges = document.edges.iter().fold(
        BTreeMap::<&str, Vec<&FlowEdge>>::new(),
        |mut by_scope, edge| {
            by_scope
                .entry(edge.data.scope.as_str())
                .or_default()
                .push(edge);
            by_scope
        },
    );
    let baseline_nodes = baseline
        .nodes()
        .map(|node| (node.id.to_string(), node.clone()))
        .collect::<BTreeMap<_, _>>();
    let mut graph = baseline.clone();
    graph.schema_version = document.schema_version;
    graph.main = build_subgraph(
        "main",
        &document.roots.main,
        &nodes,
        &baseline_nodes,
        &edges,
    )?;
    for declaration in &mut graph.declarations {
        let WorkflowDeclaration::Process(process) = declaration else {
            continue;
        };
        let process_id = process.id.to_string();
        let flow_process = nodes.get(process_id.as_str()).copied().ok_or_else(|| {
            RenderErrorResponse::document(
                format!("missing process container `{process_id}`"),
                json!({ "processId": process_id }),
            )
        })?;
        process.display_name = flow_process.data.title.clone();
        process.description = flow_process.data.description.clone();
        process.name_source = parse_name_source(&flow_process.data.name_source);
        if !document
            .roots
            .processes
            .iter()
            .any(|root| root == &process_id)
        {
            return Err(RenderErrorResponse::document(
                format!("missing process root `{process_id}`"),
                json!({ "processId": process_id }),
            ));
        }
        let body = flow_process
            .data
            .children
            .iter()
            .find(|child| child.slot == "body")
            .ok_or_else(|| {
                RenderErrorResponse::document(
                    format!("process container `{process_id}` is missing its body"),
                    json!({ "processId": process_id, "child": "body" }),
                )
            })?;
        process.body =
            build_subgraph(&body.scope, &body.node_ids, &nodes, &baseline_nodes, &edges)?;
    }
    Ok(graph)
}

fn flatten_subgraph(
    graph: &WorkflowSubgraph,
    scope: &str,
    parent_id: Option<String>,
    nodes: &mut Vec<FlowNode>,
    edges: &mut Vec<FlowEdge>,
) {
    for edge in &graph.edges {
        edges.push(flow_edge(edge, scope));
    }
    for node in &graph.nodes {
        let id = node.id.to_string();
        let child_groups = child_groups(node);
        nodes.push(FlowNode {
            id: id.clone(),
            node_type: node_kind(node).to_string(),
            parent_id: parent_id.clone(),
            data: node_data(node, child_groups.clone()),
        });
        for child in child_groups {
            if let Some(subgraph) = child_subgraph(node, &child.slot) {
                flatten_subgraph(subgraph, &child.scope, Some(id.clone()), nodes, edges);
            }
        }
    }
}

fn node_data(node: &WorkflowNode, children: Vec<ChildGroup>) -> NodeData {
    let (operation, effect) = match &node.kind {
        WorkflowNodeKind::Call { operation, .. } => (Some(operation.clone()), None),
        WorkflowNodeKind::Effect { effect, .. } => (None, Some(effect_name(effect).to_string())),
        _ => (None, None),
    };
    let source = match &node.kind {
        WorkflowNodeKind::Opaque { source } => Some(source.clone()),
        _ => None,
    };
    let binding = match &node.kind {
        WorkflowNodeKind::Data { binding, .. }
        | WorkflowNodeKind::Call { binding, .. }
        | WorkflowNodeKind::Effect { binding, .. }
        | WorkflowNodeKind::Computation { binding, .. }
        | WorkflowNodeKind::Container(WorkflowContainer::If { binding, .. })
        | WorkflowNodeKind::Container(WorkflowContainer::ListComprehension { binding, .. }) => {
            binding.clone()
        }
        WorkflowNodeKind::Container(WorkflowContainer::For { binding, .. }) => {
            Some(binding.clone())
        }
        _ => None,
    };
    let target = match &node.kind {
        WorkflowNodeKind::StateUpdate { target, .. } => Some(target.clone()),
        _ => None,
    };
    let expression = match &node.kind {
        WorkflowNodeKind::Computation { expression, .. }
        | WorkflowNodeKind::StateUpdate { expression, .. } => Some(expression.clone()),
        _ => None,
    };
    let condition = match &node.kind {
        WorkflowNodeKind::Container(WorkflowContainer::If { condition, .. })
        | WorkflowNodeKind::Container(WorkflowContainer::While { condition, .. }) => {
            Some(condition.clone())
        }
        _ => None,
    };
    let iterable = match &node.kind {
        WorkflowNodeKind::Container(WorkflowContainer::For { iterable, .. }) => {
            Some(iterable.clone())
        }
        _ => None,
    };
    let clauses = match &node.kind {
        WorkflowNodeKind::Container(WorkflowContainer::ListComprehension { clauses, .. }) => {
            clauses.iter().map(editable_clause).collect()
        }
        _ => Vec::new(),
    };
    NodeData {
        kind: node_kind(node).to_string(),
        title: node.name.clone(),
        description: node.description.clone(),
        name_source: name_source(node.name_source),
        operation,
        effect,
        fields: editable_fields(node),
        binding,
        target,
        expression,
        condition,
        iterable,
        clauses,
        source,
        children,
    }
}

fn editable_clause(clause: &WorkflowListComprehensionClause) -> EditableComprehensionClause {
    match clause {
        WorkflowListComprehensionClause::For { binding, iterable } => {
            EditableComprehensionClause::For {
                binding: binding.clone(),
                iterable: iterable.clone(),
            }
        }
        WorkflowListComprehensionClause::If { condition } => EditableComprehensionClause::If {
            condition: condition.clone(),
        },
    }
}

fn flow_edge(edge: &WorkflowEdge, scope: &str) -> FlowEdge {
    let (kind, variable, version) = match &edge.kind {
        WorkflowEdgeKind::Sequence => ("sequence".to_string(), None, None),
        WorkflowEdgeKind::DataDependency { variable, version } => {
            ("data".to_string(), Some(variable.clone()), Some(*version))
        }
    };
    FlowEdge {
        id: edge.id.clone(),
        source: edge.from.to_string(),
        target: edge.to.to_string(),
        data: EdgeData {
            kind,
            scope: scope.to_string(),
            variable,
            version,
        },
    }
}

fn child_groups(node: &WorkflowNode) -> Vec<ChildGroup> {
    let group = |slot: &str, graph: &WorkflowSubgraph| ChildGroup {
        slot: slot.to_string(),
        scope: format!("container:{}:{slot}", node.id),
        node_ids: node_ids(graph),
    };
    match &node.kind {
        WorkflowNodeKind::Container(WorkflowContainer::If {
            then_graph,
            else_graph,
            ..
        }) => [
            then_graph.as_deref().map(|graph| group("then", graph)),
            else_graph.as_deref().map(|graph| group("else", graph)),
        ]
        .into_iter()
        .flatten()
        .collect(),
        WorkflowNodeKind::Container(WorkflowContainer::For { body, .. }) => body
            .as_deref()
            .map(|graph| vec![group("body", graph)])
            .unwrap_or_default(),
        WorkflowNodeKind::Container(WorkflowContainer::While { body, .. }) => body
            .as_deref()
            .map(|graph| vec![group("body", graph)])
            .unwrap_or_default(),
        WorkflowNodeKind::Container(WorkflowContainer::ListComprehension { element, .. }) => {
            element
                .as_deref()
                .map(|graph| vec![group("element", graph)])
                .unwrap_or_default()
        }
        _ => Vec::new(),
    }
}

fn child_subgraph<'a>(node: &'a WorkflowNode, slot: &str) -> Option<&'a WorkflowSubgraph> {
    match (&node.kind, slot) {
        (WorkflowNodeKind::Container(WorkflowContainer::If { then_graph, .. }), "then") => {
            then_graph.as_deref()
        }
        (WorkflowNodeKind::Container(WorkflowContainer::If { else_graph, .. }), "else") => {
            else_graph.as_deref()
        }
        (WorkflowNodeKind::Container(WorkflowContainer::For { body, .. }), "body") => {
            body.as_deref()
        }
        (WorkflowNodeKind::Container(WorkflowContainer::While { body, .. }), "body") => {
            body.as_deref()
        }
        (
            WorkflowNodeKind::Container(WorkflowContainer::ListComprehension { element, .. }),
            "element",
        ) => element.as_deref(),
        _ => None,
    }
}

fn build_subgraph(
    scope: &str,
    node_ids: &[String],
    flow_nodes: &BTreeMap<&str, &FlowNode>,
    baseline_nodes: &BTreeMap<String, WorkflowNode>,
    flow_edges: &BTreeMap<&str, Vec<&FlowEdge>>,
) -> Result<WorkflowSubgraph, RenderErrorResponse> {
    let mut nodes = Vec::with_capacity(node_ids.len());
    for id in node_ids {
        let flow = flow_nodes.get(id.as_str()).copied().ok_or_else(|| {
            RenderErrorResponse::document(
                format!("scope `{scope}` references missing flow node `{id}`"),
                json!({ "scope": scope, "nodeId": id }),
            )
        })?;
        let mut node = baseline_nodes.get(id).cloned().ok_or_else(|| {
            RenderErrorResponse::document(
                format!("flow node `{id}` does not exist in workflow version"),
                json!({ "nodeId": id }),
            )
        })?;
        apply_editable_data(&mut node, &flow.data)?;
        rebuild_children(
            &mut node,
            &flow.data.children,
            flow_nodes,
            baseline_nodes,
            flow_edges,
        )?;
        nodes.push(node);
    }
    let edges = flow_edges
        .get(scope)
        .into_iter()
        .flatten()
        .map(|edge| workflow_edge(edge))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(WorkflowSubgraph { nodes, edges })
}

fn workflow_edge(edge: &FlowEdge) -> Result<WorkflowEdge, RenderErrorResponse> {
    let parse_id = |id: &str| {
        serde_json::from_value::<WorkflowNodeId>(serde_json::Value::String(id.to_string())).map_err(
            |error| {
                RenderErrorResponse::document(
                    format!("invalid workflow node id `{id}`: {error}"),
                    json!({ "nodeId": id }),
                )
            },
        )
    };
    let kind = match edge.data.kind.as_str() {
        "sequence" => WorkflowEdgeKind::Sequence,
        "data" => WorkflowEdgeKind::DataDependency {
            variable: edge.data.variable.clone().ok_or_else(|| {
                RenderErrorResponse::document(
                    format!("data edge `{}` is missing `data.variable`", edge.id),
                    json!({ "edgeId": edge.id }),
                )
            })?,
            version: edge.data.version.ok_or_else(|| {
                RenderErrorResponse::document(
                    format!("data edge `{}` is missing `data.version`", edge.id),
                    json!({ "edgeId": edge.id }),
                )
            })?,
        },
        kind => {
            return Err(RenderErrorResponse::document(
                format!("edge `{}` has unknown kind `{kind}`", edge.id),
                json!({ "edgeId": edge.id, "kind": kind }),
            ));
        }
    };
    Ok(WorkflowEdge {
        id: edge.id.clone(),
        from: parse_id(&edge.source)?,
        to: parse_id(&edge.target)?,
        kind,
    })
}

fn rebuild_children(
    node: &mut WorkflowNode,
    children: &[ChildGroup],
    flow_nodes: &BTreeMap<&str, &FlowNode>,
    baseline_nodes: &BTreeMap<String, WorkflowNode>,
    flow_edges: &BTreeMap<&str, Vec<&FlowEdge>>,
) -> Result<(), RenderErrorResponse> {
    let build = |slot: &str| -> Result<Option<Box<WorkflowSubgraph>>, RenderErrorResponse> {
        children
            .iter()
            .find(|child| child.slot == slot)
            .map(|child| {
                build_subgraph(
                    &child.scope,
                    &child.node_ids,
                    flow_nodes,
                    baseline_nodes,
                    flow_edges,
                )
                .map(Box::new)
            })
            .transpose()
    };
    match &mut node.kind {
        WorkflowNodeKind::Container(WorkflowContainer::If {
            then_graph,
            else_graph,
            ..
        }) => {
            *then_graph = build("then")?;
            *else_graph = build("else")?;
        }
        WorkflowNodeKind::Container(WorkflowContainer::For { body, .. }) => {
            *body = build("body")?;
        }
        WorkflowNodeKind::Container(WorkflowContainer::While { body, .. }) => {
            *body = build("body")?;
        }
        WorkflowNodeKind::Container(WorkflowContainer::ListComprehension { element, .. }) => {
            *element = build("element")?;
        }
        _ => {}
    }
    Ok(())
}

fn apply_editable_data(
    node: &mut WorkflowNode,
    data: &NodeData,
) -> Result<(), RenderErrorResponse> {
    let node_id = node.id.to_string();
    node.name = data.title.clone();
    node.description = data.description.clone();
    node.name_source = parse_name_source(&data.name_source);
    if let WorkflowNodeKind::Opaque { source } = &mut node.kind
        && let Some(updated) = &data.source
    {
        *source = updated.clone();
    }
    match &mut node.kind {
        WorkflowNodeKind::Data { binding, .. }
        | WorkflowNodeKind::Call { binding, .. }
        | WorkflowNodeKind::Effect { binding, .. } => {
            *binding = data.binding.clone();
        }
        WorkflowNodeKind::Computation {
            binding,
            expression,
        } => {
            *binding = data.binding.clone();
            *expression = required_text(&node_id, data.expression.as_ref(), "expression")?;
        }
        WorkflowNodeKind::StateUpdate { target, expression } => {
            *target = required_text(&node_id, data.target.as_ref(), "target")?;
            *expression = required_text(&node_id, data.expression.as_ref(), "expression")?;
        }
        WorkflowNodeKind::Container(WorkflowContainer::If {
            binding, condition, ..
        }) => {
            *binding = data.binding.clone();
            *condition = required_text(&node_id, data.condition.as_ref(), "condition")?;
        }
        WorkflowNodeKind::Container(WorkflowContainer::For {
            binding, iterable, ..
        }) => {
            *binding = required_text(&node_id, data.binding.as_ref(), "binding")?;
            *iterable = required_text(&node_id, data.iterable.as_ref(), "iterable")?;
        }
        WorkflowNodeKind::Container(WorkflowContainer::While { condition, .. }) => {
            *condition = required_text(&node_id, data.condition.as_ref(), "condition")?;
        }
        WorkflowNodeKind::Container(WorkflowContainer::ListComprehension {
            binding,
            clauses,
            ..
        }) => {
            *binding = data.binding.clone();
            *clauses = data.clauses.iter().map(workflow_clause).collect();
        }
        _ => {}
    }
    let expression_text = match &mut node.kind {
        WorkflowNodeKind::Data { expression, .. }
        | WorkflowNodeKind::Call { expression, .. }
        | WorkflowNodeKind::Effect { expression, .. }
        | WorkflowNodeKind::Terminal { expression, .. } => Some(expression),
        _ => None,
    };
    if let Some(expression_text) = expression_text {
        let mut expression = parse_expression(expression_text).map_err(|error| {
            RenderErrorResponse::document(
                format!(
                    "stored expression for node `{}` did not parse: {error}",
                    node.id
                ),
                json!({ "nodeId": node.id.to_string(), "reason": error.to_string() }),
            )
        })?;
        apply_fields(&mut expression, &data.fields);
        *expression_text = canonical_expression_source(&expression).map_err(|error| {
            RenderErrorResponse::document(
                format!(
                    "edited fields for node `{}` are not sourceable: {error}",
                    node.id
                ),
                json!({ "nodeId": node.id.to_string(), "reason": error.to_string() }),
            )
        })?;
    }
    Ok(())
}

fn required_text(
    node_id: &str,
    value: Option<&String>,
    field: &'static str,
) -> Result<String, RenderErrorResponse> {
    value.cloned().ok_or_else(|| {
        RenderErrorResponse::document(
            format!("flow node `{node_id}` is missing `data.{field}`"),
            json!({ "nodeId": node_id, "field": field }),
        )
    })
}

fn workflow_clause(clause: &EditableComprehensionClause) -> WorkflowListComprehensionClause {
    match clause {
        EditableComprehensionClause::For { binding, iterable } => {
            WorkflowListComprehensionClause::For {
                binding: binding.clone(),
                iterable: iterable.clone(),
            }
        }
        EditableComprehensionClause::If { condition } => WorkflowListComprehensionClause::If {
            condition: condition.clone(),
        },
    }
}

fn editable_fields(node: &WorkflowNode) -> BTreeMap<String, EditableValue> {
    let expression = match &node.kind {
        WorkflowNodeKind::Data { expression, .. }
        | WorkflowNodeKind::Call { expression, .. }
        | WorkflowNodeKind::Effect { expression, .. }
        | WorkflowNodeKind::Terminal { expression, .. } => expression,
        _ => return BTreeMap::new(),
    };
    let Ok(expression) = parse_expression(expression) else {
        return BTreeMap::new();
    };
    if let Some(fields) = receiver_fields(&expression) {
        return fields
            .iter()
            .filter_map(|(name, value)| {
                EditableValue::from_expr(value).map(|value| (name.to_string(), value))
            })
            .collect();
    }
    match &expression {
        Expr::SleepFor(value) | Expr::SleepUntil(value) => EditableValue::from_expr(value)
            .map(|value| BTreeMap::from([("duration".to_string(), value)]))
            .unwrap_or_default(),
        Expr::WaitSignal { name } => BTreeMap::from([(
            "signal".to_string(),
            EditableValue::String(name.to_string()),
        )]),
        _ => BTreeMap::new(),
    }
}

fn apply_fields(expression: &mut Expr, fields: &BTreeMap<String, EditableValue>) {
    if let Some(entries) = receiver_fields_mut(expression) {
        for (name, value) in fields {
            if let Some((_, expression)) = entries
                .iter_mut()
                .find(|(existing, _)| existing.as_str() == name)
            {
                *expression = value.to_expr();
            } else {
                entries.push((name.clone().into(), value.to_expr()));
            }
        }
        return;
    }
    match expression {
        Expr::SleepFor(value) | Expr::SleepUntil(value) => {
            if let Some(duration) = fields.get("duration") {
                **value = duration.to_expr();
            }
        }
        Expr::WaitSignal { name } => {
            if let Some(EditableValue::String(signal)) = fields.get("signal") {
                *name = signal.clone().into();
            }
        }
        _ => {}
    }
}

fn receiver_fields(expression: &Expr) -> Option<&Vec<(compact_str::CompactString, Expr)>> {
    match expression {
        Expr::ReceiverCall { args, .. } => args.first().and_then(|arg| match arg {
            Expr::Record(fields) => Some(fields),
            _ => None,
        }),
        Expr::Await(inner) | Expr::ResultUnwrap(inner) => receiver_fields(inner),
        _ => None,
    }
}

fn receiver_fields_mut(
    expression: &mut Expr,
) -> Option<&mut Vec<(compact_str::CompactString, Expr)>> {
    match expression {
        Expr::ReceiverCall { args, .. } => args.first_mut().and_then(|arg| match arg {
            Expr::Record(fields) => Some(fields),
            _ => None,
        }),
        Expr::Await(inner) | Expr::ResultUnwrap(inner) => receiver_fields_mut(inner),
        _ => None,
    }
}

impl EditableValue {
    fn from_expr(expression: &Expr) -> Option<Self> {
        match expression {
            Expr::Null => Some(Self::Null),
            Expr::Bool(value) => Some(Self::Bool(*value)),
            Expr::Number(value) => Some(Self::Number(*value)),
            Expr::String(value) => Some(Self::String(value.to_string())),
            Expr::List(values) => values
                .iter()
                .map(Self::from_expr)
                .collect::<Option<Vec<_>>>()
                .map(Self::List),
            Expr::Record(entries) => entries
                .iter()
                .map(|(key, value)| Some((key.to_string(), Self::from_expr(value)?)))
                .collect::<Option<BTreeMap<_, _>>>()
                .map(Self::Object),
            _ => None,
        }
    }

    fn to_expr(&self) -> Expr {
        match self {
            Self::Null => Expr::Null,
            Self::Bool(value) => Expr::Bool(*value),
            Self::Number(value) => Expr::Number(*value),
            Self::String(value) => Expr::String(value.clone().into()),
            Self::List(values) => Expr::List(values.iter().map(Self::to_expr).collect()),
            Self::Object(entries) => Expr::Record(
                entries
                    .iter()
                    .map(|(key, value)| (key.clone().into(), value.to_expr()))
                    .collect(),
            ),
        }
    }
}

fn node_ids(graph: &WorkflowSubgraph) -> Vec<String> {
    graph.nodes.iter().map(|node| node.id.to_string()).collect()
}

fn node_kind(node: &WorkflowNode) -> &'static str {
    match node.kind {
        WorkflowNodeKind::Data { .. } => "data",
        WorkflowNodeKind::Call { .. } => "call",
        WorkflowNodeKind::Effect { .. } => "effect",
        WorkflowNodeKind::Computation { .. } => "computation",
        WorkflowNodeKind::StateUpdate { .. } => "state_update",
        WorkflowNodeKind::Terminal { .. } => "terminal",
        WorkflowNodeKind::Container(_) => "container",
        WorkflowNodeKind::Opaque { .. } => "opaque",
    }
}

fn effect_name(effect: &lashlang::WorkflowEffectKind) -> &'static str {
    use lashlang::WorkflowEffectKind;
    match effect {
        WorkflowEffectKind::StartProcess => "start_process",
        WorkflowEffectKind::AwaitJoin => "await_join",
        WorkflowEffectKind::SignalRun => "signal_run",
        WorkflowEffectKind::WaitSignal => "wait_signal",
        WorkflowEffectKind::Sleep => "sleep",
        WorkflowEffectKind::Cancel => "cancel",
        WorkflowEffectKind::Print => "print",
        WorkflowEffectKind::Yield => "yield",
        WorkflowEffectKind::Wake => "wake",
        WorkflowEffectKind::Break => "break",
        WorkflowEffectKind::Continue => "continue",
    }
}

fn name_source(source: WorkflowNodeNameSource) -> String {
    match source {
        WorkflowNodeNameSource::Label => "label",
        WorkflowNodeNameSource::Derived => "derived",
    }
    .to_string()
}

fn parse_name_source(source: &str) -> WorkflowNodeNameSource {
    if source == "label" {
        WorkflowNodeNameSource::Label
    } else {
        WorkflowNodeNameSource::Derived
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn promoted_constructs_flatten_and_rebuild_as_typed_nodes() {
        let input = r#"type State = { count: int }

process worker() {
  finish 1
}

schema = Type { state: State }
runs = [start worker(), start worker()]
state = { count: 0 }
state.count = 1
while state.count < 2 {
  state.count = state.count + 1
}
for item in [1, 2] {
  state.count = item
  introduced = item
}
finish [state, introduced]
"#;
        let graph = lashlang::workflow_graph_from_source(input).expect("project promoted graph");
        let source = lashlang::workflow_graph_to_source(&graph).expect("render promoted graph");
        let document = document_from_graph(1, source.clone(), graph.clone());

        for kind in [
            "data",
            "computation",
            "state_update",
            "container",
            "terminal",
        ] {
            assert!(
                document.nodes.iter().any(|node| node.node_type == kind),
                "missing flattened {kind} node"
            );
        }
        assert!(!document.nodes.iter().any(|node| node.node_type == "opaque"));
        assert!(document.nodes.iter().any(|node| {
            node.data.title == "while"
                && node.data.condition.as_deref() == Some("(state.count < 2)")
                && node.data.children.iter().any(|child| child.slot == "body")
        }));
        assert!(document.nodes.iter().any(|node| {
            node.node_type == "state_update"
                && node.data.target.as_deref() == Some("state.count")
                && node.data.expression.is_some()
        }));
        assert!(document.nodes.iter().any(|node| {
            node.node_type == "computation"
                && node.data.binding.as_deref() == Some("runs")
                && node.data.expression.as_deref() == Some("[start worker(), start worker()]")
        }));

        let rebuilt = graph_from_document(document, &graph).expect("rebuild promoted graph");
        assert_eq!(
            lashlang::workflow_graph_to_source(&rebuilt).expect("render rebuilt graph"),
            source
        );
        assert_eq!(
            lashlang::workflow_graph_from_source(&source).expect("reproject rebuilt source"),
            graph
        );
    }
}
