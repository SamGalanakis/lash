use std::collections::BTreeMap;

use axum::http::StatusCode;
use lashlang::GraphRenderError;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WorkflowDocument {
    pub schema_version: u32,
    pub version: u64,
    pub source: String,
    pub nodes: Vec<FlowNode>,
    pub edges: Vec<FlowEdge>,
    pub roots: GraphRoots,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GraphRoots {
    pub main: Vec<String>,
    #[serde(default)]
    pub processes: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FlowNode {
    pub id: String,
    #[serde(rename = "type")]
    pub node_type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_id: Option<String>,
    pub data: NodeData,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NodeData {
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subkind: Option<String>,
    pub title: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default = "derived_name_source")]
    pub name_source: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub operation: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effect: Option<String>,
    #[serde(default)]
    pub fields: BTreeMap<String, EditableValue>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub binding: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expression: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub condition: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub iterable: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub clauses: Vec<EditableComprehensionClause>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub children: Vec<ChildGroup>,
}

fn derived_name_source() -> String {
    "derived".to_string()
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EditableComprehensionClause {
    For { binding: String, iterable: String },
    If { condition: String },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChildGroup {
    pub slot: String,
    pub scope: String,
    pub node_ids: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum EditableValue {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    List(Vec<EditableValue>),
    Object(BTreeMap<String, EditableValue>),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FlowEdge {
    pub id: String,
    pub source: String,
    pub target: String,
    pub data: EdgeData,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EdgeData {
    pub kind: String,
    pub scope: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub variable: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<u32>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DisplayState {
    pub messages: Vec<String>,
    pub statuses: BTreeMap<String, String>,
    pub lists: BTreeMap<String, Vec<String>>,
    pub lights: BTreeMap<String, String>,
    pub progress: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub highlighted: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DisplayDelta {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub messages_appended: Vec<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub statuses: BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub list_items_appended: BTreeMap<String, Vec<String>>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub lights: BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub progress: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub highlighted: Option<String>,
}

impl DisplayDelta {
    pub(crate) fn merge(&mut self, other: Self) {
        self.messages_appended.extend(other.messages_appended);
        self.statuses.extend(other.statuses);
        for (list, items) in other.list_items_appended {
            self.list_items_appended
                .entry(list)
                .or_default()
                .extend(items);
        }
        self.lights.extend(other.lights);
        if other.progress.is_some() {
            self.progress = other.progress;
        }
        if other.highlighted.is_some() {
            self.highlighted = other.highlighted;
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    Started,
    Succeeded,
    Waiting,
    Failed,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RunEvent {
    pub run_id: String,
    pub workflow_version: u64,
    pub sequence: u64,
    pub node_id: String,
    pub status: RunStatus,
    #[serde(default)]
    pub display_delta: DisplayDelta,
    pub display: DisplayState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ErrorBody {
    pub error: ErrorDetail,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ErrorDetail {
    pub code: String,
    pub message: String,
    pub details: Value,
}

#[derive(Clone, Debug)]
pub struct RenderErrorResponse {
    pub(crate) status: StatusCode,
    pub(crate) body: ErrorBody,
}

impl RenderErrorResponse {
    pub(crate) fn render(error: GraphRenderError) -> Self {
        let message = error.to_string();
        let (code, details) = match &error {
            GraphRenderError::UnsupportedSchemaVersion { found, expected } => (
                "unsupported_schema_version",
                json!({ "found": found, "expected": expected }),
            ),
            GraphRenderError::DuplicateNodeId { id } => ("duplicate_node_id", json!({ "id": id })),
            GraphRenderError::UnknownNodeReference {
                edge_id,
                endpoint,
                node_id,
            } => (
                "unknown_node_reference",
                json!({ "edgeId": edge_id, "endpoint": endpoint, "nodeId": node_id }),
            ),
            GraphRenderError::MissingRequiredChild { node_id, child } => (
                "missing_required_child",
                json!({ "nodeId": node_id, "child": child }),
            ),
            GraphRenderError::InvalidNodePayload { node_id, message } => (
                "invalid_node_payload",
                json!({ "nodeId": node_id, "reason": message }),
            ),
            GraphRenderError::InvalidExpression {
                node_id,
                field,
                message,
            } => (
                "invalid_expression",
                json!({ "nodeId": node_id, "field": field, "reason": message }),
            ),
            GraphRenderError::InvalidAssignmentTarget {
                node_id,
                field,
                message,
            } => (
                "invalid_assignment_target",
                json!({ "nodeId": node_id, "field": field, "reason": message }),
            ),
            GraphRenderError::InvalidOpaqueSource { node_id, message } => (
                "invalid_opaque_source",
                json!({ "nodeId": node_id, "reason": message }),
            ),
            GraphRenderError::DuplicateProcessName { name } => {
                ("duplicate_process_name", json!({ "name": name }))
            }
            GraphRenderError::CanonicalSource(_) => ("canonical_source", json!({})),
            GraphRenderError::RenderedSourceInvalid { message } => {
                ("rendered_source_invalid", json!({ "reason": message }))
            }
        };
        Self::new(StatusCode::UNPROCESSABLE_ENTITY, code, message, details)
    }

    pub(crate) fn projection(error: lashlang::WorkflowGraphBuildError) -> Self {
        Self::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "projection_failed",
            error.to_string(),
            json!({}),
        )
    }

    pub(crate) fn document(message: impl Into<String>, details: Value) -> Self {
        Self::new(
            StatusCode::UNPROCESSABLE_ENTITY,
            "invalid_graph_document",
            message,
            details,
        )
    }

    pub(crate) fn invalid_expression(
        node_id: &str,
        field: &'static str,
        message: impl Into<String>,
    ) -> Self {
        let message = message.into();
        Self::new(
            StatusCode::UNPROCESSABLE_ENTITY,
            "invalid_expression",
            format!("node `{node_id}` has invalid `{field}` expression text: {message}"),
            json!({ "nodeId": node_id, "field": field, "reason": message }),
        )
    }

    pub(crate) fn invalid_assignment_target(
        node_id: &str,
        field: &'static str,
        message: impl Into<String>,
    ) -> Self {
        let message = message.into();
        Self::new(
            StatusCode::UNPROCESSABLE_ENTITY,
            "invalid_assignment_target",
            format!("node `{node_id}` has invalid `{field}` assignment target text: {message}"),
            json!({ "nodeId": node_id, "field": field, "reason": message }),
        )
    }

    pub(crate) fn invalid_node_payload(node_id: &str, message: impl Into<String>) -> Self {
        let message = message.into();
        Self::new(
            StatusCode::UNPROCESSABLE_ENTITY,
            "invalid_node_payload",
            format!("node `{node_id}` has an invalid payload: {message}"),
            json!({ "nodeId": node_id, "reason": message }),
        )
    }

    pub(crate) fn unknown_node_kind(node_id: &str, kind: &str, subkind: Option<&str>) -> Self {
        Self::new(
            StatusCode::UNPROCESSABLE_ENTITY,
            "unknown_node_kind",
            match subkind {
                Some(subkind) => {
                    format!("flow node `{node_id}` has unknown kind `{kind}:{subkind}`")
                }
                None => format!("flow node `{node_id}` has unknown kind `{kind}`"),
            },
            json!({ "nodeId": node_id, "kind": kind, "subkind": subkind }),
        )
    }

    pub(crate) fn run_preparation(message: impl ToString) -> Self {
        Self::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "run_preparation_failed",
            message.to_string(),
            json!({}),
        )
    }

    pub(crate) fn version_conflict(submitted: u64, current: u64) -> Self {
        Self::new(
            StatusCode::CONFLICT,
            "version_conflict",
            format!("workflow version {submitted} is stale; current version is {current}"),
            json!({ "submitted": submitted, "current": current }),
        )
    }

    pub(crate) fn unknown_workflow(id: &str) -> Self {
        Self::new(
            StatusCode::NOT_FOUND,
            "unknown_workflow",
            format!("workflow example `{id}` does not exist"),
            json!({ "id": id }),
        )
    }

    fn new(
        status: StatusCode,
        code: impl Into<String>,
        message: impl Into<String>,
        details: Value,
    ) -> Self {
        Self {
            status,
            body: ErrorBody {
                error: ErrorDetail {
                    code: code.into(),
                    message: message.into(),
                    details,
                },
            },
        }
    }
}
