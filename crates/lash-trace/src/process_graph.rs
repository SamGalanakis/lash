use std::collections::{BTreeMap, BTreeSet};
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

use crate::{
    TraceEvent, TraceProcessMap, TraceProcessStatus, TraceProcessTrackingEvent, TraceRecord,
    TraceSink, TraceSinkError,
};

/// Trace-derived process graph snapshot for hosts and debugging tools.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceProcessGraph {
    pub process_id: String,
    pub session_id: String,
    pub module_ref: String,
    pub process_ref: String,
    pub process_name: String,
    pub status: TraceProcessStatus,
    pub nodes: Vec<TraceProcessGraphNode>,
    pub edges: Vec<TraceProcessGraphEdge>,
    pub children: Vec<TraceProcessGraphChildLink>,
}

/// Observed process node state.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TraceProcessNodeStatus {
    #[default]
    Unobserved,
    Running,
    Completed,
    Failed,
}

/// Observed branch-edge selection state.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TraceProcessEdgeSelection {
    #[default]
    Unknown,
    Selected,
    Rejected,
}

/// Trace-derived process graph node.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceProcessGraphNode {
    pub id: String,
    pub kind: String,
    pub label: String,
    pub status: TraceProcessNodeStatus,
    pub first_timestamp: Option<String>,
    pub last_timestamp: Option<String>,
    pub duration_ms: Option<i64>,
    pub latest_error: Option<String>,
    pub occurrence: Option<u64>,
}

impl TraceProcessGraphNode {
    fn unobserved(
        id: impl Into<String>,
        kind: impl Into<String>,
        label: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            kind: kind.into(),
            label: label.into(),
            status: TraceProcessNodeStatus::Unobserved,
            first_timestamp: None,
            last_timestamp: None,
            duration_ms: None,
            latest_error: None,
            occurrence: None,
        }
    }
}

/// Trace-derived process graph edge.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceProcessGraphEdge {
    pub id: String,
    pub from: String,
    pub to: String,
    pub label: String,
    pub selection: TraceProcessEdgeSelection,
}

/// Link from an observed parent process node to a child process.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceProcessGraphChildLink {
    pub parent_process_id: String,
    pub parent_node_id: String,
    pub child_process_id: String,
    pub child_process_name: String,
}

/// In-memory store that reduces process-tracking trace records into graph snapshots.
#[derive(Default)]
pub struct TraceProcessGraphStore {
    inner: Mutex<TraceProcessGraphState>,
}

#[derive(Default)]
struct TraceProcessGraphState {
    seen_event_keys: BTreeSet<String>,
    processes: BTreeMap<String, TraceProcessGraphAccumulator>,
}

#[derive(Clone, Debug)]
struct TraceProcessGraphAccumulator {
    process_id: String,
    session_id: String,
    module_ref: String,
    process_ref: String,
    process_name: String,
    status: TraceProcessStatus,
    nodes: BTreeMap<String, TraceProcessGraphNode>,
    edges: BTreeMap<String, TraceProcessGraphEdge>,
    children: Vec<TraceProcessGraphChildLink>,
}

impl TraceProcessGraphStore {
    /// Returns a snapshot for one observed process.
    pub fn graph(&self, process_id: &str) -> Option<TraceProcessGraph> {
        self.inner
            .lock()
            .ok()?
            .processes
            .get(process_id)
            .map(TraceProcessGraphAccumulator::to_graph)
    }

    /// Returns snapshots for all observed processes in stable process-id order.
    pub fn graphs(&self) -> Vec<TraceProcessGraph> {
        self.inner
            .lock()
            .map(|state| {
                state
                    .processes
                    .values()
                    .map(TraceProcessGraphAccumulator::to_graph)
                    .collect()
            })
            .unwrap_or_default()
    }
}

impl TraceSink for TraceProcessGraphStore {
    fn append(&self, record: &TraceRecord) -> Result<(), TraceSinkError> {
        let TraceEvent::ProcessTracking { event } = &record.event else {
            return Ok(());
        };
        let event_key = process_tracking_event_key(event);
        let mut state = self
            .inner
            .lock()
            .map_err(|_| TraceSinkError::LockPoisoned)?;
        if !state.seen_event_keys.insert(event_key.to_string()) {
            return Ok(());
        }
        reduce_process_tracking_event(&mut state, event, &record.timestamp);
        Ok(())
    }
}

impl TraceProcessGraphAccumulator {
    fn new(
        process_id: &str,
        session_id: &str,
        module_ref: &str,
        process_ref: &str,
        process_name: &str,
    ) -> Self {
        Self {
            process_id: process_id.to_string(),
            session_id: session_id.to_string(),
            module_ref: module_ref.to_string(),
            process_ref: process_ref.to_string(),
            process_name: process_name.to_string(),
            status: TraceProcessStatus::Running,
            nodes: BTreeMap::new(),
            edges: BTreeMap::new(),
            children: Vec::new(),
        }
    }

    fn to_graph(&self) -> TraceProcessGraph {
        TraceProcessGraph {
            process_id: self.process_id.clone(),
            session_id: self.session_id.clone(),
            module_ref: self.module_ref.clone(),
            process_ref: self.process_ref.clone(),
            process_name: self.process_name.clone(),
            status: self.status,
            nodes: self.nodes.values().cloned().collect(),
            edges: self.edges.values().cloned().collect(),
            children: self.children.clone(),
        }
    }
}

fn reduce_process_tracking_event(
    state: &mut TraceProcessGraphState,
    event: &TraceProcessTrackingEvent,
    timestamp: &str,
) {
    match event {
        TraceProcessTrackingEvent::ProcessStarted {
            process_id,
            session_id,
            module_ref,
            process_ref,
            process_name,
            process_map,
            ..
        } => seed_process_graph(
            graph_mut(
                state,
                process_id,
                session_id,
                module_ref,
                process_ref,
                process_name,
            ),
            process_map,
        ),
        TraceProcessTrackingEvent::ProcessFinished {
            process_id,
            session_id,
            module_ref,
            process_ref,
            process_name,
            status,
            ..
        } => {
            graph_mut(
                state,
                process_id,
                session_id,
                module_ref,
                process_ref,
                process_name,
            )
            .status = *status;
        }
        TraceProcessTrackingEvent::NodeStarted {
            process_id,
            session_id,
            module_ref,
            process_ref,
            process_name,
            node_id,
            node_kind,
            label,
            occurrence,
            ..
        } => {
            let node = node_mut(
                state,
                TraceProcessNodeIdentity {
                    process_id,
                    session_id,
                    module_ref,
                    process_ref,
                    process_name,
                    node_id,
                    node_kind,
                    label,
                },
            );
            if node.first_timestamp.is_none() {
                node.first_timestamp = Some(timestamp.to_string());
            }
            node.last_timestamp = Some(timestamp.to_string());
            node.status = TraceProcessNodeStatus::Running;
            node.occurrence = Some(*occurrence);
        }
        TraceProcessTrackingEvent::NodeCompleted {
            process_id,
            session_id,
            module_ref,
            process_ref,
            process_name,
            node_id,
            node_kind,
            label,
            occurrence,
            ..
        } => {
            let node = node_mut(
                state,
                TraceProcessNodeIdentity {
                    process_id,
                    session_id,
                    module_ref,
                    process_ref,
                    process_name,
                    node_id,
                    node_kind,
                    label,
                },
            );
            node.last_timestamp = Some(timestamp.to_string());
            node.duration_ms = duration_ms(node.first_timestamp.as_deref(), Some(timestamp));
            node.status = TraceProcessNodeStatus::Completed;
            node.occurrence = Some(*occurrence);
        }
        TraceProcessTrackingEvent::NodeFailed {
            process_id,
            session_id,
            module_ref,
            process_ref,
            process_name,
            node_id,
            node_kind,
            label,
            occurrence,
            error,
            ..
        } => {
            let node = node_mut(
                state,
                TraceProcessNodeIdentity {
                    process_id,
                    session_id,
                    module_ref,
                    process_ref,
                    process_name,
                    node_id,
                    node_kind,
                    label,
                },
            );
            node.last_timestamp = Some(timestamp.to_string());
            node.duration_ms = duration_ms(node.first_timestamp.as_deref(), Some(timestamp));
            node.status = TraceProcessNodeStatus::Failed;
            node.latest_error = Some(error.clone());
            node.occurrence = Some(*occurrence);
        }
        TraceProcessTrackingEvent::BranchSelected {
            process_id,
            session_id,
            module_ref,
            process_ref,
            process_name,
            node_id,
            occurrence,
            edge_id,
            ..
        } => {
            let graph = graph_mut(
                state,
                process_id,
                session_id,
                module_ref,
                process_ref,
                process_name,
            );
            if let Some(node) = graph.nodes.get_mut(node_id) {
                node.status = TraceProcessNodeStatus::Completed;
                node.last_timestamp = Some(timestamp.to_string());
                node.occurrence = Some(*occurrence);
            }
            let selected_from = graph.edges.get(edge_id).map(|edge| edge.from.clone());
            if let Some(edge) = graph.edges.get_mut(edge_id) {
                edge.selection = TraceProcessEdgeSelection::Selected;
            }
            if let Some(selected_from) = selected_from {
                for edge in graph.edges.values_mut() {
                    if edge.from == selected_from
                        && matches!(edge.label.as_str(), "then" | "else")
                        && edge.id != *edge_id
                    {
                        edge.selection = TraceProcessEdgeSelection::Rejected;
                    }
                }
            }
        }
        TraceProcessTrackingEvent::ChildStarted {
            process_id,
            session_id,
            module_ref,
            process_ref,
            process_name,
            parent_process_id,
            parent_node_id,
            child_process_id,
            child_process_name,
            ..
        } => {
            let graph = graph_mut(
                state,
                process_id,
                session_id,
                module_ref,
                process_ref,
                process_name,
            );
            if !graph.children.iter().any(|child| {
                child.parent_node_id == *parent_node_id
                    && child.child_process_id == *child_process_id
            }) {
                graph.children.push(TraceProcessGraphChildLink {
                    parent_process_id: parent_process_id.clone(),
                    parent_node_id: parent_node_id.clone(),
                    child_process_id: child_process_id.clone(),
                    child_process_name: child_process_name.clone(),
                });
            }
        }
    }
}

fn seed_process_graph(graph: &mut TraceProcessGraphAccumulator, process_map: &TraceProcessMap) {
    graph.status = TraceProcessStatus::Running;
    for node in &process_map.nodes {
        graph.nodes.entry(node.id.clone()).or_insert_with(|| {
            TraceProcessGraphNode::unobserved(
                node.id.clone(),
                node.kind.clone(),
                node.label.clone(),
            )
        });
    }
    for edge in &process_map.edges {
        graph
            .edges
            .entry(edge.id.clone())
            .or_insert_with(|| TraceProcessGraphEdge {
                id: edge.id.clone(),
                from: edge.from.clone(),
                to: edge.to.clone(),
                label: edge.label.clone(),
                selection: TraceProcessEdgeSelection::Unknown,
            });
    }
}

#[derive(Clone, Copy)]
struct TraceProcessNodeIdentity<'event> {
    process_id: &'event str,
    session_id: &'event str,
    module_ref: &'event str,
    process_ref: &'event str,
    process_name: &'event str,
    node_id: &'event str,
    node_kind: &'event str,
    label: &'event str,
}

fn graph_mut<'a>(
    state: &'a mut TraceProcessGraphState,
    process_id: &str,
    session_id: &str,
    module_ref: &str,
    process_ref: &str,
    process_name: &str,
) -> &'a mut TraceProcessGraphAccumulator {
    state
        .processes
        .entry(process_id.to_string())
        .or_insert_with(|| {
            TraceProcessGraphAccumulator::new(
                process_id,
                session_id,
                module_ref,
                process_ref,
                process_name,
            )
        })
}

fn node_mut<'a>(
    state: &'a mut TraceProcessGraphState,
    identity: TraceProcessNodeIdentity<'_>,
) -> &'a mut TraceProcessGraphNode {
    graph_mut(
        state,
        identity.process_id,
        identity.session_id,
        identity.module_ref,
        identity.process_ref,
        identity.process_name,
    )
    .nodes
    .entry(identity.node_id.to_string())
    .or_insert_with(|| {
        TraceProcessGraphNode::unobserved(identity.node_id, identity.node_kind, identity.label)
    })
}

fn process_tracking_event_key(event: &TraceProcessTrackingEvent) -> &str {
    match event {
        TraceProcessTrackingEvent::ProcessStarted { event_key, .. }
        | TraceProcessTrackingEvent::ProcessFinished { event_key, .. }
        | TraceProcessTrackingEvent::NodeStarted { event_key, .. }
        | TraceProcessTrackingEvent::NodeCompleted { event_key, .. }
        | TraceProcessTrackingEvent::NodeFailed { event_key, .. }
        | TraceProcessTrackingEvent::BranchSelected { event_key, .. }
        | TraceProcessTrackingEvent::ChildStarted { event_key, .. } => event_key,
    }
}

fn duration_ms(first: Option<&str>, last: Option<&str>) -> Option<i64> {
    let first = chrono::DateTime::parse_from_rfc3339(first?).ok()?;
    let last = chrono::DateTime::parse_from_rfc3339(last?).ok()?;
    Some((last - first).num_milliseconds().max(0))
}

#[cfg(test)]
mod tests {
    use chrono::{TimeZone, Utc};
    use serde_json::json;

    use super::*;
    use crate::{TraceBranchSelection, TraceContext, TraceProcessMapEdge, TraceProcessMapNode};

    fn append_at(store: &TraceProcessGraphStore, event: TraceProcessTrackingEvent, ms: i64) {
        store
            .append(&TraceRecord::new_with_timestamp(
                TraceContext::default().for_session("session-1"),
                TraceEvent::ProcessTracking { event },
                Utc.timestamp_millis_opt(ms).single().expect("timestamp"),
            ))
            .expect("append process tracking event");
    }

    fn started_event(event_key: &str) -> TraceProcessTrackingEvent {
        TraceProcessTrackingEvent::ProcessStarted {
            event_key: event_key.to_string(),
            process_id: "process-1".to_string(),
            session_id: "session-1".to_string(),
            module_ref: "module-1".to_string(),
            process_ref: "process:0".to_string(),
            process_name: "main".to_string(),
            process_map: TraceProcessMap {
                module_ref: "module-1".to_string(),
                process_ref: "process:0".to_string(),
                process_name: "main".to_string(),
                nodes: vec![TraceProcessMapNode {
                    id: "branch".to_string(),
                    kind: "branch".to_string(),
                    label: "if ready".to_string(),
                }],
                edges: vec![
                    TraceProcessMapEdge {
                        id: "then-edge".to_string(),
                        from: "branch".to_string(),
                        to: "then".to_string(),
                        label: "then".to_string(),
                    },
                    TraceProcessMapEdge {
                        id: "else-edge".to_string(),
                        from: "branch".to_string(),
                        to: "else".to_string(),
                        label: "else".to_string(),
                    },
                ],
            },
        }
    }

    fn node_started(event_key: &str, occurrence: u64) -> TraceProcessTrackingEvent {
        TraceProcessTrackingEvent::NodeStarted {
            event_key: event_key.to_string(),
            process_id: "process-1".to_string(),
            session_id: "session-1".to_string(),
            module_ref: "module-1".to_string(),
            process_ref: "process:0".to_string(),
            process_name: "main".to_string(),
            node_id: "branch".to_string(),
            node_kind: "branch".to_string(),
            label: "if ready".to_string(),
            occurrence,
        }
    }

    fn node_completed(event_key: &str, occurrence: u64) -> TraceProcessTrackingEvent {
        TraceProcessTrackingEvent::NodeCompleted {
            event_key: event_key.to_string(),
            process_id: "process-1".to_string(),
            session_id: "session-1".to_string(),
            module_ref: "module-1".to_string(),
            process_ref: "process:0".to_string(),
            process_name: "main".to_string(),
            node_id: "branch".to_string(),
            node_kind: "branch".to_string(),
            label: "if ready".to_string(),
            occurrence,
        }
    }

    fn node_failed(event_key: &str, occurrence: u64) -> TraceProcessTrackingEvent {
        TraceProcessTrackingEvent::NodeFailed {
            event_key: event_key.to_string(),
            process_id: "process-1".to_string(),
            session_id: "session-1".to_string(),
            module_ref: "module-1".to_string(),
            process_ref: "process:0".to_string(),
            process_name: "main".to_string(),
            node_id: "branch".to_string(),
            node_kind: "branch".to_string(),
            label: "if ready".to_string(),
            occurrence,
            error: "not ready".to_string(),
        }
    }

    #[test]
    fn graph_store_seeds_static_map_on_process_start() {
        let store = TraceProcessGraphStore::default();

        append_at(&store, started_event("start"), 1_000);

        let graph = store.graph("process-1").expect("graph");
        assert_eq!(graph.status, TraceProcessStatus::Running);
        assert_eq!(graph.nodes[0].status, TraceProcessNodeStatus::Unobserved);
        assert_eq!(graph.edges[0].selection, TraceProcessEdgeSelection::Unknown);
    }

    #[test]
    fn graph_store_ignores_duplicate_event_keys() {
        let store = TraceProcessGraphStore::default();

        append_at(&store, node_started("same-key", 1), 1_000);
        append_at(&store, node_completed("same-key", 1), 1_250);

        let graph = store.graph("process-1").expect("graph");
        assert_eq!(graph.nodes[0].status, TraceProcessNodeStatus::Running);
    }

    #[test]
    fn graph_store_updates_completed_node_duration() {
        let store = TraceProcessGraphStore::default();

        append_at(&store, node_started("start-node", 1), 1_000);
        append_at(&store, node_completed("complete-node", 2), 1_750);

        let graph = store.graph("process-1").expect("graph");
        let node = &graph.nodes[0];
        assert_eq!(node.status, TraceProcessNodeStatus::Completed);
        assert_eq!(node.duration_ms, Some(750));
        assert_eq!(node.occurrence, Some(2));
    }

    #[test]
    fn graph_store_updates_failed_node_error() {
        let store = TraceProcessGraphStore::default();

        append_at(&store, node_started("start-node", 1), 1_000);
        append_at(&store, node_failed("fail-node", 3), 1_125);

        let graph = store.graph("process-1").expect("graph");
        let node = &graph.nodes[0];
        assert_eq!(node.status, TraceProcessNodeStatus::Failed);
        assert_eq!(node.latest_error.as_deref(), Some("not ready"));
    }

    #[test]
    fn graph_store_marks_selected_and_rejected_branch_edges() {
        let store = TraceProcessGraphStore::default();

        append_at(&store, started_event("start"), 1_000);
        append_at(
            &store,
            TraceProcessTrackingEvent::BranchSelected {
                event_key: "branch".to_string(),
                process_id: "process-1".to_string(),
                session_id: "session-1".to_string(),
                module_ref: "module-1".to_string(),
                process_ref: "process:0".to_string(),
                process_name: "main".to_string(),
                node_id: "branch".to_string(),
                occurrence: 1,
                edge_id: "then-edge".to_string(),
                selected: TraceBranchSelection::Then,
            },
            1_100,
        );

        let graph = store.graph("process-1").expect("graph");
        assert_eq!(
            graph
                .edges
                .iter()
                .find(|edge| edge.id == "then-edge")
                .map(|edge| edge.selection),
            Some(TraceProcessEdgeSelection::Selected)
        );
        assert_eq!(
            graph
                .edges
                .iter()
                .find(|edge| edge.id == "else-edge")
                .map(|edge| edge.selection),
            Some(TraceProcessEdgeSelection::Rejected)
        );
    }

    #[test]
    fn graph_store_records_child_links_once() {
        let store = TraceProcessGraphStore::default();
        let event = TraceProcessTrackingEvent::ChildStarted {
            event_key: "child".to_string(),
            process_id: "process-1".to_string(),
            session_id: "session-1".to_string(),
            module_ref: "module-1".to_string(),
            process_ref: "process:0".to_string(),
            process_name: "main".to_string(),
            parent_process_id: "process-1".to_string(),
            parent_node_id: "branch".to_string(),
            occurrence: 1,
            child_process_id: "process-2".to_string(),
            child_module_ref: "module-1".to_string(),
            child_process_ref: "child:0".to_string(),
            child_process_name: "child".to_string(),
        };

        append_at(&store, event.clone(), 1_000);
        append_at(&store, event, 1_100);

        let graph = store.graph("process-1").expect("graph");
        assert_eq!(graph.children.len(), 1);
        assert_eq!(graph.children[0].child_process_id, "process-2");
    }

    #[test]
    fn graph_store_sets_process_finished_status() {
        let store = TraceProcessGraphStore::default();

        append_at(&store, started_event("start"), 1_000);
        append_at(
            &store,
            TraceProcessTrackingEvent::ProcessFinished {
                event_key: "finish".to_string(),
                process_id: "process-1".to_string(),
                session_id: "session-1".to_string(),
                module_ref: "module-1".to_string(),
                process_ref: "process:0".to_string(),
                process_name: "main".to_string(),
                status: TraceProcessStatus::Completed,
                error: None,
            },
            1_500,
        );

        let graph = store.graph("process-1").expect("graph");
        assert_eq!(graph.status, TraceProcessStatus::Completed);
    }

    #[test]
    fn graph_dtos_serialize_status_and_selection_as_snake_case() {
        let node_status = serde_json::to_value(TraceProcessNodeStatus::Unobserved)
            .expect("serialize node status");
        let edge_selection = serde_json::to_value(TraceProcessEdgeSelection::Rejected)
            .expect("serialize edge selection");

        assert_eq!(node_status, json!("unobserved"));
        assert_eq!(edge_selection, json!("rejected"));
    }
}
