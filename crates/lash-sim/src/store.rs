use std::collections::BTreeMap;

use serde_json::{Value, json};

use crate::effects::{DurableEffectJournal, sleep_effect_envelope};
use crate::runtime_contracts::{RuntimeTurnObservation, runtime_turn_contract};
use crate::scheduler::{BoundaryEvent, BoundaryKind};
use crate::trace::{
    AbstractWorldSummary, DurableEffectAbstractSummary, SessionAbstractSummary,
    WorkerAbstractSummary,
};
use crate::workers::SimWorkerTopology;

#[derive(Clone, Debug, Default)]
pub struct ModelStore {
    sessions: BTreeMap<String, ModelSession>,
    durable_effects: BTreeMap<String, ModelDurableEffect>,
    workers: BTreeMap<String, ModelWorker>,
    #[allow(clippy::struct_field_names)]
    durable_journal: DurableEffectJournal,
    worker_topology: SimWorkerTopology,
    total_events: usize,
}

impl ModelStore {
    pub fn open_session(&mut self, alias: impl Into<String>) {
        let alias = alias.into();
        self.sessions
            .entry(alias.clone())
            .or_insert_with(|| ModelSession::new(alias))
            .opened = true;
    }

    pub fn apply_boundary(&mut self, event: &BoundaryEvent) -> Value {
        let observed = self.project_boundary(event);
        self.apply_observed_boundary(event, &observed);
        observed
    }

    pub fn apply_observed_boundary(&mut self, event: &BoundaryEvent, observed: &Value) {
        self.total_events += 1;
        match event.kind {
            BoundaryKind::Ingress => {
                self.open_session(event.actor_alias.clone());
                let session = self
                    .sessions
                    .get_mut(&event.actor_alias)
                    .expect("session was opened");
                session.ingress_count += 1;
            }
            BoundaryKind::Provider => {
                let session = self.ensure_session(event.actor_alias.clone());
                let text = observed
                    .get("provider_output")
                    .and_then(Value::as_str)
                    .or_else(|| event.payload.get("text").and_then(Value::as_str))
                    .unwrap_or("")
                    .to_string();
                session.provider_outputs.push(text);
                if let Some(provider_exchange_count) = observed
                    .get("provider_exchange_count")
                    .and_then(Value::as_u64)
                    .map(|value| value as usize)
                {
                    session
                        .provider_exchange_counts
                        .push(provider_exchange_count);
                }
                if let Some(graph_node_count) = observed
                    .get("graph_node_count")
                    .and_then(Value::as_u64)
                    .map(|value| value as usize)
                {
                    session.graph_node_counts.push(graph_node_count);
                }
                if let Some(transcript_message_count) = observed
                    .get("transcript_message_count")
                    .and_then(Value::as_u64)
                    .map(|value| value as usize)
                {
                    session
                        .transcript_message_counts
                        .push(transcript_message_count);
                }
            }
            BoundaryKind::Tool => {
                let session = self.ensure_session(event.actor_alias.clone());
                let output = observed
                    .get("tool_output")
                    .and_then(Value::as_str)
                    .or_else(|| event.payload.get("output").and_then(Value::as_str))
                    .unwrap_or("")
                    .to_string();
                session.tool_outputs.push(output);
            }
            BoundaryKind::DurableEffect => {
                let session_alias = boundary_session_alias(event);
                let key = observed
                    .get("durable_key")
                    .and_then(Value::as_str)
                    .or_else(|| event.payload.get("durable_key").and_then(Value::as_str))
                    .unwrap_or(&event.boundary_id)
                    .to_string();
                self.ensure_session(session_alias.clone())
                    .durable_effect_keys
                    .push(key.clone());
                self.durable_effects.insert(
                    key.clone(),
                    ModelDurableEffect::from_observed(key, observed),
                );
            }
            BoundaryKind::Worker => {
                let worker_alias = observed
                    .get("worker_alias")
                    .and_then(Value::as_str)
                    .unwrap_or(&event.actor_alias)
                    .to_string();
                self.workers.insert(
                    worker_alias.clone(),
                    ModelWorker::from_observed(worker_alias, observed),
                );
            }
            BoundaryKind::Observer => {
                let session = self.ensure_session(event.actor_alias.clone());
                let turn_index = observed
                    .get("turn_index")
                    .and_then(Value::as_u64)
                    .unwrap_or(session.provider_outputs.len() as u64)
                    as usize;
                session.observer_turn_indices.push(turn_index);
            }
            BoundaryKind::LeaseTime => {
                let session = self.ensure_session(event.actor_alias.clone());
                let tick = event
                    .payload
                    .get("tick")
                    .and_then(Value::as_u64)
                    .unwrap_or(event.at);
                session.lease_time_ticks.push(tick);
            }
        }
    }

    pub fn summary(&self) -> AbstractWorldSummary {
        let sessions = self
            .sessions
            .values()
            .map(ModelSession::summary)
            .collect::<Vec<_>>();
        let durable_effects = self
            .durable_effects
            .values()
            .map(ModelDurableEffect::summary)
            .collect::<Vec<_>>();
        let workers = self
            .workers
            .values()
            .map(ModelWorker::summary)
            .collect::<Vec<_>>();
        AbstractWorldSummary::with_digest(
            self.sessions.len(),
            self.total_events,
            sessions,
            durable_effects,
            workers,
        )
    }

    fn project_boundary(&mut self, event: &BoundaryEvent) -> Value {
        match event.kind {
            BoundaryKind::Ingress => json!({
                "session": event.actor_alias,
                "opened": true,
                "ingress_count": self
                    .sessions
                    .get(&event.actor_alias)
                    .map_or(1, |session| session.ingress_count + 1),
            }),
            BoundaryKind::Provider => {
                let turn_index = self
                    .sessions
                    .get(&event.actor_alias)
                    .map_or(1, |session| session.provider_outputs.len() + 1);
                let text = event
                    .payload
                    .get("text")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                let provider_exchange_count = event
                    .payload
                    .get("expected_provider_exchange_count")
                    .and_then(Value::as_u64)
                    .unwrap_or(turn_index as u64)
                    as usize;
                let graph_node_count = event
                    .payload
                    .get("expected_graph_node_count")
                    .and_then(Value::as_u64)
                    .unwrap_or((turn_index * 2) as u64)
                    as usize;
                let transcript_message_count = event
                    .payload
                    .get("expected_transcript_message_count")
                    .and_then(Value::as_u64)
                    .unwrap_or((turn_index * 2) as u64)
                    as usize;
                let runtime_contract = runtime_turn_contract(
                    &RuntimeTurnObservation {
                        session_id: event.actor_alias.clone(),
                        turn_index,
                        assistant_message: text.clone(),
                        graph_node_count,
                        transcript_message_count,
                        activity_count: 1,
                        provider_exchange_count,
                    },
                    &event.actor_alias,
                    turn_index,
                    &text,
                    provider_exchange_count,
                );
                json!({
                    "session": event.actor_alias,
                    "runtime_session_id": event.actor_alias,
                    "turn_index": turn_index,
                    "success": true,
                    "provider_output": text,
                    "provider_script": event.payload.get("script").cloned().unwrap_or(Value::Null),
                    "provider_exchange_count": provider_exchange_count,
                    "graph_node_count": graph_node_count,
                    "transcript_message_count": transcript_message_count,
                    "activity_count_nonzero": true,
                    "provider_kind": "openai-compatible",
                    "runtime_invariants": {
                        "session_id": true,
                        "turn_index": true,
                        "graph_non_empty": true,
                        "transcript_contains_provider_output": true,
                        "activity_count_nonzero": true,
                    },
                    "runtime_contract": runtime_contract,
                })
            }
            BoundaryKind::Tool => {
                let output = event
                    .payload
                    .get("output")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                json!({
                    "session": event.actor_alias,
                    "tool_output": output,
                    "tool_name": event.payload.get("tool").cloned().unwrap_or(Value::Null),
                })
            }
            BoundaryKind::DurableEffect => {
                let durable_key = event
                    .payload
                    .get("durable_key")
                    .and_then(Value::as_str)
                    .unwrap_or(&event.boundary_id)
                    .to_string();
                let result = event
                    .payload
                    .get("result")
                    .cloned()
                    .unwrap_or_else(|| json!({"completed": true}));
                if let Some(runtime_effect) = event.payload.get("runtime_effect") {
                    let effect_id = runtime_effect
                        .get("effect_id")
                        .and_then(Value::as_str)
                        .unwrap_or(&event.boundary_id);
                    let duration_ms = runtime_effect
                        .get("duration_ms")
                        .and_then(Value::as_u64)
                        .unwrap_or(0);
                    let envelope = sleep_effect_envelope(
                        event.actor_alias.clone(),
                        effect_id.to_string(),
                        durable_key,
                        duration_ms,
                    );
                    self.durable_journal
                        .complete_runtime_effect(envelope, result)
                } else {
                    self.durable_journal.complete(durable_key, result)
                }
            }
            BoundaryKind::Worker => {
                let session_alias = boundary_session_alias(event);
                self.worker_topology
                    .run_stale_completion_script(event.actor_alias.clone(), session_alias)
            }
            BoundaryKind::Observer => {
                let turn_index = self
                    .sessions
                    .get(&event.actor_alias)
                    .map_or(0, |session| session.provider_outputs.len());
                json!({
                    "session": event.actor_alias,
                    "turn_index": turn_index,
                    "graph_node_count": event.payload
                        .get("expected_graph_node_count")
                        .and_then(Value::as_u64)
                        .unwrap_or((turn_index * 2) as u64),
                    "transcript_message_count": event.payload
                        .get("expected_transcript_message_count")
                        .and_then(Value::as_u64)
                        .unwrap_or((turn_index * 2) as u64),
                    "observer_invariants": {
                        "session_id": true,
                        "turn_index_converged": true,
                        "graph_non_empty": turn_index > 0,
                        "transcript_message_count_converged": true,
                    },
                })
            }
            BoundaryKind::LeaseTime => {
                let tick = event
                    .payload
                    .get("tick")
                    .and_then(Value::as_u64)
                    .unwrap_or(event.at);
                let previous_tick = self
                    .sessions
                    .get(&event.actor_alias)
                    .and_then(|session| session.lease_time_ticks.last().copied());
                json!({
                    "session": event.actor_alias,
                    "lease_time_tick": tick,
                    "monotonic": previous_tick.map_or(true, |previous| previous <= tick),
                })
            }
        }
    }

    fn ensure_session(&mut self, alias: impl Into<String>) -> &mut ModelSession {
        let alias = alias.into();
        self.sessions
            .entry(alias.clone())
            .or_insert_with(|| ModelSession::new(alias))
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct ModelSession {
    alias: String,
    opened: bool,
    ingress_count: usize,
    provider_outputs: Vec<String>,
    provider_exchange_counts: Vec<usize>,
    graph_node_counts: Vec<usize>,
    transcript_message_counts: Vec<usize>,
    tool_outputs: Vec<String>,
    observer_turn_indices: Vec<usize>,
    durable_effect_keys: Vec<String>,
    lease_time_ticks: Vec<u64>,
}

impl ModelSession {
    fn new(alias: String) -> Self {
        Self {
            alias,
            opened: false,
            ingress_count: 0,
            provider_outputs: Vec::new(),
            provider_exchange_counts: Vec::new(),
            graph_node_counts: Vec::new(),
            transcript_message_counts: Vec::new(),
            tool_outputs: Vec::new(),
            observer_turn_indices: Vec::new(),
            durable_effect_keys: Vec::new(),
            lease_time_ticks: Vec::new(),
        }
    }

    fn summary(&self) -> SessionAbstractSummary {
        SessionAbstractSummary {
            alias: self.alias.clone(),
            opened: self.opened,
            ingress_count: self.ingress_count,
            provider_outputs: self.provider_outputs.clone(),
            provider_exchange_counts: self.provider_exchange_counts.clone(),
            graph_node_counts: self.graph_node_counts.clone(),
            transcript_message_counts: self.transcript_message_counts.clone(),
            tool_outputs: self.tool_outputs.clone(),
            observer_turn_indices: self.observer_turn_indices.clone(),
            durable_effect_keys: self.durable_effect_keys.clone(),
            lease_time_ticks: self.lease_time_ticks.clone(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ModelDurableEffect {
    durable_key: String,
    execution_count: usize,
    replay_count: usize,
    result_digest: String,
}

impl ModelDurableEffect {
    fn from_observed(durable_key: String, observed: &Value) -> Self {
        Self {
            durable_key,
            execution_count: observed
                .get("execution_count")
                .and_then(Value::as_u64)
                .unwrap_or(0) as usize,
            replay_count: observed
                .get("replay_count")
                .and_then(Value::as_u64)
                .unwrap_or(0) as usize,
            result_digest: observed
                .get("result_digest")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string(),
        }
    }

    fn summary(&self) -> DurableEffectAbstractSummary {
        DurableEffectAbstractSummary {
            durable_key: self.durable_key.clone(),
            execution_count: self.execution_count,
            replay_count: self.replay_count,
            result_digest: self.result_digest.clone(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ModelWorker {
    worker_alias: String,
    session_alias: String,
    active_incarnation_id: String,
    active_fencing_token: u64,
    lease_owner_changes: usize,
    stale_completion_rejections: usize,
}

impl ModelWorker {
    fn from_observed(worker_alias: String, observed: &Value) -> Self {
        Self {
            worker_alias,
            session_alias: observed
                .get("session")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string(),
            active_incarnation_id: observed
                .get("active_owner")
                .and_then(|owner| owner.get("incarnation_id"))
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string(),
            active_fencing_token: observed
                .get("active_fencing_token")
                .and_then(Value::as_u64)
                .unwrap_or(0),
            lease_owner_changes: usize::from(
                observed
                    .get("lease_owner_changed")
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
            ),
            stale_completion_rejections: usize::from(
                observed
                    .get("stale_completion_rejected")
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
            ),
        }
    }

    fn summary(&self) -> WorkerAbstractSummary {
        WorkerAbstractSummary {
            worker_alias: self.worker_alias.clone(),
            session_alias: self.session_alias.clone(),
            active_incarnation_id: self.active_incarnation_id.clone(),
            active_fencing_token: self.active_fencing_token,
            lease_owner_changes: self.lease_owner_changes,
            stale_completion_rejections: self.stale_completion_rejections,
        }
    }
}

fn boundary_session_alias(event: &BoundaryEvent) -> String {
    event
        .payload
        .get("session")
        .and_then(Value::as_str)
        .unwrap_or(&event.actor_alias)
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduler::{BoundaryEvent, BoundaryKind};

    #[test]
    fn model_store_keeps_cross_session_outputs_isolated() {
        let mut store = ModelStore::default();
        store.apply_boundary(&BoundaryEvent::new(
            "open-1",
            "session-001",
            BoundaryKind::Ingress,
            0,
            "session.open",
            json!({}),
        ));
        store.apply_boundary(&BoundaryEvent::new(
            "open-2",
            "session-002",
            BoundaryKind::Ingress,
            0,
            "session.open",
            json!({}),
        ));
        store.apply_boundary(&BoundaryEvent::new(
            "p1",
            "session-001",
            BoundaryKind::Provider,
            1,
            "provider",
            json!({"text": "one"}),
        ));
        store.apply_boundary(&BoundaryEvent::new(
            "p2",
            "session-002",
            BoundaryKind::Provider,
            1,
            "provider",
            json!({"text": "two"}),
        ));

        let summary = store.summary();
        assert_eq!(summary.session_count, 2);
        assert_eq!(summary.sessions[0].provider_outputs, vec!["one"]);
        assert_eq!(summary.sessions[1].provider_outputs, vec!["two"]);
        assert_ne!(
            summary.sessions[0].provider_outputs,
            summary.sessions[1].provider_outputs
        );
    }

    #[test]
    fn model_store_projects_semantic_boundary_summaries() {
        let mut store = ModelStore::default();
        store.apply_boundary(&BoundaryEvent::new(
            "open-1",
            "session-001",
            BoundaryKind::Ingress,
            0,
            "session.open",
            json!({}),
        ));
        store.apply_boundary(&BoundaryEvent::new(
            "provider-1",
            "session-001",
            BoundaryKind::Provider,
            1,
            "provider.chat.stream",
            json!({"text": "answer for session-001"}),
        ));
        store.apply_boundary(&BoundaryEvent::new(
            "observer-1",
            "session-001",
            BoundaryKind::Observer,
            2,
            "observer.snapshot",
            json!({}),
        ));
        store.apply_boundary(&BoundaryEvent::new(
            "effect-1",
            "session-001",
            BoundaryKind::DurableEffect,
            3,
            "durable.sleep.complete",
            json!({"durable_key": "sleep/session-001", "result": {"done": true}}),
        ));
        store.apply_boundary(&BoundaryEvent::new(
            "effect-1-replay",
            "session-001",
            BoundaryKind::DurableEffect,
            4,
            "durable.sleep.replay",
            json!({"durable_key": "sleep/session-001", "result": {"done": false}}),
        ));
        store.apply_boundary(&BoundaryEvent::new(
            "worker-1",
            "worker-001",
            BoundaryKind::Worker,
            5,
            "worker.stale-completion-rejected",
            json!({"session": "session-001"}),
        ));

        let summary = store.summary();
        assert_eq!(summary.sessions[0].observer_turn_indices, vec![1]);
        assert_eq!(summary.durable_effects[0].execution_count, 1);
        assert_eq!(summary.durable_effects[0].replay_count, 1);
        assert_eq!(summary.workers[0].stale_completion_rejections, 1);
    }
}
