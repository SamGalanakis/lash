use std::collections::VecDeque;

use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const BOUNDARY_EVENT_SCHEMA: &str = "lash.sim.boundary-event.v1";

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BoundaryKind {
    Ingress,
    Provider,
    Tool,
    DurableEffect,
    Worker,
    Observer,
    LeaseTime,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct BoundaryEvent {
    pub boundary_id: String,
    pub actor_alias: String,
    pub kind: BoundaryKind,
    pub at: u64,
    pub label: String,
    #[serde(default)]
    pub payload: Value,
}

impl BoundaryEvent {
    pub fn new(
        boundary_id: impl Into<String>,
        actor_alias: impl Into<String>,
        kind: BoundaryKind,
        at: u64,
        label: impl Into<String>,
        payload: Value,
    ) -> Self {
        Self {
            boundary_id: boundary_id.into(),
            actor_alias: actor_alias.into(),
            kind,
            at,
            label: label.into(),
            payload,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct DeliveredBoundary {
    pub schema: String,
    pub sequence: usize,
    pub boundary_id: String,
    pub actor_alias: String,
    pub kind: BoundaryKind,
    pub at: u64,
    pub label: String,
    #[serde(default)]
    pub payload: Value,
    #[serde(default)]
    pub observed: Value,
}

impl DeliveredBoundary {
    fn from_event(sequence: usize, event: BoundaryEvent, observed: Value) -> Self {
        Self {
            schema: BOUNDARY_EVENT_SCHEMA.to_string(),
            sequence,
            boundary_id: event.boundary_id,
            actor_alias: event.actor_alias,
            kind: event.kind,
            at: event.at,
            label: event.label,
            payload: event.payload,
            observed,
        }
    }

    pub fn as_event(&self) -> BoundaryEvent {
        BoundaryEvent {
            boundary_id: self.boundary_id.clone(),
            actor_alias: self.actor_alias.clone(),
            kind: self.kind,
            at: self.at,
            label: self.label.clone(),
            payload: self.payload.clone(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct BoundaryScheduler {
    pending: Vec<BoundaryEvent>,
    seed: u64,
    sequence: usize,
}

impl BoundaryScheduler {
    pub fn new(seed: u64) -> Self {
        Self {
            pending: Vec::new(),
            seed,
            sequence: 0,
        }
    }

    pub fn with_events(seed: u64, events: impl IntoIterator<Item = BoundaryEvent>) -> Self {
        let mut scheduler = Self::new(seed);
        for event in events {
            scheduler.schedule(event);
        }
        scheduler
    }

    pub fn schedule(&mut self, event: BoundaryEvent) {
        self.pending.push(event);
    }

    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    pub fn deliver_next(&mut self, observed: Value) -> Option<DeliveredBoundary> {
        let index = self.next_index()?;
        Some(self.deliver_index(index, observed))
    }

    pub fn deliver_next_with(
        &mut self,
        observe: impl FnOnce(&BoundaryEvent) -> Value,
    ) -> Option<DeliveredBoundary> {
        let index = self.next_index()?;
        let observed = observe(&self.pending[index]);
        Some(self.deliver_index(index, observed))
    }

    pub fn deliver_boundary(
        &mut self,
        boundary_id: &str,
        observed: Value,
    ) -> Option<DeliveredBoundary> {
        let index = self
            .pending
            .iter()
            .position(|event| event.boundary_id == boundary_id)?;
        Some(self.deliver_index(index, observed))
    }

    pub fn deliver_boundary_with(
        &mut self,
        boundary_id: &str,
        observe: impl FnOnce(&BoundaryEvent) -> Value,
    ) -> Option<DeliveredBoundary> {
        let index = self
            .pending
            .iter()
            .position(|event| event.boundary_id == boundary_id)?;
        let observed = observe(&self.pending[index]);
        Some(self.deliver_index(index, observed))
    }

    fn deliver_index(&mut self, index: usize, observed: Value) -> DeliveredBoundary {
        let event = self.pending.remove(index);
        let sequence = self.sequence;
        self.sequence += 1;
        DeliveredBoundary::from_event(sequence, event, observed)
    }

    fn next_index(&mut self) -> Option<usize> {
        let min_at = self.pending.iter().map(|event| event.at).min()?;
        let mut candidates = self
            .pending
            .iter()
            .enumerate()
            .filter(|(_, event)| event.at == min_at)
            .collect::<Vec<_>>();
        candidates.sort_by(|(_, left), (_, right)| {
            left.actor_alias
                .cmp(&right.actor_alias)
                .then_with(|| left.kind_name().cmp(right.kind_name()))
                .then_with(|| left.boundary_id.cmp(&right.boundary_id))
        });
        let choice = if candidates.len() == 1 {
            0
        } else {
            (next_seed(&mut self.seed) as usize) % candidates.len()
        };
        Some(candidates[choice].0)
    }
}

impl BoundaryEvent {
    fn kind_name(&self) -> &'static str {
        match self.kind {
            BoundaryKind::Ingress => "ingress",
            BoundaryKind::Provider => "provider",
            BoundaryKind::Tool => "tool",
            BoundaryKind::DurableEffect => "durable_effect",
            BoundaryKind::Worker => "worker",
            BoundaryKind::Observer => "observer",
            BoundaryKind::LeaseTime => "lease_time",
        }
    }
}

pub(crate) fn next_seed(seed: &mut u64) -> u64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *seed
}

#[derive(Clone, Debug, Default)]
pub struct BoundaryDeliveryLog {
    events: VecDeque<DeliveredBoundary>,
}

impl BoundaryDeliveryLog {
    pub fn push(&mut self, event: DeliveredBoundary) {
        self.events.push_back(event);
    }

    pub fn into_vec(self) -> Vec<DeliveredBoundary> {
        self.events.into_iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn scheduler_delivers_seeded_boundaries_without_polling_futures() {
        let events = [
            BoundaryEvent::new(
                "ingress-a",
                "session-a",
                BoundaryKind::Ingress,
                9,
                "session.open",
                json!({}),
            ),
            BoundaryEvent::new(
                "provider-a",
                "session-a",
                BoundaryKind::Provider,
                10,
                "p",
                json!({}),
            ),
            BoundaryEvent::new(
                "tool-a",
                "session-a",
                BoundaryKind::Tool,
                10,
                "t",
                json!({}),
            ),
            BoundaryEvent::new(
                "lease-time-b",
                "session-b",
                BoundaryKind::LeaseTime,
                11,
                "lease.clock",
                json!({}),
            ),
        ];
        let mut first = BoundaryScheduler::with_events(7, events.clone());
        let mut second = BoundaryScheduler::with_events(7, events);

        let first_ids = drain_ids(&mut first);
        let second_ids = drain_ids(&mut second);

        assert_eq!(first_ids, second_ids);
        assert_eq!(first_ids.len(), 4);
        assert_eq!(first_ids.first().map(String::as_str), Some("ingress-a"));
        assert_eq!(first_ids.last().map(String::as_str), Some("lease-time-b"));
    }

    fn drain_ids(scheduler: &mut BoundaryScheduler) -> Vec<String> {
        let mut ids = Vec::new();
        while let Some(event) = scheduler.deliver_next(json!({"ok": true})) {
            ids.push(event.boundary_id);
        }
        ids
    }
}
