use std::collections::BTreeMap;

use lash_core::{
    LeaseOwnerIdentity, SessionExecutionLease, SessionExecutionLeaseCompletion,
    SessionExecutionLeaseFence,
};
use serde_json::{Value, json};

#[derive(Clone, Debug, Default)]
pub struct SimWorkerTopology {
    workers: BTreeMap<String, WorkerState>,
    active_session_leases: BTreeMap<String, SessionExecutionLease>,
    next_fencing_token: u64,
}

impl SimWorkerTopology {
    pub fn run_stale_completion_script(
        &mut self,
        worker_alias: impl Into<String>,
        session_alias: impl Into<String>,
    ) -> Value {
        let worker_alias = worker_alias.into();
        let session_alias = session_alias.into();
        let stale_owner =
            LeaseOwnerIdentity::opaque(&worker_alias, format!("{worker_alias}:incarnation-001"));
        let live_owner =
            LeaseOwnerIdentity::opaque(&worker_alias, format!("{worker_alias}:incarnation-002"));
        let stale_lease = self.claim_session_lease(&session_alias, stale_owner);
        let live_lease = self.claim_session_lease(&session_alias, live_owner.clone());
        let stale_completion = stale_lease.completion();
        let stale_accepted = self.complete_session_lease(&stale_completion);
        let state = self
            .workers
            .entry(worker_alias.clone())
            .or_insert_with(|| WorkerState::new(worker_alias.clone()));
        state.incarnation_id = live_owner.incarnation_id.clone();
        state.lease_owner_changes += 1;
        if stale_accepted {
            state.accepted_completions += 1;
        } else {
            state.stale_completion_rejections += 1;
        }
        json!({
            "worker_alias": worker_alias,
            "session": session_alias,
            "initial_owner": owner_json(&stale_lease.owner),
            "active_owner": owner_json(&live_lease.owner),
            "active_fencing_token": live_lease.fencing_token,
            "stale_completion_rejected": !stale_accepted,
            "lease_owner_changed": !stale_lease.owner.same_incarnation(&live_lease.owner),
        })
    }

    fn claim_session_lease(
        &mut self,
        session_alias: &str,
        owner: LeaseOwnerIdentity,
    ) -> SessionExecutionLease {
        self.next_fencing_token += 1;
        let lease = SessionExecutionLease {
            session_id: session_alias.to_string(),
            lease_token: format!("lease-{}", self.next_fencing_token),
            owner,
            fencing_token: self.next_fencing_token,
            claimed_at_epoch_ms: self.next_fencing_token * 1000,
            expires_at_epoch_ms: self.next_fencing_token * 1000 + 30_000,
        };
        self.active_session_leases
            .insert(session_alias.to_string(), lease.clone());
        lease
    }

    fn complete_session_lease(&mut self, completion: &SessionExecutionLeaseCompletion) -> bool {
        let Some(active) = self.active_session_leases.get(&completion.session_id) else {
            return false;
        };
        if completion_matches(active, completion) {
            self.active_session_leases.remove(&completion.session_id);
            true
        } else {
            false
        }
    }
}

#[derive(Clone, Debug)]
struct WorkerState {
    incarnation_id: String,
    lease_owner_changes: usize,
    stale_completion_rejections: usize,
    accepted_completions: usize,
}

impl WorkerState {
    fn new(worker_alias: String) -> Self {
        Self {
            incarnation_id: format!("{worker_alias}:incarnation-000"),
            lease_owner_changes: 0,
            stale_completion_rejections: 0,
            accepted_completions: 0,
        }
    }
}

fn completion_matches(
    active: &SessionExecutionLease,
    completion: &SessionExecutionLeaseCompletion,
) -> bool {
    fence_matches(&active.fence(), completion)
}

fn fence_matches(
    active: &SessionExecutionLeaseFence,
    completion: &SessionExecutionLeaseCompletion,
) -> bool {
    active.session_id == completion.session_id
        && active.lease_token == completion.lease_token
        && active.fencing_token == completion.fencing_token
        && active.owner.same_incarnation(&completion.owner)
}

fn owner_json(owner: &LeaseOwnerIdentity) -> Value {
    json!({
        "owner_id": owner.owner_id,
        "incarnation_id": owner.incarnation_id,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stale_completion_from_prior_worker_incarnation_is_rejected() {
        let mut topology = SimWorkerTopology::default();
        let observed = topology.run_stale_completion_script("worker-001", "session-001");

        assert_eq!(observed["stale_completion_rejected"], true);
        assert_eq!(observed["lease_owner_changed"], true);
        assert_eq!(
            observed["active_owner"]["incarnation_id"],
            "worker-001:incarnation-002"
        );
    }
}
