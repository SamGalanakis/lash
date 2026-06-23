//! Dialect-independent queued-work claim logic shared by durable backends.
//!
//! The SQL backends (sqlite, postgres) load candidate batch rows ordered by
//! `enqueue_seq` and pre-filtered to ready batches that are not held by a
//! live claim, then apply the same pure state machine: a delivery-policy
//! boundary gate, slot-policy/merge-key prefix grouping, and fencing-token /
//! lease derivation. That state machine lives here so the backends own only
//! their SQL reads and writes while the claim contract has a single
//! implementation, exercised against every backend by the shared
//! `runtime_persistence` conformance suite.

use sha2::{Digest, Sha256};

use super::LeaseOwnerIdentity;
use super::StoreError;
use crate::{
    DeliveryPolicy, MergeKey, QueuedWorkClaim, QueuedWorkClaimBoundary, QueuedWorkCompletion,
    SlotPolicy,
};

/// Decoded claim-relevant fields of one ready queued-work batch row.
///
/// Backends build these from their candidate rows, presented in
/// `enqueue_seq` ascending order and already filtered to
/// `available_at_ms <= now` with no live claim.
#[derive(Clone, Debug)]
pub struct ClaimCandidate {
    pub enqueue_seq: u64,
    pub claim_fencing_token: u64,
    pub delivery_policy: DeliveryPolicy,
    pub slot_policy: SlotPolicy,
    pub merge_key: MergeKey,
}

/// How many candidate rows a backend should scan when selecting up to
/// `max_batches` claimable batches. Joinable groups are matched as a prefix,
/// so scanning a bounded surplus keeps one round trip sufficient.
pub fn claim_scan_limit(max_batches: usize) -> i64 {
    (max_batches as i64).saturating_add(32)
}

/// Select the prefix of `candidates` that a single claim may take.
///
/// Returns the number of leading candidates to claim (`0` means no claim):
///
/// * An [`QueuedWorkClaimBoundary::ActiveTurnCheckpoint`] boundary only
///   admits work whose head batch is
///   [`DeliveryPolicy::EarliestSafeBoundary`].
/// * An [`SlotPolicy::Exclusive`] head claims exactly one batch.
/// * A [`SlotPolicy::Join`] head extends through immediately following
///   `Join` batches with the same delivery policy and merge key, up to
///   `max_batches`.
pub fn select_claim_prefix(
    candidates: &[ClaimCandidate],
    boundary: QueuedWorkClaimBoundary,
    max_batches: usize,
) -> usize {
    if max_batches == 0 {
        return 0;
    }
    let Some(first) = candidates.first() else {
        return 0;
    };
    if boundary == QueuedWorkClaimBoundary::ActiveTurnCheckpoint
        && first.delivery_policy != DeliveryPolicy::EarliestSafeBoundary
    {
        return 0;
    }
    if first.slot_policy != SlotPolicy::Join {
        return 1;
    }
    let mut selected = 1;
    for candidate in &candidates[1..] {
        if selected >= max_batches
            || candidate.slot_policy != SlotPolicy::Join
            || candidate.delivery_policy != first.delivery_policy
            || candidate.merge_key != first.merge_key
        {
            break;
        }
        selected += 1;
    }
    selected
}

/// A freshly derived lease for a selected claim prefix.
///
/// The fencing token advances past the head batch's last observed token, the
/// claim id is stable for (head batch, fencing token), and the lease token is
/// an opaque proof-of-ownership digest the backend stamps on every claimed
/// row.
#[derive(Clone, Debug)]
pub struct QueuedWorkClaimLease {
    pub claim_id: String,
    pub lease_token: String,
    pub fencing_token: u64,
    pub claimed_at_epoch_ms: u64,
    pub expires_at_epoch_ms: u64,
}

impl QueuedWorkClaimLease {
    pub fn derive(
        head: &ClaimCandidate,
        session_id: &str,
        owner: &LeaseOwnerIdentity,
        now_epoch_ms: u64,
        lease_ttl_ms: u64,
    ) -> Self {
        let fencing_token = head.claim_fencing_token.saturating_add(1);
        let claim_id = format!("qwc:{}:{fencing_token}", head.enqueue_seq);
        let lease_token = format!(
            "{:x}",
            Sha256::digest(
                format!(
                    "{}:{}:{}:{}:{}",
                    session_id, owner.owner_id, owner.incarnation_id, claim_id, now_epoch_ms
                )
                .as_bytes(),
            )
        );
        Self {
            claim_id,
            lease_token,
            fencing_token,
            claimed_at_epoch_ms: now_epoch_ms,
            expires_at_epoch_ms: now_epoch_ms.saturating_add(lease_ttl_ms),
        }
    }
}

/// Derive the durable id for a newly enqueued batch.
///
/// `nonce` disambiguates batches enqueued within the same millisecond;
/// backends whose id uniqueness already comes from elsewhere pass `None`.
pub fn derive_batch_id(
    session_id: &str,
    source_key: Option<&str>,
    now_epoch_ms: u64,
    nonce: Option<u64>,
) -> String {
    let mut seed = format!("{session_id}:{source_key:?}:{now_epoch_ms}");
    if let Some(nonce) = nonce {
        seed.push_str(&format!(":{nonce}"));
    }
    format!("qwb:{:x}", Sha256::digest(seed.as_bytes()))
}

/// Apply the shared lease-renewal decision: the lease holds only when every
/// batch row in the claim accepted the new expiry stamp.
pub fn renewed_claim(
    claim: &QueuedWorkClaim,
    renewed_rows: usize,
    expires_at_epoch_ms: u64,
) -> Result<QueuedWorkClaim, StoreError> {
    if renewed_rows != claim.batches.len() {
        return Err(StoreError::QueuedWorkClaimExpired {
            session_id: claim.session_id.clone(),
            claim_id: claim.claim_id.clone(),
        });
    }
    Ok(QueuedWorkClaim {
        expires_at_epoch_ms,
        ..claim.clone()
    })
}

/// Apply the shared completion-fencing decision: a completion may delete its
/// batches only when the live store still shows the claim owning every one
/// of them (`owned_rows` rows matched the claim id + lease token).
pub fn ensure_completion_owns_all_batches(
    completed: &QueuedWorkCompletion,
    owned_rows: usize,
) -> Result<(), StoreError> {
    if owned_rows != completed.batch_ids.len() {
        return Err(StoreError::QueuedWorkClaimExpired {
            session_id: completed.session_id.clone(),
            claim_id: completed.claim_id.clone(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(
        enqueue_seq: u64,
        delivery_policy: DeliveryPolicy,
        slot_policy: SlotPolicy,
        merge_key: MergeKey,
    ) -> ClaimCandidate {
        ClaimCandidate {
            enqueue_seq,
            claim_fencing_token: 0,
            delivery_policy,
            slot_policy,
            merge_key,
        }
    }

    #[test]
    fn exclusive_head_claims_exactly_one() {
        let candidates = vec![
            candidate(
                1,
                DeliveryPolicy::EarliestSafeBoundary,
                SlotPolicy::Exclusive,
                MergeKey::Never,
            ),
            candidate(
                2,
                DeliveryPolicy::EarliestSafeBoundary,
                SlotPolicy::Exclusive,
                MergeKey::Never,
            ),
        ];
        assert_eq!(
            select_claim_prefix(&candidates, QueuedWorkClaimBoundary::Idle, 8),
            1
        );
    }

    #[test]
    fn join_head_groups_matching_prefix_up_to_max() {
        let join = |seq| {
            candidate(
                seq,
                DeliveryPolicy::EarliestSafeBoundary,
                SlotPolicy::Join,
                MergeKey::PayloadDefault,
            )
        };
        let candidates = vec![join(1), join(2), join(3), join(4)];
        assert_eq!(
            select_claim_prefix(&candidates, QueuedWorkClaimBoundary::Idle, 3),
            3
        );
    }

    #[test]
    fn join_group_breaks_on_policy_or_merge_key_mismatch() {
        let candidates = vec![
            candidate(
                1,
                DeliveryPolicy::EarliestSafeBoundary,
                SlotPolicy::Join,
                MergeKey::Group("a".to_string()),
            ),
            candidate(
                2,
                DeliveryPolicy::EarliestSafeBoundary,
                SlotPolicy::Join,
                MergeKey::Group("b".to_string()),
            ),
        ];
        assert_eq!(
            select_claim_prefix(&candidates, QueuedWorkClaimBoundary::Idle, 8),
            1
        );
    }

    #[test]
    fn active_turn_checkpoint_boundary_gates_on_delivery_policy() {
        let candidates = vec![candidate(
            1,
            DeliveryPolicy::AfterCurrentTurnCommit,
            SlotPolicy::Exclusive,
            MergeKey::Never,
        )];
        assert_eq!(
            select_claim_prefix(
                &candidates,
                QueuedWorkClaimBoundary::ActiveTurnCheckpoint,
                8
            ),
            0
        );
    }

    #[test]
    fn lease_derivation_is_deterministic_and_advances_fencing() {
        let head = ClaimCandidate {
            enqueue_seq: 7,
            claim_fencing_token: 2,
            delivery_policy: DeliveryPolicy::EarliestSafeBoundary,
            slot_policy: SlotPolicy::Exclusive,
            merge_key: MergeKey::Never,
        };
        let owner = LeaseOwnerIdentity::opaque("owner", "owner:incarnation");
        let lease = QueuedWorkClaimLease::derive(&head, "session", &owner, 1_000, 250);
        assert_eq!(lease.fencing_token, 3);
        assert_eq!(lease.claim_id, "qwc:7:3");
        assert_eq!(lease.claimed_at_epoch_ms, 1_000);
        assert_eq!(lease.expires_at_epoch_ms, 1_250);
        let again = QueuedWorkClaimLease::derive(&head, "session", &owner, 1_000, 250);
        assert_eq!(lease.lease_token, again.lease_token);
    }

    #[test]
    fn batch_id_includes_optional_nonce() {
        let plain = derive_batch_id("session", Some("key"), 1_000, None);
        let nonced = derive_batch_id("session", Some("key"), 1_000, Some(1));
        assert_ne!(plain, nonced);
        assert!(plain.starts_with("qwb:"));
    }
}
