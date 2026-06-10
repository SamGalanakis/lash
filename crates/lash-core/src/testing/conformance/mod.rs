//! Backend-agnostic conformance suites for durable-backend traits.
//!
//! Each suite is parameterized over a factory that produces a *fresh* backend
//! instance and asserts the trait's contract invariants. Run the same suite
//! against every implementation (the production backend and any in-memory test
//! double) so the contract has one executable source of truth and the doubles
//! can't drift from production behavior.
//!
//! Suites panic on the first violated invariant — call them from a
//! `#[tokio::test]`. Embedders with custom backends can run them via
//! `lash::testing::conformance`.

mod attachment_store;
mod effect_host;
mod helpers;
mod host_event_store;
mod lashlang_artifact_store;
mod live_replay;
mod process_registry;
mod runtime_persistence;

pub use attachment_store::*;
pub use effect_host::*;
pub use helpers::*;
pub use host_event_store::*;
pub use lashlang_artifact_store::*;
pub use live_replay::*;
pub use process_registry::*;
pub use runtime_persistence::*;

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::{
    AgentFrameReason, AgentFrameRecord, AttachmentId, AttachmentIntent, CausalRef, DeliveryPolicy,
    EffectHost, EffectScope, LiveReplayGapReason, LiveReplayResult, LiveReplayStore,
    LiveReplayStoreError, LiveReplaySubscribeResult, MergeKey, ModelSpec, PluginSessionSnapshot,
    ProtocolEvent, ProtocolTurnOptions, QueuedWorkBatch, QueuedWorkBatchDraft,
    QueuedWorkClaimBoundary, QueuedWorkPayload, RuntimeCommit, RuntimeEffectCommand,
    RuntimeEffectController, RuntimeEffectControllerError, RuntimeEffectEnvelope,
    RuntimeEffectKind, RuntimeEffectLocalExecutor, RuntimeEffectOutcome, RuntimeInvocation,
    RuntimePersistence, RuntimeScope, RuntimeSessionState, RuntimeSubject, RuntimeTurnCommitStamp,
    ScopedEffectController, SessionMeta, SessionNodePayload, SessionNodeRecord,
    SessionObservationEvent, SessionObservationEventPayload, SessionPolicy,
    SessionProcessEventKind, SessionQueueEventKind, SessionReadScope, SessionRelation,
    SessionRevision, SlotPolicy, StoreError, TokenLedgerEntry, TokenUsage, ToolState, TurnActivity,
    TurnEvent, TurnInput,
};
use crate::{
    AttachmentStore, AttachmentStoreError, AttachmentStorePersistence, DurabilityTier,
    LashlangArtifactStore,
};
use crate::{
    LashSchema, ProcessAwaitOutput, ProcessEventAppendRequest, ProcessEventSemanticsSpec,
    ProcessEventType, ProcessExternalRef, ProcessHandleDescriptor, ProcessInput,
    ProcessLeaseCompletion, ProcessListFilter, ProcessProvenance, ProcessRecord,
    ProcessRegistration, ProcessRegistry, ProcessStatusFilter, ProcessTerminalState,
    ProcessValueSelector, ProcessWakeDedupeKey,
    ProcessWakeDelivery, ProcessWakeSpec, SessionScope, SessionScopeId, WaitKind, WaitState,
};
use lash_sansio::{AttachmentCreateMeta, ImageMediaType, MediaType};

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn in_memory_attachment_store_satisfies_conformance() {
        attachment_store(
            || Arc::new(crate::InMemoryAttachmentStore::new()) as Arc<dyn AttachmentStore>,
            AttachmentStorePersistence::Ephemeral,
        )
        .await;
    }

    #[tokio::test]
    async fn in_memory_lashlang_artifact_store_satisfies_conformance() {
        lashlang_artifact_store(
            || {
                Arc::new(crate::InMemoryLashlangArtifactStore::new())
                    as Arc<dyn LashlangArtifactStore>
            },
            DurabilityTier::Inline,
        )
        .await;
    }

    #[tokio::test]
    async fn in_memory_host_event_store_satisfies_conformance() {
        host_event_store(
            || Arc::new(crate::InMemoryHostEventStore::default()) as Arc<dyn crate::HostEventStore>,
            DurabilityTier::Inline,
        )
        .await;
    }

    #[tokio::test]
    async fn in_memory_live_replay_store_satisfies_conformance() {
        live_replay_store(|| {
            Arc::new(crate::InMemoryLiveReplayStore::default()) as Arc<dyn LiveReplayStore>
        })
        .await;
        live_replay_store_capacity_trim(|| {
            Arc::new(crate::InMemoryLiveReplayStore::with_bounds(
                1,
                Duration::from_secs(120),
            )) as Arc<dyn LiveReplayStore>
        })
        .await;
        live_replay_store_ttl_trim(
            || {
                Arc::new(crate::InMemoryLiveReplayStore::with_bounds(
                    16,
                    Duration::from_millis(1),
                )) as Arc<dyn LiveReplayStore>
            },
            Duration::from_millis(20),
        )
        .await;
    }

    #[tokio::test]
    async fn in_memory_process_registry_satisfies_conformance() {
        process_registry(|| {
            Arc::new(crate::TestLocalProcessRegistry::default()) as Arc<dyn ProcessRegistry>
        })
        .await;
    }

    #[tokio::test]
    async fn inline_effect_host_satisfies_conformance() {
        effect_host(|| Arc::new(crate::InlineEffectHost::default())).await;
    }

    #[tokio::test]
    async fn recording_effect_host_records_selected_scope_and_envelope() {
        let host = RecordingEffectHost::default();
        let scope = EffectScope::runtime_operation("host-event:button-1");
        let scoped = host.scoped(scope.clone()).expect("scoped controller");
        let envelope = RuntimeEffectEnvelope::new(
            crate::RuntimeInvocation::effect(
                RuntimeScope::new("session-1"),
                "sleep-effect",
                RuntimeEffectKind::Sleep,
                "host-event:button-1:sleep-effect",
            ),
            RuntimeEffectCommand::Sleep { duration_ms: 0 },
        );

        let outcome = scoped
            .controller()
            .execute_effect(envelope, RuntimeEffectLocalExecutor::unavailable())
            .await
            .expect("execute sleep");

        assert!(matches!(outcome, RuntimeEffectOutcome::Sleep));
        assert_eq!(host.selected_scopes(), vec![scope.clone()]);
        let records = host.records();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].effect_scope, scope);
        assert_eq!(records[0].runtime_scope, RuntimeScope::new("session-1"));
        assert_eq!(records[0].effect_id, "sleep-effect");
        assert_eq!(records[0].effect_kind, RuntimeEffectKind::Sleep);
        assert_eq!(
            records[0].replay_key.as_deref(),
            Some("host-event:button-1:sleep-effect")
        );
    }

    #[test]
    fn module_artifact_rejects_corrupted_store_bytes() {
        let err = lashlang::ModuleArtifact::from_store_bytes(b"not an artifact")
            .expect_err("corrupted artifact bytes must be rejected");
        assert!(matches!(err, lashlang::ModuleArtifactError::Codec(_)));
    }
}
