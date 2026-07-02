mod file_store;

pub use file_store::FileAttachmentStore;

use std::collections::{BTreeSet, HashMap};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use lash_sansio::{AttachmentCreateMeta, AttachmentId, AttachmentMeta, AttachmentRef};
use sha2::{Digest, Sha256};

use crate::store::{AttachmentIntent, AttachmentManifest};

#[derive(Debug, thiserror::Error)]
pub enum AttachmentStoreError {
    #[error("attachment `{0}` was not found")]
    NotFound(AttachmentId),
    #[error("attachment store I/O failed at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("attachment store metadata is unavailable for `{0}`")]
    MissingMeta(AttachmentId),
    #[error("attachment store metadata decode failed for `{id}`: {source}")]
    MetadataDecode {
        id: AttachmentId,
        #[source]
        source: serde_json::Error,
    },
    #[error("attachment manifest write failed: {0}")]
    ManifestRecordFailed(String),
    #[error("attachment store backend failed: {0}")]
    Backend(String),
}

#[derive(Clone, Debug)]
pub struct StoredAttachment {
    pub meta: AttachmentMeta,
    pub bytes: Vec<u8>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttachmentStorePersistence {
    Ephemeral,
    Durable,
}

impl AttachmentStorePersistence {
    /// Map the attachment-store persistence signal onto the shared
    /// [`DurabilityTier`](crate::DurabilityTier): `Ephemeral -> Inline`,
    /// `Durable -> Durable`. Lets consistency checks read every wired store's
    /// tier uniformly without a separate `durability_tier()` method here.
    pub fn durability_tier(self) -> crate::DurabilityTier {
        match self {
            Self::Ephemeral => crate::DurabilityTier::Inline,
            Self::Durable => crate::DurabilityTier::Durable,
        }
    }
}

#[async_trait::async_trait]
pub trait AttachmentStore: Send + Sync {
    fn persistence(&self) -> AttachmentStorePersistence {
        AttachmentStorePersistence::Ephemeral
    }

    /// Attachment refs written by this store that still need their
    /// write-ahead manifest rows stamped by the next runtime commit.
    ///
    /// Plain stores return an empty set. [`SessionScopedAttachmentStore`]
    /// overrides this so attachments created through downstream tools,
    /// process execution, and other runtime services are committed by the
    /// same final turn transaction that makes them reachable from session
    /// state.
    fn pending_manifest_commit_ids(&self) -> Vec<AttachmentId> {
        Vec::new()
    }

    /// Clear attachment refs that were stamped committed by a successful
    /// runtime commit.
    fn mark_manifest_committed(&self, _ids: &[AttachmentId]) {}

    async fn put(
        &self,
        bytes: Vec<u8>,
        meta: AttachmentCreateMeta,
    ) -> Result<AttachmentRef, AttachmentStoreError>;

    async fn get(&self, id: &AttachmentId) -> Result<StoredAttachment, AttachmentStoreError>;
}

#[derive(Default)]
pub struct InMemoryAttachmentStore {
    attachments: Mutex<HashMap<AttachmentId, StoredAttachment>>,
}

impl InMemoryAttachmentStore {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait::async_trait]
impl AttachmentStore for InMemoryAttachmentStore {
    async fn put(
        &self,
        bytes: Vec<u8>,
        meta: AttachmentCreateMeta,
    ) -> Result<AttachmentRef, AttachmentStoreError> {
        let meta = stored_meta(&bytes, meta);
        let reference = meta.as_ref();
        let stored = StoredAttachment { meta, bytes };
        self.attachments
            .lock()
            .expect("attachment store lock")
            .insert(reference.id.clone(), stored);
        Ok(reference)
    }

    async fn get(&self, id: &AttachmentId) -> Result<StoredAttachment, AttachmentStoreError> {
        self.attachments
            .lock()
            .expect("attachment store lock")
            .get(id)
            .cloned()
            .ok_or_else(|| AttachmentStoreError::NotFound(id.clone()))
    }
}

pub fn content_id(bytes: &[u8]) -> AttachmentId {
    AttachmentId::new(format!("{:x}", Sha256::digest(bytes)))
}

/// Session-scoped wrapper that records a write-ahead intent in
/// [`AttachmentManifest`] before delegating each `put` to the backing
/// [`AttachmentStore`]. The intent row durably captures "this session
/// is about to write these bytes," so if the process dies between
/// `put` and the next committed runtime state, a later GC sweep can
/// reconcile the orphaned bytes by walking
/// [`AttachmentManifest::list_uncommitted`].
///
/// Constructed by the runtime when both a durable [`AttachmentStore`]
/// and a [`RuntimePersistence`](crate::RuntimePersistence) backend
/// (which also implements [`AttachmentManifest`]) are wired up. Other
/// callers — tests, hosts using only ephemeral storage — keep the
/// plain inner store and skip the manifest entirely.
pub struct SessionScopedAttachmentStore {
    inner: Arc<dyn AttachmentStore>,
    manifest: Arc<dyn AttachmentManifest>,
    session_id: String,
    pending_manifest_commit_ids: Mutex<BTreeSet<AttachmentId>>,
}

impl SessionScopedAttachmentStore {
    pub fn new(
        inner: Arc<dyn AttachmentStore>,
        manifest: Arc<dyn AttachmentManifest>,
        session_id: impl Into<String>,
    ) -> Self {
        Self {
            inner,
            manifest,
            session_id: session_id.into(),
            pending_manifest_commit_ids: Mutex::new(BTreeSet::new()),
        }
    }

    pub fn inner(&self) -> &Arc<dyn AttachmentStore> {
        &self.inner
    }

    pub fn manifest(&self) -> &Arc<dyn AttachmentManifest> {
        &self.manifest
    }
}

#[async_trait::async_trait]
impl AttachmentStore for SessionScopedAttachmentStore {
    fn persistence(&self) -> AttachmentStorePersistence {
        self.inner.persistence()
    }

    fn pending_manifest_commit_ids(&self) -> Vec<AttachmentId> {
        self.pending_manifest_commit_ids
            .lock()
            .expect("attachment manifest commit tracker lock")
            .iter()
            .cloned()
            .collect()
    }

    fn mark_manifest_committed(&self, ids: &[AttachmentId]) {
        if ids.is_empty() {
            return;
        }
        let mut pending = self
            .pending_manifest_commit_ids
            .lock()
            .expect("attachment manifest commit tracker lock");
        for id in ids {
            pending.remove(id);
        }
    }

    async fn put(
        &self,
        bytes: Vec<u8>,
        meta: AttachmentCreateMeta,
    ) -> Result<AttachmentRef, AttachmentStoreError> {
        let attachment_id = content_id(&bytes);
        let intent = AttachmentIntent {
            attachment_id: attachment_id.clone(),
            session_id: self.session_id.clone(),
            canonical_uri: format!("sha256:{attachment_id}"),
            intent_at_epoch_ms: now_epoch_ms(),
        };
        // Record intent first. If this fails the bytes never land,
        // matching the write-ahead guarantee.
        self.manifest.record_intent(intent).map_err(|err| {
            AttachmentStoreError::ManifestRecordFailed(format!(
                "failed to record attachment intent for `{attachment_id}`: {err}"
            ))
        })?;
        let reference = self.inner.put(bytes, meta).await?;
        if reference.id != attachment_id {
            return Err(AttachmentStoreError::Backend(format!(
                "attachment store returned id `{}` after manifest intent for `{attachment_id}`",
                reference.id
            )));
        }
        self.pending_manifest_commit_ids
            .lock()
            .expect("attachment manifest commit tracker lock")
            .insert(reference.id.clone());
        Ok(reference)
    }

    async fn get(&self, id: &AttachmentId) -> Result<StoredAttachment, AttachmentStoreError> {
        self.inner.get(id).await
    }
}

fn now_epoch_ms() -> u64 {
    <crate::SystemClock as crate::Clock>::timestamp_ms(&crate::SystemClock)
}

/// Adapter that exposes the [`AttachmentManifest`] supertrait of an
/// `Arc<dyn RuntimePersistence>` as an `Arc<dyn AttachmentManifest>`.
/// Rust's trait-object upcasting does not yet allow direct coercion
/// between the two; this thin forwarder is the bridge.
pub(crate) struct PersistenceManifestAdapter(pub Arc<dyn crate::RuntimePersistence>);

impl AttachmentManifest for PersistenceManifestAdapter {
    fn record_intent(&self, intent: AttachmentIntent) -> Result<(), crate::StoreError> {
        AttachmentManifest::record_intent(&*self.0, intent)
    }

    fn commit_refs(
        &self,
        session_id: &str,
        attachment_ids: &[AttachmentId],
    ) -> Result<(), crate::StoreError> {
        AttachmentManifest::commit_refs(&*self.0, session_id, attachment_ids)
    }

    fn list_uncommitted(
        &self,
        older_than_epoch_ms: u64,
    ) -> Result<Vec<crate::AttachmentManifestEntry>, crate::StoreError> {
        AttachmentManifest::list_uncommitted(&*self.0, older_than_epoch_ms)
    }

    fn forget(&self, attachment_id: &AttachmentId) -> Result<(), crate::StoreError> {
        AttachmentManifest::forget(&*self.0, attachment_id)
    }
}

fn stored_meta(bytes: &[u8], meta: AttachmentCreateMeta) -> AttachmentMeta {
    AttachmentMeta::new(
        content_id(bytes),
        meta.media_type,
        bytes.len() as u64,
        meta.width,
        meta.height,
        meta.label,
    )
}

pub async fn resolve_llm_request_attachments(
    mut request: crate::llm::types::LlmRequest,
    store: &dyn AttachmentStore,
) -> Result<crate::llm::types::LlmRequest, AttachmentStoreError> {
    for attachment in &mut request.attachments {
        let Some(reference) = attachment.reference.as_ref() else {
            continue;
        };
        if !attachment.data.is_empty() {
            continue;
        }
        let stored = store.get(&reference.id).await?;
        attachment.mime = stored.meta.media_type.canonical_mime().to_string();
        attachment.data = stored.bytes;
    }
    Ok(request)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lash_sansio::{ImageMediaType, MediaType};
    use std::collections::HashSet;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[derive(Default)]
    struct RecordingManifest {
        intents: Mutex<Vec<AttachmentIntent>>,
        committed: Mutex<Vec<(String, AttachmentId)>>,
    }

    impl AttachmentManifest for RecordingManifest {
        fn record_intent(&self, intent: AttachmentIntent) -> Result<(), crate::StoreError> {
            self.intents.lock().expect("lock intents").push(intent);
            Ok(())
        }

        fn commit_refs(
            &self,
            session_id: &str,
            attachment_ids: &[AttachmentId],
        ) -> Result<(), crate::StoreError> {
            let mut committed = self.committed.lock().expect("lock committed attachments");
            committed.extend(
                attachment_ids
                    .iter()
                    .cloned()
                    .map(|attachment_id| (session_id.to_string(), attachment_id)),
            );
            Ok(())
        }

        fn list_uncommitted(
            &self,
            older_than_epoch_ms: u64,
        ) -> Result<Vec<crate::AttachmentManifestEntry>, crate::StoreError> {
            let committed = self
                .committed
                .lock()
                .expect("lock committed attachments")
                .iter()
                .cloned()
                .collect::<HashSet<_>>();
            Ok(self
                .intents
                .lock()
                .expect("lock intents")
                .iter()
                .filter(|intent| intent.intent_at_epoch_ms <= older_than_epoch_ms)
                .filter(|intent| {
                    !committed.contains(&(intent.session_id.clone(), intent.attachment_id.clone()))
                })
                .map(|intent| crate::AttachmentManifestEntry {
                    attachment_id: intent.attachment_id.clone(),
                    session_id: intent.session_id.clone(),
                    canonical_uri: intent.canonical_uri.clone(),
                    intent_at_epoch_ms: intent.intent_at_epoch_ms,
                    committed_at_epoch_ms: None,
                })
                .collect())
        }

        fn forget(&self, _attachment_id: &AttachmentId) -> Result<(), crate::StoreError> {
            Ok(())
        }
    }

    #[derive(Default)]
    struct RecordingRuntimePersistence {
        inner: crate::InMemorySessionStore,
        manifest: RecordingManifest,
    }

    impl AttachmentManifest for RecordingRuntimePersistence {
        fn record_intent(&self, intent: AttachmentIntent) -> Result<(), crate::StoreError> {
            self.manifest.record_intent(intent)
        }

        fn commit_refs(
            &self,
            session_id: &str,
            attachment_ids: &[AttachmentId],
        ) -> Result<(), crate::StoreError> {
            self.manifest.commit_refs(session_id, attachment_ids)
        }

        fn list_uncommitted(
            &self,
            older_than_epoch_ms: u64,
        ) -> Result<Vec<crate::AttachmentManifestEntry>, crate::StoreError> {
            self.manifest.list_uncommitted(older_than_epoch_ms)
        }

        fn forget(&self, attachment_id: &AttachmentId) -> Result<(), crate::StoreError> {
            self.manifest.forget(attachment_id)
        }
    }

    // Pass-through wrapper: every persistence segment delegates to the inner
    // in-memory store; only the attachment manifest is replaced with the
    // recording double above.
    #[async_trait::async_trait]
    impl crate::SessionCommitStore for RecordingRuntimePersistence {
        async fn load_session(
            &self,
            scope: crate::SessionReadScope,
        ) -> Result<Option<crate::PersistedSessionRead>, crate::StoreError> {
            crate::SessionCommitStore::load_session(&self.inner, scope).await
        }

        async fn load_node(
            &self,
            node_id: &str,
        ) -> Result<Option<crate::SessionNodeRecord>, crate::StoreError> {
            crate::SessionCommitStore::load_node(&self.inner, node_id).await
        }

        async fn commit_runtime_state(
            &self,
            commit: crate::RuntimeCommit,
        ) -> Result<crate::RuntimeCommitResult, crate::StoreError> {
            crate::SessionCommitStore::commit_runtime_state(&self.inner, commit).await
        }

        async fn save_session_meta(
            &self,
            meta: crate::SessionMeta,
        ) -> Result<(), crate::StoreError> {
            crate::SessionCommitStore::save_session_meta(&self.inner, meta).await
        }

        async fn load_session_meta(&self) -> Result<Option<crate::SessionMeta>, crate::StoreError> {
            crate::SessionCommitStore::load_session_meta(&self.inner).await
        }
    }

    #[async_trait::async_trait]
    impl crate::SessionExecutionLeaseStore for RecordingRuntimePersistence {
        async fn try_claim_session_execution_lease(
            &self,
            session_id: &str,
            owner: &crate::LeaseOwnerIdentity,
            lease_ttl_ms: u64,
        ) -> Result<crate::SessionExecutionLeaseClaimOutcome, crate::StoreError> {
            crate::SessionExecutionLeaseStore::try_claim_session_execution_lease(
                &self.inner,
                session_id,
                owner,
                lease_ttl_ms,
            )
            .await
        }

        async fn reclaim_session_execution_lease(
            &self,
            session_id: &str,
            owner: &crate::LeaseOwnerIdentity,
            observed_holder: &crate::SessionExecutionLeaseFence,
            lease_ttl_ms: u64,
        ) -> Result<crate::SessionExecutionLeaseClaimOutcome, crate::StoreError> {
            crate::SessionExecutionLeaseStore::reclaim_session_execution_lease(
                &self.inner,
                session_id,
                owner,
                observed_holder,
                lease_ttl_ms,
            )
            .await
        }

        async fn renew_session_execution_lease(
            &self,
            fence: &crate::SessionExecutionLeaseFence,
            lease_ttl_ms: u64,
        ) -> Result<crate::SessionExecutionLease, crate::StoreError> {
            crate::SessionExecutionLeaseStore::renew_session_execution_lease(
                &self.inner,
                fence,
                lease_ttl_ms,
            )
            .await
        }

        async fn release_session_execution_lease(
            &self,
            completion: &crate::SessionExecutionLeaseCompletion,
        ) -> Result<(), crate::StoreError> {
            crate::SessionExecutionLeaseStore::release_session_execution_lease(
                &self.inner,
                completion,
            )
            .await
        }
    }

    #[async_trait::async_trait]
    impl crate::TurnInputStore for RecordingRuntimePersistence {
        async fn enqueue_pending_turn_input(
            &self,
            input: crate::PendingTurnInputDraft,
        ) -> Result<crate::PendingTurnInput, crate::StoreError> {
            crate::TurnInputStore::enqueue_pending_turn_input(&self.inner, input).await
        }

        async fn list_pending_turn_inputs(
            &self,
            session_id: &str,
        ) -> Result<Vec<crate::PendingTurnInput>, crate::StoreError> {
            crate::TurnInputStore::list_pending_turn_inputs(&self.inner, session_id).await
        }

        async fn cancel_pending_turn_inputs(
            &self,
            session_id: &str,
            targets: &[crate::PendingTurnInputCancelTarget],
        ) -> Result<Vec<crate::PendingTurnInputCancelResult>, crate::StoreError> {
            crate::TurnInputStore::cancel_pending_turn_inputs(&self.inner, session_id, targets)
                .await
        }

        async fn cancel_pending_turn_input_suffix(
            &self,
            session_id: &str,
            anchor: &crate::PendingTurnInputCancelTarget,
        ) -> Result<crate::PendingTurnInputSuffixCancelOutcome, crate::StoreError> {
            crate::TurnInputStore::cancel_pending_turn_input_suffix(&self.inner, session_id, anchor)
                .await
        }

        async fn claim_active_turn_inputs(
            &self,
            session_id: &str,
            session_execution_lease: &crate::SessionExecutionLeaseFence,
            owner: &crate::LeaseOwnerIdentity,
            turn_id: &str,
            checkpoint: crate::CheckpointKind,
            lease_ttl_ms: u64,
            max_inputs: usize,
        ) -> Result<Option<crate::TurnInputClaim>, crate::StoreError> {
            crate::TurnInputStore::claim_active_turn_inputs(
                &self.inner,
                session_id,
                session_execution_lease,
                owner,
                turn_id,
                checkpoint,
                lease_ttl_ms,
                max_inputs,
            )
            .await
        }

        async fn claim_next_turn_inputs(
            &self,
            session_id: &str,
            session_execution_lease: &crate::SessionExecutionLeaseFence,
            owner: &crate::LeaseOwnerIdentity,
            lease_ttl_ms: u64,
            max_inputs: usize,
        ) -> Result<Option<crate::TurnInputClaim>, crate::StoreError> {
            crate::TurnInputStore::claim_next_turn_inputs(
                &self.inner,
                session_id,
                session_execution_lease,
                owner,
                lease_ttl_ms,
                max_inputs,
            )
            .await
        }

        async fn abandon_turn_input_claim(
            &self,
            claim: &crate::TurnInputClaim,
        ) -> Result<(), crate::StoreError> {
            crate::TurnInputStore::abandon_turn_input_claim(&self.inner, claim).await
        }
    }

    #[async_trait::async_trait]
    impl crate::QueuedWorkStore for RecordingRuntimePersistence {
        async fn enqueue_queued_work(
            &self,
            batch: crate::QueuedWorkBatchDraft,
        ) -> Result<crate::QueuedWorkBatch, crate::StoreError> {
            crate::QueuedWorkStore::enqueue_queued_work(&self.inner, batch).await
        }

        async fn claim_leading_ready_session_command(
            &self,
            session_id: &str,
            session_execution_lease: &crate::SessionExecutionLeaseFence,
            owner: &crate::LeaseOwnerIdentity,
            lease_ttl_ms: u64,
        ) -> Result<Option<crate::QueuedWorkClaim>, crate::StoreError> {
            crate::QueuedWorkStore::claim_leading_ready_session_command(
                &self.inner,
                session_id,
                session_execution_lease,
                owner,
                lease_ttl_ms,
            )
            .await
        }

        async fn claim_ready_queued_work(
            &self,
            session_id: &str,
            session_execution_lease: &crate::SessionExecutionLeaseFence,
            owner: &crate::LeaseOwnerIdentity,
            boundary: crate::QueuedWorkClaimBoundary,
            lease_ttl_ms: u64,
            max_batches: usize,
        ) -> Result<Option<crate::QueuedWorkClaim>, crate::StoreError> {
            crate::QueuedWorkStore::claim_ready_queued_work(
                &self.inner,
                session_id,
                session_execution_lease,
                owner,
                boundary,
                lease_ttl_ms,
                max_batches,
            )
            .await
        }

        async fn renew_queued_work_claim(
            &self,
            claim: &crate::QueuedWorkClaim,
            lease_ttl_ms: u64,
        ) -> Result<crate::QueuedWorkClaim, crate::StoreError> {
            crate::QueuedWorkStore::renew_queued_work_claim(&self.inner, claim, lease_ttl_ms).await
        }

        async fn abandon_queued_work_claim(
            &self,
            claim: &crate::QueuedWorkClaim,
        ) -> Result<(), crate::StoreError> {
            crate::QueuedWorkStore::abandon_queued_work_claim(&self.inner, claim).await
        }

        async fn cancel_queued_work_batch(
            &self,
            session_id: &str,
            batch_id: &str,
        ) -> Result<Option<crate::QueuedWorkBatch>, crate::StoreError> {
            crate::QueuedWorkStore::cancel_queued_work_batch(&self.inner, session_id, batch_id)
                .await
        }

        async fn list_queued_work(
            &self,
            session_id: &str,
        ) -> Result<Vec<crate::QueuedWorkBatch>, crate::StoreError> {
            crate::QueuedWorkStore::list_queued_work(&self.inner, session_id).await
        }

        async fn list_pending_queued_work(
            &self,
            session_id: &str,
        ) -> Result<Vec<crate::QueuedWorkBatch>, crate::StoreError> {
            crate::QueuedWorkStore::list_pending_queued_work(&self.inner, session_id).await
        }
    }

    #[async_trait::async_trait]
    impl crate::StoreMaintenance for RecordingRuntimePersistence {
        async fn tombstone_nodes(&self, ids: &[String]) -> Result<(), crate::StoreError> {
            crate::StoreMaintenance::tombstone_nodes(&self.inner, ids).await
        }

        async fn vacuum(&self) -> Result<crate::VacuumReport, crate::StoreError> {
            crate::StoreMaintenance::vacuum(&self.inner).await
        }

        async fn gc_unreachable(&self) -> Result<crate::GcReport, crate::StoreError> {
            crate::StoreMaintenance::gc_unreachable(&self.inner).await
        }
    }

    fn meta() -> AttachmentCreateMeta {
        AttachmentCreateMeta::new(
            MediaType::Image(ImageMediaType::Png),
            Some(1),
            Some(1),
            Some("pixel".to_string()),
        )
    }

    fn system_epoch_ms_for_test() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock must be after Unix epoch")
            .as_millis() as u64
    }

    #[tokio::test]
    async fn memory_store_dedupes_by_bytes() {
        let store = InMemoryAttachmentStore::new();
        let a = store.put(vec![1, 2, 3], meta()).await.expect("put a");
        let b = store.put(vec![1, 2, 3], meta()).await.expect("put b");
        assert_eq!(a.id, b.id);
        assert_eq!(a.byte_len, 3);
        assert_eq!(store.get(&a.id).await.expect("get").bytes, vec![1, 2, 3]);
    }

    #[tokio::test]
    async fn memory_store_assigns_identity_and_byte_len_from_bytes() {
        let store = InMemoryAttachmentStore::new();
        let reference = store.put(vec![4, 5, 6, 7], meta()).await.expect("put");

        assert_eq!(reference.id, content_id(&[4, 5, 6, 7]));
        assert_eq!(reference.byte_len, 4);
    }

    #[tokio::test]
    async fn session_scoped_attachment_store_satisfies_conformance() {
        crate::testing::conformance::attachment_store(
            || {
                let manifest: Arc<dyn AttachmentManifest> = Arc::new(RecordingManifest::default());
                Arc::new(SessionScopedAttachmentStore::new(
                    Arc::new(InMemoryAttachmentStore::new()),
                    manifest,
                    "session-scoped-conformance",
                )) as Arc<dyn AttachmentStore>
            },
            AttachmentStorePersistence::Ephemeral,
        )
        .await;
    }

    #[tokio::test]
    async fn session_scoped_store_tracks_successful_puts_until_commit_mark() {
        let manifest = Arc::new(RecordingManifest::default());
        let manifest_for_store: Arc<dyn AttachmentManifest> = manifest.clone();
        let store = SessionScopedAttachmentStore::new(
            Arc::new(InMemoryAttachmentStore::new()),
            manifest_for_store,
            "session-1",
        );

        let reference = store.put(vec![8, 9, 10], meta()).await.expect("put");

        assert_eq!(
            manifest.intents.lock().expect("lock intents")[0].attachment_id,
            reference.id
        );
        assert_eq!(
            store.pending_manifest_commit_ids(),
            vec![reference.id.clone()]
        );

        store.mark_manifest_committed(&[AttachmentId::new("other")]);
        assert_eq!(
            store.pending_manifest_commit_ids(),
            vec![reference.id.clone()]
        );

        store.mark_manifest_committed(std::slice::from_ref(&reference.id));
        assert!(store.pending_manifest_commit_ids().is_empty());
    }

    #[tokio::test]
    async fn session_scoped_store_records_intent_timestamp_from_system_clock() {
        let manifest = Arc::new(RecordingManifest::default());
        let manifest_for_store: Arc<dyn AttachmentManifest> = manifest.clone();
        let store = SessionScopedAttachmentStore::new(
            Arc::new(InMemoryAttachmentStore::new()),
            manifest_for_store,
            "session-clock",
        );

        let before_put_epoch_ms = system_epoch_ms_for_test();
        let reference = store.put(vec![11, 12, 13], meta()).await.expect("put");
        let after_put_epoch_ms = system_epoch_ms_for_test();

        let intents = manifest.intents.lock().expect("lock intents");
        assert_eq!(intents.len(), 1);
        let intent = &intents[0];
        assert_eq!(intent.attachment_id, reference.id);
        assert!(
            intent.intent_at_epoch_ms > 1_000_000_000_000,
            "intent timestamp should be a real epoch millis value, got {}",
            intent.intent_at_epoch_ms
        );
        assert!(
            intent.intent_at_epoch_ms >= before_put_epoch_ms.saturating_sub(1000),
            "intent timestamp {} should be close to or after put start {}",
            intent.intent_at_epoch_ms,
            before_put_epoch_ms
        );
        assert!(
            intent.intent_at_epoch_ms <= after_put_epoch_ms.saturating_add(1000),
            "intent timestamp {} should be close to or before put finish {}",
            intent.intent_at_epoch_ms,
            after_put_epoch_ms
        );
    }

    #[test]
    fn persistence_manifest_adapter_forwards_to_wrapped_runtime_persistence() {
        let runtime = Arc::new(RecordingRuntimePersistence::default());
        let persistence: Arc<dyn crate::RuntimePersistence> = runtime.clone();
        let adapter = PersistenceManifestAdapter(persistence);
        let attachment_id = AttachmentId::new("adapter-forwarding");
        let intent = AttachmentIntent {
            attachment_id: attachment_id.clone(),
            session_id: "adapter-session".to_string(),
            canonical_uri: "sha256:adapter-forwarding".to_string(),
            intent_at_epoch_ms: 10,
        };

        adapter.record_intent(intent).expect("record intent");
        let uncommitted = adapter
            .list_uncommitted(10)
            .expect("list uncommitted through adapter");
        assert_eq!(uncommitted.len(), 1);
        assert_eq!(uncommitted[0].attachment_id, attachment_id);
        assert_eq!(uncommitted[0].session_id, "adapter-session");
        assert_eq!(uncommitted[0].canonical_uri, "sha256:adapter-forwarding");
        assert_eq!(uncommitted[0].intent_at_epoch_ms, 10);
        assert!(uncommitted[0].committed_at_epoch_ms.is_none());

        adapter
            .commit_refs("adapter-session", std::slice::from_ref(&attachment_id))
            .expect("commit refs through adapter");
        assert!(
            adapter
                .list_uncommitted(10)
                .expect("list after commit through adapter")
                .is_empty()
        );
        assert_eq!(
            runtime
                .manifest
                .committed
                .lock()
                .expect("lock committed attachments")
                .as_slice(),
            &[("adapter-session".to_string(), attachment_id)]
        );
    }
}
