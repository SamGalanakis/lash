use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

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
    #[error("attachment manifest write failed: {0}")]
    ManifestRecordFailed(String),
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

pub trait AttachmentStore: Send + Sync {
    fn persistence(&self) -> AttachmentStorePersistence {
        AttachmentStorePersistence::Ephemeral
    }

    fn put(
        &self,
        bytes: Vec<u8>,
        meta: AttachmentCreateMeta,
    ) -> Result<AttachmentRef, AttachmentStoreError>;

    fn get(&self, id: &AttachmentId) -> Result<StoredAttachment, AttachmentStoreError>;
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

impl AttachmentStore for InMemoryAttachmentStore {
    fn put(
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

    fn get(&self, id: &AttachmentId) -> Result<StoredAttachment, AttachmentStoreError> {
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
/// `put` and the next durable turn commit, a later GC sweep can
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
        }
    }

    pub fn inner(&self) -> &Arc<dyn AttachmentStore> {
        &self.inner
    }

    pub fn manifest(&self) -> &Arc<dyn AttachmentManifest> {
        &self.manifest
    }
}

impl AttachmentStore for SessionScopedAttachmentStore {
    fn persistence(&self) -> AttachmentStorePersistence {
        self.inner.persistence()
    }

    fn put(
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
        self.inner.put(bytes, meta)
    }

    fn get(&self, id: &AttachmentId) -> Result<StoredAttachment, AttachmentStoreError> {
        self.inner.get(id)
    }
}

fn now_epoch_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
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

pub fn resolve_llm_request_attachments(
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
        let stored = store.get(&reference.id)?;
        attachment.mime = stored.meta.media_type.canonical_mime().to_string();
        attachment.data = stored.bytes;
    }
    Ok(request)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lash_sansio::{ImageMediaType, MediaType};

    fn meta() -> AttachmentCreateMeta {
        AttachmentCreateMeta::new(
            MediaType::Image(ImageMediaType::Png),
            Some(1),
            Some(1),
            Some("pixel".to_string()),
        )
    }

    #[test]
    fn memory_store_dedupes_by_bytes() {
        let store = InMemoryAttachmentStore::new();
        let a = store.put(vec![1, 2, 3], meta()).expect("put a");
        let b = store.put(vec![1, 2, 3], meta()).expect("put b");
        assert_eq!(a.id, b.id);
        assert_eq!(a.byte_len, 3);
        assert_eq!(store.get(&a.id).expect("get").bytes, vec![1, 2, 3]);
    }

    #[test]
    fn memory_store_assigns_identity_and_byte_len_from_bytes() {
        let store = InMemoryAttachmentStore::new();
        let reference = store.put(vec![4, 5, 6, 7], meta()).expect("put");

        assert_eq!(reference.id, content_id(&[4, 5, 6, 7]));
        assert_eq!(reference.byte_len, 4);
    }
}
