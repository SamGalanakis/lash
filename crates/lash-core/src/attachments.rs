use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use lash_sansio::{AttachmentCreateMeta, AttachmentId, AttachmentMeta, AttachmentRef};
use sha2::{Digest, Sha256};

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

pub struct FileAttachmentStore {
    root: PathBuf,
    meta: Mutex<HashMap<AttachmentId, AttachmentMeta>>,
}

impl FileAttachmentStore {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            meta: Mutex::new(HashMap::new()),
        }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    fn path_for_id(&self, id: &AttachmentId) -> PathBuf {
        let id = id.as_str();
        let prefix = id.get(..2).unwrap_or(id);
        self.root.join("sha256").join(prefix).join(id)
    }

    fn meta_path_for_id(&self, id: &AttachmentId) -> PathBuf {
        self.path_for_id(id).with_extension("json")
    }
}

impl AttachmentStore for FileAttachmentStore {
    fn persistence(&self) -> AttachmentStorePersistence {
        AttachmentStorePersistence::Durable
    }

    fn put(
        &self,
        bytes: Vec<u8>,
        meta: AttachmentCreateMeta,
    ) -> Result<AttachmentRef, AttachmentStoreError> {
        let meta = stored_meta(&bytes, meta);
        let path = self.path_for_id(&meta.id);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|source| AttachmentStoreError::Io {
                path: parent.to_path_buf(),
                source,
            })?;
        }
        if !path.exists() {
            fs::write(&path, &bytes).map_err(|source| AttachmentStoreError::Io {
                path: path.clone(),
                source,
            })?;
        }
        let meta_path = self.meta_path_for_id(&meta.id);
        let meta_bytes = serde_json::to_vec_pretty(&meta).expect("attachment metadata serializes");
        fs::write(&meta_path, meta_bytes).map_err(|source| AttachmentStoreError::Io {
            path: meta_path.clone(),
            source,
        })?;
        let reference = meta.as_ref();
        self.meta
            .lock()
            .expect("attachment metadata lock")
            .insert(reference.id.clone(), meta);
        Ok(reference)
    }

    fn get(&self, id: &AttachmentId) -> Result<StoredAttachment, AttachmentStoreError> {
        let path = self.path_for_id(id);
        let bytes = fs::read(&path).map_err(|source| {
            if source.kind() == std::io::ErrorKind::NotFound {
                AttachmentStoreError::NotFound(id.clone())
            } else {
                AttachmentStoreError::Io {
                    path: path.clone(),
                    source,
                }
            }
        })?;
        let meta = if let Some(meta) = self
            .meta
            .lock()
            .expect("attachment metadata lock")
            .get(id)
            .cloned()
        {
            meta
        } else {
            let meta_path = self.meta_path_for_id(id);
            let meta_bytes = fs::read(&meta_path).map_err(|source| {
                if source.kind() == std::io::ErrorKind::NotFound {
                    AttachmentStoreError::MissingMeta(id.clone())
                } else {
                    AttachmentStoreError::Io {
                        path: meta_path.clone(),
                        source,
                    }
                }
            })?;
            serde_json::from_slice(&meta_bytes).map_err(|source| AttachmentStoreError::Io {
                path: meta_path,
                source: std::io::Error::new(std::io::ErrorKind::InvalidData, source),
            })?
        };
        Ok(StoredAttachment { meta, bytes })
    }
}

pub fn content_id(bytes: &[u8]) -> AttachmentId {
    AttachmentId::new(format!("{:x}", Sha256::digest(bytes)))
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

    #[test]
    fn file_store_reads_after_write() {
        let temp = tempfile::tempdir().expect("tempdir");
        let store = FileAttachmentStore::new(temp.path());
        let reference = store.put(vec![9, 8, 7], meta()).expect("put");
        let stored = store.get(&reference.id).expect("get");
        assert_eq!(stored.bytes, vec![9, 8, 7]);
        assert_eq!(stored.meta.label.as_deref(), Some("pixel"));
    }
}
