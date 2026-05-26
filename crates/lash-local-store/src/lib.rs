use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use lash_core::{
    AttachmentCreateMeta, AttachmentId, AttachmentMeta, AttachmentRef, AttachmentStore,
    AttachmentStoreError, AttachmentStorePersistence, StoredAttachment,
};

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
        let meta = AttachmentMeta::new(
            lash_core::attachments::content_id(&bytes),
            meta.media_type,
            bytes.len() as u64,
            meta.width,
            meta.height,
            meta.label,
        );
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

#[cfg(test)]
mod tests {
    use super::*;
    use lash_core::{ImageMediaType, MediaType};

    fn meta() -> AttachmentCreateMeta {
        AttachmentCreateMeta::new(
            MediaType::Image(ImageMediaType::Png),
            Some(1),
            Some(1),
            Some("pixel".to_string()),
        )
    }

    #[test]
    fn file_store_round_trips_bytes_and_metadata() {
        let temp = tempfile::tempdir().expect("tempdir");
        let store = FileAttachmentStore::new(temp.path());
        let reference = store.put(vec![1, 2, 3], meta()).expect("put");
        let stored = store.get(&reference.id).expect("get");

        assert_eq!(stored.bytes, vec![1, 2, 3]);
        assert_eq!(stored.meta.id, reference.id);
        assert_eq!(stored.meta.byte_len, 3);
    }
}
