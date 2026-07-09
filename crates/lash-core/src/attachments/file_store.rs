use std::fs;
use std::path::{Path, PathBuf};

use lash_sansio::{AttachmentCreateMeta, AttachmentId, AttachmentMeta, AttachmentRef};

use super::{
    AttachmentStore, AttachmentStoreError, AttachmentStorePersistence, StoredAttachment,
    content_id, session_storage_namespace,
};

pub struct FileAttachmentStore {
    root: PathBuf,
}

impl FileAttachmentStore {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    fn path_for_id(&self, id: &AttachmentId) -> PathBuf {
        self.path_in_root(&self.root, id)
    }

    fn path_for_session(&self, session_id: &str, id: &AttachmentId) -> PathBuf {
        let root = self
            .root
            .join("sessions")
            .join(session_storage_namespace(session_id));
        self.path_in_root(&root, id)
    }

    fn path_in_root(&self, root: &Path, id: &AttachmentId) -> PathBuf {
        let id = id.as_str();
        let prefix = id.get(..2).unwrap_or(id);
        root.join("sha256").join(prefix).join(id)
    }
}

/// Write `bytes` to `final_path` crash-atomically: stage into a sibling
/// `<final>.tmp`, flush it, then `rename` into place. A `rename` within the
/// same directory is atomic on POSIX, so a reader (or a crash) ever sees either
/// the old contents or the complete new contents — never a half-written file.
/// The temp file is removed on any failure so a crashed write leaves no
/// `.tmp` litter behind.
fn write_atomic(final_path: &Path, bytes: &[u8]) -> Result<(), AttachmentStoreError> {
    let mut tmp_os = final_path.as_os_str().to_os_string();
    tmp_os.push(".tmp");
    let tmp_path = PathBuf::from(tmp_os);

    let io_err = |path: &Path, source: std::io::Error| AttachmentStoreError::Io {
        path: path.to_path_buf(),
        source,
    };

    let write_result = (|| {
        let mut file = fs::File::create(&tmp_path).map_err(|source| io_err(&tmp_path, source))?;
        std::io::Write::write_all(&mut file, bytes).map_err(|source| io_err(&tmp_path, source))?;
        // Best-effort durability for the staged bytes before the rename.
        file.sync_all()
            .map_err(|source| io_err(&tmp_path, source))?;
        fs::rename(&tmp_path, final_path).map_err(|source| io_err(final_path, source))
    })();

    if write_result.is_err() {
        // Never leave a partial temp file behind.
        let _ = fs::remove_file(&tmp_path);
    }
    write_result
}

#[async_trait::async_trait]
impl AttachmentStore for FileAttachmentStore {
    fn persistence(&self) -> AttachmentStorePersistence {
        AttachmentStorePersistence::Durable
    }

    async fn put(
        &self,
        bytes: Vec<u8>,
        meta: AttachmentCreateMeta,
    ) -> Result<AttachmentRef, AttachmentStoreError> {
        let meta = AttachmentMeta::new(
            content_id(&bytes),
            meta.media_type,
            bytes.len() as u64,
            meta.width,
            meta.height,
            meta.label,
        );
        let path = self.path_for_id(&meta.id);
        put_at_path(path, bytes, meta)
    }

    async fn get(&self, id: &AttachmentId) -> Result<StoredAttachment, AttachmentStoreError> {
        get_at_path(self.path_for_id(id), id)
    }

    async fn delete(&self, id: &AttachmentId) -> Result<(), AttachmentStoreError> {
        delete_at_path(self.path_for_id(id))
    }

    async fn put_for_session(
        &self,
        session_id: &str,
        bytes: Vec<u8>,
        meta: AttachmentCreateMeta,
    ) -> Result<AttachmentRef, AttachmentStoreError> {
        let meta = AttachmentMeta::new(
            content_id(&bytes),
            meta.media_type,
            bytes.len() as u64,
            meta.width,
            meta.height,
            meta.label,
        );
        put_at_path(self.path_for_session(session_id, &meta.id), bytes, meta)
    }

    async fn get_for_session(
        &self,
        session_id: &str,
        id: &AttachmentId,
    ) -> Result<StoredAttachment, AttachmentStoreError> {
        get_at_path(self.path_for_session(session_id, id), id)
    }

    async fn delete_for_session(
        &self,
        session_id: &str,
        id: &AttachmentId,
    ) -> Result<(), AttachmentStoreError> {
        delete_at_path(self.path_for_session(session_id, id))
    }
}

fn put_at_path(
    path: PathBuf,
    bytes: Vec<u8>,
    meta: AttachmentMeta,
) -> Result<AttachmentRef, AttachmentStoreError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|source| AttachmentStoreError::Io {
            path: parent.to_path_buf(),
            source,
        })?;
    }
    if !path.exists() {
        write_atomic(&path, &bytes)?;
    }
    Ok(meta.as_ref())
}

fn get_at_path(path: PathBuf, id: &AttachmentId) -> Result<StoredAttachment, AttachmentStoreError> {
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
    Ok(StoredAttachment { bytes })
}

fn delete_at_path(path: PathBuf) -> Result<(), AttachmentStoreError> {
    match fs::remove_file(&path) {
        Ok(()) => Ok(()),
        Err(source) if source.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(source) => Err(AttachmentStoreError::Io { path, source }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ImageMediaType, MediaType};
    use std::sync::Mutex;

    fn meta() -> AttachmentCreateMeta {
        AttachmentCreateMeta::new(
            MediaType::Image(ImageMediaType::Png),
            Some(1),
            Some(1),
            Some("pixel".to_string()),
        )
    }

    #[tokio::test]
    async fn file_store_round_trips_bytes_and_metadata() {
        let temp = tempfile::tempdir().expect("tempdir");
        let store = FileAttachmentStore::new(temp.path());
        let reference = store.put(vec![1, 2, 3], meta()).await.expect("put");
        let stored = store.get(&reference.id).await.expect("get");

        assert_eq!(stored.bytes, vec![1, 2, 3]);
        assert_eq!(reference.byte_len, 3);
    }

    // Finding 4: `put` must write crash-atomically (stage into `<final>.tmp`,
    // then rename). After a successful put there must be no leftover `.tmp`
    // files in the content directory — proof that the temp file was renamed
    // into place rather than written in situ.
    #[tokio::test]
    async fn file_store_writes_atomically_without_temp_litter() {
        let temp = tempfile::tempdir().expect("tempdir");
        let store = FileAttachmentStore::new(temp.path());
        let reference = store.put(vec![9, 8, 7, 6], meta()).await.expect("put");

        let final_path = store.path_for_id(&reference.id);
        assert!(final_path.exists(), "content file must be in place");

        let mut tmp_files = Vec::new();
        let dir = final_path.parent().expect("content dir");
        for entry in fs::read_dir(dir).expect("read content dir") {
            let path = entry.expect("dir entry").path();
            if path.extension().and_then(|ext| ext.to_str()) == Some("tmp") {
                tmp_files.push(path);
            }
        }
        assert!(
            tmp_files.is_empty(),
            "atomic write must not leave .tmp files behind: {tmp_files:?}"
        );

        // The bytes round-trip in full (no truncation from a partial write).
        let stored = store.get(&reference.id).await.expect("get");
        assert_eq!(stored.bytes, vec![9, 8, 7, 6]);
    }

    // A stale `<final>.tmp` left by a crashed prior write must not block a
    // subsequent successful put — the temp file is recreated/truncated.
    #[tokio::test]
    async fn file_store_overwrites_stale_temp_file() {
        let temp = tempfile::tempdir().expect("tempdir");
        let store = FileAttachmentStore::new(temp.path());
        let content_id = content_id(&[1, 1, 1]);
        let id = AttachmentId::new(content_id.to_string());
        let final_path = store.path_for_id(&id);
        let parent = final_path.parent().expect("parent");
        fs::create_dir_all(parent).expect("mkdir");
        let mut tmp_os = final_path.as_os_str().to_os_string();
        tmp_os.push(".tmp");
        fs::write(PathBuf::from(tmp_os), b"stale partial write").expect("seed stale tmp");

        let reference = store
            .put(vec![1, 1, 1], meta())
            .await
            .expect("put over stale tmp");
        let stored = store.get(&reference.id).await.expect("get");
        assert_eq!(stored.bytes, vec![1, 1, 1]);
    }

    #[tokio::test]
    async fn file_store_session_namespaces_isolate_identical_content() {
        let temp = tempfile::tempdir().expect("tempdir");
        let store = FileAttachmentStore::new(temp.path());
        let first = store
            .put_for_session("session-a", vec![7, 7, 7], meta())
            .await
            .expect("put first");
        let second = store
            .put_for_session("session-b", vec![7, 7, 7], meta())
            .await
            .expect("put second");
        assert_eq!(first.id, second.id);
        assert_ne!(
            store.path_for_session("session-a", &first.id),
            store.path_for_session("session-b", &second.id)
        );

        store
            .delete_for_session("session-b", &second.id)
            .await
            .expect("delete second");
        assert!(matches!(
            store.get_for_session("session-b", &second.id).await,
            Err(AttachmentStoreError::NotFound(_))
        ));
        assert_eq!(
            store
                .get_for_session("session-a", &first.id)
                .await
                .expect("first survives")
                .bytes,
            vec![7, 7, 7]
        );
    }

    // Runs the backend-agnostic `AttachmentStore` conformance suite against
    // the file-backed implementation. The same suite runs against the
    // in-memory store, so both backends are held to one contract.
    #[tokio::test]
    async fn file_attachment_store_satisfies_conformance() {
        use std::sync::Arc;

        use crate::testing::conformance::ReopenableAttachmentStore;

        // Each `make()` call needs its own root that outlives the returned
        // store. Keep the tempdirs alive for the duration of the suite.
        let dirs: Arc<Mutex<Vec<tempfile::TempDir>>> = Arc::new(Mutex::new(Vec::new()));
        crate::testing::conformance::attachment_store_reopenable(
            || {
                let dir = tempfile::tempdir().expect("tempdir");
                let open =
                    Arc::new(FileAttachmentStore::new(dir.path())) as Arc<dyn AttachmentStore>;
                let reopen =
                    Arc::new(FileAttachmentStore::new(dir.path())) as Arc<dyn AttachmentStore>;
                dirs.lock().expect("dirs lock").push(dir);
                ReopenableAttachmentStore { open, reopen }
            },
            AttachmentStorePersistence::Durable,
        )
        .await;
    }
}
