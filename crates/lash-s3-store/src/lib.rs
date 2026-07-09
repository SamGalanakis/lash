//! S3-compatible attachment storage for Lash.
//!
//! This crate stores attachment bytes in any S3-compatible object store,
//! including AWS S3 and MinIO. Runtime metadata and attachment manifests remain
//! in the configured [`lash_core::RuntimePersistence`] backend.

use lash_core::{
    AttachmentCreateMeta, AttachmentId, AttachmentMeta, AttachmentRef, AttachmentStore,
    AttachmentStoreError, AttachmentStorePersistence, StoredAttachment,
};
use object_store::aws::AmazonS3Builder;
use object_store::path::Path;
use object_store::{ObjectStore, ObjectStoreExt};
use std::sync::Arc;
use url::Url;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct S3AttachmentStoreConfig {
    pub endpoint_url: Option<String>,
    pub region: String,
    pub bucket: String,
    pub prefix: Option<String>,
    pub access_key_id: Option<String>,
    pub secret_access_key: Option<String>,
    pub path_style: bool,
}

impl S3AttachmentStoreConfig {
    pub fn new(bucket: impl Into<String>, region: impl Into<String>) -> Self {
        Self {
            endpoint_url: None,
            region: region.into(),
            bucket: bucket.into(),
            prefix: None,
            access_key_id: None,
            secret_access_key: None,
            path_style: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct S3AttachmentStoreBuilder {
    config: S3AttachmentStoreConfig,
}

impl S3AttachmentStoreBuilder {
    pub fn new(bucket: impl Into<String>, region: impl Into<String>) -> Self {
        Self {
            config: S3AttachmentStoreConfig::new(bucket, region),
        }
    }

    pub fn from_config(config: S3AttachmentStoreConfig) -> Self {
        Self { config }
    }

    pub fn endpoint_url(mut self, endpoint_url: impl Into<String>) -> Self {
        self.config.endpoint_url = Some(endpoint_url.into());
        self
    }

    pub fn prefix(mut self, prefix: impl Into<String>) -> Self {
        self.config.prefix = Some(prefix.into());
        self
    }

    pub fn access_key_id(mut self, access_key_id: impl Into<String>) -> Self {
        self.config.access_key_id = Some(access_key_id.into());
        self
    }

    pub fn secret_access_key(mut self, secret_access_key: impl Into<String>) -> Self {
        self.config.secret_access_key = Some(secret_access_key.into());
        self
    }

    pub fn path_style(mut self, path_style: bool) -> Self {
        self.config.path_style = path_style;
        self
    }

    pub fn build(self) -> Result<S3AttachmentStore, AttachmentStoreError> {
        S3AttachmentStore::from_config(self.config)
    }
}

#[derive(Clone)]
pub struct S3AttachmentStore {
    store: Arc<dyn ObjectStore>,
    prefix: Option<String>,
}

impl std::fmt::Debug for S3AttachmentStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("S3AttachmentStore")
            .field("prefix", &self.prefix)
            .finish_non_exhaustive()
    }
}

impl S3AttachmentStore {
    pub fn builder(
        bucket: impl Into<String>,
        region: impl Into<String>,
    ) -> S3AttachmentStoreBuilder {
        S3AttachmentStoreBuilder::new(bucket, region)
    }

    pub fn from_config(config: S3AttachmentStoreConfig) -> Result<Self, AttachmentStoreError> {
        let prefix = normalize_prefix(config.prefix.as_deref());
        let mut builder = AmazonS3Builder::from_env()
            .with_region(config.region)
            .with_bucket_name(config.bucket)
            .with_virtual_hosted_style_request(!config.path_style);

        if let Some(endpoint) = config.endpoint_url {
            if endpoint_uses_http(&endpoint) {
                builder = builder.with_allow_http(true);
            }
            builder = builder.with_endpoint(endpoint);
        }
        if let Some(access_key_id) = config.access_key_id {
            builder = builder.with_access_key_id(access_key_id);
        }
        if let Some(secret_access_key) = config.secret_access_key {
            builder = builder.with_secret_access_key(secret_access_key);
        }

        let store = builder.build().map_err(|err| {
            AttachmentStoreError::Backend(format!("failed to build S3 store: {err}"))
        })?;

        Ok(Self {
            store: Arc::new(store),
            prefix,
        })
    }

    pub fn from_object_store(store: Arc<dyn ObjectStore>, prefix: Option<String>) -> Self {
        Self {
            store,
            prefix: normalize_prefix(prefix.as_deref()),
        }
    }

    fn content_path(&self, id: &AttachmentId) -> Result<Path, AttachmentStoreError> {
        self.path_for(None, id)
    }

    fn session_content_path(
        &self,
        session_id: &str,
        id: &AttachmentId,
    ) -> Result<Path, AttachmentStoreError> {
        self.path_for(Some(session_id), id)
    }

    fn path_for(
        &self,
        session_id: Option<&str>,
        id: &AttachmentId,
    ) -> Result<Path, AttachmentStoreError> {
        let hash = id.as_str();
        let first = hash.get(..2).unwrap_or(hash);
        let mut path = String::new();
        if let Some(prefix) = &self.prefix {
            path.push_str(prefix);
            path.push('/');
        }
        if let Some(session_id) = session_id {
            path.push_str("sessions/");
            path.push_str(&lash_core::attachments::session_storage_namespace(
                session_id,
            ));
            path.push('/');
        }
        path.push_str("sha256/");
        path.push_str(first);
        path.push('/');
        path.push_str(hash);
        Path::parse(path).map_err(|err| {
            AttachmentStoreError::Backend(format!("invalid S3 attachment path for `{id}`: {err}"))
        })
    }
}

#[async_trait::async_trait]
impl AttachmentStore for S3AttachmentStore {
    fn persistence(&self) -> AttachmentStorePersistence {
        AttachmentStorePersistence::Durable
    }

    async fn put(
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
        put_at_path(&*self.store, self.content_path(&meta.id)?, bytes, meta).await
    }

    async fn get(&self, id: &AttachmentId) -> Result<StoredAttachment, AttachmentStoreError> {
        get_at_path(&*self.store, self.content_path(id)?, id).await
    }

    async fn delete(&self, id: &AttachmentId) -> Result<(), AttachmentStoreError> {
        delete_at_path(&*self.store, self.content_path(id)?).await
    }

    async fn put_for_session(
        &self,
        session_id: &str,
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
        put_at_path(
            &*self.store,
            self.session_content_path(session_id, &meta.id)?,
            bytes,
            meta,
        )
        .await
    }

    async fn get_for_session(
        &self,
        session_id: &str,
        id: &AttachmentId,
    ) -> Result<StoredAttachment, AttachmentStoreError> {
        get_at_path(&*self.store, self.session_content_path(session_id, id)?, id).await
    }

    async fn delete_for_session(
        &self,
        session_id: &str,
        id: &AttachmentId,
    ) -> Result<(), AttachmentStoreError> {
        delete_at_path(&*self.store, self.session_content_path(session_id, id)?).await
    }
}

async fn put_at_path(
    store: &dyn ObjectStore,
    content_path: Path,
    bytes: Vec<u8>,
    meta: AttachmentMeta,
) -> Result<AttachmentRef, AttachmentStoreError> {
    let reference = meta.as_ref();
    store
        .put(&content_path, bytes.into())
        .await
        .map_err(|err| {
            AttachmentStoreError::Backend(format!("failed to write `{content_path}`: {err}"))
        })?;

    Ok(reference)
}

async fn get_at_path(
    store: &dyn ObjectStore,
    content_path: Path,
    id: &AttachmentId,
) -> Result<StoredAttachment, AttachmentStoreError> {
    let bytes = store
        .get(&content_path)
        .await
        .map_err(|err| map_object_store_get_error(err, id))?
        .bytes()
        .await
        .map_err(|err| {
            AttachmentStoreError::Backend(format!("failed to read `{content_path}`: {err}"))
        })?
        .to_vec();

    Ok(StoredAttachment { bytes })
}

async fn delete_at_path(store: &dyn ObjectStore, path: Path) -> Result<(), AttachmentStoreError> {
    match store.delete(&path).await {
        Ok(()) => {}
        Err(object_store::Error::NotFound { .. }) => {}
        Err(err) => {
            return Err(AttachmentStoreError::Backend(format!(
                "failed to delete `{path}`: {err}"
            )));
        }
    }
    Ok(())
}

fn map_object_store_get_error(err: object_store::Error, id: &AttachmentId) -> AttachmentStoreError {
    match err {
        object_store::Error::NotFound { .. } => AttachmentStoreError::NotFound(id.clone()),
        err => AttachmentStoreError::Backend(err.to_string()),
    }
}

fn normalize_prefix(prefix: Option<&str>) -> Option<String> {
    prefix
        .map(|value| value.trim_matches('/').to_string())
        .filter(|value| !value.is_empty())
}

fn endpoint_uses_http(endpoint: &str) -> bool {
    Url::parse(endpoint)
        .map(|url| url.scheme() == "http")
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lash_core::testing::conformance::ReopenableAttachmentStore;
    use lash_core::{ImageMediaType, MediaType};

    #[test]
    fn normalizes_prefixes() {
        assert_eq!(normalize_prefix(None), None);
        assert_eq!(normalize_prefix(Some("")), None);
        assert_eq!(
            normalize_prefix(Some("/lash/e2e/")),
            Some("lash/e2e".to_string())
        );
    }

    #[tokio::test]
    async fn minio_attachment_store_satisfies_conformance_when_configured() {
        let Some(config) = minio_config_from_env() else {
            eprintln!("skipping MinIO conformance: LASH_MINIO_ENDPOINT is not set");
            return;
        };
        lash_core::testing::conformance::attachment_store_reopenable(
            || {
                let open =
                    Arc::new(S3AttachmentStore::from_config(config.clone()).expect("open store"))
                        as Arc<dyn AttachmentStore>;
                let reopen =
                    Arc::new(S3AttachmentStore::from_config(config.clone()).expect("reopen store"))
                        as Arc<dyn AttachmentStore>;
                ReopenableAttachmentStore { open, reopen }
            },
            AttachmentStorePersistence::Durable,
        )
        .await;
    }

    #[tokio::test]
    async fn duplicate_puts_are_content_addressed_when_minio_configured() {
        let Some(config) = minio_config_from_env() else {
            eprintln!("skipping MinIO duplicate-put test: LASH_MINIO_ENDPOINT is not set");
            return;
        };
        let store = S3AttachmentStore::from_config(config).expect("store");
        let meta = AttachmentCreateMeta::new(
            MediaType::Image(ImageMediaType::Png),
            Some(1),
            Some(1),
            Some("pixel".to_string()),
        );
        let first = store
            .put(vec![1, 2, 3], meta.clone())
            .await
            .expect("first put");
        let second = store.put(vec![1, 2, 3], meta).await.expect("second put");
        assert_eq!(first.id, second.id);
    }

    #[tokio::test]
    async fn delete_removes_content_and_is_idempotent_when_minio_configured() {
        let Some(config) = minio_config_from_env() else {
            eprintln!("skipping MinIO delete test: LASH_MINIO_ENDPOINT is not set");
            return;
        };
        let store = S3AttachmentStore::from_config(config).expect("store");
        let meta = AttachmentCreateMeta::new(
            MediaType::Image(ImageMediaType::Png),
            Some(1),
            Some(1),
            Some("pixel".to_string()),
        );
        let reference = store.put(vec![4, 5, 6, 7], meta).await.expect("put");
        store
            .get(&reference.id)
            .await
            .expect("present before delete");

        store.delete(&reference.id).await.expect("delete content");
        let err = store
            .get(&reference.id)
            .await
            .expect_err("content must be gone after delete");
        assert!(
            matches!(err, AttachmentStoreError::NotFound(_)),
            "deleted content must map to NotFound, got {err:?}"
        );

        // Idempotent: deleting already-absent content succeeds.
        store
            .delete(&reference.id)
            .await
            .expect("delete of absent content is a no-op");
    }

    fn minio_config_from_env() -> Option<S3AttachmentStoreConfig> {
        let endpoint_url = std::env::var("LASH_MINIO_ENDPOINT").ok()?;
        let bucket =
            std::env::var("LASH_MINIO_BUCKET").unwrap_or_else(|_| "lash-attachments".into());
        let region = std::env::var("LASH_MINIO_REGION").unwrap_or_else(|_| "us-east-1".into());
        Some(S3AttachmentStoreConfig {
            endpoint_url: Some(endpoint_url),
            region,
            bucket,
            prefix: Some(format!(
                "tests/{}",
                std::env::var("LASH_MINIO_PREFIX").unwrap_or_else(|_| uuid_like_suffix())
            )),
            access_key_id: Some(
                std::env::var("LASH_MINIO_ACCESS_KEY").unwrap_or_else(|_| "minioadmin".into()),
            ),
            secret_access_key: Some(
                std::env::var("LASH_MINIO_SECRET_KEY").unwrap_or_else(|_| "minioadmin".into()),
            ),
            path_style: true,
        })
    }

    fn uuid_like_suffix() -> String {
        format!(
            "{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        )
    }
}
