use std::sync::Arc;

use crate::{AttachmentCreateMeta, AttachmentRef, AttachmentStore, AttachmentStoreError};

#[derive(Clone)]
pub struct ToolAttachmentClient {
    pub(super) store: Arc<dyn AttachmentStore>,
}

impl ToolAttachmentClient {
    pub async fn put(
        &self,
        data: Vec<u8>,
        meta: AttachmentCreateMeta,
    ) -> Result<AttachmentRef, AttachmentStoreError> {
        self.store.put(data, meta).await
    }
}
