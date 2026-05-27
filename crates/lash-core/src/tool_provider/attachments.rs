use std::sync::Arc;

use crate::{AttachmentCreateMeta, AttachmentRef, AttachmentStore, AttachmentStoreError};

#[derive(Clone)]
pub struct ToolAttachmentControl {
    pub(super) store: Arc<dyn AttachmentStore>,
}

impl ToolAttachmentControl {
    pub fn put(
        &self,
        data: Vec<u8>,
        meta: AttachmentCreateMeta,
    ) -> Result<AttachmentRef, AttachmentStoreError> {
        self.store.put(data, meta)
    }
}
