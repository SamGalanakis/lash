//! Runs the backend-agnostic `AttachmentStore` conformance suite against the
//! file-backed implementation. The same suite runs against the in-memory store
//! in lash-core, so both backends are held to one contract.

use std::sync::{Arc, Mutex};

use lash_core::testing::conformance::ReopenableAttachmentStore;
use lash_core::{AttachmentStore, AttachmentStorePersistence};
use lash_local_store::FileAttachmentStore;
use tempfile::TempDir;

#[test]
fn file_attachment_store_satisfies_conformance() {
    // Each `make()` call needs its own root that outlives the returned store.
    // Keep the tempdirs alive for the duration of the suite.
    let dirs: Arc<Mutex<Vec<TempDir>>> = Arc::new(Mutex::new(Vec::new()));
    lash_core::testing::conformance::attachment_store_reopenable(
        || {
            let dir = tempfile::tempdir().expect("tempdir");
            let open = Arc::new(FileAttachmentStore::new(dir.path())) as Arc<dyn AttachmentStore>;
            let reopen = Arc::new(FileAttachmentStore::new(dir.path())) as Arc<dyn AttachmentStore>;
            dirs.lock().expect("dirs lock").push(dir);
            ReopenableAttachmentStore { open, reopen }
        },
        AttachmentStorePersistence::Durable,
    );
}
