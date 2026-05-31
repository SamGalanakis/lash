//! Runs the backend-agnostic `AttachmentStore` conformance suite against the
//! file-backed implementation. The same suite runs against the in-memory store
//! in lash-core, so both backends are held to one contract.

use std::sync::{Arc, Mutex};

use lash_core::{AttachmentStore, AttachmentStorePersistence};
use lash_local_store::FileAttachmentStore;
use tempfile::TempDir;

#[test]
fn file_attachment_store_satisfies_conformance() {
    // Each `make()` call needs its own root that outlives the returned store.
    // Keep the tempdirs alive for the duration of the suite.
    let dirs: Arc<Mutex<Vec<TempDir>>> = Arc::new(Mutex::new(Vec::new()));
    lash_core::testing::conformance::attachment_store(
        || {
            let dir = tempfile::tempdir().expect("tempdir");
            let store = FileAttachmentStore::new(dir.path());
            dirs.lock().expect("dirs lock").push(dir);
            Arc::new(store) as Arc<dyn AttachmentStore>
        },
        AttachmentStorePersistence::Durable,
    );
}
