//! Shared fixtures for the conformance suites: paired handles opened
//! against the same durable backing store, used by the `*_reopenable`
//! suite variants.

use super::*;

/// A pair of [`ProcessRegistry`] handles opened against the same durable
/// backing store.
pub struct ReopenableProcessRegistry {
    pub open: Arc<dyn ProcessRegistry>,
    pub reopen: Arc<dyn ProcessRegistry>,
}

/// A pair of [`RuntimePersistence`] handles opened against the same durable
/// backing store.
pub struct ReopenableRuntimePersistence {
    pub open: Arc<dyn RuntimePersistence>,
    pub reopen: Arc<dyn RuntimePersistence>,
}

/// A pair of [`AttachmentStore`](crate::AttachmentStore) handles opened against
/// the same durable backing store.
pub struct ReopenableAttachmentStore {
    pub open: Arc<dyn crate::AttachmentStore>,
    pub reopen: Arc<dyn crate::AttachmentStore>,
}

/// A pair of [`LashlangArtifactStore`] handles opened against the same durable
/// backing store.
pub struct ReopenableLashlangArtifactStore {
    pub open: Arc<dyn crate::LashlangArtifactStore>,
    pub reopen: Arc<dyn crate::LashlangArtifactStore>,
}

/// A pair of [`HostEventStore`](crate::HostEventStore) handles opened against
/// the same durable backing store.
pub struct ReopenableHostEventStore {
    pub open: Arc<dyn crate::HostEventStore>,
    pub reopen: Arc<dyn crate::HostEventStore>,
}
