//! Runs the backend-agnostic `ProcessRegistry` conformance suite against the
//! SQLite implementation. The same suite runs against the in-memory registry
//! in lash-core, so both backends are held to one contract.

use std::sync::Arc;

use lash_core::{DurabilityTier, LashlangArtifactStore, ProcessRegistry, RuntimePersistence};
use lash_sqlite_store::{SqliteProcessRegistry, Store};

#[tokio::test]
async fn sqlite_process_registry_satisfies_conformance() {
    lash_core::testing::conformance::process_registry(|| {
        Arc::new(SqliteProcessRegistry::memory().expect("memory registry"))
            as Arc<dyn ProcessRegistry>
    })
    .await;
}

#[tokio::test]
async fn sqlite_store_satisfies_runtime_persistence_conformance() {
    lash_core::testing::conformance::runtime_persistence(|| {
        Arc::new(Store::memory().expect("memory store")) as Arc<dyn RuntimePersistence>
    })
    .await;
}

#[test]
fn sqlite_store_satisfies_lashlang_artifact_store_conformance() {
    lash_core::testing::conformance::lashlang_artifact_store(
        || Arc::new(Store::memory().expect("memory store")) as Arc<dyn LashlangArtifactStore>,
        DurabilityTier::Durable,
    );
}
