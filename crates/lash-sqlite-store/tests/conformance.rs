//! Runs the backend-agnostic `ProcessRegistry` conformance suite against the
//! SQLite implementation. The same suite runs against the in-memory registry
//! in lash-core, so both backends are held to one contract.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use lash_core::testing::conformance::{
    ReopenableLashlangArtifactStore, ReopenableProcessRegistry, ReopenableRuntimePersistence,
};
use lash_core::{DurabilityTier, LashlangArtifactStore, ProcessRegistry, RuntimePersistence};
use lash_sqlite_store::{SqliteProcessRegistry, Store};
use tempfile::TempDir;

fn fresh_db_path(dirs: &Arc<Mutex<Vec<TempDir>>>, file_name: &str) -> PathBuf {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join(file_name);
    dirs.lock().expect("dirs lock").push(dir);
    path
}

fn open_registry(path: &Path) -> Arc<dyn ProcessRegistry> {
    Arc::new(SqliteProcessRegistry::open(path).expect("file registry")) as Arc<dyn ProcessRegistry>
}

fn open_store(path: &Path) -> Arc<dyn RuntimePersistence> {
    Arc::new(Store::open(path).expect("file store")) as Arc<dyn RuntimePersistence>
}

fn open_artifact_store(path: &Path) -> Arc<dyn LashlangArtifactStore> {
    Arc::new(Store::open(path).expect("file store")) as Arc<dyn LashlangArtifactStore>
}

#[tokio::test]
async fn sqlite_process_registry_satisfies_conformance() {
    let dirs = Arc::new(Mutex::new(Vec::new()));
    lash_core::testing::conformance::process_registry_reopenable(|| {
        let path = fresh_db_path(&dirs, "processes.db");
        ReopenableProcessRegistry {
            open: open_registry(&path),
            reopen: open_registry(&path),
        }
    })
    .await;
}

#[tokio::test]
async fn sqlite_store_satisfies_runtime_persistence_conformance() {
    let dirs = Arc::new(Mutex::new(Vec::new()));
    lash_core::testing::conformance::runtime_persistence_reopenable(|| {
        let path = fresh_db_path(&dirs, "session.db");
        ReopenableRuntimePersistence {
            open: open_store(&path),
            reopen: open_store(&path),
        }
    })
    .await;
}

#[test]
fn sqlite_store_satisfies_lashlang_artifact_store_conformance() {
    let dirs = Arc::new(Mutex::new(Vec::new()));
    lash_core::testing::conformance::lashlang_artifact_store_reopenable(
        || {
            let path = fresh_db_path(&dirs, "artifacts.db");
            ReopenableLashlangArtifactStore {
                open: open_artifact_store(&path),
                reopen: open_artifact_store(&path),
            }
        },
        DurabilityTier::Durable,
    );
}
