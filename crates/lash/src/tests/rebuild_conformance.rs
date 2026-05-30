//! Runs the public
//! [`runtime_rebuild_and_worker_recovery`](crate::testing::conformance::runtime_rebuild_and_worker_recovery)
//! conformance suite against this crate's store backends. The suite proves cold
//! rebuild of a trigger-mutated session and durable worker recovery across every
//! `ProcessInput` variant the worker runs.
//!
//! Two backends differing in the attachment/artifact tier over a SQLite session
//! store: the `.in_memory_stores()` defaults and explicit durable stores. (A
//! fully in-memory session store has no public factory, and peer-coherence
//! rejects a durable session factory paired with an explicit ephemeral artifact
//! store, so the session store is SQLite in both.)

use super::*;
use crate::testing::conformance::{RuntimeRebuildBackend, runtime_rebuild_and_worker_recovery};

fn fresh_sqlite_session_backend(
    root: &std::path::Path,
) -> (
    std::path::PathBuf,
    Arc<dyn lash_core::SessionStoreFactory>,
    Arc<dyn lash_core::ProcessRegistry>,
) {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static SCENARIO: AtomicUsize = AtomicUsize::new(0);
    let dir = root.join(format!(
        "scenario-{}",
        SCENARIO.fetch_add(1, Ordering::SeqCst)
    ));
    std::fs::create_dir_all(&dir).expect("create scenario dir");
    let store_factory = Arc::new(lash_sqlite_store::SqliteSessionStoreFactory::new(
        dir.join("sessions"),
    )) as Arc<dyn lash_core::SessionStoreFactory>;
    let registry = Arc::new(
        lash_sqlite_store::SqliteProcessRegistry::open(&dir.join("processes.db"))
            .expect("open process registry"),
    ) as Arc<dyn lash_core::ProcessRegistry>;
    (dir, store_factory, registry)
}

#[tokio::test]
async fn runtime_rebuild_and_worker_recovery_with_in_memory_artifact() {
    let root = tempfile::tempdir().expect("tempdir");
    let root_path = root.path().to_path_buf();
    runtime_rebuild_and_worker_recovery(move || {
        let (_dir, store_factory, registry) = fresh_sqlite_session_backend(&root_path);
        RuntimeRebuildBackend {
            process_registry: registry,
            build_core: Box::new(move |builder| {
                builder
                    .store_factory(Arc::clone(&store_factory))
                    .in_memory_stores()
                    .build()
                    .expect("build core")
            }),
        }
    })
    .await;
}

#[tokio::test]
async fn runtime_rebuild_and_worker_recovery_with_durable_stores() {
    let root = tempfile::tempdir().expect("tempdir");
    let root_path = root.path().to_path_buf();
    runtime_rebuild_and_worker_recovery(move || {
        let (dir, store_factory, registry) = fresh_sqlite_session_backend(&root_path);
        let attachment = Arc::new(crate::persistence::FileAttachmentStore::new(
            dir.join("attachments"),
        )) as Arc<dyn lash_core::AttachmentStore>;
        let artifact = Arc::new(
            lash_sqlite_store::Store::open(&dir.join("artifacts.db"))
                .expect("open durable artifact store"),
        ) as Arc<dyn lash_core::LashlangArtifactStore>;
        RuntimeRebuildBackend {
            process_registry: registry,
            build_core: Box::new(move |builder| {
                builder
                    .store_factory(Arc::clone(&store_factory))
                    .attachment_store(Arc::clone(&attachment))
                    .lashlang_artifact_store(Arc::clone(&artifact))
                    .advanced()
                    .effect_controller(Arc::new(
                        crate::advanced::InlineRuntimeEffectController::default(),
                    ))
                    .build()
                    .expect("build core")
            }),
        }
    })
    .await;
}
