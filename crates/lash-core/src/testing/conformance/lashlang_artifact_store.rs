//! [`LashlangArtifactStore`] conformance: module-artifact round trips.

use super::*;

// ---------------------------------------------------------------------------
// LashlangArtifactStore conformance
// ---------------------------------------------------------------------------

/// Run the full [`LashlangArtifactStore`] conformance suite against the backend
/// produced by `make`. `make` must return a fresh, empty store on each call.
/// `expected_tier` is the tier this backend declares (`Inline` for in-memory,
/// `Durable` for Sqlite-backed).
pub async fn lashlang_artifact_store<F>(make: F, expected_tier: DurabilityTier)
where
    F: Fn() -> Arc<dyn LashlangArtifactStore>,
{
    artifact_put_get_round_trips(make()).await;
    artifact_get_unknown_is_none(make()).await;
    artifact_reports_declared_tier(make(), expected_tier);
}

/// Run the full [`LashlangArtifactStore`] suite plus durable reopen checks.
pub async fn lashlang_artifact_store_reopenable<F>(make: F, expected_tier: DurabilityTier)
where
    F: Fn() -> ReopenableLashlangArtifactStore,
{
    lashlang_artifact_store(|| make().open, expected_tier).await;
    lashlang_artifact_store_survives_reopen(make()).await;
}

fn sample_artifact() -> lashlang::ModuleArtifact {
    let program = lashlang::parse("process echo(value: str) { finish value }")
        .expect("sample lashlang module parses");
    lashlang::ModuleArtifact::from_program(program).expect("module artifact builds")
}

async fn artifact_put_get_round_trips(store: Arc<dyn LashlangArtifactStore>) {
    let artifact = sample_artifact();
    store
        .put_module_artifact(&artifact)
        .await
        .expect("put module artifact");
    let loaded = store
        .get_module_artifact(&artifact.module_ref)
        .await
        .expect("get module artifact")
        .expect("artifact present after put");

    assert_eq!(loaded.module_ref, artifact.module_ref);
    assert_eq!(loaded.host_requirements_ref, artifact.host_requirements_ref);
    assert_eq!(loaded.exports, artifact.exports);
    assert_eq!(
        loaded.to_store_bytes().expect("re-encode loaded artifact"),
        artifact
            .to_store_bytes()
            .expect("re-encode source artifact"),
        "stored artifact must round-trip byte-identically"
    );
}

async fn artifact_get_unknown_is_none(store: Arc<dyn LashlangArtifactStore>) {
    let unknown = sample_artifact().module_ref;
    let result = store
        .get_module_artifact(&unknown)
        .await
        .expect("get of an unknown ref must not error");
    assert!(
        result.is_none(),
        "an unknown module ref must return Ok(None), not a backend error"
    );
}

fn artifact_reports_declared_tier(store: Arc<dyn LashlangArtifactStore>, expected: DurabilityTier) {
    assert_eq!(
        store.durability_tier(),
        expected,
        "durability tier must match the backend"
    );
}

async fn lashlang_artifact_store_survives_reopen(factory: ReopenableLashlangArtifactStore) {
    let artifact = sample_artifact();
    factory
        .open
        .put_module_artifact(&artifact)
        .await
        .expect("put module artifact before reopen");
    let loaded = factory
        .reopen
        .get_module_artifact(&artifact.module_ref)
        .await
        .expect("get module artifact after reopen")
        .expect("artifact present after reopen");
    assert_eq!(loaded.module_ref, artifact.module_ref);
    assert_eq!(loaded.host_requirements_ref, artifact.host_requirements_ref);
    assert_eq!(loaded.exports, artifact.exports);
}
