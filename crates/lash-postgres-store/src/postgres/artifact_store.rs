#[async_trait::async_trait]
impl lashlang::LashlangArtifactStore for PostgresLashlangArtifactStore {
    fn durability_tier(&self) -> DurabilityTier {
        DurabilityTier::Durable
    }

    async fn put_module_artifact(
        &self,
        artifact: &lashlang::ModuleArtifact,
    ) -> Result<(), lashlang::ArtifactStoreError> {
        let bytes = artifact
            .to_store_bytes()
            .map_err(lashlang::ArtifactStoreError::from)?;
        sqlx::query(
            "INSERT INTO lash_lashlang_artifacts (module_ref, artifact_bytes)
             VALUES ($1, $2)
             ON CONFLICT (module_ref) DO UPDATE SET artifact_bytes = EXCLUDED.artifact_bytes",
        )
        .bind(artifact.module_ref.as_str())
        .bind(bytes)
        .execute(&self.pool)
        .await
        .map_err(|err| lashlang::ArtifactStoreError::Backend(err.to_string()))?;
        Ok(())
    }

    async fn get_module_artifact(
        &self,
        module_ref: &lashlang::ModuleRef,
    ) -> Result<Option<Arc<lashlang::ModuleArtifact>>, lashlang::ArtifactStoreError> {
        let bytes: Option<Vec<u8>> = sqlx::query_scalar(
            "SELECT artifact_bytes FROM lash_lashlang_artifacts WHERE module_ref = $1",
        )
        .bind(module_ref.as_str())
        .fetch_optional(&self.pool)
        .await
        .map_err(|err| lashlang::ArtifactStoreError::Backend(err.to_string()))?;
        bytes
            .map(|bytes| {
                lashlang::ModuleArtifact::from_store_bytes(&bytes)
                    .map(Arc::new)
                    .map_err(lashlang::ArtifactStoreError::from)
            })
            .transpose()
    }

    async fn put_artifact_bytes(
        &self,
        artifact_ref: &str,
        _descriptor: &str,
        bytes: &[u8],
    ) -> Result<(), lashlang::ArtifactStoreError> {
        sqlx::query(
            "INSERT INTO lash_lashlang_artifacts (module_ref, artifact_bytes)
             VALUES ($1, $2)
             ON CONFLICT (module_ref) DO UPDATE SET artifact_bytes = EXCLUDED.artifact_bytes",
        )
        .bind(artifact_ref)
        .bind(bytes)
        .execute(&self.pool)
        .await
        .map_err(|err| lashlang::ArtifactStoreError::Backend(err.to_string()))?;
        Ok(())
    }

    async fn get_artifact_bytes(
        &self,
        artifact_ref: &str,
    ) -> Result<Option<Vec<u8>>, lashlang::ArtifactStoreError> {
        sqlx::query_scalar(
            "SELECT artifact_bytes FROM lash_lashlang_artifacts WHERE module_ref = $1",
        )
        .bind(artifact_ref)
        .fetch_optional(&self.pool)
        .await
        .map_err(|err| lashlang::ArtifactStoreError::Backend(err.to_string()))
    }
}
