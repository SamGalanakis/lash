//! Compiled sources for the Rust snippets on `docs/rlm.html`.

use lash::provider::ProviderHandle;

async fn rlm_core(provider: ProviderHandle, model_id: &str) -> anyhow::Result<()> {
    // docs:start:rlm-core
    use std::sync::Arc;

    use lash::TurnInput;

    let core = lash::RlmCore::builder()
        .plugins(lash::plugins::runtime_plugin_stack())
        .provider(provider)
        .model(
            lash::ModelSpec::from_token_limits(model_id, None, 200_000, None)
                .expect("valid model metadata"),
        )
        .effect_host(Arc::new(lash::durability::InlineEffectHost::default()))
        .lashlang_artifact_store(Arc::new(
            lash::persistence::InMemoryLashlangArtifactStore::new(),
        ))
        .attachment_store(Arc::new(lash::persistence::InMemoryAttachmentStore::new()))
        .build()?;

    let session = core.session("task-42").open().await?;
    let output = session
        .turn(TurnInput::text(
            "Inspect the task and finish a concise result.",
        ))
        .run()
        .await?;
    // docs:end:rlm-core
    Ok(())
}
