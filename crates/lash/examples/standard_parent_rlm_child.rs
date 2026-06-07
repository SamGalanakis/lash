use std::sync::Arc;

use lash::{LashCore, ModeId, ModePreset};

#[tokio::main]
async fn main() -> lash::Result<()> {
    let core = LashCore::builder()
        .install_mode(ModePreset::standard())
        .install_mode(ModePreset::rlm())
        .default_mode(ModeId::standard())
        .model(
            lash::ModelSpec::from_token_limits("example-model", None, 200_000, None)
                .expect("valid model spec"),
        )
        .effect_host(Arc::new(lash::durability::InlineEffectHost::default()))
        .lashlang_artifact_store(Arc::new(
            lash::persistence::InMemoryLashlangArtifactStore::new(),
        ))
        .attachment_store(Arc::new(lash::persistence::InMemoryAttachmentStore::new()))
        .build()?;

    let parent = core.session("main").standard().open().await?;
    let child = core.session("child").rlm().parent("main").open().await?;

    println!("parent mode: {}", parent.mode());
    println!("child mode: {}", child.mode());
    Ok(())
}
