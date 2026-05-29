use lash::{LashCore, ModeId, ModePreset};

#[tokio::main]
async fn main() -> lash::Result<()> {
    let core = LashCore::builder()
        .install_mode(ModePreset::standard())
        .install_mode(ModePreset::rlm())
        .default_mode(ModeId::standard())
        .model(
            lash::ModelSpec::from_token_limits("example-model", None, 200_000, None, None)
                .expect("valid model spec"),
        )
        .in_memory_stores()
        .build()?;

    let parent = core.session("main").standard().open().await?;
    let child = core.session("child").rlm().parent("main").open().await?;

    println!("parent mode: {}", parent.mode());
    println!("child mode: {}", child.mode());
    Ok(())
}
