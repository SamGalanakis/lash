use lash::{LashCore, ModeId, ModePreset};

#[tokio::main]
async fn main() -> lash::Result<()> {
    let core = LashCore::builder()
        .install_mode(ModePreset::standard())
        .install_mode(ModePreset::rlm())
        .default_mode(ModeId::standard())
        .model("example-model", None)
        .max_context_tokens(200_000)
        .build()?;

    let parent = core.session("main").standard().open().await?;
    let child = core.session("child").rlm().parent("main").open().await?;

    println!("parent mode: {}", parent.mode());
    println!("child mode: {}", child.mode());
    Ok(())
}
