//! Compiled source for the Rust snippets on `docs/index.html`.

// docs:start:hello-lash
use std::sync::Arc;

use lash::{LashCore, ModelSpec, TurnInput, provider::ProviderHandle};
use lash_provider_openai::{OPENROUTER_BASE_URL, OpenAiCompatibleProvider};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let api_key = std::env::var("OPENROUTER_API_KEY")?;
    let provider = ProviderHandle::new(
        OpenAiCompatibleProvider::new(api_key, OPENROUTER_BASE_URL).into_components(),
    );

    let model = ModelSpec::from_token_limits("anthropic/claude-sonnet-4.6", None, 200_000, None)
        .map_err(anyhow::Error::msg)?;

    // one LashCore per app, cloned freely.
    let core = lash::LashCore::standard_builder()
        .provider(provider)
        .model(model)
        .effect_host(Arc::new(lash::durability::InlineEffectHost::default()))
        .attachment_store(Arc::new(lash::persistence::InMemoryAttachmentStore::new()))
        .build()?;

    // one session per chat / task; run one turn; read settled prose.
    let session = core.session("hello-1").open().await?;
    let result = session
        .turn(TurnInput::text("Say hi in one short sentence."))
        .run()
        .await?;

    println!("{}", result.assistant_message().unwrap_or_default());
    Ok(())
}
// docs:end:hello-lash
