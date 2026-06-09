//! Compiled source for the Rust snippets on `docs/quickstart.html`.

// docs:start:hello-lash
use std::sync::Arc;

use lash::{LashCore, TurnInput, provider::ProviderHandle};
use lash_provider_openai::{OPENROUTER_BASE_URL, OpenAiCompatibleProvider};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // build a provider handle. substitute your own creds + base URL.
    let api_key = std::env::var("OPENROUTER_API_KEY")?;
    let provider = ProviderHandle::new(
        OpenAiCompatibleProvider::new(api_key, OPENROUTER_BASE_URL).into_components(),
    );

    // one LashCore per app, cloned freely.
    let core = LashCore::standard()
        .provider(provider)
        .model(
            lash::ModelSpec::from_token_limits("anthropic/claude-sonnet-4.6", None, 200_000, None)
                .expect("valid model metadata"),
        )
        .effect_host(Arc::new(lash::durability::InlineEffectHost::default()))
        .lashlang_artifact_store(Arc::new(
            lash::persistence::InMemoryLashlangArtifactStore::new(),
        ))
        .attachment_store(Arc::new(lash::persistence::InMemoryAttachmentStore::new()))
        .build()?;

    // one session per chat / task.
    let session = core.session("hello-1").open().await?;

    // run one turn; read settled prose from the terminal result.
    let result = session
        .turn(TurnInput::text("Say hi in one short sentence."))
        .run()
        .await?;

    let prose = result.assistant_message().unwrap_or_default();
    println!("{prose}");
    Ok(())
}
// docs:end:hello-lash
