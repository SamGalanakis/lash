async fn update_session_config_result_is_awaitable(
    session: &lash::LashSession,
    provider: Option<lash::provider::ProviderHandle>,
    model: Option<lash::ModelSpec>,
    prompt: Option<lash::prompt::PromptLayer>,
) -> lash::Result<()> {
    session
        .admin()
        .config()
        .update_session_config(provider, model, prompt)
        .await?;
    Ok(())
}

fn main() {
    let _ = update_session_config_result_is_awaitable;
}
