use std::future::Future;

pub(super) fn model_spec(
    model: impl Into<String>,
    variant: Option<String>,
    context_window_tokens: usize,
) -> lash_core::ModelSpec {
    lash_core::ModelSpec::from_token_limits(model, variant, context_window_tokens, None, None)
        .expect("valid model spec")
}

pub(super) fn mock_model_spec() -> lash_core::ModelSpec {
    model_spec("mock-model", None, 200_000)
}

pub(super) fn test_current_epoch_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or_default()
}

pub(super) fn run_async_test_on_large_stack<F, Fut, T>(name: &str, test: F) -> T
where
    F: FnOnce() -> Fut + Send + 'static,
    Fut: Future<Output = T> + 'static,
    T: Send + 'static,
{
    std::thread::Builder::new()
        .name(name.to_string())
        .stack_size(16 * 1024 * 1024)
        .spawn(|| {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("tokio runtime")
                .block_on(test())
        })
        .expect("spawn large-stack test thread")
        .join()
        .expect("large-stack test thread")
}
