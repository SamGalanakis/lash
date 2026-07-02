//! Compiled sources for the Rust snippets on `docs/architecture/providers.html`.

use std::sync::Arc;

use lash::provider::{
    LlmRequest, LlmResponse, LlmTransportError, Provider, ProviderComponents, ProviderHandle,
    ProviderOptions,
};

// docs:start:admission-window
/// Host-owned admission: bounded in-flight windows per traffic class,
/// wrapped around the provider the host installs. Breakers, AIMD windows,
/// and backpressure metrics slot into the same `complete()` seam.
#[derive(Debug)]
struct AdmissionGate {
    inner: Box<dyn Provider>,
    interactive_slots: Arc<tokio::sync::Semaphore>,
    batch_slots: Arc<tokio::sync::Semaphore>,
}

impl Clone for AdmissionGate {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone_boxed(),
            // Shared handles: every clone admits through the same windows.
            interactive_slots: Arc::clone(&self.interactive_slots),
            batch_slots: Arc::clone(&self.batch_slots),
        }
    }
}

#[async_trait::async_trait]
impl Provider for AdmissionGate {
    async fn complete(&mut self, request: LlmRequest) -> Result<LlmResponse, LlmTransportError> {
        // Class traffic by the session identity the host already owns:
        // deliberate ids for its own pipelines, and a default lane for ids
        // it did not mint (lash-spawned child sessions).
        let lane = if request.scope.session_id.starts_with("batch:") {
            &self.batch_slots
        } else {
            &self.interactive_slots
        };
        // The permit drops on every exit path — success, failure, or a
        // cancelled turn — so an aborted call never leaks a slot.
        let _slot = lane.acquire().await.expect("admission gate closed");
        self.inner.complete(request).await
    }

    // Forward `close()` explicitly: the default impl is a no-op and would
    // silently skip the inner provider's transport shutdown.
    async fn close(&self) -> Result<(), LlmTransportError> {
        self.inner.close().await
    }

    fn kind(&self) -> &'static str {
        self.inner.kind()
    }
    fn options(&self) -> ProviderOptions {
        self.inner.options()
    }
    fn set_options(&mut self, options: ProviderOptions) {
        self.inner.set_options(options);
    }
    fn serialize_config(&self) -> serde_json::Value {
        self.inner.serialize_config()
    }
    fn requires_streaming(&self) -> bool {
        self.inner.requires_streaming()
    }
    fn clone_boxed(&self) -> Box<dyn Provider> {
        Box::new(self.clone())
    }
}
// docs:end:admission-window

fn install_admission_gate(components: ProviderComponents) -> ProviderHandle {
    // docs:start:admission-wrap
    let interactive_slots = Arc::new(tokio::sync::Semaphore::new(8));
    let batch_slots = Arc::new(tokio::sync::Semaphore::new(2));
    let handle = ProviderHandle::new(components.map_provider(|inner| {
        Box::new(AdmissionGate {
            inner,
            interactive_slots,
            batch_slots,
        })
    }));
    // docs:end:admission-wrap
    handle
}
