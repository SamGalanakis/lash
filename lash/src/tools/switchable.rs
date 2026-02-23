use std::sync::{Arc, RwLock};

use crate::{
    DynamicStateSnapshot, ProgressSender, ResolvedProjection, ToolDefinition, ToolProvider,
    ToolResult,
};

#[derive(Clone)]
pub struct SwitchableTools {
    inner: Arc<RwLock<Arc<dyn ToolProvider>>>,
}

impl SwitchableTools {
    pub fn new(initial: Arc<dyn ToolProvider>) -> Self {
        Self {
            inner: Arc::new(RwLock::new(initial)),
        }
    }

    pub fn swap(&self, next: Arc<dyn ToolProvider>) {
        *self.inner.write().expect("switchable tools lock poisoned") = next;
    }

    fn current(&self) -> Arc<dyn ToolProvider> {
        self.inner
            .read()
            .expect("switchable tools lock poisoned")
            .clone()
    }
}

#[async_trait::async_trait]
impl ToolProvider for SwitchableTools {
    fn definitions(&self) -> Vec<ToolDefinition> {
        self.current().definitions()
    }

    fn dynamic_projection(&self) -> Option<ResolvedProjection> {
        self.current().dynamic_projection()
    }

    fn dynamic_snapshot(&self) -> Option<DynamicStateSnapshot> {
        self.current().dynamic_snapshot()
    }

    fn fork_dynamic_with_snapshot(
        &self,
        snapshot: DynamicStateSnapshot,
    ) -> Option<Arc<dyn ToolProvider>> {
        self.current().fork_dynamic_with_snapshot(snapshot)
    }

    fn dynamic_capabilities_payload_json(&self) -> Option<String> {
        self.current().dynamic_capabilities_payload_json()
    }

    fn dynamic_generation(&self) -> Option<u64> {
        self.current().dynamic_generation()
    }

    async fn execute(&self, name: &str, args: &serde_json::Value) -> ToolResult {
        self.current().execute(name, args).await
    }

    async fn execute_streaming(
        &self,
        name: &str,
        args: &serde_json::Value,
        progress: Option<&ProgressSender>,
    ) -> ToolResult {
        self.current().execute_streaming(name, args, progress).await
    }
}
