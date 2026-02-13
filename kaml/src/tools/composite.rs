use std::sync::Arc;

use crate::{ToolDefinition, ToolProvider, ToolResult};

/// Combines multiple `ToolProvider`s into one.
///
/// ```rust,ignore
/// let tools = CompositeTools::new()
///     .add(WebSearch::new(tavily_key))
///     .add(FetchUrl::new())
///     .add(my_custom_provider);
/// ```
pub struct CompositeTools {
    providers: Vec<Box<dyn ToolProvider>>,
}

impl CompositeTools {
    pub fn new() -> Self {
        Self {
            providers: Vec::new(),
        }
    }

    pub fn add(mut self, provider: impl ToolProvider) -> Self {
        self.providers.push(Box::new(provider));
        self
    }

    /// Add an `Arc<dyn ToolProvider>` (e.g. to share a tool set with sub-agents).
    pub fn add_arc(mut self, provider: Arc<dyn ToolProvider>) -> Self {
        self.providers.push(Box::new(ArcToolProvider(provider)));
        self
    }
}

/// Wrapper so an `Arc<dyn ToolProvider>` can live inside a `Box<dyn ToolProvider>`.
struct ArcToolProvider(Arc<dyn ToolProvider>);

#[async_trait::async_trait]
impl ToolProvider for ArcToolProvider {
    fn definitions(&self) -> Vec<ToolDefinition> {
        self.0.definitions()
    }
    async fn execute(&self, name: &str, args: &serde_json::Value) -> ToolResult {
        self.0.execute(name, args).await
    }
}

impl Default for CompositeTools {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl ToolProvider for CompositeTools {
    fn definitions(&self) -> Vec<ToolDefinition> {
        self.providers
            .iter()
            .flat_map(|p| p.definitions())
            .collect()
    }

    async fn execute(&self, name: &str, args: &serde_json::Value) -> ToolResult {
        for provider in &self.providers {
            if provider.definitions().iter().any(|d| d.name == name) {
                return provider.execute(name, args).await;
            }
        }
        ToolResult {
            success: false,
            result: serde_json::json!(format!("Unknown tool: {name}")),
        }
    }
}
