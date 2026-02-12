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
