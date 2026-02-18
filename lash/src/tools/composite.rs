use std::collections::HashMap;
use std::sync::Arc;

use crate::{ProgressSender, ToolDefinition, ToolProvider, ToolResult};

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
    tool_index: HashMap<String, usize>,
    definitions_cache: Vec<ToolDefinition>,
}

impl CompositeTools {
    pub fn new() -> Self {
        Self {
            providers: Vec::new(),
            tool_index: HashMap::new(),
            definitions_cache: Vec::new(),
        }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn add(mut self, provider: impl ToolProvider) -> Self {
        let defs = provider.definitions();
        let idx = self.providers.len();
        for def in &defs {
            self.tool_index.entry(def.name.clone()).or_insert(idx);
        }
        self.definitions_cache.extend(defs);
        self.providers.push(Box::new(provider));
        self
    }

    /// Add an `Arc<dyn ToolProvider>` (e.g. to share a tool set with sub-agents).
    pub fn add_arc(mut self, provider: Arc<dyn ToolProvider>) -> Self {
        let defs = provider.definitions();
        let idx = self.providers.len();
        for def in &defs {
            self.tool_index.entry(def.name.clone()).or_insert(idx);
        }
        self.definitions_cache.extend(defs);
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
    async fn execute_streaming(
        &self,
        name: &str,
        args: &serde_json::Value,
        progress: Option<&ProgressSender>,
    ) -> ToolResult {
        self.0.execute_streaming(name, args, progress).await
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
        self.definitions_cache.clone()
    }

    async fn execute(&self, name: &str, args: &serde_json::Value) -> ToolResult {
        if let Some(idx) = self.tool_index.get(name)
            && let Some(provider) = self.providers.get(*idx)
        {
            return provider.execute(name, args).await;
        }
        ToolResult::err_fmt(format_args!("Unknown tool: {name}"))
    }

    async fn execute_streaming(
        &self,
        name: &str,
        args: &serde_json::Value,
        progress: Option<&ProgressSender>,
    ) -> ToolResult {
        if let Some(idx) = self.tool_index.get(name)
            && let Some(provider) = self.providers.get(*idx)
        {
            return provider.execute_streaming(name, args, progress).await;
        }
        ToolResult::err_fmt(format_args!("Unknown tool: {name}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ToolParam;

    struct MockProviderA;

    #[async_trait::async_trait]
    impl ToolProvider for MockProviderA {
        fn definitions(&self) -> Vec<ToolDefinition> {
            vec![ToolDefinition {
                name: "tool_a".into(),
                description: "Tool A".into(),
                params: vec![],
                returns: "str".into(),
                hidden: false,
            }]
        }
        async fn execute(&self, _name: &str, _args: &serde_json::Value) -> ToolResult {
            ToolResult::ok(serde_json::json!("result_a"))
        }
    }

    struct MockProviderB;

    #[async_trait::async_trait]
    impl ToolProvider for MockProviderB {
        fn definitions(&self) -> Vec<ToolDefinition> {
            vec![ToolDefinition {
                name: "tool_b".into(),
                description: "Tool B".into(),
                params: vec![ToolParam::typed("x", "int")],
                returns: "int".into(),
                hidden: false,
            }]
        }
        async fn execute(&self, _name: &str, _args: &serde_json::Value) -> ToolResult {
            ToolResult::ok(serde_json::json!(42))
        }
    }

    // ── CompositeTools ──

    #[tokio::test]
    async fn composite_combines_definitions() {
        let tools = CompositeTools::new().add(MockProviderA).add(MockProviderB);
        let defs = tools.definitions();
        assert_eq!(defs.len(), 2);
        let names: Vec<&str> = defs.iter().map(|d| d.name.as_str()).collect();
        assert!(names.contains(&"tool_a"));
        assert!(names.contains(&"tool_b"));
    }

    #[tokio::test]
    async fn composite_routes_execute() {
        let tools = CompositeTools::new().add(MockProviderA).add(MockProviderB);
        let r = tools.execute("tool_a", &serde_json::json!({})).await;
        assert!(r.success);
        assert_eq!(r.result, serde_json::json!("result_a"));

        let r = tools.execute("tool_b", &serde_json::json!({"x": 1})).await;
        assert!(r.success);
        assert_eq!(r.result, serde_json::json!(42));
    }

    #[tokio::test]
    async fn composite_unknown_tool() {
        let tools = CompositeTools::new().add(MockProviderA);
        let r = tools.execute("nonexistent", &serde_json::json!({})).await;
        assert!(!r.success);
    }
}
