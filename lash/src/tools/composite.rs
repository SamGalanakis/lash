use std::sync::Arc;

use crate::{ProgressSender, ToolDefinition, ToolProvider, ToolResult};

use super::ToolSet;

/// Thin compatibility wrapper around `ToolSet`.
///
/// ```rust,ignore
/// let tools = CompositeTools::new()
///     .add(WebSearch::new(tavily_key))
///     .add(FetchUrl::new())
///     .add(my_custom_provider);
/// ```
pub struct CompositeTools {
    inner: ToolSet,
}

impl CompositeTools {
    pub fn new() -> Self {
        Self {
            inner: ToolSet::new(),
        }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn add<T: ToolProvider>(mut self, provider: T) -> Self {
        self.inner = self.inner + provider;
        self
    }

    /// Add an `Arc<dyn ToolProvider>` (e.g. to share a tool set with sub-agents).
    pub fn add_arc(mut self, provider: Arc<dyn ToolProvider>) -> Self {
        self.inner = self.inner + provider;
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
        self.inner.definitions()
    }

    async fn execute(&self, name: &str, args: &serde_json::Value) -> ToolResult {
        self.inner.execute(name, args).await
    }

    async fn execute_streaming(
        &self,
        name: &str,
        args: &serde_json::Value,
        progress: Option<&ProgressSender>,
    ) -> ToolResult {
        self.inner.execute_streaming(name, args, progress).await
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
                description: vec![crate::ToolText::new(
                    "Tool A",
                    [
                        crate::ExecutionMode::Repl,
                        crate::ExecutionMode::NativeTools,
                    ],
                )],
                params: vec![],
                returns: "str".into(),
                examples: vec![],
                hidden: false,
                inject_into_prompt: true,
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
                description: vec![crate::ToolText::new(
                    "Tool B",
                    [
                        crate::ExecutionMode::Repl,
                        crate::ExecutionMode::NativeTools,
                    ],
                )],
                params: vec![ToolParam::typed("x", "int")],
                returns: "int".into(),
                examples: vec![],
                hidden: false,
                inject_into_prompt: true,
            }]
        }
        async fn execute(&self, _name: &str, _args: &serde_json::Value) -> ToolResult {
            ToolResult::ok(serde_json::json!(42))
        }
    }

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
