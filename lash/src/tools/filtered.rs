use std::collections::BTreeSet;
use std::sync::Arc;

use crate::{ProgressSender, ToolDefinition, ToolProvider, ToolResult};

#[derive(Clone)]
pub struct FilteredTools {
    inner: Arc<dyn ToolProvider>,
    allowed: Arc<BTreeSet<String>>,
    definitions_cache: Vec<ToolDefinition>,
}

impl FilteredTools {
    pub fn new(inner: Arc<dyn ToolProvider>, allowed: BTreeSet<String>) -> Self {
        let definitions_cache = inner
            .definitions()
            .into_iter()
            .filter(|d| allowed.contains(&d.name))
            .collect();
        Self {
            inner,
            allowed: Arc::new(allowed),
            definitions_cache,
        }
    }

    fn allows(&self, name: &str) -> bool {
        self.allowed.contains(name)
    }
}

#[async_trait::async_trait]
impl ToolProvider for FilteredTools {
    fn definitions(&self) -> Vec<ToolDefinition> {
        self.definitions_cache.clone()
    }

    async fn execute(&self, name: &str, args: &serde_json::Value) -> ToolResult {
        if !self.allows(name) {
            return ToolResult::err_fmt(format_args!("Unknown tool: {name}"));
        }
        self.inner.execute(name, args).await
    }

    async fn execute_streaming(
        &self,
        name: &str,
        args: &serde_json::Value,
        progress: Option<&ProgressSender>,
    ) -> ToolResult {
        if !self.allows(name) {
            return ToolResult::err_fmt(format_args!("Unknown tool: {name}"));
        }
        self.inner.execute_streaming(name, args, progress).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[async_trait::async_trait]
    impl ToolProvider for MockTool {
        fn definitions(&self) -> Vec<ToolDefinition> {
            vec![ToolDefinition {
                name: "probe".into(),
                description: vec![crate::ToolText::new(
                    "probe",
                    [crate::ExecutionMode::Repl, crate::ExecutionMode::Standard],
                )],
                params: vec![],
                returns: "str".into(),
                examples: vec![],
                hidden: false,
                inject_into_prompt: true,
            }]
        }

        async fn execute(&self, _name: &str, _args: &serde_json::Value) -> ToolResult {
            ToolResult::ok(serde_json::json!("ok"))
        }
    }

    struct MockTool;

    #[tokio::test]
    async fn filtered_tools_reject_unlisted_tools() {
        let inner: Arc<dyn ToolProvider> = Arc::new(MockTool);
        let filtered = FilteredTools::new(inner, BTreeSet::from(["probe".to_string()]));

        let result = filtered.execute("other", &serde_json::json!({})).await;

        assert!(!result.success);
        assert_eq!(result.result, serde_json::json!("Unknown tool: other"));
    }
}
