use std::collections::BTreeMap;
use std::sync::Arc;

use crate::{ProgressSender, ToolDefinition, ToolExecutionContext, ToolProvider, ToolResult};

pub(crate) struct CompositeToolProvider {
    tools: BTreeMap<String, (ToolDefinition, usize)>,
    providers: Vec<(Arc<dyn ToolProvider>, Vec<String>)>,
}

impl CompositeToolProvider {
    pub(crate) fn from_providers(providers: Vec<Arc<dyn ToolProvider>>) -> Self {
        let mut tools = BTreeMap::new();
        let mut entries = Vec::new();
        for provider in providers {
            let tool_names = provider
                .definitions()
                .into_iter()
                .map(|def| {
                    let name = def.name.clone();
                    tools.insert(name.clone(), (def, entries.len()));
                    name
                })
                .collect::<Vec<_>>();
            entries.push((provider, tool_names));
        }
        Self {
            tools,
            providers: entries,
        }
    }
}

#[async_trait::async_trait]
impl ToolProvider for CompositeToolProvider {
    fn definitions(&self) -> Vec<ToolDefinition> {
        self.tools.values().map(|(def, _)| def.clone()).collect()
    }

    fn dynamic_generation(&self) -> Option<u64> {
        self.providers
            .iter()
            .filter_map(|(provider, _)| provider.dynamic_generation())
            .max()
    }

    async fn execute(&self, name: &str, args: &serde_json::Value) -> ToolResult {
        match self.tools.get(name) {
            Some((_, provider_idx)) => self.providers[*provider_idx].0.execute(name, args).await,
            None => ToolResult::err_fmt(format_args!("Unknown tool: {name}")),
        }
    }

    async fn execute_with_context(
        &self,
        name: &str,
        args: &serde_json::Value,
        context: &ToolExecutionContext,
    ) -> ToolResult {
        match self.tools.get(name) {
            Some((_, provider_idx)) => {
                self.providers[*provider_idx]
                    .0
                    .execute_with_context(name, args, context)
                    .await
            }
            None => ToolResult::err_fmt(format_args!("Unknown tool: {name}")),
        }
    }

    async fn execute_streaming(
        &self,
        name: &str,
        args: &serde_json::Value,
        progress: Option<&ProgressSender>,
    ) -> ToolResult {
        match self.tools.get(name) {
            Some((_, provider_idx)) => {
                self.providers[*provider_idx]
                    .0
                    .execute_streaming(name, args, progress)
                    .await
            }
            None => ToolResult::err_fmt(format_args!("Unknown tool: {name}")),
        }
    }

    async fn execute_streaming_with_context(
        &self,
        name: &str,
        args: &serde_json::Value,
        context: &ToolExecutionContext,
        progress: Option<&ProgressSender>,
    ) -> ToolResult {
        match self.tools.get(name) {
            Some((_, provider_idx)) => {
                self.providers[*provider_idx]
                    .0
                    .execute_streaming_with_context(name, args, context, progress)
                    .await
            }
            None => ToolResult::err_fmt(format_args!("Unknown tool: {name}")),
        }
    }
}
