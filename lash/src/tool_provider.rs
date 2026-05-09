use std::collections::BTreeMap;
use std::sync::Arc;

use crate::plugin::ToolHookHost;
use crate::{ToolDefinition, ToolResult};

/// A message sent from the sandbox to the host during execution.
#[derive(Clone, Debug)]
pub struct SandboxMessage {
    pub text: String,
    /// "final", "tool_output", or other host-rendered progress events.
    pub kind: String,
}

/// Sender for streaming progress messages from tools (e.g. live bash output).
pub type ProgressSender = tokio::sync::mpsc::UnboundedSender<SandboxMessage>;

#[derive(Clone)]
pub struct ToolExecutionContext {
    pub session_id: String,
    pub host: Arc<dyn ToolHookHost>,
    pub cancellation_token: Option<tokio_util::sync::CancellationToken>,
    pub async_task_id: Option<String>,
    pub turn_context: crate::TurnContext,
    /// The id of the in-flight tool call that is invoking this tool. Set by
    /// the runtime tool dispatcher; tools should propagate it onto any
    /// `DirectRequest::originating_tool_call_id` they issue so the trace
    /// renderer can group fan-out LLM calls under the parent tool entry.
    pub tool_call_id: Option<String>,
}

impl ToolExecutionContext {
    pub fn with_async_task(
        mut self,
        task_id: impl Into<String>,
        cancellation_token: tokio_util::sync::CancellationToken,
    ) -> Self {
        self.async_task_id = Some(task_id.into());
        self.cancellation_token = Some(cancellation_token);
        self
    }
}

/// Trait for providing tools to the sandbox. Implement this per-project.
#[async_trait::async_trait]
pub trait ToolProvider: Send + Sync + 'static {
    fn definitions(&self) -> Vec<ToolDefinition>;
    async fn execute(&self, name: &str, args: &serde_json::Value) -> ToolResult;

    async fn execute_with_context(
        &self,
        name: &str,
        args: &serde_json::Value,
        _context: &ToolExecutionContext,
    ) -> ToolResult {
        self.execute(name, args).await
    }

    /// Execute with progress streaming. Default: delegates to execute().
    async fn execute_streaming(
        &self,
        name: &str,
        args: &serde_json::Value,
        _progress: Option<&ProgressSender>,
    ) -> ToolResult {
        self.execute(name, args).await
    }

    /// Execute with progress streaming and session context. Default: delegates to
    /// `execute_with_context()`.
    async fn execute_streaming_with_context(
        &self,
        name: &str,
        args: &serde_json::Value,
        context: &ToolExecutionContext,
        _progress: Option<&ProgressSender>,
    ) -> ToolResult {
        self.execute_with_context(name, args, context).await
    }
}

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
