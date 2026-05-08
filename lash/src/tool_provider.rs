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
    fn dynamic_snapshot(&self) -> Option<crate::dynamic::DynamicStateSnapshot> {
        None
    }
    fn fork_dynamic_with_snapshot(
        &self,
        _snapshot: crate::dynamic::DynamicStateSnapshot,
    ) -> Option<Arc<dyn ToolProvider>> {
        None
    }
    fn dynamic_generation(&self) -> Option<u64> {
        None
    }
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
