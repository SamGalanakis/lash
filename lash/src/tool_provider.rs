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

/// Per-call environment for [`ToolProvider::execute`]. Fields are sealed so
/// the runtime can add capabilities without breaking tool authors.
#[derive(Clone)]
pub struct ToolContext {
    pub(crate) session_id: String,
    pub(crate) host: Arc<dyn ToolHookHost>,
    pub(crate) cancellation_token: Option<tokio_util::sync::CancellationToken>,
    pub(crate) async_task_id: Option<String>,
    pub(crate) turn_context: crate::TurnContext,
    /// The id of the in-flight tool call that is invoking this tool. Set by
    /// the runtime tool dispatcher; tools should propagate it onto any
    /// `DirectRequest::originating_tool_call_id` they issue so the trace
    /// renderer can group fan-out LLM calls under the parent tool entry.
    pub(crate) tool_call_id: Option<String>,
}

impl ToolContext {
    pub(crate) fn new(
        session_id: String,
        host: Arc<dyn ToolHookHost>,
        turn_context: crate::TurnContext,
        tool_call_id: Option<String>,
    ) -> Self {
        Self {
            session_id,
            host,
            cancellation_token: None,
            async_task_id: None,
            turn_context,
            tool_call_id,
        }
    }

    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    pub fn host(&self) -> &Arc<dyn ToolHookHost> {
        &self.host
    }

    pub fn cancellation_token(&self) -> Option<&tokio_util::sync::CancellationToken> {
        self.cancellation_token.as_ref()
    }

    pub fn async_task_id(&self) -> Option<&str> {
        self.async_task_id.as_deref()
    }

    pub fn turn_context(&self) -> &crate::TurnContext {
        &self.turn_context
    }

    pub fn tool_call_id(&self) -> Option<&str> {
        self.tool_call_id.as_deref()
    }

    /// Shortcut for [`TurnContext::plugin_context`](crate::TurnContext::plugin_context).
    pub fn plugin_context<T: 'static>(&self, plugin_id: &'static str) -> Option<&T> {
        self.turn_context.plugin_context::<T>(plugin_id)
    }

    pub fn with_async_task(
        mut self,
        task_id: impl Into<String>,
        cancellation_token: tokio_util::sync::CancellationToken,
    ) -> Self {
        self.async_task_id = Some(task_id.into());
        self.cancellation_token = Some(cancellation_token);
        self
    }

    /// Constructor reserved for `lash::testing` helpers. Do not call directly;
    /// use [`lash::testing::mock_tool_context`] instead.
    #[cfg(any(test, feature = "testing"))]
    #[doc(hidden)]
    pub fn __for_testing(
        session_id: String,
        host: Arc<dyn ToolHookHost>,
        turn_context: crate::TurnContext,
        tool_call_id: Option<String>,
    ) -> Self {
        Self::new(session_id, host, turn_context, tool_call_id)
    }
}

/// Per-call inputs handed to [`ToolProvider::execute`].
///
/// Fields are `pub` because `ToolCall` is a transient borrow; consumers
/// typically destructure (`let ToolCall { name, args, .. } = call`). The
/// stable surface lives on [`ToolContext`] (sealed) and the runtime's
/// dispatcher, which constructs `ToolCall` values.
pub struct ToolCall<'a> {
    pub name: &'a str,
    pub args: &'a serde_json::Value,
    pub context: &'a ToolContext,
    pub progress: Option<&'a ProgressSender>,
}

/// Trait for providing tools to the sandbox. Implement this per-project.
///
/// Implementations supply a list of [`ToolDefinition`]s and a single
/// [`execute`](Self::execute) method that handles every call. Tools that
/// need session state read it from `call.context`; tools that stream
/// progress send through `call.progress`.
#[async_trait::async_trait]
pub trait ToolProvider: Send + Sync + 'static {
    fn definitions(&self) -> Vec<ToolDefinition>;
    async fn execute(&self, call: ToolCall<'_>) -> ToolResult;
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

    async fn execute(&self, call: ToolCall<'_>) -> ToolResult {
        match self.tools.get(call.name) {
            Some((_, provider_idx)) => self.providers[*provider_idx].0.execute(call).await,
            None => ToolResult::err_fmt(format_args!("Unknown tool: {}", call.name)),
        }
    }
}
