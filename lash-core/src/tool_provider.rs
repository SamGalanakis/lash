use std::collections::BTreeMap;
use std::sync::Arc;

use crate::plugin::{DirectCompletion, PluginError, SessionHandle, SessionSnapshot, ToolHookHost};
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolSessionModel {
    pub model: String,
    pub model_variant: Option<String>,
}

#[derive(Clone)]
pub struct ToolSessionControl {
    host: Arc<dyn ToolHookHost>,
}

impl ToolSessionControl {
    pub async fn create_session(
        &self,
        request: crate::SessionCreateRequest,
    ) -> Result<SessionHandle, PluginError> {
        self.host.create_session(request).await
    }

    pub async fn close_session(&self, session_id: &str) -> Result<(), PluginError> {
        self.host.close_session(session_id).await
    }

    pub async fn start_turn_stream(
        &self,
        session_id: &str,
        input: crate::TurnInput,
    ) -> Result<crate::plugin::SessionTurnHandle, PluginError> {
        self.host.start_turn_stream(session_id, input).await
    }

    pub async fn await_turn(&self, turn_id: &str) -> Result<crate::AssembledTurn, PluginError> {
        self.host.await_turn(turn_id).await
    }

    pub async fn cancel_turn(&self, turn_id: &str) -> Result<(), PluginError> {
        self.host.cancel_turn(turn_id).await
    }
}

#[async_trait::async_trait]
impl crate::plugin::SessionLifecycleHost for ToolSessionControl {
    async fn create_session(
        &self,
        request: crate::SessionCreateRequest,
    ) -> Result<SessionHandle, PluginError> {
        ToolSessionControl::create_session(self, request).await
    }

    async fn close_session(&self, session_id: &str) -> Result<(), PluginError> {
        ToolSessionControl::close_session(self, session_id).await
    }
}

#[derive(Clone)]
pub struct ToolTaskControl {
    session_id: String,
    host: Arc<dyn ToolHookHost>,
}

impl ToolTaskControl {
    pub async fn register_background_task(
        &self,
        spec: crate::ManagedTaskSpec,
        cancel: Option<crate::ManagedTaskCancel>,
    ) -> Result<(), PluginError> {
        self.host
            .register_background_task(&self.session_id, spec, cancel)
            .await
    }

    pub async fn unregister_background_task(&self, task_id: &str) {
        self.unregister_background_task_for_session(&self.session_id, task_id)
            .await;
    }

    pub async fn complete_background_task(&self, task_id: &str, run_state: crate::ManagedRunState) {
        self.complete_background_task_for_session(&self.session_id, task_id, run_state)
            .await;
    }

    pub async fn transition_background_task_live_state(
        &self,
        task_id: &str,
        run_state: crate::ManagedRunState,
    ) {
        self.transition_background_task_live_state_for_session(
            &self.session_id,
            task_id,
            run_state,
        )
        .await;
    }

    pub async fn unregister_background_task_for_session(&self, session_id: &str, task_id: &str) {
        self.host
            .unregister_background_task(session_id, task_id)
            .await;
    }

    pub async fn complete_background_task_for_session(
        &self,
        session_id: &str,
        task_id: &str,
        run_state: crate::ManagedRunState,
    ) {
        self.host
            .complete_background_task(session_id, task_id, run_state)
            .await;
    }

    pub async fn transition_background_task_live_state_for_session(
        &self,
        session_id: &str,
        task_id: &str,
        run_state: crate::ManagedRunState,
    ) {
        self.host
            .transition_background_task_live_state(session_id, task_id, run_state)
            .await;
    }
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

    pub async fn session_model(&self) -> Result<ToolSessionModel, PluginError> {
        let snapshot = self.session_snapshot().await?;
        Ok(ToolSessionModel {
            model: snapshot.policy.model,
            model_variant: snapshot.policy.model_variant,
        })
    }

    pub async fn session_snapshot(&self) -> Result<SessionSnapshot, PluginError> {
        self.host.snapshot_session(&self.session_id).await
    }

    pub async fn tool_catalog(&self) -> Result<Vec<serde_json::Value>, PluginError> {
        self.host.tool_catalog(&self.session_id).await
    }

    pub async fn set_tools_availability(
        &self,
        names: &[String],
        availability: Option<crate::ToolAvailability>,
    ) -> Result<u64, PluginError> {
        self.host
            .set_tools_availability(&self.session_id, names, availability)
            .await
    }

    pub fn sessions(&self) -> ToolSessionControl {
        ToolSessionControl {
            host: Arc::clone(&self.host),
        }
    }

    pub fn tasks(&self) -> ToolTaskControl {
        ToolTaskControl {
            session_id: self.session_id.clone(),
            host: Arc::clone(&self.host),
        }
    }

    pub async fn direct_completion(
        &self,
        mut request: crate::DirectRequest,
        usage_source: &str,
    ) -> Result<DirectCompletion, PluginError> {
        if request.session_id.is_none() {
            request.session_id = Some(self.session_id.clone());
        }
        if request.originating_tool_call_id.is_none() {
            request.originating_tool_call_id = self.tool_call_id.clone();
        }
        self.host.direct_completion(request, usage_source).await
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

    /// Shortcut for [`TurnContext::plugin_input`](crate::TurnContext::plugin_input).
    pub fn plugin_input<T: 'static>(&self, plugin_id: &'static str) -> Option<&T> {
        self.turn_context.plugin_input::<T>(plugin_id)
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

    /// Constructor reserved for `lash_core::testing` helpers. Do not call directly;
    /// use [`lash_core::testing::mock_tool_context`] instead.
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
