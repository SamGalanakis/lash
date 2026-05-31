use std::sync::Arc;

use lash_sansio::llm::types::ProviderReplayMeta;
use serde::{Deserialize, Serialize};

use crate::plugin::{PluginError, RuntimeSessionHost, SessionSnapshot};
use crate::{AttachmentStore, ToolContract, ToolManifest, ToolResult};

mod attachments;
mod direct_completion;
mod dispatch;
mod process;
pub(crate) mod process_events;
mod session;

pub use attachments::ToolAttachmentControl;
pub use direct_completion::ToolDirectCompletionControl;
pub use dispatch::ToolDispatchControl;
pub use process::ToolProcessControl;
pub use process_events::ToolProcessEventControl;
pub use session::{ToolSessionControl, ToolSessionModel};

/// A message sent from the sandbox to the host during execution.
#[derive(Clone, Debug)]
pub struct SandboxMessage {
    pub text: String,
    /// "tool_output" or another host-rendered progress event kind.
    pub kind: String,
}

/// Sender for streaming progress messages from tools (e.g. live bash output).
pub type ProgressSender = tokio::sync::mpsc::UnboundedSender<SandboxMessage>;

/// Per-call environment for [`ToolProvider::execute`]. Fields are sealed so
/// the runtime can add capabilities without breaking tool authors.
#[derive(Clone)]
pub struct ToolContext<'run> {
    pub(crate) session_id: String,
    pub(crate) agent_frame_id: crate::AgentFrameId,
    pub(crate) host: Arc<dyn RuntimeSessionHost>,
    pub(crate) processes: Arc<dyn crate::ProcessService>,
    pub(crate) effect_controller: crate::runtime::RuntimeEffectControllerHandle<'run>,
    pub(crate) runtime_dispatch: Option<Arc<crate::tool_dispatch::ToolDispatchContext<'run>>>,
    pub(crate) cancellation_token: Option<tokio_util::sync::CancellationToken>,
    pub(crate) async_process_id: Option<String>,
    pub(crate) process_events: Option<ToolProcessEventContext>,
    pub(crate) attachment_store: Arc<dyn AttachmentStore>,
    pub(crate) direct_completions: crate::DirectCompletionClient<'run>,
    pub(crate) prepared_payload: serde_json::Value,
    /// The id of the in-flight tool call that is invoking this tool.
    pub(crate) tool_call_id: Option<String>,
    pub(crate) attempt_number: u32,
    pub(crate) max_attempts: u32,
    pub(crate) replay_key: Option<String>,
    pub(crate) parent_invocation: Option<crate::RuntimeInvocation>,
}

#[derive(Clone)]
pub(crate) struct ToolProcessEventContext {
    process_id: String,
    registry: Arc<dyn crate::ProcessRegistry>,
    wake_target_scope: Option<crate::ProcessScope>,
    store: Option<Arc<dyn crate::RuntimePersistence>>,
    host: Arc<dyn RuntimeSessionHost>,
}

pub(crate) struct ToolContextBuilder<'run> {
    session_id: String,
    agent_frame_id: crate::AgentFrameId,
    host: Arc<dyn RuntimeSessionHost>,
    processes: Arc<dyn crate::ProcessService>,
    effect_controller: crate::runtime::RuntimeEffectControllerHandle<'run>,
    runtime_dispatch: Option<Arc<crate::tool_dispatch::ToolDispatchContext<'run>>>,
    cancellation_token: Option<tokio_util::sync::CancellationToken>,
    async_process_id: Option<String>,
    process_events: Option<ToolProcessEventContext>,
    attachment_store: Arc<dyn AttachmentStore>,
    direct_completions: crate::DirectCompletionClient<'run>,
    prepared_payload: serde_json::Value,
    tool_call_id: Option<String>,
    parent_invocation: Option<crate::RuntimeInvocation>,
}

impl<'run> ToolContextBuilder<'run> {
    pub(crate) fn from_dispatch(
        dispatch: Arc<crate::tool_dispatch::ToolDispatchContext<'run>>,
    ) -> Self {
        Self {
            session_id: dispatch.session_id.clone(),
            agent_frame_id: dispatch.agent_frame_id.clone(),
            host: Arc::clone(&dispatch.host),
            processes: Arc::clone(&dispatch.processes),
            effect_controller: dispatch.effect_controller.clone(),
            runtime_dispatch: Some(Arc::clone(&dispatch)),
            cancellation_token: None,
            async_process_id: None,
            process_events: None,
            attachment_store: Arc::clone(&dispatch.attachment_store),
            direct_completions: dispatch.direct_completions.clone(),
            prepared_payload: serde_json::Value::Null,
            tool_call_id: None,
            parent_invocation: dispatch.parent_invocation.clone(),
        }
    }

    #[cfg(any(test, feature = "testing"))]
    pub(crate) fn tool_call_id(mut self, tool_call_id: impl Into<Option<String>>) -> Self {
        self.tool_call_id = tool_call_id.into();
        self
    }

    pub(crate) fn prepared_call(mut self, call: &PreparedToolCall) -> Self {
        self.tool_call_id = Some(call.call_id.clone());
        self.prepared_payload = call.prepared_payload.clone();
        self
    }

    pub(crate) fn cancellation_token(
        mut self,
        cancellation_token: Option<tokio_util::sync::CancellationToken>,
    ) -> Self {
        self.cancellation_token = cancellation_token;
        self
    }

    pub(crate) fn async_process(
        mut self,
        process_id: impl Into<String>,
        cancellation_token: tokio_util::sync::CancellationToken,
    ) -> Self {
        self.async_process_id = Some(process_id.into());
        self.cancellation_token = Some(cancellation_token);
        self
    }

    pub(crate) fn process_events(
        mut self,
        process_id: impl Into<String>,
        registry: Arc<dyn crate::ProcessRegistry>,
        wake_target_scope: Option<crate::ProcessScope>,
        store: Option<Arc<dyn crate::RuntimePersistence>>,
    ) -> Self {
        self.process_events = Some(ToolProcessEventContext {
            process_id: process_id.into(),
            registry,
            wake_target_scope,
            store,
            host: Arc::clone(&self.host),
        });
        self
    }

    pub(crate) fn parent_invocation(mut self, metadata: Option<crate::RuntimeInvocation>) -> Self {
        self.parent_invocation = metadata;
        self
    }

    pub(crate) fn build(self) -> ToolContext<'run> {
        ToolContext {
            session_id: self.session_id,
            agent_frame_id: self.agent_frame_id,
            host: self.host,
            processes: self.processes,
            effect_controller: self.effect_controller,
            runtime_dispatch: self.runtime_dispatch,
            cancellation_token: self.cancellation_token,
            async_process_id: self.async_process_id,
            process_events: self.process_events,
            attachment_store: self.attachment_store,
            direct_completions: self.direct_completions,
            prepared_payload: self.prepared_payload,
            tool_call_id: self.tool_call_id,
            attempt_number: 1,
            max_attempts: 1,
            replay_key: None,
            parent_invocation: self.parent_invocation,
        }
    }
}

impl<'run> ToolContext<'run> {
    #[cfg(any(test, feature = "testing"))]
    pub(crate) fn builder(
        session_id: String,
        host: Arc<dyn RuntimeSessionHost>,
        processes: Arc<dyn crate::ProcessService>,
        effect_controller: crate::runtime::RuntimeEffectControllerHandle<'run>,
        attachment_store: Arc<dyn AttachmentStore>,
        direct_completions: crate::DirectCompletionClient<'run>,
    ) -> ToolContextBuilder<'run> {
        ToolContextBuilder {
            session_id,
            agent_frame_id: String::new(),
            host,
            processes,
            effect_controller,
            runtime_dispatch: None,
            cancellation_token: None,
            async_process_id: None,
            process_events: None,
            attachment_store,
            direct_completions,
            prepared_payload: serde_json::Value::Null,
            tool_call_id: None,
            parent_invocation: None,
        }
    }

    pub(crate) fn from_dispatch(
        dispatch: Arc<crate::tool_dispatch::ToolDispatchContext<'run>>,
    ) -> ToolContextBuilder<'run> {
        ToolContextBuilder::from_dispatch(dispatch)
    }

    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    pub fn agent_frame_id(&self) -> &str {
        &self.agent_frame_id
    }

    pub fn sessions(&self) -> ToolSessionControl {
        ToolSessionControl {
            session_id: self.session_id.clone(),
            host: Arc::clone(&self.host),
        }
    }

    pub fn dispatch(&self) -> ToolDispatchControl<'run> {
        ToolDispatchControl {
            context: self.clone(),
        }
    }

    pub fn processes(&self) -> ToolProcessControl<'run> {
        ToolProcessControl {
            session_id: self.session_id.clone(),
            agent_frame_id: self.agent_frame_id.clone(),
            processes: Arc::clone(&self.processes),
            effect_controller: self.effect_controller.clone(),
            parent_invocation: self.parent_invocation.clone(),
            tool_call_id: self.tool_call_id.clone(),
        }
    }

    pub fn direct_completions(&self) -> ToolDirectCompletionControl<'run> {
        ToolDirectCompletionControl {
            session_id: self.session_id.clone(),
            tool_call_id: self.tool_call_id.clone(),
            direct_completions: self.direct_completions.clone(),
        }
    }

    pub fn attachments(&self) -> ToolAttachmentControl {
        ToolAttachmentControl {
            store: Arc::clone(&self.attachment_store),
        }
    }

    pub fn process_events(&self) -> ToolProcessEventControl {
        ToolProcessEventControl {
            context: self.process_events.clone(),
        }
    }

    pub fn cancellation_token(&self) -> Option<&tokio_util::sync::CancellationToken> {
        self.cancellation_token.as_ref()
    }

    pub fn async_process_id(&self) -> Option<&str> {
        self.async_process_id.as_deref()
    }

    pub fn tool_call_id(&self) -> Option<&str> {
        self.tool_call_id.as_deref()
    }

    pub fn prepared_payload(&self) -> &serde_json::Value {
        &self.prepared_payload
    }

    pub fn decode_prepared_payload<T>(&self) -> Result<T, serde_json::Error>
    where
        T: serde::de::DeserializeOwned,
    {
        serde_json::from_value(self.prepared_payload.clone())
    }

    pub fn attempt_number(&self) -> u32 {
        self.attempt_number
    }

    pub fn max_attempts(&self) -> u32 {
        self.max_attempts
    }

    pub fn replay_key(&self) -> Option<&str> {
        self.replay_key.as_deref()
    }

    pub fn with_async_process(
        mut self,
        process_id: impl Into<String>,
        cancellation_token: tokio_util::sync::CancellationToken,
    ) -> Self {
        self.async_process_id = Some(process_id.into());
        self.cancellation_token = Some(cancellation_token);
        self
    }

    #[cfg(any(test, feature = "testing"))]
    #[doc(hidden)]
    pub fn with_process_events_for_testing(
        mut self,
        process_id: impl Into<String>,
        registry: Arc<dyn crate::ProcessRegistry>,
    ) -> Self {
        self.process_events = Some(ToolProcessEventContext {
            process_id: process_id.into(),
            registry,
            wake_target_scope: None,
            store: None,
            host: Arc::clone(&self.host),
        });
        self
    }

    pub(crate) fn with_retry_context(
        mut self,
        tool_name: &str,
        attempt_number: u32,
        max_attempts: u32,
    ) -> Self {
        self.attempt_number = attempt_number.max(1);
        self.max_attempts = max_attempts.max(1);
        self.replay_key = self
            .tool_call_id
            .as_ref()
            .map(|call_id| format!("lash-tool:{}:{call_id}:{tool_name}", self.session_id));
        self
    }

    pub(crate) fn with_prepared_payload(mut self, payload: serde_json::Value) -> Self {
        self.prepared_payload = payload;
        self
    }

    /// Constructor reserved for `lash_core::testing` helpers. Do not call directly;
    /// use [`lash_core::testing::mock_tool_context`] instead.
    #[cfg(any(test, feature = "testing"))]
    #[doc(hidden)]
    pub fn __for_testing(
        session_id: String,
        host: Arc<dyn RuntimeSessionHost>,
        processes: Arc<dyn crate::ProcessService>,
        attachment_store: Arc<dyn AttachmentStore>,
        direct_completions: crate::DirectCompletionClient<'static>,
        tool_call_id: Option<String>,
    ) -> ToolContext<'static> {
        ToolContext::builder(
            session_id,
            host,
            processes,
            crate::runtime::RuntimeEffectControllerHandle::shared(Arc::new(
                crate::InlineRuntimeEffectController::default(),
            )),
            attachment_store,
            direct_completions,
        )
        .tool_call_id(tool_call_id)
        .build()
    }
}

/// Runtime-prepared executable tool call.
///
/// The raw model/provider identity remains visible, but any argument rewrites
/// and provider-owned context projections are frozen before the call crosses a
/// runtime effect or process boundary.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PreparedToolCall {
    pub call_id: String,
    pub tool_name: String,
    pub args: serde_json::Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub replay: Option<ProviderReplayMeta>,
    #[serde(default, skip_serializing_if = "serde_json::Value::is_null")]
    pub prepared_payload: serde_json::Value,
}

impl PreparedToolCall {
    pub fn identity(call: crate::sansio::PendingToolCall) -> Self {
        Self {
            call_id: call.call_id,
            tool_name: call.tool_name,
            args: call.args,
            replay: call.replay,
            prepared_payload: serde_json::Value::Null,
        }
    }

    pub fn from_parts(
        call_id: impl Into<String>,
        tool_name: impl Into<String>,
        args: serde_json::Value,
        replay: Option<ProviderReplayMeta>,
        prepared_payload: serde_json::Value,
    ) -> Self {
        Self {
            call_id: call_id.into(),
            tool_name: tool_name.into(),
            args,
            replay,
            prepared_payload,
        }
    }
}

#[derive(Clone)]
pub struct ToolPrepareContext {
    session_id: String,
    host: Arc<dyn RuntimeSessionHost>,
    turn_context: crate::TurnContext,
    tool_call_id: Option<String>,
}

impl ToolPrepareContext {
    pub(crate) fn new(
        session_id: String,
        host: Arc<dyn RuntimeSessionHost>,
        turn_context: crate::TurnContext,
        tool_call_id: Option<String>,
    ) -> Self {
        Self {
            session_id,
            host,
            turn_context,
            tool_call_id,
        }
    }

    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    pub fn tool_call_id(&self) -> Option<&str> {
        self.tool_call_id.as_deref()
    }

    pub fn turn_context(&self) -> &crate::TurnContext {
        &self.turn_context
    }

    pub fn plugin_input<T>(&self, plugin_id: &'static str) -> Option<&T>
    where
        T: 'static,
    {
        self.turn_context.plugin_input::<T>(plugin_id)
    }

    pub async fn session_snapshot(&self) -> Result<SessionSnapshot, PluginError> {
        self.host.snapshot_session(&self.session_id).await
    }

    pub async fn tool_catalog(&self) -> Result<Vec<serde_json::Value>, PluginError> {
        self.host.tool_catalog(&self.session_id).await
    }

    pub async fn shared_tool_catalog(
        &self,
    ) -> Result<std::sync::Arc<Vec<serde_json::Value>>, PluginError> {
        self.host.shared_tool_catalog(&self.session_id).await
    }
}

/// Inputs handed to [`ToolProvider::prepare_tool_call`].
pub struct ToolPrepareCall<'a> {
    pub pending: crate::sansio::PendingToolCall,
    pub context: &'a ToolPrepareContext,
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
    pub context: &'a ToolContext<'a>,
    pub progress: Option<&'a ProgressSender>,
}

/// Trait for providing tools to the sandbox. Implement this per-project.
///
/// Implementations supply cheap [`ToolManifest`]s, lazily resolved
/// [`ToolContract`]s, and a single
/// [`execute`](Self::execute) method that handles every call. Tools that
/// need session state read it from `call.context`; tools that stream
/// progress send through `call.progress`.
#[async_trait::async_trait]
pub trait ToolProvider: Send + Sync + 'static {
    fn tool_manifests(&self) -> Vec<ToolManifest>;
    fn resolve_manifest(&self, name: &str) -> Option<ToolManifest> {
        self.tool_manifests()
            .into_iter()
            .find(|manifest| manifest.name == name)
    }
    fn resolve_contract(&self, name: &str) -> Option<Arc<ToolContract>>;
    async fn prepare_tool_call(
        &self,
        call: ToolPrepareCall<'_>,
    ) -> Result<PreparedToolCall, ToolResult> {
        Ok(PreparedToolCall::identity(call.pending))
    }
    async fn execute(&self, call: ToolCall<'_>) -> ToolResult;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_context_builder_carries_call_payload_and_cancellation_state() {
        let cancellation = tokio_util::sync::CancellationToken::new();
        let prepared = PreparedToolCall::from_parts(
            "call-1",
            "demo_tool",
            serde_json::json!({ "input": true }),
            None,
            serde_json::json!({ "prepared": true }),
        );

        let context = ToolContext::builder(
            "session-1".to_string(),
            Arc::new(crate::testing::MockSessionManager::default()),
            Arc::new(crate::UnavailableProcessService),
            crate::runtime::RuntimeEffectControllerHandle::shared(Arc::new(
                crate::InlineRuntimeEffectController::default(),
            )),
            Arc::new(crate::InMemoryAttachmentStore::new()),
            crate::DirectCompletionClient::unavailable(
                "direct completions are unavailable in this test context",
            ),
        )
        .prepared_call(&prepared)
        .cancellation_token(Some(cancellation.clone()))
        .async_process("process-1", cancellation.clone())
        .build();

        assert_eq!(context.session_id(), "session-1");
        assert_eq!(context.tool_call_id(), Some("call-1"));
        assert_eq!(
            context.prepared_payload(),
            &serde_json::json!({ "prepared": true })
        );
        assert_eq!(context.async_process_id(), Some("process-1"));
        assert!(context.cancellation_token().is_some());
    }
}
