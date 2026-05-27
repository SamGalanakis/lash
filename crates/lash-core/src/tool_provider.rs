use std::sync::Arc;

use lash_sansio::llm::types::ProviderReplayMeta;
use serde::{Deserialize, Serialize};

use crate::plugin::{
    DirectCompletion, PluginError, RuntimeSessionHost, SessionHandle, SessionSnapshot,
};
use crate::{
    AttachmentCreateMeta, AttachmentRef, AttachmentStore, AttachmentStoreError, ProcessRecord,
    ToolContract, ToolInvocation, ToolInvocationReply, ToolManifest, ToolResult,
};

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
    /// The id of the in-flight tool call that is invoking this tool. Set by
    /// the runtime tool dispatcher; tools should propagate it onto any
    /// `DirectRequest::originating_tool_call_id` they issue so the trace
    /// renderer can group fan-out LLM calls under the parent tool entry.
    pub(crate) tool_call_id: Option<String>,
    pub(crate) attempt_number: u32,
    pub(crate) max_attempts: u32,
    pub(crate) idempotency_key: Option<String>,
    pub(crate) tool_effect_metadata: Option<crate::EffectInvocationMetadata>,
}

#[derive(Clone)]
pub(crate) struct ToolProcessEventContext {
    process_id: String,
    registry: Arc<dyn crate::ProcessRegistry>,
    wake_target_scope_key: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolSessionModel {
    pub model: String,
    pub model_variant: Option<String>,
}

#[derive(Clone)]
pub struct ToolSessionControl {
    session_id: String,
    host: Arc<dyn RuntimeSessionHost>,
}

impl ToolSessionControl {
    pub async fn model(&self) -> Result<ToolSessionModel, PluginError> {
        let snapshot = self.snapshot_current().await?;
        Ok(ToolSessionModel {
            model: snapshot.policy.model.id,
            model_variant: snapshot.policy.model.variant,
        })
    }

    pub async fn snapshot_current(&self) -> Result<SessionSnapshot, PluginError> {
        self.snapshot(&self.session_id).await
    }

    pub async fn snapshot(
        &self,
        session_id: impl AsRef<str>,
    ) -> Result<SessionSnapshot, PluginError> {
        self.host.snapshot_session(session_id.as_ref()).await
    }

    pub async fn create_session(
        &self,
        request: crate::SessionCreateRequest,
    ) -> Result<SessionHandle, PluginError> {
        self.host.create_session(request).await
    }

    pub async fn close_session(&self, session_id: &str) -> Result<(), PluginError> {
        self.host.close_session(session_id).await
    }

    pub async fn start_turn(
        &self,
        session_id: &str,
        input: crate::TurnInput,
    ) -> Result<crate::AssembledTurn, PluginError> {
        self.host.start_turn(session_id, input).await
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
}

#[async_trait::async_trait]
impl RuntimeSessionHost for ToolSessionControl {
    async fn create_session(
        &self,
        request: crate::SessionCreateRequest,
    ) -> Result<SessionHandle, PluginError> {
        ToolSessionControl::create_session(self, request).await
    }

    async fn close_session(&self, session_id: &str) -> Result<(), PluginError> {
        ToolSessionControl::close_session(self, session_id).await
    }

    async fn start_turn(
        &self,
        session_id: &str,
        input: crate::TurnInput,
    ) -> Result<crate::AssembledTurn, PluginError> {
        ToolSessionControl::start_turn(self, session_id, input).await
    }
}

#[derive(Clone)]
pub struct ToolDispatchControl<'run> {
    context: ToolContext<'run>,
}

impl<'run> ToolDispatchControl<'run> {
    pub async fn batch(&self, calls: Vec<ToolInvocation>) -> Vec<ToolInvocationReply> {
        let Some(dispatch) = self.context.runtime_dispatch.clone() else {
            return calls
                .into_iter()
                .map(|_| {
                    ToolInvocationReply::error(serde_json::json!(
                        "tool batch dispatch is unavailable outside runtime tool execution"
                    ))
                })
                .collect();
        };
        let indexed_calls = calls.into_iter().enumerate().collect::<Vec<_>>();
        crate::tool_dispatch::schedule_tool_batch(
            indexed_calls,
            |(index, _)| *index,
            {
                let dispatch = Arc::clone(&dispatch);
                move |(_, call)| {
                    crate::tool_dispatch::resolve_tool_scheduling(&dispatch, &call.name)
                }
            },
            |(_, call)| {
                let dispatch = Arc::clone(&dispatch);
                let mut child_context = self.context.clone();
                async move {
                    child_context.tool_call_id = Some(call.id.clone());
                    child_context.prepared_payload = serde_json::Value::Null;
                    let outcome = crate::tool_dispatch::dispatch_tool_call_with_execution_context(
                        dispatch.as_ref(),
                        call.name,
                        call.args,
                        None,
                        child_context,
                    )
                    .await;
                    let mut record = outcome.record;
                    record.call_id = Some(call.id);
                    ToolInvocationReply::from_output(record.output.clone()).with_record(record)
                }
            },
        )
        .await
    }
}

#[derive(Clone)]
pub struct ToolProcessControl<'run> {
    session_id: String,
    processes: Arc<dyn crate::ProcessService>,
    effect_controller: crate::runtime::RuntimeEffectControllerHandle<'run>,
    effect_metadata: Option<crate::EffectInvocationMetadata>,
}

impl ToolProcessControl<'_> {
    fn process_scope(&self) -> crate::ProcessOpScope<'_> {
        crate::ProcessOpScope::new()
            .with_effect_metadata(self.effect_metadata.clone())
            .with_effect_controller(self.effect_controller.as_controller())
    }

    pub async fn list_handles(&self) -> Result<Vec<crate::ProcessHandleGrantEntry>, PluginError> {
        self.processes
            .list_visible(&self.session_id, self.process_scope())
            .await
    }

    pub async fn validate_handles(&self, handle_ids: &[String]) -> Result<(), PluginError> {
        self.processes
            .validate_visible(&self.session_id, handle_ids)
            .await
    }

    pub async fn cancel(&self, process_id: &str) -> Result<ProcessRecord, PluginError> {
        self.processes
            .cancel(&self.session_id, process_id, self.process_scope())
            .await
    }

    pub async fn cancel_all(&self) -> Result<Vec<ProcessRecord>, PluginError> {
        self.processes
            .cancel_all(&self.session_id, self.process_scope())
            .await
    }

    pub async fn transfer_handles(
        &self,
        to_session_id: &str,
        process_ids: Vec<String>,
    ) -> Result<(), PluginError> {
        self.processes
            .transfer(
                &self.session_id,
                to_session_id,
                process_ids,
                self.process_scope(),
            )
            .await
    }

    pub async fn cancel_unreferenced_handles(
        &self,
        keep_process_ids: Vec<String>,
    ) -> Result<Vec<ProcessRecord>, PluginError> {
        self.processes
            .cancel_unreferenced(&self.session_id, keep_process_ids, self.process_scope())
            .await
    }
}

#[derive(Clone)]
pub struct ToolDirectCompletionControl<'run> {
    session_id: String,
    tool_call_id: Option<String>,
    direct_completions: crate::DirectCompletionClient<'run>,
}

impl ToolDirectCompletionControl<'_> {
    pub async fn complete(
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
        self.direct_completions
            .direct_completion(request, usage_source)
            .await
    }
}

#[derive(Clone)]
pub struct ToolAttachmentControl {
    store: Arc<dyn AttachmentStore>,
}

impl ToolAttachmentControl {
    pub fn put(
        &self,
        data: Vec<u8>,
        meta: AttachmentCreateMeta,
    ) -> Result<AttachmentRef, AttachmentStoreError> {
        self.store.put(data, meta)
    }
}

#[derive(Clone)]
pub struct ToolProcessEventControl {
    context: Option<ToolProcessEventContext>,
}

impl ToolProcessEventControl {
    pub async fn emit(
        &self,
        event_type: impl Into<String>,
        payload: serde_json::Value,
    ) -> Result<crate::ProcessEvent, PluginError> {
        self.emit_request(crate::ProcessEventAppendRequest::new(event_type, payload))
            .await
    }

    pub async fn emit_request(
        &self,
        request: crate::ProcessEventAppendRequest,
    ) -> Result<crate::ProcessEvent, PluginError> {
        let Some(process) = self.context.as_ref() else {
            return Err(PluginError::Session(
                "process event emission is unavailable outside a durable process".to_string(),
            ));
        };
        process
            .registry
            .append_event(
                &process.process_id,
                request.with_optional_wake_target_scope_key(process.wake_target_scope_key.clone()),
            )
            .await
    }
}

impl<'run> ToolContext<'run> {
    pub(crate) fn new(
        session_id: String,
        host: Arc<dyn RuntimeSessionHost>,
        processes: Arc<dyn crate::ProcessService>,
        effect_controller: crate::runtime::RuntimeEffectControllerHandle<'run>,
        attachment_store: Arc<dyn AttachmentStore>,
        direct_completions: crate::DirectCompletionClient<'run>,
        tool_call_id: Option<String>,
    ) -> Self {
        Self {
            session_id,
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
            tool_call_id,
            attempt_number: 1,
            max_attempts: 1,
            idempotency_key: None,
            tool_effect_metadata: None,
        }
    }

    pub fn session_id(&self) -> &str {
        &self.session_id
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
            processes: Arc::clone(&self.processes),
            effect_controller: self.effect_controller.clone(),
            effect_metadata: self.tool_effect_metadata.clone(),
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

    pub fn idempotency_key(&self) -> Option<&str> {
        self.idempotency_key.as_deref()
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

    pub(crate) fn with_cancellation_token(
        mut self,
        cancellation_token: Option<tokio_util::sync::CancellationToken>,
    ) -> Self {
        self.cancellation_token = cancellation_token;
        self
    }

    pub(crate) fn with_runtime_dispatch(
        mut self,
        dispatch: Arc<crate::tool_dispatch::ToolDispatchContext<'run>>,
    ) -> Self {
        self.runtime_dispatch = Some(dispatch);
        self
    }

    pub(crate) fn with_process_events(
        mut self,
        process_id: impl Into<String>,
        registry: Arc<dyn crate::ProcessRegistry>,
        wake_target_scope_key: Option<String>,
    ) -> Self {
        self.process_events = Some(ToolProcessEventContext {
            process_id: process_id.into(),
            registry,
            wake_target_scope_key,
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
        self.idempotency_key = self
            .tool_call_id
            .as_ref()
            .map(|call_id| format!("lash-tool:{}:{call_id}:{tool_name}", self.session_id));
        self
    }

    pub(crate) fn with_tool_effect_metadata(
        mut self,
        metadata: Option<crate::EffectInvocationMetadata>,
    ) -> Self {
        self.tool_effect_metadata = metadata;
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
        ToolContext::new(
            session_id,
            host,
            processes,
            crate::runtime::RuntimeEffectControllerHandle::shared(Arc::new(
                crate::InlineRuntimeEffectController::default(),
            )),
            attachment_store,
            direct_completions,
            tool_call_id,
        )
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
