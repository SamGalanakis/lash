use std::collections::BTreeMap;
use std::sync::Arc;

use crate::plugin::{
    DirectCompletion, PluginError, RuntimeSessionHost, SessionHandle, SessionSnapshot,
};
use crate::{
    AttachmentCreateMeta, AttachmentRef, AttachmentStore, AttachmentStoreError, ToolContract,
    ToolManifest, ToolResult,
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
pub struct ToolContext {
    pub(crate) session_id: String,
    pub(crate) host: Arc<dyn RuntimeSessionHost>,
    pub(crate) cancellation_token: Option<tokio_util::sync::CancellationToken>,
    pub(crate) async_task_id: Option<String>,
    pub(crate) turn_context: crate::TurnContext,
    pub(crate) attachment_store: Arc<dyn AttachmentStore>,
    /// The id of the in-flight tool call that is invoking this tool. Set by
    /// the runtime tool dispatcher; tools should propagate it onto any
    /// `DirectRequest::originating_tool_call_id` they issue so the trace
    /// renderer can group fan-out LLM calls under the parent tool entry.
    pub(crate) tool_call_id: Option<String>,
    pub(crate) attempt_number: u32,
    pub(crate) max_attempts: u32,
    pub(crate) idempotency_key: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolSessionModel {
    pub model: String,
    pub model_variant: Option<String>,
}

#[derive(Clone)]
pub struct ToolSessionControl {
    host: Arc<dyn RuntimeSessionHost>,
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

    async fn start_turn_stream(
        &self,
        session_id: &str,
        input: crate::TurnInput,
    ) -> Result<crate::plugin::SessionTurnHandle, PluginError> {
        ToolSessionControl::start_turn_stream(self, session_id, input).await
    }

    async fn await_turn(&self, turn_id: &str) -> Result<crate::AssembledTurn, PluginError> {
        ToolSessionControl::await_turn(self, turn_id).await
    }

    async fn cancel_turn(&self, turn_id: &str) -> Result<(), PluginError> {
        ToolSessionControl::cancel_turn(self, turn_id).await
    }
}

#[derive(Clone)]
pub struct ToolTaskControl {
    session_id: String,
    host: Arc<dyn RuntimeSessionHost>,
}

impl ToolTaskControl {
    pub async fn register_background_task(
        &self,
        spec: crate::BackgroundTaskRegistration,
        cancel: Option<crate::LocalBackgroundTaskCancel>,
    ) -> Result<(), PluginError> {
        self.host
            .register_background_task(&self.session_id, spec, cancel)
            .await
    }

    pub async fn unregister_background_task(&self, task_id: &str) {
        self.unregister_background_task_for_session(&self.session_id, task_id)
            .await;
    }

    pub async fn complete_background_task(&self, task_id: &str, state: crate::BackgroundTaskState) {
        self.complete_background_task_for_session(&self.session_id, task_id, state)
            .await;
    }

    pub async fn transition_background_task_live_state(
        &self,
        task_id: &str,
        state: crate::BackgroundTaskState,
    ) {
        self.transition_background_task_live_state_for_session(&self.session_id, task_id, state)
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
        state: crate::BackgroundTaskState,
    ) {
        self.host
            .complete_background_task(session_id, task_id, state)
            .await;
    }

    pub async fn transition_background_task_live_state_for_session(
        &self,
        session_id: &str,
        task_id: &str,
        state: crate::BackgroundTaskState,
    ) {
        self.host
            .transition_background_task_live_state(session_id, task_id, state)
            .await;
    }

    pub async fn validate_async_handles_visible(
        &self,
        handle_ids: &[String],
    ) -> Result<(), PluginError> {
        self.host
            .validate_async_handles_visible(&self.session_id, handle_ids)
            .await
    }

    pub async fn transfer_async_handles_to_session(
        &self,
        successor_session_id: &str,
        handle_ids: &[String],
    ) -> Result<(), PluginError> {
        self.host
            .transfer_async_handles(&self.session_id, successor_session_id, handle_ids)
            .await
    }

    pub async fn cancel_unreferenced_async_handles(
        &self,
        keep_handle_ids: &[String],
    ) -> Result<Vec<crate::BackgroundTaskRecord>, PluginError> {
        self.host
            .cancel_unreferenced_async_handles(&self.session_id, keep_handle_ids)
            .await
    }

    pub async fn cancel_all_background_tasks_for_session(
        &self,
        session_id: &str,
    ) -> Result<Vec<crate::BackgroundTaskRecord>, PluginError> {
        self.host.cancel_all_background_tasks(session_id).await
    }
}

impl ToolContext {
    pub(crate) fn new(
        session_id: String,
        host: Arc<dyn RuntimeSessionHost>,
        turn_context: crate::TurnContext,
        attachment_store: Arc<dyn AttachmentStore>,
        tool_call_id: Option<String>,
    ) -> Self {
        Self {
            session_id,
            host,
            cancellation_token: None,
            async_task_id: None,
            turn_context,
            attachment_store,
            tool_call_id,
            attempt_number: 1,
            max_attempts: 1,
            idempotency_key: None,
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
        self.snapshot_current_session().await
    }

    pub async fn snapshot_current_session(&self) -> Result<SessionSnapshot, PluginError> {
        self.snapshot_session(&self.session_id).await
    }

    pub async fn snapshot_session(
        &self,
        session_id: impl AsRef<str>,
    ) -> Result<SessionSnapshot, PluginError> {
        self.host.snapshot_session(session_id.as_ref()).await
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

    pub fn attempt_number(&self) -> u32 {
        self.attempt_number
    }

    pub fn max_attempts(&self) -> u32 {
        self.max_attempts
    }

    pub fn idempotency_key(&self) -> Option<&str> {
        self.idempotency_key.as_deref()
    }

    pub fn put_attachment(
        &self,
        data: Vec<u8>,
        meta: AttachmentCreateMeta,
    ) -> Result<AttachmentRef, AttachmentStoreError> {
        self.attachment_store.put(data, meta)
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

    /// Constructor reserved for `lash_core::testing` helpers. Do not call directly;
    /// use [`lash_core::testing::mock_tool_context`] instead.
    #[cfg(any(test, feature = "testing"))]
    #[doc(hidden)]
    pub fn __for_testing(
        session_id: String,
        host: Arc<dyn RuntimeSessionHost>,
        turn_context: crate::TurnContext,
        attachment_store: Arc<dyn AttachmentStore>,
        tool_call_id: Option<String>,
    ) -> Self {
        Self::new(
            session_id,
            host,
            turn_context,
            attachment_store,
            tool_call_id,
        )
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
    async fn execute(&self, call: ToolCall<'_>) -> ToolResult;
}

pub(crate) struct CompositeToolProvider {
    tools: std::sync::RwLock<BTreeMap<String, (ToolManifest, usize)>>,
    providers: Vec<(Arc<dyn ToolProvider>, Vec<String>)>,
}

impl CompositeToolProvider {
    pub(crate) fn from_providers(providers: Vec<Arc<dyn ToolProvider>>) -> Self {
        let mut tools = BTreeMap::new();
        let mut entries = Vec::new();
        for provider in providers {
            let tool_names = provider
                .tool_manifests()
                .into_iter()
                .map(|manifest| {
                    let name = manifest.name.clone();
                    tools.insert(name.clone(), (manifest, entries.len()));
                    name
                })
                .collect::<Vec<_>>();
            entries.push((provider, tool_names));
        }
        Self {
            tools: std::sync::RwLock::new(tools),
            providers: entries,
        }
    }
}

#[async_trait::async_trait]
impl ToolProvider for CompositeToolProvider {
    fn tool_manifests(&self) -> Vec<ToolManifest> {
        self.tools
            .read()
            .expect("composite tool provider lock poisoned")
            .values()
            .map(|(manifest, _)| manifest.clone())
            .collect()
    }

    fn resolve_manifest(&self, name: &str) -> Option<ToolManifest> {
        if let Some((manifest, _)) = self
            .tools
            .read()
            .expect("composite tool provider lock poisoned")
            .get(name)
        {
            return Some(manifest.clone());
        }
        for (provider_idx, (provider, _)) in self.providers.iter().enumerate() {
            if let Some(manifest) = provider.resolve_manifest(name) {
                self.tools
                    .write()
                    .expect("composite tool provider lock poisoned")
                    .insert(name.to_string(), (manifest.clone(), provider_idx));
                return Some(manifest);
            }
        }
        None
    }

    fn resolve_contract(&self, name: &str) -> Option<Arc<ToolContract>> {
        let provider_idx = self.resolve_manifest(name).and_then(|_| {
            self.tools
                .read()
                .expect("composite tool provider lock poisoned")
                .get(name)
                .map(|(_, provider_idx)| *provider_idx)
        })?;
        self.providers[provider_idx].0.resolve_contract(name)
    }

    async fn execute(&self, call: ToolCall<'_>) -> ToolResult {
        let provider_idx = self.resolve_manifest(call.name).and_then(|_| {
            self.tools
                .read()
                .expect("composite tool provider lock poisoned")
                .get(call.name)
                .map(|(_, provider_idx)| *provider_idx)
        });
        match provider_idx {
            Some(provider_idx) => self.providers[provider_idx].0.execute(call).await,
            None => ToolResult::err_fmt(format_args!("Unknown tool: {}", call.name)),
        }
    }
}
