use std::collections::HashSet;
use std::future::Future;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use lash_sansio::llm::types::ProviderReplayMeta;
use serde::{Deserialize, Serialize};

use crate::plugin::{
    PluginError, SessionGraphService, SessionLifecycleService, SessionSnapshot, SessionStateService,
};
use crate::{AttachmentStore, ToolContract, ToolManifest, ToolResult};

mod attachments;
mod direct_completion;
mod dispatch;
mod process;
pub(crate) mod process_events;
mod session;
mod triggers;

pub use attachments::ToolAttachmentClient;
pub use direct_completion::ToolDirectCompletionClient;
pub use dispatch::ToolDispatchClient;
pub use process::ToolSessionProcessAdmin;
pub use process_events::ToolProcessEventClient;
pub use session::{ToolSessionAdmin, ToolSessionModel};
pub use triggers::ToolTriggerClient;

/// A message sent from the sandbox to the host during execution.
#[derive(Clone, Debug)]
pub struct SandboxMessage {
    pub text: String,
    /// "tool_output" or another host-rendered progress event kind.
    pub kind: String,
}

/// Sender for streaming progress messages from tools (e.g. live bash output).
pub type ProgressSender = tokio::sync::mpsc::UnboundedSender<SandboxMessage>;

#[derive(Clone, Default)]
pub(crate) struct ToolCompletionState {
    key: Arc<Mutex<Option<crate::AwaitEventKey>>>,
}

impl ToolCompletionState {
    fn store(
        &self,
        key: crate::AwaitEventKey,
    ) -> Result<crate::AwaitEventKey, crate::RuntimeError> {
        let mut guard = self.key.lock().map_err(|_| {
            crate::RuntimeError::new(
                "tool_completion_state_poisoned",
                "tool completion key state lock poisoned",
            )
        })?;
        if let Some(existing) = guard.as_ref() {
            return Ok(existing.clone());
        }
        *guard = Some(key.clone());
        Ok(key)
    }

    pub(crate) fn take(&self) -> Result<Option<crate::AwaitEventKey>, crate::RuntimeError> {
        self.key.lock().map(|mut guard| guard.take()).map_err(|_| {
            crate::RuntimeError::new(
                "tool_completion_state_poisoned",
                "tool completion key state lock poisoned",
            )
        })
    }
}

#[derive(Clone, Default)]
pub(crate) struct ToolDurableEffectState {
    step_ids: Arc<Mutex<HashSet<String>>>,
    process_event_sequence: Arc<AtomicU64>,
}

impl ToolDurableEffectState {
    fn reserve_step(&self, step_id: &str) -> Result<(), crate::RuntimeError> {
        let mut guard = self.step_ids.lock().map_err(|_| {
            crate::RuntimeError::new(
                "durable_effect_state_poisoned",
                "durable effect step state lock poisoned",
            )
        })?;
        if !guard.insert(step_id.to_string()) {
            return Err(crate::RuntimeError::new(
                "durable_effect_duplicate_step_id",
                format!("durable effect step id `{step_id}` was already used by this tool call"),
            ));
        }
        Ok(())
    }

    fn next_process_event_sequence(&self) -> u64 {
        self.process_event_sequence.fetch_add(1, Ordering::Relaxed)
    }
}

/// Per-call environment for [`ToolProvider::execute`]. Fields are sealed so
/// the runtime can add capabilities without breaking tool authors.
#[derive(Clone)]
pub struct ToolContext<'run> {
    pub(crate) session_id: String,
    pub(crate) agent_frame_id: crate::AgentFrameId,
    pub(crate) sessions: Arc<dyn SessionStateService>,
    pub(crate) session_lifecycle: Arc<dyn SessionLifecycleService>,
    pub(crate) processes: Arc<dyn crate::ProcessService>,
    pub(crate) process_cancel_ability: Arc<dyn crate::ProcessCancelAbility>,
    pub(crate) effect_controller: crate::runtime::RuntimeEffectControllerHandle<'run>,
    pub(crate) runtime_dispatch: Option<Arc<crate::tool_dispatch::ToolDispatchContext<'run>>>,
    pub(crate) cancellation_token: Option<tokio_util::sync::CancellationToken>,
    pub(crate) async_process_id: Option<String>,
    pub(crate) runtime_process_id: Option<String>,
    pub(crate) process_events: Option<ToolProcessEventContext>,
    pub(crate) attachment_store: Arc<dyn AttachmentStore>,
    pub(crate) direct_completions: crate::DirectCompletionClient<'run>,
    pub(crate) prepared_payload: serde_json::Value,
    /// The id of the in-flight tool call that is invoking this tool.
    pub(crate) tool_call_id: Option<String>,
    pub(crate) attempt_number: u32,
    pub(crate) max_attempts: u32,
    pub(crate) replay_key: Option<String>,
    pub(crate) completion: ToolCompletionState,
    pub(crate) durable_effects: ToolDurableEffectState,
    pub(crate) parent_invocation: Option<crate::RuntimeInvocation>,
    pub(crate) execution_env_spec: crate::ProcessExecutionEnvSpec,
    pub(crate) lashlang_execution_call_site: Option<ToolLashlangExecutionCallSite>,
}

#[derive(Clone)]
pub struct ToolLashlangExecutionCallSite {
    sink: Arc<dyn lash_trace::TraceSink>,
    base_context: lash_trace::TraceContext,
    identity: lash_trace::TraceLashlangExecutionIdentity,
    parent_node_id: String,
    occurrence: u64,
}

impl ToolLashlangExecutionCallSite {
    pub fn new(
        sink: Arc<dyn lash_trace::TraceSink>,
        base_context: lash_trace::TraceContext,
        identity: lash_trace::TraceLashlangExecutionIdentity,
        parent_node_id: impl Into<String>,
        occurrence: u64,
    ) -> Self {
        Self {
            sink,
            base_context,
            identity,
            parent_node_id: parent_node_id.into(),
            occurrence,
        }
    }
}

#[derive(Clone)]
pub(crate) struct ToolProcessEventContext {
    process_id: String,
    registry: Arc<dyn crate::ProcessRegistry>,
    store: Option<Arc<dyn crate::RuntimePersistence>>,
    session_store_factory: Option<Arc<dyn crate::SessionStoreFactory>>,
    session_graph: Arc<dyn SessionGraphService>,
    queued_work_poke: Option<crate::QueuedWorkPoke>,
}

pub(crate) struct ToolContextBuilder<'run> {
    session_id: String,
    agent_frame_id: crate::AgentFrameId,
    sessions: Arc<dyn SessionStateService>,
    session_lifecycle: Arc<dyn SessionLifecycleService>,
    session_graph: Arc<dyn SessionGraphService>,
    processes: Arc<dyn crate::ProcessService>,
    process_cancel_ability: Arc<dyn crate::ProcessCancelAbility>,
    effect_controller: crate::runtime::RuntimeEffectControllerHandle<'run>,
    runtime_dispatch: Option<Arc<crate::tool_dispatch::ToolDispatchContext<'run>>>,
    cancellation_token: Option<tokio_util::sync::CancellationToken>,
    async_process_id: Option<String>,
    runtime_process_id: Option<String>,
    process_events: Option<ToolProcessEventContext>,
    attachment_store: Arc<dyn AttachmentStore>,
    direct_completions: crate::DirectCompletionClient<'run>,
    prepared_payload: serde_json::Value,
    tool_call_id: Option<String>,
    completion: ToolCompletionState,
    durable_effects: ToolDurableEffectState,
    parent_invocation: Option<crate::RuntimeInvocation>,
    execution_env_spec: crate::ProcessExecutionEnvSpec,
    lashlang_execution_call_site: Option<ToolLashlangExecutionCallSite>,
}

impl<'run> ToolContextBuilder<'run> {
    pub(crate) fn from_dispatch(
        dispatch: Arc<crate::tool_dispatch::ToolDispatchContext<'run>>,
    ) -> Self {
        Self {
            session_id: dispatch.session_id.clone(),
            agent_frame_id: dispatch.agent_frame_id.clone(),
            sessions: Arc::clone(&dispatch.sessions),
            session_lifecycle: Arc::clone(&dispatch.session_lifecycle),
            session_graph: Arc::clone(&dispatch.session_graph),
            processes: Arc::clone(&dispatch.processes),
            process_cancel_ability: Arc::clone(&dispatch.process_cancel_ability),
            effect_controller: dispatch.effect_controller.clone(),
            runtime_dispatch: Some(Arc::clone(&dispatch)),
            cancellation_token: None,
            async_process_id: None,
            runtime_process_id: None,
            process_events: None,
            attachment_store: Arc::clone(&dispatch.attachment_store),
            direct_completions: dispatch.direct_completions.clone(),
            prepared_payload: serde_json::Value::Null,
            tool_call_id: None,
            completion: ToolCompletionState::default(),
            durable_effects: ToolDurableEffectState::default(),
            parent_invocation: dispatch.parent_invocation.clone(),
            execution_env_spec: dispatch.execution_env_spec.clone(),
            lashlang_execution_call_site: None,
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

    pub(crate) fn runtime_process_id(mut self, process_id: Option<String>) -> Self {
        self.runtime_process_id = process_id;
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
        store: Option<Arc<dyn crate::RuntimePersistence>>,
        session_store_factory: Option<Arc<dyn crate::SessionStoreFactory>>,
        queued_work_poke: Option<crate::QueuedWorkPoke>,
    ) -> Self {
        self.process_events = Some(ToolProcessEventContext {
            process_id: process_id.into(),
            registry,
            store,
            session_store_factory,
            session_graph: Arc::clone(&self.session_graph),
            queued_work_poke,
        });
        self
    }

    pub(crate) fn parent_invocation(mut self, metadata: Option<crate::RuntimeInvocation>) -> Self {
        self.parent_invocation = metadata;
        self
    }

    pub(crate) fn lashlang_execution_call_site(
        mut self,
        call_site: Option<ToolLashlangExecutionCallSite>,
    ) -> Self {
        self.lashlang_execution_call_site = call_site;
        self
    }

    pub(crate) fn build(self) -> ToolContext<'run> {
        ToolContext {
            session_id: self.session_id,
            agent_frame_id: self.agent_frame_id,
            sessions: self.sessions,
            session_lifecycle: self.session_lifecycle,
            processes: self.processes,
            process_cancel_ability: self.process_cancel_ability,
            effect_controller: self.effect_controller,
            runtime_dispatch: self.runtime_dispatch,
            cancellation_token: self.cancellation_token,
            async_process_id: self.async_process_id,
            runtime_process_id: self.runtime_process_id,
            process_events: self.process_events,
            attachment_store: self.attachment_store,
            direct_completions: self.direct_completions,
            prepared_payload: self.prepared_payload,
            tool_call_id: self.tool_call_id,
            attempt_number: 1,
            max_attempts: 1,
            replay_key: None,
            completion: self.completion,
            durable_effects: self.durable_effects,
            parent_invocation: self.parent_invocation,
            execution_env_spec: self.execution_env_spec,
            lashlang_execution_call_site: self.lashlang_execution_call_site,
        }
    }
}

impl<'run> ToolContext<'run> {
    #[cfg(any(test, feature = "testing"))]
    #[expect(
        clippy::too_many_arguments,
        reason = "testing constructor mirrors the sealed runtime tool context dependencies"
    )]
    pub(crate) fn builder(
        session_id: String,
        sessions: Arc<dyn SessionStateService>,
        session_lifecycle: Arc<dyn SessionLifecycleService>,
        session_graph: Arc<dyn SessionGraphService>,
        processes: Arc<dyn crate::ProcessService>,
        process_cancel_ability: Arc<dyn crate::ProcessCancelAbility>,
        effect_controller: crate::runtime::RuntimeEffectControllerHandle<'run>,
        attachment_store: Arc<dyn AttachmentStore>,
        direct_completions: crate::DirectCompletionClient<'run>,
    ) -> ToolContextBuilder<'run> {
        ToolContextBuilder {
            session_id,
            agent_frame_id: String::new(),
            sessions,
            session_lifecycle,
            session_graph,
            processes,
            process_cancel_ability,
            effect_controller,
            runtime_dispatch: None,
            cancellation_token: None,
            async_process_id: None,
            runtime_process_id: None,
            process_events: None,
            attachment_store,
            direct_completions,
            prepared_payload: serde_json::Value::Null,
            tool_call_id: None,
            completion: ToolCompletionState::default(),
            durable_effects: ToolDurableEffectState::default(),
            parent_invocation: None,
            execution_env_spec: crate::ProcessExecutionEnvSpec::new(
                crate::PluginOptions::default(),
                crate::SessionPolicy::default(),
            ),
            lashlang_execution_call_site: None,
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

    pub fn sessions(&self) -> ToolSessionAdmin<'run> {
        ToolSessionAdmin {
            session_id: self.session_id.clone(),
            sessions: Arc::clone(&self.sessions),
            session_lifecycle: Arc::clone(&self.session_lifecycle),
            effect_controller: self.effect_controller.clone(),
        }
    }

    pub fn dispatch(&self) -> ToolDispatchClient<'run> {
        ToolDispatchClient {
            context: self.clone(),
        }
    }

    pub fn triggers(&self) -> ToolTriggerClient<'run> {
        ToolTriggerClient {
            context: self.clone(),
        }
    }

    pub fn processes(&self) -> ToolSessionProcessAdmin<'run> {
        ToolSessionProcessAdmin {
            session_id: self.session_id.clone(),
            agent_frame_id: self.agent_frame_id.clone(),
            processes: Arc::clone(&self.processes),
            process_cancel_ability: Arc::clone(&self.process_cancel_ability),
            effect_controller: self.effect_controller.clone(),
            parent_invocation: self.parent_invocation.clone(),
            tool_call_id: self.tool_call_id.clone(),
            execution_env_spec: self.execution_env_spec.clone(),
        }
    }

    pub fn emit_lashlang_child_process_started(
        &self,
        process_id: impl Into<String>,
        child_entry_name: Option<String>,
    ) {
        let Some(call_site) = &self.lashlang_execution_call_site else {
            return;
        };
        let child = lash_trace::TraceLashlangChildExecution {
            scope: call_site.identity.scope.clone(),
            subject: lash_trace::TraceRuntimeSubject::Process {
                process_id: process_id.into(),
            },
            module_ref: None,
            entry_ref: None,
            entry_name: child_entry_name,
        };
        let child_graph_key = child.graph_key();
        let event = lash_trace::TraceLashlangExecutionEvent::ChildStarted {
            event_key: format!(
                "lashlang_execution:{}:child:{}:{}:{}",
                call_site.identity.graph_key(),
                call_site.parent_node_id,
                call_site.occurrence,
                child_graph_key
            ),
            identity: call_site.identity.clone(),
            parent_node_id: call_site.parent_node_id.clone(),
            occurrence: call_site.occurrence,
            child,
        };
        let mut context = lash_trace::TraceContext::default()
            .for_session(call_site.identity.scope.session_id.clone());
        if let Some(turn_id) = &call_site.identity.scope.turn_id {
            context = context.for_turn(turn_id.clone());
        }
        if let Some(turn_index) = call_site.identity.scope.turn_index {
            context = context.for_turn_index(turn_index);
        }
        if let Some(protocol_iteration) = call_site.identity.scope.protocol_iteration {
            context = context.for_protocol_iteration(protocol_iteration);
        }
        if let lash_trace::TraceRuntimeSubject::Effect { effect_id, .. } =
            &call_site.identity.subject
        {
            context.effect_id = Some(effect_id.clone());
        }
        crate::trace::emit_trace(
            &Some(Arc::clone(&call_site.sink)),
            &call_site.base_context,
            context,
            lash_trace::TraceEvent::LashlangExecution { event },
        );
    }

    pub fn direct_completions(&self) -> ToolDirectCompletionClient<'run> {
        ToolDirectCompletionClient {
            session_id: self.session_id.clone(),
            tool_call_id: self.tool_call_id.clone(),
            direct_completions: self.direct_completions.clone(),
        }
    }

    pub fn attachments(&self) -> ToolAttachmentClient {
        ToolAttachmentClient {
            store: Arc::clone(&self.attachment_store),
        }
    }

    pub fn process_events(&self) -> ToolProcessEventClient {
        ToolProcessEventClient {
            context: self.process_events.clone(),
        }
    }

    /// Borrow this tool call's durable effect boundary.
    ///
    /// This is available only while executing a prepared tool call under a
    /// controller that explicitly supports durable tool effects. The returned
    /// facade records JSON steps and await-event waits in the caller's existing
    /// effect log; it does not expose the underlying workflow engine context.
    pub fn durable_effects(&self) -> Result<ToolDurableEffects<'_, 'run>, crate::RuntimeError> {
        let Some(tool_call_id) = self.tool_call_id.as_deref() else {
            return Err(crate::RuntimeError::new(
                "durable_effects_missing_call_id",
                "durable effects require a prepared tool call id",
            ));
        };
        if tool_call_id.trim().is_empty() {
            return Err(crate::RuntimeError::new(
                "durable_effects_missing_call_id",
                "durable effects require a non-empty prepared tool call id",
            ));
        }
        let scoped = self.effect_controller.scoped();
        if !scoped.controller().supports_durable_effects() {
            return Err(crate::RuntimeError::new(
                "durable_effects_unavailable",
                "this effect controller does not support durable tool effects",
            ));
        }
        Ok(ToolDurableEffects { context: self })
    }

    pub fn cancellation_token(&self) -> Option<&tokio_util::sync::CancellationToken> {
        self.cancellation_token.as_ref()
    }

    pub fn async_process_id(&self) -> Option<&str> {
        self.async_process_id.as_deref()
    }

    pub fn runtime_process_id(&self) -> Option<&str> {
        self.async_process_id
            .as_deref()
            .or(self.runtime_process_id.as_deref())
            .or_else(|| {
                self.process_events
                    .as_ref()
                    .map(|context| context.process_id.as_str())
            })
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

    /// Obtain the durable completion key for this call, required before returning
    /// [`ToolResult::Pending`](crate::ToolResult::Pending).
    ///
    /// A tool that defers its outcome (waiting on a webhook, human approval, or another
    /// service) calls this, hands the returned [`AwaitEventKey`](crate::AwaitEventKey)
    /// to whatever will complete the work out-of-band, and then returns
    /// `ToolResult::Pending(..)`. The key names the durable wait the runtime parks the
    /// call on; the external resolver delivers the result against it later.
    ///
    /// The key is stored on the context and consumed by the dispatcher when the tool
    /// returns `Pending`. Returning `Pending` without first calling this fails the call
    /// with `pending_tool_missing_completion_key`. Calls made outside a prepared tool
    /// invocation (no tool call id) fail with `tool_completion_key_missing_call_id`.
    pub async fn completion_key(&self) -> Result<crate::AwaitEventKey, crate::RuntimeError> {
        let tool_call_id = self.tool_call_id.clone().ok_or_else(|| {
            crate::RuntimeError::new(
                "tool_completion_key_missing_call_id",
                "completion keys require a prepared tool call id",
            )
        })?;
        let scoped = self.effect_controller.scoped();
        let key = scoped
            .controller()
            .await_event_key(
                scoped.execution_scope(),
                crate::AwaitEventWaitIdentity::tool_completion(tool_call_id),
            )
            .await?;
        self.completion.store(key)
    }

    pub(crate) fn take_completion_key(
        &self,
    ) -> Result<Option<crate::AwaitEventKey>, crate::RuntimeError> {
        self.completion.take()
    }

    pub fn with_async_process(
        mut self,
        process_id: impl Into<String>,
        cancellation_token: tokio_util::sync::CancellationToken,
    ) -> Self {
        self.async_process_id = Some(process_id.into());
        self.runtime_process_id = self.async_process_id.clone();
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
            store: None,
            session_store_factory: None,
            session_graph: Arc::new(crate::plugin::NoopSessionManager),
            queued_work_poke: None,
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
    #[expect(
        clippy::too_many_arguments,
        reason = "test-only constructor mirrors the sealed runtime tool context"
    )]
    pub fn __for_testing(
        session_id: String,
        sessions: Arc<dyn SessionStateService>,
        session_lifecycle: Arc<dyn SessionLifecycleService>,
        session_graph: Arc<dyn SessionGraphService>,
        processes: Arc<dyn crate::ProcessService>,
        attachment_store: Arc<dyn AttachmentStore>,
        direct_completions: crate::DirectCompletionClient<'static>,
        tool_call_id: Option<String>,
    ) -> ToolContext<'static> {
        ToolContext::builder(
            session_id,
            sessions,
            session_lifecycle,
            session_graph,
            processes,
            Arc::new(crate::DefaultProcessCancelAbility),
            crate::runtime::RuntimeEffectControllerHandle::shared(Arc::new(
                crate::InlineRuntimeEffectController,
            )),
            attachment_store,
            direct_completions,
        )
        .tool_call_id(tool_call_id)
        .build()
    }

    /// Constructor reserved for tests that need a custom process-cancel host
    /// ability. Do not call directly; prefer public testing helpers when they
    /// cover the case.
    #[cfg(any(test, feature = "testing"))]
    #[doc(hidden)]
    #[expect(
        clippy::too_many_arguments,
        reason = "test-only constructor mirrors the sealed runtime context"
    )]
    pub fn __for_testing_with_process_cancel_ability(
        session_id: String,
        sessions: Arc<dyn SessionStateService>,
        session_lifecycle: Arc<dyn SessionLifecycleService>,
        session_graph: Arc<dyn SessionGraphService>,
        processes: Arc<dyn crate::ProcessService>,
        process_cancel_ability: Arc<dyn crate::ProcessCancelAbility>,
        attachment_store: Arc<dyn AttachmentStore>,
        direct_completions: crate::DirectCompletionClient<'static>,
        tool_call_id: Option<String>,
    ) -> ToolContext<'static> {
        ToolContext::builder(
            session_id,
            sessions,
            session_lifecycle,
            session_graph,
            processes,
            process_cancel_ability,
            crate::runtime::RuntimeEffectControllerHandle::shared(Arc::new(
                crate::InlineRuntimeEffectController,
            )),
            attachment_store,
            direct_completions,
        )
        .tool_call_id(tool_call_id)
        .build()
    }
}

/// Durable effect operations available to advanced in-process tools.
///
/// The facade borrows the caller's existing runtime effect boundary. It records
/// JSON-only local steps and awaits host-signed event keys without exposing
/// Restate, Temporal, or any other workflow-native context.
pub struct ToolDurableEffects<'ctx, 'run> {
    context: &'ctx ToolContext<'run>,
}

impl<'ctx, 'run> ToolDurableEffects<'ctx, 'run> {
    pub async fn run_json<F, Fut>(
        &self,
        step_id: impl Into<String>,
        input: serde_json::Value,
        run: F,
    ) -> Result<serde_json::Value, crate::RuntimeError>
    where
        F: FnOnce(serde_json::Value) -> Fut + Send + 'run,
        Fut: Future<Output = Result<serde_json::Value, crate::RuntimeError>> + Send + 'run,
    {
        let step_id = step_id.into();
        if step_id.trim().is_empty() {
            return Err(crate::RuntimeError::new(
                "durable_effect_empty_step_id",
                "durable effect step id must be non-empty",
            ));
        }
        self.context.durable_effects.reserve_step(&step_id)?;
        let invocation = self.step_invocation(
            format!("durable-step:{step_id}"),
            crate::RuntimeEffectKind::DurableStep,
            format!("durable-step:{step_id}"),
        )?;
        let outcome = self
            .context
            .effect_controller
            .controller()
            .execute_effect(
                crate::RuntimeEffectEnvelope::new(
                    invocation,
                    crate::RuntimeEffectCommand::DurableStep {
                        step_id,
                        input: input.clone(),
                    },
                ),
                crate::RuntimeEffectLocalExecutor::durable_step(run),
            )
            .await
            .map_err(crate::RuntimeEffectControllerError::into_runtime_error)?;
        outcome
            .into_durable_step()
            .map_err(crate::RuntimeEffectControllerError::into_runtime_error)
    }

    pub async fn external_event_key(
        &self,
        key: impl Into<String>,
    ) -> Result<crate::AwaitEventKey, crate::RuntimeError> {
        let key = key.into();
        if key.trim().is_empty() {
            return Err(crate::RuntimeError::new(
                "durable_effect_empty_event_key",
                "durable effect external event key must be non-empty",
            ));
        }
        let scoped = self.context.effect_controller.scoped();
        scoped
            .controller()
            .await_event_key(
                scoped.execution_scope(),
                crate::AwaitEventWaitIdentity::Custom { key },
            )
            .await
    }

    pub async fn await_event_json(
        &self,
        key: crate::AwaitEventKey,
    ) -> Result<serde_json::Value, crate::RuntimeError> {
        let invocation = self.step_invocation(
            format!("await-event:{}", key.key_id),
            crate::RuntimeEffectKind::AwaitEvent,
            format!("await-event:{}", key.key_id),
        )?;
        let cancellation = self.context.cancellation_token.clone().unwrap_or_default();
        let outcome = self
            .context
            .effect_controller
            .controller()
            .execute_effect(
                crate::RuntimeEffectEnvelope::new(
                    invocation,
                    crate::RuntimeEffectCommand::AwaitEvent { key },
                ),
                crate::RuntimeEffectLocalExecutor::await_event(cancellation, None),
            )
            .await
            .map_err(crate::RuntimeEffectControllerError::into_runtime_error)?;
        match outcome
            .into_await_event()
            .map_err(crate::RuntimeEffectControllerError::into_runtime_error)?
        {
            crate::Resolution::Ok(value) => Ok(value),
            crate::Resolution::Err(err) => Err(crate::RuntimeError::new(err.code, err.message)),
            crate::Resolution::Timeout => Err(crate::RuntimeError::new(
                "durable_effect_event_timeout",
                "durable effect external event wait timed out",
            )),
            crate::Resolution::Cancelled => Err(crate::RuntimeError::new(
                "durable_effect_event_cancelled",
                "durable effect external event wait was cancelled",
            )),
        }
    }

    pub async fn emit_process_event(
        &self,
        event_type: impl Into<String>,
        payload: serde_json::Value,
    ) -> Result<crate::ProcessEvent, crate::RuntimeError> {
        let Some(process) = self.context.process_events.as_ref() else {
            return Err(crate::RuntimeError::new(
                "durable_effect_process_event_unavailable",
                "durable effect process events are unavailable outside a durable process",
            ));
        };
        let event_type = event_type.into();
        if event_type.trim().is_empty() {
            return Err(crate::RuntimeError::new(
                "durable_effect_empty_process_event_type",
                "durable effect process event type must be non-empty",
            ));
        }
        let tool_call_id = self.context.tool_call_id.as_deref().ok_or_else(|| {
            crate::RuntimeError::new(
                "durable_effects_missing_call_id",
                "durable effects require a prepared tool call id",
            )
        })?;
        let sequence = self.context.durable_effects.next_process_event_sequence();
        let request = crate::ProcessEventAppendRequest::new(event_type, payload).with_replay_key(
            format!("tool:{tool_call_id}:durable-process-event:{sequence}"),
        );
        self.context
            .process_events()
            .emit_request(request)
            .await
            .map_err(|err| {
                crate::RuntimeError::new(
                    "durable_effect_process_event_append_failed",
                    err.to_string(),
                )
            })
            .and_then(|event| {
                if event.process_id == process.process_id {
                    Ok(event)
                } else {
                    Err(crate::RuntimeError::new(
                        "durable_effect_process_event_process_mismatch",
                        "process event append returned an event for a different process",
                    ))
                }
            })
    }

    fn step_invocation(
        &self,
        effect_id_suffix: impl Into<String>,
        kind: crate::RuntimeEffectKind,
        replay_suffix: impl AsRef<str>,
    ) -> Result<crate::RuntimeInvocation, crate::RuntimeError> {
        let tool_call_id = self.context.tool_call_id.as_deref().ok_or_else(|| {
            crate::RuntimeError::new(
                "durable_effects_missing_call_id",
                "durable effects require a prepared tool call id",
            )
        })?;
        let effect_id_suffix = effect_id_suffix.into();
        if let Some(parent) = self.context.parent_invocation.as_ref() {
            return Ok(crate::runtime::causal::child_effect_invocation(
                parent,
                format!("{tool_call_id}:{effect_id_suffix}"),
                kind,
                replay_suffix,
            ));
        }
        let scoped = self.context.effect_controller.scoped();
        let replay_key = format!(
            "{}:tool:{tool_call_id}:{}",
            scoped.scope_id(),
            replay_suffix.as_ref()
        );
        Ok(crate::RuntimeInvocation::effect(
            crate::RuntimeScope::new(self.context.session_id.clone()),
            format!("{tool_call_id}:{effect_id_suffix}"),
            kind,
            replay_key,
        ))
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
    sessions: Arc<dyn SessionStateService>,
    turn_context: crate::TurnContext,
    tool_call_id: Option<String>,
}

impl ToolPrepareContext {
    pub(crate) fn new(
        session_id: String,
        sessions: Arc<dyn SessionStateService>,
        turn_context: crate::TurnContext,
        tool_call_id: Option<String>,
    ) -> Self {
        Self {
            session_id,
            sessions,
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
        self.sessions.snapshot_session(&self.session_id).await
    }

    pub async fn tool_catalog(&self) -> Result<Vec<serde_json::Value>, PluginError> {
        self.sessions.tool_catalog(&self.session_id).await
    }

    pub async fn shared_tool_catalog(
        &self,
    ) -> Result<std::sync::Arc<Vec<serde_json::Value>>, PluginError> {
        self.sessions.shared_tool_catalog(&self.session_id).await
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
    use crate::ProcessRegistry;
    use crate::RuntimeEffectController;
    use std::sync::atomic::{AtomicU64, Ordering};

    struct NoDurableEffectController;

    #[async_trait::async_trait]
    impl crate::RuntimeEffectController for NoDurableEffectController {
        async fn execute_effect(
            &self,
            _envelope: crate::RuntimeEffectEnvelope,
            _local_executor: crate::RuntimeEffectLocalExecutor<'_>,
        ) -> Result<crate::RuntimeEffectOutcome, crate::RuntimeEffectControllerError> {
            Err(crate::RuntimeEffectControllerError::new(
                "unexpected_effect",
                "test controller should not execute effects",
            ))
        }
    }

    fn test_context_with_controller(
        tool_call_id: Option<String>,
        controller: Arc<dyn crate::RuntimeEffectController>,
    ) -> ToolContext<'static> {
        ToolContext::builder(
            "session-1".to_string(),
            Arc::new(crate::testing::MockSessionManager::default()),
            Arc::new(crate::testing::MockSessionManager::default()),
            Arc::new(crate::testing::MockSessionManager::default()),
            Arc::new(crate::UnavailableProcessService),
            Arc::new(crate::DefaultProcessCancelAbility),
            crate::runtime::RuntimeEffectControllerHandle::shared(controller),
            Arc::new(crate::InMemoryAttachmentStore::new()),
            crate::DirectCompletionClient::unavailable(
                "direct completions are unavailable in this test context",
            ),
        )
        .tool_call_id(tool_call_id)
        .build()
    }

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
            Arc::new(crate::testing::MockSessionManager::default()),
            Arc::new(crate::testing::MockSessionManager::default()),
            Arc::new(crate::UnavailableProcessService),
            Arc::new(crate::DefaultProcessCancelAbility),
            crate::runtime::RuntimeEffectControllerHandle::shared(Arc::new(
                crate::InlineRuntimeEffectController,
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

    #[test]
    fn durable_effects_requires_prepared_call_id_and_supporting_controller() {
        let missing_call =
            test_context_with_controller(None, Arc::new(crate::InlineRuntimeEffectController));
        let err = match missing_call.durable_effects() {
            Ok(_) => panic!("missing prepared tool call id should fail"),
            Err(err) => err,
        };
        assert_eq!(err.code.as_str(), "durable_effects_missing_call_id");

        let unsupported = test_context_with_controller(
            Some("call-1".to_string()),
            Arc::new(NoDurableEffectController),
        );
        let err = match unsupported.durable_effects() {
            Ok(_) => panic!("unsupported controller should fail"),
            Err(err) => err,
        };
        assert_eq!(err.code.as_str(), "durable_effects_unavailable");
    }

    #[tokio::test]
    async fn durable_run_json_executes_and_maps_closure_errors() {
        let context = test_context_with_controller(
            Some("call-run-json".to_string()),
            Arc::new(crate::InlineRuntimeEffectController),
        );
        let durable = context.durable_effects().expect("durable effects");
        let value = durable
            .run_json(
                "create",
                serde_json::json!({ "x": 1 }),
                |input| async move { Ok(serde_json::json!({ "seen": input["x"] })) },
            )
            .await
            .expect("durable step");
        assert_eq!(value, serde_json::json!({ "seen": 1 }));

        let err = durable
            .run_json("fail", serde_json::json!({}), |_| async {
                Err(crate::RuntimeError::new(
                    "durable_step_failed",
                    "step failed",
                ))
            })
            .await
            .expect_err("closure error");
        assert_eq!(err.code.as_str(), "durable_step_failed");
        assert_eq!(err.message, "step failed");
    }

    #[tokio::test]
    async fn durable_run_json_rejects_empty_or_duplicate_step_ids_before_running() {
        let context = test_context_with_controller(
            Some("call-step-ids".to_string()),
            Arc::new(crate::InlineRuntimeEffectController),
        );
        let durable = context.durable_effects().expect("durable effects");
        let runs = Arc::new(AtomicU64::new(0));

        let err = durable
            .run_json("", serde_json::Value::Null, {
                let runs = Arc::clone(&runs);
                move |_| async move {
                    runs.fetch_add(1, Ordering::Relaxed);
                    Ok(serde_json::Value::Null)
                }
            })
            .await
            .expect_err("empty step id");
        assert_eq!(err.code.as_str(), "durable_effect_empty_step_id");
        assert_eq!(runs.load(Ordering::Relaxed), 0);

        durable
            .run_json("same", serde_json::Value::Null, {
                let runs = Arc::clone(&runs);
                move |_| async move {
                    runs.fetch_add(1, Ordering::Relaxed);
                    Ok(serde_json::Value::Null)
                }
            })
            .await
            .expect("first step");
        let err = durable
            .run_json("same", serde_json::Value::Null, {
                let runs = Arc::clone(&runs);
                move |_| async move {
                    runs.fetch_add(1, Ordering::Relaxed);
                    Ok(serde_json::Value::Null)
                }
            })
            .await
            .expect_err("duplicate step id");
        assert_eq!(err.code.as_str(), "durable_effect_duplicate_step_id");
        assert_eq!(runs.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn durable_external_event_key_is_custom_and_stable() {
        let context = test_context_with_controller(
            Some("call-event-key".to_string()),
            Arc::new(crate::InlineRuntimeEffectController),
        );
        let durable = context.durable_effects().expect("durable effects");
        let first = durable
            .external_event_key("tool-event-stable")
            .await
            .expect("first key");
        let second = durable
            .external_event_key("tool-event-stable")
            .await
            .expect("second key");

        assert_eq!(first, second);
        assert_eq!(
            first.wait,
            crate::AwaitEventWaitIdentity::Custom {
                key: "tool-event-stable".to_string()
            }
        );
    }

    #[tokio::test]
    async fn durable_await_event_json_maps_terminal_resolutions() {
        let controller = Arc::new(crate::InlineRuntimeEffectController);
        let context =
            test_context_with_controller(Some("call-await-event".to_string()), controller.clone());
        let durable = context.durable_effects().expect("durable effects");

        let ok_key = durable
            .external_event_key("tool-event-ok")
            .await
            .expect("ok key");
        controller
            .resolve_await_event(
                &ok_key,
                crate::Resolution::Ok(serde_json::json!({ "answer": 42 })),
            )
            .await
            .expect("resolve ok");
        let value = durable
            .await_event_json(ok_key)
            .await
            .expect("await ok value");
        assert_eq!(value, serde_json::json!({ "answer": 42 }));

        let err_key = durable
            .external_event_key("tool-event-err")
            .await
            .expect("err key");
        controller
            .resolve_await_event(
                &err_key,
                crate::Resolution::Err(crate::ExternalCompletionError::new(
                    "external_bad",
                    "external failed",
                )),
            )
            .await
            .expect("resolve err");
        let err = durable
            .await_event_json(err_key)
            .await
            .expect_err("await err value");
        assert_eq!(err.code.as_str(), "external_bad");

        let cancelled_key = durable
            .external_event_key("tool-event-cancelled")
            .await
            .expect("cancelled key");
        controller
            .resolve_await_event(&cancelled_key, crate::Resolution::Cancelled)
            .await
            .expect("resolve cancelled");
        let err = durable
            .await_event_json(cancelled_key)
            .await
            .expect_err("await cancelled value");
        assert_eq!(err.code.as_str(), "durable_effect_event_cancelled");

        let timeout_key = durable
            .external_event_key("tool-event-timeout")
            .await
            .expect("timeout key");
        controller
            .resolve_await_event(&timeout_key, crate::Resolution::Timeout)
            .await
            .expect("resolve timeout");
        let err = durable
            .await_event_json(timeout_key)
            .await
            .expect_err("await timeout value");
        assert_eq!(err.code.as_str(), "durable_effect_event_timeout");
    }

    #[tokio::test]
    async fn durable_emit_process_event_requires_process_and_appends_inside_process() {
        let context = test_context_with_controller(
            Some("call-no-process".to_string()),
            Arc::new(crate::InlineRuntimeEffectController),
        );
        let err = context
            .durable_effects()
            .expect("durable effects")
            .emit_process_event("tool.event", serde_json::json!({}))
            .await
            .expect_err("outside process");
        assert_eq!(
            err.code.as_str(),
            "durable_effect_process_event_unavailable"
        );

        let registry = Arc::new(crate::TestLocalProcessRegistry::default());
        let process_id = "process:durable-tool-event";
        registry
            .register_process(
                crate::ProcessRegistration::new(
                    process_id,
                    crate::ProcessInput::External {
                        metadata: serde_json::json!({}),
                    },
                    crate::ProcessProvenance::host("test"),
                )
                .with_extra_event_types([crate::ProcessEventType {
                    name: "tool.event".to_string(),
                    payload_schema: crate::LashSchema::any(),
                    semantics: crate::ProcessEventSemanticsSpec::default(),
                }]),
            )
            .await
            .expect("register process");
        let registry_dyn: Arc<dyn crate::ProcessRegistry> = registry;
        let context = test_context_with_controller(
            Some("call-process-event".to_string()),
            Arc::new(crate::InlineRuntimeEffectController),
        )
        .with_process_events_for_testing(process_id, registry_dyn);

        let event = context
            .durable_effects()
            .expect("durable effects")
            .emit_process_event("tool.event", serde_json::json!({ "ok": true }))
            .await
            .expect("process event");
        assert_eq!(event.process_id, process_id);
        assert_eq!(event.event_type, "tool.event");
        assert_eq!(event.payload, serde_json::json!({ "ok": true }));
        assert_eq!(
            event.invocation.replay_key(),
            Some("tool:call-process-event:durable-process-event:0")
        );

        let append_err = context
            .durable_effects()
            .expect("durable effects")
            .emit_process_event("undeclared.event", serde_json::json!({}))
            .await
            .expect_err("undeclared event type must fail the append");
        assert_eq!(
            append_err.code.as_str(),
            "durable_effect_process_event_append_failed"
        );
    }
}
