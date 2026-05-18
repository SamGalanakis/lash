use futures_util::FutureExt;
use serde::{Deserialize, Serialize};
use sha2::Digest;
use std::collections::HashMap;
use std::future::Future;
use std::panic::AssertUnwindSafe;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{Mutex, mpsc};
use tokio_util::sync::CancellationToken;

use crate::llm::transport::LlmTransportError;
use crate::llm::types::{LlmRequest, LlmResponse};
use crate::plugin::{DirectCompletion, DirectLlmCompletion, PluginMessage};
use crate::provider::ProviderHandle;
use crate::sansio::{
    CompletedToolCall, EffectId, ExecutionSurfaceSync, LlmCallError, PendingToolCall,
};
use crate::session_model::TokenUsage;
use crate::{
    CheckpointKind, DirectRequest, ExecResponse, LlmRequest as CoreLlmRequest, PluginError,
    RuntimeError,
};

use super::host::{
    BackgroundCancelPolicy, BackgroundTaskCompletion, BackgroundTaskRecord,
    BackgroundTaskRegistration, BackgroundTaskRegistry, BackgroundTaskState,
};
use super::session_manager::{CurrentSessionCapability, UsageCapability};
use super::{RuntimeStreamEvent, RuntimeTurnDriver};

/// Where a runtime effect originated.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EffectOrigin {
    Turn,
    DirectCompletion { usage_source: String },
    DirectLlmCompletion { usage_source: String },
}

/// Durable category for a runtime effect.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeEffectKind {
    LlmCall,
    DirectCompletion,
    DirectLlmCompletion,
    ToolCall,
    ExecCode,
    Checkpoint,
    SyncExecutionSurface,
    Sleep,
}

impl RuntimeEffectKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::LlmCall => "llm_call",
            Self::DirectCompletion => "direct_completion",
            Self::DirectLlmCompletion => "direct_llm_completion",
            Self::ToolCall => "tool_call",
            Self::ExecCode => "exec_code",
            Self::Checkpoint => "checkpoint",
            Self::SyncExecutionSurface => "sync_execution_surface",
            Self::Sleep => "sleep",
        }
    }
}

/// Borrowed durable effect boundary for one runtime execution.
///
/// `RuntimeCoreConfig::effect_host` remains the default in-process fallback.
/// Durable integrations should instead create one scope per externally
/// identified run and pass it to the scoped runtime entrypoints, making the
/// turn identity part of the idempotency contract rather than a tracing-only
/// hint.
#[derive(Clone, Copy)]
pub struct RuntimeEffectScope<'run> {
    host: &'run dyn RuntimeEffectHost,
    turn_id: &'run str,
}

impl<'run> RuntimeEffectScope<'run> {
    pub fn new(
        host: &'run dyn RuntimeEffectHost,
        turn_id: &'run str,
    ) -> Result<Self, RuntimeError> {
        if turn_id.is_empty() {
            return Err(RuntimeError {
                code: "missing_effect_scope_turn_id".to_string(),
                message: "scoped durable runs require a non-empty stable turn_id".to_string(),
            });
        }
        Ok(Self { host, turn_id })
    }

    pub fn host(&self) -> &'run dyn RuntimeEffectHost {
        self.host
    }

    pub fn turn_id(&self) -> &'run str {
        self.turn_id
    }
}

/// Runtime-internal handle for effect-host references carried through
/// per-turn execution contexts.
#[derive(Clone)]
pub(crate) enum RuntimeEffectHostHandle<'run> {
    Borrowed(&'run dyn RuntimeEffectHost),
    Shared(Arc<dyn RuntimeEffectHost>),
}

impl<'run> RuntimeEffectHostHandle<'run> {
    pub(crate) fn borrowed(host: &'run dyn RuntimeEffectHost) -> Self {
        Self::Borrowed(host)
    }

    pub(crate) fn shared(host: Arc<dyn RuntimeEffectHost>) -> Self {
        Self::Shared(host)
    }

    pub(crate) fn as_host(&self) -> &dyn RuntimeEffectHost {
        match self {
            Self::Borrowed(host) => *host,
            Self::Shared(host) => host.as_ref(),
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct RuntimeEffectHostPtr {
    host: *const dyn RuntimeEffectHost,
}

unsafe impl Send for RuntimeEffectHostPtr {}
unsafe impl Sync for RuntimeEffectHostPtr {}

impl RuntimeEffectHostPtr {
    fn new(host: &dyn RuntimeEffectHost) -> Self {
        // SAFETY: the pointer is only dereferenced while the task-local scope
        // that created it is still awaiting. See `as_host` for the matching
        // safety boundary.
        let host = unsafe {
            std::mem::transmute::<
                *const dyn RuntimeEffectHost,
                *const (dyn RuntimeEffectHost + 'static),
            >(host as *const dyn RuntimeEffectHost)
        };
        Self { host }
    }

    fn as_host(self) -> &'static dyn RuntimeEffectHost {
        // SAFETY: `RuntimeEffectHostPtr` values are only installed by
        // `scope_active_effect_host`, which awaits the scoped future before
        // the borrowed host can go out of scope. Tokio task-local values do
        // not propagate into detached spawned tasks, so this pointer is never
        // used as a background-task escape hatch.
        unsafe { &*self.host }
    }
}

#[derive(Clone)]
pub(crate) struct ActiveRuntimeEffectScope {
    host: RuntimeEffectHostPtr,
    turn_id: Arc<str>,
}

impl ActiveRuntimeEffectScope {
    pub(crate) fn host(&self) -> &'static dyn RuntimeEffectHost {
        self.host.as_host()
    }

    pub(crate) fn turn_id(&self) -> &str {
        &self.turn_id
    }
}

tokio::task_local! {
    static ACTIVE_RUNTIME_EFFECT_SCOPE: ActiveRuntimeEffectScope;
}

pub(crate) async fn scope_active_effect_host<F>(
    effect_scope: RuntimeEffectScope<'_>,
    future: F,
) -> F::Output
where
    F: std::future::Future,
{
    ACTIVE_RUNTIME_EFFECT_SCOPE
        .scope(
            ActiveRuntimeEffectScope {
                host: RuntimeEffectHostPtr::new(effect_scope.host()),
                turn_id: Arc::from(effect_scope.turn_id()),
            },
            future,
        )
        .await
}

pub(crate) fn active_effect_scope() -> Option<ActiveRuntimeEffectScope> {
    ACTIVE_RUNTIME_EFFECT_SCOPE.try_with(Clone::clone).ok()
}

/// Serializable metadata attached to every host-run runtime effect.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EffectInvocationMetadata {
    pub session_id: String,
    pub origin: EffectOrigin,
    pub turn_id: Option<String>,
    pub turn_index: Option<usize>,
    pub mode_iteration: Option<usize>,
    pub effect_id: String,
    pub effect_kind: RuntimeEffectKind,
    pub idempotency_key: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn_checkpoint: Option<serde_json::Value>,
}

/// Host-facing invocation context. The cancellation token is intentionally
/// live process state; the rest of the invocation is serializable metadata.
#[derive(Clone)]
pub struct EffectInvocation {
    pub metadata: EffectInvocationMetadata,
    pub cancellation: CancellationToken,
}

impl EffectInvocation {
    pub fn new(metadata: EffectInvocationMetadata, cancellation: CancellationToken) -> Self {
        Self {
            metadata,
            cancellation,
        }
    }
}

/// One-shot executor for the built-in local implementation of turn effects.
///
/// Custom hosts can persist or schedule based on [`EffectInvocation`], then
/// call the typed executor method when they want Lash's default in-process
/// behavior.
pub struct TurnEffectLocalExecutor<'a, 'run> {
    driver: &'a mut RuntimeTurnDriver<'run>,
    machine: &'a mut crate::TurnMachine,
    event_tx: mpsc::Sender<RuntimeStreamEvent>,
    cancellation: CancellationToken,
}

/// One-shot executor for the built-in local implementation of direct effects.
pub struct DirectEffectLocalExecutor {
    current: CurrentSessionCapability,
    usage_capability: UsageCapability,
    provider: ProviderHandle,
}

pub type BackgroundTaskFuture =
    Pin<Box<dyn Future<Output = BackgroundTaskCompletion> + Send + 'static>>;

/// One-shot local executor for a background run container.
pub struct BackgroundTaskExecutor {
    run: Box<dyn FnOnce(CancellationToken) -> BackgroundTaskFuture + Send + 'static>,
}

impl BackgroundTaskExecutor {
    pub fn new<F, Fut>(run: F) -> Self
    where
        F: FnOnce(CancellationToken) -> Fut + Send + 'static,
        Fut: Future<Output = BackgroundTaskCompletion> + Send + 'static,
    {
        Self {
            run: Box::new(move |cancellation| Box::pin(run(cancellation))),
        }
    }

    fn run(self, cancellation: CancellationToken) -> BackgroundTaskFuture {
        (self.run)(cancellation)
    }
}

impl<'a, 'run> TurnEffectLocalExecutor<'a, 'run> {
    pub(in crate::runtime) fn new(
        driver: &'a mut RuntimeTurnDriver<'run>,
        machine: &'a mut crate::TurnMachine,
        event_tx: mpsc::Sender<RuntimeStreamEvent>,
        cancellation: CancellationToken,
    ) -> Self {
        Self {
            driver,
            machine,
            event_tx,
            cancellation,
        }
    }

    pub async fn llm_call(
        self,
        request: Arc<LlmRequest>,
    ) -> (Result<LlmResponse, LlmCallError>, bool) {
        let mode_iteration = self.machine.mode_iteration();
        self.driver
            .run_standard_llm_call(request, mode_iteration, &self.event_tx, &self.cancellation)
            .await
    }

    pub async fn tool_call(
        self,
        call: PendingToolCall,
        metadata: EffectInvocationMetadata,
    ) -> CompletedToolCall {
        let tool_name = call.tool_name.clone();
        let mut results = self.tool_batch(vec![(call, metadata)]).await;
        results
            .pop()
            .unwrap_or_else(|| missing_tool_result_completed_call(tool_name))
    }

    pub(in crate::runtime) async fn tool_batch(
        self,
        calls: Vec<(PendingToolCall, EffectInvocationMetadata)>,
    ) -> Vec<CompletedToolCall> {
        self.driver
            .run_tool_calls(calls, &self.event_tx, &self.cancellation)
            .await
    }

    pub async fn exec_code(self, code: String) -> Result<ExecResponse, String> {
        let mode_iteration = self.machine.mode_iteration();
        let messages = self.machine.message_sequence();
        self.driver
            .run_exec_code(&code, messages, mode_iteration, &self.event_tx)
            .await
    }

    pub async fn checkpoint(
        self,
        checkpoint: CheckpointKind,
    ) -> Result<(Vec<PluginMessage>, Vec<PluginMessage>), RuntimeError> {
        self.driver
            .run_checkpoint(self.machine, checkpoint, &self.event_tx)
            .await
    }

    pub async fn sync_execution_surface(
        self,
        update_machine_config: bool,
    ) -> Result<Option<ExecutionSurfaceSync>, String> {
        self.driver
            .refresh_execution_surface(self.machine, update_machine_config)
            .await
            .map_err(|err| err.to_string())
    }
}

impl DirectEffectLocalExecutor {
    pub(in crate::runtime) fn new(
        current: CurrentSessionCapability,
        usage_capability: UsageCapability,
        provider: ProviderHandle,
    ) -> Self {
        Self {
            current,
            usage_capability,
            provider,
        }
    }

    pub async fn direct_completion(
        &mut self,
        request: DirectRequest,
        normalized_request: CoreLlmRequest,
        model: String,
        usage_source: String,
    ) -> Result<DirectCompletion, PluginError> {
        let originating_tool_call_id = request.originating_tool_call_id.clone();
        let (response, usage) = self
            .run_direct_llm_request(
                normalized_request,
                usage_source,
                model,
                originating_tool_call_id,
            )
            .await?;
        Ok(DirectCompletion {
            text: response.full_text,
            usage,
        })
    }

    pub async fn direct_llm_completion(
        &mut self,
        request: CoreLlmRequest,
        usage_source: String,
    ) -> Result<DirectLlmCompletion, PluginError> {
        let model = request.model.clone();
        let (response, usage) = self
            .run_direct_llm_request(request, usage_source, model, None)
            .await?;
        Ok(DirectLlmCompletion { response, usage })
    }

    async fn run_direct_llm_request(
        &mut self,
        request: CoreLlmRequest,
        usage_source: String,
        usage_model: String,
        originating_tool_call_id: Option<String>,
    ) -> Result<(LlmResponse, TokenUsage), PluginError> {
        let llm_call_id =
            self.emit_direct_llm_trace_started(&request, originating_tool_call_id.as_deref());
        let response = match self.provider.complete(request).await {
            Ok(response) => response,
            Err(err) => {
                self.emit_direct_llm_trace_failed(
                    llm_call_id.as_deref(),
                    originating_tool_call_id.as_deref(),
                    &err,
                );
                return Err(PluginError::Session(err.message.clone()));
            }
        };
        self.emit_direct_llm_trace_completed(
            llm_call_id.as_deref(),
            originating_tool_call_id.as_deref(),
            &response,
        );

        let usage = token_usage_from_llm(&response.usage);
        self.usage_capability
            .record_token_usage(&usage_source, &usage_model, &usage);
        self.usage_capability
            .persist_current_usage_ledger(&self.current)
            .await?;
        Ok((response, usage))
    }

    fn emit_direct_llm_trace_started(
        &self,
        request: &CoreLlmRequest,
        originating_tool_call_id: Option<&str>,
    ) -> Option<String> {
        self.current.host.core.trace_sink.as_ref()?;
        let llm_call_id = uuid::Uuid::new_v4().to_string();
        emit_llm_trace_started(
            &self.current.host.core.trace_sink,
            &self.current.host.core.trace_context,
            self.direct_trace_context(Some(&llm_call_id), originating_tool_call_id),
            request,
        );
        Some(llm_call_id)
    }

    fn emit_direct_llm_trace_completed(
        &self,
        llm_call_id: Option<&str>,
        originating_tool_call_id: Option<&str>,
        response: &LlmResponse,
    ) {
        let Some(llm_call_id) = llm_call_id else {
            return;
        };
        emit_llm_trace_completed(
            &self.current.host.core.trace_sink,
            &self.current.host.core.trace_context,
            self.direct_trace_context(Some(llm_call_id), originating_tool_call_id),
            response,
            0,
            None,
        );
    }

    fn emit_direct_llm_trace_failed(
        &self,
        llm_call_id: Option<&str>,
        originating_tool_call_id: Option<&str>,
        err: &LlmTransportError,
    ) {
        let Some(llm_call_id) = llm_call_id else {
            return;
        };
        emit_llm_trace_failed(
            &self.current.host.core.trace_sink,
            &self.current.host.core.trace_context,
            self.direct_trace_context(Some(llm_call_id), originating_tool_call_id),
            LlmTraceFailure::from(err),
            None,
        );
    }

    fn direct_trace_context(
        &self,
        llm_call_id: Option<&str>,
        originating_tool_call_id: Option<&str>,
    ) -> lash_trace::TraceContext {
        let mut context =
            lash_trace::TraceContext::default().for_session(self.current.session_id.clone());
        if let Some(llm_call_id) = llm_call_id {
            context = context.for_llm_call(llm_call_id.to_string());
        }
        if let Some(originating_tool_call_id) = originating_tool_call_id {
            context = context.for_originating_tool_call(originating_tool_call_id.to_string());
        }
        context
    }
}

fn missing_tool_result_completed_call(tool_name: String) -> CompletedToolCall {
    CompletedToolCall {
        call_id: String::new(),
        tool_name: tool_name.clone(),
        args: serde_json::Value::Null,
        model_return: crate::ModelToolReturn {
            call_id: String::new(),
            tool_name,
            parts: vec![crate::ModelToolReturnPart::Text(
                "[Tool execution failed]\nmissing tool result".to_string(),
            )],
        },
        output: crate::ToolCallOutput::failure(crate::ToolFailure::runtime(
            crate::ToolFailureClass::Internal,
            "tool_result_missing",
            "tool execution completed without a result",
        )),
        duration_ms: 0,
        replay: None,
    }
}

pub(in crate::runtime) fn token_usage_from_llm(usage: &crate::llm::types::LlmUsage) -> TokenUsage {
    TokenUsage {
        input_tokens: usage.input_tokens,
        output_tokens: usage.output_tokens,
        cached_input_tokens: usage.cached_input_tokens,
        reasoning_tokens: usage.reasoning_tokens,
    }
}

pub(in crate::runtime) fn emit_llm_trace_started(
    trace_sink: &Option<Arc<dyn lash_trace::TraceSink>>,
    base_context: &lash_trace::TraceContext,
    context: lash_trace::TraceContext,
    request: &CoreLlmRequest,
) {
    crate::trace::emit_trace(
        trace_sink,
        base_context,
        context,
        lash_trace::TraceEvent::LlmCallStarted {
            request: crate::trace::trace_llm_request(request),
        },
    );
}

pub(in crate::runtime) fn emit_llm_trace_completed(
    trace_sink: &Option<Arc<dyn lash_trace::TraceSink>>,
    base_context: &lash_trace::TraceContext,
    context: lash_trace::TraceContext,
    response: &LlmResponse,
    duration_ms: u64,
    stream_summary: Option<serde_json::Value>,
) {
    crate::trace::emit_trace(
        trace_sink,
        base_context,
        context,
        lash_trace::TraceEvent::LlmCallCompleted {
            response: crate::trace::trace_llm_response(
                response.full_text.clone(),
                duration_ms,
                Some(response.terminal_reason),
                crate::trace::trace_output_parts(&response.parts),
            ),
            usage: Some(crate::trace::trace_usage_from_llm(&response.usage)),
            provider_usage: response.provider_usage.clone(),
            stream_summary,
        },
    );
}

pub(in crate::runtime) struct LlmTraceFailure {
    message: String,
    retryable: bool,
    terminal_reason: crate::LlmTerminalReason,
    code: Option<String>,
    raw: Option<String>,
}

impl From<&LlmTransportError> for LlmTraceFailure {
    fn from(err: &LlmTransportError) -> Self {
        Self {
            message: err.message.clone(),
            retryable: err.retryable,
            terminal_reason: err.terminal_reason,
            code: err.code.clone(),
            raw: err.raw.clone(),
        }
    }
}

impl From<&LlmCallError> for LlmTraceFailure {
    fn from(err: &LlmCallError) -> Self {
        Self {
            message: err.message.clone(),
            retryable: err.retryable,
            terminal_reason: err.terminal_reason,
            code: err.code.clone(),
            raw: err.raw.clone(),
        }
    }
}

pub(in crate::runtime) fn emit_llm_trace_failed(
    trace_sink: &Option<Arc<dyn lash_trace::TraceSink>>,
    base_context: &lash_trace::TraceContext,
    context: lash_trace::TraceContext,
    failure: LlmTraceFailure,
    stream_summary: Option<serde_json::Value>,
) {
    crate::trace::emit_trace(
        trace_sink,
        base_context,
        context,
        lash_trace::TraceEvent::LlmCallFailed {
            error: lash_trace::TraceError {
                message: failure.message,
                retryable: failure.retryable,
                terminal_reason: Some(failure.terminal_reason.code().to_string()),
                code: failure.code,
                raw: failure.raw,
            },
            stream_summary,
        },
    );
}

/// Boundary for nondeterministic runtime work.
#[async_trait::async_trait]
pub trait RuntimeEffectHost: Send + Sync {
    async fn llm_call(
        &self,
        _invocation: EffectInvocation,
        request: Arc<LlmRequest>,
        executor: TurnEffectLocalExecutor<'_, '_>,
    ) -> (Result<LlmResponse, LlmCallError>, bool) {
        executor.llm_call(request).await
    }

    async fn direct_completion(
        &self,
        _invocation: EffectInvocation,
        request: DirectRequest,
        normalized_request: CoreLlmRequest,
        model: String,
        usage_source: String,
        mut executor: DirectEffectLocalExecutor,
    ) -> Result<DirectCompletion, PluginError> {
        executor
            .direct_completion(request, normalized_request, model, usage_source)
            .await
    }

    async fn direct_llm_completion(
        &self,
        _invocation: EffectInvocation,
        request: CoreLlmRequest,
        usage_source: String,
        mut executor: DirectEffectLocalExecutor,
    ) -> Result<DirectLlmCompletion, PluginError> {
        executor.direct_llm_completion(request, usage_source).await
    }

    async fn tool_call(
        &self,
        invocation: EffectInvocation,
        call: PendingToolCall,
        executor: TurnEffectLocalExecutor<'_, '_>,
    ) -> CompletedToolCall {
        executor.tool_call(call, invocation.metadata).await
    }

    async fn exec_code(
        &self,
        _invocation: EffectInvocation,
        code: String,
        executor: TurnEffectLocalExecutor<'_, '_>,
    ) -> Result<ExecResponse, String> {
        executor.exec_code(code).await
    }

    async fn checkpoint(
        &self,
        _invocation: EffectInvocation,
        checkpoint: CheckpointKind,
        executor: TurnEffectLocalExecutor<'_, '_>,
    ) -> Result<(Vec<PluginMessage>, Vec<PluginMessage>), RuntimeError> {
        executor.checkpoint(checkpoint).await
    }

    async fn sync_execution_surface(
        &self,
        _invocation: EffectInvocation,
        update_machine_config: bool,
        executor: TurnEffectLocalExecutor<'_, '_>,
    ) -> Result<Option<ExecutionSurfaceSync>, String> {
        executor.sync_execution_surface(update_machine_config).await
    }

    async fn sleep(&self, _invocation: EffectInvocation, duration: Duration) {
        tokio::time::sleep(duration).await;
    }

    async fn start_background_task(
        &self,
        registry: Arc<dyn BackgroundTaskRegistry>,
        registration: BackgroundTaskRegistration,
        _executor: BackgroundTaskExecutor,
    ) -> Result<BackgroundTaskRecord, PluginError> {
        let _ = registry;
        Err(PluginError::Session(format!(
            "background task execution is unavailable for `{}`",
            registration.id
        )))
    }

    async fn request_background_task_cancel(
        &self,
        registry: Arc<dyn BackgroundTaskRegistry>,
        task_id: &str,
        reason: Option<String>,
    ) -> Result<BackgroundTaskRecord, PluginError> {
        registry.request_cancel(task_id, reason).await
    }
}

/// Default in-process effect host.
#[derive(Clone, Default)]
pub struct LocalRuntimeEffectHost {
    background_tasks: Arc<Mutex<HashMap<String, LocalBackgroundExecution>>>,
}

#[async_trait::async_trait]
impl RuntimeEffectHost for LocalRuntimeEffectHost {
    async fn start_background_task(
        &self,
        registry: Arc<dyn BackgroundTaskRegistry>,
        registration: BackgroundTaskRegistration,
        executor: BackgroundTaskExecutor,
    ) -> Result<BackgroundTaskRecord, PluginError> {
        let task_id = registration.id.clone();
        let running = registry.mark_running(&task_id).await?;
        let cancellation = CancellationToken::new();
        let task_cancellation = cancellation.clone();
        let registry_for_task = Arc::clone(&registry);
        let tasks = Arc::clone(&self.background_tasks);
        let effect_host = self.clone();
        let task_id_for_task = task_id.clone();
        let handle = tokio::spawn(async move {
            let future = executor.run(task_cancellation);
            let completion = match RuntimeEffectScope::new(&effect_host, &task_id_for_task) {
                Ok(effect_scope) => {
                    AssertUnwindSafe(scope_active_effect_host(effect_scope, future))
                        .catch_unwind()
                        .await
                }
                Err(_) => AssertUnwindSafe(future).catch_unwind().await,
            };
            let completion = match completion {
                Ok(completion) if completion.state.is_terminal() => completion,
                Ok(completion) => BackgroundTaskCompletion {
                    state: BackgroundTaskState::Failed,
                    summary: Some(format!(
                        "background task returned non-terminal state {:?}",
                        completion.state
                    )),
                    output: completion.output,
                },
                Err(_) => BackgroundTaskCompletion {
                    state: BackgroundTaskState::Failed,
                    summary: Some("background task panicked".to_string()),
                    output: Some(crate::ToolCallOutput::failure(crate::ToolFailure::runtime(
                        crate::ToolFailureClass::Internal,
                        "background_task_panicked",
                        "background task panicked",
                    ))),
                },
            };
            let _ = registry_for_task
                .complete(&task_id_for_task, completion)
                .await;
            tasks.lock().await.remove(&task_id_for_task);
        });
        self.background_tasks.lock().await.insert(
            task_id,
            LocalBackgroundExecution {
                cancellation,
                abort: handle.abort_handle(),
            },
        );
        drop(handle);
        Ok(running)
    }

    async fn request_background_task_cancel(
        &self,
        registry: Arc<dyn BackgroundTaskRegistry>,
        task_id: &str,
        reason: Option<String>,
    ) -> Result<BackgroundTaskRecord, PluginError> {
        let record = registry.request_cancel(task_id, reason.clone()).await?;
        let execution = self.background_tasks.lock().await.get(task_id).cloned();
        if let Some(execution) = execution {
            execution.cancellation.cancel();
            if matches!(record.cancel_policy, BackgroundCancelPolicy::AbortLocal) {
                execution.abort.abort();
                self.background_tasks.lock().await.remove(task_id);
                let message = reason.unwrap_or_else(|| "background task was cancelled".to_string());
                return registry
                    .complete(
                        task_id,
                        BackgroundTaskCompletion {
                            state: BackgroundTaskState::Cancelled,
                            summary: Some(message.clone()),
                            output: Some(crate::ToolCallOutput::cancelled(
                                crate::ToolCancellation::runtime(message),
                            )),
                        },
                    )
                    .await;
            }
        }
        Ok(record)
    }
}

#[derive(Clone)]
struct LocalBackgroundExecution {
    cancellation: CancellationToken,
    abort: tokio::task::AbortHandle,
}

impl std::fmt::Debug for LocalRuntimeEffectHost {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LocalRuntimeEffectHost")
            .finish_non_exhaustive()
    }
}

pub(crate) fn turn_idempotency_key(
    session_id: &str,
    turn_id: &str,
    turn_index: usize,
    mode_iteration: usize,
    kind: RuntimeEffectKind,
    effect_id: EffectId,
) -> String {
    format!(
        "{session_id}:{turn_id}:{turn_index}:{mode_iteration}:{}:{}",
        kind.as_str(),
        effect_id.0
    )
}

pub(crate) fn direct_effect_invocation(
    session_id: &str,
    usage_source: &str,
    effect_kind: RuntimeEffectKind,
    idempotency_discriminator: String,
    turn_id: Option<&str>,
) -> EffectInvocation {
    let origin = match effect_kind {
        RuntimeEffectKind::DirectCompletion => EffectOrigin::DirectCompletion {
            usage_source: usage_source.to_string(),
        },
        RuntimeEffectKind::DirectLlmCompletion => EffectOrigin::DirectLlmCompletion {
            usage_source: usage_source.to_string(),
        },
        _ => unreachable!("direct invocation requires a direct effect kind"),
    };
    let idempotency_key = match turn_id.filter(|value| !value.is_empty()) {
        Some(turn_id) => format!(
            "{session_id}:{turn_id}:direct:{}:{usage_source}:{idempotency_discriminator}",
            effect_kind.as_str()
        ),
        None => format!(
            "{session_id}:direct:{}:{usage_source}:{idempotency_discriminator}",
            effect_kind.as_str()
        ),
    };
    EffectInvocation::new(
        EffectInvocationMetadata {
            session_id: session_id.to_string(),
            origin,
            turn_id: turn_id.map(str::to_string),
            turn_index: None,
            mode_iteration: None,
            effect_id: idempotency_discriminator,
            effect_kind,
            idempotency_key,
            turn_checkpoint: None,
        },
        CancellationToken::new(),
    )
}

pub(crate) fn tool_retry_sleep_invocation(
    parent: &EffectInvocationMetadata,
    tool_name: &str,
    attempt: u32,
    cancellation: CancellationToken,
) -> EffectInvocation {
    let effect_id = format!("{}:{tool_name}:attempt:{attempt}:sleep", parent.effect_id);
    let idempotency_key = format!(
        "{}:{tool_name}:attempt:{attempt}:sleep",
        parent.idempotency_key
    );
    EffectInvocation::new(
        EffectInvocationMetadata {
            session_id: parent.session_id.clone(),
            origin: parent.origin.clone(),
            turn_id: parent.turn_id.clone(),
            turn_index: parent.turn_index,
            mode_iteration: parent.mode_iteration,
            effect_id,
            effect_kind: RuntimeEffectKind::Sleep,
            idempotency_key,
            turn_checkpoint: parent.turn_checkpoint.clone(),
        },
        cancellation,
    )
}

pub(crate) fn direct_request_discriminator<T>(
    request: &T,
    explicit_key: Option<&str>,
    parent_tool_call_id: Option<&str>,
) -> String
where
    T: Serialize,
{
    if let Some(explicit_key) = explicit_key.filter(|key| !key.is_empty()) {
        return match parent_tool_call_id.filter(|id| !id.is_empty()) {
            Some(parent) => format!("tool:{parent}:request:{explicit_key}"),
            None => format!("request:{explicit_key}"),
        };
    }
    let bytes = serde_json::to_vec(request).unwrap_or_else(|_| b"unserializable".to_vec());
    let digest = format!("{:x}", sha2::Sha256::digest(&bytes));
    match parent_tool_call_id.filter(|id| !id.is_empty()) {
        Some(parent) => format!("tool:{parent}:sha256:{digest}"),
        None => format!("sha256:{digest}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_effect_invocation_preserves_metadata_shape() {
        let invocation = direct_effect_invocation(
            "s",
            "tool",
            RuntimeEffectKind::DirectCompletion,
            "request:k".to_string(),
            None,
        );

        assert_eq!(invocation.metadata.session_id, "s");
        assert_eq!(
            invocation.metadata.origin,
            EffectOrigin::DirectCompletion {
                usage_source: "tool".to_string()
            }
        );
        assert!(
            invocation
                .metadata
                .idempotency_key
                .starts_with("s:direct:direct_completion:tool:request:k")
        );
        assert!(invocation.metadata.turn_checkpoint.is_none());
    }
}
