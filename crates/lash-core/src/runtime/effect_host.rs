use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::mpsc;
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
    ToolBatch,
    ExecCode,
    Checkpoint,
    SyncExecutionSurface,
}

impl RuntimeEffectKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::LlmCall => "llm_call",
            Self::DirectCompletion => "direct_completion",
            Self::DirectLlmCompletion => "direct_llm_completion",
            Self::ToolBatch => "tool_batch",
            Self::ExecCode => "exec_code",
            Self::Checkpoint => "checkpoint",
            Self::SyncExecutionSurface => "sync_execution_surface",
        }
    }
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
pub struct TurnEffectLocalExecutor<'a> {
    driver: &'a mut RuntimeTurnDriver,
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

impl<'a> TurnEffectLocalExecutor<'a> {
    pub(in crate::runtime) fn new(
        driver: &'a mut RuntimeTurnDriver,
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

    pub async fn tool_batch(self, calls: Vec<PendingToolCall>) -> Vec<CompletedToolCall> {
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
        executor: TurnEffectLocalExecutor<'_>,
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

    async fn tool_batch(
        &self,
        _invocation: EffectInvocation,
        calls: Vec<PendingToolCall>,
        executor: TurnEffectLocalExecutor<'_>,
    ) -> Vec<CompletedToolCall> {
        executor.tool_batch(calls).await
    }

    async fn exec_code(
        &self,
        _invocation: EffectInvocation,
        code: String,
        executor: TurnEffectLocalExecutor<'_>,
    ) -> Result<ExecResponse, String> {
        executor.exec_code(code).await
    }

    async fn checkpoint(
        &self,
        _invocation: EffectInvocation,
        checkpoint: CheckpointKind,
        executor: TurnEffectLocalExecutor<'_>,
    ) -> Result<(Vec<PluginMessage>, Vec<PluginMessage>), RuntimeError> {
        executor.checkpoint(checkpoint).await
    }

    async fn sync_execution_surface(
        &self,
        _invocation: EffectInvocation,
        update_machine_config: bool,
        executor: TurnEffectLocalExecutor<'_>,
    ) -> Result<Option<ExecutionSurfaceSync>, String> {
        executor.sync_execution_surface(update_machine_config).await
    }
}

/// Default in-process effect host.
#[derive(Debug, Default)]
pub struct LocalRuntimeEffectHost;

#[async_trait::async_trait]
impl RuntimeEffectHost for LocalRuntimeEffectHost {}

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
) -> EffectInvocation {
    let effect_id = uuid::Uuid::new_v4().to_string();
    let origin = match effect_kind {
        RuntimeEffectKind::DirectCompletion => EffectOrigin::DirectCompletion {
            usage_source: usage_source.to_string(),
        },
        RuntimeEffectKind::DirectLlmCompletion => EffectOrigin::DirectLlmCompletion {
            usage_source: usage_source.to_string(),
        },
        _ => unreachable!("direct invocation requires a direct effect kind"),
    };
    EffectInvocation::new(
        EffectInvocationMetadata {
            session_id: session_id.to_string(),
            origin,
            turn_id: None,
            turn_index: None,
            mode_iteration: None,
            effect_id: effect_id.clone(),
            effect_kind,
            idempotency_key: format!("{session_id}:direct:{}:{effect_id}", effect_kind.as_str()),
            turn_checkpoint: None,
        },
        CancellationToken::new(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_effect_invocation_preserves_metadata_shape() {
        let invocation = direct_effect_invocation("s", "tool", RuntimeEffectKind::DirectCompletion);

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
                .starts_with("s:direct:direct_completion:")
        );
        assert!(invocation.metadata.turn_checkpoint.is_none());
    }
}
