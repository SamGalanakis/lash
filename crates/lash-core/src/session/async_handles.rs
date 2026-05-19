use std::sync::Arc;

use serde_json::json;

use super::execution_context::ModeExecutionContext;
use super::tool_execution::ModeToolReply;
use crate::tool_dispatch::dispatch_tool_call_with_execution_context;
use crate::{
    BackgroundClosePolicy, BackgroundTaskCompletion, BackgroundTaskInput, BackgroundTaskKind,
    BackgroundTaskLocalExecutor, BackgroundTaskRegistration, BackgroundTaskScope,
    BackgroundTaskState, SandboxMessage, ToolCallOutput, ToolCallRecord, ToolCallStatus,
    ToolContext,
};

const ASYNC_TOOL_HANDLE_KIND: &str = "task";

fn task_state_for_output(output: &ToolCallOutput) -> BackgroundTaskState {
    match output.status() {
        ToolCallStatus::Success => BackgroundTaskState::Completed,
        ToolCallStatus::Failure => BackgroundTaskState::Failed,
        ToolCallStatus::Cancelled => BackgroundTaskState::Cancelled,
    }
}

fn completion_from_output(output: ToolCallOutput) -> BackgroundTaskCompletion {
    BackgroundTaskCompletion {
        state: task_state_for_output(&output),
        summary: None,
        output: Some(output),
    }
}

fn tool_output_from_completion(completion: BackgroundTaskCompletion) -> ToolCallOutput {
    if let Some(output) = completion.output {
        return output;
    }
    match completion.state {
        BackgroundTaskState::Completed => ToolCallOutput::success(json!({
            "state": "completed",
        })),
        BackgroundTaskState::Cancelled => {
            ToolCallOutput::cancelled(crate::ToolCancellation::runtime(
                completion
                    .summary
                    .unwrap_or_else(|| "background task was cancelled".to_string()),
            ))
        }
        BackgroundTaskState::Failed => {
            let message = completion
                .summary
                .unwrap_or_else(|| "background task failed".to_string());
            ToolCallOutput::failure(crate::ToolFailure::tool(
                crate::ToolFailureClass::Execution,
                "background_task_failed",
                message,
            ))
        }
        BackgroundTaskState::Pending
        | BackgroundTaskState::Scheduled
        | BackgroundTaskState::Running
        | BackgroundTaskState::Waiting
        | BackgroundTaskState::CancelRequested => {
            ToolCallOutput::failure(crate::ToolFailure::tool(
                crate::ToolFailureClass::Execution,
                "background_task_not_terminal",
                "background task did not reach a terminal state",
            ))
        }
    }
}

impl ModeExecutionContext<'_> {
    pub(super) fn async_tool_handle_value(id: &str, tool_name: &str) -> serde_json::Value {
        json!({
            "__handle__": ASYNC_TOOL_HANDLE_KIND,
            "id": id,
            "tool": tool_name,
        })
    }

    pub(super) fn parse_async_tool_handle(
        handle: &serde_json::Value,
    ) -> Result<(String, Option<String>), String> {
        let kind = handle
            .get("__handle__")
            .and_then(|value| value.as_str())
            .ok_or_else(|| "Invalid async handle: missing `__handle__`".to_string())?;
        if kind != ASYNC_TOOL_HANDLE_KIND {
            return Err(format!("Invalid async handle kind: {kind}"));
        }
        let id = handle
            .get("id")
            .and_then(|value| value.as_str())
            .filter(|value| !value.is_empty())
            .ok_or_else(|| "Invalid async handle: missing `id`".to_string())?;
        let tool_name = handle
            .get("tool")
            .and_then(|value| value.as_str())
            .map(str::to_string);
        Ok((id.to_string(), tool_name))
    }

    pub(super) async fn start_async_tool_call(
        &self,
        call_id: String,
        tool_name: String,
        args: serde_json::Value,
    ) -> ModeToolReply {
        let handle_id = uuid::Uuid::new_v4().to_string();
        let registration = BackgroundTaskRegistration::new(
            handle_id.clone(),
            BackgroundTaskKind::Tool,
            tool_name.clone(),
            BackgroundTaskScope {
                session_id: self.session_id.clone(),
            },
            BackgroundTaskInput::ToolCall {
                call_id: call_id.clone(),
                tool_name: tool_name.clone(),
                args: args.clone(),
            },
        )
        .with_close_policy(BackgroundClosePolicy::Transfer);

        let (progress_tx, mut progress_rx) =
            tokio::sync::mpsc::unbounded_channel::<SandboxMessage>();
        let event_tx = self.dispatch.event_tx.clone();
        tokio::spawn(async move {
            while let Some(message) = progress_rx.recv().await {
                if message.kind != "lashlang_code" {
                    let _ = event_tx
                        .send(crate::SessionEvent::Message {
                            text: message.text,
                            kind: message.kind,
                        })
                        .await;
                }
            }
        });

        let dispatch = Arc::new(crate::tool_dispatch::ToolDispatchContext {
            plugins: Arc::clone(&self.dispatch.plugins),
            tools: Arc::clone(&self.dispatch.tools),
            surface: Arc::clone(&self.dispatch.surface),
            host: Arc::clone(&self.dispatch.host),
            effect_controller: crate::runtime::RuntimeEffectControllerHandle::shared(Arc::clone(
                &self.detached_effect_controller,
            )),
            direct_completions: crate::DirectCompletionClient::unavailable(
                "direct completions are unavailable from detached async tool execution",
            ),
            session_id: self.dispatch.session_id.clone(),
            event_tx: self.dispatch.event_tx.clone(),
            turn_injection_bridge: self.dispatch.turn_injection_bridge.clone(),
            attachment_store: Arc::clone(&self.dispatch.attachment_store),
            turn_context: self.dispatch.turn_context.clone(),
        });
        let task_handle_id = handle_id.clone();
        let task_tool_name = tool_name.clone();
        let task_args = args.clone();
        let async_call_id = call_id.clone();
        if let Err(err) = self
            .dispatch
            .host
            .start_background_task(
                &self.session_id,
                registration,
                BackgroundTaskLocalExecutor::new(move |cancellation| async move {
                    let tool_context = ToolContext::new(
                        dispatch.session_id.clone(),
                        Arc::clone(&dispatch.host),
                        dispatch.turn_context.clone(),
                        Arc::clone(&dispatch.attachment_store),
                        dispatch.direct_completions.clone(),
                        Some(async_call_id),
                    )
                    .with_async_task(task_handle_id.clone(), cancellation);
                    let outcome = dispatch_tool_call_with_execution_context(
                        &dispatch,
                        task_tool_name,
                        task_args,
                        Some(&progress_tx),
                        tool_context,
                    )
                    .await;
                    drop(progress_tx);
                    completion_from_output(outcome.record.output)
                }),
            )
            .await
        {
            return ModeToolReply::error(json!(err.to_string()));
        }

        let handle_value = Self::async_tool_handle_value(&handle_id, &tool_name);
        let record = ToolCallRecord {
            call_id: Some(call_id),
            tool: tool_name,
            args,
            output: ToolCallOutput::success(handle_value.clone()),
            duration_ms: 0,
        };
        ModeToolReply::success(handle_value).with_record(record)
    }

    pub(super) async fn await_async_tool_handle(&self, handle: serde_json::Value) -> ModeToolReply {
        let (handle_id, _hinted_tool_name) = match Self::parse_async_tool_handle(&handle) {
            Ok(parsed) => parsed,
            Err(err) => return ModeToolReply::error(json!(err)),
        };
        match self.dispatch.host.await_background_task(&handle_id).await {
            Ok(completion) => ModeToolReply::from_output(tool_output_from_completion(completion)),
            Err(err) => ModeToolReply::error(json!(err.to_string())),
        }
    }

    pub(super) async fn cancel_async_tool_handle(
        &self,
        handle: serde_json::Value,
    ) -> ModeToolReply {
        let (handle_id, _hinted_tool_name) = match Self::parse_async_tool_handle(&handle) {
            Ok(parsed) => parsed,
            Err(err) => return ModeToolReply::error(json!(err)),
        };
        match self
            .dispatch
            .host
            .cancel_background_task(&self.session_id, &handle_id)
            .await
        {
            Ok(status) => ModeToolReply::success(Self::background_task_status_value(&status)),
            Err(err) => ModeToolReply::error(json!(err.to_string())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plugin::{PluginHost, StaticPluginFactory};
    use crate::runtime::RuntimeEffectControllerHandle;
    use crate::tool_dispatch::ToolDispatchContext;
    use crate::{
        ExecutionMode, ToolCall, ToolDefinition, ToolFailureClass, ToolProvider, ToolResult,
        ToolRetryPolicy,
    };
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex as StdMutex};

    #[derive(Default)]
    struct SleepRecordingEffectController {
        sleeps: Arc<StdMutex<usize>>,
    }

    impl SleepRecordingEffectController {
        fn sleep_count(&self) -> usize {
            *self.sleeps.lock().expect("sleep count")
        }
    }

    #[async_trait::async_trait]
    impl crate::RuntimeEffectController for SleepRecordingEffectController {
        async fn execute_effect(
            &self,
            _envelope: crate::RuntimeEffectEnvelope,
            _local_executor: crate::RuntimeEffectLocalExecutor<'_>,
        ) -> Result<crate::RuntimeEffectOutcome, crate::RuntimeEffectControllerError> {
            *self.sleeps.lock().expect("sleep count") += 1;
            Ok(crate::RuntimeEffectOutcome::Sleep)
        }

        async fn start_background_task(
            &self,
            _registry: Arc<dyn crate::BackgroundTaskRegistry>,
            registration: crate::BackgroundTaskRegistration,
            _local_executor: crate::BackgroundTaskLocalExecutor,
        ) -> Result<crate::BackgroundTaskRecord, crate::PluginError> {
            Err(crate::PluginError::Session(format!(
                "sleep recording controller cannot start background task `{}`",
                registration.id
            )))
        }

        async fn request_background_task_cancel(
            &self,
            registry: Arc<dyn crate::BackgroundTaskRegistry>,
            task_id: &str,
            reason: Option<String>,
        ) -> Result<crate::BackgroundTaskRecord, crate::PluginError> {
            registry.request_cancel(task_id, reason).await
        }
    }

    struct RetryAsyncTool {
        attempts: Arc<AtomicUsize>,
    }

    fn retry_async_tool_definition() -> ToolDefinition {
        ToolDefinition::raw(
            "async_retry",
            "Retries once when run as an async handle.",
            serde_json::json!({
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }),
            serde_json::json!({ "type": "object", "additionalProperties": true }),
        )
        .with_retry_policy(ToolRetryPolicy::safe(2, 1, 1))
    }

    #[async_trait::async_trait]
    impl ToolProvider for RetryAsyncTool {
        fn tool_manifests(&self) -> Vec<crate::ToolManifest> {
            vec![retry_async_tool_definition().manifest()]
        }

        fn resolve_contract(&self, name: &str) -> Option<Arc<crate::ToolContract>> {
            (name == "async_retry").then(|| Arc::new(retry_async_tool_definition().contract()))
        }

        async fn execute(&self, _call: ToolCall<'_>) -> ToolResult {
            let attempt = self.attempts.fetch_add(1, Ordering::SeqCst);
            if attempt == 0 {
                return ToolResult::retryable_failure(
                    ToolFailureClass::External,
                    "transient",
                    "transient failure",
                    Some(1),
                );
            }
            ToolResult::ok(serde_json::json!({ "ok": true }))
        }
    }

    #[tokio::test]
    async fn async_handle_detached_tool_retry_sleep_uses_owned_detached_effect_controller() {
        let provider: Arc<dyn ToolProvider> = Arc::new(RetryAsyncTool {
            attempts: Arc::new(AtomicUsize::new(0)),
        });
        let plugins = PluginHost::new(vec![Arc::new(StaticPluginFactory::new(
            "async_retry_test",
            crate::PluginSpec::new().with_tool_provider(Arc::clone(&provider)),
        ))])
        .build_standard_session("root", None)
        .expect("plugin session");
        let tools = plugins.tools();
        let surface = plugins
            .tool_surface("session", ExecutionMode::standard())
            .expect("tool surface");
        let scoped_recorder = Arc::new(SleepRecordingEffectController::default());
        let detached_recorder = Arc::new(SleepRecordingEffectController::default());
        let (event_tx, _event_rx) = tokio::sync::mpsc::channel(8);
        let dispatch = Arc::new(ToolDispatchContext {
            plugins,
            tools,
            surface,
            host: Arc::new(crate::testing::MockSessionManager::default()),
            effect_controller: RuntimeEffectControllerHandle::shared(scoped_recorder.clone()),
            direct_completions: crate::DirectCompletionClient::unavailable(
                "direct completions are unavailable in this test context",
            ),
            session_id: "session".to_string(),
            event_tx,
            turn_injection_bridge: crate::TurnInjectionBridge::new(),
            attachment_store: Arc::new(crate::InMemoryAttachmentStore::new()),
            turn_context: crate::TurnContext::default(),
        });
        let context = ModeExecutionContext::new(
            "session".to_string(),
            ExecutionMode::standard(),
            dispatch,
            detached_recorder.clone(),
            Arc::new(crate::InMemoryAttachmentStore::new()),
            Arc::new(crate::ChronologicalProjection::default()),
            None,
            crate::TurnContext::default(),
        );

        let started = context
            .start_async_tool_call(
                "async-call-1".to_string(),
                "async_retry".to_string(),
                serde_json::json!({}),
            )
            .await;
        let crate::ToolCallOutcome::Success(handle) = started.output.outcome else {
            panic!("expected async handle output");
        };
        let awaited = context
            .await_async_tool_handle(handle.to_json_value())
            .await;

        assert!(awaited.output.is_success());
        assert_eq!(scoped_recorder.sleep_count(), 0);
        assert_eq!(detached_recorder.sleep_count(), 1);
    }
}
