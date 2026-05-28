use serde_json::json;

use super::execution_context::RuntimeExecutionContext;
use super::tool_execution::ToolInvocationReply;
use crate::tool_dispatch::ToolPreparationOutcome;
use crate::{
    ProcessHandleDescriptor, ProcessInput, ProcessRegistration, ToolCallOutput, ToolCallRecord,
};

const PROCESS_HANDLE_KIND: &str = "process";

impl RuntimeExecutionContext<'_> {
    pub(super) fn process_handle_value(id: &str, tool_name: &str) -> serde_json::Value {
        let _ = tool_name;
        crate::lashlang_bridge::process_handle_json(id)
    }

    pub(super) fn process_status_value(status: &crate::ProcessRecord) -> serde_json::Value {
        json!({
            "process_id": status.id,
            "terminal": terminal_status_label(status.terminal.as_ref()),
        })
    }

    pub(super) fn parse_process_handle(
        handle: &serde_json::Value,
    ) -> Result<(String, Option<String>), String> {
        let kind = handle
            .get("__handle__")
            .and_then(|value| value.as_str())
            .ok_or_else(|| "Invalid process handle: missing `__handle__`".to_string())?;
        if kind != PROCESS_HANDLE_KIND {
            return Err(format!("Invalid process handle kind: {kind}"));
        }
        let id = handle
            .get("id")
            .and_then(|value| value.as_str())
            .filter(|value| !value.is_empty())
            .ok_or_else(|| "Invalid process handle: missing `id`".to_string())?;
        let tool_name = handle
            .get("tool")
            .and_then(|value| value.as_str())
            .map(str::to_string);
        Ok((id.to_string(), tool_name))
    }

    pub(super) async fn start_tool_process(
        &self,
        call_id: String,
        tool_name: String,
        args: serde_json::Value,
    ) -> ToolInvocationReply {
        let handle_id = call_id.clone();
        let pending_call = crate::sansio::PendingToolCall {
            call_id: call_id.clone(),
            tool_name: tool_name.clone(),
            args: args.clone(),
            replay: None,
        };
        let prepared_call = match self.prepare_tool_call(pending_call).await {
            ToolPreparationOutcome::Prepared(prepared) => prepared,
            ToolPreparationOutcome::Completed(outcome) => {
                let mut record = outcome.record;
                record.call_id = Some(call_id);
                return ToolInvocationReply::from_output(record.output.clone()).with_record(record);
            }
        };
        let registration = ProcessRegistration::new(
            handle_id.clone(),
            ProcessInput::ToolCall {
                call: prepared_call.clone(),
            },
        );
        if let Err(err) = self
            .dispatch
            .processes
            .start(
                &self.session_id,
                registration,
                crate::ProcessStartOptions::new()
                    .with_wake_session_id(self.session_id.clone())
                    .with_descriptor(ProcessHandleDescriptor::new(
                        Some("tool"),
                        Some(tool_name.clone()),
                    )),
                self.process_scope(self.effect_metadata.clone()),
            )
            .await
        {
            return ToolInvocationReply::error(json!(err.to_string()));
        }

        let handle_value = Self::process_handle_value(&handle_id, &tool_name);
        let record = ToolCallRecord {
            call_id: Some(call_id),
            tool: prepared_call.tool_name,
            args: prepared_call.args,
            output: ToolCallOutput::success(handle_value.clone()),
            duration_ms: 0,
        };
        ToolInvocationReply::success(handle_value).with_record(record)
    }

    fn recorded_process_reply(
        call_id: String,
        tool: impl Into<String>,
        args: serde_json::Value,
        output: ToolCallOutput,
        started: std::time::Instant,
    ) -> ToolInvocationReply {
        let record = ToolCallRecord {
            call_id: Some(call_id),
            tool: tool.into(),
            args,
            output: output.clone(),
            duration_ms: started.elapsed().as_millis() as u64,
        };
        ToolInvocationReply::from_output(output).with_record(record)
    }

    fn recorded_process_error(
        call_id: String,
        tool: &'static str,
        args: serde_json::Value,
        message: impl Into<String>,
        started: std::time::Instant,
    ) -> ToolInvocationReply {
        let output = ToolInvocationReply::error(json!(message.into())).output;
        Self::recorded_process_reply(call_id, tool, args, output, started)
    }

    pub(super) async fn await_process_handle(
        &self,
        call_id: String,
        handle: serde_json::Value,
    ) -> ToolInvocationReply {
        let started = std::time::Instant::now();
        let args = json!({ "handle": handle.clone() });
        let (handle_id, _hinted_tool_name) = match Self::parse_process_handle(&handle) {
            Ok(parsed) => parsed,
            Err(err) => {
                return Self::recorded_process_error(call_id, "await_process", args, err, started);
            }
        };
        if let Err(err) = self
            .dispatch
            .processes
            .validate_visible(&self.session_id, std::slice::from_ref(&handle_id))
            .await
        {
            return Self::recorded_process_error(
                call_id,
                "await_process",
                args,
                err.to_string(),
                started,
            );
        }
        let output = self
            .await_process_with_cancellation(
                &handle_id,
                self.effect_metadata.clone(),
                self.cancellation_token.clone(),
            )
            .await;
        let output = match output {
            Ok(output) => output.into_tool_output(),
            Err(err) => ToolInvocationReply::error(json!(err.to_string())).output,
        };
        Self::recorded_process_reply(call_id, "await_process", args, output, started)
    }

    pub(super) async fn cancel_process_handle(
        &self,
        call_id: String,
        handle: serde_json::Value,
    ) -> ToolInvocationReply {
        let started = std::time::Instant::now();
        let args = json!({ "handle": handle.clone() });
        let (handle_id, _hinted_tool_name) = match Self::parse_process_handle(&handle) {
            Ok(parsed) => parsed,
            Err(err) => {
                return Self::recorded_process_error(call_id, "cancel_process", args, err, started);
            }
        };
        if let Err(err) = self
            .dispatch
            .processes
            .validate_visible(&self.session_id, std::slice::from_ref(&handle_id))
            .await
        {
            return Self::recorded_process_error(
                call_id,
                "cancel_process",
                args,
                err.to_string(),
                started,
            );
        }
        let output = match self
            .dispatch
            .processes
            .cancel(
                &self.session_id,
                &handle_id,
                self.process_scope(self.effect_metadata.clone()),
            )
            .await
        {
            Ok(status) => ToolCallOutput::success(Self::process_status_value(&status)),
            Err(err) => ToolInvocationReply::error(json!(err.to_string())).output,
        };
        Self::recorded_process_reply(call_id, "cancel_process", args, output, started)
    }
}

fn terminal_status_label(terminal: Option<&crate::ProcessTerminalSemantics>) -> &'static str {
    match terminal.map(|terminal| terminal.state) {
        None => "running",
        Some(crate::ProcessTerminalState::Completed) => "completed",
        Some(crate::ProcessTerminalState::Failed) => "failed",
        Some(crate::ProcessTerminalState::Cancelled) => "cancelled",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plugin::PluginHost;
    use crate::runtime::RuntimeEffectControllerHandle;
    use crate::tool_dispatch::ToolDispatchContext;
    use crate::{
        PreparedToolCall, ProcessRegistry, ToolCall, ToolDefinition, ToolPrepareCall, ToolProvider,
        ToolResult,
    };
    use std::collections::BTreeMap;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct PrepareRecordingTool {
        prepares: Arc<AtomicUsize>,
    }

    fn process_tool_definition() -> ToolDefinition {
        ToolDefinition::raw(
            "tool:process_prepare",
            "process_prepare",
            "Records preparation before background registration.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "input": { "type": "string" }
                },
                "additionalProperties": false
            }),
            serde_json::json!({ "type": "object", "additionalProperties": true }),
        )
    }

    #[async_trait::async_trait]
    impl ToolProvider for PrepareRecordingTool {
        fn tool_manifests(&self) -> Vec<crate::ToolManifest> {
            vec![process_tool_definition().manifest()]
        }

        fn resolve_contract(&self, name: &str) -> Option<Arc<crate::ToolContract>> {
            (name == "process_prepare").then(|| Arc::new(process_tool_definition().contract()))
        }

        async fn prepare_tool_call(
            &self,
            call: ToolPrepareCall<'_>,
        ) -> Result<PreparedToolCall, ToolResult> {
            self.prepares.fetch_add(1, Ordering::SeqCst);
            Ok(PreparedToolCall::from_parts(
                call.pending.call_id,
                call.pending.tool_name,
                call.pending.args,
                call.pending.replay,
                serde_json::json!({ "prepared": true }),
            ))
        }

        async fn execute(&self, call: ToolCall<'_>) -> ToolResult {
            ToolResult::ok(serde_json::json!({
                "payload": call.context.prepared_payload().clone(),
            }))
        }
    }

    #[tokio::test]
    async fn process_handle_start_registers_prepared_tool_call() {
        let prepares = Arc::new(AtomicUsize::new(0));
        let provider: Arc<dyn ToolProvider> = Arc::new(PrepareRecordingTool {
            prepares: Arc::clone(&prepares),
        });
        let plugins = PluginHost::empty()
            .build_session("root", None)
            .expect("plugin session");
        let tools = Arc::clone(&provider);
        let surface = Arc::new(crate::ToolSurface::from_tools(
            provider.tool_manifests(),
            BTreeMap::new(),
        ));
        let host = Arc::new(crate::testing::MockSessionManager::default());
        let (event_tx, _event_rx) = tokio::sync::mpsc::channel(8);
        let dispatch = Arc::new(ToolDispatchContext {
            plugins,
            tools,
            surface,
            host: host.clone(),
            processes: host.clone(),
            effect_controller: RuntimeEffectControllerHandle::shared(Arc::new(
                crate::InlineRuntimeEffectController::default(),
            )),
            direct_completions: crate::DirectCompletionClient::unavailable(
                "direct completions are unavailable in this test context",
            ),
            tool_effect_metadata: None,
            session_id: "session".to_string(),
            event_tx,
            turn_injection_bridge: crate::TurnInjectionBridge::new(),
            attachment_store: Arc::new(crate::InMemoryAttachmentStore::new()),
            turn_context: crate::TurnContext::default(),
        });
        let context = RuntimeExecutionContext::new(
            "session".to_string(),
            dispatch,
            Default::default(),
            Arc::new(lashlang::InMemoryLashlangArtifactStore::new()),
            Arc::new(crate::InMemoryAttachmentStore::new()),
            Arc::new(crate::ChronologicalProjection::default()),
            None,
            crate::TurnContext::default(),
        );

        let started = context
            .start_tool_process(
                "async-call-1".to_string(),
                "process_prepare".to_string(),
                serde_json::json!({ "input": "live" }),
            )
            .await;
        let crate::ToolCallOutcome::Success(handle) = started.output.outcome else {
            panic!("expected process handle output");
        };
        assert_eq!(
            handle
                .to_json_value()
                .get("id")
                .and_then(|value| value.as_str()),
            Some("async-call-1")
        );
        assert_eq!(prepares.load(Ordering::SeqCst), 1);
        let record = host
            .process_registry
            .get_process("async-call-1")
            .await
            .expect("registered process");
        let ProcessInput::ToolCall { call } = record.input.as_ref() else {
            panic!("expected prepared tool call process input");
        };
        assert_eq!(call.tool_name, "process_prepare");
        assert_eq!(call.args, serde_json::json!({ "input": "live" }));
        assert_eq!(
            call.prepared_payload,
            serde_json::json!({ "prepared": true })
        );

        let awaited = context
            .await_process_handle("await-async-call-1".to_string(), handle.to_json_value())
            .await;

        assert!(awaited.output.is_success());
        let record = awaited.record.expect("await record");
        assert_eq!(record.call_id.as_deref(), Some("await-async-call-1"));
        assert_eq!(record.tool, "await_process");
    }

    #[tokio::test]
    async fn process_handle_await_and_cancel_require_session_grant() {
        let provider: Arc<dyn ToolProvider> = Arc::new(PrepareRecordingTool {
            prepares: Arc::new(AtomicUsize::new(0)),
        });
        let plugins = PluginHost::empty()
            .build_session("root", None)
            .expect("plugin session");
        let surface = Arc::new(crate::ToolSurface::from_tools(
            provider.tool_manifests(),
            BTreeMap::new(),
        ));
        let host = Arc::new(crate::testing::MockSessionManager::default());
        host.process_registry
            .register_process(ProcessRegistration::new(
                "hidden-process",
                ProcessInput::External {
                    metadata: serde_json::Value::Null,
                },
            ))
            .await
            .expect("register hidden process");
        let (event_tx, _event_rx) = tokio::sync::mpsc::channel(8);
        let dispatch = Arc::new(ToolDispatchContext {
            plugins,
            tools: provider,
            surface,
            host: host.clone(),
            processes: host,
            effect_controller: RuntimeEffectControllerHandle::shared(Arc::new(
                crate::InlineRuntimeEffectController::default(),
            )),
            direct_completions: crate::DirectCompletionClient::unavailable(
                "direct completions are unavailable in this test context",
            ),
            tool_effect_metadata: None,
            session_id: "session".to_string(),
            event_tx,
            turn_injection_bridge: crate::TurnInjectionBridge::new(),
            attachment_store: Arc::new(crate::InMemoryAttachmentStore::new()),
            turn_context: crate::TurnContext::default(),
        });
        let context = RuntimeExecutionContext::new(
            "session".to_string(),
            dispatch,
            Default::default(),
            Arc::new(lashlang::InMemoryLashlangArtifactStore::new()),
            Arc::new(crate::InMemoryAttachmentStore::new()),
            Arc::new(crate::ChronologicalProjection::default()),
            None,
            crate::TurnContext::default(),
        );
        let handle = json!({
            "__handle__": "process",
            "id": "hidden-process"
        });

        let awaited = context
            .await_process_handle("await-hidden-process".to_string(), handle.clone())
            .await;
        let cancelled = context
            .cancel_process_handle("cancel-hidden-process".to_string(), handle)
            .await;

        assert!(!awaited.output.is_success());
        assert!(!cancelled.output.is_success());
        assert_eq!(
            awaited
                .record
                .as_ref()
                .and_then(|record| record.call_id.as_deref()),
            Some("await-hidden-process")
        );
        assert_eq!(
            cancelled
                .record
                .as_ref()
                .and_then(|record| record.call_id.as_deref()),
            Some("cancel-hidden-process")
        );
    }
}
