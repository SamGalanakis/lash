use serde_json::json;

use super::execution_context::ModeExecutionContext;
use super::tool_execution::ModeToolReply;
use crate::tool_dispatch::ToolPreparationOutcome;
use crate::{
    ProcessClosePolicy, ProcessInput, ProcessRegistration, ProcessScope, ToolCallOutput,
    ToolCallRecord,
};

const PROCESS_HANDLE_KIND: &str = "process";

impl ModeExecutionContext<'_> {
    pub(super) fn process_handle_value(id: &str, tool_name: &str) -> serde_json::Value {
        let _ = tool_name;
        json!({
            "__handle__": PROCESS_HANDLE_KIND,
            "id": id,
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
    ) -> ModeToolReply {
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
                return ModeToolReply::from_output(record.output.clone()).with_record(record);
            }
        };
        let mut registration = ProcessRegistration::new(
            handle_id.clone(),
            tool_name.clone(),
            ProcessScope {
                session_id: self.session_id.clone(),
            },
            ProcessInput::ToolCall {
                call: prepared_call.clone(),
            },
        )
        .with_tags(["tool"])
        .with_handle_visible(true)
        .with_close_policy(ProcessClosePolicy::Transfer);
        if let Some(metadata) = self.effect_metadata.as_ref() {
            registration = registration.with_metadata(serde_json::json!({
                "tool_effect_metadata": metadata,
            }));
        }

        if let Err(err) = self
            .dispatch
            .host
            .start_process_scoped(
                &self.session_id,
                registration,
                self.effect_metadata.clone(),
                Some(self.dispatch.effect_controller.as_controller()),
            )
            .await
        {
            return ModeToolReply::error(json!(err.to_string()));
        }

        let handle_value = Self::process_handle_value(&handle_id, &tool_name);
        let record = ToolCallRecord {
            call_id: Some(call_id),
            tool: prepared_call.tool_name,
            args: prepared_call.args,
            output: ToolCallOutput::success(handle_value.clone()),
            duration_ms: 0,
        };
        ModeToolReply::success(handle_value).with_record(record)
    }

    pub(super) async fn await_process_handle(&self, handle: serde_json::Value) -> ModeToolReply {
        let (handle_id, _hinted_tool_name) = match Self::parse_process_handle(&handle) {
            Ok(parsed) => parsed,
            Err(err) => return ModeToolReply::error(json!(err)),
        };
        match self
            .dispatch
            .host
            .await_process_scoped(
                &handle_id,
                self.effect_metadata.clone(),
                Some(self.dispatch.effect_controller.as_controller()),
            )
            .await
        {
            Ok(output) => ModeToolReply::from_output(output.into_tool_output()),
            Err(err) => ModeToolReply::error(json!(err.to_string())),
        }
    }

    pub(super) async fn cancel_process_handle(&self, handle: serde_json::Value) -> ModeToolReply {
        let (handle_id, _hinted_tool_name) = match Self::parse_process_handle(&handle) {
            Ok(parsed) => parsed,
            Err(err) => return ModeToolReply::error(json!(err)),
        };
        match self
            .dispatch
            .host
            .cancel_process_scoped(
                &self.session_id,
                &handle_id,
                self.effect_metadata.clone(),
                Some(self.dispatch.effect_controller.as_controller()),
            )
            .await
        {
            Ok(status) => ModeToolReply::success(Self::process_status_value(&status)),
            Err(err) => ModeToolReply::error(json!(err.to_string())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plugin::PluginHost;
    use crate::runtime::RuntimeEffectControllerHandle;
    use crate::tool_dispatch::ToolDispatchContext;
    use crate::{
        ExecutionMode, PreparedToolCall, ProcessRegistry, ToolCall, ToolDefinition,
        ToolPrepareCall, ToolProvider, ToolResult,
    };
    use std::collections::BTreeMap;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct PrepareRecordingTool {
        prepares: Arc<AtomicUsize>,
    }

    fn process_tool_definition() -> ToolDefinition {
        ToolDefinition::raw(
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
            .build_standard_session("root", None)
            .expect("plugin session");
        let tools = Arc::clone(&provider);
        let surface = Arc::new(crate::ToolSurface::from_tools(
            provider.tool_manifests(),
            ExecutionMode::standard(),
            BTreeMap::new(),
        ));
        let host = Arc::new(crate::testing::MockSessionManager::default());
        let (event_tx, _event_rx) = tokio::sync::mpsc::channel(8);
        let dispatch = Arc::new(ToolDispatchContext {
            plugins,
            tools,
            surface,
            host: host.clone(),
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
        let context = ModeExecutionContext::new(
            "session".to_string(),
            ExecutionMode::standard(),
            dispatch,
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
            .get("async-call-1")
            .await
            .expect("registered process");
        let ProcessInput::ToolCall { call } = record.input else {
            panic!("expected prepared tool call process input");
        };
        assert_eq!(call.tool_name, "process_prepare");
        assert_eq!(call.args, serde_json::json!({ "input": "live" }));
        assert_eq!(
            call.prepared_payload,
            serde_json::json!({ "prepared": true })
        );

        let awaited = context.await_process_handle(handle.to_json_value()).await;

        assert!(awaited.output.is_success());
    }
}
