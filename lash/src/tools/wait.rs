use crate::{
    PromptRequest, ToolDefinition, ToolExecutionContext, ToolParam, ToolProvider, ToolResult,
};

#[derive(Clone, Default)]
pub struct WaitTool;

impl WaitTool {
    pub fn new() -> Self {
        Self
    }

    async fn execute_wait_with_context(
        &self,
        args: &serde_json::Value,
        context: &ToolExecutionContext,
    ) -> ToolResult {
        let seconds = match args.get("seconds").and_then(|value| value.as_u64()) {
            Some(value) if value > 0 => value,
            _ => {
                return ToolResult::err_fmt(
                    "Invalid seconds: expected positive integer number of seconds",
                );
            }
        };

        let request =
            PromptRequest::freeform("Pausing briefly before continuing.").with_wait(seconds);

        match context.host.prompt_user(request).await {
            Ok(crate::PromptResponse::Text { text }) => {
                let resumed_early =
                    text.trim() == crate::WAIT_PROMPT_RESUME_EARLY_TOKEN || text.trim().is_empty();
                ToolResult::ok(serde_json::json!({
                    "seconds": seconds,
                    "resumed_early": resumed_early,
                }))
            }
            Ok(_) => ToolResult::ok(serde_json::json!({
                "seconds": seconds,
                "resumed_early": false,
            })),
            Err(err) => ToolResult::err(serde_json::json!(err.to_string())),
        }
    }
}

#[async_trait::async_trait]
impl ToolProvider for WaitTool {
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![ToolDefinition {
            name: "wait".into(),
            description: "Pause briefly, then resume automatically. Use this to wait before polling, retrying, or checking for external changes.".into(),
            params: vec![ToolParam {
                name: "seconds".into(),
                r#type: "int".into(),
                description: "How long to wait before resuming automatically.".into(),
                default_value: None,
                required: true,
            }],
            returns: "json".into(),
            examples: vec!["wait(seconds=5)".into()],
            enabled: true,
            injected: true,
            input_schema_override: None,
            output_schema_override: None,
        }]
    }

    async fn execute(&self, name: &str, _args: &serde_json::Value) -> ToolResult {
        ToolResult::err_fmt(format_args!(
            "`{name}` requires session context and cannot run without it"
        ))
    }

    async fn execute_with_context(
        &self,
        name: &str,
        args: &serde_json::Value,
        context: &ToolExecutionContext,
    ) -> ToolResult {
        match name {
            "wait" => self.execute_wait_with_context(args, context).await,
            _ => ToolResult::err_fmt(format_args!("Unknown tool: {name}")),
        }
    }

    async fn execute_streaming_with_context(
        &self,
        name: &str,
        args: &serde_json::Value,
        context: &ToolExecutionContext,
        _progress: Option<&crate::ProgressSender>,
    ) -> ToolResult {
        self.execute_with_context(name, args, context).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::Mutex;

    use crate::plugin::SessionTurnHandle;
    use crate::{
        PluginError, PromptResponse, SessionHandle, SessionManager, SessionSnapshot, TurnInput,
    };

    #[derive(Default)]
    struct WaitingManager {
        requests: Mutex<Vec<PromptRequest>>,
        response: Mutex<Option<PromptResponse>>,
    }

    #[async_trait::async_trait]
    impl SessionManager for WaitingManager {
        async fn snapshot_current(&self) -> Result<SessionSnapshot, PluginError> {
            Err(PluginError::Session("snapshot unavailable".to_string()))
        }

        async fn snapshot_session(
            &self,
            _session_id: &str,
        ) -> Result<SessionSnapshot, PluginError> {
            Err(PluginError::Session("snapshot unavailable".to_string()))
        }

        async fn tool_catalog(
            &self,
            _session_id: &str,
        ) -> Result<Vec<serde_json::Value>, PluginError> {
            Err(PluginError::Session("tool catalog unavailable".to_string()))
        }

        async fn create_session(
            &self,
            _request: crate::SessionCreateRequest,
        ) -> Result<SessionHandle, PluginError> {
            Err(PluginError::Session(
                "session creation unavailable".to_string(),
            ))
        }

        async fn close_session(&self, _session_id: &str) -> Result<(), PluginError> {
            Err(PluginError::Session(
                "session close unavailable".to_string(),
            ))
        }

        async fn start_turn_stream(
            &self,
            _session_id: &str,
            _input: TurnInput,
        ) -> Result<SessionTurnHandle, PluginError> {
            Err(PluginError::Session(
                "turn streaming unavailable".to_string(),
            ))
        }

        async fn await_turn(&self, _turn_id: &str) -> Result<crate::AssembledTurn, PluginError> {
            Err(PluginError::Session("await turn unavailable".to_string()))
        }

        async fn cancel_turn(&self, _turn_id: &str) -> Result<(), PluginError> {
            Err(PluginError::Session("cancel turn unavailable".to_string()))
        }

        async fn prompt_user(&self, request: PromptRequest) -> Result<PromptResponse, PluginError> {
            self.requests.lock().expect("requests").push(request);
            self.response
                .lock()
                .expect("response")
                .clone()
                .ok_or_else(|| PluginError::Session("prompt response missing".to_string()))
        }
    }

    #[tokio::test]
    async fn wait_tool_uses_wait_prompt_request() {
        let tool = WaitTool::new();
        let manager = Arc::new(WaitingManager {
            requests: Mutex::new(Vec::new()),
            response: Mutex::new(Some(PromptResponse::Text {
                text: crate::WAIT_PROMPT_TIMEOUT_TOKEN.to_string(),
            })),
        });

        let result = tool
            .execute_with_context(
                "wait",
                &serde_json::json!({ "seconds": 5 }),
                &ToolExecutionContext {
                    session_id: "root".to_string(),
                    host: manager.clone(),
                    cancellation_token: None,
                    async_task_id: None,
                },
            )
            .await;

        assert!(result.success);
        assert_eq!(result.result["seconds"], serde_json::json!(5));
        assert_eq!(result.result["resumed_early"], serde_json::json!(false));
        let requests = manager.requests.lock().expect("requests");
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].wait.as_ref().map(|wait| wait.seconds), Some(5));
        assert!(requests[0].is_wait());
    }
}
