use crate::{
    PromptRequest, PromptSelectionMode, ToolDefinition, ToolExecutionContext, ToolExecutionMode,
    ToolProvider, ToolResult,
};

use super::object_schema;

#[derive(Clone, Default)]
pub struct AskTool;

impl AskTool {
    pub fn new() -> Self {
        Self
    }

    async fn execute_ask_with_context(
        &self,
        args: &serde_json::Value,
        context: &ToolExecutionContext,
    ) -> ToolResult {
        let question = match super::require_str(args, "question") {
            Ok(question) => question,
            Err(err) => return err,
        };
        let options = match parse_options(args) {
            Ok(options) => options,
            Err(err) => return err,
        };
        let selection_mode = match parse_selection_mode(args, !options.is_empty()) {
            Ok(mode) => mode,
            Err(err) => return err,
        };
        let allow_note = match parse_allow_note(args, !options.is_empty()) {
            Ok(value) => value,
            Err(err) => return err,
        };
        let request = if options.is_empty() {
            PromptRequest::freeform(question.to_string())
        } else {
            let request = match selection_mode {
                PromptSelectionMode::Single => PromptRequest::single(question.to_string(), options),
                PromptSelectionMode::Multi => PromptRequest::multi(question.to_string(), options),
            };
            if allow_note {
                request.with_optional_note()
            } else {
                request
            }
        };

        match context.host.prompt_user(request).await {
            Ok(answer) => ToolResult::ok(serde_json::json!(answer)),
            Err(err) => ToolResult::err(serde_json::json!(err.to_string())),
        }
    }
}

fn parse_options(args: &serde_json::Value) -> Result<Vec<String>, ToolResult> {
    let Some(value) = args.get("options") else {
        return Ok(Vec::new());
    };
    if value.is_null() {
        return Ok(Vec::new());
    }
    let Some(items) = value.as_array() else {
        return Err(ToolResult::err_fmt(
            "Invalid options: expected list of strings",
        ));
    };
    let mut out = Vec::with_capacity(items.len());
    for (idx, item) in items.iter().enumerate() {
        let Some(option) = item.as_str() else {
            return Err(ToolResult::err_fmt(format!(
                "Invalid options[{idx}]: expected non-empty string"
            )));
        };
        if option.trim().is_empty() {
            return Err(ToolResult::err_fmt(format!(
                "Invalid options[{idx}]: expected non-empty string"
            )));
        }
        out.push(option.to_string());
    }
    Ok(out)
}

fn parse_selection_mode(
    args: &serde_json::Value,
    has_options: bool,
) -> Result<PromptSelectionMode, ToolResult> {
    let Some(value) = args.get("selection_mode") else {
        return Ok(PromptSelectionMode::Single);
    };
    let Some(mode) = value.as_str() else {
        return Err(ToolResult::err_fmt(
            "Invalid selection_mode: expected \"single\" or \"multi\"",
        ));
    };
    let selection_mode = match mode {
        "single" => PromptSelectionMode::Single,
        "multi" => PromptSelectionMode::Multi,
        _ => {
            return Err(ToolResult::err_fmt(
                "Invalid selection_mode: expected \"single\" or \"multi\"",
            ));
        }
    };
    if !has_options && matches!(selection_mode, PromptSelectionMode::Multi) {
        return Err(ToolResult::err_fmt(
            "Invalid selection_mode: \"multi\" requires non-empty options",
        ));
    }
    Ok(selection_mode)
}

fn parse_allow_note(args: &serde_json::Value, has_options: bool) -> Result<bool, ToolResult> {
    let Some(value) = args.get("allow_note") else {
        return Ok(false);
    };
    if value.is_null() {
        return Ok(false);
    }
    let Some(allow_note) = value.as_bool() else {
        return Err(ToolResult::err_fmt(
            "Invalid allow_note: expected true or false",
        ));
    };
    if allow_note && !has_options {
        return Err(ToolResult::err_fmt(
            "Invalid allow_note: requires non-empty options",
        ));
    }
    Ok(allow_note)
}

#[async_trait::async_trait]
impl ToolProvider for AskTool {
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![
            ToolDefinition::new(
                "ask",
                "Pause and ask the user a targeted question, then wait for the answer before continuing. Use this only when you are genuinely blocked, need the user's decision, or must request a value that cannot be inferred safely. Prefer doing the work without asking when a reasonable default can be discovered from local context. Provide `options` when there are roughly 2–6 discrete choices (pick/confirm/choose-between); omit it for open-ended responses where the user needs to type something. Returns structured JSON: free-form answers use `{ kind: \"text\", text }`, single-choice answers use `{ kind: \"single\", selection, note? }`, and multi-choice answers use `{ kind: \"multi\", selections, note? }`.",
                object_schema(
                    serde_json::json!({
                        "question": {
                            "type": "string",
                            "description": "Question to show the user."
                        },
                        "options": {
                            "type": ["array", "null"],
                            "items": { "type": "string" },
                            "description": "Optional list of short choices. Prefer passing `options` whenever the answer can be expressed as a short choice list; omit or pass null only for genuinely free-form input."
                        },
                        "selection_mode": {
                            "type": "string",
                            "enum": ["single", "multi"],
                            "default": "single",
                            "description": "Optional selection mode when `options` are provided: `single` (default) or `multi`."
                        },
                        "allow_note": {
                            "type": "boolean",
                            "default": false,
                            "description": "Optional. When true and `options` are provided, lets the user attach a free-form note alongside their selection."
                        }
                    }),
                    &["question"],
                ),
                serde_json::json!({ "type": "object", "additionalProperties": true }),
            )
            .with_examples(vec![
                "ask(question=\"Which environment should I use?\", options=[\"staging\", \"prod\"])"
                    .into(),
                "ask(question=\"Which checks should I run?\", options=[\"unit\", \"lint\", \"e2e\"], selection_mode=\"multi\")".into(),
                "ask(question=\"Which direction should I take?\", options=[\"minimal\", \"full\"], allow_note=true)".into(),
            ])
            .with_discovery(crate::tools::discovery_metadata(
                "user",
                &["prompt_user", "request_input"],
            ))
            .with_execution_mode(ToolExecutionMode::Parallel),
        ]
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
            "ask" => self.execute_ask_with_context(args, context).await,
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

    use crate::plugin::{
        DirectCompletionHost, DynamicToolHost, MonitorHost, PromptHost, SessionGraphHost,
        SessionLifecycleHost, SessionSnapshotHost, SessionTurnHandle, TaskHost, ToolCatalogHost,
        TraceHost, TurnHost,
    };
    use crate::{PluginError, PromptResponse, SessionHandle, SessionSnapshot, TurnInput};

    #[derive(Default)]
    struct PromptingManager {
        requests: Mutex<Vec<PromptRequest>>,
        response: Mutex<Option<PromptResponse>>,
    }

    #[async_trait::async_trait]
    impl SessionSnapshotHost for PromptingManager {
        async fn snapshot_current(&self) -> Result<SessionSnapshot, PluginError> {
            Err(PluginError::Session("snapshot unavailable".to_string()))
        }

        async fn snapshot_session(
            &self,
            _session_id: &str,
        ) -> Result<SessionSnapshot, PluginError> {
            Err(PluginError::Session("snapshot unavailable".to_string()))
        }
    }

    #[async_trait::async_trait]
    impl ToolCatalogHost for PromptingManager {
        async fn tool_catalog(
            &self,
            _session_id: &str,
        ) -> Result<Vec<serde_json::Value>, PluginError> {
            Err(PluginError::Session("tool catalog unavailable".to_string()))
        }
    }

    impl DynamicToolHost for PromptingManager {}

    #[async_trait::async_trait]
    impl SessionLifecycleHost for PromptingManager {
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
    }

    #[async_trait::async_trait]
    impl TurnHost for PromptingManager {
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
    }

    #[async_trait::async_trait]
    impl PromptHost for PromptingManager {
        async fn prompt_user(&self, request: PromptRequest) -> Result<PromptResponse, PluginError> {
            self.requests.lock().expect("requests").push(request);
            self.response
                .lock()
                .expect("response")
                .clone()
                .ok_or_else(|| PluginError::Session("prompt response missing".to_string()))
        }
    }

    impl TaskHost for PromptingManager {}
    impl MonitorHost for PromptingManager {}
    impl SessionGraphHost for PromptingManager {}
    impl DirectCompletionHost for PromptingManager {}
    impl TraceHost for PromptingManager {}

    #[tokio::test]
    async fn ask_tool_returns_structured_user_selection() {
        let tool = AskTool::new();
        let manager = Arc::new(PromptingManager {
            requests: Mutex::new(Vec::new()),
            response: Mutex::new(Some(PromptResponse::Single {
                selection: "b".to_string(),
                note: None,
            })),
        });

        let result = tool
            .execute_with_context(
                "ask",
                &serde_json::json!({
                    "question": "Choose one",
                    "options": ["a", "b"]
                }),
                &ToolExecutionContext {
                    session_id: "root".to_string(),
                    host: manager.clone(),
                    cancellation_token: None,
                    async_task_id: None,
                },
            )
            .await;

        assert!(result.success);
        assert_eq!(
            result.result,
            serde_json::json!({
                "kind": "single",
                "selection": "b",
            })
        );
        assert_eq!(
            manager.requests.lock().expect("requests").as_slice(),
            &[PromptRequest::single(
                "Choose one",
                vec!["a".to_string(), "b".to_string()]
            )]
        );
    }

    #[test]
    fn ask_tool_definition_prefers_structured_options() {
        let tool = AskTool::new();
        let definition = tool.definitions().into_iter().next().expect("definition");
        let options = definition
            .parameter_metadata()
            .into_iter()
            .find(|param| param["name"] == "options")
            .expect("options param");

        assert!(
            definition
                .description
                .contains("Provide `options` when there are roughly 2–6 discrete choices"),
            "description should bias the model toward structured choices with a concrete threshold"
        );
        assert!(
            options["description"]
                .as_str()
                .is_some_and(|text| text.contains("Prefer passing `options`")),
            "options param should explain when to use structured choices"
        );
        assert_eq!(definition.output_schema["type"], "object");
    }

    #[tokio::test]
    async fn ask_tool_forwards_optional_note_requests() {
        let tool = AskTool::new();
        let manager = Arc::new(PromptingManager {
            requests: Mutex::new(Vec::new()),
            response: Mutex::new(Some(PromptResponse::Single {
                selection: "full".to_string(),
                note: Some("keep the transcript path stable".to_string()),
            })),
        });

        let result = tool
            .execute_with_context(
                "ask",
                &serde_json::json!({
                    "question": "Which direction should I take?",
                    "options": ["minimal", "full"],
                    "allow_note": true,
                }),
                &ToolExecutionContext {
                    session_id: "root".to_string(),
                    host: manager.clone(),
                    cancellation_token: None,
                    async_task_id: None,
                },
            )
            .await;

        assert!(result.success);
        assert_eq!(
            manager.requests.lock().expect("requests").as_slice(),
            &[PromptRequest::single(
                "Which direction should I take?",
                vec!["minimal".to_string(), "full".to_string()]
            )
            .with_optional_note()]
        );
        assert_eq!(
            result.result,
            serde_json::json!({
                "kind": "single",
                "selection": "full",
                "note": "keep the transcript path stable",
            })
        );
    }

    #[tokio::test]
    async fn ask_tool_streaming_execution_preserves_session_context() {
        let tool = AskTool::new();
        let manager = Arc::new(PromptingManager {
            requests: Mutex::new(Vec::new()),
            response: Mutex::new(Some(PromptResponse::Single {
                selection: "b".to_string(),
                note: None,
            })),
        });

        let result = tool
            .execute_streaming_with_context(
                "ask",
                &serde_json::json!({
                    "question": "Choose one",
                    "options": ["a", "b"]
                }),
                &ToolExecutionContext {
                    session_id: "root".to_string(),
                    host: manager.clone(),
                    cancellation_token: None,
                    async_task_id: None,
                },
                None,
            )
            .await;

        assert!(result.success);
        assert_eq!(
            manager.requests.lock().expect("requests").as_slice(),
            &[PromptRequest::single(
                "Choose one",
                vec!["a".to_string(), "b".to_string()]
            )]
        );
    }
}
