use std::sync::{Arc, Mutex};

use lash::plugin::StaticPluginFactory;
use lash::{
    PluginError, PluginFactory, PluginSpec, ToolDefinition, ToolDiscoveryMetadata,
    ToolExecutionContext, ToolExecutionMode, ToolProvider, ToolResult,
};
use lash::{PromptRequest, PromptResponse, PromptSelectionMode};
use serde_json::json;

use crate::event::{AppEvent, AppEventTx};

#[derive(Clone, Default)]
pub(crate) struct CliPromptBridge {
    tx: Arc<Mutex<Option<AppEventTx>>>,
}

impl CliPromptBridge {
    pub(crate) fn set_event_tx(&self, tx: AppEventTx) {
        *self.tx.lock().expect("prompt bridge") = Some(tx);
    }

    pub(crate) async fn prompt_user(
        &self,
        request: PromptRequest,
    ) -> Result<PromptResponse, PluginError> {
        let tx = self
            .tx
            .lock()
            .map_err(|_| PluginError::Session("prompt bridge poisoned".to_string()))?
            .clone()
            .ok_or_else(|| PluginError::Session("prompt UI is unavailable".to_string()))?;
        let (response_tx, response_rx) = std::sync::mpsc::channel();
        tx.send(AppEvent::Prompt {
            request,
            response_tx,
        })
        .map_err(|_| PluginError::Session("prompt UI is closed".to_string()))?;
        tokio::task::spawn_blocking(move || response_rx.recv())
            .await
            .map_err(|err| PluginError::Session(format!("prompt task failed: {err}")))?
            .map_err(|_| PluginError::Session("prompt response channel closed".to_string()))
    }
}

#[async_trait::async_trait]
impl lash_plugin_plan_mode::PlanModePrompt for CliPromptBridge {
    async fn prompt_user(&self, request: PromptRequest) -> Result<PromptResponse, PluginError> {
        self.prompt_user(request).await
    }
}

#[derive(Clone)]
struct CliAskTool {
    prompt: CliPromptBridge,
}

impl CliAskTool {
    fn new(prompt: CliPromptBridge) -> Self {
        Self { prompt }
    }

    async fn execute_ask(&self, args: &serde_json::Value) -> ToolResult {
        let question = match require_str(args, "question") {
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

        match self.prompt.prompt_user(request).await {
            Ok(answer) => ToolResult::ok(json!(answer)),
            Err(err) => ToolResult::err(json!(err.to_string())),
        }
    }
}

#[async_trait::async_trait]
impl ToolProvider for CliAskTool {
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![
            ToolDefinition::new(
                "ask",
                "Pause and ask the user a targeted question, then wait for the answer before continuing. Use this only when you are genuinely blocked, need the user's decision, or must request a value that cannot be inferred safely. Prefer doing the work without asking when a reasonable default can be discovered from local context. Provide `options` when there are roughly 2-6 discrete choices; omit it for open-ended responses.",
                object_schema(
                    json!({
                        "question": { "type": "string", "description": "Question to show the user." },
                        "options": {
                            "type": ["array", "null"],
                            "items": { "type": "string" },
                            "description": "Optional list of short choices."
                        },
                        "selection_mode": {
                            "type": "string",
                            "enum": ["single", "multi"],
                            "default": "single"
                        },
                        "allow_note": {
                            "type": "boolean",
                            "default": false
                        }
                    }),
                    &["question"],
                ),
                json!({ "type": "object", "additionalProperties": true }),
            )
            .with_examples(vec![
                "ask(question=\"Which environment should I use?\", options=[\"staging\", \"prod\"])"
                    .into(),
                "ask(question=\"Which checks should I run?\", options=[\"unit\", \"lint\", \"e2e\"], selection_mode=\"multi\")".into(),
            ])
            .with_discovery(ToolDiscoveryMetadata {
                namespace: Some("user".to_string()),
                aliases: vec!["prompt_user".to_string(), "request_input".to_string()],
            })
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
        _context: &ToolExecutionContext,
    ) -> ToolResult {
        match name {
            "ask" => self.execute_ask(args).await,
            _ => ToolResult::err_fmt(format_args!("Unknown tool: {name}")),
        }
    }

    async fn execute_streaming_with_context(
        &self,
        name: &str,
        args: &serde_json::Value,
        context: &ToolExecutionContext,
        _progress: Option<&lash::ProgressSender>,
    ) -> ToolResult {
        self.execute_with_context(name, args, context).await
    }
}

pub(crate) fn cli_ask_plugin_factory(prompt: CliPromptBridge) -> Arc<dyn PluginFactory> {
    Arc::new(StaticPluginFactory::new(
        "ask",
        PluginSpec::new().with_tool_provider(Arc::new(CliAskTool::new(prompt))),
    ))
}

fn require_str<'a>(args: &'a serde_json::Value, name: &str) -> Result<&'a str, ToolResult> {
    args.get(name)
        .and_then(serde_json::Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| ToolResult::err_fmt(format!("Invalid {name}: expected non-empty string")))
}

fn object_schema(properties: serde_json::Value, required: &[&str]) -> serde_json::Value {
    json!({
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": false
    })
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
