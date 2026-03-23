use crate::{PromptBridge, ToolDefinition, ToolParam, ToolProvider, ToolResult};

#[derive(Clone)]
pub struct AskTool {
    prompt_bridge: PromptBridge,
}

impl AskTool {
    pub fn new(prompt_bridge: PromptBridge) -> Self {
        Self { prompt_bridge }
    }

    async fn execute_ask(&self, args: &serde_json::Value) -> ToolResult {
        let question = match super::require_str(args, "question") {
            Ok(question) => question,
            Err(err) => return err,
        };
        let options = match parse_options(args) {
            Ok(options) => options,
            Err(err) => return err,
        };
        match self
            .prompt_bridge
            .prompt(question.to_string(), options)
            .await
        {
            Ok(answer) => ToolResult::ok(serde_json::json!(answer)),
            Err(err) => ToolResult::err(serde_json::json!(err)),
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

#[async_trait::async_trait]
impl ToolProvider for AskTool {
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![ToolDefinition {
            name: "ask".into(),
            description: "Pause and ask the user a targeted question, then wait for the answer before continuing. Use this only when you are genuinely blocked, need the user's decision, or must request a value that cannot be inferred safely. Prefer doing the work without asking when a reasonable default can be discovered from local context. When a short list of concrete answers would be sufficient, always provide `options`; omit `options` only for genuinely free-form responses.".into(),
            params: vec![
                ToolParam {
                    name: "question".into(),
                    r#type: "str".into(),
                    description: "Question to show the user.".into(),
                    default_value: None,
                    required: true,
                },
                ToolParam {
                    name: "options".into(),
                    r#type: "list".into(),
                    description:
                        "Optional list of short choices. Prefer passing `options` whenever the answer can be expressed as a short choice list; omit or pass null only for genuinely free-form input."
                            .into(),
                    default_value: None,
                    required: false,
                },
            ],
            returns: "str".into(),
            examples: vec![
                "ask(question=\"Which environment should I use?\", options=[\"staging\", \"prod\"])"
                    .into(),
            ],
            enabled: true,
            injected: true,
            input_schema_override: None,
            output_schema_override: None,
        }]
    }
    async fn execute(&self, name: &str, args: &serde_json::Value) -> ToolResult {
        match name {
            "ask" => self.execute_ask(args).await,
            _ => ToolResult::err_fmt(format_args!("Unknown tool: {name}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::mpsc::unbounded_channel;

    #[tokio::test]
    async fn ask_tool_returns_user_selection() {
        let bridge = PromptBridge::new();
        let (tx, mut rx) = unbounded_channel();
        bridge.set_sender(tx);
        let tool = AskTool::new(bridge);

        let response_task = tokio::spawn(async move {
            let prompt = rx.recv().await.expect("prompt");
            assert_eq!(prompt.question, "Choose one");
            assert_eq!(prompt.options, vec!["a".to_string(), "b".to_string()]);
            prompt.response_tx.send("b".to_string()).expect("response");
        });

        let result = tool
            .execute(
                "ask",
                &serde_json::json!({
                    "question": "Choose one",
                    "options": ["a", "b"]
                }),
            )
            .await;

        response_task.await.expect("response task");
        assert!(result.success);
        assert_eq!(result.result, serde_json::json!("b"));
    }

    #[test]
    fn ask_tool_definition_prefers_structured_options() {
        let tool = AskTool::new(PromptBridge::new());
        let definition = tool.definitions().into_iter().next().expect("definition");
        let options = definition
            .params
            .iter()
            .find(|param| param.name == "options")
            .expect("options param");

        assert!(
            definition.description.contains("always provide `options`"),
            "description should bias the model toward structured choices"
        );
        assert!(
            options.description.contains("Prefer passing `options`"),
            "options param should explain when to use structured choices"
        );
    }
}
