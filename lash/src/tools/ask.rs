use crate::{ExecutionMode, PromptBridge, ToolDefinition, ToolParam, ToolProvider, ToolResult};

#[derive(Clone)]
pub struct AskTool {
    prompt_bridge: PromptBridge,
    headless: bool,
}

impl AskTool {
    pub fn new(prompt_bridge: PromptBridge, headless: bool) -> Self {
        Self {
            prompt_bridge,
            headless,
        }
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
        if self.headless {
            return Vec::new();
        }

        vec![ToolDefinition {
            name: "ask".into(),
            description: vec![crate::ToolText::new(
                "Pause and ask the user a targeted question, then wait for the answer before continuing. Use this only when you are genuinely blocked, need the user's decision, or must request a value that cannot be inferred safely. Prefer doing the work without asking when a reasonable default can be discovered from local context. Omit `options` for free-form input, or provide a short list of choices.",
                [ExecutionMode::Standard],
            )],
            params: vec![
                ToolParam {
                    name: "question".into(),
                    r#type: "str".into(),
                    description: "Question to show the user.".into(),
                    required: true,
                },
                ToolParam {
                    name: "options".into(),
                    r#type: "list".into(),
                    description:
                        "Optional list of short choices. Omit or pass null for free-form input."
                            .into(),
                    required: false,
                },
            ],
            returns: "str".into(),
            examples: vec![crate::ToolText::new(
                "ask(question=\"Which environment should I use?\", options=[\"staging\", \"prod\"])",
                [ExecutionMode::Standard],
            )],
            hidden: false,
            inject_into_prompt: true,
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
        let tool = AskTool::new(bridge, false);

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
    fn ask_tool_is_hidden_in_headless_mode() {
        let tool = AskTool::new(PromptBridge::new(), true);
        assert!(tool.definitions().is_empty());
    }
}
