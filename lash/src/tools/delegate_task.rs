use std::sync::Arc;

use serde_json::json;
use tokio::sync::mpsc;

use crate::{
    Agent, AgentConfig, AgentEvent, Session, SessionConfig, ToolDefinition, ToolParam,
    ToolProvider, ToolResult,
};

/// Spawns an autonomous sub-agent with its own REPL and tool access.
pub struct DelegateTask {
    tools: Arc<dyn ToolProvider>,
    model: String,
    api_key: String,
    base_url: String,
    max_iterations: usize,
}

impl DelegateTask {
    pub fn new(
        tools: Arc<dyn ToolProvider>,
        model: &str,
        api_key: &str,
        base_url: &str,
    ) -> Self {
        Self {
            tools,
            model: model.to_string(),
            api_key: api_key.to_string(),
            base_url: base_url.to_string(),
            max_iterations: 10,
        }
    }
}

#[async_trait::async_trait]
impl ToolProvider for DelegateTask {
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![ToolDefinition {
            name: "delegate_task".into(),
            description: "Spawn a sub-agent to handle a task autonomously. It has its own REPL and full tool access. Returns {\"result\": str, \"context\": [str]}.".into(),
            params: vec![ToolParam::typed("prompt", "str")],
            returns: "dict".into(),
        }]
    }

    async fn execute(&self, _name: &str, args: &serde_json::Value) -> ToolResult {
        let prompt = args
            .get("prompt")
            .and_then(|v| v.as_str())
            .unwrap_or_default();

        if prompt.is_empty() {
            return ToolResult {
                success: false,
                result: json!("Missing required parameter: prompt"),
            };
        }

        // Create a new session with the base tools (no delegate_task)
        let session =
            match Session::new(Arc::clone(&self.tools), SessionConfig::default()).await {
                Ok(s) => s,
                Err(e) => {
                    return ToolResult {
                        success: false,
                        result: json!(format!("Failed to create sub-agent session: {e}")),
                    };
                }
            };

        let config = AgentConfig {
            model: self.model.clone(),
            api_key: self.api_key.clone(),
            base_url: self.base_url.clone(),
            max_iterations: self.max_iterations,
            sub_agent: true,
            ..Default::default()
        };

        let mut agent = Agent::new(session, config);
        let messages = vec![crate::agent::ChatMsg {
            role: "user".to_string(),
            content: prompt.to_string(),
        }];

        let (event_tx, mut event_rx) = mpsc::channel::<AgentEvent>(100);

        // Run agent in a spawned task so we can drain events concurrently
        let run_handle = tokio::spawn(async move { agent.run(messages, event_tx).await });

        let mut final_message: Option<String> = None;
        let mut context: Vec<String> = Vec::new();
        let mut current_prose = String::new();

        while let Some(event) = event_rx.recv().await {
            match event {
                AgentEvent::TextDelta { content } => {
                    current_prose.push_str(&content);
                }
                AgentEvent::Message { text, kind } => {
                    if kind == "final" {
                        final_message = Some(text);
                    } else if kind == "progress" {
                        context.push(text);
                    }
                }
                AgentEvent::CodeBlock { .. } => {
                    // Preceding prose was intermediate — capture as context
                    let trimmed = current_prose.trim().to_string();
                    if !trimmed.is_empty() {
                        context.push(trimmed);
                    }
                    current_prose.clear();
                }
                AgentEvent::Done => break,
                AgentEvent::Error { .. } => {}
                _ => {}
            }
        }

        // Wait for the agent task to finish
        let _ = run_handle.await;

        let result = if let Some(msg) = final_message {
            msg
        } else {
            current_prose.trim().to_string()
        };

        ToolResult {
            success: true,
            result: json!({"result": result, "context": context}),
        }
    }
}
