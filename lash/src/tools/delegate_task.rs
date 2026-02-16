use std::sync::Arc;

use serde_json::json;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::provider::DelegateModels;
use crate::{
    Agent, AgentConfig, AgentEvent, Message, MessageRole, Part, PartKind, ProgressSender,
    PruneState, SandboxMessage, Session, SessionConfig, Store, ToolDefinition, ToolParam,
    ToolProvider, ToolResult,
};

/// Delegate tier determines model choice and turn limits.
enum Tier {
    Search,
    Task,
    Deep,
}

/// Shared delegate sub-agent core.
#[allow(dead_code)]
struct DelegateInner {
    tools: Arc<dyn ToolProvider>,
    config: AgentConfig,
    store: Arc<Store>,
    cancel: CancellationToken,
    parent_id: String,
    tool_name: &'static str,
    description: &'static str,
}

fn pick_model(config: &AgentConfig, models: &Option<DelegateModels>, tier: &Tier) -> String {
    if let Some(m) = models {
        match tier {
            Tier::Search => {
                if let Some(ref q) = m.quick {
                    return q.clone();
                }
            }
            Tier::Task => {
                if let Some(ref b) = m.balanced {
                    return b.clone();
                }
            }
            Tier::Deep => {
                if let Some(ref t) = m.thorough {
                    return t.clone();
                }
            }
        }
    }
    config.model.clone()
}

impl DelegateInner {
    fn new(
        tools: Arc<dyn ToolProvider>,
        config: &AgentConfig,
        delegate_models: Option<DelegateModels>,
        store: Arc<Store>,
        cancel: CancellationToken,
        parent_id: String,
        tier: Tier,
    ) -> Self {
        let model = pick_model(config, &delegate_models, &tier);
        let (tool_name, description) = match tier {
            Tier::Search => (
                "delegate_search",
                "Spawn a fast sub-agent for quick lookups, searches, and simple questions.",
            ),
            Tier::Task => (
                "delegate_task",
                "Spawn a sub-agent to handle a task autonomously. It has its own REPL and full tool access.",
            ),
            Tier::Deep => (
                "delegate_deep",
                "Spawn a thorough sub-agent for complex analysis, multi-step reasoning, or deep investigation.",
            ),
        };

        Self {
            tools,
            config: AgentConfig {
                model,
                provider: config.provider.clone(),
                sub_agent: true,
                include_soul: matches!(tier, Tier::Deep),
                max_turns: None,
                llm_log_path: config.llm_log_path.clone(),
                headless: config.headless,
                ..Default::default()
            },
            store,
            cancel,
            parent_id,
            tool_name,
            description,
        }
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: self.tool_name.into(),
            description: format!(
                "{} Returns {{\"result\": str, \"context\": [str]}}.",
                self.description
            ),
            params: vec![ToolParam::typed("prompt", "str")],
            returns: "dict".into(),
            hidden: false,
        }
    }

    async fn execute_streaming(
        &self,
        args: &serde_json::Value,
        progress: Option<&ProgressSender>,
    ) -> ToolResult {
        if let Some(tx) = progress {
            let prompt = args.get("prompt").and_then(|v| v.as_str()).unwrap_or("");
            let _ = tx.send(SandboxMessage {
                text: json!({ "name": self.tool_name, "task": prompt }).to_string(),
                kind: "delegate_start".into(),
            });
        }
        self.execute(args).await
    }

    async fn execute(&self, args: &serde_json::Value) -> ToolResult {
        let prompt = args
            .get("prompt")
            .and_then(|v| v.as_str())
            .unwrap_or_default();

        if prompt.is_empty() {
            return ToolResult::err(json!("Missing required parameter: prompt"));
        }

        // Generate agent ID for the sub-agent (used for task ownership + state)
        let agent_id = uuid::Uuid::new_v4().to_string();

        // Create a new session with the base tools (no delegate tools)
        let session = match Session::new(
            Arc::clone(&self.tools),
            SessionConfig::default(),
            &agent_id,
        )
        .await
        {
            Ok(s) => s,
            Err(e) => {
                return ToolResult::err_fmt(format_args!(
                    "Failed to create sub-agent session: {e}"
                ));
            }
        };

        let mut agent = Agent::new(
            session,
            self.config.clone(),
            Arc::clone(&self.store),
            Some(agent_id),
        );
        let messages = vec![Message {
            id: uuid::Uuid::new_v4().to_string(),
            role: MessageRole::User,
            parts: vec![Part {
                id: "p0".to_string(),
                kind: PartKind::Text,
                content: prompt.to_string(),
                prune_state: PruneState::Intact,
            }],
        }];

        let (event_tx, mut event_rx) = mpsc::channel::<AgentEvent>(100);
        let cancel = self.cancel.child_token();

        // Run agent in a spawned task so we can drain events concurrently
        let run_handle =
            tokio::spawn(async move { agent.run(messages, vec![], event_tx, cancel, 0).await });

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
                AgentEvent::SubAgentDone { .. } => break,
                _ => {}
            }
        }

        // Wait for the agent task to finish
        let _ = run_handle.await;

        let result = if let Some(msg) = final_message {
            msg
        } else if !current_prose.trim().is_empty() {
            current_prose.trim().to_string()
        } else {
            // No done() was called — fall back to accumulated context
            context.join("\n\n")
        };

        ToolResult::ok(json!({"result": result, "context": context}))
    }
}

// ─── Public delegate types ───

macro_rules! delegate_type {
    ($(#[$meta:meta])* $name:ident, $tier:expr) => {
        $(#[$meta])*
        pub struct $name(DelegateInner);

        impl $name {
            pub fn new(
                tools: Arc<dyn ToolProvider>,
                config: &AgentConfig,
                delegate_models: Option<DelegateModels>,
                store: Arc<Store>,
                cancel: CancellationToken,
                parent_id: String,
            ) -> Self {
                Self(DelegateInner::new(tools, config, delegate_models, store, cancel, parent_id, $tier))
            }
        }

        #[async_trait::async_trait]
        impl ToolProvider for $name {
            fn definitions(&self) -> Vec<ToolDefinition> { vec![self.0.definition()] }
            async fn execute(&self, _name: &str, args: &serde_json::Value) -> ToolResult {
                self.0.execute(args).await
            }
            async fn execute_streaming(
                &self,
                _name: &str,
                args: &serde_json::Value,
                progress: Option<&ProgressSender>,
            ) -> ToolResult {
                self.0.execute_streaming(args, progress).await
            }
        }
    };
}

delegate_type!(/// Balanced sub-agent for general-purpose autonomous tasks.
    DelegateTask, Tier::Task);
delegate_type!(/// Fast sub-agent for quick lookups and searches.
    DelegateSearch, Tier::Search);
delegate_type!(/// Thorough sub-agent for deep analysis and complex reasoning.
    DelegateDeep, Tier::Deep);
