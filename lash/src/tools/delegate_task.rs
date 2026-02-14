use std::sync::Arc;

use serde_json::json;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::{
    Agent, AgentConfig, AgentEvent, Message, MessageRole, Part, PartKind, PruneState,
    Session, SessionConfig, Store, ToolDefinition, ToolParam, ToolProvider, ToolResult,
};
use crate::provider::DelegateModels;

/// Delegate tier determines model choice and turn limits.
enum Tier {
    Search,
    Task,
    Deep,
}

/// Shared delegate sub-agent core.
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
        let (tool_name, description, max_turns) = match tier {
            Tier::Search => (
                "delegate_search",
                "Spawn a fast sub-agent for quick lookups, searches, and simple questions.",
                5usize,
            ),
            Tier::Task => (
                "delegate_task",
                "Spawn a sub-agent to handle a task autonomously. It has its own REPL and full tool access.",
                10,
            ),
            Tier::Deep => (
                "delegate_deep",
                "Spawn a thorough sub-agent for complex analysis, multi-step reasoning, or deep investigation.",
                20,
            ),
        };

        Self {
            tools,
            config: AgentConfig {
                model,
                provider: config.provider.clone(),
                sub_agent: true,
                max_turns: Some(max_turns),
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
            description: format!("{} Returns {{\"result\": str, \"context\": [str]}}.", self.description),
            params: vec![ToolParam::typed("prompt", "str")],
            returns: "dict".into(),
            hidden: false,
        }
    }

    async fn execute(&self, args: &serde_json::Value) -> ToolResult {
        let prompt = args
            .get("prompt")
            .and_then(|v| v.as_str())
            .unwrap_or_default();

        if prompt.is_empty() {
            return ToolResult::err(json!("Missing required parameter: prompt"));
        }

        // Create a new session with the base tools (no delegate tools)
        let session =
            match Session::new(Arc::clone(&self.tools), SessionConfig::default()).await {
                Ok(s) => s,
                Err(e) => {
                    return ToolResult::err(json!(format!("Failed to create sub-agent session: {e}")));
                }
            };

        let mut agent = Agent::new(
            session,
            self.config.clone(),
            Arc::clone(&self.store),
            None,
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
            tokio::spawn(async move { agent.run(messages, vec![], event_tx, cancel).await });

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
                AgentEvent::SubAgentDone { .. } => break,
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

        ToolResult::ok(json!({"result": result, "context": context}))
    }
}

// ─── Public delegate types ───

/// Balanced sub-agent for general-purpose autonomous tasks.
pub struct DelegateTask(DelegateInner);

impl DelegateTask {
    pub fn new(
        tools: Arc<dyn ToolProvider>,
        config: &AgentConfig,
        delegate_models: Option<DelegateModels>,
        store: Arc<Store>,
        cancel: CancellationToken,
        parent_id: String,
    ) -> Self {
        Self(DelegateInner::new(tools, config, delegate_models, store, cancel, parent_id, Tier::Task))
    }
}

#[async_trait::async_trait]
impl ToolProvider for DelegateTask {
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![self.0.definition()]
    }
    async fn execute(&self, _name: &str, args: &serde_json::Value) -> ToolResult {
        self.0.execute(args).await
    }
}

/// Fast sub-agent for quick lookups and searches.
pub struct DelegateSearch(DelegateInner);

impl DelegateSearch {
    pub fn new(
        tools: Arc<dyn ToolProvider>,
        config: &AgentConfig,
        delegate_models: Option<DelegateModels>,
        store: Arc<Store>,
        cancel: CancellationToken,
        parent_id: String,
    ) -> Self {
        Self(DelegateInner::new(tools, config, delegate_models, store, cancel, parent_id, Tier::Search))
    }
}

#[async_trait::async_trait]
impl ToolProvider for DelegateSearch {
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![self.0.definition()]
    }
    async fn execute(&self, _name: &str, args: &serde_json::Value) -> ToolResult {
        self.0.execute(args).await
    }
}

/// Thorough sub-agent for deep analysis and complex reasoning.
pub struct DelegateDeep(DelegateInner);

impl DelegateDeep {
    pub fn new(
        tools: Arc<dyn ToolProvider>,
        config: &AgentConfig,
        delegate_models: Option<DelegateModels>,
        store: Arc<Store>,
        cancel: CancellationToken,
        parent_id: String,
    ) -> Self {
        Self(DelegateInner::new(tools, config, delegate_models, store, cancel, parent_id, Tier::Deep))
    }
}

#[async_trait::async_trait]
impl ToolProvider for DelegateDeep {
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![self.0.definition()]
    }
    async fn execute(&self, _name: &str, args: &serde_json::Value) -> ToolResult {
        self.0.execute(args).await
    }
}
