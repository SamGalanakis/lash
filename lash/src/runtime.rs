use std::path::PathBuf;
use std::sync::Arc;

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::agent::{
    Agent, AgentConfig, AgentEvent, Message, MessageRole, Part, PartKind, PruneState, TokenUsage,
};
use crate::instructions::{FsInstructionSource, InstructionSource};
use crate::provider::Provider;
use crate::{Session, SessionError, ToolProvider};

/// Runtime execution mode for a turn.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub enum RunMode {
    Normal,
    Plan,
}

/// Host-provided per-turn input.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TurnInput {
    pub user_message: String,
    #[serde(default)]
    pub images_png: Vec<Vec<u8>>,
    #[serde(default)]
    pub mode: Option<RunMode>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plan_file: Option<String>,
}

/// Serializable host-owned session envelope.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct AgentStateEnvelope {
    pub agent_id: String,
    #[serde(default)]
    pub messages: Vec<Message>,
    #[serde(default)]
    pub iteration: usize,
    #[serde(default)]
    pub token_usage: TokenUsage,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task_state: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subagent_state: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repl_snapshot: Option<Vec<u8>>,
}

impl Default for AgentStateEnvelope {
    fn default() -> Self {
        Self {
            agent_id: "root".to_string(),
            messages: Vec::new(),
            iteration: 0,
            token_usage: TokenUsage::default(),
            task_state: None,
            subagent_state: None,
            repl_snapshot: None,
        }
    }
}

/// Output from a completed runtime turn.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TurnResult {
    pub state: AgentStateEnvelope,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub final_message: Option<String>,
    pub done: bool,
}

/// Host event sink for streaming runtime events.
#[async_trait::async_trait]
pub trait EventSink: Send + Sync {
    async fn emit(&self, event: AgentEvent);
}

/// No-op sink useful for callers that only care about final state.
pub struct NoopEventSink;

#[async_trait::async_trait]
impl EventSink for NoopEventSink {
    async fn emit(&self, _event: AgentEvent) {}
}

/// Runtime config used by embedders to construct an engine.
#[derive(Clone)]
pub struct RuntimeConfig {
    pub model: String,
    pub provider: Provider,
    pub max_context_tokens: Option<usize>,
    pub include_soul: bool,
    pub llm_log_path: Option<PathBuf>,
    pub headless: bool,
    pub preamble: Option<String>,
    pub soul: Option<String>,
    pub instruction_source: Arc<dyn InstructionSource>,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        let cfg = AgentConfig::default();
        Self {
            model: cfg.model,
            provider: cfg.provider,
            max_context_tokens: cfg.max_context_tokens,
            include_soul: cfg.include_soul,
            llm_log_path: cfg.llm_log_path,
            headless: cfg.headless,
            preamble: cfg.preamble,
            soul: cfg.soul,
            instruction_source: Arc::new(FsInstructionSource::new()),
        }
    }
}

impl From<RuntimeConfig> for AgentConfig {
    fn from(value: RuntimeConfig) -> Self {
        Self {
            model: value.model,
            provider: value.provider,
            max_context_tokens: value.max_context_tokens,
            sub_agent: false,
            reasoning_effort: None,
            max_turns: None,
            include_soul: value.include_soul,
            llm_log_path: value.llm_log_path,
            headless: value.headless,
            preamble: value.preamble,
            soul: value.soul,
            instruction_source: value.instruction_source,
        }
    }
}

/// Generic runtime engine for CLI or programmatic embedding.
pub struct RuntimeEngine {
    agent: Option<Agent>,
    state: AgentStateEnvelope,
}

impl RuntimeEngine {
    /// Build a runtime from host-provided state + tools.
    pub async fn from_state(
        config: RuntimeConfig,
        tools: Arc<dyn ToolProvider>,
        mut state: AgentStateEnvelope,
    ) -> Result<Self, SessionError> {
        if state.agent_id.is_empty() {
            state.agent_id = "root".to_string();
        }
        let session = Session::new(tools, &state.agent_id, config.headless).await?;
        let mut agent = Agent::new(session, config.into(), Some(state.agent_id.clone()));
        if let Some(snapshot) = state.repl_snapshot.clone() {
            agent.restore(&snapshot).await?;
        }
        Ok(Self {
            agent: Some(agent),
            state,
        })
    }

    /// Wrap an existing agent (used by CLI adapter migration).
    pub fn from_agent(agent: Agent, state: AgentStateEnvelope) -> Self {
        Self {
            agent: Some(agent),
            state,
        }
    }

    /// Export current host-owned state envelope.
    pub fn export_state(&self) -> AgentStateEnvelope {
        self.state.clone()
    }

    /// Replace the host-owned state envelope.
    pub fn set_state(&mut self, state: AgentStateEnvelope) {
        self.state = state;
    }

    /// Update model on the underlying agent.
    pub fn set_model(&mut self, model: String) {
        if let Some(agent) = self.agent.as_mut() {
            agent.set_model(model);
        }
    }

    /// Reset the REPL session on the underlying agent.
    pub async fn reset_session(&mut self) -> Result<(), SessionError> {
        let Some(agent) = self.agent.as_mut() else {
            return Err(SessionError::Protocol(
                "runtime agent not available".to_string(),
            ));
        };
        agent.reset_session().await
    }

    /// Explicitly snapshot REPL state; does not run an LLM turn.
    pub async fn snapshot_repl(&mut self) -> Result<Vec<u8>, SessionError> {
        let Some(agent) = self.agent.as_mut() else {
            return Err(SessionError::Protocol(
                "runtime agent not available".to_string(),
            ));
        };
        let Some(blob) = agent.snapshot().await else {
            return Err(SessionError::Protocol(
                "failed to snapshot runtime repl".to_string(),
            ));
        };
        self.state.repl_snapshot = Some(blob.clone());
        Ok(blob)
    }

    /// Explicitly restore REPL state from an opaque snapshot blob.
    pub async fn restore_repl(&mut self, snapshot: &[u8]) -> Result<(), SessionError> {
        let Some(agent) = self.agent.as_mut() else {
            return Err(SessionError::Protocol(
                "runtime agent not available".to_string(),
            ));
        };
        agent.restore(snapshot).await?;
        self.state.repl_snapshot = Some(snapshot.to_vec());
        Ok(())
    }

    /// Run a single turn and stream events to the host sink.
    pub async fn run_turn(
        &mut self,
        input: TurnInput,
        events: &dyn EventSink,
        cancel: CancellationToken,
    ) -> TurnResult {
        let mut messages = self.state.messages.clone();
        let mode = input.mode.unwrap_or(RunMode::Normal);
        let plan_file = input
            .plan_file
            .clone()
            .unwrap_or_else(|| ".lash_plan".into());
        let mode_msg = match mode {
            RunMode::Plan => Some(format!(
                "## Plan Mode\n\n\
                You are in PLAN mode. Think, explore, and design — do NOT execute changes.\n\n\
                **Rules:**\n\
                - READ-ONLY: Do not modify project files or run destructive commands\n\
                - You MAY use: read_file, glob, grep, ls, web_search, fetch_url, agent_call\n\
                - You MUST NOT use: edit_file, find_replace, write_file (except the plan file), or shell with write commands\n\
                - Exception: Write your plan to `{}` using write_file\n\n\
                **Workflow:**\n\
                1. Understand the request — ask clarifying questions using message(kind=\"final\")\n\
                2. Explore the codebase — read files, search for patterns, understand architecture\n\
                3. Design your approach — consider tradeoffs, identify risks\n\
                4. Write a clear, step-by-step plan to the plan file\n\n\
                When the user switches back to normal mode, you will execute the plan.",
                plan_file
            )),
            RunMode::Normal => input.plan_file.as_ref().and_then(|path| {
                let plan_content = std::fs::read_to_string(path).unwrap_or_default();
                if plan_content.is_empty() {
                    None
                } else {
                    Some(format!(
                        "## Executing Plan\n\n\
                        You are executing a plan from a previous planning session. \
                        Your planning context (exploration, reasoning, findings) is in `_history`.\n\n\
                        **Plan file:** `{}`\n\n\
                        ---\n{}\n---\n\n\
                        **Available context:**\n\
                        - `_history.search(\"pattern\")` — search planning exploration\n\
                        - `_history.user_messages()` — original user requests\n\
                        - `_mem` — persistent memory (fully preserved)\n\n\
                        Execute the plan step by step.",
                        path, plan_content
                    ))
                }
            }),
        };
        if let Some(content) = mode_msg {
            let sys_id = format!("m{}", messages.len());
            messages.push(Message {
                id: sys_id.clone(),
                role: MessageRole::System,
                parts: vec![Part {
                    id: format!("{}.p0", sys_id),
                    kind: PartKind::Text,
                    content,
                    prune_state: PruneState::Intact,
                }],
            });
        }

        let user_id = format!("m{}", messages.len());
        messages.push(Message {
            id: user_id.clone(),
            role: MessageRole::User,
            parts: vec![Part {
                id: format!("{}.p0", user_id),
                kind: PartKind::Text,
                content: input.user_message,
                prune_state: PruneState::Intact,
            }],
        });

        self.run_prepared_turn(messages, input.images_png, events, cancel)
            .await
    }

    /// Run a turn using host-prepared message history.
    pub async fn run_prepared_turn(
        &mut self,
        messages: Vec<Message>,
        images_png: Vec<Vec<u8>>,
        events: &dyn EventSink,
        cancel: CancellationToken,
    ) -> TurnResult {
        let mut agent = self
            .agent
            .take()
            .expect("runtime engine agent must be available");
        let run_offset = self.state.iteration;
        let (event_tx, mut event_rx) = mpsc::channel::<AgentEvent>(100);
        let run_task = tokio::spawn(async move {
            let (new_messages, new_iteration) = agent
                .run(messages, images_png, event_tx, cancel, run_offset)
                .await;
            (agent, new_messages, new_iteration)
        });

        let mut final_message: Option<String> = None;
        let mut done = false;
        let mut usage = self.state.token_usage.clone();
        while let Some(event) = event_rx.recv().await {
            match &event {
                AgentEvent::Message { text, kind } if kind == "final" => {
                    final_message = Some(text.clone());
                }
                AgentEvent::TokenUsage { cumulative, .. } => {
                    usage = cumulative.clone();
                }
                AgentEvent::Done => {
                    done = true;
                }
                _ => {}
            }
            events.emit(event).await;
        }

        let (agent, new_messages, new_iteration) = match run_task.await {
            Ok(v) => v,
            Err(_) => {
                // Keep prior state on panic/join error.
                return TurnResult {
                    state: self.state.clone(),
                    final_message: None,
                    done: false,
                };
            }
        };

        self.agent = Some(agent);
        self.state.messages = new_messages;
        self.state.iteration = new_iteration;
        self.state.token_usage = usage;

        TurnResult {
            state: self.state.clone(),
            final_message,
            done,
        }
    }
}
