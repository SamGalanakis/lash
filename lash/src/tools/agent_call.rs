use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::{Arc, Mutex as StdMutex};

use serde_json::json;
use tokio::sync::Notify;

use crate::plugin::{
    PluginError, PluginFactory, PluginRegistrar, PluginSessionContext, SessionPlugin,
    SessionReadyContext, ToolSurfaceContribution, ToolSurfaceOverride,
};
use crate::provider::AgentModels;
use crate::{
    AgentConfig, AgentEvent, InputItem, PluginSession, ProgressSender, PromptContribution,
    SandboxMessage, SessionConfigOverrides, SessionCreateRequest, SessionStartPoint,
    ToolDefinition, ToolExecutionContext, ToolParam, ToolProvider, ToolResult, TurnInput,
};

use super::require_str;

/// Intelligence tier determines model choice, tool access, and turn limits.
enum Tier {
    Low,
    Medium,
    High,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AgentCallConfig {
    pub low_tier_execution_mode: crate::ExecutionMode,
}

impl Default for AgentCallConfig {
    fn default() -> Self {
        Self {
            low_tier_execution_mode: crate::ExecutionMode::Standard,
        }
    }
}

impl Tier {
    fn from_str(s: &str) -> Option<Self> {
        match s {
            "low" => Some(Tier::Low),
            "medium" => Some(Tier::Medium),
            "high" => Some(Tier::High),
            _ => None,
        }
    }

    fn as_str(&self) -> &'static str {
        match self {
            Tier::Low => "low",
            Tier::Medium => "medium",
            Tier::High => "high",
        }
    }
}

fn pick_model_and_variant(
    config: &AgentConfig,
    models: &Option<AgentModels>,
    tier: &Tier,
) -> (String, Option<String>) {
    if let Some(m) = models {
        match tier {
            Tier::Low => {
                if let Some(ref q) = m.low {
                    let variant = preferred_override_variant(&config.provider, q, tier);
                    return (q.clone(), variant);
                }
            }
            Tier::Medium => {
                if let Some(ref b) = m.medium {
                    let variant = preferred_override_variant(&config.provider, b, tier);
                    return (b.clone(), variant);
                }
            }
            Tier::High => {
                if let Some(ref t) = m.high {
                    let variant = preferred_override_variant(&config.provider, t, tier);
                    return (t.clone(), variant);
                }
            }
        }
    }

    if let Some((model, variant)) = config.provider.default_agent_model(tier.as_str()) {
        return (model.to_string(), variant.map(str::to_string));
    }

    let model = config.model.clone();
    let variant = config
        .provider
        .default_model_variant(&model)
        .map(str::to_string)
        .or_else(|| config.model_variant.clone());
    (model, variant)
}

fn preferred_override_variant(
    provider: &crate::Provider,
    model: &str,
    tier: &Tier,
) -> Option<String> {
    let tier_variant = tier.as_str();
    if provider.supported_variants(model).contains(&tier_variant) {
        return Some(tier_variant.to_string());
    }
    provider.default_model_variant(model).map(str::to_string)
}

fn low_tier_denied_tools() -> HashSet<&'static str> {
    [
        "apply_patch",
        "agent_call",
        "agent_result",
        "agent_kill",
        "ask",
    ]
    .into_iter()
    .collect()
}

/// A running delegated child session managed by AgentCall.
struct RunningAgent {
    session_id: String,
    turn_id: String,
    host: Arc<dyn crate::SessionManager>,
    /// Accumulated prose output from the agent (drainable).
    buffer: Arc<StdMutex<String>>,
    /// Final result once the agent completes.
    result: Arc<StdMutex<Option<serde_json::Value>>>,
    /// Notified when the agent finishes.
    done_notify: Arc<Notify>,
}

/// Single agent-call tool that spawns delegated child sessions at different intelligence tiers.
/// Returns a handle immediately; use agent_result/agent_kill to interact.
pub struct AgentCall {
    #[cfg(test)]
    plugins: Arc<PluginSession>,
    config: AgentConfig,
    execution_mode: crate::ExecutionMode,
    tool_config: AgentCallConfig,
    agent_models: Option<AgentModels>,
    agents: Arc<StdMutex<HashMap<String, RunningAgent>>>,
}

impl AgentCall {
    pub fn new(
        plugins: Arc<PluginSession>,
        config: &AgentConfig,
        tool_config: AgentCallConfig,
        agent_models: Option<AgentModels>,
    ) -> Self {
        #[cfg(not(test))]
        let _ = &plugins;
        Self {
            #[cfg(test)]
            plugins,
            config: config.clone(),
            execution_mode: config.execution_mode,
            tool_config,
            agent_models,
            agents: Arc::new(StdMutex::new(HashMap::new())),
        }
    }

    fn build_agent_config(&self, tier: &Tier) -> AgentConfig {
        let (model, model_variant) = pick_model_and_variant(&self.config, &self.agent_models, tier);
        AgentConfig {
            model,
            model_variant,
            session_id: self.config.session_id.clone(),
            provider: self.config.provider.clone(),
            sub_agent: true,
            include_soul: matches!(tier, Tier::High),
            max_context_tokens: self.config.max_context_tokens,
            max_turns: None,
            llm_log_path: self.config.llm_log_path.clone(),
            prompt_overrides: self.config.prompt_overrides.clone(),
            prompt_renderer: Arc::clone(&self.config.prompt_renderer),
            instruction_source: self.config.instruction_source.clone(),
            execution_mode: match tier {
                Tier::Low => self.tool_config.low_tier_execution_mode,
                Tier::Medium | Tier::High => self.config.execution_mode,
            },
            ..Default::default()
        }
    }

    fn tool_surface_for_tier(&self, tier: &Tier) -> ToolSurfaceContribution {
        let denied = match tier {
            Tier::Low => low_tier_denied_tools()
                .into_iter()
                .map(str::to_string)
                .collect::<BTreeSet<_>>(),
            Tier::Medium | Tier::High => BTreeSet::from(["ask".to_string()]),
        };
        ToolSurfaceContribution {
            overrides: denied
                .into_iter()
                .map(|tool_name| ToolSurfaceOverride {
                    tool_name,
                    enabled: Some(false),
                    injected: Some(false),
                })
                .collect(),
            tool_list_notes: Vec::new(),
        }
    }

    #[cfg(test)]
    fn session_plugins_for_tier(
        &self,
        agent_id: &str,
        tier: &Tier,
    ) -> Result<Arc<PluginSession>, PluginError> {
        let execution_mode = self.build_agent_config(tier).execution_mode;
        self.plugins.fork_for_agent_with_tool_surface(
            agent_id,
            execution_mode,
            self.tool_surface_for_tier(tier),
        )
    }

    #[cfg(test)]
    fn visible_tool_names_for_tier(&self, tier: &Tier) -> Result<Vec<String>, PluginError> {
        let session = self.session_plugins_for_tier("__tier_probe__", tier)?;
        let surface = session.execution_surface(
            session.agent_id(),
            self.build_agent_config(tier).execution_mode,
        );
        Ok(surface
            .enabled_tools()
            .into_iter()
            .map(|tool| tool.name)
            .collect())
    }

    fn build_create_request(&self, agent_id: String, tier: &Tier) -> SessionCreateRequest {
        let agent_config = self.build_agent_config(tier);
        SessionCreateRequest {
            agent_id: Some(agent_id),
            start: SessionStartPoint::Empty,
            config_overrides: SessionConfigOverrides {
                model: Some(agent_config.model),
                model_variant: agent_config.model_variant,
                max_context_tokens: agent_config.max_context_tokens,
                execution_mode: Some(agent_config.execution_mode),
                context_folding: Some(agent_config.context_folding),
                session_id: agent_config.session_id,
                max_turns: agent_config.max_turns,
                include_soul: Some(agent_config.include_soul),
                sub_agent: Some(true),
            },
            tool_surface: self.tool_surface_for_tier(tier),
            initial_messages: Vec::new(),
        }
    }

    /// Spawn a child session in the background and return its handle ID.
    async fn spawn_agent(
        &self,
        args: &serde_json::Value,
        context: &ToolExecutionContext,
    ) -> ToolResult {
        let prompt = args
            .get("prompt")
            .and_then(|v| v.as_str())
            .unwrap_or_default();

        if prompt.is_empty() {
            return ToolResult::err(json!("Missing required parameter: prompt"));
        }

        let intelligence = args
            .get("intelligence")
            .and_then(|v| v.as_str())
            .unwrap_or_default();

        let tier = match Tier::from_str(intelligence) {
            Some(t) => t,
            None => {
                return ToolResult::err(json!(
                    "Missing or invalid 'intelligence' parameter: must be \"low\", \"medium\", or \"high\""
                ));
            }
        };

        let schema = args
            .get("schema")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let agent_id = uuid::Uuid::new_v4().to_string();
        let session = match context
            .host
            .create_session(self.build_create_request(agent_id, &tier))
            .await
        {
            Ok(session) => session,
            Err(err) => {
                return ToolResult::err_fmt(format_args!(
                    "Failed to create delegate session: {err}"
                ));
            }
        };
        let agent_execution_mode = session.config.execution_mode;

        // Build the user message, optionally appending schema instructions
        let user_content = if let Some(ref schema_str) = schema {
            let model_name = serde_json::from_str::<serde_json::Value>(schema_str)
                .ok()
                .and_then(|v| v.get("title").and_then(|t| t.as_str()).map(String::from))
                .unwrap_or_else(|| "Result".to_string());
            match agent_execution_mode {
                crate::ExecutionMode::Repl => format!(
                    "{prompt}\n\nInside `<repl>`, end with `finish {{ ... }}` using a single JSON-compatible record matching this schema exactly:\n{schema_str}\n\nDo not return prose in `finish`. The record should represent a `{model_name}` object."
                ),
                crate::ExecutionMode::Standard => format!(
                    "{prompt}\n\nReturn your final answer as a single JSON object matching this schema exactly:\n{schema_str}\n\nDo not wrap it in markdown fences or extra commentary."
                ),
            }
        } else {
            prompt.to_string()
        };

        let mut turn = match context
            .host
            .start_turn_stream(
                &session.session_id,
                TurnInput {
                    items: vec![InputItem::Text { text: user_content }],
                    image_blobs: HashMap::new(),
                    mode: None,
                },
            )
            .await
        {
            Ok(turn) => turn,
            Err(err) => {
                let _ = context.host.close_session(&session.session_id).await;
                return ToolResult::err_fmt(format_args!(
                    "Failed to start delegate session: {err}"
                ));
            }
        };

        // Set up shared state
        let buffer = Arc::new(StdMutex::new(String::new()));
        let context_chunks = Arc::new(StdMutex::new(Vec::new()));
        let result: Arc<StdMutex<Option<serde_json::Value>>> = Arc::new(StdMutex::new(None));
        let done_notify = Arc::new(Notify::new());

        let buf_clone = buffer.clone();
        let ctx_clone = context_chunks.clone();
        let res_clone = result.clone();
        let done_clone = done_notify.clone();
        let host = Arc::clone(&context.host);
        let prompt_for_meta = prompt.to_string();
        let session_id = session.session_id.clone();
        let turn_id = turn.turn_id.clone();
        let session_config = session.config.clone();

        // Spawn event drainer
        tokio::spawn(async move {
            let mut final_message: Option<String> = None;
            let mut current_prose = String::new();

            while let Some(event) = turn.events.recv().await {
                match event {
                    AgentEvent::TextDelta { content } => {
                        current_prose.push_str(&content);
                        buf_clone.lock().unwrap().push_str(&content);
                    }
                    AgentEvent::Message { text, kind } => {
                        if kind == "final" {
                            final_message = Some(text);
                        }
                    }
                    AgentEvent::CodeBlock { .. } => {
                        let trimmed = current_prose.trim().to_string();
                        if !trimmed.is_empty() {
                            ctx_clone.lock().unwrap().push(trimmed);
                        }
                        current_prose.clear();
                    }
                    _ => {}
                }
            }

            let assembled = host.await_turn(&turn_id).await;
            let mut result_json = match assembled {
                Ok(turn) => {
                    let result_text = if let Some(msg) = final_message {
                        msg
                    } else if !current_prose.trim().is_empty() {
                        current_prose.trim().to_string()
                    } else if !turn.assistant_output.raw_text.trim().is_empty() {
                        turn.assistant_output.raw_text.trim().to_string()
                    } else {
                        ctx_clone.lock().unwrap().join("\n\n")
                    };
                    let status = match turn.status {
                        crate::TurnStatus::Completed => "completed",
                        crate::TurnStatus::Interrupted => "interrupted",
                        crate::TurnStatus::Failed => "failed",
                    };
                    json!({
                        "result": result_text,
                        "context": ctx_clone.lock().unwrap().clone(),
                        "_sub_agent": {
                            "task": prompt_for_meta,
                            "session_id": session_id,
                            "turn_id": turn_id,
                            "config": session_config,
                            "usage": turn.token_usage,
                            "tool_calls": turn.tool_calls.len(),
                            "iterations": turn.state.iteration,
                            "status": status,
                        }
                    })
                }
                Err(err) => json!({
                    "result": "",
                    "context": [],
                    "_sub_agent": {
                        "task": prompt_for_meta,
                        "session_id": session_id,
                        "turn_id": turn_id,
                        "config": session_config,
                        "usage": crate::TokenUsage::default(),
                        "tool_calls": 0,
                        "iterations": 0,
                        "status": "failed",
                        "error": err.to_string(),
                    }
                }),
            };

            if result_json
                .get("result")
                .and_then(|value| value.as_str())
                .is_some_and(|value| value.is_empty())
                && !current_prose.trim().is_empty()
            {
                result_json["result"] = json!(current_prose.trim());
            }

            *res_clone.lock().unwrap() = Some(result_json);
            done_clone.notify_waiters();
        });

        // Store the running agent
        self.agents.lock().unwrap().insert(
            session.session_id.clone(),
            RunningAgent {
                session_id: session.session_id.clone(),
                turn_id: turn.turn_id,
                host: Arc::clone(&context.host),
                buffer,
                result,
                done_notify,
            },
        );

        ToolResult::ok(json!({
            "__handle__": "agent",
            "id": session.session_id,
            "config": session.config,
        }))
    }

    /// Wait for agent completion and return the result.
    async fn agent_result(
        &self,
        id: &str,
        timeout: Option<f64>,
        progress: Option<&ProgressSender>,
    ) -> ToolResult {
        let (result_arc, done_notify, buffer, session_id, turn_id, host) = {
            let agents = self.agents.lock().unwrap();
            match agents.get(id) {
                Some(a) => (
                    a.result.clone(),
                    a.done_notify.clone(),
                    a.buffer.clone(),
                    a.session_id.clone(),
                    a.turn_id.clone(),
                    Arc::clone(&a.host),
                ),
                None => return ToolResult::err_fmt(format_args!("No agent with id: {id}")),
            }
        };

        let deadline =
            timeout.map(|t| tokio::time::Instant::now() + std::time::Duration::from_secs_f64(t));

        let mut sent_len = 0usize;

        loop {
            let done = result_arc.lock().unwrap().is_some();

            // Stream new output to progress sender
            if let Some(tx) = progress {
                let buf = buffer.lock().unwrap();
                if buf.len() > sent_len {
                    let new_chunk = &buf[sent_len..];
                    let _ = tx.send(SandboxMessage {
                        text: new_chunk.to_string(),
                        kind: "tool_output".into(),
                    });
                    sent_len = buf.len();
                }
            }

            if done {
                break;
            }

            if let Some(dl) = deadline
                && tokio::time::Instant::now() >= dl
            {
                let _ = host.cancel_turn(&turn_id).await;
                let _ =
                    tokio::time::timeout(std::time::Duration::from_secs(5), done_notify.notified())
                        .await;
                let _ = host.close_session(&session_id).await;
                self.agents.lock().unwrap().remove(id);
                return ToolResult::err(json!("Agent timed out"));
            }

            tokio::select! {
                _ = done_notify.notified() => {}
                _ = tokio::time::sleep(std::time::Duration::from_millis(50)) => {}
            }
        }

        let result = result_arc.lock().unwrap().clone().unwrap_or(json!(null));
        ToolResult::ok(result)
    }

    /// Cancel a running agent.
    async fn agent_kill(&self, id: &str) -> ToolResult {
        let (turn_id, session_id, done_notify, host) = {
            let agents = self.agents.lock().unwrap();
            match agents.get(id) {
                Some(a) => (
                    a.turn_id.clone(),
                    a.session_id.clone(),
                    a.done_notify.clone(),
                    Arc::clone(&a.host),
                ),
                None => return ToolResult::err_fmt(format_args!("No agent with id: {id}")),
            }
        };

        let _ = host.cancel_turn(&turn_id).await;
        let _ =
            tokio::time::timeout(std::time::Duration::from_secs(5), done_notify.notified()).await;
        let _ = host.close_session(&session_id).await;
        self.agents.lock().unwrap().remove(id);
        ToolResult::ok(json!(null))
    }
}

#[async_trait::async_trait]
impl ToolProvider for AgentCall {
    fn definitions(&self) -> Vec<ToolDefinition> {
        agent_call_definitions(
            self.execution_mode,
            self.tool_config.low_tier_execution_mode,
        )
    }

    async fn execute(&self, name: &str, args: &serde_json::Value) -> ToolResult {
        match name {
            "agent_call" => ToolResult::err_fmt("agent_call requires session context"),
            "agent_result" => {
                let id = match require_str(args, "id") {
                    Ok(s) => s,
                    Err(e) => return e,
                };
                let timeout = args.get("timeout").and_then(|v| v.as_f64());
                self.agent_result(id, timeout, None).await
            }
            "agent_kill" => {
                let id = match require_str(args, "id") {
                    Ok(s) => s,
                    Err(e) => return e,
                };
                self.agent_kill(id).await
            }
            _ => ToolResult::err_fmt(format_args!("Unknown tool: {name}")),
        }
    }

    async fn execute_streaming(
        &self,
        name: &str,
        args: &serde_json::Value,
        progress: Option<&ProgressSender>,
    ) -> ToolResult {
        match name {
            "agent_result" => {
                let id = match require_str(args, "id") {
                    Ok(s) => s,
                    Err(e) => return e,
                };
                let timeout = args.get("timeout").and_then(|v| v.as_f64());
                self.agent_result(id, timeout, progress).await
            }
            "agent_call" => ToolResult::err_fmt("agent_call requires session context"),
            _ => self.execute(name, args).await,
        }
    }

    async fn execute_with_context(
        &self,
        name: &str,
        args: &serde_json::Value,
        context: &ToolExecutionContext,
    ) -> ToolResult {
        match name {
            "agent_call" => self.spawn_agent(args, context).await,
            _ => self.execute(name, args).await,
        }
    }

    async fn execute_streaming_with_context(
        &self,
        name: &str,
        args: &serde_json::Value,
        context: &ToolExecutionContext,
        progress: Option<&ProgressSender>,
    ) -> ToolResult {
        match name {
            "agent_call" => self.spawn_agent(args, context).await,
            "agent_result" => {
                let id = match require_str(args, "id") {
                    Ok(s) => s,
                    Err(e) => return e,
                };
                let timeout = args.get("timeout").and_then(|v| v.as_f64());
                self.agent_result(id, timeout, progress).await
            }
            _ => self.execute_with_context(name, args, context).await,
        }
    }
}

fn agent_call_prompt_contributions() -> Vec<PromptContribution> {
    vec![
        PromptContribution::guidance(
            "### Delegation\nUse `agent_call` for scoped sub-tasks. Each delegate runs in its own session. Prefer low-intelligence delegates for read-only lookup or summarization work, and avoid overlapping file edits across concurrent delegates.",
        ),
        PromptContribution::guidance(
            "### Agent Lifecycle\n`agent_result(id)` blocks until the child session finishes and returns an object in `result.value` with fields like `result`, `context`, and `_sub_agent` (including session/config metadata). The agent ID remains valid afterwards, so you can call `agent_result` again or use `agent_kill` to clean up.",
        ),
    ]
}

fn agent_call_definitions(
    execution_mode: crate::ExecutionMode,
    low_tier_execution_mode: crate::ExecutionMode,
) -> Vec<ToolDefinition> {
    let low_tier_summary = match low_tier_execution_mode {
        crate::ExecutionMode::Standard => "by default, low runs in standard mode",
        crate::ExecutionMode::Repl => "in this session, low runs in repl mode",
    };
    let (agent_call_description, agent_call_examples) = match execution_mode {
        crate::ExecutionMode::Repl => (
            format!(
                "Spawn a child session for scoped work and return a handle. In REPL mode, use `call agent_result {{ id: handle.value.id }}` or `call agent_kill {{ id: handle.value.id }}` with the returned id. Use `intelligence=\"low\"` for fast read-only work; {}. Medium/high inherit the parent execution mode.",
                low_tier_summary
            ),
            vec![
                r#"handle = call agent_call { prompt: "Summarize the auth flow", intelligence: "low" }"#.into(),
                r#"result = call agent_result { id: handle.value.id }"#.into(),
            ],
        ),
        crate::ExecutionMode::Standard => (
            format!(
                "Spawn a child session for scoped work and return a handle. Use `agent_result(id)` or `agent_kill(id)` with the returned id. Use `intelligence=\"low\"` for fast read-only work; {}. Medium/high inherit the parent execution mode.",
                low_tier_summary
            ),
            vec![
                "handle = agent_call(prompt=\"Summarize the auth flow\", intelligence=\"low\")"
                    .into(),
            ],
        ),
    };
    vec![
        ToolDefinition {
            name: "agent_call".into(),
            description: agent_call_description,
            params: vec![
                ToolParam::typed("prompt", "str"),
                ToolParam::typed("intelligence", "str"),
                ToolParam {
                    name: "schema".into(),
                    r#type: "str".into(),
                    description: "JSON schema to include in the agent's prompt as output guidance (not enforced at runtime)".into(),
                    required: false,
                },
            ],
            returns: "dict".into(),
            examples: agent_call_examples,
            enabled: true,
            injected: true,
        },
        ToolDefinition {
            name: "agent_result".into(),
            description: "Wait for a child session to finish and return its final result.".into(),
            params: vec![
                ToolParam::typed("id", "str"),
                ToolParam::optional("timeout", "float"),
            ],
            returns: "dict".into(),
            examples: vec![],
            enabled: true,
            injected: true,
        },
        ToolDefinition {
            name: "agent_kill".into(),
            description: "Cancel a running child session.".into(),
            params: vec![ToolParam::typed("id", "str")],
            returns: "None".into(),
            examples: vec![],
            enabled: true,
            injected: true,
        },
    ]
}

struct AgentCallPluginProvider {
    config: AgentConfig,
    execution_mode: crate::ExecutionMode,
    tool_config: AgentCallConfig,
    agent_models: Option<AgentModels>,
    delegate: StdMutex<Option<Arc<AgentCall>>>,
}

impl AgentCallPluginProvider {
    fn bind(&self, session: Arc<PluginSession>) {
        let delegate = AgentCall::new(
            session,
            &self.config,
            self.tool_config,
            self.agent_models.clone(),
        );
        *self
            .delegate
            .lock()
            .expect("agent_call delegate lock poisoned") = Some(Arc::new(delegate));
    }
}

#[async_trait::async_trait]
impl ToolProvider for AgentCallPluginProvider {
    fn definitions(&self) -> Vec<ToolDefinition> {
        agent_call_definitions(
            self.execution_mode,
            self.tool_config.low_tier_execution_mode,
        )
    }

    async fn execute(&self, name: &str, args: &serde_json::Value) -> ToolResult {
        let delegate = self
            .delegate
            .lock()
            .expect("agent_call delegate lock poisoned")
            .clone();
        let Some(delegate) = delegate else {
            return ToolResult::err_fmt("agent_call plugin is not ready");
        };
        delegate.execute(name, args).await
    }

    async fn execute_with_context(
        &self,
        name: &str,
        args: &serde_json::Value,
        context: &ToolExecutionContext,
    ) -> ToolResult {
        let delegate = self
            .delegate
            .lock()
            .expect("agent_call delegate lock poisoned")
            .clone();
        let Some(delegate) = delegate else {
            return ToolResult::err_fmt("agent_call plugin is not ready");
        };
        delegate.execute_with_context(name, args, context).await
    }

    async fn execute_streaming(
        &self,
        name: &str,
        args: &serde_json::Value,
        progress: Option<&ProgressSender>,
    ) -> ToolResult {
        let delegate = self
            .delegate
            .lock()
            .expect("agent_call delegate lock poisoned")
            .clone();
        let Some(delegate) = delegate else {
            return ToolResult::err_fmt("agent_call plugin is not ready");
        };
        delegate.execute_streaming(name, args, progress).await
    }

    async fn execute_streaming_with_context(
        &self,
        name: &str,
        args: &serde_json::Value,
        context: &ToolExecutionContext,
        progress: Option<&ProgressSender>,
    ) -> ToolResult {
        let delegate = self
            .delegate
            .lock()
            .expect("agent_call delegate lock poisoned")
            .clone();
        let Some(delegate) = delegate else {
            return ToolResult::err_fmt("agent_call plugin is not ready");
        };
        delegate
            .execute_streaming_with_context(name, args, context, progress)
            .await
    }
}

pub struct AgentCallPluginFactory {
    config: AgentConfig,
    tool_config: AgentCallConfig,
    agent_models: Option<AgentModels>,
}

impl AgentCallPluginFactory {
    pub fn new(
        config: AgentConfig,
        tool_config: AgentCallConfig,
        agent_models: Option<AgentModels>,
    ) -> Self {
        Self {
            config,
            tool_config,
            agent_models,
        }
    }
}

impl PluginFactory for AgentCallPluginFactory {
    fn id(&self) -> &'static str {
        "agent_call"
    }

    fn build(&self, ctx: &PluginSessionContext) -> Result<Arc<dyn SessionPlugin>, PluginError> {
        let mut config = self.config.clone();
        config.execution_mode = ctx.execution_mode;
        Ok(Arc::new(AgentCallPlugin {
            provider: Arc::new(AgentCallPluginProvider {
                config,
                execution_mode: ctx.execution_mode,
                tool_config: self.tool_config,
                agent_models: self.agent_models.clone(),
                delegate: StdMutex::new(None),
            }),
        }))
    }
}

struct AgentCallPlugin {
    provider: Arc<AgentCallPluginProvider>,
}

impl SessionPlugin for AgentCallPlugin {
    fn id(&self) -> &'static str {
        "agent_call"
    }

    fn register(&self, reg: &mut PluginRegistrar) -> Result<(), PluginError> {
        reg.tools()
            .provider(Arc::clone(&self.provider) as Arc<dyn ToolProvider>)?;
        reg.prompt().contribute(Arc::new(|_ctx| {
            Box::pin(async move { Ok(agent_call_prompt_contributions()) })
        }));
        Ok(())
    }

    fn session_ready(&self, ctx: SessionReadyContext) -> Result<(), PluginError> {
        let session = ctx
            .host
            .session(&ctx.agent_id)
            .map_err(|err| PluginError::Session(err.to_string()))?;
        self.provider.bind(session);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::Provider;
    use std::sync::Mutex;

    #[derive(Default)]
    struct MockSessionManager {
        created: Mutex<Vec<crate::SessionCreateRequest>>,
        cancelled: Mutex<Vec<String>>,
        closed: Mutex<Vec<String>>,
    }

    #[async_trait::async_trait]
    impl crate::SessionManager for MockSessionManager {
        async fn snapshot_current(&self) -> Result<crate::SessionSnapshot, crate::PluginError> {
            Ok(crate::AgentStateEnvelope::default())
        }

        async fn snapshot_session(
            &self,
            _session_id: &str,
        ) -> Result<crate::SessionSnapshot, crate::PluginError> {
            Ok(crate::AgentStateEnvelope::default())
        }

        async fn tool_catalog(
            &self,
            _session_id: &str,
        ) -> Result<Vec<serde_json::Value>, crate::PluginError> {
            Ok(Vec::new())
        }

        async fn create_session(
            &self,
            request: crate::SessionCreateRequest,
        ) -> Result<crate::SessionHandle, crate::PluginError> {
            self.created.lock().unwrap().push(request.clone());
            Ok(crate::SessionHandle {
                session_id: "child-session".to_string(),
                config: crate::SessionConfigSnapshot {
                    provider_kind: crate::provider::ProviderKind::Codex,
                    model: request
                        .config_overrides
                        .model
                        .unwrap_or_else(|| "mock-model".to_string()),
                    model_variant: request.config_overrides.model_variant,
                    execution_mode: request
                        .config_overrides
                        .execution_mode
                        .unwrap_or(crate::ExecutionMode::Standard),
                    context_folding: request.config_overrides.context_folding.unwrap_or_default(),
                    context_window: request
                        .config_overrides
                        .max_context_tokens
                        .map(|tokens| tokens as u64),
                    max_turns: request.config_overrides.max_turns.map(|turns| turns as u64),
                    include_soul: request.config_overrides.include_soul.unwrap_or(false),
                    sub_agent: request.config_overrides.sub_agent.unwrap_or(false),
                },
            })
        }

        async fn close_session(&self, session_id: &str) -> Result<(), crate::PluginError> {
            self.closed.lock().unwrap().push(session_id.to_string());
            Ok(())
        }

        async fn start_turn_stream(
            &self,
            session_id: &str,
            _input: crate::TurnInput,
        ) -> Result<crate::plugin::SessionTurnHandle, crate::PluginError> {
            let (tx, rx) = tokio::sync::mpsc::channel(4);
            tokio::spawn(async move {
                let _ = tx
                    .send(crate::AgentEvent::TextDelta {
                        content: "delegate".to_string(),
                    })
                    .await;
                let _ = tx
                    .send(crate::AgentEvent::Message {
                        text: "delegate result".to_string(),
                        kind: "final".to_string(),
                    })
                    .await;
            });
            Ok(crate::plugin::SessionTurnHandle {
                turn_id: "turn-1".to_string(),
                session_id: session_id.to_string(),
                config: crate::SessionConfigSnapshot {
                    provider_kind: crate::provider::ProviderKind::Codex,
                    model: "gpt-5.3-codex-spark".to_string(),
                    model_variant: Some("low".to_string()),
                    execution_mode: crate::ExecutionMode::Standard,
                    context_folding: crate::ContextFoldingConfig::default(),
                    context_window: Some(128_000),
                    max_turns: None,
                    include_soul: false,
                    sub_agent: true,
                },
                events: rx,
            })
        }

        async fn await_turn(
            &self,
            _turn_id: &str,
        ) -> Result<crate::AssembledTurn, crate::PluginError> {
            Ok(crate::AssembledTurn {
                state: crate::AgentStateEnvelope {
                    agent_id: "child-session".to_string(),
                    execution_mode: crate::ExecutionMode::Standard,
                    context_folding: crate::ContextFoldingConfig::default(),
                    iteration: 2,
                    ..Default::default()
                },
                status: crate::TurnStatus::Completed,
                assistant_output: crate::AssistantOutput {
                    safe_text: "delegate result".to_string(),
                    raw_text: "delegate result".to_string(),
                    state: crate::OutputState::Usable,
                },
                done_reason: crate::DoneReason::ModelStop,
                execution: crate::ExecutionSummary {
                    mode: crate::ExecutionMode::Standard,
                    had_tool_calls: true,
                    had_code_execution: false,
                },
                token_usage: crate::TokenUsage {
                    input_tokens: 3,
                    output_tokens: 4,
                    cached_input_tokens: 0,
                    reasoning_tokens: 2,
                },
                tool_calls: vec![crate::ToolCallRecord {
                    tool: "read_file".to_string(),
                    args: json!({"path":"foo"}),
                    result: json!("bar"),
                    success: true,
                    duration_ms: 12,
                }],
                code_outputs: Vec::new(),
                errors: Vec::new(),
            })
        }

        async fn cancel_turn(&self, turn_id: &str) -> Result<(), crate::PluginError> {
            self.cancelled.lock().unwrap().push(turn_id.to_string());
            Ok(())
        }
    }

    fn codex_provider() -> Provider {
        Provider::Codex {
            access_token: "tok".into(),
            refresh_token: "ref".into(),
            expires_at: u64::MAX,
            account_id: Some("acct".into()),
            options: crate::provider::ProviderOptions::default(),
        }
    }

    #[test]
    fn codex_uses_explicit_tier_models_when_no_overrides() {
        let config = AgentConfig {
            model: "custom-parent-model".to_string(),
            provider: codex_provider(),
            ..Default::default()
        };

        let (m_low, r_low) = pick_model_and_variant(&config, &None, &Tier::Low);
        assert_eq!(m_low, "gpt-5.3-codex-spark");
        assert_eq!(r_low.as_deref(), Some("low"));

        let (m_mid, r_mid) = pick_model_and_variant(&config, &None, &Tier::Medium);
        assert_eq!(m_mid, "gpt-5.4");
        assert_eq!(r_mid.as_deref(), Some("medium"));

        let (m_high, r_high) = pick_model_and_variant(&config, &None, &Tier::High);
        assert_eq!(m_high, "gpt-5.4");
        assert_eq!(r_high.as_deref(), Some("high"));
    }

    #[test]
    fn override_model_keeps_override_and_inferrs_reasoning() {
        let config = AgentConfig {
            model: "custom-parent-model".to_string(),
            provider: codex_provider(),
            ..Default::default()
        };
        let models = Some(AgentModels {
            low: None,
            medium: None,
            high: Some("gpt-5.4".to_string()),
        });

        let (m, r) = pick_model_and_variant(&config, &models, &Tier::High);
        assert_eq!(m, "gpt-5.4");
        assert_eq!(r.as_deref(), Some("high"));
    }

    #[test]
    fn agent_call_docs_are_mode_specific() {
        let repl = test_agent_call(crate::ExecutionMode::Repl)
            .definitions()
            .into_iter()
            .find(|def| def.name == "agent_call")
            .expect("agent_call definition");
        let standard = test_agent_call(crate::ExecutionMode::Standard)
            .definitions()
            .into_iter()
            .find(|def| def.name == "agent_call")
            .expect("agent_call definition");

        assert!(repl.description.contains("return a handle"));
        assert!(
            repl.description
                .contains("call agent_result { id: handle.value.id }")
        );
        assert!(!standard.description.contains("AgentHandle"));
        assert!(
            !standard
                .description
                .contains("call agent_result { id: handle.value.id }")
        );
    }

    #[test]
    fn agent_lifecycle_tools_are_prompt_injected() {
        let defs = test_agent_call(crate::ExecutionMode::Standard).definitions();
        for name in ["agent_call", "agent_result", "agent_kill"] {
            assert!(defs.iter().any(|def| def.name == name && def.injected));
        }
    }

    struct MockBaseTool;

    #[async_trait::async_trait]
    impl ToolProvider for MockBaseTool {
        fn definitions(&self) -> Vec<ToolDefinition> {
            vec![
                ToolDefinition {
                    name: "mock_base".into(),
                    description: "mock".into(),
                    params: vec![],
                    returns: "None".into(),
                    examples: vec![],
                    enabled: true,
                    injected: true,
                },
                ToolDefinition {
                    name: "read_file".into(),
                    description: "mock read".into(),
                    params: vec![],
                    returns: "str".into(),
                    examples: vec![],
                    enabled: true,
                    injected: true,
                },
                ToolDefinition {
                    name: "apply_patch".into(),
                    description: "mock patch".into(),
                    params: vec![],
                    returns: "None".into(),
                    examples: vec![],
                    enabled: true,
                    injected: true,
                },
            ]
        }

        async fn execute(&self, _name: &str, _args: &serde_json::Value) -> ToolResult {
            ToolResult::ok(serde_json::json!(null))
        }
    }

    fn test_agent_call(execution_mode: crate::ExecutionMode) -> AgentCall {
        let config = AgentConfig {
            model: "custom-parent-model".to_string(),
            provider: codex_provider(),
            execution_mode,
            ..Default::default()
        };
        let plugins = crate::PluginHost::new(vec![
            Arc::new(crate::PluginSpecFactory::new(
                "mock_base",
                Arc::new(|_ctx| {
                    Ok(crate::PluginSpec::new()
                        .with_tool_provider(Arc::new(MockBaseTool) as Arc<dyn ToolProvider>))
                }),
            )),
            Arc::new(AgentCallPluginFactory::new(
                config.clone(),
                AgentCallConfig::default(),
                None,
            )),
        ])
        .build_session("root", execution_mode, None)
        .expect("plugins");
        AgentCall::new(plugins, &config, AgentCallConfig::default(), None)
    }

    #[test]
    fn low_tier_subagent_tools_do_not_include_agent_call() {
        let agent_call = test_agent_call(crate::ExecutionMode::Standard);
        let tool_names = agent_call
            .visible_tool_names_for_tier(&Tier::Low)
            .expect("tier tools");
        assert!(!tool_names.iter().any(|n| n == "agent_call"));
        assert!(tool_names.iter().any(|n| n == "mock_base"));
        assert!(tool_names.iter().any(|n| n == "read_file"));
        assert!(!tool_names.iter().any(|n| n == "apply_patch"));
    }

    #[test]
    fn medium_and_high_tier_subagent_tools_include_agent_call() {
        let agent_call = test_agent_call(crate::ExecutionMode::Standard);

        let medium_names = agent_call
            .visible_tool_names_for_tier(&Tier::Medium)
            .expect("medium tools");
        assert!(medium_names.iter().any(|n| n == "agent_call"));
        assert!(medium_names.iter().any(|n| n == "mock_base"));
        assert!(medium_names.iter().any(|n| n == "apply_patch"));

        let high_names = agent_call
            .visible_tool_names_for_tier(&Tier::High)
            .expect("high tools");
        assert!(high_names.iter().any(|n| n == "agent_call"));
        assert!(high_names.iter().any(|n| n == "mock_base"));
        assert!(high_names.iter().any(|n| n == "apply_patch"));
    }

    #[test]
    fn low_tier_subagent_config_is_read_only() {
        let agent_call = test_agent_call(crate::ExecutionMode::Standard);
        let low = agent_call.build_agent_config(&Tier::Low);
        let medium = agent_call.build_agent_config(&Tier::Medium);
        assert_eq!(
            low.execution_mode,
            AgentCallConfig::default().low_tier_execution_mode
        );
        assert_eq!(medium.execution_mode, agent_call.config.execution_mode);
    }

    #[test]
    fn low_tier_execution_mode_is_host_configurable() {
        let config = AgentConfig {
            execution_mode: crate::ExecutionMode::Standard,
            ..AgentConfig::default()
        };
        let plugins = crate::PluginHost::new(vec![
            Arc::new(crate::PluginSpecFactory::new(
                "mock_base",
                Arc::new(|_ctx| {
                    Ok(crate::PluginSpec::new()
                        .with_tool_provider(Arc::new(MockBaseTool) as Arc<dyn ToolProvider>))
                }),
            )),
            Arc::new(AgentCallPluginFactory::new(
                config.clone(),
                AgentCallConfig {
                    low_tier_execution_mode: crate::ExecutionMode::Repl,
                },
                None,
            )),
        ])
        .build_standard_session("root", None)
        .expect("plugins");
        let agent_call = AgentCall::new(
            plugins,
            &config,
            AgentCallConfig {
                low_tier_execution_mode: crate::ExecutionMode::Repl,
            },
            None,
        );

        let low = agent_call.build_agent_config(&Tier::Low);
        let medium = agent_call.build_agent_config(&Tier::Medium);

        assert_eq!(low.execution_mode, crate::ExecutionMode::Repl);
        assert_eq!(medium.execution_mode, crate::ExecutionMode::Standard);
    }

    #[tokio::test]
    async fn agent_call_uses_session_manager_and_returns_child_session_metadata() {
        let agent_call = test_agent_call(crate::ExecutionMode::Standard);
        let host = Arc::new(MockSessionManager::default());
        let context = crate::ToolExecutionContext {
            session_id: "root".to_string(),
            host: host.clone(),
        };

        let handle = agent_call
            .execute_with_context(
                "agent_call",
                &json!({
                    "prompt": "Summarize the auth flow",
                    "intelligence": "low"
                }),
                &context,
            )
            .await;

        assert!(handle.success);
        assert_eq!(
            handle.result.get("id").and_then(|value| value.as_str()),
            Some("child-session")
        );
        assert_eq!(
            handle
                .result
                .get("config")
                .and_then(|value| value.get("sub_agent"))
                .and_then(|value| value.as_bool()),
            Some(true)
        );

        let result = agent_call
            .execute_streaming_with_context(
                "agent_result",
                &json!({ "id": "child-session" }),
                &context,
                None,
            )
            .await;

        assert!(result.success);
        assert_eq!(
            result.result.get("result").and_then(|value| value.as_str()),
            Some("delegate result")
        );
        assert_eq!(
            result
                .result
                .get("_sub_agent")
                .and_then(|value| value.get("session_id"))
                .and_then(|value| value.as_str()),
            Some("child-session")
        );
        assert_eq!(
            result
                .result
                .get("_sub_agent")
                .and_then(|value| value.get("status"))
                .and_then(|value| value.as_str()),
            Some("completed")
        );

        let repeated = agent_call
            .execute_streaming_with_context(
                "agent_result",
                &json!({ "id": "child-session" }),
                &context,
                None,
            )
            .await;
        assert!(repeated.success);
        assert_eq!(
            repeated
                .result
                .get("result")
                .and_then(|value| value.as_str()),
            Some("delegate result")
        );

        let created = host.created.lock().unwrap();
        assert_eq!(created.len(), 1);
        assert_eq!(
            created[0]
                .tool_surface
                .overrides
                .iter()
                .find(|override_| override_.tool_name == "apply_patch")
                .and_then(|override_| override_.enabled),
            Some(false)
        );
    }
}
