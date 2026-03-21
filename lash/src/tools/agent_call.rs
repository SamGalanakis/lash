mod policy;
mod runner;

use std::collections::HashMap;
use std::sync::{Arc, Mutex as StdMutex};

use crate::plugin::{
    PluginError, PluginFactory, PluginRegistrar, PluginSessionContext, SessionPlugin,
    SessionReadyContext,
};
use crate::provider::AgentModels;
use crate::{
    PluginSession, ProgressSender, SessionPolicy, ToolDefinition, ToolExecutionContext,
    ToolProvider, ToolResult,
};

use super::require_str;
#[cfg(test)]
use policy::pick_model_and_variant;
use policy::{Tier, agent_call_definitions, agent_call_prompt_contributions};
use runner::RunningAgent;

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

/// Single agent-call tool that spawns delegated child sessions at different intelligence tiers.
/// Returns a handle immediately; use agent_result/agent_kill to interact.
pub struct AgentCall {
    base_tools: Arc<dyn ToolProvider>,
    policy: SessionPolicy,
    execution_mode: crate::ExecutionMode,
    tool_config: AgentCallConfig,
    agent_models: Option<AgentModels>,
    agents: Arc<StdMutex<HashMap<String, RunningAgent>>>,
}

impl AgentCall {
    pub fn new(
        base_tools: Arc<dyn ToolProvider>,
        policy: &SessionPolicy,
        tool_config: AgentCallConfig,
        agent_models: Option<AgentModels>,
    ) -> Self {
        Self {
            base_tools,
            policy: policy.clone(),
            execution_mode: policy.execution_mode,
            tool_config,
            agent_models,
            agents: Arc::new(StdMutex::new(HashMap::new())),
        }
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

struct AgentCallPluginProvider {
    policy: SessionPolicy,
    execution_mode: crate::ExecutionMode,
    tool_config: AgentCallConfig,
    agent_models: Option<AgentModels>,
    delegate: StdMutex<Option<Arc<AgentCall>>>,
}

impl AgentCallPluginProvider {
    fn bind(&self, session: Arc<PluginSession>) {
        let base_tools = session.tools();
        let delegate = AgentCall::new(
            base_tools,
            &self.policy,
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
    policy: SessionPolicy,
    tool_config: AgentCallConfig,
    agent_models: Option<AgentModels>,
}

impl AgentCallPluginFactory {
    pub fn new(
        policy: SessionPolicy,
        tool_config: AgentCallConfig,
        agent_models: Option<AgentModels>,
    ) -> Self {
        Self {
            policy,
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
        let mut policy = self.policy.clone();
        policy.execution_mode = ctx.execution_mode;
        Ok(Arc::new(AgentCallPlugin {
            provider: Arc::new(AgentCallPluginProvider {
                policy,
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
    use crate::plugin::StaticPluginFactory;
    use crate::provider::Provider;
    use serde_json::json;
    use std::sync::Mutex;
    use tokio::sync::Notify;

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
                policy: request.policy.unwrap_or_else(|| crate::SessionPolicy {
                    provider: Provider::Codex {
                        access_token: "t".into(),
                        refresh_token: "r".into(),
                        expires_at: 0,
                        account_id: None,
                        options: crate::ProviderOptions::default(),
                    },
                    model: "mock-model".to_string(),
                    execution_mode: crate::ExecutionMode::Standard,
                    max_context_tokens: Some(128_000),
                    ..Default::default()
                }),
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
                policy: crate::SessionPolicy {
                    provider: Provider::Codex {
                        access_token: "t".into(),
                        refresh_token: "r".into(),
                        expires_at: 0,
                        account_id: None,
                        options: crate::ProviderOptions::default(),
                    },
                    model: "gpt-5.4-mini".to_string(),
                    model_variant: Some("low".to_string()),
                    recall_agent_model: None,
                    session_id: None,
                    execution_mode: crate::ExecutionMode::Standard,
                    context_strategy: crate::default_context_strategy(),
                    max_context_tokens: Some(128_000),
                    max_turns: None,
                    include_soul: false,
                    sub_agent: true,
                },
                events: rx,
            })
        }

        async fn await_turn(
            &self,
            turn_id: &str,
        ) -> Result<crate::AssembledTurn, crate::PluginError> {
            let cancelled = self
                .cancelled
                .lock()
                .unwrap()
                .iter()
                .any(|id| id == turn_id);
            Ok(crate::AssembledTurn {
                state: crate::AgentStateEnvelope {
                    agent_id: "child-session".to_string(),
                    policy: crate::SessionPolicy {
                        execution_mode: crate::ExecutionMode::Standard,
                        context_strategy: crate::default_context_strategy(),
                        ..Default::default()
                    },
                    iteration: 2,
                    ..Default::default()
                },
                status: if cancelled {
                    crate::TurnStatus::Interrupted
                } else {
                    crate::TurnStatus::Completed
                },
                assistant_output: crate::AssistantOutput {
                    safe_text: "delegate result".to_string(),
                    raw_text: "delegate result".to_string(),
                    state: crate::OutputState::Usable,
                },
                has_plugin_visible_output: false,
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
                    call_id: Some("tc1".to_string()),
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
        let config = SessionPolicy {
            model: "custom-parent-model".to_string(),
            provider: codex_provider(),
            ..Default::default()
        };

        let (m_low, r_low) = pick_model_and_variant(&config, &None, &Tier::Low);
        assert_eq!(m_low, "gpt-5.4-mini");
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
        let config = SessionPolicy {
            model: "custom-parent-model".to_string(),
            provider: codex_provider(),
            ..Default::default()
        };
        let models = Some(AgentModels {
            low: None,
            recall_agent: None,
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
        let config = SessionPolicy {
            model: "custom-parent-model".to_string(),
            provider: codex_provider(),
            execution_mode,
            ..Default::default()
        };
        let plugins = crate::PluginHost::new(vec![
            Arc::new(StaticPluginFactory::new(
                "mock_base",
                crate::PluginSpec::new()
                    .with_tool_provider(Arc::new(MockBaseTool) as Arc<dyn ToolProvider>),
            )),
            Arc::new(AgentCallPluginFactory::new(
                config.clone(),
                AgentCallConfig::default(),
                None,
            )),
        ])
        .build_session("root", execution_mode, None)
        .expect("plugins");
        let base_tools = plugins.tools();
        AgentCall::new(base_tools, &config, AgentCallConfig::default(), None)
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
        let low = agent_call.build_session_policy(&Tier::Low);
        let medium = agent_call.build_session_policy(&Tier::Medium);
        assert_eq!(
            low.execution_mode,
            AgentCallConfig::default().low_tier_execution_mode
        );
        assert_eq!(medium.execution_mode, agent_call.policy.execution_mode);
    }

    #[test]
    fn low_tier_execution_mode_is_host_configurable() {
        let config = SessionPolicy {
            execution_mode: crate::ExecutionMode::Standard,
            ..SessionPolicy::default()
        };
        let plugins = crate::PluginHost::new(vec![
            Arc::new(StaticPluginFactory::new(
                "mock_base",
                crate::PluginSpec::new()
                    .with_tool_provider(Arc::new(MockBaseTool) as Arc<dyn ToolProvider>),
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
        let base_tools = plugins.tools();
        let agent_call = AgentCall::new(
            base_tools,
            &config,
            AgentCallConfig {
                low_tier_execution_mode: crate::ExecutionMode::Repl,
            },
            None,
        );

        let low = agent_call.build_session_policy(&Tier::Low);
        let medium = agent_call.build_session_policy(&Tier::Medium);

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
                .get("execution_mode")
                .and_then(|value| value.as_str()),
            Some("standard")
        );
        assert!(handle.result.get("config").is_none());

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
            result.result.get("status").and_then(|value| value.as_str()),
            Some("completed")
        );
        let result_text = result.result.to_string();
        assert!(!result_text.contains("access_token"));
        assert!(!result_text.contains("refresh_token"));

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
        assert_eq!(
            host.closed.lock().unwrap().as_slice(),
            ["child-session".to_string()]
        );

        let created = host.created.lock().unwrap();
        assert_eq!(created.len(), 1);
        let tool_names = created[0].context_surface.tool_providers[0]
            .definitions()
            .into_iter()
            .map(|tool| tool.name)
            .collect::<Vec<_>>();
        assert!(!tool_names.iter().any(|name| name == "apply_patch"));
    }

    #[tokio::test]
    async fn agent_result_streams_delegate_start_before_tool_output() {
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

        let (progress_tx, mut progress_rx) = tokio::sync::mpsc::unbounded_channel();
        let result = agent_call
            .execute_streaming_with_context(
                "agent_result",
                &json!({ "id": "child-session" }),
                &context,
                Some(&progress_tx),
            )
            .await;
        assert!(result.success);

        let first = progress_rx.recv().await.expect("delegate_start message");
        assert_eq!(first.kind, "delegate_start");
        let first_json: serde_json::Value =
            serde_json::from_str(&first.text).expect("delegate_start payload");
        assert_eq!(
            first_json.get("task").and_then(|value| value.as_str()),
            Some("Summarize the auth flow")
        );
        assert_eq!(
            first_json.get("model").and_then(|value| value.as_str()),
            handle.result.get("model").and_then(|value| value.as_str())
        );

        let second = progress_rx.recv().await.expect("tool_output message");
        assert_eq!(second.kind, "tool_output");
        assert!(second.text.contains("delegate"));
    }

    #[tokio::test]
    async fn killed_agent_remains_queryable() {
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

        let killed = agent_call
            .execute_with_context("agent_kill", &json!({ "id": "child-session" }), &context)
            .await;
        assert!(killed.success);

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
            result.result.get("status").and_then(|value| value.as_str()),
            Some("interrupted")
        );
        assert_eq!(
            host.cancelled.lock().unwrap().as_slice(),
            ["turn-1".to_string()]
        );
        assert_eq!(
            host.closed.lock().unwrap().as_slice(),
            ["child-session".to_string()]
        );
    }

    #[tokio::test]
    async fn timed_out_agent_remains_queryable() {
        let agent_call = test_agent_call(crate::ExecutionMode::Standard);
        let host = Arc::new(MockSessionManager::default());
        let done_notify = Arc::new(Notify::new());
        let result = Arc::new(StdMutex::new(None));

        agent_call.agents.lock().unwrap().insert(
            "child-session".to_string(),
            RunningAgent {
                session_id: "child-session".to_string(),
                turn_id: "turn-1".to_string(),
                host: host.clone(),
                task: "Summarize the auth flow".to_string(),
                model: "mock-model".to_string(),
                model_variant: Some("low".to_string()),
                buffer: Arc::new(StdMutex::new(String::new())),
                result: result.clone(),
                done_notify: done_notify.clone(),
                session_closed: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            },
        );

        let result_clone = result.clone();
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            *result_clone.lock().unwrap() = Some(json!({
                "result": "delegate result",
                "status": "interrupted"
            }));
            done_notify.notify_waiters();
        });

        let timed_out = agent_call
            .agent_result("child-session", Some(0.0), None)
            .await;
        assert!(!timed_out.success);
        assert_eq!(timed_out.result, json!("Agent timed out"));

        let repeated = agent_call.agent_result("child-session", None, None).await;
        assert!(repeated.success);
        assert_eq!(
            repeated
                .result
                .get("result")
                .and_then(|value| value.as_str()),
            Some("delegate result")
        );
        assert_eq!(
            host.cancelled.lock().unwrap().as_slice(),
            ["turn-1".to_string()]
        );
        assert_eq!(
            host.closed.lock().unwrap().as_slice(),
            ["child-session".to_string()]
        );
    }
}
