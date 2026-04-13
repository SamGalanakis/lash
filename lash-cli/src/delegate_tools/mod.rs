mod policy;
mod runner;

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex as StdMutex};

use lash::plugin::{
    PluginError, PluginFactory, PluginRegistrar, PluginSessionContext, SessionPlugin,
    SessionReadyContext,
};
use lash::provider::AgentModels;
use lash::{
    PluginSession, ProgressSender, SessionPolicy, ToolDefinition, ToolExecutionContext,
    ToolProvider, ToolResult,
};

#[cfg(test)]
use policy::Tier;
use policy::{delegate_prompt_contributions, delegate_tool_definitions};
use runner::RunningDelegate;

#[derive(Clone)]
struct FilteredToolProvider {
    inner: Arc<dyn ToolProvider>,
    allowed: HashSet<String>,
}

impl FilteredToolProvider {
    fn new(
        inner: Arc<dyn ToolProvider>,
        allowed: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        Self {
            inner,
            allowed: allowed.into_iter().map(Into::into).collect(),
        }
    }

    fn allows(&self, name: &str) -> bool {
        self.allowed.contains(name)
    }
}

#[async_trait::async_trait]
impl ToolProvider for FilteredToolProvider {
    fn definitions(&self) -> Vec<ToolDefinition> {
        self.inner
            .definitions()
            .into_iter()
            .filter(|definition| self.allows(&definition.name))
            .collect()
    }

    async fn execute(&self, name: &str, args: &serde_json::Value) -> ToolResult {
        if !self.allows(name) {
            return ToolResult::err_fmt(format_args!("Unknown tool: {name}"));
        }
        self.inner.execute(name, args).await
    }

    async fn execute_with_context(
        &self,
        name: &str,
        args: &serde_json::Value,
        context: &ToolExecutionContext,
    ) -> ToolResult {
        if !self.allows(name) {
            return ToolResult::err_fmt(format_args!("Unknown tool: {name}"));
        }
        self.inner.execute_with_context(name, args, context).await
    }

    async fn execute_streaming(
        &self,
        name: &str,
        args: &serde_json::Value,
        progress: Option<&ProgressSender>,
    ) -> ToolResult {
        if !self.allows(name) {
            return ToolResult::err_fmt(format_args!("Unknown tool: {name}"));
        }
        self.inner.execute_streaming(name, args, progress).await
    }

    async fn execute_streaming_with_context(
        &self,
        name: &str,
        args: &serde_json::Value,
        context: &ToolExecutionContext,
        progress: Option<&ProgressSender>,
    ) -> ToolResult {
        if !self.allows(name) {
            return ToolResult::err_fmt(format_args!("Unknown tool: {name}"));
        }
        self.inner
            .execute_streaming_with_context(name, args, context, progress)
            .await
    }
}

fn require_str<'a>(args: &'a serde_json::Value, key: &str) -> Result<&'a str, ToolResult> {
    args.get(key)
        .and_then(|value| value.as_str())
        .filter(|value| !value.is_empty())
        .ok_or_else(|| ToolResult::err_fmt(format_args!("Missing required parameter: {key}")))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DelegateToolConfig {
    pub low_tier_execution_mode: lash::ExecutionMode,
}

impl Default for DelegateToolConfig {
    fn default() -> Self {
        Self {
            low_tier_execution_mode: lash::ExecutionMode::Standard,
        }
    }
}

struct DelegateTools {
    base_tools: Arc<dyn ToolProvider>,
    policy: SessionPolicy,
    execution_mode: lash::ExecutionMode,
    tool_config: DelegateToolConfig,
    agent_models: Option<AgentModels>,
    delegates: Arc<StdMutex<HashMap<String, RunningDelegate>>>,
}

impl DelegateTools {
    fn new(
        base_tools: Arc<dyn ToolProvider>,
        policy: &SessionPolicy,
        tool_config: DelegateToolConfig,
        agent_models: Option<AgentModels>,
    ) -> Self {
        Self {
            base_tools,
            policy: policy.clone(),
            execution_mode: policy.execution_mode,
            tool_config,
            agent_models,
            delegates: Arc::new(StdMutex::new(HashMap::new())),
        }
    }
}

#[async_trait::async_trait]
impl ToolProvider for DelegateTools {
    fn definitions(&self) -> Vec<ToolDefinition> {
        delegate_tool_definitions(
            self.execution_mode,
            self.tool_config.low_tier_execution_mode,
        )
    }

    async fn execute(&self, name: &str, args: &serde_json::Value) -> ToolResult {
        match name {
            "agent_call" => ToolResult::err_fmt("agent_call requires session context"),
            "predict" => ToolResult::err_fmt("predict requires session context"),
            "agent_result" => {
                let id = match require_str(args, "id") {
                    Ok(id) => id,
                    Err(err) => return err,
                };
                let timeout = args.get("timeout").and_then(|value| value.as_f64());
                self.agent_result(id, timeout, None).await
            }
            "agent_kill" => {
                let id = match require_str(args, "id") {
                    Ok(id) => id,
                    Err(err) => return err,
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
                    Ok(id) => id,
                    Err(err) => return err,
                };
                let timeout = args.get("timeout").and_then(|value| value.as_f64());
                self.agent_result(id, timeout, progress).await
            }
            "agent_call" => ToolResult::err_fmt("agent_call requires session context"),
            "predict" => ToolResult::err_fmt("predict requires session context"),
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
            "predict" => self.spawn_predict(args, context).await,
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
            "predict" => self.spawn_predict(args, context).await,
            "agent_result" => {
                let id = match require_str(args, "id") {
                    Ok(id) => id,
                    Err(err) => return err,
                };
                let timeout = args.get("timeout").and_then(|value| value.as_f64());
                self.agent_result(id, timeout, progress).await
            }
            _ => self.execute_with_context(name, args, context).await,
        }
    }
}

struct DelegateToolsPluginProvider {
    policy: SessionPolicy,
    execution_mode: lash::ExecutionMode,
    tool_config: DelegateToolConfig,
    agent_models: Option<AgentModels>,
    delegate_tools: StdMutex<Option<Arc<DelegateTools>>>,
}

impl DelegateToolsPluginProvider {
    fn bind(&self, session: Arc<PluginSession>) {
        let delegate_tools = DelegateTools::new(
            session.tools(),
            &self.policy,
            self.tool_config,
            self.agent_models.clone(),
        );
        *self
            .delegate_tools
            .lock()
            .expect("delegate tools lock poisoned") = Some(Arc::new(delegate_tools));
    }

    fn bound(&self) -> Option<Arc<DelegateTools>> {
        self.delegate_tools
            .lock()
            .expect("delegate tools lock poisoned")
            .clone()
    }
}

#[async_trait::async_trait]
impl ToolProvider for DelegateToolsPluginProvider {
    fn definitions(&self) -> Vec<ToolDefinition> {
        delegate_tool_definitions(
            self.execution_mode,
            self.tool_config.low_tier_execution_mode,
        )
    }

    async fn execute(&self, name: &str, args: &serde_json::Value) -> ToolResult {
        let Some(delegate_tools) = self.bound() else {
            return ToolResult::err_fmt("delegate tools are not ready");
        };
        delegate_tools.execute(name, args).await
    }

    async fn execute_with_context(
        &self,
        name: &str,
        args: &serde_json::Value,
        context: &ToolExecutionContext,
    ) -> ToolResult {
        let Some(delegate_tools) = self.bound() else {
            return ToolResult::err_fmt("delegate tools are not ready");
        };
        delegate_tools
            .execute_with_context(name, args, context)
            .await
    }

    async fn execute_streaming(
        &self,
        name: &str,
        args: &serde_json::Value,
        progress: Option<&ProgressSender>,
    ) -> ToolResult {
        let Some(delegate_tools) = self.bound() else {
            return ToolResult::err_fmt("delegate tools are not ready");
        };
        delegate_tools.execute_streaming(name, args, progress).await
    }

    async fn execute_streaming_with_context(
        &self,
        name: &str,
        args: &serde_json::Value,
        context: &ToolExecutionContext,
        progress: Option<&ProgressSender>,
    ) -> ToolResult {
        let Some(delegate_tools) = self.bound() else {
            return ToolResult::err_fmt("delegate tools are not ready");
        };
        delegate_tools
            .execute_streaming_with_context(name, args, context, progress)
            .await
    }
}

pub(crate) struct DelegateToolsPluginFactory {
    policy: SessionPolicy,
    tool_config: DelegateToolConfig,
    agent_models: Option<AgentModels>,
}

impl DelegateToolsPluginFactory {
    pub(crate) fn new(
        policy: SessionPolicy,
        tool_config: DelegateToolConfig,
        agent_models: Option<AgentModels>,
    ) -> Self {
        Self {
            policy,
            tool_config,
            agent_models,
        }
    }
}

impl PluginFactory for DelegateToolsPluginFactory {
    fn id(&self) -> &'static str {
        "delegate_tools"
    }

    fn build(&self, ctx: &PluginSessionContext) -> Result<Arc<dyn SessionPlugin>, PluginError> {
        let mut policy = self.policy.clone();
        policy.execution_mode = ctx.execution_mode;
        Ok(Arc::new(DelegateToolsPlugin {
            provider: Arc::new(DelegateToolsPluginProvider {
                policy,
                execution_mode: ctx.execution_mode,
                tool_config: self.tool_config,
                agent_models: self.agent_models.clone(),
                delegate_tools: StdMutex::new(None),
            }),
        }))
    }
}

struct DelegateToolsPlugin {
    provider: Arc<DelegateToolsPluginProvider>,
}

impl SessionPlugin for DelegateToolsPlugin {
    fn id(&self) -> &'static str {
        "delegate_tools"
    }

    fn register(&self, reg: &mut PluginRegistrar) -> Result<(), PluginError> {
        reg.tools()
            .provider(Arc::clone(&self.provider) as Arc<dyn ToolProvider>)?;
        reg.prompt().contribute(Arc::new(|_ctx| {
            Box::pin(async move { Ok(delegate_prompt_contributions()) })
        }));
        Ok(())
    }

    fn session_ready(&self, ctx: SessionReadyContext) -> Result<(), PluginError> {
        let session = ctx
            .host
            .session(&ctx.session_id)
            .map_err(|err| PluginError::Session(err.to_string()))?;
        self.provider.bind(session);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use lash::plugin::StaticPluginFactory;
    use lash::provider::Provider;
    use lash::{AssembledTurn, ExecutionSummary, OutputState, TurnStatus};
    use serde_json::json;
    use std::sync::Mutex;
    use tokio::sync::Notify;

    #[derive(Default)]
    struct MockSessionManager {
        created: Mutex<Vec<lash::SessionCreateRequest>>,
        cancelled: Mutex<Vec<String>>,
        closed: Mutex<Vec<String>>,
    }

    #[async_trait::async_trait]
    impl lash::SessionManager for MockSessionManager {
        async fn snapshot_current(&self) -> Result<lash::SessionSnapshot, lash::PluginError> {
            Ok(lash::SessionStateEnvelope::default())
        }

        async fn snapshot_session(
            &self,
            _session_id: &str,
        ) -> Result<lash::SessionSnapshot, lash::PluginError> {
            Ok(lash::SessionStateEnvelope::default())
        }

        async fn tool_catalog(
            &self,
            _session_id: &str,
        ) -> Result<Vec<serde_json::Value>, lash::PluginError> {
            Ok(Vec::new())
        }

        async fn create_session(
            &self,
            request: lash::SessionCreateRequest,
        ) -> Result<lash::SessionHandle, lash::PluginError> {
            self.created.lock().unwrap().push(request.clone());
            let mut policy = request.policy.unwrap_or_else(|| lash::SessionPolicy {
                provider: Provider::Codex {
                    access_token: "t".into(),
                    refresh_token: "r".into(),
                    expires_at: 0,
                    account_id: None,
                    options: lash::ProviderOptions::default(),
                },
                model: "mock-model".to_string(),
                execution_mode: lash::ExecutionMode::Standard,
                max_context_tokens: Some(128_000),
                ..Default::default()
            });
            if request.parent_session_id.is_some() {
                policy.session_id = Some("child-session".to_string());
            }
            Ok(lash::SessionHandle {
                session_id: "child-session".to_string(),
                parent_session_id: request.parent_session_id,
                policy,
            })
        }

        async fn close_session(&self, session_id: &str) -> Result<(), lash::PluginError> {
            self.closed.lock().unwrap().push(session_id.to_string());
            Ok(())
        }

        async fn start_turn_stream(
            &self,
            session_id: &str,
            _input: lash::TurnInput,
        ) -> Result<lash::SessionTurnHandle, lash::PluginError> {
            let (tx, rx) = tokio::sync::mpsc::channel(4);
            tokio::spawn(async move {
                let _ = tx
                    .send(lash::SessionEvent::TextDelta {
                        content: "delegate".to_string(),
                    })
                    .await;
                let _ = tx
                    .send(lash::SessionEvent::Message {
                        text: "delegate result".to_string(),
                        kind: "final".to_string(),
                    })
                    .await;
            });
            Ok(lash::SessionTurnHandle {
                turn_id: "turn-1".to_string(),
                session_id: session_id.to_string(),
                policy: lash::SessionPolicy {
                    provider: Provider::Codex {
                        access_token: "t".into(),
                        refresh_token: "r".into(),
                        expires_at: 0,
                        account_id: None,
                        options: lash::ProviderOptions::default(),
                    },
                    model: "gpt-5.4-mini".to_string(),
                    model_variant: Some("low".to_string()),
                    session_id: Some(session_id.to_string()),
                    execution_mode: lash::ExecutionMode::Standard,
                    max_context_tokens: Some(128_000),
                    max_turns: None,
                    context_approach: lash::ContextApproach::default(),
                },
                events: rx,
            })
        }

        async fn await_turn(&self, turn_id: &str) -> Result<AssembledTurn, lash::PluginError> {
            let cancelled = self
                .cancelled
                .lock()
                .unwrap()
                .iter()
                .any(|id| id == turn_id);
            Ok(lash::AssembledTurn {
                state: lash::SessionStateEnvelope {
                    session_id: "child-session".to_string(),
                    policy: lash::SessionPolicy {
                        session_id: Some("child-session".to_string()),
                        execution_mode: lash::ExecutionMode::Standard,
                        ..Default::default()
                    },
                    iteration: 2,
                    ..Default::default()
                },
                status: if cancelled {
                    TurnStatus::Interrupted
                } else {
                    TurnStatus::Completed
                },
                assistant_output: lash::AssistantOutput {
                    safe_text: "delegate result".to_string(),
                    raw_text: "delegate result".to_string(),
                    state: OutputState::Usable,
                },
                has_plugin_visible_output: false,
                done_reason: lash::DoneReason::ModelStop,
                execution: ExecutionSummary {
                    mode: lash::ExecutionMode::Standard,
                    had_tool_calls: true,
                    had_code_execution: false,
                },
                token_usage: lash::TokenUsage {
                    input_tokens: 11,
                    output_tokens: 7,
                    cached_input_tokens: 3,
                    reasoning_tokens: 2,
                },
                tool_calls: vec![lash::ToolCallRecord {
                    call_id: Some("call-1".to_string()),
                    tool: "read_file".to_string(),
                    args: json!({"path":"Cargo.toml"}),
                    result: json!("contents"),
                    success: true,
                    duration_ms: 5,
                }],
                errors: Vec::new(),
                typed_finish: None,
            })
        }

        async fn cancel_turn(&self, turn_id: &str) -> Result<(), lash::PluginError> {
            self.cancelled.lock().unwrap().push(turn_id.to_string());
            Ok(())
        }
    }

    #[test]
    fn docs_are_mode_specific() {
        let rlm = test_delegate_tools(lash::ExecutionMode::Rlm)
            .definitions()
            .into_iter()
            .find(|definition| definition.name == "agent_call")
            .expect("agent_call definition");
        let standard = test_delegate_tools(lash::ExecutionMode::Standard)
            .definitions()
            .into_iter()
            .find(|definition| definition.name == "agent_call")
            .expect("agent_call definition");

        assert!(rlm.description.contains("return a handle"));
        assert!(
            rlm.description
                .contains("call agent_result { id: handle.value.id }")
        );
        assert!(
            !standard
                .description
                .contains("call agent_result { id: handle.value.id }")
        );
    }

    #[test]
    fn lifecycle_tools_are_prompt_injected() {
        let definitions = test_delegate_tools(lash::ExecutionMode::Standard).definitions();
        for name in ["agent_call", "agent_result", "agent_kill"] {
            assert!(
                definitions
                    .iter()
                    .any(|definition| definition.name == name && definition.injected)
            );
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
                    input_schema_override: None,
                    output_schema_override: None,
                },
                ToolDefinition {
                    name: "read_file".into(),
                    description: "mock read".into(),
                    params: vec![],
                    returns: "str".into(),
                    examples: vec![],
                    enabled: true,
                    injected: true,
                    input_schema_override: None,
                    output_schema_override: None,
                },
                ToolDefinition {
                    name: "apply_patch".into(),
                    description: "mock patch".into(),
                    params: vec![],
                    returns: "None".into(),
                    examples: vec![],
                    enabled: true,
                    injected: true,
                    input_schema_override: None,
                    output_schema_override: None,
                },
            ]
        }

        async fn execute(&self, _name: &str, _args: &serde_json::Value) -> ToolResult {
            ToolResult::ok(json!(null))
        }
    }

    fn test_delegate_tools(execution_mode: lash::ExecutionMode) -> DelegateTools {
        let policy = lash::SessionPolicy {
            model: "custom-parent-model".to_string(),
            provider: codex_provider(),
            execution_mode,
            ..Default::default()
        };
        let plugins = lash::PluginHost::new(vec![
            Arc::new(StaticPluginFactory::new(
                "mock_base",
                lash::PluginSpec::new()
                    .with_tool_provider(Arc::new(MockBaseTool) as Arc<dyn ToolProvider>),
            )),
            Arc::new(DelegateToolsPluginFactory::new(
                policy.clone(),
                DelegateToolConfig::default(),
                None,
            )),
        ])
        .build_session(
            "root",
            execution_mode,
            policy.context_approach.clone(),
            None,
        )
        .expect("plugins");
        DelegateTools::new(
            plugins.tools(),
            &policy,
            DelegateToolConfig::default(),
            None,
        )
    }

    fn codex_provider() -> Provider {
        Provider::Codex {
            access_token: "token".into(),
            refresh_token: "refresh".into(),
            expires_at: 0,
            account_id: None,
            options: lash::ProviderOptions::default(),
        }
    }

    #[test]
    fn low_tier_child_session_tools_do_not_include_agent_call() {
        let delegate_tools = test_delegate_tools(lash::ExecutionMode::Standard);
        let tool_names = delegate_tools
            .visible_tool_names_for_tier(&Tier::Low)
            .expect("tier tools");
        assert!(!tool_names.iter().any(|name| name == "agent_call"));
        assert!(tool_names.iter().any(|name| name == "mock_base"));
        assert!(tool_names.iter().any(|name| name == "read_file"));
        assert!(!tool_names.iter().any(|name| name == "apply_patch"));
    }

    #[test]
    fn medium_and_high_tier_child_session_tools_include_agent_call() {
        let delegate_tools = test_delegate_tools(lash::ExecutionMode::Standard);

        let medium_names = delegate_tools
            .visible_tool_names_for_tier(&Tier::Medium)
            .expect("medium tools");
        assert!(medium_names.iter().any(|name| name == "agent_call"));
        assert!(medium_names.iter().any(|name| name == "mock_base"));
        assert!(medium_names.iter().any(|name| name == "apply_patch"));

        let high_names = delegate_tools
            .visible_tool_names_for_tier(&Tier::High)
            .expect("high tools");
        assert!(high_names.iter().any(|name| name == "agent_call"));
        assert!(high_names.iter().any(|name| name == "mock_base"));
        assert!(high_names.iter().any(|name| name == "apply_patch"));
    }

    #[test]
    fn low_tier_child_session_policy_uses_host_configured_execution_mode() {
        let delegate_tools = test_delegate_tools(lash::ExecutionMode::Standard);
        let low = delegate_tools.build_session_policy(&Tier::Low);
        let medium = delegate_tools.build_session_policy(&Tier::Medium);
        assert_eq!(
            low.execution_mode,
            DelegateToolConfig::default().low_tier_execution_mode
        );
        assert_eq!(medium.execution_mode, delegate_tools.policy.execution_mode);
    }

    #[test]
    fn low_tier_execution_mode_is_host_configurable() {
        let policy = lash::SessionPolicy {
            execution_mode: lash::ExecutionMode::Standard,
            ..Default::default()
        };
        let plugins = lash::PluginHost::new(vec![
            Arc::new(StaticPluginFactory::new(
                "mock_base",
                lash::PluginSpec::new()
                    .with_tool_provider(Arc::new(MockBaseTool) as Arc<dyn ToolProvider>),
            )),
            Arc::new(DelegateToolsPluginFactory::new(
                policy.clone(),
                DelegateToolConfig {
                    low_tier_execution_mode: lash::ExecutionMode::Rlm,
                },
                None,
            )),
        ])
        .build_standard_session("root", None)
        .expect("plugins");

        let delegate_tools = DelegateTools::new(
            plugins.tools(),
            &policy,
            DelegateToolConfig {
                low_tier_execution_mode: lash::ExecutionMode::Rlm,
            },
            None,
        );

        let low = delegate_tools.build_session_policy(&Tier::Low);
        let medium = delegate_tools.build_session_policy(&Tier::Medium);
        assert_eq!(low.execution_mode, lash::ExecutionMode::Rlm);
        assert_eq!(medium.execution_mode, lash::ExecutionMode::Standard);
    }

    #[tokio::test]
    async fn agent_call_uses_session_manager_and_returns_child_session_metadata() {
        let delegate_tools = test_delegate_tools(lash::ExecutionMode::Standard);
        let host = Arc::new(MockSessionManager::default());
        let context = lash::ToolExecutionContext {
            session_id: "root".to_string(),
            host: host.clone(),
        };

        let handle = delegate_tools
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
                .get("parent_session_id")
                .and_then(|value| value.as_str()),
            Some("root")
        );
        assert_eq!(
            handle
                .result
                .get("execution_mode")
                .and_then(|value| value.as_str()),
            Some("standard")
        );

        let result = delegate_tools
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

        let repeated = delegate_tools
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
        assert!(host.closed.lock().unwrap().is_empty());

        let created = host.created.lock().unwrap();
        assert_eq!(created.len(), 1);
        assert_eq!(created[0].parent_session_id.as_deref(), Some("root"));
        let tool_names = created[0].context_surface.tool_providers[0]
            .definitions()
            .into_iter()
            .map(|tool| tool.name)
            .collect::<Vec<_>>();
        assert!(!tool_names.iter().any(|name| name == "apply_patch"));
    }

    #[tokio::test]
    async fn agent_result_streams_delegate_start_before_tool_output() {
        let delegate_tools = test_delegate_tools(lash::ExecutionMode::Standard);
        let host = Arc::new(MockSessionManager::default());
        let context = lash::ToolExecutionContext {
            session_id: "root".to_string(),
            host: host.clone(),
        };

        let handle = delegate_tools
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
        let result = delegate_tools
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
            first_json
                .get("parent_session_id")
                .and_then(|value| value.as_str()),
            Some("root")
        );

        let second = progress_rx.recv().await.expect("tool_output message");
        assert_eq!(second.kind, "tool_output");
        assert!(second.text.contains("delegate"));
    }

    #[tokio::test]
    async fn killed_agent_remains_queryable() {
        let delegate_tools = test_delegate_tools(lash::ExecutionMode::Standard);
        let host = Arc::new(MockSessionManager::default());
        let context = lash::ToolExecutionContext {
            session_id: "root".to_string(),
            host: host.clone(),
        };

        let handle = delegate_tools
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

        let killed = delegate_tools
            .execute_with_context("agent_kill", &json!({ "id": "child-session" }), &context)
            .await;
        assert!(killed.success);

        let result = delegate_tools
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
        assert!(host.closed.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn timed_out_agent_remains_queryable() {
        let delegate_tools = test_delegate_tools(lash::ExecutionMode::Standard);
        let host = Arc::new(MockSessionManager::default());
        let done_notify = Arc::new(Notify::new());
        let result = Arc::new(StdMutex::new(None));

        delegate_tools.delegates.lock().unwrap().insert(
            "child-session".to_string(),
            RunningDelegate {
                session_id: "child-session".to_string(),
                parent_session_id: Some("root".to_string()),
                turn_id: "turn-1".to_string(),
                host: host.clone(),
                task: "Summarize the auth flow".to_string(),
                model: "mock-model".to_string(),
                model_variant: Some("low".to_string()),
                buffer: Arc::new(StdMutex::new(String::new())),
                result: result.clone(),
                done_notify: done_notify.clone(),
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

        let timed_out = delegate_tools
            .agent_result("child-session", Some(0.0), None)
            .await;
        assert!(!timed_out.success);
        assert_eq!(timed_out.result, json!("Agent timed out"));

        let repeated = delegate_tools
            .agent_result("child-session", None, None)
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
            host.cancelled.lock().unwrap().as_slice(),
            ["turn-1".to_string()]
        );
        assert!(host.closed.lock().unwrap().is_empty());
    }
}
