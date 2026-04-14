mod policy;
mod runner;

use std::collections::HashSet;
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DelegateToolConfig {
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

    async fn execute(&self, name: &str, _args: &serde_json::Value) -> ToolResult {
        match name {
            "delegate" => ToolResult::err_fmt("delegate requires session context"),
            _ => ToolResult::err_fmt(format_args!("Unknown tool: {name}")),
        }
    }

    async fn execute_streaming(
        &self,
        name: &str,
        args: &serde_json::Value,
        _progress: Option<&ProgressSender>,
    ) -> ToolResult {
        match name {
            "delegate" => ToolResult::err_fmt("delegate requires session context"),
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
            "delegate" => self.delegate(args, context, None).await,
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
            "delegate" => self.delegate(args, context, progress).await,
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

pub struct DelegateToolsPluginFactory {
    policy: SessionPolicy,
    tool_config: DelegateToolConfig,
    agent_models: Option<AgentModels>,
}

impl DelegateToolsPluginFactory {
    pub fn new(
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
    use std::time::Duration;
    use tokio_util::sync::CancellationToken;

    struct MockSessionManager {
        created: Mutex<Vec<lash::SessionCreateRequest>>,
        cancelled: Mutex<Vec<String>>,
        closed: Mutex<Vec<String>>,
        event_delay: Duration,
    }

    impl Default for MockSessionManager {
        fn default() -> Self {
            Self {
                created: Mutex::new(Vec::new()),
                cancelled: Mutex::new(Vec::new()),
                closed: Mutex::new(Vec::new()),
                event_delay: Duration::from_millis(0),
            }
        }
    }

    impl MockSessionManager {
        fn with_event_delay(mut self, delay: Duration) -> Self {
            self.event_delay = delay;
            self
        }
    }

    #[async_trait::async_trait]
    impl lash::SessionManager for MockSessionManager {
        async fn snapshot_current(&self) -> Result<lash::SessionSnapshot, lash::PluginError> {
            Ok(lash::PersistedSessionState::default())
        }

        async fn snapshot_session(
            &self,
            _session_id: &str,
        ) -> Result<lash::SessionSnapshot, lash::PluginError> {
            Ok(lash::PersistedSessionState::default())
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
            let delay = self.event_delay;
            tokio::spawn(async move {
                if !delay.is_zero() {
                    tokio::time::sleep(delay).await;
                }
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
                typed_finish: Some(json!({"answer": "delegate result"})),
            })
        }

        async fn cancel_turn(&self, turn_id: &str) -> Result<(), lash::PluginError> {
            self.cancelled.lock().unwrap().push(turn_id.to_string());
            Ok(())
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
    fn lifecycle_tools_are_prompt_injected() {
        let definitions = test_delegate_tools(lash::ExecutionMode::Standard).definitions();
        assert_eq!(definitions.len(), 1);
        assert_eq!(definitions[0].name, "delegate");
        assert!(definitions[0].injected);
    }

    #[test]
    fn low_tier_child_session_tools_do_not_include_delegate() {
        let delegate_tools = test_delegate_tools(lash::ExecutionMode::Standard);
        let tool_names = delegate_tools
            .visible_tool_names_for_tier(&Tier::Low)
            .expect("tier tools");
        assert!(!tool_names.iter().any(|name| name == "delegate"));
        assert!(tool_names.iter().any(|name| name == "mock_base"));
        assert!(tool_names.iter().any(|name| name == "read_file"));
        assert!(!tool_names.iter().any(|name| name == "apply_patch"));
    }

    #[test]
    fn medium_and_high_tier_child_session_tools_include_delegate() {
        let delegate_tools = test_delegate_tools(lash::ExecutionMode::Standard);

        let medium_names = delegate_tools
            .visible_tool_names_for_tier(&Tier::Medium)
            .expect("medium tools");
        assert!(medium_names.iter().any(|name| name == "delegate"));
        assert!(medium_names.iter().any(|name| name == "mock_base"));
        assert!(medium_names.iter().any(|name| name == "apply_patch"));

        let high_names = delegate_tools
            .visible_tool_names_for_tier(&Tier::High)
            .expect("high tools");
        assert!(high_names.iter().any(|name| name == "delegate"));
        assert!(high_names.iter().any(|name| name == "mock_base"));
        assert!(high_names.iter().any(|name| name == "apply_patch"));
    }

    #[tokio::test]
    async fn delegate_uses_session_manager_and_returns_child_result() {
        let delegate_tools = test_delegate_tools(lash::ExecutionMode::Standard);
        let host = Arc::new(MockSessionManager::default());
        let context = lash::ToolExecutionContext {
            session_id: "root".to_string(),
            host: host.clone(),
            cancellation_token: None,
            async_task_id: None,
        };

        let result = delegate_tools
            .execute_with_context(
                "delegate",
                &json!({
                    "task": "Summarize the auth flow",
                    "intelligence": "low"
                }),
                &context,
            )
            .await;

        assert!(result.success);
        assert_eq!(
            result.result.get("status").and_then(|value| value.as_str()),
            Some("completed")
        );
        assert_eq!(
            result.result.get("result"),
            Some(&json!({"answer": "delegate result"}))
        );
        assert_eq!(
            result
                .result
                .get("session")
                .and_then(|value| value.get("id"))
                .and_then(|value| value.as_str()),
            Some("child-session")
        );
        assert_eq!(
            result
                .result
                .get("session")
                .and_then(|value| value.get("parent_session_id"))
                .and_then(|value| value.as_str()),
            Some("root")
        );
        assert!(host.closed.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn delegate_streaming_emits_delegate_progress_messages() {
        let delegate_tools = test_delegate_tools(lash::ExecutionMode::Standard);
        let host = Arc::new(MockSessionManager::default());
        let context = lash::ToolExecutionContext {
            session_id: "root".to_string(),
            host,
            cancellation_token: None,
            async_task_id: None,
        };

        let (progress, mut progress_rx) = tokio::sync::mpsc::unbounded_channel();
        let result = delegate_tools
            .execute_streaming_with_context(
                "delegate",
                &json!({
                    "task": "Summarize the auth flow",
                    "intelligence": "low"
                }),
                &context,
                Some(&progress),
            )
            .await;

        assert!(result.success);
        let first = progress_rx.recv().await.expect("delegate_start message");
        assert_eq!(first.kind, "delegate_start");
        let second = progress_rx.recv().await.expect("tool output");
        assert_eq!(second.kind, "tool_output");
        assert!(second.text.contains("delegate"));
    }

    #[tokio::test]
    async fn delegate_respects_cancellation_token() {
        let delegate_tools = test_delegate_tools(lash::ExecutionMode::Standard);
        let host = Arc::new(MockSessionManager::default().with_event_delay(Duration::from_secs(5)));
        let cancellation = CancellationToken::new();
        cancellation.cancel();
        let context = lash::ToolExecutionContext {
            session_id: "root".to_string(),
            host: Arc::clone(&host) as Arc<dyn lash::SessionManager>,
            cancellation_token: Some(cancellation),
            async_task_id: Some("task-1".to_string()),
        };

        let result = delegate_tools
            .execute_with_context(
                "delegate",
                &json!({
                    "task": "Summarize the auth flow",
                    "intelligence": "low"
                }),
                &context,
            )
            .await;

        assert!(result.success);
        assert_eq!(
            result.result.get("status").and_then(|value| value.as_str()),
            Some("interrupted")
        );
        assert_eq!(
            host.cancelled.lock().unwrap().as_slice(),
            &["turn-1".to_string()]
        );
    }

    #[test]
    fn rlm_delegate_docs_explain_start_and_await() {
        let definition =
            delegate_tool_definitions(lash::ExecutionMode::Rlm, lash::ExecutionMode::Standard)
                .into_iter()
                .find(|definition| definition.name == "delegate")
                .expect("delegate definition");
        assert!(definition.description.contains("start call delegate"));
        assert!(
            definition
                .examples
                .iter()
                .any(|example| example.contains("await handle"))
        );
    }
}
