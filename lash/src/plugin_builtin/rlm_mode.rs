use std::sync::Arc;

#[path = "rlm_support.rs"]
mod rlm_support;

use crate::plugin::{
    ModeNativeToolsPlugin, ModeSessionPlugin, PluginError, PluginFactory, PluginRegistrar,
    PluginSessionContext, SessionPlugin,
};
use crate::{
    ExecutionMode, ProgressSender, SessionError, ToolResult, ToolResultProjectionPluginConfig,
};

use self::rlm_support::{SearchToolsProvider, bound_variables_prompt_contributions};

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
#[derive(Default)]
pub struct RlmModePluginConfig {
    pub observe_projection: ToolResultProjectionPluginConfig,
}

pub struct BuiltinRlmModePluginFactory {
    config: RlmModePluginConfig,
}

impl BuiltinRlmModePluginFactory {
    pub fn new(config: RlmModePluginConfig) -> Self {
        Self { config }
    }
}

impl Default for BuiltinRlmModePluginFactory {
    fn default() -> Self {
        Self::new(RlmModePluginConfig::default())
    }
}

impl PluginFactory for BuiltinRlmModePluginFactory {
    fn id(&self) -> &'static str {
        "mode_rlm"
    }

    fn build(&self, ctx: &PluginSessionContext) -> Result<Arc<dyn SessionPlugin>, PluginError> {
        Ok(Arc::new(RlmModePlugin {
            active: matches!(ctx.execution_mode, ExecutionMode::Rlm),
            provider: Arc::new(SearchToolsProvider::new()),
            config: self.config.clone(),
        }))
    }
}

struct RlmModePlugin {
    active: bool,
    provider: Arc<SearchToolsProvider>,
    config: RlmModePluginConfig,
}

impl SessionPlugin for RlmModePlugin {
    fn id(&self) -> &'static str {
        "mode_rlm"
    }

    fn register(&self, reg: &mut PluginRegistrar) -> Result<(), PluginError> {
        if !self.active {
            return Ok(());
        }
        reg.mode().session(Arc::new(RlmModeSession {
            config: self.config.clone(),
        }))?;
        // The native-tools slot is owned by `StandardModePlugin`, which
        // surfaces the execution-mode-agnostic runtime-control tools
        // (`monitor`, `tasks_list`, `tasks_stop`) in both modes. RLM has
        // no extra native tools to add — orchestration primitives like
        // `parallel { }` / `start call` live in lashlang itself — so we
        // don't claim the slot here.
        reg.tools()
            .provider(Arc::clone(&self.provider) as Arc<dyn crate::ToolProvider>)?;
        reg.prompt().contribute(Arc::new(move |ctx| {
            Box::pin(async move { Ok(bound_variables_prompt_contributions(&ctx)) })
        }));
        Ok(())
    }
}

struct RlmModeSession {
    config: RlmModePluginConfig,
}

#[async_trait::async_trait]
impl ModeSessionPlugin for RlmModeSession {
    async fn initialize_session(
        &self,
        session: &mut crate::Session,
        session_id: &str,
    ) -> Result<(), SessionError> {
        session.set_rlm_observe_projection_config(self.config.observe_projection.clone());
        session.start_rlm_runtime(session_id).await
    }

    async fn restore_session(
        &self,
        session: &mut crate::Session,
        state: &crate::runtime::PersistedSessionState,
    ) -> Result<(), SessionError> {
        if let Some(snapshot) = state.execution_state_snapshot.clone() {
            session.restore_execution_state(&snapshot).await?;
        }
        for body in state
            .session_graph
            .active_path_plugins(crate::INTERNAL_RLM_GLOBALS_PATCH_PLUGIN_TYPE)
        {
            let patch = serde_json::from_value::<crate::RlmGlobalsPatchPluginBody>(body.clone())
                .map_err(|err| {
                    SessionError::Protocol(format!("invalid RLM globals patch node: {err}"))
                })?;
            session.apply_rlm_globals_patch(&patch).await?;
        }
        Ok(())
    }

    async fn append_session_nodes(
        &self,
        session: &mut crate::Session,
        nodes: &[crate::SessionAppendNode],
    ) -> Result<(), SessionError> {
        for node in nodes {
            let crate::SessionAppendNode::Plugin { plugin_type, body } = node else {
                continue;
            };
            if plugin_type != crate::INTERNAL_RLM_GLOBALS_PATCH_PLUGIN_TYPE {
                continue;
            }
            let patch = serde_json::from_value::<crate::RlmGlobalsPatchPluginBody>(body.clone())
                .map_err(|err| {
                    SessionError::Protocol(format!("invalid RLM globals patch node body: {err}"))
                })?;
            session.apply_rlm_globals_patch(&patch).await?;
        }
        Ok(())
    }

    fn configure_runtime_from_request(
        &self,
        runtime: &mut crate::runtime::LashRuntime,
        request: &crate::SessionCreateRequest,
    ) {
        if let crate::ModeExtras::Rlm(extras) = &request.mode_extras {
            runtime.set_repl_termination(extras.termination.clone());
        }
    }
}

struct RlmModeNativeTools;

#[async_trait::async_trait]
impl ModeNativeToolsPlugin for RlmModeNativeTools {
    fn definitions(&self) -> Vec<crate::ToolDefinition> {
        Vec::new()
    }

    async fn execute(
        &self,
        _context: &crate::tool_dispatch::ToolDispatchContext,
        _name: &str,
        _args: &serde_json::Value,
        _progress: Option<&ProgressSender>,
    ) -> Option<ToolResult> {
        None
    }
}
