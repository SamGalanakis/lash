use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use tokio::task::JoinSet;

use super::*;
use crate::dynamic::DynamicCapabilityDef;

pub struct PluginRegistrar {
    tool_names: BTreeSet<String>,
    tool_providers: Vec<Arc<dyn ToolProvider>>,
    capability_defs: BTreeMap<String, DynamicCapabilityDef>,
    prompt_sections: Vec<String>,
    prompt_contributors: Vec<PromptContributor>,
    before_turn_hooks: Vec<BeforeTurnHook>,
    before_tool_call_hooks: Vec<BeforeToolCallHook>,
    after_tool_call_hooks: Vec<AfterToolCallHook>,
    after_turn_hooks: Vec<AfterTurnHook>,
    turn_committed_hooks: Vec<TurnCommittedHook>,
    session_restored_hooks: Vec<SessionRestoredHook>,
    external_ops: BTreeMap<String, RegisteredExternalOp>,
}

impl PluginRegistrar {
    fn new() -> Self {
        Self {
            tool_names: BTreeSet::new(),
            tool_providers: Vec::new(),
            capability_defs: BTreeMap::new(),
            prompt_sections: Vec::new(),
            prompt_contributors: Vec::new(),
            before_turn_hooks: Vec::new(),
            before_tool_call_hooks: Vec::new(),
            after_tool_call_hooks: Vec::new(),
            after_turn_hooks: Vec::new(),
            turn_committed_hooks: Vec::new(),
            session_restored_hooks: Vec::new(),
            external_ops: BTreeMap::new(),
        }
    }

    pub fn register_tool_provider(
        &mut self,
        provider: Arc<dyn ToolProvider>,
    ) -> Result<(), PluginError> {
        for def in provider.definitions() {
            if !self.tool_names.insert(def.name.clone()) {
                return Err(PluginError::Registration(format!(
                    "duplicate plugin tool name `{}`",
                    def.name
                )));
            }
        }
        self.tool_providers.push(provider);
        Ok(())
    }

    pub fn register_capability(&mut self, def: DynamicCapabilityDef) -> Result<(), PluginError> {
        if self.capability_defs.contains_key(&def.id) {
            return Err(PluginError::Registration(format!(
                "duplicate plugin capability `{}`",
                def.id
            )));
        }
        self.capability_defs.insert(def.id.clone(), def);
        Ok(())
    }

    pub fn register_prompt_section(&mut self, section: impl Into<String>) {
        let section = section.into();
        if !section.trim().is_empty() {
            self.prompt_sections.push(section);
        }
    }

    pub fn register_prompt_contributor(&mut self, contributor: PromptContributor) {
        self.prompt_contributors.push(contributor);
    }

    pub fn before_turn(&mut self, hook: BeforeTurnHook) {
        self.before_turn_hooks.push(hook);
    }

    pub fn before_tool_call(&mut self, hook: BeforeToolCallHook) {
        self.before_tool_call_hooks.push(hook);
    }

    pub fn after_tool_call(&mut self, hook: AfterToolCallHook) {
        self.after_tool_call_hooks.push(hook);
    }

    pub fn after_turn(&mut self, hook: AfterTurnHook) {
        self.after_turn_hooks.push(hook);
    }

    pub fn on_turn_committed(&mut self, hook: TurnCommittedHook) {
        self.turn_committed_hooks.push(hook);
    }

    pub fn on_session_restored(&mut self, hook: SessionRestoredHook) {
        self.session_restored_hooks.push(hook);
    }

    pub fn register_external_op(
        &mut self,
        def: ExternalOpDef,
        handler: ExternalInvokeHandler,
    ) -> Result<(), PluginError> {
        if self.external_ops.contains_key(&def.name) {
            return Err(PluginError::Registration(format!(
                "duplicate external invoke name `{}`",
                def.name
            )));
        }
        self.external_ops
            .insert(def.name.clone(), RegisteredExternalOp { def, handler });
        Ok(())
    }
}

#[derive(Clone)]
pub struct PluginHost {
    factories: Arc<Vec<Arc<dyn PluginFactory>>>,
}

impl PluginHost {
    pub fn empty() -> Self {
        Self {
            factories: Arc::new(Vec::new()),
        }
    }

    pub fn new(factories: Vec<Arc<dyn PluginFactory>>) -> Self {
        Self {
            factories: Arc::new(factories),
        }
    }

    pub fn factories(&self) -> &[Arc<dyn PluginFactory>] {
        self.factories.as_ref().as_slice()
    }

    pub fn build_session(
        &self,
        agent_id: impl Into<String>,
        snapshot: Option<&PluginSessionSnapshot>,
    ) -> Result<Arc<PluginSession>, PluginError> {
        let ctx = PluginSessionContext {
            agent_id: agent_id.into(),
        };
        let mut plugins = Vec::new();
        let mut reg = PluginRegistrar::new();
        for factory in self.factories() {
            let plugin = factory.build(&ctx)?;
            plugin.register(&mut reg)?;
            plugins.push(plugin);
        }

        let session = Arc::new(PluginSession {
            host: self.clone(),
            agent_id: ctx.agent_id,
            plugins,
            tool_providers: reg.tool_providers,
            capability_defs: reg.capability_defs,
            prompt_sections: reg.prompt_sections,
            prompt_contributors: reg.prompt_contributors,
            before_turn_hooks: reg.before_turn_hooks,
            before_tool_call_hooks: reg.before_tool_call_hooks,
            after_tool_call_hooks: reg.after_tool_call_hooks,
            after_turn_hooks: reg.after_turn_hooks,
            turn_committed_hooks: reg.turn_committed_hooks,
            session_restored_hooks: reg.session_restored_hooks,
            external_ops: reg.external_ops,
        });
        if let Some(snapshot) = snapshot {
            session.restore(snapshot)?;
        }
        Ok(session)
    }

    pub async fn invoke_external_sessionless(
        &self,
        name: &str,
        args: serde_json::Value,
    ) -> Result<ToolResult, PluginError> {
        let session = self.build_session("__external__", None)?;
        session
            .invoke_external(name, args, None, false, Arc::new(NoopSessionManager))
            .await
            .map_err(|err| PluginError::Invoke(err.to_string()))
    }
}

pub struct PluginSession {
    host: PluginHost,
    agent_id: String,
    plugins: Vec<Arc<dyn SessionPlugin>>,
    tool_providers: Vec<Arc<dyn ToolProvider>>,
    capability_defs: BTreeMap<String, DynamicCapabilityDef>,
    prompt_sections: Vec<String>,
    prompt_contributors: Vec<PromptContributor>,
    before_turn_hooks: Vec<BeforeTurnHook>,
    before_tool_call_hooks: Vec<BeforeToolCallHook>,
    after_tool_call_hooks: Vec<AfterToolCallHook>,
    after_turn_hooks: Vec<AfterTurnHook>,
    turn_committed_hooks: Vec<TurnCommittedHook>,
    session_restored_hooks: Vec<SessionRestoredHook>,
    external_ops: BTreeMap<String, RegisteredExternalOp>,
}

impl PluginSession {
    pub fn agent_id(&self) -> &str {
        &self.agent_id
    }

    pub fn host(&self) -> &PluginHost {
        &self.host
    }

    pub fn tool_providers(&self) -> &[Arc<dyn ToolProvider>] {
        &self.tool_providers
    }

    pub fn capability_defs(&self) -> &BTreeMap<String, DynamicCapabilityDef> {
        &self.capability_defs
    }

    pub fn prompt_sections(&self) -> &[String] {
        &self.prompt_sections
    }

    pub fn prompt_contributors(&self) -> &[PromptContributor] {
        &self.prompt_contributors
    }

    pub fn external_ops(&self) -> Vec<ExternalOpDef> {
        self.external_ops
            .values()
            .map(|op| op.def.clone())
            .collect()
    }

    pub fn collect_prompt_contributions(
        &self,
        ctx: PromptHookContext,
    ) -> Result<Vec<PromptContribution>, PluginError> {
        let mut out = self
            .prompt_sections
            .iter()
            .map(|content| PromptContribution {
                section: PromptSectionName::PluginExtensions,
                priority: 0,
                content: content.clone(),
            })
            .collect::<Vec<_>>();
        for contributor in &self.prompt_contributors {
            out.extend(contributor(ctx.clone())?);
        }
        out.sort_by(|a, b| {
            a.section
                .as_str()
                .cmp(b.section.as_str())
                .then(a.priority.cmp(&b.priority))
        });
        Ok(out)
    }

    pub async fn before_turn(
        &self,
        ctx: TurnHookContext,
    ) -> Result<Vec<PluginDirective>, PluginError> {
        let mut directives = Vec::new();
        for hook in &self.before_turn_hooks {
            directives.extend(hook(ctx.clone()).await?);
        }
        Ok(directives)
    }

    pub async fn before_tool_call(
        &self,
        ctx: ToolCallHookContext,
    ) -> Result<Vec<PluginDirective>, PluginError> {
        let mut directives = Vec::new();
        for hook in &self.before_tool_call_hooks {
            directives.extend(hook(ctx.clone()).await?);
        }
        Ok(directives)
    }

    pub async fn after_tool_call(
        &self,
        ctx: ToolResultHookContext,
    ) -> Result<Vec<PluginDirective>, PluginError> {
        let mut directives = Vec::new();
        for hook in &self.after_tool_call_hooks {
            directives.extend(hook(ctx.clone()).await?);
        }
        Ok(directives)
    }

    pub async fn after_turn(
        &self,
        ctx: TurnResultHookContext,
    ) -> Result<Vec<PluginDirective>, PluginError> {
        let mut directives = Vec::new();
        for hook in &self.after_turn_hooks {
            directives.extend(hook(ctx.clone()).await?);
        }
        Ok(directives)
    }

    pub async fn on_turn_committed(&self, turn: &AssembledTurn) {
        let mut tasks = JoinSet::new();
        for hook in &self.turn_committed_hooks {
            let hook = Arc::clone(hook);
            let turn = turn.clone();
            tasks.spawn(async move { hook(turn).await });
        }

        while let Some(result) = tasks.join_next().await {
            match result {
                Ok(Ok(())) => {}
                Ok(Err(err)) => tracing::warn!("plugin turn hook failed: {err}"),
                Err(err) => tracing::warn!("plugin turn hook task failed: {err}"),
            }
        }
    }

    pub async fn on_session_restored(&self, state: &AgentStateEnvelope) {
        let mut tasks = JoinSet::new();
        for hook in &self.session_restored_hooks {
            let hook = Arc::clone(hook);
            let state = state.clone();
            tasks.spawn(async move { hook(state).await });
        }

        while let Some(result) = tasks.join_next().await {
            match result {
                Ok(Ok(())) => {}
                Ok(Err(err)) => tracing::warn!("plugin restore hook failed: {err}"),
                Err(err) => tracing::warn!("plugin restore hook task failed: {err}"),
            }
        }
    }

    pub fn snapshot(&self) -> Result<PluginSessionSnapshot, PluginError> {
        let mut plugins = BTreeMap::new();
        for plugin in &self.plugins {
            let mut writer = InMemorySnapshotWriter::default();
            let meta = plugin.snapshot(&mut writer)?;
            plugins.insert(
                plugin.id().to_string(),
                PluginSnapshotEntry {
                    meta,
                    artifacts: writer.finish(),
                },
            );
        }
        Ok(PluginSessionSnapshot { plugins })
    }

    pub fn restore(&self, snapshot: &PluginSessionSnapshot) -> Result<(), PluginError> {
        for plugin in &self.plugins {
            let Some(entry) = snapshot.plugins.get(plugin.id()) else {
                continue;
            };
            let reader = InMemorySnapshotReader { entry };
            plugin.restore(&entry.meta, &reader)?;
        }
        Ok(())
    }

    pub fn fork_for_agent(
        &self,
        agent_id: impl Into<String>,
    ) -> Result<Arc<PluginSession>, PluginError> {
        let snapshot = self.snapshot()?;
        self.host.build_session(agent_id, Some(&snapshot))
    }

    pub async fn invoke_external(
        &self,
        name: &str,
        args: serde_json::Value,
        session_id: Option<String>,
        default_to_current_session: bool,
        host: Arc<dyn SessionManager>,
    ) -> Result<ToolResult, ExternalInvokeError> {
        let Some(op) = self.external_ops.get(name).cloned() else {
            return Err(ExternalInvokeError::Unknown(name.to_string()));
        };

        let effective_session = session_id.or_else(|| {
            if default_to_current_session && !self.agent_id.is_empty() {
                Some(self.agent_id.clone())
            } else {
                None
            }
        });

        match (op.def.session_param, effective_session.as_ref()) {
            (SessionParam::Required, None) => {
                return Err(ExternalInvokeError::MissingSession(name.to_string()));
            }
            (SessionParam::Forbidden, Some(_)) => {
                return Err(ExternalInvokeError::UnexpectedSession(name.to_string()));
            }
            _ => {}
        }

        Ok((op.handler)(
            ExternalInvokeContext {
                session_id: effective_session,
                host,
            },
            args,
        )
        .await)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ExternalInvokeError {
    #[error("unknown external invoke `{0}`")]
    Unknown(String),
    #[error("external invoke `{0}` requires a session")]
    MissingSession(String),
    #[error("external invoke `{0}` does not accept a session")]
    UnexpectedSession(String),
}

#[derive(Clone)]
pub struct RuntimeServices {
    pub tools: Arc<dyn ToolProvider>,
    pub plugins: Arc<PluginSession>,
}

pub(crate) struct NoopSessionManager;

#[async_trait::async_trait]
impl SessionManager for NoopSessionManager {
    async fn snapshot_current(&self) -> Result<SessionSnapshot, PluginError> {
        Err(PluginError::Session(
            "session snapshots are unavailable in this runtime".to_string(),
        ))
    }

    async fn snapshot_session(&self, _session_id: &str) -> Result<SessionSnapshot, PluginError> {
        Err(PluginError::Session(
            "session lookup is unavailable in this runtime".to_string(),
        ))
    }

    async fn create_session(
        &self,
        _request: SessionCreateRequest,
    ) -> Result<SessionHandle, PluginError> {
        Err(PluginError::Session(
            "session creation is unavailable in this runtime".to_string(),
        ))
    }

    async fn close_session(&self, _session_id: &str) -> Result<(), PluginError> {
        Err(PluginError::Session(
            "session closing is unavailable in this runtime".to_string(),
        ))
    }

    async fn start_turn(
        &self,
        _session_id: &str,
        _input: TurnInput,
    ) -> Result<AssembledTurn, PluginError> {
        Err(PluginError::Session(
            "session execution is unavailable in this runtime".to_string(),
        ))
    }
}

impl RuntimeServices {
    pub fn new(tools: Arc<dyn ToolProvider>, plugins: Arc<PluginSession>) -> Self {
        Self { tools, plugins }
    }

    pub fn tools_only(
        tools: Arc<dyn ToolProvider>,
        agent_id: impl Into<String>,
    ) -> Result<Self, PluginError> {
        let host = PluginHost::empty();
        let plugins = host.build_session(agent_id, None)?;
        Ok(Self { tools, plugins })
    }
}
