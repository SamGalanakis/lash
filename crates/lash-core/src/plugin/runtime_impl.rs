use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::sync::{Mutex as StdMutex, Weak};

use super::*;

#[derive(Clone)]
pub struct PluginHost {
    factories: Arc<Vec<Arc<dyn PluginFactory>>>,
    lashlang_abilities: lashlang::LashlangAbilities,
    sessions: Arc<StdMutex<BTreeMap<String, Weak<PluginSession>>>>,
}

struct BuildPluginSessionRequest<'a> {
    session_id: String,
    parent_session_id: Option<String>,
    execution_mode: ExecutionMode,
    standard_context_approach: Option<crate::StandardContextApproach>,
    snapshot: Option<&'a PluginSessionSnapshot>,
    tool_surface_overlay: ToolSurfaceContribution,
    tool_snapshot: Option<crate::ToolState>,
    authority: SessionAuthorityContext,
}

#[derive(Clone, Debug, Default)]
pub struct SessionAuthorityContext {
    pub tool_access: SessionToolAccess,
    pub subagent: Option<SubagentSessionContext>,
}

impl PluginHost {
    pub fn empty() -> Self {
        Self::new(Vec::new())
    }

    pub fn new(factories: Vec<Arc<dyn PluginFactory>>) -> Self {
        let override_ids: BTreeSet<&'static str> =
            factories.iter().map(|factory| factory.id()).collect();
        let mut all_factories = super::builtin_plugin_factories();
        if !override_ids.is_empty() {
            all_factories.retain(|factory| !override_ids.contains(factory.id()));
        }
        all_factories.extend(factories);
        Self {
            factories: Arc::new(all_factories),
            lashlang_abilities: lashlang::LashlangAbilities::default(),
            sessions: Arc::new(StdMutex::new(BTreeMap::new())),
        }
    }

    pub fn with_lashlang_abilities(mut self, abilities: lashlang::LashlangAbilities) -> Self {
        self.lashlang_abilities = abilities;
        self
    }

    pub fn isolated_registry(&self) -> Self {
        Self {
            factories: Arc::clone(&self.factories),
            lashlang_abilities: self.lashlang_abilities,
            sessions: Arc::new(StdMutex::new(BTreeMap::new())),
        }
    }

    pub fn lashlang_abilities(&self) -> lashlang::LashlangAbilities {
        self.lashlang_abilities
    }

    pub fn factories(&self) -> &[Arc<dyn PluginFactory>] {
        self.factories.as_ref().as_slice()
    }

    pub fn supports_standard_context_approach(
        &self,
        standard_context_approach: &crate::StandardContextApproach,
    ) -> bool {
        let required = standard_context_approach.kind();
        self.factories().iter().any(|factory| {
            factory
                .supported_standard_context_approaches()
                .contains(&required)
        })
    }

    pub fn build_standard_session(
        &self,
        session_id: impl Into<String>,
        snapshot: Option<&PluginSessionSnapshot>,
    ) -> Result<Arc<PluginSession>, PluginError> {
        self.build_session(
            session_id,
            ExecutionMode::standard(),
            Some(crate::StandardContextApproach::default()),
            snapshot,
        )
    }

    pub fn build_session(
        &self,
        session_id: impl Into<String>,
        execution_mode: ExecutionMode,
        standard_context_approach: Option<crate::StandardContextApproach>,
        snapshot: Option<&PluginSessionSnapshot>,
    ) -> Result<Arc<PluginSession>, PluginError> {
        self.build_session_with_surface(
            session_id,
            execution_mode,
            standard_context_approach,
            snapshot,
            ToolSurfaceContribution::default(),
            None,
        )
    }

    /// Variant of [`build_session`] that records the caller as the
    /// parent of the new session. Plugin factories read
    /// [`PluginSessionContext::is_root_session`] to gate root-only
    /// behavior; anything that goes through the plain `build_session`
    /// is treated as a root session by default.
    pub fn build_session_with_parent(
        &self,
        session_id: impl Into<String>,
        parent_session_id: Option<String>,
        execution_mode: ExecutionMode,
        standard_context_approach: Option<crate::StandardContextApproach>,
        snapshot: Option<&PluginSessionSnapshot>,
        authority: SessionAuthorityContext,
    ) -> Result<Arc<PluginSession>, PluginError> {
        self.build_session_with_parent_and_surface(
            session_id,
            parent_session_id,
            execution_mode,
            standard_context_approach,
            snapshot,
            ToolSurfaceContribution::default(),
            None,
            authority,
        )
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "public plugin-host boundary keeps parent, snapshot, tool overlay, and authority explicit"
    )]
    pub fn build_session_with_parent_and_surface(
        &self,
        session_id: impl Into<String>,
        parent_session_id: Option<String>,
        execution_mode: ExecutionMode,
        standard_context_approach: Option<crate::StandardContextApproach>,
        snapshot: Option<&PluginSessionSnapshot>,
        tool_surface_overlay: ToolSurfaceContribution,
        tool_snapshot: Option<crate::ToolState>,
        authority: SessionAuthorityContext,
    ) -> Result<Arc<PluginSession>, PluginError> {
        self.build_session_inner(BuildPluginSessionRequest {
            session_id: session_id.into(),
            parent_session_id,
            execution_mode,
            standard_context_approach,
            snapshot,
            tool_surface_overlay,
            tool_snapshot,
            authority,
        })
    }

    pub fn build_session_with_surface(
        &self,
        session_id: impl Into<String>,
        execution_mode: ExecutionMode,
        standard_context_approach: Option<crate::StandardContextApproach>,
        snapshot: Option<&PluginSessionSnapshot>,
        tool_surface_overlay: ToolSurfaceContribution,
        tool_snapshot: Option<crate::ToolState>,
    ) -> Result<Arc<PluginSession>, PluginError> {
        self.build_session_inner(BuildPluginSessionRequest {
            session_id: session_id.into(),
            parent_session_id: None,
            execution_mode,
            standard_context_approach,
            snapshot,
            tool_surface_overlay,
            tool_snapshot,
            authority: SessionAuthorityContext::default(),
        })
    }

    fn build_session_inner(
        &self,
        request: BuildPluginSessionRequest<'_>,
    ) -> Result<Arc<PluginSession>, PluginError> {
        let BuildPluginSessionRequest {
            session_id,
            parent_session_id,
            execution_mode,
            standard_context_approach,
            snapshot,
            tool_surface_overlay,
            tool_snapshot,
            authority,
        } = request;
        if execution_mode == ExecutionMode::standard() {
            let approach = standard_context_approach.as_ref().ok_or_else(|| {
                PluginError::Registration(
                    "standard execution requires a standard context approach".to_string(),
                )
            })?;
            if matches!(
                approach,
                crate::StandardContextApproach::ObservationalMemory(_)
            ) && !self.supports_standard_context_approach(approach)
            {
                return Err(PluginError::Registration(format!(
                    "standard context approach `{:?}` requires a supporting plugin factory on this plugin host",
                    approach.kind()
                )));
            }
        } else if standard_context_approach.is_some() {
            return Err(PluginError::Registration(
                "standard context approach only applies to standard execution mode".to_string(),
            ));
        }
        let ctx = PluginSessionContext {
            session_id,
            execution_mode: execution_mode.clone(),
            standard_context_approach: standard_context_approach.clone(),
            tool_access: authority.tool_access.clone(),
            subagent: authority.subagent.clone(),
            lashlang_abilities: self.lashlang_abilities,
            parent_session_id,
        };
        let session_id = ctx.session_id.clone();
        let mut plugins = Vec::new();
        let mut reg = PluginRegistrar::new();
        for factory in self.factories() {
            let plugin = factory.build(&ctx)?;
            reg.registering_plugin_id = Some(plugin.id().to_string());
            plugin.register(&mut reg)?;
            reg.registering_plugin_id = None;
            plugins.push(plugin);
        }
        let mut contributions = reg.contributions;
        let mode_session = contributions.mode_session.take().ok_or_else(|| {
            PluginError::Registration(format!(
                "missing mode session capability for {:?}",
                execution_mode
            ))
        })?;
        let mode_protocol_driver = contributions.mode_protocol_driver.take();
        for provider in &contributions.mode_native_tools {
            for manifest in provider.tool_manifests() {
                if !reg.tool_names.insert(manifest.name.clone()) {
                    return Err(PluginError::Registration(format!(
                        "duplicate mode native tool name `{}`",
                        manifest.name
                    )));
                }
            }
        }
        contributions.mode_session = Some(mode_session);
        contributions.mode_protocol_driver = mode_protocol_driver;
        contributions
            .turn_context_transforms
            .sort_by_key(|entry| std::cmp::Reverse(entry.0));
        contributions
            .history_rewriters
            .sort_by_key(|entry| std::cmp::Reverse(entry.0));
        let base_tools: Arc<dyn ToolProvider> =
            Arc::new(crate::tool_provider::CompositeToolProvider::from_providers(
                contributions.tool_providers.clone(),
            ));
        let registry = match tool_snapshot {
            Some(snapshot) => Arc::new(
                crate::ToolRegistry::from_tool_provider(base_tools)
                    .map_err(|err| {
                        PluginError::Registration(format!("failed to build tool registry: {err}"))
                    })?
                    .fork_with_state(snapshot)
                    .map_err(|err| {
                        PluginError::Session(format!(
                            "tool state cannot be applied to this plugin host session: {err}"
                        ))
                    })?,
            ),
            None => Arc::new(crate::ToolRegistry::from_tool_provider(base_tools).map_err(
                |err| PluginError::Registration(format!("failed to build tool registry: {err}")),
            )?),
        };
        let tools = Arc::clone(&registry) as Arc<dyn ToolProvider>;

        let session = Arc::new(PluginSession {
            host: self.clone(),
            session_id: ctx.session_id,
            execution_mode: execution_mode.clone(),
            plugins,
            tools,
            tool_registry: registry,
            tool_surface_overlay,
            tool_access: authority.tool_access,
            subagent: authority.subagent,
            lashlang_abilities: self.lashlang_abilities,
            contributions,
        });
        self.register_session(&session_id, &session)?;
        let ready = SessionReadyContext {
            session_id: session.session_id.clone(),
            execution_mode,
            standard_context_approach,
            host: self.clone(),
        };
        for plugin in &session.plugins {
            plugin.session_ready(ready.clone())?;
        }
        if let Some(snapshot) = snapshot {
            session.restore(snapshot)?;
        }
        Ok(session)
    }

    pub async fn invoke_plugin_action_sessionless(
        &self,
        name: &str,
        args: serde_json::Value,
    ) -> Result<ToolResult, PluginError> {
        let session = self.build_standard_session(
            format!("__external__-{}", uuid::Uuid::new_v4().simple()),
            None,
        )?;
        session
            .invoke_plugin_action(
                name,
                args,
                None,
                false,
                Arc::new(NoopSessionManager),
                Arc::new(crate::UnavailableProcessService),
            )
            .await
            .map_err(|err| PluginError::Invoke(err.to_string()))
    }

    fn register_session(
        &self,
        session_id: &str,
        session: &Arc<PluginSession>,
    ) -> Result<(), PluginError> {
        let mut sessions = self.sessions.lock().map_err(|_| {
            PluginError::Session("plugin host session registry poisoned".to_string())
        })?;
        if let Some(existing) = sessions.get(session_id).and_then(Weak::upgrade) {
            if !Arc::ptr_eq(&existing, session) {
                return Err(PluginError::Session(format!(
                    "session `{session_id}` is already registered on this plugin host"
                )));
            }
            return Ok(());
        }
        sessions.insert(session_id.to_string(), Arc::downgrade(session));
        Ok(())
    }

    pub fn unregister_session(&self, session_id: &str) -> Result<(), PluginError> {
        let mut sessions = self.sessions.lock().map_err(|_| {
            PluginError::Session("plugin host session registry poisoned".to_string())
        })?;
        sessions.remove(session_id);
        Ok(())
    }

    pub fn session(&self, session_id: &str) -> Result<Arc<PluginSession>, PluginActionInvokeError> {
        let mut sessions = self
            .sessions
            .lock()
            .map_err(|_| PluginActionInvokeError::SessionRegistryPoisoned)?;
        let Some(weak) = sessions.get(session_id).cloned() else {
            return Err(PluginActionInvokeError::UnknownSession(
                session_id.to_string(),
            ));
        };
        match weak.upgrade() {
            Some(session) => Ok(session),
            None => {
                sessions.remove(session_id);
                Err(PluginActionInvokeError::UnknownSession(
                    session_id.to_string(),
                ))
            }
        }
    }

    pub async fn invoke_plugin_action_for_session(
        &self,
        session_id: &str,
        name: &str,
        args: serde_json::Value,
        host: Arc<dyn RuntimeSessionHost>,
        processes: Arc<dyn crate::ProcessService>,
    ) -> Result<ToolResult, PluginActionInvokeError> {
        let session = self.session(session_id)?;
        session
            .invoke_plugin_action(
                name,
                args,
                Some(session_id.to_string()),
                false,
                host,
                processes,
            )
            .await
    }
}
