use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::sync::{Mutex as StdMutex, Weak};

use super::*;

#[derive(Clone)]
pub struct PluginHost {
    factories: Arc<Vec<Arc<dyn PluginFactory>>>,
    lashlang_abilities: lashlang::LashlangAbilities,
    lashlang_language_features: lashlang::LashlangLanguageFeatures,
    lashlang_resources: lashlang::ResourceCatalog,
    sessions: Arc<StdMutex<BTreeMap<String, Weak<PluginSession>>>>,
}

struct BuildPluginSessionRequest<'a> {
    session_id: String,
    parent_session_id: Option<String>,
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
        let lashlang_abilities = all_factories.iter().fold(
            lashlang::LashlangAbilities::default(),
            |abilities, factory| abilities.union(factory.lashlang_abilities()),
        );
        let lashlang_language_features = all_factories.iter().fold(
            lashlang::LashlangLanguageFeatures::default(),
            |features, factory| features.union(factory.lashlang_language_features()),
        );
        let lashlang_resources = all_factories.iter().fold(
            lashlang::ResourceCatalog::new(),
            |mut resources, factory| {
                resources.extend(factory.lashlang_resources());
                resources
            },
        );
        Self {
            factories: Arc::new(all_factories),
            lashlang_abilities,
            lashlang_language_features,
            lashlang_resources,
            sessions: Arc::new(StdMutex::new(BTreeMap::new())),
        }
    }

    pub fn with_lashlang_abilities(mut self, abilities: lashlang::LashlangAbilities) -> Self {
        self.lashlang_abilities = abilities;
        self
    }

    pub fn with_lashlang_language_features(
        mut self,
        language_features: lashlang::LashlangLanguageFeatures,
    ) -> Self {
        self.lashlang_language_features = language_features;
        self
    }

    pub fn with_lashlang_resources(mut self, resources: lashlang::ResourceCatalog) -> Self {
        self.lashlang_resources = resources;
        self
    }

    pub fn isolated_registry(&self) -> Self {
        Self {
            factories: Arc::clone(&self.factories),
            lashlang_abilities: self.lashlang_abilities,
            lashlang_language_features: self.lashlang_language_features,
            lashlang_resources: self.lashlang_resources.clone(),
            sessions: Arc::new(StdMutex::new(BTreeMap::new())),
        }
    }

    pub fn lashlang_abilities(&self) -> lashlang::LashlangAbilities {
        self.lashlang_abilities
    }

    pub fn lashlang_language_features(&self) -> lashlang::LashlangLanguageFeatures {
        self.lashlang_language_features
    }

    pub fn lashlang_resources(&self) -> lashlang::ResourceCatalog {
        self.lashlang_resources.clone()
    }

    pub fn factories(&self) -> &[Arc<dyn PluginFactory>] {
        self.factories.as_ref().as_slice()
    }

    pub fn build_session(
        &self,
        session_id: impl Into<String>,
        snapshot: Option<&PluginSessionSnapshot>,
    ) -> Result<Arc<PluginSession>, PluginError> {
        self.build_session_with_surface(
            session_id,
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
        snapshot: Option<&PluginSessionSnapshot>,
        authority: SessionAuthorityContext,
    ) -> Result<Arc<PluginSession>, PluginError> {
        self.build_session_with_parent_and_surface(
            session_id,
            parent_session_id,
            snapshot,
            ToolSurfaceContribution::default(),
            None,
            authority,
        )
    }

    pub fn build_session_with_parent_and_surface(
        &self,
        session_id: impl Into<String>,
        parent_session_id: Option<String>,
        snapshot: Option<&PluginSessionSnapshot>,
        tool_surface_overlay: ToolSurfaceContribution,
        tool_snapshot: Option<crate::ToolState>,
        authority: SessionAuthorityContext,
    ) -> Result<Arc<PluginSession>, PluginError> {
        self.build_session_inner(BuildPluginSessionRequest {
            session_id: session_id.into(),
            parent_session_id,
            snapshot,
            tool_surface_overlay,
            tool_snapshot,
            authority,
        })
    }

    pub fn build_session_with_surface(
        &self,
        session_id: impl Into<String>,
        snapshot: Option<&PluginSessionSnapshot>,
        tool_surface_overlay: ToolSurfaceContribution,
        tool_snapshot: Option<crate::ToolState>,
    ) -> Result<Arc<PluginSession>, PluginError> {
        self.build_session_inner(BuildPluginSessionRequest {
            session_id: session_id.into(),
            parent_session_id: None,
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
            snapshot,
            tool_surface_overlay,
            tool_snapshot,
            authority,
        } = request;
        let ctx = PluginSessionContext {
            session_id,
            tool_access: authority.tool_access.clone(),
            subagent: authority.subagent.clone(),
            lashlang_abilities: self.lashlang_abilities,
            lashlang_language_features: self.lashlang_language_features,
            parent_session_id,
        };
        let session_id = ctx.session_id.clone();
        let mut tool_snapshot = tool_snapshot;
        if let Some(snapshot) = &mut tool_snapshot {
            let hidden_tools = &authority.tool_access.hidden_tools;
            if !hidden_tools.is_empty() {
                snapshot.retain(|name, _| !hidden_tools.contains(name));
            }
        }
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
        let protocol_session = contributions.protocol_session.take().ok_or_else(|| {
            PluginError::Registration("missing protocol session capability".to_string())
        })?;
        let protocol_driver = contributions.protocol_driver.take().ok_or_else(|| {
            PluginError::Registration("missing protocol driver capability".to_string())
        })?;
        contributions.protocol_session = Some(protocol_session);
        contributions.protocol_driver = Some(protocol_driver);
        contributions
            .turn_context_transforms
            .sort_by_key(|entry| std::cmp::Reverse(entry.0));
        contributions
            .context_compactors
            .sort_by_key(|entry| std::cmp::Reverse(entry.0));
        let host_events = crate::HostEventCatalog::from_events(contributions.host_events.clone())
            .map_err(|message| {
            PluginError::Registration(format!("invalid host event catalog: {message}"))
        })?;
        let mut lashlang_resources = self.lashlang_resources.clone();
        for event in host_events.events() {
            lashlang_resources
                .add_trigger_source_constructor(
                    event.source_type().split('.'),
                    lashlang::TypeExpr::Object(Vec::new()),
                    event.payload_type().clone(),
                )
                .map_err(|err| {
                    PluginError::Registration(format!(
                        "invalid host event trigger source `{}.{}`: {err}",
                        event.alias, event.event
                    ))
                })?;
        }
        let registry = match tool_snapshot {
            Some(snapshot) => Arc::new(
                crate::ToolRegistry::from_tool_providers(contributions.tool_providers.clone())
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
            None => Arc::new(
                crate::ToolRegistry::from_tool_providers(contributions.tool_providers.clone())
                    .map_err(|err| {
                        PluginError::Registration(format!("failed to build tool registry: {err}"))
                    })?,
            ),
        };
        let tools = Arc::clone(&registry) as Arc<dyn ToolProvider>;

        let session = Arc::new(PluginSession {
            host: self.clone(),
            session_id: ctx.session_id,
            plugins,
            tools,
            tool_registry: registry,
            tool_surface_overlay,
            tool_access: authority.tool_access,
            subagent: authority.subagent,
            lashlang_abilities: self.lashlang_abilities,
            lashlang_language_features: self.lashlang_language_features,
            lashlang_resources,
            host_events,
            contributions,
        });
        self.register_session(&session_id, &session)?;
        let ready = SessionReadyContext {
            session_id: session.session_id.clone(),
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
        let session = self.build_session(
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
                Arc::new(NoopSessionManager),
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

    #[expect(
        clippy::too_many_arguments,
        reason = "host action invocation wires the runtime service bundle at the plugin boundary"
    )]
    pub async fn invoke_plugin_action_for_session(
        &self,
        session_id: &str,
        name: &str,
        args: serde_json::Value,
        sessions: Arc<dyn SessionStateService>,
        session_lifecycle: Arc<dyn SessionLifecycleService>,
        session_graph: Arc<dyn SessionGraphService>,
        processes: Arc<dyn crate::ProcessService>,
    ) -> Result<ToolResult, PluginActionInvokeError> {
        let session = self.session(session_id)?;
        session
            .invoke_plugin_action(
                name,
                args,
                Some(session_id.to_string()),
                false,
                sessions,
                session_lifecycle,
                session_graph,
                processes,
            )
            .await
    }
}
