use super::*;
use crate::runtime::host::{EmbeddedRuntimeHost, ProcessRuntimeHost};

struct SessionCreatePlan {
    session_id: String,
    relation: SessionRelation,
    parent_session_id: Option<String>,
    start_snapshot: SessionSnapshot,
    policy: SessionPolicy,
    initial_runtime_state: SessionSnapshot,
    plugin_authority: crate::plugin::SessionAuthorityContext,
    plugin_source: crate::SessionPluginSource,
    plugin_options: crate::PluginOptions,
    context_surface: crate::SessionContextSurface,
    store_binding: Option<Arc<dyn crate::store::RuntimePersistence>>,
    usage_source: Option<String>,
    first_turn_input: Option<crate::PluginMessage>,
}

impl ManagedSessionCapability {
    pub(in crate::runtime::session_manager) fn build_runtime_state(
        &self,
        session_id: String,
        request: &SessionCreateRequest,
        mut base: SessionSnapshot,
        policy: &SessionPolicy,
    ) -> SessionSnapshot {
        normalize_session_graph(&mut base);
        base.session_id = session_id;
        base.policy = policy.clone();
        append_session_nodes_to_state(&mut base, &request.initial_nodes);
        normalize_session_graph(&mut base);
        base
    }

    async fn resolve_create_plan(
        &self,
        current: &CurrentSessionCapability,
        request: SessionCreateRequest,
    ) -> Result<SessionCreatePlan, crate::PluginError> {
        let session_id = request
            .session_id
            .clone()
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        if session_id == current.session_id || self.registry.lock().await.contains_key(&session_id)
        {
            return Err(crate::PluginError::Session(format!(
                "session `{session_id}` already exists"
            )));
        }
        let parent_session_id = request.relation.parent_session_id().map(ToOwned::to_owned);
        let start_snapshot = match &request.start {
            SessionStartPoint::Empty => SessionSnapshot {
                session_id: session_id.clone(),
                ..Default::default()
            },
            SessionStartPoint::CurrentSession => current.snapshot.to_snapshot(),
            SessionStartPoint::ExistingSession { session_id } => {
                current.snapshot_by_id(self, session_id).await?
            }
            SessionStartPoint::Snapshot { snapshot } => (**snapshot).clone(),
        };
        let mut policy = request
            .policy
            .clone()
            .unwrap_or_else(|| match &request.start {
                SessionStartPoint::Empty => current.policy.clone(),
                _ => start_snapshot.policy.clone(),
            });
        if parent_session_id.is_some() {
            policy.session_id = Some(session_id.clone());
        }
        let initial_runtime_state = self.build_runtime_state(
            session_id.clone(),
            &request,
            start_snapshot.clone(),
            &policy,
        );
        let plugin_authority = crate::plugin::SessionAuthorityContext {
            tool_access: request.tool_access.clone(),
            subagent: request.subagent.clone(),
        };
        let store_binding = match &current.host.session_store_factory {
            Some(factory) => {
                let store = factory
                    .create_store(&SessionStoreCreateRequest {
                        session_id: session_id.clone(),
                        relation: request.relation.clone(),
                        policy: policy.clone(),
                    })
                    .map_err(|message| {
                        crate::PluginError::Session(child_store_factory_error(
                            &session_id,
                            parent_session_id.as_deref(),
                            message,
                        ))
                    })?;
                validate_child_store_binding(
                    store.as_ref(),
                    &session_id,
                    parent_session_id.as_deref(),
                )
                .await?;
                Some(store)
            }
            None => None,
        };
        Ok(SessionCreatePlan {
            session_id,
            relation: request.relation,
            parent_session_id,
            start_snapshot,
            policy,
            initial_runtime_state,
            plugin_authority,
            plugin_source: request.plugin_source,
            plugin_options: request.plugin_options,
            context_surface: request.context_surface,
            store_binding,
            usage_source: request.usage_source,
            first_turn_input: request.first_turn_input,
        })
    }

    async fn materialize_create_plan(
        &self,
        current: &CurrentSessionCapability,
        plan: &SessionCreatePlan,
    ) -> Result<LashRuntime, crate::PluginError> {
        let plugins = match plan.plugin_source {
            crate::SessionPluginSource::CurrentHostFresh => current
                .plugins
                .host()
                .build_session_with_parent(
                    &plan.session_id,
                    plan.parent_session_id.clone(),
                    None,
                    plan.plugin_authority.clone(),
                )
                .map_err(|err| crate::PluginError::Session(err.to_string()))?,
            crate::SessionPluginSource::CurrentSessionFork => current
                .plugins
                .fork_for_child_session(
                    &plan.session_id,
                    plan.parent_session_id.clone(),
                    plan.plugin_authority.clone(),
                )
                .map_err(|err| crate::PluginError::Session(err.to_string()))?,
        };
        let mut runtime = match (&current.host.process_registry, &plan.store_binding) {
            (Some(executor), Some(store)) => {
                let host = ProcessRuntimeHost::new(
                    EmbeddedRuntimeHost {
                        core: current.host.core.clone(),
                        session_store_factory: current.host.session_store_factory.clone(),
                    },
                    Arc::clone(executor),
                );
                LashRuntime::from_persistent_background_state(
                    plan.policy.clone(),
                    host,
                    crate::PersistentRuntimeServices::new(plugins, Arc::clone(store)),
                    plan.initial_runtime_state.clone(),
                )
                .await
            }
            (Some(executor), None) => {
                let host = ProcessRuntimeHost::new(
                    EmbeddedRuntimeHost {
                        core: current.host.core.clone(),
                        session_store_factory: current.host.session_store_factory.clone(),
                    },
                    Arc::clone(executor),
                );
                LashRuntime::from_background_state(
                    plan.policy.clone(),
                    host,
                    RuntimeServices::new(plugins),
                    plan.initial_runtime_state.clone(),
                )
                .await
            }
            (None, Some(store)) => {
                let host = EmbeddedRuntimeHost {
                    core: current.host.core.clone(),
                    session_store_factory: current.host.session_store_factory.clone(),
                };
                LashRuntime::from_persistent_embedded_state(
                    plan.policy.clone(),
                    host,
                    crate::PersistentRuntimeServices::new(plugins, Arc::clone(store)),
                    plan.initial_runtime_state.clone(),
                )
                .await
            }
            (None, None) => {
                let host = EmbeddedRuntimeHost {
                    core: current.host.core.clone(),
                    session_store_factory: current.host.session_store_factory.clone(),
                };
                LashRuntime::from_embedded_state(
                    plan.policy.clone(),
                    host,
                    RuntimeServices::new(plugins),
                    plan.initial_runtime_state.clone(),
                )
                .await
            }
        }
        .map_err(|err| crate::PluginError::Session(err.to_string()))?;
        let protocol_session = runtime
            .session
            .as_ref()
            .map(|session| Arc::clone(session.plugins().protocol_session()));
        if let Some(protocol_session) = protocol_session {
            protocol_session
                .configure_runtime_from_request(
                    crate::plugin::ProtocolRuntimeContext::new(&mut runtime),
                    &SessionCreateRequest {
                        session_id: Some(plan.session_id.clone()),
                        relation: plan.relation.clone(),
                        start: SessionStartPoint::Snapshot {
                            snapshot: Box::new(plan.start_snapshot.clone()),
                        },
                        policy: Some(plan.policy.clone()),
                        plugin_source: plan.plugin_source,
                        initial_nodes: Vec::new(),
                        first_turn_input: plan.first_turn_input.clone(),
                        tool_access: plan.plugin_authority.tool_access.clone(),
                        subagent: plan.plugin_authority.subagent.clone(),
                        context_surface: plan.context_surface.clone(),
                        plugin_options: plan.plugin_options.clone(),
                        usage_source: plan.usage_source.clone(),
                    },
                )
                .map_err(|err| crate::PluginError::Session(err.to_string()))?;
        }
        if let Some(session) = runtime.session.as_mut() {
            session.set_context_surface(
                plan.context_surface.tool_providers.clone(),
                plan.context_surface.prompt_contributions.clone(),
                plan.context_surface.include_base_tools,
            );
        }
        Ok(runtime)
    }

    async fn register_materialized_session(
        &self,
        usage: &UsageCapability,
        plan: SessionCreatePlan,
        runtime: LashRuntime,
    ) -> Result<SessionHandle, crate::PluginError> {
        if let Some(store) = &plan.store_binding {
            let mut persisted_state = runtime.export_persisted_state();
            super::normalize_session_graph(&mut persisted_state);
            persisted_state.graph_replace_required = true;
            let commit = crate::store::RuntimeCommit::persisted_state(&persisted_state, &[]);
            let result = store
                .commit_runtime_state(commit)
                .await
                .map_err(|err| crate::PluginError::Session(err.to_string()))?;
            persisted_state.apply_persisted_commit_result(result);
        }
        self.registry
            .lock()
            .await
            .insert(plan.session_id.clone(), RuntimeHandle::new(runtime));
        if let crate::SessionRelation::Handoff {
            parent_session_id, ..
        } = &plan.relation
        {
            self.active_handoff_continuations
                .lock()
                .await
                .insert(parent_session_id.clone(), plan.session_id.clone());
        }
        if let Some(source) = &plan.usage_source {
            usage
                .child_sources
                .lock()
                .expect("child usage sources lock")
                .insert(plan.session_id.clone(), source.clone());
        }
        if let Some(seed) = plan.first_turn_input.clone() {
            self.pending_first_turn_inputs
                .lock()
                .expect("pending first turn inputs lock")
                .insert(plan.session_id.clone(), seed);
        }
        Ok(SessionHandle {
            session_id: plan.session_id,
            parent_session_id: plan.parent_session_id,
            policy: plan.policy,
        })
    }

    pub(in crate::runtime::session_manager) async fn create_session(
        &self,
        current: &CurrentSessionCapability,
        usage: &UsageCapability,
        request: SessionCreateRequest,
    ) -> Result<SessionHandle, crate::PluginError> {
        let plan = self.resolve_create_plan(current, request).await?;
        let runtime = self.materialize_create_plan(current, &plan).await?;
        self.register_materialized_session(usage, plan, runtime)
            .await
    }

    pub(in crate::runtime::session_manager) async fn close_session(
        &self,
        current: &CurrentSessionCapability,
        usage: &UsageCapability,
        session_id: &str,
    ) -> Result<(), crate::PluginError> {
        if session_id == current.session_id {
            return Err(crate::PluginError::Session(
                "cannot close the current session".to_string(),
            ));
        }
        if self
            .turns
            .lock()
            .await
            .values()
            .any(|turn| turn.session_id == session_id)
        {
            return Err(crate::PluginError::Session(format!(
                "cannot close session `{session_id}` while a turn is running"
            )));
        }
        self.registry.lock().await.remove(session_id);
        {
            let mut continuations = self.active_handoff_continuations.lock().await;
            continuations.remove(session_id);
            continuations.retain(|_, successor| successor != session_id);
        }
        usage
            .child_sources
            .lock()
            .expect("child usage sources lock")
            .remove(session_id);
        self.pending_first_turn_inputs
            .lock()
            .expect("pending first turn inputs lock")
            .remove(session_id);
        current.plugins.host().unregister_session(session_id)?;
        Ok(())
    }

    pub(in crate::runtime::session_manager) async fn take_first_turn_input(
        &self,
        session_id: &str,
    ) -> Result<Option<crate::PluginMessage>, crate::PluginError> {
        Ok(self
            .pending_first_turn_inputs
            .lock()
            .expect("pending first turn inputs lock")
            .remove(session_id))
    }

    pub(in crate::runtime::session_manager) async fn inject_turn_input(
        &self,
        session_id: &str,
        input: crate::InjectedTurnInput,
    ) -> Result<(), crate::PluginError> {
        let runtime_handle = {
            let registry = self.registry.lock().await;
            registry.get(session_id).cloned()
        };
        let Some(runtime_handle) = runtime_handle else {
            return Err(crate::PluginError::Session(format!(
                "unknown or inactive session `{session_id}` for turn input injection"
            )));
        };
        let runtime = runtime_handle.runtime.lock().await;
        let Some(session) = runtime.session.as_ref() else {
            return Err(crate::PluginError::Session(format!(
                "session `{session_id}` has no live turn-input bridge"
            )));
        };
        session
            .turn_input_injection_bridge()
            .enqueue(vec![input])
            .map_err(crate::PluginError::Session)
    }
}

fn child_store_guidance(parent_session_id: Option<&str>) -> String {
    let parent = parent_session_id
        .map(|id| format!(" for parent session `{id}`"))
        .unwrap_or_default();
    format!(
        "Managed child sessions require a store for the child session id{parent}. \
         Do not wrap a single pre-opened root store in LashCoreBuilder::store_factory; \
         pass root-only stores with SessionBuilder::store(...) and configure \
         LashCoreBuilder::child_store_factory(...) for managed children."
    )
}

fn child_store_factory_error(
    session_id: &str,
    parent_session_id: Option<&str>,
    message: String,
) -> String {
    format!(
        "failed to create store for child session `{session_id}`: {message}. {}",
        child_store_guidance(parent_session_id)
    )
}

async fn validate_child_store_binding(
    store: &dyn crate::RuntimePersistence,
    session_id: &str,
    parent_session_id: Option<&str>,
) -> Result<(), crate::PluginError> {
    let meta = store.load_session_meta().await.map_err(|err| {
        crate::PluginError::Session(format!(
            "failed to inspect store for child session `{session_id}`: {err}. {}",
            child_store_guidance(parent_session_id)
        ))
    })?;
    if let Some(meta) = meta
        && meta.session_id != session_id
    {
        return Err(crate::PluginError::Session(format!(
            "configured child session store is already bound to session `{}` and cannot be used for child session `{session_id}`. {}",
            meta.session_id,
            child_store_guidance(parent_session_id)
        )));
    }
    Ok(())
}
