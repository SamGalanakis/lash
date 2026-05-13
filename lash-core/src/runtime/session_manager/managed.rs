use super::*;
use crate::runtime::host::{BackgroundRuntimeHost, EmbeddedRuntimeHost};

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

    pub(in crate::runtime::session_manager) async fn create_session(
        &self,
        current: &CurrentSessionCapability,
        usage: &UsageCapability,
        request: SessionCreateRequest,
    ) -> Result<SessionHandle, crate::PluginError> {
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
        let snapshot = match &request.start {
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
                _ => snapshot.policy.clone(),
            });
        if parent_session_id.is_some() {
            policy.session_id = Some(session_id.clone());
        }
        let state = self.build_runtime_state(session_id.clone(), &request, snapshot, &policy);
        let authority = crate::plugin::SessionAuthorityContext {
            tool_access: request.tool_access.clone(),
            subagent: request.subagent.clone(),
        };
        let plugins = match request.plugin_mode {
            crate::SessionPluginMode::Fresh => current
                .plugins
                .host()
                .build_session_with_parent(
                    &session_id,
                    parent_session_id.clone(),
                    policy.execution_mode.clone(),
                    policy.standard_context_approach.clone(),
                    None,
                    authority,
                )
                .map_err(|err| crate::PluginError::Session(err.to_string()))?,
            crate::SessionPluginMode::InheritCurrent => current
                .plugins
                .fork_for_child_session(
                    &session_id,
                    parent_session_id.clone(),
                    policy.execution_mode.clone(),
                    policy.standard_context_approach.clone(),
                    authority,
                )
                .map_err(|err| crate::PluginError::Session(err.to_string()))?,
        };
        let session_store = match &current.host.session_store_factory {
            Some(factory) => {
                let store = factory
                    .create_store(&SessionStoreCreateRequest {
                        session_id: session_id.clone(),
                        parent_session_id: parent_session_id.clone(),
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
        let mut runtime = match (&current.host.session_task_executor, &session_store) {
            (Some(executor), Some(store)) => {
                let host = BackgroundRuntimeHost::new(
                    EmbeddedRuntimeHost {
                        core: current.host.core.clone(),
                        session_store_factory: current.host.session_store_factory.clone(),
                    },
                    Arc::clone(executor),
                );
                LashRuntime::from_persistent_background_state(
                    policy.clone(),
                    host,
                    crate::PersistentRuntimeServices::new(plugins, Arc::clone(store)),
                    state,
                )
                .await
            }
            (Some(executor), None) => {
                let host = BackgroundRuntimeHost::new(
                    EmbeddedRuntimeHost {
                        core: current.host.core.clone(),
                        session_store_factory: current.host.session_store_factory.clone(),
                    },
                    Arc::clone(executor),
                );
                LashRuntime::from_background_state(
                    policy.clone(),
                    host,
                    RuntimeServices::new(plugins),
                    state,
                )
                .await
            }
            (None, Some(store)) => {
                let host = EmbeddedRuntimeHost {
                    core: current.host.core.clone(),
                    session_store_factory: current.host.session_store_factory.clone(),
                };
                LashRuntime::from_persistent_embedded_state(
                    policy.clone(),
                    host,
                    crate::PersistentRuntimeServices::new(plugins, Arc::clone(store)),
                    state,
                )
                .await
            }
            (None, None) => {
                let host = EmbeddedRuntimeHost {
                    core: current.host.core.clone(),
                    session_store_factory: current.host.session_store_factory.clone(),
                };
                LashRuntime::from_embedded_state(
                    policy.clone(),
                    host,
                    RuntimeServices::new(plugins),
                    state,
                )
                .await
            }
        }
        .map_err(|err| crate::PluginError::Session(err.to_string()))?;
        let mode_session = runtime
            .session
            .as_ref()
            .map(|session| Arc::clone(session.plugins().mode_session()));
        if let Some(mode_session) = mode_session {
            mode_session.configure_runtime_from_request(
                crate::plugin::ModeRuntimeContext::new(&mut runtime),
                &request,
            );
        }
        if let Some(session) = runtime.session.as_mut() {
            session.set_context_surface(
                request.context_surface.tool_providers.clone(),
                request.context_surface.prompt_contributions.clone(),
                request.context_surface.include_base_tools,
            );
        }
        if let Some(store) = &session_store {
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
            .insert(session_id.clone(), RuntimeHandle::new(runtime));
        if let crate::SessionRelation::Handoff {
            parent_session_id, ..
        } = &request.relation
        {
            self.active_handoff_continuations
                .lock()
                .await
                .insert(parent_session_id.clone(), session_id.clone());
        }
        if let Some(source) = &request.usage_source {
            usage
                .child_sources
                .lock()
                .expect("child usage sources lock")
                .insert(session_id.clone(), source.clone());
        }
        if let Some(seed) = request.first_turn_input.clone() {
            self.pending_first_turn_inputs
                .lock()
                .expect("pending first turn inputs lock")
                .insert(session_id.clone(), seed);
        }
        Ok(SessionHandle {
            session_id,
            parent_session_id,
            policy,
        })
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
