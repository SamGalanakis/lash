use super::create_plan::SessionCreatePlan;
use super::*;
use crate::runtime::host::{EmbeddedRuntimeHost, ProcessRuntimeHost};

pub(in crate::runtime::session_manager) struct MaterializedSession {
    pub(in crate::runtime::session_manager) runtime: LashRuntime,
    pub(in crate::runtime::session_manager) store_binding:
        Option<Arc<dyn crate::store::RuntimePersistence>>,
}

enum RuntimeMaterializationBinding {
    ProcessPersistent {
        host: ProcessRuntimeHost,
        services: crate::PersistentRuntimeServices,
    },
    ProcessEphemeral {
        host: ProcessRuntimeHost,
        services: RuntimeServices,
    },
    EmbeddedPersistent {
        host: EmbeddedRuntimeHost,
        services: crate::PersistentRuntimeServices,
    },
    EmbeddedEphemeral {
        host: EmbeddedRuntimeHost,
        services: RuntimeServices,
    },
}

impl RuntimeMaterializationBinding {
    fn resolve(
        current: &CurrentSessionCapability,
        plugins: Arc<crate::PluginSession>,
        store_binding: Option<Arc<dyn crate::store::RuntimePersistence>>,
    ) -> Self {
        let embedded = embedded_host(current);
        match (current.host.process_registry.clone(), store_binding) {
            (Some(process_registry), Some(store)) => Self::ProcessPersistent {
                host: ProcessRuntimeHost::new(embedded, process_registry),
                services: crate::PersistentRuntimeServices::new(plugins, store),
            },
            (Some(process_registry), None) => Self::ProcessEphemeral {
                host: ProcessRuntimeHost::new(embedded, process_registry),
                services: RuntimeServices::new(plugins),
            },
            (None, Some(store)) => Self::EmbeddedPersistent {
                host: embedded,
                services: crate::PersistentRuntimeServices::new(plugins, store),
            },
            (None, None) => Self::EmbeddedEphemeral {
                host: embedded,
                services: RuntimeServices::new(plugins),
            },
        }
    }

    async fn materialize(
        self,
        policy: SessionPolicy,
        initial_runtime_state: SessionSnapshot,
    ) -> Result<LashRuntime, SessionError> {
        match self {
            Self::ProcessPersistent { host, services } => {
                LashRuntime::from_persistent_background_state(
                    policy,
                    host,
                    services,
                    initial_runtime_state,
                )
                .await
            }
            Self::ProcessEphemeral { host, services } => {
                LashRuntime::from_background_state(policy, host, services, initial_runtime_state)
                    .await
            }
            Self::EmbeddedPersistent { host, services } => {
                LashRuntime::from_persistent_embedded_state(
                    policy,
                    host,
                    services,
                    initial_runtime_state,
                )
                .await
            }
            Self::EmbeddedEphemeral { host, services } => {
                LashRuntime::from_embedded_state(policy, host, services, initial_runtime_state)
                    .await
            }
        }
    }
}

pub(in crate::runtime::session_manager) async fn materialize_session_create_plan(
    current: &CurrentSessionCapability,
    plan: &SessionCreatePlan,
) -> Result<MaterializedSession, crate::PluginError> {
    let plugins = build_session_plugins(current, plan)?;
    let store_binding = bind_session_store(current, plan).await?;
    let binding = RuntimeMaterializationBinding::resolve(current, plugins, store_binding.clone());
    let mut runtime = binding
        .materialize(plan.policy.clone(), plan.initial_runtime_state.clone())
        .await
        .map_err(|err| crate::PluginError::Session(err.to_string()))?;

    configure_protocol_runtime(&mut runtime, &plan.protocol_request)?;
    if let Some(session) = runtime.session.as_mut() {
        session.set_context_surface(
            plan.context_surface.tool_providers.clone(),
            plan.context_surface.prompt_contributions.clone(),
            plan.context_surface.include_base_tools,
        )?;
    }

    Ok(MaterializedSession {
        runtime,
        store_binding,
    })
}

fn build_session_plugins(
    current: &CurrentSessionCapability,
    plan: &SessionCreatePlan,
) -> Result<Arc<crate::PluginSession>, crate::PluginError> {
    match plan.plugin_source {
        crate::SessionPluginSource::CurrentHostFresh => current
            .plugins
            .host()
            .build_session_with_parent(
                &plan.session_id,
                plan.parent_session_id.clone(),
                None,
                plan.plugin_authority.clone(),
            )
            .map_err(|err| crate::PluginError::Session(err.to_string())),
        crate::SessionPluginSource::CurrentSessionFork => current
            .plugins
            .fork_for_child_session(
                &plan.session_id,
                plan.parent_session_id.clone(),
                plan.plugin_authority.clone(),
            )
            .map_err(|err| crate::PluginError::Session(err.to_string())),
    }
}

async fn bind_session_store(
    current: &CurrentSessionCapability,
    plan: &SessionCreatePlan,
) -> Result<Option<Arc<dyn crate::store::RuntimePersistence>>, crate::PluginError> {
    let Some(factory) = &current.host.session_store_factory else {
        return Ok(None);
    };
    let store = factory
        .create_store(&SessionStoreCreateRequest {
            session_id: plan.session_id.clone(),
            relation: plan.relation.clone(),
            policy: plan.policy.clone(),
        })
        .map_err(|message| {
            crate::PluginError::Session(child_store_factory_error(
                &plan.session_id,
                plan.parent_session_id.as_deref(),
                message,
            ))
        })?;
    validate_child_store_binding(
        store.as_ref(),
        &plan.session_id,
        plan.parent_session_id.as_deref(),
    )
    .await?;
    Ok(Some(store))
}

fn embedded_host(current: &CurrentSessionCapability) -> EmbeddedRuntimeHost {
    EmbeddedRuntimeHost {
        core: current.host.core.clone(),
        session_store_factory: current.host.session_store_factory.clone(),
    }
}

fn configure_protocol_runtime(
    runtime: &mut LashRuntime,
    request: &SessionCreateRequest,
) -> Result<(), crate::PluginError> {
    let protocol_session = runtime
        .session
        .as_ref()
        .map(|session| Arc::clone(session.plugins().protocol_session()));
    if let Some(protocol_session) = protocol_session {
        protocol_session
            .configure_runtime_from_request(
                crate::plugin::ProtocolRuntimeContext::new(runtime),
                request,
            )
            .map_err(|err| crate::PluginError::Session(err.to_string()))?;
    }
    Ok(())
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
