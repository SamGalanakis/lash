use super::*;

pub(in crate::runtime::session_manager) struct SessionCreatePlan {
    pub(in crate::runtime::session_manager) session_id: String,
    pub(in crate::runtime::session_manager) relation: SessionRelation,
    pub(in crate::runtime::session_manager) parent_session_id: Option<String>,
    pub(in crate::runtime::session_manager) policy: SessionPolicy,
    pub(in crate::runtime::session_manager) initial_runtime_state: SessionSnapshot,
    pub(in crate::runtime::session_manager) plugin_authority:
        crate::plugin::SessionAuthorityContext,
    pub(in crate::runtime::session_manager) plugin_source: crate::SessionPluginSource,
    pub(in crate::runtime::session_manager) context_surface: crate::SessionContextSurface,
    pub(in crate::runtime::session_manager) protocol_request: SessionCreateRequest,
    pub(in crate::runtime::session_manager) usage_source: Option<String>,
    pub(in crate::runtime::session_manager) first_turn_input: Option<crate::PluginMessage>,
}

pub(in crate::runtime::session_manager) async fn resolve_session_create_plan(
    managed: &ManagedSessionCapability,
    current: &CurrentSessionCapability,
    mut request: SessionCreateRequest,
) -> Result<SessionCreatePlan, crate::PluginError> {
    let session_id = request
        .session_id
        .take()
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
    request.session_id = Some(session_id.clone());
    if session_id == current.session_id || managed.registry.lock().await.contains_key(&session_id) {
        return Err(crate::PluginError::Session(format!(
            "session `{session_id}` already exists"
        )));
    }

    let parent_session_id = request.relation.parent_session_id().map(ToOwned::to_owned);
    let start_snapshot = resolve_start_snapshot(managed, current, &request, &session_id).await?;
    let policy = resolve_session_policy(current, &request, &start_snapshot, &session_id);
    request.policy = Some(policy.clone());
    let initial_runtime_state =
        build_runtime_state(session_id.clone(), &request, start_snapshot, &policy);
    let plugin_authority = crate::plugin::SessionAuthorityContext {
        tool_access: request.tool_access.clone(),
        subagent: request.subagent.clone(),
    };

    Ok(SessionCreatePlan {
        session_id,
        relation: request.relation.clone(),
        parent_session_id,
        policy,
        initial_runtime_state,
        plugin_authority,
        plugin_source: request.plugin_source,
        context_surface: request.context_surface.clone(),
        usage_source: request.usage_source.clone(),
        first_turn_input: request.first_turn_input.clone(),
        protocol_request: request,
    })
}

async fn resolve_start_snapshot(
    managed: &ManagedSessionCapability,
    current: &CurrentSessionCapability,
    request: &SessionCreateRequest,
    session_id: &str,
) -> Result<SessionSnapshot, crate::PluginError> {
    match &request.start {
        SessionStartPoint::Empty => Ok(SessionSnapshot {
            session_id: session_id.to_string(),
            ..Default::default()
        }),
        SessionStartPoint::CurrentSession => Ok(current.snapshot.to_snapshot()),
        SessionStartPoint::ExistingSession { session_id } => {
            current.snapshot_by_id(managed, session_id).await
        }
        SessionStartPoint::Snapshot { snapshot } => Ok((**snapshot).clone()),
    }
}

fn resolve_session_policy(
    current: &CurrentSessionCapability,
    request: &SessionCreateRequest,
    start_snapshot: &SessionSnapshot,
    session_id: &str,
) -> SessionPolicy {
    let mut policy = request
        .policy
        .clone()
        .unwrap_or_else(|| match &request.start {
            SessionStartPoint::Empty => current.policy.clone(),
            _ => start_snapshot.policy.clone(),
        });
    if request.relation.parent_session_id().is_some() {
        policy.session_id = Some(session_id.to_string());
    }
    policy
}

fn build_runtime_state(
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
