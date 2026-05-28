use super::create_plan::{SessionCreatePlan, resolve_session_create_plan};
use super::materialize::{MaterializedSession, materialize_session_create_plan};
use super::*;

impl ManagedSessionCapability {
    async fn register_materialized_session(
        &self,
        usage: &UsageCapability,
        plan: SessionCreatePlan,
        materialized: MaterializedSession,
    ) -> Result<SessionHandle, crate::PluginError> {
        if let Some(store) = &materialized.store_binding {
            let mut persisted_state = materialized.runtime.export_persisted_state();
            super::normalize_session_graph(&mut persisted_state);
            persisted_state.graph_replace_required = true;
            let commit = crate::store::RuntimeCommit::persisted_state(&persisted_state, &[]);
            let result = store
                .commit_runtime_state(commit)
                .await
                .map_err(|err| crate::PluginError::Session(err.to_string()))?;
            persisted_state.apply_persisted_commit_result(result);
        }
        self.registry.lock().await.insert(
            plan.session_id.clone(),
            RuntimeHandle::new(materialized.runtime),
        );
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
        let plan = resolve_session_create_plan(self, current, request).await?;
        let materialized = materialize_session_create_plan(current, &plan).await?;
        self.register_materialized_session(usage, plan, materialized)
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

}
