use super::*;

impl LashRuntime {
    pub fn session_id(&self) -> &str {
        &self.state.session_id
    }

    pub(super) fn stamp_live_plugin_state(&mut self) {
        if let Some(session) = self.session.as_ref() {
            let snapshot = session.plugins().tool_registry().export_state();
            self.state.tool_state_generation = Some(snapshot.generation());
            self.state.tool_state_snapshot = Some(snapshot);
            self.state.plugin_snapshot = session.plugins().snapshot().ok();
            self.state.plugin_snapshot_revision =
                Some(session.plugins().snapshot_revision_fingerprint());
        } else {
            self.state.tool_state_generation = None;
            self.state.tool_state_snapshot = None;
            self.state.plugin_snapshot = None;
            self.state.plugin_snapshot_revision = None;
        }
    }
    pub(super) fn active_tool_catalog(&self) -> Result<Vec<serde_json::Value>, crate::PluginError> {
        self.active_tool_catalog_shared()
            .map(|catalog| catalog.as_ref().clone())
    }

    pub(super) fn active_tool_catalog_shared(
        &self,
    ) -> Result<Arc<Vec<serde_json::Value>>, crate::PluginError> {
        self.session
            .as_ref()
            .map(|session| {
                session
                    .shared_tool_catalog(&self.state.session_id, self.policy.execution_mode.clone())
            })
            .unwrap_or_else(|| Ok(Arc::new(Vec::new())))
    }

    pub fn tool_state(&self) -> Result<crate::ToolState, SessionError> {
        let Some(session) = self.session.as_ref() else {
            return Err(SessionError::Protocol(
                "runtime session not available".to_string(),
            ));
        };
        Ok(session.plugins().tool_registry().export_state())
    }
    /// Override mode-owned turn options for this session.
    pub fn set_mode_turn_options(&mut self, options: crate::ModeTurnOptions) {
        self.state.mode_turn_options = options.clone();
        self.mode_turn_options = options;
    }

    /// Export current session state for inspection/UI purposes.
    /// This keeps persistence-heavy snapshots untouched; callers that need a
    /// fully persisted view should use `export_persisted_state`.
    pub fn export_state(&self) -> SessionStateEnvelope {
        self.state.export_state()
    }

    pub fn read_view(&self) -> crate::SessionReadView {
        crate::SessionReadView::from_runtime_state(
            &self.state,
            self.policy.clone(),
            self.mode_turn_options.clone(),
        )
    }

    /// Export the narrow persistence snapshot used by stores and resume logic.
    pub fn export_persistence_state(&self) -> RuntimeSessionState {
        self.state.clone()
    }

    pub fn apply_persistence_state(&mut self, state: RuntimeSessionState) {
        self.set_persisted_state(state);
    }

    pub(crate) fn export_graph_first_state(&self) -> RuntimeSessionState {
        self.state.clone()
    }

    /// Export a persistence-ready state envelope with dynamic/plugin snapshots
    /// refreshed from the live session.
    pub fn export_persisted_state(&self) -> RuntimeSessionState {
        let mut state = self.state.clone();
        state.mode_turn_options = self.mode_turn_options.clone();
        if let Some(session) = self.session.as_ref() {
            let snapshot = session.plugins().tool_registry().export_state();
            state.tool_state_generation = Some(snapshot.generation());
            state.tool_state_snapshot = Some(snapshot);
            state.plugin_snapshot = session.plugins().snapshot().ok();
            state.plugin_snapshot_revision =
                Some(session.plugins().snapshot_revision_fingerprint());
        }
        normalize_session_graph(&mut state);
        state
    }

    pub fn usage_report(&self) -> SessionUsageReport {
        let mut entries = self.state.token_ledger.clone();
        let drained = self.shared_token_ledger.lock().expect("token ledger lock");
        for entry in drained.iter().cloned() {
            merge_ledger_entry(&mut entries, entry);
        }
        SessionUsageReport::from_entries(&entries)
    }

    pub async fn await_background_work(&mut self) -> Result<(), SessionError> {
        if self.process_sync_needed.swap(false, Ordering::AcqRel) {
            self.refresh_session_graph_from_store().await;
        }
        Ok(())
    }

    pub(super) async fn refresh_session_graph_from_store(&mut self) {
        let Some(store) = self
            .session
            .as_ref()
            .and_then(|session| session.history_store())
        else {
            return;
        };
        let scope = match self.residency {
            crate::Residency::KeepAll => crate::store::SessionReadScope::FullGraph,
            crate::Residency::ActivePathOnly => crate::store::SessionReadScope::ActivePath {
                leaf_node_id: self.state.session_graph.leaf_node_id.clone(),
            },
        };
        let read = match store.load_session(scope).await {
            Ok(Some(read)) => read,
            Ok(None) => return,
            Err(err) => {
                tracing::warn!("failed to refresh session graph from store: {err}");
                return;
            }
        };
        let has_newer_graph = self.state.head_revision != Some(read.head_revision)
            || read.graph.leaf_node_id != self.state.session_graph.leaf_node_id
            || read.checkpoint_ref != self.state.checkpoint_ref;
        if !has_newer_graph {
            return;
        }
        let head = crate::store::SessionHead {
            session_id: read.session_id.clone(),
            head_revision: read.head_revision,
            graph: read.graph,
            config: read.config.clone(),
            checkpoint_ref: read.checkpoint_ref.clone(),
            token_ledger: merge_usage_delta_entries(read.token_ledger),
        };
        apply_session_head(&mut self.state, &head);
        apply_session_checkpoint(&mut self.state, read.checkpoint);
    }

    pub(super) fn runtime_session_manager(
        &self,
    ) -> Result<Arc<RuntimeSessionManager>, PluginActionInvokeError> {
        Ok(Arc::new(RuntimeSessionManager::new(
            self, true, None, None,
        )?))
    }

    pub(super) fn runtime_session_manager_for_turn(
        &self,
        child_usage_event_relay: Option<ChildUsageEventRelay>,
    ) -> Result<Arc<RuntimeSessionManager>, PluginActionInvokeError> {
        self.runtime_session_manager_for_turn_with_lease(child_usage_event_relay, None)
    }

    pub(super) fn runtime_session_manager_for_turn_with_lease(
        &self,
        child_usage_event_relay: Option<ChildUsageEventRelay>,
        turn_lease: Option<crate::RuntimeTurnLease>,
    ) -> Result<Arc<RuntimeSessionManager>, PluginActionInvokeError> {
        Ok(Arc::new(RuntimeSessionManager::new(
            self,
            false,
            child_usage_event_relay,
            turn_lease,
        )?))
    }

    pub fn session_manager(&self) -> Result<Arc<dyn RuntimeSessionHost>, PluginActionInvokeError> {
        self.runtime_session_manager()
            .map(|manager| manager as Arc<dyn RuntimeSessionHost>)
    }

    /// The plugin session bound to the currently active runtime session, if any.
    pub fn plugin_session(&self) -> Option<Arc<crate::PluginSession>> {
        self.session.as_ref().map(|s| Arc::clone(s.plugins()))
    }

    pub fn turn_input_injection_bridge(
        &self,
    ) -> Result<crate::TurnInputInjectionBridge, SessionError> {
        let Some(session) = self.session.as_ref() else {
            return Err(SessionError::Protocol(
                "runtime session not available".to_string(),
            ));
        };
        Ok(session.turn_input_injection_bridge().clone())
    }

    /// Run the registered history rewrite pipeline against the current
    /// state, applying the resulting messages back onto the runtime.
    /// Returns true when at least one rewriter produced a summary or
    /// otherwise mutated the message list.
    pub async fn rewrite_history(
        &mut self,
        trigger: crate::RewriteTrigger,
    ) -> Result<bool, PluginActionInvokeError> {
        let manager = self.runtime_session_manager()?;
        let Some(plugin_session) = self.session.as_ref().map(|s| Arc::clone(s.plugins())) else {
            return Err(PluginActionInvokeError::Unknown(
                "runtime session not available".to_string(),
            ));
        };
        let ctx = crate::RewriteContext {
            session_id: self.state.session_id.clone(),
            trigger,
            state: self.read_view(),
            host: manager,
        };
        let input = crate::HistoryState::from_state(&self.state.export_state());
        let baseline_messages = input.messages.len();
        let outcome = plugin_session
            .rewrite_history(&ctx, input)
            .await
            .map_err(|err| {
                PluginActionInvokeError::Unknown(format!("rewrite_history failed: {err}"))
            })?;
        let mutated =
            outcome.metadata.produced_summary || outcome.messages.len() != baseline_messages;
        if mutated {
            self.state
                .replace_active_read_state(&outcome.messages, &outcome.tool_calls);
            if let Some(session) = self.session.as_ref() {
                self.state.tool_state_snapshot =
                    Some(session.plugins().tool_registry().export_state());
                self.state.plugin_snapshot = session.plugins().snapshot().ok();
                self.state.plugin_snapshot_revision =
                    Some(session.plugins().snapshot_revision_fingerprint());
            }
        }
        Ok(mutated)
    }

    pub(super) fn session_policy(&self) -> SessionPolicy {
        self.policy.clone()
    }

    pub(super) async fn notify_session_config_changed(&self, previous: SessionPolicy) {
        let Some(session) = self.session.as_ref() else {
            return;
        };
        let current = self.session_policy();
        if current == previous {
            return;
        }
        let Ok(host) = self.runtime_session_manager() else {
            return;
        };
        session
            .plugins()
            .emit_runtime_event(crate::PluginLifecycleEvent::SessionConfigChanged(Box::new(
                SessionConfigChangedContext {
                    session_id: self.state.session_id.clone(),
                    previous,
                    current,
                    host,
                },
            )))
            .await;
    }

    pub(super) async fn apply_session_config_mutations(&mut self, previous: SessionPolicy) {
        let Some(session) = self.session.as_ref() else {
            return;
        };
        let current = self.session_policy();
        if current == previous {
            return;
        }
        let Ok(host) = self.runtime_session_manager() else {
            return;
        };
        self.policy = session
            .plugins()
            .mutate_session_config(
                SessionConfigChangedContext {
                    session_id: self.state.session_id.clone(),
                    previous,
                    current,
                    host,
                },
                self.policy.clone(),
            )
            .await;
        self.state.policy = self.policy.clone();
    }
}
