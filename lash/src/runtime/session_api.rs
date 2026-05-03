use super::*;

impl LashRuntime {
    pub fn session_id(&self) -> &str {
        &self.state.session_id
    }

    pub(super) fn stamp_live_plugin_state(&mut self) {
        if let Some(session) = self.session.as_ref() {
            if let Some(dynamic_tools) = session.plugins().dynamic_tools() {
                let snapshot = dynamic_tools.export_state();
                self.state.dynamic_state_generation = Some(snapshot.base_generation);
                self.state.dynamic_state_snapshot = Some(snapshot);
            } else {
                self.state.dynamic_state_generation = None;
                self.state.dynamic_state_snapshot = None;
            }
            self.state.plugin_snapshot = session.plugins().snapshot().ok();
            self.state.plugin_snapshot_revision =
                Some(session.plugins().snapshot_revision_fingerprint());
        } else {
            self.state.dynamic_state_generation = None;
            self.state.dynamic_state_snapshot = None;
            self.state.plugin_snapshot = None;
            self.state.plugin_snapshot_revision = None;
        }
    }
    pub(super) fn active_tool_catalog(&self) -> Vec<serde_json::Value> {
        self.active_tool_catalog_shared().as_ref().clone()
    }

    pub(super) fn active_tool_catalog_shared(&self) -> Arc<Vec<serde_json::Value>> {
        self.session
            .as_ref()
            .map(|session| {
                session
                    .shared_tool_catalog(&self.state.session_id, self.policy.execution_mode.clone())
            })
            .unwrap_or_else(|| Arc::new(Vec::new()))
    }

    pub fn dynamic_tool_state(&self) -> Result<crate::DynamicStateSnapshot, SessionError> {
        let Some(session) = self.session.as_ref() else {
            return Err(SessionError::Protocol(
                "runtime session not available".to_string(),
            ));
        };
        let Some(dynamic_tools) = session.plugins().dynamic_tools() else {
            return Err(SessionError::Protocol(
                "dynamic tools are unavailable in this runtime session".to_string(),
            ));
        };
        Ok(dynamic_tools.export_state())
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
    pub fn export_persistence_state(&self) -> PersistedSessionState {
        self.state.clone()
    }

    pub fn apply_persistence_state(&mut self, state: PersistedSessionState) {
        self.set_persisted_state(state);
    }

    pub(crate) fn export_graph_first_state(&self) -> PersistedSessionState {
        self.state.clone()
    }

    /// Export a persistence-ready state envelope with dynamic/plugin snapshots
    /// refreshed from the live session.
    pub fn export_persisted_state(&self) -> PersistedSessionState {
        let mut state = self.state.clone();
        state.mode_turn_options = self.mode_turn_options.clone();
        if let Some(session) = self.session.as_ref() {
            if let Some(dynamic_tools) = session.plugins().dynamic_tools() {
                let snapshot = dynamic_tools.export_state();
                state.dynamic_state_generation = Some(snapshot.base_generation);
                state.dynamic_state_snapshot = Some(snapshot);
            } else {
                state.dynamic_state_generation = None;
                state.dynamic_state_snapshot = None;
            }
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
        let manager = self
            .runtime_session_manager()
            .map_err(|err| SessionError::Protocol(err.to_string()))?;
        manager
            .await_hidden_tasks(&self.state.session_id)
            .await
            .map_err(|err| SessionError::Protocol(format!("session task failed: {err}")))?;
        if self.background_sync_needed.swap(false, Ordering::AcqRel) {
            self.refresh_session_graph_from_store().await;
        }
        self.refresh_session_tool_surface().await?;
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
        let Some(head_meta) = store.load_session_head_meta().await else {
            return;
        };
        let has_newer_graph = head_meta.graph_node_count > self.state.persisted_graph_node_count
            || head_meta.leaf_node_id != self.state.session_graph.leaf_node_id
            || head_meta.checkpoint_ref != self.state.checkpoint_ref;
        if !has_newer_graph {
            return;
        }
        let mut graph = store.load_session_graph().await;
        graph.set_leaf_node_id(head_meta.leaf_node_id.clone());
        let head = crate::store::SessionHead {
            session_id: head_meta.session_id.clone(),
            graph,
            config: head_meta.config.clone(),
            checkpoint_ref: head_meta.checkpoint_ref.clone(),
            token_ledger: merge_usage_delta_entries(store.load_usage_deltas().await),
        };
        apply_session_head(&mut self.state, &head);
        let checkpoint =
            load_session_checkpoint(store.as_ref(), head.checkpoint_ref.as_ref()).await;
        apply_session_checkpoint(&mut self.state, checkpoint);
    }

    pub(super) fn runtime_session_manager(
        &self,
    ) -> Result<Arc<dyn RuntimeSessionHost>, ExternalInvokeError> {
        self.runtime_session_manager_with_prompt_bridge(None)
    }

    pub(super) fn runtime_session_manager_for_turn(
        &self,
        prompt_bridge: Option<HostPromptBridge>,
        child_usage_event_relay: Option<ChildUsageEventRelay>,
    ) -> Result<Arc<dyn RuntimeSessionHost>, ExternalInvokeError> {
        Ok(Arc::new(RuntimeSessionManager::new(
            self,
            prompt_bridge,
            false,
            child_usage_event_relay,
        )?))
    }

    pub(super) fn runtime_session_manager_with_prompt_bridge(
        &self,
        prompt_bridge: Option<HostPromptBridge>,
    ) -> Result<Arc<dyn RuntimeSessionHost>, ExternalInvokeError> {
        Ok(Arc::new(RuntimeSessionManager::new(
            self,
            prompt_bridge,
            true,
            None,
        )?))
    }

    pub fn session_manager(&self) -> Result<Arc<dyn RuntimeSessionHost>, ExternalInvokeError> {
        self.runtime_session_manager()
    }

    /// The plugin session bound to the currently active runtime session, if any.
    pub fn plugin_session(&self) -> Option<Arc<crate::PluginSession>> {
        self.session.as_ref().map(|s| Arc::clone(s.plugins()))
    }

    /// Run the registered history rewrite pipeline against the current
    /// state, applying the resulting messages back onto the runtime.
    /// Returns true when at least one rewriter produced a summary or
    /// otherwise mutated the message list.
    pub async fn rewrite_history(
        &mut self,
        trigger: crate::RewriteTrigger,
    ) -> Result<bool, ExternalInvokeError> {
        let manager = self.runtime_session_manager()?;
        let Some(plugin_session) = self.session.as_ref().map(|s| Arc::clone(s.plugins())) else {
            return Err(ExternalInvokeError::Unknown(
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
                ExternalInvokeError::Unknown(format!("rewrite_history failed: {err}"))
            })?;
        let mutated =
            outcome.metadata.produced_summary || outcome.messages.len() != baseline_messages;
        if mutated {
            self.state
                .replace_active_read_state(&outcome.messages, &outcome.tool_calls);
            if let Some(session) = self.session.as_ref() {
                self.state.dynamic_state_snapshot = session
                    .plugins()
                    .dynamic_tools()
                    .map(|tools| tools.export_state());
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
            .emit_runtime_event(crate::PluginRuntimeEvent::SessionConfigChanged(Box::new(
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
