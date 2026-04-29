use super::*;

impl LashRuntime {
    pub(super) async fn from_host_state(
        policy: SessionPolicy,
        host: RuntimeHost,
        services: RuntimeServices,
        mut state: PersistedSessionState,
    ) -> Result<Self, SessionError> {
        if state.session_id.is_empty() {
            state.session_id = uuid::Uuid::new_v4().to_string();
        }
        // Defaulted state (e.g. `PersistedSessionState::default()` used
        // by fresh-session constructors) carries an unconfigured policy.
        // Fill it in from the caller's policy so tests and hosts that
        // pass a real policy alongside default state don't trip the
        // max_context_tokens guard below.
        if state.policy.provider.kind() == "unconfigured" {
            state.policy = policy.clone();
        }
        normalize_session_graph(&mut state);
        if policy.max_context_tokens.is_none() {
            return Err(SessionError::Protocol(
                "session policy missing max_context_tokens; hosts must supply explicit model metadata"
                    .to_string(),
            ));
        }
        let mut session = Session::new(
            services.clone(),
            &state.session_id,
            state.policy.execution_mode.clone(),
        )
        .await?;
        if let Some(dynamic_state) = state.dynamic_state_snapshot.clone()
            && let Some(dynamic_tools) = session.plugins().dynamic_tools()
            && let Err(err) = dynamic_tools.apply_state(dynamic_state)
        {
            tracing::warn!("failed to restore dynamic tool state from checkpoint: {err}");
        }
        if let Some(snapshot) = state.plugin_snapshot.clone() {
            session
                .plugins()
                .restore(&snapshot)
                .map_err(|err| SessionError::Protocol(err.to_string()))?;
        }
        let mode_session = Arc::clone(session.plugins().mode_session());
        let session_id = state.session_id.clone();
        mode_session
            .restore_session(
                crate::plugin::ModeSessionContext::new(&mut session, &session_id),
                &state,
            )
            .await?;
        clear_persisted_runtime_caches(&mut state);
        session
            .plugins()
            .emit_runtime_event(crate::PluginRuntimeEvent::SessionRestored(
                state.read_view(),
            ))
            .await;
        let mode_turn_options = state.mode_turn_options.clone();
        Ok(Self {
            session: Some(session),
            policy,
            host,
            services,
            state,
            runtime_scope_id: Arc::<str>::from(uuid::Uuid::new_v4().to_string()),
            managed_sessions: Arc::new(Mutex::new(HashMap::new())),
            managed_turns: Arc::new(Mutex::new(HashMap::new())),
            overflow_recovery_attempted: false,
            mode_turn_options,
            shared_token_ledger: Arc::new(std::sync::Mutex::new(Vec::new())),
            background_sync_needed: Arc::new(AtomicBool::new(false)),
            pending_first_turn_inputs: Arc::new(std::sync::Mutex::new(HashMap::new())),
            turn_phase_probe: None,
        })
    }

    /// Build a runtime for an embedded host with no background worker support.
    pub async fn from_embedded_state(
        policy: SessionPolicy,
        host: EmbeddedRuntimeHost,
        services: RuntimeServices,
        state: PersistedSessionState,
    ) -> Result<Self, SessionError> {
        Self::from_host_state(policy, host.into(), services, state).await
    }

    /// Build a runtime for a host that supports background plugin work.
    pub async fn from_background_state(
        policy: SessionPolicy,
        host: BackgroundRuntimeHost,
        services: RuntimeServices,
        state: PersistedSessionState,
    ) -> Result<Self, SessionError> {
        Self::from_host_state(policy, host.into(), services, state).await
    }

    /// Build a runtime for an embedded host with persistent store support.
    pub async fn from_persistent_embedded_state(
        policy: SessionPolicy,
        host: EmbeddedRuntimeHost,
        services: PersistentRuntimeServices,
        state: PersistedSessionState,
    ) -> Result<Self, SessionError> {
        Self::from_host_state(policy, host.into(), services.into_runtime_services(), state).await
    }

    /// Build a runtime for a background-capable host with persistent store support.
    pub async fn from_persistent_background_state(
        policy: SessionPolicy,
        host: BackgroundRuntimeHost,
        services: PersistentRuntimeServices,
        state: PersistedSessionState,
    ) -> Result<Self, SessionError> {
        Self::from_host_state(policy, host.into(), services.into_runtime_services(), state).await
    }

    /// Embedder-preferred constructor: build a `LashRuntime` from a
    /// shared `RuntimeEnvironment`.
    ///
    /// Everything expensive (plugin factories, HTTP client pool, prompt
    /// template, path resolver) lives on the environment and is
    /// reused across every runtime the embedder builds. This call is
    /// O(plugin-session-registration + state-hydration), not
    /// O(full-infrastructure-init).
    ///
    /// * `env` — the shared environment. `env.plugin_host` must be set.
    /// * `policy` — per-session policy (model, provider, execution mode).
    /// * `state` — persisted session state (empty for a fresh session).
    /// * `store` — per-session store. `None` builds an embedded runtime
    ///   with no persistence; `Some` builds a persistent
    ///   background-capable runtime.
    pub async fn from_environment(
        env: &RuntimeEnvironment,
        policy: SessionPolicy,
        mut state: PersistedSessionState,
        store: Option<Arc<dyn crate::store::RuntimePersistence>>,
    ) -> Result<Self, SessionError> {
        // ActivePathOnly without a store is a data-loss footgun: trim
        // drops orphans from RAM with nowhere to reload them from.
        if matches!(env.residency, Residency::ActivePathOnly) && store.is_none() {
            return Err(SessionError::Protocol(
                "Residency::ActivePathOnly requires a persistent store — \
                 without one, trimmed orphans are irrecoverable"
                    .to_string(),
            ));
        }
        // Heal FIRST (against the full resident set), then trim.
        // `heal_orphaned_leaf` is driven by `normalize_session_graph`
        // which runs again inside `from_host_state`. Running it here
        // too lets us trim safely before delegating.
        normalize_session_graph(&mut state);
        apply_residency_on_load(&mut state, env.residency);
        let plugin_host = env.plugin_host.as_ref().ok_or_else(|| {
            SessionError::Protocol(
                "RuntimeEnvironment.plugin_host is required for from_environment".to_string(),
            )
        })?;
        let plugin_session = plugin_host
            .build_session(
                state.session_id.as_str(),
                policy.execution_mode.clone(),
                policy.context_approach.clone(),
                state.plugin_snapshot.as_ref(),
            )
            .map_err(|err| SessionError::Protocol(err.to_string()))?;
        let core = env.to_runtime_core_config();
        let embedded = EmbeddedRuntimeHost::new(core);
        let runtime = if let Some(store) = store {
            let services = PersistentRuntimeServices::new_with_bridges(
                plugin_session,
                crate::session::TurnInjectionBridge::new(),
                crate::session::TurnInputInjectionBridge::new(),
                store,
            );
            match env.session_task_executor.as_ref() {
                Some(executor) => {
                    let host = BackgroundRuntimeHost::new(embedded, Arc::clone(executor));
                    Self::from_persistent_background_state(policy, host, services, state).await?
                }
                None => {
                    Self::from_persistent_embedded_state(policy, embedded, services, state).await?
                }
            }
        } else {
            let services = RuntimeServices::new(plugin_session);
            match env.session_task_executor.as_ref() {
                Some(executor) => {
                    let host = BackgroundRuntimeHost::new(embedded, Arc::clone(executor));
                    Self::from_background_state(policy, host, services, state).await?
                }
                None => Self::from_embedded_state(policy, embedded, services, state).await?,
            }
        };
        Ok(runtime)
    }

    /// Persist any dirty state and drop the runtime, returning a lightweight
    /// handle the embedder can cache and resume later via
    /// [`LashRuntime::resume`]. This is the webserver-embedder handoff
    /// primitive: the handle holds only the session id, policy, and store
    /// reference — no graph nodes, no plugin session, no HTTP client.
    pub async fn park(mut self) -> Result<ParkedSession, SessionError> {
        let store = self.services.store.clone().ok_or_else(|| {
            SessionError::Protocol(
                "park() requires a persistent runtime (store is not set)".to_string(),
            )
        })?;
        let session_id = self.state.session_id.clone();
        let policy = self.policy.clone();
        // Flush any dirty resident state to the store before dropping.
        persist_runtime_state(store.as_ref(), &mut self.state).await;
        // Drain pending tombstones if any. Under KeepHistory this is a
        // no-op (tombstones never get added). Under DropOrphans,
        // Phase-9's not-yet-wired rewrite path would have populated the
        // set — wired fully in Phase 10's vacuum() design.
        Ok(ParkedSession {
            session_id,
            store,
            policy,
        })
    }

    /// Resume a previously parked session against a shared environment.
    /// Loads only the active-path graph when
    /// `env.residency == ActivePathOnly`; under `KeepAll`
    /// loads the full graph (current behavior).
    pub async fn resume(
        parked: ParkedSession,
        env: &RuntimeEnvironment,
    ) -> Result<Self, SessionError> {
        // Under ActivePathOnly, skip the full-graph load: fetch head
        // metadata + the active-path chain only. SQLite impls can
        // override `load_active_path_graph` with a recursive CTE for a
        // real O(active-path) query; the default still loads the full
        // graph then forks, which is correct but slower on large
        // histories.
        let loaded = match env.residency {
            Residency::KeepAll => {
                crate::store::load_persisted_session_state(parked.store.as_ref()).await
            }
            Residency::ActivePathOnly => {
                crate::store::load_persisted_session_state_active_path(parked.store.as_ref()).await
            }
        };
        let state = loaded.unwrap_or_else(|| PersistedSessionState {
            session_id: parked.session_id.clone(),
            policy: parked.policy.clone(),
            ..PersistedSessionState::default()
        });
        Self::from_environment(env, parked.policy, state, Some(parked.store)).await
    }

    /// Opt-in async read for historic (non-active-path) nodes under
    /// `Residency::ActivePathOnly`. Plugins that walk the full graph
    /// call this instead of `session_graph().find_node()` so missing
    /// nodes surface as `Ok(None)` rather than silently missing.
    pub async fn get_historic_node(
        &self,
        node_id: &str,
    ) -> Result<Option<crate::SessionNodeRecord>, SessionError> {
        if let Some(node) = self.state.session_graph.find_node(node_id) {
            return Ok(Some(node.clone()));
        }
        let store = self.services.store.clone().ok_or_else(|| {
            SessionError::Protocol("get_historic_node() requires a persistent runtime".to_string())
        })?;
        Ok(store.get_node(node_id).await)
    }

    /// Store-resident node IDs that are NOT reachable from the current
    /// leaf — i.e. orphans eligible for tombstoning. lash owns RAM; the
    /// host owns disk lifecycle, so this is a primitive the host calls
    /// on its own schedule (e.g. every N turns, or off-peak).
    ///
    /// Typical autonomous-agent loop:
    ///
    /// ```ignore
    /// let orphans = runtime.orphaned_node_ids().await?;
    /// if !orphans.is_empty() {
    ///     store.tombstone_nodes(&orphans).await;
    /// }
    /// // And less often:
    /// store.vacuum().await;
    /// ```
    pub async fn orphaned_node_ids(&self) -> Result<Vec<String>, SessionError> {
        let store = self.services.store.clone().ok_or_else(|| {
            SessionError::Protocol("orphaned_node_ids() requires a persistent runtime".to_string())
        })?;
        let active: std::collections::HashSet<&str> = self
            .state
            .session_graph
            .active_path_nodes()
            .iter()
            .map(|node| node.node_id.as_str())
            .collect();
        let full = store.load_session_graph().await;
        Ok(full
            .nodes
            .iter()
            .filter(|node| !active.contains(node.node_id.as_str()))
            .map(|node| node.node_id.clone())
            .collect())
    }
}
