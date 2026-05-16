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
        let services = services.with_attachment_store(Arc::clone(&host.core.attachment_store));
        let mut session = Session::new(
            services.clone(),
            &state.session_id,
            state.policy.execution_mode.clone(),
        )
        .await?;
        if let Some(tool_state) = state.tool_state_snapshot.clone()
            && let Err(err) = session.plugins().tool_registry().apply_state(tool_state)
        {
            tracing::warn!("failed to restore tool state from checkpoint: {err}");
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
        state.discard_runtime_snapshots();
        session
            .plugins()
            .emit_runtime_event(crate::PluginRuntimeEvent::SessionRestored(
                crate::SessionReadView::from_persisted_state(&state),
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
            active_handoff_continuations: Arc::new(Mutex::new(HashMap::new())),
            managed_turns: Arc::new(Mutex::new(HashMap::new())),
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
                policy.standard_context_approach.clone(),
                state.plugin_snapshot.as_ref(),
            )
            .map_err(|err| SessionError::Protocol(err.to_string()))?;
        let core = RuntimeCoreConfig {
            attachment_store: Arc::clone(&env.attachment_store),
            prompt: env.prompt.clone(),
            trace_sink: env.trace_sink.clone(),
            trace_level: env.trace_level,
            trace_context: env.trace_context.clone(),
            termination: env.termination.clone(),
        };
        let mut embedded = EmbeddedRuntimeHost::new(core);
        if let Some(factory) = env.session_store_factory.as_ref() {
            embedded = embedded.with_session_store_factory(Arc::clone(factory));
        }
        let runtime = if let Some(store) = store {
            let services = PersistentRuntimeServices::new_with_bridges(
                plugin_session,
                crate::session::TurnInjectionBridge::new(),
                crate::session::TurnInputInjectionBridge::new(),
                store,
            );
            match env.background_task_host.as_ref() {
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
            match env.background_task_host.as_ref() {
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
        let commit = crate::store::RuntimeCommit::persisted_state(&self.state, &[]);
        let result = store.commit_runtime_state(commit).await.map_err(|err| {
            SessionError::Protocol(format!("failed to persist runtime state: {err}"))
        })?;
        self.state.apply_persisted_commit_result(result);
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
        // ActivePathOnly is an exact store capability. Stores that do
        // not support it must return UnsupportedReadScope; resume does
        // not fall back to a full graph load.
        let loaded = match env.residency {
            Residency::KeepAll => {
                crate::store::load_persisted_session_state(parked.store.as_ref()).await
            }
            Residency::ActivePathOnly => {
                crate::store::load_persisted_session_state_active_path(parked.store.as_ref(), None)
                    .await
            }
        }
        .map_err(|err| SessionError::Protocol(format!("failed to load runtime state: {err}")))?;
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
        store
            .load_node(node_id)
            .await
            .map_err(|err| SessionError::Protocol(format!("failed to load historic node: {err}")))
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
        let Some(read) = store
            .load_session(crate::store::SessionReadScope::FullGraph)
            .await
            .map_err(|err| SessionError::Protocol(format!("failed to load full graph: {err}")))?
        else {
            return Ok(Vec::new());
        };
        let active: std::collections::HashSet<&str> = read
            .graph
            .active_path_nodes()
            .iter()
            .map(|node| node.node_id.as_str())
            .collect();
        Ok(read
            .graph
            .nodes
            .iter()
            .filter(|node| !active.contains(node.node_id.as_str()))
            .map(|node| node.node_id.clone())
            .collect())
    }
}
