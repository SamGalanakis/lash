use super::*;
use crate::runtime::host::{BackgroundRuntimeHost, EmbeddedRuntimeHost};
use std::sync::atomic::Ordering;

#[async_trait::async_trait]
impl SessionManager for RuntimeSessionManager {
    async fn snapshot_current(&self) -> Result<SessionSnapshot, crate::PluginError> {
        let mut snapshot = self.current_snapshot.to_snapshot();
        super::normalize_session_graph(&mut snapshot);
        self.enrich_current_snapshot(&mut snapshot);
        Ok(snapshot)
    }

    async fn snapshot_session(
        &self,
        session_id: &str,
    ) -> Result<SessionSnapshot, crate::PluginError> {
        self.snapshot_by_id(session_id).await
    }

    async fn tool_catalog(
        &self,
        session_id: &str,
    ) -> Result<Vec<serde_json::Value>, crate::PluginError> {
        self.tool_catalog_by_id(session_id).await
    }

    async fn dynamic_tool_state(
        &self,
        session_id: &str,
    ) -> Result<crate::DynamicStateSnapshot, crate::PluginError> {
        if session_id == self.current_session_id {
            if let Some(runtime) = self.registry.lock().await.get(session_id).cloned() {
                let runtime = runtime.lock().await;
                return runtime
                    .dynamic_tool_state()
                    .map_err(|err| crate::PluginError::Session(err.to_string()));
            }
            return Ok(self.current_dynamic_tools()?.export_state());
        }

        let runtime = {
            let registry = self.registry.lock().await;
            registry.get(session_id).cloned()
        }
        .ok_or_else(|| crate::PluginError::Session(format!("unknown session `{session_id}`")))?;
        let runtime = runtime.lock().await;
        runtime
            .dynamic_tool_state()
            .map_err(|err| crate::PluginError::Session(err.to_string()))
    }

    async fn apply_dynamic_tool_state(
        &self,
        session_id: &str,
        snapshot: crate::DynamicStateSnapshot,
    ) -> Result<u64, crate::PluginError> {
        if session_id == self.current_session_id {
            if let Some(runtime) = self.registry.lock().await.get(session_id).cloned() {
                let mut runtime = runtime.lock().await;
                return runtime
                    .apply_dynamic_tool_state(snapshot)
                    .await
                    .map_err(|err| crate::PluginError::Session(err.to_string()));
            }
            let dynamic_tools = self.current_dynamic_tools()?;
            return dynamic_tools
                .apply_state(snapshot)
                .map_err(|err| crate::PluginError::Session(err.to_string()));
        }

        let runtime = {
            let registry = self.registry.lock().await;
            registry.get(session_id).cloned()
        }
        .ok_or_else(|| crate::PluginError::Session(format!("unknown session `{session_id}`")))?;
        let mut runtime = runtime.lock().await;
        runtime
            .apply_dynamic_tool_state(snapshot)
            .await
            .map_err(|err| crate::PluginError::Session(err.to_string()))
    }

    async fn create_session(
        &self,
        request: SessionCreateRequest,
    ) -> Result<SessionHandle, crate::PluginError> {
        let session_id = request
            .session_id
            .clone()
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        if session_id == self.current_session_id
            || self.registry.lock().await.contains_key(&session_id)
        {
            return Err(crate::PluginError::Session(format!(
                "session `{session_id}` already exists"
            )));
        }
        let snapshot = match &request.start {
            SessionStartPoint::Empty => SessionSnapshot {
                session_id: session_id.clone(),
                ..Default::default()
            },
            SessionStartPoint::CurrentSession => self.current_snapshot.to_snapshot(),
            SessionStartPoint::ExistingSession { session_id } => {
                self.snapshot_by_id(session_id).await?
            }
            SessionStartPoint::Snapshot { snapshot } => (**snapshot).clone(),
        };
        let mut policy = request
            .policy
            .clone()
            .unwrap_or_else(|| match &request.start {
                SessionStartPoint::Empty => self.current_policy.clone(),
                _ => snapshot.policy.clone(),
            });
        if request.parent_session_id.is_some() {
            policy.session_id = Some(session_id.clone());
        }
        let state = self.build_runtime_state(session_id.clone(), &request, snapshot, &policy);
        let plugins = match request.plugin_mode {
            crate::SessionPluginMode::Fresh => self
                .current_plugins
                .host()
                .build_session_with_parent(
                    &session_id,
                    request.parent_session_id.clone(),
                    policy.execution_mode.clone(),
                    policy.standard_context_approach.clone(),
                    None,
                )
                .map_err(|err| crate::PluginError::Session(err.to_string()))?,
            crate::SessionPluginMode::InheritCurrent => self
                .current_plugins
                .fork_for_session(
                    &session_id,
                    policy.execution_mode.clone(),
                    policy.standard_context_approach.clone(),
                )
                .map_err(|err| crate::PluginError::Session(err.to_string()))?,
        };
        let session_store = match &self.current_host.session_store_factory {
            Some(factory) => Some(
                factory
                    .create_store(&SessionStoreCreateRequest {
                        session_id: session_id.clone(),
                        parent_session_id: request.parent_session_id.clone(),
                        policy: policy.clone(),
                    })
                    .map_err(crate::PluginError::Session)?,
            ),
            None => None,
        };
        let mut runtime = match (&self.current_host.session_task_executor, &session_store) {
            (Some(executor), Some(store)) => {
                let host = BackgroundRuntimeHost::new(
                    EmbeddedRuntimeHost {
                        core: self.current_host.core.clone(),
                        session_store_factory: self.current_host.session_store_factory.clone(),
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
                        core: self.current_host.core.clone(),
                        session_store_factory: self.current_host.session_store_factory.clone(),
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
                    core: self.current_host.core.clone(),
                    session_store_factory: self.current_host.session_store_factory.clone(),
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
                    core: self.current_host.core.clone(),
                    session_store_factory: self.current_host.session_store_factory.clone(),
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
            super::commit_runtime_state(store.as_ref(), &mut persisted_state, &[])
                .await
                .map_err(|err| crate::PluginError::Session(err.to_string()))?;
        }
        self.registry
            .lock()
            .await
            .insert(session_id.clone(), Arc::new(Mutex::new(runtime)));
        // Stash the usage_source label so await_turn can tag the child's
        // token cost in the parent's ledger.
        if let Some(source) = &request.usage_source {
            self.child_usage_sources
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
            parent_session_id: request.parent_session_id,
            policy,
        })
    }

    async fn emit_trace_event(
        &self,
        context: lash_trace::TraceContext,
        event: lash_trace::TraceEvent,
    ) -> Result<(), crate::PluginError> {
        crate::trace::emit_trace(
            &self.current_host.core.trace_sink,
            &self.current_host.core.trace_context,
            context.for_session(self.current_session_id.clone()),
            event,
        );
        Ok(())
    }

    async fn close_session(&self, session_id: &str) -> Result<(), crate::PluginError> {
        if session_id == self.current_session_id {
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
        self.child_usage_sources
            .lock()
            .expect("child usage sources lock")
            .remove(session_id);
        self.pending_first_turn_inputs
            .lock()
            .expect("pending first turn inputs lock")
            .remove(session_id);
        self.current_plugins.host().unregister_session(session_id)?;
        Ok(())
    }

    async fn start_turn_stream(
        &self,
        session_id: &str,
        input: TurnInput,
    ) -> Result<crate::plugin::SessionTurnHandle, crate::PluginError> {
        if self
            .turns
            .lock()
            .await
            .values()
            .any(|turn| turn.session_id == session_id)
        {
            return Err(crate::PluginError::Session(format!(
                "session `{session_id}` already has a running turn"
            )));
        }
        let runtime = {
            let registry = self.registry.lock().await;
            registry.get(session_id).cloned()
        }
        .ok_or_else(|| crate::PluginError::Session(format!("unknown session `{session_id}`")))?;
        let policy = {
            let runtime = runtime.lock().await;
            runtime.session_policy()
        };
        let turn_id = uuid::Uuid::new_v4().to_string();
        let cancel = CancellationToken::new();
        let (event_tx, event_rx) = mpsc::channel::<SessionEvent>(100);
        let usage_source = self
            .child_usage_sources
            .lock()
            .expect("child usage sources lock")
            .get(session_id)
            .cloned()
            .unwrap_or_else(|| "child".to_string());
        let runtime_clone = Arc::clone(&runtime);
        let cancel_clone = cancel.clone();
        let sink = ChannelEventSink {
            tx: event_tx,
            live_usage: Some(LiveChildUsageForwarder {
                turn_id: turn_id.clone(),
                session_id: session_id.to_string(),
                source: usage_source,
                model: policy.model.clone(),
                token_ledger: Arc::clone(&self.token_ledger),
                child_turn_live_usage: Arc::clone(&self.child_turn_live_usage),
                relay: self.child_usage_event_relay.clone(),
            }),
        };
        let task = tokio::spawn(async move {
            let mut runtime = runtime_clone.lock().await;
            runtime
                .refresh_session_tool_surface()
                .await
                .map_err(|err| crate::PluginError::Session(err.to_string()))?;
            runtime
                .stream_turn(input, &sink, cancel_clone)
                .await
                .map_err(|err| crate::PluginError::Session(err.to_string()))
        });
        self.turns.lock().await.insert(
            turn_id.clone(),
            ManagedSessionTurn {
                session_id: session_id.to_string(),
                cancel,
                task,
            },
        );
        Ok(crate::plugin::SessionTurnHandle {
            turn_id,
            session_id: session_id.to_string(),
            policy,
            events: event_rx,
        })
    }

    async fn await_turn(&self, turn_id: &str) -> Result<AssembledTurn, crate::PluginError> {
        let managed = self
            .turns
            .lock()
            .await
            .remove(turn_id)
            .ok_or_else(|| crate::PluginError::Session(format!("unknown turn `{turn_id}`")))?;
        let session_id = managed.session_id.clone();
        let turn = managed
            .task
            .await
            .map_err(|err| crate::PluginError::Session(format!("turn task failed: {err}")))?;
        let live_reported = self
            .child_turn_live_usage
            .lock()
            .expect("child turn live usage lock")
            .remove(turn_id)
            .unwrap_or_default();
        if let Ok(turn) = &turn {
            let source = self
                .child_usage_sources
                .lock()
                .expect("child usage sources lock")
                .get(&session_id)
                .cloned()
                .unwrap_or_else(|| "child".to_string());
            if let Some(remainder) = subtract_usage(&live_reported, &turn.token_usage) {
                self.record_token_usage(&source, &turn.state.policy.model, &remainder);
            }
        }
        self.persist_current_usage_ledger().await?;
        turn
    }

    async fn cancel_turn(&self, turn_id: &str) -> Result<(), crate::PluginError> {
        let turns = self.turns.lock().await;
        let managed = turns
            .get(turn_id)
            .ok_or_else(|| crate::PluginError::Session(format!("unknown turn `{turn_id}`")))?;
        managed.cancel.cancel();
        Ok(())
    }

    async fn spawn_hidden_task(
        &self,
        session_id: &str,
        label: &str,
        task: crate::plugin::PluginSessionTask,
    ) -> Result<(), crate::PluginError> {
        if session_id != self.current_session_id {
            let known = self.registry.lock().await.contains_key(session_id);
            if !known {
                return Err(crate::PluginError::Session(format!(
                    "unknown session `{session_id}`"
                )));
            }
        }
        let Some(executor) = &self.current_host.session_task_executor else {
            return Err(crate::PluginError::Session(
                "session tasks are unavailable in this runtime".to_string(),
            ));
        };
        if session_id == self.current_session_id {
            self.background_sync_needed.store(true, Ordering::Release);
        }
        executor
            .spawn_hidden(&self.background_scope_key(session_id), label, task)
            .await
    }

    async fn await_hidden_tasks(&self, session_id: &str) -> Result<(), crate::PluginError> {
        let Some(executor) = &self.current_host.session_task_executor else {
            return Ok(());
        };
        executor
            .await_hidden(&self.background_scope_key(session_id))
            .await
    }

    async fn spawn_managed_task(
        &self,
        session_id: &str,
        spec: crate::ManagedTaskSpec,
        task: crate::plugin::PluginSessionTask,
    ) -> Result<(), crate::PluginError> {
        if session_id != self.current_session_id {
            let known = self.registry.lock().await.contains_key(session_id);
            if !known {
                return Err(crate::PluginError::Session(format!(
                    "unknown session `{session_id}`"
                )));
            }
        }
        let Some(executor) = &self.current_host.session_task_executor else {
            return Err(crate::PluginError::Session(
                "managed session tasks are unavailable in this runtime".to_string(),
            ));
        };
        if session_id == self.current_session_id {
            self.background_sync_needed.store(true, Ordering::Release);
        }
        executor
            .spawn_managed(&self.background_scope_key(session_id), spec, task)
            .await
    }

    async fn cancel_managed_task(
        &self,
        session_id: &str,
        task_id: &str,
    ) -> Result<(), crate::PluginError> {
        let Some(executor) = &self.current_host.session_task_executor else {
            return Ok(());
        };
        executor
            .cancel_managed(&self.background_scope_key(session_id), task_id)
            .await
    }

    async fn register_background_task(
        &self,
        session_id: &str,
        spec: crate::ManagedTaskSpec,
        cancel: Option<crate::ManagedTaskCancel>,
    ) -> Result<(), crate::PluginError> {
        let Some(executor) = &self.current_host.session_task_executor else {
            return Err(crate::PluginError::Session(
                "background task registry is unavailable in this runtime".to_string(),
            ));
        };
        executor
            .register_external(&self.background_scope_key(session_id), spec, cancel)
            .await
    }

    async fn complete_background_task(
        &self,
        session_id: &str,
        task_id: &str,
        run_state: crate::ManagedRunState,
    ) {
        let Some(executor) = &self.current_host.session_task_executor else {
            return;
        };
        executor
            .mark_terminal(&self.background_scope_key(session_id), task_id, run_state)
            .await;
    }

    async fn transition_background_task_live_state(
        &self,
        session_id: &str,
        task_id: &str,
        run_state: crate::ManagedRunState,
    ) {
        let Some(executor) = &self.current_host.session_task_executor else {
            return;
        };
        executor
            .mark_live_state(&self.background_scope_key(session_id), task_id, run_state)
            .await;
    }

    async fn take_first_turn_input(
        &self,
        session_id: &str,
    ) -> Result<Option<crate::PluginMessage>, crate::PluginError> {
        Ok(self
            .pending_first_turn_inputs
            .lock()
            .expect("pending first turn inputs lock")
            .remove(session_id))
    }

    async fn inject_turn_input(
        &self,
        session_id: &str,
        message: crate::PluginMessage,
    ) -> Result<(), crate::PluginError> {
        let runtime_arc = {
            let registry = self.registry.lock().await;
            registry.get(session_id).cloned()
        };
        let Some(runtime_arc) = runtime_arc else {
            return Err(crate::PluginError::Session(format!(
                "unknown or inactive session `{session_id}` for turn input injection"
            )));
        };
        let runtime = runtime_arc.lock().await;
        let Some(session) = runtime.session.as_ref() else {
            return Err(crate::PluginError::Session(format!(
                "session `{session_id}` has no live turn-input bridge"
            )));
        };
        session
            .turn_input_injection_bridge()
            .enqueue(vec![crate::InjectedTurnInput { message }])
            .map_err(crate::PluginError::Session)
    }

    async fn list_background_tasks(
        &self,
        session_id: &str,
    ) -> Result<Vec<crate::ManagedTaskStatus>, crate::PluginError> {
        let Some(executor) = &self.current_host.session_task_executor else {
            return Err(crate::PluginError::Session(
                "background task registry is unavailable in this runtime".to_string(),
            ));
        };
        Ok(executor
            .list_managed(&self.background_scope_key(session_id))
            .await)
    }

    async fn cancel_background_task(
        &self,
        session_id: &str,
        task_id: &str,
    ) -> Result<crate::ManagedTaskStatus, crate::PluginError> {
        let Some(executor) = &self.current_host.session_task_executor else {
            return Err(crate::PluginError::Session(
                "background task registry is unavailable in this runtime".to_string(),
            ));
        };
        let scope_key = self.background_scope_key(session_id);
        let Some(status) = executor.get_managed(&scope_key, task_id).await else {
            return Err(crate::PluginError::Session(format!(
                "unknown background task `{task_id}`"
            )));
        };
        match status.kind {
            crate::ManagedTaskKind::Monitor => {
                let monitor_id = task_id
                    .strip_prefix("monitor:")
                    .unwrap_or(task_id)
                    .to_string();
                self.stop_monitor(session_id, &monitor_id).await?;
                executor
                    .mark_terminal(&scope_key, task_id, crate::ManagedRunState::Cancelled)
                    .await;
            }
            crate::ManagedTaskKind::Subagent => {
                // Producer-owned cancel: ask the task owner to terminate via
                // its registered cancellation channel. For now we just mark
                // the record; subagent host wires up a close path in its
                // register_background_task adapter.
                executor.cancel_managed(&scope_key, task_id).await?;
                executor
                    .mark_terminal(&scope_key, task_id, crate::ManagedRunState::Cancelled)
                    .await;
            }
            _ => {
                executor.cancel_managed(&scope_key, task_id).await?;
                executor
                    .mark_terminal(&scope_key, task_id, crate::ManagedRunState::Cancelled)
                    .await;
            }
        }
        let updated = executor
            .get_managed(&scope_key, task_id)
            .await
            .unwrap_or(status);
        Ok(updated)
    }

    async fn monitor_snapshot(
        &self,
        session_id: &str,
    ) -> Result<crate::MonitorSnapshot, crate::PluginError> {
        self.ensure_registered_monitor_specs(session_id).await?;
        let result = self
            .invoke_monitor_external(session_id, "monitor.status", serde_json::json!({}))
            .await?;
        if !result.success {
            return Err(crate::PluginError::Session(result.result.to_string()));
        }
        serde_json::from_value(result.result)
            .map_err(|err| crate::PluginError::Session(format!("invalid monitor status: {err}")))
    }

    async fn take_monitor_updates(
        &self,
        session_id: &str,
    ) -> Result<crate::MonitorUpdateBatch, crate::PluginError> {
        self.ensure_registered_monitor_specs(session_id).await?;
        let result = self
            .invoke_monitor_external(session_id, "monitor.take_updates", serde_json::json!({}))
            .await?;
        if !result.success {
            return Err(crate::PluginError::Session(result.result.to_string()));
        }
        serde_json::from_value(result.result)
            .map_err(|err| crate::PluginError::Session(format!("invalid monitor updates: {err}")))
    }

    async fn start_monitor(
        &self,
        session_id: &str,
        spec: crate::MonitorSpec,
    ) -> Result<crate::MonitorSnapshot, crate::PluginError> {
        self.ensure_registered_monitor_specs(session_id).await?;
        let result = self
            .invoke_monitor_external(
                session_id,
                "monitor.start",
                serde_json::json!({ "spec": spec }),
            )
            .await?;
        if !result.success {
            return Err(crate::PluginError::Session(result.result.to_string()));
        }
        serde_json::from_value(result.result)
            .map_err(|err| crate::PluginError::Session(format!("invalid monitor status: {err}")))
    }

    async fn stop_monitor(
        &self,
        session_id: &str,
        monitor_id: &str,
    ) -> Result<crate::MonitorSnapshot, crate::PluginError> {
        self.ensure_registered_monitor_specs(session_id).await?;
        let result = self
            .invoke_monitor_external(
                session_id,
                "monitor.stop",
                serde_json::json!({ "id": monitor_id }),
            )
            .await?;
        if !result.success {
            return Err(crate::PluginError::Session(result.result.to_string()));
        }
        serde_json::from_value(result.result)
            .map_err(|err| crate::PluginError::Session(format!("invalid monitor status: {err}")))
    }

    async fn prompt_user(
        &self,
        request: crate::PromptRequest,
    ) -> Result<crate::PromptResponse, crate::PluginError> {
        let Some(prompt_bridge) = &self.current_prompt_bridge else {
            return Err(crate::PluginError::Session(
                "user prompts are unavailable in this session".to_string(),
            ));
        };
        prompt_bridge.prompt(request).await
    }

    async fn append_session_nodes(
        &self,
        session_id: &str,
        request: crate::AppendSessionNodesRequest,
    ) -> Result<crate::AppendSessionNodesResult, crate::PluginError> {
        if let Some(runtime) = {
            let registry = self.registry.lock().await;
            registry.get(session_id).cloned()
        } {
            let mut runtime = runtime.lock().await;
            return runtime
                .append_session_nodes(request)
                .await
                .map_err(|err| crate::PluginError::Session(err.to_string()));
        }

        if session_id != self.current_session_id {
            return Err(crate::PluginError::Session(format!(
                "unknown session `{session_id}`"
            )));
        }

        let Some(store) = &self.current_store else {
            return Err(crate::PluginError::Session(
                "session graph mutation requires a runtime store".to_string(),
            ));
        };

        let mut state = if self.persist_usage_to_store {
            self.current_snapshot_for_store_write().await
        } else {
            let mut state = self.current_snapshot.to_snapshot();
            super::normalize_session_graph(&mut state);
            state
        };
        let usage_deltas = if self.persist_usage_to_store {
            self.merge_drained_token_ledger(&mut state)
        } else {
            Vec::new()
        };
        if let Some(required) = request.requires_ancestor_node_id.as_deref()
            && !state.session_graph.active_path_contains(required)
        {
            return Ok(crate::AppendSessionNodesResult::StaleBranch {
                current_leaf_node_id: state.session_graph.leaf_node_id.clone(),
            });
        }
        let node_ids = append_session_nodes_to_state(&mut state, &request.nodes);
        let leaf_node_id = state.session_graph.leaf_node_id.clone().unwrap_or_default();
        super::commit_runtime_state(store.as_ref(), &mut state, &usage_deltas)
            .await
            .map_err(|err| crate::PluginError::Session(err.to_string()))?;
        self.background_sync_needed.store(true, Ordering::Release);
        Ok(crate::AppendSessionNodesResult::Appended {
            node_ids,
            leaf_node_id,
        })
    }

    async fn direct_completion(
        &self,
        request: crate::DirectRequest,
        usage_source: &str,
    ) -> Result<crate::DirectCompletion, crate::PluginError> {
        let mut provider = self.current_policy.provider.clone();
        let model = provider.resolve_model(&request.model);
        if let Some(variant) = request.model_variant.as_deref() {
            provider
                .validate_variant(&model, variant)
                .map_err(crate::PluginError::Session)?;
        }
        provider
            .ensure_ready()
            .await
            .map_err(|err| crate::PluginError::Session(err.message.clone()))?;
        let llm_request = crate::direct::build_llm_request(&provider, request, model.clone());
        let llm_call_id = if self.current_host.core.trace_sink.is_some() {
            let id = uuid::Uuid::new_v4().to_string();
            crate::trace::emit_trace(
                &self.current_host.core.trace_sink,
                &self.current_host.core.trace_context,
                lash_trace::TraceContext::default()
                    .for_session(self.current_session_id.clone())
                    .for_llm_call(id.clone()),
                lash_trace::TraceEvent::LlmCallStarted {
                    request: crate::trace::trace_llm_request(&llm_request),
                },
            );
            Some(id)
        } else {
            None
        };
        let response = match provider.complete(llm_request).await {
            Ok(response) => response,
            Err(err) => {
                if let Some(llm_call_id) = llm_call_id {
                    crate::trace::emit_trace(
                        &self.current_host.core.trace_sink,
                        &self.current_host.core.trace_context,
                        lash_trace::TraceContext::default()
                            .for_session(self.current_session_id.clone())
                            .for_llm_call(llm_call_id),
                        lash_trace::TraceEvent::LlmCallFailed {
                            error: lash_trace::TraceError {
                                message: err.message.clone(),
                                retryable: err.retryable,
                                code: err.code.clone(),
                                raw: err.raw.clone(),
                            },
                            stream_summary: None,
                        },
                    );
                }
                return Err(crate::PluginError::Session(err.message.clone()));
            }
        };
        if let Some(llm_call_id) = llm_call_id {
            crate::trace::emit_trace(
                &self.current_host.core.trace_sink,
                &self.current_host.core.trace_context,
                lash_trace::TraceContext::default()
                    .for_session(self.current_session_id.clone())
                    .for_llm_call(llm_call_id),
                lash_trace::TraceEvent::LlmCallCompleted {
                    response: crate::trace::trace_llm_response(
                        response.full_text.clone(),
                        0,
                        crate::trace::trace_output_parts(&response.parts),
                    ),
                    usage: Some(crate::trace::trace_usage_from_llm(&response.usage)),
                    provider_usage: response.provider_usage.clone(),
                    stream_summary: None,
                },
            );
        }
        let usage = TokenUsage {
            input_tokens: response.usage.input_tokens,
            output_tokens: response.usage.output_tokens,
            cached_input_tokens: response.usage.cached_input_tokens,
            reasoning_tokens: response.usage.reasoning_tokens,
        };
        self.record_token_usage(usage_source, &model, &usage);
        self.persist_current_usage_ledger().await?;
        Ok(crate::DirectCompletion {
            text: response.full_text,
            usage,
        })
    }
}
