use super::*;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use lash_trace::{
    TraceBranchSelection, TraceContext, TraceEvent, TraceLabelMetadata,
    TraceLashlangChildExecution, TraceLashlangExecutionEvent, TraceLashlangExecutionIdentity,
    TraceLashlangMap, TraceLashlangMapEdge, TraceLashlangMapNode, TraceLashlangStatus,
    TraceRuntimeScope, TraceRuntimeSubject, TraceSink,
};

impl RuntimeSessionServices {
    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime::session_manager::process_runners) async fn run_lashlang_process(
        &self,
        registration: crate::ProcessRegistration,
        registry: Arc<dyn crate::ProcessRegistry>,
        module_ref: ::lashlang::ModuleRef,
        process_ref: ::lashlang::ProcessRef,
        required_surface_ref: ::lashlang::RequiredSurfaceRef,
        process_name: String,
        args: serde_json::Map<String, serde_json::Value>,
        execution_context: crate::ProcessExecutionContext,
        scoped_effect_controller: crate::ScopedEffectController<'_>,
        cancellation: tokio_util::sync::CancellationToken,
    ) -> crate::ProcessAwaitOutput {
        let artifact = match self
            .current
            .host
            .core
            .durability
            .lashlang_artifact_store
            .get_module_artifact(&module_ref)
            .await
        {
            Ok(Some(artifact)) => artifact,
            Ok(None) => {
                return process_lashlang_failure(
                    "process_module_artifact_missing",
                    format!("missing lashlang module artifact `{module_ref}`"),
                    None,
                );
            }
            Err(err) => {
                return process_lashlang_failure(
                    "process_module_artifact_load_failed",
                    format!("failed to load lashlang module artifact `{module_ref}`: {err}"),
                    None,
                );
            }
        };
        if artifact.required_surface_ref != required_surface_ref {
            return process_lashlang_failure(
                "process_required_surface_mismatch",
                format!(
                    "lashlang process `{process_name}` requested surface {}, artifact has {}",
                    required_surface_ref, artifact.required_surface_ref
                ),
                None,
            );
        }
        if artifact.process_ref(&process_name) != Some(&process_ref) {
            return process_lashlang_failure(
                "process_ref_mismatch",
                format!(
                    "lashlang module `{module_ref}` does not export process `{process_name}` as requested ref {:?}",
                    process_ref
                ),
                None,
            );
        }
        let tool_surface = match self.current.plugins.tool_surface(&self.current.session_id) {
            Ok(surface) => surface,
            Err(err) => {
                return process_lashlang_failure(
                    "process_tool_surface_failed",
                    err.to_string(),
                    None,
                );
            }
        };
        let lashlang_abilities = crate::runtime::builder::lashlang_abilities_for_process_registry(
            self.current.plugins.lashlang_abilities(),
            self.current.host.process_registry.is_some(),
        );
        let current_surface = crate::session::lashlang_surface_from_tool_surface(
            &tool_surface,
            lashlang_abilities,
            self.current.plugins.lashlang_language_features(),
            self.current.plugins.lashlang_resources(),
        );
        if let Err(err) =
            lashlang_surface_satisfies_requirements(&artifact.required_surface, &current_surface)
        {
            return process_lashlang_failure(
                "process_surface_incompatible",
                format!(
                    "lashlang process `{process_name}` is incompatible with this host surface: {err}"
                ),
                None,
            );
        }
        let compiled = match self
            .current
            .host
            .core
            .durability
            .lashlang_process_cache
            .lock()
        {
            Ok(mut cache) => cache.get_or_compile(&artifact, &process_ref, &required_surface_ref),
            Err(_) => Err(::lashlang::RuntimeError::ValueError {
                message: "lashlang compiled process cache lock poisoned".to_string(),
            }),
        };
        let compiled = match compiled {
            Ok(compiled) => compiled,
            Err(err) => {
                return process_lashlang_failure(
                    "process_compile_failed",
                    format!("failed to compile process `{process_name}`: {err}"),
                    None,
                );
            }
        };
        let lashlang_execution_sink = self
            .current
            .host
            .core
            .tracing
            .lashlang_execution_sink
            .clone();
        let lashlang_execution_context = self.current.host.core.tracing.trace_context.clone();
        let lashlang_execution_trace = LashlangExecutionTraceContext {
            sink: &lashlang_execution_sink,
            base_context: &lashlang_execution_context,
            session_id: &self.current.session_id,
        };
        emit_process_started_trace(
            lashlang_execution_trace,
            &registration.id,
            &artifact,
            &process_ref,
            &process_name,
        );
        let mut globals = ::lashlang::Record::with_capacity(args.len());
        for (name, value) in args {
            globals.insert(name, ::lashlang::from_json(value));
        }
        let mut state = ::lashlang::State::from_snapshot(::lashlang::Snapshot { globals });

        let run_context = ProcessRunContext::builder(self)
            .surface(tool_surface)
            .scoped_effect_controller(scoped_effect_controller)
            .causal_invocation(execution_context.causal_invocation.clone())
            .build()
            .map_err(|err| {
                process_lashlang_failure("process_run_context_failed", err.to_string(), None)
            });
        let run_context = match run_context {
            Ok(run_context) => run_context,
            Err(output) => return output,
        };
        let mut ctx = crate::RuntimeExecutionContext::new(
            self.current.session_id.clone(),
            run_context.dispatch(),
            lashlang_abilities,
            self.current.plugins.lashlang_language_features(),
            Arc::clone(&self.current.host.core.durability.lashlang_artifact_store),
            Arc::clone(&self.current.host.core.durability.attachment_store),
            Arc::new(crate::ChronologicalProjection::default()),
            None,
            crate::TurnContext::default(),
        )
        .with_execution_env_spec(current_execution_env_spec(&self.current))
        .with_process_registration_context(&registration)
        .with_cancellation_token(cancellation.clone());
        if let Some(invocation) = execution_context.causal_invocation.clone() {
            ctx = ctx.with_parent_invocation(invocation);
        }

        let host = LashlangProcessHost {
            ctx,
            registry: Arc::clone(&registry),
            process_id: registration.id.clone(),
            session_id: self.current.session_id.clone(),
            module_ref: module_ref.clone(),
            process_ref: process_ref.clone(),
            process_name: process_name.clone(),
            lashlang_execution_sink: lashlang_execution_sink.clone(),
            lashlang_execution_context: lashlang_execution_context.clone(),
            store: self.current.store.clone(),
            session_store_factory: self.current.host.session_store_factory.clone(),
            queued_work_poke: self.current.host.queued_work_poke.clone(),
            sleep_sequence: AtomicU64::new(0),
            event_sequence: AtomicU64::new(0),
            resource_tool_sequence: AtomicU64::new(0),
            signal_send_sequence: AtomicU64::new(0),
            signal_wait_ordinals: tokio::sync::Mutex::new(BTreeMap::new()),
        };
        let env = ::lashlang::ExecutionEnvironment::new(&host).process();

        let output = {
            tokio::select! {
                _ = cancellation.cancelled() => process_lashlang_cancelled("lashlang process was cancelled"),
                result = ::lashlang::execute(compiled.as_ref(), &mut state, &env) => {
                    if cancellation.is_cancelled() {
                        process_lashlang_cancelled("lashlang process was cancelled")
                    } else {
                        process_lashlang_execution_result(result)
                    }
                }
            }
        };
        drop(env);
        drop(host);
        run_context.shutdown().await;
        emit_process_finished_trace(
            lashlang_execution_trace,
            &registration.id,
            &module_ref,
            &process_ref,
            &process_name,
            &output,
        );
        output
    }
}

fn current_execution_env_spec(
    current: &CurrentSessionCapability,
) -> crate::ProcessExecutionEnvSpec {
    let state = current.snapshot.to_runtime_state();
    state.process_execution_env_spec(&current.policy)
}

struct LashlangProcessHost<'run> {
    ctx: crate::RuntimeExecutionContext<'run>,
    registry: Arc<dyn crate::ProcessRegistry>,
    process_id: String,
    session_id: String,
    module_ref: ::lashlang::ModuleRef,
    process_ref: ::lashlang::ProcessRef,
    process_name: String,
    lashlang_execution_sink: Option<Arc<dyn TraceSink>>,
    lashlang_execution_context: TraceContext,
    store: Option<Arc<dyn crate::RuntimePersistence>>,
    session_store_factory: Option<Arc<dyn crate::SessionStoreFactory>>,
    queued_work_poke: Option<crate::QueuedWorkPoke>,
    sleep_sequence: AtomicU64,
    /// Per-execution ordinal for wake/yield emissions. Deterministic replay
    /// re-issues `process_event` calls in the same order, so the Nth emission
    /// gets the same ordinal — and thus the same replay key — across a
    /// crash-recovery re-run, making the append idempotent on redelivery.
    event_sequence: AtomicU64,
    resource_tool_sequence: AtomicU64,
    signal_send_sequence: AtomicU64,
    signal_wait_ordinals: tokio::sync::Mutex<BTreeMap<String, u64>>,
}

impl LashlangProcessHost<'_> {
    fn resource_payload(
        &self,
        args: &[::lashlang::Value],
    ) -> Result<serde_json::Value, ::lashlang::ExecutionHostError> {
        let mut payload = if let [::lashlang::Value::Record(record)] = args {
            crate::lashlang_bridge::lashlang_value_to_json(&::lashlang::Value::Record(Arc::clone(
                record,
            )))?
        } else {
            serde_json::json!({
                "args": args
                    .iter()
                    .map(crate::lashlang_bridge::lashlang_value_to_json)
                    .collect::<Result<Vec<_>, _>>()?,
            })
        };
        payload.as_object_mut().ok_or_else(|| {
            ::lashlang::ExecutionHostError::new("module operation payload must be an object")
        })?;
        Ok(payload)
    }

    fn execution_identity(&self) -> TraceLashlangExecutionIdentity {
        process_execution_identity(
            &self.session_id,
            &self.process_id,
            &self.module_ref,
            &self.process_ref,
            &self.process_name,
        )
    }

    fn resource_tool_call_id(
        &self,
        host_operation: &str,
        call_site: Option<&::lashlang::LashlangExecutionCallSite>,
    ) -> String {
        if let Some(call_site) = call_site {
            return format!(
                "lashlang:{}:resource:{}:{}:{}",
                self.process_id, host_operation, call_site.site.node_id, call_site.occurrence
            );
        }
        let ordinal = self.resource_tool_sequence.fetch_add(1, Ordering::Relaxed);
        format!(
            "lashlang:{}:resource:{}:ordinal:{}",
            self.process_id, host_operation, ordinal
        )
    }
}

impl LashlangProcessHost<'_> {
    async fn resource_operation(
        &self,
        operation: String,
        receiver: ::lashlang::Value,
        args: Vec<::lashlang::Value>,
        call_site: Option<::lashlang::LashlangExecutionCallSite>,
    ) -> Result<::lashlang::Value, ::lashlang::ExecutionHostError> {
        let receiver = match &receiver {
            ::lashlang::Value::Resource(receiver) => receiver,
            _ => {
                return Err(::lashlang::ExecutionHostError::new(format!(
                    "module operation `{operation}` requires a module authority receiver"
                )));
            }
        };
        let host_operation = self
            .ctx
            .resolve_lashlang_host_operation(receiver, &operation)
            .map_err(::lashlang::ExecutionHostError::new)?;
        if host_operation.starts_with("triggers.") {
            let payload = self.resource_payload(&args)?;
            let value = self
                .ctx
                .perform_lashlang_trigger_operation(&host_operation, payload)
                .await
                .map_err(::lashlang::ExecutionHostError::new)?;
            return Ok(::lashlang::from_json(value));
        }
        let manifest = self.ctx.callable_tool_manifest(&host_operation).ok_or_else(|| {
            ::lashlang::ExecutionHostError::new(format!(
                "module operation `{}` resolved to unavailable host operation `{host_operation}`",
                operation
            ))
        })?;
        let call_id = self.resource_tool_call_id(&host_operation, call_site.as_ref());
        let call_site = call_site.and_then(|call_site| {
            self.lashlang_execution_sink.as_ref().map(|sink| {
                crate::ToolLashlangExecutionCallSite::new(
                    Arc::clone(sink),
                    self.lashlang_execution_context.clone(),
                    self.execution_identity(),
                    call_site.site.node_id,
                    call_site.occurrence,
                )
            })
        });
        let payload = self.resource_payload(&args)?;
        let reply = if let Some(call_site) = call_site {
            self.ctx
                .call_tool_with_lashlang_execution_call_site(
                    call_id,
                    manifest.name.clone(),
                    payload,
                    0,
                    call_site,
                )
                .await
        } else {
            self.ctx
                .call_tool(call_id, manifest.name.clone(), payload, 0)
                .await
        };
        protocol_reply_to_lashlang_value(reply)
    }

    async fn await_handle(
        &self,
        handle: ::lashlang::Value,
    ) -> Result<::lashlang::Value, ::lashlang::ExecutionHostError> {
        let reply = self
            .ctx
            .await_tool_handle(
                uuid::Uuid::new_v4().to_string(),
                crate::lashlang_bridge::lashlang_value_to_json(&handle)?,
            )
            .await;
        protocol_reply_to_lashlang_value(reply)
    }

    async fn cancel_handle(
        &self,
        handle: ::lashlang::Value,
    ) -> Result<::lashlang::Value, ::lashlang::ExecutionHostError> {
        let reply = self
            .ctx
            .cancel_tool_handle(
                uuid::Uuid::new_v4().to_string(),
                crate::lashlang_bridge::lashlang_value_to_json(&handle)?,
            )
            .await;
        protocol_reply_to_lashlang_value(reply)
    }

    async fn start_process(
        &self,
        start: ::lashlang::ProcessStart,
    ) -> Result<::lashlang::Value, ::lashlang::ExecutionHostError> {
        let (registration, label) = self
            .ctx
            .prepare_lashlang_process_start(start)
            .await
            .map_err(::lashlang::ExecutionHostError::new)?;
        let reply = self.ctx.start_lashlang_process(registration, label).await;
        protocol_reply_to_lashlang_value(reply)
    }

    async fn process_event(
        &self,
        event: ::lashlang::ProcessEvent,
    ) -> Result<(), ::lashlang::ExecutionHostError> {
        let event_type = match event.kind {
            ::lashlang::ProcessEventKind::Yield => "process.yield",
            ::lashlang::ProcessEventKind::Wake => "process.wake",
        };
        // Deterministic per-emission key: a crash-recovery re-run replays the
        // process from the start, re-issuing wake/yield events in the same order,
        // so the Nth emission keys to the same value and the append dedupes
        // instead of redelivering a duplicate wake/yield.
        let ordinal = self.event_sequence.fetch_add(1, Ordering::Relaxed);
        let result = self
            .registry
            .append_event(
                &self.process_id,
                crate::ProcessEventAppendRequest::new(
                    event_type,
                    crate::lashlang_bridge::process_event_payload(&event.value)?,
                )
                .with_replay_key(format!("process:{}:event:{ordinal}", self.process_id)),
            )
            .await
            .map_err(|err| ::lashlang::ExecutionHostError::new(err.to_string()))?;
        crate::tool_provider::process_events::enqueue_wake_delivery(
            self.store.clone(),
            self.session_store_factory.as_ref(),
            result.wake_delivery,
            Some(self.ctx.session_graph_service()),
            self.queued_work_poke.as_ref(),
        )
        .await
        .map_err(|err| ::lashlang::ExecutionHostError::new(err.to_string()))?;
        Ok(())
    }

    async fn sleep(
        &self,
        sleep: ::lashlang::Sleep,
    ) -> Result<::lashlang::Value, ::lashlang::ExecutionHostError> {
        let duration_ms = crate::lashlang_bridge::sleep_duration_ms(sleep.kind, &sleep.value)?;
        let sequence = self.sleep_sequence.fetch_add(1, Ordering::Relaxed);
        let scope = format!("process:{}", self.process_id);
        self.ctx
            .sleep_lashlang(&scope, sequence, duration_ms)
            .await
            .map_err(|err| ::lashlang::ExecutionHostError::new(err.to_string()))?;
        Ok(::lashlang::Value::Null)
    }

    async fn wait_signal(
        &self,
        name: String,
    ) -> Result<::lashlang::Value, ::lashlang::ExecutionHostError> {
        let event_type = crate::process_signal_event_type(&name)
            .map_err(|err| ::lashlang::ExecutionHostError::new(err.to_string()))?;
        let event_ordinal = {
            let mut ordinals = self.signal_wait_ordinals.lock().await;
            let ordinal = ordinals.entry(name.clone()).or_insert(0);
            *ordinal += 1;
            *ordinal
        };
        let key = crate::process_signal_wait_key(&self.process_id, &name, event_ordinal);
        let waiting_replay_key = format!(
            "process:{}:waiting:signal.{}:{event_ordinal}",
            self.process_id, name
        );
        let since_ms = self
            .wait_since_ms(&key, &waiting_replay_key)
            .await
            .map_err(|err| ::lashlang::ExecutionHostError::new(err.to_string()))?;
        let wait = crate::WaitState {
            since_ms,
            kind: crate::WaitKind::Signal {
                name: name.clone(),
                event_type: event_type.clone(),
                key: key.clone(),
                ordinal: event_ordinal,
            },
        };
        self.registry
            .set_process_wait(&self.process_id, wait.clone())
            .await
            .map_err(|err| ::lashlang::ExecutionHostError::new(err.to_string()))?;
        self.registry
            .append_event(
                &self.process_id,
                crate::ProcessEventAppendRequest::new(
                    "process.waiting",
                    serde_json::json!({ "wait": wait }),
                )
                .with_replay_key(waiting_replay_key),
            )
            .await
            .map_err(|err| ::lashlang::ExecutionHostError::new(err.to_string()))?;
        let payload = self
            .ctx
            .await_process_event_lashlang(
                Arc::clone(&self.registry),
                &self.process_id,
                &name,
                &event_type,
                event_ordinal,
            )
            .await
            .map_err(|err| ::lashlang::ExecutionHostError::new(err.to_string()))?;
        self.registry
            .clear_process_wait(&self.process_id)
            .await
            .map_err(|err| ::lashlang::ExecutionHostError::new(err.to_string()))?;
        self.registry
            .append_event(
                &self.process_id,
                crate::ProcessEventAppendRequest::new(
                    "process.resumed",
                    serde_json::json!({
                        "signal": name,
                        "key": key,
                        "ordinal": event_ordinal,
                    }),
                )
                .with_replay_key(format!(
                    "process:{}:resumed:signal.{}:{event_ordinal}",
                    self.process_id, name
                )),
            )
            .await
            .map_err(|err| ::lashlang::ExecutionHostError::new(err.to_string()))?;
        Ok(::lashlang::from_json(payload))
    }

    async fn wait_since_ms(
        &self,
        key: &str,
        waiting_replay_key: &str,
    ) -> Result<u64, crate::PluginError> {
        if let Some(since_ms) =
            self.registry
                .get_process(&self.process_id)
                .await
                .and_then(|record| {
                    let wait = record.wait?;
                    match &wait.kind {
                        crate::WaitKind::Signal { key: wait_key, .. } if wait_key == key => {
                            Some(wait.since_ms)
                        }
                        _ => None,
                    }
                })
        {
            return Ok(since_ms);
        }

        for event in self
            .registry
            .events_after(&self.process_id, 0)
            .await?
            .into_iter()
            .rev()
        {
            if event.event_type != "process.waiting"
                || event.invocation.replay_key() != Some(waiting_replay_key)
            {
                continue;
            }
            let Some(wait_value) = event.payload.get("wait") else {
                continue;
            };
            let Ok(wait) = serde_json::from_value::<crate::WaitState>(wait_value.clone()) else {
                continue;
            };
            match &wait.kind {
                crate::WaitKind::Signal { key: wait_key, .. } if wait_key == key => {
                    return Ok(wait.since_ms);
                }
                _ => {}
            }
        }

        Ok(crate::current_epoch_ms())
    }

    async fn signal_run(
        &self,
        signal: ::lashlang::ProcessSignal,
    ) -> Result<::lashlang::Value, ::lashlang::ExecutionHostError> {
        let target = process_id_from_lashlang_handle(&signal.run)?;
        let payload = crate::lashlang_bridge::lashlang_value_to_json(&signal.payload)?;
        let sequence = self.signal_send_sequence.fetch_add(1, Ordering::Relaxed);
        let signal_id = format!(
            "lashlang:{}:signal.{}:{sequence}",
            self.process_id, signal.name
        );
        self.ctx
            .signal_lashlang_process(
                Arc::clone(&self.registry),
                &target,
                &signal.name,
                signal_id,
                payload,
            )
            .await
            .map_err(|err| ::lashlang::ExecutionHostError::new(err.to_string()))?;
        Ok(::lashlang::Value::Null)
    }
}

impl ::lashlang::ExecutionHost for LashlangProcessHost<'_> {
    async fn perform(
        &self,
        op: ::lashlang::AbilityOp,
    ) -> Result<::lashlang::AbilityResult, ::lashlang::ExecutionHostError> {
        match op {
            ::lashlang::AbilityOp::ResourceOperation(operation) => self
                .resource_operation(
                    operation.operation,
                    operation.receiver,
                    operation.args,
                    operation.call_site,
                )
                .await
                .map(::lashlang::AbilityResult::Value),
            ::lashlang::AbilityOp::Await(handle) => self
                .await_handle(handle)
                .await
                .map(::lashlang::AbilityResult::Value),
            ::lashlang::AbilityOp::Cancel(handle) => self
                .cancel_handle(handle)
                .await
                .map(::lashlang::AbilityResult::Value),
            ::lashlang::AbilityOp::StartProcess(start) => self
                .start_process(*start)
                .await
                .map(::lashlang::AbilityResult::Value),
            ::lashlang::AbilityOp::ProcessEvent(event) => {
                self.process_event(event).await?;
                Ok(::lashlang::AbilityResult::Unit)
            }
            ::lashlang::AbilityOp::Sleep(sleep) => self
                .sleep(sleep)
                .await
                .map(::lashlang::AbilityResult::Value),
            ::lashlang::AbilityOp::WaitSignal { name } => self
                .wait_signal(name)
                .await
                .map(::lashlang::AbilityResult::Value),
            ::lashlang::AbilityOp::SignalRun(signal) => self
                .signal_run(signal)
                .await
                .map(::lashlang::AbilityResult::Value),
            ::lashlang::AbilityOp::Print(_) => Err(::lashlang::ExecutionHostError::new(
                "`print` is not available inside lashlang process bodies",
            )),
            ::lashlang::AbilityOp::Submit(value)
            | ::lashlang::AbilityOp::Finish(value)
            | ::lashlang::AbilityOp::Fail(value) => Ok(::lashlang::AbilityResult::Value(value)),
        }
    }

    async fn yield_now(&self) {
        tokio::task::yield_now().await;
    }

    fn observe_lashlang_execution(&self, observation: ::lashlang::LashlangExecutionObservation) {
        let identity = self.execution_identity();
        let event = match observation {
            ::lashlang::LashlangExecutionObservation::NodeStarted { site, occurrence } => {
                TraceLashlangExecutionEvent::NodeStarted {
                    event_key: lashlang_execution_event_key(
                        &self.process_id,
                        format!("node:{}:{occurrence}:started", site.node_id),
                    ),
                    identity: identity.clone(),
                    node_id: site.node_id,
                    node_kind: site.node_kind,
                    label: site.label,
                    occurrence,
                }
            }
            ::lashlang::LashlangExecutionObservation::NodeCompleted { site, occurrence } => {
                TraceLashlangExecutionEvent::NodeCompleted {
                    event_key: lashlang_execution_event_key(
                        &self.process_id,
                        format!("node:{}:{occurrence}:completed", site.node_id),
                    ),
                    identity: identity.clone(),
                    node_id: site.node_id,
                    node_kind: site.node_kind,
                    label: site.label,
                    occurrence,
                }
            }
            ::lashlang::LashlangExecutionObservation::NodeFailed {
                site,
                occurrence,
                error,
            } => TraceLashlangExecutionEvent::NodeFailed {
                event_key: lashlang_execution_event_key(
                    &self.process_id,
                    format!("node:{}:{occurrence}:failed", site.node_id),
                ),
                identity: identity.clone(),
                node_id: site.node_id,
                node_kind: site.node_kind,
                label: site.label,
                occurrence,
                error,
            },
            ::lashlang::LashlangExecutionObservation::BranchSelected {
                site,
                occurrence,
                edge_id,
                selected,
            } => TraceLashlangExecutionEvent::BranchSelected {
                event_key: lashlang_execution_event_key(
                    &self.process_id,
                    format!("branch:{}:{occurrence}:{edge_id}", site.node_id),
                ),
                identity: identity.clone(),
                node_id: site.node_id,
                occurrence,
                edge_id,
                selected: match selected {
                    ::lashlang::ProcessBranchSelection::Then => TraceBranchSelection::Then,
                    ::lashlang::ProcessBranchSelection::Else => TraceBranchSelection::Else,
                },
            },
            ::lashlang::LashlangExecutionObservation::ChildStarted {
                site,
                occurrence,
                child,
            } => TraceLashlangExecutionEvent::ChildStarted {
                event_key: lashlang_execution_event_key(
                    &self.process_id,
                    format!("child:{}:{occurrence}:{}", site.node_id, child.process_id),
                ),
                identity,
                parent_node_id: site.node_id,
                occurrence,
                child: TraceLashlangChildExecution {
                    scope: TraceRuntimeScope::new(self.session_id.clone()),
                    subject: TraceRuntimeSubject::Process {
                        process_id: child.process_id,
                    },
                    module_ref: Some(child.module_ref.to_string()),
                    entry_ref: Some(::lashlang::process_ref_key(&child.process_ref)),
                    entry_name: Some(child.process_name),
                },
            },
        };
        let trace = LashlangExecutionTraceContext {
            sink: &self.lashlang_execution_sink,
            base_context: &self.lashlang_execution_context,
            session_id: &self.session_id,
        };
        emit_lashlang_execution_trace(trace, event);
    }
}

#[derive(Clone, Copy)]
struct LashlangExecutionTraceContext<'trace> {
    sink: &'trace Option<Arc<dyn TraceSink>>,
    base_context: &'trace TraceContext,
    session_id: &'trace str,
}

fn emit_process_started_trace(
    trace: LashlangExecutionTraceContext<'_>,
    process_id: &str,
    artifact: &::lashlang::ModuleArtifact,
    process_ref: &::lashlang::ProcessRef,
    process_name: &str,
) {
    if trace.sink.is_none() {
        return;
    }
    emit_lashlang_execution_trace(
        trace,
        TraceLashlangExecutionEvent::ExecutionStarted {
            event_key: lashlang_execution_event_key(process_id, "started"),
            identity: process_execution_identity(
                trace.session_id,
                process_id,
                &artifact.module_ref,
                process_ref,
                process_name,
            ),
            execution_map: trace_lashlang_process_map(artifact, process_ref, process_name),
        },
    );
}

fn emit_process_finished_trace(
    trace: LashlangExecutionTraceContext<'_>,
    process_id: &str,
    module_ref: &::lashlang::ModuleRef,
    process_ref: &::lashlang::ProcessRef,
    process_name: &str,
    output: &crate::ProcessAwaitOutput,
) {
    if trace.sink.is_none() {
        return;
    }
    let (status, error) = match output {
        crate::ProcessAwaitOutput::Success { .. } => (TraceLashlangStatus::Completed, None),
        crate::ProcessAwaitOutput::Failure { message, .. } => {
            (TraceLashlangStatus::Failed, Some(message.clone()))
        }
        crate::ProcessAwaitOutput::Cancelled { message, .. } => {
            (TraceLashlangStatus::Cancelled, Some(message.clone()))
        }
    };
    emit_lashlang_execution_trace(
        trace,
        TraceLashlangExecutionEvent::ExecutionFinished {
            event_key: lashlang_execution_event_key(process_id, "finished"),
            identity: process_execution_identity(
                trace.session_id,
                process_id,
                module_ref,
                process_ref,
                process_name,
            ),
            status,
            error,
        },
    );
}

fn emit_lashlang_execution_trace(
    trace: LashlangExecutionTraceContext<'_>,
    event: TraceLashlangExecutionEvent,
) {
    crate::trace::emit_trace(
        trace.sink,
        trace.base_context,
        TraceContext::default().for_session(trace.session_id.to_string()),
        TraceEvent::LashlangExecution { event },
    );
}

fn lashlang_execution_event_key(process_id: &str, suffix: impl std::fmt::Display) -> String {
    format!("lashlang_execution:{process_id}:{suffix}")
}

fn process_execution_identity(
    session_id: &str,
    process_id: &str,
    module_ref: &::lashlang::ModuleRef,
    process_ref: &::lashlang::ProcessRef,
    process_name: &str,
) -> TraceLashlangExecutionIdentity {
    TraceLashlangExecutionIdentity {
        scope: TraceRuntimeScope::new(session_id.to_string()),
        subject: TraceRuntimeSubject::Process {
            process_id: process_id.to_string(),
        },
        module_ref: module_ref.to_string(),
        entry_kind: "process".to_string(),
        entry_ref: Some(::lashlang::process_ref_key(process_ref)),
        entry_name: process_name.to_string(),
    }
}

fn trace_lashlang_process_map(
    artifact: &::lashlang::ModuleArtifact,
    process_ref: &::lashlang::ProcessRef,
    process_name: &str,
) -> TraceLashlangMap {
    let Some(map) = ::lashlang::map_lashlang_process(
        artifact,
        process_ref,
        ::lashlang::LashlangMapOptions {
            include_reachable_processes: true,
        },
    ) else {
        return TraceLashlangMap {
            module_ref: artifact.module_ref.to_string(),
            entry_kind: "process".to_string(),
            entry_ref: Some(::lashlang::process_ref_key(process_ref)),
            entry_name: process_name.to_string(),
            nodes: Vec::new(),
            edges: Vec::new(),
        };
    };
    TraceLashlangMap {
        module_ref: map.module_ref.to_string(),
        entry_kind: "process".to_string(),
        entry_ref: map.entry_ref.as_ref().map(::lashlang::process_ref_key),
        entry_name: process_name.to_string(),
        nodes: map
            .nodes
            .into_iter()
            .map(|node| TraceLashlangMapNode {
                id: node.id,
                kind: node.kind,
                label: node.label,
                label_metadata: node.label_metadata.map(|label| TraceLabelMetadata {
                    title: label.title.to_string(),
                    description: label.description.map(|description| description.to_string()),
                }),
            })
            .collect(),
        edges: map
            .edges
            .into_iter()
            .map(|edge| TraceLashlangMapEdge {
                id: edge.id,
                from: edge.from,
                to: edge.to,
                label: edge.label,
            })
            .collect(),
    }
}

fn protocol_reply_to_lashlang_value(
    reply: crate::ToolInvocationReply,
) -> Result<::lashlang::Value, ::lashlang::ExecutionHostError> {
    crate::lashlang_bridge::protocol_tool_reply_to_lashlang_value(reply)
}

fn process_id_from_lashlang_handle(
    handle: &::lashlang::Value,
) -> Result<String, ::lashlang::ExecutionHostError> {
    let value = crate::lashlang_bridge::lashlang_value_to_json(handle)?;
    let kind = value
        .get("__handle__")
        .and_then(|value| value.as_str())
        .ok_or_else(|| {
            ::lashlang::ExecutionHostError::new("signal_run expects a process handle")
        })?;
    if kind != "process" {
        return Err(::lashlang::ExecutionHostError::new(format!(
            "signal_run expects a process handle, got `{kind}`"
        )));
    }
    value
        .get("id")
        .and_then(|value| value.as_str())
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .ok_or_else(|| {
            ::lashlang::ExecutionHostError::new("signal_run process handle is missing `id`")
        })
}

fn lashlang_surface_satisfies_requirements(
    required: &::lashlang::SurfaceRequirements,
    current: &::lashlang::LashlangSurface,
) -> Result<(), String> {
    let abilities = required.abilities;
    let current_abilities = current.abilities;
    if abilities.processes && !current_abilities.processes {
        return Err("processes are not available".to_string());
    }
    if abilities.sleep && !current_abilities.sleep {
        return Err("sleep is not available".to_string());
    }
    if abilities.process_signals && !current_abilities.process_signals {
        return Err("process signals are not available".to_string());
    }
    if abilities.triggers && !current_abilities.triggers {
        return Err("triggers are not available".to_string());
    }
    if required.language_features.label_annotations && !current.language_features.label_annotations
    {
        return Err("label annotations are not available".to_string());
    }

    for (_, module) in required.resources.module_instances() {
        let current_module = current
            .resources
            .resolve_module_path(&module.path)
            .ok_or_else(|| format!("module `{}` is not available", module.alias))?;
        if current_module.resource_type != module.resource_type {
            return Err(format!(
                "module `{}` has type `{}`, expected `{}`",
                module.alias, current_module.resource_type, module.resource_type
            ));
        }
        for (operation, required_binding) in &module.operations {
            match current.resources.resolve_module_operation(
                &module.resource_type,
                &module.alias,
                operation,
            ) {
                Some(current_binding) if current_binding == required_binding => {}
                Some(current_binding) => {
                    return Err(format!(
                        "module `{}` operation `{operation}` resolves to `{}`, expected `{}`",
                        module.alias,
                        current_binding.host_operation,
                        required_binding.host_operation
                    ));
                }
                None => {
                    return Err(format!(
                        "module `{}` does not expose operation `{operation}`",
                        module.alias
                    ));
                }
            }
        }
    }

    for (resource_type, required_type) in required.resources.resource_types() {
        if !current.resources.has_resource_type(resource_type) {
            return Err(format!("resource type `{resource_type}` is not available"));
        }
        for (operation, required_binding) in &required_type.operations {
            let current_binding = current
                .resources
                .resolve_operation(resource_type, operation)
                .ok_or_else(|| {
                    format!(
                        "resource type `{resource_type}` does not expose operation `{operation}`"
                    )
                })?;
            if current_binding.input_ty != required_binding.input_ty {
                return Err(format!(
                    "resource type `{resource_type}` operation `{operation}` has incompatible input type"
                ));
            }
            if current_binding.output_ty != required_binding.output_ty {
                return Err(format!(
                    "resource type `{resource_type}` operation `{operation}` has incompatible output type"
                ));
            }
        }
    }
    for (name, required_data_type) in required.resources.named_data_types() {
        let current_data_type = current
            .resources
            .resolve_named_data_type(name)
            .ok_or_else(|| format!("host data type `{name}` is not available"))?;
        if current_data_type != required_data_type {
            return Err(format!(
                "host data type `{name}` has incompatible structure"
            ));
        }
    }
    for (path, required_binding) in required.resources.value_constructors() {
        let current_binding = current
            .resources
            .resolve_value_constructor(&path.split('.').collect::<Vec<_>>())
            .ok_or_else(|| format!("value constructor `{path}` is not available"))?;
        if current_binding.input_ty != required_binding.input_ty {
            return Err(format!(
                "value constructor `{path}` has incompatible input type"
            ));
        }
        if current_binding.output_ty != required_binding.output_ty {
            return Err(format!(
                "value constructor `{path}` has incompatible output type"
            ));
        }
    }
    for (source_ty, required_binding) in required.resources.trigger_sources() {
        let current_binding = current
            .resources
            .resolve_trigger_source(source_ty)
            .ok_or_else(|| format!("trigger source type `{source_ty}` is not available"))?;
        if current_binding != required_binding {
            return Err(format!(
                "trigger source type `{source_ty}` has incompatible event type"
            ));
        }
    }

    Ok(())
}

fn process_lashlang_execution_result(
    result: Result<::lashlang::ExecutionOutcome, ::lashlang::RuntimeError>,
) -> crate::ProcessAwaitOutput {
    match result {
        Ok(::lashlang::ExecutionOutcome::Finished(value)) => crate::ProcessAwaitOutput::Success {
            value: crate::lashlang_bridge::lashlang_value_to_json(&value)
                .unwrap_or(serde_json::Value::Null),
            control: None,
        },
        Ok(::lashlang::ExecutionOutcome::Failed(value)) => process_lashlang_failure(
            "process_failed",
            value.to_string(),
            Some(
                crate::lashlang_bridge::lashlang_value_to_json(&value)
                    .unwrap_or(serde_json::Value::Null),
            ),
        ),
        Ok(::lashlang::ExecutionOutcome::Continued) => crate::ProcessAwaitOutput::Success {
            value: serde_json::Value::Null,
            control: None,
        },
        Err(err) => process_lashlang_failure("process_runtime_error", err.to_string(), None),
    }
}

fn process_lashlang_failure(
    code: &str,
    message: String,
    raw: Option<serde_json::Value>,
) -> crate::ProcessAwaitOutput {
    crate::ProcessAwaitOutput::Failure {
        class: crate::ToolFailureClass::Execution,
        code: code.to_string(),
        message,
        raw,
        control: None,
    }
}

fn process_lashlang_cancelled(message: impl Into<String>) -> crate::ProcessAwaitOutput {
    crate::ProcessAwaitOutput::Cancelled {
        message: message.into(),
        raw: None,
        control: None,
    }
}
