use super::*;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use lash_trace::{
    TraceBranchSelection, TraceContext, TraceEvent, TraceProcessMap, TraceProcessMapEdge,
    TraceProcessMapNode, TraceProcessStatus, TraceProcessTrackingEvent, TraceSink,
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
        cancellation: tokio_util::sync::CancellationToken,
    ) -> crate::ProcessAwaitOutput {
        let artifact = match self
            .current
            .host
            .core
            .durability
            .lashlang_artifact_store
            .get_module_artifact(&module_ref)
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
        let process_tracking_sink = self.current.host.core.tracing.process_tracking_sink.clone();
        let process_tracking_context = self.current.host.core.tracing.trace_context.clone();
        let process_tracking_trace = ProcessTrackingTraceContext {
            sink: &process_tracking_sink,
            base_context: &process_tracking_context,
            session_id: &self.current.session_id,
        };
        emit_process_started_trace(
            process_tracking_trace,
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
            Arc::clone(&self.current.host.core.durability.lashlang_artifact_store),
            Arc::clone(&self.current.host.core.durability.attachment_store),
            Arc::new(crate::ChronologicalProjection::default()),
            None,
            crate::TurnContext::default(),
        )
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
            process_tracking_sink: process_tracking_sink.clone(),
            process_tracking_context: process_tracking_context.clone(),
            wake_target_scope: execution_context.wake_target_scope,
            store: self.current.store.clone(),
            cancellation: cancellation.clone(),
            sleep_sequence: AtomicU64::new(0),
            event_sequence: AtomicU64::new(0),
            signal_sequence: tokio::sync::Mutex::new(0),
        };
        let env = ::lashlang::ExecutionEnvironment::new(&host).process();

        let output = {
            tokio::select! {
                _ = cancellation.cancelled() => process_lashlang_cancelled("lashlang process was cancelled"),
                result = ::lashlang::execute(compiled.as_ref(), &mut state, &env) => {
                    process_lashlang_execution_result(result)
                }
            }
        };
        drop(env);
        drop(host);
        run_context.shutdown().await;
        emit_process_finished_trace(
            process_tracking_trace,
            &registration.id,
            &module_ref,
            &process_ref,
            &process_name,
            &output,
        );
        output
    }
}

struct LashlangProcessHost<'run> {
    ctx: crate::RuntimeExecutionContext<'run>,
    registry: Arc<dyn crate::ProcessRegistry>,
    process_id: String,
    session_id: String,
    module_ref: ::lashlang::ModuleRef,
    process_ref: ::lashlang::ProcessRef,
    process_name: String,
    process_tracking_sink: Option<Arc<dyn TraceSink>>,
    process_tracking_context: TraceContext,
    wake_target_scope: Option<crate::ProcessScope>,
    store: Option<Arc<dyn crate::RuntimePersistence>>,
    cancellation: tokio_util::sync::CancellationToken,
    sleep_sequence: AtomicU64,
    /// Per-execution ordinal for wake/yield emissions. Deterministic replay
    /// re-issues `process_event` calls in the same order, so the Nth emission
    /// gets the same ordinal — and thus the same replay key — across a
    /// crash-recovery re-run, making the append idempotent on redelivery.
    event_sequence: AtomicU64,
    signal_sequence: tokio::sync::Mutex<u64>,
}

impl LashlangProcessHost<'_> {
    fn resource_payload(
        &self,
        _receiver: &::lashlang::Value,
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
}

impl LashlangProcessHost<'_> {
    async fn resource_operation(
        &self,
        operation: String,
        receiver: ::lashlang::Value,
        args: Vec<::lashlang::Value>,
    ) -> Result<::lashlang::Value, ::lashlang::ExecutionHostError> {
        if !matches!(&receiver, ::lashlang::Value::Resource(_)) {
            return Err(::lashlang::ExecutionHostError::new(format!(
                "module operation `{operation}` requires a module authority receiver"
            )));
        }
        if operation.starts_with("triggers.") {
            let payload = self.resource_payload(&receiver, &args)?;
            let value = self
                .ctx
                .perform_lashlang_trigger_operation(&operation, payload)
                .await
                .map_err(::lashlang::ExecutionHostError::new)?;
            return Ok(::lashlang::from_json(value));
        }
        let manifest = self.ctx.callable_tool_manifest(&operation).ok_or_else(|| {
            ::lashlang::ExecutionHostError::new(format!(
                "module operation `{operation}` is unavailable in this session"
            ))
        })?;
        let reply = self
            .ctx
            .call_tool(
                uuid::Uuid::new_v4().to_string(),
                manifest.name.clone(),
                self.resource_payload(&receiver, &args)?,
                0,
            )
            .await;
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
                .with_replay_key(format!("process:{}:event:{ordinal}", self.process_id))
                .with_optional_wake_target_scope(self.wake_target_scope.clone()),
            )
            .await
            .map_err(|err| ::lashlang::ExecutionHostError::new(err.to_string()))?;
        crate::tool_provider::process_events::enqueue_wake_delivery(
            self.store.as_deref(),
            result.wake_delivery,
            Some(self.ctx.session_graph_service()),
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

    async fn wait_signal(&self) -> Result<::lashlang::Value, ::lashlang::ExecutionHostError> {
        let after_sequence = *self.signal_sequence.lock().await;
        let wait =
            self.registry
                .wait_event_after(&self.process_id, "process.signal", after_sequence);
        let event = tokio::select! {
            _ = self.cancellation.cancelled() => {
                return Err(::lashlang::ExecutionHostError::new("wait signal was cancelled"));
            }
            event = wait => event.map_err(|err| ::lashlang::ExecutionHostError::new(err.to_string()))?,
        };
        *self.signal_sequence.lock().await = event.sequence;
        Ok(::lashlang::from_json(
            event
                .payload
                .get("payload")
                .cloned()
                .unwrap_or(event.payload),
        ))
    }

    async fn signal_run(
        &self,
        signal: ::lashlang::ProcessSignal,
    ) -> Result<::lashlang::Value, ::lashlang::ExecutionHostError> {
        let target = process_id_from_lashlang_handle(&signal.run)?;
        let payload = crate::lashlang_bridge::lashlang_value_to_json(&signal.payload)?;
        self.registry
            .append_event(
                &target,
                crate::ProcessEventAppendRequest::new(
                    "process.signal",
                    serde_json::json!({
                        "payload": payload,
                        "timestamp": chrono::Utc::now().to_rfc3339(),
                    }),
                ),
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
                .resource_operation(operation.operation, operation.receiver, operation.args)
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
            ::lashlang::AbilityOp::WaitSignal => self
                .wait_signal()
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

    fn observe_process_tracking(&self, observation: ::lashlang::ProcessTrackingObservation) {
        let event = match observation {
            ::lashlang::ProcessTrackingObservation::NodeStarted { site, occurrence } => {
                TraceProcessTrackingEvent::NodeStarted {
                    event_key: process_tracking_event_key(
                        &self.process_id,
                        format!("node:{}:{occurrence}:started", site.node_id),
                    ),
                    process_id: self.process_id.clone(),
                    session_id: self.session_id.clone(),
                    module_ref: self.module_ref.to_string(),
                    process_ref: ::lashlang::process_ref_key(&self.process_ref),
                    process_name: self.process_name.clone(),
                    node_id: site.node_id,
                    node_kind: site.node_kind,
                    label: site.label,
                    occurrence,
                }
            }
            ::lashlang::ProcessTrackingObservation::NodeCompleted { site, occurrence } => {
                TraceProcessTrackingEvent::NodeCompleted {
                    event_key: process_tracking_event_key(
                        &self.process_id,
                        format!("node:{}:{occurrence}:completed", site.node_id),
                    ),
                    process_id: self.process_id.clone(),
                    session_id: self.session_id.clone(),
                    module_ref: self.module_ref.to_string(),
                    process_ref: ::lashlang::process_ref_key(&self.process_ref),
                    process_name: self.process_name.clone(),
                    node_id: site.node_id,
                    node_kind: site.node_kind,
                    label: site.label,
                    occurrence,
                }
            }
            ::lashlang::ProcessTrackingObservation::NodeFailed {
                site,
                occurrence,
                error,
            } => TraceProcessTrackingEvent::NodeFailed {
                event_key: process_tracking_event_key(
                    &self.process_id,
                    format!("node:{}:{occurrence}:failed", site.node_id),
                ),
                process_id: self.process_id.clone(),
                session_id: self.session_id.clone(),
                module_ref: self.module_ref.to_string(),
                process_ref: ::lashlang::process_ref_key(&self.process_ref),
                process_name: self.process_name.clone(),
                node_id: site.node_id,
                node_kind: site.node_kind,
                label: site.label,
                occurrence,
                error,
            },
            ::lashlang::ProcessTrackingObservation::BranchSelected {
                site,
                occurrence,
                edge_id,
                selected,
            } => TraceProcessTrackingEvent::BranchSelected {
                event_key: process_tracking_event_key(
                    &self.process_id,
                    format!("branch:{}:{occurrence}:{edge_id}", site.node_id),
                ),
                process_id: self.process_id.clone(),
                session_id: self.session_id.clone(),
                module_ref: self.module_ref.to_string(),
                process_ref: ::lashlang::process_ref_key(&self.process_ref),
                process_name: self.process_name.clone(),
                node_id: site.node_id,
                occurrence,
                edge_id,
                selected: match selected {
                    ::lashlang::ProcessBranchSelection::Then => TraceBranchSelection::Then,
                    ::lashlang::ProcessBranchSelection::Else => TraceBranchSelection::Else,
                },
            },
            ::lashlang::ProcessTrackingObservation::ChildStarted {
                site,
                occurrence,
                child,
            } => TraceProcessTrackingEvent::ChildStarted {
                event_key: process_tracking_event_key(
                    &self.process_id,
                    format!("child:{}:{occurrence}:{}", site.node_id, child.process_id),
                ),
                process_id: self.process_id.clone(),
                session_id: self.session_id.clone(),
                module_ref: self.module_ref.to_string(),
                process_ref: ::lashlang::process_ref_key(&self.process_ref),
                process_name: self.process_name.clone(),
                parent_process_id: self.process_id.clone(),
                parent_node_id: site.node_id,
                occurrence,
                child_process_id: child.process_id,
                child_module_ref: child.module_ref.to_string(),
                child_process_ref: ::lashlang::process_ref_key(&child.process_ref),
                child_process_name: child.process_name,
            },
        };
        let trace = ProcessTrackingTraceContext {
            sink: &self.process_tracking_sink,
            base_context: &self.process_tracking_context,
            session_id: &self.session_id,
        };
        emit_process_tracking_trace(trace, event);
    }
}

#[derive(Clone, Copy)]
struct ProcessTrackingTraceContext<'trace> {
    sink: &'trace Option<Arc<dyn TraceSink>>,
    base_context: &'trace TraceContext,
    session_id: &'trace str,
}

fn emit_process_started_trace(
    trace: ProcessTrackingTraceContext<'_>,
    process_id: &str,
    artifact: &::lashlang::ModuleArtifact,
    process_ref: &::lashlang::ProcessRef,
    process_name: &str,
) {
    if trace.sink.is_none() {
        return;
    }
    emit_process_tracking_trace(
        trace,
        TraceProcessTrackingEvent::ProcessStarted {
            event_key: process_tracking_event_key(process_id, "started"),
            process_id: process_id.to_string(),
            session_id: trace.session_id.to_string(),
            module_ref: artifact.module_ref.to_string(),
            process_ref: ::lashlang::process_ref_key(process_ref),
            process_name: process_name.to_string(),
            process_map: trace_process_map(artifact, process_ref, process_name),
        },
    );
}

fn emit_process_finished_trace(
    trace: ProcessTrackingTraceContext<'_>,
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
        crate::ProcessAwaitOutput::Success { .. } => (TraceProcessStatus::Completed, None),
        crate::ProcessAwaitOutput::Failure { message, .. } => {
            (TraceProcessStatus::Failed, Some(message.clone()))
        }
        crate::ProcessAwaitOutput::Cancelled { message, .. } => {
            (TraceProcessStatus::Cancelled, Some(message.clone()))
        }
    };
    emit_process_tracking_trace(
        trace,
        TraceProcessTrackingEvent::ProcessFinished {
            event_key: process_tracking_event_key(process_id, "finished"),
            process_id: process_id.to_string(),
            session_id: trace.session_id.to_string(),
            module_ref: module_ref.to_string(),
            process_ref: ::lashlang::process_ref_key(process_ref),
            process_name: process_name.to_string(),
            status,
            error,
        },
    );
}

fn emit_process_tracking_trace(
    trace: ProcessTrackingTraceContext<'_>,
    event: TraceProcessTrackingEvent,
) {
    crate::trace::emit_trace(
        trace.sink,
        trace.base_context,
        TraceContext::default().for_session(trace.session_id.to_string()),
        TraceEvent::ProcessTracking { event },
    );
}

fn process_tracking_event_key(process_id: &str, suffix: impl std::fmt::Display) -> String {
    format!("process_tracking:{process_id}:{suffix}")
}

fn trace_process_map(
    artifact: &::lashlang::ModuleArtifact,
    process_ref: &::lashlang::ProcessRef,
    process_name: &str,
) -> TraceProcessMap {
    let Some(map) = ::lashlang::map_process(
        artifact,
        process_ref,
        ::lashlang::ProcessMapOptions {
            include_reachable_processes: true,
        },
    ) else {
        return TraceProcessMap {
            module_ref: artifact.module_ref.to_string(),
            process_ref: ::lashlang::process_ref_key(process_ref),
            process_name: process_name.to_string(),
            nodes: Vec::new(),
            edges: Vec::new(),
        };
    };
    TraceProcessMap {
        module_ref: map.module_ref.to_string(),
        process_ref: ::lashlang::process_ref_key(&map.process_ref),
        process_name: process_name.to_string(),
        nodes: map
            .nodes
            .into_iter()
            .map(|node| TraceProcessMapNode {
                id: node.id,
                kind: node.kind,
                label: node.label,
            })
            .collect(),
        edges: map
            .edges
            .into_iter()
            .map(|edge| TraceProcessMapEdge {
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
            ::lashlang::ExecutionHostError::new("signal run expects a process handle")
        })?;
    if kind != "process" {
        return Err(::lashlang::ExecutionHostError::new(format!(
            "signal run expects a process handle, got `{kind}`"
        )));
    }
    value
        .get("id")
        .and_then(|value| value.as_str())
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .ok_or_else(|| {
            ::lashlang::ExecutionHostError::new("signal run process handle is missing `id`")
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

    for module in required.resources.module_instances.values() {
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
    }

    for (resource_type, required_type) in &required.resources.resource_types {
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
            if current_binding.host_operation != required_binding.host_operation {
                return Err(format!(
                    "resource type `{resource_type}` operation `{operation}` resolves to `{}`, expected `{}`",
                    current_binding.host_operation, required_binding.host_operation
                ));
            }
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
    for (path, required_binding) in &required.resources.value_constructors {
        let current_binding = current
            .resources
            .value_constructors
            .get(path)
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
    for (source_ty, required_binding) in &required.resources.trigger_sources {
        let current_binding = current
            .resources
            .trigger_sources
            .get(source_ty)
            .ok_or_else(|| format!("trigger source type `{source_ty}` is not available"))?;
        if current_binding.event_ty != required_binding.event_ty {
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
