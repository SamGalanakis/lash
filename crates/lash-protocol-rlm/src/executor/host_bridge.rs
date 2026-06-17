use std::collections::{BTreeMap, HashSet};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use lash_core::{
    AttachmentRef, ExecImage, RuntimeExecutionContext, TextProjectionMetadata,
    ToolChildExecutionTraceHook, ToolInvocationReply, TraceBranchSelection, TraceContext,
    TraceEvent, TraceRecord, TraceRuntimeSubject, TraceSink,
};
use lash_lashlang_runtime::{
    LASHLANG_ENGINE_KIND, LashlangProcessInput, TraceLashlangChildExecution,
    TraceLashlangExecutionEvent, TraceLashlangExecutionIdentity, lashlang_process_event_types,
    lashlang_process_signal_event_types, lashlang_type_expr_schema, lashlang_value_to_json,
    prepare_lashlang_process_start, protocol_tool_output_to_lashlang_value,
    resolve_lashlang_module_operation, sleep_duration_ms,
};
use lash_plugin_tool_output_budget::{ToolOutputBudgetConfig, project_observation_text};
use lashlang::{
    AbilityOp, AbilityResult, ExecutionHost, ExecutionHostError, ProcessSignal, ProcessStart,
    ProjectedFuture, Record as FlowRecord, Sleep, Value as FlowValue,
};
use serde_json::Value;

use crate::projection::{flow_to_json_value, format_output_value};

pub(super) struct HostBridge<'run> {
    ctx: RuntimeExecutionContext<'run>,
    observe_projection: ToolOutputBudgetConfig,
    tool_result_projectors: Vec<crate::RlmToolResultProjector>,
    observations: Mutex<Vec<String>>,
    observation_truncation: Mutex<Vec<TextProjectionMetadata>>,
    printed_images: Mutex<Vec<AttachmentRef>>,
    tool_calls: Mutex<Vec<lash_core::ToolCallRecord>>,
    tool_images: Mutex<Vec<ExecImage>>,
    next_tool_index: Mutex<usize>,
    sleep_sequence: AtomicU64,
    lashlang_execution_trace: Option<LashlangExecutionTrace>,
    host_environment: lashlang::LashlangHostEnvironment,
    artifact_store: std::sync::Arc<dyn lashlang::LashlangArtifactStore>,
}

impl<'run> HostBridge<'run> {
    pub(super) fn new(
        ctx: RuntimeExecutionContext<'run>,
        observe_projection: ToolOutputBudgetConfig,
        tool_result_projectors: Vec<crate::RlmToolResultProjector>,
        lashlang_execution_trace: Option<LashlangExecutionTrace>,
        host_environment: lashlang::LashlangHostEnvironment,
        artifact_store: std::sync::Arc<dyn lashlang::LashlangArtifactStore>,
    ) -> Self {
        Self {
            ctx,
            observe_projection,
            tool_result_projectors,
            observations: Mutex::new(Vec::new()),
            observation_truncation: Mutex::new(Vec::new()),
            printed_images: Mutex::new(Vec::new()),
            tool_calls: Mutex::new(Vec::new()),
            tool_images: Mutex::new(Vec::new()),
            next_tool_index: Mutex::new(0),
            sleep_sequence: AtomicU64::new(0),
            lashlang_execution_trace,
            host_environment,
            artifact_store,
        }
    }

    fn next_index(&self) -> usize {
        let mut guard = self
            .next_tool_index
            .lock()
            .expect("tool index lock poisoned");
        let next = *guard;
        *guard += 1;
        next
    }

    fn consume_reply(
        &self,
        tool_name: &str,
        reply: ToolInvocationReply,
    ) -> Result<FlowValue, ExecutionHostError> {
        let projected_tool_name = reply
            .record
            .as_ref()
            .map(|record| record.tool.as_str())
            .unwrap_or(tool_name)
            .to_string();
        if let Some(record) = reply.record {
            self.tool_calls
                .lock()
                .map_err(|_| ExecutionHostError::new("tool call buffer poisoned"))?
                .push(record);
        }
        if reply.output.is_success() {
            let value = reply.output.value_for_projection();
            for projector in &self.tool_result_projectors {
                if let Some(value) = projector(&projected_tool_name, &value) {
                    return Ok(value);
                }
            }
            protocol_tool_output_to_lashlang_value(&reply.output)
        } else {
            protocol_tool_output_to_lashlang_value(&reply.output)
        }
    }

    pub(super) fn into_collected(self) -> CollectedExecutionOutput {
        CollectedExecutionOutput {
            observations: self.observations.into_inner().unwrap_or_default(),
            observation_truncation: self.observation_truncation.into_inner().unwrap_or_default(),
            printed_images: self.printed_images.into_inner().unwrap_or_default(),
            tool_calls: self.tool_calls.into_inner().unwrap_or_default(),
            tool_images: self.tool_images.into_inner().unwrap_or_default(),
        }
    }
}

#[derive(Clone)]
pub(super) struct LashlangExecutionTrace {
    sink: std::sync::Arc<dyn TraceSink>,
    base_context: TraceContext,
    identity: TraceLashlangExecutionIdentity,
}

impl LashlangExecutionTrace {
    #[allow(dead_code)]
    pub(super) fn new(
        sink: std::sync::Arc<dyn TraceSink>,
        base_context: TraceContext,
        identity: TraceLashlangExecutionIdentity,
    ) -> Self {
        Self {
            sink,
            base_context,
            identity,
        }
    }

    pub(super) fn identity(&self) -> &TraceLashlangExecutionIdentity {
        &self.identity
    }

    pub(super) fn event_key(&self, suffix: impl std::fmt::Display) -> String {
        format!("lashlang_execution:{}:{suffix}", self.identity.graph_key())
    }

    pub(super) fn tool_child_execution_trace_hook(
        &self,
        call_site: lashlang::LashlangExecutionCallSite,
    ) -> ToolChildExecutionTraceHook {
        let trace = self.clone();
        let parent_node_id = call_site.site.node_id;
        let occurrence = call_site.occurrence;
        ToolChildExecutionTraceHook::new(move |started| {
            let child = TraceLashlangChildExecution {
                scope: trace.identity.scope.clone(),
                subject: TraceRuntimeSubject::Process {
                    process_id: started.process_id,
                },
                module_ref: None,
                entry_ref: None,
                entry_name: started.child_entry_name,
            };
            let child_graph_key = child.graph_key();
            trace.emit(TraceLashlangExecutionEvent::ChildStarted {
                event_key: format!(
                    "lashlang_execution:{}:child:{}:{}:{}",
                    trace.identity.graph_key(),
                    parent_node_id,
                    occurrence,
                    child_graph_key
                ),
                identity: trace.identity.clone(),
                parent_node_id: parent_node_id.clone(),
                occurrence,
                child,
            });
        })
    }

    pub(super) fn emit(&self, event: TraceLashlangExecutionEvent) {
        let mut context = self.base_context.clone();
        context.session_id = Some(self.identity.scope.session_id.clone());
        context.turn_id = self.identity.scope.turn_id.clone();
        context.turn_index = self.identity.scope.turn_index;
        context.protocol_iteration = self.identity.scope.protocol_iteration;
        if let TraceRuntimeSubject::Effect { effect_id, .. } = &self.identity.subject {
            context.effect_id = Some(effect_id.clone());
        }
        let _ = self.sink.append(&TraceRecord::new(
            context,
            TraceEvent::LashlangExecution { event },
        ));
    }
}

impl HostBridge<'_> {
    async fn resource_operation(
        &self,
        operation: String,
        receiver: FlowValue,
        args: Vec<FlowValue>,
        call_site: Option<lashlang::LashlangExecutionCallSite>,
    ) -> Result<FlowValue, ExecutionHostError> {
        let receiver = match &receiver {
            FlowValue::Resource(receiver) => receiver,
            _ => {
                return Err(ExecutionHostError::new(format!(
                    "module operation `{operation}` requires a module authority receiver"
                )));
            }
        };
        let host_operation =
            resolve_lashlang_module_operation(&self.host_environment, receiver, &operation)?;
        let mut payload = if let [FlowValue::Record(record)] = args.as_slice() {
            flow_record_json(record).await
        } else {
            serde_json::json!({
                "args": flow_values_to_json(&args).await,
            })
        };
        payload
            .as_object_mut()
            .ok_or_else(|| ExecutionHostError::new("module operation payload must be an object"))?;
        if let Some(trigger_operation) =
            lashlang::TriggerHostOperation::from_host_operation(&host_operation)
        {
            return self.trigger_operation(trigger_operation, payload).await;
        }
        let index = self.next_index();
        let call_id = uuid::Uuid::new_v4().to_string();
        let call_site = call_site.and_then(|call_site| {
            self.lashlang_execution_trace
                .as_ref()
                .map(|trace| trace.tool_child_execution_trace_hook(call_site))
        });
        let reply = if let Some(call_site) = call_site {
            self.ctx
                .call_tool_with_child_execution_trace_hook(
                    call_id,
                    host_operation.clone(),
                    payload,
                    index,
                    call_site,
                )
                .await
        } else {
            self.ctx
                .call_tool(call_id, host_operation.clone(), payload, index)
                .await
        };
        self.consume_reply(&host_operation, reply)
    }

    async fn trigger_operation(
        &self,
        operation: lashlang::TriggerHostOperation,
        payload: Value,
    ) -> Result<FlowValue, ExecutionHostError> {
        match operation {
            lashlang::TriggerHostOperation::Register => self.register_trigger(payload).await,
            lashlang::TriggerHostOperation::List => self.list_triggers(payload).await,
            lashlang::TriggerHostOperation::Cancel => self.cancel_trigger(payload).await,
        }
    }

    async fn register_trigger(&self, payload: Value) -> Result<FlowValue, ExecutionHostError> {
        let request = lashlang::TriggerRegistrationRequest::decode(&payload)
            .map_err(|err| ExecutionHostError::new(err.to_string()))?;
        let store = self.trigger_store()?;
        let artifact = self
            .artifact_store
            .get_module_artifact(&request.target.module_ref)
            .await
            .map_err(|err| {
                ExecutionHostError::new(format!("failed to load lashlang module artifact: {err}"))
            })?
            .ok_or_else(|| {
                ExecutionHostError::new(format!(
                    "missing lashlang module artifact `{}` for trigger target `{}`",
                    request.target.module_ref, request.target.process_name
                ))
            })?;
        let compatibility =
            lashlang::check_trigger_compatibility(lashlang::TriggerCompatibilityRequest {
                artifact: artifact.as_ref(),
                definition: &request.target,
                source_type: &request.source.source_type,
                inputs: &request.inputs,
            })
            .map_err(|err| ExecutionHostError::new(err.to_string()))?;
        let source_key = store
            .source_key_for_subscription(&request.source.source_type, &request.source.value)
            .await
            .map_err(|err| ExecutionHostError::new(err.to_string()))?;
        let target = trigger_target_process_input(&request.target).map_err(|err| {
            ExecutionHostError::new(format!("failed to encode trigger target: {err}"))
        })?;
        let target_identity = lashlang_process_identity_for_definition(&request.target);
        let process = artifact
            .canonical_ir
            .process(&request.target.process_name)
            .ok_or_else(|| {
                ExecutionHostError::new(format!(
                    "trigger target artifact `{}` is missing process `{}`",
                    request.target.module_ref, request.target.process_name
                ))
            })?;
        let event_types = lashlang_process_event_types()
            .into_iter()
            .chain(lashlang_process_signal_event_types(process))
            .collect::<Vec<_>>();
        let env_ref = self
            .ctx
            .captured_process_execution_env_ref()
            .await
            .map_err(|err| ExecutionHostError::new(err.to_string()))?;
        let record = store
            .register_subscription(lash_core::TriggerSubscriptionDraft {
                registrant: self.ctx.trigger_registration_originator(),
                env_ref,
                wake_target: self.ctx.trigger_registration_wake_target(),
                name: request.name.clone(),
                source_type: request.source.source_type.clone(),
                source_key,
                source: request.source.value.clone(),
                payload_schema: lash_core::LashSchema::new(lashlang_type_expr_schema(
                    &compatibility.resolved_event_type,
                )),
                target,
                target_identity,
                event_types,
                input_template: core_trigger_input_template(&request.inputs),
                target_label: Some(request.target.process_name.clone()),
            })
            .await
            .map_err(|err| ExecutionHostError::new(err.to_string()))?;
        Ok(lashlang::from_json(serde_json::json!({
            "type": "trigger_handle",
            "id": record.handle,
            "name": record.name,
            "source_type": record.source_type,
            "source_key": record.source_key,
        })))
    }

    async fn list_triggers(&self, payload: Value) -> Result<FlowValue, ExecutionHostError> {
        let request = lashlang::TriggerListRequest::decode(&payload)
            .map_err(|err| ExecutionHostError::new(err.to_string()))?;
        let store = self.trigger_store()?;
        let mut filter = lash_core::TriggerSubscriptionFilter::for_session(self.ctx.session_id());
        filter.name = request.name;
        filter.source_type = request.source_type;
        filter.enabled = request.enabled;
        filter.target = request
            .target
            .as_ref()
            .map(lashlang_process_definition_for_identity);
        let registrations = store
            .list_subscriptions(filter)
            .await
            .map_err(|err| ExecutionHostError::new(err.to_string()))?
            .iter()
            .map(lash_core::TriggerRegistration::from)
            .collect::<Vec<_>>();
        let value = serde_json::to_value(registrations).map_err(|err| {
            ExecutionHostError::new(format!("failed to encode trigger registrations: {err}"))
        })?;
        Ok(lashlang::from_json(value))
    }

    async fn cancel_trigger(&self, payload: Value) -> Result<FlowValue, ExecutionHostError> {
        let request = lashlang::TriggerCancelRequest::decode(&payload)
            .map_err(|err| ExecutionHostError::new(err.to_string()))?;
        let store = self.trigger_store()?;
        let cancelled = store
            .cancel_subscription(self.ctx.session_id(), &request.handle)
            .await
            .map_err(|err| ExecutionHostError::new(err.to_string()))?;
        Ok(FlowValue::Bool(cancelled))
    }

    fn trigger_store(
        &self,
    ) -> Result<std::sync::Arc<dyn lash_core::TriggerStore>, ExecutionHostError> {
        self.ctx
            .trigger_store()
            .ok_or_else(|| ExecutionHostError::new("trigger store is unavailable in this runtime"))
    }

    async fn start_process(&self, start: ProcessStart) -> Result<FlowValue, ExecutionHostError> {
        let prepared =
            prepare_lashlang_process_start(std::sync::Arc::clone(&self.artifact_store), start)
                .await
                .map_err(ExecutionHostError::new)?;
        let reply = self
            .ctx
            .start_child_process(prepared.registration, LASHLANG_ENGINE_KIND, prepared.label)
            .await;
        self.consume_reply("start_process", reply)
    }

    async fn await_handle(&self, handle: FlowValue) -> Result<FlowValue, ExecutionHostError> {
        let reply = self
            .ctx
            .await_tool_handle(
                uuid::Uuid::new_v4().to_string(),
                handle_to_json(&handle).await?,
            )
            .await;
        self.consume_reply("await_handle", reply)
    }

    async fn cancel_handle(&self, handle: FlowValue) -> Result<FlowValue, ExecutionHostError> {
        let reply = self
            .ctx
            .cancel_tool_handle(
                uuid::Uuid::new_v4().to_string(),
                handle_to_json(&handle).await?,
            )
            .await;
        self.consume_reply("cancel_handle", reply)
    }

    async fn signal_run(&self, signal: ProcessSignal) -> Result<FlowValue, ExecutionHostError> {
        let handle = handle_to_json(&signal.run).await?;
        let payload = handle_to_json(&signal.payload).await?;
        let reply = self
            .ctx
            .signal_tool_handle(
                uuid::Uuid::new_v4().to_string(),
                handle,
                signal.name,
                payload,
            )
            .await;
        self.consume_reply("signal_run", reply)?;
        // `signal_run` evaluates to null in the language; the appended event is
        // recorded as a tool call but not surfaced as the expression value.
        Ok(FlowValue::Null)
    }

    async fn print(&self, value: FlowValue) -> Result<(), ExecutionHostError> {
        let attachment_store = self.ctx.attachment_store();
        let images = collect_printed_images(&value, attachment_store.as_ref()).await?;
        let raw_text = format_output_value(&value).await;
        let (_projected_text, metadata) =
            project_observation_text(&raw_text, &self.observe_projection);
        self.observations
            .lock()
            .map_err(|_| ExecutionHostError::new("observation buffer poisoned"))?
            .push(raw_text);
        self.observation_truncation
            .lock()
            .map_err(|_| ExecutionHostError::new("observation metadata buffer poisoned"))?
            .push(metadata);
        if !images.is_empty() {
            self.printed_images
                .lock()
                .map_err(|_| ExecutionHostError::new("printed image buffer poisoned"))?
                .extend(images);
        }
        Ok(())
    }

    async fn sleep(&self, sleep: Sleep) -> Result<FlowValue, ExecutionHostError> {
        let duration_ms = sleep_duration_ms(sleep.kind, &sleep.value)?;
        let sequence = self.sleep_sequence.fetch_add(1, Ordering::Relaxed);
        self.ctx
            .sleep_process("foreground", sequence, duration_ms)
            .await
            .map_err(|err| ExecutionHostError::new(err.to_string()))?;
        Ok(FlowValue::Null)
    }
}

impl ExecutionHost for HostBridge<'_> {
    async fn perform(&self, op: AbilityOp) -> Result<AbilityResult, ExecutionHostError> {
        match op {
            AbilityOp::ResourceOperation(operation) => {
                let lashlang::ResourceOperation {
                    operation,
                    receiver,
                    args,
                    call_site,
                } = operation;
                self.resource_operation(operation, receiver, args, call_site)
                    .await
                    .map(AbilityResult::Value)
            }
            AbilityOp::StartProcess(start) => {
                self.start_process(*start).await.map(AbilityResult::Value)
            }
            AbilityOp::Await(handle) => self.await_handle(handle).await.map(AbilityResult::Value),
            AbilityOp::Cancel(handle) => self.cancel_handle(handle).await.map(AbilityResult::Value),
            AbilityOp::Print(value) => {
                self.print(value).await?;
                Ok(AbilityResult::Unit)
            }
            AbilityOp::Sleep(sleep) => self.sleep(sleep).await.map(AbilityResult::Value),
            AbilityOp::ProcessEvent(_) => Err(ExecutionHostError::new(
                "process events are only available inside lashlang process bodies",
            )),
            AbilityOp::WaitSignal { .. } => Err(ExecutionHostError::new(
                "`wait_signal` is only available inside lashlang process bodies",
            )),
            AbilityOp::SignalRun(signal) => self.signal_run(signal).await.map(AbilityResult::Value),
            AbilityOp::Submit(value) | AbilityOp::Finish(value) | AbilityOp::Fail(value) => {
                Ok(AbilityResult::Value(value))
            }
        }
    }

    async fn yield_now(&self) {
        tokio::task::yield_now().await;
    }

    fn observe_lashlang_execution(&self, observation: lashlang::LashlangExecutionObservation) {
        let Some(trace) = &self.lashlang_execution_trace else {
            return;
        };
        let identity = trace.identity().clone();
        let event = match observation {
            lashlang::LashlangExecutionObservation::NodeStarted { site, occurrence } => {
                TraceLashlangExecutionEvent::NodeStarted {
                    event_key: trace
                        .event_key(format!("node:{}:{occurrence}:started", site.node_id)),
                    identity,
                    node_id: site.node_id,
                    node_kind: site.node_kind,
                    label: site.label,
                    occurrence,
                }
            }
            lashlang::LashlangExecutionObservation::NodeCompleted { site, occurrence } => {
                TraceLashlangExecutionEvent::NodeCompleted {
                    event_key: trace
                        .event_key(format!("node:{}:{occurrence}:completed", site.node_id)),
                    identity,
                    node_id: site.node_id,
                    node_kind: site.node_kind,
                    label: site.label,
                    occurrence,
                }
            }
            lashlang::LashlangExecutionObservation::NodeFailed {
                site,
                occurrence,
                error,
            } => TraceLashlangExecutionEvent::NodeFailed {
                event_key: trace.event_key(format!("node:{}:{occurrence}:failed", site.node_id)),
                identity,
                node_id: site.node_id,
                node_kind: site.node_kind,
                label: site.label,
                occurrence,
                error,
            },
            lashlang::LashlangExecutionObservation::BranchSelected {
                site,
                occurrence,
                edge_id,
                selected,
            } => TraceLashlangExecutionEvent::BranchSelected {
                event_key: trace
                    .event_key(format!("branch:{}:{occurrence}:{edge_id}", site.node_id)),
                identity,
                node_id: site.node_id,
                occurrence,
                edge_id,
                selected: match selected {
                    lashlang::ProcessBranchSelection::Then => TraceBranchSelection::Then,
                    lashlang::ProcessBranchSelection::Else => TraceBranchSelection::Else,
                },
            },
            lashlang::LashlangExecutionObservation::ChildStarted {
                site,
                occurrence,
                child,
            } => TraceLashlangExecutionEvent::ChildStarted {
                event_key: trace.event_key(format!(
                    "child:{}:{occurrence}:{}",
                    site.node_id, child.process_id
                )),
                identity,
                parent_node_id: site.node_id,
                occurrence,
                child: TraceLashlangChildExecution {
                    scope: trace.identity().scope.clone(),
                    subject: TraceRuntimeSubject::Process {
                        process_id: child.process_id,
                    },
                    module_ref: Some(child.module_ref.to_string()),
                    entry_ref: Some(lashlang::process_ref_key(&child.process_ref)),
                    entry_name: Some(child.process_name),
                },
            },
        };
        trace.emit(event);
    }
}

async fn handle_to_json(value: &FlowValue) -> Result<Value, ExecutionHostError> {
    match value {
        FlowValue::Projected(_) => Ok(flow_to_json_value(value).await),
        _ => lashlang_value_to_json(value),
    }
}

fn lashlang_process_input_for_definition(
    definition: &lashlang::ProcessDefinitionIdentity,
) -> LashlangProcessInput {
    LashlangProcessInput {
        module_ref: definition.module_ref.clone(),
        process_ref: definition.process_ref.clone(),
        host_requirements_ref: definition.host_requirements_ref.clone(),
        process_name: definition.process_name.clone(),
        args: serde_json::Map::new(),
    }
}

fn lashlang_process_definition_for_identity(
    definition: &lashlang::ProcessDefinitionIdentity,
) -> serde_json::Value {
    lashlang_process_input_for_definition(definition).definition()
}

fn lashlang_process_identity_for_definition(
    definition: &lashlang::ProcessDefinitionIdentity,
) -> lash_core::ProcessIdentity {
    lash_core::ProcessIdentity::new(LASHLANG_ENGINE_KIND)
        .with_label(Some(definition.process_name.clone()))
        .with_definition(Some(lashlang_process_definition_for_identity(definition)))
}

fn trigger_target_process_input(
    definition: &lashlang::ProcessDefinitionIdentity,
) -> Result<lash_core::ProcessInput, serde_json::Error> {
    lashlang_process_input_for_definition(definition).into_process_input()
}

fn core_trigger_input_template(
    input: &lashlang::TriggerInputTemplate,
) -> BTreeMap<String, lash_core::TriggerInputBinding> {
    input
        .entries()
        .map(|(name, binding)| {
            let binding = match binding {
                lashlang::TriggerInputBinding::Event => lash_core::TriggerInputBinding::Event,
                lashlang::TriggerInputBinding::Fixed { value } => {
                    lash_core::TriggerInputBinding::Fixed {
                        value: value.clone(),
                    }
                }
            };
            (name.to_string(), binding)
        })
        .collect()
}

fn flow_values_to_json<'a>(values: &'a [FlowValue]) -> ProjectedFuture<'a, Vec<Value>> {
    Box::pin(async move {
        let mut out = Vec::with_capacity(values.len());
        for value in values {
            out.push(flow_to_json_value(value).await);
        }
        out
    })
}

fn flow_record_json<'a>(record: &'a FlowRecord) -> ProjectedFuture<'a, Value> {
    Box::pin(async move {
        let mut object = serde_json::Map::with_capacity(record.len());
        for (key, value) in record.iter() {
            object.insert(key.to_string(), flow_to_json_value(value).await);
        }
        Value::Object(object)
    })
}

pub(super) struct CollectedExecutionOutput {
    pub(super) observations: Vec<String>,
    pub(super) observation_truncation: Vec<TextProjectionMetadata>,
    pub(super) printed_images: Vec<AttachmentRef>,
    pub(super) tool_calls: Vec<lash_core::ToolCallRecord>,
    pub(super) tool_images: Vec<ExecImage>,
}

async fn collect_printed_images(
    value: &FlowValue,
    attachment_store: &dyn lash_core::AttachmentStore,
) -> Result<Vec<AttachmentRef>, ExecutionHostError> {
    let mut seen = HashSet::new();
    let mut images = Vec::new();
    collect_printed_images_inner(value, attachment_store, &mut seen, &mut images).await?;
    Ok(images)
}

fn collect_printed_images_inner<'a>(
    value: &'a FlowValue,
    attachment_store: &'a dyn lash_core::AttachmentStore,
    seen: &'a mut HashSet<String>,
    images: &'a mut Vec<AttachmentRef>,
) -> ProjectedFuture<'a, Result<(), ExecutionHostError>> {
    Box::pin(async move {
        match value {
            FlowValue::Image(image) => {
                if !seen.insert(image.id.clone()) {
                    return Ok(());
                }
                let reference = attachment_store
                    .get(&lash_core::AttachmentId::new(image.id.clone()))
                    .await
                    .ok()
                    .map(|stored| stored.meta.as_ref())
                    .ok_or_else(|| {
                        ExecutionHostError::new(format!(
                            "image bytes for `{}` are unavailable or were pruned",
                            image.id
                        ))
                    })?;
                images.push(reference);
            }
            FlowValue::List(values) => {
                for value in values.iter() {
                    collect_printed_images_inner(value, attachment_store, seen, images).await?;
                }
            }
            FlowValue::Record(record) => {
                for (_, value) in record.iter() {
                    collect_printed_images_inner(value, attachment_store, seen, images).await?;
                }
            }
            FlowValue::Projected(value) => {
                collect_printed_images_inner(
                    &value.materialize_async().await,
                    attachment_store,
                    seen,
                    images,
                )
                .await?;
            }
            FlowValue::Null
            | FlowValue::Bool(_)
            | FlowValue::Number(_)
            | FlowValue::String(_)
            | FlowValue::Resource(_) => {}
        }
        Ok(())
    })
}
