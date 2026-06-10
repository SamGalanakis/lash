use std::sync::Arc;

use tokio::sync::mpsc::Sender;
use tokio_util::sync::CancellationToken;

use crate::tool_dispatch::ToolDispatchContext;
use crate::{TurnActivity, TurnActivityId, TurnEvent};

pub(crate) fn lashlang_surface_from_tool_surface(
    surface: &crate::ToolSurface,
    abilities: lashlang::LashlangAbilities,
    language_features: lashlang::LashlangLanguageFeatures,
    host_resources: lashlang::ResourceCatalog,
) -> lashlang::LashlangSurface {
    let mut resources = lashlang_resources_from_tool_surface(surface);
    resources.extend(host_resources);
    lashlang::LashlangSurface::new(resources, abilities).with_language_features(language_features)
}

pub(crate) fn lashlang_resources_from_tool_surface(
    surface: &crate::ToolSurface,
) -> lashlang::ResourceCatalog {
    let mut catalog = lashlang::ResourceCatalog::new();
    for entry in surface.tools.iter() {
        if entry.availability.is_callable() {
            let agent_surface = entry
                .manifest
                .agent_surface
                .executable_for(&entry.manifest.name);
            catalog.add_module_operation(
                agent_surface.module_path.iter().map(String::as_str),
                agent_surface.authority_type.clone(),
                agent_surface.operation.clone(),
                entry.manifest.name.clone(),
                lashlang::TypeExpr::Any,
                lashlang::TypeExpr::Any,
            );
        }
    }
    catalog
}

#[derive(Clone)]
pub struct RuntimeExecutionContext<'run> {
    pub(super) session_id: String,
    pub(super) dispatch: Arc<ToolDispatchContext<'run>>,
    lashlang_abilities: lashlang::LashlangAbilities,
    lashlang_language_features: lashlang::LashlangLanguageFeatures,
    lashlang_surface: lashlang::LashlangSurface,
    lashlang_artifact_store: Arc<dyn lashlang::LashlangArtifactStore>,
    attachment_store: Arc<dyn crate::AttachmentStore>,
    chronological_projection: Arc<crate::ChronologicalProjection>,
    protocol_extension: Option<crate::ProtocolTurnExtensionHandle>,
    turn_context: crate::TurnContext,
    execution_env_spec: crate::ProcessExecutionEnvSpec,
    process_originator: Option<crate::ProcessOriginator>,
    process_env_ref: Option<crate::ProcessExecutionEnvRef>,
    process_wake_target: Option<crate::SessionScope>,
    pub(super) parent_invocation: Option<crate::RuntimeInvocation>,
    lashlang_execution_sink: Option<Arc<dyn lash_trace::TraceSink>>,
    lashlang_execution_context: lash_trace::TraceContext,
    pub(super) turn_event_tx: Option<Sender<TurnActivity>>,
    pub(super) cancellation_token: Option<CancellationToken>,
    /// Process ids started by THIS execution context. Possession of a handle
    /// the run itself created is sufficient capability to await/cancel it —
    /// run-local children are not session handle grants (the ephemeral
    /// execution scope must never appear in durable grant state).
    started_process_ids: Arc<std::sync::Mutex<std::collections::HashSet<String>>>,
}

impl<'run> RuntimeExecutionContext<'run> {
    pub(crate) fn drain_tool_host_event_outcomes(
        &self,
    ) -> Result<Vec<crate::tool_dispatch::ToolHostEventEffectOutcome>, crate::PluginError> {
        self.dispatch
            .host_event_outcomes
            .drain()
            .map_err(crate::PluginError::Session)
    }

    pub(super) fn process_scope(
        &self,
        parent_invocation: Option<crate::RuntimeInvocation>,
    ) -> crate::ProcessOpScope<'_> {
        crate::ProcessOpScope::new(self.dispatch.effect_controller.scoped())
            .with_parent_invocation(parent_invocation)
            .with_agent_frame_id(Some(self.dispatch.agent_frame_id.clone()))
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "code execution bridge carries explicit per-turn runtime dependencies"
    )]
    pub(crate) fn new(
        session_id: String,
        dispatch: Arc<ToolDispatchContext<'run>>,
        lashlang_abilities: lashlang::LashlangAbilities,
        lashlang_language_features: lashlang::LashlangLanguageFeatures,
        lashlang_artifact_store: Arc<dyn lashlang::LashlangArtifactStore>,
        attachment_store: Arc<dyn crate::AttachmentStore>,
        chronological_projection: Arc<crate::ChronologicalProjection>,
        protocol_extension: Option<crate::ProtocolTurnExtensionHandle>,
        turn_context: crate::TurnContext,
    ) -> Self {
        let lashlang_surface = lashlang_surface_from_tool_surface(
            &dispatch.surface,
            lashlang_abilities,
            lashlang_language_features,
            dispatch.plugins.lashlang_resources(),
        );
        Self {
            session_id,
            dispatch,
            lashlang_abilities,
            lashlang_language_features,
            lashlang_surface,
            lashlang_artifact_store,
            attachment_store,
            chronological_projection,
            protocol_extension,
            turn_context,
            execution_env_spec: crate::ProcessExecutionEnvSpec::new(
                crate::PluginOptions::default(),
                crate::SessionPolicy::default(),
            ),
            process_originator: None,
            started_process_ids: Arc::default(),
            process_env_ref: None,
            process_wake_target: None,
            parent_invocation: None,
            lashlang_execution_sink: None,
            lashlang_execution_context: lash_trace::TraceContext::default(),
            turn_event_tx: None,
            cancellation_token: None,
        }
    }

    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    pub fn attachment_store(&self) -> Arc<dyn crate::AttachmentStore> {
        Arc::clone(&self.attachment_store)
    }

    pub async fn put_lashlang_module_artifact(
        &self,
        artifact: &lashlang::ModuleArtifact,
    ) -> Result<(), String> {
        self.lashlang_artifact_store
            .put_module_artifact(artifact)
            .await
            .map_err(|err| err.to_string())
    }

    pub fn chronological_projection(&self) -> Arc<crate::ChronologicalProjection> {
        Arc::clone(&self.chronological_projection)
    }

    pub fn protocol_extension<T: 'static>(&self) -> Option<&T> {
        self.protocol_extension
            .as_ref()
            .and_then(|extension| extension.as_any().downcast_ref::<T>())
    }

    pub fn turn_context(&self) -> &crate::TurnContext {
        &self.turn_context
    }

    pub(crate) fn session_graph_service(&self) -> &dyn crate::plugin::SessionGraphService {
        self.dispatch.session_graph.as_ref()
    }

    pub(super) async fn emit_turn_activity(
        &self,
        correlation_id: TurnActivityId,
        event: TurnEvent,
    ) {
        if let Some(tx) = &self.turn_event_tx {
            let _ = tx.send(TurnActivity::new(correlation_id, event)).await;
        }
    }

    pub(crate) fn with_turn_event_sender(mut self, turn_event_tx: Sender<TurnActivity>) -> Self {
        self.turn_event_tx = Some(turn_event_tx);
        self
    }

    pub(crate) fn with_parent_invocation(mut self, metadata: crate::RuntimeInvocation) -> Self {
        self.parent_invocation = Some(metadata);
        self
    }

    pub(crate) fn with_execution_env_spec(
        mut self,
        execution_env_spec: crate::ProcessExecutionEnvSpec,
    ) -> Self {
        self.execution_env_spec = execution_env_spec;
        self
    }

    pub(crate) fn with_process_registration_context(
        mut self,
        registration: &crate::ProcessRegistration,
    ) -> Self {
        self.process_originator = Some(registration.provenance.originator.clone());
        self.process_env_ref = registration.env_ref.clone();
        self.process_wake_target = registration.wake_target.clone();
        self
    }

    /// Spawn provenance for children started by this context, present only
    /// when this context executes a process: children inherit the chain's
    /// originator and wake target instead of the ephemeral execution scope.
    pub(super) fn record_started_process(&self, process_id: &str) {
        self.started_process_ids
            .lock()
            .expect("started process ids lock")
            .insert(process_id.to_string());
    }

    pub(super) fn is_run_local_process(&self, process_id: &str) -> bool {
        self.started_process_ids
            .lock()
            .expect("started process ids lock")
            .contains(process_id)
    }

    pub(crate) fn process_spawn_provenance(&self) -> Option<crate::ProcessSpawnProvenance> {
        self.process_originator
            .clone()
            .map(|originator| crate::ProcessSpawnProvenance {
                originator,
                wake_target: self.process_wake_target.clone(),
            })
    }

    pub(super) async fn attach_captured_process_execution_env(
        &self,
        registration: crate::ProcessRegistration,
    ) -> Result<crate::ProcessRegistration, crate::PluginError> {
        if registration.env_ref.is_some() {
            return Ok(registration);
        }
        match registration.input.as_ref() {
            crate::ProcessInput::ToolCall { .. } | crate::ProcessInput::LashlangProcess { .. } => {
                let env_ref = self.captured_process_execution_env_ref().await?;
                Ok(registration.with_execution_env_ref(Some(env_ref)))
            }
            crate::ProcessInput::External { .. } | crate::ProcessInput::SessionTurn { .. } => {
                Ok(registration)
            }
        }
    }

    async fn captured_process_execution_env_ref(
        &self,
    ) -> Result<crate::ProcessExecutionEnvRef, crate::PluginError> {
        if let Some(env_ref) = self.process_env_ref.clone() {
            return Ok(env_ref);
        }
        crate::persist_process_execution_env(
            self.lashlang_artifact_store.as_ref(),
            &self.execution_env_spec,
        )
        .await
    }

    pub(crate) fn with_lashlang_execution_trace(
        mut self,
        sink: Option<Arc<dyn lash_trace::TraceSink>>,
        context: lash_trace::TraceContext,
    ) -> Self {
        self.lashlang_execution_sink = sink;
        self.lashlang_execution_context = context;
        self
    }

    pub fn parent_invocation(&self) -> Option<&crate::RuntimeInvocation> {
        self.parent_invocation.as_ref()
    }

    pub fn lashlang_execution_sink(&self) -> Option<Arc<dyn lash_trace::TraceSink>> {
        self.lashlang_execution_sink.clone()
    }

    pub fn lashlang_execution_context(&self) -> &lash_trace::TraceContext {
        &self.lashlang_execution_context
    }

    pub(crate) fn with_cancellation_token(mut self, cancellation_token: CancellationToken) -> Self {
        self.cancellation_token = Some(cancellation_token);
        self
    }

    pub(crate) fn tool_scheduling(&self, name: &str) -> crate::ToolScheduling {
        crate::tool_dispatch::resolve_tool_scheduling(&self.dispatch, name)
    }

    pub fn callable_tool_manifest(&self, name: &str) -> Option<crate::ToolManifest> {
        crate::tool_dispatch::resolve_callable_manifest(&self.dispatch, name)
    }

    pub fn callable_tool_manifest_by_id(&self, id: &crate::ToolId) -> Option<crate::ToolManifest> {
        crate::tool_dispatch::resolve_callable_manifest_by_id(&self.dispatch, id)
    }

    pub fn resolve_lashlang_host_operation(
        &self,
        receiver: &lashlang::ResourceHandle,
        operation: &str,
    ) -> Result<String, String> {
        self.lashlang_surface
            .resources
            .resolve_module_operation(&receiver.resource_type, &receiver.alias, operation)
            .map(|binding| binding.host_operation.clone())
            .ok_or_else(|| {
                format!(
                    "module `{}` of type `{}` does not expose operation `{operation}`",
                    receiver.alias, receiver.resource_type
                )
            })
    }

    pub async fn prepare_lashlang_process_start(
        &self,
        start: lashlang::ProcessStart,
    ) -> Result<(crate::ProcessRegistration, Option<String>), String> {
        let display_name = Some(start.process_name.clone());
        let artifact = self
            .lashlang_artifact_store
            .get_module_artifact(&start.module_ref)
            .await
            .map_err(|err| format!("failed to load lashlang module artifact: {err}"))?
            .ok_or_else(|| {
                format!(
                    "missing lashlang module artifact `{}` for process `{}`",
                    start.module_ref, start.process_name
                )
            })?;
        if artifact.required_surface_ref != start.required_surface_ref {
            return Err(format!(
                "lashlang module artifact `{}` required surface mismatch: process requested {}, artifact has {}",
                start.module_ref, start.required_surface_ref, artifact.required_surface_ref
            ));
        }
        if artifact.process_ref(&start.process_name) != Some(&start.process_ref) {
            return Err(format!(
                "lashlang module artifact `{}` does not export process `{}` as requested ref {:?}",
                start.module_ref, start.process_name, start.process_ref
            ));
        }
        let args = match serde_json::to_value(lashlang::Value::Record(Arc::new(start.args)))
            .map_err(|err| format!("failed to serialize process args: {err}"))?
        {
            serde_json::Value::Object(map) => map,
            _ => return Err("process args must serialize as a record".to_string()),
        };
        let signal_event_types = artifact
            .canonical_ir
            .process(&start.process_name)
            .map(crate::lashlang_process_signal_event_types)
            .unwrap_or_default();
        let process_id = format!("process:{}", uuid::Uuid::new_v4());
        let registration = crate::ProcessRegistration::session_start_draft(
            process_id,
            crate::ProcessInput::LashlangProcess {
                module_ref: start.module_ref,
                process_ref: start.process_ref,
                required_surface_ref: start.required_surface_ref,
                process_name: start.process_name,
                args,
            },
        )
        .with_extra_event_types(
            crate::lashlang_process_event_types()
                .into_iter()
                .chain(signal_event_types),
        );
        Ok((registration, display_name))
    }

    pub fn lashlang_surface(&self) -> &lashlang::LashlangSurface {
        &self.lashlang_surface
    }

    pub fn lashlang_abilities(&self) -> lashlang::LashlangAbilities {
        self.lashlang_abilities
    }

    pub fn lashlang_language_features(&self) -> lashlang::LashlangLanguageFeatures {
        self.lashlang_language_features
    }

    pub fn link_lashlang_module(
        &self,
        program: lashlang::Program,
    ) -> Result<lashlang::LinkedModule, String> {
        lashlang::LinkedModule::link(program, self.lashlang_surface())
            .map_err(|err| err.to_string())
    }

    pub async fn perform_lashlang_trigger_operation(
        &self,
        operation: &str,
        payload: serde_json::Value,
    ) -> Result<serde_json::Value, String> {
        match lashlang::TriggerHostOperation::from_host_operation(operation) {
            Some(lashlang::TriggerHostOperation::Register) => self
                .register_host_event_subscription(payload)
                .await
                .map_err(|err| err.to_string()),
            Some(lashlang::TriggerHostOperation::List) => self
                .list_host_event_subscriptions(payload)
                .await
                .map_err(|err| err.to_string()),
            Some(lashlang::TriggerHostOperation::Cancel) => self
                .cancel_host_event_subscription(payload)
                .await
                .map_err(|err| err.to_string()),
            None => Err(format!("unknown trigger operation `{operation}`")),
        }
    }

    async fn register_host_event_subscription(
        &self,
        payload: serde_json::Value,
    ) -> Result<serde_json::Value, crate::PluginError> {
        let router = self.dispatch.host_event_router.as_ref().ok_or_else(|| {
            crate::PluginError::Session(
                "host event store is unavailable in this runtime".to_string(),
            )
        })?;
        let request = lashlang::TriggerRegistrationRequest::decode(&payload)
            .map_err(|err| crate::PluginError::Session(err.to_string()))?;
        let source_type = request.source.source_type.clone();
        let source_value = request.source.value.clone();
        let source = request.source.to_json();
        let event_type = lashlang::event_type_for_source(
            &self.dispatch.plugins.lashlang_resources(),
            &source_type,
        )
        .map_err(|err| crate::PluginError::Session(err.to_string()))?;
        let validation = crate::plugin::validate_target_process(
            &request.target,
            &event_type,
            &request.inputs,
            self.lashlang_artifact_store.as_ref(),
        )
        .await?;
        let store = router.store();
        let source_key = store
            .source_key_for_subscription(&source_type, &source_value)
            .await?;
        let env_ref = match self.process_env_ref.clone() {
            Some(env_ref) => env_ref,
            None => {
                crate::persist_process_execution_env(
                    self.lashlang_artifact_store.as_ref(),
                    &self.execution_env_spec,
                )
                .await?
            }
        };
        let registrant = self.process_originator.clone().unwrap_or_else(|| {
            crate::ProcessOriginator::session(crate::SessionScope::new(self.session_id.clone()))
        });
        let wake_target = self
            .process_wake_target
            .clone()
            .or_else(|| match &registrant {
                crate::ProcessOriginator::Session { scope } => Some(scope.clone()),
                crate::ProcessOriginator::Host => None,
            });
        let record = store
            .register_subscription(crate::TriggerSubscriptionDraft {
                registrant,
                env_ref,
                wake_target,
                name: request.name,
                source_type,
                source_key,
                source,
                event_ty: validation.event_ty,
                module_ref: request.target.module_ref,
                required_surface_ref: request.target.required_surface_ref,
                process_ref: request.target.process_ref,
                process_name: request.target.process_name,
                input_template: validation.inputs,
            })
            .await?;
        Ok(crate::plugin::trigger_handle_json(&record.handle))
    }

    async fn list_host_event_subscriptions(
        &self,
        payload: serde_json::Value,
    ) -> Result<serde_json::Value, crate::PluginError> {
        let router = self.dispatch.host_event_router.as_ref().ok_or_else(|| {
            crate::PluginError::Session(
                "host event store is unavailable in this runtime".to_string(),
            )
        })?;
        let request = lashlang::TriggerListRequest::decode(&payload)
            .map_err(|err| crate::PluginError::Session(err.to_string()))?;
        let mut filter = crate::TriggerSubscriptionFilter::for_session(&self.session_id);
        filter.target = request.target;
        filter.name = request.name;
        filter.source_type = request.source_type;
        filter.enabled = request.enabled;
        let registrations = router
            .store()
            .list_subscriptions(filter)
            .await?
            .iter()
            .map(crate::TriggerRegistration::from)
            .collect::<Vec<_>>();
        serde_json::to_value(registrations).map_err(|err| {
            crate::PluginError::Session(format!("failed to encode trigger registrations: {err}"))
        })
    }

    async fn cancel_host_event_subscription(
        &self,
        payload: serde_json::Value,
    ) -> Result<serde_json::Value, crate::PluginError> {
        let router = self.dispatch.host_event_router.as_ref().ok_or_else(|| {
            crate::PluginError::Session(
                "host event store is unavailable in this runtime".to_string(),
            )
        })?;
        let request = lashlang::TriggerCancelRequest::decode(&payload)
            .map_err(|err| crate::PluginError::Session(err.to_string()))?;
        let changed = router
            .store()
            .cancel_subscription(&self.session_id, &request.handle)
            .await?;
        Ok(serde_json::json!(changed))
    }

    pub fn tool_argument_projection_policy(
        &self,
        name: &str,
    ) -> crate::ToolArgumentProjectionPolicy {
        crate::tool_dispatch::resolve_tool_argument_projection_policy(&self.dispatch, name)
    }

    pub async fn start_lashlang_process(
        &self,
        registration: crate::ProcessRegistration,
        label: Option<String>,
    ) -> crate::ToolInvocationReply {
        let registration = match self
            .attach_captured_process_execution_env(registration)
            .await
        {
            Ok(registration) => registration,
            Err(err) => {
                return crate::ToolInvocationReply::error(serde_json::json!(err.to_string()));
            }
        };
        let process_id = registration.id.clone();
        let mut options = crate::ProcessStartOptions::new()
            .with_descriptor(crate::ProcessHandleDescriptor::new(Some("lashlang"), label));
        if let Some(spawn) = self.process_spawn_provenance() {
            options = options.with_spawn_provenance(spawn);
        }
        match self
            .dispatch
            .processes
            .start(
                &self.session_id,
                registration,
                options,
                self.process_scope(self.parent_invocation.clone()),
            )
            .await
        {
            Ok(_) => {
                self.record_started_process(&process_id);
                crate::ToolInvocationReply::success(crate::lashlang_bridge::process_handle_json(
                    &process_id,
                ))
            }
            Err(err) => crate::ToolInvocationReply::error(serde_json::json!(err.to_string())),
        }
    }

    pub async fn sleep_lashlang(
        &self,
        scope: &str,
        sequence: u64,
        duration_ms: u64,
    ) -> Result<(), crate::RuntimeEffectControllerError> {
        let cancellation = self.cancellation_token.clone().unwrap_or_default();
        let invocation = crate::runtime::causal::lashlang_sleep_invocation(
            &self.session_id,
            self.parent_invocation.as_ref(),
            scope,
            sequence,
        );
        let outcome = self
            .dispatch
            .effect_controller
            .controller()
            .execute_effect(
                crate::RuntimeEffectEnvelope::new(
                    invocation,
                    crate::RuntimeEffectCommand::Sleep { duration_ms },
                ),
                crate::RuntimeEffectLocalExecutor::sleep(cancellation),
            )
            .await?;
        match outcome {
            crate::RuntimeEffectOutcome::Sleep => Ok(()),
            other => Err(crate::RuntimeEffectControllerError::new(
                "runtime_effect_wrong_outcome",
                format!("expected sleep outcome, got {}", other.kind().as_str()),
            )),
        }
    }

    pub async fn await_process_event_lashlang(
        &self,
        registry: Arc<dyn crate::ProcessRegistry>,
        process_id: &str,
        signal_name: &str,
        event_type: &str,
        event_ordinal: u64,
    ) -> Result<serde_json::Value, crate::RuntimeEffectControllerError> {
        let cancellation = self.cancellation_token.clone().unwrap_or_default();
        let key = crate::process_signal_wait_key(process_id, signal_name, event_ordinal);
        let invocation = crate::runtime::causal::lashlang_await_event_invocation(
            &self.session_id,
            self.parent_invocation.as_ref(),
            process_id,
            signal_name,
            event_ordinal,
        );
        let outcome = self
            .dispatch
            .effect_controller
            .controller()
            .execute_effect(
                crate::RuntimeEffectEnvelope::new(
                    invocation,
                    crate::RuntimeEffectCommand::AwaitEvent { key: key.clone() },
                ),
                crate::RuntimeEffectLocalExecutor::await_process_event(
                    key,
                    registry,
                    process_id.to_string(),
                    event_type.to_string(),
                    event_ordinal,
                    cancellation,
                ),
            )
            .await?;
        outcome.into_await_event()
    }

    pub async fn signal_lashlang_process(
        &self,
        registry: Arc<dyn crate::ProcessRegistry>,
        process_id: &str,
        signal_name: &str,
        signal_id: String,
        payload: serde_json::Value,
    ) -> Result<crate::ProcessEvent, crate::RuntimeEffectControllerError> {
        let event_type = crate::process_signal_event_type(signal_name)?;
        let replay_key = format!("process:{process_id}:signal.{signal_name}:{signal_id}");
        let command = crate::ProcessCommand::Signal {
            process_id: process_id.to_string(),
            signal_name: signal_name.to_string(),
            signal_id,
            request: crate::ProcessEventAppendRequest::new(event_type, payload)
                .with_replay_key(replay_key),
        };
        let effect_id = command.effect_id();
        let invocation = crate::runtime::causal::process_effect_invocation(
            &self.session_id,
            self.parent_invocation.clone(),
            &effect_id,
        );
        let outcome = self
            .dispatch
            .effect_controller
            .controller()
            .execute_effect(
                crate::RuntimeEffectEnvelope::new(
                    invocation,
                    crate::RuntimeEffectCommand::process(command),
                ),
                crate::RuntimeEffectLocalExecutor::process_control(registry),
            )
            .await?;
        match outcome.into_process()? {
            crate::ProcessEffectOutcome::Signal { event } => Ok(event),
            other => Err(crate::RuntimeEffectControllerError::new(
                "runtime_effect_wrong_outcome",
                format!("expected signal outcome, got {other:?}"),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool_dispatch::ToolDispatchContext;
    use crate::{ToolCall, ToolProvider, ToolResult};

    struct NoopTools;

    #[async_trait::async_trait]
    impl ToolProvider for NoopTools {
        fn tool_manifests(&self) -> Vec<crate::ToolManifest> {
            Vec::new()
        }

        fn resolve_contract(&self, _name: &str) -> Option<Arc<crate::ToolContract>> {
            None
        }

        async fn execute(&self, _call: ToolCall<'_>) -> ToolResult {
            ToolResult::err_fmt("not used")
        }
    }

    #[test]
    fn tool_argument_projection_policy_resolves_from_active_surface_and_defaults_unknown() {
        let tool = crate::ToolDefinition::raw(
            "tool:seedy",
            "seedy",
            "Seed-aware",
            crate::ToolDefinition::default_input_schema(),
            serde_json::json!({ "type": "string" }),
        )
        .with_argument_projection(
            crate::ToolArgumentProjectionPolicy::preserve_projected_refs_in_field("seed"),
        );
        let plugins = crate::plugin::PluginHost::empty()
            .build_session("session", None)
            .expect("plugin session");
        let (event_tx, _event_rx) = tokio::sync::mpsc::channel(1);
        let dispatch = Arc::new(ToolDispatchContext {
            plugins,
            tools: Arc::new(NoopTools),
            surface: Arc::new(crate::ToolSurface::from_tools(
                vec![tool.manifest()],
                std::collections::BTreeMap::new(),
            )),
            sessions: Arc::new(crate::testing::MockSessionManager::default()),
            session_lifecycle: Arc::new(crate::testing::MockSessionManager::default()),
            session_graph: Arc::new(crate::testing::MockSessionManager::default()),
            processes: Arc::new(crate::UnavailableProcessService),
            process_cancel_ability: Arc::new(crate::DefaultProcessCancelAbility),
            host_event_router: None,
            effect_controller: crate::runtime::RuntimeEffectControllerHandle::shared(Arc::new(
                crate::InlineRuntimeEffectController,
            )),
            direct_completions: crate::DirectCompletionClient::unavailable(
                "direct completions are unavailable in this test context",
            ),
            parent_invocation: None,
            execution_env_spec: crate::ProcessExecutionEnvSpec::new(
                crate::PluginOptions::default(),
                crate::SessionPolicy::default(),
            ),
            session_id: "session".to_string(),
            agent_frame_id: String::new(),
            event_tx,
            checkpoint_messages: crate::tool_dispatch::CheckpointMessageBuffer::default(),
            host_event_outcomes: crate::tool_dispatch::ToolHostEventOutcomeBuffer::default(),
            attachment_store: Arc::new(crate::InMemoryAttachmentStore::new()),
            turn_context: crate::TurnContext::default(),
        });
        let ctx = RuntimeExecutionContext::new(
            "session".to_string(),
            dispatch,
            Default::default(),
            Default::default(),
            Arc::new(lashlang::InMemoryLashlangArtifactStore::new()),
            Arc::new(crate::InMemoryAttachmentStore::new()),
            Arc::new(crate::ChronologicalProjection::default()),
            None,
            crate::TurnContext::default(),
        );

        assert_eq!(
            ctx.tool_argument_projection_policy("seedy"),
            crate::ToolArgumentProjectionPolicy::preserve_projected_refs_in_field("seed")
        );
        assert_eq!(
            ctx.tool_argument_projection_policy("missing"),
            crate::ToolArgumentProjectionPolicy::MaterializeProjectedValues
        );
    }

    #[tokio::test]
    async fn prepare_lashlang_process_start_captures_tool_ids_and_explicit_input() {
        let tool = crate::ToolDefinition::raw(
            "tool:alpha",
            "alpha",
            "Alpha tool.",
            crate::ToolDefinition::default_input_schema(),
            serde_json::json!({ "type": "object", "additionalProperties": true }),
        );
        let plugins = crate::plugin::PluginHost::empty()
            .build_session("session", None)
            .expect("plugin session");
        let (event_tx, _event_rx) = tokio::sync::mpsc::channel(1);
        let dispatch = Arc::new(ToolDispatchContext {
            plugins,
            tools: Arc::new(NoopTools),
            surface: Arc::new(crate::ToolSurface::from_tools(
                vec![tool.manifest()],
                std::collections::BTreeMap::new(),
            )),
            sessions: Arc::new(crate::testing::MockSessionManager::default()),
            session_lifecycle: Arc::new(crate::testing::MockSessionManager::default()),
            session_graph: Arc::new(crate::testing::MockSessionManager::default()),
            processes: Arc::new(crate::UnavailableProcessService),
            process_cancel_ability: Arc::new(crate::DefaultProcessCancelAbility),
            host_event_router: None,
            effect_controller: crate::runtime::RuntimeEffectControllerHandle::shared(Arc::new(
                crate::InlineRuntimeEffectController,
            )),
            direct_completions: crate::DirectCompletionClient::unavailable(
                "direct completions are unavailable in this test context",
            ),
            parent_invocation: None,
            execution_env_spec: crate::ProcessExecutionEnvSpec::new(
                crate::PluginOptions::default(),
                crate::SessionPolicy::default(),
            ),
            session_id: "session".to_string(),
            agent_frame_id: String::new(),
            event_tx,
            checkpoint_messages: crate::tool_dispatch::CheckpointMessageBuffer::default(),
            host_event_outcomes: crate::tool_dispatch::ToolHostEventOutcomeBuffer::default(),
            attachment_store: Arc::new(crate::InMemoryAttachmentStore::new()),
            turn_context: crate::TurnContext::default(),
        });
        let ctx = RuntimeExecutionContext::new(
            "session".to_string(),
            dispatch,
            lashlang::LashlangAbilities::default().with_processes(),
            Default::default(),
            Arc::new(lashlang::InMemoryLashlangArtifactStore::new()),
            Arc::new(crate::InMemoryAttachmentStore::new()),
            Arc::new(crate::ChronologicalProjection::default()),
            None,
            crate::TurnContext::default(),
        );
        let mut input = lashlang::Record::new();
        input.insert("root".to_string(), lashlang::Value::String(".".into()));
        let linked = ctx
            .link_lashlang_module(
                lashlang::parse("process scan(root: str) { finish root }").expect("process module"),
            )
            .expect("link process module");
        ctx.put_lashlang_module_artifact(&linked.artifact)
            .await
            .expect("store module artifact");
        let process_ref = linked
            .artifact
            .process_ref("scan")
            .expect("scan process ref")
            .clone();
        let (registration, label) = ctx
            .prepare_lashlang_process_start(lashlang::ProcessStart {
                module_ref: linked.module_ref.clone(),
                process_ref,
                required_surface_ref: linked.required_surface_ref.clone(),
                process_name: "scan".to_string(),
                args: input,
            })
            .await
            .expect("process start should prepare");

        assert_eq!(label.as_deref(), Some("scan"));
        assert!(
            registration
                .event_types
                .iter()
                .any(|event_type| event_type.name == "process.wake")
        );
        let crate::ProcessInput::LashlangProcess {
            args, process_name, ..
        } = registration.input.as_ref()
        else {
            panic!("expected lashlang process input");
        };
        assert_eq!(process_name, "scan");
        assert_eq!(args.get("root"), Some(&serde_json::json!(".")));
    }

    #[test]
    fn lashlang_surface_reflects_host_abilities() {
        let tool = crate::ToolDefinition::raw(
            "tool:alpha",
            "alpha",
            "Alpha tool.",
            crate::ToolDefinition::default_input_schema(),
            serde_json::json!({ "type": "object", "additionalProperties": true }),
        );
        let plugins = crate::plugin::PluginHost::empty()
            .build_session("session", None)
            .expect("plugin session");
        let (event_tx, _event_rx) = tokio::sync::mpsc::channel(1);
        let dispatch = Arc::new(ToolDispatchContext {
            plugins,
            tools: Arc::new(NoopTools),
            surface: Arc::new(crate::ToolSurface::from_tools(
                vec![tool.manifest()],
                std::collections::BTreeMap::new(),
            )),
            sessions: Arc::new(crate::testing::MockSessionManager::default()),
            session_lifecycle: Arc::new(crate::testing::MockSessionManager::default()),
            session_graph: Arc::new(crate::testing::MockSessionManager::default()),
            processes: Arc::new(crate::UnavailableProcessService),
            process_cancel_ability: Arc::new(crate::DefaultProcessCancelAbility),
            host_event_router: None,
            effect_controller: crate::runtime::RuntimeEffectControllerHandle::shared(Arc::new(
                crate::InlineRuntimeEffectController,
            )),
            direct_completions: crate::DirectCompletionClient::unavailable(
                "direct completions are unavailable in this test context",
            ),
            parent_invocation: None,
            execution_env_spec: crate::ProcessExecutionEnvSpec::new(
                crate::PluginOptions::default(),
                crate::SessionPolicy::default(),
            ),
            session_id: "session".to_string(),
            agent_frame_id: String::new(),
            event_tx,
            checkpoint_messages: crate::tool_dispatch::CheckpointMessageBuffer::default(),
            host_event_outcomes: crate::tool_dispatch::ToolHostEventOutcomeBuffer::default(),
            attachment_store: Arc::new(crate::InMemoryAttachmentStore::new()),
            turn_context: crate::TurnContext::default(),
        });
        let ctx = RuntimeExecutionContext::new(
            "session".to_string(),
            dispatch,
            lashlang::LashlangAbilities::default()
                .with_sleep()
                .with_processes()
                .with_process_signals(),
            Default::default(),
            Arc::new(lashlang::InMemoryLashlangArtifactStore::new()),
            Arc::new(crate::InMemoryAttachmentStore::new()),
            Arc::new(crate::ChronologicalProjection::default()),
            None,
            crate::TurnContext::default(),
        );

        let surface = ctx.lashlang_surface();

        assert!(std::ptr::eq(surface, ctx.lashlang_surface()));
        assert!(surface.abilities.processes);
        assert!(surface.abilities.sleep);
        assert!(surface.abilities.process_signals);
        assert!(!surface.abilities.triggers);
        assert!(
            surface
                .resources
                .resolve_operation("Tools", "alpha")
                .is_some()
        );
    }

    #[test]
    fn lashlang_surface_reflects_host_resource_contributions() {
        let mut resources = lashlang::ResourceCatalog::new();
        resources
            .add_trigger_source_constructor(
                ["clock", "Alarm"],
                lashlang::TypeExpr::Object(vec![lashlang::TypeField {
                    name: "at".into(),
                    ty: lashlang::TypeExpr::Str,
                    optional: false,
                }]),
                lashlang::NamedDataType::object(
                    "clock.Tick",
                    vec![lashlang::TypeField {
                        name: "fired_at".into(),
                        ty: lashlang::TypeExpr::Str,
                        optional: false,
                    }],
                )
                .expect("valid clock tick type"),
            )
            .expect("valid clock trigger source");
        let plugin_host = crate::plugin::PluginHost::empty();
        let mut merged_resources = plugin_host.lashlang_resources();
        merged_resources.extend(resources);
        let plugins = plugin_host
            .with_lashlang_resources(merged_resources)
            .build_session("session", None)
            .expect("plugin session");
        let (event_tx, _event_rx) = tokio::sync::mpsc::channel(1);
        let dispatch = Arc::new(ToolDispatchContext {
            plugins,
            tools: Arc::new(NoopTools),
            surface: Arc::new(crate::ToolSurface::from_tools(
                Vec::new(),
                std::collections::BTreeMap::new(),
            )),
            sessions: Arc::new(crate::testing::MockSessionManager::default()),
            session_lifecycle: Arc::new(crate::testing::MockSessionManager::default()),
            session_graph: Arc::new(crate::testing::MockSessionManager::default()),
            processes: Arc::new(crate::UnavailableProcessService),
            process_cancel_ability: Arc::new(crate::DefaultProcessCancelAbility),
            host_event_router: None,
            effect_controller: crate::runtime::RuntimeEffectControllerHandle::shared(Arc::new(
                crate::InlineRuntimeEffectController,
            )),
            direct_completions: crate::DirectCompletionClient::unavailable(
                "direct completions are unavailable in this test context",
            ),
            parent_invocation: None,
            execution_env_spec: crate::ProcessExecutionEnvSpec::new(
                crate::PluginOptions::default(),
                crate::SessionPolicy::default(),
            ),
            session_id: "session".to_string(),
            agent_frame_id: String::new(),
            event_tx,
            checkpoint_messages: crate::tool_dispatch::CheckpointMessageBuffer::default(),
            host_event_outcomes: crate::tool_dispatch::ToolHostEventOutcomeBuffer::default(),
            attachment_store: Arc::new(crate::InMemoryAttachmentStore::new()),
            turn_context: crate::TurnContext::default(),
        });
        let ctx = RuntimeExecutionContext::new(
            "session".to_string(),
            dispatch,
            lashlang::LashlangAbilities::default()
                .with_processes()
                .with_triggers(),
            Default::default(),
            Arc::new(lashlang::InMemoryLashlangArtifactStore::new()),
            Arc::new(crate::InMemoryAttachmentStore::new()),
            Arc::new(crate::ChronologicalProjection::default()),
            None,
            crate::TurnContext::default(),
        );

        let surface = ctx.lashlang_surface();

        assert!(
            surface
                .resources
                .resolve_value_constructor(&["clock", "Alarm"])
                .is_some()
        );
        assert!(
            surface
                .resources
                .resolve_trigger_source("clock.Alarm")
                .is_some()
        );
        lashlang::LinkedModule::link(
            lashlang::parse(
                r#"
                process remember(tick: clock.Tick) {
                  finish true
                }

                source = clock.Alarm({ at: "08:00" })
                await triggers.register({
                  source: source,
                  target: remember,
                  inputs: { tick: trigger.event }
                })?
                "#,
            )
            .expect("parse trigger registry module"),
            surface,
        )
        .expect("host resource contribution should be linkable");
    }
}
