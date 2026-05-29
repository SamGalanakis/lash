use std::sync::Arc;

use tokio::sync::mpsc::Sender;
use tokio_util::sync::CancellationToken;

use crate::tool_dispatch::ToolDispatchContext;
use crate::{TurnActivity, TurnActivityId, TurnEvent};

pub(crate) fn lashlang_surface_from_tool_surface(
    surface: &crate::ToolSurface,
    abilities: lashlang::LashlangAbilities,
    host_resources: lashlang::ResourceCatalog,
) -> lashlang::LashlangSurface {
    let mut resources = lashlang_resources_from_tool_surface(surface);
    resources.extend(host_resources);
    lashlang::LashlangSurface::new(resources, abilities)
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
            catalog.add_module_instance(
                agent_surface.module_path.iter().map(String::as_str),
                agent_surface.authority_type.clone(),
            );
            catalog.add_operation(
                agent_surface.authority_type,
                agent_surface.operation,
                entry.manifest.name.clone(),
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
    lashlang_artifact_store: Arc<dyn lashlang::LashlangArtifactStore>,
    attachment_store: Arc<dyn crate::AttachmentStore>,
    chronological_projection: Arc<crate::ChronologicalProjection>,
    protocol_extension: Option<crate::ProtocolTurnExtensionHandle>,
    turn_context: crate::TurnContext,
    pub(super) parent_invocation: Option<crate::RuntimeInvocation>,
    pub(super) turn_lease: Option<crate::RuntimeTurnLease>,
    pub(super) turn_event_tx: Option<Sender<TurnActivity>>,
    pub(super) cancellation_token: Option<CancellationToken>,
}

impl<'run> RuntimeExecutionContext<'run> {
    pub(super) fn process_scope(
        &self,
        parent_invocation: Option<crate::RuntimeInvocation>,
    ) -> crate::ProcessOpScope<'_> {
        crate::ProcessOpScope::new()
            .with_parent_invocation(parent_invocation)
            .with_effect_controller(self.dispatch.effect_controller.as_controller())
            .with_turn_lease(self.turn_lease.clone())
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
        lashlang_artifact_store: Arc<dyn lashlang::LashlangArtifactStore>,
        attachment_store: Arc<dyn crate::AttachmentStore>,
        chronological_projection: Arc<crate::ChronologicalProjection>,
        protocol_extension: Option<crate::ProtocolTurnExtensionHandle>,
        turn_context: crate::TurnContext,
    ) -> Self {
        Self {
            session_id,
            dispatch,
            lashlang_abilities,
            lashlang_artifact_store,
            attachment_store,
            chronological_projection,
            protocol_extension,
            turn_context,
            parent_invocation: None,
            turn_lease: None,
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

    pub fn put_lashlang_module_artifact(
        &self,
        artifact: &lashlang::ModuleArtifact,
    ) -> Result<(), String> {
        self.lashlang_artifact_store
            .put_module_artifact(artifact)
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

    pub(crate) fn runtime_host(&self) -> &dyn crate::plugin::RuntimeSessionHost {
        self.dispatch.host.as_ref()
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

    pub(crate) fn with_turn_lease(mut self, turn_lease: Option<crate::RuntimeTurnLease>) -> Self {
        self.turn_lease = turn_lease;
        self
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

    pub fn prepare_lashlang_process_start(
        &self,
        start: lashlang::ProcessStart,
    ) -> Result<(crate::ProcessRegistration, Option<String>), String> {
        let display_name = Some(start.process_name.clone());
        let artifact = self
            .lashlang_artifact_store
            .get_module_artifact(&start.module_ref)
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
        let process_id = format!("process:{}", uuid::Uuid::new_v4());
        let registration = crate::ProcessRegistration::new(
            process_id,
            crate::ProcessInput::LashlangProcess {
                module_ref: start.module_ref,
                process_ref: start.process_ref,
                required_surface_ref: start.required_surface_ref,
                process_name: start.process_name,
                args,
            },
        )
        .with_extra_event_types(crate::lashlang_process_event_types());
        Ok((registration, display_name))
    }

    pub fn lashlang_surface(&self) -> lashlang::LashlangSurface {
        lashlang_surface_from_tool_surface(
            &self.dispatch.surface,
            self.lashlang_abilities,
            self.dispatch.plugins.lashlang_resources(),
        )
    }

    pub fn lashlang_abilities(&self) -> lashlang::LashlangAbilities {
        self.lashlang_abilities
    }

    pub fn link_lashlang_module(
        &self,
        program: lashlang::Program,
    ) -> Result<lashlang::LinkedModule, String> {
        lashlang::LinkedModule::link(program, self.lashlang_surface())
            .map_err(|err| err.to_string())
    }

    pub fn install_lashlang_trigger_source(
        &self,
        source: &str,
    ) -> Result<crate::SessionTriggerInstallReport, String> {
        self.dispatch
            .plugins
            .install_lashlang_trigger_source(
                source,
                self.lashlang_surface(),
                self.lashlang_artifact_store.as_ref(),
            )
            .map_err(|err| err.to_string())
    }

    pub fn install_linked_lashlang_trigger_source(
        &self,
        source: &str,
        linked: &lashlang::LinkedModule,
    ) -> Result<crate::SessionTriggerInstallReport, String> {
        self.dispatch
            .plugins
            .install_linked_lashlang_trigger_source(
                source,
                linked,
                self.lashlang_artifact_store.as_ref(),
            )
            .map_err(|err| err.to_string())
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
        let process_id = registration.id.clone();
        match self
            .dispatch
            .processes
            .start(
                &self.session_id,
                registration,
                crate::ProcessStartOptions::new()
                    .with_descriptor(crate::ProcessHandleDescriptor::new(Some("lashlang"), label)),
                self.process_scope(self.parent_invocation.clone()),
            )
            .await
        {
            Ok(_) => crate::ToolInvocationReply::success(
                crate::lashlang_bridge::process_handle_json(&process_id),
            ),
            Err(err) => crate::ToolInvocationReply::error(serde_json::json!(err.to_string())),
        }
    }

    pub(crate) async fn sleep_lashlang_process(
        &self,
        process_id: &str,
        sequence: u64,
        duration_ms: u64,
    ) -> Result<(), crate::RuntimeEffectControllerError> {
        let cancellation = self.cancellation_token.clone().unwrap_or_default();
        let invocation = crate::runtime::causal::process_sleep_invocation(
            &self.session_id,
            self.parent_invocation.as_ref(),
            process_id,
            sequence,
        );
        let outcome = self
            .dispatch
            .effect_controller
            .as_controller()
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
            host: Arc::new(crate::testing::MockSessionManager::default()),
            processes: Arc::new(crate::UnavailableProcessService),
            effect_controller: crate::runtime::RuntimeEffectControllerHandle::shared(Arc::new(
                crate::InlineRuntimeEffectController::default(),
            )),
            direct_completions: crate::DirectCompletionClient::unavailable(
                "direct completions are unavailable in this test context",
            ),
            parent_invocation: None,
            session_id: "session".to_string(),
            agent_frame_id: String::new(),
            event_tx,
            checkpoint_messages: crate::tool_dispatch::CheckpointMessageBuffer::default(),
            attachment_store: Arc::new(crate::InMemoryAttachmentStore::new()),
            turn_context: crate::TurnContext::default(),
        });
        let ctx = RuntimeExecutionContext::new(
            "session".to_string(),
            dispatch,
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

    #[test]
    fn prepare_lashlang_process_start_captures_tool_ids_and_explicit_input() {
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
            host: Arc::new(crate::testing::MockSessionManager::default()),
            processes: Arc::new(crate::UnavailableProcessService),
            effect_controller: crate::runtime::RuntimeEffectControllerHandle::shared(Arc::new(
                crate::InlineRuntimeEffectController::default(),
            )),
            direct_completions: crate::DirectCompletionClient::unavailable(
                "direct completions are unavailable in this test context",
            ),
            parent_invocation: None,
            session_id: "session".to_string(),
            agent_frame_id: String::new(),
            event_tx,
            checkpoint_messages: crate::tool_dispatch::CheckpointMessageBuffer::default(),
            attachment_store: Arc::new(crate::InMemoryAttachmentStore::new()),
            turn_context: crate::TurnContext::default(),
        });
        let ctx = RuntimeExecutionContext::new(
            "session".to_string(),
            dispatch,
            lashlang::LashlangAbilities::default().with_processes(),
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
    fn lashlang_surface_reflects_host_abilities_without_default_cron() {
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
            host: Arc::new(crate::testing::MockSessionManager::default()),
            processes: Arc::new(crate::UnavailableProcessService),
            effect_controller: crate::runtime::RuntimeEffectControllerHandle::shared(Arc::new(
                crate::InlineRuntimeEffectController::default(),
            )),
            direct_completions: crate::DirectCompletionClient::unavailable(
                "direct completions are unavailable in this test context",
            ),
            parent_invocation: None,
            session_id: "session".to_string(),
            agent_frame_id: String::new(),
            event_tx,
            checkpoint_messages: crate::tool_dispatch::CheckpointMessageBuffer::default(),
            attachment_store: Arc::new(crate::InMemoryAttachmentStore::new()),
            turn_context: crate::TurnContext::default(),
        });
        let ctx = RuntimeExecutionContext::new(
            "session".to_string(),
            dispatch,
            lashlang::LashlangAbilities::default()
                .with_processes()
                .with_process_lifecycle(),
            Arc::new(lashlang::InMemoryLashlangArtifactStore::new()),
            Arc::new(crate::InMemoryAttachmentStore::new()),
            Arc::new(crate::ChronologicalProjection::default()),
            None,
            crate::TurnContext::default(),
        );

        let surface = ctx.lashlang_surface();

        assert!(surface.abilities.processes);
        assert!(surface.abilities.process_sleep);
        assert!(surface.abilities.process_signals);
        assert!(!surface.abilities.triggers);
        assert!(!surface.abilities.schedules.cron);
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
        resources.add_module_instance(["ui", "button"], "Button");
        resources.add_trigger_event("Button", "pressed", lashlang::TypeExpr::Any);
        let plugins = crate::plugin::PluginHost::empty()
            .with_lashlang_resources(resources)
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
            host: Arc::new(crate::testing::MockSessionManager::default()),
            processes: Arc::new(crate::UnavailableProcessService),
            effect_controller: crate::runtime::RuntimeEffectControllerHandle::shared(Arc::new(
                crate::InlineRuntimeEffectController::default(),
            )),
            direct_completions: crate::DirectCompletionClient::unavailable(
                "direct completions are unavailable in this test context",
            ),
            parent_invocation: None,
            session_id: "session".to_string(),
            agent_frame_id: String::new(),
            event_tx,
            checkpoint_messages: crate::tool_dispatch::CheckpointMessageBuffer::default(),
            attachment_store: Arc::new(crate::InMemoryAttachmentStore::new()),
            turn_context: crate::TurnContext::default(),
        });
        let ctx = RuntimeExecutionContext::new(
            "session".to_string(),
            dispatch,
            lashlang::LashlangAbilities::default()
                .with_processes()
                .with_triggers(),
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
                .trigger_event("Button", "pressed")
                .is_some()
        );
        lashlang::LinkedModule::link(
            lashlang::parse(
                r#"
                process remember(event: any) {
                  finish event
                }

                trigger remembered on ui.button.pressed as event
                  -> remember(event: event)
                "#,
            )
            .expect("parse trigger module"),
            surface,
        )
        .expect("host resource contribution should be linkable");
    }
}
