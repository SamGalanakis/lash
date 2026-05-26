use std::sync::Arc;

use tokio::sync::mpsc::Sender;
use tokio_util::sync::CancellationToken;

use crate::tool_dispatch::ToolDispatchContext;
use crate::{TurnActivity, TurnActivityId, TurnEvent};

pub(crate) fn lashlang_surface_from_tool_surface(
    surface: &crate::ToolSurface,
    abilities: lashlang::LashlangAbilities,
) -> lashlang::LashlangSurface {
    lashlang::LashlangSurface::new(lashlang_resources_from_tool_surface(surface), abilities)
}

pub(crate) fn lashlang_resources_from_tool_surface(
    surface: &crate::ToolSurface,
) -> lashlang::ResourceCatalog {
    let mut catalog = lashlang::ResourceCatalog::new();
    catalog.add_alias("TOOL", "default");
    for entry in surface.tools.iter() {
        if entry.availability.is_callable() {
            catalog.add_operation(
                "TOOL",
                entry.manifest.name.clone(),
                entry.manifest.name.clone(),
            );
        }
    }
    catalog
}

#[derive(Clone)]
pub struct ModeExecutionContext<'run> {
    pub(super) session_id: String,
    pub(super) execution_mode: crate::ExecutionMode,
    pub(super) dispatch: Arc<ToolDispatchContext<'run>>,
    lashlang_abilities: lashlang::LashlangAbilities,
    attachment_store: Arc<dyn crate::AttachmentStore>,
    chronological_projection: Arc<crate::ChronologicalProjection>,
    mode_extension: Option<crate::ModeTurnExtensionHandle>,
    turn_context: crate::TurnContext,
    pub(super) effect_metadata: Option<crate::EffectInvocationMetadata>,
    pub(super) turn_lease: Option<crate::RuntimeTurnLease>,
    pub(super) turn_event_tx: Option<Sender<TurnActivity>>,
    pub(super) cancellation_token: Option<CancellationToken>,
}

impl<'run> ModeExecutionContext<'run> {
    pub(super) fn process_request_scope(
        &self,
        effect_metadata: Option<crate::EffectInvocationMetadata>,
    ) -> crate::ProcessRequestScope<'_> {
        crate::ProcessRequestScope::new()
            .with_effect_metadata(effect_metadata)
            .with_effect_controller(self.dispatch.effect_controller.as_controller())
            .with_turn_lease(self.turn_lease.clone())
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "mode execution bridge carries explicit per-turn runtime dependencies"
    )]
    pub(crate) fn new(
        session_id: String,
        execution_mode: crate::ExecutionMode,
        dispatch: Arc<ToolDispatchContext<'run>>,
        lashlang_abilities: lashlang::LashlangAbilities,
        attachment_store: Arc<dyn crate::AttachmentStore>,
        chronological_projection: Arc<crate::ChronologicalProjection>,
        mode_extension: Option<crate::ModeTurnExtensionHandle>,
        turn_context: crate::TurnContext,
    ) -> Self {
        Self {
            session_id,
            execution_mode,
            dispatch,
            lashlang_abilities,
            attachment_store,
            chronological_projection,
            mode_extension,
            turn_context,
            effect_metadata: None,
            turn_lease: None,
            turn_event_tx: None,
            cancellation_token: None,
        }
    }

    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    pub fn execution_mode(&self) -> &crate::ExecutionMode {
        &self.execution_mode
    }

    pub fn attachment_store(&self) -> Arc<dyn crate::AttachmentStore> {
        Arc::clone(&self.attachment_store)
    }

    pub fn chronological_projection(&self) -> Arc<crate::ChronologicalProjection> {
        Arc::clone(&self.chronological_projection)
    }

    pub fn mode_extension<T: 'static>(&self) -> Option<&T> {
        self.mode_extension
            .as_ref()
            .and_then(|extension| extension.as_any().downcast_ref::<T>())
    }

    pub fn turn_context(&self) -> &crate::TurnContext {
        &self.turn_context
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

    pub(crate) fn with_effect_metadata(
        mut self,
        metadata: crate::EffectInvocationMetadata,
    ) -> Self {
        self.effect_metadata = Some(metadata);
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

    pub(crate) fn tool_execution_mode(&self, name: &str) -> crate::ToolExecutionMode {
        crate::tool_dispatch::resolve_tool_execution_mode(&self.dispatch, name)
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
        let display_name = Some(start.process.clone());
        let linked_module = match start.linked_module {
            Some(linked_module) => linked_module,
            None => self.link_lashlang_module(start.module)?,
        };
        let module_version = linked_module.module_version.clone();
        let args = match serde_json::to_value(&lashlang::Value::Record(Arc::new(start.args)))
            .map_err(|err| format!("failed to serialize process args: {err}"))?
        {
            serde_json::Value::Object(map) => map,
            _ => return Err("process args must serialize as a record".to_string()),
        };
        let process_id = format!("process:{}", uuid::Uuid::new_v4());
        let registration = crate::ProcessRegistration::new(
            process_id,
            crate::ProcessInput::LashlangProcess {
                module_version,
                linked_module,
                process_name: start.process,
                args,
            },
        )
        .with_extra_event_types(crate::lashlang_process_event_types());
        Ok((registration, display_name))
    }

    pub fn lashlang_surface(&self) -> lashlang::LashlangSurface {
        lashlang_surface_from_tool_surface(&self.dispatch.surface, self.lashlang_abilities)
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
    ) -> crate::ModeToolReply {
        let process_id = registration.id.clone();
        let execution_context = crate::ProcessExecutionContext::default()
            .with_tool_effect_metadata(self.effect_metadata.clone())
            .with_wake_session_id(self.session_id.clone());
        match self
            .dispatch
            .host
            .start_process(
                crate::ProcessStartRequest::new(&self.session_id, registration, execution_context)
                    .with_descriptor(crate::ProcessHandleDescriptor::new(Some("lashlang"), label))
                    .with_scope(
                        crate::ProcessRequestScope::new()
                            .with_effect_metadata(self.effect_metadata.clone())
                            .with_effect_controller(
                                self.dispatch.effect_controller.as_controller(),
                            ),
                    ),
            )
            .await
        {
            Ok(_) => crate::ModeToolReply::success(crate::lashlang_bridge::process_handle_json(
                &process_id,
            )),
            Err(err) => crate::ModeToolReply::error(serde_json::json!(err.to_string())),
        }
    }

    pub(crate) async fn sleep_lashlang_process(
        &self,
        process_id: &str,
        sequence: u64,
        duration_ms: u64,
    ) -> Result<(), crate::RuntimeEffectControllerError> {
        let cancellation = self
            .cancellation_token
            .clone()
            .unwrap_or_else(CancellationToken::new);
        let metadata = if let Some(parent) = self.effect_metadata.as_ref() {
            crate::EffectInvocationMetadata {
                session_id: parent.session_id.clone(),
                origin: parent.origin.clone(),
                turn_id: parent.turn_id.clone(),
                turn_index: parent.turn_index,
                mode_iteration: parent.mode_iteration,
                effect_id: format!("{}:process:{process_id}:sleep:{sequence}", parent.effect_id),
                effect_kind: crate::RuntimeEffectKind::Sleep,
                idempotency_key: format!(
                    "{}:process:{process_id}:sleep:{sequence}",
                    parent.idempotency_key
                ),
                turn_checkpoint_hash: parent.turn_checkpoint_hash.clone(),
            }
        } else {
            let effect_id = format!("process:{process_id}:sleep:{sequence}");
            crate::EffectInvocationMetadata {
                session_id: self.session_id.clone(),
                origin: crate::EffectOrigin::Turn,
                turn_id: None,
                turn_index: None,
                mode_iteration: None,
                effect_id: effect_id.clone(),
                effect_kind: crate::RuntimeEffectKind::Sleep,
                idempotency_key: effect_id,
                turn_checkpoint_hash: None,
            }
        };
        let outcome = self
            .dispatch
            .effect_controller
            .as_controller()
            .execute_effect(
                crate::RuntimeEffectEnvelope::new(
                    metadata,
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
    use crate::{ExecutionMode, ToolCall, ToolProvider, ToolResult};

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
            .build_standard_session("session", None)
            .expect("plugin session");
        let (event_tx, _event_rx) = tokio::sync::mpsc::channel(1);
        let dispatch = Arc::new(ToolDispatchContext {
            plugins,
            tools: Arc::new(NoopTools),
            surface: Arc::new(crate::ToolSurface::from_tools(
                vec![tool.manifest()],
                ExecutionMode::standard(),
                std::collections::BTreeMap::new(),
            )),
            host: Arc::new(crate::testing::MockSessionManager::default()),
            effect_controller: crate::runtime::RuntimeEffectControllerHandle::shared(Arc::new(
                crate::InlineRuntimeEffectController::default(),
            )),
            direct_completions: crate::DirectCompletionClient::unavailable(
                "direct completions are unavailable in this test context",
            ),
            tool_effect_metadata: None,
            session_id: "session".to_string(),
            event_tx,
            turn_injection_bridge: crate::TurnInjectionBridge::new(),
            attachment_store: Arc::new(crate::InMemoryAttachmentStore::new()),
            turn_context: crate::TurnContext::default(),
        });
        let ctx = ModeExecutionContext::new(
            "session".to_string(),
            ExecutionMode::standard(),
            dispatch,
            Default::default(),
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
            .build_standard_session("session", None)
            .expect("plugin session");
        let (event_tx, _event_rx) = tokio::sync::mpsc::channel(1);
        let dispatch = Arc::new(ToolDispatchContext {
            plugins,
            tools: Arc::new(NoopTools),
            surface: Arc::new(crate::ToolSurface::from_tools(
                vec![tool.manifest()],
                ExecutionMode::standard(),
                std::collections::BTreeMap::new(),
            )),
            host: Arc::new(crate::testing::MockSessionManager::default()),
            effect_controller: crate::runtime::RuntimeEffectControllerHandle::shared(Arc::new(
                crate::InlineRuntimeEffectController::default(),
            )),
            direct_completions: crate::DirectCompletionClient::unavailable(
                "direct completions are unavailable in this test context",
            ),
            tool_effect_metadata: None,
            session_id: "session".to_string(),
            event_tx,
            turn_injection_bridge: crate::TurnInjectionBridge::new(),
            attachment_store: Arc::new(crate::InMemoryAttachmentStore::new()),
            turn_context: crate::TurnContext::default(),
        });
        let ctx = ModeExecutionContext::new(
            "session".to_string(),
            ExecutionMode::standard(),
            dispatch,
            lashlang::LashlangAbilities::default().with_processes(),
            Arc::new(crate::InMemoryAttachmentStore::new()),
            Arc::new(crate::ChronologicalProjection::default()),
            None,
            crate::TurnContext::default(),
        );
        let mut input = lashlang::Record::new();
        input.insert("root".to_string(), lashlang::Value::String(".".into()));
        let (registration, label) = ctx
            .prepare_lashlang_process_start(lashlang::ProcessStart {
                module: lashlang::parse("process scan(root: str) { finish root }")
                    .expect("process module"),
                linked_module: None,
                process: "scan".to_string(),
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
            .build_standard_session("session", None)
            .expect("plugin session");
        let (event_tx, _event_rx) = tokio::sync::mpsc::channel(1);
        let dispatch = Arc::new(ToolDispatchContext {
            plugins,
            tools: Arc::new(NoopTools),
            surface: Arc::new(crate::ToolSurface::from_tools(
                vec![tool.manifest()],
                ExecutionMode::standard(),
                std::collections::BTreeMap::new(),
            )),
            host: Arc::new(crate::testing::MockSessionManager::default()),
            effect_controller: crate::runtime::RuntimeEffectControllerHandle::shared(Arc::new(
                crate::InlineRuntimeEffectController::default(),
            )),
            direct_completions: crate::DirectCompletionClient::unavailable(
                "direct completions are unavailable in this test context",
            ),
            tool_effect_metadata: None,
            session_id: "session".to_string(),
            event_tx,
            turn_injection_bridge: crate::TurnInjectionBridge::new(),
            attachment_store: Arc::new(crate::InMemoryAttachmentStore::new()),
            turn_context: crate::TurnContext::default(),
        });
        let ctx = ModeExecutionContext::new(
            "session".to_string(),
            ExecutionMode::standard(),
            dispatch,
            lashlang::LashlangAbilities::default()
                .with_processes()
                .with_process_lifecycle(),
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
                .resolve_operation("TOOL", "alpha")
                .is_some()
        );
    }
}
