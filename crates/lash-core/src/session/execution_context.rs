use std::sync::Arc;

use tokio::sync::mpsc::Sender;
use tokio_util::sync::CancellationToken;

use crate::tool_dispatch::ToolDispatchContext;
use crate::{TurnActivity, TurnActivityId, TurnEvent};

#[derive(Clone)]
pub struct ModeExecutionContext<'run> {
    pub(super) session_id: String,
    pub(super) execution_mode: crate::ExecutionMode,
    pub(super) dispatch: Arc<ToolDispatchContext<'run>>,
    attachment_store: Arc<dyn crate::AttachmentStore>,
    chronological_projection: Arc<crate::ChronologicalProjection>,
    mode_extension: Option<crate::ModeTurnExtensionHandle>,
    turn_context: crate::TurnContext,
    pub(super) effect_metadata: Option<crate::EffectInvocationMetadata>,
    pub(super) turn_event_tx: Option<Sender<TurnActivity>>,
    pub(super) cancellation_token: Option<CancellationToken>,
}

impl<'run> ModeExecutionContext<'run> {
    #[allow(
        clippy::too_many_arguments,
        reason = "mode execution bridge carries explicit per-turn runtime dependencies"
    )]
    pub(crate) fn new(
        session_id: String,
        execution_mode: crate::ExecutionMode,
        dispatch: Arc<ToolDispatchContext<'run>>,
        attachment_store: Arc<dyn crate::AttachmentStore>,
        chronological_projection: Arc<crate::ChronologicalProjection>,
        mode_extension: Option<crate::ModeTurnExtensionHandle>,
        turn_context: crate::TurnContext,
    ) -> Self {
        Self {
            session_id,
            execution_mode,
            dispatch,
            attachment_store,
            chronological_projection,
            mode_extension,
            turn_context,
            effect_metadata: None,
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
        start: lashlang::ProcessBlockStart,
    ) -> Result<(crate::ProcessRegistration, Option<String>), String> {
        let display_name = start.name.as_ref().map(lashlang_process_name).transpose()?;
        let timeout_ms = start
            .timeout_ms
            .as_ref()
            .map(lashlang_process_timeout_ms)
            .transpose()?;
        let input = start
            .input
            .as_ref()
            .map(lashlang_process_input)
            .transpose()?
            .unwrap_or_default();
        let mut tool_bindings = Vec::with_capacity(start.tool_names.len());
        for name in start.tool_names {
            let manifest = self
                .callable_tool_manifest(&name)
                .ok_or_else(|| format!("tool `{name}` is unavailable in this session"))?;
            tool_bindings.push(crate::LashlangProcessToolBinding {
                name,
                tool_id: manifest.id.clone(),
            });
        }
        let program = serde_json::to_value(&start.program)
            .map_err(|err| format!("failed to serialize lashlang process block: {err}"))?;
        let process_id = format!("process:{}", uuid::Uuid::new_v4());
        let registration = crate::ProcessRegistration::new(
            process_id,
            crate::ProcessInput::LashlangBlock {
                program,
                input,
                tool_bindings,
                timeout_ms,
                display_name: display_name.clone(),
            },
        )
        .with_extra_event_types(crate::lashlang_process_event_types());
        Ok((registration, display_name))
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
            .start_process_scoped(
                &self.session_id,
                registration,
                Some(crate::ProcessHandleDescriptor::new(Some("lashlang"), label)),
                execution_context,
                self.effect_metadata.clone(),
                Some(self.dispatch.effect_controller.as_controller()),
            )
            .await
        {
            Ok(_) => crate::ModeToolReply::success(crate::lashlang_bridge::process_handle_json(
                &process_id,
            )),
            Err(err) => crate::ModeToolReply::error(serde_json::json!(err.to_string())),
        }
    }
}

fn lashlang_process_name(value: &lashlang::Value) -> Result<String, String> {
    match value {
        lashlang::Value::String(value) => Ok(value.to_string()),
        _ => Err("process `name` must be a string".to_string()),
    }
}

fn lashlang_process_timeout_ms(value: &lashlang::Value) -> Result<u64, String> {
    match value {
        lashlang::Value::Number(value)
            if value.is_finite()
                && *value >= 0.0
                && value.fract() == 0.0
                && *value <= u64::MAX as f64 =>
        {
            Ok(*value as u64)
        }
        _ => Err("process `timeout_ms` must be a non-negative integer".to_string()),
    }
}

fn lashlang_process_input(
    value: &lashlang::Value,
) -> Result<serde_json::Map<String, serde_json::Value>, String> {
    if value.as_record().is_none() {
        return Err("process `input` must be a record".to_string());
    }
    match serde_json::to_value(value)
        .map_err(|err| format!("failed to serialize process input: {err}"))?
    {
        serde_json::Value::Object(map) => Ok(map),
        _ => Err("process `input` must be a record".to_string()),
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
            Arc::new(crate::InMemoryAttachmentStore::new()),
            Arc::new(crate::ChronologicalProjection::default()),
            None,
            crate::TurnContext::default(),
        );
        let mut input = lashlang::Record::new();
        input.insert("root".to_string(), lashlang::Value::String(".".into()));
        let (registration, label) = ctx
            .prepare_lashlang_process_start(lashlang::ProcessBlockStart {
                program: lashlang::Program::block(Vec::new()),
                tool_names: vec!["alpha".to_string()],
                name: Some(lashlang::Value::String("scan".into())),
                timeout_ms: Some(lashlang::Value::Number(42.0)),
                input: Some(lashlang::Value::Record(Arc::new(input))),
            })
            .expect("process start should prepare");

        assert_eq!(label.as_deref(), Some("scan"));
        assert!(
            registration
                .event_types
                .iter()
                .any(|event_type| event_type.name == "process.wake")
        );
        let crate::ProcessInput::LashlangBlock {
            input,
            tool_bindings,
            timeout_ms,
            display_name,
            ..
        } = registration.input
        else {
            panic!("expected lashlang process input");
        };
        assert_eq!(input.get("root"), Some(&serde_json::json!(".")));
        assert_eq!(timeout_ms, Some(42));
        assert_eq!(display_name.as_deref(), Some("scan"));
        assert_eq!(tool_bindings.len(), 1);
        assert_eq!(tool_bindings[0].name, "alpha");
        assert_eq!(tool_bindings[0].tool_id, crate::ToolId::new("tool:alpha"));
    }
}
