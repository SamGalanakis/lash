use std::sync::Arc;

use tokio::sync::{mpsc::Sender, mpsc::UnboundedSender};
use tokio_util::sync::CancellationToken;

use super::async_handles::AsyncToolHandleMap;
use crate::tool_dispatch::ToolDispatchContext;
use crate::{SandboxMessage, TurnActivity, TurnActivityId, TurnEvent};

#[derive(Clone)]
pub struct ModeExecutionContext {
    pub(super) session_id: String,
    pub(super) execution_mode: crate::ExecutionMode,
    pub(super) dispatch: Arc<ToolDispatchContext>,
    pub(super) async_tool_handles: AsyncToolHandleMap,
    pub(super) message_tx: Option<UnboundedSender<SandboxMessage>>,
    attachment_store: Arc<dyn crate::AttachmentStore>,
    chronological_projection: Arc<crate::ChronologicalProjection>,
    mode_extension: Option<crate::ModeTurnExtensionHandle>,
    turn_context: crate::TurnContext,
    pub(super) turn_event_tx: Option<Sender<TurnActivity>>,
    pub(super) cancellation_token: Option<CancellationToken>,
}

impl ModeExecutionContext {
    #[allow(
        clippy::too_many_arguments,
        reason = "mode execution bridge carries explicit per-turn runtime dependencies"
    )]
    pub(super) fn new(
        session_id: String,
        execution_mode: crate::ExecutionMode,
        dispatch: Arc<ToolDispatchContext>,
        async_tool_handles: AsyncToolHandleMap,
        message_tx: Option<UnboundedSender<SandboxMessage>>,
        attachment_store: Arc<dyn crate::AttachmentStore>,
        chronological_projection: Arc<crate::ChronologicalProjection>,
        mode_extension: Option<crate::ModeTurnExtensionHandle>,
        turn_context: crate::TurnContext,
    ) -> Self {
        Self {
            session_id,
            execution_mode,
            dispatch,
            async_tool_handles,
            message_tx,
            attachment_store,
            chronological_projection,
            mode_extension,
            turn_context,
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

    pub(crate) fn with_cancellation_token(mut self, cancellation_token: CancellationToken) -> Self {
        self.cancellation_token = Some(cancellation_token);
        self
    }

    pub(crate) fn tool_execution_mode(&self, name: &str) -> crate::ToolExecutionMode {
        crate::tool_dispatch::resolve_tool_execution_mode(&self.dispatch, name)
    }

    pub fn tool_argument_projection_policy(
        &self,
        name: &str,
    ) -> crate::ToolArgumentProjectionPolicy {
        crate::tool_dispatch::resolve_tool_argument_projection_policy(&self.dispatch, name)
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
            Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
            None,
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
}
