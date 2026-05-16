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
}
