use std::sync::Arc;

use crate::{MessageSequence, SessionGraph, SessionReadView, SessionStateEnvelope, ToolCallRecord};

use super::{LashRuntime, PersistedSessionState};

#[derive(Clone, Debug)]
pub(super) struct RuntimeReadSnapshot {
    state: SessionStateEnvelope,
}

impl RuntimeReadSnapshot {
    pub(super) fn from_persisted(state: &PersistedSessionState) -> Self {
        Self {
            state: state.export_state(),
        }
    }

    pub(super) fn from_exported(state: &SessionStateEnvelope) -> Self {
        Self {
            state: state.clone(),
        }
    }

    pub(super) fn with_policy(mut self, policy: crate::SessionPolicy) -> Self {
        self.state.policy = policy;
        self
    }

    pub(super) fn with_iteration(mut self, iteration: usize) -> Self {
        self.state.iteration = iteration;
        self
    }

    pub(super) fn with_mode_turn_options(mut self, options: crate::ModeTurnOptions) -> Self {
        self.state.mode_turn_options = options;
        self
    }

    pub(super) fn read_view(&self) -> SessionReadView {
        SessionReadView::from_state(&self.state)
    }

    pub(super) fn derived_message_view(
        &self,
        base_graph: Arc<SessionGraph>,
        messages: MessageSequence,
        tool_calls: Arc<Vec<ToolCallRecord>>,
    ) -> SessionReadView {
        SessionReadView::from_graph_message_sequence(&self.state, base_graph, messages, tool_calls)
    }

    #[cfg(test)]
    pub(super) fn export_state(&self) -> SessionStateEnvelope {
        self.state.clone()
    }
}

impl LashRuntime {
    pub(super) fn read_snapshot(&self) -> RuntimeReadSnapshot {
        RuntimeReadSnapshot::from_persisted(&self.state)
            .with_policy(self.policy.clone())
            .with_mode_turn_options(self.mode_turn_options.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Message, MessageRole, Part, PartKind, PruneState, SessionPolicy};

    fn text_message(id: &str, role: MessageRole, content: &str) -> Message {
        Message {
            id: id.to_string(),
            role,
            parts: vec![Part {
                id: format!("{id}.p0"),
                kind: PartKind::Text,
                content: content.to_string(),
                attachment: None,
                tool_call_id: None,
                tool_name: None,
                tool_item_id: None,
                tool_signature: None,
                prune_state: PruneState::Intact,
                reasoning_meta: None,
                response_meta: None,
            }]
            .into(),
            user_input: None,
            origin: None,
        }
    }

    #[test]
    fn snapshot_stamps_runtime_metadata_on_exported_state() {
        let mut policy = SessionPolicy::default();
        policy.model = "snapshot-model".to_string();
        let snapshot = RuntimeReadSnapshot::from_persisted(&PersistedSessionState {
            session_id: "session-1".to_string(),
            iteration: 3,
            ..PersistedSessionState::default()
        })
        .with_policy(policy.clone())
        .with_iteration(7)
        .with_mode_turn_options(crate::ModeTurnOptions::Unit);

        let exported = snapshot.export_state();
        assert_eq!(exported.session_id, "session-1");
        assert_eq!(exported.policy.model, "snapshot-model");
        assert_eq!(exported.iteration, 7);
    }

    #[test]
    fn derived_message_view_projects_messages_and_tool_calls_with_metadata() {
        let user = text_message("u1", MessageRole::User, "hello");
        let assistant = text_message("a1", MessageRole::Assistant, "hi");
        let tool_call = ToolCallRecord {
            call_id: Some("call-1".to_string()),
            tool: "grep".to_string(),
            args: serde_json::json!({ "query": "hello" }),
            result: serde_json::json!({ "matches": [] }),
            success: true,
            duration_ms: 4,
        };
        let base_graph = Arc::new(SessionGraph::from_active_read_state(
            std::slice::from_ref(&user),
            &[],
        ));
        let snapshot = RuntimeReadSnapshot::from_persisted(&PersistedSessionState {
            session_id: "session-1".to_string(),
            iteration: 2,
            session_graph: base_graph.as_ref().clone(),
            ..PersistedSessionState::default()
        });

        let view = snapshot.derived_message_view(
            Arc::clone(&base_graph),
            MessageSequence::from_base(vec![user, assistant].into()),
            Arc::new(vec![tool_call.clone()]),
        );

        assert_eq!(view.session_id(), "session-1");
        assert_eq!(view.iteration(), 2);
        assert_eq!(
            view.messages()
                .iter()
                .map(|message| message.id.as_str())
                .collect::<Vec<_>>(),
            vec!["u1", "a1"]
        );
        assert_eq!(view.tool_calls(), std::slice::from_ref(&tool_call));
    }
}
