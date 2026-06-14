use lash::LashSession;
use lash_core::runtime::RuntimeSessionState;
use lash_core::{Message, MessageRole, SessionMessageTreeNode};

use crate::app::{App, timeline_from_read_view};
use crate::overlay::TreeSelection;
use crate::persistence::persist_committed_runtime_state;
use crate::session_log::SessionLogger;

pub async fn current_message_tree(session: &LashSession) -> Vec<SessionMessageTreeNode> {
    session.read_view().message_tree()
}

#[allow(clippy::too_many_arguments)]
pub async fn switch_to_tree_selection(
    session: &LashSession,
    logger: &SessionLogger,
    app: &mut App,
    history: &mut Vec<Message>,
    selection: TreeSelection,
) -> Result<(), String> {
    let target_leaf = if matches!(selection.message.role, MessageRole::User) {
        selection.parent_node_id.clone()
    } else {
        Some(selection.node_id.clone())
    };
    let seeded_input = if matches!(selection.message.role, MessageRole::User) {
        crate::overlay::tree_message_preview(&selection.message)
    } else {
        String::new()
    };

    // Fast path: if the target leaf already matches the runtime's
    // current leaf AND we have nothing to seed into the editor, the
    // branch is a no-op. Skip the full `branch_to_node` rebuild —
    // `from_state` walks the plugin host and re-projects the
    // transcript, which is expensive to pay for a visible no-op.
    let current_leaf = session
        .read_view()
        .materialized_session_graph()
        .leaf_node_id
        .clone();
    if current_leaf == target_leaf && seeded_input.is_empty() {
        return Ok(());
    }

    let state = session
        .admin()
        .state()
        .branch_to_node(target_leaf)
        .await
        .map_err(|err| err.to_string())?;
    let read_view = state.read_view();
    *history = read_view.messages().to_vec();

    app.stop_turn();
    app.timeline = timeline_from_read_view(&read_view, &app.ui_projection_state());
    app.usage.token_usage = state.token_usage.clone();
    app.usage.last_prompt_usage = state.last_prompt_usage.clone();
    // Branching to a different leaf means the handle maps from the
    // old path (shell sessions, subagent tasks) are no longer valid
    // — reset them so a stale id from the abandoned branch doesn't
    // leak into the new branch's projector dispatch.
    app.activity_state.reset();
    app.editor.pending_images.clear();
    app.editor.pending_large_pastes.clear();
    app.set_input(seeded_input);
    app.update_suggestions();
    app.invalidate_height_cache();
    app.scroll_to_bottom();

    let mut persistence_state = RuntimeSessionState::from_snapshot(state);
    persist_committed_runtime_state(logger.store().as_ref(), &mut persistence_state).await;

    Ok(())
}
