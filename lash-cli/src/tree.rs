use lash::{DynamicStateSnapshot, LashRuntime, Message, MessageRole, SessionMessageTreeNode};

use crate::app::{App, projected_blocks_from_state};
use crate::overlay::TreeSelection;
use crate::persist_root_session_state;
use crate::session_log::SessionLogger;

pub fn current_message_tree(runtime: &LashRuntime) -> Vec<SessionMessageTreeNode> {
    let mut state = runtime.export_state();
    state
        .session_graph
        .merge_active_projection(&state.messages, &state.tool_calls);
    state.session_graph.message_tree()
}

#[allow(clippy::too_many_arguments)]
pub async fn switch_to_tree_selection(
    runtime: &mut LashRuntime,
    logger: &SessionLogger,
    app: &mut App,
    history: &mut Vec<Message>,
    selection: TreeSelection,
    dynamic_state: &DynamicStateSnapshot,
) -> Result<(), String> {
    let target_leaf = if matches!(selection.message.role, MessageRole::User) {
        selection.parent_node_id.clone()
    } else {
        Some(selection.node_id.clone())
    };
    let seeded_input = if matches!(selection.message.role, MessageRole::User) {
        selection
            .message
            .user_input
            .as_ref()
            .map(|input| input.display_text.clone())
            .filter(|text| !text.trim().is_empty())
            .unwrap_or_else(|| crate::overlay::tree_message_preview(&selection.message))
    } else {
        String::new()
    };

    // Fast path: if the target leaf already matches the runtime's
    // current leaf AND we have nothing to seed into the editor, the
    // branch is a no-op. Skip the full `branch_to_node` rebuild —
    // `from_state` walks the plugin host and re-projects the
    // transcript, which is expensive to pay for a visible no-op.
    let current_leaf = runtime.export_state().session_graph.leaf_node_id.clone();
    if current_leaf == target_leaf && seeded_input.is_empty() {
        return Ok(());
    }

    let mut state = runtime
        .branch_to_node(target_leaf)
        .await
        .map_err(|err| err.to_string())?;
    *history = state.messages.clone();

    app.stop_turn();
    app.blocks =
        projected_blocks_from_state(&state.messages, &state.tool_calls, &app.ui_resume_state());
    app.token_usage = state.token_usage.clone();
    app.last_prompt_usage = state.last_prompt_usage.clone();
    // Branching to a different leaf means the handle maps from the
    // old path (shell sessions, delegate tasks) are no longer valid
    // — reset them so a stale id from the abandoned branch doesn't
    // leak into the new branch's projector dispatch.
    app.activity_state.reset();
    app.editor.pending_images.clear();
    app.editor.pending_large_pastes.clear();
    app.set_input(seeded_input);
    app.update_suggestions();
    app.invalidate_height_cache();
    app.scroll_to_bottom();

    persist_root_session_state(
        logger.store().as_ref(),
        &mut state,
        &app.ui_resume_state(),
        dynamic_state,
    );

    Ok(())
}
