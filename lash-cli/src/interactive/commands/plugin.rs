use std::sync::Arc;

use lash::session_model::Message;
use lash::*;

use crate::app::{App, projected_blocks_from_state};
use crate::push_system_message;

use super::super::runtime::sync_runtime_tool_surface;

pub(super) async fn handle_plugin(
    name: String,
    argument: Option<String>,
    app: &mut App,
    runtime: &mut Option<LashRuntime>,
    history: &mut Vec<Message>,
    session_manager: &Arc<dyn SessionManager>,
) -> anyhow::Result<bool> {
    // `/compact` is a built-in plugin command but its handler needs
    // to mutate the live runtime state, which the in-tree plugin
    // command surface cannot do today. Route it through
    // `LashRuntime::rewrite_history` directly so the rewritten
    // messages take effect immediately.
    if name == "/compact" {
        let Some(rt) = runtime.as_mut() else {
            push_system_message(app, "Compaction is unavailable while a turn is running.");
            return Ok(false);
        };
        let trigger = lash::RewriteTrigger::Manual {
            instructions: argument,
        };
        match rt.rewrite_history(trigger).await {
            Ok(true) => {
                let state = rt.export_state();
                history.clear();
                let projected_messages = state.project_messages();
                let projected_tool_calls = state.project_tool_calls();
                history.extend(projected_messages.clone());
                app.blocks = projected_blocks_from_state(
                    &projected_messages,
                    &projected_tool_calls,
                    &app.ui_resume_state(),
                );
                app.invalidate_height_cache();
                app.scroll_to_bottom();
                push_system_message(app, "Compaction summary inserted.");
            }
            Ok(false) => push_system_message(
                app,
                "Nothing to compact yet — the conversation is still short.",
            ),
            Err(err) => push_system_message(app, format!("Compaction failed: {err}")),
        }
        return Ok(false);
    }
    let plugin_session = runtime.as_ref().and_then(|rt| rt.plugin_session());
    let Some(plugin_session) = plugin_session else {
        push_system_message(
            app,
            format!("Plugin command `{name}` is unavailable (no active session)."),
        );
        return Ok(false);
    };
    match plugin_session
        .invoke_command(&name, argument, Arc::clone(session_manager))
        .await
    {
        Ok(outcome) => {
            if let Err(err) = sync_runtime_tool_surface(runtime).await {
                push_system_message(app, format!("Failed to sync tool surface: {err}"));
            }
            match outcome {
                lash::CommandOutcome::Handled => {}
                lash::CommandOutcome::Message(msg) => push_system_message(app, msg),
                lash::CommandOutcome::Error(msg) => {
                    push_system_message(app, format!("Plugin command `{name}` failed: {msg}"));
                }
            }
        }
        Err(err) => {
            push_system_message(app, format!("Plugin command `{name}` error: {err}"));
        }
    }
    Ok(false)
}
