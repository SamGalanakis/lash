use std::sync::Arc;

use lash::session_model::Message;
use lash::*;
use tokio::sync::mpsc;
use tokio::task;
use tokio_util::sync::CancellationToken;

use crate::app::{App, DisplayBlock};
use crate::event::AppEvent;
use crate::fork;
use crate::resume;
use crate::session_log::{self, SessionLogger};
use crate::turn_runner::RuntimeRunResult;
use crate::{hash12, push_system_message, sync_ui_extensions};

use super::super::helpers::TurnReplayPayload;
use super::super::runtime::{apply_pending_reconfigure, send_user_message};

#[allow(clippy::too_many_arguments)]
pub(super) async fn handle_clear(
    app: &mut App,
    plugin_host: &PluginHost,
    runtime: &mut Option<LashRuntime>,
    history: &mut Vec<Message>,
    turn_counter: &mut usize,
    last_turn: &mut Option<TurnReplayPayload>,
    active_stream_id: &mut u64,
    current_model_variant: &Option<String>,
    current_execution_mode: &ExecutionMode,
    session_manager: &mut Arc<dyn SessionManager>,
    pending_clear_after_return: &mut bool,
) -> anyhow::Result<bool> {
    app.clear();
    app.set_model_variant(current_model_variant.clone());
    history.clear();
    *turn_counter = 0;
    *last_turn = None;
    app.token_usage = TokenUsage::default();
    *active_stream_id = active_stream_id.wrapping_add(1);
    if let Some(rt) = runtime.as_mut() {
        let _ = rt.reset_session().await;
        let mut state = SessionStateEnvelope {
            session_id: "root".to_string(),
            policy: SessionPolicy {
                execution_mode: *current_execution_mode,
                ..rt.export_state().policy
            },
            session_graph: lash::SessionGraph::default(),
            iteration: *turn_counter,
            token_usage: app.token_usage.clone(),
            last_prompt_usage: None,
        };
        state.replace_projection(history, &[]);
        rt.set_persisted_state(lash::PersistedSessionState::from_state(state));
        match rt.session_manager() {
            Ok(manager) => *session_manager = manager,
            Err(err) => {
                push_system_message(app, format!("Failed to refresh session manager: {}", err))
            }
        }
        let ui_extensions = app.ui_extensions_handle();
        sync_ui_extensions(
            app,
            ui_extensions.as_ref(),
            plugin_host,
            Arc::clone(session_manager),
        )
        .await;
        *pending_clear_after_return = false;
    } else {
        *pending_clear_after_return = true;
    }
    Ok(false)
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn handle_retry(
    app: &mut App,
    logger: &mut SessionLogger,
    dynamic_tools: &Arc<DynamicToolProvider>,
    runtime: &mut Option<LashRuntime>,
    history: &mut Vec<Message>,
    last_turn: &Option<TurnReplayPayload>,
    runtime_return_rx: &mut Option<tokio::sync::oneshot::Receiver<RuntimeRunResult>>,
    cancel_token: &mut Option<CancellationToken>,
    active_stream_id: &mut u64,
    current_execution_mode: &mut ExecutionMode,
    desired_dynamic: &mut DynamicStateSnapshot,
    pending_reconfigure: &mut bool,
    toolset_hash: &mut String,
    app_tx: &mpsc::UnboundedSender<AppEvent>,
) -> anyhow::Result<bool> {
    if let Some(previous) = last_turn.clone() {
        if let Err(e) =
            apply_pending_reconfigure(dynamic_tools, desired_dynamic, pending_reconfigure, runtime)
                .await
        {
            push_system_message(
                app,
                format!("Pending runtime reconfigure failed; retry blocked: {}", e),
            );
            return Ok(false);
        }
        *toolset_hash = hash12(
            &serde_json::to_vec(&dynamic_tools.definitions()).unwrap_or_else(|_| b"[]".to_vec()),
        );
        *current_execution_mode = previous.execution_mode;
        let current_dynamic_state = dynamic_tools.export_state();
        send_user_message(
            previous.prepared_turn.clone(),
            previous.turn_input.clone(),
            app,
            None,
            logger,
            runtime,
            history,
            runtime_return_rx,
            cancel_token,
            active_stream_id,
            app_tx,
            &current_dynamic_state,
        )
        .await;
    } else {
        push_system_message(app, "No previous turn payload to retry yet.");
    }
    Ok(false)
}

pub(super) async fn handle_fork(
    app: &mut App,
    logger: &mut SessionLogger,
    _dynamic_tools: &Arc<DynamicToolProvider>,
    runtime: &mut Option<LashRuntime>,
    provider: &ProviderHandle,
    current_model_variant: &Option<String>,
    _toolset_hash: &str,
) -> anyhow::Result<bool> {
    match fork::fork_current_session(
        runtime.as_mut(),
        logger,
        provider,
        &app.model,
        app.context_window
            .expect("app context_window must be set before forking"),
        current_model_variant.as_deref(),
    )
    .await
    {
        Ok((child_filename, child_session_name)) => {
            let exe = match std::env::current_exe() {
                Ok(exe) => exe,
                Err(err) => {
                    push_system_message(
                        app,
                        format!("Fork created but launcher lookup failed: {}", err),
                    );
                    return Ok(false);
                }
            };
            let child_args = vec!["--resume".to_string(), child_filename.clone()];
            match fork::spawn_in_new_terminal(&exe, &child_args) {
                Ok(()) => push_system_message(
                    app,
                    format!("Forked into `{}` ({})", child_session_name, child_filename),
                ),
                Err(err) => {
                    push_system_message(app, format!("Fork created but launch failed: {}", err))
                }
            }
        }
        Err(err) => push_system_message(app, format!("Fork failed: {}", err)),
    }
    Ok(false)
}

pub(super) fn handle_tree(app: &mut App, runtime: &Option<LashRuntime>) -> anyhow::Result<bool> {
    if app.has_prompt() {
        push_system_message(app, "Close the active prompt before opening /tree.");
        return Ok(false);
    }
    let Some(rt) = runtime.as_ref() else {
        push_system_message(
            app,
            "Branch navigation is unavailable while a turn is running.",
        );
        return Ok(false);
    };
    let roots = crate::tree::current_message_tree(rt);
    if roots.is_empty() {
        push_system_message(app, "No messages yet.");
    } else {
        app.show_tree(roots);
    }
    Ok(false)
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn handle_resume(
    name: Option<String>,
    app: &mut App,
    logger: &SessionLogger,
    plugin_host: &PluginHost,
    dynamic_tools: &Arc<DynamicToolProvider>,
    runtime: &mut Option<LashRuntime>,
    history: &mut Vec<Message>,
    turn_counter: &mut usize,
    last_turn: &mut Option<TurnReplayPayload>,
    provider: &ProviderHandle,
    current_model_variant: &mut Option<String>,
    current_execution_mode: &mut ExecutionMode,
    session_manager: &mut Arc<dyn SessionManager>,
    desired_dynamic: &mut DynamicStateSnapshot,
    model_catalog: &CachedModelCatalog,
    toolset_hash: &mut String,
) -> anyhow::Result<bool> {
    if let Some(filename) = name {
        match resume::load_resumed_session(
            &filename,
            app,
            history,
            runtime,
            turn_counter,
            current_execution_mode,
            provider,
            current_model_variant,
            dynamic_tools,
            desired_dynamic,
            model_catalog,
        )
        .await
        {
            Ok(()) => {
                *last_turn = None;
                if let Some(rt) = runtime.as_ref() {
                    match rt.session_manager() {
                        Ok(manager) => *session_manager = manager,
                        Err(err) => push_system_message(
                            app,
                            format!("Failed to refresh session manager: {}", err),
                        ),
                    }
                }
                let ui_extensions = app.ui_extensions_handle();
                sync_ui_extensions(
                    app,
                    ui_extensions.as_ref(),
                    plugin_host,
                    Arc::clone(session_manager),
                )
                .await;
                *toolset_hash = hash12(
                    &serde_json::to_vec(&dynamic_tools.definitions())
                        .unwrap_or_else(|_| b"[]".to_vec()),
                );
            }
            Err(err) => {
                app.blocks.push(DisplayBlock::SystemMessage(err));
                app.invalidate_height_cache();
                app.scroll_to_bottom();
            }
        }
    } else {
        const SESSION_PICKER_LIMIT: usize = 50;
        let current_session_id = logger.session_id.clone();
        let mut sessions = task::spawn_blocking(move || {
            let mut s = session_log::list_recent_sessions(SESSION_PICKER_LIMIT + 1);
            s.retain(|si| si.session_id != current_session_id);
            s
        })
        .await
        .unwrap_or_default();
        if sessions.is_empty() {
            app.blocks.push(DisplayBlock::SystemMessage(
                "No sessions found.".to_string(),
            ));
            app.invalidate_height_cache();
            app.scroll_to_bottom();
        } else {
            sessions.truncate(SESSION_PICKER_LIMIT);
            app.show_session_picker(sessions);
        }
    }
    Ok(false)
}

pub(super) fn handle_skills(app: &mut App) -> anyhow::Result<bool> {
    app.skills = SkillCatalog::from_dirs(&crate::paths::default_skill_dirs());
    let items: Vec<(String, String)> = app
        .skills
        .iter()
        .map(|s| (s.name.clone(), s.description.clone()))
        .collect();
    if items.is_empty() {
        app.blocks.push(DisplayBlock::SystemMessage(
            "No skills found.\n\
             Add skill directories to ~/.lash/skills/ or .agents/lash/skills/\n\
             Each skill is a directory with a SKILL.md file."
                .to_string(),
        ));
        app.invalidate_height_cache();
        app.scroll_to_bottom();
    } else {
        app.show_skill_picker(items);
    }
    Ok(false)
}
