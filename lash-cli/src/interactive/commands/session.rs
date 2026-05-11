use lash::session_model::Message;
use lash::{CachedModelCatalog, SessionPolicy, ToolState};
use lash_embed::{LashSession, advanced::ExecutionMode, provider::ProviderHandle};
use tokio::task;
use tokio_util::sync::CancellationToken;

use crate::app::{App, UiTimelineItem};
use crate::fork;
use crate::resume;
use crate::session_bootstrap::{CliSessionOpener, OpenedCliLashSession};
use crate::session_log::{self, SessionLogger};
use crate::turn_runner::RuntimeRunResult;
use crate::{SkillCatalog, hash12, push_system_message};

use super::super::helpers::TurnReplayPayload;
use super::super::runtime::{apply_pending_reconfigure, send_user_message};

#[allow(clippy::too_many_arguments)]
async fn activate_opened_session(
    opened: OpenedCliLashSession,
    app: &mut App,
    logger: &mut SessionLogger,
    runtime: &mut Option<LashSession>,
    history: &mut Vec<Message>,
    turn_counter: &mut usize,
    current_execution_mode: &mut ExecutionMode,
    current_model_variant: &mut Option<String>,
    desired_tool_state: &mut ToolState,
    model_catalog: &CachedModelCatalog,
    provider: &ProviderHandle,
) -> Result<(), String> {
    *runtime = Some(opened.session);
    *logger = opened.logger;
    resume::load_resumed_session(
        opened.bootstrap.filename(),
        app,
        logger,
        history,
        runtime,
        turn_counter,
        current_execution_mode,
        provider,
        current_model_variant,
        desired_tool_state,
        model_catalog,
    )
    .await?;
    let Some(session) = runtime.as_ref() else {
        return Err("opened session was not installed".to_string());
    };
    session
        .control()
        .tools()
        .refresh_surface()
        .await
        .map_err(|err| err.to_string())?;
    *desired_tool_state = session
        .control()
        .tools()
        .state()
        .await
        .map_err(|err| err.to_string())?;
    Ok(())
}

fn fallback_policy_for_session_switch(
    _runtime: &Option<LashSession>,
    provider: &ProviderHandle,
    app: &App,
    current_model_variant: &mut Option<String>,
    current_execution_mode: &mut ExecutionMode,
) -> SessionPolicy {
    SessionPolicy {
        provider: provider.clone(),
        model: app.model.clone(),
        model_variant: current_model_variant.clone(),
        max_context_tokens: app.context_window.map(|window| window as usize),
        execution_mode: current_execution_mode.clone(),
        ..SessionPolicy::default()
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn handle_clear(
    app: &mut App,
    runtime_factory: &CliSessionOpener,
    logger: &mut SessionLogger,
    runtime: &mut Option<LashSession>,
    history: &mut Vec<Message>,
    turn_counter: &mut usize,
    last_turn: &mut Option<TurnReplayPayload>,
    active_stream_id: &mut u64,
    provider: &ProviderHandle,
    current_model_variant: &mut Option<String>,
    current_execution_mode: &mut ExecutionMode,
    desired_tool_state: &mut ToolState,
    pending_clear_after_return: &mut bool,
) -> anyhow::Result<bool> {
    *active_stream_id = active_stream_id.wrapping_add(1);
    if runtime.is_none() {
        *pending_clear_after_return = true;
        return Ok(false);
    }
    let policy = fallback_policy_for_session_switch(
        runtime,
        provider,
        app,
        current_model_variant,
        current_execution_mode,
    );
    let opened = runtime_factory.fresh(policy).await?;
    *runtime = Some(opened.session);
    *logger = opened.logger;
    let Some(session) = runtime.as_ref() else {
        return Err(anyhow::anyhow!("opened session was not installed"));
    };
    if let Some(rt) = runtime.as_ref() {
        rt.control()
            .tools()
            .refresh_surface()
            .await
            .map_err(|err| anyhow::anyhow!(err.to_string()))?;
        let session_id = rt.session_id();
        let policy = rt.policy_snapshot();
        app.session_id = session_id;
        *current_execution_mode = policy.execution_mode;
        *current_model_variant = policy.model_variant;
    }
    app.session_name = opened.bootstrap.session_name();
    *desired_tool_state = session.control().tools().state().await?;
    history.clear();
    *turn_counter = 0;
    *last_turn = None;
    app.clear();
    app.set_model_variant(current_model_variant.clone());
    app.timeline.push(UiTimelineItem::SystemMessage(format!(
        "Started new session: {}",
        app.session_name
    )));
    *pending_clear_after_return = false;
    Ok(false)
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn handle_retry(
    app: &mut App,
    logger: &mut SessionLogger,
    runtime: &mut Option<LashSession>,
    history: &mut Vec<Message>,
    last_turn: &Option<TurnReplayPayload>,
    runtime_return_rx: &mut Option<tokio::sync::oneshot::Receiver<RuntimeRunResult>>,
    cancel_token: &mut Option<CancellationToken>,
    active_stream_id: &mut u64,
    current_execution_mode: &mut ExecutionMode,
    desired_tool_state: &mut ToolState,
    pending_reconfigure: &mut bool,
    toolset_hash: &mut String,
    app_tx: &crate::event::AppEventTx,
) -> anyhow::Result<bool> {
    if let Some(previous) = last_turn.clone() {
        if let Err(e) =
            apply_pending_reconfigure(desired_tool_state, pending_reconfigure, runtime).await
        {
            push_system_message(
                app,
                format!("Pending runtime reconfigure failed; retry blocked: {}", e),
            );
            return Ok(false);
        }
        let definitions = match runtime.as_ref() {
            Some(session) => session
                .control()
                .tools()
                .active_definitions()
                .await
                .unwrap_or_default(),
            None => Vec::new(),
        };
        *toolset_hash =
            hash12(&serde_json::to_vec(&definitions).unwrap_or_else(|_| b"[]".to_vec()));
        *current_execution_mode = previous.execution_mode;
        let current_tool_state = desired_tool_state.clone();
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
            &current_tool_state,
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
    runtime: &mut Option<LashSession>,
    provider: &ProviderHandle,
    current_model_variant: &Option<String>,
    _toolset_hash: &str,
) -> anyhow::Result<bool> {
    match fork::fork_current_session(
        runtime.as_ref(),
        logger,
        provider,
        &app.model,
        app.context_window
            .expect("app context_window must be set before forking"),
        current_model_variant.as_deref(),
    )
    .await
    {
        Ok(forked) => {
            let fallback_command = fork_resume_command(&forked.session_id);
            let exe = match fork::resolve_resume_executable() {
                Ok(exe) => exe,
                Err(err) => {
                    push_system_message(
                        app,
                        fork_launch_fallback_message(
                            &forked,
                            &fallback_command,
                            &format!("launcher lookup failed: {}", err),
                        ),
                    );
                    return Ok(false);
                }
            };
            let child_args = vec!["--resume".to_string(), forked.session_id.clone()];
            match fork::spawn_in_new_terminal(&exe, &child_args) {
                Ok(()) => push_system_message(
                    app,
                    format!(
                        "Forked into `{}` ({})",
                        forked.session_name, forked.session_id
                    ),
                ),
                Err(err) => push_system_message(
                    app,
                    fork_launch_fallback_message(
                        &forked,
                        &fallback_command,
                        &format!("launch failed: {}", err),
                    ),
                ),
            }
        }
        Err(err) => push_system_message(app, format!("Fork failed: {}", err)),
    }
    Ok(false)
}

fn fork_resume_command(session_id: &str) -> String {
    format!("lash --resume {session_id}")
}

fn fork_launch_fallback_message(
    forked: &fork::ForkedSession,
    fallback_command: &str,
    reason: &str,
) -> String {
    format!(
        "Fork `{}` was created, but {}.\nResume it with:\n{}",
        forked.session_name, reason, fallback_command
    )
}

pub(super) async fn handle_tree(
    app: &mut App,
    runtime: &Option<LashSession>,
) -> anyhow::Result<bool> {
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
    let roots = crate::tree::current_message_tree(rt).await;
    if roots.is_empty() {
        push_system_message(app, "No messages yet.");
    } else {
        app.show_tree(roots);
    }
    Ok(false)
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn switch_to_session_identifier(
    identifier: &str,
    app: &mut App,
    logger: &mut SessionLogger,
    runtime_factory: &CliSessionOpener,
    runtime: &mut Option<LashSession>,
    history: &mut Vec<Message>,
    turn_counter: &mut usize,
    provider: &ProviderHandle,
    current_model_variant: &mut Option<String>,
    current_execution_mode: &mut ExecutionMode,
    desired_tool_state: &mut ToolState,
    model_catalog: &CachedModelCatalog,
    toolset_hash: &mut String,
) -> anyhow::Result<()> {
    let policy = fallback_policy_for_session_switch(
        runtime,
        provider,
        app,
        current_model_variant,
        current_execution_mode,
    );
    let opened = runtime_factory.resume(identifier, policy).await?;
    activate_opened_session(
        opened,
        app,
        logger,
        runtime,
        history,
        turn_counter,
        current_execution_mode,
        current_model_variant,
        desired_tool_state,
        model_catalog,
        provider,
    )
    .await
    .map_err(anyhow::Error::msg)?;
    *toolset_hash = hash12(
        &serde_json::to_vec(&desired_tool_state.definitions()).unwrap_or_else(|_| b"[]".to_vec()),
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn handle_resume(
    name: Option<String>,
    app: &mut App,
    logger: &mut SessionLogger,
    runtime_factory: &CliSessionOpener,
    runtime: &mut Option<LashSession>,
    history: &mut Vec<Message>,
    turn_counter: &mut usize,
    last_turn: &mut Option<TurnReplayPayload>,
    provider: &ProviderHandle,
    current_model_variant: &mut Option<String>,
    current_execution_mode: &mut ExecutionMode,
    desired_tool_state: &mut ToolState,
    model_catalog: &CachedModelCatalog,
    toolset_hash: &mut String,
) -> anyhow::Result<bool> {
    if let Some(filename) = name {
        match switch_to_session_identifier(
            &filename,
            app,
            logger,
            runtime_factory,
            runtime,
            history,
            turn_counter,
            provider,
            current_model_variant,
            current_execution_mode,
            desired_tool_state,
            model_catalog,
            toolset_hash,
        )
        .await
        {
            Ok(()) => {
                *last_turn = None;
                app.dirty = true;
            }
            Err(err) => {
                app.timeline
                    .push(UiTimelineItem::SystemMessage(err.to_string()));
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
            app.timeline.push(UiTimelineItem::SystemMessage(
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
        app.timeline.push(UiTimelineItem::SystemMessage(
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fork_fallback_message_uses_resume_id_command() {
        let forked = fork::ForkedSession {
            session_id: "session-123".to_string(),
            session_name: "quiet-forest".to_string(),
        };
        let command = fork_resume_command(&forked.session_id);

        assert_eq!(command, "lash --resume session-123");
        let message = fork_launch_fallback_message(&forked, &command, "launch failed: no tty");
        assert!(message.contains("Fork `quiet-forest` was created"));
        assert!(message.contains("Resume it with:\nlash --resume session-123"));
        assert!(!message.contains(".db"));
    }
}
