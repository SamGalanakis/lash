//! The default input composer: sending, queueing, injecting, and editing
//! turns, plus slash-command dispatch and `!` shell escapes.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use lash::LashSession;
use lash_tui_extensions::TuiExtensions;

use crate::app::{App, PreparedTurn, TurnSubmissionRoute, UiTimelineItem};
use crate::editor::SuggestionKind;
use crate::turn_runner::make_turn_input;
use crate::ui_effects::push_system_message;
use crate::ui_trace::UiTraceRecorder;

use crate::interactive::commands::{
    ParsedSlashCommand, handle_parsed_slash_command, parse_slash_command,
    slash_command_runs_out_of_band_while_running,
};
use crate::interactive::helpers::{
    TurnReplayPayload, record_queue_current_turn_input, record_queue_turn,
};
use crate::interactive::runtime::{
    enqueue_prepared_turn, make_injected_plugin_message, refresh_queued_work_snapshot,
    send_user_message,
};

use super::SessionCtx;
use super::modals::{
    can_focus_process_dock, cancel_selected_process, process_dock_has_focus,
    show_selected_process_overview,
};

pub(in crate::interactive) fn slash_command_blocked_while_working_message(
    command_text: &str,
) -> String {
    let command = command_text
        .split_whitespace()
        .next()
        .filter(|command| command.starts_with('/'))
        .unwrap_or("slash command");
    format!(
        "Cannot run `{command}` while Lash is working. Wait for the current turn to finish or press Esc to interrupt it."
    )
}

fn block_slash_command_while_working(app: &mut App, command_text: &str) {
    push_system_message(
        app,
        slash_command_blocked_while_working_message(command_text),
    );
}

/// `!command` shell escape: the command text when the draft is one.
fn shell_escape_command(input: &str) -> Option<&str> {
    input
        .strip_prefix('!')
        .map(str::trim)
        .filter(|cmd| !cmd.is_empty())
}

async fn enqueue_prepared_turn_for_cli(
    queued: PreparedTurn,
    app: &mut App,
    ui_trace: &mut Option<UiTraceRecorder>,
    runtime: &Option<LashSession>,
    delivery_policy: lash_core::DeliveryPolicy,
    slot_policy: lash_core::SlotPolicy,
    show_preview: bool,
) -> bool {
    let Some(session) = runtime.as_ref() else {
        push_system_message(
            app,
            "Cannot queue this input while the session is switching.".to_string(),
        );
        app.restore_prepared_turn(queued);
        return false;
    };
    match enqueue_prepared_turn(session, &queued, delivery_policy, slot_policy).await {
        Ok(()) => {
            if show_preview {
                record_queue_turn(ui_trace, &queued);
            }
            app.cache_draft_presentation(queued);
            if show_preview && let Err(err) = refresh_queued_work_snapshot(app, runtime).await {
                push_system_message(app, format!("Failed to refresh durable queue: {err}"));
            }
            true
        }
        Err(err) => {
            push_system_message(app, format!("Failed to queue input durably: {err}"));
            app.restore_prepared_turn(queued);
            false
        }
    }
}

/// Cancel the most recently queued full turn and restore it into the
/// composer for editing.
pub(super) async fn restore_last_durable_full_turn(app: &mut App, runtime: &Option<LashSession>) {
    let Some(batch) = app
        .visible_turn_batches_for_editing()
        .max_by_key(|batch| batch.enqueue_seq)
        .cloned()
    else {
        return;
    };
    let Some(session) = runtime.as_ref() else {
        push_system_message(
            app,
            "Cannot edit queued input while the session is switching.".to_string(),
        );
        return;
    };
    match session.cancel_queued_work_batch(&batch.batch_id).await {
        Ok(Some(cancelled)) => {
            if let Some(turn) = app.take_prepared_turn_for_queued_batch(&cancelled) {
                app.restore_prepared_turn(turn);
                app.update_suggestions();
            } else {
                push_system_message(app, "Queued input was cancelled.".to_string());
            }
        }
        Ok(None) => {
            push_system_message(
                app,
                "Queued input is already being processed; it was not restored.".to_string(),
            );
        }
        Err(err) => push_system_message(app, format!("Failed to edit queued input: {err}")),
    }
    if let Err(err) = refresh_queued_work_snapshot(app, runtime).await {
        push_system_message(app, format!("Failed to refresh durable queue: {err}"));
    }
}

pub(in crate::interactive) fn selected_slash_command_suggestion(
    app: &App,
    ui_extensions: &TuiExtensions,
) -> Option<(String, ParsedSlashCommand)> {
    if app.suggestion_kind() != SuggestionKind::Command {
        return None;
    }
    let command_text = app
        .suggestions()
        .get(app.suggestion_idx())
        .map(|suggestion| suggestion.name.clone())?;
    parse_slash_command(&command_text, &app.skills, ui_extensions)
        .map(|command| (command_text, command))
}

/// Record the slash command in the UI trace and dispatch it through the
/// shared slash-command context.
async fn run_slash_command(
    ctx: &mut SessionCtx<'_>,
    cmd: ParsedSlashCommand,
    command_text: &str,
) -> anyhow::Result<bool> {
    if let Some(recorder) = ctx.ui_trace.as_mut() {
        recorder.record_slash_command(command_text.to_string());
    }
    handle_parsed_slash_command(cmd, ctx.slash_ctx()).await
}

/// Default input dispatch when no modal overlay is active: suggestion
/// navigation/completion, Tab/Enter send-or-queue, shell escapes, slash
/// commands, and raw editing keys.
pub(super) async fn handle_input_mode_key(
    key: KeyEvent,
    ctx: &mut SessionCtx<'_>,
) -> anyhow::Result<bool> {
    match key.code {
        // Tab: complete selected suggestion
        KeyCode::Tab if ctx.app.has_suggestions() => {
            if let Some(recorder) = ctx.ui_trace.as_mut() {
                recorder.record_suggestion_complete();
            }
            ctx.app.complete_suggestion();
            ctx.app.update_suggestions();
        }
        KeyCode::Tab if can_focus_process_dock(ctx.app) => {
            ctx.app.select_next_process();
        }
        KeyCode::BackTab if can_focus_process_dock(ctx.app) => {
            ctx.app.select_previous_process();
        }
        KeyCode::Tab => return handle_tab_submit(ctx).await,
        // Up/Down: navigate suggestions when popup is visible
        KeyCode::Up if ctx.app.has_suggestions() => {
            if let Some(recorder) = ctx.ui_trace.as_mut() {
                recorder.record_suggestion_up();
            }
            ctx.app.editor.suggestion_up();
        }
        KeyCode::Down if ctx.app.has_suggestions() => {
            if let Some(recorder) = ctx.ui_trace.as_mut() {
                recorder.record_suggestion_down();
            }
            ctx.app.editor.suggestion_down();
        }
        // Enter with the suggestion popup open accepts the highlighted entry,
        // matching Tab. Shift/Alt+Enter still falls through to insert a newline.
        KeyCode::Enter
            if ctx.app.has_suggestions()
                && !key
                    .modifiers
                    .intersects(KeyModifiers::SHIFT | KeyModifiers::ALT) =>
        {
            return handle_suggestion_enter(ctx).await;
        }
        KeyCode::Enter if process_dock_has_focus(ctx.app) => {
            show_selected_process_overview(ctx.app);
        }
        KeyCode::Delete if process_dock_has_focus(ctx.app) => {
            cancel_selected_process(ctx.app, ctx.runtime).await;
        }
        KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            ctx.app.editor.kill_to_line_start();
            ctx.app.update_suggestions();
        }
        KeyCode::Char('k') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            ctx.app.editor.kill_to_line_end();
            ctx.app.update_suggestions();
        }
        KeyCode::Enter => {
            // Shift+Enter or Alt+Enter → insert newline
            if key.modifiers.contains(KeyModifiers::SHIFT)
                || key.modifiers.contains(KeyModifiers::ALT)
            {
                ctx.app.insert_char('\n');
                ctx.app.update_suggestions();
                return Ok(false);
            }
            return handle_enter_submit(ctx).await;
        }
        KeyCode::Backspace => {
            if let Some(recorder) = ctx.ui_trace.as_mut() {
                recorder.record_input_backspace();
            }
            ctx.app.editor.backspace();
            ctx.app.update_suggestions();
        }
        KeyCode::Delete => {
            if let Some(recorder) = ctx.ui_trace.as_mut() {
                recorder.record_input_delete();
            }
            ctx.app.editor.delete();
            ctx.app.update_suggestions();
        }
        KeyCode::Left if key.modifiers.contains(KeyModifiers::CONTROL) => {
            if let Some(recorder) = ctx.ui_trace.as_mut() {
                recorder.record_move_cursor_left();
            }
            ctx.app.editor.move_cursor_word_left();
        }
        KeyCode::Right if key.modifiers.contains(KeyModifiers::CONTROL) => {
            if let Some(recorder) = ctx.ui_trace.as_mut() {
                recorder.record_move_cursor_right();
            }
            ctx.app.editor.move_cursor_word_right();
        }
        KeyCode::Left => {
            if let Some(recorder) = ctx.ui_trace.as_mut() {
                recorder.record_move_cursor_left();
            }
            ctx.app.editor.move_cursor_left();
        }
        KeyCode::Right => {
            if let Some(recorder) = ctx.ui_trace.as_mut() {
                recorder.record_move_cursor_right();
            }
            ctx.app.editor.move_cursor_right();
        }
        KeyCode::Home => {
            if let Some(recorder) = ctx.ui_trace.as_mut() {
                recorder.record_move_cursor_home();
            }
            ctx.app.editor.move_cursor_home();
        }
        KeyCode::End => {
            if let Some(recorder) = ctx.ui_trace.as_mut() {
                recorder.record_move_cursor_end();
            }
            ctx.app.editor.move_cursor_end();
        }
        KeyCode::Up => {
            if let Some(recorder) = ctx.ui_trace.as_mut() {
                recorder.record_history_up();
            }
            ctx.app.history_up();
        }
        KeyCode::Down => {
            if let Some(recorder) = ctx.ui_trace.as_mut() {
                recorder.record_history_down();
            }
            ctx.app.editor.history_down();
        }
        KeyCode::Char(c) => {
            if let Some(recorder) = ctx.ui_trace.as_mut() {
                recorder.record_input_insert_text(c.to_string());
            }
            ctx.app.editor.insert_text(&c.to_string());
            ctx.app.update_suggestions();
        }
        _ => {}
    }
    Ok(false)
}

/// Tab on a draft: queue it for the next turn while running, or send it
/// immediately (including slash commands) when idle.
async fn handle_tab_submit(ctx: &mut SessionCtx<'_>) -> anyhow::Result<bool> {
    let Some(queued) = ctx.app.try_take_prepared_turn() else {
        push_system_message(
            ctx.app,
            "Wait for pasted images to finish processing before sending or queueing this draft.",
        );
        return Ok(false);
    };
    ctx.app.update_suggestions();
    let parsed_command =
        parse_slash_command(&queued.display_text, &ctx.app.skills, ctx.ui_extensions);
    let is_host_slash_command = parsed_command.is_some();
    if queued.is_empty() || shell_escape_command(&queued.display_text).is_some() {
        ctx.app.restore_prepared_turn(queued);
        return Ok(false);
    }
    if ctx.app.turn_active() {
        if let Some(cmd) = parsed_command
            && slash_command_runs_out_of_band_while_running(&cmd)
        {
            let command_text = queued.display_text.clone();
            return run_slash_command(ctx, cmd, &command_text).await;
        }
        if is_host_slash_command {
            block_slash_command_while_working(ctx.app, &queued.display_text);
            ctx.app.restore_prepared_turn(queued);
            return Ok(false);
        }
        enqueue_prepared_turn_for_cli(
            queued,
            ctx.app,
            ctx.ui_trace,
            ctx.runtime,
            lash_core::DeliveryPolicy::AfterCurrentTurnCommit,
            lash_core::SlotPolicy::Exclusive,
            true,
        )
        .await;
        return Ok(false);
    }
    if ctx.runtime.is_none() {
        push_system_message(
            ctx.app,
            "Cannot send this input while the session is switching.".to_string(),
        );
        ctx.app.restore_prepared_turn(queued);
        return Ok(false);
    }

    if let Some(cmd) = parsed_command {
        let command_text = queued.display_text.clone();
        return run_slash_command(ctx, cmd, &command_text).await;
    }

    let turn_input = make_turn_input(&queued);
    send_user_message(
        queued.clone(),
        turn_input.clone(),
        ctx.app,
        ctx.ui_trace.as_mut(),
        ctx.logger,
        ctx.runtime,
        ctx.history,
        ctx.runtime_return_rx,
        ctx.cancel_token,
        ctx.active_stream_id,
        ctx.app_tx,
    )
    .await;
    *ctx.last_turn = Some(TurnReplayPayload {
        turn_input,
        prepared_turn: queued,
        execution_mode: ctx.current_execution_mode.clone(),
    });
    Ok(false)
}

/// Enter while the suggestion popup is open: run a highlighted slash
/// command, otherwise complete the suggestion like Tab.
async fn handle_suggestion_enter(ctx: &mut SessionCtx<'_>) -> anyhow::Result<bool> {
    if let Some((command_text, cmd)) = selected_slash_command_suggestion(ctx.app, ctx.ui_extensions)
    {
        if let Some(recorder) = ctx.ui_trace.as_mut() {
            recorder.record_suggestion_complete();
            recorder.record_slash_command(command_text.clone());
        }
        ctx.app.set_input(String::new());
        ctx.app.update_suggestions();
        if ctx.app.turn_active() {
            if slash_command_runs_out_of_band_while_running(&cmd) {
                return handle_parsed_slash_command(cmd, ctx.slash_ctx()).await;
            }
            block_slash_command_while_working(ctx.app, &command_text);
            ctx.app.set_input(command_text);
            ctx.app.update_suggestions();
            return Ok(false);
        }
        return handle_parsed_slash_command(cmd, ctx.slash_ctx()).await;
    }
    if let Some(recorder) = ctx.ui_trace.as_mut() {
        recorder.record_suggestion_complete();
    }
    ctx.app.complete_suggestion();
    ctx.app.update_suggestions();
    Ok(false)
}

/// Enter on a draft: inject into the active turn, queue, run a slash
/// command or shell escape, or send a fresh turn.
async fn handle_enter_submit(ctx: &mut SessionCtx<'_>) -> anyhow::Result<bool> {
    let Some(queued) = ctx.app.try_take_prepared_turn() else {
        push_system_message(
            ctx.app,
            "Wait for pasted images to finish processing before sending or queueing this draft.",
        );
        return Ok(false);
    };
    ctx.app.update_suggestions();
    if queued.is_empty() {
        return Ok(false);
    }

    let parsed_command =
        parse_slash_command(&queued.display_text, &ctx.app.skills, ctx.ui_extensions);
    let is_host_slash_command = parsed_command.is_some();

    if ctx.app.turn_active() {
        if let Some(cmd) = parsed_command
            && slash_command_runs_out_of_band_while_running(&cmd)
        {
            let command_text = queued.display_text.clone();
            return run_slash_command(ctx, cmd, &command_text).await;
        }
        if is_host_slash_command {
            block_slash_command_while_working(ctx.app, &queued.display_text);
            ctx.app.restore_prepared_turn(queued);
            return Ok(false);
        }
        if shell_escape_command(&queued.display_text).is_some() {
            push_system_message(
                ctx.app,
                "Shell escapes cannot be injected into the active turn. Wait for completion or use `Tab` to queue a later turn.",
            );
            ctx.app.restore_prepared_turn(queued);
            return Ok(false);
        }
        match ctx.app.route_turn_submission(ctx.runtime.is_some()) {
            TurnSubmissionRoute::QueueNextFullTurn => {
                enqueue_prepared_turn_for_cli(
                    queued,
                    ctx.app,
                    ctx.ui_trace,
                    ctx.runtime,
                    lash_core::DeliveryPolicy::AfterCurrentTurnCommit,
                    lash_core::SlotPolicy::Exclusive,
                    true,
                )
                .await;
                return Ok(false);
            }
            TurnSubmissionRoute::BlockedSessionSwitch => {
                push_system_message(
                    ctx.app,
                    "Cannot send this input while the session is switching.".to_string(),
                );
                ctx.app.restore_prepared_turn(queued);
                return Ok(false);
            }
            TurnSubmissionRoute::SendNow => {
                push_system_message(
                    ctx.app,
                    "Cannot start a new turn while the current turn is active.".to_string(),
                );
                ctx.app.restore_prepared_turn(queued);
                return Ok(false);
            }
            TurnSubmissionRoute::InjectActiveTurn => {}
        }
        let injection = lash_core::InjectedTurnInput {
            id: Some(queued.draft_id.clone()),
            message: make_injected_plugin_message(&queued).await,
        };
        let Some(session) = ctx.runtime.as_ref() else {
            push_system_message(
                ctx.app,
                "Current-turn injection is unavailable while the session is switching.",
            );
            ctx.app.restore_prepared_turn(queued);
            return Ok(false);
        };
        match session
            .control()
            .injection()
            .inject_turn_inputs(vec![injection])
            .await
        {
            Ok(()) => {
                record_queue_current_turn_input(ctx.ui_trace, &queued);
                ctx.app.cache_draft_presentation(queued.clone());
                if let Err(err) = refresh_queued_work_snapshot(ctx.app, ctx.runtime).await {
                    push_system_message(ctx.app, format!("Failed to refresh durable queue: {err}"));
                }
            }
            Err(err) => {
                push_system_message(
                    ctx.app,
                    format!("Failed to queue current-turn injection: {}", err),
                );
                ctx.app.restore_prepared_turn(queued);
            }
        }
        return Ok(false);
    }
    if ctx.runtime.is_none() {
        push_system_message(
            ctx.app,
            "Cannot send this input while the session is switching.".to_string(),
        );
        ctx.app.restore_prepared_turn(queued);
        return Ok(false);
    }

    // Shell escape: !command
    if let Some(cmd_str) = shell_escape_command(&queued.display_text) {
        run_shell_escape(ctx.app, &queued, cmd_str).await;
        return Ok(false);
    }

    // Try slash command
    if let Some(cmd) = parsed_command {
        let command_text = queued.display_text.clone();
        return run_slash_command(ctx, cmd, &command_text).await;
    }

    // Handle "quit"/"exit" without slash prefix
    if queued.display_text == "quit" || queued.display_text == "exit" {
        return Ok(true);
    }

    // Regular user message.
    match ctx.app.route_turn_submission(ctx.runtime.is_some()) {
        TurnSubmissionRoute::SendNow => {
            let turn_input = make_turn_input(&queued);
            send_user_message(
                queued.clone(),
                turn_input.clone(),
                ctx.app,
                ctx.ui_trace.as_mut(),
                ctx.logger,
                ctx.runtime,
                ctx.history,
                ctx.runtime_return_rx,
                ctx.cancel_token,
                ctx.active_stream_id,
                ctx.app_tx,
            )
            .await;
            *ctx.last_turn = Some(TurnReplayPayload {
                turn_input,
                prepared_turn: queued,
                execution_mode: ctx.current_execution_mode.clone(),
            });
        }
        TurnSubmissionRoute::QueueNextFullTurn => {
            enqueue_prepared_turn_for_cli(
                queued,
                ctx.app,
                ctx.ui_trace,
                ctx.runtime,
                lash_core::DeliveryPolicy::AfterCurrentTurnCommit,
                lash_core::SlotPolicy::Exclusive,
                true,
            )
            .await;
        }
        TurnSubmissionRoute::InjectActiveTurn => {
            push_system_message(
                ctx.app,
                "Cannot inject into a turn from the idle input path.".to_string(),
            );
            ctx.app.restore_prepared_turn(queued);
        }
        TurnSubmissionRoute::BlockedSessionSwitch => {
            push_system_message(
                ctx.app,
                "Cannot send this input while the session is switching.".to_string(),
            );
            ctx.app.restore_prepared_turn(queued);
        }
    }
    Ok(false)
}

/// Run a `!command` shell escape and append its output to the timeline.
async fn run_shell_escape(app: &mut App, queued: &PreparedTurn, cmd_str: &str) {
    if cmd_str.is_empty() {
        return;
    }
    app.push_prepared_user_input(queued);
    app.invalidate_height_cache();

    use tokio::process::Command as TokioCommand;
    let result = tokio::time::timeout(
        std::time::Duration::from_secs(30),
        TokioCommand::new("bash").arg("-c").arg(cmd_str).output(),
    )
    .await;

    match result {
        Ok(Ok(output)) => {
            let stdout = String::from_utf8_lossy(&output.stdout).to_string();
            let stderr = String::from_utf8_lossy(&output.stderr).to_string();
            let error = if !output.status.success() {
                let mut err = stderr.clone();
                if let Some(code) = output.status.code() {
                    if !err.is_empty() && !err.ends_with('\n') {
                        err.push('\n');
                    }
                    err.push_str(&format!("[exit code: {}]", code));
                }
                Some(err)
            } else if !stderr.is_empty() {
                Some(stderr)
            } else {
                None
            };
            app.timeline.push(UiTimelineItem::ShellOutput {
                command: cmd_str.to_string(),
                output: stdout.trim_end().to_string(),
                error: error.map(|e| e.trim_end().to_string()),
            });
        }
        Ok(Err(e)) => {
            app.timeline.push(UiTimelineItem::Error(format!(
                "Failed to run '{}': {}",
                cmd_str, e
            )));
        }
        Err(_) => {
            app.timeline.push(UiTimelineItem::Error(format!(
                "Command '{}' timed out after 30s. Try a narrower command or run it in smaller steps.",
                cmd_str
            )));
        }
    }
    app.invalidate_height_cache();
    app.scroll_to_bottom();
}
