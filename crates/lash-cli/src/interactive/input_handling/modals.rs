//! Key handlers for the focused modal overlays (document, skill picker,
//! session picker, tree, process overview, ask-prompt) plus the process
//! dock helpers shared with the input composer.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use lash::LashSession;
use lash_core::runtime::ExecutionScope;

use crate::app::{App, UiTimelineItem};
use crate::config::{LashConfig, ThemeName};
use crate::overlay::CommandPaletteAction;
use crate::render;
use crate::theme;
use crate::ui_effects::push_system_message;

use super::SessionCtx;
use super::mouse::viewport_metrics;

use crate::interactive::commands::switch_to_session_identifier;

pub(super) async fn handle_command_palette_key(
    key: KeyEvent,
    ctx: &mut SessionCtx<'_>,
) -> anyhow::Result<bool> {
    let viewport_height = ctx
        .terminal
        .size()
        .map(|(_, height)| height.saturating_sub(8).max(4) as usize)
        .unwrap_or(10);
    match key.code {
        KeyCode::Up | KeyCode::Char('k') => ctx.app.command_palette_up(),
        KeyCode::Down | KeyCode::Char('j') => ctx.app.command_palette_down(),
        KeyCode::PageUp => ctx.app.command_palette_page_up(viewport_height),
        KeyCode::PageDown => ctx.app.command_palette_page_down(viewport_height),
        KeyCode::Home => ctx.app.command_palette_home(),
        KeyCode::End => ctx.app.command_palette_end(),
        KeyCode::Backspace => ctx.app.command_palette_backspace_query(),
        KeyCode::Char(ch)
            if !key
                .modifiers
                .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT) =>
        {
            ctx.app.command_palette_insert_query_char(ch);
        }
        KeyCode::Enter => {
            let Some(action) = ctx.app.take_command_palette_action() else {
                return Ok(false);
            };
            return execute_command_palette_action(action, ctx).await;
        }
        _ => {}
    }
    Ok(false)
}

async fn execute_command_palette_action(
    action: CommandPaletteAction,
    ctx: &mut SessionCtx<'_>,
) -> anyhow::Result<bool> {
    match action {
        CommandPaletteAction::Builtin(command) => {
            let slash_ctx = ctx.slash_ctx();
            super::super::commands::handle_builtin_command(command, slash_ctx).await
        }
        CommandPaletteAction::InsertDraft(draft) => {
            ctx.app.set_input(draft);
            ctx.app.update_suggestions();
            Ok(false)
        }
        CommandPaletteAction::Theme(choice) => {
            apply_theme_choice(choice, ctx.app, ctx.provider);
            Ok(false)
        }
    }
}

fn apply_theme_choice(choice: ThemeName, app: &mut App, provider: &lash::provider::ProviderHandle) {
    theme::set_active_theme(choice);
    let mut config =
        LashConfig::load(&crate::paths::config_file()).unwrap_or_else(|| LashConfig::new(provider));
    config.upsert_provider(provider);
    config.theme = choice;
    match config.save(&crate::paths::config_file()) {
        Ok(()) => app.show_toast(
            format!("Theme set to {}", choice.label()),
            crate::app::ToastKind::Info,
        ),
        Err(err) => push_system_message(app, format!("Theme changed, but saving failed: {err}")),
    }
    app.dirty = true;
}

pub(super) fn can_focus_process_dock(app: &App) -> bool {
    app.input().trim().is_empty() && !app.has_suggestions() && !app.processes.is_empty()
}

pub(super) fn process_dock_has_focus(app: &App) -> bool {
    app.input().trim().is_empty() && app.selected_process().is_some()
}

pub(super) fn show_selected_process_overview(app: &mut App) {
    let Some(overview) = app.selected_process_overview_state() else {
        return;
    };
    app.show_process_overview(overview);
}

pub(super) async fn cancel_selected_process(app: &mut App, runtime: &Option<LashSession>) {
    let Some(process) = app.selected_process().cloned() else {
        return;
    };
    if process.status.is_terminal() {
        push_system_message(
            app,
            format!(
                "Process `{}` is already {}.",
                process.label,
                process.status.label()
            ),
        );
        return;
    }
    let Some(session) = runtime.as_ref() else {
        push_system_message(
            app,
            "Cannot cancel process while the session is switching.".to_string(),
        );
        return;
    };
    let processes = session.admin().processes();
    let effect_host = session.effect_host();
    let scoped_effect_controller =
        match effect_host.scoped(ExecutionScope::process(process.process_id.clone())) {
            Ok(controller) => controller,
            Err(err) => {
                push_system_message(app, format!("Failed to scope process cancellation: {err}"));
                return;
            }
        };
    match processes
        .cancel(&process.process_id, scoped_effect_controller)
        .await
    {
        Ok(summary) => {
            push_system_message(
                app,
                format!(
                    "Process `{}` cancellation requested: {}.",
                    process.label,
                    summary.status.label()
                ),
            );
            match processes.list().await {
                Ok(processes) => app.update_processes(
                    processes
                        .into_iter()
                        .map(crate::ui_effects::observed_to_handle_summary)
                        .collect(),
                ),
                Err(err) => {
                    push_system_message(app, format!("Failed to refresh process list: {err}"));
                }
            }
        }
        Err(err) => push_system_message(app, format!("Failed to cancel process: {err}")),
    }
}

pub(super) fn handle_document_key(key: KeyEvent, ctx: &mut SessionCtx<'_>) -> anyhow::Result<()> {
    let (width, height) = ctx.terminal.size()?;
    let body_height = height.saturating_sub(1).max(1);
    let viewport_height = body_height.saturating_sub(4).max(1) as usize;
    let inner_width = width.saturating_sub(8).max(1) as usize;
    match key.code {
        KeyCode::PageUp => ctx.app.document_scroll_up(viewport_height),
        KeyCode::PageDown => {
            let max_scroll = ctx
                .app
                .document_state()
                .map(|document| render::document_max_scroll(document, inner_width, viewport_height))
                .unwrap_or(0);
            ctx.app.document_scroll_down(viewport_height, max_scroll);
        }
        KeyCode::Home => ctx.app.document_scroll_up(usize::MAX),
        KeyCode::End => {
            let max_scroll = ctx
                .app
                .document_state()
                .map(|document| render::document_max_scroll(document, inner_width, viewport_height))
                .unwrap_or(0);
            ctx.app.document_scroll_down(usize::MAX, max_scroll);
        }
        _ => {}
    }
    Ok(())
}

/// Skill-picker overlay navigation: up/down move the highlight, Enter
/// inserts the chosen skill as a `$name ` draft.
pub(super) fn handle_skill_picker_key(key: KeyEvent, ctx: &mut SessionCtx<'_>) {
    let app = &mut *ctx.app;
    match key.code {
        KeyCode::Up | KeyCode::Char('k') => app.skill_picker_up(),
        KeyCode::Down | KeyCode::Char('j') => app.skill_picker_down(),
        KeyCode::Enter => {
            if let Some(name) = app.take_skill_pick() {
                app.set_input(format!("${} ", name));
            }
        }
        _ => {}
    }
}

/// Session-picker overlay navigation: up/down move the highlight, Enter
/// switches to the selected session.
pub(super) async fn handle_session_picker_key(key: KeyEvent, ctx: &mut SessionCtx<'_>) {
    match key.code {
        KeyCode::Up => ctx.app.session_picker_up(),
        KeyCode::Down => ctx.app.session_picker_down(),
        KeyCode::Backspace => ctx.app.session_picker_backspace_query(),
        KeyCode::Char(ch)
            if !key
                .modifiers
                .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT) =>
        {
            ctx.app.session_picker_insert_query_char(ch);
        }
        KeyCode::Enter => {
            let Some(filename) = ctx.app.take_session_pick() else {
                return;
            };
            if ctx.app.turn_active() {
                ctx.app.note_manual_interrupt_requested();
                if let Some(token) = ctx.cancel_token.take() {
                    token.cancel();
                }
                *ctx.runtime_return_rx = None;
                *ctx.active_stream_id = (*ctx.active_stream_id).wrapping_add(1);
            }
            match switch_to_session_identifier(
                &filename,
                ctx.app,
                ctx.logger,
                ctx.runtime_factory,
                ctx.runtime,
                ctx.history,
                ctx.turn_counter,
                ctx.provider,
                ctx.current_model_variant,
                ctx.current_execution_mode,
                ctx.active_tool_state,
                ctx.model_catalog,
                ctx.toolset_hash,
            )
            .await
            {
                Ok(()) => {
                    *ctx.last_turn = None;
                    ctx.app.dirty = true;
                }
                Err(err) => {
                    ctx.app
                        .timeline
                        .push(UiTimelineItem::SystemMessage(err.to_string()));
                    ctx.app.invalidate_height_cache();
                    ctx.app.scroll_to_bottom();
                }
            }
        }
        _ => {} // ignore other keys while picker is open
    }
}

/// Tree (branch) overlay navigation: up/down move the highlight,
/// Ctrl/Alt+Left/Right switch branches, Enter switches to the selected
/// node. Returns the `dispatch_key_event` result (always `Ok(false)` today;
/// the `Result` keeps the `?`/`.await` ergonomics of the switch call).
pub(super) async fn handle_tree_key(
    key: KeyEvent,
    ctx: &mut SessionCtx<'_>,
) -> anyhow::Result<bool> {
    match key.code {
        KeyCode::Up | KeyCode::Char('k') => ctx.app.tree_up(),
        KeyCode::Down | KeyCode::Char('j') => ctx.app.tree_down(),
        KeyCode::Left
            if key
                .modifiers
                .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT) =>
        {
            ctx.app.tree_prev_branch();
        }
        KeyCode::Right
            if key
                .modifiers
                .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT) =>
        {
            ctx.app.tree_next_branch();
        }
        KeyCode::Enter => {
            let Some(selection) = ctx.app.take_tree_pick() else {
                return Ok(false);
            };
            let Some(rt) = ctx.runtime.as_ref() else {
                push_system_message(
                    ctx.app,
                    "Branch navigation is unavailable while a turn is running.",
                );
                return Ok(false);
            };
            match crate::tree::switch_to_tree_selection(
                rt,
                ctx.logger,
                ctx.app,
                ctx.history,
                selection,
            )
            .await
            {
                Ok(()) => {
                    ctx.app.dirty = true;
                }
                Err(err) => {
                    push_system_message(ctx.app, format!("Branch switch failed: {err}"));
                }
            }
        }
        _ => {}
    }
    Ok(false)
}

/// Process-overview modal: Enter dismisses, Delete cancels the selected
/// process and dismisses.
pub(super) async fn handle_process_overview_key(key: KeyEvent, ctx: &mut SessionCtx<'_>) {
    match key.code {
        KeyCode::Enter => ctx.app.dismiss_process_overview(),
        KeyCode::Delete => {
            cancel_selected_process(ctx.app, ctx.runtime).await;
            ctx.app.dismiss_process_overview();
        }
        _ => {}
    }
}

/// Ask-dialog (prompt) key handling: navigation, multi-select toggles,
/// note-field focus, and text entry.
pub(super) fn handle_prompt_key(key: KeyEvent, ctx: &mut SessionCtx<'_>) {
    let app = &mut *ctx.app;
    let ui_trace = &mut *ctx.ui_trace;
    let editing_text = app.is_prompt_text_entry();
    let viewport_height = ctx
        .terminal
        .size()
        .map(|(width, height)| render::history_viewport_height(app, width, height))
        .unwrap_or(12)
        .max(1);
    match key.code {
        KeyCode::PageUp => {
            app.prompt_scroll_up(viewport_height);
        }
        KeyCode::PageDown => {
            let max_scroll = viewport_metrics(app, ctx.terminal).prompt_max_scroll;
            app.prompt_scroll_down(viewport_height, max_scroll);
        }
        KeyCode::Up if !editing_text && !app.prompt_has_options() => {
            app.prompt_scroll_up(1);
        }
        KeyCode::Down if !editing_text && !app.prompt_has_options() => {
            let max_scroll = viewport_metrics(app, ctx.terminal).prompt_max_scroll;
            app.prompt_scroll_down(1, max_scroll);
        }
        KeyCode::Tab if app.prompt_supports_note() => {
            if let Some(recorder) = ui_trace.as_mut() {
                recorder.record_prompt_toggle_note_focus();
            }
            app.prompt_toggle_note_focus();
        }
        KeyCode::Up if !editing_text => {
            if let Some(recorder) = ui_trace.as_mut() {
                recorder.record_prompt_up();
            }
            app.prompt_up();
        }
        KeyCode::Down if !editing_text => {
            if let Some(recorder) = ui_trace.as_mut() {
                recorder.record_prompt_down();
            }
            app.prompt_down();
        }
        KeyCode::Char(' ') if app.is_prompt_multi_select() && !editing_text => {
            if let Some(recorder) = ui_trace.as_mut() {
                recorder.record_prompt_toggle_current_option();
            }
            app.prompt_toggle_current_option();
        }
        KeyCode::BackTab if editing_text => {
            if let Some(recorder) = ui_trace.as_mut() {
                recorder.record_prompt_insert_text("\n");
            }
            app.prompt_insert_text("\n");
        }
        KeyCode::Enter => {
            if let Some(recorder) = ui_trace.as_mut() {
                recorder.record_submit_prompt();
            }
            let _ = app.take_prompt_response();
        }
        KeyCode::Char(c) if editing_text => {
            if let Some(recorder) = ui_trace.as_mut() {
                recorder.record_prompt_insert_text(c.to_string());
            }
            app.prompt_insert_text(&c.to_string());
        }
        KeyCode::Backspace if editing_text => {
            if let Some(recorder) = ui_trace.as_mut() {
                recorder.record_prompt_backspace();
            }
            app.prompt_backspace();
        }
        _ => {}
    }
}
