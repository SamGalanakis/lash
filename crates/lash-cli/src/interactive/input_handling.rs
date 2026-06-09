use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use crossterm::event::{
    Event as TermEvent, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseEvent,
};
use lash::{LashSession, ModeId, provider::ProviderHandle};
use lash_core::session_model::Message;
use lash_core::{ToolState, runtime::EffectScope};
use lash_tui::{InputEvent as TuiInputEvent, Terminal, normalize_event};
use lash_tui_extensions::{TuiExtensionContext, TuiExtensions, TuiInputOutcome, TuiSurfaceSlot};
use tokio::task;
use tokio_util::sync::CancellationToken;

use crate::app::{App, PreparedTurn, TurnSubmissionRoute, UiTimelineItem};
use crate::editor::SuggestionKind;
use crate::event::AppEvent;
use crate::input_items::insert_inline_marker;
use crate::model_catalog::CachedModelCatalog;
use crate::render;
use crate::session_log::SessionLogger;
use crate::turn_runner::{RuntimeRunResult, make_turn_input};
use crate::ui_trace::UiTraceRecorder;
use crate::{
    Args, apply_ui_host_effects, normalize_prepared_turn_for_dispatch, push_system_message,
    shell_escape_command,
};

use super::commands::{
    SlashCommandCtx, handle_parsed_slash_command, parse_slash_command,
    slash_command_runs_out_of_band_while_running, switch_to_session_identifier,
};
use super::helpers::{
    TurnReplayPayload, is_copy_shortcut, key_chord_from_event, queued_turn_edit_matches,
    record_queue_current_turn_input, record_queue_turn, should_preserve_selection_for_key,
};
use super::runtime::{
    copy_selected_text_or_last_response, enqueue_prepared_turn, make_injected_plugin_message,
    refresh_queued_work_snapshot, send_user_message,
};

pub(super) fn slash_command_blocked_while_working_message(command_text: &str) -> String {
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

pub(super) fn handle_surface_input(
    ui_extensions: &TuiExtensions,
    event: &TuiInputEvent,
    session: Option<&LashSession>,
    app: &mut App,
) -> bool {
    let Some(session) = session else {
        return false;
    };
    match ui_extensions.handle_input(
        event,
        TuiExtensionContext {
            actions: &session.plugin_actions(),
        },
    ) {
        TuiInputOutcome::Ignored => false,
        TuiInputOutcome::Handled(effects) => {
            apply_ui_host_effects(app, effects);
            app.dirty = true;
            true
        }
    }
}

async fn restore_last_durable_full_turn(app: &mut App, runtime: &Option<LashSession>) {
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

fn can_focus_process_dock(app: &App) -> bool {
    app.input().trim().is_empty() && !app.has_suggestions() && !app.processes.is_empty()
}

fn process_dock_has_focus(app: &App) -> bool {
    app.input().trim().is_empty() && app.selected_process().is_some()
}

fn show_selected_process_overview(app: &mut App) {
    let Some(overview) = app.selected_process_overview_state() else {
        return;
    };
    app.show_process_overview(overview);
}

async fn cancel_selected_process(app: &mut App, runtime: &Option<LashSession>) {
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
    let process_control = session.control().processes();
    let effect_host = session.effect_host().await;
    let scoped_effect_controller =
        match effect_host.scoped(EffectScope::process(process.process_id.clone())) {
            Ok(controller) => controller,
            Err(err) => {
                push_system_message(app, format!("Failed to scope process cancellation: {err}"));
                return;
            }
        };
    match process_control
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
            match process_control.list().await {
                Ok(processes) => app.update_processes(processes),
                Err(err) => {
                    push_system_message(app, format!("Failed to refresh process list: {err}"));
                }
            }
        }
        Err(err) => push_system_message(app, format!("Failed to cancel process: {err}")),
    }
}

/// Viewport-derived measurements the scroll handlers need; computed from
/// the current terminal size against the live layout.
struct ViewportMetrics {
    width: usize,
    history_height: usize,
    prompt_max_scroll: usize,
}

fn viewport_metrics(app: &App, terminal: &Terminal) -> ViewportMetrics {
    let (width, height) = terminal.size().unwrap_or((80, 24));
    let history_height = render::history_viewport_height(app, width, height);
    let prompt_max_scroll = if let Some(prompt) = app.prompt_state() {
        let prompt_area = render::input_area(app, width, height);
        let visible_height = prompt_area.height as usize;
        let inner_width = prompt_area.width.saturating_sub(2) as usize;
        render::prompt_max_scroll(prompt, inner_width.max(1), visible_height)
    } else {
        0
    };
    ViewportMetrics {
        width: width as usize,
        history_height,
        prompt_max_scroll,
    }
}

/// Scroll the history viewport down by `amount`, clamped against the
/// current layout.
fn scroll_history_down(app: &mut App, terminal: &Terminal, amount: usize) {
    let metrics = viewport_metrics(app, terminal);
    app.scroll_down(amount, metrics.history_height, metrics.width);
}

pub(super) fn selected_slash_command_suggestion(
    app: &App,
    ui_extensions: &TuiExtensions,
) -> Option<(String, super::commands::ParsedSlashCommand)> {
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

/// A scroll keypress, resolved to a direction and amount. PageUp/PageDown
/// scroll the focused document/prompt overlay first, then history when no
/// modal owns focus. Bare Up/Down scroll a review-only prompt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ScrollInputAction {
    PromptUp(usize),
    PromptDown(usize),
    HistoryUp(usize),
    HistoryDown(usize),
}

fn classify_always_on_scroll_key(
    key: KeyEvent,
    app: &App,
    viewport_height: usize,
) -> Option<ScrollInputAction> {
    if app.has_prompt() {
        if key.code == KeyCode::PageUp {
            return Some(ScrollInputAction::PromptUp(viewport_height));
        }
        if key.code == KeyCode::PageDown {
            return Some(ScrollInputAction::PromptDown(viewport_height));
        }
        if !app.is_prompt_text_entry() && !app.prompt_has_options() {
            if key.code == KeyCode::Up {
                return Some(ScrollInputAction::PromptUp(1));
            }
            if key.code == KeyCode::Down {
                return Some(ScrollInputAction::PromptDown(1));
            }
        }
    }

    if key.code == KeyCode::PageUp {
        return Some(ScrollInputAction::HistoryUp(viewport_height));
    }
    if key.code == KeyCode::PageDown {
        return Some(ScrollInputAction::HistoryDown(viewport_height));
    }

    None
}

fn apply_scroll_input_action(
    app: &mut App,
    terminal: &Terminal,
    ui_trace: &mut Option<UiTraceRecorder>,
    action: ScrollInputAction,
) {
    match action {
        ScrollInputAction::PromptUp(amount) => app.prompt_scroll_up(amount),
        ScrollInputAction::PromptDown(amount) => {
            let max_scroll = viewport_metrics(app, terminal).prompt_max_scroll;
            app.prompt_scroll_down(amount, max_scroll);
        }
        ScrollInputAction::HistoryUp(amount) => {
            if let Some(recorder) = ui_trace.as_mut() {
                recorder.record_scroll_up(amount);
            }
            app.scroll_up(amount);
        }
        ScrollInputAction::HistoryDown(amount) => {
            if let Some(recorder) = ui_trace.as_mut() {
                recorder.record_scroll_down(amount);
            }
            scroll_history_down(app, terminal, amount);
        }
    }
    app.dirty = true;
}

pub(super) fn handle_mouse_event(
    mouse: MouseEvent,
    app: &mut App,
    terminal: &Terminal,
    ui_trace: &mut Option<UiTraceRecorder>,
    ui_extensions: &TuiExtensions,
    runtime: &Option<LashSession>,
) -> anyhow::Result<()> {
    use crossterm::event::{MouseButton, MouseEventKind};
    // Some terminals (notably kitty) can paint transient hover/search
    // decorations on top of the alt-screen. Repaint on any mouse event
    // so ignored motion events do not leave visual artifacts behind.
    app.dirty = true;
    let ha = app.history_area;
    let (term_width, term_height) = terminal.size()?;
    let prompt_area = render::input_area(app, term_width, term_height);
    let input_area = render::input_content_area(app, term_width, term_height);
    let workspace_active = app
        .ui_extensions()
        .has_surface_in_slot(TuiSurfaceSlot::Workspace);
    let in_history = !workspace_active
        && mouse.row >= ha.y
        && mouse.row < ha.y + ha.height
        && mouse.column >= ha.x
        && mouse.column < ha.x + ha.width;
    let in_input = input_area.width > 0
        && input_area.height > 0
        && mouse.row >= input_area.y
        && mouse.row < input_area.y + input_area.height
        && mouse.column >= input_area.x
        && mouse.column < input_area.x + input_area.width;

    if app.has_prompt() {
        let prompt_review_height = app.prompt_state().map_or(0, |prompt| {
            let inner_width = prompt_area.width.saturating_sub(2) as usize;
            render::prompt_render_snapshot(prompt, inner_width.max(1), prompt_area.height as usize)
                .review_viewport_height
        });
        let in_prompt_review = prompt_area.width > 0
            && prompt_area.height > 0
            && mouse.row >= prompt_area.y
            && mouse.row < prompt_area.y + prompt_review_height as u16
            && mouse.column >= prompt_area.x
            && mouse.column < prompt_area.x + prompt_area.width;
        match mouse.kind {
            MouseEventKind::ScrollUp if !app.prompt_uses_split_layout() || in_prompt_review => {
                app.prompt_scroll_up(3);
                app.dirty = true;
                return Ok(());
            }
            MouseEventKind::ScrollDown if !app.prompt_uses_split_layout() || in_prompt_review => {
                let max_scroll = viewport_metrics(app, terminal).prompt_max_scroll;
                app.prompt_scroll_down(3, max_scroll);
                app.dirty = true;
                return Ok(());
            }
            _ => {}
        }
    }

    if let Some(event) = normalize_event(&TermEvent::Mouse(mouse))
        && handle_surface_input(ui_extensions, &event, runtime.as_ref(), app)
    {
        return Ok(());
    }

    match mouse.kind {
        MouseEventKind::Down(MouseButton::Left) if in_history => {
            app.clear_input_selection();
            tracing::debug!(
                row = mouse.row,
                column = mouse.column,
                "selection started from mouse down"
            );
            let vrow = app.scroll_offset + (mouse.row - ha.y) as usize;
            app.selection.anchor = (mouse.column, vrow);
            app.selection.end = (mouse.column, vrow);
            app.selection.active = true;
            app.selection.visible = false;
            app.dirty = true;
        }
        MouseEventKind::Down(MouseButton::Left) if in_input => {
            app.clear_selection();
            if let Some(offset) = render::input_byte_offset_for_screen_position(
                app,
                term_width,
                term_height,
                mouse.column,
                mouse.row,
            ) {
                tracing::debug!(
                    row = mouse.row,
                    column = mouse.column,
                    offset,
                    "input selection started from mouse down"
                );
                app.start_input_selection(offset);
                app.dirty = true;
            }
        }
        MouseEventKind::Drag(MouseButton::Left) if app.selection.active => {
            let col = mouse.column.clamp(ha.x, ha.x + ha.width.saturating_sub(1));

            // Auto-scroll when dragging above or below the history area
            let scroll_lines = 3usize;
            if mouse.row < ha.y {
                if let Some(recorder) = ui_trace.as_mut() {
                    recorder.record_scroll_up(scroll_lines);
                }
                app.scroll_up(scroll_lines);
            } else if mouse.row >= ha.y + ha.height {
                let (width, height) = terminal.size()?;
                let vh = render::history_viewport_height(app, width, height);
                let vw = width as usize;
                if let Some(recorder) = ui_trace.as_mut() {
                    recorder.record_scroll_down(scroll_lines);
                }
                app.scroll_down(scroll_lines, vh, vw);
            }

            let clamped_row = mouse.row.clamp(ha.y, ha.y + ha.height.saturating_sub(1));
            let vrow = app.scroll_offset + (clamped_row - ha.y) as usize;
            app.selection.end = (col, vrow);
            app.selection.visible = true;
            tracing::debug!(
                row = mouse.row,
                column = mouse.column,
                selection_anchor = ?app.selection.anchor,
                selection_end = ?app.selection.end,
                "selection updated from mouse drag"
            );
            app.dirty = true;
        }
        MouseEventKind::Drag(MouseButton::Left) if app.input_selection_active() => {
            let col = mouse.column.clamp(
                input_area.x,
                input_area.x + input_area.width.saturating_sub(1),
            );
            let row = mouse.row.clamp(
                input_area.y,
                input_area.y + input_area.height.saturating_sub(1),
            );
            if let Some(offset) = render::input_byte_offset_for_screen_position(
                app,
                term_width,
                term_height,
                col,
                row,
            ) {
                app.update_input_selection(offset);
                tracing::debug!(
                    row = mouse.row,
                    column = mouse.column,
                    offset,
                    "input selection updated from mouse drag"
                );
                app.dirty = true;
            }
        }
        MouseEventKind::Up(MouseButton::Left) if app.selection.active => {
            tracing::debug!(
                selection_anchor = ?app.selection.anchor,
                selection_end = ?app.selection.end,
                selection_visible = app.selection.visible,
                "selection finished on mouse up"
            );
            app.selection.active = false;
        }
        MouseEventKind::Up(MouseButton::Left) if app.input_selection_active() => {
            tracing::debug!(
                input_selection_range = ?app.input_selection_range(),
                "input selection finished on mouse up"
            );
            app.finish_input_selection();
        }
        MouseEventKind::ScrollUp => {
            // Scroll extends selection if actively dragging, otherwise just scrolls
            let scroll_lines = 3;
            if let Some(recorder) = ui_trace.as_mut() {
                recorder.record_scroll_up(scroll_lines);
            }
            app.scroll_up(scroll_lines);
            if app.selection.active {
                // Extend selection to top of viewport after scroll
                let vrow = app.scroll_offset;
                app.selection.end = (ha.x, vrow);
                app.selection.visible = true;
            }
            app.dirty = true;
        }
        MouseEventKind::ScrollDown => {
            let (width, height) = terminal.size()?;
            let vh = render::history_viewport_height(app, width, height);
            let vw = width as usize;
            let scroll_lines = 3;
            if let Some(recorder) = ui_trace.as_mut() {
                recorder.record_scroll_down(scroll_lines);
            }
            app.scroll_down(scroll_lines, vh, vw);
            if app.selection.active {
                // Extend selection to bottom of viewport after scroll
                let vrow = app.scroll_offset + vh.saturating_sub(1);
                app.selection.end = (ha.x + ha.width, vrow);
                app.selection.visible = true;
            }
            app.dirty = true;
        }
        _ => {}
    }
    Ok(())
}

/// Bundle of the long-lived interactive-loop state every key handler
/// needs. The run loop borrows its locals into this once per key event
/// and hands `&mut SessionCtx` to [`dispatch_key_event`]; the per-mode
/// handlers reach through `ctx.field` directly rather than re-exploding
/// the bag into locals.
pub(super) struct SessionCtx<'a> {
    pub app: &'a mut App,
    pub terminal: &'a mut Terminal,
    pub ui_trace: &'a mut Option<UiTraceRecorder>,
    pub logger: &'a mut SessionLogger,
    pub args: &'a Args,
    pub paused: &'a Arc<AtomicBool>,
    pub ui_extensions: &'a TuiExtensions,
    pub runtime_factory: &'a crate::session_bootstrap::CliSessionOpener,
    pub lash_config: &'a crate::config::LashConfig,
    pub runtime: &'a mut Option<LashSession>,
    pub history: &'a mut Vec<Message>,
    pub turn_counter: &'a mut usize,
    pub last_turn: &'a mut Option<TurnReplayPayload>,
    pub runtime_return_rx: &'a mut Option<tokio::sync::oneshot::Receiver<RuntimeRunResult>>,
    pub cancel_token: &'a mut Option<CancellationToken>,
    pub active_stream_id: &'a mut u64,
    pub provider: &'a mut ProviderHandle,
    pub current_model_variant: &'a mut Option<String>,
    pub current_execution_mode: &'a mut ModeId,
    pub active_tool_state: &'a mut ToolState,
    pub model_catalog: &'a CachedModelCatalog,
    pub toolset_hash: &'a mut String,
    pub app_tx: &'a crate::event::AppEventTx,
    pub pending_clear_after_return: &'a mut bool,
}

/// The modal overlay that currently owns keyboard focus, derived once from
/// `app.overlay` so modal precedence lives in data instead of a duplicated
/// `if app.has_X()` ladder. Empty pickers/trees collapse to `None` so they
/// fall through to the input composer, matching the old `has_*` guards.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ActiveModal {
    None,
    Document,
    SkillPicker,
    SessionPicker,
    Tree,
    ProcessOverview,
    Prompt,
}

fn active_modal(app: &App) -> ActiveModal {
    if app.has_document() {
        ActiveModal::Document
    } else if app.has_skill_picker() {
        ActiveModal::SkillPicker
    } else if app.has_session_picker() {
        ActiveModal::SessionPicker
    } else if app.has_tree() {
        ActiveModal::Tree
    } else if app.has_process_overview() {
        ActiveModal::ProcessOverview
    } else if app.has_prompt() {
        ActiveModal::Prompt
    } else {
        ActiveModal::None
    }
}

/// Route a key press to the handler for the active mode. Order matters and
/// mirrors the historical fall-through: global shortcuts win first, then
/// focused modal overlays, then history scrolling, surface/shortcut routing,
/// and finally the default input dispatch.
pub(super) async fn dispatch_key_event(
    key: KeyEvent,
    ctx: &mut SessionCtx<'_>,
) -> anyhow::Result<bool> {
    // With kitty keyboard protocol, ignore Release/Repeat events
    if key.kind != KeyEventKind::Press {
        return Ok(false);
    }
    ctx.app.dirty = true;

    if let Some(result) = handle_global_shortcut_key(key, ctx).await? {
        return Ok(result);
    }

    match active_modal(ctx.app) {
        ActiveModal::Document => {
            handle_document_key(key, ctx)?;
            return Ok(false);
        }
        ActiveModal::SkillPicker => {
            handle_skill_picker_key(key, ctx);
            return Ok(false);
        }
        ActiveModal::SessionPicker => {
            handle_session_picker_key(key, ctx).await;
            return Ok(false);
        }
        ActiveModal::Tree => return handle_tree_key(key, ctx).await,
        ActiveModal::ProcessOverview => {
            match key.code {
                KeyCode::Enter => ctx.app.dismiss_process_overview(),
                KeyCode::Delete => {
                    cancel_selected_process(ctx.app, ctx.runtime).await;
                    ctx.app.dismiss_process_overview();
                }
                _ => {}
            }
            return Ok(false);
        }
        ActiveModal::Prompt => {
            handle_prompt_key(key, ctx);
            return Ok(false);
        }
        ActiveModal::None => {}
    }

    let (width, height) = ctx.terminal.size()?;
    let vh = render::history_viewport_height(ctx.app, width, height);
    if let Some(action) = classify_always_on_scroll_key(key, ctx.app, vh) {
        apply_scroll_input_action(ctx.app, ctx.terminal, ctx.ui_trace, action);
        return Ok(false);
    }

    if let Some(event) = normalize_event(&TermEvent::Key(key))
        && handle_surface_input(ctx.ui_extensions, &event, ctx.runtime.as_ref(), ctx.app)
    {
        return Ok(false);
    }

    if let Some(chord) = key_chord_from_event(key)
        && let Some(shortcut) = ctx.ui_extensions.shortcut_for(chord)
    {
        let Some(session) = ctx.runtime.as_ref() else {
            push_system_message(ctx.app, "No active session for UI shortcut.".to_string());
            return Ok(false);
        };
        match ctx
            .ui_extensions
            .invoke_shortcut(
                &shortcut,
                TuiExtensionContext {
                    actions: &session.plugin_actions(),
                },
            )
            .await
        {
            Ok(effects) => apply_ui_host_effects(ctx.app, effects),
            Err(err) => push_system_message(ctx.app, err),
        }
        return Ok(false);
    }

    handle_input_mode_key(key, ctx).await
}

/// Global shortcuts that fire regardless of the active mode (selection
/// copy, Ctrl+C dismiss/quit, expand toggles, undo/redo, paste, Esc).
/// Returns `Some(ret)` when the key was fully handled here, where `ret`
/// is the `dispatch_key_event` return value; `None` to fall through.
async fn handle_global_shortcut_key(
    key: KeyEvent,
    ctx: &mut SessionCtx<'_>,
) -> anyhow::Result<Option<bool>> {
    let app = &mut *ctx.app;
    let terminal = &*ctx.terminal;
    let ui_trace = &mut *ctx.ui_trace;
    let app_tx = ctx.app_tx;
    let cancel_token = &mut *ctx.cancel_token;
    let copy_shortcut = is_copy_shortcut(key);
    tracing::debug!(
        code = ?key.code,
        modifiers = ?key.modifiers,
        kind = ?key.kind,
        state = ?key.state,
        selection_visible = app.selection.visible,
        selection_active = app.selection.active,
        input_selection_visible = app.has_input_selection(),
        input_selection_active = app.input_selection_active(),
        copy_shortcut,
        preserve_selection = should_preserve_selection_for_key(key),
        "received key event"
    );
    // Clear any active history selection on plain keypress.
    if app.selection.visible && !should_preserve_selection_for_key(key) {
        tracing::debug!("clearing selection on plain keypress");
        app.clear_selection();
    }

    // Active selection copy should win before generic Ctrl+C handling.
    if (app.selection.visible || app.has_input_selection()) && copy_shortcut {
        tracing::debug!("selection copy took precedence over generic key handling");
        copy_selected_text_or_last_response(app, terminal.size().ok());
        return Ok(Some(false));
    }

    // CTRL+C: close transient UI, cancel active work, clear draft, then
    // quit only from an idle empty state.
    if !key.modifiers.contains(KeyModifiers::SHIFT)
        && key.modifiers.contains(KeyModifiers::CONTROL)
        && matches!(key.code, KeyCode::Char(c) if c.eq_ignore_ascii_case(&'c'))
    {
        tracing::debug!("handling Ctrl+C as cancel/dismiss/quit");
        if app.has_suggestions() {
            app.dismiss_suggestions();
            return Ok(Some(false));
        }
        if dismiss_active_modal(app, ui_trace) {
            return Ok(Some(false));
        }
        if app.turn_active() {
            app.note_manual_interrupt_requested();
            if let Some(token) = cancel_token.take() {
                token.cancel();
            }
            return Ok(Some(false));
        }
        if !app.input().is_empty() || app.has_pending_input_payload() {
            app.clear_draft();
            return Ok(Some(false));
        }
        return Ok(Some(true));
    }

    // ALT+O: reliable full expand toggle across most terminals.
    if key.modifiers.contains(KeyModifiers::ALT)
        && matches!(key.code, KeyCode::Char(c) if c.eq_ignore_ascii_case(&'o'))
    {
        app.toggle_full_expand();
        return Ok(Some(false));
    }

    if queued_turn_edit_matches(key) {
        restore_last_durable_full_turn(app, ctx.runtime).await;
        return Ok(Some(false));
    }

    // CTRL+O: cycle expand (0↔1)
    if key.modifiers.contains(KeyModifiers::CONTROL)
        && matches!(key.code, KeyCode::Char(c) if c.eq_ignore_ascii_case(&'o'))
    {
        app.cycle_expand();
        return Ok(Some(false));
    }

    // CTRL+SHIFT+Z: redo the most recently undone edit.
    if key.modifiers.contains(KeyModifiers::CONTROL)
        && key.modifiers.contains(KeyModifiers::SHIFT)
        && matches!(key.code, KeyCode::Char(c) if c.eq_ignore_ascii_case(&'z'))
    {
        if app.editor_redo() {
            app.update_suggestions();
        }
        return Ok(Some(false));
    }

    // CTRL+Z: undo the most recent edit to the input draft.
    if key.modifiers.contains(KeyModifiers::CONTROL)
        && !key.modifiers.contains(KeyModifiers::SHIFT)
        && matches!(key.code, KeyCode::Char(c) if c.eq_ignore_ascii_case(&'z'))
    {
        if app.editor_undo() {
            app.update_suggestions();
        }
        return Ok(Some(false));
    }

    // ALT+Z: redo fallback for terminals that swallow CTRL+SHIFT+Z.
    if key.modifiers.contains(KeyModifiers::ALT)
        && matches!(key.code, KeyCode::Char(c) if c.eq_ignore_ascii_case(&'z'))
    {
        if app.editor_redo() {
            app.update_suggestions();
        }
        return Ok(Some(false));
    }

    // CTRL+Y / CTRL+SHIFT+C: copy current selection when present,
    // otherwise the last assistant response.
    if copy_shortcut {
        tracing::debug!("copy shortcut matched without active selection precedence");
        copy_selected_text_or_last_response(app, terminal.size().ok());
        return Ok(Some(false));
    }

    // CTRL+SHIFT+V: always paste text from clipboard
    if key.modifiers.contains(KeyModifiers::CONTROL)
        && key.modifiers.contains(KeyModifiers::SHIFT)
        && key.code == KeyCode::Char('V')
    {
        if let Ok(mut clipboard) = arboard::Clipboard::new()
            && let Ok(text) = clipboard.get_text()
        {
            app.insert_pasted_text(&text);
            app.update_suggestions();
        }
        return Ok(Some(false));
    }

    // CTRL+V: paste image from clipboard (no text fallback)
    if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('v') {
        if let Ok(mut clipboard) = arboard::Clipboard::new()
            && let Ok(img_data) = clipboard.get_image()
        {
            let image_id = app.next_image_marker_id();
            let marker = format!("[Image #{}]", image_id);
            insert_inline_marker(app, &marker);
            app.begin_pending_image(image_id);
            app.update_suggestions();
            let app_tx = app_tx.clone();
            let w = img_data.width as u32;
            let h = img_data.height as u32;
            let bytes = img_data.bytes.into_owned();
            tokio::spawn(async move {
                let png = task::spawn_blocking(move || {
                    let rgba = image::RgbaImage::from_raw(w, h, bytes)
                        .ok_or_else(|| anyhow::anyhow!("Failed to decode pasted image data."))?;
                    let mut png_buf = std::io::Cursor::new(Vec::new());
                    rgba.write_to(&mut png_buf, image::ImageFormat::Png)
                        .map_err(|err| anyhow::anyhow!("Failed to encode pasted image: {err}"))?;
                    Ok::<_, anyhow::Error>(png_buf.into_inner())
                })
                .await
                .unwrap_or_else(|err| {
                    Err(anyhow::anyhow!("Failed to process pasted image: {err}"))
                });
                let _ = app_tx.send(AppEvent::ClipboardImageReady { id: image_id, png });
            });
        }
        return Ok(Some(false));
    }

    // Escape dismisses the active modal (precedence resolved once via
    // `active_modal`), otherwise interrupts a running turn.
    if key.code == KeyCode::Esc {
        match active_modal(app) {
            ActiveModal::Document => app.dismiss_document(),
            ActiveModal::Prompt => {
                if let Some(recorder) = ui_trace.as_mut() {
                    recorder.record_prompt_dismiss();
                }
                app.dismiss_prompt();
            }
            ActiveModal::Tree => app.dismiss_tree(),
            ActiveModal::SkillPicker => app.dismiss_skill_picker(),
            ActiveModal::SessionPicker => app.dismiss_session_picker(),
            ActiveModal::ProcessOverview => app.dismiss_process_overview(),
            ActiveModal::None => {
                if app.selected_process().is_some() {
                    app.clear_process_selection();
                } else if app.turn_active() {
                    // Interrupt running session
                    app.note_manual_interrupt_requested();
                    if let Some(token) = cancel_token.take() {
                        token.cancel();
                    }
                }
                // When idle with no dialog: no-op
            }
        }
        return Ok(Some(false));
    }

    Ok(None)
}

fn dismiss_active_modal(app: &mut App, ui_trace: &mut Option<UiTraceRecorder>) -> bool {
    match active_modal(app) {
        ActiveModal::Document => app.dismiss_document(),
        ActiveModal::Prompt => {
            if let Some(recorder) = ui_trace.as_mut() {
                recorder.record_prompt_dismiss();
            }
            app.dismiss_prompt();
        }
        ActiveModal::Tree => app.dismiss_tree(),
        ActiveModal::SkillPicker => app.dismiss_skill_picker(),
        ActiveModal::SessionPicker => app.dismiss_session_picker(),
        ActiveModal::ProcessOverview => app.dismiss_process_overview(),
        ActiveModal::None => return false,
    }
    true
}

fn handle_document_key(key: KeyEvent, ctx: &mut SessionCtx<'_>) -> anyhow::Result<()> {
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
fn handle_skill_picker_key(key: KeyEvent, ctx: &mut SessionCtx<'_>) {
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
async fn handle_session_picker_key(key: KeyEvent, ctx: &mut SessionCtx<'_>) {
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
async fn handle_tree_key(key: KeyEvent, ctx: &mut SessionCtx<'_>) -> anyhow::Result<bool> {
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

/// Ask-dialog (prompt) key handling: navigation, multi-select toggles,
/// note-field focus, and text entry.
fn handle_prompt_key(key: KeyEvent, ctx: &mut SessionCtx<'_>) {
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

/// Default input dispatch when no modal overlay is active: suggestion
/// navigation/completion, Tab/Enter send-or-queue, shell escapes, slash
/// commands, and raw editing keys.
async fn handle_input_mode_key(key: KeyEvent, ctx: &mut SessionCtx<'_>) -> anyhow::Result<bool> {
    let app = &mut *ctx.app;
    let terminal = &mut *ctx.terminal;
    let ui_trace = &mut *ctx.ui_trace;
    let logger = &mut *ctx.logger;
    let args = ctx.args;
    let paused = ctx.paused;
    let ui_extensions = ctx.ui_extensions;
    let runtime_factory = ctx.runtime_factory;
    let lash_config = ctx.lash_config;
    let runtime = &mut *ctx.runtime;
    let history = &mut *ctx.history;
    let turn_counter = &mut *ctx.turn_counter;
    let last_turn = &mut *ctx.last_turn;
    let runtime_return_rx = &mut *ctx.runtime_return_rx;
    let cancel_token = &mut *ctx.cancel_token;
    let active_stream_id = &mut *ctx.active_stream_id;
    let provider = &mut *ctx.provider;
    let current_model_variant = &mut *ctx.current_model_variant;
    let current_execution_mode = &mut *ctx.current_execution_mode;
    let active_tool_state = &mut *ctx.active_tool_state;
    let model_catalog = ctx.model_catalog;
    let toolset_hash = &mut *ctx.toolset_hash;
    let app_tx = ctx.app_tx;
    let pending_clear_after_return = &mut *ctx.pending_clear_after_return;

    match key.code {
        // Tab: complete selected suggestion
        KeyCode::Tab if app.has_suggestions() => {
            if let Some(recorder) = ui_trace.as_mut() {
                recorder.record_suggestion_complete();
            }
            app.complete_suggestion();
            app.update_suggestions();
        }
        KeyCode::Tab if can_focus_process_dock(app) => {
            app.select_next_process();
        }
        KeyCode::BackTab if can_focus_process_dock(app) => {
            app.select_previous_process();
        }
        KeyCode::Tab => {
            let Some(queued) = app.try_take_prepared_turn() else {
                push_system_message(
                    app,
                    "Wait for pasted images to finish processing before sending or queueing this draft.",
                );
                return Ok(false);
            };
            let queued = normalize_prepared_turn_for_dispatch(queued, &app.skills);
            app.update_suggestions();
            let parsed_command =
                parse_slash_command(&queued.display_text, &app.skills, ui_extensions);
            let is_host_slash_command = parsed_command.is_some();
            if queued.is_empty() || shell_escape_command(&queued.display_text).is_some() {
                app.restore_prepared_turn(queued);
                return Ok(false);
            }
            if app.turn_active() {
                if let Some(cmd) = parsed_command
                    && slash_command_runs_out_of_band_while_running(&cmd)
                {
                    if let Some(recorder) = ui_trace.as_mut() {
                        recorder.record_slash_command(queued.display_text.clone());
                    }
                    if handle_parsed_slash_command(
                        cmd,
                        SlashCommandCtx {
                            terminal: &mut *terminal,
                            app: &mut *app,
                            logger: &mut *logger,
                            args,
                            paused,
                            ui_extensions,
                            runtime_factory,
                            lash_config,
                            runtime: &mut *runtime,
                            history: &mut *history,
                            turn_counter: &mut *turn_counter,
                            last_turn: &mut *last_turn,
                            runtime_return_rx: &mut *runtime_return_rx,
                            cancel_token: &mut *cancel_token,
                            active_stream_id: &mut *active_stream_id,
                            provider: &mut *provider,
                            current_model_variant: &mut *current_model_variant,
                            current_execution_mode: &mut *current_execution_mode,
                            active_tool_state: &mut *active_tool_state,
                            model_catalog,
                            toolset_hash: &mut *toolset_hash,
                            app_tx,
                            pending_clear_after_return: &mut *pending_clear_after_return,
                        },
                    )
                    .await?
                    {
                        return Ok(true);
                    }
                    return Ok(false);
                }
                if is_host_slash_command {
                    block_slash_command_while_working(app, &queued.display_text);
                    app.restore_prepared_turn(queued);
                    return Ok(false);
                }
                enqueue_prepared_turn_for_cli(
                    queued,
                    app,
                    ui_trace,
                    runtime,
                    lash_core::DeliveryPolicy::AfterCurrentTurnCommit,
                    lash_core::SlotPolicy::Exclusive,
                    true,
                )
                .await;
                return Ok(false);
            }
            if runtime.is_none() {
                push_system_message(
                    app,
                    "Cannot send this input while the session is switching.".to_string(),
                );
                app.restore_prepared_turn(queued);
                return Ok(false);
            }

            if let Some(cmd) = parsed_command {
                if let Some(recorder) = ui_trace.as_mut() {
                    recorder.record_slash_command(queued.display_text.clone());
                }
                if handle_parsed_slash_command(
                    cmd,
                    SlashCommandCtx {
                        terminal: &mut *terminal,
                        app: &mut *app,
                        logger: &mut *logger,
                        args,
                        paused,
                        ui_extensions,
                        runtime_factory,
                        lash_config,
                        runtime: &mut *runtime,
                        history: &mut *history,
                        turn_counter: &mut *turn_counter,
                        last_turn: &mut *last_turn,
                        runtime_return_rx: &mut *runtime_return_rx,
                        cancel_token: &mut *cancel_token,
                        active_stream_id: &mut *active_stream_id,
                        provider: &mut *provider,
                        current_model_variant: &mut *current_model_variant,
                        current_execution_mode: &mut *current_execution_mode,
                        active_tool_state: &mut *active_tool_state,
                        model_catalog,
                        toolset_hash: &mut *toolset_hash,
                        app_tx,
                        pending_clear_after_return: &mut *pending_clear_after_return,
                    },
                )
                .await?
                {
                    return Ok(true);
                }
                return Ok(false);
            }

            let turn_input = make_turn_input(&queued);
            send_user_message(
                queued.clone(),
                turn_input.clone(),
                app,
                ui_trace.as_mut(),
                logger,
                runtime,
                history,
                runtime_return_rx,
                cancel_token,
                active_stream_id,
                app_tx,
            )
            .await;
            *last_turn = Some(TurnReplayPayload {
                turn_input,
                prepared_turn: queued,
                execution_mode: current_execution_mode.clone(),
            });
        }
        // Up/Down: navigate suggestions when popup is visible
        KeyCode::Up if app.has_suggestions() => {
            if let Some(recorder) = ui_trace.as_mut() {
                recorder.record_suggestion_up();
            }
            app.editor.suggestion_up();
        }
        KeyCode::Down if app.has_suggestions() => {
            if let Some(recorder) = ui_trace.as_mut() {
                recorder.record_suggestion_down();
            }
            app.editor.suggestion_down();
        }
        // Enter with the suggestion popup open accepts the highlighted entry,
        // matching Tab. Shift/Alt+Enter still falls through to insert a newline.
        KeyCode::Enter
            if app.has_suggestions()
                && !key
                    .modifiers
                    .intersects(KeyModifiers::SHIFT | KeyModifiers::ALT) =>
        {
            if let Some((command_text, cmd)) = selected_slash_command_suggestion(app, ui_extensions)
            {
                if let Some(recorder) = ui_trace.as_mut() {
                    recorder.record_suggestion_complete();
                    recorder.record_slash_command(command_text.clone());
                }
                app.set_input(String::new());
                app.update_suggestions();
                if app.turn_active() {
                    if slash_command_runs_out_of_band_while_running(&cmd) {
                        if handle_parsed_slash_command(
                            cmd,
                            SlashCommandCtx {
                                terminal: &mut *terminal,
                                app: &mut *app,
                                logger: &mut *logger,
                                args,
                                paused,
                                ui_extensions,
                                runtime_factory,
                                lash_config,
                                runtime: &mut *runtime,
                                history: &mut *history,
                                turn_counter: &mut *turn_counter,
                                last_turn: &mut *last_turn,
                                runtime_return_rx: &mut *runtime_return_rx,
                                cancel_token: &mut *cancel_token,
                                active_stream_id: &mut *active_stream_id,
                                provider: &mut *provider,
                                current_model_variant: &mut *current_model_variant,
                                current_execution_mode: &mut *current_execution_mode,
                                active_tool_state: &mut *active_tool_state,
                                model_catalog,
                                toolset_hash: &mut *toolset_hash,
                                app_tx,
                                pending_clear_after_return: &mut *pending_clear_after_return,
                            },
                        )
                        .await?
                        {
                            return Ok(true);
                        }
                        return Ok(false);
                    }
                    block_slash_command_while_working(app, &command_text);
                    app.set_input(command_text);
                    app.update_suggestions();
                    return Ok(false);
                }
                if handle_parsed_slash_command(
                    cmd,
                    SlashCommandCtx {
                        terminal: &mut *terminal,
                        app: &mut *app,
                        logger: &mut *logger,
                        args,
                        paused,
                        ui_extensions,
                        runtime_factory,
                        lash_config,
                        runtime: &mut *runtime,
                        history: &mut *history,
                        turn_counter: &mut *turn_counter,
                        last_turn: &mut *last_turn,
                        runtime_return_rx: &mut *runtime_return_rx,
                        cancel_token: &mut *cancel_token,
                        active_stream_id: &mut *active_stream_id,
                        provider: &mut *provider,
                        current_model_variant: &mut *current_model_variant,
                        current_execution_mode: &mut *current_execution_mode,
                        active_tool_state: &mut *active_tool_state,
                        model_catalog,
                        toolset_hash: &mut *toolset_hash,
                        app_tx,
                        pending_clear_after_return: &mut *pending_clear_after_return,
                    },
                )
                .await?
                {
                    return Ok(true);
                }
                return Ok(false);
            }
            if let Some(recorder) = ui_trace.as_mut() {
                recorder.record_suggestion_complete();
            }
            app.complete_suggestion();
            app.update_suggestions();
        }
        KeyCode::Enter if process_dock_has_focus(app) => {
            show_selected_process_overview(app);
        }
        KeyCode::Delete if process_dock_has_focus(app) => {
            cancel_selected_process(app, runtime).await;
        }
        KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            app.editor.kill_to_line_start();
            app.update_suggestions();
        }
        KeyCode::Char('k') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            app.editor.kill_to_line_end();
            app.update_suggestions();
        }
        KeyCode::Enter => {
            // Shift+Enter or Alt+Enter → insert newline
            if key.modifiers.contains(KeyModifiers::SHIFT)
                || key.modifiers.contains(KeyModifiers::ALT)
            {
                app.insert_char('\n');
                app.update_suggestions();
                return Ok(false);
            }

            let Some(queued) = app.try_take_prepared_turn() else {
                push_system_message(
                    app,
                    "Wait for pasted images to finish processing before sending or queueing this draft.",
                );
                return Ok(false);
            };
            let queued = normalize_prepared_turn_for_dispatch(queued, &app.skills);
            app.update_suggestions();
            if queued.is_empty() {
                return Ok(false);
            }

            let parsed_command =
                parse_slash_command(&queued.display_text, &app.skills, ui_extensions);
            let is_host_slash_command = parsed_command.is_some();

            if app.turn_active() {
                if let Some(cmd) = parsed_command
                    && slash_command_runs_out_of_band_while_running(&cmd)
                {
                    if let Some(recorder) = ui_trace.as_mut() {
                        recorder.record_slash_command(queued.display_text.clone());
                    }
                    if handle_parsed_slash_command(
                        cmd,
                        SlashCommandCtx {
                            terminal: &mut *terminal,
                            app: &mut *app,
                            logger: &mut *logger,
                            args,
                            paused,
                            ui_extensions,
                            runtime_factory,
                            lash_config,
                            runtime: &mut *runtime,
                            history: &mut *history,
                            turn_counter: &mut *turn_counter,
                            last_turn: &mut *last_turn,
                            runtime_return_rx: &mut *runtime_return_rx,
                            cancel_token: &mut *cancel_token,
                            active_stream_id: &mut *active_stream_id,
                            provider: &mut *provider,
                            current_model_variant: &mut *current_model_variant,
                            current_execution_mode: &mut *current_execution_mode,
                            active_tool_state: &mut *active_tool_state,
                            model_catalog,
                            toolset_hash: &mut *toolset_hash,
                            app_tx,
                            pending_clear_after_return: &mut *pending_clear_after_return,
                        },
                    )
                    .await?
                    {
                        return Ok(true);
                    }
                    return Ok(false);
                }
                if is_host_slash_command {
                    block_slash_command_while_working(app, &queued.display_text);
                    app.restore_prepared_turn(queued);
                    return Ok(false);
                }
                if shell_escape_command(&queued.display_text).is_some() {
                    push_system_message(
                        app,
                        "Shell escapes cannot be injected into the active turn. Wait for completion or use `Tab` to queue a later turn.",
                    );
                    app.restore_prepared_turn(queued);
                    return Ok(false);
                }
                match app.route_turn_submission(runtime.is_some()) {
                    TurnSubmissionRoute::QueueNextFullTurn => {
                        enqueue_prepared_turn_for_cli(
                            queued,
                            app,
                            ui_trace,
                            runtime,
                            lash_core::DeliveryPolicy::AfterCurrentTurnCommit,
                            lash_core::SlotPolicy::Exclusive,
                            true,
                        )
                        .await;
                        return Ok(false);
                    }
                    TurnSubmissionRoute::BlockedSessionSwitch => {
                        push_system_message(
                            app,
                            "Cannot send this input while the session is switching.".to_string(),
                        );
                        app.restore_prepared_turn(queued);
                        return Ok(false);
                    }
                    TurnSubmissionRoute::SendNow => {
                        push_system_message(
                            app,
                            "Cannot start a new turn while the current turn is active.".to_string(),
                        );
                        app.restore_prepared_turn(queued);
                        return Ok(false);
                    }
                    TurnSubmissionRoute::InjectActiveTurn => {}
                }
                let injection = lash_core::InjectedTurnInput {
                    id: Some(queued.draft_id.clone()),
                    message: make_injected_plugin_message(&queued).await,
                };
                let Some(session) = runtime.as_ref() else {
                    push_system_message(
                        app,
                        "Current-turn injection is unavailable while the session is switching.",
                    );
                    app.restore_prepared_turn(queued);
                    return Ok(false);
                };
                match session
                    .control()
                    .injection()
                    .inject_turn_inputs(vec![injection])
                    .await
                {
                    Ok(()) => {
                        record_queue_current_turn_input(ui_trace, &queued);
                        app.cache_draft_presentation(queued.clone());
                        if let Err(err) = refresh_queued_work_snapshot(app, runtime).await {
                            push_system_message(
                                app,
                                format!("Failed to refresh durable queue: {err}"),
                            );
                        }
                    }
                    Err(err) => {
                        push_system_message(
                            app,
                            format!("Failed to queue current-turn injection: {}", err),
                        );
                        app.restore_prepared_turn(queued);
                    }
                }
                return Ok(false);
            }
            if runtime.is_none() {
                push_system_message(
                    app,
                    "Cannot send this input while the session is switching.".to_string(),
                );
                app.restore_prepared_turn(queued);
                return Ok(false);
            }

            // Shell escape: !command
            if let Some(cmd_str) = shell_escape_command(&queued.display_text) {
                if !cmd_str.is_empty() {
                    app.push_prepared_user_input(&queued);
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
                return Ok(false);
            }

            // Try slash command
            if let Some(cmd) = parse_slash_command(&queued.display_text, &app.skills, ui_extensions)
            {
                if let Some(recorder) = ui_trace.as_mut() {
                    recorder.record_slash_command(queued.display_text.clone());
                }
                if handle_parsed_slash_command(
                    cmd,
                    SlashCommandCtx {
                        terminal: &mut *terminal,
                        app: &mut *app,
                        logger: &mut *logger,
                        args,
                        paused,
                        ui_extensions,
                        runtime_factory,
                        lash_config,
                        runtime: &mut *runtime,
                        history: &mut *history,
                        turn_counter: &mut *turn_counter,
                        last_turn: &mut *last_turn,
                        runtime_return_rx: &mut *runtime_return_rx,
                        cancel_token: &mut *cancel_token,
                        active_stream_id: &mut *active_stream_id,
                        provider: &mut *provider,
                        current_model_variant: &mut *current_model_variant,
                        current_execution_mode: &mut *current_execution_mode,
                        active_tool_state: &mut *active_tool_state,
                        model_catalog,
                        toolset_hash: &mut *toolset_hash,
                        app_tx,
                        pending_clear_after_return: &mut *pending_clear_after_return,
                    },
                )
                .await?
                {
                    return Ok(true);
                }
                return Ok(false);
            }

            // Handle "quit"/"exit" without slash prefix
            if queued.display_text == "quit" || queued.display_text == "exit" {
                return Ok(true);
            }

            // Regular user message.
            match app.route_turn_submission(runtime.is_some()) {
                TurnSubmissionRoute::SendNow => {
                    let turn_input = make_turn_input(&queued);
                    send_user_message(
                        queued.clone(),
                        turn_input.clone(),
                        app,
                        ui_trace.as_mut(),
                        logger,
                        runtime,
                        history,
                        runtime_return_rx,
                        cancel_token,
                        active_stream_id,
                        app_tx,
                    )
                    .await;
                    *last_turn = Some(TurnReplayPayload {
                        turn_input,
                        prepared_turn: queued,
                        execution_mode: current_execution_mode.clone(),
                    });
                }
                TurnSubmissionRoute::QueueNextFullTurn => {
                    enqueue_prepared_turn_for_cli(
                        queued,
                        app,
                        ui_trace,
                        runtime,
                        lash_core::DeliveryPolicy::AfterCurrentTurnCommit,
                        lash_core::SlotPolicy::Exclusive,
                        true,
                    )
                    .await;
                }
                TurnSubmissionRoute::InjectActiveTurn => {
                    push_system_message(
                        app,
                        "Cannot inject into a turn from the idle input path.".to_string(),
                    );
                    app.restore_prepared_turn(queued);
                }
                TurnSubmissionRoute::BlockedSessionSwitch => {
                    push_system_message(
                        app,
                        "Cannot send this input while the session is switching.".to_string(),
                    );
                    app.restore_prepared_turn(queued);
                }
            }
        }
        KeyCode::Backspace => {
            if let Some(recorder) = ui_trace.as_mut() {
                recorder.record_input_backspace();
            }
            app.editor.backspace();
            app.update_suggestions();
        }
        KeyCode::Delete => {
            if let Some(recorder) = ui_trace.as_mut() {
                recorder.record_input_delete();
            }
            app.editor.delete();
            app.update_suggestions();
        }
        KeyCode::Left if key.modifiers.contains(KeyModifiers::CONTROL) => {
            if let Some(recorder) = ui_trace.as_mut() {
                recorder.record_move_cursor_left();
            }
            app.editor.move_cursor_word_left();
        }
        KeyCode::Right if key.modifiers.contains(KeyModifiers::CONTROL) => {
            if let Some(recorder) = ui_trace.as_mut() {
                recorder.record_move_cursor_right();
            }
            app.editor.move_cursor_word_right();
        }
        KeyCode::Left => {
            if let Some(recorder) = ui_trace.as_mut() {
                recorder.record_move_cursor_left();
            }
            app.editor.move_cursor_left();
        }
        KeyCode::Right => {
            if let Some(recorder) = ui_trace.as_mut() {
                recorder.record_move_cursor_right();
            }
            app.editor.move_cursor_right();
        }
        KeyCode::Home => {
            if let Some(recorder) = ui_trace.as_mut() {
                recorder.record_move_cursor_home();
            }
            app.editor.move_cursor_home();
        }
        KeyCode::End => {
            if let Some(recorder) = ui_trace.as_mut() {
                recorder.record_move_cursor_end();
            }
            app.editor.move_cursor_end();
        }
        KeyCode::Up => {
            if let Some(recorder) = ui_trace.as_mut() {
                recorder.record_history_up();
            }
            app.history_up();
        }
        KeyCode::Down => {
            if let Some(recorder) = ui_trace.as_mut() {
                recorder.record_history_down();
            }
            app.editor.history_down();
        }
        KeyCode::Char(c) => {
            if let Some(recorder) = ui_trace.as_mut() {
                recorder.record_input_insert_text(c.to_string());
            }
            app.editor.insert_text(&c.to_string());
            app.update_suggestions();
        }
        _ => {}
    }
    Ok(false)
}
