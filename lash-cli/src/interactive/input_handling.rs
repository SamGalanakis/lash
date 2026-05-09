use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use crossterm::event::{
    Event as TermEvent, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseEvent,
};
use lash::session_model::Message;
use lash::*;
use lash_embed::LashSession;
use lash_tui::{InputEvent as TuiInputEvent, Terminal, normalize_event};
use lash_tui_extensions::{TuiExtensionContext, TuiExtensions, TuiInputOutcome, TuiSurfaceSlot};
use tokio::task;
use tokio_util::sync::CancellationToken;

use crate::app::{App, PreparedTurn, UiTimelineItem};
use crate::editor::SuggestionKind;
use crate::event::AppEvent;
use crate::input_items::insert_inline_marker;
use crate::render;
use crate::session_log::SessionLogger;
use crate::turn_runner::{RuntimeRunResult, make_turn_input};
use crate::ui_action::{UiAction, UiActionContext, UiActionOutcome, apply_ui_action};
use crate::ui_trace::UiTraceRecorder;
use crate::{
    Args, apply_ui_host_effects, hash12, normalize_prepared_turn_for_dispatch, push_system_message,
    shell_escape_command,
};

use super::commands::{
    handle_parsed_slash_command, parse_slash_command, slash_command_runs_out_of_band_while_running,
    switch_to_session_identifier,
};
use super::helpers::{
    TurnReplayPayload, is_copy_shortcut, key_chord_from_event, monitor_wake_message,
    queued_turn_edit_matches, record_queue_pending_steer, record_queue_turn,
    should_preserve_selection_for_key,
};
use super::runtime::{
    apply_pending_reconfigure, copy_selected_text_or_last_response, make_injected_plugin_message,
    send_user_message,
};

pub(super) fn handle_surface_input(
    ui_extensions: &TuiExtensions,
    event: &TuiInputEvent,
    plugin_host: &PluginHost,
    session_manager: &Arc<dyn RuntimeSessionHost>,
    app: &mut App,
) -> bool {
    let tui_session = crate::tui_extension_session::LegacyTuiExtensionSession {
        plugin_host,
        session_id: app.session_id.as_str(),
        session_manager: Arc::clone(session_manager),
    };
    match ui_extensions.handle_input(
        event,
        TuiExtensionContext {
            session: &tui_session,
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

pub(super) fn current_ui_action_context(app: &App, terminal: &Terminal) -> UiActionContext {
    let (width, height) = terminal.size().unwrap_or((80, 24));
    let viewport_height = render::history_viewport_height(app, width, height);
    let prompt_max_scroll = if let Some(prompt) = app.prompt_state() {
        let prompt_area = render::input_area(app, width, height);
        let visible_height = prompt_area.height as usize;
        let inner_width = prompt_area.width.saturating_sub(2) as usize;
        render::prompt_max_scroll(prompt, inner_width.max(1), visible_height)
    } else {
        0
    };
    UiActionContext {
        viewport_width: width as usize,
        viewport_height,
        prompt_max_scroll,
    }
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

pub(super) fn apply_terminal_action(
    app: &mut App,
    terminal: &Terminal,
    action: UiAction,
) -> UiActionOutcome {
    let context = current_ui_action_context(app, terminal);
    apply_ui_action(app, action, context)
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum ScrollInputAction {
    Prompt(UiAction),
    History {
        action: UiAction,
        trace_amount: usize,
    },
}

fn classify_always_on_scroll_key(
    key: KeyEvent,
    app: &App,
    viewport_height: usize,
) -> Option<ScrollInputAction> {
    let half_page = viewport_height / 2;
    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);

    if app.has_prompt() {
        if ctrl && key.code == KeyCode::Char('u') {
            return Some(ScrollInputAction::Prompt(UiAction::PromptScrollUp(
                half_page,
            )));
        }
        if ctrl && key.code == KeyCode::Char('d') {
            return Some(ScrollInputAction::Prompt(UiAction::PromptScrollDown(
                half_page,
            )));
        }
        if key.code == KeyCode::PageUp {
            return Some(ScrollInputAction::Prompt(UiAction::PromptScrollUp(
                viewport_height,
            )));
        }
        if key.code == KeyCode::PageDown {
            return Some(ScrollInputAction::Prompt(UiAction::PromptScrollDown(
                viewport_height,
            )));
        }
        if !app.is_prompt_text_entry() && !app.prompt_has_options() {
            if key.code == KeyCode::Up {
                return Some(ScrollInputAction::Prompt(UiAction::PromptScrollUp(1)));
            }
            if key.code == KeyCode::Down {
                return Some(ScrollInputAction::Prompt(UiAction::PromptScrollDown(1)));
            }
        }
    }

    if ctrl && key.code == KeyCode::Char('u') {
        return Some(ScrollInputAction::History {
            action: UiAction::ScrollUp(half_page),
            trace_amount: half_page,
        });
    }
    if ctrl && key.code == KeyCode::Char('d') {
        return Some(ScrollInputAction::History {
            action: UiAction::ScrollDown(half_page),
            trace_amount: half_page,
        });
    }
    if key.code == KeyCode::PageUp {
        return Some(ScrollInputAction::History {
            action: UiAction::ScrollUp(viewport_height),
            trace_amount: viewport_height,
        });
    }
    if key.code == KeyCode::PageDown {
        return Some(ScrollInputAction::History {
            action: UiAction::ScrollDown(viewport_height),
            trace_amount: viewport_height,
        });
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
        ScrollInputAction::Prompt(action) => {
            let _ = apply_terminal_action(app, terminal, action);
        }
        ScrollInputAction::History {
            action,
            trace_amount,
        } => {
            if let Some(recorder) = ui_trace.as_mut() {
                match action {
                    UiAction::ScrollUp(_) => recorder.record_scroll_up(trace_amount),
                    UiAction::ScrollDown(_) => recorder.record_scroll_down(trace_amount),
                    _ => {}
                }
            }
            let _ = apply_terminal_action(app, terminal, action);
        }
    }
    app.dirty = true;
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn activate_foreground_session_handoff(
    app: &mut App,
    session_id: String,
    _history: &mut Vec<lash::session_model::Message>,
    _runtime: &mut Option<LashSession>,
    _turn_counter: &mut usize,
    _current_execution_mode: &mut ExecutionMode,
    _current_model_variant: &mut Option<String>,
    session_manager: &mut Arc<dyn RuntimeSessionHost>,
    _ui_extensions: &TuiExtensions,
    _plugin_host: &PluginHost,
) -> bool {
    let queued_turn = match session_manager.take_first_turn_input(&session_id).await {
        Ok(Some(seed)) if !seed.content.trim().is_empty() => {
            Some(PreparedTurn::prepare_with_effective_text(
                seed.content.clone(),
                seed.content,
                Vec::new(),
            ))
        }
        Ok(_) => None,
        Err(err) => {
            push_system_message(
                app,
                format!("Failed to read first turn for session `{session_id}`: {err}"),
            );
            None
        }
    };

    app.dirty = true;

    if let Some(turn) = queued_turn {
        app.queue_turn(turn);
    }

    true
}

pub(super) async fn enqueue_pending_monitor_wakes(
    app: &mut App,
    session: &LashSession,
) -> Result<usize, String> {
    let wakes = app.take_pending_monitor_wakes();
    if wakes.is_empty() {
        return Ok(0);
    }
    let messages = wakes
        .iter()
        .map(|input| InjectedTurnInput {
            id: None,
            message: monitor_wake_message(input),
        })
        .collect::<Vec<_>>();
    session
        .control()
        .inject_turn_inputs(messages)
        .await
        .map_err(|err| err.to_string())?;
    app.mark_monitor_wakes_in_flight(&wakes);
    Ok(wakes.len())
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn process_pending_monitor_wakes(
    app: &mut App,
    ui_trace: &mut Option<UiTraceRecorder>,
    logger: &mut SessionLogger,
    runtime: &mut Option<LashSession>,
    history: &mut Vec<lash::session_model::Message>,
    runtime_return_rx: &mut Option<tokio::sync::oneshot::Receiver<RuntimeRunResult>>,
    cancel_token: &mut Option<CancellationToken>,
    active_stream_id: &mut u64,
    app_tx: &crate::event::AppEventTx,
    desired_dynamic: &DynamicStateSnapshot,
) -> Result<bool, String> {
    if !app.running && runtime_return_rx.is_none() && runtime.is_none() {
        return Ok(false);
    }
    let Some(session) = runtime.as_ref() else {
        return Ok(false);
    };
    let injected = enqueue_pending_monitor_wakes(app, session).await?;
    if injected == 0 {
        return Ok(false);
    }
    if app.running || runtime_return_rx.is_some() || app.has_queued_messages() {
        return Ok(false);
    }

    let prepared_turn = PreparedTurn::prepare(String::new(), Vec::new(), &app.skills);
    let turn_input = TurnInput {
        items: Vec::new(),
        image_blobs: Default::default(),
        user_input: None,
        mode: Some(RunMode::Normal),
        mode_turn_options: None,
        trace_turn_id: None,
        mode_extension: None,
        turn_context: lash::TurnContext::default(),
    };
    send_user_message(
        prepared_turn,
        turn_input,
        app,
        ui_trace.as_mut(),
        logger,
        runtime,
        history,
        runtime_return_rx,
        cancel_token,
        active_stream_id,
        app_tx,
        desired_dynamic,
    )
    .await;
    Ok(true)
}

pub(super) fn handle_mouse_event(
    mouse: MouseEvent,
    app: &mut App,
    terminal: &Terminal,
    ui_trace: &mut Option<UiTraceRecorder>,
    ui_extensions: &TuiExtensions,
    plugin_host: &PluginHost,
    session_manager: &Arc<dyn RuntimeSessionHost>,
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
                let _ = apply_terminal_action(app, terminal, UiAction::PromptScrollUp(3));
                app.dirty = true;
                return Ok(());
            }
            MouseEventKind::ScrollDown if !app.prompt_uses_split_layout() || in_prompt_review => {
                let _ = apply_terminal_action(app, terminal, UiAction::PromptScrollDown(3));
                app.dirty = true;
                return Ok(());
            }
            _ => {}
        }
    }

    if let Some(event) = normalize_event(&TermEvent::Mouse(mouse))
        && handle_surface_input(ui_extensions, &event, plugin_host, session_manager, app)
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

#[allow(clippy::too_many_arguments)]
pub(super) async fn handle_key_event(
    key: KeyEvent,
    app: &mut App,
    terminal: &mut Terminal,
    ui_trace: &mut Option<UiTraceRecorder>,
    logger: &mut SessionLogger,
    args: &Args,
    paused: &Arc<AtomicBool>,
    plugin_host: &PluginHost,
    ui_extensions: &TuiExtensions,
    runtime_factory: &crate::session_bootstrap::CliSessionOpener,
    lash_config: &lash::provider::LashConfig,
    runtime: &mut Option<LashSession>,
    history: &mut Vec<Message>,
    turn_counter: &mut usize,
    last_turn: &mut Option<TurnReplayPayload>,
    runtime_return_rx: &mut Option<tokio::sync::oneshot::Receiver<RuntimeRunResult>>,
    cancel_token: &mut Option<CancellationToken>,
    active_stream_id: &mut u64,
    provider: &mut ProviderHandle,
    current_model_variant: &mut Option<String>,
    current_execution_mode: &mut ExecutionMode,
    session_manager: &mut Arc<dyn RuntimeSessionHost>,
    desired_dynamic: &mut DynamicStateSnapshot,
    pending_reconfigure: &mut bool,
    model_catalog: &CachedModelCatalog,
    toolset_hash: &mut String,
    app_tx: &crate::event::AppEventTx,
    pending_clear_after_return: &mut bool,
) -> anyhow::Result<bool> {
    // With kitty keyboard protocol, ignore Release/Repeat events
    if key.kind != KeyEventKind::Press {
        return Ok(false);
    }
    app.dirty = true;
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
        let _ = apply_terminal_action(app, terminal, UiAction::ClearSelection);
    }

    // Active selection copy should win before generic Ctrl+C handling.
    if (app.selection.visible || app.has_input_selection()) && copy_shortcut {
        tracing::debug!("selection copy took precedence over generic key handling");
        copy_selected_text_or_last_response(app, terminal.size().ok());
        return Ok(false);
    }

    // CTRL+C: dismiss prompt if active, else quit
    if !key.modifiers.contains(KeyModifiers::SHIFT)
        && key.modifiers.contains(KeyModifiers::CONTROL)
        && matches!(key.code, KeyCode::Char(c) if c.eq_ignore_ascii_case(&'c'))
    {
        tracing::debug!(
            has_prompt = app.has_prompt(),
            "handling Ctrl+C as dismiss/quit"
        );
        if app.has_prompt() {
            if let Some(recorder) = ui_trace.as_mut() {
                recorder.record_prompt_dismiss();
            }
            let _ = apply_terminal_action(app, terminal, UiAction::DismissPrompt);
            return Ok(false);
        }
        return Ok(true);
    }

    // ALT+O: reliable full expand toggle across most terminals.
    if key.modifiers.contains(KeyModifiers::ALT)
        && matches!(key.code, KeyCode::Char(c) if c.eq_ignore_ascii_case(&'o'))
    {
        app.toggle_full_expand();
        return Ok(false);
    }

    if queued_turn_edit_matches(key) {
        if let Some((turn, _was_pending)) = app.take_last_queued_turn() {
            app.restore_prepared_turn(turn);
            app.update_suggestions();
        }
        return Ok(false);
    }

    // CTRL+O: cycle expand (0↔1)
    if key.modifiers.contains(KeyModifiers::CONTROL)
        && matches!(key.code, KeyCode::Char(c) if c.eq_ignore_ascii_case(&'o'))
    {
        app.cycle_expand();
        return Ok(false);
    }

    // CTRL+SHIFT+Z: redo the most recently undone edit.
    if key.modifiers.contains(KeyModifiers::CONTROL)
        && key.modifiers.contains(KeyModifiers::SHIFT)
        && matches!(key.code, KeyCode::Char(c) if c.eq_ignore_ascii_case(&'z'))
    {
        if app.editor_redo() {
            app.update_suggestions();
        }
        return Ok(false);
    }

    // CTRL+Z: undo the most recent edit to the input draft.
    if key.modifiers.contains(KeyModifiers::CONTROL)
        && !key.modifiers.contains(KeyModifiers::SHIFT)
        && matches!(key.code, KeyCode::Char(c) if c.eq_ignore_ascii_case(&'z'))
    {
        if app.editor_undo() {
            app.update_suggestions();
        }
        return Ok(false);
    }

    // ALT+Z: redo fallback for terminals that swallow CTRL+SHIFT+Z.
    if key.modifiers.contains(KeyModifiers::ALT)
        && matches!(key.code, KeyCode::Char(c) if c.eq_ignore_ascii_case(&'z'))
    {
        if app.editor_redo() {
            app.update_suggestions();
        }
        return Ok(false);
    }

    // CTRL+Y / CTRL+SHIFT+C: copy current selection when present,
    // otherwise the last assistant response.
    if copy_shortcut {
        tracing::debug!("copy shortcut matched without active selection precedence");
        copy_selected_text_or_last_response(app, terminal.size().ok());
        return Ok(false);
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
        return Ok(false);
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
        return Ok(false);
    }

    // Escape key behavior depends on state
    if key.code == KeyCode::Esc {
        if app.has_prompt() {
            if let Some(recorder) = ui_trace.as_mut() {
                recorder.record_prompt_dismiss();
            }
            app.dismiss_prompt();
        } else if app.has_tree() {
            let _ = apply_terminal_action(app, terminal, UiAction::DismissTree);
        } else if app.has_skill_picker() {
            let _ = apply_terminal_action(app, terminal, UiAction::DismissSkillPicker);
        } else if app.has_session_picker() {
            let _ = apply_terminal_action(app, terminal, UiAction::DismissSessionPicker);
        } else if app.running {
            // Interrupt running session
            app.note_manual_interrupt_requested();
            if let Some(token) = cancel_token.take() {
                token.cancel();
            }
        }
        // When idle with no dialog: no-op
        return Ok(false);
    }

    // ── Always-on scroll keys (work in all states) ──
    {
        let (width, height) = terminal.size()?;
        let vh = render::history_viewport_height(app, width, height);
        if let Some(action) = classify_always_on_scroll_key(key, app, vh) {
            apply_scroll_input_action(app, terminal, ui_trace, action);
            return Ok(false);
        }
    }

    // ── Skill picker key handling ──
    if app.has_skill_picker() {
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => {
                let _ = apply_terminal_action(app, terminal, UiAction::SkillPickerUp);
            }
            KeyCode::Down | KeyCode::Char('j') => {
                let _ = apply_terminal_action(app, terminal, UiAction::SkillPickerDown);
            }
            KeyCode::Enter => {
                if let UiActionOutcome::SkillPicked(Some(name)) =
                    apply_terminal_action(app, terminal, UiAction::SubmitSkillPicker)
                {
                    app.set_input(format!("${} ", name));
                }
            }
            _ => {}
        }
        return Ok(false);
    }

    // ── Session picker key handling ──
    if app.has_session_picker() {
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => {
                let _ = apply_terminal_action(app, terminal, UiAction::SessionPickerUp);
            }
            KeyCode::Down | KeyCode::Char('j') => {
                let _ = apply_terminal_action(app, terminal, UiAction::SessionPickerDown);
            }
            KeyCode::Enter => {
                if let UiActionOutcome::SessionPicked(Some(filename)) =
                    apply_terminal_action(app, terminal, UiAction::SubmitSessionPicker)
                {
                    match switch_to_session_identifier(
                        &filename,
                        app,
                        logger,
                        runtime_factory,
                        lash_config,
                        runtime,
                        history,
                        turn_counter,
                        provider,
                        current_model_variant,
                        current_execution_mode,
                        session_manager,
                        desired_dynamic,
                        model_catalog,
                        toolset_hash,
                    )
                    .await
                    {
                        Ok(()) => {
                            app.dirty = true;
                        }
                        Err(err) => {
                            app.timeline
                                .push(UiTimelineItem::SystemMessage(err.to_string()));
                            app.invalidate_height_cache();
                            app.scroll_to_bottom();
                        }
                    }
                }
            }
            _ => {} // ignore other keys while picker is open
        }
        return Ok(false);
    }

    // ── Tree overlay key handling ──
    if app.has_tree() {
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => {
                let _ = apply_terminal_action(app, terminal, UiAction::TreeUp);
            }
            KeyCode::Down | KeyCode::Char('j') => {
                let _ = apply_terminal_action(app, terminal, UiAction::TreeDown);
            }
            KeyCode::Left | KeyCode::Right
                if key
                    .modifiers
                    .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT) =>
            {
                let action = if key.code == KeyCode::Left {
                    UiAction::TreePrevBranch
                } else {
                    UiAction::TreeNextBranch
                };
                let _ = apply_terminal_action(app, terminal, action);
            }
            KeyCode::Enter => {
                if let UiActionOutcome::TreePicked(Some(selection)) =
                    apply_terminal_action(app, terminal, UiAction::SubmitTree)
                {
                    let current_dynamic_state = desired_dynamic.clone();
                    let Some(rt) = runtime.as_ref() else {
                        push_system_message(
                            app,
                            "Branch navigation is unavailable while a turn is running.",
                        );
                        return Ok(false);
                    };
                    match crate::tree::switch_to_tree_selection(
                        rt,
                        logger,
                        app,
                        history,
                        selection,
                        &current_dynamic_state,
                    )
                    .await
                    {
                        Ok(()) => {
                            match rt.session_manager().await {
                                Ok(manager) => *session_manager = manager,
                                Err(err) => push_system_message(
                                    app,
                                    format!("Failed to refresh session manager: {}", err),
                                ),
                            }
                            app.dirty = true;
                        }
                        Err(err) => {
                            push_system_message(app, format!("Branch switch failed: {err}"));
                        }
                    }
                }
            }
            _ => {}
        }
        return Ok(false);
    }

    // ── Prompt (ask dialog) key handling ──
    if app.has_prompt() {
        let editing_text = app.is_prompt_text_entry();
        match key.code {
            KeyCode::Tab if app.prompt_supports_note() => {
                if let Some(recorder) = ui_trace.as_mut() {
                    recorder.record_prompt_toggle_note_focus();
                }
                let _ = apply_terminal_action(app, terminal, UiAction::PromptToggleNoteFocus);
            }
            KeyCode::Up if !editing_text => {
                if let Some(recorder) = ui_trace.as_mut() {
                    recorder.record_prompt_up();
                }
                let _ = apply_terminal_action(app, terminal, UiAction::PromptUp);
            }
            KeyCode::Down if !editing_text => {
                if let Some(recorder) = ui_trace.as_mut() {
                    recorder.record_prompt_down();
                }
                let _ = apply_terminal_action(app, terminal, UiAction::PromptDown);
            }
            KeyCode::Char(' ') if app.is_prompt_multi_select() && !editing_text => {
                if let Some(recorder) = ui_trace.as_mut() {
                    recorder.record_prompt_toggle_current_option();
                }
                let _ = apply_terminal_action(app, terminal, UiAction::PromptToggleCurrentOption);
            }
            KeyCode::BackTab if editing_text => {
                if let Some(recorder) = ui_trace.as_mut() {
                    recorder.record_prompt_insert_text("\n");
                }
                let _ = apply_terminal_action(
                    app,
                    terminal,
                    UiAction::PromptInsertText("\n".to_string()),
                );
            }
            KeyCode::Enter => {
                if let Some(recorder) = ui_trace.as_mut() {
                    recorder.record_submit_prompt();
                }
                let _ = apply_terminal_action(app, terminal, UiAction::SubmitPrompt);
            }
            KeyCode::Char(c) if editing_text => {
                if let Some(recorder) = ui_trace.as_mut() {
                    recorder.record_prompt_insert_text(c.to_string());
                }
                let _ =
                    apply_terminal_action(app, terminal, UiAction::PromptInsertText(c.to_string()));
            }
            KeyCode::Backspace if editing_text => {
                if let Some(recorder) = ui_trace.as_mut() {
                    recorder.record_prompt_backspace();
                }
                let _ = apply_terminal_action(app, terminal, UiAction::PromptBackspace);
            }
            _ => {}
        }
        return Ok(false);
    }

    if let Some(event) = normalize_event(&TermEvent::Key(key))
        && handle_surface_input(ui_extensions, &event, plugin_host, session_manager, app)
    {
        return Ok(false);
    }

    if let Some(chord) = key_chord_from_event(key)
        && let Some(shortcut) = ui_extensions.shortcut_for(chord)
    {
        let tui_session = crate::tui_extension_session::LegacyTuiExtensionSession {
            plugin_host,
            session_id: app.session_id.as_str(),
            session_manager: Arc::clone(session_manager),
        };
        match ui_extensions
            .invoke_shortcut(
                &shortcut,
                TuiExtensionContext {
                    session: &tui_session,
                },
            )
            .await
        {
            Ok(effects) => apply_ui_host_effects(app, effects),
            Err(err) => push_system_message(app, err),
        }
        return Ok(false);
    }

    match key.code {
        // Tab: complete selected suggestion
        KeyCode::Tab if app.has_suggestions() => {
            if let Some(recorder) = ui_trace.as_mut() {
                recorder.record_suggestion_complete();
            }
            let _ = apply_terminal_action(app, terminal, UiAction::SuggestionComplete);
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
            if queued.is_empty()
                || shell_escape_command(&queued.display_text).is_some()
                || (is_host_slash_command && !app.running)
            {
                app.restore_prepared_turn(queued);
                return Ok(false);
            }
            if app.running {
                if let Some(cmd) = parsed_command
                    && slash_command_runs_out_of_band_while_running(&cmd)
                {
                    if let Some(recorder) = ui_trace.as_mut() {
                        recorder.record_slash_command(queued.display_text.clone());
                    }
                    if handle_parsed_slash_command(
                        cmd,
                        terminal,
                        app,
                        logger,
                        args,
                        paused,
                        plugin_host,
                        ui_extensions,
                        runtime_factory,
                        lash_config,
                        runtime,
                        history,
                        turn_counter,
                        last_turn,
                        runtime_return_rx,
                        cancel_token,
                        active_stream_id,
                        provider,
                        current_model_variant,
                        current_execution_mode,
                        session_manager,
                        desired_dynamic,
                        pending_reconfigure,
                        model_catalog,
                        toolset_hash,
                        app_tx,
                        pending_clear_after_return,
                    )
                    .await?
                    {
                        return Ok(true);
                    }
                    return Ok(false);
                }
                record_queue_turn(ui_trace, &queued);
                app.queue_turn(queued.clone());
                return Ok(false);
            }
            if runtime.is_none() {
                tracing::debug!(
                    queued = queued.display_text,
                    app_running = app.running,
                    runtime_return_rx_present = runtime_return_rx.is_some(),
                    "queueing turn because runtime handoff is still in progress"
                );
                record_queue_turn(ui_trace, &queued);
                app.queue_turn(queued);
                return Ok(false);
            }

            let turn_input = make_turn_input(&queued);
            let current_dynamic_state = desired_dynamic.clone();
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
                &current_dynamic_state,
            )
            .await;
            *last_turn = Some(TurnReplayPayload {
                prepared_turn: queued,
                turn_input,
                execution_mode: current_execution_mode.clone(),
            });
        }
        // Up/Down: navigate suggestions when popup is visible
        KeyCode::Up if app.has_suggestions() => {
            if let Some(recorder) = ui_trace.as_mut() {
                recorder.record_suggestion_up();
            }
            let _ = apply_terminal_action(app, terminal, UiAction::SuggestionUp);
        }
        KeyCode::Down if app.has_suggestions() => {
            if let Some(recorder) = ui_trace.as_mut() {
                recorder.record_suggestion_down();
            }
            let _ = apply_terminal_action(app, terminal, UiAction::SuggestionDown);
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
                if app.running {
                    if slash_command_runs_out_of_band_while_running(&cmd) {
                        if handle_parsed_slash_command(
                            cmd,
                            terminal,
                            app,
                            logger,
                            args,
                            paused,
                            plugin_host,
                            ui_extensions,
                            runtime_factory,
                            lash_config,
                            runtime,
                            history,
                            turn_counter,
                            last_turn,
                            runtime_return_rx,
                            cancel_token,
                            active_stream_id,
                            provider,
                            current_model_variant,
                            current_execution_mode,
                            session_manager,
                            desired_dynamic,
                            pending_reconfigure,
                            model_catalog,
                            toolset_hash,
                            app_tx,
                            pending_clear_after_return,
                        )
                        .await?
                        {
                            return Ok(true);
                        }
                        return Ok(false);
                    }
                    let queued =
                        PreparedTurn::prepare(command_text.clone(), Vec::new(), &app.skills);
                    record_queue_turn(ui_trace, &queued);
                    app.queue_turn(queued);
                    return Ok(false);
                }
                if handle_parsed_slash_command(
                    cmd,
                    terminal,
                    app,
                    logger,
                    args,
                    paused,
                    plugin_host,
                    ui_extensions,
                    runtime_factory,
                    lash_config,
                    runtime,
                    history,
                    turn_counter,
                    last_turn,
                    runtime_return_rx,
                    cancel_token,
                    active_stream_id,
                    provider,
                    current_model_variant,
                    current_execution_mode,
                    session_manager,
                    desired_dynamic,
                    pending_reconfigure,
                    model_catalog,
                    toolset_hash,
                    app_tx,
                    pending_clear_after_return,
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
            let _ = apply_terminal_action(app, terminal, UiAction::SuggestionComplete);
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

            if app.running {
                if let Some(cmd) = parsed_command
                    && slash_command_runs_out_of_band_while_running(&cmd)
                {
                    if let Some(recorder) = ui_trace.as_mut() {
                        recorder.record_slash_command(queued.display_text.clone());
                    }
                    if handle_parsed_slash_command(
                        cmd,
                        terminal,
                        app,
                        logger,
                        args,
                        paused,
                        plugin_host,
                        ui_extensions,
                        runtime_factory,
                        lash_config,
                        runtime,
                        history,
                        turn_counter,
                        last_turn,
                        runtime_return_rx,
                        cancel_token,
                        active_stream_id,
                        provider,
                        current_model_variant,
                        current_execution_mode,
                        session_manager,
                        desired_dynamic,
                        pending_reconfigure,
                        model_catalog,
                        toolset_hash,
                        app_tx,
                        pending_clear_after_return,
                    )
                    .await?
                    {
                        return Ok(true);
                    }
                    return Ok(false);
                }
                if is_host_slash_command {
                    record_queue_turn(ui_trace, &queued);
                    app.queue_turn(queued.clone());
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
                let injection = lash::InjectedTurnInput {
                    id: None,
                    message: make_injected_plugin_message(&queued),
                };
                let Some(session) = runtime.as_ref() else {
                    push_system_message(
                        app,
                        "Current-turn injection is unavailable while the session is switching.",
                    );
                    app.restore_prepared_turn(queued);
                    return Ok(false);
                };
                match session.control().inject_turn_inputs(vec![injection]).await {
                    Ok(()) => {
                        record_queue_pending_steer(ui_trace, &queued);
                        app.queue_pending_steer(queued.clone());
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
                tracing::debug!(
                    queued = queued.display_text,
                    app_running = app.running,
                    runtime_return_rx_present = runtime_return_rx.is_some(),
                    "queueing turn because runtime handoff is still in progress"
                );
                record_queue_turn(ui_trace, &queued);
                app.queue_turn(queued);
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
                    terminal,
                    app,
                    logger,
                    args,
                    paused,
                    plugin_host,
                    ui_extensions,
                    runtime_factory,
                    lash_config,
                    runtime,
                    history,
                    turn_counter,
                    last_turn,
                    runtime_return_rx,
                    cancel_token,
                    active_stream_id,
                    provider,
                    current_model_variant,
                    current_execution_mode,
                    session_manager,
                    desired_dynamic,
                    pending_reconfigure,
                    model_catalog,
                    toolset_hash,
                    app_tx,
                    pending_clear_after_return,
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

            // Regular user message — send to the active session
            if let Err(e) =
                apply_pending_reconfigure(desired_dynamic, pending_reconfigure, runtime).await
            {
                push_system_message(
                    app,
                    format!(
                        "Pending runtime reconfigure failed; message not sent: {}",
                        e
                    ),
                );
                return Ok(false);
            }
            *toolset_hash = hash12(
                &serde_json::to_vec(
                    &desired_dynamic
                        .tools
                        .values()
                        .map(|spec| spec.definition.clone())
                        .collect::<Vec<_>>(),
                )
                .unwrap_or_else(|_| b"[]".to_vec()),
            );
            let turn_input = make_turn_input(&queued);
            let current_dynamic_state = desired_dynamic.clone();
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
                &current_dynamic_state,
            )
            .await;
            *last_turn = Some(TurnReplayPayload {
                prepared_turn: queued,
                turn_input,
                execution_mode: current_execution_mode.clone(),
            });
        }
        KeyCode::Backspace => {
            if let Some(recorder) = ui_trace.as_mut() {
                recorder.record_input_backspace();
            }
            let _ = apply_terminal_action(app, terminal, UiAction::InputBackspace);
        }
        KeyCode::Delete => {
            if let Some(recorder) = ui_trace.as_mut() {
                recorder.record_input_delete();
            }
            let _ = apply_terminal_action(app, terminal, UiAction::InputDelete);
        }
        KeyCode::Left if key.modifiers.contains(KeyModifiers::CONTROL) => {
            if let Some(recorder) = ui_trace.as_mut() {
                recorder.record_move_cursor_left();
            }
            let _ = apply_terminal_action(app, terminal, UiAction::MoveCursorWordLeft);
        }
        KeyCode::Right if key.modifiers.contains(KeyModifiers::CONTROL) => {
            if let Some(recorder) = ui_trace.as_mut() {
                recorder.record_move_cursor_right();
            }
            let _ = apply_terminal_action(app, terminal, UiAction::MoveCursorWordRight);
        }
        KeyCode::Left => {
            if let Some(recorder) = ui_trace.as_mut() {
                recorder.record_move_cursor_left();
            }
            let _ = apply_terminal_action(app, terminal, UiAction::MoveCursorLeft);
        }
        KeyCode::Right => {
            if let Some(recorder) = ui_trace.as_mut() {
                recorder.record_move_cursor_right();
            }
            let _ = apply_terminal_action(app, terminal, UiAction::MoveCursorRight);
        }
        KeyCode::Home => {
            if let Some(recorder) = ui_trace.as_mut() {
                recorder.record_move_cursor_home();
            }
            let _ = apply_terminal_action(app, terminal, UiAction::MoveCursorHome);
        }
        KeyCode::End => {
            if let Some(recorder) = ui_trace.as_mut() {
                recorder.record_move_cursor_end();
            }
            let _ = apply_terminal_action(app, terminal, UiAction::MoveCursorEnd);
        }
        KeyCode::Up => {
            if let Some(recorder) = ui_trace.as_mut() {
                recorder.record_history_up();
            }
            let _ = apply_terminal_action(app, terminal, UiAction::HistoryUp);
        }
        KeyCode::Down => {
            if let Some(recorder) = ui_trace.as_mut() {
                recorder.record_history_down();
            }
            let _ = apply_terminal_action(app, terminal, UiAction::HistoryDown);
        }
        KeyCode::Char(c) => {
            if let Some(recorder) = ui_trace.as_mut() {
                recorder.record_input_insert_text(c.to_string());
            }
            let _ = apply_terminal_action(app, terminal, UiAction::InputInsertText(c.to_string()));
        }
        _ => {}
    }
    Ok(false)
}
