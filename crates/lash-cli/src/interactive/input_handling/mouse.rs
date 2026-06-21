//! Mouse handling (selection, drag, scroll wheels) and the shared viewport
//! scroll math used by both mouse and keyboard scrolling.

use crossterm::event::{Event as TermEvent, KeyCode, KeyEvent, MouseEvent};
use lash::LashSession;
use lash_tui::{Terminal, normalize_event};
use lash_tui_extensions::{TuiExtensions, TuiSurfaceSlot};

use crate::app::App;
use crate::render;
use crate::ui_trace::UiTraceRecorder;

use super::handle_surface_input;

/// Viewport-derived measurements the scroll handlers need; computed from
/// the current terminal size against the live layout.
pub(super) struct ViewportMetrics {
    pub(super) width: usize,
    pub(super) history_height: usize,
    pub(super) prompt_max_scroll: usize,
}

pub(super) fn viewport_metrics(app: &App, terminal: &Terminal) -> ViewportMetrics {
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

/// A scroll keypress, resolved to a direction and amount. PageUp/PageDown
/// scroll the focused document/prompt overlay first, then history when no
/// modal owns focus. Bare Up/Down scroll a review-only prompt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ScrollInputAction {
    PromptUp(usize),
    PromptDown(usize),
    HistoryUp(usize),
    HistoryDown(usize),
}

pub(super) fn classify_always_on_scroll_key(
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

pub(super) fn apply_scroll_input_action(
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

pub(in crate::interactive) fn handle_mouse_event(
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
    if matches!(mouse.kind, MouseEventKind::Up(MouseButton::Left))
        && app.take_suppressed_mouse_up_after_selection_clear()
    {
        tracing::debug!(
            row = mouse.row,
            column = mouse.column,
            "mouse up suppressed after clearing text selection"
        );
        return Ok(());
    }
    let ha = app.history_area;
    let (term_width, term_height) = terminal.size()?;
    let prompt_area = render::input_area(app, term_width, term_height);
    let input_area = render::input_content_area(app, term_width, term_height);
    let body_area = lash_tui::Rect::new(0, 0, term_width, term_height);
    if app.has_document()
        && let Some(document) = app.document_state()
        && let Some(document_area) = render::document_overlay_content_area(body_area)
    {
        let document_scroll = document.scroll_offset;
        let document_max_scroll = render::document_max_scroll(
            document,
            document_area.width as usize,
            document_area.height as usize,
        );
        let in_document = document_area.width > 0
            && document_area.height > 0
            && mouse.row >= document_area.y
            && mouse.row < document_area.y + document_area.height
            && mouse.column >= document_area.x
            && mouse.column < document_area.x + document_area.width;
        match mouse.kind {
            MouseEventKind::ScrollUp => {
                app.document_scroll_up(3);
                app.dirty = true;
                return Ok(());
            }
            MouseEventKind::ScrollDown => {
                app.document_scroll_down(3, document_max_scroll);
                app.dirty = true;
                return Ok(());
            }
            MouseEventKind::Down(MouseButton::Left) if in_document => {
                app.clear_input_selection();
                let vrow = document_scroll + (mouse.row - document_area.y) as usize;
                app.selection.anchor = (mouse.column, vrow);
                app.selection.end = (mouse.column, vrow);
                app.selection.active = true;
                app.selection.visible = false;
                app.dirty = true;
                return Ok(());
            }
            MouseEventKind::Drag(MouseButton::Left) if app.selection.active => {
                let col = mouse.column.clamp(
                    document_area.x,
                    document_area.x + document_area.width.saturating_sub(1),
                );
                let row = mouse.row.clamp(
                    document_area.y,
                    document_area.y + document_area.height.saturating_sub(1),
                );
                let vrow = document_scroll + (row - document_area.y) as usize;
                app.selection.end = (col, vrow);
                app.selection.visible = true;
                app.dirty = true;
                return Ok(());
            }
            MouseEventKind::Up(MouseButton::Left) if app.selection.active => {
                app.selection.active = false;
                app.dirty = true;
                return Ok(());
            }
            _ => {}
        }
    }
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

    if matches!(mouse.kind, MouseEventKind::Down(MouseButton::Left))
        && app.has_visible_text_selection()
        && !in_history
        && !in_input
    {
        app.clear_text_selection();
        app.suppress_mouse_up_after_selection_clear();
        tracing::debug!(
            row = mouse.row,
            column = mouse.column,
            "cleared text selection and suppressed click target"
        );
        return Ok(());
    }

    let selection_drag_active = app.selection.active || app.input_selection_active();
    if !selection_drag_active
        && let Some(event) = normalize_event(&TermEvent::Mouse(mouse))
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
