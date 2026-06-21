//! Global shortcuts that fire regardless of the active mode: selection
//! copy, Ctrl+C dismiss/cancel/quit, expand toggles, undo/redo, clipboard
//! pastes, queued-turn edit, and Esc.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use tokio::task;

use crate::app::App;
use crate::event::AppEvent;
use crate::input_items::insert_inline_marker;

use crate::interactive::helpers::{
    is_copy_shortcut, queued_turn_edit_matches, should_preserve_selection_for_key,
};
use crate::interactive::runtime::copy_selected_text_or_last_response;

use super::turns::restore_last_durable_full_turn;
use super::{SessionCtx, dismiss_active_modal};

/// Returns `Some(ret)` when the key was fully handled here, where `ret`
/// is the `dispatch_key_event` return value; `None` to fall through.
pub(super) async fn handle_global_shortcut_key(
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
    if key.code == KeyCode::Esc && app.has_text_selection() {
        tracing::debug!("clearing text selection on Escape");
        app.clear_text_selection();
        return Ok(Some(false));
    }

    // Clear any active history selection on plain keypress.
    if app.has_visible_output_selection() && !should_preserve_selection_for_key(key) {
        tracing::debug!("clearing selection on plain keypress");
        app.clear_selection();
    }

    // Active selection copy should win before generic Ctrl+C handling.
    if app.has_text_selection() && copy_shortcut {
        tracing::debug!("selection copy took precedence over generic key handling");
        copy_selected_text_or_last_response(app, terminal.size().ok());
        app.clear_text_selection();
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
            paste_clipboard_image(app, app_tx, img_data);
        }
        return Ok(Some(false));
    }

    // Escape dismisses the active modal (precedence resolved once via
    // `dismiss_active_modal`), otherwise interrupts a running turn.
    if key.code == KeyCode::Esc {
        if !dismiss_active_modal(app, ui_trace) {
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
        return Ok(Some(false));
    }

    Ok(None)
}

/// Insert an `[Image #n]` marker for a pasted clipboard image and encode it
/// to PNG off the UI thread.
fn paste_clipboard_image(
    app: &mut App,
    app_tx: &crate::event::AppEventTx,
    img_data: arboard::ImageData<'_>,
) {
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
        .unwrap_or_else(|err| Err(anyhow::anyhow!("Failed to process pasted image: {err}")));
        let _ = app_tx.send(AppEvent::ClipboardImageReady { id: image_id, png });
    });
}
