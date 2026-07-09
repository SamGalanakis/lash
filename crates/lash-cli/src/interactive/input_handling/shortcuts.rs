//! Global shortcuts that fire regardless of the active mode: selection
//! copy, Ctrl+C dismiss/cancel/quit, expand toggles, undo/redo, clipboard
//! pastes, queued-turn edit, and Esc.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use lash::provider::ProviderHandle;
use tokio::task;

use crate::app::App;
use crate::command;
use crate::config::ThemeName;
use crate::event::AppEvent;
use crate::input_items::insert_inline_marker;
use crate::model_selection::provider_display_label;
use crate::overlay::{CommandPaletteAction, CommandPaletteItem};

use crate::interactive::helpers::{
    is_copy_shortcut, queued_turn_edit_matches, should_preserve_selection_for_key,
};
use crate::interactive::runtime::{copy_active_selection, copy_selected_text_or_last_response};

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
        copy_active_selection(app, terminal.size().ok());
        return Ok(Some(false));
    }

    if is_command_palette_shortcut(key) {
        let items = command_palette_items(app, ctx.provider);
        app.show_command_palette(items);
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

fn is_command_palette_shortcut(key: KeyEvent) -> bool {
    if key.modifiers.contains(KeyModifiers::SHIFT) {
        return false;
    }
    if key.modifiers.contains(KeyModifiers::CONTROL)
        && matches!(key.code, KeyCode::Char(c) if c.eq_ignore_ascii_case(&'p'))
    {
        return true;
    }

    // Some terminal stacks deliver Ctrl+P as the raw DLE byte rather than
    // `Char('p')` with a CONTROL modifier. Treat that byte as the same chord.
    key.code == KeyCode::Char('\u{10}')
        && !key
            .modifiers
            .intersects(KeyModifiers::SHIFT | KeyModifiers::ALT | KeyModifiers::SUPER)
}

pub(crate) fn command_palette_items(
    app: &App,
    provider: &ProviderHandle,
) -> Vec<CommandPaletteItem> {
    let mut items = Vec::new();

    for theme in ThemeName::all() {
        items.push(
            CommandPaletteItem::new(
                "Settings",
                format!("Theme: {}", theme.label()),
                theme.description(),
                CommandPaletteAction::Theme(theme),
            )
            .footer(theme.as_str())
            .current(theme == crate::theme::active_theme()),
        );
    }

    items.push(
        CommandPaletteItem::new(
            "Settings",
            "Model",
            format!("Current: {}", app.model),
            CommandPaletteAction::InsertDraft("/model ".to_string()),
        )
        .footer("/model <name>")
        .current(true),
    );

    let supported_efforts = crate::model_selection::supported_efforts(provider, &app.model);
    if supported_efforts.is_empty() {
        items.push(
            CommandPaletteItem::new(
                "Settings",
                "Variant",
                "This model does not expose configurable variants.",
                CommandPaletteAction::Builtin(command::Command::Variant(None)),
            )
            .footer("/variant")
            .current(true),
        );
    } else {
        items.push(
            CommandPaletteItem::new(
                "Settings",
                "Variant: Default",
                "Use the provider-recommended default variant.",
                CommandPaletteAction::Builtin(command::Command::Variant(Some(
                    "default".to_string(),
                ))),
            )
            .footer("/variant default")
            .current(app.model_variant.is_none()),
        );
        for variant in &supported_efforts {
            items.push(
                CommandPaletteItem::new(
                    "Settings",
                    format!("Variant: {variant}"),
                    format!("Set `{}` to `{variant}`.", app.model),
                    CommandPaletteAction::Builtin(command::Command::Variant(Some(variant.clone()))),
                )
                .footer(format!("/variant {variant}"))
                .current(app.model_variant.as_deref() == Some(variant.as_str())),
            );
        }
    }

    items.push(
        CommandPaletteItem::new(
            "Settings",
            "Provider",
            format!("Current: {}", provider_display_label(provider)),
            CommandPaletteAction::Builtin(command::Command::ChangeProvider),
        )
        .footer("/provider"),
    );
    items.push(
        CommandPaletteItem::new(
            "Settings",
            "Execution Mode",
            format!(
                "Current: {}. Locked for this session.",
                app.execution_mode_label
            ),
            CommandPaletteAction::Builtin(command::Command::Mode(None)),
        )
        .footer("/mode"),
    );
    items.push(
        CommandPaletteItem::new(
            "Settings",
            "Logout Provider",
            "Remove stored credentials for the active provider.",
            CommandPaletteAction::Builtin(command::Command::Logout),
        )
        .footer("/logout"),
    );

    items.extend([
        CommandPaletteItem::new(
            "Session",
            "New Session",
            "Reset conversation and start fresh.",
            CommandPaletteAction::Builtin(command::Command::Clear),
        )
        .footer("/clear"),
        CommandPaletteItem::new(
            "Session",
            "Resume Session",
            "Search previous sessions.",
            CommandPaletteAction::Builtin(command::Command::Resume(None)),
        )
        .footer("/resume"),
        CommandPaletteItem::new(
            "Session",
            "Browse Tree",
            "Browse and switch conversation branches.",
            CommandPaletteAction::Builtin(command::Command::Tree),
        )
        .footer("/tree"),
        CommandPaletteItem::new(
            "Session",
            "Fork Session",
            "Open this session in a new terminal.",
            CommandPaletteAction::Builtin(command::Command::Fork),
        )
        .footer("/fork"),
        CommandPaletteItem::new(
            "Session",
            "Compact Context",
            "Open a compaction frame seeded by a summary.",
            CommandPaletteAction::Builtin(command::Command::Compact(None)),
        )
        .footer("/compact"),
        CommandPaletteItem::new(
            "Session",
            "Retry Last Turn",
            "Replay the previous turn payload.",
            CommandPaletteAction::Builtin(command::Command::Retry),
        )
        .footer("/retry"),
    ]);

    items.extend([
        CommandPaletteItem::new(
            "Help",
            "Runtime Info",
            "Show session, provider, model, path, and tool info.",
            CommandPaletteAction::Builtin(command::Command::Info),
        )
        .footer("/info"),
        CommandPaletteItem::new(
            "Help",
            "Keyboard Controls",
            "Show keyboard shortcuts.",
            CommandPaletteAction::Builtin(command::Command::Controls),
        )
        .footer("/controls"),
        CommandPaletteItem::new(
            "Help",
            "Commands Help",
            "Show commands and loaded skills.",
            CommandPaletteAction::Builtin(command::Command::Help),
        )
        .footer("/help"),
        CommandPaletteItem::new(
            "Help",
            "Skills",
            "Browse loaded skills.",
            CommandPaletteAction::Builtin(command::Command::Skills),
        )
        .footer("/skills"),
        CommandPaletteItem::new(
            "Help",
            "Version",
            "Show lash-cli and lash-sansio versions.",
            CommandPaletteAction::Builtin(command::Command::Version),
        )
        .footer("/version"),
        CommandPaletteItem::new(
            "Help",
            "Exit",
            "Quit Lash.",
            CommandPaletteAction::Builtin(command::Command::Exit),
        )
        .footer("/exit"),
    ]);

    items
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

#[cfg(test)]
mod tests {
    use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyEventState, KeyModifiers};

    use super::is_command_palette_shortcut;

    fn key(code: KeyCode, modifiers: KeyModifiers) -> KeyEvent {
        KeyEvent {
            code,
            modifiers,
            kind: KeyEventKind::Press,
            state: KeyEventState::NONE,
        }
    }

    #[test]
    fn command_palette_shortcut_accepts_modified_ctrl_p() {
        assert!(is_command_palette_shortcut(key(
            KeyCode::Char('p'),
            KeyModifiers::CONTROL
        )));
    }

    #[test]
    fn command_palette_shortcut_accepts_raw_ctrl_p_byte() {
        assert!(is_command_palette_shortcut(key(
            KeyCode::Char('\u{10}'),
            KeyModifiers::NONE
        )));
    }

    #[test]
    fn command_palette_shortcut_rejects_plain_p() {
        assert!(!is_command_palette_shortcut(key(
            KeyCode::Char('p'),
            KeyModifiers::NONE
        )));
    }
}
