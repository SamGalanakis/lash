//! Keyboard bindings that vary by terminal/environment, plus the
//! authoritative shortcut-help rows derived from them.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use lash_tui_extensions::TuiExtensions;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum QueuedTurnEditBinding {
    AltUp,
    ShiftLeft,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CopyBinding {
    CtrlShiftC,
    CtrlY,
}

impl CopyBinding {
    pub(crate) fn display(self) -> &'static str {
        match self {
            Self::CtrlShiftC => "Ctrl+Shift+C",
            Self::CtrlY => "Ctrl+Y",
        }
    }

    pub(crate) fn matches(self, key: KeyEvent) -> bool {
        match self {
            Self::CtrlShiftC => {
                key.modifiers.contains(KeyModifiers::CONTROL)
                    && key.modifiers.contains(KeyModifiers::SHIFT)
                    && matches!(key.code, KeyCode::Char(c) if c.eq_ignore_ascii_case(&'c'))
            }
            Self::CtrlY => {
                key.modifiers.contains(KeyModifiers::CONTROL)
                    && matches!(key.code, KeyCode::Char(c) if c.eq_ignore_ascii_case(&'y'))
            }
        }
    }
}

impl QueuedTurnEditBinding {
    pub(crate) fn display(self) -> &'static str {
        match self {
            Self::AltUp => "Alt+Up",
            Self::ShiftLeft => "Shift+Left",
        }
    }

    pub(crate) fn matches(self, key: KeyEvent) -> bool {
        match self {
            Self::AltUp => key.modifiers.contains(KeyModifiers::ALT) && key.code == KeyCode::Up,
            Self::ShiftLeft => {
                key.modifiers.contains(KeyModifiers::SHIFT) && key.code == KeyCode::Left
            }
        }
    }
}

fn queued_turn_edit_binding_from_hints(
    term_program: Option<&str>,
    inside_tmux: bool,
    inside_vscode: bool,
) -> QueuedTurnEditBinding {
    if inside_tmux {
        return QueuedTurnEditBinding::ShiftLeft;
    }

    match term_program.map(|value| value.to_ascii_lowercase()) {
        Some(value)
            if matches!(
                value.as_str(),
                "apple_terminal" | "warpterminal" | "warp" | "vscode"
            ) =>
        {
            QueuedTurnEditBinding::ShiftLeft
        }
        _ if inside_vscode => QueuedTurnEditBinding::ShiftLeft,
        _ => QueuedTurnEditBinding::AltUp,
    }
}

pub(crate) fn queued_turn_edit_binding() -> QueuedTurnEditBinding {
    queued_turn_edit_binding_from_hints(
        std::env::var("TERM_PROGRAM").ok().as_deref(),
        std::env::var_os("TMUX").is_some(),
        std::env::var_os("VSCODE_INJECTION").is_some(),
    )
}

pub(crate) fn copy_binding() -> CopyBinding {
    copy_binding_from_env(std::env::var("LASH_COPY_BINDING").ok().as_deref())
}

pub(crate) fn copy_binding_from_env(value: Option<&str>) -> CopyBinding {
    match value
        .map(|value| value.trim().to_ascii_lowercase())
        .as_deref()
    {
        Some("ctrl-shift-c") | Some("ctrl_shift_c") => CopyBinding::CtrlShiftC,
        Some("ctrl-y") | Some("ctrl_y") => CopyBinding::CtrlY,
        Some("ctrl-c") | Some("ctrl_c") | None | Some("") => CopyBinding::CtrlShiftC,
        Some(_) => CopyBinding::CtrlShiftC,
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ShortcutHelpRow {
    pub(crate) keys: String,
    pub(crate) description: String,
}

impl ShortcutHelpRow {
    fn new(keys: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            keys: keys.into(),
            description: description.into(),
        }
    }
}

pub(crate) fn shortcut_help_rows(
    ui_extensions: &TuiExtensions,
    spaced_history_arrows: bool,
) -> Vec<ShortcutHelpRow> {
    let history_arrows = if spaced_history_arrows {
        "Up / Down"
    } else {
        "Up/Down"
    };

    let mut lines = vec![
        ShortcutHelpRow::new(
            "Ctrl+C",
            "Close popup/overlay, cancel turn, clear draft, or quit",
        ),
        ShortcutHelpRow::new("Esc", "Close overlay or cancel running turn"),
        ShortcutHelpRow::new("Enter", "Submit; early-inject while running"),
        ShortcutHelpRow::new("Tab", "Queue for next turn; submit draft when idle"),
        ShortcutHelpRow::new("PgUp / PgDn", "Scroll history or document overlay"),
        ShortcutHelpRow::new("Ctrl+U", "Delete draft text to line start"),
        ShortcutHelpRow::new("Ctrl+K", "Delete draft text to line end"),
        ShortcutHelpRow::new("Up (empty draft)", "Edit last queued turn"),
        ShortcutHelpRow::new(
            queued_turn_edit_binding().display(),
            "Edit last queued turn",
        ),
        ShortcutHelpRow::new("Shift+Enter", "Insert newline"),
        ShortcutHelpRow::new("Ctrl+V", "Paste image as inline [Image #n]"),
        ShortcutHelpRow::new("Ctrl+Shift+V", "Paste text only"),
        ShortcutHelpRow::new(copy_binding().display(), "Copy selection or last response"),
    ];

    for shortcut in ui_extensions.shortcut_specs() {
        lines.push(ShortcutHelpRow::new(
            shortcut.chord.display(),
            shortcut.description,
        ));
    }

    lines.extend([
        ShortcutHelpRow::new("Ctrl+O", "Cycle tool expansion"),
        ShortcutHelpRow::new("Alt+O", "Full expansion"),
        ShortcutHelpRow::new(history_arrows, "Input history"),
    ]);

    lines
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn queued_turn_edit_binding_defaults_to_alt_up() {
        assert_eq!(
            queued_turn_edit_binding_from_hints(None, false, false),
            QueuedTurnEditBinding::AltUp
        );
        assert_eq!(
            queued_turn_edit_binding_from_hints(Some("ghostty"), false, false),
            QueuedTurnEditBinding::AltUp
        );
    }

    #[test]
    fn queued_turn_edit_binding_falls_back_when_alt_up_is_unreliable() {
        for term_program in ["Apple_Terminal", "WarpTerminal", "warp", "vscode"] {
            assert_eq!(
                queued_turn_edit_binding_from_hints(Some(term_program), false, false),
                QueuedTurnEditBinding::ShiftLeft
            );
        }
        assert_eq!(
            queued_turn_edit_binding_from_hints(None, true, false),
            QueuedTurnEditBinding::ShiftLeft
        );
        assert_eq!(
            queued_turn_edit_binding_from_hints(None, false, true),
            QueuedTurnEditBinding::ShiftLeft
        );
    }

    #[test]
    fn shortcut_rows_have_no_duplicate_default_keys() {
        let rows = shortcut_help_rows(&TuiExtensions::default(), true);
        let mut seen = HashSet::new();
        let mut duplicates = Vec::new();
        for row in &rows {
            if !seen.insert(row.keys.as_str()) {
                duplicates.push(row.keys.clone());
            }
        }

        assert!(duplicates.is_empty(), "duplicate shortcuts: {duplicates:?}");
        assert!(
            rows.iter()
                .any(|row| row.keys == "Ctrl+C" && row.description.contains("cancel"))
        );
        assert!(
            rows.iter()
                .any(|row| row.keys == "Ctrl+Shift+C" && row.description.contains("Copy"))
        );
        assert!(
            rows.iter()
                .any(|row| row.keys == "Ctrl+U" && row.description.contains("line start"))
        );
        assert!(
            rows.iter()
                .any(|row| row.keys == "Ctrl+K" && row.description.contains("line end"))
        );
        assert!(
            rows.iter()
                .any(|row| row.keys == "PgUp / PgDn" && row.description.contains("document"))
        );
        assert!(
            !rows
                .iter()
                .any(|row| row.keys.contains("Ctrl+U") && row.description.contains("Scroll"))
        );
    }
}
