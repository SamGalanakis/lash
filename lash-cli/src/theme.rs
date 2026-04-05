use ratatui::style::{Color, Modifier, Style};

// ── Formwork: warm olive-tinted blacks ──────────────────────────────
pub const FORM: Color = Color::Rgb(14, 13, 11);
pub const FORM_DEEP: Color = Color::Rgb(8, 8, 7);
pub const FORM_RAISED: Color = Color::Rgb(20, 20, 18);

// ── Ash: structural greys ──────────────────────────────────────────
pub const ASH: Color = Color::Rgb(42, 42, 40);
pub const ASH_LIGHT: Color = Color::Rgb(58, 58, 52);
pub const ASH_MID: Color = Color::Rgb(74, 74, 68);
pub const ASH_TEXT: Color = Color::Rgb(90, 90, 80);

// ── Chalk: text hierarchy ──────────────────────────────────────────
pub const CHALK_DIM: Color = Color::Rgb(122, 122, 112);
pub const CHALK_MID: Color = Color::Rgb(200, 196, 184);
pub const CHALK: Color = Color::Rgb(232, 228, 208);

// ── Accent colors ──────────────────────────────────────────────────
pub const SODIUM: Color = Color::Rgb(232, 163, 60);
pub const LICHEN: Color = Color::Rgb(138, 158, 108);
pub const ERROR: Color = Color::Rgb(204, 68, 68);
pub const PATCH_FRAME: Color = Color::Rgb(128, 94, 38);
pub const PATCH_ADD: Color = Color::Rgb(126, 164, 92);
pub const PATCH_HUNK: Color = Color::Rgb(154, 142, 106);
pub const PATCH_ADD_GUTTER_BG: Color = Color::Rgb(39, 55, 31);
pub const PATCH_ADD_LINE_BG: Color = Color::Rgb(25, 37, 23);
pub const PATCH_REMOVE_GUTTER_BG: Color = Color::Rgb(78, 36, 32);
pub const PATCH_REMOVE_LINE_BG: Color = Color::Rgb(46, 24, 21);
pub const PATCH_CONTEXT_GUTTER_BG: Color = Color::Rgb(26, 25, 22);
pub const PATCH_CONTEXT_LINE_BG: Color = Color::Rgb(19, 18, 16);
pub const PATCH_HUNK_GUTTER_BG: Color = Color::Rgb(41, 34, 22);
pub const PATCH_HUNK_LINE_BG: Color = Color::Rgb(28, 24, 18);

// ── Character constants ────────────────────────────────────────────
pub const PROMPT_CHAR: &str = "❯";
pub const STATUS_SEP: &str = " · ";

// ── Style helpers ──────────────────────────────────────────────────

/// Sodium bold `❯` prompt character
pub fn prompt() -> Style {
    Style::default().fg(SODIUM).add_modifier(Modifier::BOLD)
}

/// User-typed text after the prompt
pub fn user_input() -> Style {
    Style::default().fg(CHALK_MID)
}

/// Sigil for an inline resolved skill/command token inside user text.
pub fn resolved_token_sigil() -> Style {
    Style::default()
        .fg(SODIUM)
        .bg(FORM_RAISED)
        .add_modifier(Modifier::BOLD)
}

/// Body for an inline resolved skill/command token inside user text.
pub fn resolved_token() -> Style {
    Style::default().fg(CHALK).bg(FORM_RAISED)
}

/// Inline image markers inside the draft editor.
pub fn image_marker() -> Style {
    Style::default().fg(SODIUM).add_modifier(Modifier::BOLD)
}

/// Assistant text marker (readable warm grey)
pub fn assistant_bar() -> Style {
    Style::default().fg(ASH_TEXT)
}

/// Assistant prose in the history
pub fn assistant_text() -> Style {
    Style::default().fg(CHALK)
}

/// System/progress messages
pub fn system_output() -> Style {
    Style::default().fg(ASH_TEXT)
}

/// Error text
pub fn error() -> Style {
    Style::default().fg(ERROR)
}

/// Code block content
pub fn code_content() -> Style {
    Style::default().fg(CHALK_DIM)
}

/// Code block border characters (│, └───, etc.)
pub fn code_chrome() -> Style {
    Style::default().fg(ASH)
}

/// Expanded code block left border — dimmed sodium "scribe line"
pub fn code_scribe() -> Style {
    Style::default().fg(Color::Rgb(100, 72, 28))
}

/// Exploration lane chrome — softer amber than code/edit rails.
pub fn explore_marker() -> Style {
    Style::default()
        .fg(Color::Rgb(122, 94, 48))
        .add_modifier(Modifier::BOLD)
}

/// Exploration section label.
pub fn explore_label() -> Style {
    Style::default().fg(CHALK_DIM).add_modifier(Modifier::BOLD)
}

/// Code block header (▶ code, ▼ code)
pub fn code_header() -> Style {
    Style::default().fg(ASH_MID)
}

/// Tool call success line
pub fn tool_success() -> Style {
    Style::default().fg(ASH_TEXT)
}

/// Tool call failure line
pub fn tool_failure() -> Style {
    Style::default().fg(ERROR)
}

/// Help bar key labels
pub fn help_key() -> Style {
    Style::default().fg(SODIUM).add_modifier(Modifier::BOLD)
}

/// Help bar descriptions
pub fn help_desc() -> Style {
    Style::default().fg(ASH_MID)
}

/// Input area border
pub fn input_border() -> Style {
    Style::default().fg(ASH)
}

/// "lash" title in status bar
pub fn app_title() -> Style {
    Style::default().fg(SODIUM).add_modifier(Modifier::BOLD)
}

/// Model name in status bar
pub fn model_name() -> Style {
    Style::default().fg(CHALK_DIM)
}

/// Status bar separator ( · )
pub fn status_separator() -> Style {
    Style::default().fg(ASH_MID)
}

/// Status bar and help bar background
pub fn bar_bg() -> Style {
    Style::default().bg(FORM_RAISED)
}

/// Bottom turn-status strip background.
pub fn turn_status_bar() -> Style {
    Style::default().bg(FORM_RAISED)
}

/// Animated LASH wordmark letters in the turn-status strip.
pub fn turn_status_brand() -> Style {
    Style::default().fg(CHALK).add_modifier(Modifier::BOLD)
}

/// Animated slash accent in the turn-status strip.
pub fn turn_status_slash() -> Style {
    Style::default().fg(SODIUM).add_modifier(Modifier::BOLD)
}

/// Primary state label in the bottom turn-status strip.
pub fn turn_status_state() -> Style {
    Style::default().fg(CHALK).add_modifier(Modifier::BOLD)
}

/// Elapsed turn time in the bottom turn-status strip.
pub fn turn_status_elapsed() -> Style {
    Style::default().fg(CHALK_DIM)
}

/// History area background
pub fn history_bg() -> Style {
    Style::default().bg(FORM)
}

/// System message text (e.g. /help output)
pub fn system_message() -> Style {
    Style::default().fg(ASH_TEXT)
}

/// Markdown heading (bold + sodium)
pub fn heading() -> Style {
    Style::default().fg(SODIUM).add_modifier(Modifier::BOLD)
}

/// Markdown inline code
pub fn inline_code() -> Style {
    Style::default().fg(SODIUM)
}

/// Patch rail and inset chrome
pub fn patch_frame() -> Style {
    Style::default().fg(PATCH_FRAME)
}

/// Patch status label
pub fn patch_label() -> Style {
    Style::default().fg(SODIUM).add_modifier(Modifier::BOLD)
}

/// Added-line or positive count styling
pub fn patch_add() -> Style {
    Style::default().fg(PATCH_ADD)
}

/// Removed-line or negative count styling
pub fn patch_remove() -> Style {
    Style::default().fg(ERROR)
}

/// Diff gutter for inserted lines.
pub fn patch_diff_add_gutter() -> Style {
    Style::default()
        .fg(LICHEN)
        .bg(PATCH_ADD_GUTTER_BG)
        .add_modifier(Modifier::BOLD)
}

/// Diff body for inserted lines.
pub fn patch_diff_add_line() -> Style {
    Style::default().fg(CHALK).bg(PATCH_ADD_LINE_BG)
}

/// Diff gutter for removed lines.
pub fn patch_diff_remove_gutter() -> Style {
    Style::default()
        .fg(ERROR)
        .bg(PATCH_REMOVE_GUTTER_BG)
        .add_modifier(Modifier::BOLD)
}

/// Diff body for removed lines.
pub fn patch_diff_remove_line() -> Style {
    Style::default().fg(CHALK).bg(PATCH_REMOVE_LINE_BG)
}

/// Diff gutter for context rows.
pub fn patch_diff_context_gutter() -> Style {
    Style::default().fg(ASH_TEXT).bg(PATCH_CONTEXT_GUTTER_BG)
}

/// Diff body for context rows.
pub fn patch_diff_context_line() -> Style {
    Style::default().fg(CHALK_DIM).bg(PATCH_CONTEXT_LINE_BG)
}

/// Diff gutter for hunk headers.
pub fn patch_diff_hunk_gutter() -> Style {
    Style::default().fg(PATCH_HUNK).bg(PATCH_HUNK_GUTTER_BG)
}

/// Diff body for hunk headers.
pub fn patch_diff_hunk_line() -> Style {
    Style::default()
        .fg(PATCH_HUNK)
        .bg(PATCH_HUNK_LINE_BG)
        .add_modifier(Modifier::BOLD)
}

/// Diff body for non-standard meta lines such as truncation notices.
pub fn patch_diff_meta_line() -> Style {
    Style::default().fg(ASH_TEXT).add_modifier(Modifier::DIM)
}

/// Delegate rail chrome — muted lichen for tree connectors.
pub fn delegate_chrome() -> Style {
    Style::default().fg(Color::Rgb(98, 118, 72))
}

/// Delegate rail marker — bold lichen for the ◆ diamond.
pub fn delegate_marker() -> Style {
    Style::default().fg(LICHEN).add_modifier(Modifier::BOLD)
}

/// Delegate child tool summary — dimmed chalk for nested tool names.
pub fn delegate_child() -> Style {
    Style::default().fg(CHALK_DIM)
}

/// Completed plan-step marker.
pub fn plan_done_marker() -> Style {
    Style::default().fg(LICHEN).add_modifier(Modifier::BOLD)
}

/// In-progress plan-step marker.
pub fn plan_active_marker() -> Style {
    Style::default().fg(SODIUM).add_modifier(Modifier::BOLD)
}

/// Pending plan-step marker.
pub fn plan_pending_marker() -> Style {
    Style::default().fg(CHALK_DIM).add_modifier(Modifier::BOLD)
}

/// Turn separator line between conversation turns.
pub fn turn_separator() -> Style {
    Style::default().fg(ASH)
}

/// Edit lane chrome — lichen green for file modifications.
pub fn edit_lane() -> Style {
    Style::default().fg(LICHEN)
}

/// Edit lane chrome — bold lichen for summary prefix.
pub fn edit_lane_bold() -> Style {
    Style::default().fg(LICHEN).add_modifier(Modifier::BOLD)
}

/// Shell command lane chrome — muted chalk for mundane operations.
pub fn shell_lane() -> Style {
    Style::default().fg(CHALK_DIM)
}

/// Error block border.
pub fn error_border() -> Style {
    Style::default().fg(ERROR)
}

/// Error block title.
pub fn error_title() -> Style {
    Style::default().fg(ERROR).add_modifier(Modifier::BOLD)
}
