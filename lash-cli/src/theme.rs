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

/// Assistant prose in the history
pub fn assistant_text() -> Style {
    Style::default().fg(CHALK)
}

/// System/progress messages
pub fn system_output() -> Style {
    Style::default().fg(ASH_TEXT)
}

/// Tool call success (lichen green)
pub fn success() -> Style {
    Style::default().fg(LICHEN)
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

/// Code block header (▶ python, ▼ python)
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

/// "waiting for agent..." text
pub fn waiting() -> Style {
    Style::default().fg(ASH_MID)
}

/// Spinner character
pub fn spinner() -> Style {
    Style::default().fg(SODIUM)
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

/// Status text next to spinner
pub fn status_text() -> Style {
    Style::default().fg(ASH_MID)
}

/// Status bar and help bar background
pub fn bar_bg() -> Style {
    Style::default().bg(FORM_RAISED)
}

/// History area background
pub fn history_bg() -> Style {
    Style::default().bg(FORM)
}

/// System message text (e.g. /help output)
pub fn system_message() -> Style {
    Style::default().fg(ASH_TEXT)
}

/// Scroll mode indicator in status bar
pub fn scroll_indicator() -> Style {
    Style::default().fg(SODIUM).add_modifier(Modifier::BOLD)
}

/// Markdown heading (bold + sodium)
pub fn heading() -> Style {
    Style::default().fg(SODIUM).add_modifier(Modifier::BOLD)
}

/// Markdown inline code
pub fn inline_code() -> Style {
    Style::default().fg(SODIUM)
}

/// Image attachment badge
pub fn image_attachment() -> Style {
    Style::default().fg(SODIUM)
}
