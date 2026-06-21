use std::sync::atomic::{AtomicU8, Ordering};

use crate::config::ThemeName;
use lash_tui::{Color, Modifier, Style};

static ACTIVE_THEME: AtomicU8 = AtomicU8::new(ThemeName::Lash as u8);

pub fn active_theme() -> ThemeName {
    match ACTIVE_THEME.load(Ordering::Relaxed) {
        value if value == ThemeName::System as u8 => ThemeName::System,
        _ => ThemeName::Lash,
    }
}

pub fn set_active_theme(theme: ThemeName) {
    ACTIVE_THEME.store(theme as u8, Ordering::Relaxed);
}

fn themed_for(theme: ThemeName, lash: Color, system: Color) -> Color {
    match theme {
        ThemeName::Lash => lash,
        ThemeName::System => system,
    }
}

fn themed(lash: Color, system: Color) -> Color {
    themed_for(active_theme(), lash, system)
}

// ─── Pigments ────────────────────────────────────────────────────────────────
// Raw paints named by their physical source. Do not reach for these when you
// mean "muted text" or "body text" — those ideas live in the token layer
// below. Pigments are for places that genuinely need a specific swatch
// (patch line backgrounds, selection highlight, brand identity).

pub const FORM: Color = Color::rgb(14, 13, 11);
pub const FORM_DEEP: Color = Color::rgb(8, 8, 7);
pub const FORM_RAISED: Color = Color::rgb(20, 20, 18);

pub const ASH: Color = Color::rgb(42, 42, 40);
pub const ASH_LIGHT: Color = Color::rgb(58, 58, 52);
pub const ASH_MID: Color = Color::rgb(74, 74, 68);
pub const ASH_TEXT: Color = Color::rgb(90, 90, 80);

pub const CHALK_DIM: Color = Color::rgb(122, 122, 112);
pub const CHALK_MID: Color = Color::rgb(200, 196, 184);
pub const CHALK: Color = Color::rgb(232, 228, 208);

pub const SODIUM: Color = Color::rgb(232, 163, 60);
pub const LICHEN: Color = Color::rgb(138, 158, 108);
pub const ERROR: Color = Color::rgb(204, 68, 68);
pub const SELECTION_BG: Color = Color::rgb(54, 44, 23);

pub const PATCH_FRAME: Color = Color::rgb(128, 94, 38);
pub const PATCH_ADD: Color = Color::rgb(126, 164, 92);
pub const PATCH_HUNK: Color = Color::rgb(154, 142, 106);
pub const PATCH_ADD_GUTTER_BG: Color = Color::rgb(39, 55, 31);
pub const PATCH_ADD_LINE_BG: Color = Color::rgb(25, 37, 23);
pub const PATCH_REMOVE_GUTTER_BG: Color = Color::rgb(78, 36, 32);
pub const PATCH_REMOVE_LINE_BG: Color = Color::rgb(46, 24, 21);
pub const PATCH_CONTEXT_GUTTER_BG: Color = Color::rgb(26, 25, 22);
pub const PATCH_CONTEXT_LINE_BG: Color = Color::rgb(19, 18, 16);
pub const PATCH_HUNK_GUTTER_BG: Color = Color::rgb(41, 34, 22);
pub const PATCH_HUNK_LINE_BG: Color = Color::rgb(28, 24, 18);

pub const PROMPT_CHAR: &str = "❯";

// ─── Tokens ──────────────────────────────────────────────────────────────────
// Semantic vocabulary. Every style function in this module composes from
// these. Call sites elsewhere should prefer tokens over pigments when the
// intent is a role ("muted body text") rather than a swatch.
//
// Text hierarchy, brightest to faintest:
//   primary  → body & headings (full contrast)
//   muted    → secondary text, labels, subheadings
//   subtle   → tertiary text, inline code, nested list items
//   faint    → tool output, system chatter, separators between strong tokens
//
// Brand is reserved for identity, navigation affordances, and state that
// needs to read as "active/current" — NOT for grammatical decoration
// (headings, inline code, every bold word).

pub fn text_primary() -> Color {
    themed(CHALK, Color::default_foreground())
}

pub fn text_muted() -> Color {
    themed(CHALK_MID, Color::ansi(7))
}

pub fn text_subtle() -> Color {
    themed(CHALK_DIM, Color::ansi(8))
}

pub fn text_faint() -> Color {
    themed(ASH_TEXT, Color::ansi(8))
}

pub fn brand() -> Color {
    themed(SODIUM, Color::ansi(3))
}

pub fn state_ok() -> Color {
    themed(LICHEN, Color::ansi(2))
}

pub fn state_error() -> Color {
    themed(ERROR, Color::ansi(1))
}

pub fn surface_base() -> Color {
    surface_base_for(active_theme())
}

pub fn surface_raised() -> Color {
    surface_raised_for(active_theme())
}

pub fn surface_deep() -> Color {
    surface_deep_for(active_theme())
}

pub fn rule() -> Color {
    themed(ASH_LIGHT, Color::ansi(8))
}

pub fn border_dim() -> Color {
    themed(ASH_MID, Color::ansi(8))
}

pub fn border_faint() -> Color {
    themed(ASH, Color::ansi(8))
}

pub fn selection_bg() -> Color {
    selection_bg_for(active_theme())
}

pub fn empty_state_logo_enabled() -> bool {
    empty_state_logo_enabled_for(active_theme())
}

pub fn input_rules_enabled() -> bool {
    matches!(active_theme(), ThemeName::Lash)
}

fn surface_base_for(theme: ThemeName) -> Color {
    themed_for(theme, FORM, Color::default_background())
}

fn surface_raised_for(theme: ThemeName) -> Color {
    themed_for(theme, FORM_RAISED, Color::default_background())
}

fn surface_deep_for(theme: ThemeName) -> Color {
    themed_for(theme, FORM_DEEP, Color::default_background())
}

fn selection_bg_for(theme: ThemeName) -> Color {
    themed_for(theme, SELECTION_BG, Color::ansi(6))
}

fn empty_state_logo_enabled_for(theme: ThemeName) -> bool {
    matches!(theme, ThemeName::Lash)
}

// ─── Semantic styles ─────────────────────────────────────────────────────────
// Named by use site. Always composed from tokens above — never from raw
// pigments except where a specific paint is deliberate (patch gutters,
// selection backgrounds).

pub fn prompt() -> Style {
    Style::default().fg(brand()).add_modifier(Modifier::Bold)
}

pub fn user_input() -> Style {
    Style::default().fg(text_muted())
}

pub fn slash_command_slash() -> Style {
    Style::default().fg(brand()).add_modifier(Modifier::Bold)
}

pub fn image_marker() -> Style {
    Style::default().fg(brand()).add_modifier(Modifier::Bold)
}

pub fn assistant_bar() -> Style {
    Style::default().fg(text_faint())
}

pub fn assistant_text() -> Style {
    Style::default().fg(text_primary())
}

// Model reasoning / "thinking" trace. Pi renders this as italic + muted
// gray so it sits visibly below the assistant's actual answer without
// competing with prose; we follow the same contract using lash's
// existing faint text token (ASH_TEXT) plus italic. Reusing `text_faint`
// keeps the palette tight — no new pigment — and keeps reasoning in the
// same hierarchy tier as other "system chatter" surfaces.
pub fn assistant_reasoning() -> Style {
    Style::default()
        .fg(text_faint())
        .add_modifier(Modifier::Italic)
}

pub fn assistant_reasoning_bar() -> Style {
    Style::default().fg(text_faint())
}

pub fn nested_list_item() -> Style {
    Style::default().fg(text_subtle())
}

pub fn system_output() -> Style {
    Style::default().fg(text_faint())
}

pub fn system_message() -> Style {
    Style::default().fg(text_faint())
}

pub fn error() -> Style {
    Style::default().fg(state_error())
}

pub fn code_content() -> Style {
    Style::default().fg(text_subtle())
}

pub fn code_keyword() -> Style {
    Style::default().fg(brand()).add_modifier(Modifier::Bold)
}

pub fn code_string() -> Style {
    Style::default().fg(state_ok())
}

pub fn code_comment() -> Style {
    Style::default()
        .fg(text_faint())
        .add_modifier(Modifier::Italic)
}

pub fn code_chrome() -> Style {
    chrome_rule()
}

pub fn chrome_rule() -> Style {
    chrome_rule_for(active_theme())
}

fn chrome_rule_for(theme: ThemeName) -> Style {
    match theme {
        ThemeName::Lash => Style::default().fg(themed_for(theme, ASH, Color::ansi(8))),
        ThemeName::System => Style::default()
            .fg(themed_for(theme, ASH_TEXT, Color::ansi(8)))
            .add_modifier(Modifier::Dim),
    }
}

pub fn explore_marker() -> Style {
    Style::default()
        .fg(themed(Color::rgb(122, 94, 48), Color::ansi(3)))
        .add_modifier(Modifier::Bold)
}

pub fn explore_label() -> Style {
    Style::default()
        .fg(text_subtle())
        .add_modifier(Modifier::Bold)
}

pub fn code_header() -> Style {
    Style::default().fg(border_dim())
}

pub fn tool_success() -> Style {
    Style::default().fg(text_faint())
}

pub fn tool_failure() -> Style {
    Style::default().fg(state_error())
}

pub fn help_key() -> Style {
    Style::default().fg(brand()).add_modifier(Modifier::Bold)
}

pub fn help_desc() -> Style {
    Style::default().fg(border_dim())
}

// Headings and inline code deliberately avoid brand(). The critique: SODIUM
// was doing quadruple duty (brand + user identity + headings + inline code),
// which dilutes it everywhere. Headings earn hierarchy from weight and the
// blank line the markdown renderer places around them. Inline code is
// self-describing prose and should recede, not pop.

pub fn heading() -> Style {
    Style::default()
        .fg(text_primary())
        .add_modifier(Modifier::Bold)
}

pub fn subheading() -> Style {
    Style::default()
        .fg(text_muted())
        .add_modifier(Modifier::Bold)
}

pub fn inline_code() -> Style {
    Style::default().fg(text_subtle())
}

pub fn patch_frame() -> Style {
    Style::default().fg(themed(PATCH_FRAME, Color::ansi(3)))
}

pub fn patch_label() -> Style {
    Style::default().fg(brand()).add_modifier(Modifier::Bold)
}

pub fn patch_add() -> Style {
    Style::default().fg(themed(PATCH_ADD, Color::ansi(2)))
}

pub fn patch_remove() -> Style {
    Style::default().fg(state_error())
}

pub fn patch_diff_add_gutter() -> Style {
    Style::default()
        .fg(state_ok())
        .bg(themed(PATCH_ADD_GUTTER_BG, Color::ansi(2)))
        .add_modifier(Modifier::Bold)
}

pub fn patch_diff_add_line() -> Style {
    Style::default()
        .fg(text_primary())
        .bg(themed(PATCH_ADD_LINE_BG, Color::default_background()))
}

pub fn patch_diff_remove_gutter() -> Style {
    Style::default()
        .fg(state_error())
        .bg(themed(PATCH_REMOVE_GUTTER_BG, Color::ansi(1)))
        .add_modifier(Modifier::Bold)
}

pub fn patch_diff_remove_line() -> Style {
    Style::default()
        .fg(text_primary())
        .bg(themed(PATCH_REMOVE_LINE_BG, Color::default_background()))
}

pub fn patch_diff_context_gutter() -> Style {
    Style::default()
        .fg(text_faint())
        .bg(themed(PATCH_CONTEXT_GUTTER_BG, Color::default_background()))
}

pub fn patch_diff_context_line() -> Style {
    Style::default()
        .fg(text_subtle())
        .bg(themed(PATCH_CONTEXT_LINE_BG, Color::default_background()))
}

pub fn patch_diff_hunk_gutter() -> Style {
    Style::default()
        .fg(themed(PATCH_HUNK, Color::ansi(3)))
        .bg(themed(PATCH_HUNK_GUTTER_BG, Color::default_background()))
}

pub fn patch_diff_hunk_line() -> Style {
    Style::default()
        .fg(themed(PATCH_HUNK, Color::ansi(3)))
        .bg(themed(PATCH_HUNK_LINE_BG, Color::default_background()))
        .add_modifier(Modifier::Bold)
}

pub fn patch_diff_meta_line() -> Style {
    Style::default()
        .fg(text_faint())
        .add_modifier(Modifier::Dim)
}

pub fn subagent_marker() -> Style {
    Style::default().fg(state_ok()).add_modifier(Modifier::Bold)
}

pub fn subagent_child() -> Style {
    Style::default().fg(text_subtle())
}

pub fn turn_separator() -> Style {
    match active_theme() {
        ThemeName::Lash => Style::default().fg(rule()),
        ThemeName::System => chrome_rule(),
    }
}

pub fn selected_row() -> Style {
    Style::default()
        .fg(text_primary())
        .bg(selection_bg())
        .add_modifier(Modifier::Bold)
}

pub fn edit_lane_bold() -> Style {
    Style::default().fg(state_ok()).add_modifier(Modifier::Bold)
}

pub fn turn_status_brand() -> Style {
    match active_theme() {
        ThemeName::Lash => Style::default()
            .fg(text_primary())
            .add_modifier(Modifier::Bold),
        ThemeName::System => Style::default().fg(text_subtle()),
    }
}

pub fn turn_status_slash() -> Style {
    slash_command_slash()
}

pub fn turn_status_state() -> Style {
    match active_theme() {
        ThemeName::Lash => Style::default()
            .fg(text_primary())
            .add_modifier(Modifier::Bold),
        ThemeName::System => Style::default().fg(text_muted()),
    }
}

pub fn turn_status_elapsed() -> Style {
    Style::default().fg(text_subtle())
}

pub fn process_selected_chrome() -> Style {
    Style::default().bg(selection_bg())
}

pub fn process_selected_indicator() -> Style {
    Style::default()
        .fg(brand())
        .bg(selection_bg())
        .add_modifier(Modifier::Bold)
}

pub fn process_selected_badge() -> Style {
    Style::default()
        .fg(text_primary())
        .bg(selection_bg())
        .add_modifier(Modifier::Bold)
}

pub fn process_selected_label() -> Style {
    Style::default()
        .fg(text_primary())
        .bg(selection_bg())
        .add_modifier(Modifier::Bold)
}

pub fn error_border() -> Style {
    Style::default().fg(state_error())
}

pub fn error_title() -> Style {
    Style::default()
        .fg(state_error())
        .add_modifier(Modifier::Bold)
}

// Text-hierarchy styles exposed for call sites that need "subtle text" /
// "faint text" without reaching for a pigment directly. Prefer these over
// `Style::default().fg(theme::CHALK_DIM)` etc. in new code.

pub fn text_subtle_style() -> Style {
    Style::default().fg(text_subtle())
}

pub fn text_faint_style() -> Style {
    Style::default().fg(text_faint())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn system_theme_keeps_structural_surfaces_on_terminal_background() {
        assert_eq!(
            surface_base_for(ThemeName::System),
            Color::default_background()
        );
        assert_eq!(
            surface_raised_for(ThemeName::System),
            Color::default_background()
        );
        assert_eq!(
            surface_deep_for(ThemeName::System),
            Color::default_background()
        );
        assert_eq!(selection_bg_for(ThemeName::System), Color::ansi(6));
        assert_ne!(
            selection_bg_for(ThemeName::System),
            Color::default_background()
        );
        assert!(!empty_state_logo_enabled_for(ThemeName::System));
    }

    #[test]
    fn lash_theme_keeps_custom_structural_surfaces() {
        assert_eq!(surface_base_for(ThemeName::Lash), FORM);
        assert_eq!(surface_raised_for(ThemeName::Lash), FORM_RAISED);
        assert_eq!(surface_deep_for(ThemeName::Lash), FORM_DEEP);
        assert_eq!(selection_bg_for(ThemeName::Lash), SELECTION_BG);
        assert!(empty_state_logo_enabled_for(ThemeName::Lash));
    }

    #[test]
    fn system_theme_uses_dim_terminal_chrome_rules() {
        let style = chrome_rule_for(ThemeName::System);
        assert_eq!(style.fg, Some(Color::ansi(8)));
        assert!(style.dim);
        assert_eq!(style.bg, None);

        let style = chrome_rule_for(ThemeName::Lash);
        assert_eq!(style.fg, Some(ASH));
        assert!(!style.dim);
        assert_eq!(style.bg, None);
    }
}
