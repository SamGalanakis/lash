use lash_tui::{Color, Modifier, Style};

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
    CHALK
}

pub fn text_muted() -> Color {
    CHALK_MID
}

pub fn text_subtle() -> Color {
    CHALK_DIM
}

pub fn text_faint() -> Color {
    ASH_TEXT
}

pub fn brand() -> Color {
    SODIUM
}

pub fn state_ok() -> Color {
    LICHEN
}

pub fn state_error() -> Color {
    ERROR
}

pub fn surface_base() -> Color {
    FORM
}

pub fn surface_raised() -> Color {
    FORM_RAISED
}

pub fn surface_deep() -> Color {
    FORM_DEEP
}

pub fn rule() -> Color {
    ASH_LIGHT
}

pub fn border_dim() -> Color {
    ASH_MID
}

pub fn border_faint() -> Color {
    ASH
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
    Style::default().fg(border_faint())
}

pub fn explore_marker() -> Style {
    Style::default()
        .fg(Color::rgb(122, 94, 48))
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
    Style::default().fg(PATCH_FRAME)
}

pub fn patch_label() -> Style {
    Style::default().fg(brand()).add_modifier(Modifier::Bold)
}

pub fn patch_add() -> Style {
    Style::default().fg(PATCH_ADD)
}

pub fn patch_remove() -> Style {
    Style::default().fg(state_error())
}

pub fn patch_diff_add_gutter() -> Style {
    Style::default()
        .fg(state_ok())
        .bg(PATCH_ADD_GUTTER_BG)
        .add_modifier(Modifier::Bold)
}

pub fn patch_diff_add_line() -> Style {
    Style::default().fg(text_primary()).bg(PATCH_ADD_LINE_BG)
}

pub fn patch_diff_remove_gutter() -> Style {
    Style::default()
        .fg(state_error())
        .bg(PATCH_REMOVE_GUTTER_BG)
        .add_modifier(Modifier::Bold)
}

pub fn patch_diff_remove_line() -> Style {
    Style::default().fg(text_primary()).bg(PATCH_REMOVE_LINE_BG)
}

pub fn patch_diff_context_gutter() -> Style {
    Style::default()
        .fg(text_faint())
        .bg(PATCH_CONTEXT_GUTTER_BG)
}

pub fn patch_diff_context_line() -> Style {
    Style::default().fg(text_subtle()).bg(PATCH_CONTEXT_LINE_BG)
}

pub fn patch_diff_hunk_gutter() -> Style {
    Style::default().fg(PATCH_HUNK).bg(PATCH_HUNK_GUTTER_BG)
}

pub fn patch_diff_hunk_line() -> Style {
    Style::default()
        .fg(PATCH_HUNK)
        .bg(PATCH_HUNK_LINE_BG)
        .add_modifier(Modifier::Bold)
}

pub fn patch_diff_meta_line() -> Style {
    Style::default()
        .fg(text_faint())
        .add_modifier(Modifier::Dim)
}

pub fn delegate_marker() -> Style {
    Style::default().fg(state_ok()).add_modifier(Modifier::Bold)
}

pub fn delegate_child() -> Style {
    Style::default().fg(text_subtle())
}

pub fn plan_done_marker() -> Style {
    Style::default().fg(state_ok()).add_modifier(Modifier::Bold)
}

pub fn plan_active_marker() -> Style {
    Style::default().fg(brand()).add_modifier(Modifier::Bold)
}

pub fn plan_pending_marker() -> Style {
    Style::default()
        .fg(text_subtle())
        .add_modifier(Modifier::Bold)
}

pub fn turn_separator() -> Style {
    Style::default().fg(rule())
}

pub fn edit_lane_bold() -> Style {
    Style::default().fg(state_ok()).add_modifier(Modifier::Bold)
}

pub fn turn_status_brand() -> Style {
    Style::default()
        .fg(text_primary())
        .add_modifier(Modifier::Bold)
}

pub fn turn_status_slash() -> Style {
    slash_command_slash()
}

pub fn turn_status_state() -> Style {
    Style::default()
        .fg(text_primary())
        .add_modifier(Modifier::Bold)
}

pub fn turn_status_elapsed() -> Style {
    Style::default().fg(text_subtle())
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
