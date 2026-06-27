#[cfg(test)]
use std::cell::Cell;
#[cfg(not(test))]
use std::sync::atomic::{AtomicU8, Ordering};

use crate::config::ThemeName;
use lash_tui::{Color, Modifier, Style};

#[cfg(not(test))]
static ACTIVE_THEME: AtomicU8 = AtomicU8::new(ThemeName::Lash as u8);

#[cfg(test)]
thread_local! {
    static ACTIVE_THEME: Cell<ThemeName> = const { Cell::new(ThemeName::Lash) };
}

#[cfg(not(test))]
pub fn active_theme() -> ThemeName {
    match ACTIVE_THEME.load(Ordering::Relaxed) {
        value if value == ThemeName::System as u8 => ThemeName::System,
        _ => ThemeName::Lash,
    }
}

#[cfg(test)]
pub fn active_theme() -> ThemeName {
    ACTIVE_THEME.with(Cell::get)
}

#[cfg(not(test))]
pub fn set_active_theme(theme: ThemeName) {
    ACTIVE_THEME.store(theme as u8, Ordering::Relaxed);
}

#[cfg(test)]
pub fn set_active_theme(theme: ThemeName) {
    ACTIVE_THEME.with(|active| active.set(theme));
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

pub const CHALK_DIM: Color = Color::rgb(160, 158, 145);
pub const CHALK_MID: Color = Color::rgb(200, 196, 184);
pub const CHALK: Color = Color::rgb(232, 228, 208);

// Code ink reads brighter than tertiary prose: a fenced block is a focal
// surface, so its body sits near chalk (docs `--code-ink`) on the faint code
// surface rather than receding into the muted-gray tier.
pub const CODE_INK: Color = Color::rgb(216, 211, 189);
pub const CODE_COMMENT: Color = Color::rgb(140, 138, 126);
// Inline-code chip: one perceptible step above the code surface so it actually
// reads as a chip against the body background, without becoming a loud fill.
pub const INLINE_CHIP: Color = Color::rgb(33, 29, 24);

pub const SODIUM: Color = Color::rgb(232, 163, 60);
pub const LICHEN: Color = Color::rgb(138, 158, 108);
pub const ERROR: Color = Color::rgb(204, 68, 68);
pub const SELECTION_BG: Color = Color::rgb(54, 44, 23);

// Diff / patch surfaces. The whole hunk sits on one faint code surface so it
// reads as a single editorial block; changed lines lift off it with a clean,
// low-chroma tint that fills the row edge-to-edge (never a ragged block). The
// +/- signs carry the brand's diff accents — the same reds and sages the docs
// use for `--tok-deleted` / `--tok-inserted` — while line numbers stay quiet.
pub const CODE_SURFACE: Color = Color::rgb(27, 24, 20); // docs --code-bg #1b1814
pub const PATCH_RAIL: Color = Color::rgb(95, 94, 84); // calm but perceptible left rail (~3:1)
pub const PATCH_DELETED: Color = Color::rgb(215, 114, 114); // docs --tok-deleted #d77272
pub const PATCH_INSERTED: Color = Color::rgb(182, 199, 154); // docs --tok-inserted #b6c79a
// Tuned for equal *perceived* weight: the deletion red is eased off its earlier
// saturation while the insertion green gains chroma, so a symmetric edit reads
// balanced instead of biasing the eye toward the deletion.
pub const PATCH_ADD_LINE_BG: Color = Color::rgb(26, 46, 31);
pub const PATCH_REMOVE_LINE_BG: Color = Color::rgb(56, 27, 27);

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
    brand_for(active_theme())
}

pub fn state_ok() -> Color {
    themed(LICHEN, Color::ansi(2))
}

pub fn state_error() -> Color {
    themed(ERROR, Color::ansi(1))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Surface {
    background: Option<Color>,
}

impl Surface {
    pub fn fill(self) -> Style {
        match self.background {
            Some(background) => Style::default().bg(background),
            None => Style::default(),
        }
    }

    pub fn apply(self, style: Style) -> Style {
        match self.background {
            Some(background) => style.bg(background),
            None => style,
        }
    }
}

pub fn surface_base() -> Surface {
    surface_base_for(active_theme())
}

pub fn surface_raised() -> Surface {
    surface_raised_for(active_theme())
}

pub fn surface_deep() -> Surface {
    surface_deep_for(active_theme())
}

pub fn terminal_background() -> Option<Color> {
    terminal_background_for(active_theme())
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

pub fn selected_row_bg() -> Color {
    selected_row_bg_for(active_theme())
}

pub fn empty_state_logo_enabled() -> bool {
    empty_state_logo_enabled_for(active_theme())
}

pub fn input_rules_enabled() -> bool {
    matches!(active_theme(), ThemeName::Lash)
}

fn surface_base_for(theme: ThemeName) -> Surface {
    surface_for(theme, FORM)
}

fn surface_raised_for(theme: ThemeName) -> Surface {
    surface_for(theme, FORM_RAISED)
}

fn surface_deep_for(theme: ThemeName) -> Surface {
    surface_for(theme, FORM_DEEP)
}

fn terminal_background_for(theme: ThemeName) -> Option<Color> {
    match theme {
        ThemeName::Lash => Some(FORM),
        ThemeName::System => None,
    }
}

fn brand_for(theme: ThemeName) -> Color {
    themed_for(theme, SODIUM, Color::ansi(6))
}

fn surface_for(theme: ThemeName, lash: Color) -> Surface {
    match theme {
        ThemeName::Lash => Surface {
            background: Some(lash),
        },
        ThemeName::System => Surface { background: None },
    }
}

fn selection_bg_for(theme: ThemeName) -> Color {
    themed_for(theme, SELECTION_BG, Color::ansi(6))
}

fn selected_row_bg_for(theme: ThemeName) -> Color {
    themed_for(theme, SELECTION_BG, Color::ansi(8))
}

fn empty_state_logo_enabled_for(_theme: ThemeName) -> bool {
    true
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
    Style::default().fg(themed(CODE_INK, Color::ansi(7)))
}

pub fn code_keyword() -> Style {
    // Calm, sodium-free syntax emphasis: keywords earn hierarchy from weight, not
    // the reserved accent. Splashing sodium across every `fn`/`let` would break
    // the "accent means action" discipline the rest of the palette keeps.
    Style::default()
        .fg(text_primary())
        .add_modifier(Modifier::Bold)
}

pub fn code_string() -> Style {
    Style::default().fg(themed(PATCH_INSERTED, Color::ansi(2)))
}

pub fn code_comment() -> Style {
    Style::default()
        .fg(themed(CODE_COMMENT, Color::ansi(8)))
        .add_modifier(Modifier::Italic)
}

pub fn code_chrome() -> Style {
    // The fenced-code rail and rules need to actually declare "this is code" in a
    // dim room, so Lash lifts them off the near-invisible ash to the perceptible
    // rail tone. System keeps the dim terminal chrome.
    match active_theme() {
        ThemeName::Lash => Style::default().fg(PATCH_RAIL),
        ThemeName::System => chrome_rule(),
    }
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
    // Inline code is the load-bearing noun in a sentence, so it must not be the
    // faintest thing on the line. It recedes by *hue* (neutral, never the sodium
    // accent) and is marked as literal by a faint code-surface chip — not by
    // dimming below the surrounding prose.
    Style::default()
        .fg(themed(CHALK_MID, Color::ansi(7)))
        .bg(themed(INLINE_CHIP, Color::default_background()))
}

pub fn patch_frame() -> Style {
    Style::default().fg(themed(PATCH_RAIL, Color::ansi(8)))
}

pub fn patch_label() -> Style {
    Style::default().fg(brand()).add_modifier(Modifier::Bold)
}

pub fn patch_add() -> Style {
    Style::default().fg(themed(PATCH_INSERTED, Color::ansi(2)))
}

pub fn patch_remove() -> Style {
    Style::default().fg(themed(PATCH_DELETED, Color::ansi(1)))
}

// Per-row diff styles. Each changed row is painted from three slots — line
// number, +/- sign, and body — that share one row-wide background. The System
// theme keeps the terminal's own background (no tint blocks) and signals the
// change with ANSI accents on the sign instead, per the System-theme contract.
#[derive(Clone, Copy, Debug)]
pub struct PatchRowStyles {
    pub number: Style,
    pub sign: Style,
    pub body: Style,
}

fn patch_row(
    bg: Color,
    number: Color,
    sign: Color,
    body: Color,
    sign_bold: bool,
) -> PatchRowStyles {
    let mut sign_style = Style::default().fg(sign).bg(bg);
    if sign_bold {
        sign_style = sign_style.add_modifier(Modifier::Bold);
    }
    PatchRowStyles {
        number: Style::default().fg(number).bg(bg),
        sign: sign_style,
        body: Style::default().fg(body).bg(bg),
    }
}

pub fn patch_diff_add() -> PatchRowStyles {
    patch_row(
        themed(PATCH_ADD_LINE_BG, Color::default_background()),
        text_subtle(),
        themed(PATCH_INSERTED, Color::ansi(2)),
        text_primary(),
        true,
    )
}

pub fn patch_diff_remove() -> PatchRowStyles {
    patch_row(
        themed(PATCH_REMOVE_LINE_BG, Color::default_background()),
        text_subtle(),
        themed(PATCH_DELETED, Color::ansi(1)),
        text_muted(),
        true,
    )
}

pub fn patch_diff_context() -> PatchRowStyles {
    let bg = themed(CODE_SURFACE, Color::default_background());
    // Context line numbers are navigational text you read to orient inside a
    // hunk, so they clear ~4:1 — quiet, but never the faint chatter tier. They
    // still sit a notch below the changed-row numbers so the changes lead.
    let number = themed(Color::rgb(118, 116, 106), Color::ansi(8));
    patch_row(bg, number, text_faint(), text_subtle(), false)
}

pub fn patch_diff_hunk() -> PatchRowStyles {
    let bg = themed(CODE_SURFACE, Color::default_background());
    let mut styles = patch_row(bg, text_faint(), text_faint(), text_subtle(), false);
    styles.body = styles.body.add_modifier(Modifier::Bold);
    styles
}

pub fn patch_diff_meta() -> PatchRowStyles {
    let bg = themed(CODE_SURFACE, Color::default_background());
    let faint = Style::default()
        .fg(text_faint())
        .bg(bg)
        .add_modifier(Modifier::Dim);
    PatchRowStyles {
        number: faint,
        sign: faint,
        body: faint,
    }
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
        .bg(selected_row_bg())
        .add_modifier(Modifier::Bold)
}

pub fn list_item() -> Style {
    match active_theme() {
        ThemeName::Lash => Style::default().fg(text_subtle()),
        ThemeName::System => Style::default().fg(text_muted()),
    }
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
        assert_eq!(surface_base_for(ThemeName::System).background, None);
        assert_eq!(surface_raised_for(ThemeName::System).background, None);
        assert_eq!(surface_deep_for(ThemeName::System).background, None);
        assert_eq!(surface_base_for(ThemeName::System).fill().bg, None);
        assert_eq!(
            surface_raised_for(ThemeName::System)
                .apply(Style::default().fg(Color::ansi(7)))
                .bg,
            None
        );
        assert_eq!(selection_bg_for(ThemeName::System), Color::ansi(6));
        assert_eq!(selected_row_bg_for(ThemeName::System), Color::ansi(8));
        assert_ne!(
            selection_bg_for(ThemeName::System),
            Color::default_background()
        );
        assert!(empty_state_logo_enabled_for(ThemeName::System));
        assert_eq!(terminal_background_for(ThemeName::System), None);
    }

    #[test]
    fn lash_theme_keeps_custom_structural_surfaces() {
        assert_eq!(surface_base_for(ThemeName::Lash).background, Some(FORM));
        assert_eq!(
            surface_raised_for(ThemeName::Lash).background,
            Some(FORM_RAISED)
        );
        assert_eq!(
            surface_deep_for(ThemeName::Lash).background,
            Some(FORM_DEEP)
        );
        assert_eq!(surface_base_for(ThemeName::Lash).fill().bg, Some(FORM));
        assert_eq!(
            surface_raised_for(ThemeName::Lash)
                .apply(Style::default().fg(Color::ansi(7)))
                .bg,
            Some(FORM_RAISED)
        );
        assert_eq!(selection_bg_for(ThemeName::Lash), SELECTION_BG);
        assert_eq!(selected_row_bg_for(ThemeName::Lash), SELECTION_BG);
        assert!(empty_state_logo_enabled_for(ThemeName::Lash));
        assert_eq!(terminal_background_for(ThemeName::Lash), Some(FORM));
    }

    #[test]
    fn system_theme_uses_terminal_cyan_for_brand_accent() {
        assert_eq!(brand_for(ThemeName::System), Color::ansi(6));
        assert_eq!(brand_for(ThemeName::Lash), SODIUM);
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
