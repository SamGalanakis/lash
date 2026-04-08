use lash_tui::{Color, Modifier, Style};

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

pub fn prompt() -> Style {
    Style::default().fg(SODIUM).add_modifier(Modifier::Bold)
}

pub fn user_input() -> Style {
    Style::default().fg(CHALK_MID)
}

pub fn image_marker() -> Style {
    Style::default().fg(SODIUM).add_modifier(Modifier::Bold)
}

pub fn assistant_bar() -> Style {
    Style::default().fg(ASH_TEXT)
}

pub fn assistant_text() -> Style {
    Style::default().fg(CHALK)
}

pub fn nested_list_item() -> Style {
    Style::default().fg(CHALK_DIM)
}

pub fn system_output() -> Style {
    Style::default().fg(ASH_TEXT)
}

pub fn system_message() -> Style {
    Style::default().fg(ASH_TEXT)
}

pub fn error() -> Style {
    Style::default().fg(ERROR)
}

pub fn code_content() -> Style {
    Style::default().fg(CHALK_DIM)
}

pub fn code_keyword() -> Style {
    Style::default().fg(SODIUM).add_modifier(Modifier::Bold)
}

pub fn code_string() -> Style {
    Style::default().fg(LICHEN)
}

pub fn code_comment() -> Style {
    Style::default().fg(ASH_TEXT).add_modifier(Modifier::Italic)
}

pub fn code_chrome() -> Style {
    Style::default().fg(ASH)
}

pub fn code_scribe() -> Style {
    Style::default().fg(Color::rgb(100, 72, 28))
}

pub fn explore_marker() -> Style {
    Style::default()
        .fg(Color::rgb(122, 94, 48))
        .add_modifier(Modifier::Bold)
}

pub fn explore_label() -> Style {
    Style::default().fg(CHALK_DIM).add_modifier(Modifier::Bold)
}

pub fn code_header() -> Style {
    Style::default().fg(ASH_MID)
}

pub fn tool_success() -> Style {
    Style::default().fg(ASH_TEXT)
}

pub fn tool_failure() -> Style {
    Style::default().fg(ERROR)
}

pub fn help_key() -> Style {
    Style::default().fg(SODIUM).add_modifier(Modifier::Bold)
}

pub fn help_desc() -> Style {
    Style::default().fg(ASH_MID)
}

pub fn heading() -> Style {
    Style::default().fg(SODIUM).add_modifier(Modifier::Bold)
}

pub fn subheading() -> Style {
    Style::default().fg(CHALK_MID).add_modifier(Modifier::Bold)
}

pub fn inline_code() -> Style {
    Style::default().fg(SODIUM)
}

pub fn patch_frame() -> Style {
    Style::default().fg(PATCH_FRAME)
}

pub fn patch_label() -> Style {
    Style::default().fg(SODIUM).add_modifier(Modifier::Bold)
}

pub fn patch_add() -> Style {
    Style::default().fg(PATCH_ADD)
}

pub fn patch_remove() -> Style {
    Style::default().fg(ERROR)
}

pub fn patch_diff_add_gutter() -> Style {
    Style::default()
        .fg(LICHEN)
        .bg(PATCH_ADD_GUTTER_BG)
        .add_modifier(Modifier::Bold)
}

pub fn patch_diff_add_line() -> Style {
    Style::default().fg(CHALK).bg(PATCH_ADD_LINE_BG)
}

pub fn patch_diff_remove_gutter() -> Style {
    Style::default()
        .fg(ERROR)
        .bg(PATCH_REMOVE_GUTTER_BG)
        .add_modifier(Modifier::Bold)
}

pub fn patch_diff_remove_line() -> Style {
    Style::default().fg(CHALK).bg(PATCH_REMOVE_LINE_BG)
}

pub fn patch_diff_context_gutter() -> Style {
    Style::default().fg(ASH_TEXT).bg(PATCH_CONTEXT_GUTTER_BG)
}

pub fn patch_diff_context_line() -> Style {
    Style::default().fg(CHALK_DIM).bg(PATCH_CONTEXT_LINE_BG)
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
    Style::default().fg(ASH_TEXT).add_modifier(Modifier::Dim)
}

pub fn delegate_marker() -> Style {
    Style::default().fg(LICHEN).add_modifier(Modifier::Bold)
}

pub fn delegate_child() -> Style {
    Style::default().fg(CHALK_DIM)
}

pub fn plan_done_marker() -> Style {
    Style::default().fg(LICHEN).add_modifier(Modifier::Bold)
}

pub fn plan_active_marker() -> Style {
    Style::default().fg(SODIUM).add_modifier(Modifier::Bold)
}

pub fn plan_pending_marker() -> Style {
    Style::default().fg(CHALK_DIM).add_modifier(Modifier::Bold)
}

pub fn turn_separator() -> Style {
    Style::default().fg(ASH_LIGHT)
}

pub fn edit_lane_bold() -> Style {
    Style::default().fg(LICHEN).add_modifier(Modifier::Bold)
}

pub fn turn_status_brand() -> Style {
    Style::default().fg(CHALK).add_modifier(Modifier::Bold)
}

pub fn turn_status_slash() -> Style {
    Style::default().fg(SODIUM).add_modifier(Modifier::Bold)
}

pub fn turn_status_state() -> Style {
    Style::default().fg(CHALK).add_modifier(Modifier::Bold)
}

pub fn turn_status_elapsed() -> Style {
    Style::default().fg(CHALK_DIM)
}

pub fn error_border() -> Style {
    Style::default().fg(ERROR)
}

pub fn error_title() -> Style {
    Style::default().fg(ERROR).add_modifier(Modifier::Bold)
}
