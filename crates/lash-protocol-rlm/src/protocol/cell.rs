//! Finalize / non-streaming Lashlang cell handling. All paired-tag grammar
//! lives in [`crate::cell_scan`]; this module only layers the
//! extraction-and-projection conveniences the driver needs.

use crate::cell_scan::first_lashlang_cell_span;

pub(super) struct CellExtraction {
    pub(super) prose: String,
    pub(super) code: String,
    pub(super) lashlang_cell_count: usize,
}

pub fn contains_lashlang_cell(text: &str) -> bool {
    first_lashlang_cell_span(text).is_some()
}

pub fn project_visible_assistant_prose(text: &str) -> String {
    let Some(span) = first_lashlang_cell_span(text) else {
        return text.to_string();
    };
    text[..span.start_tag_start].trim_end().to_string()
}

pub(super) fn extract_lashlang_cell(text: &str) -> Option<CellExtraction> {
    let span = first_lashlang_cell_span(text)?;
    let code = text[span.body_start..span.body_end].to_string();
    Some(CellExtraction {
        prose: text[..span.start_tag_start].trim_end().to_string(),
        code,
        lashlang_cell_count: 1,
    })
}
